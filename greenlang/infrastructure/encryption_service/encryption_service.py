# -*- coding: utf-8 -*-
"""
Core Encryption Service - AES-256-GCM Implementation (SEC-003)

Provides the core AES-256-GCM encryption operations using the
``cryptography`` library. Integrates with envelope encryption
for secure key management via AWS KMS.

Security properties:
    - AES-256-GCM: NIST-approved authenticated encryption
    - 256-bit keys (32 bytes): Quantum-resistant key size
    - 96-bit nonces (12 bytes): Recommended for AES-GCM
    - 128-bit auth tags (16 bytes): Full integrity protection

The service follows zero-hallucination principles: all operations
are deterministic using well-audited cryptographic primitives.
No LLM calls are made in the encryption/decryption path.

Example:
    >>> config = EncryptionServiceConfig(kms_key_id="alias/greenlang-cmk")
    >>> svc = await EncryptionService.create(config)
    >>> encrypted = await svc.encrypt(
    ...     b"sensitive data",
    ...     context={"tenant_id": "t-1", "data_class": "pii"},
    ... )
    >>> plaintext = await svc.decrypt(encrypted, context={"tenant_id": "t-1"})
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

if TYPE_CHECKING:
    from greenlang.infrastructure.encryption_service import (
        EncryptionServiceConfig,
        EncryptedData,
    )
    from greenlang.infrastructure.encryption_service.key_management import KeyManager

logger = logging.getLogger(__name__)

# Constants
AES_KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12  # 96 bits - recommended for AES-GCM
AUTH_TAG_SIZE = 16  # 128 bits


class EncryptionService:
    """Core AES-256-GCM encryption service.

    Uses envelope encryption with AWS KMS for key management.
    All encryption uses 256-bit keys with 96-bit nonces.

    This class should be instantiated via the async ``create()`` factory
    method to ensure proper initialization of async dependencies.

    Attributes:
        config: Service configuration.
        key_manager: Manages DEK lifecycle and caching.

    Example:
        >>> svc = await EncryptionService.create(config)
        >>> encrypted = await svc.encrypt(b"data", {"tenant_id": "t-1"})
    """

    def __init__(
        self,
        key_manager: KeyManager,
        config: EncryptionServiceConfig,
    ) -> None:
        """Initialize EncryptionService.

        Use ``EncryptionService.create()`` for proper async initialization.

        Args:
            key_manager: KeyManager instance for DEK operations.
            config: Service configuration.
        """
        self._key_manager = key_manager
        self._config = config
        self._metrics: Optional[Any] = None
        self._audit_logger: Optional[Any] = None

        logger.info(
            "EncryptionService initialized  kms_key=%s  cache_ttl=%ds",
            config.kms_key_id,
            config.dek_cache_ttl_seconds,
        )

    @classmethod
    async def create(
        cls,
        config: EncryptionServiceConfig,
        kms_provider: Optional[Any] = None,
    ) -> EncryptionService:
        """Factory method to create an EncryptionService instance.

        This is the recommended way to instantiate the service as it
        properly initializes all async dependencies.

        Args:
            config: Service configuration.
            kms_provider: Optional KMS provider. If None, creates an
                AWSKMSProvider using the config.

        Returns:
            Fully initialized EncryptionService.

        Raises:
            EncryptionKeyError: If KMS initialization fails.
        """
        from greenlang.infrastructure.encryption_service import (
            EncryptionServiceConfig,
        )
        from greenlang.infrastructure.encryption_service.envelope_encryption import (
            EnvelopeEncryptionService,
        )
        from greenlang.infrastructure.encryption_service.key_management import (
            KeyManager,
        )

        # Create KMS provider if not provided
        if kms_provider is None:
            try:
                from greenlang.governance.security.kms.aws_kms import AWSKMSProvider
                from greenlang.governance.security.kms.base_kms import KMSConfig

                kms_config = KMSConfig(
                    provider="aws",
                    key_id=config.kms_key_id,
                    region=config.kms_region,
                    max_retries=config.max_retries,
                    retry_base_delay=config.retry_base_delay,
                )
                kms_provider = AWSKMSProvider(kms_config)
                logger.info("Created AWSKMSProvider for encryption service")
            except ImportError as e:
                logger.warning(
                    "AWS KMS provider not available: %s. "
                    "Running in local-only mode.",
                    e,
                )
                kms_provider = None

        # Create envelope encryption service
        envelope_service = EnvelopeEncryptionService(kms_provider, config)

        # Create key manager
        key_manager = KeyManager(envelope_service, config)

        return cls(key_manager, config)

    async def encrypt(
        self,
        plaintext: bytes,
        context: Dict[str, str],
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Encrypt data using AES-256-GCM with envelope encryption.

        The encryption flow:
        1. Get or generate a DEK from cache/KMS
        2. Generate a cryptographically random nonce (12 bytes)
        3. Encrypt with AES-256-GCM
        4. Return EncryptedData with wrapped DEK

        Args:
            plaintext: Data to encrypt.
            context: Encryption context (tenant_id, data_class required
                when encryption_context_required=True).
            aad: Additional Authenticated Data bound to ciphertext.
                Useful for binding context that must match on decryption.

        Returns:
            EncryptedData with ciphertext, nonce, tag, and wrapped DEK.

        Raises:
            ValueError: If required context fields are missing.
            EncryptionError: If encryption fails.
        """
        from greenlang.infrastructure.encryption_service import (
            EncryptedData,
            EncryptionError,
        )

        start_time = datetime.now(timezone.utc)

        # Validate context
        self._validate_context(context)

        try:
            # Get or generate DEK
            plaintext_dek, encrypted_dek, key_version = (
                await self._key_manager.get_or_generate_dek(context)
            )

            # Generate cryptographically random nonce
            nonce = self._generate_nonce()

            # Encrypt with AES-256-GCM
            ciphertext, auth_tag = self._encrypt_bytes(
                plaintext, plaintext_dek, nonce, aad
            )

            elapsed_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Audit log (never log plaintext or keys)
            if self._config.enable_audit:
                logger.info(
                    "Encryption completed  "
                    "tenant=%s  data_class=%s  key_version=%s  "
                    "plaintext_size=%d  ciphertext_size=%d  elapsed=%.1fms",
                    context.get("tenant_id", "unknown"),
                    context.get("data_class", "unknown"),
                    key_version,
                    len(plaintext),
                    len(ciphertext),
                    elapsed_ms,
                )

            return EncryptedData(
                ciphertext=ciphertext,
                nonce=nonce,
                auth_tag=auth_tag,
                encrypted_dek=encrypted_dek,
                key_version=key_version,
                encryption_context=context,
                algorithm="AES-256-GCM",
            )

        except InvalidTag as e:
            logger.error("AES-GCM encryption failed: %s", e)
            raise EncryptionError(f"Encryption failed: {e}") from e
        except Exception as e:
            logger.error("Encryption failed: %s", e, exc_info=True)
            raise EncryptionError(f"Encryption failed: {e}") from e

    async def decrypt(
        self,
        encrypted_data: EncryptedData,
        context: Dict[str, str],
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Decrypt data using AES-256-GCM.

        The decryption flow:
        1. Unwrap the DEK via KMS (or cache)
        2. Verify the authentication tag
        3. Decrypt with AES-256-GCM
        4. Return plaintext

        Args:
            encrypted_data: EncryptedData from encrypt().
            context: Must match the original encryption context.
            aad: Must match the original AAD if provided during encryption.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ContextMismatchError: If context doesn't match.
            IntegrityError: If authentication tag verification fails.
            DecryptionError: If decryption fails.
        """
        from greenlang.infrastructure.encryption_service import (
            DecryptionError,
            IntegrityError,
            ContextMismatchError,
        )

        start_time = datetime.now(timezone.utc)

        # Validate context
        self._validate_context(context)

        # Verify context matches (for required fields)
        if self._config.encryption_context_required:
            stored_context = encrypted_data.encryption_context
            if stored_context:
                for key in ["tenant_id", "data_class"]:
                    if key in stored_context and key in context:
                        if stored_context[key] != context[key]:
                            raise ContextMismatchError(
                                f"Context mismatch for '{key}': "
                                f"expected '{stored_context[key]}', "
                                f"got '{context[key]}'"
                            )

        try:
            # Get DEK from cache or decrypt via KMS
            plaintext_dek = await self._key_manager.decrypt_dek(
                encrypted_data.encrypted_dek, context
            )

            # Decrypt with AES-256-GCM
            plaintext = self._decrypt_bytes(
                encrypted_data.ciphertext,
                plaintext_dek,
                encrypted_data.nonce,
                encrypted_data.auth_tag,
                aad,
            )

            elapsed_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Audit log
            if self._config.enable_audit:
                logger.info(
                    "Decryption completed  "
                    "tenant=%s  data_class=%s  key_version=%s  "
                    "ciphertext_size=%d  plaintext_size=%d  elapsed=%.1fms",
                    context.get("tenant_id", "unknown"),
                    context.get("data_class", "unknown"),
                    encrypted_data.key_version,
                    len(encrypted_data.ciphertext),
                    len(plaintext),
                    elapsed_ms,
                )

            return plaintext

        except InvalidTag as e:
            logger.error(
                "Authentication tag verification failed for tenant=%s",
                context.get("tenant_id", "unknown"),
            )
            raise IntegrityError(
                "Authentication tag verification failed. "
                "Data may have been tampered with."
            ) from e
        except ContextMismatchError:
            raise
        except Exception as e:
            logger.error("Decryption failed: %s", e, exc_info=True)
            raise DecryptionError(f"Decryption failed: {e}") from e

    def _encrypt_bytes(
        self,
        data: bytes,
        key: bytes,
        nonce: bytes,
        aad: Optional[bytes] = None,
    ) -> tuple[bytes, bytes]:
        """Low-level AES-256-GCM encryption.

        Args:
            data: Plaintext bytes to encrypt.
            key: 256-bit (32 byte) AES key.
            nonce: 96-bit (12 byte) nonce. NEVER reuse with same key.
            aad: Optional Additional Authenticated Data.

        Returns:
            Tuple of (ciphertext, auth_tag).

        Raises:
            ValueError: If key or nonce size is incorrect.
        """
        if len(key) != AES_KEY_SIZE:
            raise ValueError(f"Key must be {AES_KEY_SIZE} bytes, got {len(key)}")
        if len(nonce) != NONCE_SIZE:
            raise ValueError(f"Nonce must be {NONCE_SIZE} bytes, got {len(nonce)}")

        aesgcm = AESGCM(key)

        # AESGCM.encrypt returns ciphertext + tag concatenated
        ct_with_tag = aesgcm.encrypt(nonce, data, aad)

        # Split: tag is last 16 bytes
        ciphertext = ct_with_tag[:-AUTH_TAG_SIZE]
        auth_tag = ct_with_tag[-AUTH_TAG_SIZE:]

        return ciphertext, auth_tag

    def _decrypt_bytes(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        auth_tag: bytes,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Low-level AES-256-GCM decryption.

        Args:
            ciphertext: Encrypted data bytes.
            key: 256-bit (32 byte) AES key.
            nonce: 96-bit (12 byte) nonce used during encryption.
            auth_tag: 128-bit (16 byte) authentication tag.
            aad: Optional Additional Authenticated Data (must match).

        Returns:
            Decrypted plaintext bytes.

        Raises:
            InvalidTag: If authentication tag verification fails.
            ValueError: If key, nonce, or tag size is incorrect.
        """
        if len(key) != AES_KEY_SIZE:
            raise ValueError(f"Key must be {AES_KEY_SIZE} bytes, got {len(key)}")
        if len(nonce) != NONCE_SIZE:
            raise ValueError(f"Nonce must be {NONCE_SIZE} bytes, got {len(nonce)}")
        if len(auth_tag) != AUTH_TAG_SIZE:
            raise ValueError(
                f"Auth tag must be {AUTH_TAG_SIZE} bytes, got {len(auth_tag)}"
            )

        aesgcm = AESGCM(key)

        # Reconstruct ciphertext + tag for decryption
        ct_with_tag = ciphertext + auth_tag

        return aesgcm.decrypt(nonce, ct_with_tag, aad)

    def _validate_context(self, context: Dict[str, str]) -> None:
        """Validate encryption context contains required fields.

        Args:
            context: Encryption context dictionary.

        Raises:
            ValueError: If required fields are missing.
        """
        if not self._config.encryption_context_required:
            return

        if not context:
            raise ValueError("Encryption context is required")

        required_fields = ["tenant_id", "data_class"]
        missing = [f for f in required_fields if not context.get(f)]

        if missing:
            raise ValueError(
                f"Missing required encryption context fields: {missing}"
            )

    @staticmethod
    def _generate_nonce() -> bytes:
        """Generate a 96-bit (12 byte) cryptographically random nonce.

        Uses ``secrets.token_bytes`` which is suitable for
        cryptographic use.

        Returns:
            12-byte random nonce.
        """
        return secrets.token_bytes(NONCE_SIZE)

    def invalidate_cache(self, context: Optional[Dict[str, str]] = None) -> None:
        """Invalidate DEK cache.

        Args:
            context: Specific context to invalidate. If None, invalidates
                the entire cache.
        """
        self._key_manager.invalidate(context)
        logger.info(
            "Cache invalidated  context=%s",
            context if context else "all",
        )

    async def rotate_dek(self, context: Dict[str, str]) -> str:
        """Force rotation of DEK for the given context.

        This invalidates the cached DEK and generates a new one
        on the next encryption operation.

        Args:
            context: Encryption context for the DEK to rotate.

        Returns:
            New key version identifier.
        """
        self._key_manager.rotate(context)
        logger.info(
            "DEK rotation initiated  tenant=%s  data_class=%s",
            context.get("tenant_id", "unknown"),
            context.get("data_class", "unknown"),
        )

        # Generate new DEK immediately
        _, _, key_version = await self._key_manager.get_or_generate_dek(context)
        return key_version
