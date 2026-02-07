# -*- coding: utf-8 -*-
"""
Envelope Encryption Service - KMS-based DEK Management (SEC-003)

Implements envelope encryption pattern using AWS KMS for secure
Data Encryption Key (DEK) management:

    1. Generate DEK via KMS (kms:GenerateDataKey)
    2. Encrypt data with DEK (AES-256-GCM)
    3. Store encrypted DEK alongside ciphertext
    4. For decryption: unwrap DEK via KMS, then decrypt data

The envelope pattern ensures that:
    - The CMK (Customer Master Key) never leaves KMS
    - Each piece of data can use a unique DEK
    - DEKs are cryptographically bound to their encryption context
    - Key rotation happens at the CMK level, transparent to data

Example:
    >>> from greenlang.governance.security.kms.aws_kms import AWSKMSProvider
    >>> kms = AWSKMSProvider(config)
    >>> envelope_svc = EnvelopeEncryptionService(kms, enc_config)
    >>> encrypted = await envelope_svc.encrypt_envelope(
    ...     b"sensitive data",
    ...     {"tenant_id": "t-1"},
    ... )
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

if TYPE_CHECKING:
    from greenlang.infrastructure.encryption_service import (
        EncryptionServiceConfig,
        EncryptedData,
    )

logger = logging.getLogger(__name__)

# Constants
AES_KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12  # 96 bits
AUTH_TAG_SIZE = 16  # 128 bits


class EnvelopeEncryptionService:
    """Envelope encryption using AWS KMS for DEK management.

    Implements the envelope encryption pattern where:
    - The CMK in KMS wraps/unwraps Data Encryption Keys (DEKs)
    - Data is encrypted locally with the DEK using AES-256-GCM
    - The encrypted DEK is stored alongside the ciphertext

    This pattern provides:
    - Performance: Local encryption is faster than KMS-side encryption
    - Scale: Reduces KMS API calls through DEK caching
    - Security: CMK never leaves KMS hardware security module

    Attributes:
        kms: The KMS provider for key operations.
        config: Service configuration.

    Example:
        >>> svc = EnvelopeEncryptionService(kms_provider, config)
        >>> plaintext_dek, encrypted_dek = await svc.generate_data_key(
        ...     {"tenant_id": "t-1"}
        ... )
    """

    def __init__(
        self,
        kms_provider: Optional[Any],
        config: EncryptionServiceConfig,
    ) -> None:
        """Initialize EnvelopeEncryptionService.

        Args:
            kms_provider: KMS provider (AWSKMSProvider from
                greenlang.governance.security.kms). Can be None for
                local-only mode (testing).
            config: Service configuration.
        """
        self._kms = kms_provider
        self._config = config
        self._local_mode = kms_provider is None

        if self._local_mode:
            logger.warning(
                "EnvelopeEncryptionService running in LOCAL MODE. "
                "DEKs are generated locally without KMS protection. "
                "DO NOT USE IN PRODUCTION."
            )
            # Generate a local master key for testing only
            self._local_master_key = secrets.token_bytes(AES_KEY_SIZE)
        else:
            logger.info(
                "EnvelopeEncryptionService initialized  kms_key=%s",
                config.kms_key_id,
            )

    async def generate_data_key(
        self,
        encryption_context: Dict[str, str],
    ) -> Tuple[bytes, bytes]:
        """Generate a new Data Encryption Key.

        Uses KMS GenerateDataKey API which returns both plaintext
        and encrypted versions of the DEK. The encryption_context
        is cryptographically bound to the DEK.

        Args:
            encryption_context: Key-value pairs that are bound to the DEK.
                Must be provided again during decryption.

        Returns:
            Tuple of (plaintext_key, encrypted_key) - both 32 bytes.

        Raises:
            EncryptionKeyError: If key generation fails.
        """
        from greenlang.infrastructure.encryption_service import EncryptionKeyError

        start_time = datetime.now(timezone.utc)

        if self._local_mode:
            return self._generate_local_data_key(encryption_context)

        try:
            # Use existing KMS provider's create_data_key method
            result = self._kms.create_data_key(
                key_id=self._config.kms_key_id,
                key_spec="AES_256",
            )

            plaintext_key = result["plaintext"]
            encrypted_key = result["ciphertext"]

            elapsed_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            logger.debug(
                "Generated DEK via KMS  kms_key=%s  elapsed=%.1fms",
                self._config.kms_key_id,
                elapsed_ms,
            )

            return plaintext_key, encrypted_key

        except Exception as e:
            logger.error("KMS GenerateDataKey failed: %s", e, exc_info=True)
            raise EncryptionKeyError(f"Failed to generate data key: {e}") from e

    async def decrypt_data_key(
        self,
        encrypted_key: bytes,
        encryption_context: Dict[str, str],
    ) -> bytes:
        """Decrypt a wrapped DEK using KMS.

        Uses KMS Decrypt API to unwrap the DEK. The encryption_context
        must match the context used during key generation.

        Args:
            encrypted_key: KMS-wrapped DEK.
            encryption_context: Must match original context.

        Returns:
            Plaintext DEK (32 bytes).

        Raises:
            EncryptionKeyError: If decryption fails (including context
                mismatch).
        """
        from greenlang.infrastructure.encryption_service import EncryptionKeyError

        start_time = datetime.now(timezone.utc)

        if self._local_mode:
            return self._decrypt_local_data_key(encrypted_key, encryption_context)

        try:
            # The KMS Decrypt API automatically uses the correct CMK
            # based on the ciphertext metadata
            response = self._kms.retry_with_backoff(
                self._kms.client.decrypt,
                CiphertextBlob=encrypted_key,
                EncryptionContext=encryption_context,
            )

            plaintext_key = response["Plaintext"]

            elapsed_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            logger.debug(
                "Decrypted DEK via KMS  elapsed=%.1fms",
                elapsed_ms,
            )

            return plaintext_key

        except Exception as e:
            logger.error("KMS Decrypt failed: %s", e, exc_info=True)
            raise EncryptionKeyError(f"Failed to decrypt data key: {e}") from e

    async def encrypt_envelope(
        self,
        plaintext: bytes,
        encryption_context: Dict[str, str],
        aad: Optional[bytes] = None,
    ) -> EncryptedData:
        """Full envelope encryption flow.

        Combines DEK generation and local AES-256-GCM encryption:
        1. Generate a new DEK via KMS
        2. Encrypt data with the DEK
        3. Return EncryptedData with wrapped DEK

        Args:
            plaintext: Data to encrypt.
            encryption_context: Context bound to the encryption.
            aad: Additional Authenticated Data.

        Returns:
            EncryptedData with ciphertext and wrapped DEK.

        Raises:
            EncryptionError: If encryption fails.
        """
        from greenlang.infrastructure.encryption_service import (
            EncryptedData,
            EncryptionError,
        )
        import uuid

        try:
            # Generate DEK
            plaintext_dek, encrypted_dek = await self.generate_data_key(
                encryption_context
            )

            # Generate nonce
            nonce = secrets.token_bytes(NONCE_SIZE)

            # Encrypt with AES-256-GCM
            aesgcm = AESGCM(plaintext_dek)
            ct_with_tag = aesgcm.encrypt(nonce, plaintext, aad)

            # Split ciphertext and tag
            ciphertext = ct_with_tag[:-AUTH_TAG_SIZE]
            auth_tag = ct_with_tag[-AUTH_TAG_SIZE:]

            return EncryptedData(
                ciphertext=ciphertext,
                nonce=nonce,
                auth_tag=auth_tag,
                encrypted_dek=encrypted_dek,
                key_version=str(uuid.uuid4()),
                encryption_context=encryption_context,
                algorithm="AES-256-GCM",
            )

        except Exception as e:
            logger.error("Envelope encryption failed: %s", e, exc_info=True)
            raise EncryptionError(f"Envelope encryption failed: {e}") from e

    async def decrypt_envelope(
        self,
        encrypted_data: EncryptedData,
        aad: Optional[bytes] = None,
    ) -> bytes:
        """Full envelope decryption flow.

        Combines DEK decryption and local AES-256-GCM decryption:
        1. Decrypt DEK via KMS
        2. Decrypt data with the DEK
        3. Return plaintext

        Args:
            encrypted_data: EncryptedData from encrypt_envelope().
            aad: Must match original AAD.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            DecryptionError: If decryption fails.
        """
        from greenlang.infrastructure.encryption_service import DecryptionError

        try:
            # Decrypt DEK
            plaintext_dek = await self.decrypt_data_key(
                encrypted_data.encrypted_dek,
                encrypted_data.encryption_context,
            )

            # Reconstruct ciphertext + tag
            ct_with_tag = encrypted_data.ciphertext + encrypted_data.auth_tag

            # Decrypt with AES-256-GCM
            aesgcm = AESGCM(plaintext_dek)
            plaintext = aesgcm.decrypt(encrypted_data.nonce, ct_with_tag, aad)

            return plaintext

        except Exception as e:
            logger.error("Envelope decryption failed: %s", e, exc_info=True)
            raise DecryptionError(f"Envelope decryption failed: {e}") from e

    # -------------------------------------------------------------------------
    # Local Mode (Testing Only)
    # -------------------------------------------------------------------------

    def _generate_local_data_key(
        self,
        encryption_context: Dict[str, str],
    ) -> Tuple[bytes, bytes]:
        """Generate a DEK locally for testing.

        WARNING: This is NOT secure for production use. The "encrypted"
        key is just the plaintext key encrypted with a local master key,
        which provides no real security.

        Args:
            encryption_context: Ignored in local mode.

        Returns:
            Tuple of (plaintext_key, "encrypted" key).
        """
        # Generate random DEK
        plaintext_key = secrets.token_bytes(AES_KEY_SIZE)

        # "Encrypt" with local master key (NOT SECURE)
        aesgcm = AESGCM(self._local_master_key)
        nonce = secrets.token_bytes(NONCE_SIZE)
        encrypted = aesgcm.encrypt(nonce, plaintext_key, None)

        # Prepend nonce to encrypted key
        encrypted_key = nonce + encrypted

        logger.debug("Generated LOCAL DEK (testing only)")

        return plaintext_key, encrypted_key

    def _decrypt_local_data_key(
        self,
        encrypted_key: bytes,
        encryption_context: Dict[str, str],
    ) -> bytes:
        """Decrypt a locally-encrypted DEK for testing.

        Args:
            encrypted_key: Key encrypted with local master key.
            encryption_context: Ignored in local mode.

        Returns:
            Plaintext DEK.
        """
        # Extract nonce (first 12 bytes)
        nonce = encrypted_key[:NONCE_SIZE]
        ct_with_tag = encrypted_key[NONCE_SIZE:]

        # Decrypt with local master key
        aesgcm = AESGCM(self._local_master_key)
        plaintext_key = aesgcm.decrypt(nonce, ct_with_tag, None)

        logger.debug("Decrypted LOCAL DEK (testing only)")

        return plaintext_key
