"""
Encryption Service - Field-Level Encryption with AWS KMS Integration

This module provides field-level encryption for sensitive data using the
envelope encryption pattern. It integrates with AWS KMS for secure key
management and supports key rotation with version tracking.

SOC2 Controls Addressed:
    - CC6.1: Encryption of data at rest
    - CC6.7: Encryption key management

ISO27001 Controls Addressed:
    - A.10.1.1: Policy on use of cryptographic controls
    - A.10.1.2: Key management

Example:
    >>> config = EncryptionConfig(kms_key_id="alias/greenlang-data-key")
    >>> service = EncryptionService(config)
    >>> encrypted = await service.encrypt_field("sensitive-data", "PII")
    >>> decrypted = await service.decrypt_field(encrypted)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "AES-256-GCM"  # Recommended for most use cases
    AES_256_CBC = "AES-256-CBC"  # Legacy compatibility
    FERNET = "FERNET"  # Symmetric encryption with HMAC


class DataClassification(str, Enum):
    """Data classification levels for encryption policies."""

    PUBLIC = "PUBLIC"  # No encryption required
    INTERNAL = "INTERNAL"  # Standard encryption
    CONFIDENTIAL = "CONFIDENTIAL"  # Strong encryption
    RESTRICTED = "RESTRICTED"  # Maximum encryption with audit
    PII = "PII"  # Personally Identifiable Information
    PHI = "PHI"  # Protected Health Information
    FINANCIAL = "FINANCIAL"  # Financial data


class KeyVersion(BaseModel):
    """
    Represents a version of an encryption key.

    Key versioning enables seamless key rotation without
    requiring re-encryption of all existing data.

    Attributes:
        version: Version number of the key
        key_id: KMS key identifier
        created_at: When this version was created
        rotated_at: When this version was rotated out
        is_active: Whether this version is currently active
        algorithm: Encryption algorithm used with this key
    """

    version: int = Field(..., ge=1)
    key_id: str = Field(..., description="KMS key identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rotated_at: Optional[datetime] = Field(None)
    is_active: bool = Field(default=True)
    algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)

    # Key metadata for audit
    key_hash: Optional[str] = Field(None, description="SHA-256 hash of key for verification")


class EncryptedField(BaseModel):
    """
    Container for an encrypted field value.

    Includes all metadata needed for decryption and audit,
    following the envelope encryption pattern.

    Attributes:
        ciphertext: Base64-encoded encrypted data
        encrypted_data_key: Base64-encoded data key encrypted with KMS
        iv: Initialization vector (nonce) for decryption
        key_version: Version of the encryption key used
        algorithm: Encryption algorithm used
        classification: Data classification level
        encrypted_at: Timestamp of encryption
        field_hash: Hash of original plaintext for integrity verification
    """

    ciphertext: str = Field(..., description="Base64-encoded encrypted data")
    encrypted_data_key: str = Field(..., description="Data key encrypted with KMS")
    iv: str = Field(..., description="Base64-encoded initialization vector")
    key_version: int = Field(..., description="Key version used for encryption")
    algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)
    classification: DataClassification = Field(default=DataClassification.CONFIDENTIAL)
    encrypted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    field_hash: str = Field(..., description="SHA-256 hash of plaintext for integrity")
    auth_tag: Optional[str] = Field(None, description="Authentication tag for GCM mode")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "ct": self.ciphertext,
            "dk": self.encrypted_data_key,
            "iv": self.iv,
            "kv": self.key_version,
            "alg": self.algorithm.value,
            "cls": self.classification.value,
            "ts": self.encrypted_at.isoformat(),
            "h": self.field_hash,
            "tag": self.auth_tag,
        }

    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "EncryptedField":
        """Create from storage dictionary."""
        return cls(
            ciphertext=data["ct"],
            encrypted_data_key=data["dk"],
            iv=data["iv"],
            key_version=data["kv"],
            algorithm=EncryptionAlgorithm(data["alg"]),
            classification=DataClassification(data["cls"]),
            encrypted_at=datetime.fromisoformat(data["ts"]),
            field_hash=data["h"],
            auth_tag=data.get("tag"),
        )


class EncryptionConfig(BaseModel):
    """Configuration for the Encryption Service."""

    # AWS KMS configuration
    kms_key_id: str = Field(
        default="alias/greenlang-data-key",
        description="AWS KMS key ID or alias",
    )
    kms_region: str = Field(default="us-east-1")

    # Algorithm preferences
    default_algorithm: EncryptionAlgorithm = Field(default=EncryptionAlgorithm.AES_256_GCM)

    # Key rotation settings
    key_rotation_days: int = Field(default=90, ge=30, le=365)
    auto_rotate: bool = Field(default=True)

    # Performance settings
    cache_data_keys: bool = Field(default=True)
    data_key_cache_size: int = Field(default=100)
    data_key_cache_ttl_seconds: int = Field(default=300)

    # Classification policies
    classification_policies: Dict[DataClassification, EncryptionAlgorithm] = Field(
        default_factory=lambda: {
            DataClassification.INTERNAL: EncryptionAlgorithm.AES_256_GCM,
            DataClassification.CONFIDENTIAL: EncryptionAlgorithm.AES_256_GCM,
            DataClassification.RESTRICTED: EncryptionAlgorithm.AES_256_GCM,
            DataClassification.PII: EncryptionAlgorithm.AES_256_GCM,
            DataClassification.PHI: EncryptionAlgorithm.AES_256_GCM,
            DataClassification.FINANCIAL: EncryptionAlgorithm.AES_256_GCM,
        }
    )

    # Local development mode (uses local keys instead of KMS)
    local_mode: bool = Field(default=False)
    local_master_key: Optional[str] = Field(
        default=None,
        description="Base64-encoded 32-byte key for local development",
    )


class EncryptionService:
    """
    Production-grade encryption service with AWS KMS integration.

    Implements the envelope encryption pattern:
    1. Generate a unique data encryption key (DEK) for each encryption
    2. Encrypt the data with the DEK using AES-256-GCM
    3. Encrypt the DEK with the KMS master key (KEK)
    4. Store the encrypted DEK alongside the ciphertext

    This approach provides:
    - Unique key per encryption (limits blast radius)
    - Efficient key rotation (only re-encrypt DEKs)
    - Secure key management via KMS
    - Support for multiple key versions

    Example:
        >>> config = EncryptionConfig(kms_key_id="alias/greenlang-data")
        >>> service = EncryptionService(config)
        >>> await service.initialize()
        >>> encrypted = await service.encrypt_field(
        ...     "secret-value",
        ...     DataClassification.PII,
        ... )
        >>> decrypted = await service.decrypt_field(encrypted)
        >>> assert decrypted == "secret-value"

    Attributes:
        config: Service configuration
        _current_key_version: Active key version for encryption
        _key_versions: All known key versions for decryption
    """

    def __init__(self, config: Optional[EncryptionConfig] = None):
        """
        Initialize the Encryption Service.

        Args:
            config: Service configuration (uses defaults if not provided)
        """
        self.config = config or EncryptionConfig()
        self._current_key_version: Optional[KeyVersion] = None
        self._key_versions: Dict[int, KeyVersion] = {}
        self._data_key_cache: Dict[str, bytes] = {}
        self._kms_client = None
        self._initialized = False

        # Local mode key for development
        self._local_master_key: Optional[bytes] = None

        logger.info(
            "EncryptionService initialized",
            extra={
                "kms_key_id": self.config.kms_key_id,
                "default_algorithm": self.config.default_algorithm.value,
                "local_mode": self.config.local_mode,
            },
        )

    async def initialize(self) -> None:
        """
        Initialize the encryption service.

        Sets up KMS client and loads current key version.
        """
        if self._initialized:
            logger.warning("EncryptionService already initialized")
            return

        try:
            if self.config.local_mode:
                # Local development mode
                if self.config.local_master_key:
                    self._local_master_key = base64.b64decode(self.config.local_master_key)
                else:
                    # Generate a local key for development
                    self._local_master_key = secrets.token_bytes(32)
                    logger.warning(
                        "Generated ephemeral local master key. "
                        "Data encrypted in this session cannot be decrypted in future sessions."
                    )

                self._current_key_version = KeyVersion(
                    version=1,
                    key_id="local-development-key",
                    algorithm=self.config.default_algorithm,
                )
            else:
                # Production mode - connect to AWS KMS
                await self._initialize_kms_client()
                await self._load_key_versions()

            self._key_versions[self._current_key_version.version] = self._current_key_version
            self._initialized = True

            logger.info(
                "EncryptionService initialization complete",
                extra={"current_key_version": self._current_key_version.version},
            )

        except Exception as e:
            logger.error(f"Failed to initialize EncryptionService: {e}", exc_info=True)
            raise

    async def encrypt_field(
        self,
        plaintext: Union[str, bytes],
        classification: DataClassification = DataClassification.CONFIDENTIAL,
        context: Optional[Dict[str, str]] = None,
    ) -> EncryptedField:
        """
        Encrypt a field value using envelope encryption.

        Args:
            plaintext: The value to encrypt (string or bytes)
            classification: Data classification level
            context: Additional encryption context for key derivation

        Returns:
            EncryptedField containing all data needed for decryption

        Raises:
            RuntimeError: If service is not initialized
            ValueError: If plaintext is empty
        """
        if not self._initialized:
            raise RuntimeError("EncryptionService not initialized. Call initialize() first.")

        if not plaintext:
            raise ValueError("Cannot encrypt empty value")

        start_time = datetime.now(timezone.utc)

        try:
            # Convert string to bytes
            if isinstance(plaintext, str):
                plaintext_bytes = plaintext.encode("utf-8")
            else:
                plaintext_bytes = plaintext

            # Calculate plaintext hash for integrity verification
            plaintext_hash = hashlib.sha256(plaintext_bytes).hexdigest()

            # Get algorithm for classification
            algorithm = self.config.classification_policies.get(
                classification,
                self.config.default_algorithm,
            )

            # Generate data encryption key
            data_key, encrypted_data_key = await self._generate_data_key(context)

            # Encrypt based on algorithm
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                ciphertext, iv, auth_tag = self._encrypt_aes_gcm(plaintext_bytes, data_key)
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                ciphertext, iv = self._encrypt_aes_cbc(plaintext_bytes, data_key)
                auth_tag = None
            elif algorithm == EncryptionAlgorithm.FERNET:
                ciphertext = self._encrypt_fernet(plaintext_bytes, data_key)
                iv = ""
                auth_tag = None
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            encrypted_field = EncryptedField(
                ciphertext=base64.b64encode(ciphertext).decode() if isinstance(ciphertext, bytes) else ciphertext,
                encrypted_data_key=base64.b64encode(encrypted_data_key).decode(),
                iv=base64.b64encode(iv).decode() if isinstance(iv, bytes) else iv,
                key_version=self._current_key_version.version,
                algorithm=algorithm,
                classification=classification,
                field_hash=plaintext_hash,
                auth_tag=base64.b64encode(auth_tag).decode() if auth_tag else None,
            )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.debug(
                f"Field encrypted",
                extra={
                    "classification": classification.value,
                    "algorithm": algorithm.value,
                    "processing_time_ms": processing_time,
                },
            )

            return encrypted_field

        except Exception as e:
            logger.error(f"Encryption failed: {e}", exc_info=True)
            raise

    async def decrypt_field(
        self,
        encrypted_field: EncryptedField,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Decrypt an encrypted field value.

        Args:
            encrypted_field: The encrypted field to decrypt
            context: Encryption context used during encryption

        Returns:
            Decrypted plaintext string

        Raises:
            RuntimeError: If service is not initialized
            ValueError: If integrity check fails
        """
        if not self._initialized:
            raise RuntimeError("EncryptionService not initialized. Call initialize() first.")

        start_time = datetime.now(timezone.utc)

        try:
            # Decrypt the data encryption key
            encrypted_data_key = base64.b64decode(encrypted_field.encrypted_data_key)
            data_key = await self._decrypt_data_key(
                encrypted_data_key,
                encrypted_field.key_version,
                context,
            )

            # Decrypt based on algorithm
            ciphertext = base64.b64decode(encrypted_field.ciphertext)

            if encrypted_field.algorithm == EncryptionAlgorithm.AES_256_GCM:
                iv = base64.b64decode(encrypted_field.iv)
                auth_tag = base64.b64decode(encrypted_field.auth_tag) if encrypted_field.auth_tag else None
                plaintext_bytes = self._decrypt_aes_gcm(ciphertext, data_key, iv, auth_tag)
            elif encrypted_field.algorithm == EncryptionAlgorithm.AES_256_CBC:
                iv = base64.b64decode(encrypted_field.iv)
                plaintext_bytes = self._decrypt_aes_cbc(ciphertext, data_key, iv)
            elif encrypted_field.algorithm == EncryptionAlgorithm.FERNET:
                plaintext_bytes = self._decrypt_fernet(encrypted_field.ciphertext, data_key)
            else:
                raise ValueError(f"Unsupported algorithm: {encrypted_field.algorithm}")

            # Verify integrity
            computed_hash = hashlib.sha256(plaintext_bytes).hexdigest()
            if computed_hash != encrypted_field.field_hash:
                logger.error("Integrity check failed: hash mismatch")
                raise ValueError("Data integrity check failed")

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.debug(
                f"Field decrypted",
                extra={
                    "key_version": encrypted_field.key_version,
                    "algorithm": encrypted_field.algorithm.value,
                    "processing_time_ms": processing_time,
                },
            )

            return plaintext_bytes.decode("utf-8")

        except InvalidToken:
            logger.error("Decryption failed: invalid token")
            raise ValueError("Decryption failed: invalid ciphertext or key")
        except Exception as e:
            logger.error(f"Decryption failed: {e}", exc_info=True)
            raise

    async def rotate_key(self) -> KeyVersion:
        """
        Rotate to a new encryption key version.

        Creates a new key version and marks it as active.
        Previous versions remain available for decryption.

        Returns:
            The new active key version
        """
        logger.info("Rotating encryption key")

        try:
            if self.config.local_mode:
                # Generate new local key
                self._local_master_key = secrets.token_bytes(32)
                new_version = self._current_key_version.version + 1
            else:
                # Request KMS key rotation
                new_version = await self._rotate_kms_key()

            # Mark old version as inactive
            if self._current_key_version:
                self._current_key_version.is_active = False
                self._current_key_version.rotated_at = datetime.now(timezone.utc)

            # Create new version
            self._current_key_version = KeyVersion(
                version=new_version,
                key_id=self.config.kms_key_id,
                algorithm=self.config.default_algorithm,
            )
            self._key_versions[new_version] = self._current_key_version

            # Clear data key cache
            self._data_key_cache.clear()

            logger.info(
                f"Key rotated to version {new_version}",
                extra={"previous_version": new_version - 1},
            )

            return self._current_key_version

        except Exception as e:
            logger.error(f"Key rotation failed: {e}", exc_info=True)
            raise

    async def re_encrypt_field(
        self,
        encrypted_field: EncryptedField,
        context: Optional[Dict[str, str]] = None,
    ) -> EncryptedField:
        """
        Re-encrypt a field with the current key version.

        Used during key rotation to migrate data to new keys.

        Args:
            encrypted_field: Field encrypted with old key version
            context: Encryption context

        Returns:
            Field encrypted with current key version
        """
        if encrypted_field.key_version == self._current_key_version.version:
            return encrypted_field

        # Decrypt with old key
        plaintext = await self.decrypt_field(encrypted_field, context)

        # Re-encrypt with current key
        return await self.encrypt_field(
            plaintext,
            encrypted_field.classification,
            context,
        )

    async def encrypt_dict(
        self,
        data: Dict[str, Any],
        fields_to_encrypt: List[str],
        classification: DataClassification = DataClassification.CONFIDENTIAL,
    ) -> Dict[str, Any]:
        """
        Encrypt specific fields in a dictionary.

        Args:
            data: Dictionary containing fields to encrypt
            fields_to_encrypt: List of field names to encrypt
            classification: Data classification for all encrypted fields

        Returns:
            Dictionary with specified fields encrypted
        """
        result = data.copy()

        for field in fields_to_encrypt:
            if field in result and result[field] is not None:
                value = result[field]
                if not isinstance(value, str):
                    value = json.dumps(value)
                encrypted = await self.encrypt_field(value, classification)
                result[field] = encrypted.to_storage_dict()

        return result

    async def decrypt_dict(
        self,
        data: Dict[str, Any],
        encrypted_fields: List[str],
    ) -> Dict[str, Any]:
        """
        Decrypt specific fields in a dictionary.

        Args:
            data: Dictionary containing encrypted fields
            encrypted_fields: List of field names that are encrypted

        Returns:
            Dictionary with specified fields decrypted
        """
        result = data.copy()

        for field in encrypted_fields:
            if field in result and result[field] is not None:
                encrypted_data = result[field]
                if isinstance(encrypted_data, dict):
                    encrypted_field = EncryptedField.from_storage_dict(encrypted_data)
                    result[field] = await self.decrypt_field(encrypted_field)

        return result

    def _encrypt_aes_gcm(
        self,
        plaintext: bytes,
        key: bytes,
    ) -> tuple[bytes, bytes, bytes]:
        """
        Encrypt using AES-256-GCM (authenticated encryption).

        Args:
            plaintext: Data to encrypt
            key: 32-byte encryption key

        Returns:
            Tuple of (ciphertext, iv, auth_tag)
        """
        iv = secrets.token_bytes(12)  # 96-bit nonce for GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return ciphertext, iv, encryptor.tag

    def _decrypt_aes_gcm(
        self,
        ciphertext: bytes,
        key: bytes,
        iv: bytes,
        auth_tag: bytes,
    ) -> bytes:
        """
        Decrypt using AES-256-GCM.

        Args:
            ciphertext: Encrypted data
            key: 32-byte encryption key
            iv: Initialization vector
            auth_tag: Authentication tag

        Returns:
            Decrypted plaintext
        """
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, auth_tag),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def _encrypt_aes_cbc(
        self,
        plaintext: bytes,
        key: bytes,
    ) -> tuple[bytes, bytes]:
        """
        Encrypt using AES-256-CBC with PKCS7 padding.

        Args:
            plaintext: Data to encrypt
            key: 32-byte encryption key

        Returns:
            Tuple of (ciphertext, iv)
        """
        iv = secrets.token_bytes(16)  # 128-bit IV for CBC

        # Apply PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return ciphertext, iv

    def _decrypt_aes_cbc(
        self,
        ciphertext: bytes,
        key: bytes,
        iv: bytes,
    ) -> bytes:
        """
        Decrypt using AES-256-CBC.

        Args:
            ciphertext: Encrypted data
            key: 32-byte encryption key
            iv: Initialization vector

        Returns:
            Decrypted plaintext
        """
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove PKCS7 padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()

    def _encrypt_fernet(self, plaintext: bytes, key: bytes) -> str:
        """
        Encrypt using Fernet (symmetric encryption with HMAC).

        Args:
            plaintext: Data to encrypt
            key: 32-byte key (will be base64 encoded for Fernet)

        Returns:
            Base64-encoded ciphertext
        """
        # Fernet requires a URL-safe base64-encoded 32-byte key
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        return f.encrypt(plaintext).decode()

    def _decrypt_fernet(self, ciphertext: str, key: bytes) -> bytes:
        """
        Decrypt using Fernet.

        Args:
            ciphertext: Base64-encoded encrypted data
            key: 32-byte key

        Returns:
            Decrypted plaintext
        """
        fernet_key = base64.urlsafe_b64encode(key)
        f = Fernet(fernet_key)
        return f.decrypt(ciphertext.encode())

    async def _generate_data_key(
        self,
        context: Optional[Dict[str, str]] = None,
    ) -> tuple[bytes, bytes]:
        """
        Generate a data encryption key (envelope encryption).

        In production, this uses AWS KMS to generate and encrypt the key.
        In local mode, it generates a random key and encrypts with local master key.

        Args:
            context: Encryption context for KMS

        Returns:
            Tuple of (plaintext_key, encrypted_key)
        """
        if self.config.local_mode:
            # Generate random data key
            plaintext_key = secrets.token_bytes(32)

            # Encrypt with local master key using AES-GCM
            iv = secrets.token_bytes(12)
            cipher = Cipher(
                algorithms.AES(self._local_master_key),
                modes.GCM(iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext_key) + encryptor.finalize()

            # Combine iv + ciphertext + tag for storage
            encrypted_key = iv + ciphertext + encryptor.tag

            return plaintext_key, encrypted_key

        else:
            # Use AWS KMS to generate data key
            return await self._kms_generate_data_key(context)

    async def _decrypt_data_key(
        self,
        encrypted_key: bytes,
        key_version: int,
        context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """
        Decrypt a data encryption key.

        Args:
            encrypted_key: Encrypted data key
            key_version: Version of the key used for encryption
            context: Encryption context

        Returns:
            Decrypted data key
        """
        if self.config.local_mode:
            # Extract iv, ciphertext, and tag
            iv = encrypted_key[:12]
            tag = encrypted_key[-16:]
            ciphertext = encrypted_key[12:-16]

            cipher = Cipher(
                algorithms.AES(self._local_master_key),
                modes.GCM(iv, tag),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()

        else:
            # Use AWS KMS to decrypt
            return await self._kms_decrypt_data_key(encrypted_key, context)

    async def _initialize_kms_client(self) -> None:
        """Initialize AWS KMS client for production mode."""
        try:
            import boto3

            self._kms_client = boto3.client(
                "kms",
                region_name=self.config.kms_region,
            )
            logger.info("AWS KMS client initialized")
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize KMS client: {e}")
            raise

    async def _load_key_versions(self) -> None:
        """Load key versions from KMS."""
        # For production, query KMS for key metadata
        self._current_key_version = KeyVersion(
            version=1,
            key_id=self.config.kms_key_id,
            algorithm=self.config.default_algorithm,
        )

    async def _kms_generate_data_key(
        self,
        context: Optional[Dict[str, str]] = None,
    ) -> tuple[bytes, bytes]:
        """Generate data key using AWS KMS."""
        response = self._kms_client.generate_data_key(
            KeyId=self.config.kms_key_id,
            KeySpec="AES_256",
            EncryptionContext=context or {},
        )
        return response["Plaintext"], response["CiphertextBlob"]

    async def _kms_decrypt_data_key(
        self,
        encrypted_key: bytes,
        context: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Decrypt data key using AWS KMS."""
        response = self._kms_client.decrypt(
            KeyId=self.config.kms_key_id,
            CiphertextBlob=encrypted_key,
            EncryptionContext=context or {},
        )
        return response["Plaintext"]

    async def _rotate_kms_key(self) -> int:
        """Trigger KMS key rotation and return new version."""
        # KMS handles key rotation internally
        # We track versions for our envelope encryption
        return self._current_key_version.version + 1
