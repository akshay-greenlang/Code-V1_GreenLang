# -*- coding: utf-8 -*-
"""
Field-Level Encryption - Database Column Encryption (SEC-003)

Provides utilities for encrypting/decrypting individual database
fields with support for searchable HMAC indexes. Enables transparent
encryption of sensitive columns (PII, secrets) while maintaining
query capability via blind indexes.

Features:
- Encrypt/decrypt individual field values
- Pack/unpack encrypted data for database storage
- Searchable HMAC indexes for encrypted columns
- Support for various data types (str, int, dict, etc.)

Security considerations:
- Each field is encrypted with tenant + field name in context
- HMAC indexes enable equality searches without revealing plaintext
- Different index keys per field prevent cross-column analysis
- All serialization is deterministic for consistent encryption

Example:
    >>> encryptor = FieldEncryptor(encryption_service)
    >>> encrypted = await encryptor.encrypt_field(
    ...     "john.doe@example.com",
    ...     field_name="email",
    ...     tenant_id="t-1",
    ...     data_class="pii",
    ... )
    >>> original = await encryptor.decrypt_field(
    ...     encrypted, "email", "t-1", "pii"
    ... )
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from greenlang.infrastructure.encryption_service import (
        EncryptedData,
    )
    from greenlang.infrastructure.encryption_service.encryption_service import (
        EncryptionService,
    )

logger = logging.getLogger(__name__)


class FieldEncryptor:
    """Encrypt/decrypt individual database fields.

    Provides transparent encryption for sensitive columns with support
    for searchable HMAC indexes. Uses the core EncryptionService for
    AES-256-GCM encryption with envelope pattern.

    The encryption context binds each encrypted value to:
    - tenant_id: Multi-tenancy isolation
    - field_name: Column name (prevents moving encrypted data)
    - data_class: Classification level (pii, secret, etc.)

    Attributes:
        encryption: Core EncryptionService instance.

    Example:
        >>> encryptor = FieldEncryptor(encryption_service)
        >>> encrypted_email = await encryptor.encrypt_field(
        ...     "user@example.com",
        ...     field_name="email",
        ...     tenant_id="t-acme",
        ... )
    """

    def __init__(self, encryption_service: EncryptionService) -> None:
        """Initialize FieldEncryptor.

        Args:
            encryption_service: Core EncryptionService for encryption ops.
        """
        self._encryption = encryption_service
        logger.debug("FieldEncryptor initialized")

    async def encrypt_field(
        self,
        value: Any,
        field_name: str,
        tenant_id: str,
        data_class: str = "pii",
    ) -> str:
        """Encrypt a field value for database storage.

        The value is serialized to bytes, encrypted with AES-256-GCM,
        and packed into a base64-encoded string suitable for VARCHAR
        columns.

        Args:
            value: Value to encrypt (str, int, dict, etc.).
            field_name: Database column name.
            tenant_id: Tenant identifier for isolation.
            data_class: Data classification (pii, secret, etc.).

        Returns:
            Base64-encoded encrypted value for database storage.

        Raises:
            EncryptionError: If encryption fails.
        """
        start_time = datetime.now(timezone.utc)

        # Serialize value to bytes
        serialized = self._serialize(value)

        # Build encryption context
        context = {
            "tenant_id": tenant_id,
            "field_name": field_name,
            "data_class": data_class,
        }

        # Encrypt
        encrypted = await self._encryption.encrypt(serialized, context)

        # Pack into storable format
        packed = self._pack_encrypted(encrypted)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.debug(
            "Field encrypted  field=%s  tenant=%s  class=%s  "
            "input_size=%d  output_size=%d  elapsed=%.1fms",
            field_name,
            tenant_id,
            data_class,
            len(serialized),
            len(packed),
            elapsed_ms,
        )

        return packed

    async def decrypt_field(
        self,
        encrypted_value: str,
        field_name: str,
        tenant_id: str,
        data_class: str = "pii",
    ) -> Any:
        """Decrypt a field value from database storage.

        Args:
            encrypted_value: Base64-encoded encrypted value.
            field_name: Database column name.
            tenant_id: Tenant identifier.
            data_class: Data classification.

        Returns:
            Original decrypted value.

        Raises:
            DecryptionError: If decryption fails.
            ContextMismatchError: If context doesn't match.
        """
        start_time = datetime.now(timezone.utc)

        # Unpack
        encrypted = self._unpack_encrypted(encrypted_value)

        # Build context
        context = {
            "tenant_id": tenant_id,
            "field_name": field_name,
            "data_class": data_class,
        }

        # Decrypt
        plaintext = await self._encryption.decrypt(encrypted, context)

        # Deserialize
        value = self._deserialize(plaintext)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.debug(
            "Field decrypted  field=%s  tenant=%s  elapsed=%.1fms",
            field_name,
            tenant_id,
            elapsed_ms,
        )

        return value

    def create_search_index(
        self,
        value: Any,
        field_name: str,
        index_key: bytes,
    ) -> str:
        """Create searchable HMAC index for encrypted field.

        Allows equality searches on encrypted columns without revealing
        the plaintext. The HMAC output is deterministic for the same
        input, enabling blind index lookups.

        Security notes:
        - Use a different index_key per field to prevent cross-column analysis
        - Index reveals equality only (same hash = same value)
        - Timing-safe comparison should be used when querying

        Args:
            value: Original value to index.
            field_name: Column name (included in HMAC input).
            index_key: Secret key for HMAC (derived from master key).

        Returns:
            Hex-encoded HMAC suitable for database indexing.

        Example:
            >>> index = encryptor.create_search_index(
            ...     "user@example.com",
            ...     "email",
            ...     index_key,
            ... )
            >>> # Query: SELECT * FROM users WHERE email_index = %s
        """
        # Serialize value
        serialized = self._serialize(value)

        # Include field name to prevent index collision across fields
        message = f"{field_name}:{serialized.hex()}".encode("utf-8")

        # HMAC-SHA256
        h = hmac.new(index_key, message, hashlib.sha256)
        index_value = h.hexdigest()

        logger.debug(
            "Search index created  field=%s  index_prefix=%s...",
            field_name,
            index_value[:8],
        )

        return index_value

    @staticmethod
    def derive_index_key(
        master_key: bytes,
        field_name: str,
        tenant_id: str,
    ) -> bytes:
        """Derive a field-specific index key from master key.

        Uses HKDF-like key derivation (simplified) to create unique
        index keys per field and tenant. This prevents cross-field
        and cross-tenant index analysis.

        Args:
            master_key: Base key material (32+ bytes).
            field_name: Database column name.
            tenant_id: Tenant identifier.

        Returns:
            32-byte derived key for HMAC operations.
        """
        # Combine context
        context = f"index:{tenant_id}:{field_name}".encode("utf-8")

        # HMAC-based key derivation
        derived = hmac.new(master_key, context, hashlib.sha256).digest()

        return derived

    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes.

        Handles common types: bytes, str, numbers, dicts, lists.

        Args:
            value: Value to serialize.

        Returns:
            UTF-8 encoded bytes.
        """
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, (int, float, bool)):
            return json.dumps(value).encode("utf-8")
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        if value is None:
            return b"null"

        # Fallback to string representation
        return str(value).encode("utf-8")

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to original value.

        Attempts JSON parsing first, falls back to string.

        Args:
            data: UTF-8 encoded bytes.

        Returns:
            Deserialized value.
        """
        try:
            text = data.decode("utf-8")
            # Try JSON parse (handles numbers, bools, dicts, lists, null)
            return json.loads(text)
        except json.JSONDecodeError:
            # Plain string
            return text
        except UnicodeDecodeError:
            # Binary data, return as-is
            return data

    def _pack_encrypted(self, encrypted: EncryptedData) -> str:
        """Pack EncryptedData into storable base64 string.

        Format: Base64(JSON({
            ct: base64(ciphertext),
            n: base64(nonce),
            t: base64(auth_tag),
            k: base64(encrypted_dek),
            v: key_version,
            a: algorithm,
        }))

        Args:
            encrypted: EncryptedData to pack.

        Returns:
            Base64-encoded string.
        """
        packed = {
            "ct": base64.b64encode(encrypted.ciphertext).decode("ascii"),
            "n": base64.b64encode(encrypted.nonce).decode("ascii"),
            "t": base64.b64encode(encrypted.auth_tag).decode("ascii"),
            "k": base64.b64encode(encrypted.encrypted_dek).decode("ascii"),
            "v": encrypted.key_version,
            "a": encrypted.algorithm,
        }

        json_str = json.dumps(packed, separators=(",", ":"))
        return base64.b64encode(json_str.encode("utf-8")).decode("ascii")

    def _unpack_encrypted(self, packed: str) -> EncryptedData:
        """Unpack stored base64 string to EncryptedData.

        Args:
            packed: Base64-encoded string from _pack_encrypted.

        Returns:
            EncryptedData instance.

        Raises:
            ValueError: If packed data is malformed.
        """
        from greenlang.infrastructure.encryption_service import EncryptedData

        try:
            json_str = base64.b64decode(packed).decode("utf-8")
            data = json.loads(json_str)

            return EncryptedData(
                ciphertext=base64.b64decode(data["ct"]),
                nonce=base64.b64decode(data["n"]),
                auth_tag=base64.b64decode(data["t"]),
                encrypted_dek=base64.b64decode(data["k"]),
                key_version=data["v"],
                encryption_context={},  # Context rebuilt from function args
                algorithm=data.get("a", "AES-256-GCM"),
            )

        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid encrypted field format: {e}") from e


# ---------------------------------------------------------------------------
# SQLAlchemy Integration (Optional)
# ---------------------------------------------------------------------------

try:
    from sqlalchemy import TypeDecorator, String, event
    from sqlalchemy.engine import Engine

    class EncryptedString(TypeDecorator):
        """SQLAlchemy type for transparently encrypted string columns.

        Note: This type decorator provides a placeholder for integration.
        Full transparent encryption requires async context which SQLAlchemy's
        sync type system doesn't support. For production use, encrypt/decrypt
        at the application layer using FieldEncryptor directly.

        For async applications, use FieldEncryptor methods explicitly
        before/after ORM operations.

        Usage (sync fallback):
            class User(Base):
                email = Column(EncryptedString(512, "email"), nullable=False)

        For full async encryption:
            # Before save
            user.email = await field_encryptor.encrypt_field(
                email, "email", tenant_id
            )
            session.add(user)

            # After load
            email = await field_encryptor.decrypt_field(
                user.email, "email", tenant_id
            )
        """

        impl = String
        cache_ok = False  # Encryption makes caching unsafe

        def __init__(
            self,
            length: int,
            field_name: str,
            data_class: str = "pii",
        ) -> None:
            """Initialize EncryptedString type.

            Args:
                length: Maximum string length for database column.
                field_name: Column name for encryption context.
                data_class: Data classification (pii, secret, etc.).
            """
            super().__init__(length)
            self.field_name = field_name
            self.data_class = data_class

        def process_bind_param(self, value, dialect):
            """Process value before database insert/update.

            Note: This is a sync method. For full async encryption,
            use FieldEncryptor.encrypt_field() at the application layer.
            """
            if value is None:
                return None

            # In sync context, we can't use async encryption
            # Return value as-is - encryption should be done at app layer
            logger.warning(
                "EncryptedString.process_bind_param called in sync context. "
                "For proper encryption, use FieldEncryptor.encrypt_field() "
                "before ORM operations."
            )
            return value

        def process_result_value(self, value, dialect):
            """Process value after database select.

            Note: This is a sync method. For full async decryption,
            use FieldEncryptor.decrypt_field() at the application layer.
            """
            if value is None:
                return None

            # In sync context, we can't use async decryption
            # Return value as-is - decryption should be done at app layer
            return value

    logger.debug("SQLAlchemy EncryptedString type registered")

except ImportError:
    # SQLAlchemy not available
    EncryptedString = None  # type: ignore
    logger.debug("SQLAlchemy not available - EncryptedString type not registered")


# ---------------------------------------------------------------------------
# Batch Operations
# ---------------------------------------------------------------------------


class BatchFieldEncryptor:
    """Batch encryption/decryption for bulk operations.

    Provides efficient batch processing of multiple fields, useful
    for bulk imports, exports, or migrations.

    Example:
        >>> batch = BatchFieldEncryptor(field_encryptor)
        >>> encrypted_records = await batch.encrypt_records(
        ...     records,
        ...     fields=["email", "phone", "ssn"],
        ...     tenant_id="t-1",
        ... )
    """

    def __init__(self, field_encryptor: FieldEncryptor) -> None:
        """Initialize BatchFieldEncryptor.

        Args:
            field_encryptor: FieldEncryptor instance.
        """
        self._encryptor = field_encryptor

    async def encrypt_records(
        self,
        records: list[Dict[str, Any]],
        fields: list[str],
        tenant_id: str,
        data_class: str = "pii",
    ) -> list[Dict[str, Any]]:
        """Encrypt specified fields in multiple records.

        Args:
            records: List of record dictionaries.
            fields: Field names to encrypt.
            tenant_id: Tenant identifier.
            data_class: Data classification.

        Returns:
            Records with specified fields encrypted.
        """
        start_time = datetime.now(timezone.utc)
        encrypted_count = 0

        result = []
        for record in records:
            encrypted_record = record.copy()
            for field in fields:
                if field in encrypted_record and encrypted_record[field] is not None:
                    encrypted_record[field] = await self._encryptor.encrypt_field(
                        encrypted_record[field],
                        field,
                        tenant_id,
                        data_class,
                    )
                    encrypted_count += 1
            result.append(encrypted_record)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            "Batch encryption completed  records=%d  fields=%d  "
            "encrypted=%d  elapsed=%.1fms",
            len(records),
            len(fields),
            encrypted_count,
            elapsed_ms,
        )

        return result

    async def decrypt_records(
        self,
        records: list[Dict[str, Any]],
        fields: list[str],
        tenant_id: str,
        data_class: str = "pii",
    ) -> list[Dict[str, Any]]:
        """Decrypt specified fields in multiple records.

        Args:
            records: List of record dictionaries.
            fields: Field names to decrypt.
            tenant_id: Tenant identifier.
            data_class: Data classification.

        Returns:
            Records with specified fields decrypted.
        """
        start_time = datetime.now(timezone.utc)
        decrypted_count = 0

        result = []
        for record in records:
            decrypted_record = record.copy()
            for field in fields:
                if field in decrypted_record and decrypted_record[field] is not None:
                    try:
                        decrypted_record[field] = await self._encryptor.decrypt_field(
                            decrypted_record[field],
                            field,
                            tenant_id,
                            data_class,
                        )
                        decrypted_count += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to decrypt field %s in record: %s",
                            field,
                            e,
                        )
                        # Keep encrypted value on failure
            result.append(decrypted_record)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            "Batch decryption completed  records=%d  fields=%d  "
            "decrypted=%d  elapsed=%.1fms",
            len(records),
            len(fields),
            decrypted_count,
            elapsed_ms,
        )

        return result
