# -*- coding: utf-8 -*-
"""
Encryption Audit Logger - SEC-003: Encryption at Rest

Emits structured encryption events as JSON log records for Loki ingestion.
Every encryption event (encrypt, decrypt, key rotation, KMS calls) is captured
with a consistent schema that includes correlation IDs, tenant isolation, and
performance metrics.

**SECURITY INVARIANT:** This module NEVER logs sensitive data including:
- Plaintext data being encrypted/decrypted
- Encryption keys (DEKs, KEKs)
- Key material of any kind
- Ciphertext contents

Only opaque identifiers (key version, tenant ID, data class) are included.

Classes:
    - EncryptionAuditEventType: Enumeration of all auditable encryption events.
    - EncryptionAuditEvent: Structured encryption event data class.
    - EncryptionAuditLogger: Async logger that emits JSON events and writes to DB.

Example:
    >>> audit = EncryptionAuditLogger()
    >>> await audit.log_encryption(
    ...     tenant_id="t-corp",
    ...     key_version="v-123",
    ...     data_class="pii",
    ...     duration_ms=1.5,
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event Types
# ---------------------------------------------------------------------------


class EncryptionAuditEventType(str, Enum):
    """Enumeration of all auditable encryption events.

    Values are lowercase snake_case strings suitable for use as Loki labels.
    """

    # Core encryption operations
    ENCRYPTION_PERFORMED = "encryption_performed"
    DECRYPTION_PERFORMED = "decryption_performed"
    ENCRYPTION_FAILED = "encryption_failed"
    DECRYPTION_FAILED = "decryption_failed"

    # Key lifecycle events
    KEY_GENERATED = "key_generated"
    KEY_ROTATED = "key_rotated"
    KEY_ACCESSED = "key_accessed"
    KEY_EXPIRED = "key_expired"
    KEY_REVOKED = "key_revoked"

    # Cache events
    KEY_CACHE_HIT = "key_cache_hit"
    KEY_CACHE_MISS = "key_cache_miss"
    KEY_CACHE_INVALIDATED = "key_cache_invalidated"
    KEY_CACHE_EVICTED = "key_cache_evicted"

    # KMS events
    KMS_CALL = "kms_call"
    KMS_ERROR = "kms_error"
    KMS_ENCRYPT_DEK = "kms_encrypt_dek"
    KMS_DECRYPT_DEK = "kms_decrypt_dek"

    # Field encryption events
    FIELD_ENCRYPTED = "field_encrypted"
    FIELD_DECRYPTED = "field_decrypted"
    BATCH_ENCRYPTION = "batch_encryption"
    BATCH_DECRYPTION = "batch_decryption"


# ---------------------------------------------------------------------------
# Fields that must NEVER appear in log output
# ---------------------------------------------------------------------------

_REDACTED_FIELDS = frozenset({
    "plaintext",
    "ciphertext",
    "key",
    "dek",
    "kek",
    "secret",
    "data_key",
    "encryption_key",
    "decryption_key",
    "key_material",
    "raw_key",
    "private_key",
    "password",
    "token",
    "nonce",
    "auth_tag",
    "iv",
    "aad",
})


# ---------------------------------------------------------------------------
# Event Data Class
# ---------------------------------------------------------------------------


@dataclass
class EncryptionAuditEvent:
    """Structured representation of an encryption audit event.

    Attributes:
        event_type: The type of encryption event.
        operation: Operation name (encrypt, decrypt, generate_key, etc.).
        data_class: Data classification (pii, secret, confidential, internal).
        tenant_id: UUID of the tenant.
        key_version: Version identifier for the encryption key used.
        success: Whether the operation succeeded.
        error_message: Error message if operation failed (sanitized).
        correlation_id: Request correlation / trace ID.
        client_ip: Client IP address.
        duration_ms: Operation duration in milliseconds.
        timestamp: UTC datetime of the event.
        metadata: Additional event-specific key-value pairs.
    """

    event_type: EncryptionAuditEventType
    operation: str
    data_class: Optional[str] = None
    tenant_id: Optional[str] = None
    key_version: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    correlation_id: Optional[str] = None
    client_ip: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a log-safe dictionary.

        Strips any keys in ``metadata`` that match the redaction set.
        NEVER includes plaintext data, keys, or sensitive values.

        Returns:
            Dictionary safe for JSON serialization and log emission.
        """
        # Filter out any sensitive fields from metadata
        safe_metadata = {
            k: v
            for k, v in self.metadata.items()
            if k.lower() not in _REDACTED_FIELDS
        }

        return {
            "event_type": self.event_type.value,
            "event_category": "encryption",
            "operation": self.operation,
            "data_class": self.data_class,
            "tenant_id": self.tenant_id,
            "key_version": self.key_version,
            "success": self.success,
            "error_message": self.error_message,
            "correlation_id": self.correlation_id,
            "client_ip": self.client_ip,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": safe_metadata,
        }


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------


class EncryptionAuditLogger:
    """Emits structured encryption audit events as JSON log records.

    Uses a dedicated ``greenlang.encryption.audit`` logger so that Loki pipeline
    stages can match on the logger name and apply appropriate labels.

    Optionally writes to ``security.encryption_audit_log`` database table
    for long-term retention and compliance queries.

    **SECURITY:** Never logs plaintext data, keys, or sensitive values.
    Only logs metadata about encryption operations.

    Example:
        >>> audit = EncryptionAuditLogger()
        >>> await audit.log_encryption(
        ...     tenant_id="t-1",
        ...     key_version="v-123",
        ...     data_class="pii",
        ...     duration_ms=1.5,
        ... )
    """

    def __init__(self, db_pool: Optional[Any] = None) -> None:
        """Initialize the encryption audit logger.

        Args:
            db_pool: Optional database connection pool for writing to
                security.encryption_audit_log table.
        """
        self._pool = db_pool
        self._audit_logger = logging.getLogger("greenlang.encryption.audit")

    # ------------------------------------------------------------------
    # Core log method
    # ------------------------------------------------------------------

    async def log_event(
        self,
        event_type: EncryptionAuditEventType,
        operation: str,
        *,
        data_class: Optional[str] = None,
        tenant_id: Optional[str] = None,
        key_version: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        correlation_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an encryption audit event.

        1. Emit structured JSON log for Loki ingestion
        2. Write to security.encryption_audit_log (async, fire-and-forget)

        Args:
            event_type: The type of encryption event.
            operation: Operation name (encrypt, decrypt, etc.).
            data_class: Data classification level.
            tenant_id: UUID of the tenant.
            key_version: Version identifier for the key used.
            success: Whether the operation succeeded.
            error_message: Error message if operation failed.
            correlation_id: Request correlation / trace ID.
            client_ip: Client IP address.
            duration_ms: Operation duration in milliseconds.
            metadata: Additional event-specific key-value pairs.
        """
        event = EncryptionAuditEvent(
            event_type=event_type,
            operation=operation,
            data_class=data_class,
            tenant_id=tenant_id,
            key_version=key_version,
            success=success,
            error_message=error_message,
            correlation_id=correlation_id,
            client_ip=client_ip,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        # Determine log level based on success/failure
        level = logging.INFO
        if not success:
            level = logging.WARNING

        payload = event.to_dict()

        # Emit as JSON string with extra fields for Loki labels
        self._audit_logger.log(
            level,
            json.dumps(payload, default=str),
            extra={
                "event_type": event.event_type.value,
                "encryption_result": "success" if success else "failure",
                "tenant_id": event.tenant_id or "",
                "data_class": event.data_class or "",
            },
        )

        # Write to database (fire-and-forget)
        if self._pool:
            asyncio.create_task(self._write_to_db(event))

    async def _write_to_db(self, event: EncryptionAuditEvent) -> None:
        """Write audit event to database.

        Silently fails if database write fails to avoid disrupting
        the main encryption operation.

        Args:
            event: The encryption audit event to persist.
        """
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO security.encryption_audit_log
                    (event_type, data_class, tenant_id, key_version, operation,
                     success, error_message, correlation_id, client_ip, performed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    event.event_type.value,
                    event.data_class,
                    event.tenant_id,
                    event.key_version,
                    event.operation,
                    event.success,
                    event.error_message,
                    event.correlation_id,
                    event.client_ip,
                    event.timestamp,
                )
        except Exception as e:
            logger.warning(
                "Failed to write encryption audit to DB: %s",
                e,
                exc_info=False,
            )

    # ------------------------------------------------------------------
    # Convenience: encryption events
    # ------------------------------------------------------------------

    async def log_encryption(
        self,
        *,
        data_class: Optional[str] = None,
        tenant_id: Optional[str] = None,
        key_version: Optional[str] = None,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a successful encryption operation.

        Args:
            data_class: Data classification level.
            tenant_id: UUID of the tenant.
            key_version: Version of the key used.
            duration_ms: Operation duration in milliseconds.
            correlation_id: Request correlation ID.
            client_ip: Client IP address.
            **kwargs: Additional metadata (sensitive keys auto-redacted).
        """
        await self.log_event(
            EncryptionAuditEventType.ENCRYPTION_PERFORMED,
            "encrypt",
            data_class=data_class,
            tenant_id=tenant_id,
            key_version=key_version,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            client_ip=client_ip,
            success=True,
            metadata=kwargs,
        )

    async def log_decryption(
        self,
        *,
        data_class: Optional[str] = None,
        tenant_id: Optional[str] = None,
        key_version: Optional[str] = None,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a successful decryption operation.

        Args:
            data_class: Data classification level.
            tenant_id: UUID of the tenant.
            key_version: Version of the key used.
            duration_ms: Operation duration in milliseconds.
            correlation_id: Request correlation ID.
            client_ip: Client IP address.
            **kwargs: Additional metadata (sensitive keys auto-redacted).
        """
        await self.log_event(
            EncryptionAuditEventType.DECRYPTION_PERFORMED,
            "decrypt",
            data_class=data_class,
            tenant_id=tenant_id,
            key_version=key_version,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            client_ip=client_ip,
            success=True,
            metadata=kwargs,
        )

    async def log_encryption_failure(
        self,
        error: str,
        *,
        data_class: Optional[str] = None,
        tenant_id: Optional[str] = None,
        key_version: Optional[str] = None,
        correlation_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a failed encryption operation.

        Args:
            error: Sanitized error message (no sensitive data).
            data_class: Data classification level.
            tenant_id: UUID of the tenant.
            key_version: Version of the key attempted.
            correlation_id: Request correlation ID.
            client_ip: Client IP address.
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.ENCRYPTION_FAILED,
            "encrypt",
            data_class=data_class,
            tenant_id=tenant_id,
            key_version=key_version,
            success=False,
            error_message=error,
            correlation_id=correlation_id,
            client_ip=client_ip,
            metadata=kwargs,
        )

    async def log_decryption_failure(
        self,
        error: str,
        *,
        data_class: Optional[str] = None,
        tenant_id: Optional[str] = None,
        key_version: Optional[str] = None,
        correlation_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a failed decryption operation.

        Args:
            error: Sanitized error message (no sensitive data).
            data_class: Data classification level.
            tenant_id: UUID of the tenant.
            key_version: Version of the key attempted.
            correlation_id: Request correlation ID.
            client_ip: Client IP address.
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.DECRYPTION_FAILED,
            "decrypt",
            data_class=data_class,
            tenant_id=tenant_id,
            key_version=key_version,
            success=False,
            error_message=error,
            correlation_id=correlation_id,
            client_ip=client_ip,
            metadata=kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience: key lifecycle events
    # ------------------------------------------------------------------

    async def log_key_generated(
        self,
        *,
        key_version: str,
        tenant_id: Optional[str] = None,
        key_type: str = "dek",
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a key generation event.

        Args:
            key_version: Version identifier for the new key.
            tenant_id: UUID of the tenant.
            key_type: Type of key (dek, kek).
            correlation_id: Request correlation ID.
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.KEY_GENERATED,
            "generate_key",
            key_version=key_version,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            metadata={"key_type": key_type, **kwargs},
        )

    async def log_key_rotated(
        self,
        *,
        new_key_version: str,
        previous_key_version: Optional[str] = None,
        tenant_id: Optional[str] = None,
        key_type: str = "dek",
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a key rotation event.

        Args:
            new_key_version: Version identifier for the new key.
            previous_key_version: Version of the rotated key.
            tenant_id: UUID of the tenant.
            key_type: Type of key (dek, kek).
            correlation_id: Request correlation ID.
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.KEY_ROTATED,
            "rotate_key",
            key_version=new_key_version,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            metadata={
                "key_type": key_type,
                "previous_key_version": previous_key_version,
                **kwargs,
            },
        )

    # ------------------------------------------------------------------
    # Convenience: cache events
    # ------------------------------------------------------------------

    async def log_cache_hit(
        self,
        *,
        key_version: str,
        tenant_id: Optional[str] = None,
        key_type: str = "dek",
        **kwargs: Any,
    ) -> None:
        """Log a DEK cache hit.

        Args:
            key_version: Version of the cached key.
            tenant_id: UUID of the tenant.
            key_type: Type of key (dek).
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.KEY_CACHE_HIT,
            "cache_lookup",
            key_version=key_version,
            tenant_id=tenant_id,
            metadata={"key_type": key_type, **kwargs},
        )

    async def log_cache_miss(
        self,
        *,
        tenant_id: Optional[str] = None,
        key_type: str = "dek",
        **kwargs: Any,
    ) -> None:
        """Log a DEK cache miss.

        Args:
            tenant_id: UUID of the tenant.
            key_type: Type of key (dek).
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.KEY_CACHE_MISS,
            "cache_lookup",
            tenant_id=tenant_id,
            metadata={"key_type": key_type, **kwargs},
        )

    async def log_cache_invalidated(
        self,
        *,
        key_version: Optional[str] = None,
        tenant_id: Optional[str] = None,
        reason: str = "manual",
        **kwargs: Any,
    ) -> None:
        """Log a cache invalidation event.

        Args:
            key_version: Version of the invalidated key (if specific).
            tenant_id: UUID of the tenant.
            reason: Reason for invalidation (manual, rotation, expiry).
            **kwargs: Additional metadata.
        """
        await self.log_event(
            EncryptionAuditEventType.KEY_CACHE_INVALIDATED,
            "cache_invalidate",
            key_version=key_version,
            tenant_id=tenant_id,
            metadata={"reason": reason, **kwargs},
        )

    # ------------------------------------------------------------------
    # Convenience: KMS events
    # ------------------------------------------------------------------

    async def log_kms_call(
        self,
        operation: str,
        *,
        success: bool = True,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a KMS API call.

        Args:
            operation: KMS operation (encrypt, decrypt, generate_data_key).
            success: Whether the call succeeded.
            duration_ms: Call duration in milliseconds.
            error_message: Error message if call failed.
            tenant_id: UUID of the tenant.
            correlation_id: Request correlation ID.
            **kwargs: Additional metadata.
        """
        event_type = (
            EncryptionAuditEventType.KMS_CALL
            if success
            else EncryptionAuditEventType.KMS_ERROR
        )

        await self.log_event(
            event_type,
            f"kms_{operation}",
            tenant_id=tenant_id,
            success=success,
            duration_ms=duration_ms,
            error_message=error_message,
            correlation_id=correlation_id,
            metadata={"kms_operation": operation, **kwargs},
        )


__all__ = [
    "EncryptionAuditEventType",
    "EncryptionAuditEvent",
    "EncryptionAuditLogger",
]
