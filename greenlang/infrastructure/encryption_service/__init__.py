# -*- coding: utf-8 -*-
"""
GreenLang Encryption Service - SEC-003: Encryption at Rest

Production-grade encryption service implementing AES-256-GCM with envelope
encryption via AWS KMS for key management. Provides field-level encryption
for sensitive database columns with searchable HMAC indexes.

Follows the GreenLang zero-hallucination principle: all cryptographic
operations are deterministic, using the ``cryptography`` library for
AES-256-GCM and AWS KMS for key management.

Sub-modules:
    encryption_service   - Core AES-256-GCM encryption operations.
    envelope_encryption  - KMS-based envelope encryption (DEK wrapping).
    key_management       - DEK caching and lifecycle management.
    field_encryption     - Database field-level encryption utilities.
    encryption_audit     - Audit logging for encryption operations.
    encryption_metrics   - Prometheus metrics for encryption observability.
    api                  - REST API endpoints for encryption service.

Quick start:
    >>> from greenlang.infrastructure.encryption_service import (
    ...     EncryptionService,
    ...     EncryptionServiceConfig,
    ...     FieldEncryptor,
    ... )
    >>> config = EncryptionServiceConfig(kms_key_id="alias/greenlang-cmk")
    >>> svc = await EncryptionService.create(config)
    >>> encrypted = await svc.encrypt(b"sensitive data", {"tenant_id": "t-1"})
    >>> plaintext = await svc.decrypt(encrypted, {"tenant_id": "t-1"})

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EncryptionServiceConfig:
    """Top-level configuration for the Encryption Service (SEC-003).

    Attributes:
        kms_key_id: AWS KMS Customer Master Key ARN or alias.
            Example: ``"alias/greenlang-cmk"`` or full ARN.
        kms_region: AWS region for KMS operations. Uses default if None.
        dek_cache_ttl_seconds: TTL for cached Data Encryption Keys.
            Default 300 (5 minutes) to limit exposure.
        dek_cache_max_size: Maximum number of DEKs to cache in memory.
            Default 1000.
        encryption_context_required: Require encryption context for all
            operations. Default True for audit trail compliance.
        enable_metrics: Emit Prometheus metrics for operations.
        enable_audit: Log encryption operations for audit trail.
        max_retries: Maximum retry attempts for KMS operations.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    kms_key_id: str
    kms_region: Optional[str] = None
    dek_cache_ttl_seconds: int = 300
    dek_cache_max_size: int = 1000
    encryption_context_required: bool = True
    enable_metrics: bool = True
    enable_audit: bool = True
    max_retries: int = 3
    retry_base_delay: float = 1.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class EncryptedData:
    """Result of an encryption operation.

    Contains all components needed to decrypt the data later:
    ciphertext, nonce, authentication tag, and the KMS-wrapped DEK.

    Attributes:
        ciphertext: The encrypted data bytes.
        nonce: 12-byte (96-bit) nonce used for AES-GCM encryption.
            NEVER reuse a nonce with the same key.
        auth_tag: 16-byte (128-bit) authentication tag for integrity.
        encrypted_dek: KMS-wrapped Data Encryption Key.
        key_version: Unique identifier for key version tracking.
        encryption_context: Key-value context bound to the encryption.
        algorithm: Encryption algorithm identifier. Default AES-256-GCM.
    """

    ciphertext: bytes
    nonce: bytes
    auth_tag: bytes
    encrypted_dek: bytes
    key_version: str
    encryption_context: Dict[str, str] = field(default_factory=dict)
    algorithm: str = "AES-256-GCM"


@dataclass
class DecryptedData:
    """Result of a decryption operation.

    Attributes:
        plaintext: The decrypted data bytes.
        key_version: Version of the key used for decryption.
    """

    plaintext: bytes
    key_version: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EncryptionError(Exception):
    """Base exception for encryption service errors."""

    pass


class EncryptionKeyError(EncryptionError):
    """Error related to encryption key operations."""

    pass


class DecryptionError(EncryptionError):
    """Error during decryption operations."""

    pass


class ContextMismatchError(DecryptionError):
    """Encryption context does not match during decryption."""

    pass


class IntegrityError(DecryptionError):
    """Authentication tag verification failed - data may be tampered."""

    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from greenlang.infrastructure.encryption_service.encryption_service import (  # noqa: E402
    EncryptionService,
)
from greenlang.infrastructure.encryption_service.envelope_encryption import (  # noqa: E402
    EnvelopeEncryptionService,
)
from greenlang.infrastructure.encryption_service.key_management import (  # noqa: E402
    KeyManager,
    DEKCache,
    CachedDEK,
)
from greenlang.infrastructure.encryption_service.field_encryption import (  # noqa: E402
    FieldEncryptor,
)
from greenlang.infrastructure.encryption_service.encryption_audit import (  # noqa: E402
    EncryptionAuditEventType,
    EncryptionAuditEvent,
    EncryptionAuditLogger,
)
from greenlang.infrastructure.encryption_service.encryption_metrics import (  # noqa: E402
    EncryptionMetrics,
)

__all__ = [
    # Config
    "EncryptionServiceConfig",
    # Data classes
    "EncryptedData",
    "DecryptedData",
    # Exceptions
    "EncryptionError",
    "EncryptionKeyError",
    "DecryptionError",
    "ContextMismatchError",
    "IntegrityError",
    # Services
    "EncryptionService",
    "EnvelopeEncryptionService",
    "KeyManager",
    "DEKCache",
    "CachedDEK",
    "FieldEncryptor",
    # Audit
    "EncryptionAuditEventType",
    "EncryptionAuditEvent",
    "EncryptionAuditLogger",
    # Metrics
    "EncryptionMetrics",
]
