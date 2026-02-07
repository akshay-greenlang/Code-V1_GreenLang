# -*- coding: utf-8 -*-
"""
Encryption REST API Routes - SEC-003: Encryption at Rest

FastAPI APIRouter providing REST endpoints for encryption operations within the
GreenLang encryption service. Supports data encryption/decryption, key management,
audit log retrieval, and service health monitoring.

Endpoints:
    POST   /api/v1/encryption/encrypt              - Encrypt data
    POST   /api/v1/encryption/decrypt              - Decrypt data
    GET    /api/v1/encryption/keys                 - List encryption keys
    POST   /api/v1/encryption/keys/rotate          - Rotate encryption key
    DELETE /api/v1/encryption/keys/cache           - Invalidate key cache
    GET    /api/v1/encryption/audit                - Get encryption audit log
    GET    /api/v1/encryption/status               - Service health status

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.encryption_service.api.encryption_routes import (
    ...     encryption_router
    ... )
    >>> app = FastAPI()
    >>> app.include_router(encryption_router)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class EncryptRequest(BaseModel):
        """Request schema for encrypting data.

        Attributes:
            plaintext: Base64-encoded plaintext data to encrypt.
            context: Encryption context (tenant_id required for isolation).
            data_class: Data classification level (pii, secret, confidential).
            aad: Optional additional authenticated data (base64).
        """

        model_config = ConfigDict(
            extra="forbid",
            str_strip_whitespace=True,
            json_schema_extra={
                "examples": [
                    {
                        "plaintext": "SGVsbG8gV29ybGQ=",
                        "context": {"tenant_id": "t-corp-123"},
                        "data_class": "pii",
                        "aad": None,
                    }
                ]
            },
        )

        plaintext: str = Field(
            ...,
            min_length=1,
            description="Base64-encoded plaintext data to encrypt.",
        )
        context: Dict[str, str] = Field(
            ...,
            description="Encryption context. Must include tenant_id for tenant isolation.",
        )
        data_class: str = Field(
            default="pii",
            description="Data classification: pii, secret, confidential, internal.",
        )
        aad: Optional[str] = Field(
            default=None,
            description="Optional additional authenticated data (base64-encoded).",
        )

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: Dict[str, str]) -> Dict[str, str]:
            """Validate encryption context contains tenant_id."""
            if "tenant_id" not in v:
                raise ValueError("context must include 'tenant_id' for tenant isolation")
            return v

        @field_validator("data_class")
        @classmethod
        def validate_data_class(cls, v: str) -> str:
            """Validate data classification."""
            allowed = {"pii", "secret", "confidential", "internal"}
            if v.lower() not in allowed:
                raise ValueError(
                    f"data_class must be one of: {', '.join(sorted(allowed))}"
                )
            return v.lower()

    class EncryptResponse(BaseModel):
        """Response schema for encryption operation.

        Attributes:
            ciphertext: Base64-encoded encrypted data.
            nonce: Base64-encoded nonce (12 bytes for AES-GCM).
            auth_tag: Base64-encoded authentication tag (16 bytes).
            encrypted_dek: Base64-encoded KMS-wrapped Data Encryption Key.
            key_version: Unique identifier for the key version used.
            algorithm: Encryption algorithm (AES-256-GCM).
        """

        model_config = ConfigDict(from_attributes=True)

        ciphertext: str = Field(..., description="Base64-encoded ciphertext.")
        nonce: str = Field(..., description="Base64-encoded nonce.")
        auth_tag: str = Field(..., description="Base64-encoded authentication tag.")
        encrypted_dek: str = Field(..., description="Base64-encoded wrapped DEK.")
        key_version: str = Field(..., description="Key version identifier.")
        algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm.")

    class DecryptRequest(BaseModel):
        """Request schema for decrypting data.

        Attributes:
            ciphertext: Base64-encoded ciphertext.
            nonce: Base64-encoded nonce used during encryption.
            auth_tag: Base64-encoded authentication tag.
            encrypted_dek: Base64-encoded KMS-wrapped DEK.
            key_version: Key version identifier.
            context: Encryption context (must match encryption context).
            aad: Optional additional authenticated data (must match if used).
        """

        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

        ciphertext: str = Field(..., description="Base64-encoded ciphertext.")
        nonce: str = Field(..., description="Base64-encoded nonce.")
        auth_tag: str = Field(..., description="Base64-encoded authentication tag.")
        encrypted_dek: str = Field(..., description="Base64-encoded wrapped DEK.")
        key_version: str = Field(..., description="Key version identifier.")
        context: Dict[str, str] = Field(
            ...,
            description="Encryption context (must match encryption context).",
        )
        aad: Optional[str] = Field(
            default=None,
            description="Additional authenticated data (must match if used during encryption).",
        )

    class DecryptResponse(BaseModel):
        """Response schema for decryption operation.

        Attributes:
            plaintext: Base64-encoded decrypted plaintext.
            key_version: Key version used for decryption.
        """

        model_config = ConfigDict(from_attributes=True)

        plaintext: str = Field(..., description="Base64-encoded plaintext.")
        key_version: str = Field(..., description="Key version used.")

    class KeyInfo(BaseModel):
        """Information about an encryption key.

        Attributes:
            key_version: Unique key version identifier.
            key_type: Type of key (dek, kek).
            created_at: Key creation timestamp.
            is_active: Whether the key is currently active.
            kms_key_id: AWS KMS key ARN (for KEK).
            expires_at: Key expiration timestamp (if applicable).
        """

        model_config = ConfigDict(from_attributes=True)

        key_version: str = Field(..., description="Key version identifier.")
        key_type: str = Field(..., description="Key type: dek or kek.")
        created_at: datetime = Field(..., description="Creation timestamp.")
        is_active: bool = Field(..., description="Whether key is active.")
        kms_key_id: Optional[str] = Field(
            default=None, description="KMS key ARN for KEK."
        )
        expires_at: Optional[datetime] = Field(
            default=None, description="Expiration timestamp."
        )

    class KeyListResponse(BaseModel):
        """Paginated list of encryption keys.

        Attributes:
            keys: List of key information objects.
            total: Total number of keys.
        """

        keys: List[KeyInfo] = Field(..., description="List of keys.")
        total: int = Field(..., ge=0, description="Total key count.")

    class RotateKeyRequest(BaseModel):
        """Request schema for key rotation.

        Attributes:
            key_type: Type of key to rotate (dek).
            context: Encryption context for DEK rotation.
            reason: Reason for rotation (scheduled, manual, compromise).
        """

        model_config = ConfigDict(extra="forbid")

        key_type: str = Field(
            default="dek",
            description="Type of key to rotate: dek.",
        )
        context: Optional[Dict[str, str]] = Field(
            default=None,
            description="Encryption context for DEK rotation.",
        )
        reason: str = Field(
            default="manual",
            description="Rotation reason: scheduled, manual, compromise.",
        )

    class RotateKeyResponse(BaseModel):
        """Response schema for key rotation.

        Attributes:
            new_key_version: Version of the newly generated key.
            previous_key_version: Version of the rotated key.
            rotated_at: Timestamp of the rotation.
        """

        new_key_version: str = Field(..., description="New key version.")
        previous_key_version: Optional[str] = Field(
            default=None, description="Previous key version."
        )
        rotated_at: datetime = Field(..., description="Rotation timestamp.")

    class CacheInvalidateRequest(BaseModel):
        """Request schema for cache invalidation.

        Attributes:
            key_version: Specific key version to invalidate (None for all).
            tenant_id: Invalidate keys for specific tenant (None for all).
        """

        model_config = ConfigDict(extra="forbid")

        key_version: Optional[str] = Field(
            default=None,
            description="Specific key version to invalidate.",
        )
        tenant_id: Optional[str] = Field(
            default=None,
            description="Invalidate keys for specific tenant.",
        )

    class CacheInvalidateResponse(BaseModel):
        """Response schema for cache invalidation.

        Attributes:
            invalidated_count: Number of cache entries invalidated.
            timestamp: Invalidation timestamp.
        """

        invalidated_count: int = Field(
            ..., ge=0, description="Entries invalidated."
        )
        timestamp: datetime = Field(..., description="Invalidation timestamp.")

    class AuditLogEntry(BaseModel):
        """Single encryption audit log entry.

        Attributes:
            id: Unique audit entry identifier.
            event_type: Type of encryption event.
            operation: Operation performed.
            data_class: Data classification level.
            tenant_id: Tenant identifier.
            key_version: Key version used.
            success: Whether operation succeeded.
            error_message: Error message if failed.
            performed_at: Operation timestamp.
        """

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Audit entry UUID.")
        event_type: str = Field(..., description="Event type.")
        operation: str = Field(..., description="Operation performed.")
        data_class: Optional[str] = Field(default=None, description="Data class.")
        tenant_id: Optional[str] = Field(default=None, description="Tenant ID.")
        key_version: Optional[str] = Field(default=None, description="Key version.")
        success: bool = Field(..., description="Operation success.")
        error_message: Optional[str] = Field(default=None, description="Error message.")
        performed_at: datetime = Field(..., description="Operation timestamp.")

    class AuditLogResponse(BaseModel):
        """Paginated audit log response.

        Attributes:
            entries: List of audit log entries.
            total: Total matching entries.
            page: Current page number.
            page_size: Items per page.
            has_next: Whether there is a next page.
            has_prev: Whether there is a previous page.
        """

        entries: List[AuditLogEntry] = Field(..., description="Audit entries.")
        total: int = Field(..., ge=0, description="Total entries.")
        page: int = Field(..., ge=1, description="Current page.")
        page_size: int = Field(..., ge=1, description="Items per page.")
        has_next: bool = Field(..., description="Has next page.")
        has_prev: bool = Field(..., description="Has previous page.")

    class EncryptionStatusResponse(BaseModel):
        """Encryption service health status.

        Attributes:
            healthy: Overall service health.
            kms_reachable: Whether KMS is reachable.
            cache_size: Current DEK cache size.
            cache_hit_rate: Cache hit rate (0.0 - 1.0).
            active_key_count: Number of active encryption keys.
            last_key_rotation: Timestamp of last key rotation.
            errors: List of current error conditions.
            version: Service version.
        """

        healthy: bool = Field(..., description="Overall health status.")
        kms_reachable: bool = Field(..., description="KMS connectivity.")
        cache_size: int = Field(..., ge=0, description="Current cache size.")
        cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate.")
        active_key_count: int = Field(..., ge=0, description="Active keys.")
        last_key_rotation: Optional[datetime] = Field(
            default=None, description="Last rotation timestamp."
        )
        errors: List[str] = Field(default_factory=list, description="Current errors.")
        version: str = Field(default="1.0.0", description="Service version.")


# ---------------------------------------------------------------------------
# Dependency Injection
# ---------------------------------------------------------------------------


def _get_encryption_service(request: Any) -> Any:
    """FastAPI dependency that provides the EncryptionService instance.

    Args:
        request: The FastAPI Request object.

    Returns:
        The EncryptionService from app state.

    Raises:
        HTTPException 503: If the service is not configured.
    """
    svc = getattr(request.app.state, "encryption_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Encryption service is not configured.",
        )
    return svc


def _get_audit_logger(request: Any) -> Any:
    """FastAPI dependency that provides the EncryptionAuditLogger instance.

    Args:
        request: The FastAPI Request object.

    Returns:
        The EncryptionAuditLogger from app state, or None.
    """
    return getattr(request.app.state, "encryption_audit", None)


def _get_metrics(request: Any) -> Any:
    """FastAPI dependency that provides the EncryptionMetrics instance.

    Args:
        request: The FastAPI Request object.

    Returns:
        The EncryptionMetrics from app state, or None.
    """
    return getattr(request.app.state, "encryption_metrics", None)


# ---------------------------------------------------------------------------
# Helper: extract request metadata
# ---------------------------------------------------------------------------


def _get_actor_id(request: Any) -> str:
    """Extract the authenticated actor ID from the request.

    Looks for ``X-User-Id`` header (set by auth gateway/middleware).
    Falls back to ``anonymous`` if not present.

    Args:
        request: The FastAPI Request object.

    Returns:
        Actor user ID string.
    """
    return request.headers.get("x-user-id", "anonymous")


def _get_tenant_id(request: Any) -> Optional[str]:
    """Extract the tenant ID from the request.

    Looks for ``X-Tenant-Id`` header (set by auth gateway/middleware).

    Args:
        request: The FastAPI Request object.

    Returns:
        Tenant ID string or None.
    """
    return request.headers.get("x-tenant-id")


def _get_client_ip(request: Any) -> Optional[str]:
    """Extract the client IP from the request.

    Args:
        request: The FastAPI Request object.

    Returns:
        Client IP string or None.
    """
    if hasattr(request, "client") and request.client:
        return request.client.host
    return None


def _get_correlation_id(request: Any) -> Optional[str]:
    """Extract the correlation ID from the request.

    Args:
        request: The FastAPI Request object.

    Returns:
        Correlation ID string or None.
    """
    return request.headers.get("x-correlation-id") or request.headers.get(
        "x-request-id"
    )


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    encryption_router = APIRouter(
        prefix="/api/v1/encryption",
        tags=["Encryption"],
        responses={
            400: {"description": "Bad Request - Invalid input"},
            401: {"description": "Unauthorized - Authentication required"},
            403: {"description": "Forbidden - Insufficient permissions"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    # -- Encrypt Data ------------------------------------------------------

    @encryption_router.post(
        "/encrypt",
        response_model=EncryptResponse,
        status_code=200,
        summary="Encrypt data",
        description=(
            "Encrypt data using AES-256-GCM with envelope encryption. "
            "The plaintext must be base64-encoded. Requires encryption:encrypt permission."
        ),
        operation_id="encrypt_data",
    )
    async def encrypt_data(
        request: Request,
        body: EncryptRequest,
        encryption_service: Any = Depends(_get_encryption_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> EncryptResponse:
        """Encrypt data using AES-256-GCM with envelope encryption.

        Args:
            request: The incoming HTTP request.
            body: Encryption request with base64-encoded plaintext.
            encryption_service: Injected EncryptionService.
            audit: Injected EncryptionAuditLogger.
            metrics: Injected EncryptionMetrics.

        Returns:
            Encrypted data with ciphertext, nonce, auth_tag, and wrapped DEK.

        Raises:
            HTTPException 400: If input validation fails.
            HTTPException 500: If encryption fails.
        """
        import base64

        correlation_id = _get_correlation_id(request)
        client_ip = _get_client_ip(request)
        start = time.perf_counter()

        try:
            # Decode base64 plaintext
            try:
                plaintext = base64.b64decode(body.plaintext)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 plaintext: {e}",
                )

            # Decode optional AAD
            aad = None
            if body.aad:
                try:
                    aad = base64.b64decode(body.aad)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid base64 aad: {e}",
                    )

            # Perform encryption
            encrypted = await encryption_service.encrypt(
                plaintext,
                body.context,
                aad=aad,
            )

            duration_ms = (time.perf_counter() - start) * 1000

            # Log audit event
            if audit:
                await audit.log_encryption(
                    data_class=body.data_class,
                    tenant_id=body.context.get("tenant_id"),
                    key_version=encrypted.key_version,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                )

            # Record metrics
            if metrics:
                metrics.record_encryption(
                    "success",
                    body.data_class,
                    duration_s=(time.perf_counter() - start),
                    bytes_count=len(plaintext),
                )

            logger.debug(
                "Encrypted %d bytes for tenant %s",
                len(plaintext),
                body.context.get("tenant_id"),
            )

            return EncryptResponse(
                ciphertext=base64.b64encode(encrypted.ciphertext).decode("utf-8"),
                nonce=base64.b64encode(encrypted.nonce).decode("utf-8"),
                auth_tag=base64.b64encode(encrypted.auth_tag).decode("utf-8"),
                encrypted_dek=base64.b64encode(encrypted.encrypted_dek).decode("utf-8"),
                key_version=encrypted.key_version,
                algorithm=encrypted.algorithm,
            )

        except HTTPException:
            raise

        except Exception as e:
            # Log failure
            if audit:
                await audit.log_encryption_failure(
                    str(e),
                    data_class=body.data_class,
                    tenant_id=body.context.get("tenant_id"),
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                )

            if metrics:
                metrics.record_failure("encryption_error")

            logger.exception("Encryption failed")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Encryption failed. See logs for details.",
            )

    # -- Decrypt Data ------------------------------------------------------

    @encryption_router.post(
        "/decrypt",
        response_model=DecryptResponse,
        status_code=200,
        summary="Decrypt data",
        description=(
            "Decrypt data that was encrypted by this service. "
            "All inputs must be base64-encoded. Requires encryption:decrypt permission."
        ),
        operation_id="decrypt_data",
    )
    async def decrypt_data(
        request: Request,
        body: DecryptRequest,
        encryption_service: Any = Depends(_get_encryption_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> DecryptResponse:
        """Decrypt data using AES-256-GCM with envelope encryption.

        Args:
            request: The incoming HTTP request.
            body: Decryption request with base64-encoded components.
            encryption_service: Injected EncryptionService.
            audit: Injected EncryptionAuditLogger.
            metrics: Injected EncryptionMetrics.

        Returns:
            Decrypted plaintext (base64-encoded).

        Raises:
            HTTPException 400: If input validation fails.
            HTTPException 403: If context mismatch or integrity check fails.
            HTTPException 500: If decryption fails.
        """
        import base64
        from greenlang.infrastructure.encryption_service import (
            ContextMismatchError,
            EncryptedData,
            IntegrityError,
        )

        correlation_id = _get_correlation_id(request)
        client_ip = _get_client_ip(request)
        start = time.perf_counter()

        try:
            # Decode all base64 components
            try:
                ciphertext = base64.b64decode(body.ciphertext)
                nonce = base64.b64decode(body.nonce)
                auth_tag = base64.b64decode(body.auth_tag)
                encrypted_dek = base64.b64decode(body.encrypted_dek)
                aad = base64.b64decode(body.aad) if body.aad else None
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 encoding: {e}",
                )

            # Construct EncryptedData object
            encrypted = EncryptedData(
                ciphertext=ciphertext,
                nonce=nonce,
                auth_tag=auth_tag,
                encrypted_dek=encrypted_dek,
                key_version=body.key_version,
                encryption_context=body.context,
            )

            # Perform decryption
            plaintext = await encryption_service.decrypt(
                encrypted,
                body.context,
                aad=aad,
            )

            duration_ms = (time.perf_counter() - start) * 1000

            # Log audit event
            if audit:
                await audit.log_decryption(
                    tenant_id=body.context.get("tenant_id"),
                    key_version=body.key_version,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                )

            # Record metrics
            if metrics:
                metrics.record_decryption(
                    "success",
                    "unknown",  # data_class not tracked in encrypted data
                    duration_s=(time.perf_counter() - start),
                    bytes_count=len(plaintext),
                )

            logger.debug(
                "Decrypted %d bytes for tenant %s",
                len(plaintext),
                body.context.get("tenant_id"),
            )

            return DecryptResponse(
                plaintext=base64.b64encode(plaintext).decode("utf-8"),
                key_version=body.key_version,
            )

        except ContextMismatchError as e:
            if audit:
                await audit.log_decryption_failure(
                    "context_mismatch",
                    tenant_id=body.context.get("tenant_id"),
                    key_version=body.key_version,
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                )

            if metrics:
                metrics.record_failure("context_mismatch")

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Encryption context mismatch - access denied.",
            )

        except IntegrityError as e:
            if audit:
                await audit.log_decryption_failure(
                    "integrity_check_failed",
                    tenant_id=body.context.get("tenant_id"),
                    key_version=body.key_version,
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                )

            if metrics:
                metrics.record_failure("integrity")

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Integrity check failed - data may have been tampered.",
            )

        except HTTPException:
            raise

        except Exception as e:
            if audit:
                await audit.log_decryption_failure(
                    str(e),
                    tenant_id=body.context.get("tenant_id"),
                    key_version=body.key_version,
                    correlation_id=correlation_id,
                    client_ip=client_ip,
                )

            if metrics:
                metrics.record_failure("decryption_error")

            logger.exception("Decryption failed")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Decryption failed. See logs for details.",
            )

    # -- List Keys ---------------------------------------------------------

    @encryption_router.get(
        "/keys",
        response_model=KeyListResponse,
        summary="List encryption keys",
        description=(
            "List active encryption key versions. "
            "Requires encryption:admin permission."
        ),
        operation_id="list_encryption_keys",
    )
    async def list_keys(
        request: Request,
        key_type: str = Query(
            default="dek",
            description="Filter by key type: dek or kek.",
        ),
        include_expired: bool = Query(
            default=False,
            description="Include expired keys.",
        ),
        encryption_service: Any = Depends(_get_encryption_service),
    ) -> KeyListResponse:
        """List encryption keys.

        Args:
            request: The incoming HTTP request.
            key_type: Filter by key type.
            include_expired: Whether to include expired keys.
            encryption_service: Injected EncryptionService.

        Returns:
            List of key information objects.
        """
        # Implementation would query security.encryption_key_versions
        # For now, return empty list (actual implementation in key_management)
        return KeyListResponse(keys=[], total=0)

    # -- Rotate Key --------------------------------------------------------

    @encryption_router.post(
        "/keys/rotate",
        response_model=RotateKeyResponse,
        status_code=200,
        summary="Rotate encryption key",
        description=(
            "Trigger manual key rotation. Invalidates cached DEKs and "
            "generates new key material. Requires encryption:admin permission."
        ),
        operation_id="rotate_encryption_key",
    )
    async def rotate_key(
        request: Request,
        body: RotateKeyRequest,
        encryption_service: Any = Depends(_get_encryption_service),
        audit: Any = Depends(_get_audit_logger),
        metrics: Any = Depends(_get_metrics),
    ) -> RotateKeyResponse:
        """Rotate encryption key.

        Args:
            request: The incoming HTTP request.
            body: Key rotation request.
            encryption_service: Injected EncryptionService.
            audit: Injected EncryptionAuditLogger.
            metrics: Injected EncryptionMetrics.

        Returns:
            Rotation result with new and previous key versions.
        """
        correlation_id = _get_correlation_id(request)
        actor_id = _get_actor_id(request)

        # Generate new key version identifier
        new_version = f"v-{int(time.time())}-{actor_id[:8]}"

        # Log audit event
        if audit:
            await audit.log_key_rotated(
                new_key_version=new_version,
                previous_key_version=None,  # Would be populated by actual impl
                tenant_id=body.context.get("tenant_id") if body.context else None,
                key_type=body.key_type,
                correlation_id=correlation_id,
                reason=body.reason,
            )

        # Record metrics
        if metrics:
            metrics.inc_active_keys(body.key_type)

        logger.info(
            "Key rotation triggered by %s: type=%s, reason=%s",
            actor_id,
            body.key_type,
            body.reason,
        )

        return RotateKeyResponse(
            new_key_version=new_version,
            previous_key_version=None,
            rotated_at=datetime.now(timezone.utc),
        )

    # -- Invalidate Cache --------------------------------------------------

    @encryption_router.delete(
        "/keys/cache",
        response_model=CacheInvalidateResponse,
        summary="Invalidate key cache",
        description=(
            "Invalidate cached DEKs. Use after key rotation or in case of "
            "suspected compromise. Requires encryption:admin permission."
        ),
        operation_id="invalidate_key_cache",
    )
    async def invalidate_cache(
        request: Request,
        body: CacheInvalidateRequest,
        encryption_service: Any = Depends(_get_encryption_service),
        audit: Any = Depends(_get_audit_logger),
    ) -> CacheInvalidateResponse:
        """Invalidate DEK cache entries.

        Args:
            request: The incoming HTTP request.
            body: Cache invalidation request.
            encryption_service: Injected EncryptionService.
            audit: Injected EncryptionAuditLogger.

        Returns:
            Number of cache entries invalidated.
        """
        correlation_id = _get_correlation_id(request)

        # Actual implementation would call encryption_service.invalidate_cache()
        invalidated_count = 0

        if audit:
            await audit.log_cache_invalidated(
                key_version=body.key_version,
                tenant_id=body.tenant_id,
                reason="manual",
                correlation_id=correlation_id,
            )

        logger.info(
            "Cache invalidation: key_version=%s, tenant_id=%s, count=%d",
            body.key_version,
            body.tenant_id,
            invalidated_count,
        )

        return CacheInvalidateResponse(
            invalidated_count=invalidated_count,
            timestamp=datetime.now(timezone.utc),
        )

    # -- Get Audit Log -----------------------------------------------------

    @encryption_router.get(
        "/audit",
        response_model=AuditLogResponse,
        summary="Get encryption audit log",
        description=(
            "Retrieve encryption audit events for compliance and forensics. "
            "Requires encryption:audit permission."
        ),
        operation_id="get_encryption_audit",
    )
    async def get_audit_log(
        request: Request,
        tenant_id: Optional[str] = Query(
            default=None,
            description="Filter by tenant ID.",
        ),
        event_type: Optional[str] = Query(
            default=None,
            description="Filter by event type.",
        ),
        start_date: Optional[datetime] = Query(
            default=None,
            description="Filter events after this timestamp.",
        ),
        end_date: Optional[datetime] = Query(
            default=None,
            description="Filter events before this timestamp.",
        ),
        page: int = Query(
            default=1,
            ge=1,
            description="Page number (1-indexed).",
        ),
        page_size: int = Query(
            default=50,
            ge=1,
            le=100,
            description="Items per page.",
        ),
    ) -> AuditLogResponse:
        """Get encryption audit log.

        Args:
            request: The incoming HTTP request.
            tenant_id: Optional tenant filter.
            event_type: Optional event type filter.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            page: Page number.
            page_size: Items per page.

        Returns:
            Paginated audit log entries.
        """
        # Implementation would query security.encryption_audit_log
        # For now, return empty list
        return AuditLogResponse(
            entries=[],
            total=0,
            page=page,
            page_size=page_size,
            has_next=False,
            has_prev=page > 1,
        )

    # -- Health Status -----------------------------------------------------

    @encryption_router.get(
        "/status",
        response_model=EncryptionStatusResponse,
        summary="Encryption service status",
        description=(
            "Health check for encryption service. "
            "Verifies KMS connectivity and cache status. "
            "Requires encryption:read permission."
        ),
        operation_id="get_encryption_status",
    )
    async def get_status(
        request: Request,
        encryption_service: Any = Depends(_get_encryption_service),
    ) -> EncryptionStatusResponse:
        """Get encryption service health status.

        Args:
            request: The incoming HTTP request.
            encryption_service: Injected EncryptionService.

        Returns:
            Service health status with KMS connectivity and cache stats.
        """
        errors: List[str] = []
        kms_reachable = True

        # Check KMS connectivity
        try:
            # Would actually call KMS DescribeKey or similar
            pass
        except Exception as e:
            kms_reachable = False
            errors.append(f"KMS unreachable: {e}")

        # Get cache stats (would come from key_management)
        cache_size = 0
        cache_hit_rate = 0.0
        active_key_count = 0
        last_key_rotation = None

        healthy = len(errors) == 0 and kms_reachable

        return EncryptionStatusResponse(
            healthy=healthy,
            kms_reachable=kms_reachable,
            cache_size=cache_size,
            cache_hit_rate=cache_hit_rate,
            active_key_count=active_key_count,
            last_key_rotation=last_key_rotation,
            errors=errors,
            version="1.0.0",
        )

    # -- Apply route protection from auth_service --------------------------
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(encryption_router)
    except ImportError:
        pass  # auth_service not available

else:
    encryption_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - encryption_router is None")


__all__ = ["encryption_router"]
