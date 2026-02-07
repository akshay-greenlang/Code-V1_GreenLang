# -*- coding: utf-8 -*-
"""
PII Service REST API Routes - SEC-011

FastAPI router providing REST endpoints for PII detection, redaction,
tokenization, enforcement policies, allowlist management, and quarantine
operations.

Endpoints:
    POST /api/v1/pii/detect - Detect PII in content
    POST /api/v1/pii/redact - Redact PII from content
    POST /api/v1/pii/tokenize - Create reversible token
    POST /api/v1/pii/detokenize - Retrieve original from token
    GET /api/v1/pii/policies - List enforcement policies
    PUT /api/v1/pii/policies/{pii_type} - Update policy
    GET /api/v1/pii/allowlist - List allowlist entries
    POST /api/v1/pii/allowlist - Add entry
    DELETE /api/v1/pii/allowlist/{id} - Remove entry
    GET /api/v1/pii/quarantine - List quarantined items
    POST /api/v1/pii/quarantine/{id}/release - Release item
    POST /api/v1/pii/quarantine/{id}/delete - Delete item
    GET /api/v1/pii/metrics - Get PII metrics

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# Create the router with prefix and tags
pii_router = APIRouter(
    prefix="/api/v1/pii",
    tags=["pii"],
)


# =============================================================================
# Request/Response Models
# =============================================================================


class DetectRequest(BaseModel):
    """Request model for PII detection."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "content": "My SSN is 123-45-6789 and email is test@example.com",
                "use_ml": False,
                "apply_allowlist": True,
                "min_confidence": 0.8,
                "source": "api",
            }
        },
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=10_000_000,
        description="Content to scan for PII",
    )
    use_ml: bool = Field(
        default=False,
        description="Use ML-based detection (slower but more accurate)",
    )
    apply_allowlist: bool = Field(
        default=True,
        description="Filter results through allowlist",
    )
    min_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold (0.0-1.0)",
    )
    source: str = Field(
        default="api",
        max_length=50,
        description="Source identifier for metrics",
    )


class PIIDetectionResponse(BaseModel):
    """Individual PII detection result."""

    model_config = ConfigDict(extra="forbid")

    pii_type: str = Field(..., description="Type of PII detected")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")
    value_hash: str = Field(..., description="SHA-256 hash of detected value")
    context: Optional[str] = Field(
        default=None,
        description="Surrounding context (redacted)",
    )


class DetectResponse(BaseModel):
    """Response model for PII detection."""

    model_config = ConfigDict(extra="forbid")

    detections: List[PIIDetectionResponse] = Field(
        default_factory=list,
        description="List of PII detections",
    )
    detection_count: int = Field(..., description="Total number of detections")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class RedactRequest(BaseModel):
    """Request model for PII redaction."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "content": "Contact john.doe@company.com or call 555-1234",
                "use_ml": False,
                "apply_allowlist": True,
                "strategy": "replace",
            }
        },
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=10_000_000,
        description="Content to redact PII from",
    )
    use_ml: bool = Field(
        default=False,
        description="Use ML-based detection",
    )
    apply_allowlist: bool = Field(
        default=True,
        description="Filter through allowlist",
    )
    strategy: str = Field(
        default="replace",
        pattern=r"^(replace|mask|hash|tokenize)$",
        description="Redaction strategy: replace, mask, hash, or tokenize",
    )


class RedactResponse(BaseModel):
    """Response model for PII redaction."""

    model_config = ConfigDict(extra="forbid")

    original_content: str = Field(..., description="Original content (for verification)")
    redacted_content: str = Field(..., description="Content with PII redacted")
    detections: List[PIIDetectionResponse] = Field(
        default_factory=list,
        description="List of PII detections that were redacted",
    )
    redaction_count: int = Field(..., description="Number of redactions applied")


class TokenizeRequest(BaseModel):
    """Request model for PII tokenization."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "value": "john.doe@company.com",
                "pii_type": "email",
            }
        },
    )

    value: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="PII value to tokenize",
    )
    pii_type: str = Field(
        ...,
        max_length=30,
        description="Type of PII being tokenized",
    )


class TokenizeResponse(BaseModel):
    """Response model for PII tokenization."""

    model_config = ConfigDict(extra="forbid")

    token: str = Field(..., description="Reversible token for the PII value")
    pii_type: str = Field(..., description="Type of PII tokenized")
    expires_at: datetime = Field(..., description="Token expiration timestamp")


class DetokenizeRequest(BaseModel):
    """Request model for PII detokenization."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "token": "[TOKEN:abc123def456]",
            }
        },
    )

    token: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Token to detokenize",
    )


class DetokenizeResponse(BaseModel):
    """Response model for PII detokenization."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(..., description="Original PII value")
    pii_type: str = Field(..., description="Type of PII")


class EnforcementPolicyResponse(BaseModel):
    """Enforcement policy details."""

    model_config = ConfigDict(extra="forbid")

    pii_type: str = Field(..., description="PII type this policy applies to")
    action: str = Field(..., description="Enforcement action")
    min_confidence: float = Field(..., description="Minimum confidence threshold")
    contexts: List[str] = Field(..., description="Applicable contexts")
    notify: bool = Field(..., description="Whether to send notifications")
    quarantine_ttl_hours: int = Field(..., description="Quarantine retention hours")
    custom_placeholder: Optional[str] = Field(
        default=None,
        description="Custom redaction placeholder",
    )


class EnforcementPolicyUpdateRequest(BaseModel):
    """Request model for updating an enforcement policy."""

    model_config = ConfigDict(extra="forbid")

    action: Optional[str] = Field(
        default=None,
        pattern=r"^(allow|redact|block|quarantine|transform)$",
        description="Enforcement action",
    )
    min_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    contexts: Optional[List[str]] = Field(
        default=None,
        description="Applicable contexts",
    )
    notify: Optional[bool] = Field(
        default=None,
        description="Whether to send notifications",
    )
    quarantine_ttl_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=8760,
        description="Quarantine retention hours",
    )
    custom_placeholder: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Custom redaction placeholder",
    )


class AllowlistEntryResponse(BaseModel):
    """Allowlist entry details."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(..., description="Entry unique identifier")
    pii_type: str = Field(..., description="PII type this entry applies to")
    pattern: str = Field(..., description="Pattern to match")
    pattern_type: str = Field(..., description="Pattern type (regex, exact, prefix, etc.)")
    reason: str = Field(..., description="Reason for allowlisting")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID or null for global")
    enabled: bool = Field(..., description="Whether the entry is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")


class AllowlistEntryCreateRequest(BaseModel):
    """Request model for creating an allowlist entry."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "pii_type": "email",
                "pattern": ".*@example.com$",
                "pattern_type": "regex",
                "reason": "Test domain emails are not personal",
            }
        },
    )

    pii_type: str = Field(
        ...,
        max_length=30,
        description="PII type this entry applies to",
    )
    pattern: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Pattern to match",
    )
    pattern_type: str = Field(
        default="regex",
        pattern=r"^(regex|exact|prefix|suffix|contains)$",
        description="Pattern type",
    )
    reason: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Reason for allowlisting",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Optional expiration timestamp",
    )


class QuarantineItemResponse(BaseModel):
    """Quarantine item details."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(..., description="Item unique identifier")
    pii_type: str = Field(..., description="Type of PII detected")
    source_type: str = Field(..., description="Source type (api, storage, streaming)")
    source_location: Optional[str] = Field(default=None, description="Source location/path")
    detected_at: datetime = Field(..., description="Detection timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    status: str = Field(..., description="Item status (pending, released, deleted)")
    detection_confidence: float = Field(..., description="Detection confidence")


class PIIMetricsResponse(BaseModel):
    """PII service metrics summary."""

    model_config = ConfigDict(extra="forbid")

    total_detections: int = Field(..., description="Total detections since startup")
    total_blocked_requests: int = Field(..., description="Total blocked requests")
    total_tokenizations: int = Field(..., description="Total tokenization operations")
    total_detokenizations: int = Field(..., description="Total detokenization operations")
    quarantine_items: int = Field(..., description="Current quarantine items")
    pending_remediations: int = Field(..., description="Pending remediation items")
    allowlist_entries: int = Field(..., description="Total allowlist entries")


class SuccessResponse(BaseModel):
    """Generic success response."""

    model_config = ConfigDict(extra="forbid")

    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Success message")


class ErrorResponse(BaseModel):
    """Error response model."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")


# =============================================================================
# Dependency Functions
# =============================================================================


async def get_pii_service(request: Request) -> Any:
    """Get the PII service instance from app state.

    Args:
        request: FastAPI request object.

    Returns:
        PIIService instance.

    Raises:
        HTTPException: If PII service is not available.
    """
    pii_service = getattr(request.app.state, "pii_service", None)
    if pii_service is None:
        logger.warning("PII service not available in app state")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PII service not available",
        )
    return pii_service


async def get_current_user(request: Request) -> Any:
    """Get the current authenticated user from request state.

    Args:
        request: FastAPI request object.

    Returns:
        User/AuthContext object.

    Raises:
        HTTPException: If user is not authenticated.
    """
    auth = getattr(request.state, "auth", None)
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth


# =============================================================================
# Detection Endpoints
# =============================================================================


@pii_router.post(
    "/detect",
    response_model=DetectResponse,
    responses={
        200: {"description": "PII detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="Detect PII in content",
    description="Scan content for PII using regex and optional ML-based detection.",
)
async def detect_pii(
    request_body: DetectRequest,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> DetectResponse:
    """Detect PII in the provided content.

    Args:
        request_body: Detection request with content and options.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Detection results with list of found PII.
    """
    start_time = time.monotonic()

    try:
        # Build detection options
        options = {
            "use_ml": request_body.use_ml,
            "apply_allowlist": request_body.apply_allowlist,
            "min_confidence": request_body.min_confidence,
            "source": request_body.source,
            "tenant_id": getattr(current_user, "tenant_id", None),
        }

        # Perform detection
        detections = await pii_service.detect(request_body.content, options)

        processing_time = (time.monotonic() - start_time) * 1000

        return DetectResponse(
            detections=[
                PIIDetectionResponse(
                    pii_type=d.pii_type.value if hasattr(d.pii_type, "value") else str(d.pii_type),
                    confidence=d.confidence,
                    start=d.start,
                    end=d.end,
                    value_hash=d.value_hash,
                    context=getattr(d, "context", None),
                )
                for d in detections
            ],
            detection_count=len(detections),
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("PII detection failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detection failed",
        )


@pii_router.post(
    "/redact",
    response_model=RedactResponse,
    responses={
        200: {"description": "PII redaction completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="Redact PII from content",
    description="Detect and redact PII from content using specified strategy.",
)
async def redact_pii(
    request_body: RedactRequest,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> RedactResponse:
    """Detect and redact PII from the provided content.

    Args:
        request_body: Redaction request with content and options.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Redaction result with modified content.
    """
    try:
        options = {
            "use_ml": request_body.use_ml,
            "apply_allowlist": request_body.apply_allowlist,
            "strategy": request_body.strategy,
            "tenant_id": getattr(current_user, "tenant_id", None),
        }

        result = await pii_service.redact(request_body.content, options)

        return RedactResponse(
            original_content=request_body.content,
            redacted_content=result.redacted_content,
            detections=[
                PIIDetectionResponse(
                    pii_type=d.pii_type.value if hasattr(d.pii_type, "value") else str(d.pii_type),
                    confidence=d.confidence,
                    start=d.start,
                    end=d.end,
                    value_hash=d.value_hash,
                    context=getattr(d, "context", None),
                )
                for d in result.detections
            ],
            redaction_count=len(result.detections),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("PII redaction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Redaction failed",
        )


# =============================================================================
# Tokenization Endpoints
# =============================================================================


@pii_router.post(
    "/tokenize",
    response_model=TokenizeResponse,
    responses={
        200: {"description": "Tokenization completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="Tokenize a PII value",
    description="Create a reversible token for a PII value.",
)
async def tokenize_pii(
    request_body: TokenizeRequest,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> TokenizeResponse:
    """Create a reversible token for a PII value.

    Args:
        request_body: Tokenization request.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Token and expiration details.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", "default")

        token = await pii_service.tokenize(
            request_body.value,
            request_body.pii_type,
            tenant_id,
        )

        # Default expiration is 90 days
        expires_at = datetime.now(timezone.utc) + timedelta(days=90)

        return TokenizeResponse(
            token=token,
            pii_type=request_body.pii_type,
            expires_at=expires_at,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Tokenization failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tokenization failed",
        )


@pii_router.post(
    "/detokenize",
    response_model=DetokenizeResponse,
    responses={
        200: {"description": "Detokenization completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Access denied"},
        404: {"description": "Token not found"},
        410: {"description": "Token expired"},
        503: {"description": "Service unavailable"},
    },
    summary="Detokenize a token",
    description="Retrieve the original PII value from a token.",
)
async def detokenize_pii(
    request_body: DetokenizeRequest,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> DetokenizeResponse:
    """Retrieve the original PII value from a token.

    Args:
        request_body: Detokenization request.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Original PII value and type.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", "default")
        user_id = str(getattr(current_user, "user_id", "unknown"))

        result = await pii_service.detokenize(
            request_body.token,
            tenant_id,
            user_id,
        )

        return DetokenizeResponse(
            value=result.value,
            pii_type=result.pii_type,
        )

    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Token not found",
            )
        elif "expired" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Token has expired",
            )
        elif "unauthorized" in error_msg or "denied" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this token",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
    except Exception as e:
        logger.error("Detokenization failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detokenization failed",
        )


# =============================================================================
# Policy Management Endpoints
# =============================================================================


@pii_router.get(
    "/policies",
    response_model=List[EnforcementPolicyResponse],
    responses={
        200: {"description": "List of enforcement policies"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="List enforcement policies",
    description="Get all configured PII enforcement policies.",
)
async def list_policies(
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> List[EnforcementPolicyResponse]:
    """List all enforcement policies.

    Args:
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        List of enforcement policies.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)
        policies = await pii_service.get_policies(tenant_id)

        return [
            EnforcementPolicyResponse(
                pii_type=p.pii_type.value if hasattr(p.pii_type, "value") else str(p.pii_type),
                action=p.action.value if hasattr(p.action, "value") else str(p.action),
                min_confidence=p.min_confidence,
                contexts=p.contexts,
                notify=p.notify,
                quarantine_ttl_hours=p.quarantine_ttl_hours,
                custom_placeholder=p.custom_placeholder,
            )
            for p in policies
        ]

    except Exception as e:
        logger.error("Failed to list policies: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list policies",
        )


@pii_router.put(
    "/policies/{pii_type}",
    response_model=EnforcementPolicyResponse,
    responses={
        200: {"description": "Policy updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"description": "Unauthorized"},
        404: {"description": "Policy not found"},
        503: {"description": "Service unavailable"},
    },
    summary="Update enforcement policy",
    description="Update an enforcement policy for a specific PII type.",
)
async def update_policy(
    pii_type: str,
    request_body: EnforcementPolicyUpdateRequest,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> EnforcementPolicyResponse:
    """Update an enforcement policy.

    Args:
        pii_type: The PII type to update policy for.
        request_body: Policy update request.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Updated enforcement policy.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)

        # Build update dict from non-None fields
        updates = {
            k: v
            for k, v in request_body.model_dump().items()
            if v is not None
        }

        policy = await pii_service.update_policy(pii_type, updates, tenant_id)

        return EnforcementPolicyResponse(
            pii_type=policy.pii_type.value if hasattr(policy.pii_type, "value") else str(policy.pii_type),
            action=policy.action.value if hasattr(policy.action, "value") else str(policy.action),
            min_confidence=policy.min_confidence,
            contexts=policy.contexts,
            notify=policy.notify,
            quarantine_ttl_hours=policy.quarantine_ttl_hours,
            custom_placeholder=policy.custom_placeholder,
        )

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Policy not found for PII type: {pii_type}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to update policy: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update policy",
        )


# =============================================================================
# Allowlist Management Endpoints
# =============================================================================


@pii_router.get(
    "/allowlist",
    response_model=List[AllowlistEntryResponse],
    responses={
        200: {"description": "List of allowlist entries"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="List allowlist entries",
    description="Get allowlist entries, optionally filtered by PII type.",
)
async def list_allowlist(
    request: Request,
    pii_type: Optional[str] = Query(
        default=None,
        description="Filter by PII type",
    ),
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> List[AllowlistEntryResponse]:
    """List allowlist entries.

    Args:
        request: FastAPI request object.
        pii_type: Optional PII type filter.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        List of allowlist entries.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)
        entries = await pii_service.list_allowlist(pii_type, tenant_id)

        return [
            AllowlistEntryResponse(
                id=e.id,
                pii_type=e.pii_type.value if hasattr(e.pii_type, "value") else str(e.pii_type),
                pattern=e.pattern,
                pattern_type=e.pattern_type,
                reason=e.reason,
                tenant_id=e.tenant_id,
                enabled=e.enabled,
                created_at=e.created_at,
                expires_at=e.expires_at,
            )
            for e in entries
        ]

    except Exception as e:
        logger.error("Failed to list allowlist: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list allowlist",
        )


@pii_router.post(
    "/allowlist",
    response_model=AllowlistEntryResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Allowlist entry created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="Add allowlist entry",
    description="Add a new pattern to the PII detection allowlist.",
)
async def add_allowlist_entry(
    request_body: AllowlistEntryCreateRequest,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> AllowlistEntryResponse:
    """Add a new allowlist entry.

    Args:
        request_body: Allowlist entry creation request.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Created allowlist entry.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)
        user_id = getattr(current_user, "user_id", None)

        entry = await pii_service.add_allowlist_entry(
            pii_type=request_body.pii_type,
            pattern=request_body.pattern,
            pattern_type=request_body.pattern_type,
            reason=request_body.reason,
            tenant_id=tenant_id,
            created_by=user_id,
            expires_at=request_body.expires_at,
        )

        return AllowlistEntryResponse(
            id=entry.id,
            pii_type=entry.pii_type.value if hasattr(entry.pii_type, "value") else str(entry.pii_type),
            pattern=entry.pattern,
            pattern_type=entry.pattern_type,
            reason=entry.reason,
            tenant_id=entry.tenant_id,
            enabled=entry.enabled,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to add allowlist entry: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add allowlist entry",
        )


@pii_router.delete(
    "/allowlist/{entry_id}",
    response_model=SuccessResponse,
    responses={
        200: {"description": "Allowlist entry removed"},
        401: {"description": "Unauthorized"},
        404: {"description": "Entry not found"},
        503: {"description": "Service unavailable"},
    },
    summary="Remove allowlist entry",
    description="Remove an allowlist entry by ID.",
)
async def remove_allowlist_entry(
    entry_id: UUID,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> SuccessResponse:
    """Remove an allowlist entry.

    Args:
        entry_id: The entry ID to remove.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Success confirmation.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)

        await pii_service.remove_allowlist_entry(entry_id, tenant_id)

        return SuccessResponse(
            success=True,
            message=f"Allowlist entry {entry_id} removed",
        )

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Allowlist entry not found: {entry_id}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to remove allowlist entry: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove allowlist entry",
        )


# =============================================================================
# Quarantine Management Endpoints
# =============================================================================


@pii_router.get(
    "/quarantine",
    response_model=List[QuarantineItemResponse],
    responses={
        200: {"description": "List of quarantine items"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="List quarantine items",
    description="Get items currently in quarantine.",
)
async def list_quarantine(
    request: Request,
    status_filter: Optional[str] = Query(
        default="pending",
        alias="status",
        pattern=r"^(pending|released|deleted|all)$",
        description="Filter by status",
    ),
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> List[QuarantineItemResponse]:
    """List quarantine items.

    Args:
        request: FastAPI request object.
        status_filter: Status filter.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        List of quarantine items.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)

        items = await pii_service.list_quarantine(
            status=status_filter if status_filter != "all" else None,
            tenant_id=tenant_id,
        )

        return [
            QuarantineItemResponse(
                id=item.id,
                pii_type=item.pii_type.value if hasattr(item.pii_type, "value") else str(item.pii_type),
                source_type=item.source_type,
                source_location=item.source_location,
                detected_at=item.detected_at,
                expires_at=item.expires_at,
                status=item.status,
                detection_confidence=item.detection_confidence,
            )
            for item in items
        ]

    except Exception as e:
        logger.error("Failed to list quarantine: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list quarantine",
        )


@pii_router.post(
    "/quarantine/{item_id}/release",
    response_model=SuccessResponse,
    responses={
        200: {"description": "Item released from quarantine"},
        401: {"description": "Unauthorized"},
        404: {"description": "Item not found"},
        503: {"description": "Service unavailable"},
    },
    summary="Release from quarantine",
    description="Release an item from quarantine after review.",
)
async def release_from_quarantine(
    item_id: UUID,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> SuccessResponse:
    """Release an item from quarantine.

    Args:
        item_id: The item ID to release.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Success confirmation.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)
        user_id = getattr(current_user, "user_id", None)

        await pii_service.release_from_quarantine(
            item_id,
            tenant_id=tenant_id,
            reviewed_by=user_id,
        )

        return SuccessResponse(
            success=True,
            message=f"Quarantine item {item_id} released",
        )

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Quarantine item not found: {item_id}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to release from quarantine: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to release from quarantine",
        )


@pii_router.post(
    "/quarantine/{item_id}/delete",
    response_model=SuccessResponse,
    responses={
        200: {"description": "Item deleted from quarantine"},
        401: {"description": "Unauthorized"},
        404: {"description": "Item not found"},
        503: {"description": "Service unavailable"},
    },
    summary="Delete from quarantine",
    description="Permanently delete a quarantined item.",
)
async def delete_from_quarantine(
    item_id: UUID,
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> SuccessResponse:
    """Delete an item from quarantine.

    Args:
        item_id: The item ID to delete.
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Success confirmation.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)
        user_id = getattr(current_user, "user_id", None)

        await pii_service.delete_from_quarantine(
            item_id,
            tenant_id=tenant_id,
            deleted_by=user_id,
        )

        return SuccessResponse(
            success=True,
            message=f"Quarantine item {item_id} deleted",
        )

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Quarantine item not found: {item_id}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error("Failed to delete from quarantine: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete from quarantine",
        )


# =============================================================================
# Metrics Endpoint
# =============================================================================


@pii_router.get(
    "/metrics",
    response_model=PIIMetricsResponse,
    responses={
        200: {"description": "PII service metrics"},
        401: {"description": "Unauthorized"},
        503: {"description": "Service unavailable"},
    },
    summary="Get PII metrics",
    description="Get current PII service metrics summary.",
)
async def get_pii_metrics(
    request: Request,
    pii_service: Any = Depends(get_pii_service),
    current_user: Any = Depends(get_current_user),
) -> PIIMetricsResponse:
    """Get PII service metrics.

    Args:
        request: FastAPI request object.
        pii_service: Injected PII service.
        current_user: Authenticated user.

    Returns:
        Metrics summary.
    """
    try:
        tenant_id = getattr(current_user, "tenant_id", None)

        metrics = await pii_service.get_metrics(tenant_id)

        return PIIMetricsResponse(
            total_detections=metrics.get("total_detections", 0),
            total_blocked_requests=metrics.get("total_blocked_requests", 0),
            total_tokenizations=metrics.get("total_tokenizations", 0),
            total_detokenizations=metrics.get("total_detokenizations", 0),
            quarantine_items=metrics.get("quarantine_items", 0),
            pending_remediations=metrics.get("pending_remediations", 0),
            allowlist_entries=metrics.get("allowlist_entries", 0),
        )

    except Exception as e:
        logger.error("Failed to get metrics: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics",
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "pii_router",
    # Request models
    "DetectRequest",
    "RedactRequest",
    "TokenizeRequest",
    "DetokenizeRequest",
    "EnforcementPolicyUpdateRequest",
    "AllowlistEntryCreateRequest",
    # Response models
    "DetectResponse",
    "RedactResponse",
    "TokenizeResponse",
    "DetokenizeResponse",
    "PIIDetectionResponse",
    "EnforcementPolicyResponse",
    "AllowlistEntryResponse",
    "QuarantineItemResponse",
    "PIIMetricsResponse",
    "SuccessResponse",
    "ErrorResponse",
]
