# -*- coding: utf-8 -*-
"""
Designation Validation Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for validating the legal designation status of protected areas,
checking for PADDD events, and retrieving designation history timelines.

Endpoints:
    POST /designation/validate          - Validate designation status
    GET  /designation/status/{area_id}  - Get current designation status
    GET  /designation/history/{area_id} - Get designation history

Auth: eudr-pav:designation:{create|read}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, DesignationValidator Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    get_designation_validator,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    DesignationHistoryEntry,
    DesignationHistoryResponse,
    DesignationStatusEnum,
    DesignationStatusResponse,
    DesignationValidateRequest,
    DesignationValidateResponse,
    DesignationValidationEntry,
    ErrorResponse,
    MetadataSchema,
    ProtectedAreaTypeEnum,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/designation", tags=["Designation Validation"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /designation/validate
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=DesignationValidateResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate designation status of protected areas",
    description=(
        "Validate the current legal designation status of protected areas "
        "affecting a plot or specified by ID. Checks for PADDD events, "
        "legal status verification, and designation currency."
    ),
    responses={
        200: {"description": "Designation validation completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def validate_designation(
    request: Request,
    body: DesignationValidateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:designation:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> DesignationValidateResponse:
    """Validate designation status of protected areas.

    Args:
        body: Validation request with area IDs or spatial parameters.
        user: Authenticated user with designation:create permission.

    Returns:
        DesignationValidateResponse with validation results.
    """
    start = time.monotonic()

    try:
        engine = get_designation_validator()
        result = engine.validate(
            area_ids=body.area_ids,
            plot_id=body.plot_id,
            latitude=float(body.plot_center.latitude) if body.plot_center else None,
            longitude=float(body.plot_center.longitude) if body.plot_center else None,
            radius_km=float(body.radius_km) if body.radius_km else None,
            check_paddd=body.check_paddd,
            check_legal_status=body.check_legal_status,
        )

        validations = []
        valid_count = 0
        invalid_count = 0
        paddd_count = 0

        for v in result.get("validations", []):
            entry = DesignationValidationEntry(
                area_id=v.get("area_id", ""),
                area_name=v.get("area_name", ""),
                designation_status=DesignationStatusEnum(v.get("designation_status", "unknown")),
                is_valid=v.get("is_valid", True),
                designation_date=v.get("designation_date"),
                last_verified=v.get("last_verified"),
                paddd_events=v.get("paddd_events", 0),
                has_active_paddd=v.get("has_active_paddd", False),
                legal_basis=v.get("legal_basis"),
                verification_notes=v.get("verification_notes"),
            )
            validations.append(entry)
            if entry.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            if entry.has_active_paddd:
                paddd_count += 1

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"validate_designation:{body.area_ids or body.plot_id}",
            str(len(validations)),
        )

        logger.info(
            "Designation validated: areas=%d valid=%d invalid=%d paddd=%d operator=%s",
            len(validations),
            valid_count,
            invalid_count,
            paddd_count,
            user.operator_id or user.user_id,
        )

        return DesignationValidateResponse(
            validations=validations,
            total_validated=len(validations),
            valid_count=valid_count,
            invalid_count=invalid_count,
            paddd_affected_count=paddd_count,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DesignationValidator", "WDPA", "PADDDtracker"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Designation validation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Designation validation failed",
        )


# ---------------------------------------------------------------------------
# GET /designation/status/{area_id}
# ---------------------------------------------------------------------------


@router.get(
    "/status/{area_id}",
    response_model=DesignationStatusResponse,
    summary="Get current designation status",
    description=(
        "Retrieve the current designation status of a protected area "
        "including governance type, management authority, IUCN category, "
        "and active PADDD event indicators."
    ),
    responses={
        200: {"description": "Designation status retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Area not found"},
    },
)
async def get_designation_status(
    area_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:designation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DesignationStatusResponse:
    """Get current designation status for a protected area.

    Args:
        area_id: Protected area identifier.
        user: Authenticated user with designation:read permission.

    Returns:
        DesignationStatusResponse with current status.
    """
    start = time.monotonic()

    try:
        engine = get_designation_validator()
        result = engine.get_status(area_id=area_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protected area not found: {area_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"designation_status:{area_id}",
            result.get("designation_status", "unknown"),
        )

        logger.info(
            "Designation status: area_id=%s status=%s valid=%s operator=%s",
            area_id,
            result.get("designation_status", "unknown"),
            result.get("is_valid", True),
            user.operator_id or user.user_id,
        )

        return DesignationStatusResponse(
            area_id=area_id,
            area_name=result.get("area_name", ""),
            designation_status=DesignationStatusEnum(result.get("designation_status", "unknown")),
            is_valid=result.get("is_valid", True),
            designation_date=result.get("designation_date"),
            area_type=ProtectedAreaTypeEnum(result.get("area_type", "other")),
            iucn_category=result.get("iucn_category"),
            governance_type=result.get("governance_type"),
            management_authority=result.get("management_authority"),
            has_active_paddd=result.get("has_active_paddd", False),
            last_verified=result.get("last_verified"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DesignationValidator", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Designation status retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Designation status retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /designation/history/{area_id}
# ---------------------------------------------------------------------------


@router.get(
    "/history/{area_id}",
    response_model=DesignationHistoryResponse,
    summary="Get designation history",
    description=(
        "Retrieve the complete designation history timeline for a protected "
        "area including original designation, amendments, PADDD events, "
        "and status changes."
    ),
    responses={
        200: {"description": "Designation history retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Area not found"},
    },
)
async def get_designation_history(
    area_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:designation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DesignationHistoryResponse:
    """Get designation history for a protected area.

    Args:
        area_id: Protected area identifier.
        user: Authenticated user with designation:read permission.

    Returns:
        DesignationHistoryResponse with history timeline.
    """
    start = time.monotonic()

    try:
        engine = get_designation_validator()
        result = engine.get_history(area_id=area_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protected area not found: {area_id}",
            )

        history = []
        for h in result.get("history", []):
            history.append(
                DesignationHistoryEntry(
                    event_date=h.get("event_date"),
                    previous_status=(
                        DesignationStatusEnum(h["previous_status"])
                        if h.get("previous_status") else None
                    ),
                    new_status=DesignationStatusEnum(h.get("new_status", "unknown")),
                    event_type=h.get("event_type", "unknown"),
                    legal_reference=h.get("legal_reference"),
                    notes=h.get("notes"),
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"designation_history:{area_id}", str(len(history)),
        )

        logger.info(
            "Designation history: area_id=%s events=%d operator=%s",
            area_id,
            len(history),
            user.operator_id or user.user_id,
        )

        return DesignationHistoryResponse(
            area_id=area_id,
            area_name=result.get("area_name", ""),
            current_status=DesignationStatusEnum(result.get("current_status", "unknown")),
            history=history,
            total_events=len(history),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DesignationValidator", "WDPA", "PADDDtracker"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Designation history retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Designation history retrieval failed",
        )
