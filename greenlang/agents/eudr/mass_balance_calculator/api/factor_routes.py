# -*- coding: utf-8 -*-
"""
Factor Routes - AGENT-EUDR-011 Mass Balance Calculator API

Endpoints for conversion factor validation, reference data lookup,
custom factor registration, and factor usage history. Conversion
factors represent the yield ratio (output_mass / input_mass) for
each processing step, validated against EUDR reference data with
configurable warn (5%) and reject (15%) deviation thresholds.

Endpoints:
    POST  /factors/validate              - Validate a conversion factor
    GET   /factors/reference/{commodity} - Get reference factors for commodity
    POST  /factors/custom                - Register custom factor with approval
    GET   /factors/history/{facility_id} - Get factor usage history

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011, Feature 3 (Conversion Factor Validation)
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.mass_balance_calculator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_mbc_service,
    get_pagination,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity_path,
    validate_facility_id,
)
from greenlang.agents.eudr.mass_balance_calculator.api.schemas import (
    ConversionStatusSchema,
    FactorHistoryEntrySchema,
    FactorHistorySchema,
    FactorRegistrationResultSchema,
    FactorValidationResultSchema,
    ProvenanceInfo,
    ReferenceFactorDetailSchema,
    ReferenceFactorsSchema,
    RegisterCustomFactorSchema,
    ValidateFactorSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversion Factors"])

# ---------------------------------------------------------------------------
# Reference conversion factors (industry standard yield ratios)
# ---------------------------------------------------------------------------

_REFERENCE_FACTORS: Dict[str, Dict[str, float]] = {
    "cocoa": {
        "fermentation": 0.92, "drying": 0.88, "roasting": 0.85,
        "winnowing": 0.80, "grinding": 0.98, "pressing": 0.45,
        "conching": 0.97, "tempering": 0.99,
    },
    "coffee": {
        "wet_processing": 0.60, "dry_processing": 0.50,
        "hulling": 0.80, "polishing": 0.98, "roasting": 0.82,
    },
    "oil_palm": {
        "sterilization": 0.95, "threshing": 0.65,
        "digestion": 0.90, "extraction": 0.22,
        "clarification": 0.95, "refining": 0.92, "fractionation": 0.90,
    },
    "wood": {
        "debarking": 0.90, "sawing": 0.55, "planing": 0.90,
        "kiln_drying": 0.92, "milling": 0.85,
    },
    "rubber": {
        "coagulation": 0.60, "sheeting": 0.95,
        "smoking": 0.88, "crumbling": 0.92,
    },
    "soya": {
        "cleaning": 0.98, "dehulling": 0.92, "flaking": 0.97,
        "solvent_extraction": 0.82, "refining": 0.92,
    },
    "cattle": {
        "slaughtering": 0.55, "deboning": 0.70, "tanning": 0.30,
    },
}

# Acceptable deviation thresholds
_WARN_DEVIATION = 0.05   # 5%
_REJECT_DEVIATION = 0.15  # 15%

# In-memory stores
_custom_factor_store: Dict[str, Dict] = {}
_factor_history_store: Dict[str, List[Dict]] = {}

def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# POST /factors/validate
# ---------------------------------------------------------------------------

@router.post(
    "/factors/validate",
    response_model=FactorValidationResultSchema,
    summary="Validate a conversion factor",
    description=(
        "Validate a reported conversion factor (yield ratio) against "
        "reference data for the given commodity and process. Returns "
        "validation status (validated/warned/rejected) with deviation "
        "details. Warn threshold: 5%, Reject threshold: 15%."
    ),
    responses={
        200: {"description": "Factor validated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def validate_factor(
    request: Request,
    body: ValidateFactorSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:factors:validate")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FactorValidationResultSchema:
    """Validate a conversion factor against reference data.

    Args:
        body: Validation request with commodity, process_name, and yield_ratio.
        user: Authenticated user with factors:validate permission.

    Returns:
        FactorValidationResultSchema with validation result and deviation.
    """
    start = time.monotonic()
    try:
        factor_id = str(uuid.uuid4())
        now = utcnow()
        commodity_lower = body.commodity.strip().lower()

        # Look up reference data
        commodity_factors = _REFERENCE_FACTORS.get(commodity_lower, {})
        reference_ratio = commodity_factors.get(body.process_name.strip().lower())

        # Calculate deviation and determine status
        deviation_percent: Optional[float] = None
        acceptable_min: Optional[float] = None
        acceptable_max: Optional[float] = None
        validation_status = ConversionStatusSchema.VALIDATED
        message = "Conversion factor validated successfully."

        if reference_ratio is not None:
            deviation = abs(body.yield_ratio - reference_ratio) / reference_ratio
            deviation_percent = round(deviation * 100.0, 2)
            acceptable_min = round(reference_ratio * (1.0 - _REJECT_DEVIATION), 4)
            acceptable_max = min(1.0, round(reference_ratio * (1.0 + _REJECT_DEVIATION), 4))

            if deviation > _REJECT_DEVIATION:
                validation_status = ConversionStatusSchema.REJECTED
                message = (
                    f"Conversion factor rejected: {deviation_percent:.1f}% "
                    f"deviation exceeds {_REJECT_DEVIATION * 100:.0f}% threshold."
                )
            elif deviation > _WARN_DEVIATION:
                validation_status = ConversionStatusSchema.WARNED
                message = (
                    f"Conversion factor warning: {deviation_percent:.1f}% "
                    f"deviation exceeds {_WARN_DEVIATION * 100:.0f}% threshold."
                )
        else:
            validation_status = ConversionStatusSchema.PENDING
            message = (
                f"No reference factor available for {commodity_lower}/"
                f"{body.process_name}. Factor recorded as pending."
            )

        provenance_hash = _compute_provenance_hash(body.model_dump(mode="json"))
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        # Record in history
        _record_factor_usage(
            factor_id=factor_id,
            commodity=commodity_lower,
            process_name=body.process_name,
            yield_ratio=body.yield_ratio,
            validation_status=validation_status,
            facility_id=body.facility_id,
            timestamp=now,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Factor validation: commodity=%s process=%s ratio=%.4f "
            "status=%s deviation=%s%%",
            commodity_lower,
            body.process_name,
            body.yield_ratio,
            validation_status.value,
            deviation_percent,
        )

        return FactorValidationResultSchema(
            factor_id=factor_id,
            commodity=commodity_lower,
            process_name=body.process_name,
            yield_ratio=body.yield_ratio,
            reference_ratio=reference_ratio,
            deviation_percent=deviation_percent,
            acceptable_range_min=acceptable_min,
            acceptable_range_max=acceptable_max,
            validation_status=validation_status,
            message=message,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
            timestamp=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to validate factor: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate conversion factor",
        )

# ---------------------------------------------------------------------------
# GET /factors/reference/{commodity}
# ---------------------------------------------------------------------------

@router.get(
    "/factors/reference/{commodity}",
    response_model=ReferenceFactorsSchema,
    summary="Get reference factors for commodity",
    description=(
        "Retrieve all reference conversion factors (yield ratios) for "
        "a given EUDR commodity, including acceptable ranges."
    ),
    responses={
        200: {"description": "Reference factors retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_reference_factors(
    request: Request,
    commodity: str = Depends(validate_commodity_path),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:factors:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ReferenceFactorsSchema:
    """Get reference conversion factors for a commodity.

    Args:
        commodity: EUDR commodity name (e.g. cocoa, coffee, oil_palm).
        user: Authenticated user with factors:read permission.

    Returns:
        ReferenceFactorsSchema with all process-specific yield ratios.

    Raises:
        HTTPException: 404 if commodity has no reference data.
    """
    start = time.monotonic()
    try:
        commodity_factors = _REFERENCE_FACTORS.get(commodity)

        if commodity_factors is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No reference factors available for commodity: {commodity}",
            )

        factors = []
        for process_name, yield_ratio in commodity_factors.items():
            acceptable_min = round(yield_ratio * (1.0 - _REJECT_DEVIATION), 4)
            acceptable_max = min(1.0, round(yield_ratio * (1.0 + _REJECT_DEVIATION), 4))
            factors.append(ReferenceFactorDetailSchema(
                process_name=process_name,
                yield_ratio=yield_ratio,
                acceptable_range_min=acceptable_min,
                acceptable_range_max=acceptable_max,
            ))

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return ReferenceFactorsSchema(
            commodity=commodity,
            factors=factors,
            source="EUDR reference data (ISO 22095:2020)",
            processing_time_ms=elapsed_ms,
            timestamp=utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get reference factors for %s: %s",
            commodity, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve reference factors",
        )

# ---------------------------------------------------------------------------
# POST /factors/custom
# ---------------------------------------------------------------------------

@router.post(
    "/factors/custom",
    response_model=FactorRegistrationResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Register custom conversion factor",
    description=(
        "Register a custom conversion factor for a specific commodity "
        "and process. The factor is automatically validated against "
        "reference data and flagged if deviation exceeds thresholds."
    ),
    responses={
        201: {"description": "Custom factor registered"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_custom_factor(
    request: Request,
    body: RegisterCustomFactorSchema,
    user: AuthUser = Depends(
        require_permission("eudr-mbc:factors:custom:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FactorRegistrationResultSchema:
    """Register a custom conversion factor.

    Args:
        body: Custom factor parameters including commodity, process,
            yield ratio, and justification.
        user: Authenticated user with factors:custom:create permission.

    Returns:
        FactorRegistrationResultSchema with registration result.
    """
    start = time.monotonic()
    try:
        factor_id = str(uuid.uuid4())
        now = utcnow()
        commodity_lower = body.commodity.strip().lower()

        # Auto-validate against reference
        commodity_factors = _REFERENCE_FACTORS.get(commodity_lower, {})
        reference_ratio = commodity_factors.get(body.process_name.strip().lower())

        deviation_percent: Optional[float] = None
        validation_status = ConversionStatusSchema.VALIDATED
        message = "Custom factor registered and validated."

        if reference_ratio is not None:
            deviation = abs(body.yield_ratio - reference_ratio) / reference_ratio
            deviation_percent = round(deviation * 100.0, 2)

            if deviation > _REJECT_DEVIATION:
                validation_status = ConversionStatusSchema.REJECTED
                message = (
                    f"Custom factor rejected: {deviation_percent:.1f}% deviation "
                    f"exceeds {_REJECT_DEVIATION * 100:.0f}% threshold. "
                    f"Requires manual approval."
                )
            elif deviation > _WARN_DEVIATION:
                validation_status = ConversionStatusSchema.WARNED
                message = (
                    f"Custom factor registered with warning: {deviation_percent:.1f}% "
                    f"deviation exceeds {_WARN_DEVIATION * 100:.0f}% threshold."
                )
        else:
            validation_status = ConversionStatusSchema.PENDING
            message = (
                f"No reference factor for {commodity_lower}/{body.process_name}. "
                f"Custom factor registered as pending review."
            )

        provenance_hash = _compute_provenance_hash(body.model_dump(mode="json"))
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        # Store custom factor
        _custom_factor_store[factor_id] = {
            "factor_id": factor_id,
            "commodity": commodity_lower,
            "process_name": body.process_name,
            "yield_ratio": body.yield_ratio,
            "input_material": body.input_material,
            "output_material": body.output_material,
            "facility_id": body.facility_id,
            "source": body.source,
            "validation_status": validation_status,
            "deviation_percent": deviation_percent,
            "metadata": body.metadata,
            "created_by": user.user_id,
            "created_at": now,
        }

        # Record in history
        _record_factor_usage(
            factor_id=factor_id,
            commodity=commodity_lower,
            process_name=body.process_name,
            yield_ratio=body.yield_ratio,
            validation_status=validation_status,
            facility_id=body.facility_id,
            timestamp=now,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Custom factor registered: id=%s commodity=%s process=%s "
            "ratio=%.4f status=%s",
            factor_id,
            commodity_lower,
            body.process_name,
            body.yield_ratio,
            validation_status.value,
        )

        return FactorRegistrationResultSchema(
            factor_id=factor_id,
            commodity=commodity_lower,
            process_name=body.process_name,
            yield_ratio=body.yield_ratio,
            validation_status=validation_status,
            deviation_percent=deviation_percent,
            message=message,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to register custom factor: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register custom conversion factor",
        )

# ---------------------------------------------------------------------------
# GET /factors/history/{facility_id}
# ---------------------------------------------------------------------------

@router.get(
    "/factors/history/{facility_id}",
    response_model=FactorHistorySchema,
    summary="Get factor usage history",
    description=(
        "Retrieve conversion factor usage history for a facility, "
        "including validation status at time of application."
    ),
    responses={
        200: {"description": "Factor history retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_factor_history(
    request: Request,
    facility_id: str = Depends(validate_facility_id),
    commodity: Optional[str] = Query(
        None, description="Filter by commodity"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-mbc:factors:history:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FactorHistorySchema:
    """Get conversion factor usage history for a facility.

    Args:
        facility_id: Facility identifier.
        commodity: Optional commodity filter.
        pagination: Pagination parameters.
        user: Authenticated user with factors:history:read permission.

    Returns:
        FactorHistorySchema with factor usage entries.
    """
    start = time.monotonic()
    try:
        history = _factor_history_store.get(facility_id, [])

        # Filter by commodity if specified
        if commodity:
            commodity_lower = commodity.strip().lower()
            history = [h for h in history if h.get("commodity") == commodity_lower]

        # Sort by applied_at descending
        history.sort(key=lambda h: h.get("applied_at", ""), reverse=True)

        total = len(history)
        paginated = history[pagination.offset: pagination.offset + pagination.limit]

        factors = [FactorHistoryEntrySchema(**h) for h in paginated]

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return FactorHistorySchema(
            facility_id=facility_id,
            factors=factors,
            total_count=total,
            processing_time_ms=elapsed_ms,
            timestamp=utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get factor history for %s: %s",
            facility_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve factor usage history",
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_factor_usage(
    factor_id: str,
    commodity: str,
    process_name: str,
    yield_ratio: float,
    validation_status: ConversionStatusSchema,
    facility_id: Optional[str],
    timestamp: datetime,
) -> None:
    """Record a factor usage event in the history store.

    Args:
        factor_id: Factor identifier.
        commodity: Commodity name.
        process_name: Processing step.
        yield_ratio: Yield ratio.
        validation_status: Validation result.
        facility_id: Facility identifier.
        timestamp: Usage timestamp.
    """
    if facility_id is None:
        return

    if facility_id not in _factor_history_store:
        _factor_history_store[facility_id] = []

    _factor_history_store[facility_id].append({
        "factor_id": factor_id,
        "commodity": commodity,
        "process_name": process_name,
        "yield_ratio": yield_ratio,
        "validation_status": validation_status,
        "applied_at": timestamp,
        "facility_id": facility_id,
    })

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
