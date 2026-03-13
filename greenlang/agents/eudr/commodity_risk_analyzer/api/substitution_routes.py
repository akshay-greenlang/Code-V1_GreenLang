# -*- coding: utf-8 -*-
"""
Substitution Routes - AGENT-EUDR-018 Commodity Risk Analyzer API

Endpoints for commodity substitution risk detection including substitution
detection, supplier switching history, active alerts, declaration
verification, and pattern analysis.

Endpoints:
    POST /substitution/detect                 - Detect substitution
    GET  /substitution/{supplier_id}/history  - Switching history
    GET  /substitution/alerts                 - Active alerts
    POST /substitution/verify                 - Verify declaration
    GET  /substitution/patterns               - Pattern analysis

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018, Substitution Risk Analyzer Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.commodity_risk_analyzer.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_pagination,
    get_substitution_risk_analyzer,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity_type,
    validate_date_range,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.schemas import (
    CommodityTypeEnum,
    DetectedSwitch,
    PaginatedMeta,
    ProvenanceInfo,
    RiskLevelEnum,
    SeveritySummaryEnum,
    SubstitutionAlertEntry,
    SubstitutionAlertResponse,
    SubstitutionDetectRequest,
    SubstitutionHistoryEntry,
    SubstitutionHistoryResponse,
    SubstitutionPatternEntry,
    SubstitutionPatternResponse,
    SubstitutionResponse,
    SubstitutionVerifyRequest,
    SubstitutionVerifyResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Substitution Risk"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_substitution_store: Dict[str, SubstitutionResponse] = {}
_alert_store: List[SubstitutionAlertEntry] = []


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /substitution/detect
# ---------------------------------------------------------------------------


@router.post(
    "/substitution/detect",
    response_model=SubstitutionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect commodity substitution",
    description=(
        "Analyze a supplier's commodity declaration history to detect potential "
        "commodity substitution events. Compares the current declaration against "
        "historical patterns and flags anomalies with confidence scores."
    ),
    responses={
        200: {"description": "Substitution analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def detect_substitution(
    request: Request,
    body: SubstitutionDetectRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:substitution:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SubstitutionResponse:
    """Detect commodity substitution for a supplier.

    Args:
        body: Detection request with supplier history and current declaration.
        user: Authenticated user with substitution:write permission.

    Returns:
        SubstitutionResponse with detected switches and risk score.
    """
    start = time.monotonic()
    try:
        detected_switches: List[DetectedSwitch] = []
        alerts: List[str] = []

        # Analyze origin country switches
        current = body.current_declaration
        for hist in body.commodity_history:
            # Check for origin country switch
            if hist.origin_country != current.origin_country:
                confidence = Decimal("0.75")
                switch = DetectedSwitch(
                    from_commodity=hist.commodity_type,
                    to_commodity=current.commodity_type,
                    from_country=hist.origin_country,
                    to_country=current.origin_country,
                    detection_confidence=confidence,
                    risk_impact=RiskLevelEnum.HIGH,
                )
                detected_switches.append(switch)
                alerts.append(
                    f"Origin country switch detected: {hist.origin_country} -> "
                    f"{current.origin_country} for {current.commodity_type.value}"
                )

            # Check for commodity type switch
            if hist.commodity_type != current.commodity_type:
                confidence = Decimal("0.90")
                switch = DetectedSwitch(
                    from_commodity=hist.commodity_type,
                    to_commodity=current.commodity_type,
                    from_country=hist.origin_country,
                    to_country=current.origin_country,
                    detection_confidence=confidence,
                    risk_impact=RiskLevelEnum.CRITICAL,
                )
                detected_switches.append(switch)
                alerts.append(
                    f"Commodity type switch detected: {hist.commodity_type.value} -> "
                    f"{current.commodity_type.value}"
                )

        # Deduplicate switches by (from_commodity, to_commodity, from_country, to_country)
        seen = set()
        unique_switches: List[DetectedSwitch] = []
        for sw in detected_switches:
            key = (
                sw.from_commodity.value,
                sw.to_commodity.value,
                sw.from_country,
                sw.to_country,
            )
            if key not in seen:
                seen.add(key)
                unique_switches.append(sw)

        # Calculate risk score
        risk_score = Decimal("0.0")
        if unique_switches:
            max_confidence = max(s.detection_confidence for s in unique_switches)
            risk_score = min(Decimal("100.0"), max_confidence * Decimal("100.0"))

        risk_level = RiskLevelEnum.LOW
        if risk_score >= Decimal("75.0"):
            risk_level = RiskLevelEnum.CRITICAL
        elif risk_score >= Decimal("50.0"):
            risk_level = RiskLevelEnum.HIGH
        elif risk_score >= Decimal("25.0"):
            risk_level = RiskLevelEnum.MEDIUM

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            body.model_dump_json(), f"{body.supplier_id}:{risk_score}"
        )

        response = SubstitutionResponse(
            supplier_id=body.supplier_id,
            risk_score=risk_score.quantize(Decimal("0.01")),
            risk_level=risk_level,
            detected_switches=unique_switches,
            alerts=list(set(alerts)),
            total_declarations_analyzed=len(body.commodity_history) + 1,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

        _substitution_store[body.supplier_id] = response

        # Create alerts for high-risk detections
        for alert_msg in alerts:
            _alert_store.append(
                SubstitutionAlertEntry(
                    supplier_id=body.supplier_id,
                    commodity_type=current.commodity_type,
                    severity=(
                        SeveritySummaryEnum.CRITICAL
                        if risk_score >= Decimal("75.0")
                        else SeveritySummaryEnum.HIGH
                    ),
                    message=alert_msg,
                )
            )

        logger.info(
            "Substitution detection completed: supplier=%s switches=%d risk=%s",
            body.supplier_id,
            len(unique_switches),
            risk_score,
        )

        return response

    except Exception as exc:
        logger.error("Substitution detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Substitution detection failed",
        )


# ---------------------------------------------------------------------------
# GET /substitution/{supplier_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/substitution/{supplier_id}/history",
    response_model=SubstitutionHistoryResponse,
    summary="Get substitution switching history",
    description="Retrieve the historical commodity switching events for a supplier.",
    responses={
        200: {"description": "Switching history"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_switching_history(
    supplier_id: str,
    request: Request,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:substitution:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SubstitutionHistoryResponse:
    """Get substitution switching history for a supplier.

    Args:
        supplier_id: Supplier identifier.
        pagination: Pagination parameters.
        user: Authenticated user with substitution:read permission.

    Returns:
        SubstitutionHistoryResponse with historical events.
    """
    result = _substitution_store.get(supplier_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No substitution data for supplier: {supplier_id}",
        )

    # Build history from detected switches
    history: List[SubstitutionHistoryEntry] = []
    for switch in result.detected_switches:
        history.append(
            SubstitutionHistoryEntry(
                event_date=switch.detected_at.date(),
                from_commodity=switch.from_commodity,
                to_commodity=switch.to_commodity,
                risk_score=result.risk_score,
                resolution=None,
            )
        )

    total = len(history)
    page = history[pagination.offset: pagination.offset + pagination.limit]

    return SubstitutionHistoryResponse(
        supplier_id=supplier_id,
        history=page,
        total_switches=total,
        meta=PaginatedMeta(
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=(pagination.offset + pagination.limit) < total,
        ),
    )


# ---------------------------------------------------------------------------
# GET /substitution/alerts
# ---------------------------------------------------------------------------


@router.get(
    "/substitution/alerts",
    response_model=SubstitutionAlertResponse,
    summary="Get active substitution alerts",
    description="Retrieve all active substitution risk alerts with severity summary.",
    responses={
        200: {"description": "Active alerts"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_alerts(
    request: Request,
    severity: Optional[str] = Query(
        None,
        description="Filter by severity (low, medium, high, critical)",
    ),
    acknowledged: Optional[bool] = Query(
        None,
        description="Filter by acknowledgement status",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:substitution:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SubstitutionAlertResponse:
    """Get active substitution alerts.

    Args:
        severity: Optional severity filter.
        acknowledged: Optional acknowledgement filter.
        user: Authenticated user with substitution:read permission.

    Returns:
        SubstitutionAlertResponse with filtered alerts.
    """
    alerts = _alert_store.copy()

    if severity:
        try:
            sev_enum = SeveritySummaryEnum(severity.lower())
            alerts = [a for a in alerts if a.severity == sev_enum]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity: {severity}. Must be low, medium, high, or critical.",
            )

    if acknowledged is not None:
        alerts = [a for a in alerts if a.acknowledged == acknowledged]

    severity_summary: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for alert in alerts:
        severity_summary[alert.severity.value] += 1

    return SubstitutionAlertResponse(
        alerts=alerts,
        total_count=len(alerts),
        severity_summary=severity_summary,
    )


# ---------------------------------------------------------------------------
# POST /substitution/verify
# ---------------------------------------------------------------------------


@router.post(
    "/substitution/verify",
    response_model=SubstitutionVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify commodity declaration",
    description=(
        "Verify a commodity declaration against supporting evidence documents. "
        "Returns verification result with confidence score and identified discrepancies."
    ),
    responses={
        200: {"description": "Verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def verify_declaration(
    request: Request,
    body: SubstitutionVerifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:substitution:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SubstitutionVerifyResponse:
    """Verify a commodity declaration against evidence.

    Args:
        body: Verification request with declaration and evidence IDs.
        user: Authenticated user with substitution:write permission.

    Returns:
        SubstitutionVerifyResponse with verification result.
    """
    # Simplified verification logic (production delegates to engine)
    evidence_count = len(body.supporting_evidence)
    confidence = min(
        Decimal("1.0"),
        Decimal(str(evidence_count)) * Decimal("0.25"),
    )
    verified = confidence >= Decimal("0.75")

    discrepancies: List[str] = []
    if evidence_count < 3:
        discrepancies.append(
            "Insufficient supporting evidence (minimum 3 documents recommended)"
        )
    if not verified:
        discrepancies.append(
            "Verification confidence below threshold (0.75)"
        )

    logger.info(
        "Declaration verification: supplier=%s verified=%s confidence=%s",
        body.supplier_id or "unknown",
        verified,
        confidence,
    )

    return SubstitutionVerifyResponse(
        verified=verified,
        confidence=confidence,
        discrepancies=discrepancies,
        evidence_reviewed=evidence_count,
    )


# ---------------------------------------------------------------------------
# GET /substitution/patterns
# ---------------------------------------------------------------------------


@router.get(
    "/substitution/patterns",
    response_model=SubstitutionPatternResponse,
    summary="Get substitution patterns",
    description=(
        "Analyze substitution patterns across all monitored suppliers "
        "to identify systemic risks and recurring substitution behaviors."
    ),
    responses={
        200: {"description": "Pattern analysis results"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_patterns(
    request: Request,
    date_range: Dict[str, Optional[date]] = Depends(validate_date_range),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:substitution:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SubstitutionPatternResponse:
    """Get substitution patterns across suppliers.

    Args:
        date_range: Optional date range filter.
        user: Authenticated user with substitution:read permission.

    Returns:
        SubstitutionPatternResponse with detected patterns.
    """
    # Aggregate patterns from stored substitution results
    origin_switches = 0
    commodity_swaps = 0
    affected_suppliers = set()

    for supplier_id, result in _substitution_store.items():
        for switch in result.detected_switches:
            affected_suppliers.add(supplier_id)
            if switch.from_country != switch.to_country:
                origin_switches += 1
            if switch.from_commodity != switch.to_commodity:
                commodity_swaps += 1

    patterns: List[SubstitutionPatternEntry] = []
    if origin_switches > 0:
        patterns.append(
            SubstitutionPatternEntry(
                pattern_type="origin_switch",
                frequency=origin_switches,
                affected_suppliers=len(affected_suppliers),
                risk_level=RiskLevelEnum.HIGH,
                description=(
                    f"Origin country switching detected across "
                    f"{len(affected_suppliers)} supplier(s)"
                ),
            )
        )
    if commodity_swaps > 0:
        patterns.append(
            SubstitutionPatternEntry(
                pattern_type="commodity_swap",
                frequency=commodity_swaps,
                affected_suppliers=len(affected_suppliers),
                risk_level=RiskLevelEnum.CRITICAL,
                description=(
                    f"Commodity type swaps detected across "
                    f"{len(affected_suppliers)} supplier(s)"
                ),
            )
        )

    today = date.today()
    return SubstitutionPatternResponse(
        patterns=patterns,
        total_patterns=len(patterns),
        analysis_period_start=date_range.get("start_date") or today.replace(month=1, day=1),
        analysis_period_end=date_range.get("end_date") or today,
    )
