# -*- coding: utf-8 -*-
"""
Severity Classification Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for five-tier severity classification using weighted scoring across
area (0.25), deforestation rate (0.20), proximity (0.25), protected area
overlay (0.15), and post-cutoff timing (0.15). Severity levels: CRITICAL,
HIGH, MEDIUM, LOW, INFORMATIONAL.

Endpoints:
    POST /severity/classify       - Classify alert severity
    POST /severity/reclassify     - Reclassify existing alert severity
    GET  /severity/thresholds     - Get current threshold configuration
    GET  /severity/distribution   - Get severity distribution across alerts

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, SeverityClassifier Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    get_das_config,
    get_severity_classifier,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    AlertSeverityEnum,
    ErrorResponse,
    MetadataSchema,
    ProvenanceInfo,
    SeverityClassifyRequest,
    SeverityClassifyResponse,
    SeverityDistributionEntry,
    SeverityDistributionResponse,
    SeverityReclassifyRequest,
    SeverityReclassifyResponse,
    SeverityScoreBreakdown,
    SeverityThresholdEntry,
    SeverityThresholdsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/severity", tags=["Severity Classification"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /severity/classify
# ---------------------------------------------------------------------------


@router.post(
    "/classify",
    response_model=SeverityClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify alert severity",
    description=(
        "Classify the severity of a deforestation alert using weighted multi-"
        "dimensional scoring: area (0.25), deforestation rate (0.20), proximity "
        "to supply chain plots (0.25), protected area overlay (0.15), and "
        "post-cutoff timing (0.15). Multipliers applied for protected areas "
        "(1.5x) and post-cutoff events (2.0x)."
    ),
    responses={
        200: {"description": "Severity classified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def classify_severity(
    request: Request,
    body: SeverityClassifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:severity:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SeverityClassifyResponse:
    """Classify the severity of a deforestation alert.

    Args:
        body: Classification request with alert metrics.
        user: Authenticated user with severity:create permission.

    Returns:
        SeverityClassifyResponse with classification result.
    """
    start = time.monotonic()

    try:
        engine = get_severity_classifier()
        result = engine.classify(
            alert_id=body.alert_id,
            area_ha=float(body.area_ha),
            deforestation_rate_ha_per_day=float(body.deforestation_rate_ha_per_day)
            if body.deforestation_rate_ha_per_day else None,
            proximity_km=float(body.proximity_km)
            if body.proximity_km else None,
            in_protected_area=body.in_protected_area,
            is_post_cutoff=body.is_post_cutoff,
            confidence=float(body.confidence) if body.confidence else None,
            custom_weights={k: float(v) for k, v in body.custom_weights.items()}
            if body.custom_weights else None,
        )

        breakdown_data = result.get("breakdown", {})
        breakdown = SeverityScoreBreakdown(
            area_score=Decimal(str(breakdown_data.get("area_score", 0))),
            area_weight=Decimal(str(breakdown_data.get("area_weight", "0.25"))),
            rate_score=Decimal(str(breakdown_data.get("rate_score", 0))),
            rate_weight=Decimal(str(breakdown_data.get("rate_weight", "0.20"))),
            proximity_score=Decimal(str(breakdown_data.get("proximity_score", 0))),
            proximity_weight=Decimal(str(breakdown_data.get("proximity_weight", "0.25"))),
            protected_score=Decimal(str(breakdown_data.get("protected_score", 0))),
            protected_weight=Decimal(str(breakdown_data.get("protected_weight", "0.15"))),
            timing_score=Decimal(str(breakdown_data.get("timing_score", 0))),
            timing_weight=Decimal(str(breakdown_data.get("timing_weight", "0.15"))),
            weighted_total=Decimal(str(breakdown_data.get("weighted_total", 0))),
            multiplier_applied=Decimal(str(breakdown_data.get("multiplier_applied", 1)))
            if breakdown_data.get("multiplier_applied") else None,
            final_score=Decimal(str(result.get("score", 0))),
        )

        severity = AlertSeverityEnum(result.get("severity", "medium"))
        score = Decimal(str(result.get("score", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"classify:{body.alert_id}:{body.area_ha}",
            str(severity.value),
        )

        logger.info(
            "Severity classified: alert_id=%s severity=%s score=%s operator=%s",
            body.alert_id,
            severity.value,
            score,
            user.operator_id or user.user_id,
        )

        return SeverityClassifyResponse(
            alert_id=body.alert_id,
            severity=severity,
            score=score,
            breakdown=breakdown,
            previous_severity=AlertSeverityEnum(result["previous_severity"])
            if result.get("previous_severity") else None,
            classification_reason=result.get("classification_reason", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["SeverityClassifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Severity classification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Severity classification failed",
        )


# ---------------------------------------------------------------------------
# POST /severity/reclassify
# ---------------------------------------------------------------------------


@router.post(
    "/reclassify",
    response_model=SeverityReclassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Reclassify existing alert severity",
    description=(
        "Reclassify the severity of an existing alert with updated metrics "
        "or a forced severity override. Records the reclassification in the "
        "audit trail with reason and operator identity."
    ),
    responses={
        200: {"description": "Severity reclassified"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def reclassify_severity(
    request: Request,
    body: SeverityReclassifyRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:severity:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> SeverityReclassifyResponse:
    """Reclassify the severity of an existing alert.

    Args:
        body: Reclassification request with updated metrics.
        user: Authenticated user with severity:update permission.

    Returns:
        SeverityReclassifyResponse with reclassification result.
    """
    start = time.monotonic()

    try:
        engine = get_severity_classifier()
        result = engine.reclassify(
            alert_id=body.alert_id,
            reason=body.reason,
            new_area_ha=float(body.new_area_ha) if body.new_area_ha else None,
            new_proximity_km=float(body.new_proximity_km) if body.new_proximity_km else None,
            force_severity=body.force_severity.value if body.force_severity else None,
            reclassified_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"reclassify:{body.alert_id}:{body.reason}",
            str(result.get("new_severity", "medium")),
        )

        logger.info(
            "Severity reclassified: alert_id=%s %s->%s operator=%s",
            body.alert_id,
            result.get("previous_severity", "unknown"),
            result.get("new_severity", "unknown"),
            user.operator_id or user.user_id,
        )

        return SeverityReclassifyResponse(
            alert_id=body.alert_id,
            previous_severity=AlertSeverityEnum(result.get("previous_severity", "medium")),
            new_severity=AlertSeverityEnum(result.get("new_severity", "medium")),
            score=Decimal(str(result.get("score", 0))),
            reason=body.reason,
            reclassified_by=user.user_id,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["SeverityClassifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Severity reclassification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Severity reclassification failed",
        )


# ---------------------------------------------------------------------------
# GET /severity/thresholds
# ---------------------------------------------------------------------------


@router.get(
    "/thresholds",
    response_model=SeverityThresholdsResponse,
    summary="Get current severity threshold configuration",
    description=(
        "Retrieve the current severity classification thresholds including "
        "area thresholds (critical >=50 ha, high >=10 ha, medium >=1 ha), "
        "proximity thresholds, scoring weights, and multipliers."
    ),
    responses={
        200: {"description": "Thresholds retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_thresholds(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:severity:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SeverityThresholdsResponse:
    """Get current severity threshold configuration.

    Args:
        user: Authenticated user with severity:read permission.

    Returns:
        SeverityThresholdsResponse with threshold configuration.
    """
    start = time.monotonic()

    try:
        config = get_das_config()

        thresholds = [
            SeverityThresholdEntry(
                severity=AlertSeverityEnum.CRITICAL,
                area_threshold_ha=config.critical_area_threshold_ha,
                proximity_threshold_km=config.proximity_critical_km,
                score_range_min=Decimal("0.80"),
                score_range_max=Decimal("1.00"),
            ),
            SeverityThresholdEntry(
                severity=AlertSeverityEnum.HIGH,
                area_threshold_ha=config.high_area_threshold_ha,
                proximity_threshold_km=config.proximity_high_km,
                score_range_min=Decimal("0.60"),
                score_range_max=Decimal("0.80"),
            ),
            SeverityThresholdEntry(
                severity=AlertSeverityEnum.MEDIUM,
                area_threshold_ha=config.medium_area_threshold_ha,
                proximity_threshold_km=config.proximity_medium_km,
                score_range_min=Decimal("0.40"),
                score_range_max=Decimal("0.60"),
            ),
            SeverityThresholdEntry(
                severity=AlertSeverityEnum.LOW,
                score_range_min=Decimal("0.20"),
                score_range_max=Decimal("0.40"),
            ),
            SeverityThresholdEntry(
                severity=AlertSeverityEnum.INFORMATIONAL,
                score_range_min=Decimal("0.00"),
                score_range_max=Decimal("0.20"),
            ),
        ]

        weights = {
            "area": config.area_weight,
            "rate": config.rate_weight,
            "proximity": config.proximity_weight,
            "protected": config.protected_weight,
            "timing": config.timing_weight,
        }

        multipliers = {
            "protected_area": config.protected_area_multiplier,
            "post_cutoff": config.post_cutoff_multiplier,
        }

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "severity_thresholds",
            str(len(thresholds)),
        )

        logger.info(
            "Severity thresholds retrieved: operator=%s",
            user.operator_id or user.user_id,
        )

        return SeverityThresholdsResponse(
            thresholds=thresholds,
            weights=weights,
            multipliers=multipliers,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["DeforestationAlertSystemConfig"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Threshold retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Severity threshold retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /severity/distribution
# ---------------------------------------------------------------------------


@router.get(
    "/distribution",
    response_model=SeverityDistributionResponse,
    summary="Get severity distribution across alerts",
    description=(
        "Retrieve the distribution of alerts across severity levels including "
        "counts, percentages, and average area and score per level."
    ),
    responses={
        200: {"description": "Distribution retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_distribution(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:severity:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SeverityDistributionResponse:
    """Get severity distribution across all alerts.

    Args:
        user: Authenticated user with severity:read permission.

    Returns:
        SeverityDistributionResponse with distribution data.
    """
    start = time.monotonic()

    try:
        engine = get_severity_classifier()
        result = engine.get_distribution()

        distribution = []
        for entry in result.get("distribution", []):
            distribution.append(
                SeverityDistributionEntry(
                    severity=AlertSeverityEnum(entry.get("severity", "medium")),
                    count=entry.get("count", 0),
                    percentage=Decimal(str(entry.get("percentage", 0))),
                    average_area_ha=Decimal(str(entry.get("average_area_ha", 0)))
                    if entry.get("average_area_ha") is not None else None,
                    average_score=Decimal(str(entry.get("average_score", 0)))
                    if entry.get("average_score") is not None else None,
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            "severity_distribution",
            str(result.get("total_alerts", 0)),
        )

        logger.info(
            "Severity distribution retrieved: total=%d operator=%s",
            result.get("total_alerts", 0),
            user.operator_id or user.user_id,
        )

        return SeverityDistributionResponse(
            distribution=distribution,
            total_alerts=result.get("total_alerts", 0),
            average_severity_score=Decimal(str(result.get("average_severity_score", 0)))
            if result.get("average_severity_score") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["SeverityClassifier"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Severity distribution retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Severity distribution retrieval failed",
        )
