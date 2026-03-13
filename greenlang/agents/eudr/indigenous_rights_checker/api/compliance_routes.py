# -*- coding: utf-8 -*-
"""
Compliance Reporting Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for generating indigenous rights compliance reports and
performing full compliance assessments covering territory overlaps,
FPIC verification, violation status, and consultation completeness
for EUDR due diligence obligations.

Endpoints:
    GET  /compliance/report/{plot_id}   - Generate compliance report for a plot
    POST /compliance/assess             - Full compliance assessment

Compliance scoring: 0-100 scale combining overlap risk, FPIC status,
violation count, and consultation coverage. Thresholds:
    >= 80: COMPLIANT
    60-79: AT_RISK
    40-59: REQUIRES_ASSESSMENT
    < 40:  NON_COMPLIANT

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, ComplianceReporter Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.indigenous_rights_checker.api.dependencies import (
    AuthUser,
    get_compliance_reporter,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    ComplianceAssessRequest,
    ComplianceAssessResponse,
    CompliancePlotSummary,
    ComplianceReportResponse,
    ComplianceStatusEnum,
    ErrorResponse,
    FPICStatusEnum,
    MetadataSchema,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance Reporting"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# GET /compliance/report/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/report/{plot_id}",
    response_model=ComplianceReportResponse,
    summary="Generate compliance report for a plot",
    description=(
        "Generate a comprehensive indigenous rights compliance report for "
        "a specific supply chain plot. The report aggregates territory "
        "overlap analysis, FPIC verification status, active violations, "
        "consultation completeness, and affected community information "
        "into a single compliance score and status determination."
    ),
    responses={
        200: {"description": "Compliance report generated"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
    },
)
async def generate_compliance_report(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceReportResponse:
    """Generate a compliance report for a specific plot.

    Args:
        plot_id: Plot identifier to generate report for.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceReportResponse with aggregated compliance assessment.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_reporter()
        result = engine.generate_report(
            plot_id=plot_id,
            generated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plot not found or no data available: {plot_id}",
            )

        # Determine compliance status from score
        score = Decimal(str(result.get("overall_score", 0)))
        compliance_status = ComplianceStatusEnum.COMPLIANT
        if result.get("compliance_status"):
            compliance_status = ComplianceStatusEnum(result["compliance_status"])
        elif score < Decimal("40"):
            compliance_status = ComplianceStatusEnum.NON_COMPLIANT
        elif score < Decimal("60"):
            compliance_status = ComplianceStatusEnum.REQUIRES_ASSESSMENT
        elif score < Decimal("80"):
            compliance_status = ComplianceStatusEnum.AT_RISK

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_report:{plot_id}",
            f"{score}:{compliance_status.value}",
        )

        logger.info(
            "Compliance report generated: plot_id=%s score=%s status=%s operator=%s",
            plot_id,
            score,
            compliance_status.value,
            user.operator_id or user.user_id,
        )

        return ComplianceReportResponse(
            report_id=result.get("report_id", ""),
            plot_id=plot_id,
            compliance_status=compliance_status,
            overall_score=score,
            territory_overlaps=result.get("territory_overlaps", 0),
            fpic_status=FPICStatusEnum(
                result.get("fpic_status", "not_required")
            ),
            active_violations=result.get("active_violations", 0),
            consultations_completed=result.get("consultations_completed", 0),
            risk_factors=result.get("risk_factors", []),
            recommendations=result.get("recommendations", []),
            affected_communities=result.get("affected_communities", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "IndigenousRightsChecker",
                    "ComplianceReporter",
                    "TerritoryManager",
                    "FPICVerifier",
                    "ViolationDetector",
                    "ConsultationTracker",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance report generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance report generation failed",
        )


# ---------------------------------------------------------------------------
# POST /compliance/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=ComplianceAssessResponse,
    status_code=status.HTTP_200_OK,
    summary="Full compliance assessment",
    description=(
        "Perform a comprehensive indigenous rights compliance assessment "
        "across multiple plots. Optionally includes territory overlap "
        "analysis, FPIC verification, violation detection, and consultation "
        "review. Returns per-plot summaries and aggregate compliance status. "
        "Supports up to 100 plots per request."
    ),
    responses={
        200: {"description": "Compliance assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request (max 100 plots)"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_compliance(
    request: Request,
    body: ComplianceAssessRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:compliance:read")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> ComplianceAssessResponse:
    """Perform full compliance assessment across multiple plots.

    Args:
        body: Compliance assessment request with plot IDs and options.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceAssessResponse with per-plot and aggregate results.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_reporter()
        result = engine.assess_compliance(
            plot_ids=body.plot_ids,
            supplier_id=body.supplier_id,
            commodity=body.commodity.value if body.commodity else None,
            include_overlap_analysis=body.include_overlap_analysis,
            include_fpic_verification=body.include_fpic_verification,
            include_violation_check=body.include_violation_check,
            include_consultation_review=body.include_consultation_review,
            assessed_by=user.user_id,
        )

        # Build per-plot summaries
        plots: List[CompliancePlotSummary] = []
        compliant_count = 0
        non_compliant_count = 0
        at_risk_count = 0

        for plot_data in result.get("plots", []):
            plot_score = Decimal(str(plot_data.get("score", 0)))

            # Determine per-plot status
            if plot_data.get("compliance_status"):
                plot_status = ComplianceStatusEnum(plot_data["compliance_status"])
            elif plot_score >= Decimal("80"):
                plot_status = ComplianceStatusEnum.COMPLIANT
            elif plot_score >= Decimal("60"):
                plot_status = ComplianceStatusEnum.AT_RISK
            elif plot_score >= Decimal("40"):
                plot_status = ComplianceStatusEnum.REQUIRES_ASSESSMENT
            else:
                plot_status = ComplianceStatusEnum.NON_COMPLIANT

            if plot_status == ComplianceStatusEnum.COMPLIANT:
                compliant_count += 1
            elif plot_status == ComplianceStatusEnum.NON_COMPLIANT:
                non_compliant_count += 1
            elif plot_status in (
                ComplianceStatusEnum.AT_RISK,
                ComplianceStatusEnum.REQUIRES_ASSESSMENT,
            ):
                at_risk_count += 1

            plots.append(
                CompliancePlotSummary(
                    plot_id=plot_data.get("plot_id", ""),
                    compliance_status=plot_status,
                    score=plot_score,
                    overlap_count=plot_data.get("overlap_count", 0),
                    fpic_status=FPICStatusEnum(plot_data.get("fpic_status"))
                    if plot_data.get("fpic_status") else None,
                    violation_count=plot_data.get("violation_count", 0),
                    issues=plot_data.get("issues", []),
                )
            )

        # Determine aggregate compliance
        overall_score = Decimal(str(result.get("overall_score", 0)))
        if result.get("overall_compliance"):
            overall_compliance = ComplianceStatusEnum(result["overall_compliance"])
        elif non_compliant_count > 0:
            overall_compliance = ComplianceStatusEnum.NON_COMPLIANT
        elif at_risk_count > 0:
            overall_compliance = ComplianceStatusEnum.AT_RISK
        elif overall_score >= Decimal("80"):
            overall_compliance = ComplianceStatusEnum.COMPLIANT
        else:
            overall_compliance = ComplianceStatusEnum.REQUIRES_ASSESSMENT

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_assess:{len(body.plot_ids)}:{body.supplier_id}",
            f"{overall_score}:{overall_compliance.value}",
        )

        logger.info(
            "Compliance assessment: plots=%d compliant=%d non_compliant=%d at_risk=%d score=%s operator=%s",
            len(body.plot_ids),
            compliant_count,
            non_compliant_count,
            at_risk_count,
            overall_score,
            user.operator_id or user.user_id,
        )

        return ComplianceAssessResponse(
            assessment_id=result.get("assessment_id", ""),
            overall_compliance=overall_compliance,
            overall_score=overall_score,
            total_plots_assessed=len(plots),
            compliant_plots=compliant_count,
            non_compliant_plots=non_compliant_count,
            at_risk_plots=at_risk_count,
            plots=plots,
            risk_factors=result.get("risk_factors", []),
            recommendations=result.get("recommendations", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "IndigenousRightsChecker",
                    "ComplianceReporter",
                    "TerritoryManager",
                    "FPICVerifier",
                    "ViolationDetector",
                    "ConsultationTracker",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance assessment failed",
        )
