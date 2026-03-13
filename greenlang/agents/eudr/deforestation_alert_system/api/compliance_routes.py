# -*- coding: utf-8 -*-
"""
Compliance Impact Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for compliance impact assessment mapping deforestation alerts to
affected suppliers, products, market restrictions, remediation actions, and
estimated financial impact with auto-assessment and market restriction
thresholds per EUDR Articles 2, 9, 10, 11.

Endpoints:
    POST /compliance/assess                        - Assess compliance impact
    GET  /compliance/{alert_id}/affected-products  - Get affected products
    GET  /compliance/recommendations               - Get compliance recommendations
    POST /compliance/remediation                   - Create remediation plan

Market Restriction: Triggered at HIGH severity
Remediation Plan: Required for non-compliant events

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, ComplianceImpactAssessor Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    get_compliance_assessor,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    AffectedProductEntry,
    AffectedProductsResponse,
    AffectedSupplierEntry,
    AlertSeverityEnum,
    ComplianceAssessRequest,
    ComplianceAssessResponse,
    ComplianceOutcomeEnum,
    ComplianceRecommendationEntry,
    ComplianceRecommendationsResponse,
    EUDRCommodityEnum,
    ErrorResponse,
    MetadataSchema,
    ProvenanceInfo,
    RemediationActionEnum,
    RemediationPlanRequest,
    RemediationPlanResponse,
    WorkflowPriorityEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compliance", tags=["Compliance Impact"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /compliance/assess
# ---------------------------------------------------------------------------


@router.post(
    "/assess",
    response_model=ComplianceAssessResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess compliance impact of deforestation alert",
    description=(
        "Perform a comprehensive EUDR compliance impact assessment for a "
        "deforestation alert. Determines affected suppliers, products, market "
        "restrictions, financial impact, and generates remediation "
        "recommendations per EUDR Articles 2, 9, 10, 11."
    ),
    responses={
        200: {"description": "Compliance impact assessed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_compliance(
    request: Request,
    body: ComplianceAssessRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:compliance:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ComplianceAssessResponse:
    """Assess EUDR compliance impact of a deforestation alert.

    Args:
        body: Assessment request with alert ID and options.
        user: Authenticated user with compliance:create permission.

    Returns:
        ComplianceAssessResponse with assessment results.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.assess(
            alert_id=body.alert_id,
            include_affected_products=body.include_affected_products,
            include_financial_impact=body.include_financial_impact,
            include_recommendations=body.include_recommendations,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            operator_id=body.operator_id or user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        outcome = ComplianceOutcomeEnum(result.get("compliance_outcome", "requires_investigation"))
        severity = AlertSeverityEnum(result.get("severity", "medium"))

        suppliers = []
        for s in result.get("affected_suppliers", []):
            suppliers.append(
                AffectedSupplierEntry(
                    supplier_id=s.get("supplier_id", ""),
                    supplier_name=s.get("supplier_name"),
                    country_code=s.get("country_code"),
                    risk_level=s.get("risk_level", "medium"),
                    commodities=[EUDRCommodityEnum(c) for c in s.get("commodities", [])],
                    estimated_volume_affected=Decimal(str(s.get("estimated_volume_affected", 0)))
                    if s.get("estimated_volume_affected") is not None else None,
                )
            )

        products = []
        for p in result.get("affected_products", []):
            products.append(
                AffectedProductEntry(
                    product_id=p.get("product_id", ""),
                    product_name=p.get("product_name"),
                    commodity=EUDRCommodityEnum(p.get("commodity", "wood")),
                    cn_code=p.get("cn_code"),
                    estimated_value_eur=Decimal(str(p.get("estimated_value_eur", 0)))
                    if p.get("estimated_value_eur") is not None else None,
                    market_restriction_risk=p.get("market_restriction_risk", "low"),
                )
            )

        recommendations = []
        for r in result.get("recommendations", []):
            recommendations.append(
                ComplianceRecommendationEntry(
                    recommendation_id=r.get("recommendation_id", ""),
                    action=RemediationActionEnum(r.get("action", "enhanced_monitoring")),
                    priority=WorkflowPriorityEnum(r.get("priority", "medium")),
                    description=r.get("description", ""),
                    estimated_timeline_days=r.get("estimated_timeline_days"),
                    regulatory_reference=r.get("regulatory_reference"),
                )
            )

        market_restriction = result.get("market_restriction_triggered", False)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compliance_assess:{body.alert_id}",
            str(outcome.value),
        )

        logger.info(
            "Compliance assessed: alert_id=%s outcome=%s market_restriction=%s "
            "suppliers=%d products=%d operator=%s",
            body.alert_id,
            outcome.value,
            market_restriction,
            len(suppliers),
            len(products),
            user.operator_id or user.user_id,
        )

        return ComplianceAssessResponse(
            alert_id=body.alert_id,
            compliance_outcome=outcome,
            severity=severity,
            is_post_cutoff=result.get("is_post_cutoff", False),
            affected_suppliers=suppliers,
            affected_products=products,
            total_affected_suppliers=len(suppliers),
            total_affected_products=len(products),
            estimated_financial_impact_eur=Decimal(str(result.get("estimated_financial_impact_eur", 0)))
            if result.get("estimated_financial_impact_eur") is not None else None,
            market_restriction_triggered=market_restriction,
            recommendations=recommendations,
            regulatory_articles=result.get("regulatory_articles", ["Art. 2", "Art. 9", "Art. 10"]),
            assessment_rationale=result.get("assessment_rationale", ""),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "ComplianceImpactAssessor",
                    "DeforestationAlertSystem",
                    "EUDR Supply Chain Data",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Compliance assessment failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance impact assessment failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/{alert_id}/affected-products
# ---------------------------------------------------------------------------


@router.get(
    "/{alert_id}/affected-products",
    response_model=AffectedProductsResponse,
    summary="Get affected products for an alert",
    description=(
        "Retrieve the list of products and suppliers affected by a "
        "deforestation alert, including commodity types, CN codes, "
        "and estimated financial exposure."
    ),
    responses={
        200: {"description": "Affected products retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def get_affected_products(
    alert_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AffectedProductsResponse:
    """Get affected products for a deforestation alert.

    Args:
        alert_id: Alert identifier.
        user: Authenticated user with compliance:read permission.

    Returns:
        AffectedProductsResponse with product and supplier data.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.get_affected_products(alert_id=alert_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )

        products = []
        for p in result.get("affected_products", []):
            products.append(
                AffectedProductEntry(
                    product_id=p.get("product_id", ""),
                    product_name=p.get("product_name"),
                    commodity=EUDRCommodityEnum(p.get("commodity", "wood")),
                    cn_code=p.get("cn_code"),
                    estimated_value_eur=Decimal(str(p.get("estimated_value_eur", 0)))
                    if p.get("estimated_value_eur") is not None else None,
                    market_restriction_risk=p.get("market_restriction_risk", "low"),
                )
            )

        suppliers = []
        for s in result.get("affected_suppliers", []):
            suppliers.append(
                AffectedSupplierEntry(
                    supplier_id=s.get("supplier_id", ""),
                    supplier_name=s.get("supplier_name"),
                    country_code=s.get("country_code"),
                    risk_level=s.get("risk_level", "medium"),
                    commodities=[EUDRCommodityEnum(c) for c in s.get("commodities", [])],
                    estimated_volume_affected=Decimal(str(s.get("estimated_volume_affected", 0)))
                    if s.get("estimated_volume_affected") is not None else None,
                )
            )

        total_value = sum(
            (p.estimated_value_eur or Decimal("0")) for p in products
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"affected_products:{alert_id}",
            str(len(products)),
        )

        logger.info(
            "Affected products retrieved: alert_id=%s products=%d suppliers=%d operator=%s",
            alert_id,
            len(products),
            len(suppliers),
            user.operator_id or user.user_id,
        )

        return AffectedProductsResponse(
            alert_id=alert_id,
            affected_products=products,
            affected_suppliers=suppliers,
            total_products=len(products),
            total_suppliers=len(suppliers),
            total_estimated_value_eur=total_value if total_value > 0 else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceImpactAssessor", "EUDR Supply Chain Data"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Affected products retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Affected products retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /compliance/recommendations
# ---------------------------------------------------------------------------


@router.get(
    "/recommendations",
    response_model=ComplianceRecommendationsResponse,
    summary="Get compliance recommendations",
    description=(
        "Retrieve prioritized compliance remediation recommendations based "
        "on current alert severity and compliance status."
    ),
    responses={
        200: {"description": "Recommendations retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_recommendations(
    request: Request,
    severity: Optional[AlertSeverityEnum] = Query(
        None, description="Filter by alert severity"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by commodity"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:compliance:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ComplianceRecommendationsResponse:
    """Get compliance recommendations.

    Args:
        severity: Optional severity filter.
        commodity: Optional commodity filter.
        user: Authenticated user with compliance:read permission.

    Returns:
        ComplianceRecommendationsResponse with recommendations.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.get_recommendations(
            severity=severity.value if severity else None,
            commodity=commodity.value if commodity else None,
        )

        recommendations = []
        for r in result.get("recommendations", []):
            recommendations.append(
                ComplianceRecommendationEntry(
                    recommendation_id=r.get("recommendation_id", ""),
                    action=RemediationActionEnum(r.get("action", "enhanced_monitoring")),
                    priority=WorkflowPriorityEnum(r.get("priority", "medium")),
                    description=r.get("description", ""),
                    estimated_timeline_days=r.get("estimated_timeline_days"),
                    regulatory_reference=r.get("regulatory_reference"),
                )
            )

        urgent_count = sum(
            1 for r in recommendations if r.priority == WorkflowPriorityEnum.URGENT
        )
        total_timeline = sum(
            (r.estimated_timeline_days or 0) for r in recommendations
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"recommendations:{severity}:{commodity}",
            str(len(recommendations)),
        )

        logger.info(
            "Compliance recommendations: total=%d urgent=%d operator=%s",
            len(recommendations),
            urgent_count,
            user.operator_id or user.user_id,
        )

        return ComplianceRecommendationsResponse(
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            urgent_count=urgent_count,
            estimated_total_timeline_days=total_timeline if total_timeline > 0 else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceImpactAssessor", "EUDR Best Practices"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Recommendations retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance recommendations retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /compliance/remediation
# ---------------------------------------------------------------------------


@router.post(
    "/remediation",
    response_model=RemediationPlanResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create remediation plan for compliance incident",
    description=(
        "Create a formal remediation plan for a non-compliant deforestation "
        "alert specifying actions, responsible parties, target dates, and "
        "affected supply chain entities."
    ),
    responses={
        201: {"description": "Remediation plan created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Alert not found"},
    },
)
async def create_remediation_plan(
    request: Request,
    body: RemediationPlanRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:compliance:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> RemediationPlanResponse:
    """Create a remediation plan for a compliance incident.

    Args:
        body: Remediation plan request.
        user: Authenticated user with compliance:create permission.

    Returns:
        RemediationPlanResponse with created plan.
    """
    start = time.monotonic()

    try:
        engine = get_compliance_assessor()
        result = engine.create_remediation_plan(
            alert_id=body.alert_id,
            actions=[a.value for a in body.actions],
            target_completion_date=body.target_completion_date,
            responsible_party=body.responsible_party,
            description=body.description,
            affected_supplier_ids=body.affected_supplier_ids,
            affected_product_ids=body.affected_product_ids,
            estimated_cost_eur=float(body.estimated_cost_eur) if body.estimated_cost_eur else None,
            created_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {body.alert_id}",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"remediation:{body.alert_id}:{body.responsible_party}",
            str(result.get("plan_id", "")),
        )

        logger.info(
            "Remediation plan created: plan_id=%s alert_id=%s actions=%d operator=%s",
            result.get("plan_id", ""),
            body.alert_id,
            len(body.actions),
            user.operator_id or user.user_id,
        )

        return RemediationPlanResponse(
            plan_id=result.get("plan_id", ""),
            alert_id=body.alert_id,
            status="active",
            actions=body.actions,
            target_completion_date=body.target_completion_date,
            responsible_party=body.responsible_party,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ComplianceImpactAssessor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Remediation plan creation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Remediation plan creation failed",
        )
