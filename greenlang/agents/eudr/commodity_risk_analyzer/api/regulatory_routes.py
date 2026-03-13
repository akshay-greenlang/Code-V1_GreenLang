# -*- coding: utf-8 -*-
"""
Regulatory Routes - AGENT-EUDR-018 Commodity Risk Analyzer API

Endpoints for EUDR regulatory compliance including requirements lookup,
compliance checking, penalty risk assessment, regulatory updates,
and documentation requirements.

Endpoints:
    GET  /regulatory/{commodity_id}/requirements    - Requirements
    POST /regulatory/check-compliance               - Compliance check
    GET  /regulatory/penalty-risk                    - Penalty risk
    GET  /regulatory/updates                         - Regulatory updates
    GET  /regulatory/documentation-requirements      - Documentation needed

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018, Regulatory Compliance Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.commodity_risk_analyzer.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_regulatory_compliance_engine,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity_type,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.schemas import (
    ArticleRequirement,
    CommodityTypeEnum,
    ComplianceCheckRequest,
    ComplianceCheckResponse,
    ComplianceGap,
    ComplianceStatusEnum,
    DocumentationRequirementEntry,
    DocumentationRequirementsResponse,
    PenaltyCategoryEnum,
    PenaltyRiskFactor,
    PenaltyRiskResponse,
    ProvenanceInfo,
    RegulatoryRequirementsResponse,
    RegulatoryUpdateEntry,
    RegulatoryUpdatesResponse,
    RemediationStep,
    SeveritySummaryEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Regulatory Compliance"])

# ---------------------------------------------------------------------------
# Static regulatory data
# ---------------------------------------------------------------------------

_EUDR_ARTICLES: List[ArticleRequirement] = [
    ArticleRequirement(
        article_id="Art. 3",
        article_title="Prohibition",
        requirement_text="Relevant commodities and products shall not be placed on or made available on the Union market or exported unless they are deforestation-free, produced in accordance with the relevant legislation of the country of production, and covered by a due diligence statement.",
        mandatory=True,
        documentation_types=["due_diligence_statement", "geolocation_data", "supplier_declaration"],
    ),
    ArticleRequirement(
        article_id="Art. 4(2)(a)",
        article_title="Due Diligence - Description",
        requirement_text="Description of the relevant commodity or relevant product, including the trade name and type of product, as well as the common and scientific name of the species.",
        mandatory=True,
        documentation_types=["product_description", "species_identification"],
    ),
    ArticleRequirement(
        article_id="Art. 4(2)(b)",
        article_title="Due Diligence - Quantity",
        requirement_text="Quantity of the relevant commodity or relevant product.",
        mandatory=True,
        documentation_types=["quantity_declaration", "trade_document"],
    ),
    ArticleRequirement(
        article_id="Art. 4(2)(f)",
        article_title="Due Diligence - Geolocation",
        requirement_text="Geolocation coordinates of all plots of land where the relevant commodities were produced, and date or time range of production.",
        mandatory=True,
        documentation_types=["gps_coordinates", "geolocation_data", "satellite_image"],
    ),
    ArticleRequirement(
        article_id="Art. 9",
        article_title="Risk Assessment",
        requirement_text="Operators shall assess the risk that relevant commodities or products are not compliant with Article 3.",
        mandatory=True,
        documentation_types=["risk_assessment_report", "country_risk_data"],
    ),
    ArticleRequirement(
        article_id="Art. 10",
        article_title="Risk Mitigation",
        requirement_text="Where the risk assessment identifies a non-negligible risk, operators shall carry out risk mitigation measures.",
        mandatory=True,
        documentation_types=["risk_mitigation_plan", "enhanced_due_diligence"],
    ),
    ArticleRequirement(
        article_id="Art. 12",
        article_title="Simplified Due Diligence",
        requirement_text="Simplified due diligence applicable for commodities from low-risk countries as classified by the Commission.",
        mandatory=False,
        documentation_types=["country_classification", "simplified_declaration"],
    ),
]

_DOCUMENTATION_REQUIREMENTS: List[DocumentationRequirementEntry] = [
    DocumentationRequirementEntry(document_type="geolocation_data", description="GPS coordinates of all production plots in WGS84 format", mandatory=True, applicable_articles=["Art. 4(2)(f)"], accepted_formats=["GeoJSON", "KML", "CSV", "Shapefile"]),
    DocumentationRequirementEntry(document_type="supplier_declaration", description="Signed declaration from each supplier in the supply chain", mandatory=True, applicable_articles=["Art. 3", "Art. 4"], accepted_formats=["PDF", "signed_digital"]),
    DocumentationRequirementEntry(document_type="due_diligence_statement", description="Complete due diligence statement per Article 4", mandatory=True, applicable_articles=["Art. 3", "Art. 4"], accepted_formats=["PDF", "XML"]),
    DocumentationRequirementEntry(document_type="risk_assessment_report", description="Documented risk assessment per Article 9", mandatory=True, applicable_articles=["Art. 9"], accepted_formats=["PDF", "JSON"]),
    DocumentationRequirementEntry(document_type="satellite_image", description="Satellite imagery for forest cover verification", mandatory=False, applicable_articles=["Art. 4(2)(f)", "Art. 9"], accepted_formats=["GeoTIFF", "PNG", "JPEG"]),
    DocumentationRequirementEntry(document_type="certificate", description="Third-party sustainability certificates (FSC, RSPO, RA, etc.)", mandatory=False, applicable_articles=["Art. 10"], accepted_formats=["PDF"]),
    DocumentationRequirementEntry(document_type="trade_document", description="Customs and trade documentation (invoices, bills of lading)", mandatory=True, applicable_articles=["Art. 4(2)(b)"], accepted_formats=["PDF", "EDI"]),
    DocumentationRequirementEntry(document_type="audit_report", description="Third-party audit reports for enhanced due diligence", mandatory=False, applicable_articles=["Art. 10"], accepted_formats=["PDF"]),
]


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _resolve_commodity(commodity_id: str) -> str:
    """Resolve commodity_id to commodity type string."""
    normalized = commodity_id.strip().lower()
    valid = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
    if normalized in valid:
        return normalized
    from greenlang.agents.eudr.commodity_risk_analyzer.api.commodity_routes import (
        _profile_store,
    )

    profile = _profile_store.get(commodity_id)
    if profile:
        return profile.commodity_type.value
    return normalized


# ---------------------------------------------------------------------------
# GET /regulatory/{commodity_id}/requirements
# ---------------------------------------------------------------------------


@router.get(
    "/regulatory/{commodity_id}/requirements",
    response_model=RegulatoryRequirementsResponse,
    summary="Get EUDR regulatory requirements",
    description=(
        "Retrieve all EUDR article requirements applicable to a specific "
        "commodity, including mandatory documentation and evidence types."
    ),
    responses={
        200: {"description": "Regulatory requirements"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_requirements(
    commodity_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:regulatory:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RegulatoryRequirementsResponse:
    """Get EUDR regulatory requirements for a commodity.

    Args:
        commodity_id: Commodity identifier or type name.
        user: Authenticated user with regulatory:read permission.

    Returns:
        RegulatoryRequirementsResponse with applicable requirements.
    """
    ct = _resolve_commodity(commodity_id)
    valid = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
    if ct not in valid:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Regulatory data not found for commodity: {commodity_id}",
        )

    # All EUDR articles apply to all 7 commodities
    articles = _EUDR_ARTICLES.copy()

    # Collect all required documentation types
    doc_types = set()
    for art in articles:
        if art.mandatory:
            doc_types.update(art.documentation_types)

    return RegulatoryRequirementsResponse(
        commodity_type=CommodityTypeEnum(ct),
        articles=articles,
        documentation_needed=sorted(doc_types),
        total_requirements=len(articles),
    )


# ---------------------------------------------------------------------------
# POST /regulatory/check-compliance
# ---------------------------------------------------------------------------


@router.post(
    "/regulatory/check-compliance",
    response_model=ComplianceCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check regulatory compliance",
    description=(
        "Perform a compliance check for a commodity against EUDR requirements. "
        "Identifies gaps and provides remediation recommendations."
    ),
    responses={
        200: {"description": "Compliance check completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def check_compliance(
    request: Request,
    body: ComplianceCheckRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:regulatory:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ComplianceCheckResponse:
    """Check regulatory compliance for a commodity.

    Args:
        body: Compliance check request with supplier data and documentation.
        user: Authenticated user with regulatory:write permission.

    Returns:
        ComplianceCheckResponse with compliance score and gaps.
    """
    start = time.monotonic()
    try:
        # Determine which mandatory articles are satisfied by provided docs
        provided_docs = set(body.documentation)
        certifications = body.supplier_data.get("certifications", [])

        gaps: List[ComplianceGap] = []
        remediation_steps: List[RemediationStep] = []
        articles_assessed: List[str] = []
        satisfied = 0
        total_mandatory = 0

        for art in _EUDR_ARTICLES:
            articles_assessed.append(art.article_id)
            if not art.mandatory:
                continue
            total_mandatory += 1

            # Check if any required doc type is provided
            has_doc = bool(set(art.documentation_types) & provided_docs)
            if has_doc:
                satisfied += 1
            else:
                gaps.append(
                    ComplianceGap(
                        article_reference=art.article_id,
                        requirement=art.requirement_text[:200],
                        severity=SeveritySummaryEnum.HIGH,
                    )
                )
                step_num = len(remediation_steps) + 1
                remediation_steps.append(
                    RemediationStep(
                        step_number=step_num,
                        action=f"Provide {', '.join(art.documentation_types)} for {art.article_id}",
                        priority=SeveritySummaryEnum.HIGH,
                        estimated_effort="2-4 hours",
                    )
                )

        # Compute compliance score
        compliance_score = Decimal("0.0")
        if total_mandatory > 0:
            compliance_score = (
                Decimal(str(satisfied)) / Decimal(str(total_mandatory)) * Decimal("100.0")
            ).quantize(Decimal("0.01"))

        # Determine status
        if compliance_score >= Decimal("100.0"):
            compliance_status = ComplianceStatusEnum.COMPLIANT
        elif compliance_score >= Decimal("50.0"):
            compliance_status = ComplianceStatusEnum.PARTIALLY_COMPLIANT
        else:
            compliance_status = ComplianceStatusEnum.NON_COMPLIANT

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            body.model_dump_json(), f"{body.commodity_type.value}:{compliance_score}"
        )

        logger.info(
            "Compliance check completed: commodity=%s score=%s status=%s gaps=%d",
            body.commodity_type.value,
            compliance_score,
            compliance_status.value,
            len(gaps),
        )

        return ComplianceCheckResponse(
            commodity_type=body.commodity_type,
            compliance_score=compliance_score,
            compliance_status=compliance_status,
            gaps=gaps,
            remediation_steps=remediation_steps,
            articles_assessed=articles_assessed,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

    except Exception as exc:
        logger.error("Compliance check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Compliance check failed",
        )


# ---------------------------------------------------------------------------
# GET /regulatory/penalty-risk
# ---------------------------------------------------------------------------


@router.get(
    "/regulatory/penalty-risk",
    response_model=PenaltyRiskResponse,
    summary="Get penalty risk assessment",
    description=(
        "Assess the penalty risk for a commodity per EUDR Article 25, "
        "including estimated fine ranges and contributing risk factors."
    ),
    responses={
        200: {"description": "Penalty risk assessment"},
        400: {"model": ErrorResponse, "description": "Invalid commodity type"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_penalty_risk(
    request: Request,
    commodity_type: Optional[str] = Depends(validate_commodity_type),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:regulatory:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PenaltyRiskResponse:
    """Get penalty risk assessment for a commodity.

    Args:
        commodity_type: EUDR commodity type.
        user: Authenticated user with regulatory:read permission.

    Returns:
        PenaltyRiskResponse with penalty category and fine estimates.
    """
    ct = commodity_type or "cocoa"
    ct_enum = CommodityTypeEnum(ct)

    # Assess penalty risk based on commodity type (simplified)
    risk_factors: List[PenaltyRiskFactor] = [
        PenaltyRiskFactor(
            factor_name="Documentation completeness",
            factor_score=Decimal("35.0"),
            description="Level of documentation available for due diligence compliance",
        ),
        PenaltyRiskFactor(
            factor_name="Geolocation coverage",
            factor_score=Decimal("40.0"),
            description="GPS coordinate coverage of production plots",
        ),
        PenaltyRiskFactor(
            factor_name="Supply chain transparency",
            factor_score=Decimal("30.0"),
            description="Visibility into multi-tier supply chain actors",
        ),
    ]

    overall_risk = sum(f.factor_score for f in risk_factors) / Decimal(str(len(risk_factors)))

    if overall_risk >= Decimal("70.0"):
        category = PenaltyCategoryEnum.CRITICAL
        fine_range = {"min": Decimal("500000"), "max": Decimal("4000000")}
    elif overall_risk >= Decimal("50.0"):
        category = PenaltyCategoryEnum.SEVERE
        fine_range = {"min": Decimal("100000"), "max": Decimal("500000")}
    elif overall_risk >= Decimal("25.0"):
        category = PenaltyCategoryEnum.SIGNIFICANT
        fine_range = {"min": Decimal("10000"), "max": Decimal("100000")}
    else:
        category = PenaltyCategoryEnum.MINOR
        fine_range = {"min": Decimal("1000"), "max": Decimal("10000")}

    return PenaltyRiskResponse(
        commodity_type=ct_enum,
        penalty_category=category,
        estimated_fine_range=fine_range,
        risk_factors=risk_factors,
        overall_penalty_risk=overall_risk.quantize(Decimal("0.01")),
    )


# ---------------------------------------------------------------------------
# GET /regulatory/updates
# ---------------------------------------------------------------------------


@router.get(
    "/regulatory/updates",
    response_model=RegulatoryUpdatesResponse,
    summary="Get regulatory updates",
    description="Retrieve recent EUDR regulatory updates and changes.",
    responses={
        200: {"description": "Regulatory updates"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_regulatory_updates(
    request: Request,
    commodity_type: Optional[str] = Depends(validate_commodity_type),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:regulatory:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RegulatoryUpdatesResponse:
    """Get recent regulatory updates.

    Args:
        commodity_type: Optional commodity type filter.
        user: Authenticated user with regulatory:read permission.

    Returns:
        RegulatoryUpdatesResponse with recent updates.
    """
    updates: List[RegulatoryUpdateEntry] = [
        RegulatoryUpdateEntry(
            title="EUDR Enforcement Date Confirmed",
            summary="EU Commission confirms December 30, 2025 enforcement date for large operators and June 30, 2026 for SMEs.",
            effective_date=date(2025, 12, 30),
            impact_level=SeveritySummaryEnum.CRITICAL,
            affected_commodities=[ct for ct in CommodityTypeEnum],
        ),
        RegulatoryUpdateEntry(
            title="Country Benchmarking System Published",
            summary="EU Commission publishes initial country risk benchmarking classifications for simplified due diligence eligibility.",
            effective_date=date(2026, 1, 15),
            impact_level=SeveritySummaryEnum.HIGH,
            affected_commodities=[ct for ct in CommodityTypeEnum],
        ),
        RegulatoryUpdateEntry(
            title="Geolocation Accuracy Requirements Clarified",
            summary="European Commission guidance clarifies minimum GPS accuracy requirements for production plot coordinates.",
            effective_date=date(2026, 2, 1),
            impact_level=SeveritySummaryEnum.MEDIUM,
            affected_commodities=[ct for ct in CommodityTypeEnum],
        ),
    ]

    if commodity_type:
        ct_enum = CommodityTypeEnum(commodity_type)
        updates = [
            u for u in updates if ct_enum in u.affected_commodities
        ]

    return RegulatoryUpdatesResponse(
        updates=updates,
        total_updates=len(updates),
    )


# ---------------------------------------------------------------------------
# GET /regulatory/documentation-requirements
# ---------------------------------------------------------------------------


@router.get(
    "/regulatory/documentation-requirements",
    response_model=DocumentationRequirementsResponse,
    summary="Get documentation requirements",
    description=(
        "Retrieve all documentation requirements needed for EUDR compliance, "
        "including mandatory and optional document types with accepted formats."
    ),
    responses={
        200: {"description": "Documentation requirements"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_documentation_requirements(
    request: Request,
    commodity_type: Optional[str] = Depends(validate_commodity_type),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:regulatory:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DocumentationRequirementsResponse:
    """Get documentation requirements for EUDR compliance.

    Args:
        commodity_type: Optional commodity type filter.
        user: Authenticated user with regulatory:read permission.

    Returns:
        DocumentationRequirementsResponse with document requirements.
    """
    reqs = _DOCUMENTATION_REQUIREMENTS.copy()
    ct_enum = CommodityTypeEnum(commodity_type) if commodity_type else None

    mandatory_count = sum(1 for r in reqs if r.mandatory)

    return DocumentationRequirementsResponse(
        commodity_type=ct_enum,
        requirements=reqs,
        total_requirements=len(reqs),
        mandatory_count=mandatory_count,
    )
