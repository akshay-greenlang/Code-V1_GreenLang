# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-030: Documentation Generator

REST API endpoints for EUDR Due Diligence Statement generation and
documentation management. Provides 12 endpoints for DDS generation,
Article 9 data assembly, risk and mitigation documentation, compliance
package building, DDS validation, regulatory submission, document
version history, and health monitoring.

Endpoint Summary (12):
    POST /generate-dds                           - Generate a new DDS
    GET  /dds/{dds_id}                          - Get DDS details
    GET  /dds                                    - List DDS documents with filters
    POST /assemble-article9/{operator_id}        - Assemble Article 9 package
    POST /document-risk/{assessment_id}          - Document risk assessment
    POST /document-mitigation/{strategy_id}      - Document mitigation measures
    POST /build-package/{dds_id}                 - Build compliance package
    POST /submit/{dds_id}                        - Submit DDS to EU IS
    GET  /submissions/{submission_id}/status      - Get submission status
    POST /validate/{dds_id}                      - Validate DDS completeness
    GET  /versions/{document_id}                 - Get document version history
    GET  /health                                 - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-documentation-generator:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 11, 12, 13, 14-16, 29, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.documentation_generator.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class ProductEntryRequest(GreenLangBase):
    """Product entry for DDS generation."""

    description: str = Field(
        ..., description="Product description"
    )
    quantity: Optional[str] = Field(
        None, description="Quantity with unit (e.g. '1000 kg')"
    )
    hs_code: Optional[str] = Field(
        None, description="Harmonized System commodity code"
    )
    trade_name: Optional[str] = Field(
        None, description="Trade name of the product"
    )
    scientific_name: Optional[str] = Field(
        None, description="Scientific name (for biological products)"
    )


class GeolocationRequest(GreenLangBase):
    """Geolocation reference for DDS generation."""

    latitude: Optional[float] = Field(
        None, description="Latitude coordinate (WGS 84)"
    )
    longitude: Optional[float] = Field(
        None, description="Longitude coordinate (WGS 84)"
    )
    coordinates: Optional[List[List[float]]] = Field(
        None,
        description="Polygon coordinates for plot boundary (GeoJSON format)",
    )
    country_code: str = Field(
        ..., description="ISO 3166-1 alpha-2 country code"
    )
    plot_id: Optional[str] = Field(
        None, description="Plot or parcel identifier"
    )
    region: Optional[str] = Field(
        None, description="Region or administrative area"
    )


class SupplierRequest(GreenLangBase):
    """Supplier reference for DDS generation."""

    supplier_id: str = Field(
        ..., description="Supplier identifier"
    )
    name: str = Field(
        ..., description="Supplier name"
    )
    tier: int = Field(
        default=1, ge=1, le=10, description="Supply chain tier (1=direct)"
    )
    country_code: Optional[str] = Field(
        None, description="Supplier country (ISO 3166-1 alpha-2)"
    )
    certification: Optional[str] = Field(
        None, description="Certification status or reference"
    )


class GenerateDDSRequest(GreenLangBase):
    """Request body for generating a DDS."""

    operator_id: str = Field(
        ..., description="EUDR operator identifier"
    )
    commodity: str = Field(
        ..., description="EUDR regulated commodity (e.g. 'cocoa', 'palm_oil')"
    )
    products: List[ProductEntryRequest] = Field(
        ..., min_length=1, description="Product entries (at least 1)"
    )
    geolocations: List[GeolocationRequest] = Field(
        ..., min_length=1, description="Geolocation references (at least 1)"
    )
    suppliers: List[SupplierRequest] = Field(
        ..., min_length=1, description="Supplier references (at least 1)"
    )
    risk_assessment_id: Optional[str] = Field(
        None, description="Risk assessment identifier from EUDR-028"
    )
    mitigation_strategy_id: Optional[str] = Field(
        None, description="Mitigation strategy identifier from EUDR-029"
    )
    reference_number: Optional[str] = Field(
        None, description="Operator reference number for the DDS"
    )


class AssembleArticle9Request(GreenLangBase):
    """Request body for assembling Article 9 data."""

    commodity: str = Field(
        ..., description="EUDR regulated commodity"
    )
    products: List[ProductEntryRequest] = Field(
        ..., min_length=1, description="Product entries"
    )
    geolocations: List[GeolocationRequest] = Field(
        ..., min_length=1, description="Geolocation references"
    )
    suppliers: List[SupplierRequest] = Field(
        ..., min_length=1, description="Supplier references"
    )
    production_date: Optional[str] = Field(
        None, description="Date of production (ISO 8601)"
    )


class DocumentRiskRequest(GreenLangBase):
    """Request body for documenting a risk assessment."""

    operator_id: str = Field(
        ..., description="EUDR operator identifier"
    )
    commodity: str = Field(
        ..., description="EUDR regulated commodity"
    )
    composite_score: str = Field(
        ..., description="Composite risk score (0-100 as string)"
    )
    risk_level: str = Field(
        ..., description="Risk level (negligible/low/standard/high/critical)"
    )
    contributing_factors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Contributing factors with dimension, score, and weight",
    )
    country_scores: Optional[List[Dict[str, Any]]] = Field(
        None, description="Country-level risk scores"
    )
    supplier_scores: Optional[List[Dict[str, Any]]] = Field(
        None, description="Supplier-level risk scores"
    )


class DocumentMitigationRequest(GreenLangBase):
    """Request body for documenting mitigation measures."""

    operator_id: str = Field(
        ..., description="EUDR operator identifier"
    )
    commodity: str = Field(
        ..., description="EUDR regulated commodity"
    )
    pre_mitigation_score: str = Field(
        ..., description="Risk score before mitigation (0-100 as string)"
    )
    post_mitigation_score: str = Field(
        ..., description="Risk score after mitigation (0-100 as string)"
    )
    measures: Optional[List[Dict[str, Any]]] = Field(
        None, description="Mitigation measure summaries"
    )
    verification_status: Optional[str] = Field(
        None,
        description="Verification status (pending/verified/partial/insufficient)",
    )


class ErrorResponse(GreenLangBase):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_code: str = Field(
        "internal_error", description="Error classification"
    )
    timestamp: Optional[str] = Field(
        None, description="Error timestamp"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/documentation-generator",
    tags=["EUDR Documentation Generator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/generate-dds",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Generate a new Due Diligence Statement",
    description=(
        "Generate a complete Due Diligence Statement per EUDR Article 4(2). "
        "Assembles Article 9 data elements, documents risk assessment "
        "and mitigation measures if provided, and produces a structured DDS "
        "with provenance hashing. The DDS can then be validated and "
        "submitted to the EU Information System."
    ),
)
async def generate_dds(
    request: GenerateDDSRequest,
) -> Dict[str, Any]:
    """Generate a complete Due Diligence Statement.

    Args:
        request: DDS generation request with operator, commodity,
                products, geolocations, and suppliers.

    Returns:
        DDSDocument data with provenance hash.
    """
    try:
        service = get_service()

        # Convert Pydantic models to dictionaries for the service layer
        products = [p.model_dump(mode="json") for p in request.products]
        geolocations = [g.model_dump(mode="json") for g in request.geolocations]
        suppliers = [s.model_dump(mode="json") for s in request.suppliers]

        dds = await service.generate_dds(
            operator_id=request.operator_id,
            commodity=request.commodity,
            products=products,
            geolocations=geolocations,
            suppliers=suppliers,
            risk_assessment_id=request.risk_assessment_id,
            mitigation_strategy_id=request.mitigation_strategy_id,
            reference_number=request.reference_number,
        )

        if isinstance(dds, dict):
            return dds
        return dds.model_dump(mode="json") if hasattr(dds, "model_dump") else dict(dds)

    except ValueError as e:
        logger.warning(
            f"generate_dds validation error: {str(e)[:500]}"
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"generate_dds failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"DDS generation failed: {str(e)[:200]}",
        )


@router.get(
    "/dds/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get DDS details",
    description=(
        "Retrieve the full details of a Due Diligence Statement by its "
        "identifier, including Article 9 data, risk documentation, "
        "mitigation documentation, and provenance data."
    ),
)
async def get_dds(
    dds_id: str,
) -> Dict[str, Any]:
    """Get a DDS by identifier.

    Args:
        dds_id: DDS identifier.

    Returns:
        DDS data dictionary.
    """
    try:
        service = get_service()
        dds = await service.get_dds(dds_id)

        if dds is None:
            raise HTTPException(
                status_code=404,
                detail=f"DDS {dds_id} not found",
            )

        if isinstance(dds, dict):
            return dds
        return dds.model_dump(mode="json") if hasattr(dds, "model_dump") else dict(dds)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"get_dds failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"DDS lookup failed: {str(e)[:200]}",
        )


@router.get(
    "/dds",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="List DDS documents with filters",
    description=(
        "List Due Diligence Statements with optional filters by operator, "
        "commodity, and status. Returns a list of DDS summaries."
    ),
)
async def list_dds(
    operator_id: Optional[str] = Query(
        None, description="Filter by operator ID"
    ),
    commodity: Optional[str] = Query(
        None, description="Filter by EUDR commodity"
    ),
    status: Optional[str] = Query(
        None, description="Filter by DDS status"
    ),
) -> List[Dict[str, Any]]:
    """List DDS documents with optional filtering.

    Args:
        operator_id: Optional operator ID filter.
        commodity: Optional commodity filter.
        status: Optional status filter.

    Returns:
        List of DDS data dictionaries.
    """
    try:
        service = get_service()
        documents = await service.list_dds(
            operator_id=operator_id,
            commodity=commodity,
            status=status,
        )

        results = []
        for d in documents:
            if isinstance(d, dict):
                results.append(d)
            elif hasattr(d, "model_dump"):
                results.append(d.model_dump(mode="json"))
            else:
                results.append(dict(d))

        return results

    except Exception as e:
        logger.error(
            f"list_dds failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"DDS listing failed: {str(e)[:200]}",
        )


@router.post(
    "/assemble-article9/{operator_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Assemble Article 9 data package",
    description=(
        "Assemble all data elements required by EUDR Article 9 including "
        "product descriptions, quantities, country of production, "
        "geolocation coordinates, and supplier chain information. "
        "Returns a completeness assessment."
    ),
)
async def assemble_article9(
    operator_id: str,
    request: AssembleArticle9Request,
) -> Dict[str, Any]:
    """Assemble an Article 9 data package.

    Args:
        operator_id: EUDR operator identifier (path parameter).
        request: Article 9 assembly request with products, geolocations,
                and suppliers.

    Returns:
        Article9Package data with completeness assessment.
    """
    try:
        service = get_service()

        products = [p.model_dump(mode="json") for p in request.products]
        geolocations = [g.model_dump(mode="json") for g in request.geolocations]
        suppliers = [s.model_dump(mode="json") for s in request.suppliers]

        package = await service.assemble_article9(
            operator_id=operator_id,
            commodity=request.commodity,
            products=products,
            geolocations=geolocations,
            suppliers=suppliers,
            production_date=request.production_date,
        )

        if isinstance(package, dict):
            return package
        return package.model_dump(mode="json") if hasattr(package, "model_dump") else dict(package)

    except ValueError as e:
        logger.warning(
            f"assemble_article9 validation error: {str(e)[:500]}"
        )
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"assemble_article9 failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Article 9 assembly failed: {str(e)[:200]}",
        )


@router.post(
    "/document-risk/{assessment_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Document risk assessment",
    description=(
        "Create structured documentation of risk assessment results "
        "for DDS inclusion. Documents composite scores, risk level "
        "classifications, contributing factor breakdowns, and "
        "regulatory cross-references per EUDR Article 10."
    ),
)
async def document_risk(
    assessment_id: str,
    request: DocumentRiskRequest,
) -> Dict[str, Any]:
    """Document a risk assessment for DDS inclusion.

    Args:
        assessment_id: Risk assessment identifier (path parameter).
        request: Risk documentation request with scores and factors.

    Returns:
        RiskAssessmentDoc data with regulatory references.
    """
    try:
        service = get_service()
        doc = await service.document_risk_assessment(
            assessment_id=assessment_id,
            operator_id=request.operator_id,
            commodity=request.commodity,
            composite_score=request.composite_score,
            risk_level=request.risk_level,
            contributing_factors=request.contributing_factors,
            country_scores=request.country_scores,
            supplier_scores=request.supplier_scores,
        )

        if isinstance(doc, dict):
            return doc
        return doc.model_dump(mode="json") if hasattr(doc, "model_dump") else dict(doc)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"document_risk failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Risk documentation failed: {str(e)[:200]}",
        )


@router.post(
    "/document-mitigation/{strategy_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Document mitigation measures",
    description=(
        "Create structured documentation of mitigation measures "
        "for DDS inclusion. Documents before/after risk scores, "
        "measure summaries, effectiveness evidence, and Article 11 "
        "compliance status."
    ),
)
async def document_mitigation(
    strategy_id: str,
    request: DocumentMitigationRequest,
) -> Dict[str, Any]:
    """Document mitigation measures for DDS inclusion.

    Args:
        strategy_id: Mitigation strategy identifier (path parameter).
        request: Mitigation documentation request with scores and measures.

    Returns:
        MitigationDoc data with Article 11 compliance assessment.
    """
    try:
        service = get_service()
        doc = await service.document_mitigation(
            strategy_id=strategy_id,
            operator_id=request.operator_id,
            commodity=request.commodity,
            pre_mitigation_score=request.pre_mitigation_score,
            post_mitigation_score=request.post_mitigation_score,
            measures=request.measures,
            verification_status=request.verification_status,
        )

        if isinstance(doc, dict):
            return doc
        return doc.model_dump(mode="json") if hasattr(doc, "model_dump") else dict(doc)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"document_mitigation failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Mitigation documentation failed: {str(e)[:200]}",
        )


@router.post(
    "/build-package/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Build compliance package",
    description=(
        "Build a complete compliance package combining DDS, Article 9 data, "
        "risk documentation, mitigation documentation, and supporting "
        "evidence into a submission-ready bundle with integrity hashes "
        "for each component."
    ),
)
async def build_compliance_package(
    dds_id: str,
) -> Dict[str, Any]:
    """Build a compliance package for a DDS.

    Args:
        dds_id: DDS identifier.

    Returns:
        CompliancePackage data with integrity hashes.
    """
    try:
        service = get_service()
        package = await service.build_compliance_package(dds_id=dds_id)

        if isinstance(package, dict):
            return package
        return package.model_dump(mode="json") if hasattr(package, "model_dump") else dict(package)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"build_compliance_package failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Compliance package build failed: {str(e)[:200]}",
        )


@router.post(
    "/submit/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Submit DDS to EU Information System",
    description=(
        "Submit a validated DDS to the EU Information System. "
        "The DDS must be in a validated or compliant state. "
        "Returns a submission record with EU IS reference number "
        "and status tracking information."
    ),
)
async def submit_dds(
    dds_id: str,
) -> Dict[str, Any]:
    """Submit a DDS to the EU Information System.

    Args:
        dds_id: DDS identifier.

    Returns:
        SubmissionRecord data with EU IS reference.
    """
    try:
        service = get_service()
        submission = await service.submit_dds(dds_id=dds_id)

        if isinstance(submission, dict):
            return submission
        return submission.model_dump(mode="json") if hasattr(submission, "model_dump") else dict(submission)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(
            f"submit_dds failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"DDS submission failed: {str(e)[:200]}",
        )


@router.get(
    "/submissions/{submission_id}/status",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get submission status",
    description=(
        "Retrieve the current status of a DDS submission to the EU "
        "Information System, including receipt confirmation, retry count, "
        "and deadline information."
    ),
)
async def get_submission_status(
    submission_id: str,
) -> Dict[str, Any]:
    """Get the current status of a DDS submission.

    Args:
        submission_id: Submission identifier.

    Returns:
        SubmissionRecord data.
    """
    try:
        service = get_service()
        status = await service.get_submission_status(
            submission_id=submission_id,
        )

        if isinstance(status, dict):
            return status
        return status.model_dump(mode="json") if hasattr(status, "model_dump") else dict(status)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"get_submission_status failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Submission status lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/validate/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Validate DDS completeness",
    description=(
        "Validate a DDS for completeness and regulatory compliance. "
        "Checks all required Article 9 elements, risk assessment "
        "documentation, mitigation documentation where required, "
        "and structural integrity. Returns pass/fail with detailed "
        "issue list."
    ),
)
async def validate_dds(
    dds_id: str,
) -> Dict[str, Any]:
    """Validate DDS completeness and compliance.

    Args:
        dds_id: DDS identifier.

    Returns:
        ValidationResult with issues and pass/fail status.
    """
    try:
        service = get_service()
        result = await service.validate_dds(dds_id=dds_id)

        if isinstance(result, dict):
            return result
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"validate_dds failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"DDS validation failed: {str(e)[:200]}",
        )


@router.get(
    "/versions/{document_id}",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="Get document version history",
    description=(
        "Retrieve the version history for a document including all "
        "recorded versions with timestamps, actors, and change summaries. "
        "Supports EUDR Article 12(3) five-year retention requirement."
    ),
)
async def get_version_history(
    document_id: str,
) -> List[Dict[str, Any]]:
    """Get the version history for a document.

    Args:
        document_id: Document identifier.

    Returns:
        List of version records ordered by version number.
    """
    try:
        service = get_service()
        versions = await service.get_version_history(
            document_id=document_id,
        )

        results = []
        for v in versions:
            if isinstance(v, dict):
                results.append(v)
            elif hasattr(v, "model_dump"):
                results.append(v.model_dump(mode="json"))
            else:
                results.append(dict(v))

        return results

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            f"get_version_history failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Version history lookup failed: {str(e)[:200]}",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Health check",
    description=(
        "Returns the health status of the Documentation Generator "
        "including engine availability, database connectivity, Redis "
        "connectivity, and in-memory store statistics."
    ),
)
async def health_check() -> Dict[str, Any]:
    """Perform a health check on the Documentation Generator.

    Returns:
        Dictionary with component health statuses.
    """
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error(
            f"health_check failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        return {
            "agent_id": "GL-EUDR-DGN-030",
            "status": "error",
            "error": str(e)[:200],
        }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the Documentation Generator API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
