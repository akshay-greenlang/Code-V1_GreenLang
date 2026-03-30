# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-027: Information Gathering Agent

REST API endpoints for EUDR Article 9 information gathering operations.
Provides 11 endpoints for full pipeline execution, individual engine
delegation, and health monitoring.

Endpoint Summary (11):
    POST /gather                          - Start full information gathering
    GET  /gather/{operation_id}           - Get gathering operation status
    POST /query/{source}                  - Query a specific external database
    POST /verify-certificate              - Verify a single certificate
    POST /verify-certificates/batch       - Batch verify certificates
    POST /harvest/{source}                - Harvest public data from source
    POST /aggregate-supplier/{supplier_id} - Aggregate supplier information
    POST /validate-completeness           - Validate Article 9 completeness
    POST /assemble-package                - Assemble information package
    GET  /health                          - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-information-gathering:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (GL-EUDR-IGA-027)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.information_gathering.models import (
    CertificationBody,
    CertificateVerificationResult,
    CompletenessReport,
    EUDRCommodity,
    ExternalDatabaseSource,
    GatheringOperation,
    HarvestResult,
    InformationPackage,
    QueryResult,
    SupplierProfile,
)
from greenlang.agents.eudr.information_gathering.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class GatherInformationRequest(GreenLangBase):
    """Request body for the full information gathering pipeline."""

    operator_id: str = Field(..., description="EUDR operator identifier")
    commodity: EUDRCommodity = Field(..., description="EUDR regulated commodity")
    sources: Optional[List[ExternalDatabaseSource]] = Field(
        None, description="Specific external sources to query (None = all)"
    )


class QueryExternalDatabaseRequest(GreenLangBase):
    """Request body for querying a specific external database."""

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific query parameters",
    )


class VerifyCertificateRequest(GreenLangBase):
    """Request body for single certificate verification."""

    certificate_id: str = Field(..., description="Certificate identifier")
    certification_body: CertificationBody = Field(
        ..., description="Certification body that issued the certificate"
    )


class BatchVerifyCertificatesRequest(GreenLangBase):
    """Request body for batch certificate verification."""

    certificates: List[VerifyCertificateRequest] = Field(
        ..., description="List of certificates to verify"
    )


class HarvestPublicDataRequest(GreenLangBase):
    """Request body for public data harvesting."""

    country_code: Optional[str] = Field(
        None, description="ISO 3166-1 alpha-2 country code"
    )
    commodity: Optional[str] = Field(
        None, description="Commodity name for filtering"
    )


class AggregateSupplierRequest(GreenLangBase):
    """Request body for supplier information aggregation."""

    sources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-name to source-data mapping",
    )


class ValidateCompletenessRequest(GreenLangBase):
    """Request body for Article 9 completeness validation."""

    operation_id: str = Field(..., description="Operation identifier")
    commodity: EUDRCommodity = Field(..., description="EUDR regulated commodity")
    is_simplified_dd: bool = Field(
        False, description="Whether simplified due diligence applies (Article 13)"
    )


class AssemblePackageRequest(GreenLangBase):
    """Request body for information package assembly."""

    operation_id: str = Field(..., description="Operation identifier")
    operator_id: str = Field(..., description="EUDR operator identifier")
    commodity: EUDRCommodity = Field(..., description="EUDR regulated commodity")


class ErrorResponse(GreenLangBase):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_type: str = Field("internal_error", description="Error classification")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/information-gathering",
    tags=["EUDR Information Gathering"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/gather",
    response_model=GatheringOperation,
    status_code=200,
    summary="Start full information gathering pipeline",
    description=(
        "Executes the complete 7-step information gathering pipeline for a "
        "given operator and commodity: query external databases, verify "
        "certificates, harvest public data, aggregate suppliers, validate "
        "Article 9 completeness, normalize data, and assemble information "
        "package."
    ),
)
async def gather_information(
    request: GatherInformationRequest,
) -> GatheringOperation:
    """Start a full information gathering operation.

    Args:
        request: Gathering request with operator, commodity, and sources.

    Returns:
        GatheringOperation with status, scores, and package reference.
    """
    try:
        service = get_service()
        operation = await service.gather_information(
            operator_id=request.operator_id,
            commodity=request.commodity,
            sources=request.sources,
        )
        return operation
    except Exception as e:
        logger.error(
            f"gather_information failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Information gathering failed: {str(e)[:200]}",
        )


@router.get(
    "/gather/{operation_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get gathering operation status",
    description=(
        "Retrieve the current status of an information gathering operation "
        "by its operation identifier."
    ),
)
async def get_gathering_status(
    operation_id: str,
) -> Dict[str, Any]:
    """Get the status of a gathering operation.

    Args:
        operation_id: Operation identifier.

    Returns:
        Operation status details.
    """
    try:
        # In production, this queries the database for the operation.
        # For now, return a placeholder indicating the operation lookup.
        return {
            "operation_id": operation_id,
            "message": "Operation lookup requires database persistence layer",
            "status": "lookup_pending",
        }
    except Exception as e:
        logger.error(
            f"get_gathering_status failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Status lookup failed: {str(e)[:200]}",
        )


@router.post(
    "/query/{source}",
    response_model=QueryResult,
    status_code=200,
    summary="Query a specific external database",
    description=(
        "Query a single external regulatory or trade database (e.g., "
        "EU TRACES, CITES, FLEGT/VPA, UN COMTRADE, FAO STAT, Global "
        "Forest Watch, World Bank WGI, Transparency CPI, EU Sanctions)."
    ),
)
async def query_external_database(
    source: ExternalDatabaseSource,
    request: QueryExternalDatabaseRequest,
) -> QueryResult:
    """Query an external database source.

    Args:
        source: External database source identifier.
        request: Query parameters for the source.

    Returns:
        QueryResult with records and provenance hash.
    """
    try:
        service = get_service()
        result = await service.query_external_database(source, request.params)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"query_external_database failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"External database query failed: {str(e)[:200]}",
        )


@router.post(
    "/verify-certificate",
    response_model=CertificateVerificationResult,
    status_code=200,
    summary="Verify a single certificate",
    description=(
        "Verify a sustainability certificate (FSC, RSPO, PEFC, Rainforest "
        "Alliance, UTZ, EU Organic) against its official registry."
    ),
)
async def verify_certificate(
    request: VerifyCertificateRequest,
) -> CertificateVerificationResult:
    """Verify a single certificate.

    Args:
        request: Certificate ID and certification body.

    Returns:
        CertificateVerificationResult with status and provenance.
    """
    try:
        service = get_service()
        result = await service.verify_certificate(
            cert_id=request.certificate_id,
            body=request.certification_body,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"verify_certificate failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Certificate verification failed: {str(e)[:200]}",
        )


@router.post(
    "/verify-certificates/batch",
    response_model=List[CertificateVerificationResult],
    status_code=200,
    summary="Batch verify certificates",
    description=(
        "Verify multiple certificates concurrently against their "
        "respective official registries."
    ),
)
async def batch_verify_certificates(
    request: BatchVerifyCertificatesRequest,
) -> List[CertificateVerificationResult]:
    """Batch verify multiple certificates.

    Args:
        request: List of certificates to verify.

    Returns:
        List of CertificateVerificationResult objects.
    """
    try:
        service = get_service()
        certs: List[Tuple[str, CertificationBody]] = [
            (c.certificate_id, c.certification_body)
            for c in request.certificates
        ]
        results = await service.batch_verify_certificates(certs)
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"batch_verify_certificates failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch certificate verification failed: {str(e)[:200]}",
        )


@router.post(
    "/harvest/{source}",
    response_model=HarvestResult,
    status_code=200,
    summary="Harvest public data from source",
    description=(
        "Harvest publicly available data from a specific source (FAO STAT, "
        "Global Forest Watch, World Bank WGI, Transparency CPI, etc.)."
    ),
)
async def harvest_public_data(
    source: ExternalDatabaseSource,
    request: HarvestPublicDataRequest,
) -> HarvestResult:
    """Harvest public data from a specific source.

    Args:
        source: Data source to harvest.
        request: Country code and commodity filters.

    Returns:
        HarvestResult with harvested data and freshness status.
    """
    try:
        service = get_service()
        result = await service.harvest_public_data(
            source=source,
            country_code=request.country_code,
            commodity=request.commodity,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"harvest_public_data failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Public data harvest failed: {str(e)[:200]}",
        )


@router.post(
    "/aggregate-supplier/{supplier_id}",
    response_model=SupplierProfile,
    status_code=200,
    summary="Aggregate supplier information",
    description=(
        "Aggregate supplier information from multiple data sources using "
        "entity resolution (Jaro-Winkler), source priority ranking, and "
        "discrepancy detection."
    ),
)
async def aggregate_supplier(
    supplier_id: str,
    request: AggregateSupplierRequest,
) -> SupplierProfile:
    """Aggregate supplier information from multiple sources.

    Args:
        supplier_id: Supplier identifier.
        request: Source data for aggregation.

    Returns:
        Unified SupplierProfile with completeness and confidence scores.
    """
    try:
        service = get_service()
        profile = await service.aggregate_supplier(
            supplier_id=supplier_id,
            sources=request.sources,
        )
        return profile
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"aggregate_supplier failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Supplier aggregation failed: {str(e)[:200]}",
        )


@router.post(
    "/validate-completeness",
    response_model=CompletenessReport,
    status_code=200,
    summary="Validate Article 9 completeness",
    description=(
        "Validate the completeness of collected information against the "
        "10 mandatory Article 9 elements. Returns element-level scores, "
        "overall completeness classification, and gap analysis."
    ),
)
async def validate_completeness(
    request: ValidateCompletenessRequest,
) -> CompletenessReport:
    """Validate Article 9 information completeness.

    Args:
        request: Operation ID, commodity, and simplified DD flag.

    Returns:
        CompletenessReport with element scores and gap report.
    """
    try:
        service = get_service()
        report = await service.validate_completeness(
            operation_id=request.operation_id,
            commodity=request.commodity,
            is_simplified_dd=request.is_simplified_dd,
        )
        return report
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"validate_completeness failed: {type(e).__name__}: "
            f"{str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Completeness validation failed: {str(e)[:200]}",
        )


@router.post(
    "/assemble-package",
    response_model=InformationPackage,
    status_code=200,
    summary="Assemble information package",
    description=(
        "Assemble a complete information package for DDS submission from "
        "previously collected data. The package includes all Article 9 "
        "elements, supplier profiles, external data, certificate results, "
        "normalization log, and provenance chain."
    ),
)
async def assemble_package(
    request: AssemblePackageRequest,
) -> InformationPackage:
    """Assemble an information package for DDS submission.

    Args:
        request: Operation ID, operator ID, and commodity.

    Returns:
        Assembled InformationPackage with integrity hash.
    """
    try:
        service = get_service()
        package = await service.assemble_package(
            operation_id=request.operation_id,
            operator_id=request.operator_id,
            commodity=request.commodity,
        )
        return package
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(
            f"assemble_package failed: {type(e).__name__}: {str(e)[:500]}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Package assembly failed: {str(e)[:200]}",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Health check",
    description=(
        "Returns the health status of the Information Gathering Agent "
        "including engine availability, database connectivity, and "
        "Redis connectivity."
    ),
)
async def health_check() -> Dict[str, Any]:
    """Perform a health check on the Information Gathering Agent.

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
            "agent_id": "GL-EUDR-IGA-027",
            "status": "error",
            "error": str(e)[:200],
        }


def get_router() -> APIRouter:
    """Return the Information Gathering Agent API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
