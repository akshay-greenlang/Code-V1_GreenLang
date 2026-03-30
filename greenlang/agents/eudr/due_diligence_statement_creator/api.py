# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-037: Due Diligence Statement Creator

REST API endpoints for EUDR Due Diligence Statement (DDS) lifecycle
management. Provides 30+ endpoints for DDS creation, assembly, validation,
geolocation formatting, risk data integration, supply chain compilation,
compliance checking, document packaging, digital signing, version control,
amendment handling, and EU Information System submission.

Endpoint Summary (30+):
    POST /create-dds                                   - Create new DDS
    POST /assemble-dds                                 - Assemble DDS from data
    GET  /dds                                          - List DDS records
    GET  /dds/{statement_id}                           - Get DDS by ID
    GET  /dds/{statement_id}/summary                   - Get DDS summary
    PUT  /dds/{statement_id}/status                    - Update DDS status
    DELETE /dds/{statement_id}                          - Withdraw DDS
    POST /dds/{statement_id}/geolocations              - Format geolocations
    POST /dds/{statement_id}/geolocations/batch        - Batch format geolocations
    GET  /dds/{statement_id}/geolocations/geojson      - Export as GeoJSON
    POST /dds/{statement_id}/risk-references           - Integrate risk data
    POST /dds/{statement_id}/risk-references/batch     - Batch integrate risk data
    GET  /dds/{statement_id}/risk-references/overall   - Get overall risk level
    POST /dds/{statement_id}/supply-chain              - Compile supply chain
    GET  /dds/{statement_id}/supply-chain/completeness - Validate supply chain
    GET  /dds/{statement_id}/supply-chain/countries    - Get country summary
    POST /dds/{statement_id}/validate                  - Validate DDS compliance
    GET  /dds/{statement_id}/compliance                - Get compliance report
    POST /dds/{statement_id}/documents                 - Add document to package
    POST /dds/{statement_id}/package                   - Create submission package
    GET  /dds/{statement_id}/package/validate          - Validate package
    GET  /dds/{statement_id}/package/manifest          - Get document manifest
    POST /dds/{statement_id}/sign                      - Apply digital signature
    GET  /dds/{statement_id}/signature/validate        - Validate signature
    POST /dds/{statement_id}/amend                     - Create amendment
    GET  /dds/{statement_id}/versions                  - Get version history
    GET  /dds/{statement_id}/versions/latest           - Get latest version
    GET  /dds/{statement_id}/amendments                - Get amendment records
    POST /dds/{statement_id}/submit                    - Submit to EU IS
    GET  /health                                       - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-dds-creator:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 12, 13, 14, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import Field

from greenlang.agents.eudr.due_diligence_statement_creator.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class CreateDDSRequest(GreenLangBase):
    """Request body for creating a new DDS."""
    operator_id: str = Field(..., description="EUDR operator identifier")
    operator_name: str = Field(..., description="Operator legal name")
    operator_address: str = Field(default="", description="Operator address")
    operator_eori_number: str = Field(default="", description="EORI number")
    statement_type: str = Field(default="placing", description="DDS type")
    commodities: List[str] = Field(..., description="Commodity types")
    language: str = Field(default="en", description="Primary language")


class AssembleDDSRequest(GreenLangBase):
    """Request body for assembling a DDS with full data."""
    operator_id: str = Field(..., description="EUDR operator identifier")
    operator_name: str = Field(..., description="Operator legal name")
    operator_address: str = Field(default="", description="Operator address")
    operator_eori_number: str = Field(default="", description="EORI number")
    statement_type: str = Field(default="placing", description="DDS type")
    commodities: List[str] = Field(..., description="Commodity types")
    product_descriptions: Optional[List[Dict[str, Any]]] = Field(None, description="Products")
    hs_codes: Optional[List[str]] = Field(None, description="HS codes")
    total_quantity: float = Field(default=0.0, description="Total quantity")
    quantity_unit: str = Field(default="metric_tonnes", description="Unit")
    countries_of_production: Optional[List[str]] = Field(None, description="Countries")
    compliance_declaration: str = Field(default="", description="Compliance declaration")
    deforestation_free: bool = Field(default=False, description="Deforestation-free")
    legally_produced: bool = Field(default=False, description="Legally produced")
    language: str = Field(default="en", description="Primary language")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class UpdateStatusRequest(GreenLangBase):
    """Request body for updating DDS status."""
    status: str = Field(..., description="New DDS status")


class FormatGeolocationRequest(GreenLangBase):
    """Request body for formatting geolocation data."""
    plot_id: str = Field(..., description="Plot identifier")
    latitude: float = Field(..., description="Latitude (WGS84)")
    longitude: float = Field(..., description="Longitude (WGS84)")
    area_hectares: float = Field(default=0.0, description="Area in hectares")
    polygon_coordinates: Optional[List[List[float]]] = Field(None, description="Polygon")
    country_code: str = Field(default="", description="ISO country code")
    collection_method: str = Field(default="gps_field_survey", description="Method")


class IntegrateRiskRequest(GreenLangBase):
    """Request body for integrating risk data."""
    risk_id: str = Field(..., description="Risk assessment identifier")
    source_agent: str = Field(..., description="Source EUDR agent ID")
    risk_category: str = Field(..., description="Risk category")
    risk_level: str = Field(default="standard", description="Risk level")
    risk_score: float = Field(default=0.0, description="Risk score (0-100)")
    factors: Optional[List[str]] = Field(None, description="Risk factors")
    mitigation_measures: Optional[List[str]] = Field(None, description="Mitigations")
    data_sources: Optional[List[str]] = Field(None, description="Data sources")


class CompileSupplyChainRequest(GreenLangBase):
    """Request body for compiling supply chain data."""
    supply_chain_id: str = Field(..., description="Supply chain identifier")
    commodity: str = Field(..., description="Commodity type")
    suppliers: Optional[List[Dict[str, Any]]] = Field(None, description="Suppliers")
    countries_of_production: Optional[List[str]] = Field(None, description="Countries")
    chain_of_custody_model: str = Field(default="segregation", description="CoC model")
    traceability_score: float = Field(default=0.0, description="Traceability score")


class AddDocumentRequest(GreenLangBase):
    """Request body for adding a document to the package."""
    document_type: str = Field(..., description="Document type")
    filename: str = Field(..., description="Filename")
    size_bytes: int = Field(default=0, description="File size in bytes")
    mime_type: str = Field(default="application/pdf", description="MIME type")
    hash_sha256: str = Field(default="", description="SHA-256 hash")
    description: str = Field(default="", description="Document description")
    issuing_authority: str = Field(default="", description="Issuing authority")
    language: str = Field(default="en", description="Document language")


class ApplySignatureRequest(GreenLangBase):
    """Request body for applying a digital signature."""
    signer_name: str = Field(..., description="Signer name")
    signer_role: str = Field(default="", description="Signer role")
    signer_organization: str = Field(default="", description="Signer organization")
    signature_type: str = Field(default="qualified_electronic", description="Signature type")
    signed_hash: str = Field(default="", description="SHA-256 hash of signed content")


class CreateAmendmentRequest(GreenLangBase):
    """Request body for creating a DDS amendment."""
    reason: str = Field(..., description="Amendment reason")
    description: str = Field(..., description="Description of changes")
    previous_version: int = Field(..., description="Version being amended")
    changed_fields: Optional[List[str]] = Field(None, description="Changed fields")
    changed_by: str = Field(default="", description="User making the amendment")
    approved_by: str = Field(default="", description="User approving the amendment")


class SubmitDDSRequest(GreenLangBase):
    """Request body for submitting a DDS to the EU IS."""
    additional_documents: Optional[List[Dict[str, Any]]] = Field(None, description="Extra docs")


class ErrorResponse(GreenLangBase):
    """Standard error response body."""
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error", description="Error classification")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/dds-creator",
    tags=["EUDR Due Diligence Statement Creator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# DDS Core Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/create-dds",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Create a new Due Diligence Statement",
    description="Create a new DDS record in draft status with operator information and commodity types.",
)
async def create_dds(request: CreateDDSRequest) -> Dict[str, Any]:
    """Create a new DDS record."""
    try:
        service = get_service()
        result = await service.create_statement(
            operator_id=request.operator_id,
            operator_name=request.operator_name,
            commodities=request.commodities,
            statement_type=request.statement_type,
            operator_address=request.operator_address,
            operator_eori_number=request.operator_eori_number,
            language=request.language,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"create_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post(
    "/assemble-dds",
    response_model=Dict[str, Any],
    summary="Assemble a complete DDS from data",
    description="Assemble a DDS with all Article 4 mandatory data in a single request.",
)
async def assemble_dds(request: AssembleDDSRequest) -> Dict[str, Any]:
    """Assemble a complete DDS from provided data."""
    try:
        service = get_service()
        result = await service.assemble_statement(
            operator_id=request.operator_id,
            operator_name=request.operator_name,
            commodities=request.commodities,
            statement_type=request.statement_type,
            operator_address=request.operator_address,
            operator_eori_number=request.operator_eori_number,
            product_descriptions=request.product_descriptions,
            hs_codes=request.hs_codes,
            total_quantity=request.total_quantity,
            quantity_unit=request.quantity_unit,
            countries_of_production=request.countries_of_production,
            compliance_declaration=request.compliance_declaration,
            deforestation_free=request.deforestation_free,
            legally_produced=request.legally_produced,
            language=request.language,
            metadata=request.metadata,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"assemble_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/dds", response_model=List[Dict[str, Any]], summary="List DDS records")
async def list_dds(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List DDS records with optional filters."""
    try:
        service = get_service()
        results = await service.list_statements(
            operator_id=operator_id, status=status, commodity=commodity,
        )
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error(f"list_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/dds/{statement_id}", response_model=Dict[str, Any], summary="Get DDS by ID")
async def get_dds(statement_id: str) -> Dict[str, Any]:
    """Get a DDS by its identifier."""
    try:
        service = get_service()
        result = await service.get_statement(statement_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"DDS {statement_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/dds/{statement_id}/summary", response_model=Dict[str, Any], summary="Get DDS summary")
async def get_dds_summary(statement_id: str) -> Dict[str, Any]:
    """Get a lightweight summary of a DDS."""
    try:
        service = get_service()
        result = await service.get_statement_summary(statement_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"DDS {statement_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_dds_summary failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.put("/dds/{statement_id}/status", response_model=Dict[str, Any], summary="Update DDS status")
async def update_dds_status(statement_id: str, request: UpdateStatusRequest) -> Dict[str, Any]:
    """Update the status of a DDS."""
    try:
        service = get_service()
        result = await service.update_statement_status(statement_id, request.status)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"update_dds_status failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.delete("/dds/{statement_id}", response_model=Dict[str, Any], summary="Withdraw DDS")
async def withdraw_dds(
    statement_id: str,
    reason: str = Query(default="", description="Withdrawal reason"),
) -> Dict[str, Any]:
    """Withdraw a DDS from the system."""
    try:
        service = get_service()
        result = await service.withdraw_statement(statement_id, reason=reason)
        return result if isinstance(result, dict) else {"status": "withdrawn", "statement_id": statement_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"withdraw_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Geolocation Endpoints
# ---------------------------------------------------------------------------


@router.post("/dds/{statement_id}/geolocations", response_model=Dict[str, Any], summary="Format geolocation")
async def format_geolocation(statement_id: str, request: FormatGeolocationRequest) -> Dict[str, Any]:
    """Format geolocation data per Article 9 requirements."""
    try:
        service = get_service()
        result = await service.format_geolocation(
            statement_id=statement_id,
            plot_id=request.plot_id,
            latitude=request.latitude,
            longitude=request.longitude,
            area_hectares=request.area_hectares,
            polygon_coordinates=request.polygon_coordinates,
            country_code=request.country_code,
            collection_method=request.collection_method,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"format_geolocation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post(
    "/dds/{statement_id}/geolocations/batch",
    response_model=List[Dict[str, Any]],
    summary="Batch format geolocations",
)
async def batch_format_geolocations(
    statement_id: str,
    geolocations: List[FormatGeolocationRequest],
) -> List[Dict[str, Any]]:
    """Format multiple geolocation records in batch."""
    try:
        service = get_service()
        results = await service.format_geolocations_batch(
            statement_id=statement_id,
            geolocations=[g.model_dump(mode="json") for g in geolocations],
        )
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error(f"batch_format_geolocations failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/geolocations/geojson",
    response_model=Dict[str, Any],
    summary="Export GeoJSON",
)
async def export_geojson(statement_id: str) -> Dict[str, Any]:
    """Export geolocation data as a GeoJSON FeatureCollection."""
    try:
        service = get_service()
        return await service.export_geojson(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"export_geojson failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Risk Data Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/risk-references",
    response_model=Dict[str, Any],
    summary="Integrate risk data",
)
async def integrate_risk(statement_id: str, request: IntegrateRiskRequest) -> Dict[str, Any]:
    """Integrate a risk assessment reference into the DDS."""
    try:
        service = get_service()
        result = await service.integrate_risk(
            statement_id=statement_id,
            risk_id=request.risk_id,
            source_agent=request.source_agent,
            risk_category=request.risk_category,
            risk_level=request.risk_level,
            risk_score=request.risk_score,
            factors=request.factors,
            mitigation_measures=request.mitigation_measures,
            data_sources=request.data_sources,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"integrate_risk failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post(
    "/dds/{statement_id}/risk-references/batch",
    response_model=List[Dict[str, Any]],
    summary="Batch integrate risk data",
)
async def batch_integrate_risk(
    statement_id: str,
    risk_data: List[IntegrateRiskRequest],
) -> List[Dict[str, Any]]:
    """Integrate multiple risk assessment references in batch."""
    try:
        service = get_service()
        results = await service.integrate_risk_batch(
            statement_id=statement_id,
            risk_data=[r.model_dump(mode="json") for r in risk_data],
        )
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error(f"batch_integrate_risk failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/risk-references/overall",
    response_model=Dict[str, Any],
    summary="Get overall risk level",
)
async def get_overall_risk(statement_id: str) -> Dict[str, Any]:
    """Compute the overall risk level from all references."""
    try:
        service = get_service()
        return await service.get_overall_risk(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"get_overall_risk failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Supply Chain Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/supply-chain",
    response_model=Dict[str, Any],
    summary="Compile supply chain",
)
async def compile_supply_chain(
    statement_id: str, request: CompileSupplyChainRequest,
) -> Dict[str, Any]:
    """Compile supply chain data from upstream agents."""
    try:
        service = get_service()
        result = await service.compile_supply_chain(
            statement_id=statement_id,
            supply_chain_id=request.supply_chain_id,
            commodity=request.commodity,
            suppliers=request.suppliers,
            countries_of_production=request.countries_of_production,
            chain_of_custody_model=request.chain_of_custody_model,
            traceability_score=request.traceability_score,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"compile_supply_chain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/supply-chain/completeness",
    response_model=Dict[str, Any],
    summary="Validate supply chain completeness",
)
async def validate_supply_chain(statement_id: str) -> Dict[str, Any]:
    """Validate the completeness of supply chain data."""
    try:
        service = get_service()
        return await service.validate_supply_chain(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"validate_supply_chain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/supply-chain/countries",
    response_model=Dict[str, int],
    summary="Get country summary",
)
async def get_countries_summary(statement_id: str) -> Dict[str, int]:
    """Get a summary of supplier counts by country."""
    try:
        service = get_service()
        return await service.get_supply_chain_countries(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"get_countries_summary failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Compliance Validation Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/validate",
    response_model=Dict[str, Any],
    summary="Validate DDS compliance",
)
async def validate_dds(statement_id: str) -> Dict[str, Any]:
    """Validate a DDS against all Article 4 requirements."""
    try:
        service = get_service()
        result = await service.validate_statement(statement_id)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"validate_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/compliance",
    response_model=Dict[str, Any],
    summary="Get compliance report",
)
async def get_compliance_report(statement_id: str) -> Dict[str, Any]:
    """Get the compliance validation report for a DDS."""
    try:
        service = get_service()
        result = await service.get_compliance_report(statement_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No compliance report for DDS {statement_id}",
            )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_compliance_report failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Document Packaging Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/documents",
    response_model=Dict[str, Any],
    summary="Add document to package",
)
async def add_document(statement_id: str, request: AddDocumentRequest) -> Dict[str, Any]:
    """Add a supporting document to the DDS evidence package."""
    try:
        service = get_service()
        result = await service.add_document(
            statement_id=statement_id,
            document_type=request.document_type,
            filename=request.filename,
            size_bytes=request.size_bytes,
            mime_type=request.mime_type,
            hash_sha256=request.hash_sha256,
            description=request.description,
            issuing_authority=request.issuing_authority,
            language=request.language,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"add_document failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post(
    "/dds/{statement_id}/package",
    response_model=Dict[str, Any],
    summary="Create submission package",
)
async def create_package(statement_id: str) -> Dict[str, Any]:
    """Create a submission package for the EU Information System."""
    try:
        service = get_service()
        result = await service.create_submission_package(statement_id)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"create_package failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/package/validate",
    response_model=Dict[str, Any],
    summary="Validate package",
)
async def validate_package(statement_id: str) -> Dict[str, Any]:
    """Validate the submission package before upload."""
    try:
        service = get_service()
        return await service.validate_package(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"validate_package failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/package/manifest",
    response_model=Dict[str, Any],
    summary="Get document manifest",
)
async def get_manifest(statement_id: str) -> Dict[str, Any]:
    """Get the document manifest for the submission package."""
    try:
        service = get_service()
        return await service.get_package_manifest(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"get_manifest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Digital Signature Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/sign",
    response_model=Dict[str, Any],
    summary="Apply digital signature",
)
async def apply_signature(
    statement_id: str, request: ApplySignatureRequest,
) -> Dict[str, Any]:
    """Apply a digital signature to a DDS per eIDAS Regulation."""
    try:
        service = get_service()
        result = await service.apply_signature(
            statement_id=statement_id,
            signer_name=request.signer_name,
            signer_role=request.signer_role,
            signer_organization=request.signer_organization,
            signature_type=request.signature_type,
            signed_hash=request.signed_hash,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"apply_signature failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/signature/validate",
    response_model=Dict[str, Any],
    summary="Validate signature",
)
async def validate_signature(statement_id: str) -> Dict[str, Any]:
    """Validate the digital signature on a DDS."""
    try:
        service = get_service()
        return await service.validate_signature(statement_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"validate_signature failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Amendment / Version Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/amend",
    response_model=Dict[str, Any],
    summary="Create amendment",
)
async def create_amendment(
    statement_id: str, request: CreateAmendmentRequest,
) -> Dict[str, Any]:
    """Create an amendment to a submitted DDS."""
    try:
        service = get_service()
        result = await service.create_amendment(
            statement_id=statement_id,
            reason=request.reason,
            description=request.description,
            previous_version=request.previous_version,
            changed_fields=request.changed_fields,
            changed_by=request.changed_by,
            approved_by=request.approved_by,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"create_amendment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/versions",
    response_model=List[Dict[str, Any]],
    summary="Get version history",
)
async def get_versions(statement_id: str) -> List[Dict[str, Any]]:
    """Get the complete version history for a DDS."""
    try:
        service = get_service()
        results = await service.get_versions(statement_id)
        return [
            r.model_dump(mode="json") if hasattr(r, "model_dump") else r
            for r in results
        ]
    except Exception as e:
        logger.error(f"get_versions failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/versions/latest",
    response_model=Dict[str, Any],
    summary="Get latest version",
)
async def get_latest_version(statement_id: str) -> Dict[str, Any]:
    """Get the latest version of a DDS."""
    try:
        service = get_service()
        result = await service.get_latest_version(statement_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No versions for DDS {statement_id}",
            )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_latest_version failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get(
    "/dds/{statement_id}/amendments",
    response_model=List[Dict[str, Any]],
    summary="Get amendment records",
)
async def get_amendments(statement_id: str) -> List[Dict[str, Any]]:
    """Get all amendment records for a DDS."""
    try:
        service = get_service()
        results = await service.get_amendments(statement_id)
        return [
            r.model_dump(mode="json") if hasattr(r, "model_dump") else r
            for r in results
        ]
    except Exception as e:
        logger.error(f"get_amendments failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Submission Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/dds/{statement_id}/submit",
    response_model=Dict[str, Any],
    summary="Submit DDS to EU IS",
)
async def submit_dds(
    statement_id: str, request: Optional[SubmitDDSRequest] = None,
) -> Dict[str, Any]:
    """Submit a DDS to the EU Information System."""
    try:
        service = get_service()
        result = await service.submit_statement(
            statement_id=statement_id,
            additional_documents=(
                request.additional_documents if request else None
            ),
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"submit_dds failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------


@router.get("/health", response_model=Dict[str, Any], summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Perform a health check on the Due Diligence Statement Creator."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error(f"health_check failed: {e}", exc_info=True)
        return {
            "agent_id": "GL-EUDR-DDSC-037",
            "status": "error",
            "error": str(e)[:200],
        }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the Due Diligence Statement Creator API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
