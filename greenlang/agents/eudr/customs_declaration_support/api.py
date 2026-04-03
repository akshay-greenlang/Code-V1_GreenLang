# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-039: Customs Declaration Support

REST API endpoints for EUDR customs declaration lifecycle management.
Provides 28+ endpoints for declaration creation, SAD form generation,
CN/HS code operations, origin verification, customs value calculation,
EUDR compliance checking, customs submission, and health monitoring.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 (GL-EUDR-CDS-039)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 12, 31; EU UCC 952/2013
Status: Production Ready
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.customs_declaration_support.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class CreateDeclarationRequest(GreenLangBase):
    """Request body for creating a customs declaration."""
    operator_id: str = Field(..., description="EUDR operator identifier")
    operator_name: str = Field(default="", description="EUDR operator name")
    operator_eori: str = Field(default="", description="EORI number")
    commodities: List[str] = Field(default_factory=list, description="List of commodities")
    country_of_origin: str = Field(default="", description="Country of origin")
    declaration_type: str = Field(default="import", description="Declaration type")
    purpose: str = Field(default="free_circulation", description="Declaration purpose")
    customs_system: str = Field(default="AIS", description="Target customs system")
    incoterms: str = Field(default="CIF", description="Incoterms")
    port_of_entry: str = Field(default="", description="Port of entry code")
    currency: str = Field(default="EUR", description="Currency")
    dds_reference_numbers: Optional[List[str]] = Field(
        None, description="DDS reference numbers",
    )


class MapCNCodesRequest(GreenLangBase):
    """Request body for mapping a commodity to CN codes."""
    commodity: str = Field(..., description="EUDR commodity type")
    product_description: str = Field(default="", description="Product description")


class ValidateHSCodeRequest(GreenLangBase):
    """Request body for validating an HS code."""
    hs_code: str = Field(..., min_length=6, max_length=6, description="6-digit HS code")


class CalculateTariffRequest(GreenLangBase):
    """Request body for calculating tariff."""
    cn_code: str = Field(..., description="CN code")
    customs_value: float = Field(default=0, description="Customs value")
    quantity: float = Field(default=0, description="Quantity")
    origin_country: str = Field(default="", description="Country of origin")
    currency: str = Field(default="EUR", description="Currency")
    declaration_id: str = Field(default="", description="Declaration ID")


class VerifyOriginRequest(GreenLangBase):
    """Request body for origin verification."""
    declared_origin: str = Field(..., description="Declared country of origin")
    supply_chain_origins: List[str] = Field(default_factory=list, description="Supply chain origins")
    dds_reference: str = Field(default="", description="DDS reference")
    supply_chain_data: Optional[Dict[str, Any]] = Field(
        None, description="Supply chain traceability data (legacy)",
    )
    certificate_ref: str = Field(default="", description="Certificate of origin ref")


class RunComplianceCheckRequest(GreenLangBase):
    """Request body for running compliance checks."""
    dds_reference: str = Field(default="", description="DDS reference")
    cn_codes: List[str] = Field(default_factory=list, description="CN codes")
    declared_origin: str = Field(default="", description="Declared origin")
    supply_chain_origins: List[str] = Field(default_factory=list, description="Supply chain origins")
    deforestation_free: bool = Field(default=False, description="Deforestation-free")
    risk_level: str = Field(default="standard", description="Risk level")
    country_code: str = Field(default="", description="Country code")


class GenerateSADFormRequest(GreenLangBase):
    """Request body for generating a SAD form."""
    consignor_name: str = Field(default="", description="Consignor name")
    consignor_eori: str = Field(default="", description="Consignor EORI")
    consignee_name: str = Field(default="", description="Consignee name")
    consignee_eori: str = Field(default="", description="Consignee EORI")


class SubmitDeclarationRequest(GreenLangBase):
    """Request body for submitting a declaration."""
    system: str = Field(..., description="Target customs system (ncts/ais)")


class UpdateStatusRequest(GreenLangBase):
    """Request body for updating declaration status."""
    status: str = Field(..., description="New status")
    mrn: Optional[str] = Field(None, description="MRN if assigned")
    notes: str = Field(default="", description="Update notes")


class CalculateValueRequest(GreenLangBase):
    """Request body for customs value calculation."""
    fob_value: float = Field(default=0, description="FOB value")
    freight_cost: float = Field(default=0, description="Freight cost")
    insurance_cost: float = Field(default=0, description="Insurance cost")
    incoterms: str = Field(default="CIF", description="Incoterms basis")
    transaction_value: float = Field(default=0, gt=-1, description="Transaction value (legacy)")
    currency: str = Field(default="EUR", description="Original currency")


class ConvertCurrencyRequest(GreenLangBase):
    """Request body for currency conversion."""
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Source currency")
    to_currency: str = Field(default="EUR", description="Target currency")


class ErrorResponse(GreenLangBase):
    """Standard error response body."""
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error", description="Error classification")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/customs-declaration",
    tags=["EUDR Customs Declaration Support"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Declaration CRUD Endpoints
# ---------------------------------------------------------------------------

@router.post("/declarations", summary="Create customs declaration")
async def create_declaration(request: CreateDeclarationRequest) -> Any:
    """Create a new customs declaration."""
    try:
        service = get_service()
        result = await service.create_declaration(
            operator_id=request.operator_id,
            declaration_data=request.model_dump(mode="json"),
            commodities=request.commodities,
            country_of_origin=request.country_of_origin,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("create_declaration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/declarations", summary="List declarations")
async def list_declarations(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> Any:
    """List declarations with optional filters."""
    try:
        service = get_service()
        results = await service.list_declarations(
            operator_id=operator_id, status=status,
        )
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("list_declarations failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/declarations/{declaration_id}", summary="Get declaration")
async def get_declaration(declaration_id: str) -> Any:
    """Get declaration by identifier."""
    try:
        service = get_service()
        result = await service.get_declaration(declaration_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Declaration {declaration_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_declaration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.put("/declarations/{declaration_id}/status", summary="Update status")
async def update_declaration_status(
    declaration_id: str, request: UpdateStatusRequest,
) -> Any:
    """Update declaration status."""
    try:
        service = get_service()
        result = await service.update_declaration_status(declaration_id, request.status)
        return result if isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("update_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.delete("/declarations/{declaration_id}", summary="Cancel declaration")
async def cancel_declaration(declaration_id: str) -> Any:
    """Cancel a customs declaration."""
    try:
        service = get_service()
        return await service.cancel_declaration(declaration_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("cancel_declaration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# CN Code Endpoints
# ---------------------------------------------------------------------------

@router.post("/cn-codes/map", summary="Map commodity to CN codes")
async def map_cn_codes(request: MapCNCodesRequest) -> Any:
    """Map an EUDR commodity to its CN codes."""
    try:
        service = get_service()
        results = await service.map_cn_codes(commodity=request.commodity)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("map_cn_codes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/cn-codes/{cn_code}", summary="Get CN code details")
async def get_cn_code(cn_code: str) -> Any:
    """Get detailed information for a specific CN code."""
    try:
        service = get_service()
        result = await service.lookup_cn_code(cn_code)
        if result is None:
            raise HTTPException(status_code=404, detail=f"CN code {cn_code} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_cn_code failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# HS Code Endpoints
# ---------------------------------------------------------------------------

@router.post("/hs-codes/validate", summary="Validate HS code")
async def validate_hs_code(request: ValidateHSCodeRequest) -> Any:
    """Validate a 6-digit HS code."""
    try:
        service = get_service()
        result = await service.validate_hs_code(request.hs_code)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("validate_hs_code failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/hs-codes/validate/batch", summary="Batch validate HS codes")
async def batch_validate_hs_codes(hs_codes: List[str]) -> Any:
    """Batch validate multiple HS codes."""
    try:
        service = get_service()
        results = await service.validate_hs_codes_batch(hs_codes)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("batch_validate_hs_codes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Tariff / Value Endpoints
# ---------------------------------------------------------------------------

@router.post("/declarations/{declaration_id}/tariff", summary="Calculate tariff")
async def calculate_tariff(
    declaration_id: str, request: CalculateTariffRequest,
) -> Any:
    """Calculate tariff for a declaration."""
    try:
        service = get_service()
        result = await service.calculate_tariff(
            declaration_id=declaration_id,
            cn_code=request.cn_code,
            customs_value=request.customs_value,
            quantity=request.quantity,
            origin_country=request.origin_country,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("calculate_tariff failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/declarations/{declaration_id}/customs-value", summary="Calculate customs value")
async def calculate_customs_value(
    declaration_id: str, request: CalculateValueRequest,
) -> Any:
    """Calculate customs value with CIF/FOB breakdown."""
    try:
        service = get_service()
        result = await service.calculate_customs_value(
            transaction_value=Decimal(str(request.fob_value or request.transaction_value)),
            currency=request.currency,
            incoterms=request.incoterms,
            freight_cost=Decimal(str(request.freight_cost)) if request.freight_cost else None,
            insurance_cost=Decimal(str(request.insurance_cost)) if request.insurance_cost else None,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("calculate_customs_value failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/declarations/{declaration_id}/tariff/summary", summary="Get tariff summary")
async def get_tariff_summary(declaration_id: str) -> Any:
    """Get tariff summary for a declaration."""
    try:
        service = get_service()
        result = await service.get_tariff_summary(declaration_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Tariff summary not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_tariff_summary failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/currency/convert", summary="Convert currency")
async def convert_currency(request: ConvertCurrencyRequest) -> Any:
    """Convert amount between currencies."""
    try:
        service = get_service()
        result = await service.convert_currency(
            amount=Decimal(str(request.amount)),
            from_currency=request.from_currency,
            to_currency=request.to_currency,
        )
        # Handle both Decimal and dict returns (mock vs real)
        if isinstance(result, dict):
            return result
        return {
            "amount": float(request.amount),
            "from_currency": request.from_currency,
            "to_currency": request.to_currency,
            "converted_amount": float(result),
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("convert_currency failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Origin Verification Endpoints
# ---------------------------------------------------------------------------

@router.post("/declarations/{declaration_id}/origin/verify", summary="Verify origin")
async def verify_origin(declaration_id: str, request: VerifyOriginRequest) -> Any:
    """Verify country of origin against supply chain data."""
    try:
        service = get_service()
        result = await service.verify_origin(
            declared_origin=request.declared_origin,
            declaration_id=declaration_id,
            supply_chain_origins=request.supply_chain_origins,
            dds_reference=request.dds_reference,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("verify_origin failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/declarations/{declaration_id}/origin/verify/batch", summary="Batch verify origins")
async def batch_verify_origins(declaration_id: str, origins: List[Dict[str, Any]]) -> Any:
    """Batch verify multiple country origins."""
    try:
        service = get_service()
        results = await service.verify_origin_batch(origins)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("batch_verify_origins failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Compliance Endpoints
# ---------------------------------------------------------------------------

@router.post("/declarations/{declaration_id}/compliance", summary="Run compliance checks")
async def run_compliance_check(
    declaration_id: str, request: RunComplianceCheckRequest,
) -> Any:
    """Run EUDR compliance checks for a declaration."""
    try:
        service = get_service()
        result = await service.run_compliance_check(
            declaration_id=declaration_id,
            dds_reference=request.dds_reference,
            cn_codes=request.cn_codes,
            declared_origin=request.declared_origin,
            supply_chain_origins=request.supply_chain_origins,
            deforestation_free=request.deforestation_free,
            risk_level=request.risk_level,
            country_code=request.country_code,
        )
        return result if isinstance(result, dict) else result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("run_compliance_check failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/declarations/{declaration_id}/compliance", summary="Get compliance report")
async def get_compliance_report(declaration_id: str) -> Any:
    """Get EUDR compliance check results for a declaration."""
    try:
        service = get_service()
        result = await service.get_compliance_report(declaration_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Compliance report not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_compliance_report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post(
    "/declarations/{declaration_id}/compliance/dds-check",
    summary="Check DDS reference",
)
async def check_dds_reference(declaration_id: str, request: dict) -> Any:
    """Check DDS reference number validity."""
    try:
        service = get_service()
        result = await service.check_dds_reference(
            declaration_id=declaration_id,
            dds_reference=request.get("dds_reference", ""),
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error("check_dds_reference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# SAD Form Endpoints
# ---------------------------------------------------------------------------

@router.post("/declarations/{declaration_id}/sad-form", summary="Generate SAD form")
async def generate_sad_form(declaration_id: str) -> Any:
    """Generate a SAD form for a declaration."""
    try:
        service = get_service()
        result = await service.generate_sad_form(declaration_id=declaration_id)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("generate_sad_form failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Submission Endpoints
# ---------------------------------------------------------------------------

@router.post("/declarations/{declaration_id}/submit", summary="Submit to customs")
async def submit_declaration(
    declaration_id: str, request: SubmitDeclarationRequest,
) -> Any:
    """Submit a declaration to customs authority."""
    try:
        service = get_service()
        result = await service.submit_declaration(
            declaration_id=declaration_id,
            system=request.system,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("submit_declaration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/declarations/{declaration_id}/submit/status", summary="Check submission status")
async def check_submission_status(declaration_id: str) -> Any:
    """Check submission status."""
    try:
        service = get_service()
        result = await service.check_submission_status(mrn="", system="ncts")
        return result if isinstance(result, dict) else result
    except Exception as e:
        logger.error("check_submission_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# MRN Endpoints
# ---------------------------------------------------------------------------

@router.get("/mrn/{mrn}/status", summary="Get MRN status")
async def get_mrn_status(mrn: str) -> Any:
    """Get MRN status."""
    try:
        service = get_service()
        result = await service.get_mrn_status(mrn=mrn)
        return result if isinstance(result, dict) else result
    except Exception as e:
        logger.error("get_mrn_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------

@router.get("/health", summary="Health check")
async def health_check() -> Any:
    """Perform a health check on the Customs Declaration Support agent."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error("health_check failed: %s", e, exc_info=True)
        return {"agent_id": "GL-EUDR-CDS-039", "status": "error", "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Aliases for backward compatibility
# ---------------------------------------------------------------------------

MapCNCodeRequest = MapCNCodesRequest
GenerateSADRequest = GenerateSADFormRequest
CalculateValueRequestLegacy = CalculateValueRequest
MapCNCodeRequest = MapCNCodesRequest


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the Customs Declaration Support API router."""
    return router
