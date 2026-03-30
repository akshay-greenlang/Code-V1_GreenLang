# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-036: EU Information System Interface

REST API endpoints for EU Information System interface operations.
Provides 12 endpoints for DDS management, operator registration,
geolocation formatting, package assembly, status tracking, audit
trail access, and health monitoring.

Endpoint Summary (12):
    POST /create-dds                              - Create new DDS
    POST /validate-dds/{dds_id}                   - Validate DDS
    POST /submit-dds/{dds_id}                     - Submit DDS to EU IS
    GET  /dds/{dds_id}                            - Get DDS details
    GET  /dds                                     - List DDS with filters
    POST /register-operator                       - Register operator
    POST /format-geolocation                      - Format coordinates
    POST /assemble-package                        - Assemble document package
    GET  /status/{dds_id}                         - Check submission status
    GET  /audit/{entity_type}/{entity_id}         - Get audit trail
    POST /audit-report/{entity_type}/{entity_id}  - Generate audit report
    GET  /health                                  - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-eu-information-system-interface:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 (GL-EUDR-EUIS-036)
Regulation: EU 2023/1115 (EUDR) Articles 4, 12, 13, 14, 31, 33
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.eu_information_system_interface.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class CreateDDSRequest(GreenLangBase):
    """Request body for creating a Due Diligence Statement."""

    operator_id: str = Field(..., description="GreenLang operator identifier")
    eori_number: str = Field(..., description="EORI number")
    dds_type: str = Field(
        ..., description="DDS type (placing/making_available/export)"
    )
    commodity_lines: List[Dict[str, Any]] = Field(
        ..., description="Commodity line items"
    )
    risk_assessment_id: Optional[str] = Field(
        None, description="Risk assessment reference"
    )
    mitigation_plan_id: Optional[str] = Field(
        None, description="Mitigation plan reference"
    )
    improvement_plan_id: Optional[str] = Field(
        None, description="Improvement plan reference (EUDR-035)"
    )


class RegisterOperatorRequest(GreenLangBase):
    """Request body for registering an operator."""

    operator_id: str = Field(..., description="GreenLang operator identifier")
    eori_number: str = Field(..., description="EORI number")
    operator_type: str = Field(
        ..., description="Operator type (operator/trader/sme_operator/sme_trader)"
    )
    company_name: str = Field(..., description="Legal entity name")
    member_state: str = Field(
        ..., description="EU Member State code (2-letter)"
    )
    address: str = Field(default="", description="Registered address")
    contact_email: str = Field(default="", description="Contact email")


class FormatGeolocationRequest(GreenLangBase):
    """Request body for formatting geolocation data."""

    coordinates: List[Dict[str, Any]] = Field(
        ..., description="Coordinate list with lat/lng keys"
    )
    country_code: str = Field(
        ..., description="ISO 3166-1 alpha-2 country code"
    )
    region: str = Field(default="", description="Sub-national region")
    area_hectares: Optional[str] = Field(
        None, description="Plot area in hectares (decimal string)"
    )


class AssemblePackageRequest(GreenLangBase):
    """Request body for assembling a document package."""

    dds_id: str = Field(..., description="Associated DDS identifier")
    documents: List[Dict[str, Any]] = Field(
        ..., description="Documents with type, content, size"
    )


class CheckStatusRequest(GreenLangBase):
    """Request body for checking DDS status."""

    eu_reference: str = Field(
        ..., description="EU IS reference number"
    )


class ErrorResponse(GreenLangBase):
    """Standard error response body."""

    detail: str = Field(..., description="Error description")
    error_code: str = Field("internal_error", description="Error classification")
    timestamp: Optional[str] = Field(None, description="Error timestamp")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/eu-information-system-interface",
    tags=["EUDR EU Information System Interface"],
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
    "/create-dds",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Create a Due Diligence Statement",
    description=(
        "Create a new DDS with operator information, commodity lines, "
        "and risk assessment references per EUDR Article 4."
    ),
)
async def create_dds(request: CreateDDSRequest) -> Dict[str, Any]:
    """Create a new Due Diligence Statement."""
    try:
        service = get_service()
        dds = await service.create_dds(
            operator_id=request.operator_id,
            eori_number=request.eori_number,
            dds_type=request.dds_type,
            commodity_lines=request.commodity_lines,
            risk_assessment_id=request.risk_assessment_id,
            mitigation_plan_id=request.mitigation_plan_id,
            improvement_plan_id=request.improvement_plan_id,
        )
        if isinstance(dds, dict):
            return dds
        return dds.model_dump(mode="json") if hasattr(dds, "model_dump") else dict(dds)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("create_dds failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"DDS creation failed: {str(e)[:200]}")


@router.post(
    "/validate-dds/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Validate a DDS",
    description="Validate a DDS against EUDR Article 4(2) requirements.",
)
async def validate_dds(dds_id: str) -> Dict[str, Any]:
    """Validate a Due Diligence Statement."""
    try:
        service = get_service()
        return await service.validate_dds(dds_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("validate_dds failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"DDS validation failed: {str(e)[:200]}")


@router.post(
    "/submit-dds/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Submit DDS to EU Information System",
    description="Submit a validated DDS to the EU Information System.",
)
async def submit_dds(dds_id: str) -> Dict[str, Any]:
    """Submit a DDS to the EU Information System."""
    try:
        service = get_service()
        result = await service.submit_dds(dds_id)
        if isinstance(result, dict):
            return result
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("submit_dds failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"DDS submission failed: {str(e)[:200]}")


@router.get(
    "/dds/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Get DDS details",
    description="Retrieve a DDS by its identifier.",
)
async def get_dds(dds_id: str) -> Dict[str, Any]:
    """Get DDS details by identifier."""
    try:
        service = get_service()
        dds = await service.get_dds(dds_id)
        if dds is None:
            raise HTTPException(status_code=404, detail=f"DDS {dds_id} not found")
        if isinstance(dds, dict):
            return dds
        return dds.model_dump(mode="json") if hasattr(dds, "model_dump") else dict(dds)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_dds failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"DDS lookup failed: {str(e)[:200]}")


@router.get(
    "/dds",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="List DDS",
    description="List Due Diligence Statements with optional filters.",
)
async def list_dds(
    operator_id: Optional[str] = Query(None, description="Filter by operator"),
    status: Optional[str] = Query(None, description="Filter by status"),
) -> List[Dict[str, Any]]:
    """List DDS with optional filtering."""
    try:
        service = get_service()
        dds_list = await service.list_dds(operator_id=operator_id, status=status)
        results = []
        for d in dds_list:
            if isinstance(d, dict):
                results.append(d)
            elif hasattr(d, "model_dump"):
                results.append(d.model_dump(mode="json"))
            else:
                results.append(dict(d))
        return results
    except Exception as e:
        logger.error("list_dds failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"DDS listing failed: {str(e)[:200]}")


@router.post(
    "/register-operator",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Register operator",
    description="Register an operator in the EU Information System.",
)
async def register_operator(request: RegisterOperatorRequest) -> Dict[str, Any]:
    """Register an operator in the EU Information System."""
    try:
        service = get_service()
        reg = await service.register_operator(
            operator_id=request.operator_id,
            eori_number=request.eori_number,
            operator_type=request.operator_type,
            company_name=request.company_name,
            member_state=request.member_state,
            address=request.address,
            contact_email=request.contact_email,
        )
        if isinstance(reg, dict):
            return reg
        return reg.model_dump(mode="json") if hasattr(reg, "model_dump") else dict(reg)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("register_operator failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)[:200]}")


@router.post(
    "/format-geolocation",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Format geolocation data",
    description="Format coordinates to EU Information System specifications.",
)
async def format_geolocation(request: FormatGeolocationRequest) -> Dict[str, Any]:
    """Format geolocation data for EU IS submission."""
    try:
        service = get_service()
        result = await service.format_geolocation(
            coordinates=request.coordinates,
            country_code=request.country_code,
            region=request.region,
            area_hectares=request.area_hectares,
        )
        if isinstance(result, dict):
            return result
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("format_geolocation failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"Geolocation formatting failed: {str(e)[:200]}")


@router.post(
    "/assemble-package",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Assemble document package",
    description="Assemble all documents into a submission package.",
)
async def assemble_package(request: AssemblePackageRequest) -> Dict[str, Any]:
    """Assemble a document package for DDS submission."""
    try:
        service = get_service()
        package = await service.assemble_package(
            dds_id=request.dds_id,
            documents=request.documents,
        )
        if isinstance(package, dict):
            return package
        return package.model_dump(mode="json") if hasattr(package, "model_dump") else dict(package)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("assemble_package failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"Package assembly failed: {str(e)[:200]}")


@router.get(
    "/status/{dds_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Check submission status",
    description="Check DDS submission status in the EU Information System.",
)
async def check_status(
    dds_id: str,
    eu_reference: str = Query(..., description="EU IS reference number"),
) -> Dict[str, Any]:
    """Check DDS submission status."""
    try:
        service = get_service()
        result = await service.check_status(dds_id=dds_id, eu_reference=eu_reference)
        if isinstance(result, dict):
            return result
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("check_status failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)[:200]}")


@router.get(
    "/audit/{entity_type}/{entity_id}",
    response_model=List[Dict[str, Any]],
    status_code=200,
    summary="Get audit trail",
    description="Get Article 31 audit trail for an entity.",
)
async def get_audit_trail(entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
    """Get audit trail for a specific entity."""
    try:
        service = get_service()
        records = await service.get_audit_trail(entity_type, entity_id)
        results = []
        for r in records:
            if isinstance(r, dict):
                results.append(r)
            elif hasattr(r, "model_dump"):
                results.append(r.model_dump(mode="json"))
            else:
                results.append(dict(r))
        return results
    except Exception as e:
        logger.error("get_audit_trail failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit trail lookup failed: {str(e)[:200]}")


@router.post(
    "/audit-report/{entity_type}/{entity_id}",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Generate audit report",
    description="Generate Article 31 audit report for competent authority.",
)
async def generate_audit_report(entity_type: str, entity_id: str) -> Dict[str, Any]:
    """Generate an audit report for a specific entity."""
    try:
        service = get_service()
        return await service.generate_audit_report(entity_type, entity_id)
    except Exception as e:
        logger.error("generate_audit_report failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit report failed: {str(e)[:200]}")


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Health check",
    description="Returns health status of the EU Information System Interface.",
)
async def health_check() -> Dict[str, Any]:
    """Perform a health check."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error("health_check failed: %s: %s", type(e).__name__, str(e)[:500], exc_info=True)
        return {"agent_id": "GL-EUDR-EUIS-036", "status": "error", "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the EU Information System Interface API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
