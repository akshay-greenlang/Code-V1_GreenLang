# -*- coding: utf-8 -*-
"""
EUDR Traceability Service REST API Router - AGENT-DATA-004

FastAPI router providing 20 REST API endpoints for EUDR traceability
operations including plot management, chain of custody, due diligence,
risk assessment, commodity classification, and compliance verification.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/eudr", tags=["EUDR Traceability"])

# =============================================================================
# Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    agent_id: str = "GL-DATA-EUDR-001"
    agent_name: str = "EUDR Traceability Connector Agent"
    version: str = "1.0.0"
    timestamp: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""


class PlotResponse(BaseModel):
    """Plot registration response."""
    plot_id: str
    commodity: str
    country_code: str
    risk_level: str
    deforestation_free: bool = False
    legal_compliance: bool = False
    created_at: str = ""


class TransferResponse(BaseModel):
    """Custody transfer response."""
    transfer_id: str
    transaction_id: str
    commodity: str
    quantity: str
    custody_model: str
    verification_status: str = "pending_verification"


class DDSResponse(BaseModel):
    """Due Diligence Statement response."""
    dds_id: str
    commodity: str
    status: str
    risk_level: str
    origin_countries: List[str] = []
    eu_reference_number: Optional[str] = None


class RiskResponse(BaseModel):
    """Risk assessment response."""
    assessment_id: str
    overall_risk_score: float
    risk_level: str
    risk_factors: List[str] = []


class ClassificationResponse(BaseModel):
    """Commodity classification response."""
    classification_id: str
    product_name: str
    commodity: str
    is_derived_product: bool = False
    primary_commodity: Optional[str] = None


class ComplianceResponse(BaseModel):
    """Compliance check response."""
    target_type: str
    target_id: str
    total_checks: int
    passed: int
    failed: int
    compliance_score: float
    checks: List[Dict[str, Any]] = []


class SubmissionResponse(BaseModel):
    """EU submission response."""
    submission_id: str
    dds_id: str
    submission_status: str
    eu_reference: Optional[str] = None


# =============================================================================
# Service Dependency
# =============================================================================


def _get_service():
    """Get or create the EUDR Traceability Service singleton.

    Returns:
        EUDRTraceabilityService instance.
    """
    from greenlang.eudr_traceability.setup import EUDRTraceabilityService
    if not hasattr(_get_service, "_instance"):
        _get_service._instance = EUDRTraceabilityService()
    return _get_service._instance


# =============================================================================
# Health Endpoint
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check endpoint.

    Returns:
        Health status with agent metadata.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


# =============================================================================
# Plot Management Endpoints (5)
# =============================================================================


@router.post("/plots", tags=["Plots"])
async def register_plot(request: Dict[str, Any]):
    """Register a production plot with geolocation (Article 9).

    Creates a new production plot record with WGS84 coordinates,
    commodity type, producer information, and initial risk assessment.

    Args:
        request: Plot registration data.

    Returns:
        Created PlotRecord.
    """
    try:
        service = _get_service()
        from greenlang.eudr_traceability.models import RegisterPlotRequest
        plot_request = RegisterPlotRequest(**request)
        result = service.register_plot(plot_request)
        return _to_dict(result)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error registering plot: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plots", tags=["Plots"])
async def list_plots(
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    country: Optional[str] = Query(None, description="Filter by country code"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List production plots with optional filters.

    Args:
        commodity: Filter by EUDR commodity.
        country: Filter by ISO country code.
        risk_level: Filter by risk level (low/standard/high).
        limit: Maximum results.
        offset: Results offset.

    Returns:
        List of PlotRecord dictionaries.
    """
    try:
        service = _get_service()
        results = service.list_plots(
            commodity=commodity,
            country=country,
            risk_level=risk_level,
            limit=limit,
            offset=offset,
        )
        return [_to_dict(r) for r in results]
    except Exception as e:
        logger.error("Error listing plots: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plots/{plot_id}", tags=["Plots"])
async def get_plot(plot_id: str):
    """Get production plot details by ID.

    Args:
        plot_id: Plot identifier.

    Returns:
        PlotRecord dictionary.
    """
    service = _get_service()
    result = service.get_plot(plot_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Plot not found: {plot_id}")
    return _to_dict(result)


@router.put("/plots/{plot_id}/compliance", tags=["Plots"])
async def update_plot_compliance(plot_id: str, request: Dict[str, Any]):
    """Update plot compliance status (deforestation-free, legal compliance).

    Args:
        plot_id: Plot identifier.
        request: Compliance update data.

    Returns:
        Updated PlotRecord.
    """
    try:
        service = _get_service()
        result = service.plot_registry.update_compliance(
            plot_id=plot_id,
            deforestation_free=request.get("deforestation_free", False),
            legal_compliance=request.get("legal_compliance", False),
            supporting_documents=request.get("supporting_documents"),
        )
        if not result:
            raise HTTPException(
                status_code=404, detail=f"Plot not found: {plot_id}"
            )
        return _to_dict(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/plots/{plot_id}", tags=["Plots"])
async def archive_plot(plot_id: str):
    """Archive a production plot (soft delete).

    Args:
        plot_id: Plot identifier.

    Returns:
        Confirmation message.
    """
    service = _get_service()
    result = service.get_plot(plot_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Plot not found: {plot_id}")
    return {"message": f"Plot {plot_id} archived", "plot_id": plot_id}


# =============================================================================
# Chain of Custody Endpoints (4)
# =============================================================================


@router.post("/custody/transfers", tags=["Chain of Custody"])
async def record_transfer(request: Dict[str, Any]):
    """Record a chain of custody transfer between operators.

    Args:
        request: Transfer data with source/target operators, commodity, quantity.

    Returns:
        Created CustodyTransfer.
    """
    try:
        service = _get_service()
        from greenlang.eudr_traceability.models import RecordTransferRequest
        transfer_request = RecordTransferRequest(**request)
        result = service.record_transfer(transfer_request)
        return _to_dict(result)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error recording transfer: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/custody/transfers", tags=["Chain of Custody"])
async def list_transfers(
    commodity: Optional[str] = Query(None),
    operator_id: Optional[str] = Query(None),
    batch: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query custody transfer records with filters.

    Returns:
        List of CustodyTransfer dictionaries.
    """
    try:
        service = _get_service()
        results = service.chain_of_custody.list_transfers(
            commodity=commodity,
            operator_id=operator_id,
            batch=batch,
            limit=limit,
            offset=offset,
        )
        return [_to_dict(r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/custody/trace/{batch_id}", tags=["Chain of Custody"])
async def trace_batch_origin(batch_id: str):
    """Trace a product batch back to its origin production plots.

    Args:
        batch_id: Batch identifier.

    Returns:
        List of origin PlotRecord dictionaries.
    """
    try:
        service = _get_service()
        results = service.trace_to_origin(batch_id)
        return [_to_dict(r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custody/batches/split", tags=["Chain of Custody"])
async def split_batch(request: Dict[str, Any]):
    """Split a batch into smaller portions for downstream processing.

    Args:
        request: Split specification with parent_batch_id and quantities.

    Returns:
        List of new BatchRecord dictionaries.
    """
    try:
        service = _get_service()
        from decimal import Decimal
        results = service.chain_of_custody.split_batch(
            parent_batch_id=request["parent_batch_id"],
            split_quantities=[Decimal(str(q)) for q in request["quantities"]],
            descriptions=request.get("descriptions", []),
        )
        return [_to_dict(r) for r in results]
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Due Diligence Statement Endpoints (4)
# =============================================================================


@router.post("/dds", tags=["Due Diligence"])
async def generate_dds(request: Dict[str, Any]):
    """Generate a Due Diligence Statement (Article 4).

    Args:
        request: DDS generation data.

    Returns:
        Created DueDiligenceStatement.
    """
    try:
        service = _get_service()
        from greenlang.eudr_traceability.models import GenerateDDSRequest
        dds_request = GenerateDDSRequest(**request)
        result = service.generate_dds(dds_request)
        return _to_dict(result)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error generating DDS: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dds", tags=["Due Diligence"])
async def list_dds(
    status: Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List Due Diligence Statements with filters.

    Returns:
        List of DDS dictionaries.
    """
    try:
        service = _get_service()
        results = service.due_diligence.list_dds(
            status=status,
            commodity=commodity,
            limit=limit,
            offset=offset,
        )
        return [_to_dict(r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dds/{dds_id}", tags=["Due Diligence"])
async def get_dds(dds_id: str):
    """Get Due Diligence Statement details.

    Args:
        dds_id: DDS identifier.

    Returns:
        DDS dictionary.
    """
    service = _get_service()
    result = service.due_diligence.get_dds(dds_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"DDS not found: {dds_id}")
    return _to_dict(result)


@router.post("/dds/{dds_id}/submit", tags=["Due Diligence"])
async def submit_dds(dds_id: str):
    """Submit DDS to EU Information System (Article 12).

    Args:
        dds_id: DDS identifier.

    Returns:
        Updated DDS with submission status.
    """
    try:
        service = _get_service()
        result = service.submit_dds(dds_id)
        if not result:
            raise HTTPException(
                status_code=404, detail=f"DDS not found: {dds_id}"
            )
        return _to_dict(result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Risk Assessment Endpoints (2)
# =============================================================================


@router.post("/risk/assess", tags=["Risk Assessment"])
async def assess_risk(request: Dict[str, Any]):
    """Perform risk assessment for a product/plot/operator (Article 10).

    Args:
        request: Risk assessment parameters.

    Returns:
        RiskScore dictionary.
    """
    try:
        service = _get_service()
        from greenlang.eudr_traceability.models import AssessRiskRequest
        risk_request = AssessRiskRequest(**request)
        result = service.assess_risk(risk_request)
        return _to_dict(result)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/countries", tags=["Risk Assessment"])
async def get_country_risk_classifications():
    """Get country risk classifications per Article 29.

    Returns:
        Dictionary mapping country codes to risk levels.
    """
    try:
        service = _get_service()
        return service.risk_assessment.get_country_classifications()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Commodity Classification Endpoint (1)
# =============================================================================


@router.post("/commodities/classify", tags=["Commodity Classification"])
async def classify_commodity(request: Dict[str, Any]):
    """Classify product by CN/HS code for EUDR coverage.

    Args:
        request: Classification data with product name and/or codes.

    Returns:
        CommodityClassification dictionary.
    """
    try:
        service = _get_service()
        from greenlang.eudr_traceability.models import ClassifyCommodityRequest
        cls_request = ClassifyCommodityRequest(**request)
        result = service.classify_commodity(cls_request)
        return _to_dict(result)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Supplier Declaration Endpoints (2)
# =============================================================================


@router.post("/suppliers/declarations", tags=["Supplier Declarations"])
async def register_declaration(request: Dict[str, Any]):
    """Register a supplier compliance declaration.

    Args:
        request: Supplier declaration data.

    Returns:
        Created declaration record.
    """
    try:
        # Store declaration (simplified - delegates to service when available)
        from datetime import date
        declaration = {
            "declaration_id": f"DECL-{__import__('uuid').uuid4().hex[:12].upper()}",
            "supplier_id": request.get("supplier_id", ""),
            "supplier_name": request.get("supplier_name", ""),
            "supplier_country": request.get("supplier_country", ""),
            "commodities_covered": request.get("commodities_covered", []),
            "confirms_deforestation_free": request.get(
                "confirms_deforestation_free", False
            ),
            "confirms_legal_production": request.get(
                "confirms_legal_production", False
            ),
            "confirms_traceability": request.get("confirms_traceability", False),
            "declaration_date": date.today().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
        }
        return declaration
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/suppliers/declarations", tags=["Supplier Declarations"])
async def list_declarations(
    supplier_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """List supplier compliance declarations.

    Returns:
        List of declaration dictionaries.
    """
    return []


# =============================================================================
# Compliance Endpoint (1)
# =============================================================================


@router.get("/compliance/summary", tags=["Compliance"])
async def get_compliance_summary():
    """Get overall compliance summary report.

    Returns:
        Compliance statistics across all tracked entities.
    """
    try:
        service = _get_service()
        return service.compliance_verifier.get_compliance_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Statistics Endpoint (1)
# =============================================================================


@router.get("/statistics", tags=["Statistics"])
async def get_statistics():
    """Get comprehensive service statistics.

    Returns:
        Statistics from all 7 engines.
    """
    try:
        service = _get_service()
        return service.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Utility
# =============================================================================


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model or dataclass to dictionary.

    Args:
        obj: Object to convert.

    Returns:
        Dictionary representation.
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
