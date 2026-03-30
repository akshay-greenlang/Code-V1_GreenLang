# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-033: Continuous Monitoring Agent

REST API endpoints for supply chain monitoring, deforestation alert
correlation, compliance auditing, change detection, risk score tracking,
data freshness validation, and regulatory change tracking.

Endpoint Summary (30+):
    POST /scan-supply-chain                        - Run supply chain scan
    GET  /scans                                    - List supply chain scans
    GET  /scans/{scan_id}                          - Get scan details
    GET  /alerts                                   - List monitoring alerts
    POST /check-deforestation                      - Check deforestation alerts
    GET  /deforestation-records                    - List deforestation records
    GET  /deforestation-records/{monitor_id}       - Get deforestation record
    GET  /investigations                           - List investigations
    GET  /investigations/{investigation_id}        - Get investigation
    POST /run-compliance-audit                     - Run compliance audit
    GET  /audits                                   - List compliance audits
    GET  /audits/{audit_id}                        - Get audit details
    POST /detect-changes                           - Detect entity changes
    GET  /changes                                  - List detected changes
    GET  /changes/{detection_id}                   - Get change details
    POST /monitor-risk-scores                      - Monitor risk scores
    GET  /risk-monitors                            - List risk monitors
    GET  /risk-monitors/{monitor_id}               - Get risk monitor
    POST /validate-freshness                       - Validate data freshness
    GET  /freshness-records                        - List freshness records
    GET  /freshness-records/{freshness_id}         - Get freshness record
    GET  /freshness-report                         - Generate freshness report
    POST /check-regulatory                         - Check regulatory updates
    GET  /regulatory-records                       - List regulatory records
    GET  /regulatory-records/{tracking_id}         - Get regulatory record
    POST /generate-summary                         - Generate monitoring summary
    GET  /summaries                                - List monitoring summaries
    GET  /summaries/{summary_id}                   - Get monitoring summary
    GET  /health                                   - Health check
    GET  /dashboard                                - Dashboard data

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import Field

from greenlang.agents.eudr.continuous_monitoring.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class ScanSupplyChainRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    suppliers: List[Dict[str, Any]] = Field(..., description="Supplier data list")

class CheckDeforestationRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    alerts: List[Dict[str, Any]] = Field(..., description="Deforestation alerts from EUDR-020")
    supply_chain_entities: Optional[List[Dict[str, Any]]] = Field(None, description="Supply chain entities")

class RunComplianceAuditRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    operator_data: Dict[str, Any] = Field(..., description="Operator compliance data")

class DetectChangesRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    entity_snapshots: List[Dict[str, Any]] = Field(..., description="Entity old/new state pairs")

class MonitorRiskScoresRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    entity_id: str = Field(..., description="Entity to monitor")
    score_history: List[Dict[str, Any]] = Field(..., description="Historical score data")
    entity_type: str = Field(default="supplier", description="Entity type")
    incidents: Optional[List[Dict[str, Any]]] = Field(None, description="Related incidents")

class ValidateFreshnessRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    entities: List[Dict[str, Any]] = Field(..., description="Entity data with timestamps")

class CheckRegulatoryRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    updates: Optional[List[Dict[str, Any]]] = Field(None, description="Regulatory updates")

class GenerateSummaryRequest(GreenLangBase):
    operator_id: str = Field(..., description="Operator identifier")
    period_start: Optional[str] = Field(None, description="Period start ISO date")
    period_end: Optional[str] = Field(None, description="Period end ISO date")

class ErrorResponse(GreenLangBase):
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/continuous-monitoring",
    tags=["EUDR Continuous Monitoring"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Supply Chain Scan Endpoints
# ---------------------------------------------------------------------------

@router.post("/scan-supply-chain", response_model=Dict[str, Any], summary="Run supply chain scan")
async def scan_supply_chain(request: ScanSupplyChainRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.scan_supply_chain(request.operator_id, request.suppliers)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"scan_supply_chain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/scans", response_model=List[Dict[str, Any]], summary="List supply chain scans")
async def list_scans(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_scans(operator_id=operator_id, status=status)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/scans/{scan_id}", response_model=Dict[str, Any], summary="Get scan details")
async def get_scan(scan_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_scan(scan_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/alerts", response_model=List[Dict[str, Any]], summary="List monitoring alerts")
async def list_alerts(
    operator_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_alerts(operator_id=operator_id, severity=severity)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Deforestation Endpoints
# ---------------------------------------------------------------------------

@router.post("/check-deforestation", response_model=Dict[str, Any], summary="Check deforestation alerts")
async def check_deforestation(request: CheckDeforestationRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.check_deforestation(
            request.operator_id, request.alerts, request.supply_chain_entities,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error(f"check_deforestation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/deforestation-records", response_model=List[Dict[str, Any]], summary="List deforestation records")
async def list_deforestation_records(
    operator_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_deforestation_records(operator_id=operator_id)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/deforestation-records/{monitor_id}", response_model=Dict[str, Any], summary="Get deforestation record")
async def get_deforestation_record(monitor_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_deforestation_record(monitor_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Record {monitor_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/investigations", response_model=List[Dict[str, Any]], summary="List investigations")
async def list_investigations(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_investigations(operator_id=operator_id, status=status)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/investigations/{investigation_id}", response_model=Dict[str, Any], summary="Get investigation")
async def get_investigation(investigation_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_investigation(investigation_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Investigation {investigation_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Compliance Audit Endpoints
# ---------------------------------------------------------------------------

@router.post("/run-compliance-audit", response_model=Dict[str, Any], summary="Run compliance audit")
async def run_compliance_audit(request: RunComplianceAuditRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.run_compliance_audit(request.operator_id, request.operator_data)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error(f"run_compliance_audit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/audits", response_model=List[Dict[str, Any]], summary="List compliance audits")
async def list_audits(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_audits(operator_id=operator_id, status=status)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/audits/{audit_id}", response_model=Dict[str, Any], summary="Get audit details")
async def get_audit(audit_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_audit(audit_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Audit {audit_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Change Detection Endpoints
# ---------------------------------------------------------------------------

@router.post("/detect-changes", response_model=List[Dict[str, Any]], summary="Detect entity changes")
async def detect_changes(request: DetectChangesRequest) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.detect_changes(request.operator_id, request.entity_snapshots)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error(f"detect_changes failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/changes", response_model=List[Dict[str, Any]], summary="List detected changes")
async def list_changes(
    operator_id: Optional[str] = Query(None),
    change_type: Optional[str] = Query(None),
    impact: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_changes(
            operator_id=operator_id, change_type=change_type, impact=impact,
        )
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/changes/{detection_id}", response_model=Dict[str, Any], summary="Get change details")
async def get_change(detection_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_change(detection_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Change {detection_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Risk Score Monitor Endpoints
# ---------------------------------------------------------------------------

@router.post("/monitor-risk-scores", response_model=Dict[str, Any], summary="Monitor risk scores")
async def monitor_risk_scores(request: MonitorRiskScoresRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.monitor_risk_scores(
            request.operator_id, request.entity_id,
            request.score_history, request.entity_type, request.incidents,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error(f"monitor_risk_scores failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/risk-monitors", response_model=List[Dict[str, Any]], summary="List risk monitors")
async def list_risk_monitors(
    operator_id: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    trend: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_risk_monitors(
            operator_id=operator_id, entity_id=entity_id, trend=trend,
        )
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/risk-monitors/{monitor_id}", response_model=Dict[str, Any], summary="Get risk monitor")
async def get_risk_monitor(monitor_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_risk_monitor(monitor_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Risk monitor {monitor_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Data Freshness Endpoints
# ---------------------------------------------------------------------------

@router.post("/validate-freshness", response_model=Dict[str, Any], summary="Validate data freshness")
async def validate_freshness(request: ValidateFreshnessRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.validate_freshness(request.operator_id, request.entities)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error(f"validate_freshness failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/freshness-records", response_model=List[Dict[str, Any]], summary="List freshness records")
async def list_freshness_records(
    operator_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_freshness_records(operator_id=operator_id)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/freshness-records/{freshness_id}", response_model=Dict[str, Any], summary="Get freshness record")
async def get_freshness_record(freshness_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_freshness_record(freshness_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Freshness record {freshness_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/freshness-report", response_model=Dict[str, Any], summary="Generate freshness report")
async def freshness_report(
    operator_id: str = Query(..., description="Operator identifier"),
) -> Dict[str, Any]:
    try:
        service = get_service()
        return await service.generate_freshness_report(operator_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Regulatory Tracking Endpoints
# ---------------------------------------------------------------------------

@router.post("/check-regulatory", response_model=Dict[str, Any], summary="Check regulatory updates")
async def check_regulatory(request: CheckRegulatoryRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.check_regulatory(request.operator_id, request.updates)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error(f"check_regulatory failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/regulatory-records", response_model=List[Dict[str, Any]], summary="List regulatory records")
async def list_regulatory_records(
    operator_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_regulatory_records(operator_id=operator_id)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/regulatory-records/{tracking_id}", response_model=Dict[str, Any], summary="Get regulatory record")
async def get_regulatory_record(tracking_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_regulatory_record(tracking_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Regulatory record {tracking_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Summary & Dashboard Endpoints
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_model=Dict[str, Any], summary="Dashboard data")
async def get_dashboard(
    operator_id: str = Query(..., description="Operator identifier"),
) -> Dict[str, Any]:
    try:
        service = get_service()
        return await service.get_dashboard(operator_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------

@router.get("/health", response_model=Dict[str, Any], summary="Health check")
async def health_check() -> Dict[str, Any]:
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        return {"agent_id": "GL-EUDR-CM-033", "status": "error", "error": str(e)[:200]}


def get_router() -> APIRouter:
    """Return the Continuous Monitoring API router."""
    return router
