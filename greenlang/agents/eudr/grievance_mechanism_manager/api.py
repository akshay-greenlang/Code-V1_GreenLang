# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-032: Grievance Mechanism Manager

REST API endpoints for grievance analytics, root cause analysis, mediation
workflows, remediation tracking, risk scoring, collective grievances, and
regulatory reporting.

Endpoint Summary (25+):
    POST /analyze-patterns                         - Run pattern analytics
    GET  /analytics                                - List analytics records
    GET  /analytics/{analytics_id}                 - Get analytics record
    POST /analyze-root-cause                       - Perform root cause analysis
    GET  /root-causes                              - List root causes
    GET  /root-causes/{root_cause_id}              - Get root cause
    POST /initiate-mediation                       - Start mediation
    POST /mediations/{id}/advance                  - Advance mediation stage
    POST /mediations/{id}/sessions                 - Record session
    POST /mediations/{id}/agreements               - Record agreement
    POST /mediations/{id}/settlement               - Set settlement
    GET  /mediations                               - List mediations
    GET  /mediations/{id}                          - Get mediation
    POST /create-remediation                       - Create remediation
    POST /remediations/{id}/progress               - Update progress
    POST /remediations/{id}/verify                 - Verify remediation
    POST /remediations/{id}/satisfaction            - Record satisfaction
    GET  /remediations                             - List remediations
    GET  /remediations/{id}                        - Get remediation
    POST /compute-risk-score                       - Compute risk score
    GET  /risk-scores                              - List risk scores
    GET  /risk-scores/{id}                         - Get risk score
    POST /create-collective                        - Create collective grievance
    POST /collectives/{id}/demands                 - Add demands
    POST /collectives/{id}/status                  - Update status
    GET  /collectives                              - List collectives
    GET  /collectives/{id}                         - Get collective
    POST /generate-report                          - Generate regulatory report
    GET  /reports                                  - List reports
    GET  /reports/{id}                             - Get report
    GET  /health                                   - Health check

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 (GL-EUDR-GMM-032)
Status: Production Ready
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from greenlang.agents.eudr.grievance_mechanism_manager.setup import get_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class AnalyzePatternsRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    grievances: List[Dict[str, Any]] = Field(..., description="Grievance data list")

class AnalyzeRootCauseRequest(BaseModel):
    grievance_id: str = Field(..., description="Grievance ID from EUDR-031")
    operator_id: str = Field(..., description="Operator identifier")
    grievance_data: Dict[str, Any] = Field(..., description="Grievance details")
    method: Optional[str] = Field(None, description="Analysis method override")

class InitiateMediationRequest(BaseModel):
    grievance_id: str = Field(..., description="Grievance ID from EUDR-031")
    operator_id: str = Field(..., description="Operator identifier")
    parties: List[Dict[str, Any]] = Field(..., description="Involved parties")
    mediator_type: str = Field(default="internal", description="Mediator type")
    mediator_id: Optional[str] = Field(None, description="Mediator ID")

class RecordSessionRequest(BaseModel):
    summary: str = Field(default="", description="Session summary")
    duration_minutes: int = Field(default=120, description="Session duration")
    attendees: List[str] = Field(default_factory=list, description="Attendees")
    outcomes: List[str] = Field(default_factory=list, description="Outcomes")

class RecordAgreementRequest(BaseModel):
    clause: str = Field(..., description="Agreement clause")
    agreed_by: List[str] = Field(default_factory=list, description="Agreeing parties")

class SetSettlementRequest(BaseModel):
    settlement_terms: Dict[str, Any] = Field(..., description="Settlement terms")
    status: str = Field(default="accepted", description="Settlement status")

class CreateRemediationRequest(BaseModel):
    grievance_id: str = Field(..., description="Grievance ID from EUDR-031")
    operator_id: str = Field(..., description="Operator identifier")
    remediation_type: str = Field(..., description="Remediation type")
    actions: Optional[List[Dict[str, Any]]] = Field(None, description="Actions")

class UpdateProgressRequest(BaseModel):
    completion_percentage: float = Field(..., description="Completion %")
    status: Optional[str] = Field(None, description="Status override")

class VerifyRemediationRequest(BaseModel):
    verification_evidence: List[Dict[str, Any]] = Field(..., description="Evidence")
    effectiveness_indicators: Optional[Dict[str, Any]] = Field(None)

class RecordSatisfactionRequest(BaseModel):
    satisfaction_score: float = Field(..., ge=1, le=5, description="Score 1-5")

class ComputeRiskScoreRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    scope: str = Field(..., description="Scope: operator/supplier/commodity/region")
    scope_identifier: str = Field(..., description="Entity within scope")
    grievances: List[Dict[str, Any]] = Field(..., description="Grievance data")

class CreateCollectiveRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    title: str = Field(..., description="Collective grievance title")
    individual_ids: Optional[List[str]] = Field(None, description="Individual grievance IDs")
    description: str = Field(default="", description="Description")
    category: str = Field(default="process", description="Category")
    lead_complainant_id: Optional[str] = Field(None)
    affected_count: int = Field(default=1, ge=1)

class AddDemandsRequest(BaseModel):
    demands: List[Dict[str, Any]] = Field(..., description="Demands list")

class UpdateStatusRequest(BaseModel):
    status: str = Field(..., description="New status")

class GenerateReportRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    report_type: str = Field(..., description="Report type")
    grievances: Optional[List[Dict[str, Any]]] = Field(None)
    remediations: Optional[List[Dict[str, Any]]] = Field(None)

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/grievance-mechanism-manager",
    tags=["EUDR Grievance Mechanism Manager"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Analytics Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze-patterns", response_model=Dict[str, Any], summary="Run pattern analytics")
async def analyze_patterns(request: AnalyzePatternsRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.analyze_patterns(request.operator_id, request.grievances)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"analyze_patterns failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/analytics", response_model=List[Dict[str, Any]], summary="List analytics")
async def list_analytics(
    operator_id: Optional[str] = Query(None),
    pattern_type: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_analytics(operator_id=operator_id, pattern_type=pattern_type)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/analytics/{analytics_id}", response_model=Dict[str, Any], summary="Get analytics")
async def get_analytics(analytics_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_analytics(analytics_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Analytics {analytics_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Root Cause Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze-root-cause", response_model=Dict[str, Any], summary="Analyze root cause")
async def analyze_root_cause(request: AnalyzeRootCauseRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.analyze_root_cause(
            request.grievance_id, request.operator_id,
            request.grievance_data, request.method,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/root-causes", response_model=List[Dict[str, Any]], summary="List root causes")
async def list_root_causes(
    grievance_id: Optional[str] = Query(None),
    operator_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_root_causes(grievance_id=grievance_id, operator_id=operator_id)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/root-causes/{root_cause_id}", response_model=Dict[str, Any], summary="Get root cause")
async def get_root_cause(root_cause_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_root_cause(root_cause_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Root cause {root_cause_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Mediation Endpoints
# ---------------------------------------------------------------------------

@router.post("/initiate-mediation", response_model=Dict[str, Any], summary="Initiate mediation")
async def initiate_mediation(request: InitiateMediationRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.initiate_mediation(
            request.grievance_id, request.operator_id,
            request.parties, request.mediator_type, request.mediator_id,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/mediations/{mediation_id}/advance", response_model=Dict[str, Any], summary="Advance stage")
async def advance_mediation(mediation_id: str, target_stage: Optional[str] = Query(None)) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.advance_mediation(mediation_id, target_stage)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/mediations/{mediation_id}/sessions", response_model=Dict[str, Any], summary="Record session")
async def record_session(mediation_id: str, request: RecordSessionRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.record_mediation_session(mediation_id, request.model_dump(mode="json"))
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/mediations/{mediation_id}/agreements", response_model=Dict[str, Any], summary="Record agreement")
async def record_agreement(mediation_id: str, request: RecordAgreementRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.record_mediation_agreement(mediation_id, request.model_dump(mode="json"))
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/mediations/{mediation_id}/settlement", response_model=Dict[str, Any], summary="Set settlement")
async def set_settlement(mediation_id: str, request: SetSettlementRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.set_mediation_settlement(mediation_id, request.settlement_terms, request.status)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/mediations", response_model=List[Dict[str, Any]], summary="List mediations")
async def list_mediations(
    operator_id: Optional[str] = Query(None),
    grievance_id: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_mediations(operator_id=operator_id, grievance_id=grievance_id, stage=stage)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/mediations/{mediation_id}", response_model=Dict[str, Any], summary="Get mediation")
async def get_mediation(mediation_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_mediation(mediation_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Mediation {mediation_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Remediation Endpoints
# ---------------------------------------------------------------------------

@router.post("/create-remediation", response_model=Dict[str, Any], summary="Create remediation")
async def create_remediation(request: CreateRemediationRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.create_remediation(
            request.grievance_id, request.operator_id,
            request.remediation_type, request.actions,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/remediations/{remediation_id}/progress", response_model=Dict[str, Any], summary="Update progress")
async def update_progress(remediation_id: str, request: UpdateProgressRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.update_remediation_progress(
            remediation_id, request.completion_percentage, request.status,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/remediations/{remediation_id}/verify", response_model=Dict[str, Any], summary="Verify remediation")
async def verify_remediation(remediation_id: str, request: VerifyRemediationRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.verify_remediation(
            remediation_id, request.verification_evidence, request.effectiveness_indicators,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/remediations/{remediation_id}/satisfaction", response_model=Dict[str, Any], summary="Record satisfaction")
async def record_satisfaction(remediation_id: str, request: RecordSatisfactionRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.record_remediation_satisfaction(remediation_id, request.satisfaction_score)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/remediations", response_model=List[Dict[str, Any]], summary="List remediations")
async def list_remediations(
    grievance_id: Optional[str] = Query(None),
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_remediations(grievance_id=grievance_id, operator_id=operator_id, status=status)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/remediations/{remediation_id}", response_model=Dict[str, Any], summary="Get remediation")
async def get_remediation(remediation_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_remediation(remediation_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Remediation {remediation_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Risk Score Endpoints
# ---------------------------------------------------------------------------

@router.post("/compute-risk-score", response_model=Dict[str, Any], summary="Compute risk score")
async def compute_risk_score(request: ComputeRiskScoreRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.compute_risk_score(
            request.operator_id, request.scope, request.scope_identifier, request.grievances,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/risk-scores", response_model=List[Dict[str, Any]], summary="List risk scores")
async def list_risk_scores(
    operator_id: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_risk_scores(operator_id=operator_id, scope=scope, risk_level=risk_level)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/risk-scores/{risk_score_id}", response_model=Dict[str, Any], summary="Get risk score")
async def get_risk_score(risk_score_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_risk_score(risk_score_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Risk score {risk_score_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Collective Grievance Endpoints
# ---------------------------------------------------------------------------

@router.post("/create-collective", response_model=Dict[str, Any], summary="Create collective")
async def create_collective(request: CreateCollectiveRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.create_collective(
            request.operator_id, request.title, request.individual_ids,
            request.description, request.category, request.lead_complainant_id,
            request.affected_count,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/collectives/{collective_id}/demands", response_model=Dict[str, Any], summary="Add demands")
async def add_demands(collective_id: str, request: AddDemandsRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.add_collective_demands(collective_id, request.demands)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.post("/collectives/{collective_id}/status", response_model=Dict[str, Any], summary="Update status")
async def update_collective_status(collective_id: str, request: UpdateStatusRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.update_collective_status(collective_id, request.status)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/collectives", response_model=List[Dict[str, Any]], summary="List collectives")
async def list_collectives(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_collectives(operator_id=operator_id, status=status)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/collectives/{collective_id}", response_model=Dict[str, Any], summary="Get collective")
async def get_collective(collective_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_collective(collective_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Collective {collective_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Report Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-report", response_model=Dict[str, Any], summary="Generate report")
async def generate_report(request: GenerateReportRequest) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.generate_regulatory_report(
            request.operator_id, request.report_type,
            request.grievances, request.remediations,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/reports", response_model=List[Dict[str, Any]], summary="List reports")
async def list_reports(
    operator_id: Optional[str] = Query(None),
    report_type: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    try:
        service = get_service()
        results = await service.list_reports(operator_id=operator_id, report_type=report_type)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:200])

@router.get("/reports/{report_id}", response_model=Dict[str, Any], summary="Get report")
async def get_report(report_id: str) -> Dict[str, Any]:
    try:
        service = get_service()
        result = await service.get_report(report_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
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
        return {"agent_id": "GL-EUDR-GMM-032", "status": "error", "error": str(e)[:200]}


def get_router() -> APIRouter:
    """Return the Grievance Mechanism Manager API router."""
    return router
