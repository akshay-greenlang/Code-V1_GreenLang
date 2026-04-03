# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-031: Stakeholder Engagement Tool

REST API endpoints for EUDR stakeholder engagement lifecycle management.
Provides 30+ endpoints for stakeholder mapping, FPIC workflow management,
grievance mechanism operations, consultation records, multi-channel
communications, engagement quality assessment, compliance reporting,
and health monitoring.

Endpoint Summary (30+):
    POST /map-stakeholder                              - Map new stakeholder
    GET  /stakeholders                                  - List stakeholders
    GET  /stakeholders/{stakeholder_id}                 - Get stakeholder details
    POST /initiate-fpic                                 - Start FPIC workflow
    POST /fpic/{fpic_id}/advance-stage                  - Advance FPIC stage
    POST /fpic/{fpic_id}/record-consent                 - Record consent
    GET  /fpic/{fpic_id}                                - Get FPIC workflow
    GET  /fpic                                          - List FPIC workflows
    POST /submit-grievance                              - Submit complaint
    POST /grievances/{grievance_id}/triage              - Triage grievance
    POST /grievances/{grievance_id}/investigate          - Investigate grievance
    POST /grievances/{grievance_id}/resolve              - Resolve grievance
    POST /grievances/{grievance_id}/appeal               - Appeal grievance
    GET  /grievances/{grievance_id}                      - Get grievance
    GET  /grievances                                     - List grievances
    POST /create-consultation                            - Create consultation
    POST /consultations/{consultation_id}/participants   - Add participants
    POST /consultations/{consultation_id}/outcomes       - Record outcomes
    POST /consultations/{consultation_id}/evidence       - Attach evidence
    POST /consultations/{consultation_id}/finalize       - Finalize consultation
    GET  /consultations/{consultation_id}                - Get consultation
    GET  /consultations                                  - List consultations
    POST /send-communication                             - Send communication
    POST /schedule-communication                         - Schedule communication
    POST /send-campaign                                  - Send campaign
    GET  /communications/{communication_id}              - Get communication
    GET  /communications                                 - List communications
    POST /assess-engagement/{stakeholder_id}             - Assess engagement
    GET  /assessments/{assessment_id}                    - Get assessment
    POST /generate-report                                - Generate report
    GET  /reports/{report_id}                            - Get report
    GET  /reports/{report_id}/export                     - Export report
    GET  /health                                         - Health check

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and check
    eudr-stakeholder-engagement:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR) Articles 2, 4, 8, 9, 10, 11, 12, 29, 31
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import Field

from greenlang.agents.eudr.stakeholder_engagement.setup import get_service
from greenlang.schemas import GreenLangBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class MapStakeholderRequest(GreenLangBase):
    """Request body for mapping a stakeholder."""
    operator_id: str = Field(..., description="EUDR operator identifier")
    name: str = Field(..., description="Stakeholder name")
    type: str = Field(default="civil_society", description="Stakeholder type")
    country_code: str = Field(default="", description="ISO 3166-1 country code")
    region: str = Field(default="", description="Geographic region")
    contact_info: Optional[Dict[str, Any]] = Field(None, description="Contact info")
    supply_chain_nodes: Optional[List[str]] = Field(None, description="Supply chain nodes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extra metadata")


class InitiateFPICRequest(GreenLangBase):
    """Request body for initiating an FPIC workflow."""
    operator_id: str = Field(..., description="EUDR operator identifier")
    stakeholder_id: str = Field(..., description="Stakeholder identifier")
    supply_chain_node: str = Field(default="", description="Supply chain node")


class AdvanceStageRequest(GreenLangBase):
    """Request body for advancing an FPIC stage."""
    next_stage: str = Field(..., description="Target FPIC stage")
    evidence: Optional[Dict[str, Any]] = Field(None, description="Stage evidence")


class RecordConsentRequest(GreenLangBase):
    """Request body for recording FPIC consent."""
    consent_status: str = Field(..., description="Consent status")
    agreement_terms: Optional[Dict[str, Any]] = Field(None, description="Agreement terms")


class SubmitGrievanceRequest(GreenLangBase):
    """Request body for submitting a grievance."""
    operator_id: str = Field(..., description="Operator identifier")
    description: str = Field(..., description="Complaint description")
    complainant_name: str = Field(default="anonymous", description="Complainant name")
    stakeholder_id: Optional[str] = Field(None, description="Linked stakeholder")
    channel: str = Field(default="web_portal", description="Intake channel")
    language: str = Field(default="en", description="Language of submission")
    supply_chain_node: str = Field(default="", description="Supply chain node")


class InvestigateRequest(GreenLangBase):
    """Request body for investigating a grievance."""
    investigator: str = Field(default="", description="Investigator name")
    findings: str = Field(default="", description="Investigation findings")
    evidence_collected: Optional[List[str]] = Field(None, description="Evidence refs")
    root_cause: str = Field(default="", description="Root cause")


class ResolveGrievanceRequest(GreenLangBase):
    """Request body for resolving a grievance."""
    resolution_type: str = Field(default="", description="Resolution type")
    actions_taken: Optional[List[str]] = Field(None, description="Actions taken")
    remediation: str = Field(default="", description="Remediation provided")
    preventive_measures: Optional[List[str]] = Field(None, description="Preventive measures")
    resolved_by: str = Field(default="", description="Resolver identity")


class CreateConsultationRequest(GreenLangBase):
    """Request body for creating a consultation."""
    operator_id: str = Field(..., description="Operator identifier")
    title: str = Field(..., description="Consultation title")
    type: str = Field(default="community_meeting", description="Consultation type")
    objectives: Optional[List[str]] = Field(None, description="Objectives")
    location: str = Field(default="", description="Location")
    date: Optional[str] = Field(None, description="Date (ISO 8601)")
    duration_minutes: int = Field(default=0, description="Duration in minutes")
    language: str = Field(default="en", description="Language")


class SendCommunicationRequest(GreenLangBase):
    """Request body for sending a communication."""
    operator_id: str = Field(..., description="Operator identifier")
    stakeholder_ids: List[str] = Field(..., description="Target stakeholders")
    message: str = Field(..., description="Message content")
    channel: str = Field(default="email", description="Communication channel")
    subject: str = Field(default="", description="Subject line")
    language: str = Field(default="en", description="Language")


class ScheduleCommunicationRequest(GreenLangBase):
    """Request body for scheduling a communication."""
    operator_id: str = Field(..., description="Operator identifier")
    stakeholder_ids: List[str] = Field(..., description="Target stakeholders")
    message: str = Field(..., description="Message content")
    scheduled_at: str = Field(..., description="Scheduled time (ISO 8601)")
    channel: str = Field(default="email", description="Channel")
    subject: str = Field(default="", description="Subject line")
    language: str = Field(default="en", description="Language")


class SendCampaignRequest(GreenLangBase):
    """Request body for sending a campaign."""
    operator_id: str = Field(..., description="Operator identifier")
    stakeholder_ids: List[str] = Field(..., description="Target stakeholders")
    message: str = Field(..., description="Message content")
    channel: str = Field(default="email", description="Channel")
    subject: str = Field(default="", description="Subject line")
    language: str = Field(default="en", description="Language")


class AssessEngagementRequest(GreenLangBase):
    """Request body for assessing engagement quality."""
    operator_id: str = Field(..., description="Operator identifier")
    period: Optional[Dict[str, str]] = Field(None, description="Period start/end")
    engagement_data: Optional[Dict[str, Any]] = Field(None, description="Engagement data")


class GenerateReportRequest(GreenLangBase):
    """Request body for generating a compliance report."""
    operator_id: str = Field(..., description="Operator identifier")
    report_type: str = Field(..., description="Report type")
    year: Optional[int] = Field(None, description="Report year (for annual reports)")
    period: Optional[Dict[str, str]] = Field(None, description="Period start/end")


class ErrorResponse(GreenLangBase):
    """Standard error response body."""
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error", description="Error classification")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/stakeholder-engagement",
    tags=["EUDR Stakeholder Engagement"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Stakeholder Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/map-stakeholder",
    response_model=Dict[str, Any],
    status_code=200,
    summary="Map a new stakeholder",
    description="Register a new stakeholder in the engagement registry with type classification and rights analysis.",
)
async def map_stakeholder(request: MapStakeholderRequest) -> Dict[str, Any]:
    """Map a new stakeholder to the engagement registry."""
    try:
        service = get_service()
        data = request.model_dump(mode="json")
        result = await service.map_stakeholder(
            operator_id=request.operator_id,
            stakeholder_data=data,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("map_stakeholder failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/stakeholders", response_model=List[Dict[str, Any]], summary="List stakeholders")
async def list_stakeholders(
    operator_id: Optional[str] = Query(None),
    stakeholder_type: Optional[str] = Query(None),
    country_code: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List stakeholders with optional filters."""
    try:
        service = get_service()
        results = await service.list_stakeholders(operator_id=operator_id, stakeholder_type=stakeholder_type, country_code=country_code)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("list_stakeholders failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/stakeholders/{stakeholder_id}", response_model=Dict[str, Any], summary="Get stakeholder details")
async def get_stakeholder(stakeholder_id: str) -> Dict[str, Any]:
    """Get stakeholder by identifier."""
    try:
        service = get_service()
        result = await service.get_stakeholder(stakeholder_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Stakeholder {stakeholder_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_stakeholder failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# FPIC Endpoints
# ---------------------------------------------------------------------------

@router.post("/initiate-fpic", response_model=Dict[str, Any], summary="Start FPIC workflow")
async def initiate_fpic(request: InitiateFPICRequest) -> Dict[str, Any]:
    """Initiate a new FPIC workflow."""
    try:
        service = get_service()
        result = await service.initiate_fpic(
            operator_id=request.operator_id,
            stakeholder_id=request.stakeholder_id,
            supply_chain_node=request.supply_chain_node,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("initiate_fpic failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/fpic/{fpic_id}/advance-stage", response_model=Dict[str, Any], summary="Advance FPIC stage")
async def advance_fpic_stage(fpic_id: str, request: AdvanceStageRequest) -> Dict[str, Any]:
    """Advance an FPIC workflow to the next stage."""
    try:
        service = get_service()
        result = await service.advance_fpic_stage(
            fpic_id=fpic_id, next_stage=request.next_stage, evidence=request.evidence,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("advance_fpic_stage failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/fpic/{fpic_id}/record-consent", response_model=Dict[str, Any], summary="Record consent")
async def record_fpic_consent(fpic_id: str, request: RecordConsentRequest) -> Dict[str, Any]:
    """Record consent status for an FPIC workflow."""
    try:
        service = get_service()
        result = await service.record_fpic_consent(
            fpic_id=fpic_id, consent_status=request.consent_status,
            agreement_terms=request.agreement_terms,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("record_fpic_consent failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/fpic/{fpic_id}", response_model=Dict[str, Any], summary="Get FPIC workflow")
async def get_fpic_workflow(fpic_id: str) -> Dict[str, Any]:
    """Get FPIC workflow by identifier."""
    try:
        service = get_service()
        result = await service.get_fpic_workflow(fpic_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"FPIC workflow {fpic_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_fpic_workflow failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/fpic", response_model=List[Dict[str, Any]], summary="List FPIC workflows")
async def list_fpic_workflows(
    operator_id: Optional[str] = Query(None),
    stakeholder_id: Optional[str] = Query(None),
    current_stage: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List FPIC workflows with optional filters."""
    try:
        service = get_service()
        results = await service.list_fpic_workflows(operator_id=operator_id, stakeholder_id=stakeholder_id, current_stage=current_stage)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("list_fpic_workflows failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Grievance Endpoints
# ---------------------------------------------------------------------------

@router.post("/submit-grievance", response_model=Dict[str, Any], summary="Submit complaint")
async def submit_grievance(request: SubmitGrievanceRequest) -> Dict[str, Any]:
    """Submit a new grievance."""
    try:
        service = get_service()
        result = await service.submit_grievance(
            operator_id=request.operator_id,
            complaint_data=request.model_dump(mode="json"),
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("submit_grievance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/grievances/{grievance_id}/triage", response_model=Dict[str, Any], summary="Triage grievance")
async def triage_grievance(grievance_id: str) -> Dict[str, Any]:
    """Triage a grievance by classifying severity and category."""
    try:
        service = get_service()
        result = await service.triage_grievance(grievance_id)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("triage_grievance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/grievances/{grievance_id}/investigate", summary="Investigate grievance")
async def investigate_grievance(grievance_id: str, request: InvestigateRequest) -> Dict[str, Any]:
    """Record investigation notes for a grievance."""
    try:
        service = get_service()
        await service.investigate_grievance(grievance_id, request.model_dump(mode="json"))
        return {"status": "investigation_recorded", "grievance_id": grievance_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("investigate_grievance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/grievances/{grievance_id}/resolve", response_model=Dict[str, Any], summary="Resolve grievance")
async def resolve_grievance(grievance_id: str, request: ResolveGrievanceRequest) -> Dict[str, Any]:
    """Resolve a grievance with specified actions."""
    try:
        service = get_service()
        result = await service.resolve_grievance(grievance_id, request.model_dump(mode="json"))
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("resolve_grievance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/grievances/{grievance_id}/appeal", response_model=Dict[str, Any], summary="Appeal grievance")
async def appeal_grievance(grievance_id: str, appeal_reason: str = Query(...)) -> Dict[str, Any]:
    """Appeal a resolved grievance."""
    try:
        service = get_service()
        result = await service.appeal_grievance(grievance_id, appeal_reason)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("appeal_grievance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/grievances/{grievance_id}", response_model=Dict[str, Any], summary="Get grievance")
async def get_grievance(grievance_id: str) -> Dict[str, Any]:
    """Get grievance by identifier."""
    try:
        service = get_service()
        result = await service.get_grievance(grievance_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Grievance {grievance_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_grievance failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/grievances", response_model=List[Dict[str, Any]], summary="List grievances")
async def list_grievances(
    operator_id: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List grievances with optional filters."""
    try:
        service = get_service()
        results = await service.list_grievances(operator_id=operator_id, severity=severity, status=status, category=category)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("list_grievances failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Consultation Endpoints
# ---------------------------------------------------------------------------

@router.post("/create-consultation", response_model=Dict[str, Any], summary="Create consultation")
async def create_consultation(request: CreateConsultationRequest) -> Dict[str, Any]:
    """Create a new consultation record."""
    try:
        service = get_service()
        result = await service.create_consultation(
            operator_id=request.operator_id,
            consultation_data=request.model_dump(mode="json"),
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("create_consultation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/consultations/{consultation_id}/participants", summary="Add participants")
async def add_participants(consultation_id: str, participants: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add participants to a consultation."""
    try:
        service = get_service()
        await service.add_consultation_participants(consultation_id, participants)
        return {"status": "participants_added", "consultation_id": consultation_id}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("add_participants failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/consultations/{consultation_id}/outcomes", summary="Record outcomes")
async def record_outcomes(consultation_id: str, outcomes: List[str], commitments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Record outcomes and commitments from a consultation."""
    try:
        service = get_service()
        await service.record_consultation_outcomes(consultation_id, outcomes, commitments)
        return {"status": "outcomes_recorded", "consultation_id": consultation_id}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("record_outcomes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/consultations/{consultation_id}/evidence", summary="Attach evidence")
async def attach_evidence(consultation_id: str, evidence_files: List[str]) -> Dict[str, Any]:
    """Attach evidence files to a consultation."""
    try:
        service = get_service()
        await service.attach_consultation_evidence(consultation_id, evidence_files)
        return {"status": "evidence_attached", "consultation_id": consultation_id}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("attach_evidence failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/consultations/{consultation_id}/finalize", response_model=Dict[str, Any], summary="Finalize consultation")
async def finalize_consultation(consultation_id: str) -> Dict[str, Any]:
    """Finalize a consultation record, making it immutable."""
    try:
        service = get_service()
        result = await service.finalize_consultation(consultation_id)
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("finalize_consultation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/consultations/{consultation_id}", response_model=Dict[str, Any], summary="Get consultation")
async def get_consultation(consultation_id: str) -> Dict[str, Any]:
    """Get consultation by identifier."""
    try:
        service = get_service()
        result = await service.get_consultation(consultation_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Consultation {consultation_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_consultation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/consultations", response_model=List[Dict[str, Any]], summary="List consultations")
async def list_consultations(
    operator_id: Optional[str] = Query(None),
    consultation_type: Optional[str] = Query(None),
    is_finalized: Optional[bool] = Query(None),
) -> List[Dict[str, Any]]:
    """List consultations with optional filters."""
    try:
        service = get_service()
        results = await service.list_consultations(operator_id=operator_id, consultation_type=consultation_type, is_finalized=is_finalized)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("list_consultations failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Communication Endpoints
# ---------------------------------------------------------------------------

@router.post("/send-communication", response_model=Dict[str, Any], summary="Send communication")
async def send_communication(request: SendCommunicationRequest) -> Dict[str, Any]:
    """Send a communication to stakeholders."""
    try:
        service = get_service()
        result = await service.send_communication(
            operator_id=request.operator_id,
            stakeholder_ids=request.stakeholder_ids,
            message=request.message,
            channel=request.channel,
            subject=request.subject,
            language=request.language,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("send_communication failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/schedule-communication", summary="Schedule communication")
async def schedule_communication(request: ScheduleCommunicationRequest) -> Dict[str, Any]:
    """Schedule a communication for future delivery."""
    try:
        service = get_service()
        schedule_id = await service.schedule_communication(
            operator_id=request.operator_id,
            stakeholder_ids=request.stakeholder_ids,
            message=request.message,
            scheduled_at=request.scheduled_at,
            channel=request.channel,
            subject=request.subject,
            language=request.language,
        )
        return {"schedule_id": schedule_id, "status": "scheduled"}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("schedule_communication failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.post("/send-campaign", response_model=List[Dict[str, Any]], summary="Send campaign")
async def send_campaign(request: SendCampaignRequest) -> List[Dict[str, Any]]:
    """Send a coordinated multi-stakeholder campaign."""
    try:
        service = get_service()
        results = await service.send_campaign(request.model_dump(mode="json"))
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("send_campaign failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/communications/{communication_id}", response_model=Dict[str, Any], summary="Get communication")
async def get_communication(communication_id: str) -> Dict[str, Any]:
    """Get communication by identifier."""
    try:
        service = get_service()
        result = await service.get_communication(communication_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Communication {communication_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_communication failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/communications", response_model=List[Dict[str, Any]], summary="List communications")
async def list_communications(
    operator_id: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    delivery_status: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """List communications with optional filters."""
    try:
        service = get_service()
        results = await service.list_communications(operator_id=operator_id, channel=channel, delivery_status=delivery_status)
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in results]
    except Exception as e:
        logger.error("list_communications failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Assessment Endpoints
# ---------------------------------------------------------------------------

@router.post("/assess-engagement/{stakeholder_id}", response_model=Dict[str, Any], summary="Assess engagement quality")
async def assess_engagement(stakeholder_id: str, request: AssessEngagementRequest) -> Dict[str, Any]:
    """Assess engagement quality for a stakeholder."""
    try:
        service = get_service()
        result = await service.assess_engagement(
            operator_id=request.operator_id,
            stakeholder_id=stakeholder_id,
            period=request.period,
            engagement_data=request.engagement_data,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except Exception as e:
        logger.error("assess_engagement failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/assessments/{assessment_id}", response_model=Dict[str, Any], summary="Get assessment")
async def get_assessment(assessment_id: str) -> Dict[str, Any]:
    """Get assessment by identifier."""
    try:
        service = get_service()
        result = await service.get_assessment(assessment_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Assessment {assessment_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_assessment failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Report Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-report", response_model=Dict[str, Any], summary="Generate compliance report")
async def generate_report(request: GenerateReportRequest) -> Dict[str, Any]:
    """Generate a compliance report."""
    try:
        service = get_service()
        result = await service.generate_report(
            operator_id=request.operator_id,
            report_type=request.report_type,
            year=request.year,
            period=request.period,
        )
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("generate_report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/reports/{report_id}", response_model=Dict[str, Any], summary="Get report")
async def get_report(report_id: str) -> Dict[str, Any]:
    """Get report by identifier."""
    try:
        service = get_service()
        result = await service.get_report(report_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        return result if isinstance(result, dict) else result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


@router.get("/reports/{report_id}/export", summary="Export report")
async def export_report(
    report_id: str,
    format: str = Query(default="json", description="Export format"),
    language: str = Query(default="en", description="Export language"),
) -> Response:
    """Export a report in the specified format."""
    try:
        service = get_service()
        content = await service.export_report(report_id, format=format, language=language)
        media_type = "application/json" if format == "json" else "application/octet-stream"
        return Response(content=content, media_type=media_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("export_report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------------------------

@router.get("/health", response_model=Dict[str, Any], summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Perform a health check on the Stakeholder Engagement Tool."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        logger.error("health_check failed: %s", e, exc_info=True)
        return {"agent_id": "GL-EUDR-SET-031", "status": "error", "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the Stakeholder Engagement Tool API router.

    Used by ``auth_setup.configure_auth()`` to include the router
    in the main FastAPI application.

    Returns:
        The configured APIRouter instance.
    """
    return router
