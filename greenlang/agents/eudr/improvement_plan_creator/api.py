# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-035: Improvement Plan Creator

REST API endpoints for improvement plan creation, finding aggregation,
gap analysis, action generation, root cause mapping, prioritization,
progress tracking, stakeholder coordination, and reporting.

Endpoint Summary (30+):
    POST /create-plan                              - Create a full improvement plan
    GET  /plans                                    - List improvement plans
    GET  /plans/{plan_id}                          - Get plan details
    POST /update-plan-status/{plan_id}             - Update plan status
    POST /aggregate-findings                       - Aggregate findings from agents
    GET  /aggregations                             - List aggregations
    GET  /aggregations/{aggregation_id}            - Get aggregation details
    POST /analyze-gaps                             - Analyze compliance gaps
    GET  /gaps/{plan_id}                           - Get gaps for a plan
    POST /generate-actions                         - Generate SMART actions
    GET  /actions/{plan_id}                        - Get actions for a plan
    POST /update-action-status                     - Update action status
    POST /analyze-root-causes                      - Perform root cause analysis
    GET  /root-causes/{plan_id}                    - Get root causes for a plan
    POST /build-fishbone                           - Build fishbone analysis
    POST /prioritize-actions                       - Prioritize plan actions
    POST /capture-progress/{plan_id}               - Capture progress snapshot
    GET  /progress/{plan_id}                       - Get progress history
    POST /check-overdue/{plan_id}                  - Check for overdue actions
    POST /review-effectiveness/{plan_id}           - Review action effectiveness
    POST /assign-stakeholders                      - Assign stakeholders to action
    POST /send-notification                        - Send stakeholder notification
    POST /send-bulk-notifications                  - Send bulk notifications
    POST /acknowledge/{action_id}/{stakeholder_id} - Acknowledge assignment
    GET  /pending-acknowledgments/{plan_id}        - Get pending acknowledgments
    POST /generate-report/{plan_id}                - Generate plan report
    GET  /dashboard                                - Dashboard data
    GET  /health                                   - Health check

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from greenlang.agents.eudr.improvement_plan_creator.setup import get_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class CreatePlanRequest(BaseModel):
    """Request body for creating an improvement plan."""
    operator_id: str = Field(..., description="Operator identifier")
    findings: List[Dict[str, Any]] = Field(..., description="Raw findings from agents")
    commodity: Optional[str] = Field(None, description="EUDR commodity")
    title: str = Field(default="", description="Plan title")
    description: str = Field(default="", description="Plan description")


class UpdatePlanStatusRequest(BaseModel):
    """Request body for updating plan status."""
    new_status: str = Field(..., description="Target plan status")


class AggregateFindingsRequest(BaseModel):
    """Request body for finding aggregation."""
    operator_id: str = Field(..., description="Operator identifier")
    findings: List[Dict[str, Any]] = Field(..., description="Raw findings")
    plan_id: str = Field(default="", description="Associated plan ID")


class AnalyzeGapsRequest(BaseModel):
    """Request body for gap analysis."""
    aggregation_id: str = Field(..., description="Aggregation to analyze")
    plan_id: str = Field(default="", description="Plan identifier")


class GenerateActionsRequest(BaseModel):
    """Request body for action generation."""
    plan_id: str = Field(..., description="Plan with gaps to generate actions for")


class UpdateActionStatusRequest(BaseModel):
    """Request body for action status update."""
    plan_id: str = Field(..., description="Plan identifier")
    action_id: str = Field(..., description="Action identifier")
    new_status: str = Field(..., description="New action status")


class AnalyzeRootCausesRequest(BaseModel):
    """Request body for root cause analysis."""
    plan_id: str = Field(..., description="Plan identifier")


class BuildFishboneRequest(BaseModel):
    """Request body for fishbone analysis."""
    plan_id: str = Field(..., description="Plan identifier")
    gap_id: str = Field(..., description="Gap identifier")


class PrioritizeActionsRequest(BaseModel):
    """Request body for action prioritization."""
    plan_id: str = Field(..., description="Plan identifier")


class AssignStakeholdersRequest(BaseModel):
    """Request body for stakeholder assignment."""
    plan_id: str = Field(..., description="Plan identifier")
    action_id: str = Field(..., description="Action identifier")
    stakeholders: List[Dict[str, Any]] = Field(..., description="Stakeholder list")


class SendNotificationRequest(BaseModel):
    """Request body for sending a notification."""
    action_id: str = Field(..., description="Action identifier")
    stakeholder_id: str = Field(..., description="Stakeholder identifier")
    subject: str = Field(..., description="Notification subject")
    body: str = Field(..., description="Notification body")


class SendBulkNotificationsRequest(BaseModel):
    """Request body for bulk notifications."""
    plan_id: str = Field(..., description="Plan identifier")
    action_id: str = Field(..., description="Action identifier")
    subject: str = Field(..., description="Notification subject")
    body: str = Field(..., description="Notification body")


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/improvement-plan-creator",
    tags=["EUDR Improvement Plan Creator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Plan Endpoints
# ---------------------------------------------------------------------------


@router.post("/create-plan", summary="Create a full improvement plan")
async def create_plan(req: CreatePlanRequest) -> Dict[str, Any]:
    """Create a comprehensive improvement plan via the full pipeline."""
    try:
        service = get_service()
        from greenlang.agents.eudr.improvement_plan_creator.models import Finding
        findings = [Finding(**f) for f in req.findings]
        result = await service.create_improvement_plan(
            operator_id=req.operator_id,
            findings=findings,
            commodity=req.commodity,
            title=req.title,
            description=req.description,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("create_plan failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans", summary="List improvement plans")
async def list_plans(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List improvement plans with optional filters."""
    try:
        service = get_service()
        results = await service.list_plans(operator_id, status, commodity, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_plans failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/plans/{plan_id}", summary="Get plan details")
async def get_plan(plan_id: str) -> Dict[str, Any]:
    """Get a specific improvement plan by ID."""
    try:
        service = get_service()
        result = await service.get_plan(plan_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_plan failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-plan-status/{plan_id}", summary="Update plan status")
async def update_plan_status(plan_id: str, req: UpdatePlanStatusRequest) -> Dict[str, Any]:
    """Update the status of an improvement plan."""
    try:
        service = get_service()
        from greenlang.agents.eudr.improvement_plan_creator.models import PlanStatus
        result = await service.update_plan_status(
            plan_id, PlanStatus(req.new_status)
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("update_plan_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Finding Aggregation Endpoints
# ---------------------------------------------------------------------------


@router.post("/aggregate-findings", summary="Aggregate findings from agents")
async def aggregate_findings(req: AggregateFindingsRequest) -> Dict[str, Any]:
    """Aggregate and deduplicate findings from upstream agents."""
    try:
        service = get_service()
        from greenlang.agents.eudr.improvement_plan_creator.models import Finding
        findings = [Finding(**f) for f in req.findings]
        engine = service._engines.get("finding_aggregator")
        if not engine:
            raise HTTPException(status_code=503, detail="FindingAggregator not available")
        result = await engine.aggregate_findings(req.operator_id, findings, req.plan_id)
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("aggregate_findings failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregations", summary="List aggregations")
async def list_aggregations(
    operator_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List finding aggregations."""
    try:
        service = get_service()
        engine = service._engines.get("finding_aggregator")
        if not engine:
            return []
        results = await engine.list_aggregations(operator_id, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_aggregations failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregations/{aggregation_id}", summary="Get aggregation details")
async def get_aggregation(aggregation_id: str) -> Dict[str, Any]:
    """Get a specific finding aggregation by ID."""
    try:
        service = get_service()
        engine = service._engines.get("finding_aggregator")
        if not engine:
            raise HTTPException(status_code=503, detail="FindingAggregator not available")
        result = await engine.get_aggregation(aggregation_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Aggregation {aggregation_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_aggregation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Gap Analysis Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze-gaps", summary="Analyze compliance gaps")
async def analyze_gaps(req: AnalyzeGapsRequest) -> List[Dict[str, Any]]:
    """Analyze compliance gaps from an aggregation."""
    try:
        service = get_service()
        agg_engine = service._engines.get("finding_aggregator")
        gap_engine = service._engines.get("gap_analyzer")
        if not agg_engine or not gap_engine:
            raise HTTPException(status_code=503, detail="Required engines not available")
        aggregation = await agg_engine.get_aggregation(req.aggregation_id)
        if not aggregation:
            raise HTTPException(status_code=404, detail=f"Aggregation {req.aggregation_id} not found")
        gaps = await gap_engine.analyze_gaps(aggregation, req.plan_id)
        return [g.model_dump(mode="json") for g in gaps]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_gaps failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gaps/{plan_id}", summary="Get gaps for a plan")
async def get_gaps(plan_id: str) -> List[Dict[str, Any]]:
    """Get identified gaps for a plan."""
    try:
        service = get_service()
        plan = await service.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return [g.model_dump(mode="json") for g in plan.gaps]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_gaps failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Action Generation Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate-actions", summary="Generate SMART actions")
async def generate_actions(req: GenerateActionsRequest) -> List[Dict[str, Any]]:
    """Generate SMART improvement actions for a plan's gaps."""
    try:
        service = get_service()
        plan = await service.get_plan(req.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {req.plan_id} not found")
        engine = service._engines.get("action_generator")
        if not engine:
            raise HTTPException(status_code=503, detail="ActionGenerator not available")
        actions = await engine.generate_actions(plan.gaps, req.plan_id)
        return [a.model_dump(mode="json") for a in actions]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_actions failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions/{plan_id}", summary="Get actions for a plan")
async def get_actions(plan_id: str) -> List[Dict[str, Any]]:
    """Get improvement actions for a plan."""
    try:
        service = get_service()
        plan = await service.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return [a.model_dump(mode="json") for a in plan.actions]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_actions failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-action-status", summary="Update action status")
async def update_action_status(req: UpdateActionStatusRequest) -> Dict[str, Any]:
    """Update the status of an improvement action."""
    try:
        service = get_service()
        result = await service.update_action_status(
            req.plan_id, req.action_id, req.new_status
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Action not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("update_action_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Root Cause Analysis Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze-root-causes", summary="Perform root cause analysis")
async def analyze_root_causes(req: AnalyzeRootCausesRequest) -> List[Dict[str, Any]]:
    """Perform 5-Whys root cause analysis for a plan's gaps."""
    try:
        service = get_service()
        plan = await service.get_plan(req.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {req.plan_id} not found")
        engine = service._engines.get("root_cause_mapper")
        if not engine:
            raise HTTPException(status_code=503, detail="RootCauseMapper not available")
        causes = await engine.analyze_root_causes(plan.gaps, req.plan_id)
        return [rc.model_dump(mode="json") for rc in causes]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("analyze_root_causes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/root-causes/{plan_id}", summary="Get root causes for a plan")
async def get_root_causes(plan_id: str) -> List[Dict[str, Any]]:
    """Get root causes identified for a plan."""
    try:
        service = get_service()
        plan = await service.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return [rc.model_dump(mode="json") for rc in plan.root_causes]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_root_causes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-fishbone", summary="Build fishbone analysis")
async def build_fishbone(req: BuildFishboneRequest) -> Dict[str, Any]:
    """Build a fishbone (Ishikawa) diagram analysis for a gap."""
    try:
        service = get_service()
        result = await service.build_fishbone(req.plan_id, req.gap_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Plan or gap not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("build_fishbone failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Prioritization Endpoints
# ---------------------------------------------------------------------------


@router.post("/prioritize-actions", summary="Prioritize plan actions")
async def prioritize_actions(req: PrioritizeActionsRequest) -> List[Dict[str, Any]]:
    """Run Eisenhower + risk-based prioritization on plan actions."""
    try:
        service = get_service()
        plan = await service.get_plan(req.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {req.plan_id} not found")
        engine = service._engines.get("prioritization_engine")
        if not engine:
            raise HTTPException(status_code=503, detail="PrioritizationEngine not available")
        ranked = await engine.prioritize_actions(plan.actions, plan.gaps, plan.root_causes)
        return [a.model_dump(mode="json") for a in ranked]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("prioritize_actions failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Progress Tracking Endpoints
# ---------------------------------------------------------------------------


@router.post("/capture-progress/{plan_id}", summary="Capture progress snapshot")
async def capture_progress(plan_id: str) -> Dict[str, Any]:
    """Capture a point-in-time progress snapshot for a plan."""
    try:
        service = get_service()
        result = await service.capture_progress(plan_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("capture_progress failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{plan_id}", summary="Get progress history")
async def get_progress(plan_id: str) -> List[Dict[str, Any]]:
    """Get progress snapshot history for a plan."""
    try:
        service = get_service()
        results = await service.get_progress_snapshots(plan_id)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("get_progress failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-overdue/{plan_id}", summary="Check for overdue actions")
async def check_overdue(plan_id: str) -> List[Dict[str, Any]]:
    """Check for overdue actions in a plan."""
    try:
        service = get_service()
        results = await service.check_overdue(plan_id)
        return [a.model_dump(mode="json") for a in results]
    except Exception as e:
        logger.error("check_overdue failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review-effectiveness/{plan_id}", summary="Review action effectiveness")
async def review_effectiveness(plan_id: str) -> Dict[str, Any]:
    """Review effectiveness of completed actions in a plan."""
    try:
        service = get_service()
        return await service.review_effectiveness(plan_id)
    except Exception as e:
        logger.error("review_effectiveness failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Stakeholder Coordination Endpoints
# ---------------------------------------------------------------------------


@router.post("/assign-stakeholders", summary="Assign stakeholders to action")
async def assign_stakeholders(req: AssignStakeholdersRequest) -> List[Dict[str, Any]]:
    """Assign stakeholders with RACI roles to an action."""
    try:
        service = get_service()
        results = await service.assign_stakeholders(
            req.plan_id, req.action_id, req.stakeholders
        )
        return [r.model_dump(mode="json") for r in results]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("assign_stakeholders failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-notification", summary="Send stakeholder notification")
async def send_notification(req: SendNotificationRequest) -> Dict[str, Any]:
    """Send a notification to a stakeholder."""
    try:
        service = get_service()
        result = await service.send_notification(
            req.action_id, req.stakeholder_id, req.subject, req.body
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("send_notification failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-bulk-notifications", summary="Send bulk notifications")
async def send_bulk_notifications(req: SendBulkNotificationsRequest) -> List[Dict[str, Any]]:
    """Send notifications to all stakeholders for an action."""
    try:
        service = get_service()
        plan = await service.get_plan(req.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {req.plan_id} not found")
        action = None
        for a in plan.actions:
            if a.action_id == req.action_id:
                action = a
                break
        if not action:
            raise HTTPException(status_code=404, detail=f"Action {req.action_id} not found")
        engine = service._engines.get("stakeholder_coordinator")
        if not engine:
            raise HTTPException(status_code=503, detail="StakeholderCoordinator not available")
        results = await engine.send_bulk_notifications(action, req.subject, req.body)
        return [r.model_dump(mode="json") for r in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("send_bulk_notifications failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/acknowledge/{action_id}/{stakeholder_id}",
    summary="Acknowledge assignment",
)
async def acknowledge_assignment(action_id: str, stakeholder_id: str) -> Dict[str, Any]:
    """Record stakeholder acknowledgment of assignment."""
    try:
        service = get_service()
        engine = service._engines.get("stakeholder_coordinator")
        if not engine:
            raise HTTPException(status_code=503, detail="StakeholderCoordinator not available")
        result = await engine.acknowledge(action_id, stakeholder_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Assignment not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("acknowledge failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending-acknowledgments/{plan_id}", summary="Get pending acknowledgments")
async def get_pending_acknowledgments(plan_id: str) -> List[Dict[str, Any]]:
    """Get stakeholders who have not yet acknowledged their assignments."""
    try:
        service = get_service()
        plan = await service.get_plan(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        engine = service._engines.get("stakeholder_coordinator")
        if not engine:
            return []
        results = await engine.get_pending_acknowledgments(plan.actions)
        return [r.model_dump(mode="json") for r in results]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pending_ack failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Report Generation Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate-report/{plan_id}", summary="Generate plan report")
async def generate_report(plan_id: str) -> Dict[str, Any]:
    """Generate a comprehensive improvement plan report."""
    try:
        service = get_service()
        result = await service.generate_report(plan_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("generate_report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Dashboard & Health Endpoints
# ---------------------------------------------------------------------------


@router.get("/dashboard", summary="Dashboard data")
async def dashboard(
    operator_id: str = Query(..., description="Operator identifier"),
) -> Dict[str, Any]:
    """Get aggregated dashboard data for an operator."""
    try:
        service = get_service()
        return await service.get_dashboard(operator_id)
    except Exception as e:
        logger.error("dashboard failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Return agent health status."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)[:200]}
