# -*- coding: utf-8 -*-
"""
FastAPI Router - AGENT-EUDR-034: Annual Review Scheduler Agent

REST API endpoints for review cycle management, deadline tracking,
checklist generation, entity coordination, year-over-year comparison,
calendar management, and notification dispatch.

Endpoint Summary (30+):
    POST /create-cycle                             - Create a review cycle
    POST /schedule-tasks/{cycle_id}                - Schedule tasks for a cycle
    POST /update-cycle-status/{cycle_id}           - Update cycle status
    GET  /cycles                                   - List review cycles
    GET  /cycles/{cycle_id}                        - Get cycle details
    GET  /active-cycles                            - Get active cycles
    POST /update-task-status                       - Update a task status
    POST /register-deadline                        - Register a deadline
    POST /check-deadlines                          - Check approaching deadlines
    POST /submit-to-authority                      - Submit to competent authority
    GET  /submissions/{submission_id}              - Track submission status
    GET  /deadline-records                         - List deadline records
    POST /generate-checklist                       - Generate a review checklist
    POST /customize-checklist                      - Customize checklist for commodity
    POST /track-checklist-completion               - Track checklist item completion
    GET  /checklists                               - List checklists
    GET  /checklists/{checklist_id}                - Get checklist details
    POST /identify-entities                        - Identify review entities
    POST /cascade-reviews                          - Cascade reviews to child entities
    POST /track-dependencies                       - Track entity dependencies
    POST /aggregate-completion                     - Aggregate entity completion
    GET  /coordinations                            - List coordination records
    POST /compare-years                            - Compare year-over-year data
    POST /comparison-report/{comparison_id}        - Generate comparison report
    GET  /comparisons                              - List comparison records
    POST /add-calendar-event                       - Add a calendar event
    GET  /upcoming-events                          - Get upcoming calendar events
    POST /generate-ical                            - Generate iCal export
    GET  /calendar-records                         - List calendar records
    POST /send-notification                        - Send a notification
    POST /send-notification-batch                  - Send batch notifications
    POST /acknowledge-notification/{notification_id} - Acknowledge a notification
    POST /escalate-overdue                         - Escalate overdue notifications
    GET  /notification-records                     - List notification records
    POST /generate-summary                         - Generate review summary
    GET  /health                                   - Health check
    GET  /dashboard                                - Dashboard data

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (GL-EUDR-ARS-034)
Status: Production Ready
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from greenlang.agents.eudr.annual_review_scheduler.setup import get_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class CreateCycleRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(..., description="Year under review")
    commodities: Optional[List[str]] = Field(None, description="Commodities covered")
    start_date: Optional[str] = Field(None, description="Explicit start date (ISO)")

class ScheduleTasksRequest(BaseModel):
    additional_tasks: Optional[List[Dict[str, Any]]] = Field(None, description="Custom tasks")

class UpdateCycleStatusRequest(BaseModel):
    new_status: str = Field(..., description="Target status")

class UpdateTaskStatusRequest(BaseModel):
    cycle_id: str = Field(..., description="Cycle identifier")
    task_id: str = Field(..., description="Task identifier")
    new_status: str = Field(..., description="New task status")

class RegisterDeadlineRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    deadline_type: str = Field(..., description="Type of deadline")
    title: str = Field(..., description="Deadline title")
    due_date: str = Field(..., description="Due date (ISO)")
    article_reference: str = Field(default="", description="EUDR article reference")
    responsible_entity: Optional[str] = None
    review_year: int = Field(default=0, description="Review year")

class CheckDeadlinesRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    review_year: Optional[int] = None

class SubmitToAuthorityRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    deadline_id: str = Field(..., description="Deadline being fulfilled")
    submission_data: Dict[str, Any] = Field(..., description="Submission payload")

class GenerateChecklistRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    commodity: str = Field(default="general", description="Commodity type")
    cycle_id: str = Field(default="", description="Associated cycle ID")
    custom_items: Optional[List[Dict[str, Any]]] = None

class CustomizeChecklistRequest(BaseModel):
    checklist_id: str = Field(..., description="Checklist identifier")
    commodity: str = Field(..., description="Commodity to add items for")

class TrackChecklistCompletionRequest(BaseModel):
    checklist_id: str = Field(..., description="Checklist identifier")
    item_id: str = Field(..., description="Item identifier")
    new_status: str = Field(..., description="New status")
    notes: str = Field(default="", description="Reviewer notes")

class IdentifyEntitiesRequest(BaseModel):
    operator_id: str = Field(..., description="Root operator identifier")
    entities: List[Dict[str, Any]] = Field(..., description="Entity data list")
    cycle_id: str = Field(default="", description="Associated cycle ID")

class CascadeReviewsRequest(BaseModel):
    coordination_id: str = Field(..., description="Coordination record ID")
    child_entities: Optional[List[Dict[str, Any]]] = None

class TrackDependenciesRequest(BaseModel):
    coordination_id: str = Field(..., description="Coordination record ID")

class AggregateCompletionRequest(BaseModel):
    coordination_id: str = Field(..., description="Coordination record ID")

class CompareYearsRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    data_points: List[Dict[str, Any]] = Field(..., description="Year data points")

class AddCalendarEventRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    event_type: str = Field(..., description="Event type")
    title: str = Field(..., description="Event title")
    start_date: str = Field(..., description="Start date (ISO)")
    end_date: Optional[str] = None
    description: str = Field(default="", description="Event description")
    all_day: bool = Field(default=True)
    review_year: int = Field(default=0)

class GenerateIcalRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    review_year: Optional[int] = None

class SendNotificationRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    channel: str = Field(default="email", description="Delivery channel")
    recipient: str = Field(..., description="Recipient identifier")
    subject: str = Field(..., description="Subject line")
    body: str = Field(..., description="Message body")
    cycle_id: str = Field(default="")

class SendNotificationBatchRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    notifications: List[Dict[str, Any]] = Field(..., description="Notification list")
    cycle_id: str = Field(default="")

class EscalateOverdueRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    cycle_id: str = Field(default="")
    hours_threshold: Optional[int] = None

class GenerateSummaryRequest(BaseModel):
    operator_id: str = Field(..., description="Operator identifier")
    review_year: int = Field(default=0)

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")
    error_code: str = Field(default="internal_error")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/eudr/annual-review-scheduler",
    tags=["EUDR Annual Review Scheduler"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)


# ---------------------------------------------------------------------------
# Review Cycle Endpoints
# ---------------------------------------------------------------------------

@router.post("/create-cycle", summary="Create a review cycle")
async def create_cycle(req: CreateCycleRequest) -> Dict[str, Any]:
    """Create a new annual review cycle."""
    try:
        service = get_service()
        start_date = None
        if req.start_date:
            start_date = datetime.fromisoformat(req.start_date)
        result = await service.create_review_cycle(
            operator_id=req.operator_id,
            review_year=req.review_year,
            commodities=req.commodities,
            start_date=start_date,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("create_cycle failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule-tasks/{cycle_id}", summary="Schedule tasks for a cycle")
async def schedule_tasks(cycle_id: str, req: ScheduleTasksRequest) -> Dict[str, Any]:
    """Generate and schedule review tasks for a cycle."""
    try:
        service = get_service()
        result = await service.schedule_tasks(cycle_id, req.additional_tasks)
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("schedule_tasks failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-cycle-status/{cycle_id}", summary="Update cycle status")
async def update_cycle_status(cycle_id: str, req: UpdateCycleStatusRequest) -> Dict[str, Any]:
    """Transition a review cycle to a new status."""
    try:
        service = get_service()
        from greenlang.agents.eudr.annual_review_scheduler.models import ReviewCycleStatus
        result = await service.update_cycle_status(
            cycle_id, ReviewCycleStatus(req.new_status),
        )
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("update_cycle_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cycles", summary="List review cycles")
async def list_cycles(
    operator_id: Optional[str] = Query(None),
    review_year: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List review cycles with optional filters."""
    try:
        service = get_service()
        results = await service.list_cycles(operator_id, review_year, status, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_cycles failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cycles/{cycle_id}", summary="Get cycle details")
async def get_cycle(cycle_id: str) -> Dict[str, Any]:
    """Get a specific review cycle by ID."""
    try:
        service = get_service()
        result = await service.get_cycle(cycle_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Cycle {cycle_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_cycle failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-cycles", summary="Get active review cycles")
async def get_active_cycles(
    operator_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """Get all active review cycles."""
    try:
        service = get_service()
        results = await service.get_active_cycles(operator_id)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("get_active_cycles failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-task-status", summary="Update task status")
async def update_task_status(req: UpdateTaskStatusRequest) -> Dict[str, Any]:
    """Update the status of a task within a cycle."""
    try:
        service = get_service()
        from greenlang.agents.eudr.annual_review_scheduler.models import ChecklistItemStatus
        result = await service.update_task_status(
            req.cycle_id, req.task_id, ChecklistItemStatus(req.new_status),
        )
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("update_task_status failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Deadline Endpoints
# ---------------------------------------------------------------------------

@router.post("/register-deadline", summary="Register a deadline")
async def register_deadline(req: RegisterDeadlineRequest) -> Dict[str, Any]:
    """Register a new deadline for tracking."""
    try:
        service = get_service()
        from greenlang.agents.eudr.annual_review_scheduler.models import DeadlineType
        due_date = datetime.fromisoformat(req.due_date)
        result = await service.register_deadline(
            operator_id=req.operator_id,
            deadline_type=DeadlineType(req.deadline_type),
            title=req.title,
            due_date=due_date,
            article_reference=req.article_reference,
            responsible_entity=req.responsible_entity,
            review_year=req.review_year,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("register_deadline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-deadlines", summary="Check approaching deadlines")
async def check_deadlines(req: CheckDeadlinesRequest) -> Dict[str, Any]:
    """Check all deadlines for approaching and overdue items."""
    try:
        service = get_service()
        result = await service.check_approaching_deadlines(
            req.operator_id, req.review_year,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("check_deadlines failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-to-authority", summary="Submit to competent authority")
async def submit_to_authority(req: SubmitToAuthorityRequest) -> Dict[str, Any]:
    """Submit compliance documentation to competent authority."""
    try:
        service = get_service()
        return await service.submit_to_authority(
            req.operator_id, req.deadline_id, req.submission_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("submit_to_authority failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/submissions/{submission_id}", summary="Track submission status")
async def track_submission(submission_id: str) -> Dict[str, Any]:
    """Track the status of a previous submission."""
    try:
        service = get_service()
        result = await service.track_submission_status(submission_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("track_submission failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deadline-records", summary="List deadline records")
async def list_deadline_records(
    operator_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List deadline tracking records."""
    try:
        service = get_service()
        results = await service.list_deadline_records(operator_id, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_deadline_records failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Checklist Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-checklist", summary="Generate a review checklist")
async def generate_checklist(req: GenerateChecklistRequest) -> Dict[str, Any]:
    """Generate a commodity-specific review checklist."""
    try:
        service = get_service()
        result = await service.generate_checklist(
            req.operator_id, req.commodity, req.cycle_id, req.custom_items,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("generate_checklist failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/customize-checklist", summary="Customize checklist for commodity")
async def customize_checklist(req: CustomizeChecklistRequest) -> Dict[str, Any]:
    """Add commodity-specific items to an existing checklist."""
    try:
        service = get_service()
        result = await service.customize_for_commodity(req.checklist_id, req.commodity)
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("customize_checklist failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-checklist-completion", summary="Track checklist item completion")
async def track_checklist_completion(req: TrackChecklistCompletionRequest) -> Dict[str, Any]:
    """Update the status of a checklist item."""
    try:
        service = get_service()
        from greenlang.agents.eudr.annual_review_scheduler.models import ChecklistItemStatus
        result = await service.track_checklist_completion(
            req.checklist_id, req.item_id, ChecklistItemStatus(req.new_status), req.notes,
        )
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("track_checklist_completion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checklists", summary="List checklists")
async def list_checklists(
    operator_id: Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List generated checklists."""
    try:
        service = get_service()
        results = await service.list_checklists(operator_id, commodity, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_checklists failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checklists/{checklist_id}", summary="Get checklist details")
async def get_checklist(checklist_id: str) -> Dict[str, Any]:
    """Get a specific checklist by ID."""
    try:
        service = get_service()
        result = await service.get_checklist(checklist_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Checklist {checklist_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_checklist failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entity Coordination Endpoints
# ---------------------------------------------------------------------------

@router.post("/identify-entities", summary="Identify review entities")
async def identify_entities(req: IdentifyEntitiesRequest) -> Dict[str, Any]:
    """Identify entities requiring review."""
    try:
        service = get_service()
        result = await service.identify_review_entities(
            req.operator_id, req.entities, req.cycle_id,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("identify_entities failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cascade-reviews", summary="Cascade reviews to child entities")
async def cascade_reviews(req: CascadeReviewsRequest) -> Dict[str, Any]:
    """Cascade review requirements to child entities."""
    try:
        service = get_service()
        result = await service.cascade_reviews(
            req.coordination_id, req.child_entities,
        )
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("cascade_reviews failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track-dependencies", summary="Track entity dependencies")
async def track_dependencies(req: TrackDependenciesRequest) -> Dict[str, Any]:
    """Evaluate dependency status for coordinated entities."""
    try:
        service = get_service()
        result = await service.track_dependencies(req.coordination_id)
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("track_dependencies failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/aggregate-completion", summary="Aggregate entity completion")
async def aggregate_completion(req: AggregateCompletionRequest) -> Dict[str, Any]:
    """Aggregate completion status across coordinated entities."""
    try:
        service = get_service()
        result = await service.aggregate_completion(req.coordination_id)
        return result.model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("aggregate_completion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coordinations", summary="List coordination records")
async def list_coordinations(
    operator_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List entity coordination records."""
    try:
        service = get_service()
        results = await service.list_coordination_records(operator_id, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_coordinations failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Year Comparison Endpoints
# ---------------------------------------------------------------------------

@router.post("/compare-years", summary="Compare year-over-year data")
async def compare_years(req: CompareYearsRequest) -> Dict[str, Any]:
    """Compare EUDR compliance data across multiple years."""
    try:
        service = get_service()
        result = await service.compare_years(req.operator_id, req.data_points)
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("compare_years failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comparison-report/{comparison_id}", summary="Generate comparison report")
async def generate_comparison_report(comparison_id: str) -> Dict[str, Any]:
    """Generate a detailed year-over-year comparison report."""
    try:
        service = get_service()
        return await service.generate_comparison_report(comparison_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("comparison_report failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparisons", summary="List comparison records")
async def list_comparisons(
    operator_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List year comparison records."""
    try:
        service = get_service()
        results = await service.list_comparison_records(operator_id, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_comparisons failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Calendar Endpoints
# ---------------------------------------------------------------------------

@router.post("/add-calendar-event", summary="Add a calendar event")
async def add_calendar_event(req: AddCalendarEventRequest) -> Dict[str, Any]:
    """Add an event to the compliance calendar."""
    try:
        service = get_service()
        from greenlang.agents.eudr.annual_review_scheduler.models import CalendarEventType
        start_date = datetime.fromisoformat(req.start_date)
        end_date = datetime.fromisoformat(req.end_date) if req.end_date else None
        result = await service.add_calendar_event(
            operator_id=req.operator_id,
            event_type=CalendarEventType(req.event_type),
            title=req.title,
            start_date=start_date,
            end_date=end_date,
            description=req.description,
            all_day=req.all_day,
            review_year=req.review_year,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("add_calendar_event failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming-events", summary="Get upcoming calendar events")
async def get_upcoming_events(
    operator_id: Optional[str] = Query(None),
    days_ahead: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    """Get upcoming events within a specified time window."""
    try:
        service = get_service()
        results = await service.get_upcoming_events(operator_id, days_ahead, limit=limit)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("get_upcoming_events failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-ical", summary="Generate iCal export")
async def generate_ical(req: GenerateIcalRequest) -> Dict[str, Any]:
    """Generate iCal-formatted calendar data."""
    try:
        service = get_service()
        ical_data = await service.generate_ical(req.operator_id, req.review_year)
        return {"operator_id": req.operator_id, "ical_data": ical_data}
    except Exception as e:
        logger.error("generate_ical failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar-records", summary="List calendar records")
async def list_calendar_records(
    operator_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List calendar records."""
    try:
        service = get_service()
        results = await service.list_calendar_records(operator_id, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_calendar_records failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Notification Endpoints
# ---------------------------------------------------------------------------

@router.post("/send-notification", summary="Send a notification")
async def send_notification(req: SendNotificationRequest) -> Dict[str, Any]:
    """Send a notification via a specified channel."""
    try:
        service = get_service()
        from greenlang.agents.eudr.annual_review_scheduler.models import NotificationChannel
        result = await service.send_notification(
            operator_id=req.operator_id,
            channel=NotificationChannel(req.channel),
            recipient=req.recipient,
            subject=req.subject,
            body=req.body,
            cycle_id=req.cycle_id,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("send_notification failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-notification-batch", summary="Send batch notifications")
async def send_notification_batch(req: SendNotificationBatchRequest) -> Dict[str, Any]:
    """Send a batch of notifications."""
    try:
        service = get_service()
        result = await service.send_notification_batch(
            req.operator_id, req.notifications, req.cycle_id,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("send_notification_batch failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acknowledge-notification/{notification_id}", summary="Acknowledge notification")
async def acknowledge_notification(notification_id: str) -> Dict[str, Any]:
    """Acknowledge a received notification."""
    try:
        service = get_service()
        result = await service.acknowledge_notification(notification_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Notification {notification_id} not found")
        return result.model_dump(mode="json")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("acknowledge_notification failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/escalate-overdue", summary="Escalate overdue notifications")
async def escalate_overdue(req: EscalateOverdueRequest) -> Dict[str, Any]:
    """Escalate unacknowledged notifications to the next tier."""
    try:
        service = get_service()
        result = await service.escalate_overdue(
            req.operator_id, req.cycle_id, req.hours_threshold,
        )
        return result.model_dump(mode="json")
    except Exception as e:
        logger.error("escalate_overdue failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notification-records", summary="List notification records")
async def list_notification_records(
    operator_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List notification batch records."""
    try:
        service = get_service()
        results = await service.list_notification_records(operator_id, limit, offset)
        return [r.model_dump(mode="json") for r in results]
    except Exception as e:
        logger.error("list_notification_records failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Summary & Health Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate-summary", summary="Generate review summary")
async def generate_summary(req: GenerateSummaryRequest) -> Dict[str, Any]:
    """Generate an overall review summary."""
    try:
        service = get_service()
        return await service.generate_summary(req.operator_id, req.review_year)
    except Exception as e:
        logger.error("generate_summary failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", summary="Health check")
async def health_check() -> Dict[str, Any]:
    """Return agent health status."""
    try:
        service = get_service()
        return await service.health_check()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)[:200]}


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
