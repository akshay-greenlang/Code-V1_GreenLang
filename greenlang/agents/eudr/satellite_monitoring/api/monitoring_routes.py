# -*- coding: utf-8 -*-
"""
Monitoring Routes - AGENT-EUDR-003 Satellite Monitoring API

Endpoints for continuous monitoring schedule management including
creation, retrieval, update, deletion, result history, and manual
execution of monitoring schedules.

Endpoints:
    POST   /schedule                - Create monitoring schedule
    GET    /schedule/{schedule_id}  - Get schedule details
    PUT    /schedule/{schedule_id}  - Update schedule
    DELETE /schedule/{schedule_id}  - Delete schedule
    GET    /results/{plot_id}       - Get monitoring results
    POST   /execute                 - Trigger manual monitoring execution

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.satellite_monitoring.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_continuous_monitor,
    get_pagination,
    get_satellite_service,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.satellite_monitoring.api.schemas import (
    CreateMonitoringApiRequest,
    MonitoringExecuteRequest,
    MonitoringResultResponse,
    MonitoringScheduleResponse,
    PaginatedMeta,
    UpdateMonitoringApiRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Continuous Monitoring"])


# ---------------------------------------------------------------------------
# In-memory schedule store (replaced by database in production)
# ---------------------------------------------------------------------------

_schedule_store: Dict[str, Dict[str, Any]] = {}


def _get_schedule_store() -> Dict[str, Dict[str, Any]]:
    """Return the schedule store. Replaceable for testing."""
    return _schedule_store


# ---------------------------------------------------------------------------
# POST /schedule
# ---------------------------------------------------------------------------


@router.post(
    "/schedule",
    response_model=MonitoringScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create monitoring schedule",
    description=(
        "Create a continuous monitoring schedule for a production plot. "
        "Configures the interval (daily to quarterly), priority level, "
        "and analysis depth for automated deforestation monitoring."
    ),
    responses={
        201: {"description": "Monitoring schedule created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_monitoring_schedule(
    body: CreateMonitoringApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:monitoring:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> MonitoringScheduleResponse:
    """Create a continuous monitoring schedule for a plot.

    Registers a new monitoring schedule with the continuous monitor.
    The plot will be automatically analyzed at the configured interval.

    Args:
        body: Schedule creation request with plot and monitoring parameters.
        user: Authenticated user with monitoring:write permission.

    Returns:
        MonitoringScheduleResponse with created schedule details.

    Raises:
        HTTPException: 400 if request invalid, 500 on error.
    """
    start = time.monotonic()
    schedule_id = f"mon-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Create monitoring schedule: user=%s plot_id=%s commodity=%s "
        "interval=%s priority=%s",
        user.user_id,
        body.plot_id,
        body.commodity,
        body.interval,
        body.priority,
    )

    try:
        monitor = get_continuous_monitor()

        result = monitor.create_schedule(
            schedule_id=schedule_id,
            plot_id=body.plot_id,
            polygon_vertices=body.polygon_vertices,
            commodity=body.commodity,
            country_code=body.country_code,
            interval=body.interval,
            priority=body.priority,
            analysis_level=body.analysis_level,
            alert_on_change=body.alert_on_change,
        )

        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Store schedule for retrieval
        store = _get_schedule_store()
        store[schedule_id] = {
            "schedule_id": schedule_id,
            "plot_id": body.plot_id,
            "commodity": body.commodity,
            "country_code": body.country_code,
            "interval": body.interval,
            "priority": body.priority,
            "analysis_level": body.analysis_level,
            "active": True,
            "alert_on_change": body.alert_on_change,
            "polygon_vertices": body.polygon_vertices,
            "user_id": user.user_id,
            "operator_id": user.operator_id or user.user_id,
            "total_executions": 0,
            "next_execution": getattr(result, "next_execution", None),
            "last_execution": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Monitoring schedule created: schedule_id=%s plot_id=%s "
            "elapsed_ms=%.1f",
            schedule_id,
            body.plot_id,
            elapsed * 1000,
        )

        return MonitoringScheduleResponse(
            schedule_id=schedule_id,
            plot_id=body.plot_id,
            commodity=body.commodity,
            country_code=body.country_code,
            interval=body.interval,
            priority=body.priority,
            analysis_level=body.analysis_level,
            active=True,
            alert_on_change=body.alert_on_change,
            next_execution=getattr(result, "next_execution", None),
            last_execution=None,
            total_executions=0,
            created_at=now,
            updated_at=now,
        )

    except ValueError as exc:
        logger.warning(
            "Create schedule error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Create schedule failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Monitoring schedule creation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /schedule/{schedule_id}
# ---------------------------------------------------------------------------


@router.get(
    "/schedule/{schedule_id}",
    response_model=MonitoringScheduleResponse,
    summary="Get monitoring schedule details",
    description="Retrieve details of a specific monitoring schedule.",
    responses={
        200: {"description": "Schedule details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Schedule not found"},
    },
)
async def get_monitoring_schedule(
    schedule_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:monitoring:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> MonitoringScheduleResponse:
    """Get monitoring schedule details.

    Args:
        schedule_id: Schedule identifier.
        user: Authenticated user with monitoring:read permission.

    Returns:
        MonitoringScheduleResponse with schedule details.

    Raises:
        HTTPException: 404 if schedule not found, 403 if unauthorized.
    """
    logger.info(
        "Get schedule: user=%s schedule_id=%s",
        user.user_id,
        schedule_id,
    )

    store = _get_schedule_store()
    schedule = store.get(schedule_id)

    if schedule is None:
        try:
            monitor = get_continuous_monitor()
            schedule_data = monitor.get_schedule(schedule_id=schedule_id)
            if schedule_data is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Schedule {schedule_id} not found",
                )
            schedule = schedule_data if isinstance(schedule_data, dict) else {
                "schedule_id": schedule_id,
                "plot_id": getattr(schedule_data, "plot_id", ""),
                "commodity": getattr(schedule_data, "commodity", ""),
                "country_code": getattr(schedule_data, "country_code", ""),
                "interval": getattr(schedule_data, "interval", "monthly"),
                "priority": getattr(schedule_data, "priority", "medium"),
                "analysis_level": getattr(schedule_data, "analysis_level", "standard"),
                "active": getattr(schedule_data, "active", True),
                "alert_on_change": getattr(schedule_data, "alert_on_change", True),
                "operator_id": getattr(schedule_data, "operator_id", ""),
            }
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Schedule {schedule_id} not found",
            )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    schedule_owner = schedule.get("operator_id", schedule.get("user_id", ""))
    if schedule_owner != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this schedule",
        )

    return MonitoringScheduleResponse(
        schedule_id=schedule.get("schedule_id", schedule_id),
        plot_id=schedule.get("plot_id", ""),
        commodity=schedule.get("commodity", ""),
        country_code=schedule.get("country_code", ""),
        interval=schedule.get("interval", "monthly"),
        priority=schedule.get("priority", "medium"),
        analysis_level=schedule.get("analysis_level", "standard"),
        active=schedule.get("active", True),
        alert_on_change=schedule.get("alert_on_change", True),
        next_execution=schedule.get("next_execution"),
        last_execution=schedule.get("last_execution"),
        total_executions=schedule.get("total_executions", 0),
    )


# ---------------------------------------------------------------------------
# PUT /schedule/{schedule_id}
# ---------------------------------------------------------------------------


@router.put(
    "/schedule/{schedule_id}",
    response_model=MonitoringScheduleResponse,
    summary="Update monitoring schedule",
    description=(
        "Update an existing monitoring schedule. Supports updating "
        "interval, priority, analysis level, active status, and "
        "alert configuration."
    ),
    responses={
        200: {"description": "Updated schedule"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Schedule not found"},
    },
)
async def update_monitoring_schedule(
    schedule_id: str,
    body: UpdateMonitoringApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:monitoring:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> MonitoringScheduleResponse:
    """Update an existing monitoring schedule.

    Args:
        schedule_id: Schedule identifier.
        body: Update request with fields to modify.
        user: Authenticated user with monitoring:write permission.

    Returns:
        MonitoringScheduleResponse with updated schedule.

    Raises:
        HTTPException: 404 if schedule not found, 403 if unauthorized.
    """
    logger.info(
        "Update schedule: user=%s schedule_id=%s",
        user.user_id,
        schedule_id,
    )

    store = _get_schedule_store()
    schedule = store.get(schedule_id)

    if schedule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {schedule_id} not found",
        )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    schedule_owner = schedule.get("operator_id", schedule.get("user_id", ""))
    if schedule_owner != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this schedule",
        )

    # Apply updates
    now = datetime.now(timezone.utc).replace(microsecond=0)

    if body.interval is not None:
        schedule["interval"] = body.interval
    if body.priority is not None:
        schedule["priority"] = body.priority
    if body.analysis_level is not None:
        schedule["analysis_level"] = body.analysis_level
    if body.active is not None:
        schedule["active"] = body.active
    if body.alert_on_change is not None:
        schedule["alert_on_change"] = body.alert_on_change
    schedule["updated_at"] = now.isoformat()

    # Notify continuous monitor of changes
    try:
        monitor = get_continuous_monitor()
        monitor.update_schedule(
            schedule_id=schedule_id,
            interval=body.interval,
            priority=body.priority,
            analysis_level=body.analysis_level,
            active=body.active,
            alert_on_change=body.alert_on_change,
        )
    except Exception as exc:
        logger.warning(
            "Monitor update notification failed: %s", exc
        )

    logger.info(
        "Schedule updated: schedule_id=%s plot_id=%s",
        schedule_id,
        schedule.get("plot_id", ""),
    )

    return MonitoringScheduleResponse(
        schedule_id=schedule_id,
        plot_id=schedule.get("plot_id", ""),
        commodity=schedule.get("commodity", ""),
        country_code=schedule.get("country_code", ""),
        interval=schedule.get("interval", "monthly"),
        priority=schedule.get("priority", "medium"),
        analysis_level=schedule.get("analysis_level", "standard"),
        active=schedule.get("active", True),
        alert_on_change=schedule.get("alert_on_change", True),
        next_execution=schedule.get("next_execution"),
        last_execution=schedule.get("last_execution"),
        total_executions=schedule.get("total_executions", 0),
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# DELETE /schedule/{schedule_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/schedule/{schedule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete monitoring schedule",
    description="Delete a monitoring schedule. Stops all future executions.",
    responses={
        204: {"description": "Schedule deleted"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Schedule not found"},
    },
)
async def delete_monitoring_schedule(
    schedule_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:monitoring:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> None:
    """Delete a monitoring schedule.

    Args:
        schedule_id: Schedule identifier.
        user: Authenticated user with monitoring:write permission.

    Raises:
        HTTPException: 404 if schedule not found, 403 if unauthorized.
    """
    logger.info(
        "Delete schedule: user=%s schedule_id=%s",
        user.user_id,
        schedule_id,
    )

    store = _get_schedule_store()
    schedule = store.get(schedule_id)

    if schedule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {schedule_id} not found",
        )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    schedule_owner = schedule.get("operator_id", schedule.get("user_id", ""))
    if schedule_owner != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this schedule",
        )

    # Remove from pipeline
    try:
        monitor = get_continuous_monitor()
        monitor.delete_schedule(schedule_id=schedule_id)
    except Exception as exc:
        logger.warning(
            "Monitor delete notification failed: %s", exc
        )

    del store[schedule_id]

    logger.info(
        "Schedule deleted: schedule_id=%s plot_id=%s",
        schedule_id,
        schedule.get("plot_id", ""),
    )


# ---------------------------------------------------------------------------
# GET /results/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/results/{plot_id}",
    response_model=Dict[str, Any],
    summary="Get monitoring results for a plot",
    description=(
        "Retrieve paginated monitoring execution results for a production "
        "plot. Returns individual execution results sorted by date descending."
    ),
    responses={
        200: {"description": "Monitoring results"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_monitoring_results(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:monitoring:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get paginated monitoring results for a plot.

    Args:
        plot_id: Plot identifier.
        pagination: Pagination parameters.
        user: Authenticated user with monitoring:read permission.

    Returns:
        Dictionary with monitoring result items and pagination metadata.
    """
    logger.info(
        "Monitoring results: user=%s plot_id=%s limit=%d offset=%d",
        user.user_id,
        plot_id,
        pagination.limit,
        pagination.offset,
    )

    try:
        monitor = get_continuous_monitor()

        results = monitor.get_results(
            plot_id=plot_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        items = getattr(results, "items", [])
        total = getattr(results, "total", len(items))

        return {
            "plot_id": plot_id,
            "items": [
                {
                    "result_id": getattr(item, "result_id", ""),
                    "schedule_id": getattr(item, "schedule_id", ""),
                    "execution_date": str(getattr(item, "execution_date", "")),
                    "deforestation_detected": getattr(item, "deforestation_detected", False),
                    "change_classification": getattr(item, "change_classification", "no_change"),
                    "ndvi_current": getattr(item, "ndvi_current", 0.0),
                    "ndvi_delta": getattr(item, "ndvi_delta", 0.0),
                    "confidence": getattr(item, "confidence", 0.0),
                    "forest_loss_ha": getattr(item, "forest_loss_ha", 0.0),
                    "alerts_generated": getattr(item, "alerts_generated", 0),
                }
                for item in items
            ],
            "meta": {
                "total": total,
                "limit": pagination.limit,
                "offset": pagination.offset,
                "has_more": (pagination.offset + pagination.limit) < total,
            },
        }

    except Exception as exc:
        logger.error(
            "Monitoring results failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        return {
            "plot_id": plot_id,
            "items": [],
            "meta": {
                "total": 0,
                "limit": pagination.limit,
                "offset": pagination.offset,
                "has_more": False,
            },
        }


# ---------------------------------------------------------------------------
# POST /execute
# ---------------------------------------------------------------------------


@router.post(
    "/execute",
    response_model=MonitoringResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger manual monitoring execution",
    description=(
        "Manually trigger an immediate monitoring execution for a "
        "specific schedule. Returns the execution result directly."
    ),
    responses={
        200: {"description": "Execution result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Schedule not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def execute_monitoring(
    body: MonitoringExecuteRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:monitoring:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> MonitoringResultResponse:
    """Trigger manual monitoring execution for a schedule.

    Args:
        body: Execute request with schedule_id.
        user: Authenticated user with monitoring:write permission.

    Returns:
        MonitoringResultResponse with execution results.

    Raises:
        HTTPException: 404 if schedule not found, 500 on error.
    """
    start = time.monotonic()
    result_id = f"res-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Manual execution: user=%s schedule_id=%s",
        user.user_id,
        body.schedule_id,
    )

    # Verify schedule exists
    store = _get_schedule_store()
    schedule = store.get(body.schedule_id)

    if schedule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule {body.schedule_id} not found",
        )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    schedule_owner = schedule.get("operator_id", schedule.get("user_id", ""))
    if schedule_owner != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to execute this schedule",
        )

    try:
        monitor = get_continuous_monitor()

        result = monitor.execute(schedule_id=body.schedule_id)

        elapsed = time.monotonic() - start
        now = datetime.now(timezone.utc).replace(microsecond=0)

        # Update schedule counters
        schedule["total_executions"] = schedule.get("total_executions", 0) + 1
        schedule["last_execution"] = now.isoformat()

        logger.info(
            "Manual execution completed: schedule_id=%s plot_id=%s "
            "deforestation=%s elapsed_ms=%.1f",
            body.schedule_id,
            schedule.get("plot_id", ""),
            getattr(result, "deforestation_detected", False),
            elapsed * 1000,
        )

        return MonitoringResultResponse(
            result_id=getattr(result, "result_id", result_id),
            schedule_id=body.schedule_id,
            plot_id=schedule.get("plot_id", ""),
            execution_date=now,
            deforestation_detected=getattr(result, "deforestation_detected", False),
            change_classification=getattr(result, "change_classification", "no_change"),
            ndvi_current=getattr(result, "ndvi_current", 0.0),
            ndvi_delta=getattr(result, "ndvi_delta", 0.0),
            confidence=getattr(result, "confidence", 0.0),
            forest_loss_ha=getattr(result, "forest_loss_ha", 0.0),
            alerts_generated=getattr(result, "alerts_generated", 0),
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Execution error: user=%s schedule_id=%s error=%s",
            user.user_id,
            body.schedule_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Execution failed: user=%s schedule_id=%s error=%s",
            user.user_id,
            body.schedule_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Monitoring execution failed due to an internal error",
        )
