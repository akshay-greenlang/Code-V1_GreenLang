"""
Agent Execution Router

This module provides REST API endpoints for agent execution:
- POST /v1/agents/{agent_id}/execute - Execute agent
- GET /v1/agents/{agent_id}/executions - List executions
- GET /v1/agents/{agent_id}/executions/{execution_id} - Get execution
- GET /v1/agents/{agent_id}/executions/{execution_id}/result - Get result
- POST /v1/agents/{agent_id}/executions/{execution_id}/cancel - Cancel
- GET /v1/executions/metrics - Get execution metrics
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, Field

from services.execution import (
    AgentExecutionService,
    ExecutionContext,
    ExecutionResult,
    ExecutionStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton execution service instance
# In production, this would be injected via dependency injection
_execution_service: Optional[AgentExecutionService] = None


def get_execution_service() -> AgentExecutionService:
    """
    Get or create the execution service singleton.

    Returns:
        AgentExecutionService instance
    """
    global _execution_service
    if _execution_service is None:
        _execution_service = AgentExecutionService(
            config={
                "agent_defaults": {
                    "enable_provenance": True,
                    "enable_cost_tracking": True,
                },
            }
        )
    return _execution_service


# Request/Response Models

class ExecutionRequest(BaseModel):
    """Request to execute an agent."""

    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    version: Optional[str] = Field(None, description="Specific version to execute")
    async_mode: bool = Field(False, description="Run asynchronously")
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Execution timeout")
    priority: int = Field(5, ge=1, le=10, description="Execution priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    # Tenant and user info (in production, from auth token)
    tenant_id: str = Field("default-tenant", description="Tenant ID")
    user_id: str = Field("default-user", description="User ID")


class ExecutionResponse(BaseModel):
    """Execution response model."""

    execution_id: str
    agent_id: str
    status: str
    progress: Optional[int] = Field(None, description="Progress percentage (0-100)")
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


class ExecutionResultResponse(BaseModel):
    """Execution result response model."""

    execution_id: str
    agent_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    provenance_hash: str
    cost: Optional[Dict[str, float]] = None
    metrics: Dict[str, Any]


class ExecutionListResponse(BaseModel):
    """Paginated list of executions."""

    data: List[ExecutionResponse]
    meta: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Execution metrics response."""

    counters: Dict[str, int]
    gauges: Dict[str, float]


# Helper functions

async def execute_agent_background(
    service: AgentExecutionService,
    agent_id: str,
    input_data: Dict[str, Any],
    context: ExecutionContext,
) -> None:
    """
    Background task for async agent execution.

    Args:
        service: Execution service instance
        agent_id: Agent to execute
        input_data: Input data
        context: Execution context
    """
    try:
        result = await service.execute_agent(agent_id, input_data, context)
        logger.info(
            f"Background execution completed: {result.execution_id} - {result.status}"
        )
    except Exception as e:
        logger.error(f"Background execution failed: {e}", exc_info=True)


def execution_record_to_response(record: Any) -> ExecutionResponse:
    """
    Convert execution record to API response.

    Args:
        record: Execution record from service

    Returns:
        ExecutionResponse for API
    """
    return ExecutionResponse(
        execution_id=record.execution_id,
        agent_id=record.agent_id,
        status=record.status.value if hasattr(record.status, "value") else record.status,
        progress=record.progress,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        duration_ms=record.duration_ms,
    )


def execution_result_to_response(result: ExecutionResult) -> ExecutionResultResponse:
    """
    Convert execution result to API response.

    Args:
        result: Execution result from service

    Returns:
        ExecutionResultResponse for API
    """
    cost_dict = None
    if result.cost:
        cost_dict = {
            "compute_cost_usd": result.cost.compute_cost_usd,
            "llm_cost_usd": result.cost.llm_cost_usd,
            "total_usd": result.cost.total_usd,
        }

    metrics_dict = {
        "start_time": result.metrics.start_time.isoformat(),
        "end_time": result.metrics.end_time.isoformat() if result.metrics.end_time else None,
        "duration_ms": result.metrics.duration_ms,
        "input_size_bytes": result.metrics.input_size_bytes,
        "output_size_bytes": result.metrics.output_size_bytes,
        "llm_tokens_input": result.metrics.llm_tokens_input,
        "llm_tokens_output": result.metrics.llm_tokens_output,
    }

    return ExecutionResultResponse(
        execution_id=result.execution_id,
        agent_id=result.agent_id,
        status=result.status.value if hasattr(result.status, "value") else result.status,
        result=result.result,
        error=result.error,
        provenance_hash=result.provenance_hash,
        cost=cost_dict,
        metrics=metrics_dict,
    )


# Endpoints

@router.post(
    "/{agent_id}/execute",
    response_model=ExecutionResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute agent",
    description="Execute an agent with the given input data.",
    responses={
        200: {"description": "Execution completed (sync mode)"},
        202: {"description": "Execution started (async mode)", "model": ExecutionResponse},
        400: {"description": "Invalid input data"},
        404: {"description": "Agent not found"},
        500: {"description": "Execution failed"},
    },
)
async def execute_agent(
    agent_id: str,
    request: ExecutionRequest,
    background_tasks: BackgroundTasks,
    # current_user: User = Depends(get_current_user),
) -> ExecutionResultResponse:
    """
    Execute an agent.

    - Validates input against agent schema
    - Tracks execution cost and metrics
    - Returns immediately if async_mode=True with 202 status
    - Enforces timeout limits
    - Provides full provenance tracking
    """
    logger.info(
        f"Executing agent {agent_id}",
        extra={
            "agent_id": agent_id,
            "async_mode": request.async_mode,
            "tenant_id": request.tenant_id,
        }
    )

    service = get_execution_service()

    # Build execution context
    context = ExecutionContext(
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        timeout_seconds=request.timeout_seconds,
        priority=request.priority,
        metadata=request.metadata,
    )

    if request.async_mode:
        # Start execution in background and return immediately
        # First create a pending record
        import uuid
        execution_id = str(uuid.uuid4())

        # Queue background execution
        background_tasks.add_task(
            execute_agent_background,
            service,
            agent_id,
            request.input_data,
            context,
        )

        # Return 202 Accepted with execution ID
        # Note: In a real implementation, you would create a pending record first
        # and return that execution_id for tracking
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail={
                "message": "Execution started",
                "agent_id": agent_id,
                "status": "PENDING",
                "note": "Use GET /executions/{execution_id} to check status",
            },
        )

    try:
        # Execute synchronously
        result = await service.execute_agent(
            agent_id=agent_id,
            input_data=request.input_data,
            context=context,
        )

        # Check for failed execution
        if result.status == ExecutionStatus.FAILED:
            logger.warning(f"Execution failed: {result.error}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": result.error,
                    "execution_id": result.execution_id,
                    "provenance_hash": result.provenance_hash,
                },
            )

        return execution_result_to_response(result)

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}",
        )


@router.get(
    "/{agent_id}/executions",
    response_model=ExecutionListResponse,
    summary="List executions",
    description="Get a paginated list of executions for an agent.",
)
async def list_executions(
    agent_id: str,
    execution_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tenant_id: str = Query("default-tenant", description="Tenant ID"),
    # current_user: User = Depends(get_current_user),
) -> ExecutionListResponse:
    """
    List executions for an agent.

    Supports filtering by:
    - status: Execution status (PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED)
    - start_date, end_date: Date range
    """
    logger.info(f"Listing executions for agent {agent_id}")

    service = get_execution_service()

    # Convert status string to enum if provided
    status_enum = None
    if execution_status:
        try:
            status_enum = ExecutionStatus(execution_status.upper())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {execution_status}. Valid values: {[s.value for s in ExecutionStatus]}",
            )

    # Get executions from service
    executions = await service.list_executions(
        agent_id=agent_id,
        tenant_id=tenant_id,
        status=status_enum,
        limit=limit,
        offset=offset,
    )

    # Convert to response models
    response_data = [execution_record_to_response(ex) for ex in executions]

    return ExecutionListResponse(
        data=response_data,
        meta={
            "total": len(executions),  # In production, would be total count from DB
            "limit": limit,
            "offset": offset,
            "agent_id": agent_id,
        },
    )


@router.get(
    "/{agent_id}/executions/{execution_id}",
    response_model=ExecutionResponse,
    summary="Get execution status",
    description="Get the current status of an execution.",
)
async def get_execution(
    agent_id: str,
    execution_id: str,
    # current_user: User = Depends(get_current_user),
) -> ExecutionResponse:
    """
    Get execution status.

    Returns current status and progress for running executions.
    """
    logger.info(f"Getting execution {execution_id}")

    service = get_execution_service()

    # Get execution from service
    execution = await service.get_execution(execution_id)

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )

    # Verify agent_id matches
    if execution.agent_id != agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found for agent {agent_id}",
        )

    return execution_record_to_response(execution)


@router.get(
    "/{agent_id}/executions/{execution_id}/result",
    response_model=ExecutionResultResponse,
    summary="Get execution result",
    description="Get the full result of a completed execution.",
)
async def get_execution_result(
    agent_id: str,
    execution_id: str,
    # current_user: User = Depends(get_current_user),
) -> ExecutionResultResponse:
    """
    Get execution result.

    Returns the complete result including:
    - Output data
    - Provenance hash
    - Cost breakdown
    - Execution metrics
    """
    logger.info(f"Getting result for execution {execution_id}")

    service = get_execution_service()

    # Get full result from service
    result = await service.get_execution_result(execution_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )

    # Verify agent_id matches
    if result.agent_id != agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found for agent {agent_id}",
        )

    return execution_result_to_response(result)


@router.post(
    "/{agent_id}/executions/{execution_id}/cancel",
    response_model=ExecutionResponse,
    summary="Cancel execution",
    description="Cancel a running execution.",
)
async def cancel_execution(
    agent_id: str,
    execution_id: str,
    force: bool = Query(False, description="Force cancellation"),
    # current_user: User = Depends(get_current_user),
) -> ExecutionResponse:
    """
    Cancel a running execution.

    - Graceful cancellation by default
    - Force=True for immediate termination
    """
    logger.info(f"Cancelling execution {execution_id} (force={force})")

    service = get_execution_service()

    # Get execution first to verify it exists
    execution = await service.get_execution(execution_id)

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found",
        )

    # Verify agent_id matches
    if execution.agent_id != agent_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found for agent {agent_id}",
        )

    # Check if execution can be cancelled
    if execution.status not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel execution with status {execution.status}",
        )

    # Cancel the execution
    cancelled = await service.cancel_execution(execution_id)

    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel execution {execution_id}",
        )

    # Get updated execution status
    updated_execution = await service.get_execution(execution_id)

    if updated_execution:
        return execution_record_to_response(updated_execution)

    # Return with cancelled status
    execution.status = ExecutionStatus.CANCELLED
    return execution_record_to_response(execution)


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get execution metrics",
    description="Get execution service metrics for monitoring.",
)
async def get_metrics() -> MetricsResponse:
    """
    Get execution metrics.

    Returns:
    - counters: Total executions, success/failure counts
    - gauges: Active executions, average duration
    """
    service = get_execution_service()
    metrics = service.get_metrics()

    return MetricsResponse(
        counters=metrics["counters"],
        gauges=metrics["gauges"],
    )
