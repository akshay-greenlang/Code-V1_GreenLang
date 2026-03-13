# -*- coding: utf-8 -*-
"""
Workflow CRUD Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for workflow lifecycle management including creation, listing,
retrieval, deletion, validation, and cloning of EUDR due diligence
workflows. Workflows define and manage the three-phase due diligence
process per EUDR Article 8: information gathering (Art. 9), risk
assessment (Art. 10), and risk mitigation (Art. 11).

Endpoints (6):
    POST /workflows             - Create a new due diligence workflow
    GET  /workflows             - List workflows with filters and pagination
    GET  /workflows/{id}        - Get workflow details by ID
    DELETE /workflows/{id}      - Soft-delete (archive) a workflow
    POST /workflows/{id}/validate - Validate workflow DAG definition
    POST /workflows/{id}/clone  - Clone an existing workflow

RBAC Permissions:
    eudr-ddo:workflows:create  - Create and clone workflows
    eudr-ddo:workflows:read    - View workflow status and details
    eudr-ddo:workflows:manage  - Manage workflows
    eudr-ddo:workflows:delete  - Archive workflows

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_ddo_service,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity,
    validate_status_filter,
    validate_workflow_type,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    CreateWorkflowRequest,
    DueDiligencePhase,
    EUDRCommodity,
    WorkflowDefinition,
    WorkflowStatus,
    WorkflowStatusResponse,
    WorkflowType,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Workflow Management"])


# ---------------------------------------------------------------------------
# POST /workflows -- Create a new workflow
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new due diligence workflow",
    description=(
        "Create a new EUDR due diligence workflow with DAG-based agent "
        "topology. Supports standard (25-agent), simplified (Article 13), "
        "and custom workflow types across all 7 EUDR commodities."
    ),
    responses={
        201: {"description": "Workflow created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def create_workflow(
    request: Request,
    body: CreateWorkflowRequest,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:create")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> WorkflowStatusResponse:
    """Create a new due diligence workflow.

    Creates the workflow definition (DAG), initializes the state machine,
    and returns the workflow identifier for subsequent operations.

    Args:
        request: FastAPI request object.
        body: Workflow creation request with operator, commodity, and
            country parameters.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with the created workflow details.

    Raises:
        HTTPException: 400 if parameters are invalid, 401/403 for auth.
    """
    logger.info(
        "create_workflow: user=%s operator=%s commodity=%s type=%s",
        user.user_id,
        body.operator_id,
        body.commodity,
        body.workflow_type,
    )

    try:
        service = get_ddo_service()
        result = service.create_workflow(body)
        result.request_id = body.request_id
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /workflows -- List workflows
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="List due diligence workflows",
    description=(
        "List all due diligence workflows for the authenticated tenant "
        "with optional filters for status, commodity, workflow type, "
        "and pagination support."
    ),
    responses={
        200: {"description": "Workflows retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_workflows(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    pagination: PaginationParams = Depends(get_pagination),
    commodity: Optional[str] = Depends(validate_commodity),
    workflow_type: Optional[str] = Depends(validate_workflow_type),
    status_filter: Optional[str] = Depends(validate_status_filter),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """List workflows with filters and pagination.

    Returns a paginated list of workflows accessible to the authenticated
    user's tenant, with optional filtering by status, commodity, and
    workflow type.

    Args:
        request: FastAPI request object.
        user: Authenticated and authorized user.
        pagination: Pagination parameters (limit, offset).
        commodity: Optional commodity filter.
        workflow_type: Optional workflow type filter.
        status_filter: Optional status filter.

    Returns:
        Dictionary with workflows array, pagination metadata, and total count.
    """
    logger.info(
        "list_workflows: user=%s commodity=%s type=%s status=%s limit=%d offset=%d",
        user.user_id,
        commodity,
        workflow_type,
        status_filter,
        pagination.limit,
        pagination.offset,
    )

    service = get_ddo_service()

    # The state manager provides a list_workflows helper
    # In production this queries PostgreSQL with filters
    all_workflows = service._state_manager.list_workflows(
        tenant_id=user.tenant_id,
        commodity=commodity,
        workflow_type=workflow_type,
        status_filter=status_filter,
        limit=pagination.limit,
        offset=pagination.offset,
    )

    total_count = service._state_manager.count_workflows(
        tenant_id=user.tenant_id,
        commodity=commodity,
        workflow_type=workflow_type,
        status_filter=status_filter,
    )

    workflow_summaries = []
    for wf in all_workflows:
        workflow_summaries.append({
            "workflow_id": wf.workflow_id,
            "status": wf.status.value if hasattr(wf.status, "value") else str(wf.status),
            "commodity": wf.commodity.value if wf.commodity and hasattr(wf.commodity, "value") else wf.commodity,
            "workflow_type": wf.workflow_type.value if hasattr(wf.workflow_type, "value") else str(wf.workflow_type),
            "current_phase": wf.current_phase.value if hasattr(wf.current_phase, "value") else str(wf.current_phase),
            "progress_pct": str(wf.progress_pct),
            "operator_id": wf.operator_id,
            "created_at": wf.created_at.isoformat() if wf.created_at else None,
        })

    return {
        "workflows": workflow_summaries,
        "meta": {
            "total": total_count,
            "limit": pagination.limit,
            "offset": pagination.offset,
            "has_more": (pagination.offset + pagination.limit) < total_count,
        },
    }


# ---------------------------------------------------------------------------
# GET /workflows/{workflow_id} -- Get workflow details
# ---------------------------------------------------------------------------


@router.get(
    "/{workflow_id}",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get workflow details",
    description=(
        "Get detailed status and progress information for a specific "
        "due diligence workflow including agent execution statuses, "
        "quality gate results, and ETA."
    ),
    responses={
        200: {"description": "Workflow retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def get_workflow(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> WorkflowStatusResponse:
    """Get detailed workflow information.

    Retrieves the current status, phase, agent progress, quality gate
    results, and estimated time to completion for the specified workflow.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse with full workflow details.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "get_workflow: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    try:
        service = get_ddo_service()
        return service.get_workflow_status(workflow_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# DELETE /workflows/{workflow_id} -- Soft-delete (archive) workflow
# ---------------------------------------------------------------------------


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_200_OK,
    summary="Archive a workflow",
    description=(
        "Soft-delete (archive) a due diligence workflow. Archived workflows "
        "are retained for the EUDR Article 31 5-year record keeping period "
        "but are hidden from default listing."
    ),
    responses={
        200: {"description": "Workflow archived successfully"},
        400: {"model": ErrorResponse, "description": "Cannot archive running workflow"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def delete_workflow(
    request: Request,
    workflow_id: str,
    reason: Optional[str] = Query(
        default=None,
        description="Reason for archiving the workflow",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:delete")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> Dict[str, Any]:
    """Archive (soft-delete) a workflow.

    Marks the workflow as archived while retaining all data for the
    EUDR Article 31 5-year record keeping requirement.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        reason: Optional archival reason for audit trail.
        user: Authenticated and authorized user.

    Returns:
        Confirmation with workflow ID and archive timestamp.

    Raises:
        HTTPException: 400 if workflow is running, 404 if not found.
    """
    logger.info(
        "delete_workflow: user=%s workflow_id=%s reason=%s",
        user.user_id,
        workflow_id,
        reason,
    )

    service = get_ddo_service()
    try:
        state = service._state_manager.get_state(workflow_id)
        if state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found",
            )

        if state.status == WorkflowStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot archive a running workflow. Cancel or pause first.",
            )

        service._state_manager.archive_workflow(
            workflow_id,
            reason=reason or f"Archived by {user.user_id}",
        )

        return {
            "workflow_id": workflow_id,
            "status": "archived",
            "archived_at": _utcnow().isoformat(),
            "archived_by": user.user_id,
            "reason": reason,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/validate -- Validate workflow DAG
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/validate",
    status_code=status.HTTP_200_OK,
    summary="Validate workflow DAG definition",
    description=(
        "Validate the workflow DAG for acyclicity, connectivity, and "
        "completeness. Checks for circular dependencies, missing agent "
        "nodes, and quality gate placement."
    ),
    responses={
        200: {"description": "Workflow validation result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
    },
)
async def validate_workflow(
    request: Request,
    workflow_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Validate a workflow DAG definition.

    Runs the full validation suite on the workflow DAG: acyclicity check,
    topological sort, dependency resolution, quality gate placement, and
    critical path calculation.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with validation result, errors, and DAG metadata.

    Raises:
        HTTPException: 404 if workflow not found.
    """
    logger.info(
        "validate_workflow: user=%s workflow_id=%s",
        user.user_id,
        workflow_id,
    )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    definition = service._workflow_engine.get_definition(state.definition_id)
    if definition is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow definition not found for {workflow_id}",
        )

    is_valid, validation_errors = service._workflow_engine.validate_definition(
        definition
    )

    # Calculate execution layers and critical path if valid
    dag_metadata = {}
    if is_valid:
        layers = service._workflow_engine.get_execution_layers(definition)
        dag_metadata = {
            "total_nodes": len(definition.nodes),
            "total_edges": len(definition.edges),
            "execution_layers": len(layers),
            "quality_gates": definition.quality_gates,
            "is_acyclic": True,
        }

    return {
        "workflow_id": workflow_id,
        "valid": is_valid,
        "errors": validation_errors,
        "dag_metadata": dag_metadata,
        "validated_at": _utcnow().isoformat(),
        "validated_by": user.user_id,
    }


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/clone -- Clone an existing workflow
# ---------------------------------------------------------------------------


@router.post(
    "/{workflow_id}/clone",
    response_model=WorkflowStatusResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Clone an existing workflow",
    description=(
        "Clone an existing workflow definition to create a new workflow "
        "with the same DAG topology, quality gate thresholds, and agent "
        "configuration. Optionally override commodity or workflow type."
    ),
    responses={
        201: {"description": "Workflow cloned successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Source workflow not found"},
    },
)
async def clone_workflow(
    request: Request,
    workflow_id: str,
    name: Optional[str] = Query(
        default=None,
        description="Name for the cloned workflow",
    ),
    commodity: Optional[str] = Depends(validate_commodity),
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:create")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> WorkflowStatusResponse:
    """Clone an existing workflow.

    Creates a new workflow based on the source workflow's DAG definition.
    The new workflow starts in CREATED status with no execution history.

    Args:
        request: FastAPI request object.
        workflow_id: Source workflow identifier to clone from.
        name: Optional name for the new workflow.
        commodity: Optional commodity override for the clone.
        user: Authenticated and authorized user.

    Returns:
        WorkflowStatusResponse for the newly cloned workflow.

    Raises:
        HTTPException: 404 if source workflow not found.
    """
    logger.info(
        "clone_workflow: user=%s source_workflow_id=%s commodity=%s",
        user.user_id,
        workflow_id,
        commodity,
    )

    service = get_ddo_service()
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source workflow {workflow_id} not found",
        )

    # Build clone request from the source workflow
    clone_commodity = (
        EUDRCommodity(commodity) if commodity else state.commodity
    )

    clone_request = CreateWorkflowRequest(
        workflow_type=state.workflow_type,
        commodity=clone_commodity,
        operator_id=state.operator_id or user.operator_id,
        operator_name=state.operator_name,
        country_codes=state.country_codes,
    )

    try:
        result = service.create_workflow(clone_request)
        logger.info(
            "Workflow cloned: source=%s new=%s",
            workflow_id,
            result.workflow_id,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
