# -*- coding: utf-8 -*-
"""
Template Management Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for managing workflow templates that define pre-configured DAG
topologies for each EUDR commodity. Templates specify agent inclusion,
dependency edges, quality gate thresholds, and execution parameters
specific to the supply chain archetype of each commodity.

Endpoints (4):
    GET  /templates              - List all available workflow templates
    GET  /templates/commodity/{commodity} - Get template for a specific commodity
    POST /templates              - Create a custom workflow template
    GET  /templates/{template_id} - Get template details by ID

RBAC Permissions:
    eudr-ddo:templates:read   - View templates
    eudr-ddo:templates:manage - Create/modify custom templates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_ddo_service,
    get_workflow_engine,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    EUDRCommodity,
    SUPPORTED_COMMODITIES,
    WorkflowDefinition,
    WorkflowType,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/templates", tags=["Template Management"])


# ---------------------------------------------------------------------------
# GET /templates -- List all templates
# ---------------------------------------------------------------------------


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="List all workflow templates",
    description=(
        "List all available workflow templates including built-in commodity "
        "templates (cattle, cocoa, coffee, palm_oil, rubber, soya, wood) "
        "and any custom templates created by the organization."
    ),
    responses={
        200: {"description": "Templates retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_templates(
    request: Request,
    workflow_type: Optional[str] = Query(
        default=None,
        description="Filter by workflow type: standard, simplified",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:templates:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """List all available workflow templates.

    Returns built-in commodity templates for all 7 EUDR commodities
    and any custom templates created by the authenticated user's
    organization.

    Args:
        request: FastAPI request object.
        workflow_type: Optional filter for workflow type.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with list of templates and metadata.
    """
    logger.info(
        "list_templates: user=%s type_filter=%s",
        user.user_id,
        workflow_type,
    )

    engine = get_workflow_engine()
    templates = []

    # Built-in commodity templates
    for commodity in SUPPORTED_COMMODITIES:
        for wf_type in ["standard", "simplified"]:
            if workflow_type and wf_type != workflow_type:
                continue

            template = engine.get_commodity_template(commodity, wf_type)
            if template is not None:
                templates.append({
                    "template_id": f"{commodity}_{wf_type}",
                    "name": f"{commodity.replace('_', ' ').title()} - {wf_type.title()} Due Diligence",
                    "commodity": commodity,
                    "workflow_type": wf_type,
                    "agent_count": len(template.nodes) if hasattr(template, "nodes") else 0,
                    "quality_gates": template.quality_gates if hasattr(template, "quality_gates") else [],
                    "is_builtin": True,
                    "description": f"Pre-configured {wf_type} due diligence workflow for {commodity.replace('_', ' ')}",
                })

    # Custom templates from the database
    custom_templates = engine.list_custom_templates(tenant_id=user.tenant_id)
    for ct in custom_templates:
        if workflow_type and ct.workflow_type.value != workflow_type:
            continue
        templates.append({
            "template_id": ct.definition_id,
            "name": ct.name,
            "commodity": ct.commodity.value if ct.commodity else None,
            "workflow_type": ct.workflow_type.value,
            "agent_count": len(ct.nodes),
            "quality_gates": ct.quality_gates,
            "is_builtin": False,
            "description": ct.description,
        })

    return {
        "templates": templates,
        "total": len(templates),
    }


# ---------------------------------------------------------------------------
# GET /templates/commodity/{commodity} -- Get commodity template
# ---------------------------------------------------------------------------


@router.get(
    "/commodity/{commodity}",
    status_code=status.HTTP_200_OK,
    summary="Get workflow template for a commodity",
    description=(
        "Get the pre-configured workflow template for a specific EUDR "
        "commodity. Returns the full DAG definition with agent nodes, "
        "dependency edges, quality gate thresholds, and execution parameters."
    ),
    responses={
        200: {"description": "Template retrieved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid commodity"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Template not found"},
    },
)
async def get_commodity_template(
    request: Request,
    commodity: str,
    workflow_type: str = Query(
        default="standard",
        description="Workflow type: standard or simplified",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:templates:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get the workflow template for a specific commodity.

    Returns the complete DAG definition for the specified commodity and
    workflow type, including all agent nodes, dependency edges, quality
    gate configurations, and execution parameters.

    Args:
        request: FastAPI request object.
        commodity: EUDR commodity (cattle, cocoa, coffee, palm_oil, rubber, soya, wood).
        workflow_type: Workflow type (standard or simplified).
        user: Authenticated and authorized user.

    Returns:
        Dictionary with template definition and metadata.

    Raises:
        HTTPException: 400 if invalid commodity, 404 if template not found.
    """
    logger.info(
        "get_commodity_template: user=%s commodity=%s type=%s",
        user.user_id,
        commodity,
        workflow_type,
    )

    if commodity not in SUPPORTED_COMMODITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid commodity: {commodity}. "
                f"Valid values: {', '.join(SUPPORTED_COMMODITIES)}"
            ),
        )

    if workflow_type not in ("standard", "simplified"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="workflow_type must be 'standard' or 'simplified'",
        )

    engine = get_workflow_engine()
    template = engine.get_commodity_template(commodity, workflow_type)

    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found for {commodity} ({workflow_type})",
        )

    return {
        "template_id": f"{commodity}_{workflow_type}",
        "commodity": commodity,
        "workflow_type": workflow_type,
        "definition": template.model_dump(mode="json") if hasattr(template, "model_dump") else {},
        "agent_count": len(template.nodes) if hasattr(template, "nodes") else 0,
        "edge_count": len(template.edges) if hasattr(template, "edges") else 0,
        "quality_gates": template.quality_gates if hasattr(template, "quality_gates") else [],
    }


# ---------------------------------------------------------------------------
# POST /templates -- Create custom template
# ---------------------------------------------------------------------------


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a custom workflow template",
    description=(
        "Create a custom workflow template with operator-defined agent "
        "selection, dependency edges, and quality gate thresholds. "
        "Custom templates must pass DAG validation before being saved."
    ),
    responses={
        201: {"description": "Template created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid template definition"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def create_template(
    request: Request,
    body: WorkflowDefinition,
    user: AuthUser = Depends(require_permission("eudr-ddo:templates:manage")),
    _rate: AuthUser = Depends(rate_limit_write),
) -> Dict[str, Any]:
    """Create a custom workflow template.

    Validates the DAG definition for acyclicity and completeness,
    then stores the template for reuse. The template can be used
    when creating workflows with workflow_type='custom'.

    Args:
        request: FastAPI request object.
        body: WorkflowDefinition with nodes, edges, and quality gates.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with template ID, validation result, and metadata.

    Raises:
        HTTPException: 400 if definition is invalid.
    """
    logger.info(
        "create_template: user=%s name=%s type=%s commodity=%s",
        user.user_id,
        body.name,
        body.workflow_type,
        body.commodity,
    )

    engine = get_workflow_engine()

    # Set the creator
    body.created_by = user.user_id

    # Validate the definition
    is_valid, validation_errors = engine.validate_definition(body)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Template validation failed",
                "errors": validation_errors,
            },
        )

    # Store the template
    engine.store_custom_template(body, tenant_id=user.tenant_id)

    return {
        "template_id": body.definition_id,
        "name": body.name,
        "commodity": body.commodity.value if body.commodity else None,
        "workflow_type": body.workflow_type.value,
        "agent_count": len(body.nodes),
        "edge_count": len(body.edges),
        "valid": True,
        "created_at": utcnow().isoformat(),
        "created_by": user.user_id,
    }


# ---------------------------------------------------------------------------
# GET /templates/{template_id} -- Get template by ID
# ---------------------------------------------------------------------------


@router.get(
    "/{template_id}",
    status_code=status.HTTP_200_OK,
    summary="Get template details by ID",
    description=(
        "Get the full definition of a workflow template by its unique "
        "identifier. Returns agent nodes, dependency edges, quality gate "
        "configurations, and template metadata."
    ),
    responses={
        200: {"description": "Template retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Template not found"},
    },
)
async def get_template(
    request: Request,
    template_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:templates:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get template details by ID.

    Retrieves the complete template definition including all agent
    nodes, dependency edges, quality gate thresholds, and metadata.

    Args:
        request: FastAPI request object.
        template_id: Unique template identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with full template definition and metadata.

    Raises:
        HTTPException: 404 if template not found.
    """
    logger.info(
        "get_template: user=%s template_id=%s",
        user.user_id,
        template_id,
    )

    engine = get_workflow_engine()

    # Check built-in templates first (format: commodity_type)
    parts = template_id.rsplit("_", 1)
    if len(parts) == 2 and parts[0] in SUPPORTED_COMMODITIES:
        commodity, wf_type = parts
        if wf_type in ("standard", "simplified"):
            template = engine.get_commodity_template(commodity, wf_type)
            if template is not None:
                return {
                    "template_id": template_id,
                    "name": f"{commodity.replace('_', ' ').title()} - {wf_type.title()} Due Diligence",
                    "commodity": commodity,
                    "workflow_type": wf_type,
                    "is_builtin": True,
                    "definition": template.model_dump(mode="json") if hasattr(template, "model_dump") else {},
                    "agent_count": len(template.nodes) if hasattr(template, "nodes") else 0,
                    "edge_count": len(template.edges) if hasattr(template, "edges") else 0,
                    "quality_gates": template.quality_gates if hasattr(template, "quality_gates") else [],
                }

    # Check custom templates
    template = engine.get_custom_template(template_id, tenant_id=user.tenant_id)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template {template_id} not found",
        )

    return {
        "template_id": template.definition_id,
        "name": template.name,
        "commodity": template.commodity.value if template.commodity else None,
        "workflow_type": template.workflow_type.value,
        "is_builtin": False,
        "definition": template.model_dump(mode="json") if hasattr(template, "model_dump") else {},
        "agent_count": len(template.nodes),
        "edge_count": len(template.edges),
        "quality_gates": template.quality_gates,
        "created_at": template.created_at.isoformat() if template.created_at else None,
        "created_by": template.created_by,
    }
