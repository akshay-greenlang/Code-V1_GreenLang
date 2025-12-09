"""
Agent CRUD Router

This module provides REST API endpoints for agent management:
- POST /v1/agents - Create new agent
- GET /v1/agents - List all agents
- GET /v1/agents/{agent_id} - Get agent by ID
- PATCH /v1/agents/{agent_id} - Update agent
- DELETE /v1/agents/{agent_id} - Delete agent
- POST /v1/agents/{agent_id}/versions - Create new version
- GET /v1/agents/{agent_id}/versions - List versions
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

# Import dependencies and registry service
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from registry.service import AgentRegistryService
from app.dependencies import get_registry_service, get_tenant_id, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models

class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""

    agent_id: str = Field(..., description="Unique agent identifier (category/name)")
    name: str = Field(..., description="Human-readable name")
    version: str = Field("1.0.0", description="Initial version")
    description: Optional[str] = Field(None, description="Agent description")
    category: str = Field(..., description="Agent category")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    entrypoint: str = Field(..., description="Python entrypoint")
    deterministic: bool = Field(True, description="Is agent deterministic")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    regulatory_frameworks: List[str] = Field(default_factory=list)


class AgentUpdateRequest(BaseModel):
    """Request to update an agent."""

    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class AgentResponse(BaseModel):
    """Agent response model."""

    id: str
    agent_id: str
    name: str
    version: str
    state: str
    category: str
    tags: List[str]
    description: Optional[str]
    tenant_id: str
    created_at: datetime
    updated_at: datetime
    invocation_count: int
    success_rate: float


class AgentListResponse(BaseModel):
    """Paginated list of agents."""

    data: List[AgentResponse]
    meta: Dict[str, Any]


class VersionCreateRequest(BaseModel):
    """Request to create a new version."""

    version: str = Field(..., description="Semantic version (X.Y.Z)")
    changelog: Optional[str] = None


class VersionResponse(BaseModel):
    """Version response model."""

    version: str
    agent_id: str
    created_at: datetime
    artifact_url: Optional[str]
    is_latest: bool


class StateTransitionRequest(BaseModel):
    """Request to transition agent state."""

    target_state: str = Field(..., description="Target state")
    reason: Optional[str] = None


# Endpoints

@router.post(
    "",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new agent",
    description="Register a new agent with the given specification. Initial state will be DRAFT.",
)
async def create_agent(
    request: AgentCreateRequest,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> AgentResponse:
    """
    Create a new agent.

    - Validates agent_id format (category/name)
    - Checks for duplicates
    - Sets initial state to DRAFT
    - Creates audit log entry
    """
    logger.info(f"Creating agent: {request.agent_id}")

    tenant_id = get_tenant_id(http_request)
    user_id = get_user_id(http_request)

    try:
        agent = await service.create_agent(
            name=request.agent_id,
            version=request.version,
            category=request.category,
            author=user_id or "anonymous",
            description=request.description or "",
            tags=request.tags,
            regulatory_frameworks=request.regulatory_frameworks,
            tenant_id=tenant_id,
        )

        return AgentResponse(
            id=str(agent.id),
            agent_id=request.agent_id,
            name=request.name,
            version=agent.version,
            state=agent.status.upper(),
            category=agent.category,
            tags=agent.tags or [],
            description=agent.description,
            tenant_id=str(agent.tenant_id) if agent.tenant_id else "default",
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            invocation_count=0,
            success_rate=1.0,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List all agents",
    description="Get a paginated list of agents with optional filtering.",
)
async def list_agents(
    http_request: Request,
    category: Optional[str] = Query(None, description="Filter by category"),
    state: Optional[str] = Query(None, description="Filter by state"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    service: AgentRegistryService = Depends(get_registry_service),
) -> AgentListResponse:
    """
    List agents with filtering and pagination.

    Supports filtering by:
    - category: Agent category
    - state: Lifecycle state (DRAFT, CERTIFIED, etc.)
    - tags: Comma-separated list of tags
    - search: Text search in name and description
    """
    logger.info(f"Listing agents: category={category}, state={state}, limit={limit}")

    tenant_id = get_tenant_id(http_request)
    tag_list = tags.split(",") if tags else None

    # Map state to status
    status_filter = state.lower() if state else None

    if search:
        # Use search endpoint
        agents, total = await service.search_agents(
            query=search,
            category=category,
            status=status_filter,
            tags=tag_list,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )
    else:
        # Use list endpoint
        agents, total = await service.list_agents(
            category=category,
            status=status_filter,
            tags=tag_list,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    # Convert to response format
    agent_responses = [
        AgentResponse(
            id=str(a.id),
            agent_id=a.name,
            name=a.name,
            version=a.version,
            state=a.status.upper(),
            category=a.category,
            tags=a.tags or [],
            description=a.description,
            tenant_id=str(a.tenant_id) if a.tenant_id else "default",
            created_at=a.created_at,
            updated_at=a.updated_at,
            invocation_count=a.downloads,
            success_rate=1.0,
        )
        for a in agents
    ]

    return AgentListResponse(
        data=agent_responses,
        meta={
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(agents) < total,
        },
    )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent by ID",
    description="Retrieve a specific agent by its identifier.",
)
async def get_agent(
    agent_id: str,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> AgentResponse:
    """
    Get agent by ID.

    Returns the agent details including latest version and metrics.
    """
    logger.info(f"Getting agent: {agent_id}")

    tenant_id = get_tenant_id(http_request)

    # Try to get by name first (agent_id is typically the name)
    agent = await service.get_agent_by_name(agent_id, tenant_id)

    # If not found by name, try by UUID
    if not agent:
        try:
            agent_uuid = UUID(agent_id)
            agent = await service.get_agent(agent_uuid, tenant_id)
        except (ValueError, TypeError):
            pass

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return AgentResponse(
        id=str(agent.id),
        agent_id=agent.name,
        name=agent.name,
        version=agent.version,
        state=agent.status.upper(),
        category=agent.category,
        tags=agent.tags or [],
        description=agent.description,
        tenant_id=str(agent.tenant_id) if agent.tenant_id else "default",
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        invocation_count=agent.downloads,
        success_rate=1.0,
    )


@router.patch(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Update agent",
    description="Update agent metadata. Cannot update agent_id or created_at.",
)
async def update_agent(
    agent_id: str,
    request: AgentUpdateRequest,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> AgentResponse:
    """
    Update agent metadata.

    Supports partial updates - only provided fields are updated.
    Creates audit log entry.
    """
    logger.info(f"Updating agent: {agent_id}")

    tenant_id = get_tenant_id(http_request)
    user_id = get_user_id(http_request)

    # Get agent by name first
    agent = await service.get_agent_by_name(agent_id, tenant_id)
    if not agent:
        try:
            agent_uuid = UUID(agent_id)
            agent = await service.get_agent(agent_uuid, tenant_id)
        except (ValueError, TypeError):
            pass

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    # Build updates dict
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.description is not None:
        updates["description"] = request.description
    if request.tags is not None:
        updates["tags"] = request.tags

    try:
        updated_agent = await service.update_agent(
            agent_id=agent.id,
            updates=updates,
            tenant_id=tenant_id,
            user_id=user_id,
        )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )

    if not updated_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return AgentResponse(
        id=str(updated_agent.id),
        agent_id=updated_agent.name,
        name=updated_agent.name,
        version=updated_agent.version,
        state=updated_agent.status.upper(),
        category=updated_agent.category,
        tags=updated_agent.tags or [],
        description=updated_agent.description,
        tenant_id=str(updated_agent.tenant_id) if updated_agent.tenant_id else "default",
        created_at=updated_agent.created_at,
        updated_at=updated_agent.updated_at,
        invocation_count=updated_agent.downloads,
        success_rate=1.0,
    )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent",
    description="Soft delete an agent. CERTIFIED agents cannot be deleted.",
)
async def delete_agent(
    agent_id: str,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> None:
    """
    Delete an agent (soft delete).

    - Cannot delete CERTIFIED agents
    - Transitions to RETIRED state
    - Creates audit log entry
    """
    logger.info(f"Deleting agent: {agent_id}")

    tenant_id = get_tenant_id(http_request)
    user_id = get_user_id(http_request)

    # Get agent by name first
    agent = await service.get_agent_by_name(agent_id, tenant_id)
    if not agent:
        try:
            agent_uuid = UUID(agent_id)
            agent = await service.get_agent(agent_uuid, tenant_id)
        except (ValueError, TypeError):
            pass

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    try:
        deleted = await service.delete_agent(
            agent_id=agent.id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        )


@router.post(
    "/{agent_id}/versions",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new version",
    description="Create a new version of an agent.",
)
async def create_version(
    agent_id: str,
    request: VersionCreateRequest,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> VersionResponse:
    """
    Create a new version.

    - Validates semantic version format
    - Checks version uniqueness
    - Marks as latest version
    """
    logger.info(f"Creating version {request.version} for agent {agent_id}")

    tenant_id = get_tenant_id(http_request)
    user_id = get_user_id(http_request)

    # Get agent
    agent = await service.get_agent_by_name(agent_id, tenant_id)
    if not agent:
        try:
            agent_uuid = UUID(agent_id)
            agent = await service.get_agent(agent_uuid, tenant_id)
        except (ValueError, TypeError):
            pass

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    try:
        version = await service.create_version(
            agent_id=agent.id,
            version=request.version,
            changelog=request.changelog or "",
            user_id=user_id,
        )

        return VersionResponse(
            version=version.version,
            agent_id=str(version.agent_id),
            created_at=version.created_at,
            artifact_url=version.artifact_path,
            is_latest=version.is_latest,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/{agent_id}/versions",
    response_model=List[VersionResponse],
    summary="List versions",
    description="Get all versions of an agent.",
)
async def list_versions(
    agent_id: str,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> List[VersionResponse]:
    """
    List all versions of an agent.

    Returns versions sorted by semantic version (newest first).
    """
    logger.info(f"Listing versions for agent {agent_id}")

    tenant_id = get_tenant_id(http_request)

    # Get agent
    agent = await service.get_agent_by_name(agent_id, tenant_id)
    if not agent:
        try:
            agent_uuid = UUID(agent_id)
            agent = await service.get_agent(agent_uuid, tenant_id)
        except (ValueError, TypeError):
            pass

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    versions = await service.list_versions(agent.id)

    return [
        VersionResponse(
            version=v.version,
            agent_id=str(v.agent_id),
            created_at=v.created_at,
            artifact_url=v.artifact_path,
            is_latest=v.is_latest,
        )
        for v in versions
    ]


@router.post(
    "/{agent_id}/transition",
    response_model=AgentResponse,
    summary="Transition state",
    description="Transition agent to a new lifecycle state.",
)
async def transition_state(
    agent_id: str,
    request: StateTransitionRequest,
    http_request: Request,
    service: AgentRegistryService = Depends(get_registry_service),
) -> AgentResponse:
    """
    Transition agent state.

    Valid transitions:
    - DRAFT -> EXPERIMENTAL, RETIRED
    - EXPERIMENTAL -> CERTIFIED, DRAFT, RETIRED
    - CERTIFIED -> DEPRECATED
    - DEPRECATED -> RETIRED, CERTIFIED
    """
    logger.info(f"Transitioning agent {agent_id} to {request.target_state}")

    tenant_id = get_tenant_id(http_request)
    user_id = get_user_id(http_request)

    # Get agent
    agent = await service.get_agent_by_name(agent_id, tenant_id)
    if not agent:
        try:
            agent_uuid = UUID(agent_id)
            agent = await service.get_agent(agent_uuid, tenant_id)
        except (ValueError, TypeError):
            pass

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    # Map target state to service methods
    target = request.target_state.lower()

    try:
        if target == "published" or target == "certified":
            # Publish the agent
            await service.publish_agent(
                agent_id=agent.id,
                version=agent.version,
                release_notes=request.reason,
                user_id=user_id,
            )
            # Refresh agent data
            agent = await service.get_agent(agent.id, tenant_id)
        elif target == "deprecated":
            agent = await service.deprecate_agent(
                agent_id=agent.id,
                user_id=user_id,
            )
        else:
            # For other transitions, update status directly
            agent = await service.update_agent(
                agent_id=agent.id,
                updates={"status": target},
                tenant_id=tenant_id,
                user_id=user_id,
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    return AgentResponse(
        id=str(agent.id),
        agent_id=agent.name,
        name=agent.name,
        version=agent.version,
        state=agent.status.upper(),
        category=agent.category,
        tags=agent.tags or [],
        description=agent.description,
        tenant_id=str(agent.tenant_id) if agent.tenant_id else "default",
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        invocation_count=agent.downloads,
        success_rate=1.0,
    )
