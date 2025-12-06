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

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

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
    # current_user: User = Depends(get_current_user),
) -> AgentResponse:
    """
    Create a new agent.

    - Validates agent_id format (category/name)
    - Checks for duplicates
    - Sets initial state to DRAFT
    - Creates audit log entry
    """
    logger.info(f"Creating agent: {request.agent_id}")

    # TODO: Call registry service
    # agent = await registry_service.register_agent(spec, tenant_id)

    # Placeholder response
    return AgentResponse(
        id="agent-000001",
        agent_id=request.agent_id,
        name=request.name,
        version=request.version,
        state="DRAFT",
        category=request.category,
        tags=request.tags,
        description=request.description,
        tenant_id="tenant-1",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        invocation_count=0,
        success_rate=1.0,
    )


@router.get(
    "",
    response_model=AgentListResponse,
    summary="List all agents",
    description="Get a paginated list of agents with optional filtering.",
)
async def list_agents(
    category: Optional[str] = Query(None, description="Filter by category"),
    state: Optional[str] = Query(None, description="Filter by state"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    # current_user: User = Depends(get_current_user),
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

    # TODO: Call registry service
    # agents = await registry_service.list_agents(filters, pagination, tenant_id)

    # Placeholder response
    return AgentListResponse(
        data=[],
        meta={
            "total": 0,
            "limit": limit,
            "offset": offset,
            "has_more": False,
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
    # current_user: User = Depends(get_current_user),
) -> AgentResponse:
    """
    Get agent by ID.

    Returns the agent details including latest version and metrics.
    """
    logger.info(f"Getting agent: {agent_id}")

    # TODO: Call registry service
    # agent = await registry_service.get_agent(agent_id, tenant_id)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Agent {agent_id} not found",
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
    # current_user: User = Depends(get_current_user),
) -> AgentResponse:
    """
    Update agent metadata.

    Supports partial updates - only provided fields are updated.
    Creates audit log entry.
    """
    logger.info(f"Updating agent: {agent_id}")

    # TODO: Call registry service
    # agent = await registry_service.update_agent(agent_id, updates, tenant_id)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Agent {agent_id} not found",
    )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent",
    description="Soft delete an agent. CERTIFIED agents cannot be deleted.",
)
async def delete_agent(
    agent_id: str,
    # current_user: User = Depends(get_current_user),
) -> None:
    """
    Delete an agent (soft delete).

    - Cannot delete CERTIFIED agents
    - Transitions to RETIRED state
    - Creates audit log entry
    """
    logger.info(f"Deleting agent: {agent_id}")

    # TODO: Call registry service
    # await registry_service.delete_agent(agent_id, tenant_id)


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
    # current_user: User = Depends(get_current_user),
) -> VersionResponse:
    """
    Create a new version.

    - Validates semantic version format
    - Checks version uniqueness
    - Marks as latest version
    """
    logger.info(f"Creating version {request.version} for agent {agent_id}")

    # TODO: Call version manager
    # version = await version_manager.create_version(agent_id, request.version)

    return VersionResponse(
        version=request.version,
        agent_id=agent_id,
        created_at=datetime.utcnow(),
        artifact_url=None,
        is_latest=True,
    )


@router.get(
    "/{agent_id}/versions",
    response_model=List[VersionResponse],
    summary="List versions",
    description="Get all versions of an agent.",
)
async def list_versions(
    agent_id: str,
    # current_user: User = Depends(get_current_user),
) -> List[VersionResponse]:
    """
    List all versions of an agent.

    Returns versions sorted by semantic version (newest first).
    """
    logger.info(f"Listing versions for agent {agent_id}")

    # TODO: Call version manager
    # versions = await version_manager.list_versions(agent_id)

    return []


@router.post(
    "/{agent_id}/transition",
    response_model=AgentResponse,
    summary="Transition state",
    description="Transition agent to a new lifecycle state.",
)
async def transition_state(
    agent_id: str,
    request: StateTransitionRequest,
    # current_user: User = Depends(get_current_user),
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

    # TODO: Call registry service
    # agent = await registry_service.transition_state(
    #     agent_id, target_state, current_user.id, tenant_id
    # )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Agent {agent_id} not found",
    )
