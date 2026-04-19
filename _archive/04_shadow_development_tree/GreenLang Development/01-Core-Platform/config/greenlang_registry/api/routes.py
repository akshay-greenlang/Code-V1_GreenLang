"""
API Routes for GreenLang Agent Registry

This module implements the 4 core API endpoints:
1. POST /api/v1/registry/agents - Publish agent
2. GET /api/v1/registry/agents - List agents (paginated)
3. GET /api/v1/registry/agents/{id} - Get agent details
4. POST /api/v1/registry/agents/{id}/promote - Promote agent state

All endpoints use async database operations with SQLAlchemy 2.0.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload

from greenlang_registry.db.client import get_database_client
from greenlang_registry.db.models import (
    Agent,
    AgentVersion,
    StateTransition,
    AuditLog,
    LifecycleState as DBLifecycleState,
)
from greenlang_registry.models import (
    PublishRequest,
    PublishResponse,
    PromoteRequest,
    PromoteResponse,
    AgentMetadata,
    AgentVersion as AgentVersionModel,
    AgentDetail,
    ListAgentsResponse,
    ErrorResponse,
    LifecycleState,
    SemanticVersion,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================


def parse_semantic_version(version: str) -> dict:
    """Parse a version string into semantic version components."""
    import re

    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
    match = re.match(pattern, version)
    if not match:
        return {"major": 0, "minor": 0, "patch": 0}

    return {
        "major": int(match.group(1)),
        "minor": int(match.group(2)),
        "patch": int(match.group(3)),
        "prerelease": match.group(4),
        "build": match.group(5),
    }


def agent_to_metadata(agent: Agent) -> AgentMetadata:
    """Convert SQLAlchemy Agent model to Pydantic AgentMetadata."""
    tags = agent.tags if isinstance(agent.tags, list) else (agent.tags or {}).get("tags", [])

    return AgentMetadata(
        agent_id=agent.agent_id,
        name=agent.name,
        description=agent.description,
        domain=agent.domain,
        type=agent.type,
        category=agent.category,
        tags=tags if isinstance(tags, list) else [],
        created_by=agent.created_by,
        team=agent.team,
        tenant_id=agent.tenant_id,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


def version_to_model(version: AgentVersion) -> AgentVersionModel:
    """Convert SQLAlchemy AgentVersion to Pydantic model."""
    semantic = None
    if version.semantic_version:
        try:
            semantic = SemanticVersion(**version.semantic_version)
        except Exception:
            pass

    capabilities = version.capabilities
    if isinstance(capabilities, dict):
        capabilities = capabilities.get("capabilities", [])

    return AgentVersionModel(
        version_id=version.version_id,
        agent_id=version.agent_id,
        version=version.version,
        semantic_version=semantic,
        lifecycle_state=LifecycleState(version.lifecycle_state),
        container_image=version.container_image,
        image_digest=version.image_digest,
        metadata=version.metadata,
        runtime_requirements=version.runtime_requirements,
        capabilities=capabilities,
        created_at=version.created_at,
        published_at=version.published_at,
        deprecated_at=version.deprecated_at,
    )


async def log_audit(
    session,
    action: str,
    agent_id: Optional[str] = None,
    version_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    performed_by: Optional[str] = None,
    details: Optional[dict] = None,
    request: Optional[Request] = None,
    status: str = "success",
) -> None:
    """Create an audit log entry."""
    ip_address = None
    user_agent = None

    if request:
        # Get client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        else:
            ip_address = request.client.host if request.client else None

        user_agent = request.headers.get("User-Agent")

    audit_entry = AuditLog(
        agent_id=agent_id,
        version_id=version_id,
        action=action,
        performed_by=performed_by,
        tenant_id=tenant_id,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=getattr(request.state, "request_id", None) if request else None,
        status=status,
    )

    session.add(audit_entry)


# =============================================================================
# Endpoint 1: POST /agents - Publish Agent
# =============================================================================


@router.post(
    "/agents",
    response_model=PublishResponse,
    status_code=201,
    responses={
        201: {"description": "Agent published successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Version already exists"},
    },
    summary="Publish a new agent or version",
    description="""
    Publish a new agent or a new version of an existing agent.

    If the agent_id doesn't exist, a new agent is created.
    If the agent_id exists, a new version is added (if version doesn't exist).

    The new version starts in 'draft' lifecycle state.
    """,
)
async def publish_agent(
    request: Request,
    body: PublishRequest,
) -> PublishResponse:
    """
    Publish a new agent version to the registry.

    Args:
        request: FastAPI request object
        body: Publish request with agent and version details

    Returns:
        PublishResponse with success status and version details

    Raises:
        HTTPException: If version already exists or validation fails
    """
    start_time = datetime.utcnow()
    logger.info(f"Publishing agent: {body.agent_id}:{body.version}")

    client = get_database_client()

    async with client.session() as session:
        # Check if agent exists
        result = await session.execute(
            select(Agent).where(Agent.agent_id == body.agent_id)
        )
        existing_agent = result.scalar_one_or_none()

        version_id = f"{body.agent_id}:{body.version}"

        # Check if version already exists
        if existing_agent:
            version_result = await session.execute(
                select(AgentVersion).where(AgentVersion.version_id == version_id)
            )
            if version_result.scalar_one_or_none():
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "VersionExists",
                        "message": f"Version {body.version} already exists for agent {body.agent_id}",
                    },
                )

        # Create or update agent
        if existing_agent:
            # Update existing agent metadata
            existing_agent.name = body.name
            existing_agent.description = body.description
            existing_agent.domain = body.domain
            existing_agent.type = body.type
            existing_agent.category = body.category
            existing_agent.tags = {"tags": body.tags} if body.tags else None
            existing_agent.updated_at = datetime.utcnow()
            agent = existing_agent
        else:
            # Create new agent
            agent = Agent(
                agent_id=body.agent_id,
                name=body.name,
                description=body.description,
                domain=body.domain,
                type=body.type,
                category=body.category,
                tags={"tags": body.tags} if body.tags else None,
                created_by=body.team,  # Use team as creator if no auth
                team=body.team,
                tenant_id=body.tenant_id,
            )
            session.add(agent)

        # Parse semantic version
        semantic_version = parse_semantic_version(body.version)

        # Prepare capabilities and runtime requirements as dicts
        capabilities_dict = None
        if body.capabilities:
            capabilities_dict = {"capabilities": [c.model_dump() for c in body.capabilities]}

        runtime_dict = None
        if body.runtime_requirements:
            runtime_dict = body.runtime_requirements.model_dump()

        # Create new version
        new_version = AgentVersion(
            version_id=version_id,
            agent_id=body.agent_id,
            version=body.version,
            semantic_version=semantic_version,
            lifecycle_state=DBLifecycleState.DRAFT,
            container_image=body.container_image,
            image_digest=body.image_digest,
            metadata=body.metadata,
            runtime_requirements=runtime_dict,
            capabilities=capabilities_dict,
            published_at=datetime.utcnow(),
        )
        session.add(new_version)

        # Create initial state transition
        state_transition = StateTransition(
            version_id=version_id,
            from_state=None,
            to_state=DBLifecycleState.DRAFT,
            transitioned_by=body.team,
            reason="Initial publish",
            metadata={"publish_request": body.model_dump(exclude_none=True)},
        )
        session.add(state_transition)

        # Audit log
        await log_audit(
            session=session,
            action="publish",
            agent_id=body.agent_id,
            version_id=version_id,
            tenant_id=body.tenant_id,
            performed_by=body.team,
            details={
                "version": body.version,
                "container_image": body.container_image,
            },
            request=request,
        )

        # Commit transaction (handled by context manager)
        await session.flush()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Published agent {body.agent_id}:{body.version} in {processing_time:.2f}ms"
        )

        return PublishResponse(
            success=True,
            agent_id=body.agent_id,
            version_id=version_id,
            version=body.version,
            lifecycle_state=LifecycleState.DRAFT,
            message=f"Successfully published {body.agent_id} version {body.version}",
            created_at=new_version.created_at,
        )


# =============================================================================
# Endpoint 2: GET /agents - List Agents (Paginated)
# =============================================================================


@router.get(
    "/agents",
    response_model=ListAgentsResponse,
    responses={
        200: {"description": "List of agents"},
    },
    summary="List all agents",
    description="""
    List all registered agents with pagination and filtering.

    Supports filtering by:
    - domain: Classification domain
    - type: Agent type
    - tenant_id: Tenant identifier
    - search: Text search in name and description
    """,
)
async def list_agents(
    request: Request,
    domain: Optional[str] = Query(None, description="Filter by domain"),
    type: Optional[str] = Query(None, description="Filter by type"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
) -> ListAgentsResponse:
    """
    List agents with pagination and filtering.

    Args:
        request: FastAPI request object
        domain: Filter by classification domain
        type: Filter by agent type
        tenant_id: Filter by tenant
        search: Text search in name and description
        page: Page number (1-indexed)
        page_size: Number of items per page
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)

    Returns:
        ListAgentsResponse with paginated agent list
    """
    logger.info(f"Listing agents: page={page}, page_size={page_size}")

    client = get_database_client()

    async with client.session() as session:
        # Build query
        query = select(Agent)

        # Apply filters
        if domain:
            query = query.where(Agent.domain == domain)
        if type:
            query = query.where(Agent.type == type)
        if tenant_id:
            query = query.where(Agent.tenant_id == tenant_id)
        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                or_(
                    Agent.name.ilike(search_pattern),
                    Agent.description.ilike(search_pattern),
                )
            )

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply sorting
        sort_column = getattr(Agent, sort_by, Agent.created_at)
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # Execute query
        result = await session.execute(query)
        agents = result.scalars().all()

        # Calculate pagination info
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1

        # Convert to response models
        agent_list = [agent_to_metadata(agent) for agent in agents]

        return ListAgentsResponse(
            agents=agent_list,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev,
        )


# =============================================================================
# Endpoint 3: GET /agents/{id} - Get Agent Details
# =============================================================================


@router.get(
    "/agents/{agent_id}",
    response_model=AgentDetail,
    responses={
        200: {"description": "Agent details"},
        404: {"model": ErrorResponse, "description": "Agent not found"},
    },
    summary="Get agent details",
    description="""
    Get detailed information about a specific agent, including all versions.
    """,
)
async def get_agent(
    request: Request,
    agent_id: str,
) -> AgentDetail:
    """
    Get detailed information about a specific agent.

    Args:
        request: FastAPI request object
        agent_id: Agent identifier

    Returns:
        AgentDetail with agent metadata and all versions

    Raises:
        HTTPException: If agent not found
    """
    logger.info(f"Getting agent details: {agent_id}")

    client = get_database_client()

    async with client.session() as session:
        # Query agent with versions
        result = await session.execute(
            select(Agent)
            .options(selectinload(Agent.versions))
            .where(Agent.agent_id == agent_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Agent '{agent_id}' not found",
                },
            )

        # Convert to response model
        agent_metadata = agent_to_metadata(agent)
        versions = [version_to_model(v) for v in agent.versions]

        # Sort versions by semantic version (descending)
        versions.sort(
            key=lambda v: (
                v.semantic_version.major if v.semantic_version else 0,
                v.semantic_version.minor if v.semantic_version else 0,
                v.semantic_version.patch if v.semantic_version else 0,
            ),
            reverse=True,
        )

        # Find latest non-deprecated version
        latest_version = None
        for v in versions:
            if v.lifecycle_state != LifecycleState.DEPRECATED:
                latest_version = v
                break

        return AgentDetail(
            agent=agent_metadata,
            versions=versions,
            latest_version=latest_version,
            version_count=len(versions),
        )


# =============================================================================
# Endpoint 4: POST /agents/{id}/promote - Promote Agent State
# =============================================================================


@router.post(
    "/agents/{agent_id}/promote",
    response_model=PromoteResponse,
    responses={
        200: {"description": "Agent promoted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid state transition"},
        404: {"model": ErrorResponse, "description": "Agent not found"},
    },
    summary="Promote agent to next lifecycle state",
    description="""
    Promote an agent version to a new lifecycle state.

    Valid state transitions:
    - draft -> experimental
    - experimental -> certified
    - experimental -> deprecated
    - certified -> deprecated
    """,
)
async def promote_agent(
    request: Request,
    agent_id: str,
    body: PromoteRequest,
    version: Optional[str] = Query(None, description="Version to promote (latest if not specified)"),
) -> PromoteResponse:
    """
    Promote an agent version to a new lifecycle state.

    Args:
        request: FastAPI request object
        agent_id: Agent identifier
        body: Promotion request with target state
        version: Optional version to promote (uses latest if not specified)

    Returns:
        PromoteResponse with transition details

    Raises:
        HTTPException: If agent not found or invalid state transition
    """
    logger.info(f"Promoting agent {agent_id} to {body.target_state.value}")

    client = get_database_client()

    async with client.session() as session:
        # Build query for version
        if version:
            version_id = f"{agent_id}:{version}"
            query = select(AgentVersion).where(AgentVersion.version_id == version_id)
        else:
            # Get latest version that's not deprecated
            query = (
                select(AgentVersion)
                .where(AgentVersion.agent_id == agent_id)
                .where(AgentVersion.lifecycle_state != DBLifecycleState.DEPRECATED)
                .order_by(AgentVersion.created_at.desc())
                .limit(1)
            )

        result = await session.execute(query)
        agent_version = result.scalar_one_or_none()

        if not agent_version:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Agent version not found for '{agent_id}'",
                },
            )

        # Validate state transition
        current_state = agent_version.lifecycle_state
        target_state = body.target_state.value

        if not DBLifecycleState.can_transition(current_state, target_state):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "InvalidTransition",
                    "message": f"Cannot transition from '{current_state}' to '{target_state}'",
                    "valid_transitions": DBLifecycleState.VALID_TRANSITIONS.get(
                        current_state, []
                    ),
                },
            )

        # Update version state
        agent_version.lifecycle_state = target_state
        if target_state == DBLifecycleState.DEPRECATED:
            agent_version.deprecated_at = datetime.utcnow()

        # Record state transition
        transition = StateTransition(
            version_id=agent_version.version_id,
            from_state=current_state,
            to_state=target_state,
            transitioned_by=body.promoted_by,
            reason=body.reason,
            metadata=body.metadata,
        )
        session.add(transition)

        # Audit log
        await log_audit(
            session=session,
            action="promote",
            agent_id=agent_id,
            version_id=agent_version.version_id,
            performed_by=body.promoted_by,
            details={
                "from_state": current_state,
                "to_state": target_state,
                "reason": body.reason,
            },
            request=request,
        )

        await session.flush()

        logger.info(
            f"Promoted {agent_version.version_id}: {current_state} -> {target_state}"
        )

        return PromoteResponse(
            success=True,
            version_id=agent_version.version_id,
            from_state=LifecycleState(current_state),
            to_state=LifecycleState(target_state),
            message=f"Successfully promoted to {target_state}",
            transitioned_at=transition.transitioned_at,
        )


# =============================================================================
# Additional Endpoints (Bonus)
# =============================================================================


@router.get(
    "/agents/{agent_id}/versions/{version}",
    response_model=AgentVersionModel,
    responses={
        200: {"description": "Version details"},
        404: {"model": ErrorResponse, "description": "Version not found"},
    },
    summary="Get specific version",
    description="Get details for a specific agent version.",
)
async def get_version(
    request: Request,
    agent_id: str,
    version: str,
) -> AgentVersionModel:
    """Get a specific version of an agent."""
    logger.info(f"Getting version: {agent_id}:{version}")

    client = get_database_client()

    async with client.session() as session:
        version_id = f"{agent_id}:{version}"
        result = await session.execute(
            select(AgentVersion).where(AgentVersion.version_id == version_id)
        )
        agent_version = result.scalar_one_or_none()

        if not agent_version:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Version '{version}' not found for agent '{agent_id}'",
                },
            )

        return version_to_model(agent_version)
