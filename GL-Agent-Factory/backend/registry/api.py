"""
Agent Registry FastAPI Endpoints

This module provides REST API endpoints for the Agent Registry:
- POST /agents - Register new agent
- GET /agents - List all agents (with filters)
- GET /agents/{id} - Get agent by ID
- GET /agents/{id}/versions - List versions
- PUT /agents/{id} - Update agent
- DELETE /agents/{id} - Soft delete
- POST /agents/{id}/publish - Publish version
- GET /agents/search - Search agents

All endpoints support multi-tenant isolation and comprehensive audit logging.

Example:
    >>> from fastapi import FastAPI
    >>> from backend.registry.api import router
    >>> app = FastAPI()
    >>> app.include_router(router, prefix="/v1/registry")
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse

from backend.registry.models import (
    AgentStatus,
    AgentCreateRequest,
    AgentUpdateRequest,
    AgentResponse,
    AgentListResponse,
    AgentSearchRequest,
    VersionCreateRequest,
    VersionResponse,
    PublishRequest,
    PublishResponse,
    CertificationStatus,
)

logger = logging.getLogger(__name__)

# Create router with prefix for registry endpoints
router = APIRouter(
    prefix="/registry",
    tags=["Agent Registry"],
    responses={
        404: {"description": "Agent not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)


# Dependency injection stubs (to be replaced with actual implementations)
async def get_registry_service():
    """
    Get the registry service instance.

    In production, this would return the actual AgentRegistryService
    with database session dependency.
    """
    from backend.registry.service import AgentRegistryService
    # TODO: Initialize with actual database session
    return AgentRegistryService(session=None)


async def get_current_user():
    """
    Get the current authenticated user.

    In production, this would validate JWT and return user info.
    """
    # Placeholder - return mock user
    return {"id": "user-001", "tenant_id": "tenant-001", "email": "user@greenlang.io"}


# =============================================================================
# Agent CRUD Endpoints
# =============================================================================


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new agent",
    description="""
    Register a new agent in the registry.

    The agent is created in DRAFT status and must be published to become
    available for download. A unique name is required.

    **Permissions:** Authenticated user with 'agent:create' permission.
    """,
    responses={
        201: {"description": "Agent created successfully"},
        409: {"description": "Agent with this name already exists"},
    },
)
async def create_agent(
    request: AgentCreateRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> AgentResponse:
    """
    Create a new agent in the registry.

    Args:
        request: Agent creation request with name, version, category, etc.
        background_tasks: FastAPI background task queue
        service: Registry service instance
        current_user: Authenticated user information

    Returns:
        Created agent record

    Raises:
        HTTPException: 409 if agent name already exists
    """
    logger.info(f"Creating agent: {request.name} by user {current_user['id']}")

    try:
        agent = await service.create_agent(
            name=request.name,
            version=request.version,
            description=request.description,
            category=request.category,
            author=request.author,
            pack_yaml=request.pack_yaml,
            generated_code=request.generated_code,
            tags=request.tags,
            regulatory_frameworks=request.regulatory_frameworks,
            documentation_url=request.documentation_url,
            repository_url=request.repository_url,
            license=request.license,
            tenant_id=current_user.get("tenant_id"),
        )

        # Schedule background audit log
        background_tasks.add_task(
            _log_audit_event,
            action="agent.create",
            agent_id=str(agent.id),
            user_id=current_user["id"],
        )

        return AgentResponse(
            id=agent.id,
            name=agent.name,
            version=agent.version,
            description=agent.description,
            category=agent.category,
            status=AgentStatus(agent.status),
            author=agent.author,
            checksum=agent.checksum,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            downloads=agent.downloads,
            tags=agent.tags,
            regulatory_frameworks=agent.regulatory_frameworks,
            certification_status=[],
            documentation_url=agent.documentation_url,
            repository_url=agent.repository_url,
            license=agent.license,
            version_count=1,
            latest_version=agent.version,
        )

    except ValueError as e:
        if "already exists" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent with name '{request.name}' already exists",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/agents",
    response_model=AgentListResponse,
    summary="List all agents",
    description="""
    Get a paginated list of agents with optional filtering.

    Supports filtering by category, status, tags, and regulatory frameworks.
    Results are paginated and can be sorted by various fields.

    **Permissions:** Authenticated user with 'agent:read' permission.
    """,
)
async def list_agents(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status (draft/published/deprecated)"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    frameworks: Optional[str] = Query(None, description="Filter by regulatory frameworks (comma-separated)"),
    author: Optional[str] = Query(None, description="Filter by author"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: str = Query("created_at", description="Sort field (name/created_at/downloads/updated_at)"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> AgentListResponse:
    """
    List agents with filtering and pagination.

    Args:
        category: Filter by agent category
        status: Filter by lifecycle status
        tags: Comma-separated list of tags to filter
        frameworks: Comma-separated list of regulatory frameworks
        author: Filter by agent author
        limit: Maximum number of results
        offset: Pagination offset
        sort_by: Field to sort by
        sort_order: Sort direction (asc/desc)
        service: Registry service instance
        current_user: Authenticated user

    Returns:
        Paginated list of agents with metadata
    """
    logger.info(f"Listing agents: category={category}, status={status}, limit={limit}")

    # Parse comma-separated values
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    framework_list = [f.strip() for f in frameworks.split(",")] if frameworks else None

    # Get agents from service
    agents, total = await service.list_agents(
        category=category,
        status=status,
        tags=tag_list,
        regulatory_frameworks=framework_list,
        author=author,
        tenant_id=current_user.get("tenant_id"),
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )

    # Convert to response models
    agent_responses = [
        AgentResponse(
            id=a.id,
            name=a.name,
            version=a.version,
            description=a.description,
            category=a.category,
            status=AgentStatus(a.status),
            author=a.author,
            checksum=a.checksum,
            created_at=a.created_at,
            updated_at=a.updated_at,
            downloads=a.downloads,
            tags=a.tags or [],
            regulatory_frameworks=a.regulatory_frameworks or [],
            certification_status=[
                CertificationStatus(**c) for c in (a.certification_status or [])
            ],
            documentation_url=a.documentation_url,
            repository_url=a.repository_url,
            license=a.license,
            version_count=len(a.versions) if hasattr(a, "versions") and a.versions else 0,
            latest_version=a.version,
        )
        for a in agents
    ]

    return AgentListResponse(
        data=agent_responses,
        meta={
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
        },
    )


@router.get(
    "/agents/search",
    response_model=AgentListResponse,
    summary="Search agents",
    description="""
    Search for agents using full-text search.

    Searches across agent name, description, tags, and categories.
    Results are ranked by relevance.

    **Permissions:** Authenticated user with 'agent:read' permission.
    """,
)
async def search_agents(
    q: str = Query(..., min_length=1, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> AgentListResponse:
    """
    Search agents by text query.

    Args:
        q: Search query string
        category: Optional category filter
        status: Optional status filter
        tags: Optional tag filter (comma-separated)
        limit: Maximum results
        offset: Pagination offset
        service: Registry service
        current_user: Authenticated user

    Returns:
        Search results with relevance ranking
    """
    logger.info(f"Searching agents: query='{q}'")

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    agents, total = await service.search_agents(
        query=q,
        category=category,
        status=status,
        tags=tag_list,
        tenant_id=current_user.get("tenant_id"),
        limit=limit,
        offset=offset,
    )

    # Convert to response models
    agent_responses = [
        AgentResponse(
            id=a.id,
            name=a.name,
            version=a.version,
            description=a.description,
            category=a.category,
            status=AgentStatus(a.status),
            author=a.author,
            checksum=a.checksum,
            created_at=a.created_at,
            updated_at=a.updated_at,
            downloads=a.downloads,
            tags=a.tags or [],
            regulatory_frameworks=a.regulatory_frameworks or [],
            certification_status=[
                CertificationStatus(**c) for c in (a.certification_status or [])
            ],
            documentation_url=a.documentation_url,
            repository_url=a.repository_url,
            license=a.license,
            version_count=len(a.versions) if hasattr(a, "versions") and a.versions else 0,
            latest_version=a.version,
        )
        for a in agents
    ]

    return AgentListResponse(
        data=agent_responses,
        meta={
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
            "query": q,
        },
    )


@router.get(
    "/agents/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent by ID",
    description="""
    Retrieve a specific agent by its UUID.

    Returns full agent details including all versions and certifications.

    **Permissions:** Authenticated user with 'agent:read' permission.
    """,
)
async def get_agent(
    agent_id: UUID,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> AgentResponse:
    """
    Get agent by ID.

    Args:
        agent_id: Agent UUID
        service: Registry service
        current_user: Authenticated user

    Returns:
        Agent details

    Raises:
        HTTPException: 404 if agent not found
    """
    logger.info(f"Getting agent: {agent_id}")

    agent = await service.get_agent(
        agent_id=agent_id,
        tenant_id=current_user.get("tenant_id"),
    )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    # Get version count
    versions = await service.list_versions(agent_id=agent_id)
    latest_version = next((v for v in versions if v.is_latest), None)

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        version=agent.version,
        description=agent.description,
        category=agent.category,
        status=AgentStatus(agent.status),
        author=agent.author,
        checksum=agent.checksum,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        downloads=agent.downloads,
        tags=agent.tags or [],
        regulatory_frameworks=agent.regulatory_frameworks or [],
        certification_status=[
            CertificationStatus(**c) for c in (agent.certification_status or [])
        ],
        documentation_url=agent.documentation_url,
        repository_url=agent.repository_url,
        license=agent.license,
        version_count=len(versions),
        latest_version=latest_version.version if latest_version else agent.version,
    )


@router.put(
    "/agents/{agent_id}",
    response_model=AgentResponse,
    summary="Update agent",
    description="""
    Update agent metadata.

    Only description, tags, frameworks, and URLs can be updated.
    Name, category, and author are immutable after creation.

    **Permissions:** Agent owner or admin with 'agent:update' permission.
    """,
)
async def update_agent(
    agent_id: UUID,
    request: AgentUpdateRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> AgentResponse:
    """
    Update agent metadata.

    Args:
        agent_id: Agent UUID to update
        request: Update request with fields to modify
        background_tasks: Background task queue
        service: Registry service
        current_user: Authenticated user

    Returns:
        Updated agent

    Raises:
        HTTPException: 404 if agent not found, 403 if not authorized
    """
    logger.info(f"Updating agent: {agent_id}")

    # Build update dict from non-None values
    updates = {k: v for k, v in request.dict().items() if v is not None}

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided",
        )

    try:
        agent = await service.update_agent(
            agent_id=agent_id,
            updates=updates,
            tenant_id=current_user.get("tenant_id"),
            user_id=current_user["id"],
        )

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # Schedule audit log
        background_tasks.add_task(
            _log_audit_event,
            action="agent.update",
            agent_id=str(agent_id),
            user_id=current_user["id"],
            details={"updates": list(updates.keys())},
        )

        return AgentResponse(
            id=agent.id,
            name=agent.name,
            version=agent.version,
            description=agent.description,
            category=agent.category,
            status=AgentStatus(agent.status),
            author=agent.author,
            checksum=agent.checksum,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
            downloads=agent.downloads,
            tags=agent.tags or [],
            regulatory_frameworks=agent.regulatory_frameworks or [],
            certification_status=[
                CertificationStatus(**c) for c in (agent.certification_status or [])
            ],
            documentation_url=agent.documentation_url,
            repository_url=agent.repository_url,
            license=agent.license,
            version_count=0,
            latest_version=agent.version,
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this agent",
        )


@router.delete(
    "/agents/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent",
    description="""
    Soft delete an agent.

    Published agents are deprecated (not deleted). Draft agents are
    permanently removed. This action cannot be undone.

    **Permissions:** Agent owner or admin with 'agent:delete' permission.
    """,
)
async def delete_agent(
    agent_id: UUID,
    background_tasks: BackgroundTasks,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> None:
    """
    Delete (or deprecate) an agent.

    Args:
        agent_id: Agent UUID to delete
        background_tasks: Background task queue
        service: Registry service
        current_user: Authenticated user

    Raises:
        HTTPException: 404 if agent not found, 403 if not authorized
    """
    logger.info(f"Deleting agent: {agent_id}")

    try:
        success = await service.delete_agent(
            agent_id=agent_id,
            tenant_id=current_user.get("tenant_id"),
            user_id=current_user["id"],
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # Schedule audit log
        background_tasks.add_task(
            _log_audit_event,
            action="agent.delete",
            agent_id=str(agent_id),
            user_id=current_user["id"],
        )

    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this agent",
        )


# =============================================================================
# Version Management Endpoints
# =============================================================================


@router.get(
    "/agents/{agent_id}/versions",
    response_model=List[VersionResponse],
    summary="List agent versions",
    description="""
    Get all versions of an agent.

    Versions are sorted by semantic version (newest first).
    Includes changelog, breaking change indicators, and download counts.

    **Permissions:** Authenticated user with 'agent:read' permission.
    """,
)
async def list_versions(
    agent_id: UUID,
    include_deprecated: bool = Query(False, description="Include deprecated versions"),
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> List[VersionResponse]:
    """
    List all versions of an agent.

    Args:
        agent_id: Agent UUID
        include_deprecated: Whether to include deprecated versions
        service: Registry service
        current_user: Authenticated user

    Returns:
        List of version records

    Raises:
        HTTPException: 404 if agent not found
    """
    logger.info(f"Listing versions for agent: {agent_id}")

    versions = await service.list_versions(
        agent_id=agent_id,
        include_deprecated=include_deprecated,
    )

    return [
        VersionResponse(
            id=v.id,
            agent_id=v.agent_id,
            version=v.version,
            changelog=v.changelog,
            breaking_changes=v.breaking_changes,
            release_notes=v.release_notes,
            artifact_path=v.artifact_path,
            checksum=v.checksum,
            created_at=v.created_at,
            published_at=v.published_at,
            deprecated_at=v.deprecated_at,
            is_latest=v.is_latest,
            downloads=v.downloads,
        )
        for v in versions
    ]


@router.post(
    "/agents/{agent_id}/versions",
    response_model=VersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new version",
    description="""
    Create a new version of an agent.

    The new version must be higher than all existing versions.
    Optionally include updated pack_yaml and generated_code.

    **Permissions:** Agent owner or admin with 'agent:update' permission.
    """,
)
async def create_version(
    agent_id: UUID,
    request: VersionCreateRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> VersionResponse:
    """
    Create a new version for an agent.

    Args:
        agent_id: Agent UUID
        request: Version creation request
        background_tasks: Background task queue
        service: Registry service
        current_user: Authenticated user

    Returns:
        Created version record

    Raises:
        HTTPException: 404 if agent not found, 400 if version invalid
    """
    logger.info(f"Creating version {request.version} for agent: {agent_id}")

    try:
        version = await service.create_version(
            agent_id=agent_id,
            version=request.version,
            changelog=request.changelog,
            breaking_changes=request.breaking_changes,
            release_notes=request.release_notes,
            pack_yaml=request.pack_yaml,
            generated_code=request.generated_code,
            user_id=current_user["id"],
        )

        # Schedule audit log
        background_tasks.add_task(
            _log_audit_event,
            action="version.create",
            agent_id=str(agent_id),
            user_id=current_user["id"],
            details={"version": request.version},
        )

        return VersionResponse(
            id=version.id,
            agent_id=version.agent_id,
            version=version.version,
            changelog=version.changelog,
            breaking_changes=version.breaking_changes,
            release_notes=version.release_notes,
            artifact_path=version.artifact_path,
            checksum=version.checksum,
            created_at=version.created_at,
            published_at=version.published_at,
            deprecated_at=version.deprecated_at,
            is_latest=version.is_latest,
            downloads=version.downloads,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/agents/{agent_id}/versions/{version}",
    response_model=VersionResponse,
    summary="Get specific version",
    description="""
    Get a specific version of an agent.

    Use 'latest' as version to get the latest version.

    **Permissions:** Authenticated user with 'agent:read' permission.
    """,
)
async def get_version(
    agent_id: UUID,
    version: str,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> VersionResponse:
    """
    Get a specific version of an agent.

    Args:
        agent_id: Agent UUID
        version: Version string or 'latest'
        service: Registry service
        current_user: Authenticated user

    Returns:
        Version details

    Raises:
        HTTPException: 404 if version not found
    """
    logger.info(f"Getting version {version} for agent: {agent_id}")

    ver = await service.get_version(
        agent_id=agent_id,
        version=version,
    )

    if not ver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found for agent {agent_id}",
        )

    return VersionResponse(
        id=ver.id,
        agent_id=ver.agent_id,
        version=ver.version,
        changelog=ver.changelog,
        breaking_changes=ver.breaking_changes,
        release_notes=ver.release_notes,
        artifact_path=ver.artifact_path,
        checksum=ver.checksum,
        created_at=ver.created_at,
        published_at=ver.published_at,
        deprecated_at=ver.deprecated_at,
        is_latest=ver.is_latest,
        downloads=ver.downloads,
    )


# =============================================================================
# Publishing Workflow Endpoints
# =============================================================================


@router.post(
    "/agents/{agent_id}/publish",
    response_model=PublishResponse,
    summary="Publish agent version",
    description="""
    Publish an agent version to make it publicly available.

    Publishing:
    - Transitions agent status from draft to published
    - Makes the agent available for download
    - Optionally certifies for regulatory frameworks
    - Generates artifact and checksum

    **Permissions:** Agent owner or admin with 'agent:publish' permission.
    """,
)
async def publish_agent(
    agent_id: UUID,
    request: PublishRequest,
    background_tasks: BackgroundTasks,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> PublishResponse:
    """
    Publish an agent version.

    Args:
        agent_id: Agent UUID
        request: Publish request with version and optional certifications
        background_tasks: Background task queue
        service: Registry service
        current_user: Authenticated user

    Returns:
        Publish result with artifact URL and checksum

    Raises:
        HTTPException: 404 if agent not found, 400 if already published
    """
    logger.info(f"Publishing agent {agent_id} version {request.version}")

    try:
        result = await service.publish_agent(
            agent_id=agent_id,
            version=request.version,
            release_notes=request.release_notes,
            certifications=request.certifications,
            user_id=current_user["id"],
        )

        # Schedule background tasks
        background_tasks.add_task(
            _log_audit_event,
            action="agent.publish",
            agent_id=str(agent_id),
            user_id=current_user["id"],
            details={"version": request.version},
        )

        return PublishResponse(
            success=True,
            agent_id=agent_id,
            version=request.version,
            published_at=result.published_at,
            artifact_url=result.artifact_path,
            checksum=result.checksum,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/agents/{agent_id}/deprecate",
    response_model=AgentResponse,
    summary="Deprecate agent",
    description="""
    Deprecate an agent.

    Deprecation:
    - Marks the agent as deprecated
    - Agent remains available but shows deprecation warning
    - No new versions can be published

    **Permissions:** Agent owner or admin with 'agent:delete' permission.
    """,
)
async def deprecate_agent(
    agent_id: UUID,
    background_tasks: BackgroundTasks,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> AgentResponse:
    """
    Deprecate an agent.

    Args:
        agent_id: Agent UUID
        background_tasks: Background task queue
        service: Registry service
        current_user: Authenticated user

    Returns:
        Deprecated agent record

    Raises:
        HTTPException: 404 if agent not found
    """
    logger.info(f"Deprecating agent: {agent_id}")

    agent = await service.deprecate_agent(
        agent_id=agent_id,
        user_id=current_user["id"],
    )

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found",
        )

    # Schedule audit log
    background_tasks.add_task(
        _log_audit_event,
        action="agent.deprecate",
        agent_id=str(agent_id),
        user_id=current_user["id"],
    )

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        version=agent.version,
        description=agent.description,
        category=agent.category,
        status=AgentStatus(agent.status),
        author=agent.author,
        checksum=agent.checksum,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
        downloads=agent.downloads,
        tags=agent.tags or [],
        regulatory_frameworks=agent.regulatory_frameworks or [],
        certification_status=[
            CertificationStatus(**c) for c in (agent.certification_status or [])
        ],
        documentation_url=agent.documentation_url,
        repository_url=agent.repository_url,
        license=agent.license,
        version_count=0,
        latest_version=agent.version,
    )


# =============================================================================
# Download Endpoint
# =============================================================================


@router.get(
    "/agents/{agent_id}/download",
    summary="Download agent",
    description="""
    Download an agent package.

    Downloads the latest version by default. Specify version parameter
    for a specific version.

    **Permissions:** Authenticated user with 'agent:download' permission.
    """,
)
async def download_agent(
    agent_id: UUID,
    version: Optional[str] = Query(None, description="Version to download (default: latest)"),
    background_tasks: BackgroundTasks = None,
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> JSONResponse:
    """
    Download agent package.

    Args:
        agent_id: Agent UUID
        version: Optional version (default: latest)
        background_tasks: Background task queue
        service: Registry service
        current_user: Authenticated user

    Returns:
        Download URL and metadata

    Raises:
        HTTPException: 404 if agent/version not found
    """
    logger.info(f"Downloading agent: {agent_id}, version: {version or 'latest'}")

    download_info = await service.get_download(
        agent_id=agent_id,
        version=version,
        user_id=current_user["id"],
    )

    if not download_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found or not published",
        )

    # Increment download counter in background
    if background_tasks:
        background_tasks.add_task(
            service.increment_download,
            agent_id=agent_id,
            version=download_info["version"],
        )

    return JSONResponse(
        content={
            "agent_id": str(agent_id),
            "version": download_info["version"],
            "download_url": download_info["artifact_path"],
            "checksum": download_info["checksum"],
            "size_bytes": download_info.get("size_bytes"),
        }
    )


# =============================================================================
# Statistics Endpoint
# =============================================================================


@router.get(
    "/stats",
    response_model=Dict[str, Any],
    summary="Get registry statistics",
    description="""
    Get overall registry statistics.

    Includes total agents, downloads, categories, and recent activity.

    **Permissions:** Authenticated user with 'registry:read' permission.
    """,
)
async def get_stats(
    service=Depends(get_registry_service),
    current_user: Dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get registry statistics.

    Args:
        service: Registry service
        current_user: Authenticated user

    Returns:
        Statistics dictionary
    """
    stats = await service.get_statistics(
        tenant_id=current_user.get("tenant_id"),
    )

    return {
        "total_agents": stats.get("total_agents", 0),
        "total_versions": stats.get("total_versions", 0),
        "total_downloads": stats.get("total_downloads", 0),
        "by_status": stats.get("by_status", {}),
        "by_category": stats.get("by_category", {}),
        "recent_activity": stats.get("recent_activity", []),
    }


# =============================================================================
# Helper Functions
# =============================================================================


async def _log_audit_event(
    action: str,
    agent_id: str,
    user_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an audit event (background task).

    Args:
        action: Action type (agent.create, agent.update, etc.)
        agent_id: Agent identifier
        user_id: User who performed the action
        details: Additional details to log
    """
    logger.info(
        f"Audit: {action} on agent {agent_id} by user {user_id}",
        extra={
            "action": action,
            "agent_id": agent_id,
            "user_id": user_id,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    # TODO: Persist to audit log table
