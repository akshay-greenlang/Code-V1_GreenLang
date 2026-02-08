# -*- coding: utf-8 -*-
"""
Agent Registry REST API Router - AGENT-FOUND-007: Agent Registry & Service Catalog

FastAPI router providing 20 endpoints for agent registration, discovery,
health checking, dependency resolution, capability matching, service
catalog, provenance tracking, and metrics.

All endpoints are mounted under ``/api/v1/agent-registry``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; agent registry router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RegisterAgentRequest(BaseModel):
        """Request body for registering an agent."""
        agent_id: str = Field(..., description="Unique agent identifier")
        name: str = Field(..., description="Human-readable agent name")
        description: str = Field(..., description="Agent description")
        version: str = Field(..., description="Semantic version string")
        layer: str = Field(..., description="Agent layer")
        sectors: List[str] = Field(default_factory=list, description="Applicable sectors")
        capabilities: List[Dict[str, Any]] = Field(default_factory=list, description="Agent capabilities")
        tags: List[str] = Field(default_factory=list, description="Searchable tags")
        execution_mode: str = Field(default="legacy_http", description="Execution mode")
        legacy_http_config: Optional[Dict[str, Any]] = Field(None, description="Legacy HTTP config")
        container_spec: Optional[Dict[str, Any]] = Field(None, description="Container spec for GLIP v1")
        resource_profile: Optional[Dict[str, Any]] = Field(None, description="Resource profile")
        dependencies: List[Dict[str, Any]] = Field(default_factory=list, description="Dependencies")
        author: Optional[str] = Field(None, description="Agent author")
        documentation_url: Optional[str] = Field(None, description="Documentation URL")

    class UpdateAgentRequest(BaseModel):
        """Request body for updating an agent."""
        name: Optional[str] = Field(None, description="New name")
        description: Optional[str] = Field(None, description="New description")
        tags: Optional[List[str]] = Field(None, description="New tags")
        health_status: Optional[str] = Field(None, description="New health status")
        documentation_url: Optional[str] = Field(None, description="New docs URL")

    class SetHealthRequest(BaseModel):
        """Request body for setting agent health."""
        status: str = Field(..., description="Health status value")
        version: Optional[str] = Field(None, description="Specific version")

    class ResolveRequest(BaseModel):
        """Request body for dependency resolution."""
        agent_ids: List[str] = Field(..., description="Agent IDs to resolve")
        include_optional: bool = Field(default=False, description="Include optional deps")
        fail_on_missing: bool = Field(default=True, description="Fail on missing deps")

    class QueryRequest(BaseModel):
        """Request body for advanced queries."""
        layer: Optional[str] = Field(None, description="Filter by layer")
        sector: Optional[str] = Field(None, description="Filter by sector")
        capability: Optional[str] = Field(None, description="Filter by capability")
        tags: List[str] = Field(default_factory=list, description="Filter by tags")
        health_status: Optional[str] = Field(None, description="Filter by health")
        search_text: Optional[str] = Field(None, description="Text search")
        limit: int = Field(default=100, ge=1, le=1000, description="Max results")
        offset: int = Field(default=0, ge=0, description="Offset")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/agent-registry",
        tags=["agent-registry"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Any) -> Any:
    """Extract AgentRegistryService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        AgentRegistryService instance.

    Raises:
        HTTPException: If the service is not configured.
    """
    service = getattr(request.app.state, "agent_registry_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Agent Registry service not configured",
        )
    return service


# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    # 1. Health check
    @router.get("/health", summary="Agent Registry health check")
    async def health_check(request: Request) -> Dict[str, Any]:
        """Return agent registry service health status."""
        service = _get_service(request)
        return {
            "status": "healthy",
            "started": service._started,
            "agents_registered": service.registry.count,
            "health_checks_performed": service.health_checker.total_checks,
        }

    # 2. Register agent
    @router.post("/agents", summary="Register an agent", status_code=201)
    async def register_agent(body: RegisterAgentRequest, request: Request) -> Dict[str, Any]:
        """Register a new agent in the registry."""
        from greenlang.agent_registry.models import AgentMetadataEntry
        service = _get_service(request)
        try:
            metadata = AgentMetadataEntry(**body.model_dump())
            provenance_hash = service.registry.register_agent(metadata)
            if service.config.enable_audit:
                service.provenance.record(
                    "agent", body.agent_id, "register", provenance_hash,
                )
            return {
                "agent_id": body.agent_id,
                "version": body.version,
                "provenance_hash": provenance_hash,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 3. List agents
    @router.get("/agents", summary="List registered agents")
    async def list_agents(
        request: Request,
        layer: Optional[str] = Query(None),
        sector: Optional[str] = Query(None),
        capability: Optional[str] = Query(None),
        tag: Optional[str] = Query(None),
        health: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        """List agents with optional filters."""
        from greenlang.agent_registry.models import (
            AgentHealthStatus, AgentLayer, SectorClassification,
        )
        service = _get_service(request)
        try:
            layer_enum = AgentLayer(layer) if layer else None
            sector_enum = SectorClassification(sector) if sector else None
            health_enum = AgentHealthStatus(health) if health else None
            tags = [tag] if tag else []

            agents = service.registry.list_agents(
                layer=layer_enum,
                sector=sector_enum,
                capability=capability,
                tags=tags,
                health=health_enum,
                search=search,
                limit=limit,
                offset=offset,
            )
            return {
                "agents": [a.model_dump(mode="json") for a in agents],
                "count": len(agents),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 4. Get agent by ID
    @router.get("/agents/{agent_id}", summary="Get agent by ID")
    async def get_agent(
        agent_id: str, request: Request,
        version: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Get a specific agent's metadata."""
        service = _get_service(request)
        metadata = service.registry.get_agent(agent_id, version)
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return metadata.model_dump(mode="json")

    # 5. Update agent
    @router.put("/agents/{agent_id}", summary="Update an agent")
    async def update_agent(
        agent_id: str, body: UpdateAgentRequest, request: Request,
    ) -> Dict[str, Any]:
        """Update an existing agent's metadata."""
        service = _get_service(request)
        try:
            updates = {k: v for k, v in body.model_dump().items() if v is not None}
            provenance_hash = service.registry.update_agent(agent_id, updates)
            if service.config.enable_audit:
                service.provenance.record(
                    "agent", agent_id, "update", provenance_hash,
                )
            return {"agent_id": agent_id, "provenance_hash": provenance_hash}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 6. Unregister agent
    @router.delete("/agents/{agent_id}", summary="Unregister an agent")
    async def unregister_agent(
        agent_id: str, request: Request,
        version: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Remove an agent from the registry."""
        service = _get_service(request)
        removed = service.registry.unregister_agent(agent_id, version)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        if service.config.enable_audit:
            service.provenance.record("agent", agent_id, "unregister", "")
        return {"removed": True, "agent_id": agent_id, "version": version}

    # 7. List versions
    @router.get("/agents/{agent_id}/versions", summary="List agent versions")
    async def list_versions(agent_id: str, request: Request) -> Dict[str, Any]:
        """List all versions of an agent."""
        service = _get_service(request)
        versions = service.registry.list_versions(agent_id)
        if not versions:
            raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
        return {"agent_id": agent_id, "versions": versions}

    # 8. Hot-reload agent
    @router.post("/agents/{agent_id}/reload", summary="Hot-reload an agent")
    async def hot_reload_agent(
        agent_id: str, body: RegisterAgentRequest, request: Request,
    ) -> Dict[str, Any]:
        """Hot-reload an agent with new metadata."""
        from greenlang.agent_registry.models import AgentMetadataEntry
        service = _get_service(request)
        try:
            metadata = AgentMetadataEntry(**body.model_dump())
            success = service.registry.hot_reload_agent(agent_id, metadata)
            if success and service.config.enable_audit:
                service.provenance.record(
                    "agent", agent_id, "reload", metadata.provenance_hash,
                )
            return {"agent_id": agent_id, "reloaded": success}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 9. Check agent health
    @router.get("/agents/{agent_id}/health", summary="Check agent health")
    async def check_agent_health(agent_id: str, request: Request) -> Dict[str, Any]:
        """Run a health check probe on an agent."""
        service = _get_service(request)
        result = service.health_checker.check_health(agent_id)
        return result.model_dump(mode="json")

    # 10. Set agent health
    @router.put("/agents/{agent_id}/health", summary="Set agent health")
    async def set_agent_health(
        agent_id: str, body: SetHealthRequest, request: Request,
    ) -> Dict[str, Any]:
        """Manually set health status for an agent."""
        from greenlang.agent_registry.models import AgentHealthStatus
        service = _get_service(request)
        try:
            status = AgentHealthStatus(body.status)
            success = service.health_checker.set_health(agent_id, status, body.version)
            if not success:
                raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
            return {"agent_id": agent_id, "health_status": body.status}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # 11. Get health history
    @router.get("/agents/{agent_id}/health/history", summary="Get health history")
    async def get_health_history(
        agent_id: str, request: Request,
        limit: int = Query(50, ge=1, le=500),
    ) -> Dict[str, Any]:
        """Get health check history for an agent."""
        service = _get_service(request)
        history = service.health_checker.get_health_history(agent_id, limit)
        return {
            "agent_id": agent_id,
            "history": [h.model_dump(mode="json") for h in history],
        }

    # 12. Get unhealthy agents
    @router.get("/health/unhealthy", summary="Get unhealthy agents")
    async def get_unhealthy_agents(request: Request) -> Dict[str, Any]:
        """Get all unhealthy or degraded agents."""
        service = _get_service(request)
        unhealthy = service.health_checker.get_unhealthy_agents()
        return {"unhealthy_agents": unhealthy, "count": len(unhealthy)}

    # 13. Health summary
    @router.get("/health/summary", summary="Get health summary")
    async def get_health_summary(request: Request) -> Dict[str, Any]:
        """Get health status counts across all agents."""
        service = _get_service(request)
        return service.health_checker.get_health_summary()

    # 14. Resolve dependencies
    @router.post("/dependencies/resolve", summary="Resolve agent dependencies")
    async def resolve_dependencies(
        body: ResolveRequest, request: Request,
    ) -> Dict[str, Any]:
        """Resolve dependency graph and return topological order."""
        service = _get_service(request)
        result = service.dependency_resolver.resolve(
            body.agent_ids, body.include_optional, body.fail_on_missing,
        )
        return result.model_dump(mode="json")

    # 15. Get dependents
    @router.get("/dependencies/{agent_id}/dependents", summary="Get dependents")
    async def get_dependents(agent_id: str, request: Request) -> Dict[str, Any]:
        """Get agents that depend on the given agent."""
        service = _get_service(request)
        dependents = service.dependency_resolver.get_dependents(agent_id)
        return {"agent_id": agent_id, "dependents": dependents}

    # 16. Get dependency tree
    @router.get("/dependencies/{agent_id}/tree", summary="Get dependency tree")
    async def get_dependency_tree(agent_id: str, request: Request) -> Dict[str, Any]:
        """Get nested dependency tree for an agent."""
        service = _get_service(request)
        tree = service.dependency_resolver.get_dependency_tree(agent_id)
        return tree

    # 17. Capability matrix
    @router.get("/capabilities/matrix", summary="Get capability matrix")
    async def get_capability_matrix(request: Request) -> Dict[str, Any]:
        """Get the capability-to-agent mapping matrix."""
        service = _get_service(request)
        matrix = service.capability_matcher.get_capability_matrix()
        return {"matrix": matrix, "capability_count": len(matrix)}

    # 18. Get provenance chain
    @router.get("/provenance/{entity_id}", summary="Get provenance chain")
    async def get_provenance_chain(entity_id: str, request: Request) -> List[Dict[str, Any]]:
        """Get the provenance chain for an entity."""
        service = _get_service(request)
        chain = service.provenance.get_chain(entity_id)
        return [e.model_dump(mode="json") for e in chain]

    # 19. Registry statistics
    @router.get("/statistics", summary="Get registry statistics")
    async def get_statistics(request: Request) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        service = _get_service(request)
        return service.registry.get_statistics()

    # 20. Metrics summary
    @router.get("/metrics", summary="Get metrics summary")
    async def get_metrics(request: Request) -> Dict[str, Any]:
        """Get service metrics summary."""
        service = _get_service(request)
        return service.get_metrics()


__all__ = [
    "router",
    "FASTAPI_AVAILABLE",
]
