# -*- coding: utf-8 -*-
"""
Agent Registry Service Setup - AGENT-FOUND-007: Agent Registry & Service Catalog

Provides ``configure_agent_registry(app)`` which wires up the Agent
Registry SDK (registry, health checker, dependency resolver, capability
matcher, provenance tracker) and mounts the REST API.

Also exposes ``get_agent_registry(app)`` for programmatic access and
the ``AgentRegistryService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.agent_registry.setup import configure_agent_registry
    >>> app = FastAPI()
    >>> configure_agent_registry(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-007 Agent Registry & Service Catalog
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from greenlang.agent_registry.config import AgentRegistryConfig, get_config
from greenlang.agent_registry.registry import AgentRegistry
from greenlang.agent_registry.health_checker import HealthChecker
from greenlang.agent_registry.dependency_resolver import DependencyResolver
from greenlang.agent_registry.capability_matcher import CapabilityMatcher
from greenlang.agent_registry.provenance import ProvenanceTracker
from greenlang.agent_registry.metrics import PROMETHEUS_AVAILABLE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# AgentRegistryService facade
# ===================================================================

_singleton_lock = threading.Lock()
_singleton_instance: Optional[AgentRegistryService] = None


class AgentRegistryService:
    """Unified facade over the Agent Registry SDK.

    Aggregates AgentRegistry, HealthChecker, DependencyResolver,
    CapabilityMatcher, and ProvenanceTracker through a single entry
    point with lifecycle management and metrics.

    Attributes:
        registry: AgentRegistry instance.
        health_checker: HealthChecker instance.
        dependency_resolver: DependencyResolver instance.
        capability_matcher: CapabilityMatcher instance.
        provenance: ProvenanceTracker instance.
        config: AgentRegistryConfig instance.

    Example:
        >>> service = AgentRegistryService()
        >>> provenance_hash = service.registry.register_agent(metadata)
        >>> result = service.health_checker.check_health("GL-MRV-X-001")
    """

    def __init__(
        self,
        config: Optional[AgentRegistryConfig] = None,
    ) -> None:
        """Initialize the Agent Registry Service facade.

        Args:
            config: Optional config. Uses global config if None.
        """
        self.config = config or get_config()

        # Initialize sub-components
        self.registry = AgentRegistry(config=self.config)
        self.health_checker = HealthChecker(
            registry=self.registry, config=self.config,
        )
        self.dependency_resolver = DependencyResolver(registry=self.registry)
        self.capability_matcher = CapabilityMatcher(registry=self.registry)
        self.provenance = ProvenanceTracker()

        # Internal state
        self._started = False
        self._start_time: Optional[float] = None

        # Internal metrics
        self._total_operations = 0

        logger.info("AgentRegistryService facade created")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the agent registry service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("AgentRegistryService already started; skipping")
            return

        logger.info("AgentRegistryService starting up...")
        self._start_time = time.time()
        self._started = True
        logger.info("AgentRegistryService startup complete")

    def shutdown(self) -> None:
        """Shutdown the agent registry service and release resources."""
        if not self._started:
            return

        self._started = False
        self._start_time = None
        logger.info("AgentRegistryService shut down")

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def register_agent(self, metadata: Any) -> str:
        """Register an agent and record provenance.

        Args:
            metadata: AgentMetadataEntry to register.

        Returns:
            SHA-256 provenance hash.
        """
        self._total_operations += 1
        provenance_hash = self.registry.register_agent(metadata)

        if self.config.enable_audit:
            self.provenance.record(
                entity_type="agent",
                entity_id=metadata.agent_id,
                action="register",
                data_hash=provenance_hash,
            )

        return provenance_hash

    def unregister_agent(
        self, agent_id: str, version: Optional[str] = None,
    ) -> bool:
        """Unregister an agent and record provenance.

        Args:
            agent_id: Agent ID to remove.
            version: Specific version or None for all.

        Returns:
            True if agent was removed.
        """
        self._total_operations += 1
        removed = self.registry.unregister_agent(agent_id, version)

        if removed and self.config.enable_audit:
            self.provenance.record(
                entity_type="agent",
                entity_id=agent_id,
                action="unregister",
                data_hash="",
            )

        return removed

    def check_health(self, agent_id: str) -> Any:
        """Run a health check on an agent.

        Args:
            agent_id: Agent ID to probe.

        Returns:
            HealthCheckResult.
        """
        self._total_operations += 1
        return self.health_checker.check_health(agent_id)

    def resolve_dependencies(
        self,
        agent_ids: List[str],
        include_optional: bool = False,
        fail_on_missing: bool = True,
    ) -> Any:
        """Resolve dependencies for a set of agents.

        Args:
            agent_ids: Root agent IDs to resolve.
            include_optional: Include optional dependencies.
            fail_on_missing: Fail if dependencies missing.

        Returns:
            DependencyResolutionOutput.
        """
        self._total_operations += 1
        return self.dependency_resolver.resolve(
            agent_ids, include_optional, fail_on_missing,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent registry service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.time() - self._start_time

        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "uptime_seconds": round(uptime, 2),
            "total_operations": self._total_operations,
            "agents_registered": self.registry.count,
            "health_checks_total": self.health_checker.total_checks,
            "provenance_entries": self.provenance.entry_count,
            "config": {
                "max_agents": self.config.max_agents,
                "max_versions_per_agent": self.config.max_versions_per_agent,
                "strict_mode": self.config.strict_mode,
                "enable_hot_reload": self.config.enable_hot_reload,
                "enable_audit": self.config.enable_audit,
            },
        }

    def get_service_catalog(self) -> List[Dict[str, Any]]:
        """Get the full service catalog.

        Returns:
            List of ServiceCatalogEntry dicts for all agents.
        """
        from greenlang.agent_registry.models import ServiceCatalogEntry
        entries: List[Dict[str, Any]] = []
        for agent_id in self.registry.get_all_agent_ids():
            metadata = self.registry.get_agent(agent_id)
            if metadata is not None:
                entry = ServiceCatalogEntry.from_metadata(metadata)
                entries.append(entry.model_dump(mode="json"))
        return entries


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> AgentRegistryService:
    """Get or create the singleton AgentRegistryService instance.

    Returns:
        The singleton AgentRegistryService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = AgentRegistryService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_agent_registry(
    app: Any,
    config: Optional[AgentRegistryConfig] = None,
) -> AgentRegistryService:
    """Configure the Agent Registry Service on a FastAPI application.

    Creates the AgentRegistryService, stores it in app.state, mounts
    the agent registry API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional agent registry config.

    Returns:
        AgentRegistryService instance.
    """
    global _singleton_instance

    service = AgentRegistryService(config=config)

    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.agent_registry_service = service

    # Mount API router
    try:
        from greenlang.agent_registry.api.router import router as registry_router
        if registry_router is not None:
            app.include_router(registry_router)
            logger.info("Agent Registry API router mounted")
    except ImportError:
        logger.warning("Agent Registry router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Agent Registry service configured on app")
    return service


def get_agent_registry(app: Any) -> AgentRegistryService:
    """Get the AgentRegistryService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        AgentRegistryService instance.

    Raises:
        RuntimeError: If agent registry service not configured.
    """
    service = getattr(app.state, "agent_registry_service", None)
    if service is None:
        raise RuntimeError(
            "Agent Registry service not configured. "
            "Call configure_agent_registry(app) first."
        )
    return service


def get_router() -> Any:
    """Get the agent registry API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.agent_registry.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "AgentRegistryService",
    "configure_agent_registry",
    "get_agent_registry",
    "get_router",
]
