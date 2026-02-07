# -*- coding: utf-8 -*-
"""
Assumptions Service Setup - AGENT-FOUND-004: Assumptions Registry

Provides ``configure_assumptions_service(app)`` which wires up the
Assumptions Registry SDK (registry, scenarios, validator, provenance,
dependencies) and mounts the REST API.

Also exposes ``get_assumptions_service(app)`` for programmatic access
and the ``AssumptionsService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.assumptions.setup import configure_assumptions_service
    >>> app = FastAPI()
    >>> configure_assumptions_service(app)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-004 Assumptions Registry
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from greenlang.assumptions.config import AssumptionsConfig, get_config
from greenlang.assumptions.registry import AssumptionRegistry
from greenlang.assumptions.scenarios import ScenarioManager
from greenlang.assumptions.validator import AssumptionValidator
from greenlang.assumptions.provenance import ProvenanceTracker
from greenlang.assumptions.dependencies import DependencyTracker
from greenlang.assumptions.metrics import PROMETHEUS_AVAILABLE

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
# AssumptionsService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["AssumptionsService"] = None


class AssumptionsService:
    """Unified facade over the Assumptions Registry SDK.

    Manages assumption registry, scenario management, validation,
    provenance tracking, and dependency analysis through a single
    entry point.

    Attributes:
        registry: AssumptionRegistry instance.
        scenario_manager: ScenarioManager instance.
        validator: AssumptionValidator instance.
        provenance: ProvenanceTracker instance.
        dependency_tracker: DependencyTracker instance.
        config: AssumptionsConfig instance.

    Example:
        >>> service = AssumptionsService()
        >>> a = service.registry.create(
        ...     "ef.test", "Test", "Test EF", "emission_factor",
        ...     "float", 10.0, user_id="test", change_reason="init",
        ...     metadata_source="EPA",
        ... )
        >>> print(service.registry.get_value("ef.test"))
    """

    def __init__(
        self,
        config: Optional[AssumptionsConfig] = None,
    ) -> None:
        """Initialize the Assumptions Service facade.

        Args:
            config: Optional assumptions config. Uses global config if None.
        """
        self.config = config or get_config()
        self.provenance = ProvenanceTracker()
        self.validator = AssumptionValidator()
        self.registry = AssumptionRegistry(
            config=self.config,
            validator=self.validator,
            provenance=self.provenance,
        )
        self.scenario_manager = ScenarioManager(config=self.config)
        self.dependency_tracker = DependencyTracker()
        self._started = False

        logger.info("AssumptionsService facade created")

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_registry(self) -> AssumptionRegistry:
        """Get the AssumptionRegistry instance.

        Returns:
            The AssumptionRegistry used by this service.
        """
        return self.registry

    def get_scenario_manager(self) -> ScenarioManager:
        """Get the ScenarioManager instance.

        Returns:
            The ScenarioManager used by this service.
        """
        return self.scenario_manager

    def get_validator(self) -> AssumptionValidator:
        """Get the AssumptionValidator instance.

        Returns:
            The AssumptionValidator used by this service.
        """
        return self.validator

    def get_provenance(self) -> ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            The ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_dependency_tracker(self) -> DependencyTracker:
        """Get the DependencyTracker instance.

        Returns:
            The DependencyTracker used by this service.
        """
        return self.dependency_tracker

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get assumptions service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "assumptions_count": self.registry.count,
            "scenarios_count": self.scenario_manager.count,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the assumptions service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("AssumptionsService already started; skipping")
            return

        logger.info("AssumptionsService starting up...")
        self._started = True
        logger.info("AssumptionsService startup complete")

    def shutdown(self) -> None:
        """Shutdown the assumptions service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("AssumptionsService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> AssumptionsService:
    """Get or create the singleton AssumptionsService instance.

    Returns:
        The singleton AssumptionsService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = AssumptionsService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


def configure_assumptions_service(
    app: Any,
    config: Optional[AssumptionsConfig] = None,
) -> AssumptionsService:
    """Configure the Assumptions Service on a FastAPI application.

    Creates the AssumptionsService, stores it in app.state, mounts
    the assumptions API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional assumptions config.

    Returns:
        AssumptionsService instance.
    """
    global _singleton_instance

    service = AssumptionsService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.assumptions_service = service

    # Mount assumptions API router
    try:
        from greenlang.assumptions.api.router import router as assumptions_router
        if assumptions_router is not None:
            app.include_router(assumptions_router)
            logger.info("Assumptions service API router mounted")
    except ImportError:
        logger.warning("Assumptions router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("Assumptions service configured on app")
    return service


def get_assumptions_service(app: Any) -> AssumptionsService:
    """Get the AssumptionsService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        AssumptionsService instance.

    Raises:
        RuntimeError: If assumptions service not configured.
    """
    service = getattr(app.state, "assumptions_service", None)
    if service is None:
        raise RuntimeError(
            "Assumptions service not configured. "
            "Call configure_assumptions_service(app) first."
        )
    return service


def get_router() -> Any:
    """Get the assumptions API router.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.assumptions.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "AssumptionsService",
    "configure_assumptions_service",
    "get_assumptions_service",
    "get_router",
]
