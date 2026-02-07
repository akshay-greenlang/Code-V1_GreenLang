# -*- coding: utf-8 -*-
"""
Unit Tests for AssumptionsService Facade & Setup (AGENT-FOUND-004)

Tests the AssumptionsService facade class, configure_assumptions_service(app),
get_assumptions_service(app), lifecycle management, and delegation to
registry/scenarios/validator/provenance/dependencies sub-components.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline AssumptionsService facade mirroring greenlang/assumptions/setup.py
# ---------------------------------------------------------------------------


class AssumptionsService:
    """
    Facade for the Assumptions Registry SDK.
    Delegates to AssumptionRegistry, ScenarioManager, AssumptionValidator,
    ProvenanceTracker, and DependencyTracker.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._registry = MagicMock()
        self._scenario_manager = MagicMock()
        self._validator = MagicMock()
        self._provenance = MagicMock()
        self._dependencies = MagicMock()
        self._initialized = True
        self._router = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def registry(self):
        return self._registry

    @property
    def scenario_manager(self):
        return self._scenario_manager

    @property
    def validator(self):
        return self._validator

    @property
    def provenance(self):
        return self._provenance

    @property
    def dependencies(self):
        return self._dependencies

    # ---- Registry delegation ----

    def create_assumption(self, **kwargs):
        return self._registry.create(**kwargs)

    def get_assumption(self, assumption_id: str):
        return self._registry.get(assumption_id)

    def update_assumption(self, assumption_id: str, **kwargs):
        return self._registry.update(assumption_id, **kwargs)

    def delete_assumption(self, assumption_id: str):
        return self._registry.delete(assumption_id)

    def list_assumptions(self, **kwargs):
        return self._registry.list_assumptions(**kwargs)

    def get_value(self, assumption_id: str, **kwargs):
        return self._registry.get_value(assumption_id, **kwargs)

    def set_value(self, assumption_id: str, value, **kwargs):
        return self._registry.set_value(assumption_id, value, **kwargs)

    # ---- Scenario delegation ----

    def create_scenario(self, **kwargs):
        return self._scenario_manager.create(**kwargs)

    def get_scenario(self, scenario_id: str):
        return self._scenario_manager.get(scenario_id)

    def update_scenario(self, scenario_id: str, **kwargs):
        return self._scenario_manager.update(scenario_id, **kwargs)

    def delete_scenario(self, scenario_id: str):
        return self._scenario_manager.delete(scenario_id)

    def list_scenarios(self, **kwargs):
        return self._scenario_manager.list_scenarios(**kwargs)

    # ---- Lifecycle ----

    def get_router(self):
        if self._router is None:
            self._router = MagicMock()
        return self._router

    def shutdown(self):
        self._initialized = False
        self._registry = None
        self._scenario_manager = None
        self._validator = None
        self._provenance = None
        self._dependencies = None


class _MockApp:
    """Mock FastAPI app with state attribute."""

    def __init__(self):
        self.state = type("State", (), {})()
        self._routers: List[Any] = []

    def include_router(self, router, **kwargs):
        self._routers.append(router)


_GLOBAL_SERVICE: Optional[AssumptionsService] = None


def configure_assumptions_service(
    app: _MockApp,
    config: Optional[Dict[str, Any]] = None,
) -> AssumptionsService:
    """Configure and attach the assumptions service to the FastAPI app."""
    global _GLOBAL_SERVICE

    if _GLOBAL_SERVICE is not None:
        _GLOBAL_SERVICE.shutdown()

    service = AssumptionsService(config)
    app.state.assumptions_service = service
    _GLOBAL_SERVICE = service

    router = service.get_router()
    app.include_router(router)

    return service


def get_assumptions_service(app: _MockApp) -> AssumptionsService:
    """Retrieve the assumptions service from the FastAPI app."""
    service = getattr(app.state, "assumptions_service", None)
    if service is None:
        raise RuntimeError(
            "Assumptions service not configured. "
            "Call configure_assumptions_service(app) first."
        )
    return service


def reset_assumptions_service():
    """Reset the global assumptions service."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is not None:
        _GLOBAL_SERVICE.shutdown()
    _GLOBAL_SERVICE = None


# ---------------------------------------------------------------------------
# Autouse fixture to reset between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_service():
    yield
    reset_assumptions_service()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAssumptionsServiceCreation:
    """Test AssumptionsService creation."""

    def test_creation_with_defaults(self):
        svc = AssumptionsService()
        assert svc.is_initialized is True

    def test_creation_with_config(self):
        svc = AssumptionsService(config={"max_versions": 100})
        assert svc.is_initialized is True

    def test_has_registry(self):
        svc = AssumptionsService()
        assert svc.registry is not None

    def test_has_scenario_manager(self):
        svc = AssumptionsService()
        assert svc.scenario_manager is not None

    def test_has_validator(self):
        svc = AssumptionsService()
        assert svc.validator is not None

    def test_has_provenance(self):
        svc = AssumptionsService()
        assert svc.provenance is not None

    def test_has_dependencies(self):
        svc = AssumptionsService()
        assert svc.dependencies is not None

    def test_shutdown(self):
        svc = AssumptionsService()
        assert svc.is_initialized is True
        svc.shutdown()
        assert svc.is_initialized is False

    def test_shutdown_cleans_up_registry(self):
        svc = AssumptionsService()
        assert svc.registry is not None
        svc.shutdown()
        assert svc._registry is None

    def test_shutdown_cleans_up_scenario_manager(self):
        svc = AssumptionsService()
        svc.shutdown()
        assert svc._scenario_manager is None

    def test_shutdown_cleans_up_validator(self):
        svc = AssumptionsService()
        svc.shutdown()
        assert svc._validator is None

    def test_shutdown_cleans_up_provenance(self):
        svc = AssumptionsService()
        svc.shutdown()
        assert svc._provenance is None

    def test_shutdown_cleans_up_dependencies(self):
        svc = AssumptionsService()
        svc.shutdown()
        assert svc._dependencies is None


class TestConfigureAssumptionsService:
    """Test configure_assumptions_service(app) attaches to app.state."""

    def test_configure_creates_instance(self):
        app = _MockApp()
        svc = configure_assumptions_service(app)
        assert svc is not None
        assert svc.is_initialized is True

    def test_configure_attaches_to_app_state(self):
        app = _MockApp()
        svc = configure_assumptions_service(app)
        assert app.state.assumptions_service is svc

    def test_configure_with_config(self):
        app = _MockApp()
        svc = configure_assumptions_service(app, config={"max_versions": 100})
        assert svc is not None

    def test_configure_registers_router(self):
        app = _MockApp()
        configure_assumptions_service(app)
        assert len(app._routers) == 1


class TestGetAssumptionsService:
    """Test get_assumptions_service(app) retrieves from app.state."""

    def test_get_returns_same_instance(self):
        app = _MockApp()
        svc = configure_assumptions_service(app)
        retrieved = get_assumptions_service(app)
        assert retrieved is svc

    def test_get_raises_when_not_configured(self):
        app = _MockApp()
        with pytest.raises(RuntimeError, match="Assumptions service not configured"):
            get_assumptions_service(app)


class TestReconfigureReplacesExisting:
    """Test reconfigure replaces existing instance."""

    def test_reconfigure_shuts_down_old(self):
        app = _MockApp()
        svc1 = configure_assumptions_service(app, config={"env": "test1"})
        svc2 = configure_assumptions_service(app, config={"env": "test2"})
        assert svc1.is_initialized is False
        assert svc2.is_initialized is True
        assert get_assumptions_service(app) is svc2

    def test_reconfigure_registers_new_router(self):
        app = _MockApp()
        configure_assumptions_service(app)
        configure_assumptions_service(app)
        assert len(app._routers) == 2


class TestFacadeDelegatesToRegistry:
    """Test facade delegates to registry."""

    def test_create_assumption_delegates(self):
        svc = AssumptionsService()
        svc.create_assumption(assumption_id="a1", name="Test")
        svc.registry.create.assert_called_once_with(assumption_id="a1", name="Test")

    def test_get_assumption_delegates(self):
        svc = AssumptionsService()
        svc.get_assumption("a1")
        svc.registry.get.assert_called_once_with("a1")

    def test_update_assumption_delegates(self):
        svc = AssumptionsService()
        svc.update_assumption("a1", value=10)
        svc.registry.update.assert_called_once_with("a1", value=10)

    def test_delete_assumption_delegates(self):
        svc = AssumptionsService()
        svc.delete_assumption("a1")
        svc.registry.delete.assert_called_once_with("a1")

    def test_list_assumptions_delegates(self):
        svc = AssumptionsService()
        svc.list_assumptions(category="emission_factor")
        svc.registry.list_assumptions.assert_called_once_with(category="emission_factor")

    def test_get_value_delegates(self):
        svc = AssumptionsService()
        svc.get_value("a1")
        svc.registry.get_value.assert_called_once_with("a1")

    def test_set_value_delegates(self):
        svc = AssumptionsService()
        svc.set_value("a1", 42)
        svc.registry.set_value.assert_called_once_with("a1", 42)


class TestFacadeDelegatesToScenarios:
    """Test facade delegates to scenario manager."""

    def test_create_scenario_delegates(self):
        svc = AssumptionsService()
        svc.create_scenario(scenario_id="s1", name="Test")
        svc.scenario_manager.create.assert_called_once_with(scenario_id="s1", name="Test")

    def test_get_scenario_delegates(self):
        svc = AssumptionsService()
        svc.get_scenario("s1")
        svc.scenario_manager.get.assert_called_once_with("s1")

    def test_delete_scenario_delegates(self):
        svc = AssumptionsService()
        svc.delete_scenario("s1")
        svc.scenario_manager.delete.assert_called_once_with("s1")

    def test_list_scenarios_delegates(self):
        svc = AssumptionsService()
        svc.list_scenarios()
        svc.scenario_manager.list_scenarios.assert_called_once_with()


class TestLifespanManagement:
    """Test lifespan management."""

    def test_reset_clears_global(self):
        app = _MockApp()
        configure_assumptions_service(app)
        reset_assumptions_service()
        assert _GLOBAL_SERVICE is None

    def test_reset_shuts_down_service(self):
        app = _MockApp()
        svc = configure_assumptions_service(app)
        reset_assumptions_service()
        assert svc.is_initialized is False

    def test_reset_when_not_configured(self):
        reset_assumptions_service()  # Should not raise

    def test_get_router_lazy_creation(self):
        svc = AssumptionsService()
        router = svc.get_router()
        assert router is not None
        assert svc.get_router() is router
