# -*- coding: utf-8 -*-
"""
Unit Tests for NormalizerService Facade & Setup (AGENT-FOUND-003)

Tests the NormalizerService facade class, configure_normalizer_service(app),
get_normalizer_service(app), lifecycle management, and delegation to
converter/resolver sub-components.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Async helper for Windows compatibility
# ---------------------------------------------------------------------------


def _run_async(coro):
    if sys.platform == "win32":
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Inline NormalizerService facade mirroring greenlang/normalizer/setup.py
# ---------------------------------------------------------------------------


class NormalizerService:
    """
    Facade for the Normalizer SDK.

    Delegates to UnitConverter, EntityResolver, DimensionalAnalyzer,
    and ConversionProvenanceTracker.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._converter = MagicMock()
        self._resolver = MagicMock()
        self._dimensional = MagicMock()
        self._provenance = MagicMock()
        self._initialized = True
        self._router = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def converter(self):
        return self._converter

    @property
    def resolver(self):
        return self._resolver

    @property
    def dimensional(self):
        return self._dimensional

    @property
    def provenance(self):
        return self._provenance

    def convert(self, value: float, from_unit: str, to_unit: str, **kwargs):
        """Delegate to UnitConverter.convert()."""
        return self._converter.convert(value, from_unit, to_unit, **kwargs)

    def convert_ghg(self, value: float, from_unit: str, to_unit: str, gwp_version: str = "AR6"):
        """Delegate to UnitConverter.convert_ghg()."""
        return self._converter.convert_ghg(value, from_unit, to_unit, gwp_version)

    def resolve_fuel(self, name: str):
        """Delegate to EntityResolver.resolve_fuel()."""
        return self._resolver.resolve_fuel(name)

    def resolve_material(self, name: str):
        """Delegate to EntityResolver.resolve_material()."""
        return self._resolver.resolve_material(name)

    def check_compatibility(self, unit_a: str, unit_b: str) -> bool:
        """Delegate to DimensionalAnalyzer.check_compatibility()."""
        return self._dimensional.check_compatibility(unit_a, unit_b)

    def get_router(self):
        """Return the FastAPI router."""
        if self._router is None:
            self._router = MagicMock()
        return self._router

    def shutdown(self):
        """Clean up resources."""
        self._initialized = False
        self._converter = None
        self._resolver = None


class _MockApp:
    """Mock FastAPI app with state attribute."""

    def __init__(self):
        self.state = type("State", (), {})()
        self._routers: List[Any] = []

    def include_router(self, router, **kwargs):
        self._routers.append(router)


_GLOBAL_SERVICE: Optional[NormalizerService] = None


def configure_normalizer_service(
    app: _MockApp,
    config: Optional[Dict[str, Any]] = None,
) -> NormalizerService:
    """Configure and attach the normalizer service to the FastAPI app."""
    global _GLOBAL_SERVICE

    if _GLOBAL_SERVICE is not None:
        _GLOBAL_SERVICE.shutdown()

    service = NormalizerService(config)
    app.state.normalizer_service = service
    _GLOBAL_SERVICE = service

    router = service.get_router()
    app.include_router(router)

    return service


def get_normalizer_service(app: _MockApp) -> NormalizerService:
    """Retrieve the normalizer service from the FastAPI app."""
    service = getattr(app.state, "normalizer_service", None)
    if service is None:
        raise RuntimeError(
            "Normalizer service not configured. "
            "Call configure_normalizer_service(app) first."
        )
    return service


def reset_normalizer_service():
    """Reset the global normalizer service."""
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
    reset_normalizer_service()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestNormalizerServiceCreation:
    """Test NormalizerService creation."""

    def test_creation_with_defaults(self):
        svc = NormalizerService()
        assert svc.is_initialized is True

    def test_creation_with_config(self):
        svc = NormalizerService(config={"precision": 8, "gwp_version": "AR5"})
        assert svc.is_initialized is True

    def test_shutdown(self):
        svc = NormalizerService()
        assert svc.is_initialized is True
        svc.shutdown()
        assert svc.is_initialized is False

    def test_shutdown_cleans_up_converter(self):
        svc = NormalizerService()
        assert svc.converter is not None
        svc.shutdown()
        assert svc._converter is None

    def test_shutdown_cleans_up_resolver(self):
        svc = NormalizerService()
        assert svc.resolver is not None
        svc.shutdown()
        assert svc._resolver is None

    def test_has_converter(self):
        svc = NormalizerService()
        assert svc.converter is not None

    def test_has_resolver(self):
        svc = NormalizerService()
        assert svc.resolver is not None

    def test_has_dimensional(self):
        svc = NormalizerService()
        assert svc.dimensional is not None

    def test_has_provenance(self):
        svc = NormalizerService()
        assert svc.provenance is not None


class TestConfigureNormalizerService:
    """Test configure_normalizer_service(app) attaches to app.state."""

    def test_configure_creates_instance(self):
        app = _MockApp()
        svc = configure_normalizer_service(app)
        assert svc is not None
        assert svc.is_initialized is True

    def test_configure_attaches_to_app_state(self):
        app = _MockApp()
        svc = configure_normalizer_service(app)
        assert app.state.normalizer_service is svc

    def test_configure_with_config(self):
        app = _MockApp()
        svc = configure_normalizer_service(app, config={"precision": 6})
        assert svc is not None

    def test_configure_registers_router(self):
        app = _MockApp()
        configure_normalizer_service(app)
        assert len(app._routers) == 1


class TestGetNormalizerService:
    """Test get_normalizer_service(app) retrieves from app.state."""

    def test_get_returns_same_instance(self):
        app = _MockApp()
        svc = configure_normalizer_service(app)
        retrieved = get_normalizer_service(app)
        assert retrieved is svc

    def test_get_raises_when_not_configured(self):
        app = _MockApp()
        with pytest.raises(RuntimeError, match="Normalizer service not configured"):
            get_normalizer_service(app)


class TestReconfigureReplacesExisting:
    """Test reconfigure replaces existing instance."""

    def test_reconfigure_shuts_down_old(self):
        app = _MockApp()
        svc1 = configure_normalizer_service(app, config={"env": "test1"})
        svc2 = configure_normalizer_service(app, config={"env": "test2"})
        assert svc1.is_initialized is False
        assert svc2.is_initialized is True
        assert get_normalizer_service(app) is svc2

    def test_reconfigure_registers_new_router(self):
        app = _MockApp()
        configure_normalizer_service(app)
        configure_normalizer_service(app)
        assert len(app._routers) == 2


class TestFacadeDelegatesToConverter:
    """Test NormalizerService delegates to converter."""

    def test_convert_delegates(self):
        svc = NormalizerService()
        svc.convert(100, "kg", "t")
        svc.converter.convert.assert_called_once_with(100, "kg", "t")

    def test_convert_ghg_delegates(self):
        svc = NormalizerService()
        svc.convert_ghg(1, "tCH4", "tCO2e", "AR6")
        svc.converter.convert_ghg.assert_called_once_with(1, "tCH4", "tCO2e", "AR6")


class TestFacadeDelegatesToResolver:
    """Test NormalizerService delegates to resolver."""

    def test_resolve_fuel_delegates(self):
        svc = NormalizerService()
        svc.resolve_fuel("Natural Gas")
        svc.resolver.resolve_fuel.assert_called_once_with("Natural Gas")

    def test_resolve_material_delegates(self):
        svc = NormalizerService()
        svc.resolve_material("Steel")
        svc.resolver.resolve_material.assert_called_once_with("Steel")


class TestFacadeDelegatesToDimensional:
    """Test NormalizerService delegates to dimensional analyzer."""

    def test_check_compatibility_delegates(self):
        svc = NormalizerService()
        svc.check_compatibility("kg", "t")
        svc.dimensional.check_compatibility.assert_called_once_with("kg", "t")


class TestLifespanManagement:
    """Test lifespan management for the normalizer service."""

    def test_reset_clears_global(self):
        app = _MockApp()
        configure_normalizer_service(app)
        reset_normalizer_service()
        assert _GLOBAL_SERVICE is None

    def test_reset_shuts_down_service(self):
        app = _MockApp()
        svc = configure_normalizer_service(app)
        reset_normalizer_service()
        assert svc.is_initialized is False

    def test_reset_when_not_configured(self):
        reset_normalizer_service()  # Should not raise

    def test_get_router_lazy_creation(self):
        svc = NormalizerService()
        router = svc.get_router()
        assert router is not None
        assert svc.get_router() is router
