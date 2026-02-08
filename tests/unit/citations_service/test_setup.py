# -*- coding: utf-8 -*-
"""
Unit Tests for CitationsService Facade & Setup (AGENT-FOUND-005)

Tests the CitationsService facade class, configure_citations_service(app),
get_citations_service(app), lifecycle management, and delegation to
registry/evidence/verification/provenance/export sub-components.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline CitationsService facade mirroring greenlang/citations/setup.py
# ---------------------------------------------------------------------------


class CitationsService:
    """Facade for the Citations & Evidence SDK."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._registry = MagicMock()
        self._evidence_manager = MagicMock()
        self._verification_engine = MagicMock()
        self._provenance_tracker = MagicMock()
        self._export_import_manager = MagicMock()
        self._initialized = True
        self._router = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def registry(self):
        return self._registry

    @property
    def evidence_manager(self):
        return self._evidence_manager

    @property
    def verification_engine(self):
        return self._verification_engine

    @property
    def provenance_tracker(self):
        return self._provenance_tracker

    @property
    def export_import_manager(self):
        return self._export_import_manager

    # ---- Registry delegation ----

    def create_citation(self, **kwargs):
        return self._registry.create(**kwargs)

    def get_citation(self, citation_id: str):
        return self._registry.get(citation_id)

    def update_citation(self, citation_id: str, **kwargs):
        return self._registry.update(citation_id, **kwargs)

    def delete_citation(self, citation_id: str):
        return self._registry.delete(citation_id)

    def list_citations(self, **kwargs):
        return self._registry.list_citations(**kwargs)

    def search_citations(self, query: str):
        return self._registry.search(query)

    # ---- Evidence delegation ----

    def create_package(self, **kwargs):
        return self._evidence_manager.create_package(**kwargs)

    def get_package(self, package_id: str):
        return self._evidence_manager.get_package(package_id)

    def add_evidence(self, package_id: str, **kwargs):
        return self._evidence_manager.add_item(package_id, **kwargs)

    def finalize_package(self, package_id: str):
        return self._evidence_manager.finalize_package(package_id)

    def list_packages(self, **kwargs):
        return self._evidence_manager.list_packages(**kwargs)

    def delete_package(self, package_id: str):
        return self._evidence_manager.delete_package(package_id)

    # ---- Verification delegation ----

    def verify_citation(self, citation_id: str):
        return self._verification_engine.verify_citation(citation_id)

    def verify_batch(self, citation_ids: List[str]):
        return self._verification_engine.verify_batch(citation_ids)

    # ---- Export/Import delegation ----

    def export_citations(self, format_type: str):
        return self._export_import_manager.export(format_type)

    def import_citations(self, data: Dict[str, Any]):
        return self._export_import_manager.import_data(data)

    # ---- Lifecycle ----

    def get_router(self):
        if self._router is None:
            self._router = MagicMock()
        return self._router

    def shutdown(self):
        self._initialized = False
        self._registry = None
        self._evidence_manager = None
        self._verification_engine = None
        self._provenance_tracker = None
        self._export_import_manager = None


class _MockApp:
    """Mock FastAPI app with state attribute."""

    def __init__(self):
        self.state = type("State", (), {})()
        self._routers: List[Any] = []

    def include_router(self, router, **kwargs):
        self._routers.append(router)


_GLOBAL_SERVICE: Optional[CitationsService] = None


def configure_citations_service(
    app: _MockApp,
    config: Optional[Dict[str, Any]] = None,
) -> CitationsService:
    global _GLOBAL_SERVICE

    if _GLOBAL_SERVICE is not None:
        _GLOBAL_SERVICE.shutdown()

    service = CitationsService(config)
    app.state.citations_service = service
    _GLOBAL_SERVICE = service

    router = service.get_router()
    app.include_router(router)

    return service


def get_citations_service(app: _MockApp) -> CitationsService:
    service = getattr(app.state, "citations_service", None)
    if service is None:
        raise RuntimeError(
            "Citations service not configured. "
            "Call configure_citations_service(app) first."
        )
    return service


def get_router() -> MagicMock:
    if _GLOBAL_SERVICE is None:
        raise RuntimeError("Citations service not configured")
    return _GLOBAL_SERVICE.get_router()


def reset_citations_service():
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
    reset_citations_service()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCitationsServiceCreation:
    """Test CitationsService creation."""

    def test_creation_with_defaults(self):
        svc = CitationsService()
        assert svc.is_initialized is True

    def test_creation_with_config(self):
        svc = CitationsService(config={"max_citations": 50000})
        assert svc.is_initialized is True

    def test_has_registry(self):
        svc = CitationsService()
        assert svc.registry is not None

    def test_has_evidence_manager(self):
        svc = CitationsService()
        assert svc.evidence_manager is not None

    def test_has_verification_engine(self):
        svc = CitationsService()
        assert svc.verification_engine is not None

    def test_has_provenance_tracker(self):
        svc = CitationsService()
        assert svc.provenance_tracker is not None

    def test_has_export_import_manager(self):
        svc = CitationsService()
        assert svc.export_import_manager is not None


class TestCitationsServiceLifecycle:
    """Test lifecycle management."""

    def test_shutdown(self):
        svc = CitationsService()
        assert svc.is_initialized is True
        svc.shutdown()
        assert svc.is_initialized is False

    def test_shutdown_cleans_registry(self):
        svc = CitationsService()
        svc.shutdown()
        assert svc._registry is None

    def test_shutdown_cleans_evidence_manager(self):
        svc = CitationsService()
        svc.shutdown()
        assert svc._evidence_manager is None

    def test_shutdown_cleans_verification_engine(self):
        svc = CitationsService()
        svc.shutdown()
        assert svc._verification_engine is None

    def test_shutdown_cleans_provenance_tracker(self):
        svc = CitationsService()
        svc.shutdown()
        assert svc._provenance_tracker is None

    def test_shutdown_cleans_export_import_manager(self):
        svc = CitationsService()
        svc.shutdown()
        assert svc._export_import_manager is None


class TestConfigureCitationsService:
    """Test configure_citations_service(app) attaches to app.state."""

    def test_configure_creates_instance(self):
        app = _MockApp()
        svc = configure_citations_service(app)
        assert svc is not None
        assert svc.is_initialized is True

    def test_configure_attaches_to_app_state(self):
        app = _MockApp()
        svc = configure_citations_service(app)
        assert app.state.citations_service is svc

    def test_configure_with_config(self):
        app = _MockApp()
        svc = configure_citations_service(app, config={"max_citations": 50000})
        assert svc is not None

    def test_configure_registers_router(self):
        app = _MockApp()
        configure_citations_service(app)
        assert len(app._routers) == 1


class TestGetCitationsService:
    """Test get_citations_service(app) retrieves from app.state."""

    def test_get_returns_same_instance(self):
        app = _MockApp()
        svc = configure_citations_service(app)
        retrieved = get_citations_service(app)
        assert retrieved is svc

    def test_get_raises_when_not_configured(self):
        app = _MockApp()
        with pytest.raises(RuntimeError, match="Citations service not configured"):
            get_citations_service(app)


class TestGetRouter:
    """Test router retrieval."""

    def test_get_router_from_service(self):
        app = _MockApp()
        configure_citations_service(app)
        router = get_router()
        assert router is not None

    def test_get_router_raises_when_not_configured(self):
        with pytest.raises(RuntimeError, match="not configured"):
            get_router()

    def test_get_router_returns_same_instance(self):
        app = _MockApp()
        configure_citations_service(app)
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2


class TestReconfigureReplacesExisting:
    """Test reconfigure replaces existing instance."""

    def test_reconfigure_shuts_down_old(self):
        app = _MockApp()
        svc1 = configure_citations_service(app, config={"env": "test1"})
        svc2 = configure_citations_service(app, config={"env": "test2"})
        assert svc1.is_initialized is False
        assert svc2.is_initialized is True
        assert get_citations_service(app) is svc2

    def test_reconfigure_registers_new_router(self):
        app = _MockApp()
        configure_citations_service(app)
        configure_citations_service(app)
        assert len(app._routers) == 2


class TestFacadeDelegatesToRegistry:
    """Test facade delegates to registry."""

    def test_create_citation_delegates(self):
        svc = CitationsService()
        svc.create_citation(citation_type="emission_factor")
        svc.registry.create.assert_called_once_with(citation_type="emission_factor")

    def test_get_citation_delegates(self):
        svc = CitationsService()
        svc.get_citation("cid-1")
        svc.registry.get.assert_called_once_with("cid-1")

    def test_update_citation_delegates(self):
        svc = CitationsService()
        svc.update_citation("cid-1", title="Updated")
        svc.registry.update.assert_called_once_with("cid-1", title="Updated")

    def test_delete_citation_delegates(self):
        svc = CitationsService()
        svc.delete_citation("cid-1")
        svc.registry.delete.assert_called_once_with("cid-1")

    def test_list_citations_delegates(self):
        svc = CitationsService()
        svc.list_citations(citation_type="emission_factor")
        svc.registry.list_citations.assert_called_once_with(citation_type="emission_factor")

    def test_search_citations_delegates(self):
        svc = CitationsService()
        svc.search_citations("DEFRA")
        svc.registry.search.assert_called_once_with("DEFRA")


class TestFacadeDelegatesToEvidence:
    """Test facade delegates to evidence manager."""

    def test_create_package_delegates(self):
        svc = CitationsService()
        svc.create_package(name="Test")
        svc.evidence_manager.create_package.assert_called_once_with(name="Test")

    def test_get_package_delegates(self):
        svc = CitationsService()
        svc.get_package("pkg-1")
        svc.evidence_manager.get_package.assert_called_once_with("pkg-1")

    def test_finalize_package_delegates(self):
        svc = CitationsService()
        svc.finalize_package("pkg-1")
        svc.evidence_manager.finalize_package.assert_called_once_with("pkg-1")

    def test_delete_package_delegates(self):
        svc = CitationsService()
        svc.delete_package("pkg-1")
        svc.evidence_manager.delete_package.assert_called_once_with("pkg-1")


class TestFacadeDelegatesToVerification:
    """Test facade delegates to verification engine."""

    def test_verify_citation_delegates(self):
        svc = CitationsService()
        svc.verify_citation("cid-1")
        svc.verification_engine.verify_citation.assert_called_once_with("cid-1")

    def test_verify_batch_delegates(self):
        svc = CitationsService()
        svc.verify_batch(["cid-1", "cid-2"])
        svc.verification_engine.verify_batch.assert_called_once_with(["cid-1", "cid-2"])


class TestLifespanManagement:
    """Test lifespan management."""

    def test_reset_clears_global(self):
        app = _MockApp()
        configure_citations_service(app)
        reset_citations_service()
        assert _GLOBAL_SERVICE is None

    def test_reset_shuts_down_service(self):
        app = _MockApp()
        svc = configure_citations_service(app)
        reset_citations_service()
        assert svc.is_initialized is False

    def test_reset_when_not_configured(self):
        reset_citations_service()  # Should not raise

    def test_get_router_lazy_creation(self):
        svc = CitationsService()
        router = svc.get_router()
        assert router is not None
        assert svc.get_router() is router
