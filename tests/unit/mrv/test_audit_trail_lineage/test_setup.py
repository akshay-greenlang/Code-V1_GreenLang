# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage.setup - AGENT-MRV-030.

Tests the service facade for the Audit Trail & Lineage Agent
(GL-MRV-X-042) including initialization, engine access, delegation
methods, health check, and router retrieval.

Coverage:
- Service singleton creation
- All delegation methods (record_event, verify_chain, trace_lineage, etc.)
- Health check endpoint
- get_router returns FastAPI router
- Engine initialization and access

Target: ~30 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.setup import (
        AuditTrailLineageService,
        get_service,
        get_router,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

# Fallback: try importing from __init__ if setup module doesn't exist yet
if not SETUP_AVAILABLE:
    try:
        from greenlang.agents.mrv.audit_trail_lineage import (
            AGENT_ID,
            AGENT_COMPONENT,
            VERSION,
            get_version,
            get_agent_info,
        )
        INIT_AVAILABLE = True
    except ImportError:
        INIT_AVAILABLE = False
else:
    INIT_AVAILABLE = True

_SKIP_SETUP = pytest.mark.skipif(
    not SETUP_AVAILABLE,
    reason="AuditTrailLineageService (setup.py) not available",
)

_SKIP_INIT = pytest.mark.skipif(
    not INIT_AVAILABLE,
    reason="audit_trail_lineage __init__ not available",
)

ORG_ID = "org-test-setup"
YEAR = 2025


# ==============================================================================
# SERVICE SINGLETON TESTS
# ==============================================================================


@_SKIP_SETUP
class TestServiceSingleton:
    """Test service singleton pattern."""

    def test_get_service_returns_instance(self):
        """Test get_service returns an AuditTrailLineageService instance."""
        service = get_service()
        assert isinstance(service, AuditTrailLineageService)

    def test_get_service_returns_same_instance(self):
        """Test get_service returns the same singleton."""
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2

    def test_service_direct_construction(self):
        """Test AuditTrailLineageService can be constructed directly."""
        service = AuditTrailLineageService()
        assert service is not None


# ==============================================================================
# DELEGATION METHOD TESTS
# ==============================================================================


@_SKIP_SETUP
class TestDelegationMethods:
    """Test service delegation methods."""

    def test_record_event_delegation(self):
        """Test service.record_event delegates to AuditEventEngine."""
        service = AuditTrailLineageService()
        result = service.record_event(
            event_type="DATA_INGESTED",
            agent_id="GL-MRV-S1-001",
            scope="scope_1",
            category=None,
            organization_id=ORG_ID,
            reporting_year=YEAR,
            payload={"test": True},
        )
        assert result["success"] is True

    def test_verify_chain_delegation(self):
        """Test service.verify_chain delegates to AuditEventEngine."""
        service = AuditTrailLineageService()
        service.record_event(
            event_type="DATA_INGESTED",
            agent_id="GL-MRV-S1-001",
            scope="scope_1",
            category=None,
            organization_id=ORG_ID,
            reporting_year=YEAR,
        )
        result = service.verify_chain(ORG_ID, YEAR)
        assert result["valid"] is True

    def test_get_events_delegation(self):
        """Test service.get_events delegates correctly."""
        service = AuditTrailLineageService()
        result = service.get_events(ORG_ID, YEAR)
        assert result["success"] is True

    def test_get_event_by_id_delegation(self):
        """Test service.get_event delegates correctly."""
        service = AuditTrailLineageService()
        r = service.record_event(
            event_type="DATA_INGESTED",
            agent_id="GL-MRV-S1-001",
            scope="scope_1",
            category=None,
            organization_id=ORG_ID,
            reporting_year=YEAR,
        )
        evt = service.get_event(r["event_id"])
        assert evt is not None

    def test_export_chain_delegation(self):
        """Test service.export_chain delegates correctly."""
        service = AuditTrailLineageService()
        result = service.export_chain(ORG_ID, YEAR)
        assert result["success"] is True

    def test_get_statistics_delegation(self):
        """Test service.get_event_statistics delegates correctly."""
        service = AuditTrailLineageService()
        result = service.get_event_statistics(ORG_ID, YEAR)
        assert result["success"] is True


# ==============================================================================
# HEALTH CHECK TESTS
# ==============================================================================


@_SKIP_SETUP
class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check(self):
        """Test health check returns healthy status."""
        service = AuditTrailLineageService()
        result = service.health_check()
        assert result["status"] in ["healthy", "ok", "UP"]

    def test_health_check_has_agent_id(self):
        """Test health check includes agent_id."""
        service = AuditTrailLineageService()
        result = service.health_check()
        assert result.get("agent_id") == "GL-MRV-X-042" or "GL-MRV-X-042" in str(result)

    def test_health_check_has_version(self):
        """Test health check includes version."""
        service = AuditTrailLineageService()
        result = service.health_check()
        assert "version" in result or "1.0.0" in str(result)

    def test_health_check_has_engines(self):
        """Test health check reports engine status."""
        service = AuditTrailLineageService()
        result = service.health_check()
        assert "engines" in result or "components" in result


# ==============================================================================
# ROUTER RETRIEVAL TESTS
# ==============================================================================


@_SKIP_SETUP
class TestGetRouter:
    """Test get_router returns a FastAPI router."""

    def test_get_router_returns_object(self):
        """Test get_router returns a non-None object."""
        router = get_router()
        assert router is not None

    def test_get_router_has_routes(self):
        """Test router has defined routes."""
        router = get_router()
        assert hasattr(router, "routes")
        assert len(router.routes) > 0


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


@_SKIP_SETUP
class TestEngineInitialization:
    """Test engine initialization via service."""

    def test_audit_event_engine_accessible(self):
        """Test AuditEventEngine is accessible via service."""
        service = AuditTrailLineageService()
        assert hasattr(service, "audit_event_engine") or hasattr(service, "_audit_event_engine")

    def test_service_has_all_engines(self):
        """Test service initializes all 7 engines."""
        service = AuditTrailLineageService()
        engine_attrs = [
            "audit_event_engine", "lineage_graph_engine",
            "evidence_packager_engine", "compliance_tracer_engine",
            "change_detector_engine", "compliance_checker_engine",
            "pipeline_engine",
        ]
        for attr in engine_attrs:
            # Allow either public or private attribute names
            has_attr = hasattr(service, attr) or hasattr(service, f"_{attr}")
            assert has_attr, f"Service missing engine: {attr}"


# ==============================================================================
# INIT MODULE INTEGRATION TESTS (fallback if setup.py not yet built)
# ==============================================================================


@_SKIP_INIT
class TestInitModuleIntegration:
    """Test __init__.py module-level access as setup fallback."""

    def test_agent_id_accessible(self):
        """Test AGENT_ID is accessible from __init__."""
        from greenlang.agents.mrv.audit_trail_lineage import AGENT_ID
        assert AGENT_ID == "GL-MRV-X-042"

    def test_version_accessible(self):
        """Test VERSION is accessible from __init__."""
        from greenlang.agents.mrv.audit_trail_lineage import VERSION
        assert VERSION == "1.0.0"

    def test_get_version_callable(self):
        """Test get_version is callable from __init__."""
        from greenlang.agents.mrv.audit_trail_lineage import get_version
        assert get_version() == "1.0.0"

    def test_get_agent_info_callable(self):
        """Test get_agent_info is callable from __init__."""
        from greenlang.agents.mrv.audit_trail_lineage import get_agent_info
        info = get_agent_info()
        assert info["agent_id"] == "GL-MRV-X-042"

    def test_all_exports_defined(self):
        """Test __all__ lists expected exports."""
        from greenlang.agents.mrv.audit_trail_lineage import __all__
        assert "AGENT_ID" in __all__
        assert "AGENT_COMPONENT" in __all__
        assert "VERSION" in __all__
        assert "AuditEventEngine" in __all__
        assert "get_version" in __all__
        assert "get_agent_info" in __all__
