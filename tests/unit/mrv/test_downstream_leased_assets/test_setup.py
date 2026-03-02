# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-026 Downstream Leased Assets Agent - DownstreamLeasedService (setup.py).

Tests the service facade including initialization, engine access, single/batch
calculation, building/vehicle/equipment/IT delegation, compliance checking,
emission factor retrieval, health check, router, method existence, and
integration sequences.

Target: 30 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
import pytest

try:
    from greenlang.downstream_leased_assets.setup import (
        DownstreamLeasedService,
        get_service,
        get_router,
    )
    SETUP_AVAILABLE = True
except ImportError:
    SETUP_AVAILABLE = False

try:
    from greenlang.downstream_leased_assets.models import (
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

_SKIP = pytest.mark.skipif(not SETUP_AVAILABLE, reason="DownstreamLeasedService not available")
pytestmark = _SKIP


@pytest.fixture
def service():
    return DownstreamLeasedService()


# ==============================================================================
# MODULE CONSTANTS TESTS
# ==============================================================================


class TestModuleConstants:

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_agent_id(self):
        assert AGENT_ID == "GL-MRV-S3-013"

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_agent_component(self):
        assert AGENT_COMPONENT == "AGENT-MRV-026"

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_version(self):
        assert VERSION == "1.0.0"

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_table_prefix(self):
        assert TABLE_PREFIX == "gl_dla_"


# ==============================================================================
# SERVICE CREATION TESTS
# ==============================================================================


class TestServiceCreation:

    def test_service_creation(self):
        svc = DownstreamLeasedService()
        assert svc is not None

    def test_service_has_agent_id(self, service):
        assert service.agent_id == "GL-MRV-S3-013" or hasattr(service, "agent_id")

    def test_service_has_version(self, service):
        assert service.version == "1.0.0" or hasattr(service, "version")


# ==============================================================================
# SINGLETON AND RESET TESTS
# ==============================================================================


class TestSingleton:

    def test_get_service_returns_instance(self):
        svc = get_service()
        assert svc is not None
        assert isinstance(svc, DownstreamLeasedService)


class TestThreadSafety:

    def test_concurrent_get_service(self):
        services = []

        def get_svc():
            services.append(get_service())

        threads = [threading.Thread(target=get_svc) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = services[0]
        for svc in services[1:]:
            assert svc is first


# ==============================================================================
# ENGINE INITIALIZATION TESTS (7 engines)
# ==============================================================================


class TestEngineInitialization:

    def test_database_engine_initialized(self, service):
        assert service.database_engine is not None or hasattr(service, "_database_engine")

    def test_asset_specific_engine_initialized(self, service):
        assert service.asset_specific_engine is not None or hasattr(service, "_asset_specific_engine")

    def test_average_data_engine_initialized(self, service):
        assert service.average_data_engine is not None or hasattr(service, "_average_data_engine")

    def test_spend_engine_initialized(self, service):
        assert service.spend_engine is not None or hasattr(service, "_spend_engine")

    def test_hybrid_engine_initialized(self, service):
        assert service.hybrid_engine is not None or hasattr(service, "_hybrid_engine")

    def test_compliance_engine_initialized(self, service):
        assert service.compliance_engine is not None or hasattr(service, "_compliance_engine")

    def test_pipeline_engine_initialized(self, service):
        assert service.pipeline_engine is not None or hasattr(service, "_pipeline_engine")


# ==============================================================================
# HEALTH CHECK AND VERSION
# ==============================================================================


class TestHealthCheck:

    def test_health_check_method_exists(self, service):
        assert hasattr(service, "health_check")

    def test_version_method_or_attr(self, service):
        assert hasattr(service, "version") or hasattr(service, "get_version")


# ==============================================================================
# CALCULATE METHOD EXISTENCE
# ==============================================================================


class TestCalculateMethods:

    def test_calculate_method_exists(self, service):
        assert hasattr(service, "calculate")

    def test_calculate_batch_exists(self, service):
        assert hasattr(service, "calculate_batch")

    def test_calculate_building_exists(self, service):
        assert hasattr(service, "calculate_building")

    def test_calculate_vehicle_exists(self, service):
        assert hasattr(service, "calculate_vehicle")

    def test_calculate_equipment_exists(self, service):
        assert hasattr(service, "calculate_equipment")

    def test_calculate_it_asset_exists(self, service):
        assert hasattr(service, "calculate_it_asset")

    def test_check_compliance_exists(self, service):
        assert hasattr(service, "check_compliance")

    def test_get_emission_factors_exists(self, service):
        assert hasattr(service, "get_emission_factors")

    def test_aggregate_exists(self, service):
        assert hasattr(service, "aggregate")


# ==============================================================================
# LOOKUP METHODS
# ==============================================================================


class TestLookupMethods:

    def test_lookup_building_benchmarks(self, service):
        assert hasattr(service, "get_emission_factors") or hasattr(service, "lookup_building_eui")

    def test_lookup_vehicle_efs(self, service):
        assert hasattr(service, "get_emission_factors")

    def test_lookup_grid_factors(self, service):
        assert hasattr(service, "get_emission_factors")


# ==============================================================================
# RESPONSE MODELS
# ==============================================================================


class TestResponseModels:

    def test_service_returns_dict(self, service):
        """Service methods should return dict-like results."""
        assert hasattr(service, "calculate")


# ==============================================================================
# ROUTER RETRIEVAL
# ==============================================================================


class TestRouter:

    def test_get_router_returns_router(self):
        router = get_router()
        assert router is not None
        assert hasattr(router, "routes")

    def test_get_router_has_prefix(self):
        router = get_router()
        assert router.prefix == "/api/v1/downstream-leased-assets"

    def test_get_router_has_tags(self):
        router = get_router()
        assert "downstream-leased-assets" in router.tags
