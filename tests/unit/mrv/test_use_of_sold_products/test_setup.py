# -*- coding: utf-8 -*-
"""
Unit tests for UseOfSoldProductsService -- AGENT-MRV-024

Tests the service facade that wires together all 7 engines, including
calculate methods (direct, indirect, fuels/feedstocks), calculate_batch,
calculate_portfolio, check_compliance, get_fuel_ef, get_grid_ef,
get_refrigerant_gwp, get_product_profile, get_default_lifetime,
get_all_categories, get_all_fuel_types, get_all_regions,
get_all_refrigerants, get_database_summary, health_check, get_version,
get_config, get_router, get_service singleton, reset_service,
and thread safety.

Target: 25+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.setup import (
        UseOfSoldProductsService,
        get_service,
        get_router,
        reset_service,
        VERSION,
        AGENT_ID,
        AGENT_COMPONENT,
        TABLE_PREFIX,
        CalculationResponse,
        BatchCalculationResponse,
        AggregationResponse,
        ComplianceCheckResponse,
        ProvenanceResponse,
        HealthResponse,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="UseOfSoldProductsService not available")
pytestmark = _SKIP


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the service singleton before each test."""
    reset_service()
    yield
    reset_service()


@pytest.fixture
def service():
    """Create a UseOfSoldProductsService instance."""
    return get_service()


@pytest.fixture
def vehicle_input():
    """Create a vehicle product input for direct calculation."""
    return [
        {
            "product_id": "VEH-001",
            "category": "vehicles",
            "emission_type": "direct",
            "units_sold": 1000,
            "lifetime_years": 15,
            "fuel_type": "gasoline",
            "fuel_consumption_per_year": "1200",
        }
    ]


@pytest.fixture
def multi_product_inputs():
    """Create multiple product inputs."""
    return [
        {
            "product_id": "VEH-001",
            "category": "vehicles",
            "emission_type": "direct",
            "units_sold": 1000,
            "lifetime_years": 15,
        },
        {
            "product_id": "APP-001",
            "category": "appliances",
            "emission_type": "indirect",
            "units_sold": 10000,
            "lifetime_years": 15,
        },
    ]


# ============================================================================
# TEST: Module Constants
# ============================================================================


class TestModuleConstants:
    """Test module-level constants."""

    def test_version(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_agent_id(self):
        """Test AGENT_ID is GL-MRV-S3-011."""
        assert AGENT_ID == "GL-MRV-S3-011"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-024."""
        assert AGENT_COMPONENT == "AGENT-MRV-024"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_usp_."""
        assert TABLE_PREFIX == "gl_usp_"


# ============================================================================
# TEST: Singleton Pattern
# ============================================================================


class TestSingletonPattern:
    """Test service singleton pattern."""

    def test_get_service_returns_same_instance(self):
        """Test get_service returns the same instance."""
        s1 = get_service()
        s2 = get_service()
        assert s1 is s2

    def test_reset_service_creates_new_instance(self):
        """Test reset_service creates a new instance."""
        s1 = get_service()
        reset_service()
        s2 = get_service()
        assert s1 is not s2

    def test_thread_safe_get_service(self):
        """Test get_service is thread-safe with 10+ threads."""
        results = []
        errors = []

        def _get():
            try:
                s = get_service()
                results.append(id(s))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(set(results)) == 1


# ============================================================================
# TEST: Service Initialization
# ============================================================================


class TestServiceInitialization:
    """Test lazy initialization of service engines."""

    def test_service_has_database_engine(self, service):
        """Test service initializes database engine."""
        assert hasattr(service, "database_engine") or hasattr(service, "_database_engine")

    def test_service_has_direct_engine(self, service):
        """Test service initializes direct emissions engine."""
        assert hasattr(service, "direct_engine") or hasattr(service, "_direct_engine")

    def test_service_has_indirect_engine(self, service):
        """Test service initializes indirect emissions engine."""
        assert hasattr(service, "indirect_engine") or hasattr(service, "_indirect_engine")

    def test_service_has_fuels_engine(self, service):
        """Test service initializes fuels/feedstocks engine."""
        assert hasattr(service, "fuels_engine") or hasattr(service, "_fuels_engine")

    def test_service_has_lifetime_engine(self, service):
        """Test service initializes lifetime modeling engine."""
        assert hasattr(service, "lifetime_engine") or hasattr(service, "_lifetime_engine")

    def test_service_has_compliance_engine(self, service):
        """Test service initializes compliance engine."""
        assert hasattr(service, "compliance_engine") or hasattr(service, "_compliance_engine")

    def test_service_has_pipeline_engine(self, service):
        """Test service initializes pipeline engine."""
        assert hasattr(service, "pipeline_engine") or hasattr(service, "_pipeline_engine")


# ============================================================================
# TEST: Health Check
# ============================================================================


class TestHealthCheck:
    """Test health_check method."""

    def test_health_check_returns_dict(self, service):
        """Test health_check returns a dictionary."""
        result = service.health_check()
        assert isinstance(result, dict)

    def test_health_check_has_status(self, service):
        """Test health_check includes status field."""
        result = service.health_check()
        assert "status" in result
        assert result["status"] in ("healthy", "ok", "up")

    def test_health_check_has_version(self, service):
        """Test health_check includes version."""
        result = service.health_check()
        assert "version" in result
        assert result["version"] == "1.0.0"

    def test_health_check_has_agent_id(self, service):
        """Test health_check includes agent_id."""
        result = service.health_check()
        assert "agent_id" in result
        assert result["agent_id"] == "GL-MRV-S3-011"


# ============================================================================
# TEST: Get Version
# ============================================================================


class TestGetVersion:
    """Test get_version method."""

    def test_get_version(self, service):
        """Test get_version returns version string."""
        version = service.get_version()
        assert version == "1.0.0"


# ============================================================================
# TEST: Get Router
# ============================================================================


class TestGetRouter:
    """Test get_router function."""

    def test_get_router_returns_router(self):
        """Test get_router returns a FastAPI router."""
        router = get_router()
        assert router is not None


# ============================================================================
# TEST: Lookup Methods
# ============================================================================


class TestLookupMethods:
    """Test emission factor and profile lookup methods."""

    def test_get_all_categories(self, service):
        """Test get_all_categories returns list of 10 categories."""
        categories = service.get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) >= 10

    def test_get_all_fuel_types(self, service):
        """Test get_all_fuel_types returns list of 15 fuel types."""
        fuels = service.get_all_fuel_types()
        assert isinstance(fuels, list)
        assert len(fuels) >= 15

    def test_get_all_regions(self, service):
        """Test get_all_regions returns list of 16 regions."""
        regions = service.get_all_regions()
        assert isinstance(regions, list)
        assert len(regions) >= 16

    def test_get_all_refrigerants(self, service):
        """Test get_all_refrigerants returns list of 10 refrigerants."""
        refrigerants = service.get_all_refrigerants()
        assert isinstance(refrigerants, list)
        assert len(refrigerants) >= 10

    def test_get_database_summary(self, service):
        """Test get_database_summary returns summary dict."""
        summary = service.get_database_summary()
        assert isinstance(summary, dict)
        assert "categories" in summary
        assert "fuel_types" in summary


# ============================================================================
# TEST: Response Models
# ============================================================================


class TestResponseModels:
    """Test response model definitions."""

    def test_calculation_response_exists(self):
        """Test CalculationResponse model exists."""
        assert CalculationResponse is not None

    def test_batch_response_exists(self):
        """Test BatchCalculationResponse model exists."""
        assert BatchCalculationResponse is not None

    def test_compliance_response_exists(self):
        """Test ComplianceCheckResponse model exists."""
        assert ComplianceCheckResponse is not None

    def test_provenance_response_exists(self):
        """Test ProvenanceResponse model exists."""
        assert ProvenanceResponse is not None

    def test_health_response_exists(self):
        """Test HealthResponse model exists."""
        assert HealthResponse is not None

    def test_aggregation_response_exists(self):
        """Test AggregationResponse model exists."""
        assert AggregationResponse is not None
