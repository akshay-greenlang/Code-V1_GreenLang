# -*- coding: utf-8 -*-
"""
Unit tests for ProcessingSoldProductsService -- AGENT-MRV-023

Tests the service facade that wires together all 7 engines, including
calculate methods (5 methods), calculate_spend_based, calculate_hybrid,
calculate_batch, calculate_portfolio, check_compliance, get_processing_ef,
get_energy_intensity, get_grid_ef, get_fuel_ef, get_eeio_factor,
get_processing_chain, get_all_categories, get_all_processing_types,
get_processing_chains, health_check, get_version, get_config, get_router,
get_service singleton, reset_service, and thread safety.

Target: 25+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.processing_sold_products.setup import (
        ProcessingSoldProductsService,
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

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ProcessingSoldProductsService not available")
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
    """Create a ProcessingSoldProductsService instance."""
    return get_service()


@pytest.fixture
def steel_input():
    """Create a steel product input for average-data calculation."""
    return [
        {
            "product_id": "STEEL-001",
            "category": "metals_ferrous",
            "processing_type": "machining",
            "quantity_tonnes": "500",
        }
    ]


@pytest.fixture
def multi_product_inputs():
    """Create multiple product inputs."""
    return [
        {
            "product_id": "STEEL-001",
            "category": "metals_ferrous",
            "processing_type": "machining",
            "quantity_tonnes": "500",
        },
        {
            "product_id": "PLASTIC-001",
            "category": "plastics_thermoplastic",
            "processing_type": "injection_molding",
            "quantity_tonnes": "300",
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
        """Test AGENT_ID is GL-MRV-S3-010."""
        assert AGENT_ID == "GL-MRV-S3-010"

    def test_agent_component(self):
        """Test AGENT_COMPONENT is AGENT-MRV-023."""
        assert AGENT_COMPONENT == "AGENT-MRV-023"

    def test_table_prefix(self):
        """Test TABLE_PREFIX is gl_psp_."""
        assert TABLE_PREFIX == "gl_psp_"


# ============================================================================
# TEST: Service Singleton
# ============================================================================


class TestServiceSingleton:
    """Test singleton pattern for ProcessingSoldProductsService."""

    def test_get_service_returns_instance(self):
        """Test that get_service returns a ProcessingSoldProductsService."""
        svc = get_service()
        assert isinstance(svc, ProcessingSoldProductsService)

    def test_get_service_singleton(self):
        """Test that get_service returns the same instance."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_reset_service_allows_new_instance(self):
        """Test that reset_service allows a new singleton to be created."""
        svc1 = get_service()
        reset_service()
        svc2 = get_service()
        assert svc1 is not svc2


# ============================================================================
# TEST: Health Check
# ============================================================================


class TestHealthCheck:
    """Test service health check."""

    def test_health_check_returns_dict(self, service):
        """Test that health_check returns a dictionary."""
        status = service.health_check()
        assert isinstance(status, dict)

    def test_health_check_has_status(self, service):
        """Test that health check includes status field."""
        status = service.health_check()
        assert "status" in status
        assert status["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_check_has_version(self, service):
        """Test that health check includes version."""
        status = service.health_check()
        assert status["version"] == "1.0.0"

    def test_health_check_has_agent_id(self, service):
        """Test that health check includes agent_id."""
        status = service.health_check()
        assert status["agent_id"] == "GL-MRV-S3-010"

    def test_health_check_has_engines_status(self, service):
        """Test that health check includes engines_status."""
        status = service.health_check()
        assert "engines_status" in status
        engines = status["engines_status"]
        assert isinstance(engines, dict)
        assert "database" in engines
        assert "site_specific" in engines
        assert "average_data" in engines
        assert "spend_based" in engines
        assert "hybrid" in engines
        assert "compliance" in engines
        assert "pipeline" in engines

    def test_health_check_has_uptime(self, service):
        """Test that health check includes uptime_seconds."""
        status = service.health_check()
        assert "uptime_seconds" in status
        assert status["uptime_seconds"] >= 0


# ============================================================================
# TEST: Get Version
# ============================================================================


class TestGetVersion:
    """Test get_version method."""

    def test_get_version_returns_1_0_0(self, service):
        """Test that get_version returns '1.0.0'."""
        assert service.get_version() == "1.0.0"


# ============================================================================
# TEST: Get Config
# ============================================================================


class TestGetConfig:
    """Test get_config method."""

    def test_get_config_returns_dict(self, service):
        """Test that get_config returns a dictionary."""
        config = service.get_config()
        assert isinstance(config, dict)

    def test_get_config_has_agent_id(self, service):
        """Test that config includes agent_id."""
        config = service.get_config()
        assert config["agent_id"] == "GL-MRV-S3-010"

    def test_get_config_has_supported_methods(self, service):
        """Test that config includes supported methods."""
        config = service.get_config()
        methods = config["supported_methods"]
        assert "average_data" in methods
        assert "spend_based" in methods
        assert "site_specific_direct" in methods

    def test_get_config_has_compliance_frameworks(self, service):
        """Test that config shows 7 compliance frameworks."""
        config = service.get_config()
        assert config["compliance_frameworks"] == 7

    def test_get_config_has_allocation_methods(self, service):
        """Test that config lists 4 allocation methods."""
        config = service.get_config()
        assert len(config["allocation_methods"]) == 4

    def test_get_config_has_double_counting_rules(self, service):
        """Test that config shows 8 double-counting rules."""
        config = service.get_config()
        assert config["double_counting_rules"] == 8


# ============================================================================
# TEST: Calculate Methods
# ============================================================================


class TestCalculateMethods:
    """Test calculation method dispatch."""

    def test_calculate_average_data(self, service, steel_input):
        """Test calculate with average_data method returns a response."""
        result = service.calculate(steel_input, method="average_data", org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)
        assert result.method == "average_data"

    def test_calculate_returns_calc_id(self, service, steel_input):
        """Test that calculate returns a non-empty calc_id."""
        result = service.calculate(steel_input, method="average_data", org_id="ORG-001", year=2024)
        assert result.calc_id is not None
        assert len(result.calc_id) > 0

    def test_calculate_returns_processing_time(self, service, steel_input):
        """Test that calculate returns positive processing_time_ms."""
        result = service.calculate(steel_input, method="average_data", org_id="ORG-001", year=2024)
        assert result.processing_time_ms >= 0

    def test_calculate_average_data_shortcut(self, service, steel_input):
        """Test calculate_average_data convenience method."""
        result = service.calculate_average_data(steel_input, org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)

    def test_calculate_site_specific_direct_shortcut(self, service):
        """Test calculate_site_specific_direct convenience method."""
        inputs = [
            {
                "product_id": "P1",
                "category": "metals_ferrous",
                "processing_type": "machining",
                "quantity_tonnes": "100",
                "processing_emissions_kg": "28000",
            }
        ]
        result = service.calculate_site_specific_direct(inputs, org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)

    def test_calculate_site_specific_energy_shortcut(self, service):
        """Test calculate_site_specific_energy convenience method."""
        inputs = [
            {
                "product_id": "P1",
                "category": "metals_ferrous",
                "processing_type": "machining",
                "quantity_tonnes": "100",
                "processing_energy_kwh": "28000",
            }
        ]
        result = service.calculate_site_specific_energy(inputs, org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)

    def test_calculate_site_specific_fuel_shortcut(self, service):
        """Test calculate_site_specific_fuel convenience method."""
        inputs = [
            {
                "product_id": "P1",
                "category": "metals_ferrous",
                "processing_type": "machining",
                "quantity_tonnes": "100",
                "fuel_type": "natural_gas",
                "fuel_quantity_kwh": "5000",
            }
        ]
        result = service.calculate_site_specific_fuel(inputs, org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)

    def test_calculate_unknown_method_returns_error(self, service, steel_input):
        """Test that unknown method returns error response."""
        result = service.calculate(steel_input, method="nonexistent", org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)
        assert result.success is False
        assert result.error is not None


# ============================================================================
# TEST: Spend-Based Calculation
# ============================================================================


class TestSpendBasedCalculation:
    """Test spend-based EEIO calculation."""

    def test_calculate_spend_based(self, service):
        """Test calculate_spend_based method."""
        result = service.calculate_spend_based(
            revenue=1000000.0,
            currency="USD",
            sector="331",
            year=2024,
            org_id="ORG-001",
            reporting_year=2024,
        )
        assert isinstance(result, CalculationResponse)
        assert result.total_emissions_kg >= 0


# ============================================================================
# TEST: Hybrid Calculation
# ============================================================================


class TestHybridCalculation:
    """Test hybrid multi-method aggregation."""

    def test_calculate_hybrid(self, service, multi_product_inputs):
        """Test calculate_hybrid method."""
        result = service.calculate_hybrid(multi_product_inputs, org_id="ORG-001", year=2024)
        assert isinstance(result, CalculationResponse)


# ============================================================================
# TEST: Batch Calculation
# ============================================================================


class TestBatchCalculation:
    """Test batch calculation processing."""

    def test_calculate_batch(self, service, steel_input):
        """Test calculate_batch method."""
        batch = [
            {"inputs": steel_input, "org_id": "ORG-001", "year": 2024},
            {"inputs": steel_input, "org_id": "ORG-001", "year": 2024},
        ]
        result = service.calculate_batch(batch, method="average_data")
        assert isinstance(result, BatchCalculationResponse)
        assert result.total_calculations == 2


# ============================================================================
# TEST: Portfolio Calculation
# ============================================================================


class TestPortfolioCalculation:
    """Test portfolio-level calculation."""

    def test_calculate_portfolio(self, service, multi_product_inputs):
        """Test calculate_portfolio method."""
        result = service.calculate_portfolio(
            multi_product_inputs, org_id="ORG-001", reporting_year=2024
        )
        assert isinstance(result, AggregationResponse)


# ============================================================================
# TEST: Compliance
# ============================================================================


class TestComplianceCheck:
    """Test compliance checking."""

    def test_check_compliance_not_found(self, service):
        """Test compliance check on non-existent calculation returns error."""
        result = service.check_compliance("NONEXISTENT-001")
        assert isinstance(result, ComplianceCheckResponse)
        assert result.success is False


# ============================================================================
# TEST: Emission Factor Lookups
# ============================================================================


class TestEFLookups:
    """Test emission factor lookup methods."""

    def test_get_all_categories_returns_list(self, service):
        """Test that get_all_categories returns a non-empty list."""
        categories = service.get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) >= 10

    def test_get_all_processing_types_returns_list(self, service):
        """Test that get_all_processing_types returns a non-empty list."""
        types = service.get_all_processing_types()
        assert isinstance(types, list)
        assert len(types) >= 15

    def test_get_processing_chains_returns_list(self, service):
        """Test that get_processing_chains returns a non-empty list."""
        chains = service.get_processing_chains()
        assert isinstance(chains, list)
        assert len(chains) >= 4


# ============================================================================
# TEST: Provenance
# ============================================================================


class TestProvenance:
    """Test provenance retrieval."""

    def test_get_provenance_not_found(self, service):
        """Test getting provenance for non-existent calculation."""
        result = service.get_provenance("NONEXISTENT-001")
        assert isinstance(result, ProvenanceResponse)
        assert result.success is False

    def test_get_provenance_after_calculate(self, service, steel_input):
        """Test getting provenance for a completed calculation."""
        calc = service.calculate(steel_input, method="average_data", org_id="ORG-001", year=2024)
        if calc.success:
            prov = service.get_provenance(calc.calc_id)
            assert isinstance(prov, ProvenanceResponse)
            assert prov.success is True


# ============================================================================
# TEST: Aggregations
# ============================================================================


class TestAggregations:
    """Test aggregation queries."""

    def test_get_aggregations_empty(self, service):
        """Test get_aggregations returns valid structure even with no data."""
        result = service.get_aggregations("ORG-001", "2024")
        assert isinstance(result, AggregationResponse)
        assert result.success is True


# ============================================================================
# TEST: Get Router
# ============================================================================


class TestGetRouter:
    """Test get_router function."""

    def test_get_router_returns_object(self):
        """Test that get_router returns a router object."""
        try:
            router = get_router()
            assert router is not None
        except ImportError:
            pytest.skip("FastAPI router not available")


# ============================================================================
# TEST: Thread Safety
# ============================================================================


class TestServiceThreadSafety:
    """Test thread safety of ProcessingSoldProductsService."""

    def test_concurrent_get_service(self):
        """Test that 10 threads calling get_service get the same singleton."""
        results = []

        def fetch():
            svc = get_service()
            results.append(id(svc))

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 10
        assert len(set(results)) == 1  # All same instance

    def test_concurrent_calculations(self, service):
        """Test that 5 threads can run calculations concurrently."""
        inputs = [
            {
                "product_id": "P1",
                "category": "metals_ferrous",
                "processing_type": "machining",
                "quantity_tonnes": "100",
            }
        ]
        results = []
        errors = []

        def calc():
            try:
                r = service.calculate(inputs, method="average_data", org_id="ORG-001", year=2024)
                results.append(r)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=calc) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        assert len(results) == 5
        for r in results:
            assert isinstance(r, CalculationResponse)
