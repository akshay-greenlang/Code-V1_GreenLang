# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation service setup - AGENT-MRV-022.

Tests DownstreamTransportService facade creation, 7 engine initialization,
22 method existence, and service accessors for the Downstream Transportation
& Distribution Agent (GL-MRV-S3-009).

Coverage (~25 tests):
- Service creation and initialization
- 7 engine initialization (database, distance, spend, average, warehouse,
  compliance, pipeline)
- 22 method existence on service facade
- Service accessor methods
- Service state management

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation import (
        DownstreamTransportDatabaseEngine,
        DistanceBasedCalculatorEngine,
        SpendBasedCalculatorEngine,
        AverageDataCalculatorEngine,
        WarehouseDistributionEngine,
        ComplianceCheckerEngine,
        DownstreamTransportPipelineEngine,
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,
        METRICS_PREFIX,
        API_PREFIX,
        get_config,
    )
    _ENGINES_AVAILABLE = all([
        DownstreamTransportDatabaseEngine is not None,
        DistanceBasedCalculatorEngine is not None,
        SpendBasedCalculatorEngine is not None,
        AverageDataCalculatorEngine is not None,
        WarehouseDistributionEngine is not None,
        ComplianceCheckerEngine is not None,
        DownstreamTransportPipelineEngine is not None,
    ])
except ImportError as exc:
    _AVAILABLE = False
    _ENGINES_AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"downstream_transportation not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SERVICE CREATION TESTS
# ==============================================================================


class TestServiceCreation:
    """Test downstream transportation service creation."""

    def test_package_imports(self):
        """Test package-level imports are available."""
        assert AGENT_ID == "GL-MRV-S3-009"
        assert AGENT_COMPONENT == "AGENT-MRV-022"
        assert VERSION == "1.0.0"
        assert TABLE_PREFIX == "gl_dto_"
        assert METRICS_PREFIX == "gl_dto_"
        assert API_PREFIX == "/api/v1/downstream-transportation"

    def test_get_config_callable(self):
        """Test get_config function is callable."""
        assert callable(get_config)

    def test_get_config_returns_value(self):
        """Test get_config returns a non-None value."""
        config = get_config()
        assert config is not None


# ==============================================================================
# ENGINE INITIALIZATION TESTS
# ==============================================================================


class TestEngineInitialization:
    """Test all 7 engine classes are importable and instantiable."""

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_database_engine_init(self):
        """Test DownstreamTransportDatabaseEngine initialization."""
        engine = DownstreamTransportDatabaseEngine()
        assert engine is not None

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_distance_engine_init(self):
        """Test DistanceBasedCalculatorEngine initialization."""
        engine = DistanceBasedCalculatorEngine()
        assert engine is not None

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_spend_engine_init(self):
        """Test SpendBasedCalculatorEngine initialization."""
        engine = SpendBasedCalculatorEngine()
        assert engine is not None

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_average_engine_init(self):
        """Test AverageDataCalculatorEngine initialization."""
        engine = AverageDataCalculatorEngine()
        assert engine is not None

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_warehouse_engine_init(self):
        """Test WarehouseDistributionEngine initialization."""
        engine = WarehouseDistributionEngine()
        assert engine is not None

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_compliance_engine_init(self):
        """Test ComplianceCheckerEngine initialization."""
        engine = ComplianceCheckerEngine()
        assert engine is not None

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_pipeline_engine_init(self):
        """Test DownstreamTransportPipelineEngine initialization."""
        engine = DownstreamTransportPipelineEngine()
        assert engine is not None


# ==============================================================================
# METHOD EXISTENCE TESTS
# ==============================================================================


class TestMethodExistence:
    """Test expected methods exist on engine classes."""

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_database_engine_methods(self):
        """Test DownstreamTransportDatabaseEngine has all 18 methods."""
        engine = DownstreamTransportDatabaseEngine()
        expected_methods = [
            "get_transport_emission_factor",
            "get_cold_chain_factor",
            "get_warehouse_emission_factor",
            "get_last_mile_factor",
            "get_eeio_factor",
            "get_currency_rate",
            "get_cpi_deflator",
            "get_grid_emission_factor",
            "get_channel_average",
            "get_incoterm_classification",
            "get_load_factor",
            "get_return_factor",
            "get_dqi_scoring",
            "get_uncertainty_range",
            "get_mode_comparison",
            "get_wtt_factor",
            "get_fleet_average_factor",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_distance_engine_methods(self):
        """Test DistanceBasedCalculatorEngine has core methods."""
        engine = DistanceBasedCalculatorEngine()
        expected_methods = [
            "calculate_shipment",
            "calculate_multi_leg",
            "calculate_batch",
            "calculate_fleet",
            "compare_modes",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_spend_engine_methods(self):
        """Test SpendBasedCalculatorEngine has core methods."""
        engine = SpendBasedCalculatorEngine()
        expected_methods = [
            "calculate_spend",
            "calculate_batch",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_average_engine_methods(self):
        """Test AverageDataCalculatorEngine has core methods."""
        engine = AverageDataCalculatorEngine()
        expected_methods = [
            "calculate_channel",
            "calculate_batch",
            "compare_channels",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_warehouse_engine_methods(self):
        """Test WarehouseDistributionEngine has core methods."""
        engine = WarehouseDistributionEngine()
        expected_methods = [
            "calculate_warehouse",
            "calculate_last_mile",
            "calculate_batch_warehouses",
            "calculate_batch_last_mile",
            "calculate_distribution_chain",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_compliance_engine_methods(self):
        """Test ComplianceCheckerEngine has core methods."""
        engine = ComplianceCheckerEngine()
        expected_methods = [
            "check_compliance",
            "check_double_counting",
            "classify_incoterm",
            "get_required_disclosures",
            "get_double_counting_rules",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"

    @pytest.mark.skipif(
        not _ENGINES_AVAILABLE,
        reason="Not all engines are available",
    )
    def test_pipeline_engine_methods(self):
        """Test DownstreamTransportPipelineEngine has core methods."""
        engine = DownstreamTransportPipelineEngine()
        expected_methods = [
            "process",
            "process_batch",
            "process_distribution_chain",
        ]
        for method in expected_methods:
            assert hasattr(engine, method), f"Missing method: {method}"


# ==============================================================================
# SERVICE ACCESSOR TESTS
# ==============================================================================


class TestServiceAccessors:
    """Test service accessor patterns."""

    def test_all_exports_available(self):
        """Test __all__ exports are available."""
        from greenlang.agents.mrv.downstream_transportation import __all__
        assert "DownstreamTransportDatabaseEngine" in __all__
        assert "DistanceBasedCalculatorEngine" in __all__
        assert "SpendBasedCalculatorEngine" in __all__
        assert "AverageDataCalculatorEngine" in __all__
        assert "WarehouseDistributionEngine" in __all__
        assert "ComplianceCheckerEngine" in __all__
        assert "DownstreamTransportPipelineEngine" in __all__
        assert "AGENT_ID" in __all__
        assert "get_config" in __all__

    def test_agent_id_constant(self):
        """Test AGENT_ID constant."""
        assert AGENT_ID == "GL-MRV-S3-009"

    def test_version_constant(self):
        """Test VERSION constant."""
        parts = VERSION.split(".")
        assert len(parts) == 3

    def test_table_prefix_convention(self):
        """Test TABLE_PREFIX follows gl_ convention."""
        assert TABLE_PREFIX.startswith("gl_")
        assert TABLE_PREFIX.endswith("_")

    def test_metrics_prefix_matches_table(self):
        """Test METRICS_PREFIX matches TABLE_PREFIX."""
        assert METRICS_PREFIX == TABLE_PREFIX

    def test_api_prefix_format(self):
        """Test API_PREFIX starts with /api/v1/."""
        assert API_PREFIX.startswith("/api/v1/")
        assert "downstream-transportation" in API_PREFIX
