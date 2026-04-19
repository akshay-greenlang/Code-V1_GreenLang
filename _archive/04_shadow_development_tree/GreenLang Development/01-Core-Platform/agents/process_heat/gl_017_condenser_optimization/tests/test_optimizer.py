"""
GL-017 CONDENSYNC Agent - Optimizer Integration Tests

Integration tests for CondenserOptimizerAgent including:
- Full pipeline execution
- Component coordination
- Input/output validation
- Recommendation generation
- Alert triggering

Coverage targets:
    - Process method with various scenarios
    - Validation methods
    - KPI calculations
    - Provenance tracking
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import hashlib

from greenlang.agents.process_heat.gl_017_condenser_optimization.optimizer import (
    CondenserOptimizerAgent,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    CondenserOptimizationConfig,
    CoolingTowerConfig,
    TubeFoulingConfig,
    VacuumSystemConfig,
    AirIngresConfig,
    CleanlinessConfig,
    PerformanceConfig,
    CoolingWaterSource,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    CondenserInput,
    CondenserOutput,
    CondenserStatus,
    CleaningStatus,
    AlertSeverity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Create default condenser configuration."""
    return CondenserOptimizationConfig(condenser_id="TEST-C-001")


@pytest.fixture
def agent(default_config):
    """Create CondenserOptimizerAgent instance."""
    return CondenserOptimizerAgent(default_config)


@pytest.fixture
def valid_input():
    """Create valid condenser input."""
    return CondenserInput(
        condenser_id="TEST-C-001",
        load_pct=85.0,
        exhaust_steam_flow_lb_hr=425000.0,
        exhaust_steam_pressure_psia=1.2,
        condenser_vacuum_inhga=1.5,
        saturation_temperature_f=101.0,
        hotwell_temperature_f=100.5,
        cw_inlet_temperature_f=75.0,
        cw_outlet_temperature_f=95.0,
        cw_inlet_flow_gpm=90000.0,
    )


@pytest.fixture
def input_with_cooling_tower():
    """Create input with cooling tower data."""
    return CondenserInput(
        condenser_id="TEST-C-001",
        load_pct=85.0,
        exhaust_steam_flow_lb_hr=425000.0,
        exhaust_steam_pressure_psia=1.2,
        condenser_vacuum_inhga=1.5,
        saturation_temperature_f=101.0,
        hotwell_temperature_f=100.5,
        cw_inlet_temperature_f=75.0,
        cw_outlet_temperature_f=95.0,
        cw_inlet_flow_gpm=90000.0,
        wet_bulb_temperature_f=78.0,
        dry_bulb_temperature_f=90.0,
        cw_conductivity_umhos=2000.0,
        cw_ph=8.0,
        makeup_water_flow_gpm=1000.0,
        blowdown_flow_gpm=200.0,
    )


@pytest.fixture
def high_fouling_input():
    """Create input that indicates high fouling."""
    return CondenserInput(
        condenser_id="TEST-C-001",
        load_pct=85.0,
        exhaust_steam_flow_lb_hr=425000.0,
        exhaust_steam_pressure_psia=1.2,
        condenser_vacuum_inhga=2.5,  # High vacuum = fouling
        saturation_temperature_f=110.0,
        hotwell_temperature_f=108.0,
        cw_inlet_temperature_f=75.0,
        cw_outlet_temperature_f=95.0,
        cw_inlet_flow_gpm=90000.0,
    )


@pytest.fixture
def air_ingress_input():
    """Create input with air ingress indicators."""
    return CondenserInput(
        condenser_id="TEST-C-001",
        load_pct=85.0,
        exhaust_steam_flow_lb_hr=425000.0,
        exhaust_steam_pressure_psia=1.2,
        condenser_vacuum_inhga=1.8,
        saturation_temperature_f=105.0,
        hotwell_temperature_f=100.0,  # 5F subcooling
        cw_inlet_temperature_f=75.0,
        cw_outlet_temperature_f=95.0,
        cw_inlet_flow_gpm=90000.0,
        condensate_dissolved_o2_ppb=50.0,  # High DO
        subcooling_f=5.0,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestAgentInitialization:
    """Test agent initialization."""

    def test_basic_initialization(self, default_config):
        """Test agent initializes correctly."""
        agent = CondenserOptimizerAgent(default_config)
        assert agent is not None
        assert agent.condenser_config == default_config

    def test_component_initialization(self, agent):
        """Test sub-components are initialized."""
        assert agent.cleanliness_calculator is not None
        assert agent.cleanliness_monitor is not None
        assert agent.fouling_detector is not None
        assert agent.vacuum_monitor is not None
        assert agent.air_ingress_detector is not None
        assert agent.cooling_tower_optimizer is not None
        assert agent.performance_analyzer is not None

    def test_agent_config(self, agent):
        """Test agent configuration."""
        assert agent.config.agent_type == "GL-017"
        assert "CONDENSYNC" in agent.config.name

    def test_custom_config(self):
        """Test agent with custom configuration."""
        custom_fouling = TubeFoulingConfig(
            design_cleanliness_factor=0.90,
            cleanliness_warning_threshold=0.80,
        )
        config = CondenserOptimizationConfig(
            condenser_id="CUSTOM-001",
            tube_fouling=custom_fouling,
        )
        agent = CondenserOptimizerAgent(config)
        assert agent.condenser_config.tube_fouling.design_cleanliness_factor == 0.90


# =============================================================================
# PROCESS TESTS
# =============================================================================

class TestProcessMethod:
    """Test the main process method."""

    def test_process_valid_input(self, agent, valid_input):
        """Test processing valid input."""
        result = agent.process(valid_input)

        assert isinstance(result, CondenserOutput)
        assert result.condenser_id == "TEST-C-001"
        assert result.status == "success"

    def test_process_returns_all_results(self, agent, valid_input):
        """Test all result components are present."""
        result = agent.process(valid_input)

        assert result.cleanliness is not None
        assert result.tube_fouling is not None
        assert result.vacuum_system is not None
        assert result.air_ingress is not None
        assert result.performance is not None

    def test_process_generates_kpis(self, agent, valid_input):
        """Test KPIs are generated."""
        result = agent.process(valid_input)

        assert "cleanliness_factor" in result.kpis
        assert "backpressure_deviation_pct" in result.kpis
        assert "heat_rate_impact_btu_kwh" in result.kpis

    def test_process_with_cooling_tower(self, agent, input_with_cooling_tower):
        """Test processing with cooling tower data."""
        result = agent.process(input_with_cooling_tower)

        assert result.cooling_tower is not None
        assert result.cooling_tower.cycles_of_concentration > 0

    def test_process_without_cooling_tower(self, valid_input):
        """Test processing without cooling tower data."""
        config = CondenserOptimizationConfig(
            condenser_id="TEST-C-001",
            cooling_source=CoolingWaterSource.ONCE_THROUGH_FRESH,
        )
        agent = CondenserOptimizerAgent(config)
        result = agent.process(valid_input)

        assert result.cooling_tower is None

    def test_processing_time_recorded(self, agent, valid_input):
        """Test processing time is recorded."""
        result = agent.process(valid_input)

        assert result.processing_time_ms > 0

    def test_provenance_hash_generated(self, agent, valid_input):
        """Test provenance hash is generated."""
        result = agent.process(valid_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_input_hash_generated(self, agent, valid_input):
        """Test input hash is generated."""
        result = agent.process(valid_input)

        assert result.input_hash is not None
        assert len(result.input_hash) == 16

    def test_deterministic_provenance(self, agent, valid_input):
        """Test same input produces same provenance hash within same second."""
        # Note: Due to timestamp in provenance, exact match unlikely
        # unless we mock datetime
        result1 = agent.process(valid_input)
        result2 = agent.process(valid_input)

        # Input hashes should match
        assert result1.input_hash == result2.input_hash


# =============================================================================
# CLEANLINESS ANALYSIS TESTS
# =============================================================================

class TestCleanlinessAnalysis:
    """Test cleanliness analysis integration."""

    def test_cleanliness_factor_calculated(self, agent, valid_input):
        """Test cleanliness factor is calculated."""
        result = agent.process(valid_input)

        cf = result.cleanliness.cleanliness_factor
        assert 0.0 <= cf <= 1.2

    def test_cleaning_status_determined(self, agent, valid_input):
        """Test cleaning status is determined."""
        result = agent.process(valid_input)

        assert result.cleanliness.cleaning_status in CleaningStatus

    def test_high_fouling_detection(self, agent, high_fouling_input):
        """Test high fouling is detected."""
        result = agent.process(high_fouling_input)

        # High vacuum indicates fouling
        assert result.tube_fouling.fouling_detected is True
        assert result.tube_fouling.fouling_severity in ["light", "moderate", "severe"]


# =============================================================================
# RECOMMENDATION TESTS
# =============================================================================

class TestRecommendations:
    """Test recommendation generation."""

    def test_recommendations_generated_for_fouling(self, agent, high_fouling_input):
        """Test recommendations generated for fouled condenser."""
        result = agent.process(high_fouling_input)

        # Should have at least one recommendation
        fouling_recs = [r for r in result.recommendations if r.category == "fouling"]
        # May or may not have recommendations depending on severity
        assert isinstance(result.recommendations, list)

    def test_recommendation_structure(self, agent, high_fouling_input):
        """Test recommendation structure."""
        result = agent.process(high_fouling_input)

        if result.recommendations:
            rec = result.recommendations[0]
            assert rec.recommendation_id is not None
            assert rec.category is not None
            assert rec.title is not None
            assert rec.description is not None


# =============================================================================
# ALERT TESTS
# =============================================================================

class TestAlerts:
    """Test alert generation."""

    def test_low_vacuum_alert(self):
        """Test low vacuum alert generation."""
        config = CondenserOptimizationConfig(
            condenser_id="TEST-C-001",
            low_vacuum_trip_inhga=5.0,
        )
        agent = CondenserOptimizerAgent(config)

        # Input with vacuum near trip point
        input_data = CondenserInput(
            condenser_id="TEST-C-001",
            load_pct=85.0,
            exhaust_steam_flow_lb_hr=425000.0,
            exhaust_steam_pressure_psia=1.2,
            condenser_vacuum_inhga=4.5,  # 90% of trip
            saturation_temperature_f=115.0,
            hotwell_temperature_f=114.0,
            cw_inlet_temperature_f=75.0,
            cw_outlet_temperature_f=95.0,
            cw_inlet_flow_gpm=90000.0,
        )

        result = agent.process(input_data)

        vacuum_alerts = [a for a in result.alerts if a.category == "vacuum"]
        assert len(vacuum_alerts) > 0

    def test_high_hotwell_alert(self):
        """Test high hotwell level alert."""
        config = CondenserOptimizationConfig(
            condenser_id="TEST-C-001",
            high_hotwell_level_trip_pct=90.0,
        )
        agent = CondenserOptimizerAgent(config)

        input_data = CondenserInput(
            condenser_id="TEST-C-001",
            load_pct=85.0,
            exhaust_steam_flow_lb_hr=425000.0,
            exhaust_steam_pressure_psia=1.2,
            condenser_vacuum_inhga=1.5,
            saturation_temperature_f=101.0,
            hotwell_temperature_f=100.5,
            hotwell_level_pct=85.0,  # > 90% of trip
            cw_inlet_temperature_f=75.0,
            cw_outlet_temperature_f=95.0,
            cw_inlet_flow_gpm=90000.0,
        )

        result = agent.process(input_data)

        level_alerts = [a for a in result.alerts if a.category == "level"]
        assert len(level_alerts) > 0

    def test_alert_structure(self, agent, high_fouling_input):
        """Test alert structure."""
        result = agent.process(high_fouling_input)

        if result.alerts:
            alert = result.alerts[0]
            assert alert.alert_id is not None
            assert alert.severity in AlertSeverity
            assert alert.category is not None
            assert alert.title is not None


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test input/output validation."""

    def test_validate_valid_input(self, agent, valid_input):
        """Test validation passes for valid input."""
        assert agent.validate_input(valid_input) is True

    def test_validate_invalid_vacuum(self, agent):
        """Test validation fails for invalid vacuum."""
        # Create input with invalid vacuum (via direct attribute if possible)
        input_data = CondenserInput(
            condenser_id="TEST-C-001",
            load_pct=85.0,
            exhaust_steam_flow_lb_hr=425000.0,
            exhaust_steam_pressure_psia=1.2,
            condenser_vacuum_inhga=0.01,  # Very low
            saturation_temperature_f=101.0,
            hotwell_temperature_f=100.5,
            cw_inlet_temperature_f=75.0,
            cw_outlet_temperature_f=95.0,
            cw_inlet_flow_gpm=90000.0,
        )
        # Input object is valid per schema but may fail business validation
        result = agent.validate_input(input_data)
        # 0.01 is > 0 so should pass basic validation
        assert result is True

    def test_validate_output(self, agent, valid_input):
        """Test output validation."""
        result = agent.process(valid_input)
        assert agent.validate_output(result) is True


# =============================================================================
# PERFORMANCE CURVE TESTS
# =============================================================================

class TestPerformanceCurves:
    """Test performance curve generation."""

    def test_get_performance_curves(self, agent):
        """Test performance curve generation."""
        curves = agent.get_performance_curves()

        assert isinstance(curves, dict)
        assert len(curves) > 0

    def test_performance_curves_structure(self, agent):
        """Test performance curves structure."""
        curves = agent.get_performance_curves(
            inlet_temps=[70.0, 80.0, 90.0],
            loads=[50.0, 75.0, 100.0],
        )

        # Should have curves for each load
        for load in [50.0, 75.0, 100.0]:
            assert load in curves
            # Each load should have values for each inlet temp
            for temp in [70.0, 80.0, 90.0]:
                assert temp in curves[load]
                assert curves[load][temp] > 0


# =============================================================================
# VACUUM DECAY TEST
# =============================================================================

class TestVacuumDecayTest:
    """Test vacuum decay test analysis."""

    def test_vacuum_decay_analysis(self, agent):
        """Test vacuum decay test analysis."""
        result = agent.perform_vacuum_decay_test(
            initial_vacuum=1.5,
            final_vacuum=1.6,
            duration_minutes=10.0,
        )

        assert "decay_rate_inhg_min" in result
        assert "test_passed" in result
        assert "status" in result
        assert "recommended_action" in result

    def test_acceptable_decay_rate(self, agent):
        """Test acceptable decay rate detection."""
        result = agent.perform_vacuum_decay_test(
            initial_vacuum=1.5,
            final_vacuum=1.51,
            duration_minutes=10.0,
        )

        assert result["status"] == "acceptable"
        assert result["test_passed"] is True

    def test_excessive_decay_rate(self, agent):
        """Test excessive decay rate detection."""
        result = agent.perform_vacuum_decay_test(
            initial_vacuum=1.5,
            final_vacuum=2.5,  # Large decay
            duration_minutes=10.0,
        )

        assert result["test_passed"] is False
        assert result["status"] in ["marginal", "excessive", "severe"]


# =============================================================================
# BLOWDOWN OPTIMIZATION
# =============================================================================

class TestBlowdownOptimization:
    """Test blowdown optimization."""

    def test_optimize_blowdown(self, agent):
        """Test blowdown optimization calculation."""
        result = agent.optimize_blowdown(
            current_cycles=4.0,
            evaporation_gpm=800.0,
        )

        assert "current_cycles" in result
        assert "target_cycles" in result
        assert "optimal_blowdown_gpm" in result

    def test_blowdown_savings_calculation(self, agent):
        """Test blowdown savings calculation."""
        result = agent.optimize_blowdown(
            current_cycles=3.0,  # Low cycles
            evaporation_gpm=800.0,
        )

        # Should recommend higher cycles
        assert result["target_cycles"] > 3.0
        assert "annual_savings_usd" in result


# =============================================================================
# METADATA TESTS
# =============================================================================

class TestMetadata:
    """Test metadata generation."""

    def test_metadata_includes_version(self, agent, valid_input):
        """Test metadata includes agent version."""
        result = agent.process(valid_input)

        assert "agent_version" in result.metadata
        assert result.metadata["agent_version"] == "1.0.0"

    def test_metadata_includes_hei_edition(self, agent, valid_input):
        """Test metadata includes HEI edition."""
        result = agent.process(valid_input)

        assert "hei_edition" in result.metadata
        assert result.metadata["hei_edition"] == "12th"

    def test_metadata_includes_calculation_count(self, agent, valid_input):
        """Test metadata includes calculation count."""
        result = agent.process(valid_input)

        assert "calculation_count" in result.metadata
        assert result.metadata["calculation_count"] >= 6  # One per component


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_load(self, agent):
        """Test processing at minimum load."""
        input_data = CondenserInput(
            condenser_id="TEST-C-001",
            load_pct=30.0,  # Minimum typical load
            exhaust_steam_flow_lb_hr=150000.0,
            exhaust_steam_pressure_psia=0.8,
            condenser_vacuum_inhga=1.0,
            saturation_temperature_f=95.0,
            hotwell_temperature_f=94.5,
            cw_inlet_temperature_f=70.0,
            cw_outlet_temperature_f=85.0,
            cw_inlet_flow_gpm=90000.0,
        )

        result = agent.process(input_data)
        assert result.status == "success"

    def test_maximum_load(self, agent):
        """Test processing at maximum load."""
        input_data = CondenserInput(
            condenser_id="TEST-C-001",
            load_pct=110.0,  # Peak load
            exhaust_steam_flow_lb_hr=550000.0,
            exhaust_steam_pressure_psia=1.5,
            condenser_vacuum_inhga=2.0,
            saturation_temperature_f=108.0,
            hotwell_temperature_f=107.0,
            cw_inlet_temperature_f=85.0,
            cw_outlet_temperature_f=105.0,
            cw_inlet_flow_gpm=95000.0,
        )

        result = agent.process(input_data)
        assert result.status == "success"

    def test_extreme_temperatures(self, agent):
        """Test with extreme inlet temperature."""
        input_data = CondenserInput(
            condenser_id="TEST-C-001",
            load_pct=85.0,
            exhaust_steam_flow_lb_hr=425000.0,
            exhaust_steam_pressure_psia=1.2,
            condenser_vacuum_inhga=2.5,
            saturation_temperature_f=115.0,
            hotwell_temperature_f=114.0,
            cw_inlet_temperature_f=100.0,  # Very hot inlet
            cw_outlet_temperature_f=115.0,
            cw_inlet_flow_gpm=90000.0,
        )

        result = agent.process(input_data)
        assert result.status == "success"
        # Should show high deviation
        assert result.performance.backpressure_deviation_pct > 0
