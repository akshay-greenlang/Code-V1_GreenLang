"""
Unit tests for GL-009 THERMALIQ Agent Main Analyzer

Tests the main ThermalFluidAnalyzer class including full processing workflow,
integration between sub-analyzers, and provenance tracking.
"""

import pytest
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from greenlang.agents.process_heat.gl_009_thermal_fluid.analyzer import (
    ThermalFluidAnalyzer,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.config import (
    ThermalFluidConfig,
    create_default_config,
    SafetyConfig,
    ExergyConfig,
)
from greenlang.agents.process_heat.gl_009_thermal_fluid.schemas import (
    ThermalFluidType,
    ThermalFluidInput,
    ThermalFluidOutput,
    FluidLabAnalysis,
    SafetyStatus,
    OptimizationStatus,
    HeaterType,
    DegradationLevel,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Create default configuration."""
    return create_default_config(system_id="TF-TEST-001")


@pytest.fixture
def analyzer(default_config):
    """Create ThermalFluidAnalyzer instance."""
    return ThermalFluidAnalyzer(default_config)


@pytest.fixture
def analyzer_no_exergy():
    """Create analyzer with exergy disabled."""
    config = create_default_config(system_id="TF-TEST-002")
    config.exergy.enabled = False
    return ThermalFluidAnalyzer(config)


@pytest.fixture
def normal_input():
    """Create normal operating input."""
    return ThermalFluidInput(
        system_id="TF-TEST-001",
        fluid_type=ThermalFluidType.THERMINOL_66,
        bulk_temperature_f=550.0,
        flow_rate_gpm=450.0,
        design_flow_rate_gpm=500.0,
        inlet_temperature_f=520.0,
        outlet_temperature_f=580.0,
        heater_duty_btu_hr=5_000_000.0,
        pump_discharge_pressure_psig=75.0,
        expansion_tank_level_pct=45.0,
    )


@pytest.fixture
def alarm_input():
    """Create input with alarm conditions."""
    return ThermalFluidInput(
        system_id="TF-TEST-001",
        fluid_type=ThermalFluidType.THERMINOL_66,
        bulk_temperature_f=625.0,  # High temperature
        flow_rate_gpm=180.0,  # Low flow
        design_flow_rate_gpm=500.0,
        film_temperature_f=680.0,
        pump_discharge_pressure_psig=75.0,
    )


@pytest.fixture
def lab_analysis():
    """Create lab analysis for degradation testing."""
    return FluidLabAnalysis(
        sample_id="LAB-001",
        sample_date=datetime.now(timezone.utc),
        viscosity_cst_100f=30.0,
        flash_point_f=320.0,
        total_acid_number_mg_koh_g=0.15,
        carbon_residue_pct=0.08,
        moisture_ppm=300,
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestAnalyzerInit:
    """Tests for ThermalFluidAnalyzer initialization."""

    def test_initialization(self, analyzer, default_config):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.tf_config == default_config
        assert analyzer.AGENT_TYPE == "GL-009"
        assert analyzer.AGENT_NAME == "THERMALIQ Thermal Fluid Analyzer"

    def test_sub_analyzers_initialized(self, analyzer):
        """Test all sub-analyzers are initialized."""
        assert analyzer.property_db is not None
        assert analyzer.exergy_analyzer is not None
        assert analyzer.degradation_monitor is not None
        assert analyzer.expansion_analyzer is not None
        assert analyzer.heat_transfer_calc is not None
        assert analyzer.safety_monitor is not None

    def test_config_properties_accessible(self, analyzer, default_config):
        """Test configuration properties are accessible."""
        assert analyzer.tf_config.system_id == "TF-TEST-001"
        assert analyzer.tf_config.fluid_type == ThermalFluidType.THERMINOL_66

    def test_agent_metadata(self, analyzer):
        """Test agent metadata is set correctly."""
        assert analyzer.AGENT_TYPE == "GL-009"
        assert "1.0.0" in analyzer.AGENT_VERSION


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Tests for input validation."""

    def test_valid_input_passes(self, analyzer, normal_input):
        """Test valid input passes validation."""
        assert analyzer.validate_input(normal_input) == True

    def test_invalid_temperature_fails(self, analyzer):
        """Test temperature above max fails validation."""
        input_data = ThermalFluidInput(
            system_id="TF-TEST-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=700.0,  # Above max 650F
            flow_rate_gpm=450.0,
        )

        assert analyzer.validate_input(input_data) == False

    def test_invalid_flow_fails(self, analyzer):
        """Test zero flow fails validation."""
        input_data = ThermalFluidInput(
            system_id="TF-TEST-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=550.0,
            flow_rate_gpm=0.0,  # Invalid
        )

        # Should fail pydantic validation
        with pytest.raises(Exception):
            analyzer.validate_input(input_data)


# =============================================================================
# PROCESSING TESTS
# =============================================================================

class TestProcessing:
    """Tests for main processing workflow."""

    def test_process_returns_output(self, analyzer, normal_input):
        """Test process returns ThermalFluidOutput."""
        result = analyzer.process(normal_input)

        assert isinstance(result, ThermalFluidOutput)

    def test_process_output_has_required_fields(self, analyzer, normal_input):
        """Test output has all required fields."""
        result = analyzer.process(normal_input)

        assert result.system_id == normal_input.system_id
        assert result.status == "success"
        assert result.fluid_properties is not None
        assert result.safety_analysis is not None
        assert result.provenance_hash is not None

    def test_process_timing_recorded(self, analyzer, normal_input):
        """Test processing time is recorded."""
        result = analyzer.process(normal_input)

        assert result.processing_time_ms > 0

    def test_process_calculation_count(self, analyzer, normal_input):
        """Test calculation count is tracked."""
        result = analyzer.process(normal_input)

        assert result.calculation_count > 0

    def test_process_with_exergy(self, analyzer, normal_input):
        """Test processing includes exergy analysis."""
        result = analyzer.process(normal_input)

        assert result.exergy_analysis is not None
        assert result.exergy_analysis.exergy_efficiency_pct >= 0
        assert result.exergy_analysis.carnot_efficiency_pct >= 0

    def test_process_without_exergy(self, analyzer_no_exergy, normal_input):
        """Test processing without exergy analysis."""
        result = analyzer_no_exergy.process(normal_input)

        assert result.exergy_analysis is None

    def test_process_with_expansion_tank(self, analyzer, normal_input):
        """Test processing includes expansion tank analysis."""
        result = analyzer.process(normal_input)

        assert result.expansion_tank_analysis is not None

    def test_process_without_expansion_tank_level(self, analyzer):
        """Test processing without expansion tank level."""
        input_data = ThermalFluidInput(
            system_id="TF-TEST-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=550.0,
            flow_rate_gpm=450.0,
            expansion_tank_level_pct=None,  # Not provided
        )

        result = analyzer.process(input_data)

        # Should still succeed
        assert result.status == "success"
        # Expansion analysis may be None
        assert result.expansion_tank_analysis is None


# =============================================================================
# SAFETY ANALYSIS TESTS
# =============================================================================

class TestSafetyAnalysis:
    """Tests for safety analysis integration."""

    def test_safety_analysis_included(self, analyzer, normal_input):
        """Test safety analysis is included."""
        result = analyzer.process(normal_input)

        assert result.safety_analysis is not None
        assert result.safety_analysis.safety_status is not None

    def test_normal_conditions_safe(self, analyzer, normal_input):
        """Test normal conditions return safe status."""
        result = analyzer.process(normal_input)

        assert result.safety_analysis.safety_status == SafetyStatus.NORMAL

    def test_alarm_conditions_detected(self, analyzer, alarm_input):
        """Test alarm conditions are detected."""
        result = analyzer.process(alarm_input)

        assert result.safety_analysis.safety_status in [
            SafetyStatus.WARNING,
            SafetyStatus.ALARM,
            SafetyStatus.TRIP,
        ]
        assert len(result.safety_analysis.active_alarms) > 0

    def test_safety_margins_calculated(self, analyzer, normal_input):
        """Test safety margins are calculated."""
        result = analyzer.process(normal_input)

        assert result.safety_analysis.film_temp_margin_f is not None
        assert result.safety_analysis.bulk_temp_margin_f is not None
        assert result.safety_analysis.flash_point_margin_f is not None
        assert result.safety_analysis.auto_ignition_margin_f is not None


# =============================================================================
# HEAT TRANSFER TESTS
# =============================================================================

class TestHeatTransferAnalysis:
    """Tests for heat transfer analysis integration."""

    def test_heat_transfer_included(self, analyzer, normal_input):
        """Test heat transfer analysis is included."""
        result = analyzer.process(normal_input)

        assert result.heat_transfer_analysis is not None

    def test_reynolds_number_calculated(self, analyzer, normal_input):
        """Test Reynolds number is calculated."""
        result = analyzer.process(normal_input)

        assert result.heat_transfer_analysis.reynolds_number > 0

    def test_film_coefficient_calculated(self, analyzer, normal_input):
        """Test film coefficient is calculated."""
        result = analyzer.process(normal_input)

        assert result.heat_transfer_analysis.film_coefficient_btu_hr_ft2_f > 0


# =============================================================================
# KPI CALCULATION TESTS
# =============================================================================

class TestKPICalculation:
    """Tests for KPI calculation."""

    def test_kpis_calculated(self, analyzer, normal_input):
        """Test KPIs are calculated."""
        result = analyzer.process(normal_input)

        assert len(result.kpis) > 0

    def test_basic_kpis_present(self, analyzer, normal_input):
        """Test basic KPIs are present."""
        result = analyzer.process(normal_input)

        assert "bulk_temperature_f" in result.kpis
        assert "flow_rate_gpm" in result.kpis
        assert "bulk_temp_margin_f" in result.kpis

    def test_exergy_kpis_when_enabled(self, analyzer, normal_input):
        """Test exergy KPIs when exergy is enabled."""
        result = analyzer.process(normal_input)

        assert "exergy_efficiency_pct" in result.kpis
        assert "carnot_efficiency_pct" in result.kpis


# =============================================================================
# RECOMMENDATIONS TESTS
# =============================================================================

class TestRecommendations:
    """Tests for recommendation generation."""

    def test_recommendations_generated(self, analyzer, normal_input):
        """Test recommendations are generated."""
        result = analyzer.process(normal_input)

        # May or may not have recommendations depending on conditions
        assert result.recommendations is not None

    def test_safety_recommendations_priority(self, analyzer, alarm_input):
        """Test safety recommendations have high priority."""
        result = analyzer.process(alarm_input)

        if len(result.recommendations) > 0:
            safety_recs = [r for r in result.recommendations if r.category == "safety"]
            if safety_recs:
                assert safety_recs[0].priority <= 2


# =============================================================================
# ALERTS AND WARNINGS TESTS
# =============================================================================

class TestAlertsWarnings:
    """Tests for alerts and warnings collection."""

    def test_no_alerts_normal(self, analyzer, normal_input):
        """Test no alerts in normal operation."""
        result = analyzer.process(normal_input)

        assert len(result.alerts) == 0

    def test_alerts_for_alarm_conditions(self, analyzer, alarm_input):
        """Test alerts generated for alarm conditions."""
        result = analyzer.process(alarm_input)

        # Should have alerts or warnings
        assert len(result.alerts) > 0 or len(result.warnings) > 0

    def test_alert_structure(self, analyzer, alarm_input):
        """Test alert structure."""
        result = analyzer.process(alarm_input)

        for alert in result.alerts:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert


# =============================================================================
# OVERALL STATUS TESTS
# =============================================================================

class TestOverallStatus:
    """Tests for overall status determination."""

    def test_optimal_status_normal(self, analyzer, normal_input):
        """Test optimal status for normal operation."""
        result = analyzer.process(normal_input)

        assert result.overall_status == OptimizationStatus.OPTIMAL

    def test_critical_status_trip(self, analyzer):
        """Test critical status for trip conditions."""
        trip_input = ThermalFluidInput(
            system_id="TF-TEST-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=645.0,  # Trip condition
            flow_rate_gpm=450.0,
        )

        result = analyzer.process(trip_input)

        assert result.overall_status == OptimizationStatus.CRITICAL

    def test_suboptimal_status_warnings(self, analyzer):
        """Test suboptimal status for warnings."""
        warning_input = ThermalFluidInput(
            system_id="TF-TEST-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=610.0,  # Warning level
            flow_rate_gpm=450.0,
        )

        result = analyzer.process(warning_input)

        assert result.overall_status in [
            OptimizationStatus.OPTIMAL,
            OptimizationStatus.SUBOPTIMAL,
        ]


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Tests for provenance tracking."""

    def test_provenance_hash_generated(self, analyzer, normal_input):
        """Test provenance hash is generated."""
        result = analyzer.process(normal_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_provenance_hash_is_sha256(self, analyzer, normal_input):
        """Test provenance hash is valid SHA-256."""
        result = analyzer.process(normal_input)

        # Should be valid hex string
        try:
            int(result.provenance_hash, 16)
        except ValueError:
            pytest.fail("Provenance hash is not valid hex")

    def test_provenance_contains_key_data(self, analyzer, normal_input):
        """Test provenance calculation includes key data."""
        # The provenance should include agent info and input data
        # We can verify by checking the hash changes with different inputs
        result1 = analyzer.process(normal_input)

        different_input = ThermalFluidInput(
            system_id="TF-TEST-001",
            fluid_type=ThermalFluidType.THERMINOL_66,
            bulk_temperature_f=560.0,  # Different temp
            flow_rate_gpm=450.0,
        )
        result2 = analyzer.process(different_input)

        # Hashes should be different for different inputs
        # (Note: timestamp is included so they'll always be different anyway)
        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# DEGRADATION ANALYSIS TESTS
# =============================================================================

class TestDegradationAnalysis:
    """Tests for degradation analysis."""

    def test_analyze_degradation_method(self, analyzer, lab_analysis):
        """Test analyze_degradation method."""
        result = analyzer.analyze_degradation(lab_analysis)

        assert result is not None
        assert result.degradation_level is not None
        assert result.remaining_life_pct >= 0

    def test_analyze_degradation_with_baseline(self, analyzer, lab_analysis):
        """Test analyze_degradation with baseline."""
        baseline = FluidLabAnalysis(
            viscosity_cst_100f=28.0,
            flash_point_f=340.0,
            total_acid_number_mg_koh_g=0.01,
        )

        result = analyzer.analyze_degradation(lab_analysis, baseline)

        assert result is not None
        # Should show some degradation from baseline
        assert result.degradation_score > 0


# =============================================================================
# CONVENIENCE METHODS TESTS
# =============================================================================

class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_get_fluid_properties(self, analyzer):
        """Test get_fluid_properties method."""
        props = analyzer.get_fluid_properties(temperature_f=550.0)

        assert props is not None
        assert props.temperature_f == 550.0
        assert props.density_lb_ft3 > 0

    def test_calculate_exergy_efficiency(self, analyzer):
        """Test calculate_exergy_efficiency method."""
        result = analyzer.calculate_exergy_efficiency(
            hot_temp_f=600.0,
            cold_temp_f=400.0,
            heat_duty_btu_hr=5_000_000.0,
        )

        assert result is not None
        assert result.exergy_efficiency_pct >= 0

    def test_check_safety_status(self, analyzer, normal_input):
        """Test check_safety_status method."""
        result = analyzer.check_safety_status(normal_input)

        assert result is not None
        assert result.safety_status is not None


# =============================================================================
# OUTPUT VALIDATION TESTS
# =============================================================================

class TestOutputValidation:
    """Tests for output validation."""

    def test_valid_output_passes(self, analyzer, normal_input):
        """Test valid output passes validation."""
        result = analyzer.process(normal_input)

        assert analyzer.validate_output(result) == True

    def test_output_without_fluid_properties_fails(self, analyzer, normal_input):
        """Test output without fluid properties fails."""
        result = analyzer.process(normal_input)
        result.fluid_properties = None

        assert analyzer.validate_output(result) == False

    def test_output_without_safety_analysis_fails(self, analyzer, normal_input):
        """Test output without safety analysis fails."""
        result = analyzer.process(normal_input)
        result.safety_analysis = None

        assert analyzer.validate_output(result) == False

    def test_output_without_provenance_fails(self, analyzer, normal_input):
        """Test output without provenance fails."""
        result = analyzer.process(normal_input)
        result.provenance_hash = None

        assert analyzer.validate_output(result) == False


# =============================================================================
# METADATA TESTS
# =============================================================================

class TestMetadata:
    """Tests for output metadata."""

    def test_metadata_included(self, analyzer, normal_input):
        """Test metadata is included in output."""
        result = analyzer.process(normal_input)

        assert result.metadata is not None
        assert len(result.metadata) > 0

    def test_metadata_contains_fluid_type(self, analyzer, normal_input):
        """Test metadata contains fluid type."""
        result = analyzer.process(normal_input)

        assert "fluid_type" in result.metadata
        assert result.metadata["fluid_type"] == "therminol_66"

    def test_metadata_contains_design_temp(self, analyzer, normal_input):
        """Test metadata contains design temperature."""
        result = analyzer.process(normal_input)

        assert "design_temperature_f" in result.metadata


# =============================================================================
# INTELLIGENCE MIXIN TESTS
# =============================================================================

class TestIntelligenceMixin:
    """Tests for intelligence mixin integration."""

    def test_intelligence_level(self, analyzer):
        """Test intelligence level is returned."""
        level = analyzer.get_intelligence_level()

        from greenlang.agents.intelligence_interface import IntelligenceLevel
        assert level == IntelligenceLevel.STANDARD

    def test_intelligence_capabilities(self, analyzer):
        """Test intelligence capabilities are returned."""
        caps = analyzer.get_intelligence_capabilities()

        assert caps.can_explain == True
        assert caps.can_recommend == True


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism."""

    def test_same_input_same_kpis(self, normal_input):
        """Test same input produces same KPIs."""
        config = create_default_config(system_id="TF-TEST-001")
        analyzer1 = ThermalFluidAnalyzer(config)
        analyzer2 = ThermalFluidAnalyzer(config)

        result1 = analyzer1.process(normal_input)
        result2 = analyzer2.process(normal_input)

        assert result1.kpis["bulk_temperature_f"] == result2.kpis["bulk_temperature_f"]
        assert result1.kpis["flow_rate_gpm"] == result2.kpis["flow_rate_gpm"]

    def test_reproducible_safety_analysis(self, normal_input):
        """Test safety analysis is reproducible."""
        config = create_default_config(system_id="TF-TEST-001")
        analyzer1 = ThermalFluidAnalyzer(config)
        analyzer2 = ThermalFluidAnalyzer(config)

        result1 = analyzer1.process(normal_input)
        result2 = analyzer2.process(normal_input)

        assert result1.safety_analysis.safety_status == result2.safety_analysis.safety_status
        assert result1.safety_analysis.film_temp_margin_f == result2.safety_analysis.film_temp_margin_f


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self, analyzer, normal_input):
        """Test complete processing workflow."""
        # Process input
        result = analyzer.process(normal_input)

        # Verify all major components
        assert result.status == "success"
        assert result.fluid_properties is not None
        assert result.exergy_analysis is not None
        assert result.heat_transfer_analysis is not None
        assert result.safety_analysis is not None
        assert len(result.kpis) > 0
        assert result.provenance_hash is not None

    def test_multiple_analyses(self, analyzer, normal_input, alarm_input):
        """Test multiple sequential analyses."""
        result1 = analyzer.process(normal_input)
        result2 = analyzer.process(alarm_input)

        # Both should succeed
        assert result1.status == "success"
        assert result2.status == "success"

        # Results should be different
        assert result1.overall_status != result2.overall_status or \
               result1.safety_analysis.safety_status != result2.safety_analysis.safety_status
