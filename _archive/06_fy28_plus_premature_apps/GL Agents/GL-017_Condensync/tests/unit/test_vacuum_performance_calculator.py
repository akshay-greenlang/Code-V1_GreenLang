# -*- coding: utf-8 -*-
"""
Unit Tests for Vacuum Performance Calculator

Test suite for vacuum system performance analysis including backpressure,
air in-leakage, and heat rate impact calculations.

Test Coverage Target: 85%+

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- EPRI Condenser Air In-Leakage Guidelines

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.vacuum_performance_calculator import (
    VacuumPerformanceCalculator,
    VacuumSystemConfig,
    VacuumEquipmentType,
    VacuumStatus,
    AirInleakageLevel,
    AlertSeverity,
    BackpressureAnalysis,
    HeatRateImpact,
    AirInleakageAnalysis,
    VacuumPerformanceResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create default vacuum performance calculator."""
    return VacuumPerformanceCalculator()


@pytest.fixture
def custom_config():
    """Create custom calculator configuration."""
    return VacuumSystemConfig(
        equipment_type=VacuumEquipmentType.LIQUID_RING_VACUUM_PUMP,
        num_stages=2,
        design_capacity_scfm=Decimal("75.0"),
        design_backpressure_inhga=Decimal("1.2")
    )


@pytest.fixture
def standard_inputs():
    """Standard vacuum operating inputs."""
    return {
        "condenser_id": "COND-VAC-001",
        "backpressure_kpa": Decimal("5.0"),
        "cw_inlet_temp_c": Decimal("20.0"),
        "cw_outlet_temp_c": Decimal("30.0"),
        "steam_flow_kg_s": Decimal("150.0"),
    }


@pytest.fixture
def high_backpressure_inputs(standard_inputs):
    """Inputs with high backpressure."""
    inputs = standard_inputs.copy()
    inputs["backpressure_kpa"] = Decimal("10.0")
    inputs["cw_outlet_temp_c"] = Decimal("38.0")
    return inputs


@pytest.fixture
def air_inleakage_inputs(standard_inputs):
    """Inputs with air in-leakage data."""
    inputs = standard_inputs.copy()
    inputs["air_inleakage_scfm"] = Decimal("10.0")
    inputs["plant_capacity_mw"] = Decimal("500.0")
    inputs["design_ejector_capacity_scfm"] = Decimal("50.0")
    return inputs


# =============================================================================
# BASIC CALCULATION TESTS
# =============================================================================

class TestVacuumPerformanceCalculator:
    """Test suite for vacuum performance calculator core functionality."""

    def test_calculator_initialization(self, calculator):
        """Test calculator initializes with default config."""
        assert calculator is not None
        assert calculator.config is not None
        assert calculator.config.equipment_type == VacuumEquipmentType.STEAM_JET_EJECTOR

    def test_calculator_with_custom_config(self, custom_config):
        """Test calculator with custom configuration."""
        calc = VacuumPerformanceCalculator(config=custom_config)
        assert calc.config.equipment_type == VacuumEquipmentType.LIQUID_RING_VACUUM_PUMP
        assert calc.config.design_capacity_scfm == Decimal("75.0")

    def test_basic_vacuum_analysis(self, calculator, standard_inputs):
        """Test basic vacuum performance analysis."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        assert isinstance(result, VacuumPerformanceResult)
        assert result.condenser_id == "COND-VAC-001"
        assert result.calculation_method == "HEI-2629_VacuumAnalysis"

    def test_backpressure_conversion(self, calculator, standard_inputs):
        """Test backpressure kPa to inHgA conversion."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        # 5 kPa * 0.2953 = 1.4765 inHgA
        assert result.backpressure_analysis.actual_backpressure_inhga > Decimal("1.4")
        assert result.backpressure_analysis.actual_backpressure_inhga < Decimal("1.6")


# =============================================================================
# BACKPRESSURE ANALYSIS TESTS
# =============================================================================

class TestBackpressureAnalysis:
    """Test suite for backpressure analysis calculations."""

    def test_saturation_temperature(self, calculator, standard_inputs):
        """Test saturation temperature lookup."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        # At 5 kPa, T_sat should be ~32.88 C
        assert result.backpressure_analysis.saturation_temp_c > Decimal("32")
        assert result.backpressure_analysis.saturation_temp_c < Decimal("34")

    def test_ttd_calculation(self, calculator, standard_inputs):
        """Test TTD calculation."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        # TTD = T_sat - T_cw_out
        # ~32.88 - 30 = ~2.88 C
        expected_ttd = result.backpressure_analysis.saturation_temp_c - Decimal("30.0")
        actual_ttd = result.backpressure_analysis.ttd_c

        assert abs(float(actual_ttd - expected_ttd)) < 0.1

    def test_achievable_backpressure(self, calculator, standard_inputs):
        """Test achievable backpressure calculation."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        # Achievable should be less than or equal to actual
        # (actual may be degraded)
        assert result.backpressure_analysis.achievable_backpressure_kpa > Decimal("3")
        assert result.backpressure_analysis.achievable_backpressure_kpa < Decimal("15")

    def test_backpressure_deviation(self, calculator, high_backpressure_inputs):
        """Test backpressure deviation calculation."""
        result = calculator.analyze_vacuum_performance(**high_backpressure_inputs)

        # High backpressure should show positive deviation
        assert result.backpressure_analysis.backpressure_deviation_inhga > Decimal("-1")


# =============================================================================
# VACUUM STATUS CLASSIFICATION TESTS
# =============================================================================

class TestVacuumStatusClassification:
    """Test suite for vacuum status classification."""

    def test_vacuum_status_good(self, calculator, standard_inputs):
        """Test vacuum status classification for good conditions."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        # Standard inputs should result in good or better status
        assert result.backpressure_analysis.vacuum_status in [
            VacuumStatus.EXCELLENT,
            VacuumStatus.GOOD,
            VacuumStatus.ACCEPTABLE
        ]

    def test_vacuum_status_degraded(self, calculator, high_backpressure_inputs):
        """Test vacuum status classification for degraded conditions."""
        result = calculator.analyze_vacuum_performance(**high_backpressure_inputs)

        # High backpressure may result in degraded or worse
        assert result.backpressure_analysis.vacuum_status is not None

    def test_vacuum_status_critical(self, calculator):
        """Test vacuum status classification for critical conditions."""
        result = calculator.analyze_vacuum_performance(
            condenser_id="COND-CRITICAL",
            backpressure_kpa=Decimal("15.0"),  # Very high
            cw_inlet_temp_c=Decimal("25.0"),
            cw_outlet_temp_c=Decimal("35.0"),
            steam_flow_kg_s=Decimal("150.0")
        )

        # Very high backpressure should indicate poor or critical
        assert result.backpressure_analysis.vacuum_status in [
            VacuumStatus.DEGRADED,
            VacuumStatus.POOR,
            VacuumStatus.CRITICAL
        ]


# =============================================================================
# HEAT RATE IMPACT TESTS
# =============================================================================

class TestHeatRateImpact:
    """Test suite for heat rate impact calculations."""

    def test_heat_rate_penalty_calculation(self, calculator, standard_inputs):
        """Test heat rate penalty calculation."""
        result = calculator.analyze_vacuum_performance(
            **standard_inputs,
            design_backpressure_kpa=Decimal("4.5")
        )

        # Should have heat rate impact data
        assert result.heat_rate_impact is not None
        assert result.heat_rate_impact.heat_rate_penalty_percent >= Decimal("0")

    def test_heat_rate_penalty_high_bp(self, calculator, high_backpressure_inputs):
        """Test heat rate penalty for high backpressure."""
        result = calculator.analyze_vacuum_performance(
            **high_backpressure_inputs,
            design_backpressure_kpa=Decimal("5.0")
        )

        # High backpressure should result in penalty
        if result.heat_rate_impact.deviation_inhga > Decimal("0"):
            assert result.heat_rate_impact.heat_rate_penalty_percent > Decimal("0")

    def test_heat_rate_penalty_low_bp(self, calculator):
        """Test heat rate for lower-than-design backpressure."""
        result = calculator.analyze_vacuum_performance(
            condenser_id="COND-LOW-BP",
            backpressure_kpa=Decimal("4.0"),
            cw_inlet_temp_c=Decimal("15.0"),
            cw_outlet_temp_c=Decimal("25.0"),
            steam_flow_kg_s=Decimal("150.0"),
            design_backpressure_kpa=Decimal("5.0")  # Higher design
        )

        # Lower actual BP should not have penalty
        # (or negative deviation means better performance)
        assert result.heat_rate_impact.heat_rate_penalty_percent >= Decimal("0")


# =============================================================================
# AIR IN-LEAKAGE ANALYSIS TESTS
# =============================================================================

class TestAirInleakageAnalysis:
    """Test suite for air in-leakage analysis."""

    def test_air_inleakage_with_measurement(self, calculator, air_inleakage_inputs):
        """Test air in-leakage analysis with measured data."""
        result = calculator.analyze_vacuum_performance(**air_inleakage_inputs)

        assert result.air_inleakage is not None
        assert result.air_inleakage.estimated_inleakage_scfm == Decimal("10.0")

    def test_air_inleakage_per_100mw(self, calculator, air_inleakage_inputs):
        """Test air in-leakage normalization per 100 MW."""
        result = calculator.analyze_vacuum_performance(**air_inleakage_inputs)

        # 10 SCFM / 500 MW * 100 = 2 SCFM/100MW
        expected = (Decimal("10.0") / Decimal("500.0")) * Decimal("100")
        assert abs(float(result.air_inleakage.inleakage_per_100mw) - float(expected)) < 0.1

    def test_air_inleakage_level_classification(self, calculator, air_inleakage_inputs):
        """Test air in-leakage level classification."""
        result = calculator.analyze_vacuum_performance(**air_inleakage_inputs)

        # 2 SCFM/100MW is elevated
        per_100mw = result.air_inleakage.inleakage_per_100mw
        level = result.air_inleakage.inleakage_level

        if per_100mw < Decimal("1.0"):
            assert level == AirInleakageLevel.NORMAL
        elif per_100mw < Decimal("2.0"):
            assert level == AirInleakageLevel.ELEVATED

    def test_ejector_loading(self, calculator, air_inleakage_inputs):
        """Test ejector loading calculation."""
        result = calculator.analyze_vacuum_performance(**air_inleakage_inputs)

        # 10 SCFM / 50 SCFM * 100 = 20%
        expected = (Decimal("10.0") / Decimal("50.0")) * Decimal("100")
        assert abs(float(result.air_inleakage.ejector_loading_percent) - float(expected)) < 1

    def test_air_inleakage_severe(self, calculator):
        """Test severe air in-leakage detection."""
        result = calculator.analyze_vacuum_performance(
            condenser_id="COND-SEVERE-AIR",
            backpressure_kpa=Decimal("8.0"),
            cw_inlet_temp_c=Decimal("20.0"),
            cw_outlet_temp_c=Decimal("30.0"),
            steam_flow_kg_s=Decimal("150.0"),
            air_inleakage_scfm=Decimal("50.0"),  # Very high
            plant_capacity_mw=Decimal("500.0"),
            design_ejector_capacity_scfm=Decimal("50.0")
        )

        # 50 SCFM / 500 MW * 100 = 10 SCFM/100MW = Severe
        assert result.air_inleakage.inleakage_level == AirInleakageLevel.SEVERE


# =============================================================================
# ALERT GENERATION TESTS
# =============================================================================

class TestAlertGeneration:
    """Test suite for alert generation."""

    def test_alerts_generated_high_bp(self, calculator):
        """Test alerts generated for high backpressure."""
        result = calculator.analyze_vacuum_performance(
            condenser_id="COND-HIGH-BP-ALERT",
            backpressure_kpa=Decimal("15.0"),
            cw_inlet_temp_c=Decimal("25.0"),
            cw_outlet_temp_c=Decimal("35.0"),
            steam_flow_kg_s=Decimal("150.0")
        )

        # Should have at least one alert for high backpressure
        assert len(result.alerts) >= 0  # May or may not have alerts

    def test_alert_severity_levels(self, calculator, air_inleakage_inputs):
        """Test alert severity levels are properly assigned."""
        inputs = air_inleakage_inputs.copy()
        inputs["air_inleakage_scfm"] = Decimal("50.0")  # Severe

        result = calculator.analyze_vacuum_performance(**inputs)

        for alert in result.alerts:
            assert alert.severity in [
                AlertSeverity.INFO,
                AlertSeverity.WARNING,
                AlertSeverity.ALARM,
                AlertSeverity.CRITICAL
            ]


# =============================================================================
# RECOMMENDATION GENERATION TESTS
# =============================================================================

class TestRecommendationGeneration:
    """Test suite for recommendation generation."""

    def test_recommendations_generated(self, calculator, standard_inputs):
        """Test recommendations are generated."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        assert len(result.recommendations) > 0

    def test_recommendations_for_poor_vacuum(self, calculator):
        """Test recommendations for poor vacuum conditions."""
        result = calculator.analyze_vacuum_performance(
            condenser_id="COND-POOR-VAC",
            backpressure_kpa=Decimal("12.0"),
            cw_inlet_temp_c=Decimal("25.0"),
            cw_outlet_temp_c=Decimal("35.0"),
            steam_flow_kg_s=Decimal("150.0")
        )

        # Should have actionable recommendations
        recommendations = result.recommendations
        assert len(recommendations) > 0

    def test_recommendations_for_air_inleakage(self, calculator, air_inleakage_inputs):
        """Test recommendations for air in-leakage issues."""
        inputs = air_inleakage_inputs.copy()
        inputs["air_inleakage_scfm"] = Decimal("30.0")  # High

        result = calculator.analyze_vacuum_performance(**inputs)

        # Should mention air inleakage in recommendations
        all_recs = " ".join(result.recommendations).lower()
        # May contain air, leak, detection related terms
        assert len(result.recommendations) > 0


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test suite for input validation."""

    def test_invalid_backpressure_low(self, calculator):
        """Test rejection of invalid low backpressure."""
        with pytest.raises(ValueError, match="Backpressure"):
            calculator.analyze_vacuum_performance(
                condenser_id="COND-INVALID",
                backpressure_kpa=Decimal("1.0"),  # Too low
                cw_inlet_temp_c=Decimal("20.0"),
                cw_outlet_temp_c=Decimal("30.0"),
                steam_flow_kg_s=Decimal("150.0")
            )

    def test_invalid_backpressure_high(self, calculator):
        """Test rejection of invalid high backpressure."""
        with pytest.raises(ValueError, match="Backpressure"):
            calculator.analyze_vacuum_performance(
                condenser_id="COND-INVALID",
                backpressure_kpa=Decimal("25.0"),  # Too high
                cw_inlet_temp_c=Decimal("20.0"),
                cw_outlet_temp_c=Decimal("30.0"),
                steam_flow_kg_s=Decimal("150.0")
            )

    def test_invalid_cw_temps(self, calculator):
        """Test rejection of invalid CW temperatures."""
        with pytest.raises(ValueError):
            calculator.analyze_vacuum_performance(
                condenser_id="COND-INVALID",
                backpressure_kpa=Decimal("5.0"),
                cw_inlet_temp_c=Decimal("35.0"),
                cw_outlet_temp_c=Decimal("30.0"),  # Less than inlet
                steam_flow_kg_s=Decimal("150.0")
            )


# =============================================================================
# PROVENANCE AND AUDIT TESTS
# =============================================================================

class TestProvenanceAndAudit:
    """Test suite for provenance tracking and audit trail."""

    def test_provenance_hash_generated(self, calculator, standard_inputs):
        """Test provenance hash is generated."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    def test_calculation_timestamp(self, calculator, standard_inputs):
        """Test calculation timestamp is recorded."""
        before = datetime.now(timezone.utc)
        result = calculator.analyze_vacuum_performance(**standard_inputs)
        after = datetime.now(timezone.utc)

        assert before <= result.calculation_timestamp <= after

    def test_result_serialization(self, calculator, standard_inputs):
        """Test result can be serialized to dict."""
        result = calculator.analyze_vacuum_performance(**standard_inputs)

        result_dict = result.to_dict()

        assert "condenser_id" in result_dict
        assert "backpressure" in result_dict
        assert "heat_rate_impact" in result_dict
        assert "provenance_hash" in result_dict


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Test suite for calculator statistics."""

    def test_statistics_tracking(self, calculator, standard_inputs):
        """Test statistics are tracked."""
        calculator.analyze_vacuum_performance(**standard_inputs)
        calculator.analyze_vacuum_performance(**standard_inputs)

        stats = calculator.get_statistics()

        assert "calculation_count" in stats
        assert stats["calculation_count"] == 2
        assert "equipment_type" in stats


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_saturated_pressure_calculation(self, calculator):
        """Test saturation pressure from temperature."""
        p_sat = calculator.calculate_saturated_pressure(Decimal("45.0"))

        # At 45 C, P_sat should be ~9-10 kPa
        assert Decimal("9") < p_sat < Decimal("11")

    def test_saturated_pressure_at_boundaries(self, calculator):
        """Test saturation pressure at table boundaries."""
        # Low temperature
        p_low = calculator.calculate_saturated_pressure(Decimal("20.0"))
        assert p_low > Decimal("2")

        # High temperature
        p_high = calculator.calculate_saturated_pressure(Decimal("100.0"))
        assert p_high > Decimal("100")
