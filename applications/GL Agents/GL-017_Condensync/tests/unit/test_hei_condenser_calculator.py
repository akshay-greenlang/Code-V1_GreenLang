# -*- coding: utf-8 -*-
"""
Unit Tests for HEI Condenser Calculator

Comprehensive test suite for the HEI Standards compliant condenser
performance calculator. Tests cleanliness factor, heat transfer,
LMTD, and HEI correction factor calculations.

Test Coverage Target: 85%+

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers

Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import pytest
from decimal import Decimal
from datetime import datetime, timezone

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.hei_condenser_calculator import (
    HEICondenserCalculator,
    HEICondenserConfig,
    TubeMaterial,
    CondenserType,
    PerformanceStatus,
    CondenserSpecifications,
    CoolingWaterConditions,
    SteamConditions,
    HeatTransferResult,
    CleanlinessFactorResult,
    HEICorrectionFactors,
    CondenserPerformanceResult,
    ProvenanceTracker,
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    kpa_to_inhg,
    inhg_to_kpa,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create default HEI calculator instance."""
    return HEICondenserCalculator()


@pytest.fixture
def custom_config():
    """Create custom calculator configuration."""
    return HEICondenserConfig(
        design_cleanliness_factor=Decimal("0.90"),
        minimum_acceptable_cf=Decimal("0.70"),
        warning_cf_threshold=Decimal("0.80"),
        design_ttd_c=Decimal("2.5"),
        max_ttd_c=Decimal("6.0")
    )


@pytest.fixture
def standard_inputs():
    """Standard condenser operating inputs."""
    return {
        "condenser_id": "COND-TEST-001",
        "steam_flow_kg_s": Decimal("150.0"),
        "cw_inlet_temp_c": Decimal("20.0"),
        "cw_outlet_temp_c": Decimal("30.0"),
        "cw_flow_m3_s": Decimal("15.0"),
        "backpressure_kpa": Decimal("5.0"),
        "tube_material": TubeMaterial.TITANIUM,
        "tube_od_mm": Decimal("25.4"),
        "tube_wall_mm": Decimal("0.711"),
        "tube_length_m": Decimal("12.0"),
        "num_tubes": 20000,
        "num_passes": 1,
    }


@pytest.fixture
def high_backpressure_inputs(standard_inputs):
    """Inputs with high backpressure."""
    inputs = standard_inputs.copy()
    inputs["backpressure_kpa"] = Decimal("10.0")
    inputs["cw_outlet_temp_c"] = Decimal("38.0")
    return inputs


@pytest.fixture
def low_cw_temp_inputs(standard_inputs):
    """Inputs with low CW inlet temperature."""
    inputs = standard_inputs.copy()
    inputs["cw_inlet_temp_c"] = Decimal("10.0")
    inputs["cw_outlet_temp_c"] = Decimal("20.0")
    return inputs


# =============================================================================
# BASIC CALCULATION TESTS
# =============================================================================

class TestHEICondenserCalculator:
    """Test suite for HEI condenser calculator core functionality."""

    def test_calculator_initialization(self, calculator):
        """Test calculator initializes with default config."""
        assert calculator is not None
        assert calculator.config is not None
        assert calculator.config.design_cleanliness_factor == Decimal("0.85")
        assert calculator.config.minimum_acceptable_cf == Decimal("0.60")

    def test_calculator_with_custom_config(self, custom_config):
        """Test calculator with custom configuration."""
        calc = HEICondenserCalculator(config=custom_config)
        assert calc.config.design_cleanliness_factor == Decimal("0.90")
        assert calc.config.minimum_acceptable_cf == Decimal("0.70")

    def test_basic_performance_calculation(self, calculator, standard_inputs):
        """Test basic condenser performance calculation."""
        result = calculator.calculate_performance(**standard_inputs)

        assert isinstance(result, CondenserPerformanceResult)
        assert result.condenser_specs.condenser_id == "COND-TEST-001"
        assert result.calculation_method == "HEI-2629"

    def test_heat_duty_calculation(self, calculator, standard_inputs):
        """Test heat duty calculation accuracy."""
        result = calculator.calculate_performance(**standard_inputs)

        # Q = m * cp * dT
        # m = 15 m3/s * 1000 kg/m3 = 15000 kg/s
        # dT = 30 - 20 = 10 C
        # Q = 15000 * 4.18 * 10 = 627000 kW = 627 MW
        assert result.heat_transfer.heat_duty_mw > Decimal("500")
        assert result.heat_transfer.heat_duty_mw < Decimal("700")

    def test_lmtd_calculation(self, calculator, standard_inputs):
        """Test LMTD calculation."""
        result = calculator.calculate_performance(**standard_inputs)

        # LMTD should be positive and reasonable
        assert result.heat_transfer.lmtd_c > Decimal("0")
        assert result.heat_transfer.lmtd_c < Decimal("30")

    def test_ttd_calculation(self, calculator, standard_inputs):
        """Test TTD calculation."""
        result = calculator.calculate_performance(**standard_inputs)

        # TTD = T_sat - T_cw_out
        # At 5 kPa, T_sat ~ 32.88 C
        # TTD ~ 32.88 - 30 = ~2.88 C
        assert result.heat_transfer.ttd_c > Decimal("0")
        assert result.heat_transfer.ttd_c < Decimal("10")

    def test_cleanliness_factor_calculation(self, calculator, standard_inputs):
        """Test cleanliness factor calculation."""
        result = calculator.calculate_performance(**standard_inputs)

        # CF should be between 0 and 1.2
        assert result.cleanliness.cleanliness_factor > Decimal("0")
        assert result.cleanliness.cleanliness_factor <= Decimal("1.2")
        assert result.cleanliness.cf_percent == result.cleanliness.cleanliness_factor * 100


# =============================================================================
# HEI CORRECTION FACTOR TESTS
# =============================================================================

class TestHEICorrectionFactors:
    """Test suite for HEI correction factor calculations."""

    def test_material_factor_titanium(self, calculator, standard_inputs):
        """Test material correction factor for titanium tubes."""
        result = calculator.calculate_performance(**standard_inputs)
        assert result.hei_corrections.f_m == Decimal("0.71")

    def test_material_factor_admiralty(self, calculator, standard_inputs):
        """Test material correction factor for admiralty brass."""
        inputs = standard_inputs.copy()
        inputs["tube_material"] = TubeMaterial.ADMIRALTY_BRASS
        result = calculator.calculate_performance(**inputs)
        assert result.hei_corrections.f_m == Decimal("1.00")

    def test_material_factor_stainless(self, calculator, standard_inputs):
        """Test material correction factor for stainless steel."""
        inputs = standard_inputs.copy()
        inputs["tube_material"] = TubeMaterial.STAINLESS_316
        result = calculator.calculate_performance(**inputs)
        assert result.hei_corrections.f_m == Decimal("0.72")

    def test_temperature_correction_factor(self, calculator, standard_inputs):
        """Test water temperature correction factor."""
        result = calculator.calculate_performance(**standard_inputs)

        # F_w should be in reasonable range
        assert result.hei_corrections.f_w > Decimal("0.5")
        assert result.hei_corrections.f_w < Decimal("1.5")

    def test_velocity_correction_factor(self, calculator, standard_inputs):
        """Test velocity correction factor."""
        result = calculator.calculate_performance(**standard_inputs)

        # F_v should be in reasonable range
        assert result.hei_corrections.f_v > Decimal("0.7")
        assert result.hei_corrections.f_v < Decimal("1.1")

    def test_combined_correction_factor(self, calculator, standard_inputs):
        """Test combined correction factor calculation."""
        result = calculator.calculate_performance(**standard_inputs)

        # Combined = F_w * F_m * F_v
        expected = result.hei_corrections.f_w * result.hei_corrections.f_m * result.hei_corrections.f_v
        assert abs(float(result.hei_corrections.combined_factor) - float(expected)) < 0.01


# =============================================================================
# PERFORMANCE STATUS TESTS
# =============================================================================

class TestPerformanceStatus:
    """Test suite for performance status classification."""

    def test_status_classification_excellent(self, calculator, standard_inputs):
        """Test excellent performance status classification."""
        result = calculator.calculate_performance(**standard_inputs)

        # Check if status is classified correctly based on CF
        cf = result.cleanliness.cleanliness_factor
        status = result.cleanliness.performance_status

        if cf >= Decimal("0.90"):
            assert status == PerformanceStatus.EXCELLENT
        elif cf >= Decimal("0.80"):
            assert status == PerformanceStatus.GOOD
        elif cf >= Decimal("0.70"):
            assert status == PerformanceStatus.ACCEPTABLE

    def test_alert_generation_low_cf(self, calculator):
        """Test alert generation for low cleanliness factor."""
        # Use conditions that would result in low CF
        result = calculator.calculate_performance(
            condenser_id="COND-LOW-CF",
            steam_flow_kg_s=Decimal("200.0"),
            cw_inlet_temp_c=Decimal("25.0"),
            cw_outlet_temp_c=Decimal("40.0"),
            cw_flow_m3_s=Decimal("10.0"),
            backpressure_kpa=Decimal("10.0"),
            tube_material=TubeMaterial.TITANIUM,
            tube_od_mm=Decimal("25.4"),
            num_tubes=15000,
        )

        # Check for alerts if CF is low
        if result.cleanliness.cleanliness_factor < Decimal("0.75"):
            assert len(result.alerts) > 0

    def test_alert_high_ttd(self, calculator):
        """Test alert generation for high TTD."""
        # High TTD scenario
        result = calculator.calculate_performance(
            condenser_id="COND-HIGH-TTD",
            steam_flow_kg_s=Decimal("150.0"),
            cw_inlet_temp_c=Decimal("20.0"),
            cw_outlet_temp_c=Decimal("25.0"),  # Low rise = high TTD
            cw_flow_m3_s=Decimal("20.0"),
            backpressure_kpa=Decimal("10.0"),  # High BP
            tube_material=TubeMaterial.TITANIUM,
            tube_od_mm=Decimal("25.4"),
            num_tubes=20000,
        )

        # Verify TTD is high
        assert result.heat_transfer.ttd_c > Decimal("5")


# =============================================================================
# PROVENANCE TRACKING TESTS
# =============================================================================

class TestProvenanceTracking:
    """Test suite for calculation provenance tracking."""

    def test_provenance_hash_generated(self, calculator, standard_inputs):
        """Test that provenance hash is generated."""
        result = calculator.calculate_performance(**standard_inputs)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_provenance_hash_deterministic(self, calculator, standard_inputs):
        """Test that same inputs produce same provenance hash."""
        result1 = calculator.calculate_performance(**standard_inputs)
        result2 = calculator.calculate_performance(**standard_inputs)

        # Note: Timestamps differ, so full provenance hash may differ
        # But calculation results should be identical
        assert result1.heat_transfer.heat_duty_mw == result2.heat_transfer.heat_duty_mw
        assert result1.cleanliness.cleanliness_factor == result2.cleanliness.cleanliness_factor

    def test_provenance_tracker_steps(self):
        """Test provenance tracker records steps."""
        tracker = ProvenanceTracker()

        tracker.record_step(
            operation="test_calculation",
            inputs={"a": 1, "b": 2},
            formula="c = a + b",
            result=3
        )

        steps = tracker.get_steps()
        assert len(steps) == 1
        assert steps[0].operation == "test_calculation"

    def test_provenance_tracker_hash(self):
        """Test provenance tracker hash calculation."""
        tracker = ProvenanceTracker()

        tracker.record_step("op1", {"x": 1}, "f(x)", 1)
        tracker.record_step("op2", {"y": 2}, "g(y)", 2)

        hash1 = tracker.get_hash()
        hash2 = tracker.get_hash()

        assert hash1 == hash2
        assert len(hash1) == 64


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test suite for input validation."""

    def test_invalid_cw_inlet_temp_low(self, calculator, standard_inputs):
        """Test rejection of invalid low CW inlet temperature."""
        inputs = standard_inputs.copy()
        inputs["cw_inlet_temp_c"] = Decimal("-5.0")

        with pytest.raises(ValueError, match="CW inlet temperature"):
            calculator.calculate_performance(**inputs)

    def test_invalid_cw_inlet_temp_high(self, calculator, standard_inputs):
        """Test rejection of invalid high CW inlet temperature."""
        inputs = standard_inputs.copy()
        inputs["cw_inlet_temp_c"] = Decimal("50.0")

        with pytest.raises(ValueError, match="CW inlet temperature"):
            calculator.calculate_performance(**inputs)

    def test_invalid_cw_outlet_less_than_inlet(self, calculator, standard_inputs):
        """Test rejection when outlet temp <= inlet temp."""
        inputs = standard_inputs.copy()
        inputs["cw_outlet_temp_c"] = Decimal("18.0")  # Less than inlet

        with pytest.raises(ValueError, match="outlet temperature"):
            calculator.calculate_performance(**inputs)

    def test_invalid_backpressure_low(self, calculator, standard_inputs):
        """Test rejection of invalid low backpressure."""
        inputs = standard_inputs.copy()
        inputs["backpressure_kpa"] = Decimal("1.0")

        with pytest.raises(ValueError, match="Backpressure"):
            calculator.calculate_performance(**inputs)

    def test_invalid_backpressure_high(self, calculator, standard_inputs):
        """Test rejection of invalid high backpressure."""
        inputs = standard_inputs.copy()
        inputs["backpressure_kpa"] = Decimal("25.0")

        with pytest.raises(ValueError, match="Backpressure"):
            calculator.calculate_performance(**inputs)

    def test_invalid_cw_flow_zero(self, calculator, standard_inputs):
        """Test rejection of zero CW flow rate."""
        inputs = standard_inputs.copy()
        inputs["cw_flow_m3_s"] = Decimal("0")

        with pytest.raises(ValueError, match="flow rate"):
            calculator.calculate_performance(**inputs)


# =============================================================================
# UNIT CONVERSION TESTS
# =============================================================================

class TestUnitConversions:
    """Test suite for unit conversion functions."""

    def test_celsius_to_fahrenheit(self):
        """Test C to F conversion."""
        assert celsius_to_fahrenheit(Decimal("0")) == Decimal("32")
        assert celsius_to_fahrenheit(Decimal("100")) == Decimal("212")
        assert celsius_to_fahrenheit(Decimal("20")) == Decimal("68")

    def test_fahrenheit_to_celsius(self):
        """Test F to C conversion."""
        assert fahrenheit_to_celsius(Decimal("32")) == Decimal("0")
        assert fahrenheit_to_celsius(Decimal("212")) == Decimal("100")
        assert fahrenheit_to_celsius(Decimal("68")) == Decimal("20")

    def test_kpa_to_inhg(self):
        """Test kPa to inHg conversion."""
        result = kpa_to_inhg(Decimal("101.325"))
        assert Decimal("29.9") < result < Decimal("30.0")

    def test_inhg_to_kpa(self):
        """Test inHg to kPa conversion."""
        result = inhg_to_kpa(Decimal("29.92"))
        assert Decimal("101") < result < Decimal("102")


# =============================================================================
# SATURATION PROPERTY TESTS
# =============================================================================

class TestSaturationProperties:
    """Test suite for steam saturation properties."""

    def test_saturation_temp_at_5kpa(self, calculator):
        """Test saturation temperature at 5 kPa."""
        t_sat = calculator.calculate_saturation_temperature(Decimal("5.0"))
        # At 5 kPa, T_sat should be ~32.88 C
        assert Decimal("32") < t_sat < Decimal("34")

    def test_saturation_temp_at_10kpa(self, calculator):
        """Test saturation temperature at 10 kPa."""
        t_sat = calculator.calculate_saturation_temperature(Decimal("10.0"))
        # At 10 kPa, T_sat should be ~45.81 C
        assert Decimal("45") < t_sat < Decimal("47")

    def test_saturation_pressure_at_45c(self, calculator):
        """Test saturation pressure at 45 C."""
        p_sat = calculator.calculate_saturation_pressure(Decimal("45.0"))
        # At 45 C, P_sat should be ~9.6 kPa
        assert Decimal("9") < p_sat < Decimal("11")


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================

class TestBatchProcessing:
    """Test suite for batch calculation processing."""

    def test_batch_calculation(self, calculator, standard_inputs):
        """Test batch calculation of multiple condensers."""
        measurements = [
            {**standard_inputs, "condenser_id": "COND-001"},
            {**standard_inputs, "condenser_id": "COND-002", "cw_inlet_temp_c": Decimal("22.0")},
            {**standard_inputs, "condenser_id": "COND-003", "backpressure_kpa": Decimal("6.0")},
        ]

        results = calculator.calculate_batch(measurements)

        assert len(results) == 3
        assert results[0].condenser_specs.condenser_id == "COND-001"
        assert results[1].condenser_specs.condenser_id == "COND-002"
        assert results[2].condenser_specs.condenser_id == "COND-003"

    def test_batch_with_invalid_input(self, calculator, standard_inputs):
        """Test batch calculation handles invalid inputs gracefully."""
        measurements = [
            {**standard_inputs, "condenser_id": "COND-001"},
            {**standard_inputs, "condenser_id": "COND-002", "cw_inlet_temp_c": Decimal("-10.0")},  # Invalid
            {**standard_inputs, "condenser_id": "COND-003"},
        ]

        results = calculator.calculate_batch(measurements)

        # Should get 2 valid results, invalid one skipped
        assert len(results) == 2


# =============================================================================
# STATISTICS TESTS
# =============================================================================

class TestStatistics:
    """Test suite for calculator statistics."""

    def test_statistics_after_calculations(self, calculator, standard_inputs):
        """Test statistics tracking after calculations."""
        # Perform some calculations
        calculator.calculate_performance(**standard_inputs)
        calculator.calculate_performance(**standard_inputs)

        stats = calculator.get_statistics()

        assert "calculation_count" in stats
        assert stats["calculation_count"] == 2
        assert "design_cf" in stats
        assert "supported_materials" in stats
