# -*- coding: utf-8 -*-
"""
Unit Tests for SteamTrapEnergyLossCalculator

Comprehensive test suite covering all calculator functionality including:
- Energy loss calculations for all failure modes
- Steam loss rate using Napier equation
- ROI and financial calculations
- Carbon emission calculations
- Provenance tracking
- Edge cases and error handling

Test Coverage Target: 90%+

Author: GL-BackendDeveloper
Date: December 2025
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
import json

import sys
import os

# Add the calculators directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculators.steam_trap_energy_loss_calculator import (
    SteamTrapEnergyLossCalculator,
    EnergyLossConfig,
    FailureMode,
    TrapType,
    SeverityLevel,
    FuelType,
    TrapSpecifications,
    SteamConditions,
    SteamLossResult,
    EnergyLossMetrics,
    CarbonEmissionResult,
    ROIResult,
    EnergyLossAnalysisResult,
    ProvenanceTracker,
    ProvenanceStep,
    STEAM_PROPERTIES_TABLE,
    DISCHARGE_COEFFICIENTS,
    EPA_EMISSION_FACTORS,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator():
    """Create a default calculator instance."""
    return SteamTrapEnergyLossCalculator()


@pytest.fixture
def custom_config():
    """Create a custom configuration for testing."""
    return EnergyLossConfig(
        steam_cost_usd_per_1000lb=Decimal("15.00"),
        fuel_cost_usd_per_mmbtu=Decimal("5.00"),
        fuel_type=FuelType.NATURAL_GAS,
        boiler_efficiency=Decimal("0.85"),
        operating_hours_per_year=8000,
        carbon_price_usd_per_ton=Decimal("75.00"),
        discount_rate=Decimal("0.08"),
        trap_lifetime_years=10
    )


@pytest.fixture
def calculator_with_config(custom_config):
    """Create a calculator with custom configuration."""
    return SteamTrapEnergyLossCalculator(config=custom_config)


# ============================================================================
# BASIC INITIALIZATION TESTS
# ============================================================================

class TestCalculatorInitialization:
    """Test calculator initialization."""

    def test_default_initialization(self, calculator):
        """Test default calculator initialization."""
        assert calculator is not None
        assert calculator.config is not None
        assert calculator.config.boiler_efficiency == Decimal("0.82")

    def test_custom_config_initialization(self, calculator_with_config, custom_config):
        """Test initialization with custom config."""
        assert calculator_with_config.config.boiler_efficiency == Decimal("0.85")
        assert calculator_with_config.config.fuel_type == FuelType.NATURAL_GAS

    def test_statistics_initialization(self, calculator):
        """Test statistics are properly initialized."""
        stats = calculator.get_statistics()
        assert stats["calculation_count"] == 0
        assert "supported_trap_types" in stats
        assert "supported_failure_modes" in stats


# ============================================================================
# STEAM LOSS CALCULATION TESTS
# ============================================================================

class TestSteamLossCalculation:
    """Test steam loss rate calculations."""

    def test_blow_through_failure_high_loss(self, calculator):
        """Test blow-through failure produces high steam loss."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-001",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        # Blow-through should have 90-100% loss
        assert float(result.steam_loss.steam_loss_lb_hr) > 0
        assert result.steam_loss.failure_severity in [
            SeverityLevel.CRITICAL, SeverityLevel.HIGH
        ]

    def test_leaking_failure_moderate_loss(self, calculator):
        """Test leaking failure produces moderate steam loss."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-002",
            failure_mode=FailureMode.LEAKING,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        # Leaking should have 10-50% loss
        assert float(result.steam_loss.steam_loss_lb_hr) > 0
        assert result.steam_loss.failure_severity in [
            SeverityLevel.MEDIUM, SeverityLevel.LOW, SeverityLevel.HIGH
        ]

    def test_blocked_failure_zero_loss(self, calculator):
        """Test blocked failure produces zero steam loss."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-003",
            failure_mode=FailureMode.BLOCKED,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        # Blocked trap should have zero steam loss
        assert float(result.steam_loss.steam_loss_lb_hr) == 0

    def test_normal_operation_minimal_loss(self, calculator):
        """Test normal operation produces minimal/no loss."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-004",
            failure_mode=FailureMode.NORMAL,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result.steam_loss.failure_severity == SeverityLevel.NONE

    def test_larger_orifice_higher_loss(self, calculator):
        """Test that larger orifice produces higher steam loss."""
        result_small = calculator.calculate_energy_loss(
            trap_id="TEST-005A",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=3.0,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        result_large = calculator.calculate_energy_loss(
            trap_id="TEST-005B",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=12.0,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert float(result_large.steam_loss.steam_loss_lb_hr) > \
               float(result_small.steam_loss.steam_loss_lb_hr)

    def test_higher_pressure_higher_loss(self, calculator):
        """Test that higher pressure produces higher steam loss."""
        result_low = calculator.calculate_energy_loss(
            trap_id="TEST-006A",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=2.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        result_high = calculator.calculate_energy_loss(
            trap_id="TEST-006B",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=20.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert float(result_high.steam_loss.steam_loss_lb_hr) > \
               float(result_low.steam_loss.steam_loss_lb_hr)


# ============================================================================
# TRAP TYPE TESTS
# ============================================================================

class TestTrapTypes:
    """Test different trap type calculations."""

    @pytest.mark.parametrize("trap_type", list(TrapType))
    def test_all_trap_types(self, calculator, trap_type):
        """Test calculation works for all trap types."""
        result = calculator.calculate_energy_loss(
            trap_id=f"TEST-{trap_type.value}",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=trap_type
        )

        assert result is not None
        assert result.trap_specs.trap_type == trap_type

    def test_venturi_highest_discharge_coefficient(self, calculator):
        """Test venturi has highest discharge coefficient."""
        result_venturi = calculator.calculate_energy_loss(
            trap_id="TEST-VENTURI",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.VENTURI
        )

        result_mechanical = calculator.calculate_energy_loss(
            trap_id="TEST-MECHANICAL",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.MECHANICAL
        )

        # Venturi has higher C_d, so higher flow for same conditions
        assert float(result_venturi.steam_loss.steam_loss_lb_hr) > \
               float(result_mechanical.steam_loss.steam_loss_lb_hr)


# ============================================================================
# ENERGY METRICS TESTS
# ============================================================================

class TestEnergyMetrics:
    """Test energy loss metrics calculations."""

    def test_energy_conversion_consistency(self, calculator):
        """Test energy unit conversions are consistent."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-ENERGY",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        metrics = result.energy_metrics

        # BTU/hr to kW conversion check (1 kW = 3412 BTU/hr)
        expected_kw = float(metrics.energy_loss_btu_hr) * 0.000293071
        assert abs(float(metrics.energy_loss_kw) - expected_kw) < 0.1

    def test_annual_energy_calculation(self, calculator):
        """Test annual energy loss calculation."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-ANNUAL",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        metrics = result.energy_metrics
        hours_per_year = calculator.config.operating_hours_per_year

        # Annual should be hourly * operating hours (accounting for efficiency)
        expected_annual = float(metrics.fuel_waste_mmbtu_hr) * hours_per_year
        assert abs(float(metrics.annual_energy_loss_mmbtu) - expected_annual) < 0.1


# ============================================================================
# CARBON EMISSION TESTS
# ============================================================================

class TestCarbonEmissions:
    """Test carbon emission calculations."""

    def test_natural_gas_emission_factor(self, calculator):
        """Test natural gas emission factor is applied correctly."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-CO2",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        # EPA natural gas factor is 53.07 kg CO2/MMBtu
        expected_factor = Decimal("53.07")
        assert result.carbon_emissions.emission_factor_kg_per_mmbtu == expected_factor

    def test_carbon_penalty_calculation(self, calculator_with_config):
        """Test carbon penalty is calculated correctly."""
        result = calculator_with_config.calculate_energy_loss(
            trap_id="TEST-PENALTY",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        # Penalty = CO2_tons * carbon_price
        co2_tons = result.carbon_emissions.co2_tons_year
        carbon_price = calculator_with_config.config.carbon_price_usd_per_ton
        expected_penalty = co2_tons * carbon_price

        assert result.carbon_emissions.carbon_penalty_usd_year == expected_penalty.quantize(
            Decimal("0.01")
        )

    @pytest.mark.parametrize("fuel_type", list(FuelType))
    def test_all_fuel_types_have_emission_factors(self, fuel_type):
        """Test all fuel types have emission factors defined."""
        assert fuel_type in EPA_EMISSION_FACTORS


# ============================================================================
# ROI CALCULATION TESTS
# ============================================================================

class TestROICalculations:
    """Test ROI and financial calculations."""

    def test_roi_calculation_with_replacement_cost(self, calculator):
        """Test ROI is calculated when replacement cost is provided."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-ROI",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=200.0
        )

        assert result.roi_analysis is not None
        assert result.roi_analysis.replacement_cost_usd == Decimal("200.00")

    def test_no_roi_without_replacement_cost(self, calculator):
        """Test no ROI analysis without replacement cost."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-NO-ROI",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result.roi_analysis is None

    def test_simple_payback_calculation(self, calculator):
        """Test simple payback period calculation."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-PAYBACK",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=200.0
        )

        roi = result.roi_analysis
        if float(roi.annual_savings_usd) > 0:
            expected_payback_years = 200.0 / float(roi.annual_savings_usd)
            expected_payback_days = expected_payback_years * 365
            assert abs(float(roi.simple_payback_days) - expected_payback_days) < 1

    def test_npv_positive_for_good_investment(self, calculator):
        """Test NPV is positive for high-loss traps."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-NPV",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=12.0,  # Large orifice = high loss
            pressure_bar_g=20.0,       # High pressure
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=200.0
        )

        # High loss should result in positive NPV
        assert float(result.roi_analysis.npv_lifetime_usd) > 0

    def test_recommendation_based_on_payback(self, calculator):
        """Test recommendations are appropriate for payback period."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-REC",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=12.0,
            pressure_bar_g=20.0,
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=100.0
        )

        roi = result.roi_analysis
        # Fast payback should result in urgent recommendation
        if float(roi.simple_payback_months) < 3:
            assert "IMMEDIATE" in roi.recommendation or "HIGH" in roi.recommendation


# ============================================================================
# PROVENANCE TRACKING TESTS
# ============================================================================

class TestProvenanceTracking:
    """Test provenance tracking and audit trail."""

    def test_provenance_hash_generated(self, calculator):
        """Test provenance hash is generated."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-PROV",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_deterministic_results(self, calculator):
        """Test same inputs produce same outputs (determinism)."""
        result1 = calculator.calculate_energy_loss(
            trap_id="TEST-DETERM",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        result2 = calculator.calculate_energy_loss(
            trap_id="TEST-DETERM",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result1.steam_loss.steam_loss_lb_hr == result2.steam_loss.steam_loss_lb_hr
        assert result1.annual_energy_cost_usd == result2.annual_energy_cost_usd

    def test_provenance_tracker_class(self):
        """Test ProvenanceTracker class directly."""
        tracker = ProvenanceTracker()

        tracker.record_step(
            operation="test_operation",
            inputs={"a": 1, "b": 2},
            formula="c = a + b",
            result=3
        )

        steps = tracker.get_steps()
        assert len(steps) == 1
        assert steps[0].operation == "test_operation"
        assert steps[0].result == 3

        hash_value = tracker.get_hash()
        assert len(hash_value) == 64


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_orifice_diameter_negative(self, calculator):
        """Test negative orifice diameter raises error."""
        with pytest.raises(ValueError, match="Invalid orifice diameter"):
            calculator.calculate_energy_loss(
                trap_id="TEST-ERR1",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=-5.0,
                pressure_bar_g=10.0,
                trap_type=TrapType.THERMODYNAMIC
            )

    def test_invalid_orifice_diameter_too_large(self, calculator):
        """Test orifice diameter > 100mm raises error."""
        with pytest.raises(ValueError, match="Invalid orifice diameter"):
            calculator.calculate_energy_loss(
                trap_id="TEST-ERR2",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=150.0,
                pressure_bar_g=10.0,
                trap_type=TrapType.THERMODYNAMIC
            )

    def test_invalid_pressure_negative(self, calculator):
        """Test negative pressure raises error."""
        with pytest.raises(ValueError, match="Invalid pressure"):
            calculator.calculate_energy_loss(
                trap_id="TEST-ERR3",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=6.35,
                pressure_bar_g=-5.0,
                trap_type=TrapType.THERMODYNAMIC
            )

    def test_invalid_pressure_too_high(self, calculator):
        """Test pressure > 100 bar raises error."""
        with pytest.raises(ValueError, match="Invalid pressure"):
            calculator.calculate_energy_loss(
                trap_id="TEST-ERR4",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=6.35,
                pressure_bar_g=150.0,
                trap_type=TrapType.THERMODYNAMIC
            )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_pressure(self, calculator):
        """Test calculation at zero gauge pressure (atmospheric)."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-ZERO-P",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=0.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result is not None
        assert float(result.steam_conditions.saturation_temp_c) == 100.0

    def test_high_pressure_boundary(self, calculator):
        """Test calculation at high pressure boundary."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-HIGH-P",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=50.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result is not None
        assert float(result.steam_conditions.saturation_temp_c) > 250.0

    def test_very_small_orifice(self, calculator):
        """Test calculation with very small orifice."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-SMALL",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=0.5,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        assert result is not None
        # Small orifice should have small loss
        assert float(result.steam_loss.steam_loss_lb_hr) < 10

    def test_custom_leak_severity_override(self, calculator):
        """Test custom leak severity override."""
        result_normal = calculator.calculate_energy_loss(
            trap_id="TEST-SEV1",
            failure_mode=FailureMode.LEAKING,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        result_override = calculator.calculate_energy_loss(
            trap_id="TEST-SEV2",
            failure_mode=FailureMode.LEAKING,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            leak_severity=0.9  # Force 90% leak
        )

        # Override should produce higher loss
        assert float(result_override.steam_loss.steam_loss_lb_hr) > \
               float(result_normal.steam_loss.steam_loss_lb_hr)


# ============================================================================
# BATCH PROCESSING TESTS
# ============================================================================

class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_calculation(self, calculator):
        """Test batch calculation of multiple traps."""
        trap_params = [
            {
                "trap_id": "BATCH-001",
                "failure_mode": FailureMode.BLOW_THROUGH,
                "orifice_diameter_mm": 6.35,
                "pressure_bar_g": 10.0,
                "trap_type": TrapType.THERMODYNAMIC
            },
            {
                "trap_id": "BATCH-002",
                "failure_mode": FailureMode.LEAKING,
                "orifice_diameter_mm": 8.0,
                "pressure_bar_g": 15.0,
                "trap_type": TrapType.MECHANICAL
            },
            {
                "trap_id": "BATCH-003",
                "failure_mode": FailureMode.BLOCKED,
                "orifice_diameter_mm": 5.0,
                "pressure_bar_g": 5.0,
                "trap_type": TrapType.THERMOSTATIC
            },
        ]

        results = calculator.calculate_batch(trap_params)

        assert len(results) == 3
        assert results[0].trap_specs.trap_id == "BATCH-001"
        assert results[1].trap_specs.trap_id == "BATCH-002"
        assert results[2].trap_specs.trap_id == "BATCH-003"


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestSerialization:
    """Test result serialization."""

    def test_to_dict_method(self, calculator):
        """Test to_dict produces valid dictionary."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-DICT",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=200.0
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["trap_id"] == "TEST-DICT"
        assert "steam_loss_lb_hr" in result_dict
        assert "annual_energy_cost_usd" in result_dict
        assert "roi_analysis" in result_dict

    def test_json_serializable(self, calculator):
        """Test result is JSON serializable."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-JSON",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=200.0
        )

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)

        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["trap_id"] == "TEST-JSON"


# ============================================================================
# STEAM PROPERTIES TABLE TESTS
# ============================================================================

class TestSteamPropertiesTable:
    """Test steam properties table and interpolation."""

    def test_steam_properties_at_known_pressure(self, calculator):
        """Test steam properties at known pressure points."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-PROPS",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC
        )

        # At 10 bar, saturation temp should be ~184.1C
        assert abs(float(result.steam_conditions.saturation_temp_c) - 184.1) < 1.0

    def test_interpolation_between_table_points(self, calculator):
        """Test interpolation between table pressure points."""
        result = calculator.calculate_energy_loss(
            trap_id="TEST-INTERP",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=7.0,  # Between 6 and 8 bar in table
            trap_type=TrapType.THERMODYNAMIC
        )

        # Temperature should be between 165C (6 bar) and 175.4C (8 bar)
        temp = float(result.steam_conditions.saturation_temp_c)
        assert 165.0 < temp < 175.4


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Test thread safety of calculator."""

    def test_calculation_count_thread_safe(self, calculator):
        """Test calculation count is thread-safe."""
        import threading

        def run_calculation():
            calculator.calculate_energy_loss(
                trap_id="THREAD-TEST",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=6.35,
                pressure_bar_g=10.0,
                trap_type=TrapType.THERMODYNAMIC
            )

        threads = [threading.Thread(target=run_calculation) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = calculator.get_statistics()
        assert stats["calculation_count"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
