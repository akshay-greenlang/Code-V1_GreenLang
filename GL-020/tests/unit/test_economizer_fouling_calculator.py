"""
GL-020 ECONOPULSE: Economizer Fouling Calculator Test Suite

Comprehensive test coverage for the advanced economizer fouling calculator.
Tests include unit tests, edge cases, provenance verification, and
calculation accuracy validation.

Author: GL-BackendDeveloper
Test Coverage Target: 85%+
"""

from __future__ import annotations

import hashlib
import json
import math
import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from calculators.economizer_fouling_calculator import (
    # Enumerations
    FoulingSide,
    FoulingMechanism,
    CleaningMethod,
    TrendModel,
    # Constants
    ASME_REFERENCE_CONDITIONS,
    TEMA_FOULING_FACTORS,
    FOULING_SEVERITY_THRESHOLDS,
    CO2_EMISSION_FACTORS,
    # Data Classes
    FoulingMeasurement,
    FoulingFactorResult,
    FoulingRateResult,
    HeatLossResult,
    FuelPenaltyResult,
    CarbonPenaltyResult,
    CleaningComparisonResult,
    CleaningIntervalResult,
    # Functions
    calculate_fouling_factor_from_u_values,
    calculate_cleanliness_trend,
    predict_fouling_rate,
    calculate_heat_loss_from_fouling,
    calculate_fuel_penalty,
    calculate_carbon_penalty,
    compare_cleaning_effectiveness,
    optimize_cleaning_interval,
    clear_calculation_cache,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_measurements() -> List[FoulingMeasurement]:
    """Create sample fouling measurements for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    measurements = []

    # Simulate fouling progression over 30 days
    for day in range(31):
        # U-value degrades linearly
        u_clean = 10.0
        degradation_factor = 1 - (day * 0.005)  # 0.5% per day
        u_current = u_clean * degradation_factor

        measurements.append(FoulingMeasurement(
            timestamp=base_time + timedelta(days=day),
            u_value_current=max(u_current, 7.0),  # Don't go below 7.0
            u_value_clean=u_clean,
            gas_inlet_temp_f=500.0 + day * 0.5,  # Slight temperature increase
            gas_outlet_temp_f=300.0 + day * 0.8,
            water_inlet_temp_f=200.0,
            water_outlet_temp_f=280.0,
            heat_duty_mmbtu_hr=50.0 - day * 0.2
        ))

    return measurements


@pytest.fixture
def clean_economizer() -> dict:
    """Clean economizer parameters."""
    return {
        "u_clean": 10.0,
        "u_current": 10.0,
        "heat_transfer_area_ft2": 5000.0,
        "lmtd_f": 150.0,
        "design_duty_mmbtu_hr": 75.0,
    }


@pytest.fixture
def fouled_economizer() -> dict:
    """Moderately fouled economizer parameters."""
    return {
        "u_clean": 10.0,
        "u_current": 8.5,
        "heat_transfer_area_ft2": 5000.0,
        "lmtd_f": 150.0,
        "design_duty_mmbtu_hr": 75.0,
    }


@pytest.fixture
def heavily_fouled_economizer() -> dict:
    """Heavily fouled economizer parameters."""
    return {
        "u_clean": 10.0,
        "u_current": 6.0,
        "heat_transfer_area_ft2": 5000.0,
        "lmtd_f": 150.0,
        "design_duty_mmbtu_hr": 75.0,
    }


# =============================================================================
# TEST: FOULING FACTOR CALCULATION
# =============================================================================

class TestFoulingFactorCalculation:
    """Tests for calculate_fouling_factor_from_u_values function."""

    def test_basic_fouling_factor_calculation(self):
        """Test basic fouling factor calculation."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0,
            gas_side_fraction=0.70
        )

        assert isinstance(result, FoulingFactorResult)
        assert result.rf_total > 0
        assert result.rf_gas_side > 0
        assert result.rf_water_side > 0
        assert result.cleanliness_factor == Decimal("85.00")

    def test_fouling_factor_partitioning(self):
        """Test that gas and water side fouling sum to total."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.0,
            u_clean=10.0,
            gas_side_fraction=0.70
        )

        # Gas + water should equal total
        total = result.rf_gas_side + result.rf_water_side
        assert abs(float(total - result.rf_total)) < 1e-9

    def test_clean_condition_zero_fouling(self):
        """Test that clean condition yields zero fouling factor."""
        result = calculate_fouling_factor_from_u_values(
            u_current=10.0,
            u_clean=10.0,
            gas_side_fraction=0.70
        )

        assert result.rf_total == Decimal("0.000000")
        assert result.cleanliness_factor == Decimal("100.00")
        assert result.severity_level == "clean"

    def test_severity_level_classification(self):
        """Test severity level classification based on fouling factor."""
        # Light fouling
        result_light = calculate_fouling_factor_from_u_values(
            u_current=9.5,
            u_clean=10.0
        )
        assert result_light.severity_level in ["clean", "light"]

        # Moderate fouling (Rf ~ 0.003)
        result_moderate = calculate_fouling_factor_from_u_values(
            u_current=7.5,
            u_clean=10.0
        )
        # Rf = 1/7.5 - 1/10 = 0.0333 which is moderate/heavy

        # Severe fouling
        result_severe = calculate_fouling_factor_from_u_values(
            u_current=5.0,
            u_clean=10.0
        )
        assert result_severe.severity_level in ["severe", "critical"]

    def test_provenance_hash_uniqueness(self):
        """Test that different inputs produce different provenance hashes."""
        result1 = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0
        )
        result2 = calculate_fouling_factor_from_u_values(
            u_current=8.0,
            u_clean=10.0
        )

        assert result1.provenance_hash != result2.provenance_hash

    def test_provenance_tracking(self):
        """Test provenance tracking returns complete record."""
        result, provenance = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0,
            track_provenance=True
        )

        assert provenance is not None
        assert provenance.provenance_hash != ""
        assert len(provenance.steps) > 0
        assert provenance.verify_integrity()

    def test_invalid_u_current_negative(self):
        """Test that negative U-value raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            calculate_fouling_factor_from_u_values(
                u_current=-1.0,
                u_clean=10.0
            )

    def test_invalid_u_current_exceeds_clean(self):
        """Test that U_current > U_clean raises error."""
        with pytest.raises(ValueError, match="cannot exceed"):
            calculate_fouling_factor_from_u_values(
                u_current=11.0,
                u_clean=10.0
            )

    def test_invalid_gas_side_fraction(self):
        """Test that invalid gas side fraction raises error."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_fouling_factor_from_u_values(
                u_current=8.5,
                u_clean=10.0,
                gas_side_fraction=1.5
            )

    def test_decimal_precision(self):
        """Test that output maintains proper Decimal precision."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0
        )

        # Check that Decimal has 6 decimal places
        assert isinstance(result.rf_total, Decimal)
        # The result should be quantized to 6 decimal places
        str_value = str(result.rf_total)
        if '.' in str_value:
            decimal_places = len(str_value.split('.')[1])
            assert decimal_places == 6


# =============================================================================
# TEST: CLEANLINESS TREND ANALYSIS
# =============================================================================

class TestCleanlinessTrend:
    """Tests for calculate_cleanliness_trend function."""

    def test_basic_trend_calculation(self, sample_measurements):
        """Test basic cleanliness trend calculation."""
        result = calculate_cleanliness_trend(
            measurements=sample_measurements,
            trend_window_days=30
        )

        assert "current_cf" in result
        assert "cf_trend_per_day" in result
        assert "days_to_threshold" in result
        assert "statistical_metrics" in result
        assert "provenance_hash" in result

    def test_degrading_trend_detection(self, sample_measurements):
        """Test detection of degrading cleanliness trend."""
        result = calculate_cleanliness_trend(
            measurements=sample_measurements
        )

        # Our sample data simulates degradation
        assert result["cf_trend_per_day"] < 0
        assert result["trend_direction"] == "degrading"

    def test_statistical_metrics(self, sample_measurements):
        """Test statistical metrics in trend analysis."""
        result = calculate_cleanliness_trend(
            measurements=sample_measurements
        )

        stats = result["statistical_metrics"]
        assert "mean" in stats
        assert "std_dev" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["n_samples"] == len(sample_measurements)

    def test_insufficient_data_error(self):
        """Test that single measurement raises error."""
        single_measurement = [FoulingMeasurement(
            timestamp=datetime.now(timezone.utc),
            u_value_current=8.5,
            u_value_clean=10.0,
            gas_inlet_temp_f=500.0,
            gas_outlet_temp_f=300.0,
            water_inlet_temp_f=200.0,
            water_outlet_temp_f=280.0
        )]

        with pytest.raises(ValueError, match="At least 2"):
            calculate_cleanliness_trend(single_measurement)

    def test_provenance_tracking_trend(self, sample_measurements):
        """Test provenance tracking in trend analysis."""
        result, provenance = calculate_cleanliness_trend(
            measurements=sample_measurements,
            track_provenance=True
        )

        assert provenance is not None
        assert provenance.verify_integrity()


# =============================================================================
# TEST: FOULING RATE PREDICTION
# =============================================================================

class TestFoulingRatePrediction:
    """Tests for predict_fouling_rate function."""

    def test_linear_model_prediction(self, sample_measurements):
        """Test fouling rate prediction with linear model."""
        result = predict_fouling_rate(
            measurements=sample_measurements,
            model=TrendModel.LINEAR
        )

        assert isinstance(result, FoulingRateResult)
        assert result.rate_model == "linear"
        assert result.fouling_rate >= 0  # Expect positive rate for degrading data

    def test_asymptotic_model_prediction(self, sample_measurements):
        """Test fouling rate prediction with asymptotic model."""
        result = predict_fouling_rate(
            measurements=sample_measurements,
            model=TrendModel.ASYMPTOTIC
        )

        assert result.rate_model == "asymptotic"

    def test_falling_rate_model(self, sample_measurements):
        """Test fouling rate prediction with falling rate model."""
        result = predict_fouling_rate(
            measurements=sample_measurements,
            model=TrendModel.FALLING_RATE
        )

        assert result.rate_model == "falling_rate"

    def test_power_law_model(self, sample_measurements):
        """Test fouling rate prediction with power law model."""
        result = predict_fouling_rate(
            measurements=sample_measurements,
            model=TrendModel.POWER_LAW
        )

        assert result.rate_model == "power_law"

    def test_time_to_threshold_calculation(self, sample_measurements):
        """Test time to threshold calculation."""
        result = predict_fouling_rate(
            measurements=sample_measurements,
            threshold_rf=0.005
        )

        # Should have a finite time to threshold or already past
        assert result.time_to_threshold_hours >= 0

    def test_confidence_level(self, sample_measurements):
        """Test that confidence level is calculated."""
        result = predict_fouling_rate(
            measurements=sample_measurements
        )

        assert 0 <= float(result.confidence_level) <= 1


# =============================================================================
# TEST: HEAT LOSS CALCULATION
# =============================================================================

class TestHeatLossCalculation:
    """Tests for calculate_heat_loss_from_fouling function."""

    def test_basic_heat_loss(self, fouled_economizer):
        """Test basic heat loss calculation."""
        result = calculate_heat_loss_from_fouling(
            u_current=fouled_economizer["u_current"],
            u_clean=fouled_economizer["u_clean"],
            heat_transfer_area_ft2=fouled_economizer["heat_transfer_area_ft2"],
            lmtd_f=fouled_economizer["lmtd_f"],
            design_duty_mmbtu_hr=fouled_economizer["design_duty_mmbtu_hr"]
        )

        assert isinstance(result, HeatLossResult)
        assert result.heat_loss_mmbtu_hr > 0
        assert result.heat_loss_percent > 0
        assert result.u_value_degradation_percent > 0

    def test_zero_heat_loss_clean(self, clean_economizer):
        """Test zero heat loss for clean economizer."""
        result = calculate_heat_loss_from_fouling(
            u_current=clean_economizer["u_current"],
            u_clean=clean_economizer["u_clean"],
            heat_transfer_area_ft2=clean_economizer["heat_transfer_area_ft2"],
            lmtd_f=clean_economizer["lmtd_f"],
            design_duty_mmbtu_hr=clean_economizer["design_duty_mmbtu_hr"]
        )

        assert result.heat_loss_mmbtu_hr == Decimal("0.0000")

    def test_temperature_penalty(self, heavily_fouled_economizer):
        """Test temperature penalty calculation for heavy fouling."""
        result = calculate_heat_loss_from_fouling(
            u_current=heavily_fouled_economizer["u_current"],
            u_clean=heavily_fouled_economizer["u_clean"],
            heat_transfer_area_ft2=heavily_fouled_economizer["heat_transfer_area_ft2"],
            lmtd_f=heavily_fouled_economizer["lmtd_f"],
            design_duty_mmbtu_hr=heavily_fouled_economizer["design_duty_mmbtu_hr"]
        )

        assert result.temperature_penalty_f > 0

    def test_provenance_tracking_heat_loss(self, fouled_economizer):
        """Test provenance tracking in heat loss calculation."""
        result, provenance = calculate_heat_loss_from_fouling(
            u_current=fouled_economizer["u_current"],
            u_clean=fouled_economizer["u_clean"],
            heat_transfer_area_ft2=fouled_economizer["heat_transfer_area_ft2"],
            lmtd_f=fouled_economizer["lmtd_f"],
            design_duty_mmbtu_hr=fouled_economizer["design_duty_mmbtu_hr"],
            track_provenance=True
        )

        assert provenance is not None
        assert provenance.verify_integrity()


# =============================================================================
# TEST: FUEL PENALTY CALCULATION
# =============================================================================

class TestFuelPenaltyCalculation:
    """Tests for calculate_fuel_penalty function."""

    def test_basic_fuel_penalty(self):
        """Test basic fuel penalty calculation."""
        result = calculate_fuel_penalty(
            heat_loss_mmbtu_hr=2.5,
            boiler_efficiency=0.85,
            fuel_cost_per_mmbtu=5.0,
            operating_hours_per_year=8000.0
        )

        assert isinstance(result, FuelPenaltyResult)
        assert result.fuel_penalty_mmbtu_hr > 0
        assert result.cost_per_hour > 0
        assert result.cost_per_year > 0

    def test_zero_heat_loss_zero_penalty(self):
        """Test zero fuel penalty for zero heat loss."""
        result = calculate_fuel_penalty(
            heat_loss_mmbtu_hr=0.0,
            boiler_efficiency=0.85,
            fuel_cost_per_mmbtu=5.0
        )

        assert result.fuel_penalty_mmbtu_hr == Decimal("0.0000")
        assert result.cost_per_hour == Decimal("0.00")

    def test_annual_cost_calculation(self):
        """Test annual cost calculation accuracy."""
        result = calculate_fuel_penalty(
            heat_loss_mmbtu_hr=1.0,
            boiler_efficiency=0.90,
            fuel_cost_per_mmbtu=10.0,
            operating_hours_per_year=8000.0
        )

        # Manual calculation:
        # fuel_penalty = 1.0 / 0.90 = 1.111 MMBtu/hr
        # cost_per_hour = 1.111 * 10.0 = 11.11 $/hr
        # annual = 11.11 * 8000 = 88,888.89 $/yr
        expected_annual = (1.0 / 0.90) * 10.0 * 8000

        assert abs(float(result.cost_per_year) - expected_annual) < 1.0

    def test_different_fuel_types(self):
        """Test with different fuel types."""
        for fuel_type in ["natural_gas", "fuel_oil_no2", "coal_bituminous"]:
            result = calculate_fuel_penalty(
                heat_loss_mmbtu_hr=1.0,
                boiler_efficiency=0.85,
                fuel_cost_per_mmbtu=5.0,
                fuel_type=fuel_type
            )
            assert result.fuel_type == fuel_type


# =============================================================================
# TEST: CARBON PENALTY CALCULATION
# =============================================================================

class TestCarbonPenaltyCalculation:
    """Tests for calculate_carbon_penalty function."""

    def test_basic_carbon_penalty(self):
        """Test basic carbon penalty calculation."""
        result = calculate_carbon_penalty(
            fuel_penalty_mmbtu_hr=2.0,
            operating_hours_per_year=8000.0,
            fuel_type="natural_gas",
            carbon_price_per_tonne=50.0
        )

        assert isinstance(result, CarbonPenaltyResult)
        assert result.co2_penalty_kg_hr > 0
        assert result.co2_penalty_tonnes_yr > 0
        assert result.carbon_cost_per_year > 0

    def test_zero_carbon_for_biomass(self):
        """Test zero carbon for biomass fuel (carbon neutral)."""
        result = calculate_carbon_penalty(
            fuel_penalty_mmbtu_hr=2.0,
            fuel_type="biomass",
            carbon_price_per_tonne=50.0
        )

        assert result.co2_penalty_kg_hr == Decimal("0.00")

    def test_coal_higher_than_gas(self):
        """Test that coal has higher CO2 than natural gas."""
        result_gas = calculate_carbon_penalty(
            fuel_penalty_mmbtu_hr=1.0,
            fuel_type="natural_gas"
        )
        result_coal = calculate_carbon_penalty(
            fuel_penalty_mmbtu_hr=1.0,
            fuel_type="coal_bituminous"
        )

        assert result_coal.co2_penalty_kg_hr > result_gas.co2_penalty_kg_hr


# =============================================================================
# TEST: CLEANING COMPARISON
# =============================================================================

class TestCleaningComparison:
    """Tests for compare_cleaning_effectiveness function."""

    def test_basic_cleaning_comparison(self):
        """Test basic cleaning effectiveness comparison."""
        result = compare_cleaning_effectiveness(
            u_before=7.5,
            u_after=9.0,
            u_clean=10.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0
        )

        assert isinstance(result, CleaningComparisonResult)
        assert result.rf_before > result.rf_after
        assert result.cleaning_effectiveness > 0
        assert result.u_value_recovery_percent > 0
        assert result.annual_savings > 0

    def test_perfect_cleaning_effectiveness(self):
        """Test 100% cleaning effectiveness."""
        result = compare_cleaning_effectiveness(
            u_before=7.5,
            u_after=10.0,  # Fully restored
            u_clean=10.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0
        )

        assert float(result.cleaning_effectiveness) == pytest.approx(1.0, abs=0.01)
        assert float(result.u_value_recovery_percent) == pytest.approx(100.0, abs=1.0)

    def test_cleaning_method_recorded(self):
        """Test that cleaning method is recorded."""
        result = compare_cleaning_effectiveness(
            u_before=7.5,
            u_after=9.0,
            u_clean=10.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            heat_transfer_area_ft2=5000.0,
            lmtd_f=150.0,
            cleaning_method=CleaningMethod.WATER_WASHING
        )

        assert result.cleaning_method == "water_washing"


# =============================================================================
# TEST: CLEANING INTERVAL OPTIMIZATION
# =============================================================================

class TestCleaningIntervalOptimization:
    """Tests for optimize_cleaning_interval function."""

    def test_basic_interval_optimization(self):
        """Test basic cleaning interval optimization."""
        result = optimize_cleaning_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert isinstance(result, CleaningIntervalResult)
        assert result.optimal_interval_hours > 0
        assert result.optimal_interval_days > 0
        assert result.cleaning_cycles_per_year > 0

    def test_higher_fouling_shorter_interval(self):
        """Test that higher fouling rate leads to shorter interval."""
        result_low = optimize_cleaning_interval(
            fouling_rate=1e-7,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )
        result_high = optimize_cleaning_interval(
            fouling_rate=1e-5,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        assert result_high.optimal_interval_hours < result_low.optimal_interval_hours

    def test_net_savings_vs_current(self):
        """Test net savings calculation against current interval."""
        result = optimize_cleaning_interval(
            fouling_rate=1e-6,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0,
            current_interval_hours=168.0  # Current: weekly
        )

        # Should calculate savings vs current interval
        assert result.net_savings_vs_current is not None

    def test_zero_fouling_maximum_interval(self):
        """Test that zero fouling rate gives maximum interval."""
        result = optimize_cleaning_interval(
            fouling_rate=0.0,
            cleaning_cost=50.0,
            fuel_cost_per_mmbtu=5.0,
            boiler_efficiency=0.85,
            boiler_heat_input_mmbtu_hr=100.0
        )

        # Should be at or near maximum practical interval
        assert float(result.optimal_interval_hours) >= 168.0  # At least 1 week


# =============================================================================
# TEST: CACHE FUNCTIONALITY
# =============================================================================

class TestCacheFunctionality:
    """Tests for cache clearing functionality."""

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        # Perform some calculations to populate cache
        calculate_fouling_factor_from_u_values(u_current=8.5, u_clean=10.0)
        calculate_fouling_factor_from_u_values(u_current=8.0, u_clean=10.0)

        # Clear cache
        cleared = clear_calculation_cache()

        # Should return 0 or more (depends on caching policy)
        assert cleared >= 0


# =============================================================================
# TEST: DATACLASS IMMUTABILITY
# =============================================================================

class TestDataclassImmutability:
    """Tests for frozen dataclass immutability."""

    def test_fouling_measurement_immutable(self):
        """Test that FoulingMeasurement is immutable."""
        measurement = FoulingMeasurement(
            timestamp=datetime.now(timezone.utc),
            u_value_current=8.5,
            u_value_clean=10.0,
            gas_inlet_temp_f=500.0,
            gas_outlet_temp_f=300.0,
            water_inlet_temp_f=200.0,
            water_outlet_temp_f=280.0
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            measurement.u_value_current = 9.0

    def test_fouling_factor_result_immutable(self):
        """Test that FoulingFactorResult is immutable."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0
        )

        with pytest.raises(Exception):
            result.rf_total = Decimal("0")


# =============================================================================
# TEST: CONSTANTS AND ENUMERATIONS
# =============================================================================

class TestConstantsAndEnumerations:
    """Tests for module constants and enumerations."""

    def test_tema_fouling_factors_complete(self):
        """Test that TEMA fouling factors are defined."""
        assert "boiler_feedwater_treated" in TEMA_FOULING_FACTORS
        assert "flue_gas_natural_gas" in TEMA_FOULING_FACTORS
        assert all(v > 0 for v in TEMA_FOULING_FACTORS.values())

    def test_fouling_severity_thresholds_ordered(self):
        """Test that severity thresholds are properly ordered."""
        thresholds = FOULING_SEVERITY_THRESHOLDS
        assert thresholds["clean"] < thresholds["light"]
        assert thresholds["light"] < thresholds["moderate"]
        assert thresholds["moderate"] < thresholds["heavy"]
        assert thresholds["heavy"] < thresholds["severe"]
        assert thresholds["severe"] < thresholds["critical"]

    def test_co2_emission_factors_defined(self):
        """Test that CO2 emission factors are defined."""
        assert "natural_gas" in CO2_EMISSION_FACTORS
        assert "coal_bituminous" in CO2_EMISSION_FACTORS
        assert CO2_EMISSION_FACTORS["coal_bituminous"] > CO2_EMISSION_FACTORS["natural_gas"]

    def test_fouling_side_enumeration(self):
        """Test FoulingSide enumeration values."""
        assert FoulingSide.GAS_SIDE.value == "gas_side"
        assert FoulingSide.WATER_SIDE.value == "water_side"
        assert FoulingSide.BOTH.value == "both"

    def test_cleaning_method_enumeration(self):
        """Test CleaningMethod enumeration values."""
        assert CleaningMethod.SOOT_BLOWING_STEAM.value == "soot_blowing_steam"
        assert CleaningMethod.WATER_WASHING.value == "water_washing"


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_fouling(self):
        """Test handling of very small fouling factors."""
        result = calculate_fouling_factor_from_u_values(
            u_current=9.999,
            u_clean=10.0
        )

        assert result.rf_total >= Decimal("0")
        assert result.severity_level == "clean"

    def test_maximum_fouling(self):
        """Test handling of maximum practical fouling."""
        result = calculate_fouling_factor_from_u_values(
            u_current=3.0,  # Very heavily fouled
            u_clean=10.0
        )

        assert result.rf_total > Decimal("0.01")
        assert result.severity_level == "critical"

    def test_equal_gas_water_fraction(self):
        """Test with equal gas and water side fractions."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.0,
            u_clean=10.0,
            gas_side_fraction=0.50
        )

        assert result.rf_gas_side == result.rf_water_side

    def test_all_gas_side_fouling(self):
        """Test with all fouling on gas side."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.0,
            u_clean=10.0,
            gas_side_fraction=1.0
        )

        assert result.rf_water_side == Decimal("0.000000")
        assert result.rf_gas_side == result.rf_total


# =============================================================================
# TEST: PROVENANCE HASH VERIFICATION
# =============================================================================

class TestProvenanceVerification:
    """Tests for provenance hash integrity."""

    def test_hash_reproducibility(self):
        """Test that same inputs produce same hash."""
        result1 = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0,
            gas_side_fraction=0.70
        )
        result2 = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0,
            gas_side_fraction=0.70
        )

        assert result1.provenance_hash == result2.provenance_hash

    def test_hash_is_sha256(self):
        """Test that hash is valid SHA-256 format."""
        result = calculate_fouling_factor_from_u_values(
            u_current=8.5,
            u_clean=10.0
        )

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
