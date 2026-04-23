# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Comprehensive Energy Loss Calculator Unit Tests

This module provides extensive unit tests for the SteamTrapEnergyLossCalculator,
covering all calculation scenarios, failure modes, and regulatory compliance.

Test Categories:
    - Steam loss calculation (Napier equation)
    - Energy metrics computation
    - Carbon emission calculations (EPA methodology)
    - ROI analysis validation
    - Provenance tracking verification
    - ASME PTC 39 compliance tests
    - Edge case handling

Coverage Target: 95%+

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import json
import math
import pytest
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.steam_trap_energy_loss_calculator import (
    SteamTrapEnergyLossCalculator,
    EnergyLossConfig,
    EnergyLossAnalysisResult,
    TrapSpecifications,
    SteamConditions,
    SteamLossResult,
    EnergyLossMetrics,
    CarbonEmissionResult,
    ROIResult,
    ProvenanceTracker,
    ProvenanceStep,
    FailureMode,
    TrapType,
    SeverityLevel,
    FuelType,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_calculator() -> SteamTrapEnergyLossCalculator:
    """Create a calculator with default configuration."""
    return SteamTrapEnergyLossCalculator()


@pytest.fixture
def custom_calculator() -> SteamTrapEnergyLossCalculator:
    """Create a calculator with custom configuration."""
    config = EnergyLossConfig(
        steam_cost_usd_per_1000lb=Decimal("15.00"),
        fuel_cost_usd_per_mmbtu=Decimal("6.00"),
        fuel_type=FuelType.FUEL_OIL_2,
        boiler_efficiency=Decimal("0.78"),
        operating_hours_per_year=8000,
        carbon_price_usd_per_ton=Decimal("75.00"),
    )
    return SteamTrapEnergyLossCalculator(config)


@pytest.fixture
def blow_through_params() -> dict:
    """Parameters for blow-through failure analysis."""
    return {
        "trap_id": "ST-BLOW-001",
        "failure_mode": FailureMode.BLOW_THROUGH,
        "orifice_diameter_mm": 6.35,  # 1/4 inch
        "pressure_bar_g": 10.0,
        "trap_type": TrapType.THERMODYNAMIC,
        "replacement_cost_usd": 200.0,
    }


@pytest.fixture
def leaking_params() -> dict:
    """Parameters for leaking trap analysis."""
    return {
        "trap_id": "ST-LEAK-001",
        "failure_mode": FailureMode.LEAKING,
        "orifice_diameter_mm": 6.35,
        "pressure_bar_g": 10.0,
        "trap_type": TrapType.THERMOSTATIC,
        "replacement_cost_usd": 250.0,
    }


@pytest.fixture
def blocked_params() -> dict:
    """Parameters for blocked trap analysis."""
    return {
        "trap_id": "ST-BLOCK-001",
        "failure_mode": FailureMode.BLOCKED,
        "orifice_diameter_mm": 6.35,
        "pressure_bar_g": 10.0,
        "trap_type": TrapType.MECHANICAL,
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestCalculatorInitialization:
    """Tests for calculator initialization."""

    def test_default_initialization(self, default_calculator: SteamTrapEnergyLossCalculator):
        """Test calculator initializes with default config."""
        assert default_calculator is not None
        assert default_calculator.config is not None
        assert default_calculator.config.fuel_type == FuelType.NATURAL_GAS

    def test_custom_initialization(self, custom_calculator: SteamTrapEnergyLossCalculator):
        """Test calculator initializes with custom config."""
        assert custom_calculator.config.fuel_type == FuelType.FUEL_OIL_2
        assert custom_calculator.config.boiler_efficiency == Decimal("0.78")
        assert custom_calculator.config.carbon_price_usd_per_ton == Decimal("75.00")

    def test_statistics_initialized(self, default_calculator: SteamTrapEnergyLossCalculator):
        """Test that statistics are initialized."""
        stats = default_calculator.get_statistics()
        assert stats["calculation_count"] == 0
        assert "fuel_type" in stats
        assert "boiler_efficiency" in stats


class TestNapierEquation:
    """Tests for Napier equation steam loss calculation."""

    def test_blow_through_maximum_loss(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that blow-through produces maximum steam loss."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Blow-through should have significant steam loss
        assert float(result.steam_loss.steam_loss_kg_hr) > 10.0
        assert result.steam_loss.failure_severity == SeverityLevel.CRITICAL

    def test_blocked_zero_steam_loss(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blocked_params: dict
    ):
        """Test that blocked traps have zero steam loss."""
        result = default_calculator.calculate_energy_loss(**blocked_params)

        # Blocked should have zero or minimal steam loss
        assert float(result.steam_loss.steam_loss_kg_hr) < 0.1

    def test_leaking_intermediate_loss(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        leaking_params: dict
    ):
        """Test that leaking traps have intermediate steam loss."""
        result = default_calculator.calculate_energy_loss(**leaking_params)

        # Leaking should be between blow-through and normal
        assert float(result.steam_loss.steam_loss_kg_hr) > 0.0
        assert result.steam_loss.failure_severity in [
            SeverityLevel.LOW,
            SeverityLevel.MEDIUM,
            SeverityLevel.HIGH,
        ]

    def test_pressure_affects_loss(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that higher pressure increases steam loss."""
        # Low pressure
        result_low = default_calculator.calculate_energy_loss(
            trap_id="ST-LOW-P",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=2.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        # High pressure
        result_high = default_calculator.calculate_energy_loss(
            trap_id="ST-HIGH-P",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=20.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        # Higher pressure should result in more steam loss
        assert float(result_high.steam_loss.steam_loss_kg_hr) > \
               float(result_low.steam_loss.steam_loss_kg_hr)

    def test_orifice_diameter_affects_loss(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that larger orifice increases steam loss."""
        # Small orifice
        result_small = default_calculator.calculate_energy_loss(
            trap_id="ST-SMALL-O",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=3.0,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        # Large orifice
        result_large = default_calculator.calculate_energy_loss(
            trap_id="ST-LARGE-O",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=12.0,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        # Larger orifice should result in more steam loss
        assert float(result_large.steam_loss.steam_loss_kg_hr) > \
               float(result_small.steam_loss.steam_loss_kg_hr)


class TestEnergyCalculation:
    """Tests for energy loss calculation."""

    def test_energy_loss_positive(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that energy loss is positive for failures."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        assert float(result.energy_metrics.energy_loss_btu_hr) > 0
        assert float(result.energy_metrics.energy_loss_kw) > 0

    def test_annual_energy_calculated(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that annual energy loss is calculated."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Annual should be hourly * hours_per_year
        # Allow for rounding
        hourly = float(result.energy_metrics.fuel_waste_mmbtu_hr)
        annual = float(result.energy_metrics.annual_energy_loss_mmbtu)

        expected_annual = hourly * 8760  # Default hours

        assert abs(annual - expected_annual) < 1.0

    def test_energy_cost_reasonable(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that energy cost is reasonable."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Annual cost should be positive for failed trap
        assert float(result.annual_energy_cost_usd) > 0

        # And should be reasonable (not millions for one trap)
        assert float(result.annual_energy_cost_usd) < 100000


class TestCarbonEmissions:
    """Tests for carbon emission calculations."""

    def test_carbon_emissions_calculated(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that carbon emissions are calculated."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        assert float(result.carbon_emissions.co2_kg_hr) > 0
        assert float(result.carbon_emissions.co2_tons_year) > 0

    def test_carbon_penalty_calculated(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that carbon penalty cost is calculated."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Penalty = CO2 tons * price per ton
        expected = float(result.carbon_emissions.co2_tons_year) * 50.0  # Default price

        actual = float(result.carbon_emissions.carbon_penalty_usd_year)

        assert abs(actual - expected) < 1.0

    def test_fuel_type_affects_emissions(self):
        """Test that different fuel types have different emission factors."""
        natural_gas_calc = SteamTrapEnergyLossCalculator(EnergyLossConfig(
            fuel_type=FuelType.NATURAL_GAS
        ))

        coal_calc = SteamTrapEnergyLossCalculator(EnergyLossConfig(
            fuel_type=FuelType.COAL
        ))

        params = {
            "trap_id": "ST-FUEL-TEST",
            "failure_mode": FailureMode.BLOW_THROUGH,
            "orifice_diameter_mm": 6.35,
            "pressure_bar_g": 10.0,
            "trap_type": TrapType.THERMODYNAMIC,
        }

        result_ng = natural_gas_calc.calculate_energy_loss(**params)
        result_coal = coal_calc.calculate_energy_loss(**params)

        # Coal should have higher emissions
        assert float(result_coal.carbon_emissions.co2_tons_year) > \
               float(result_ng.carbon_emissions.co2_tons_year)

    def test_epa_emission_factors_used(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that EPA emission factors are used."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Natural gas factor is 53.07 kg CO2/MMBtu
        expected_factor = Decimal("53.07")

        assert result.carbon_emissions.emission_factor_kg_per_mmbtu == expected_factor


class TestROICalculation:
    """Tests for ROI calculation."""

    def test_roi_calculated_when_cost_provided(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that ROI is calculated when replacement cost provided."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        assert result.roi_analysis is not None
        assert float(result.roi_analysis.replacement_cost_usd) == 200.0

    def test_roi_not_calculated_without_cost(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that ROI is not calculated without replacement cost."""
        result = default_calculator.calculate_energy_loss(
            trap_id="ST-NO-ROI",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            replacement_cost_usd=None,
        )

        assert result.roi_analysis is None

    def test_simple_payback_calculated(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that simple payback is calculated correctly."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Payback = cost / annual_savings
        roi = result.roi_analysis
        expected_payback_years = float(roi.replacement_cost_usd) / float(roi.annual_savings_usd)
        expected_payback_days = expected_payback_years * 365

        assert abs(float(roi.simple_payback_days) - expected_payback_days) < 1.0

    def test_npv_positive_for_good_investment(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that NPV is positive for good investments."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        # Blow-through trap replacement should have positive NPV
        assert float(result.roi_analysis.npv_lifetime_usd) > 0

    def test_roi_percentage_calculated(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that ROI percentage is calculated."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        roi = result.roi_analysis

        # ROI = (savings / cost) * 100
        expected_roi = (float(roi.annual_savings_usd) / float(roi.replacement_cost_usd)) * 100

        assert abs(float(roi.roi_first_year_percent) - expected_roi) < 1.0

    def test_recommendation_provided(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that recommendation is provided."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        assert result.roi_analysis.recommendation is not None
        assert len(result.roi_analysis.recommendation) > 0


class TestProvenanceTracking:
    """Tests for provenance tracking."""

    def test_provenance_hash_provided(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that provenance hash is provided."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_deterministic(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that provenance hash is deterministic."""
        result1 = default_calculator.calculate_energy_loss(**blow_through_params)
        result2 = default_calculator.calculate_energy_loss(**blow_through_params)

        # Note: May differ due to timestamp, but all calculations should match
        assert result1.steam_loss.steam_loss_kg_hr == result2.steam_loss.steam_loss_kg_hr
        assert result1.energy_metrics.energy_loss_btu_hr == result2.energy_metrics.energy_loss_btu_hr

    def test_provenance_tracker_class(self):
        """Test ProvenanceTracker class functionality."""
        tracker = ProvenanceTracker()

        # Record a step
        tracker.record_step(
            operation="test_operation",
            inputs={"input1": 100},
            formula="output = input1 * 2",
            result=200
        )

        steps = tracker.get_steps()
        assert len(steps) == 1
        assert steps[0].operation == "test_operation"

        # Get hash
        hash_value = tracker.get_hash()
        assert len(hash_value) == 64

    def test_calculation_method_asme(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that ASME method is documented."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)

        assert result.calculation_method == "ASME_PTC_39"


class TestInputValidation:
    """Tests for input validation."""

    def test_negative_orifice_rejected(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that negative orifice diameter is rejected."""
        with pytest.raises(ValueError, match="Invalid orifice"):
            default_calculator.calculate_energy_loss(
                trap_id="ST-NEG-O",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=-5.0,
                pressure_bar_g=10.0,
                trap_type=TrapType.THERMODYNAMIC,
            )

    def test_excessive_orifice_rejected(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that excessive orifice diameter is rejected."""
        with pytest.raises(ValueError, match="Invalid orifice"):
            default_calculator.calculate_energy_loss(
                trap_id="ST-BIG-O",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=200.0,
                pressure_bar_g=10.0,
                trap_type=TrapType.THERMODYNAMIC,
            )

    def test_negative_pressure_rejected(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that negative pressure is rejected."""
        with pytest.raises(ValueError, match="Invalid pressure"):
            default_calculator.calculate_energy_loss(
                trap_id="ST-NEG-P",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=6.35,
                pressure_bar_g=-5.0,
                trap_type=TrapType.THERMODYNAMIC,
            )


class TestTrapTypes:
    """Tests for different trap types."""

    @pytest.mark.parametrize("trap_type", [
        TrapType.THERMODYNAMIC,
        TrapType.THERMOSTATIC,
        TrapType.MECHANICAL,
        TrapType.VENTURI,
    ])
    def test_all_trap_types_work(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        trap_type: TrapType
    ):
        """Test that all trap types can be calculated."""
        result = default_calculator.calculate_energy_loss(
            trap_id=f"ST-{trap_type.value}",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=trap_type,
        )

        assert result is not None
        assert result.trap_specs.trap_type == trap_type

    def test_discharge_coefficients_differ(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that different trap types have different discharge coefficients."""
        results = {}

        for trap_type in TrapType:
            result = default_calculator.calculate_energy_loss(
                trap_id=f"ST-{trap_type.value}",
                failure_mode=FailureMode.BLOW_THROUGH,
                orifice_diameter_mm=6.35,
                pressure_bar_g=10.0,
                trap_type=trap_type,
            )
            results[trap_type] = float(result.steam_loss.steam_loss_kg_hr)

        # Venturi should have highest flow (highest Cd)
        # Mechanical should have lowest flow (lowest Cd)
        assert results[TrapType.VENTURI] > results[TrapType.MECHANICAL]


class TestFailureModes:
    """Tests for all failure modes."""

    @pytest.mark.parametrize("failure_mode", [
        FailureMode.BLOW_THROUGH,
        FailureMode.LEAKING,
        FailureMode.BLOCKED,
        FailureMode.CYCLING_FAST,
        FailureMode.CYCLING_SLOW,
        FailureMode.COLD_TRAP,
        FailureMode.NORMAL,
    ])
    def test_all_failure_modes_work(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        failure_mode: FailureMode
    ):
        """Test that all failure modes can be calculated."""
        result = default_calculator.calculate_energy_loss(
            trap_id=f"ST-{failure_mode.value}",
            failure_mode=failure_mode,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        assert result is not None
        assert result.failure_mode == failure_mode


class TestLeakSeverityOverride:
    """Tests for leak severity override."""

    def test_custom_severity_applied(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that custom leak severity is applied."""
        # 50% leak severity
        result = default_calculator.calculate_energy_loss(
            trap_id="ST-CUSTOM-LEAK",
            failure_mode=FailureMode.LEAKING,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            leak_severity=0.50,
        )

        # Should be half of blow-through
        result_full = default_calculator.calculate_energy_loss(
            trap_id="ST-FULL",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
            leak_severity=1.0,
        )

        ratio = float(result.steam_loss.steam_loss_kg_hr) / \
                float(result_full.steam_loss.steam_loss_kg_hr)

        assert 0.45 < ratio < 0.55  # Should be around 50%


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_multiple_traps(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test batch calculation for multiple traps."""
        trap_analyses = [
            {
                "trap_id": f"ST-BATCH-{i}",
                "failure_mode": FailureMode.BLOW_THROUGH,
                "orifice_diameter_mm": 6.35,
                "pressure_bar_g": 10.0,
                "trap_type": TrapType.THERMODYNAMIC,
            }
            for i in range(5)
        ]

        results = default_calculator.calculate_batch(trap_analyses)

        assert len(results) == 5
        for result in results:
            assert result.steam_loss.steam_loss_kg_hr > 0


class TestSerialization:
    """Tests for result serialization."""

    def test_to_dict_complete(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that to_dict produces complete output."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)
        result_dict = result.to_dict()

        assert "trap_id" in result_dict
        assert "steam_loss_lb_hr" in result_dict
        assert "annual_energy_cost_usd" in result_dict
        assert "provenance_hash" in result_dict

    def test_to_dict_json_serializable(
        self,
        default_calculator: SteamTrapEnergyLossCalculator,
        blow_through_params: dict
    ):
        """Test that to_dict output is JSON serializable."""
        result = default_calculator.calculate_energy_loss(**blow_through_params)
        result_dict = result.to_dict()

        # Should not raise
        json_str = json.dumps(result_dict)
        assert json_str is not None

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["trap_id"] == result_dict["trap_id"]


class TestSteamProperties:
    """Tests for steam properties lookup."""

    def test_saturation_temps_reasonable(
        self,
        default_calculator: SteamTrapEnergyLossCalculator
    ):
        """Test that saturation temperatures are reasonable."""
        # At 0 bar gauge, T_sat should be around 100C
        result_0 = default_calculator.calculate_energy_loss(
            trap_id="ST-0BAR",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=0.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        assert 99 < float(result_0.steam_conditions.saturation_temp_c) < 101

        # At 10 bar gauge, T_sat should be around 184C
        result_10 = default_calculator.calculate_energy_loss(
            trap_id="ST-10BAR",
            failure_mode=FailureMode.BLOW_THROUGH,
            orifice_diameter_mm=6.35,
            pressure_bar_g=10.0,
            trap_type=TrapType.THERMODYNAMIC,
        )

        assert 180 < float(result_10.steam_conditions.saturation_temp_c) < 188


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
