# -*- coding: utf-8 -*-
"""
Unit Tests: Vacuum Calculator

Comprehensive tests for vacuum/backpressure optimization calculations including:
- Optimal backpressure calculation
- Economic optimization (MW gain vs CW pump cost)
- Heat rate sensitivity analysis
- Achievable vacuum limits
- CW flow optimization

Standards Reference:
- EPRI Condenser Performance Guidelines
- ASME PTC 12.2

Target Coverage: 85%+
Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import (
    VacuumOptimizationInput,
    VacuumOptimizationResult,
    CondenserConfig,
    CondenserReading,
    TubeMaterial,
    WaterSource,
    AssertionHelpers,
    ProvenanceCalculator,
    saturation_temp_from_pressure,
    pressure_from_saturation_temp,
    OPERATING_LIMITS,
    TEST_SEED,
)


# =============================================================================
# VACUUM CALCULATOR IMPLEMENTATION FOR TESTING
# =============================================================================

class VacuumCalculator:
    """
    Vacuum/backpressure optimization calculator.

    Optimizes condenser vacuum for maximum economic benefit considering:
    - Turbine heat rate improvement from lower backpressure
    - CW pumping power cost
    - Achievable vacuum based on CW inlet temperature
    - Equipment limitations
    """

    VERSION = "1.0.0"

    # Heat rate sensitivity (kJ/kWh per kPa change)
    # Typical value: 45-55 kJ/kWh per kPa for modern units
    DEFAULT_HEAT_RATE_SENSITIVITY = 50.0  # kJ/kWh per kPa

    # Minimum approach temperature (ITD - TTD)
    MIN_APPROACH_TEMP_C = 5.0

    # CW pump affinity law exponent
    CW_PUMP_AFFINITY_EXPONENT = 2.5  # Power varies with flow^2.5

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize vacuum calculator."""
        self.config = config or {}
        self.heat_rate_sensitivity = self.config.get(
            "heat_rate_sensitivity",
            self.DEFAULT_HEAT_RATE_SENSITIVITY
        )
        self.min_backpressure = self.config.get("min_backpressure_kpa", 2.5)
        self.max_backpressure = self.config.get("max_backpressure_kpa", 15.0)

    def calculate_achievable_vacuum(
        self,
        cw_inlet_temp_c: float,
        approach_temp_c: float = 5.0
    ) -> float:
        """
        Calculate achievable vacuum based on CW inlet temperature.

        The minimum achievable pressure is limited by thermodynamics:
        T_sat >= T_cw_in + approach

        Args:
            cw_inlet_temp_c: CW inlet temperature (C)
            approach_temp_c: Minimum approach temperature (C)

        Returns:
            Minimum achievable backpressure (kPa absolute)
        """
        min_sat_temp = cw_inlet_temp_c + approach_temp_c
        min_pressure = pressure_from_saturation_temp(min_sat_temp)

        # Cannot go below equipment limit
        return max(min_pressure, self.min_backpressure)

    def calculate_heat_rate_impact(
        self,
        backpressure_change_kpa: float,
        unit_load_mw: float,
        base_heat_rate_kj_kwh: float
    ) -> Dict[str, float]:
        """
        Calculate heat rate impact from backpressure change.

        Args:
            backpressure_change_kpa: Change in backpressure (negative = improvement)
            unit_load_mw: Current unit load (MW)
            base_heat_rate_kj_kwh: Base heat rate (kJ/kWh)

        Returns:
            Dictionary with heat rate metrics
        """
        # Heat rate change
        heat_rate_change = backpressure_change_kpa * self.heat_rate_sensitivity

        # New heat rate
        new_heat_rate = base_heat_rate_kj_kwh + heat_rate_change

        # Fuel savings (approximate)
        # Lower heat rate -> less fuel for same output
        if base_heat_rate_kj_kwh > 0:
            efficiency_improvement = -heat_rate_change / base_heat_rate_kj_kwh
        else:
            efficiency_improvement = 0.0

        # MW equivalent (power that could be generated with saved fuel)
        mw_equivalent = unit_load_mw * efficiency_improvement

        return {
            "heat_rate_change_kj_kwh": heat_rate_change,
            "new_heat_rate_kj_kwh": new_heat_rate,
            "efficiency_improvement_pct": efficiency_improvement * 100,
            "mw_equivalent": mw_equivalent,
        }

    def calculate_mw_gain(
        self,
        current_backpressure_kpa: float,
        optimal_backpressure_kpa: float,
        unit_load_mw: float,
        base_heat_rate_kj_kwh: float
    ) -> float:
        """
        Calculate MW gain from backpressure improvement.

        Args:
            current_backpressure_kpa: Current backpressure
            optimal_backpressure_kpa: Optimal backpressure
            unit_load_mw: Unit load
            base_heat_rate_kj_kwh: Base heat rate

        Returns:
            MW gain (positive = improvement)
        """
        backpressure_change = optimal_backpressure_kpa - current_backpressure_kpa
        impact = self.calculate_heat_rate_impact(
            backpressure_change,
            unit_load_mw,
            base_heat_rate_kj_kwh
        )
        return -impact["mw_equivalent"]  # Negative change = positive gain

    def calculate_cw_pump_power(
        self,
        flow_m3_s: float,
        base_flow_m3_s: float,
        base_power_kw: float
    ) -> float:
        """
        Calculate CW pump power using affinity laws.

        Power varies with flow^2.5 (approximately)

        Args:
            flow_m3_s: Current flow rate
            base_flow_m3_s: Reference flow rate
            base_power_kw: Reference power

        Returns:
            Pump power (kW)
        """
        if base_flow_m3_s <= 0 or base_power_kw <= 0:
            return 0.0

        flow_ratio = flow_m3_s / base_flow_m3_s
        return base_power_kw * (flow_ratio ** self.CW_PUMP_AFFINITY_EXPONENT)

    def calculate_annual_value(
        self,
        mw_value: float,
        electricity_price_usd_mwh: float,
        capacity_factor: float = 0.85,
        hours_per_year: int = 8760
    ) -> float:
        """
        Calculate annual value of MW improvement.

        Args:
            mw_value: MW improvement
            electricity_price_usd_mwh: Electricity price ($/MWh)
            capacity_factor: Expected capacity factor
            hours_per_year: Hours in year

        Returns:
            Annual value (USD)
        """
        operating_hours = hours_per_year * capacity_factor
        return mw_value * electricity_price_usd_mwh * operating_hours

    def optimize_vacuum(
        self,
        inputs: VacuumOptimizationInput,
        design_cw_flow_m3_s: float = None,
        design_pump_power_kw: float = None
    ) -> VacuumOptimizationResult:
        """
        Optimize condenser vacuum for maximum economic benefit.

        Args:
            inputs: Vacuum optimization inputs
            design_cw_flow_m3_s: Design CW flow rate
            design_pump_power_kw: Design pump power

        Returns:
            VacuumOptimizationResult with recommendations
        """
        design_cw_flow = design_cw_flow_m3_s or inputs.cw_flow_m3_s
        design_pump_power = design_pump_power_kw or inputs.cw_pump_power_kw

        # Calculate achievable vacuum limit
        achievable_min = self.calculate_achievable_vacuum(inputs.cw_inlet_temp_c)

        # Search for optimal backpressure
        best_backpressure = inputs.current_backpressure_kpa
        best_net_benefit = 0.0

        # Try different backpressure levels
        for bp_trial in np.linspace(achievable_min, self.max_backpressure, 50):
            # Calculate MW gain from lower backpressure
            mw_gain = self.calculate_mw_gain(
                inputs.current_backpressure_kpa,
                bp_trial,
                inputs.unit_load_mw,
                inputs.heat_rate_kj_kwh
            )

            # Estimate flow change needed (simplified)
            # Lower backpressure may require more CW flow
            flow_factor = 1.0
            if bp_trial < inputs.current_backpressure_kpa:
                # Need more flow for lower backpressure
                bp_reduction = inputs.current_backpressure_kpa - bp_trial
                flow_factor = 1.0 + 0.05 * bp_reduction  # 5% more flow per kPa

            new_flow = inputs.cw_flow_m3_s * flow_factor
            new_pump_power = self.calculate_cw_pump_power(
                new_flow, design_cw_flow, design_pump_power
            )

            # Calculate net benefit
            pump_power_change = new_pump_power - inputs.cw_pump_power_kw
            pump_power_change_mw = pump_power_change / 1000

            net_mw_benefit = mw_gain - pump_power_change_mw

            if net_mw_benefit > best_net_benefit:
                best_net_benefit = net_mw_benefit
                best_backpressure = bp_trial
                best_cw_flow = new_flow

        # Calculate annual savings
        annual_savings = self.calculate_annual_value(
            best_net_benefit,
            inputs.electricity_price_usd_mwh
        )

        # Determine if economically optimal
        is_economic = best_backpressure < inputs.current_backpressure_kpa

        # Sensitivity analysis
        sensitivity = {
            "mw_per_kpa": self.calculate_mw_gain(
                inputs.current_backpressure_kpa,
                inputs.current_backpressure_kpa - 1.0,
                inputs.unit_load_mw,
                inputs.heat_rate_kj_kwh
            ),
            "achievable_min_kpa": achievable_min,
            "current_vs_achievable_kpa": inputs.current_backpressure_kpa - achievable_min,
        }

        # Generate provenance hash
        input_data = inputs.to_dict()
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        return VacuumOptimizationResult(
            optimal_backpressure_kpa=round(best_backpressure, 3),
            current_backpressure_kpa=inputs.current_backpressure_kpa,
            mw_gain_potential=round(best_net_benefit, 3),
            annual_savings_usd=round(annual_savings, 2),
            cw_flow_recommendation_m3_s=round(best_cw_flow, 2),
            economic_optimum=is_economic,
            sensitivity_analysis=sensitivity,
            provenance_hash=provenance_hash,
        )

    def calculate_vacuum_curves(
        self,
        cw_inlet_temps_c: List[float],
        unit_load_mw: float,
        base_heat_rate_kj_kwh: float
    ) -> Dict[float, Dict[str, float]]:
        """
        Generate vacuum performance curves for different CW inlet temps.

        Args:
            cw_inlet_temps_c: List of CW inlet temperatures
            unit_load_mw: Unit load
            base_heat_rate_kj_kwh: Base heat rate

        Returns:
            Dictionary mapping CW temp to performance metrics
        """
        curves = {}

        for cw_temp in cw_inlet_temps_c:
            achievable_bp = self.calculate_achievable_vacuum(cw_temp)
            sat_temp = saturation_temp_from_pressure(achievable_bp)

            # Calculate expected TTD and CW rise
            expected_ttd = 3.0  # Typical TTD for clean condenser
            expected_cw_out = sat_temp - expected_ttd
            expected_cw_rise = expected_cw_out - cw_temp

            curves[cw_temp] = {
                "achievable_backpressure_kpa": achievable_bp,
                "saturation_temp_c": sat_temp,
                "expected_cw_outlet_c": expected_cw_out,
                "expected_cw_rise_c": expected_cw_rise,
            }

        return curves


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator() -> VacuumCalculator:
    """Create vacuum calculator instance."""
    return VacuumCalculator()


@pytest.fixture
def calculator_with_config() -> VacuumCalculator:
    """Create vacuum calculator with custom config."""
    config = {
        "heat_rate_sensitivity": 45.0,
        "min_backpressure_kpa": 2.0,
        "max_backpressure_kpa": 12.0,
    }
    return VacuumCalculator(config)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestAchievableVacuum:
    """Tests for achievable vacuum calculation."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_achievable_vacuum_cold_water(self, calculator: VacuumCalculator):
        """Test achievable vacuum with cold CW inlet."""
        cw_inlet = 15.0  # Cold inlet
        achievable = calculator.calculate_achievable_vacuum(cw_inlet)

        # With 15C inlet and 5C approach, min sat temp = 20C
        # This gives approximately 2.3 kPa
        assert achievable > 0
        assert achievable < 5.0  # Should achieve good vacuum

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_achievable_vacuum_warm_water(self, calculator: VacuumCalculator):
        """Test achievable vacuum with warm CW inlet."""
        cw_inlet = 30.0  # Warm inlet
        achievable = calculator.calculate_achievable_vacuum(cw_inlet)

        # With 30C inlet, vacuum will be worse
        assert achievable > 3.0  # Minimum achievable will be higher

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_achievable_vacuum_hot_water(self, calculator: VacuumCalculator):
        """Test achievable vacuum with hot CW inlet."""
        cw_inlet = 38.0  # Hot summer day
        achievable = calculator.calculate_achievable_vacuum(cw_inlet)

        # Should still return valid value
        assert achievable > 5.0
        assert achievable < 15.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_achievable_vacuum_respects_minimum(self, calculator: VacuumCalculator):
        """Test achievable vacuum respects equipment minimum."""
        cw_inlet = 5.0  # Very cold

        # Even with very cold water, can't go below equipment limit
        achievable = calculator.calculate_achievable_vacuum(cw_inlet)
        assert achievable >= calculator.min_backpressure

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.parametrize("cw_inlet,approach,expected_min", [
        (20.0, 5.0, 2.5),   # Typical - limited by equipment
        (25.0, 5.0, 3.0),   # Moderate
        (30.0, 5.0, 4.5),   # Warm
        (35.0, 5.0, 6.0),   # Hot
    ])
    def test_achievable_vacuum_parametric(
        self,
        calculator: VacuumCalculator,
        cw_inlet, approach, expected_min
    ):
        """Test achievable vacuum with parametric values."""
        achievable = calculator.calculate_achievable_vacuum(cw_inlet, approach)

        # Should be at least the expected minimum
        assert achievable >= expected_min - 0.5


class TestHeatRateImpact:
    """Tests for heat rate impact calculations."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_heat_rate_improvement(self, calculator: VacuumCalculator):
        """Test heat rate improvement from lower backpressure."""
        result = calculator.calculate_heat_rate_impact(
            backpressure_change_kpa=-1.0,  # 1 kPa reduction
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        # Lower backpressure -> lower heat rate
        assert result["heat_rate_change_kj_kwh"] < 0
        assert result["efficiency_improvement_pct"] > 0
        assert result["mw_equivalent"] > 0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_heat_rate_degradation(self, calculator: VacuumCalculator):
        """Test heat rate degradation from higher backpressure."""
        result = calculator.calculate_heat_rate_impact(
            backpressure_change_kpa=1.0,  # 1 kPa increase
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        # Higher backpressure -> higher heat rate
        assert result["heat_rate_change_kj_kwh"] > 0
        assert result["efficiency_improvement_pct"] < 0
        assert result["mw_equivalent"] < 0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_heat_rate_zero_change(self, calculator: VacuumCalculator):
        """Test heat rate with no backpressure change."""
        result = calculator.calculate_heat_rate_impact(
            backpressure_change_kpa=0.0,
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        assert result["heat_rate_change_kj_kwh"] == 0.0
        assert result["efficiency_improvement_pct"] == 0.0
        assert result["mw_equivalent"] == 0.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_heat_rate_sensitivity_configurable(self, calculator_with_config: VacuumCalculator):
        """Test heat rate sensitivity is configurable."""
        result = calculator_with_config.calculate_heat_rate_impact(
            backpressure_change_kpa=-1.0,
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        # With sensitivity of 45 kJ/kWh per kPa
        expected_change = -45.0
        assert result["heat_rate_change_kj_kwh"] == expected_change

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.parametrize("bp_change,expected_sign", [
        (-2.0, 1),   # Improvement
        (-1.0, 1),   # Improvement
        (-0.5, 1),   # Small improvement
        (0.0, 0),    # No change
        (0.5, -1),   # Small degradation
        (1.0, -1),   # Degradation
        (2.0, -1),   # Major degradation
    ])
    def test_heat_rate_change_direction(
        self,
        calculator: VacuumCalculator,
        bp_change, expected_sign
    ):
        """Test heat rate change has correct sign."""
        result = calculator.calculate_heat_rate_impact(
            bp_change, 500.0, 9500.0
        )

        if expected_sign > 0:
            assert result["mw_equivalent"] > 0
        elif expected_sign < 0:
            assert result["mw_equivalent"] < 0
        else:
            assert result["mw_equivalent"] == 0


class TestMWGainCalculation:
    """Tests for MW gain calculation."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_mw_gain_from_improvement(self, calculator: VacuumCalculator):
        """Test MW gain from backpressure improvement."""
        mw_gain = calculator.calculate_mw_gain(
            current_backpressure_kpa=6.0,
            optimal_backpressure_kpa=4.0,  # 2 kPa improvement
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        assert mw_gain > 0  # Should have positive gain

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_mw_gain_zero_change(self, calculator: VacuumCalculator):
        """Test MW gain with no change."""
        mw_gain = calculator.calculate_mw_gain(
            current_backpressure_kpa=5.0,
            optimal_backpressure_kpa=5.0,
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        assert mw_gain == 0.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_mw_loss_from_degradation(self, calculator: VacuumCalculator):
        """Test MW loss from backpressure degradation."""
        mw_gain = calculator.calculate_mw_gain(
            current_backpressure_kpa=5.0,
            optimal_backpressure_kpa=7.0,  # Worse backpressure
            unit_load_mw=500.0,
            base_heat_rate_kj_kwh=9500.0
        )

        assert mw_gain < 0  # Should show loss

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_mw_gain_scales_with_load(self, calculator: VacuumCalculator):
        """Test MW gain scales with unit load."""
        gain_low_load = calculator.calculate_mw_gain(5.0, 4.0, 250.0, 9500.0)
        gain_high_load = calculator.calculate_mw_gain(5.0, 4.0, 500.0, 9500.0)

        # Double the load -> approximately double the gain
        assert abs(gain_high_load / gain_low_load - 2.0) < 0.1


class TestCWPumpPower:
    """Tests for CW pump power calculations."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_pump_power_at_design(self, calculator: VacuumCalculator):
        """Test pump power at design flow."""
        power = calculator.calculate_cw_pump_power(
            flow_m3_s=15.0,
            base_flow_m3_s=15.0,
            base_power_kw=2000.0
        )

        assert power == 2000.0  # At design, power equals base

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_pump_power_above_design(self, calculator: VacuumCalculator):
        """Test pump power above design flow."""
        power = calculator.calculate_cw_pump_power(
            flow_m3_s=18.0,  # 20% above design
            base_flow_m3_s=15.0,
            base_power_kw=2000.0
        )

        # Power increases faster than flow (affinity law)
        assert power > 2000.0 * 1.2  # More than 20% increase

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_pump_power_below_design(self, calculator: VacuumCalculator):
        """Test pump power below design flow."""
        power = calculator.calculate_cw_pump_power(
            flow_m3_s=12.0,  # 20% below design
            base_flow_m3_s=15.0,
            base_power_kw=2000.0
        )

        # Power decreases faster than flow
        assert power < 2000.0 * 0.8

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_pump_power_zero_flow(self, calculator: VacuumCalculator):
        """Test pump power with zero flow."""
        power = calculator.calculate_cw_pump_power(
            flow_m3_s=0.0,
            base_flow_m3_s=15.0,
            base_power_kw=2000.0
        )

        assert power == 0.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_pump_power_invalid_base_flow(self, calculator: VacuumCalculator):
        """Test pump power with invalid base flow."""
        power = calculator.calculate_cw_pump_power(
            flow_m3_s=15.0,
            base_flow_m3_s=0.0,
            base_power_kw=2000.0
        )

        assert power == 0.0


class TestAnnualValue:
    """Tests for annual value calculations."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_annual_value_calculation(self, calculator: VacuumCalculator):
        """Test annual value calculation."""
        value = calculator.calculate_annual_value(
            mw_value=1.0,
            electricity_price_usd_mwh=50.0,
            capacity_factor=0.85
        )

        # 1 MW * $50/MWh * 8760 hours * 0.85 = $372,300
        expected = 1.0 * 50.0 * 8760 * 0.85
        assert abs(value - expected) < 1.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_annual_value_zero_mw(self, calculator: VacuumCalculator):
        """Test annual value with zero MW."""
        value = calculator.calculate_annual_value(0.0, 50.0)
        assert value == 0.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_annual_value_high_price(self, calculator: VacuumCalculator):
        """Test annual value with high electricity price."""
        value_low = calculator.calculate_annual_value(1.0, 50.0)
        value_high = calculator.calculate_annual_value(1.0, 100.0)

        assert value_high == 2 * value_low

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.parametrize("cf,expected_hours", [
        (1.0, 8760),
        (0.85, 7446),
        (0.50, 4380),
        (0.0, 0),
    ])
    def test_annual_value_capacity_factor(
        self,
        calculator: VacuumCalculator,
        cf, expected_hours
    ):
        """Test annual value with different capacity factors."""
        value = calculator.calculate_annual_value(1.0, 50.0, capacity_factor=cf)
        expected = 1.0 * 50.0 * expected_hours
        assert abs(value - expected) < 1.0


class TestVacuumOptimization:
    """Tests for vacuum optimization."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_optimization_basic(
        self,
        calculator: VacuumCalculator,
        vacuum_optimization_input: VacuumOptimizationInput
    ):
        """Test basic vacuum optimization."""
        result = calculator.optimize_vacuum(vacuum_optimization_input)

        assert isinstance(result, VacuumOptimizationResult)
        assert result.optimal_backpressure_kpa > 0
        assert result.optimal_backpressure_kpa <= calculator.max_backpressure

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_optimization_returns_provenance(
        self,
        calculator: VacuumCalculator,
        vacuum_optimization_input: VacuumOptimizationInput
    ):
        """Test optimization returns valid provenance hash."""
        result = calculator.optimize_vacuum(vacuum_optimization_input)

        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # Valid hex

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_optimization_includes_sensitivity(
        self,
        calculator: VacuumCalculator,
        vacuum_optimization_input: VacuumOptimizationInput
    ):
        """Test optimization includes sensitivity analysis."""
        result = calculator.optimize_vacuum(vacuum_optimization_input)

        assert "mw_per_kpa" in result.sensitivity_analysis
        assert "achievable_min_kpa" in result.sensitivity_analysis
        assert "current_vs_achievable_kpa" in result.sensitivity_analysis

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_optimization_already_optimal(self, calculator: VacuumCalculator):
        """Test optimization when already at optimum."""
        inputs = VacuumOptimizationInput(
            current_backpressure_kpa=3.0,  # Already very good
            unit_load_mw=500.0,
            cw_inlet_temp_c=20.0,
            cw_flow_m3_s=15.0,
            cw_pump_power_kw=2000.0,
            heat_rate_kj_kwh=9500.0,
            electricity_price_usd_mwh=50.0,
        )

        result = calculator.optimize_vacuum(inputs)

        # Should show minimal improvement potential
        assert result.mw_gain_potential < 1.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_optimization_degraded_condenser(self, calculator: VacuumCalculator):
        """Test optimization with degraded condenser."""
        inputs = VacuumOptimizationInput(
            current_backpressure_kpa=10.0,  # High backpressure
            unit_load_mw=500.0,
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=12.0,  # Low flow
            cw_pump_power_kw=1500.0,
            heat_rate_kj_kwh=10000.0,  # Degraded heat rate
            electricity_price_usd_mwh=50.0,
        )

        result = calculator.optimize_vacuum(inputs)

        # Should show significant improvement potential
        assert result.mw_gain_potential > 0
        assert result.optimal_backpressure_kpa < inputs.current_backpressure_kpa

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_optimization_high_price_sensitivity(self, calculator: VacuumCalculator):
        """Test that higher prices justify more CW pumping."""
        low_price = VacuumOptimizationInput(
            current_backpressure_kpa=6.0,
            unit_load_mw=500.0,
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=15.0,
            cw_pump_power_kw=2000.0,
            heat_rate_kj_kwh=9500.0,
            electricity_price_usd_mwh=30.0,  # Low price
        )

        high_price = VacuumOptimizationInput(
            current_backpressure_kpa=6.0,
            unit_load_mw=500.0,
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=15.0,
            cw_pump_power_kw=2000.0,
            heat_rate_kj_kwh=9500.0,
            electricity_price_usd_mwh=100.0,  # High price
        )

        result_low = calculator.optimize_vacuum(low_price)
        result_high = calculator.optimize_vacuum(high_price)

        # Higher price should show higher annual savings
        assert result_high.annual_savings_usd > result_low.annual_savings_usd


class TestVacuumCurves:
    """Tests for vacuum performance curve generation."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_vacuum_curves_generation(self, calculator: VacuumCalculator):
        """Test vacuum curves are generated for all temps."""
        temps = [15.0, 20.0, 25.0, 30.0, 35.0]

        curves = calculator.calculate_vacuum_curves(
            temps, unit_load_mw=500.0, base_heat_rate_kj_kwh=9500.0
        )

        assert len(curves) == len(temps)
        for temp in temps:
            assert temp in curves
            assert "achievable_backpressure_kpa" in curves[temp]
            assert "saturation_temp_c" in curves[temp]

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_vacuum_curves_monotonic(self, calculator: VacuumCalculator):
        """Test vacuum curves show worse vacuum with higher inlet temp."""
        temps = [15.0, 20.0, 25.0, 30.0, 35.0]

        curves = calculator.calculate_vacuum_curves(temps, 500.0, 9500.0)

        # Backpressure should increase with CW inlet temperature
        prev_bp = 0.0
        for temp in temps:
            bp = curves[temp]["achievable_backpressure_kpa"]
            assert bp >= prev_bp
            prev_bp = bp

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_vacuum_curves_ttd_reasonable(self, calculator: VacuumCalculator):
        """Test vacuum curves have reasonable temperature differences."""
        temps = [20.0, 25.0, 30.0]

        curves = calculator.calculate_vacuum_curves(temps, 500.0, 9500.0)

        for temp in temps:
            sat_temp = curves[temp]["saturation_temp_c"]
            cw_out = curves[temp]["expected_cw_outlet_c"]

            # TTD should be positive
            ttd = sat_temp - cw_out
            assert ttd > 0

            # CW rise should be reasonable (5-20C typically)
            cw_rise = curves[temp]["expected_cw_rise_c"]
            assert 3.0 <= cw_rise <= 25.0


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.golden
    def test_optimization_is_deterministic(
        self,
        calculator: VacuumCalculator,
        vacuum_optimization_input: VacuumOptimizationInput
    ):
        """Test optimization produces same result repeatedly."""
        results = [
            calculator.optimize_vacuum(vacuum_optimization_input)
            for _ in range(10)
        ]

        # All optimal backpressures should be identical
        bp_values = [r.optimal_backpressure_kpa for r in results]
        assert len(set(bp_values)) == 1

        # All hashes should be identical
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 1

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.golden
    def test_heat_rate_impact_is_deterministic(self, calculator: VacuumCalculator):
        """Test heat rate impact is deterministic."""
        results = [
            calculator.calculate_heat_rate_impact(-1.0, 500.0, 9500.0)
            for _ in range(100)
        ]

        # All results should be identical
        mw_values = [r["mw_equivalent"] for r in results]
        assert len(set(mw_values)) == 1


class TestProvenanceTracking:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_different_inputs_different_hash(self, calculator: VacuumCalculator):
        """Test different inputs produce different hashes."""
        input1 = VacuumOptimizationInput(
            current_backpressure_kpa=5.0,
            unit_load_mw=500.0,
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=15.0,
            cw_pump_power_kw=2000.0,
            heat_rate_kj_kwh=9500.0,
            electricity_price_usd_mwh=50.0,
        )

        input2 = VacuumOptimizationInput(
            current_backpressure_kpa=6.0,  # Different
            unit_load_mw=500.0,
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=15.0,
            cw_pump_power_kw=2000.0,
            heat_rate_kj_kwh=9500.0,
            electricity_price_usd_mwh=50.0,
        )

        result1 = calculator.optimize_vacuum(input1)
        result2 = calculator.optimize_vacuum(input2)

        assert result1.provenance_hash != result2.provenance_hash


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_very_low_load(self, calculator: VacuumCalculator):
        """Test with very low unit load."""
        inputs = VacuumOptimizationInput(
            current_backpressure_kpa=5.0,
            unit_load_mw=50.0,  # Very low load
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=5.0,
            cw_pump_power_kw=500.0,
            heat_rate_kj_kwh=10000.0,
            electricity_price_usd_mwh=50.0,
        )

        result = calculator.optimize_vacuum(inputs)

        assert result.mw_gain_potential >= 0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_very_high_load(self, calculator: VacuumCalculator):
        """Test with very high unit load."""
        inputs = VacuumOptimizationInput(
            current_backpressure_kpa=5.0,
            unit_load_mw=1200.0,  # Large unit
            cw_inlet_temp_c=25.0,
            cw_flow_m3_s=40.0,
            cw_pump_power_kw=6000.0,
            heat_rate_kj_kwh=9200.0,
            electricity_price_usd_mwh=50.0,
        )

        result = calculator.optimize_vacuum(inputs)

        assert result.optimal_backpressure_kpa > 0

    @pytest.mark.unit
    @pytest.mark.vacuum
    def test_extreme_cw_temperature(self, calculator: VacuumCalculator):
        """Test with extreme CW inlet temperature."""
        inputs = VacuumOptimizationInput(
            current_backpressure_kpa=10.0,
            unit_load_mw=500.0,
            cw_inlet_temp_c=40.0,  # Very hot
            cw_flow_m3_s=20.0,
            cw_pump_power_kw=3000.0,
            heat_rate_kj_kwh=10500.0,
            electricity_price_usd_mwh=50.0,
        )

        result = calculator.optimize_vacuum(inputs)

        # Achievable vacuum should be limited by high CW temp
        assert result.sensitivity_analysis["achievable_min_kpa"] > 5.0


class TestPerformance:
    """Performance tests."""

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.performance
    def test_optimization_speed(
        self,
        calculator: VacuumCalculator,
        vacuum_optimization_input: VacuumOptimizationInput,
        performance_timer
    ):
        """Test optimization completes within target time."""
        timer = performance_timer()

        with timer:
            for _ in range(100):
                calculator.optimize_vacuum(vacuum_optimization_input)

        # 100 optimizations should complete in < 1 second
        assert timer.elapsed < 1.0

    @pytest.mark.unit
    @pytest.mark.vacuum
    @pytest.mark.performance
    def test_curve_generation_speed(
        self,
        calculator: VacuumCalculator,
        performance_timer
    ):
        """Test curve generation completes quickly."""
        temps = list(range(10, 41, 1))  # 31 temperatures
        timer = performance_timer()

        with timer:
            for _ in range(100):
                calculator.calculate_vacuum_curves(temps, 500.0, 9500.0)

        # Should complete quickly
        assert timer.elapsed < 1.0
