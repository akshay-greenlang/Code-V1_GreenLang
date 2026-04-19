"""
GL-020 ECONOPULSE - Economizer Efficiency Calculator Unit Tests

Comprehensive unit tests for EconomizerEfficiencyCalculator with 95%+ coverage target.
Tests effectiveness calculation, heat recovery ratio, gas-side/water-side efficiency,
and design deviation analysis.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
import math
import hashlib
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EconomizerConfig, EconomizerType, FlowArrangement,
    TemperatureReading, FlowReading, PerformanceBaseline
)


# =============================================================================
# MOCK CALCULATOR CLASS FOR TESTING
# =============================================================================

@dataclass
class EfficiencyResult:
    """Result of efficiency calculation."""
    effectiveness: float
    heat_recovery_ratio: float
    gas_side_efficiency: float
    water_side_efficiency: float
    design_deviation_pct: float
    actual_heat_duty_kw: float
    maximum_possible_duty_kw: float
    capacity_ratio: float
    performance_index: float
    provenance_hash: str
    calculation_timestamp: datetime


class EconomizerEfficiencyCalculator:
    """
    Efficiency calculator for economizer performance monitoring.

    Calculates:
    - Heat exchanger effectiveness
    - Heat recovery ratio
    - Gas-side efficiency
    - Water-side efficiency
    - Design deviation
    - Performance index
    """

    VERSION = "1.0.0"
    NAME = "EconomizerEfficiencyCalculator"
    AGENT_ID = "GL-020"

    def __init__(self):
        self._tracker = None

    def calculate_effectiveness(
        self,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float,
        C_hot: float = None,
        C_cold: float = None
    ) -> float:
        """
        Calculate heat exchanger effectiveness.

        Effectiveness = Actual heat transfer / Maximum possible heat transfer

        For counter-flow:
        epsilon = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)
        when C_cold < C_hot

        Args:
            T_hot_in: Hot fluid inlet temperature (C)
            T_hot_out: Hot fluid outlet temperature (C)
            T_cold_in: Cold fluid inlet temperature (C)
            T_cold_out: Cold fluid outlet temperature (C)
            C_hot: Hot side heat capacity rate (W/K) - optional
            C_cold: Cold side heat capacity rate (W/K) - optional

        Returns:
            Effectiveness (0 to 1)
        """
        delta_T_max = T_hot_in - T_cold_in

        if delta_T_max <= 0:
            return 0.0

        # If heat capacity rates provided, use full NTU-effectiveness method
        if C_hot is not None and C_cold is not None:
            C_min = min(C_hot, C_cold)
            C_max = max(C_hot, C_cold)

            if C_min <= 0:
                return 0.0

            # Actual heat transfer
            Q_actual = C_hot * (T_hot_in - T_hot_out)

            # Maximum possible heat transfer
            Q_max = C_min * delta_T_max

            effectiveness = Q_actual / Q_max if Q_max > 0 else 0.0
        else:
            # Simplified effectiveness based on cold side temperature rise
            delta_T_cold = T_cold_out - T_cold_in
            effectiveness = delta_T_cold / delta_T_max

        # Clamp to valid range
        return max(0.0, min(1.0, effectiveness))

    def calculate_heat_recovery_ratio(
        self,
        actual_heat_duty_kw: float,
        available_heat_kw: float
    ) -> float:
        """
        Calculate heat recovery ratio.

        HRR = Actual recovered heat / Available heat in flue gas

        Args:
            actual_heat_duty_kw: Actual heat recovered (kW)
            available_heat_kw: Total available heat in flue gas (kW)

        Returns:
            Heat recovery ratio (0 to 1)
        """
        if available_heat_kw <= 0:
            return 0.0

        if actual_heat_duty_kw < 0:
            raise ValueError("Actual heat duty cannot be negative")

        hrr = actual_heat_duty_kw / available_heat_kw

        return max(0.0, min(1.0, hrr))

    def calculate_gas_side_efficiency(
        self,
        T_gas_in: float,
        T_gas_out: float,
        T_gas_reference: float = 15.0
    ) -> float:
        """
        Calculate gas-side efficiency (percentage of sensible heat recovered).

        Gas efficiency = (T_gas_in - T_gas_out) / (T_gas_in - T_reference) * 100

        Args:
            T_gas_in: Flue gas inlet temperature (C)
            T_gas_out: Flue gas outlet temperature (C)
            T_gas_reference: Reference temperature, typically ambient (C)

        Returns:
            Gas-side efficiency percentage
        """
        delta_T_available = T_gas_in - T_gas_reference

        if delta_T_available <= 0:
            return 0.0

        delta_T_actual = T_gas_in - T_gas_out

        efficiency = (delta_T_actual / delta_T_available) * 100

        return max(0.0, min(100.0, efficiency))

    def calculate_water_side_efficiency(
        self,
        T_water_in: float,
        T_water_out: float,
        T_design_water_out: float,
        T_gas_in: float
    ) -> float:
        """
        Calculate water-side efficiency (actual vs possible water temperature rise).

        Water efficiency = (T_water_out - T_water_in) / (T_design_water_out - T_water_in) * 100

        Args:
            T_water_in: Water inlet temperature (C)
            T_water_out: Actual water outlet temperature (C)
            T_design_water_out: Design water outlet temperature (C)
            T_gas_in: Gas inlet temperature for max possible (C)

        Returns:
            Water-side efficiency percentage
        """
        design_rise = T_design_water_out - T_water_in

        if design_rise <= 0:
            return 0.0

        actual_rise = T_water_out - T_water_in

        efficiency = (actual_rise / design_rise) * 100

        return max(0.0, min(100.0, efficiency))

    def calculate_design_deviation(
        self,
        actual_value: float,
        design_value: float
    ) -> float:
        """
        Calculate percentage deviation from design.

        Deviation = ((Actual - Design) / Design) * 100

        Args:
            actual_value: Actual measured value
            design_value: Design value

        Returns:
            Deviation percentage (negative = below design, positive = above)
        """
        if design_value == 0:
            raise ValueError("Design value cannot be zero")

        deviation = ((actual_value - design_value) / design_value) * 100

        return deviation

    def calculate_capacity_ratio(
        self,
        C_hot: float,
        C_cold: float
    ) -> float:
        """
        Calculate heat capacity ratio (C_min / C_max).

        Args:
            C_hot: Hot side heat capacity rate (W/K)
            C_cold: Cold side heat capacity rate (W/K)

        Returns:
            Capacity ratio (0 to 1)
        """
        if C_hot <= 0 or C_cold <= 0:
            raise ValueError("Heat capacity rates must be positive")

        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)

        return C_min / C_max

    def calculate_performance_index(
        self,
        current_u_value: float,
        design_u_value: float,
        current_effectiveness: float,
        design_effectiveness: float
    ) -> float:
        """
        Calculate overall performance index (0-100).

        PI = 0.6 * (U_current/U_design) + 0.4 * (eff_current/eff_design) * 100

        Args:
            current_u_value: Current U-value (W/m2.K)
            design_u_value: Design U-value (W/m2.K)
            current_effectiveness: Current effectiveness
            design_effectiveness: Design effectiveness

        Returns:
            Performance index (0 to 100)
        """
        if design_u_value <= 0 or design_effectiveness <= 0:
            raise ValueError("Design values must be positive")

        u_ratio = min(current_u_value / design_u_value, 1.0)
        eff_ratio = min(current_effectiveness / design_effectiveness, 1.0)

        # Weighted combination: 60% U-value, 40% effectiveness
        pi = (0.6 * u_ratio + 0.4 * eff_ratio) * 100

        return max(0.0, min(100.0, pi))

    def calculate_available_heat(
        self,
        gas_flow_kg_s: float,
        gas_cp_kj_kg_k: float,
        T_gas_in: float,
        T_reference: float = 15.0
    ) -> float:
        """
        Calculate available heat in flue gas.

        Args:
            gas_flow_kg_s: Flue gas mass flow rate (kg/s)
            gas_cp_kj_kg_k: Flue gas specific heat (kJ/kg.K)
            T_gas_in: Flue gas inlet temperature (C)
            T_reference: Reference temperature (C)

        Returns:
            Available heat in kW
        """
        if gas_flow_kg_s <= 0:
            return 0.0

        delta_T = T_gas_in - T_reference
        available_heat = gas_flow_kg_s * gas_cp_kj_kg_k * delta_T

        return max(0.0, available_heat)

    def calculate_maximum_possible_duty(
        self,
        gas_flow_kg_s: float,
        water_flow_kg_s: float,
        gas_cp_kj_kg_k: float,
        water_cp_kj_kg_k: float,
        T_gas_in: float,
        T_water_in: float
    ) -> float:
        """
        Calculate maximum possible heat duty (Q_max).

        Q_max = C_min * (T_hot_in - T_cold_in)

        Args:
            gas_flow_kg_s: Flue gas mass flow rate (kg/s)
            water_flow_kg_s: Water mass flow rate (kg/s)
            gas_cp_kj_kg_k: Flue gas specific heat (kJ/kg.K)
            water_cp_kj_kg_k: Water specific heat (kJ/kg.K)
            T_gas_in: Flue gas inlet temperature (C)
            T_water_in: Water inlet temperature (C)

        Returns:
            Maximum possible heat duty in kW
        """
        C_hot = gas_flow_kg_s * gas_cp_kj_kg_k
        C_cold = water_flow_kg_s * water_cp_kj_kg_k
        C_min = min(C_hot, C_cold)

        delta_T_max = T_gas_in - T_water_in
        Q_max = C_min * delta_T_max

        return max(0.0, Q_max)

    def calculate_all(
        self,
        economizer: EconomizerConfig,
        temperatures: Dict[str, TemperatureReading],
        flows: Dict[str, FlowReading],
        current_u_value: float,
        water_cp_kj_kg_k: float = 4.2,
        gas_cp_kj_kg_k: float = 1.1
    ) -> EfficiencyResult:
        """
        Perform complete efficiency calculation.

        Args:
            economizer: Economizer configuration
            temperatures: Temperature readings dict
            flows: Flow readings dict
            current_u_value: Current measured U-value (W/m2.K)
            water_cp_kj_kg_k: Water specific heat capacity
            gas_cp_kj_kg_k: Flue gas specific heat capacity

        Returns:
            EfficiencyResult with all calculated values
        """
        # Extract temperatures
        T_gas_in = temperatures["gas_inlet"].value_c
        T_gas_out = temperatures["gas_outlet"].value_c
        T_water_in = temperatures["water_inlet"].value_c
        T_water_out = temperatures["water_outlet"].value_c

        # Extract flows
        water_flow = flows["water"].value
        gas_flow = flows["flue_gas"].value

        # Calculate heat capacity rates (W/K)
        C_hot = gas_flow * gas_cp_kj_kg_k * 1000
        C_cold = water_flow * water_cp_kj_kg_k * 1000

        # Calculate effectiveness
        effectiveness = self.calculate_effectiveness(
            T_hot_in=T_gas_in,
            T_hot_out=T_gas_out,
            T_cold_in=T_water_in,
            T_cold_out=T_water_out,
            C_hot=C_hot,
            C_cold=C_cold
        )

        # Calculate actual heat duty (from water side)
        actual_heat_duty = water_flow * water_cp_kj_kg_k * (T_water_out - T_water_in)

        # Calculate available heat
        available_heat = self.calculate_available_heat(
            gas_flow_kg_s=gas_flow,
            gas_cp_kj_kg_k=gas_cp_kj_kg_k,
            T_gas_in=T_gas_in
        )

        # Calculate heat recovery ratio
        heat_recovery_ratio = self.calculate_heat_recovery_ratio(
            actual_heat_duty_kw=actual_heat_duty,
            available_heat_kw=available_heat
        )

        # Calculate gas-side efficiency
        gas_side_efficiency = self.calculate_gas_side_efficiency(
            T_gas_in=T_gas_in,
            T_gas_out=T_gas_out
        )

        # Calculate water-side efficiency
        water_side_efficiency = self.calculate_water_side_efficiency(
            T_water_in=T_water_in,
            T_water_out=T_water_out,
            T_design_water_out=economizer.design_water_outlet_c,
            T_gas_in=T_gas_in
        )

        # Calculate design deviation
        design_deviation = self.calculate_design_deviation(
            actual_value=actual_heat_duty,
            design_value=economizer.design_heat_duty_kw
        )

        # Calculate maximum possible duty
        max_possible_duty = self.calculate_maximum_possible_duty(
            gas_flow_kg_s=gas_flow,
            water_flow_kg_s=water_flow,
            gas_cp_kj_kg_k=gas_cp_kj_kg_k,
            water_cp_kj_kg_k=water_cp_kj_kg_k,
            T_gas_in=T_gas_in,
            T_water_in=T_water_in
        )

        # Calculate capacity ratio
        capacity_ratio = self.calculate_capacity_ratio(C_hot, C_cold)

        # Calculate design effectiveness (approximation)
        design_effectiveness = self.calculate_effectiveness(
            T_hot_in=economizer.design_gas_inlet_c,
            T_hot_out=economizer.design_gas_outlet_c,
            T_cold_in=economizer.design_water_inlet_c,
            T_cold_out=economizer.design_water_outlet_c
        )

        # Calculate performance index
        performance_index = self.calculate_performance_index(
            current_u_value=current_u_value,
            design_u_value=economizer.design_u_value_w_m2k,
            current_effectiveness=effectiveness,
            design_effectiveness=design_effectiveness
        )

        # Generate provenance hash
        provenance_data = f"{T_gas_in},{T_gas_out},{T_water_in},{T_water_out},{water_flow},{gas_flow},{current_u_value}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return EfficiencyResult(
            effectiveness=effectiveness,
            heat_recovery_ratio=heat_recovery_ratio,
            gas_side_efficiency=gas_side_efficiency,
            water_side_efficiency=water_side_efficiency,
            design_deviation_pct=design_deviation,
            actual_heat_duty_kw=actual_heat_duty,
            maximum_possible_duty_kw=max_possible_duty,
            capacity_ratio=capacity_ratio,
            performance_index=performance_index,
            provenance_hash=provenance_hash,
            calculation_timestamp=datetime.now(timezone.utc)
        )


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.critical
class TestEconomizerEfficiencyCalculator:
    """Comprehensive test suite for EconomizerEfficiencyCalculator."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization(self):
        """Test EconomizerEfficiencyCalculator initializes correctly."""
        calculator = EconomizerEfficiencyCalculator()

        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "EconomizerEfficiencyCalculator"
        assert calculator.AGENT_ID == "GL-020"

    # =========================================================================
    # EFFECTIVENESS CALCULATION TESTS
    # =========================================================================

    def test_effectiveness_standard(self):
        """Test effectiveness calculation for standard case."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=400.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=175.0
        )

        # epsilon = (175-100)/(400-100) = 75/300 = 0.25
        assert effectiveness == pytest.approx(0.25, rel=0.01)

    def test_effectiveness_with_capacity_rates(self):
        """Test effectiveness with heat capacity rates."""
        calculator = EconomizerEfficiencyCalculator()

        # C_hot = 18 * 1.1 * 1000 = 19,800 W/K
        # C_cold = 12.5 * 4.2 * 1000 = 52,500 W/K
        # C_min = 19,800 W/K (gas side)
        C_hot = 19800.0
        C_cold = 52500.0

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=380.0,
            T_hot_out=175.0,
            T_cold_in=105.0,
            T_cold_out=145.0,
            C_hot=C_hot,
            C_cold=C_cold
        )

        # Q_actual = C_hot * (T_hot_in - T_hot_out) = 19800 * 205 = 4,059,000 W = 4059 kW
        # Q_max = C_min * (T_hot_in - T_cold_in) = 19800 * 275 = 5,445,000 W = 5445 kW
        # effectiveness = 4059 / 5445 = 0.745
        assert effectiveness == pytest.approx(0.745, rel=0.02)

    def test_effectiveness_zero_delta_t(self):
        """Test effectiveness is zero when no temperature difference."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=150.0,
            T_hot_out=140.0,
            T_cold_in=150.0,  # Same as hot inlet
            T_cold_out=145.0
        )

        assert effectiveness == 0.0

    def test_effectiveness_no_heat_transfer(self):
        """Test effectiveness is zero with no heat transfer."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=300.0,
            T_hot_out=300.0,  # No cooling
            T_cold_in=100.0,
            T_cold_out=100.0  # No heating
        )

        assert effectiveness == 0.0

    def test_effectiveness_clamped_max(self):
        """Test effectiveness is clamped to 1.0 max."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=200.0,
            T_hot_out=150.0,
            T_cold_in=100.0,
            T_cold_out=195.0  # Very close to hot inlet
        )

        assert effectiveness <= 1.0

    @pytest.mark.parametrize("T_hot_in,T_cold_in,T_cold_out,expected_eff", [
        (400.0, 100.0, 175.0, 0.25),
        (380.0, 105.0, 145.0, 0.145),
        (350.0, 100.0, 200.0, 0.40),
        (500.0, 150.0, 290.0, 0.40),
        (300.0, 100.0, 150.0, 0.25),
    ])
    def test_effectiveness_parametrized(
        self,
        T_hot_in, T_cold_in, T_cold_out, expected_eff
    ):
        """Test effectiveness with parametrized inputs."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=T_hot_in,
            T_hot_out=T_hot_in - 100,
            T_cold_in=T_cold_in,
            T_cold_out=T_cold_out
        )

        assert effectiveness == pytest.approx(expected_eff, rel=0.02)

    # =========================================================================
    # HEAT RECOVERY RATIO TESTS
    # =========================================================================

    def test_heat_recovery_ratio_standard(self):
        """Test heat recovery ratio calculation."""
        calculator = EconomizerEfficiencyCalculator()

        hrr = calculator.calculate_heat_recovery_ratio(
            actual_heat_duty_kw=2100.0,
            available_heat_kw=7000.0
        )

        # HRR = 2100 / 7000 = 0.30
        assert hrr == pytest.approx(0.30, rel=0.01)

    def test_heat_recovery_ratio_full_recovery(self):
        """Test heat recovery ratio at 100%."""
        calculator = EconomizerEfficiencyCalculator()

        hrr = calculator.calculate_heat_recovery_ratio(
            actual_heat_duty_kw=5000.0,
            available_heat_kw=5000.0
        )

        assert hrr == pytest.approx(1.0, rel=0.01)

    def test_heat_recovery_ratio_zero_available(self):
        """Test heat recovery ratio with zero available heat."""
        calculator = EconomizerEfficiencyCalculator()

        hrr = calculator.calculate_heat_recovery_ratio(
            actual_heat_duty_kw=2000.0,
            available_heat_kw=0.0
        )

        assert hrr == 0.0

    def test_heat_recovery_ratio_negative_duty_raises(self):
        """Test heat recovery ratio raises error for negative duty."""
        calculator = EconomizerEfficiencyCalculator()

        with pytest.raises(ValueError, match="cannot be negative"):
            calculator.calculate_heat_recovery_ratio(
                actual_heat_duty_kw=-100.0,
                available_heat_kw=5000.0
            )

    # =========================================================================
    # GAS-SIDE EFFICIENCY TESTS
    # =========================================================================

    def test_gas_side_efficiency_standard(self):
        """Test gas-side efficiency calculation."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_gas_side_efficiency(
            T_gas_in=380.0,
            T_gas_out=175.0,
            T_gas_reference=15.0
        )

        # Efficiency = (380-175) / (380-15) * 100 = 205/365 * 100 = 56.16%
        assert efficiency == pytest.approx(56.16, rel=0.02)

    def test_gas_side_efficiency_high_recovery(self):
        """Test gas-side efficiency with high recovery."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_gas_side_efficiency(
            T_gas_in=400.0,
            T_gas_out=100.0,  # Very low outlet
            T_gas_reference=15.0
        )

        # Efficiency = (400-100) / (400-15) * 100 = 300/385 * 100 = 77.9%
        assert efficiency == pytest.approx(77.9, rel=0.02)

    def test_gas_side_efficiency_no_cooling(self):
        """Test gas-side efficiency with no gas cooling."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_gas_side_efficiency(
            T_gas_in=300.0,
            T_gas_out=300.0,  # No change
            T_gas_reference=15.0
        )

        assert efficiency == 0.0

    def test_gas_side_efficiency_below_reference(self):
        """Test gas-side efficiency when inlet below reference."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_gas_side_efficiency(
            T_gas_in=10.0,  # Below reference
            T_gas_out=8.0,
            T_gas_reference=15.0
        )

        assert efficiency == 0.0

    # =========================================================================
    # WATER-SIDE EFFICIENCY TESTS
    # =========================================================================

    def test_water_side_efficiency_design(self):
        """Test water-side efficiency at design conditions."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_water_side_efficiency(
            T_water_in=105.0,
            T_water_out=145.0,
            T_design_water_out=145.0,
            T_gas_in=380.0
        )

        # At design, efficiency should be 100%
        assert efficiency == pytest.approx(100.0, rel=0.01)

    def test_water_side_efficiency_below_design(self):
        """Test water-side efficiency below design."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_water_side_efficiency(
            T_water_in=105.0,
            T_water_out=130.0,  # Below design
            T_design_water_out=145.0,
            T_gas_in=380.0
        )

        # Efficiency = (130-105)/(145-105) * 100 = 25/40 * 100 = 62.5%
        assert efficiency == pytest.approx(62.5, rel=0.02)

    def test_water_side_efficiency_no_heating(self):
        """Test water-side efficiency with no heating."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_water_side_efficiency(
            T_water_in=105.0,
            T_water_out=105.0,  # No change
            T_design_water_out=145.0,
            T_gas_in=380.0
        )

        assert efficiency == 0.0

    # =========================================================================
    # DESIGN DEVIATION TESTS
    # =========================================================================

    def test_design_deviation_at_design(self):
        """Test design deviation is zero at design conditions."""
        calculator = EconomizerEfficiencyCalculator()

        deviation = calculator.calculate_design_deviation(
            actual_value=2500.0,
            design_value=2500.0
        )

        assert deviation == 0.0

    def test_design_deviation_below_design(self):
        """Test design deviation below design."""
        calculator = EconomizerEfficiencyCalculator()

        deviation = calculator.calculate_design_deviation(
            actual_value=2000.0,
            design_value=2500.0
        )

        # Deviation = (2000-2500)/2500 * 100 = -20%
        assert deviation == pytest.approx(-20.0, rel=0.01)

    def test_design_deviation_above_design(self):
        """Test design deviation above design."""
        calculator = EconomizerEfficiencyCalculator()

        deviation = calculator.calculate_design_deviation(
            actual_value=3000.0,
            design_value=2500.0
        )

        # Deviation = (3000-2500)/2500 * 100 = +20%
        assert deviation == pytest.approx(20.0, rel=0.01)

    def test_design_deviation_zero_design_raises(self):
        """Test design deviation raises error for zero design."""
        calculator = EconomizerEfficiencyCalculator()

        with pytest.raises(ValueError, match="cannot be zero"):
            calculator.calculate_design_deviation(
                actual_value=1000.0,
                design_value=0.0
            )

    # =========================================================================
    # CAPACITY RATIO TESTS
    # =========================================================================

    def test_capacity_ratio_gas_limited(self):
        """Test capacity ratio when gas is limiting."""
        calculator = EconomizerEfficiencyCalculator()

        # C_hot = 19,800 W/K, C_cold = 52,500 W/K
        ratio = calculator.calculate_capacity_ratio(
            C_hot=19800.0,
            C_cold=52500.0
        )

        # Ratio = 19800/52500 = 0.377
        assert ratio == pytest.approx(0.377, rel=0.01)

    def test_capacity_ratio_water_limited(self):
        """Test capacity ratio when water is limiting."""
        calculator = EconomizerEfficiencyCalculator()

        ratio = calculator.calculate_capacity_ratio(
            C_hot=50000.0,
            C_cold=30000.0
        )

        # Ratio = 30000/50000 = 0.60
        assert ratio == pytest.approx(0.60, rel=0.01)

    def test_capacity_ratio_equal(self):
        """Test capacity ratio when equal."""
        calculator = EconomizerEfficiencyCalculator()

        ratio = calculator.calculate_capacity_ratio(
            C_hot=40000.0,
            C_cold=40000.0
        )

        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_capacity_ratio_zero_raises(self):
        """Test capacity ratio raises error for zero values."""
        calculator = EconomizerEfficiencyCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_capacity_ratio(
                C_hot=0.0,
                C_cold=40000.0
            )

    # =========================================================================
    # PERFORMANCE INDEX TESTS
    # =========================================================================

    def test_performance_index_at_design(self):
        """Test performance index at design conditions."""
        calculator = EconomizerEfficiencyCalculator()

        pi = calculator.calculate_performance_index(
            current_u_value=45.0,
            design_u_value=45.0,
            current_effectiveness=0.75,
            design_effectiveness=0.75
        )

        # At design, PI should be 100
        assert pi == pytest.approx(100.0, rel=0.01)

    def test_performance_index_degraded(self):
        """Test performance index with degraded performance."""
        calculator = EconomizerEfficiencyCalculator()

        pi = calculator.calculate_performance_index(
            current_u_value=35.0,  # 78% of design
            design_u_value=45.0,
            current_effectiveness=0.60,  # 80% of design
            design_effectiveness=0.75
        )

        # PI = (0.6 * 0.778 + 0.4 * 0.80) * 100 = (0.467 + 0.32) * 100 = 78.7
        assert pi == pytest.approx(78.7, rel=0.02)

    def test_performance_index_severely_degraded(self):
        """Test performance index with severely degraded performance."""
        calculator = EconomizerEfficiencyCalculator()

        pi = calculator.calculate_performance_index(
            current_u_value=25.0,  # 56% of design
            design_u_value=45.0,
            current_effectiveness=0.45,  # 60% of design
            design_effectiveness=0.75
        )

        # PI should be significantly below 100
        assert pi < 60.0
        assert pi >= 0.0

    def test_performance_index_clamped(self):
        """Test performance index is clamped to 0-100."""
        calculator = EconomizerEfficiencyCalculator()

        # Test above 100 (current better than design)
        pi_high = calculator.calculate_performance_index(
            current_u_value=50.0,  # Above design
            design_u_value=45.0,
            current_effectiveness=0.80,  # Above design
            design_effectiveness=0.75
        )

        assert pi_high == 100.0

    # =========================================================================
    # COMPLETE CALCULATION TESTS
    # =========================================================================

    def test_calculate_all_clean_operation(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test complete calculation for clean operation."""
        calculator = EconomizerEfficiencyCalculator()

        result = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings,
            current_u_value=45.0
        )

        # Validate all outputs
        assert 0 < result.effectiveness < 1
        assert 0 < result.heat_recovery_ratio < 1
        assert 0 < result.gas_side_efficiency < 100
        assert 0 < result.water_side_efficiency < 100
        assert result.actual_heat_duty_kw > 0
        assert result.maximum_possible_duty_kw > 0
        assert 0 < result.capacity_ratio <= 1
        assert 0 < result.performance_index <= 100
        assert len(result.provenance_hash) == 64

    def test_calculate_all_fouled_operation(
        self,
        bare_tube_economizer,
        fouled_operation_temperatures,
        design_flow_readings
    ):
        """Test complete calculation for fouled operation."""
        calculator = EconomizerEfficiencyCalculator()

        result = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=fouled_operation_temperatures,
            flows=design_flow_readings,
            current_u_value=32.0  # Degraded U-value
        )

        # Fouled should show reduced efficiency
        assert result.effectiveness < 0.75  # Below typical design
        assert result.performance_index < 90.0  # Below design performance

    def test_calculate_all_reduced_load(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        reduced_flow_readings
    ):
        """Test complete calculation at reduced load."""
        calculator = EconomizerEfficiencyCalculator()

        result = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=reduced_flow_readings,
            current_u_value=45.0
        )

        # At reduced load, heat duty should be lower
        assert result.actual_heat_duty_kw < bare_tube_economizer.design_heat_duty_kw

    def test_calculate_all_reproducibility(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test calculation reproducibility."""
        calculator = EconomizerEfficiencyCalculator()

        results = []
        for _ in range(5):
            result = calculator.calculate_all(
                economizer=bare_tube_economizer,
                temperatures=clean_operation_temperatures,
                flows=design_flow_readings,
                current_u_value=45.0
            )
            results.append(result)

        # All provenance hashes should match
        first_hash = results[0].provenance_hash
        for result in results[1:]:
            assert result.provenance_hash == first_hash

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_efficiency_calculation_speed(self, benchmark):
        """Test efficiency calculation meets performance target."""
        calculator = EconomizerEfficiencyCalculator()

        def run_calculation():
            return calculator.calculate_effectiveness(
                T_hot_in=380.0,
                T_hot_out=175.0,
                T_cold_in=105.0,
                T_cold_out=145.0,
                C_hot=19800.0,
                C_cold=52500.0
            )

        result = benchmark(run_calculation)
        assert result > 0

    @pytest.mark.performance
    def test_batch_efficiency_throughput(
        self,
        bare_tube_economizer,
        benchmark_sensor_data
    ):
        """Test batch efficiency calculation throughput."""
        calculator = EconomizerEfficiencyCalculator()
        import time

        num_calculations = 5000
        start = time.time()

        for i in range(num_calculations):
            data = benchmark_sensor_data[i % len(benchmark_sensor_data)]
            try:
                calculator.calculate_effectiveness(
                    T_hot_in=data["gas_inlet_c"],
                    T_hot_out=data["gas_outlet_c"],
                    T_cold_in=data["water_inlet_c"],
                    T_cold_out=data["water_outlet_c"]
                )
            except (ValueError, ZeroDivisionError):
                pass

        duration = time.time() - start
        throughput = num_calculations / duration

        assert throughput > 50000  # >50,000 calculations per second


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestEfficiencyEdgeCases:
    """Edge case tests for efficiency calculations."""

    def test_very_low_effectiveness(self):
        """Test with very low effectiveness."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=400.0,
            T_hot_out=398.0,
            T_cold_in=100.0,
            T_cold_out=101.0  # Very small temperature rise
        )

        # epsilon = 1/300 = 0.0033
        assert effectiveness >= 0.0
        assert effectiveness < 0.01

    def test_effectiveness_at_different_economizer_types(
        self,
        bare_tube_economizer,
        finned_tube_economizer,
        extended_surface_economizer
    ):
        """Test effectiveness calculation for different economizer types."""
        calculator = EconomizerEfficiencyCalculator()

        for economizer in [bare_tube_economizer, finned_tube_economizer, extended_surface_economizer]:
            effectiveness = calculator.calculate_effectiveness(
                T_hot_in=economizer.design_gas_inlet_c,
                T_hot_out=economizer.design_gas_outlet_c,
                T_cold_in=economizer.design_water_inlet_c,
                T_cold_out=economizer.design_water_outlet_c
            )

            assert 0 < effectiveness < 1

    def test_very_high_temperature_difference(self):
        """Test with very high temperature difference."""
        calculator = EconomizerEfficiencyCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=800.0,
            T_hot_out=200.0,
            T_cold_in=50.0,
            T_cold_out=400.0
        )

        # epsilon = (400-50)/(800-50) = 350/750 = 0.467
        assert effectiveness == pytest.approx(0.467, rel=0.02)

    def test_gas_efficiency_very_high_inlet(self):
        """Test gas-side efficiency with very high inlet temperature."""
        calculator = EconomizerEfficiencyCalculator()

        efficiency = calculator.calculate_gas_side_efficiency(
            T_gas_in=1000.0,
            T_gas_out=200.0,
            T_gas_reference=15.0
        )

        # High temperature should yield high recovery
        assert efficiency > 80.0

    def test_design_deviation_extreme(self):
        """Test design deviation with extreme values."""
        calculator = EconomizerEfficiencyCalculator()

        # 90% below design
        deviation_low = calculator.calculate_design_deviation(
            actual_value=250.0,
            design_value=2500.0
        )
        assert deviation_low == pytest.approx(-90.0, rel=0.01)

        # 100% above design
        deviation_high = calculator.calculate_design_deviation(
            actual_value=5000.0,
            design_value=2500.0
        )
        assert deviation_high == pytest.approx(100.0, rel=0.01)


# =============================================================================
# INTEGRATION TESTS WITH FIXTURES
# =============================================================================

@pytest.mark.unit
class TestEfficiencyWithFixtures:
    """Tests using comprehensive fixtures."""

    def test_varying_load_efficiency(self, varying_load_temperatures, design_flow_readings, bare_tube_economizer):
        """Test efficiency at different load levels."""
        calculator = EconomizerEfficiencyCalculator()

        efficiencies = []
        for load_data in varying_load_temperatures:
            effectiveness = calculator.calculate_effectiveness(
                T_hot_in=load_data["gas_inlet"].value_c,
                T_hot_out=load_data["gas_outlet"].value_c,
                T_cold_in=load_data["water_inlet"].value_c,
                T_cold_out=load_data["water_outlet"].value_c
            )
            efficiencies.append((load_data["load_pct"], effectiveness))

        # At higher loads, effectiveness may vary
        # All should be valid
        for load_pct, eff in efficiencies:
            assert 0 <= eff <= 1, f"Invalid effectiveness at {load_pct}% load"

    def test_clean_vs_fouled_performance(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        fouled_operation_temperatures,
        design_flow_readings
    ):
        """Test performance difference between clean and fouled."""
        calculator = EconomizerEfficiencyCalculator()

        result_clean = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings,
            current_u_value=45.0
        )

        result_fouled = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=fouled_operation_temperatures,
            flows=design_flow_readings,
            current_u_value=32.0
        )

        # Fouled should have lower performance
        assert result_fouled.effectiveness < result_clean.effectiveness
        assert result_fouled.performance_index < result_clean.performance_index
        assert result_fouled.actual_heat_duty_kw < result_clean.actual_heat_duty_kw
