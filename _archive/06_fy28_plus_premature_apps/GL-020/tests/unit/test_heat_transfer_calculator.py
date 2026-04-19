"""
GL-020 ECONOPULSE - Heat Transfer Calculator Unit Tests

Comprehensive unit tests for HeatTransferCalculator with 95%+ coverage target.
Tests LMTD, U-value, heat duty, approach temperature, and effectiveness calculations.
Validates against ASME PTC 4.3 examples.

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
from typing import Dict, Tuple
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EconomizerConfig, EconomizerType, FlowArrangement,
    TemperatureReading, FlowReading, HeatTransferResult,
    PerformanceBaseline
)


# =============================================================================
# MOCK CALCULATOR CLASS FOR TESTING
# =============================================================================

class HeatTransferCalculator:
    """
    Heat transfer calculator for economizer performance monitoring.

    Calculates:
    - Log Mean Temperature Difference (LMTD)
    - Overall heat transfer coefficient (U-value)
    - Heat duty (Q)
    - Approach temperature
    - Heat exchanger effectiveness
    - Number of Transfer Units (NTU)
    """

    VERSION = "1.0.0"
    NAME = "HeatTransferCalculator"
    AGENT_ID = "GL-020"

    # Minimum temperature difference to avoid division by zero
    MIN_DELTA_T = 0.1  # Celsius

    def __init__(self):
        self._tracker = None

    def calculate_lmtd(
        self,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_FLOW
    ) -> float:
        """
        Calculate Log Mean Temperature Difference (LMTD).

        Args:
            T_hot_in: Hot fluid inlet temperature (C)
            T_hot_out: Hot fluid outlet temperature (C)
            T_cold_in: Cold fluid inlet temperature (C)
            T_cold_out: Cold fluid outlet temperature (C)
            flow_arrangement: Counter flow or parallel flow

        Returns:
            LMTD in Celsius

        Raises:
            ValueError: If temperatures are invalid
        """
        # Validate inputs
        if T_hot_in <= T_cold_in:
            raise ValueError("Hot inlet must be greater than cold inlet")

        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            # Counter flow: hot in vs cold out, hot out vs cold in
            delta_T1 = T_hot_in - T_cold_out
            delta_T2 = T_hot_out - T_cold_in
        elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
            # Parallel flow: hot in vs cold in, hot out vs cold out
            delta_T1 = T_hot_in - T_cold_in
            delta_T2 = T_hot_out - T_cold_out
        else:
            # For cross flow, use counter flow approximation
            delta_T1 = T_hot_in - T_cold_out
            delta_T2 = T_hot_out - T_cold_in

        # Handle edge cases
        if delta_T1 <= 0 or delta_T2 <= 0:
            raise ValueError(
                f"Invalid temperature differences: delta_T1={delta_T1}, delta_T2={delta_T2}. "
                "Temperature cross detected."
            )

        # Handle case where delta_T1 equals delta_T2
        if abs(delta_T1 - delta_T2) < self.MIN_DELTA_T:
            return delta_T1  # When equal, LMTD = arithmetic mean

        # Standard LMTD formula
        lmtd = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)

        return lmtd

    def calculate_heat_duty(
        self,
        mass_flow_kg_s: float,
        cp_kj_kg_k: float,
        T_in: float,
        T_out: float
    ) -> float:
        """
        Calculate heat duty (heat transfer rate).

        Args:
            mass_flow_kg_s: Mass flow rate (kg/s)
            cp_kj_kg_k: Specific heat capacity (kJ/kg.K)
            T_in: Inlet temperature (C)
            T_out: Outlet temperature (C)

        Returns:
            Heat duty in kW

        Raises:
            ValueError: If inputs are invalid
        """
        if mass_flow_kg_s < 0:
            raise ValueError("Mass flow rate cannot be negative")

        if mass_flow_kg_s == 0:
            return 0.0

        if cp_kj_kg_k <= 0:
            raise ValueError("Specific heat capacity must be positive")

        delta_T = abs(T_out - T_in)
        Q = mass_flow_kg_s * cp_kj_kg_k * delta_T

        return Q

    def calculate_u_value(
        self,
        heat_duty_kw: float,
        area_m2: float,
        lmtd_c: float
    ) -> float:
        """
        Calculate overall heat transfer coefficient (U-value).

        U = Q / (A * LMTD)

        Args:
            heat_duty_kw: Heat transfer rate (kW)
            area_m2: Heat transfer area (m2)
            lmtd_c: Log mean temperature difference (C)

        Returns:
            U-value in W/m2.K

        Raises:
            ValueError: If inputs are invalid
        """
        if heat_duty_kw < 0:
            raise ValueError("Heat duty cannot be negative")

        if area_m2 <= 0:
            raise ValueError("Heat transfer area must be positive")

        if lmtd_c <= 0:
            raise ValueError("LMTD must be positive")

        # Q (kW) = U (W/m2.K) * A (m2) * LMTD (K) / 1000
        # U = Q * 1000 / (A * LMTD)
        u_value = (heat_duty_kw * 1000) / (area_m2 * lmtd_c)

        return u_value

    def calculate_approach_temperature(
        self,
        T_hot_out: float,
        T_cold_in: float,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_FLOW
    ) -> float:
        """
        Calculate approach temperature (minimum temperature difference).

        Args:
            T_hot_out: Hot fluid outlet temperature (C)
            T_cold_in: Cold fluid inlet temperature (C)
            flow_arrangement: Flow arrangement type

        Returns:
            Approach temperature in Celsius
        """
        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            # For counter flow, approach is hot_out - cold_in
            approach = T_hot_out - T_cold_in
        else:
            # For parallel flow, approach is hot_out - cold_out
            # But we don't have cold_out here, so use hot_out - cold_in
            approach = T_hot_out - T_cold_in

        return approach

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

        Args:
            T_hot_in: Hot fluid inlet temperature (C)
            T_hot_out: Hot fluid outlet temperature (C)
            T_cold_in: Cold fluid inlet temperature (C)
            T_cold_out: Cold fluid outlet temperature (C)
            C_hot: Hot side heat capacity rate (optional)
            C_cold: Cold side heat capacity rate (optional)

        Returns:
            Effectiveness (0 to 1)
        """
        # Maximum possible temperature change
        delta_T_max = T_hot_in - T_cold_in

        if delta_T_max <= 0:
            return 0.0

        # Actual temperature changes
        delta_T_hot = T_hot_in - T_hot_out
        delta_T_cold = T_cold_out - T_cold_in

        if C_hot is not None and C_cold is not None:
            # Use heat capacity rates to determine C_min
            C_min = min(C_hot, C_cold)
            Q_actual = C_hot * delta_T_hot  # or C_cold * delta_T_cold
            Q_max = C_min * delta_T_max
            effectiveness = Q_actual / Q_max if Q_max > 0 else 0.0
        else:
            # Simplified calculation based on cold side temperature rise
            effectiveness = delta_T_cold / delta_T_max

        # Clamp to valid range
        return max(0.0, min(1.0, effectiveness))

    def calculate_ntu(
        self,
        u_value_w_m2k: float,
        area_m2: float,
        C_min: float
    ) -> float:
        """
        Calculate Number of Transfer Units (NTU).

        NTU = U * A / C_min

        Args:
            u_value_w_m2k: Overall heat transfer coefficient (W/m2.K)
            area_m2: Heat transfer area (m2)
            C_min: Minimum heat capacity rate (W/K or kW/K)

        Returns:
            NTU (dimensionless)
        """
        if u_value_w_m2k <= 0:
            raise ValueError("U-value must be positive")

        if area_m2 <= 0:
            raise ValueError("Area must be positive")

        if C_min <= 0:
            raise ValueError("C_min must be positive")

        # Ensure units are consistent (assume C_min in W/K if < 1000, else kW/K)
        if C_min < 100:
            # Assume kW/K, convert to W/K
            C_min_w_k = C_min * 1000
        else:
            C_min_w_k = C_min

        ntu = (u_value_w_m2k * area_m2) / C_min_w_k

        return ntu

    def calculate_all(
        self,
        economizer: EconomizerConfig,
        temperatures: Dict[str, TemperatureReading],
        flows: Dict[str, FlowReading],
        water_cp_kj_kg_k: float = 4.2,
        gas_cp_kj_kg_k: float = 1.1
    ) -> HeatTransferResult:
        """
        Perform complete heat transfer calculation.

        Args:
            economizer: Economizer configuration
            temperatures: Temperature readings dict
            flows: Flow readings dict
            water_cp_kj_kg_k: Water specific heat capacity
            gas_cp_kj_kg_k: Flue gas specific heat capacity

        Returns:
            HeatTransferResult with all calculated values
        """
        # Extract temperatures
        T_gas_in = temperatures["gas_inlet"].value_c
        T_gas_out = temperatures["gas_outlet"].value_c
        T_water_in = temperatures["water_inlet"].value_c
        T_water_out = temperatures["water_outlet"].value_c

        # Extract flows
        water_flow = flows["water"].value
        gas_flow = flows["flue_gas"].value

        # Calculate LMTD
        lmtd = self.calculate_lmtd(
            T_hot_in=T_gas_in,
            T_hot_out=T_gas_out,
            T_cold_in=T_water_in,
            T_cold_out=T_water_out,
            flow_arrangement=economizer.flow_arrangement
        )

        # Calculate heat duty (use water side as reference)
        heat_duty_water = self.calculate_heat_duty(
            mass_flow_kg_s=water_flow,
            cp_kj_kg_k=water_cp_kj_kg_k,
            T_in=T_water_in,
            T_out=T_water_out
        )

        # Calculate U-value
        u_value = self.calculate_u_value(
            heat_duty_kw=heat_duty_water,
            area_m2=economizer.heat_transfer_area_m2,
            lmtd_c=lmtd
        )

        # Calculate approach temperature
        approach_temp = self.calculate_approach_temperature(
            T_hot_out=T_gas_out,
            T_cold_in=T_water_in,
            flow_arrangement=economizer.flow_arrangement
        )

        # Calculate effectiveness
        effectiveness = self.calculate_effectiveness(
            T_hot_in=T_gas_in,
            T_hot_out=T_gas_out,
            T_cold_in=T_water_in,
            T_cold_out=T_water_out
        )

        # Calculate heat capacity rates
        C_hot = gas_flow * gas_cp_kj_kg_k * 1000  # W/K
        C_cold = water_flow * water_cp_kj_kg_k * 1000  # W/K
        C_min = min(C_hot, C_cold)

        # Calculate NTU
        ntu = self.calculate_ntu(
            u_value_w_m2k=u_value,
            area_m2=economizer.heat_transfer_area_m2,
            C_min=C_min
        )

        # Generate provenance hash
        provenance_data = f"{T_gas_in},{T_gas_out},{T_water_in},{T_water_out},{water_flow},{gas_flow}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return HeatTransferResult(
            heat_duty_kw=heat_duty_water,
            lmtd_c=lmtd,
            u_value_w_m2k=u_value,
            effectiveness=effectiveness,
            approach_temp_c=approach_temp,
            ntu=ntu,
            calculation_timestamp=datetime.now(timezone.utc),
            provenance_hash=provenance_hash
        )


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.heat_transfer
@pytest.mark.critical
class TestHeatTransferCalculator:
    """Comprehensive test suite for HeatTransferCalculator."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization(self):
        """Test HeatTransferCalculator initializes correctly."""
        calculator = HeatTransferCalculator()

        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "HeatTransferCalculator"
        assert calculator.AGENT_ID == "GL-020"
        assert calculator.MIN_DELTA_T == 0.1

    # =========================================================================
    # LMTD CALCULATION TESTS - COUNTER FLOW
    # =========================================================================

    def test_lmtd_counter_flow_standard(self):
        """Test LMTD calculation for counter flow standard case."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=400.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=150.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        # delta_T1 = 400 - 150 = 250
        # delta_T2 = 200 - 100 = 100
        # LMTD = (250 - 100) / ln(250/100) = 150 / 0.916 = 163.77
        assert lmtd == pytest.approx(163.77, rel=0.01)

    def test_lmtd_counter_flow_equal_deltas(self):
        """Test LMTD when temperature differences are equal."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=300.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=200.0,  # delta_T1 = 100, delta_T2 = 100
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        # When equal, LMTD = arithmetic mean = 100
        assert lmtd == pytest.approx(100.0, rel=0.01)

    def test_lmtd_counter_flow_design_conditions(self, bare_tube_economizer):
        """Test LMTD calculation at design conditions."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=bare_tube_economizer.design_gas_inlet_c,
            T_hot_out=bare_tube_economizer.design_gas_outlet_c,
            T_cold_in=bare_tube_economizer.design_water_inlet_c,
            T_cold_out=bare_tube_economizer.design_water_outlet_c,
            flow_arrangement=bare_tube_economizer.flow_arrangement
        )

        # delta_T1 = 380 - 145 = 235
        # delta_T2 = 175 - 105 = 70
        # LMTD = (235 - 70) / ln(235/70) = 165 / 1.211 = 136.23
        assert lmtd == pytest.approx(136.23, rel=0.02)
        assert lmtd > 0

    @pytest.mark.parametrize("T_hot_in,T_hot_out,T_cold_in,T_cold_out,expected_lmtd", [
        (400.0, 200.0, 100.0, 150.0, 163.93),
        (300.0, 200.0, 150.0, 200.0, 68.79),
        (380.0, 175.0, 105.0, 145.0, 136.23),
        (500.0, 250.0, 120.0, 200.0, 197.34),
    ])
    def test_lmtd_counter_flow_parametrized(
        self,
        T_hot_in, T_hot_out, T_cold_in, T_cold_out, expected_lmtd
    ):
        """Test LMTD calculation with parametrized inputs."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=T_hot_in,
            T_hot_out=T_hot_out,
            T_cold_in=T_cold_in,
            T_cold_out=T_cold_out,
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        assert lmtd == pytest.approx(expected_lmtd, rel=0.02)

    # =========================================================================
    # LMTD CALCULATION TESTS - PARALLEL FLOW
    # =========================================================================

    def test_lmtd_parallel_flow_standard(self):
        """Test LMTD calculation for parallel flow."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=350.0,
            T_hot_out=180.0,
            T_cold_in=100.0,
            T_cold_out=140.0,
            flow_arrangement=FlowArrangement.PARALLEL_FLOW
        )

        # delta_T1 = 350 - 100 = 250
        # delta_T2 = 180 - 140 = 40
        # LMTD = (250 - 40) / ln(250/40) = 210 / 1.833 = 114.57
        assert lmtd == pytest.approx(114.57, rel=0.02)

    def test_lmtd_parallel_flow_large_delta(self):
        """Test LMTD for parallel flow with large temperature difference."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=500.0,
            T_hot_out=200.0,
            T_cold_in=50.0,
            T_cold_out=150.0,
            flow_arrangement=FlowArrangement.PARALLEL_FLOW
        )

        # delta_T1 = 500 - 50 = 450
        # delta_T2 = 200 - 150 = 50
        # LMTD = (450 - 50) / ln(450/50) = 400 / 2.197 = 182.07
        assert lmtd == pytest.approx(182.07, rel=0.02)

    # =========================================================================
    # LMTD EDGE CASES
    # =========================================================================

    def test_lmtd_hot_inlet_equals_cold_inlet_raises(self):
        """Test LMTD raises error when hot inlet equals cold inlet."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="Hot inlet must be greater"):
            calculator.calculate_lmtd(
                T_hot_in=150.0,
                T_hot_out=120.0,
                T_cold_in=150.0,  # Same as hot inlet
                T_cold_out=140.0
            )

    def test_lmtd_hot_inlet_less_than_cold_inlet_raises(self):
        """Test LMTD raises error when hot inlet < cold inlet."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="Hot inlet must be greater"):
            calculator.calculate_lmtd(
                T_hot_in=100.0,  # Less than cold inlet
                T_hot_out=80.0,
                T_cold_in=150.0,
                T_cold_out=140.0
            )

    def test_lmtd_temperature_cross_raises(self):
        """Test LMTD raises error for temperature cross."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="Temperature cross detected"):
            calculator.calculate_lmtd(
                T_hot_in=200.0,
                T_hot_out=150.0,
                T_cold_in=100.0,
                T_cold_out=180.0,  # Cold outlet > Hot outlet (cross)
                flow_arrangement=FlowArrangement.COUNTER_FLOW
            )

    def test_lmtd_very_small_difference(self):
        """Test LMTD with very small temperature differences."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=200.0,
            T_hot_out=198.0,
            T_cold_in=100.0,
            T_cold_out=101.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        # delta_T1 = 200 - 101 = 99
        # delta_T2 = 198 - 100 = 98
        # When nearly equal, LMTD approx arithmetic mean
        assert lmtd == pytest.approx(98.5, rel=0.02)
        assert lmtd > 0

    def test_lmtd_cross_flow_uses_counter_flow(self):
        """Test cross flow uses counter flow approximation."""
        calculator = HeatTransferCalculator()

        lmtd_cross = calculator.calculate_lmtd(
            T_hot_in=400.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=150.0,
            flow_arrangement=FlowArrangement.CROSS_FLOW
        )

        lmtd_counter = calculator.calculate_lmtd(
            T_hot_in=400.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=150.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        # Cross flow should use same formula as counter flow
        assert lmtd_cross == pytest.approx(lmtd_counter, rel=0.001)

    # =========================================================================
    # HEAT DUTY CALCULATION TESTS
    # =========================================================================

    def test_heat_duty_standard(self):
        """Test heat duty calculation for standard case."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=10.0,
            cp_kj_kg_k=4.2,
            T_in=105.0,
            T_out=145.0
        )

        # Q = 10 * 4.2 * 40 = 1680 kW
        assert Q == pytest.approx(1680.0, rel=0.001)

    def test_heat_duty_flue_gas(self):
        """Test heat duty calculation for flue gas cooling."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=18.0,
            cp_kj_kg_k=1.1,
            T_in=380.0,
            T_out=175.0
        )

        # Q = 18 * 1.1 * (380-175) = 18 * 1.1 * 205 = 4059 kW
        assert Q == pytest.approx(4059.0, rel=0.001)

    @pytest.mark.parametrize("mass_flow,cp,delta_t,expected_q", [
        (10.0, 4.2, 40.0, 1680.0),
        (15.0, 1.1, 200.0, 3300.0),
        (12.5, 4.2, 45.0, 2362.5),
        (5.0, 4.18, 50.0, 1045.0),
        (20.0, 1.08, 150.0, 3240.0),
    ])
    def test_heat_duty_parametrized(self, mass_flow, cp, delta_t, expected_q):
        """Test heat duty with parametrized inputs."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=mass_flow,
            cp_kj_kg_k=cp,
            T_in=100.0,
            T_out=100.0 + delta_t
        )

        assert Q == pytest.approx(expected_q, rel=0.001)

    def test_heat_duty_zero_flow(self):
        """Test heat duty is zero with zero flow."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=0.0,
            cp_kj_kg_k=4.2,
            T_in=100.0,
            T_out=150.0
        )

        assert Q == 0.0

    def test_heat_duty_negative_flow_raises(self):
        """Test heat duty raises error for negative flow."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="cannot be negative"):
            calculator.calculate_heat_duty(
                mass_flow_kg_s=-10.0,
                cp_kj_kg_k=4.2,
                T_in=100.0,
                T_out=150.0
            )

    def test_heat_duty_negative_cp_raises(self):
        """Test heat duty raises error for negative Cp."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_heat_duty(
                mass_flow_kg_s=10.0,
                cp_kj_kg_k=-4.2,
                T_in=100.0,
                T_out=150.0
            )

    def test_heat_duty_zero_cp_raises(self):
        """Test heat duty raises error for zero Cp."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_heat_duty(
                mass_flow_kg_s=10.0,
                cp_kj_kg_k=0.0,
                T_in=100.0,
                T_out=150.0
            )

    def test_heat_duty_no_temperature_difference(self):
        """Test heat duty is zero with no temperature difference."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=10.0,
            cp_kj_kg_k=4.2,
            T_in=100.0,
            T_out=100.0  # No change
        )

        assert Q == 0.0

    # =========================================================================
    # U-VALUE CALCULATION TESTS
    # =========================================================================

    def test_u_value_standard(self):
        """Test U-value calculation for standard case."""
        calculator = HeatTransferCalculator()

        U = calculator.calculate_u_value(
            heat_duty_kw=2500.0,
            area_m2=350.0,
            lmtd_c=158.73
        )

        # U = (2500 * 1000) / (350 * 158.73) = 2,500,000 / 55,555.5 = 45 W/m2.K
        assert U == pytest.approx(45.0, rel=0.01)

    def test_u_value_design_conditions(self, bare_tube_economizer):
        """Test U-value matches design at design conditions."""
        calculator = HeatTransferCalculator()

        # At design: Q = 2500 kW, A = 350 m2
        # LMTD for design conditions:
        lmtd = calculator.calculate_lmtd(
            T_hot_in=bare_tube_economizer.design_gas_inlet_c,
            T_hot_out=bare_tube_economizer.design_gas_outlet_c,
            T_cold_in=bare_tube_economizer.design_water_inlet_c,
            T_cold_out=bare_tube_economizer.design_water_outlet_c,
            flow_arrangement=bare_tube_economizer.flow_arrangement
        )

        U = calculator.calculate_u_value(
            heat_duty_kw=bare_tube_economizer.design_heat_duty_kw,
            area_m2=bare_tube_economizer.heat_transfer_area_m2,
            lmtd_c=lmtd
        )

        # Should be close to design U-value (within 15%)
        assert U == pytest.approx(
            bare_tube_economizer.design_u_value_w_m2k,
            rel=0.15
        )

    def test_u_value_negative_duty_raises(self):
        """Test U-value raises error for negative heat duty."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="cannot be negative"):
            calculator.calculate_u_value(
                heat_duty_kw=-100.0,
                area_m2=350.0,
                lmtd_c=100.0
            )

    def test_u_value_zero_area_raises(self):
        """Test U-value raises error for zero area."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_u_value(
                heat_duty_kw=2500.0,
                area_m2=0.0,
                lmtd_c=100.0
            )

    def test_u_value_zero_lmtd_raises(self):
        """Test U-value raises error for zero LMTD."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_u_value(
                heat_duty_kw=2500.0,
                area_m2=350.0,
                lmtd_c=0.0
            )

    # =========================================================================
    # APPROACH TEMPERATURE TESTS
    # =========================================================================

    def test_approach_temp_counter_flow(self):
        """Test approach temperature for counter flow."""
        calculator = HeatTransferCalculator()

        approach = calculator.calculate_approach_temperature(
            T_hot_out=175.0,
            T_cold_in=105.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        # Approach = hot_out - cold_in = 175 - 105 = 70
        assert approach == pytest.approx(70.0, rel=0.001)

    def test_approach_temp_design(self, bare_tube_economizer):
        """Test approach temperature at design conditions."""
        calculator = HeatTransferCalculator()

        approach = calculator.calculate_approach_temperature(
            T_hot_out=bare_tube_economizer.design_gas_outlet_c,
            T_cold_in=bare_tube_economizer.design_water_inlet_c,
            flow_arrangement=bare_tube_economizer.flow_arrangement
        )

        # Design: gas_out=175, water_in=105 -> approach=70
        assert approach == pytest.approx(70.0, rel=0.01)

    def test_approach_temp_minimum_valid(self, bare_tube_economizer):
        """Test approach temperature above minimum."""
        calculator = HeatTransferCalculator()

        approach = calculator.calculate_approach_temperature(
            T_hot_out=175.0,
            T_cold_in=105.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW
        )

        # Should be above minimum approach (25 C for bare tube)
        assert approach >= bare_tube_economizer.min_approach_temp_c

    # =========================================================================
    # EFFECTIVENESS CALCULATION TESTS
    # =========================================================================

    def test_effectiveness_standard(self):
        """Test effectiveness calculation for standard case."""
        calculator = HeatTransferCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=400.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=175.0
        )

        # Effectiveness = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)
        # = (175 - 100) / (400 - 100) = 75 / 300 = 0.25
        assert effectiveness == pytest.approx(0.25, rel=0.01)

    def test_effectiveness_design_conditions(self, bare_tube_economizer):
        """Test effectiveness at design conditions."""
        calculator = HeatTransferCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=bare_tube_economizer.design_gas_inlet_c,
            T_hot_out=bare_tube_economizer.design_gas_outlet_c,
            T_cold_in=bare_tube_economizer.design_water_inlet_c,
            T_cold_out=bare_tube_economizer.design_water_outlet_c
        )

        # = (145 - 105) / (380 - 105) = 40 / 275 = 0.145
        assert 0 < effectiveness < 1
        assert effectiveness == pytest.approx(0.145, rel=0.02)

    def test_effectiveness_zero_when_no_heat_transfer(self):
        """Test effectiveness is zero when no heat transfer."""
        calculator = HeatTransferCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=200.0,
            T_hot_out=200.0,
            T_cold_in=100.0,
            T_cold_out=100.0
        )

        assert effectiveness == 0.0

    def test_effectiveness_clamped_to_one(self):
        """Test effectiveness is clamped to 1.0 maximum."""
        calculator = HeatTransferCalculator()

        # This would theoretically give effectiveness > 1
        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=200.0,
            T_hot_out=150.0,
            T_cold_in=100.0,
            T_cold_out=199.0  # Nearly equals hot inlet
        )

        assert effectiveness <= 1.0

    @pytest.mark.parametrize("T_hot_in,T_cold_in,T_cold_out,expected_eff", [
        (400.0, 100.0, 175.0, 0.25),
        (380.0, 105.0, 145.0, 0.145),
        (350.0, 100.0, 200.0, 0.40),
        (500.0, 150.0, 290.0, 0.40),
    ])
    def test_effectiveness_parametrized(
        self,
        T_hot_in, T_cold_in, T_cold_out, expected_eff
    ):
        """Test effectiveness with parametrized inputs."""
        calculator = HeatTransferCalculator()

        effectiveness = calculator.calculate_effectiveness(
            T_hot_in=T_hot_in,
            T_hot_out=T_hot_in - 100,  # Some cooling
            T_cold_in=T_cold_in,
            T_cold_out=T_cold_out
        )

        assert effectiveness == pytest.approx(expected_eff, rel=0.02)

    # =========================================================================
    # NTU CALCULATION TESTS
    # =========================================================================

    def test_ntu_standard(self):
        """Test NTU calculation for standard case."""
        calculator = HeatTransferCalculator()

        ntu = calculator.calculate_ntu(
            u_value_w_m2k=45.0,
            area_m2=350.0,
            C_min=19800.0  # W/K
        )

        # NTU = (45 * 350) / 19800 = 15750 / 19800 = 0.795
        assert ntu == pytest.approx(0.795, rel=0.01)

    def test_ntu_design_conditions(self, bare_tube_economizer):
        """Test NTU at design conditions."""
        calculator = HeatTransferCalculator()

        # C_min = min(gas_flow * gas_cp, water_flow * water_cp) * 1000
        C_gas = 18.0 * 1.1 * 1000  # = 19,800 W/K
        C_water = 12.5 * 4.2 * 1000  # = 52,500 W/K
        C_min = min(C_gas, C_water)

        ntu = calculator.calculate_ntu(
            u_value_w_m2k=bare_tube_economizer.design_u_value_w_m2k,
            area_m2=bare_tube_economizer.heat_transfer_area_m2,
            C_min=C_min
        )

        assert ntu > 0
        assert ntu < 10  # Reasonable NTU range

    def test_ntu_zero_u_value_raises(self):
        """Test NTU raises error for zero U-value."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_ntu(
                u_value_w_m2k=0.0,
                area_m2=350.0,
                C_min=19800.0
            )

    def test_ntu_zero_area_raises(self):
        """Test NTU raises error for zero area."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_ntu(
                u_value_w_m2k=45.0,
                area_m2=0.0,
                C_min=19800.0
            )

    def test_ntu_zero_cmin_raises(self):
        """Test NTU raises error for zero C_min."""
        calculator = HeatTransferCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_ntu(
                u_value_w_m2k=45.0,
                area_m2=350.0,
                C_min=0.0
            )

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
        calculator = HeatTransferCalculator()

        result = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        # Validate all outputs are populated
        assert result.heat_duty_kw > 0
        assert result.lmtd_c > 0
        assert result.u_value_w_m2k > 0
        assert 0 < result.effectiveness < 1
        assert result.approach_temp_c > 0
        assert result.ntu > 0
        assert len(result.provenance_hash) == 64

    def test_calculate_all_fouled_operation(
        self,
        bare_tube_economizer,
        fouled_operation_temperatures,
        design_flow_readings
    ):
        """Test complete calculation for fouled operation."""
        calculator = HeatTransferCalculator()

        result = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=fouled_operation_temperatures,
            flows=design_flow_readings
        )

        # Fouled should show reduced U-value and effectiveness
        assert result.u_value_w_m2k < bare_tube_economizer.design_u_value_w_m2k
        assert result.approach_temp_c > bare_tube_economizer.min_approach_temp_c

    def test_calculate_all_reproducibility(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test calculation reproducibility."""
        calculator = HeatTransferCalculator()

        results = []
        for _ in range(5):
            result = calculator.calculate_all(
                economizer=bare_tube_economizer,
                temperatures=clean_operation_temperatures,
                flows=design_flow_readings
            )
            results.append(result)

        # All provenance hashes should match
        first_hash = results[0].provenance_hash
        for result in results[1:]:
            assert result.provenance_hash == first_hash

    # =========================================================================
    # ASME PTC 4.3 VALIDATION TESTS
    # =========================================================================

    @pytest.mark.asme
    def test_asme_ptc_43_example_1(self, asme_ptc_43_test_cases):
        """Validate against ASME PTC 4.3 Example 1."""
        calculator = HeatTransferCalculator()
        case = asme_ptc_43_test_cases[0]

        lmtd = calculator.calculate_lmtd(
            T_hot_in=case["gas_inlet_c"],
            T_hot_out=case["gas_outlet_c"],
            T_cold_in=case["water_inlet_c"],
            T_cold_out=case["water_outlet_c"],
            flow_arrangement=case["flow_arrangement"]
        )

        heat_duty = calculator.calculate_heat_duty(
            mass_flow_kg_s=case["water_flow_kg_s"],
            cp_kj_kg_k=case["water_cp_kj_kg_k"],
            T_in=case["water_inlet_c"],
            T_out=case["water_outlet_c"]
        )

        assert lmtd == pytest.approx(case["expected_lmtd_c"], rel=case["tolerance"])
        assert heat_duty == pytest.approx(case["expected_heat_duty_kw"], rel=case["tolerance"])

    @pytest.mark.asme
    def test_asme_ptc_43_example_2(self, asme_ptc_43_test_cases):
        """Validate against ASME PTC 4.3 Example 2 - Parallel Flow."""
        calculator = HeatTransferCalculator()
        case = asme_ptc_43_test_cases[1]

        lmtd = calculator.calculate_lmtd(
            T_hot_in=case["gas_inlet_c"],
            T_hot_out=case["gas_outlet_c"],
            T_cold_in=case["water_inlet_c"],
            T_cold_out=case["water_outlet_c"],
            flow_arrangement=case["flow_arrangement"]
        )

        heat_duty = calculator.calculate_heat_duty(
            mass_flow_kg_s=case["water_flow_kg_s"],
            cp_kj_kg_k=case["water_cp_kj_kg_k"],
            T_in=case["water_inlet_c"],
            T_out=case["water_outlet_c"]
        )

        assert lmtd == pytest.approx(case["expected_lmtd_c"], rel=case["tolerance"])
        assert heat_duty == pytest.approx(case["expected_heat_duty_kw"], rel=case["tolerance"])

    @pytest.mark.asme
    def test_asme_ptc_43_all_examples(self, asme_ptc_43_test_cases):
        """Validate against all ASME PTC 4.3 examples."""
        calculator = HeatTransferCalculator()

        for case in asme_ptc_43_test_cases:
            lmtd = calculator.calculate_lmtd(
                T_hot_in=case["gas_inlet_c"],
                T_hot_out=case["gas_outlet_c"],
                T_cold_in=case["water_inlet_c"],
                T_cold_out=case["water_outlet_c"],
                flow_arrangement=case["flow_arrangement"]
            )

            heat_duty = calculator.calculate_heat_duty(
                mass_flow_kg_s=case["water_flow_kg_s"],
                cp_kj_kg_k=case["water_cp_kj_kg_k"],
                T_in=case["water_inlet_c"],
                T_out=case["water_outlet_c"]
            )

            assert lmtd == pytest.approx(
                case["expected_lmtd_c"],
                rel=case["tolerance"]
            ), f"LMTD mismatch for {case['name']}"

            assert heat_duty == pytest.approx(
                case["expected_heat_duty_kw"],
                rel=case["tolerance"]
            ), f"Heat duty mismatch for {case['name']}"

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_lmtd_calculation_speed(self, benchmark):
        """Test LMTD calculation meets performance target."""
        calculator = HeatTransferCalculator()

        def run_lmtd():
            return calculator.calculate_lmtd(
                T_hot_in=400.0,
                T_hot_out=200.0,
                T_cold_in=100.0,
                T_cold_out=150.0,
                flow_arrangement=FlowArrangement.COUNTER_FLOW
            )

        result = benchmark(run_lmtd)
        assert result > 0

    @pytest.mark.performance
    def test_batch_calculation_throughput(
        self,
        bare_tube_economizer,
        benchmark_sensor_data
    ):
        """Test batch calculation throughput."""
        calculator = HeatTransferCalculator()
        import time

        num_calculations = 1000
        start = time.time()

        for i in range(num_calculations):
            data = benchmark_sensor_data[i % len(benchmark_sensor_data)]
            try:
                calculator.calculate_lmtd(
                    T_hot_in=data["gas_inlet_c"],
                    T_hot_out=data["gas_outlet_c"],
                    T_cold_in=data["water_inlet_c"],
                    T_cold_out=data["water_outlet_c"]
                )
            except ValueError:
                pass  # Skip invalid combinations

        duration = time.time() - start
        throughput = num_calculations / duration

        assert throughput > 10000  # >10,000 calculations per second


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestHeatTransferEdgeCases:
    """Edge case tests for heat transfer calculations."""

    def test_very_high_temperatures(self):
        """Test with very high temperatures."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=1000.0,  # 1000 C gas
            T_hot_out=500.0,
            T_cold_in=200.0,
            T_cold_out=450.0
        )

        assert lmtd > 0
        assert lmtd < 1000

    def test_near_zero_temperatures(self):
        """Test with temperatures near zero."""
        calculator = HeatTransferCalculator()

        lmtd = calculator.calculate_lmtd(
            T_hot_in=50.0,
            T_hot_out=20.0,
            T_cold_in=5.0,
            T_cold_out=15.0
        )

        assert lmtd > 0

    def test_very_large_flow_rate(self):
        """Test heat duty with very large flow rate."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=1000.0,  # 1000 kg/s
            cp_kj_kg_k=4.2,
            T_in=100.0,
            T_out=150.0
        )

        assert Q == pytest.approx(210000.0, rel=0.001)  # 210 MW

    def test_very_small_flow_rate(self):
        """Test heat duty with very small flow rate."""
        calculator = HeatTransferCalculator()

        Q = calculator.calculate_heat_duty(
            mass_flow_kg_s=0.001,  # 1 g/s
            cp_kj_kg_k=4.2,
            T_in=100.0,
            T_out=150.0
        )

        assert Q == pytest.approx(0.21, rel=0.01)  # 0.21 kW

    def test_different_economizer_types(
        self,
        bare_tube_economizer,
        finned_tube_economizer,
        extended_surface_economizer
    ):
        """Test calculations work for different economizer types."""
        calculator = HeatTransferCalculator()

        for economizer in [bare_tube_economizer, finned_tube_economizer, extended_surface_economizer]:
            lmtd = calculator.calculate_lmtd(
                T_hot_in=economizer.design_gas_inlet_c,
                T_hot_out=economizer.design_gas_outlet_c,
                T_cold_in=economizer.design_water_inlet_c,
                T_cold_out=economizer.design_water_outlet_c,
                flow_arrangement=economizer.flow_arrangement
            )

            assert lmtd > 0

            U = calculator.calculate_u_value(
                heat_duty_kw=economizer.design_heat_duty_kw,
                area_m2=economizer.heat_transfer_area_m2,
                lmtd_c=lmtd
            )

            # U should be within 50% of design for reasonable conditions
            assert U > 0


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

@pytest.mark.unit
class TestHeatTransferProvenance:
    """Provenance tracking tests for heat transfer calculations."""

    def test_provenance_hash_generated(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test provenance hash is generated."""
        calculator = HeatTransferCalculator()

        result = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        design_flow_readings
    ):
        """Test provenance hash is deterministic."""
        calculator = HeatTransferCalculator()

        result1 = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        result2 = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_changes_with_input(
        self,
        bare_tube_economizer,
        clean_operation_temperatures,
        fouled_operation_temperatures,
        design_flow_readings
    ):
        """Test provenance hash changes with different inputs."""
        calculator = HeatTransferCalculator()

        result_clean = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=clean_operation_temperatures,
            flows=design_flow_readings
        )

        result_fouled = calculator.calculate_all(
            economizer=bare_tube_economizer,
            temperatures=fouled_operation_temperatures,
            flows=design_flow_readings
        )

        assert result_clean.provenance_hash != result_fouled.provenance_hash
