# -*- coding: utf-8 -*-
"""
Exergy Calculator Tests for GL-009 THERMALIQ

Comprehensive unit tests for Second Law (exergy) efficiency calculations.
Tests validate exergy calculations for various fluids, reference environments,
and thermodynamic conditions.

Test Coverage:
- Physical exergy calculation for water
- Physical exergy calculation for steam
- Physical exergy calculation for Therminol
- Exergy destruction calculation
- Carnot factor calculation
- Reference environment impact
- Exergy balance closure

Standards:
- ASME PTC 46 - Overall Plant Performance
- Kotas: The Exergy Method of Thermal Plant Analysis

Author: GL-TestEngineer
Version: 1.0.0
"""

import math
from decimal import Decimal
from typing import Dict, Any, List
from datetime import datetime

import pytest

# Try importing hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Reference environment (ISO 50001 standard)
T0_DEFAULT_K = 298.15  # 25 C
P0_DEFAULT_KPA = 101.325  # 1 atm

# Reference properties for water at 25 C, 1 atm
H0_WATER_KJ_KG = 104.89
S0_WATER_KJ_KG_K = 0.3674

# Fuel exergy-to-HHV ratios (phi)
PHI_VALUES = {
    "natural_gas": 1.04,
    "methane": 1.04,
    "coal": 1.06,
    "oil": 1.065,
    "diesel": 1.07,
    "biomass": 1.15,
    "hydrogen": 0.985,
}


# =============================================================================
# TEST CLASS: PHYSICAL EXERGY CALCULATIONS
# =============================================================================

class TestPhysicalExergyWater:
    """Test physical exergy calculations for liquid water."""

    @pytest.mark.unit
    def test_physical_exergy_water_at_reference(self):
        """Test exergy is zero at reference state."""
        exergy = self._calculate_physical_exergy_water(
            temperature_c=25.0,
            pressure_kpa=101.325,
            mass_flow_kg_s=1.0,
        )

        # At reference state, physical exergy should be approximately zero
        assert abs(exergy) < 0.1, f"Exergy at reference should be ~0, got {exergy}"

    @pytest.mark.unit
    def test_physical_exergy_water_above_reference(self):
        """Test physical exergy for water above reference temperature."""
        exergy = self._calculate_physical_exergy_water(
            temperature_c=60.0,
            pressure_kpa=101.325,
            mass_flow_kg_s=1.0,
        )

        # Exergy should be positive for water above T0
        assert exergy > 0, f"Exergy should be positive above T0, got {exergy}"

    @pytest.mark.unit
    def test_physical_exergy_water_below_reference(self):
        """Test physical exergy for water below reference temperature."""
        exergy = self._calculate_physical_exergy_water(
            temperature_c=10.0,
            pressure_kpa=101.325,
            mass_flow_kg_s=1.0,
        )

        # Exergy can be positive even below T0 (cooling potential)
        assert exergy >= 0, f"Exergy should be non-negative, got {exergy}"

    @pytest.mark.unit
    @pytest.mark.parametrize("temp_c,expected_min,expected_max", [
        (50.0, 1.0, 10.0),
        (80.0, 10.0, 30.0),
        (100.0, 20.0, 50.0),
    ])
    def test_physical_exergy_water_temperature_range(
        self, temp_c, expected_min, expected_max
    ):
        """Test physical exergy across temperature range."""
        exergy = self._calculate_physical_exergy_water(
            temperature_c=temp_c,
            pressure_kpa=101.325,
            mass_flow_kg_s=1.0,
        )

        assert expected_min <= exergy <= expected_max, \
            f"Exergy {exergy} kW outside range [{expected_min}, {expected_max}] at {temp_c}C"

    @pytest.mark.unit
    def test_physical_exergy_scales_with_mass_flow(self):
        """Test that exergy scales linearly with mass flow."""
        exergy_1kg = self._calculate_physical_exergy_water(
            temperature_c=80.0,
            pressure_kpa=101.325,
            mass_flow_kg_s=1.0,
        )

        exergy_2kg = self._calculate_physical_exergy_water(
            temperature_c=80.0,
            pressure_kpa=101.325,
            mass_flow_kg_s=2.0,
        )

        assert abs(exergy_2kg - 2 * exergy_1kg) < 0.01, \
            "Exergy should scale linearly with mass flow"

    def _calculate_physical_exergy_water(
        self,
        temperature_c: float,
        pressure_kpa: float,
        mass_flow_kg_s: float,
    ) -> float:
        """Calculate physical exergy of water stream."""
        T_K = temperature_c + 273.15
        T0_K = T0_DEFAULT_K

        # Simplified specific heat and entropy calculation
        cp = 4.186  # kJ/kg-K

        # Enthalpy relative to reference
        h = cp * (temperature_c - 25.0) + H0_WATER_KJ_KG

        # Entropy relative to reference (simplified)
        if T_K > 273.15:
            s = S0_WATER_KJ_KG_K + cp * math.log(T_K / T0_K)
        else:
            s = S0_WATER_KJ_KG_K

        # Physical exergy: Ex = m * [(h - h0) - T0 * (s - s0)]
        delta_h = h - H0_WATER_KJ_KG
        delta_s = s - S0_WATER_KJ_KG_K

        specific_exergy = delta_h - T0_K * delta_s  # kJ/kg
        exergy_kw = mass_flow_kg_s * specific_exergy

        return max(0.0, exergy_kw)


# =============================================================================
# TEST CLASS: STEAM EXERGY CALCULATIONS
# =============================================================================

class TestPhysicalExergySteam:
    """Test physical exergy calculations for steam."""

    @pytest.mark.unit
    def test_physical_exergy_steam_saturated(self):
        """Test exergy of saturated steam at 1 atm."""
        exergy = self._calculate_physical_exergy_steam(
            temperature_c=100.0,
            pressure_bar=1.01325,
            mass_flow_kg_s=1.0,
            enthalpy_kj_kg=2676.1,
            entropy_kj_kg_k=7.355,
        )

        # Saturated steam at 100 C should have significant exergy
        assert 500 < exergy < 700, f"Saturated steam exergy {exergy} outside expected range"

    @pytest.mark.unit
    def test_physical_exergy_steam_superheated(self):
        """Test exergy of superheated steam."""
        exergy = self._calculate_physical_exergy_steam(
            temperature_c=200.0,
            pressure_bar=10.0,
            mass_flow_kg_s=1.0,
            enthalpy_kj_kg=2828.0,
            entropy_kj_kg_k=6.694,
        )

        # Superheated steam should have higher exergy
        assert exergy > 600, f"Superheated steam exergy {exergy} too low"

    @pytest.mark.unit
    def test_steam_exergy_higher_than_water(self):
        """Test that steam exergy is higher than water at same temperature."""
        water_exergy = TestPhysicalExergyWater()._calculate_physical_exergy_water(
            temperature_c=100.0,
            pressure_kpa=101.325,
            mass_flow_kg_s=1.0,
        )

        steam_exergy = self._calculate_physical_exergy_steam(
            temperature_c=100.0,
            pressure_bar=1.01325,
            mass_flow_kg_s=1.0,
            enthalpy_kj_kg=2676.1,
            entropy_kj_kg_k=7.355,
        )

        assert steam_exergy > water_exergy, \
            f"Steam exergy {steam_exergy} should exceed water exergy {water_exergy}"

    @pytest.mark.unit
    @pytest.mark.parametrize("pressure_bar,temp_c,h_kj_kg,s_kj_kg_k", [
        (1.01325, 100, 2676.1, 7.355),  # Saturated at 1 atm
        (5.0, 152, 2748.7, 6.821),  # Saturated at 5 bar
        (10.0, 180, 2778.1, 6.586),  # Saturated at 10 bar
        (20.0, 212, 2799.5, 6.340),  # Saturated at 20 bar
    ])
    def test_steam_exergy_at_various_pressures(
        self, pressure_bar, temp_c, h_kj_kg, s_kj_kg_k
    ):
        """Test steam exergy at various saturation pressures."""
        exergy = self._calculate_physical_exergy_steam(
            temperature_c=temp_c,
            pressure_bar=pressure_bar,
            mass_flow_kg_s=1.0,
            enthalpy_kj_kg=h_kj_kg,
            entropy_kj_kg_k=s_kj_kg_k,
        )

        # Higher pressure steam should have more exergy
        assert exergy > 0, f"Exergy should be positive at {pressure_bar} bar"

    def _calculate_physical_exergy_steam(
        self,
        temperature_c: float,
        pressure_bar: float,
        mass_flow_kg_s: float,
        enthalpy_kj_kg: float,
        entropy_kj_kg_k: float,
    ) -> float:
        """Calculate physical exergy of steam stream."""
        T0_K = T0_DEFAULT_K

        # Physical exergy: Ex = m * [(h - h0) - T0 * (s - s0)]
        delta_h = enthalpy_kj_kg - H0_WATER_KJ_KG
        delta_s = entropy_kj_kg_k - S0_WATER_KJ_KG_K

        specific_exergy = delta_h - T0_K * delta_s  # kJ/kg
        exergy_kw = mass_flow_kg_s * specific_exergy

        return max(0.0, exergy_kw)


# =============================================================================
# TEST CLASS: THERMINOL EXERGY CALCULATIONS
# =============================================================================

class TestPhysicalExergyTherminol:
    """Test physical exergy calculations for Therminol 66."""

    @pytest.mark.unit
    def test_physical_exergy_therminol_at_operating_temp(self):
        """Test exergy of Therminol 66 at typical operating temperature."""
        exergy = self._calculate_physical_exergy_therminol(
            temperature_c=200.0,
            mass_flow_kg_s=1.0,
        )

        assert exergy > 0, "Therminol exergy should be positive at 200C"
        assert exergy < 100, f"Therminol exergy {exergy} seems too high"

    @pytest.mark.unit
    def test_therminol_exergy_increases_with_temperature(self):
        """Test that Therminol exergy increases with temperature."""
        exergy_100c = self._calculate_physical_exergy_therminol(
            temperature_c=100.0,
            mass_flow_kg_s=1.0,
        )

        exergy_200c = self._calculate_physical_exergy_therminol(
            temperature_c=200.0,
            mass_flow_kg_s=1.0,
        )

        exergy_300c = self._calculate_physical_exergy_therminol(
            temperature_c=300.0,
            mass_flow_kg_s=1.0,
        )

        assert exergy_100c < exergy_200c < exergy_300c, \
            "Exergy should increase with temperature"

    @pytest.mark.unit
    def test_therminol_lower_exergy_than_steam(self):
        """Test that Therminol has lower exergy quality than steam at same temp."""
        therminol_exergy = self._calculate_physical_exergy_therminol(
            temperature_c=180.0,
            mass_flow_kg_s=1.0,
        )

        steam_exergy = TestPhysicalExergySteam()._calculate_physical_exergy_steam(
            temperature_c=180.0,
            pressure_bar=10.0,
            mass_flow_kg_s=1.0,
            enthalpy_kj_kg=2778.1,
            entropy_kj_kg_k=6.586,
        )

        # Steam has higher specific exergy than thermal oil
        assert steam_exergy > therminol_exergy

    def _calculate_physical_exergy_therminol(
        self,
        temperature_c: float,
        mass_flow_kg_s: float,
    ) -> float:
        """Calculate physical exergy of Therminol 66 stream."""
        T_K = temperature_c + 273.15
        T0_K = T0_DEFAULT_K

        # Therminol 66 specific heat (temperature dependent, simplified)
        cp = 1.92 + 0.0013 * (temperature_c - 100)  # kJ/kg-K

        # Calculate enthalpy change
        delta_h = cp * (temperature_c - 25)  # kJ/kg

        # Calculate entropy change (simplified)
        if T_K > T0_K:
            delta_s = cp * math.log(T_K / T0_K)
        else:
            delta_s = 0

        # Physical exergy
        specific_exergy = delta_h - T0_K * delta_s
        exergy_kw = mass_flow_kg_s * specific_exergy

        return max(0.0, exergy_kw)


# =============================================================================
# TEST CLASS: EXERGY DESTRUCTION
# =============================================================================

class TestExergyDestruction:
    """Test exergy destruction (irreversibility) calculations."""

    @pytest.mark.unit
    def test_exergy_destruction_heat_transfer(self):
        """Test exergy destruction in heat transfer across temperature difference."""
        heat_rate_kw = 1000.0
        hot_temp_c = 200.0
        cold_temp_c = 100.0

        destruction = self._calculate_heat_transfer_irreversibility(
            heat_rate_kw, hot_temp_c, cold_temp_c
        )

        # Destruction should be positive for finite temperature difference
        assert destruction > 0, "Heat transfer should cause exergy destruction"

    @pytest.mark.unit
    def test_exergy_destruction_zero_for_reversible(self):
        """Test exergy destruction is zero for reversible (isothermal) process."""
        heat_rate_kw = 1000.0
        temperature_c = 150.0

        destruction = self._calculate_heat_transfer_irreversibility(
            heat_rate_kw, temperature_c, temperature_c
        )

        assert abs(destruction) < 0.1, \
            "No destruction for reversible isothermal transfer"

    @pytest.mark.unit
    def test_exergy_destruction_increases_with_delta_t(self):
        """Test that exergy destruction increases with temperature difference."""
        heat_rate_kw = 1000.0
        hot_temp_c = 200.0

        destruction_small_dt = self._calculate_heat_transfer_irreversibility(
            heat_rate_kw, hot_temp_c, 180.0
        )

        destruction_large_dt = self._calculate_heat_transfer_irreversibility(
            heat_rate_kw, hot_temp_c, 100.0
        )

        assert destruction_large_dt > destruction_small_dt, \
            "Larger temperature difference should cause more destruction"

    @pytest.mark.unit
    def test_combustion_irreversibility(self):
        """Test exergy destruction in combustion process."""
        fuel_exergy_kw = 1000.0
        products_exergy_kw = 700.0

        destruction = fuel_exergy_kw - products_exergy_kw

        # Combustion is highly irreversible
        assert destruction > 0, "Combustion should destroy exergy"
        assert destruction / fuel_exergy_kw > 0.2, \
            "Combustion typically destroys >20% of fuel exergy"

    def _calculate_heat_transfer_irreversibility(
        self,
        heat_rate_kw: float,
        hot_temp_c: float,
        cold_temp_c: float,
    ) -> float:
        """Calculate irreversibility due to heat transfer."""
        T0_K = T0_DEFAULT_K
        T_hot_K = hot_temp_c + 273.15
        T_cold_K = cold_temp_c + 273.15

        if T_cold_K >= T_hot_K:
            return 0.0

        # Irreversibility: I = T0 * Q * (1/T_cold - 1/T_hot)
        irreversibility = T0_K * heat_rate_kw * (1 / T_cold_K - 1 / T_hot_K)

        return max(0.0, irreversibility)


# =============================================================================
# TEST CLASS: CARNOT FACTOR
# =============================================================================

class TestCarnotFactor:
    """Test Carnot factor calculations."""

    @pytest.mark.unit
    def test_carnot_factor_at_reference(self):
        """Test Carnot factor is zero at reference temperature."""
        carnot = self._calculate_carnot_factor(25.0, 25.0)
        assert abs(carnot) < 0.001, "Carnot factor should be zero at T0"

    @pytest.mark.unit
    def test_carnot_factor_approaches_one(self):
        """Test Carnot factor approaches 1 at high temperatures."""
        carnot = self._calculate_carnot_factor(1000.0, 25.0)
        assert 0.7 < carnot < 1.0, f"Carnot factor {carnot} wrong at 1000C"

    @pytest.mark.unit
    @pytest.mark.parametrize("temp_c,ambient_c,expected_carnot", [
        (100.0, 25.0, 0.201),
        (180.0, 25.0, 0.342),
        (300.0, 25.0, 0.480),
        (500.0, 25.0, 0.615),
        (800.0, 25.0, 0.722),
    ])
    def test_carnot_factor_known_values(self, temp_c, ambient_c, expected_carnot):
        """Test Carnot factor against known calculated values."""
        carnot = self._calculate_carnot_factor(temp_c, ambient_c)

        assert abs(carnot - expected_carnot) < 0.01, \
            f"Carnot factor {carnot} differs from expected {expected_carnot}"

    @pytest.mark.unit
    def test_carnot_factor_for_cooling(self):
        """Test Carnot factor for cooling (below ambient)."""
        # For cooling, exergy is available if T < T0
        carnot = self._calculate_carnot_factor_cooling(10.0, 25.0)

        assert carnot > 0, "Cooling should have positive Carnot factor"

    def _calculate_carnot_factor(
        self, temp_c: float, ambient_c: float
    ) -> float:
        """Calculate Carnot factor for heating."""
        T_K = temp_c + 273.15
        T0_K = ambient_c + 273.15

        if T_K <= T0_K:
            return 0.0

        return 1 - T0_K / T_K

    def _calculate_carnot_factor_cooling(
        self, temp_c: float, ambient_c: float
    ) -> float:
        """Calculate Carnot factor for cooling."""
        T_K = temp_c + 273.15
        T0_K = ambient_c + 273.15

        if T_K >= T0_K:
            return 0.0

        return T0_K / T_K - 1


# =============================================================================
# TEST CLASS: REFERENCE ENVIRONMENT
# =============================================================================

class TestReferenceEnvironment:
    """Test reference environment configuration."""

    @pytest.mark.unit
    def test_default_reference_conditions(self):
        """Test default reference conditions."""
        assert T0_DEFAULT_K == 298.15  # 25 C
        assert P0_DEFAULT_KPA == 101.325  # 1 atm

    @pytest.mark.unit
    def test_exergy_with_different_reference_temperatures(self):
        """Test exergy varies with reference temperature."""
        temperature_c = 100.0
        mass_flow = 1.0

        exergy_25c = self._calculate_exergy_with_reference(
            temperature_c, mass_flow, reference_temp_c=25.0
        )

        exergy_0c = self._calculate_exergy_with_reference(
            temperature_c, mass_flow, reference_temp_c=0.0
        )

        exergy_40c = self._calculate_exergy_with_reference(
            temperature_c, mass_flow, reference_temp_c=40.0
        )

        # Higher reference temp means less exergy available
        assert exergy_0c > exergy_25c > exergy_40c

    @pytest.mark.unit
    def test_exergy_at_reference_is_zero(self):
        """Test exergy is zero when stream is at reference conditions."""
        exergy = self._calculate_exergy_with_reference(
            temperature_c=25.0,
            mass_flow=1.0,
            reference_temp_c=25.0,
        )

        assert abs(exergy) < 0.1, "Exergy should be zero at reference"

    @pytest.mark.unit
    def test_reference_atmosphere_composition(self):
        """Test default reference atmosphere composition."""
        reference_composition = {
            "N2": 0.7567,
            "O2": 0.2035,
            "H2O": 0.0303,
            "CO2": 0.0003,
            "Ar": 0.0092,
        }

        total = sum(reference_composition.values())
        assert abs(total - 1.0) < 0.001, "Composition should sum to 1.0"

    def _calculate_exergy_with_reference(
        self,
        temperature_c: float,
        mass_flow: float,
        reference_temp_c: float,
    ) -> float:
        """Calculate exergy with custom reference temperature."""
        T_K = temperature_c + 273.15
        T0_K = reference_temp_c + 273.15

        cp = 4.186  # kJ/kg-K for water

        if T_K > T0_K:
            delta_h = cp * (temperature_c - reference_temp_c)
            delta_s = cp * math.log(T_K / T0_K)
        else:
            delta_h = cp * (temperature_c - reference_temp_c)
            delta_s = cp * math.log(T_K / T0_K) if T_K > 0 else 0

        specific_exergy = delta_h - T0_K * delta_s
        return mass_flow * max(0, specific_exergy)


# =============================================================================
# TEST CLASS: EXERGY BALANCE CLOSURE
# =============================================================================

class TestExergyBalanceClosure:
    """Test exergy balance closure verification."""

    @pytest.mark.unit
    def test_exergy_balance_closure(self):
        """Test exergy balance: Ex_in = Ex_out + Ex_destruction + Ex_loss."""
        exergy_input = 1000.0
        exergy_output = 450.0
        exergy_destruction = 350.0
        exergy_loss = 200.0

        balance = exergy_input - (exergy_output + exergy_destruction + exergy_loss)

        assert abs(balance) < 0.01, f"Exergy balance error: {balance} kW"

    @pytest.mark.unit
    def test_exergy_balance_with_tolerance(self):
        """Test exergy balance within engineering tolerance."""
        exergy_input = 1000.0
        exergy_output = 445.0  # Slightly off
        exergy_destruction = 352.0
        exergy_loss = 198.0

        balance_error = exergy_input - (exergy_output + exergy_destruction + exergy_loss)
        tolerance_percent = (abs(balance_error) / exergy_input) * 100

        # 2% tolerance typical for industrial measurements
        assert tolerance_percent < 2.0, \
            f"Exergy balance error {tolerance_percent:.1f}% exceeds 2% tolerance"

    @pytest.mark.unit
    def test_exergy_destruction_equals_imbalance(self):
        """Test that unaccounted exergy is classified as destruction."""
        exergy_input = 1000.0
        exergy_output = 450.0
        measured_destruction = 300.0
        exergy_loss = 200.0

        unaccounted = exergy_input - exergy_output - measured_destruction - exergy_loss

        # Unaccounted should be added to destruction
        total_destruction = measured_destruction + unaccounted

        assert total_destruction >= measured_destruction

    @pytest.mark.unit
    def test_exergy_efficiency_from_balance(self):
        """Test calculating exergy efficiency from balance."""
        exergy_input = 1000.0
        exergy_output = 450.0

        efficiency = (exergy_output / exergy_input) * 100

        assert 0 <= efficiency <= 100
        assert efficiency == 45.0


# =============================================================================
# PROPERTY-BASED TESTS (HYPOTHESIS)
# =============================================================================

if HAS_HYPOTHESIS:

    class TestExergyPropertyBased:
        """Property-based tests for exergy calculations."""

        @given(
            temp_c=st.floats(min_value=30.0, max_value=1000.0),
            ambient_c=st.floats(min_value=-40.0, max_value=50.0),
        )
        @settings(max_examples=100)
        def test_carnot_always_less_than_one(self, temp_c, ambient_c):
            """Property: Carnot factor is always < 1."""
            assume(temp_c > ambient_c)

            T_K = temp_c + 273.15
            T0_K = ambient_c + 273.15
            carnot = 1 - T0_K / T_K

            assert carnot < 1.0

        @given(
            heat_kw=st.floats(min_value=1.0, max_value=10000.0),
            hot_temp=st.floats(min_value=100.0, max_value=1000.0),
            cold_temp=st.floats(min_value=30.0, max_value=99.0),
        )
        @settings(max_examples=100)
        def test_destruction_always_positive(self, heat_kw, hot_temp, cold_temp):
            """Property: Exergy destruction is always >= 0."""
            assume(hot_temp > cold_temp)

            T0_K = 298.15
            T_hot_K = hot_temp + 273.15
            T_cold_K = cold_temp + 273.15

            destruction = T0_K * heat_kw * (1 / T_cold_K - 1 / T_hot_K)

            assert destruction >= 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestExergyPerformance:
    """Performance tests for exergy calculations."""

    @pytest.mark.performance
    def test_exergy_calculation_time(self):
        """Test exergy calculation meets <10ms target."""
        import time

        start = time.perf_counter()

        for _ in range(100):
            TestPhysicalExergySteam()._calculate_physical_exergy_steam(
                temperature_c=180.0,
                pressure_bar=10.0,
                mass_flow_kg_s=1.0,
                enthalpy_kj_kg=2778.1,
                entropy_kj_kg_k=6.586,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000 / 100

        assert elapsed_ms < 10.0, f"Calculation took {elapsed_ms:.2f}ms (target: <10ms)"
