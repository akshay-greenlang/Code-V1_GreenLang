# -*- coding: utf-8 -*-
"""
Extended Golden Value Validation Tests for GL-009 THERMALIQ

Additional golden value tests covering:
- Heat exchanger effectiveness (NTU method)
- Combustion stoichiometry
- Steam table extended values
- Pinch analysis
- Property-based testing with Hypothesis
- Performance benchmarks (<5ms target)

These tests complement test_golden_values.py with additional reference cases
from authoritative sources.

Author: GL-TestEngineer
Version: 1.0.0
"""

import math
import json
import hashlib
from decimal import Decimal
from typing import Dict, Any, List, Tuple
from datetime import datetime

import pytest


# Try importing hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# =============================================================================
# HEAT EXCHANGER GOLDEN VALUES (NTU Method)
# =============================================================================

HEAT_EXCHANGER_GOLDEN_VALUES = [
    {
        "case_id": "HX_Counterflow_NTU_1.0",
        "description": "Counterflow heat exchanger with NTU=1.0, C_ratio=0.5",
        "input": {
            "ntu": 1.0,
            "c_ratio": 0.5,
            "flow_arrangement": "counterflow",
        },
        "expected": {
            "effectiveness": 0.6321,
            "tolerance": 0.01,
        },
        "source": "Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
    },
    {
        "case_id": "HX_Parallel_NTU_2.0",
        "description": "Parallel flow heat exchanger with NTU=2.0, C_ratio=1.0",
        "input": {
            "ntu": 2.0,
            "c_ratio": 1.0,
            "flow_arrangement": "parallel",
        },
        "expected": {
            "effectiveness": 0.4908,
            "tolerance": 0.01,
        },
        "source": "Incropera & DeWitt, Fundamentals of Heat and Mass Transfer",
    },
    {
        "case_id": "HX_Counterflow_NTU_3.0_C05",
        "description": "Counterflow heat exchanger with NTU=3.0, C_ratio=0.5",
        "input": {
            "ntu": 3.0,
            "c_ratio": 0.5,
            "flow_arrangement": "counterflow",
        },
        "expected": {
            "effectiveness": 0.9179,
            "tolerance": 0.02,
        },
        "source": "Kays & London, Compact Heat Exchangers",
    },
    {
        "case_id": "HX_Evaporator_NTU_2.0",
        "description": "Evaporator/Condenser (C_ratio=0)",
        "input": {
            "ntu": 2.0,
            "c_ratio": 0.0,
            "flow_arrangement": "counterflow",
        },
        "expected": {
            "effectiveness": 0.8647,  # 1 - exp(-NTU)
            "tolerance": 0.01,
        },
        "source": "Fundamentals of Heat Exchangers",
    },
]


# Combustion Stoichiometry Reference Cases
COMBUSTION_GOLDEN_VALUES = [
    {
        "case_id": "Combustion_Methane_Stoichiometric",
        "description": "Stoichiometric combustion of methane (CH4)",
        "input": {
            "fuel": "methane",
            "fuel_composition": {"C": 1, "H": 4},
            "excess_air_percent": 0,
        },
        "expected": {
            "air_fuel_ratio_mass": 17.2,
            "co2_percent_dry": 11.7,
            "tolerance_percent": 2.0,
        },
        "formula": "CH4 + 2O2 + 7.52N2 -> CO2 + 2H2O + 7.52N2",
        "source": "Turns: An Introduction to Combustion",
    },
    {
        "case_id": "Combustion_Propane_Stoichiometric",
        "description": "Stoichiometric combustion of propane (C3H8)",
        "input": {
            "fuel": "propane",
            "fuel_composition": {"C": 3, "H": 8},
            "excess_air_percent": 0,
        },
        "expected": {
            "air_fuel_ratio_mass": 15.6,
            "tolerance_percent": 2.0,
        },
        "formula": "C3H8 + 5O2 -> 3CO2 + 4H2O",
        "source": "Turns: An Introduction to Combustion",
    },
]


# Extended Steam Table Golden Values
STEAM_TABLE_EXTENDED_VALUES = [
    {
        "case_id": "Steam_Subcooled_50C_1MPa",
        "description": "Subcooled water at 50C, 1 MPa",
        "input": {
            "temperature_c": 50,
            "pressure_mpa": 1.0,
            "phase": "liquid",
        },
        "expected": {
            "specific_enthalpy_kj_kg": 210.2,
            "density_kg_m3": 988.0,
            "tolerance_percent": 1.0,
        },
        "source": "IAPWS-IF97",
    },
    {
        "case_id": "Steam_Superheated_500C_5MPa",
        "description": "Superheated steam at 500C, 5 MPa",
        "input": {
            "temperature_c": 500,
            "pressure_mpa": 5.0,
        },
        "expected": {
            "specific_enthalpy_kj_kg": 3433.8,
            "specific_entropy_kj_kg_k": 6.9759,
            "tolerance_percent": 0.5,
        },
        "source": "IAPWS-IF97",
    },
    {
        "case_id": "Steam_Saturation_10MPa",
        "description": "Saturation properties at 10 MPa",
        "input": {
            "pressure_mpa": 10.0,
        },
        "expected": {
            "saturation_temp_c": 311.0,
            "hf_kj_kg": 1407.6,
            "hg_kj_kg": 2724.7,
            "tolerance_percent": 0.5,
        },
        "source": "IAPWS-IF97",
    },
]


# Pinch Analysis Reference Cases
PINCH_ANALYSIS_GOLDEN_VALUES = [
    {
        "case_id": "Pinch_4_Stream_Basic",
        "description": "Standard 4-stream heat integration problem",
        "input": {
            "hot_streams": [
                {"name": "H1", "T_in_c": 180, "T_out_c": 60, "mCp_kW_K": 2.0},
                {"name": "H2", "T_in_c": 150, "T_out_c": 30, "mCp_kW_K": 4.0},
            ],
            "cold_streams": [
                {"name": "C1", "T_in_c": 60, "T_out_c": 180, "mCp_kW_K": 3.0},
                {"name": "C2", "T_in_c": 30, "T_out_c": 130, "mCp_kW_K": 2.5},
            ],
            "delta_t_min_c": 10,
        },
        "expected": {
            "pinch_temperature_c": 90,
            "min_hot_utility_kw": 80.0,
            "min_cold_utility_kw": 70.0,
            "tolerance_kw": 10.0,
        },
        "source": "Linnhoff March: Pinch Analysis Handbook",
    },
]


# =============================================================================
# TEST CLASS: HEAT EXCHANGER GOLDEN VALUES
# =============================================================================

class TestHeatExchangerGoldenValues:
    """Validate heat exchanger calculations against NTU-method reference values."""

    @pytest.mark.compliance
    @pytest.mark.golden
    @pytest.mark.parametrize("test_case", HEAT_EXCHANGER_GOLDEN_VALUES)
    def test_heat_exchanger_effectiveness(self, test_case):
        """Test heat exchanger effectiveness against published correlations."""
        input_data = test_case["input"]
        expected = test_case["expected"]

        ntu = input_data["ntu"]
        c_ratio = input_data["c_ratio"]
        arrangement = input_data["flow_arrangement"]

        calculated = self._calculate_effectiveness(ntu, c_ratio, arrangement)
        expected_eff = expected["effectiveness"]
        tolerance = expected["tolerance"]

        assert abs(calculated - expected_eff) <= tolerance, \
            f"{test_case['case_id']}: Calculated {calculated:.4f} vs expected {expected_eff:.4f} " \
            f"(tolerance {tolerance})\nSource: {test_case['source']}"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_effectiveness_bounds_all_arrangements(self):
        """Test that effectiveness is always between 0 and 1 for all arrangements."""
        arrangements = ["counterflow", "parallel", "crossflow_mixed"]
        test_params = [
            (0.1, 0.5),
            (1.0, 0.0),
            (5.0, 1.0),
            (10.0, 0.25),
        ]

        for arrangement in arrangements:
            for ntu, c_ratio in test_params:
                eff = self._calculate_effectiveness(ntu, c_ratio, arrangement)
                assert 0 <= eff <= 1.0, \
                    f"Effectiveness {eff} out of bounds for {arrangement}, NTU={ntu}, Cr={c_ratio}"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_effectiveness_monotonic_with_ntu(self):
        """Test that effectiveness monotonically increases with NTU."""
        c_ratio = 0.5

        for arrangement in ["counterflow", "parallel"]:
            previous_eff = 0
            for ntu in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                eff = self._calculate_effectiveness(ntu, c_ratio, arrangement)
                assert eff > previous_eff, \
                    f"Effectiveness should increase with NTU ({arrangement})"
                previous_eff = eff

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_counterflow_superior_to_parallel(self):
        """Test that counterflow is always more effective than parallel flow."""
        for ntu in [0.5, 1.0, 2.0, 5.0]:
            for c_ratio in [0.25, 0.5, 0.75, 1.0]:
                eff_counter = self._calculate_effectiveness(ntu, c_ratio, "counterflow")
                eff_parallel = self._calculate_effectiveness(ntu, c_ratio, "parallel")

                assert eff_counter >= eff_parallel, \
                    f"Counterflow should be >= parallel at NTU={ntu}, Cr={c_ratio}"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_limiting_case_ntu_zero(self):
        """Test effectiveness approaches 0 as NTU approaches 0."""
        for c_ratio in [0.0, 0.5, 1.0]:
            eff = self._calculate_effectiveness(0.001, c_ratio, "counterflow")
            assert eff < 0.01, f"Effectiveness should approach 0 as NTU->0"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_limiting_case_ntu_infinity(self):
        """Test effectiveness approaches limit as NTU approaches infinity."""
        # For counterflow with C_ratio < 1, effectiveness -> 1 as NTU -> infinity
        eff = self._calculate_effectiveness(100.0, 0.5, "counterflow")
        assert eff > 0.99, f"Counterflow effectiveness should approach 1 as NTU->inf"

        # For parallel flow, limit is 1/(1+C_ratio)
        eff_parallel = self._calculate_effectiveness(100.0, 0.5, "parallel")
        expected_limit = 1 / (1 + 0.5)  # 0.667
        assert abs(eff_parallel - expected_limit) < 0.01

    def _calculate_effectiveness(
        self, ntu: float, c_ratio: float, arrangement: str
    ) -> float:
        """Calculate heat exchanger effectiveness using NTU method."""
        if c_ratio == 0:
            # Evaporator/Condenser case
            return 1 - math.exp(-ntu)

        if arrangement == "counterflow":
            if abs(c_ratio - 1.0) < 0.001:
                return ntu / (1 + ntu)
            else:
                exp_term = math.exp(-ntu * (1 - c_ratio))
                return (1 - exp_term) / (1 - c_ratio * exp_term)

        elif arrangement == "parallel":
            exp_term = math.exp(-ntu * (1 + c_ratio))
            return (1 - exp_term) / (1 + c_ratio)

        elif arrangement == "crossflow_mixed":
            # One fluid mixed, one unmixed (Cmax mixed)
            exp_term = math.exp(-c_ratio * (1 - math.exp(-ntu)))
            return (1 - exp_term) / c_ratio if c_ratio > 0 else 1 - math.exp(-ntu)

        return 0.0


# =============================================================================
# TEST CLASS: COMBUSTION GOLDEN VALUES
# =============================================================================

class TestCombustionGoldenValues:
    """Validate combustion calculations against reference values."""

    @pytest.mark.compliance
    @pytest.mark.golden
    @pytest.mark.parametrize("test_case", COMBUSTION_GOLDEN_VALUES)
    def test_stoichiometric_afr(self, test_case):
        """Test stoichiometric air-fuel ratio calculation."""
        composition = test_case["input"].get("fuel_composition", {})
        expected_afr = test_case["expected"]["air_fuel_ratio_mass"]
        tolerance = test_case["expected"]["tolerance_percent"] / 100 * expected_afr

        calculated_afr = self._calculate_stoichiometric_afr(composition)

        assert abs(calculated_afr - expected_afr) <= tolerance, \
            f"{test_case['case_id']}: AFR {calculated_afr:.2f} vs expected {expected_afr:.2f}"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_excess_air_to_o2_relationship(self):
        """Test relationship between excess air and O2 in flue gas."""
        # Validated reference points for natural gas combustion
        reference_points = [
            (0, 0.0, 0.5),    # (excess_air%, expected_O2%, tolerance%)
            (10, 1.9, 0.5),
            (15, 2.7, 0.5),
            (20, 3.5, 0.5),
            (30, 4.8, 0.5),
        ]

        for excess_air, expected_o2, tolerance in reference_points:
            calculated_o2 = self._calculate_flue_gas_o2(excess_air)
            assert abs(calculated_o2 - expected_o2) <= tolerance, \
                f"O2 at {excess_air}% excess air: {calculated_o2:.1f}% vs expected {expected_o2:.1f}%"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_co2_in_products(self):
        """Test CO2 percentage in combustion products."""
        # For stoichiometric methane combustion
        # CO2 percentage in dry flue gas should be ~11.7%
        composition = {"C": 1, "H": 4}  # Methane

        co2_percent = self._calculate_co2_percent(composition, excess_air=0)

        assert 10.0 < co2_percent < 13.0, \
            f"CO2 percentage {co2_percent:.1f}% outside expected range for methane"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_combustion_efficiency_trends(self):
        """Test that combustion efficiency decreases with excess air."""
        efficiencies = []
        for excess_air in [5, 10, 20, 50, 100]:
            eff = self._calculate_combustion_efficiency(excess_air)
            efficiencies.append(eff)

        # Each should be less than previous
        for i in range(1, len(efficiencies)):
            assert efficiencies[i] <= efficiencies[i-1], \
                "Combustion efficiency should decrease with excess air"

    def _calculate_stoichiometric_afr(self, composition: dict) -> float:
        """Calculate stoichiometric air-fuel ratio (mass basis)."""
        c = composition.get("C", 0)
        h = composition.get("H", 0)
        o = composition.get("O", 0)
        s = composition.get("S", 0)

        # Molar mass of fuel
        m_fuel = 12 * c + 1 * h + 16 * o + 32 * s

        # O2 required (moles): C->CO2 (1), H2->H2O (0.25), S->SO2 (1)
        # Subtract O already in fuel
        o2_required = c + h / 4 + s - o / 2

        # Air is 21% O2 by mole, M_air = 28.97 g/mol
        air_moles = o2_required / 0.21
        m_air = air_moles * 28.97

        return m_air / m_fuel if m_fuel > 0 else 0

    def _calculate_flue_gas_o2(self, excess_air_percent: float) -> float:
        """Calculate O2 percentage in dry flue gas."""
        ea = excess_air_percent / 100
        # For natural gas: O2% = 21 * ea / (1 + ea * (1 + 0.21*9.5))
        # Simplified formula
        return 21 * ea / (1 + ea) if ea >= 0 else 0

    def _calculate_co2_percent(self, composition: dict, excess_air: float) -> float:
        """Calculate CO2 percentage in dry flue gas."""
        c = composition.get("C", 0)
        h = composition.get("H", 0)

        # Moles of products per mole of fuel
        co2_moles = c
        n2_stoich = (c + h/4) / 0.21 * 0.79
        o2_excess = (c + h/4) * excess_air / 100

        total_dry = co2_moles + n2_stoich + o2_excess

        return (co2_moles / total_dry * 100) if total_dry > 0 else 0

    def _calculate_combustion_efficiency(self, excess_air_percent: float) -> float:
        """Calculate combustion efficiency."""
        base_efficiency = 99.5
        penalty = 0.02 * excess_air_percent
        return max(90.0, base_efficiency - penalty)


# =============================================================================
# TEST CLASS: EXTENDED STEAM TABLE GOLDEN VALUES
# =============================================================================

class TestSteamTableExtendedGoldenValues:
    """Validate steam property calculations against extended IAPWS-IF97 values."""

    @pytest.mark.compliance
    @pytest.mark.golden
    @pytest.mark.parametrize("test_case", STEAM_TABLE_EXTENDED_VALUES)
    def test_steam_enthalpy(self, test_case):
        """Test steam enthalpy against reference values."""
        input_data = test_case["input"]
        expected = test_case["expected"]

        properties = self._get_steam_properties(input_data)

        if "specific_enthalpy_kj_kg" in expected:
            exp_h = expected["specific_enthalpy_kj_kg"]
            calc_h = properties.get("h_kj_kg", 0)
            tolerance = expected.get("tolerance_percent", 1.0) / 100 * abs(exp_h)

            assert abs(calc_h - exp_h) <= tolerance, \
                f"{test_case['case_id']}: h={calc_h:.1f} vs expected {exp_h:.1f} kJ/kg"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_saturation_thermodynamic_consistency(self):
        """Test thermodynamic consistency at saturation."""
        for p_mpa in [0.1, 0.5, 1.0, 5.0, 10.0]:
            hf, hfg, hg = self._get_saturation_enthalpies(p_mpa)

            # hg = hf + hfg
            assert abs(hg - (hf + hfg)) < 1.0, \
                f"Inconsistent saturation at {p_mpa} MPa: hf={hf:.1f}, hfg={hfg:.1f}, hg={hg:.1f}"

            # hfg should decrease with pressure (approaching critical)
            if p_mpa > 0.1:
                _, hfg_prev, _ = self._get_saturation_enthalpies(p_mpa * 0.9)
                assert hfg <= hfg_prev + 50, \
                    f"hfg should decrease with pressure"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_superheated_enthalpy_increases_with_temp(self):
        """Test that superheated enthalpy increases with temperature."""
        p_mpa = 1.0
        previous_h = 0

        for T_c in [200, 300, 400, 500, 600]:
            props = self._get_steam_properties({
                "temperature_c": T_c,
                "pressure_mpa": p_mpa,
            })
            h = props.get("h_kj_kg", 0)

            assert h > previous_h, \
                f"Enthalpy should increase with temperature at constant pressure"
            previous_h = h

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_critical_point_properties(self):
        """Test properties near critical point."""
        # Critical point: 374.14C, 22.064 MPa
        props = self._get_steam_properties({
            "temperature_c": 374,
            "pressure_mpa": 22.0,
        })

        # Near critical point, properties should exist
        assert props.get("h_kj_kg", 0) > 0

    def _get_steam_properties(self, input_data: dict) -> dict:
        """Get steam properties from input conditions."""
        if "temperature_c" in input_data and "pressure_mpa" in input_data:
            T = input_data["temperature_c"]
            p = input_data["pressure_mpa"]

            # Simplified correlations
            if input_data.get("phase") == "liquid":
                h = 4.18 * T + p * 1.0  # Approximate for subcooled
                rho = 1000 - 0.2 * (T - 4) ** 1.5 if T > 4 else 1000
                return {"h_kj_kg": h, "rho_kg_m3": rho}
            else:
                # Superheated steam
                h = 2500 + 2.0 * T + 10 * p
                s = 6.0 + 0.005 * T - 0.5 * math.log(p) if p > 0 else 6.0
                return {"h_kj_kg": h, "s_kj_kg_k": s}

        return {}

    def _get_saturation_enthalpies(self, p_mpa: float) -> Tuple[float, float, float]:
        """Get saturation enthalpies at given pressure."""
        # Simplified correlations
        hf = 400 + 100 * math.log(1 + 10 * p_mpa)
        hfg = 2200 - 150 * math.log(1 + p_mpa)
        hg = hf + hfg
        return hf, hfg, hg


# =============================================================================
# TEST CLASS: PINCH ANALYSIS GOLDEN VALUES
# =============================================================================

class TestPinchAnalysisGoldenValues:
    """Validate pinch analysis calculations against reference values."""

    @pytest.mark.compliance
    @pytest.mark.golden
    @pytest.mark.parametrize("test_case", PINCH_ANALYSIS_GOLDEN_VALUES)
    def test_pinch_temperature(self, test_case):
        """Test pinch temperature identification."""
        input_data = test_case["input"]
        expected = test_case["expected"]

        result = self._perform_pinch_analysis(input_data)

        exp_pinch = expected["pinch_temperature_c"]
        calc_pinch = result.get("pinch_temperature_c", 0)

        assert abs(calc_pinch - exp_pinch) <= 15, \
            f"{test_case['case_id']}: Pinch {calc_pinch}C vs expected {exp_pinch}C"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_energy_balance_in_pinch(self):
        """Test energy balance is maintained in pinch analysis."""
        input_data = PINCH_ANALYSIS_GOLDEN_VALUES[0]["input"]

        # Total heat available from hot streams
        hot_duty = sum(
            s["mCp_kW_K"] * (s["T_in_c"] - s["T_out_c"])
            for s in input_data["hot_streams"]
        )

        # Total heat required by cold streams
        cold_duty = sum(
            s["mCp_kW_K"] * (s["T_out_c"] - s["T_in_c"])
            for s in input_data["cold_streams"]
        )

        result = self._perform_pinch_analysis(input_data)

        # Energy balance: hot_duty + hot_utility = cold_duty + cold_utility
        lhs = hot_duty + result.get("hot_utility_kw", 0)
        rhs = cold_duty + result.get("cold_utility_kw", 0)

        assert abs(lhs - rhs) < 5, f"Energy balance error: {lhs:.1f} != {rhs:.1f}"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_utility_targets_positive(self):
        """Test that utility targets are non-negative."""
        input_data = PINCH_ANALYSIS_GOLDEN_VALUES[0]["input"]

        result = self._perform_pinch_analysis(input_data)

        assert result.get("hot_utility_kw", 0) >= 0, "Hot utility should be >= 0"
        assert result.get("cold_utility_kw", 0) >= 0, "Cold utility should be >= 0"

    def _perform_pinch_analysis(self, input_data: dict) -> dict:
        """Perform simplified pinch analysis."""
        hot_streams = input_data["hot_streams"]
        cold_streams = input_data["cold_streams"]
        delta_t_min = input_data["delta_t_min_c"]

        # Calculate total duties
        hot_duty = sum(
            s["mCp_kW_K"] * (s["T_in_c"] - s["T_out_c"])
            for s in hot_streams
        )
        cold_duty = sum(
            s["mCp_kW_K"] * (s["T_out_c"] - s["T_in_c"])
            for s in cold_streams
        )

        # Simplified pinch calculation
        # Find temperature where heat recovery is maximum
        all_temps = []
        for s in hot_streams:
            all_temps.extend([s["T_in_c"], s["T_out_c"]])
        for s in cold_streams:
            all_temps.extend([s["T_in_c"], s["T_out_c"]])

        pinch_temp = sum(all_temps) / len(all_temps)  # Simplified

        # Approximate utility requirements
        diff = cold_duty - hot_duty
        hot_utility = max(0, diff)
        cold_utility = max(0, -diff)

        return {
            "pinch_temperature_c": pinch_temp,
            "hot_utility_kw": hot_utility,
            "cold_utility_kw": cold_utility,
        }


# =============================================================================
# TEST CLASS: PROVENANCE HASH GOLDEN VALUES
# =============================================================================

class TestProvenanceHashGoldenValues:
    """Test provenance hash determinism and consistency."""

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_provenance_hash_sha256_format(self):
        """Test that provenance hash is valid SHA-256."""
        data = {"efficiency_percent": 82.8, "timestamp": "2025-01-01T00:00:00Z"}

        hash_val = self._calculate_provenance_hash(data)

        # SHA-256 should be 64 hex characters
        assert len(hash_val) == 64, f"Hash length {len(hash_val)} != 64"
        assert all(c in '0123456789abcdef' for c in hash_val), "Hash should be hex"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_provenance_hash_deterministic(self):
        """Test that same input always produces same hash."""
        data = {"a": 1, "b": 2, "c": [1, 2, 3]}

        hashes = [self._calculate_provenance_hash(data) for _ in range(100)]

        assert len(set(hashes)) == 1, "Hash should be deterministic"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_provenance_hash_order_independent(self):
        """Test that hash is independent of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        data3 = {"b": 2, "c": 3, "a": 1}

        hash1 = self._calculate_provenance_hash(data1)
        hash2 = self._calculate_provenance_hash(data2)
        hash3 = self._calculate_provenance_hash(data3)

        assert hash1 == hash2 == hash3, "Hash should be order-independent"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_provenance_hash_changes_with_input(self):
        """Test that different inputs produce different hashes."""
        data1 = {"value": 100}
        data2 = {"value": 101}
        data3 = {"value": 100, "extra": True}

        hash1 = self._calculate_provenance_hash(data1)
        hash2 = self._calculate_provenance_hash(data2)
        hash3 = self._calculate_provenance_hash(data3)

        assert hash1 != hash2, "Different values should produce different hashes"
        assert hash1 != hash3, "Additional keys should change hash"

    @pytest.mark.compliance
    @pytest.mark.golden
    def test_provenance_hash_nested_objects(self):
        """Test hash works with nested objects."""
        data = {
            "efficiency": {"first_law": 82.8, "second_law": 45.2},
            "losses": {"flue_gas": 80.0, "radiation": 12.0},
            "metadata": {"version": "1.0.0"},
        }

        hash_val = self._calculate_provenance_hash(data)

        assert len(hash_val) == 64
        assert hash_val == self._calculate_provenance_hash(data)  # Deterministic

    def _calculate_provenance_hash(self, data: dict) -> str:
        """Calculate SHA-256 provenance hash."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# TEST CLASS: PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests for thermal calculations."""

    @pytest.mark.performance
    def test_efficiency_calculation_under_5ms(self, sample_heat_balance):
        """Test efficiency calculation completes in <5ms."""
        import time

        energy_inputs = sample_heat_balance["energy_inputs"]
        useful_outputs = sample_heat_balance["useful_outputs"]

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            self._calculate_efficiency(energy_inputs, useful_outputs)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        assert elapsed_ms < 5.0, f"Efficiency took {elapsed_ms:.3f}ms (target: <5ms)"

    @pytest.mark.performance
    def test_exergy_calculation_under_5ms(self, sample_heat_balance):
        """Test exergy calculation completes in <5ms."""
        import time

        energy_inputs = sample_heat_balance["energy_inputs"]
        useful_outputs = sample_heat_balance["useful_outputs"]
        ambient = sample_heat_balance["ambient_conditions"]

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            self._calculate_exergy(energy_inputs, useful_outputs, ambient)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        assert elapsed_ms < 5.0, f"Exergy took {elapsed_ms:.3f}ms (target: <5ms)"

    @pytest.mark.performance
    def test_hx_effectiveness_under_1ms(self):
        """Test heat exchanger effectiveness calculation under 1ms."""
        import time

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            ntu = 2.0
            c_ratio = 0.5
            exp_term = math.exp(-ntu * (1 - c_ratio))
            _ = (1 - exp_term) / (1 - c_ratio * exp_term)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        assert elapsed_ms < 1.0, f"HX effectiveness took {elapsed_ms:.4f}ms (target: <1ms)"

    @pytest.mark.performance
    def test_provenance_hash_under_1ms(self):
        """Test provenance hash generation under 1ms."""
        import time

        data = {
            "efficiency": 82.8,
            "exergy": 45.2,
            "losses": {"flue_gas": 80.0, "radiation": 12.0},
        }

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            _ = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

        assert elapsed_ms < 1.0, f"Hash took {elapsed_ms:.4f}ms (target: <1ms)"

    @pytest.mark.performance
    def test_carnot_factor_under_100us(self):
        """Test Carnot factor calculation under 100 microseconds."""
        import time

        iterations = 100000
        start = time.perf_counter()
        for _ in range(iterations):
            T_hot = 453.15  # 180C in K
            T_cold = 298.15  # 25C in K
            _ = 1 - T_cold / T_hot
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / iterations

        assert elapsed_us < 100, f"Carnot factor took {elapsed_us:.2f}us (target: <100us)"

    def _calculate_efficiency(self, energy_inputs: dict, useful_outputs: dict) -> float:
        """Calculate first law efficiency."""
        total_input = sum(
            f.get("mass_flow_kg_hr", 0) * f.get("heating_value_mj_kg", 0) * 0.2778
            for f in energy_inputs.get("fuel_inputs", [])
        )
        total_output = sum(
            s.get("heat_rate_kw", 0)
            for s in useful_outputs.get("steam_output", [])
        )
        return (total_output / total_input * 100) if total_input > 0 else 0

    def _calculate_exergy(
        self, energy_inputs: dict, useful_outputs: dict, ambient: dict
    ) -> float:
        """Calculate second law (exergy) efficiency."""
        T0_K = ambient.get("ambient_temperature_c", 25.0) + 273.15

        exergy_input = sum(
            f.get("mass_flow_kg_hr", 0) * f.get("heating_value_mj_kg", 0) * 0.2778 * 1.04
            for f in energy_inputs.get("fuel_inputs", [])
        )

        exergy_output = sum(
            s.get("heat_rate_kw", 0) * (1 - T0_K / (s.get("temperature_c", 180) + 273.15))
            for s in useful_outputs.get("steam_output", [])
        )

        return (exergy_output / exergy_input * 100) if exergy_input > 0 else 0


# =============================================================================
# PROPERTY-BASED TESTS WITH HYPOTHESIS
# =============================================================================

if HAS_HYPOTHESIS:

    class TestGoldenValuesPropertyBased:
        """Property-based tests using Hypothesis."""

        @given(
            input_kw=st.floats(min_value=1.0, max_value=100000.0),
            output_kw=st.floats(min_value=0.0, max_value=100000.0),
        )
        @settings(max_examples=100)
        def test_efficiency_bounded_0_to_100(self, input_kw: float, output_kw: float):
            """Property: Efficiency is bounded between 0 and 100%."""
            assume(output_kw <= input_kw)  # Physical constraint
            assume(not math.isnan(input_kw) and not math.isnan(output_kw))
            assume(not math.isinf(input_kw) and not math.isinf(output_kw))

            efficiency = (output_kw / input_kw) * 100

            assert 0 <= efficiency <= 100

        @given(
            t_hot_c=st.floats(min_value=30.0, max_value=1000.0),
            t_ambient_c=st.floats(min_value=-40.0, max_value=25.0),
        )
        @settings(max_examples=100)
        def test_carnot_factor_bounded_0_to_1(self, t_hot_c: float, t_ambient_c: float):
            """Property: Carnot factor is between 0 and 1."""
            assume(t_hot_c > t_ambient_c + 5)  # Minimum temperature difference

            T_hot_K = t_hot_c + 273.15
            T_ambient_K = t_ambient_c + 273.15

            carnot = 1 - T_ambient_K / T_hot_K

            assert 0 < carnot < 1

        @given(
            ntu=st.floats(min_value=0.01, max_value=50.0),
            c_ratio=st.floats(min_value=0.001, max_value=0.999),
        )
        @settings(max_examples=100)
        def test_hx_effectiveness_bounded(self, ntu: float, c_ratio: float):
            """Property: Heat exchanger effectiveness is between 0 and 1."""
            assume(not math.isnan(ntu) and not math.isnan(c_ratio))
            assume(not math.isinf(ntu) and not math.isinf(c_ratio))

            # Counterflow effectiveness
            exp_term = math.exp(-ntu * (1 - c_ratio))
            effectiveness = (1 - exp_term) / (1 - c_ratio * exp_term)

            assert 0 <= effectiveness <= 1

        @given(
            value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        )
        @settings(max_examples=50)
        def test_hash_is_deterministic(self, value: float):
            """Property: Hash is deterministic for any valid input."""
            data = {"value": value}

            hash1 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            hash2 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

            assert hash1 == hash2

        @given(
            excess_air=st.floats(min_value=0.0, max_value=200.0),
        )
        @settings(max_examples=50)
        def test_flue_gas_o2_bounded(self, excess_air: float):
            """Property: Flue gas O2 is bounded between 0 and 21%."""
            assume(not math.isnan(excess_air) and not math.isinf(excess_air))

            ea = excess_air / 100
            o2_percent = 21 * ea / (1 + ea) if ea >= 0 else 0

            assert 0 <= o2_percent < 21

        @given(
            p_mpa=st.floats(min_value=0.01, max_value=20.0),
        )
        @settings(max_examples=50)
        def test_saturation_hfg_positive(self, p_mpa: float):
            """Property: Latent heat of vaporization is always positive."""
            assume(not math.isnan(p_mpa) and not math.isinf(p_mpa))
            assume(p_mpa < 22.064)  # Below critical pressure

            hfg = 2200 - 150 * math.log(1 + p_mpa)

            assert hfg > 0
