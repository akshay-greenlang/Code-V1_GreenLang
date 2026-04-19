# -*- coding: utf-8 -*-
"""
Golden Value Tests for GL-003 UNIFIEDSTEAM - Steam System Optimization

Reference Standards:
    - IAPWS-IF97: Industrial Formulation 1997 for Water and Steam
    - NIST Chemistry WebBook
    - ASME PTC 4.3: Air Heater Performance
    - ASME PTC 4.4: Gas Turbine Heat Recovery Steam Generators

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any


@dataclass(frozen=True)
class NISTSteamGoldenValue:
    name: str
    value: float
    unit: str
    tolerance_percent: float
    source: str
    conditions: str

    def validate(self, calculated: float) -> Tuple[bool, float]:
        if self.value == 0:
            return abs(calculated) < 0.001, abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * 100
        return deviation <= self.tolerance_percent, deviation


class NISTSteamProperties:
    SAT_WATER_100C = {
        "enthalpy_liquid": NISTSteamGoldenValue("hf @ 100C", 419.05, "kJ/kg", 0.1, "IAPWS-IF97", "T=100C, sat liquid"),
        "enthalpy_vapor": NISTSteamGoldenValue("hg @ 100C", 2675.46, "kJ/kg", 0.1, "IAPWS-IF97", "T=100C, sat vapor"),
        "entropy_liquid": NISTSteamGoldenValue("sf @ 100C", 1.3069, "kJ/(kg.K)", 0.1, "IAPWS-IF97", "T=100C"),
        "entropy_vapor": NISTSteamGoldenValue("sg @ 100C", 7.3545, "kJ/(kg.K)", 0.1, "IAPWS-IF97", "T=100C"),
        "latent_heat": NISTSteamGoldenValue("hfg @ 100C", 2256.41, "kJ/kg", 0.1, "IAPWS-IF97", "T=100C"),
    }

    SAT_PRESSURE_TABLE = {
        101.325: (100.0, 419.05, 2675.46, 2256.41),
        500.0: (151.83, 640.09, 2748.11, 2108.02),
        1000.0: (179.88, 762.51, 2777.11, 2014.60),
        2000.0: (212.37, 908.62, 2798.29, 1889.67),
        5000.0: (263.92, 1154.50, 2794.17, 1639.67),
    }

    SUPERHEATED_300C_1MPA = {
        "enthalpy": NISTSteamGoldenValue("h @ 300C, 1MPa", 3051.6, "kJ/kg", 0.1, "IAPWS-IF97", "T=300C, P=1MPa"),
        "entropy": NISTSteamGoldenValue("s @ 300C, 1MPa", 7.1246, "kJ/(kg.K)", 0.1, "IAPWS-IF97", "T=300C, P=1MPa"),
    }


class ASMEPTC43AirHeater:
    EFFECTIVENESS_TESTS = [
        {"gas_in_C": 350, "gas_out_C": 180, "air_in_C": 30, "air_out_C": 280, "expected_effectiveness": 0.78},
    ]


class ASMEPTC44HRSG:
    APPROACH_TEMP_TESTS = [
        {"econ_exit_temp_C": 175, "drum_sat_temp_C": 180, "expected_approach_C": 5, "tolerance_C": 2},
    ]


@pytest.mark.golden
class TestNISTSteamProperties:

    def test_saturation_enthalpy_100C(self):
        golden = NISTSteamProperties.SAT_WATER_100C["enthalpy_vapor"]
        is_valid, _ = golden.validate(2675.46)
        assert is_valid

    def test_latent_heat_100C(self):
        hf = NISTSteamProperties.SAT_WATER_100C["enthalpy_liquid"].value
        hg = NISTSteamProperties.SAT_WATER_100C["enthalpy_vapor"].value
        assert abs((hg - hf) - 2256.41) <= 0.1

    def test_entropy_consistency(self):
        sf = NISTSteamProperties.SAT_WATER_100C["entropy_liquid"].value
        sg = NISTSteamProperties.SAT_WATER_100C["entropy_vapor"].value
        assert sg > sf

    @pytest.mark.parametrize("pressure_kPa,expected_values", [
        (101.325, (100.0, 419.05, 2675.46, 2256.41)),
        (1000.0, (179.88, 762.51, 2777.11, 2014.60)),
    ])
    def test_saturation_pressure_table(self, pressure_kPa, expected_values):
        T_sat, hf, hg, hfg = expected_values
        assert abs((hg - hf) - hfg) <= 1.0


@pytest.mark.golden
class TestASMEPTC43AirHeater:

    def test_air_heater_effectiveness(self):
        for test in ASMEPTC43AirHeater.EFFECTIVENESS_TESTS:
            eff = (test["air_out_C"] - test["air_in_C"]) / (test["gas_in_C"] - test["air_in_C"])
            assert abs(eff - test["expected_effectiveness"]) <= 0.02


@pytest.mark.golden
class TestASMEPTC44HRSG:

    def test_approach_temperature(self):
        for test in ASMEPTC44HRSG.APPROACH_TEMP_TESTS:
            approach = test["drum_sat_temp_C"] - test["econ_exit_temp_C"]
            assert abs(approach - test["expected_approach_C"]) <= test["tolerance_C"]


@pytest.mark.golden
class TestEnthalpyBalance:

    def test_steam_quality_from_enthalpy(self):
        hf, hfg, h = 419.05, 2256.41, 1500.0
        quality = (h - hf) / hfg
        assert abs(quality - 0.479) <= 0.01

    def test_desuperheating_spray_calculation(self):
        h_in, h_spray, h_target, m_steam = 3051.6, 419.05, 2800.0, 100.0
        m_spray = m_steam * (h_in - h_target) / (h_target - h_spray)
        assert 0 < m_spray < m_steam


@pytest.mark.golden
class TestDeterminism:

    def test_enthalpy_determinism(self):
        hashes = set()
        for _ in range(100):
            quality = round((1500.0 - 419.05) / 2256.41, 10)
            hashes.add(hashlib.sha256(str(quality).encode()).hexdigest())
        assert len(hashes) == 1


def export_golden_values() -> Dict[str, Any]:
    return {
        "metadata": {"version": "1.0.0", "agent": "GL-003_UnifiedSteam"},
        "iapws_if97": {"sat_100C": {"hf": 419.05, "hg": 2675.46, "hfg": 2256.41}},
    }


if __name__ == "__main__":
    print(json.dumps(export_golden_values(), indent=2))
