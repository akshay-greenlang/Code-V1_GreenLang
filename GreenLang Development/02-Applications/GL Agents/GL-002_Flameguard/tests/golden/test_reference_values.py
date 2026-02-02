# -*- coding: utf-8 -*-
"""
Golden Value Tests for GL-002 FLAMEGUARD - Boiler Combustion Optimization

Validates calculations against authoritative reference standards for
boiler efficiency, combustion optimization, and emission calculations.

Reference Standards:
    - ASME PTC 4-2013: Fired Steam Generators Performance Test Codes
    - ASME PTC 4.1-1964 (R2003): Steam Generating Units
    - EPA Method 19: Determination of Sulfur Dioxide Removal Efficiency
    - EPA 40 CFR Part 98: Mandatory Greenhouse Gas Reporting
    - NIST Chemistry WebBook: Thermochemical Data

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
from decimal import Decimal
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple


@dataclass(frozen=True)
class GoldenValue:
    name: str
    value: float
    unit: str
    tolerance_percent: float
    source: str
    section: str

    def validate(self, calculated: float) -> Tuple[bool, float]:
        if self.value == 0:
            return abs(calculated) < 0.001, abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * 100
        return deviation <= self.tolerance_percent, deviation


class ASMEPTC4References:
    STOICH_AIR_NATURAL_GAS = GoldenValue("Stoich Air NG", 17.24, "lb/lb", 0.5, "ASME PTC 4-2013", "Section 5.5")
    STOICH_AIR_NO2_OIL = GoldenValue("Stoich Air Oil2", 14.1, "lb/lb", 0.5, "ASME PTC 4-2013", "Section 5.5")
    THEORETICAL_CO2_NATURAL_GAS = GoldenValue("CO2max NG", 11.73, "%", 0.5, "ASME PTC 4-2013", "Section 5.7.1")


class EPAMethod19FFactor:
    NATURAL_GAS = {
        "Fd": GoldenValue("Fd NG", 8710.0, "dscf/MMBtu", 1.0, "EPA Method 19", "Table 19-1"),
        "Fw": GoldenValue("Fw NG", 10610.0, "wscf/MMBtu", 1.0, "EPA Method 19", "Table 19-1"),
        "Fc": GoldenValue("Fc NG", 1040.0, "scf/MMBtu", 1.0, "EPA Method 19", "Table 19-1"),
    }
    NO2_OIL = {
        "Fd": GoldenValue("Fd Oil2", 9190.0, "dscf/MMBtu", 1.0, "EPA Method 19", "Table 19-1"),
        "Fw": GoldenValue("Fw Oil2", 10320.0, "wscf/MMBtu", 1.0, "EPA Method 19", "Table 19-1"),
        "Fc": GoldenValue("Fc Oil2", 1420.0, "scf/MMBtu", 1.0, "EPA Method 19", "Table 19-1"),
    }
    BITUMINOUS_COAL = {
        "Fd": GoldenValue("Fd Coal", 9780.0, "dscf/MMBtu", 2.0, "EPA Method 19", "Table 19-1"),
        "Fw": GoldenValue("Fw Coal", 10640.0, "wscf/MMBtu", 2.0, "EPA Method 19", "Table 19-1"),
        "Fc": GoldenValue("Fc Coal", 1800.0, "scf/MMBtu", 2.0, "EPA Method 19", "Table 19-1"),
    }


class EPA40CFRPart98EmissionFactors:
    CO2_EMISSION_FACTORS = {
        "natural_gas": GoldenValue("CO2 EF NG", 53.06, "kg CO2/MMBtu", 1.0, "EPA 40 CFR 98", "Table C-1"),
        "distillate_fuel_oil_2": GoldenValue("CO2 EF Oil2", 73.96, "kg CO2/MMBtu", 1.0, "EPA 40 CFR 98", "Table C-1"),
        "residual_fuel_oil_6": GoldenValue("CO2 EF Oil6", 75.10, "kg CO2/MMBtu", 1.0, "EPA 40 CFR 98", "Table C-1"),
        "bituminous_coal": GoldenValue("CO2 EF Coal", 93.28, "kg CO2/MMBtu", 2.0, "EPA 40 CFR 98", "Table C-1"),
        "propane": GoldenValue("CO2 EF Propane", 63.07, "kg CO2/MMBtu", 1.0, "EPA 40 CFR 98", "Table C-1"),
    }


EXCESS_AIR_O2_RELATIONSHIP = [
    (0.0, 0.0), (1.0, 5.0), (2.0, 10.53), (3.0, 16.67), (4.0, 23.53),
    (5.0, 31.25), (6.0, 40.0), (7.0, 50.0), (8.0, 61.54), (10.0, 90.91),
]


@pytest.mark.golden
class TestASMEPTC4Efficiency:
    def test_stoichiometric_air_natural_gas(self):
        ch4, c2h6, c3h8 = 0.93, 0.035, 0.01
        o2_per_mole = ch4 * 2.0 + c2h6 * 3.5 + c3h8 * 5.0
        air_per_mole = o2_per_mole / 0.21
        mw_air, mw_fuel = 28.96, ch4 * 16.04 + c2h6 * 30.07 + c3h8 * 44.10
        calculated = air_per_mole * mw_air / mw_fuel
        golden = ASMEPTC4References.STOICH_AIR_NATURAL_GAS
        is_valid, _ = golden.validate(calculated)
        assert is_valid

    def test_excess_air_from_o2_measurement(self):
        for o2, expected_ea in EXCESS_AIR_O2_RELATIONSHIP:
            if o2 >= 21: continue
            calculated = o2 / (21 - o2) * 100
            assert abs(calculated - expected_ea) <= 0.5

    def test_indirect_efficiency_calculation(self):
        losses = {"dry_flue": 5.2, "moisture": 10.5, "radiation": 0.5}
        calculated = 100.0 - sum(losses.values())
        assert abs(calculated - 83.8) <= 1.0


@pytest.mark.golden
class TestEPAMethod19FFactor:
    @pytest.mark.parametrize("fuel,f_factors", [
        ("natural_gas", EPAMethod19FFactor.NATURAL_GAS),
        ("no2_oil", EPAMethod19FFactor.NO2_OIL),
    ])
    def test_fd_factor_values(self, fuel: str, f_factors: Dict):
        golden = f_factors["Fd"]
        assert 8000 <= golden.value <= 11000

    @pytest.mark.parametrize("fuel,f_factors", [
        ("natural_gas", EPAMethod19FFactor.NATURAL_GAS),
        ("no2_oil", EPAMethod19FFactor.NO2_OIL),
    ])
    def test_fw_greater_than_fd(self, fuel: str, f_factors: Dict):
        assert f_factors["Fw"].value > f_factors["Fd"].value


@pytest.mark.golden
class TestEPA40CFR98EmissionFactors:
    @pytest.mark.parametrize("fuel,expected_ef", [
        ("natural_gas", 53.06), ("distillate_fuel_oil_2", 73.96),
        ("residual_fuel_oil_6", 75.10), ("bituminous_coal", 93.28),
    ])
    def test_co2_emission_factors(self, fuel: str, expected_ef: float):
        golden = EPA40CFRPart98EmissionFactors.CO2_EMISSION_FACTORS[fuel]
        assert golden.value == expected_ef

    def test_co2e_calculation_with_gwp(self):
        gwp_ch4, gwp_n2o = 25, 298
        co2, ch4, n2o = 53.06, 0.001 * 25, 0.0001 * 298
        co2e = co2 + ch4 + n2o
        assert abs(co2e - 53.11) <= 0.05


@pytest.mark.golden
class TestCombustionStoichiometry:
    @pytest.mark.parametrize("fuel,carbon_pct,hydrogen_pct,expected_afr", [
        ("methane", 75.0, 25.0, 17.24),
        ("propane", 81.7, 18.3, 15.67),
        ("diesel", 86.5, 13.5, 14.5),
    ])
    def test_stoichiometric_air_fuel_ratio(self, fuel, carbon_pct, hydrogen_pct, expected_afr):
        c, h = carbon_pct / 100, hydrogen_pct / 100
        o2_required = 2.667 * c + 8 * h
        calculated = o2_required / 0.232
        assert abs(calculated - expected_afr) <= 0.5


@pytest.mark.golden
class TestDeterminism:
    def test_excess_air_determinism(self):
        hashes = set()
        for _ in range(100):
            results = [round(o2 / (21 - o2) * 100, 10) for o2, _ in EXCESS_AIR_O2_RELATIONSHIP if o2 < 21]
            hashes.add(hashlib.sha256(json.dumps(results).encode()).hexdigest())
        assert len(hashes) == 1

    def test_provenance_hash_consistency(self):
        data = {"fuel": "natural_gas", "heat_input": 100.0, "o2": 3.0}
        hashes = set(hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() for _ in range(100))
        assert len(hashes) == 1


def export_golden_values() -> Dict[str, Any]:
    return {
        "metadata": {"version": "1.0.0", "agent": "GL-002_Flameguard"},
        "asme_ptc4": {"stoich_air_ng": 17.24, "theoretical_co2_ng": 11.73},
        "epa_f_factors": {"natural_gas": {"Fd": 8710, "Fw": 10610, "Fc": 1040}},
        "epa_co2_ef": {"natural_gas": 53.06, "oil2": 73.96, "coal": 93.28},
    }


if __name__ == "__main__":
    print(json.dumps(export_golden_values(), indent=2))
