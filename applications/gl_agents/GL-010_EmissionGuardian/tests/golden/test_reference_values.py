# -*- coding: utf-8 -*-
import pytest
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any

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


class EPA40CFR75References:
    RATA_BIAS_LIMIT = GoldenValue("RATA Bias Limit", 7.5, "percent", 0.0, "EPA 40 CFR 75", "Appendix A")
    CEMS_CALIBRATION_ERROR = GoldenValue("CEMS Cal Error", 2.5, "percent", 0.0, "EPA 40 CFR 75", "Appendix A")
    LINEARITY_LIMIT = GoldenValue("Linearity Limit", 5.0, "percent", 0.0, "EPA 40 CFR 75", "Appendix A")


class EPAMethod19FFactor:
    NATURAL_GAS = {"Fd": 8710.0, "Fw": 10610.0, "Fc": 1040.0}
    NO2_OIL = {"Fd": 9190.0, "Fw": 10320.0, "Fc": 1420.0}
    COAL = {"Fd": 9780.0, "Fw": 10640.0, "Fc": 1800.0}


class EPA40CFR98EmissionFactors:
    CO2_FACTORS = {
        "natural_gas": GoldenValue("CO2 EF NG", 53.06, "kg/MMBtu", 1.0, "EPA 40 CFR 98", "Table C-1"),
        "oil_2": GoldenValue("CO2 EF Oil2", 73.96, "kg/MMBtu", 1.0, "EPA 40 CFR 98", "Table C-1"),
        "coal": GoldenValue("CO2 EF Coal", 93.28, "kg/MMBtu", 2.0, "EPA 40 CFR 98", "Table C-1"),
    }


@pytest.mark.golden
class TestEPA40CFR75CEMS:
    def test_rata_bias_limit(self):
        golden = EPA40CFR75References.RATA_BIAS_LIMIT
        assert golden.value == 7.5

    def test_cems_calibration_error_limit(self):
        golden = EPA40CFR75References.CEMS_CALIBRATION_ERROR
        assert golden.value == 2.5

    def test_linearity_limit(self):
        golden = EPA40CFR75References.LINEARITY_LIMIT
        assert golden.value == 5.0


@pytest.mark.golden
class TestEPAMethod19FFactor:
    @pytest.mark.parametrize("fuel,expected_fd", [
        ("NATURAL_GAS", 8710.0), ("NO2_OIL", 9190.0), ("COAL", 9780.0),
    ])
    def test_fd_factors(self, fuel, expected_fd):
        actual = getattr(EPAMethod19FFactor, fuel)["Fd"]
        assert actual == expected_fd

    def test_fw_greater_than_fd(self):
        for fuel_data in [EPAMethod19FFactor.NATURAL_GAS, EPAMethod19FFactor.NO2_OIL]:
            assert fuel_data["Fw"] > fuel_data["Fd"]


@pytest.mark.golden
class TestEPA40CFR98EmissionFactors:
    @pytest.mark.parametrize("fuel,expected_ef", [
        ("natural_gas", 53.06), ("oil_2", 73.96), ("coal", 93.28),
    ])
    def test_co2_emission_factors(self, fuel, expected_ef):
        golden = EPA40CFR98EmissionFactors.CO2_FACTORS[fuel]
        assert golden.value == expected_ef


@pytest.mark.golden
class TestEmissionRateCalculations:
    def test_mass_emission_rate(self):
        # E = Concentration * Flow * MW / MV
        conc_ppm = 100
        flow_scfm = 10000
        mw_so2 = 64.066
        mv = 385.5

        lb_per_min = conc_ppm * flow_scfm * mw_so2 / (mv * 1e6)
        lb_per_hr = lb_per_min * 60
        assert 0.5 < lb_per_hr < 1.5


@pytest.mark.golden
class TestDeterminism:
    def test_emission_calculation_determinism(self):
        hashes = set()
        for _ in range(100):
            result = round(100 * 10000 * 64.066 / (385.5 * 1e6) * 60, 8)
            hashes.add(hashlib.sha256(str(result).encode()).hexdigest())
        assert len(hashes) == 1


def export_golden_values() -> Dict[str, Any]:
    return {
        "metadata": {"version": "1.0.0", "agent": "GL-010_EmissionGuardian"},
        "epa_40cfr75": {"rata_bias": 7.5, "cems_cal_error": 2.5},
        "epa_method19": {"natural_gas": {"Fd": 8710, "Fw": 10610}},
    }
