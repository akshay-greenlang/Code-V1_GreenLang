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


class NISTSteamProperties:
    SAT_WATER_100C = {
        "enthalpy_liquid": GoldenValue("hf @ 100C", 419.05, "kJ/kg", 0.1, "IAPWS-IF97", "T=100C"),
        "enthalpy_vapor": GoldenValue("hg @ 100C", 2675.46, "kJ/kg", 0.1, "IAPWS-IF97", "T=100C"),
        "latent_heat": GoldenValue("hfg @ 100C", 2256.41, "kJ/kg", 0.1, "IAPWS-IF97", "T=100C"),
    }


class ASMEPTC1911References:
    """ASME PTC 19.11 Steam Quality Measurement References."""
    CALORIMETER_ACCURACY = GoldenValue("Calorimeter Accuracy", 0.5, "percent", 50.0, "ASME PTC 19.11", "Section 4")
    THROTTLING_CALORIMETER_MIN_QUALITY = GoldenValue("Min Quality for Throttling", 0.95, "fraction", 2.0, "ASME PTC 19.11", "Section 5")


class SteamQualityCalculations:
    QUALITY_TESTS = [
        {"h_mix": 2500.0, "hf": 419.05, "hfg": 2256.41, "expected_quality": 0.922},
        {"h_mix": 2000.0, "hf": 419.05, "hfg": 2256.41, "expected_quality": 0.701},
        {"h_mix": 2675.46, "hf": 419.05, "hfg": 2256.41, "expected_quality": 1.000},
    ]


class CarryoverRiskReferences:
    CARRYOVER_THRESHOLDS = {
        "low": {"quality_min": 0.99, "description": "Minimal moisture"},
        "medium": {"quality_min": 0.97, "description": "Acceptable moisture"},
        "high": {"quality_min": 0.95, "description": "Elevated moisture"},
        "critical": {"quality_min": 0.0, "description": "Excessive moisture"},
    }


class SeparatorEfficiencyReferences:
    CYCLONE_EFFICIENCY = GoldenValue("Cyclone Separator Eff", 0.98, "fraction", 3.0, "Vendor", "Data Sheet")
    MESH_PAD_EFFICIENCY = GoldenValue("Mesh Pad Eff", 0.99, "fraction", 2.0, "Vendor", "Data Sheet")


@pytest.mark.golden
class TestNISTSteamProperties:
    def test_saturation_enthalpy_100C(self):
        golden = NISTSteamProperties.SAT_WATER_100C["enthalpy_vapor"]
        is_valid, _ = golden.validate(2675.46)
        assert is_valid

    def test_latent_heat_consistency(self):
        hf = NISTSteamProperties.SAT_WATER_100C["enthalpy_liquid"].value
        hg = NISTSteamProperties.SAT_WATER_100C["enthalpy_vapor"].value
        hfg = NISTSteamProperties.SAT_WATER_100C["latent_heat"].value
        assert abs((hg - hf) - hfg) <= 0.01


@pytest.mark.golden
class TestASMEPTC1911:
    def test_throttling_calorimeter_min_quality(self):
        golden = ASMEPTC1911References.THROTTLING_CALORIMETER_MIN_QUALITY
        assert 0.94 <= golden.value <= 0.96


@pytest.mark.golden
class TestSteamQualityCalculations:
    @pytest.mark.parametrize("test_case", SteamQualityCalculations.QUALITY_TESTS)
    def test_steam_quality_from_enthalpy(self, test_case):
        h_mix = test_case["h_mix"]
        hf = test_case["hf"]
        hfg = test_case["hfg"]
        expected = test_case["expected_quality"]

        calculated = (h_mix - hf) / hfg
        assert abs(calculated - expected) <= 0.002


@pytest.mark.golden
class TestCarryoverRisk:
    def test_carryover_threshold_ordering(self):
        thresholds = CarryoverRiskReferences.CARRYOVER_THRESHOLDS
        assert thresholds["low"]["quality_min"] > thresholds["medium"]["quality_min"]
        assert thresholds["medium"]["quality_min"] > thresholds["high"]["quality_min"]


@pytest.mark.golden
class TestSeparatorEfficiency:
    def test_separator_efficiency_values(self):
        cyclone = SeparatorEfficiencyReferences.CYCLONE_EFFICIENCY.value
        mesh_pad = SeparatorEfficiencyReferences.MESH_PAD_EFFICIENCY.value
        assert mesh_pad >= cyclone


@pytest.mark.golden
class TestDeterminism:
    def test_quality_calculation_determinism(self):
        hashes = set()
        for _ in range(100):
            quality = round((2500.0 - 419.05) / 2256.41, 8)
            hashes.add(hashlib.sha256(str(quality).encode()).hexdigest())
        assert len(hashes) == 1


def export_golden_values() -> Dict[str, Any]:
    return {
        "metadata": {"version": "1.0.0", "agent": "GL-012_SteamQual"},
        "iapws_if97": {"sat_100C": {"hf": 419.05, "hg": 2675.46, "hfg": 2256.41}},
        "asme_ptc1911": {"min_quality_throttling": 0.95},
    }
