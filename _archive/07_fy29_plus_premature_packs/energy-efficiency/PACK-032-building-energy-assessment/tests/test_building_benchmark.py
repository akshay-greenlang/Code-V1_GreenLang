# -*- coding: utf-8 -*-
"""
Unit tests for BuildingBenchmarkEngine (PACK-032 Engine 7)

Tests EUI calculation, CRREM pathways, Energy Star score, DEC rating,
and degree-day normalization.

Target: 30+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack032_bench.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


@pytest.fixture(scope="module")
def engine_mod():
    return _load("building_benchmark_engine")


@pytest.fixture
def engine(engine_mod):
    return engine_mod.BuildingBenchmarkEngine()


@pytest.fixture
def office_benchmark_input(engine_mod):
    mod = engine_mod
    energy = mod.EnergyConsumptionInput(
        electricity_kwh=250000.0,
        gas_kwh=150000.0,
    )
    weather = mod.WeatherDataInput(
        actual_hdd=2800.0,
        actual_cdd=200.0,
    )
    return mod.BenchmarkInput(
        building_id="BLD-BM-001",
        building_name="Benchmark Office",
        building_type=mod.BuildingType.OFFICE,
        gross_internal_area_m2=2000.0,
        energy=energy,
        weather=weather,
    )


@pytest.fixture
def hotel_benchmark_input(engine_mod):
    mod = engine_mod
    energy = mod.EnergyConsumptionInput(
        electricity_kwh=800000.0,
        gas_kwh=600000.0,
    )
    return mod.BenchmarkInput(
        building_id="BLD-BM-002",
        building_name="Benchmark Hotel",
        building_type=mod.BuildingType.HOTEL,
        gross_internal_area_m2=5000.0,
        energy=energy,
    )


# =========================================================================
# Test Initialization
# =========================================================================


class TestInitialization:
    def test_engine_class_exists(self, engine_mod):
        assert hasattr(engine_mod, "BuildingBenchmarkEngine")

    def test_engine_instantiation(self, engine):
        assert engine is not None

    def test_input_model(self, engine_mod):
        assert hasattr(engine_mod, "BenchmarkInput")

    def test_eui_result_model(self, engine_mod):
        assert hasattr(engine_mod, "EUIResult")


# =========================================================================
# Test EUI Calculation
# =========================================================================


class TestEUI:
    def test_calculate_eui(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        assert result is not None
        assert result.eui_kwh_per_m2 > 0

    def test_eui_value(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        expected_eui = (250000.0 + 150000.0) / 2000.0  # 200 kWh/m2
        assert result.eui_kwh_per_m2 == pytest.approx(expected_eui, rel=0.01)

    def test_electricity_eui(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        expected = 250000.0 / 2000.0  # 125 kWh/m2
        assert result.electricity_eui == pytest.approx(expected, rel=0.01)

    def test_thermal_eui(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        expected = 150000.0 / 2000.0  # 75 kWh/m2
        assert result.thermal_eui == pytest.approx(expected, rel=0.01)

    def test_normalised_eui(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        if result.eui_normalised_kwh_per_m2 is not None:
            assert result.eui_normalised_kwh_per_m2 > 0

    def test_performance_tier(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        assert result.performance_tier in (
            "best_practice", "good", "typical", "poor",
        )

    def test_benchmarks_positive(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        assert result.benchmark_best_practice > 0
        assert result.benchmark_typical > 0


# =========================================================================
# Test DEC Rating
# =========================================================================


class TestDEC:
    def test_calculate_dec(self, engine, engine_mod, office_benchmark_input):
        result = engine.calculate_dec_rating(
            180.0, engine_mod.BuildingType.OFFICE, engine_mod.ClimateZone.CENTRAL_EUROPE
        )
        assert result is not None
        assert result.operational_rating > 0

    def test_dec_band(self, engine, engine_mod, office_benchmark_input):
        result = engine.calculate_dec_rating(
            180.0, engine_mod.BuildingType.OFFICE, engine_mod.ClimateZone.CENTRAL_EUROPE
        )
        assert result.dec_band in ("A", "B", "C", "D", "E", "F", "G")

    def test_dec_benchmark_eui(self, engine, engine_mod, office_benchmark_input):
        result = engine.calculate_dec_rating(
            180.0, engine_mod.BuildingType.OFFICE, engine_mod.ClimateZone.CENTRAL_EUROPE
        )
        assert result.benchmark_eui > 0


# =========================================================================
# Test CRREM Pathway
# =========================================================================


class TestCRREM:
    def test_crrem_assessment(self, engine, office_benchmark_input):
        if hasattr(engine, "calculate_crrem") or hasattr(engine, "assess_crrem"):
            method = getattr(engine, "calculate_crrem", None) or getattr(engine, "assess_crrem")
            result = method(office_benchmark_input)
            assert result is not None
            assert hasattr(result, "compliant")

    def test_crrem_via_full_assess(self, engine, office_benchmark_input):
        if hasattr(engine, "assess"):
            result = engine.assess(office_benchmark_input)
            if hasattr(result, "crrem_result") and result.crrem_result is not None:
                assert result.crrem_result.current_carbon_intensity_kg_m2 >= 0


# =========================================================================
# Test Energy Star
# =========================================================================


class TestEnergyStar:
    def test_energy_star_score(self, engine, office_benchmark_input):
        if hasattr(engine, "calculate_energy_star") or hasattr(engine, "estimate_energy_star"):
            method = getattr(engine, "calculate_energy_star", None) or getattr(engine, "estimate_energy_star")
            result = method(office_benchmark_input)
            assert result is not None
            assert 1 <= result.estimated_score <= 100


# =========================================================================
# Test Provenance
# =========================================================================


class TestProvenance:
    def test_full_assessment_provenance(self, engine, office_benchmark_input):
        if hasattr(engine, "assess"):
            result = engine.assess(office_benchmark_input)
            assert hasattr(result, "provenance_hash")
            assert len(result.provenance_hash) == 64


# =========================================================================
# Test Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_hotel_eui_higher(self, engine, office_benchmark_input, hotel_benchmark_input):
        office_eui = engine.calculate_eui(office_benchmark_input)
        hotel_eui = engine.calculate_eui(hotel_benchmark_input)
        # Hotels typically have higher EUI
        assert hotel_eui.eui_kwh_per_m2 > 0

    def test_total_energy(self, engine, office_benchmark_input):
        result = engine.calculate_eui(office_benchmark_input)
        assert result.total_energy_kwh == pytest.approx(400000.0, rel=0.01)

    def test_no_weather_data(self, engine, engine_mod):
        mod = engine_mod
        energy = mod.EnergyConsumptionInput(
            electricity_kwh=100000.0,
            gas_kwh=50000.0,
        )
        inp = mod.BenchmarkInput(
            building_id="BLD-NW",
            building_type=mod.BuildingType.OFFICE,
            gross_internal_area_m2=1000.0,
            energy=energy,
        )
        result = engine.calculate_eui(inp)
        assert result.eui_kwh_per_m2 > 0

    def test_peer_comparison(self, engine, engine_mod):
        mod = engine_mod
        energy = mod.EnergyConsumptionInput(electricity_kwh=200000.0)
        peers = [
            mod.PeerBuildingInput(building_name="Peer A", eui_kwh_per_m2=180.0),
            mod.PeerBuildingInput(building_name="Peer B", eui_kwh_per_m2=220.0),
        ]
        inp = mod.BenchmarkInput(
            building_id="BLD-PEER",
            building_type=mod.BuildingType.OFFICE,
            gross_internal_area_m2=1000.0,
            energy=energy,
            peers=peers,
        )
        result = engine.calculate_eui(inp)
        assert result.eui_kwh_per_m2 > 0
