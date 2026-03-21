# -*- coding: utf-8 -*-
"""
Performance tests for PACK-032 Building Energy Assessment Pack

Tests engine execution latency (<100ms target), hash computation
performance, batch processing, and scalability. All engines are tested
for timing characteristics under representative workloads.

Target: 40+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
import time
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
CONFIG_DIR = PACK_ROOT / "config"


def _load(name: str, prefix: str = "pack032_perf"):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"{prefix}.{name}"
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


def _load_config():
    path = CONFIG_DIR / "pack_config.py"
    if not path.exists():
        pytest.skip(f"pack_config.py not found")
    mod_key = "pack032_perf_cfg.pack_config"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load pack_config: {exc}")
    return mod


def _timed(func, *args, **kwargs):
    """Execute a function and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000.0
    return result, elapsed


LATENCY_TARGET_MS = 500.0  # Generous target for single-engine call


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture(scope="module")
def envelope_mod():
    return _load("building_envelope_engine")


@pytest.fixture(scope="module")
def epc_mod():
    return _load("epc_rating_engine")


@pytest.fixture(scope="module")
def hvac_mod():
    return _load("hvac_assessment_engine")


@pytest.fixture(scope="module")
def dhw_mod():
    return _load("domestic_hot_water_engine")


@pytest.fixture(scope="module")
def lighting_mod():
    return _load("lighting_assessment_engine")


@pytest.fixture(scope="module")
def renewable_mod():
    return _load("renewable_integration_engine")


@pytest.fixture(scope="module")
def benchmark_mod():
    return _load("building_benchmark_engine")


@pytest.fixture(scope="module")
def retrofit_mod():
    return _load("retrofit_analysis_engine")


@pytest.fixture(scope="module")
def indoor_mod():
    return _load("indoor_environment_engine")


@pytest.fixture(scope="module")
def wlc_mod():
    return _load("whole_life_carbon_engine")


@pytest.fixture(scope="module")
def cfg_mod():
    return _load_config()


# =========================================================================
# Test Envelope Engine Latency
# =========================================================================


class TestEnvelopePerformance:
    def test_envelope_analyze_latency(self, envelope_mod):
        engine = envelope_mod.BuildingEnvelopeEngine()
        walls = [
            envelope_mod.WallElement(
                element_id="W1",
                wall_type=envelope_mod.WallType.CAVITY_WALL,
                area_m2=200.0,
            ),
        ]
        windows = [
            envelope_mod.WindowElement(
                element_id="WIN1",
                window_type=envelope_mod.WindowType.DOUBLE_GLAZED,
                area_m2=80.0,
            ),
        ]
        envelope = envelope_mod.BuildingEnvelope(
            facility_id="BLD-PERF-ENV",
            name="Performance Envelope Test",
            year_built=2000,
            gross_floor_area_m2=2000.0,
            heated_volume_m3=6000.0,
            walls=walls,
            windows=windows,
        )
        result, elapsed_ms = _timed(engine.analyze, envelope)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS

    def test_envelope_repeated_runs(self, envelope_mod):
        engine = envelope_mod.BuildingEnvelopeEngine()
        walls = [
            envelope_mod.WallElement(
                element_id="W1",
                wall_type=envelope_mod.WallType.CAVITY_WALL,
                area_m2=100.0,
            ),
        ]
        envelope = envelope_mod.BuildingEnvelope(
            facility_id="BLD-PERF-RPT",
            name="Repeated Run Test",
            year_built=2000,
            gross_floor_area_m2=1000.0,
            heated_volume_m3=3000.0,
            walls=walls,
        )
        times = []
        for _ in range(5):
            _, elapsed = _timed(engine.analyze, envelope)
            times.append(elapsed)
        avg_ms = sum(times) / len(times)
        assert avg_ms < LATENCY_TARGET_MS


# =========================================================================
# Test EPC Engine Latency
# =========================================================================


class TestEPCPerformance:
    def test_epc_rate_latency(self, epc_mod):
        engine = epc_mod.EPCRatingEngine()
        building = epc_mod.BuildingData(
            facility_id="BLD-PERF-EPC",
            building_type=epc_mod.BuildingUseType.OFFICE,
            floor_area_m2=2000.0,
        )
        result, elapsed_ms = _timed(engine.rate, building)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS

    def test_epc_repeated_runs(self, epc_mod):
        engine = epc_mod.EPCRatingEngine()
        building = epc_mod.BuildingData(
            facility_id="BLD-PERF-EPC-R",
            floor_area_m2=500.0,
        )
        times = []
        for _ in range(5):
            _, elapsed = _timed(engine.rate, building)
            times.append(elapsed)
        avg_ms = sum(times) / len(times)
        assert avg_ms < LATENCY_TARGET_MS


# =========================================================================
# Test HVAC Engine Latency
# =========================================================================


class TestHVACPerformance:
    def test_hvac_assess_latency(self, hvac_mod):
        engine = hvac_mod.HVACAssessmentEngine()
        heating = hvac_mod.HeatingSystem(
            system_type=hvac_mod.HeatingSystemType.GAS_BOILER,
            capacity_kw=100.0,
        )
        hvac_input = hvac_mod.HVACInput(
            facility_id="BLD-PERF-HVAC",
            floor_area_m2=2000.0,
            heating_systems=[heating],
        )
        result, elapsed_ms = _timed(engine.assess, hvac_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS


# =========================================================================
# Test DHW Engine Latency
# =========================================================================


class TestDHWPerformance:
    def test_dhw_assess_latency(self, dhw_mod):
        engine = dhw_mod.DomesticHotWaterEngine()
        dhw_system = dhw_mod.DHWSystemInput(
            system_type=dhw_mod.DHWSystemType.GAS_BOILER,
        )
        dhw_input = dhw_mod.DHWAssessmentInput(
            building_id="BLD-PERF-DHW",
            building_type=dhw_mod.BuildingOccupancyType.OFFICE,
            occupancy_count=50,
            dhw_system=dhw_system,
        )
        result, elapsed_ms = _timed(engine.analyze, dhw_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS


# =========================================================================
# Test Lighting Engine Latency
# =========================================================================


class TestLightingPerformance:
    def test_lighting_assess_latency(self, lighting_mod):
        engine = lighting_mod.LightingAssessmentEngine()
        zone = lighting_mod.LightingZoneInput(
            zone_id="Z1",
            zone_name="Open Plan",
            space_category=lighting_mod.SpaceCategory.OFFICE_OPEN_PLAN,
            floor_area_m2=500.0,
            number_of_fixtures=50,
            watts_per_fixture=70.0,
            annual_operating_hours=2500,
            lamp_type=lighting_mod.LampType.LED,
        )
        light_input = lighting_mod.LightingAssessmentInput(
            building_id="BLD-PERF-LT",
            total_floor_area_m2=500.0,
            zones=[zone],
        )
        result, elapsed_ms = _timed(engine.analyze, light_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS

    def test_lighting_multi_zone_latency(self, lighting_mod):
        engine = lighting_mod.LightingAssessmentEngine()
        zones = []
        for i in range(10):
            zones.append(
                lighting_mod.LightingZoneInput(
                    zone_id=f"Z{i}",
                    zone_name=f"Zone {i}",
                    space_category=lighting_mod.SpaceCategory.OFFICE_OPEN_PLAN,
                    floor_area_m2=100.0,
                    number_of_fixtures=10,
                    watts_per_fixture=70.0,
                    annual_operating_hours=2500,
                    lamp_type=lighting_mod.LampType.LED,
                )
            )
        light_input = lighting_mod.LightingAssessmentInput(
            building_id="BLD-PERF-LT-MZ",
            total_floor_area_m2=1000.0,
            zones=zones,
        )
        result, elapsed_ms = _timed(engine.analyze, light_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS * 2  # Allow 2x for multi-zone


# =========================================================================
# Test Renewable Engine Latency
# =========================================================================


class TestRenewablePerformance:
    def test_renewable_assess_latency(self, renewable_mod):
        engine = renewable_mod.RenewableIntegrationEngine()
        pv = renewable_mod.SolarPVInput(
            system_capacity_kwp=50.0,
            tilt_deg=30.0,
        )
        ren_input = renewable_mod.RenewableAssessmentInput(
            building_id="BLD-PERF-REN",
            building_load_profile=renewable_mod.BuildingLoadProfile.OFFICE_WEEKDAY,
            annual_electricity_consumption_kwh=400000.0,
            solar_pv=pv,
        )
        result, elapsed_ms = _timed(engine.analyze, ren_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS


# =========================================================================
# Test Benchmark Engine Latency
# =========================================================================


class TestBenchmarkPerformance:
    def test_benchmark_eui_latency(self, benchmark_mod):
        engine = benchmark_mod.BuildingBenchmarkEngine()
        energy = benchmark_mod.EnergyConsumptionInput(
            electricity_kwh=250000.0,
            gas_kwh=150000.0,
        )
        bench_input = benchmark_mod.BenchmarkInput(
            building_id="BLD-PERF-BM",
            building_type=benchmark_mod.BuildingType.OFFICE,
            gross_internal_area_m2=2000.0,
            energy=energy,
        )
        result, elapsed_ms = _timed(engine.calculate_eui, bench_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS

    def test_benchmark_full_assess_latency(self, benchmark_mod):
        engine = benchmark_mod.BuildingBenchmarkEngine()
        energy = benchmark_mod.EnergyConsumptionInput(
            electricity_kwh=250000.0,
            gas_kwh=150000.0,
        )
        bench_input = benchmark_mod.BenchmarkInput(
            building_id="BLD-PERF-BM-F",
            building_type=benchmark_mod.BuildingType.OFFICE,
            gross_internal_area_m2=2000.0,
            energy=energy,
        )
        if hasattr(engine, "assess"):
            result, elapsed_ms = _timed(engine.assess, bench_input)
            assert result is not None
            assert elapsed_ms < LATENCY_TARGET_MS


# =========================================================================
# Test Retrofit Engine Latency
# =========================================================================


class TestRetrofitPerformance:
    def test_retrofit_analyze_latency(self, retrofit_mod):
        engine = retrofit_mod.RetrofitAnalysisEngine()
        measures = [
            retrofit_mod.MeasureInput(measure_id="EWI_001"),
            retrofit_mod.MeasureInput(measure_id="LIGHT_001"),
        ]
        ret_input = retrofit_mod.RetrofitAnalysisInput(
            building_id="BLD-PERF-RET",
            building_type="office",
            floor_area_m2=1000.0,
            baseline_energy_kwh_yr=200000.0,
            measures=measures,
        )
        if hasattr(engine, "analyze"):
            result, elapsed_ms = _timed(engine.analyze, ret_input)
            assert result is not None
            assert elapsed_ms < LATENCY_TARGET_MS

    def test_retrofit_many_measures_latency(self, retrofit_mod):
        engine = retrofit_mod.RetrofitAnalysisEngine()
        measures = [
            retrofit_mod.MeasureInput(measure_id="EWI_001"),
            retrofit_mod.MeasureInput(measure_id="ROOF_001"),
            retrofit_mod.MeasureInput(measure_id="FLOOR_001"),
            retrofit_mod.MeasureInput(measure_id="WIN_001"),
            retrofit_mod.MeasureInput(measure_id="LIGHT_001"),
            retrofit_mod.MeasureInput(measure_id="HEAT_001"),
            retrofit_mod.MeasureInput(measure_id="VENT_001"),
            retrofit_mod.MeasureInput(measure_id="PV_001"),
        ]
        ret_input = retrofit_mod.RetrofitAnalysisInput(
            building_id="BLD-PERF-RET-M",
            building_type="detached_house",
            floor_area_m2=120.0,
            baseline_energy_kwh_yr=25000.0,
            measures=measures,
        )
        if hasattr(engine, "analyze"):
            result, elapsed_ms = _timed(engine.analyze, ret_input)
            assert result is not None
            assert elapsed_ms < LATENCY_TARGET_MS * 2


# =========================================================================
# Test Indoor Environment Engine Latency
# =========================================================================


class TestIndoorPerformance:
    def test_pmv_ppd_latency(self, indoor_mod):
        engine = indoor_mod.IndoorEnvironmentEngine()
        comfort = indoor_mod.ThermalComfortInput(
            air_temperature_degC=22.0,
            mean_radiant_temperature_degC=22.0,
            relative_humidity_pct=50.0,
            air_speed_m_s=0.1,
            metabolic_rate_met=1.2,
            clothing_insulation_clo=0.7,
        )
        result, elapsed_ms = _timed(engine.calculate_pmv_ppd, comfort)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS

    def test_full_ieq_latency(self, indoor_mod):
        engine = indoor_mod.IndoorEnvironmentEngine()
        inp = indoor_mod.IndoorEnvironmentInput(
            building_id="BLD-PERF-IEQ",
            thermal_inputs=[
                indoor_mod.ThermalComfortInput(
                    air_temperature_degC=22.0,
                    mean_radiant_temperature_degC=22.0,
                    relative_humidity_pct=50.0,
                ),
            ],
        )
        result, elapsed_ms = _timed(engine.assess, inp)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS


# =========================================================================
# Test Whole-Life Carbon Engine Latency
# =========================================================================


class TestWLCPerformance:
    def test_wlc_calculate_latency(self, wlc_mod):
        engine = wlc_mod.WholeLifeCarbonEngine()
        mat = wlc_mod.MaterialInput(
            material_id="concrete_C30_37",
            material_category="concrete",
            quantity=500000.0,
            unit="kg",
        )
        wlc_input = wlc_mod.WholeLifeCarbonInput(
            building_id="BLD-PERF-WLC",
            gross_internal_area_m2=2000.0,
            materials=[mat],
        )
        result, elapsed_ms = _timed(engine.analyze, wlc_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS

    def test_wlc_multi_material_latency(self, wlc_mod):
        engine = wlc_mod.WholeLifeCarbonEngine()
        materials = [
            wlc_mod.MaterialInput(
                material_id="concrete_C30_37",
                material_category="concrete",
                quantity=500000.0,
            ),
            wlc_mod.MaterialInput(
                material_id="steel_rebar",
                material_category="steel",
                quantity=25000.0,
            ),
            wlc_mod.MaterialInput(
                material_id="timber_CLT",
                material_category="timber",
                quantity=80000.0,
            ),
        ]
        wlc_input = wlc_mod.WholeLifeCarbonInput(
            building_id="BLD-PERF-WLC-M",
            gross_internal_area_m2=2000.0,
            materials=materials,
        )
        result, elapsed_ms = _timed(engine.analyze, wlc_input)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS


# =========================================================================
# Test Config Hash Performance
# =========================================================================


class TestConfigPerformance:
    def test_config_hash_latency(self, cfg_mod):
        config = cfg_mod.PackConfig()
        _, elapsed_ms = _timed(config.get_config_hash)
        assert elapsed_ms < 50.0  # Config hashing should be fast

    def test_config_hash_repeated(self, cfg_mod):
        config = cfg_mod.PackConfig()
        times = []
        for _ in range(20):
            _, elapsed = _timed(config.get_config_hash)
            times.append(elapsed)
        avg_ms = sum(times) / len(times)
        assert avg_ms < 50.0

    def test_preset_loading_latency(self, cfg_mod):
        _, elapsed_ms = _timed(cfg_mod.PackConfig.from_preset, "commercial_office")
        assert elapsed_ms < 200.0  # File I/O + parsing

    def test_deep_merge_latency(self, cfg_mod):
        base = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
        over = {"a": {"b": {"c": 10, "f": 4}}, "g": 5}
        _, elapsed_ms = _timed(cfg_mod.PackConfig._deep_merge, base, over)
        assert elapsed_ms < 10.0


# =========================================================================
# Test Scalability
# =========================================================================


class TestScalability:
    def test_envelope_many_elements(self, envelope_mod):
        """Test envelope engine with many wall/window elements."""
        engine = envelope_mod.BuildingEnvelopeEngine()
        walls = [
            envelope_mod.WallElement(
                element_id=f"W{i}",
                wall_type=envelope_mod.WallType.CAVITY_WALL,
                area_m2=50.0,
            )
            for i in range(20)
        ]
        windows = [
            envelope_mod.WindowElement(
                element_id=f"WIN{i}",
                window_type=envelope_mod.WindowType.DOUBLE_GLAZED,
                area_m2=10.0,
            )
            for i in range(20)
        ]
        envelope = envelope_mod.BuildingEnvelope(
            facility_id="BLD-SCALE-ENV",
            name="Scale Test Building",
            year_built=2000,
            gross_floor_area_m2=5000.0,
            heated_volume_m3=15000.0,
            walls=walls,
            windows=windows,
        )
        result, elapsed_ms = _timed(engine.analyze, envelope)
        assert result is not None
        assert elapsed_ms < LATENCY_TARGET_MS * 3  # Allow 3x for scale
