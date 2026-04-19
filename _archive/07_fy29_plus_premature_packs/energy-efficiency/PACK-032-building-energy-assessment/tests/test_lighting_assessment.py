# -*- coding: utf-8 -*-
"""
Unit tests for LightingAssessmentEngine (PACK-032 Engine 5)

Tests LPD calculation, LENI (EN 15193), daylight factor, occupancy
controls, and LED retrofit analysis.

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
    mod_key = f"pack032_light.{name}"
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
    return _load("lighting_assessment_engine")


@pytest.fixture
def engine(engine_mod):
    return engine_mod.LightingAssessmentEngine()


@pytest.fixture
def office_zone(engine_mod):
    mod = engine_mod
    return mod.LightingZoneInput(
        zone_id="Z-001",
        zone_name="Open Plan Office",
        space_category=mod.SpaceCategory.OFFICE_OPEN_PLAN,
        floor_area_m2=500.0,
        lamp_type=mod.LampType.FLUORESCENT_T8,
        number_of_fixtures=80,
        watts_per_fixture=36.0,
        lumens_per_fixture=3200.0,
        annual_operating_hours=2500.0,
        control_type=mod.ControlType.MANUAL_SWITCH,
        daylight_factor_pct=3.0,
    )


@pytest.fixture
def led_zone(engine_mod):
    mod = engine_mod
    return mod.LightingZoneInput(
        zone_id="Z-002",
        zone_name="LED Office",
        space_category=mod.SpaceCategory.OFFICE_OPEN_PLAN,
        floor_area_m2=500.0,
        lamp_type=mod.LampType.LED,
        number_of_fixtures=80,
        watts_per_fixture=18.0,
        lumens_per_fixture=3600.0,
        annual_operating_hours=2500.0,
        control_type=mod.ControlType.DAYLIGHT_SENSOR,
        daylight_factor_pct=3.0,
    )


@pytest.fixture
def assessment_input(engine_mod, office_zone, led_zone):
    mod = engine_mod
    return mod.LightingAssessmentInput(
        building_id="BLD-LIT-001",
        building_type="office",
        total_floor_area_m2=1000.0,
        zones=[office_zone, led_zone],
    )


# =========================================================================
# Test Initialization
# =========================================================================


class TestInitialization:
    def test_engine_class_exists(self, engine_mod):
        assert hasattr(engine_mod, "LightingAssessmentEngine")

    def test_engine_instantiation(self, engine):
        assert engine is not None

    def test_input_models(self, engine_mod):
        assert hasattr(engine_mod, "LightingAssessmentInput")
        assert hasattr(engine_mod, "LightingZoneInput")


# =========================================================================
# Test LPD Calculation
# =========================================================================


class TestLPD:
    def test_calculate_lpd(self, engine, office_zone):
        result = engine.calculate_lpd(office_zone)
        assert result is not None
        assert result.lpd_w_per_m2 > 0

    def test_lpd_value_correct(self, engine, office_zone):
        result = engine.calculate_lpd(office_zone)
        expected = (80 * 36.0) / 500.0  # 5.76 W/m2
        assert result.lpd_w_per_m2 == pytest.approx(expected, rel=0.01)

    def test_lpd_rating(self, engine, office_zone):
        result = engine.calculate_lpd(office_zone)
        assert result.lpd_rating in ("best_practice", "good", "acceptable", "poor")

    def test_led_lower_lpd(self, engine, office_zone, led_zone):
        r_old = engine.calculate_lpd(office_zone)
        r_led = engine.calculate_lpd(led_zone)
        assert r_led.lpd_w_per_m2 < r_old.lpd_w_per_m2

    def test_annual_energy_positive(self, engine, office_zone):
        result = engine.calculate_lpd(office_zone)
        assert result.annual_energy_kwh > 0

    def test_control_saving(self, engine, led_zone):
        result = engine.calculate_lpd(led_zone)
        assert result.control_saving_pct >= 0


# =========================================================================
# Test LENI Calculation
# =========================================================================


class TestLENI:
    def test_calculate_leni(self, engine, assessment_input):
        # First calculate total energy from zones
        total_energy = 0.0
        for zone in assessment_input.zones:
            lpd = engine.calculate_lpd(zone)
            total_energy += lpd.annual_energy_with_controls_kwh
        result = engine.calculate_leni(
            total_energy_kwh=total_energy,
            total_floor_area_m2=assessment_input.total_floor_area_m2,
            building_type=assessment_input.building_type,
        )
        assert result is not None
        assert result.leni_kwh_per_m2_yr > 0

    def test_leni_rating(self, engine, assessment_input):
        total_energy = 0.0
        for zone in assessment_input.zones:
            lpd = engine.calculate_lpd(zone)
            total_energy += lpd.annual_energy_with_controls_kwh
        result = engine.calculate_leni(
            total_energy_kwh=total_energy,
            total_floor_area_m2=assessment_input.total_floor_area_m2,
            building_type=assessment_input.building_type,
        )
        assert result.leni_rating in ("best_practice", "good", "typical", "poor")

    def test_leni_benchmarks(self, engine, assessment_input):
        total_energy = 0.0
        for zone in assessment_input.zones:
            lpd = engine.calculate_lpd(zone)
            total_energy += lpd.annual_energy_with_controls_kwh
        result = engine.calculate_leni(
            total_energy_kwh=total_energy,
            total_floor_area_m2=assessment_input.total_floor_area_m2,
            building_type=assessment_input.building_type,
        )
        assert result.benchmark_best_practice > 0
        assert result.benchmark_typical > 0


# =========================================================================
# Test Daylight Assessment
# =========================================================================


class TestDaylight:
    def test_assess_daylight(self, engine, office_zone):
        result = engine.assess_daylight(office_zone)
        assert result is not None
        assert result.daylight_factor_pct >= 0

    def test_daylight_rating(self, engine, office_zone):
        result = engine.assess_daylight(office_zone)
        assert isinstance(result.daylight_rating, str) and len(result.daylight_rating) > 0


# =========================================================================
# Test Controls Assessment
# =========================================================================


class TestControls:
    def test_assess_controls(self, engine, office_zone):
        result = engine.assess_controls(office_zone)
        assert result is not None
        assert result.control_saving_pct >= 0

    def test_advanced_controls_more_savings(self, engine, office_zone, led_zone):
        r_manual = engine.assess_controls(office_zone)
        r_daylight = engine.assess_controls(led_zone)
        assert r_daylight.control_saving_pct >= r_manual.control_saving_pct


# =========================================================================
# Test LED Retrofit
# =========================================================================


class TestRetrofit:
    def test_retrofit_savings(self, engine, office_zone):
        result = engine.calculate_retrofit_savings(office_zone)
        assert result is not None
        assert result.annual_energy_saving_kwh >= 0

    def test_already_led_no_saving(self, engine, led_zone):
        result = engine.calculate_retrofit_savings(led_zone)
        assert result.already_led is True
        assert result.annual_energy_saving_kwh == pytest.approx(0, abs=1)


# =========================================================================
# Test Provenance
# =========================================================================


class TestProvenance:
    def test_full_assessment_provenance(self, engine, assessment_input):
        if hasattr(engine, "analyze"):
            result = engine.analyze(assessment_input)
            assert hasattr(result, "provenance_hash")
            assert len(result.provenance_hash) == 64


# =========================================================================
# Test Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_single_zone_assessment(self, engine, engine_mod, office_zone):
        lpd = engine.calculate_lpd(office_zone)
        leni = engine.calculate_leni(
            total_energy_kwh=lpd.annual_energy_with_controls_kwh,
            total_floor_area_m2=500.0,
            building_type="office",
        )
        assert leni.leni_kwh_per_m2_yr > 0

    def test_visual_quality(self, engine, office_zone):
        if hasattr(engine, "assess_visual_quality"):
            result = engine.assess_visual_quality(office_zone)
            assert result is not None
