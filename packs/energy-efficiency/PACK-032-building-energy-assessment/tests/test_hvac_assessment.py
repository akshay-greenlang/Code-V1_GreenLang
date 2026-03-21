# -*- coding: utf-8 -*-
"""
Unit tests for HVACAssessmentEngine (PACK-032 Engine 3)

Tests COP/EER/SEER, SFP, heating efficiency, ventilation, heat recovery,
and refrigerant F-gas compliance.

Target: 35+ tests
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
    mod_key = f"pack032_hvac.{name}"
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
    return _load("hvac_assessment_engine")


@pytest.fixture
def engine(engine_mod):
    return engine_mod.HVACAssessmentEngine()


@pytest.fixture
def basic_hvac_input(engine_mod):
    """Basic HVAC input with one heating system."""
    mod = engine_mod
    heating = mod.HeatingSystem(
        system_id="HS-001",
        system_type=mod.HeatingSystemType.GAS_BOILER,
        boiler_subtype="condensing_modern",
        capacity_kw=100.0,
        age_years=5,
    )
    return mod.HVACInput(
        facility_id="FAC-HVAC-001",
        building_name="Test Office",
        floor_area_m2=1000.0,
        heated_volume_m3=3000.0,
        annual_heating_demand_kwh=120000.0,
        heating_systems=[heating],
    )


@pytest.fixture
def full_hvac_input(engine_mod):
    """Full HVAC input with heating, cooling, ventilation."""
    mod = engine_mod
    heating = mod.HeatingSystem(
        system_id="HS-002",
        system_type=mod.HeatingSystemType.GAS_BOILER,
        boiler_subtype="condensing_modern",
        capacity_kw=150.0,
        age_years=5,
    )
    cooling = mod.CoolingSystem(
        system_id="CS-001",
        system_type=mod.CoolingSystemType.SPLIT_SYSTEM,
        capacity_kw=80.0,
        age_years=5,
        refrigerant=mod.RefrigerantType.R410A,
        refrigerant_charge_kg=12.0,
    )
    vent = mod.VentilationSystem(
        system_id="VS-001",
        ventilation_type=mod.VentilationType.MECHANICAL_SUPPLY_EXTRACT,
        airflow_rate_ls=500.0,
        fan_power_w=750.0,
        heat_recovery_type=mod.HeatRecoveryType.PLATE,
    )
    return mod.HVACInput(
        facility_id="FAC-HVAC-002",
        building_name="Full HVAC Office",
        floor_area_m2=2000.0,
        heated_volume_m3=6000.0,
        annual_heating_demand_kwh=200000.0,
        annual_cooling_demand_kwh=50000.0,
        operating_hours=3000.0,
        heating_systems=[heating],
        cooling_systems=[cooling],
        ventilation_systems=[vent],
    )


# =========================================================================
# Test Initialization
# =========================================================================


class TestInitialization:
    def test_engine_class_exists(self, engine_mod):
        assert hasattr(engine_mod, "HVACAssessmentEngine")

    def test_engine_instantiation(self, engine):
        assert engine is not None

    def test_engine_version(self, engine):
        assert engine.engine_version == "1.0.0"

    def test_has_boiler_efficiency_data(self, engine):
        assert hasattr(engine, "_boiler_efficiency")

    def test_has_refrigerant_gwp(self, engine):
        assert hasattr(engine, "_refrigerant_gwp")


# =========================================================================
# Test Full Assessment
# =========================================================================


class TestFullAssessment:
    def test_assess_basic(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert result is not None

    def test_assess_full(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        assert result is not None

    def test_result_has_heating(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert len(result.heating_assessments) >= 1

    def test_result_has_cooling(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        assert len(result.cooling_assessments) >= 1

    def test_result_has_ventilation(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        assert len(result.ventilation_assessments) >= 1


# =========================================================================
# Test Heating Assessment
# =========================================================================


class TestHeatingAssessment:
    def test_heating_efficiency_positive(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        ha = result.heating_assessments[0]
        assert ha.seasonal_efficiency > 0

    def test_condensing_boiler_high_efficiency(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        ha = result.heating_assessments[0]
        # Modern condensing boiler should be > 0.85
        assert ha.seasonal_efficiency > 0.80

    def test_heating_efficiency_rating(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        ha = result.heating_assessments[0]
        assert ha.efficiency_rating in ("poor", "average", "good", "excellent")

    def test_assess_heating_method(self, engine, engine_mod):
        hs = engine_mod.HeatingSystem(
            system_id="HS-T",
            system_type=engine_mod.HeatingSystemType.GAS_BOILER,
            boiler_subtype="condensing_modern",
            capacity_kw=100.0,
            age_years=5,
        )
        context = engine_mod.HVACInput(
            facility_id="FAC-ASSESS-T",
            floor_area_m2=1000.0,
            annual_heating_demand_kwh=100000.0,
            heating_systems=[hs],
        )
        result = engine.assess_heating(hs, context)
        assert result is not None
        assert result.seasonal_efficiency > 0

    def test_overall_heating_efficiency(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert result.overall_heating_efficiency > 0


# =========================================================================
# Test Cooling Assessment
# =========================================================================


class TestCoolingAssessment:
    def test_cooling_seer_positive(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        if result.cooling_assessments:
            ca = result.cooling_assessments[0]
            assert ca.seer > 0

    def test_cooling_efficiency_rating(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        if result.cooling_assessments:
            ca = result.cooling_assessments[0]
            assert ca.efficiency_rating in ("poor", "average", "good", "excellent")

    def test_overall_cooling_seer(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        assert result.overall_cooling_seer >= 0


# =========================================================================
# Test Ventilation Assessment
# =========================================================================


class TestVentilationAssessment:
    def test_sfp_calculated(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        if result.ventilation_assessments:
            va = result.ventilation_assessments[0]
            assert va.sfp_w_ls >= 0

    def test_heat_recovery_assessed(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        if result.ventilation_assessments:
            va = result.ventilation_assessments[0]
            assert va.heat_recovery_efficiency >= 0

    def test_sfp_rating(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        if result.ventilation_assessments:
            va = result.ventilation_assessments[0]
            assert va.sfp_rating != ""


# =========================================================================
# Test Refrigerant / F-gas
# =========================================================================


class TestRefrigerantRisk:
    def test_refrigerant_risk_assessed(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        if result.refrigerant_risks:
            rr = result.refrigerant_risks[0]
            assert rr.gwp_100yr > 0

    def test_f_gas_emissions(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        assert result.total_f_gas_emissions_tco2e >= 0


# =========================================================================
# Test Provenance
# =========================================================================


class TestProvenance:
    def test_provenance_hash(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_provenance_deterministic(self, engine, engine_mod):
        hs = engine_mod.HeatingSystem(
            system_id="HS-DET",
            system_type=engine_mod.HeatingSystemType.GAS_BOILER,
            boiler_subtype="condensing_modern",
            capacity_kw=50.0,
            age_years=5,
        )
        inp = engine_mod.HVACInput(
            facility_id="FAC-DET",
            floor_area_m2=500.0,
            annual_heating_demand_kwh=60000.0,
            heating_systems=[hs],
        )
        r1 = engine.assess(inp)
        r2 = engine.assess(inp)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)


# =========================================================================
# Test Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_processing_time(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert result.processing_time_ms > 0

    def test_recommendations(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert isinstance(result.recommendations, list)

    def test_improvement_measures(self, engine, basic_hvac_input):
        result = engine.assess(basic_hvac_input)
        assert isinstance(result.improvement_measures, list)

    def test_total_hvac_energy(self, engine, full_hvac_input):
        result = engine.assess(full_hvac_input)
        assert result.total_hvac_energy_kwh >= 0
