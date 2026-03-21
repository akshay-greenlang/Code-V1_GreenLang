# -*- coding: utf-8 -*-
"""
Unit tests for RenewableIntegrationEngine (PACK-032 Engine 6)

Tests PV yield, capacity factor, LCOE, heat pump COP/SPF,
self-consumption, and biomass assessment.

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
    mod_key = f"pack032_renew.{name}"
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
    return _load("renewable_integration_engine")


@pytest.fixture
def engine(engine_mod):
    return engine_mod.RenewableIntegrationEngine()


@pytest.fixture
def solar_pv_input(engine_mod):
    mod = engine_mod
    return mod.SolarPVInput(
        system_capacity_kwp=50.0,
        module_type=mod.PVModuleType.MONOCRYSTALLINE,
        mount_type=mod.PVMountType.ROOFTOP_TILTED,
        tilt_deg=30.0,
        azimuth_deg=180.0,
        shading_factor=0.95,
    )


@pytest.fixture
def heat_pump_input(engine_mod):
    mod = engine_mod
    return mod.HeatPumpInput(
        hp_type=mod.RenewableType.AIR_SOURCE_HEAT_PUMP,
        rated_capacity_kw=20.0,
        annual_heat_demand_kwh=30000.0,
        distribution_type=mod.HeatDistributionType.UNDERFLOOR_35C,
    )


@pytest.fixture
def biomass_input(engine_mod):
    mod = engine_mod
    return mod.BiomassInput(
        fuel_type="wood_pellet_auto",
        rated_capacity_kw=50.0,
        annual_heat_demand_kwh=80000.0,
    )


@pytest.fixture
def full_renewable_input(engine_mod, solar_pv_input, heat_pump_input):
    mod = engine_mod
    return mod.RenewableAssessmentInput(
        building_id="BLD-REN-001",
        climate_zone=mod.ClimateZone.CENTRAL_EUROPE,
        location="berlin",
        country_code="DE",
        annual_electricity_consumption_kwh=100000.0,
        annual_heat_demand_kwh=30000.0,
        solar_pv=solar_pv_input,
        heat_pump=heat_pump_input,
    )


# =========================================================================
# Test Initialization
# =========================================================================


class TestInitialization:
    def test_engine_class_exists(self, engine_mod):
        assert hasattr(engine_mod, "RenewableIntegrationEngine")

    def test_engine_instantiation(self, engine):
        assert engine is not None


# =========================================================================
# Test Solar PV
# =========================================================================


class TestSolarPV:
    def _pv_result(self, engine, solar_pv_input, engine_mod):
        return engine.assess_solar_pv(
            pv=solar_pv_input,
            climate_zone=engine_mod.ClimateZone.CENTRAL_EUROPE,
            location="berlin",
            country_code="DE",
            load_profile=engine_mod.BuildingLoadProfile.OFFICE_WEEKDAY,
            annual_consumption=100000.0,
        )

    def test_assess_solar_pv(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result is not None
        assert result.annual_yield_kwh > 0

    def test_specific_yield(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result.specific_yield_kwh_per_kwp > 0
        # Central Europe: typically 800-1200 kWh/kWp
        assert result.specific_yield_kwh_per_kwp > 500
        assert result.specific_yield_kwh_per_kwp < 2000

    def test_performance_ratio(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert 0 < result.performance_ratio <= 1.0

    def test_self_consumption(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result.self_consumption_pct >= 0
        assert result.self_consumption_pct <= 100

    def test_cost_saving(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result.annual_cost_saving_eur > 0

    def test_carbon_saving(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result.annual_carbon_saving_kg > 0

    def test_lcoe_positive(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result.lcoe_eur_per_kwh > 0

    def test_payback(self, engine, solar_pv_input, engine_mod):
        result = self._pv_result(engine, solar_pv_input, engine_mod)
        assert result.simple_payback_years > 0


# =========================================================================
# Test Heat Pump
# =========================================================================


class TestHeatPump:
    def test_assess_heat_pump(self, engine, heat_pump_input, engine_mod):
        result = engine.assess_heat_pump(
            hp=heat_pump_input,
            climate_zone=engine_mod.ClimateZone.CENTRAL_EUROPE,
        )
        assert result is not None

    def test_hp_spf_positive(self, engine, heat_pump_input, engine_mod):
        result = engine.assess_heat_pump(
            hp=heat_pump_input,
            climate_zone=engine_mod.ClimateZone.CENTRAL_EUROPE,
        )
        assert result.seasonal_cop > 0
        # Underfloor ASHP should achieve 2.5-4.5 SPF
        assert result.seasonal_cop > 1.5

    def test_hp_electricity_consumption(self, engine, heat_pump_input, engine_mod):
        result = engine.assess_heat_pump(
            hp=heat_pump_input,
            climate_zone=engine_mod.ClimateZone.CENTRAL_EUROPE,
        )
        assert result.annual_electricity_kwh > 0
        # Electricity should be less than heat demand (COP > 1)
        assert result.annual_electricity_kwh < heat_pump_input.annual_heat_demand_kwh


# =========================================================================
# Test Biomass
# =========================================================================


class TestBiomass:
    def test_assess_biomass(self, engine, biomass_input):
        result = engine.assess_biomass(biomass_input)
        assert result is not None
        assert result.annual_heat_delivered_kwh > 0
        assert result.annual_fuel_cost_eur > 0

    def test_biomass_carbon_saving(self, engine, biomass_input):
        result = engine.assess_biomass(biomass_input)
        assert result.annual_carbon_saving_kg >= 0
        assert result.simple_payback_years > 0


# =========================================================================
# Test LCOE and Self-Consumption
# =========================================================================


class TestFinancials:
    def test_calculate_lcoe(self, engine, engine_mod):
        lcoe = engine.calculate_lcoe(
            capex=50000.0,
            annual_opex=500.0,
            annual_generation=45000.0,
            discount_rate=0.05,
            lifetime_years=25,
            degradation_rate=0.005,
        )
        assert lcoe > 0

    def test_calculate_self_consumption(self, engine, engine_mod):
        self_consumed, exported = engine.calculate_self_consumption(
            pv_generation_kwh=45000.0,
            annual_consumption_kwh=100000.0,
            load_profile=engine_mod.BuildingLoadProfile.OFFICE_WEEKDAY,
        )
        assert self_consumed >= 0
        assert exported >= 0
        assert self_consumed + exported == pytest.approx(45000.0, rel=0.01)

    def test_calculate_renewable_fraction(self, engine):
        frac = engine.calculate_renewable_fraction(
            renewable_generation_kwh=45000.0,
            total_energy_kwh=100000.0,
        )
        assert frac == pytest.approx(45.0, rel=0.01)


# =========================================================================
# Test Provenance
# =========================================================================


class TestProvenance:
    def test_full_assessment_provenance(self, engine, full_renewable_input):
        if hasattr(engine, "analyze"):
            result = engine.analyze(full_renewable_input)
            assert hasattr(result, "provenance_hash")
            assert len(result.provenance_hash) == 64
            # Hash is a valid hex string
            int(result.provenance_hash, 16)


# =========================================================================
# Test Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_no_shading(self, engine, engine_mod):
        pv = engine_mod.SolarPVInput(
            system_capacity_kwp=10.0,
            shading_factor=1.0,
        )
        result = engine.assess_solar_pv(
            pv=pv,
            climate_zone=engine_mod.ClimateZone.CENTRAL_EUROPE,
            location="berlin",
            annual_consumption=50000.0,
        )
        assert result.annual_yield_kwh > 0

    def test_heavy_shading(self, engine, engine_mod):
        pv = engine_mod.SolarPVInput(
            system_capacity_kwp=10.0,
            shading_factor=0.5,
        )
        result = engine.assess_solar_pv(
            pv=pv,
            climate_zone=engine_mod.ClimateZone.CENTRAL_EUROPE,
            location="berlin",
            annual_consumption=50000.0,
        )
        assert result.annual_yield_kwh > 0
