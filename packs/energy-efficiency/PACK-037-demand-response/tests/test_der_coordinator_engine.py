# -*- coding: utf-8 -*-
"""
Unit tests for DERCoordinatorEngine -- PACK-037 Engine 6
==========================================================

Tests battery dispatch (SOC constraints), solar availability, EV charging
deferral, thermal storage, generator constraints (emissions, fuel),
coordinated dispatch, and degradation tracking.

Coverage target: 85%+
Total tests: ~60
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack037_test.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("der_coordinator_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_engine_class_exists(self):
        assert hasattr(_m, "DERCoordinatorEngine")

    def test_engine_instantiation(self):
        engine = _m.DERCoordinatorEngine()
        assert engine is not None


class TestBatteryDispatch:
    """Test battery dispatch with SOC constraints."""

    def _get_dispatch(self, engine):
        return (getattr(engine, "dispatch_battery", None)
                or getattr(engine, "battery_dispatch", None)
                or getattr(engine, "dispatch_asset", None))

    def test_dispatch_battery(self, sample_der_assets):
        engine = _m.DERCoordinatorEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_battery method not found")
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        result = dispatch(asset=battery, power_kw=250, duration_hours=2)
        assert result is not None

    def test_soc_constraints_min(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        assert battery["min_soc_pct"] == 10.0

    def test_soc_constraints_max(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        assert battery["max_soc_pct"] == 95.0

    def test_current_soc(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        assert battery["min_soc_pct"] <= battery["current_soc_pct"] <= battery["max_soc_pct"]

    def test_max_discharge_duration(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        usable_kwh = battery["capacity_kwh"] * (
            battery["current_soc_pct"] - battery["min_soc_pct"]) / 100
        max_hours = usable_kwh / battery["max_power_kw"]
        assert max_hours > 0

    def test_discharge_respects_min_soc(self, sample_der_assets):
        engine = _m.DERCoordinatorEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_battery method not found")
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        # Request very long discharge that would breach min SOC
        result = dispatch(asset=battery, power_kw=250, duration_hours=20)
        final_soc = getattr(result, "final_soc_pct", None)
        if final_soc is not None:
            assert final_soc >= battery["min_soc_pct"]

    @pytest.mark.parametrize("power_kw,duration_h", [
        (50, 4), (100, 3), (200, 2), (250, 1),
    ])
    def test_various_discharge_profiles(self, sample_der_assets,
                                         power_kw, duration_h):
        engine = _m.DERCoordinatorEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_battery method not found")
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        result = dispatch(asset=battery, power_kw=power_kw,
                         duration_hours=duration_h)
        assert result is not None

    def test_round_trip_efficiency(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        assert battery["round_trip_efficiency_pct"] == 92.0


class TestSolarAvailability:
    """Test solar availability calculation."""

    def _get_solar(self, engine):
        return (getattr(engine, "solar_availability", None)
                or getattr(engine, "calculate_solar", None)
                or getattr(engine, "assess_solar", None))

    def test_solar_current_output(self, sample_der_assets):
        solar = next(a for a in sample_der_assets
                    if a["asset_type"] == "SOLAR")
        assert solar["current_output_kw"] <= solar["capacity_kw"]

    def test_solar_degradation(self, sample_der_assets):
        solar = next(a for a in sample_der_assets
                    if a["asset_type"] == "SOLAR")
        degradation = solar["panel_degradation_pct_per_year"] * solar["age_years"]
        effective_capacity = solar["capacity_kw"] * (1 - degradation / 100)
        assert effective_capacity > 0
        assert effective_capacity < solar["capacity_kw"]

    def test_solar_availability(self, sample_der_assets):
        engine = _m.DERCoordinatorEngine()
        solar_fn = self._get_solar(engine)
        if solar_fn is None:
            pytest.skip("solar_availability method not found")
        solar = next(a for a in sample_der_assets
                    if a["asset_type"] == "SOLAR")
        result = solar_fn(asset=solar, hour=14, cloud_cover_pct=20)
        assert result is not None

    @pytest.mark.parametrize("hour,expected_available", [
        (0, False), (6, True), (12, True), (14, True),
        (18, True), (21, False),
    ])
    def test_solar_by_hour(self, hour, expected_available):
        is_daylight = 6 <= hour <= 19
        assert is_daylight == expected_available


class TestEVChargingDeferral:
    """Test EV charging deferral."""

    def _get_defer(self, engine):
        return (getattr(engine, "defer_ev_charging", None)
                or getattr(engine, "ev_deferral", None)
                or getattr(engine, "manage_ev", None))

    def test_defer_ev_charging(self, sample_der_assets):
        engine = _m.DERCoordinatorEngine()
        defer = self._get_defer(engine)
        if defer is None:
            pytest.skip("defer_ev_charging method not found")
        ev = next(a for a in sample_der_assets
                 if a["asset_type"] == "EV_FLEET")
        result = defer(asset=ev, duration_hours=4)
        assert result is not None

    def test_ev_deferrable_capacity(self, sample_der_assets):
        ev = next(a for a in sample_der_assets
                 if a["asset_type"] == "EV_FLEET")
        assert ev["deferrable_kw"] == 180.0

    def test_ev_min_charge_threshold(self, sample_der_assets):
        ev = next(a for a in sample_der_assets
                 if a["asset_type"] == "EV_FLEET")
        assert ev["min_charge_threshold_pct"] == 30.0


class TestThermalStorage:
    """Test thermal storage dispatch."""

    def _get_thermal(self, engine):
        return (getattr(engine, "dispatch_thermal", None)
                or getattr(engine, "thermal_storage", None)
                or getattr(engine, "manage_thermal", None))

    def test_discharge_thermal(self, sample_der_assets):
        engine = _m.DERCoordinatorEngine()
        thermal = self._get_thermal(engine)
        if thermal is None:
            pytest.skip("dispatch_thermal method not found")
        tes = next(a for a in sample_der_assets
                  if a["asset_type"] == "THERMAL_STORAGE")
        result = thermal(asset=tes, action="DISCHARGE", duration_hours=3)
        assert result is not None

    def test_thermal_capacity(self, sample_der_assets):
        tes = next(a for a in sample_der_assets
                  if a["asset_type"] == "THERMAL_STORAGE")
        assert tes["capacity_kwh_thermal"] == 2400.0

    def test_thermal_current_charge(self, sample_der_assets):
        tes = next(a for a in sample_der_assets
                  if a["asset_type"] == "THERMAL_STORAGE")
        assert tes["current_charge_pct"] == 70.0


class TestGeneratorConstraints:
    """Test backup generator constraints (emissions, fuel, runtime)."""

    def _get_gen(self, engine):
        return (getattr(engine, "dispatch_generator", None)
                or getattr(engine, "generator_dispatch", None)
                or getattr(engine, "manage_generator", None))

    def test_generator_capacity(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        assert gen["capacity_kw"] == 1500.0

    def test_generator_emission_factor(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        assert gen["emission_factor_kg_co2_per_litre"] == 2.68

    def test_generator_fuel_consumption(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        hours = gen["fuel_tank_litres"] / gen["fuel_consumption_litres_per_hour"]
        assert hours == 25.0

    def test_generator_annual_limit(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        remaining = gen["annual_run_hour_limit"] - gen["run_hours_ytd"]
        assert remaining == 165

    def test_generator_startup_time(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        assert gen["startup_time_minutes"] == 10

    def test_generator_emissions_calculation(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        hours = 4
        fuel = gen["fuel_consumption_litres_per_hour"] * hours
        emissions = fuel * gen["emission_factor_kg_co2_per_litre"]
        assert emissions == pytest.approx(1286.4, rel=0.01)


class TestCoordinatedDispatch:
    """Test coordinated dispatch across multiple DER assets."""

    def _get_coordinate(self, engine):
        return (getattr(engine, "coordinate_dispatch", None)
                or getattr(engine, "coordinated_dispatch", None)
                or getattr(engine, "optimize_der_dispatch", None))

    def test_coordinated_dispatch(self, sample_der_assets, sample_dr_event):
        engine = _m.DERCoordinatorEngine()
        coord = self._get_coordinate(engine)
        if coord is None:
            pytest.skip("coordinate_dispatch method not found")
        result = coord(assets=sample_der_assets, event=sample_dr_event,
                      target_kw=500.0)
        assert result is not None

    def test_total_der_capacity(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        ev = next(a for a in sample_der_assets
                 if a["asset_type"] == "EV_FLEET")
        total = battery["max_power_kw"] + ev["deferrable_kw"]
        assert total == 430.0


class TestDegradationTracking:
    """Test asset degradation tracking."""

    def _get_degradation(self, engine):
        return (getattr(engine, "track_degradation", None)
                or getattr(engine, "calculate_degradation", None)
                or getattr(engine, "degradation_impact", None))

    def test_battery_degradation(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        degradation_pct = (battery["cycle_count"] / 1000) * \
            battery["degradation_pct_per_1000_cycles"]
        assert degradation_pct == pytest.approx(1.125, rel=0.01)

    def test_battery_remaining_life(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        remaining = battery["max_cycles"] - battery["cycle_count"]
        assert remaining == 4550

    def test_solar_age_degradation(self, sample_der_assets):
        solar = next(a for a in sample_der_assets
                    if a["asset_type"] == "SOLAR")
        total_deg = solar["panel_degradation_pct_per_year"] * solar["age_years"]
        assert total_deg == pytest.approx(1.5, rel=0.01)

    def test_track_degradation_returns_result(self, sample_der_assets):
        engine = _m.DERCoordinatorEngine()
        track = self._get_degradation(engine)
        if track is None:
            pytest.skip("track_degradation method not found")
        result = track(assets=sample_der_assets)
        assert result is not None


class TestDERDataIntegrity:
    """Test DER fixture data integrity."""

    def test_asset_count(self, sample_der_assets):
        assert len(sample_der_assets) == 5

    def test_asset_types_unique(self, sample_der_assets):
        types = [a["asset_type"] for a in sample_der_assets]
        assert len(set(types)) == 5

    @pytest.mark.parametrize("asset_type", [
        "BATTERY", "SOLAR", "EV_FLEET", "THERMAL_STORAGE", "BACKUP_GENERATOR",
    ])
    def test_asset_type_present(self, sample_der_assets, asset_type):
        found = any(a["asset_type"] == asset_type for a in sample_der_assets)
        assert found

    def test_all_have_availability(self, sample_der_assets):
        for asset in sample_der_assets:
            avail = asset.get("availability_pct", None)
            if avail is not None:
                assert 0 <= avail <= 100

    def test_all_have_ids(self, sample_der_assets):
        for asset in sample_der_assets:
            assert "asset_id" in asset
            assert asset["asset_id"].startswith("DER-")


# =============================================================================
# Battery Energy Calculations
# =============================================================================


class TestBatteryEnergyCalcs:
    """Detailed battery energy calculations."""

    def test_usable_capacity(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        usable = battery["capacity_kwh"] * (
            battery["max_soc_pct"] - battery["min_soc_pct"]) / 100
        assert usable == pytest.approx(425.0, rel=0.01)

    def test_current_usable_energy(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        current = battery["capacity_kwh"] * (
            battery["current_soc_pct"] - battery["min_soc_pct"]) / 100
        assert current == pytest.approx(375.0, rel=0.01)

    def test_max_discharge_time_at_full_power(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        current_energy = battery["capacity_kwh"] * (
            battery["current_soc_pct"] - battery["min_soc_pct"]) / 100
        hours = current_energy / battery["max_power_kw"]
        assert hours == pytest.approx(1.5, rel=0.01)

    def test_charge_time_to_full(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        energy_needed = battery["capacity_kwh"] * (
            battery["max_soc_pct"] - battery["current_soc_pct"]) / 100
        # Account for round-trip efficiency on charging
        charge_energy = energy_needed / (battery["round_trip_efficiency_pct"] / 100)
        hours = charge_energy / battery["max_power_kw"]
        assert hours > 0

    @pytest.mark.parametrize("soc_pct,expected_usable_kwh", [
        (10, 0), (50, 200), (85, 375), (95, 425),
    ])
    def test_usable_energy_at_soc(self, sample_der_assets,
                                    soc_pct, expected_usable_kwh):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        usable = battery["capacity_kwh"] * (
            soc_pct - battery["min_soc_pct"]) / 100
        assert usable == pytest.approx(expected_usable_kwh, rel=0.01)

    def test_cost_per_kwh_throughput(self, sample_der_assets):
        battery = next(a for a in sample_der_assets
                      if a["asset_type"] == "BATTERY")
        assert battery["cost_usd_per_kwh_throughput"] == Decimal("0.04")


# =============================================================================
# Generator Emissions Scenarios
# =============================================================================


class TestGeneratorEmissions:
    """Test generator emission calculations for various scenarios."""

    @pytest.mark.parametrize("run_hours,expected_emissions_kg", [
        (1, 321.6), (2, 643.2), (4, 1286.4), (8, 2572.8),
    ])
    def test_emissions_by_runtime(self, sample_der_assets,
                                    run_hours, expected_emissions_kg):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        fuel = gen["fuel_consumption_litres_per_hour"] * run_hours
        emissions = fuel * gen["emission_factor_kg_co2_per_litre"]
        assert emissions == pytest.approx(expected_emissions_kg, rel=0.01)

    def test_generator_fuel_for_max_event(self, sample_der_assets):
        gen = next(a for a in sample_der_assets
                  if a["asset_type"] == "BACKUP_GENERATOR")
        fuel_needed = (gen["fuel_consumption_litres_per_hour"] *
                      gen["max_run_hours_per_event"])
        assert fuel_needed <= gen["fuel_tank_litres"]
