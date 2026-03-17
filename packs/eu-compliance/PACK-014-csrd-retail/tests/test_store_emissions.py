# -*- coding: utf-8 -*-
"""
Unit tests for StoreEmissionsEngine -- PACK-014 CSRD Retail Engine 1
=====================================================================

Tests store-level Scope 1 and Scope 2 emissions calculations including
energy consumption, refrigerant leakage, fleet emissions, multi-store
consolidation, F-gas compliance, and dual-reporting (location vs market).

Coverage target: 85%+
Total tests: ~44
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "..", "engines")


def _load(name: str):
    path = os.path.join(ENGINES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("store_emissions_engine")

StoreEmissionsEngine = _m.StoreEmissionsEngine
StoreData = _m.StoreData
StoreType = _m.StoreType
EnergySource = _m.EnergySource
EnergyConsumption = _m.EnergyConsumption
RefrigerantType = _m.RefrigerantType
RefrigerantData = _m.RefrigerantData
FleetVehicleType = _m.FleetVehicleType
FleetData = _m.FleetData
StoreEmissionsResult = _m.StoreEmissionsResult
MultiStoreConsolidationResult = _m.MultiStoreConsolidationResult
FGasComplianceResult = _m.FGasComplianceResult
GRID_EMISSION_FACTORS = _m.GRID_EMISSION_FACTORS
RESIDUAL_MIX_FACTORS = _m.RESIDUAL_MIX_FACTORS
FUEL_EMISSION_FACTORS = _m.FUEL_EMISSION_FACTORS
REFRIGERANT_GWP = _m.REFRIGERANT_GWP
TYPICAL_LEAKAGE_RATES = _m.TYPICAL_LEAKAGE_RATES
FLEET_FUEL_FACTORS = _m.FLEET_FUEL_FACTORS
FLEET_DISTANCE_FACTORS = _m.FLEET_DISTANCE_FACTORS
F_GAS_PHASE_DOWN_SCHEDULE = _m.F_GAS_PHASE_DOWN_SCHEDULE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**overrides) -> StoreData:
    """Build a minimal valid StoreData with optional overrides."""
    defaults = dict(
        store_id="S001",
        store_name="Test Store",
        store_type=StoreType.STANDARD,
        country="DE",
        floor_area_sqm=1000.0,
        employees=20,
        energy_consumption=[],
        refrigerants=[],
        fleet=[],
    )
    defaults.update(overrides)
    return StoreData(**defaults)


# ===================================================================
# TestInitialization
# ===================================================================


class TestInitialization:
    """Engine instantiation tests."""

    def test_default(self):
        engine = StoreEmissionsEngine()
        assert engine is not None

    def test_with_grid_factors(self):
        engine = StoreEmissionsEngine()
        assert engine._grid_factors is not None
        assert len(engine._grid_factors) > 0

    def test_with_fuel_factors(self):
        engine = StoreEmissionsEngine()
        assert "natural_gas" in engine._fuel_factors

    def test_with_gwp_dict(self):
        engine = StoreEmissionsEngine()
        assert RefrigerantType.R404A in engine._refrigerant_gwp


# ===================================================================
# TestStoreTypes
# ===================================================================


class TestStoreTypes:
    """Validate all 7 StoreType enum members."""

    def test_all_types_defined(self):
        assert len(StoreType) == 7

    def test_enum_values(self):
        expected = {
            "flagship", "standard", "express", "outlet",
            "warehouse", "dark_store", "pop_up",
        }
        actual = {s.value for s in StoreType}
        assert actual == expected

    def test_grocery_store_uses_standard(self):
        store = _make_store(store_type=StoreType.STANDARD)
        assert store.store_type == StoreType.STANDARD

    def test_warehouse_type(self):
        store = _make_store(store_type=StoreType.WAREHOUSE)
        assert store.store_type == StoreType.WAREHOUSE


# ===================================================================
# TestEnergyCalculation
# ===================================================================


class TestEnergyCalculation:
    """Energy-related emission calculations."""

    def test_electricity_scope2(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            country="DE",
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=100_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 100 MWh * 0.338 tCO2e/MWh = 33.8
        assert result.scope2_location_tco2e == pytest.approx(33.8, rel=1e-2)

    def test_natural_gas_scope1(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.NATURAL_GAS, quantity_kwh=50_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 50 MWh * 0.202 tCO2e/MWh = 10.1
        assert result.heating_tco2e == pytest.approx(10.1, rel=1e-2)

    def test_district_heating(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.DISTRICT_HEATING, quantity_kwh=80_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 80 MWh * 0.180 = 14.4
        assert result.scope2_location_tco2e == pytest.approx(14.4, rel=1e-2)

    def test_solar_avoided_emissions(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            country="DE",
            energy_consumption=[
                EnergyConsumption(source=EnergySource.SOLAR_PV, quantity_kwh=20_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # Avoided = 20 MWh * 0.338 = 6.76
        assert result.renewable_generation_tco2e_avoided == pytest.approx(6.76, rel=1e-2)

    def test_total_energy(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=60_000),
                EnergyConsumption(source=EnergySource.NATURAL_GAS, quantity_kwh=40_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert result.energy_intensity_kwh_per_sqm == pytest.approx(100.0, rel=1e-2)

    def test_energy_intensity_kwh_per_sqm(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            floor_area_sqm=500.0,
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=75_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 75_000 / 500 = 150
        assert result.energy_intensity_kwh_per_sqm == pytest.approx(150.0, rel=1e-2)


# ===================================================================
# TestRefrigerantEmissions
# ===================================================================


class TestRefrigerantEmissions:
    """Refrigerant leakage emission tests."""

    def test_r404a_leakage(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            refrigerants=[
                RefrigerantData(
                    refrigerant_type=RefrigerantType.R404A,
                    charge_kg=200.0,
                    leakage_rate_pct=15.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # leakage = 200 * 15/100 = 30 kg, emissions = 30 * 3922 / 1000 = 117.66
        assert result.refrigerant_tco2e == pytest.approx(117.66, rel=1e-2)

    def test_r134a_leakage(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            refrigerants=[
                RefrigerantData(
                    refrigerant_type=RefrigerantType.R134A,
                    charge_kg=100.0,
                    leakage_rate_pct=10.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # leakage = 100 * 10/100 = 10 kg, emissions = 10 * 1430 / 1000 = 14.3
        assert result.refrigerant_tco2e == pytest.approx(14.3, rel=1e-2)

    def test_top_up_method(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            refrigerants=[
                RefrigerantData(
                    refrigerant_type=RefrigerantType.R404A,
                    charge_kg=200.0,
                    top_up_kg=50.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # top-up overrides: 50 * 3922 / 1000 = 196.1
        assert result.refrigerant_tco2e == pytest.approx(196.1, rel=1e-2)

    def test_gwp_values(self):
        assert REFRIGERANT_GWP[RefrigerantType.R404A] == 3922
        assert REFRIGERANT_GWP[RefrigerantType.R134A] == 1430
        assert REFRIGERANT_GWP[RefrigerantType.R290] == 3
        assert REFRIGERANT_GWP[RefrigerantType.R744_CO2] == 1

    def test_leakage_rates(self):
        assert TYPICAL_LEAKAGE_RATES[StoreType.FLAGSHIP] == 18.0
        assert TYPICAL_LEAKAGE_RATES[StoreType.WAREHOUSE] == 10.0
        assert TYPICAL_LEAKAGE_RATES[StoreType.POP_UP] == 25.0


# ===================================================================
# TestFleetEmissions
# ===================================================================


class TestFleetEmissions:
    """Fleet vehicle emission tests."""

    def test_delivery_van(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            fleet=[
                FleetData(
                    vehicle_type=FleetVehicleType.DELIVERY_VAN,
                    count=5,
                    fuel_consumption_litres=10_000.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 10000 * 0.002676 = 26.76 tCO2e
        assert result.fleet_tco2e == pytest.approx(26.76, rel=1e-2)

    def test_electric_van_zero(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            fleet=[
                FleetData(
                    vehicle_type=FleetVehicleType.ELECTRIC_VAN,
                    count=3,
                    distance_km=50_000.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert result.fleet_tco2e == pytest.approx(0.0, abs=1e-6)

    def test_forklift(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            fleet=[
                FleetData(
                    vehicle_type=FleetVehicleType.FORKLIFT_DIESEL,
                    count=2,
                    fuel_consumption_litres=2_000.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 2000 * 0.002676 = 5.352
        assert result.fleet_tco2e == pytest.approx(5.352, rel=1e-2)

    def test_total_fleet(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            fleet=[
                FleetData(
                    vehicle_type=FleetVehicleType.DELIVERY_VAN,
                    count=3,
                    fuel_consumption_litres=5_000.0,
                ),
                FleetData(
                    vehicle_type=FleetVehicleType.TRUCK,
                    count=1,
                    fuel_consumption_litres=8_000.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 5000*0.002676 + 8000*0.002676 = 13.38 + 21.408 = 34.788
        assert result.fleet_tco2e == pytest.approx(34.788, rel=1e-2)


# ===================================================================
# TestMultiStoreConsolidation
# ===================================================================


class TestMultiStoreConsolidation:
    """Multi-store consolidation tests."""

    def _two_stores(self):
        s1 = _make_store(
            store_id="A", store_name="Store A",
            store_type=StoreType.STANDARD, country="DE",
            floor_area_sqm=1000.0, employees=20,
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=100_000),
            ],
        )
        s2 = _make_store(
            store_id="B", store_name="Store B",
            store_type=StoreType.FLAGSHIP, country="FR",
            floor_area_sqm=2000.0, employees=40,
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=200_000),
            ],
        )
        return [s1, s2]

    def test_two_stores_sum(self):
        engine = StoreEmissionsEngine()
        stores = self._two_stores()
        consolidated = engine.consolidate_stores(stores)
        assert consolidated.total_stores == 2
        expected = (100 * 0.338) + (200 * 0.055)  # DE + FR grid factors
        assert consolidated.total_tco2e == pytest.approx(expected, rel=1e-2)

    def test_drill_down_by_type(self):
        engine = StoreEmissionsEngine()
        stores = self._two_stores()
        consolidated = engine.consolidate_stores(stores)
        assert "standard" in consolidated.by_store_type
        assert "flagship" in consolidated.by_store_type

    def test_drill_down_by_country(self):
        engine = StoreEmissionsEngine()
        stores = self._two_stores()
        consolidated = engine.consolidate_stores(stores)
        assert "DE" in consolidated.by_country
        assert "FR" in consolidated.by_country

    def test_emissions_per_sqm(self):
        engine = StoreEmissionsEngine()
        stores = self._two_stores()
        consolidated = engine.consolidate_stores(stores)
        total_sqm = 1000.0 + 2000.0
        assert consolidated.avg_emissions_per_sqm == pytest.approx(
            consolidated.total_tco2e / total_sqm, rel=1e-4
        )

    def test_emissions_per_employee(self):
        engine = StoreEmissionsEngine()
        stores = self._two_stores()
        consolidated = engine.consolidate_stores(stores)
        total_emp = 20 + 40
        assert consolidated.avg_emissions_per_employee == pytest.approx(
            consolidated.total_tco2e / total_emp, rel=1e-4
        )


# ===================================================================
# TestFGasCompliance
# ===================================================================


class TestFGasCompliance:
    """F-gas Regulation compliance tests."""

    def test_compliant_facility(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            refrigerants=[
                RefrigerantData(
                    refrigerant_type=RefrigerantType.R290,
                    charge_kg=50.0,
                    leakage_rate_pct=5.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert result.fgas_compliance is not None
        # R290 is natural with GWP=3, not an HFC -> compliant
        assert result.fgas_compliance.compliant is True

    def test_non_compliant(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            refrigerants=[
                RefrigerantData(
                    refrigerant_type=RefrigerantType.R404A,
                    charge_kg=500.0,
                    leakage_rate_pct=15.0,
                ),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert result.fgas_compliance is not None
        # R404A GWP=3922, avg > 2500, charge > 10 => non-compliant
        assert result.fgas_compliance.compliant is False

    def test_phase_down_schedule(self):
        assert F_GAS_PHASE_DOWN_SCHEDULE[2030] == 21.0
        assert F_GAS_PHASE_DOWN_SCHEDULE[2050] == 0.0


# ===================================================================
# TestScope2Methods
# ===================================================================


class TestScope2Methods:
    """Location-based vs market-based Scope 2 tests."""

    def test_location_based(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            country="DE",
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=100_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # 100 MWh * 0.338 = 33.8
        assert result.scope2_location_tco2e == pytest.approx(33.8, rel=1e-2)

    def test_market_based(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            country="DE",
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=100_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # Residual mix DE = 0.493 -> 100 * 0.493 = 49.3
        assert result.scope2_market_tco2e == pytest.approx(49.3, rel=1e-2)

    def test_dual_reporting(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            country="DE",
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=100_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert result.scope2_location_tco2e != result.scope2_market_tco2e

    def test_renewable_ppa(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            country="DE",
            has_ppa=True,
            ppa_emission_factor=0.010,
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=100_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        # Market-based with PPA: 100 MWh * 0.010 = 1.0
        assert result.scope2_market_tco2e == pytest.approx(1.0, rel=1e-2)


# ===================================================================
# TestProvenance
# ===================================================================


class TestProvenance:
    """Provenance hash tests."""

    def test_hash_64char(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=50_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=50_000),
            ],
        )
        r1 = engine.calculate_store_emissions(store)
        r2 = engine.calculate_store_emissions(store)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input(self):
        engine = StoreEmissionsEngine()
        s1 = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=50_000),
            ],
        )
        s2 = _make_store(
            store_id="S002",
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=99_000),
            ],
        )
        r1 = engine.calculate_store_emissions(s1)
        r2 = engine.calculate_store_emissions(s2)
        assert r1.provenance_hash != r2.provenance_hash


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_store(self):
        engine = StoreEmissionsEngine()
        store = _make_store()
        result = engine.calculate_store_emissions(store)
        assert result.total_tco2e == pytest.approx(0.0, abs=1e-6)

    def test_zero_area_not_allowed(self):
        with pytest.raises(Exception):
            _make_store(floor_area_sqm=0.0)

    def test_large_store_chain(self):
        engine = StoreEmissionsEngine()
        stores = [
            _make_store(
                store_id=f"S{i:03d}",
                store_name=f"Store {i}",
                energy_consumption=[
                    EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=10_000),
                ],
            )
            for i in range(50)
        ]
        consolidated = engine.consolidate_stores(stores)
        assert consolidated.total_stores == 50
        assert consolidated.total_tco2e > 0

    def test_result_fields(self):
        engine = StoreEmissionsEngine()
        store = _make_store(
            energy_consumption=[
                EnergyConsumption(source=EnergySource.ELECTRICITY, quantity_kwh=10_000),
            ],
        )
        result = engine.calculate_store_emissions(store)
        assert isinstance(result, StoreEmissionsResult)
        assert result.store_id == "S001"
        assert result.store_name == "Test Store"
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0
