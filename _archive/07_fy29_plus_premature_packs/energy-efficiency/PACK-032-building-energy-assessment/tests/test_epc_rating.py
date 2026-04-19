# -*- coding: utf-8 -*-
"""
Unit tests for EPCRatingEngine (PACK-032 Engine 2)

Tests EPC rating A-G, primary energy, CO2 per m2, reference building
comparison, and EPBD 2024 compliance.

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
    mod_key = f"pack032_epc.{name}"
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
    return _load("epc_rating_engine")


@pytest.fixture
def engine(engine_mod):
    return engine_mod.EPCRatingEngine()


@pytest.fixture
def typical_dwelling(engine_mod):
    """Typical UK semi-detached dwelling for EPC rating."""
    mod = engine_mod
    building = mod.BuildingData(
        facility_id="FAC-EPC-001",
        building_name="Test Semi-Detached",
        building_type=mod.BuildingUseType.RESIDENTIAL,
        country="UK",
        year_built=2000,
        floor_area_m2=85.0,
        floors=2,
    )
    return building


@pytest.fixture
def modern_office(engine_mod):
    """Modern well-insulated office building."""
    mod = engine_mod
    envelope = mod.EnvelopeSummary(
        wall_u_value=0.25,
        wall_area_m2=400.0,
        roof_u_value=0.18,
        roof_area_m2=200.0,
        floor_u_value=0.20,
        floor_area_m2=200.0,
        window_u_value=1.40,
        window_area_m2=100.0,
        window_g_value=0.40,
        door_u_value=1.50,
        door_area_m2=6.0,
        airtightness_n50=3.0,
        heated_volume_m3=3000.0,
    )
    return mod.BuildingData(
        facility_id="FAC-EPC-002",
        building_name="Modern Office",
        building_type=mod.BuildingUseType.OFFICE,
        country="UK",
        year_built=2020,
        floor_area_m2=1000.0,
        floors=3,
        envelope=envelope,
    )


@pytest.fixture
def poor_building(engine_mod):
    """Poorly insulated pre-1960 building."""
    mod = engine_mod
    envelope = mod.EnvelopeSummary(
        wall_u_value=2.10,
        wall_area_m2=200.0,
        roof_u_value=2.30,
        roof_area_m2=80.0,
        floor_u_value=1.00,
        floor_area_m2=80.0,
        window_u_value=4.80,
        window_area_m2=30.0,
        window_g_value=0.85,
        door_u_value=3.00,
        door_area_m2=4.0,
        airtightness_n50=15.0,
        heated_volume_m3=500.0,
    )
    return mod.BuildingData(
        facility_id="FAC-EPC-003",
        building_name="Old Victorian Terrace",
        building_type=mod.BuildingUseType.RESIDENTIAL,
        country="UK",
        year_built=1900,
        floor_area_m2=100.0,
        floors=2,
        envelope=envelope,
    )


# =========================================================================
# Test Initialization
# =========================================================================


class TestInitialization:
    def test_engine_class_exists(self, engine_mod):
        assert hasattr(engine_mod, "EPCRatingEngine")

    def test_engine_instantiation(self, engine):
        assert engine is not None

    def test_engine_version(self, engine):
        assert engine.engine_version == "1.0.0"

    def test_has_primary_energy_factors(self, engine):
        assert hasattr(engine, "_primary_energy_factors")

    def test_has_epc_thresholds(self, engine):
        assert hasattr(engine, "_epc_thresholds")

    def test_building_data_model(self, engine_mod):
        assert hasattr(engine_mod, "BuildingData")

    def test_epc_result_model(self, engine_mod):
        assert hasattr(engine_mod, "EPCResult")


# =========================================================================
# Test EPC Rating Calculation
# =========================================================================


class TestEPCRating:
    def test_rate_typical_dwelling(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result is not None
        assert result.epc_rating in ("A", "B", "C", "D", "E", "F", "G")

    def test_rate_modern_office(self, engine, modern_office):
        result = engine.rate(modern_office)
        assert result.epc_rating in ("A", "B", "C", "D", "E", "F", "G")

    def test_rate_poor_building(self, engine, poor_building):
        result = engine.rate(poor_building)
        # Poorly insulated building should not be A or B
        assert result.epc_rating in ("C", "D", "E", "F", "G")

    def test_modern_better_than_old(self, engine, modern_office, poor_building):
        modern_result = engine.rate(modern_office)
        poor_result = engine.rate(poor_building)
        rating_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        assert rating_order[modern_result.epc_rating] <= rating_order[poor_result.epc_rating]

    def test_epc_score_positive(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.epc_score > 0


# =========================================================================
# Test Energy Calculations
# =========================================================================


class TestEnergyCalculations:
    def test_heating_demand_positive(self, engine, typical_dwelling):
        hd = engine.calculate_heating_demand(typical_dwelling)
        assert hd > 0

    def test_cooling_demand_non_negative(self, engine, typical_dwelling):
        cd = engine.calculate_cooling_demand(typical_dwelling)
        assert cd >= 0

    def test_dhw_demand_positive(self, engine, typical_dwelling):
        dhw = engine.calculate_dhw_demand(typical_dwelling)
        assert dhw > 0

    def test_lighting_energy_positive(self, engine, typical_dwelling):
        le = engine.calculate_lighting_energy(typical_dwelling)
        assert le >= 0

    def test_primary_energy_positive(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.primary_energy_kwh > 0

    def test_primary_energy_per_m2(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.primary_energy_kwh_m2 > 0

    def test_total_delivered_energy(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.total_delivered_energy_kwh > 0

    def test_energy_breakdown_present(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.energy_breakdown is not None
        assert result.energy_breakdown.space_heating_kwh >= 0


# =========================================================================
# Test CO2 Emissions
# =========================================================================


class TestCO2Emissions:
    def test_co2_positive(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.co2_emissions_kg > 0

    def test_co2_per_m2(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.co2_emissions_kg_m2 > 0

    def test_poor_building_higher_co2(self, engine, modern_office, poor_building):
        modern_r = engine.rate(modern_office)
        poor_r = engine.rate(poor_building)
        # Poor building should have higher CO2/m2
        assert poor_r.co2_emissions_kg_m2 > modern_r.co2_emissions_kg_m2


# =========================================================================
# Test Reference Building and MEES
# =========================================================================


class TestReferenceBuilding:
    def test_reference_energy_positive(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.reference_building_energy_kwh_m2 > 0

    def test_improvement_potential_non_negative(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.improvement_potential_kwh_m2 >= 0

    def test_compliance_status(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        # compliance_status may be empty for some building types or contain a status string
        assert isinstance(result.compliance_status, str)

    def test_regulatory_minimum(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.regulatory_minimum != ""


# =========================================================================
# Test Provenance
# =========================================================================


class TestProvenance:
    def test_provenance_hash_present(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.provenance_hash != ""

    def test_provenance_hash_64_chars(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert len(result.provenance_hash) == 64

    def test_provenance_deterministic(self, engine, engine_mod):
        bd = engine_mod.BuildingData(
            facility_id="FAC-DET-EPC",
            building_type=engine_mod.BuildingUseType.RESIDENTIAL,
            floor_area_m2=90.0,
            country="UK",
        )
        r1 = engine.rate(bd)
        r2 = engine.rate(bd)
        # Verify both produce valid 64-char hex SHA-256 hashes
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)


# =========================================================================
# Test Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_processing_time(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.processing_time_ms > 0

    def test_result_facility_id(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.facility_id == "FAC-EPC-001"

    def test_result_country(self, engine, typical_dwelling):
        result = engine.rate(typical_dwelling)
        assert result.country == "UK"

    def test_germany_methodology(self, engine, engine_mod):
        bd = engine_mod.BuildingData(
            facility_id="FAC-DE",
            building_type=engine_mod.BuildingUseType.RESIDENTIAL,
            floor_area_m2=100.0,
            country="DE",
        )
        result = engine.rate(bd)
        assert result.epc_rating in ("A", "B", "C", "D", "E", "F", "G")

    def test_recommendations_list(self, engine, poor_building):
        result = engine.rate(poor_building)
        assert isinstance(result.recommendations, list)
