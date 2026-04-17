# -*- coding: utf-8 -*-
"""Tests for EPA GHG Emission Factors Hub parser (F010)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.epa_ghg_hub import (
    parse_epa_ghg_hub,
    iter_stationary_combustion,
    iter_mobile_combustion,
    iter_electricity,
    iter_steam_and_heat,
    iter_scope3_upstream,
    _normalize_fuel,
    _normalize_unit,
    _safe_float,
)
from greenlang.factors.etl.qa import validate_factor_dict


# ---- Fixtures ----

@pytest.fixture
def epa_metadata():
    return {
        "source": "EPA",
        "version": "2024",
        "source_publication": "EPA GHG Emission Factors Hub",
        "source_url": "https://www.epa.gov/ghgemissionfactors",
    }


@pytest.fixture
def epa_full_payload(epa_metadata):
    """Realistic EPA Hub payload with all 5 sections."""
    return {
        "metadata": epa_metadata,
        "stationary_combustion": [
            {
                "fuel_type": "Natural Gas",
                "unit": "scf",
                "co2_factor": 0.05444,
                "ch4_factor": 0.00000103,
                "n2o_factor": 0.0000001,
                "geography": "US",
            },
            {
                "fuel_type": "Distillate Fuel Oil No. 2",
                "unit": "gallon",
                "co2_factor": 10.18,
                "ch4_factor": 0.00041,
                "n2o_factor": 0.000082,
                "geography": "US",
            },
            {
                "fuel_type": "Motor Gasoline",
                "unit": "gallon",
                "co2_factor": 8.78,
                "ch4_factor": 0.00035,
                "n2o_factor": 0.000082,
                "geography": "US",
            },
            {
                "fuel_type": "Propane",
                "unit": "gallon",
                "co2_factor": 5.72,
                "ch4_factor": 0.00023,
                "n2o_factor": 0.000046,
                "geography": "US",
            },
            {
                "fuel_type": "Kerosene",
                "unit": "gallon",
                "co2_factor": 9.75,
                "ch4_factor": 0.00039,
                "n2o_factor": 0.000078,
                "geography": "US",
            },
            {
                "fuel_type": "Anthracite Coal",
                "unit": "short ton",
                "co2_factor": 2602.42,
                "ch4_factor": 0.024,
                "n2o_factor": 0.035,
                "geography": "US",
            },
            {
                "fuel_type": "Bituminous Coal",
                "unit": "short ton",
                "co2_factor": 2328.44,
                "ch4_factor": 0.024,
                "n2o_factor": 0.035,
                "geography": "US",
            },
            {
                "fuel_type": "Sub-Bituminous Coal",
                "unit": "short ton",
                "co2_factor": 1685.72,
                "ch4_factor": 0.024,
                "n2o_factor": 0.035,
                "geography": "US",
            },
            {
                "fuel_type": "LPG",
                "unit": "gallon",
                "co2_factor": 5.68,
                "ch4_factor": 0.00023,
                "n2o_factor": 0.000046,
                "geography": "US",
            },
            {
                "fuel_type": "Residual Fuel Oil No. 6",
                "unit": "gallon",
                "co2_factor": 11.27,
                "ch4_factor": 0.00045,
                "n2o_factor": 0.000091,
                "geography": "US",
            },
        ],
        "mobile_combustion": [
            {
                "fuel_type": "Motor Gasoline",
                "vehicle_type": "passenger_car",
                "unit": "gallon",
                "co2_factor": 8.78,
                "ch4_factor": 0.000247,
                "n2o_factor": 0.000221,
                "geography": "US",
            },
            {
                "fuel_type": "Diesel Fuel",
                "vehicle_type": "light_duty_truck",
                "unit": "gallon",
                "co2_factor": 10.21,
                "ch4_factor": 0.000051,
                "n2o_factor": 0.000121,
                "geography": "US",
            },
            {
                "fuel_type": "Jet Fuel",
                "vehicle_type": "aircraft",
                "unit": "gallon",
                "co2_factor": 9.75,
                "ch4_factor": 0.00015,
                "n2o_factor": 0.00010,
                "geography": "US",
            },
            {
                "fuel_type": "Residual Fuel Oil No. 6",
                "vehicle_type": "ocean_vessel",
                "unit": "gallon",
                "co2_factor": 11.27,
                "ch4_factor": 0.00057,
                "n2o_factor": 0.000018,
                "geography": "US",
            },
        ],
        "electricity": [
            {
                "subregion": "AKGD",
                "co2_factor": 0.000476,
                "ch4_factor": 0.0000000227,
                "n2o_factor": 0.00000000608,
                "geography_level": "grid_zone",
            },
            {
                "subregion": "CAMX",
                "co2_factor": 0.000225,
                "ch4_factor": 0.0000000085,
                "n2o_factor": 0.00000000198,
                "geography_level": "grid_zone",
            },
            {
                "subregion": "ERCT",
                "co2_factor": 0.000379,
                "ch4_factor": 0.0000000199,
                "n2o_factor": 0.00000000407,
                "geography_level": "grid_zone",
            },
            {
                "subregion": "FRCC",
                "co2_factor": 0.000401,
                "ch4_factor": 0.0000000215,
                "n2o_factor": 0.00000000354,
                "geography_level": "grid_zone",
            },
            {
                "state": "CA",
                "co2_factor": 0.000210,
                "ch4_factor": 0.0000000078,
                "n2o_factor": 0.00000000185,
                "geography_level": "state",
            },
            {
                "state": "TX",
                "co2_factor": 0.000395,
                "ch4_factor": 0.0000000200,
                "n2o_factor": 0.00000000410,
                "geography_level": "state",
            },
        ],
        "steam_and_heat": [
            {
                "fuel_type": "steam",
                "unit": "mmBtu",
                "co2_factor": 66.33,
                "ch4_factor": 0.00125,
                "n2o_factor": 0.000975,
                "geography": "US",
            },
        ],
        "scope3_upstream": [
            {
                "fuel_type": "Natural Gas",
                "unit": "scf",
                "co2_factor": 0.01228,
                "ch4_factor": 0.000237,
                "n2o_factor": 0.0000000423,
                "geography": "US",
            },
            {
                "fuel_type": "Motor Gasoline",
                "unit": "gallon",
                "co2_factor": 2.13,
                "ch4_factor": 0.0104,
                "n2o_factor": 0.0000279,
                "geography": "US",
            },
            {
                "fuel_type": "Diesel Fuel",
                "unit": "gallon",
                "co2_factor": 2.63,
                "ch4_factor": 0.0122,
                "n2o_factor": 0.0000343,
                "geography": "US",
            },
        ],
    }


# ---- Unit/Fuel normalization ----


def test_normalize_fuel_known():
    assert _normalize_fuel("Natural Gas") == "natural_gas"
    assert _normalize_fuel("Motor Gasoline") == "gasoline"
    assert _normalize_fuel("Distillate Fuel Oil No. 2") == "diesel"
    assert _normalize_fuel("Jet Fuel") == "jet_fuel"
    assert _normalize_fuel("LPG") == "lpg"


def test_normalize_fuel_unknown_falls_back_to_slug():
    result = _normalize_fuel("Some New Fuel Type 2025")
    assert result == "some_new_fuel_type_2025"


def test_normalize_unit_known():
    assert _normalize_unit("scf") == "scf"
    assert _normalize_unit("gallon") == "gallons"
    assert _normalize_unit("short ton") == "tonnes"
    assert _normalize_unit("mmBtu") == "mmbtu"
    assert _normalize_unit("kwh") == "kwh"


def test_normalize_unit_unknown_falls_back_to_slug():
    assert _normalize_unit("weird_unit") == "weird_unit"


def test_safe_float():
    assert _safe_float(3.14) == 3.14
    assert _safe_float("2.5") == 2.5
    assert _safe_float(None) == 0.0
    assert _safe_float("not_a_number") == 0.0
    assert _safe_float(None, 99.0) == 99.0


# ---- Section parsers ----


def test_stationary_combustion_produces_valid_factors(epa_metadata):
    rows = [
        {
            "fuel_type": "Natural Gas",
            "unit": "scf",
            "co2_factor": 0.05444,
            "ch4_factor": 0.00000103,
            "n2o_factor": 0.0000001,
            "geography": "US",
        },
    ]
    results = list(iter_stationary_combustion(rows, epa_metadata, 2024))
    assert len(results) == 1
    f = results[0]
    assert f["factor_id"] == "EF:EPA:stat_natural_gas:US:2024:v1"
    assert f["fuel_type"] == "natural_gas"
    assert f["unit"] == "scf"
    assert f["scope"] == "1"
    assert f["boundary"] == "combustion"
    assert f["vectors"]["CO2"] == 0.05444
    ok, errs = validate_factor_dict(f)
    assert ok, errs


def test_mobile_combustion_includes_vehicle_type(epa_metadata):
    rows = [
        {
            "fuel_type": "Motor Gasoline",
            "vehicle_type": "passenger_car",
            "unit": "gallon",
            "co2_factor": 8.78,
            "ch4_factor": 0.000247,
            "n2o_factor": 0.000221,
        },
    ]
    results = list(iter_mobile_combustion(rows, epa_metadata, 2024))
    assert len(results) == 1
    f = results[0]
    assert "mob_gasoline_passenger_car" in f["factor_id"]
    assert "mobile_combustion" in f["tags"]
    assert "passenger_car" in f["tags"]


def test_electricity_grid_zone_factor(epa_metadata):
    rows = [
        {
            "subregion": "CAMX",
            "co2_factor": 0.000225,
            "ch4_factor": 0.0000000085,
            "n2o_factor": 0.00000000198,
            "geography_level": "grid_zone",
        },
    ]
    results = list(iter_electricity(rows, epa_metadata, 2024))
    assert len(results) == 1
    f = results[0]
    assert f["factor_id"] == "EF:EPA:elec:CAMX:2024:v1"
    assert f["scope"] == "2"
    assert f["geography_level"] == "grid_zone"
    assert f["unit"] == "kwh"


def test_electricity_state_level(epa_metadata):
    rows = [
        {
            "state": "CA",
            "co2_factor": 0.000210,
            "ch4_factor": 0.0000000078,
            "n2o_factor": 0.00000000185,
            "geography_level": "state",
        },
    ]
    results = list(iter_electricity(rows, epa_metadata, 2024))
    assert len(results) == 1
    f = results[0]
    assert f["geography"] == "CA"
    assert f["geography_level"] == "state"


def test_steam_and_heat_factor(epa_metadata):
    rows = [
        {
            "fuel_type": "steam",
            "unit": "mmBtu",
            "co2_factor": 66.33,
            "ch4_factor": 0.00125,
            "n2o_factor": 0.000975,
        },
    ]
    results = list(iter_steam_and_heat(rows, epa_metadata, 2024))
    assert len(results) == 1
    f = results[0]
    assert f["scope"] == "2"
    assert "steam_heat" in f["tags"]


def test_scope3_upstream_wtt(epa_metadata):
    rows = [
        {
            "fuel_type": "Natural Gas",
            "unit": "scf",
            "co2_factor": 0.01228,
            "ch4_factor": 0.000237,
            "n2o_factor": 0.0000000423,
        },
    ]
    results = list(iter_scope3_upstream(rows, epa_metadata, 2024))
    assert len(results) == 1
    f = results[0]
    assert f["scope"] == "3"
    assert f["boundary"] == "WTT"
    assert "wtt" in f["tags"]


# ---- Full parser ----


def test_parse_full_payload(epa_full_payload):
    """Parse full payload and verify total factor count."""
    factors = parse_epa_ghg_hub(epa_full_payload)
    # 10 stationary + 4 mobile + 6 electricity + 1 steam + 3 scope3 = 24
    assert len(factors) == 24


def test_all_factors_pass_qa(epa_full_payload):
    """Every factor from the EPA parser passes QA validation."""
    factors = parse_epa_ghg_hub(epa_full_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(epa_full_payload):
    """All factor IDs are unique within a parse run."""
    factors = parse_epa_ghg_hub(epa_full_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_all_factors_have_required_fields(epa_full_payload):
    """Every factor has the core required fields."""
    factors = parse_epa_ghg_hub(epa_full_payload)
    for f in factors:
        assert f["factor_id"].startswith("EF:")
        assert f["vectors"]["CO2"] >= 0
        assert f["vectors"]["CH4"] >= 0
        assert f["vectors"]["N2O"] >= 0
        assert f["gwp_100yr"]["gwp_set"] == "IPCC_AR6_100"
        assert f["license_info"]["redistribution_allowed"] is True
        assert f["provenance"]["source_org"] == "EPA"
        assert "epa" in f["tags"]


def test_empty_payload_returns_empty():
    factors = parse_epa_ghg_hub({})
    assert factors == []


def test_metadata_version_year_extraction():
    data = {
        "metadata": {"version": "2025"},
        "stationary_combustion": [
            {
                "fuel_type": "Natural Gas",
                "unit": "scf",
                "co2_factor": 0.054,
                "ch4_factor": 0.000001,
                "n2o_factor": 0.0000001,
            },
        ],
    }
    factors = parse_epa_ghg_hub(data)
    assert factors[0]["factor_id"].endswith(":2025:v1")
    assert factors[0]["valid_from"] == "2025-01-01"


def test_duplicate_factor_ids_are_deduplicated():
    """If two rows produce the same factor_id, only the first is kept."""
    data = {
        "metadata": {"version": "2024"},
        "stationary_combustion": [
            {"fuel_type": "Natural Gas", "unit": "scf", "co2_factor": 0.054, "ch4_factor": 0.000001, "n2o_factor": 0.0000001},
            {"fuel_type": "Natural Gas", "unit": "scf", "co2_factor": 0.055, "ch4_factor": 0.000001, "n2o_factor": 0.0000001},
        ],
    }
    factors = parse_epa_ghg_hub(data)
    assert len(factors) == 1
    assert factors[0]["vectors"]["CO2"] == 0.054  # first wins


def test_compliance_frameworks_set(epa_full_payload):
    factors = parse_epa_ghg_hub(epa_full_payload)
    for f in factors:
        assert "GHG_Protocol" in f["compliance_frameworks"]
        assert "EPA_MRR" in f["compliance_frameworks"]
