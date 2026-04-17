# -*- coding: utf-8 -*-
"""Tests for DESNZ/DEFRA full UK parser (F012)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.desnz_uk import parse_desnz_uk
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def desnz_full_payload():
    return {
        "metadata": {"source": "DESNZ", "version": "2024", "region": "UK"},
        "scope1_fuels": [
            {"fuel_type": "Natural Gas", "unit": "kwh", "co2_factor": 0.18293, "ch4_factor": 0.00034, "n2o_factor": 0.00026},
            {"fuel_type": "Gas Oil", "unit": "litres", "co2_factor": 2.75776, "ch4_factor": 0.00031, "n2o_factor": 0.00805},
            {"fuel_type": "Kerosene", "unit": "litres", "co2_factor": 2.54039, "ch4_factor": 0.00020, "n2o_factor": 0.00596},
            {"fuel_type": "Diesel", "unit": "litres", "co2_factor": 2.70398, "ch4_factor": 0.00027, "n2o_factor": 0.02647},
            {"fuel_type": "Petrol", "unit": "litres", "co2_factor": 2.31440, "ch4_factor": 0.00348, "n2o_factor": 0.04853},
            {"fuel_type": "LPG", "unit": "litres", "co2_factor": 1.55537, "ch4_factor": 0.00058, "n2o_factor": 0.00100},
            {"fuel_type": "Coal Industrial", "unit": "kg", "co2_factor": 2.25670, "ch4_factor": 0.00014, "n2o_factor": 0.03390},
            {"fuel_type": "Fuel Oil", "unit": "litres", "co2_factor": 3.17851, "ch4_factor": 0.00098, "n2o_factor": 0.00774},
        ],
        "scope1_bioenergy": [
            {"fuel_type": "Wood Chips", "unit": "kg", "co2_factor": 0.01077, "ch4_factor": 0.00059, "n2o_factor": 0.00350, "biogenic_co2": 1.31985},
            {"fuel_type": "Biogas", "unit": "kwh", "co2_factor": 0.00022, "ch4_factor": 0.00002, "n2o_factor": 0.00026},
        ],
        "scope2_electricity": [
            {"type": "grid_average", "co2_factor": 0.20707, "ch4_factor": 0.00026, "n2o_factor": 0.00219},
            {"type": "generation", "co2_factor": 0.19338, "ch4_factor": 0.00023, "n2o_factor": 0.00199},
            {"type": "transmission_distribution", "co2_factor": 0.01369, "ch4_factor": 0.00003, "n2o_factor": 0.00020},
        ],
        "scope2_heat_steam": [
            {"type": "district_heat", "unit": "kwh", "co2_factor": 0.17090, "ch4_factor": 0.00041, "n2o_factor": 0.00087},
            {"type": "steam", "unit": "kwh", "co2_factor": 0.19400, "ch4_factor": 0.00054, "n2o_factor": 0.00098},
        ],
        "scope3_wtt": [
            {"activity": "natural_gas_wtt", "unit": "kwh", "co2_factor": 0.02540, "ch4_factor": 0.00071, "n2o_factor": 0.00004},
            {"activity": "diesel_wtt", "unit": "litres", "co2_factor": 0.62574, "ch4_factor": 0.01050, "n2o_factor": 0.00081},
        ],
        "scope3_freight": [
            {"activity": "hgv_all_diesel", "unit": "tonne_km", "co2_factor": 0.10432, "ch4_factor": 0.00001, "n2o_factor": 0.00154},
            {"activity": "rail_freight", "unit": "tonne_km", "co2_factor": 0.02724, "ch4_factor": 0.00001, "n2o_factor": 0.00041},
        ],
        "scope3_business_travel": [
            {"activity": "domestic_flight", "unit": "km", "co2_factor": 0.24587, "ch4_factor": 0.00002, "n2o_factor": 0.00236},
            {"activity": "short_haul_flight", "unit": "km", "co2_factor": 0.15353, "ch4_factor": 0.00001, "n2o_factor": 0.00147},
            {"activity": "long_haul_flight", "unit": "km", "co2_factor": 0.19309, "ch4_factor": 0.00001, "n2o_factor": 0.00185},
            {"activity": "hotel_night_uk", "unit": "km", "co2_factor": 10.20, "ch4_factor": 0.001, "n2o_factor": 0.003},
        ],
        "scope3_water": [
            {"activity": "water_supply", "unit": "m3", "co2_factor": 0.14900, "ch4_factor": 0.00002, "n2o_factor": 0.00019},
            {"activity": "water_treatment", "unit": "m3", "co2_factor": 0.27200, "ch4_factor": 0.00084, "n2o_factor": 0.02670},
        ],
        "scope3_waste": [
            {"activity": "landfill_mixed_msw", "unit": "tonnes", "co2_factor": 467.05, "ch4_factor": 16.41, "n2o_factor": 0.034},
            {"activity": "incineration_mixed_msw", "unit": "tonnes", "co2_factor": 21.354, "ch4_factor": 0.001, "n2o_factor": 0.857},
        ],
        "scope3_materials": [
            {"activity": "paper", "unit": "kg", "co2_factor": 0.91930, "ch4_factor": 0.00015, "n2o_factor": 0.00012},
        ],
    }


def test_parse_full_payload(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    # 8 s1 + 2 bio + 3 s2e + 2 s2h + 2 wtt + 2 freight + 4 travel + 2 water + 2 waste + 1 material = 28
    assert len(factors) == 28


def test_all_factors_pass_qa(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_scope1_fuels_parsed(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    s1 = [f for f in factors if f["scope"] == "1" and "bioenergy" not in f["tags"]]
    assert len(s1) == 8
    for f in s1:
        assert f["geography"] == "UK"
        assert f["boundary"] == "combustion"


def test_bioenergy_tagged(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    bio = [f for f in factors if "bioenergy" in f["tags"]]
    assert len(bio) == 2


def test_scope2_electricity_types(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    s2e = [f for f in factors if "electricity" in f["tags"] and f["scope"] == "2"]
    assert len(s2e) == 3
    types = {f["fuel_type"] for f in s2e}
    assert "electricity_grid_average" in types
    assert "electricity_generation" in types


def test_scope3_sections(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    s3 = [f for f in factors if f["scope"] == "3"]
    assert len(s3) == 13  # 2+2+4+2+2+1


def test_gwp_set_is_ar5(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    for f in factors:
        assert f["gwp_100yr"]["gwp_set"] == "IPCC_AR5_100"
        assert f["gwp_100yr"]["N2O_gwp"] == 265  # AR5 value


def test_license_is_ogl_uk(desnz_full_payload):
    factors = parse_desnz_uk(desnz_full_payload)
    for f in factors:
        assert f["license_info"]["license"] == "OGL-UK-v3"
        assert f["license_info"]["redistribution_allowed"] is True


def test_empty_payload():
    assert parse_desnz_uk({}) == []


def test_partial_payload():
    data = {
        "metadata": {"version": "2024", "region": "UK"},
        "scope1_fuels": [
            {"fuel_type": "Natural Gas", "unit": "kwh", "co2_factor": 0.183, "ch4_factor": 0.0003, "n2o_factor": 0.0003},
        ],
    }
    factors = parse_desnz_uk(data)
    assert len(factors) == 1
    assert factors[0]["scope"] == "1"
