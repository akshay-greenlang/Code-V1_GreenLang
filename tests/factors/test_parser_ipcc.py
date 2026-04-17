# -*- coding: utf-8 -*-
"""Tests for IPCC Tier 1 defaults parser (F013)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.ipcc_defaults import parse_ipcc_defaults
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def ipcc_payload():
    return {
        "metadata": {"source": "IPCC", "version": "2019_refinement"},
        "energy_stationary": [
            {"category": "crude_oil", "unit": "gj", "co2_factor": 73.3, "ch4_factor": 0.003, "n2o_factor": 0.0006},
            {"category": "natural_gas", "unit": "gj", "co2_factor": 56.1, "ch4_factor": 0.001, "n2o_factor": 0.0001},
            {"category": "anthracite", "unit": "gj", "co2_factor": 98.3, "ch4_factor": 0.001, "n2o_factor": 0.0015},
            {"category": "coking_coal", "unit": "gj", "co2_factor": 94.6, "ch4_factor": 0.001, "n2o_factor": 0.0015},
            {"category": "lignite", "unit": "gj", "co2_factor": 101.0, "ch4_factor": 0.001, "n2o_factor": 0.0015},
            {"category": "peat", "unit": "gj", "co2_factor": 106.0, "ch4_factor": 0.001, "n2o_factor": 0.0015},
        ],
        "energy_mobile": [
            {"category": "gasoline", "unit": "gj", "co2_factor": 69.3, "ch4_factor": 0.033, "n2o_factor": 0.0032},
            {"category": "diesel_oil", "unit": "gj", "co2_factor": 74.1, "ch4_factor": 0.0039, "n2o_factor": 0.0039},
            {"category": "jet_kerosene", "unit": "gj", "co2_factor": 71.5, "ch4_factor": 0.0005, "n2o_factor": 0.002},
        ],
        "industrial_processes": [
            {"category": "cement_clinker", "unit": "tonnes", "co2_factor": 520.0, "ch4_factor": 0.0, "n2o_factor": 0.0},
            {"category": "lime_production", "unit": "tonnes", "co2_factor": 750.0, "ch4_factor": 0.0, "n2o_factor": 0.0},
            {"category": "steel_bof", "unit": "tonnes", "co2_factor": 1460.0, "ch4_factor": 0.001, "n2o_factor": 0.0},
            {"category": "aluminum_primary", "unit": "tonnes", "co2_factor": 1500.0, "ch4_factor": 0.0, "n2o_factor": 0.0},
        ],
        "agriculture": [
            {"category": "enteric_fermentation_cattle", "unit": "kg", "co2_factor": 0.0, "ch4_factor": 0.068, "n2o_factor": 0.0},
            {"category": "rice_cultivation", "unit": "kg", "co2_factor": 0.0, "ch4_factor": 0.0012, "n2o_factor": 0.0},
            {"category": "manure_management", "unit": "kg", "co2_factor": 0.0, "ch4_factor": 0.0021, "n2o_factor": 0.0005},
        ],
        "lulucf": [
            {"category": "forest_land_to_cropland", "unit": "kg", "co2_factor": 0.45, "ch4_factor": 0.0, "n2o_factor": 0.0},
        ],
        "waste": [
            {"category": "solid_waste_disposal", "unit": "tonnes", "co2_factor": 0.0, "ch4_factor": 58.0, "n2o_factor": 0.0},
            {"category": "incineration", "unit": "tonnes", "co2_factor": 364.0, "ch4_factor": 0.02, "n2o_factor": 0.05},
            {"category": "wastewater_domestic", "unit": "tonnes", "co2_factor": 0.0, "ch4_factor": 0.48, "n2o_factor": 0.026},
        ],
    }


def test_parse_full_payload(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    # 6 + 3 + 4 + 3 + 1 + 3 = 20
    assert len(factors) == 20


def test_all_factors_pass_qa(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_dqs_is_global_defaults(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    for f in factors:
        assert f["dqs"]["temporal"] == 3
        assert f["dqs"]["geographical"] == 2
        assert f["dqs"]["methodological"] == 5


def test_geography_is_global(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    for f in factors:
        assert f["geography"] == "GLOBAL"
        assert f["geography_level"] == "global"


def test_tags_include_ipcc_tier1(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    for f in factors:
        assert "ipcc" in f["tags"]
        assert "tier1" in f["tags"]


def test_industrial_processes_scope1(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    ind = [f for f in factors if "industrial" in f["tags"]]
    assert len(ind) == 4
    for f in ind:
        assert f["scope"] == "1"


def test_waste_is_scope3(ipcc_payload):
    factors = parse_ipcc_defaults(ipcc_payload)
    waste = [f for f in factors if "waste" in f["tags"] and f["factor_id"].startswith("EF:IPCC:waste")]
    assert len(waste) == 3
    for f in waste:
        assert f["scope"] == "3"


def test_empty_payload():
    assert parse_ipcc_defaults({}) == []


def test_version_year_extraction():
    data = {
        "metadata": {"version": "2019_refinement"},
        "energy_stationary": [
            {"category": "crude_oil", "unit": "gj", "co2_factor": 73.3, "ch4_factor": 0.003, "n2o_factor": 0.0006},
        ],
    }
    factors = parse_ipcc_defaults(data)
    assert "2019" in factors[0]["factor_id"]
