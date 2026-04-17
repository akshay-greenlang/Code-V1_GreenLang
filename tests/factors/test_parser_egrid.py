# -*- coding: utf-8 -*-
"""Tests for eGRID electricity grid parser (F011)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.egrid import (
    parse_egrid,
    iter_subregions,
    iter_states,
    parse_national,
    _convert_lb_mwh_to_kg_kwh,
)
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def egrid_metadata():
    return {
        "source": "eGRID",
        "version": "2022",
        "source_publication": "eGRID2022",
        "source_url": "https://www.epa.gov/egrid",
    }


@pytest.fixture
def egrid_full_payload(egrid_metadata):
    return {
        "metadata": egrid_metadata,
        "subregions": [
            {"acronym": "AKGD", "name": "ASCC Alaska Grid", "co2_lb_mwh": 1050.2, "ch4_lb_mwh": 0.121, "n2o_lb_mwh": 0.015},
            {"acronym": "AKMS", "name": "ASCC Miscellaneous", "co2_lb_mwh": 495.3, "ch4_lb_mwh": 0.057, "n2o_lb_mwh": 0.008},
            {"acronym": "CAMX", "name": "WECC California", "co2_lb_mwh": 496.0, "ch4_lb_mwh": 0.038, "n2o_lb_mwh": 0.005},
            {"acronym": "ERCT", "name": "ERCOT All", "co2_lb_mwh": 835.5, "ch4_lb_mwh": 0.053, "n2o_lb_mwh": 0.009},
            {"acronym": "FRCC", "name": "FRCC All", "co2_lb_mwh": 883.2, "ch4_lb_mwh": 0.073, "n2o_lb_mwh": 0.009},
            {"acronym": "MROE", "name": "MRO East", "co2_lb_mwh": 1203.0, "ch4_lb_mwh": 0.099, "n2o_lb_mwh": 0.017},
            {"acronym": "MROW", "name": "MRO West", "co2_lb_mwh": 1035.6, "ch4_lb_mwh": 0.097, "n2o_lb_mwh": 0.016},
            {"acronym": "NEWE", "name": "NPCC New England", "co2_lb_mwh": 447.3, "ch4_lb_mwh": 0.059, "n2o_lb_mwh": 0.006},
            {"acronym": "NWPP", "name": "WECC Northwest", "co2_lb_mwh": 619.8, "ch4_lb_mwh": 0.045, "n2o_lb_mwh": 0.008},
            {"acronym": "NYCW", "name": "NPCC NYC/Westchester", "co2_lb_mwh": 568.1, "ch4_lb_mwh": 0.035, "n2o_lb_mwh": 0.004},
            {"acronym": "NYLI", "name": "NPCC Long Island", "co2_lb_mwh": 1062.5, "ch4_lb_mwh": 0.048, "n2o_lb_mwh": 0.007},
            {"acronym": "NYUP", "name": "NPCC Upstate NY", "co2_lb_mwh": 249.2, "ch4_lb_mwh": 0.014, "n2o_lb_mwh": 0.003},
            {"acronym": "RFCE", "name": "RFC East", "co2_lb_mwh": 660.5, "ch4_lb_mwh": 0.049, "n2o_lb_mwh": 0.009},
            {"acronym": "RFCM", "name": "RFC Michigan", "co2_lb_mwh": 1143.2, "ch4_lb_mwh": 0.085, "n2o_lb_mwh": 0.016},
            {"acronym": "RFCW", "name": "RFC West", "co2_lb_mwh": 1104.8, "ch4_lb_mwh": 0.091, "n2o_lb_mwh": 0.016},
            {"acronym": "RMPA", "name": "WECC Rockies", "co2_lb_mwh": 1289.4, "ch4_lb_mwh": 0.104, "n2o_lb_mwh": 0.019},
            {"acronym": "SPNO", "name": "SPP North", "co2_lb_mwh": 1273.2, "ch4_lb_mwh": 0.110, "n2o_lb_mwh": 0.019},
            {"acronym": "SPSO", "name": "SPP South", "co2_lb_mwh": 977.8, "ch4_lb_mwh": 0.076, "n2o_lb_mwh": 0.013},
            {"acronym": "SRMV", "name": "SERC Mississippi Valley", "co2_lb_mwh": 753.4, "ch4_lb_mwh": 0.058, "n2o_lb_mwh": 0.008},
            {"acronym": "SRMW", "name": "SERC Midwest", "co2_lb_mwh": 1466.2, "ch4_lb_mwh": 0.137, "n2o_lb_mwh": 0.022},
            {"acronym": "SRSO", "name": "SERC South", "co2_lb_mwh": 893.2, "ch4_lb_mwh": 0.064, "n2o_lb_mwh": 0.010},
            {"acronym": "SRTV", "name": "SERC Tennessee Valley", "co2_lb_mwh": 846.7, "ch4_lb_mwh": 0.072, "n2o_lb_mwh": 0.012},
            {"acronym": "SRVC", "name": "SERC Virginia/Carolina", "co2_lb_mwh": 618.4, "ch4_lb_mwh": 0.046, "n2o_lb_mwh": 0.009},
            {"acronym": "AZNM", "name": "WECC Southwest", "co2_lb_mwh": 864.5, "ch4_lb_mwh": 0.058, "n2o_lb_mwh": 0.010},
            {"acronym": "HIOA", "name": "HICC Oahu", "co2_lb_mwh": 1507.8, "ch4_lb_mwh": 0.154, "n2o_lb_mwh": 0.022},
            {"acronym": "HIMS", "name": "HICC Miscellaneous", "co2_lb_mwh": 1345.2, "ch4_lb_mwh": 0.135, "n2o_lb_mwh": 0.019},
        ],
        "states": [
            {"state": "AL", "co2_lb_mwh": 820.5, "ch4_lb_mwh": 0.067, "n2o_lb_mwh": 0.010},
            {"state": "AK", "co2_lb_mwh": 850.3, "ch4_lb_mwh": 0.098, "n2o_lb_mwh": 0.012},
            {"state": "AZ", "co2_lb_mwh": 780.2, "ch4_lb_mwh": 0.052, "n2o_lb_mwh": 0.009},
            {"state": "CA", "co2_lb_mwh": 430.1, "ch4_lb_mwh": 0.033, "n2o_lb_mwh": 0.005},
            {"state": "CO", "co2_lb_mwh": 1105.0, "ch4_lb_mwh": 0.089, "n2o_lb_mwh": 0.016},
            {"state": "CT", "co2_lb_mwh": 380.5, "ch4_lb_mwh": 0.050, "n2o_lb_mwh": 0.005},
            {"state": "FL", "co2_lb_mwh": 883.2, "ch4_lb_mwh": 0.073, "n2o_lb_mwh": 0.009},
            {"state": "GA", "co2_lb_mwh": 815.0, "ch4_lb_mwh": 0.060, "n2o_lb_mwh": 0.009},
            {"state": "HI", "co2_lb_mwh": 1450.0, "ch4_lb_mwh": 0.148, "n2o_lb_mwh": 0.021},
            {"state": "IL", "co2_lb_mwh": 680.3, "ch4_lb_mwh": 0.056, "n2o_lb_mwh": 0.010},
            {"state": "NY", "co2_lb_mwh": 410.2, "ch4_lb_mwh": 0.029, "n2o_lb_mwh": 0.004},
            {"state": "TX", "co2_lb_mwh": 835.5, "ch4_lb_mwh": 0.053, "n2o_lb_mwh": 0.009},
            {"state": "WA", "co2_lb_mwh": 165.0, "ch4_lb_mwh": 0.011, "n2o_lb_mwh": 0.002},
        ],
        "national": {
            "co2_lb_mwh": 852.3,
            "ch4_lb_mwh": 0.067,
            "n2o_lb_mwh": 0.010,
        },
    }


# ---- Unit conversion ----

def test_lb_mwh_to_kg_kwh_conversion():
    # 1000 lb/MWh = 1000 * 0.453592 / 1000 = 0.453592 kg/kWh
    result = _convert_lb_mwh_to_kg_kwh(1000.0)
    assert abs(result - 0.453592) < 1e-6


def test_lb_mwh_to_kg_kwh_zero():
    assert _convert_lb_mwh_to_kg_kwh(0.0) == 0.0


# ---- Subregion parser ----

def test_subregion_parser_produces_valid_factors(egrid_metadata):
    rows = [
        {"acronym": "CAMX", "name": "WECC California", "co2_lb_mwh": 496.0, "ch4_lb_mwh": 0.038, "n2o_lb_mwh": 0.005},
    ]
    results = list(iter_subregions(rows, egrid_metadata, 2022))
    assert len(results) == 1
    f = results[0]
    assert f["factor_id"] == "EF:eGRID:sub_CAMX:CAMX:2022:v1"
    assert f["geography"] == "CAMX"
    assert f["geography_level"] == "grid_zone"
    assert f["scope"] == "2"
    assert f["unit"] == "kwh"
    ok, errs = validate_factor_dict(f)
    assert ok, errs


def test_subregion_skips_empty_acronym(egrid_metadata):
    rows = [{"acronym": "", "co2_lb_mwh": 100}]
    results = list(iter_subregions(rows, egrid_metadata, 2022))
    assert len(results) == 0


# ---- State parser ----

def test_state_parser_produces_valid_factors(egrid_metadata):
    rows = [
        {"state": "CA", "co2_lb_mwh": 430.1, "ch4_lb_mwh": 0.033, "n2o_lb_mwh": 0.005},
    ]
    results = list(iter_states(rows, egrid_metadata, 2022))
    assert len(results) == 1
    f = results[0]
    assert f["factor_id"] == "EF:eGRID:state_CA:US-CA:2022:v1"
    assert f["geography"] == "US-CA"
    assert f["geography_level"] == "state"
    ok, errs = validate_factor_dict(f)
    assert ok, errs


def test_state_skips_invalid_code(egrid_metadata):
    rows = [{"state": "INVALID", "co2_lb_mwh": 100}]
    results = list(iter_states(rows, egrid_metadata, 2022))
    assert len(results) == 0


# ---- National parser ----

def test_national_parser(egrid_metadata):
    data = {"co2_lb_mwh": 852.3, "ch4_lb_mwh": 0.067, "n2o_lb_mwh": 0.010}
    f = parse_national(data, egrid_metadata, 2022)
    assert f is not None
    assert f["factor_id"] == "EF:eGRID:national:US:2022:v1"
    assert f["geography"] == "US"
    assert f["geography_level"] == "country"
    ok, errs = validate_factor_dict(f)
    assert ok, errs


def test_national_parser_none():
    assert parse_national(None, {}, 2022) is None
    assert parse_national({}, {}, 2022) is None


# ---- Full parser ----

def test_parse_full_payload(egrid_full_payload):
    factors = parse_egrid(egrid_full_payload)
    # 26 subregions + 13 states + 1 national = 40
    assert len(factors) == 40


def test_all_factors_pass_qa(egrid_full_payload):
    factors = parse_egrid(egrid_full_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(egrid_full_payload):
    factors = parse_egrid(egrid_full_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_all_factors_are_scope2_kwh(egrid_full_payload):
    factors = parse_egrid(egrid_full_payload)
    for f in factors:
        assert f["scope"] == "2"
        assert f["unit"] == "kwh"
        assert f["fuel_type"] == "electricity_grid"


def test_co2_values_positive_and_reasonable(egrid_full_payload):
    factors = parse_egrid(egrid_full_payload)
    for f in factors:
        co2 = f["vectors"]["CO2"]
        # kg CO2/kWh should be between 0 and 1 for grid electricity
        assert 0 <= co2 <= 1.0, f"CO2={co2} for {f['factor_id']}"


def test_empty_payload():
    assert parse_egrid({}) == []


def test_version_extraction():
    data = {
        "metadata": {"version": "2023"},
        "subregions": [
            {"acronym": "CAMX", "co2_lb_mwh": 496.0, "ch4_lb_mwh": 0.038, "n2o_lb_mwh": 0.005},
        ],
    }
    factors = parse_egrid(data)
    assert factors[0]["factor_id"].endswith(":2023:v1")
