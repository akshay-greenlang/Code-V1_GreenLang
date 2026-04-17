# -*- coding: utf-8 -*-
"""Tests for Green-e residual mix parser (F017)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.green_e import parse_green_e
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def green_e_payload():
    return {
        "metadata": {"source": "Green-e", "version": "2023"},
        "residual_mix": [
            {"region": "US-NEPOOL", "state": "MA", "co2_lb_mwh": 680.5, "ch4_lb_mwh": 0.052, "n2o_lb_mwh": 0.008},
            {"region": "US-NYCW", "state": "NY", "co2_lb_mwh": 520.3, "ch4_lb_mwh": 0.035, "n2o_lb_mwh": 0.005},
            {"region": "US-CAMX", "state": "CA", "co2_lb_mwh": 410.2, "ch4_lb_mwh": 0.028, "n2o_lb_mwh": 0.004},
            {"region": "US-ERCT", "state": "TX", "co2_lb_mwh": 835.1, "ch4_lb_mwh": 0.053, "n2o_lb_mwh": 0.009},
            {"region": "US-RFCW", "state": "OH", "co2_lb_mwh": 1105.0, "ch4_lb_mwh": 0.091, "n2o_lb_mwh": 0.016},
        ],
    }


def test_parse_full_payload(green_e_payload):
    factors = parse_green_e(green_e_payload)
    assert len(factors) == 5


def test_all_factors_pass_qa(green_e_payload):
    factors = parse_green_e(green_e_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(green_e_payload):
    factors = parse_green_e(green_e_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_all_scope2_market_based(green_e_payload):
    factors = parse_green_e(green_e_payload)
    for f in factors:
        assert f["scope"] == "2"
        assert "market_based" in f["tags"]
        assert f["fuel_type"] == "electricity_residual_mix"
        assert f["unit"] == "kwh"


def test_lb_mwh_converted_to_kg_kwh(green_e_payload):
    factors = parse_green_e(green_e_payload)
    for f in factors:
        co2 = f["vectors"]["CO2"]
        # kg CO2/kWh should be between 0 and 1
        assert 0 <= co2 <= 1.0, f"CO2={co2} for {f['factor_id']}"


def test_tags_include_green_e(green_e_payload):
    factors = parse_green_e(green_e_payload)
    for f in factors:
        assert "green_e" in f["tags"]
        assert "residual_mix" in f["tags"]


def test_license_not_redistributable(green_e_payload):
    factors = parse_green_e(green_e_payload)
    for f in factors:
        assert f["license_info"]["redistribution_allowed"] is False


def test_empty_payload():
    assert parse_green_e({}) == []


def test_already_in_kg_kwh():
    """If values are already small (< 1), don't re-convert."""
    data = {
        "metadata": {"version": "2023"},
        "residual_mix": [
            {"region": "US-TEST", "co2": 0.35, "ch4": 0.00002, "n2o": 0.000003},
        ],
    }
    factors = parse_green_e(data)
    assert len(factors) == 1
    assert abs(factors[0]["vectors"]["CO2"] - 0.35) < 0.01
