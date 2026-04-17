# -*- coding: utf-8 -*-
"""Tests for GHG Protocol factors parser (F015)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.ghg_protocol import parse_ghg_protocol
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def ghgp_payload():
    return {
        "metadata": {"source": "GHG Protocol", "version": "2024"},
        "cat1_purchased_goods": [
            {"sector": "agriculture", "unit": "usd", "co2e": 0.52, "co2": 0.52, "ch4": 0.0, "n2o": 0.0},
            {"sector": "chemicals", "unit": "usd", "co2e": 0.78, "co2": 0.78, "ch4": 0.0, "n2o": 0.0},
            {"sector": "food_products", "unit": "usd", "co2e": 0.65, "co2": 0.65, "ch4": 0.0, "n2o": 0.0},
            {"sector": "machinery", "unit": "usd", "co2e": 0.35, "co2": 0.35, "ch4": 0.0, "n2o": 0.0},
            {"sector": "construction", "unit": "usd", "co2e": 0.42, "co2": 0.42, "ch4": 0.0, "n2o": 0.0},
        ],
        "cat4_upstream_transport": [
            {"activity": "road_freight", "unit": "tonne_km", "co2": 0.10432, "ch4": 0.00001, "n2o": 0.00154},
            {"activity": "rail_freight", "unit": "tonne_km", "co2": 0.02724, "ch4": 0.00001, "n2o": 0.00041},
            {"activity": "sea_freight", "unit": "tonne_km", "co2": 0.01622, "ch4": 0.00001, "n2o": 0.00024},
            {"activity": "air_freight", "unit": "tonne_km", "co2": 0.60234, "ch4": 0.00002, "n2o": 0.01803},
        ],
        "cat6_business_travel": [
            {"activity": "short_haul_flight", "unit": "km", "co2": 0.15353, "ch4": 0.00001, "n2o": 0.00147},
            {"activity": "long_haul_flight", "unit": "km", "co2": 0.19309, "ch4": 0.00001, "n2o": 0.00185},
            {"activity": "rail", "unit": "km", "co2": 0.03549, "ch4": 0.00001, "n2o": 0.00050},
            {"activity": "hotel_night", "unit": "km", "co2": 10.20, "ch4": 0.001, "n2o": 0.003},
        ],
        "cat7_commuting": [
            {"activity": "car_average", "unit": "km", "co2": 0.17120, "ch4": 0.00002, "n2o": 0.00250},
            {"activity": "bus", "unit": "km", "co2": 0.08950, "ch4": 0.00001, "n2o": 0.00135},
            {"activity": "bicycle", "unit": "km", "co2": 0.0, "ch4": 0.0, "n2o": 0.0},
        ],
    }


def test_parse_full_payload(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    # 5 cat1 + 4 cat4 + 4 cat6 + 3 cat7 = 16
    assert len(factors) == 16


def test_all_factors_pass_qa(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_all_scope3(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    for f in factors:
        assert f["scope"] == "3"


def test_cat_tags(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    cat1 = [f for f in factors if "cat1" in f["tags"]]
    cat4 = [f for f in factors if "cat4" in f["tags"]]
    cat6 = [f for f in factors if "cat6" in f["tags"]]
    cat7 = [f for f in factors if "cat7" in f["tags"]]
    assert len(cat1) == 5
    assert len(cat4) == 4
    assert len(cat6) == 4
    assert len(cat7) == 3


def test_cat1_spend_based_methodology(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    cat1 = [f for f in factors if "cat1" in f["tags"]]
    for f in cat1:
        assert f["provenance"]["methodology"] == "spend_based"


def test_license_not_redistributable(ghgp_payload):
    factors = parse_ghg_protocol(ghgp_payload)
    for f in factors:
        assert f["license_info"]["redistribution_allowed"] is False


def test_empty_payload():
    assert parse_ghg_protocol({}) == []
