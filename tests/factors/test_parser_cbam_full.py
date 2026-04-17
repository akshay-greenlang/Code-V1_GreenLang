# -*- coding: utf-8 -*-
"""Tests for CBAM full coverage parser (F014)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.cbam_full import parse_cbam_full
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def cbam_full_payload():
    return {
        "metadata": {"source": "EU Commission", "version": "2024"},
        "products": {
            "iron_steel": {
                "categories": [
                    {"name": "hot_rolled_coil"},
                    {"name": "cold_rolled_sheet"},
                    {"name": "stainless_steel"},
                ],
                "by_country": {
                    "CN": {"direct_emissions_factor": 2.15, "indirect_emissions_factor": 0.85},
                    "IN": {"direct_emissions_factor": 2.45, "indirect_emissions_factor": 0.92},
                    "RU": {"direct_emissions_factor": 1.98, "indirect_emissions_factor": 0.55},
                },
            },
            "aluminum": {
                "categories": [{"name": "primary"}, {"name": "secondary"}],
                "by_country": {
                    "CN": {"direct_emissions_factor": 8.50, "indirect_emissions_factor": 7.20},
                    "RU": {"direct_emissions_factor": 2.10, "indirect_emissions_factor": 0.55},
                },
            },
            "cement": {
                "categories": [{"name": "clinker"}, {"name": "portland"}],
                "by_country": {
                    "CN": {"direct_emissions_factor": 0.82, "indirect_emissions_factor": 0.05},
                    "TR": {"direct_emissions_factor": 0.78, "indirect_emissions_factor": 0.06},
                    "IN": {"direct_emissions_factor": 0.85, "indirect_emissions_factor": 0.08},
                },
            },
            "fertilizers": {
                "categories": [{"name": "urea"}, {"name": "ammonia"}, {"name": "nitric_acid"}],
                "by_country": {
                    "CN": {"direct_emissions_factor": 1.65, "indirect_emissions_factor": 0.35},
                    "RU": {"direct_emissions_factor": 1.25, "indirect_emissions_factor": 0.20},
                },
            },
            "electricity": {
                "categories": [{"name": "grid_import"}],
                "by_country": {
                    "BA": {"direct_emissions_factor": 0.85, "indirect_emissions_factor": 0.0},
                    "RS": {"direct_emissions_factor": 0.78, "indirect_emissions_factor": 0.0},
                    "UA": {"direct_emissions_factor": 0.42, "indirect_emissions_factor": 0.0},
                },
            },
            "hydrogen": {
                "categories": [{"name": "grey"}, {"name": "blue"}],
                "by_country": {
                    "CN": {"direct_emissions_factor": 9.00, "indirect_emissions_factor": 2.00},
                },
            },
        },
    }


def test_parse_full_payload(cbam_full_payload):
    factors = parse_cbam_full(cbam_full_payload)
    # iron: 3 cats x 3 countries = 9
    # aluminum: 2 x 2 = 4
    # cement: 2 x 3 = 6
    # fertilizers: 3 x 2 = 6
    # electricity: 1 x 3 = 3
    # hydrogen: 2 x 1 = 2
    assert len(factors) == 30


def test_all_factors_pass_qa(cbam_full_payload):
    factors = parse_cbam_full(cbam_full_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(cbam_full_payload):
    factors = parse_cbam_full(cbam_full_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_all_scope3_cradle_to_gate(cbam_full_payload):
    factors = parse_cbam_full(cbam_full_payload)
    for f in factors:
        assert f["scope"] == "3"
        assert f["boundary"] == "cradle_to_gate"


def test_cbam_tags(cbam_full_payload):
    factors = parse_cbam_full(cbam_full_payload)
    for f in factors:
        assert "cbam" in f["tags"]
        assert "cbam_2026" in f["tags"]


def test_direct_indirect_summed(cbam_full_payload):
    factors = parse_cbam_full(cbam_full_payload)
    # Iron/Steel CN hot_rolled_coil: direct=2.15 + indirect=0.85 = 3.0
    iron_cn = [f for f in factors if "iron_steel" in f["factor_id"] and ":CN:" in f["factor_id"] and "hot_rolled" in f["factor_id"]]
    assert len(iron_cn) == 1
    assert abs(iron_cn[0]["vectors"]["CO2"] - 3.0) < 0.01


def test_empty_payload():
    assert parse_cbam_full({}) == []


def test_single_product():
    data = {
        "metadata": {"version": "2024"},
        "products": {
            "cement": {
                "categories": [{"name": "clinker"}],
                "by_country": {"CN": {"direct_emissions_factor": 0.82, "indirect_emissions_factor": 0.05}},
            },
        },
    }
    factors = parse_cbam_full(data)
    assert len(factors) == 1
    assert "cement" in factors[0]["tags"]
