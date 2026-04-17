# -*- coding: utf-8 -*-
"""Tests for TCR default factors parser (F016)."""

from __future__ import annotations

import pytest

from greenlang.factors.ingestion.parsers.tcr import parse_tcr
from greenlang.factors.etl.qa import validate_factor_dict


@pytest.fixture
def tcr_payload():
    return {
        "metadata": {"source": "TCR", "version": "2024"},
        "stationary_combustion": [
            {"fuel_type": "natural_gas", "unit": "mmbtu", "co2_factor": 53.06, "ch4_factor": 0.001, "n2o_factor": 0.0001},
            {"fuel_type": "diesel", "unit": "gallons", "co2_factor": 10.21, "ch4_factor": 0.00041, "n2o_factor": 0.000082},
            {"fuel_type": "propane", "unit": "gallons", "co2_factor": 5.72, "ch4_factor": 0.00023, "n2o_factor": 0.000046},
        ],
        "mobile_combustion": [
            {"fuel_type": "gasoline_passenger_car", "unit": "gallons", "co2_factor": 8.78, "ch4_factor": 0.00025, "n2o_factor": 0.00022},
            {"fuel_type": "diesel_heavy_duty", "unit": "gallons", "co2_factor": 10.21, "ch4_factor": 0.00005, "n2o_factor": 0.00012},
        ],
        "electricity": [
            {"type": "us_average", "unit": "kwh", "co2_factor": 0.000386, "ch4_factor": 0.0000000186, "n2o_factor": 0.00000000369},
        ],
        "refrigerants": [
            {"type": "r_410a", "unit": "kg", "co2_factor": 2088.0, "ch4_factor": 0.0, "n2o_factor": 0.0},
            {"type": "r_134a", "unit": "kg", "co2_factor": 1430.0, "ch4_factor": 0.0, "n2o_factor": 0.0},
        ],
    }


def test_parse_full_payload(tcr_payload):
    factors = parse_tcr(tcr_payload)
    # 3 + 2 + 1 + 2 = 8
    assert len(factors) == 8


def test_all_factors_pass_qa(tcr_payload):
    factors = parse_tcr(tcr_payload)
    for f in factors:
        ok, errs = validate_factor_dict(f)
        assert ok, f"QA failed for {f['factor_id']}: {errs}"


def test_factor_ids_unique(tcr_payload):
    factors = parse_tcr(tcr_payload)
    ids = [f["factor_id"] for f in factors]
    assert len(ids) == len(set(ids))


def test_tcr_tags(tcr_payload):
    factors = parse_tcr(tcr_payload)
    for f in factors:
        assert "tcr" in f["tags"]
        assert "grp" in f["tags"]


def test_gwp_ar5(tcr_payload):
    factors = parse_tcr(tcr_payload)
    for f in factors:
        assert f["gwp_100yr"]["gwp_set"] == "IPCC_AR5_100"


def test_scope_assignment(tcr_payload):
    factors = parse_tcr(tcr_payload)
    stat = [f for f in factors if "stationary" in f["tags"]]
    elec = [f for f in factors if "electricity" in f["tags"]]
    for f in stat:
        assert f["scope"] == "1"
    for f in elec:
        assert f["scope"] == "2"


def test_empty_payload():
    assert parse_tcr({}) == []
