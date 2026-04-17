# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.quality.validators and etl.qa."""

from __future__ import annotations

import copy

import pytest

from greenlang.factors.etl.qa import validate_factor_dict
from greenlang.factors.quality.validators import validate_canonical_row


def _make_valid_factor_dict():
    return {
        "factor_id": "EF:US:diesel:2024:v1",
        "fuel_type": "diesel",
        "unit": "litres",
        "geography": "US",
        "vectors": {"CO2": 2.68, "CH4": 0.0001, "N2O": 0.00001},
        "gwp_100yr": {"co2e_total": 2.7, "gwp_set": "IPCC_AR6_100"},
        "valid_from": "2024-01-01",
        "factor_status": "certified",
    }


# --- validate_factor_dict tests ---


def test_validate_factor_dict_valid():
    ok, errors = validate_factor_dict(_make_valid_factor_dict())
    assert ok is True
    assert errors == []


def test_missing_factor_id_fails():
    d = _make_valid_factor_dict()
    del d["factor_id"]
    ok, errors = validate_factor_dict(d)
    assert ok is False
    assert any("factor_id" in e for e in errors)


def test_factor_id_prefix_required():
    d = _make_valid_factor_dict()
    d["factor_id"] = "WRONG:US:diesel:2024:v1"
    ok, errors = validate_factor_dict(d)
    assert ok is False
    assert any("EF:" in e for e in errors)


def test_missing_co2_vector_fails():
    d = _make_valid_factor_dict()
    del d["vectors"]["CO2"]
    ok, errors = validate_factor_dict(d)
    assert ok is False
    assert any("CO2" in e for e in errors)


def test_negative_vector_fails():
    d = _make_valid_factor_dict()
    d["vectors"]["CH4"] = -0.5
    ok, errors = validate_factor_dict(d)
    assert ok is False
    assert any("CH4" in e for e in errors)


def test_non_numeric_vector_fails():
    d = _make_valid_factor_dict()
    d["vectors"]["N2O"] = "abc"
    ok, errors = validate_factor_dict(d)
    assert ok is False
    assert any("N2O" in e for e in errors)


def test_negative_co2e_total_fails():
    d = _make_valid_factor_dict()
    d["gwp_100yr"]["co2e_total"] = -100.0
    ok, errors = validate_factor_dict(d)
    assert ok is False
    assert any("co2e_total" in e for e in errors)


# --- validate_canonical_row tests ---


def test_canonical_row_valid():
    ok, errors = validate_canonical_row(_make_valid_factor_dict())
    assert ok is True
    assert errors == []


def test_canonical_outlier_co2e():
    d = _make_valid_factor_dict()
    d["gwp_100yr"]["co2e_total"] = 2e8
    ok, errors = validate_canonical_row(d)
    assert ok is False
    assert any("outlier" in e for e in errors)


def test_canonical_invalid_year():
    d = _make_valid_factor_dict()
    d["valid_from"] = "1980-01-01"
    ok, errors = validate_canonical_row(d)
    assert ok is False
    assert any("year" in e.lower() or "plausible" in e.lower() for e in errors)


def test_canonical_invalid_geo():
    d = _make_valid_factor_dict()
    d["geography"] = "1Z"  # 2-char but not matching ^[A-Z]{2}$
    ok, errors = validate_canonical_row(d)
    assert ok is False
    assert any("ISO2" in e or "geography" in e for e in errors)


def test_canonical_unknown_unit():
    d = _make_valid_factor_dict()
    d["unit"] = "unknown_unit_xyz"
    ok, errors = validate_canonical_row(d)
    assert ok is False
    assert any("unit" in e for e in errors)
