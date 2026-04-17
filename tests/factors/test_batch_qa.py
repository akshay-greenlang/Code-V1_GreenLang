# -*- coding: utf-8 -*-
"""Tests for batch QA runner (F020)."""

from __future__ import annotations

import pytest

from greenlang.factors.quality.batch_qa import (
    BatchQAReport,
    FactorQAResult,
    run_batch_qa_on_dicts,
    _run_qa_on_factor,
)


def _good_factor(fid: str = "EF:EPA:stat_natural_gas:US:2024:v1", **overrides):
    base = {
        "factor_id": fid,
        "fuel_type": "natural_gas",
        "unit": "scf",
        "geography": "US",
        "vectors": {"CO2": 0.05444, "CH4": 0.000001, "N2O": 0.0000001},
        "gwp_100yr": {"gwp_set": "IPCC_AR6_100", "CH4_gwp": 28, "N2O_gwp": 273},
        "scope": "1",
        "boundary": "cradle_to_gate",
        "provenance": {
            "source_org": "EPA",
            "methodology": "IPCC_Tier_1",
            "source_year": 2024,
        },
        "valid_from": "2024-01-01",
        "dqs": {
            "temporal": 5,
            "geographical": 4,
            "technological": 4,
            "representativeness": 4,
            "methodological": 5,
        },
        "license_info": {
            "license": "US-Public-Domain",
            "redistribution_allowed": True,
            "commercial_use_allowed": True,
            "attribution_required": False,
        },
        "tags": ["epa", "natural_gas"],
        "factor_status": "preview",
    }
    base.update(overrides)
    return base


# ---- FactorQAResult ----

def test_factor_qa_result_to_dict():
    r = FactorQAResult(factor_id="EF:test:1", passed=True, promoted=True, new_status="certified")
    d = r.to_dict()
    assert d["factor_id"] == "EF:test:1"
    assert d["passed"] is True
    assert d["promoted"] is True


# ---- BatchQAReport ----

def test_batch_qa_report_to_dict():
    report = BatchQAReport(edition_id="2024.04.0", total_factors=10, total_passed=8, total_failed=2)
    d = report.to_dict()
    assert d["edition_id"] == "2024.04.0"
    assert d["pass_rate"] == 0.8
    assert d["total_factors"] == 10


def test_batch_qa_report_to_json():
    report = BatchQAReport(edition_id="test", total_factors=1, total_passed=1, total_failed=0)
    j = report.to_json()
    assert '"edition_id": "test"' in j


def test_batch_qa_report_all_passed():
    report = BatchQAReport(edition_id="t", total_factors=5, total_passed=5, total_failed=0)
    assert report.all_passed
    report2 = BatchQAReport(edition_id="t", total_factors=5, total_passed=4, total_failed=1)
    assert not report2.all_passed


def test_batch_qa_report_empty_not_passed():
    report = BatchQAReport(edition_id="t", total_factors=0)
    assert not report.all_passed


# ---- _run_qa_on_factor ----

def test_qa_on_valid_factor():
    fd = _good_factor()
    r = _run_qa_on_factor(fd, {}, False)
    assert r.passed
    assert len(r.gate_errors) == 0


def test_qa_on_missing_factor_id():
    fd = _good_factor()
    fd.pop("factor_id")
    r = _run_qa_on_factor(fd, {}, False)
    assert not r.passed
    assert any("Q1" in e for e in r.gate_errors)


def test_qa_on_negative_vector():
    fd = _good_factor()
    fd["vectors"]["CH4"] = -1.0
    r = _run_qa_on_factor(fd, {}, False)
    assert not r.passed


def test_qa_on_missing_vectors():
    fd = _good_factor()
    fd["vectors"] = {}
    r = _run_qa_on_factor(fd, {}, False)
    assert not r.passed


def test_qa_warns_on_missing_license_info():
    fd = _good_factor()
    fd.pop("license_info")
    r = _run_qa_on_factor(fd, {}, False)
    assert r.passed  # warning, not error
    assert any("Q4" in w for w in r.gate_warnings)


def test_qa_warns_on_missing_dqs():
    fd = _good_factor()
    fd["dqs"] = {"temporal": 5}  # missing other dimensions
    r = _run_qa_on_factor(fd, {}, False)
    assert r.passed
    assert any("Q5" in w for w in r.gate_warnings)


def test_qa_warns_on_missing_provenance():
    fd = _good_factor()
    fd["provenance"] = {}
    r = _run_qa_on_factor(fd, {}, False)
    assert r.passed
    assert any("Q6" in w for w in r.gate_warnings)


def test_qa_outlier_co2e():
    fd = _good_factor()
    fd["gwp_100yr"]["co2e_total"] = 2e8  # extreme
    r = _run_qa_on_factor(fd, {}, False)
    assert not r.passed
    assert any("outlier" in e for e in r.gate_errors)


def test_qa_invalid_year():
    fd = _good_factor()
    fd["valid_from"] = "1980-01-01"
    r = _run_qa_on_factor(fd, {}, False)
    assert not r.passed


# ---- run_batch_qa_on_dicts ----

def test_batch_qa_all_valid():
    factors = [_good_factor(f"EF:EPA:f{i}:US:2024:v1") for i in range(5)]
    report = run_batch_qa_on_dicts(factors, edition_id="test", max_workers=1)
    assert report.total_factors == 5
    assert report.total_passed == 5
    assert report.total_failed == 0
    assert report.all_passed


def test_batch_qa_mixed():
    good = _good_factor("EF:EPA:good:US:2024:v1")
    bad = _good_factor("EF:EPA:bad:US:2024:v1")
    bad["vectors"] = {}  # fails Q1
    report = run_batch_qa_on_dicts([good, bad], edition_id="test", max_workers=1)
    assert report.total_passed == 1
    assert report.total_failed == 1


def test_batch_qa_empty():
    report = run_batch_qa_on_dicts([], edition_id="test")
    assert report.total_factors == 0
    assert report.total_passed == 0
    assert not report.all_passed


def test_batch_qa_parallel():
    factors = [_good_factor(f"EF:EPA:p{i}:US:2024:v1") for i in range(20)]
    report = run_batch_qa_on_dicts(factors, edition_id="test", max_workers=4)
    assert report.total_factors == 20
    assert report.total_passed == 20


def test_batch_qa_sorted_results():
    factors = [_good_factor(f"EF:EPA:z{i}:US:2024:v1") for i in range(3)]
    report = run_batch_qa_on_dicts(factors, edition_id="test", max_workers=1)
    ids = [r.factor_id for r in report.per_factor]
    assert ids == sorted(ids)


def test_batch_qa_pass_rate():
    factors = [_good_factor(f"EF:EPA:r{i}:US:2024:v1") for i in range(4)]
    factors[0]["vectors"] = {}  # fail
    report = run_batch_qa_on_dicts(factors, edition_id="test", max_workers=1)
    d = report.to_dict()
    assert d["pass_rate"] == 0.75
