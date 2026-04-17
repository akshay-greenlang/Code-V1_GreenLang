# -*- coding: utf-8 -*-
"""Tests for cross-source consistency checker (F022)."""

from __future__ import annotations

import pytest

from greenlang.factors.quality.cross_source import (
    ConsistencyCheck,
    ConsistencyReport,
    SourceValue,
    check_cross_source_consistency,
    _compute_divergence,
    _extract_activity_key,
)


def _factor(fid, fuel="natural_gas", geo="US", scope="1", source_id="epa", co2=0.05, **kw):
    d = {
        "factor_id": fid,
        "fuel_type": fuel,
        "geography": geo,
        "scope": scope,
        "unit": "scf",
        "vectors": {"CO2": co2, "CH4": 0.001, "N2O": 0.0001},
        "gwp_100yr": {},
        "provenance": {"source_org": source_id.upper()},
        "source_id": source_id,
    }
    d.update(kw)
    return d


# ---- _compute_divergence ----

def test_divergence_identical():
    assert _compute_divergence([1.0, 1.0, 1.0]) == 0.0


def test_divergence_20_percent():
    d = _compute_divergence([1.0, 1.2])
    assert abs(d - 0.20) < 0.01


def test_divergence_100_percent():
    d = _compute_divergence([1.0, 2.0])
    assert abs(d - 1.0) < 0.01


def test_divergence_single_value():
    assert _compute_divergence([5.0]) == 0.0


def test_divergence_empty():
    assert _compute_divergence([]) == 0.0


def test_divergence_all_zero():
    assert _compute_divergence([0.0, 0.0]) == 0.0


# ---- _extract_activity_key ----

def test_activity_key_from_dict():
    ak = _extract_activity_key(_factor("EF:EPA:ng:US:2024:v1"))
    assert ak == "natural_gas|US|1"


def test_activity_key_case_normalized():
    f = _factor("EF:X:1", fuel="Natural Gas", geo="us")
    ak = _extract_activity_key(f)
    assert ak == "natural gas|US|1"


# ---- SourceValue ----

def test_source_value_to_dict():
    sv = SourceValue(factor_id="EF:X:1", source_id="epa", co2_value=0.05, co2e_total=0.05, unit="scf")
    d = sv.to_dict()
    assert d["factor_id"] == "EF:X:1"
    assert d["co2_value"] == 0.05


# ---- ConsistencyCheck ----

def test_consistency_check_to_dict():
    c = ConsistencyCheck(
        activity_key="ng|US|1", fuel_type="ng", geography="US", scope="1",
        max_divergence=0.15, severity="ok", detail="within tolerance",
    )
    d = c.to_dict()
    assert d["severity"] == "ok"
    assert d["max_divergence"] == 0.15


# ---- ConsistencyReport ----

def test_report_no_issues():
    r = ConsistencyReport(edition_id="test", total_ok=3)
    assert not r.has_issues
    assert not r.needs_review


def test_report_has_warnings():
    r = ConsistencyReport(edition_id="test", total_warnings=1)
    assert r.has_issues
    assert not r.needs_review


def test_report_needs_review():
    r = ConsistencyReport(edition_id="test", total_reviews=1)
    assert r.has_issues
    assert r.needs_review


def test_report_to_dict():
    r = ConsistencyReport(edition_id="test", total_activities=2, total_ok=1, total_warnings=1)
    d = r.to_dict()
    assert d["edition_id"] == "test"
    assert d["total_activities"] == 2


# ---- check_cross_source_consistency ----

def test_consistent_sources():
    """Two sources with same CO2 value -> ok."""
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa", co2=0.05),
        _factor("EF:DEFRA:ng:US:2024:v1", source_id="defra", co2=0.05),
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    assert report.total_activities == 1
    assert report.total_ok == 1
    assert not report.has_issues


def test_warning_threshold():
    """25% divergence -> warning."""
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa", co2=1.0),
        _factor("EF:DEFRA:ng:US:2024:v1", source_id="defra", co2=1.25),
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    assert report.total_warnings == 1
    assert report.has_issues
    assert not report.needs_review


def test_review_threshold():
    """60% divergence -> review_required."""
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa", co2=1.0),
        _factor("EF:IPCC:ng:US:2024:v1", source_id="ipcc", co2=1.6),
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    assert report.total_reviews == 1
    assert report.needs_review


def test_single_source_skipped():
    """Only one source -> not compared."""
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa", co2=1.0),
        _factor("EF:EPA:ng:US:2024:v2", source_id="epa", co2=2.0),
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    assert report.total_activities == 0
    assert not report.has_issues


def test_different_activities_separate():
    """Different fuels are compared separately."""
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", fuel="natural_gas", source_id="epa", co2=1.0),
        _factor("EF:DEFRA:ng:US:2024:v1", fuel="natural_gas", source_id="defra", co2=1.0),
        _factor("EF:EPA:diesel:US:2024:v1", fuel="diesel", source_id="epa", co2=2.0),
        _factor("EF:DEFRA:diesel:US:2024:v1", fuel="diesel", source_id="defra", co2=3.0),  # 50% divergence
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    assert report.total_activities == 2
    severities = {c.fuel_type: c.severity for c in report.checks}
    assert severities["natural_gas"] == "ok"
    assert severities["diesel"] in ("warning", "review_required")


def test_three_sources():
    """Three sources compared — worst pair drives severity."""
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa", co2=1.0),
        _factor("EF:DEFRA:ng:US:2024:v1", source_id="defra", co2=1.1),
        _factor("EF:IPCC:ng:US:2024:v1", source_id="ipcc", co2=1.8),
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    assert report.total_activities == 1
    # 80% divergence (1.0 vs 1.8) -> review_required
    assert report.needs_review


def test_empty_factors():
    report = check_cross_source_consistency([], edition_id="test")
    assert report.total_activities == 0
    assert not report.has_issues


def test_custom_thresholds():
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa", co2=1.0),
        _factor("EF:DEFRA:ng:US:2024:v1", source_id="defra", co2=1.1),
    ]
    # 10% divergence, set warning threshold to 5%
    report = check_cross_source_consistency(
        factors, edition_id="test",
        warning_threshold=0.05,
        review_threshold=0.30,
    )
    assert report.total_warnings == 1


def test_report_checks_sorted():
    factors = [
        _factor("EF:EPA:diesel:US:2024:v1", fuel="diesel", source_id="epa", co2=1.0),
        _factor("EF:DEFRA:diesel:US:2024:v1", fuel="diesel", source_id="defra", co2=1.0),
        _factor("EF:EPA:coal:US:2024:v1", fuel="coal", source_id="epa", co2=2.0),
        _factor("EF:DEFRA:coal:US:2024:v1", fuel="coal", source_id="defra", co2=2.0),
    ]
    report = check_cross_source_consistency(factors, edition_id="test")
    keys = [c.activity_key for c in report.checks]
    assert keys == sorted(keys)
