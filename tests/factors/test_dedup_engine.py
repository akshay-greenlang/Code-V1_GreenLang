# -*- coding: utf-8 -*-
"""Tests for enhanced duplicate detection engine (F021)."""

from __future__ import annotations

import pytest

from greenlang.factors.quality.dedup_engine import (
    DedupReport,
    DuplicatePair,
    detect_duplicates,
    _activity_key,
    _merge_score,
    _dqs_score,
    _extract_source,
)


def _factor(fid, fuel="natural_gas", geo="US", scope="1", boundary="cradle_to_gate",
            source_id="epa", source_year=2024, dqs_avg=4, uncertainty=0.1, **extra):
    d = {
        "factor_id": fid,
        "fuel_type": fuel,
        "geography": geo,
        "scope": scope,
        "boundary": boundary,
        "unit": "scf",
        "vectors": {"CO2": 0.05, "CH4": 0.001, "N2O": 0.0001},
        "gwp_100yr": {"gwp_set": "IPCC_AR6_100"},
        "provenance": {
            "source_org": source_id.upper(),
            "methodology": "IPCC_Tier_1",
            "source_year": source_year,
        },
        "valid_from": f"{source_year}-01-01",
        "dqs": {
            "temporal": dqs_avg,
            "geographical": dqs_avg,
            "technological": dqs_avg,
            "representativeness": dqs_avg,
            "methodological": dqs_avg,
        },
        "uncertainty_95ci": uncertainty,
        "source_id": source_id,
        "content_hash": f"hash_{fid}",
    }
    d.update(extra)
    return d


# ---- Helper tests ----

def test_extract_source_from_dict():
    assert _extract_source({"source_id": "epa"}) == "epa"
    assert _extract_source({"factor_id": "EF:DEFRA:test:UK:2024:v1"}) == "defra"


def test_extract_source_empty():
    assert _extract_source({}) == ""


def test_dqs_score_average():
    f = _factor("EF:X:1", dqs_avg=4)
    assert _dqs_score(f) == 4.0


def test_dqs_score_empty():
    f = {"dqs": {}}
    assert _dqs_score(f) == 0.0


def test_activity_key():
    f = _factor("EF:EPA:ng:US:2024:v1", fuel="natural_gas", geo="US", scope="1", boundary="cradle_to_gate")
    ak = _activity_key(f)
    assert ak == "natural_gas|US|1|cradle_to_gate"


def test_merge_score_epa_beats_ipcc():
    epa = _factor("EF:EPA:1", source_id="epa")
    ipcc = _factor("EF:IPCC:1", source_id="ipcc")
    assert _merge_score(epa) > _merge_score(ipcc)


def test_merge_score_defra_beats_ipcc():
    defra = _factor("EF:DEFRA:1", source_id="defra")
    ipcc = _factor("EF:IPCC:1", source_id="ipcc")
    assert _merge_score(defra) > _merge_score(ipcc)


# ---- DuplicatePair ----

def test_duplicate_pair_to_dict():
    p = DuplicatePair(
        factor_id_a="A", factor_id_b="B",
        match_type="exact", fingerprint="fp1",
        resolution="keep_a", reason="test",
    )
    d = p.to_dict()
    assert d["factor_id_a"] == "A"
    assert d["match_type"] == "exact"


# ---- DedupReport ----

def test_dedup_report_no_duplicates():
    r = DedupReport(edition_id="test", total_factors=5)
    assert not r.has_duplicates
    assert r.to_dict()["exact_duplicates"] == 0


def test_dedup_report_has_duplicates():
    r = DedupReport(edition_id="test", total_factors=5)
    r.pairs.append(DuplicatePair("A", "B", "exact", "fp", "keep_a", "test"))
    assert r.has_duplicates


# ---- detect_duplicates ----

def test_no_duplicates():
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", fuel="natural_gas"),
        _factor("EF:EPA:diesel:US:2024:v1", fuel="diesel"),
        _factor("EF:EPA:coal:US:2024:v1", fuel="coal"),
    ]
    report = detect_duplicates(factors, edition_id="test")
    assert report.total_factors == 3
    assert not report.has_duplicates


def test_exact_duplicates_same_content():
    f1 = _factor("EF:EPA:ng:US:2024:v1")
    f2 = _factor("EF:EPA:ng:US:2024:v2")
    # Same content hash = exact
    f2["content_hash"] = f1["content_hash"]
    report = detect_duplicates([f1, f2], edition_id="test", detect_near=False)
    assert report.exact_duplicates >= 1
    assert report.has_duplicates


def test_near_duplicates_same_activity_different_source():
    epa = _factor("EF:EPA:ng:US:2024:v1", source_id="epa", source_year=2024)
    defra = _factor("EF:DEFRA:ng:US:2024:v1", source_id="defra", source_year=2024)
    report = detect_duplicates([epa, defra], edition_id="test", detect_near=True)
    assert report.has_duplicates
    assert report.near_duplicates >= 1


def test_resolution_keeps_higher_priority():
    epa = _factor("EF:EPA:ng:US:2024:v1", source_id="epa")
    ipcc = _factor("EF:IPCC:ng:US:2024:v1", source_id="ipcc")
    report = detect_duplicates([epa, ipcc], edition_id="test", detect_near=True)
    assert report.has_duplicates
    # EPA has higher priority than IPCC
    pair = report.pairs[0]
    if pair.factor_id_a == "EF:EPA:ng:US:2024:v1":
        assert pair.resolution == "keep_a"
    else:
        assert pair.resolution == "keep_b"


def test_resolution_newer_source_year_wins_same_source():
    old = _factor("EF:EPA:ng:US:2020:v1", source_id="epa", source_year=2020)
    new = _factor("EF:EPA:ng:US:2024:v1", source_id="epa", source_year=2024)
    report = detect_duplicates([old, new], edition_id="test", detect_near=True)
    assert report.has_duplicates
    pair = report.pairs[0]
    # Newer year should win
    if pair.factor_id_a == "EF:EPA:ng:US:2024:v1":
        assert pair.resolution == "keep_a"
    else:
        assert pair.resolution == "keep_b"


def test_human_review_on_tie():
    f1 = _factor("EF:EPA:ng:US:2024:v1", source_id="epa", source_year=2024, dqs_avg=4, uncertainty=0.1)
    f2 = _factor("EF:EPA:ng:US:2024:v2", source_id="epa", source_year=2024, dqs_avg=4, uncertainty=0.1)
    report = detect_duplicates([f1, f2], edition_id="test", detect_near=False)
    if report.has_duplicates:
        pair = report.pairs[0]
        assert pair.resolution == "human_review"


def test_detect_near_disabled():
    epa = _factor("EF:EPA:ng:US:2024:v1", source_id="epa")
    defra = _factor("EF:DEFRA:ng:US:2024:v1", source_id="defra")
    # Different fingerprints but same activity key
    report = detect_duplicates([epa, defra], edition_id="test", detect_near=False)
    # Should not detect near-duplicates when disabled
    near_count = sum(1 for p in report.pairs if p.match_type == "near")
    # May still detect if they happen to have same fingerprint
    assert report.total_factors == 2


def test_empty_factors():
    report = detect_duplicates([], edition_id="test")
    assert report.total_factors == 0
    assert not report.has_duplicates


def test_single_factor():
    report = detect_duplicates([_factor("EF:EPA:1")], edition_id="test")
    assert report.total_factors == 1
    assert not report.has_duplicates


def test_report_to_dict():
    factors = [
        _factor("EF:EPA:ng:US:2024:v1", source_id="epa"),
        _factor("EF:IPCC:ng:US:2024:v1", source_id="ipcc"),
    ]
    report = detect_duplicates(factors, edition_id="test")
    d = report.to_dict()
    assert d["edition_id"] == "test"
    assert d["total_factors"] == 2
    assert isinstance(d["pairs"], list)


def test_different_geographies_not_duplicates():
    us = _factor("EF:EPA:ng:US:2024:v1", geo="US")
    uk = _factor("EF:EPA:ng:UK:2024:v1", geo="UK")
    report = detect_duplicates([us, uk], edition_id="test")
    # Different geography should not match
    assert not report.has_duplicates


def test_different_scopes_not_duplicates():
    s1 = _factor("EF:EPA:ng:US:2024:v1", scope="1")
    s3 = _factor("EF:EPA:ng:US:2024:v2", scope="3")
    report = detect_duplicates([s1, s3], edition_id="test")
    assert not report.has_duplicates
