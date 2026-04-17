# -*- coding: utf-8 -*-
"""Tests for the comprehensive evaluation suite (F044)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenlang.factors.matching.evaluation import (
    CaseResult,
    EvalMetrics,
    EvalReport,
    GoldCase,
    MatchEvaluator,
    _dcg,
    _extract_fuel_type,
    _ndcg,
    _reciprocal_rank,
    _compute_metrics,
    load_gold_cases,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---- GoldCase ----

def test_gold_case_from_dict():
    d = {"id": "g001", "activity": "diesel", "expected_fuel_type": "diesel"}
    c = GoldCase.from_dict(d)
    assert c.id == "g001"
    assert c.activity == "diesel"
    assert c.expected_fuel_type == "diesel"
    assert c.geography is None
    assert c.difficulty == "normal"


def test_gold_case_from_dict_full():
    d = {
        "id": "g100", "activity": "diesel US",
        "expected_fuel_type": "diesel", "geography": "US",
        "domain": "energy", "difficulty": "hard", "scope": "1",
    }
    c = GoldCase.from_dict(d)
    assert c.geography == "US"
    assert c.domain == "energy"
    assert c.difficulty == "hard"
    assert c.scope == "1"


# ---- Helper functions ----

def test_extract_fuel_type():
    assert _extract_fuel_type("US:diesel:gallons:1:combustion:IPCC") == "diesel"
    assert _extract_fuel_type("EU:electricity:kwh:2:combustion:IPCC") == "electricity"
    assert _extract_fuel_type("single") == ""


def test_reciprocal_rank_found():
    assert _reciprocal_rank(["diesel", "gas", "coal"], "diesel") == 1.0
    assert _reciprocal_rank(["gas", "diesel", "coal"], "diesel") == 0.5
    assert _reciprocal_rank(["gas", "coal", "diesel"], "diesel") == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found():
    assert _reciprocal_rank(["gas", "coal"], "diesel") == 0.0
    assert _reciprocal_rank([], "diesel") == 0.0


def test_dcg_basic():
    # relevances: [1, 0, 1] -> DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5
    dcg = _dcg([1.0, 0.0, 1.0], k=3)
    assert dcg == pytest.approx(1.0 + 0.5, rel=1e-3)


def test_dcg_empty():
    assert _dcg([], k=5) == 0.0


def test_dcg_respects_k():
    dcg_2 = _dcg([1.0, 1.0, 1.0], k=2)
    dcg_3 = _dcg([1.0, 1.0, 1.0], k=3)
    assert dcg_2 < dcg_3


def test_ndcg_perfect():
    assert _ndcg([1.0, 1.0, 1.0], k=3) == pytest.approx(1.0)


def test_ndcg_worst():
    assert _ndcg([0.0, 0.0, 1.0], k=3) < 1.0


def test_ndcg_no_relevance():
    assert _ndcg([0.0, 0.0, 0.0], k=3) == 0.0


# ---- CaseResult ----

def test_case_result_hit():
    r = CaseResult(
        case_id="g1", expected_fuel_type="diesel",
        matched_fuel_types=["diesel", "gas"], matched_factor_ids=["EF:1", "EF:2"],
        hit_at_1=True, hit_at_3=True, hit_at_5=True, hit_at_10=True,
        reciprocal_rank=1.0,
    )
    assert r.hit_at_1 is True
    assert r.reciprocal_rank == 1.0


def test_case_result_miss():
    r = CaseResult(
        case_id="g2", expected_fuel_type="diesel",
        matched_fuel_types=["gas", "coal"], matched_factor_ids=["EF:1", "EF:2"],
        hit_at_1=False, hit_at_3=False, hit_at_5=False, hit_at_10=False,
        reciprocal_rank=0.0,
    )
    assert r.hit_at_1 is False


# ---- _compute_metrics ----

def test_compute_metrics_perfect():
    results = [
        CaseResult("g1", "diesel", ["diesel"], ["EF:1"],
                   True, True, True, True, 1.0, latency_ms=5.0),
        CaseResult("g2", "gas", ["gas"], ["EF:2"],
                   True, True, True, True, 1.0, latency_ms=3.0),
    ]
    m = _compute_metrics(results)
    assert m.total_cases == 2
    assert m.precision_at_1 == 1.0
    assert m.mrr == 1.0
    assert m.avg_latency_ms == 4.0


def test_compute_metrics_half():
    results = [
        CaseResult("g1", "diesel", ["diesel"], ["EF:1"],
                   True, True, True, True, 1.0),
        CaseResult("g2", "gas", ["coal"], ["EF:2"],
                   False, False, False, False, 0.0),
    ]
    m = _compute_metrics(results)
    assert m.precision_at_1 == 0.5
    assert m.mrr == 0.5
    assert m.misses_at_1 == 1


def test_compute_metrics_empty():
    m = _compute_metrics([])
    assert m.total_cases == 0
    assert m.precision_at_1 == 0.0


# ---- EvalReport ----

def test_report_summary():
    report = EvalReport(mode="lexical")
    report.overall = EvalMetrics(
        total_cases=100, precision_at_1=0.85, precision_at_3=0.92,
        precision_at_5=0.95, precision_at_10=0.98, mrr=0.88,
        ndcg_at_5=0.90, ndcg_at_10=0.92, avg_latency_ms=5.0,
    )
    summary = report.summary()
    assert "lexical" in summary
    assert "85.00%" in summary
    assert "100 cases" in summary


def test_report_to_dict():
    report = EvalReport(mode="hybrid", timestamp="2026-04-17T00:00:00")
    report.overall = EvalMetrics(total_cases=50, precision_at_1=0.80, mrr=0.85)
    d = report.to_dict()
    assert d["mode"] == "hybrid"
    assert d["overall"]["precision_at_1"] == 0.80
    assert d["overall"]["mrr"] == 0.85


def test_report_domain_breakdown():
    report = EvalReport(mode="test")
    report.by_domain = {
        "energy": EvalMetrics(total_cases=30, precision_at_1=0.90, mrr=0.92),
        "transport": EvalMetrics(total_cases=20, precision_at_1=0.75, mrr=0.80),
    }
    summary = report.summary()
    assert "energy" in summary
    assert "transport" in summary


# ---- load_gold_cases ----

def test_load_gold_cases():
    cases = load_gold_cases(FIXTURES_DIR / "gold_eval_full.json")
    assert len(cases) >= 500
    for c in cases:
        assert "id" in c
        assert "activity" in c
        assert "expected_fuel_type" in c


def test_load_gold_cases_has_domains():
    cases = load_gold_cases(FIXTURES_DIR / "gold_eval_full.json")
    domains = set(c.get("domain") for c in cases if c.get("domain"))
    assert len(domains) >= 5


def test_load_gold_cases_has_difficulties():
    cases = load_gold_cases(FIXTURES_DIR / "gold_eval_full.json")
    hard = [c for c in cases if c.get("difficulty") == "hard"]
    assert len(hard) >= 50


# ---- MatchEvaluator integration ----

def test_evaluator_runs(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    evaluator = MatchEvaluator(memory_catalog, eid)
    cases = [
        {"id": "t1", "activity": "diesel", "expected_fuel_type": "diesel"},
        {"id": "t2", "activity": "electricity", "expected_fuel_type": "electricity"},
    ]
    report = evaluator.evaluate(cases, mode="test")
    assert report.mode == "test"
    assert report.overall.total_cases == 2
    assert len(report.case_results) == 2


def test_evaluator_with_domain(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    evaluator = MatchEvaluator(memory_catalog, eid)
    cases = [
        {"id": "t1", "activity": "diesel", "expected_fuel_type": "diesel", "domain": "energy"},
        {"id": "t2", "activity": "electricity", "expected_fuel_type": "electricity", "domain": "energy"},
        {"id": "t3", "activity": "diesel truck", "expected_fuel_type": "diesel", "domain": "transport"},
    ]
    report = evaluator.evaluate(cases, mode="test")
    assert "energy" in report.by_domain
    assert "transport" in report.by_domain
    assert report.by_domain["energy"].total_cases == 2


def test_evaluator_ab_compare(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    evaluator = MatchEvaluator(memory_catalog, eid)
    cases = [
        {"id": "t1", "activity": "diesel", "expected_fuel_type": "diesel"},
    ]
    configs = {
        "lexical": {},
        "custom": {"match_limit": 5},
    }
    reports = evaluator.ab_compare(cases, configs)
    assert "lexical" in reports
    assert "custom" in reports
    assert reports["lexical"].overall.total_cases == 1


def test_evaluator_full_gold_set(memory_catalog):
    """Run full 500+ case evaluation and verify metrics are reported."""
    eid = memory_catalog.get_default_edition_id()
    evaluator = MatchEvaluator(memory_catalog, eid)
    cases = load_gold_cases(FIXTURES_DIR / "gold_eval_full.json")
    report = evaluator.evaluate(cases, mode="lexical")

    assert report.overall.total_cases >= 500
    assert 0.0 <= report.overall.precision_at_1 <= 1.0
    assert 0.0 <= report.overall.mrr <= 1.0
    assert 0.0 <= report.overall.ndcg_at_10 <= 1.0
    assert report.overall.avg_latency_ms >= 0

    # Print report for visibility
    print(report.summary())
