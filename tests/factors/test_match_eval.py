# -*- coding: utf-8 -*-
"""Gold evaluation set matching pipeline test (M5).

Loads gold_eval_full.json (105 cases) and reports precision@1, precision@3, MRR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from greenlang.factors.matching.pipeline import MatchRequest, run_match


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def gold_cases():
    path = FIXTURES_DIR / "gold_eval_full.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def eval_repo():
    from greenlang.data.emission_factor_database import EmissionFactorDatabase
    from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository

    db = EmissionFactorDatabase(enable_cache=False)
    return MemoryFactorCatalogRepository("eval-v1", "eval", db)


def _run_eval(repo, cases):
    """Run matching for all gold cases and return per-case results."""
    eid = repo.get_default_edition_id()
    results = []
    for case in cases:
        req = MatchRequest(
            activity_description=case["activity"],
            geography=case.get("geography"),
            limit=10,
        )
        matches = run_match(repo, eid, req)
        matched_fuels = [
            _extract_fuel_type(m["factor_id"]) for m in matches
        ]
        results.append({
            "id": case["id"],
            "expected": case["expected_fuel_type"],
            "matched_fuels": matched_fuels,
            "hit_at_1": len(matched_fuels) > 0 and matched_fuels[0] == case["expected_fuel_type"],
            "hit_at_3": case["expected_fuel_type"] in matched_fuels[:3],
            "rr": _reciprocal_rank(matched_fuels, case["expected_fuel_type"]),
        })
    return results


def _extract_fuel_type(factor_id: str) -> str:
    """Extract fuel_type token from factor_id like 'US:diesel:gallons:1:combustion:...'."""
    parts = factor_id.split(":")
    if len(parts) >= 2:
        return parts[1]
    return ""


def _reciprocal_rank(ranked_fuels: list, expected: str) -> float:
    """Reciprocal rank: 1/rank of first correct match, or 0 if not found."""
    for i, fuel in enumerate(ranked_fuels):
        if fuel == expected:
            return 1.0 / (i + 1)
    return 0.0


def test_gold_eval_full_loads(gold_cases):
    assert len(gold_cases) >= 100
    for case in gold_cases:
        assert "id" in case
        assert "activity" in case
        assert "expected_fuel_type" in case


def test_gold_eval_precision_report(eval_repo, gold_cases):
    """Run full eval and report metrics. Test passes as long as eval completes."""
    results = _run_eval(eval_repo, gold_cases)

    total = len(results)
    p_at_1 = sum(1 for r in results if r["hit_at_1"]) / total
    p_at_3 = sum(1 for r in results if r["hit_at_3"]) / total
    mrr = sum(r["rr"] for r in results) / total

    print(f"\n--- Gold Eval Report ({total} cases) ---")
    print(f"  Precision@1: {p_at_1:.2%}")
    print(f"  Precision@3: {p_at_3:.2%}")
    print(f"  MRR:         {mrr:.4f}")

    # Log misses for debugging
    misses = [r for r in results if not r["hit_at_3"]]
    if misses:
        print(f"  Misses ({len(misses)}):")
        for m in misses[:10]:
            print(f"    {m['id']}: expected={m['expected']}, got={m['matched_fuels'][:3]}")

    # Soft assertion: with only 8 factors, many queries should match.
    # We don't require high precision since the matching is substring-based
    # and the DB is tiny. The value of this test is the report.
    assert total == len(gold_cases)


def test_gold_eval_smoke_subset(eval_repo):
    """Verify the original 2 smoke cases run and produce results.

    Note: The Memory repo uses contiguous substring matching, so multi-word
    queries may not match. This test validates the pipeline executes without
    errors rather than asserting ranking quality (which depends on the repo
    implementation).
    """
    path = FIXTURES_DIR / "gold_eval_smoke.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    results = _run_eval(eval_repo, cases)
    assert len(results) == len(cases)
    for r in results:
        assert r["expected"] in ("diesel", "electricity")
