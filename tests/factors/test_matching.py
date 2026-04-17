# -*- coding: utf-8 -*-
"""Tests for greenlang.factors.matching.pipeline (F042 hybrid search)."""

from __future__ import annotations

import pytest

from greenlang.factors.matching.pipeline import (
    HybridConfig,
    MatchRequest,
    _reciprocal_rank_fusion,
    _token_overlap,
    run_match,
)


# ---- Token overlap (existing) ----

def test_token_overlap_scoring():
    assert _token_overlap("diesel combustion", "diesel combustion stationary") == pytest.approx(1.0)
    assert _token_overlap("diesel", "electricity grid") == pytest.approx(0.0)
    assert 0.0 < _token_overlap("diesel combustion", "diesel fuel oil") < 1.0
    assert _token_overlap("", "anything") == pytest.approx(0.0)


# ---- RRF ----

def test_rrf_single_ranking():
    rankings = [["A", "B", "C"]]
    weights = [1.0]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = [f for f, _ in fused]
    assert ids == ["A", "B", "C"]


def test_rrf_two_identical_rankings():
    rankings = [["A", "B", "C"], ["A", "B", "C"]]
    weights = [0.5, 0.5]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = [f for f, _ in fused]
    assert ids == ["A", "B", "C"]


def test_rrf_two_different_rankings():
    rankings = [["A", "B", "C"], ["C", "B", "A"]]
    weights = [0.5, 0.5]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = [f for f, _ in fused]
    # With k=60: A gets 0.5/61 + 0.5/63, B gets 0.5/62 + 0.5/62, C gets 0.5/63 + 0.5/61
    # A and C have same score (symmetric), B has slightly less
    # A and C tie; A wins alphabetically due to sort stability
    assert len(ids) == 3
    assert set(ids) == {"A", "B", "C"}


def test_rrf_weighted_bias():
    """With high lexical weight, lexical ranking wins."""
    rankings = [["A", "B", "C"], ["C", "B", "A"]]
    weights = [0.9, 0.1]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = [f for f, _ in fused]
    assert ids[0] == "A"  # Lexical first item wins with high weight


def test_rrf_semantic_bias():
    """With high semantic weight, semantic ranking wins."""
    rankings = [["A", "B", "C"], ["C", "B", "A"]]
    weights = [0.1, 0.9]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = [f for f, _ in fused]
    assert ids[0] == "C"  # Semantic first item wins with high weight


def test_rrf_disjoint_sets():
    """Items only in one ranking still appear in fusion."""
    rankings = [["A", "B"], ["C", "D"]]
    weights = [0.5, 0.5]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = set(f for f, _ in fused)
    assert ids == {"A", "B", "C", "D"}


def test_rrf_scores_positive():
    rankings = [["A", "B", "C"]]
    weights = [1.0]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    for _, score in fused:
        assert score > 0


def test_rrf_empty_rankings():
    fused = _reciprocal_rank_fusion([], [], k=60)
    assert fused == []


def test_rrf_one_empty_ranking():
    rankings = [["A", "B"], []]
    weights = [0.5, 0.5]
    fused = _reciprocal_rank_fusion(rankings, weights, k=60)
    ids = [f for f, _ in fused]
    assert ids == ["A", "B"]


# ---- HybridConfig ----

def test_hybrid_config_defaults():
    cfg = HybridConfig()
    assert cfg.lexical_weight == 0.4
    assert cfg.semantic_weight == 0.6
    assert cfg.dqs_boost_factor == 0.01
    assert cfg.semantic_k == 50
    assert cfg.rrf_k == 60
    assert cfg.enable_semantic is True


def test_hybrid_config_custom():
    cfg = HybridConfig(lexical_weight=0.7, semantic_weight=0.3, rrf_k=100)
    assert cfg.lexical_weight == 0.7
    assert cfg.semantic_weight == 0.3
    assert cfg.rrf_k == 100


# ---- run_match (lexical-only, backward compat) ----

def test_run_match_returns_results(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(memory_catalog, eid, req)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "factor_id" in results[0]
    assert "score" in results[0]


def test_match_score_ordering(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="diesel", limit=10)
    results = run_match(memory_catalog, eid, req)
    if len(results) >= 2:
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


def test_match_geography_filter(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="electricity", geography="US", limit=20)
    results = run_match(memory_catalog, eid, req)
    assert isinstance(results, list)


def test_match_fuel_type_filter(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="diesel", fuel_type="diesel", limit=20)
    results = run_match(memory_catalog, eid, req)
    assert isinstance(results, list)


def test_match_scope_filter(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="electricity", scope="2", limit=20)
    results = run_match(memory_catalog, eid, req)
    assert isinstance(results, list)


def test_match_limit_respected(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="diesel", limit=3)
    results = run_match(memory_catalog, eid, req)
    assert len(results) <= 3


def test_match_empty_query_defaults(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="", limit=5)
    results = run_match(memory_catalog, eid, req)
    assert isinstance(results, list)


def test_match_explanation_has_mode(memory_catalog):
    eid = memory_catalog.get_default_edition_id()
    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(memory_catalog, eid, req)
    if results:
        assert "lexical" in results[0]["explanation"]["why"]


# ---- run_match (hybrid mode with mock semantic) ----

def test_match_hybrid_with_semantic(memory_catalog):
    """Hybrid match with mock semantic search function."""
    eid = memory_catalog.get_default_edition_id()

    # Get some real factor IDs from the repo
    factors, _ = memory_catalog.list_factors(eid, limit=5)
    factor_ids = [f.factor_id for f in factors]

    def mock_semantic(query, edition_id, k):
        return [
            {"factor_id": fid, "similarity": 0.9 - i * 0.1}
            for i, fid in enumerate(factor_ids[:3])
        ]

    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(
        memory_catalog, eid, req,
        semantic_search_fn=mock_semantic,
    )
    assert isinstance(results, list)
    assert len(results) > 0
    if results:
        assert "hybrid" in results[0]["explanation"]["why"]


def test_match_hybrid_fallback_on_semantic_error(memory_catalog):
    """Falls back to lexical when semantic search raises."""
    eid = memory_catalog.get_default_edition_id()

    def broken_semantic(query, edition_id, k):
        raise RuntimeError("Connection refused")

    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(
        memory_catalog, eid, req,
        semantic_search_fn=broken_semantic,
    )
    assert isinstance(results, list)
    assert len(results) > 0
    if results:
        assert "lexical" in results[0]["explanation"]["why"]


def test_match_hybrid_disabled_by_config(memory_catalog):
    """Semantic disabled in config -> lexical only."""
    eid = memory_catalog.get_default_edition_id()

    def mock_semantic(query, edition_id, k):
        return [{"factor_id": "EF:mock", "similarity": 0.99}]

    cfg = HybridConfig(enable_semantic=False)
    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(
        memory_catalog, eid, req,
        semantic_search_fn=mock_semantic,
        hybrid_config=cfg,
    )
    if results:
        assert "lexical" in results[0]["explanation"]["why"]


def test_match_hybrid_empty_semantic(memory_catalog):
    """Empty semantic results -> lexical-only mode."""
    eid = memory_catalog.get_default_edition_id()

    def empty_semantic(query, edition_id, k):
        return []

    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(
        memory_catalog, eid, req,
        semantic_search_fn=empty_semantic,
    )
    if results:
        assert "lexical" in results[0]["explanation"]["why"]


# ---- run_match with reranking callback ----

def test_match_with_reranker(memory_catalog):
    """Verify rerank_fn callback is applied."""
    eid = memory_catalog.get_default_edition_id()

    def mock_reranker(query, candidates):
        # Reverse the order
        return list(reversed(candidates))

    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(
        memory_catalog, eid, req,
        rerank_fn=mock_reranker,
    )
    if results:
        assert "llm_reranked" in results[0]["explanation"]["why"]


def test_match_reranker_failure_fallback(memory_catalog):
    """Failed reranker falls back to base ranking."""
    eid = memory_catalog.get_default_edition_id()

    def broken_reranker(query, candidates):
        raise RuntimeError("LLM unavailable")

    req = MatchRequest(activity_description="diesel", limit=5)
    results = run_match(
        memory_catalog, eid, req,
        rerank_fn=broken_reranker,
    )
    assert isinstance(results, list)
    if results:
        assert "lexical" in results[0]["explanation"]["why"]
