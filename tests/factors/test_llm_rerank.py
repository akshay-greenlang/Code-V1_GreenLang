# -*- coding: utf-8 -*-
"""Tests for LLM-assisted reranking (F043)."""

from __future__ import annotations

import json

import pytest

from greenlang.factors.matching.llm_rerank import (
    LLMRerankConfig,
    LLMReranker,
    RerankResult,
    StubLLMProvider,
    _build_rerank_prompt,
    _parse_rerank_response,
)


# ---- Test data ----

@pytest.fixture
def candidates():
    return [
        {"factor_id": "EF:diesel:US", "score": 0.9},
        {"factor_id": "EF:diesel:EU", "score": 0.8},
        {"factor_id": "EF:gasoline:US", "score": 0.7},
        {"factor_id": "EF:coal:US", "score": 0.6},
        {"factor_id": "EF:electricity:US", "score": 0.5},
    ]


# ---- LLMRerankConfig ----

def test_config_defaults():
    cfg = LLMRerankConfig()
    assert cfg.provider == "anthropic"
    assert cfg.max_candidates == 20
    assert cfg.temperature == 0.0
    assert cfg.rate_limit_rpm == 10
    assert cfg.enabled is True


def test_config_custom():
    cfg = LLMRerankConfig(provider="openai", max_candidates=10, rate_limit_rpm=5)
    assert cfg.provider == "openai"
    assert cfg.max_candidates == 10
    assert cfg.rate_limit_rpm == 5


# ---- _build_rerank_prompt ----

def test_build_prompt_contains_activity(candidates):
    prompt = _build_rerank_prompt("diesel combustion US", candidates)
    assert "diesel combustion US" in prompt


def test_build_prompt_contains_all_factor_ids(candidates):
    prompt = _build_rerank_prompt("diesel", candidates)
    for c in candidates:
        assert c["factor_id"] in prompt


def test_build_prompt_asks_for_json(candidates):
    prompt = _build_rerank_prompt("diesel", candidates)
    assert "ranked_factor_ids" in prompt
    assert "JSON" in prompt


# ---- _parse_rerank_response ----

def test_parse_valid_response(candidates):
    response = json.dumps({"ranked_factor_ids": [
        "EF:coal:US", "EF:diesel:US", "EF:diesel:EU", "EF:gasoline:US", "EF:electricity:US"
    ]})
    result = _parse_rerank_response(response, candidates)
    assert result[0]["factor_id"] == "EF:coal:US"
    assert len(result) == 5


def test_parse_response_with_markdown_fences(candidates):
    response = '```json\n{"ranked_factor_ids": ["EF:diesel:EU", "EF:diesel:US"]}\n```'
    result = _parse_rerank_response(response, candidates)
    assert result[0]["factor_id"] == "EF:diesel:EU"
    assert len(result) == 5  # Missing ones appended


def test_parse_invalid_json_returns_original(candidates):
    result = _parse_rerank_response("not valid json", candidates)
    assert result == candidates


def test_parse_missing_key_returns_original(candidates):
    result = _parse_rerank_response('{"wrong_key": []}', candidates)
    assert result == candidates


def test_parse_injected_factor_id_filtered(candidates):
    """LLM tries to inject a new factor_id — it gets filtered out."""
    response = json.dumps({"ranked_factor_ids": [
        "EF:INJECTED:FAKE",  # Not in original candidates
        "EF:diesel:US",
        "EF:diesel:EU",
    ]})
    result = _parse_rerank_response(response, candidates)
    # Only valid IDs kept
    assert result[0]["factor_id"] == "EF:diesel:US"
    assert len(result) == 5  # All originals preserved
    assert "EF:INJECTED:FAKE" not in [r["factor_id"] for r in result]


def test_parse_partial_ranking_appends_missing(candidates):
    response = json.dumps({"ranked_factor_ids": ["EF:diesel:US"]})
    result = _parse_rerank_response(response, candidates)
    assert result[0]["factor_id"] == "EF:diesel:US"
    assert len(result) == 5  # All original candidates preserved


def test_parse_duplicate_ids_deduplicated(candidates):
    response = json.dumps({"ranked_factor_ids": [
        "EF:diesel:US", "EF:diesel:US", "EF:diesel:EU"
    ]})
    result = _parse_rerank_response(response, candidates)
    factor_ids = [r["factor_id"] for r in result]
    assert len(factor_ids) == len(set(factor_ids))


def test_parse_preserves_original_data(candidates):
    """Ensure original candidate data (score, etc.) is preserved."""
    response = json.dumps({"ranked_factor_ids": [
        "EF:diesel:EU", "EF:diesel:US"
    ]})
    result = _parse_rerank_response(response, candidates)
    eu = next(r for r in result if r["factor_id"] == "EF:diesel:EU")
    assert eu["score"] == 0.8  # Original score preserved


# ---- StubLLMProvider ----

def test_stub_provider_returns_json(candidates):
    provider = StubLLMProvider()
    prompt = _build_rerank_prompt("diesel", candidates)
    response = provider.complete(prompt, LLMRerankConfig())
    parsed = json.loads(response)
    assert "ranked_factor_ids" in parsed


def test_stub_provider_reverses_order(candidates):
    provider = StubLLMProvider()
    prompt = _build_rerank_prompt("diesel", candidates)
    response = provider.complete(prompt, LLMRerankConfig())
    parsed = json.loads(response)
    ids = parsed["ranked_factor_ids"]
    original_ids = [c["factor_id"] for c in candidates]
    assert ids == list(reversed(original_ids))


# ---- LLMReranker ----

def test_reranker_with_stub(candidates):
    cfg = LLMRerankConfig(provider="stub", enabled=True)
    reranker = LLMReranker(cfg)
    result = reranker.rerank("diesel combustion US", candidates)
    assert isinstance(result, RerankResult)
    assert result.reranked_by_llm is True
    assert result.candidates_in == 5
    assert result.candidates_out == 5
    assert result.latency_ms >= 0


def test_reranker_stub_reverses(candidates):
    cfg = LLMRerankConfig(provider="stub", enabled=True)
    reranker = LLMReranker(cfg)
    result = reranker.rerank("diesel", candidates)
    # Stub reverses order
    assert result.reranked[0]["factor_id"] == "EF:electricity:US"


def test_reranker_disabled(candidates):
    cfg = LLMRerankConfig(enabled=False)
    reranker = LLMReranker(cfg)
    result = reranker.rerank("diesel", candidates)
    assert result.reranked_by_llm is False
    assert result.reranked == candidates


def test_reranker_respects_max_candidates():
    cfg = LLMRerankConfig(provider="stub", max_candidates=3)
    reranker = LLMReranker(cfg)
    candidates = [
        {"factor_id": f"EF:{i}", "score": 1.0 - i * 0.1}
        for i in range(10)
    ]
    result = reranker.rerank("test", candidates)
    assert result.candidates_in == 10
    assert result.candidates_out == 10  # All candidates preserved


def test_reranker_rate_limit():
    cfg = LLMRerankConfig(provider="stub", rate_limit_rpm=2)
    reranker = LLMReranker(cfg)
    candidates = [{"factor_id": "EF:1", "score": 0.9}]

    # First 2 calls should succeed
    r1 = reranker.rerank("test1", candidates)
    r2 = reranker.rerank("test2", candidates)
    assert r1.reranked_by_llm is True
    assert r2.reranked_by_llm is True

    # 3rd call should be rate limited
    r3 = reranker.rerank("test3", candidates)
    assert r3.reranked_by_llm is False


def test_reranker_as_callback(candidates):
    cfg = LLMRerankConfig(provider="stub")
    reranker = LLMReranker(cfg)
    fn = reranker.as_rerank_fn()
    result = fn("diesel", candidates)
    assert isinstance(result, list)
    assert len(result) == 5


def test_reranker_provider_error_fallback(candidates):
    """Provider error -> returns original candidates."""

    class FailingProvider:
        def complete(self, prompt, config):
            raise RuntimeError("API timeout")

    cfg = LLMRerankConfig(enabled=True)
    reranker = LLMReranker(cfg, provider=FailingProvider())
    result = reranker.rerank("diesel", candidates)
    assert result.reranked_by_llm is False
    assert result.reranked == candidates


# ---- RerankResult ----

def test_rerank_result_fields():
    result = RerankResult(
        reranked=[{"factor_id": "EF:1", "score": 0.9}],
        model_used="test",
        latency_ms=15.5,
        candidates_in=5,
        candidates_out=5,
        reranked_by_llm=True,
    )
    assert result.model_used == "test"
    assert result.latency_ms == 15.5
    assert result.reranked_by_llm is True


# ---- Value immutability guarantee ----

def test_reranking_never_mutates_values(candidates):
    """Core safety: LLM reranking must never change factor values."""
    cfg = LLMRerankConfig(provider="stub")
    reranker = LLMReranker(cfg)

    # Deep copy originals for comparison
    import copy
    originals = copy.deepcopy(candidates)

    result = reranker.rerank("diesel", candidates)

    # Verify all original candidates present with unchanged values
    result_map = {r["factor_id"]: r for r in result.reranked}
    for orig in originals:
        fid = orig["factor_id"]
        assert fid in result_map, f"Factor {fid} missing from reranked results"
        assert result_map[fid]["score"] == orig["score"], f"Score mutated for {fid}"
