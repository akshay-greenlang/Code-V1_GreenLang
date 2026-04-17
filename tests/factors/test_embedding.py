# -*- coding: utf-8 -*-
"""Tests for the embedding pipeline (F041)."""

from __future__ import annotations

import math

import pytest

from greenlang.factors.matching.embedding import (
    EmbeddingConfig,
    EmbeddingPipeline,
    EmbeddingStats,
    StubEmbeddingModel,
    build_search_text,
)
from greenlang.factors.matching.pgvector_index import EmbeddingRecord


# ---- build_search_text ----

def test_search_text_from_dict():
    d = {
        "fuel_type": "diesel",
        "geography": "US",
        "scope": "1",
        "boundary": "combustion",
        "tags": ["stationary", "fossil"],
        "notes": "EPA 2024",
        "unit": "gallons",
        "source_org": "EPA",
    }
    text = build_search_text(d)
    assert "diesel" in text
    assert "US" in text
    assert "1" in text
    assert "combustion" in text
    assert "stationary" in text
    assert "EPA" in text
    assert "gallons" in text


def test_search_text_from_dict_minimal():
    d = {"fuel_type": "coal"}
    text = build_search_text(d)
    assert text.strip() == "coal"


def test_search_text_from_dict_empty():
    assert build_search_text({}) == ""


def test_search_text_from_record(sample_factor):
    text = build_search_text(sample_factor)
    assert len(text) > 0
    assert sample_factor.fuel_type in text
    assert sample_factor.geography in text


def test_search_text_from_record_includes_scope(sample_factor):
    text = build_search_text(sample_factor)
    assert sample_factor.scope.value in text


# ---- StubEmbeddingModel ----

def test_stub_model_dimension():
    model = StubEmbeddingModel(dim=384)
    assert model.dimension == 384


def test_stub_model_encode_single():
    model = StubEmbeddingModel(dim=384)
    vec = model.encode_single("diesel combustion")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec)


def test_stub_model_is_unit_normalized():
    model = StubEmbeddingModel(dim=384)
    vec = model.encode_single("diesel combustion")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_stub_model_deterministic():
    model = StubEmbeddingModel(dim=384)
    v1 = model.encode_single("diesel US")
    v2 = model.encode_single("diesel US")
    assert v1 == v2


def test_stub_model_different_text_different_vectors():
    model = StubEmbeddingModel(dim=384)
    v1 = model.encode_single("diesel")
    v2 = model.encode_single("electricity")
    assert v1 != v2


def test_stub_model_batch_encode():
    model = StubEmbeddingModel(dim=384)
    texts = ["diesel", "electricity", "natural gas"]
    vecs = model.encode(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


def test_stub_model_case_insensitive():
    model = StubEmbeddingModel(dim=384)
    v1 = model.encode_single("Diesel Combustion")
    v2 = model.encode_single("diesel combustion")
    assert v1 == v2


# ---- EmbeddingConfig ----

def test_config_defaults():
    cfg = EmbeddingConfig()
    assert cfg.model_name == "all-MiniLM-L6-v2"
    assert cfg.embedding_dim == 384
    assert cfg.batch_size == 1000
    assert cfg.use_stub is False


def test_config_custom():
    cfg = EmbeddingConfig(model_name="test-model", batch_size=500, use_stub=True)
    assert cfg.model_name == "test-model"
    assert cfg.batch_size == 500
    assert cfg.use_stub is True


# ---- EmbeddingPipeline ----

def test_pipeline_from_config_stub():
    cfg = EmbeddingConfig(use_stub=True, embedding_dim=384)
    pipeline = EmbeddingPipeline.from_config(cfg)
    assert pipeline.dimension == 384


def test_pipeline_embed_text():
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    vec = pipeline.embed_text("diesel combustion")
    assert len(vec) == 384


def test_pipeline_embed_query():
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    vec = pipeline.embed_query("diesel US scope 1")
    assert len(vec) == 384
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


def test_pipeline_embed_factors_dicts():
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    factors = [
        {"factor_id": "EF:1", "fuel_type": "diesel", "geography": "US", "content_hash": "h1"},
        {"factor_id": "EF:2", "fuel_type": "electricity", "geography": "EU", "content_hash": "h2"},
    ]
    records, stats = pipeline.embed_factors("ed1", factors)
    assert len(records) == 2
    assert stats.total_factors == 2
    assert stats.embedded == 2
    assert stats.skipped_cached == 0
    assert stats.errors == 0
    assert all(isinstance(r, EmbeddingRecord) for r in records)
    assert records[0].edition_id == "ed1"
    assert records[0].factor_id == "EF:1"
    assert len(records[0].embedding) == 384


def test_pipeline_embed_factors_with_cache():
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    factors = [
        {"factor_id": "EF:1", "fuel_type": "diesel", "content_hash": "h1"},
        {"factor_id": "EF:2", "fuel_type": "electricity", "content_hash": "h2"},
        {"factor_id": "EF:3", "fuel_type": "coal", "content_hash": "h3"},
    ]
    cached = {"EF:1": "h1", "EF:2": "h2"}  # These are up-to-date
    records, stats = pipeline.embed_factors("ed1", factors, cached_hashes=cached)
    assert len(records) == 1  # Only EF:3 needed embedding
    assert stats.embedded == 1
    assert stats.skipped_cached == 2


def test_pipeline_embed_factors_stale_cache():
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    factors = [
        {"factor_id": "EF:1", "fuel_type": "diesel", "content_hash": "h1_new"},
    ]
    cached = {"EF:1": "h1_old"}  # Hash changed -> needs re-embedding
    records, stats = pipeline.embed_factors("ed1", factors, cached_hashes=cached)
    assert len(records) == 1
    assert stats.embedded == 1
    assert stats.skipped_cached == 0


def test_pipeline_embed_factors_records(emission_db):
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    factors = list(emission_db.factors.values())[:5]
    records, stats = pipeline.embed_factors("test-ed", factors)
    assert len(records) == 5
    assert stats.embedded == 5
    for rec in records:
        assert len(rec.embedding) == 384
        assert len(rec.search_text) > 0


def test_pipeline_embed_factors_empty():
    cfg = EmbeddingConfig(use_stub=True)
    pipeline = EmbeddingPipeline.from_config(cfg)
    records, stats = pipeline.embed_factors("ed1", [])
    assert records == []
    assert stats.total_factors == 0
    assert stats.embedded == 0


def test_pipeline_from_config_fallback():
    """Falls back to stub if sentence-transformers not available."""
    cfg = EmbeddingConfig(use_stub=False)
    pipeline = EmbeddingPipeline.from_config(cfg)
    # Should not raise — either loads real model or falls back to stub
    assert pipeline.dimension > 0


# ---- EmbeddingStats ----

def test_stats_defaults():
    stats = EmbeddingStats()
    assert stats.total_factors == 0
    assert stats.embedded == 0
    assert stats.skipped_cached == 0
    assert stats.errors == 0
