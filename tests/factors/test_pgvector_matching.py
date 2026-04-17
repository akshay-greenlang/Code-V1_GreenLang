# -*- coding: utf-8 -*-
"""Tests for pgvector semantic index (F040).

Since CI has no Postgres+pgvector, these test the module structure,
config, and in-memory behavior. Integration tests require GL_FACTORS_PG_DSN.
"""

from __future__ import annotations

import pytest

from greenlang.factors.matching.pgvector_index import (
    DEFAULT_EMBEDDING_DIM,
    EMBEDDING_DIM_MINILM,
    EMBEDDING_DIM_MPNET,
    EMBEDDING_DIM_OPENAI,
    EmbeddingRecord,
    PgVectorConfig,
    PgVectorSemanticIndex,
)
from greenlang.factors.matching.semantic_index import NoopSemanticIndex, SemanticIndex


# ---- Dimension constants ----

def test_embedding_dim_constants():
    assert EMBEDDING_DIM_MINILM == 384
    assert EMBEDDING_DIM_MPNET == 768
    assert EMBEDDING_DIM_OPENAI == 1536
    assert DEFAULT_EMBEDDING_DIM == 384


# ---- PgVectorConfig ----

def test_config_defaults():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    assert cfg.dsn == "postgresql://localhost/test"
    assert cfg.embedding_dim == 384
    assert cfg.schema_name == "factors_catalog"
    assert cfg.table_name == "factor_embeddings"
    assert cfg.hnsw_m == 16
    assert cfg.hnsw_ef_construction == 64
    assert cfg.hnsw_ef_search == 40
    assert cfg.distance_metric == "cosine"
    assert cfg.max_results == 50


def test_config_custom():
    cfg = PgVectorConfig(
        dsn="postgresql://host/db",
        embedding_dim=768,
        hnsw_m=24,
        hnsw_ef_construction=128,
    )
    assert cfg.embedding_dim == 768
    assert cfg.hnsw_m == 24
    assert cfg.hnsw_ef_construction == 128


# ---- EmbeddingRecord ----

def test_embedding_record():
    rec = EmbeddingRecord(
        edition_id="2026.04.0",
        factor_id="EF:diesel:US",
        embedding=[0.1] * 384,
        search_text="diesel combustion US scope 1",
        content_hash="abc123",
    )
    assert rec.edition_id == "2026.04.0"
    assert rec.factor_id == "EF:diesel:US"
    assert len(rec.embedding) == 384
    assert rec.search_text == "diesel combustion US scope 1"
    assert rec.content_hash == "abc123"


# ---- PgVectorSemanticIndex (without connection) ----

def test_index_construction():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    assert idx._config is cfg
    assert idx._conn is None
    assert idx._initialized is False
    assert idx.is_available is False


def test_index_full_table_name():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    assert idx.full_table_name == "factors_catalog.factor_embeddings"


def test_index_custom_table_name():
    cfg = PgVectorConfig(
        dsn="postgresql://localhost/test",
        schema_name="custom_schema",
        table_name="custom_embeddings",
    )
    idx = PgVectorSemanticIndex(cfg)
    assert idx.full_table_name == "custom_schema.custom_embeddings"


def test_search_returns_empty_when_not_available():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    results = idx.search("edition-1", [0.1] * 384, k=10)
    assert results == []


def test_search_returns_empty_for_empty_vector():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    results = idx.search("edition-1", [], k=10)
    assert results == []


def test_embed_text_returns_empty():
    """Protocol method returns empty (embedding done externally)."""
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    assert idx.embed_text("diesel combustion") == []


def test_upsert_returns_zero_when_not_available():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    rec = EmbeddingRecord("ed1", "EF:1", [0.0] * 384, "test", "h1")
    count = idx.upsert_embeddings("ed1", [rec])
    assert count == 0


def test_delete_edition_returns_zero_when_not_available():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    assert idx.delete_edition("ed1") == 0


def test_count_embeddings_returns_zero_when_not_available():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    assert idx.count_embeddings("ed1") == 0


def test_get_stale_factors_returns_all_when_not_available():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    hashes = {"EF:1": "h1", "EF:2": "h2"}
    stale = idx.get_stale_factors("ed1", hashes)
    assert set(stale) == {"EF:1", "EF:2"}


def test_get_stale_factors_empty_hashes():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    assert idx.get_stale_factors("ed1", {}) == []


def test_close_safe_when_no_connection():
    cfg = PgVectorConfig(dsn="postgresql://localhost/test")
    idx = PgVectorSemanticIndex(cfg)
    idx.close()  # Should not raise
    assert idx._conn is None
    assert idx._initialized is False


# ---- NoopSemanticIndex compatibility ----

def test_noop_index_embed_text():
    idx = NoopSemanticIndex()
    assert idx.embed_text("diesel") == []


def test_noop_index_search():
    idx = NoopSemanticIndex()
    assert idx.search("ed1", [0.1] * 384, 10) == []


# ---- create_semantic_index factory ----

def test_create_semantic_index_no_dsn(monkeypatch):
    monkeypatch.delenv("GL_FACTORS_PG_DSN", raising=False)
    from greenlang.factors.matching.pgvector_index import create_semantic_index
    idx = create_semantic_index()
    # Falls back to NoopSemanticIndex
    assert isinstance(idx, NoopSemanticIndex)


def test_create_semantic_index_with_dsn_no_pg(monkeypatch):
    """If dsn provided but Postgres unreachable, returns index with is_available=False."""
    monkeypatch.setenv("GL_FACTORS_PG_DSN", "postgresql://localhost:5555/nonexistent")
    from greenlang.factors.matching.pgvector_index import create_semantic_index
    idx = create_semantic_index()
    # Will try to connect but fail gracefully
    assert not idx.is_available or isinstance(idx, NoopSemanticIndex)
