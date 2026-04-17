# -*- coding: utf-8 -*-
"""Tests for Postgres factor catalog repository (F031).

Since these tests can't connect to a real Postgres instance in CI,
they test the module structure, SQL generation helpers, and config.
Integration tests require GL_FACTORS_PG_DSN to be set.
"""

from __future__ import annotations

import pytest

from greenlang.factors.catalog_repository_pg import (
    PgPoolConfig,
    PostgresFactorCatalogRepository,
    _status_visibility_sql,
)
from greenlang.factors.catalog_repository import (
    EditionRow,
    FactorCatalogRepository,
)


# ---- PgPoolConfig ----

def test_pool_config_defaults():
    cfg = PgPoolConfig(dsn="postgresql://localhost/test")
    assert cfg.min_size == 2
    assert cfg.max_size == 20
    assert cfg.max_idle == 300.0
    assert cfg.max_lifetime == 3600.0


def test_pool_config_custom():
    cfg = PgPoolConfig(dsn="postgresql://host/db", min_size=5, max_size=50)
    assert cfg.min_size == 5
    assert cfg.max_size == 50


# ---- _status_visibility_sql ----

def test_visibility_sql_community():
    sql = _status_visibility_sql(include_preview=False, include_connector=False)
    assert "'certified'" in sql
    assert "'preview'" not in sql
    assert "'connector_only'" not in sql
    assert "'deprecated'" in sql  # excluded


def test_visibility_sql_pro():
    sql = _status_visibility_sql(include_preview=True, include_connector=False)
    assert "'certified'" in sql
    assert "'preview'" in sql
    assert "'connector_only'" not in sql


def test_visibility_sql_enterprise():
    sql = _status_visibility_sql(include_preview=True, include_connector=True)
    assert "'certified'" in sql
    assert "'preview'" in sql
    assert "'connector_only'" in sql


# ---- PostgresFactorCatalogRepository class ----

def test_inherits_abc():
    assert issubclass(PostgresFactorCatalogRepository, FactorCatalogRepository)


def test_has_all_abstract_methods():
    """Verify Postgres repo implements all abstract methods from the base."""
    abc_methods = {
        "list_editions",
        "get_default_edition_id",
        "resolve_edition",
        "get_changelog",
        "get_manifest_dict",
        "list_factors",
        "get_factor",
        "search_factors",
        "search_facets",
        "coverage_stats",
        "list_factor_summaries",
    }
    for method in abc_methods:
        assert hasattr(PostgresFactorCatalogRepository, method), f"Missing method: {method}"
        func = getattr(PostgresFactorCatalogRepository, method)
        assert callable(func), f"Not callable: {method}"


def test_constructor_stores_config():
    cfg = PgPoolConfig(dsn="postgresql://localhost/test")
    repo = PostgresFactorCatalogRepository(cfg)
    assert repo._config is cfg
    assert repo._pool is None


def test_has_write_methods():
    """Verify write methods exist."""
    assert hasattr(PostgresFactorCatalogRepository, "upsert_edition")
    assert hasattr(PostgresFactorCatalogRepository, "insert_factors")


def test_has_full_text_search():
    """Verify search uses tsvector-based queries (check source)."""
    import inspect
    src = inspect.getsource(PostgresFactorCatalogRepository.search_factors)
    assert "plainto_tsquery" in src
    assert "ts_rank" in src
    assert "search_tsv" in src


def test_schema_namespace():
    """Verify all queries use factors_catalog schema prefix."""
    import inspect
    src = inspect.getsource(PostgresFactorCatalogRepository)
    # Count occurrences of the schema prefix
    assert src.count("factors_catalog.") >= 10


# ---- EditionRow reuse ----

def test_edition_row_from_catalog():
    row = EditionRow(
        edition_id="2026.04.0",
        status="stable",
        label="April 2026",
        manifest_hash="abc123",
        changelog_json='["Added EPA factors"]',
    )
    assert row.edition_id == "2026.04.0"
    assert row.status == "stable"
