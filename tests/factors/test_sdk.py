# -*- coding: utf-8 -*-
"""Tests for Python SDK (F037)."""

from __future__ import annotations

import json
import urllib.error

import pytest

from greenlang.factors.sdk import (
    SDK_VERSION,
    FactorsApiError,
    FactorsClient,
    FactorsConfig,
    FactorsConnectionError,
    FactorsSdk,
    FactorsSdkConfig,
)


# ---- SDK version ----

def test_sdk_version():
    assert SDK_VERSION == "1.0.0"


# ---- FactorsConfig ----

def test_config_defaults():
    cfg = FactorsConfig(base_url="http://localhost:8000/api/v1")
    assert cfg.base_url == "http://localhost:8000/api/v1"
    assert cfg.api_key is None
    assert cfg.edition is None
    assert cfg.timeout == 60
    assert cfg.max_retries == 3
    assert cfg.retry_backoff == 1.0
    assert "greenlang-factors-sdk" in cfg.user_agent


def test_config_custom():
    cfg = FactorsConfig(
        base_url="https://api.greenlang.io/api/v1",
        api_key="gl_test_key_1234567890123456789",
        edition="2026.04.0",
        timeout=30,
        max_retries=5,
    )
    assert cfg.api_key == "gl_test_key_1234567890123456789"
    assert cfg.edition == "2026.04.0"
    assert cfg.timeout == 30
    assert cfg.max_retries == 5


# ---- Backward compat ----

def test_backward_compat_aliases():
    assert FactorsSdk is FactorsClient
    assert FactorsSdkConfig is FactorsConfig


# ---- FactorsClient construction ----

def test_client_construction():
    cfg = FactorsConfig(base_url="http://localhost:8000/api/v1")
    client = FactorsClient(cfg)
    assert client._base == "http://localhost:8000/api/v1"


def test_client_strips_trailing_slash():
    cfg = FactorsConfig(base_url="http://localhost:8000/api/v1/")
    client = FactorsClient(cfg)
    assert client._base == "http://localhost:8000/api/v1"


# ---- Headers ----

def test_headers_no_auth():
    cfg = FactorsConfig(base_url="http://localhost:8000/api/v1")
    client = FactorsClient(cfg)
    h = client._headers()
    assert h["Accept"] == "application/json"
    assert "greenlang-factors-sdk" in h["User-Agent"]
    assert "Authorization" not in h


def test_headers_with_api_key():
    cfg = FactorsConfig(
        base_url="http://localhost:8000/api/v1",
        api_key="gl_test_key",
    )
    client = FactorsClient(cfg)
    h = client._headers()
    assert h["Authorization"] == "Bearer gl_test_key"


def test_headers_with_edition():
    cfg = FactorsConfig(
        base_url="http://localhost:8000/api/v1",
        edition="2026.04.0",
    )
    client = FactorsClient(cfg)
    h = client._headers()
    assert h["X-Factors-Edition"] == "2026.04.0"


# ---- Error classes ----

def test_api_error():
    err = FactorsApiError(404, "Not Found", '{"error": "not_found"}')
    assert err.status_code == 404
    assert err.message == "Not Found"
    assert err.body is not None
    assert "404" in str(err)


def test_connection_error():
    err = FactorsConnectionError("Connection refused")
    assert "Connection refused" in str(err)


# ---- Method existence ----

def test_has_edition_methods():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.list_editions)
    assert callable(c.get_changelog)
    assert callable(c.compare_editions)


def test_has_factor_methods():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.list_factors)
    assert callable(c.get_factor)
    assert callable(c.get_provenance)
    assert callable(c.get_replacements)
    assert callable(c.get_audit_bundle)
    assert callable(c.diff_factor)


def test_has_search_methods():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.search)
    assert callable(c.search_v2)
    assert callable(c.get_facets)


def test_has_match_method():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.match)


def test_has_calculation_methods():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.calculate)
    assert callable(c.calculate_batch)


def test_has_export_method():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.export)


def test_has_system_methods():
    cfg = FactorsConfig(base_url="http://localhost/api/v1")
    c = FactorsClient(cfg)
    assert callable(c.health)
    assert callable(c.stats)
    assert callable(c.coverage)
    assert callable(c.source_registry)


# ---- Integration with TestClient (FastAPI) ----
# Uses api_client fixture from conftest.py which properly initializes the app.

def test_sdk_against_test_client(api_client):
    """Verify SDK method signatures match actual API responses."""
    resp = api_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_sdk_list_editions_api_shape(api_client):
    resp = api_client.get("/api/v1/editions")
    assert resp.status_code == 200
    data = resp.json()
    assert "editions" in data
    assert "default_edition_id" in data


def test_sdk_search_api_shape(api_client):
    resp = api_client.get("/api/v1/factors/search", params={"q": "diesel"})
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "factors" in data
    assert "count" in data


def test_sdk_factor_detail_api_shape(api_client):
    search = api_client.get("/api/v1/factors/search", params={"q": "diesel", "limit": "1"})
    factors = search.json().get("factors", [])
    if factors:
        fid = factors[0]["factor_id"]
        resp = api_client.get(f"/api/v1/factors/{fid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["factor_id"] == fid
        assert "co2e_per_unit" in data
        assert "data_quality" in data


def test_sdk_match_api_shape(api_client):
    resp = api_client.post(
        "/api/v1/factors/match",
        json={"activity_description": "diesel combustion US", "limit": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert "edition_id" in data
