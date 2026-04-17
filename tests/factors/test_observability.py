# -*- coding: utf-8 -*-
"""Tests for Factors observability: Prometheus metrics + health checks (F070-F073)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.observability.health import (
    ComponentHealth,
    ComponentStatus,
    HealthStatus,
    _check_cache,
    _check_database,
    _check_edition,
    get_health_status,
)
from greenlang.factors.observability.prometheus_exporter import (
    FactorsMetrics,
    get_factors_metrics,
)


# ── Prometheus Metrics ──────────────────────────────────────────────


class TestFactorsMetrics:
    """FactorsMetrics fallback mode (no prometheus_client)."""

    def test_singleton(self):
        m1 = get_factors_metrics()
        m2 = get_factors_metrics()
        assert m1 is m2

    def test_record_api_request_fallback(self):
        m = FactorsMetrics()
        m.record_api_request("GET", "/factors/search", 200, 0.05)
        assert m.fallback_store.counters.get("api:GET:/factors/search:200", 0) >= 1

    def test_track_api_call_success(self):
        m = FactorsMetrics()
        with m.track_api_call("POST", "/factors/match"):
            pass
        assert m.fallback_store.counters.get("api:POST:/factors/match:200", 0) >= 1

    def test_track_api_call_error(self):
        m = FactorsMetrics()
        with pytest.raises(ValueError):
            with m.track_api_call("GET", "/factors/detail"):
                raise ValueError("boom")
        assert m.fallback_store.counters.get("api:GET:/factors/detail:500", 0) >= 1

    def test_record_search_results(self):
        m = FactorsMetrics()
        m.record_search_results(42)
        assert 42 in m.fallback_store.histograms.get("search_results", [])

    def test_record_match_score(self):
        m = FactorsMetrics()
        m.record_match_score(0.87)
        assert 0.87 in m.fallback_store.histograms.get("match_score", [])

    def test_set_edition_factor_count(self):
        m = FactorsMetrics()
        m.set_edition_factor_count("ed-2026-04", "stable", 102000)
        assert m.fallback_store.gauges["edition:ed-2026-04:stable"] == 102000

    def test_record_ingestion(self):
        m = FactorsMetrics()
        m.record_ingestion("defra_2025", 5000, "ok")
        assert m.fallback_store.counters["ingest:defra_2025:ok"] == 5000

    def test_record_watch_change(self):
        m = FactorsMetrics()
        m.record_watch_change("epa_ghg", "updated")
        assert m.fallback_store.counters["watch:epa_ghg:updated"] == 1

    def test_record_qa_failure(self):
        m = FactorsMetrics()
        m.record_qa_failure("unit_consistency")
        assert m.fallback_store.counters["qa:unit_consistency"] == 1


# ── Component Health Checks ─────────────────────────────────────────


class TestCheckDatabase:
    def test_healthy_database(self):
        repo = MagicMock()
        repo.coverage_stats.return_value = {"total_factors": 100000}
        result = _check_database(repo)
        assert result.status == ComponentStatus.HEALTHY
        assert result.details["total_factors"] == 100000

    def test_unavailable_database(self):
        repo = MagicMock()
        repo.coverage_stats.side_effect = ConnectionError("DB down")
        result = _check_database(repo)
        assert result.status == ComponentStatus.UNAVAILABLE
        assert "DB down" in result.message

    def test_no_coverage_stats_method(self):
        repo = object()  # no coverage_stats attribute
        result = _check_database(repo)
        assert result.status == ComponentStatus.HEALTHY


class TestCheckCache:
    @patch("greenlang.factors.observability.health._get_redis", side_effect=ImportError)
    def test_import_error_degrades(self, _mock):
        # If the import itself fails, we catch the exception
        result = _check_cache()
        assert result.status == ComponentStatus.DEGRADED

    def test_no_redis_configured(self):
        with patch("greenlang.factors.observability.health._get_redis", return_value=None):
            result = _check_cache()
        assert result.status == ComponentStatus.DEGRADED
        assert "not configured" in result.message

    def test_redis_healthy(self):
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        with patch("greenlang.factors.observability.health._get_redis", return_value=mock_redis):
            result = _check_cache()
        assert result.status == ComponentStatus.HEALTHY


class TestCheckEdition:
    def test_edition_found(self):
        repo = MagicMock()
        repo.resolve_edition.return_value = "ed-2026-04"
        result = _check_edition(repo)
        assert result.status == ComponentStatus.HEALTHY
        assert result.details["edition_id"] == "ed-2026-04"

    def test_no_edition(self):
        repo = MagicMock()
        repo.resolve_edition.return_value = None
        result = _check_edition(repo)
        assert result.status == ComponentStatus.DEGRADED

    def test_edition_error(self):
        repo = MagicMock()
        repo.resolve_edition.side_effect = RuntimeError("timeout")
        result = _check_edition(repo)
        assert result.status == ComponentStatus.DEGRADED


# ── Full Health Status ───────────────────────────────────────────────


class TestGetHealthStatus:
    def _mock_repo(self, db_ok=True, edition_ok=True):
        repo = MagicMock()
        if db_ok:
            repo.coverage_stats.return_value = {"total_factors": 50000, "certified": 42000}
        else:
            repo.coverage_stats.side_effect = ConnectionError("down")
        if edition_ok:
            repo.resolve_edition.return_value = "ed-2026-04"
        else:
            repo.resolve_edition.return_value = None
        return repo

    def test_healthy(self):
        repo = self._mock_repo()
        with patch("greenlang.factors.observability.health._check_cache") as mock_cache:
            mock_cache.return_value = ComponentHealth(
                name="cache", status=ComponentStatus.HEALTHY, message="ok"
            )
            status = get_health_status(repo)
        assert status.status == "healthy"
        assert status.is_healthy
        assert status.http_status == 200
        assert status.factor_count == 50000
        assert status.certified_count == 42000
        assert status.edition_id == "ed-2026-04"

    def test_degraded_when_cache_down(self):
        repo = self._mock_repo()
        with patch("greenlang.factors.observability.health._check_cache") as mock_cache:
            mock_cache.return_value = ComponentHealth(
                name="cache", status=ComponentStatus.DEGRADED, message="no redis"
            )
            status = get_health_status(repo)
        assert status.status == "degraded"
        assert status.http_status == 200  # degraded still returns 200

    def test_unavailable_when_db_down(self):
        repo = self._mock_repo(db_ok=False)
        with patch("greenlang.factors.observability.health._check_cache") as mock_cache:
            mock_cache.return_value = ComponentHealth(
                name="cache", status=ComponentStatus.HEALTHY, message="ok"
            )
            status = get_health_status(repo)
        assert status.status == "unavailable"
        assert status.http_status == 503

    def test_to_dict_structure(self):
        repo = self._mock_repo()
        with patch("greenlang.factors.observability.health._check_cache") as mock_cache:
            mock_cache.return_value = ComponentHealth(
                name="cache", status=ComponentStatus.HEALTHY, message="ok"
            )
            status = get_health_status(repo)
        d = status.to_dict()
        assert "status" in d
        assert "version" in d
        assert "timestamp" in d
        assert "components" in d
        assert "database" in d["components"]
        assert "cache" in d["components"]
        assert "edition" in d["components"]
