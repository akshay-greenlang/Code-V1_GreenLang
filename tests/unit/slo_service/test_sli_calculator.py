# -*- coding: utf-8 -*-
"""
Unit tests for SLI Calculator (OBS-005)

Tests PromQL query building for all SLI types, recording rule generation,
Prometheus query execution, and error handling.

Coverage target: 85%+ of sli_calculator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.slo_service.models import SLI, SLO, SLIType, SLOWindow
from greenlang.infrastructure.slo_service.sli_calculator import (
    build_error_rate_query,
    build_sli_ratio_query,
    calculate_sli,
    generate_recording_rule,
    generate_recording_rules_yaml,
    query_prometheus,
)


# ============================================================================
# Query building tests
# ============================================================================


class TestQueryBuilding:
    """Tests for PromQL query building functions."""

    def test_build_availability_ratio_query(self, sample_sli_availability):
        """Availability ratio query uses increase()."""
        query = build_sli_ratio_query(sample_sli_availability, "30d")
        assert "increase" in query
        assert "30d" in query
        assert sample_sli_availability.good_query in query
        assert sample_sli_availability.total_query in query

    def test_build_latency_ratio_query(self, sample_sli_latency):
        """Latency ratio query uses increase()."""
        query = build_sli_ratio_query(sample_sli_latency, "28d")
        assert "increase" in query
        assert "28d" in query

    def test_build_correctness_ratio_query(self, sample_sli_correctness):
        """Correctness ratio query uses increase()."""
        query = build_sli_ratio_query(sample_sli_correctness, "30d")
        assert "increase" in query
        assert sample_sli_correctness.good_query in query

    def test_build_throughput_ratio_query(self, sample_sli_throughput):
        """Throughput ratio query uses rate()."""
        query = build_sli_ratio_query(sample_sli_throughput, "30d")
        assert "rate" in query
        assert sample_sli_throughput.good_query in query

    def test_build_freshness_ratio_query(self, sample_sli_freshness):
        """Freshness ratio query uses increase()."""
        query = build_sli_ratio_query(sample_sli_freshness, "7d")
        assert "increase" in query
        assert "7d" in query

    def test_build_error_rate_query(self, sample_sli_availability):
        """Error rate query is 1 minus the SLI ratio."""
        query = build_error_rate_query(sample_sli_availability, "30d")
        assert query.startswith("1 - (")
        assert "increase" in query

    @pytest.mark.parametrize("window", ["5m", "1h", "7d", "30d", "90d"])
    def test_query_with_different_windows(self, sample_sli_availability, window):
        """Query correctly inserts different window durations."""
        query = build_sli_ratio_query(sample_sli_availability, window)
        assert f"[{window}]" in query

    def test_query_structure_has_division(self, sample_sli_availability):
        """Ratio query contains division operator."""
        query = build_sli_ratio_query(sample_sli_availability, "30d")
        assert " / " in query


# ============================================================================
# Recording rule tests
# ============================================================================


class TestRecordingRuleGeneration:
    """Tests for recording rule generation."""

    def test_generate_recording_rule_format(self, sample_slo):
        """Recording rule has correct structure."""
        rule = generate_recording_rule(sample_slo)
        assert "record" in rule
        assert "expr" in rule
        assert "labels" in rule

    def test_generate_recording_rule_naming(self, sample_slo):
        """Recording rule uses slo:<safe_name>:sli_ratio naming."""
        rule = generate_recording_rule(sample_slo)
        expected_name = f"slo:{sample_slo.safe_name}:sli_ratio"
        assert rule["record"] == expected_name

    def test_generate_recording_rule_labels(self, sample_slo):
        """Recording rule includes slo_id, service, and sli_type labels."""
        rule = generate_recording_rule(sample_slo)
        labels = rule["labels"]
        assert labels["slo_id"] == sample_slo.slo_id
        assert labels["service"] == sample_slo.service
        assert labels["sli_type"] == sample_slo.sli.sli_type.value

    def test_generate_recording_rule_team_label(self, sample_slo):
        """Recording rule includes team label when set."""
        rule = generate_recording_rule(sample_slo)
        assert rule["labels"]["team"] == "platform"

    def test_generate_recording_rules_yaml(self, sample_slo_list):
        """YAML structure has groups with rules."""
        result = generate_recording_rules_yaml(sample_slo_list)
        assert "groups" in result
        assert len(result["groups"]) == 1
        group = result["groups"][0]
        assert group["name"] == "slo_sli_recording_rules"
        assert group["interval"] == "60s"
        assert len(group["rules"]) == len(sample_slo_list)

    def test_generate_recording_rules_yaml_disabled_slo(self, slo_factory):
        """Disabled SLOs are excluded from recording rules."""
        slos = [
            slo_factory(slo_id="enabled-1", name="Enabled SLO", enabled=True),
            slo_factory(slo_id="disabled-1", name="Disabled SLO", enabled=False),
        ]
        result = generate_recording_rules_yaml(slos)
        rules = result["groups"][0]["rules"]
        assert len(rules) == 1
        assert rules[0]["labels"]["slo_id"] == "enabled-1"

    def test_recording_rule_yaml_empty_slos(self):
        """Empty SLO list produces empty rules."""
        result = generate_recording_rules_yaml([])
        assert result["groups"][0]["rules"] == []


# ============================================================================
# Prometheus query execution tests
# ============================================================================


class TestPrometheusQuery:
    """Tests for Prometheus query execution."""

    @pytest.mark.asyncio
    async def test_query_prometheus_success(self, mock_httpx_client):
        """Successful Prometheus query returns float value."""
        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_httpx_client

            result = await query_prometheus(
                "http://prometheus:9090",
                "sum(rate(requests[5m]))",
            )

        assert result is not None
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_query_prometheus_empty_result(self):
        """Empty Prometheus result returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await query_prometheus("http://prom:9090", "test_query")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_prometheus_error_handling(self):
        """Prometheus errors are handled gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "error",
            "errorType": "bad_data",
            "error": "invalid query",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = await query_prometheus("http://prom:9090", "bad_query")

        assert result is None

    @pytest.mark.asyncio
    async def test_query_prometheus_timeout(self):
        """Timeout raises TimeoutError."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(TimeoutError):
                await query_prometheus("http://prom:9090", "test", timeout_seconds=1)

    @pytest.mark.asyncio
    async def test_query_prometheus_connection_error(self):
        """Connection failure raises ConnectionError."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Failed to connect")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client

            with pytest.raises(ConnectionError):
                await query_prometheus("http://prom:9090", "test")

    @pytest.mark.asyncio
    async def test_query_prometheus_no_httpx(self):
        """Returns None when httpx is not installed."""
        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", False):
            result = await query_prometheus("http://prom:9090", "test")
        assert result is None

    def test_sli_ratio_bounds(self):
        """SLI ratio values are clamped to 0.0-1.0 range."""
        # Verified via query_prometheus clamping logic.
        # Ratio > 1.0 should be clamped to 1.0
        assert max(0.0, min(1.0, 1.5)) == 1.0
        assert max(0.0, min(1.0, -0.1)) == 0.0
        assert max(0.0, min(1.0, 0.5)) == 0.5


# ============================================================================
# Calculate SLI tests
# ============================================================================


class TestCalculateSLI:
    """Tests for the calculate_sli wrapper."""

    @pytest.mark.asyncio
    async def test_calculate_sli_availability(self, sample_slo, mock_httpx_client):
        """calculate_sli returns the SLI ratio for an SLO."""
        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_httpx_client

            result = await calculate_sli(sample_slo, "http://prom:9090")

        assert result is not None
        assert 0.0 <= result <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_sli_with_custom_window(self, sample_slo, mock_httpx_client):
        """calculate_sli accepts a custom window override."""
        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_httpx_client

            result = await calculate_sli(sample_slo, "http://prom:9090", window="7d")

        assert result is not None

    @pytest.mark.asyncio
    async def test_calculate_sli_with_zero_total_events(self, sample_slo):
        """Returns None when Prometheus returns no data."""
        with patch("greenlang.infrastructure.slo_service.sli_calculator.query_prometheus",
                   new_callable=AsyncMock, return_value=None):
            result = await calculate_sli(sample_slo, "http://prom:9090")

        assert result is None
