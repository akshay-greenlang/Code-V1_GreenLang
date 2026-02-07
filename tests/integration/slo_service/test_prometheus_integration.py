# -*- coding: utf-8 -*-
"""
Prometheus Integration Tests for SLO Service (OBS-005)

Tests SLI query execution, recording rule evaluation, PromQL syntax
validation, and Prometheus connection handling with mocked HTTP.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.slo_service.models import SLI, SLO, SLIType
from greenlang.infrastructure.slo_service.sli_calculator import (
    build_sli_ratio_query,
    calculate_sli,
    query_prometheus,
)


def _run_async(coro):
    """Run an async coroutine synchronously.

    Temporarily restores the real socket module so that asyncio can create
    an event loop (which requires ``socket.socketpair()``).  The parent
    integration conftest blocks all socket access, so we restore the original
    socket class captured at import time by our SLO conftest.
    """
    import socket as _sock_mod
    from tests.integration.slo_service.conftest import (
        _ORIGINAL_SOCKET,
        _ORIGINAL_CREATE_CONNECTION,
    )

    saved = _sock_mod.socket
    saved_cc = _sock_mod.create_connection
    _sock_mod.socket = _ORIGINAL_SOCKET
    _sock_mod.create_connection = _ORIGINAL_CREATE_CONNECTION
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    finally:
        _sock_mod.socket = saved
        _sock_mod.create_connection = saved_cc


@pytest.mark.integration
class TestPrometheusIntegration:
    """Prometheus query integration tests (mocked HTTP)."""

    @pytest.fixture
    def sample_slo(self):
        """Create a test SLO."""
        sli = SLI(
            name="api_avail",
            sli_type=SLIType.AVAILABILITY,
            good_query='http_requests_total{code!~"5.."}',
            total_query="http_requests_total",
        )
        return SLO(
            slo_id="prom-test-slo",
            name="Prom Test SLO",
            service="api-gateway",
            sli=sli,
            target=99.9,
        )

    def test_query_prometheus_for_sli(self, sample_slo):
        """Query Prometheus for SLI value."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [1707307200, "0.9995"]}],
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            result = _run_async(calculate_sli(sample_slo, "http://prom:9090"))

        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_promql_syntax_validation(self, sample_slo):
        """Generated PromQL has valid structure."""
        query = build_sli_ratio_query(sample_slo.sli, "30d")
        # Must have balanced parentheses
        assert query.count("(") == query.count(")")
        # Must have balanced brackets
        assert query.count("[") == query.count("]")
        # Must contain the window
        assert "[30d]" in query

    def test_prometheus_connection_retry(self):
        """Connection errors propagate as ConnectionError."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Failed to connect")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            with pytest.raises(ConnectionError):
                _run_async(query_prometheus("http://prom:9090", "test"))

    def test_prometheus_timeout_handling(self):
        """Timeout errors propagate as TimeoutError."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            with pytest.raises(TimeoutError):
                _run_async(query_prometheus("http://prom:9090", "test", 5))

    def test_multiple_window_queries(self, sample_slo):
        """SLI can be queried with different windows."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {}, "value": [0, "0.9990"]}],
            },
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE", True), \
             patch("greenlang.infrastructure.slo_service.sli_calculator.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client

            for window in ["5m", "1h", "7d", "30d"]:
                result = _run_async(calculate_sli(
                    sample_slo, "http://prom:9090", window=window,
                ))
                assert result is not None

    def test_recording_rule_evaluation(self, sample_slo):
        """Recording rules produce valid PromQL expressions."""
        from greenlang.infrastructure.slo_service.sli_calculator import generate_recording_rule
        rule = generate_recording_rule(sample_slo)
        expr = rule["expr"]
        # Expression should be non-empty and contain query components
        assert len(expr) > 0
        assert "increase" in expr or "rate" in expr

    def test_burn_rate_calculation_from_prometheus(self, sample_slo):
        """Burn rate can be calculated from Prometheus-returned SLI."""
        from greenlang.infrastructure.slo_service.burn_rate import calculate_burn_rate

        # Simulate SLI of 0.998 (error rate 0.002)
        error_rate = 1.0 - 0.998
        burn_rate = calculate_burn_rate(error_rate, sample_slo.error_budget_fraction)
        assert burn_rate == pytest.approx(2.0)  # 0.002 / 0.001

    def test_error_budget_from_prometheus(self, sample_slo):
        """Error budget computed from Prometheus SLI data."""
        from greenlang.infrastructure.slo_service.error_budget import calculate_error_budget

        budget = calculate_error_budget(sample_slo, current_sli=0.9995)
        assert budget.slo_id == sample_slo.slo_id
        assert budget.consumed_percent == pytest.approx(50.0, rel=0.01)
