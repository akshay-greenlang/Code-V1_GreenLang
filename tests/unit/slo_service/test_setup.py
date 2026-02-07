# -*- coding: utf-8 -*-
"""
Unit tests for SLO Service Setup / SLOService Facade (OBS-005)

Tests the SLOService facade class covering CRUD delegation, error budget
management, burn rate checks, rule/dashboard generation, compliance
reporting, evaluation loop, health checks, and the configure/get helpers.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.slo_service.config import SLOServiceConfig
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    BurnRateWindow,
    ErrorBudget,
    SLI,
    SLIType,
    SLO,
    SLOReport,
    SLOWindow,
)
from greenlang.infrastructure.slo_service.setup import (
    SLOService,
    configure_slo_service,
    get_slo_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config():
    """Create a test SLOServiceConfig."""
    return SLOServiceConfig(
        service_name="test-slo",
        environment="test",
        prometheus_url="http://localhost:9090",
        redis_url="redis://localhost:6379/15",
        evaluation_interval_seconds=5,
        budget_exhausted_action="alert_only",
        alerting_bridge_enabled=False,
    )


@pytest.fixture
def slo_service(test_config):
    """Create a SLOService instance."""
    return SLOService(config=test_config)


@pytest.fixture
def sample_sli():
    """Create a sample SLI."""
    return SLI(
        name="api_avail",
        sli_type=SLIType.AVAILABILITY,
        good_query='http_requests_total{code!~"5.."}',
        total_query="http_requests_total",
    )


@pytest.fixture
def sample_slo_data(sample_sli):
    """Create sample SLO creation data."""
    return {
        "slo_id": "test-slo-1",
        "name": "Test SLO 1",
        "service": "api-gateway",
        "sli": sample_sli.to_dict(),
        "target": 99.9,
        "team": "platform",
    }


@pytest.fixture
def populated_service(slo_service, sample_sli):
    """SLO service with pre-populated SLOs."""
    for i in range(3):
        slo = SLO(
            slo_id=f"test-slo-{i}",
            name=f"Test SLO {i}",
            service=f"service-{i % 2}",
            sli=sample_sli,
            target=99.9,
            team=f"team-{i % 2}",
        )
        slo_service.manager.create(slo)
    return slo_service


@pytest.fixture
def service_with_budgets(populated_service):
    """SLO service with budget history."""
    from greenlang.infrastructure.slo_service.error_budget import (
        calculate_error_budget,
    )

    for slo in populated_service.manager.list_all():
        budget = calculate_error_budget(slo, current_sli=0.9995)
        if slo.slo_id not in populated_service._budget_store:
            populated_service._budget_store[slo.slo_id] = []
        populated_service._budget_store[slo.slo_id].append(budget)

    return populated_service


# ---------------------------------------------------------------------------
# SLOService initialization
# ---------------------------------------------------------------------------


class TestSLOServiceInit:
    """Tests for SLOService initialization."""

    def test_init_with_config(self, test_config):
        """Service initializes with provided config."""
        svc = SLOService(config=test_config)
        assert svc.config is test_config
        assert svc.manager is not None
        assert svc.bridge is not None
        assert svc._running is False
        assert svc._evaluation_task is None

    def test_init_defaults(self):
        """Service initializes with default config from env."""
        svc = SLOService()
        assert svc.config is not None
        assert svc.manager is not None

    def test_init_budget_store_empty(self, slo_service):
        """Budget store starts empty."""
        assert slo_service._budget_store == {}

    def test_init_report_store_empty(self, slo_service):
        """Report store starts empty."""
        assert slo_service._report_store == {}


# ---------------------------------------------------------------------------
# SLO CRUD
# ---------------------------------------------------------------------------


class TestSLOServiceCRUD:
    """Tests for SLO CRUD operations."""

    def test_create_slo(self, slo_service, sample_slo_data):
        """Create an SLO through the service."""
        result = slo_service.create_slo(sample_slo_data)
        assert result.slo_id == "test-slo-1"
        assert result.name == "Test SLO 1"
        assert result.target == 99.9

    def test_create_slo_with_window(self, slo_service, sample_slo_data):
        """Create an SLO with a custom window."""
        sample_slo_data["window"] = "7d"
        result = slo_service.create_slo(sample_slo_data)
        assert result.window == SLOWindow.SEVEN_DAYS

    def test_get_slo(self, populated_service):
        """Get an SLO by ID."""
        result = populated_service.get_slo("test-slo-0")
        assert result is not None
        assert result.slo_id == "test-slo-0"

    def test_get_slo_not_found(self, populated_service):
        """Get nonexistent SLO returns None."""
        result = populated_service.get_slo("nonexistent")
        assert result is None

    def test_list_slos(self, populated_service):
        """List all SLOs."""
        result = populated_service.list_slos()
        assert len(result) == 3

    def test_list_slos_by_service(self, populated_service):
        """List SLOs filtered by service."""
        result = populated_service.list_slos(service="service-0")
        assert len(result) == 2
        assert all(s.service == "service-0" for s in result)

    def test_list_slos_by_team(self, populated_service):
        """List SLOs filtered by team."""
        result = populated_service.list_slos(team="team-0")
        assert len(result) == 2

    def test_list_slos_enabled_filter(self, populated_service):
        """List SLOs filtered by enabled status."""
        result = populated_service.list_slos(enabled=True)
        assert len(result) == 3  # All enabled by default

    def test_list_slos_include_deleted(self, populated_service):
        """List SLOs including deleted."""
        populated_service.delete_slo("test-slo-0")
        visible = populated_service.list_slos()
        assert len(visible) == 2

        all_slos = populated_service.list_slos(include_deleted=True)
        assert len(all_slos) == 3

    def test_update_slo(self, populated_service):
        """Update an SLO."""
        result = populated_service.update_slo("test-slo-0", {"description": "Updated"})
        assert result.description == "Updated"
        assert result.version == 2

    def test_update_slo_not_found(self, populated_service):
        """Update nonexistent SLO raises KeyError."""
        with pytest.raises(KeyError):
            populated_service.update_slo("nonexistent", {"description": "x"})

    def test_delete_slo(self, populated_service):
        """Delete an SLO."""
        result = populated_service.delete_slo("test-slo-0")
        assert result is True

    def test_delete_slo_not_found(self, populated_service):
        """Delete nonexistent SLO returns False."""
        result = populated_service.delete_slo("nonexistent")
        assert result is False

    def test_get_slo_history(self, populated_service):
        """Get SLO version history."""
        populated_service.update_slo("test-slo-0", {"description": "v2"})
        populated_service.update_slo("test-slo-0", {"description": "v3"})
        history = populated_service.get_slo_history("test-slo-0")
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Error budget
# ---------------------------------------------------------------------------


class TestSLOServiceBudget:
    """Tests for error budget operations."""

    def test_get_error_budget_no_history(self, populated_service):
        """Get budget with no history returns fresh budget."""
        result = populated_service.get_error_budget("test-slo-0")
        assert result["slo_id"] == "test-slo-0"
        assert result["consumed_percent"] == 0.0
        assert result["status"] == "healthy"

    def test_get_error_budget_with_history(self, service_with_budgets):
        """Get budget with history returns latest."""
        result = service_with_budgets.get_error_budget("test-slo-0")
        assert result["slo_id"] == "test-slo-0"
        assert result["consumed_percent"] > 0

    def test_get_error_budget_not_found(self, populated_service):
        """Get budget for nonexistent SLO raises KeyError."""
        with pytest.raises(KeyError):
            populated_service.get_error_budget("nonexistent")

    def test_get_budget_history_empty(self, populated_service):
        """Budget history is empty when no evaluations done."""
        result = populated_service.get_budget_history("test-slo-0")
        assert result == []

    def test_get_budget_history_with_data(self, service_with_budgets):
        """Budget history returns stored snapshots."""
        result = service_with_budgets.get_budget_history("test-slo-0", days=30)
        assert len(result) >= 1

    def test_forecast_budget_no_history(self, populated_service):
        """Forecast returns no data message when no history."""
        result = populated_service.forecast_budget("test-slo-0")
        assert result["slo_id"] == "test-slo-0"
        assert result["exhaustion_forecast"] is None
        assert result["message"] == "No budget data available yet"

    def test_forecast_budget_with_history(self, service_with_budgets):
        """Forecast returns consumption rate when history exists."""
        result = service_with_budgets.forecast_budget("test-slo-0")
        assert result["slo_id"] == "test-slo-0"
        assert "consumption_rate_per_hour" in result
        assert "current_consumed_percent" in result

    def test_forecast_budget_not_found(self, populated_service):
        """Forecast for nonexistent SLO raises KeyError."""
        with pytest.raises(KeyError):
            populated_service.forecast_budget("nonexistent")

    def test_check_budget_policy_no_history(self, populated_service):
        """Policy check with no history returns healthy status."""
        result = populated_service.check_budget_policy("test-slo-0")
        assert result["budget_status"] == "healthy"
        assert result["action_required"] is False

    def test_check_budget_policy_with_history(self, service_with_budgets):
        """Policy check with history returns policy result."""
        result = service_with_budgets.check_budget_policy("test-slo-0")
        assert "slo_id" in result or "budget_status" in result

    def test_check_budget_policy_not_found(self, populated_service):
        """Policy check for nonexistent SLO raises KeyError."""
        with pytest.raises(KeyError):
            populated_service.check_budget_policy("nonexistent")


# ---------------------------------------------------------------------------
# Burn rate
# ---------------------------------------------------------------------------


class TestSLOServiceBurnRate:
    """Tests for burn rate operations."""

    def test_get_burn_rates(self, populated_service):
        """Get burn rates for all windows."""
        result = populated_service.get_burn_rates("test-slo-0")
        assert result["slo_id"] == "test-slo-0"
        assert "fast" in result["burn_rates"]
        assert "medium" in result["burn_rates"]
        assert "slow" in result["burn_rates"]

    def test_get_burn_rates_specific_window(self, populated_service):
        """Get burn rate for a specific window."""
        result = populated_service.get_burn_rates("test-slo-0", window="fast")
        assert "fast" in result["burn_rates"]
        assert len(result["burn_rates"]) == 1

    def test_get_burn_rates_not_found(self, populated_service):
        """Burn rates for nonexistent SLO raises KeyError."""
        with pytest.raises(KeyError):
            populated_service.get_burn_rates("nonexistent")

    def test_burn_rate_includes_threshold(self, populated_service):
        """Burn rate response includes threshold."""
        result = populated_service.get_burn_rates("test-slo-0")
        fast = result["burn_rates"]["fast"]
        assert fast["threshold"] == 14.4
        assert fast["severity"] == "critical"

    def test_check_burn_rate_alerts(self, populated_service):
        """Check burn rate alerts returns alert status."""
        result = populated_service.check_burn_rate_alerts("test-slo-0")
        assert result["slo_id"] == "test-slo-0"
        assert "alerts" in result
        assert result["any_firing"] is False

    def test_check_burn_rate_alerts_not_found(self, populated_service):
        """Check alerts for nonexistent SLO raises KeyError."""
        with pytest.raises(KeyError):
            populated_service.check_burn_rate_alerts("nonexistent")


# ---------------------------------------------------------------------------
# Rule / Dashboard generation
# ---------------------------------------------------------------------------


class TestSLOServiceGeneration:
    """Tests for rule and dashboard generation."""

    def test_generate_recording_rules(self, populated_service, tmp_path):
        """Generate recording rules to a temp path."""
        output = str(tmp_path / "recording_rules.yaml")
        result = populated_service.generate_recording_rules(output_path=output)
        assert result["status"] == "generated"
        assert result["total_rules"] > 0
        assert result["slo_count"] == 3

    def test_generate_alert_rules(self, populated_service, tmp_path):
        """Generate alert rules to a temp path."""
        output = str(tmp_path / "alert_rules.yaml")
        result = populated_service.generate_alert_rules(output_path=output)
        assert result["status"] == "generated"
        assert result["total_rules"] > 0

    def test_generate_dashboards(self, populated_service, tmp_path):
        """Generate dashboards to a temp directory."""
        result = populated_service.generate_dashboards(output_dir=str(tmp_path))
        assert result["status"] == "generated"
        assert result["dashboard_count"] > 0

    def test_generate_recording_rules_error_handling(self, slo_service):
        """Recording rule generation handles errors gracefully."""
        # No SLOs and invalid path
        result = slo_service.generate_recording_rules(
            output_path="/nonexistent/deeply/nested/rules.yaml"
        )
        # With no SLOs, it should still try to write
        assert result["status"] in ("generated", "error")

    def test_generate_alert_rules_error_handling(self, slo_service):
        """Alert rule generation handles errors gracefully."""
        result = slo_service.generate_alert_rules(
            output_path="/nonexistent/deeply/nested/alerts.yaml"
        )
        assert result["status"] in ("generated", "error")

    def test_generate_dashboards_error_handling(self, slo_service):
        """Dashboard generation handles errors gracefully."""
        result = slo_service.generate_dashboards(
            output_dir="/nonexistent/deeply/nested/dashboards"
        )
        assert result["status"] in ("generated", "error")


# ---------------------------------------------------------------------------
# Compliance reporting
# ---------------------------------------------------------------------------


class TestSLOServiceReporting:
    """Tests for compliance reporting."""

    def test_generate_weekly_report(self, service_with_budgets):
        """Generate a weekly compliance report."""
        result = service_with_budgets.generate_weekly_report()
        assert result["report_type"] == "weekly"
        assert result["total_slos"] == 3

    def test_generate_monthly_report(self, service_with_budgets):
        """Generate a monthly compliance report."""
        result = service_with_budgets.generate_monthly_report(month=1, year=2026)
        assert result["report_type"] == "monthly"
        assert result["total_slos"] == 3

    def test_generate_quarterly_report(self, service_with_budgets):
        """Generate a quarterly compliance report."""
        result = service_with_budgets.generate_quarterly_report(quarter=1, year=2026)
        assert result["report_type"] == "quarterly"
        assert result["total_slos"] == 3

    def test_list_reports_empty(self, slo_service):
        """List reports when none exist."""
        result = slo_service.list_reports()
        assert result == []

    def test_list_reports_with_data(self, service_with_budgets):
        """List reports after generating some."""
        service_with_budgets.generate_weekly_report()
        service_with_budgets.generate_weekly_report()
        result = service_with_budgets.list_reports()
        assert len(result) == 2

    def test_list_reports_filter_by_type(self, service_with_budgets):
        """List reports filtered by type."""
        service_with_budgets.generate_weekly_report()
        service_with_budgets.generate_monthly_report(month=1, year=2026)
        result = service_with_budgets.list_reports(report_type="weekly")
        assert all(r["report_type"] == "weekly" for r in result)

    def test_list_reports_limit(self, service_with_budgets):
        """List reports respects limit parameter."""
        for _ in range(5):
            service_with_budgets.generate_weekly_report()
        result = service_with_budgets.list_reports(limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestSLOServiceHealth:
    """Tests for health check."""

    def test_health_check(self, populated_service):
        """Health check returns correct status."""
        result = populated_service.health_check()
        assert result["healthy"] is True
        assert result["service"] == "slo-service"
        assert result["total_slos"] == 3
        assert result["enabled_slos"] == 3
        assert result["evaluation_loop_running"] is False

    def test_health_check_empty(self, slo_service):
        """Health check with no SLOs."""
        result = slo_service.health_check()
        assert result["total_slos"] == 0
        assert result["enabled_slos"] == 0


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


class TestSLOServiceEvaluation:
    """Tests for SLO evaluation."""

    def test_evaluate_all(self, populated_service):
        """Evaluate all SLOs stores budget snapshots."""
        import socket as _sock_mod

        # Save and restore real sockets for event loop creation
        saved = _sock_mod.socket
        saved_cc = _sock_mod.create_connection
        try:
            from tests.integration.slo_service.conftest import (
                _ORIGINAL_SOCKET,
                _ORIGINAL_CREATE_CONNECTION,
            )
            _sock_mod.socket = _ORIGINAL_SOCKET
            _sock_mod.create_connection = _ORIGINAL_CREATE_CONNECTION
        except (ImportError, AttributeError):
            pass  # Not in integration test context, sockets are fine

        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "success",
                "data": {
                    "resultType": "vector",
                    "result": [{"metric": {}, "value": [0, "0.9995"]}],
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

                loop = asyncio.new_event_loop()
                try:
                    results = loop.run_until_complete(
                        populated_service.evaluate_all()
                    )
                finally:
                    loop.close()

            assert len(results) == 3
            for result in results:
                assert result["status"] == "evaluated"
                assert result["sli_value"] > 0
                assert "budget_status" in result

            # Budget store should have entries
            assert len(populated_service._budget_store) == 3
        finally:
            _sock_mod.socket = saved
            _sock_mod.create_connection = saved_cc

    def test_evaluate_all_with_error(self, populated_service):
        """Evaluate handles SLI errors gracefully."""
        import socket as _sock_mod

        saved = _sock_mod.socket
        saved_cc = _sock_mod.create_connection
        try:
            from tests.integration.slo_service.conftest import (
                _ORIGINAL_SOCKET,
                _ORIGINAL_CREATE_CONNECTION,
            )
            _sock_mod.socket = _ORIGINAL_SOCKET
            _sock_mod.create_connection = _ORIGINAL_CREATE_CONNECTION
        except (ImportError, AttributeError):
            pass

        try:
            with patch(
                "greenlang.infrastructure.slo_service.sli_calculator.HTTPX_AVAILABLE",
                False,
            ):
                loop = asyncio.new_event_loop()
                try:
                    results = loop.run_until_complete(
                        populated_service.evaluate_all()
                    )
                finally:
                    loop.close()

            # Should return results even with errors
            assert len(results) == 3
            for result in results:
                assert result["status"] in ("evaluated", "error")
        finally:
            _sock_mod.socket = saved
            _sock_mod.create_connection = saved_cc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestSLOServiceHelpers:
    """Tests for internal helper methods."""

    def test_get_all_latest_budgets_empty(self, slo_service):
        """Get latest budgets when store is empty."""
        result = slo_service._get_all_latest_budgets()
        assert result == []

    def test_get_all_latest_budgets(self, service_with_budgets):
        """Get latest budgets when store has data."""
        result = service_with_budgets._get_all_latest_budgets()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# configure_slo_service / get_slo_service
# ---------------------------------------------------------------------------


class TestConfigureSLOService:
    """Tests for the setup entry point functions."""

    def test_configure_without_app(self, test_config):
        """Configure SLO service without FastAPI app."""
        svc = configure_slo_service(config=test_config)
        assert isinstance(svc, SLOService)
        assert svc.config is test_config

    def test_configure_with_fastapi_app(self, test_config):
        """Configure SLO service with a FastAPI app."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = FastAPI()
        svc = configure_slo_service(app=app, config=test_config)
        assert isinstance(svc, SLOService)
        assert app.state.slo_service is svc

    def test_get_slo_service_configured(self, test_config):
        """Get SLO service from a configured app."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = FastAPI()
        expected = configure_slo_service(app=app, config=test_config)
        result = get_slo_service(app)
        assert result is expected

    def test_get_slo_service_not_configured(self):
        """Get SLO service from unconfigured app raises RuntimeError."""
        try:
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("FastAPI not installed")

        app = FastAPI()
        with pytest.raises(RuntimeError, match="not configured"):
            get_slo_service(app)

    def test_configure_default_config(self):
        """Configure with default config from env."""
        svc = configure_slo_service()
        assert isinstance(svc, SLOService)
