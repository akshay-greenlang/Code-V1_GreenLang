# -*- coding: utf-8 -*-
"""
Unit tests for Error Budget Calculator (OBS-005)

Tests budget calculation at different target levels, status determination,
consumption rates, exhaustion forecasting, budget policies, and Redis
caching.

Coverage target: 85%+ of error_budget.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from greenlang.infrastructure.slo_service.error_budget import (
    BudgetCache,
    budget_consumption_rate,
    calculate_error_budget,
    check_budget_policy,
    forecast_exhaustion,
)
from greenlang.infrastructure.slo_service.models import BudgetStatus, ErrorBudget


class TestErrorBudgetCalculation:
    """Tests for the calculate_error_budget function."""

    def test_budget_calculation_99_95_target(self, slo_factory):
        """99.95% target with 99.99% SLI consumes ~20% of budget."""
        slo = slo_factory(target=99.95)
        budget = calculate_error_budget(slo, current_sli=0.9999)
        # Allowed error = 0.0005, actual error = 0.0001
        # Consumed = 0.0001/0.0005 = 20%
        assert budget.consumed_percent == pytest.approx(20.0, rel=0.01)
        assert budget.status == BudgetStatus.WARNING

    def test_budget_calculation_99_9_target(self, slo_factory):
        """99.9% target with 99.95% SLI consumes 50% of budget."""
        slo = slo_factory(target=99.9)
        budget = calculate_error_budget(slo, current_sli=0.9995)
        # Allowed error = 0.001, actual error = 0.0005
        # Consumed = 0.0005/0.001 = 50%
        assert budget.consumed_percent == pytest.approx(50.0, rel=0.01)
        assert budget.status == BudgetStatus.CRITICAL

    def test_budget_calculation_99_0_target(self, slo_factory):
        """99.0% target with 99.5% SLI consumes 50% of budget."""
        slo = slo_factory(target=99.0)
        budget = calculate_error_budget(slo, current_sli=0.995)
        assert budget.consumed_percent == pytest.approx(50.0, rel=0.01)

    def test_budget_calculation_100_target(self, slo_factory):
        """100% target means zero error budget; any error = exhausted."""
        slo = slo_factory(target=100.0)
        budget = calculate_error_budget(slo, current_sli=0.999)
        assert budget.consumed_percent == 100.0
        assert budget.status == BudgetStatus.EXHAUSTED

    def test_budget_calculation_100_target_no_errors(self, slo_factory):
        """100% target with 100% SLI is healthy."""
        slo = slo_factory(target=100.0)
        budget = calculate_error_budget(slo, current_sli=1.0)
        assert budget.consumed_percent == 0.0

    def test_budget_remaining_percent(self, slo_factory):
        """remaining_percent = 100 - consumed_percent."""
        slo = slo_factory(target=99.9)
        budget = calculate_error_budget(slo, current_sli=0.9995)
        assert budget.remaining_percent == pytest.approx(100.0 - budget.consumed_percent)

    def test_budget_status_healthy(self, slo_factory):
        """Budget is HEALTHY when consumed < 20%."""
        slo = slo_factory(target=99.9)
        budget = calculate_error_budget(slo, current_sli=0.99999)
        assert budget.status == BudgetStatus.HEALTHY

    def test_budget_status_warning(self, slo_factory):
        """Budget is WARNING when consumed >= 20% and < 50%."""
        slo = slo_factory(target=99.9)
        # 30% consumed: error rate = 0.001 * 0.3 = 0.0003
        budget = calculate_error_budget(slo, current_sli=0.9997)
        assert budget.status == BudgetStatus.WARNING

    def test_budget_status_critical(self, slo_factory):
        """Budget is CRITICAL when consumed >= 50% and < 100%."""
        slo = slo_factory(target=99.9)
        budget = calculate_error_budget(slo, current_sli=0.9993)
        assert budget.status == BudgetStatus.CRITICAL

    def test_budget_status_exhausted(self, slo_factory):
        """Budget is EXHAUSTED when consumed >= 100%."""
        slo = slo_factory(target=99.9)
        budget = calculate_error_budget(slo, current_sli=0.998)
        assert budget.status == BudgetStatus.EXHAUSTED


class TestConsumptionRate:
    """Tests for budget_consumption_rate."""

    def test_budget_consumption_rate(self):
        """Consumption rate is calculated as %/hour."""
        rate = budget_consumption_rate(10.0, 60.0)
        assert rate == pytest.approx(10.0)  # 10% in 60 min = 10%/hr

    def test_budget_consumption_rate_zero_elapsed(self):
        """Zero elapsed time returns 0 rate."""
        assert budget_consumption_rate(50.0, 0.0) == 0.0


class TestForecastExhaustion:
    """Tests for forecast_exhaustion."""

    def test_forecast_exhaustion_date(self):
        """Positive rate produces a future date."""
        result = forecast_exhaustion(50.0, 5.0)
        assert result is not None
        assert result > datetime.now(timezone.utc)

    def test_forecast_no_exhaustion(self):
        """Zero rate returns None (no exhaustion projected)."""
        result = forecast_exhaustion(50.0, 0.0)
        assert result is None

    def test_forecast_already_exhausted(self):
        """100% consumed returns roughly now."""
        result = forecast_exhaustion(100.0, 5.0)
        assert result is not None
        diff = abs((result - datetime.now(timezone.utc)).total_seconds())
        assert diff < 2  # within 2 seconds of now


class TestBudgetPolicy:
    """Tests for check_budget_policy."""

    def test_check_budget_policy_freeze(self, sample_error_budget_exhausted, slo_factory):
        """freeze_deployments policy triggers action when exhausted."""
        result = check_budget_policy(sample_error_budget_exhausted, "freeze_deployments")
        assert result["action_required"] is True
        assert result["action"] == "freeze_deployments"

    def test_check_budget_policy_alert_only(self, sample_error_budget_exhausted):
        """alert_only policy triggers alert action when exhausted."""
        result = check_budget_policy(sample_error_budget_exhausted, "alert_only")
        assert result["action_required"] is True
        assert result["action"] == "alert_only"

    def test_check_budget_policy_none(self, sample_error_budget_exhausted):
        """none policy does not trigger action."""
        result = check_budget_policy(sample_error_budget_exhausted, "none")
        assert result["action_required"] is False
        assert result["action"] == "none"

    def test_check_budget_policy_critical_alerts(self, sample_error_budget_critical):
        """Critical budget triggers alert_only for freeze policy."""
        result = check_budget_policy(sample_error_budget_critical, "freeze_deployments")
        assert result["action_required"] is True
        assert result["action"] == "alert_only"

    def test_check_budget_policy_healthy_no_action(self, sample_error_budget):
        """Healthy budget requires no action."""
        result = check_budget_policy(sample_error_budget, "freeze_deployments")
        assert result["action_required"] is False


class TestBudgetCache:
    """Tests for the Redis-backed BudgetCache."""

    def test_redis_cache_hit(self, mock_redis_with_data):
        """Cache returns ErrorBudget on hit."""
        cache = BudgetCache(mock_redis_with_data)
        budget = cache.get("api-availability-99-9")
        assert budget is not None
        assert isinstance(budget, ErrorBudget)

    def test_redis_cache_miss(self, mock_redis):
        """Cache returns None on miss."""
        cache = BudgetCache(mock_redis)
        result = cache.get("nonexistent")
        assert result is None

    def test_record_budget_snapshot(self, mock_redis, sample_error_budget):
        """Cache set stores budget data."""
        cache = BudgetCache(mock_redis)
        result = cache.set(sample_error_budget)
        assert result is True
        mock_redis.setex.assert_called_once()

    def test_cache_invalidation(self, mock_redis):
        """Cache invalidation deletes the key."""
        cache = BudgetCache(mock_redis)
        result = cache.invalidate("test-slo")
        assert result is True
        mock_redis.delete.assert_called_once()

    def test_cache_error_handling_get(self):
        """Cache get gracefully handles Redis errors."""
        bad_redis = MagicMock()
        bad_redis.get.side_effect = Exception("Redis down")
        cache = BudgetCache(bad_redis)
        assert cache.get("test") is None

    def test_cache_error_handling_set(self, sample_error_budget):
        """Cache set gracefully handles Redis errors."""
        bad_redis = MagicMock()
        bad_redis.setex.side_effect = Exception("Redis down")
        cache = BudgetCache(bad_redis)
        assert cache.set(sample_error_budget) is False

    def test_cache_key_format(self, mock_redis):
        """Cache keys use the correct prefix."""
        cache = BudgetCache(mock_redis, prefix="gl:slo:budget:")
        cache.get("my-slo-id")
        mock_redis.get.assert_called_with("gl:slo:budget:my-slo-id")
