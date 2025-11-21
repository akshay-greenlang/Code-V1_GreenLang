# -*- coding: utf-8 -*-
"""
Tests for Budget Tracking and Enforcement

Tests:
- Cost calculation
- Budget enforcement
- Multi-level budgets (request, hour, day, month)
- Budget alerts
- Usage tracking
"""

import uuid
from datetime import datetime, timedelta

import pytest

from greenlang.intelligence.budget import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock
    Budget,
    BudgetExceededError,
    BudgetMetrics,
    BudgetPeriod,
    BudgetTracker,
    ModelPricing,
    STANDARD_PRICING,
    Usage,
)


class TestModelPricing:
    """Test model pricing"""

    def test_calculate_cost(self):
        """Test cost calculation"""
        pricing = ModelPricing("gpt-4", 0.03, 0.06)

        # 1000 input, 500 output tokens
        cost = pricing.calculate_cost(1000, 500)

        expected = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert abs(cost - expected) < 0.0001
        assert cost == 0.06  # 0.03 + 0.03

    def test_standard_pricing(self):
        """Test standard pricing is available"""
        assert "gpt-4" in STANDARD_PRICING
        assert "gpt-3.5-turbo" in STANDARD_PRICING
        assert "claude-3-sonnet-20240229" in STANDARD_PRICING


class TestBudget:
    """Test budget configuration"""

    def test_init_defaults(self):
        """Test default budget"""
        budget = Budget()

        assert budget.max_cost_per_request == 0.10
        assert budget.max_tokens_per_request == 4000
        assert budget.max_cost_per_hour == 10.00
        assert budget.max_cost_per_day == 100.00
        assert budget.max_cost_per_month == 1000.00

    def test_custom_budget(self):
        """Test custom budget"""
        budget = Budget(
            max_cost_per_request=0.50,
            max_cost_per_hour=20.00,
        )

        assert budget.max_cost_per_request == 0.50
        assert budget.max_cost_per_hour == 20.00


class TestUsage:
    """Test usage record"""

    def test_total_tokens(self):
        """Test total tokens calculation"""
        usage = Usage(
            request_id="test",
            model="gpt-4",
            input_tokens=500,
            output_tokens=300,
            cost=0.05,
        )

        assert usage.total_tokens == 800


class TestBudgetTracker:
    """Test budget tracker"""

    @pytest.fixture
    def budget(self):
        """Create test budget"""
        return Budget(
            max_cost_per_request=0.10,
            max_cost_per_hour=1.00,
            max_cost_per_day=10.00,
        )

    @pytest.fixture
    def tracker(self, budget):
        """Create budget tracker"""
        return BudgetTracker(budget=budget, enable_enforcement=True)

    def test_init(self, tracker):
        """Test initialization"""
        assert tracker.enable_enforcement is True
        assert len(tracker.usage_history) == 0

    def test_calculate_cost(self, tracker):
        """Test cost calculation"""
        cost = tracker.calculate_cost("gpt-4", 1000, 500)

        # GPT-4: $0.03/1k input, $0.06/1k output
        expected = (1000 / 1000) * 0.03 + (500 / 1000) * 0.06
        assert abs(cost - expected) < 0.0001

    def test_record_usage(self, tracker):
        """Test recording usage"""
        usage = tracker.record_usage(
            request_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            model="gpt-4",
            input_tokens=500,
            output_tokens=300,
            agent_id="test_agent",
        )

        assert usage.cost > 0
        assert len(tracker.usage_history) == 1
        assert tracker.metrics.total_requests == 1

    def test_check_budget_within_limit(self, tracker):
        """Test budget check when within limit"""
        # Small request, should pass
        try:
            tracker.check_budget("gpt-3.5-turbo", 500, 300)
        except BudgetExceededError:
            pytest.fail("Budget check should not raise for small request")

    def test_check_budget_exceeds_request_limit(self, tracker):
        """Test budget check exceeds per-request limit"""
        # Large request, exceeds $0.10 limit
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget("gpt-4", 10000, 5000)

        assert exc_info.value.budget_type == BudgetPeriod.REQUEST

    def test_check_budget_exceeds_hourly_limit(self, tracker):
        """Test budget check exceeds hourly limit"""
        # Add usage up to hourly limit
        for i in range(10):
            tracker.record_usage(
                request_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                model="gpt-4",
                input_tokens=1000,
                output_tokens=500,
            )

        # Next request should exceed hourly limit
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget("gpt-4", 1000, 500)

        assert exc_info.value.budget_type == BudgetPeriod.HOUR

    def test_check_budget_token_limit(self, tracker):
        """Test token limit enforcement"""
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget("gpt-4", 10000, 0)

        assert exc_info.value.budget_type == BudgetPeriod.REQUEST

    def test_cost_by_model(self, tracker):
        """Test cost breakdown by model"""
        tracker.record_usage("req1", "gpt-4", 1000, 500)
        tracker.record_usage("req2", "gpt-3.5-turbo", 1000, 500)
        tracker.record_usage("req3", "gpt-4", 500, 300)

        metrics = tracker.get_metrics()

        assert "gpt-4" in metrics.cost_by_model
        assert "gpt-3.5-turbo" in metrics.cost_by_model
        assert metrics.cost_by_model["gpt-4"] > metrics.cost_by_model["gpt-3.5-turbo"]

    def test_cost_by_agent(self, tracker):
        """Test cost breakdown by agent"""
        tracker.record_usage("req1", "gpt-4", 1000, 500, agent_id="agent1")
        tracker.record_usage("req2", "gpt-4", 1000, 500, agent_id="agent2")
        tracker.record_usage("req3", "gpt-4", 500, 300, agent_id="agent1")

        metrics = tracker.get_metrics()

        assert "agent1" in metrics.cost_by_agent
        assert "agent2" in metrics.cost_by_agent
        assert metrics.cost_by_agent["agent1"] > metrics.cost_by_agent["agent2"]

    def test_budget_utilization(self, tracker):
        """Test budget utilization calculation"""
        # Add usage (50% of hourly budget)
        for _ in range(5):
            tracker.record_usage(
                str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                "gpt-4",
                1000,
                500,
            )

        util = tracker.get_budget_utilization(BudgetPeriod.HOUR)

        assert 0.4 < util < 0.6  # Should be around 50%

    def test_budget_alerts(self, tracker):
        """Test budget alert thresholds"""
        # Add usage up to 85% of hourly budget
        target_cost = tracker.budget.max_cost_per_hour * 0.85

        while tracker.metrics.hourly_cost < target_cost:
            tracker.record_usage(
                str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                "gpt-4",
                500,
                300,
            )

        # Should trigger 80% alert
        alert = tracker.should_alert(BudgetPeriod.HOUR)
        assert alert is not None
        assert alert == 0.8

    def test_disabled_enforcement(self):
        """Test with enforcement disabled"""
        budget = Budget(max_cost_per_request=0.01)
        tracker = BudgetTracker(budget=budget, enable_enforcement=False)

        # Should not raise even though exceeds budget
        try:
            tracker.check_budget("gpt-4", 10000, 5000)
        except BudgetExceededError:
            pytest.fail("Should not enforce budget when disabled")

    def test_period_cost_calculation(self, tracker):
        """Test period cost calculation"""
        # Add old usage (should not count)
        old_usage = Usage(
            request_id="old",
            model="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            cost=0.06,
            timestamp=DeterministicClock.now() - timedelta(hours=2),
        )
        tracker.usage_history.append(old_usage)

        # Add recent usage (should count)
        tracker.record_usage("recent", "gpt-4", 500, 300)

        hourly_cost = tracker._calculate_period_cost(hours=1)

        # Old usage should not be included
        assert hourly_cost < 0.06


class TestBudgetMetrics:
    """Test budget metrics"""

    def test_init(self):
        """Test initialization"""
        metrics = BudgetMetrics()

        assert metrics.total_cost == 0.0
        assert metrics.total_requests == 0


class TestBudgetIntegration:
    """Integration tests"""

    def test_realistic_usage_pattern(self):
        """Test realistic usage pattern"""
        budget = Budget(
            max_cost_per_request=0.20,
            max_cost_per_hour=5.00,
            max_cost_per_day=50.00,
        )
        tracker = BudgetTracker(budget=budget)

        # Simulate 10 requests
        models = ["gpt-4", "gpt-3.5-turbo"]
        total_cost = 0

        for i in range(10):
            model = models[i % 2]
            input_tokens = 500 + (i * 50)
            output_tokens = 300

            try:
                tracker.check_budget(model, input_tokens, output_tokens)
                usage = tracker.record_usage(
                    str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                    model,
                    input_tokens,
                    output_tokens,
                    agent_id=f"agent{i % 3}",
                )
                total_cost += usage.cost
            except BudgetExceededError as e:
                print(f"Budget exceeded at request {i+1}: {e}")
                break

        metrics = tracker.get_metrics()

        assert metrics.total_requests > 0
        assert metrics.total_cost > 0
        assert abs(metrics.total_cost - total_cost) < 0.0001

    def test_budget_enforcement_prevents_overspending(self):
        """Test that enforcement prevents overspending"""
        budget = Budget(
            max_cost_per_hour=0.10,  # Very low limit
        )
        tracker = BudgetTracker(budget=budget)

        total_cost = 0
        requests = 0

        # Try to make many requests
        for i in range(100):
            try:
                tracker.check_budget("gpt-4", 500, 300)
                usage = tracker.record_usage(
                    str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                    "gpt-4",
                    500,
                    300,
                )
                total_cost += usage.cost
                requests += 1
            except BudgetExceededError:
                break

        # Should have stopped before exceeding budget
        assert total_cost <= budget.max_cost_per_hour
        assert requests < 100


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v", "-s"])
