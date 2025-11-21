# -*- coding: utf-8 -*-
"""
Unit tests for budget enforcement and error handling

Tests Budget class functionality including:
- check() raises when would exceed cap
- add() raises when would exceed cap
- BudgetExceeded error details
- Budget.merge() aggregation
- Budget.reset() functionality
- Token and dollar cap enforcement
- Remaining budget calculations
"""

import pytest
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded


class TestBudgetCheck:
    """Test Budget.check() method"""

    def test_check_passes_when_under_cap(self):
        """check() should pass when usage would stay under cap"""
        budget = Budget(max_usd=0.50)
        # Should not raise
        budget.check(add_usd=0.30, add_tokens=3000)

    def test_check_raises_when_exceeds_dollar_cap(self):
        """check() should raise BudgetExceeded when would exceed dollar cap"""
        budget = Budget(max_usd=0.50)

        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check(add_usd=0.60, add_tokens=1000)

        assert exc_info.value.spent_usd == 0.60
        assert exc_info.value.max_usd == 0.50
        assert "Dollar cap exceeded" in str(exc_info.value)

    def test_check_raises_when_exceeds_token_cap(self):
        """check() should raise BudgetExceeded when would exceed token cap"""
        budget = Budget(max_usd=1.00, max_tokens=4000)

        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check(add_usd=0.10, add_tokens=5000)

        assert exc_info.value.spent_tokens == 5000
        assert exc_info.value.max_tokens == 4000
        assert "Token cap exceeded" in str(exc_info.value)

    def test_check_with_existing_usage(self):
        """check() should consider existing usage"""
        budget = Budget(max_usd=0.50)
        budget.spent_usd = 0.40

        # Adding 0.15 would exceed 0.50 cap
        with pytest.raises(BudgetExceeded):
            budget.check(add_usd=0.15, add_tokens=1000)

    def test_check_exact_cap_passes(self):
        """check() should pass when exactly at cap"""
        budget = Budget(max_usd=0.50)
        # Should not raise
        budget.check(add_usd=0.50, add_tokens=1000)

    def test_check_token_cap_priority(self):
        """check() should check token cap before dollar cap"""
        budget = Budget(max_usd=1.00, max_tokens=4000)

        # Token cap exceeded (checked first)
        with pytest.raises(BudgetExceeded, match="Token cap exceeded"):
            budget.check(add_usd=0.10, add_tokens=5000)


class TestBudgetAdd:
    """Test Budget.add() method"""

    def test_add_updates_spent_amounts(self):
        """add() should update spent_usd and spent_tokens"""
        budget = Budget(max_usd=0.50)

        budget.add(add_usd=0.10, add_tokens=1000)

        assert budget.spent_usd == 0.10
        assert budget.spent_tokens == 1000

    def test_add_accumulates_usage(self):
        """add() should accumulate usage across multiple calls"""
        budget = Budget(max_usd=0.50)

        budget.add(add_usd=0.10, add_tokens=1000)
        budget.add(add_usd=0.15, add_tokens=1500)
        budget.add(add_usd=0.05, add_tokens=500)

        assert budget.spent_usd == 0.30
        assert budget.spent_tokens == 3000

    def test_add_raises_when_would_exceed_cap(self):
        """add() should raise BudgetExceeded when would exceed cap"""
        budget = Budget(max_usd=0.50)

        with pytest.raises(BudgetExceeded):
            budget.add(add_usd=0.60, add_tokens=1000)

        # Budget should NOT be updated after failure
        assert budget.spent_usd == 0.0
        assert budget.spent_tokens == 0

    def test_add_raises_on_second_call_if_exceeds(self):
        """add() should raise on second call if total would exceed"""
        budget = Budget(max_usd=0.50)

        budget.add(add_usd=0.40, add_tokens=4000)

        with pytest.raises(BudgetExceeded):
            budget.add(add_usd=0.20, add_tokens=2000)

        # Budget should still show first call only
        assert budget.spent_usd == 0.40
        assert budget.spent_tokens == 4000

    def test_add_with_token_cap(self):
        """add() should enforce token cap"""
        budget = Budget(max_usd=1.00, max_tokens=10000)

        budget.add(add_usd=0.10, add_tokens=3000)
        budget.add(add_usd=0.15, add_tokens=4000)

        # This would exceed token cap
        with pytest.raises(BudgetExceeded, match="Token cap exceeded"):
            budget.add(add_usd=0.10, add_tokens=5000)

        # Budget shows partial usage
        assert budget.spent_tokens == 7000


class TestBudgetExceededError:
    """Test BudgetExceeded exception details"""

    def test_error_includes_spent_amount(self):
        """BudgetExceeded should include spent amount"""
        budget = Budget(max_usd=0.50)

        try:
            budget.add(add_usd=0.60, add_tokens=6000)
        except BudgetExceeded as e:
            assert e.spent_usd == 0.60
            assert e.max_usd == 0.50
            assert e.spent_tokens == 6000

    def test_error_includes_token_details(self):
        """BudgetExceeded should include token cap details"""
        budget = Budget(max_usd=1.00, max_tokens=4000)

        try:
            budget.add(add_usd=0.10, add_tokens=5000)
        except BudgetExceeded as e:
            assert e.spent_tokens == 5000
            assert e.max_tokens == 4000

    def test_error_string_representation(self):
        """BudgetExceeded should have helpful string representation"""
        budget = Budget(max_usd=0.50, max_tokens=4000)

        try:
            budget.add(add_usd=0.60, add_tokens=6000)
        except BudgetExceeded as e:
            error_str = str(e)
            assert "0.60" in error_str  # Spent
            assert "0.50" in error_str  # Max
            assert "6000" in error_str or "6,000" in error_str  # Tokens

    def test_error_message_different_for_token_vs_dollar(self):
        """BudgetExceeded should have different message for token vs dollar"""
        # Dollar cap exceeded
        budget1 = Budget(max_usd=0.50, max_tokens=10000)
        try:
            budget1.add(add_usd=0.60, add_tokens=1000)
        except BudgetExceeded as e:
            assert "Dollar cap exceeded" in e.message

        # Token cap exceeded
        budget2 = Budget(max_usd=1.00, max_tokens=4000)
        try:
            budget2.add(add_usd=0.10, add_tokens=5000)
        except BudgetExceeded as e:
            assert "Token cap exceeded" in e.message


class TestBudgetMerge:
    """Test Budget.merge() functionality"""

    def test_merge_aggregates_spending(self):
        """merge() should aggregate spending from another budget"""
        workflow_budget = Budget(max_usd=2.00)

        agent1_budget = Budget(max_usd=0.50, spent_usd=0.30, spent_tokens=3000)
        agent2_budget = Budget(max_usd=0.50, spent_usd=0.25, spent_tokens=2500)

        workflow_budget.merge(agent1_budget)
        workflow_budget.merge(agent2_budget)

        assert workflow_budget.spent_usd == 0.55
        assert workflow_budget.spent_tokens == 5500

    def test_merge_raises_if_would_exceed_cap(self):
        """merge() should raise BudgetExceeded if merged total exceeds cap"""
        workflow_budget = Budget(max_usd=0.50)

        agent_budget = Budget(max_usd=1.00, spent_usd=0.60, spent_tokens=6000)

        with pytest.raises(BudgetExceeded):
            workflow_budget.merge(agent_budget)

    def test_merge_empty_budget(self):
        """merge() should handle empty budget (zero spending)"""
        workflow_budget = Budget(max_usd=1.00)

        empty_budget = Budget(max_usd=0.50)  # No spending

        workflow_budget.merge(empty_budget)

        assert workflow_budget.spent_usd == 0.0
        assert workflow_budget.spent_tokens == 0

    def test_merge_multiple_budgets(self):
        """merge() should support merging multiple budgets"""
        workflow_budget = Budget(max_usd=5.00, max_tokens=50000)

        budgets = [
            Budget(max_usd=1.00, spent_usd=0.50, spent_tokens=5000),
            Budget(max_usd=1.00, spent_usd=0.30, spent_tokens=3000),
            Budget(max_usd=1.00, spent_usd=0.40, spent_tokens=4000),
        ]

        for b in budgets:
            workflow_budget.merge(b)

        assert workflow_budget.spent_usd == 1.20
        assert workflow_budget.spent_tokens == 12000


class TestBudgetReset:
    """Test Budget.reset() functionality"""

    def test_reset_clears_counters(self):
        """reset() should clear spent_usd and spent_tokens"""
        budget = Budget(max_usd=0.50)
        budget.add(add_usd=0.30, add_tokens=3000)

        assert budget.spent_usd == 0.30
        assert budget.spent_tokens == 3000

        budget.reset()

        assert budget.spent_usd == 0.0
        assert budget.spent_tokens == 0

    def test_reset_preserves_caps(self):
        """reset() should preserve max_usd and max_tokens"""
        budget = Budget(max_usd=0.50, max_tokens=4000)
        budget.add(add_usd=0.30, add_tokens=3000)

        budget.reset()

        assert budget.max_usd == 0.50
        assert budget.max_tokens == 4000

    def test_reset_allows_reuse(self):
        """reset() should allow budget reuse"""
        budget = Budget(max_usd=0.50)

        # First use
        budget.add(add_usd=0.40, add_tokens=4000)

        # Reset
        budget.reset()

        # Second use (should not raise)
        budget.add(add_usd=0.40, add_tokens=4000)

        assert budget.spent_usd == 0.40

    def test_reset_multiple_times(self):
        """reset() should work multiple times"""
        budget = Budget(max_usd=0.50)

        for _ in range(5):
            budget.add(add_usd=0.10, add_tokens=1000)
            budget.reset()

        assert budget.spent_usd == 0.0
        assert budget.spent_tokens == 0


class TestTokenCapEnforcement:
    """Test token cap enforcement"""

    def test_token_cap_enforced(self):
        """Token cap should be enforced"""
        budget = Budget(max_usd=1.00, max_tokens=4000)

        with pytest.raises(BudgetExceeded, match="Token cap exceeded"):
            budget.add(add_usd=0.10, add_tokens=5000)

    def test_token_cap_exact(self):
        """Exact token cap should be allowed"""
        budget = Budget(max_usd=1.00, max_tokens=4000)

        # Should not raise
        budget.add(add_usd=0.10, add_tokens=4000)

        assert budget.spent_tokens == 4000

    def test_token_cap_none_allows_unlimited(self):
        """max_tokens=None should allow unlimited tokens"""
        budget = Budget(max_usd=1.00, max_tokens=None)

        # Should not raise even with very high token count
        budget.add(add_usd=0.10, add_tokens=1000000)

        assert budget.spent_tokens == 1000000

    def test_token_cap_accumulation(self):
        """Token cap should apply to accumulated tokens"""
        budget = Budget(max_usd=1.00, max_tokens=10000)

        budget.add(add_usd=0.10, add_tokens=3000)
        budget.add(add_usd=0.15, add_tokens=4000)

        # This exceeds 10000 token cap
        with pytest.raises(BudgetExceeded, match="Token cap exceeded"):
            budget.add(add_usd=0.10, add_tokens=4000)


class TestDollarCapEnforcement:
    """Test dollar cap enforcement"""

    def test_dollar_cap_enforced(self):
        """Dollar cap should be enforced"""
        budget = Budget(max_usd=0.50)

        with pytest.raises(BudgetExceeded, match="Dollar cap exceeded"):
            budget.add(add_usd=0.60, add_tokens=1000)

    def test_dollar_cap_exact(self):
        """Exact dollar cap should be allowed"""
        budget = Budget(max_usd=0.50)

        # Should not raise
        budget.add(add_usd=0.50, add_tokens=5000)

        assert budget.spent_usd == 0.50

    def test_dollar_cap_accumulation(self):
        """Dollar cap should apply to accumulated spending"""
        budget = Budget(max_usd=0.50)

        budget.add(add_usd=0.20, add_tokens=2000)
        budget.add(add_usd=0.15, add_tokens=1500)

        # This exceeds 0.50 cap
        with pytest.raises(BudgetExceeded, match="Dollar cap exceeded"):
            budget.add(add_usd=0.20, add_tokens=2000)

    def test_dollar_cap_precision(self):
        """Dollar cap should handle floating point precision"""
        budget = Budget(max_usd=0.10)

        # Multiple small additions
        for _ in range(10):
            budget.add(add_usd=0.01, add_tokens=100)

        # Should be exactly at cap
        assert abs(budget.spent_usd - 0.10) < 0.0001


class TestBudgetRemainingCalculations:
    """Test remaining budget calculations"""

    def test_remaining_usd_initial(self):
        """remaining_usd should equal max_usd initially"""
        budget = Budget(max_usd=0.50)
        assert budget.remaining_usd == 0.50

    def test_remaining_usd_after_spending(self):
        """remaining_usd should decrease after spending"""
        budget = Budget(max_usd=0.50)
        budget.add(add_usd=0.30, add_tokens=3000)

        assert budget.remaining_usd == 0.20

    def test_remaining_usd_zero_when_exhausted(self):
        """remaining_usd should be zero when budget exhausted"""
        budget = Budget(max_usd=0.50)
        budget.add(add_usd=0.50, add_tokens=5000)

        assert budget.remaining_usd == 0.0

    def test_remaining_usd_never_negative(self):
        """remaining_usd should never go negative"""
        budget = Budget(max_usd=0.50)
        budget.spent_usd = 0.60  # Simulate over-spending (shouldn't happen in practice)

        assert budget.remaining_usd == 0.0

    def test_remaining_tokens_initial(self):
        """remaining_tokens should equal max_tokens initially"""
        budget = Budget(max_usd=0.50, max_tokens=4000)
        assert budget.remaining_tokens == 4000

    def test_remaining_tokens_after_spending(self):
        """remaining_tokens should decrease after spending"""
        budget = Budget(max_usd=0.50, max_tokens=4000)
        budget.add(add_usd=0.10, add_tokens=1000)

        assert budget.remaining_tokens == 3000

    def test_remaining_tokens_none_when_no_cap(self):
        """remaining_tokens should be None when no token cap"""
        budget = Budget(max_usd=0.50, max_tokens=None)
        assert budget.remaining_tokens is None

    def test_remaining_tokens_zero_when_exhausted(self):
        """remaining_tokens should be zero when token budget exhausted"""
        budget = Budget(max_usd=1.00, max_tokens=4000)
        budget.add(add_usd=0.10, add_tokens=4000)

        assert budget.remaining_tokens == 0

    def test_is_exhausted_dollar_cap(self):
        """is_exhausted should be True when dollar cap reached"""
        budget = Budget(max_usd=0.50)
        assert not budget.is_exhausted

        budget.add(add_usd=0.50, add_tokens=1000)
        assert budget.is_exhausted

    def test_is_exhausted_token_cap(self):
        """is_exhausted should be True when token cap reached"""
        budget = Budget(max_usd=1.00, max_tokens=4000)
        assert not budget.is_exhausted

        budget.add(add_usd=0.10, add_tokens=4000)
        assert budget.is_exhausted

    def test_is_exhausted_either_cap(self):
        """is_exhausted should be True if either cap reached"""
        # Token cap reached first
        budget1 = Budget(max_usd=1.00, max_tokens=4000)
        budget1.add(add_usd=0.10, add_tokens=4000)
        assert budget1.is_exhausted

        # Dollar cap reached first
        budget2 = Budget(max_usd=0.50, max_tokens=10000)
        budget2.add(add_usd=0.50, add_tokens=1000)
        assert budget2.is_exhausted
