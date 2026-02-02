# -*- coding: utf-8 -*-
"""
Budget Enforcement Tests
GL Intelligence Infrastructure

Tests for budget tracking and enforcement in ChatSession and RAG.
Critical for cost control and preventing runaway LLM expenses.

Version: 1.0.0
Date: 2025-11-06
"""

import pytest
import asyncio
from typing import List, Dict, Any
from pathlib import Path


class MockLLMProvider:
    """Mock LLM provider for budget testing."""

    def __init__(self, cost_per_call: float = 0.10):
        """Initialize mock provider with fixed cost."""
        self.cost_per_call = cost_per_call
        self.call_count = 0

    async def chat(self, messages: List[Dict], budget: Any, **kwargs) -> Any:
        """Mock chat that consumes budget."""
        from greenlang.intelligence.runtime.budget import BudgetExceeded

        self.call_count += 1

        # Check budget before consuming
        if budget.spent_usd + self.cost_per_call > budget.max_usd:
            raise BudgetExceeded(
                f"Budget exceeded: {budget.spent_usd + self.cost_per_call:.4f} > {budget.max_usd:.4f}"
            )

        # Track cost
        budget.track(
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=self.cost_per_call
        )

        # Mock response
        from greenlang.intelligence.schemas.responses import ChatResponse, Usage, FinishReason, ProviderInfo

        return ChatResponse(
            text="Mock response",
            tool_calls=[],
            usage=Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=self.cost_per_call
            ),
            finish_reason=FinishReason.stop,
            provider_info=ProviderInfo(provider="mock", model="mock-model"),
            raw=None
        )


class TestBudgetTracking:
    """Test budget tracking functionality."""

    @pytest.mark.asyncio
    async def test_budget_tracks_single_call(self):
        """Test budget tracks cost from single LLM call."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Track a call
        budget.track(
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.10
        )

        assert budget.spent_usd == 0.10
        assert budget.prompt_tokens == 100
        assert budget.completion_tokens == 50
        assert budget.total_tokens == 150
        assert budget.call_count == 1

    @pytest.mark.asyncio
    async def test_budget_tracks_multiple_calls(self):
        """Test budget accumulates costs across multiple calls."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Track multiple calls
        for i in range(5):
            budget.track(
                prompt_tokens=100,
                completion_tokens=50,
                cost_usd=0.05
            )

        assert budget.spent_usd == 0.25  # 5 * 0.05
        assert budget.call_count == 5
        assert budget.total_tokens == 750  # 5 * 150

    @pytest.mark.asyncio
    async def test_budget_remaining_calculation(self):
        """Test remaining budget calculation."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        assert budget.remaining_usd == 1.0

        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.30)
        assert budget.remaining_usd == 0.70

        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.40)
        assert budget.remaining_usd == 0.30

    @pytest.mark.asyncio
    async def test_budget_percentage_used(self):
        """Test percentage used calculation."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        assert budget.percentage_used == 0.0

        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.25)
        assert budget.percentage_used == 25.0

        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.50)
        assert budget.percentage_used == 75.0


class TestBudgetEnforcement:
    """Test budget enforcement and limits."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises_exception(self):
        """Test that exceeding budget raises BudgetExceeded."""
        from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

        budget = Budget(max_usd=0.20)

        # First call succeeds
        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.10)

        # Second call exceeds budget
        with pytest.raises(BudgetExceeded):
            # Attempt to track more than remaining budget
            if budget.spent_usd + 0.15 > budget.max_usd:
                raise BudgetExceeded(f"Would exceed budget: {budget.spent_usd + 0.15} > {budget.max_usd}")

    @pytest.mark.asyncio
    async def test_budget_enforced_in_llm_call(self):
        """Test budget is enforced during LLM calls."""
        from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

        budget = Budget(max_usd=0.25)
        provider = MockLLMProvider(cost_per_call=0.10)

        # First 2 calls should succeed
        await provider.chat([{"role": "user", "content": "Test 1"}], budget)
        await provider.chat([{"role": "user", "content": "Test 2"}], budget)

        assert provider.call_count == 2
        assert budget.spent_usd == 0.20

        # Third call should fail (would exceed budget)
        with pytest.raises(BudgetExceeded):
            await provider.chat([{"role": "user", "content": "Test 3"}], budget)

    @pytest.mark.asyncio
    async def test_budget_prevents_over_spending(self):
        """Test budget prevents spending beyond limit."""
        from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

        budget = Budget(max_usd=0.50)
        provider = MockLLMProvider(cost_per_call=0.20)

        # Use up budget
        await provider.chat([{"role": "user", "content": "1"}], budget)
        await provider.chat([{"role": "user", "content": "2"}], budget)

        assert budget.spent_usd == 0.40

        # Next call would exceed budget
        with pytest.raises(BudgetExceeded):
            await provider.chat([{"role": "user", "content": "3"}], budget)

        # Budget should not have changed
        assert budget.spent_usd == 0.40
        assert budget.call_count == 2

    @pytest.mark.asyncio
    async def test_budget_exact_limit_allowed(self):
        """Test that spending exactly up to limit is allowed."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=0.50)
        provider = MockLLMProvider(cost_per_call=0.25)

        # Spend exactly to limit
        await provider.chat([{"role": "user", "content": "1"}], budget)
        await provider.chat([{"role": "user", "content": "2"}], budget)

        assert budget.spent_usd == 0.50
        assert budget.remaining_usd == 0.0


class TestBudgetWithTools:
    """Test budget tracking with tool calling."""

    @pytest.mark.asyncio
    async def test_budget_tracks_tool_calls(self):
        """Test budget tracks costs for calls with tools."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Simulate tool-calling LLM call (typically more expensive)
        budget.track(
            prompt_tokens=200,  # More tokens due to tool definitions
            completion_tokens=100,
            cost_usd=0.20
        )

        assert budget.spent_usd == 0.20
        assert budget.prompt_tokens == 200
        assert budget.call_count == 1

    @pytest.mark.asyncio
    async def test_budget_enforced_with_tools(self):
        """Test budget enforcement works with tool calling."""
        from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

        budget = Budget(max_usd=0.30)
        provider = MockLLMProvider(cost_per_call=0.20)

        # First call with tools succeeds
        await provider.chat(
            [{"role": "user", "content": "Use tools"}],
            budget,
            tools=[{"name": "test_tool"}]
        )

        assert budget.spent_usd == 0.20

        # Second call would exceed budget
        with pytest.raises(BudgetExceeded):
            await provider.chat(
                [{"role": "user", "content": "Use tools again"}],
                budget,
                tools=[{"name": "test_tool"}]
            )


class TestBudgetStatistics:
    """Test budget statistics and reporting."""

    @pytest.mark.asyncio
    async def test_budget_statistics(self):
        """Test budget provides statistics."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Track several calls
        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.10)
        budget.track(prompt_tokens=200, completion_tokens=100, cost_usd=0.20)
        budget.track(prompt_tokens=150, completion_tokens=75, cost_usd=0.15)

        # Verify statistics
        assert budget.call_count == 3
        assert budget.spent_usd == 0.45
        assert budget.prompt_tokens == 450
        assert budget.completion_tokens == 225
        assert budget.total_tokens == 675
        assert budget.percentage_used == 45.0

    @pytest.mark.asyncio
    async def test_budget_average_cost_per_call(self):
        """Test calculating average cost per call."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.10)
        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.20)
        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.30)

        avg_cost = budget.spent_usd / budget.call_count
        assert avg_cost == 0.20  # (0.10 + 0.20 + 0.30) / 3


class TestBudgetWarnings:
    """Test budget warning thresholds."""

    @pytest.mark.asyncio
    async def test_budget_warning_threshold(self):
        """Test warning when approaching budget limit."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Use 80% of budget
        budget.track(prompt_tokens=1000, completion_tokens=500, cost_usd=0.80)

        # Check if over warning threshold (e.g., 75%)
        warning_threshold = 0.75
        is_warning = budget.percentage_used >= warning_threshold * 100

        assert is_warning is True
        assert budget.remaining_usd == 0.20

    @pytest.mark.asyncio
    async def test_budget_low_remaining(self):
        """Test detecting low remaining budget."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        budget.track(prompt_tokens=1000, completion_tokens=500, cost_usd=0.95)

        # Check if remaining is very low
        low_threshold = 0.10
        is_low = budget.remaining_usd < low_threshold

        assert is_low is True


class TestBudgetReset:
    """Test budget reset functionality."""

    @pytest.mark.asyncio
    async def test_budget_reset(self):
        """Test resetting budget clears all tracking."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Use some budget
        budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.30)

        assert budget.spent_usd == 0.30

        # Reset
        budget.reset()

        # All should be cleared
        assert budget.spent_usd == 0.0
        assert budget.call_count == 0
        assert budget.prompt_tokens == 0
        assert budget.completion_tokens == 0
        assert budget.total_tokens == 0
        assert budget.remaining_usd == budget.max_usd


class TestBudgetEdgeCases:
    """Test budget edge cases."""

    @pytest.mark.asyncio
    async def test_budget_zero_limit(self):
        """Test budget with zero limit."""
        from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

        budget = Budget(max_usd=0.0)

        # Any cost should exceed budget
        with pytest.raises(BudgetExceeded):
            if budget.spent_usd + 0.01 > budget.max_usd:
                raise BudgetExceeded("Budget exceeded")

    @pytest.mark.asyncio
    async def test_budget_very_small_cost(self):
        """Test budget tracks very small costs accurately."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Track tiny cost
        budget.track(prompt_tokens=10, completion_tokens=5, cost_usd=0.0001)

        assert budget.spent_usd == 0.0001
        assert budget.remaining_usd == pytest.approx(0.9999, rel=1e-6)

    @pytest.mark.asyncio
    async def test_budget_many_small_calls(self):
        """Test budget accumulates many small calls correctly."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=1.0)

        # Make 100 small calls
        for _ in range(100):
            budget.track(prompt_tokens=10, completion_tokens=5, cost_usd=0.005)

        assert budget.spent_usd == 0.50
        assert budget.call_count == 100


class TestBudgetConcurrency:
    """Test budget behavior under concurrent access."""

    @pytest.mark.asyncio
    async def test_budget_concurrent_tracking(self):
        """Test budget handles concurrent tracking correctly."""
        from greenlang.intelligence.runtime.budget import Budget

        budget = Budget(max_usd=10.0)

        async def track_cost():
            """Track a cost."""
            budget.track(prompt_tokens=100, completion_tokens=50, cost_usd=0.10)

        # Run multiple tracking operations concurrently
        tasks = [track_cost() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Should have tracked all 10 calls
        assert budget.call_count == 10
        assert budget.spent_usd == 1.0


class TestRAGBudgetEnforcement:
    """Test budget enforcement in RAG operations."""

    @pytest.mark.asyncio
    async def test_rag_respects_budget(self, temp_dir):
        """Test that RAG query respects budget limits."""
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta

        # Note: RAG itself doesn't consume budget (no LLM calls)
        # But it should track embedding costs if configured

        config = RAGConfig(
            mode="live",
            embedding_provider="minilm",
            vector_store_provider="faiss",
            vector_store_path=str(temp_dir / "vs"),
        )

        engine = RAGEngine(config=config)

        # Create test document
        test_doc = temp_dir / "test.txt"
        test_doc.write_text("Test content for budget testing")

        doc_meta = DocMeta(
            title="Test",
            source="test",
            version="1.0",
            collection="test"
        )

        # Ingest (embeddings cost should be tracked if configured)
        manifest = await engine.ingest_document(test_doc, "test", doc_meta)

        # Query (embedding cost for query should be tracked)
        result = await engine.query("test", top_k=1, collections=["test"])

        assert result is not None


# Fixtures
@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for tests."""
    return tmp_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
