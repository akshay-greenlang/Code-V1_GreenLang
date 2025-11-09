"""
Tests for CalculatorAgent
"""

import pytest
from greenlang.agents.templates import CalculatorAgent


class TestCalculatorAgent:
    """Test CalculatorAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        agent = CalculatorAgent()
        assert agent is not None
        assert len(agent.formulas) == 0

    @pytest.mark.asyncio
    async def test_register_formula(self):
        """Test registering a formula."""
        agent = CalculatorAgent()

        def add(a, b):
            return a + b

        agent.register_formula("add", add, required_inputs=["a", "b"])

        assert "add" in agent.formulas

    @pytest.mark.asyncio
    async def test_calculate_success(self):
        """Test successful calculation."""
        agent = CalculatorAgent()

        def multiply(a, b):
            return a * b

        agent.register_formula("multiply", multiply, required_inputs=["a", "b"])

        result = await agent.calculate(
            formula_name="multiply",
            inputs={"a": 5, "b": 3}
        )

        assert result.success is True
        assert result.value == 15

    @pytest.mark.asyncio
    async def test_calculate_missing_formula(self):
        """Test calculation with missing formula."""
        agent = CalculatorAgent()

        result = await agent.calculate(
            formula_name="nonexistent",
            inputs={"a": 5}
        )

        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_calculate_missing_inputs(self):
        """Test calculation with missing inputs."""
        agent = CalculatorAgent()

        def add(a, b):
            return a + b

        agent.register_formula("add", add, required_inputs=["a", "b"])

        result = await agent.calculate(
            formula_name="add",
            inputs={"a": 5}  # Missing 'b'
        )

        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_batch_calculate(self):
        """Test batch calculation."""
        agent = CalculatorAgent()

        def square(x):
            return x ** 2

        agent.register_formula("square", square, required_inputs=["x"])

        inputs_list = [
            {"x": 2},
            {"x": 3},
            {"x": 4},
        ]

        results = await agent.batch_calculate(
            formula_name="square",
            inputs_list=inputs_list
        )

        assert len(results) == 3
        assert results[0].value == 4
        assert results[1].value == 9
        assert results[2].value == 16

    @pytest.mark.asyncio
    async def test_batch_calculate_parallel(self):
        """Test parallel batch calculation."""
        agent = CalculatorAgent()

        def add(a, b):
            return a + b

        agent.register_formula("add", add, required_inputs=["a", "b"])

        inputs_list = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
        ]

        results = await agent.batch_calculate(
            formula_name="add",
            inputs_list=inputs_list,
            parallel=True
        )

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_get_stats(self):
        """Test getting agent statistics."""
        agent = CalculatorAgent()
        stats = agent.get_stats()

        assert "total_calculations" in stats
        assert "cache_size" in stats

    def test_clear_cache(self):
        """Test clearing cache."""
        agent = CalculatorAgent()
        agent.clear_cache()
        assert len(agent._cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
