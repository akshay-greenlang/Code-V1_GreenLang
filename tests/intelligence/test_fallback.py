"""
Tests for Model Fallback Chains

Tests:
- Fallback chain execution
- Circuit breaker pattern
- Retry logic
- Quality checks
- Failure simulation
"""

import asyncio

import pytest

from greenlang.intelligence.fallback import (
    CircuitBreaker,
    CircuitState,
    DEFAULT_FALLBACK_CHAIN,
    FallbackManager,
    FallbackReason,
    ModelConfig,
)


class TestCircuitBreaker:
    """Test circuit breaker"""

    @pytest.fixture
    def circuit(self):
        """Create circuit breaker"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2,
        )

    def test_init(self, circuit):
        """Test initialization"""
        assert circuit.state == CircuitState.CLOSED
        assert circuit.failure_count == 0

    def test_record_success(self, circuit):
        """Test recording success"""
        circuit.record_success()
        assert circuit.failure_count == 0

    def test_record_failure_opens_circuit(self, circuit):
        """Test circuit opens after threshold failures"""
        for _ in range(3):
            circuit.record_failure()

        assert circuit.state == CircuitState.OPEN

    def test_can_execute_when_closed(self, circuit):
        """Test can execute when circuit is closed"""
        assert circuit.can_execute() is True

    def test_cannot_execute_when_open(self, circuit):
        """Test cannot execute when circuit is open"""
        # Force open
        for _ in range(3):
            circuit.record_failure()

        assert circuit.can_execute() is False

    @pytest.mark.asyncio
    async def test_recovery_to_half_open(self, circuit):
        """Test recovery to half-open state"""
        # Force open
        for _ in range(3):
            circuit.record_failure()

        assert circuit.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should transition to half-open
        assert circuit.can_execute() is True
        assert circuit.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self, circuit):
        """Test half-open closes after successes"""
        # Force to half-open
        circuit.state = CircuitState.HALF_OPEN

        # Record successes
        for _ in range(2):
            circuit.record_success()

        assert circuit.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self, circuit):
        """Test half-open reopens on failure"""
        # Force to half-open
        circuit.state = CircuitState.HALF_OPEN

        circuit.record_failure()

        assert circuit.state == CircuitState.OPEN


class TestFallbackManager:
    """Test fallback manager"""

    @pytest.fixture
    def manager(self):
        """Create fallback manager"""
        chain = [
            ModelConfig("gpt-4", "openai", max_retries=2, priority=0),
            ModelConfig("gpt-3.5-turbo", "openai", max_retries=2, priority=1),
        ]
        return FallbackManager(fallback_chain=chain)

    def test_init(self, manager):
        """Test initialization"""
        assert len(manager.fallback_chain) == 2
        assert manager.enable_circuit_breaker is True

    @pytest.mark.asyncio
    async def test_execute_success_first_model(self, manager):
        """Test successful execution with first model"""
        call_count = {"count": 0}

        async def execute_fn(config):
            call_count["count"] += 1
            return {"response": "success", "model": config.model}

        result = await manager.execute_with_fallback(execute_fn)

        assert result.success is True
        assert result.model_used == "gpt-4"
        assert result.fallback_count == 0
        assert call_count["count"] == 1

    @pytest.mark.asyncio
    async def test_execute_fallback(self, manager):
        """Test fallback to second model"""
        call_count = {"count": 0}

        async def execute_fn(config):
            call_count["count"] += 1
            if config.model == "gpt-4":
                raise Exception("Rate limit exceeded")
            return {"response": "success", "model": config.model}

        result = await manager.execute_with_fallback(execute_fn)

        assert result.success is True
        assert result.model_used == "gpt-3.5-turbo"
        assert result.fallback_count == 1
        assert call_count["count"] > 1

    @pytest.mark.asyncio
    async def test_execute_all_fail(self, manager):
        """Test when all models fail"""
        async def execute_fn(config):
            raise Exception("All models fail")

        result = await manager.execute_with_fallback(execute_fn)

        assert result.success is False
        assert result.model_used == "none"
        assert result.fallback_count == len(manager.fallback_chain) - 1

    @pytest.mark.asyncio
    async def test_retry_logic(self, manager):
        """Test retry logic"""
        attempt_counts = {"gpt-4": 0}

        async def execute_fn(config):
            attempt_counts[config.model] = attempt_counts.get(config.model, 0) + 1
            if config.model == "gpt-4" and attempt_counts[config.model] < 3:
                raise Exception("Temporary failure")
            return {"response": "success"}

        result = await manager.execute_with_fallback(execute_fn)

        # Should retry GPT-4 before fallback
        assert attempt_counts["gpt-4"] >= 2

    @pytest.mark.asyncio
    async def test_quality_check_triggers_fallback(self, manager):
        """Test quality check triggers fallback"""
        async def execute_fn(config):
            return {"response": "low quality", "model": config.model}

        def quality_check_fn(response):
            if response.get("model") == "gpt-4":
                return 0.5  # Low quality
            return 0.9  # Good quality

        result = await manager.execute_with_fallback(
            execute_fn,
            quality_check_fn=quality_check_fn,
            min_quality=0.8,
        )

        # Should fallback due to quality
        assert result.model_used == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_timeout_handling(self, manager):
        """Test timeout handling"""
        async def execute_fn(config):
            if config.model == "gpt-4":
                await asyncio.sleep(100)  # Timeout
            return {"response": "success"}

        result = await manager.execute_with_fallback(execute_fn)

        # Should fallback due to timeout
        assert result.model_used == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, manager):
        """Test circuit breaker prevents retries"""
        async def execute_fn(config):
            raise Exception("Always fails")

        # Trigger circuit breaker for GPT-4
        for _ in range(3):
            await manager.execute_with_fallback(execute_fn)

        # Check circuit state
        circuit = manager.circuit_breakers.get("gpt-4")
        assert circuit is not None
        # Circuit should be open after multiple failures

    def test_route_by_complexity(self, manager):
        """Test routing by query complexity"""
        # Short query -> cheap models first
        short_chain = manager.route_by_complexity("short")
        assert short_chain[0].model == "gpt-3.5-turbo"

        # Long query -> best models first
        long_query = "a" * 300
        long_chain = manager.route_by_complexity(long_query)
        assert long_chain[0].model == "gpt-4"

    def test_metrics_tracking(self, manager):
        """Test metrics tracking"""
        metrics = manager.get_metrics()

        assert "total_requests" in metrics
        assert "fallback_counts" in metrics
        assert "success_counts" in metrics


class TestModelConfig:
    """Test model configuration"""

    def test_avg_cost(self):
        """Test average cost calculation"""
        config = ModelConfig(
            "gpt-4",
            "openai",
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
        )

        assert config.avg_cost_per_1k == 0.045


class TestDefaultFallbackChain:
    """Test default fallback chain"""

    def test_chain_exists(self):
        """Test default chain is defined"""
        assert len(DEFAULT_FALLBACK_CHAIN) > 0

    def test_chain_ordering(self):
        """Test chain is ordered by priority"""
        priorities = [c.priority for c in DEFAULT_FALLBACK_CHAIN]
        assert priorities == sorted(priorities)


class TestFallbackIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_realistic_scenario(self):
        """Test realistic fallback scenario"""
        manager = FallbackManager()

        # Simulate API with rate limiting
        api_calls = {"count": 0}

        async def realistic_execute(config):
            api_calls["count"] += 1

            # Simulate rate limiting for first few calls
            if api_calls["count"] <= 2 and config.model == "gpt-4o":
                raise Exception("429 Rate limit exceeded")

            # Simulate occasional timeout
            if api_calls["count"] == 4:
                await asyncio.sleep(100)

            return {"response": f"Success with {config.model}"}

        result = await manager.execute_with_fallback(realistic_execute)

        assert result.success is True
        assert result.fallback_count >= 1

    @pytest.mark.asyncio
    async def test_cost_optimization(self):
        """Test cost optimization with fallback"""
        chain = [
            ModelConfig("gpt-4", "openai", cost_per_1k_input=0.03, cost_per_1k_output=0.06),
            ModelConfig("gpt-3.5-turbo", "openai", cost_per_1k_input=0.0005, cost_per_1k_output=0.0015),
        ]

        manager = FallbackManager(fallback_chain=chain)

        # GPT-4 fails, falls back to cheaper GPT-3.5
        async def execute_fn(config):
            if config.model == "gpt-4":
                raise Exception("GPT-4 unavailable")
            return {"response": "success"}

        result = await manager.execute_with_fallback(execute_fn)

        assert result.success is True
        assert result.model_used == "gpt-3.5-turbo"
        # Cost should be lower than if GPT-4 succeeded
        assert result.cost < 0.05


if __name__ == "__main__":
    """Run tests"""
    pytest.main([__file__, "-v", "-s"])
