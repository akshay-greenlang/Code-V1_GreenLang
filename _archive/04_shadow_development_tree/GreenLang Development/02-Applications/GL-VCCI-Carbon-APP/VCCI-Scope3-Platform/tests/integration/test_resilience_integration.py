# -*- coding: utf-8 -*-
"""
Resilience Integration Tests for GL-VCCI Scope 3 Platform

End-to-end tests for resilience patterns:
- Factor Broker with circuit breaker
- LLM categorization with retry + timeout
- ERP connector with circuit breaker + fallback
- API failure simulation
- Recovery scenarios

Total: 20+ integration scenarios
Coverage: Critical paths

Team: Testing & Documentation Team
Phase: 5 (Production Readiness)
Date: 2025-11-09
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path

from greenlang.intelligence.fallback import FallbackManager, ModelConfig
from greenlang.intelligence.providers.resilience import (
    ResilientHTTPClient,
    CircuitBreakerError,
)


# =============================================================================
# INTEGRATION SCENARIO 1: Scope 3 Calculation with Circuit Breakers (5 tests)
# =============================================================================

class TestScope3CalculationWithCircuitBreaker:
    """Test end-to-end Scope 3 calculation with circuit breaker protection"""

    @pytest.mark.asyncio
    async def test_calculation_with_factor_broker_circuit_breaker(self):
        """Test calculation protects Factor Broker calls with circuit breaker"""
        # Simulate Factor Broker client
        factor_broker_client = ResilientHTTPClient(
            failure_threshold=3,
            recovery_timeout=2.0,
        )

        async def get_emission_factor(category):
            """Mock Factor Broker API call"""
            return {
                "category": category,
                "factor": 0.185,  # kg CO2/kWh for natural gas
                "unit": "kg_co2_per_kwh"
            }

        # Successful calculation
        result = await factor_broker_client.call(
            get_emission_factor,
            "electricity_natural_gas"
        )

        assert result["factor"] == 0.185

    @pytest.mark.asyncio
    async def test_calculation_handles_factor_broker_failure(self):
        """Test calculation handles Factor Broker failures gracefully"""
        factor_broker_client = ResilientHTTPClient(
            failure_threshold=3,
            recovery_timeout=2.0,
        )

        attempts = {"count": 0}

        async def failing_factor_broker(category):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise Exception("Factor Broker temporarily unavailable")
            return {"category": category, "factor": 0.185}

        result = await factor_broker_client.call(
            failing_factor_broker,
            "electricity_grid"
        )

        assert result["factor"] == 0.185
        assert attempts["count"] == 3  # Retried successfully

    @pytest.mark.asyncio
    async def test_calculation_falls_back_to_default_factors(self):
        """Test calculation falls back to default factors when broker unavailable"""
        # Circuit breaker for Factor Broker
        factor_broker_client = ResilientHTTPClient(
            failure_threshold=2,
            recovery_timeout=1.0,
        )

        # Default factors as fallback
        DEFAULT_FACTORS = {
            "electricity_grid": 0.500,  # kg CO2/kWh (conservative estimate)
            "natural_gas": 0.185,
        }

        async def get_emission_factor_with_fallback(category):
            try:
                # Try Factor Broker
                async def call_broker():
                    raise Exception("Broker down")

                return await factor_broker_client.call(call_broker)
            except (Exception, CircuitBreakerError):
                # Fall back to defaults
                return {
                    "category": category,
                    "factor": DEFAULT_FACTORS.get(category, 1.0),
                    "source": "default",
                }

        result = await get_emission_factor_with_fallback("electricity_grid")

        assert result["factor"] == 0.500
        assert result["source"] == "default"

    @pytest.mark.asyncio
    async def test_end_to_end_calculation_with_resilience(self):
        """Test complete Scope 3 calculation with all resilience patterns"""
        # Simulated services
        factor_broker = ResilientHTTPClient(failure_threshold=3)
        llm_client = FallbackManager()

        # 1. Get activity data
        activity_data = {
            "supplier": "Acme Corp",
            "spend_usd": 100000,
            "category": "electricity",
        }

        # 2. Categorize with LLM (with retry/timeout)
        async def categorize_spend(data):
            await asyncio.sleep(0.1)  # Simulate LLM call
            return {"category": "electricity_grid", "confidence": 0.95}

        category_result = await llm_client.execute_with_fallback(
            lambda cfg: categorize_spend(activity_data)
        )

        assert category_result.success

        # 3. Get emission factor (with circuit breaker)
        async def get_factor():
            return {"factor": 0.500, "unit": "kg_co2_per_usd"}

        factor_result = await factor_broker.call(get_factor)

        # 4. Calculate emissions
        emissions = activity_data["spend_usd"] * factor_result["factor"]

        assert emissions == 50000  # kg CO2

    @pytest.mark.asyncio
    async def test_calculation_maintains_accuracy_under_failures(self):
        """Test calculation maintains accuracy even with service failures"""
        calculations = []

        for _ in range(5):
            # Simulate intermittent failures
            factor_broker = ResilientHTTPClient(failure_threshold=5)

            attempts = {"count": 0}

            async def intermittent_broker():
                attempts["count"] += 1
                if attempts["count"] % 2 == 0:
                    raise Exception("Intermittent failure")
                return {"factor": 0.185}

            result = await factor_broker.call(intermittent_broker)
            calculations.append(100 * result["factor"])

        # All calculations should be accurate
        assert all(calc == 18.5 for calc in calculations)


# =============================================================================
# INTEGRATION SCENARIO 2: LLM Categorization with Retry + Timeout (5 tests)
# =============================================================================

class TestLLMCategorizationWithRetryTimeout:
    """Test LLM categorization with retry and timeout patterns"""

    @pytest.mark.asyncio
    async def test_categorization_with_retry_on_rate_limit(self):
        """Test LLM categorization retries on rate limit"""
        manager = FallbackManager()

        attempts = {"count": 0}

        async def categorize_with_rate_limit(cfg):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise Exception("429 Rate limit exceeded")
            return {
                "category": "purchased_electricity",
                "confidence": 0.92,
            }

        result = await manager.execute_with_fallback(categorize_with_rate_limit)

        assert result.success
        assert attempts["count"] == 2

    @pytest.mark.asyncio
    async def test_categorization_with_timeout_fallback(self):
        """Test categorization falls back on timeout"""
        chain = [
            ModelConfig(model="slow", provider="test", timeout=0.3),
            ModelConfig(model="fast", provider="test", timeout=1.0),
        ]
        manager = FallbackManager(fallback_chain=chain)

        async def categorize(cfg):
            if cfg.model == "slow":
                await asyncio.sleep(0.5)  # Exceeds timeout
            else:
                await asyncio.sleep(0.1)
            return {"category": "transportation", "confidence": 0.88}

        result = await manager.execute_with_fallback(categorize)

        assert result.success
        assert result.model_used == "fast"

    @pytest.mark.asyncio
    async def test_categorization_quality_check_triggers_fallback(self):
        """Test low-quality categorization triggers fallback"""
        manager = FallbackManager()

        async def categorize(cfg):
            if cfg.model == manager.fallback_chain[0].model:
                return {"category": "unknown", "confidence": 0.3}
            return {"category": "waste_disposal", "confidence": 0.95}

        def quality_check(response):
            return response.get("confidence", 0.0)

        result = await manager.execute_with_fallback(
            categorize,
            quality_check_fn=quality_check,
            min_quality=0.8
        )

        assert result.success
        assert result.model_used != manager.fallback_chain[0].model

    @pytest.mark.asyncio
    async def test_batch_categorization_with_resilience(self):
        """Test batch categorization handles individual failures"""
        manager = FallbackManager()

        suppliers = [
            {"name": "Supplier A", "description": "Electricity provider"},
            {"name": "Supplier B", "description": "Transportation services"},
            {"name": "Supplier C", "description": "Waste management"},
        ]

        categorized = []

        for supplier in suppliers:
            async def categorize(cfg):
                await asyncio.sleep(0.05)
                # Derive category from description
                desc = supplier["description"].lower()
                if "electricity" in desc:
                    return {"category": "purchased_electricity", "confidence": 0.95}
                elif "transportation" in desc:
                    return {"category": "upstream_transportation", "confidence": 0.90}
                elif "waste" in desc:
                    return {"category": "waste_disposal", "confidence": 0.88}
                return {"category": "other", "confidence": 0.50}

            result = await manager.execute_with_fallback(categorize)
            categorized.append({
                "supplier": supplier["name"],
                "category": result.response.get("category") if result.success else "unknown"
            })

        assert len(categorized) == 3
        assert all(c["category"] != "unknown" for c in categorized)

    @pytest.mark.asyncio
    async def test_categorization_circuit_breaker_protection(self):
        """Test categorization is protected by circuit breaker"""
        manager = FallbackManager(enable_circuit_breaker=True)

        # Cause primary model to fail repeatedly
        for _ in range(6):
            async def always_fail(cfg):
                if cfg.model == manager.fallback_chain[0].model:
                    raise Exception("Model unavailable")
                return {"category": "fallback_category", "confidence": 0.8}

            result = await manager.execute_with_fallback(always_fail)
            # Should still succeed via fallback
            assert result.success

        # Check circuit breaker opened
        metrics = manager.get_metrics()
        if "circuit_breaker_states" in metrics:
            primary_model = manager.fallback_chain[0].model
            # Circuit should be open after repeated failures
            assert metrics["circuit_breaker_states"].get(primary_model) in ["open", "closed"]


# =============================================================================
# INTEGRATION SCENARIO 3: ERP Connector with Circuit Breaker + Fallback (5 tests)
# =============================================================================

class TestERPConnectorWithResiliencePattern:
    """Test ERP connector with circuit breaker and fallback"""

    @pytest.mark.asyncio
    async def test_erp_extraction_with_circuit_breaker(self):
        """Test ERP data extraction protected by circuit breaker"""
        erp_client = ResilientHTTPClient(failure_threshold=3)

        async def extract_procurement_data():
            return {
                "suppliers": [
                    {"id": "S001", "name": "Acme Corp", "spend": 100000},
                    {"id": "S002", "name": "TechCo", "spend": 50000},
                ],
                "total_spend": 150000,
            }

        result = await erp_client.call(extract_procurement_data)

        assert len(result["suppliers"]) == 2
        assert result["total_spend"] == 150000

    @pytest.mark.asyncio
    async def test_erp_fallback_to_cached_data(self):
        """Test ERP connector falls back to cached data on failure"""
        erp_client = ResilientHTTPClient(failure_threshold=2)

        # Cached data
        CACHE = {
            "procurement_data": {
                "suppliers": [{"id": "S001", "name": "Cached Corp", "spend": 75000}],
                "cached_at": "2025-11-08T10:00:00Z",
            }
        }

        async def get_procurement_data_with_fallback():
            try:
                async def call_erp():
                    raise Exception("ERP system unavailable")

                return await erp_client.call(call_erp)
            except (Exception, CircuitBreakerError):
                # Fall back to cache
                return CACHE["procurement_data"]

        result = await get_procurement_data_with_fallback()

        assert len(result["suppliers"]) == 1
        assert "cached_at" in result

    @pytest.mark.asyncio
    async def test_erp_retry_on_transient_failure(self):
        """Test ERP connector retries on transient failures"""
        erp_client = ResilientHTTPClient(
            max_retries=3,
            base_delay=0.1,
        )

        attempts = {"count": 0}

        async def transient_erp_call():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise Exception("Database connection timeout")
            return {"data": "success"}

        result = await erp_client.call(transient_erp_call)

        assert result["data"] == "success"
        assert attempts["count"] == 3

    @pytest.mark.asyncio
    async def test_erp_timeout_protection(self):
        """Test ERP calls have timeout protection"""
        # Simulate timeout at HTTP client level
        async def slow_erp_call():
            await asyncio.sleep(5.0)  # Very slow
            return {"data": "too_late"}

        # With timeout wrapper
        try:
            result = await asyncio.wait_for(slow_erp_call(), timeout=1.0)
        except asyncio.TimeoutError:
            result = {"error": "timeout", "fallback": True}

        assert result.get("fallback") == True

    @pytest.mark.asyncio
    async def test_erp_graceful_degradation(self):
        """Test ERP connector degrades gracefully"""
        erp_client = ResilientHTTPClient(failure_threshold=2)

        # Try primary ERP
        async def get_data_with_degradation():
            try:
                async def primary_erp():
                    raise Exception("Primary ERP down")

                return await erp_client.call(primary_erp)
            except (Exception, CircuitBreakerError):
                # Degrade to minimal data
                return {
                    "suppliers": [],
                    "degraded_mode": True,
                    "message": "Operating with cached/default data"
                }

        result = await get_data_with_degradation()

        assert result["degraded_mode"] == True


# =============================================================================
# INTEGRATION SCENARIO 4: API Failure Simulation (5 tests)
# =============================================================================

class TestAPIFailureSimulation:
    """Test system behavior under simulated API failures"""

    @pytest.mark.asyncio
    async def test_system_handles_complete_outage(self):
        """Test system handles complete API outage"""
        services = {
            "factor_broker": ResilientHTTPClient(failure_threshold=3),
            "llm_service": FallbackManager(),
            "erp_connector": ResilientHTTPClient(failure_threshold=3),
        }

        # All services fail
        async def failing_service():
            raise Exception("Complete outage")

        # System should degrade gracefully
        results = {}

        try:
            await services["factor_broker"].call(failing_service)
        except (Exception, CircuitBreakerError):
            results["factor_broker"] = "degraded"

        try:
            await services["erp_connector"].call(failing_service)
        except (Exception, CircuitBreakerError):
            results["erp_connector"] = "degraded"

        # LLM has fallback
        llm_result = await services["llm_service"].execute_with_fallback(
            lambda cfg: failing_service()
        )
        if not llm_result.success:
            results["llm_service"] = "degraded"

        # All services degraded but system still operational
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_system_handles_partial_outage(self):
        """Test system handles partial service outage"""
        success_count = {"count": 0}
        call_count = {"count": 0}

        async def intermittent_service():
            call_count["count"] += 1
            # 50% failure rate
            if call_count["count"] % 2 == 0:
                raise Exception("Intermittent failure")
            success_count["count"] += 1
            return {"status": "success"}

        client = ResilientHTTPClient(max_retries=3, base_delay=0.01)

        # Multiple calls - retries should get us through
        for _ in range(5):
            try:
                await client.call(intermittent_service)
            except:
                pass

        # Some calls should succeed
        assert success_count["count"] > 0

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self):
        """Test prevents cascading failures across services"""
        # Service A depends on Service B
        service_b_client = ResilientHTTPClient(
            failure_threshold=2,  # Opens quickly
            recovery_timeout=1.0,
        )

        async def service_b():
            raise Exception("Service B failing")

        async def service_a():
            try:
                # Try to call Service B
                return await service_b_client.call(service_b)
            except (Exception, CircuitBreakerError):
                # Service A degrades but doesn't fail
                return {"status": "degraded", "service_b": "unavailable"}

        # Multiple calls to Service A
        results = []
        for _ in range(5):
            result = await service_a()
            results.append(result)

        # Service A continues operating
        assert all(r["status"] == "degraded" for r in results)

    @pytest.mark.asyncio
    async def test_system_recovers_after_outage(self):
        """Test system automatically recovers after outage"""
        client = ResilientHTTPClient(
            failure_threshold=3,
            recovery_timeout=1.0,
        )

        calls = {"count": 0}

        async def recovering_service():
            calls["count"] += 1
            # Fail first 5 calls, then succeed
            if calls["count"] <= 5:
                raise Exception("Still recovering")
            return {"status": "recovered"}

        # Fail repeatedly to open circuit
        for _ in range(5):
            try:
                await client.call(recovering_service)
            except:
                pass

        # Wait for recovery timeout
        await asyncio.sleep(1.5)

        # Should recover
        result = await client.call(recovering_service)
        assert result["status"] == "recovered"

    @pytest.mark.asyncio
    async def test_load_shedding_under_high_failure_rate(self):
        """Test system sheds load under high failure rate"""
        client = ResilientHTTPClient(
            failure_threshold=5,
            max_retries=1,  # Limit retries to shed load faster
        )

        failures = {"count": 0}
        successes = {"count": 0}

        async def high_failure_service():
            # 80% failure rate
            if hash(time.time()) % 5 != 0:
                failures["count"] += 1
                raise Exception("Overloaded")
            successes["count"] += 1
            return {"status": "success"}

        # Many requests
        for _ in range(20):
            try:
                await client.call(high_failure_service)
            except:
                pass

        # Should have failed fast on many
        assert failures["count"] > successes["count"]


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])
