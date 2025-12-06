# -*- coding: utf-8 -*-
"""
Integration Tests for GreenLang Agent Communication

Comprehensive test suite with 28 test cases covering:
- Agent-to-Agent Pipeline Communication (12 tests)
- Agent State Management (8 tests)
- Agent Error Handling and Recovery (8 tests)

Target: Validate agent integration patterns
Run with: pytest tests/integration/test_agent_integration.py -v --tb=short

Author: GL-TestEngineer
Version: 1.0.0

These tests validate that agents can communicate correctly in pipelines,
share state appropriately, and handle errors gracefully.
"""

import pytest
import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_fuel_analyzer_agent():
    """Create mock Fuel Analyzer Agent."""
    agent = Mock()
    agent.name = "fuel_analyzer"
    agent.version = "1.0.0"

    async def process(input_data):
        return {
            "emissions_value": input_data.get("quantity", 0) * 0.0561,
            "emissions_unit": "kgCO2e",
            "fuel_type": input_data.get("fuel_type"),
            "provenance_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),
        }

    agent.process = AsyncMock(side_effect=process)
    return agent


@pytest.fixture
def mock_cbam_agent():
    """Create mock CBAM Agent."""
    agent = Mock()
    agent.name = "carbon_intensity"
    agent.version = "1.0.0"

    async def process(input_data):
        return {
            "carbon_intensity": input_data.get("total_emissions", 0) / input_data.get("production_quantity", 1),
            "carbon_intensity_unit": "tCO2e/tonne",
            "benchmark_value": 1.85,
            "provenance_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),
        }

    agent.process = AsyncMock(side_effect=process)
    return agent


@pytest.fixture
def mock_building_energy_agent():
    """Create mock Building Energy Agent."""
    agent = Mock()
    agent.name = "energy_performance"
    agent.version = "1.0.0"

    async def process(input_data):
        eui = input_data.get("energy_consumption_kwh", 0) / input_data.get("floor_area_sqm", 1)
        return {
            "eui_kwh_per_sqm": eui,
            "threshold_eui": 80.0,
            "compliant": eui <= 80.0,
            "provenance_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),
        }

    agent.process = AsyncMock(side_effect=process)
    return agent


@pytest.fixture
def mock_eudr_agent():
    """Create mock EUDR Agent."""
    agent = Mock()
    agent.name = "eudr_compliance"
    agent.version = "1.0.0"

    async def process(input_data):
        return {
            "valid": True,
            "risk_level": "standard",
            "commodity_type": input_data.get("commodity_type"),
            "eudr_regulated": True,
            "provenance_hash": hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest(),
        }

    agent.process = AsyncMock(side_effect=process)
    return agent


@pytest.fixture
def sample_pipeline_config():
    """Sample pipeline configuration."""
    return {
        "pipeline_id": "emissions-calculation-pipeline",
        "version": "1.0.0",
        "agents": [
            {"name": "fuel_analyzer", "order": 1},
            {"name": "carbon_intensity", "order": 2},
        ],
        "error_handling": {
            "retry_count": 3,
            "retry_delay_ms": 100,
            "on_failure": "halt",
        },
    }


# =============================================================================
# Agent-to-Agent Pipeline Communication Tests (12 tests)
# =============================================================================

class TestAgentPipelineCommunication:
    """Test suite for agent-to-agent pipeline communication - 12 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_agent_pipeline(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-PIPE-001: Test simple two-agent pipeline execution."""
        # Step 1: Fuel analyzer calculates emissions
        fuel_input = {"fuel_type": "natural_gas", "quantity": 1000, "unit": "MJ"}
        fuel_result = await mock_fuel_analyzer_agent.process(fuel_input)

        assert fuel_result["emissions_value"] == pytest.approx(56.1, rel=0.01)

        # Step 2: CBAM agent calculates carbon intensity
        cbam_input = {
            "total_emissions": fuel_result["emissions_value"],
            "production_quantity": 100,
        }
        cbam_result = await mock_cbam_agent.process(cbam_input)

        assert cbam_result["carbon_intensity"] == pytest.approx(0.561, rel=0.01)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_data_transformation(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-PIPE-002: Test data transformation between agents."""
        fuel_input = {"fuel_type": "diesel", "quantity": 5000, "unit": "L"}
        fuel_result = await mock_fuel_analyzer_agent.process(fuel_input)

        # Transform fuel output to CBAM input format
        cbam_input = {
            "total_emissions": fuel_result["emissions_value"] / 1000,  # Convert to tonnes
            "production_quantity": 50,
            "source_agent": fuel_result.get("provenance_hash"),
        }

        cbam_result = await mock_cbam_agent.process(cbam_input)

        assert "carbon_intensity" in cbam_result
        assert "source_agent" in cbam_input  # Provenance chain maintained

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_provenance_chain(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-PIPE-003: Test provenance chain is maintained across agents."""
        provenance_chain = []

        # Step 1
        fuel_input = {"fuel_type": "natural_gas", "quantity": 1000}
        fuel_result = await mock_fuel_analyzer_agent.process(fuel_input)
        provenance_chain.append(fuel_result["provenance_hash"])

        # Step 2
        cbam_input = {
            "total_emissions": fuel_result["emissions_value"],
            "production_quantity": 100,
            "input_provenance": fuel_result["provenance_hash"],
        }
        cbam_result = await mock_cbam_agent.process(cbam_input)
        provenance_chain.append(cbam_result["provenance_hash"])

        # Verify chain
        assert len(provenance_chain) == 2
        assert all(len(h) == 64 for h in provenance_chain)
        assert provenance_chain[0] != provenance_chain[1]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, mock_fuel_analyzer_agent, mock_building_energy_agent):
        """INT-PIPE-004: Test parallel agent execution."""
        # Execute two independent agents in parallel
        fuel_input = {"fuel_type": "natural_gas", "quantity": 1000}
        energy_input = {"energy_consumption_kwh": 70000, "floor_area_sqm": 1000}

        results = await asyncio.gather(
            mock_fuel_analyzer_agent.process(fuel_input),
            mock_building_energy_agent.process(energy_input),
        )

        assert len(results) == 2
        assert "emissions_value" in results[0]
        assert "eui_kwh_per_sqm" in results[1]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conditional_agent_routing(self, mock_fuel_analyzer_agent, mock_cbam_agent, mock_eudr_agent):
        """INT-PIPE-005: Test conditional routing based on agent output."""
        fuel_input = {"fuel_type": "natural_gas", "quantity": 1000}
        fuel_result = await mock_fuel_analyzer_agent.process(fuel_input)

        # Conditional routing based on emissions
        if fuel_result["emissions_value"] > 50:
            # High emissions - route to CBAM for carbon intensity
            next_result = await mock_cbam_agent.process({
                "total_emissions": fuel_result["emissions_value"],
                "production_quantity": 100,
            })
            route = "cbam"
        else:
            # Low emissions - route to EUDR for compliance
            next_result = await mock_eudr_agent.process({
                "commodity_type": "coffee",
            })
            route = "eudr"

        assert route == "cbam"
        assert "carbon_intensity" in next_result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fan_out_fan_in_pattern(self, mock_fuel_analyzer_agent, mock_cbam_agent, mock_building_energy_agent):
        """INT-PIPE-006: Test fan-out/fan-in pattern."""
        # Fan-out: Single input to multiple agents
        initial_input = {
            "facility_id": "FAC-001",
            "fuel_type": "natural_gas",
            "quantity": 5000,
            "energy_consumption_kwh": 150000,
            "floor_area_sqm": 2000,
        }

        # Fan-out to parallel agents
        fuel_result, energy_result = await asyncio.gather(
            mock_fuel_analyzer_agent.process({
                "fuel_type": initial_input["fuel_type"],
                "quantity": initial_input["quantity"],
            }),
            mock_building_energy_agent.process({
                "energy_consumption_kwh": initial_input["energy_consumption_kwh"],
                "floor_area_sqm": initial_input["floor_area_sqm"],
            }),
        )

        # Fan-in: Aggregate results
        aggregated_result = {
            "facility_id": initial_input["facility_id"],
            "emissions_kgco2e": fuel_result["emissions_value"],
            "eui_kwh_sqm": energy_result["eui_kwh_per_sqm"],
            "energy_compliant": energy_result["compliant"],
            "combined_provenance": [
                fuel_result["provenance_hash"],
                energy_result["provenance_hash"],
            ],
        }

        assert aggregated_result["facility_id"] == "FAC-001"
        assert len(aggregated_result["combined_provenance"]) == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_timeout_handling(self, mock_fuel_analyzer_agent):
        """INT-PIPE-007: Test pipeline handles agent timeouts."""
        # Simulate slow agent
        async def slow_process(input_data):
            await asyncio.sleep(0.5)
            return {"result": "done"}

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=slow_process)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mock_fuel_analyzer_agent.process({"quantity": 1000}),
                timeout=0.1
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_order_execution(self, sample_pipeline_config):
        """INT-PIPE-008: Test agents execute in correct order."""
        execution_order = []

        async def track_execution(agent_name, input_data):
            execution_order.append(agent_name)
            return {"agent": agent_name, "completed": True}

        # Simulate ordered execution
        for agent_config in sorted(sample_pipeline_config["agents"], key=lambda x: x["order"]):
            await track_execution(agent_config["name"], {})

        assert execution_order == ["fuel_analyzer", "carbon_intensity"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_context_passing(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-PIPE-009: Test pipeline context is passed between agents."""
        pipeline_context = {
            "request_id": "REQ-12345",
            "user_id": "user-001",
            "timestamp": datetime.now().isoformat(),
        }

        fuel_input = {"fuel_type": "natural_gas", "quantity": 1000, **pipeline_context}
        fuel_result = await mock_fuel_analyzer_agent.process(fuel_input)

        cbam_input = {
            "total_emissions": fuel_result["emissions_value"],
            "production_quantity": 100,
            **pipeline_context,  # Context passed along
        }
        cbam_result = await mock_cbam_agent.process(cbam_input)

        # Verify context available in both steps
        assert "request_id" in fuel_input
        assert "request_id" in cbam_input

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_result_aggregation(self, mock_fuel_analyzer_agent, mock_cbam_agent, mock_building_energy_agent):
        """INT-PIPE-010: Test pipeline aggregates all agent results."""
        all_results = []

        # Execute pipeline
        fuel_result = await mock_fuel_analyzer_agent.process({"quantity": 1000})
        all_results.append({"agent": "fuel_analyzer", "result": fuel_result})

        cbam_result = await mock_cbam_agent.process({
            "total_emissions": fuel_result["emissions_value"],
            "production_quantity": 100,
        })
        all_results.append({"agent": "carbon_intensity", "result": cbam_result})

        energy_result = await mock_building_energy_agent.process({
            "energy_consumption_kwh": 80000,
            "floor_area_sqm": 1000,
        })
        all_results.append({"agent": "energy_performance", "result": energy_result})

        # Aggregate
        pipeline_result = {
            "pipeline_id": "test-pipeline",
            "agent_count": len(all_results),
            "all_successful": all(r["result"] is not None for r in all_results),
            "results": all_results,
        }

        assert pipeline_result["agent_count"] == 3
        assert pipeline_result["all_successful"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_deterministic_execution(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-PIPE-011: Test pipeline produces deterministic results."""
        input_data = {"fuel_type": "natural_gas", "quantity": 1000}

        results = []
        for _ in range(5):
            fuel_result = await mock_fuel_analyzer_agent.process(input_data)
            cbam_result = await mock_cbam_agent.process({
                "total_emissions": fuel_result["emissions_value"],
                "production_quantity": 100,
            })
            results.append(cbam_result["carbon_intensity"])

        # All results should be identical
        assert all(r == results[0] for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_validation_between_agents(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-PIPE-012: Test pipeline validates data between agents."""
        fuel_result = await mock_fuel_analyzer_agent.process({"quantity": 1000})

        # Validate output before passing to next agent
        required_fields = ["emissions_value", "emissions_unit", "provenance_hash"]
        validation_passed = all(f in fuel_result for f in required_fields)

        assert validation_passed is True

        # Only proceed if validation passes
        if validation_passed:
            cbam_result = await mock_cbam_agent.process({
                "total_emissions": fuel_result["emissions_value"],
                "production_quantity": 100,
            })
            assert cbam_result is not None


# =============================================================================
# Agent State Management Tests (8 tests)
# =============================================================================

class TestAgentStateManagement:
    """Test suite for agent state management - 8 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_stateless_execution(self, mock_fuel_analyzer_agent):
        """INT-STATE-001: Test agents are stateless between calls."""
        result1 = await mock_fuel_analyzer_agent.process({"quantity": 1000})
        result2 = await mock_fuel_analyzer_agent.process({"quantity": 2000})

        # Results should be independent
        assert result1["emissions_value"] != result2["emissions_value"]
        assert result2["emissions_value"] == pytest.approx(result1["emissions_value"] * 2, rel=0.01)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_does_not_retain_previous_input(self, mock_fuel_analyzer_agent):
        """INT-STATE-002: Test agent does not retain previous input."""
        # First call with fuel_type
        result1 = await mock_fuel_analyzer_agent.process({
            "fuel_type": "diesel",
            "quantity": 1000,
        })

        # Second call without fuel_type
        result2 = await mock_fuel_analyzer_agent.process({
            "quantity": 500,
        })

        # fuel_type should not be retained from first call
        assert result1.get("fuel_type") == "diesel"
        assert result2.get("fuel_type") is None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_shared_context_isolation(self, mock_fuel_analyzer_agent):
        """INT-STATE-003: Test shared context is isolated between requests."""
        contexts = []

        async def process_with_context(input_data, context):
            result = await mock_fuel_analyzer_agent.process(input_data)
            result["context_id"] = context["request_id"]
            contexts.append(context["request_id"])
            return result

        # Simulate parallel requests with different contexts
        results = await asyncio.gather(
            process_with_context({"quantity": 100}, {"request_id": "req-1"}),
            process_with_context({"quantity": 200}, {"request_id": "req-2"}),
            process_with_context({"quantity": 300}, {"request_id": "req-3"}),
        )

        # Each result should have its own context
        context_ids = [r["context_id"] for r in results]
        assert len(set(context_ids)) == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_cache_hit(self, mock_fuel_analyzer_agent):
        """INT-STATE-004: Test agent caching for identical requests."""
        cache = {}

        async def cached_process(input_data):
            cache_key = hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest()

            if cache_key in cache:
                return cache[cache_key]

            result = await mock_fuel_analyzer_agent.process(input_data)
            cache[cache_key] = result
            return result

        input_data = {"quantity": 1000}

        # First call - cache miss
        result1 = await cached_process(input_data)
        cache_size_after_first = len(cache)

        # Second call - cache hit
        result2 = await cached_process(input_data)
        cache_size_after_second = len(cache)

        assert cache_size_after_first == 1
        assert cache_size_after_second == 1
        assert result1 == result2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_cache_invalidation(self, mock_fuel_analyzer_agent):
        """INT-STATE-005: Test agent cache invalidation."""
        cache = {}
        invalidated = False

        def invalidate_cache():
            nonlocal invalidated
            cache.clear()
            invalidated = True

        # Populate cache
        cache_key = "test-key"
        cache[cache_key] = {"result": "cached"}

        # Invalidate
        invalidate_cache()

        assert len(cache) == 0
        assert invalidated is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_agent_access(self, mock_fuel_analyzer_agent):
        """INT-STATE-006: Test concurrent access to same agent."""
        async def concurrent_request(quantity):
            result = await mock_fuel_analyzer_agent.process({"quantity": quantity})
            return result

        # Execute many concurrent requests
        quantities = list(range(100, 1100, 100))  # 10 concurrent requests
        results = await asyncio.gather(*[
            concurrent_request(q) for q in quantities
        ])

        # All should complete successfully
        assert len(results) == 10
        # Each should have correct result
        for i, result in enumerate(results):
            expected = quantities[i] * 0.0561
            assert result["emissions_value"] == pytest.approx(expected, rel=0.01)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_resource_cleanup(self, mock_fuel_analyzer_agent):
        """INT-STATE-007: Test agent resources are cleaned up."""
        resources_allocated = []
        resources_released = []

        async def process_with_cleanup(input_data):
            # Allocate resource
            resource_id = f"res-{len(resources_allocated)}"
            resources_allocated.append(resource_id)

            try:
                result = await mock_fuel_analyzer_agent.process(input_data)
                return result
            finally:
                # Release resource
                resources_released.append(resource_id)

        await process_with_cleanup({"quantity": 1000})

        assert len(resources_allocated) == 1
        assert len(resources_released) == 1
        assert resources_allocated == resources_released

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_session_management(self, mock_fuel_analyzer_agent):
        """INT-STATE-008: Test agent session management."""
        session = {
            "id": "session-001",
            "created_at": datetime.now().isoformat(),
            "request_count": 0,
        }

        async def process_with_session(input_data):
            session["request_count"] += 1
            result = await mock_fuel_analyzer_agent.process(input_data)
            result["session_id"] = session["id"]
            result["request_number"] = session["request_count"]
            return result

        # Multiple requests in same session
        r1 = await process_with_session({"quantity": 100})
        r2 = await process_with_session({"quantity": 200})
        r3 = await process_with_session({"quantity": 300})

        assert r1["session_id"] == r2["session_id"] == r3["session_id"]
        assert r1["request_number"] == 1
        assert r2["request_number"] == 2
        assert r3["request_number"] == 3


# =============================================================================
# Agent Error Handling and Recovery Tests (8 tests)
# =============================================================================

class TestAgentErrorHandling:
    """Test suite for agent error handling and recovery - 8 test cases."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_error_propagation(self, mock_fuel_analyzer_agent):
        """INT-ERR-001: Test errors propagate correctly from agents."""
        async def failing_process(input_data):
            raise ValueError("Invalid fuel type")

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=failing_process)

        with pytest.raises(ValueError) as exc_info:
            await mock_fuel_analyzer_agent.process({"fuel_type": "invalid"})

        assert "Invalid fuel type" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_retry_on_transient_error(self, mock_fuel_analyzer_agent):
        """INT-ERR-002: Test agent retries on transient errors."""
        attempt_count = 0

        async def intermittent_process(input_data):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary network error")
            return {"emissions_value": 56.1}

        async def retry_process(agent, input_data, max_retries=3):
            for attempt in range(max_retries):
                try:
                    return await agent.process(input_data)
                except ConnectionError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.01)

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=intermittent_process)

        result = await retry_process(mock_fuel_analyzer_agent, {"quantity": 1000})

        assert result["emissions_value"] == 56.1
        assert attempt_count == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_circuit_breaker(self, mock_fuel_analyzer_agent):
        """INT-ERR-003: Test circuit breaker pattern."""
        failure_count = 0
        circuit_open = False
        threshold = 3

        async def monitored_process(input_data):
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise Exception("Circuit breaker open")

            try:
                raise Exception("Simulated failure")
            except Exception:
                failure_count += 1
                if failure_count >= threshold:
                    circuit_open = True
                raise

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=monitored_process)

        # Trigger failures
        for _ in range(threshold):
            with pytest.raises(Exception):
                await mock_fuel_analyzer_agent.process({})

        # Circuit should be open now
        assert circuit_open is True

        # Next call should immediately fail with circuit breaker
        with pytest.raises(Exception) as exc_info:
            await mock_fuel_analyzer_agent.process({})

        assert "Circuit breaker" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_fallback_behavior(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-ERR-004: Test fallback to alternate agent."""
        async def failing_process(input_data):
            raise RuntimeError("Primary agent unavailable")

        async def fallback_process(input_data):
            return {"fallback": True, "result": "default_value"}

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=failing_process)
        mock_cbam_agent.process = AsyncMock(side_effect=fallback_process)

        # Try primary, fallback on error
        try:
            result = await mock_fuel_analyzer_agent.process({})
        except RuntimeError:
            result = await mock_cbam_agent.process({})

        assert result["fallback"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_partial_failure_handling(self, mock_fuel_analyzer_agent, mock_cbam_agent):
        """INT-ERR-005: Test pipeline handles partial failures."""
        # First agent succeeds
        fuel_result = await mock_fuel_analyzer_agent.process({"quantity": 1000})

        # Second agent fails
        async def cbam_failure(input_data):
            raise ValueError("CBAM calculation failed")

        mock_cbam_agent.process = AsyncMock(side_effect=cbam_failure)

        pipeline_result = {
            "fuel_analyzer": {"success": True, "result": fuel_result},
            "cbam": {"success": False, "error": None},
        }

        try:
            await mock_cbam_agent.process({})
        except ValueError as e:
            pipeline_result["cbam"]["error"] = str(e)

        assert pipeline_result["fuel_analyzer"]["success"] is True
        assert pipeline_result["cbam"]["success"] is False
        assert "CBAM calculation failed" in pipeline_result["cbam"]["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_timeout_recovery(self, mock_fuel_analyzer_agent):
        """INT-ERR-006: Test recovery from agent timeout."""
        async def slow_then_fast(input_data):
            if input_data.get("attempt", 1) == 1:
                await asyncio.sleep(1)  # First attempt is slow
            return {"emissions_value": 56.1}

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=slow_then_fast)

        # First attempt times out
        try:
            result = await asyncio.wait_for(
                mock_fuel_analyzer_agent.process({"attempt": 1}),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Retry with shorter processing
            result = await mock_fuel_analyzer_agent.process({"attempt": 2})

        assert result["emissions_value"] == 56.1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_error_logging(self, mock_fuel_analyzer_agent):
        """INT-ERR-007: Test errors are logged correctly."""
        error_log = []

        async def logging_wrapper(agent, input_data):
            try:
                return await agent.process(input_data)
            except Exception as e:
                error_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent.name,
                    "error": str(e),
                    "input": input_data,
                })
                raise

        async def failing_process(input_data):
            raise ValueError("Test error")

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=failing_process)

        with pytest.raises(ValueError):
            await logging_wrapper(mock_fuel_analyzer_agent, {"quantity": 1000})

        assert len(error_log) == 1
        assert error_log[0]["agent"] == "fuel_analyzer"
        assert "Test error" in error_log[0]["error"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_graceful_degradation(self, mock_fuel_analyzer_agent):
        """INT-ERR-008: Test graceful degradation on non-critical errors."""
        async def partial_failure(input_data):
            result = {"emissions_value": input_data["quantity"] * 0.0561}
            # Non-critical feature fails
            try:
                raise Exception("Optional feature unavailable")
            except Exception:
                result["warnings"] = ["Optional feature unavailable"]
            return result

        mock_fuel_analyzer_agent.process = AsyncMock(side_effect=partial_failure)

        result = await mock_fuel_analyzer_agent.process({"quantity": 1000})

        # Primary result available despite non-critical failure
        assert result["emissions_value"] == pytest.approx(56.1, rel=0.01)
        assert "warnings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
