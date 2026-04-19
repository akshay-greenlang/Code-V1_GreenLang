# -*- coding: utf-8 -*-
"""
Phase 5 Critical Path Compliance Test Suite

CRITICAL: These tests validate regulatory compliance requirements for deterministic agents.
All tests in this suite MUST pass 100% for production deployment.

Test Categories:
1. Determinism Tests - Verify identical outputs for identical inputs (byte-for-byte)
2. No LLM Dependency Tests - Verify no ChatSession or API dependencies
3. Performance Benchmarks - Verify <10ms execution time (100x faster than AI)
4. Deprecation Warning Tests - Verify deprecated AI agents show warnings
5. Audit Trail Tests - Verify complete provenance and logging
6. Reproducibility Tests - Verify cross-run consistency

Agents Under Test:
- FuelAgent (fuel_agent.py) - CRITICAL PATH
- GridFactorAgent (grid_factor_agent.py) - CRITICAL PATH
- BoilerAgent (boiler_agent.py) - CRITICAL PATH
- CarbonAgent (carbon_agent.py) - CRITICAL PATH (deterministic version)

Regulatory Compliance:
- ISO 14064-1 (GHG Accounting)
- GHG Protocol Corporate Standard
- SOC 2 Type II (Deterministic Controls)

Author: GreenLang Framework Team
Date: November 2025
Version: 1.0.0
"""

import pytest
import time
import hashlib
import json
import warnings
import sys
import subprocess
from typing import Dict, Any, List
from pathlib import Path

# Import CRITICAL PATH agents
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.grid_factor_agent import GridFactorAgent
from greenlang.agents.boiler_agent import BoilerAgent
from greenlang.agents.carbon_agent import CarbonAgent


# ============================================================================
# MARK: Test Configuration
# ============================================================================

# Critical Path agents that must be 100% deterministic
CRITICAL_PATH_AGENTS = {
    "fuel": FuelAgent,
    "grid_factor": GridFactorAgent,
    "boiler": BoilerAgent,
    "carbon": CarbonAgent,
}

# Deprecated AI agents that should trigger warnings
DEPRECATED_AI_AGENTS = [
    "greenlang.agents.fuel_agent_ai",
    "greenlang.agents.grid_factor_agent_ai",
]

# Performance targets
PERFORMANCE_TARGET_MS = 10.0  # Target: <10ms execution time
DETERMINISM_ITERATIONS = 10  # Run each calculation 10 times


# ============================================================================
# MARK: A. Determinism Tests (CRITICAL)
# ============================================================================

@pytest.mark.critical_path
class TestDeterminism:
    """
    Test that CRITICAL PATH agents produce identical outputs for identical inputs.

    WHY THIS IS CRITICAL:
    - Regulatory audits require reproducible emissions calculations
    - Financial transactions based on carbon credits need exact numbers
    - ISO 14064-1 requires deterministic GHG accounting
    - SOC 2 controls require deterministic processing
    """

    def test_fuel_agent_determinism_natural_gas(
        self,
        sample_fuel_consumption_natural_gas,
        hash_result,
        assert_deterministic_result
    ):
        """Test FuelAgent produces identical results for natural gas consumption."""
        agent = FuelAgent()
        results = []
        hashes = []

        # Run calculation 10 times
        for i in range(DETERMINISM_ITERATIONS):
            result = agent.run(sample_fuel_consumption_natural_gas)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for i, result in enumerate(results):
            assert result["success"], f"Iteration {i} failed: {result.get('error')}"

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results: {unique_hashes}"

        # Compare first and last results byte-for-byte
        assert_deterministic_result(results[0], results[-1])

        # Check exact emission value
        expected_emissions = 5310.0  # 1000 therms * 5.31 kgCO2e/therm
        for result in results:
            actual = result["data"]["co2e_emissions_kg"]
            assert actual == expected_emissions, f"Expected {expected_emissions}, got {actual}"

    def test_fuel_agent_determinism_electricity(
        self,
        sample_fuel_consumption_electricity,
        hash_result,
        assert_deterministic_result
    ):
        """Test FuelAgent produces identical results for electricity consumption."""
        agent = FuelAgent()
        results = []
        hashes = []

        for i in range(DETERMINISM_ITERATIONS):
            result = agent.run(sample_fuel_consumption_electricity)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for i, result in enumerate(results):
            assert result["success"], f"Iteration {i} failed"

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results"

        # Compare first and last results
        assert_deterministic_result(results[0], results[-1])

    def test_fuel_agent_determinism_diesel(
        self,
        sample_fuel_consumption_diesel,
        hash_result
    ):
        """Test FuelAgent produces identical results for diesel consumption."""
        agent = FuelAgent()
        results = []
        hashes = []

        for i in range(DETERMINISM_ITERATIONS):
            result = agent.run(sample_fuel_consumption_diesel)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for result in results:
            assert result["success"]

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results"

    def test_fuel_agent_determinism_multiple_inputs(
        self,
        determinism_test_inputs,
        hash_result
    ):
        """Test FuelAgent determinism across multiple input variations."""
        agent = FuelAgent()

        for input_data in determinism_test_inputs:
            results = []
            hashes = []

            # Run each input 10 times
            for i in range(DETERMINISM_ITERATIONS):
                result = agent.run(input_data)
                results.append(result)
                hashes.append(hash_result(result))

            # Check all are successful
            for result in results:
                assert result["success"], f"Failed for input: {input_data}"

            # Check determinism
            unique_hashes = set(hashes)
            assert len(unique_hashes) == 1, f"Non-deterministic for input {input_data}"

    def test_grid_factor_agent_determinism(
        self,
        sample_grid_factor_request,
        hash_result
    ):
        """Test GridFactorAgent produces identical results."""
        agent = GridFactorAgent()
        results = []
        hashes = []

        for i in range(DETERMINISM_ITERATIONS):
            result = agent.run(sample_grid_factor_request)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for result in results:
            assert result["success"]

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results"

        # Check exact emission factor value
        expected_factor = 0.385  # US grid intensity
        for result in results:
            actual = result["data"]["emission_factor"]
            assert actual == expected_factor, f"Expected {expected_factor}, got {actual}"

    def test_grid_factor_agent_determinism_multiple_countries(
        self,
        cross_country_test_data,
        hash_result
    ):
        """Test GridFactorAgent determinism across multiple countries."""
        agent = GridFactorAgent()

        for country_data in cross_country_test_data:
            results = []
            hashes = []

            for i in range(DETERMINISM_ITERATIONS):
                result = agent.run(country_data)
                results.append(result)
                hashes.append(hash_result(result))

            # Check all successful or all fail consistently
            success_states = [r["success"] for r in results]
            assert len(set(success_states)) == 1, f"Inconsistent success state for {country_data}"

            # If successful, check determinism
            if results[0]["success"]:
                unique_hashes = set(hashes)
                assert len(unique_hashes) == 1, f"Non-deterministic for {country_data}"

    def test_boiler_agent_determinism_thermal_output(
        self,
        sample_boiler_thermal_output,
        hash_result
    ):
        """Test BoilerAgent produces identical results with thermal output input."""
        agent = BoilerAgent()
        results = []
        hashes = []

        for i in range(DETERMINISM_ITERATIONS):
            result = agent.run(sample_boiler_thermal_output)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for result in results:
            assert result["success"]

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results"

    def test_boiler_agent_determinism_fuel_consumption(
        self,
        sample_boiler_fuel_consumption,
        hash_result
    ):
        """Test BoilerAgent produces identical results with fuel consumption input."""
        agent = BoilerAgent()
        results = []
        hashes = []

        for i in range(DETERMINISM_ITERATIONS):
            result = agent.run(sample_boiler_fuel_consumption)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for result in results:
            assert result["success"]

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results"

    def test_carbon_agent_determinism(
        self,
        sample_carbon_aggregation,
        hash_result
    ):
        """Test CarbonAgent produces identical aggregation results."""
        agent = CarbonAgent()
        results = []
        hashes = []

        for i in range(DETERMINISM_ITERATIONS):
            result = agent.execute(sample_carbon_aggregation)
            # Convert AgentResult to dict for consistency
            result_dict = {
                "success": result.success,
                "data": result.data,
                "metadata": result.metadata if hasattr(result, "metadata") else {}
            }
            results.append(result_dict)
            hashes.append(hash_result(result_dict))

        # All results must be successful
        for result in results:
            assert result["success"]

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, f"Non-deterministic! Got {len(unique_hashes)} different results"

        # Check exact total
        expected_total = 9787.5  # Sum of all emissions
        for result in results:
            actual = result["data"]["total_co2e_kg"]
            assert actual == expected_total, f"Expected {expected_total}, got {actual}"


# ============================================================================
# MARK: B. No LLM Dependency Tests
# ============================================================================

@pytest.mark.critical_path
class TestNoLLMDependencies:
    """
    Verify CRITICAL PATH agents don't use LLM dependencies.

    WHY THIS IS CRITICAL:
    - LLM calls are non-deterministic (even with temperature=0, seed)
    - API failures can't affect regulatory calculations
    - Performance requirements (100x faster than AI)
    - Cost control (no API charges for critical path)
    - Data privacy (no emissions data sent to third parties)
    """

    def test_fuel_agent_no_chatsession_import(self):
        """Verify FuelAgent doesn't import ChatSession."""
        import greenlang.agents.fuel_agent as fuel_module
        import inspect

        source = inspect.getsource(fuel_module)

        # Check for banned imports
        assert "ChatSession" not in source, "FuelAgent imports ChatSession (NOT ALLOWED)"
        assert "from greenlang.intelligence.chatsession" not in source, "FuelAgent imports from chatsession module"
        assert "from greenlang.intelligence.rag" not in source, "FuelAgent imports RAG engine"

    def test_fuel_agent_no_temperature_parameter(self):
        """Verify FuelAgent has no temperature parameter."""
        import greenlang.agents.fuel_agent as fuel_module
        import inspect

        source = inspect.getsource(fuel_module)

        # Check for LLM parameters
        assert "temperature=" not in source, "FuelAgent has temperature parameter (LLM indicator)"
        assert "seed=" not in source or "random.seed" in source, "FuelAgent has LLM seed parameter"

    def test_fuel_agent_no_api_keys(self):
        """Verify FuelAgent doesn't use API keys."""
        import greenlang.agents.fuel_agent as fuel_module
        import inspect

        source = inspect.getsource(fuel_module)

        # Check for API key usage
        assert "ANTHROPIC_API_KEY" not in source, "FuelAgent uses Anthropic API key"
        assert "OPENAI_API_KEY" not in source, "FuelAgent uses OpenAI API key"
        assert "anthropic" not in source.lower(), "FuelAgent imports anthropic"
        assert "openai" not in source.lower() or "# openai" in source.lower(), "FuelAgent imports openai"

    def test_grid_factor_agent_no_llm_dependencies(self):
        """Verify GridFactorAgent has no LLM dependencies."""
        import greenlang.agents.grid_factor_agent as grid_module
        import inspect

        source = inspect.getsource(grid_module)

        # Check for banned patterns
        banned = ["ChatSession", "temperature=", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        for pattern in banned:
            assert pattern not in source, f"GridFactorAgent contains banned pattern: {pattern}"

    def test_boiler_agent_no_llm_dependencies(self):
        """Verify BoilerAgent has no LLM dependencies."""
        import greenlang.agents.boiler_agent as boiler_module
        import inspect

        source = inspect.getsource(boiler_module)

        # Check for banned patterns
        banned = ["ChatSession", "temperature=", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        for pattern in banned:
            assert pattern not in source, f"BoilerAgent contains banned pattern: {pattern}"

    def test_carbon_agent_no_llm_dependencies(self):
        """Verify CarbonAgent has no LLM dependencies."""
        import greenlang.agents.carbon_agent as carbon_module
        import inspect

        source = inspect.getsource(carbon_module)

        # Check for banned patterns
        banned = ["ChatSession", "temperature=", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        for pattern in banned:
            assert pattern not in source, f"CarbonAgent contains banned pattern: {pattern}"

    def test_all_critical_path_agents_no_rag_engine(self):
        """Verify no CRITICAL PATH agent uses RAG engine."""
        for agent_name, agent_class in CRITICAL_PATH_AGENTS.items():
            import inspect
            source = inspect.getsource(inspect.getmodule(agent_class))

            assert "from greenlang.intelligence.rag" not in source, f"{agent_name} imports RAG engine"
            assert "RAGEngine" not in source, f"{agent_name} uses RAGEngine"


# ============================================================================
# MARK: C. Performance Benchmarks
# ============================================================================

@pytest.mark.critical_path
class TestPerformanceBenchmarks:
    """
    Test that deterministic agents meet performance requirements.

    REQUIREMENTS:
    - Target: <10ms execution time
    - 100x faster than AI versions (AI: ~1000ms, Deterministic: <10ms)
    - No network calls
    - Minimal memory allocation
    """

    def test_fuel_agent_performance_target(
        self,
        sample_fuel_consumption_natural_gas,
        performance_benchmark
    ):
        """Test FuelAgent meets <10ms performance target."""
        agent = FuelAgent()

        # Warm up (first call may be slower due to cache)
        agent.run(sample_fuel_consumption_natural_gas)

        # Benchmark
        result, execution_time_ms = performance_benchmark(
            agent.run,
            sample_fuel_consumption_natural_gas,
            PERFORMANCE_TARGET_MS
        )

        assert result["success"], "Calculation failed"
        assert execution_time_ms < PERFORMANCE_TARGET_MS, \
            f"FuelAgent too slow: {execution_time_ms:.2f}ms (target: <{PERFORMANCE_TARGET_MS}ms)"

    def test_fuel_agent_average_performance(
        self,
        sample_fuel_consumption_natural_gas
    ):
        """Test FuelAgent average performance over 100 runs."""
        agent = FuelAgent()
        times = []

        # Warm up
        for _ in range(10):
            agent.run(sample_fuel_consumption_natural_gas)

        # Benchmark 100 runs
        for _ in range(100):
            start = time.perf_counter()
            result = agent.run(sample_fuel_consumption_natural_gas)
            end = time.perf_counter()

            assert result["success"]
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        print(f"\nFuelAgent Performance:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  Min: {min_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")

        assert avg_time < PERFORMANCE_TARGET_MS, f"Average time {avg_time:.2f}ms exceeds target"

    def test_grid_factor_agent_performance(
        self,
        sample_grid_factor_request,
        performance_benchmark
    ):
        """Test GridFactorAgent meets performance target."""
        agent = GridFactorAgent()

        # Warm up
        agent.run(sample_grid_factor_request)

        # Benchmark
        result, execution_time_ms = performance_benchmark(
            agent.run,
            sample_grid_factor_request,
            PERFORMANCE_TARGET_MS
        )

        assert result["success"]
        assert execution_time_ms < PERFORMANCE_TARGET_MS, \
            f"GridFactorAgent too slow: {execution_time_ms:.2f}ms"

    def test_boiler_agent_performance(
        self,
        sample_boiler_fuel_consumption,
        performance_benchmark
    ):
        """Test BoilerAgent meets performance target."""
        agent = BoilerAgent()

        # Warm up
        agent.run(sample_boiler_fuel_consumption)

        # Benchmark
        result, execution_time_ms = performance_benchmark(
            agent.run,
            sample_boiler_fuel_consumption,
            PERFORMANCE_TARGET_MS
        )

        assert result["success"]
        assert execution_time_ms < PERFORMANCE_TARGET_MS, \
            f"BoilerAgent too slow: {execution_time_ms:.2f}ms"

    def test_carbon_agent_performance(
        self,
        sample_carbon_aggregation
    ):
        """Test CarbonAgent meets performance target."""
        agent = CarbonAgent()

        # Warm up
        agent.execute(sample_carbon_aggregation)

        # Benchmark
        start = time.perf_counter()
        result = agent.execute(sample_carbon_aggregation)
        end = time.perf_counter()

        execution_time_ms = (end - start) * 1000

        assert result.success
        assert execution_time_ms < PERFORMANCE_TARGET_MS, \
            f"CarbonAgent too slow: {execution_time_ms:.2f}ms"

    def test_performance_comparison_100x_improvement(
        self,
        sample_fuel_consumption_natural_gas
    ):
        """Verify deterministic agent is 100x faster than AI version."""
        agent = FuelAgent()

        # Warm up
        agent.run(sample_fuel_consumption_natural_gas)

        # Benchmark deterministic version
        times = []
        for _ in range(50):
            start = time.perf_counter()
            agent.run(sample_fuel_consumption_natural_gas)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_deterministic_ms = sum(times) / len(times)

        # AI version typically takes ~1000ms (with API calls)
        # We expect deterministic to be <10ms
        # That's >100x improvement
        expected_ai_time_ms = 1000.0
        speedup = expected_ai_time_ms / avg_deterministic_ms

        print(f"\nPerformance Improvement:")
        print(f"  Deterministic: {avg_deterministic_ms:.3f}ms")
        print(f"  AI (expected): {expected_ai_time_ms:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x")

        assert speedup > 100, f"Expected >100x speedup, got {speedup:.1f}x"


# ============================================================================
# MARK: D. Deprecation Warning Tests
# ============================================================================

@pytest.mark.critical_path
class TestDeprecationWarnings:
    """
    Verify deprecated AI agents show clear warnings.

    WHY THIS IS CRITICAL:
    - Prevent accidental use of AI agents for regulatory calculations
    - Guide developers to correct deterministic versions
    - Maintain backward compatibility during migration
    """

    def test_fuel_agent_ai_deprecation_warning(self):
        """Test FuelAgentAI triggers DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import should trigger warning
            from greenlang.agents.fuel_agent_ai import FuelAgentAI

            # Check warning was raised
            assert len(w) >= 1, "No deprecation warning raised"

            # Check it's a DeprecationWarning
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w), \
                "Warning is not DeprecationWarning"

            # Check message mentions FuelAgent
            warning_messages = [str(warning.message) for warning in w]
            assert any("FuelAgent" in msg for msg in warning_messages), \
                "Warning doesn't mention FuelAgent"

            # Check message mentions CRITICAL PATH
            assert any("CRITICAL PATH" in msg for msg in warning_messages), \
                "Warning doesn't mention CRITICAL PATH"

    def test_grid_factor_agent_ai_deprecation_warning(self):
        """Test GridFactorAgentAI triggers DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from greenlang.agents.grid_factor_agent_ai import GridFactorAgentAI

            assert len(w) >= 1, "No deprecation warning raised"
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

            warning_messages = [str(warning.message) for warning in w]
            assert any("GridFactorAgent" in msg for msg in warning_messages)

    def test_deprecation_warning_messages_are_clear(self):
        """Test deprecation warnings provide clear guidance."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from greenlang.agents.fuel_agent_ai import FuelAgentAI

            # Get deprecation warnings
            dep_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]

            assert len(dep_warnings) > 0, "No deprecation warnings found"

            # Check message quality
            for warning in dep_warnings:
                msg = str(warning.message)

                # Should mention what's deprecated
                assert "deprecated" in msg.lower(), "Message doesn't say 'deprecated'"

                # Should tell user what to use instead
                assert "instead" in msg.lower() or "use" in msg.lower(), \
                    "Message doesn't provide alternative"


# ============================================================================
# MARK: E. Audit Trail Tests
# ============================================================================

@pytest.mark.critical_path
class TestAuditTrails:
    """
    Verify all CRITICAL PATH agents have complete audit trails.

    REQUIREMENTS (SOC 2, ISO 14064-1):
    - Input logging
    - Calculation steps logged
    - Output logging
    - Timestamps
    - Version tracking
    - Data provenance
    """

    def test_fuel_agent_audit_trail_completeness(
        self,
        sample_fuel_consumption_natural_gas,
        assert_complete_audit_trail
    ):
        """Test FuelAgent provides complete audit trail."""
        agent = FuelAgent()
        result = agent.run(sample_fuel_consumption_natural_gas)

        assert_complete_audit_trail(result)

        # Check specific fields
        assert "metadata" in result
        metadata = result["metadata"]

        assert "agent_id" in metadata
        assert metadata["agent_id"] == "fuel"

        assert "calculation" in metadata
        assert "×" in metadata["calculation"], "Calculation doesn't show formula"

    def test_fuel_agent_audit_trail_input_tracking(
        self,
        sample_fuel_consumption_natural_gas
    ):
        """Test FuelAgent tracks input parameters in audit trail."""
        agent = FuelAgent()
        result = agent.run(sample_fuel_consumption_natural_gas)

        assert result["success"]
        data = result["data"]

        # Input values should be in output
        assert data["fuel_type"] == sample_fuel_consumption_natural_gas["fuel_type"]
        assert data["consumption_amount"] == sample_fuel_consumption_natural_gas["amount"]
        assert data["consumption_unit"] == sample_fuel_consumption_natural_gas["unit"]
        assert data["country"] == sample_fuel_consumption_natural_gas["country"]

    def test_fuel_agent_audit_trail_calculation_details(
        self,
        sample_fuel_consumption_natural_gas
    ):
        """Test FuelAgent provides calculation details in audit trail."""
        agent = FuelAgent()
        result = agent.run(sample_fuel_consumption_natural_gas)

        assert result["success"]
        data = result["data"]
        metadata = result["metadata"]

        # Must include emission factor
        assert "emission_factor" in data
        assert data["emission_factor"] > 0

        # Must include calculation formula
        assert "calculation" in metadata
        calc = metadata["calculation"]

        # Formula should show: amount × factor
        assert str(sample_fuel_consumption_natural_gas["amount"]) in calc
        assert "kgCO2e" in calc

    def test_grid_factor_agent_audit_trail(
        self,
        sample_grid_factor_request
    ):
        """Test GridFactorAgent provides complete audit trail."""
        agent = GridFactorAgent()
        result = agent.run(sample_grid_factor_request)

        assert result["success"]

        # Check metadata
        assert "metadata" in result
        metadata = result["metadata"]

        assert "agent_id" in metadata
        assert metadata["agent_id"] == "grid_factor"

        # Check data provenance
        data = result["data"]
        assert "source" in data, "Missing data source"
        assert "version" in data, "Missing data version"
        assert "last_updated" in data, "Missing last_updated timestamp"

    def test_boiler_agent_audit_trail(
        self,
        sample_boiler_fuel_consumption
    ):
        """Test BoilerAgent provides complete audit trail."""
        agent = BoilerAgent()
        result = agent.run(sample_boiler_fuel_consumption)

        assert result["success"]

        # Check metadata
        assert "metadata" in result
        metadata = result["metadata"]

        assert "agent_id" in metadata
        assert "calculation" in metadata
        assert "efficiency_used" in metadata

        # Check efficiency tracking
        assert "efficiency" in result["data"]
        assert "thermal_efficiency_percent" in result["data"]

    def test_carbon_agent_audit_trail(
        self,
        sample_carbon_aggregation
    ):
        """Test CarbonAgent provides complete audit trail."""
        agent = CarbonAgent()
        result = agent.execute(sample_carbon_aggregation)

        assert result.success

        # Check breakdown tracking
        assert "emissions_breakdown" in result.data
        breakdown = result.data["emissions_breakdown"]

        assert len(breakdown) > 0, "No emissions breakdown"

        # Each item should have source tracking
        for item in breakdown:
            assert "source" in item, "Missing source in breakdown"
            assert "co2e_kg" in item, "Missing emissions in breakdown"
            assert "percentage" in item, "Missing percentage in breakdown"

    def test_audit_trail_version_tracking(self):
        """Test all agents include version information."""
        agents_to_test = [
            (FuelAgent(), "fuel"),
            (GridFactorAgent(), "grid_factor"),
            (BoilerAgent(), "boiler"),
        ]

        for agent, agent_id in agents_to_test:
            assert hasattr(agent, "version"), f"{agent_id} missing version attribute"
            assert agent.version is not None, f"{agent_id} version is None"
            assert len(agent.version) > 0, f"{agent_id} version is empty"


# ============================================================================
# MARK: F. Reproducibility Tests
# ============================================================================

@pytest.mark.critical_path
class TestReproducibility:
    """
    Test cross-run reproducibility (Python interpreter restart simulation).

    REQUIREMENTS:
    - Results must be identical across different Python sessions
    - No dependency on execution order
    - No dependency on cache state
    - No dependency on random seeds (unless explicitly set)
    """

    def test_fuel_agent_cross_run_reproducibility(
        self,
        sample_fuel_consumption_natural_gas,
        hash_result
    ):
        """Test FuelAgent produces identical results across 'sessions'."""
        # Simulate multiple sessions by creating new agent instances
        results = []
        hashes = []

        for session_num in range(5):
            # Create fresh agent (simulates new Python session)
            agent = FuelAgent()

            # Clear any caches
            if hasattr(agent, "clear_cache"):
                agent.clear_cache()

            # Run calculation
            result = agent.run(sample_fuel_consumption_natural_gas)
            results.append(result)
            hashes.append(hash_result(result))

        # All results must be successful
        for i, result in enumerate(results):
            assert result["success"], f"Session {i} failed"

        # All hashes must be identical
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 1, \
            f"Non-reproducible across sessions! Got {len(unique_hashes)} different results"

    def test_grid_factor_agent_cache_independence(
        self,
        sample_grid_factor_request
    ):
        """Test GridFactorAgent results don't depend on cache state."""
        # Test 1: Fresh agent, no cache
        agent1 = GridFactorAgent()
        result1 = agent1.run(sample_grid_factor_request)

        # Test 2: Warm cache
        agent2 = GridFactorAgent()
        agent2.run(sample_grid_factor_request)  # Warm up
        result2 = agent2.run(sample_grid_factor_request)  # Use cache

        # Test 3: Cleared cache
        agent3 = GridFactorAgent()
        agent3.run(sample_grid_factor_request)  # Warm up
        # Grid factor agent doesn't have clear_cache, but that's OK
        result3 = agent3.run(sample_grid_factor_request)

        # All results must be identical
        assert result1["success"] and result2["success"] and result3["success"]

        assert result1["data"]["emission_factor"] == result2["data"]["emission_factor"]
        assert result2["data"]["emission_factor"] == result3["data"]["emission_factor"]

    def test_execution_order_independence(
        self,
        determinism_test_inputs
    ):
        """Test that execution order doesn't affect results."""
        agent = FuelAgent()

        # Run in original order
        results_forward = []
        for input_data in determinism_test_inputs:
            result = agent.run(input_data)
            results_forward.append(result)

        # Run in reverse order
        results_reverse = []
        for input_data in reversed(determinism_test_inputs):
            result = agent.run(input_data)
            results_reverse.insert(0, result)

        # Results should match regardless of order
        for i, (forward, reverse) in enumerate(zip(results_forward, results_reverse)):
            assert forward["success"] == reverse["success"], f"Different success state at index {i}"

            if forward["success"]:
                assert forward["data"]["co2e_emissions_kg"] == reverse["data"]["co2e_emissions_kg"], \
                    f"Different emissions at index {i}"

    def test_parallel_execution_consistency(
        self,
        sample_fuel_consumption_natural_gas
    ):
        """Test that parallel executions produce consistent results."""
        from concurrent.futures import ThreadPoolExecutor

        agent = FuelAgent()

        def run_calculation():
            return agent.run(sample_fuel_consumption_natural_gas)

        # Run 20 calculations in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_calculation) for _ in range(20)]
            results = [future.result() for future in futures]

        # All results must be successful
        for result in results:
            assert result["success"]

        # All emissions values must be identical
        emissions_values = [r["data"]["co2e_emissions_kg"] for r in results]
        unique_emissions = set(emissions_values)

        assert len(unique_emissions) == 1, \
            f"Inconsistent results in parallel execution: {unique_emissions}"


# ============================================================================
# MARK: G. Integration Tests
# ============================================================================

@pytest.mark.critical_path
class TestCriticalPathIntegration:
    """
    Test integration of multiple CRITICAL PATH agents.

    SCENARIO: Complete facility emissions calculation
    - Use GridFactorAgent to get emission factors
    - Use FuelAgent to calculate fuel emissions
    - Use BoilerAgent to calculate boiler emissions
    - Use CarbonAgent to aggregate total emissions
    """

    def test_end_to_end_facility_emissions_determinism(self):
        """Test end-to-end facility calculation is deterministic."""
        # Run complete calculation 5 times
        results = []

        for run_num in range(5):
            # Step 1: Get grid emission factor
            grid_agent = GridFactorAgent()
            grid_result = grid_agent.run({
                "country": "US",
                "fuel_type": "electricity",
                "unit": "kWh",
                "year": 2025
            })
            assert grid_result["success"]

            # Step 2: Calculate electricity emissions
            fuel_agent = FuelAgent()
            elec_result = fuel_agent.run({
                "fuel_type": "electricity",
                "amount": 10000.0,
                "unit": "kWh",
                "country": "US",
                "year": 2025
            })
            assert elec_result["success"]

            # Step 3: Calculate natural gas emissions
            gas_result = fuel_agent.run({
                "fuel_type": "natural_gas",
                "amount": 500.0,
                "unit": "therms",
                "country": "US",
                "year": 2025
            })
            assert gas_result["success"]

            # Step 4: Calculate boiler emissions
            boiler_agent = BoilerAgent()
            boiler_result = boiler_agent.run({
                "boiler_type": "condensing",
                "fuel_type": "natural_gas",
                "fuel_consumption": {
                    "value": 200.0,
                    "unit": "therms"
                },
                "efficiency": 0.92,
                "country": "US",
                "year": 2025
            })
            assert boiler_result["success"]

            # Step 5: Aggregate with CarbonAgent
            carbon_agent = CarbonAgent()
            total_result = carbon_agent.execute({
                "emissions": [
                    {
                        "fuel_type": "electricity",
                        "co2e_emissions_kg": elec_result["data"]["co2e_emissions_kg"],
                        "source": "facility"
                    },
                    {
                        "fuel_type": "natural_gas",
                        "co2e_emissions_kg": gas_result["data"]["co2e_emissions_kg"],
                        "source": "heaters"
                    },
                    {
                        "fuel_type": "natural_gas",
                        "co2e_emissions_kg": boiler_result["data"]["co2e_emissions_kg"],
                        "source": "boiler"
                    }
                ],
                "building_area": 50000,
                "occupancy": 250
            })
            assert total_result.success

            # Store total emissions
            results.append(total_result.data["total_co2e_kg"])

        # All runs must produce identical results
        unique_results = set(results)
        assert len(unique_results) == 1, \
            f"Non-deterministic end-to-end! Got {len(unique_results)} different totals: {unique_results}"

        print(f"\nEnd-to-End Determinism: ✓ All 5 runs produced {results[0]:.2f} kg CO2e")

    def test_integration_performance(self):
        """Test integrated calculation meets performance requirements."""
        start = time.perf_counter()

        # Complete facility calculation
        grid_agent = GridFactorAgent()
        fuel_agent = FuelAgent()
        boiler_agent = BoilerAgent()
        carbon_agent = CarbonAgent()

        # Execute all calculations
        grid_result = grid_agent.run({
            "country": "US",
            "fuel_type": "electricity",
            "unit": "kWh",
            "year": 2025
        })

        elec_result = fuel_agent.run({
            "fuel_type": "electricity",
            "amount": 10000.0,
            "unit": "kWh",
            "country": "US"
        })

        gas_result = fuel_agent.run({
            "fuel_type": "natural_gas",
            "amount": 500.0,
            "unit": "therms",
            "country": "US"
        })

        boiler_result = boiler_agent.run({
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "fuel_consumption": {"value": 200.0, "unit": "therms"},
            "efficiency": 0.92,
            "country": "US"
        })

        total_result = carbon_agent.execute({
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": elec_result["data"]["co2e_emissions_kg"]},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": gas_result["data"]["co2e_emissions_kg"]},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": boiler_result["data"]["co2e_emissions_kg"]}
            ]
        })

        end = time.perf_counter()
        total_time_ms = (end - start) * 1000

        # All steps must succeed
        assert grid_result["success"]
        assert elec_result["success"]
        assert gas_result["success"]
        assert boiler_result["success"]
        assert total_result.success

        # Total time should be reasonable (<100ms for full pipeline)
        assert total_time_ms < 100.0, \
            f"Integrated calculation too slow: {total_time_ms:.2f}ms (target: <100ms)"

        print(f"\nIntegrated Performance: {total_time_ms:.2f}ms for complete facility calculation")


# ============================================================================
# MARK: H. Compliance Summary
# ============================================================================

@pytest.mark.critical_path
def test_compliance_summary(capsys):
    """
    Generate compliance summary report.

    This test always passes but prints a summary of compliance status.
    """
    print("\n" + "="*80)
    print("PHASE 5 CRITICAL PATH COMPLIANCE SUMMARY")
    print("="*80)

    print("\n✓ CRITICAL PATH AGENTS:")
    for agent_name, agent_class in CRITICAL_PATH_AGENTS.items():
        print(f"  - {agent_name}: {agent_class.__name__}")

    print("\n✓ COMPLIANCE REQUIREMENTS:")
    print("  - Complete Determinism (byte-for-byte identical outputs)")
    print("  - Zero LLM Dependencies (no ChatSession, no API calls)")
    print("  - Performance Target (<10ms execution time)")
    print("  - Complete Audit Trails (full provenance tracking)")
    print("  - Cross-Run Reproducibility (session-independent)")

    print("\n✓ REGULATORY STANDARDS:")
    print("  - ISO 14064-1 (GHG Accounting)")
    print("  - GHG Protocol Corporate Standard")
    print("  - SOC 2 Type II (Deterministic Controls)")

    print("\n✓ DEPRECATION WARNINGS:")
    print("  - fuel_agent_ai.py → Use fuel_agent.py")
    print("  - grid_factor_agent_ai.py → Use grid_factor_agent.py")

    print("\n" + "="*80)
    print("Run pytest with -v flag for detailed test results")
    print("="*80 + "\n")

    assert True, "Compliance summary generated"
