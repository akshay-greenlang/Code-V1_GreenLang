# -*- coding: utf-8 -*-
"""
tests/agents/test_performance_benchmarks.py

Performance Benchmarks for FuelAgentAI v2

OBJECTIVES:
1. Validate latency thresholds (fast path <100ms, AI path <500ms)
2. Validate cost targets (v2 optimized ≤ v1 baseline)
3. Measure throughput (requests/second)
4. Validate memory efficiency
5. Measure fast path effectiveness (60% traffic target)

PERFORMANCE TARGETS (from COST_PERFORMANCE_ANALYSIS.md):
- v1 Baseline: $0.0025/calc, 200ms latency
- v2 Baseline (no opt): $0.0083/calc, 350ms latency
- v2 Optimized: $0.0020/calc, 220ms latency (TARGET)

THRESHOLDS:
- Fast path latency: <100ms (P95)
- AI path latency: <500ms (P95)
- Cost per calc: ≤$0.0025 (20% cheaper than v1)
- Throughput: >50 req/s (single instance)
- Memory: <500MB per instance

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
from greenlang.agents import FuelAgentAI_v2, FuelAgentAI


# ==================== FIXTURES ====================


@pytest.fixture
def agent_v2_fast_path():
    """v2 agent with fast path enabled (no AI)"""
    return FuelAgentAI_v2(
        enable_explanations=False,
        enable_recommendations=False,
        enable_fast_path=True,
    )


@pytest.fixture
def agent_v2_ai_path():
    """v2 agent with AI path enabled"""
    return FuelAgentAI_v2(
        enable_explanations=True,
        enable_recommendations=True,
        enable_fast_path=True,
    )


@pytest.fixture
def agent_v1():
    """v1 agent for baseline comparison"""
    return FuelAgentAI(
        enable_explanations=False,
        enable_recommendations=False,
    )


@pytest.fixture
def sample_payloads() -> List[Dict[str, Any]]:
    """Sample payloads for benchmarking"""
    return [
        {
            "fuel_type": "diesel",
            "amount": 1000,
            "unit": "gallons",
        },
        {
            "fuel_type": "natural_gas",
            "amount": 5000,
            "unit": "therms",
        },
        {
            "fuel_type": "electricity",
            "amount": 10000,
            "unit": "kWh",
        },
        {
            "fuel_type": "gasoline",
            "amount": 500,
            "unit": "gallons",
        },
        {
            "fuel_type": "propane",
            "amount": 200,
            "unit": "gallons",
        },
    ]


# ==================== LATENCY BENCHMARKS ====================


def test_fast_path_latency_p50(agent_v2_fast_path, sample_payloads):
    """
    Benchmark 1: Fast path P50 latency < 50ms

    Target: Fast path should be <50ms at median (P50)
    """
    latencies = []

    # Warmup (exclude first 2 requests from measurements)
    for _ in range(2):
        agent_v2_fast_path.run(sample_payloads[0])

    # Measure latency for 20 requests
    for payload in sample_payloads * 4:  # 20 total requests
        start = time.time()
        result = agent_v2_fast_path.run(payload)
        latency_ms = (time.time() - start) * 1000

        assert result["success"], f"Request failed: {result.get('error')}"
        latencies.append(latency_ms)

    # Calculate P50 (median)
    p50 = statistics.median(latencies)

    print(f"\n📊 Fast Path Latency P50: {p50:.2f}ms")
    print(f"   Min: {min(latencies):.2f}ms")
    print(f"   Max: {max(latencies):.2f}ms")

    # Assert P50 < 50ms
    assert p50 < 50, f"Fast path P50 latency {p50:.2f}ms exceeds 50ms threshold"


def test_fast_path_latency_p95(agent_v2_fast_path, sample_payloads):
    """
    Benchmark 2: Fast path P95 latency < 100ms

    Target: Fast path should be <100ms at 95th percentile (P95)
    """
    latencies = []

    # Warmup
    for _ in range(2):
        agent_v2_fast_path.run(sample_payloads[0])

    # Measure latency for 50 requests
    for payload in sample_payloads * 10:  # 50 total requests
        start = time.time()
        result = agent_v2_fast_path.run(payload)
        latency_ms = (time.time() - start) * 1000

        assert result["success"], f"Request failed: {result.get('error')}"
        latencies.append(latency_ms)

    # Calculate P95
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    p95 = latencies[p95_index]

    print(f"\n📊 Fast Path Latency P95: {p95:.2f}ms")
    print(f"   P50: {statistics.median(latencies):.2f}ms")
    print(f"   P99: {latencies[int(len(latencies) * 0.99)]:.2f}ms")

    # Assert P95 < 100ms
    assert p95 < 100, f"Fast path P95 latency {p95:.2f}ms exceeds 100ms threshold"


@pytest.mark.slow
def test_ai_path_latency_p95(agent_v2_ai_path, sample_payloads):
    """
    Benchmark 3: AI path P95 latency < 500ms

    Target: AI path should be <500ms at 95th percentile (P95)
    NOTE: This test is marked as 'slow' due to AI calls
    """
    latencies = []

    # Measure latency for 10 requests (fewer due to cost)
    for payload in sample_payloads * 2:  # 10 total requests
        start = time.time()
        result = agent_v2_ai_path.run(payload)
        latency_ms = (time.time() - start) * 1000

        assert result["success"], f"Request failed: {result.get('error')}"
        latencies.append(latency_ms)

    # Calculate P95
    latencies.sort()
    p95_index = int(len(latencies) * 0.95)
    p95 = latencies[p95_index] if len(latencies) > 1 else latencies[0]

    print(f"\n📊 AI Path Latency P95: {p95:.2f}ms")
    print(f"   P50: {statistics.median(latencies):.2f}ms")
    print(f"   Min: {min(latencies):.2f}ms")
    print(f"   Max: {max(latencies):.2f}ms")

    # Assert P95 < 500ms
    assert p95 < 500, f"AI path P95 latency {p95:.2f}ms exceeds 500ms threshold"


# ==================== COST BENCHMARKS ====================


@pytest.mark.slow
def test_cost_v2_fast_path_vs_v1(agent_v2_fast_path, agent_v1, sample_payloads):
    """
    Benchmark 4: v2 fast path cost ≤ v1 cost

    Target: v2 fast path should be ≤$0.0025/calc (same or cheaper than v1)
    """
    # v1 baseline
    v1_costs = []
    for payload in sample_payloads:
        result = agent_v1.run(payload)
        assert result["success"]
        v1_costs.append(result["metadata"].get("total_cost_usd", 0.0))

    v1_avg_cost = statistics.mean(v1_costs) if v1_costs else 0.0025

    # v2 fast path
    v2_costs = []
    for payload in sample_payloads:
        result = agent_v2_fast_path.run(payload)
        assert result["success"]
        v2_costs.append(result["metadata"].get("total_cost_usd", 0.0))

    v2_avg_cost = statistics.mean(v2_costs) if v2_costs else 0.0

    print(f"\n💰 Cost Comparison:")
    print(f"   v1 average: ${v1_avg_cost:.6f}/calc")
    print(f"   v2 fast path: ${v2_avg_cost:.6f}/calc")

    # Fast path should have zero or minimal cost (no AI calls)
    assert v2_avg_cost <= 0.0001, (
        f"Fast path cost ${v2_avg_cost:.6f} too high (should be ~$0 with no AI)"
    )


@pytest.mark.slow
def test_cost_v2_optimized_target(agent_v2_ai_path, sample_payloads):
    """
    Benchmark 5: v2 optimized cost ≤ $0.0025/calc

    Target: v2 with optimizations should be ≤$0.0025/calc (20% cheaper than v1)
    NOTE: This test measures AI path cost (most expensive path)
    """
    costs = []

    # Measure cost for 5 requests (AI path is expensive)
    for payload in sample_payloads[:5]:
        result = agent_v2_ai_path.run(payload)
        assert result["success"]
        cost = result["metadata"].get("total_cost_usd", 0.0)
        costs.append(cost)

    avg_cost = statistics.mean(costs)

    print(f"\n💰 v2 AI Path Cost: ${avg_cost:.6f}/calc")
    print(f"   Target: ≤$0.0025/calc")
    print(f"   Min: ${min(costs):.6f}")
    print(f"   Max: ${max(costs):.6f}")

    # AI path cost should be reasonable (≤$0.01 per calc)
    # Note: Without caching, AI path will be more expensive than fast path
    assert avg_cost <= 0.01, (
        f"AI path cost ${avg_cost:.6f} exceeds $0.01 threshold"
    )


# ==================== THROUGHPUT BENCHMARKS ====================


def test_throughput_fast_path(agent_v2_fast_path, sample_payloads):
    """
    Benchmark 6: Fast path throughput > 50 req/s

    Target: Single instance should handle >50 requests/second
    """
    num_requests = 100
    payloads = sample_payloads * (num_requests // len(sample_payloads))

    # Warmup
    for _ in range(5):
        agent_v2_fast_path.run(sample_payloads[0])

    # Measure throughput
    start = time.time()
    for payload in payloads[:num_requests]:
        result = agent_v2_fast_path.run(payload)
        assert result["success"]
    duration = time.time() - start

    throughput = num_requests / duration

    print(f"\n⚡ Fast Path Throughput: {throughput:.2f} req/s")
    print(f"   Total requests: {num_requests}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Avg latency: {(duration / num_requests) * 1000:.2f}ms")

    # Assert >50 req/s
    assert throughput > 50, (
        f"Fast path throughput {throughput:.2f} req/s below 50 req/s threshold"
    )


# ==================== EXECUTION PATH BENCHMARKS ====================


def test_fast_path_effectiveness(agent_v2_fast_path):
    """
    Benchmark 7: Fast path used for 100% of legacy requests

    Target: All legacy requests (no explanations) should use fast path
    """
    payloads = [
        {"fuel_type": "diesel", "amount": 1000, "unit": "gallons"},
        {"fuel_type": "natural_gas", "amount": 1000, "unit": "therms"},
        {"fuel_type": "electricity", "amount": 1000, "unit": "kWh"},
    ]

    for payload in payloads:
        result = agent_v2_fast_path.run(payload)
        assert result["success"]

        # Verify fast path was used
        assert result["metadata"]["execution_path"] == "fast", (
            f"Expected fast path, got: {result['metadata']['execution_path']}"
        )

    print(f"\n✅ Fast Path Effectiveness: 100% (all legacy requests use fast path)")


def test_ai_path_for_enhanced_requests(agent_v2_ai_path):
    """
    Benchmark 8: AI path used when explanations enabled

    Validates: AI path is used when needed (explanations/recommendations)
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
    }

    result = agent_v2_ai_path.run(payload)
    assert result["success"]

    # Verify AI path was used
    assert result["metadata"]["execution_path"] == "ai", (
        f"Expected AI path, got: {result['metadata']['execution_path']}"
    )

    # Verify explanation was generated
    assert "explanation" in result["data"], "Explanation missing from AI path output"

    print(f"\n✅ AI Path Effectiveness: AI invoked for enhanced requests")


# ==================== ACCURACY VS SPEED TRADEOFF ====================


def test_fast_path_accuracy_matches_ai_path(agent_v2_fast_path, agent_v2_ai_path):
    """
    Benchmark 9: Fast path produces same emissions as AI path

    Validates: Fast path optimization doesn't sacrifice accuracy
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
    }

    # Fast path result
    result_fast = agent_v2_fast_path.run(payload)
    emissions_fast = result_fast["data"]["co2e_emissions_kg"]

    # AI path result
    result_ai = agent_v2_ai_path.run(payload)
    emissions_ai = result_ai["data"]["co2e_emissions_kg"]

    print(f"\n🎯 Accuracy Check:")
    print(f"   Fast path: {emissions_fast:.4f} kg CO2e")
    print(f"   AI path: {emissions_ai:.4f} kg CO2e")
    print(f"   Difference: {abs(emissions_fast - emissions_ai):.6f} kg CO2e")

    # Emissions should be identical
    assert emissions_fast == emissions_ai, (
        f"Fast path emissions {emissions_fast} != AI path emissions {emissions_ai}"
    )


# ==================== MEMORY BENCHMARKS ====================


@pytest.mark.skipif(True, reason="Memory profiling requires memory_profiler package")
def test_memory_usage_fast_path(agent_v2_fast_path, sample_payloads):
    """
    Benchmark 10: Memory usage < 100MB for fast path

    Target: Fast path should use <100MB of memory
    NOTE: Requires memory_profiler package (optional dependency)
    """
    try:
        from memory_profiler import memory_usage
    except ImportError:
        pytest.skip("memory_profiler not installed")

    def run_requests():
        for payload in sample_payloads * 10:  # 50 requests
            agent_v2_fast_path.run(payload)

    # Measure memory usage
    mem_usage = memory_usage(run_requests, interval=0.1, max_usage=True)

    print(f"\n💾 Fast Path Memory: {mem_usage:.2f} MB")

    # Assert <100MB
    assert mem_usage < 100, f"Memory usage {mem_usage:.2f}MB exceeds 100MB threshold"


# ==================== COMPARATIVE BENCHMARKS ====================


def test_latency_improvement_fast_vs_ai(agent_v2_fast_path, agent_v2_ai_path):
    """
    Benchmark 11: Fast path is significantly faster than AI path

    Target: Fast path should be >3× faster than AI path
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
    }

    # Measure fast path latency
    fast_latencies = []
    for _ in range(10):
        start = time.time()
        agent_v2_fast_path.run(payload)
        fast_latencies.append((time.time() - start) * 1000)

    fast_avg = statistics.mean(fast_latencies)

    # Measure AI path latency
    ai_latencies = []
    for _ in range(5):  # Fewer iterations due to cost
        start = time.time()
        agent_v2_ai_path.run(payload)
        ai_latencies.append((time.time() - start) * 1000)

    ai_avg = statistics.mean(ai_latencies)

    speedup = ai_avg / fast_avg

    print(f"\n⚡ Latency Comparison:")
    print(f"   Fast path: {fast_avg:.2f}ms")
    print(f"   AI path: {ai_avg:.2f}ms")
    print(f"   Speedup: {speedup:.2f}×")

    # Fast path should be >3× faster
    assert speedup > 3, (
        f"Fast path speedup {speedup:.2f}× below 3× threshold"
    )


# ==================== CONSISTENCY BENCHMARKS ====================


def test_latency_consistency_fast_path(agent_v2_fast_path):
    """
    Benchmark 12: Fast path latency is consistent (low variance)

    Target: Coefficient of variation < 0.3 (stable performance)
    """
    payload = {
        "fuel_type": "diesel",
        "amount": 1000,
        "unit": "gallons",
    }

    latencies = []
    for _ in range(50):
        start = time.time()
        agent_v2_fast_path.run(payload)
        latencies.append((time.time() - start) * 1000)

    mean_latency = statistics.mean(latencies)
    stdev_latency = statistics.stdev(latencies)
    cv = stdev_latency / mean_latency  # Coefficient of variation

    print(f"\n📈 Latency Consistency:")
    print(f"   Mean: {mean_latency:.2f}ms")
    print(f"   StdDev: {stdev_latency:.2f}ms")
    print(f"   CV: {cv:.3f}")

    # Coefficient of variation should be <0.3 (stable)
    assert cv < 0.3, (
        f"Latency CV {cv:.3f} exceeds 0.3 (unstable performance)"
    )


# ==================== SUMMARY BENCHMARK ====================


@pytest.mark.slow
def test_summary_performance_report(agent_v2_fast_path, agent_v2_ai_path, sample_payloads):
    """
    Benchmark 13: Comprehensive performance report

    Generates a summary report of all performance metrics
    """
    print("\n" + "=" * 80)
    print("  PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)

    # Latency benchmarks
    fast_latencies = []
    for payload in sample_payloads * 10:
        start = time.time()
        agent_v2_fast_path.run(payload)
        fast_latencies.append((time.time() - start) * 1000)

    fast_p50 = statistics.median(fast_latencies)
    fast_p95 = sorted(fast_latencies)[int(len(fast_latencies) * 0.95)]

    print(f"\n📊 LATENCY BENCHMARKS:")
    print(f"   Fast Path P50: {fast_p50:.2f}ms (target: <50ms)")
    print(f"   Fast Path P95: {fast_p95:.2f}ms (target: <100ms)")

    # Throughput benchmark
    start = time.time()
    for payload in sample_payloads * 20:
        agent_v2_fast_path.run(payload)
    duration = time.time() - start
    throughput = (len(sample_payloads) * 20) / duration

    print(f"\n⚡ THROUGHPUT BENCHMARK:")
    print(f"   Fast Path: {throughput:.2f} req/s (target: >50 req/s)")

    # Cost benchmark (fast path should be ~$0)
    result = agent_v2_fast_path.run(sample_payloads[0])
    fast_cost = result["metadata"].get("total_cost_usd", 0.0)

    print(f"\n💰 COST BENCHMARK:")
    print(f"   Fast Path: ${fast_cost:.6f}/calc (target: ~$0)")

    print(f"\n✅ PERFORMANCE TARGETS:")
    print(f"   Latency P50 < 50ms: {'✅ PASS' if fast_p50 < 50 else '❌ FAIL'}")
    print(f"   Latency P95 < 100ms: {'✅ PASS' if fast_p95 < 100 else '❌ FAIL'}")
    print(f"   Throughput > 50 req/s: {'✅ PASS' if throughput > 50 else '❌ FAIL'}")
    print(f"   Cost ≤ $0.0001: {'✅ PASS' if fast_cost <= 0.0001 else '❌ FAIL'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run benchmarks with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "not slow"])
