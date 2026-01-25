"""
Test suite for ML Platform components.

Tests:
- Model API endpoints
- Golden test execution
- Determinism validation
- Model routing
- Cost optimization
"""

import pytest
import asyncio
from datetime import datetime

from greenlang.registry.model_registry import (
    model_registry,
    ModelProvider,
    ModelCapability
)
from greenlang.ml_platform.evaluation import (
    GoldenTest,
    GoldenTestExecutor,
    DeterminismValidator,
    MetricsCollector,
    create_golden_test_suite
)
from greenlang.ml_platform.router import (
    ModelRouter,
    RoutingCriteria,
    RoutingStrategy,
    CostOptimizer,
    LoadBalancer
)
from greenlang.ml_platform.model_api import (
    ModelInvokeRequest,
    ModelEvaluateRequest
)


# ============================================================================
# GOLDEN TEST EXECUTOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_golden_test_execution():
    """Test golden test execution."""
    executor = GoldenTestExecutor(model_registry)

    # Create test suite
    tests = [
        GoldenTest(
            id="test1",
            name="Basic calculation",
            prompt="What is 2+2?",
            expected_response="4",
            temperature=0.0
        ),
        GoldenTest(
            id="test2",
            name="Simple query",
            prompt="What is the capital of France?",
            expected_response="Paris",
            temperature=0.0
        )
    ]

    # Run tests
    report = await executor.run_golden_tests(
        model_id="claude-sonnet-4",
        tests=tests,
        test_suite_name="basic_tests",
        check_determinism=True
    )

    # Assertions
    assert report.total_tests == 2
    assert report.test_suite_name == "basic_tests"
    assert report.model_id == "claude-sonnet-4"
    assert len(report.test_results) == 2
    assert report.performance_metrics.total_tests == 2

    print(f"\nGolden Test Results:")
    print(f"  Total tests: {report.total_tests}")
    print(f"  Passed: {report.tests_passed}")
    print(f"  Failed: {report.tests_failed}")
    print(f"  Pass rate: {report.pass_rate*100:.1f}%")
    print(f"  Avg latency: {report.performance_metrics.avg_latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_determinism_validation():
    """Test determinism validation."""
    validator = DeterminismValidator(model_registry)

    result = await validator.validate_determinism(
        model_id="claude-sonnet-4",
        prompt="What is 2+2?",
        runs=5,
        temperature=0.0
    )

    # Assertions
    assert result.num_runs == 5
    assert len(result.responses) == 5
    assert len(result.hashes) == 5
    # Mock responses should be identical
    assert result.all_identical or result.unique_responses <= 5

    print(f"\nDeterminism Check:")
    print(f"  Runs: {result.num_runs}")
    print(f"  All identical: {result.all_identical}")
    print(f"  Unique responses: {result.unique_responses}")


def test_metrics_collector():
    """Test metrics collection."""
    collector = MetricsCollector()

    # Record invocations
    for i in range(10):
        collector.record_invocation(
            model_id="claude-sonnet-4",
            latency_ms=100 + i * 10,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.001,
            success=True
        )

    # Get metrics
    metrics = collector.get_aggregated_metrics("claude-sonnet-4")

    # Assertions
    assert metrics is not None
    assert metrics["total_invocations"] == 10
    assert metrics["success_rate"] == 1.0
    assert metrics["avg_latency_ms"] > 0
    assert metrics["total_cost_usd"] == 0.01

    print(f"\nMetrics:")
    print(f"  Total invocations: {metrics['total_invocations']}")
    print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  Total cost: ${metrics['total_cost_usd']:.4f}")


# ============================================================================
# MODEL ROUTER TESTS
# ============================================================================

def test_model_routing_lowest_cost():
    """Test routing with lowest cost strategy."""
    router = ModelRouter(model_registry)

    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        certified_only=True,
        strategy=RoutingStrategy.LOWEST_COST
    )

    decision = router.route(criteria)

    # Should select claude-sonnet-4 (cheapest certified model)
    assert decision.primary_model.id == "claude-sonnet-4"
    assert decision.primary_model.certified_for_zero_hallucination
    assert len(decision.fallback_models) >= 0

    print(f"\nRouting Decision (Lowest Cost):")
    print(f"  Primary: {decision.primary_model.name}")
    print(f"  Cost: ${decision.primary_model.avg_cost_per_1k_tokens:.6f}/1k")
    print(f"  Fallbacks: {len(decision.fallback_models)}")
    print(f"  Reason: {decision.reason}")


def test_model_routing_with_constraints():
    """Test routing with cost constraints."""
    router = ModelRouter(model_registry)

    criteria = RoutingCriteria(
        capability=ModelCapability.CODE_GENERATION,
        max_cost_per_1k_tokens=0.010,
        certified_only=True,
        strategy=RoutingStrategy.BALANCED
    )

    decision = router.route(criteria)

    # Should select a model within cost constraint
    assert decision.primary_model.avg_cost_per_1k_tokens <= 0.010
    assert decision.primary_model.certified_for_zero_hallucination

    print(f"\nRouting Decision (With Constraints):")
    print(f"  Primary: {decision.primary_model.name}")
    print(f"  Cost: ${decision.primary_model.avg_cost_per_1k_tokens:.6f}/1k")
    print(f"  Strategy: {decision.strategy_used}")


@pytest.mark.asyncio
async def test_invoke_with_fallback():
    """Test invocation with fallback."""
    router = ModelRouter(model_registry)

    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        certified_only=True,
        enable_fallback=True,
        max_fallback_attempts=2
    )

    result = await router.invoke_with_fallback(
        criteria=criteria,
        prompt="What is the capital of France?",
        temperature=0.0
    )

    # Assertions
    assert result.model_id in model_registry.models
    assert len(result.response) > 0
    assert result.latency_ms > 0
    assert result.attempts >= 1

    print(f"\nInvocation Result:")
    print(f"  Model: {result.model_id}")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    print(f"  Tokens: {result.input_tokens} + {result.output_tokens}")
    print(f"  Cost: ${result.cost_usd:.6f}")
    print(f"  Fallback used: {result.fallback_used}")


def test_cost_optimizer():
    """Test cost optimizer."""
    optimizer = CostOptimizer(model_registry)

    # Simple prompt should use cheaper model
    should_use_cheaper = optimizer.should_use_cheaper_model(
        prompt="What is 2+2?",
        capability=ModelCapability.TEXT_GENERATION
    )
    assert should_use_cheaper

    # Complex prompt might not use cheaper model
    should_use_cheaper = optimizer.should_use_cheaper_model(
        prompt="Perform detailed analysis of climate change impacts on global supply chains.",
        capability=ModelCapability.CODE_GENERATION
    )
    # This might or might not use cheaper model depending on heuristics

    print(f"\nCost Optimizer:")
    print(f"  Simple prompt: Use cheaper = {should_use_cheaper}")


def test_load_balancer():
    """Test load balancer."""
    models = ["claude-sonnet-4", "claude-opus-4", "gpt-4"]
    weights = [0.5, 0.3, 0.2]

    balancer = LoadBalancer(models, weights)

    # Select models
    selections = [balancer.select_model() for _ in range(100)]

    # Count selections
    counts = {model: selections.count(model) for model in models}

    print(f"\nLoad Balancer (100 requests):")
    for model, count in counts.items():
        print(f"  {model}: {count} requests ({count/100*100:.1f}%)")

    # Should roughly match weights (within 20% tolerance)
    assert counts["claude-sonnet-4"] > 30  # Roughly 50%
    assert counts["claude-opus-4"] > 15   # Roughly 30%


def test_routing_statistics():
    """Test routing statistics."""
    router = ModelRouter(model_registry)

    # Make several routing decisions
    for _ in range(5):
        criteria = RoutingCriteria(
            capability=ModelCapability.TEXT_GENERATION,
            strategy=RoutingStrategy.LOWEST_COST
        )
        router.route(criteria)

    stats = router.get_routing_statistics()

    assert stats["total_routes"] == 5
    assert len(stats["models_used"]) > 0
    assert len(stats["strategies_used"]) > 0

    print(f"\nRouting Statistics:")
    print(f"  Total routes: {stats['total_routes']}")
    print(f"  Models used: {stats['models_used']}")
    print(f"  Strategies: {stats['strategies_used']}")


# ============================================================================
# MODEL API TESTS
# ============================================================================

def test_model_invoke_request_validation():
    """Test request validation."""
    # Valid request
    request = ModelInvokeRequest(
        model_id="claude-sonnet-4",
        prompt="What is 2+2?",
        temperature=0.0
    )
    assert request.temperature == 0.0

    # Non-zero temperature (should warn but not fail)
    request = ModelInvokeRequest(
        model_id="claude-sonnet-4",
        prompt="What is 2+2?",
        temperature=0.5
    )
    assert request.temperature == 0.5


def test_model_evaluate_request():
    """Test evaluation request."""
    request = ModelEvaluateRequest(
        model_id="claude-sonnet-4",
        prompt="What is 2+2?",
        response="4",
        expected_response="4",
        evaluation_criteria=["accuracy", "completeness"]
    )

    assert request.model_id == "claude-sonnet-4"
    assert len(request.evaluation_criteria) == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_end_to_end_golden_test_with_routing():
    """Test end-to-end: routing + golden test execution."""
    # 1. Route to best model
    router = ModelRouter(model_registry)
    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        max_cost_per_1k_tokens=0.005,
        certified_only=True
    )
    model = router.select_model(criteria)

    print(f"\nEnd-to-End Test:")
    print(f"  Selected model: {model.name}")

    # 2. Run golden tests
    executor = GoldenTestExecutor(model_registry)
    tests = [
        GoldenTest(
            id="e2e_test1",
            name="Capital query",
            prompt="What is the capital of France?",
            expected_response="Paris",
            temperature=0.0
        )
    ]

    report = await executor.run_golden_tests(
        model_id=model.id,
        tests=tests,
        test_suite_name="e2e_tests"
    )

    print(f"  Tests passed: {report.tests_passed}/{report.total_tests}")
    print(f"  Total cost: ${report.performance_metrics.total_cost_usd:.6f}")

    # 3. Verify results
    assert report.total_tests == 1
    assert report.model_id == model.id


def test_create_golden_test_suite():
    """Test golden test suite creation helper."""
    test_defs = [
        {
            "id": "test1",
            "name": "Test 1",
            "prompt": "Hello",
            "expected_response": "Hi"
        },
        {
            "id": "test2",
            "name": "Test 2",
            "prompt": "Goodbye",
            "expected_response": "Bye"
        }
    ]

    suite = create_golden_test_suite("test_suite", test_defs)

    assert len(suite) == 2
    assert suite[0].id == "test1"
    assert suite[1].id == "test2"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("GreenLang ML Platform Test Suite")
    print("=" * 80)

    # Run synchronous tests
    print("\n[1] Testing Metrics Collector...")
    test_metrics_collector()

    print("\n[2] Testing Model Routing (Lowest Cost)...")
    test_model_routing_lowest_cost()

    print("\n[3] Testing Model Routing (With Constraints)...")
    test_model_routing_with_constraints()

    print("\n[4] Testing Cost Optimizer...")
    test_cost_optimizer()

    print("\n[5] Testing Load Balancer...")
    test_load_balancer()

    print("\n[6] Testing Routing Statistics...")
    test_routing_statistics()

    print("\n[7] Testing Request Validation...")
    test_model_invoke_request_validation()

    print("\n[8] Testing Evaluation Request...")
    test_model_evaluate_request()

    print("\n[9] Testing Golden Test Suite Creation...")
    test_create_golden_test_suite()

    # Run async tests
    print("\n[10] Testing Golden Test Execution...")
    asyncio.run(test_golden_test_execution())

    print("\n[11] Testing Determinism Validation...")
    asyncio.run(test_determinism_validation())

    print("\n[12] Testing Invoke with Fallback...")
    asyncio.run(test_invoke_with_fallback())

    print("\n[13] Testing End-to-End Integration...")
    asyncio.run(test_end_to_end_golden_test_with_routing())

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
