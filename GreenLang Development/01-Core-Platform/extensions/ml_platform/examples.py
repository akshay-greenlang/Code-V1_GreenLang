"""
GreenLang ML Platform - Example Usage

Demonstrates complete workflows using:
- Model API for invocation
- Golden test execution
- Determinism validation
- Model routing with cost optimization
"""

import asyncio
from datetime import datetime

from greenlang.registry.model_registry import (
    model_registry,
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
    ModelInvokeResponse
)


# ============================================================================
# EXAMPLE 1: Basic Model Routing
# ============================================================================

def example_1_basic_routing():
    """Example 1: Select best model based on criteria."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Model Routing")
    print("=" * 80)

    router = ModelRouter()

    # Scenario: Need a certified model for text generation, lowest cost
    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        certified_only=True,
        strategy=RoutingStrategy.LOWEST_COST
    )

    decision = router.route(criteria)

    print(f"\n✓ Selected Model: {decision.primary_model.name}")
    print(f"  Provider: {decision.primary_model.provider}")
    print(f"  Cost: ${decision.primary_model.avg_cost_per_1k_tokens:.6f} per 1k tokens")
    print(f"  Certified: {decision.primary_model.certified_for_zero_hallucination}")
    print(f"  Reason: {decision.reason}")
    print(f"  Fallbacks available: {len(decision.fallback_models)}")


# ============================================================================
# EXAMPLE 2: Cost-Constrained Routing
# ============================================================================

def example_2_cost_constrained():
    """Example 2: Route with strict cost constraints."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Cost-Constrained Routing")
    print("=" * 80)

    router = ModelRouter()

    # Scenario: Need code generation under $0.005 per 1k tokens
    criteria = RoutingCriteria(
        capability=ModelCapability.CODE_GENERATION,
        max_cost_per_1k_tokens=0.005,
        certified_only=True,
        strategy=RoutingStrategy.BALANCED
    )

    try:
        decision = router.route(criteria)
        print(f"\n✓ Found model within budget:")
        print(f"  Model: {decision.primary_model.name}")
        print(f"  Cost: ${decision.primary_model.avg_cost_per_1k_tokens:.6f}/1k tokens")
        print(f"  Under budget: {decision.primary_model.avg_cost_per_1k_tokens <= 0.005}")
    except ValueError as e:
        print(f"\n✗ No models found: {e}")


# ============================================================================
# EXAMPLE 3: Golden Test Execution
# ============================================================================

async def example_3_golden_tests():
    """Example 3: Run golden tests to validate model behavior."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Golden Test Execution")
    print("=" * 80)

    # Create test suite for carbon footprint calculations
    tests = [
        GoldenTest(
            id="cf_test_1",
            name="Basic carbon footprint",
            prompt="What is a carbon footprint?",
            expected_response="A carbon footprint is the total amount of greenhouse gas emissions...",
            temperature=0.0,
            tags=["carbon", "basic"]
        ),
        GoldenTest(
            id="cf_test_2",
            name="Scope 1 emissions",
            prompt="Define Scope 1 emissions",
            expected_response="Scope 1 emissions are direct GHG emissions from sources owned or controlled...",
            temperature=0.0,
            tags=["carbon", "scopes"]
        ),
        GoldenTest(
            id="cf_test_3",
            name="GHG Protocol",
            prompt="What is the GHG Protocol?",
            expected_response="The GHG Protocol is the most widely used international accounting tool...",
            temperature=0.0,
            tags=["protocol"]
        )
    ]

    # Run tests
    executor = GoldenTestExecutor()
    report = await executor.run_golden_tests(
        model_id="claude-sonnet-4",
        tests=tests,
        test_suite_name="carbon_footprint_basic",
        check_determinism=True,
        determinism_runs=3
    )

    # Display results
    print(f"\n✓ Test Execution Complete:")
    print(f"  Total tests: {report.total_tests}")
    print(f"  Passed: {report.tests_passed} ({report.pass_rate*100:.1f}%)")
    print(f"  Failed: {report.tests_failed}")
    print(f"  Duration: {report.duration_seconds:.2f}s")

    print(f"\n  Performance Metrics:")
    print(f"    Avg latency: {report.performance_metrics.avg_latency_ms:.2f}ms")
    print(f"    P50 latency: {report.performance_metrics.p50_latency_ms:.2f}ms")
    print(f"    P95 latency: {report.performance_metrics.p95_latency_ms:.2f}ms")
    print(f"    Total tokens: {report.performance_metrics.total_tokens}")
    print(f"    Total cost: ${report.performance_metrics.total_cost_usd:.6f}")

    # Show individual test results
    print(f"\n  Test Results:")
    for result in report.test_results:
        status_icon = "✓" if result.passed else "✗"
        print(f"    {status_icon} {result.test_name}")
        print(f"      Exact match: {result.exact_match}")
        print(f"      Similarity: {result.similarity_score:.2%}")
        print(f"      Latency: {result.latency_ms:.2f}ms")

    # Show determinism results
    if report.determinism_results:
        print(f"\n  Determinism Validation:")
        for det_result in report.determinism_results:
            icon = "✓" if det_result.all_identical else "✗"
            print(f"    {icon} Test {det_result.test_id}")
            print(f"      Runs: {det_result.num_runs}")
            print(f"      Unique responses: {det_result.unique_responses}")


# ============================================================================
# EXAMPLE 4: Determinism Validation
# ============================================================================

async def example_4_determinism():
    """Example 4: Validate bit-perfect reproducibility."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Determinism Validation")
    print("=" * 80)

    validator = DeterminismValidator()

    # Test critical calculation
    prompt = "Calculate emissions: 100 kWh * 0.5 kgCO2e/kWh"

    print(f"\nValidating determinism for prompt:")
    print(f"  '{prompt}'")
    print(f"  Running 5 iterations...")

    result = await validator.validate_determinism(
        model_id="claude-sonnet-4",
        prompt=prompt,
        runs=5,
        temperature=0.0
    )

    print(f"\n✓ Determinism Check Results:")
    print(f"  All identical: {result.all_identical}")
    print(f"  Unique responses: {result.unique_responses}")
    print(f"  Variance detected: {result.variance_detected}")

    if result.all_identical:
        print(f"  ✓ PASS: Model produces bit-perfect reproducible outputs")
    else:
        print(f"  ✗ FAIL: Model outputs vary across runs")
        print(f"  Unique hashes:")
        for i, hash_val in enumerate(set(result.hashes), 1):
            print(f"    {i}. {hash_val}")


# ============================================================================
# EXAMPLE 5: Invocation with Automatic Fallback
# ============================================================================

async def example_5_fallback():
    """Example 5: Invoke model with automatic fallback."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Invocation with Automatic Fallback")
    print("=" * 80)

    router = ModelRouter()

    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        certified_only=True,
        enable_fallback=True,
        max_fallback_attempts=3,
        strategy=RoutingStrategy.LOWEST_COST
    )

    prompt = "Explain carbon neutrality in climate policy"

    print(f"\nInvoking model with fallback enabled...")
    print(f"  Prompt: '{prompt[:50]}...'")
    print(f"  Max attempts: {criteria.max_fallback_attempts + 1}")

    result = await router.invoke_with_fallback(
        criteria=criteria,
        prompt=prompt,
        temperature=0.0
    )

    print(f"\n✓ Invocation Complete:")
    print(f"  Model used: {result.model_id}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Fallback used: {result.fallback_used}")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    print(f"  Tokens: {result.input_tokens} + {result.output_tokens} = {result.input_tokens + result.output_tokens}")
    print(f"  Cost: ${result.cost_usd:.6f}")
    print(f"  Response length: {len(result.response)} chars")


# ============================================================================
# EXAMPLE 6: Cost Optimization
# ============================================================================

def example_6_cost_optimization():
    """Example 6: Automatic cost optimization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Cost Optimization")
    print("=" * 80)

    optimizer = CostOptimizer()

    # Test with different prompt types
    test_cases = [
        ("What is 2+2?", "simple calculation"),
        ("List the greenhouse gases", "simple list"),
        ("Perform detailed lifecycle assessment of solar panels including manufacturing, transport, installation, operation, and end-of-life disposal", "complex analysis")
    ]

    print("\nCost optimization recommendations:")
    for prompt, description in test_cases:
        should_use_cheaper = optimizer.should_use_cheaper_model(
            prompt=prompt,
            capability=ModelCapability.TEXT_GENERATION
        )

        recommendation = "Use cheaper model" if should_use_cheaper else "Use primary model"
        print(f"\n  Prompt: '{prompt[:60]}...'")
        print(f"  Type: {description}")
        print(f"  Recommendation: {recommendation}")


# ============================================================================
# EXAMPLE 7: Load Balancing
# ============================================================================

def example_7_load_balancing():
    """Example 7: Load balance across models."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Load Balancing")
    print("=" * 80)

    # Set up load balancer with weights
    models = ["claude-sonnet-4", "claude-opus-4", "gpt-4"]
    weights = [0.6, 0.3, 0.1]  # 60% sonnet, 30% opus, 10% gpt-4

    balancer = LoadBalancer(models, weights)

    print(f"\nLoad balancer configuration:")
    print(f"  Models: {models}")
    print(f"  Weights: {weights}")

    # Simulate 100 requests
    selections = [balancer.select_model() for _ in range(100)]
    counts = {model: selections.count(model) for model in models}

    print(f"\nDistribution after 100 requests:")
    for model, count in counts.items():
        percentage = count / 100 * 100
        print(f"  {model}: {count} requests ({percentage:.1f}%)")


# ============================================================================
# EXAMPLE 8: Metrics Collection
# ============================================================================

def example_8_metrics():
    """Example 8: Collect and analyze performance metrics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Metrics Collection")
    print("=" * 80)

    collector = MetricsCollector()

    # Simulate 20 invocations
    print("\nSimulating 20 model invocations...")
    for i in range(20):
        collector.record_invocation(
            model_id="claude-sonnet-4",
            latency_ms=100 + i * 5,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.00045,
            success=True if i < 19 else False  # 1 failure
        )

    # Get metrics
    metrics = collector.get_aggregated_metrics("claude-sonnet-4")

    print(f"\n✓ Aggregated Metrics:")
    print(f"  Total invocations: {metrics['total_invocations']}")
    print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  P50 latency: {metrics['p50_latency_ms']:.2f}ms")
    print(f"  P95 latency: {metrics['p95_latency_ms']:.2f}ms")
    print(f"  P99 latency: {metrics['p99_latency_ms']:.2f}ms")
    print(f"  Total cost: ${metrics['total_cost_usd']:.6f}")
    print(f"  Avg cost/request: ${metrics['avg_cost_per_request']:.6f}")


# ============================================================================
# EXAMPLE 9: Complete Workflow
# ============================================================================

async def example_9_complete_workflow():
    """Example 9: Complete end-to-end workflow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Complete End-to-End Workflow")
    print("=" * 80)

    # Step 1: Route to best model
    print("\nStep 1: Model Routing")
    router = ModelRouter()
    criteria = RoutingCriteria(
        capability=ModelCapability.TEXT_GENERATION,
        max_cost_per_1k_tokens=0.005,
        certified_only=True,
        strategy=RoutingStrategy.BALANCED
    )
    model = router.select_model(criteria)
    print(f"  ✓ Selected: {model.name}")

    # Step 2: Create golden tests
    print("\nStep 2: Creating Golden Test Suite")
    tests = [
        GoldenTest(
            id="workflow_test_1",
            name="Carbon definition",
            prompt="What is carbon dioxide equivalent?",
            expected_response="CO2 equivalent is a measure...",
            temperature=0.0
        )
    ]
    print(f"  ✓ Created {len(tests)} tests")

    # Step 3: Run tests
    print("\nStep 3: Running Golden Tests")
    executor = GoldenTestExecutor()
    report = await executor.run_golden_tests(
        model_id=model.id,
        tests=tests,
        test_suite_name="workflow_test"
    )
    print(f"  ✓ Pass rate: {report.pass_rate*100:.1f}%")
    print(f"  ✓ Avg latency: {report.performance_metrics.avg_latency_ms:.2f}ms")

    # Step 4: Validate determinism
    print("\nStep 4: Validating Determinism")
    validator = DeterminismValidator()
    det_result = await validator.validate_determinism(
        model_id=model.id,
        prompt=tests[0].prompt,
        runs=3,
        temperature=0.0
    )
    print(f"  ✓ All identical: {det_result.all_identical}")

    # Step 5: Get routing statistics
    print("\nStep 5: Routing Statistics")
    stats = router.get_routing_statistics()
    print(f"  ✓ Total routes: {stats['total_routes']}")

    print("\n✓ Workflow Complete!")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("GreenLang ML Platform - Example Usage")
    print("=" * 80)

    # Synchronous examples
    example_1_basic_routing()
    example_2_cost_constrained()
    example_6_cost_optimization()
    example_7_load_balancing()
    example_8_metrics()

    # Asynchronous examples
    await example_3_golden_tests()
    await example_4_determinism()
    await example_5_fallback()
    await example_9_complete_workflow()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
