# -*- coding: utf-8 -*-
"""
Example Usage of A/B Testing Framework for Process Heat Agents

Demonstrates practical use cases for comparing Process Heat agent variants:
  - Boiler efficiency model comparison
  - Thermal command optimization
  - Combustion diagnostics improvements
  - Steam optimization strategies

Run these examples to understand the framework:
    >>> python -m greenlang.ml.experimentation.examples
"""

import numpy as np
from greenlang.ml.experimentation import ABTestManager, MetricType, TestType


def example_1_boiler_efficiency_comparison():
    """
    Example 1: Compare boiler efficiency models.

    Scenario: Testing new boiler optimization model (GL-002) against baseline.
    Metrics: Thermal efficiency (%), fuel consumption (kg/h), emissions (g/kWh)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Boiler Efficiency Model Comparison")
    print("=" * 70)

    manager = ABTestManager()

    # Create experiment
    exp_id = manager.create_experiment(
        name="boiler_efficiency_v2_vs_baseline",
        variants=["baseline_gl002_v1", "optimized_gl002_v2"],
        traffic_split={"baseline_gl002_v1": 0.5, "optimized_gl002_v2": 0.5},
        metric_type=MetricType.CONTINUOUS,
        test_type=TestType.WELCH_T,
    )

    print(f"Created experiment: {exp_id}")
    print(f"Variants: baseline (50%), optimized (50%)")

    # Simulate operations over 200 time periods
    np.random.seed(42)
    baseline_efficiency = np.random.normal(loc=88.5, scale=2.0, size=200)
    optimized_efficiency = np.random.normal(loc=90.2, scale=1.8, size=200)

    for i in range(200):
        # Deterministic variant assignment based on timestamp
        variant = manager.assign_variant(f"boiler_run_{i}", exp_id)

        if variant == "baseline_gl002_v1":
            efficiency = baseline_efficiency[i]
        else:
            efficiency = optimized_efficiency[i]

        manager.record_metric(exp_id, variant, "thermal_efficiency", efficiency)

    # Analyze results
    result = manager.analyze_results(exp_id)

    print(f"\nResults after {result.current_sample_size} observations:")
    print(f"  Baseline mean efficiency: {result.variant_results['baseline_gl002_v1'].mean:.2f}%")
    print(f"  Optimized mean efficiency: {result.variant_results['optimized_gl002_v2'].mean:.2f}%")
    print(f"  Difference: {result.variant_results['optimized_gl002_v2'].mean - result.variant_results['baseline_gl002_v1'].mean:.2f}%")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Effect size (Cohen's d): {result.effect_size:.4f}")
    print(f"  Statistically significant: {result.is_significant}")
    print(f"  Winner: {result.winner}")

    # Get Prometheus metrics
    prometheus_metrics = manager.export_prometheus_metrics(exp_id)
    print(f"\nPrometheus metrics available for monitoring")


def example_2_combustion_diagnostics_strategy():
    """
    Example 2: Compare combustion diagnostics strategies.

    Scenario: Testing new fuel characterization algorithm (GL-005) impact
    on anomaly detection accuracy.
    Metrics: Anomaly detection rate (% accuracy)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Combustion Diagnostics Strategy Comparison")
    print("=" * 70)

    manager = ABTestManager()

    # Create experiment
    exp_id = manager.create_experiment(
        name="combustion_anomaly_detection_v2",
        variants=["rule_based", "ml_enhanced"],
        traffic_split={"rule_based": 0.5, "ml_enhanced": 0.5},
        metric_type=MetricType.CONTINUOUS,
        test_type=TestType.WELCH_T,
    )

    print(f"Created experiment: {exp_id}")

    # Simulate diagnostic sessions
    np.random.seed(123)
    rule_accuracy = np.random.normal(loc=82.5, scale=5.0, size=150)
    ml_accuracy = np.random.normal(loc=87.3, scale=4.2, size=150)

    for i in range(150):
        variant = manager.assign_variant(f"diagnostic_session_{i}", exp_id)

        if variant == "rule_based":
            accuracy = rule_accuracy[i]
        else:
            accuracy = ml_accuracy[i]

        manager.record_metric(exp_id, variant, "anomaly_detection_accuracy", accuracy)

    # Analyze
    result = manager.analyze_results(exp_id)

    print(f"\nResults after {result.current_sample_size} observations:")
    print(f"  Rule-based accuracy: {result.variant_results['rule_based'].mean:.2f}%")
    print(f"  ML-enhanced accuracy: {result.variant_results['ml_enhanced'].mean:.2f}%")
    print(f"  Improvement: {result.variant_results['ml_enhanced'].mean - result.variant_results['rule_based'].mean:.2f}%")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Effect size: {result.effect_size:.4f}")
    print(f"  Recommended samples needed: {result.min_sample_size}")
    print(f"  Winner: {result.winner}")


def example_3_steam_optimization_variants():
    """
    Example 3: Compare steam distribution optimization strategies.

    Scenario: Testing unified steam distribution model (GL-003) variants
    Metric: Steam loss (%)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Steam Distribution Optimization Comparison")
    print("=" * 70)

    manager = ABTestManager()

    # Create multi-variant experiment
    exp_id = manager.create_experiment(
        name="steam_distribution_optimization",
        variants=["baseline", "flash_recovery_enabled", "condensate_return_optimized"],
        traffic_split={
            "baseline": 0.34,
            "flash_recovery_enabled": 0.33,
            "condensate_return_optimized": 0.33,
        },
        metric_type=MetricType.CONTINUOUS,
        test_type=TestType.WELCH_T,
    )

    print(f"Created experiment: {exp_id}")
    print(f"Testing 3 variants with ~equal traffic split")

    # Simulate steam system operations
    np.random.seed(456)
    baseline_loss = np.random.normal(loc=8.5, scale=1.2, size=180)
    flash_loss = np.random.normal(loc=6.2, scale=1.1, size=180)
    condensate_loss = np.random.normal(loc=5.8, scale=1.0, size=180)

    data = {
        "baseline": baseline_loss,
        "flash_recovery_enabled": flash_loss,
        "condensate_return_optimized": condensate_loss,
    }

    for i in range(180):
        variant = manager.assign_variant(f"steam_operation_{i}", exp_id)
        loss = data[variant][i]
        manager.record_metric(exp_id, variant, "steam_loss_percent", loss)

    # Analyze
    result = manager.analyze_results(exp_id)

    print(f"\nResults after {result.current_sample_size} observations:")
    for variant in result.variant_results:
        metrics = result.variant_results[variant]
        print(f"  {variant}:")
        print(f"    - Mean loss: {metrics.mean:.2f}%")
        print(f"    - Std dev: {metrics.std:.2f}%")
        print(f"    - Samples: {metrics.n_samples}")

    print(f"\nComparison:")
    print(f"  Winner: {result.winner}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Effect size: {result.effect_size:.4f}")


def example_4_deterministic_assignment():
    """
    Example 4: Demonstrate deterministic variant assignment.

    Shows that the same user always gets the same variant.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Deterministic Variant Assignment")
    print("=" * 70)

    manager = ABTestManager()

    exp_id = manager.create_experiment(
        name="deterministic_test",
        variants=["variant_a", "variant_b"],
        traffic_split={"variant_a": 0.5, "variant_b": 0.5},
    )

    # Test determinism
    user_id = "boiler_unit_123"
    assignments = []

    for _ in range(5):
        variant = manager.assign_variant(user_id, exp_id)
        assignments.append(variant)

    print(f"User '{user_id}' assigned variant: {assignments[0]}")
    print(f"Consistency check (5 assignments):")
    for i, variant in enumerate(assignments):
        print(f"  Assignment {i+1}: {variant}")

    assert len(set(assignments)) == 1, "Assignments not deterministic!"
    print("[OK] Assignments are deterministic (same user -> same variant)")


def example_5_sample_size_planning():
    """
    Example 5: Plan sample size for desired statistical power.

    Shows how to determine experiment duration needed.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Sample Size Planning")
    print("=" * 70)

    from greenlang.ml.experimentation import StatisticalAnalyzer

    # Different effect sizes and their required samples
    effect_sizes = [0.1, 0.2, 0.5, 0.8]

    print("Sample size required per variant (α=0.05, power=0.80):\n")
    print(f"{'Effect Size':<15} {'Sample/Variant':<20} {'Total Samples':<15}")
    print("-" * 50)

    for effect_size in effect_sizes:
        sample_per_variant = StatisticalAnalyzer.calculate_sample_size(
            effect_size=effect_size,
            alpha=0.05,
            power=0.80,
        )
        total = sample_per_variant * 2

        print(f"{effect_size:<15.1f} {sample_per_variant:<20} {total:<15}")

    print("\nFor typical Process Heat improvements:")
    print("  - Small effect (0.2): ~400 observations per variant (~800 total)")
    print("  - Medium effect (0.5): ~64 observations per variant (~128 total)")
    print("  - Large effect (0.8): ~25 observations per variant (~50 total)")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GreenLang A/B Testing Framework - Process Heat Agent Examples")
    print("=" * 70)

    example_1_boiler_efficiency_comparison()
    example_2_combustion_diagnostics_strategy()
    example_3_steam_optimization_variants()
    example_4_deterministic_assignment()
    example_5_sample_size_planning()

    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
