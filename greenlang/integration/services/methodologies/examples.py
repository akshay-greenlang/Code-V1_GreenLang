# -*- coding: utf-8 -*-
"""
Methodologies Module - Usage Examples

This script demonstrates various use cases for the Methodologies module,
including data quality assessment, uncertainty quantification, and
Monte Carlo simulation.

Run: python -m services.methodologies.examples
"""

from datetime import datetime
from greenlang.utilities.determinism import DeterministicClock


def example_1_simple_emission_calculation():
    """Example 1: Simple emission calculation with uncertainty."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Simple Emission Calculation with Uncertainty")
    print("=" * 70)

    from services.methodologies import MonteCarloSimulator

    # Initialize simulator with fixed seed for reproducibility
    simulator = MonteCarloSimulator(seed=42)

    # Calculate emission: E = Activity × Emission Factor
    result = simulator.simple_propagation(
        activity_data=1000.0,  # kg of material
        activity_uncertainty=0.1,  # 10% uncertainty
        emission_factor=2.5,  # kg CO2e per kg material
        factor_uncertainty=0.15,  # 15% uncertainty
        iterations=10000,
    )

    print(f"\nActivity Data: 1,000 kg ± 10%")
    print(f"Emission Factor: 2.5 kg CO2e/kg ± 15%")
    print(f"\nResults (10,000 Monte Carlo iterations):")
    print(f"  Mean Emission: {result.mean:.2f} kg CO2e")
    print(f"  Std Deviation: {result.std_dev:.2f} kg CO2e")
    print(f"  Coefficient of Variation: {result.coefficient_of_variation:.1%}")
    print(f"\nConfidence Intervals:")
    print(f"  90% CI: [{result.p5:.2f}, {result.p95:.2f}] kg CO2e")
    print(f"  95% CI: [{result.confidence_90_lower:.2f}, {result.confidence_90_upper:.2f}] kg CO2e")
    print(f"\nPerformance:")
    print(f"  Computation Time: {result.computation_time:.3f} seconds")

    if result.sensitivity_indices:
        print(f"\nSensitivity Analysis:")
        for param, sensitivity in result.sensitivity_indices.items():
            print(f"  {param}: {abs(sensitivity):.1%}")


def example_2_data_quality_assessment():
    """Example 2: Comprehensive data quality assessment."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Data Quality Assessment (ILCD Pedigree Matrix)")
    print("=" * 70)

    from services.methodologies import PedigreeScore, PedigreeMatrixEvaluator

    # Create pedigree score for primary data from supplier
    pedigree = PedigreeScore(
        reliability=1,  # Verified measurements
        completeness=2,  # Good sample size
        temporal=1,  # <3 years old
        geographical=2,  # Similar region
        technological=1,  # Same technology
        reference_year=2024,
        data_year=2023,
        notes="Supplier-specific primary data with measurements",
    )

    print(f"\nPedigree Scores (1=excellent, 5=poor):")
    print(f"  Reliability: {pedigree.reliability} - {PedigreeMatrixEvaluator().get_dimension_description('reliability', pedigree.reliability)}")
    print(f"  Completeness: {pedigree.completeness}")
    print(f"  Temporal: {pedigree.temporal}")
    print(f"  Geographical: {pedigree.geographical}")
    print(f"  Technological: {pedigree.technological}")
    print(f"\nAverage Score: {pedigree.average_score:.2f}")
    print(f"Quality Label: {pedigree.quality_label}")

    # Generate quality report
    evaluator = PedigreeMatrixEvaluator()
    report = evaluator.generate_quality_report(pedigree)

    print(f"\nData Quality Index: {report['dqi_score']:.2f}/100")
    print(f"Combined Uncertainty: {report['combined_uncertainty']:.1%}")

    if report["improvement_opportunities"]:
        print(f"\nImprovement Opportunities:")
        for dim in report["improvement_opportunities"]:
            print(f"  - {dim.title()}")
    else:
        print(f"\nNo significant improvement opportunities identified.")


def example_3_dqi_calculation():
    """Example 3: DQI calculation with multiple components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Data Quality Index (DQI) Calculation")
    print("=" * 70)

    from services.methodologies import (
        PedigreeScore,
        DQICalculator,
    )

    calculator = DQICalculator()

    # Scenario A: High quality data
    print("\nScenario A: Primary Data (Excellent Quality)")
    pedigree_a = PedigreeScore(
        reliability=1,
        completeness=1,
        temporal=1,
        geographical=1,
        technological=1,
    )

    dqi_a = calculator.calculate_dqi(
        pedigree_score=pedigree_a,
        factor_source="primary_measured",
        data_tier=1,
        assessed_by="data_team",
    )

    print(f"  DQI Score: {dqi_a.score:.2f}/100")
    print(f"  Quality Label: {dqi_a.quality_label}")
    print(f"  Components:")
    print(f"    - Pedigree: {dqi_a.pedigree_contribution:.2f}")
    print(f"    - Source: {dqi_a.source_contribution:.2f}")
    print(f"    - Tier: {dqi_a.tier_contribution:.2f}")

    # Scenario B: Database data
    print("\nScenario B: Database Data (Good Quality)")
    pedigree_b = PedigreeScore(
        reliability=2,
        completeness=2,
        temporal=2,
        geographical=2,
        technological=2,
    )

    dqi_b = calculator.calculate_dqi(
        pedigree_score=pedigree_b,
        factor_source="ecoinvent",
        data_tier=2,
    )

    print(f"  DQI Score: {dqi_b.score:.2f}/100")
    print(f"  Quality Label: {dqi_b.quality_label}")

    # Scenario C: Estimated data
    print("\nScenario C: Estimated Data (Fair Quality)")
    pedigree_c = PedigreeScore(
        reliability=3,
        completeness=3,
        temporal=3,
        geographical=3,
        technological=3,
    )

    dqi_c = calculator.calculate_dqi(
        pedigree_score=pedigree_c,
        factor_source="proxy",
        data_tier=3,
    )

    print(f"  DQI Score: {dqi_c.score:.2f}/100")
    print(f"  Quality Label: {dqi_c.quality_label}")

    # Generate report with recommendations
    report_c = calculator.generate_dqi_report(dqi_c)
    if report_c["recommendations"]:
        print(f"\n  Recommendations:")
        for rec in report_c["recommendations"]:
            print(f"    - {rec}")


def example_4_uncertainty_quantification():
    """Example 4: Uncertainty quantification by category."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Category-Based Uncertainty Quantification")
    print("=" * 70)

    from services.methodologies import UncertaintyQuantifier

    quantifier = UncertaintyQuantifier()

    # Different categories with different tiers
    categories = [
        ("electricity", 1, 1000.0),
        ("natural_gas", 2, 500.0),
        ("road_transport", 2, 200.0),
        ("plastics", 3, 100.0),
    ]

    print("\nCategory-Based Uncertainties:")
    print(f"\n{'Category':<20} {'Tier':<6} {'Value':<12} {'Uncertainty':<15} {'Range':<30}")
    print("-" * 85)

    total_emission = 0
    total_uncertainty = 0

    for category, tier, value in categories:
        result = quantifier.quantify_uncertainty(
            mean=value, category=category, tier=tier
        )

        range_str = f"[{result.confidence_95_lower:.1f}, {result.confidence_95_upper:.1f}]"
        uncertainty_pct = f"{result.relative_std_dev:.1%}"

        print(
            f"{category:<20} {tier:<6} {value:<12.1f} {uncertainty_pct:<15} {range_str:<30}"
        )

        total_emission += result.mean
        total_uncertainty += result.variance

    total_std = total_uncertainty**0.5
    total_cv = total_std / total_emission if total_emission > 0 else 0

    print("-" * 85)
    print(f"{'TOTAL':<20} {'':<6} {total_emission:<12.1f} {total_cv:<15.1%}")


def example_5_complete_workflow():
    """Example 5: Complete workflow with quality tracking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Workflow (Quality + Uncertainty)")
    print("=" * 70)

    from services.methodologies import (
        PedigreeScore,
        DQICalculator,
        MonteCarloSimulator,
    )

    print("\nCalculating emissions for electricity consumption...")

    # Step 1: Assess data quality
    print("\n1. Data Quality Assessment")
    pedigree = PedigreeScore(
        reliability=1,
        completeness=2,
        temporal=1,
        geographical=2,
        technological=1,
    )

    dqi_calc = DQICalculator()
    dqi = dqi_calc.calculate_dqi(
        pedigree_score=pedigree, factor_source="ecoinvent", data_tier=1
    )

    print(f"   DQI Score: {dqi.score:.2f}/100 ({dqi.quality_label})")

    # Step 2: Calculate emission with uncertainty
    print("\n2. Monte Carlo Simulation")
    simulator = MonteCarloSimulator(seed=42)
    result = simulator.simple_propagation(
        activity_data=50000.0,  # 50,000 kWh
        activity_uncertainty=0.08,  # 8% uncertainty
        emission_factor=0.5,  # 0.5 kg CO2e/kWh
        factor_uncertainty=0.12,  # 12% uncertainty
        iterations=10000,
    )

    print(f"   Iterations: {result.iterations:,}")
    print(f"   Mean Emission: {result.mean:.2f} kg CO2e")
    print(f"   Computation Time: {result.computation_time:.3f}s")

    # Step 3: Generate comprehensive report
    print("\n3. Comprehensive Report")
    print("   " + "-" * 60)
    print(f"   Electricity Consumption: 50,000 kWh")
    print(f"   Emission Factor: 0.5 kg CO2e/kWh")
    print(f"   ")
    print(f"   RESULTS:")
    print(f"   Total Emission: {result.mean:.2f} ± {result.std_dev:.2f} kg CO2e")
    print(f"   Relative Uncertainty: {result.coefficient_of_variation:.1%}")
    print(f"   ")
    print(f"   90% Confidence Interval:")
    print(f"     Lower Bound (p5): {result.p5:.2f} kg CO2e")
    print(f"     Median (p50): {result.p50:.2f} kg CO2e")
    print(f"     Upper Bound (p95): {result.p95:.2f} kg CO2e")
    print(f"   ")
    print(f"   Data Quality: {dqi.quality_label} (DQI: {dqi.score:.2f}/100)")
    print(f"   Timestamp: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("   " + "-" * 60)


def example_6_analytical_vs_monte_carlo():
    """Example 6: Analytical vs Monte Carlo comparison."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Analytical vs Monte Carlo Comparison")
    print("=" * 70)

    from services.methodologies import UncertaintyQuantifier

    quantifier = UncertaintyQuantifier()

    print("\nComparing propagation methods for: E = Activity × Factor")
    print(f"Activity: 1,000 ± 10%")
    print(f"Factor: 2.5 ± 15%")

    # Analytical method
    print("\n1. Analytical Propagation (Taylor Series)")
    result_analytical = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
        method="analytical",
    )

    print(f"   Mean: {result_analytical.mean:.2f}")
    print(f"   Std Dev: {result_analytical.std_dev:.2f}")
    print(f"   CV: {result_analytical.relative_std_dev:.2%}")
    print(f"   95% CI: [{result_analytical.confidence_95_lower:.2f}, {result_analytical.confidence_95_upper:.2f}]")

    # Monte Carlo method
    print("\n2. Monte Carlo Propagation (10,000 iterations)")
    result_mc = quantifier.propagate_simple(
        activity_mean=1000.0,
        activity_uncertainty=0.1,
        factor_mean=2.5,
        factor_uncertainty=0.15,
        method="monte_carlo",
    )

    print(f"   Mean: {result_mc.mean:.2f}")
    print(f"   Std Dev: {result_mc.std_dev:.2f}")
    print(f"   CV: {result_mc.relative_std_dev:.2%}")
    print(f"   95% CI: [{result_mc.confidence_95_lower:.2f}, {result_mc.confidence_95_upper:.2f}]")

    # Comparison
    print("\n3. Comparison")
    mean_diff = abs(result_mc.mean - result_analytical.mean)
    std_diff = abs(result_mc.std_dev - result_analytical.std_dev)

    print(f"   Mean difference: {mean_diff:.2f} ({mean_diff/result_analytical.mean:.2%})")
    print(f"   Std Dev difference: {std_diff:.2f} ({std_diff/result_analytical.std_dev:.2%})")
    print(f"\n   Conclusion: Both methods give similar results")
    print(f"   Analytical is ~1000× faster, Monte Carlo more accurate for complex cases")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 15 + "METHODOLOGIES MODULE - USAGE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the key features of the Methodologies module:")
    print("  1. Simple Emission Calculation with Uncertainty")
    print("  2. Data Quality Assessment (ILCD Pedigree Matrix)")
    print("  3. Data Quality Index (DQI) Calculation")
    print("  4. Category-Based Uncertainty Quantification")
    print("  5. Complete Workflow (Quality + Uncertainty)")
    print("  6. Analytical vs Monte Carlo Comparison")

    try:
        example_1_simple_emission_calculation()
        example_2_data_quality_assessment()
        example_3_dqi_calculation()
        example_4_uncertainty_quantification()
        example_5_complete_workflow()
        example_6_analytical_vs_monte_carlo()

        print("\n" + "=" * 70)
        print(" " * 20 + "ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
