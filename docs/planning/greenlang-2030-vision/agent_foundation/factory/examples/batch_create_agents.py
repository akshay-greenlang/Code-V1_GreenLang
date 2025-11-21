# -*- coding: utf-8 -*-
"""
Example: Batch Create Multiple Agents using Agent Factory.

This example demonstrates how to create multiple agents in parallel
for maximum throughput and efficiency.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factory.agent_factory import AgentFactory, AgentSpecification


def main():
    """Create multiple calculator agents in batch."""

    print("=" * 80)
    print("GreenLang Agent Factory - Batch Agent Creation Example")
    print("=" * 80)
    print()

    # Initialize factory
    factory = AgentFactory(
        output_directory=Path("./generated_agents"),
        parallel_execution=True,
        cache_templates=True,
        max_workers=8
    )

    # Define multiple agent specifications
    agent_specs = [
        AgentSpecification(
            name="Scope1EmissionsCalculator",
            type="calculator",
            description="Calculate Scope 1 direct GHG emissions",
            input_schema={
                "fuel_consumption": "float",
                "fuel_type": "str",
                "emission_factor": "float"
            },
            output_schema={
                "emissions_co2": "float",
                "emissions_ch4": "float",
                "emissions_n2o": "float",
                "total_co2e": "float"
            },
            calculation_formulas={
                "emissions_co2": "fuel_consumption * emission_factor * 0.95",
                "emissions_ch4": "fuel_consumption * emission_factor * 0.04",
                "emissions_n2o": "fuel_consumption * emission_factor * 0.01",
                "total_co2e": "emissions_co2 + (emissions_ch4 * 25) + (emissions_n2o * 298)"
            },
            test_coverage_target=85
        ),

        AgentSpecification(
            name="Scope2EmissionsCalculator",
            type="calculator",
            description="Calculate Scope 2 indirect GHG emissions from purchased energy",
            input_schema={
                "electricity_consumption": "float",
                "grid_emission_factor": "float",
                "location": "str"
            },
            output_schema={
                "emissions_location_based": "float",
                "emissions_market_based": "float"
            },
            calculation_formulas={
                "emissions_location_based": "electricity_consumption * grid_emission_factor",
                "emissions_market_based": "electricity_consumption * grid_emission_factor * 0.8"
            },
            test_coverage_target=85
        ),

        AgentSpecification(
            name="Scope3EmissionsCalculator",
            type="calculator",
            description="Calculate Scope 3 value chain GHG emissions",
            input_schema={
                "activity_data": "float",
                "category": "int",
                "emission_factor": "float"
            },
            output_schema={
                "emissions": "float",
                "category_name": "str",
                "uncertainty": "float"
            },
            calculation_formulas={
                "emissions": "activity_data * emission_factor",
                "uncertainty": "emissions * 0.2"  # 20% uncertainty estimate
            },
            test_coverage_target=85
        ),

        AgentSpecification(
            name="CarbonIntensityCalculator",
            type="calculator",
            description="Calculate carbon intensity metrics",
            input_schema={
                "total_emissions": "float",
                "revenue": "float",
                "employees": "int",
                "floor_area": "float"
            },
            output_schema={
                "intensity_per_revenue": "float",
                "intensity_per_employee": "float",
                "intensity_per_sqm": "float"
            },
            calculation_formulas={
                "intensity_per_revenue": "total_emissions / revenue",
                "intensity_per_employee": "total_emissions / employees",
                "intensity_per_sqm": "total_emissions / floor_area"
            },
            test_coverage_target=85
        ),

        AgentSpecification(
            name="NetZeroProgressCalculator",
            type="calculator",
            description="Calculate progress towards net-zero targets",
            input_schema={
                "current_emissions": "float",
                "baseline_emissions": "float",
                "target_year": "int",
                "current_year": "int"
            },
            output_schema={
                "reduction_achieved": "float",
                "reduction_required": "float",
                "on_track": "bool",
                "years_remaining": "int"
            },
            calculation_formulas={
                "reduction_achieved": "(baseline_emissions - current_emissions) / baseline_emissions * 100",
                "years_remaining": "target_year - current_year",
                "reduction_required": "100 - reduction_achieved"
            },
            test_coverage_target=85
        )
    ]

    # Create agents in batch
    print(f"Creating {len(agent_specs)} agents in parallel...")
    print()

    start_time = time.perf_counter()

    results = factory.create_agent_batch(
        agent_specs,
        parallel=True,
        generate_tests=True,
        generate_docs=True,
        create_pack=False,
        validate=True
    )

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000

    # Display results
    print()
    print("=" * 80)
    print("BATCH CREATION RESULTS")
    print("=" * 80)
    print()

    successful = 0
    failed = 0
    total_loc = 0
    total_tests = 0
    quality_scores = []

    for i, result in enumerate(results):
        if result.success:
            successful += 1
            total_loc += result.lines_of_code
            total_tests += result.test_count
            quality_scores.append(result.quality_score)

            print(f"✓ {result.agent_name}")
            print(f"  Generation Time: {result.generation_time_ms:.2f}ms")
            print(f"  Quality Score: {result.quality_score:.1f}%")
            print(f"  LOC: {result.lines_of_code}, Tests: {result.test_count}")
            print()
        else:
            failed += 1
            print(f"✗ {agent_specs[i].name} - FAILED")
            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")
            print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print(f"Total Agents: {len(agent_specs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    print(f"Total Time: {total_time_ms:.2f}ms")
    print(f"Average Time per Agent: {total_time_ms/len(agent_specs):.2f}ms")
    print()

    print(f"Total Lines of Code: {total_loc}")
    print(f"Total Tests Generated: {total_tests}")
    print()

    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)

        print(f"Quality Scores:")
        print(f"  Average: {avg_quality:.1f}%")
        print(f"  Range: {min_quality:.1f}% - {max_quality:.1f}%")
        print()

    # Factory statistics
    stats = factory.get_metrics()
    print("=" * 80)
    print("FACTORY STATISTICS")
    print("=" * 80)
    print()
    print(f"Total Agents Created: {stats['agents_created']}")
    print(f"Average Generation Time: {stats['average_generation_time_ms']:.2f}ms")
    print(f"Fastest: {stats['fastest_ms']:.2f}ms")
    print(f"Slowest: {stats['slowest_ms']:.2f}ms")
    print()

    print("All agents generated in:", Path("./generated_agents").absolute())
    print()


if __name__ == "__main__":
    main()
