"""
Example: Create a Calculator Agent using Agent Factory.

This example demonstrates how to use the Agent Factory to generate
a production-ready calculator agent in <100ms with zero-hallucination guarantees.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factory.agent_factory import AgentFactory, AgentSpecification


def main():
    """Create a Carbon Emissions Calculator agent."""

    print("=" * 80)
    print("GreenLang Agent Factory - Calculator Agent Example")
    print("=" * 80)
    print()

    # Initialize factory
    factory = AgentFactory(
        output_directory=Path("./generated_agents"),
        parallel_execution=True,
        cache_templates=True
    )

    # Define agent specification
    spec = AgentSpecification(
        name="CarbonEmissionsCalculator",
        type="calculator",
        description="Calculate carbon emissions from activity data using GHG Protocol methodologies",

        # Schema definitions
        input_schema={
            "activity_data": "float",
            "emission_factor": "float",
            "activity_unit": "str",
            "scope": "int"
        },
        output_schema={
            "emissions": "float",
            "emissions_unit": "str",
            "intensity": "float",
            "scope": "int"
        },

        # Calculation formulas (zero-hallucination)
        calculation_formulas={
            "emissions": "activity_data * emission_factor",
            "intensity": "emissions / activity_data"
        },

        # Validation rules
        validation_rules=[
            {
                "name": "positive_values",
                "condition": "lambda x: all(v >= 0 for v in x.values() if isinstance(v, (int, float)))"
            },
            {
                "name": "valid_scope",
                "condition": "lambda x: x.get('scope', 1) in [1, 2, 3]"
            }
        ],

        # Performance targets
        performance_targets={
            "latency_ms": 500,
            "throughput_rps": 1000
        },

        # Quality requirements
        test_coverage_target=90,
        documentation_required=True,

        # Compliance
        compliance_frameworks=["GHG Protocol", "ISO 14064"],
        audit_requirements=True
    )

    # Create agent
    print("Creating agent...")
    print(f"  Name: {spec.name}")
    print(f"  Type: {spec.type}")
    print(f"  Description: {spec.description}")
    print()

    result = factory.create_agent(spec)

    # Display results
    print()
    print("=" * 80)
    print("CREATION RESULTS")
    print("=" * 80)
    print()

    if result.success:
        print("✓ Agent created successfully!")
        print()
        print(f"Agent ID: {result.agent_id}")
        print(f"Generation Time: {result.generation_time_ms:.2f}ms")
        print()

        print("Quality Metrics:")
        print(f"  Overall Quality Score: {result.quality_score:.1f}%")
        print(f"  Lines of Code: {result.lines_of_code}")
        print(f"  Test Count: {result.test_count}")
        print(f"  Deployable: {'Yes' if result.deployable else 'No'}")
        print()

        print("Generated Files:")
        print(f"  Code: {result.code_path}")
        print(f"  Tests: {result.test_path}")
        print(f"  Docs: {result.documentation_path}")
        if result.pack_id:
            print(f"  Pack ID: {result.pack_id}")
        print()

        if result.validation_result:
            print("Validation Details:")
            print(f"  Code Quality: {result.validation_result.code_quality_score:.1f}%")
            print(f"  Test Coverage: {result.validation_result.test_coverage:.1f}%")
            print(f"  Documentation: {result.validation_result.documentation_score:.1f}%")
            print(f"  Security: {result.validation_result.security_score:.1f}%")
            print()

            if result.validation_result.warnings:
                print(f"Warnings ({len(result.validation_result.warnings)}):")
                for warning in result.validation_result.warnings[:5]:
                    print(f"  - {warning['message']}")
                if len(result.validation_result.warnings) > 5:
                    print(f"  ... and {len(result.validation_result.warnings) - 5} more")
                print()

        print("Next Steps:")
        print("  1. Review generated code and tests")
        print("  2. Customize agent logic if needed")
        print("  3. Run tests: pytest", result.test_path)
        print("  4. Deploy to production when ready")
        print()

    else:
        print("✗ Agent creation failed!")
        print()
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
            print()

    # Factory statistics
    stats = factory.get_metrics()
    print("=" * 80)
    print("FACTORY STATISTICS")
    print("=" * 80)
    print()
    print(f"Total Agents Created: {stats['agents_created']}")
    print(f"Average Generation Time: {stats['average_generation_time_ms']:.2f}ms")
    print()


if __name__ == "__main__":
    main()
