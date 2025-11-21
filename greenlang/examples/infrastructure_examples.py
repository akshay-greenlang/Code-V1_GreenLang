# -*- coding: utf-8 -*-
"""
GreenLang Infrastructure Examples
Practical examples demonstrating all infrastructure components.
"""

import asyncio
import pandas as pd
from datetime import datetime
from greenlang.determinism import DeterministicClock


# ============================================================================
# EXAMPLE 1: Complete Data Intake Pipeline with Validation and Provenance
# ============================================================================

async def example_data_intake_pipeline():
    """
    Complete data intake pipeline with validation and provenance tracking.
    """
    from greenlang.agents.templates import IntakeAgent, DataFormat
    from greenlang.validation import ValidationFramework, SchemaValidator, RulesEngine, Rule, RuleOperator
    from greenlang.provenance import ProvenanceTracker

    print("=" * 60)
    print("EXAMPLE 1: Data Intake Pipeline")
    print("=" * 60)

    # Setup provenance tracking
    tracker = ProvenanceTracker(name="data_intake_pipeline")

    with tracker.track_operation("intake_and_validate"):
        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "facility": {"type": "string"},
                "emissions": {"type": "number", "minimum": 0},
                "date": {"type": "string"}
            },
            "required": ["facility", "emissions", "date"]
        }

        # Create intake agent
        intake_agent = IntakeAgent(schema=schema)

        # Sample data
        sample_data = pd.DataFrame({
            "facility": ["Plant A", "Plant B", "Plant C"],
            "emissions": [1000.5, 1500.2, 800.3],
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"]
        })

        # Ingest data
        result = await intake_agent.ingest(
            data=sample_data,
            format=DataFormat.CSV,
            validate=True
        )

        if result.success:
            print(f"✓ Ingested {result.rows_read} rows successfully")
            print(f"  Valid rows: {result.rows_valid}")

            # Track in provenance
            tracker.add_metadata("rows_ingested", result.rows_read)
            tracker.add_metadata("validation_passed", True)

        else:
            print("✗ Ingestion failed:")
            for issue in result.validation_issues:
                print(f"  {issue.severity}: {issue.message}")

    # Save provenance
    record = tracker.get_record()
    print(f"\n✓ Provenance tracked: {record.record_id}")
    print(f"  Operations: {len(tracker.chain_of_custody)}")


# ============================================================================
# EXAMPLE 2: Emissions Calculation with Parallel Processing
# ============================================================================

async def example_emissions_calculation():
    """
    High-performance emissions calculation with parallel processing.
    """
    from greenlang.agents.templates import CalculatorAgent
    from greenlang.cache import CacheManager
    from greenlang.telemetry import get_metrics_collector

    print("\n" + "=" * 60)
    print("EXAMPLE 2: Emissions Calculation (Parallel)")
    print("=" * 60)

    # Initialize metrics
    metrics = get_metrics_collector()

    # Create calculator agent
    agent = CalculatorAgent(config={
        "thread_workers": 4,
        "process_workers": 2
    })

    # Register emission calculation formula
    def calculate_scope1_emissions(activity_data: float, emission_factor: float) -> float:
        """Calculate Scope 1 emissions."""
        return activity_data * emission_factor

    agent.register_formula(
        "scope1_emissions",
        calculate_scope1_emissions,
        required_inputs=["activity_data", "emission_factor"]
    )

    # Prepare batch calculations
    inputs_list = [
        {"activity_data": 1000, "emission_factor": 2.5, "unit": "kg CO2e"},
        {"activity_data": 1500, "emission_factor": 3.2, "unit": "kg CO2e"},
        {"activity_data": 800, "emission_factor": 1.8, "unit": "kg CO2e"},
        {"activity_data": 2000, "emission_factor": 2.9, "unit": "kg CO2e"},
        {"activity_data": 1200, "emission_factor": 2.1, "unit": "kg CO2e"},
    ]

    # Execute in parallel
    start_time = DeterministicClock.now()
    results = await agent.batch_calculate(
        formula_name="scope1_emissions",
        inputs_list=inputs_list,
        parallel=True,
        use_processes=False  # Use thread pool
    )
    duration = (DeterministicClock.now() - start_time).total_seconds()

    # Process results
    total_emissions = sum(r.value for r in results if r.success)

    print(f"✓ Calculated {len(results)} emissions in {duration:.3f}s")
    print(f"  Total emissions: {total_emissions:.2f} kg CO2e")
    print(f"  Success rate: {sum(1 for r in results if r.success)}/{len(results)}")

    # Track metrics
    metrics.increment("calculations.completed", len(results))
    metrics.record("calculation.duration", duration)

    # Show agent stats
    stats = agent.get_stats()
    print(f"\nAgent Statistics:")
    print(f"  Total calculations: {stats['total_calculations']}")
    print(f"  Parallel batches: {stats['parallel_calculations']}")
    print(f"  Cache size: {stats['cache_size']}")


# ============================================================================
# EXAMPLE 3: Multi-Format Report Generation with Charts
# ============================================================================

async def example_report_generation():
    """
    Generate reports in multiple formats with charts.
    """
    from greenlang.agents.templates import ReportingAgent, ReportFormat, ComplianceFramework

    print("\n" + "=" * 60)
    print("EXAMPLE 3: Report Generation")
    print("=" * 60)

    # Create reporting agent
    agent = ReportingAgent()

    # Sample emissions data
    emissions_data = pd.DataFrame({
        "category": ["Scope 1", "Scope 2", "Scope 3"],
        "emissions": [1500.5, 2300.8, 4200.3],
        "percentage": [18.8, 28.8, 52.4]
    })

    # Generate JSON report
    json_result = await agent.generate_report(
        data=emissions_data,
        format=ReportFormat.JSON
    )
    print(f"✓ Generated JSON report ({len(json_result.data)} bytes)")

    # Generate Excel report
    excel_result = await agent.generate_report(
        data=emissions_data,
        format=ReportFormat.EXCEL
    )
    print(f"✓ Generated Excel report ({len(excel_result.data)} bytes)")

    # Generate Markdown report
    md_result = await agent.generate_report(
        data=emissions_data,
        format=ReportFormat.MARKDOWN
    )
    print(f"✓ Generated Markdown report")
    print("\nMarkdown Preview:")
    print(md_result.data[:200] + "...")

    # Generate HTML report with charts
    chart_configs = [
        {
            "type": "bar",
            "x": "category",
            "y": "emissions",
            "title": "Emissions by Scope"
        }
    ]

    try:
        html_result = await agent.generate_with_charts(
            data=emissions_data,
            chart_configs=chart_configs,
            format=ReportFormat.HTML
        )
        print(f"\n✓ Generated HTML report with charts")
        print(f"  Charts included: {html_result.metadata.get('charts_count', 0)}")
    except ImportError:
        print("\n⚠ matplotlib not available, skipping chart generation")

    # Show agent stats
    stats = agent.get_stats()
    print(f"\nAgent Statistics:")
    print(f"  Total reports: {stats['total_reports']}")
    print(f"  By format: {stats['reports_by_format']}")


# ============================================================================
# EXAMPLE 4: Comprehensive Validation Pipeline
# ============================================================================

async def example_validation_pipeline():
    """
    Multi-layer validation with schema, rules, and quality checks.
    """
    from greenlang.validation import (
        ValidationFramework,
        SchemaValidator,
        RulesEngine,
        Rule,
        RuleOperator,
        DataQualityValidator
    )

    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multi-Layer Validation")
    print("=" * 60)

    # Create validation framework
    framework = ValidationFramework()

    # Layer 1: JSON Schema validation
    schema = {
        "type": "object",
        "properties": {
            "emissions": {"type": "number"},
            "facility": {"type": "string"},
            "year": {"type": "integer"}
        },
        "required": ["emissions", "facility", "year"]
    }
    schema_validator = SchemaValidator(schema)
    framework.add_validator("schema", schema_validator.validate)

    # Layer 2: Business rules
    rules_engine = RulesEngine()
    rules_engine.add_rule(Rule(
        name="emissions_non_negative",
        field="emissions",
        operator=RuleOperator.GREATER_EQUAL,
        value=0,
        message="Emissions cannot be negative"
    ))
    rules_engine.add_rule(Rule(
        name="valid_year",
        field="year",
        operator=RuleOperator.GREATER_EQUAL,
        value=2000,
        message="Year must be 2000 or later"
    ))
    framework.add_validator("business_rules", rules_engine.validate)

    # Test with valid data
    valid_data = {
        "emissions": 1500.5,
        "facility": "Plant A",
        "year": 2024
    }

    result = framework.validate(valid_data)
    print(f"\nValid Data Test:")
    print(f"  Status: {'✓ PASSED' if result.valid else '✗ FAILED'}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")

    # Test with invalid data
    invalid_data = {
        "emissions": -100,  # Negative emissions
        "facility": "Plant B",
        "year": 1999  # Too old
    }

    result = framework.validate(invalid_data)
    print(f"\nInvalid Data Test:")
    print(f"  Status: {'✓ PASSED' if result.valid else '✗ FAILED'}")
    print(f"  Errors: {len(result.errors)}")
    for error in result.errors:
        print(f"    - {error.field}: {error.message}")


# ============================================================================
# EXAMPLE 5: Cache-Optimized Data Pipeline
# ============================================================================

async def example_cached_pipeline():
    """
    Data pipeline with intelligent caching.
    """
    from greenlang.cache import CacheManager, initialize_cache_manager, get_cache_manager

    print("\n" + "=" * 60)
    print("EXAMPLE 5: Cache-Optimized Pipeline")
    print("=" * 60)

    # Initialize cache manager (L1 only for example)
    initialize_cache_manager(
        enable_l1=True,
        enable_l2=False,
        enable_l3=False
    )

    cache_manager = get_cache_manager()

    # Expensive computation
    async def compute_emission_factors():
        """Simulate expensive computation."""
        print("  Computing emission factors (expensive)...")
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "electricity": 0.5,
            "natural_gas": 2.0,
            "diesel": 2.7
        }

    # First call - cache miss
    print("\nFirst call (cache miss):")
    start = DeterministicClock.now()
    factors = await cache_manager.get_or_compute(
        key="emission_factors_2024",
        compute_fn=compute_emission_factors,
        ttl=3600
    )
    duration_miss = (DeterministicClock.now() - start).total_seconds()
    print(f"  Duration: {duration_miss:.3f}s")
    print(f"  Factors: {factors}")

    # Second call - cache hit
    print("\nSecond call (cache hit):")
    start = DeterministicClock.now()
    factors = await cache_manager.get_or_compute(
        key="emission_factors_2024",
        compute_fn=compute_emission_factors,
        ttl=3600
    )
    duration_hit = (DeterministicClock.now() - start).total_seconds()
    print(f"  Duration: {duration_hit:.3f}s")
    print(f"  Speedup: {duration_miss / duration_hit:.1f}x")

    # Analytics
    analytics = cache_manager.get_analytics()
    print(f"\nCache Analytics:")
    print(f"  Total requests: {analytics.total_requests}")
    print(f"  Hit rate: {analytics.hit_rate:.1f}%")


# ============================================================================
# EXAMPLE 6: End-to-End Sustainability Workflow
# ============================================================================

async def example_complete_workflow():
    """
    Complete end-to-end sustainability data workflow.
    """
    from greenlang.agents.templates import IntakeAgent, CalculatorAgent, ReportingAgent
    from greenlang.agents.templates import DataFormat, ReportFormat
    from greenlang.provenance import ProvenanceTracker
    from greenlang.telemetry import get_logger

    print("\n" + "=" * 60)
    print("EXAMPLE 6: End-to-End Workflow")
    print("=" * 60)

    logger = get_logger(__name__)
    tracker = ProvenanceTracker(name="sustainability_workflow")

    # Step 1: Intake
    with tracker.track_operation("data_intake"):
        logger.info("Starting data intake")

        intake_agent = IntakeAgent()
        raw_data = pd.DataFrame({
            "facility": ["Plant A", "Plant B"],
            "activity": [1000, 1500],
            "fuel_type": ["natural_gas", "diesel"]
        })

        intake_result = await intake_agent.ingest(data=raw_data, format=DataFormat.CSV)
        print(f"✓ Step 1: Ingested {intake_result.rows_read} rows")

        tracker.track_data_transformation(
            source="raw_input",
            destination="validated_data",
            transformation="intake_validation",
            input_records=0,
            output_records=intake_result.rows_read
        )

    # Step 2: Calculate
    with tracker.track_operation("emissions_calculation"):
        logger.info("Calculating emissions")

        calc_agent = CalculatorAgent()

        def calculate_emissions(activity: float, factor: float) -> float:
            return activity * factor

        calc_agent.register_formula("emissions", calculate_emissions)

        # Emission factors
        factors = {"natural_gas": 2.0, "diesel": 2.7}

        results = []
        for _, row in intake_result.data.iterrows():
            result = await calc_agent.calculate(
                "emissions",
                {
                    "activity": row["activity"],
                    "factor": factors[row["fuel_type"]]
                }
            )
            results.append(result.value)

        intake_result.data["emissions"] = results
        print(f"✓ Step 2: Calculated emissions for {len(results)} facilities")

    # Step 3: Report
    with tracker.track_operation("report_generation"):
        logger.info("Generating report")

        report_agent = ReportingAgent()
        report_result = await report_agent.generate_report(
            data=intake_result.data,
            format=ReportFormat.JSON
        )

        print(f"✓ Step 3: Generated report")

    # Final summary
    print(f"\n✓ Workflow completed successfully")
    print(f"  Operations: {len(tracker.chain_of_custody)}")
    print(f"  Data lineage: {len(tracker.context.data_lineage)} transformations")

    # Show final data
    print(f"\nFinal Results:")
    print(intake_result.data.to_string())


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GREENLANG INFRASTRUCTURE EXAMPLES")
    print("=" * 60)

    try:
        await example_data_intake_pipeline()
        await example_emissions_calculation()
        await example_report_generation()
        await example_validation_pipeline()
        await example_cached_pipeline()
        await example_complete_workflow()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
