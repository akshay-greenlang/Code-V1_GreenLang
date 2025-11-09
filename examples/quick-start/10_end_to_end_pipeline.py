"""
Example 10: End-to-End Pipeline
================================

Complete pipeline demonstrating multiple infrastructure components working together.
"""

import asyncio
import pandas as pd
from greenlang.agents.templates import IntakeAgent, CalculatorAgent, ReportingAgent
from greenlang.agents.templates import DataFormat, ReportFormat
from greenlang.provenance import ProvenanceTracker
from greenlang.telemetry import get_logger
from greenlang.cache import initialize_cache_manager, get_cache_manager


async def main():
    """Run complete end-to-end pipeline."""
    # Initialize infrastructure
    logger = get_logger(__name__)
    tracker = ProvenanceTracker(name="end_to_end_pipeline")

    initialize_cache_manager(enable_l1=True)
    cache = get_cache_manager()

    print("\n" + "="*60)
    print("END-TO-END EMISSIONS PROCESSING PIPELINE")
    print("="*60)

    # Step 1: Data Intake
    print("\n[1/3] Data Intake")
    with tracker.track_operation("intake"):
        intake_agent = IntakeAgent()

        # Sample emissions data
        raw_data = pd.DataFrame({
            "facility": ["Plant A", "Plant B", "Plant C"],
            "fuel_consumption": [1000, 1500, 800],
            "fuel_type": ["natural_gas", "diesel", "natural_gas"],
            "electricity_kwh": [50000, 75000, 40000]
        })

        intake_result = await intake_agent.ingest(
            data=raw_data,
            format=DataFormat.CSV,
            validate=True
        )

        logger.info(f"Ingested {intake_result.rows_read} rows")
        print(f"  ✓ Ingested {intake_result.rows_read} rows")

    # Step 2: Calculate Emissions
    print("\n[2/3] Calculate Emissions")
    with tracker.track_operation("calculate"):
        calc_agent = CalculatorAgent()

        # Register formulas
        def fuel_emissions(fuel_amount: float, emission_factor: float) -> float:
            return fuel_amount * emission_factor

        def electricity_emissions(kwh: float, grid_factor: float) -> float:
            return kwh * grid_factor

        calc_agent.register_formula("fuel_emissions", fuel_emissions)
        calc_agent.register_formula("electricity_emissions", electricity_emissions)

        # Emission factors
        fuel_factors = {
            "natural_gas": 2.0,
            "diesel": 2.7
        }
        grid_factor = 0.5

        # Calculate for each facility
        results = []
        for _, row in intake_result.data.iterrows():
            # Fuel emissions
            fuel_result = await calc_agent.calculate(
                "fuel_emissions",
                {
                    "fuel_amount": row["fuel_consumption"],
                    "emission_factor": fuel_factors[row["fuel_type"]]
                }
            )

            # Electricity emissions
            elec_result = await calc_agent.calculate(
                "electricity_emissions",
                {
                    "kwh": row["electricity_kwh"],
                    "grid_factor": grid_factor
                }
            )

            total = fuel_result.value + elec_result.value

            results.append({
                "facility": row["facility"],
                "fuel_emissions": fuel_result.value,
                "electricity_emissions": elec_result.value,
                "total_emissions": total
            })

            logger.info(f"Calculated emissions for {row['facility']}: {total:.2f} kg CO2e")

        results_df = pd.DataFrame(results)
        print(f"  ✓ Calculated emissions for {len(results)} facilities")

    # Step 3: Generate Report
    print("\n[3/3] Generate Report")
    with tracker.track_operation("report"):
        report_agent = ReportingAgent()

        report_result = await report_agent.generate_report(
            data=results_df,
            format=ReportFormat.MARKDOWN
        )

        logger.info("Generated emissions report")
        print(f"  ✓ Generated report in {ReportFormat.MARKDOWN.value} format")

    # Show results
    print("\n" + "="*60)
    print("EMISSIONS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))

    print("\n" + "="*60)
    print("PIPELINE STATISTICS")
    print("="*60)
    print(f"  Total operations: {len(tracker.chain_of_custody)}")
    print(f"  Data transformations: {len(tracker.context.data_lineage)}")

    cache_analytics = cache.get_analytics()
    print(f"  Cache requests: {cache_analytics.total_requests}")
    print(f"  Cache hit rate: {cache_analytics.hit_rate:.1f}%")

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
