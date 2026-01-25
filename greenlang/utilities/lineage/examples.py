"""
Column Lineage Tracking Examples

This module demonstrates various usage patterns for the column-level
lineage tracking system in GreenLang applications.
"""

import pandas as pd
from datetime import datetime

from greenlang.lineage import (
    ColumnLineageTracker,
    TransformationType,
    DataClassification,
    track_lineage
)


def example_basic_emissions_lineage():
    """Example: Track lineage for emissions calculation pipeline."""
    print("\n=== Basic Emissions Calculation Lineage ===\n")

    # Initialize tracker
    tracker = ColumnLineageTracker(storage_backend="memory")

    # 1. Register source columns
    print("1. Registering source columns...")

    fuel_consumption = tracker.add_column(
        system="erp_sap",
        dataset="fuel_purchases",
        column_name="consumption_liters",
        data_type="decimal(10,2)",
        description="Monthly fuel consumption in liters",
        business_name="Fuel Consumption",
        owner="operations@company.com"
    )

    emission_factor = tracker.add_column(
        system="reference_data",
        dataset="emission_factors",
        column_name="co2_per_liter",
        data_type="decimal(10,4)",
        description="CO2 emissions per liter of fuel",
        classification=DataClassification.PUBLIC
    )

    # 2. Register calculation result
    print("2. Registering destination column...")

    co2_emissions = tracker.add_column(
        system="sustainability",
        dataset="ghg_emissions",
        column_name="scope1_co2",
        data_type="decimal(12,2)",
        description="Scope 1 CO2 emissions in kg",
        classification=DataClassification.REGULATORY,
        sox_critical=True,
        business_name="Scope 1 CO2 Emissions"
    )

    # 3. Track the transformation
    print("3. Tracking transformation...")

    with tracker.track_transformation(
        agent_name="scope1_calculator",
        transformation_type=TransformationType.CALCULATE,
        pipeline_id="PIPE-2024-001"
    ) as transform:
        transform.add_source(fuel_consumption.id)
        transform.add_source(emission_factor.id)
        transform.add_destination(co2_emissions.id)
        transform.set_formula("consumption_liters * co2_per_liter")
        transform.add_validation_rule("co2_emissions >= 0")
        transform.add_validation_rule("consumption_liters >= 0")

    # 4. Query lineage
    print("\n4. Querying lineage information:")

    upstream = tracker.graph.get_upstream(co2_emissions.id)
    print(f"   Upstream columns: {upstream}")

    impact = tracker.graph.impact_analysis(fuel_consumption.id)
    print(f"   Impact of changing fuel_consumption: {impact['total_impacted']} columns affected")

    # 5. Generate visualization
    mermaid_diagram = tracker.visualize_lineage(output_format="mermaid")
    print("\n5. Mermaid Diagram:")
    print(mermaid_diagram[:200] + "...")


def example_gdpr_compliance_tracking():
    """Example: Track PII data flow for GDPR compliance."""
    print("\n=== GDPR Compliance Lineage Tracking ===\n")

    tracker = ColumnLineageTracker(storage_backend="memory")

    # 1. Register PII columns
    print("1. Registering PII columns...")

    customer_email = tracker.add_column(
        system="crm",
        dataset="customers",
        column_name="email_address",
        data_type="varchar(255)",
        classification=DataClassification.PII,
        is_pii=True,
        gdpr_lawful_basis="consent",
        retention_days=365,
        description="Customer email address"
    )

    customer_name = tracker.add_column(
        system="crm",
        dataset="customers",
        column_name="full_name",
        data_type="varchar(255)",
        classification=DataClassification.PII,
        is_pii=True,
        gdpr_lawful_basis="contract",
        retention_days=2555,  # 7 years for financial records
        description="Customer full name"
    )

    # 2. Track anonymization
    print("2. Tracking anonymization transformation...")

    anonymous_id = tracker.add_column(
        system="analytics",
        dataset="user_metrics",
        column_name="user_hash",
        data_type="varchar(64)",
        classification=DataClassification.INTERNAL,
        is_pii=False,
        description="Anonymized user identifier"
    )

    with tracker.track_transformation(
        agent_name="anonymization_agent",
        transformation_type=TransformationType.DERIVE
    ) as t:
        t.add_source(customer_email.id)
        t.add_destination(anonymous_id.id)
        t.set_formula("SHA256(LOWER(email_address))")

    # 3. Generate GDPR compliance report
    print("\n3. Generating GDPR compliance report...")

    report = tracker.generate_compliance_report("gdpr")
    print(f"   PII columns found: {report['gdpr']['pii_columns_count']}")
    print(f"   Retention policies defined: {report['gdpr']['retention_policies_defined']}")

    for pii_col in report['gdpr']['pii_columns']:
        print(f"   - {pii_col['column']}: {pii_col['lawful_basis']} "
              f"(retention: {pii_col['retention_days']} days)")


def example_complex_supply_chain_lineage():
    """Example: Track complex supply chain data transformations."""
    print("\n=== Complex Supply Chain Lineage ===\n")

    tracker = ColumnLineageTracker(storage_backend="memory")

    # Stage 1: Raw supplier data
    print("Stage 1: Registering raw supplier data...")

    supplier_emissions = tracker.add_column(
        system="supplier_portal",
        dataset="supplier_reports",
        column_name="reported_emissions",
        data_type="decimal",
        classification=DataClassification.CONFIDENTIAL
    )

    supplier_production = tracker.add_column(
        system="supplier_portal",
        dataset="supplier_reports",
        column_name="units_produced",
        data_type="integer",
        classification=DataClassification.CONFIDENTIAL
    )

    # Stage 2: Validated data
    print("Stage 2: Data validation...")

    validated_emissions = tracker.add_column(
        system="data_quality",
        dataset="validated_supplier_data",
        column_name="emissions_validated",
        data_type="decimal"
    )

    with tracker.track_transformation("validation_agent", TransformationType.CLEANSE) as t:
        t.add_source(supplier_emissions.id)
        t.add_destination(validated_emissions.id)
        t.add_validation_rule("emissions > 0")
        t.add_validation_rule("emissions < 1000000")

    # Stage 3: Calculate intensity
    print("Stage 3: Calculating emission intensity...")

    emission_intensity = tracker.add_column(
        system="analytics",
        dataset="supplier_metrics",
        column_name="emission_intensity",
        data_type="decimal",
        description="Emissions per unit produced"
    )

    with tracker.track_transformation("intensity_calculator", TransformationType.CALCULATE) as t:
        t.add_source(validated_emissions.id)
        t.add_source(supplier_production.id)
        t.add_destination(emission_intensity.id)
        t.set_formula("emissions_validated / NULLIF(units_produced, 0)")

    # Stage 4: Aggregate for reporting
    print("Stage 4: Aggregating for Scope 3 reporting...")

    scope3_total = tracker.add_column(
        system="ghg_reporting",
        dataset="scope3_emissions",
        column_name="supplier_emissions_total",
        data_type="decimal",
        classification=DataClassification.REGULATORY,
        sox_critical=True
    )

    with tracker.track_transformation("scope3_aggregator", TransformationType.AGGREGATE) as t:
        t.add_source(emission_intensity.id)
        t.add_destination(scope3_total.id)
        t.set_formula("SUM(emission_intensity * purchase_quantity)")

    # Analyze full lineage
    print("\nLineage Analysis:")

    # Find all paths from raw to final
    paths = tracker.graph.find_transformation_path(
        supplier_emissions.id,
        scope3_total.id
    )
    print(f"   Transformation paths found: {len(paths)}")
    for i, path in enumerate(paths, 1):
        print(f"   Path {i}: {' -> '.join(p.split('.')[-1] for p in path)}")

    # Impact analysis
    impact = tracker.graph.impact_analysis(supplier_emissions.id)
    print(f"\nImpact of supplier data change:")
    print(f"   Total affected columns: {impact['total_impacted']}")
    print(f"   Critical impacts: {len(impact['critical_impacts'])}")


def example_pandas_integration():
    """Example: Automatic tracking with pandas operations."""
    print("\n=== Pandas DataFrame Integration ===\n")

    tracker = ColumnLineageTracker(storage_backend="memory")

    # Create sample dataframes
    print("1. Creating sample data...")

    purchases_df = pd.DataFrame({
        'supplier_id': ['SUP001', 'SUP002', 'SUP003'],
        'material': ['Steel', 'Aluminum', 'Plastic'],
        'quantity_kg': [1000, 500, 200],
        'price_per_kg': [2.5, 4.0, 1.5]
    })

    emission_factors_df = pd.DataFrame({
        'material': ['Steel', 'Aluminum', 'Plastic'],
        'co2_per_kg': [2.1, 8.5, 3.2]
    })

    print("2. Performing tracked transformation...")

    # Track the merge operation
    with tracker.track_transformation(
        agent_name="material_emissions_calculator",
        transformation_type=TransformationType.JOIN
    ) as t:
        # Register source columns
        for col in purchases_df.columns:
            t.add_source(f"purchases.raw.{col}")
        for col in emission_factors_df.columns:
            t.add_source(f"factors.reference.{col}")

        # Perform merge
        result_df = pd.merge(purchases_df, emission_factors_df, on='material')

        # Calculate emissions
        result_df['emissions_kg'] = result_df['quantity_kg'] * result_df['co2_per_kg']
        result_df['cost_total'] = result_df['quantity_kg'] * result_df['price_per_kg']

        # Register output columns
        for col in result_df.columns:
            t.add_destination(f"emissions.calculated.{col}")

        t.set_formula("JOIN on material, CALCULATE emissions")

    print("\n3. Results:")
    print(result_df[['supplier_id', 'material', 'emissions_kg']].to_string())

    print("\n4. Lineage tracked:")
    print(f"   Transformations recorded: {len(tracker.graph.transformations)}")


def example_sql_lineage_extraction():
    """Example: Extract lineage from SQL queries."""
    print("\n=== SQL Query Lineage Extraction ===\n")

    tracker = ColumnLineageTracker(storage_backend="memory")

    # Complex SQL query
    sql_query = """
    WITH monthly_fuel AS (
        SELECT
            vehicle_id,
            EXTRACT(MONTH FROM purchase_date) as month,
            SUM(liters) as total_liters,
            AVG(price_per_liter) as avg_price
        FROM fuel_purchases
        WHERE purchase_date >= '2024-01-01'
        GROUP BY vehicle_id, EXTRACT(MONTH FROM purchase_date)
    ),
    vehicle_emissions AS (
        SELECT
            mf.vehicle_id,
            mf.month,
            mf.total_liters,
            mf.total_liters * ef.co2_factor as co2_emissions,
            v.department
        FROM monthly_fuel mf
        JOIN vehicles v ON mf.vehicle_id = v.id
        JOIN emission_factors ef ON v.fuel_type = ef.fuel_type
    )
    SELECT
        department,
        month,
        SUM(co2_emissions) as total_emissions,
        COUNT(DISTINCT vehicle_id) as vehicle_count,
        AVG(co2_emissions) as avg_emissions_per_vehicle
    FROM vehicle_emissions
    GROUP BY department, month
    ORDER BY department, month
    """

    print("1. Parsing SQL query for lineage...")
    tracker.parse_sql_lineage(sql_query, default_system="warehouse")

    print("2. Extracted lineage information:")
    print(f"   Transformations found: {len(tracker.graph.transformations)}")

    # The parser would identify:
    # - Source tables: fuel_purchases, vehicles, emission_factors
    # - Intermediate CTEs: monthly_fuel, vehicle_emissions
    # - Final output columns: department, month, total_emissions, etc.


def example_visualization_export():
    """Example: Export lineage visualizations."""
    print("\n=== Lineage Visualization Export ===\n")

    # Create a sample lineage graph
    tracker = ColumnLineageTracker(storage_backend="memory")

    # Build a simple but interesting graph
    raw = tracker.add_column("source", "raw_data", "value", "float")
    validated = tracker.add_column("staging", "clean_data", "value", "float")
    calculated = tracker.add_column("analytics", "metrics", "score", "float", sox_critical=True)
    reported = tracker.add_column("reporting", "dashboard", "kpi", "float",
                                 classification=DataClassification.REGULATORY)

    # Create transformations
    with tracker.track_transformation("cleanse") as t:
        t.add_source(raw.id)
        t.add_destination(validated.id)

    with tracker.track_transformation("calculate") as t:
        t.add_source(validated.id)
        t.add_destination(calculated.id)

    with tracker.track_transformation("aggregate") as t:
        t.add_source(calculated.id)
        t.add_destination(reported.id)

    # Export to different formats
    print("1. Mermaid diagram format:")
    mermaid = tracker.visualize_lineage(output_format="mermaid")
    print(mermaid)

    print("\n2. HTML visualization created")
    html = tracker.visualize_lineage(output_format="html", output_file="lineage_graph.html")

    print("\n3. Export summary:")
    print("   - Mermaid: For documentation (Markdown)")
    print("   - HTML: Interactive visualization")
    print("   - Graphviz: For high-quality diagrams")
    print("   - File saved: lineage_graph.html")


def main():
    """Run all examples."""
    print("=" * 60)
    print("COLUMN LINEAGE TRACKING EXAMPLES")
    print("=" * 60)

    example_basic_emissions_lineage()
    example_gdpr_compliance_tracking()
    example_complex_supply_chain_lineage()
    example_pandas_integration()
    example_sql_lineage_extraction()
    example_visualization_export()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()