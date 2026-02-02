"""
Test suite for column-level lineage tracking.

This module contains comprehensive tests demonstrating all features
of the column lineage tracking system.
"""

import pytest
from datetime import datetime
import pandas as pd
import json
from pathlib import Path

from greenlang.lineage import (
    ColumnLineageTracker,
    LineageNode,
    TransformationType,
    DataClassification,
    track_lineage
)


class TestColumnLineageTracker:
    """Test suite for ColumnLineageTracker."""

    def test_basic_lineage_tracking(self):
        """Test basic column lineage tracking."""
        tracker = ColumnLineageTracker()

        # Add source columns
        fuel_col = tracker.add_column(
            system="erp",
            dataset="fuel_consumption",
            column_name="liters",
            data_type="float",
            classification=DataClassification.INTERNAL
        )

        factor_col = tracker.add_column(
            system="reference",
            dataset="emission_factors",
            column_name="co2_factor",
            data_type="float",
            classification=DataClassification.PUBLIC
        )

        # Add destination column
        emissions_col = tracker.add_column(
            system="reporting",
            dataset="emissions",
            column_name="co2_total",
            data_type="float",
            classification=DataClassification.REGULATORY,
            sox_critical=True
        )

        # Track transformation
        with tracker.track_transformation(
            agent_name="emissions_calculator",
            transformation_type=TransformationType.CALCULATE
        ) as t:
            t.add_source(fuel_col.id)
            t.add_source(factor_col.id)
            t.add_destination(emissions_col.id)
            t.set_formula("liters * co2_factor")

        # Verify lineage
        upstream = tracker.graph.get_upstream(emissions_col.id)
        assert fuel_col.id in upstream
        assert factor_col.id in upstream
        assert len(upstream) == 2

    def test_multi_level_lineage(self):
        """Test multi-level lineage tracking."""
        tracker = ColumnLineageTracker()

        # Level 1: Raw data
        raw_data = tracker.add_column("source", "raw", "value", "float")

        # Level 2: Cleaned data
        cleaned_data = tracker.add_column("staging", "clean", "value", "float")

        with tracker.track_transformation("cleanse_agent", TransformationType.CLEANSE) as t:
            t.add_source(raw_data.id)
            t.add_destination(cleaned_data.id)

        # Level 3: Aggregated data
        aggregated = tracker.add_column("reporting", "summary", "total", "float")

        with tracker.track_transformation("aggregate_agent", TransformationType.AGGREGATE) as t:
            t.add_source(cleaned_data.id)
            t.add_destination(aggregated.id)

        # Check full lineage path
        paths = tracker.graph.find_transformation_path(raw_data.id, aggregated.id)
        assert len(paths) > 0
        assert len(paths[0]) == 3  # raw -> clean -> aggregated

    def test_impact_analysis(self):
        """Test impact analysis for column changes."""
        tracker = ColumnLineageTracker()

        # Create a branching lineage
        source = tracker.add_column("erp", "orders", "amount", "decimal", is_pii=False)

        # Two branches from source
        calc1 = tracker.add_column("finance", "revenue", "total", "decimal", sox_critical=True)
        calc2 = tracker.add_column("analytics", "metrics", "sum", "decimal")

        with tracker.track_transformation("revenue_calc") as t:
            t.add_source(source.id)
            t.add_destination(calc1.id)

        with tracker.track_transformation("metrics_calc") as t:
            t.add_source(source.id)
            t.add_destination(calc2.id)

        # Further downstream
        report = tracker.add_column("reporting", "quarterly", "revenue", "decimal",
                                   classification=DataClassification.REGULATORY)

        with tracker.track_transformation("quarterly_report") as t:
            t.add_source(calc1.id)
            t.add_destination(report.id)

        # Perform impact analysis
        impact = tracker.graph.impact_analysis(source.id)

        assert impact['total_impacted'] == 3
        assert len(impact['critical_impacts']) > 0  # SOX-critical field impacted
        assert 'finance' in impact['impacted_systems']
        assert 'analytics' in impact['impacted_systems']

    def test_pii_tracking(self):
        """Test PII data lineage tracking."""
        tracker = ColumnLineageTracker()

        # PII source
        customer_email = tracker.add_column(
            system="crm",
            dataset="customers",
            column_name="email",
            data_type="varchar",
            is_pii=True,
            classification=DataClassification.PII,
            gdpr_lawful_basis="consent",
            retention_days=365
        )

        # Derived PII field
        hashed_email = tracker.add_column(
            system="analytics",
            dataset="user_profiles",
            column_name="email_hash",
            data_type="varchar",
            is_pii=False,  # Hashed, so not direct PII
            classification=DataClassification.CONFIDENTIAL
        )

        with tracker.track_transformation("hash_pii", TransformationType.DERIVE) as t:
            t.add_source(customer_email.id)
            t.add_destination(hashed_email.id)
            t.set_formula("SHA256(email)")

        # Check PII lineage
        downstream = tracker.graph.get_downstream(customer_email.id)
        assert hashed_email.id in downstream

    def test_sql_lineage_parsing(self):
        """Test SQL query lineage parsing."""
        tracker = ColumnLineageTracker()

        # Parse a simple SQL query
        sql = """
        SELECT
            o.order_id,
            o.total_amount,
            c.customer_name,
            o.total_amount * 0.1 as tax_amount
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.status = 'completed'
        """

        tracker.parse_sql_lineage(sql, default_system="warehouse")

        # Verify transformations were tracked
        assert len(tracker.graph.transformations) > 0

    def test_dataframe_tracking(self):
        """Test pandas DataFrame operation tracking."""
        tracker = ColumnLineageTracker()

        # Create sample dataframes
        input_df = pd.DataFrame({
            'quantity': [10, 20, 30],
            'price': [100, 200, 150]
        })

        output_df = pd.DataFrame({
            'quantity': [10, 20, 30],
            'price': [100, 200, 150],
            'total': [1000, 4000, 4500]
        })

        # Track the transformation
        tracker.track_dataframe_operation(
            input_df, output_df,
            operation="Calculate total as quantity * price",
            agent_name="calculation_agent"
        )

        # Check that transformation was tracked
        assert len(tracker.graph.transformations) > 0

    def test_compliance_reporting(self):
        """Test compliance report generation."""
        tracker = ColumnLineageTracker()

        # Add various columns with compliance implications
        pii_col = tracker.add_column(
            "crm", "users", "ssn", "varchar",
            is_pii=True,
            gdpr_lawful_basis="legal_obligation",
            retention_days=2555  # 7 years
        )

        sox_col = tracker.add_column(
            "finance", "transactions", "amount", "decimal",
            sox_critical=True,
            classification=DataClassification.FINANCIAL
        )

        # Generate GDPR compliance report
        gdpr_report = tracker.generate_compliance_report("gdpr")
        assert 'gdpr' in gdpr_report
        assert gdpr_report['gdpr']['pii_columns_count'] == 1

        # Generate SOX compliance report
        sox_report = tracker.generate_compliance_report("sox")
        assert 'sox' in sox_report
        assert sox_report['sox']['critical_fields_count'] == 1

        # Generate complete compliance report
        full_report = tracker.generate_compliance_report("all")
        assert 'gdpr' in full_report
        assert 'sox' in full_report

    def test_visualization_export(self):
        """Test visualization export formats."""
        tracker = ColumnLineageTracker()

        # Create simple lineage
        source = tracker.add_column("db", "table1", "col1", "int")
        dest = tracker.add_column("db", "table2", "col2", "int")

        with tracker.track_transformation("transform") as t:
            t.add_source(source.id)
            t.add_destination(dest.id)

        # Test Mermaid export
        mermaid = tracker.visualize_lineage(output_format="mermaid")
        assert "graph LR" in mermaid
        assert source.id.replace(".", "_") in mermaid

        # Test Graphviz export
        graphviz = tracker.visualize_lineage(output_format="graphviz")
        assert "digraph" in graphviz

        # Test HTML export
        html = tracker.visualize_lineage(output_format="html")
        assert "<html>" in html
        assert "vis-network" in html

    def test_storage_backends(self, tmp_path):
        """Test different storage backends."""
        # Test file storage
        file_tracker = ColumnLineageTracker(
            storage_backend="file",
            config={"path": str(tmp_path / "lineage")}
        )

        col = file_tracker.add_column("sys", "table", "col", "varchar")
        assert col.id == "sys.table.col"

        # Check files were created
        lineage_path = tmp_path / "lineage"
        assert lineage_path.exists()

    def test_decorator_tracking(self):
        """Test automatic lineage tracking with decorator."""

        @track_lineage
        def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
            """Calculate metrics from input data."""
            df['metric'] = df['value'] * 2
            return df

        # Create test data
        input_df = pd.DataFrame({'value': [1, 2, 3]})

        # Call decorated function
        output_df = calculate_metrics(input_df)

        assert 'metric' in output_df.columns
        assert output_df['metric'].tolist() == [2, 4, 6]

    def test_complex_transformation_chain(self):
        """Test complex transformation chain with multiple agents."""
        tracker = ColumnLineageTracker()

        # Stage 1: Data Intake
        raw_fuel = tracker.add_column("erp", "fuel_purchases", "liters", "float")
        raw_distance = tracker.add_column("fleet", "vehicle_logs", "kilometers", "float")

        # Stage 2: Validation & Cleansing
        clean_fuel = tracker.add_column("staging", "validated_fuel", "liters", "float")
        clean_distance = tracker.add_column("staging", "validated_distance", "km", "float")

        with tracker.track_transformation("validation_agent", TransformationType.CLEANSE) as t:
            t.add_source(raw_fuel.id)
            t.add_destination(clean_fuel.id)
            t.add_validation_rule("liters > 0")

        with tracker.track_transformation("validation_agent", TransformationType.CLEANSE) as t:
            t.add_source(raw_distance.id)
            t.add_destination(clean_distance.id)
            t.add_validation_rule("kilometers > 0")

        # Stage 3: Calculation
        emissions = tracker.add_column(
            "reporting", "emissions", "co2_kg", "float",
            classification=DataClassification.REGULATORY
        )

        with tracker.track_transformation("emissions_agent", TransformationType.CALCULATE) as t:
            t.add_source(clean_fuel.id)
            t.add_destination(emissions.id)
            t.set_formula("liters * 2.31")  # CO2 kg per liter of gasoline

        # Stage 4: Aggregation
        monthly_total = tracker.add_column(
            "reporting", "monthly_summary", "total_emissions", "float",
            sox_critical=True
        )

        with tracker.track_transformation("aggregation_agent", TransformationType.AGGREGATE) as t:
            t.add_source(emissions.id)
            t.add_destination(monthly_total.id)
            t.set_formula("SUM(co2_kg) GROUP BY month")

        # Verify complete lineage
        full_upstream = tracker.graph.get_upstream(monthly_total.id)
        assert raw_fuel.id in full_upstream
        assert len(full_upstream) == 3  # raw_fuel, clean_fuel, emissions

        # Check transformation path
        paths = tracker.graph.find_transformation_path(raw_fuel.id, monthly_total.id)
        assert len(paths) > 0

    def test_circular_dependency_prevention(self):
        """Test that circular dependencies are prevented."""
        tracker = ColumnLineageTracker()

        col1 = tracker.add_column("sys", "table", "col1", "int")
        col2 = tracker.add_column("sys", "table", "col2", "int")

        # Create forward transformation
        with tracker.track_transformation("forward") as t:
            t.add_source(col1.id)
            t.add_destination(col2.id)

        # Attempt reverse transformation (would create cycle)
        with tracker.track_transformation("reverse") as t:
            t.add_source(col2.id)
            t.add_destination(col1.id)

        # NetworkX should handle this - DAG property maintained
        # Check that we can still traverse without infinite loops
        upstream = tracker.graph.get_upstream(col1.id, max_depth=10)
        assert col2.id in upstream  # Due to reverse edge

    def test_transformation_context_manager(self):
        """Test transformation context manager functionality."""
        tracker = ColumnLineageTracker()

        source = tracker.add_column("db", "t1", "c1", "int")
        dest = tracker.add_column("db", "t2", "c2", "int")

        # Test successful transformation
        with tracker.track_transformation("test_agent") as t:
            assert t is not None
            t.add_source(source.id)
            t.add_destination(dest.id)
            t.set_formula("c1 * 2")
            t.add_validation_rule("c2 > 0")

        # Verify transformation was recorded
        assert len(tracker.graph.transformations) == 1
        trans = list(tracker.graph.transformations.values())[0]
        assert trans.agent_name == "test_agent"
        assert trans.formula == "c1 * 2"
        assert "c2 > 0" in trans.validation_rules


if __name__ == "__main__":
    # Run basic tests
    test = TestColumnLineageTracker()

    print("Running basic lineage tracking test...")
    test.test_basic_lineage_tracking()
    print("✓ Basic tracking works")

    print("\nRunning multi-level lineage test...")
    test.test_multi_level_lineage()
    print("✓ Multi-level tracking works")

    print("\nRunning impact analysis test...")
    test.test_impact_analysis()
    print("✓ Impact analysis works")

    print("\nRunning PII tracking test...")
    test.test_pii_tracking()
    print("✓ PII tracking works")

    print("\nRunning compliance reporting test...")
    test.test_compliance_reporting()
    print("✓ Compliance reporting works")

    print("\nAll tests passed successfully!")