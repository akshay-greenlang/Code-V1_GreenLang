"""
Integration Example: Agent + ValidationFramework
=================================================

Demonstrates how to integrate agents with ValidationFramework.
"""

import asyncio
import pandas as pd
from greenlang.agents.templates import IntakeAgent, DataFormat
from greenlang.validation import (
    ValidationFramework,
    SchemaValidator,
    RulesEngine,
    Rule,
    RuleOperator,
    DataQualityValidator
)


async def main():
    """Run Agent + ValidationFramework integration."""
    print("\nAgent + ValidationFramework Integration")
    print("=" * 60)

    # Create validation framework
    framework = ValidationFramework()

    # Add schema validation
    schema = {
        "type": "object",
        "properties": {
            "facility": {"type": "string", "minLength": 1},
            "emissions": {"type": "number", "minimum": 0},
            "year": {"type": "integer", "minimum": 2000, "maximum": 2100}
        },
        "required": ["facility", "emissions", "year"]
    }

    framework.add_validator(
        "schema",
        SchemaValidator(schema).validate
    )

    # Add business rules
    rules = RulesEngine()
    rules.add_rule(Rule(
        name="valid_emissions",
        field="emissions",
        operator=RuleOperator.GREATER_EQUAL,
        value=0,
        message="Emissions must be non-negative"
    ))
    rules.add_rule(Rule(
        name="valid_year",
        field="year",
        operator=RuleOperator.BETWEEN,
        value=(2000, 2100),
        message="Year must be between 2000 and 2100"
    ))

    framework.add_validator("rules", rules.validate)

    # Add data quality validation
    quality = DataQualityValidator(
        completeness_threshold=0.95,
        consistency_checks=True
    )

    framework.add_validator("quality", quality.validate)

    # Create intake agent with validation
    intake_agent = IntakeAgent(
        schema=schema,
        validation_framework=framework
    )

    # Test with valid data
    print("\n[Test 1] Valid Data:")
    valid_data = pd.DataFrame({
        "facility": ["Plant A", "Plant B", "Plant C"],
        "emissions": [1500.5, 2300.8, 1800.2],
        "year": [2024, 2024, 2024]
    })

    result = await intake_agent.ingest(
        data=valid_data,
        format=DataFormat.CSV,
        validate=True
    )

    print(f"  Success: {result.success}")
    print(f"  Rows read: {result.rows_read}")
    print(f"  Rows valid: {result.rows_valid}")
    print(f"  Validation issues: {len(result.validation_issues)}")

    # Test with invalid data
    print("\n[Test 2] Invalid Data:")
    invalid_data = pd.DataFrame({
        "facility": ["Plant D", "", "Plant F"],  # Empty facility name
        "emissions": [1500.5, -100, 1800.2],    # Negative emissions
        "year": [2024, 1999, 2024]               # Invalid year
    })

    result = await intake_agent.ingest(
        data=invalid_data,
        format=DataFormat.CSV,
        validate=True
    )

    print(f"  Success: {result.success}")
    print(f"  Rows read: {result.rows_read}")
    print(f"  Rows valid: {result.rows_valid}")
    print(f"  Validation issues: {len(result.validation_issues)}")

    if result.validation_issues:
        print("\n  Issues found:")
        for issue in result.validation_issues[:5]:  # Show first 5
            print(f"    - {issue.severity}: {issue.message}")

    print("\n" + "=" * 60)
    print("Integration demonstrates multi-layer validation:")
    print("  1. Schema validation (JSON Schema)")
    print("  2. Business rules validation")
    print("  3. Data quality validation")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
