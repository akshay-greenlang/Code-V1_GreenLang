# -*- coding: utf-8 -*-
"""
Example 3: Data Validation
===========================

Demonstrates ValidationFramework with schema and rules.
"""

import asyncio
from greenlang.validation import (
    ValidationFramework,
    SchemaValidator,
    RulesEngine,
    Rule,
    RuleOperator
)


async def main():
    """Run validation example."""
    # Create validation framework
    framework = ValidationFramework()

    # Add JSON schema validator
    schema = {
        "type": "object",
        "properties": {
            "emissions": {"type": "number"},
            "facility": {"type": "string"}
        },
        "required": ["emissions", "facility"]
    }

    schema_validator = SchemaValidator(schema)
    framework.add_validator("schema", schema_validator.validate)

    # Add business rules
    rules_engine = RulesEngine()
    rules_engine.add_rule(Rule(
        name="non_negative_emissions",
        field="emissions",
        operator=RuleOperator.GREATER_EQUAL,
        value=0,
        message="Emissions must be non-negative"
    ))

    framework.add_validator("rules", rules_engine.validate)

    # Test valid data
    valid_data = {"emissions": 1500.5, "facility": "Plant A"}
    result = framework.validate(valid_data)

    print(f"Valid data test: {'PASS' if result.valid else 'FAIL'}")

    # Test invalid data
    invalid_data = {"emissions": -100, "facility": "Plant B"}
    result = framework.validate(invalid_data)

    print(f"Invalid data test: {'PASS' if not result.valid else 'FAIL'}")
    for error in result.errors:
        print(f"  Error: {error.message}")


if __name__ == "__main__":
    asyncio.run(main())
