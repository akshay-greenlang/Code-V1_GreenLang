# GL-FOUND-X-002: Python SDK Usage Guide

## Overview

The GreenLang Python SDK provides a native Python interface to the Schema Compiler & Validator (GL-FOUND-X-002). It offers full access to validation, type coercion, unit checking, and schema management capabilities.

---

## Installation

```bash
# Install the full GreenLang SDK
pip install greenlang-sdk

# Or install with optional dependencies
pip install greenlang-sdk[jsonschema]  # Full JSON Schema validation
pip install greenlang-sdk[all]         # All optional features
```

**Requirements:**
- Python 3.9+
- pydantic >= 2.0

---

## Quick Start

```python
from greenlang.agents.foundation import SchemaCompilerAgent

# Initialize the agent
agent = SchemaCompilerAgent()

# Validate a payload
result = agent.run({
    "payload": {
        "emissions": [
            {"fuel_type": "Natural Gas", "co2e_emissions_kg": 5300.0}
        ]
    },
    "schema_id": "gl-emissions-input"
})

if result.data["is_valid"]:
    print("Validation passed!")
else:
    for error in result.data["validation_result"]["errors"]:
        print(f"Error: {error['message']}")
```

---

## Core Classes

### SchemaCompilerAgent

The main agent class for schema validation operations.

```python
from greenlang.agents.foundation import SchemaCompilerAgent
from greenlang.agents.base import AgentConfig

# Default initialization
agent = SchemaCompilerAgent()

# Custom configuration
config = AgentConfig(
    name="CustomSchemaValidator",
    description="Schema validator with strict mode",
    version="1.0.0",
    parameters={
        "enable_coercion": True,
        "enable_unit_check": True,
        "strict_mode": True,
        "generate_fixes": True
    }
)
agent = SchemaCompilerAgent(config)
```

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_coercion` | bool | True | Enable automatic type coercion |
| `enable_unit_check` | bool | True | Enable unit consistency checking |
| `strict_mode` | bool | False | Treat warnings as errors |
| `generate_fixes` | bool | True | Generate fix suggestions |

---

## Validation Methods

### Basic Validation (run)

The `run()` method is the primary interface for validation.

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Validate with schema ID
result = agent.run({
    "payload": {"name": "Test", "value": 42},
    "schema_id": "my-schema"
})

# Validate with inline schema
result = agent.run({
    "payload": {"name": "Test", "value": 42},
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number", "minimum": 0}
        },
        "required": ["name", "value"]
    }
})

# Access results
print(f"Success: {result.success}")
print(f"Valid: {result.data['is_valid']}")
print(f"Errors: {result.data['validation_result']['errors']}")
print(f"Provenance: {result.data['provenance_hash']}")
```

**Input Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `payload` | dict | Yes | Data payload to validate |
| `schema_id` | str | Conditional | Schema ID from registry |
| `schema` | dict | Conditional | Inline schema definition |
| `inline_schema` | dict | Conditional | Alias for `schema` |
| `enable_coercion` | bool | No | Override coercion setting |
| `enable_unit_check` | bool | No | Override unit check setting |
| `strict_mode` | bool | No | Override strict mode |
| `generate_fixes` | bool | No | Override fix generation |
| `unit_fields` | dict | No | Map of field paths to expected unit families |

### Convenience Method (validate)

The `validate()` method provides a simpler interface with typed output.

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Validate with typed output
output = agent.validate(
    payload={"name": "Test", "value": 42},
    schema_id="my-schema"
)

# SchemaCompilerOutput provides typed access
print(f"Valid: {output.is_valid}")
print(f"Errors: {len(output.validation_result.errors)}")
print(f"Coercions: {output.coercion_records}")
print(f"Fix suggestions: {output.fix_suggestions}")
```

---

## Working with Schemas

### Schema Registry

The agent includes a built-in schema registry for managing schemas.

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Register a new schema
entry = agent.register_schema(
    schema_id="my-custom-schema",
    schema_name="My Custom Schema",
    schema_content={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "emissions_total": {"type": "number", "minimum": 0},
            "unit": {"type": "string", "enum": ["kgCO2e", "tCO2e"]}
        },
        "required": ["project_id", "emissions_total"]
    },
    version="1.0.0",
    description="Schema for custom project data",
    tags=["custom", "project", "emissions"]
)

print(f"Registered: {entry.schema_id} v{entry.schema_version}")
print(f"Hash: {entry.content_hash}")

# Retrieve a schema
schema = agent.get_schema("my-custom-schema")
if schema:
    print(f"Schema: {schema.schema_name}")
    print(f"Content: {schema.schema_content}")

# List all schemas
all_schemas = agent.list_schemas()
for s in all_schemas:
    print(f"- {s.schema_id} ({s.schema_version})")

# List schemas by tag
emissions_schemas = agent.list_schemas(tags=["emissions"])
```

### Built-in Schemas

The registry comes pre-configured with GreenLang standard schemas:

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Access built-in schemas
emissions_schema = agent.get_schema("gl-emissions-input")
activity_schema = agent.get_schema("gl-activity-data")
result_schema = agent.get_schema("gl-calculation-result")

# Validate against built-in schema
result = agent.run({
    "payload": {
        "emissions": [
            {
                "fuel_type": "Natural Gas",
                "quantity": 1000,
                "unit": "therms",
                "co2e_emissions_kg": 5300.0,
                "scope": 1
            }
        ]
    },
    "schema_id": "gl-emissions-input"
})
```

---

## Type Coercion

The SDK includes a powerful type coercion engine that safely converts values.

### Automatic Coercion

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Payload with string values that can be coerced
payload = {
    "name": "Test",
    "count": "42",      # String that should be integer
    "value": "3.14",    # String that should be float
    "active": "true"    # String that should be boolean
}

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "value": {"type": "number"},
        "active": {"type": "boolean"}
    }
}

result = agent.run({
    "payload": payload,
    "schema": schema,
    "enable_coercion": True
})

# Access coerced payload
coerced = result.data["coerced_payload"]
print(f"count: {coerced['count']} (type: {type(coerced['count']).__name__})")
# Output: count: 42 (type: int)

# Access coercion records
for record in result.data["coercion_records"]:
    print(f"Field '{record['field']}': {record['original_value']} -> {record['coerced_value']}")
    print(f"  Type: {record['original_type']} -> {record['coerced_type']}")
```

### Using TypeCoercionEngine Directly

```python
from greenlang.agents.foundation.schema_compiler import TypeCoercionEngine, CoercionType

engine = TypeCoercionEngine()

# Coerce a single value
value, success, record = engine.coerce("42", "integer", "my_field")
print(f"Coerced value: {value}, Success: {success}")
# Output: Coerced value: 42, Success: True

# Check coercion type
if record:
    print(f"Coercion type: {record.coercion_type}")
    # Output: Coercion type: CoercionType.STRING_TO_INT

# Supported coercions
# String -> Integer: "42" -> 42
# String -> Float: "3.14" -> 3.14
# String -> Boolean: "true"/"yes"/"1" -> True
# Int -> Float: 42 -> 42.0
# Float -> Int: 42.0 -> 42 (if whole number)
# Any -> Array: value -> [value]
```

---

## Unit Consistency Checking

Validate that units across related fields are consistent.

### Using UnitConsistencyChecker

```python
from greenlang.agents.foundation.schema_compiler import UnitConsistencyChecker

checker = UnitConsistencyChecker()

# Get unit information
info = checker.get_unit_info("kgCO2e")
print(f"Unit: {info.unit}")
print(f"Family: {info.family}")        # mass_co2e
print(f"Base unit: {info.base_unit}")  # kgCO2e
print(f"Conversion: {info.conversion_factor}")

# Check consistency of multiple units
units = [
    ("emissions.scope1", "kgCO2e"),
    ("emissions.scope2", "tCO2e"),
    ("emissions.scope3", "kgCO2e")
]

result = checker.check_consistency(units)
print(f"Consistent: {result.valid}")
# Output: Consistent: True (all same family)

# Check with mixed families (produces warning)
mixed_units = [
    ("emissions.co2", "kgCO2e"),
    ("energy.consumed", "kWh")  # Different family
]

result = checker.check_consistency(mixed_units)
print(f"Warnings: {result.warnings}")

# Get conversion suggestion
conversion = checker.suggest_conversion("tCO2e", "kgCO2e")
print(f"Factor: {conversion['conversion_factor']}")  # 1000.0
print(f"Formula: {conversion['formula']}")  # tCO2e * 1000.0 = kgCO2e
```

### Unit Validation in Agent

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Get unit info through agent
info = agent.get_unit_info("MWh")
print(f"Family: {info.family}")  # energy

# Get conversion suggestion through agent
suggestion = agent.suggest_unit_conversion("MWh", "kWh")
print(f"Factor: {suggestion['conversion_factor']}")  # 1000.0

# Validate with unit field specification
result = agent.run({
    "payload": {
        "energy_data": {
            "electricity": {"value": 1000, "unit": "kWh"},
            "gas": {"value": 500, "unit": "therms"}  # Different family
        }
    },
    "schema": {"type": "object"},
    "enable_unit_check": True,
    "unit_fields": {
        "energy_data.electricity.unit": "energy",
        "energy_data.gas.unit": "energy"
    }
})

# Check for unit validation results
for validation in result.data["unit_validations"]:
    print(f"Field: {validation['field']}")
    print(f"Unit: {validation['unit']}")
    print(f"Valid: {validation['valid']}")
    print(f"Family: {validation['family']}")
```

---

## Fix Suggestions

The agent generates machine-fixable suggestions for validation errors.

```python
from greenlang.agents.foundation import SchemaCompilerAgent, FixSuggestionType

agent = SchemaCompilerAgent()

# Payload with errors
payload = {
    "name": 123,              # Should be string
    "count": "invalid",       # Should be number
    "status": "activ",        # Should be "active" (typo)
    "priority": 150           # Max is 100
}

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
        "priority": {"type": "integer", "maximum": 100}
    },
    "required": ["name"]
}

result = agent.run({
    "payload": payload,
    "schema": schema,
    "generate_fixes": True,
    "enable_coercion": False  # Disable to see suggestions
})

# Access fix suggestions
for fix in result.data["fix_suggestions"]:
    print(f"\nField: {fix['field']}")
    print(f"Type: {fix['suggestion_type']}")
    print(f"Description: {fix['description']}")
    print(f"Original: {fix['original_value']}")
    print(f"Suggested: {fix['suggested_value']}")
    print(f"Auto-fixable: {fix['auto_fixable']}")
    print(f"Confidence: {fix['confidence']}")
    if fix.get('code_snippet'):
        print(f"Code: {fix['code_snippet']}")
```

**Fix Suggestion Types:**

| Type | Description |
|------|-------------|
| `TYPE_COERCION` | Convert value to correct type |
| `VALUE_RANGE` | Adjust value to be within min/max |
| `PATTERN_MATCH` | Fix value to match pattern |
| `REQUIRED_FIELD` | Add missing required field |
| `UNIT_CONVERSION` | Convert to correct unit |
| `FORMAT_CORRECTION` | Fix format (date, email, etc.) |
| `ENUM_SUGGESTION` | Suggest closest enum value |

---

## Error Handling

### Agent Result Structure

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

result = agent.run({
    "payload": {"name": "Test"},
    "schema_id": "gl-emissions-input"
})

# Check if agent execution succeeded
if not result.success:
    print(f"Agent error: {result.error}")
    print(f"Error type: {result.data.get('exception_type')}")
else:
    # Agent succeeded - check validation result
    if result.data["is_valid"]:
        print("Validation passed!")
    else:
        # Access validation errors
        for error in result.data["validation_result"]["errors"]:
            print(f"Error at {error['field']}: {error['message']}")
            print(f"  Severity: {error['severity']}")
            print(f"  Validator: {error['validator']}")
```

### Exception Handling

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

try:
    result = agent.run({
        "payload": {"test": "data"},
        "schema_id": "nonexistent-schema"
    })

    if not result.success:
        # Handle schema not found
        if "not found" in result.error.lower():
            print(f"Schema not found: {result.error}")
        else:
            print(f"Validation error: {result.error}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Advanced Usage

### Custom Validation Pipeline

```python
from greenlang.agents.foundation import SchemaCompilerAgent
from greenlang.agents.foundation.schema_compiler import (
    SchemaRegistry,
    TypeCoercionEngine,
    UnitConsistencyChecker
)

# Create custom components
registry = SchemaRegistry()
coercion = TypeCoercionEngine()
units = UnitConsistencyChecker()

# Register custom schemas
registry.register(
    schema_id="step1-schema",
    schema_name="Step 1 Validation",
    schema_content={"type": "object", "properties": {"stage": {"type": "string"}}},
    tags=["pipeline"]
)

registry.register(
    schema_id="step2-schema",
    schema_name="Step 2 Validation",
    schema_content={"type": "object", "properties": {"result": {"type": "number"}}},
    tags=["pipeline"]
)

# Create agent with custom registry access
agent = SchemaCompilerAgent()

# Multi-step validation
def validate_pipeline(data):
    # Step 1: Validate input structure
    result1 = agent.run({
        "payload": data,
        "schema_id": "step1-schema"
    })
    if not result1.data["is_valid"]:
        return {"step": 1, "errors": result1.data["validation_result"]["errors"]}

    # Step 2: Validate business rules
    result2 = agent.run({
        "payload": data,
        "schema_id": "step2-schema"
    })
    if not result2.data["is_valid"]:
        return {"step": 2, "errors": result2.data["validation_result"]["errors"]}

    return {"success": True, "provenance": result2.data["provenance_hash"]}

# Run pipeline
result = validate_pipeline({"stage": "complete", "result": 42.5})
print(result)
```

### Batch Validation

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Validate multiple payloads
payloads = [
    {"name": "Item 1", "value": 100},
    {"name": "Item 2", "value": "invalid"},  # Error
    {"name": "Item 3", "value": 300},
    {"name": 123, "value": 400},  # Error
]

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"}
    },
    "required": ["name", "value"]
}

# Validate each payload
results = []
for i, payload in enumerate(payloads):
    result = agent.run({
        "payload": payload,
        "schema": schema
    })
    results.append({
        "index": i,
        "valid": result.data["is_valid"],
        "errors": len(result.data["validation_result"]["errors"])
    })

# Summary
valid_count = sum(1 for r in results if r["valid"])
print(f"Valid: {valid_count}/{len(payloads)}")
for r in results:
    status = "PASS" if r["valid"] else f"FAIL ({r['errors']} errors)"
    print(f"  [{r['index']}] {status}")
```

### Async Usage

```python
import asyncio
from greenlang.agents.foundation import SchemaCompilerAgent

async def validate_async(payload, schema_id):
    """Run validation in async context."""
    agent = SchemaCompilerAgent()

    # Run in thread pool for async compatibility
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: agent.run({
            "payload": payload,
            "schema_id": schema_id
        })
    )
    return result

# Usage
async def main():
    tasks = [
        validate_async({"name": "A"}, "my-schema"),
        validate_async({"name": "B"}, "my-schema"),
        validate_async({"name": "C"}, "my-schema"),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"Valid: {r.data['is_valid']}")

asyncio.run(main())
```

---

## Data Models

### SchemaCompilerOutput

```python
from greenlang.agents.foundation.schema_compiler import SchemaCompilerOutput

# Fields available in output
class SchemaCompilerOutput:
    is_valid: bool                          # Overall validation result
    validation_result: ValidationResult     # Detailed validation result
    coerced_payload: Optional[Dict]         # Payload after coercion
    coercion_records: List[CoercionRecord]  # Coercion audit trail
    fix_suggestions: List[FixSuggestion]    # Machine-fixable suggestions
    unit_validations: List[Dict]            # Unit check results
    provenance_hash: str                    # SHA-256 for audit trail
    processing_time_ms: float               # Processing duration
    schema_used: str                        # Schema ID or "inline"
```

### ValidationResult

```python
from greenlang.governance.validation.framework import ValidationResult, ValidationError

# Access validation details
result = agent.run({"payload": data, "schema": schema})
validation = result.data["validation_result"]

print(f"Valid: {validation['valid']}")
print(f"Errors: {len(validation['errors'])}")
print(f"Warnings: {len(validation['warnings'])}")

for error in validation["errors"]:
    print(f"Field: {error['field']}")
    print(f"Message: {error['message']}")
    print(f"Severity: {error['severity']}")
    print(f"Validator: {error['validator']}")
```

---

## Best Practices

### 1. Reuse Agent Instances

```python
# Good: Reuse agent instance
agent = SchemaCompilerAgent()
for payload in payloads:
    result = agent.run({"payload": payload, "schema_id": "my-schema"})

# Avoid: Creating new agent each time
for payload in payloads:
    agent = SchemaCompilerAgent()  # Inefficient
    result = agent.run(...)
```

### 2. Pre-register Frequently Used Schemas

```python
# Register once at startup
agent = SchemaCompilerAgent()
agent.register_schema(
    schema_id="frequently-used",
    schema_name="Frequently Used Schema",
    schema_content=my_schema
)

# Use by ID (faster than inline)
for payload in payloads:
    result = agent.run({
        "payload": payload,
        "schema_id": "frequently-used"  # Uses cached schema
    })
```

### 3. Use Appropriate Validation Profiles

```python
# Development: Use permissive mode
agent = SchemaCompilerAgent()
result = agent.run({
    "payload": data,
    "schema": schema,
    "strict_mode": False
})

# Production: Use strict mode
result = agent.run({
    "payload": data,
    "schema": schema,
    "strict_mode": True  # Warnings become errors
})
```

### 4. Handle Coerced Payloads

```python
# Always use coerced payload for downstream processing
result = agent.run({
    "payload": raw_data,
    "schema": schema,
    "enable_coercion": True
})

if result.data["is_valid"]:
    # Use coerced payload (types are correct)
    processed_data = result.data["coerced_payload"]
    save_to_database(processed_data)
```

---

## See Also

- [CLI Reference](cli.md)
- [REST API Reference](api.md)
- [Error Codes Reference](error_codes.md)
- [Migration Guide](migration.md)
