# GL-FOUND-X-002: Migration Guide

## Overview

This guide helps you migrate from existing validation solutions to the GreenLang Schema Compiler & Validator (GL-FOUND-X-002). It covers common migration scenarios, compatibility considerations, and step-by-step instructions.

---

## Migration Scenarios

### From JSON Schema Libraries

If you're currently using libraries like `jsonschema`, `fastjsonschema`, or similar, GL-FOUND-X-002 provides a compatible path with enhanced features.

#### From jsonschema (Python)

**Before:**

```python
import jsonschema
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "value": {"type": "number"}
    },
    "required": ["name", "value"]
}

try:
    validate(instance=data, schema=schema)
    print("Valid")
except ValidationError as e:
    print(f"Error: {e.message}")
```

**After:**

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

result = agent.run({
    "payload": data,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "number"}
        },
        "required": ["name", "value"]
    }
})

if result.data["is_valid"]:
    print("Valid")
else:
    for error in result.data["validation_result"]["errors"]:
        print(f"Error: {error['message']}")
```

**Benefits of migration:**
- Automatic type coercion (e.g., `"42"` -> `42`)
- Fix suggestions with code snippets
- Unit consistency checking
- Provenance tracking
- Integration with GreenLang ecosystem

#### Compatibility Notes

| JSON Schema Feature | GL-FOUND-X-002 Support |
|---------------------|------------------------|
| Draft-07 keywords | Full support |
| Draft-06/04 schemas | Supported with compatibility mode |
| `$ref` references | Supported |
| Custom formats | Extensible |
| Custom validators | Via rule system |

---

### From Pydantic Validation

If you're using Pydantic for validation, you can integrate GL-FOUND-X-002 alongside or instead of Pydantic validation.

**Before:**

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class Emission(BaseModel):
    fuel_type: str
    quantity: float = Field(ge=0)
    unit: str
    co2e_emissions_kg: float

    @validator('unit')
    def validate_unit(cls, v):
        allowed = ['kWh', 'therms', 'L', 'kg']
        if v not in allowed:
            raise ValueError(f'Unit must be one of {allowed}')
        return v

class EmissionsData(BaseModel):
    emissions: List[Emission]
    organization_id: Optional[str]

# Usage
try:
    data = EmissionsData(**raw_data)
except ValidationError as e:
    print(e.json())
```

**After:**

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Register equivalent schema
agent.register_schema(
    schema_id="emissions-data",
    schema_name="Emissions Data",
    schema_content={
        "type": "object",
        "properties": {
            "emissions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fuel_type": {"type": "string"},
                        "quantity": {"type": "number", "minimum": 0},
                        "unit": {"type": "string", "enum": ["kWh", "therms", "L", "kg"]},
                        "co2e_emissions_kg": {"type": "number"}
                    },
                    "required": ["fuel_type", "quantity", "unit", "co2e_emissions_kg"]
                }
            },
            "organization_id": {"type": "string"}
        },
        "required": ["emissions"]
    }
)

# Usage
result = agent.run({
    "payload": raw_data,
    "schema_id": "emissions-data"
})

if result.data["is_valid"]:
    # Use coerced payload
    validated_data = result.data["coerced_payload"]
else:
    for error in result.data["validation_result"]["errors"]:
        print(f"{error['field']}: {error['message']}")
```

**Hybrid approach (keep Pydantic models):**

```python
from pydantic import BaseModel
from greenlang.agents.foundation import SchemaCompilerAgent

class Emission(BaseModel):
    fuel_type: str
    quantity: float
    unit: str
    co2e_emissions_kg: float

class EmissionsData(BaseModel):
    emissions: list[Emission]
    organization_id: str | None = None

# Pre-validate with GreenLang for enhanced features
agent = SchemaCompilerAgent()

def validate_and_parse(raw_data: dict) -> EmissionsData:
    # Step 1: GreenLang validation with coercion and unit checks
    result = agent.run({
        "payload": raw_data,
        "schema_id": "emissions-data",
        "enable_coercion": True,
        "enable_unit_check": True
    })

    if not result.data["is_valid"]:
        raise ValueError(f"Validation failed: {result.data['validation_result']['errors']}")

    # Step 2: Parse into Pydantic model
    return EmissionsData(**result.data["coerced_payload"])
```

---

### From Custom Validation Logic

If you have custom validation code, here's how to migrate to GL-FOUND-X-002.

**Before:**

```python
def validate_emissions(data):
    errors = []

    if not isinstance(data, dict):
        return ["Data must be an object"]

    if "emissions" not in data:
        errors.append("Missing required field: emissions")
    elif not isinstance(data["emissions"], list):
        errors.append("emissions must be an array")
    else:
        for i, emission in enumerate(data["emissions"]):
            if "fuel_type" not in emission:
                errors.append(f"emissions[{i}]: Missing fuel_type")
            if "co2e_emissions_kg" not in emission:
                errors.append(f"emissions[{i}]: Missing co2e_emissions_kg")
            elif not isinstance(emission["co2e_emissions_kg"], (int, float)):
                errors.append(f"emissions[{i}]: co2e_emissions_kg must be a number")
            if "scope" in emission and emission["scope"] not in [1, 2, 3]:
                errors.append(f"emissions[{i}]: scope must be 1, 2, or 3")

    return errors

# Usage
errors = validate_emissions(raw_data)
if errors:
    print("Validation errors:", errors)
```

**After:**

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Define schema declaratively
agent.register_schema(
    schema_id="emissions-custom",
    schema_name="Custom Emissions Schema",
    schema_content={
        "type": "object",
        "properties": {
            "emissions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fuel_type": {"type": "string"},
                        "co2e_emissions_kg": {"type": "number"},
                        "scope": {"type": "integer", "enum": [1, 2, 3]}
                    },
                    "required": ["fuel_type", "co2e_emissions_kg"]
                }
            }
        },
        "required": ["emissions"]
    }
)

def validate_emissions(data):
    result = agent.run({
        "payload": data,
        "schema_id": "emissions-custom"
    })

    if result.data["is_valid"]:
        return []
    else:
        return [e["message"] for e in result.data["validation_result"]["errors"]]

# Usage unchanged
errors = validate_emissions(raw_data)
if errors:
    print("Validation errors:", errors)
```

---

### From API Validation (Marshmallow, Cerberus)

#### From Marshmallow

**Before:**

```python
from marshmallow import Schema, fields, validate, ValidationError

class EmissionSchema(Schema):
    fuel_type = fields.Str(required=True)
    quantity = fields.Float(validate=validate.Range(min=0))
    unit = fields.Str()
    co2e_emissions_kg = fields.Float(required=True)
    scope = fields.Int(validate=validate.OneOf([1, 2, 3]))

class EmissionsDataSchema(Schema):
    emissions = fields.List(fields.Nested(EmissionSchema), required=True)
    organization_id = fields.Str()

schema = EmissionsDataSchema()
try:
    result = schema.load(raw_data)
except ValidationError as e:
    print(e.messages)
```

**After:**

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

# Equivalent JSON Schema
agent.register_schema(
    schema_id="emissions-marshmallow",
    schema_name="Emissions (Marshmallow Equivalent)",
    schema_content={
        "type": "object",
        "properties": {
            "emissions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fuel_type": {"type": "string"},
                        "quantity": {"type": "number", "minimum": 0},
                        "unit": {"type": "string"},
                        "co2e_emissions_kg": {"type": "number"},
                        "scope": {"type": "integer", "enum": [1, 2, 3]}
                    },
                    "required": ["fuel_type", "co2e_emissions_kg"]
                }
            },
            "organization_id": {"type": "string"}
        },
        "required": ["emissions"]
    }
)

result = agent.run({
    "payload": raw_data,
    "schema_id": "emissions-marshmallow",
    "enable_coercion": True  # Like Marshmallow's type coercion
})

if result.data["is_valid"]:
    validated_data = result.data["coerced_payload"]
else:
    errors = {e["field"]: e["message"] for e in result.data["validation_result"]["errors"]}
    print(errors)
```

---

## Schema Migration

### Converting Existing Schemas

#### JSON Schema to GreenLang Schema

JSON Schema Draft-07 schemas work directly with GL-FOUND-X-002. You can register them as-is:

```python
# Your existing JSON Schema
existing_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {...}
}

# Register directly
agent.register_schema(
    schema_id="existing-schema",
    schema_name="Existing Schema",
    schema_content=existing_schema
)
```

#### Adding GreenLang Features

Enhance your schemas with GreenLang-specific features:

```python
# Enhanced schema with GreenLang features
enhanced_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "emissions_kg": {
            "type": "number",
            "minimum": 0,
            # GreenLang unit annotation
            "x-gl-unit": "kgCO2e"
        },
        "energy_kwh": {
            "type": "number",
            "minimum": 0,
            "x-gl-unit": "kWh"
        },
        "scope": {
            "type": "integer",
            "enum": [1, 2, 3],
            # GreenLang metadata
            "x-gl-description": "GHG Protocol emission scope"
        }
    },
    # GreenLang schema metadata
    "x-gl-schema": {
        "category": "emissions",
        "compliance": ["GHG Protocol", "ISO 14064"]
    }
}
```

---

## CLI Migration

### From Custom Validation Scripts

**Before (custom script):**

```bash
#!/bin/bash
python validate.py --input data.yaml --schema schema.json
if [ $? -ne 0 ]; then
    echo "Validation failed"
    exit 1
fi
```

**After:**

```bash
#!/bin/bash
greenlang schema validate data.yaml --schema my-schema@1.0.0
if [ $? -ne 0 ]; then
    echo "Validation failed"
    exit 1
fi
```

### From Other Validation CLIs

**From ajv-cli:**

```bash
# Before
ajv validate -s schema.json -d data.json

# After
greenlang schema validate data.json --schema-file schema.json
```

**From tv4:**

```bash
# Before
tv4 --schema schema.json data.json

# After
greenlang validate data.json --schema my-schema@1.0.0
```

---

## API Migration

### From Custom Validation Endpoints

**Before (FastAPI custom endpoint):**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jsonschema

app = FastAPI()

@app.post("/validate")
def validate_data(data: dict):
    schema = load_schema("my-schema")
    try:
        jsonschema.validate(data, schema)
        return {"valid": True}
    except jsonschema.ValidationError as e:
        return {"valid": False, "error": e.message}
```

**After (using GreenLang API):**

```python
from fastapi import FastAPI, Depends
from greenlang.schema.api.routes import router as schema_router

app = FastAPI()

# Include GreenLang schema routes
app.include_router(schema_router, prefix="/v1/schema")

# Your endpoint now uses: POST /v1/schema/validate
```

Or call the GreenLang API from your existing endpoint:

```python
from greenlang.agents.foundation import SchemaCompilerAgent

app = FastAPI()
agent = SchemaCompilerAgent()

@app.post("/validate")
def validate_data(data: dict, schema_id: str):
    result = agent.run({
        "payload": data,
        "schema_id": schema_id
    })

    return {
        "valid": result.data["is_valid"],
        "errors": result.data["validation_result"]["errors"],
        "fix_suggestions": result.data["fix_suggestions"]
    }
```

---

## Step-by-Step Migration

### Phase 1: Parallel Running (Week 1-2)

Run both old and new validation in parallel to compare results.

```python
from greenlang.agents.foundation import SchemaCompilerAgent

# Your existing validator
def old_validate(data):
    # ... existing logic
    return old_errors

# New GreenLang validator
agent = SchemaCompilerAgent()

def new_validate(data, schema_id):
    result = agent.run({
        "payload": data,
        "schema_id": schema_id
    })
    return result.data["validation_result"]["errors"]

# Parallel validation
def validate_with_comparison(data, schema_id):
    old_errors = old_validate(data)
    new_errors = new_validate(data, schema_id)

    # Log comparison for analysis
    if len(old_errors) != len(new_errors):
        logger.warning(f"Mismatch: old={len(old_errors)}, new={len(new_errors)}")

    # Return old results during transition
    return old_errors
```

### Phase 2: Feature Enablement (Week 3-4)

Enable GreenLang features while still using old validation as fallback.

```python
def validate_with_features(data, schema_id, use_coercion=True):
    result = agent.run({
        "payload": data,
        "schema_id": schema_id,
        "enable_coercion": use_coercion,
        "enable_unit_check": True
    })

    if result.data["is_valid"]:
        # Use coerced payload
        return {"valid": True, "data": result.data["coerced_payload"]}
    else:
        return {
            "valid": False,
            "errors": result.data["validation_result"]["errors"],
            "fixes": result.data["fix_suggestions"]
        }
```

### Phase 3: Full Cutover (Week 5+)

Remove old validation and use GreenLang exclusively.

```python
from greenlang.agents.foundation import SchemaCompilerAgent

agent = SchemaCompilerAgent()

def validate(data, schema_id):
    result = agent.run({
        "payload": data,
        "schema_id": schema_id,
        "enable_coercion": True,
        "enable_unit_check": True,
        "generate_fixes": True
    })

    return {
        "valid": result.data["is_valid"],
        "data": result.data["coerced_payload"] if result.data["is_valid"] else None,
        "errors": result.data["validation_result"]["errors"],
        "fixes": result.data["fix_suggestions"],
        "provenance": result.data["provenance_hash"]
    }
```

---

## Common Migration Issues

### Issue 1: Type Coercion Differences

**Problem:** GreenLang's coercion may be stricter than your current solution.

**Solution:** Review coercion settings and adjust:

```python
# Disable coercion for strict validation
result = agent.run({
    "payload": data,
    "schema_id": schema_id,
    "enable_coercion": False
})

# Or allow lossy coercion
# (Note: This is controlled per-coercion in the engine)
```

### Issue 2: Error Message Differences

**Problem:** Error messages have different format/wording.

**Solution:** Map new error formats to old format if needed:

```python
def convert_errors(gl_errors):
    """Convert GreenLang errors to old format."""
    return [
        {
            "path": e["field"],
            "msg": e["message"],
            "type": e["validator"]
        }
        for e in gl_errors
    ]
```

### Issue 3: Missing Custom Validators

**Problem:** Custom validation logic doesn't have schema equivalent.

**Solution:** Add post-validation checks:

```python
def validate_with_custom_rules(data, schema_id):
    # Standard schema validation
    result = agent.run({
        "payload": data,
        "schema_id": schema_id
    })

    errors = list(result.data["validation_result"]["errors"])

    # Add custom validation
    if result.data["is_valid"]:
        payload = result.data["coerced_payload"]

        # Custom rule: total must match sum
        if "items" in payload:
            calculated_total = sum(item["value"] for item in payload["items"])
            if payload.get("total") != calculated_total:
                errors.append({
                    "field": "total",
                    "message": f"Total {payload.get('total')} does not match sum {calculated_total}",
                    "severity": "error"
                })

    return {"valid": len(errors) == 0, "errors": errors}
```

---

## Rollback Plan

If you need to rollback, keep these in place:

1. **Feature flags:** Control which validation system is active
2. **Parallel logging:** Log both validation results for comparison
3. **Schema backups:** Keep original schemas in version control
4. **Dependency isolation:** Keep old validation library available

```python
import os

USE_GREENLANG = os.environ.get("USE_GREENLANG_VALIDATION", "false").lower() == "true"

def validate(data, schema_id):
    if USE_GREENLANG:
        return greenlang_validate(data, schema_id)
    else:
        return legacy_validate(data, schema_id)
```

---

## See Also

- [SDK Guide](sdk.md) - Detailed Python SDK documentation
- [CLI Reference](cli.md) - Command-line usage
- [API Reference](api.md) - REST API documentation
- [Error Codes](error_codes.md) - Error code reference
