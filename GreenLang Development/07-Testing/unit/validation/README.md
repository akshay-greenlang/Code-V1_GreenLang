# Validation Tests

Comprehensive test suite for GreenLang validation modules.

## Quick Start

```bash
# Run all validation tests
pytest tests/unit/validation/ -v

# Run with coverage
pytest tests/unit/validation/ --cov=greenlang.validation --cov-report=html

# Run specific test file
pytest tests/unit/validation/test_framework.py -v
```

## Test Files

### test_framework.py (561 lines)
Tests for ValidationFramework, ValidationResult, and ValidationError classes.

**Coverage:**
- ✓ Adding/removing validators
- ✓ Pre/post validator hooks
- ✓ Batch validation
- ✓ Error collection and merging
- ✓ Severity levels (ERROR, WARNING, INFO)
- ✓ Validator enabling/disabling
- ✓ Exception handling

**Example:**
```python
from greenlang.validation.framework import ValidationFramework

framework = ValidationFramework()
framework.add_validator("schema", schema_validator)
result = framework.validate(data)
```

### test_schema.py (464 lines)
Tests for JSON Schema validation with jsonschema library.

**Coverage:**
- ✓ JSON Schema Draft 7 validation
- ✓ Required fields and type checking
- ✓ Nested objects and arrays
- ✓ Format validators (email, date, uri)
- ✓ Schema constraints (min/max, pattern, enum)
- ✓ Graceful degradation without jsonschema
- ✓ Error reporting with field paths

**Example:**
```python
from greenlang.validation.schema import SchemaValidator

schema = {"type": "object", "properties": {"name": {"type": "string"}}}
validator = SchemaValidator(schema)
result = validator.validate(data)
```

### test_rules.py (730 lines)
Tests for business rules engine with all operators.

**Coverage:**
- ✓ All 12 operators (==, !=, >, >=, <, <=, in, not_in, contains, regex, is_null, not_null)
- ✓ Nested field paths (e.g., "address.city")
- ✓ Conditional rules
- ✓ Rule sets for organization
- ✓ Custom error messages
- ✓ Rule enabling/disabling
- ✓ Loading rules from config

**Example:**
```python
from greenlang.validation.rules import RulesEngine, Rule, RuleOperator

engine = RulesEngine()
rule = Rule(
    name="age_check",
    field="age",
    operator=RuleOperator.GREATER_EQUAL,
    value=18
)
engine.add_rule(rule)
result = engine.validate(data)
```

## Test Statistics

- **Total Lines**: 1,755
- **Test Classes**: 23+
- **Test Functions**: 100+
- **Coverage Target**: >90%

## Key Features Tested

### ValidationFramework
- Multi-layer validation
- Validator orchestration
- Result aggregation
- Hook system (pre/post)
- Batch processing

### SchemaValidator
- JSON Schema compliance
- Format validation
- Nested structure validation
- Error path tracking
- Fallback validation

### RulesEngine
- Business rule evaluation
- Multiple comparison operators
- Nested field access
- Conditional execution
- Rule organization

## Fixtures (conftest.py)

Shared fixtures available to all tests:
- `sample_valid_data`: Valid test data
- `sample_invalid_data`: Invalid test data
- `temp_schema_file`: Temporary schema file

## Dependencies

**Required:**
- pytest

**Optional (for full coverage):**
- jsonschema (for schema validation tests)

Tests automatically skip when optional dependencies are missing.
