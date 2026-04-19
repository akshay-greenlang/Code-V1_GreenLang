# Golden Test Suite for GL-FOUND-X-002

This directory contains comprehensive golden tests for the GreenLang Schema Compiler & Validator (GL-FOUND-X-002).

## Directory Structure

```
tests/schema/golden/
├── __init__.py              # Package documentation
├── conftest.py              # Test fixtures and configuration
├── test_golden.py           # Main golden test module
├── README.md                # This file
│
├── schemas/                 # Test schema definitions
│   ├── basic/               # Basic constraint schemas
│   │   ├── string_constraints.yaml
│   │   ├── numeric_constraints.yaml
│   │   ├── array_constraints.yaml
│   │   ├── object_constraints.yaml
│   │   ├── enum_constraints.yaml
│   │   ├── type_constraints.yaml
│   │   ├── ref_schema.yaml
│   │   └── combined_schema.yaml
│   │
│   ├── units/               # Unit validation schemas
│   │   ├── energy_units.yaml
│   │   └── mass_units.yaml
│   │
│   └── rules/               # Rule validation schemas
│       ├── conditional_rules.yaml
│       └── dependency_rules.yaml
│
├── payloads/                # Test payloads
│   ├── valid/               # Payloads that should pass (20+)
│   └── invalid/             # Payloads that should fail (30+)
│
└── expected/                # Expected validation reports
    ├── string_valid_001_report.json
    ├── missing_required_001_report.json
    └── ...
```

## Test Coverage

### Schemas (13 total)

| Schema | Constraints Tested |
|--------|-------------------|
| string_constraints.yaml | minLength, maxLength, pattern, format (email, uuid, uri, date, datetime) |
| numeric_constraints.yaml | minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf |
| array_constraints.yaml | minItems, maxItems, uniqueItems, contains |
| object_constraints.yaml | required, additionalProperties, minProperties, maxProperties, propertyNames |
| enum_constraints.yaml | enum, const |
| type_constraints.yaml | type (string, number, integer, boolean, array, object, null) |
| ref_schema.yaml | $ref, $defs |
| combined_schema.yaml | Complex multi-constraint schema |
| energy_units.yaml | x-gl-unit dimension (energy) |
| mass_units.yaml | x-gl-unit dimension (mass, emissions) |
| conditional_rules.yaml | x-gl-rules (conditional, sum validation) |
| dependency_rules.yaml | dependentRequired, oneOf |

### Valid Payloads (23 total)

All valid payloads should pass validation with zero errors.

| Payload | Tests |
|---------|-------|
| string_valid_001.yaml | All string constraints satisfied |
| string_valid_002.yaml | Minimal required fields only |
| numeric_valid_001.yaml | All numeric constraints satisfied |
| numeric_valid_002.yaml | Boundary minimum values |
| numeric_valid_003.yaml | Boundary maximum values |
| array_valid_001.yaml | All array constraints satisfied |
| array_valid_002.yaml | Single item arrays |
| object_valid_001.yaml | All object constraints satisfied |
| object_valid_002.yaml | Minimal required fields |
| enum_valid_001.yaml | Valid enum values |
| enum_valid_002.yaml | Nullable enum |
| type_valid_001.yaml | All type constraints |
| type_valid_002.yaml | Nullable and multi-type |
| energy_units_valid_001.yaml | Valid energy units |
| mass_units_valid_001.yaml | Valid mass units |
| conditional_rules_valid_001.yaml | Natural gas with methane_slip |
| conditional_rules_valid_002.yaml | Electricity with grid_region |
| conditional_rules_valid_003.yaml | Sum validation correct |
| dependency_rules_valid_001.yaml | Individual type |
| dependency_rules_valid_002.yaml | Organization type |
| ref_schema_valid_001.yaml | Valid $ref resolution |
| combined_valid_001.yaml | Complex schema |

### Invalid Payloads (44 total)

Invalid payloads should fail validation with specific error codes.

#### Structural Errors (E1xx)

| Payload | Error Code | Description |
|---------|------------|-------------|
| missing_required_001.yaml | GLSCHEMA-E100 | Missing required field |
| unknown_field_001.yaml | GLSCHEMA-E101 | Unknown field (strict mode) |
| type_mismatch_001.yaml | GLSCHEMA-E102 | String expected, number given |
| type_mismatch_002.yaml | GLSCHEMA-E102 | Integer expected, string given |
| type_mismatch_003.yaml | GLSCHEMA-E102 | Array expected, object given |
| null_violation_001.yaml | GLSCHEMA-E103 | Null where not allowed |
| property_count_violation_001.yaml | GLSCHEMA-E105 | Too many properties |

#### Constraint Errors (E2xx)

| Payload | Error Code | Description |
|---------|------------|-------------|
| range_violation_001.yaml | GLSCHEMA-E200 | Below minimum |
| range_violation_002.yaml | GLSCHEMA-E200 | Above maximum |
| range_violation_003.yaml | GLSCHEMA-E200 | At exclusive minimum |
| pattern_mismatch_001.yaml | GLSCHEMA-E201 | Username pattern |
| pattern_mismatch_002.yaml | GLSCHEMA-E201 | Phone pattern |
| enum_violation_001.yaml | GLSCHEMA-E202 | Invalid string enum |
| enum_violation_002.yaml | GLSCHEMA-E202 | Invalid numeric enum |
| length_violation_001.yaml | GLSCHEMA-E203 | Too short |
| length_violation_002.yaml | GLSCHEMA-E203 | Too long |
| unique_violation_001.yaml | GLSCHEMA-E204 | Duplicate array items |
| multiple_of_violation_001.yaml | GLSCHEMA-E205 | Not multiple of |
| format_violation_001.yaml | GLSCHEMA-E206 | Invalid email |
| format_violation_002.yaml | GLSCHEMA-E206 | Invalid date |
| format_violation_003.yaml | GLSCHEMA-E206 | Invalid UUID |
| format_violation_004.yaml | GLSCHEMA-E206 | Invalid URI |
| const_violation_001.yaml | GLSCHEMA-E207 | Const mismatch |
| contains_violation_001.yaml | GLSCHEMA-E208 | Missing required item |
| property_name_violation_001.yaml | GLSCHEMA-E209 | Invalid property name |

#### Unit Errors (E3xx)

| Payload | Error Code | Description |
|---------|------------|-------------|
| unit_missing_001.yaml | GLSCHEMA-E300 | Missing unit field |
| unit_incompatible_001.yaml | GLSCHEMA-E301 | Wrong dimension |
| unit_unknown_001.yaml | GLSCHEMA-E303 | Unrecognized unit |

#### Rule Errors (E4xx)

| Payload | Error Code | Description |
|---------|------------|-------------|
| rule_violation_001.yaml | GLSCHEMA-E400 | Natural gas missing methane_slip |
| rule_violation_002.yaml | GLSCHEMA-E402 | Sum total mismatch |
| rule_violation_003.yaml | GLSCHEMA-E400 | Biofuel missing blend |
| rule_violation_004.yaml | GLSCHEMA-E400 | Electricity missing grid_region |
| dependency_violation_001.yaml | GLSCHEMA-E403 | Property dependency |
| one_of_violation_001.yaml | GLSCHEMA-E405 | Matches multiple oneOf |
| any_of_violation_001.yaml | GLSCHEMA-E406 | Matches none of anyOf |

## Running Tests

```bash
# Run all golden tests
pytest tests/schema/golden/test_golden.py -v

# Run only valid payload tests
pytest tests/schema/golden/test_golden.py -v -k "valid"

# Run only invalid payload tests
pytest tests/schema/golden/test_golden.py -v -k "invalid"

# Run error code coverage tests
pytest tests/schema/golden/test_golden.py -v -k "error_code"

# Run with coverage report
pytest tests/schema/golden/test_golden.py --cov=greenlang.schema --cov-report=html
```

## Adding New Test Cases

### Adding a Valid Payload

1. Create a YAML file in `payloads/valid/` with naming convention `{category}_valid_{nnn}.yaml`
2. Include `_test_metadata` section:

```yaml
_test_metadata:
  schema: "basic/string_constraints.yaml"
  description: "Description of what this tests"
  expected_valid: true

# Actual payload data
field1: value1
field2: value2
```

### Adding an Invalid Payload

1. Create a YAML file in `payloads/invalid/` with naming convention `{error_type}_{nnn}.yaml`
2. Include `_test_metadata` section with expected errors:

```yaml
_test_metadata:
  schema: "basic/string_constraints.yaml"
  description: "Description of what this tests"
  expected_valid: false
  expected_errors:
    - code: "GLSCHEMA-E100"
      path: "/field_name"
      severity: "error"

# Payload data with intentional errors
field1: invalid_value
```

### Adding an Expected Report

1. Create a JSON file in `expected/` with naming convention `{payload_name}_report.json`
2. Follow the ValidationReport structure:

```json
{
  "valid": false,
  "schema_ref": {
    "schema_id": "gl://schemas/test/...",
    "version": "1.0.0"
  },
  "summary": {
    "error_count": 1,
    "warning_count": 0,
    "info_count": 0,
    "total_findings": 1
  },
  "findings": [
    {
      "code": "GLSCHEMA-E100",
      "severity": "error",
      "path": "/field",
      "message": "Human-readable message",
      "expected": {},
      "actual": {}
    }
  ]
}
```

## Error Code Reference

| Code | Name | Description |
|------|------|-------------|
| GLSCHEMA-E100 | MISSING_REQUIRED | Required field is missing |
| GLSCHEMA-E101 | UNKNOWN_FIELD | Unknown field not in schema |
| GLSCHEMA-E102 | TYPE_MISMATCH | Value type does not match schema |
| GLSCHEMA-E103 | INVALID_NULL | Null value not allowed |
| GLSCHEMA-E104 | CONTAINER_TYPE_MISMATCH | Container type mismatch |
| GLSCHEMA-E105 | PROPERTY_COUNT_VIOLATION | Too many/few properties |
| GLSCHEMA-E200 | RANGE_VIOLATION | Numeric value out of range |
| GLSCHEMA-E201 | PATTERN_MISMATCH | String does not match pattern |
| GLSCHEMA-E202 | ENUM_VIOLATION | Value not in enum |
| GLSCHEMA-E203 | LENGTH_VIOLATION | String/array length out of range |
| GLSCHEMA-E204 | UNIQUE_VIOLATION | Duplicate items in uniqueItems array |
| GLSCHEMA-E205 | MULTIPLE_OF_VIOLATION | Not a multiple of constraint |
| GLSCHEMA-E206 | FORMAT_VIOLATION | Does not match format |
| GLSCHEMA-E207 | CONST_VIOLATION | Does not match const |
| GLSCHEMA-E208 | CONTAINS_VIOLATION | Array missing required item |
| GLSCHEMA-E209 | PROPERTY_NAME_VIOLATION | Property name pattern mismatch |
| GLSCHEMA-E300 | UNIT_MISSING | Required unit field missing |
| GLSCHEMA-E301 | UNIT_INCOMPATIBLE | Unit dimension incompatible |
| GLSCHEMA-E302 | UNIT_NONCANONICAL | Unit is not canonical (warning) |
| GLSCHEMA-E303 | UNIT_UNKNOWN | Unrecognized unit |
| GLSCHEMA-E400 | RULE_VIOLATION | Cross-field rule violation |
| GLSCHEMA-E401 | CONDITIONAL_REQUIRED | Conditional requirement not met |
| GLSCHEMA-E402 | CONSISTENCY_ERROR | Cross-field consistency error |
| GLSCHEMA-E403 | DEPENDENCY_VIOLATION | Property dependency not met |
| GLSCHEMA-E405 | ONE_OF_VIOLATION | Does not match exactly one schema |
| GLSCHEMA-E406 | ANY_OF_VIOLATION | Does not match any schema |
