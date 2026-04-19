"""Invalid test payloads that should fail validation.

This package contains 30+ invalid payload test files covering all error codes:

Structural Errors (E1xx):
    - missing_required_001.yaml: GLSCHEMA-E100 (MISSING_REQUIRED)
    - unknown_field_001.yaml: GLSCHEMA-E101 (UNKNOWN_FIELD)
    - type_mismatch_001.yaml, type_mismatch_002.yaml, type_mismatch_003.yaml: GLSCHEMA-E102
    - null_violation_001.yaml: GLSCHEMA-E103 (INVALID_NULL)
    - property_count_violation_001.yaml: GLSCHEMA-E105 (PROPERTY_COUNT_VIOLATION)

Constraint Errors (E2xx):
    - range_violation_001.yaml, range_violation_002.yaml, range_violation_003.yaml: GLSCHEMA-E200
    - pattern_mismatch_001.yaml, pattern_mismatch_002.yaml: GLSCHEMA-E201
    - enum_violation_001.yaml, enum_violation_002.yaml: GLSCHEMA-E202
    - length_violation_001.yaml, length_violation_002.yaml: GLSCHEMA-E203
    - unique_violation_001.yaml: GLSCHEMA-E204
    - multiple_of_violation_001.yaml: GLSCHEMA-E205
    - format_violation_001.yaml to format_violation_004.yaml: GLSCHEMA-E206
    - const_violation_001.yaml: GLSCHEMA-E207
    - contains_violation_001.yaml: GLSCHEMA-E208
    - property_name_violation_001.yaml: GLSCHEMA-E209

Unit Errors (E3xx):
    - unit_missing_001.yaml: GLSCHEMA-E300
    - unit_incompatible_001.yaml: GLSCHEMA-E301
    - unit_unknown_001.yaml: GLSCHEMA-E303

Rule Errors (E4xx):
    - rule_violation_001.yaml to rule_violation_004.yaml: GLSCHEMA-E400
    - dependency_violation_001.yaml: GLSCHEMA-E403
    - one_of_violation_001.yaml: GLSCHEMA-E405
    - any_of_violation_001.yaml: GLSCHEMA-E406

Each payload file contains a _test_metadata section specifying:
    - schema: The schema file to validate against
    - description: Human-readable description
    - expected_valid: Always false for invalid payloads
    - expected_errors: List of expected error codes and paths
"""
