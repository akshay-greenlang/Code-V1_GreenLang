"""Valid test payloads that should pass validation.

This package contains 20 valid payload test files:
    - string_valid_001.yaml, string_valid_002.yaml
    - numeric_valid_001.yaml, numeric_valid_002.yaml, numeric_valid_003.yaml
    - array_valid_001.yaml, array_valid_002.yaml
    - object_valid_001.yaml, object_valid_002.yaml
    - enum_valid_001.yaml, enum_valid_002.yaml
    - type_valid_001.yaml, type_valid_002.yaml
    - energy_units_valid_001.yaml
    - mass_units_valid_001.yaml
    - conditional_rules_valid_001.yaml, conditional_rules_valid_002.yaml, conditional_rules_valid_003.yaml
    - dependency_rules_valid_001.yaml, dependency_rules_valid_002.yaml
    - ref_schema_valid_001.yaml
    - combined_valid_001.yaml

Each payload file contains a _test_metadata section specifying:
    - schema: The schema file to validate against
    - description: Human-readable description
    - expected_valid: Always true for valid payloads
"""
