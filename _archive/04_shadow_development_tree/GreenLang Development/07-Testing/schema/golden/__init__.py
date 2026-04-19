"""
Golden Test Suite for GL-FOUND-X-002 Schema Compiler & Validator.

This package provides comprehensive golden tests for validating the
GreenLang Schema Compiler & Validator implementation.

Directory Structure:
    schemas/
        basic/          - Basic constraint test schemas (10 schemas)
            - string_constraints.yaml
            - numeric_constraints.yaml
            - array_constraints.yaml
            - object_constraints.yaml
            - enum_constraints.yaml
            - type_constraints.yaml
            - ref_schema.yaml
            - combined_schema.yaml
        units/          - Unit validation schemas
            - energy_units.yaml
            - mass_units.yaml
        rules/          - Rule validation schemas
            - conditional_rules.yaml
            - dependency_rules.yaml

    payloads/
        valid/          - Valid test payloads (20 payloads)
        invalid/        - Invalid test payloads (30+ payloads)

    expected/           - Expected validation reports (JSON)

Test Coverage:
    - 10 test schemas covering different constraint types
    - 20 valid payloads that should pass validation
    - 30+ invalid payloads testing each error code:
        - E1xx: Structural errors (missing required, type mismatch, etc.)
        - E2xx: Constraint errors (range, pattern, enum, length, etc.)
        - E3xx: Unit errors (missing, incompatible, unknown)
        - E4xx: Rule errors (conditional, dependency, oneOf/anyOf)

Usage:
    pytest tests/schema/golden/test_golden.py -v
    pytest tests/schema/golden/test_golden.py -v -k "valid"
    pytest tests/schema/golden/test_golden.py -v -k "invalid"
    pytest tests/schema/golden/test_golden.py -v -k "error_code"

Golden tests ensure that validation behavior remains consistent
across code changes and refactoring.
"""
