"""
Test suite for GL-FOUND-X-002: GreenLang Schema Compiler & Validator.

This package contains comprehensive tests organized into:
    - unit/: Unit tests for individual modules
    - integration/: Integration tests for module interactions
    - golden/: Golden test data (schemas, payloads, expected outputs)
    - property/: Property-based tests using Hypothesis
    - security/: Security tests (ReDoS, YAML bombs, etc.)

Test Coverage Targets:
    - Unit tests: 85%+ coverage
    - Golden tests: 100+ test cases
    - Property tests: Normalization idempotency, patch monotonicity

Markers:
    - @pytest.mark.golden: Golden test data validation
    - @pytest.mark.property: Property-based tests
    - @pytest.mark.security: Security-focused tests
    - @pytest.mark.slow: Tests that take >1 second
"""
