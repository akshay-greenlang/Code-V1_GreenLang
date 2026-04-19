# -*- coding: utf-8 -*-
"""
Property-Based Tests for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator)

This package contains property-based tests using Hypothesis to verify
mathematical properties of the schema validation and normalization system.

Property-based testing generates random inputs to find edge cases that
example-based tests might miss. Each test verifies a specific property
that should hold for ALL valid inputs.

Test Modules:
    test_normalization_idempotent.py
        Property: normalize(normalize(x)) == normalize(x)
        Ensures normalization is stable and can be applied multiple times.

    test_patch_monotonic.py
        Property: errors(apply_patches(x, safe_patches)) <= errors(x)
        Ensures safe patches reduce or maintain error count.

    test_validation_determinism.py
        Property: validate(x, schema) == validate(x, schema) [always]
        Ensures validation is a pure function with deterministic output.

    test_coercion_reversible.py
        Property: reverse(coerce(x)) == x [for safe coercions]
        Ensures safe coercions can be reversed without data loss.

Running Property Tests:
    # Run all property tests
    pytest tests/schema/property/ -v

    # Run with more examples (slower but more thorough)
    pytest tests/schema/property/ -v --hypothesis-seed=0

    # Run specific test module
    pytest tests/schema/property/test_normalization_idempotent.py -v

Hypothesis Settings:
    - Default: 100 examples per property test
    - Extended: Use @settings(max_examples=1000) for thorough testing
    - Deadline: Disabled for complex tests to avoid timeouts

Key Properties Tested:

1. IDEMPOTENCY (Normalization)
   - normalize(normalize(x)) == normalize(x)
   - Multiple normalizations converge to same result
   - Coerced values remain stable

2. MONOTONICITY (Patches)
   - Safe patches never increase error count
   - Cumulative patches are monotonically non-increasing
   - Preconditions protect against invalid patches

3. DETERMINISM (Validation)
   - Same input always produces same output
   - Finding order is deterministic
   - Result hashes are reproducible

4. REVERSIBILITY (Coercion)
   - Safe coercions can be reversed
   - Original values recoverable from coercion records
   - Type information preserved

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Property Tests
"""

from __future__ import annotations

# Test module exports for easy importing
__all__ = [
    "test_normalization_idempotent",
    "test_patch_monotonic",
    "test_validation_determinism",
    "test_coercion_reversible",
]
