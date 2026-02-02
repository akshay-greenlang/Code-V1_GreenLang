# -*- coding: utf-8 -*-
"""
Property-Based Tests: Normalization Idempotency

Tests the critical property that normalize(normalize(x)) == normalize(x).
This ensures that normalization is a stable operation - applying it multiple
times produces the same result as applying it once.

This property is essential for:
    - Deterministic pipelines: Same input always produces same output
    - Composability: Chained normalizations are safe
    - Audit reproducibility: Re-running normalization doesn't change data

Uses Hypothesis to generate random payloads and verify the property holds
for all possible inputs.

GL-FOUND-X-002: Schema Compiler & Validator - Property Tests
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, assume, HealthCheck, Phase, example
from hypothesis import strategies as st

# Import the components under test
from greenlang.schema.normalizer.engine import (
    NormalizationEngine,
    NormalizationResult,
    normalize,
    is_normalization_idempotent,
    _remove_meta_block,
)
from greenlang.schema.normalizer.coercions import CoercionEngine, CoercionPolicy
from greenlang.schema.compiler.ir import SchemaIR, PropertyIR, UnitSpecIR
from greenlang.schema.units.catalog import UnitCatalog
from greenlang.schema.models.config import ValidationOptions, CoercionPolicy as ConfigCoercionPolicy


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Strategy for generating JSON-compatible primitive values
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-2**31, max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
    st.text(
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'),
            blacklist_characters='\x00'
        ),
        min_size=0,
        max_size=100
    ),
)

# Strategy for generating valid JSON keys (snake_case preferred for normalization)
json_keys = st.text(
    alphabet='abcdefghijklmnopqrstuvwxyz_0123456789',
    min_size=1,
    max_size=30
).filter(lambda x: x[0].isalpha())

# Strategy for generating coercible string values
coercible_strings = st.one_of(
    st.integers(min_value=-10000, max_value=10000).map(str),  # "42"
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000).map(str),  # "3.14"
    st.sampled_from(["true", "false", "True", "False", "TRUE", "FALSE"]),  # booleans
    st.sampled_from(["null", "Null", "NULL"]),  # null
)

# Strategy for generating numeric values with potential units
numeric_with_unit = st.one_of(
    st.integers(min_value=0, max_value=10000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=10000),
    st.tuples(
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=10000),
        st.sampled_from(['kWh', 'MWh', 'GJ', 'kg', 'tonne', 'm3', 'L'])
    ).map(lambda x: f"{x[0]} {x[1]}"),
)


def json_values_recursive(max_depth: int = 3):
    """
    Generate recursive JSON-compatible values with controlled depth.

    Args:
        max_depth: Maximum nesting depth for objects and arrays

    Returns:
        Hypothesis strategy for JSON values
    """
    if max_depth <= 0:
        return json_primitives

    return st.one_of(
        json_primitives,
        st.lists(
            st.deferred(lambda: json_values_recursive(max_depth - 1)),
            min_size=0,
            max_size=5
        ),
        st.dictionaries(
            json_keys,
            st.deferred(lambda: json_values_recursive(max_depth - 1)),
            min_size=0,
            max_size=5
        ),
    )


# Strategy for generating complete payloads
json_payloads = st.dictionaries(
    json_keys,
    json_values_recursive(max_depth=3),
    min_size=0,
    max_size=10
)


# Strategy for payloads with coercible values
payloads_with_coercible_values = st.dictionaries(
    json_keys,
    st.one_of(
        json_primitives,
        coercible_strings,
        numeric_with_unit,
    ),
    min_size=1,
    max_size=10
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def minimal_schema_ir() -> SchemaIR:
    """Create a minimal SchemaIR for testing normalization."""
    return SchemaIR(
        schema_id="test/minimal",
        version="1.0.0",
        schema_hash="a" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={},
        required_paths=set(),
    )


@pytest.fixture
def schema_ir_with_properties() -> SchemaIR:
    """Create a SchemaIR with common property definitions."""
    return SchemaIR(
        schema_id="test/with_properties",
        version="1.0.0",
        schema_hash="b" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=False),
            "/value": PropertyIR(path="/value", type="number", required=False),
            "/count": PropertyIR(path="/count", type="integer", required=False),
            "/active": PropertyIR(path="/active", type="boolean", required=False),
            "/energy": PropertyIR(path="/energy", type="number", required=False),
        },
        required_paths=set(),
    )


@pytest.fixture
def unit_catalog() -> UnitCatalog:
    """Create a unit catalog for testing."""
    return UnitCatalog()


@pytest.fixture
def default_options() -> ValidationOptions:
    """Create default validation options."""
    return ValidationOptions(
        normalize=True,
        coercion_policy=ConfigCoercionPolicy.SAFE,
    )


@pytest.fixture
def normalization_engine(schema_ir_with_properties, unit_catalog, default_options):
    """Create a NormalizationEngine for testing."""
    return NormalizationEngine(
        ir=schema_ir_with_properties,
        catalog=unit_catalog,
        options=default_options,
    )


# =============================================================================
# IDEMPOTENCY TESTS
# =============================================================================

@pytest.mark.property
class TestNormalizationIdempotency:
    """
    Property-based tests for normalization idempotency.

    The key property: normalize(normalize(x)) == normalize(x)
    """

    @given(payload=json_payloads)
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
        phases=[Phase.generate, Phase.target, Phase.shrink],
    )
    def test_idempotency_with_random_payloads(
        self,
        payload: Dict[str, Any],
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """
        Test that normalization is idempotent for random JSON payloads.

        Property: normalize(normalize(x)) == normalize(x)

        This means applying normalization twice should produce the same
        result as applying it once - a fundamental requirement for
        deterministic data pipelines.
        """
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        # First normalization
        result1 = engine.normalize(payload)

        # Remove _meta block for second normalization (meta contains timestamps)
        payload2 = _remove_meta_block(result1.normalized)

        # Second normalization
        result2 = engine.normalize(payload2)

        # Compare normalized results (excluding _meta blocks)
        normalized1 = _remove_meta_block(result1.normalized)
        normalized2 = _remove_meta_block(result2.normalized)

        assert normalized1 == normalized2, (
            f"Idempotency violation!\n"
            f"Input: {payload}\n"
            f"First normalize: {normalized1}\n"
            f"Second normalize: {normalized2}"
        )

    @given(payload=payloads_with_coercible_values)
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_idempotency_with_coercible_values(
        self,
        payload: Dict[str, Any],
        schema_ir_with_properties,
        unit_catalog,
        default_options,
    ):
        """
        Test idempotency when payloads contain coercible string values.

        Coercions like "42" -> 42 should be stable: once coerced, the
        value should not change on subsequent normalizations.
        """
        engine = NormalizationEngine(
            ir=schema_ir_with_properties,
            catalog=unit_catalog,
            options=default_options,
        )

        # First normalization
        result1 = engine.normalize(payload)

        # Second normalization
        payload2 = _remove_meta_block(result1.normalized)
        result2 = engine.normalize(payload2)

        # Compare
        normalized1 = _remove_meta_block(result1.normalized)
        normalized2 = _remove_meta_block(result2.normalized)

        assert normalized1 == normalized2, (
            f"Idempotency violation with coercible values!\n"
            f"Input: {payload}\n"
            f"First normalize: {normalized1}\n"
            f"Second normalize: {normalized2}"
        )

    @given(
        payload=json_payloads,
        iterations=st.integers(min_value=2, max_value=5)
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_multiple_normalizations_converge(
        self,
        payload: Dict[str, Any],
        iterations: int,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """
        Test that multiple normalizations always converge to the same result.

        This is a stronger form of idempotency: for any N >= 1,
        normalize^N(x) == normalize(x)
        """
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        # First normalization (the reference)
        result1 = engine.normalize(payload)
        reference = _remove_meta_block(result1.normalized)

        # Apply normalization multiple times
        current = copy.deepcopy(payload)
        for i in range(iterations):
            result = engine.normalize(current)
            current = _remove_meta_block(result.normalized)

        assert current == reference, (
            f"Convergence failure after {iterations} iterations!\n"
            f"Input: {payload}\n"
            f"Reference (1x): {reference}\n"
            f"After {iterations}x: {current}"
        )

    @given(payload=json_payloads)
    @settings(max_examples=100, deadline=None)
    def test_helper_function_idempotency(
        self,
        payload: Dict[str, Any],
        schema_ir_with_properties,
        unit_catalog,
        default_options,
    ):
        """
        Test the is_normalization_idempotent helper function.

        This function should return True for all valid payloads.
        """
        result = is_normalization_idempotent(
            payload=payload,
            ir=schema_ir_with_properties,
            catalog=unit_catalog,
            options=default_options,
        )

        assert result is True, (
            f"is_normalization_idempotent returned False for: {payload}"
        )


# =============================================================================
# SPECIFIC VALUE TESTS (Explicit Examples)
# =============================================================================

@pytest.mark.property
class TestNormalizationIdempotencyExplicit:
    """
    Explicit test cases for normalization idempotency.

    These tests cover known edge cases and boundary conditions.
    """

    @pytest.mark.parametrize("payload", [
        {},  # Empty payload
        {"name": "test"},  # Simple string
        {"value": 42},  # Integer
        {"value": 3.14},  # Float
        {"active": True},  # Boolean
        {"data": None},  # Null
        {"nested": {"key": "value"}},  # Nested object
        {"items": [1, 2, 3]},  # Array
        {"mixed": [1, "two", True, None]},  # Mixed array
    ])
    def test_idempotency_explicit_payloads(
        self,
        payload,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """Test idempotency with specific payload types."""
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        result1 = engine.normalize(payload)
        payload2 = _remove_meta_block(result1.normalized)
        result2 = engine.normalize(payload2)

        normalized1 = _remove_meta_block(result1.normalized)
        normalized2 = _remove_meta_block(result2.normalized)

        assert normalized1 == normalized2

    @pytest.mark.parametrize("value,expected_type", [
        ("42", int),  # String to integer
        ("3.14", float),  # String to float
        ("true", bool),  # String to boolean
        ("false", bool),  # String to boolean
    ])
    def test_coerced_values_remain_stable(
        self,
        value,
        expected_type,
        schema_ir_with_properties,
        unit_catalog,
    ):
        """
        Test that once a value is coerced, it remains stable.

        After coercion: "42" -> 42, subsequent normalizations should
        keep the value as 42 (integer), not coerce it again.
        """
        options = ValidationOptions(
            normalize=True,
            coercion_policy=ConfigCoercionPolicy.SAFE,
        )

        engine = NormalizationEngine(
            ir=schema_ir_with_properties,
            catalog=unit_catalog,
            options=options,
        )

        # Use a field that has a type definition
        payload = {"value": value}

        # First normalization
        result1 = engine.normalize(payload)
        normalized1 = _remove_meta_block(result1.normalized)

        # Second normalization
        result2 = engine.normalize(normalized1)
        normalized2 = _remove_meta_block(result2.normalized)

        # Values should be identical
        assert normalized1 == normalized2, (
            f"Coerced value not stable: {normalized1} != {normalized2}"
        )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.property
class TestNormalizationEdgeCases:
    """
    Test normalization idempotency for edge cases.
    """

    def test_empty_payload_idempotency(
        self,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """Empty payloads should be idempotent."""
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        payload = {}
        result1 = engine.normalize(payload)
        result2 = engine.normalize(_remove_meta_block(result1.normalized))

        assert _remove_meta_block(result1.normalized) == _remove_meta_block(result2.normalized)

    def test_deeply_nested_payload_idempotency(
        self,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """Deeply nested payloads should be idempotent."""
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        # Create deeply nested payload
        payload = {"level0": {"level1": {"level2": {"level3": {"value": 42}}}}}

        result1 = engine.normalize(payload)
        result2 = engine.normalize(_remove_meta_block(result1.normalized))

        assert _remove_meta_block(result1.normalized) == _remove_meta_block(result2.normalized)

    def test_large_array_idempotency(
        self,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """Large arrays should be idempotent."""
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        payload = {"values": list(range(100))}

        result1 = engine.normalize(payload)
        result2 = engine.normalize(_remove_meta_block(result1.normalized))

        assert _remove_meta_block(result1.normalized) == _remove_meta_block(result2.normalized)

    @given(
        unicode_text=st.text(
            alphabet=st.characters(
                whitelist_categories=('L', 'N', 'P', 'S', 'Z'),
                blacklist_characters='\x00'
            ),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_unicode_content_idempotency(
        self,
        unicode_text,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """Unicode content should be idempotent."""
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        payload = {"text": unicode_text}

        result1 = engine.normalize(payload)
        result2 = engine.normalize(_remove_meta_block(result1.normalized))

        assert _remove_meta_block(result1.normalized) == _remove_meta_block(result2.normalized)


# =============================================================================
# COERCION POLICY TESTS
# =============================================================================

@pytest.mark.property
class TestNormalizationWithDifferentPolicies:
    """
    Test idempotency holds across different coercion policies.
    """

    @pytest.mark.parametrize("policy", [
        ConfigCoercionPolicy.OFF,
        ConfigCoercionPolicy.SAFE,
        ConfigCoercionPolicy.AGGRESSIVE,
    ])
    @given(payload=json_payloads)
    @settings(max_examples=50, deadline=None)
    def test_idempotency_with_coercion_policies(
        self,
        policy,
        payload,
        minimal_schema_ir,
        unit_catalog,
    ):
        """
        Test that idempotency holds regardless of coercion policy.

        Whether coercion is OFF, SAFE, or AGGRESSIVE, applying
        normalization twice should give the same result.
        """
        options = ValidationOptions(
            normalize=True,
            coercion_policy=policy,
        )

        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=options,
        )

        result1 = engine.normalize(payload)
        result2 = engine.normalize(_remove_meta_block(result1.normalized))

        assert _remove_meta_block(result1.normalized) == _remove_meta_block(result2.normalized)


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestNormalizationDeterminism:
    """
    Test that normalization is deterministic (same input -> same output).

    This is related to but distinct from idempotency.
    """

    @given(payload=json_payloads)
    @settings(max_examples=100, deadline=None)
    def test_deterministic_results(
        self,
        payload,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """
        Test that normalizing the same payload produces identical results.

        Running normalization multiple times with the same input should
        produce byte-identical outputs (except for timestamps).
        """
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        # Normalize the same payload twice
        result1 = engine.normalize(copy.deepcopy(payload))
        result2 = engine.normalize(copy.deepcopy(payload))

        # Compare without meta blocks (which have timestamps)
        normalized1 = _remove_meta_block(result1.normalized)
        normalized2 = _remove_meta_block(result2.normalized)

        assert normalized1 == normalized2, (
            f"Non-deterministic normalization!\n"
            f"Input: {payload}\n"
            f"Result 1: {normalized1}\n"
            f"Result 2: {normalized2}"
        )

    @given(payload=json_payloads)
    @settings(max_examples=50, deadline=None)
    def test_provenance_hash_determinism(
        self,
        payload,
        minimal_schema_ir,
        unit_catalog,
        default_options,
    ):
        """
        Test that provenance hashes are deterministic.

        The same normalized payload should always produce the same
        provenance hash.
        """
        engine = NormalizationEngine(
            ir=minimal_schema_ir,
            catalog=unit_catalog,
            options=default_options,
        )

        result1 = engine.normalize(copy.deepcopy(payload))
        result2 = engine.normalize(copy.deepcopy(payload))

        assert result1.meta.provenance_hash == result2.meta.provenance_hash, (
            f"Non-deterministic provenance hash!\n"
            f"Hash 1: {result1.meta.provenance_hash}\n"
            f"Hash 2: {result2.meta.provenance_hash}"
        )
