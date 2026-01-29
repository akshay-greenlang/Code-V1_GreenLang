# -*- coding: utf-8 -*-
"""
Pytest Configuration for GL-FOUND-X-002 Property Tests

This module provides shared fixtures and configuration for property-based
tests using Hypothesis.

Fixtures:
    - Schema IR fixtures for various test scenarios
    - Unit catalog fixtures
    - Validation options fixtures
    - Coercion engine fixtures

Hypothesis Configuration:
    - Default settings for property tests
    - Profile definitions for different test scenarios

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Property Tests
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pytest
from hypothesis import settings, Verbosity, Phase, HealthCheck

# Import schema components
from greenlang.schema.compiler.ir import (
    SchemaIR,
    PropertyIR,
    NumericConstraintIR,
    StringConstraintIR,
    ArrayConstraintIR,
    UnitSpecIR,
    CompiledPattern,
)
from greenlang.schema.units.catalog import UnitCatalog
from greenlang.schema.models.config import (
    ValidationOptions,
    ValidationProfile,
    CoercionPolicy,
    UnknownFieldPolicy,
)
from greenlang.schema.normalizer.coercions import CoercionEngine


# =============================================================================
# HYPOTHESIS PROFILES
# =============================================================================

# Register custom Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.generate, Phase.target, Phase.shrink],
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

settings.register_profile(
    "exhaustive",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.generate, Phase.target, Phase.shrink],
    verbosity=Verbosity.verbose,
)

# Load profile from environment or default to 'ci'
settings.load_profile("ci")


# =============================================================================
# SCHEMA IR FIXTURES
# =============================================================================

@pytest.fixture
def minimal_schema_ir() -> SchemaIR:
    """
    Create a minimal SchemaIR for testing basic normalization.

    This schema has no properties, constraints, or requirements.
    Useful for testing normalization behavior on arbitrary payloads.
    """
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
    """
    Create a SchemaIR with common property definitions.

    Properties:
        - /name (string)
        - /value (number)
        - /count (integer)
        - /active (boolean)
        - /energy (number)
    """
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
def schema_ir_with_constraints() -> SchemaIR:
    """
    Create a SchemaIR with various constraints.

    Includes:
        - Required fields
        - Numeric constraints (min/max)
        - String constraints (length)
        - Enum constraints
    """
    return SchemaIR(
        schema_id="test/with_constraints",
        version="1.0.0",
        schema_hash="c" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/name": PropertyIR(path="/name", type="string", required=True),
            "/value": PropertyIR(path="/value", type="number", required=False),
            "/count": PropertyIR(path="/count", type="integer", required=False),
            "/status": PropertyIR(path="/status", type="string", required=False),
            "/description": PropertyIR(path="/description", type="string", required=False),
        },
        required_paths={"/name"},
        numeric_constraints={
            "/value": NumericConstraintIR(
                path="/value",
                minimum=0,
                maximum=1000,
            ),
            "/count": NumericConstraintIR(
                path="/count",
                minimum=1,
                maximum=100,
            ),
        },
        string_constraints={
            "/name": StringConstraintIR(
                path="/name",
                min_length=1,
                max_length=100,
            ),
            "/description": StringConstraintIR(
                path="/description",
                max_length=500,
            ),
        },
        enums={
            "/status": ["pending", "active", "completed", "cancelled"],
        },
    )


@pytest.fixture
def schema_ir_with_units() -> SchemaIR:
    """
    Create a SchemaIR with unit specifications.

    Includes unit specifications for energy and mass fields.
    """
    return SchemaIR(
        schema_id="test/with_units",
        version="1.0.0",
        schema_hash="e" * 64,
        compiled_at=datetime.now(),
        compiler_version="0.1.0",
        properties={
            "/energy_consumption": PropertyIR(
                path="/energy_consumption",
                type="number",
                required=False,
            ),
            "/mass": PropertyIR(
                path="/mass",
                type="number",
                required=False,
            ),
        },
        required_paths=set(),
        unit_specs={
            "/energy_consumption": UnitSpecIR(
                path="/energy_consumption",
                dimension="energy",
                canonical="kWh",
                allowed=["kWh", "MWh", "GJ", "MMBTU"],
            ),
            "/mass": UnitSpecIR(
                path="/mass",
                dimension="mass",
                canonical="kg",
                allowed=["kg", "g", "tonne", "lb"],
            ),
        },
    )


# =============================================================================
# VALIDATION OPTIONS FIXTURES
# =============================================================================

@pytest.fixture
def default_options() -> ValidationOptions:
    """Create default validation options."""
    return ValidationOptions(
        profile=ValidationProfile.STANDARD,
        normalize=True,
        coercion_policy=CoercionPolicy.SAFE,
        unknown_field_policy=UnknownFieldPolicy.WARN,
    )


@pytest.fixture
def strict_options() -> ValidationOptions:
    """Create strict validation options."""
    return ValidationOptions(
        profile=ValidationProfile.STRICT,
        normalize=True,
        coercion_policy=CoercionPolicy.OFF,
        unknown_field_policy=UnknownFieldPolicy.ERROR,
    )


@pytest.fixture
def permissive_options() -> ValidationOptions:
    """Create permissive validation options."""
    return ValidationOptions(
        profile=ValidationProfile.PERMISSIVE,
        normalize=True,
        coercion_policy=CoercionPolicy.AGGRESSIVE,
        unknown_field_policy=UnknownFieldPolicy.IGNORE,
    )


# =============================================================================
# COERCION FIXTURES
# =============================================================================

@pytest.fixture
def safe_engine() -> CoercionEngine:
    """Create a coercion engine with SAFE policy."""
    return CoercionEngine(policy=CoercionPolicy.SAFE)


@pytest.fixture
def aggressive_engine() -> CoercionEngine:
    """Create a coercion engine with AGGRESSIVE policy."""
    return CoercionEngine(policy=CoercionPolicy.AGGRESSIVE)


@pytest.fixture
def off_engine() -> CoercionEngine:
    """Create a coercion engine with coercion disabled."""
    return CoercionEngine(policy=CoercionPolicy.OFF)


# =============================================================================
# UNIT CATALOG FIXTURES
# =============================================================================

@pytest.fixture
def unit_catalog() -> UnitCatalog:
    """Create a unit catalog for testing."""
    return UnitCatalog()


# =============================================================================
# PAYLOAD FIXTURES
# =============================================================================

@pytest.fixture
def valid_payload() -> Dict[str, Any]:
    """Create a valid payload that matches schema_ir_with_constraints."""
    return {
        "name": "test_item",
        "value": 500,
        "count": 10,
        "status": "active",
    }


@pytest.fixture
def invalid_payload_missing_required() -> Dict[str, Any]:
    """Create a payload missing required field."""
    return {
        "value": 500,
        "status": "active",
    }


@pytest.fixture
def invalid_payload_constraint_violations() -> Dict[str, Any]:
    """Create a payload with constraint violations."""
    return {
        "name": "test",
        "value": -100,  # Below minimum
        "count": 200,   # Above maximum
        "status": "invalid_status",  # Invalid enum
    }


@pytest.fixture
def payload_with_coercible_values() -> Dict[str, Any]:
    """Create a payload with values that can be coerced."""
    return {
        "name": "test",
        "value": "500",     # String that can be coerced to number
        "count": "10",      # String that can be coerced to integer
        "active": "true",   # String that can be coerced to boolean
    }


# =============================================================================
# MARKER CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers for property tests."""
    config.addinivalue_line(
        "markers",
        "property: marks tests as property-based tests using Hypothesis"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that may take longer due to many examples"
    )


# =============================================================================
# TEST COLLECTION HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Add property marker to all tests in property/ directory.

    This ensures all property tests are properly marked.
    """
    for item in items:
        if "property" in item.fspath.dirname:
            item.add_marker(pytest.mark.property)
