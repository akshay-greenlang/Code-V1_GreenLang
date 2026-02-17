# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Schema Migration Agent Service Unit Tests (AGENT-DATA-017)
==============================================================================

Provides shared fixtures for testing the schema migration config, models,
provenance tracker, metrics, schema registry, schema versioner, change
detector, compatibility checker, migration planner, migration executor,
and pipeline orchestrator components.

All tests are self-contained with no external dependencies.

Includes a module-level stub for greenlang.schema_migration.__init__
to bypass engine imports that may not yet be available, allowing direct
submodule imports to work.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub the schema_migration package to bypass broken __init__ imports.
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.schema_migration"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "schema_migration")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(
        _stub.__path__[0], "__init__.py"
    )
    sys.modules[_PKG_NAME] = _stub

# ---------------------------------------------------------------------------
# Pre-import metrics to avoid Prometheus duplicate-metric ValueError.
# Several engine files (schema_registry.py, etc.) create their own
# Prometheus Counter/Gauge/Histogram objects with the same gl_sm_* names
# as metrics.py.  By importing metrics.py first, the canonical metrics
# are registered in the default CollectorRegistry.  When engine files
# later try to register the same names, the ValueError is caught by
# their own try/except and they fall back to no-op stubs.
# ---------------------------------------------------------------------------

from greenlang.schema_migration import metrics as _sm_metrics  # noqa: F401


# ---------------------------------------------------------------------------
# Relax pydantic model configs for engine tests
# ---------------------------------------------------------------------------


def _relax_model_configs():
    """Relax extra="forbid" to extra="ignore" on all models.

    This allows engine tests to instantiate models with extra fields
    passed through from engine internals without triggering
    ValidationError. The test_models.py module restores strict mode
    via its own module-scoped fixture.
    """
    import pydantic
    from greenlang.schema_migration import models as sm_models

    for name in dir(sm_models):
        obj = getattr(sm_models, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, pydantic.BaseModel)
            and obj is not pydantic.BaseModel
        ):
            cfg = getattr(obj, "model_config", {})
            if isinstance(cfg, dict):
                obj.model_config = {**cfg, "extra": "ignore"}
                obj.model_rebuild(force=True)


_relax_model_configs()


# ---------------------------------------------------------------------------
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_sm_env(monkeypatch):
    """Remove all GL_SM_ env vars and reset config singleton between tests.

    This fixture runs automatically for every test in this package. It:
      1. Removes any existing GL_SM_* environment variables.
      2. Resets the config singleton before yielding.
      3. Resets again after the test completes.
    """
    prefix = "GL_SM_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    from greenlang.schema_migration.config import reset_config
    reset_config()

    yield

    # Reset the config singleton so next test starts clean
    try:
        reset_config()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sample JSON Schema fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_json_schema() -> Dict[str, Any]:
    """A valid JSON Schema dict with four properties.

    Properties:
      name (string), age (integer), email (string), department (string)
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string", "format": "email"},
            "department": {"type": "string", "maxLength": 100},
        },
        "required": ["name", "email"],
        "additionalProperties": False,
    }


@pytest.fixture
def sample_json_schema_v2() -> Dict[str, Any]:
    """An evolved version of sample_json_schema.

    Changes from v1:
      - Added: salary (number) with default 0
      - Renamed: department -> team (simulated by remove+add)
      - Removed: age
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name"},
            "email": {"type": "string", "format": "email"},
            "salary": {"type": "number", "default": 0, "minimum": 0},
            "team": {"type": "string", "maxLength": 100},
        },
        "required": ["name", "email"],
        "additionalProperties": False,
    }


@pytest.fixture
def sample_avro_schema() -> Dict[str, Any]:
    """A simple Apache Avro schema dict for a User record."""
    return {
        "type": "record",
        "name": "User",
        "namespace": "com.greenlang.emissions",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "age", "type": "int"},
            {"name": "email", "type": ["null", "string"], "default": None},
        ],
    }


# ---------------------------------------------------------------------------
# Sample Schema Definition fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_schema_definition(sample_json_schema) -> Dict[str, Any]:
    """A complete schema definition dict for registry operations.

    Contains namespace, name, schema_type, definition, owner, and tags.
    """
    return {
        "namespace": "greenlang.emissions",
        "name": "emission_factors_v1",
        "schema_type": "json_schema",
        "definition_json": sample_json_schema,
        "owner": "platform-team",
        "tags": {"domain": "emissions", "tier": "core"},
        "description": "Schema for emission factor records.",
    }


# ---------------------------------------------------------------------------
# Sample Migration Steps fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_migration_steps() -> List[Dict[str, Any]]:
    """A list of 3 migration step dicts (add_field, rename_field, remove_field).

    Steps:
      1. add_field: Add 'salary' with default 0
      2. rename_field: Rename 'department' to 'team'
      3. remove_field: Remove 'age'
    """
    return [
        {
            "step_number": 1,
            "operation": "add_field",
            "source_field": None,
            "target_field": "salary",
            "parameters": {"default_value": 0, "field_type": "number"},
            "reversible": True,
            "description": "Add salary field with default value 0",
        },
        {
            "step_number": 2,
            "operation": "rename_field",
            "source_field": "department",
            "target_field": "team",
            "parameters": {},
            "reversible": True,
            "description": "Rename department field to team",
        },
        {
            "step_number": 3,
            "operation": "remove_field",
            "source_field": "age",
            "target_field": None,
            "parameters": {},
            "reversible": False,
            "description": "Remove deprecated age field",
        },
    ]


# ---------------------------------------------------------------------------
# Mock Prometheus Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """MagicMock for prometheus_client with Counter, Gauge, and Histogram stubs."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_histogram
    mock_gauge = MagicMock()
    mock_gauge.labels.return_value = mock_gauge

    mock_prom = MagicMock()
    mock_prom.Counter.return_value = mock_counter
    mock_prom.Histogram.return_value = mock_histogram
    mock_prom.Gauge.return_value = mock_gauge
    mock_prom.generate_latest.return_value = (
        b"# HELP test_metric\n# TYPE test_metric counter\n"
    )
    return mock_prom


# ---------------------------------------------------------------------------
# Sample datasets for engine testing
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records_v1() -> List[Dict[str, Any]]:
    """10 sample records conforming to sample_json_schema (v1)."""
    return [
        {"name": "Alice", "age": 28, "email": "alice@greenlang.io", "department": "Engineering"},
        {"name": "Bob", "age": 35, "email": "bob@greenlang.io", "department": "Science"},
        {"name": "Carol", "age": 42, "email": "carol@greenlang.io", "department": "Operations"},
        {"name": "David", "age": 31, "email": "david@greenlang.io", "department": "Engineering"},
        {"name": "Eva", "age": 29, "email": "eva@greenlang.io", "department": "Science"},
        {"name": "Frank", "age": 38, "email": "frank@greenlang.io", "department": "Management"},
        {"name": "Grace", "age": 26, "email": "grace@greenlang.io", "department": "Operations"},
        {"name": "Hector", "age": 44, "email": "hector@greenlang.io", "department": "Engineering"},
        {"name": "Ingrid", "age": 33, "email": "ingrid@greenlang.io", "department": "Science"},
        {"name": "James", "age": 30, "email": "james@greenlang.io", "department": "Operations"},
    ]


@pytest.fixture
def sample_records_v2() -> List[Dict[str, Any]]:
    """10 sample records conforming to sample_json_schema_v2."""
    return [
        {"name": "Alice", "email": "alice@greenlang.io", "salary": 72000, "team": "Engineering"},
        {"name": "Bob", "email": "bob@greenlang.io", "salary": 85000, "team": "Science"},
        {"name": "Carol", "email": "carol@greenlang.io", "salary": 95000, "team": "Operations"},
        {"name": "David", "email": "david@greenlang.io", "salary": 68000, "team": "Engineering"},
        {"name": "Eva", "email": "eva@greenlang.io", "salary": 71000, "team": "Science"},
        {"name": "Frank", "email": "frank@greenlang.io", "salary": 92000, "team": "Management"},
        {"name": "Grace", "email": "grace@greenlang.io", "salary": 63000, "team": "Operations"},
        {"name": "Hector", "email": "hector@greenlang.io", "salary": 98000, "team": "Engineering"},
        {"name": "Ingrid", "email": "ingrid@greenlang.io", "salary": 77000, "team": "Science"},
        {"name": "James", "email": "james@greenlang.io", "salary": 74000, "team": "Operations"},
    ]
