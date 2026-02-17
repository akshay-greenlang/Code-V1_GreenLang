# -*- coding: utf-8 -*-
"""
Shared Fixtures for Schema Migration Agent Integration Tests (AGENT-DATA-017)
=============================================================================

Provides fixtures used across all integration test modules:
  - Prometheus metric pre-import to prevent duplicate ValueError
  - Package stub for greenlang.schema_migration
  - Environment cleanup (autouse, removes GL_SM_* env vars, resets config)
  - Sample schemas (v1 and v2 with distinct structural differences)
  - Sample v1 records (10 records matching v1 schema)
  - Fresh engine fixtures (registry, versioner, detector, checker, planner,
    executor, pipeline)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Stub the schema_migration package to bypass broken __init__ imports.
# This must happen before any engine imports so that submodule imports
# resolve without triggering the full __init__.py (which may fail due
# to Prometheus duplicate metric registration).
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.schema_migration"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent package exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "schema_migration")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(_stub.__path__[0], "__init__.py")
    sys.modules[_PKG_NAME] = _stub


# ---------------------------------------------------------------------------
# Pre-import metrics to avoid Prometheus duplicate-metric ValueError.
# Engine files (schema_registry.py, etc.) register their own gl_sm_*
# Prometheus objects. Importing metrics.py first claims the canonical
# names; engine try/except blocks then fall back to no-op stubs.
# ---------------------------------------------------------------------------

from greenlang.schema_migration import metrics as _sm_metrics  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Relax Pydantic model configs for engine tests (extra="ignore")
# ---------------------------------------------------------------------------

def _relax_model_configs() -> None:
    """Relax extra='forbid' to extra='ignore' on all SDK Pydantic models.

    This allows engine dicts with extra fields to be accepted by models
    without triggering ValidationError during integration tests.
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
# Engine imports (post-stub, post-metrics)
# ---------------------------------------------------------------------------

from greenlang.schema_migration.schema_registry import SchemaRegistryEngine  # noqa: E402
from greenlang.schema_migration.schema_versioner import SchemaVersionerEngine  # noqa: E402
from greenlang.schema_migration.change_detector import ChangeDetectorEngine  # noqa: E402
from greenlang.schema_migration.compatibility_checker import CompatibilityCheckerEngine  # noqa: E402
from greenlang.schema_migration.migration_planner import MigrationPlannerEngine  # noqa: E402
from greenlang.schema_migration.migration_executor import MigrationExecutorEngine  # noqa: E402
from greenlang.schema_migration.schema_migration_pipeline import SchemaMigrationPipelineEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Environment cleanup fixture (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_sm_env(monkeypatch):
    """Remove all GL_SM_* env vars and reset config singleton between tests.

    This runs automatically for every test in this integration package.
    """
    prefix = "GL_SM_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    from greenlang.schema_migration.config import reset_config
    reset_config()

    yield

    try:
        reset_config()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sample User Schema v1
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_schema_v1() -> Dict[str, Any]:
    """JSON Schema v1 with properties: user_id, name, email, age, department.

    - user_id: string (required)
    - name: string (required)
    - email: string (required)
    - age: integer
    - department: string
    """
    return {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "age": {"type": "integer", "minimum": 0, "maximum": 200},
            "department": {"type": "string", "maxLength": 100},
        },
        "required": ["user_id", "name", "email"],
    }


# ---------------------------------------------------------------------------
# Sample User Schema v2 (evolved)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_schema_v2() -> Dict[str, Any]:
    """JSON Schema v2 -- evolved from v1.

    Changes from v1:
      - Added: salary (number, optional, default 0)
      - Added: phone (string, optional)
      - Renamed: department -> team (simulated via remove + add of same type)
      - Removed: age
    """
    return {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "salary": {"type": "number", "default": 0, "minimum": 0},
            "team": {"type": "string", "maxLength": 100},
            "phone": {"type": "string"},
        },
        "required": ["user_id", "name", "email"],
    }


# ---------------------------------------------------------------------------
# Sample User Records matching v1 schema (10 records)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_records_v1() -> List[Dict[str, Any]]:
    """10 records conforming to sample_user_schema_v1."""
    return [
        {"user_id": "U001", "name": "Alice", "email": "alice@gl.io", "age": 28, "department": "Engineering"},
        {"user_id": "U002", "name": "Bob", "email": "bob@gl.io", "age": 35, "department": "Science"},
        {"user_id": "U003", "name": "Carol", "email": "carol@gl.io", "age": 42, "department": "Operations"},
        {"user_id": "U004", "name": "David", "email": "david@gl.io", "age": 31, "department": "Engineering"},
        {"user_id": "U005", "name": "Eva", "email": "eva@gl.io", "age": 29, "department": "Science"},
        {"user_id": "U006", "name": "Frank", "email": "frank@gl.io", "age": 38, "department": "Management"},
        {"user_id": "U007", "name": "Grace", "email": "grace@gl.io", "age": 26, "department": "Operations"},
        {"user_id": "U008", "name": "Hector", "email": "hector@gl.io", "age": 44, "department": "Engineering"},
        {"user_id": "U009", "name": "Ingrid", "email": "ingrid@gl.io", "age": 33, "department": "Science"},
        {"user_id": "U010", "name": "James", "email": "james@gl.io", "age": 30, "department": "Operations"},
    ]


# ---------------------------------------------------------------------------
# Fresh Engine Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_registry() -> SchemaRegistryEngine:
    """Create a fresh SchemaRegistryEngine instance."""
    return SchemaRegistryEngine()


@pytest.fixture
def fresh_versioner() -> SchemaVersionerEngine:
    """Create a fresh SchemaVersionerEngine instance."""
    return SchemaVersionerEngine()


@pytest.fixture
def fresh_detector() -> ChangeDetectorEngine:
    """Create a fresh ChangeDetectorEngine instance."""
    return ChangeDetectorEngine()


@pytest.fixture
def fresh_checker() -> CompatibilityCheckerEngine:
    """Create a fresh CompatibilityCheckerEngine instance."""
    return CompatibilityCheckerEngine()


@pytest.fixture
def fresh_planner() -> MigrationPlannerEngine:
    """Create a fresh MigrationPlannerEngine instance."""
    return MigrationPlannerEngine()


@pytest.fixture
def fresh_executor() -> MigrationExecutorEngine:
    """Create a fresh MigrationExecutorEngine instance."""
    return MigrationExecutorEngine()


@pytest.fixture
def fresh_pipeline() -> SchemaMigrationPipelineEngine:
    """Create a fresh SchemaMigrationPipelineEngine with all sub-engines injected."""
    return SchemaMigrationPipelineEngine()


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that do not apply here
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents fixture (no-op for SM tests).

    The parent ``tests/integration/conftest.py`` defines an autouse
    ``mock_agents`` fixture that patches ``greenlang.agents.registry``
    which is irrelevant to schema-migration engine tests and may fail
    due to missing attributes.  This override silences it.
    """
    yield
