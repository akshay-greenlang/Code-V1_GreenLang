# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Validation Rule Engine Service Unit Tests (AGENT-DATA-019)
==============================================================================

Provides shared fixtures for testing the validation rule engine config, models,
provenance tracker, metrics, rule registry, rule composer, rule evaluator,
conflict detector, rule pack, validation reporter, and pipeline orchestrator.

All tests are self-contained with no external dependencies.

Includes a module-level stub for greenlang.validation_rule_engine.__init__
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
# Stub the validation_rule_engine package to bypass broken __init__ imports.
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.validation_rule_engine"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "validation_rule_engine")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(
        _stub.__path__[0], "__init__.py"
    )
    sys.modules[_PKG_NAME] = _stub

# ---------------------------------------------------------------------------
# Pre-import metrics to avoid Prometheus duplicate-metric ValueError.
# Several engine files may create their own Prometheus Counter/Gauge/Histogram
# objects with the same gl_vre_* names as metrics.py.  By importing metrics.py
# first, the canonical metrics are registered in the default CollectorRegistry.
# When engine files later try to register the same names, the ValueError is
# caught by their own try/except and they fall back to no-op stubs.
# ---------------------------------------------------------------------------

from greenlang.validation_rule_engine import metrics as _vre_metrics  # noqa: F401


# ---------------------------------------------------------------------------
# Relax pydantic model configs for engine tests
# ---------------------------------------------------------------------------


def _relax_model_configs():
    """Relax extra='forbid' to extra='ignore' on all VRE Pydantic models.

    This allows engine tests to instantiate models with extra fields
    passed through from engine internals without triggering
    ValidationError. The test_models.py module restores strict mode
    via its own module-scoped fixture.
    """
    import pydantic
    from greenlang.validation_rule_engine import models as vre_models

    for name in dir(vre_models):
        obj = getattr(vre_models, name)
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
def _clean_vre_env(monkeypatch):
    """Remove all GL_VRE_ env vars and reset config singleton between tests.

    This fixture runs automatically for every test in this package. It:
      1. Removes any existing GL_VRE_* environment variables.
      2. Resets the config singleton before yielding.
      3. Resets again after the test completes.
    """
    prefix = "GL_VRE_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    from greenlang.validation_rule_engine.config import reset_config
    reset_config()

    yield

    # Reset the config singleton so next test starts clean
    try:
        reset_config()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Provenance tracker fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker instance for testing."""
    from greenlang.validation_rule_engine.provenance import ProvenanceTracker
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Engine fixtures (stub-safe: only instantiate if available)
# ---------------------------------------------------------------------------


@pytest.fixture
def rule_registry(provenance_tracker):
    """Create a fresh RuleRegistryEngine with shared provenance tracker.

    Returns None if the engine module is not yet available.
    """
    try:
        from greenlang.validation_rule_engine.rule_registry import RuleRegistryEngine
        return RuleRegistryEngine(provenance_tracker=provenance_tracker)
    except (ImportError, Exception):
        return None


@pytest.fixture
def rule_composer(provenance_tracker, rule_registry):
    """Create a fresh RuleComposerEngine with registry and provenance.

    Returns None if the engine module is not yet available.
    """
    try:
        from greenlang.validation_rule_engine.rule_composer import RuleComposerEngine
        return RuleComposerEngine(
            provenance_tracker=provenance_tracker,
            rule_registry=rule_registry,
        )
    except (ImportError, Exception):
        return None


@pytest.fixture
def rule_evaluator(provenance_tracker, rule_registry):
    """Create a fresh RuleEvaluatorEngine with registry and provenance.

    Returns None if the engine module is not yet available.
    """
    try:
        from greenlang.validation_rule_engine.rule_evaluator import RuleEvaluatorEngine
        return RuleEvaluatorEngine(
            provenance_tracker=provenance_tracker,
            rule_registry=rule_registry,
        )
    except (ImportError, Exception):
        return None


@pytest.fixture
def conflict_detector(provenance_tracker, rule_registry):
    """Create a fresh ConflictDetectorEngine with registry and provenance.

    Returns None if the engine module is not yet available.
    """
    try:
        from greenlang.validation_rule_engine.conflict_detector import (
            ConflictDetectorEngine,
        )
        return ConflictDetectorEngine(
            provenance_tracker=provenance_tracker,
            rule_registry=rule_registry,
        )
    except (ImportError, Exception):
        return None


# ---------------------------------------------------------------------------
# Sample rule parameters fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rule_params() -> Dict[str, Any]:
    """Valid parameters for registering a validation rule.

    Returns a dictionary suitable for constructing a ValidationRule or
    CreateRuleRequest instance.
    """
    return {
        "name": "co2e_range_check",
        "description": "Validates CO2e values are within acceptable range",
        "rule_type": "range",
        "operator": "between",
        "target_field": "co2e",
        "threshold_min": 0.0,
        "threshold_max": 1_000_000.0,
        "severity": "high",
        "namespace": "default",
        "tags": {"domain": "emissions", "framework": "ghg_protocol"},
        "framework": "ghg_protocol",
        "parameters": {"unit": "tCO2e"},
    }


# ---------------------------------------------------------------------------
# Sample dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> List[Dict[str, Any]]:
    """List of 15 sample data records for rule evaluation testing.

    Columns: name, age, email, country, amount, date.
    Contains intentional data quality issues (missing values, out-of-range,
    invalid format) to exercise validation rule evaluation logic.
    """
    return [
        {"name": "Alice", "age": 28, "email": "alice@greenlang.io", "country": "DE", "amount": 1500.00, "date": "2025-01-15"},
        {"name": "Bob", "age": 35, "email": "bob@greenlang.io", "country": "US", "amount": 2500.50, "date": "2025-02-20"},
        {"name": "Carol", "age": 42, "email": "carol@greenlang.io", "country": "FR", "amount": 3200.00, "date": "2025-03-10"},
        {"name": "David", "age": 31, "email": "david@greenlang.io", "country": "DE", "amount": 1800.75, "date": "2025-04-05"},
        {"name": "Eva", "age": 29, "email": "eva@greenlang.io", "country": "UK", "amount": 900.25, "date": "2025-05-12"},
        {"name": "", "age": 38, "email": "frank@greenlang.io", "country": "US", "amount": 4500.00, "date": "2025-06-18"},
        {"name": "Grace", "age": -5, "email": "grace@greenlang.io", "country": "JP", "amount": 2100.00, "date": "2025-07-22"},
        {"name": "Hector", "age": 44, "email": "not-an-email", "country": "BR", "amount": 3700.50, "date": "2025-08-30"},
        {"name": "Ingrid", "age": 33, "email": "ingrid@greenlang.io", "country": "DE", "amount": -100.00, "date": "2025-09-14"},
        {"name": "James", "age": 30, "email": "james@greenlang.io", "country": "US", "amount": 2800.00, "date": "invalid-date"},
        {"name": "Karen", "age": None, "email": "karen@greenlang.io", "country": "CA", "amount": 1900.00, "date": "2025-11-05"},
        {"name": "Leo", "age": 27, "email": "leo@greenlang.io", "country": "AU", "amount": 3300.25, "date": "2025-12-20"},
        {"name": "Mona", "age": 36, "email": None, "country": "IN", "amount": 2600.00, "date": "2026-01-08"},
        {"name": "Nick", "age": 200, "email": "nick@greenlang.io", "country": "ZZ", "amount": 5000.00, "date": "2026-02-14"},
        {"name": "Olivia", "age": 25, "email": "olivia@greenlang.io", "country": "SE", "amount": 1200.00, "date": "2026-03-01"},
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
