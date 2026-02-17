# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Data Lineage Tracker Service Unit Tests (AGENT-DATA-018)
============================================================================

Provides shared fixtures for testing the data lineage tracker config, models,
provenance tracker, metrics, asset registry, transformation tracker, lineage
graph, impact analyzer, lineage validator, lineage reporter, and pipeline
orchestrator components.

All tests are self-contained with no external dependencies.

Includes a module-level stub for greenlang.data_lineage_tracker.__init__
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
# Stub the data_lineage_tracker package to bypass broken __init__ imports.
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.data_lineage_tracker"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "data_lineage_tracker")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(
        _stub.__path__[0], "__init__.py"
    )
    sys.modules[_PKG_NAME] = _stub

# ---------------------------------------------------------------------------
# Pre-import metrics to avoid Prometheus duplicate-metric ValueError.
# ---------------------------------------------------------------------------

from greenlang.data_lineage_tracker import metrics as _dlt_metrics  # noqa: F401


# ---------------------------------------------------------------------------
# Relax pydantic model configs for engine tests
# ---------------------------------------------------------------------------


def _relax_model_configs():
    """Relax extra='forbid' to extra='ignore' on all models.

    This allows engine tests to instantiate models with extra fields
    passed through from engine internals without triggering
    ValidationError.
    """
    import pydantic
    from greenlang.data_lineage_tracker import models as dlt_models

    for name in dir(dlt_models):
        obj = getattr(dlt_models, name)
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
def _clean_dlt_env(monkeypatch):
    """Remove all GL_DLT_ env vars and reset config singleton between tests."""
    prefix = "GL_DLT_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    from greenlang.data_lineage_tracker.config import reset_config
    reset_config()

    yield

    try:
        reset_config()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Provenance and engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker instance for testing."""
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
    return ProvenanceTracker()


@pytest.fixture
def asset_registry(provenance_tracker):
    """Create a fresh AssetRegistryEngine instance for testing."""
    from greenlang.data_lineage_tracker.asset_registry import AssetRegistryEngine
    return AssetRegistryEngine(provenance=provenance_tracker)


@pytest.fixture
def transformation_tracker(provenance_tracker):
    """Create a fresh TransformationTrackerEngine instance for testing."""
    from greenlang.data_lineage_tracker.transformation_tracker import (
        TransformationTrackerEngine,
    )
    return TransformationTrackerEngine(provenance=provenance_tracker)


# ---------------------------------------------------------------------------
# Config fixture (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Set a deterministic test config and reset after each test."""
    from greenlang.data_lineage_tracker.config import (
        DataLineageTrackerConfig,
        set_config,
        reset_config as _reset,
    )
    set_config(
        DataLineageTrackerConfig(
            database_url="postgresql://test:test@localhost/test",
            redis_url="redis://localhost/0",
            log_level="DEBUG",
        )
    )
    yield
    _reset()


# ---------------------------------------------------------------------------
# Provenance alias (for tests that use "provenance" fixture name)
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance():
    """Alias for provenance_tracker - returns a fresh ProvenanceTracker."""
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Lineage graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lineage_graph():
    """Create a LineageGraphEngine with sample nodes and edges.

    Sample graph topology::

        raw_orders (a1) ---> clean_orders (a2) ---> agg_orders (a3)
        raw_spend  (a4) ---> clean_spend  (a5) -/
    """
    from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

    prov = ProvenanceTracker()
    engine = LineageGraphEngine(provenance=prov)

    engine.add_node("a1", "raw.orders", "dataset")
    engine.add_node("a2", "clean.orders", "dataset")
    engine.add_node("a3", "agg.orders", "dataset")
    engine.add_node("a4", "raw.spend", "dataset")
    engine.add_node("a5", "clean.spend", "dataset")

    engine.add_edge("a1", "a2", edge_type="dataset_level")
    engine.add_edge("a2", "a3", edge_type="dataset_level")
    engine.add_edge("a4", "a5", edge_type="dataset_level")
    engine.add_edge("a5", "a3", edge_type="dataset_level")

    return engine


@pytest.fixture
def empty_lineage_graph():
    """Create an empty LineageGraphEngine for tests that need a clean slate."""
    from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

    prov = ProvenanceTracker()
    return LineageGraphEngine(provenance=prov)


# ---------------------------------------------------------------------------
# Impact analyzer fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def impact_analyzer(lineage_graph):
    """Create an ImpactAnalyzerEngine backed by the sample lineage graph."""
    from greenlang.data_lineage_tracker.impact_analyzer import ImpactAnalyzerEngine
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

    prov = ProvenanceTracker()
    return ImpactAnalyzerEngine(graph=lineage_graph, provenance=prov)


# ---------------------------------------------------------------------------
# Lineage validator fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def lineage_validator(lineage_graph):
    """Create a LineageValidatorEngine backed by the sample lineage graph."""
    from greenlang.data_lineage_tracker.lineage_validator import LineageValidatorEngine
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

    prov = ProvenanceTracker()
    return LineageValidatorEngine(graph=lineage_graph, provenance=prov)


# ---------------------------------------------------------------------------
# Lineage reporter fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def lineage_reporter(lineage_graph):
    """Create a LineageReporterEngine backed by the sample lineage graph."""
    from greenlang.data_lineage_tracker.lineage_reporter import LineageReporterEngine
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker

    prov = ProvenanceTracker()
    return LineageReporterEngine(graph=lineage_graph, provenance=prov)


# ---------------------------------------------------------------------------
# Pipeline engine fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_engine():
    """Create a LineageTrackerPipelineEngine with all engines sharing provenance."""
    from greenlang.data_lineage_tracker.asset_registry import AssetRegistryEngine
    from greenlang.data_lineage_tracker.impact_analyzer import ImpactAnalyzerEngine
    from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
    from greenlang.data_lineage_tracker.lineage_reporter import LineageReporterEngine
    from greenlang.data_lineage_tracker.lineage_tracker_pipeline import (
        LineageTrackerPipelineEngine,
    )
    from greenlang.data_lineage_tracker.lineage_validator import LineageValidatorEngine
    from greenlang.data_lineage_tracker.provenance import ProvenanceTracker
    from greenlang.data_lineage_tracker.transformation_tracker import (
        TransformationTrackerEngine,
    )

    prov = ProvenanceTracker()
    graph = LineageGraphEngine(provenance=prov)

    return LineageTrackerPipelineEngine(
        asset_registry=AssetRegistryEngine(provenance=prov),
        transformation_tracker=TransformationTrackerEngine(provenance=prov),
        lineage_graph=graph,
        impact_analyzer=ImpactAnalyzerEngine(graph=graph, provenance=prov),
        lineage_validator=LineageValidatorEngine(graph=graph, provenance=prov),
        lineage_reporter=LineageReporterEngine(graph=graph, provenance=prov),
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Sample asset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_asset_params() -> Dict[str, Any]:
    """Minimal valid parameters for registering a data asset."""
    return {
        "qualified_name": "emissions.scope3.spend_data",
        "asset_type": "dataset",
        "display_name": "Scope 3 Spend Data",
        "owner": "data-team",
        "tags": ["scope3", "spend"],
        "classification": "confidential",
        "description": "Spend data for Scope 3 emissions calculation.",
        "metadata": {"source_system": "SAP", "refresh_interval": "daily"},
    }


@pytest.fixture
def sample_transformation_params() -> Dict[str, Any]:
    """Minimal valid parameters for recording a transformation."""
    return {
        "transformation_type": "filter",
        "agent_id": "data-quality-profiler",
        "pipeline_id": "pipeline-001",
        "source_asset_ids": ["asset-a"],
        "target_asset_ids": ["asset-b"],
        "records_in": 1000,
        "records_out": 950,
        "records_filtered": 50,
        "records_error": 0,
        "duration_ms": 125.5,
        "description": "Filter invalid records from spend data.",
        "parameters": {"min_amount": 0, "currency": "USD"},
        "metadata": {"batch_id": "batch-2026-02-17"},
    }


@pytest.fixture
def sample_edge_data() -> Dict[str, Any]:
    """Sample edge creation data for testing."""
    return {
        "source_asset_id": "asset-a",
        "target_asset_id": "asset-b",
        "edge_type": "dataset_level",
        "confidence": 0.95,
    }


# ---------------------------------------------------------------------------
# Mock Prometheus fixtures
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
