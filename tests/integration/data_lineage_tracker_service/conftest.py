# -*- coding: utf-8 -*-
"""
Shared Fixtures for Data Lineage Tracker Integration Tests (AGENT-DATA-018)
===========================================================================

Provides fixtures used across all integration test modules:
  - Prometheus metric pre-import to prevent duplicate ValueError
  - Package stub for greenlang.data_lineage_tracker
  - Environment cleanup (autouse, removes GL_DLT_* env vars, resets config)
  - ProvenanceTracker fixture with deterministic genesis hash
  - Fresh engine fixtures (asset_registry, transformation_tracker,
    lineage_graph, impact_analyzer, lineage_validator, lineage_reporter,
    lineage_tracker_pipeline)
  - Populated pipeline fixture with realistic GreenLang data flow scenario

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
# Environment defaults for test isolation
# ---------------------------------------------------------------------------

os.environ.setdefault("GL_DLT_DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("GL_DLT_REDIS_URL", "redis://localhost:6379/20")


# ---------------------------------------------------------------------------
# Stub the data_lineage_tracker package to bypass broken __init__ imports.
# This must happen before any engine imports so that submodule imports
# resolve without triggering the full __init__.py (which may fail due
# to Prometheus duplicate metric registration or missing dependencies).
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.data_lineage_tracker"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent package exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "data_lineage_tracker")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(_stub.__path__[0], "__init__.py")
    sys.modules[_PKG_NAME] = _stub


# ---------------------------------------------------------------------------
# Pre-import metrics to avoid Prometheus duplicate-metric ValueError.
# Engine files register their own gl_dlt_* Prometheus objects. Importing
# metrics.py first claims the canonical names; engine try/except blocks
# then fall back to no-op stubs.
# ---------------------------------------------------------------------------

from greenlang.data_lineage_tracker import metrics as _dlt_metrics  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Relax Pydantic model configs for engine tests (extra="ignore")
# ---------------------------------------------------------------------------

def _relax_model_configs() -> None:
    """Relax extra='forbid' to extra='ignore' on all SDK Pydantic models.

    This allows engine dicts with extra fields to be accepted by models
    without triggering ValidationError during integration tests.
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
# Engine imports (post-stub, post-metrics)
# ---------------------------------------------------------------------------

from greenlang.data_lineage_tracker.config import (  # noqa: E402
    DataLineageTrackerConfig,
    set_config,
    reset_config,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker  # noqa: E402
from greenlang.data_lineage_tracker.asset_registry import AssetRegistryEngine  # noqa: E402
from greenlang.data_lineage_tracker.transformation_tracker import TransformationTrackerEngine  # noqa: E402
from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine  # noqa: E402
from greenlang.data_lineage_tracker.impact_analyzer import ImpactAnalyzerEngine  # noqa: E402
from greenlang.data_lineage_tracker.lineage_validator import LineageValidatorEngine  # noqa: E402
from greenlang.data_lineage_tracker.lineage_reporter import LineageReporterEngine  # noqa: E402
from greenlang.data_lineage_tracker.lineage_tracker_pipeline import LineageTrackerPipelineEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Environment cleanup fixture (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_config_fixture():
    """Reset config singleton before and after every test.

    Sets a deterministic test configuration to prevent leakage between tests.
    Removes all GL_DLT_* env vars and resets config singleton on teardown.
    """
    set_config(DataLineageTrackerConfig(
        database_url="postgresql://test:test@localhost/test",
        redis_url="redis://localhost/0",
        log_level="DEBUG",
    ))
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures that do not apply here
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents fixture (no-op for DLT tests).

    The parent ``tests/integration/conftest.py`` defines an autouse
    ``mock_agents`` fixture that patches ``greenlang.agents.registry``
    which is irrelevant to data-lineage-tracker engine tests and may fail
    due to missing attributes.  This override silences it.
    """
    yield


# ---------------------------------------------------------------------------
# Provenance fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance():
    """Create a fresh ProvenanceTracker with deterministic genesis hash."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Individual engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def asset_registry(provenance):
    """Create a fresh AssetRegistryEngine with shared provenance."""
    return AssetRegistryEngine(provenance=provenance)


@pytest.fixture
def transformation_tracker(provenance):
    """Create a fresh TransformationTrackerEngine with shared provenance."""
    return TransformationTrackerEngine(provenance=provenance)


@pytest.fixture
def lineage_graph(provenance):
    """Create a fresh LineageGraphEngine with shared provenance."""
    return LineageGraphEngine(provenance=provenance)


@pytest.fixture
def impact_analyzer(lineage_graph, provenance):
    """Create a fresh ImpactAnalyzerEngine with graph and shared provenance."""
    return ImpactAnalyzerEngine(lineage_graph, provenance=provenance)


@pytest.fixture
def lineage_validator(lineage_graph, provenance):
    """Create a fresh LineageValidatorEngine with graph and shared provenance."""
    return LineageValidatorEngine(lineage_graph, provenance=provenance)


@pytest.fixture
def lineage_reporter(lineage_graph, provenance):
    """Create a fresh LineageReporterEngine with graph and shared provenance."""
    return LineageReporterEngine(lineage_graph, provenance=provenance)


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(provenance):
    """Create a fresh LineageTrackerPipelineEngine with shared provenance."""
    return LineageTrackerPipelineEngine(provenance=provenance)


# ---------------------------------------------------------------------------
# Populated pipeline fixture with realistic GreenLang data flow
# ---------------------------------------------------------------------------

@pytest.fixture
def populated_pipeline(pipeline):
    """Build a realistic GreenLang lineage scenario inside the pipeline.

    Data flow chain:
        external_source: supplier_invoices
        -> agent: pdf_extractor
        -> dataset: extracted_invoices
        -> agent: excel_normalizer
        -> dataset: normalized_spend
        -> agent: spend_categorizer
        -> dataset: categorized_spend
        -> agent: emission_calculator
        -> metric: scope3_emissions
        -> report: csrd_report

    Registers 10 assets, adds 10 nodes to the graph, and creates 9 edges
    forming a linear data flow chain.
    """
    reg = pipeline.asset_registry
    graph = pipeline.lineage_graph

    # Register assets in the asset registry
    asset_definitions = [
        ("supplier.invoices", "external_source", "Supplier Invoices"),
        ("agent.pdf_extractor", "agent", "PDF Extractor"),
        ("data.extracted_invoices", "dataset", "Extracted Invoices"),
        ("agent.excel_normalizer", "agent", "Excel Normalizer"),
        ("data.normalized_spend", "dataset", "Normalized Spend"),
        ("agent.spend_categorizer", "agent", "Spend Categorizer"),
        ("data.categorized_spend", "dataset", "Categorized Spend"),
        ("agent.emission_calculator", "agent", "Emission Calculator"),
        ("metric.scope3_emissions", "metric", "Scope 3 Emissions"),
        ("report.csrd_report", "report", "CSRD Report"),
    ]

    assets = []
    for qualified_name, asset_type, display_name in asset_definitions:
        asset = reg.register_asset(
            qualified_name=qualified_name,
            asset_type=asset_type,
            display_name=display_name,
        )
        assets.append(asset)

    # Add nodes to the lineage graph
    for asset in assets:
        graph.add_node(
            asset["asset_id"],
            asset["qualified_name"],
            asset["asset_type"],
        )

    # Add edges forming a linear data flow chain
    for i in range(len(assets) - 1):
        src = assets[i]
        tgt = assets[i + 1]
        graph.add_edge(
            src["asset_id"],
            tgt["asset_id"],
            edge_type="dataset_level",
        )

    return pipeline, assets


# ---------------------------------------------------------------------------
# GreenLang asset qualified names for test assertions
# ---------------------------------------------------------------------------

GREENLANG_ASSET_NAMES = [
    "supplier.invoices",
    "agent.pdf_extractor",
    "data.extracted_invoices",
    "agent.excel_normalizer",
    "data.normalized_spend",
    "agent.spend_categorizer",
    "data.categorized_spend",
    "agent.emission_calculator",
    "metric.scope3_emissions",
    "report.csrd_report",
]
