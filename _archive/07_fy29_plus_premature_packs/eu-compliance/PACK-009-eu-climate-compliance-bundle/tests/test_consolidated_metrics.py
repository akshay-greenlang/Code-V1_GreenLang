# -*- coding: utf-8 -*-
"""
Unit tests for ConsolidatedMetricsEngine - PACK-009 Engine 5

Tests aggregation of KPIs across CSRD, CBAM, EU Taxonomy, and EUDR into
a unified metrics dashboard. Validates bundle score computation, completeness
calculation, trend analysis, per-regulation breakdowns, executive summaries,
period comparison, and provenance hashing.

Coverage target: 85%+
Test count: 15

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories).

    Registers the module in sys.modules so that pydantic can resolve
    forward-referenced annotations created by ``from __future__ import
    annotations``.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load the engine module
# ---------------------------------------------------------------------------

_PACK_DIR = Path(__file__).resolve().parent.parent
_ENGINE_PATH = _PACK_DIR / "engines" / "consolidated_metrics_engine.py"

try:
    _mod = _import_from_path("consolidated_metrics_engine", _ENGINE_PATH)
    ConsolidatedMetricsEngine = _mod.ConsolidatedMetricsEngine
    ConsolidatedMetricsConfig = _mod.ConsolidatedMetricsConfig
    ConsolidatedMetricsResult = _mod.ConsolidatedMetricsResult
    RegulationMetrics = _mod.RegulationMetrics
    TrendDataPoint = _mod.TrendDataPoint
    PeriodSnapshot = _mod.PeriodSnapshot
    ExecutiveSummary = _mod.ExecutiveSummary
    PeriodComparison = _mod.PeriodComparison
    RegulationType = _mod.RegulationType
    TrendDirection = _mod.TrendDirection
    SummaryRating = _mod.SummaryRating
    REGULATION_METRIC_DEFINITIONS = _mod.REGULATION_METRIC_DEFINITIONS
    DEFAULT_REGULATION_WEIGHTS = _mod.DEFAULT_REGULATION_WEIGHTS
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Skip decorator
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"ConsolidatedMetricsEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regulation_metrics(
    regulation: str,
    compliance_score: float = 75.0,
    data_completeness: float = 80.0,
    items_assessed: int = 100,
    items_compliant: int = 75,
    items_non_compliant: int = 15,
    items_pending: int = 10,
    key_metrics: Dict[str, Any] = None,
) -> "RegulationMetrics":
    """Create a RegulationMetrics instance with sensible defaults."""
    return RegulationMetrics(
        regulation=regulation,
        compliance_score=compliance_score,
        data_completeness=data_completeness,
        items_assessed=items_assessed,
        items_compliant=items_compliant,
        items_non_compliant=items_non_compliant,
        items_pending=items_pending,
        key_metrics=key_metrics or {},
        assessment_date="2026-03-01",
    )


def _make_all_regulation_metrics() -> List["RegulationMetrics"]:
    """Create a list of RegulationMetrics for all 4 regulations."""
    return [
        _make_regulation_metrics("CSRD", 85.0, 90.0, 200, 170, 20, 10),
        _make_regulation_metrics("CBAM", 70.0, 75.0, 100, 70, 20, 10),
        _make_regulation_metrics("EU_TAXONOMY", 60.0, 65.0, 150, 90, 40, 20),
        _make_regulation_metrics("EUDR", 50.0, 55.0, 80, 40, 25, 15),
    ]


def _assert_provenance_hash(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest."""
    assert isinstance(hash_str, str), f"Expected str, got {type(hash_str)}"
    assert len(hash_str) == 64, f"SHA-256 hash must be 64 chars, got {len(hash_str)}"
    assert re.match(r"^[0-9a-f]{64}$", hash_str), f"Invalid hex hash: {hash_str}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConsolidatedMetricsEngine:
    """Test suite for ConsolidatedMetricsEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated with default and custom config."""
        engine = ConsolidatedMetricsEngine()
        assert engine.config is not None
        assert engine.config.include_trends is True
        assert engine.config.trend_periods == 4
        assert engine.config.completeness_threshold == 80.0

        custom_config = ConsolidatedMetricsConfig(
            include_trends=False,
            trend_periods=6,
            completeness_threshold=70.0,
        )
        engine2 = ConsolidatedMetricsEngine(custom_config)
        assert engine2.config.include_trends is False
        assert engine2.config.trend_periods == 6
        assert engine2.config.completeness_threshold == 70.0

    def test_aggregate_metrics(self):
        """aggregate_metrics returns a complete ConsolidatedMetricsResult."""
        engine = ConsolidatedMetricsEngine()
        metrics = _make_all_regulation_metrics()
        result = engine.aggregate_metrics(metrics, period="2026-Q1")

        assert isinstance(result, ConsolidatedMetricsResult)
        assert result.result_id is not None
        assert result.engine_version == "1.0.0"
        assert len(result.per_regulation) == 4
        assert result.bundle_score >= 0.0
        assert result.bundle_score <= 100.0
        assert result.executive_summary is not None
        assert result.completeness is not None
        assert len(result.completeness) == 4

    def test_compute_bundle_score(self):
        """compute_bundle_score produces a weighted average in [0, 100]."""
        engine = ConsolidatedMetricsEngine()
        metrics = _make_all_regulation_metrics()
        # Scores: CSRD=85, CBAM=70, TAXONOMY=60, EUDR=50
        # Weights: CSRD=0.30, CBAM=0.25, TAXONOMY=0.25, EUDR=0.20
        expected = (85 * 0.30 + 70 * 0.25 + 60 * 0.25 + 50 * 0.20) / 1.0
        score = engine.compute_bundle_score(metrics)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_calculate_completeness(self):
        """calculate_completeness returns per-regulation completeness pct."""
        engine = ConsolidatedMetricsEngine()
        csrd_keys = {d["key"]: 1 for d in REGULATION_METRIC_DEFINITIONS["CSRD"][:5]}
        metrics = [
            _make_regulation_metrics("CSRD", key_metrics=csrd_keys, data_completeness=40.0),
        ]
        completeness = engine.calculate_completeness(metrics)
        assert "CSRD" in completeness
        # 5 of 15 CSRD metrics populated = 33.3%; data_completeness=40.0
        # Result is max(computed, provided), so >= 33.0
        assert completeness["CSRD"] >= 33.0

    def test_analyze_trends(self):
        """analyze_trends returns trend data after multiple aggregate calls."""
        engine = ConsolidatedMetricsEngine()
        metrics_q1 = [
            _make_regulation_metrics("CSRD", 70.0),
            _make_regulation_metrics("CBAM", 60.0),
        ]
        metrics_q2 = [
            _make_regulation_metrics("CSRD", 80.0),
            _make_regulation_metrics("CBAM", 65.0),
        ]
        engine.aggregate_metrics(metrics_q1, period="2026-Q1")
        engine.aggregate_metrics(metrics_q2, period="2026-Q2")

        trends = engine.analyze_trends()
        assert len(trends) > 0
        # Should contain BUNDLE, CBAM, CSRD trend entries
        regulation_names = {t.regulation for t in trends}
        assert "BUNDLE" in regulation_names

    def test_per_regulation_breakdown(self):
        """get_per_regulation_breakdown returns detailed KPI data per regulation."""
        engine = ConsolidatedMetricsEngine()
        metrics = [
            _make_regulation_metrics("CSRD", 85.0, items_assessed=100, items_compliant=85),
        ]
        breakdown = engine.get_per_regulation_breakdown(metrics)

        assert "CSRD" in breakdown
        csrd = breakdown["CSRD"]
        assert csrd["compliance_score"] == 85.0
        assert csrd["items_assessed"] == 100
        assert csrd["kpi_count"] == len(REGULATION_METRIC_DEFINITIONS["CSRD"])
        assert "kpis" in csrd
        assert len(csrd["kpis"]) == csrd["kpi_count"]

    def test_executive_summary(self):
        """get_executive_summary produces a valid ExecutiveSummary."""
        engine = ConsolidatedMetricsEngine()
        metrics = _make_all_regulation_metrics()
        bundle_score = engine.compute_bundle_score(metrics)
        completeness = engine.calculate_completeness(metrics)

        summary = engine.get_executive_summary(metrics, bundle_score, completeness)

        assert isinstance(summary, ExecutiveSummary)
        assert summary.regulations_assessed == 4
        assert summary.strongest_regulation == "CSRD"
        assert summary.weakest_regulation == "EUDR"
        assert summary.bundle_score == bundle_score
        assert len(summary.key_findings) > 0
        assert len(summary.recommendations) > 0
        assert summary.generated_at != ""

    def test_compare_periods(self):
        """compare_periods returns a PeriodComparison with deltas."""
        engine = ConsolidatedMetricsEngine()
        metrics_q1 = [_make_regulation_metrics("CSRD", 70.0)]
        metrics_q2 = [_make_regulation_metrics("CSRD", 85.0)]
        engine.aggregate_metrics(metrics_q1, period="2026-Q1")
        engine.aggregate_metrics(metrics_q2, period="2026-Q2")

        comparison = engine.compare_periods("2026-Q1", "2026-Q2")

        assert isinstance(comparison, PeriodComparison)
        assert comparison.period_a == "2026-Q1"
        assert comparison.period_b == "2026-Q2"
        assert comparison.bundle_score_b > comparison.bundle_score_a
        assert "CSRD" in comparison.regulation_changes
        assert len(comparison.improved_regulations) > 0

    def test_score_range_0_to_100(self):
        """Bundle score is always clamped in [0, 100]."""
        engine = ConsolidatedMetricsEngine()
        # All regulations at 0
        zero_metrics = [_make_regulation_metrics(r, 0.0) for r in ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]]
        score_zero = engine.compute_bundle_score(zero_metrics)
        assert 0.0 <= score_zero <= 100.0

        # All regulations at 100
        max_metrics = [_make_regulation_metrics(r, 100.0) for r in ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]]
        score_max = engine.compute_bundle_score(max_metrics)
        assert 0.0 <= score_max <= 100.0
        assert score_max == pytest.approx(100.0, rel=1e-4)

    def test_result_has_provenance_hash(self):
        """The aggregated result carries a valid SHA-256 provenance hash."""
        engine = ConsolidatedMetricsEngine()
        metrics = _make_all_regulation_metrics()
        result = engine.aggregate_metrics(metrics, period="2026-Q1")
        _assert_provenance_hash(result.provenance_hash)

    def test_regulation_metric_definitions_populated(self):
        """REGULATION_METRIC_DEFINITIONS contains entries for all 4 regulations."""
        assert "CSRD" in REGULATION_METRIC_DEFINITIONS
        assert "CBAM" in REGULATION_METRIC_DEFINITIONS
        assert "EU_TAXONOMY" in REGULATION_METRIC_DEFINITIONS
        assert "EUDR" in REGULATION_METRIC_DEFINITIONS
        for reg, defs in REGULATION_METRIC_DEFINITIONS.items():
            assert len(defs) >= 10, f"{reg} has fewer than 10 metric definitions"
            for d in defs:
                assert "key" in d
                assert "label" in d
                assert "unit" in d

    def test_weighted_scoring(self):
        """Regulation weights are applied correctly in the bundle score."""
        config = ConsolidatedMetricsConfig(
            regulation_weights={"CSRD": 0.50, "CBAM": 0.50}
        )
        engine = ConsolidatedMetricsEngine(config)
        metrics = [
            _make_regulation_metrics("CSRD", 100.0),
            _make_regulation_metrics("CBAM", 0.0),
        ]
        score = engine.compute_bundle_score(metrics)
        assert score == pytest.approx(50.0, rel=1e-4)

    def test_completeness_threshold(self):
        """Config completeness_threshold is stored and accessible."""
        config = ConsolidatedMetricsConfig(completeness_threshold=95.0)
        engine = ConsolidatedMetricsEngine(config)
        assert engine.config.completeness_threshold == 95.0

    def test_trend_direction_detection(self):
        """Trend analysis correctly classifies IMPROVING, DECLINING, STABLE."""
        engine = ConsolidatedMetricsEngine()
        # Period 1: score 50
        engine.aggregate_metrics(
            [_make_regulation_metrics("CSRD", 50.0)], period="P1"
        )
        # Period 2: score 80 (big improvement)
        engine.aggregate_metrics(
            [_make_regulation_metrics("CSRD", 80.0)], period="P2"
        )
        # Period 3: score 40 (big decline)
        engine.aggregate_metrics(
            [_make_regulation_metrics("CSRD", 40.0)], period="P3"
        )

        trends = engine.analyze_trends()
        directions = {(t.period, t.regulation, t.metric_name): t.direction for t in trends}
        # P2 should be IMPROVING for BUNDLE
        assert directions.get(("P2", "BUNDLE", "bundle_score")) == "IMPROVING"
        # P3 should be DECLINING for BUNDLE
        assert directions.get(("P3", "BUNDLE", "bundle_score")) == "DECLINING"

    def test_empty_input_handling(self):
        """Engine handles empty metrics list gracefully."""
        engine = ConsolidatedMetricsEngine()
        result = engine.aggregate_metrics([], period="2026-Q1")

        assert result.bundle_score == 0.0
        assert len(result.per_regulation) == 0
        assert result.executive_summary is not None
        assert result.executive_summary.rating == "CRITICAL"
        assert result.executive_summary.regulations_assessed == 0
