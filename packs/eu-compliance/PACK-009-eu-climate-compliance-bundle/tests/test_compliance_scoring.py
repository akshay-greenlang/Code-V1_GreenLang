# -*- coding: utf-8 -*-
"""
Unit tests for BundleComplianceScoringEngine - PACK-009 Engine 7

Tests weighted compliance scoring across CSRD, CBAM, EU Taxonomy, and
EUDR. Validates raw/risk-adjusted/deadline-boosted scoring, maturity
assessment (5 levels), heatmap generation, improvement recommendations,
benchmark comparison, and provenance hashing.

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
_ENGINE_PATH = _PACK_DIR / "engines" / "bundle_compliance_scoring_engine.py"

try:
    _mod = _import_from_path("bundle_compliance_scoring_engine", _ENGINE_PATH)
    BundleComplianceScoringEngine = _mod.BundleComplianceScoringEngine
    ScoringConfig = _mod.ScoringConfig
    BundleScoringResult = _mod.BundleScoringResult
    RegulationInput = _mod.RegulationInput
    ComplianceScore = _mod.ComplianceScore
    MaturityAssessment = _mod.MaturityAssessment
    HeatmapCell = _mod.HeatmapCell
    Recommendation = _mod.Recommendation
    BenchmarkComparison = _mod.BenchmarkComparison
    MaturityLevel = _mod.MaturityLevel
    RiskSeverity = _mod.RiskSeverity
    HeatmapStatus = _mod.HeatmapStatus
    DEFAULT_WEIGHTS = _mod.DEFAULT_WEIGHTS
    MATURITY_DEFINITIONS = _mod.MATURITY_DEFINITIONS
    INDUSTRY_BENCHMARKS = _mod.INDUSTRY_BENCHMARKS
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Skip decorator
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"BundleComplianceScoringEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regulation_input(
    regulation: str,
    items_assessed: int = 100,
    items_compliant: int = 75,
    items_non_compliant: int = 15,
    items_pending: int = 10,
    evidence_count: int = 30,
    risk_flags: List[str] = None,
    near_deadline: bool = False,
    data_quality_score: float = 80.0,
    gaps: List[str] = None,
) -> "RegulationInput":
    """Create a RegulationInput instance with sensible defaults."""
    return RegulationInput(
        regulation=regulation,
        items_assessed=items_assessed,
        items_compliant=items_compliant,
        items_non_compliant=items_non_compliant,
        items_pending=items_pending,
        evidence_count=evidence_count,
        risk_flags=risk_flags or [],
        near_deadline=near_deadline,
        data_quality_score=data_quality_score,
        gaps=gaps or [],
    )


def _make_all_inputs() -> List["RegulationInput"]:
    """Create a list of RegulationInput for all 4 regulations."""
    return [
        _make_regulation_input("CSRD", 200, 170, 20, 10, 55, data_quality_score=90.0),
        _make_regulation_input("CBAM", 100, 70, 20, 10, 35, data_quality_score=75.0),
        _make_regulation_input("EU_TAXONOMY", 150, 90, 40, 20, 25, data_quality_score=70.0),
        _make_regulation_input("EUDR", 80, 40, 25, 15, 15, data_quality_score=60.0),
    ]


def _assert_provenance_hash(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest."""
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64
    assert re.match(r"^[0-9a-f]{64}$", hash_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBundleComplianceScoringEngine:
    """Test suite for BundleComplianceScoringEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated with default and custom config."""
        engine = BundleComplianceScoringEngine()
        assert engine.config is not None
        assert engine.config.industry == "default"
        assert engine.config.maturity_levels == 5
        assert engine.config.near_deadline_boost == 0.10

        custom = ScoringConfig(
            industry="manufacturing",
            near_deadline_boost=0.15,
            risk_multiplier=1.5,
        )
        engine2 = BundleComplianceScoringEngine(custom)
        assert engine2.config.industry == "manufacturing"
        assert engine2.config.near_deadline_boost == 0.15
        assert engine2.config.risk_multiplier == 1.5

    def test_calculate_scores(self):
        """calculate_scores returns a complete BundleScoringResult."""
        engine = BundleComplianceScoringEngine()
        inputs = _make_all_inputs()
        result = engine.calculate_scores(inputs)

        assert isinstance(result, BundleScoringResult)
        assert result.result_id is not None
        assert result.engine_version == "1.0.0"
        assert len(result.per_regulation) == 4
        assert len(result.maturity_profile) == 4
        assert len(result.heatmap) > 0
        assert len(result.benchmarks) == 4
        assert result.overall_score >= 0.0
        assert result.overall_score <= 100.0

    def test_apply_weights(self):
        """apply_weights multiplies raw scores by regulation weights."""
        engine = BundleComplianceScoringEngine()
        scores = {"CSRD": 80.0, "CBAM": 70.0, "EU_TAXONOMY": 60.0, "EUDR": 50.0}
        weighted = engine.apply_weights(scores)

        weights = engine.config.regulation_weights
        for reg, raw in scores.items():
            expected = round(raw * weights.get(reg, 0.0), 4)
            assert weighted[reg] == pytest.approx(expected, rel=1e-6)

    def test_assess_maturity(self):
        """assess_maturity returns a MaturityAssessment for a regulation."""
        engine = BundleComplianceScoringEngine()

        # Level 5: score >= 80, evidence >= 50
        high_input = _make_regulation_input("CSRD", 100, 85, 10, 5, evidence_count=60)
        maturity_high = engine.assess_maturity(high_input)
        assert maturity_high.level == 5
        assert maturity_high.regulation == "CSRD"

        # Level 1: score < 20, evidence < 5
        low_input = _make_regulation_input("EUDR", 100, 10, 80, 10, evidence_count=2)
        maturity_low = engine.assess_maturity(low_input)
        assert maturity_low.level == 1

    def test_risk_adjusted_score(self):
        """Risk flags reduce the raw score by 3% * multiplier each."""
        engine = BundleComplianceScoringEngine()
        raw_score = 80.0
        risk_flags = ["missing_data", "late_filing", "unverified_supplier"]
        adjusted = engine.calculate_risk_adjusted_score(raw_score, risk_flags)
        # 3 flags * 3.0 * 1.0 = 9.0 penalty
        expected = 80.0 - 9.0
        assert adjusted == pytest.approx(expected, abs=0.01)

        # With higher multiplier
        engine2 = BundleComplianceScoringEngine(ScoringConfig(risk_multiplier=2.0))
        adjusted2 = engine2.calculate_risk_adjusted_score(raw_score, risk_flags)
        # 3 flags * 3.0 * 2.0 = 18.0 penalty
        expected2 = 80.0 - 18.0
        assert adjusted2 == pytest.approx(expected2, abs=0.01)

    def test_heatmap_data(self):
        """generate_heatmap_data produces cells with GREEN/YELLOW/ORANGE/RED."""
        engine = BundleComplianceScoringEngine()
        inputs = [_make_regulation_input("CSRD", 100, 90, 5, 5, evidence_count=50, data_quality_score=85.0)]
        heatmap = engine.generate_heatmap_data(inputs)

        assert len(heatmap) == 5  # 5 dimensions per regulation
        dimensions = {cell.dimension for cell in heatmap}
        assert "Compliance Level" in dimensions
        assert "Data Quality" in dimensions
        assert "Evidence Coverage" in dimensions
        assert "Gap Closure" in dimensions
        assert "Risk Exposure" in dimensions

        statuses = {cell.status for cell in heatmap}
        # All should be valid heatmap statuses
        valid = {HeatmapStatus.GREEN.value, HeatmapStatus.YELLOW.value,
                 HeatmapStatus.ORANGE.value, HeatmapStatus.RED.value}
        assert statuses.issubset(valid)

    def test_improvement_recommendations(self):
        """get_improvement_recommendations generates prioritized recommendations."""
        engine = BundleComplianceScoringEngine()
        inputs = [
            _make_regulation_input(
                "CSRD", 100, 45, 40, 15,
                evidence_count=8,
                data_quality_score=40.0,
                near_deadline=True,
                gaps=["Missing Scope 3", "XBRL tagging"],
            ),
        ]
        result = engine.calculate_scores(inputs)
        recommendations = result.recommendations

        assert len(recommendations) > 0
        # Highest priority recommendations should be first
        priorities = [r.priority for r in recommendations]
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        numeric = [priority_order.get(p, 9) for p in priorities]
        assert numeric == sorted(numeric), "Recommendations not sorted by priority"

    def test_compare_to_benchmark(self):
        """compare_to_benchmark compares scores against industry averages."""
        engine = BundleComplianceScoringEngine()
        inputs = _make_all_inputs()
        result = engine.calculate_scores(inputs)
        benchmarks = result.benchmarks

        assert len(benchmarks) == 4
        for bc in benchmarks:
            assert isinstance(bc, BenchmarkComparison)
            assert bc.position in ("ABOVE", "AT", "BELOW")
            assert bc.benchmark_score > 0.0

    def test_default_weights_sum(self):
        """All default weight profiles sum to 1.0."""
        for industry, weights in DEFAULT_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, (
                f"Weights for {industry} sum to {total}, expected 1.0"
            )

    def test_maturity_levels_1_to_5(self):
        """Maturity levels span 1 to 5 for all regulations."""
        engine = BundleComplianceScoringEngine()
        test_cases = [
            (5, 0, 1),   # score=5%, evidence=0 => level 1
            (25, 8, 2),  # score=25%, evidence=8 => level 2
            (45, 20, 3), # score=45%, evidence=20 => level 3
            (65, 35, 4), # score=65%, evidence=35 => level 4
            (85, 55, 5), # score=85%, evidence=55 => level 5
        ]
        for compliant, evidence, expected_level in test_cases:
            inp = _make_regulation_input(
                "CSRD", 100, compliant, 100 - compliant, 0,
                evidence_count=evidence,
            )
            assessment = engine.assess_maturity(inp)
            assert assessment.level == expected_level, (
                f"compliant={compliant}, evidence={evidence}: "
                f"expected level {expected_level}, got {assessment.level}"
            )

    def test_near_deadline_boost(self):
        """A near-deadline regulation receives a score boost."""
        engine = BundleComplianceScoringEngine()
        base_input = _make_regulation_input("CSRD", 100, 75, 20, 5, near_deadline=False)
        deadline_input = _make_regulation_input("CSRD", 100, 75, 20, 5, near_deadline=True)

        result_no_dl = engine.calculate_scores([base_input])
        result_dl = engine.calculate_scores([deadline_input])

        # The deadline-boosted score should be higher
        ws_no_dl = result_no_dl.per_regulation[0].weighted_score
        ws_dl = result_dl.per_regulation[0].weighted_score
        assert ws_dl > ws_no_dl

    def test_result_has_provenance_hash(self):
        """The scoring result carries a valid SHA-256 provenance hash."""
        engine = BundleComplianceScoringEngine()
        inputs = _make_all_inputs()
        result = engine.calculate_scores(inputs)
        _assert_provenance_hash(result.provenance_hash)

    def test_per_regulation_scores(self):
        """Per-regulation ComplianceScore objects have correct fields."""
        engine = BundleComplianceScoringEngine()
        inputs = _make_all_inputs()
        result = engine.calculate_scores(inputs)

        for cs in result.per_regulation:
            assert isinstance(cs, ComplianceScore)
            assert cs.regulation in ("CSRD", "CBAM", "EU_TAXONOMY", "EUDR")
            assert 0.0 <= cs.raw_score <= 100.0
            assert cs.weight > 0.0
            assert 1 <= cs.maturity_level <= 5
            _assert_provenance_hash(cs.provenance_hash)

    def test_overall_score_range(self):
        """Overall score is always in [0, 100]."""
        engine = BundleComplianceScoringEngine()
        # All zero
        zero_inputs = [_make_regulation_input(r, 100, 0, 100, 0) for r in ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]]
        r_zero = engine.calculate_scores(zero_inputs)
        assert 0.0 <= r_zero.overall_score <= 100.0

        # All perfect
        perf_inputs = [_make_regulation_input(r, 100, 100, 0, 0) for r in ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"]]
        r_perf = engine.calculate_scores(perf_inputs)
        assert 0.0 <= r_perf.overall_score <= 100.0
        assert r_perf.overall_score == pytest.approx(100.0, abs=1.0)

    def test_empty_regulation_input(self):
        """Engine handles an empty inputs list without error."""
        engine = BundleComplianceScoringEngine()
        result = engine.calculate_scores([])
        assert isinstance(result, BundleScoringResult)
        assert result.overall_score == 0.0
        assert len(result.per_regulation) == 0
        assert len(result.maturity_profile) == 0
        assert len(result.heatmap) == 0
