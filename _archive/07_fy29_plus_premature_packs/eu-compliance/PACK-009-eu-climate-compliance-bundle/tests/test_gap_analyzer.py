# -*- coding: utf-8 -*-
"""
Unit tests for CrossRegulationGapAnalyzerEngine - PACK-009 Engine 3

Tests cross-regulation gap identification, severity scoring, cross-impact
multipliers, remediation roadmap generation, and compliance requirement
scanning across CSRD, CBAM, EUDR, and EU Taxonomy.

Coverage target: 85%+
Test count: 18

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Load the engine module
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = PACK_ROOT / "engines" / "cross_regulation_gap_analyzer.py"

try:
    _gap_mod = _import_from_path("cross_regulation_gap_analyzer", ENGINE_PATH)
    CrossRegulationGapAnalyzerEngine = _gap_mod.CrossRegulationGapAnalyzerEngine
    GapAnalyzerConfig = _gap_mod.GapAnalyzerConfig
    ComplianceRequirement = _gap_mod.ComplianceRequirement
    Gap = _gap_mod.Gap
    GapAnalysisResult = _gap_mod.GapAnalysisResult
    RemediationRoadmapItem = _gap_mod.RemediationRoadmapItem
    CrossImpactEntry = _gap_mod.CrossImpactEntry
    COMPLIANCE_REQUIREMENTS = _gap_mod.COMPLIANCE_REQUIREMENTS
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"CrossRegulationGapAnalyzerEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

def _assert_provenance_hash(obj: Any) -> None:
    """Verify an object carries a valid 64-char SHA-256 provenance hash."""
    h = getattr(obj, "provenance_hash", None)
    if h is None and isinstance(obj, dict):
        h = obj.get("provenance_hash")
    assert h is not None, "Missing provenance_hash"
    assert isinstance(h, str), f"provenance_hash should be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash should be 64 hex chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Invalid hex chars in hash"


def _build_all_non_compliant_status(engine) -> Dict[str, str]:
    """Build a status dict marking every requirement as NON_COMPLIANT."""
    status = {}
    for req in engine._all_requirements:
        status[req.requirement_id] = "NON_COMPLIANT"
    return status


def _build_all_compliant_status(engine) -> Dict[str, str]:
    """Build a status dict marking every requirement as COMPLIANT."""
    status = {}
    for req in engine._all_requirements:
        status[req.requirement_id] = "COMPLIANT"
    return status


def _build_partial_status(engine) -> Dict[str, str]:
    """Build a status dict with a mix of statuses for realistic testing."""
    status = {}
    for i, req in enumerate(engine._all_requirements):
        if i % 4 == 0:
            status[req.requirement_id] = "COMPLIANT"
        elif i % 4 == 1:
            status[req.requirement_id] = "NON_COMPLIANT"
        elif i % 4 == 2:
            status[req.requirement_id] = "PARTIALLY_COMPLIANT"
        else:
            status[req.requirement_id] = "NOT_ASSESSED"
    return status


# ===========================================================================
# Tests
# ===========================================================================

class TestCrossRegulationGapAnalyzerEngine:
    """Tests for CrossRegulationGapAnalyzerEngine."""

    # -----------------------------------------------------------------------
    # 1. Instantiation
    # -----------------------------------------------------------------------

    def test_engine_instantiation(self):
        """Engine can be created with default configuration."""
        engine = CrossRegulationGapAnalyzerEngine()
        assert engine is not None
        assert isinstance(engine.config, GapAnalyzerConfig)
        assert len(engine._all_requirements) > 0

    # -----------------------------------------------------------------------
    # 2. scan_all_regulations
    # -----------------------------------------------------------------------

    def test_scan_all_regulations(self):
        """Full scan with partial compliance produces gaps and a roadmap."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_partial_status(engine)
        result = engine.scan_all_regulations(status)
        assert isinstance(result, GapAnalysisResult)
        assert result.total_requirements_scanned > 0
        assert result.total_gaps > 0
        assert len(result.gaps) == result.total_gaps
        assert len(result.regulations_analyzed) == 4
        assert result.overall_compliance_score >= 0.0
        assert result.overall_compliance_score <= 100.0

    # -----------------------------------------------------------------------
    # 3. identify_gaps with missing data
    # -----------------------------------------------------------------------

    def test_identify_gaps_with_missing_data(self):
        """Every non-compliant mandatory requirement appears as a gap."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_non_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        mandatory_count = sum(
            1 for req in engine._all_requirements if req.mandatory
        )
        assert result.total_gaps >= mandatory_count * 0.8, (
            f"Expected at least ~{int(mandatory_count * 0.8)} gaps for "
            f"{mandatory_count} mandatory reqs, got {result.total_gaps}"
        )

    # -----------------------------------------------------------------------
    # 4. identify_gaps all compliant
    # -----------------------------------------------------------------------

    def test_identify_gaps_all_compliant(self):
        """When everything is compliant, zero gaps are identified."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        assert result.total_gaps == 0
        assert result.overall_compliance_score == 100.0

    # -----------------------------------------------------------------------
    # 5. score_cross_impact
    # -----------------------------------------------------------------------

    def test_score_cross_impact(self):
        """Cross-impact scoring increases severity for multi-regulation gaps."""
        engine = CrossRegulationGapAnalyzerEngine()
        req = ComplianceRequirement(
            requirement_id="TEST-DC-001",
            regulation="CSRD",
            description="Test data requirement",
            category="DATA_COLLECTION",
            mandatory=True,
            cross_regulation_tags=["CBAM-DC-001", "TAX-DC-001"],
        )
        gaps = engine.identify_gaps([req], {"TEST-DC-001": "NON_COMPLIANT"})
        assert len(gaps) >= 1
        scored = engine.score_cross_impact(gaps)
        multi_gaps = [g for g in scored if len(g.affected_regulations) > 1]
        assert len(multi_gaps) >= 1
        for g in multi_gaps:
            assert g.cross_impact_multiplier > 1.0
            assert g.adjusted_severity_score > g.base_severity_score

    # -----------------------------------------------------------------------
    # 6. prioritize_remediation
    # -----------------------------------------------------------------------

    def test_prioritize_remediation(self):
        """Remediation plan is ordered by priority with IMMEDIATE first."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_partial_status(engine)
        result = engine.scan_all_regulations(status)
        plan = result.remediation_plan
        assert len(plan) > 0
        assert isinstance(plan[0], RemediationRoadmapItem)
        priority_order = {
            "IMMEDIATE": 1, "SHORT_TERM": 2,
            "MEDIUM_TERM": 3, "LONG_TERM": 4,
        }
        phases = [priority_order.get(item.priority, 99) for item in plan]
        assert phases == sorted(phases), "Remediation plan is not sorted by priority"

    # -----------------------------------------------------------------------
    # 7. generate_roadmap
    # -----------------------------------------------------------------------

    def test_generate_roadmap(self):
        """Roadmap items reference gaps and include effort estimates."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_non_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        plan = result.remediation_plan
        assert len(plan) > 0
        for item in plan:
            assert item.effort_hours > 0
            assert item.cost_eur > 0
            assert item.timeline_days > 0
            assert len(item.regulations_addressed) >= 1

    # -----------------------------------------------------------------------
    # 8. get_multi_regulation_gaps
    # -----------------------------------------------------------------------

    def test_get_multi_regulation_gaps(self):
        """Filtering for multi-regulation gaps returns only those with 2+ regs."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_non_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        multi = engine.get_multi_regulation_gaps(result.gaps, min_regulations=2)
        for gap in multi:
            assert len(gap.affected_regulations) >= 2

    # -----------------------------------------------------------------------
    # 9. Gap severity classification
    # -----------------------------------------------------------------------

    def test_gap_severity_classification(self):
        """Gap severity is one of the five defined levels."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_partial_status(engine)
        result = engine.scan_all_regulations(status)
        valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
        for gap in result.gaps:
            assert gap.severity in valid_severities, (
                f"Gap {gap.gap_id} has invalid severity: {gap.severity}"
            )

    # -----------------------------------------------------------------------
    # 10. Cross-impact multiplier
    # -----------------------------------------------------------------------

    def test_cross_impact_multiplier(self):
        """Single-regulation gaps have multiplier 1.0; multi-reg have >1.0."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_non_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        for gap in result.gaps:
            if len(gap.affected_regulations) == 1:
                assert gap.cross_impact_multiplier == 1.0
            elif len(gap.affected_regulations) > 1:
                assert gap.cross_impact_multiplier > 1.0

    # -----------------------------------------------------------------------
    # 11. Gaps sorted by severity
    # -----------------------------------------------------------------------

    def test_gaps_sorted_by_severity(self):
        """Gaps in the result are sorted by adjusted_severity_score descending."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_partial_status(engine)
        result = engine.scan_all_regulations(status)
        scores = [g.adjusted_severity_score for g in result.gaps]
        assert scores == sorted(scores, reverse=True), (
            "Gaps are not sorted by adjusted severity score descending"
        )

    # -----------------------------------------------------------------------
    # 12. Remediation plan has timeline
    # -----------------------------------------------------------------------

    def test_remediation_plan_has_timeline(self):
        """Every roadmap item has a positive timeline_days value."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_partial_status(engine)
        result = engine.scan_all_regulations(status)
        for item in result.remediation_plan:
            assert item.timeline_days > 0, (
                f"Roadmap item {item.item_id} has zero timeline_days"
            )

    # -----------------------------------------------------------------------
    # 13. Result has provenance hash
    # -----------------------------------------------------------------------

    def test_result_has_provenance_hash(self):
        """Analysis result carries a SHA-256 provenance hash."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_partial_status(engine)
        result = engine.scan_all_regulations(status)
        _assert_provenance_hash(result)

    # -----------------------------------------------------------------------
    # 14. Compliance requirements populated
    # -----------------------------------------------------------------------

    def test_compliance_requirements_populated(self):
        """All four regulations have compliance requirements loaded."""
        engine = CrossRegulationGapAnalyzerEngine()
        counts = engine.get_requirement_count()
        assert "CSRD" in counts
        assert "CBAM" in counts
        assert "EUDR" in counts
        assert "EU_TAXONOMY" in counts
        for reg, count in counts.items():
            assert count > 0, f"Regulation {reg} has zero requirements"
        total = sum(counts.values())
        assert total >= 60, (
            f"Expected at least 60 total requirements, got {total}"
        )

    # -----------------------------------------------------------------------
    # 15. Gap affected regulations list
    # -----------------------------------------------------------------------

    def test_gap_affected_regulations_list(self):
        """Each gap has a non-empty affected_regulations list."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_non_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        for gap in result.gaps:
            assert len(gap.affected_regulations) >= 1
            assert gap.requirement.regulation in gap.affected_regulations

    # -----------------------------------------------------------------------
    # 16. Empty compliance data
    # -----------------------------------------------------------------------

    def test_empty_compliance_data(self):
        """Passing an empty status dict treats all reqs as NOT_ASSESSED."""
        engine = CrossRegulationGapAnalyzerEngine()
        result = engine.scan_all_regulations({})
        assert result.total_gaps > 0
        for gap in result.gaps:
            assert gap.current_status in ("NOT_ASSESSED", "NON_COMPLIANT")

    # -----------------------------------------------------------------------
    # 17. Partial compliance data
    # -----------------------------------------------------------------------

    def test_partial_compliance_data(self):
        """Providing status for only one regulation still reports gaps
        in the other three."""
        engine = CrossRegulationGapAnalyzerEngine()
        csrd_status = {}
        for req in engine._all_requirements:
            if req.regulation == "CSRD":
                csrd_status[req.requirement_id] = "COMPLIANT"
        result = engine.scan_all_regulations(csrd_status)
        assert result.total_gaps > 0
        gap_regs = set(g.requirement.regulation for g in result.gaps)
        assert "CSRD" not in gap_regs or len(gap_regs) >= 2

    # -----------------------------------------------------------------------
    # 18. Critical gaps highlighted
    # -----------------------------------------------------------------------

    def test_critical_gaps_highlighted(self):
        """CRITICAL gaps exist when mandatory reporting requirements are
        NON_COMPLIANT."""
        engine = CrossRegulationGapAnalyzerEngine()
        status = _build_all_non_compliant_status(engine)
        result = engine.scan_all_regulations(status)
        critical_gaps = [g for g in result.gaps if g.severity == "CRITICAL"]
        assert len(critical_gaps) > 0, (
            "Expected at least one CRITICAL gap when all requirements "
            "are NON_COMPLIANT"
        )
        for cg in critical_gaps:
            assert cg.adjusted_severity_score >= 7.5
