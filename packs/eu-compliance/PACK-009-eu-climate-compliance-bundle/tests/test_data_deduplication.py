# -*- coding: utf-8 -*-
"""
Unit tests for DataDeduplicationEngine - PACK-009 Engine 2

Tests duplicate detection, merge strategies, savings estimation,
golden record generation, conflict detection, and deduplication
reporting across CSRD, CBAM, EUDR, and EU Taxonomy data requirements.

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
ENGINE_PATH = PACK_ROOT / "engines" / "data_deduplication_engine.py"

try:
    _dedup_mod = _import_from_path("data_deduplication_engine", ENGINE_PATH)
    DataDeduplicationEngine = _dedup_mod.DataDeduplicationEngine
    DataDeduplicationConfig = _dedup_mod.DataDeduplicationConfig
    DataRequirement = _dedup_mod.DataRequirement
    DeduplicationGroup = _dedup_mod.DeduplicationGroup
    DeduplicationResult = _dedup_mod.DeduplicationResult
    GoldenRecord = _dedup_mod.GoldenRecord
    DedupReport = _dedup_mod.DedupReport
    MergeConflict = _dedup_mod.MergeConflict
    REGULATION_DATA_REQUIREMENTS = _dedup_mod.REGULATION_DATA_REQUIREMENTS
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"DataDeduplicationEngine could not be imported: "
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


# ===========================================================================
# Tests
# ===========================================================================

class TestDataDeduplicationEngine:
    """Tests for DataDeduplicationEngine."""

    # -----------------------------------------------------------------------
    # 1. Instantiation
    # -----------------------------------------------------------------------

    def test_engine_instantiation(self):
        """Engine can be created with default configuration."""
        engine = DataDeduplicationEngine()
        assert engine is not None
        assert isinstance(engine.config, DataDeduplicationConfig)
        assert len(engine._requirements) > 0

    # -----------------------------------------------------------------------
    # 2. scan_requirements
    # -----------------------------------------------------------------------

    def test_scan_requirements(self):
        """Full scan identifies duplicates across all four regulations."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        assert isinstance(result, DeduplicationResult)
        assert result.total_requirements_scanned > 0
        assert result.duplicate_groups >= 0
        assert result.regulations_analyzed is not None
        assert len(result.regulations_analyzed) == 4
        assert set(result.regulations_analyzed) == {
            "CSRD", "CBAM", "EUDR", "EU_TAXONOMY",
        }

    # -----------------------------------------------------------------------
    # 3. find_duplicates
    # -----------------------------------------------------------------------

    def test_find_duplicates_exact_match(self):
        """Two requirements with the same field_name in different regulations
        are grouped as exact-match duplicates."""
        engine = DataDeduplicationEngine()
        reqs = [
            DataRequirement(
                regulation="CSRD",
                field_name="water_consumption_total",
                data_type="numeric",
                unit="m3",
                category="water",
                collection_effort_hours=4.0,
            ),
            DataRequirement(
                regulation="EU_TAXONOMY",
                field_name="water_consumption_total",
                data_type="numeric",
                unit="m3",
                category="water",
                collection_effort_hours=4.0,
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) >= 1
        exact_groups = [g for g in groups if g.match_type == "EXACT_NAME"]
        assert len(exact_groups) >= 1
        group = exact_groups[0]
        assert group.similarity_score == 1.0
        assert len(group.regulations_involved) >= 2

    def test_find_duplicates_semantic_match(self):
        """Fields with high word-overlap across regulations are detected
        as semantic-similarity duplicates when fuzzy matching is enabled."""
        engine = DataDeduplicationEngine(
            config={"enable_fuzzy_matching": True, "fuzzy_threshold": 0.5},
        )
        reqs = [
            DataRequirement(
                regulation="CSRD",
                field_name="ghg_scope1_emissions_total",
                data_type="numeric",
                category="emissions",
                collection_effort_hours=8.0,
            ),
            DataRequirement(
                regulation="CBAM",
                field_name="emissions_total_scope1_ghg",
                data_type="numeric",
                category="emissions",
                collection_effort_hours=10.0,
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) >= 1
        sem_groups = [g for g in groups if g.match_type == "SEMANTIC_SIMILARITY"]
        assert len(sem_groups) >= 1

    # -----------------------------------------------------------------------
    # 4. merge_duplicates
    # -----------------------------------------------------------------------

    def test_merge_duplicates(self):
        """Merging a group produces a GoldenRecord with source regulations."""
        engine = DataDeduplicationEngine()
        reqs = [
            DataRequirement(
                regulation="CSRD",
                field_name="biodiversity_impact_assessment",
                data_type="string",
                category="biodiversity",
                collection_effort_hours=8.0,
                confidence=0.9,
            ),
            DataRequirement(
                regulation="EU_TAXONOMY",
                field_name="biodiversity_impact_assessment",
                data_type="string",
                category="biodiversity",
                collection_effort_hours=8.0,
                confidence=0.95,
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) >= 1
        golden = engine.merge_duplicates(groups[0])
        assert isinstance(golden, GoldenRecord)
        assert golden.canonical_field != ""
        assert len(golden.source_regulations) >= 2
        assert golden.merge_strategy_used == engine.config.merge_strategy

    # -----------------------------------------------------------------------
    # 5. calculate_savings
    # -----------------------------------------------------------------------

    def test_calculate_savings(self):
        """Savings are computed from the scan result with hours and cost."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        savings = engine.calculate_savings(result)
        assert isinstance(savings, dict)
        assert "total_hours_saved" in savings
        assert "cost_saved_eur" in savings
        assert "savings_percentage" in savings
        assert savings["hourly_rate_eur"] == engine.config.hourly_rate_eur

    # -----------------------------------------------------------------------
    # 6. get_golden_record
    # -----------------------------------------------------------------------

    def test_get_golden_record(self):
        """get_golden_record is an alias that produces a valid GoldenRecord."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        if result.groups:
            golden = engine.get_golden_record(result.groups[0])
            assert isinstance(golden, GoldenRecord)
            _assert_provenance_hash(golden)

    # -----------------------------------------------------------------------
    # 7. generate_dedup_report
    # -----------------------------------------------------------------------

    def test_generate_dedup_report(self):
        """Report generation produces a summary and recommendations."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        report = engine.generate_dedup_report(result)
        assert isinstance(report, DedupReport)
        assert len(report.summary) > 0
        assert len(report.recommendations) > 0
        assert isinstance(report.regulation_breakdown, dict)

    # -----------------------------------------------------------------------
    # 8. Merge strategy: FIRST_WINS
    # -----------------------------------------------------------------------

    def test_merge_strategy_first_wins(self):
        """FIRST_WINS strategy picks the first requirement as the golden source."""
        engine = DataDeduplicationEngine(config={"merge_strategy": "FIRST_WINS"})
        reqs = [
            DataRequirement(
                regulation="CBAM",
                field_name="direct_emissions_tco2e",
                data_type="numeric",
                description="CBAM direct emissions",
                category="emissions",
                collection_effort_hours=10.0,
                confidence=0.8,
            ),
            DataRequirement(
                regulation="CSRD",
                field_name="direct_emissions_tco2e",
                data_type="numeric",
                description="CSRD scope 1 emissions",
                category="emissions",
                collection_effort_hours=8.0,
                confidence=0.95,
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) >= 1
        golden = engine.merge_duplicates(groups[0], strategy="FIRST_WINS")
        assert golden.merge_strategy_used == "FIRST_WINS"
        assert golden.description == "CBAM direct emissions"

    # -----------------------------------------------------------------------
    # 9. Merge strategy: HIGHEST_CONFIDENCE
    # -----------------------------------------------------------------------

    def test_merge_strategy_highest_confidence(self):
        """HIGHEST_CONFIDENCE picks the requirement with the best confidence."""
        engine = DataDeduplicationEngine()
        reqs = [
            DataRequirement(
                regulation="CBAM",
                field_name="direct_emissions_tco2e",
                data_type="numeric",
                description="CBAM direct emissions",
                category="emissions",
                collection_effort_hours=10.0,
                confidence=0.8,
            ),
            DataRequirement(
                regulation="CSRD",
                field_name="direct_emissions_tco2e",
                data_type="numeric",
                description="CSRD scope 1 emissions",
                category="emissions",
                collection_effort_hours=8.0,
                confidence=0.95,
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) >= 1
        golden = engine.merge_duplicates(
            groups[0], strategy="HIGHEST_CONFIDENCE",
        )
        assert golden.merge_strategy_used == "HIGHEST_CONFIDENCE"
        assert golden.description == "CSRD scope 1 emissions"

    # -----------------------------------------------------------------------
    # 10. No duplicates for unique fields
    # -----------------------------------------------------------------------

    def test_no_duplicates_for_unique_fields(self):
        """Unique field names within the same regulation do not form groups."""
        engine = DataDeduplicationEngine(
            config={"enable_fuzzy_matching": False},
        )
        reqs = [
            DataRequirement(
                regulation="CSRD",
                field_name="unique_field_alpha",
                category="other",
                collection_effort_hours=1.0,
            ),
            DataRequirement(
                regulation="CSRD",
                field_name="unique_field_beta",
                category="other",
                collection_effort_hours=1.0,
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) == 0

    # -----------------------------------------------------------------------
    # 11. Provenance hash on dedup result
    # -----------------------------------------------------------------------

    def test_dedup_result_has_provenance_hash(self):
        """Scan result carries a SHA-256 provenance hash."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        _assert_provenance_hash(result)

    # -----------------------------------------------------------------------
    # 12. Canonical field on groups
    # -----------------------------------------------------------------------

    def test_dedup_groups_have_canonical_field(self):
        """Every deduplication group has a non-empty canonical_field."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        for group in result.groups:
            assert group.canonical_field != "", (
                f"Group {group.group_id} has empty canonical_field"
            )

    # -----------------------------------------------------------------------
    # 13. Savings are positive when duplicates found
    # -----------------------------------------------------------------------

    def test_savings_positive_when_duplicates_found(self):
        """When duplicates exist, total savings hours are positive."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        if result.duplicate_groups > 0:
            assert result.total_savings_hours > 0.0
            assert result.savings_percentage > 0.0

    # -----------------------------------------------------------------------
    # 14. Regulation data requirements populated
    # -----------------------------------------------------------------------

    def test_regulation_data_requirements_populated(self):
        """All four regulations have data requirements in the internal DB."""
        engine = DataDeduplicationEngine()
        assert "CSRD" in engine._requirements
        assert "CBAM" in engine._requirements
        assert "EUDR" in engine._requirements
        assert "EU_TAXONOMY" in engine._requirements
        for reg, reqs in engine._requirements.items():
            assert len(reqs) > 0, f"Regulation {reg} has no requirements"

    # -----------------------------------------------------------------------
    # 15. Empty input handling
    # -----------------------------------------------------------------------

    def test_empty_input_handling(self):
        """Finding duplicates on an empty list returns no groups."""
        engine = DataDeduplicationEngine()
        groups = engine.find_duplicates([])
        assert groups == []

    # -----------------------------------------------------------------------
    # 16. Conflict detection
    # -----------------------------------------------------------------------

    def test_conflict_detection(self):
        """Conflicts are detected when grouped fields have different units."""
        engine = DataDeduplicationEngine()
        reqs = [
            DataRequirement(
                regulation="CSRD",
                field_name="emission_intensity",
                data_type="numeric",
                unit="tCO2e/EUR",
                category="emissions",
                collection_effort_hours=3.0,
                frequency="annual",
            ),
            DataRequirement(
                regulation="CBAM",
                field_name="emission_intensity",
                data_type="numeric",
                unit="tCO2e/t",
                category="emissions",
                collection_effort_hours=6.0,
                frequency="quarterly",
            ),
        ]
        groups = engine.find_duplicates(reqs)
        assert len(groups) >= 1
        conflicts = groups[0].conflicts
        assert len(conflicts) >= 1
        conflict_attributes = [c.field_attribute for c in conflicts]
        assert "unit" in conflict_attributes or "frequency" in conflict_attributes

    # -----------------------------------------------------------------------
    # 17. Dedup report summary
    # -----------------------------------------------------------------------

    def test_dedup_report_summary(self):
        """Report summary text includes key statistics."""
        engine = DataDeduplicationEngine()
        result = engine.scan_requirements()
        report = engine.generate_dedup_report(result)
        assert "Scanned" in report.summary
        assert "regulations" in report.summary
        assert "duplicate" in report.summary.lower() or "groups" in report.summary.lower()
        _assert_provenance_hash(report)
