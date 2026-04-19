# -*- coding: utf-8 -*-
"""
Unit tests for MultiRegulationConsistencyEngine - PACK-009 Engine 6

Tests cross-regulation data consistency checking across CSRD, CBAM,
EU Taxonomy, and EUDR. Validates tolerance-based numeric comparison,
exact match for categorical/boolean fields, conflict detection, auto-
resolution, correction propagation, reconciliation reports, and
provenance hashing.

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
_ENGINE_PATH = _PACK_DIR / "engines" / "multi_regulation_consistency_engine.py"

try:
    _mod = _import_from_path("multi_regulation_consistency_engine", _ENGINE_PATH)
    MultiRegulationConsistencyEngine = _mod.MultiRegulationConsistencyEngine
    ConsistencyConfig = _mod.ConsistencyConfig
    ConsistencyResult = _mod.ConsistencyResult
    ConsistencyCheck = _mod.ConsistencyCheck
    DataPoint = _mod.DataPoint
    ConflictResolution = _mod.ConflictResolution
    ReconciliationItem = _mod.ReconciliationItem
    ComparisonMode = _mod.ComparisonMode
    ConsistencyLevel = _mod.ConsistencyLevel
    FieldType = _mod.FieldType
    ResolutionStrategy = _mod.ResolutionStrategy
    FieldCategory = _mod.FieldCategory
    SHARED_DATA_FIELDS = _mod.SHARED_DATA_FIELDS
    _ENGINE_AVAILABLE = True
except Exception as exc:
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Skip decorator
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not _ENGINE_AVAILABLE,
    reason=f"MultiRegulationConsistencyEngine could not be imported: "
           f"{_ENGINE_IMPORT_ERROR if not _ENGINE_AVAILABLE else ''}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_point(
    regulation: str,
    field_name: str,
    value: Any,
    timestamp: str = "2026-03-01T00:00:00",
    confidence: float = 0.95,
    source: str = "test",
) -> "DataPoint":
    """Create a DataPoint instance with sensible defaults."""
    return DataPoint(
        regulation=regulation,
        field_name=field_name,
        value=value,
        timestamp=timestamp,
        confidence=confidence,
        source=source,
    )


def _assert_provenance_hash(hash_str: str) -> None:
    """Assert that a string is a valid SHA-256 hex digest."""
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64
    assert re.match(r"^[0-9a-f]{64}$", hash_str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiRegulationConsistencyEngine:
    """Test suite for MultiRegulationConsistencyEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated with default and custom config."""
        engine = MultiRegulationConsistencyEngine()
        assert engine.config is not None
        assert engine.config.tolerance_pct == 5.0
        assert engine.config.comparison_mode == ComparisonMode.TOLERANT

        custom = ConsistencyConfig(
            tolerance_pct=10.0,
            comparison_mode=ComparisonMode.STRICT,
            auto_resolve_threshold=1.0,
        )
        engine2 = MultiRegulationConsistencyEngine(custom)
        assert engine2.config.tolerance_pct == 10.0
        assert engine2.config.comparison_mode == ComparisonMode.STRICT
        assert engine2.config.auto_resolve_threshold == 1.0

    def test_check_consistency_exact_match(self):
        """Identical numeric values across regulations are flagged CONSISTENT."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10000.0),
            _make_data_point("EU_TAXONOMY", "scope1_total_emissions", 10000.0),
        ]
        result = engine.check_consistency(data_points)

        assert isinstance(result, ConsistencyResult)
        assert result.total_fields_checked == 1
        assert result.consistent_count == 1
        assert result.conflict_count == 0
        assert result.consistency_score == 100.0

    def test_check_consistency_within_tolerance(self):
        """Values within the tolerance band are detected as minor deviation."""
        config = ConsistencyConfig(tolerance_pct=5.0)
        engine = MultiRegulationConsistencyEngine(config)
        # 2% deviation (within 5% tolerance * 1x = 5% MINOR threshold)
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10200.0),
        ]
        result = engine.check_consistency(data_points)
        assert result.total_fields_checked == 1
        # Deviation ~1.98% which is > 0.5% (STRICT threshold) but <= 5% (tolerance)
        # Classified as MINOR_DEVIATION
        check = result.checks[0]
        assert check.status in (
            ConsistencyLevel.CONSISTENT.value,
            ConsistencyLevel.MINOR_DEVIATION.value,
        )

    def test_detect_conflicts(self):
        """detect_conflicts returns only non-consistent checks."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            # Consistent field
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10000.0),
            # Conflicting categorical field
            _make_data_point("CSRD", "ghg_accounting_standard", "GHG Protocol"),
            _make_data_point("CBAM", "ghg_accounting_standard", "ISO 14064"),
            _make_data_point("EU_TAXONOMY", "ghg_accounting_standard", "GHG Protocol"),
        ]
        conflicts = engine.detect_conflicts(data_points)
        assert len(conflicts) >= 1
        field_names = {c.field_name for c in conflicts}
        assert "ghg_accounting_standard" in field_names

    def test_auto_resolve(self):
        """auto_resolve returns a ConflictResolution for small deviations."""
        config = ConsistencyConfig(
            auto_resolve_threshold=3.0,
            default_resolution_strategy=ResolutionStrategy.AVERAGE,
        )
        engine = MultiRegulationConsistencyEngine(config)
        check = ConsistencyCheck(
            field_name="scope1_total_emissions",
            field_category="EMISSIONS",
            field_type="NUMERIC",
            data_points=[
                _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
                _make_data_point("CBAM", "scope1_total_emissions", 10100.0),
            ],
            status=ConsistencyLevel.MINOR_DEVIATION.value,
            max_deviation=100.0,
            deviation_pct=0.99,
        )
        resolution = engine.auto_resolve(check)
        assert resolution is not None
        assert resolution.auto_resolved is True
        assert resolution.strategy_used == ResolutionStrategy.AVERAGE.value
        assert resolution.resolved_value == pytest.approx(10050.0, rel=1e-4)

    def test_propagate_correction(self):
        """propagate_correction updates all data points for the given field."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10100.0),
            _make_data_point("CSRD", "scope2_total_emissions", 5000.0),
        ]
        corrected_value = 10050.0
        updated = engine.propagate_correction(
            "scope1_total_emissions", corrected_value, data_points
        )
        assert len(updated) == 3
        # scope1 fields should be corrected
        scope1_points = [dp for dp in updated if dp.field_name == "scope1_total_emissions"]
        assert all(dp.value == corrected_value for dp in scope1_points)
        # scope2 fields should be unchanged
        scope2_points = [dp for dp in updated if dp.field_name == "scope2_total_emissions"]
        assert scope2_points[0].value == 5000.0

    def test_consistency_score(self):
        """get_consistency_score returns a numeric score 0-100."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10000.0),
        ]
        score = engine.get_consistency_score(data_points)
        assert 0.0 <= score <= 100.0
        assert score == 100.0

    def test_reconciliation_report(self):
        """generate_reconciliation_report produces ReconciliationItem entries."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10500.0),
        ]
        result = engine.check_consistency(data_points)
        assert len(result.reconciliation_items) >= 1
        item = result.reconciliation_items[0]
        assert isinstance(item, ReconciliationItem)
        assert item.field_name == "scope1_total_emissions"
        assert "CSRD" in item.regulations_involved
        assert "CBAM" in item.regulations_involved

    def test_strict_comparison_mode(self):
        """STRICT mode flags even tiny deviations as deviations."""
        config = ConsistencyConfig(
            comparison_mode=ComparisonMode.STRICT,
            tolerance_pct=0.01,
        )
        engine = MultiRegulationConsistencyEngine(config)
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10001.0),
        ]
        result = engine.check_consistency(data_points)
        check = result.checks[0]
        # Even 0.01% deviation should be flagged in STRICT mode with 0.01% tolerance
        assert check.status != ConsistencyLevel.CONSISTENT.value or check.deviation_pct < 0.001

    def test_tolerant_comparison_mode(self):
        """TOLERANT mode allows moderate deviations within tolerance band."""
        config = ConsistencyConfig(
            comparison_mode=ComparisonMode.TOLERANT,
            tolerance_pct=10.0,
        )
        engine = MultiRegulationConsistencyEngine(config)
        # 3% deviation should be within MINOR range for 10% tolerance
        data_points = [
            _make_data_point("CSRD", "total_revenue", 100_000_000.0),
            _make_data_point("EU_TAXONOMY", "total_revenue", 103_000_000.0),
        ]
        result = engine.check_consistency(data_points)
        check = result.checks[0]
        assert check.status in (
            ConsistencyLevel.CONSISTENT.value,
            ConsistencyLevel.MINOR_DEVIATION.value,
        )

    def test_shared_data_fields_populated(self):
        """SHARED_DATA_FIELDS contains entries across multiple categories."""
        assert len(SHARED_DATA_FIELDS) >= 50
        categories_found = set()
        for field_name, field_def in SHARED_DATA_FIELDS.items():
            assert "type" in field_def
            assert "regulations" in field_def
            assert "category" in field_def
            assert len(field_def["regulations"]) >= 2
            categories_found.add(field_def["category"])
        # At least 5 categories expected
        assert len(categories_found) >= 5

    def test_result_has_provenance_hash(self):
        """The consistency result carries a valid SHA-256 provenance hash."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10000.0),
        ]
        result = engine.check_consistency(data_points)
        _assert_provenance_hash(result.provenance_hash)

    def test_conflict_resolution_strategies(self):
        """Different resolution strategies produce different resolved values."""
        # HIGHEST_CONFIDENCE strategy
        config_hc = ConsistencyConfig(
            auto_resolve_threshold=5.0,
            default_resolution_strategy=ResolutionStrategy.HIGHEST_CONFIDENCE,
        )
        engine_hc = MultiRegulationConsistencyEngine(config_hc)
        check = ConsistencyCheck(
            field_name="scope1_total_emissions",
            field_type="NUMERIC",
            data_points=[
                _make_data_point("CSRD", "scope1_total_emissions", 10000.0, confidence=0.9),
                _make_data_point("CBAM", "scope1_total_emissions", 10200.0, confidence=0.95),
            ],
            status=ConsistencyLevel.MINOR_DEVIATION.value,
            deviation_pct=1.98,
        )
        res_hc = engine_hc.auto_resolve(check)
        assert res_hc is not None
        assert res_hc.auto_resolved is True
        assert res_hc.resolved_value == 10200.0  # Higher confidence

        # MOST_RECENT strategy
        config_mr = ConsistencyConfig(
            auto_resolve_threshold=5.0,
            default_resolution_strategy=ResolutionStrategy.MOST_RECENT,
        )
        engine_mr = MultiRegulationConsistencyEngine(config_mr)
        check_mr = ConsistencyCheck(
            field_name="scope1_total_emissions",
            field_type="NUMERIC",
            data_points=[
                _make_data_point("CSRD", "scope1_total_emissions", 10000.0, timestamp="2026-01-01T00:00:00"),
                _make_data_point("CBAM", "scope1_total_emissions", 10200.0, timestamp="2026-03-01T00:00:00"),
            ],
            status=ConsistencyLevel.MINOR_DEVIATION.value,
            deviation_pct=1.98,
        )
        res_mr = engine_mr.auto_resolve(check_mr)
        assert res_mr is not None
        assert res_mr.resolved_value == 10200.0  # Most recent

    def test_empty_data_points(self):
        """Engine handles empty data points list without error."""
        engine = MultiRegulationConsistencyEngine()
        result = engine.check_consistency([])
        assert isinstance(result, ConsistencyResult)
        assert result.total_fields_checked == 0
        assert result.consistency_score == 100.0
        assert result.conflict_count == 0

    def test_mixed_consistency_levels(self):
        """A batch with consistent and conflicting fields returns mixed results."""
        engine = MultiRegulationConsistencyEngine()
        data_points = [
            # Consistent numeric field
            _make_data_point("CSRD", "scope1_total_emissions", 10000.0),
            _make_data_point("CBAM", "scope1_total_emissions", 10000.0),
            # Conflicting boolean field
            _make_data_point("CSRD", "board_sustainability_oversight", True),
            _make_data_point("EU_TAXONOMY", "board_sustainability_oversight", False),
        ]
        result = engine.check_consistency(data_points)
        assert result.total_fields_checked == 2
        assert result.consistent_count >= 1
        assert result.conflict_count >= 1
        assert 0.0 < result.consistency_score < 100.0
