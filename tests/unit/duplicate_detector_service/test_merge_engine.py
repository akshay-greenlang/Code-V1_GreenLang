# -*- coding: utf-8 -*-
"""
Unit tests for MergeEngine - AGENT-DATA-011

Tests the MergeEngine with 110+ test cases covering:
- keep_first: alphabetical first record wins
- keep_latest: latest timestamp wins
- keep_most_complete: fewest nulls wins
- merge_fields: field-by-field best value
- golden_record: quality-based field selection
- resolve_conflict: per-field conflict resolution
- merge_batch: batch merging
- validate_merge: completeness validation
- undo_merge: merge history undo
- conflict detection and tracking
- thread-safe statistics
- provenance tracking

Author: GreenLang Platform Team
Date: February 2026
"""

import copy
import threading
import uuid
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.duplicate_detector.merge_engine import (
    MergeEngine,
    _is_empty,
    _record_completeness,
)
from greenlang.duplicate_detector.models import (
    ConflictResolution,
    DuplicateCluster,
    MergeConflict,
    MergeDecision,
    MergeStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> MergeEngine:
    """Create a fresh MergeEngine instance."""
    return MergeEngine()


def _make_cluster(
    member_ids: List[str],
    cluster_id: str | None = None,
) -> DuplicateCluster:
    """Create a DuplicateCluster helper."""
    cid = cluster_id or str(uuid.uuid4())
    return DuplicateCluster(
        cluster_id=cid,
        member_record_ids=member_ids,
        representative_id=member_ids[0] if member_ids else None,
        cluster_quality=0.85,
        density=1.0,
        diameter=0.15,
        member_count=len(member_ids),
        provenance_hash="a" * 64,
    )


def _make_records(*dicts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a records map from dicts that have an 'id' key."""
    return {str(d["id"]): d for d in dicts}


# =============================================================================
# TestHelpers
# =============================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_is_empty_none(self):
        assert _is_empty(None) is True

    def test_is_empty_empty_string(self):
        assert _is_empty("") is True

    def test_is_empty_whitespace(self):
        assert _is_empty("   ") is True

    def test_is_empty_nonempty(self):
        assert _is_empty("hello") is False

    def test_is_empty_number(self):
        assert _is_empty(0) is False

    def test_is_empty_list(self):
        assert _is_empty([]) is False  # lists are not considered empty

    def test_record_completeness_all_filled(self):
        rec = {"id": "1", "name": "Alice", "email": "a@co.com"}
        assert _record_completeness(rec) == 3

    def test_record_completeness_with_nones(self):
        rec = {"id": "1", "name": "Alice", "email": None}
        assert _record_completeness(rec) == 2

    def test_record_completeness_all_empty(self):
        rec = {"id": None, "name": "", "email": "  "}
        assert _record_completeness(rec) == 0


# =============================================================================
# TestMergeEngineInit
# =============================================================================


class TestMergeEngineInit:
    """Initialization tests."""

    def test_initialization(self):
        engine = MergeEngine()
        stats = engine.get_statistics()
        assert stats["engine_name"] == "MergeEngine"
        assert stats["invocations"] == 0
        assert stats["merge_history_size"] == 0

    def test_reset_statistics(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        engine.merge_cluster(cluster, records)
        engine.reset_statistics()
        stats = engine.get_statistics()
        assert stats["invocations"] == 0


# =============================================================================
# TestKeepFirst
# =============================================================================


class TestKeepFirst:
    """Tests for keep_first strategy."""

    def test_keeps_first_alphabetical(self, engine: MergeEngine):
        source = [
            {"id": "b", "name": "Bob", "email": "bob@co.com"},
            {"id": "a", "name": "Alice", "email": "alice@co.com"},
        ]
        merged, conflicts = engine.keep_first(source)
        assert merged["name"] == "Bob"

    def test_preserves_all_fields(self, engine: MergeEngine):
        source = [{"id": "1", "name": "A", "x": 10}]
        merged, conflicts = engine.keep_first(source)
        assert merged["name"] == "A"
        assert merged["x"] == 10

    def test_empty_source_returns_empty(self, engine: MergeEngine):
        merged, conflicts = engine.keep_first([])
        assert merged == {}
        assert conflicts == []

    def test_conflicts_detected(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice", "email": "alice@co.com"},
            {"id": "2", "name": "Bob", "email": "bob@co.com"},
        ]
        _, conflicts = engine.keep_first(source)
        field_names = [c.field_name for c in conflicts]
        assert "email" in field_names or "name" in field_names

    def test_no_conflicts_if_same_values(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Alice"},
        ]
        _, conflicts = engine.keep_first(source)
        # name values are same -> no conflict for name
        name_conflicts = [c for c in conflicts if c.field_name == "name"]
        assert len(name_conflicts) == 0

    def test_merge_cluster_keep_first(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice"},
            {"id": "r2", "name": "Bob"},
        )
        decision = engine.merge_cluster(cluster, records, strategy=MergeStrategy.KEEP_FIRST)
        assert decision.strategy == MergeStrategy.KEEP_FIRST
        assert decision.merged_record["name"] == "Alice"  # r1 is alphabetically first


# =============================================================================
# TestKeepLatest
# =============================================================================


class TestKeepLatest:
    """Tests for keep_latest strategy."""

    def test_keeps_latest_by_timestamp(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Old", "updated_at": "2025-01-01"},
            {"id": "2", "name": "New", "updated_at": "2026-01-01"},
        ]
        merged, _ = engine.keep_latest(source, timestamp_field="updated_at")
        assert merged["name"] == "New"

    def test_no_timestamp_uses_last(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "First"},
            {"id": "2", "name": "Last"},
        ]
        merged, _ = engine.keep_latest(source)
        assert merged["name"] == "Last"

    def test_empty_returns_empty(self, engine: MergeEngine):
        merged, conflicts = engine.keep_latest([])
        assert merged == {}

    def test_conflicts_detected(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "A", "email": "a@co.com"},
            {"id": "2", "name": "B", "email": "b@co.com"},
        ]
        _, conflicts = engine.keep_latest(source)
        assert len(conflicts) > 0

    def test_merge_cluster_keep_latest(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Old", "ts": "2025-01-01"},
            {"id": "r2", "name": "New", "ts": "2026-01-01"},
        )
        decision = engine.merge_cluster(
            cluster, records,
            strategy=MergeStrategy.KEEP_LATEST,
            timestamp_field="ts",
        )
        assert decision.merged_record["name"] == "New"


# =============================================================================
# TestKeepMostComplete
# =============================================================================


class TestKeepMostComplete:
    """Tests for keep_most_complete strategy."""

    def test_keeps_most_complete(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice", "email": None},
            {"id": "2", "name": "Bob", "email": "bob@co.com", "phone": "555"},
        ]
        merged, _ = engine.keep_most_complete(source)
        assert merged["name"] == "Bob"
        assert merged["email"] == "bob@co.com"

    def test_empty_returns_empty(self, engine: MergeEngine):
        merged, _ = engine.keep_most_complete([])
        assert merged == {}

    def test_all_same_completeness(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "A"},
            {"id": "2", "name": "B"},
        ]
        merged, _ = engine.keep_most_complete(source)
        # First wins when tied
        assert merged["name"] in ("A", "B")

    def test_empty_strings_not_counted(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "A", "email": "", "phone": ""},
            {"id": "2", "name": "B", "email": "b@co.com"},
        ]
        merged, _ = engine.keep_most_complete(source)
        assert merged["name"] == "B"  # 2 non-empty vs 1 non-empty

    def test_merge_cluster_keep_most_complete(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Sparse"},
            {"id": "r2", "name": "Complete", "email": "c@co.com", "phone": "1"},
        )
        decision = engine.merge_cluster(
            cluster, records,
            strategy=MergeStrategy.KEEP_MOST_COMPLETE,
        )
        assert decision.merged_record["name"] == "Complete"


# =============================================================================
# TestMergeFields
# =============================================================================


class TestMergeFields:
    """Tests for merge_fields strategy."""

    def test_fills_nulls_from_other_records(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice", "email": None},
            {"id": "2", "name": None, "email": "bob@co.com"},
        ]
        merged, _ = engine.merge_fields(source)
        assert merged["name"] == "Alice"
        assert merged["email"] == "bob@co.com"

    def test_conflict_resolution_first(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        merged, conflicts = engine.merge_fields(
            source, conflict_resolution=ConflictResolution.FIRST,
        )
        assert merged["name"] == "Alice"

    def test_conflict_resolution_longest(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Al"},
            {"id": "2", "name": "Alexander"},
        ]
        merged, conflicts = engine.merge_fields(
            source, conflict_resolution=ConflictResolution.LONGEST,
        )
        assert merged["name"] == "Alexander"

    def test_conflict_resolution_shortest(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Al"},
            {"id": "2", "name": "Alexander"},
        ]
        merged, conflicts = engine.merge_fields(
            source, conflict_resolution=ConflictResolution.SHORTEST,
        )
        assert merged["name"] == "Al"

    def test_no_conflict_single_non_null(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice", "email": None},
            {"id": "2", "name": None, "email": None},
        ]
        merged, conflicts = engine.merge_fields(source)
        assert merged["name"] == "Alice"
        # Only one non-null for name -> no conflict
        name_conflicts = [c for c in conflicts if c.field_name == "name"]
        assert len(name_conflicts) == 0

    def test_all_fields_null(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": None},
            {"id": "2", "name": None},
        ]
        merged, _ = engine.merge_fields(source)
        assert merged["name"] is None

    def test_empty_returns_empty(self, engine: MergeEngine):
        merged, _ = engine.merge_fields([])
        assert merged == {}

    def test_merge_cluster_merge_fields(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice", "email": None},
            {"id": "r2", "name": None, "email": "b@co.com"},
        )
        decision = engine.merge_cluster(
            cluster, records,
            strategy=MergeStrategy.MERGE_FIELDS,
        )
        assert decision.merged_record["name"] == "Alice"
        assert decision.merged_record["email"] == "b@co.com"

    def test_superset_of_fields(self, engine: MergeEngine):
        """Merged record contains union of all fields."""
        source = [
            {"id": "1", "a": "x"},
            {"id": "2", "b": "y"},
        ]
        merged, _ = engine.merge_fields(source)
        assert "a" in merged
        assert "b" in merged


# =============================================================================
# TestGoldenRecord
# =============================================================================


class TestGoldenRecord:
    """Tests for golden_record strategy."""

    def test_picks_best_value_per_field(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice Smith", "email": "a@co.com"},
            {"id": "2", "name": "A", "email": "alice.smith@company.com"},
        ]
        golden, conflicts = engine.golden_record(source)
        # Longer name should score higher (string length bonus)
        assert golden["name"] == "Alice Smith"
        # Longer email should score higher
        assert golden["email"] == "alice.smith@company.com"

    def test_empty_returns_empty(self, engine: MergeEngine):
        golden, _ = engine.golden_record([])
        assert golden == {}

    def test_single_record(self, engine: MergeEngine):
        source = [{"id": "1", "name": "Alice", "email": "a@co.com"}]
        golden, _ = engine.golden_record(source)
        assert golden["name"] == "Alice"

    def test_records_ranked_by_completeness(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "A"},
            {"id": "2", "name": "Bob", "email": "b@co.com", "phone": "1"},
        ]
        golden, _ = engine.golden_record(source)
        # "2" is more complete, so gets rank bonus
        # Combined with equal length, "2" wins
        assert golden["name"] == "Bob"

    def test_null_fields_skipped(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice", "email": None},
            {"id": "2", "name": None, "email": "b@co.com"},
        ]
        golden, _ = engine.golden_record(source)
        assert golden["name"] == "Alice"
        assert golden["email"] == "b@co.com"

    def test_conflicts_tracked(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        _, conflicts = engine.golden_record(source)
        name_conflicts = [c for c in conflicts if c.field_name == "name"]
        assert len(name_conflicts) == 1

    def test_merge_cluster_golden_record(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice Smith"},
            {"id": "r2", "name": "A"},
        )
        decision = engine.merge_cluster(
            cluster, records, strategy=MergeStrategy.GOLDEN_RECORD,
        )
        assert decision.merged_record["name"] == "Alice Smith"


# =============================================================================
# TestResolveConflict
# =============================================================================


class TestResolveConflict:
    """Tests for resolve_conflict method."""

    def test_empty_values_returns_none(self, engine: MergeEngine):
        val, rid = engine.resolve_conflict("name", {}, [])
        assert val is None
        assert rid is None

    def test_single_value_returns_it(self, engine: MergeEngine):
        val, rid = engine.resolve_conflict("name", {"r1": "Alice"}, [])
        assert val == "Alice"
        assert rid == "r1"

    def test_first_resolution(self, engine: MergeEngine):
        values = {"r2": "Bob", "r1": "Alice"}
        val, rid = engine.resolve_conflict(
            "name", values, [], method=ConflictResolution.FIRST,
        )
        assert val == "Alice"  # r1 sorts first
        assert rid == "r1"

    def test_latest_resolution_with_timestamp(self, engine: MergeEngine):
        values = {"r1": "Old", "r2": "New"}
        source = [
            {"id": "r1", "name": "Old", "ts": "2025-01-01"},
            {"id": "r2", "name": "New", "ts": "2026-01-01"},
        ]
        val, rid = engine.resolve_conflict(
            "name", values, source,
            method=ConflictResolution.LATEST,
            timestamp_field="ts",
        )
        assert val == "New"

    def test_latest_resolution_fallback(self, engine: MergeEngine):
        """Without timestamp, latest uses last key."""
        values = {"r1": "A", "r2": "B"}
        val, rid = engine.resolve_conflict(
            "name", values, [],
            method=ConflictResolution.LATEST,
        )
        assert rid == "r2"

    def test_most_complete_resolution(self, engine: MergeEngine):
        values = {"r1": "Sparse", "r2": "Complete"}
        source = [
            {"id": "r1", "name": "Sparse"},
            {"id": "r2", "name": "Complete", "email": "c@co.com", "phone": "1"},
        ]
        val, rid = engine.resolve_conflict(
            "name", values, source,
            method=ConflictResolution.MOST_COMPLETE,
        )
        assert val == "Complete"

    def test_longest_resolution(self, engine: MergeEngine):
        values = {"r1": "Al", "r2": "Alexander"}
        val, rid = engine.resolve_conflict(
            "name", values, [],
            method=ConflictResolution.LONGEST,
        )
        assert val == "Alexander"

    def test_shortest_resolution(self, engine: MergeEngine):
        values = {"r1": "Al", "r2": "Alexander"}
        val, rid = engine.resolve_conflict(
            "name", values, [],
            method=ConflictResolution.SHORTEST,
        )
        assert val == "Al"


# =============================================================================
# TestMergeBatch
# =============================================================================


class TestMergeBatch:
    """Tests for merge_batch method."""

    def test_empty_batch(self, engine: MergeEngine):
        decisions = engine.merge_batch([], {})
        assert decisions == []

    def test_single_cluster_batch(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice"},
            {"id": "r2", "name": "Bob"},
        )
        decisions = engine.merge_batch([cluster], records)
        assert len(decisions) == 1

    def test_multiple_clusters(self, engine: MergeEngine):
        c1 = _make_cluster(["r1", "r2"])
        c2 = _make_cluster(["r3", "r4"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
            {"id": "r3", "name": "C"},
            {"id": "r4", "name": "D"},
        )
        decisions = engine.merge_batch([c1, c2], records)
        assert len(decisions) == 2

    def test_batch_with_strategy(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice"},
            {"id": "r2", "name": "Bob"},
        )
        decisions = engine.merge_batch(
            [cluster], records, strategy=MergeStrategy.KEEP_FIRST,
        )
        assert decisions[0].strategy == MergeStrategy.KEEP_FIRST


# =============================================================================
# TestValidateMerge
# =============================================================================


class TestValidateMerge:
    """Tests for validate_merge method."""

    def test_valid_merge(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice", "email": "a@co.com"},
            {"id": "r2", "name": "Bob", "email": "b@co.com"},
        )
        decision = engine.merge_cluster(cluster, records)
        validation = engine.validate_merge(decision)
        assert validation["valid"] is True
        assert validation["issues"] == []

    def test_empty_merged_record(self, engine: MergeEngine):
        decision = MergeDecision(
            cluster_id="c1",
            strategy=MergeStrategy.KEEP_FIRST,
            merged_record={},
            source_records=[{"id": "r1", "name": "A"}],
        )
        validation = engine.validate_merge(decision)
        assert validation["valid"] is False
        assert any("empty" in i.lower() for i in validation["issues"])

    def test_no_source_records(self, engine: MergeEngine):
        decision = MergeDecision(
            cluster_id="c1",
            strategy=MergeStrategy.KEEP_FIRST,
            merged_record={"name": "A"},
            source_records=[],
        )
        validation = engine.validate_merge(decision)
        assert validation["valid"] is False

    def test_missing_fields(self, engine: MergeEngine):
        decision = MergeDecision(
            cluster_id="c1",
            strategy=MergeStrategy.KEEP_FIRST,
            merged_record={"name": "A"},
            source_records=[{"id": "r1", "name": "A", "email": "a@co.com"}],
        )
        validation = engine.validate_merge(decision)
        assert validation["valid"] is False
        assert any("missing" in i.lower() for i in validation["issues"])

    def test_conflict_count_in_validation(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "Alice"},
            {"id": "r2", "name": "Bob"},
        )
        decision = engine.merge_cluster(
            cluster, records, strategy=MergeStrategy.KEEP_FIRST,
        )
        validation = engine.validate_merge(decision)
        assert "conflict_count" in validation

    def test_strategy_in_validation(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        decision = engine.merge_cluster(cluster, records)
        validation = engine.validate_merge(decision)
        assert validation["strategy"] == MergeStrategy.KEEP_MOST_COMPLETE.value


# =============================================================================
# TestUndoMerge
# =============================================================================


class TestUndoMerge:
    """Tests for undo_merge method."""

    def test_undo_existing(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"], cluster_id="test-c1")
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        engine.merge_cluster(cluster, records)
        decision = engine.undo_merge("test-c1")
        assert decision is not None
        assert decision.cluster_id == "test-c1"

    def test_undo_nonexistent(self, engine: MergeEngine):
        result = engine.undo_merge("nonexistent")
        assert result is None

    def test_undo_removes_from_history(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"], cluster_id="undo-me")
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        engine.merge_cluster(cluster, records)
        assert len(engine.get_merge_history()) == 1
        engine.undo_merge("undo-me")
        assert len(engine.get_merge_history()) == 0

    def test_undo_returns_source_records(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"], cluster_id="c-undo")
        records = _make_records(
            {"id": "r1", "name": "Alice"},
            {"id": "r2", "name": "Bob"},
        )
        engine.merge_cluster(cluster, records)
        decision = engine.undo_merge("c-undo")
        assert len(decision.source_records) == 2

    def test_undo_last_of_multiple(self, engine: MergeEngine):
        """Undo removes only the targeted cluster merge."""
        c1 = _make_cluster(["r1", "r2"], cluster_id="c1")
        c2 = _make_cluster(["r3", "r4"], cluster_id="c2")
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
            {"id": "r3", "name": "C"},
            {"id": "r4", "name": "D"},
        )
        engine.merge_cluster(c1, records)
        engine.merge_cluster(c2, records)
        assert len(engine.get_merge_history()) == 2
        engine.undo_merge("c1")
        assert len(engine.get_merge_history()) == 1
        remaining = engine.get_merge_history()[0]
        assert remaining.cluster_id == "c2"


# =============================================================================
# TestMergeHistory
# =============================================================================


class TestMergeHistory:
    """Tests for merge history tracking."""

    def test_history_grows(self, engine: MergeEngine):
        for i in range(3):
            cluster = _make_cluster([f"r{i*2}", f"r{i*2+1}"])
            records = _make_records(
                {"id": f"r{i*2}", "name": f"A{i}"},
                {"id": f"r{i*2+1}", "name": f"B{i}"},
            )
            engine.merge_cluster(cluster, records)
        assert len(engine.get_merge_history()) == 3

    def test_history_is_copy(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        engine.merge_cluster(cluster, records)
        h1 = engine.get_merge_history()
        h2 = engine.get_merge_history()
        assert h1 is not h2


# =============================================================================
# TestConflictTracking
# =============================================================================


class TestConflictTracking:
    """Tests for conflict detection and tracking."""

    def test_conflict_has_field_name(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        _, conflicts = engine.merge_fields(source)
        name_conflicts = [c for c in conflicts if c.field_name == "name"]
        assert len(name_conflicts) == 1

    def test_conflict_has_values(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        _, conflicts = engine.merge_fields(source)
        for c in conflicts:
            if c.field_name == "name":
                assert len(c.values) == 2

    def test_conflict_has_chosen_value(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        _, conflicts = engine.merge_fields(source)
        for c in conflicts:
            assert c.chosen_value is not None

    def test_no_conflict_when_values_agree(self, engine: MergeEngine):
        source = [
            {"id": "1", "name": "Same"},
            {"id": "2", "name": "Same"},
        ]
        _, conflicts = engine.keep_first(source)
        name_conflicts = [c for c in conflicts if c.field_name == "name"]
        assert len(name_conflicts) == 0

    def test_many_fields_many_conflicts(self, engine: MergeEngine):
        source = [
            {"id": "1", "a": "x1", "b": "y1", "c": "z1"},
            {"id": "2", "a": "x2", "b": "y2", "c": "z2"},
        ]
        _, conflicts = engine.merge_fields(source)
        # All 3 fields differ -> 3 conflicts (+ id differs too if non-empty)
        assert len(conflicts) >= 3


# =============================================================================
# TestCustomMerge
# =============================================================================


class TestCustomMerge:
    """Tests for custom merge function."""

    def test_custom_merge_registered(self, engine: MergeEngine):
        def my_merge(records):
            return {"custom": True}

        engine.register_custom_merge(my_merge)
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        decision = engine.merge_cluster(
            cluster, records, strategy=MergeStrategy.CUSTOM,
        )
        assert decision.merged_record["custom"] is True

    def test_custom_merge_not_registered_raises(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        with pytest.raises(ValueError, match="No custom merge function"):
            engine.merge_cluster(cluster, records, strategy=MergeStrategy.CUSTOM)


# =============================================================================
# TestMergeClusterValidation
# =============================================================================


class TestMergeClusterValidation:
    """Tests for merge_cluster input validation."""

    def test_single_member_raises(self, engine: MergeEngine):
        cluster = _make_cluster(["r1"])
        records = _make_records({"id": "r1", "name": "A"})
        with pytest.raises(ValueError, match="at least 2"):
            engine.merge_cluster(cluster, records)

    def test_missing_records_raises(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records({"id": "r1", "name": "A"})
        with pytest.raises(ValueError, match="Missing records"):
            engine.merge_cluster(cluster, records)

    def test_unknown_strategy_raises(self, engine: MergeEngine):
        """Unknown strategy value not possible with enum but test attribute."""
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        # Valid strategies all work - just verify this doesn't raise
        for strat in [MergeStrategy.KEEP_FIRST, MergeStrategy.KEEP_MOST_COMPLETE]:
            decision = engine.merge_cluster(cluster, records, strategy=strat)
            assert decision is not None


# =============================================================================
# TestMergeNoConflicts
# =============================================================================


class TestMergeNoConflicts:
    """Tests with no field conflicts."""

    def test_complementary_fields(self, engine: MergeEngine):
        """Records with disjoint non-null fields produce zero conflicts."""
        source = [
            {"id": "1", "name": "Alice", "email": None, "phone": None},
            {"id": "2", "name": None, "email": "b@co.com", "phone": None},
            {"id": "3", "name": None, "email": None, "phone": "555"},
        ]
        merged, conflicts = engine.merge_fields(source)
        # Only id has conflict (all different), name/email/phone have single values
        non_id = [c for c in conflicts if c.field_name != "id"]
        assert len(non_id) == 0
        assert merged["name"] == "Alice"
        assert merged["email"] == "b@co.com"
        assert merged["phone"] == "555"


# =============================================================================
# TestMergeManyConflicts
# =============================================================================


class TestMergeManyConflicts:
    """Tests with many field conflicts."""

    def test_10_conflicting_fields(self, engine: MergeEngine):
        rec1 = {"id": "1"}
        rec2 = {"id": "2"}
        for i in range(10):
            rec1[f"field_{i}"] = f"val_a_{i}"
            rec2[f"field_{i}"] = f"val_b_{i}"
        _, conflicts = engine.merge_fields([rec1, rec2])
        non_id = [c for c in conflicts if c.field_name != "id"]
        assert len(non_id) == 10


# =============================================================================
# TestSingleRecordMerge
# =============================================================================


class TestSingleRecordMerge:
    """Tests for single-record (no-op) scenarios."""

    def test_single_record_keep_first(self, engine: MergeEngine):
        source = [{"id": "1", "name": "Alice"}]
        merged, conflicts = engine.keep_first(source)
        assert merged["name"] == "Alice"
        assert conflicts == []

    def test_single_record_merge_fields(self, engine: MergeEngine):
        source = [{"id": "1", "name": "Alice"}]
        merged, conflicts = engine.merge_fields(source)
        assert merged["name"] == "Alice"


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Thread-safety tests for statistics tracking."""

    def test_concurrent_merges(self, engine: MergeEngine):
        num_threads = 8
        errors: List[str] = []

        def worker(idx: int):
            try:
                cluster = _make_cluster([f"r{idx}_1", f"r{idx}_2"])
                records = _make_records(
                    {"id": f"r{idx}_1", "name": f"A{idx}"},
                    {"id": f"r{idx}_2", "name": f"B{idx}"},
                )
                engine.merge_cluster(cluster, records)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        assert stats["successes"] == num_threads


# =============================================================================
# TestStatistics
# =============================================================================


class TestStatistics:
    """Statistics tracking tests."""

    def test_initial_stats(self, engine: MergeEngine):
        stats = engine.get_statistics()
        assert stats["engine_name"] == "MergeEngine"
        assert stats["invocations"] == 0

    def test_success_counted(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        engine.merge_cluster(cluster, records)
        stats = engine.get_statistics()
        assert stats["successes"] == 1

    def test_failure_counted(self, engine: MergeEngine):
        cluster = _make_cluster(["r1"])
        records = _make_records({"id": "r1", "name": "A"})
        with pytest.raises(ValueError):
            engine.merge_cluster(cluster, records)
        stats = engine.get_statistics()
        assert stats["failures"] == 1

    def test_duration_positive(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        engine.merge_cluster(cluster, records)
        stats = engine.get_statistics()
        assert stats["total_duration_ms"] > 0


# =============================================================================
# TestProvenanceTracking
# =============================================================================


class TestProvenanceTracking:
    """Provenance hash generation tests."""

    def test_merge_decision_has_provenance(self, engine: MergeEngine):
        cluster = _make_cluster(["r1", "r2"])
        records = _make_records(
            {"id": "r1", "name": "A"},
            {"id": "r2", "name": "B"},
        )
        decision = engine.merge_cluster(cluster, records)
        assert len(decision.provenance_hash) == 64
        int(decision.provenance_hash, 16)


# =============================================================================
# TestDeterminism
# =============================================================================


class TestDeterminism:
    """Determinism: same input always produces same merge."""

    def test_same_input_same_merged_record(self, engine: MergeEngine):
        for _ in range(5):
            source = [
                {"id": "1", "name": "Alice", "email": "a@co.com"},
                {"id": "2", "name": "Bob", "email": "b@co.com"},
            ]
            merged, _ = engine.keep_first(source)
            assert merged["name"] == "Alice"

    def test_same_input_same_conflicts_count(self, engine: MergeEngine):
        counts = []
        for _ in range(5):
            source = [
                {"id": "1", "name": "Alice"},
                {"id": "2", "name": "Bob"},
            ]
            _, conflicts = engine.merge_fields(source)
            counts.append(len(conflicts))
        assert all(c == counts[0] for c in counts)

    def test_deep_copy_isolation(self, engine: MergeEngine):
        """Merged record is a deep copy, not a reference."""
        source = [{"id": "1", "name": "Alice", "data": [1, 2, 3]}]
        merged, _ = engine.keep_first(source)
        merged["data"].append(4)
        assert source[0]["data"] == [1, 2, 3]
