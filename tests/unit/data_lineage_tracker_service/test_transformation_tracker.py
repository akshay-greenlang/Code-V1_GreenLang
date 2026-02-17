# -*- coding: utf-8 -*-
"""
Unit Tests for TransformationTrackerEngine - AGENT-DATA-018 Data Lineage Tracker
================================================================================

Tests all public methods of TransformationTrackerEngine (Engine 2 of 7) with
80+ tests covering recording, retrieval, search, batch operations,
transformation chains, statistics, export, clear, and thread safety.

Test Classes:
    TestRecordTransformation      - 15 tests: recording and validation
    TestGetTransformation         - 3 tests:  retrieval by ID
    TestSearchTransformations     - 10 tests: multi-criteria search
    TestBatchRecord               - 4 tests:  batch recording
    TestTransformationChain       - 8 tests:  chain traversal
    TestTransformationStatistics  - 4 tests:  aggregate statistics
    TestTransformationMisc        - 6 tests:  export, clear, properties, repr
    TestTransformationEdgeCases   - 34+ tests: validation and boundary tests

Total: 84+ tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pytest

from greenlang.data_lineage_tracker.transformation_tracker import (
    TransformationTrackerEngine,
    VALID_TRANSFORMATION_TYPES,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# TestRecordTransformation
# ============================================================================


class TestRecordTransformation:
    """Tests for TransformationTrackerEngine.record_transformation()."""

    def test_record_basic(
        self, transformation_tracker, sample_transformation_params
    ):
        """Record a basic transformation and verify all returned fields."""
        result = transformation_tracker.record_transformation(
            **sample_transformation_params
        )

        assert result is not None
        assert "id" in result
        assert result["transformation_type"] == "filter"
        assert result["agent_id"] == "data-quality-profiler"
        assert result["pipeline_id"] == "pipeline-001"
        assert result["source_asset_ids"] == ["asset-a"]
        assert result["target_asset_ids"] == ["asset-b"]
        assert result["records_in"] == 1000
        assert result["records_out"] == 950
        assert result["records_filtered"] == 50
        assert result["records_error"] == 0
        assert result["duration_ms"] == 125.5
        assert result["description"] == "Filter invalid records from spend data."
        assert result["parameters"]["min_amount"] == 0
        assert result["metadata"]["batch_id"] == "batch-2026-02-17"
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256
        assert "created_at" in result
        assert "updated_at" in result

    @pytest.mark.parametrize(
        "transformation_type", sorted(VALID_TRANSFORMATION_TYPES)
    )
    def test_all_transformation_types(
        self, transformation_tracker, transformation_type
    ):
        """Verify recording succeeds for every valid transformation type."""
        result = transformation_tracker.record_transformation(
            transformation_type=transformation_type,
            agent_id="test-agent",
            pipeline_id="test-pipeline",
            source_asset_ids=["src-1"],
            target_asset_ids=["tgt-1"],
        )
        assert result["transformation_type"] == transformation_type
        assert result["provenance_hash"] != ""

    def test_record_with_metadata(self, transformation_tracker):
        """Record with metadata dict stores it correctly."""
        meta = {"environment": "production", "version": 2}
        result = transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="spend-categorizer",
            pipeline_id="quarterly",
            source_asset_ids=["raw"],
            target_asset_ids=["agg"],
            metadata=meta,
        )
        assert result["metadata"]["environment"] == "production"
        assert result["metadata"]["version"] == 2

    def test_record_with_counts(self, transformation_tracker):
        """Record with all record count fields."""
        result = transformation_tracker.record_transformation(
            transformation_type="deduplicate",
            agent_id="dedup-agent",
            pipeline_id="pipeline-dedup",
            source_asset_ids=["source"],
            target_asset_ids=["target"],
            records_in=10000,
            records_out=9500,
            records_filtered=0,
            records_error=500,
            duration_ms=2500.0,
        )
        assert result["records_in"] == 10000
        assert result["records_out"] == 9500
        assert result["records_filtered"] == 0
        assert result["records_error"] == 500
        assert result["duration_ms"] == 2500.0

    def test_record_invalid_type(self, transformation_tracker):
        """Invalid transformation_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid transformation_type"):
            transformation_tracker.record_transformation(
                transformation_type="nonexistent_type",
                agent_id="agent",
                pipeline_id="pipeline",
                source_asset_ids=["s"],
                target_asset_ids=["t"],
            )

    def test_record_records_provenance(
        self, transformation_tracker, provenance_tracker
    ):
        """Recording must create a provenance chain entry."""
        initial = provenance_tracker.entry_count
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipeline",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        assert provenance_tracker.entry_count == initial + 1

    def test_record_records_metric(self, transformation_tracker):
        """Recording should not raise even when metrics are no-op."""
        result = transformation_tracker.record_transformation(
            transformation_type="validate",
            agent_id="validator",
            pipeline_id="pipeline",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        assert result["transformation_type"] == "validate"

    def test_record_empty_agent_id_raises(self, transformation_tracker):
        """Empty agent_id raises ValueError."""
        with pytest.raises(ValueError, match="agent_id must not be empty"):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="",
                pipeline_id="pipeline",
                source_asset_ids=["s"],
                target_asset_ids=["t"],
            )

    def test_record_empty_pipeline_id_raises(self, transformation_tracker):
        """Empty pipeline_id raises ValueError."""
        with pytest.raises(ValueError, match="pipeline_id must not be empty"):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="agent",
                pipeline_id="",
                source_asset_ids=["s"],
                target_asset_ids=["t"],
            )

    def test_record_empty_source_asset_ids_raises(self, transformation_tracker):
        """Empty source_asset_ids raises ValueError."""
        with pytest.raises(ValueError, match="source_asset_ids must not be empty"):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="agent",
                pipeline_id="pipeline",
                source_asset_ids=[],
                target_asset_ids=["t"],
            )

    def test_record_empty_target_asset_ids_raises(self, transformation_tracker):
        """Empty target_asset_ids raises ValueError."""
        with pytest.raises(ValueError, match="target_asset_ids must not be empty"):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="agent",
                pipeline_id="pipeline",
                source_asset_ids=["s"],
                target_asset_ids=[],
            )

    def test_record_with_execution_id(self, transformation_tracker):
        """execution_id is stored when provided."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipeline",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            execution_id="exec-run-42",
        )
        assert result["execution_id"] == "exec-run-42"

    def test_record_without_execution_id(self, transformation_tracker):
        """execution_id defaults to empty string when not provided."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipeline",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        assert result["execution_id"] == ""

    def test_record_with_parameters(self, transformation_tracker):
        """Parameters dict is stored correctly."""
        params = {"threshold": 0.95, "method": "z-score"}
        result = transformation_tracker.record_transformation(
            transformation_type="validate",
            agent_id="validator",
            pipeline_id="pipeline",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            parameters=params,
        )
        assert result["parameters"]["threshold"] == 0.95
        assert result["parameters"]["method"] == "z-score"


# ============================================================================
# TestGetTransformation
# ============================================================================


class TestGetTransformation:
    """Tests for TransformationTrackerEngine.get_transformation()."""

    def test_get_by_id(self, transformation_tracker):
        """Retrieve a transformation by its unique ID."""
        recorded = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipeline",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        retrieved = transformation_tracker.get_transformation(recorded["id"])
        assert retrieved is not None
        assert retrieved["id"] == recorded["id"]
        assert retrieved["transformation_type"] == "filter"

    def test_get_nonexistent(self, transformation_tracker):
        """Getting a non-existent transformation_id returns None."""
        result = transformation_tracker.get_transformation("nonexistent-id")
        assert result is None

    def test_get_empty_id_returns_none(self, transformation_tracker):
        """Getting with empty string returns None."""
        result = transformation_tracker.get_transformation("")
        assert result is None


# ============================================================================
# TestSearchTransformations
# ============================================================================


class TestSearchTransformations:
    """Tests for TransformationTrackerEngine.search_transformations()."""

    def _record_test_transformations(self, tracker):
        """Register a set of test transformations for search tests."""
        tracker.record_transformation(
            transformation_type="filter",
            agent_id="profiler",
            pipeline_id="pipeline-a",
            source_asset_ids=["s1"],
            target_asset_ids=["t1"],
            records_in=100,
            records_out=90,
        )
        tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="categorizer",
            pipeline_id="pipeline-a",
            source_asset_ids=["s2"],
            target_asset_ids=["t2"],
            records_in=200,
            records_out=50,
        )
        tracker.record_transformation(
            transformation_type="filter",
            agent_id="profiler",
            pipeline_id="pipeline-b",
            source_asset_ids=["s3"],
            target_asset_ids=["t3"],
            records_in=300,
            records_out=280,
        )
        tracker.record_transformation(
            transformation_type="join",
            agent_id="reconciler",
            pipeline_id="pipeline-b",
            source_asset_ids=["s4", "s5"],
            target_asset_ids=["t4"],
            records_in=500,
            records_out=480,
        )

    def test_by_type(self, transformation_tracker):
        """Search by transformation_type returns correct results."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.search_transformations(
            transformation_type="filter"
        )
        assert len(results) == 2
        assert all(r["transformation_type"] == "filter" for r in results)

    def test_by_agent(self, transformation_tracker):
        """Search by agent_id returns correct results."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.search_transformations(
            agent_id="profiler"
        )
        assert len(results) == 2
        assert all(r["agent_id"] == "profiler" for r in results)

    def test_by_pipeline(self, transformation_tracker):
        """Search by pipeline_id returns correct results."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.search_transformations(
            pipeline_id="pipeline-a"
        )
        assert len(results) == 2
        assert all(r["pipeline_id"] == "pipeline-a" for r in results)

    def test_by_time_range(self, transformation_tracker):
        """Search by time range returns results within bounds."""
        before = datetime.now(timezone.utc).isoformat()
        self._record_test_transformations(transformation_tracker)
        after = datetime.now(timezone.utc).isoformat()

        results = transformation_tracker.search_transformations(
            start_time=before, end_time=after
        )
        assert len(results) == 4

    def test_pagination(self, transformation_tracker):
        """Search with limit and offset for pagination."""
        self._record_test_transformations(transformation_tracker)
        page1 = transformation_tracker.search_transformations(limit=2, offset=0)
        page2 = transformation_tracker.search_transformations(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        page1_ids = {r["id"] for r in page1}
        page2_ids = {r["id"] for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_combined_filters(self, transformation_tracker):
        """Search with multiple combined filters (AND logic)."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.search_transformations(
            transformation_type="filter",
            agent_id="profiler",
        )
        assert len(results) == 2

        results = transformation_tracker.search_transformations(
            transformation_type="filter",
            pipeline_id="pipeline-b",
        )
        assert len(results) == 1

    def test_no_filters_returns_all(self, transformation_tracker):
        """Search with no filters returns all transformations."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.search_transformations()
        assert len(results) == 4

    def test_search_empty_results(self, transformation_tracker):
        """Search with non-matching filter returns empty list."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.search_transformations(
            agent_id="nonexistent-agent"
        )
        assert results == []

    def test_search_by_type_dedicated_method(self, transformation_tracker):
        """get_transformations_by_type returns matching results."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.get_transformations_by_type("join")
        assert len(results) == 1
        assert results[0]["transformation_type"] == "join"

    def test_search_by_agent_dedicated_method(self, transformation_tracker):
        """get_transformations_by_agent returns matching results."""
        self._record_test_transformations(transformation_tracker)
        results = transformation_tracker.get_transformations_by_agent("reconciler")
        assert len(results) == 1
        assert results[0]["agent_id"] == "reconciler"


# ============================================================================
# TestBatchRecord
# ============================================================================


class TestBatchRecord:
    """Tests for TransformationTrackerEngine.batch_record()."""

    def test_batch_basic(self, transformation_tracker):
        """Batch record multiple transformations successfully."""
        specs = [
            {
                "transformation_type": "filter",
                "agent_id": "agent-1",
                "pipeline_id": "pipe-1",
                "source_asset_ids": ["s1"],
                "target_asset_ids": ["t1"],
                "records_in": 100,
                "records_out": 90,
            },
            {
                "transformation_type": "aggregate",
                "agent_id": "agent-2",
                "pipeline_id": "pipe-2",
                "source_asset_ids": ["s2"],
                "target_asset_ids": ["t2"],
                "records_in": 200,
                "records_out": 50,
            },
        ]
        result = transformation_tracker.batch_record(specs)
        assert len(result["recorded"]) == 2
        assert result["failed"] == 0
        assert result["errors"] == []

    def test_batch_with_failures(self, transformation_tracker):
        """Batch record with some invalid entries continues processing."""
        specs = [
            {
                "transformation_type": "filter",
                "agent_id": "agent",
                "pipeline_id": "pipe",
                "source_asset_ids": ["s"],
                "target_asset_ids": ["t"],
            },
            {
                "transformation_type": "invalid_type",  # Invalid
                "agent_id": "agent",
                "pipeline_id": "pipe",
                "source_asset_ids": ["s"],
                "target_asset_ids": ["t"],
            },
            {
                "transformation_type": "join",
                "agent_id": "",  # Invalid: empty
                "pipeline_id": "pipe",
                "source_asset_ids": ["s"],
                "target_asset_ids": ["t"],
            },
        ]
        result = transformation_tracker.batch_record(specs)
        assert len(result["recorded"]) == 1
        assert result["failed"] == 2
        assert len(result["errors"]) == 2

    def test_batch_empty_list(self, transformation_tracker):
        """Batch record with empty list returns zero counts."""
        result = transformation_tracker.batch_record([])
        assert len(result["recorded"]) == 0
        assert result["failed"] == 0

    def test_batch_non_dict_entry(self, transformation_tracker):
        """Batch record with non-dict entry records error."""
        specs = [
            "not a dict",
            {
                "transformation_type": "filter",
                "agent_id": "agent",
                "pipeline_id": "pipe",
                "source_asset_ids": ["s"],
                "target_asset_ids": ["t"],
            },
        ]
        result = transformation_tracker.batch_record(specs)
        assert len(result["recorded"]) == 1
        assert result["failed"] == 1


# ============================================================================
# TestTransformationChain
# ============================================================================


class TestTransformationChain:
    """Tests for TransformationTrackerEngine.get_transformation_chain()."""

    def _build_chain(self, tracker):
        """Build a 3-step transformation chain: A -> B -> C -> D.

        Returns dict mapping asset IDs for verification.
        """
        tracker.record_transformation(
            transformation_type="filter",
            agent_id="step-1",
            pipeline_id="chain-pipe",
            source_asset_ids=["asset-A"],
            target_asset_ids=["asset-B"],
        )
        tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="step-2",
            pipeline_id="chain-pipe",
            source_asset_ids=["asset-B"],
            target_asset_ids=["asset-C"],
        )
        tracker.record_transformation(
            transformation_type="enrich",
            agent_id="step-3",
            pipeline_id="chain-pipe",
            source_asset_ids=["asset-C"],
            target_asset_ids=["asset-D"],
        )

    def test_chain_backward(self, transformation_tracker):
        """Backward chain from asset-D should find all 3 upstream steps."""
        self._build_chain(transformation_tracker)
        chain = transformation_tracker.get_transformation_chain(
            "asset-D", direction="backward"
        )
        assert len(chain) == 3
        types = {c["transformation_type"] for c in chain}
        assert "filter" in types
        assert "aggregate" in types
        assert "enrich" in types

    def test_chain_forward(self, transformation_tracker):
        """Forward chain from asset-A should find all 3 downstream steps."""
        self._build_chain(transformation_tracker)
        chain = transformation_tracker.get_transformation_chain(
            "asset-A", direction="forward"
        )
        assert len(chain) == 3

    def test_chain_backward_from_midpoint(self, transformation_tracker):
        """Backward chain from asset-C should find 2 upstream steps."""
        self._build_chain(transformation_tracker)
        chain = transformation_tracker.get_transformation_chain(
            "asset-C", direction="backward"
        )
        assert len(chain) == 2
        types = {c["transformation_type"] for c in chain}
        assert "filter" in types
        assert "aggregate" in types

    def test_chain_forward_from_midpoint(self, transformation_tracker):
        """Forward chain from asset-B should find 2 downstream steps."""
        self._build_chain(transformation_tracker)
        chain = transformation_tracker.get_transformation_chain(
            "asset-B", direction="forward"
        )
        assert len(chain) == 2
        types = {c["transformation_type"] for c in chain}
        assert "aggregate" in types
        assert "enrich" in types

    def test_chain_no_matches(self, transformation_tracker):
        """Chain for an asset with no transformations returns empty list."""
        self._build_chain(transformation_tracker)
        chain = transformation_tracker.get_transformation_chain(
            "nonexistent-asset", direction="backward"
        )
        assert chain == []

    def test_chain_empty_asset_id_raises(self, transformation_tracker):
        """Empty asset_id raises ValueError."""
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            transformation_tracker.get_transformation_chain("")

    def test_chain_invalid_direction_raises(self, transformation_tracker):
        """Invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="direction must be"):
            transformation_tracker.get_transformation_chain(
                "asset-A", direction="sideways"
            )

    def test_chain_handles_cycles(self, transformation_tracker):
        """Chain traversal handles cycles without infinite loop."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="cycle-agent",
            pipeline_id="cycle-pipe",
            source_asset_ids=["cycle-A"],
            target_asset_ids=["cycle-B"],
        )
        transformation_tracker.record_transformation(
            transformation_type="enrich",
            agent_id="cycle-agent",
            pipeline_id="cycle-pipe",
            source_asset_ids=["cycle-B"],
            target_asset_ids=["cycle-A"],  # Creates cycle
        )
        # Should not hang; visited set prevents infinite loop
        chain = transformation_tracker.get_transformation_chain(
            "cycle-A", direction="forward"
        )
        assert len(chain) == 2  # Both transformations found but no infinite loop


# ============================================================================
# TestTransformationStatistics
# ============================================================================


class TestTransformationStatistics:
    """Tests for TransformationTrackerEngine.get_statistics()."""

    def test_get_statistics(self, transformation_tracker):
        """Verify statistics reflect recorded transformations."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent-a",
            pipeline_id="pipe-1",
            source_asset_ids=["s1"],
            target_asset_ids=["t1"],
            records_in=100,
            records_out=90,
            records_filtered=10,
            records_error=0,
            duration_ms=50.0,
        )
        transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="agent-b",
            pipeline_id="pipe-2",
            source_asset_ids=["s2"],
            target_asset_ids=["t2"],
            records_in=200,
            records_out=50,
            records_filtered=0,
            records_error=5,
            duration_ms=150.0,
        )
        stats = transformation_tracker.get_statistics()

        assert stats["total"] == 2
        assert stats["by_type"]["filter"] == 1
        assert stats["by_type"]["aggregate"] == 1
        assert stats["by_agent"]["agent-a"] == 1
        assert stats["by_agent"]["agent-b"] == 1
        assert stats["by_pipeline"]["pipe-1"] == 1
        assert stats["by_pipeline"]["pipe-2"] == 1
        assert stats["total_records_in"] == 300
        assert stats["total_records_out"] == 140
        assert stats["total_records_filtered"] == 10
        assert stats["total_records_error"] == 5
        assert stats["avg_duration_ms"] == pytest.approx(100.0, abs=0.01)

    def test_statistics_by_type(self, transformation_tracker):
        """Statistics by_type counts correctly."""
        for _ in range(3):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="a",
                pipeline_id="p",
                source_asset_ids=["s"],
                target_asset_ids=["t"],
            )
        transformation_tracker.record_transformation(
            transformation_type="join",
            agent_id="b",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        stats = transformation_tracker.get_statistics()
        assert stats["by_type"]["filter"] == 3
        assert stats["by_type"]["join"] == 1

    def test_statistics_empty(self, transformation_tracker):
        """Statistics on empty tracker show all zeros."""
        stats = transformation_tracker.get_statistics()
        assert stats["total"] == 0
        assert stats["by_type"] == {}
        assert stats["by_agent"] == {}
        assert stats["by_pipeline"] == {}
        assert stats["avg_duration_ms"] == 0.0
        assert stats["total_records_in"] == 0
        assert stats["total_records_out"] == 0
        assert stats["total_records_filtered"] == 0
        assert stats["total_records_error"] == 0

    def test_statistics_avg_duration(self, transformation_tracker):
        """Average duration is computed correctly across all transformations."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            duration_ms=100.0,
        )
        transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="b",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            duration_ms=300.0,
        )
        stats = transformation_tracker.get_statistics()
        assert stats["avg_duration_ms"] == pytest.approx(200.0, abs=0.01)


# ============================================================================
# TestTransformationMisc
# ============================================================================


class TestTransformationMisc:
    """Miscellaneous tests: export, clear, properties, repr, thread safety."""

    def test_export(self, transformation_tracker):
        """Export all transformations as list of dicts."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="agent-2",
            pipeline_id="pipe-2",
            source_asset_ids=["s2"],
            target_asset_ids=["t2"],
        )
        exported = transformation_tracker.export_transformations()
        assert len(exported) == 2
        # Sorted by created_at
        assert exported[0]["created_at"] <= exported[1]["created_at"]

    def test_export_by_agent(self, transformation_tracker):
        """Export filtered by agent_id returns only matching."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="target-agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="other-agent",
            pipeline_id="pipe",
            source_asset_ids=["s2"],
            target_asset_ids=["t2"],
        )
        exported = transformation_tracker.export_transformations(
            agent_id="target-agent"
        )
        assert len(exported) == 1
        assert exported[0]["agent_id"] == "target-agent"

    def test_clear(self, transformation_tracker):
        """clear() removes all transformations and indexes."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        assert transformation_tracker.count > 0
        transformation_tracker.clear()
        assert transformation_tracker.count == 0

    def test_count_property(self, transformation_tracker):
        """count property returns correct count."""
        assert transformation_tracker.count == 0
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        assert transformation_tracker.count == 1

    def test_len_dunder(self, transformation_tracker):
        """__len__ returns the same as count property."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        assert len(transformation_tracker) == 1
        assert len(transformation_tracker) == transformation_tracker.count

    def test_thread_safety(self, provenance_tracker):
        """Concurrent recording from multiple threads must not corrupt state."""
        engine = TransformationTrackerEngine(provenance=provenance_tracker)
        num_threads = 10
        records_per_thread = 5
        errors: List[str] = []

        def record_batch(thread_id: int):
            try:
                for i in range(records_per_thread):
                    engine.record_transformation(
                        transformation_type="filter",
                        agent_id=f"thread-{thread_id}",
                        pipeline_id=f"pipe-{thread_id}",
                        source_asset_ids=[f"s-{thread_id}-{i}"],
                        target_asset_ids=[f"t-{thread_id}-{i}"],
                    )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=record_batch, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert engine.count == num_threads * records_per_thread


# ============================================================================
# TestTransformationEdgeCases
# ============================================================================


class TestTransformationEdgeCases:
    """Edge case and boundary tests for TransformationTrackerEngine."""

    # -- Input validation --

    def test_empty_transformation_type_raises(self, transformation_tracker):
        """Empty transformation_type raises ValueError."""
        with pytest.raises(ValueError, match="transformation_type must not be empty"):
            transformation_tracker.record_transformation(
                transformation_type="",
                agent_id="agent",
                pipeline_id="pipe",
                source_asset_ids=["s"],
                target_asset_ids=["t"],
            )

    def test_source_not_list_raises(self, transformation_tracker):
        """source_asset_ids as string (not list) raises TypeError."""
        with pytest.raises(TypeError, match="source_asset_ids must be a list"):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="agent",
                pipeline_id="pipe",
                source_asset_ids="not-a-list",
                target_asset_ids=["t"],
            )

    def test_target_not_list_raises(self, transformation_tracker):
        """target_asset_ids as string (not list) raises TypeError."""
        with pytest.raises(TypeError, match="target_asset_ids must be a list"):
            transformation_tracker.record_transformation(
                transformation_type="filter",
                agent_id="agent",
                pipeline_id="pipe",
                source_asset_ids=["s"],
                target_asset_ids="not-a-list",
            )

    # -- Negative record counts clamp to zero --

    def test_negative_records_in_clamps(self, transformation_tracker):
        """Negative records_in is clamped to 0."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            records_in=-10,
        )
        assert result["records_in"] == 0

    def test_negative_records_out_clamps(self, transformation_tracker):
        """Negative records_out is clamped to 0."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            records_out=-10,
        )
        assert result["records_out"] == 0

    def test_negative_records_filtered_clamps(self, transformation_tracker):
        """Negative records_filtered is clamped to 0."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            records_filtered=-5,
        )
        assert result["records_filtered"] == 0

    def test_negative_records_error_clamps(self, transformation_tracker):
        """Negative records_error is clamped to 0."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            records_error=-3,
        )
        assert result["records_error"] == 0

    def test_negative_duration_clamps(self, transformation_tracker):
        """Negative duration_ms is clamped to 0.0."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            duration_ms=-50.0,
        )
        assert result["duration_ms"] == 0.0

    # -- Multiple source/target assets --

    def test_multiple_source_assets(self, transformation_tracker):
        """Transformation with multiple source assets stores all."""
        result = transformation_tracker.record_transformation(
            transformation_type="join",
            agent_id="join-agent",
            pipeline_id="pipe",
            source_asset_ids=["s1", "s2", "s3"],
            target_asset_ids=["t1"],
        )
        assert result["source_asset_ids"] == ["s1", "s2", "s3"]

    def test_multiple_target_assets(self, transformation_tracker):
        """Transformation with multiple target assets stores all."""
        result = transformation_tracker.record_transformation(
            transformation_type="split",
            agent_id="split-agent",
            pipeline_id="pipe",
            source_asset_ids=["s1"],
            target_asset_ids=["t1", "t2", "t3"],
        )
        assert result["target_asset_ids"] == ["t1", "t2", "t3"]

    # -- Parameters and metadata edge cases --

    def test_none_parameters(self, transformation_tracker):
        """None parameters defaults to empty dict."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            parameters=None,
        )
        assert result["parameters"] == {}

    def test_none_metadata(self, transformation_tracker):
        """None metadata defaults to empty dict."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="agent",
            pipeline_id="pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
            metadata=None,
        )
        assert result["metadata"] == {}

    # -- Provenance chain integrity --

    def test_provenance_hashes_are_unique(self, transformation_tracker):
        """Each recorded transformation gets a unique provenance hash."""
        r1 = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a1",
            pipeline_id="p1",
            source_asset_ids=["s1"],
            target_asset_ids=["t1"],
        )
        r2 = transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="a2",
            pipeline_id="p2",
            source_asset_ids=["s2"],
            target_asset_ids=["t2"],
        )
        assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_provenance_hash_is_sha256(self, transformation_tracker):
        """Provenance hash is a valid 64-character hex SHA-256 digest."""
        result = transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        h = result["provenance_hash"]
        assert len(h) == 64
        # Verify it is valid hex
        int(h, 16)

    # -- Time index queries --

    def test_get_transformations_in_range_empty_start(
        self, transformation_tracker
    ):
        """Empty start_time returns empty list."""
        result = transformation_tracker.get_transformations_in_range("", "2099-01-01")
        assert result == []

    def test_get_transformations_in_range_empty_end(
        self, transformation_tracker
    ):
        """Empty end_time returns empty list."""
        result = transformation_tracker.get_transformations_in_range("2020-01-01", "")
        assert result == []

    def test_get_transformations_in_range_valid(self, transformation_tracker):
        """Valid range returns matching transformations."""
        now = datetime.now(timezone.utc)
        before = (now - timedelta(seconds=1)).isoformat()
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        after = (now + timedelta(seconds=5)).isoformat()

        results = transformation_tracker.get_transformations_in_range(before, after)
        assert len(results) >= 1

    # -- Dedicated retrieval methods edge cases --

    def test_get_by_agent_empty_returns_empty(self, transformation_tracker):
        """get_transformations_by_agent with empty string returns empty."""
        result = transformation_tracker.get_transformations_by_agent("")
        assert result == []

    def test_get_by_pipeline_empty_returns_empty(self, transformation_tracker):
        """get_transformations_by_pipeline with empty string returns empty."""
        result = transformation_tracker.get_transformations_by_pipeline("")
        assert result == []

    def test_get_by_type_empty_returns_empty(self, transformation_tracker):
        """get_transformations_by_type with empty string returns empty."""
        result = transformation_tracker.get_transformations_by_type("")
        assert result == []

    def test_get_by_agent_nonexistent_returns_empty(
        self, transformation_tracker
    ):
        """get_transformations_by_agent for non-existent agent returns empty."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="actual-agent",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        result = transformation_tracker.get_transformations_by_agent("fake-agent")
        assert result == []

    def test_get_by_pipeline_nonexistent_returns_empty(
        self, transformation_tracker
    ):
        """get_transformations_by_pipeline for non-existent pipeline returns empty."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="actual-pipe",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        result = transformation_tracker.get_transformations_by_pipeline("fake-pipe")
        assert result == []

    # -- Clear and re-use --

    def test_clear_then_record(self, transformation_tracker):
        """After clear, recording still works correctly."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        transformation_tracker.clear()
        assert transformation_tracker.count == 0

        transformation_tracker.record_transformation(
            transformation_type="aggregate",
            agent_id="b",
            pipeline_id="q",
            source_asset_ids=["s2"],
            target_asset_ids=["t2"],
        )
        assert transformation_tracker.count == 1

    def test_clear_resets_all_indexes(self, transformation_tracker):
        """After clear, all search indexes are empty."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        transformation_tracker.clear()

        assert transformation_tracker.search_transformations(
            transformation_type="filter"
        ) == []
        assert transformation_tracker.search_transformations(
            agent_id="a"
        ) == []
        assert transformation_tracker.search_transformations(
            pipeline_id="p"
        ) == []

    # -- repr --

    def test_repr(self, transformation_tracker):
        """__repr__ returns a meaningful string."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="a",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        r = repr(transformation_tracker)
        assert "TransformationTrackerEngine" in r
        assert "transformations=1" in r

    # -- Provenance property --

    def test_provenance_property(self, transformation_tracker, provenance_tracker):
        """provenance property returns the tracker instance."""
        assert transformation_tracker.provenance is provenance_tracker

    # -- Export empty --

    def test_export_empty_returns_empty(self, transformation_tracker):
        """Export from empty tracker returns empty list."""
        exported = transformation_tracker.export_transformations()
        assert exported == []

    def test_export_by_nonexistent_agent_returns_empty(
        self, transformation_tracker
    ):
        """Export filtered by non-existent agent returns empty list."""
        transformation_tracker.record_transformation(
            transformation_type="filter",
            agent_id="real-agent",
            pipeline_id="p",
            source_asset_ids=["s"],
            target_asset_ids=["t"],
        )
        exported = transformation_tracker.export_transformations(
            agent_id="fake-agent"
        )
        assert exported == []
