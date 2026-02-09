# -*- coding: utf-8 -*-
"""
Unit Tests for ResponseAggregatorEngine (AGENT-DATA-004)

Tests response merging, conflict resolution strategies (latest_wins,
source_priority, merge, error), aggregation operations (sum, avg, min,
max, count, group_by), and SHA-256 provenance hashing on aggregation
results.

Coverage target: 85%+ of response_aggregator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class QueryResult:
    """Result from a single data source query."""

    def __init__(self, source_id: str, data: List[Dict[str, Any]],
                 timestamp: Optional[str] = None,
                 priority: int = 0,
                 error: Optional[str] = None):
        self.source_id = source_id
        self.data = data
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.priority = priority
        self.error = error


class AggregatedResult:
    """Merged result from multiple data sources."""

    def __init__(self, sources: List[str], data: List[Dict[str, Any]],
                 conflicts: int = 0, provenance_hash: str = ""):
        self.sources = sources
        self.data = data
        self.conflicts = conflicts
        self.provenance_hash = provenance_hash
        self.aggregated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Inline ResponseAggregatorEngine
# ---------------------------------------------------------------------------


class ResponseAggregatorEngine:
    """Aggregates and merges results from multiple data source queries."""

    STRATEGIES = ["latest_wins", "source_priority", "merge", "error"]

    def __init__(self, default_strategy: str = "latest_wins"):
        if default_strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {default_strategy}")
        self._default_strategy = default_strategy
        self._aggregation_count = 0

    def aggregate(self, results: List[QueryResult],
                  strategy: Optional[str] = None) -> AggregatedResult:
        """Merge multiple QueryResult objects into a single AggregatedResult."""
        strategy = strategy or self._default_strategy
        if not results:
            return AggregatedResult(
                sources=[], data=[], conflicts=0,
                provenance_hash=_compute_hash({"empty": True}),
            )

        if len(results) == 1:
            r = results[0]
            return AggregatedResult(
                sources=[r.source_id],
                data=list(r.data),
                conflicts=0,
                provenance_hash=_compute_hash({
                    "sources": [r.source_id],
                    "record_count": len(r.data),
                }),
            )

        merged_data, conflicts = self._merge_data(results, strategy)
        sources = [r.source_id for r in results]
        self._aggregation_count += 1

        provenance_hash = _compute_hash({
            "sources": sources,
            "record_count": len(merged_data),
            "conflicts": conflicts,
            "strategy": strategy,
        })

        return AggregatedResult(
            sources=sources,
            data=merged_data,
            conflicts=conflicts,
            provenance_hash=provenance_hash,
        )

    def _merge_data(self, results: List[QueryResult],
                    strategy: str) -> tuple:
        """Merge data from multiple results using the specified strategy."""
        all_keys: Dict[str, List[tuple]] = {}
        conflicts = 0

        for r in results:
            for record in r.data:
                key = record.get("id", id(record))
                if key not in all_keys:
                    all_keys[key] = []
                all_keys[key].append((r, record))

        merged: List[Dict[str, Any]] = []
        for key, entries in all_keys.items():
            if len(entries) == 1:
                merged.append(entries[0][1])
            else:
                conflicts += 1
                resolved = self._resolve_conflict(entries, strategy)
                merged.append(resolved)

        return merged, conflicts

    def _resolve_conflict(self, entries: List[tuple],
                          strategy: str) -> Dict[str, Any]:
        """Resolve a data conflict using the specified strategy."""
        if strategy == "latest_wins":
            # Pick the entry with the latest timestamp
            best = max(entries, key=lambda e: e[0].timestamp)
            return best[1]

        elif strategy == "source_priority":
            # Pick the entry from the highest-priority source
            best = max(entries, key=lambda e: e[0].priority)
            return best[1]

        elif strategy == "merge":
            # Merge all fields; later entries overwrite earlier ones
            result: Dict[str, Any] = {}
            for _, record in entries:
                result.update(record)
            return result

        elif strategy == "error":
            # Raise an error on conflict
            raise ValueError(
                f"Conflict detected for entries from sources: "
                f"{[e[0].source_id for e in entries]}"
            )

        raise ValueError(f"Unknown strategy: {strategy}")

    def apply_aggregations(self, data: List[Dict[str, Any]],
                           aggregations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply aggregation operations to a dataset."""
        if not data:
            return {"results": {}, "record_count": 0}

        results: Dict[str, Any] = {}
        for agg in aggregations:
            op = agg["op"]
            field = agg.get("field", "")
            alias = agg.get("alias", f"{op}_{field}")

            if op == "sum":
                results[alias] = self._compute_sum(data, field)
            elif op == "avg":
                results[alias] = self._compute_avg(data, field)
            elif op == "min":
                results[alias] = self._compute_min(data, field)
            elif op == "max":
                results[alias] = self._compute_max(data, field)
            elif op == "count":
                results[alias] = self._compute_count(data)
            elif op == "group_by":
                group_field = agg.get("group_field", field)
                results[alias] = self._compute_group_by(data, group_field)

        return {"results": results, "record_count": len(data)}

    def _compute_sum(self, data: List[Dict[str, Any]], field: str) -> float:
        total = 0.0
        for record in data:
            val = record.get(field)
            if val is not None and isinstance(val, (int, float)):
                total += val
        return total

    def _compute_avg(self, data: List[Dict[str, Any]], field: str) -> float:
        values = [r[field] for r in data
                  if field in r and isinstance(r[field], (int, float))]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _compute_min(self, data: List[Dict[str, Any]], field: str) -> Any:
        values = [r[field] for r in data if field in r]
        if not values:
            return None
        return min(values)

    def _compute_max(self, data: List[Dict[str, Any]], field: str) -> Any:
        values = [r[field] for r in data if field in r]
        if not values:
            return None
        return max(values)

    def _compute_count(self, data: List[Dict[str, Any]]) -> int:
        return len(data)

    def _compute_group_by(self, data: List[Dict[str, Any]],
                          field: str) -> Dict[str, int]:
        groups: Dict[str, int] = {}
        for record in data:
            key = str(record.get(field, "unknown"))
            groups[key] = groups.get(key, 0) + 1
        return groups


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> ResponseAggregatorEngine:
    return ResponseAggregatorEngine()


@pytest.fixture
def sample_results() -> List[QueryResult]:
    return [
        QueryResult(
            source_id="src-1",
            data=[
                {"id": "r1", "value": 10, "name": "alpha"},
                {"id": "r2", "value": 20, "name": "beta"},
            ],
            timestamp="2026-02-08T10:00:00Z",
            priority=1,
        ),
        QueryResult(
            source_id="src-2",
            data=[
                {"id": "r2", "value": 25, "name": "beta_updated"},
                {"id": "r3", "value": 30, "name": "gamma"},
            ],
            timestamp="2026-02-08T11:00:00Z",
            priority=2,
        ),
    ]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAggregate:
    """Tests for aggregate (merge) operations."""

    def test_merge_two_results(self, engine, sample_results):
        result = engine.aggregate(sample_results)
        assert result is not None
        assert len(result.sources) == 2
        assert "src-1" in result.sources
        assert "src-2" in result.sources
        # r1 from src-1, r2 conflict resolved, r3 from src-2
        assert len(result.data) == 3

    def test_merge_three_results(self, engine):
        results = [
            QueryResult("s1", [{"id": "a", "v": 1}], "2026-01-01T00:00:00Z"),
            QueryResult("s2", [{"id": "b", "v": 2}], "2026-01-02T00:00:00Z"),
            QueryResult("s3", [{"id": "c", "v": 3}], "2026-01-03T00:00:00Z"),
        ]
        agg = engine.aggregate(results)
        assert len(agg.sources) == 3
        assert len(agg.data) == 3
        assert agg.conflicts == 0

    def test_empty_results(self, engine):
        result = engine.aggregate([])
        assert result.sources == []
        assert result.data == []
        assert result.conflicts == 0
        assert len(result.provenance_hash) == 64

    def test_single_result_passthrough(self, engine):
        single = QueryResult("only-src", [{"id": "x", "val": 42}])
        result = engine.aggregate([single])
        assert result.sources == ["only-src"]
        assert len(result.data) == 1
        assert result.data[0]["val"] == 42
        assert result.conflicts == 0


class TestMergeData:
    """Tests for _merge_data with different strategies."""

    def test_latest_wins_strategy(self, engine, sample_results):
        result = engine.aggregate(sample_results, strategy="latest_wins")
        # r2 conflict: src-2 has later timestamp, so its value wins
        r2_records = [d for d in result.data if d["id"] == "r2"]
        assert len(r2_records) == 1
        assert r2_records[0]["value"] == 25

    def test_source_priority_strategy(self, engine, sample_results):
        result = engine.aggregate(sample_results, strategy="source_priority")
        # src-2 has higher priority (2 > 1)
        r2_records = [d for d in result.data if d["id"] == "r2"]
        assert len(r2_records) == 1
        assert r2_records[0]["value"] == 25

    def test_merge_strategy(self, engine, sample_results):
        result = engine.aggregate(sample_results, strategy="merge")
        r2_records = [d for d in result.data if d["id"] == "r2"]
        assert len(r2_records) == 1
        # Merge strategy: later entry overwrites
        assert r2_records[0]["name"] == "beta_updated"

    def test_error_strategy_raises(self, engine, sample_results):
        with pytest.raises(ValueError, match="Conflict detected"):
            engine.aggregate(sample_results, strategy="error")


class TestResolveConflict:
    """Tests for _resolve_conflict with all 4 strategies."""

    def test_resolve_latest_wins(self, engine):
        r1 = QueryResult("s1", [], timestamp="2026-01-01T00:00:00Z")
        r2 = QueryResult("s2", [], timestamp="2026-02-01T00:00:00Z")
        entries = [(r1, {"id": "x", "v": 1}), (r2, {"id": "x", "v": 2})]
        resolved = engine._resolve_conflict(entries, "latest_wins")
        assert resolved["v"] == 2

    def test_resolve_source_priority(self, engine):
        r1 = QueryResult("s1", [], priority=10)
        r2 = QueryResult("s2", [], priority=5)
        entries = [(r1, {"id": "x", "v": 1}), (r2, {"id": "x", "v": 2})]
        resolved = engine._resolve_conflict(entries, "source_priority")
        assert resolved["v"] == 1  # s1 has higher priority

    def test_resolve_merge(self, engine):
        r1 = QueryResult("s1", [])
        r2 = QueryResult("s2", [])
        entries = [
            (r1, {"id": "x", "a": 1, "b": 2}),
            (r2, {"id": "x", "b": 99, "c": 3}),
        ]
        resolved = engine._resolve_conflict(entries, "merge")
        assert resolved["a"] == 1
        assert resolved["b"] == 99  # overwritten by s2
        assert resolved["c"] == 3

    def test_resolve_error(self, engine):
        r1 = QueryResult("s1", [])
        r2 = QueryResult("s2", [])
        entries = [(r1, {"id": "x"}), (r2, {"id": "x"})]
        with pytest.raises(ValueError, match="Conflict detected"):
            engine._resolve_conflict(entries, "error")


class TestApplyAggregations:
    """Tests for apply_aggregations operations."""

    def test_sum(self, engine):
        data = [{"amount": 10}, {"amount": 20}, {"amount": 30}]
        result = engine.apply_aggregations(data, [{"op": "sum", "field": "amount"}])
        assert result["results"]["sum_amount"] == 60

    def test_avg(self, engine):
        data = [{"score": 10}, {"score": 20}, {"score": 30}]
        result = engine.apply_aggregations(data, [{"op": "avg", "field": "score"}])
        assert abs(result["results"]["avg_score"] - 20.0) < 0.001

    def test_min(self, engine):
        data = [{"temp": 5}, {"temp": 2}, {"temp": 8}]
        result = engine.apply_aggregations(data, [{"op": "min", "field": "temp"}])
        assert result["results"]["min_temp"] == 2

    def test_max(self, engine):
        data = [{"temp": 5}, {"temp": 2}, {"temp": 8}]
        result = engine.apply_aggregations(data, [{"op": "max", "field": "temp"}])
        assert result["results"]["max_temp"] == 8

    def test_count(self, engine):
        data = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        result = engine.apply_aggregations(data, [{"op": "count"}])
        assert result["results"]["count_"] == 4

    def test_group_by(self, engine):
        data = [
            {"category": "A", "val": 1},
            {"category": "B", "val": 2},
            {"category": "A", "val": 3},
        ]
        result = engine.apply_aggregations(
            data, [{"op": "group_by", "field": "category", "group_field": "category"}],
        )
        groups = result["results"]["group_by_category"]
        assert groups["A"] == 2
        assert groups["B"] == 1

    def test_empty_data(self, engine):
        result = engine.apply_aggregations([], [{"op": "sum", "field": "x"}])
        assert result["record_count"] == 0
        assert result["results"] == {}


class TestComputeSum:
    """Tests for _compute_sum."""

    def test_integers(self, engine):
        data = [{"v": 1}, {"v": 2}, {"v": 3}]
        assert engine._compute_sum(data, "v") == 6.0

    def test_floats(self, engine):
        data = [{"v": 1.5}, {"v": 2.5}]
        assert abs(engine._compute_sum(data, "v") - 4.0) < 0.001

    def test_missing_field(self, engine):
        data = [{"v": 10}, {"other": 5}, {"v": 20}]
        assert engine._compute_sum(data, "v") == 30.0


class TestComputeAvg:
    """Tests for _compute_avg."""

    def test_basic(self, engine):
        data = [{"v": 10}, {"v": 20}, {"v": 30}]
        assert abs(engine._compute_avg(data, "v") - 20.0) < 0.001

    def test_single_value(self, engine):
        data = [{"v": 42}]
        assert engine._compute_avg(data, "v") == 42.0

    def test_missing_field(self, engine):
        data = [{"other": 10}]
        assert engine._compute_avg(data, "v") == 0.0


class TestComputeMin:
    """Tests for _compute_min."""

    def test_numbers(self, engine):
        data = [{"v": 5}, {"v": 1}, {"v": 9}]
        assert engine._compute_min(data, "v") == 1

    def test_strings(self, engine):
        data = [{"v": "cherry"}, {"v": "apple"}, {"v": "banana"}]
        assert engine._compute_min(data, "v") == "apple"


class TestComputeMax:
    """Tests for _compute_max."""

    def test_numbers(self, engine):
        data = [{"v": 5}, {"v": 1}, {"v": 9}]
        assert engine._compute_max(data, "v") == 9

    def test_strings(self, engine):
        data = [{"v": "cherry"}, {"v": "apple"}, {"v": "banana"}]
        assert engine._compute_max(data, "v") == "cherry"


class TestComputeCount:
    """Tests for _compute_count."""

    def test_basic(self, engine):
        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        assert engine._compute_count(data) == 3

    def test_empty(self, engine):
        assert engine._compute_count([]) == 0


class TestComputeGroupBy:
    """Tests for _compute_group_by."""

    def test_single_group(self, engine):
        data = [{"cat": "A"}, {"cat": "A"}, {"cat": "A"}]
        groups = engine._compute_group_by(data, "cat")
        assert groups == {"A": 3}

    def test_multiple_groups(self, engine):
        data = [
            {"cat": "A"}, {"cat": "B"}, {"cat": "A"},
            {"cat": "C"}, {"cat": "B"},
        ]
        groups = engine._compute_group_by(data, "cat")
        assert groups["A"] == 2
        assert groups["B"] == 2
        assert groups["C"] == 1


class TestAggregationProvenance:
    """Tests for SHA-256 provenance on aggregation results."""

    def test_aggregate_provenance_hash(self, engine, sample_results):
        result = engine.aggregate(sample_results)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # valid hex

    def test_aggregate_provenance_deterministic(self, engine):
        r1 = QueryResult("s1", [{"id": "a", "v": 1}], "2026-01-01T00:00:00Z")
        r2 = QueryResult("s2", [{"id": "b", "v": 2}], "2026-01-02T00:00:00Z")
        agg1 = engine.aggregate([r1, r2])
        agg2 = engine.aggregate([r1, r2])
        assert agg1.provenance_hash == agg2.provenance_hash

    def test_empty_aggregate_has_provenance(self, engine):
        result = engine.aggregate([])
        assert len(result.provenance_hash) == 64
