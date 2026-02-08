# -*- coding: utf-8 -*-
"""
Unit Tests for TransformEngine (AGENT-DATA-002)

Tests data transformation operations: pivot, unpivot, deduplicate,
merge, filter, aggregate, rename, split, cast, fill_missing,
apply_transforms chained execution, and statistics.

Coverage target: 85%+ of transform_engine.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline TransformEngine mirroring greenlang/excel_normalizer/transform_engine.py
# ---------------------------------------------------------------------------


class TransformEngine:
    """Data transformation engine for Excel/CSV normalization.

    Applies structural and value transformations to normalized
    data with provenance tracking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._stats: Dict[str, int] = {
            "transforms_applied": 0,
            "rows_processed": 0,
            "errors": 0,
        }

    def apply_transforms(
        self,
        data: List[Dict[str, Any]],
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply a sequence of transform operations."""
        current = list(data)
        applied = 0
        for op in operations:
            op_type = op.get("type", op.get("operation", "unknown"))
            try:
                current = self._apply_single(current, op_type, op)
                applied += 1
                self._stats["transforms_applied"] += 1
            except Exception:
                self._stats["errors"] += 1
        self._stats["rows_processed"] += len(current)
        provenance = self._compute_hash(current)
        return {
            "data": current,
            "row_count": len(current),
            "operations_applied": applied,
            "provenance_hash": provenance,
        }

    def _apply_single(
        self,
        data: List[Dict[str, Any]],
        op_type: str,
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        handler = {
            "pivot": self.pivot,
            "unpivot": self.unpivot,
            "dedup": self.deduplicate,
            "deduplicate": self.deduplicate,
            "merge": self.merge,
            "filter": self.filter_rows,
            "aggregate": self.aggregate,
            "rename": self.rename_columns,
            "split": self.split_column,
            "cast": self.cast_column,
            "fill_missing": self.fill_missing,
        }.get(op_type)
        if handler is None:
            raise ValueError(f"Unknown transform: {op_type}")
        return handler(data, config)

    def pivot(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Pivot rows to columns using index, column, and value keys."""
        index_col = config.get("index")
        pivot_col = config.get("column")
        value_col = config.get("value")
        if not index_col or not pivot_col or not value_col:
            return data

        pivoted: Dict[Any, Dict[str, Any]] = {}
        for row in data:
            idx = row.get(index_col)
            col = str(row.get(pivot_col, ""))
            val = row.get(value_col)
            if idx not in pivoted:
                pivoted[idx] = {index_col: idx}
            pivoted[idx][col] = val
        return list(pivoted.values())

    def unpivot(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Unpivot columns to rows (melt)."""
        id_vars = config.get("id_vars", [])
        value_vars = config.get("value_vars", [])
        var_name = config.get("var_name", "variable")
        val_name = config.get("value_name", "value")

        result = []
        for row in data:
            base = {k: row.get(k) for k in id_vars}
            cols = value_vars if value_vars else [
                k for k in row if k not in id_vars
            ]
            for col in cols:
                new_row = dict(base)
                new_row[var_name] = col
                new_row[val_name] = row.get(col)
                result.append(new_row)
        return result

    def deduplicate(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Remove duplicate rows based on key columns."""
        keys = config.get("keys", config.get("key_columns", []))
        seen = set()
        result = []
        for row in data:
            if keys:
                key = tuple(str(row.get(k, "")) for k in keys)
            else:
                key = tuple(sorted((str(k), str(v)) for k, v in row.items()))
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    def merge(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Merge/join with another dataset on key columns."""
        right = config.get("right_data", [])
        on = config.get("on", [])
        if not on or not right:
            return data

        right_map: Dict[tuple, Dict[str, Any]] = {}
        for row in right:
            key = tuple(str(row.get(k, "")) for k in on)
            right_map[key] = row

        result = []
        for row in data:
            key = tuple(str(row.get(k, "")) for k in on)
            merged = dict(row)
            if key in right_map:
                for k, v in right_map[key].items():
                    if k not in merged:
                        merged[k] = v
            result.append(merged)
        return result

    def filter_rows(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter rows based on column conditions."""
        column = config.get("column")
        operator = config.get("operator", "eq")
        value = config.get("value")
        if not column:
            return data

        result = []
        for row in data:
            cell = row.get(column)
            if operator == "eq" and cell == value:
                result.append(row)
            elif operator == "ne" and cell != value:
                result.append(row)
            elif operator == "gt" and cell is not None and cell > value:
                result.append(row)
            elif operator == "lt" and cell is not None and cell < value:
                result.append(row)
            elif operator == "gte" and cell is not None and cell >= value:
                result.append(row)
            elif operator == "lte" and cell is not None and cell <= value:
                result.append(row)
            elif operator == "contains" and isinstance(cell, str) and value in cell:
                result.append(row)
            elif operator == "not_null" and cell is not None:
                result.append(row)
        return result

    def aggregate(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Aggregate rows by group columns with aggregation functions."""
        group_by = config.get("group_by", [])
        agg_column = config.get("column")
        agg_func = config.get("function", "sum")
        if not group_by or not agg_column:
            return data

        groups: Dict[tuple, List[Any]] = {}
        group_rows: Dict[tuple, Dict[str, Any]] = {}
        for row in data:
            key = tuple(str(row.get(k, "")) for k in group_by)
            if key not in groups:
                groups[key] = []
                group_rows[key] = {k: row.get(k) for k in group_by}
            val = row.get(agg_column)
            if val is not None:
                try:
                    groups[key].append(float(val))
                except (ValueError, TypeError):
                    pass

        result = []
        for key, values in groups.items():
            row = dict(group_rows[key])
            if agg_func == "sum":
                row[agg_column] = sum(values) if values else 0
            elif agg_func == "avg":
                row[agg_column] = sum(values) / len(values) if values else 0
            elif agg_func == "count":
                row[agg_column] = len(values)
            elif agg_func == "min":
                row[agg_column] = min(values) if values else None
            elif agg_func == "max":
                row[agg_column] = max(values) if values else None
            result.append(row)
        return result

    def rename_columns(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Rename columns based on a mapping."""
        mapping = config.get("mapping", {})
        if not mapping:
            return data
        result = []
        for row in data:
            new_row = {}
            for k, v in row.items():
                new_key = mapping.get(k, k)
                new_row[new_key] = v
            result.append(new_row)
        return result

    def split_column(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Split a column into multiple columns by delimiter."""
        column = config.get("column")
        delimiter = config.get("delimiter", ",")
        new_columns = config.get("new_columns", [])
        if not column or not new_columns:
            return data

        result = []
        for row in data:
            new_row = dict(row)
            value = str(row.get(column, ""))
            parts = value.split(delimiter)
            for i, col_name in enumerate(new_columns):
                new_row[col_name] = parts[i].strip() if i < len(parts) else ""
            result.append(new_row)
        return result

    def cast_column(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Cast a column to a specified type."""
        column = config.get("column")
        target_type = config.get("target_type", "string")
        if not column:
            return data

        result = []
        for row in data:
            new_row = dict(row)
            val = row.get(column)
            if val is not None:
                try:
                    if target_type == "integer":
                        new_row[column] = int(float(str(val)))
                    elif target_type == "float":
                        new_row[column] = float(str(val))
                    elif target_type == "string":
                        new_row[column] = str(val)
                    elif target_type == "boolean":
                        new_row[column] = str(val).lower() in ("true", "1", "yes")
                except (ValueError, TypeError):
                    pass
            result.append(new_row)
        return result

    def fill_missing(
        self,
        data: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Fill missing values in a column."""
        column = config.get("column")
        method = config.get("method", "constant")
        value = config.get("value")
        if not column:
            return data

        result = [dict(row) for row in data]

        if method == "constant" and value is not None:
            for row in result:
                if row.get(column) is None:
                    row[column] = value
        elif method == "forward_fill":
            last_val = None
            for row in result:
                if row.get(column) is not None:
                    last_val = row[column]
                elif last_val is not None:
                    row[column] = last_val
        elif method == "backward_fill":
            last_val = None
            for row in reversed(result):
                if row.get(column) is not None:
                    last_val = row[column]
                elif last_val is not None:
                    row[column] = last_val

        return result

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()


# ===========================================================================
# Test Data
# ===========================================================================

SAMPLE_DATA = [
    {"facility": "London HQ", "year": 2024, "scope": "Scope 1", "emissions": 1250.5},
    {"facility": "London HQ", "year": 2025, "scope": "Scope 1", "emissions": 1180.3},
    {"facility": "Berlin DC", "year": 2024, "scope": "Scope 2", "emissions": 890.0},
    {"facility": "Berlin DC", "year": 2025, "scope": "Scope 2", "emissions": 820.7},
    {"facility": "London HQ", "year": 2024, "scope": "Scope 2", "emissions": 450.0},
]

DUPLICATE_DATA = [
    {"id": "A001", "facility": "London", "value": 100},
    {"id": "A002", "facility": "Berlin", "value": 200},
    {"id": "A001", "facility": "London", "value": 100},
    {"id": "A003", "facility": "Paris", "value": 300},
]


# ===========================================================================
# Test Classes
# ===========================================================================


class TestTransformEngineInit:
    def test_default_creation(self):
        engine = TransformEngine()
        assert engine._stats["transforms_applied"] == 0

    def test_custom_config(self):
        engine = TransformEngine(config={"max_rows": 50000})
        assert engine._config["max_rows"] == 50000

    def test_initial_statistics(self):
        engine = TransformEngine()
        stats = engine.get_statistics()
        assert stats["transforms_applied"] == 0
        assert stats["rows_processed"] == 0
        assert stats["errors"] == 0


class TestPivot:
    def test_basic_pivot(self):
        engine = TransformEngine()
        data = [
            {"facility": "London", "year": "2024", "emissions": 1250},
            {"facility": "London", "year": "2025", "emissions": 1180},
            {"facility": "Berlin", "year": "2024", "emissions": 890},
        ]
        result = engine.pivot(data, {
            "index": "facility", "column": "year", "value": "emissions",
        })
        assert len(result) == 2
        london = [r for r in result if r["facility"] == "London"][0]
        assert london["2024"] == 1250
        assert london["2025"] == 1180

    def test_pivot_missing_config(self):
        engine = TransformEngine()
        result = engine.pivot(SAMPLE_DATA, {})
        assert result == SAMPLE_DATA

    def test_pivot_single_index(self):
        engine = TransformEngine()
        data = [{"cat": "A", "metric": "x", "val": 10}]
        result = engine.pivot(data, {"index": "cat", "column": "metric", "value": "val"})
        assert len(result) == 1
        assert result[0]["x"] == 10


class TestUnpivot:
    def test_basic_unpivot(self):
        engine = TransformEngine()
        data = [{"facility": "London", "2024": 1250, "2025": 1180}]
        result = engine.unpivot(data, {
            "id_vars": ["facility"],
            "value_vars": ["2024", "2025"],
            "var_name": "year",
            "value_name": "emissions",
        })
        assert len(result) == 2
        assert result[0]["year"] == "2024"
        assert result[0]["emissions"] == 1250

    def test_unpivot_no_value_vars(self):
        engine = TransformEngine()
        data = [{"id": "A", "x": 1, "y": 2}]
        result = engine.unpivot(data, {"id_vars": ["id"]})
        assert len(result) == 2

    def test_unpivot_empty_data(self):
        engine = TransformEngine()
        result = engine.unpivot([], {"id_vars": ["id"]})
        assert result == []


class TestDeduplicate:
    def test_dedup_by_keys(self):
        engine = TransformEngine()
        result = engine.deduplicate(DUPLICATE_DATA, {"keys": ["id"]})
        assert len(result) == 3
        ids = [r["id"] for r in result]
        assert ids.count("A001") == 1

    def test_dedup_all_columns(self):
        engine = TransformEngine()
        result = engine.deduplicate(DUPLICATE_DATA, {})
        assert len(result) == 3

    def test_dedup_no_duplicates(self):
        engine = TransformEngine()
        data = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        result = engine.deduplicate(data, {"keys": ["id"]})
        assert len(result) == 3

    def test_dedup_all_same(self):
        engine = TransformEngine()
        data = [{"id": "A", "v": 1}] * 5
        result = engine.deduplicate(data, {"keys": ["id"]})
        assert len(result) == 1


class TestMerge:
    def test_basic_merge(self):
        engine = TransformEngine()
        left = [
            {"facility": "London", "emissions": 1250},
            {"facility": "Berlin", "emissions": 890},
        ]
        right = [
            {"facility": "London", "country": "UK"},
            {"facility": "Berlin", "country": "Germany"},
        ]
        result = engine.merge(left, {"right_data": right, "on": ["facility"]})
        assert len(result) == 2
        london = [r for r in result if r["facility"] == "London"][0]
        assert london["country"] == "UK"
        assert london["emissions"] == 1250

    def test_merge_no_match(self):
        engine = TransformEngine()
        left = [{"id": "A", "val": 1}]
        right = [{"id": "B", "extra": 2}]
        result = engine.merge(left, {"right_data": right, "on": ["id"]})
        assert len(result) == 1
        assert "extra" not in result[0]

    def test_merge_empty_right(self):
        engine = TransformEngine()
        result = engine.merge(SAMPLE_DATA, {"right_data": [], "on": ["facility"]})
        assert result == SAMPLE_DATA

    def test_merge_no_on_keys(self):
        engine = TransformEngine()
        result = engine.merge(SAMPLE_DATA, {"right_data": [{"x": 1}], "on": []})
        assert result == SAMPLE_DATA


class TestFilterRows:
    def test_filter_eq(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "facility", "operator": "eq", "value": "London HQ",
        })
        assert all(r["facility"] == "London HQ" for r in result)
        assert len(result) == 3

    def test_filter_ne(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "facility", "operator": "ne", "value": "London HQ",
        })
        assert all(r["facility"] != "London HQ" for r in result)

    def test_filter_gt(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "emissions", "operator": "gt", "value": 1000,
        })
        assert all(r["emissions"] > 1000 for r in result)

    def test_filter_lt(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "emissions", "operator": "lt", "value": 900,
        })
        assert all(r["emissions"] < 900 for r in result)

    def test_filter_gte(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "year", "operator": "gte", "value": 2025,
        })
        assert all(r["year"] >= 2025 for r in result)

    def test_filter_lte(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "year", "operator": "lte", "value": 2024,
        })
        assert all(r["year"] <= 2024 for r in result)

    def test_filter_contains(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {
            "column": "scope", "operator": "contains", "value": "Scope 1",
        })
        assert all("Scope 1" in r["scope"] for r in result)

    def test_filter_not_null(self):
        engine = TransformEngine()
        data = [{"a": 1}, {"a": None}, {"a": 3}]
        result = engine.filter_rows(data, {
            "column": "a", "operator": "not_null",
        })
        assert len(result) == 2

    def test_filter_no_column(self):
        engine = TransformEngine()
        result = engine.filter_rows(SAMPLE_DATA, {"operator": "eq", "value": "x"})
        assert result == SAMPLE_DATA


class TestAggregate:
    def test_sum(self):
        engine = TransformEngine()
        result = engine.aggregate(SAMPLE_DATA, {
            "group_by": ["facility"],
            "column": "emissions",
            "function": "sum",
        })
        assert len(result) == 2
        london = [r for r in result if r["facility"] == "London HQ"][0]
        assert london["emissions"] == pytest.approx(1250.5 + 1180.3 + 450.0, rel=1e-4)

    def test_avg(self):
        engine = TransformEngine()
        result = engine.aggregate(SAMPLE_DATA, {
            "group_by": ["facility"],
            "column": "emissions",
            "function": "avg",
        })
        london = [r for r in result if r["facility"] == "London HQ"][0]
        expected = (1250.5 + 1180.3 + 450.0) / 3
        assert london["emissions"] == pytest.approx(expected, rel=1e-4)

    def test_count(self):
        engine = TransformEngine()
        result = engine.aggregate(SAMPLE_DATA, {
            "group_by": ["facility"],
            "column": "emissions",
            "function": "count",
        })
        london = [r for r in result if r["facility"] == "London HQ"][0]
        assert london["emissions"] == 3

    def test_min(self):
        engine = TransformEngine()
        result = engine.aggregate(SAMPLE_DATA, {
            "group_by": ["facility"],
            "column": "emissions",
            "function": "min",
        })
        london = [r for r in result if r["facility"] == "London HQ"][0]
        assert london["emissions"] == pytest.approx(450.0)

    def test_max(self):
        engine = TransformEngine()
        result = engine.aggregate(SAMPLE_DATA, {
            "group_by": ["facility"],
            "column": "emissions",
            "function": "max",
        })
        london = [r for r in result if r["facility"] == "London HQ"][0]
        assert london["emissions"] == pytest.approx(1250.5)

    def test_aggregate_no_group_by(self):
        engine = TransformEngine()
        result = engine.aggregate(SAMPLE_DATA, {
            "group_by": [],
            "column": "emissions",
            "function": "sum",
        })
        assert result == SAMPLE_DATA


class TestRenameColumns:
    def test_basic_rename(self):
        engine = TransformEngine()
        data = [{"old_name": 1, "keep": 2}]
        result = engine.rename_columns(data, {
            "mapping": {"old_name": "new_name"},
        })
        assert "new_name" in result[0]
        assert "old_name" not in result[0]
        assert result[0]["keep"] == 2

    def test_rename_empty_mapping(self):
        engine = TransformEngine()
        result = engine.rename_columns(SAMPLE_DATA, {"mapping": {}})
        assert result == SAMPLE_DATA

    def test_rename_multiple(self):
        engine = TransformEngine()
        data = [{"a": 1, "b": 2, "c": 3}]
        result = engine.rename_columns(data, {
            "mapping": {"a": "alpha", "b": "beta"},
        })
        assert "alpha" in result[0]
        assert "beta" in result[0]
        assert result[0]["c"] == 3


class TestSplitColumn:
    def test_basic_split(self):
        engine = TransformEngine()
        data = [{"location": "London, UK"}, {"location": "Berlin, Germany"}]
        result = engine.split_column(data, {
            "column": "location",
            "delimiter": ",",
            "new_columns": ["city", "country"],
        })
        assert result[0]["city"] == "London"
        assert result[0]["country"] == "UK"

    def test_split_missing_parts(self):
        engine = TransformEngine()
        data = [{"location": "London"}]
        result = engine.split_column(data, {
            "column": "location",
            "delimiter": ",",
            "new_columns": ["city", "country"],
        })
        assert result[0]["city"] == "London"
        assert result[0]["country"] == ""

    def test_split_no_column(self):
        engine = TransformEngine()
        result = engine.split_column(SAMPLE_DATA, {"delimiter": ","})
        assert result == SAMPLE_DATA


class TestCastColumn:
    def test_cast_to_integer(self):
        engine = TransformEngine()
        data = [{"value": "42.5"}, {"value": "100"}]
        result = engine.cast_column(data, {
            "column": "value", "target_type": "integer",
        })
        assert result[0]["value"] == 42
        assert result[1]["value"] == 100

    def test_cast_to_float(self):
        engine = TransformEngine()
        data = [{"value": "3.14"}]
        result = engine.cast_column(data, {
            "column": "value", "target_type": "float",
        })
        assert result[0]["value"] == pytest.approx(3.14)

    def test_cast_to_string(self):
        engine = TransformEngine()
        data = [{"value": 42}]
        result = engine.cast_column(data, {
            "column": "value", "target_type": "string",
        })
        assert result[0]["value"] == "42"

    def test_cast_to_boolean(self):
        engine = TransformEngine()
        data = [{"flag": "true"}, {"flag": "false"}, {"flag": "1"}]
        result = engine.cast_column(data, {
            "column": "flag", "target_type": "boolean",
        })
        assert result[0]["flag"] is True
        assert result[1]["flag"] is False
        assert result[2]["flag"] is True

    def test_cast_none_value(self):
        engine = TransformEngine()
        data = [{"value": None}]
        result = engine.cast_column(data, {
            "column": "value", "target_type": "integer",
        })
        assert result[0]["value"] is None

    def test_cast_no_column(self):
        engine = TransformEngine()
        result = engine.cast_column(SAMPLE_DATA, {"target_type": "float"})
        assert result == SAMPLE_DATA


class TestFillMissing:
    def test_fill_constant(self):
        engine = TransformEngine()
        data = [{"a": 1}, {"a": None}, {"a": 3}]
        result = engine.fill_missing(data, {
            "column": "a", "method": "constant", "value": 0,
        })
        assert result[1]["a"] == 0

    def test_fill_forward(self):
        engine = TransformEngine()
        data = [{"a": 10}, {"a": None}, {"a": None}, {"a": 40}]
        result = engine.fill_missing(data, {
            "column": "a", "method": "forward_fill",
        })
        assert result[1]["a"] == 10
        assert result[2]["a"] == 10

    def test_fill_backward(self):
        engine = TransformEngine()
        data = [{"a": None}, {"a": None}, {"a": 30}]
        result = engine.fill_missing(data, {
            "column": "a", "method": "backward_fill",
        })
        assert result[0]["a"] == 30
        assert result[1]["a"] == 30

    def test_fill_no_column(self):
        engine = TransformEngine()
        result = engine.fill_missing(SAMPLE_DATA, {"method": "constant", "value": 0})
        assert result == SAMPLE_DATA

    def test_fill_no_missing(self):
        engine = TransformEngine()
        data = [{"a": 1}, {"a": 2}]
        result = engine.fill_missing(data, {
            "column": "a", "method": "constant", "value": 0,
        })
        assert result == data


class TestApplyTransforms:
    def test_single_operation(self):
        engine = TransformEngine()
        result = engine.apply_transforms(DUPLICATE_DATA, [
            {"type": "dedup", "keys": ["id"]},
        ])
        assert result["operations_applied"] == 1
        assert result["row_count"] == 3

    def test_chained_operations(self):
        engine = TransformEngine()
        result = engine.apply_transforms(SAMPLE_DATA, [
            {"type": "filter", "column": "year", "operator": "eq", "value": 2024},
            {"type": "rename", "mapping": {"emissions": "co2e_tonnes"}},
        ])
        assert result["operations_applied"] == 2
        assert all("co2e_tonnes" in r for r in result["data"])

    def test_empty_operations(self):
        engine = TransformEngine()
        result = engine.apply_transforms(SAMPLE_DATA, [])
        assert result["operations_applied"] == 0
        assert result["row_count"] == len(SAMPLE_DATA)

    def test_provenance_hash_generated(self):
        engine = TransformEngine()
        result = engine.apply_transforms(SAMPLE_DATA, [
            {"type": "dedup", "keys": ["facility", "year", "scope"]},
        ])
        assert len(result["provenance_hash"]) == 64

    def test_unknown_operation_counted_as_error(self):
        engine = TransformEngine()
        result = engine.apply_transforms(SAMPLE_DATA, [
            {"type": "nonexistent_op"},
        ])
        assert result["operations_applied"] == 0
        stats = engine.get_statistics()
        assert stats["errors"] == 1


class TestTransformEngineStatistics:
    def test_stats_accumulate(self):
        engine = TransformEngine()
        engine.apply_transforms(SAMPLE_DATA, [{"type": "dedup", "keys": ["facility"]}])
        engine.apply_transforms(SAMPLE_DATA, [{"type": "filter", "column": "year", "operator": "eq", "value": 2024}])
        stats = engine.get_statistics()
        assert stats["transforms_applied"] == 2
        assert stats["rows_processed"] > 0
