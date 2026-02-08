# -*- coding: utf-8 -*-
"""
Transform Engine - AGENT-DATA-002: Excel/CSV Normalizer

Data transformation engine that applies structural and value
transformations to normalised spreadsheet data with provenance tracking.

Supports:
    - Pivot and unpivot (wide-to-long) operations
    - Deduplication with configurable key columns and keep strategy
    - Merge/join across two datasets (inner, left, right, outer)
    - Row filtering with condition dictionaries
    - Aggregation by group with sum, avg, min, max, count
    - Column renaming
    - Column splitting by delimiter
    - Type casting with error handling
    - Missing value filling (none, zero, mean, median, forward, backward, value)
    - Date format normalisation
    - Unit normalisation (stub delegating to AGENT-FOUND-003)
    - SHA-256 provenance for every transform operation
    - Thread-safe statistics

Zero-Hallucination Guarantees:
    - All transforms are deterministic arithmetic or structural operations
    - No LLM calls in the transformation path
    - Provenance hashes track every transformation applied

Example:
    >>> from greenlang.excel_normalizer.transform_engine import TransformEngine
    >>> engine = TransformEngine()
    >>> result = engine.deduplicate(rows, key_columns=["id"])
    >>> print(result.rows_output, result.rows_removed)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel/CSV Normalizer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "TransformResult",
    "TransformEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, input_hash: str, params: str) -> str:
    """Compute SHA-256 provenance hash for a transform operation.

    Args:
        operation: Transform operation name.
        input_hash: Hash of input data.
        params: Serialised operation parameters.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    data = f"{operation}:{input_hash}:{params}"
    return hashlib.sha256(data.encode()).hexdigest()


def _hash_rows(rows: List[Dict[str, Any]]) -> str:
    """Compute SHA-256 hash of a row list for provenance.

    Args:
        rows: List of data row dictionaries.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    content = str([(sorted(r.items()) if isinstance(r, dict) else r) for r in rows[:100]])
    return hashlib.sha256(content.encode()).hexdigest()


# Date formats to try for normalisation
_DATE_FORMATS: List[str] = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
    "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%m.%d.%Y", "%d.%m.%Y",
    "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y",
    "%m/%d/%y", "%d/%m/%y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TransformResult(BaseModel):
    """Result of a transformation operation."""

    result_id: str = Field(
        default_factory=lambda: f"tx-{uuid.uuid4().hex[:12]}",
        description="Unique result identifier",
    )
    operation: str = Field(..., description="Transform operation name")
    rows_input: int = Field(default=0, ge=0, description="Input row count")
    rows_output: int = Field(default=0, ge=0, description="Output row count")
    rows_added: int = Field(default=0, ge=0, description="Rows added")
    rows_removed: int = Field(default=0, ge=0, description="Rows removed")
    rows_modified: int = Field(default=0, ge=0, description="Rows modified")
    output_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Transformed output rows",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Transform timestamp",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# TransformEngine
# ---------------------------------------------------------------------------


class TransformEngine:
    """Data transformation engine with provenance tracking.

    Applies structural and value transformations to normalised data
    rows. Every operation produces a TransformResult with provenance
    hash and processing metrics.

    Attributes:
        _config: Configuration dictionary.
        _lock: Threading lock for statistics.
        _stats: Transform statistics.

    Example:
        >>> engine = TransformEngine()
        >>> result = engine.deduplicate(rows, key_columns=["id"])
        >>> print(result.rows_output)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise TransformEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_output_rows``: int (default 1000000)
        """
        self._config = config or {}
        self._max_output: int = self._config.get("max_output_rows", 1_000_000)
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "transforms_applied": 0,
            "pivot_count": 0,
            "unpivot_count": 0,
            "deduplicate_count": 0,
            "merge_count": 0,
            "filter_count": 0,
            "aggregate_count": 0,
            "rename_count": 0,
            "split_count": 0,
            "cast_count": 0,
            "fill_count": 0,
            "date_normalize_count": 0,
            "unit_normalize_count": 0,
            "rows_input": 0,
            "rows_output": 0,
            "errors": 0,
        }
        logger.info("TransformEngine initialised: max_output=%d", self._max_output)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_transforms(
        self,
        rows: List[Dict[str, Any]],
        operations: List[Dict[str, Any]],
    ) -> TransformResult:
        """Apply a chain of transform operations sequentially.

        Args:
            rows: Input data rows.
            operations: List of operation dicts, each with "operation" key
                and operation-specific parameters.

        Returns:
            TransformResult with final output.
        """
        start = time.monotonic()
        current_rows = list(rows)

        for op in operations:
            op_name = op.get("operation", "unknown")
            try:
                result = self._dispatch_operation(current_rows, op)
                current_rows = result.output_data
            except Exception as exc:
                logger.error("Transform '%s' failed: %s", op_name, exc)
                with self._lock:
                    self._stats["errors"] += 1

        elapsed = (time.monotonic() - start) * 1000
        provenance = _compute_provenance(
            "chain", _hash_rows(rows), str(len(operations)),
        )

        return TransformResult(
            operation="chain",
            rows_input=len(rows),
            rows_output=len(current_rows),
            output_data=current_rows,
            provenance_hash=provenance,
            processing_time_ms=round(elapsed, 2),
        )

    def pivot(
        self,
        rows: List[Dict[str, Any]],
        index_cols: List[str],
        pivot_col: str,
        value_col: str,
    ) -> List[Dict[str, Any]]:
        """Pivot rows from long to wide format.

        Groups rows by index_cols and creates new columns from
        unique values in pivot_col, filling with values from value_col.

        Args:
            rows: Input data rows.
            index_cols: Columns to group by (row identifiers).
            pivot_col: Column whose values become new column names.
            value_col: Column whose values fill the new columns.

        Returns:
            List of pivoted row dictionaries.
        """
        start = time.monotonic()

        groups: Dict[tuple, Dict[str, Any]] = {}
        for row in rows:
            key = tuple(row.get(c, "") for c in index_cols)
            if key not in groups:
                groups[key] = {c: row.get(c, "") for c in index_cols}
            pval = str(row.get(pivot_col, ""))
            groups[key][pval] = row.get(value_col)

        output = list(groups.values())

        self._update_stats("pivot", len(rows), len(output))
        logger.info(
            "Pivot: %d rows -> %d rows (%.1f ms)",
            len(rows), len(output), (time.monotonic() - start) * 1000,
        )
        return output

    def unpivot(
        self,
        rows: List[Dict[str, Any]],
        id_cols: List[str],
        value_cols: List[str],
        var_name: str = "variable",
        value_name: str = "value",
    ) -> List[Dict[str, Any]]:
        """Unpivot rows from wide to long format.

        Converts columns in value_cols into rows, preserving id_cols.

        Args:
            rows: Input data rows.
            id_cols: Columns to preserve as identifiers.
            value_cols: Columns to unpivot into rows.
            var_name: Name for the variable column.
            value_name: Name for the value column.

        Returns:
            List of unpivoted row dictionaries.
        """
        start = time.monotonic()

        output: List[Dict[str, Any]] = []
        for row in rows:
            base = {c: row.get(c) for c in id_cols}
            for col in value_cols:
                new_row = dict(base)
                new_row[var_name] = col
                new_row[value_name] = row.get(col)
                output.append(new_row)

        self._update_stats("unpivot", len(rows), len(output))
        logger.info(
            "Unpivot: %d rows -> %d rows (%.1f ms)",
            len(rows), len(output), (time.monotonic() - start) * 1000,
        )
        return output

    def deduplicate(
        self,
        rows: List[Dict[str, Any]],
        key_columns: Optional[List[str]] = None,
        keep: str = "first",
    ) -> List[Dict[str, Any]]:
        """Remove duplicate rows.

        Args:
            rows: Input data rows.
            key_columns: Columns to use as duplicate key. None = all columns.
            keep: Which duplicate to keep ("first" or "last").

        Returns:
            List of deduplicated rows.
        """
        start = time.monotonic()

        seen: Dict[str, int] = {}
        result: List[Dict[str, Any]] = []

        iterable = enumerate(rows) if keep == "first" else enumerate(reversed(rows))

        for idx, row in iterable:
            if key_columns:
                key = tuple(str(row.get(c, "")) for c in key_columns)
            else:
                key = tuple(str(v) for v in row.values())

            row_hash = hashlib.sha256(str(key).encode()).hexdigest()
            if row_hash not in seen:
                seen[row_hash] = idx
                result.append(row)

        if keep == "last":
            result.reverse()

        self._update_stats("deduplicate", len(rows), len(result))
        logger.info(
            "Deduplicate: %d rows -> %d rows, %d removed (%.1f ms)",
            len(rows), len(result), len(rows) - len(result),
            (time.monotonic() - start) * 1000,
        )
        return result

    def merge(
        self,
        rows1: List[Dict[str, Any]],
        rows2: List[Dict[str, Any]],
        on: List[str],
        how: str = "inner",
    ) -> List[Dict[str, Any]]:
        """Merge two datasets by key columns.

        Args:
            rows1: Left dataset.
            rows2: Right dataset.
            on: Key columns to join on.
            how: Join type ("inner", "left", "right", "outer").

        Returns:
            List of merged row dictionaries.
        """
        start = time.monotonic()

        # Index rows2 by key
        right_index: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in rows2:
            key = tuple(str(row.get(c, "")) for c in on)
            right_index.setdefault(key, []).append(row)

        output: List[Dict[str, Any]] = []
        left_matched: set = set()

        for row1 in rows1:
            key = tuple(str(row1.get(c, "")) for c in on)
            matches = right_index.get(key, [])

            if matches:
                left_matched.add(key)
                for row2 in matches:
                    merged = dict(row1)
                    for k, v in row2.items():
                        if k not in merged:
                            merged[k] = v
                    output.append(merged)
            elif how in ("left", "outer"):
                output.append(dict(row1))

        if how in ("right", "outer"):
            for key, right_rows in right_index.items():
                if key not in left_matched:
                    output.extend(right_rows)

        self._update_stats("merge", len(rows1) + len(rows2), len(output))
        logger.info(
            "Merge (%s): %d + %d -> %d rows (%.1f ms)",
            how, len(rows1), len(rows2), len(output),
            (time.monotonic() - start) * 1000,
        )
        return output

    def filter_rows(
        self,
        rows: List[Dict[str, Any]],
        conditions: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Filter rows by conditions.

        Conditions dict maps field names to expected values or
        filter expressions. Simple equality is used for direct values.
        Prefix with '>' '<' '>=' '<=' '!=' for comparison.

        Args:
            rows: Input data rows.
            conditions: Dict of field_name -> condition.

        Returns:
            List of matching rows.
        """
        start = time.monotonic()

        output: List[Dict[str, Any]] = []
        for row in rows:
            if self._matches_conditions(row, conditions):
                output.append(row)

        self._update_stats("filter", len(rows), len(output))
        logger.info(
            "Filter: %d rows -> %d rows (%.1f ms)",
            len(rows), len(output), (time.monotonic() - start) * 1000,
        )
        return output

    def aggregate(
        self,
        rows: List[Dict[str, Any]],
        group_by: List[str],
        agg_rules: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Group rows and apply aggregation functions.

        Args:
            rows: Input data rows.
            group_by: Columns to group by.
            agg_rules: Dict of column -> function ("sum", "avg", "min",
                       "max", "count").

        Returns:
            List of aggregated row dictionaries.
        """
        start = time.monotonic()

        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for row in rows:
            key = tuple(str(row.get(c, "")) for c in group_by)
            groups.setdefault(key, []).append(row)

        output: List[Dict[str, Any]] = []
        for key, group_rows in groups.items():
            agg_row: Dict[str, Any] = {}
            for i, col in enumerate(group_by):
                agg_row[col] = key[i]

            for col, func in agg_rules.items():
                values = self._extract_floats(group_rows, col)
                agg_row[col] = self._apply_agg(values, func)

            output.append(agg_row)

        self._update_stats("aggregate", len(rows), len(output))
        logger.info(
            "Aggregate: %d rows -> %d groups (%.1f ms)",
            len(rows), len(output), (time.monotonic() - start) * 1000,
        )
        return output

    def rename_columns(
        self,
        rows: List[Dict[str, Any]],
        rename_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Rename columns in all rows.

        Args:
            rows: Input data rows.
            rename_map: Dict of old_name -> new_name.

        Returns:
            List of rows with renamed columns.
        """
        output: List[Dict[str, Any]] = []
        for row in rows:
            new_row = {}
            for k, v in row.items():
                new_key = rename_map.get(k, k)
                new_row[new_key] = v
            output.append(new_row)

        self._update_stats("rename", len(rows), len(output))
        return output

    def split_column(
        self,
        rows: List[Dict[str, Any]],
        column: str,
        delimiter: str,
        new_columns: List[str],
    ) -> List[Dict[str, Any]]:
        """Split a column into multiple columns by delimiter.

        Args:
            rows: Input data rows.
            column: Column to split.
            delimiter: Delimiter string.
            new_columns: Names for the resulting columns.

        Returns:
            List of rows with the split columns.
        """
        output: List[Dict[str, Any]] = []
        for row in rows:
            new_row = dict(row)
            value = str(row.get(column, ""))
            parts = value.split(delimiter)

            for i, col_name in enumerate(new_columns):
                new_row[col_name] = parts[i].strip() if i < len(parts) else ""

            output.append(new_row)

        self._update_stats("split", len(rows), len(output))
        return output

    def cast_column(
        self,
        rows: List[Dict[str, Any]],
        column: str,
        target_type: str,
    ) -> List[Dict[str, Any]]:
        """Type-cast a column to a target type.

        Args:
            rows: Input data rows.
            column: Column to cast.
            target_type: Target type ("int", "float", "str", "bool").

        Returns:
            List of rows with the cast column.
        """
        output: List[Dict[str, Any]] = []
        for row in rows:
            new_row = dict(row)
            value = row.get(column)
            new_row[column] = self._cast_value(value, target_type)
            output.append(new_row)

        self._update_stats("cast", len(rows), len(output))
        return output

    def fill_missing(
        self,
        rows: List[Dict[str, Any]],
        strategy: str = "none",
        fill_value: Any = None,
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fill missing values using the specified strategy.

        Args:
            rows: Input data rows.
            strategy: Fill strategy ("none", "zero", "mean", "median",
                      "forward", "backward", "value").
            fill_value: Value to use when strategy is "value".
            columns: Columns to fill. None = all columns.

        Returns:
            List of rows with filled values.
        """
        start = time.monotonic()

        if not rows:
            return []

        target_cols = columns or list(rows[0].keys())
        output = [dict(row) for row in rows]

        for col in target_cols:
            if strategy == "none":
                continue
            elif strategy == "zero":
                self._fill_constant(output, col, 0)
            elif strategy == "value":
                self._fill_constant(output, col, fill_value)
            elif strategy == "mean":
                values = self._extract_floats(rows, col)
                if values:
                    mean_val = sum(values) / len(values)
                    self._fill_constant(output, col, round(mean_val, 6))
            elif strategy == "median":
                values = sorted(self._extract_floats(rows, col))
                if values:
                    n = len(values)
                    median_val = (
                        values[n // 2] if n % 2 == 1
                        else (values[n // 2 - 1] + values[n // 2]) / 2
                    )
                    self._fill_constant(output, col, round(median_val, 6))
            elif strategy == "forward":
                self._fill_forward(output, col)
            elif strategy == "backward":
                self._fill_backward(output, col)

        self._update_stats("fill", len(rows), len(output))
        logger.info(
            "Fill missing (%s): %d rows, %d columns (%.1f ms)",
            strategy, len(rows), len(target_cols),
            (time.monotonic() - start) * 1000,
        )
        return output

    def normalize_dates(
        self,
        rows: List[Dict[str, Any]],
        date_columns: List[str],
        target_format: str = "ISO",
    ) -> List[Dict[str, Any]]:
        """Normalise date columns to a consistent format.

        Args:
            rows: Input data rows.
            date_columns: Columns containing date values.
            target_format: Target format ("ISO" for YYYY-MM-DD).

        Returns:
            List of rows with normalised dates.
        """
        output_fmt = "%Y-%m-%d" if target_format == "ISO" else target_format
        output: List[Dict[str, Any]] = []

        for row in rows:
            new_row = dict(row)
            for col in date_columns:
                value = row.get(col)
                if value is not None and str(value).strip():
                    parsed = self._parse_date(str(value).strip())
                    if parsed:
                        new_row[col] = parsed.strftime(output_fmt)
            output.append(new_row)

        self._update_stats("date_normalize", len(rows), len(output))
        return output

    def normalize_units(
        self,
        rows: List[Dict[str, Any]],
        unit_columns: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Normalise units (stub - delegates to AGENT-FOUND-003).

        This is a placeholder that logs the intent. Full unit conversion
        is delegated to the Unit & Reference Normalizer agent.

        Args:
            rows: Input data rows.
            unit_columns: Dict of column -> target_unit.

        Returns:
            Input rows unchanged (stub).
        """
        logger.info(
            "Unit normalisation requested for %d columns "
            "(delegate to AGENT-FOUND-003)",
            len(unit_columns),
        )
        self._update_stats("unit_normalize", len(rows), len(rows))
        return rows

    def get_statistics(self) -> Dict[str, Any]:
        """Return transform statistics.

        Returns:
            Dictionary with per-operation counts and totals.
        """
        with self._lock:
            return {
                "transforms_applied": self._stats["transforms_applied"],
                "by_operation": {
                    "pivot": self._stats["pivot_count"],
                    "unpivot": self._stats["unpivot_count"],
                    "deduplicate": self._stats["deduplicate_count"],
                    "merge": self._stats["merge_count"],
                    "filter": self._stats["filter_count"],
                    "aggregate": self._stats["aggregate_count"],
                    "rename": self._stats["rename_count"],
                    "split": self._stats["split_count"],
                    "cast": self._stats["cast_count"],
                    "fill": self._stats["fill_count"],
                    "date_normalize": self._stats["date_normalize_count"],
                    "unit_normalize": self._stats["unit_normalize_count"],
                },
                "rows_input": self._stats["rows_input"],
                "rows_output": self._stats["rows_output"],
                "errors": self._stats["errors"],
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _update_stats(self, operation: str, rows_in: int, rows_out: int) -> None:
        """Update statistics for a transform operation.

        Args:
            operation: Operation name.
            rows_in: Input row count.
            rows_out: Output row count.
        """
        with self._lock:
            self._stats["transforms_applied"] += 1
            stat_key = f"{operation}_count"
            if stat_key in self._stats:
                self._stats[stat_key] += 1
            self._stats["rows_input"] += rows_in
            self._stats["rows_output"] += rows_out

    def _dispatch_operation(
        self,
        rows: List[Dict[str, Any]],
        op: Dict[str, Any],
    ) -> TransformResult:
        """Dispatch a single operation from a chain.

        Args:
            rows: Input rows.
            op: Operation config dict.

        Returns:
            TransformResult with output data.
        """
        op_name = op.get("operation", "unknown")
        start = time.monotonic()
        output = rows

        if op_name == "deduplicate":
            output = self.deduplicate(
                rows,
                key_columns=op.get("key_columns"),
                keep=op.get("keep", "first"),
            )
        elif op_name == "filter":
            output = self.filter_rows(rows, conditions=op.get("conditions", {}))
        elif op_name == "rename":
            output = self.rename_columns(rows, rename_map=op.get("rename_map", {}))
        elif op_name == "fill":
            output = self.fill_missing(
                rows,
                strategy=op.get("strategy", "none"),
                fill_value=op.get("fill_value"),
                columns=op.get("columns"),
            )
        elif op_name == "cast":
            output = self.cast_column(
                rows,
                column=op.get("column", ""),
                target_type=op.get("target_type", "str"),
            )
        elif op_name == "split":
            output = self.split_column(
                rows,
                column=op.get("column", ""),
                delimiter=op.get("delimiter", ","),
                new_columns=op.get("new_columns", []),
            )
        elif op_name == "aggregate":
            output = self.aggregate(
                rows,
                group_by=op.get("group_by", []),
                agg_rules=op.get("agg_rules", {}),
            )
        elif op_name == "normalize_dates":
            output = self.normalize_dates(
                rows,
                date_columns=op.get("date_columns", []),
                target_format=op.get("target_format", "ISO"),
            )
        else:
            logger.warning("Unknown transform operation: '%s'", op_name)

        elapsed = (time.monotonic() - start) * 1000
        provenance = _compute_provenance(op_name, _hash_rows(rows), str(op))

        return TransformResult(
            operation=op_name,
            rows_input=len(rows),
            rows_output=len(output),
            output_data=output,
            provenance_hash=provenance,
            processing_time_ms=round(elapsed, 2),
        )

    def _matches_conditions(
        self,
        row: Dict[str, Any],
        conditions: Dict[str, Any],
    ) -> bool:
        """Check if a row matches all filter conditions.

        Args:
            row: Data row.
            conditions: Condition dict.

        Returns:
            True if all conditions match.
        """
        for field, condition in conditions.items():
            value = row.get(field)
            if isinstance(condition, str) and len(condition) > 1:
                # Check comparison operators
                if condition.startswith(">="):
                    if not self._compare_numeric(value, condition[2:], ">="):
                        return False
                    continue
                elif condition.startswith("<="):
                    if not self._compare_numeric(value, condition[2:], "<="):
                        return False
                    continue
                elif condition.startswith("!="):
                    if str(value) == condition[2:].strip():
                        return False
                    continue
                elif condition.startswith(">"):
                    if not self._compare_numeric(value, condition[1:], ">"):
                        return False
                    continue
                elif condition.startswith("<"):
                    if not self._compare_numeric(value, condition[1:], "<"):
                        return False
                    continue

            # Equality check
            if str(value) != str(condition):
                return False

        return True

    def _compare_numeric(
        self,
        value: Any,
        threshold: str,
        operator: str,
    ) -> bool:
        """Numeric comparison for filter conditions.

        Args:
            value: Row value.
            threshold: Threshold string.
            operator: Comparison operator.

        Returns:
            True if comparison holds.
        """
        try:
            val = float(str(value).replace(",", ""))
            thresh = float(threshold.strip().replace(",", ""))
        except (ValueError, TypeError, AttributeError):
            return False

        if operator == ">":
            return val > thresh
        elif operator == "<":
            return val < thresh
        elif operator == ">=":
            return val >= thresh
        elif operator == "<=":
            return val <= thresh
        return False

    def _extract_floats(
        self,
        rows: List[Dict[str, Any]],
        column: str,
    ) -> List[float]:
        """Extract numeric values from a column.

        Args:
            rows: Data rows.
            column: Column name.

        Returns:
            List of float values.
        """
        values: List[float] = []
        for row in rows:
            v = row.get(column)
            if v is None:
                continue
            try:
                values.append(float(str(v).replace(",", "")))
            except (ValueError, TypeError):
                continue
        return values

    def _apply_agg(
        self,
        values: List[float],
        func: str,
    ) -> Any:
        """Apply an aggregation function to a list of values.

        Args:
            values: Numeric values.
            func: Function name ("sum", "avg", "min", "max", "count").

        Returns:
            Aggregated result.
        """
        if not values:
            return 0 if func == "count" else None

        if func == "sum":
            return round(sum(values), 6)
        elif func == "avg":
            return round(sum(values) / len(values), 6)
        elif func == "min":
            return min(values)
        elif func == "max":
            return max(values)
        elif func == "count":
            return len(values)
        else:
            logger.warning("Unknown aggregation function: '%s'", func)
            return None

    def _cast_value(self, value: Any, target_type: str) -> Any:
        """Cast a single value to a target type.

        Args:
            value: Value to cast.
            target_type: Target type string.

        Returns:
            Cast value, or original if casting fails.
        """
        if value is None:
            return None

        try:
            if target_type == "int":
                return int(float(str(value).replace(",", "")))
            elif target_type == "float":
                return float(str(value).replace(",", ""))
            elif target_type == "str":
                return str(value)
            elif target_type == "bool":
                return str(value).strip().lower() in {
                    "true", "yes", "1", "y", "t", "on",
                }
            else:
                return value
        except (ValueError, TypeError):
            return value

    def _fill_constant(
        self,
        rows: List[Dict[str, Any]],
        column: str,
        value: Any,
    ) -> None:
        """Fill missing values in a column with a constant (in-place).

        Args:
            rows: Data rows (modified in-place).
            column: Column name.
            value: Fill value.
        """
        for row in rows:
            if row.get(column) is None or str(row.get(column, "")).strip() == "":
                row[column] = value

    def _fill_forward(
        self,
        rows: List[Dict[str, Any]],
        column: str,
    ) -> None:
        """Forward-fill missing values in a column (in-place).

        Args:
            rows: Data rows (modified in-place).
            column: Column name.
        """
        last_value = None
        for row in rows:
            val = row.get(column)
            if val is not None and str(val).strip():
                last_value = val
            elif last_value is not None:
                row[column] = last_value

    def _fill_backward(
        self,
        rows: List[Dict[str, Any]],
        column: str,
    ) -> None:
        """Backward-fill missing values in a column (in-place).

        Args:
            rows: Data rows (modified in-place).
            column: Column name.
        """
        next_value = None
        for row in reversed(rows):
            val = row.get(column)
            if val is not None and str(val).strip():
                next_value = val
            elif next_value is not None:
                row[column] = next_value

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Parse a date string using known formats.

        Args:
            value: Date string.

        Returns:
            Parsed datetime or None.
        """
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None
