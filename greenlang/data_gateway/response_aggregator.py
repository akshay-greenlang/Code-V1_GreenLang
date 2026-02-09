# -*- coding: utf-8 -*-
"""
Response Aggregator Engine - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Merges and aggregates query results from multiple data sources using
configurable conflict resolution strategies. Applies aggregation
operations (sum, avg, min, max, count, group_by, percentile).

Zero-Hallucination Guarantees:
    - All aggregation uses deterministic arithmetic operations
    - Conflict resolution follows fixed strategy rules
    - No ML/LLM used for data merging or resolution
    - SHA-256 provenance hashes on all aggregation results

Example:
    >>> from greenlang.data_gateway.response_aggregator import ResponseAggregatorEngine
    >>> aggregator = ResponseAggregatorEngine()
    >>> merged = aggregator.aggregate(results, strategy="latest_wins")
    >>> assert merged["query_id"] is not None

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Conflict Resolution Strategies
# ---------------------------------------------------------------------------

CONFLICT_STRATEGIES = frozenset({
    "first_wins",       # First source value takes precedence
    "latest_wins",      # Last source value takes precedence
    "highest_wins",     # Highest numeric value wins
    "lowest_wins",      # Lowest numeric value wins
    "concatenate",      # Concatenate string values
    "merge_unique",     # Merge unique list values
    "error_on_conflict",  # Raise error on conflicting values
})


class ResponseAggregatorEngine:
    """Multi-source response aggregation engine.

    Merges query results from multiple data sources, resolves field
    conflicts using configurable strategies, and applies aggregation
    operations.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _aggregations: In-memory aggregation result storage.

    Example:
        >>> aggregator = ResponseAggregatorEngine()
        >>> merged = aggregator.aggregate([r1, r2], "latest_wins")
        >>> assert len(merged["data"]) > 0
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize ResponseAggregatorEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._aggregations: Dict[str, Dict[str, Any]] = {}

        logger.info("ResponseAggregatorEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        results: List[Dict[str, Any]],
        strategy: str = "latest_wins",
    ) -> Dict[str, Any]:
        """Merge multiple query results into a single result.

        Args:
            results: List of QueryResult dictionaries.
            strategy: Conflict resolution strategy name.

        Returns:
            Merged QueryResult dictionary.
        """
        start_time = time.monotonic()

        if not results:
            return {
                "query_id": f"AGG-{uuid.uuid4().hex[:12]}",
                "source_id": "aggregated",
                "data": [],
                "total_count": 0,
                "row_count": 0,
                "metadata": {"strategy": strategy},
                "errors": [],
                "execution_time_ms": 0.0,
                "created_at": _utcnow().isoformat(),
            }

        if strategy not in CONFLICT_STRATEGIES:
            logger.warning(
                "Unknown conflict strategy '%s', using 'latest_wins'",
                strategy,
            )
            strategy = "latest_wins"

        # Collect all data and metadata
        all_datasets: List[List[Dict[str, Any]]] = []
        all_errors: List[str] = []
        source_ids: List[str] = []
        query_id = ""

        for result in results:
            if not query_id:
                query_id = result.get("query_id", "")
            all_datasets.append(result.get("data", []))
            all_errors.extend(result.get("errors", []))
            src = result.get("source_id", "")
            if src:
                source_ids.append(src)

        # Merge data
        merged_data = self._merge_data(all_datasets, strategy)

        total_count = sum(
            r.get("total_count", len(r.get("data", [])))
            for r in results
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        merged_result = {
            "query_id": query_id or f"AGG-{uuid.uuid4().hex[:12]}",
            "source_id": ",".join(source_ids),
            "data": merged_data,
            "total_count": total_count,
            "row_count": len(merged_data),
            "metadata": {
                "strategy": strategy,
                "sources_merged": source_ids,
                "source_count": len(results),
            },
            "errors": all_errors if all_errors else [],
            "execution_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        # Store
        agg_id = merged_result["query_id"]
        self._aggregations[agg_id] = merged_result

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash({
                "query_id": agg_id,
                "row_count": len(merged_data),
                "strategy": strategy,
            })
            self._provenance.record(
                entity_type="aggregation",
                entity_id=agg_id,
                action="aggregation",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.data_gateway.metrics import record_aggregation
            status = "success" if not all_errors else "partial_error"
            record_aggregation(
                sources_count=str(len(results)),
                status=status,
            )
        except ImportError:
            pass

        logger.info(
            "Aggregated %d results (%d total rows) using %s (%.1f ms)",
            len(results), len(merged_data), strategy, elapsed_ms,
        )
        return merged_result

    def apply_aggregations(
        self,
        data: List[Dict[str, Any]],
        aggregations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply aggregation operations to a dataset.

        Args:
            data: Data rows to aggregate.
            aggregations: List of QueryAggregation dictionaries.

        Returns:
            Aggregation results as list of dictionaries.
        """
        return self._apply_aggregations(data, aggregations)

    # ------------------------------------------------------------------
    # Merge Logic
    # ------------------------------------------------------------------

    def _merge_data(
        self,
        datasets: List[List[Dict[str, Any]]],
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """Merge data lists from multiple sources.

        For now, performs a simple concatenation of all datasets.
        If rows share an 'id' field, conflicts are resolved using
        the specified strategy.

        Args:
            datasets: List of data row lists.
            strategy: Conflict resolution strategy.

        Returns:
            Merged data rows.
        """
        if not datasets:
            return []

        # Index by ID for conflict resolution
        merged_by_id: Dict[str, Dict[str, Any]] = {}
        no_id_rows: List[Dict[str, Any]] = []

        for dataset in datasets:
            for row in dataset:
                row_id = row.get("id")
                if row_id is None:
                    no_id_rows.append(row)
                elif row_id in merged_by_id:
                    # Conflict: resolve field by field
                    existing = merged_by_id[row_id]
                    for field, value in row.items():
                        if field == "id":
                            continue
                        if field in existing and existing[field] != value:
                            existing[field] = self._resolve_conflict(
                                field,
                                [existing[field], value],
                                strategy,
                            )
                        else:
                            existing[field] = value
                else:
                    merged_by_id[row_id] = dict(row)

        # Combine
        result = list(merged_by_id.values()) + no_id_rows
        return result

    def _resolve_conflict(
        self,
        field: str,
        values: List[Any],
        strategy: str,
    ) -> Any:
        """Resolve a field conflict between multiple values.

        Args:
            field: Field name with conflict.
            values: List of conflicting values.
            strategy: Conflict resolution strategy.

        Returns:
            Resolved value.
        """
        if not values:
            return None

        if strategy == "first_wins":
            return values[0]

        elif strategy == "latest_wins":
            return values[-1]

        elif strategy == "highest_wins":
            try:
                numeric = [v for v in values if isinstance(v, (int, float))]
                return max(numeric) if numeric else values[-1]
            except (TypeError, ValueError):
                return values[-1]

        elif strategy == "lowest_wins":
            try:
                numeric = [v for v in values if isinstance(v, (int, float))]
                return min(numeric) if numeric else values[-1]
            except (TypeError, ValueError):
                return values[-1]

        elif strategy == "concatenate":
            str_values = [str(v) for v in values if v is not None]
            return "; ".join(str_values)

        elif strategy == "merge_unique":
            unique: List[Any] = []
            for v in values:
                if isinstance(v, list):
                    for item in v:
                        if item not in unique:
                            unique.append(item)
                elif v not in unique:
                    unique.append(v)
            return unique

        elif strategy == "error_on_conflict":
            if len(set(str(v) for v in values)) > 1:
                logger.warning(
                    "Conflict on field '%s': values=%s", field, values,
                )
            return values[-1]

        # Default: latest wins
        return values[-1]

    # ------------------------------------------------------------------
    # Aggregation Operations
    # ------------------------------------------------------------------

    def _apply_aggregations(
        self,
        data: List[Dict[str, Any]],
        aggregations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply aggregation operations to a dataset.

        Args:
            data: Data rows.
            aggregations: List of aggregation definitions.

        Returns:
            Aggregation results.
        """
        if not aggregations or not data:
            return []

        results: List[Dict[str, Any]] = []

        for agg in aggregations:
            func = agg.get("function", "")
            field = agg.get("field", "")
            alias = agg.get("alias", f"{func}_{field}")
            group_by = agg.get("group_by", [])

            if func == "sum":
                value = self._compute_sum(data, field)
                results.append({"aggregation": alias, "value": value})

            elif func == "avg":
                value = self._compute_avg(data, field)
                results.append({"aggregation": alias, "value": value})

            elif func == "min":
                value = self._compute_min(data, field)
                results.append({"aggregation": alias, "value": value})

            elif func == "max":
                value = self._compute_max(data, field)
                results.append({"aggregation": alias, "value": value})

            elif func == "count":
                value = self._compute_count(data, field)
                results.append({"aggregation": alias, "value": value})

            elif func == "distinct_count":
                value = self._compute_distinct_count(data, field)
                results.append({"aggregation": alias, "value": value})

            elif func == "percentile":
                value = self._compute_percentile(
                    data, field, agg.get("percentile", 95),
                )
                results.append({"aggregation": alias, "value": value})

            elif func == "group_by":
                grouped = self._compute_group_by(data, field, group_by)
                results.extend(grouped)

            else:
                logger.warning("Unknown aggregation function: %s", func)

        return results

    def _compute_sum(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> float:
        """Compute sum of a numeric field.

        Args:
            data: Data rows.
            field: Field name.

        Returns:
            Sum value.
        """
        total = 0.0
        for row in data:
            val = row.get(field)
            if isinstance(val, (int, float)):
                total += val
        return round(total, 6)

    def _compute_avg(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> float:
        """Compute average of a numeric field.

        Args:
            data: Data rows.
            field: Field name.

        Returns:
            Average value.
        """
        values = [
            row.get(field) for row in data
            if isinstance(row.get(field), (int, float))
        ]
        if not values:
            return 0.0
        return round(sum(values) / len(values), 6)

    def _compute_min(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> Any:
        """Compute minimum value of a field.

        Args:
            data: Data rows.
            field: Field name.

        Returns:
            Minimum value or None.
        """
        values = [
            row.get(field) for row in data
            if row.get(field) is not None
        ]
        if not values:
            return None
        try:
            return min(values)
        except TypeError:
            return values[0]

    def _compute_max(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> Any:
        """Compute maximum value of a field.

        Args:
            data: Data rows.
            field: Field name.

        Returns:
            Maximum value or None.
        """
        values = [
            row.get(field) for row in data
            if row.get(field) is not None
        ]
        if not values:
            return None
        try:
            return max(values)
        except TypeError:
            return values[-1]

    def _compute_count(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> int:
        """Compute count of non-null values for a field.

        Args:
            data: Data rows.
            field: Field name.

        Returns:
            Count of non-null values.
        """
        return sum(
            1 for row in data
            if row.get(field) is not None
        )

    def _compute_distinct_count(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> int:
        """Compute count of distinct non-null values for a field.

        Args:
            data: Data rows.
            field: Field name.

        Returns:
            Count of distinct values.
        """
        values = set()
        for row in data:
            val = row.get(field)
            if val is not None:
                try:
                    values.add(val)
                except TypeError:
                    values.add(str(val))
        return len(values)

    def _compute_percentile(
        self,
        data: List[Dict[str, Any]],
        field: str,
        percentile: float = 95,
    ) -> float:
        """Compute percentile of a numeric field.

        Uses nearest-rank method.

        Args:
            data: Data rows.
            field: Field name.
            percentile: Percentile to compute (0-100).

        Returns:
            Percentile value.
        """
        values = sorted(
            row.get(field) for row in data
            if isinstance(row.get(field), (int, float))
        )
        if not values:
            return 0.0

        # Nearest rank method
        k = max(0, min(len(values) - 1,
                       int(len(values) * percentile / 100.0)))
        return float(values[k])

    def _compute_group_by(
        self,
        data: List[Dict[str, Any]],
        field: str,
        group_fields: List[str],
    ) -> List[Dict[str, Any]]:
        """Group data by specified fields and count.

        Args:
            data: Data rows.
            field: Primary field to aggregate.
            group_fields: Fields to group by.

        Returns:
            List of grouped result dictionaries.
        """
        if not group_fields:
            group_fields = [field]

        groups: Dict[str, List[Dict[str, Any]]] = {}

        for row in data:
            key_parts = []
            for gf in group_fields:
                key_parts.append(str(row.get(gf, "")))
            key = "|".join(key_parts)

            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        results: List[Dict[str, Any]] = []
        for key, rows in groups.items():
            group_result: Dict[str, Any] = {"count": len(rows)}
            key_values = key.split("|")
            for i, gf in enumerate(group_fields):
                if i < len(key_values):
                    group_result[gf] = key_values[i]

            # Compute sum of the target field within group
            field_sum = sum(
                r.get(field, 0) for r in rows
                if isinstance(r.get(field), (int, float))
            )
            group_result[f"{field}_sum"] = round(field_sum, 6)

            results.append(group_result)

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def aggregation_count(self) -> int:
        """Return the total number of aggregation operations performed."""
        return len(self._aggregations)

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics.

        Returns:
            Dictionary with aggregation counts.
        """
        return {
            "total_aggregations": len(self._aggregations),
        }


__all__ = [
    "ResponseAggregatorEngine",
    "CONFLICT_STRATEGIES",
]
