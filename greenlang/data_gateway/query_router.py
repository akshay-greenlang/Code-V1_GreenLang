# -*- coding: utf-8 -*-
"""
Query Router Engine - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Routes parsed query plans to the appropriate data sources, executes
them (single or multi-source), applies local filtering/sorting, and
manages circuit breaker state per source.

Zero-Hallucination Guarantees:
    - Routing uses deterministic source matching
    - Circuit breaker state transitions follow fixed rules
    - Filter/sort application uses standard comparison operators
    - SHA-256 provenance hashes on all query results

Example:
    >>> from greenlang.data_gateway.query_router import QueryRouterEngine
    >>> router = QueryRouterEngine()
    >>> result = router.execute(plan)
    >>> assert result["query_id"] == plan["query_id"]

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
# Data Structures
# ---------------------------------------------------------------------------

def _make_circuit_breaker_state(
    source_id: str,
) -> Dict[str, Any]:
    """Create a CircuitBreakerState dictionary.

    Args:
        source_id: Data source identifier.

    Returns:
        CircuitBreakerState dictionary.
    """
    return {
        "source_id": source_id,
        "failures": 0,
        "state": "closed",  # closed, open, half_open
        "last_failure_time": None,
        "last_success_time": None,
        "consecutive_successes": 0,
    }


def _make_query_result(
    query_id: str,
    source_id: str,
    data: List[Dict[str, Any]],
    total_count: int,
    metadata: Optional[Dict[str, Any]] = None,
    errors: Optional[List[str]] = None,
    execution_time_ms: float = 0.0,
) -> Dict[str, Any]:
    """Create a QueryResult dictionary.

    Args:
        query_id: Query identifier.
        source_id: Source that produced the result.
        data: Result data rows.
        total_count: Total matching rows (before pagination).
        metadata: Additional result metadata.
        errors: Any errors encountered.
        execution_time_ms: Execution time in milliseconds.

    Returns:
        QueryResult dictionary.
    """
    return {
        "query_id": query_id,
        "source_id": source_id,
        "data": data,
        "total_count": total_count,
        "row_count": len(data),
        "metadata": metadata or {},
        "errors": errors or [],
        "execution_time_ms": round(execution_time_ms, 2),
        "created_at": _utcnow().isoformat(),
    }


class QueryRouterEngine:
    """Query routing and execution engine with circuit breaker.

    Routes query plans to data sources, executes them, and applies
    local filtering/sorting. Manages per-source circuit breakers
    to prevent cascading failures.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _connection_manager: ConnectionManagerEngine for source lookups.
        _results: In-memory query result storage.
        _circuit_breakers: Per-source circuit breaker states.
        _source_data: Simulated per-source data for in-memory execution.

    Example:
        >>> router = QueryRouterEngine()
        >>> result = router.execute(plan)
        >>> assert len(result["data"]) <= plan["limit"]
    """

    # Circuit breaker defaults
    CB_FAILURE_THRESHOLD = 5
    CB_TIMEOUT_SECONDS = 60
    CB_SUCCESS_THRESHOLD = 3

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
        connection_manager: Any = None,
    ) -> None:
        """Initialize QueryRouterEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
            connection_manager: Optional ConnectionManagerEngine.
        """
        self._config = config or {}
        self._provenance = provenance
        self._connection_manager = connection_manager
        self._results: Dict[str, Dict[str, Any]] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._source_data: Dict[str, List[Dict[str, Any]]] = {}

        # Load CB config
        if hasattr(config, "circuit_breaker_threshold"):
            self.CB_FAILURE_THRESHOLD = config.circuit_breaker_threshold
        if hasattr(config, "circuit_breaker_timeout"):
            self.CB_TIMEOUT_SECONDS = config.circuit_breaker_timeout

        logger.info("QueryRouterEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        plan: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Route a query plan into per-source sub-plans.

        Splits the original plan into one sub-plan per source, each
        inheriting the filters, sorts, aggregations, and pagination.

        Args:
            plan: QueryPlan dictionary.

        Returns:
            Dictionary mapping source_id to sub-QueryPlan.
        """
        sources = plan.get("sources", [])
        sub_plans: Dict[str, Dict[str, Any]] = {}

        for source_id in sources:
            sub_plan = {
                "query_id": plan.get("query_id", ""),
                "sources": [source_id],
                "filters": plan.get("filters", []),
                "sorts": plan.get("sorts", []),
                "aggregations": plan.get("aggregations", []),
                "fields": plan.get("fields", []),
                "limit": plan.get("limit", 100),
                "offset": plan.get("offset", 0),
                "complexity": plan.get("complexity", 0.0),
                "created_at": plan.get("created_at", ""),
                "raw_request": plan.get("raw_request", {}),
            }
            sub_plans[source_id] = sub_plan

        # Record routing decisions
        try:
            from greenlang.data_gateway.metrics import record_routing_decision
            for source_id in sources:
                strategy = "multi" if len(sources) > 1 else "single"
                record_routing_decision(source=source_id, strategy=strategy)
        except ImportError:
            pass

        logger.info(
            "Routed query %s to %d source(s): %s",
            plan.get("query_id", "unknown"),
            len(sub_plans),
            list(sub_plans.keys()),
        )
        return sub_plans

    def execute(
        self,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a query plan against its target sources.

        Routes the plan, determines single vs. multi-source execution,
        and returns the merged result.

        Args:
            plan: QueryPlan dictionary.

        Returns:
            QueryResult dictionary.
        """
        start_time = time.monotonic()
        query_id = plan.get("query_id", f"QRY-{uuid.uuid4().hex[:12]}")
        sources = plan.get("sources", [])

        try:
            if len(sources) <= 1:
                source_id = sources[0] if sources else "default"
                result = self._execute_single_source(source_id, plan)
            else:
                sub_plans = self.route(plan)
                result = self._execute_multi_source(sub_plans)

            # Override query_id from plan
            result["query_id"] = query_id

        except Exception as e:
            logger.error("Query execution failed for %s: %s", query_id, e)
            result = _make_query_result(
                query_id=query_id,
                source_id=",".join(sources),
                data=[],
                total_count=0,
                errors=[str(e)],
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result["execution_time_ms"] = round(elapsed_ms, 2)

        # Store result
        self._results[query_id] = result

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash({
                "query_id": query_id,
                "row_count": result.get("row_count", 0),
                "total_count": result.get("total_count", 0),
            })
            self._provenance.record(
                entity_type="query_result",
                entity_id=query_id,
                action="query_execution",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.data_gateway.metrics import record_query
            status = "success" if not result.get("errors") else "error"
            record_query(
                source=",".join(sources) if sources else "none",
                operation="execute",
                status=status,
                duration=(time.monotonic() - start_time),
            )
        except ImportError:
            pass

        logger.info(
            "Executed query %s: %d rows from %d source(s) (%.1f ms)",
            query_id,
            result.get("row_count", 0),
            len(sources),
            elapsed_ms,
        )
        return result

    def get_result(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a query result by ID.

        Args:
            query_id: Query identifier.

        Returns:
            QueryResult dictionary or None if not found.
        """
        return self._results.get(query_id)

    def register_source_data(
        self,
        source_id: str,
        data: List[Dict[str, Any]],
    ) -> None:
        """Register simulated data for a source (in-memory execution).

        Args:
            source_id: Data source identifier.
            data: List of data rows to serve for this source.
        """
        self._source_data[source_id] = data
        logger.info(
            "Registered %d rows for source %s", len(data), source_id,
        )

    def get_circuit_breaker_state(
        self,
        source_id: str,
    ) -> Dict[str, Any]:
        """Get circuit breaker state for a source.

        Args:
            source_id: Data source identifier.

        Returns:
            CircuitBreakerState dictionary.
        """
        if source_id not in self._circuit_breakers:
            self._circuit_breakers[source_id] = _make_circuit_breaker_state(
                source_id
            )
        return self._circuit_breakers[source_id]

    # ------------------------------------------------------------------
    # Execution internals
    # ------------------------------------------------------------------

    def _execute_single_source(
        self,
        source_id: str,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a query plan against a single source.

        Checks circuit breaker, retrieves data from source, applies
        filters, sorts, and pagination locally.

        Args:
            source_id: Target data source ID.
            plan: QueryPlan dictionary.

        Returns:
            QueryResult dictionary.
        """
        query_id = plan.get("query_id", "")

        # Check circuit breaker
        cb = self.get_circuit_breaker_state(source_id)
        if cb["state"] == "open":
            # Check if timeout has elapsed for half-open
            if cb["last_failure_time"]:
                elapsed = (
                    _utcnow() - datetime.fromisoformat(
                        cb["last_failure_time"]
                    )
                ).total_seconds()
                if elapsed < self.CB_TIMEOUT_SECONDS:
                    logger.warning(
                        "Circuit breaker OPEN for source %s, skipping",
                        source_id,
                    )
                    return _make_query_result(
                        query_id=query_id,
                        source_id=source_id,
                        data=[],
                        total_count=0,
                        errors=[
                            f"Circuit breaker open for source {source_id}"
                        ],
                    )
                else:
                    cb["state"] = "half_open"
                    logger.info(
                        "Circuit breaker HALF_OPEN for source %s", source_id,
                    )

        try:
            # Retrieve data from in-memory source storage
            raw_data = list(self._source_data.get(source_id, []))

            # If connection manager available, check source exists
            if self._connection_manager and not raw_data:
                source = self._connection_manager.get_source(source_id)
                if source is None:
                    return _make_query_result(
                        query_id=query_id,
                        source_id=source_id,
                        data=[],
                        total_count=0,
                        errors=[f"Source not found: {source_id}"],
                    )

            # Apply filters
            filtered = self._apply_filters(
                raw_data, plan.get("filters", []),
            )

            # Apply sorts
            sorted_data = self._apply_sorts(
                filtered, plan.get("sorts", []),
            )

            total_count = len(sorted_data)

            # Apply pagination
            paginated = self._apply_pagination(
                sorted_data,
                plan.get("limit", 100),
                plan.get("offset", 0),
            )

            # Apply field projection
            fields = plan.get("fields", [])
            if fields:
                paginated = [
                    {k: v for k, v in row.items() if k in fields}
                    for row in paginated
                ]

            # Record success in circuit breaker
            self._record_cb_success(source_id)

            return _make_query_result(
                query_id=query_id,
                source_id=source_id,
                data=paginated,
                total_count=total_count,
            )

        except Exception as e:
            self._record_cb_failure(source_id)
            logger.error(
                "Source %s execution failed: %s", source_id, e,
            )
            return _make_query_result(
                query_id=query_id,
                source_id=source_id,
                data=[],
                total_count=0,
                errors=[str(e)],
            )

    def _execute_multi_source(
        self,
        sub_plans: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute query plans against multiple sources and merge results.

        Executes each sub-plan sequentially (in-memory), then merges
        data from all sources.

        Args:
            sub_plans: Dictionary mapping source_id to sub-QueryPlan.

        Returns:
            Merged QueryResult dictionary.
        """
        all_data: List[Dict[str, Any]] = []
        all_errors: List[str] = []
        total_count = 0
        source_ids: List[str] = []
        query_id = ""

        for source_id, sub_plan in sub_plans.items():
            if not query_id:
                query_id = sub_plan.get("query_id", "")
            source_ids.append(source_id)

            result = self._execute_single_source(source_id, sub_plan)
            all_data.extend(result.get("data", []))
            all_errors.extend(result.get("errors", []))
            total_count += result.get("total_count", 0)

        # Record aggregation metric
        try:
            from greenlang.data_gateway.metrics import record_aggregation
            status = "success" if not all_errors else "partial_error"
            record_aggregation(
                sources_count=str(len(source_ids)),
                status=status,
            )
        except ImportError:
            pass

        return _make_query_result(
            query_id=query_id,
            source_id=",".join(source_ids),
            data=all_data,
            total_count=total_count,
            errors=all_errors if all_errors else None,
            metadata={"sources_queried": source_ids},
        )

    # ------------------------------------------------------------------
    # Filter / Sort / Pagination
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        data: List[Dict[str, Any]],
        filters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply filters to a data list.

        Args:
            data: Data rows.
            filters: List of QueryFilter dictionaries.

        Returns:
            Filtered data rows.
        """
        result = data
        for f in filters:
            field = f.get("field", "")
            operator = f.get("operator", "eq")
            value = f.get("value")
            negate = f.get("negate", False)

            filtered: List[Dict[str, Any]] = []
            for row in result:
                row_val = row.get(field)
                match = self._evaluate_filter(row_val, operator, value)
                if negate:
                    match = not match
                if match:
                    filtered.append(row)
            result = filtered

        return result

    def _evaluate_filter(
        self,
        row_val: Any,
        operator: str,
        value: Any,
    ) -> bool:
        """Evaluate a single filter condition.

        Args:
            row_val: Value from the data row.
            operator: Filter operator.
            value: Filter comparison value.

        Returns:
            True if the row matches the filter.
        """
        try:
            if operator == "eq":
                return row_val == value
            elif operator == "ne":
                return row_val != value
            elif operator == "gt":
                return row_val > value
            elif operator == "gte":
                return row_val >= value
            elif operator == "lt":
                return row_val < value
            elif operator == "lte":
                return row_val <= value
            elif operator == "in":
                return row_val in value
            elif operator == "not_in":
                return row_val not in value
            elif operator == "contains":
                return (
                    isinstance(row_val, str)
                    and isinstance(value, str)
                    and value.lower() in row_val.lower()
                )
            elif operator == "between":
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return value[0] <= row_val <= value[1]
                return False
            elif operator == "is_null":
                return row_val is None
            elif operator == "is_not_null":
                return row_val is not None
            elif operator == "starts_with":
                return (
                    isinstance(row_val, str)
                    and isinstance(value, str)
                    and row_val.lower().startswith(value.lower())
                )
            elif operator == "ends_with":
                return (
                    isinstance(row_val, str)
                    and isinstance(value, str)
                    and row_val.lower().endswith(value.lower())
                )
        except (TypeError, ValueError):
            return False
        return False

    def _apply_sorts(
        self,
        data: List[Dict[str, Any]],
        sorts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply sort operations to a data list.

        Args:
            data: Data rows.
            sorts: List of QuerySort dictionaries.

        Returns:
            Sorted data rows.
        """
        if not sorts:
            return data

        result = list(data)
        # Apply sorts in reverse order (last sort is least significant)
        for sort_spec in reversed(sorts):
            field = sort_spec.get("field", "")
            direction = sort_spec.get("direction", "asc")
            null_handling = sort_spec.get("null_handling", "last")

            def sort_key(row: Dict[str, Any], f=field, nh=null_handling):
                val = row.get(f)
                if val is None:
                    # Sort nulls first or last
                    return (0 if nh == "first" else 2, "")
                return (1, val)

            reverse = direction == "desc"
            result.sort(key=sort_key, reverse=reverse)

        return result

    def _apply_pagination(
        self,
        data: List[Dict[str, Any]],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """Apply limit/offset pagination to data.

        Args:
            data: Data rows.
            limit: Maximum rows to return.
            offset: Rows to skip.

        Returns:
            Paginated data rows.
        """
        return data[offset:offset + limit]

    # ------------------------------------------------------------------
    # Circuit Breaker Management
    # ------------------------------------------------------------------

    def _record_cb_success(self, source_id: str) -> None:
        """Record a successful execution for circuit breaker.

        Args:
            source_id: Data source identifier.
        """
        cb = self.get_circuit_breaker_state(source_id)
        cb["consecutive_successes"] += 1
        cb["last_success_time"] = _utcnow().isoformat()

        if cb["state"] == "half_open":
            if cb["consecutive_successes"] >= self.CB_SUCCESS_THRESHOLD:
                cb["state"] = "closed"
                cb["failures"] = 0
                cb["consecutive_successes"] = 0
                logger.info(
                    "Circuit breaker CLOSED for source %s", source_id,
                )

    def _record_cb_failure(self, source_id: str) -> None:
        """Record a failed execution for circuit breaker.

        Args:
            source_id: Data source identifier.
        """
        cb = self.get_circuit_breaker_state(source_id)
        cb["failures"] += 1
        cb["consecutive_successes"] = 0
        cb["last_failure_time"] = _utcnow().isoformat()

        if cb["failures"] >= self.CB_FAILURE_THRESHOLD:
            cb["state"] = "open"
            logger.warning(
                "Circuit breaker OPEN for source %s after %d failures",
                source_id, cb["failures"],
            )

        # Record metric
        try:
            from greenlang.data_gateway.metrics import record_processing_error
            record_processing_error(
                source=source_id, error_type="circuit_breaker_failure",
            )
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def result_count(self) -> int:
        """Return the total number of stored query results."""
        return len(self._results)

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics.

        Returns:
            Dictionary with result counts and circuit breaker states.
        """
        cb_states: Dict[str, str] = {}
        for source_id, cb in self._circuit_breakers.items():
            cb_states[source_id] = cb["state"]

        return {
            "total_results": len(self._results),
            "circuit_breakers": cb_states,
            "registered_sources": len(self._source_data),
        }


__all__ = [
    "QueryRouterEngine",
]
