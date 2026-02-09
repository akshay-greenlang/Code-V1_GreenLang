# -*- coding: utf-8 -*-
"""
Query Parser Engine - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Parses incoming data query requests into structured query plans with
filters, sorts, aggregations, and complexity scoring. Validates query
plans against configurable constraints before execution.

Zero-Hallucination Guarantees:
    - All parsing uses deterministic rule-based transformations
    - Complexity scores are computed from fixed cost formulas
    - No ML/LLM used for query interpretation
    - SHA-256 provenance hashes on all parsed query plans

Example:
    >>> from greenlang.data_gateway.query_parser import QueryParserEngine
    >>> parser = QueryParserEngine()
    >>> plan = parser.parse(request)
    >>> assert plan["query_id"].startswith("QRY-")

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

def _make_query_filter(
    field: str,
    operator: str,
    value: Any,
    negate: bool = False,
) -> Dict[str, Any]:
    """Create a QueryFilter dictionary.

    Args:
        field: Field name to filter on.
        operator: Comparison operator (eq, ne, gt, gte, lt, lte, in,
                  not_in, contains, between, is_null, is_not_null).
        value: Filter value.
        negate: Whether to negate the filter.

    Returns:
        QueryFilter dictionary.
    """
    return {
        "field": field,
        "operator": operator,
        "value": value,
        "negate": negate,
    }


def _make_query_sort(
    field: str,
    direction: str = "asc",
    null_handling: str = "last",
) -> Dict[str, Any]:
    """Create a QuerySort dictionary.

    Args:
        field: Field name to sort on.
        direction: Sort direction (asc or desc).
        null_handling: Where to place nulls (first or last).

    Returns:
        QuerySort dictionary.
    """
    return {
        "field": field,
        "direction": direction,
        "null_handling": null_handling,
    }


def _make_query_aggregation(
    function: str,
    field: str,
    alias: Optional[str] = None,
    group_by: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a QueryAggregation dictionary.

    Args:
        function: Aggregation function (sum, avg, min, max, count,
                  percentile, group_by).
        field: Field to aggregate.
        alias: Optional alias for the result.
        group_by: Optional grouping fields.

    Returns:
        QueryAggregation dictionary.
    """
    return {
        "function": function,
        "field": field,
        "alias": alias or f"{function}_{field}",
        "group_by": group_by or [],
    }


def _make_query_plan(
    query_id: str,
    sources: List[str],
    filters: List[Dict[str, Any]],
    sorts: List[Dict[str, Any]],
    aggregations: List[Dict[str, Any]],
    fields: Optional[List[str]] = None,
    limit: int = 100,
    offset: int = 0,
    complexity: float = 0.0,
    raw_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a QueryPlan dictionary.

    Args:
        query_id: Unique query identifier.
        sources: List of source IDs to query.
        filters: Parsed filter list.
        sorts: Parsed sort list.
        aggregations: Parsed aggregation list.
        fields: Optional field projection list.
        limit: Maximum result rows.
        offset: Result offset.
        complexity: Computed complexity score.
        raw_request: Original request data.

    Returns:
        QueryPlan dictionary.
    """
    return {
        "query_id": query_id,
        "sources": sources,
        "filters": filters,
        "sorts": sorts,
        "aggregations": aggregations,
        "fields": fields or [],
        "limit": limit,
        "offset": offset,
        "complexity": complexity,
        "created_at": _utcnow().isoformat(),
        "raw_request": raw_request or {},
    }


# ---------------------------------------------------------------------------
# Supported operators and functions
# ---------------------------------------------------------------------------

SUPPORTED_OPERATORS = frozenset({
    "eq", "ne", "gt", "gte", "lt", "lte",
    "in", "not_in", "contains", "between",
    "is_null", "is_not_null",
    "starts_with", "ends_with", "regex",
})

SUPPORTED_AGG_FUNCTIONS = frozenset({
    "sum", "avg", "min", "max", "count",
    "percentile", "group_by", "distinct_count",
})


class QueryParserEngine:
    """Query parsing and validation engine.

    Transforms raw query requests into structured QueryPlan dictionaries
    with deterministic complexity scoring and validation.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _plans: In-memory query plan storage.

    Example:
        >>> parser = QueryParserEngine()
        >>> plan = parser.parse({"sources": ["SRC-abc"], "filters": []})
        >>> assert plan["query_id"].startswith("QRY-")
    """

    # Complexity cost constants
    COST_BASE_PER_SOURCE = 1.0
    COST_PER_FILTER = 0.5
    COST_FILTER_BETWEEN = 2.0
    COST_FILTER_CONTAINS = 1.5
    COST_FILTER_REGEX = 2.5
    COST_PER_SORT = 0.5
    COST_PER_AGGREGATION = 2.0
    COST_AGG_PERCENTILE = 3.0
    COST_CROSS_SOURCE = 5.0

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize QueryParserEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._plans: Dict[str, Dict[str, Any]] = {}

        logger.info("QueryParserEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Parse an incoming query request into a QueryPlan.

        Extracts sources, filters, sorts, aggregations, and pagination
        from the request dictionary. Computes complexity and stores the
        resulting plan.

        Args:
            request: Raw query request dictionary with keys:
                sources (List[str]): Data source IDs.
                filters (List[Dict]): Optional filter definitions.
                sorts (List[Dict]): Optional sort definitions.
                aggregations (List[Dict]): Optional aggregation defs.
                fields (List[str]): Optional field projection.
                limit (int): Max rows (default 100).
                offset (int): Offset (default 0).

        Returns:
            QueryPlan dictionary.
        """
        start_time = time.monotonic()

        query_id = self._generate_query_id()
        sources = request.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]

        filters = self.parse_filters(request.get("filters", []))
        sorts = self.parse_sorts(request.get("sorts", []))
        aggregations = self.parse_aggregations(
            request.get("aggregations", [])
        )
        fields = request.get("fields", [])
        limit = request.get("limit", 100)
        offset = request.get("offset", 0)

        plan = _make_query_plan(
            query_id=query_id,
            sources=sources,
            filters=filters,
            sorts=sorts,
            aggregations=aggregations,
            fields=fields,
            limit=limit,
            offset=offset,
            raw_request=request,
        )

        # Compute complexity
        complexity = self.calculate_complexity(plan)
        plan["complexity"] = complexity

        # Store plan
        self._plans[query_id] = plan

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(plan)
            self._provenance.record(
                entity_type="query_plan",
                entity_id=query_id,
                action="query_execution",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.data_gateway.metrics import record_query
            record_query(
                source=",".join(sources) if sources else "none",
                operation="parse",
                status="success",
                duration=(time.monotonic() - start_time),
            )
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Parsed query %s: sources=%d, filters=%d, sorts=%d, "
            "aggs=%d, complexity=%.1f (%.1f ms)",
            query_id,
            len(sources),
            len(filters),
            len(sorts),
            len(aggregations),
            complexity,
            elapsed_ms,
        )
        return plan

    def parse_filters(
        self,
        raw_filters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse filter dictionaries into QueryFilter structures.

        Args:
            raw_filters: List of filter dicts with keys:
                field (str): Field name.
                operator (str): Comparison operator.
                value: Filter value.
                negate (bool): Optional negation flag.

        Returns:
            List of QueryFilter dictionaries.
        """
        parsed: List[Dict[str, Any]] = []
        for raw in raw_filters:
            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict filter: %s", type(raw))
                continue

            field = raw.get("field", "")
            operator = raw.get("operator", "eq")
            value = raw.get("value")
            negate = raw.get("negate", False)

            if not field:
                logger.warning("Skipping filter with empty field")
                continue

            # Normalize operator
            operator = operator.lower().strip()
            if operator not in SUPPORTED_OPERATORS:
                logger.warning(
                    "Unsupported filter operator '%s', defaulting to 'eq'",
                    operator,
                )
                operator = "eq"

            # Validate between requires list of 2
            if operator == "between":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    logger.warning(
                        "Filter 'between' requires [min, max], got %s",
                        type(value).__name__,
                    )
                    continue

            parsed.append(_make_query_filter(
                field=field,
                operator=operator,
                value=value,
                negate=negate,
            ))

        return parsed

    def parse_sorts(
        self,
        raw_sorts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse sort dictionaries into QuerySort structures.

        Args:
            raw_sorts: List of sort dicts with keys:
                field (str): Field name.
                direction (str): asc or desc (default asc).
                null_handling (str): first or last (default last).

        Returns:
            List of QuerySort dictionaries.
        """
        parsed: List[Dict[str, Any]] = []
        for raw in raw_sorts:
            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict sort: %s", type(raw))
                continue

            field = raw.get("field", "")
            if not field:
                logger.warning("Skipping sort with empty field")
                continue

            direction = raw.get("direction", "asc").lower().strip()
            if direction not in ("asc", "desc"):
                direction = "asc"

            null_handling = raw.get("null_handling", "last").lower().strip()
            if null_handling not in ("first", "last"):
                null_handling = "last"

            parsed.append(_make_query_sort(
                field=field,
                direction=direction,
                null_handling=null_handling,
            ))

        return parsed

    def parse_aggregations(
        self,
        raw_aggs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Parse aggregation dictionaries into QueryAggregation structures.

        Args:
            raw_aggs: List of aggregation dicts with keys:
                function (str): Aggregation function.
                field (str): Field to aggregate.
                alias (str): Optional result alias.
                group_by (List[str]): Optional grouping fields.

        Returns:
            List of QueryAggregation dictionaries.
        """
        parsed: List[Dict[str, Any]] = []
        for raw in raw_aggs:
            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict aggregation: %s", type(raw))
                continue

            function = raw.get("function", "").lower().strip()
            field = raw.get("field", "")

            if not function or not field:
                logger.warning(
                    "Skipping aggregation with empty function or field"
                )
                continue

            if function not in SUPPORTED_AGG_FUNCTIONS:
                logger.warning(
                    "Unsupported aggregation function '%s'", function,
                )
                continue

            alias = raw.get("alias")
            group_by = raw.get("group_by", [])
            if isinstance(group_by, str):
                group_by = [group_by]

            parsed.append(_make_query_aggregation(
                function=function,
                field=field,
                alias=alias,
                group_by=group_by,
            ))

        return parsed

    def calculate_complexity(self, plan: Dict[str, Any]) -> float:
        """Calculate the complexity score of a query plan.

        Deterministic scoring formula:
        - Base cost: 1.0 per source
        - Filter cost: 0.5 per filter, 2.0 for 'between', 1.5 for 'contains'
        - Sort cost: 0.5 per sort
        - Aggregation cost: 2.0 per aggregation, 3.0 for percentile
        - Cross-source cost: 5.0 if multiple sources

        Args:
            plan: QueryPlan dictionary.

        Returns:
            Complexity score as float.
        """
        score = 0.0
        sources = plan.get("sources", [])
        filters = plan.get("filters", [])
        sorts = plan.get("sorts", [])
        aggregations = plan.get("aggregations", [])

        # Base cost per source
        score += len(sources) * self.COST_BASE_PER_SOURCE

        # Filter costs
        for f in filters:
            op = f.get("operator", "eq")
            if op == "between":
                score += self.COST_FILTER_BETWEEN
            elif op == "contains":
                score += self.COST_FILTER_CONTAINS
            elif op == "regex":
                score += self.COST_FILTER_REGEX
            else:
                score += self.COST_PER_FILTER

        # Sort costs
        score += len(sorts) * self.COST_PER_SORT

        # Aggregation costs
        for agg in aggregations:
            func = agg.get("function", "")
            if func == "percentile":
                score += self.COST_AGG_PERCENTILE
            else:
                score += self.COST_PER_AGGREGATION

        # Cross-source penalty
        if len(sources) > 1:
            score += self.COST_CROSS_SOURCE

        return round(score, 2)

    def validate_query(
        self,
        plan: Dict[str, Any],
        max_complexity: Optional[int] = None,
    ) -> List[str]:
        """Validate a query plan and return validation errors.

        Checks:
        - At least one source specified
        - Complexity within allowed limit
        - Valid filter operators
        - Valid aggregation functions
        - Limit within bounds

        Args:
            plan: QueryPlan dictionary.
            max_complexity: Maximum allowed complexity (default from config).

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        sources = plan.get("sources", [])
        if not sources:
            errors.append("At least one data source is required")

        # Check max sources
        max_sources = 10
        if hasattr(self._config, "max_sources_per_query"):
            max_sources = self._config.max_sources_per_query
        elif isinstance(self._config, dict):
            max_sources = self._config.get("max_sources_per_query", 10)

        if len(sources) > max_sources:
            errors.append(
                f"Too many sources: {len(sources)} (max {max_sources})"
            )

        # Check complexity
        if max_complexity is None:
            if hasattr(self._config, "max_query_complexity"):
                max_complexity = self._config.max_query_complexity
            elif isinstance(self._config, dict):
                max_complexity = self._config.get("max_query_complexity", 100)
            else:
                max_complexity = 100

        complexity = plan.get("complexity", 0.0)
        if complexity > max_complexity:
            errors.append(
                f"Query complexity {complexity} exceeds maximum "
                f"{max_complexity}"
            )

        # Validate limit
        limit = plan.get("limit", 100)
        if limit < 1:
            errors.append("Limit must be at least 1")
        elif limit > 10000:
            errors.append(
                f"Limit {limit} exceeds maximum 10000"
            )

        # Validate offset
        offset = plan.get("offset", 0)
        if offset < 0:
            errors.append("Offset must be non-negative")

        # Validate filter operators
        for f in plan.get("filters", []):
            op = f.get("operator", "")
            if op and op not in SUPPORTED_OPERATORS:
                errors.append(f"Unsupported filter operator: {op}")

        # Validate aggregation functions
        for agg in plan.get("aggregations", []):
            func = agg.get("function", "")
            if func and func not in SUPPORTED_AGG_FUNCTIONS:
                errors.append(f"Unsupported aggregation function: {func}")

        return errors

    def get_plan(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a query plan by ID.

        Args:
            query_id: Query plan identifier.

        Returns:
            QueryPlan dictionary or None if not found.
        """
        return self._plans.get(query_id)

    def list_plans(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored query plans.

        Args:
            limit: Maximum number of plans to return.
            offset: Number of plans to skip.

        Returns:
            List of QueryPlan dictionaries.
        """
        plans = list(self._plans.values())
        plans.sort(key=lambda p: p.get("created_at", ""), reverse=True)
        return plans[offset:offset + limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_query_id(self) -> str:
        """Generate a unique query identifier.

        Returns:
            Query ID in format "QRY-{hex12}".
        """
        return f"QRY-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def plan_count(self) -> int:
        """Return the total number of stored query plans."""
        return len(self._plans)

    def get_statistics(self) -> Dict[str, Any]:
        """Get parser statistics.

        Returns:
            Dictionary with plan counts and complexity distribution.
        """
        plans = list(self._plans.values())
        complexities = [p.get("complexity", 0.0) for p in plans]
        return {
            "total_plans": len(plans),
            "avg_complexity": (
                round(sum(complexities) / len(complexities), 2)
                if complexities else 0.0
            ),
            "max_complexity": max(complexities) if complexities else 0.0,
            "min_complexity": min(complexities) if complexities else 0.0,
        }


__all__ = [
    "QueryParserEngine",
    "SUPPORTED_OPERATORS",
    "SUPPORTED_AGG_FUNCTIONS",
]
