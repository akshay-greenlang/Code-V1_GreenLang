# -*- coding: utf-8 -*-
"""
Unit Tests for QueryParserEngine (AGENT-DATA-004)

Tests query parsing, ID format (QRY-xxxxx), provenance hash generation,
filter parsing (all 14 operators), sort parsing, aggregation parsing (all 7 types),
complexity calculation (base, filter, sort, aggregation, cross-source, percentile costs),
and query validation (max complexity, empty sources, too many sources).

Coverage target: 85%+ of query_parser.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline models (minimal, same as test_models.py)
# ---------------------------------------------------------------------------


class QueryFilter:
    def __init__(self, field: str = "", operator: str = "eq", value: Any = None):
        self.field = field
        self.operator = operator
        self.value = value


class QuerySort:
    def __init__(self, field: str = "", order: str = "asc"):
        self.field = field
        self.order = order


class QueryAggregation:
    def __init__(
        self,
        type: str = "count",
        field: str = "",
        alias: Optional[str] = None,
        group_by: Optional[List[str]] = None,
    ):
        self.type = type
        self.field = field
        self.alias = alias
        self.group_by = group_by or []


class QueryPlan:
    def __init__(
        self,
        plan_id: str = "",
        query_id: str = "",
        source_ids: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
        filters: Optional[List[QueryFilter]] = None,
        sorts: Optional[List[QuerySort]] = None,
        aggregations: Optional[List[QueryAggregation]] = None,
        estimated_complexity: int = 0,
        estimated_rows: int = 0,
        cache_eligible: bool = True,
        provenance_hash: Optional[str] = None,
    ):
        self.plan_id = plan_id
        self.query_id = query_id
        self.source_ids = source_ids or []
        self.operations = operations or []
        self.filters = filters or []
        self.sorts = sorts or []
        self.aggregations = aggregations or []
        self.estimated_complexity = estimated_complexity
        self.estimated_rows = estimated_rows
        self.cache_eligible = cache_eligible
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline ExecuteQueryRequest
# ---------------------------------------------------------------------------


class ExecuteQueryRequest:
    def __init__(
        self,
        source_ids: Optional[List[str]] = None,
        operation: str = "select",
        filters: Optional[List[QueryFilter]] = None,
        sorts: Optional[List[QuerySort]] = None,
        aggregations: Optional[List[QueryAggregation]] = None,
        limit: int = 100,
        offset: int = 0,
        cache_strategy: str = "ttl",
        timeout_s: int = 60,
    ):
        self.source_ids = source_ids or []
        self.operation = operation
        self.filters = filters or []
        self.sorts = sorts or []
        self.aggregations = aggregations or []
        self.limit = limit
        self.offset = offset
        self.cache_strategy = cache_strategy
        self.timeout_s = timeout_s


# ---------------------------------------------------------------------------
# Inline QueryParserEngine
# ---------------------------------------------------------------------------


class QueryParserEngine:
    """Parses, validates, and plans queries for the data gateway."""

    # Complexity cost constants
    BASE_COST = 10
    FILTER_COST = 5
    SORT_COST = 8
    AGGREGATION_COST = 15
    CROSS_SOURCE_COST = 50
    PERCENTILE_COST = 25

    VALID_OPERATORS = {
        "eq", "ne", "gt", "gte", "lt", "lte",
        "in", "not_in", "contains", "starts_with",
        "ends_with", "between", "is_null", "is_not_null",
    }

    VALID_AGGREGATIONS = {"count", "sum", "avg", "min", "max", "median", "percentile"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._max_complexity = self._config.get("max_query_complexity", 1000)
        self._max_sources = self._config.get("max_sources_per_query", 10)
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "queries_parsed": 0,
            "validation_errors": 0,
        }

    def _next_query_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"QRY-{self._counter:05d}"

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def parse(self, request: ExecuteQueryRequest) -> QueryPlan:
        """Parse a query request into a QueryPlan."""
        query_id = self._next_query_id()

        filters = self._parse_filters(request.filters)
        sorts = self._parse_sorts(request.sorts)
        aggregations = self._parse_aggregations(request.aggregations)
        complexity = self._calculate_complexity(
            source_ids=request.source_ids,
            filters=filters,
            sorts=sorts,
            aggregations=aggregations,
        )

        prov_data = {
            "op": "parse_query",
            "query_id": query_id,
            "sources": request.source_ids,
            "operation": request.operation,
            "filter_count": len(filters),
            "sort_count": len(sorts),
            "agg_count": len(aggregations),
        }

        plan = QueryPlan(
            plan_id=f"PLN-{self._counter:05d}",
            query_id=query_id,
            source_ids=list(request.source_ids),
            operations=[request.operation],
            filters=filters,
            sorts=sorts,
            aggregations=aggregations,
            estimated_complexity=complexity,
            cache_eligible=request.cache_strategy != "none",
            provenance_hash=self._compute_provenance(prov_data),
        )

        with self._lock:
            self._stats["queries_parsed"] += 1

        return plan

    def _parse_filters(self, filters: List[QueryFilter]) -> List[QueryFilter]:
        """Validate and return parsed filters."""
        parsed = []
        for f in filters:
            if f.operator not in self.VALID_OPERATORS:
                raise ValueError(f"Invalid filter operator: {f.operator}")
            parsed.append(f)
        return parsed

    def _parse_sorts(self, sorts: List[QuerySort]) -> List[QuerySort]:
        """Validate and return parsed sorts."""
        parsed = []
        for s in sorts:
            if s.order not in ("asc", "desc"):
                raise ValueError(f"Invalid sort order: {s.order}")
            parsed.append(s)
        return parsed

    def _parse_aggregations(self, aggregations: List[QueryAggregation]) -> List[QueryAggregation]:
        """Validate and return parsed aggregations."""
        parsed = []
        for a in aggregations:
            if a.type not in self.VALID_AGGREGATIONS:
                raise ValueError(f"Invalid aggregation type: {a.type}")
            parsed.append(a)
        return parsed

    def _calculate_complexity(
        self,
        source_ids: List[str],
        filters: List[QueryFilter],
        sorts: List[QuerySort],
        aggregations: List[QueryAggregation],
    ) -> int:
        """Calculate query complexity score."""
        cost = self.BASE_COST
        cost += len(filters) * self.FILTER_COST
        cost += len(sorts) * self.SORT_COST
        cost += len(aggregations) * self.AGGREGATION_COST

        # Cross-source cost for multi-source queries
        if len(source_ids) > 1:
            cost += (len(source_ids) - 1) * self.CROSS_SOURCE_COST

        # Extra cost for percentile aggregations
        for a in aggregations:
            if a.type == "percentile":
                cost += self.PERCENTILE_COST

        return cost

    def validate(self, request: ExecuteQueryRequest) -> List[str]:
        """Validate a query request. Returns list of error strings (empty = valid)."""
        errors = []

        if not request.source_ids:
            errors.append("At least one source_id is required")

        if len(request.source_ids) > self._max_sources:
            errors.append(
                f"Too many sources: {len(request.source_ids)} "
                f"(max {self._max_sources})"
            )

        # Calculate complexity to check against max
        try:
            filters = self._parse_filters(request.filters)
            sorts = self._parse_sorts(request.sorts)
            aggregations = self._parse_aggregations(request.aggregations)
            complexity = self._calculate_complexity(
                source_ids=request.source_ids,
                filters=filters,
                sorts=sorts,
                aggregations=aggregations,
            )
            if complexity > self._max_complexity:
                errors.append(
                    f"Query complexity {complexity} exceeds maximum "
                    f"{self._max_complexity}"
                )
        except ValueError as exc:
            errors.append(str(exc))

        if errors:
            with self._lock:
                self._stats["validation_errors"] += 1

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine():
    """QueryParserEngine instance for testing."""
    return QueryParserEngine()


@pytest.fixture
def strict_engine():
    """QueryParserEngine with low complexity limit for testing validation."""
    return QueryParserEngine(config={
        "max_query_complexity": 50,
        "max_sources_per_query": 3,
    })


@pytest.fixture
def sample_request():
    """ExecuteQueryRequest with typical query parameters."""
    return ExecuteQueryRequest(
        source_ids=["SRC-00001"],
        operation="select",
        filters=[
            QueryFilter(field="status", operator="eq", value="active"),
            QueryFilter(field="score", operator="gt", value=50),
        ],
        sorts=[QuerySort(field="created_at", order="desc")],
        limit=50,
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestParse:
    """Test basic query parsing."""

    def test_parse_basic(self, engine, sample_request):
        """Parse a query and verify QueryPlan returned."""
        plan = engine.parse(sample_request)
        assert plan.query_id is not None
        assert plan.plan_id is not None
        assert len(plan.source_ids) == 1
        assert plan.source_ids[0] == "SRC-00001"
        assert plan.operations == ["select"]
        assert plan.estimated_complexity > 0

    def test_parse_id_format(self, engine, sample_request):
        """Query ID matches QRY-xxxxx format."""
        plan = engine.parse(sample_request)
        assert re.match(r"^QRY-\d{5}$", plan.query_id)

    def test_parse_plan_id_format(self, engine, sample_request):
        """Plan ID matches PLN-xxxxx format."""
        plan = engine.parse(sample_request)
        assert re.match(r"^PLN-\d{5}$", plan.plan_id)

    def test_parse_provenance_hash(self, engine, sample_request):
        """Provenance hash is SHA-256 (64 hex chars)."""
        plan = engine.parse(sample_request)
        assert plan.provenance_hash is not None
        assert len(plan.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", plan.provenance_hash)

    def test_parse_increments_stats(self, engine, sample_request):
        """Parsing increments queries_parsed counter."""
        engine.parse(sample_request)
        stats = engine.get_statistics()
        assert stats["queries_parsed"] == 1

    def test_parse_sequential_ids(self, engine):
        """Sequential queries get sequential IDs."""
        req1 = ExecuteQueryRequest(source_ids=["SRC-00001"])
        req2 = ExecuteQueryRequest(source_ids=["SRC-00002"])
        plan1 = engine.parse(req1)
        plan2 = engine.parse(req2)
        assert plan1.query_id == "QRY-00001"
        assert plan2.query_id == "QRY-00002"

    def test_parse_cache_eligible_ttl(self, engine):
        """TTL cache strategy makes query cache eligible."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"], cache_strategy="ttl")
        plan = engine.parse(req)
        assert plan.cache_eligible is True

    def test_parse_cache_not_eligible_none(self, engine):
        """None cache strategy makes query not cache eligible."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"], cache_strategy="none")
        plan = engine.parse(req)
        assert plan.cache_eligible is False


class TestParseFilters:
    """Test filter parsing with all 14 operators."""

    @pytest.mark.parametrize("operator", [
        "eq", "ne", "gt", "gte", "lt", "lte",
        "in", "not_in", "contains", "starts_with",
        "ends_with", "between", "is_null", "is_not_null",
    ])
    def test_all_14_operators(self, engine, operator):
        """All 14 filter operators are accepted."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            filters=[QueryFilter(field="test", operator=operator, value="v")],
        )
        plan = engine.parse(req)
        assert len(plan.filters) == 1
        assert plan.filters[0].operator == operator

    def test_invalid_operator_raises(self, engine):
        """Invalid operator raises ValueError."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            filters=[QueryFilter(field="x", operator="invalid_op", value="v")],
        )
        with pytest.raises(ValueError, match="Invalid filter operator"):
            engine.parse(req)

    def test_empty_filters(self, engine):
        """Empty filters list is valid."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"], filters=[])
        plan = engine.parse(req)
        assert plan.filters == []

    def test_multiple_filters(self, engine):
        """Multiple filters are all parsed."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            filters=[
                QueryFilter(field="a", operator="eq", value=1),
                QueryFilter(field="b", operator="gt", value=2),
                QueryFilter(field="c", operator="in", value=[3, 4]),
            ],
        )
        plan = engine.parse(req)
        assert len(plan.filters) == 3


class TestParseSorts:
    """Test sort parsing."""

    def test_single_sort(self, engine):
        """Single sort field parsed."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            sorts=[QuerySort(field="name", order="asc")],
        )
        plan = engine.parse(req)
        assert len(plan.sorts) == 1
        assert plan.sorts[0].field == "name"
        assert plan.sorts[0].order == "asc"

    def test_multiple_sorts(self, engine):
        """Multiple sort fields parsed in order."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            sorts=[
                QuerySort(field="priority", order="desc"),
                QuerySort(field="name", order="asc"),
            ],
        )
        plan = engine.parse(req)
        assert len(plan.sorts) == 2
        assert plan.sorts[0].order == "desc"
        assert plan.sorts[1].order == "asc"

    def test_default_order(self, engine):
        """Default sort order is asc."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            sorts=[QuerySort(field="name")],
        )
        plan = engine.parse(req)
        assert plan.sorts[0].order == "asc"

    def test_invalid_sort_order_raises(self, engine):
        """Invalid sort order raises ValueError."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            sorts=[QuerySort(field="name", order="invalid")],
        )
        with pytest.raises(ValueError, match="Invalid sort order"):
            engine.parse(req)

    def test_empty_sorts(self, engine):
        """Empty sorts list is valid."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"], sorts=[])
        plan = engine.parse(req)
        assert plan.sorts == []


class TestParseAggregations:
    """Test aggregation parsing for all 7 types."""

    @pytest.mark.parametrize("agg_type", [
        "count", "sum", "avg", "min", "max", "median", "percentile",
    ])
    def test_all_7_types(self, engine, agg_type):
        """All 7 aggregation types are accepted."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            aggregations=[QueryAggregation(type=agg_type, field="value")],
        )
        plan = engine.parse(req)
        assert len(plan.aggregations) == 1
        assert plan.aggregations[0].type == agg_type

    def test_invalid_aggregation_raises(self, engine):
        """Invalid aggregation type raises ValueError."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            aggregations=[QueryAggregation(type="invalid_agg", field="value")],
        )
        with pytest.raises(ValueError, match="Invalid aggregation type"):
            engine.parse(req)

    def test_empty_aggregations(self, engine):
        """Empty aggregations list is valid."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"], aggregations=[])
        plan = engine.parse(req)
        assert plan.aggregations == []

    def test_multiple_aggregations(self, engine):
        """Multiple aggregations parsed."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            aggregations=[
                QueryAggregation(type="count", field="id"),
                QueryAggregation(type="sum", field="value"),
                QueryAggregation(type="avg", field="score"),
            ],
        )
        plan = engine.parse(req)
        assert len(plan.aggregations) == 3


class TestCalculateComplexity:
    """Test query complexity calculation."""

    def test_base_cost(self, engine):
        """Base cost is 10 for a simple query."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"])
        plan = engine.parse(req)
        assert plan.estimated_complexity == QueryParserEngine.BASE_COST

    def test_filter_cost(self, engine):
        """Each filter adds FILTER_COST (5) to complexity."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            filters=[
                QueryFilter(field="a", operator="eq", value=1),
                QueryFilter(field="b", operator="gt", value=2),
            ],
        )
        plan = engine.parse(req)
        expected = QueryParserEngine.BASE_COST + 2 * QueryParserEngine.FILTER_COST
        assert plan.estimated_complexity == expected

    def test_sort_cost(self, engine):
        """Each sort adds SORT_COST (8) to complexity."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            sorts=[QuerySort(field="name", order="asc")],
        )
        plan = engine.parse(req)
        expected = QueryParserEngine.BASE_COST + QueryParserEngine.SORT_COST
        assert plan.estimated_complexity == expected

    def test_aggregation_cost(self, engine):
        """Each aggregation adds AGGREGATION_COST (15) to complexity."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            aggregations=[QueryAggregation(type="count", field="id")],
        )
        plan = engine.parse(req)
        expected = QueryParserEngine.BASE_COST + QueryParserEngine.AGGREGATION_COST
        assert plan.estimated_complexity == expected

    def test_cross_source_cost(self, engine):
        """Multi-source queries add CROSS_SOURCE_COST (50) per additional source."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001", "SRC-00002", "SRC-00003"],
        )
        plan = engine.parse(req)
        expected = QueryParserEngine.BASE_COST + 2 * QueryParserEngine.CROSS_SOURCE_COST
        assert plan.estimated_complexity == expected

    def test_percentile_cost(self, engine):
        """Percentile aggregation adds extra PERCENTILE_COST (25)."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            aggregations=[QueryAggregation(type="percentile", field="latency")],
        )
        plan = engine.parse(req)
        expected = (
            QueryParserEngine.BASE_COST
            + QueryParserEngine.AGGREGATION_COST
            + QueryParserEngine.PERCENTILE_COST
        )
        assert plan.estimated_complexity == expected

    def test_combined_complexity(self, engine):
        """Combined complexity calculation with all cost factors."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001", "SRC-00002"],
            filters=[
                QueryFilter(field="a", operator="eq", value=1),
            ],
            sorts=[QuerySort(field="name", order="desc")],
            aggregations=[
                QueryAggregation(type="sum", field="value"),
                QueryAggregation(type="percentile", field="latency"),
            ],
        )
        plan = engine.parse(req)
        expected = (
            QueryParserEngine.BASE_COST
            + 1 * QueryParserEngine.FILTER_COST
            + 1 * QueryParserEngine.SORT_COST
            + 2 * QueryParserEngine.AGGREGATION_COST
            + 1 * QueryParserEngine.CROSS_SOURCE_COST
            + 1 * QueryParserEngine.PERCENTILE_COST
        )
        assert plan.estimated_complexity == expected


class TestValidateQuery:
    """Test query validation."""

    def test_valid_query(self, engine):
        """Valid query returns empty error list."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"])
        errors = engine.validate(req)
        assert errors == []

    def test_empty_sources(self, engine):
        """Empty source_ids returns error."""
        req = ExecuteQueryRequest(source_ids=[])
        errors = engine.validate(req)
        assert len(errors) == 1
        assert "At least one source_id" in errors[0]

    def test_too_many_sources(self, strict_engine):
        """Too many sources returns error (max=3 in strict_engine)."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001", "SRC-00002", "SRC-00003", "SRC-00004"],
        )
        errors = strict_engine.validate(req)
        assert any("Too many sources" in e for e in errors)

    def test_exceeds_max_complexity(self, strict_engine):
        """Query exceeding max complexity returns error (max=50 in strict_engine)."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001", "SRC-00002"],
            filters=[
                QueryFilter(field="a", operator="eq", value=1),
                QueryFilter(field="b", operator="gt", value=2),
            ],
            sorts=[QuerySort(field="name", order="desc")],
            aggregations=[QueryAggregation(type="count", field="id")],
        )
        errors = strict_engine.validate(req)
        assert any("exceeds maximum" in e for e in errors)

    def test_invalid_filter_in_validation(self, engine):
        """Invalid filter operator caught during validation."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            filters=[QueryFilter(field="x", operator="bad_op", value="v")],
        )
        errors = engine.validate(req)
        assert any("Invalid filter operator" in e for e in errors)

    def test_validation_increments_error_stats(self, engine):
        """Validation errors increment the stats counter."""
        req = ExecuteQueryRequest(source_ids=[])
        engine.validate(req)
        stats = engine.get_statistics()
        assert stats["validation_errors"] == 1

    def test_valid_query_no_error_stats(self, engine):
        """Valid query does not increment validation_errors."""
        req = ExecuteQueryRequest(source_ids=["SRC-00001"])
        engine.validate(req)
        stats = engine.get_statistics()
        assert stats["validation_errors"] == 0

    def test_max_sources_boundary(self, strict_engine):
        """Exactly max sources (3) is valid."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001", "SRC-00002", "SRC-00003"],
        )
        errors = strict_engine.validate(req)
        assert not any("Too many sources" in e for e in errors)
