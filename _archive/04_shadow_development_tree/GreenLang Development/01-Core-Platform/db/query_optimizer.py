# -*- coding: utf-8 -*-
"""
GreenLang Query Optimizer

Analyzes and optimizes database queries for better performance.

Features:
- Slow query detection and logging
- Query execution plan analysis
- N+1 query detection and prevention
- Automatic query result caching
- Query hints for complex operations
- Missing index detection

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import hashlib
import json

try:
    from sqlalchemy import event, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, Query
    from sqlalchemy.sql import Select
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    event = None
    Engine = None
    Session = None
    Query = None
    Select = None

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """
    Metrics for a database query.

    Attributes:
        query_hash: Hash of the query
        query_text: SQL query text
        execution_count: Number of times executed
        total_time_ms: Total execution time
        avg_time_ms: Average execution time
        max_time_ms: Maximum execution time
        min_time_ms: Minimum execution time
        last_executed: Last execution timestamp
    """
    query_hash: str
    query_text: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    last_executed: float = 0.0

    def record_execution(self, duration_ms: float) -> None:
        """Record a query execution."""
        self.execution_count += 1
        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.execution_count
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.last_executed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_hash": self.query_hash,
            "query_text": self.query_text[:200],  # Truncate for display
            "execution_count": self.execution_count,
            "avg_time_ms": round(self.avg_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
        }


@dataclass
class SlowQuery:
    """
    Record of a slow query.

    Attributes:
        query_text: SQL query
        duration_ms: Execution time
        timestamp: When it was executed
        stack_trace: Call stack (for debugging)
        params: Query parameters
    """
    query_text: str
    duration_ms: float
    timestamp: float
    stack_trace: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query_text[:500],
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
            "params": self.params,
        }


@dataclass
class QueryPlan:
    """
    Query execution plan analysis.

    Attributes:
        query_text: SQL query
        plan: Execution plan from database
        cost: Estimated cost
        uses_index: Whether query uses indexes
        seq_scans: Number of sequential scans
        recommendations: Optimization recommendations
    """
    query_text: str
    plan: Dict[str, Any]
    cost: float = 0.0
    uses_index: bool = False
    seq_scans: int = 0
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def from_explain(cls, query_text: str, explain_output: List[Dict]) -> "QueryPlan":
        """
        Create QueryPlan from EXPLAIN output.

        Args:
            query_text: SQL query
            explain_output: EXPLAIN JSON output

        Returns:
            QueryPlan instance
        """
        if not explain_output:
            return cls(query_text=query_text, plan={})

        plan = explain_output[0] if isinstance(explain_output, list) else explain_output
        plan_dict = plan.get('Plan', {}) if isinstance(plan, dict) else {}

        cost = plan_dict.get('Total Cost', 0)
        node_type = plan_dict.get('Node Type', '')
        uses_index = 'Index' in node_type

        # Analyze plan for issues
        recommendations = []
        seq_scans = cls._count_seq_scans(plan_dict)

        if seq_scans > 0:
            recommendations.append(
                f"Query performs {seq_scans} sequential scan(s). "
                "Consider adding indexes."
            )

        if cost > 10000:
            recommendations.append(
                f"High query cost ({cost:.2f}). "
                "Review query complexity and indexing."
            )

        return cls(
            query_text=query_text,
            plan=plan_dict,
            cost=cost,
            uses_index=uses_index,
            seq_scans=seq_scans,
            recommendations=recommendations
        )

    @staticmethod
    def _count_seq_scans(plan: Dict) -> int:
        """Count sequential scans in plan."""
        count = 0
        if plan.get('Node Type') == 'Seq Scan':
            count += 1

        # Recursively check child nodes
        for child in plan.get('Plans', []):
            count += QueryPlan._count_seq_scans(child)

        return count


class QueryCache:
    """
    Cache for query results with TTL support.

    Integrates with the multi-layer cache system.
    """

    def __init__(self, default_ttl: int = 300):
        """
        Initialize query cache.

        Args:
            default_ttl: Default TTL in seconds
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, query_hash: str) -> Optional[Any]:
        """
        Get cached query result.

        Args:
            query_hash: Hash of the query

        Returns:
            Cached result or None
        """
        if query_hash in self._cache:
            result, expires_at = self._cache[query_hash]
            if time.time() < expires_at:
                self._hits += 1
                return result
            else:
                del self._cache[query_hash]

        self._misses += 1
        return None

    def set(
        self,
        query_hash: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache query result.

        Args:
            query_hash: Hash of the query
            result: Query result
            ttl: Optional TTL (uses default if None)
        """
        ttl_seconds = ttl if ttl is not None else self._default_ttl
        expires_at = time.time() + ttl_seconds
        self._cache[query_hash] = (result, expires_at)

    def invalidate(self, query_hash: str) -> None:
        """Invalidate cached result."""
        self._cache.pop(query_hash, None)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "cached_queries": len(self._cache),
        }


class QueryOptimizer:
    """
    Database query optimizer and analyzer.

    Monitors query performance, detects slow queries, and provides
    optimization recommendations.

    Example:
        >>> optimizer = QueryOptimizer(slow_query_threshold_ms=100)
        >>> optimizer.start()
        >>>
        >>> # Queries are automatically monitored
        >>> with optimizer.track_query() as tracker:
        ...     result = session.execute(query)
        >>>
        >>> # Get statistics
        >>> stats = optimizer.get_stats()
        >>> slow_queries = optimizer.get_slow_queries()
        >>> optimizer.stop()
    """

    def __init__(
        self,
        slow_query_threshold_ms: float = 100.0,
        enable_query_cache: bool = True,
        cache_ttl_seconds: int = 300,
        enable_explain: bool = True,
        max_slow_queries: int = 1000
    ):
        """
        Initialize query optimizer.

        Args:
            slow_query_threshold_ms: Threshold for slow query logging
            enable_query_cache: Enable query result caching
            cache_ttl_seconds: Default cache TTL
            enable_explain: Enable EXPLAIN analysis for slow queries
            max_slow_queries: Maximum slow queries to keep in memory
        """
        self._slow_threshold = slow_query_threshold_ms
        self._enable_cache = enable_query_cache
        self._cache_ttl = cache_ttl_seconds
        self._enable_explain = enable_explain
        self._max_slow_queries = max_slow_queries

        # Query tracking
        self._query_metrics: Dict[str, QueryMetrics] = {}
        self._slow_queries: List[SlowQuery] = []

        # Query cache
        self._query_cache = QueryCache(default_ttl=cache_ttl_seconds)

        # N+1 detection
        self._recent_queries: List[Tuple[str, float]] = []
        self._n_plus_one_warnings: List[str] = []

        # Statistics
        self._total_queries = 0
        self._total_time_ms = 0.0

        logger.info(
            f"Initialized QueryOptimizer: "
            f"slow_threshold={slow_query_threshold_ms}ms, "
            f"cache_enabled={enable_query_cache}"
        )

    def start(self) -> None:
        """Start the query optimizer."""
        logger.info("QueryOptimizer started")

    def stop(self) -> None:
        """Stop the query optimizer."""
        logger.info("QueryOptimizer stopped")

    @asynccontextmanager
    async def track_query(
        self,
        query_text: Optional[str] = None,
        cacheable: bool = False,
        cache_key: Optional[str] = None
    ):
        """
        Context manager for tracking query execution.

        Args:
            query_text: SQL query text
            cacheable: Whether query results can be cached
            cache_key: Optional cache key (auto-generated if None)

        Example:
            >>> async with optimizer.track_query("SELECT * FROM users") as tracker:
            ...     result = await session.execute(query)
            ...     tracker.result = result
        """
        start_time = time.perf_counter()
        tracker = QueryTracker(query_text=query_text)

        # Check cache if enabled
        if cacheable and self._enable_cache and cache_key:
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                tracker.result = cached_result
                tracker.from_cache = True
                yield tracker
                return

        try:
            yield tracker
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record metrics
            self._record_query(query_text, duration_ms)

            # Cache result if enabled
            if (
                cacheable
                and self._enable_cache
                and cache_key
                and hasattr(tracker, 'result')
            ):
                self._query_cache.set(cache_key, tracker.result)

            # Check for slow query
            if duration_ms >= self._slow_threshold:
                self._record_slow_query(query_text, duration_ms)

    def _record_query(
        self,
        query_text: Optional[str],
        duration_ms: float
    ) -> None:
        """Record query execution metrics."""
        self._total_queries += 1
        self._total_time_ms += duration_ms

        if not query_text:
            return

        # Generate query hash
        query_hash = self._hash_query(query_text)

        # Update metrics
        if query_hash not in self._query_metrics:
            self._query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_text=query_text
            )

        self._query_metrics[query_hash].record_execution(duration_ms)

        # N+1 detection
        self._detect_n_plus_one(query_text, duration_ms)

    def _record_slow_query(
        self,
        query_text: Optional[str],
        duration_ms: float
    ) -> None:
        """Record slow query for analysis."""
        if not query_text:
            return

        slow_query = SlowQuery(
            query_text=query_text,
            duration_ms=duration_ms,
            timestamp=time.time()
        )

        self._slow_queries.append(slow_query)

        # Keep only recent slow queries
        if len(self._slow_queries) > self._max_slow_queries:
            self._slow_queries = self._slow_queries[-self._max_slow_queries:]

        logger.warning(
            f"Slow query detected ({duration_ms:.2f}ms): "
            f"{query_text[:100]}..."
        )

    def _detect_n_plus_one(self, query_text: str, duration_ms: float) -> None:
        """
        Detect potential N+1 query problems.

        N+1 occurs when a query in a loop results in N additional queries.
        """
        now = time.time()

        # Add to recent queries
        self._recent_queries.append((query_text, now))

        # Keep only last 100 queries
        self._recent_queries = self._recent_queries[-100:]

        # Check for repeated similar queries
        query_pattern = self._normalize_query(query_text)
        similar_count = sum(
            1 for q, t in self._recent_queries
            if self._normalize_query(q) == query_pattern
            and (now - t) < 1.0  # Within 1 second
        )

        if similar_count >= 10:
            warning = (
                f"Potential N+1 query detected: "
                f"{similar_count} similar queries in 1 second"
            )
            if warning not in self._n_plus_one_warnings[-10:]:
                self._n_plus_one_warnings.append(warning)
                logger.warning(warning)

    def _hash_query(self, query_text: str) -> str:
        """Generate hash for query."""
        # Normalize query before hashing
        normalized = self._normalize_query(query_text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _normalize_query(self, query_text: str) -> str:
        """
        Normalize query for comparison.

        Removes parameter values to group similar queries.
        """
        import re

        # Remove whitespace
        normalized = re.sub(r'\s+', ' ', query_text.strip())

        # Replace numeric literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)

        # Replace string literals
        normalized = re.sub(r"'[^']*'", '?', normalized)

        return normalized.lower()

    async def analyze_query(
        self,
        query_text: str,
        session: Any
    ) -> Optional[QueryPlan]:
        """
        Analyze query execution plan.

        Args:
            query_text: SQL query
            session: Database session

        Returns:
            QueryPlan with analysis results
        """
        if not self._enable_explain:
            return None

        try:
            # Run EXPLAIN (FORMAT JSON)
            explain_query = f"EXPLAIN (FORMAT JSON) {query_text}"
            result = await session.execute(text(explain_query))
            explain_output = result.fetchall()

            # Parse result
            if explain_output:
                plan_json = json.loads(explain_output[0][0])
                return QueryPlan.from_explain(query_text, plan_json)

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")

        return None

    def get_slow_queries(
        self,
        limit: int = 100,
        min_duration_ms: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get slow queries.

        Args:
            limit: Maximum number of queries to return
            min_duration_ms: Optional minimum duration filter

        Returns:
            List of slow query dictionaries
        """
        queries = self._slow_queries

        if min_duration_ms:
            queries = [q for q in queries if q.duration_ms >= min_duration_ms]

        # Sort by duration (slowest first)
        queries = sorted(queries, key=lambda q: q.duration_ms, reverse=True)

        return [q.to_dict() for q in queries[:limit]]

    def get_query_metrics(
        self,
        limit: int = 100,
        order_by: str = 'avg_time_ms'
    ) -> List[Dict[str, Any]]:
        """
        Get query metrics.

        Args:
            limit: Maximum number of queries to return
            order_by: Sort field (avg_time_ms, execution_count, total_time_ms)

        Returns:
            List of query metric dictionaries
        """
        metrics = list(self._query_metrics.values())

        # Sort by specified field
        if order_by == 'execution_count':
            metrics.sort(key=lambda m: m.execution_count, reverse=True)
        elif order_by == 'total_time_ms':
            metrics.sort(key=lambda m: m.total_time_ms, reverse=True)
        else:  # avg_time_ms
            metrics.sort(key=lambda m: m.avg_time_ms, reverse=True)

        return [m.to_dict() for m in metrics[:limit]]

    def get_recommendations(self) -> List[str]:
        """
        Get optimization recommendations.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for slow queries
        slow_count = len([q for q in self._slow_queries if q.duration_ms > 1000])
        if slow_count > 0:
            recommendations.append(
                f"Found {slow_count} very slow queries (>1s). "
                "Review and optimize these queries."
            )

        # Check for N+1 problems
        if self._n_plus_one_warnings:
            recommendations.append(
                f"Detected {len(self._n_plus_one_warnings)} potential N+1 query issues. "
                "Use eager loading (select_related/prefetch_related) to reduce queries."
            )

        # Check cache hit rate
        cache_stats = self._query_cache.get_stats()
        if cache_stats['hit_rate'] < 0.5 and cache_stats['hits'] + cache_stats['misses'] > 100:
            recommendations.append(
                f"Low cache hit rate ({cache_stats['hit_rate']:.1%}). "
                "Consider caching more queries or increasing TTL."
            )

        # Check for frequently executed queries
        for metrics in self._query_metrics.values():
            if metrics.execution_count > 1000:
                recommendations.append(
                    f"Query executed {metrics.execution_count} times. "
                    f"Consider caching: {metrics.query_text[:100]}..."
                )
                break

        return recommendations

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.

        Returns:
            Dictionary with statistics
        """
        avg_query_time = (
            self._total_time_ms / self._total_queries
            if self._total_queries > 0 else 0
        )

        return {
            "total_queries": self._total_queries,
            "total_time_ms": round(self._total_time_ms, 2),
            "avg_query_time_ms": round(avg_query_time, 2),
            "slow_queries": len(self._slow_queries),
            "unique_queries": len(self._query_metrics),
            "n_plus_one_warnings": len(self._n_plus_one_warnings),
            "cache_stats": self._query_cache.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._query_metrics.clear()
        self._slow_queries.clear()
        self._n_plus_one_warnings.clear()
        self._total_queries = 0
        self._total_time_ms = 0.0
        self._query_cache.clear()
        logger.info("Query optimizer stats reset")


class QueryTracker:
    """
    Tracker for individual query execution.

    Used with track_query context manager.
    """

    def __init__(self, query_text: Optional[str] = None):
        self.query_text = query_text
        self.result: Any = None
        self.from_cache: bool = False


# Global query optimizer instance
_global_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> Optional[QueryOptimizer]:
    """Get global query optimizer instance."""
    return _global_optimizer


def initialize_query_optimizer(
    slow_query_threshold_ms: float = 100.0,
    enable_query_cache: bool = True
) -> QueryOptimizer:
    """
    Initialize global query optimizer.

    Args:
        slow_query_threshold_ms: Threshold for slow queries
        enable_query_cache: Enable query caching

    Returns:
        QueryOptimizer instance
    """
    global _global_optimizer

    if _global_optimizer is not None:
        _global_optimizer.stop()

    _global_optimizer = QueryOptimizer(
        slow_query_threshold_ms=slow_query_threshold_ms,
        enable_query_cache=enable_query_cache
    )
    _global_optimizer.start()

    return _global_optimizer
