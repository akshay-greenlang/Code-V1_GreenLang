# -*- coding: utf-8 -*-
"""
Performance optimizer for the Factors catalog (F080).

Provides query analysis, slow-query detection, automatic query plan caching,
and bulk-fetch optimization for high-throughput factor retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueryProfile:
    """Captured profile for a single query execution."""

    query_hash: str
    query_text: str
    execution_time_ms: float
    rows_returned: int
    plan_cost: float = 0.0
    used_index: bool = False
    index_name: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceReport:
    """Aggregated performance report."""

    total_queries: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    slow_query_count: int = 0
    index_hit_ratio: float = 0.0
    top_slow_queries: List[QueryProfile] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "slow_query_count": self.slow_query_count,
            "index_hit_ratio": round(self.index_hit_ratio, 4),
            "top_slow_queries": [
                {"query_hash": q.query_hash, "time_ms": round(q.execution_time_ms, 2)}
                for q in self.top_slow_queries[:10]
            ],
            "recommendations": self.recommendations,
        }


class PerformanceOptimizer:
    """
    Tracks query performance and provides optimization recommendations.

    Features:
      - Query profiling with automatic slow-query detection
      - Query plan caching for repeated patterns
      - Bulk-fetch chunking for large result sets
      - Performance report generation
    """

    SLOW_QUERY_THRESHOLD_MS = 500.0
    MAX_PROFILES = 10000

    def __init__(self, slow_threshold_ms: float = SLOW_QUERY_THRESHOLD_MS) -> None:
        self._slow_threshold = slow_threshold_ms
        self._profiles: List[QueryProfile] = []
        self._plan_cache: Dict[str, Any] = {}
        self._query_counts: Dict[str, int] = {}

    @staticmethod
    def _hash_query(query_text: str) -> str:
        return hashlib.sha256(query_text.encode()).hexdigest()[:16]

    def record_query(
        self,
        query_text: str,
        execution_time_ms: float,
        rows_returned: int,
        plan_cost: float = 0.0,
        used_index: bool = False,
        index_name: str = "",
    ) -> QueryProfile:
        """Record a query execution for profiling."""
        qhash = self._hash_query(query_text)
        profile = QueryProfile(
            query_hash=qhash,
            query_text=query_text,
            execution_time_ms=execution_time_ms,
            rows_returned=rows_returned,
            plan_cost=plan_cost,
            used_index=used_index,
            index_name=index_name,
        )
        self._profiles.append(profile)
        self._query_counts[qhash] = self._query_counts.get(qhash, 0) + 1

        # Evict old profiles to bound memory
        if len(self._profiles) > self.MAX_PROFILES:
            self._profiles = self._profiles[-self.MAX_PROFILES:]

        if execution_time_ms > self._slow_threshold:
            logger.warning(
                "Slow query detected: hash=%s time_ms=%.1f rows=%d",
                qhash, execution_time_ms, rows_returned,
            )
        return profile

    def cache_plan(self, query_text: str, plan: Any) -> None:
        """Cache an execution plan for a query pattern."""
        qhash = self._hash_query(query_text)
        self._plan_cache[qhash] = plan

    def get_cached_plan(self, query_text: str) -> Optional[Any]:
        """Retrieve a cached plan if available."""
        return self._plan_cache.get(self._hash_query(query_text))

    def get_report(self) -> PerformanceReport:
        """Generate a performance report from recorded profiles."""
        if not self._profiles:
            return PerformanceReport()

        latencies = sorted(p.execution_time_ms for p in self._profiles)
        total = len(latencies)
        index_hits = sum(1 for p in self._profiles if p.used_index)
        slow = [p for p in self._profiles if p.execution_time_ms > self._slow_threshold]

        report = PerformanceReport(
            total_queries=total,
            avg_latency_ms=sum(latencies) / total,
            p95_latency_ms=latencies[int(total * 0.95)] if total > 1 else latencies[0],
            p99_latency_ms=latencies[int(total * 0.99)] if total > 1 else latencies[0],
            slow_query_count=len(slow),
            index_hit_ratio=index_hits / total if total else 0.0,
            top_slow_queries=sorted(slow, key=lambda p: -p.execution_time_ms)[:10],
        )

        # Generate recommendations
        if report.index_hit_ratio < 0.8:
            report.recommendations.append(
                f"Index hit ratio is {report.index_hit_ratio:.0%}. "
                "Consider adding indexes on frequently queried columns."
            )
        if report.slow_query_count > total * 0.05:
            report.recommendations.append(
                f"{report.slow_query_count} slow queries ({report.slow_query_count/total:.0%}). "
                "Review query patterns and consider denormalization."
            )
        # Identify hot queries that benefit from caching
        hot = [(h, c) for h, c in self._query_counts.items() if c > 10]
        if hot:
            top_hot = sorted(hot, key=lambda x: -x[1])[:5]
            for qhash, count in top_hot:
                if qhash not in self._plan_cache:
                    report.recommendations.append(
                        f"Query {qhash} executed {count} times — cache its plan."
                    )

        return report

    @staticmethod
    def optimal_batch_size(total_rows: int, row_size_bytes: int = 500) -> int:
        """Calculate optimal batch size for bulk factor retrieval."""
        target_bytes = 4 * 1024 * 1024  # 4 MB per batch
        batch = max(100, min(5000, target_bytes // max(row_size_bytes, 1)))
        return min(batch, total_rows)

    def clear(self) -> None:
        """Reset all profiles and caches."""
        self._profiles.clear()
        self._plan_cache.clear()
        self._query_counts.clear()
