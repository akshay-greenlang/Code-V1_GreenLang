# -*- coding: utf-8 -*-
"""
Pilot usage telemetry (F092).

Captures granular usage events from pilot partners to inform product
decisions: which endpoints are most used, search patterns, error rates,
and feature adoption metrics.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    """A single usage event from a pilot partner."""

    event_id: str
    partner_id: str
    tenant_id: str
    event_type: str  # "api_call", "search", "match", "batch", "connector"
    endpoint: str
    method: str = "GET"
    status_code: int = 200
    latency_ms: float = 0.0
    result_count: int = 0
    query_params: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "partner_id": self.partner_id,
            "tenant_id": self.tenant_id,
            "event_type": self.event_type,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "latency_ms": round(self.latency_ms, 2),
            "result_count": self.result_count,
            "timestamp": self.timestamp,
        }


@dataclass
class PartnerMetrics:
    """Aggregated metrics for a single pilot partner."""

    partner_id: str
    total_requests: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    unique_endpoints: int = 0
    top_endpoints: List[str] = field(default_factory=list)
    top_search_terms: List[str] = field(default_factory=list)
    first_activity: str = ""
    last_activity: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partner_id": self.partner_id,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": round(self.error_count / max(self.total_requests, 1), 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "unique_endpoints": self.unique_endpoints,
            "top_endpoints": self.top_endpoints[:5],
            "top_search_terms": self.top_search_terms[:10],
            "first_activity": self.first_activity,
            "last_activity": self.last_activity,
        }


class PilotTelemetry:
    """
    Collects and analyzes usage telemetry from pilot partners.

    Provides:
      - Event recording
      - Per-partner metrics aggregation
      - Engagement scoring
      - Weekly activity summaries
    """

    MAX_EVENTS = 100000

    def __init__(self) -> None:
        self._events: List[UsageEvent] = []
        self._by_partner: Dict[str, List[UsageEvent]] = defaultdict(list)

    def record(self, event: UsageEvent) -> None:
        """Record a usage event."""
        if not event.timestamp:
            event.timestamp = datetime.now(timezone.utc).isoformat()
        self._events.append(event)
        self._by_partner[event.partner_id].append(event)

        # Bound memory
        if len(self._events) > self.MAX_EVENTS:
            self._events = self._events[-self.MAX_EVENTS:]

    def get_partner_metrics(self, partner_id: str) -> PartnerMetrics:
        """Compute aggregated metrics for a partner."""
        events = self._by_partner.get(partner_id, [])
        if not events:
            return PartnerMetrics(partner_id=partner_id)

        latencies = sorted(e.latency_ms for e in events)
        errors = sum(1 for e in events if e.status_code >= 400)
        endpoints = defaultdict(int)
        search_terms: Dict[str, int] = {}

        for e in events:
            endpoints[e.endpoint] += 1
            q = e.query_params.get("query", "")
            if q:
                search_terms[q] = search_terms.get(q, 0) + 1

        top_ep = sorted(endpoints.items(), key=lambda x: -x[1])
        top_st = sorted(search_terms.items(), key=lambda x: -x[1])
        total = len(events)

        return PartnerMetrics(
            partner_id=partner_id,
            total_requests=total,
            error_count=errors,
            avg_latency_ms=sum(latencies) / total,
            p95_latency_ms=latencies[int(total * 0.95)] if total > 1 else latencies[0],
            unique_endpoints=len(endpoints),
            top_endpoints=[ep for ep, _ in top_ep[:5]],
            top_search_terms=[st for st, _ in top_st[:10]],
            first_activity=events[0].timestamp,
            last_activity=events[-1].timestamp,
        )

    def engagement_score(self, partner_id: str) -> float:
        """
        Calculate an engagement score (0-100) for a partner.

        Factors: request volume, endpoint diversity, error rate, recency.
        """
        events = self._by_partner.get(partner_id, [])
        if not events:
            return 0.0

        total = len(events)
        endpoints = len(set(e.endpoint for e in events))
        errors = sum(1 for e in events if e.status_code >= 400)
        error_rate = errors / total

        # Volume score (0-30): log scale
        import math
        volume_score = min(30.0, math.log10(max(total, 1)) * 10)

        # Diversity score (0-30): unique endpoints
        diversity_score = min(30.0, endpoints * 6.0)

        # Quality score (0-20): inverse error rate
        quality_score = max(0.0, 20.0 * (1 - error_rate))

        # Recency score (0-20): days since last activity
        try:
            last = datetime.fromisoformat(events[-1].timestamp.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - last).days
            recency_score = max(0.0, 20.0 - days_ago * 2)
        except Exception:
            recency_score = 0.0

        return round(min(100.0, volume_score + diversity_score + quality_score + recency_score), 1)

    def weekly_summary(self) -> Dict[str, Any]:
        """Generate a weekly activity summary across all partners."""
        cutoff = time.time() - 7 * 86400
        recent = [e for e in self._events if e.timestamp and _ts_to_epoch(e.timestamp) > cutoff]

        by_partner: Dict[str, int] = defaultdict(int)
        by_endpoint: Dict[str, int] = defaultdict(int)
        errors = 0

        for e in recent:
            by_partner[e.partner_id] += 1
            by_endpoint[e.endpoint] += 1
            if e.status_code >= 400:
                errors += 1

        return {
            "period": "7d",
            "total_events": len(recent),
            "active_partners": len(by_partner),
            "error_count": errors,
            "top_partners": sorted(by_partner.items(), key=lambda x: -x[1])[:10],
            "top_endpoints": sorted(by_endpoint.items(), key=lambda x: -x[1])[:10],
        }

    @property
    def total_events(self) -> int:
        return len(self._events)


def _ts_to_epoch(ts: str) -> float:
    """Convert ISO timestamp to epoch seconds."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0
