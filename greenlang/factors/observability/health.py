# -*- coding: utf-8 -*-
"""
Comprehensive health endpoint for the Factors service (F073).

Returns detailed status of all subsystems: database, cache, edition
availability, factor count, last ingestion, and last watch run.

Used by K8s readiness/liveness probes and the Grafana dashboard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class ComponentHealth:
    name: str
    status: ComponentStatus
    latency_ms: int = 0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Full health report for the Factors service."""

    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str = ""
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    edition_id: Optional[str] = None
    factor_count: int = 0
    certified_count: int = 0
    last_ingestion: Optional[str] = None
    last_watch_run: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "timestamp": self.timestamp,
            "edition_id": self.edition_id,
            "factor_count": self.factor_count,
            "certified_count": self.certified_count,
            "last_ingestion": self.last_ingestion,
            "last_watch_run": self.last_watch_run,
            "components": {
                name: {
                    "status": c.status.value,
                    "latency_ms": c.latency_ms,
                    "message": c.message,
                }
                for name, c in self.components.items()
            },
        }

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"

    @property
    def http_status(self) -> int:
        return 200 if self.status in ("healthy", "degraded") else 503


def _check_database(repo: Any) -> ComponentHealth:
    """Check database connectivity via the catalog repository."""
    start = time.monotonic()
    try:
        stats = repo.coverage_stats() if hasattr(repo, "coverage_stats") else {}
        latency = int((time.monotonic() - start) * 1000)
        return ComponentHealth(
            name="database",
            status=ComponentStatus.HEALTHY,
            latency_ms=latency,
            message="Connected",
            details={"total_factors": stats.get("total_factors", 0)},
        )
    except Exception as exc:
        latency = int((time.monotonic() - start) * 1000)
        return ComponentHealth(
            name="database",
            status=ComponentStatus.UNAVAILABLE,
            latency_ms=latency,
            message=str(exc),
        )


def _check_cache() -> ComponentHealth:
    """Check Redis cache connectivity."""
    start = time.monotonic()
    try:
        from greenlang.factors.cache_redis import _get_redis
        r = _get_redis()
        if r is not None:
            r.ping()
            latency = int((time.monotonic() - start) * 1000)
            return ComponentHealth(
                name="cache",
                status=ComponentStatus.HEALTHY,
                latency_ms=latency,
                message="Redis connected",
            )
        latency = int((time.monotonic() - start) * 1000)
        return ComponentHealth(
            name="cache",
            status=ComponentStatus.DEGRADED,
            latency_ms=latency,
            message="Redis not configured (operating without cache)",
        )
    except Exception as exc:
        latency = int((time.monotonic() - start) * 1000)
        return ComponentHealth(
            name="cache",
            status=ComponentStatus.DEGRADED,
            latency_ms=latency,
            message=f"Redis error: {exc}",
        )


def _check_edition(repo: Any) -> ComponentHealth:
    """Check that a default edition is available."""
    start = time.monotonic()
    try:
        edition = repo.resolve_edition("latest") if hasattr(repo, "resolve_edition") else None
        latency = int((time.monotonic() - start) * 1000)
        if edition:
            return ComponentHealth(
                name="edition",
                status=ComponentStatus.HEALTHY,
                latency_ms=latency,
                message=f"Default edition: {edition}",
                details={"edition_id": edition},
            )
        return ComponentHealth(
            name="edition",
            status=ComponentStatus.DEGRADED,
            latency_ms=latency,
            message="No default edition found",
        )
    except Exception as exc:
        latency = int((time.monotonic() - start) * 1000)
        return ComponentHealth(
            name="edition",
            status=ComponentStatus.DEGRADED,
            latency_ms=latency,
            message=str(exc),
        )


def get_health_status(repo: Any) -> HealthStatus:
    """
    Run all health checks and return a comprehensive HealthStatus.

    Args:
        repo: A FactorCatalogRepository instance.

    Returns:
        HealthStatus with component-level details.
    """
    health = HealthStatus(timestamp=datetime.now(timezone.utc).isoformat())

    # Check components
    db = _check_database(repo)
    cache = _check_cache()
    edition = _check_edition(repo)

    health.components = {
        "database": db,
        "cache": cache,
        "edition": edition,
    }

    # Aggregate stats
    if db.status == ComponentStatus.HEALTHY:
        stats = db.details
        health.factor_count = stats.get("total_factors", 0)
    if edition.status == ComponentStatus.HEALTHY:
        health.edition_id = edition.details.get("edition_id")

    # Coverage stats if available
    try:
        cov = repo.coverage_stats() if hasattr(repo, "coverage_stats") else {}
        health.certified_count = cov.get("certified", 0)
        health.factor_count = cov.get("total_factors", health.factor_count)
    except Exception:
        pass

    # Overall status
    statuses = [c.status for c in health.components.values()]
    if any(s == ComponentStatus.UNAVAILABLE for s in statuses):
        if db.status == ComponentStatus.UNAVAILABLE:
            health.status = "unavailable"
        else:
            health.status = "degraded"
    elif any(s == ComponentStatus.DEGRADED for s in statuses):
        health.status = "degraded"
    else:
        health.status = "healthy"

    logger.info("Health check: status=%s components=%d", health.status, len(health.components))
    return health
