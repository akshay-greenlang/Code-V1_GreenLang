# -*- coding: utf-8 -*-
"""
RBAC Metrics - Authorization Service (SEC-002)

Prometheus counters, gauges, and histograms for RBAC authorization
observability.  Metrics are lazily initialized on first use so that the
module can be imported even when ``prometheus_client`` is not installed
(metrics become no-ops).

Registered metrics:
    - gl_rbac_authorization_total (Counter): Authorization decisions by result/resource/action/tenant.
    - gl_rbac_authorization_duration_seconds (Histogram): Authorization evaluation latency.
    - gl_rbac_cache_hits_total (Counter): RBAC cache hits by tenant.
    - gl_rbac_cache_misses_total (Counter): RBAC cache misses by tenant.
    - gl_rbac_roles_total (Gauge): Total roles by tenant and system flag.
    - gl_rbac_assignments_total (Gauge): Total active assignments by tenant.
    - gl_rbac_role_changes_total (Counter): Role change operations by action/tenant.

Classes:
    - RBACMetrics: Singleton-style RBAC metrics manager.

Example:
    >>> metrics = RBACMetrics()
    >>> metrics.record_authorization("allowed", "reports", "read", "t-corp", 0.002, True)
    >>> metrics.record_cache_hit("t-corp")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: lazy Prometheus metric handles
# ---------------------------------------------------------------------------


class _PrometheusHandles:
    """Lazy-initialized Prometheus metric objects.

    Metrics are created on first call to :meth:`ensure_initialized`.
    If ``prometheus_client`` is not installed, all handles remain ``None``
    and recording methods become safe no-ops.
    """

    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()
    _available: bool = False

    # Counters
    authorization_total: Any = None
    cache_hits_total: Any = None
    cache_misses_total: Any = None
    role_changes_total: Any = None

    # Gauges
    roles_total: Any = None
    assignments_total: Any = None

    # Histograms
    authorization_duration: Any = None

    @classmethod
    def ensure_initialized(cls) -> bool:
        """Create Prometheus metrics if the library is available.

        Thread-safe via a class-level lock.

        Returns:
            ``True`` if prometheus_client is available and metrics are
            registered, ``False`` otherwise.
        """
        if cls._initialized:
            return cls._available

        with cls._lock:
            if cls._initialized:
                return cls._available

            cls._initialized = True

            try:
                from prometheus_client import Counter, Gauge, Histogram
            except ImportError:
                logger.info(
                    "prometheus_client not installed; RBAC metrics are no-ops"
                )
                cls._available = False
                return False

            prefix = "gl_rbac"

            # -- Counters --------------------------------------------------

            cls.authorization_total = Counter(
                f"{prefix}_authorization_total",
                "Total authorization decisions",
                ["result", "resource", "action", "tenant_id"],
            )

            cls.cache_hits_total = Counter(
                f"{prefix}_cache_hits_total",
                "Total RBAC cache hits",
                ["tenant_id"],
            )

            cls.cache_misses_total = Counter(
                f"{prefix}_cache_misses_total",
                "Total RBAC cache misses",
                ["tenant_id"],
            )

            cls.role_changes_total = Counter(
                f"{prefix}_role_changes_total",
                "Total role change operations",
                ["action", "tenant_id"],
            )

            # -- Gauges ----------------------------------------------------

            cls.roles_total = Gauge(
                f"{prefix}_roles_total",
                "Total number of roles",
                ["tenant_id", "is_system"],
            )

            cls.assignments_total = Gauge(
                f"{prefix}_assignments_total",
                "Total active role assignments",
                ["tenant_id"],
            )

            # -- Histograms ------------------------------------------------

            cls.authorization_duration = Histogram(
                f"{prefix}_authorization_duration_seconds",
                "Authorization evaluation duration in seconds",
                ["result", "cache_hit"],
                buckets=(
                    0.0001, 0.00025, 0.0005, 0.001, 0.0025,
                    0.005, 0.01, 0.025, 0.05, 0.1,
                ),
            )

            cls._available = True
            logger.info("RBAC Prometheus metrics registered successfully")
            return True


# ---------------------------------------------------------------------------
# RBACMetrics
# ---------------------------------------------------------------------------


class RBACMetrics:
    """Manages Prometheus metrics for the RBAC authorization service.

    All recording methods are safe no-ops when ``prometheus_client`` is
    not installed.  Thread-safe.

    Example:
        >>> m = RBACMetrics()
        >>> m.record_authorization("allowed", "reports", "read", "t-1", 0.001, True)
        >>> m.record_cache_hit("t-1")
    """

    def __init__(self, prefix: str = "gl_rbac") -> None:
        """Initialize RBAC metrics.

        Args:
            prefix: Metric name prefix (used for documentation only;
                actual prefix is fixed at registration time).
        """
        self._prefix = prefix
        self._available = _PrometheusHandles.ensure_initialized()

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def record_authorization(
        self,
        result: str,
        resource: str,
        action: str,
        tenant_id: str,
        duration_s: float,
        cache_hit: bool,
    ) -> None:
        """Record an authorization decision.

        Args:
            result: Outcome (``allowed``, ``denied``).
            resource: Resource that was checked (e.g. ``reports``).
            action: Action that was checked (e.g. ``read``).
            tenant_id: UUID of the tenant.
            duration_s: Evaluation duration in seconds.
            cache_hit: Whether the result came from cache.
        """
        if not self._available:
            return

        _PrometheusHandles.authorization_total.labels(
            result=result, resource=resource, action=action, tenant_id=tenant_id
        ).inc()

        _PrometheusHandles.authorization_duration.labels(
            result=result, cache_hit=str(cache_hit).lower()
        ).observe(duration_s)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def record_cache_hit(self, tenant_id: str) -> None:
        """Record an RBAC cache hit.

        Args:
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.cache_hits_total.labels(
            tenant_id=tenant_id
        ).inc()

    def record_cache_miss(self, tenant_id: str) -> None:
        """Record an RBAC cache miss.

        Args:
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.cache_misses_total.labels(
            tenant_id=tenant_id
        ).inc()

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    def set_roles_count(
        self,
        tenant_id: str,
        is_system: bool,
        count: int,
    ) -> None:
        """Set the total roles gauge for a tenant.

        Args:
            tenant_id: UUID of the tenant.
            is_system: Whether these are system-defined roles.
            count: Current role count.
        """
        if not self._available:
            return

        _PrometheusHandles.roles_total.labels(
            tenant_id=tenant_id, is_system=str(is_system).lower()
        ).set(count)

    # ------------------------------------------------------------------
    # Assignments
    # ------------------------------------------------------------------

    def set_assignments_count(self, tenant_id: str, count: int) -> None:
        """Set the active assignments gauge for a tenant.

        Args:
            tenant_id: UUID of the tenant.
            count: Current active assignment count.
        """
        if not self._available:
            return

        _PrometheusHandles.assignments_total.labels(
            tenant_id=tenant_id
        ).set(count)

    # ------------------------------------------------------------------
    # Role changes
    # ------------------------------------------------------------------

    def record_role_change(self, action: str, tenant_id: str) -> None:
        """Record a role change operation.

        Args:
            action: Change type (``created``, ``updated``, ``deleted``,
                ``enabled``, ``disabled``).
            tenant_id: UUID of the tenant.
        """
        if not self._available:
            return

        _PrometheusHandles.role_changes_total.labels(
            action=action, tenant_id=tenant_id
        ).inc()
