# -*- coding: utf-8 -*-
"""
Health Probe Orchestration Engine - AGENT-FOUND-010: Observability & Telemetry Agent

Provides orchestrated health probing for all GreenLang platform dependencies
with support for liveness, readiness, and startup probes. Probes execute
with configurable timeouts and maintain history for trend analysis. All
probe results include SHA-256 provenance hashes for audit trails.

Zero-Hallucination Guarantees:
    - All health statuses are deterministic (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
    - Aggregated status uses worst-case logic (deterministic)
    - Timeout enforcement uses pure arithmetic on UTC timestamps
    - No probabilistic health scoring or prediction

Example:
    >>> from greenlang.observability_agent.health_checker import HealthChecker
    >>> from greenlang.observability_agent.config import ObservabilityConfig
    >>> checker = HealthChecker(ObservabilityConfig())
    >>> checker.register_probe("database", "readiness", lambda: (True, "OK"))
    >>> result = checker.run_probe("database")
    >>> print(result.status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PROBE_TYPES: Tuple[str, ...] = ("liveness", "readiness", "startup")
VALID_HEALTH_STATUSES: Tuple[str, ...] = ("HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN")

# Severity ordering for worst-case aggregation
_STATUS_SEVERITY: Dict[str, int] = {
    "HEALTHY": 0,
    "DEGRADED": 1,
    "UNHEALTHY": 2,
    "UNKNOWN": 3,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ProbeDefinition:
    """Definition of a registered health probe.

    Attributes:
        probe_id: Unique identifier for this probe.
        name: Human-readable probe name (must be unique).
        probe_type: One of liveness, readiness, startup.
        check_func: Callable that returns (success: bool, message: str).
        interval_seconds: How often the probe should run.
        timeout_seconds: Maximum execution time for the probe.
        enabled: Whether the probe is currently active.
        created_at: Registration timestamp.
    """

    probe_id: str = ""
    name: str = ""
    probe_type: str = "readiness"
    check_func: Optional[Callable[[], Tuple[bool, str]]] = None
    interval_seconds: int = 30
    timeout_seconds: int = 5
    enabled: bool = True
    created_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        """Generate probe_id if not provided."""
        if not self.probe_id:
            self.probe_id = str(uuid.uuid4())


@dataclass
class HealthProbeResult:
    """Result of a health probe execution.

    Attributes:
        result_id: Unique identifier for this result.
        probe_name: Name of the probe that produced this result.
        probe_type: Type of probe (liveness, readiness, startup).
        status: Health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN).
        message: Human-readable status message.
        latency_ms: Probe execution duration in milliseconds.
        checked_at: Timestamp when the probe was executed.
        timed_out: Whether the probe timed out.
        error: Error message if the probe failed.
        provenance_hash: SHA-256 hash for audit trail.
    """

    result_id: str = ""
    probe_name: str = ""
    probe_type: str = "readiness"
    status: str = "UNKNOWN"
    message: str = ""
    latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=_utcnow)
    timed_out: bool = False
    error: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate result_id if not provided."""
        if not self.result_id:
            self.result_id = str(uuid.uuid4())


@dataclass
class HealthStatus:
    """Aggregated health status across all probes.

    Attributes:
        overall_status: Worst-case aggregated status.
        probe_results: Individual probe results.
        total_probes: Number of probes evaluated.
        healthy_count: Number of healthy probes.
        degraded_count: Number of degraded probes.
        unhealthy_count: Number of unhealthy probes.
        unknown_count: Number of probes with unknown status.
        checked_at: Timestamp of this aggregation.
        provenance_hash: SHA-256 hash for audit trail.
    """

    overall_status: str = "UNKNOWN"
    probe_results: List[HealthProbeResult] = field(default_factory=list)
    total_probes: int = 0
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0
    unknown_count: int = 0
    checked_at: datetime = field(default_factory=_utcnow)
    provenance_hash: str = ""


# =============================================================================
# HealthChecker
# =============================================================================


class HealthChecker:
    """Orchestrated health probe execution engine.

    Manages probe registration, execution with timeouts, result history,
    and aggregated health status computation.

    Thread-safe via a reentrant lock on all mutating operations.

    Attributes:
        _config: Observability configuration.
        _probes: Registered probes keyed by name.
        _history: Probe result history keyed by probe name.
        _latest_results: Most recent result per probe.
        _total_probes_run: Running count of all probe executions.
        _executor: Thread pool for timeout enforcement.
        _lock: Thread lock for concurrent access.

    Example:
        >>> checker = HealthChecker(config)
        >>> checker.register_probe("redis", "readiness",
        ...     lambda: (True, "Connected"), interval_seconds=15)
        >>> status = checker.get_aggregated_status()
        >>> print(status.overall_status)
    """

    def __init__(self, config: Any) -> None:
        """Initialize HealthChecker.

        Args:
            config: Observability configuration. May expose
                    ``health_history_limit``, ``health_probe_workers``.
        """
        self._config = config
        self._probes: Dict[str, ProbeDefinition] = {}
        self._history: Dict[str, Deque[HealthProbeResult]] = {}
        self._latest_results: Dict[str, HealthProbeResult] = {}
        self._total_probes_run: int = 0
        self._total_failures: int = 0
        self._lock = threading.RLock()

        self._history_limit: int = getattr(config, "health_history_limit", 100)
        self._worker_count: int = getattr(config, "health_probe_workers", 4)
        self._executor = ThreadPoolExecutor(
            max_workers=self._worker_count,
            thread_name_prefix="health-probe",
        )

        logger.info(
            "HealthChecker initialized: history_limit=%d, workers=%d",
            self._history_limit,
            self._worker_count,
        )

    # ------------------------------------------------------------------
    # Probe registration
    # ------------------------------------------------------------------

    def register_probe(
        self,
        name: str,
        probe_type: str,
        check_func: Callable[[], Tuple[bool, str]],
        interval_seconds: int = 30,
        timeout_seconds: int = 5,
    ) -> ProbeDefinition:
        """Register a new health probe.

        Args:
            name: Unique probe name.
            probe_type: One of liveness, readiness, startup.
            check_func: Callable returning (success: bool, message: str).
            interval_seconds: Recommended interval between executions.
            timeout_seconds: Maximum execution time for the probe.

        Returns:
            ProbeDefinition for the registered probe.

        Raises:
            ValueError: If name is empty, type is invalid, or probe exists.
        """
        if not name or not name.strip():
            raise ValueError("Probe name must be non-empty")

        if probe_type not in VALID_PROBE_TYPES:
            raise ValueError(
                f"Invalid probe_type '{probe_type}'; must be one of {VALID_PROBE_TYPES}"
            )

        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        with self._lock:
            if name in self._probes:
                raise ValueError(f"Probe '{name}' is already registered")

            probe = ProbeDefinition(
                name=name,
                probe_type=probe_type,
                check_func=check_func,
                interval_seconds=interval_seconds,
                timeout_seconds=timeout_seconds,
            )
            self._probes[name] = probe
            self._history[name] = deque(maxlen=self._history_limit)

        logger.info(
            "Registered probe: name=%s, type=%s, interval=%ds, timeout=%ds",
            name, probe_type, interval_seconds, timeout_seconds,
        )
        return probe

    def unregister_probe(self, name: str) -> bool:
        """Unregister a health probe.

        Args:
            name: Probe name to remove.

        Returns:
            True if probe was found and removed, False otherwise.
        """
        with self._lock:
            if name not in self._probes:
                return False

            del self._probes[name]
            self._history.pop(name, None)
            self._latest_results.pop(name, None)

        logger.info("Unregistered probe: name=%s", name)
        return True

    def enable_probe(self, name: str) -> ProbeDefinition:
        """Enable a disabled probe.

        Args:
            name: Probe name to enable.

        Returns:
            Updated ProbeDefinition.

        Raises:
            ValueError: If probe not found.
        """
        with self._lock:
            probe = self._probes.get(name)
            if probe is None:
                raise ValueError(f"Probe '{name}' not found")
            probe.enabled = True
        logger.info("Enabled probe: name=%s", name)
        return probe

    def disable_probe(self, name: str) -> ProbeDefinition:
        """Disable a probe without removing it.

        Args:
            name: Probe name to disable.

        Returns:
            Updated ProbeDefinition.

        Raises:
            ValueError: If probe not found.
        """
        with self._lock:
            probe = self._probes.get(name)
            if probe is None:
                raise ValueError(f"Probe '{name}' not found")
            probe.enabled = False
        logger.info("Disabled probe: name=%s", name)
        return probe

    def list_probes(self) -> List[ProbeDefinition]:
        """List all registered probes.

        Returns:
            List of ProbeDefinition objects sorted by name.
        """
        with self._lock:
            probes = list(self._probes.values())
        probes.sort(key=lambda p: p.name)
        return probes

    # ------------------------------------------------------------------
    # Probe execution
    # ------------------------------------------------------------------

    def run_probe(self, name: str) -> HealthProbeResult:
        """Execute a single probe and record the result.

        Args:
            name: Probe name to execute.

        Returns:
            HealthProbeResult with status and latency.

        Raises:
            ValueError: If probe not found.
        """
        with self._lock:
            probe = self._probes.get(name)
            if probe is None:
                raise ValueError(f"Probe '{name}' not found")

        result = self._execute_probe_with_timeout(probe)

        with self._lock:
            self._latest_results[name] = result
            if name in self._history:
                self._history[name].append(result)
            self._total_probes_run += 1
            if result.status in ("UNHEALTHY", "UNKNOWN"):
                self._total_failures += 1

        logger.debug(
            "Probe result: name=%s, status=%s, latency=%.1fms",
            name, result.status, result.latency_ms,
        )
        return result

    def run_all_probes(
        self,
        probe_type_filter: Optional[str] = None,
    ) -> List[HealthProbeResult]:
        """Execute all enabled probes, optionally filtered by type.

        Args:
            probe_type_filter: If provided, only run probes of this type.

        Returns:
            List of HealthProbeResult objects.
        """
        with self._lock:
            probes_to_run = [
                p for p in self._probes.values()
                if p.enabled and (probe_type_filter is None or p.probe_type == probe_type_filter)
            ]

        results: List[HealthProbeResult] = []
        for probe in probes_to_run:
            result = self._execute_probe_with_timeout(probe)

            with self._lock:
                self._latest_results[probe.name] = result
                if probe.name in self._history:
                    self._history[probe.name].append(result)
                self._total_probes_run += 1
                if result.status in ("UNHEALTHY", "UNKNOWN"):
                    self._total_failures += 1

            results.append(result)

        logger.info(
            "Ran %d probes: %d healthy, %d unhealthy",
            len(results),
            sum(1 for r in results if r.status == "HEALTHY"),
            sum(1 for r in results if r.status in ("UNHEALTHY", "UNKNOWN")),
        )
        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def get_aggregated_status(self) -> HealthStatus:
        """Compute aggregated health status using worst-case logic.

        If any probe is UNHEALTHY, overall status is UNHEALTHY.
        If any probe is DEGRADED, overall status is DEGRADED.
        If no probes have been executed, overall status is UNKNOWN.

        Returns:
            HealthStatus with overall status and per-probe results.
        """
        with self._lock:
            results = list(self._latest_results.values())

        if not results:
            return HealthStatus(
                overall_status="UNKNOWN",
                checked_at=_utcnow(),
            )

        healthy = sum(1 for r in results if r.status == "HEALTHY")
        degraded = sum(1 for r in results if r.status == "DEGRADED")
        unhealthy = sum(1 for r in results if r.status == "UNHEALTHY")
        unknown = sum(1 for r in results if r.status == "UNKNOWN")

        # Worst-case aggregation
        if unhealthy > 0:
            overall = "UNHEALTHY"
        elif unknown > 0:
            overall = "DEGRADED"
        elif degraded > 0:
            overall = "DEGRADED"
        else:
            overall = "HEALTHY"

        now = _utcnow()
        status = HealthStatus(
            overall_status=overall,
            probe_results=results,
            total_probes=len(results),
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            unknown_count=unknown,
            checked_at=now,
        )
        status.provenance_hash = self._compute_status_hash(status)

        return status

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_probe_history(
        self,
        name: str,
        limit: int = 10,
    ) -> List[HealthProbeResult]:
        """Get execution history for a specific probe.

        Args:
            name: Probe name.
            limit: Maximum number of results to return.

        Returns:
            List of HealthProbeResult objects, most recent first.

        Raises:
            ValueError: If probe not found.
        """
        with self._lock:
            if name not in self._probes:
                raise ValueError(f"Probe '{name}' not found")

            history = list(self._history.get(name, []))

        history.sort(key=lambda r: r.checked_at, reverse=True)
        return history[:limit]

    def get_latest_result(self, name: str) -> Optional[HealthProbeResult]:
        """Get the most recent result for a probe.

        Args:
            name: Probe name.

        Returns:
            HealthProbeResult or None if not yet executed.
        """
        with self._lock:
            return self._latest_results.get(name)

    # ------------------------------------------------------------------
    # Default probes
    # ------------------------------------------------------------------

    def register_default_probes(self) -> List[ProbeDefinition]:
        """Register default platform probes for database, redis, and prometheus.

        These probes use simple connectivity checks that return success
        by default. Replace check_func with real implementations in production.

        Returns:
            List of registered ProbeDefinition objects.
        """
        defaults = [
            ("database", "readiness", lambda: (True, "Database connection pool active"), 30, 5),
            ("redis", "readiness", lambda: (True, "Redis connection active"), 15, 3),
            ("prometheus", "liveness", lambda: (True, "Prometheus endpoint reachable"), 60, 5),
            ("disk_space", "liveness", lambda: (True, "Sufficient disk space available"), 120, 3),
            ("memory", "liveness", lambda: (True, "Memory usage within limits"), 60, 3),
        ]

        registered: List[ProbeDefinition] = []
        for name, ptype, func, interval, timeout in defaults:
            try:
                probe = self.register_probe(name, ptype, func, interval, timeout)
                registered.append(probe)
            except ValueError:
                logger.debug("Default probe '%s' already registered, skipping", name)

        logger.info("Registered %d default probes", len(registered))
        return registered

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get health checker statistics.

        Returns:
            Dictionary with total_probes_registered, total_probes_run,
            total_failures, probes_by_type, and latest results summary.
        """
        with self._lock:
            type_counts: Dict[str, int] = {}
            for p in self._probes.values():
                type_counts[p.probe_type] = type_counts.get(p.probe_type, 0) + 1

            status_counts: Dict[str, int] = {}
            for r in self._latest_results.values():
                status_counts[r.status] = status_counts.get(r.status, 0) + 1

            return {
                "total_probes_registered": len(self._probes),
                "enabled_probes": sum(1 for p in self._probes.values() if p.enabled),
                "total_probes_run": self._total_probes_run,
                "total_failures": self._total_failures,
                "probes_by_type": type_counts,
                "latest_status_counts": status_counts,
                "history_limit": self._history_limit,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_probe_with_timeout(
        self,
        probe: ProbeDefinition,
    ) -> HealthProbeResult:
        """Execute a probe with timeout enforcement.

        Args:
            probe: ProbeDefinition to execute.

        Returns:
            HealthProbeResult with status, message, and latency.
        """
        start = time.monotonic()
        now = _utcnow()

        if probe.check_func is None:
            return HealthProbeResult(
                probe_name=probe.name,
                probe_type=probe.probe_type,
                status="UNKNOWN",
                message="No check function configured",
                checked_at=now,
                provenance_hash=self._compute_result_hash(
                    probe.name, "UNKNOWN", "No check function", 0.0, now,
                ),
            )

        try:
            future = self._executor.submit(probe.check_func)
            success, message = future.result(timeout=probe.timeout_seconds)
            elapsed_ms = (time.monotonic() - start) * 1000.0

            status = "HEALTHY" if success else "UNHEALTHY"

            result = HealthProbeResult(
                probe_name=probe.name,
                probe_type=probe.probe_type,
                status=status,
                message=message,
                latency_ms=round(elapsed_ms, 2),
                checked_at=now,
            )
            result.provenance_hash = self._compute_result_hash(
                probe.name, status, message, elapsed_ms, now,
            )
            return result

        except FuturesTimeout:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            msg = f"Probe timed out after {probe.timeout_seconds}s"
            logger.warning("Probe '%s' timed out", probe.name)

            result = HealthProbeResult(
                probe_name=probe.name,
                probe_type=probe.probe_type,
                status="UNHEALTHY",
                message=msg,
                latency_ms=round(elapsed_ms, 2),
                checked_at=now,
                timed_out=True,
            )
            result.provenance_hash = self._compute_result_hash(
                probe.name, "UNHEALTHY", msg, elapsed_ms, now,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            error_msg = f"Probe failed: {str(exc)}"
            logger.error("Probe '%s' failed: %s", probe.name, exc, exc_info=True)

            result = HealthProbeResult(
                probe_name=probe.name,
                probe_type=probe.probe_type,
                status="UNHEALTHY",
                message=error_msg,
                latency_ms=round(elapsed_ms, 2),
                checked_at=now,
                error=str(exc),
            )
            result.provenance_hash = self._compute_result_hash(
                probe.name, "UNHEALTHY", error_msg, elapsed_ms, now,
            )
            return result

    def _compute_result_hash(
        self,
        probe_name: str,
        status: str,
        message: str,
        latency_ms: float,
        timestamp: datetime,
    ) -> str:
        """Compute SHA-256 provenance hash for a probe result.

        Args:
            probe_name: Probe name.
            status: Health status.
            message: Status message.
            latency_ms: Probe latency.
            timestamp: Check timestamp.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps(
            {
                "probe_name": probe_name,
                "status": status,
                "message": message,
                "latency_ms": latency_ms,
                "timestamp": timestamp.isoformat(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _compute_status_hash(self, status: HealthStatus) -> str:
        """Compute SHA-256 provenance hash for aggregated health status.

        Args:
            status: HealthStatus to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        probe_summaries = [
            {"name": r.probe_name, "status": r.status}
            for r in status.probe_results
        ]
        payload = json.dumps(
            {
                "overall_status": status.overall_status,
                "total_probes": status.total_probes,
                "healthy": status.healthy_count,
                "unhealthy": status.unhealthy_count,
                "probes": probe_summaries,
                "timestamp": status.checked_at.isoformat(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "HealthChecker",
    "ProbeDefinition",
    "HealthProbeResult",
    "HealthStatus",
    "VALID_PROBE_TYPES",
    "VALID_HEALTH_STATUSES",
]
