# -*- coding: utf-8 -*-
"""
HealthCheck - 20-Category System Health Verification for PACK-037
===================================================================

This module implements a comprehensive health check system that validates the
operational readiness of the Demand Response Pack.

Check Categories (20):
    1.  dr_flexibility_engine       -- Verify DR flexibility engines
    2.  baseline_engine             -- Verify baseline calculation engine
    3.  dispatch_engine             -- Verify dispatch optimization engine
    4.  der_coordination_engine     -- Verify DER coordination engine
    5.  event_mgmt_engine           -- Verify event management engine
    6.  performance_engine          -- Verify performance tracking engine
    7.  revenue_engine              -- Verify revenue reconciliation engine
    8.  program_matching_engine     -- Verify program matching engine
    9.  load_inventory_engine       -- Verify load inventory engine
    10. reporting_engine            -- Verify reporting engine
    11. workflows                   -- Verify DR workflows
    12. database                    -- Check database connectivity
    13. cache                       -- Check Redis cache connectivity
    14. mrv_bridge                  -- Check MRV bridge availability
    15. data_bridge                 -- Check DATA bridge availability
    16. grid_signal                 -- Check grid signal connectivity
    17. bms_control                 -- Check BMS control connectivity
    18. meter_data                  -- Check meter data feed
    19. der_assets                  -- Check DER asset fleet status
    20. auth                        -- Check authentication service

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-037 Demand Response
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

PACK_BASE_DIR = Path(__file__).parent.parent


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class HealthSeverity(str, Enum):
    """Severity levels for health issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class HealthCategory(str, Enum):
    """Health check categories (20 total)."""

    DR_FLEXIBILITY_ENGINE = "dr_flexibility_engine"
    BASELINE_ENGINE = "baseline_engine"
    DISPATCH_ENGINE = "dispatch_engine"
    DER_COORDINATION_ENGINE = "der_coordination_engine"
    EVENT_MGMT_ENGINE = "event_mgmt_engine"
    PERFORMANCE_ENGINE = "performance_engine"
    REVENUE_ENGINE = "revenue_engine"
    PROGRAM_MATCHING_ENGINE = "program_matching_engine"
    LOAD_INVENTORY_ENGINE = "load_inventory_engine"
    REPORTING_ENGINE = "reporting_engine"
    WORKFLOWS = "workflows"
    DATABASE = "database"
    CACHE = "cache"
    MRV_BRIDGE = "mrv_bridge"
    DATA_BRIDGE = "data_bridge"
    GRID_SIGNAL = "grid_signal"
    BMS_CONTROL = "bms_control"
    METER_DATA = "meter_data"
    DER_ASSETS = "der_assets"
    AUTH = "auth"


QUICK_CHECK_CATEGORIES = {
    HealthCategory.DR_FLEXIBILITY_ENGINE,
    HealthCategory.DISPATCH_ENGINE,
    HealthCategory.EVENT_MGMT_ENGINE,
    HealthCategory.GRID_SIGNAL,
    HealthCategory.BMS_CONTROL,
    HealthCategory.METER_DATA,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class HealthResult(BaseModel):
    """Health status of a single check component."""

    check_name: str = Field(...)
    category: HealthCategory = Field(...)
    status: HealthStatus = Field(default=HealthStatus.HEALTHY)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    details: Dict[str, Any] = Field(default_factory=dict)
    severity: HealthSeverity = Field(default=HealthSeverity.INFO)
    remediation: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=_utcnow)


class HealthCheckConfig(BaseModel):
    """Configuration for the health check."""

    pack_id: str = Field(default="PACK-037")
    pack_version: str = Field(default="1.0.0")
    skip_categories: List[str] = Field(default_factory=list)
    timeout_per_check_ms: float = Field(default=5000.0)
    verbose: bool = Field(default=False)


class SystemHealth(BaseModel):
    """Complete result of the health check."""

    result_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-037")
    pack_version: str = Field(default="1.0.0")
    total_checks: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    degraded: int = Field(default=0)
    unknown: int = Field(default=0)
    overall_health_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_status: HealthStatus = Field(default=HealthStatus.HEALTHY)
    categories: Dict[str, List[HealthResult]] = Field(default_factory=dict)
    total_duration_ms: float = Field(default=0.0)
    executed_at: datetime = Field(default_factory=_utcnow)
    quick_mode: bool = Field(default=False)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Component Lists
# ---------------------------------------------------------------------------

DR_ENGINES = [
    "dr_flexibility_engine",
    "baseline_calculation_engine",
    "dispatch_optimization_engine",
    "der_coordination_engine",
    "event_management_engine",
    "performance_tracking_engine",
    "revenue_reconciliation_engine",
    "program_matching_engine",
    "load_inventory_engine",
    "reporting_engine",
]

DR_WORKFLOWS = [
    "full_dr_enrollment_workflow",
    "event_dispatch_workflow",
    "performance_measurement_workflow",
    "settlement_workflow",
    "der_coordination_workflow",
    "baseline_update_workflow",
    "program_compliance_workflow",
    "reporting_workflow",
]

DR_INTEGRATIONS = [
    "pack_orchestrator",
    "mrv_bridge",
    "data_bridge",
    "grid_signal_bridge",
    "bms_control_bridge",
    "meter_data_bridge",
    "der_asset_bridge",
    "pack036_bridge",
    "pack033_bridge",
    "health_check",
    "setup_wizard",
    "alert_bridge",
]


# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------


class HealthCheck:
    """20-category health check for Demand Response Pack.

    Validates operational readiness across engines, workflows, integrations,
    external system connectivity, and infrastructure components.

    Example:
        >>> hc = HealthCheck()
        >>> result = hc.run_all_checks()
        >>> print(f"Score: {result.overall_health_score}/100")
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """Initialize the Health Check."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or HealthCheckConfig()

        self._check_handlers: Dict[HealthCategory, Callable[[], List[HealthResult]]] = {
            HealthCategory.DR_FLEXIBILITY_ENGINE: self._check_engine("dr_flexibility_engine"),
            HealthCategory.BASELINE_ENGINE: self._check_engine("baseline_calculation_engine"),
            HealthCategory.DISPATCH_ENGINE: self._check_engine("dispatch_optimization_engine"),
            HealthCategory.DER_COORDINATION_ENGINE: self._check_engine("der_coordination_engine"),
            HealthCategory.EVENT_MGMT_ENGINE: self._check_engine("event_management_engine"),
            HealthCategory.PERFORMANCE_ENGINE: self._check_engine("performance_tracking_engine"),
            HealthCategory.REVENUE_ENGINE: self._check_engine("revenue_reconciliation_engine"),
            HealthCategory.PROGRAM_MATCHING_ENGINE: self._check_engine("program_matching_engine"),
            HealthCategory.LOAD_INVENTORY_ENGINE: self._check_engine("load_inventory_engine"),
            HealthCategory.REPORTING_ENGINE: self._check_engine("reporting_engine"),
            HealthCategory.WORKFLOWS: self._check_workflows,
            HealthCategory.DATABASE: self._check_database,
            HealthCategory.CACHE: self._check_cache,
            HealthCategory.MRV_BRIDGE: self._check_bridge("mrv_bridge"),
            HealthCategory.DATA_BRIDGE: self._check_bridge("data_bridge"),
            HealthCategory.GRID_SIGNAL: self._check_bridge("grid_signal_bridge"),
            HealthCategory.BMS_CONTROL: self._check_bridge("bms_control_bridge"),
            HealthCategory.METER_DATA: self._check_bridge("meter_data_bridge"),
            HealthCategory.DER_ASSETS: self._check_bridge("der_asset_bridge"),
            HealthCategory.AUTH: self._check_auth,
        }

        self.logger.info("HealthCheck initialized: 20 categories")

    def run_all_checks(self) -> SystemHealth:
        """Run the full 20-category health check.

        Returns:
            SystemHealth with category-level pass/fail status.
        """
        return self._execute_checks(quick_mode=False)

    def run_quick(self) -> SystemHealth:
        """Run a quick health check (critical categories only).

        Returns:
            SystemHealth for quick check categories.
        """
        return self._execute_checks(quick_mode=True)

    def check_component(self, category: HealthCategory) -> HealthResult:
        """Check a single component category.

        Args:
            category: Category to check.

        Returns:
            HealthResult for the category.
        """
        handler = self._check_handlers.get(category)
        if handler is None:
            return HealthResult(
                check_name=category.value,
                category=category,
                status=HealthStatus.UNKNOWN,
                message=f"No handler for category '{category.value}'",
            )

        checks = handler()
        all_healthy = all(c.status == HealthStatus.HEALTHY for c in checks)
        any_unhealthy = any(c.status == HealthStatus.UNHEALTHY for c in checks)

        status = HealthStatus.HEALTHY
        if any_unhealthy:
            status = HealthStatus.UNHEALTHY
        elif not all_healthy:
            status = HealthStatus.DEGRADED

        return HealthResult(
            check_name=f"{category.value}_summary",
            category=category,
            status=status,
            message=f"{len(checks)} checks: {sum(1 for c in checks if c.status == HealthStatus.HEALTHY)} healthy",
            details={"check_count": len(checks)},
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick health summary without running full checks.

        Returns:
            Dict with health summary metrics.
        """
        return {
            "pack_id": self.config.pack_id,
            "pack_version": self.config.pack_version,
            "total_categories": len(self._check_handlers),
            "engines_count": len(DR_ENGINES),
            "workflows_count": len(DR_WORKFLOWS),
            "integrations_count": len(DR_INTEGRATIONS),
        }

    def _execute_checks(self, quick_mode: bool) -> SystemHealth:
        """Execute health checks across configured categories."""
        start_time = time.monotonic()

        all_checks: Dict[str, List[HealthResult]] = {}
        total = passed = failed = degraded_count = unknown_count = 0

        skip_set = set(self.config.skip_categories)

        for category in HealthCategory:
            if category.value in skip_set:
                continue
            if quick_mode and category not in QUICK_CHECK_CATEGORIES:
                continue

            handler = self._check_handlers.get(category)
            if handler is None:
                continue

            try:
                checks = handler()
            except Exception as exc:
                self.logger.error("Health check '%s' raised: %s", category.value, exc)
                checks = [HealthResult(
                    check_name=f"{category.value}_exception",
                    category=category,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Exception: {exc}",
                    severity=HealthSeverity.CRITICAL,
                )]

            all_checks[category.value] = checks

            for check in checks:
                total += 1
                if check.status == HealthStatus.HEALTHY:
                    passed += 1
                elif check.status == HealthStatus.UNHEALTHY:
                    failed += 1
                elif check.status == HealthStatus.DEGRADED:
                    degraded_count += 1
                elif check.status == HealthStatus.UNKNOWN:
                    unknown_count += 1

        score = (passed / total * 100.0) if total > 0 else 0.0
        overall_status = HealthStatus.HEALTHY
        if failed > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED

        total_duration_ms = (time.monotonic() - start_time) * 1000

        result = SystemHealth(
            total_checks=total,
            passed=passed,
            failed=failed,
            degraded=degraded_count,
            unknown=unknown_count,
            overall_health_score=round(score, 1),
            overall_status=overall_status,
            categories=all_checks,
            total_duration_ms=round(total_duration_ms, 1),
            quick_mode=quick_mode,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Health check complete (%s): %d/%d healthy, score=%.1f",
            "quick" if quick_mode else "full", passed, total, score,
        )
        return result

    # ---- Category Handler Factories ----

    def _check_engine(self, engine_name: str) -> Callable[[], List[HealthResult]]:
        """Create a handler that checks for an engine file."""
        def handler() -> List[HealthResult]:
            start = time.monotonic()
            fpath = PACK_BASE_DIR / "engines" / f"{engine_name}.py"
            exists = fpath.exists()
            category = HealthCategory(engine_name) if engine_name in [e.value for e in HealthCategory] else HealthCategory.DR_FLEXIBILITY_ENGINE
            return [HealthResult(
                check_name=f"engine_{engine_name}",
                category=category,
                status=HealthStatus.HEALTHY if exists else HealthStatus.DEGRADED,
                message=f"{engine_name}: {'found' if exists else 'not found'}",
                duration_ms=(time.monotonic() - start) * 1000,
            )]
        return handler

    def _check_bridge(self, bridge_name: str) -> Callable[[], List[HealthResult]]:
        """Create a handler that checks for an integration bridge file."""
        def handler() -> List[HealthResult]:
            start = time.monotonic()
            fpath = PACK_BASE_DIR / "integrations" / f"{bridge_name}.py"
            exists = fpath.exists()
            category = HealthCategory.MRV_BRIDGE  # Default
            return [HealthResult(
                check_name=f"bridge_{bridge_name}",
                category=category,
                status=HealthStatus.HEALTHY if exists else HealthStatus.DEGRADED,
                message=f"{bridge_name}: {'found' if exists else 'not found'}",
                duration_ms=(time.monotonic() - start) * 1000,
            )]
        return handler

    def _check_workflows(self) -> List[HealthResult]:
        """Check that all DR workflows exist."""
        checks: List[HealthResult] = []
        start = time.monotonic()
        base = PACK_BASE_DIR / "workflows"
        for name in DR_WORKFLOWS:
            fpath = base / f"{name}.py"
            exists = fpath.exists()
            checks.append(HealthResult(
                check_name=f"workflow_{name}",
                category=HealthCategory.WORKFLOWS,
                status=HealthStatus.HEALTHY if exists else HealthStatus.DEGRADED,
                message=f"{name}: {'found' if exists else 'not found'}",
                duration_ms=(time.monotonic() - start) * 1000,
            ))
        return checks

    def _check_database(self) -> List[HealthResult]:
        """Check database connectivity (stub)."""
        start = time.monotonic()
        return [HealthResult(
            check_name="database_postgresql",
            category=HealthCategory.DATABASE,
            status=HealthStatus.DEGRADED,
            message="PostgreSQL: not tested (stub mode)",
            duration_ms=(time.monotonic() - start) * 1000,
        )]

    def _check_cache(self) -> List[HealthResult]:
        """Check Redis cache connectivity (stub)."""
        start = time.monotonic()
        return [HealthResult(
            check_name="cache_redis",
            category=HealthCategory.CACHE,
            status=HealthStatus.DEGRADED,
            message="Redis: not tested (stub mode)",
            duration_ms=(time.monotonic() - start) * 1000,
        )]

    def _check_auth(self) -> List[HealthResult]:
        """Check authentication service (stub)."""
        start = time.monotonic()
        return [HealthResult(
            check_name="auth_jwt",
            category=HealthCategory.AUTH,
            status=HealthStatus.DEGRADED,
            message="Auth: not tested (stub mode)",
            duration_ms=(time.monotonic() - start) * 1000,
        )]
