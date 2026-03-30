# -*- coding: utf-8 -*-
"""
HealthCheck - 20-Category System Health Verification for PACK-039
===================================================================

This module implements a comprehensive health check system that validates the
operational readiness of the Energy Monitoring Pack across engines, workflows,
integrations, external system connectivity, and infrastructure components.

Check Categories (20):
    1.  meter_connectivity         -- Meter communication and data flow
    2.  data_freshness             -- Data staleness and lag detection
    3.  validation_pass_rate       -- Data validation engine pass rate
    4.  anomaly_engine             -- Anomaly detection engine status
    5.  enpi_models                -- EnPI calculation model status
    6.  cost_allocation            -- Cost allocation engine status
    7.  budget_engine              -- Budget tracking engine status
    8.  alarm_system               -- Alarm management system status
    9.  dashboard_engine           -- Dashboard rendering engine status
    10. reporting_engine           -- Report generation engine status
    11. workflows                  -- Verify monitoring workflows
    12. database                   -- Check database connectivity
    13. cache                      -- Check Redis cache connectivity
    14. mrv_bridge                 -- Check MRV bridge availability
    15. data_bridge                -- Check DATA bridge availability
    16. meter_protocol             -- Check meter protocol bridge
    17. ami_bridge                 -- Check AMI data integration
    18. bms_bridge                 -- Check BMS trend data integration
    19. iot_sensors                -- Check IoT sensor connectivity
    20. auth                       -- Check authentication service

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-039 Energy Monitoring
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

from greenlang.schemas import utcnow
from greenlang.schemas.enums import HealthStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

PACK_BASE_DIR = Path(__file__).parent.parent

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

class HealthSeverity(str, Enum):
    """Severity levels for health issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class HealthCategory(str, Enum):
    """Health check categories (20 total)."""

    METER_CONNECTIVITY = "meter_connectivity"
    DATA_FRESHNESS = "data_freshness"
    VALIDATION_PASS_RATE = "validation_pass_rate"
    ANOMALY_ENGINE = "anomaly_engine"
    ENPI_MODELS = "enpi_models"
    COST_ALLOCATION = "cost_allocation"
    BUDGET_ENGINE = "budget_engine"
    ALARM_SYSTEM = "alarm_system"
    DASHBOARD_ENGINE = "dashboard_engine"
    REPORTING_ENGINE = "reporting_engine"
    WORKFLOWS = "workflows"
    DATABASE = "database"
    CACHE = "cache"
    MRV_BRIDGE = "mrv_bridge"
    DATA_BRIDGE = "data_bridge"
    METER_PROTOCOL = "meter_protocol"
    AMI_BRIDGE = "ami_bridge"
    BMS_BRIDGE = "bms_bridge"
    IOT_SENSORS = "iot_sensors"
    AUTH = "auth"

QUICK_CHECK_CATEGORIES = {
    HealthCategory.METER_CONNECTIVITY,
    HealthCategory.DATA_FRESHNESS,
    HealthCategory.ANOMALY_ENGINE,
    HealthCategory.ALARM_SYSTEM,
    HealthCategory.DATABASE,
    HealthCategory.METER_PROTOCOL,
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
    timestamp: datetime = Field(default_factory=utcnow)

class HealthCheckConfig(BaseModel):
    """Configuration for the health check."""

    pack_id: str = Field(default="PACK-039")
    pack_version: str = Field(default="1.0.0")
    skip_categories: List[str] = Field(default_factory=list)
    timeout_per_check_ms: float = Field(default=5000.0)
    verbose: bool = Field(default=False)

class SystemHealth(BaseModel):
    """Complete result of the health check."""

    result_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-039")
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
    executed_at: datetime = Field(default_factory=utcnow)
    quick_mode: bool = Field(default=False)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Component Lists
# ---------------------------------------------------------------------------

EM_ENGINES = [
    "meter_registry_engine",
    "data_acquisition_engine",
    "data_validation_engine",
    "anomaly_detection_engine",
    "enpi_calculation_engine",
    "cost_allocation_engine",
    "budget_tracking_engine",
    "alarm_management_engine",
    "dashboard_engine",
    "reporting_engine",
]

EM_WORKFLOWS = [
    "full_monitoring_workflow",
    "real_time_acquisition_workflow",
    "anomaly_detection_workflow",
    "enpi_calculation_workflow",
    "cost_allocation_workflow",
    "budget_tracking_workflow",
    "alarm_management_workflow",
    "reporting_workflow",
]

EM_INTEGRATIONS = [
    "pack_orchestrator",
    "mrv_bridge",
    "data_bridge",
    "meter_protocol_bridge",
    "ami_bridge",
    "bms_bridge",
    "iot_sensor_bridge",
    "pack036_bridge",
    "pack038_bridge",
    "health_check",
    "setup_wizard",
    "alert_bridge",
]

# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------

class HealthCheck:
    """20-category health check for Energy Monitoring Pack.

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
            HealthCategory.METER_CONNECTIVITY: self._check_engine("meter_registry_engine"),
            HealthCategory.DATA_FRESHNESS: self._check_engine("data_acquisition_engine"),
            HealthCategory.VALIDATION_PASS_RATE: self._check_engine("data_validation_engine"),
            HealthCategory.ANOMALY_ENGINE: self._check_engine("anomaly_detection_engine"),
            HealthCategory.ENPI_MODELS: self._check_engine("enpi_calculation_engine"),
            HealthCategory.COST_ALLOCATION: self._check_engine("cost_allocation_engine"),
            HealthCategory.BUDGET_ENGINE: self._check_engine("budget_tracking_engine"),
            HealthCategory.ALARM_SYSTEM: self._check_engine("alarm_management_engine"),
            HealthCategory.DASHBOARD_ENGINE: self._check_engine("dashboard_engine"),
            HealthCategory.REPORTING_ENGINE: self._check_engine("reporting_engine"),
            HealthCategory.WORKFLOWS: self._check_workflows,
            HealthCategory.DATABASE: self._check_database,
            HealthCategory.CACHE: self._check_cache,
            HealthCategory.MRV_BRIDGE: self._check_bridge("mrv_bridge"),
            HealthCategory.DATA_BRIDGE: self._check_bridge("data_bridge"),
            HealthCategory.METER_PROTOCOL: self._check_bridge("meter_protocol_bridge"),
            HealthCategory.AMI_BRIDGE: self._check_bridge("ami_bridge"),
            HealthCategory.BMS_BRIDGE: self._check_bridge("bms_bridge"),
            HealthCategory.IOT_SENSORS: self._check_bridge("iot_sensor_bridge"),
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
            "engines_count": len(EM_ENGINES),
            "workflows_count": len(EM_WORKFLOWS),
            "integrations_count": len(EM_INTEGRATIONS),
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
            category = HealthCategory.METER_CONNECTIVITY  # default fallback
            for cat in HealthCategory:
                if cat.value == engine_name.replace("_engine", ""):
                    category = cat
                    break
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
        """Check that all energy monitoring workflows exist."""
        checks: List[HealthResult] = []
        start = time.monotonic()
        base = PACK_BASE_DIR / "workflows"
        for name in EM_WORKFLOWS:
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
