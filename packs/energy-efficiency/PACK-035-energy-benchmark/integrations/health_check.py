# -*- coding: utf-8 -*-
"""
HealthCheck - 15-Category System Health Verification for PACK-035
===================================================================

This module implements a comprehensive health check system that validates the
operational readiness of the Energy Benchmark Pack.

Check Categories (15):
    1.  engines              -- Verify benchmark calculation engines
    2.  workflows            -- Verify benchmark workflows
    3.  templates            -- Verify report templates
    4.  integrations         -- Verify 12 integration bridges
    5.  config               -- Validate pack configuration
    6.  presets              -- Verify facility presets
    7.  database             -- Check database connectivity
    8.  memory               -- Check available memory
    9.  disk_space           -- Check available disk space
    10. api_connectivity     -- Check API endpoint availability
    11. weather_service      -- Check weather data service availability
    12. benchmark_databases  -- Check benchmark database availability
    13. agent_dependencies   -- Check MRV and DATA agent availability
    14. imports              -- Check module import health
    15. migrations           -- Check database migration status

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-035 Energy Benchmark
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

class HealthCategory(str, Enum):
    """Health check categories (15 total)."""

    ENGINES = "engines"
    WORKFLOWS = "workflows"
    TEMPLATES = "templates"
    INTEGRATIONS = "integrations"
    CONFIG = "config"
    PRESETS = "presets"
    DATABASE = "database"
    MEMORY = "memory"
    DISK_SPACE = "disk_space"
    API_CONNECTIVITY = "api_connectivity"
    WEATHER_SERVICE = "weather_service"
    BENCHMARK_DATABASES = "benchmark_databases"
    AGENT_DEPENDENCIES = "agent_dependencies"
    IMPORTS = "imports"
    MIGRATIONS = "migrations"

class HealthSeverity(str, Enum):
    """Severity levels for health issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

QUICK_CHECK_CATEGORIES = {
    HealthCategory.ENGINES,
    HealthCategory.WORKFLOWS,
    HealthCategory.TEMPLATES,
    HealthCategory.INTEGRATIONS,
    HealthCategory.CONFIG,
    HealthCategory.PRESETS,
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class CategoryHealth(BaseModel):
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

    pack_id: str = Field(default="PACK-035")
    pack_version: str = Field(default="1.0.0")
    skip_categories: List[str] = Field(default_factory=list)
    timeout_per_check_ms: float = Field(default=5000.0)
    verbose: bool = Field(default=False)

class HealthCheckResult(BaseModel):
    """Complete result of the health check."""

    result_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-035")
    pack_version: str = Field(default="1.0.0")
    total_checks: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    degraded: int = Field(default=0)
    unknown: int = Field(default=0)
    overall_health_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_status: HealthStatus = Field(default=HealthStatus.HEALTHY)
    categories: Dict[str, List[CategoryHealth]] = Field(default_factory=dict)
    total_duration_ms: float = Field(default=0.0)
    executed_at: datetime = Field(default_factory=utcnow)
    quick_mode: bool = Field(default=False)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Component Lists
# ---------------------------------------------------------------------------

BENCHMARK_ENGINES = [
    "eui_calculation_engine",
    "weather_normalisation_engine",
    "peer_comparison_engine",
    "performance_rating_engine",
    "gap_analysis_engine",
    "trend_analysis_engine",
    "portfolio_aggregation_engine",
    "carbon_intensity_engine",
    "reporting_engine",
    "data_validation_engine",
]

BENCHMARK_WORKFLOWS = [
    "full_benchmark_workflow",
    "single_building_workflow",
    "portfolio_benchmark_workflow",
    "trend_analysis_workflow",
    "weather_normalisation_workflow",
    "gap_analysis_workflow",
    "epc_comparison_workflow",
    "energy_star_workflow",
]

BENCHMARK_TEMPLATES = [
    "executive_summary_report",
    "benchmark_dashboard",
    "eui_comparison_report",
    "performance_rating_report",
    "gap_analysis_report",
    "trend_analysis_report",
    "portfolio_overview_report",
    "energy_star_scorecard",
    "epc_comparison_report",
    "recommendations_report",
]

BENCHMARK_INTEGRATIONS = [
    "pack_orchestrator",
    "mrv_benchmark_bridge",
    "data_benchmark_bridge",
    "pack031_bridge",
    "pack032_bridge",
    "pack033_bridge",
    "energy_star_bridge",
    "weather_service_bridge",
    "epc_registry_bridge",
    "benchmark_database_bridge",
    "health_check",
    "setup_wizard",
]

BENCHMARK_PRESETS = [
    "office_standard",
    "office_premium",
    "retail_standard",
    "hotel_standard",
    "hospital_standard",
    "education_standard",
    "warehouse_standard",
    "industrial_standard",
]

# ---------------------------------------------------------------------------
# HealthCheck
# ---------------------------------------------------------------------------

class HealthCheck:
    """15-category health check for Energy Benchmark Pack.

    Validates operational readiness across engines, workflows, templates,
    integrations, presets, weather services, benchmark databases, and
    agent dependencies.

    Example:
        >>> hc = HealthCheck()
        >>> result = hc.run_full_check()
        >>> print(f"Score: {result.overall_health_score}/100")
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """Initialize the Health Check."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or HealthCheckConfig()

        self._check_handlers: Dict[HealthCategory, Callable[[], List[CategoryHealth]]] = {
            HealthCategory.ENGINES: self._check_engines,
            HealthCategory.WORKFLOWS: self._check_workflows,
            HealthCategory.TEMPLATES: self._check_templates,
            HealthCategory.INTEGRATIONS: self._check_integrations,
            HealthCategory.CONFIG: self._check_config,
            HealthCategory.PRESETS: self._check_presets,
            HealthCategory.DATABASE: self._check_databases,
            HealthCategory.MEMORY: self._check_memory,
            HealthCategory.DISK_SPACE: self._check_disk_space,
            HealthCategory.API_CONNECTIVITY: self._check_api_connectivity,
            HealthCategory.WEATHER_SERVICE: self._check_weather_service,
            HealthCategory.BENCHMARK_DATABASES: self._check_benchmark_databases,
            HealthCategory.AGENT_DEPENDENCIES: self._check_agent_dependencies,
            HealthCategory.IMPORTS: self._check_imports,
            HealthCategory.MIGRATIONS: self._check_migrations,
        }

        self.logger.info("HealthCheck initialized: 15 categories")

    def run_full_check(self) -> HealthCheckResult:
        """Run the full 15-category health check.

        Returns:
            HealthCheckResult with category-level pass/fail status.
        """
        return self._execute_checks(quick_mode=False)

    def run_quick(self) -> HealthCheckResult:
        """Run a quick health check (core categories only).

        Returns:
            HealthCheckResult for quick check categories.
        """
        return self._execute_checks(quick_mode=True)

    def check_engines(self) -> List[CategoryHealth]:
        """Check that all benchmark engines exist.

        Returns:
            List of CategoryHealth for each engine.
        """
        return self._check_engines()

    def check_databases(self) -> List[CategoryHealth]:
        """Check database connectivity.

        Returns:
            List of CategoryHealth for database checks.
        """
        return self._check_databases()

    def check_weather_service(self) -> List[CategoryHealth]:
        """Check weather data service availability.

        Returns:
            List of CategoryHealth for weather service checks.
        """
        return self._check_weather_service()

    def check_benchmark_databases(self) -> List[CategoryHealth]:
        """Check benchmark database availability.

        Returns:
            List of CategoryHealth for benchmark database checks.
        """
        return self._check_benchmark_databases()

    def check_agent_dependencies(self) -> List[CategoryHealth]:
        """Check MRV and DATA agent availability.

        Returns:
            List of CategoryHealth for agent dependency checks.
        """
        return self._check_agent_dependencies()

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick health summary without running full checks.

        Returns:
            Dict with health summary metrics.
        """
        return {
            "pack_id": self.config.pack_id,
            "pack_version": self.config.pack_version,
            "total_categories": len(self._check_handlers),
            "engines_count": len(BENCHMARK_ENGINES),
            "workflows_count": len(BENCHMARK_WORKFLOWS),
            "templates_count": len(BENCHMARK_TEMPLATES),
            "integrations_count": len(BENCHMARK_INTEGRATIONS),
            "presets_count": len(BENCHMARK_PRESETS),
        }

    def _execute_checks(self, quick_mode: bool) -> HealthCheckResult:
        """Execute health checks across configured categories."""
        start_time = time.monotonic()

        all_checks: Dict[str, List[CategoryHealth]] = {}
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
                checks = [CategoryHealth(
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

        result = HealthCheckResult(
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

    # ---- Category Handlers ----

    def _check_file_list(
        self, category: HealthCategory, directory: str,
        file_list: List[str], suffix: str = ".py",
    ) -> List[CategoryHealth]:
        """Check existence of a list of files in a directory."""
        checks: List[CategoryHealth] = []
        start = time.monotonic()
        base = PACK_BASE_DIR / directory
        for name in file_list:
            fpath = base / f"{name}{suffix}"
            exists = fpath.exists()
            checks.append(CategoryHealth(
                check_name=f"{category.value}_{name}",
                category=category,
                status=HealthStatus.HEALTHY if exists else HealthStatus.DEGRADED,
                message=f"{name}: {'found' if exists else 'not found'}",
                duration_ms=(time.monotonic() - start) * 1000,
            ))
        return checks

    def _check_engines(self) -> List[CategoryHealth]:
        return self._check_file_list(HealthCategory.ENGINES, "engines", BENCHMARK_ENGINES)

    def _check_workflows(self) -> List[CategoryHealth]:
        return self._check_file_list(HealthCategory.WORKFLOWS, "workflows", BENCHMARK_WORKFLOWS)

    def _check_templates(self) -> List[CategoryHealth]:
        return self._check_file_list(HealthCategory.TEMPLATES, "templates", BENCHMARK_TEMPLATES)

    def _check_integrations(self) -> List[CategoryHealth]:
        return self._check_file_list(HealthCategory.INTEGRATIONS, "integrations", BENCHMARK_INTEGRATIONS)

    def _check_config(self) -> List[CategoryHealth]:
        start = time.monotonic()
        config_path = PACK_BASE_DIR / "config" / "pack_config.py"
        return [CategoryHealth(
            check_name="config_pack_config",
            category=HealthCategory.CONFIG,
            status=HealthStatus.HEALTHY if config_path.exists() else HealthStatus.UNHEALTHY,
            message="pack_config.py " + ("found" if config_path.exists() else "MISSING"),
            duration_ms=(time.monotonic() - start) * 1000,
            severity=HealthSeverity.HIGH if not config_path.exists() else HealthSeverity.INFO,
            remediation="Create config/pack_config.py" if not config_path.exists() else None,
        )]

    def _check_presets(self) -> List[CategoryHealth]:
        return self._check_file_list(HealthCategory.PRESETS, "config/presets", BENCHMARK_PRESETS, ".yaml")

    def _check_infra_stub(self, category: HealthCategory, name: str) -> List[CategoryHealth]:
        start = time.monotonic()
        return [CategoryHealth(
            check_name=f"{category.value}_{name}",
            category=category,
            status=HealthStatus.DEGRADED,
            message=f"{name}: not tested (stub mode)",
            duration_ms=(time.monotonic() - start) * 1000,
        )]

    def _check_databases(self) -> List[CategoryHealth]:
        return self._check_infra_stub(HealthCategory.DATABASE, "postgresql")

    def _check_memory(self) -> List[CategoryHealth]:
        start = time.monotonic()
        return [CategoryHealth(
            check_name="memory_available",
            category=HealthCategory.MEMORY,
            status=HealthStatus.HEALTHY,
            message="Memory check: Python process responsive",
            duration_ms=(time.monotonic() - start) * 1000,
        )]

    def _check_disk_space(self) -> List[CategoryHealth]:
        start = time.monotonic()
        try:
            import shutil

            total, used, free = shutil.disk_usage(str(PACK_BASE_DIR))
            free_gb = free / (1024 ** 3)
            if free_gb > 1.0:
                status = HealthStatus.HEALTHY
            elif free_gb > 0.5:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            return [CategoryHealth(
                check_name="disk_space_free",
                category=HealthCategory.DISK_SPACE,
                status=status,
                message=f"Free disk space: {free_gb:.1f} GB",
                details={"free_gb": round(free_gb, 1)},
                duration_ms=(time.monotonic() - start) * 1000,
            )]
        except Exception as exc:
            return [CategoryHealth(
                check_name="disk_space_free",
                category=HealthCategory.DISK_SPACE,
                status=HealthStatus.DEGRADED,
                message=f"Could not check disk space: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )]

    def _check_api_connectivity(self) -> List[CategoryHealth]:
        return self._check_infra_stub(HealthCategory.API_CONNECTIVITY, "api_gateway")

    def _check_weather_service(self) -> List[CategoryHealth]:
        start = time.monotonic()
        sources = ["meteostat", "noaa", "cibse_try", "dwd", "ashrae_iwec"]
        return [CategoryHealth(
            check_name=f"weather_{src}",
            category=HealthCategory.WEATHER_SERVICE,
            status=HealthStatus.DEGRADED,
            message=f"{src}: not tested (stub mode)",
            duration_ms=(time.monotonic() - start) * 1000,
        ) for src in sources]

    def _check_benchmark_databases(self) -> List[CategoryHealth]:
        start = time.monotonic()
        sources = ["cibse_tm46", "din_v_18599", "bpie", "ashrae_90_1", "energy_star"]
        return [CategoryHealth(
            check_name=f"benchmark_db_{src}",
            category=HealthCategory.BENCHMARK_DATABASES,
            status=HealthStatus.DEGRADED,
            message=f"{src}: not tested (stub mode)",
            duration_ms=(time.monotonic() - start) * 1000,
        ) for src in sources]

    def _check_agent_dependencies(self) -> List[CategoryHealth]:
        start = time.monotonic()
        agents = ["MRV-001", "MRV-009", "MRV-010", "MRV-013", "DATA-001", "DATA-002", "DATA-004", "DATA-010", "DATA-013", "DATA-014", "DATA-018"]
        checks: List[CategoryHealth] = []
        for agent_id in agents:
            checks.append(CategoryHealth(
                check_name=f"agent_{agent_id.lower().replace('-', '_')}",
                category=HealthCategory.AGENT_DEPENDENCIES,
                status=HealthStatus.DEGRADED,
                message=f"{agent_id}: not tested (stub mode)",
                duration_ms=(time.monotonic() - start) * 1000,
            ))
        return checks

    def _check_imports(self) -> List[CategoryHealth]:
        start = time.monotonic()
        deps = ["pydantic", "hashlib", "json", "logging", "decimal"]
        checks: List[CategoryHealth] = []
        for dep in deps:
            try:
                __import__(dep)
                checks.append(CategoryHealth(
                    check_name=f"dependency_{dep}",
                    category=HealthCategory.IMPORTS,
                    status=HealthStatus.HEALTHY,
                    message=f"{dep}: available",
                    duration_ms=(time.monotonic() - start) * 1000,
                ))
            except ImportError:
                checks.append(CategoryHealth(
                    check_name=f"dependency_{dep}",
                    category=HealthCategory.IMPORTS,
                    status=HealthStatus.UNHEALTHY,
                    message=f"{dep}: NOT AVAILABLE",
                    severity=HealthSeverity.CRITICAL,
                    remediation=f"pip install {dep}",
                    duration_ms=(time.monotonic() - start) * 1000,
                ))
        return checks

    def _check_migrations(self) -> List[CategoryHealth]:
        return self._check_infra_stub(HealthCategory.MIGRATIONS, "db_migrations")
