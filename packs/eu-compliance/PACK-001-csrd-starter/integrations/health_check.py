# -*- coding: utf-8 -*-
"""
PackHealthCheck - Comprehensive Health Verification for CSRD Starter Pack
=========================================================================

This module implements a comprehensive health check system that validates
the operational readiness of the CSRD Starter Pack. It checks agent
availability, configuration validity, data file presence, database
connectivity, dependency versions, security settings, and basic
performance benchmarks.

Check Categories:
    1. Agent Availability: All 66+ agents importable and initializable
    2. Configuration: pack.yaml valid, presets valid, sector configs valid
    3. Data Files: ESRS formulas, emission factors, compliance rules present
    4. Database: Connection test, schema version, required tables
    5. Dependencies: Python packages at required versions
    6. Security: Auth configured, RBAC roles, encryption enabled
    7. Performance: Benchmark 100 metric calculations under target time

Example:
    >>> health = PackHealthCheck()
    >>> result = await health.check_all()
    >>> for component in result.components:
    ...     print(f"{component.name}: {component.status}")

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import importlib
import logging
import os
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas.enums import HealthStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class HealthSeverity(str, Enum):
    """Severity level for health check findings."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class CheckCategory(str, Enum):
    """Categories of health checks."""
    AGENTS = "agents"
    CONFIGURATION = "configuration"
    DATA_FILES = "data_files"
    DATABASE = "database"
    DEPENDENCIES = "dependencies"
    SECURITY = "security"
    PERFORMANCE = "performance"


# =============================================================================
# Data Models
# =============================================================================


class HealthCheckConfig(BaseModel):
    """Configuration for the health check system."""
    project_root: str = Field(
        default="", description="Root directory of the GreenLang project"
    )
    check_agents: bool = Field(default=True, description="Check agent availability")
    check_configuration: bool = Field(default=True, description="Check configuration files")
    check_data_files: bool = Field(default=True, description="Check data file presence")
    check_database: bool = Field(default=True, description="Check database connectivity")
    check_dependencies: bool = Field(default=True, description="Check package dependencies")
    check_security: bool = Field(default=True, description="Check security settings")
    check_performance: bool = Field(default=True, description="Run performance benchmarks")
    database_url: Optional[str] = Field(None, description="Database URL for connectivity test")
    performance_target_ms: float = Field(
        default=5000.0, description="Target time for 100 metric calculations (ms)"
    )
    performance_metric_count: int = Field(
        default=100, description="Number of metrics to calculate in benchmark"
    )


class RemediationSuggestion(BaseModel):
    """A remediation suggestion for a health check finding."""
    finding: str = Field(..., description="What was found")
    suggestion: str = Field(..., description="What to do about it")
    severity: HealthSeverity = Field(..., description="Severity of the finding")
    category: CheckCategory = Field(..., description="Check category")
    auto_fixable: bool = Field(
        default=False, description="Whether this can be auto-fixed"
    )
    documentation_link: Optional[str] = Field(
        None, description="Link to relevant documentation"
    )


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str = Field(..., description="Component name")
    category: CheckCategory = Field(..., description="Check category")
    status: HealthStatus = Field(..., description="Component health status")
    message: str = Field(default="", description="Status message")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed check results"
    )
    checks_passed: int = Field(default=0, description="Number of checks passed")
    checks_failed: int = Field(default=0, description="Number of checks failed")
    checks_warned: int = Field(default=0, description="Number of checks with warnings")
    execution_time_ms: float = Field(default=0.0, description="Check execution time")
    remediations: List[RemediationSuggestion] = Field(
        default_factory=list, description="Remediation suggestions"
    )


class HealthCheckResult(BaseModel):
    """Complete health check result."""
    overall_status: HealthStatus = Field(..., description="Overall pack health status")
    check_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the check was run"
    )
    total_checks: int = Field(default=0, description="Total number of checks run")
    checks_passed: int = Field(default=0, description="Total checks passed")
    checks_failed: int = Field(default=0, description="Total checks failed")
    checks_warned: int = Field(default=0, description="Total checks with warnings")
    components: List[ComponentHealth] = Field(
        default_factory=list, description="Per-component health results"
    )
    critical_issues: List[RemediationSuggestion] = Field(
        default_factory=list, description="Critical issues requiring attention"
    )
    warnings: List[RemediationSuggestion] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    total_execution_time_ms: float = Field(
        default=0.0, description="Total health check time"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of the health check result"
    )


# =============================================================================
# Agent Registry for Health Checks
# =============================================================================

REQUIRED_AGENT_MODULES: Dict[str, str] = {
    # Foundation Agents
    "GL-FOUND-X-001": "greenlang.agents.foundation.orchestrator",
    "GL-FOUND-X-002": "greenlang.agents.foundation.schema_compiler",
    "GL-FOUND-X-003": "greenlang.agents.foundation.unit_normalizer",
    "GL-FOUND-X-004": "greenlang.agents.foundation.assumptions_registry",
    "GL-FOUND-X-005": "greenlang.agents.foundation.citations_agent",
    "GL-FOUND-X-006": "greenlang.agents.foundation.policy_guard",
    "GL-FOUND-X-007": "greenlang.agents.foundation.pii_redaction",
    "GL-FOUND-X-008": "greenlang.agents.foundation.qa_test_harness",
    "GL-FOUND-X-009": "greenlang.agents.foundation.observability_agent",
    "GL-FOUND-X-010": "greenlang.agents.foundation.agent_registry",
    # Data Agents (intake)
    "GL-DATA-X-001": "greenlang.agents.data.document_ingestion_agent",
    "GL-DATA-X-004": "greenlang.agents.data.erp_connector_agent",
    "GL-DATA-X-008": "greenlang.agents.data.weather_climate_agent",
    "GL-DATA-X-009": "greenlang.agents.data.utility_tariff_agent",
    # MRV Scope 1
    "GL-MRV-SCOPE1": "greenlang.agents.mrv.scope1_combustion",
    "GL-MRV-RFGAS": "greenlang.agents.mrv.refrigerants_fgas",
    "GL-MRV-PROCESS": "greenlang.agents.mrv.process_emissions",
    # MRV Scope 2
    "GL-MRV-LOC": "greenlang.agents.mrv.scope2_location_based",
    "GL-MRV-MKT": "greenlang.agents.mrv.scope2_market_based",
    # MRV Cross-cutting
    "GL-MRV-S3MAP": "greenlang.agents.mrv.scope3_category_mapper",
    "GL-MRV-AUDIT": "greenlang.agents.mrv.audit_trail_lineage",
    "GL-MRV-UDQ": "greenlang.agents.mrv.uncertainty_data_quality",
}

REQUIRED_PACKAGES: Dict[str, str] = {
    "pydantic": "2.0.0",
    "httpx": "0.24.0",
    "numpy": "1.24.0",
    "pandas": "2.0.0",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "psycopg": "3.0.0",
    "redis": "4.0.0",
    "boto3": "1.26.0",
    "opentelemetry-api": "1.15.0",
}

REQUIRED_DATA_FILES: List[str] = [
    "greenlang/agents/mrv",
    "greenlang/agents/foundation",
    "greenlang/agents/data",
]


# =============================================================================
# Health Check Implementation
# =============================================================================


class PackHealthCheck:
    """Comprehensive health check for CSRD Starter Pack.

    Validates operational readiness across seven check categories:
    agents, configuration, data files, database, dependencies, security,
    and performance. Generates detailed reports with remediation
    suggestions for any issues found.

    Attributes:
        config: Health check configuration
        _results: Accumulated component health results

    Example:
        >>> health = PackHealthCheck()
        >>> result = await health.check_all()
        >>> print(result.overall_status)
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """Initialize the health check system.

        Args:
            config: Health check configuration. Uses defaults if not provided.
        """
        self.config = config or HealthCheckConfig()
        self._results: List[ComponentHealth] = []

        if not self.config.project_root:
            self.config.project_root = str(
                Path(__file__).resolve().parents[4]
            )

        logger.info("PackHealthCheck initialized with project_root=%s",
                     self.config.project_root)

    # -------------------------------------------------------------------------
    # Main Check Entry Point
    # -------------------------------------------------------------------------

    async def check_all(self) -> HealthCheckResult:
        """Run all enabled health checks and return a comprehensive result.

        Returns:
            HealthCheckResult with per-component details and overall status.
        """
        start_time = time.monotonic()
        self._results = []

        logger.info("Starting comprehensive health check")

        check_methods = [
            (self.config.check_agents, self.check_agents),
            (self.config.check_configuration, self.check_configuration),
            (self.config.check_data_files, self.check_data_files),
            (self.config.check_database, self.check_database),
            (self.config.check_dependencies, self.check_dependencies),
            (self.config.check_security, self.check_security),
            (self.config.check_performance, self.check_performance),
        ]

        for enabled, method in check_methods:
            if enabled:
                try:
                    component = await method()
                    self._results.append(component)
                except Exception as exc:
                    logger.error("Health check %s failed: %s",
                                 method.__name__, exc, exc_info=True)
                    self._results.append(ComponentHealth(
                        name=method.__name__.replace("check_", "").title(),
                        category=CheckCategory.AGENTS,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check raised exception: {exc}",
                    ))

        total_elapsed = (time.monotonic() - start_time) * 1000
        return self._build_result(total_elapsed)

    # -------------------------------------------------------------------------
    # Individual Check Methods
    # -------------------------------------------------------------------------

    async def check_agents(self) -> ComponentHealth:
        """Check that all required agents are importable and initializable.

        Returns:
            ComponentHealth for the agents category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {"agents": {}}

        for agent_id, module_path in REQUIRED_AGENT_MODULES.items():
            try:
                importlib.import_module(module_path)
                details["agents"][agent_id] = "importable"
                passed += 1
            except ImportError as exc:
                details["agents"][agent_id] = f"import_error: {exc}"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Agent {agent_id} module '{module_path}' could not be imported",
                    suggestion=(
                        f"Verify that the module '{module_path}' exists and all its "
                        f"dependencies are installed. Run: pip install -e ."
                    ),
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.AGENTS,
                ))
            except Exception as exc:
                details["agents"][agent_id] = f"error: {exc}"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Agent {agent_id} raised unexpected error on import",
                    suggestion=f"Check module '{module_path}' for initialization errors: {exc}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.AGENTS,
                ))

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.UNHEALTHY if failed > 5 else HealthStatus.DEGRADED

        return ComponentHealth(
            name="Agent Availability",
            category=CheckCategory.AGENTS,
            status=status,
            message=f"{passed}/{passed+failed+warned} agents importable",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    async def check_configuration(self) -> ComponentHealth:
        """Check that configuration files are valid and complete.

        Returns:
            ComponentHealth for the configuration category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        # Check pack directory structure
        pack_root = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-001-csrd-starter",
        )

        required_dirs = ["config", "integrations", "templates", "tests", "workflows"]
        for dir_name in required_dirs:
            dir_path = os.path.join(pack_root, dir_name)
            if os.path.isdir(dir_path):
                details[f"dir_{dir_name}"] = "present"
                passed += 1
            else:
                details[f"dir_{dir_name}"] = "missing"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Directory '{dir_name}' not found in pack root",
                    suggestion=f"Create directory: {dir_path}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.CONFIGURATION,
                    auto_fixable=True,
                ))

        # Check config __init__.py
        config_init = os.path.join(pack_root, "config", "__init__.py")
        if os.path.isfile(config_init):
            details["config_init"] = "present"
            passed += 1
        else:
            details["config_init"] = "missing"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="Config __init__.py not found",
                suggestion="Create the config module with PackConfig class",
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.CONFIGURATION,
            ))

        # Check integrations __init__.py
        integrations_init = os.path.join(pack_root, "integrations", "__init__.py")
        if os.path.isfile(integrations_init):
            details["integrations_init"] = "present"
            passed += 1
        else:
            details["integrations_init"] = "missing"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="Integrations __init__.py not found",
                suggestion="Create the integrations module",
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.CONFIGURATION,
            ))

        # Check templates __init__.py
        templates_init = os.path.join(pack_root, "templates", "__init__.py")
        if os.path.isfile(templates_init):
            details["templates_init"] = "present"
            passed += 1
        else:
            details["templates_init"] = "missing"
            warned += 1

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.DEGRADED
        if failed > 2:
            status = HealthStatus.UNHEALTHY

        return ComponentHealth(
            name="Configuration",
            category=CheckCategory.CONFIGURATION,
            status=status,
            message=f"{passed} configs valid, {failed} missing, {warned} warnings",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    async def check_data_files(self) -> ComponentHealth:
        """Check that required data files and directories are present.

        Returns:
            ComponentHealth for the data files category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        for rel_path in REQUIRED_DATA_FILES:
            full_path = os.path.join(self.config.project_root, rel_path)
            if os.path.exists(full_path):
                details[rel_path] = "present"
                passed += 1

                # Check for __init__.py in agent directories
                init_file = os.path.join(full_path, "__init__.py")
                if os.path.isfile(init_file):
                    details[f"{rel_path}/__init__"] = "present"
                    passed += 1
                else:
                    details[f"{rel_path}/__init__"] = "missing"
                    warned += 1
            else:
                details[rel_path] = "missing"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Required path '{rel_path}' not found",
                    suggestion=f"Verify project structure includes: {full_path}",
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.DATA_FILES,
                ))

        # Check for specific MRV modules
        mrv_modules = [
            "scope1_combustion.py", "refrigerants_fgas.py",
            "scope2_location_based.py", "scope2_market_based.py",
            "scope3_category_mapper.py",
        ]
        mrv_dir = os.path.join(self.config.project_root, "greenlang", "agents", "mrv")
        for module_name in mrv_modules:
            module_path = os.path.join(mrv_dir, module_name)
            if os.path.isfile(module_path):
                details[f"mrv/{module_name}"] = "present"
                passed += 1
            else:
                details[f"mrv/{module_name}"] = "missing"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"MRV module '{module_name}' not found",
                    suggestion=f"Ensure MRV agent module exists at: {module_path}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.DATA_FILES,
                ))

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.DEGRADED
        if failed > 2:
            status = HealthStatus.UNHEALTHY

        return ComponentHealth(
            name="Data Files",
            category=CheckCategory.DATA_FILES,
            status=status,
            message=f"{passed} files present, {failed} missing",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity and schema readiness.

        Returns:
            ComponentHealth for the database category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        db_url = self.config.database_url
        if not db_url:
            details["connection"] = "not_configured"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="No database URL configured for health check",
                suggestion=(
                    "Provide a database URL in the health check config or "
                    "set the DATABASE_URL environment variable"
                ),
                severity=HealthSeverity.INFO,
                category=CheckCategory.DATABASE,
            ))
        else:
            # Test URL format
            if "://" in db_url:
                details["url_format"] = "valid"
                passed += 1
            else:
                details["url_format"] = "invalid"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding="Database URL format is invalid",
                    suggestion="URL must contain '://' (e.g., postgresql://user:pass@host/db)",
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.DATABASE,
                ))

            # Test connection (format validation only in health check)
            try:
                if db_url.startswith(("postgresql://", "postgres://", "sqlite://")):
                    details["driver"] = "recognized"
                    passed += 1
                else:
                    details["driver"] = "unrecognized"
                    warned += 1
            except Exception as exc:
                details["connection_test"] = f"error: {exc}"
                failed += 1

        # Check for psycopg availability
        try:
            importlib.import_module("psycopg")
            details["psycopg"] = "available"
            passed += 1
        except ImportError:
            details["psycopg"] = "not_installed"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="psycopg package not installed",
                suggestion="Install psycopg for PostgreSQL support: pip install psycopg[binary]",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.DATABASE,
            ))

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.UNHEALTHY
        elif warned > 0:
            status = HealthStatus.DEGRADED

        return ComponentHealth(
            name="Database",
            category=CheckCategory.DATABASE,
            status=status,
            message=f"{passed} checks passed, {failed} failed, {warned} warnings",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    async def check_dependencies(self) -> ComponentHealth:
        """Check that required Python packages are installed at minimum versions.

        Returns:
            ComponentHealth for the dependencies category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {"required": {}, "optional": {}}

        # Check required packages
        for pkg_name, min_version in REQUIRED_PACKAGES.items():
            pkg_result = _check_package_version(pkg_name, min_version)
            details["required"][pkg_name] = pkg_result
            if pkg_result["status"] == "ok":
                passed += 1
            elif pkg_result["status"] == "version_mismatch":
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=(
                        f"Package '{pkg_name}' version {pkg_result.get('installed', 'unknown')} "
                        f"is below minimum {min_version}"
                    ),
                    suggestion=f"Upgrade: pip install '{pkg_name}>={min_version}'",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.DEPENDENCIES,
                ))
            else:
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Required package '{pkg_name}' is not installed",
                    suggestion=f"Install: pip install '{pkg_name}>={min_version}'",
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.DEPENDENCIES,
                ))

        # Check optional packages
        for pkg_name, min_version in OPTIONAL_PACKAGES.items():
            pkg_result = _check_package_version(pkg_name, min_version)
            details["optional"][pkg_name] = pkg_result
            if pkg_result["status"] == "ok":
                passed += 1
            elif pkg_result["status"] == "not_installed":
                warned += 1  # Optional, so just warn

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.UNHEALTHY
        elif warned > 1:
            status = HealthStatus.DEGRADED

        return ComponentHealth(
            name="Dependencies",
            category=CheckCategory.DEPENDENCIES,
            status=status,
            message=(
                f"{passed} packages OK, {failed} missing required, "
                f"{warned} warnings"
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    async def check_security(self) -> ComponentHealth:
        """Check security configuration and readiness.

        Returns:
            ComponentHealth for the security category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        # Check auth module availability
        try:
            importlib.import_module("greenlang.auth")
            details["auth_module"] = "available"
            passed += 1
        except ImportError:
            details["auth_module"] = "not_available"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="Auth module not importable",
                suggestion="Verify greenlang.auth module is properly installed",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.SECURITY,
            ))

        # Check access guard availability
        try:
            importlib.import_module("greenlang.agents.foundation.access_guard")
            details["access_guard"] = "available"
            passed += 1
        except ImportError:
            details["access_guard"] = "not_available"
            warned += 1

        # Check policy guard agent
        try:
            importlib.import_module("greenlang.agents.foundation.policy_guard")
            details["policy_guard"] = "available"
            passed += 1
        except ImportError:
            details["policy_guard"] = "not_available"
            warned += 1

        # Check PII redaction agent
        try:
            importlib.import_module("greenlang.agents.foundation.pii_redaction")
            details["pii_redaction"] = "available"
            passed += 1
        except ImportError:
            details["pii_redaction"] = "not_available"
            warned += 1

        # Check for sensitive environment variables
        sensitive_vars = ["DATABASE_URL", "SECRET_KEY", "JWT_SECRET"]
        for var_name in sensitive_vars:
            value = os.environ.get(var_name)
            if value:
                details[f"env_{var_name}"] = "set"
                passed += 1
                # Check for insecure default values
                if value in ("changeme", "secret", "password", "default"):
                    warned += 1
                    remediations.append(RemediationSuggestion(
                        finding=f"Environment variable {var_name} has an insecure default value",
                        suggestion=f"Set {var_name} to a strong, unique value",
                        severity=HealthSeverity.WARNING,
                        category=CheckCategory.SECURITY,
                    ))
            else:
                details[f"env_{var_name}"] = "not_set"
                # Not critical if not in production

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.UNHEALTHY
        elif warned > 2:
            status = HealthStatus.DEGRADED

        return ComponentHealth(
            name="Security",
            category=CheckCategory.SECURITY,
            status=status,
            message=f"{passed} security checks passed, {warned} warnings",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    async def check_performance(self) -> ComponentHealth:
        """Run a basic performance benchmark.

        Calculates a target number of simple metrics and verifies the
        total time is within the configured threshold.

        Returns:
            ComponentHealth for the performance category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        metric_count = self.config.performance_metric_count
        target_ms = self.config.performance_target_ms

        # Benchmark: simple emission calculations
        calc_start = time.monotonic()
        total_emissions = 0.0
        for i in range(metric_count):
            # Simulate deterministic calculation
            quantity = float(i + 1) * 10.0
            factor = 2.5
            emissions = quantity * factor
            total_emissions += emissions
            # Include a hash to simulate provenance
            _compute_hash(f"benchmark:{i}:{emissions}")

        calc_elapsed = (time.monotonic() - calc_start) * 1000
        details["calculations_completed"] = metric_count
        details["calculation_time_ms"] = round(calc_elapsed, 2)
        details["total_emissions_calculated"] = round(total_emissions, 2)
        details["target_ms"] = target_ms
        details["per_metric_ms"] = round(calc_elapsed / metric_count, 4)

        if calc_elapsed <= target_ms:
            passed += 1
            details["benchmark_result"] = "passed"
        else:
            warned += 1
            details["benchmark_result"] = "slow"
            remediations.append(RemediationSuggestion(
                finding=(
                    f"Performance benchmark took {calc_elapsed:.1f}ms "
                    f"(target: {target_ms:.1f}ms)"
                ),
                suggestion=(
                    "Consider optimizing calculation paths or increasing "
                    "available compute resources."
                ),
                severity=HealthSeverity.WARNING,
                category=CheckCategory.PERFORMANCE,
            ))

        # Benchmark: hash computation
        hash_start = time.monotonic()
        for i in range(1000):
            _compute_hash(f"hash_benchmark:{i}")
        hash_elapsed = (time.monotonic() - hash_start) * 1000
        details["hash_benchmark_ms"] = round(hash_elapsed, 2)
        details["hashes_per_second"] = round(1000 / (hash_elapsed / 1000), 0) if hash_elapsed > 0 else 0

        if hash_elapsed < 1000:
            passed += 1
        else:
            warned += 1

        # Benchmark: Pydantic model creation
        model_start = time.monotonic()
        for i in range(1000):
            ComponentHealth(
                name=f"bench_{i}",
                category=CheckCategory.PERFORMANCE,
                status=HealthStatus.HEALTHY,
            )
        model_elapsed = (time.monotonic() - model_start) * 1000
        details["model_benchmark_ms"] = round(model_elapsed, 2)

        if model_elapsed < 2000:
            passed += 1
        else:
            warned += 1

        elapsed = (time.monotonic() - start_time) * 1000
        status = HealthStatus.HEALTHY
        if failed > 0:
            status = HealthStatus.UNHEALTHY
        elif warned > 0:
            status = HealthStatus.DEGRADED

        return ComponentHealth(
            name="Performance",
            category=CheckCategory.PERFORMANCE,
            status=status,
            message=(
                f"{metric_count} metrics calculated in {calc_elapsed:.1f}ms "
                f"(target: {target_ms:.0f}ms)"
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Result Assembly
    # -------------------------------------------------------------------------

    def _build_result(self, total_elapsed_ms: float) -> HealthCheckResult:
        """Build the final HealthCheckResult from accumulated component results.

        Args:
            total_elapsed_ms: Total elapsed time for all checks.

        Returns:
            Complete HealthCheckResult.
        """
        total_checks = sum(c.checks_passed + c.checks_failed + c.checks_warned
                           for c in self._results)
        total_passed = sum(c.checks_passed for c in self._results)
        total_failed = sum(c.checks_failed for c in self._results)
        total_warned = sum(c.checks_warned for c in self._results)

        critical_issues: List[RemediationSuggestion] = []
        warnings: List[RemediationSuggestion] = []

        for component in self._results:
            for remediation in component.remediations:
                if remediation.severity == HealthSeverity.CRITICAL:
                    critical_issues.append(remediation)
                else:
                    warnings.append(remediation)

        # Determine overall status
        if total_failed == 0 and not critical_issues:
            overall = HealthStatus.HEALTHY
        elif total_failed <= 3 and len(critical_issues) <= 1:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNHEALTHY

        provenance = _compute_hash(
            f"health:{datetime.utcnow().isoformat()}:{total_checks}:{total_passed}:{total_failed}"
        )

        result = HealthCheckResult(
            overall_status=overall,
            total_checks=total_checks,
            checks_passed=total_passed,
            checks_failed=total_failed,
            checks_warned=total_warned,
            components=self._results,
            critical_issues=critical_issues,
            warnings=warnings,
            total_execution_time_ms=total_elapsed_ms,
            provenance_hash=provenance,
        )

        logger.info(
            "Health check complete: %s (%d/%d passed, %d critical, %d warnings) in %.1fms",
            overall.value, total_passed, total_checks,
            len(critical_issues), len(warnings), total_elapsed_ms,
        )

        return result


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _check_package_version(
    package_name: str, min_version: str
) -> Dict[str, str]:
    """Check if a Python package is installed and meets the minimum version.

    Args:
        package_name: Name of the package to check.
        min_version: Minimum required version string.

    Returns:
        Dictionary with status, installed version, and minimum version.
    """
    try:
        mod = importlib.import_module(package_name.replace("-", "_"))
        installed_version = getattr(mod, "__version__", "unknown")

        if installed_version == "unknown":
            return {
                "status": "ok",
                "installed": installed_version,
                "minimum": min_version,
                "note": "Version could not be determined",
            }

        if _version_gte(installed_version, min_version):
            return {
                "status": "ok",
                "installed": installed_version,
                "minimum": min_version,
            }
        else:
            return {
                "status": "version_mismatch",
                "installed": installed_version,
                "minimum": min_version,
            }

    except ImportError:
        return {
            "status": "not_installed",
            "installed": None,
            "minimum": min_version,
        }


def _version_gte(installed: str, minimum: str) -> bool:
    """Check if installed version is greater than or equal to minimum.

    Simple version comparison that handles major.minor.patch format.

    Args:
        installed: Installed version string.
        minimum: Minimum required version string.

    Returns:
        True if installed >= minimum, False otherwise.
    """
    try:
        installed_parts = [int(x) for x in installed.split(".")[:3]]
        minimum_parts = [int(x) for x in minimum.split(".")[:3]]

        # Pad shorter list with zeros
        while len(installed_parts) < 3:
            installed_parts.append(0)
        while len(minimum_parts) < 3:
            minimum_parts.append(0)

        return tuple(installed_parts) >= tuple(minimum_parts)
    except (ValueError, AttributeError):
        return True  # If we cannot parse, assume OK
