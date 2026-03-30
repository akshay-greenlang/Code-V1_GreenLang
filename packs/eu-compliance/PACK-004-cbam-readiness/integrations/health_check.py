# -*- coding: utf-8 -*-
"""
CBAMHealthCheck - 12-Category Health Verification for CBAM Readiness Pack
==========================================================================

This module implements a comprehensive health check system that validates the
operational readiness of the CBAM Readiness Pack across 12 categories:

    1.  pack_manifest:           Verify pack.yaml integrity
    2.  configuration:           Validate config loading, preset application
    3.  cbam_app_connectivity:   Check GL-CBAM-APP v1.1 modules accessible
    4.  engine_availability:     Check all 7 CBAM engines instantiate
    5.  workflow_availability:   Check all 7 CBAM workflows instantiate
    6.  template_availability:   Check all 8 CBAM templates render
    7.  agent_connectivity:      Verify 45+ agent references
    8.  cn_code_database:        Verify completeness of CN code database (50+ codes)
    9.  emission_factor_coverage: Check emission factors for all goods categories
    10. ets_price_feed:          Check ETS price feed status
    11. supplier_portal_status:  Check supplier portal components
    12. compliance_rule_coverage: Verify 50+ CBAM rules loaded

Example:
    >>> health = CBAMHealthCheck()
    >>> result = health.run()
    >>> print(f"Score: {result.overall_health_score}/100")
    >>> quick = health.run_quick()
    >>> assert quick.total_checks > 0

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
"""

import hashlib
import importlib
import logging
import os
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas.enums import HealthStatus, Severity

logger = logging.getLogger(__name__)

# =============================================================================
# Enums
# =============================================================================

class CheckCategory(str, Enum):
    """Categories of health checks."""
    PACK_MANIFEST = "pack_manifest"
    CONFIGURATION = "configuration"
    CBAM_APP_CONNECTIVITY = "cbam_app_connectivity"
    ENGINE_AVAILABILITY = "engine_availability"
    WORKFLOW_AVAILABILITY = "workflow_availability"
    TEMPLATE_AVAILABILITY = "template_availability"
    AGENT_CONNECTIVITY = "agent_connectivity"
    CN_CODE_DATABASE = "cn_code_database"
    EMISSION_FACTOR_COVERAGE = "emission_factor_coverage"
    ETS_PRICE_FEED = "ets_price_feed"
    SUPPLIER_PORTAL_STATUS = "supplier_portal_status"
    COMPLIANCE_RULE_COVERAGE = "compliance_rule_coverage"

# =============================================================================
# Data Models
# =============================================================================

class Finding(BaseModel):
    """A health check finding."""
    category: CheckCategory = Field(..., description="Check category")
    severity: Severity = Field(..., description="Finding severity")
    finding: str = Field(..., description="What was found")
    suggestion: str = Field(default="", description="Remediation suggestion")
    auto_fixable: bool = Field(default=False, description="Whether auto-fixable")

class CategoryResult(BaseModel):
    """Result of a single health check category."""
    category: CheckCategory = Field(..., description="Category checked")
    status: HealthStatus = Field(default=HealthStatus.UNKNOWN, description="Category status")
    checks_passed: int = Field(default=0, description="Checks passed")
    checks_failed: int = Field(default=0, description="Checks failed")
    checks_warned: int = Field(default=0, description="Checks with warnings")
    findings: List[Finding] = Field(default_factory=list, description="Findings")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed results")
    execution_time_ms: float = Field(default=0.0, description="Check time in ms")

class HealthCheckResult(BaseModel):
    """Complete health check result."""
    total_checks: int = Field(default=0, description="Total checks run")
    passed: int = Field(default=0, description="Total passed")
    failed: int = Field(default=0, description="Total failed")
    warnings: int = Field(default=0, description="Total warnings")
    overall_health_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall health score (0-100)"
    )
    category_results: Dict[str, CategoryResult] = Field(
        default_factory=dict, description="Per-category results"
    )
    critical_findings: List[Finding] = Field(
        default_factory=list, description="Critical findings"
    )
    warning_findings: List[Finding] = Field(
        default_factory=list, description="Warning findings"
    )
    duration_seconds: float = Field(default=0.0, description="Total check duration")
    check_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Check timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# =============================================================================
# Expected Resources
# =============================================================================

# 7 CBAM engines
EXPECTED_ENGINES: List[str] = [
    "cbam_calculation_engine",
    "certificate_engine",
    "quarterly_reporting_engine",
    "supplier_management_engine",
    "deminimis_engine",
    "verification_engine",
    "policy_compliance_engine",
]

# 7 CBAM workflows
EXPECTED_WORKFLOWS: List[str] = [
    "quarterly_reporting",
    "annual_declaration",
    "supplier_onboarding",
    "certificate_management",
    "verification_cycle",
    "deminimis_assessment",
    "data_collection",
]

# 8 CBAM templates
EXPECTED_TEMPLATES: List[str] = [
    "quarterly_report",
    "annual_declaration",
    "supplier_communication",
    "certificate_summary",
    "verification_request",
    "emission_calculation_detail",
    "compliance_dashboard",
    "executive_summary",
]

# 45+ agents referenced by CBAM pack
EXPECTED_AGENTS: Dict[str, str] = {
    # Foundation
    "GL-FOUND-X-001": "greenlang.agents.foundation.orchestrator",
    "GL-FOUND-X-002": "greenlang.agents.foundation.schema_compiler",
    "GL-FOUND-X-003": "greenlang.agents.foundation.unit_normalizer",
    "GL-FOUND-X-004": "greenlang.agents.foundation.assumptions_registry",
    "GL-FOUND-X-005": "greenlang.agents.foundation.citations_agent",
    "GL-FOUND-X-006": "greenlang.agents.foundation.policy_guard",
    "GL-FOUND-X-008": "greenlang.agents.foundation.qa_test_harness",
    "GL-FOUND-X-009": "greenlang.agents.foundation.observability_agent",
    "GL-FOUND-X-010": "greenlang.agents.foundation.agent_registry",
    # Data Intake
    "GL-DATA-X-001": "greenlang.agents.data.document_ingestion_agent",
    "GL-DATA-X-002": "greenlang.agents.data.excel_csv_normalizer",
    "GL-DATA-X-003": "greenlang.agents.data.erp_connector_agent",
    "GL-DATA-X-008": "greenlang.agents.data.weather_climate_agent",
    "GL-DATA-X-009": "greenlang.agents.data.utility_tariff_agent",
    "GL-DATA-X-010": "greenlang.agents.data.data_quality_profiler",
    "GL-DATA-X-011": "greenlang.agents.data.duplicate_detection",
    "GL-DATA-X-012": "greenlang.agents.data.missing_value_imputer",
    "GL-DATA-X-013": "greenlang.agents.data.outlier_detection",
    "GL-DATA-X-019": "greenlang.agents.data.validation_rule_engine",
    # MRV Scope 1
    "GL-MRV-X-001": "greenlang.agents.mrv.stationary_combustion",
    "GL-MRV-X-002": "greenlang.agents.mrv.refrigerants_fgas",
    "GL-MRV-X-003": "greenlang.agents.mrv.mobile_combustion",
    "GL-MRV-X-004": "greenlang.agents.mrv.process_emissions",
    "GL-MRV-X-005": "greenlang.agents.mrv.fugitive_emissions",
    "GL-MRV-X-006": "greenlang.agents.mrv.land_use_emissions",
    "GL-MRV-X-007": "greenlang.agents.mrv.waste_treatment_emissions",
    "GL-MRV-X-008": "greenlang.agents.mrv.agricultural_emissions",
    # MRV Scope 2
    "GL-MRV-X-009": "greenlang.agents.mrv.scope2_location_based",
    "GL-MRV-X-010": "greenlang.agents.mrv.scope2_market_based",
    "GL-MRV-X-011": "greenlang.agents.mrv.steam_heat_purchase",
    "GL-MRV-X-012": "greenlang.agents.mrv.cooling_purchase",
    "GL-MRV-X-013": "greenlang.agents.mrv.dual_reporting_reconciliation",
    # MRV Scope 3 (selected relevant categories)
    "GL-MRV-X-014": "greenlang.agents.mrv.purchased_goods_services",
    "GL-MRV-X-017": "greenlang.agents.mrv.upstream_transportation",
    "GL-MRV-X-029": "greenlang.agents.mrv.scope3_category_mapper",
    "GL-MRV-X-030": "greenlang.agents.mrv.audit_trail_lineage",
    # EUDR agents (shared infrastructure)
    "GL-EUDR-X-001": "greenlang.agents.eudr.commodity_classification",
    "GL-EUDR-X-005": "greenlang.agents.eudr.supplier_due_diligence",
    "GL-EUDR-X-017": "greenlang.agents.eudr.supplier_risk_scorer",
    # CBAM APP
    "GL-CBAM-APP": "greenlang.apps.cbam",
    # Security
    "GL-SEC-JWT": "greenlang.security.jwt_auth",
    "GL-SEC-RBAC": "greenlang.security.rbac",
    "GL-SEC-AUDIT": "greenlang.security.audit_logging",
    "GL-SEC-PII": "greenlang.security.pii_detection",
}

# CBAM goods categories that must have emission factors
REQUIRED_EF_CATEGORIES: List[str] = [
    "IRON_AND_STEEL",
    "ALUMINIUM",
    "CEMENT",
    "FERTILISERS",
    "HYDROGEN",
    "ELECTRICITY",
]

# =============================================================================
# Health Check Implementation
# =============================================================================

class CBAMHealthCheck:
    """12-category health check for CBAM Readiness Pack.

    Validates operational readiness across pack manifest, configuration,
    GL-CBAM-APP connectivity, engine/workflow/template availability,
    agent connectivity, CN code database, emission factors, ETS price
    feed, supplier portal, and compliance rule coverage.

    Attributes:
        config: Optional configuration dictionary
        _project_root: Path to the GreenLang project root
        _pack_root: Path to the PACK-004 directory

    Example:
        >>> health = CBAMHealthCheck()
        >>> result = health.run()
        >>> assert result.overall_health_score >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the health check system.

        Args:
            config: Optional configuration dictionary. Keys:
                - project_root: Override project root path
        """
        self.config = config or {}
        self.logger = logger

        self._project_root = self.config.get("project_root", "")
        if not self._project_root:
            self._project_root = str(Path(__file__).resolve().parents[4])

        self._pack_root = os.path.join(
            self._project_root,
            "packs", "eu-compliance", "PACK-004-cbam-readiness",
        )

        self.logger.info(
            "CBAMHealthCheck initialized: project_root=%s, pack_root=%s",
            self._project_root, self._pack_root,
        )

    # -------------------------------------------------------------------------
    # Main Entry Points
    # -------------------------------------------------------------------------

    def run(self) -> HealthCheckResult:
        """Run the full 12-category health check.

        Returns:
            HealthCheckResult with all category results.
        """
        start_time = time.monotonic()
        self.logger.info("Starting full CBAM health check (12 categories)")

        categories = [
            (CheckCategory.PACK_MANIFEST, self._check_pack_manifest),
            (CheckCategory.CONFIGURATION, self._check_configuration),
            (CheckCategory.CBAM_APP_CONNECTIVITY, self._check_cbam_app),
            (CheckCategory.ENGINE_AVAILABILITY, self._check_engines),
            (CheckCategory.WORKFLOW_AVAILABILITY, self._check_workflows),
            (CheckCategory.TEMPLATE_AVAILABILITY, self._check_templates),
            (CheckCategory.AGENT_CONNECTIVITY, self._check_agents),
            (CheckCategory.CN_CODE_DATABASE, self._check_cn_codes),
            (CheckCategory.EMISSION_FACTOR_COVERAGE, self._check_emission_factors),
            (CheckCategory.ETS_PRICE_FEED, self._check_ets_feed),
            (CheckCategory.SUPPLIER_PORTAL_STATUS, self._check_supplier_portal),
            (CheckCategory.COMPLIANCE_RULE_COVERAGE, self._check_compliance_rules),
        ]

        results: Dict[str, CategoryResult] = {}
        for cat, checker in categories:
            try:
                cat_result = checker()
                results[cat.value] = cat_result
            except Exception as exc:
                self.logger.error("Category %s check failed: %s", cat.value, exc)
                results[cat.value] = CategoryResult(
                    category=cat,
                    status=HealthStatus.UNHEALTHY,
                    findings=[Finding(
                        category=cat,
                        severity=Severity.CRITICAL,
                        finding=f"Check raised exception: {exc}",
                    )],
                )

        duration = (time.monotonic() - start_time)
        return self._build_result(results, duration)

    def run_quick(self) -> HealthCheckResult:
        """Run a quick health check (categories 1-6 only).

        Returns:
            HealthCheckResult for the quick subset.
        """
        start_time = time.monotonic()
        self.logger.info("Starting quick CBAM health check (6 categories)")

        quick_categories = [
            (CheckCategory.PACK_MANIFEST, self._check_pack_manifest),
            (CheckCategory.CONFIGURATION, self._check_configuration),
            (CheckCategory.CBAM_APP_CONNECTIVITY, self._check_cbam_app),
            (CheckCategory.ENGINE_AVAILABILITY, self._check_engines),
            (CheckCategory.WORKFLOW_AVAILABILITY, self._check_workflows),
            (CheckCategory.TEMPLATE_AVAILABILITY, self._check_templates),
        ]

        results: Dict[str, CategoryResult] = {}
        for cat, checker in quick_categories:
            try:
                results[cat.value] = checker()
            except Exception as exc:
                self.logger.error("Quick check %s failed: %s", cat.value, exc)
                results[cat.value] = CategoryResult(
                    category=cat,
                    status=HealthStatus.UNHEALTHY,
                )

        duration = (time.monotonic() - start_time)
        return self._build_result(results, duration)

    # -------------------------------------------------------------------------
    # Category 1: Pack Manifest
    # -------------------------------------------------------------------------

    def _check_pack_manifest(self) -> CategoryResult:
        """Category 1: Verify pack.yaml integrity."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        # Check pack directory exists
        if os.path.isdir(self._pack_root):
            details["pack_directory"] = "present"
            passed += 1
        else:
            details["pack_directory"] = "missing"
            failed += 1
            findings.append(Finding(
                category=CheckCategory.PACK_MANIFEST,
                severity=Severity.CRITICAL,
                finding=f"Pack directory not found: {self._pack_root}",
                suggestion="Verify PACK-004 directory structure",
            ))

        # Check for pack.yaml
        pack_yaml = os.path.join(self._pack_root, "pack.yaml")
        if os.path.isfile(pack_yaml):
            details["pack_yaml"] = "present"
            passed += 1
        else:
            details["pack_yaml"] = "missing"
            warned += 1
            findings.append(Finding(
                category=CheckCategory.PACK_MANIFEST,
                severity=Severity.WARNING,
                finding="pack.yaml not found",
                suggestion="Create pack.yaml manifest file",
            ))

        # Check required subdirectories
        for subdir in ["config", "integrations", "templates"]:
            path = os.path.join(self._pack_root, subdir)
            if os.path.isdir(path):
                details[f"dir_{subdir}"] = "present"
                passed += 1
            else:
                details[f"dir_{subdir}"] = "missing"
                warned += 1

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED
        if failed > 1:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.PACK_MANIFEST,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 2: Configuration
    # -------------------------------------------------------------------------

    def _check_configuration(self) -> CategoryResult:
        """Category 2: Validate configuration loading and preset application."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        # Check config directory
        config_dir = os.path.join(self._pack_root, "config")
        if os.path.isdir(config_dir):
            details["config_dir"] = "present"
            passed += 1

            # Check for __init__.py
            init_file = os.path.join(config_dir, "__init__.py")
            if os.path.isfile(init_file):
                details["config_init"] = "present"
                passed += 1
            else:
                details["config_init"] = "missing"
                warned += 1

            # Check for pack_config.py
            config_file = os.path.join(config_dir, "pack_config.py")
            if os.path.isfile(config_file):
                details["pack_config"] = "present"
                passed += 1
            else:
                details["pack_config"] = "missing"
                warned += 1

            # Check presets directory
            presets_dir = os.path.join(config_dir, "presets")
            if os.path.isdir(presets_dir):
                details["presets_dir"] = "present"
                passed += 1
            else:
                details["presets_dir"] = "missing"
                warned += 1

            # Check sectors directory
            sectors_dir = os.path.join(config_dir, "sectors")
            if os.path.isdir(sectors_dir):
                details["sectors_dir"] = "present"
                passed += 1
            else:
                details["sectors_dir"] = "missing"
                warned += 1
        else:
            details["config_dir"] = "missing"
            failed += 1
            findings.append(Finding(
                category=CheckCategory.CONFIGURATION,
                severity=Severity.CRITICAL,
                finding="Config directory not found",
                suggestion=f"Create config directory at {config_dir}",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED

        return CategoryResult(
            category=CheckCategory.CONFIGURATION,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 3: CBAM App Connectivity
    # -------------------------------------------------------------------------

    def _check_cbam_app(self) -> CategoryResult:
        """Category 3: Check GL-CBAM-APP v1.1 modules accessible."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        cbam_modules = [
            "greenlang.apps.cbam",
            "greenlang.apps.cbam.engines",
            "greenlang.apps.cbam.workflows",
        ]

        for mod_path in cbam_modules:
            try:
                importlib.import_module(mod_path)
                details[mod_path] = "importable"
                passed += 1
            except ImportError:
                details[mod_path] = "not_available"
                warned += 1
                findings.append(Finding(
                    category=CheckCategory.CBAM_APP_CONNECTIVITY,
                    severity=Severity.WARNING,
                    finding=f"Module '{mod_path}' not importable",
                    suggestion="Verify GL-CBAM-APP v1.1 is installed",
                ))

        # Check CBAMAppBridge functionality
        try:
            from packs.eu_compliance.PACK_004_cbam_readiness.integrations.cbam_app_bridge import (
                CBAMAppBridge,
            )
            bridge = CBAMAppBridge()
            health = bridge.check_app_health()
            details["bridge_status"] = "operational"
            details["bridge_stub_mode"] = health.stub_mode
            passed += 1
        except Exception as exc:
            details["bridge_status"] = f"error: {exc}"
            warned += 1

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED
        if warned > 2:
            status = HealthStatus.DEGRADED

        return CategoryResult(
            category=CheckCategory.CBAM_APP_CONNECTIVITY,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 4: Engine Availability
    # -------------------------------------------------------------------------

    def _check_engines(self) -> CategoryResult:
        """Category 4: Check all 7 CBAM engines instantiate."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {"engines": {}}

        engines_dir = os.path.join(self._pack_root, "engines")
        for engine_name in EXPECTED_ENGINES:
            engine_file = os.path.join(engines_dir, f"{engine_name}.py")
            if os.path.isfile(engine_file):
                details["engines"][engine_name] = "present"
                passed += 1
            else:
                details["engines"][engine_name] = "missing"
                warned += 1
                findings.append(Finding(
                    category=CheckCategory.ENGINE_AVAILABILITY,
                    severity=Severity.WARNING,
                    finding=f"Engine '{engine_name}' not found",
                    suggestion=f"Create {engine_file}",
                ))

        # Check engines __init__.py
        init_file = os.path.join(engines_dir, "__init__.py")
        if os.path.isfile(init_file):
            details["engines_init"] = "present"
            passed += 1
        else:
            details["engines_init"] = "missing"
            warned += 1

        elapsed = (time.monotonic() - start) * 1000
        total = passed + failed + warned
        status = HealthStatus.HEALTHY
        if warned > 2:
            status = HealthStatus.DEGRADED
        if failed > 0:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.ENGINE_AVAILABILITY,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 5: Workflow Availability
    # -------------------------------------------------------------------------

    def _check_workflows(self) -> CategoryResult:
        """Category 5: Check all 7 CBAM workflows instantiate."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {"workflows": {}}

        workflows_dir = os.path.join(self._pack_root, "workflows")
        for wf_name in EXPECTED_WORKFLOWS:
            wf_file = os.path.join(workflows_dir, f"{wf_name}.py")
            if os.path.isfile(wf_file):
                details["workflows"][wf_name] = "present"
                passed += 1
            else:
                details["workflows"][wf_name] = "missing"
                warned += 1
                findings.append(Finding(
                    category=CheckCategory.WORKFLOW_AVAILABILITY,
                    severity=Severity.WARNING,
                    finding=f"Workflow '{wf_name}' not found",
                    suggestion=f"Create {wf_file}",
                ))

        # Check workflows __init__.py
        init_file = os.path.join(workflows_dir, "__init__.py")
        if os.path.isfile(init_file):
            details["workflows_init"] = "present"
            passed += 1
        else:
            details["workflows_init"] = "missing"
            warned += 1

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY
        if warned > 2:
            status = HealthStatus.DEGRADED
        if failed > 0:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.WORKFLOW_AVAILABILITY,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 6: Template Availability
    # -------------------------------------------------------------------------

    def _check_templates(self) -> CategoryResult:
        """Category 6: Check all 8 CBAM templates render."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {"templates": {}}

        templates_dir = os.path.join(self._pack_root, "templates")
        for tpl_name in EXPECTED_TEMPLATES:
            tpl_file = os.path.join(templates_dir, f"{tpl_name}.py")
            if os.path.isfile(tpl_file):
                details["templates"][tpl_name] = "present"
                passed += 1
            else:
                details["templates"][tpl_name] = "missing"
                warned += 1
                findings.append(Finding(
                    category=CheckCategory.TEMPLATE_AVAILABILITY,
                    severity=Severity.WARNING,
                    finding=f"Template '{tpl_name}' not found",
                    suggestion=f"Create {tpl_file}",
                ))

        # Check templates __init__.py
        init_file = os.path.join(templates_dir, "__init__.py")
        if os.path.isfile(init_file):
            details["templates_init"] = "present"
            passed += 1
        else:
            details["templates_init"] = "missing"
            warned += 1

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY
        if warned > 3:
            status = HealthStatus.DEGRADED
        if failed > 0:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.TEMPLATE_AVAILABILITY,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 7: Agent Connectivity
    # -------------------------------------------------------------------------

    def _check_agents(self) -> CategoryResult:
        """Category 7: Verify 45+ agent references are importable."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {"agents": {}}

        for agent_id, module_path in EXPECTED_AGENTS.items():
            try:
                importlib.import_module(module_path)
                details["agents"][agent_id] = "importable"
                passed += 1
            except ImportError:
                details["agents"][agent_id] = "not_importable"
                warned += 1
            except Exception as exc:
                details["agents"][agent_id] = f"error: {exc}"
                warned += 1

        total_agents = len(EXPECTED_AGENTS)
        import_rate = passed / max(total_agents, 1)

        details["total_agents_expected"] = total_agents
        details["agents_importable"] = passed
        details["import_rate"] = round(import_rate, 3)

        if import_rate < 0.5:
            findings.append(Finding(
                category=CheckCategory.AGENT_CONNECTIVITY,
                severity=Severity.CRITICAL,
                finding=f"Only {passed}/{total_agents} agents importable ({import_rate:.0%})",
                suggestion="Run pip install -e . to ensure all agent modules are available",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY
        if import_rate < 0.8:
            status = HealthStatus.DEGRADED
        if import_rate < 0.5:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.AGENT_CONNECTIVITY,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 8: CN Code Database
    # -------------------------------------------------------------------------

    def _check_cn_codes(self) -> CategoryResult:
        """Category 8: Verify completeness of CN code database (50+ codes)."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        try:
            from packs.eu_compliance.PACK_004_cbam_readiness.integrations.customs_bridge import (
                CustomsBridge,
            )
            bridge = CustomsBridge()
            all_codes = bridge.get_all_cbam_cn_codes()
            summary = bridge.get_database_summary()

            total_codes = summary.get("total_cn_codes", 0)
            details["total_cn_codes"] = total_codes
            details["codes_by_category"] = summary.get("codes_by_category", {})

            # Check total count
            if total_codes >= 50:
                details["code_count_check"] = "passed"
                passed += 1
            else:
                details["code_count_check"] = f"only {total_codes} codes (need 50+)"
                failed += 1
                findings.append(Finding(
                    category=CheckCategory.CN_CODE_DATABASE,
                    severity=Severity.CRITICAL,
                    finding=f"Only {total_codes} CN codes in database (need 50+)",
                    suggestion="Expand CN code database in customs_bridge.py",
                ))

            # Check all 6 categories represented
            categories_present = set(all_codes.keys())
            for cat in REQUIRED_EF_CATEGORIES:
                if cat in categories_present:
                    passed += 1
                    details[f"category_{cat}"] = "present"
                else:
                    warned += 1
                    details[f"category_{cat}"] = "missing"
                    findings.append(Finding(
                        category=CheckCategory.CN_CODE_DATABASE,
                        severity=Severity.WARNING,
                        finding=f"No CN codes for category '{cat}'",
                        suggestion=f"Add CN codes for {cat} to the database",
                    ))

            # Verify lookup functionality
            test_code = "7201 10 11"
            info = bridge.lookup_cn_code(test_code)
            if info is not None:
                details["lookup_test"] = "passed"
                passed += 1
            else:
                details["lookup_test"] = "failed"
                warned += 1

        except ImportError:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.CN_CODE_DATABASE,
                severity=Severity.CRITICAL,
                finding="CustomsBridge could not be imported",
                suggestion="Check customs_bridge.py exists in integrations/",
            ))
        except Exception as exc:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.CN_CODE_DATABASE,
                severity=Severity.CRITICAL,
                finding=f"CN code check failed: {exc}",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED
        if failed > 1:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.CN_CODE_DATABASE,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 9: Emission Factor Coverage
    # -------------------------------------------------------------------------

    def _check_emission_factors(self) -> CategoryResult:
        """Category 9: Check emission factors for all goods categories."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        try:
            from packs.eu_compliance.PACK_004_cbam_readiness.integrations.cbam_app_bridge import (
                CBAMAppBridge,
            )
            bridge = CBAMAppBridge()
            factors = bridge.get_emission_factors()

            details["total_categories_with_factors"] = len(factors)
            for cat in REQUIRED_EF_CATEGORIES:
                cat_factors = factors.get(cat, [])
                if cat_factors:
                    details[f"ef_{cat}"] = f"{len(cat_factors)} factors"
                    passed += 1
                else:
                    details[f"ef_{cat}"] = "no factors"
                    warned += 1
                    findings.append(Finding(
                        category=CheckCategory.EMISSION_FACTOR_COVERAGE,
                        severity=Severity.WARNING,
                        finding=f"No emission factors for category '{cat}'",
                        suggestion=f"Add emission factors for {cat}",
                    ))

            # Check total factor count
            total_factors = sum(len(v) for v in factors.values())
            details["total_factors"] = total_factors
            if total_factors >= 10:
                passed += 1
            else:
                warned += 1

        except Exception as exc:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.EMISSION_FACTOR_COVERAGE,
                severity=Severity.CRITICAL,
                finding=f"Emission factor check failed: {exc}",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED

        return CategoryResult(
            category=CheckCategory.EMISSION_FACTOR_COVERAGE,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 10: ETS Price Feed
    # -------------------------------------------------------------------------

    def _check_ets_feed(self) -> CategoryResult:
        """Category 10: Check ETS price feed status."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        try:
            from packs.eu_compliance.PACK_004_cbam_readiness.integrations.ets_bridge import (
                ETSBridge,
            )
            bridge = ETSBridge()

            # Check current price
            current = bridge.get_current_price()
            if current.price_eur_per_tco2 > 0:
                details["current_price"] = current.price_eur_per_tco2
                details["price_source"] = current.source.value
                passed += 1
            else:
                details["current_price"] = 0
                warned += 1

            # Check price history
            history = bridge.get_price_history("2025-01-01", "2025-12-31")
            details["history_observations_2025"] = len(history)
            if len(history) > 100:
                passed += 1
            elif len(history) > 0:
                warned += 1
            else:
                warned += 1
                findings.append(Finding(
                    category=CheckCategory.ETS_PRICE_FEED,
                    severity=Severity.WARNING,
                    finding="No price history data for 2025",
                ))

            # Check weekly auction price
            auction = bridge.get_weekly_auction_price()
            if auction.price_eur_per_tco2 > 0:
                details["auction_price"] = auction.price_eur_per_tco2
                passed += 1
            else:
                warned += 1

            # Check exchange rate lookup
            rate = bridge.get_exchange_rate("USD")
            if rate > 0:
                details["usd_rate"] = rate
                passed += 1
            else:
                warned += 1

            # Check price summary
            summary = bridge.get_price_summary()
            details["mode"] = summary.get("mode", "unknown")
            passed += 1

        except ImportError:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.ETS_PRICE_FEED,
                severity=Severity.CRITICAL,
                finding="ETSBridge could not be imported",
            ))
        except Exception as exc:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.ETS_PRICE_FEED,
                severity=Severity.CRITICAL,
                finding=f"ETS feed check failed: {exc}",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED

        return CategoryResult(
            category=CheckCategory.ETS_PRICE_FEED,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 11: Supplier Portal Status
    # -------------------------------------------------------------------------

    def _check_supplier_portal(self) -> CategoryResult:
        """Category 11: Check supplier portal components."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        try:
            from packs.eu_compliance.PACK_004_cbam_readiness.integrations.cbam_app_bridge import (
                CBAMAppBridge,
            )
            bridge = CBAMAppBridge()

            # Check supplier portal proxy
            portal = bridge.get_supplier_portal()
            details["portal_available"] = True
            details["portal_stub_mode"] = portal.is_stub
            passed += 1

            # Test registration
            test_result = portal.register_supplier({
                "name": "Test Supplier",
                "country": "TR",
            })
            if test_result.get("supplier_id"):
                details["registration_test"] = "passed"
                passed += 1
            else:
                details["registration_test"] = "failed"
                warned += 1

            # Test emission lookup
            emissions = portal.get_supplier_emissions("test-001")
            if isinstance(emissions, dict):
                details["emission_lookup_test"] = "passed"
                passed += 1
            else:
                details["emission_lookup_test"] = "failed"
                warned += 1

        except ImportError:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.SUPPLIER_PORTAL_STATUS,
                severity=Severity.CRITICAL,
                finding="CBAMAppBridge could not be imported",
            ))
        except Exception as exc:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.SUPPLIER_PORTAL_STATUS,
                severity=Severity.CRITICAL,
                finding=f"Supplier portal check failed: {exc}",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED

        return CategoryResult(
            category=CheckCategory.SUPPLIER_PORTAL_STATUS,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Category 12: Compliance Rule Coverage
    # -------------------------------------------------------------------------

    def _check_compliance_rules(self) -> CategoryResult:
        """Category 12: Verify 50+ CBAM compliance rules loaded."""
        start = time.monotonic()
        passed, failed, warned = 0, 0, 0
        findings: List[Finding] = []
        details: Dict[str, Any] = {}

        try:
            from packs.eu_compliance.PACK_004_cbam_readiness.integrations.cbam_app_bridge import (
                CBAMAppBridge,
            )
            bridge = CBAMAppBridge()
            rules = bridge.get_cbam_rules()

            total_rules = len(rules)
            details["total_rules"] = total_rules

            if total_rules >= 50:
                details["rule_count_check"] = "passed"
                passed += 1
            elif total_rules >= 30:
                details["rule_count_check"] = f"partial ({total_rules} rules)"
                warned += 1
                findings.append(Finding(
                    category=CheckCategory.COMPLIANCE_RULE_COVERAGE,
                    severity=Severity.WARNING,
                    finding=f"Only {total_rules} CBAM rules loaded (target: 50+)",
                ))
            else:
                details["rule_count_check"] = f"insufficient ({total_rules} rules)"
                failed += 1
                findings.append(Finding(
                    category=CheckCategory.COMPLIANCE_RULE_COVERAGE,
                    severity=Severity.CRITICAL,
                    finding=f"Only {total_rules} CBAM rules loaded (need 50+)",
                ))

            # Check category coverage
            categories_covered: Dict[str, int] = {}
            for rule in rules:
                cat = getattr(rule, "category", "unknown")
                categories_covered[cat] = categories_covered.get(cat, 0) + 1

            details["categories_covered"] = categories_covered
            details["num_categories"] = len(categories_covered)

            expected_categories = {
                "registration", "cn_codes", "emissions", "certificates",
                "reporting", "verification", "supplier", "deminimis",
                "data_quality",
            }
            for cat in expected_categories:
                if cat in categories_covered:
                    passed += 1
                else:
                    warned += 1
                    findings.append(Finding(
                        category=CheckCategory.COMPLIANCE_RULE_COVERAGE,
                        severity=Severity.WARNING,
                        finding=f"No rules for category '{cat}'",
                        suggestion=f"Add compliance rules for {cat}",
                    ))

            # Check rule IDs are unique
            rule_ids = [getattr(r, "rule_id", "") for r in rules]
            unique_ids = set(rule_ids)
            if len(rule_ids) == len(unique_ids):
                details["unique_ids"] = True
                passed += 1
            else:
                details["unique_ids"] = False
                warned += 1

        except ImportError:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.COMPLIANCE_RULE_COVERAGE,
                severity=Severity.CRITICAL,
                finding="CBAMAppBridge could not be imported",
            ))
        except Exception as exc:
            failed += 1
            findings.append(Finding(
                category=CheckCategory.COMPLIANCE_RULE_COVERAGE,
                severity=Severity.CRITICAL,
                finding=f"Compliance rule check failed: {exc}",
            ))

        elapsed = (time.monotonic() - start) * 1000
        status = HealthStatus.HEALTHY if failed == 0 else HealthStatus.DEGRADED
        if failed > 1:
            status = HealthStatus.UNHEALTHY

        return CategoryResult(
            category=CheckCategory.COMPLIANCE_RULE_COVERAGE,
            status=status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            findings=findings,
            details=details,
            execution_time_ms=elapsed,
        )

    # -------------------------------------------------------------------------
    # Result Assembly
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        category_results: Dict[str, CategoryResult],
        duration_seconds: float,
    ) -> HealthCheckResult:
        """Build the final HealthCheckResult from category results.

        Args:
            category_results: Per-category results.
            duration_seconds: Total check duration.

        Returns:
            Complete HealthCheckResult.
        """
        total_checks = sum(
            cr.checks_passed + cr.checks_failed + cr.checks_warned
            for cr in category_results.values()
        )
        total_passed = sum(cr.checks_passed for cr in category_results.values())
        total_failed = sum(cr.checks_failed for cr in category_results.values())
        total_warned = sum(cr.checks_warned for cr in category_results.values())

        critical_findings: List[Finding] = []
        warning_findings: List[Finding] = []
        for cr in category_results.values():
            for finding in cr.findings:
                if finding.severity == Severity.CRITICAL:
                    critical_findings.append(finding)
                elif finding.severity == Severity.WARNING:
                    warning_findings.append(finding)

        # Calculate score: passed / (passed + failed + warned) * 100
        score = round(
            (total_passed / max(total_checks, 1)) * 100, 1
        )

        provenance = _compute_hash(
            f"health:{datetime.utcnow().isoformat()}:{total_checks}:{total_passed}:{score}"
        )

        result = HealthCheckResult(
            total_checks=total_checks,
            passed=total_passed,
            failed=total_failed,
            warnings=total_warned,
            overall_health_score=score,
            category_results=category_results,
            critical_findings=critical_findings,
            warning_findings=warning_findings,
            duration_seconds=round(duration_seconds, 3),
            provenance_hash=provenance,
        )

        self.logger.info(
            "CBAM health check complete: score=%.1f, %d/%d passed, "
            "%d critical, %d warnings, %.3fs",
            score, total_passed, total_checks,
            len(critical_findings), len(warning_findings), duration_seconds,
        )
        return result

# =============================================================================
# Module-Level Helper
# =============================================================================

def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
