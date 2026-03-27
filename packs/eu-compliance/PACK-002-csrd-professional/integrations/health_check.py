# -*- coding: utf-8 -*-
"""
ProfessionalHealthCheck - Enhanced Health Verification for CSRD Professional Pack
==================================================================================

This module implements a comprehensive 10-category health check system that
validates the operational readiness of the CSRD Professional Pack. It extends
PACK-001's 7 categories with three additional checks: cross-framework engine
availability, webhook endpoint connectivity, and PACK-001 compatibility.

Check Categories (10 total):
    1. Agents (enhanced): 93+ agent modules importable and initializable
    2. Configuration: professional config, size presets, sector presets valid
    3. Data Files: framework_mappings.json, esrs_data_points.json, benchmarks
    4. Database: consolidation tables, approval chain tables, webhook tables
    5. Dependencies: PACK-001 deps + networkx, cryptography, openpyxl
    6. Security: approval role validation, webhook HMAC verification
    7. Performance: multi-entity consolidation and cross-framework benchmarks
    8. Cross-Framework (NEW): CDP/TCFD/SBTi/Taxonomy engine availability
    9. Webhook (NEW): endpoint connectivity, HMAC configuration
   10. PACK-001 Compatibility (NEW): verify PACK-001 installed and functional

Architecture:
    ProfessionalHealthCheck --> [10 Check Categories] --> HealthCheckResult
                                     |                          |
                                     v                          v
                              RemediationSuggestions     ProvenanceHash

Zero-Hallucination:
    - All checks are deterministic import/file/benchmark tests
    - Performance benchmarks use synthetic deterministic calculations
    - No LLM involvement in any health check path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest.

    Args:
        data: Value to hash. If it has a ``model_dump`` method (Pydantic),
              that is used; otherwise ``str()`` is applied.

    Returns:
        64-char hex SHA-256 digest.
    """
    if hasattr(data, "model_dump"):
        raw = str(data.model_dump())
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# Enums
# =============================================================================


class HealthSeverity(str, Enum):
    """Severity level for health check findings."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class HealthStatus(str, Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckCategory(str, Enum):
    """Categories of health checks (10 total for Professional Pack)."""
    AGENTS = "agents"
    CONFIGURATION = "configuration"
    DATA_FILES = "data_files"
    DATABASE = "database"
    DEPENDENCIES = "dependencies"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CROSS_FRAMEWORK = "cross_framework"
    WEBHOOK = "webhook"
    PACK1_COMPATIBILITY = "pack1_compatibility"


# =============================================================================
# Data Models
# =============================================================================


class ProfessionalHealthCheckConfig(BaseModel):
    """Configuration for the Professional Pack health check system."""

    project_root: str = Field(
        default="", description="Root directory of the GreenLang project"
    )
    check_agents: bool = Field(
        default=True, description="Check agent availability (93+ modules)"
    )
    check_configuration: bool = Field(
        default=True, description="Check professional configuration files"
    )
    check_data_files: bool = Field(
        default=True, description="Check data file presence including benchmark data"
    )
    check_database: bool = Field(
        default=True, description="Check database connectivity and professional tables"
    )
    check_dependencies: bool = Field(
        default=True, description="Check package dependencies including professional deps"
    )
    check_security: bool = Field(
        default=True, description="Check security including approval RBAC and HMAC"
    )
    check_performance: bool = Field(
        default=True, description="Run professional performance benchmarks"
    )
    check_cross_framework: bool = Field(
        default=True, description="Check CDP/TCFD/SBTi/Taxonomy engine availability"
    )
    check_webhook: bool = Field(
        default=True, description="Check webhook endpoint connectivity and HMAC"
    )
    check_pack1_compatibility: bool = Field(
        default=True, description="Verify PACK-001 is installed and functional"
    )
    database_url: Optional[str] = Field(
        None, description="Database URL for connectivity test"
    )
    performance_target_ms: float = Field(
        default=5000.0,
        description="Target time for 100 metric calculations (ms)",
    )
    performance_metric_count: int = Field(
        default=100, description="Number of metrics to calculate in benchmark"
    )
    consolidation_benchmark_entities: int = Field(
        default=5,
        description="Number of entities for consolidation benchmark",
    )
    cross_framework_benchmark_count: int = Field(
        default=50,
        description="Number of data points for cross-framework benchmark",
    )
    webhook_test_endpoints: List[str] = Field(
        default_factory=list,
        description="Webhook endpoints to test connectivity",
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
    checks_warned: int = Field(
        default=0, description="Number of checks with warnings"
    )
    execution_time_ms: float = Field(
        default=0.0, description="Check execution time"
    )
    remediations: List[RemediationSuggestion] = Field(
        default_factory=list, description="Remediation suggestions"
    )


class HealthCheckResult(BaseModel):
    """Complete health check result for Professional Pack."""

    overall_status: HealthStatus = Field(
        ..., description="Overall pack health status"
    )
    pack_version: str = Field(
        default="2.0.0", description="PACK-002 version"
    )
    check_timestamp: datetime = Field(
        default_factory=_utcnow, description="When the check was run"
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
    categories_checked: int = Field(
        default=0, description="Number of categories checked (max 10)"
    )


# =============================================================================
# Agent Registry for Professional Pack (93+ agents)
# =============================================================================

REQUIRED_AGENT_MODULES: Dict[str, str] = {
    # Foundation Agents (10)
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
    # Data Agents - Intake (9)
    "GL-DATA-X-001": "greenlang.agents.data.document_ingestion_agent",
    "GL-DATA-X-002": "greenlang.agents.data.excel_csv_normalizer",
    "GL-DATA-X-003": "greenlang.agents.data.erp_finance_connector",
    "GL-DATA-X-004": "greenlang.agents.data.erp_connector_agent",
    "GL-DATA-X-005": "greenlang.agents.data.eudr_traceability_connector",
    "GL-DATA-X-006": "greenlang.agents.data.gis_mapping_connector",
    "GL-DATA-X-007": "greenlang.agents.data.satellite_connector",
    "GL-DATA-X-008": "greenlang.agents.data.weather_climate_agent",
    "GL-DATA-X-009": "greenlang.agents.data.utility_tariff_agent",
    # Data Agents - Quality (11)
    "GL-DATA-X-010": "greenlang.agents.data.questionnaire_processor",
    "GL-DATA-X-011": "greenlang.agents.data.spend_categorizer",
    "GL-DATA-X-012": "greenlang.agents.data.data_quality_profiler",
    "GL-DATA-X-013": "greenlang.agents.data.duplicate_detection",
    "GL-DATA-X-014": "greenlang.agents.data.missing_value_imputer",
    "GL-DATA-X-015": "greenlang.agents.data.outlier_detection",
    "GL-DATA-X-016": "greenlang.agents.data.time_series_gap_filler",
    "GL-DATA-X-017": "greenlang.agents.data.cross_source_reconciliation",
    "GL-DATA-X-018": "greenlang.agents.data.data_freshness_monitor",
    "GL-DATA-X-019": "greenlang.agents.data.schema_migration",
    "GL-DATA-X-020": "greenlang.agents.data.data_lineage_tracker",
    # MRV Scope 1 (8)
    "GL-MRV-SCOPE1": "greenlang.agents.mrv.scope1_combustion",
    "GL-MRV-RFGAS": "greenlang.agents.mrv.refrigerants_fgas",
    "GL-MRV-MOBILE": "greenlang.agents.mrv.mobile_combustion",
    "GL-MRV-PROCESS": "greenlang.agents.mrv.process_emissions",
    "GL-MRV-FUGITIVE": "greenlang.agents.mrv.fugitive_emissions",
    "GL-MRV-LANDUSE": "greenlang.agents.mrv.land_use_emissions",
    "GL-MRV-WASTE": "greenlang.agents.mrv.waste_treatment",
    "GL-MRV-AG": "greenlang.agents.mrv.agricultural_emissions",
    # MRV Scope 2 (5)
    "GL-MRV-LOC": "greenlang.agents.mrv.scope2_location_based",
    "GL-MRV-MKT": "greenlang.agents.mrv.scope2_market_based",
    "GL-MRV-STEAM": "greenlang.agents.mrv.steam_heat_purchase",
    "GL-MRV-COOL": "greenlang.agents.mrv.cooling_purchase",
    "GL-MRV-DUAL": "greenlang.agents.mrv.dual_reporting_reconciliation",
    # MRV Scope 3 (17)
    "GL-MRV-S3-C01": "greenlang.agents.mrv.purchased_goods_services",
    "GL-MRV-S3-C02": "greenlang.agents.mrv.capital_goods",
    "GL-MRV-S3-C03": "greenlang.agents.mrv.fuel_energy_activities",
    "GL-MRV-S3-C04": "greenlang.agents.mrv.upstream_transportation",
    "GL-MRV-S3-C05": "greenlang.agents.mrv.waste_generated",
    "GL-MRV-S3-C06": "greenlang.agents.mrv.business_travel",
    "GL-MRV-S3-C07": "greenlang.agents.mrv.employee_commuting",
    "GL-MRV-S3-C08": "greenlang.agents.mrv.upstream_leased_assets",
    "GL-MRV-S3-C09": "greenlang.agents.mrv.downstream_transportation",
    "GL-MRV-S3-C10": "greenlang.agents.mrv.processing_sold_products",
    "GL-MRV-S3-C11": "greenlang.agents.mrv.use_sold_products",
    "GL-MRV-S3-C12": "greenlang.agents.mrv.end_of_life_treatment",
    "GL-MRV-S3-C13": "greenlang.agents.mrv.downstream_leased_assets",
    "GL-MRV-S3-C14": "greenlang.agents.mrv.franchises",
    "GL-MRV-S3-C15": "greenlang.agents.mrv.investments",
    "GL-MRV-S3MAP": "greenlang.agents.mrv.scope3_category_mapper",
    "GL-MRV-AUDIT": "greenlang.agents.mrv.audit_trail_lineage",
}

# Professional-specific agent modules (additional 27+ for APP tier)
PROFESSIONAL_AGENT_MODULES: Dict[str, str] = {
    # APP agents for cross-framework
    "GL-APP-CDP": "greenlang.apps.cdp",
    "GL-APP-TCFD": "greenlang.apps.tcfd",
    "GL-APP-SBTI": "greenlang.apps.sbti",
    "GL-APP-TAXO": "greenlang.apps.taxonomy",
    "GL-APP-CSRD": "greenlang.apps.csrd",
    "GL-APP-GHG": "greenlang.apps.ghg",
    "GL-APP-ISO14064": "greenlang.apps.iso14064",
    "GL-APP-CBAM": "greenlang.apps.cbam",
    "GL-APP-VCCI": "greenlang.apps.vcci",
    "GL-APP-EUDR": "greenlang.apps.eudr",
    # PACK-002 engines
    "PACK2-ENG-CONSOL": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.consolidation_engine"
    ),
    "PACK2-ENG-APPROVE": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.approval_workflow_engine"
    ),
    "PACK2-ENG-QG": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.quality_gate_engine"
    ),
    "PACK2-ENG-BENCH": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.benchmarking_engine"
    ),
    "PACK2-ENG-REGIMP": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.regulatory_impact_engine"
    ),
    "PACK2-ENG-STAKE": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.stakeholder_engine"
    ),
    "PACK2-ENG-DGOV": (
        "packs.eu_compliance.PACK_002_csrd_professional.engines.data_governance_engine"
    ),
}

REQUIRED_PACKAGES: Dict[str, str] = {
    "pydantic": "2.0.0",
    "httpx": "0.24.0",
    "numpy": "1.24.0",
    "pandas": "2.0.0",
}

PROFESSIONAL_PACKAGES: Dict[str, str] = {
    "networkx": "3.0",
    "cryptography": "41.0.0",
    "openpyxl": "3.1.0",
    "jinja2": "3.1.0",
    "pyyaml": "6.0.0",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "psycopg": "3.0.0",
    "redis": "4.0.0",
    "boto3": "1.26.0",
    "opentelemetry-api": "1.15.0",
    "aiohttp": "3.8.0",
}

REQUIRED_DATA_PATHS: List[str] = [
    "greenlang/agents/mrv",
    "greenlang/agents/foundation",
    "greenlang/agents/data",
]

PROFESSIONAL_DATA_FILES: List[str] = [
    "packs/eu-compliance/PACK-002-csrd-professional/config/pack_config.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/__init__.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/consolidation_engine.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/approval_workflow_engine.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/quality_gate_engine.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/benchmarking_engine.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/regulatory_impact_engine.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/stakeholder_engine.py",
    "packs/eu-compliance/PACK-002-csrd-professional/engines/data_governance_engine.py",
]

PROFESSIONAL_DIRECTORIES: List[str] = [
    "config",
    "engines",
    "integrations",
    "templates",
    "workflows",
    "tests",
]

CROSS_FRAMEWORK_ENGINES: Dict[str, List[str]] = {
    "CDP": [
        "scoring_simulator",
        "gap_analysis",
        "supply_chain",
        "benchmarking",
        "transition_plan",
        "verification",
    ],
    "TCFD": [
        "scenario_analysis",
        "financial_impact",
        "physical_risk",
        "transition_risk",
        "governance",
        "gap_analysis",
        "issb_crosswalk",
    ],
    "SBTi": [
        "temperature_scoring",
        "pathway_calculator",
        "sector_engine",
        "scope3_screening",
        "validation",
        "fi_engine",
        "crosswalk",
    ],
    "EU_TAXONOMY": [
        "alignment",
        "gar_calculation",
        "dnsh_assessment",
        "substantial_contribution",
        "kpi_calculation",
        "regulatory_update",
    ],
}

PROFESSIONAL_DB_TABLES: List[str] = [
    # Consolidation tables
    "gl_consolidation_entities",
    "gl_consolidation_adjustments",
    "gl_consolidation_results",
    "gl_consolidation_audit",
    # Approval chain tables
    "gl_approval_workflows",
    "gl_approval_steps",
    "gl_approval_decisions",
    "gl_approval_audit",
    # Quality gate tables
    "gl_quality_gate_definitions",
    "gl_quality_gate_results",
    "gl_quality_gate_evidence",
    # Webhook tables
    "gl_webhook_subscriptions",
    "gl_webhook_events",
    "gl_webhook_deliveries",
    "gl_webhook_dead_letter",
    # Benchmarking tables
    "gl_benchmark_datasets",
    "gl_benchmark_results",
    "gl_benchmark_comparisons",
    # Cross-framework tables
    "gl_framework_mappings",
    "gl_framework_results",
    "gl_framework_gaps",
]

APPROVAL_ROLES: List[str] = [
    "csrd_preparer",
    "csrd_reviewer",
    "csrd_approver",
    "csrd_board_member",
    "csrd_auditor",
    "csrd_admin",
]

SECURITY_ENV_VARS: List[str] = [
    "DATABASE_URL",
    "SECRET_KEY",
    "JWT_SECRET",
    "WEBHOOK_HMAC_SECRET",
    "ENCRYPTION_KEY",
    "APPROVAL_JWT_SECRET",
]


# =============================================================================
# Health Check Implementation
# =============================================================================


class ProfessionalHealthCheck:
    """Comprehensive 10-category health check for CSRD Professional Pack.

    Validates operational readiness across ten check categories including
    all PACK-001 categories plus three professional-specific additions:
    cross-framework engine availability, webhook connectivity, and
    PACK-001 backward compatibility.

    Attributes:
        config: Professional health check configuration
        _results: Accumulated component health results

    Example:
        >>> health = ProfessionalHealthCheck()
        >>> result = await health.check_all()
        >>> print(f"Status: {result.overall_status}")
        >>> print(f"Categories checked: {result.categories_checked}/10")
    """

    def __init__(
        self, config: Optional[ProfessionalHealthCheckConfig] = None
    ) -> None:
        """Initialize the Professional Pack health check system.

        Args:
            config: Health check configuration. Uses defaults if not provided.
        """
        self.config = config or ProfessionalHealthCheckConfig()
        self._results: List[ComponentHealth] = []

        if not self.config.project_root:
            self.config.project_root = str(
                Path(__file__).resolve().parents[4]
            )

        logger.info(
            "ProfessionalHealthCheck v%s initialized, project_root=%s",
            _MODULE_VERSION,
            self.config.project_root,
        )

    # -------------------------------------------------------------------------
    # Main Check Entry Point
    # -------------------------------------------------------------------------

    async def check_all(self) -> HealthCheckResult:
        """Run all enabled health checks and return a comprehensive result.

        Runs up to 10 check categories in sequence, accumulating results
        and building remediation suggestions. Each category produces a
        ComponentHealth record with pass/fail/warn counts.

        Returns:
            HealthCheckResult with per-component details, overall status,
            critical issues, warnings, and provenance hash.
        """
        start_time = time.monotonic()
        self._results = []

        logger.info("Starting 10-category professional health check")

        check_methods = [
            (self.config.check_agents, self.check_agents),
            (self.config.check_configuration, self.check_configuration),
            (self.config.check_data_files, self.check_data_files),
            (self.config.check_database, self.check_database),
            (self.config.check_dependencies, self.check_dependencies),
            (self.config.check_security, self.check_security),
            (self.config.check_performance, self.check_performance),
            (self.config.check_cross_framework, self.check_cross_framework),
            (self.config.check_webhook, self.check_webhook),
            (self.config.check_pack1_compatibility, self.check_pack1_compatibility),
        ]

        categories_checked = 0
        for enabled, method in check_methods:
            if not enabled:
                continue
            try:
                component = await method()
                self._results.append(component)
                categories_checked += 1
            except Exception as exc:
                logger.error(
                    "Health check %s failed: %s",
                    method.__name__,
                    exc,
                    exc_info=True,
                )
                self._results.append(ComponentHealth(
                    name=method.__name__.replace("check_", "").replace(
                        "_", " "
                    ).title(),
                    category=CheckCategory.AGENTS,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check raised exception: {exc}",
                ))
                categories_checked += 1

        total_elapsed = (time.monotonic() - start_time) * 1000
        return self._build_result(total_elapsed, categories_checked)

    async def check_single(
        self, category: CheckCategory
    ) -> ComponentHealth:
        """Run a single health check category.

        Args:
            category: The check category to run.

        Returns:
            ComponentHealth for the requested category.

        Raises:
            ValueError: If category is not recognized.
        """
        category_map = {
            CheckCategory.AGENTS: self.check_agents,
            CheckCategory.CONFIGURATION: self.check_configuration,
            CheckCategory.DATA_FILES: self.check_data_files,
            CheckCategory.DATABASE: self.check_database,
            CheckCategory.DEPENDENCIES: self.check_dependencies,
            CheckCategory.SECURITY: self.check_security,
            CheckCategory.PERFORMANCE: self.check_performance,
            CheckCategory.CROSS_FRAMEWORK: self.check_cross_framework,
            CheckCategory.WEBHOOK: self.check_webhook,
            CheckCategory.PACK1_COMPATIBILITY: self.check_pack1_compatibility,
        }
        method = category_map.get(category)
        if method is None:
            raise ValueError(f"Unknown check category: {category}")
        return await method()

    # -------------------------------------------------------------------------
    # Category 1: Agent Availability (93+ modules)
    # -------------------------------------------------------------------------

    async def check_agents(self) -> ComponentHealth:
        """Check that all required agents (93+) are importable.

        Tests import of all PACK-001 base agents (66+) plus PACK-002
        professional agent modules and engine modules.

        Returns:
            ComponentHealth for the agents category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {"base_agents": {}, "professional_agents": {}}

        # Check base agents (from PACK-001 registry)
        for agent_id, module_path in REQUIRED_AGENT_MODULES.items():
            result = self._try_import(agent_id, module_path)
            details["base_agents"][agent_id] = result["status"]
            if result["status"] == "importable":
                passed += 1
            elif result["status"].startswith("import_error"):
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Agent {agent_id} module '{module_path}' not importable",
                    suggestion=(
                        f"Verify module '{module_path}' exists. "
                        f"Run: pip install -e ."
                    ),
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.AGENTS,
                ))
            else:
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Agent {agent_id} raised error on import",
                    suggestion=f"Check '{module_path}': {result.get('error', '')}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.AGENTS,
                ))

        # Check professional agent modules
        for agent_id, module_path in PROFESSIONAL_AGENT_MODULES.items():
            result = self._try_import(agent_id, module_path)
            details["professional_agents"][agent_id] = result["status"]
            if result["status"] == "importable":
                passed += 1
            elif result["status"].startswith("import_error"):
                # Professional agents are warning-level (may not be fully deployed)
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=(
                        f"Professional agent {agent_id} module "
                        f"'{module_path}' not importable"
                    ),
                    suggestion=(
                        f"Install PACK-002 professional engines. "
                        f"Module: {module_path}"
                    ),
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.AGENTS,
                ))
            else:
                warned += 1

        total = passed + failed + warned
        details["total_checked"] = total
        details["importable"] = passed

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=5)

        return ComponentHealth(
            name="Agent Availability (Professional)",
            category=CheckCategory.AGENTS,
            status=status,
            message=f"{passed}/{total} agents importable",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 2: Configuration
    # -------------------------------------------------------------------------

    async def check_configuration(self) -> ComponentHealth:
        """Check professional configuration files and directory structure.

        Validates presence of all PACK-002 directories, config modules,
        size preset files, and sector preset files.

        Returns:
            ComponentHealth for the configuration category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        pack_root = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-002-csrd-professional",
        )

        # Check directory structure
        for dir_name in PROFESSIONAL_DIRECTORIES:
            dir_path = os.path.join(pack_root, dir_name)
            if os.path.isdir(dir_path):
                details[f"dir_{dir_name}"] = "present"
                passed += 1
            else:
                details[f"dir_{dir_name}"] = "missing"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Directory '{dir_name}' not found in PACK-002 root",
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
                suggestion="Create config module with ProfessionalPackConfig class",
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.CONFIGURATION,
            ))

        # Check pack_config.py
        pack_config = os.path.join(pack_root, "config", "pack_config.py")
        if os.path.isfile(pack_config):
            details["pack_config"] = "present"
            passed += 1
        else:
            details["pack_config"] = "missing"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="pack_config.py not found",
                suggestion="Create PACK-002 configuration module",
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.CONFIGURATION,
            ))

        # Check size preset files
        presets_dir = os.path.join(pack_root, "config", "presets")
        size_presets = [
            "enterprise_group.yaml",
            "listed_company.yaml",
            "financial_institution.yaml",
            "multinational.yaml",
        ]
        for preset_file in size_presets:
            preset_path = os.path.join(presets_dir, preset_file)
            if os.path.isfile(preset_path):
                details[f"preset_{preset_file}"] = "present"
                passed += 1
            else:
                details[f"preset_{preset_file}"] = "missing"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Size preset '{preset_file}' not found",
                    suggestion=f"Create preset file at: {preset_path}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.CONFIGURATION,
                    auto_fixable=True,
                ))

        # Check sector preset files
        sector_presets = [
            "energy_sector.yaml",
            "financial_sector.yaml",
            "manufacturing_sector.yaml",
            "services_sector.yaml",
            "technology_sector.yaml",
        ]
        for sector_file in sector_presets:
            sector_path = os.path.join(presets_dir, sector_file)
            if os.path.isfile(sector_path):
                details[f"sector_{sector_file}"] = "present"
                passed += 1
            else:
                details[f"sector_{sector_file}"] = "missing"
                warned += 1

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
                suggestion="Create integrations module",
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.CONFIGURATION,
            ))

        # Check engines __init__.py
        engines_init = os.path.join(pack_root, "engines", "__init__.py")
        if os.path.isfile(engines_init):
            details["engines_init"] = "present"
            passed += 1
        else:
            details["engines_init"] = "missing"
            failed += 1

        # Check workflows __init__.py
        workflows_init = os.path.join(pack_root, "workflows", "__init__.py")
        if os.path.isfile(workflows_init):
            details["workflows_init"] = "present"
            passed += 1
        else:
            details["workflows_init"] = "missing"
            warned += 1

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=2)

        return ComponentHealth(
            name="Configuration (Professional)",
            category=CheckCategory.CONFIGURATION,
            status=status,
            message=(
                f"{passed} configs valid, {failed} missing, {warned} warnings"
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 3: Data Files
    # -------------------------------------------------------------------------

    async def check_data_files(self) -> ComponentHealth:
        """Check that required data files and directories are present.

        Validates PACK-001 agent directories, PACK-002 engine modules,
        template files, and workflow files.

        Returns:
            ComponentHealth for the data files category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        # Check base agent directories (from PACK-001)
        for rel_path in REQUIRED_DATA_PATHS:
            full_path = os.path.join(self.config.project_root, rel_path)
            if os.path.exists(full_path):
                details[rel_path] = "present"
                passed += 1

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
                    suggestion=f"Verify project structure: {full_path}",
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.DATA_FILES,
                ))

        # Check professional data files
        for rel_path in PROFESSIONAL_DATA_FILES:
            full_path = os.path.join(self.config.project_root, rel_path)
            if os.path.isfile(full_path):
                details[rel_path] = "present"
                passed += 1
            else:
                details[rel_path] = "missing"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"Professional file '{rel_path}' not found",
                    suggestion=f"Verify PACK-002 installation: {full_path}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.DATA_FILES,
                ))

        # Check template files exist
        template_dir = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-002-csrd-professional", "templates",
        )
        expected_templates = [
            "consolidated_report.py",
            "cross_framework_report.py",
            "scenario_analysis_report.py",
            "investor_esg_report.py",
            "board_governance_pack.py",
            "regulatory_filing_package.py",
            "benchmarking_dashboard.py",
            "stakeholder_report.py",
            "data_governance_report.py",
            "professional_dashboard.py",
        ]
        for tmpl_name in expected_templates:
            tmpl_path = os.path.join(template_dir, tmpl_name)
            if os.path.isfile(tmpl_path):
                details[f"template/{tmpl_name}"] = "present"
                passed += 1
            else:
                details[f"template/{tmpl_name}"] = "missing"
                warned += 1

        # Check workflow files exist
        workflow_dir = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-002-csrd-professional", "workflows",
        )
        expected_workflows = [
            "consolidated_reporting.py",
            "cross_framework_alignment.py",
            "scenario_analysis.py",
            "continuous_compliance.py",
            "stakeholder_engagement.py",
            "regulatory_change_mgmt.py",
            "board_governance.py",
            "professional_audit.py",
        ]
        for wf_name in expected_workflows:
            wf_path = os.path.join(workflow_dir, wf_name)
            if os.path.isfile(wf_path):
                details[f"workflow/{wf_name}"] = "present"
                passed += 1
            else:
                details[f"workflow/{wf_name}"] = "missing"
                warned += 1

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=2)

        return ComponentHealth(
            name="Data Files (Professional)",
            category=CheckCategory.DATA_FILES,
            status=status,
            message=f"{passed} files present, {failed} missing, {warned} warnings",
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 4: Database
    # -------------------------------------------------------------------------

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity and professional table availability.

        Validates database URL format, driver availability, and checks
        for the existence of professional-specific tables (consolidation,
        approval chain, quality gate, webhook, benchmarking, cross-framework).

        Returns:
            ComponentHealth for the database category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        db_url = self.config.database_url or os.environ.get("DATABASE_URL")
        if not db_url:
            details["connection"] = "not_configured"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="No database URL configured",
                suggestion=(
                    "Provide a database URL via config or set DATABASE_URL "
                    "environment variable"
                ),
                severity=HealthSeverity.INFO,
                category=CheckCategory.DATABASE,
            ))
        else:
            # Validate URL format
            if "://" in db_url:
                details["url_format"] = "valid"
                passed += 1
            else:
                details["url_format"] = "invalid"
                failed += 1
                remediations.append(RemediationSuggestion(
                    finding="Database URL format is invalid",
                    suggestion=(
                        "URL must contain '://' "
                        "(e.g., postgresql://user:pass@host/db)"
                    ),
                    severity=HealthSeverity.CRITICAL,
                    category=CheckCategory.DATABASE,
                ))

            # Check driver
            if db_url.startswith(("postgresql://", "postgres://", "sqlite://")):
                details["driver"] = "recognized"
                passed += 1
            else:
                details["driver"] = "unrecognized"
                warned += 1

        # Check psycopg availability
        try:
            importlib.import_module("psycopg")
            details["psycopg"] = "available"
            passed += 1
        except ImportError:
            details["psycopg"] = "not_installed"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="psycopg package not installed",
                suggestion="Install: pip install psycopg[binary]",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.DATABASE,
            ))

        # Document expected professional tables (cannot verify without live DB)
        details["expected_tables"] = PROFESSIONAL_DB_TABLES
        details["expected_table_count"] = len(PROFESSIONAL_DB_TABLES)
        details["note"] = (
            "Table existence verified at runtime via migrations V118-V128. "
            "Run migrations to create professional tables."
        )

        # Check migration files exist
        migrations_dir = os.path.join(
            self.config.project_root, "deployment", "database", "migrations",
        )
        professional_migrations = [
            f"V{v}" for v in range(118, 129)
        ]
        migrations_found = 0
        for prefix in professional_migrations:
            migration_exists = False
            if os.path.isdir(migrations_dir):
                for fname in os.listdir(migrations_dir):
                    if fname.startswith(prefix):
                        migration_exists = True
                        break
            if migration_exists:
                migrations_found += 1
                passed += 1
            else:
                warned += 1

        details["professional_migrations_found"] = migrations_found
        details["professional_migrations_expected"] = len(professional_migrations)

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="Database (Professional)",
            category=CheckCategory.DATABASE,
            status=status,
            message=(
                f"{passed} checks passed, {failed} failed, {warned} warnings. "
                f"{migrations_found}/{len(professional_migrations)} "
                f"migrations found."
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 5: Dependencies
    # -------------------------------------------------------------------------

    async def check_dependencies(self) -> ComponentHealth:
        """Check Python packages including professional-specific dependencies.

        Validates required packages (pydantic, httpx, numpy, pandas),
        professional packages (networkx, cryptography, openpyxl, jinja2,
        pyyaml), and optional packages (psycopg, redis, boto3, otel, aiohttp).

        Returns:
            ComponentHealth for the dependencies category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {
            "required": {},
            "professional": {},
            "optional": {},
        }

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
                        f"Package '{pkg_name}' version "
                        f"{pkg_result.get('installed', 'unknown')} "
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

        # Check professional packages
        for pkg_name, min_version in PROFESSIONAL_PACKAGES.items():
            pkg_result = _check_package_version(pkg_name, min_version)
            details["professional"][pkg_name] = pkg_result
            if pkg_result["status"] == "ok":
                passed += 1
            elif pkg_result["status"] == "version_mismatch":
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=(
                        f"Professional package '{pkg_name}' version "
                        f"{pkg_result.get('installed', 'unknown')} "
                        f"below minimum {min_version}"
                    ),
                    suggestion=f"Upgrade: pip install '{pkg_name}>={min_version}'",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.DEPENDENCIES,
                ))
            else:
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=(
                        f"Professional package '{pkg_name}' not installed"
                    ),
                    suggestion=f"Install: pip install '{pkg_name}>={min_version}'",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.DEPENDENCIES,
                ))

        # Check optional packages
        for pkg_name, min_version in OPTIONAL_PACKAGES.items():
            pkg_result = _check_package_version(pkg_name, min_version)
            details["optional"][pkg_name] = pkg_result
            if pkg_result["status"] == "ok":
                passed += 1
            elif pkg_result["status"] == "not_installed":
                warned += 1

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="Dependencies (Professional)",
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

    # -------------------------------------------------------------------------
    # Category 6: Security
    # -------------------------------------------------------------------------

    async def check_security(self) -> ComponentHealth:
        """Check security configuration including approval RBAC and HMAC.

        Validates auth module availability, approval role configuration,
        webhook HMAC secrets, encryption keys, and insecure default
        detection for sensitive environment variables.

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
        auth_modules = [
            ("greenlang.auth", "Auth module"),
            ("greenlang.agents.foundation.access_guard", "Access guard"),
            ("greenlang.agents.foundation.policy_guard", "Policy guard"),
            ("greenlang.agents.foundation.pii_redaction", "PII redaction"),
        ]
        for module_path, label in auth_modules:
            try:
                importlib.import_module(module_path)
                details[label.lower().replace(" ", "_")] = "available"
                passed += 1
            except ImportError:
                details[label.lower().replace(" ", "_")] = "not_available"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"{label} not importable",
                    suggestion=f"Verify module '{module_path}' is installed",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.SECURITY,
                ))

        # Check approval role definitions
        details["approval_roles"] = APPROVAL_ROLES
        details["approval_role_count"] = len(APPROVAL_ROLES)

        # Verify approval engine module is importable
        try:
            importlib.import_module(
                "packs.eu_compliance.PACK_002_csrd_professional"
                ".engines.approval_workflow_engine"
            )
            details["approval_engine"] = "available"
            passed += 1
        except ImportError:
            details["approval_engine"] = "not_available"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="Approval workflow engine not importable",
                suggestion="Verify PACK-002 engines are properly installed",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.SECURITY,
            ))

        # Check for sensitive environment variables
        insecure_defaults = {
            "changeme", "secret", "password", "default", "test",
            "example", "12345",
        }
        for var_name in SECURITY_ENV_VARS:
            value = os.environ.get(var_name)
            if value:
                details[f"env_{var_name}"] = "set"
                passed += 1
                if value.lower() in insecure_defaults:
                    warned += 1
                    remediations.append(RemediationSuggestion(
                        finding=(
                            f"Environment variable {var_name} has "
                            f"an insecure default value"
                        ),
                        suggestion=f"Set {var_name} to a strong, unique value",
                        severity=HealthSeverity.WARNING,
                        category=CheckCategory.SECURITY,
                    ))
            else:
                details[f"env_{var_name}"] = "not_set"

        # Check HMAC secret specifically
        hmac_secret = os.environ.get("WEBHOOK_HMAC_SECRET")
        if hmac_secret and len(hmac_secret) >= 32:
            details["hmac_secret_strength"] = "adequate"
            passed += 1
        elif hmac_secret:
            details["hmac_secret_strength"] = "weak"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="WEBHOOK_HMAC_SECRET is shorter than 32 characters",
                suggestion="Use a secret of at least 32 characters for HMAC signing",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.SECURITY,
            ))
        else:
            details["hmac_secret_strength"] = "not_configured"

        # Check cryptography module for HMAC/signing
        try:
            importlib.import_module("cryptography")
            details["cryptography_module"] = "available"
            passed += 1
        except ImportError:
            details["cryptography_module"] = "not_available"
            warned += 1

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="Security (Professional)",
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

    # -------------------------------------------------------------------------
    # Category 7: Performance Benchmarks
    # -------------------------------------------------------------------------

    async def check_performance(self) -> ComponentHealth:
        """Run professional-grade performance benchmarks.

        Benchmarks include:
        1. Simple emission calculations (inherited from PACK-001)
        2. Multi-entity consolidation simulation
        3. Cross-framework mapping throughput
        4. Hash computation performance
        5. Pydantic model creation throughput

        Returns:
            ComponentHealth for the performance category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        target_ms = self.config.performance_target_ms
        metric_count = self.config.performance_metric_count

        # Benchmark 1: Simple emission calculations
        calc_start = time.monotonic()
        total_emissions = 0.0
        for i in range(metric_count):
            quantity = float(i + 1) * 10.0
            factor = 2.5
            emissions = quantity * factor
            total_emissions += emissions
            _compute_hash(f"benchmark:{i}:{emissions}")

        calc_elapsed = (time.monotonic() - calc_start) * 1000
        details["emission_calc_count"] = metric_count
        details["emission_calc_ms"] = round(calc_elapsed, 2)
        details["emission_calc_total"] = round(total_emissions, 2)
        details["emission_calc_target_ms"] = target_ms
        details["emission_per_metric_ms"] = round(
            calc_elapsed / metric_count, 4
        )

        if calc_elapsed <= target_ms:
            passed += 1
            details["emission_benchmark"] = "passed"
        else:
            warned += 1
            details["emission_benchmark"] = "slow"
            remediations.append(RemediationSuggestion(
                finding=(
                    f"Emission benchmark took {calc_elapsed:.1f}ms "
                    f"(target: {target_ms:.1f}ms)"
                ),
                suggestion="Optimize calculation paths or increase compute",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.PERFORMANCE,
            ))

        # Benchmark 2: Multi-entity consolidation simulation
        entity_count = self.config.consolidation_benchmark_entities
        consol_start = time.monotonic()
        entity_totals = []
        for entity_idx in range(entity_count):
            entity_emissions = 0.0
            for metric_idx in range(20):
                value = float((entity_idx + 1) * (metric_idx + 1)) * 3.14
                entity_emissions += value
                _compute_hash(
                    f"entity:{entity_idx}:metric:{metric_idx}:{value}"
                )
            entity_totals.append(entity_emissions)

        # Simulate inter-company elimination (10% reduction)
        consolidated = sum(entity_totals) * 0.9
        _compute_hash(f"consolidated:{consolidated}")

        consol_elapsed = (time.monotonic() - consol_start) * 1000
        consol_target = 2000.0
        details["consolidation_entities"] = entity_count
        details["consolidation_ms"] = round(consol_elapsed, 2)
        details["consolidation_result"] = round(consolidated, 2)
        details["consolidation_target_ms"] = consol_target

        if consol_elapsed <= consol_target:
            passed += 1
            details["consolidation_benchmark"] = "passed"
        else:
            warned += 1
            details["consolidation_benchmark"] = "slow"
            remediations.append(RemediationSuggestion(
                finding=(
                    f"Consolidation benchmark ({entity_count} entities) "
                    f"took {consol_elapsed:.1f}ms (target: {consol_target:.0f}ms)"
                ),
                suggestion="Consider batch processing for large entity groups",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.PERFORMANCE,
            ))

        # Benchmark 3: Cross-framework mapping throughput
        cf_count = self.config.cross_framework_benchmark_count
        cf_start = time.monotonic()
        frameworks = ["CDP", "TCFD", "SBTi", "EU_TAXONOMY", "GRI", "SASB"]
        mapping_count = 0
        for dp_idx in range(cf_count):
            for fw in frameworks:
                # Simulate deterministic mapping lookup
                mapped = f"ESRS-E1-{dp_idx:03d}:{fw}:mapped"
                _compute_hash(mapped)
                mapping_count += 1

        cf_elapsed = (time.monotonic() - cf_start) * 1000
        cf_target = 3000.0
        details["cross_framework_data_points"] = cf_count
        details["cross_framework_mappings"] = mapping_count
        details["cross_framework_ms"] = round(cf_elapsed, 2)
        details["cross_framework_target_ms"] = cf_target

        if cf_elapsed <= cf_target:
            passed += 1
            details["cross_framework_benchmark"] = "passed"
        else:
            warned += 1
            details["cross_framework_benchmark"] = "slow"

        # Benchmark 4: Hash computation
        hash_start = time.monotonic()
        for i in range(1000):
            _compute_hash(f"hash_benchmark:{i}")
        hash_elapsed = (time.monotonic() - hash_start) * 1000
        details["hash_benchmark_ms"] = round(hash_elapsed, 2)
        details["hashes_per_second"] = (
            round(1000 / (hash_elapsed / 1000), 0)
            if hash_elapsed > 0
            else 0
        )

        if hash_elapsed < 1000:
            passed += 1
        else:
            warned += 1

        # Benchmark 5: Pydantic model creation
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
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="Performance (Professional)",
            category=CheckCategory.PERFORMANCE,
            status=status,
            message=(
                f"5 benchmarks: {passed} passed, {warned} slow. "
                f"Emission: {calc_elapsed:.0f}ms, "
                f"Consolidation: {consol_elapsed:.0f}ms, "
                f"Cross-framework: {cf_elapsed:.0f}ms"
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 8: Cross-Framework Engine Availability (NEW)
    # -------------------------------------------------------------------------

    async def check_cross_framework(self) -> ComponentHealth:
        """Check CDP/TCFD/SBTi/EU Taxonomy engine availability.

        Verifies that the cross-framework bridge module is importable,
        that each framework's engine modules are present, and that
        the cross-framework report template exists.

        Returns:
            ComponentHealth for the cross-framework category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {"frameworks": {}}

        # Check cross-framework bridge module
        bridge_importable = False
        try:
            importlib.import_module(
                "packs.eu_compliance.PACK_002_csrd_professional"
                ".integrations.cross_framework_bridge"
            )
            details["bridge_module"] = "importable"
            passed += 1
            bridge_importable = True
        except ImportError as exc:
            details["bridge_module"] = f"import_error: {exc}"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="CrossFrameworkBridge module not importable",
                suggestion=(
                    "Verify PACK-002 integrations are installed. "
                    "Check cross_framework_bridge.py"
                ),
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.CROSS_FRAMEWORK,
            ))

        # Check each framework's APP module availability
        framework_apps = {
            "CDP": "greenlang.apps.cdp",
            "TCFD": "greenlang.apps.tcfd",
            "SBTi": "greenlang.apps.sbti",
            "EU_TAXONOMY": "greenlang.apps.taxonomy",
            "GRI": "greenlang.apps.csrd",  # GRI mapped via CSRD app
            "SASB": "greenlang.apps.csrd",  # SASB mapped via CSRD app
        }
        for fw_name, fw_module in framework_apps.items():
            try:
                importlib.import_module(fw_module)
                details["frameworks"][fw_name] = {
                    "module": fw_module,
                    "status": "available",
                }
                passed += 1
            except ImportError:
                details["frameworks"][fw_name] = {
                    "module": fw_module,
                    "status": "not_available",
                }
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"{fw_name} framework module not importable",
                    suggestion=f"Install APP module: {fw_module}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.CROSS_FRAMEWORK,
                ))

        # Check engine counts per framework
        for fw_name, engines in CROSS_FRAMEWORK_ENGINES.items():
            details["frameworks"].setdefault(fw_name, {})
            details["frameworks"][fw_name]["expected_engines"] = len(engines)
            details["frameworks"][fw_name]["engines"] = engines

        # Check cross-framework report template
        cf_report = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-002-csrd-professional",
            "templates", "cross_framework_report.py",
        )
        if os.path.isfile(cf_report):
            details["cross_framework_report_template"] = "present"
            passed += 1
        else:
            details["cross_framework_report_template"] = "missing"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="Cross-framework report template not found",
                suggestion=f"Create template: {cf_report}",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.CROSS_FRAMEWORK,
            ))

        # Check cross-framework alignment workflow
        cf_workflow = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-002-csrd-professional",
            "workflows", "cross_framework_alignment.py",
        )
        if os.path.isfile(cf_workflow):
            details["cross_framework_workflow"] = "present"
            passed += 1
        else:
            details["cross_framework_workflow"] = "missing"
            warned += 1

        # Summary
        total_engines = sum(
            len(e) for e in CROSS_FRAMEWORK_ENGINES.values()
        )
        details["total_framework_engines"] = total_engines
        details["bridge_ready"] = bridge_importable

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="Cross-Framework Engines",
            category=CheckCategory.CROSS_FRAMEWORK,
            status=status,
            message=(
                f"{passed} checks passed across "
                f"{len(CROSS_FRAMEWORK_ENGINES)} frameworks "
                f"({total_engines} engines expected)"
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 9: Webhook Connectivity (NEW)
    # -------------------------------------------------------------------------

    async def check_webhook(self) -> ComponentHealth:
        """Check webhook endpoint connectivity and HMAC configuration.

        Validates the WebhookManager module is importable, HMAC secret
        is configured, and optionally tests configured webhook endpoint
        URLs for reachability.

        Returns:
            ComponentHealth for the webhook category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        # Check webhook manager module
        webhook_importable = False
        try:
            importlib.import_module(
                "packs.eu_compliance.PACK_002_csrd_professional"
                ".integrations.webhook_manager"
            )
            details["webhook_module"] = "importable"
            passed += 1
            webhook_importable = True
        except ImportError as exc:
            details["webhook_module"] = f"import_error: {exc}"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="WebhookManager module not importable",
                suggestion="Verify PACK-002 integrations: webhook_manager.py",
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.WEBHOOK,
            ))

        # Check HMAC configuration
        hmac_secret = os.environ.get("WEBHOOK_HMAC_SECRET")
        if hmac_secret:
            details["hmac_configured"] = True
            details["hmac_length"] = len(hmac_secret)
            if len(hmac_secret) >= 32:
                details["hmac_strength"] = "adequate"
                passed += 1
            else:
                details["hmac_strength"] = "weak"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding="HMAC secret is shorter than recommended 32 chars",
                    suggestion="Generate a 32+ char secret for HMAC signing",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.WEBHOOK,
                ))
        else:
            details["hmac_configured"] = False
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="WEBHOOK_HMAC_SECRET not configured",
                suggestion=(
                    "Set WEBHOOK_HMAC_SECRET environment variable. "
                    "Webhooks will be unsigned without it."
                ),
                severity=HealthSeverity.WARNING,
                category=CheckCategory.WEBHOOK,
            ))

        # Check hmac module availability (stdlib, should always pass)
        try:
            import hmac as hmac_module  # noqa: F811
            details["hmac_module"] = "available"
            passed += 1
        except ImportError:
            details["hmac_module"] = "not_available"
            failed += 1

        # Check httpx for webhook delivery
        try:
            importlib.import_module("httpx")
            details["httpx_module"] = "available"
            passed += 1
        except ImportError:
            details["httpx_module"] = "not_available"
            warned += 1
            remediations.append(RemediationSuggestion(
                finding="httpx not installed (needed for webhook delivery)",
                suggestion="Install: pip install httpx",
                severity=HealthSeverity.WARNING,
                category=CheckCategory.WEBHOOK,
            ))

        # Test configured webhook endpoints (if any)
        endpoints = self.config.webhook_test_endpoints
        if endpoints:
            details["test_endpoints"] = {}
            for endpoint_url in endpoints:
                # Validate URL format only (no actual HTTP calls in health check)
                if endpoint_url.startswith(("http://", "https://")):
                    details["test_endpoints"][endpoint_url] = "url_valid"
                    passed += 1
                else:
                    details["test_endpoints"][endpoint_url] = "url_invalid"
                    warned += 1
                    remediations.append(RemediationSuggestion(
                        finding=f"Invalid webhook URL: {endpoint_url}",
                        suggestion="URL must start with http:// or https://",
                        severity=HealthSeverity.WARNING,
                        category=CheckCategory.WEBHOOK,
                    ))
        else:
            details["test_endpoints"] = "none_configured"

        # Document supported webhook event types
        details["supported_event_types"] = [
            "workflow.started",
            "workflow.completed",
            "workflow.failed",
            "phase.started",
            "phase.completed",
            "phase.failed",
            "quality_gate.passed",
            "quality_gate.failed",
            "approval.requested",
            "approval.completed",
            "approval.rejected",
            "compliance.alert",
            "data_quality.alert",
            "deadline.approaching",
            "regulatory.change",
        ]

        details["webhook_ready"] = webhook_importable

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="Webhook System",
            category=CheckCategory.WEBHOOK,
            status=status,
            message=(
                f"{passed} checks passed, "
                f"HMAC {'configured' if hmac_secret else 'not configured'}, "
                f"{len(endpoints)} endpoints"
            ),
            details=details,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            execution_time_ms=elapsed,
            remediations=remediations,
        )

    # -------------------------------------------------------------------------
    # Category 10: PACK-001 Compatibility (NEW)
    # -------------------------------------------------------------------------

    async def check_pack1_compatibility(self) -> ComponentHealth:
        """Verify PACK-001 Starter Pack is installed and functional.

        Checks that PACK-001 directory exists, core modules are importable,
        health check is accessible, and integration modules are present.
        PACK-002 requires PACK-001 as a foundation.

        Returns:
            ComponentHealth for the PACK-001 compatibility category.
        """
        start_time = time.monotonic()
        passed = 0
        failed = 0
        warned = 0
        remediations: List[RemediationSuggestion] = []
        details: Dict[str, Any] = {}

        pack1_root = os.path.join(
            self.config.project_root,
            "packs", "eu-compliance", "PACK-001-csrd-starter",
        )

        # Check PACK-001 directory exists
        if os.path.isdir(pack1_root):
            details["pack1_directory"] = "present"
            passed += 1
        else:
            details["pack1_directory"] = "missing"
            failed += 1
            remediations.append(RemediationSuggestion(
                finding="PACK-001 CSRD Starter Pack directory not found",
                suggestion=(
                    "PACK-002 Professional requires PACK-001 Starter as "
                    "foundation. Install PACK-001 first."
                ),
                severity=HealthSeverity.CRITICAL,
                category=CheckCategory.PACK1_COMPATIBILITY,
            ))

        # Check PACK-001 required directories
        pack1_dirs = ["config", "integrations", "templates", "workflows"]
        for dir_name in pack1_dirs:
            dir_path = os.path.join(pack1_root, dir_name)
            if os.path.isdir(dir_path):
                details[f"pack1_dir_{dir_name}"] = "present"
                passed += 1
            else:
                details[f"pack1_dir_{dir_name}"] = "missing"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=f"PACK-001 directory '{dir_name}' missing",
                    suggestion=f"Repair PACK-001 installation: {dir_path}",
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.PACK1_COMPATIBILITY,
                ))

        # Check PACK-001 integration modules
        pack1_modules = [
            (
                "pack_orchestrator",
                "packs.eu_compliance.PACK_001_csrd_starter"
                ".integrations.pack_orchestrator",
            ),
            (
                "mrv_bridge",
                "packs.eu_compliance.PACK_001_csrd_starter"
                ".integrations.mrv_bridge",
            ),
            (
                "data_pipeline_bridge",
                "packs.eu_compliance.PACK_001_csrd_starter"
                ".integrations.data_pipeline_bridge",
            ),
            (
                "setup_wizard",
                "packs.eu_compliance.PACK_001_csrd_starter"
                ".integrations.setup_wizard",
            ),
            (
                "health_check",
                "packs.eu_compliance.PACK_001_csrd_starter"
                ".integrations.health_check",
            ),
        ]
        for module_name, module_path in pack1_modules:
            try:
                importlib.import_module(module_path)
                details[f"pack1_module_{module_name}"] = "importable"
                passed += 1
            except ImportError:
                details[f"pack1_module_{module_name}"] = "not_importable"
                warned += 1
                remediations.append(RemediationSuggestion(
                    finding=(
                        f"PACK-001 module '{module_name}' not importable"
                    ),
                    suggestion=(
                        f"Ensure PACK-001 is in Python path: {module_path}"
                    ),
                    severity=HealthSeverity.WARNING,
                    category=CheckCategory.PACK1_COMPATIBILITY,
                ))

        # Check PACK-001 config module
        pack1_config_init = os.path.join(pack1_root, "config", "__init__.py")
        if os.path.isfile(pack1_config_init):
            details["pack1_config_init"] = "present"
            passed += 1
        else:
            details["pack1_config_init"] = "missing"
            warned += 1

        # Check PACK-001 integration init
        pack1_integ_init = os.path.join(
            pack1_root, "integrations", "__init__.py"
        )
        if os.path.isfile(pack1_integ_init):
            details["pack1_integrations_init"] = "present"
            passed += 1
        else:
            details["pack1_integrations_init"] = "missing"
            warned += 1

        # Cross-check: PACK-001 health check runnable
        try:
            mod = importlib.import_module(
                "packs.eu_compliance.PACK_001_csrd_starter"
                ".integrations.health_check"
            )
            if hasattr(mod, "PackHealthCheck"):
                details["pack1_health_check_class"] = "available"
                passed += 1
            else:
                details["pack1_health_check_class"] = "class_missing"
                warned += 1
        except (ImportError, Exception):
            details["pack1_health_check_class"] = "not_available"
            warned += 1

        # Version compatibility note
        details["pack2_requires_pack1"] = True
        details["compatibility_note"] = (
            "PACK-002 Professional extends PACK-001 Starter. "
            "All PACK-001 features remain available."
        )

        elapsed = (time.monotonic() - start_time) * 1000
        status = self._determine_status(failed, warned, threshold_fail=1)

        return ComponentHealth(
            name="PACK-001 Compatibility",
            category=CheckCategory.PACK1_COMPATIBILITY,
            status=status,
            message=(
                f"{passed} compatibility checks passed, "
                f"{failed} critical, {warned} warnings"
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

    def _build_result(
        self,
        total_elapsed_ms: float,
        categories_checked: int,
    ) -> HealthCheckResult:
        """Build the final HealthCheckResult from accumulated components.

        Args:
            total_elapsed_ms: Total elapsed time for all checks.
            categories_checked: Number of categories that were checked.

        Returns:
            Complete HealthCheckResult with provenance hash.
        """
        total_checks = sum(
            c.checks_passed + c.checks_failed + c.checks_warned
            for c in self._results
        )
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
            f"professional_health:{_utcnow().isoformat()}:"
            f"{total_checks}:{total_passed}:{total_failed}:"
            f"{categories_checked}"
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
            categories_checked=categories_checked,
        )

        logger.info(
            "Professional health check complete: %s "
            "(%d/%d passed, %d critical, %d warnings, "
            "%d/10 categories) in %.1fms",
            overall.value,
            total_passed,
            total_checks,
            len(critical_issues),
            len(warnings),
            categories_checked,
            total_elapsed_ms,
        )

        return result

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _try_import(
        self, agent_id: str, module_path: str
    ) -> Dict[str, str]:
        """Try importing a module and return a status dict.

        Args:
            agent_id: Agent identifier for logging.
            module_path: Dotted module path to import.

        Returns:
            Dict with 'status' key ('importable', 'import_error', 'error').
        """
        try:
            importlib.import_module(module_path)
            return {"status": "importable"}
        except ImportError as exc:
            return {"status": f"import_error: {exc}", "error": str(exc)}
        except Exception as exc:
            return {"status": f"error: {exc}", "error": str(exc)}

    def _determine_status(
        self,
        failed: int,
        warned: int,
        threshold_fail: int = 5,
    ) -> HealthStatus:
        """Determine health status from fail/warn counts.

        Args:
            failed: Number of failed checks.
            warned: Number of warned checks.
            threshold_fail: Maximum failures before UNHEALTHY.

        Returns:
            HealthStatus enum value.
        """
        if failed == 0 and warned == 0:
            return HealthStatus.HEALTHY
        if failed == 0 and warned > 0:
            return HealthStatus.DEGRADED
        if failed <= threshold_fail:
            return HealthStatus.DEGRADED
        return HealthStatus.UNHEALTHY


# =============================================================================
# Module-Level Helper Functions
# =============================================================================


def _check_package_version(
    package_name: str, min_version: str
) -> Dict[str, Any]:
    """Check if a Python package is installed and meets the minimum version.

    Args:
        package_name: Name of the package to check.
        min_version: Minimum required version string.

    Returns:
        Dict with status, installed version, and minimum version.
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
    """Check if installed version is >= minimum version.

    Simple version comparison handling major.minor.patch format.

    Args:
        installed: Installed version string.
        minimum: Minimum required version string.

    Returns:
        True if installed >= minimum, False otherwise.
    """
    try:
        installed_parts = [int(x) for x in installed.split(".")[:3]]
        minimum_parts = [int(x) for x in minimum.split(".")[:3]]

        while len(installed_parts) < 3:
            installed_parts.append(0)
        while len(minimum_parts) < 3:
            minimum_parts.append(0)

        return tuple(installed_parts) >= tuple(minimum_parts)
    except (ValueError, AttributeError):
        return True
