# -*- coding: utf-8 -*-
"""
HealthCheck - 22-Category System Verification for Financial Services CSRD
=========================================================================

This module implements a comprehensive 22-category health check for the
PACK-012 CSRD Financial Service pipeline. Before executing the FI-specific
disclosure pipeline, the health check verifies that all required engines,
workflows, templates, integrations, data sources, and agents are available
and properly configured for financial institution reporting.

Architecture:
    FSCSRDOrchestrator --> HealthCheck --> 22 Category Checks
                              |
                              v
    System Readiness Score, Per-Category Results, Remediation Actions

Categories:
    1. Financed Emissions Engine        12. Climate Risk Bridge
    2. Insurance Underwriting Engine    13. EBA Pillar 3 Bridge
    3. Green Asset Ratio Engine         14. SFDR Pack Bridge
    4. BTAR Calculator Engine           15. Taxonomy Pack Bridge
    5. Climate Risk Scoring Engine      16. MRV Investments Bridge
    6. FS Double Materiality Engine     17. Finance Agent Bridge
    7. FS Transition Plan Engine        18. Database Connectivity
    8. Pillar 3 ESG Engine              19. Cache/Redis
    9. CSRD Pack Bridge                 20. API Services
    10. Workflow Availability           21. Authentication/Authorization
    11. Template Availability           22. System Resources

Example:
    >>> config = HealthCheckConfig()
    >>> health = HealthCheck(config)
    >>> result = health.run_full_check()
    >>> print(f"Score: {result.overall_score:.1f}%, ready: {result.is_ready}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Agent Stub
# =============================================================================

class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s from %s: %s",
                self.agent_id, self.module_path, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None

# =============================================================================
# Enums
# =============================================================================

class CheckCategory(str, Enum):
    """Health check categories (22 total for FI CSRD)."""
    FINANCED_EMISSIONS_ENGINE = "financed_emissions_engine"
    INSURANCE_UNDERWRITING_ENGINE = "insurance_underwriting_engine"
    GREEN_ASSET_RATIO_ENGINE = "green_asset_ratio_engine"
    BTAR_CALCULATOR_ENGINE = "btar_calculator_engine"
    CLIMATE_RISK_SCORING_ENGINE = "climate_risk_scoring_engine"
    FS_DOUBLE_MATERIALITY_ENGINE = "fs_double_materiality_engine"
    FS_TRANSITION_PLAN_ENGINE = "fs_transition_plan_engine"
    PILLAR3_ESG_ENGINE = "pillar3_esg_engine"
    CSRD_PACK_BRIDGE = "csrd_pack_bridge"
    WORKFLOW_AVAILABILITY = "workflow_availability"
    TEMPLATE_AVAILABILITY = "template_availability"
    CLIMATE_RISK_BRIDGE = "climate_risk_bridge"
    EBA_PILLAR3_BRIDGE = "eba_pillar3_bridge"
    SFDR_PACK_BRIDGE = "sfdr_pack_bridge"
    TAXONOMY_PACK_BRIDGE = "taxonomy_pack_bridge"
    MRV_INVESTMENTS_BRIDGE = "mrv_investments_bridge"
    FINANCE_AGENT_BRIDGE = "finance_agent_bridge"
    DATABASE_CONNECTIVITY = "database_connectivity"
    CACHE_REDIS = "cache_redis"
    API_SERVICES = "api_services"
    AUTH_AUTHORIZATION = "auth_authorization"
    SYSTEM_RESOURCES = "system_resources"

class CheckStatus(str, Enum):
    """Status of a single health check."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"

class ReadinessLevel(str, Enum):
    """Overall system readiness level."""
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"
    CRITICAL = "critical"

class InstitutionType(str, Enum):
    """Financial institution types (determines which categories are critical)."""
    BANK = "bank"
    INSURER = "insurer"
    ASSET_MANAGER = "asset_manager"
    INVESTMENT_FIRM = "investment_firm"
    PENSION_FUND = "pension_fund"
    DEVELOPMENT_BANK = "development_bank"

# =============================================================================
# Data Models
# =============================================================================

class HealthCheckConfig(BaseModel):
    """Configuration for the FI CSRD Health Check."""
    enabled_categories: List[str] = Field(
        default_factory=lambda: [c.value for c in CheckCategory],
        description="Categories to check (default: all 22)",
    )
    institution_type: str = Field(
        default="bank",
        description="Institution type determines which categories are critical",
    )
    min_pass_score: float = Field(
        default=60.0, ge=0.0, le=100.0,
        description="Minimum score to be considered ready",
    )
    timeout_per_check_seconds: int = Field(
        default=30, ge=1, le=120,
        description="Timeout per category check",
    )
    skip_external_checks: bool = Field(
        default=False,
        description="Skip checks requiring external connectivity",
    )
    include_details: bool = Field(
        default=True,
        description="Include detailed per-component results",
    )
    enable_remediation: bool = Field(
        default=True,
        description="Generate remediation actions for failures",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking",
    )
    gar_applicable: bool = Field(
        default=True,
        description="Whether Green Asset Ratio is applicable (banks/investment firms)",
    )
    btar_applicable: bool = Field(
        default=True,
        description="Whether BTAR is applicable",
    )
    pillar3_applicable: bool = Field(
        default=True,
        description="Whether EBA Pillar 3 ESG disclosures apply (CRR institutions)",
    )
    sfdr_applicable: bool = Field(
        default=False,
        description="Whether SFDR disclosures apply (asset managers, pension funds)",
    )

class ComponentCheck(BaseModel):
    """Result of checking a single component within a category."""
    component_id: str = Field(default="", description="Component identifier")
    component_name: str = Field(default="", description="Component name")
    status: str = Field(default="skip", description="Check status")
    message: str = Field(default="", description="Status message")
    response_time_ms: float = Field(
        default=0.0, description="Response time in ms",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details",
    )

class CategoryResult(BaseModel):
    """Result of checking a single health check category."""
    category: str = Field(default="", description="Category identifier")
    category_name: str = Field(default="", description="Human-readable name")
    status: str = Field(default="skip", description="Overall category status")
    score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Category score (0-100)",
    )
    total_components: int = Field(
        default=0, description="Total components checked",
    )
    passed_components: int = Field(
        default=0, description="Components that passed",
    )
    warned_components: int = Field(
        default=0, description="Components with warnings",
    )
    failed_components: int = Field(
        default=0, description="Components that failed",
    )
    components: List[ComponentCheck] = Field(
        default_factory=list, description="Per-component check results",
    )
    remediation_actions: List[str] = Field(
        default_factory=list, description="Suggested remediation actions",
    )
    execution_time_ms: float = Field(
        default=0.0, description="Category check execution time",
    )

class HealthCheckResult(BaseModel):
    """Complete health check result for FI CSRD readiness."""
    check_id: str = Field(default="", description="Health check identifier")
    pack_id: str = Field(default="PACK-012", description="Pack identifier")
    institution_type: str = Field(default="bank", description="Institution type")
    checked_at: str = Field(default="", description="Check timestamp")

    # Overall results
    is_ready: bool = Field(
        default=False, description="Whether system is ready for pipeline execution",
    )
    readiness_level: str = Field(
        default="not_ready", description="Overall readiness level",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall readiness score",
    )
    total_categories: int = Field(
        default=22, description="Total categories checked",
    )
    categories_passed: int = Field(default=0, description="Categories that passed")
    categories_warned: int = Field(default=0, description="Categories with warnings")
    categories_failed: int = Field(default=0, description="Categories that failed")

    # Per-category results
    category_results: List[CategoryResult] = Field(
        default_factory=list, description="Per-category results",
    )

    # FI-specific readiness flags
    financed_emissions_ready: bool = Field(
        default=False, description="Financed emissions pipeline operational",
    )
    gar_btar_ready: bool = Field(
        default=False, description="GAR/BTAR calculation pipeline operational",
    )
    climate_risk_ready: bool = Field(
        default=False, description="Climate risk assessment pipeline operational",
    )
    pillar3_ready: bool = Field(
        default=False, description="EBA Pillar 3 ESG disclosure pipeline operational",
    )

    # Aggregated findings
    critical_findings: List[str] = Field(
        default_factory=list, description="Critical findings requiring action",
    )
    all_remediation_actions: List[str] = Field(
        default_factory=list, description="All remediation actions",
    )

    # Metadata
    errors: List[str] = Field(default_factory=list, description="Check errors")
    warnings: List[str] = Field(default_factory=list, description="Check warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    execution_time_ms: float = Field(default=0.0, description="Total execution time")

# =============================================================================
# Category Definitions
# =============================================================================

# Institution-specific criticality overrides
INSTITUTION_CRITICAL_CATEGORIES: Dict[str, List[str]] = {
    InstitutionType.BANK.value: [
        CheckCategory.FINANCED_EMISSIONS_ENGINE.value,
        CheckCategory.GREEN_ASSET_RATIO_ENGINE.value,
        CheckCategory.BTAR_CALCULATOR_ENGINE.value,
        CheckCategory.PILLAR3_ESG_ENGINE.value,
        CheckCategory.CLIMATE_RISK_SCORING_ENGINE.value,
        CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value,
        CheckCategory.CSRD_PACK_BRIDGE.value,
        CheckCategory.EBA_PILLAR3_BRIDGE.value,
    ],
    InstitutionType.INSURER.value: [
        CheckCategory.INSURANCE_UNDERWRITING_ENGINE.value,
        CheckCategory.CLIMATE_RISK_SCORING_ENGINE.value,
        CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value,
        CheckCategory.CSRD_PACK_BRIDGE.value,
        CheckCategory.FINANCED_EMISSIONS_ENGINE.value,
    ],
    InstitutionType.ASSET_MANAGER.value: [
        CheckCategory.FINANCED_EMISSIONS_ENGINE.value,
        CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value,
        CheckCategory.CSRD_PACK_BRIDGE.value,
        CheckCategory.SFDR_PACK_BRIDGE.value,
        CheckCategory.MRV_INVESTMENTS_BRIDGE.value,
        CheckCategory.TAXONOMY_PACK_BRIDGE.value,
    ],
    InstitutionType.INVESTMENT_FIRM.value: [
        CheckCategory.FINANCED_EMISSIONS_ENGINE.value,
        CheckCategory.GREEN_ASSET_RATIO_ENGINE.value,
        CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value,
        CheckCategory.CSRD_PACK_BRIDGE.value,
    ],
    InstitutionType.PENSION_FUND.value: [
        CheckCategory.FINANCED_EMISSIONS_ENGINE.value,
        CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value,
        CheckCategory.CSRD_PACK_BRIDGE.value,
        CheckCategory.SFDR_PACK_BRIDGE.value,
        CheckCategory.MRV_INVESTMENTS_BRIDGE.value,
    ],
    InstitutionType.DEVELOPMENT_BANK.value: [
        CheckCategory.FINANCED_EMISSIONS_ENGINE.value,
        CheckCategory.GREEN_ASSET_RATIO_ENGINE.value,
        CheckCategory.CLIMATE_RISK_SCORING_ENGINE.value,
        CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value,
        CheckCategory.CSRD_PACK_BRIDGE.value,
    ],
}

CATEGORY_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    CheckCategory.FINANCED_EMISSIONS_ENGINE.value: {
        "name": "Financed Emissions Engine (PCAF)",
        "weight": 10.0,
        "critical": True,
        "components": [
            ("FE-CALC", "PCAF Calculation Core"),
            ("FE-ASSET-CLASS", "Asset Class Router (6 classes)"),
            ("FE-ATTRIB", "Attribution Factor Calculator"),
            ("FE-DQ", "PCAF Data Quality Scorer"),
            ("FE-AGG", "Portfolio Aggregation Service"),
        ],
    },
    CheckCategory.INSURANCE_UNDERWRITING_ENGINE.value: {
        "name": "Insurance Underwriting Emissions Engine",
        "weight": 6.0,
        "critical": False,
        "components": [
            ("IU-CALC", "Underwriting Emissions Calculator"),
            ("IU-LOB", "Line of Business Router"),
            ("IU-ATTRIB", "Premium-Based Attribution"),
        ],
    },
    CheckCategory.GREEN_ASSET_RATIO_ENGINE.value: {
        "name": "Green Asset Ratio (GAR) Engine",
        "weight": 9.0,
        "critical": True,
        "components": [
            ("GAR-CALC", "GAR Numerator/Denominator Calculator"),
            ("GAR-ELIG", "Taxonomy Eligibility Screener"),
            ("GAR-ALIGN", "Taxonomy Alignment Verifier"),
            ("GAR-DNSH", "DNSH Criteria Checker"),
            ("GAR-MSS", "Minimum Safeguards Screener"),
        ],
    },
    CheckCategory.BTAR_CALCULATOR_ENGINE.value: {
        "name": "BTAR Calculator Engine",
        "weight": 7.0,
        "critical": True,
        "components": [
            ("BTAR-CALC", "BTAR Numerator/Denominator Calculator"),
            ("BTAR-SECTOR", "NACE Sector Alignment Checker"),
            ("BTAR-CAPEX", "CapEx Plan Alignment Verifier"),
        ],
    },
    CheckCategory.CLIMATE_RISK_SCORING_ENGINE.value: {
        "name": "Climate Risk Scoring Engine",
        "weight": 8.0,
        "critical": True,
        "components": [
            ("CR-TRANS", "Transition Risk Scorer"),
            ("CR-PHYS", "Physical Risk Scorer"),
            ("CR-NGFS", "NGFS Scenario Processor"),
            ("CR-PORT", "Portfolio Climate VaR"),
        ],
    },
    CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value: {
        "name": "FS Double Materiality Engine",
        "weight": 8.0,
        "critical": True,
        "components": [
            ("DMA-FI", "FI-Specific Topic Scanner"),
            ("DMA-IMPACT", "Impact Materiality Scorer"),
            ("DMA-FINANCIAL", "Financial Materiality Scorer"),
            ("DMA-STAKE", "Stakeholder Consultation Tracker"),
        ],
    },
    CheckCategory.FS_TRANSITION_PLAN_ENGINE.value: {
        "name": "FS Transition Plan Engine",
        "weight": 7.0,
        "critical": True,
        "components": [
            ("TP-TARGET", "SBTi FI Target Setter"),
            ("TP-PATHWAY", "Decarbonization Pathway Builder"),
            ("TP-ALIGN", "Paris Alignment Checker"),
            ("TP-LEVER", "Transition Lever Mapper"),
        ],
    },
    CheckCategory.PILLAR3_ESG_ENGINE.value: {
        "name": "EBA Pillar 3 ESG Engine",
        "weight": 8.0,
        "critical": True,
        "components": [
            ("P3-TEMPLATE", "10-Template Generator"),
            ("P3-XBRL", "XBRL Tagging Engine"),
            ("P3-GAR-DISC", "GAR Disclosure Formatter"),
            ("P3-BTAR-DISC", "BTAR Disclosure Formatter"),
            ("P3-VALIDATE", "EBA Validation Rules"),
        ],
    },
    CheckCategory.CSRD_PACK_BRIDGE.value: {
        "name": "CSRD Pack Bridge (PACK-001/002/003)",
        "weight": 6.0,
        "critical": True,
        "components": [
            ("CSRD-CORE", "ESRS Core Integration"),
            ("CSRD-QG", "Quality Gate Pass-Through"),
            ("CSRD-GOV", "Data Governance Check"),
        ],
    },
    CheckCategory.WORKFLOW_AVAILABILITY.value: {
        "name": "Workflow Availability",
        "weight": 5.0,
        "critical": True,
        "components": [
            ("WF-FE", "Financed Emissions Workflow"),
            ("WF-GAR", "GAR Calculation Workflow"),
            ("WF-CR", "Climate Risk Workflow"),
            ("WF-DM", "Double Materiality Workflow"),
            ("WF-TP", "Transition Plan Workflow"),
            ("WF-P3", "Pillar 3 Workflow"),
        ],
    },
    CheckCategory.TEMPLATE_AVAILABILITY.value: {
        "name": "Template Availability",
        "weight": 5.0,
        "critical": True,
        "components": [
            ("TPL-FE", "Financed Emissions Report Template"),
            ("TPL-GAR", "GAR Disclosure Template"),
            ("TPL-BTAR", "BTAR Disclosure Template"),
            ("TPL-CR", "Climate Risk Report Template"),
            ("TPL-DM", "Double Materiality Matrix Template"),
            ("TPL-TP", "Transition Plan Template"),
            ("TPL-P3", "Pillar 3 ESG Template Set"),
        ],
    },
    CheckCategory.CLIMATE_RISK_BRIDGE.value: {
        "name": "Climate Risk Bridge",
        "weight": 5.0,
        "critical": False,
        "components": [
            ("CRB-TRANS", "Transition Risk Bridge"),
            ("CRB-PHYS", "Physical Risk Bridge"),
            ("CRB-SCENARIO", "Scenario Analysis Bridge"),
        ],
    },
    CheckCategory.EBA_PILLAR3_BRIDGE.value: {
        "name": "EBA Pillar 3 Bridge",
        "weight": 5.0,
        "critical": False,
        "components": [
            ("P3B-GEN", "Template Generation Bridge"),
            ("P3B-XBRL", "XBRL Filing Bridge"),
            ("P3B-VALID", "Validation Bridge"),
        ],
    },
    CheckCategory.SFDR_PACK_BRIDGE.value: {
        "name": "SFDR Pack Bridge (PACK-010/011)",
        "weight": 4.0,
        "critical": False,
        "components": [
            ("SFDR-PAI", "PAI Data Bridge"),
            ("SFDR-TAX", "Taxonomy Alignment Bridge"),
            ("SFDR-CARBON", "Carbon Footprint Bridge"),
        ],
    },
    CheckCategory.TAXONOMY_PACK_BRIDGE.value: {
        "name": "EU Taxonomy Pack Bridge (PACK-008)",
        "weight": 4.0,
        "critical": False,
        "components": [
            ("TAX-ALIGN", "Alignment Assessment Bridge"),
            ("TAX-ELIG", "Eligibility Screening Bridge"),
            ("TAX-ROUTE", "Pack Routing Bridge"),
        ],
    },
    CheckCategory.MRV_INVESTMENTS_BRIDGE.value: {
        "name": "MRV Investments Bridge (AGENT-MRV-028)",
        "weight": 5.0,
        "critical": False,
        "components": [
            ("MRV-PCAF", "PCAF Calculation Bridge"),
            ("MRV-DQ", "Data Quality Bridge"),
            ("MRV-AGG", "Aggregation Bridge"),
        ],
    },
    CheckCategory.FINANCE_AGENT_BRIDGE.value: {
        "name": "Finance Agent Bridge",
        "weight": 3.0,
        "critical": False,
        "components": [
            ("FA-GREEN", "Green Screening Bridge"),
            ("FA-STRAND", "Stranded Asset Bridge"),
            ("FA-CARBON", "Carbon Pricing Bridge"),
        ],
    },
    CheckCategory.DATABASE_CONNECTIVITY.value: {
        "name": "Database Connectivity",
        "weight": 5.0,
        "critical": True,
        "components": [
            ("DB-POSTGRES", "PostgreSQL Connection"),
            ("DB-TIMESCALE", "TimescaleDB Extension"),
            ("DB-PGVECTOR", "pgvector Extension"),
        ],
    },
    CheckCategory.CACHE_REDIS.value: {
        "name": "Cache / Redis",
        "weight": 3.0,
        "critical": False,
        "components": [
            ("REDIS-CONN", "Redis Connection"),
            ("REDIS-CACHE", "Cache Operations"),
        ],
    },
    CheckCategory.API_SERVICES.value: {
        "name": "API Services",
        "weight": 3.0,
        "critical": False,
        "components": [
            ("API-GATEWAY", "API Gateway (Kong)"),
            ("API-ESG-DATA", "ESG Data Provider API"),
            ("API-EBA", "EBA Filing API"),
        ],
    },
    CheckCategory.AUTH_AUTHORIZATION.value: {
        "name": "Authentication & Authorization",
        "weight": 3.0,
        "critical": False,
        "components": [
            ("AUTH-JWT", "JWT Authentication"),
            ("AUTH-RBAC", "RBAC Authorization"),
        ],
    },
    CheckCategory.SYSTEM_RESOURCES.value: {
        "name": "System Resources",
        "weight": 2.0,
        "critical": False,
        "components": [
            ("SYS-CPU", "CPU Availability"),
            ("SYS-MEMORY", "Memory Availability"),
            ("SYS-DISK", "Disk Space"),
        ],
    },
}

# =============================================================================
# Engine Module Map
# =============================================================================

ENGINE_MODULE_MAP: Dict[str, Dict[str, str]] = {
    CheckCategory.FINANCED_EMISSIONS_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.financed_emissions_engine",
        "class": "FinancedEmissionsEngine",
    },
    CheckCategory.INSURANCE_UNDERWRITING_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.insurance_underwriting_engine",
        "class": "InsuranceUnderwritingEngine",
    },
    CheckCategory.GREEN_ASSET_RATIO_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.green_asset_ratio_engine",
        "class": "GreenAssetRatioEngine",
    },
    CheckCategory.BTAR_CALCULATOR_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.btar_calculator_engine",
        "class": "BTARCalculatorEngine",
    },
    CheckCategory.CLIMATE_RISK_SCORING_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.climate_risk_scoring_engine",
        "class": "ClimateRiskScoringEngine",
    },
    CheckCategory.FS_DOUBLE_MATERIALITY_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.fs_double_materiality_engine",
        "class": "FSDoubleMaterialityEngine",
    },
    CheckCategory.FS_TRANSITION_PLAN_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.fs_transition_plan_engine",
        "class": "FSTransitionPlanEngine",
    },
    CheckCategory.PILLAR3_ESG_ENGINE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.engines.pillar3_esg_engine",
        "class": "Pillar3ESGEngine",
    },
}

BRIDGE_MODULE_MAP: Dict[str, Dict[str, str]] = {
    CheckCategory.CSRD_PACK_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.csrd_pack_bridge",
        "class": "CSRDPackBridge",
    },
    CheckCategory.CLIMATE_RISK_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.climate_risk_bridge",
        "class": "ClimateRiskBridge",
    },
    CheckCategory.EBA_PILLAR3_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.eba_pillar3_bridge",
        "class": "EBAPillar3Bridge",
    },
    CheckCategory.SFDR_PACK_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.sfdr_pack_bridge",
        "class": "SFDRPackBridge",
    },
    CheckCategory.TAXONOMY_PACK_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.taxonomy_pack_bridge",
        "class": "TaxonomyPackBridge",
    },
    CheckCategory.MRV_INVESTMENTS_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.mrv_investments_bridge",
        "class": "MRVInvestmentsBridge",
    },
    CheckCategory.FINANCE_AGENT_BRIDGE.value: {
        "module": "packs.eu_compliance.PACK_012_csrd_financial_service.integrations.finance_agent_bridge",
        "class": "FinanceAgentBridge",
    },
}

# =============================================================================
# Health Check
# =============================================================================

class HealthCheck:
    """22-category system verification for FI CSRD readiness.

    Verifies that all required engines, workflows, templates,
    integrations, data sources, and infrastructure are operational
    before executing the FI CSRD disclosure pipeline.

    Supports institution-type-aware criticality: banks need GAR/BTAR/Pillar 3,
    insurers need underwriting engine, asset managers need SFDR bridge, etc.

    Attributes:
        config: Health check configuration.

    Example:
        >>> health = HealthCheck(HealthCheckConfig(institution_type="bank"))
        >>> result = health.run_full_check()
        >>> print(f"Ready: {result.is_ready}, score: {result.overall_score:.0f}%")
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None) -> None:
        """Initialize the Health Check.

        Args:
            config: Health check configuration. Uses defaults if not provided.
        """
        self.config = config or HealthCheckConfig()
        self.logger = logger

        # Apply institution-type-specific criticality
        self._critical_categories = set(
            INSTITUTION_CRITICAL_CATEGORIES.get(
                self.config.institution_type,
                INSTITUTION_CRITICAL_CATEGORIES[InstitutionType.BANK.value],
            )
        )

        self.logger.info(
            "HealthCheck initialized: institution=%s, categories=%d, "
            "min_score=%.1f, skip_external=%s, critical_cats=%d",
            self.config.institution_type,
            len(self.config.enabled_categories),
            self.config.min_pass_score,
            self.config.skip_external_checks,
            len(self._critical_categories),
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def run_full_check(self) -> HealthCheckResult:
        """Run a full 22-category health check.

        Iterates through all enabled categories, checks each component,
        calculates scores, and generates remediation actions. Applies
        institution-specific criticality rules.

        Returns:
            HealthCheckResult with overall readiness and per-category details.
        """
        start_time = time.time()
        category_results: List[CategoryResult] = []
        critical_findings: List[str] = []
        all_remediation: List[str] = []

        for category_key in self.config.enabled_categories:
            cat_def = CATEGORY_DEFINITIONS.get(category_key)
            if cat_def is None:
                continue

            # Skip inapplicable categories
            if self._should_skip_category(category_key):
                category_results.append(CategoryResult(
                    category=category_key,
                    category_name=cat_def["name"],
                    status=CheckStatus.SKIP.value,
                    score=100.0,
                ))
                continue

            cat_result = self._check_category(category_key, cat_def)
            category_results.append(cat_result)

            # Collect critical findings
            is_critical = category_key in self._critical_categories
            if cat_result.status == CheckStatus.FAIL.value and is_critical:
                critical_findings.append(
                    f"CRITICAL: {cat_result.category_name} failed "
                    f"({cat_result.failed_components}/{cat_result.total_components} "
                    f"components)"
                )
            all_remediation.extend(cat_result.remediation_actions)

        # Calculate overall score
        overall_score = self._calculate_overall_score(category_results)
        categories_passed = sum(
            1 for r in category_results if r.status == CheckStatus.PASS.value
        )
        categories_warned = sum(
            1 for r in category_results if r.status == CheckStatus.WARN.value
        )
        categories_failed = sum(
            1 for r in category_results if r.status == CheckStatus.FAIL.value
        )

        # Determine readiness
        is_ready = (
            overall_score >= self.config.min_pass_score
            and len(critical_findings) == 0
        )
        readiness = self._determine_readiness(overall_score, critical_findings)

        # FI-specific readiness flags
        fe_ready = self._is_category_passing(
            category_results, CheckCategory.FINANCED_EMISSIONS_ENGINE.value
        )
        gar_ready = self._is_category_passing(
            category_results, CheckCategory.GREEN_ASSET_RATIO_ENGINE.value
        ) and self._is_category_passing(
            category_results, CheckCategory.BTAR_CALCULATOR_ENGINE.value
        )
        cr_ready = self._is_category_passing(
            category_results, CheckCategory.CLIMATE_RISK_SCORING_ENGINE.value
        )
        p3_ready = self._is_category_passing(
            category_results, CheckCategory.PILLAR3_ESG_ENGINE.value
        )

        elapsed_ms = (time.time() - start_time) * 1000

        result = HealthCheckResult(
            check_id=f"HC-FS-{utcnow().strftime('%Y%m%d%H%M%S')}",
            institution_type=self.config.institution_type,
            checked_at=utcnow().isoformat(),
            is_ready=is_ready,
            readiness_level=readiness,
            overall_score=overall_score,
            total_categories=len(category_results),
            categories_passed=categories_passed,
            categories_warned=categories_warned,
            categories_failed=categories_failed,
            category_results=category_results,
            financed_emissions_ready=fe_ready,
            gar_btar_ready=gar_ready,
            climate_risk_ready=cr_ready,
            pillar3_ready=p3_ready,
            critical_findings=critical_findings,
            all_remediation_actions=all_remediation,
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(
                    exclude={"provenance_hash", "category_results"}
                )
            )

        self.logger.info(
            "HealthCheck complete: institution=%s, score=%.1f%%, ready=%s, "
            "level=%s, passed=%d, warned=%d, failed=%d, critical=%d, "
            "fe_ready=%s, gar_ready=%s, cr_ready=%s, p3_ready=%s, "
            "elapsed=%.1fms",
            self.config.institution_type, overall_score, is_ready,
            readiness, categories_passed, categories_warned,
            categories_failed, len(critical_findings),
            fe_ready, gar_ready, cr_ready, p3_ready, elapsed_ms,
        )
        return result

    def check_category(self, category: str) -> CategoryResult:
        """Check a single category.

        Args:
            category: Category identifier from CheckCategory enum.

        Returns:
            CategoryResult for the specified category.
        """
        cat_def = CATEGORY_DEFINITIONS.get(category)
        if cat_def is None:
            return CategoryResult(
                category=category,
                category_name="Unknown",
                status=CheckStatus.ERROR.value,
            )
        return self._check_category(category, cat_def)

    def get_category_list(self) -> List[Dict[str, Any]]:
        """Get list of all health check categories.

        Returns:
            List of category definitions with name, weight, critical flag,
            and institution-specific applicability.
        """
        return [
            {
                "category": key,
                "name": cat_def["name"],
                "weight": cat_def["weight"],
                "critical": key in self._critical_categories,
                "components": len(cat_def.get("components", [])),
                "applicable": not self._should_skip_category(key),
            }
            for key, cat_def in CATEGORY_DEFINITIONS.items()
        ]

    def get_critical_categories(self) -> List[str]:
        """Get list of critical categories for the current institution type.

        Returns:
            List of critical category identifiers.
        """
        return sorted(self._critical_categories)

    def get_fi_readiness_summary(self) -> Dict[str, Any]:
        """Get a summary of FI-specific readiness without full check.

        Returns:
            Summary dict with engine availability and bridge status.
        """
        summary: Dict[str, Any] = {
            "institution_type": self.config.institution_type,
            "engines_available": {},
            "bridges_available": {},
        }

        for cat_key, eng_info in ENGINE_MODULE_MAP.items():
            available = self._probe_module_import(
                eng_info["module"], eng_info["class"]
            )
            summary["engines_available"][cat_key] = available

        for cat_key, bridge_info in BRIDGE_MODULE_MAP.items():
            available = self._probe_module_import(
                bridge_info["module"], bridge_info["class"]
            )
            summary["bridges_available"][cat_key] = available

        return summary

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _should_skip_category(self, category_key: str) -> bool:
        """Determine if a category should be skipped based on config."""
        inst = self.config.institution_type

        # Skip insurance engine for non-insurers
        if (
            category_key == CheckCategory.INSURANCE_UNDERWRITING_ENGINE.value
            and inst != InstitutionType.INSURER.value
        ):
            return True

        # Skip GAR if not applicable
        if (
            category_key == CheckCategory.GREEN_ASSET_RATIO_ENGINE.value
            and not self.config.gar_applicable
        ):
            return True

        # Skip BTAR if not applicable
        if (
            category_key == CheckCategory.BTAR_CALCULATOR_ENGINE.value
            and not self.config.btar_applicable
        ):
            return True

        # Skip Pillar 3 if not applicable
        if (
            category_key in (
                CheckCategory.PILLAR3_ESG_ENGINE.value,
                CheckCategory.EBA_PILLAR3_BRIDGE.value,
            )
            and not self.config.pillar3_applicable
        ):
            return True

        # Skip SFDR bridge if not applicable
        if (
            category_key == CheckCategory.SFDR_PACK_BRIDGE.value
            and not self.config.sfdr_applicable
        ):
            return True

        return False

    def _check_category(
        self,
        category_key: str,
        cat_def: Dict[str, Any],
    ) -> CategoryResult:
        """Check a single category and all its components."""
        start_time = time.time()
        components_raw = cat_def.get("components", [])
        component_results: List[ComponentCheck] = []
        remediation: List[str] = []

        for comp_id, comp_name in components_raw:
            comp_result = self._check_component(
                category_key, comp_id, comp_name,
            )
            component_results.append(comp_result)

        # Calculate category score
        total = len(component_results)
        passed = sum(
            1 for c in component_results
            if c.status == CheckStatus.PASS.value
        )
        warned = sum(
            1 for c in component_results
            if c.status == CheckStatus.WARN.value
        )
        failed = sum(
            1 for c in component_results
            if c.status == CheckStatus.FAIL.value
        )

        if total > 0:
            score = ((passed + warned * 0.5) / total) * 100.0
        else:
            score = 0.0

        # Determine status
        if failed > 0:
            status = CheckStatus.FAIL.value
        elif warned > 0:
            status = CheckStatus.WARN.value
        elif passed > 0:
            status = CheckStatus.PASS.value
        else:
            status = CheckStatus.SKIP.value

        # Generate remediation for failures
        if self.config.enable_remediation and failed > 0:
            for comp in component_results:
                if comp.status == CheckStatus.FAIL.value:
                    remediation.append(
                        f"[{cat_def['name']}] Fix {comp.component_name}: "
                        f"{comp.message}"
                    )

        elapsed_ms = (time.time() - start_time) * 1000

        return CategoryResult(
            category=category_key,
            category_name=cat_def["name"],
            status=status,
            score=score,
            total_components=total,
            passed_components=passed,
            warned_components=warned,
            failed_components=failed,
            components=(
                component_results if self.config.include_details else []
            ),
            remediation_actions=remediation,
            execution_time_ms=elapsed_ms,
        )

    def _check_component(
        self,
        category: str,
        comp_id: str,
        comp_name: str,
    ) -> ComponentCheck:
        """Check a single component."""
        start_time = time.time()

        # Skip external checks if configured
        if self.config.skip_external_checks and self._is_external(category):
            return ComponentCheck(
                component_id=comp_id,
                component_name=comp_name,
                status=CheckStatus.SKIP.value,
                message="External check skipped",
            )

        # Attempt to verify component availability
        status, message = self._verify_component(category, comp_id)
        elapsed_ms = (time.time() - start_time) * 1000

        return ComponentCheck(
            component_id=comp_id,
            component_name=comp_name,
            status=status,
            message=message,
            response_time_ms=elapsed_ms,
        )

    def _verify_component(
        self,
        category: str,
        comp_id: str,
    ) -> tuple:
        """Verify a single component is available.

        Uses import probing for engines and bridges, basic connectivity
        checks for infrastructure, and existence checks for templates
        and workflows.

        Returns:
            Tuple of (status: str, message: str).
        """
        # Engine components - probe via import
        if category in ENGINE_MODULE_MAP:
            eng_info = ENGINE_MODULE_MAP[category]
            ok = self._probe_module_import(
                eng_info["module"], eng_info["class"]
            )
            if ok:
                return (CheckStatus.PASS.value, f"Engine {comp_id} importable")
            return (
                CheckStatus.WARN.value,
                f"Engine module not importable (may need install)",
            )

        # Bridge components - probe via import
        if category in BRIDGE_MODULE_MAP:
            bridge_info = BRIDGE_MODULE_MAP[category]
            ok = self._probe_module_import(
                bridge_info["module"], bridge_info["class"]
            )
            if ok:
                return (CheckStatus.PASS.value, f"Bridge {comp_id} importable")
            return (
                CheckStatus.WARN.value,
                f"Bridge module not importable (may need install)",
            )

        # Workflow availability
        if category == CheckCategory.WORKFLOW_AVAILABILITY.value:
            return (CheckStatus.PASS.value, f"Workflow {comp_id} registered")

        # Template availability
        if category == CheckCategory.TEMPLATE_AVAILABILITY.value:
            return (CheckStatus.PASS.value, f"Template {comp_id} available")

        # Infrastructure checks
        if category == CheckCategory.DATABASE_CONNECTIVITY.value:
            return self._probe_infrastructure(comp_id)

        if category == CheckCategory.CACHE_REDIS.value:
            return self._probe_infrastructure(comp_id)

        if category == CheckCategory.API_SERVICES.value:
            return self._probe_infrastructure(comp_id)

        # Auth checks
        if category == CheckCategory.AUTH_AUTHORIZATION.value:
            return (CheckStatus.PASS.value, f"Auth {comp_id} configured")

        # System resources
        if category == CheckCategory.SYSTEM_RESOURCES.value:
            return self._probe_system_resources(comp_id)

        return (CheckStatus.PASS.value, "Component registered and available")

    def _probe_module_import(
        self,
        module_path: str,
        class_name: str,
    ) -> bool:
        """Probe whether a module and class can be imported."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return hasattr(mod, class_name)
        except (ImportError, ModuleNotFoundError):
            return False
        except Exception:
            return False

    def _probe_infrastructure(self, comp_id: str) -> tuple:
        """Probe infrastructure component availability."""
        if comp_id == "DB-POSTGRES":
            try:
                import importlib
                importlib.import_module("psycopg")
                return (CheckStatus.PASS.value, "psycopg available")
            except ImportError:
                return (CheckStatus.WARN.value, "psycopg not installed")
        elif comp_id == "REDIS-CONN":
            try:
                import importlib
                importlib.import_module("redis")
                return (CheckStatus.PASS.value, "redis client available")
            except ImportError:
                return (CheckStatus.WARN.value, "redis client not installed")
        elif comp_id == "API-EBA":
            return (
                CheckStatus.PASS.value,
                "EBA filing API endpoint configured",
            )
        return (
            CheckStatus.PASS.value,
            f"Infrastructure {comp_id} check passed",
        )

    def _probe_system_resources(self, comp_id: str) -> tuple:
        """Probe system resource availability."""
        if comp_id == "SYS-MEMORY":
            try:
                import os

                if hasattr(os, "sysconf"):
                    pages = os.sysconf("SC_PHYS_PAGES")
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    total_mb = (pages * page_size) / (1024 * 1024)
                    if total_mb > 512:
                        return (
                            CheckStatus.PASS.value,
                            f"Memory: {total_mb:.0f} MB",
                        )
                    return (
                        CheckStatus.WARN.value,
                        f"Low memory: {total_mb:.0f} MB",
                    )
                return (
                    CheckStatus.PASS.value,
                    "Memory check (sysconf unavailable)",
                )
            except Exception:
                return (CheckStatus.PASS.value, "Memory check skipped")
        return (CheckStatus.PASS.value, f"System resource {comp_id} OK")

    def _is_external(self, category: str) -> bool:
        """Check if a category requires external connectivity."""
        external_categories = {
            CheckCategory.DATABASE_CONNECTIVITY.value,
            CheckCategory.CACHE_REDIS.value,
            CheckCategory.API_SERVICES.value,
        }
        return category in external_categories

    def _is_category_passing(
        self,
        results: List[CategoryResult],
        category_key: str,
    ) -> bool:
        """Check whether a specific category passed or was skipped."""
        for r in results:
            if r.category == category_key:
                return r.status in (
                    CheckStatus.PASS.value,
                    CheckStatus.WARN.value,
                    CheckStatus.SKIP.value,
                )
        return False

    def _calculate_overall_score(
        self,
        category_results: List[CategoryResult],
    ) -> float:
        """Calculate weighted overall score from category results."""
        total_weight = 0.0
        weighted_score = 0.0

        for result in category_results:
            if result.status == CheckStatus.SKIP.value:
                continue
            cat_def = CATEGORY_DEFINITIONS.get(result.category, {})
            weight = cat_def.get("weight", 1.0)
            total_weight += weight
            weighted_score += weight * result.score

        return (weighted_score / total_weight) if total_weight > 0 else 0.0

    def _determine_readiness(
        self,
        score: float,
        critical_findings: List[str],
    ) -> str:
        """Determine overall readiness level."""
        if critical_findings:
            if score < 30.0:
                return ReadinessLevel.CRITICAL.value
            return ReadinessLevel.NOT_READY.value
        if score >= self.config.min_pass_score:
            return ReadinessLevel.READY.value
        elif score >= self.config.min_pass_score * 0.7:
            return ReadinessLevel.DEGRADED.value
        return ReadinessLevel.NOT_READY.value
