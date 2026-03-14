"""
PACK-003 CSRD Enterprise Pack - Configuration Manager

This module implements the PackConfig class that loads, merges, and validates
all configuration for the CSRD Enterprise Pack. It extends PACK-002's
configuration system with enterprise-grade features including multi-tenant
isolation, SSO/SAML/SCIM, white-label branding, AI/ML predictive analytics,
IoT sensor integration, carbon credit management, supply chain ESG scoring,
regulatory filing automation, API management with GraphQL, narrative
generation, custom workflow builder, and plugin marketplace.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest (PACK-001 + PACK-002 inherited + PACK-003 additions)
    2. Size preset (global_enterprise / saas_platform / financial_enterprise / consulting_firm)
    3. Sector preset (banking / oil_gas / automotive / pharma / conglomerate)
    4. Environment overrides (CSRD_ENT_* environment variables)
    5. Explicit runtime overrides

Example:
    >>> config = PackConfig.load(
    ...     size_preset="global_enterprise",
    ...     sector_preset="conglomerate",
    ... )
    >>> print(config.pack.metadata.display_name)
    'CSRD Enterprise Pack'
    >>> print(len(config.active_agents))
    135
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enum Types
# =============================================================================


class TenantIsolationLevel(str, Enum):
    """Tenant isolation levels for multi-tenant deployments."""

    SHARED = "SHARED"
    NAMESPACE = "NAMESPACE"
    CLUSTER = "CLUSTER"
    PHYSICAL = "PHYSICAL"


class NarrativeTone(str, Enum):
    """Tone presets for narrative generation."""

    BOARD = "board"
    INVESTOR = "investor"
    REGULATORY = "regulatory"
    PUBLIC = "public"


class IoTProtocol(str, Enum):
    """Supported IoT communication protocols."""

    MQTT = "MQTT"
    HTTP = "HTTP"
    OPCUA = "OPCUA"
    MODBUS = "MODBUS"


class CarbonRegistry(str, Enum):
    """Supported carbon credit registries."""

    VCS = "VCS"
    GOLD_STANDARD = "GoldStandard"
    ACR = "ACR"
    CAR = "CAR"
    CDM = "CDM"
    ARTICLE6 = "Article6"


class FilingTarget(str, Enum):
    """Regulatory filing target registries."""

    ESAP = "ESAP"
    NATIONAL = "national_registries"


class ValidationStrictness(str, Enum):
    """Filing validation strictness levels."""

    STRICT = "strict"
    STANDARD = "standard"
    RELAXED = "relaxed"


class WorkflowStepType(str, Enum):
    """Allowed step types in custom workflows."""

    AGENT = "agent"
    APPROVAL = "approval"
    CONDITION = "condition"
    TIMER = "timer"
    NOTIFICATION = "notification"
    DATA_TRANSFORM = "data_transform"
    QUALITY_GATE = "quality_gate"
    EXTERNAL_API = "external_api"


# =============================================================================
# Pydantic Models - Core data structures (inherited from PACK-001/002)
# =============================================================================


class ComplianceReference(BaseModel):
    """Reference to a regulatory compliance standard."""

    id: str = Field(..., description="Short identifier for the regulation")
    name: str = Field(..., description="Full name of the regulation")
    regulation: str = Field(..., description="Official regulation number")
    effective_date: str = Field(..., description="Date regulation became effective")
    description: str = Field("", description="Brief description of the regulation")


class PackMetadata(BaseModel):
    """Pack manifest metadata."""

    name: str = Field(..., description="Pack identifier slug")
    version: str = Field(..., description="Semantic version string")
    display_name: str = Field(..., description="Human-readable pack name")
    description: str = Field("", description="Pack description")
    category: str = Field(..., description="Pack category (e.g., eu-compliance)")
    tier: str = Field("enterprise", description="Pack tier (starter, professional, enterprise)")
    author: str = Field("", description="Pack author or team")
    license: str = Field("Proprietary", description="License type")
    min_platform_version: str = Field("2.0.0", description="Minimum GreenLang platform version")
    release_date: str = Field("", description="Release date ISO string")
    support_tier: str = Field("enterprise-premium", description="Support tier level")
    documentation_url: str = Field("", description="URL to pack documentation")
    changelog_url: str = Field("", description="URL to changelog")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    compliance_references: List[ComplianceReference] = Field(
        default_factory=list, description="Regulatory compliance references"
    )


class AgentComponentConfig(BaseModel):
    """Configuration for a single agent component."""

    id: str = Field(..., description="Agent identifier (e.g., AGENT-MRV-001)")
    name: str = Field("", description="Human-readable agent name")
    description: str = Field("", description="Agent description")
    required: bool = Field(True, description="Whether this agent is required")
    enabled: bool = Field(True, description="Whether this agent is enabled")
    version: str = Field("", description="Agent version override")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration overrides"
    )


class ComponentsConfig(BaseModel):
    """All components included in the pack, including inherited and enterprise."""

    # Inherited from PACK-001
    apps: List[AgentComponentConfig] = Field(default_factory=list)
    data_agents: List[AgentComponentConfig] = Field(default_factory=list)
    quality_agents: List[AgentComponentConfig] = Field(default_factory=list)
    mrv_scope1: List[AgentComponentConfig] = Field(default_factory=list)
    mrv_scope2: List[AgentComponentConfig] = Field(default_factory=list)
    mrv_scope3: List[AgentComponentConfig] = Field(default_factory=list)
    foundation: List[AgentComponentConfig] = Field(default_factory=list)

    # PACK-002 Professional additions
    professional_apps: List[AgentComponentConfig] = Field(default_factory=list)
    cdp_engines: List[AgentComponentConfig] = Field(default_factory=list)
    tcfd_engines: List[AgentComponentConfig] = Field(default_factory=list)
    sbti_engines: List[AgentComponentConfig] = Field(default_factory=list)
    taxonomy_engines: List[AgentComponentConfig] = Field(default_factory=list)
    professional_engines: List[AgentComponentConfig] = Field(default_factory=list)

    # PACK-003 Enterprise additions
    predictive_engines: List[AgentComponentConfig] = Field(default_factory=list)
    narrative_engines: List[AgentComponentConfig] = Field(default_factory=list)
    iot_engines: List[AgentComponentConfig] = Field(default_factory=list)
    carbon_credit_engines: List[AgentComponentConfig] = Field(default_factory=list)
    supply_chain_engines: List[AgentComponentConfig] = Field(default_factory=list)
    filing_engines: List[AgentComponentConfig] = Field(default_factory=list)
    workflow_builder_engines: List[AgentComponentConfig] = Field(default_factory=list)
    enterprise_apps: List[AgentComponentConfig] = Field(default_factory=list)
    enterprise_data_agents: List[AgentComponentConfig] = Field(default_factory=list)
    enterprise_quality_agents: List[AgentComponentConfig] = Field(default_factory=list)
    eudr_agents: List[AgentComponentConfig] = Field(default_factory=list)

    def _all_groups(self) -> List[List[AgentComponentConfig]]:
        """Return all agent groups as a list."""
        return [
            self.apps,
            self.data_agents,
            self.quality_agents,
            self.mrv_scope1,
            self.mrv_scope2,
            self.mrv_scope3,
            self.foundation,
            self.professional_apps,
            self.cdp_engines,
            self.tcfd_engines,
            self.sbti_engines,
            self.taxonomy_engines,
            self.professional_engines,
            self.predictive_engines,
            self.narrative_engines,
            self.iot_engines,
            self.carbon_credit_engines,
            self.supply_chain_engines,
            self.filing_engines,
            self.workflow_builder_engines,
            self.enterprise_apps,
            self.enterprise_data_agents,
            self.enterprise_quality_agents,
            self.eudr_agents,
        ]

    def get_all_agent_ids(self) -> List[str]:
        """Return all agent IDs across all component groups."""
        ids: List[str] = []
        for group in self._all_groups():
            ids.extend(agent.id for agent in group)
        return ids

    def get_enabled_agent_ids(self) -> List[str]:
        """Return only enabled agent IDs across all component groups."""
        ids: List[str] = []
        for group in self._all_groups():
            ids.extend(agent.id for agent in group if agent.enabled)
        return ids

    def get_required_agent_ids(self) -> List[str]:
        """Return only required agent IDs across all component groups."""
        ids: List[str] = []
        for group in self._all_groups():
            ids.extend(agent.id for agent in group if agent.required)
        return ids

    def get_enterprise_agent_ids(self) -> List[str]:
        """Return agent IDs exclusive to PACK-003 (enterprise tier)."""
        ids: List[str] = []
        for group in [
            self.predictive_engines,
            self.narrative_engines,
            self.iot_engines,
            self.carbon_credit_engines,
            self.supply_chain_engines,
            self.filing_engines,
            self.workflow_builder_engines,
            self.enterprise_apps,
            self.enterprise_data_agents,
            self.enterprise_quality_agents,
            self.eudr_agents,
        ]:
            ids.extend(agent.id for agent in group)
        return ids

    def get_professional_agent_ids(self) -> List[str]:
        """Return agent IDs exclusive to PACK-002 (professional tier)."""
        ids: List[str] = []
        for group in [
            self.professional_apps,
            self.cdp_engines,
            self.tcfd_engines,
            self.sbti_engines,
            self.taxonomy_engines,
            self.professional_engines,
        ]:
            ids.extend(agent.id for agent in group)
        return ids

    def find_agent(self, agent_id: str) -> Optional[AgentComponentConfig]:
        """Find an agent by its ID across all groups."""
        for group in self._all_groups():
            for agent in group:
                if agent.id == agent_id:
                    return agent
        return None


class WorkflowPhaseConfig(BaseModel):
    """Configuration for a single phase within a workflow."""

    name: str = Field(..., description="Phase identifier")
    description: str = Field("", description="Phase description")
    agents: List[str] = Field(default_factory=list, description="Agent IDs in this phase")
    duration_days: int = Field(1, ge=0, description="Estimated duration in days")


class WorkflowConfig(BaseModel):
    """Configuration for a workflow orchestration."""

    display_name: str = Field(..., description="Human-readable workflow name")
    description: str = Field("", description="Workflow description")
    schedule: str = Field(
        "on_demand",
        description="Schedule type: annual, semi-annual, quarterly, monthly, continuous, on_demand",
    )
    estimated_duration_days: int = Field(1, ge=0, description="Total estimated duration")
    phases: List[WorkflowPhaseConfig] = Field(
        default_factory=list, description="Ordered list of workflow phases"
    )
    enabled: bool = Field(True, description="Whether this workflow is enabled")

    def get_all_agent_ids(self) -> List[str]:
        """Return all unique agent IDs used across all phases."""
        ids: set[str] = set()
        for phase in self.phases:
            ids.update(phase.agents)
        return sorted(ids)


class TemplateConfig(BaseModel):
    """Configuration for a report template."""

    id: str = Field(..., description="Template identifier")
    display_name: str = Field(..., description="Human-readable template name")
    description: str = Field("", description="Template description")
    format: str = Field("pdf", description="Output format (pdf, xhtml, html, zip)")
    template_file: str = Field(..., description="Path to template file")
    xbrl_taxonomy: str = Field("", description="XBRL taxonomy version if applicable")
    enabled: bool = Field(True, description="Whether this template is enabled")


class PerformanceTargets(BaseModel):
    """Performance targets for the enterprise pack."""

    data_ingestion_rps: int = Field(50000, description="Records per second for data ingestion")
    data_ingestion_max_latency_ms: int = Field(80, description="Max latency for ingestion")
    iot_events_per_second: int = Field(10000, description="IoT events per second")
    ghg_single_scope_max_seconds: int = Field(30, description="Max seconds for single scope calc")
    ghg_full_inventory_max_seconds: int = Field(300, description="Max seconds for full inventory")
    ghg_multi_entity_max_minutes: int = Field(30, description="Max minutes for multi-entity calc")
    ghg_max_data_points: int = Field(500000, description="Max data points for enterprise scale")
    report_esrs_max_seconds: int = Field(120, description="Max seconds for ESRS report")
    report_summary_max_seconds: int = Field(60, description="Max seconds for executive summary")
    report_xbrl_max_seconds: int = Field(180, description="Max seconds for XBRL tagging")
    report_consolidated_max_seconds: int = Field(300, description="Max seconds for consolidated")
    report_cross_framework_max_seconds: int = Field(120, description="Max seconds cross-framework")
    report_narrative_max_seconds: int = Field(180, description="Max seconds for narrative gen")
    report_multi_language_max_seconds: int = Field(600, description="Max seconds multi-lang")
    forecast_max_seconds: int = Field(60, description="Max seconds for emission forecast")
    anomaly_detection_max_seconds: int = Field(30, description="Max seconds for anomaly detection")
    drift_check_max_seconds: int = Field(15, description="Max seconds for drift check")
    model_retraining_max_minutes: int = Field(30, description="Max minutes for model retrain")
    scenario_single_max_seconds: int = Field(60, description="Max seconds for single scenario")
    scenario_full_max_minutes: int = Field(15, description="Max minutes for full analysis")
    scenario_monte_carlo_iterations: int = Field(50000, description="Monte Carlo iterations")
    api_p50_ms: int = Field(40, description="REST API p50 latency target")
    api_p95_ms: int = Field(200, description="REST API p95 latency target")
    api_p99_ms: int = Field(800, description="REST API p99 latency target")
    graphql_p50_ms: int = Field(60, description="GraphQL p50 latency target")
    graphql_p95_ms: int = Field(300, description="GraphQL p95 latency target")
    quality_gate_max_seconds: int = Field(30, description="Max seconds for quality gate check")
    availability_percent: float = Field(99.99, description="Target uptime percentage")
    rpo_minutes: int = Field(5, description="Recovery point objective in minutes")
    rto_minutes: int = Field(15, description="Recovery time objective in minutes")


class RequirementsConfig(BaseModel):
    """System requirements for the enterprise pack."""

    python_version: str = Field(">=3.11", description="Minimum Python version")
    postgresql_version: str = Field(">=16", description="Minimum PostgreSQL version")
    redis_version: str = Field(">=7", description="Minimum Redis version")
    timescaledb_version: str = Field(">=2.13", description="Minimum TimescaleDB version")
    min_cpu_cores: int = Field(16, description="Minimum CPU cores")
    min_memory_gb: int = Field(64, description="Minimum memory in GB")
    min_storage_gb: int = Field(1000, description="Minimum storage in GB")
    recommended_cpu_cores: int = Field(32, description="Recommended CPU cores")
    recommended_memory_gb: int = Field(128, description="Recommended memory in GB")
    recommended_storage_gb: int = Field(5000, description="Recommended storage in GB")
    database_extensions: List[str] = Field(
        default_factory=lambda: ["pgvector", "timescaledb", "pg_partman"],
        description="Required database extensions",
    )
    min_db_connections: int = Field(100, description="Minimum database connections")
    recommended_db_connections: int = Field(300, description="Recommended database connections")


# =============================================================================
# Pydantic Models - PACK-002 Professional Configuration (inherited)
# =============================================================================


class EntityDefinition(BaseModel):
    """Definition of a subsidiary entity in a corporate group."""

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Legal entity name")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    ownership_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Ownership percentage (0-100)"
    )
    consolidation_method: str = Field(
        "full", description="Consolidation method: full, proportional, equity"
    )
    parent_entity_id: Optional[str] = Field(
        None, description="Parent entity ID for hierarchical structures"
    )
    nace_codes: List[str] = Field(
        default_factory=list, description="NACE activity codes for this entity"
    )
    active: bool = Field(True, description="Whether entity is active for reporting")
    reporting_currency: str = Field("EUR", description="Local reporting currency ISO code")
    fiscal_year_end: str = Field("12-31", description="Fiscal year end date (MM-DD)")

    @field_validator("consolidation_method")
    @classmethod
    def validate_consolidation_method(cls, v: str) -> str:
        """Validate consolidation method is recognized."""
        valid = {"full", "proportional", "equity"}
        if v not in valid:
            raise ValueError(
                f"Invalid consolidation method '{v}'. Must be one of: {sorted(valid)}"
            )
        return v


class ConsolidationConfig(BaseModel):
    """Multi-entity consolidation configuration for corporate groups."""

    enabled: bool = Field(False, description="Whether multi-entity consolidation is active")
    max_subsidiaries: int = Field(500, ge=1, le=2000, description="Maximum subsidiaries supported")
    consolidation_approaches: List[str] = Field(
        default_factory=lambda: [
            "operational_control",
            "financial_control",
            "equity_share",
        ],
        description="Supported consolidation approaches",
    )
    default_approach: str = Field(
        "operational_control", description="Default consolidation approach"
    )
    intercompany_elimination: bool = Field(
        True, description="Eliminate intercompany transactions"
    )
    minority_interest_adjustment: bool = Field(
        True, description="Adjust for minority interests"
    )
    entities: List[EntityDefinition] = Field(
        default_factory=list, description="Subsidiary entity definitions"
    )
    parallel_entity_processing: bool = Field(
        True, description="Process entities in parallel"
    )
    entity_data_timeout_seconds: int = Field(
        600, description="Timeout for individual entity data collection"
    )

    @field_validator("default_approach")
    @classmethod
    def validate_default_approach(cls, v: str) -> str:
        """Validate default consolidation approach."""
        valid = {"operational_control", "financial_control", "equity_share"}
        if v not in valid:
            raise ValueError(
                f"Invalid consolidation approach '{v}'. Must be one of: {sorted(valid)}"
            )
        return v


class ApprovalLevel(BaseModel):
    """Configuration for a single approval level."""

    role: str = Field(..., description="Approval role")
    required: bool = Field(True, description="Whether this approval level is required")
    auto_approve_threshold: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Quality score threshold for auto-approval (0-100)",
    )
    escalation_timeout_hours: int = Field(
        48, ge=1, description="Hours before escalation"
    )
    delegation_enabled: bool = Field(True, description="Allow delegation")
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email"],
        description="Notification channels",
    )


class ApprovalConfig(BaseModel):
    """Approval workflow configuration with multi-level approval chain."""

    enabled: bool = Field(True, description="Whether approval workflows are active")
    levels: List[ApprovalLevel] = Field(
        default_factory=lambda: [
            ApprovalLevel(
                role="preparer", required=True, auto_approve_threshold=None,
                escalation_timeout_hours=72,
            ),
            ApprovalLevel(
                role="reviewer", required=True, auto_approve_threshold=95.0,
                escalation_timeout_hours=48,
            ),
            ApprovalLevel(
                role="approver", required=True, auto_approve_threshold=None,
                escalation_timeout_hours=48,
            ),
            ApprovalLevel(
                role="board", required=True, auto_approve_threshold=None,
                escalation_timeout_hours=168,
            ),
        ],
        description="Ordered approval levels",
    )
    require_comments_on_rejection: bool = Field(True, description="Require rejection comments")
    audit_trail_enabled: bool = Field(True, description="Track approval actions")
    parallel_approval_allowed: bool = Field(False, description="Allow parallel approvers")


class QualityGate(BaseModel):
    """Configuration for a single quality gate."""

    id: str = Field(..., description="Quality gate identifier")
    name: str = Field(..., description="Human-readable gate name")
    enabled: bool = Field(True, description="Whether this gate is active")
    threshold: float = Field(
        ..., ge=0.0, le=100.0, description="Minimum score to pass (0-100)"
    )
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Weight in overall score")
    blocking: bool = Field(True, description="Whether failing blocks the pipeline")
    checks: List[str] = Field(default_factory=list, description="Check IDs in this gate")


class QualityGateConfig(BaseModel):
    """Quality gate configuration with weighted scoring."""

    enabled: bool = Field(True, description="Whether quality gates are active")
    gates: List[QualityGate] = Field(
        default_factory=lambda: [
            QualityGate(
                id="data_completeness", name="Data Completeness Gate",
                enabled=True, threshold=90.0, weight=3.0, blocking=True,
                checks=[
                    "mandatory_fields_present", "reference_data_linked",
                    "emission_factors_resolved", "time_series_complete",
                    "entity_coverage_complete",
                ],
            ),
            QualityGate(
                id="calculation_integrity", name="Calculation Integrity Gate",
                enabled=True, threshold=95.0, weight=5.0, blocking=True,
                checks=[
                    "scope_totals_reconcile", "consolidation_balances",
                    "year_on_year_variance_explained", "unit_conversion_verified",
                    "emission_factor_sources_valid", "provenance_hashes_verified",
                ],
            ),
            QualityGate(
                id="compliance_readiness", name="Compliance Readiness Gate",
                enabled=True, threshold=85.0, weight=4.0, blocking=False,
                checks=[
                    "esrs_disclosures_complete", "xbrl_tags_valid",
                    "materiality_assessment_documented", "audit_trail_complete",
                    "cross_framework_alignment_verified",
                ],
            ),
            QualityGate(
                id="ml_model_integrity", name="ML Model Integrity Gate",
                enabled=True, threshold=90.0, weight=3.0, blocking=False,
                checks=[
                    "forecast_accuracy_above_threshold",
                    "model_drift_within_bounds",
                    "feature_importance_stable",
                    "prediction_confidence_adequate",
                ],
            ),
        ],
        description="Quality gates with weighted scoring",
    )
    overall_pass_threshold: float = Field(
        90.0, ge=0.0, le=100.0, description="Weighted average threshold for overall pass"
    )
    run_on_commit: bool = Field(True, description="Run on every data commit")
    run_on_calculation: bool = Field(True, description="Run after every calculation")


class CrossFrameworkConfig(BaseModel):
    """Cross-framework alignment configuration."""

    enabled: bool = Field(True, description="Whether cross-framework alignment is active")
    enabled_frameworks: Dict[str, bool] = Field(
        default_factory=lambda: {
            "cdp": True, "tcfd": True, "sbti": True, "taxonomy": True,
            "gri": True, "sasb": True, "issb": True, "tnfd": True,
        },
        description="Frameworks enabled for alignment mapping",
    )
    mapping_version: str = Field("2024.1", description="Cross-framework mapping version")
    auto_populate_from_esrs: bool = Field(True, description="Auto-populate from ESRS")
    gap_analysis_enabled: bool = Field(True, description="Enable gap analysis")
    reconciliation_checks: bool = Field(True, description="Run reconciliation checks")
    primary_framework: str = Field("esrs", description="Primary framework")


class ScenarioDefinition(BaseModel):
    """Definition of a single climate scenario."""

    id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Human-readable scenario name")
    source: str = Field(..., description="Scenario source (IEA, NGFS, custom)")
    temperature_outcome_c: Optional[float] = Field(
        None, description="Temperature outcome in Celsius"
    )
    time_horizons: List[int] = Field(
        default_factory=lambda: [2030, 2040, 2050],
        description="Analysis time horizons",
    )
    enabled: bool = Field(True, description="Whether this scenario is enabled")


class ScenarioConfig(BaseModel):
    """Climate scenario analysis configuration."""

    enabled: bool = Field(True, description="Whether scenario analysis is active")
    scenarios: List[ScenarioDefinition] = Field(
        default_factory=lambda: [
            ScenarioDefinition(
                id="iea_nze", name="IEA Net Zero by 2050",
                source="IEA", temperature_outcome_c=1.5,
            ),
            ScenarioDefinition(
                id="iea_aps", name="IEA Announced Pledges",
                source="IEA", temperature_outcome_c=1.7,
            ),
            ScenarioDefinition(
                id="iea_steps", name="IEA Stated Policies",
                source="IEA", temperature_outcome_c=2.5,
            ),
            ScenarioDefinition(
                id="ngfs_orderly", name="NGFS Orderly Transition",
                source="NGFS", temperature_outcome_c=1.5,
            ),
            ScenarioDefinition(
                id="ngfs_disorderly", name="NGFS Disorderly Transition",
                source="NGFS", temperature_outcome_c=1.5,
            ),
            ScenarioDefinition(
                id="ngfs_hot_house", name="NGFS Hot House World",
                source="NGFS", temperature_outcome_c=3.0,
            ),
            ScenarioDefinition(
                id="ngfs_too_little", name="NGFS Too Little Too Late",
                source="NGFS", temperature_outcome_c=2.5,
            ),
            ScenarioDefinition(
                id="custom_bau", name="Custom Business-as-Usual",
                source="custom", temperature_outcome_c=3.5, enabled=False,
            ),
        ],
        description="Climate scenarios for analysis",
    )
    monte_carlo_enabled: bool = Field(True, description="Enable Monte Carlo simulation")
    monte_carlo_iterations: int = Field(
        50000, ge=100, le=500000, description="Monte Carlo iterations"
    )
    monte_carlo_confidence_level: float = Field(
        0.95, ge=0.80, le=0.99, description="Monte Carlo confidence level"
    )
    physical_risk_enabled: bool = Field(True, description="Include physical risk")
    transition_risk_enabled: bool = Field(True, description="Include transition risk")
    financial_impact_quantification: bool = Field(True, description="Quantify financial impacts")
    time_horizons: List[int] = Field(
        default_factory=lambda: [2030, 2040, 2050], description="Default time horizons"
    )


class BenchmarkingConfig(BaseModel):
    """Peer benchmarking and ESG rating comparison configuration."""

    enabled: bool = Field(True, description="Whether benchmarking is active")
    peer_comparison_enabled: bool = Field(True, description="Compare against sector peers")
    peer_group_source: str = Field("nace_code", description="Peer group source")
    max_peer_count: int = Field(50, ge=5, le=200, description="Maximum peers")
    esg_rating_frameworks: List[str] = Field(
        default_factory=lambda: [
            "msci", "sustainalytics", "iss_esg", "cdp", "sp_global",
        ],
        description="ESG rating frameworks",
    )
    percentile_targets: Dict[str, int] = Field(
        default_factory=lambda: {
            "ghg_intensity": 25, "renewable_energy": 75,
            "water_intensity": 25, "waste_diversion": 75,
            "safety_rate": 25,
        },
        description="Target percentile rankings",
    )
    update_frequency: str = Field("monthly", description="Benchmark refresh frequency")
    cross_tenant_benchmarking: bool = Field(
        False, description="Enable anonymized cross-tenant benchmarking"
    )


class StakeholderConfig(BaseModel):
    """Stakeholder engagement configuration."""

    enabled: bool = Field(True, description="Whether stakeholder engagement is active")
    stakeholder_categories: List[str] = Field(
        default_factory=lambda: [
            "investors", "employees", "customers", "suppliers",
            "communities", "regulators", "ngos", "media",
            "board_members", "auditors",
        ],
        description="Stakeholder categories",
    )
    survey_enabled: bool = Field(True, description="Enable surveys")
    survey_languages: List[str] = Field(
        default_factory=lambda: ["en"], description="Survey languages"
    )
    response_target_per_category: int = Field(100, ge=10, description="Target responses")
    materiality_weight_in_scoring: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight in materiality scoring"
    )
    anonymous_responses_allowed: bool = Field(True, description="Allow anonymous responses")


class RegulatoryConfig(BaseModel):
    """Regulatory change monitoring configuration."""

    enabled: bool = Field(True, description="Whether regulatory monitoring is active")
    monitored_jurisdictions: List[str] = Field(
        default_factory=lambda: [
            "EU", "DE", "FR", "NL", "ES", "IT", "BE", "AT", "SE", "DK",
            "FI", "IE", "PT", "PL", "CZ", "HU", "RO", "BG", "HR", "SK",
            "SI", "LT", "LV", "EE", "CY", "MT", "LU", "GR",
        ],
        description="Monitored jurisdictions",
    )
    monitored_regulations: List[str] = Field(
        default_factory=lambda: [
            "CSRD", "ESRS", "EU_Taxonomy", "SFDR", "CBAM",
            "CSDDD", "ESEF", "EUDR", "ETS",
        ],
        description="Monitored regulations",
    )
    scan_frequency: str = Field("daily", description="Scan frequency")
    impact_assessment_auto: bool = Field(True, description="Auto-assess impact")
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "slack", "teams", "dashboard", "webhook"],
        description="Notification channels",
    )
    filing_calendars: Dict[str, str] = Field(
        default_factory=lambda: {
            "EU": "Annual, April 30",
            "DE": "Annual, April 30",
            "FR": "Annual, April 30",
            "NL": "Annual, April 30",
        },
        description="Filing calendars per jurisdiction",
    )


class DataGovernanceConfig(BaseModel):
    """Data governance configuration for enterprise deployments."""

    enabled: bool = Field(True, description="Whether data governance is active")
    data_retention_years: int = Field(10, ge=1, le=25, description="Data retention period")
    data_classification_enabled: bool = Field(True, description="Enable data classification")
    classification_levels: List[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential", "restricted"],
        description="Data classification levels",
    )
    gdpr_compliance: bool = Field(True, description="Enable GDPR compliance")
    gdpr_data_subject_access: bool = Field(True, description="Enable DSAR handling")
    gdpr_right_to_erasure: bool = Field(True, description="Enable right to erasure")
    data_lineage_tracking: bool = Field(True, description="Track data lineage")
    version_control_enabled: bool = Field(True, description="Enable version control")
    change_log_retention_years: int = Field(15, ge=1, description="Change log retention")


class WebhookConfig(BaseModel):
    """Webhook notification configuration."""

    enabled: bool = Field(True, description="Whether webhooks are active")
    endpoints: List[str] = Field(default_factory=list, description="Webhook endpoint URLs")
    channels: List[str] = Field(
        default_factory=lambda: [
            "approval", "quality_gate", "regulatory_change",
            "scenario_complete", "filing_status", "anomaly_detected",
            "tenant_event", "iot_alert",
        ],
        description="Event channels",
    )
    hmac_secret: str = Field("", description="HMAC secret for payload signing")
    retry_count: int = Field(5, ge=0, le=10, description="Retry count")
    retry_delay_seconds: int = Field(30, ge=1, description="Retry delay")
    timeout_seconds: int = Field(10, ge=1, le=60, description="Request timeout")


class AssuranceConfig(BaseModel):
    """External assurance engagement configuration."""

    enabled: bool = Field(True, description="Whether assurance is active")
    assurance_level: str = Field("reasonable", description="Assurance level")
    isae_standard: str = Field("ISAE_3000", description="Applicable ISAE standard")
    assurance_scope: List[str] = Field(
        default_factory=lambda: [
            "scope_1_emissions", "scope_2_emissions",
            "scope_3_material_categories", "esrs_e1_disclosures",
            "esrs_e2_disclosures", "esrs_s1_disclosures",
            "eu_taxonomy_kpis", "ml_model_validation",
            "iot_data_integrity",
        ],
        description="Disclosures in scope for assurance",
    )
    evidence_format: str = Field("structured", description="Evidence format")
    auditor_access_portal: bool = Field(True, description="Enable auditor portal")
    evidence_retention_years: int = Field(10, ge=5, description="Evidence retention")
    reasonable_assurance_roadmap_year: Optional[int] = Field(
        None, description="Target year for reasonable assurance"
    )

    @field_validator("assurance_level")
    @classmethod
    def validate_assurance_level(cls, v: str) -> str:
        """Validate assurance level."""
        valid = {"limited", "reasonable"}
        if v not in valid:
            raise ValueError(
                f"Invalid assurance level '{v}'. Must be one of: {sorted(valid)}"
            )
        return v


class IntensityMetricsConfig(BaseModel):
    """Intensity metrics configuration."""

    enabled: bool = Field(True, description="Whether intensity metrics are active")
    metric_types: List[str] = Field(
        default_factory=lambda: ["per_revenue", "per_employee", "per_unit"],
        description="Intensity metric types",
    )
    per_revenue_currency: str = Field("EUR", description="Currency for per-revenue")
    per_revenue_unit: str = Field("million", description="Revenue unit")
    per_unit_definitions: Dict[str, str] = Field(
        default_factory=lambda: {
            "manufacturing": "per_tonne_product",
            "energy": "per_mwh_generated",
            "financial": "per_million_aum",
            "technology": "per_server_rack",
            "transport": "per_tonne_km",
        },
        description="Sector-specific per-unit definitions",
    )
    baseline_year: Optional[int] = Field(None, description="Baseline year")
    reduction_targets: Dict[str, float] = Field(
        default_factory=dict, description="Intensity reduction targets"
    )


# =============================================================================
# Pydantic Models - PACK-003 Enterprise Configuration Models
# =============================================================================


class ResourceQuotas(BaseModel):
    """Resource quotas for a tenant."""

    max_agents: int = Field(200, ge=1, description="Maximum agents per tenant")
    max_storage_gb: int = Field(500, ge=1, description="Maximum storage per tenant in GB")
    max_api_calls_per_day: int = Field(1000000, ge=1000, description="Max daily API calls")
    max_users: int = Field(500, ge=1, description="Maximum users per tenant")


class MultiTenantConfig(BaseModel):
    """Multi-tenant isolation configuration."""

    enabled: bool = Field(False, description="Whether multi-tenant mode is active")
    isolation_level: TenantIsolationLevel = Field(
        TenantIsolationLevel.NAMESPACE,
        description="Tenant isolation level: SHARED, NAMESPACE, CLUSTER, PHYSICAL",
    )
    max_tenants: int = Field(100, ge=1, le=10000, description="Maximum tenants supported")
    resource_quotas: ResourceQuotas = Field(
        default_factory=ResourceQuotas, description="Default resource quotas per tenant"
    )
    cross_tenant_benchmarking: bool = Field(
        False,
        description="Enable anonymized cross-tenant benchmarking",
    )
    tenant_provisioning_timeout_seconds: int = Field(
        300, ge=30, description="Timeout for tenant provisioning"
    )
    data_residency_enforcement: bool = Field(
        True, description="Enforce data residency per tenant"
    )

    @field_validator("isolation_level", mode="before")
    @classmethod
    def parse_isolation_level(cls, v: Any) -> TenantIsolationLevel:
        """Parse isolation level from string or enum."""
        if isinstance(v, str):
            return TenantIsolationLevel(v.upper())
        return v


class SSOConfig(BaseModel):
    """Single Sign-On configuration."""

    saml_enabled: bool = Field(False, description="Enable SAML 2.0 SSO")
    oauth_enabled: bool = Field(False, description="Enable OAuth 2.0 / OIDC")
    scim_enabled: bool = Field(False, description="Enable SCIM user provisioning")
    idp_metadata_url: str = Field("", description="Identity Provider metadata URL")
    default_role: str = Field("viewer", description="Default role for SSO users")
    jit_provisioning: bool = Field(
        True, description="Just-in-time user provisioning on first SSO login"
    )
    allowed_domains: List[str] = Field(
        default_factory=list, description="Allowed email domains for SSO"
    )
    session_timeout_minutes: int = Field(
        480, ge=15, description="SSO session timeout in minutes"
    )
    mfa_required: bool = Field(False, description="Require MFA for all SSO users")


class WhiteLabelConfig(BaseModel):
    """White-label branding configuration."""

    enabled: bool = Field(False, description="Whether white-label branding is active")
    logo_url: str = Field("", description="URL to custom logo image")
    primary_color: str = Field("#1B5E20", description="Primary brand color (hex)")
    secondary_color: str = Field("#388E3C", description="Secondary brand color (hex)")
    accent_color: str = Field("#4CAF50", description="Accent brand color (hex)")
    font_family: str = Field("Inter, sans-serif", description="Primary font family")
    custom_domain: str = Field("", description="Custom domain for white-label deployment")
    custom_css: str = Field("", description="Custom CSS overrides (URL or inline)")
    powered_by_visible: bool = Field(True, description="Show 'Powered by GreenLang' footer")
    email_branding: bool = Field(False, description="Apply branding to email notifications")
    favicon_url: str = Field("", description="URL to custom favicon")
    report_header_logo: str = Field("", description="Logo for generated reports")


class PredictiveConfig(BaseModel):
    """AI/ML predictive analytics configuration."""

    models_enabled: List[str] = Field(
        default_factory=lambda: [
            "emission_forecast", "anomaly_detection", "drift_monitor",
        ],
        description="Enabled ML models",
    )
    forecast_horizon_months: int = Field(
        12, ge=3, le=36, description="Forecast horizon in months"
    )
    confidence_level: float = Field(
        0.95, ge=0.80, le=0.99, description="Prediction confidence level"
    )
    retrain_interval_days: int = Field(
        30, ge=7, le=365, description="Model retraining interval in days"
    )
    anomaly_sensitivity: float = Field(
        0.85, ge=0.5, le=0.99, description="Anomaly detection sensitivity (0-1)"
    )
    explainability_enabled: bool = Field(
        True, description="Enable SHAP/LIME explainability for predictions"
    )
    feature_importance_tracking: bool = Field(
        True, description="Track feature importance over time"
    )
    model_versioning: bool = Field(True, description="Version all ML models")
    auto_retrain_on_drift: bool = Field(
        True, description="Automatically retrain when drift is detected"
    )
    drift_psi_threshold: float = Field(
        0.2, ge=0.05, le=0.5,
        description="Population Stability Index threshold for drift detection",
    )


class NarrativeConfig(BaseModel):
    """AI narrative generation configuration."""

    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages for narrative generation",
    )
    tone: NarrativeTone = Field(
        NarrativeTone.REGULATORY,
        description="Default narrative tone: board, investor, regulatory, public",
    )
    fact_checking_enabled: bool = Field(
        True, description="Enable fact-checking against source data"
    )
    max_draft_tokens: int = Field(
        8000, ge=1000, le=32000, description="Maximum tokens per narrative draft"
    )
    revision_tracking: bool = Field(True, description="Track narrative revisions")
    source_citation_required: bool = Field(
        True, description="Require source citations in narratives"
    )
    template_library_enabled: bool = Field(
        True, description="Enable narrative template library"
    )
    human_review_required: bool = Field(
        True, description="Require human review before publishing"
    )

    @field_validator("tone", mode="before")
    @classmethod
    def parse_tone(cls, v: Any) -> NarrativeTone:
        """Parse tone from string or enum."""
        if isinstance(v, str):
            return NarrativeTone(v.lower())
        return v


class WorkflowBuilderConfig(BaseModel):
    """Custom workflow builder configuration."""

    max_steps: int = Field(50, ge=5, le=200, description="Maximum steps per workflow")
    allowed_step_types: List[str] = Field(
        default_factory=lambda: [
            "agent", "approval", "condition", "timer",
            "notification", "data_transform", "quality_gate", "external_api",
        ],
        description="Allowed step types in custom workflows",
    )
    template_sharing: bool = Field(
        True, description="Allow sharing workflow templates across tenants"
    )
    conditional_logic: bool = Field(True, description="Enable conditional branching")
    parallel_execution: bool = Field(True, description="Enable parallel step execution")
    timer_steps: bool = Field(True, description="Enable timer-based steps")
    human_in_loop: bool = Field(True, description="Enable human-in-the-loop steps")
    max_custom_workflows: int = Field(
        100, ge=1, le=1000, description="Maximum custom workflows per tenant"
    )
    version_control: bool = Field(True, description="Version control for workflows")


class IoTConfig(BaseModel):
    """IoT sensor integration configuration."""

    enabled: bool = Field(False, description="Whether IoT integration is active")
    protocols: List[str] = Field(
        default_factory=lambda: ["MQTT", "HTTP"],
        description="Supported IoT protocols: MQTT, HTTP, OPCUA, MODBUS",
    )
    aggregation_window_minutes: int = Field(
        15, ge=1, le=1440, description="Sensor data aggregation window in minutes"
    )
    max_devices: int = Field(1000, ge=1, le=100000, description="Maximum IoT devices")
    buffer_size_mb: int = Field(512, ge=64, le=8192, description="Ingestion buffer size in MB")
    anomaly_alerting: bool = Field(True, description="Enable anomaly alerting for sensors")
    data_retention_days: int = Field(
        365, ge=30, le=3650, description="Raw sensor data retention in days"
    )
    downsampling_enabled: bool = Field(
        True, description="Enable automatic downsampling for historical data"
    )
    downsampling_after_days: int = Field(
        90, ge=7, description="Days before downsampling kicks in"
    )

    @field_validator("protocols", mode="before")
    @classmethod
    def parse_protocols(cls, v: Any) -> List[str]:
        """Parse protocols to uppercase strings."""
        if isinstance(v, list):
            return [p.upper() if isinstance(p, str) else p for p in v]
        return v


class CarbonCreditConfig(BaseModel):
    """Carbon credit lifecycle management configuration."""

    enabled: bool = Field(False, description="Whether carbon credit management is active")
    registries_enabled: List[str] = Field(
        default_factory=lambda: ["VCS", "GoldStandard"],
        description="Enabled carbon credit registries",
    )
    auto_retirement: bool = Field(
        False, description="Automatically retire credits at year-end"
    )
    vintage_tracking: bool = Field(True, description="Track credit vintages")
    price_tracking: bool = Field(True, description="Track market prices")
    net_zero_accounting: bool = Field(
        True, description="Align credit accounting with SBTi net-zero"
    )
    additionality_verification: bool = Field(
        True, description="Require additionality verification"
    )
    buffer_pool_percent: float = Field(
        10.0, ge=0.0, le=50.0,
        description="Buffer pool percentage for reversal risk",
    )


class SupplyChainConfig(BaseModel):
    """Supply chain ESG risk management configuration."""

    enabled: bool = Field(True, description="Whether supply chain ESG is active")
    max_tiers: int = Field(4, ge=1, le=10, description="Maximum supply chain tiers to track")
    questionnaire_frequency_months: int = Field(
        6, ge=3, le=24, description="Questionnaire dispatch frequency in months"
    )
    scoring_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "environmental": 0.40,
            "social": 0.35,
            "governance": 0.25,
        },
        description="ESG dimension scoring weights",
    )
    risk_threshold: float = Field(
        0.6, ge=0.0, le=1.0,
        description="Risk score threshold for escalation (0-1)",
    )
    auto_dispatch: bool = Field(
        True, description="Automatically dispatch questionnaires on schedule"
    )
    critical_supplier_monitoring: bool = Field(
        True, description="Enable continuous monitoring for critical suppliers"
    )
    deforestation_screening: bool = Field(
        False, description="Enable EUDR deforestation screening"
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "SupplyChainConfig":
        """Validate that scoring weights sum to approximately 1.0."""
        total = sum(self.scoring_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.2f}"
            )
        return self


class FilingConfig(BaseModel):
    """Regulatory filing automation configuration."""

    enabled: bool = Field(True, description="Whether filing automation is active")
    targets: List[str] = Field(
        default_factory=lambda: ["ESAP"],
        description="Filing targets: ESAP, national_registries",
    )
    auto_submit: bool = Field(
        False, description="Automatically submit when validation passes"
    )
    validation_strictness: ValidationStrictness = Field(
        ValidationStrictness.STRICT,
        description="Validation strictness: strict, standard, relaxed",
    )
    deadline_buffer_days: int = Field(
        14, ge=1, le=90, description="Buffer days before filing deadline"
    )
    amendment_tracking: bool = Field(True, description="Track filing amendments")
    submission_receipt_archival: bool = Field(
        True, description="Archive submission receipts"
    )

    @field_validator("validation_strictness", mode="before")
    @classmethod
    def parse_strictness(cls, v: Any) -> ValidationStrictness:
        """Parse strictness from string or enum."""
        if isinstance(v, str):
            return ValidationStrictness(v.lower())
        return v


class APIManagementConfig(BaseModel):
    """API management and rate limiting configuration."""

    rate_limit_per_minute: int = Field(
        600, ge=10, le=100000, description="API rate limit per minute"
    )
    rate_limit_per_day: int = Field(
        1000000, ge=1000, le=100000000, description="API rate limit per day"
    )
    api_key_rotation_days: int = Field(
        90, ge=30, le=365, description="API key rotation interval in days"
    )
    graphql_enabled: bool = Field(True, description="Enable GraphQL API")
    webhook_max_retries: int = Field(5, ge=0, le=10, description="Webhook max retries")
    burst_limit: int = Field(
        100, ge=10, le=10000, description="API burst limit (requests per second)"
    )
    api_versioning: bool = Field(True, description="Enable API versioning")
    cors_enabled: bool = Field(True, description="Enable CORS")
    cors_allowed_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="CORS allowed origins"
    )


class MarketplaceConfig(BaseModel):
    """Plugin marketplace configuration."""

    plugins_enabled: bool = Field(False, description="Whether marketplace is active")
    max_plugins: int = Field(50, ge=1, le=500, description="Max installed plugins")
    auto_update: bool = Field(False, description="Auto-update plugins")
    sandbox_mode: bool = Field(True, description="Run plugins in sandbox")
    allowed_categories: List[str] = Field(
        default_factory=lambda: [
            "data_connector", "report_template", "calculation_engine",
            "visualization", "notification", "integration",
        ],
        description="Allowed plugin categories",
    )
    plugin_review_required: bool = Field(
        True, description="Require review before plugin activation"
    )
    plugin_resource_limits: Dict[str, int] = Field(
        default_factory=lambda: {
            "max_memory_mb": 512,
            "max_cpu_percent": 10,
            "max_storage_mb": 1024,
        },
        description="Resource limits per plugin",
    )


# =============================================================================
# ESRS and Scope 3 Configuration (inherited)
# =============================================================================


class ESRSStandardConfig(BaseModel):
    """Configuration for an individual ESRS standard."""

    id: str = Field(..., description="Standard ID (e.g., E1, S1, G1)")
    name: str = Field(..., description="Standard name")
    enabled: bool = Field(True, description="Whether this standard is active")
    mandatory: bool = Field(False, description="Whether always required")
    material: Optional[bool] = Field(None, description="Materiality assessment result")
    disclosure_requirements: List[str] = Field(
        default_factory=list, description="Active disclosure requirement IDs"
    )


class Scope3Config(BaseModel):
    """Configuration for Scope 3 category enablement."""

    enabled_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Enabled Scope 3 categories (1-15)",
    )
    priority_categories: List[int] = Field(
        default_factory=list, description="Priority categories"
    )
    screening_required: bool = Field(True, description="Whether screening is required")
    materiality_threshold_percent: float = Field(
        1.0, description="Threshold for category exclusion"
    )


class XBRLConfig(BaseModel):
    """XBRL tagging configuration."""

    enabled: bool = Field(True, description="Whether XBRL tagging is active")
    taxonomy_version: str = Field("ESRS_2023", description="XBRL taxonomy version")
    output_format: str = Field("inline_xbrl", description="XBRL output format")
    validation_level: str = Field("strict", description="Validation strictness")
    languages: List[str] = Field(
        default_factory=lambda: ["en"], description="Report languages"
    )


class GuidedModeConfig(BaseModel):
    """Guided mode configuration."""

    enabled: bool = Field(False, description="Whether guided mode is active")
    tutorial_mode: bool = Field(False, description="Show tutorial overlays")
    pre_populated_examples: bool = Field(False, description="Load example data")
    ai_assistance_level: str = Field("enhanced", description="AI assistance level")
    step_by_step: bool = Field(False, description="Enable step-by-step guidance")
    onboarding_extended: bool = Field(False, description="Extended onboarding")


class DemoConfig(BaseModel):
    """Demo mode configuration."""

    enabled: bool = Field(False, description="Whether demo mode is active")
    use_sample_data: bool = Field(False, description="Use bundled sample data")
    skip_external_apis: bool = Field(False, description="Skip external API calls")
    mock_ai_responses: bool = Field(False, description="Use mocked AI responses")
    fast_execution: bool = Field(False, description="Skip delays")
    sample_company_profile: str = Field("", description="Path to sample company profile")
    sample_esg_data: str = Field("", description="Path to sample ESG dataset")
    sample_group_profile: str = Field("", description="Path to sample group profile")
    sample_subsidiary_data: str = Field("", description="Path to sample subsidiary data")
    sample_tenant_profiles: str = Field("", description="Path to sample tenant profiles")
    sample_iot_data: str = Field("", description="Path to sample IoT data")


# =============================================================================
# EnterprisePackConfig - Combines all enterprise-tier configurations
# =============================================================================


class EnterprisePackConfig(BaseModel):
    """
    Combined enterprise-tier configuration.

    This model groups all PACK-003-specific configuration that extends
    beyond what PACK-002 provides. It includes multi-tenant, SSO,
    white-label, predictive analytics, narrative generation, workflow
    builder, IoT, carbon credits, supply chain, filing, API management,
    and marketplace, plus all inherited professional configurations.
    """

    # Inherited from PACK-002
    consolidation: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig,
        description="Multi-entity consolidation configuration",
    )
    approval: ApprovalConfig = Field(
        default_factory=ApprovalConfig, description="Approval workflow configuration"
    )
    quality_gates: QualityGateConfig = Field(
        default_factory=QualityGateConfig, description="Quality gate configuration"
    )
    cross_framework: CrossFrameworkConfig = Field(
        default_factory=CrossFrameworkConfig,
        description="Cross-framework alignment configuration",
    )
    scenarios: ScenarioConfig = Field(
        default_factory=ScenarioConfig, description="Scenario analysis configuration"
    )
    benchmarking: BenchmarkingConfig = Field(
        default_factory=BenchmarkingConfig, description="Benchmarking configuration"
    )
    stakeholder: StakeholderConfig = Field(
        default_factory=StakeholderConfig, description="Stakeholder configuration"
    )
    regulatory: RegulatoryConfig = Field(
        default_factory=RegulatoryConfig, description="Regulatory monitoring configuration"
    )
    data_governance: DataGovernanceConfig = Field(
        default_factory=DataGovernanceConfig, description="Data governance configuration"
    )
    webhooks: WebhookConfig = Field(
        default_factory=WebhookConfig, description="Webhook configuration"
    )
    assurance: AssuranceConfig = Field(
        default_factory=AssuranceConfig, description="External assurance configuration"
    )
    intensity_metrics: IntensityMetricsConfig = Field(
        default_factory=IntensityMetricsConfig, description="Intensity metrics configuration"
    )

    # PACK-003 Enterprise-exclusive
    tenant: MultiTenantConfig = Field(
        default_factory=MultiTenantConfig, description="Multi-tenant configuration"
    )
    sso: SSOConfig = Field(
        default_factory=SSOConfig, description="SSO configuration"
    )
    white_label: WhiteLabelConfig = Field(
        default_factory=WhiteLabelConfig, description="White-label configuration"
    )
    predictive: PredictiveConfig = Field(
        default_factory=PredictiveConfig, description="Predictive analytics configuration"
    )
    narrative: NarrativeConfig = Field(
        default_factory=NarrativeConfig, description="Narrative generation configuration"
    )
    workflow_builder: WorkflowBuilderConfig = Field(
        default_factory=WorkflowBuilderConfig,
        description="Custom workflow builder configuration",
    )
    iot: IoTConfig = Field(
        default_factory=IoTConfig, description="IoT sensor configuration"
    )
    carbon_credit: CarbonCreditConfig = Field(
        default_factory=CarbonCreditConfig,
        description="Carbon credit management configuration",
    )
    supply_chain: SupplyChainConfig = Field(
        default_factory=SupplyChainConfig, description="Supply chain ESG configuration"
    )
    filing: FilingConfig = Field(
        default_factory=FilingConfig, description="Regulatory filing configuration"
    )
    api_management: APIManagementConfig = Field(
        default_factory=APIManagementConfig, description="API management configuration"
    )
    marketplace: MarketplaceConfig = Field(
        default_factory=MarketplaceConfig, description="Marketplace configuration"
    )


# =============================================================================
# PresetConfig - Active preset state
# =============================================================================


class PresetConfig(BaseModel):
    """Merged configuration from size and sector presets."""

    size_preset_id: str = Field("", description="Active size preset identifier")
    sector_preset_id: str = Field("", description="Active sector preset identifier")
    esrs_standards: List[ESRSStandardConfig] = Field(
        default_factory=list, description="ESRS standard configurations"
    )
    scope3: Scope3Config = Field(
        default_factory=Scope3Config, description="Scope 3 configuration"
    )
    xbrl: XBRLConfig = Field(
        default_factory=XBRLConfig, description="XBRL configuration"
    )
    consolidation: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig, description="Consolidation configuration"
    )
    guided_mode: GuidedModeConfig = Field(
        default_factory=GuidedModeConfig, description="Guided mode configuration"
    )
    demo: DemoConfig = Field(
        default_factory=DemoConfig, description="Demo mode configuration"
    )
    emission_factor_priorities: List[str] = Field(
        default_factory=list, description="Emission factor source priorities"
    )
    sector_specific: Dict[str, Any] = Field(
        default_factory=dict, description="Sector-specific configuration"
    )


# =============================================================================
# CSRDEnterprisePackConfig - Top-level pack configuration
# =============================================================================


class CSRDEnterprisePackConfig(BaseModel):
    """
    Top-level CSRD Enterprise Pack configuration model.

    This model represents the fully merged and validated configuration
    for a CSRD Enterprise Pack deployment. It combines the base manifest,
    size preset, sector preset, enterprise configuration, and any
    runtime overrides.
    """

    metadata: PackMetadata = Field(..., description="Pack metadata")
    components: ComponentsConfig = Field(
        default_factory=ComponentsConfig, description="Component configurations"
    )
    workflows: Dict[str, WorkflowConfig] = Field(
        default_factory=dict, description="Workflow configurations"
    )
    templates: List[TemplateConfig] = Field(
        default_factory=list, description="Template configurations"
    )
    performance: PerformanceTargets = Field(
        default_factory=PerformanceTargets, description="Performance targets"
    )
    requirements: RequirementsConfig = Field(
        default_factory=RequirementsConfig, description="System requirements"
    )
    presets: PresetConfig = Field(
        default_factory=PresetConfig, description="Active preset configuration"
    )
    enterprise: EnterprisePackConfig = Field(
        default_factory=EnterprisePackConfig,
        description="Enterprise-tier configuration",
    )

    @field_validator("workflows", mode="before")
    @classmethod
    def parse_workflows(cls, v: Any) -> Dict[str, WorkflowConfig]:
        """Parse workflow definitions from YAML structure."""
        if isinstance(v, dict):
            parsed: Dict[str, WorkflowConfig] = {}
            for key, val in v.items():
                if isinstance(val, WorkflowConfig):
                    parsed[key] = val
                elif isinstance(val, dict):
                    parsed[key] = WorkflowConfig(**val)
                else:
                    parsed[key] = val
            return parsed
        return v

    def get_enabled_workflows(self) -> Dict[str, WorkflowConfig]:
        """Return only enabled workflows."""
        return {k: v for k, v in self.workflows.items() if v.enabled}

    def get_active_agent_ids(self) -> List[str]:
        """Return all enabled agent IDs from components."""
        return self.components.get_enabled_agent_ids()

    def get_enterprise_agent_ids(self) -> List[str]:
        """Return agent IDs exclusive to enterprise tier."""
        return self.components.get_enterprise_agent_ids()

    def get_professional_agent_ids(self) -> List[str]:
        """Return agent IDs exclusive to professional tier."""
        return self.components.get_professional_agent_ids()

    def get_material_esrs_standards(self) -> List[ESRSStandardConfig]:
        """Return ESRS standards marked as material or mandatory."""
        return [
            s
            for s in self.presets.esrs_standards
            if s.mandatory or s.material is True or s.enabled
        ]

    def get_enabled_frameworks(self) -> List[str]:
        """Return list of enabled cross-framework identifiers."""
        return [
            fw
            for fw, enabled in self.enterprise.cross_framework.enabled_frameworks.items()
            if enabled
        ]

    def get_enabled_scenarios(self) -> List[ScenarioDefinition]:
        """Return list of enabled scenario definitions."""
        return [s for s in self.enterprise.scenarios.scenarios if s.enabled]

    def is_multi_tenant(self) -> bool:
        """Check whether multi-tenant mode is active."""
        return self.enterprise.tenant.enabled

    def is_iot_enabled(self) -> bool:
        """Check whether IoT integration is active."""
        return self.enterprise.iot.enabled


# =============================================================================
# PackConfig - Main configuration manager
# =============================================================================


class PackConfig:
    """
    Configuration manager for PACK-003 CSRD Enterprise Pack.

    Loads and merges configuration from multiple sources in the following
    priority order (later sources override earlier):

        1. Base pack.yaml manifest
        2. Size preset (global_enterprise, saas_platform, financial_enterprise, consulting_firm)
        3. Sector preset (banking_enterprise, oil_gas_enterprise, automotive_enterprise,
           pharma_enterprise, conglomerate)
        4. Environment variables (CSRD_ENT_* prefix)
        5. Runtime overrides

    Attributes:
        pack: The fully resolved CSRDEnterprisePackConfig instance.
        config_hash: SHA-256 hash of the resolved configuration for provenance.
        loaded_at: Timestamp when configuration was loaded.
        source_files: List of files loaded during configuration resolution.

    Example:
        >>> config = PackConfig.load(
        ...     size_preset="global_enterprise",
        ...     sector_preset="conglomerate",
        ... )
        >>> print(config.pack.metadata.version)
        '1.0.0'
        >>> print(len(config.active_agents))
        135
    """

    VALID_SIZE_PRESETS = {
        "global_enterprise",
        "saas_platform",
        "financial_enterprise",
        "consulting_firm",
    }
    VALID_SECTOR_PRESETS = {
        "banking_enterprise",
        "oil_gas_enterprise",
        "automotive_enterprise",
        "pharma_enterprise",
        "conglomerate",
    }

    ENV_PREFIX = "CSRD_ENT_"

    def __init__(
        self,
        pack: CSRDEnterprisePackConfig,
        config_hash: str,
        loaded_at: datetime,
        source_files: List[str],
    ) -> None:
        """
        Initialize PackConfig with resolved configuration.

        Args:
            pack: Fully resolved pack configuration.
            config_hash: SHA-256 hash of the configuration.
            loaded_at: Timestamp of configuration loading.
            source_files: List of source files that were loaded.
        """
        self.pack = pack
        self.config_hash = config_hash
        self.loaded_at = loaded_at
        self.source_files = source_files

    @classmethod
    def load(
        cls,
        pack_dir: Optional[Union[str, Path]] = None,
        size_preset: Optional[str] = None,
        sector_preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """
        Load and merge pack configuration from all sources.

        Args:
            pack_dir: Path to the pack root directory.
            size_preset: Size preset to apply.
            sector_preset: Sector preset to apply.
            overrides: Dictionary of runtime configuration overrides.
            demo_mode: If True, loads demo configuration.

        Returns:
            Fully resolved PackConfig instance.

        Raises:
            FileNotFoundError: If pack.yaml or preset files are missing.
            ValueError: If preset names are invalid.
        """
        start_time = datetime.now()
        pack_dir = Path(pack_dir) if pack_dir else PACK_BASE_DIR
        source_files: List[str] = []

        logger.info(
            "Loading CSRD Enterprise Pack configuration from %s", pack_dir
        )

        # Step 1: Load base manifest
        manifest_path = pack_dir / "pack.yaml"
        base_config = cls._load_yaml_file(manifest_path)
        source_files.append(str(manifest_path))
        logger.info("Loaded base manifest: %s", manifest_path)

        # Step 2: Parse base config into structured form
        pack_config_dict = cls._parse_manifest(base_config)

        # Step 3: Apply size preset
        if size_preset:
            cls._validate_preset_name(size_preset, cls.VALID_SIZE_PRESETS, "size")
            preset_path = pack_dir / "config" / "presets" / f"{size_preset}.yaml"
            preset_data = cls._load_yaml_file(preset_path)
            source_files.append(str(preset_path))
            pack_config_dict = cls._merge_preset(
                pack_config_dict, preset_data, "size", size_preset
            )
            logger.info("Applied size preset: %s", size_preset)

        # Step 4: Apply sector preset
        if sector_preset:
            cls._validate_preset_name(
                sector_preset, cls.VALID_SECTOR_PRESETS, "sector"
            )
            sector_path = (
                pack_dir / "config" / "sectors" / f"{sector_preset}.yaml"
            )
            sector_data = cls._load_yaml_file(sector_path)
            source_files.append(str(sector_path))
            pack_config_dict = cls._merge_preset(
                pack_config_dict, sector_data, "sector", sector_preset
            )
            logger.info("Applied sector preset: %s", sector_preset)

        # Step 5: Apply demo mode
        if demo_mode:
            demo_config_path = pack_dir / "config" / "demo" / "demo_config.yaml"
            demo_data = cls._load_yaml_file(demo_config_path)
            source_files.append(str(demo_config_path))
            pack_config_dict = cls._merge_demo(pack_config_dict, demo_data)
            logger.info("Applied demo mode configuration")

        # Step 6: Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            pack_config_dict = cls._deep_merge(pack_config_dict, env_overrides)
            logger.info(
                "Applied %d environment variable overrides", len(env_overrides)
            )

        # Step 7: Apply runtime overrides
        if overrides:
            pack_config_dict = cls._deep_merge(pack_config_dict, overrides)
            logger.info("Applied runtime overrides")

        # Step 8: Validate and create typed config
        pack = CSRDEnterprisePackConfig(**pack_config_dict)

        # Step 9: Calculate configuration hash
        config_json = json.dumps(
            pack_config_dict, sort_keys=True, default=str
        )
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        load_duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            "Enterprise Pack configuration loaded in %.1fms (hash: %s)",
            load_duration,
            config_hash[:12],
        )

        return cls(
            pack=pack,
            config_hash=config_hash,
            loaded_at=start_time,
            source_files=source_files,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """
        Load configuration from a single YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            PackConfig instance loaded from the YAML file.
        """
        yaml_path = Path(yaml_path)
        data = cls._load_yaml_file(yaml_path)
        pack = CSRDEnterprisePackConfig(**data)
        config_json = json.dumps(data, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        return cls(
            pack=pack,
            config_hash=config_hash,
            loaded_at=datetime.now(),
            source_files=[str(yaml_path)],
        )

    @classmethod
    def from_preset(
        cls,
        size_preset: str,
        sector_preset: Optional[str] = None,
        pack_dir: Optional[Union[str, Path]] = None,
    ) -> "PackConfig":
        """
        Load configuration from a preset combination.

        Args:
            size_preset: Size preset name.
            sector_preset: Optional sector preset name.
            pack_dir: Pack root directory.

        Returns:
            PackConfig instance with the preset applied.
        """
        return cls.load(
            pack_dir=pack_dir,
            size_preset=size_preset,
            sector_preset=sector_preset,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def active_agents(self) -> List[str]:
        """Return list of all enabled agent IDs."""
        return self.pack.get_active_agent_ids()

    @property
    def enterprise_agents(self) -> List[str]:
        """Return list of PACK-003 exclusive agent IDs."""
        return self.pack.get_enterprise_agent_ids()

    @property
    def professional_agents(self) -> List[str]:
        """Return list of PACK-002 exclusive agent IDs."""
        return self.pack.get_professional_agent_ids()

    @property
    def active_workflows(self) -> Dict[str, WorkflowConfig]:
        """Return dictionary of enabled workflows."""
        return self.pack.get_enabled_workflows()

    @property
    def material_standards(self) -> List[ESRSStandardConfig]:
        """Return ESRS standards that are material or mandatory."""
        return self.pack.get_material_esrs_standards()

    @property
    def enabled_frameworks(self) -> List[str]:
        """Return list of enabled cross-framework identifiers."""
        return self.pack.get_enabled_frameworks()

    @property
    def enabled_scenarios(self) -> List[ScenarioDefinition]:
        """Return list of enabled scenario definitions."""
        return self.pack.get_enabled_scenarios()

    @property
    def consolidation_enabled(self) -> bool:
        """Check whether multi-entity consolidation is active."""
        return self.pack.enterprise.consolidation.enabled

    @property
    def multi_tenant_enabled(self) -> bool:
        """Check whether multi-tenant mode is active."""
        return self.pack.is_multi_tenant()

    @property
    def iot_enabled(self) -> bool:
        """Check whether IoT integration is active."""
        return self.pack.is_iot_enabled()

    @property
    def assurance_level(self) -> str:
        """Return current assurance level."""
        return self.pack.enterprise.assurance.assurance_level

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_agent_config(self, agent_id: str) -> Optional[AgentComponentConfig]:
        """Retrieve configuration for a specific agent."""
        return self.pack.components.find_agent(agent_id)

    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Retrieve configuration for a specific workflow."""
        return self.pack.workflows.get(workflow_id)

    def is_scope3_category_enabled(self, category: int) -> bool:
        """Check if a Scope 3 category is enabled."""
        return category in self.pack.presets.scope3.enabled_categories

    def is_esrs_standard_enabled(self, standard_id: str) -> bool:
        """Check if an ESRS standard is enabled."""
        for standard in self.pack.presets.esrs_standards:
            if standard.id == standard_id:
                return standard.enabled
        return False

    def is_framework_enabled(self, framework_id: str) -> bool:
        """Check if a cross-framework alignment is enabled."""
        return self.pack.enterprise.cross_framework.enabled_frameworks.get(
            framework_id, False
        )

    def is_scenario_enabled(self, scenario_id: str) -> bool:
        """Check if a climate scenario is enabled."""
        for scenario in self.pack.enterprise.scenarios.scenarios:
            if scenario.id == scenario_id:
                return scenario.enabled
        return False

    def get_quality_gate(self, gate_id: str) -> Optional[QualityGate]:
        """Retrieve a quality gate by ID."""
        for gate in self.pack.enterprise.quality_gates.gates:
            if gate.id == gate_id:
                return gate
        return None

    def get_entity(self, entity_id: str) -> Optional[EntityDefinition]:
        """Retrieve an entity definition by ID."""
        for entity in self.pack.enterprise.consolidation.entities:
            if entity.entity_id == entity_id:
                return entity
        return None

    def get_tenant_isolation_level(self) -> str:
        """Return the current tenant isolation level."""
        return self.pack.enterprise.tenant.isolation_level.value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full configuration to a dictionary."""
        return {
            "pack": self.pack.model_dump(),
            "config_hash": self.config_hash,
            "loaded_at": self.loaded_at.isoformat(),
            "source_files": self.source_files,
        }

    def get_provenance_hash(self) -> str:
        """Get SHA-256 hash of the configuration for audit provenance."""
        return self.config_hash

    # =========================================================================
    # Internal Methods
    # =========================================================================

    @staticmethod
    def _load_yaml_file(path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected YAML dictionary in {path}, got {type(data).__name__}"
            )
        return data

    @staticmethod
    def _validate_preset_name(
        name: str, valid_names: set, preset_type: str
    ) -> None:
        """Validate that a preset name is recognized."""
        if name not in valid_names:
            raise ValueError(
                f"Invalid {preset_type} preset '{name}'. "
                f"Valid options: {sorted(valid_names)}"
            )

    @classmethod
    def _parse_manifest(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw YAML manifest into CSRDEnterprisePackConfig structure."""
        result: Dict[str, Any] = {}

        # Metadata
        result["metadata"] = raw.get("metadata", {})

        # Components
        raw_components = raw.get("components", {})
        result["components"] = {}
        for group_key in [
            "apps", "data_agents", "quality_agents",
            "mrv_scope1", "mrv_scope2", "mrv_scope3", "foundation",
            "professional_apps", "cdp_engines", "tcfd_engines",
            "sbti_engines", "taxonomy_engines", "professional_engines",
            "predictive_engines", "narrative_engines", "iot_engines",
            "carbon_credit_engines", "supply_chain_engines",
            "filing_engines", "workflow_builder_engines",
            "enterprise_apps", "enterprise_data_agents",
            "enterprise_quality_agents", "eudr_agents",
        ]:
            agents = raw_components.get(group_key, [])
            result["components"][group_key] = [
                {
                    "id": a.get("id", ""),
                    "name": a.get("name", ""),
                    "description": a.get("description", ""),
                    "required": a.get("required", True),
                    "enabled": a.get("enabled", True),
                    "version": a.get("version", ""),
                    "config_overrides": a.get("config_overrides", {}),
                }
                for a in agents
            ]

        # Workflows
        result["workflows"] = raw.get("workflows", {})

        # Templates
        result["templates"] = raw.get("templates", [])

        # Performance targets
        raw_perf = raw.get("performance", {})
        result["performance"] = {
            "data_ingestion_rps": raw_perf.get("data_ingestion", {}).get(
                "throughput_records_per_second", 50000
            ),
            "data_ingestion_max_latency_ms": raw_perf.get("data_ingestion", {}).get(
                "max_latency_ms", 80
            ),
            "iot_events_per_second": raw_perf.get("data_ingestion", {}).get(
                "iot_events_per_second", 10000
            ),
            "ghg_single_scope_max_seconds": raw_perf.get("ghg_calculation", {}).get(
                "single_scope_max_seconds", 30
            ),
            "ghg_full_inventory_max_seconds": raw_perf.get("ghg_calculation", {}).get(
                "full_inventory_max_seconds", 300
            ),
            "ghg_multi_entity_max_minutes": raw_perf.get("ghg_calculation", {}).get(
                "multi_entity_max_minutes", 30
            ),
            "ghg_max_data_points": raw_perf.get("ghg_calculation", {}).get(
                "max_data_points", 500000
            ),
            "report_esrs_max_seconds": raw_perf.get("report_generation", {}).get(
                "esrs_disclosure_max_seconds", 120
            ),
            "report_summary_max_seconds": raw_perf.get("report_generation", {}).get(
                "executive_summary_max_seconds", 60
            ),
            "report_xbrl_max_seconds": raw_perf.get("report_generation", {}).get(
                "xbrl_tagging_max_seconds", 180
            ),
            "report_consolidated_max_seconds": raw_perf.get("report_generation", {}).get(
                "consolidated_report_max_seconds", 300
            ),
            "report_cross_framework_max_seconds": raw_perf.get(
                "report_generation", {}
            ).get("cross_framework_max_seconds", 120),
            "report_narrative_max_seconds": raw_perf.get("report_generation", {}).get(
                "narrative_generation_max_seconds", 180
            ),
            "report_multi_language_max_seconds": raw_perf.get(
                "report_generation", {}
            ).get("multi_language_max_seconds", 600),
            "forecast_max_seconds": raw_perf.get("predictive_analytics", {}).get(
                "forecast_max_seconds", 60
            ),
            "anomaly_detection_max_seconds": raw_perf.get(
                "predictive_analytics", {}
            ).get("anomaly_detection_max_seconds", 30),
            "drift_check_max_seconds": raw_perf.get("predictive_analytics", {}).get(
                "drift_check_max_seconds", 15
            ),
            "model_retraining_max_minutes": raw_perf.get(
                "predictive_analytics", {}
            ).get("model_retraining_max_minutes", 30),
            "scenario_single_max_seconds": raw_perf.get("scenario_analysis", {}).get(
                "single_scenario_max_seconds", 60
            ),
            "scenario_full_max_minutes": raw_perf.get("scenario_analysis", {}).get(
                "full_analysis_max_minutes", 15
            ),
            "scenario_monte_carlo_iterations": raw_perf.get(
                "scenario_analysis", {}
            ).get("monte_carlo_iterations", 50000),
            "api_p50_ms": raw_perf.get("api_response", {}).get("p50_ms", 40),
            "api_p95_ms": raw_perf.get("api_response", {}).get("p95_ms", 200),
            "api_p99_ms": raw_perf.get("api_response", {}).get("p99_ms", 800),
            "graphql_p50_ms": raw_perf.get("api_response", {}).get(
                "graphql_p50_ms", 60
            ),
            "graphql_p95_ms": raw_perf.get("api_response", {}).get(
                "graphql_p95_ms", 300
            ),
            "quality_gate_max_seconds": raw_perf.get("data_quality", {}).get(
                "quality_gate_max_seconds", 30
            ),
            "availability_percent": raw_perf.get("availability", {}).get(
                "target_uptime_percent", 99.99
            ),
            "rpo_minutes": raw_perf.get("availability", {}).get("rpo_minutes", 5),
            "rto_minutes": raw_perf.get("availability", {}).get("rto_minutes", 15),
        }

        # Requirements
        raw_req = raw.get("requirements", {})
        raw_runtime = raw_req.get("runtime", {})
        raw_infra = raw_req.get("infrastructure", {})
        raw_db = raw_req.get("database", {})
        result["requirements"] = {
            "python_version": raw_runtime.get("python", ">=3.11"),
            "postgresql_version": raw_runtime.get("postgresql", ">=16"),
            "redis_version": raw_runtime.get("redis", ">=7"),
            "timescaledb_version": raw_runtime.get("timescaledb", ">=2.13"),
            "min_cpu_cores": raw_infra.get("min_cpu_cores", 16),
            "min_memory_gb": raw_infra.get("min_memory_gb", 64),
            "min_storage_gb": raw_infra.get("min_storage_gb", 1000),
            "recommended_cpu_cores": raw_infra.get("recommended_cpu_cores", 32),
            "recommended_memory_gb": raw_infra.get("recommended_memory_gb", 128),
            "recommended_storage_gb": raw_infra.get("recommended_storage_gb", 5000),
            "database_extensions": raw_db.get(
                "extensions", ["pgvector", "timescaledb", "pg_partman"]
            ),
            "min_db_connections": raw_db.get("min_connections", 100),
            "recommended_db_connections": raw_db.get(
                "recommended_connections", 300
            ),
        }

        # Presets - initialize empty
        result["presets"] = {}

        # Enterprise config - initialize with defaults
        result["enterprise"] = {}

        return result

    @classmethod
    def _merge_preset(
        cls,
        base: Dict[str, Any],
        preset_data: Dict[str, Any],
        preset_type: str,
        preset_id: str,
    ) -> Dict[str, Any]:
        """Merge a preset configuration into the base configuration."""
        result = cls._deep_merge(base, {})

        if "presets" not in result:
            result["presets"] = {}

        if preset_type == "size":
            result["presets"]["size_preset_id"] = preset_id
        elif preset_type == "sector":
            result["presets"]["sector_preset_id"] = preset_id

        for key in [
            "esrs_standards", "scope3", "xbrl", "consolidation",
            "guided_mode", "demo", "emission_factor_priorities",
        ]:
            if key in preset_data:
                result["presets"][key] = preset_data[key]

        if "sector_specific" in preset_data:
            existing_sector = result["presets"].get("sector_specific", {})
            result["presets"]["sector_specific"] = cls._deep_merge(
                existing_sector, preset_data["sector_specific"]
            )

        if "agent_overrides" in preset_data:
            result = cls._apply_agent_overrides(
                result, preset_data["agent_overrides"]
            )

        if "performance" in preset_data:
            result["performance"] = cls._deep_merge(
                result.get("performance", {}), preset_data["performance"]
            )

        if "enterprise" in preset_data:
            if "enterprise" not in result:
                result["enterprise"] = {}
            result["enterprise"] = cls._deep_merge(
                result["enterprise"], preset_data["enterprise"]
            )

        # Also support "professional" key for backward compatibility
        if "professional" in preset_data:
            if "enterprise" not in result:
                result["enterprise"] = {}
            result["enterprise"] = cls._deep_merge(
                result["enterprise"], preset_data["professional"]
            )

        return result

    @classmethod
    def _merge_demo(
        cls, base: Dict[str, Any], demo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge demo configuration into base configuration."""
        result = cls._deep_merge(base, {})

        if "presets" not in result:
            result["presets"] = {}

        result["presets"]["demo"] = {
            "enabled": demo_data.get("enabled", True),
            "use_sample_data": demo_data.get("use_sample_data", True),
            "skip_external_apis": demo_data.get("skip_external_apis", True),
            "mock_ai_responses": demo_data.get("mock_ai_responses", True),
            "fast_execution": demo_data.get("fast_execution", True),
            "sample_company_profile": demo_data.get("sample_company_profile", ""),
            "sample_esg_data": demo_data.get("sample_esg_data", ""),
            "sample_group_profile": demo_data.get("sample_group_profile", ""),
            "sample_subsidiary_data": demo_data.get("sample_subsidiary_data", ""),
            "sample_tenant_profiles": demo_data.get("sample_tenant_profiles", ""),
            "sample_iot_data": demo_data.get("sample_iot_data", ""),
        }

        if "agent_overrides" in demo_data:
            result = cls._apply_agent_overrides(
                result, demo_data["agent_overrides"]
            )

        if "enterprise" in demo_data:
            if "enterprise" not in result:
                result["enterprise"] = {}
            result["enterprise"] = cls._deep_merge(
                result["enterprise"], demo_data["enterprise"]
            )

        return result

    @classmethod
    def _apply_agent_overrides(
        cls,
        config: Dict[str, Any],
        overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply agent-level overrides to component configuration."""
        components = config.get("components", {})

        for group_key in [
            "apps", "data_agents", "quality_agents",
            "mrv_scope1", "mrv_scope2", "mrv_scope3", "foundation",
            "professional_apps", "cdp_engines", "tcfd_engines",
            "sbti_engines", "taxonomy_engines", "professional_engines",
            "predictive_engines", "narrative_engines", "iot_engines",
            "carbon_credit_engines", "supply_chain_engines",
            "filing_engines", "workflow_builder_engines",
            "enterprise_apps", "enterprise_data_agents",
            "enterprise_quality_agents", "eudr_agents",
        ]:
            agents = components.get(group_key, [])
            for agent in agents:
                agent_id = agent.get("id", "")
                if agent_id in overrides:
                    agent_override = overrides[agent_id]
                    if "enabled" in agent_override:
                        agent["enabled"] = agent_override["enabled"]
                    if "config_overrides" in agent_override:
                        existing = agent.get("config_overrides", {})
                        agent["config_overrides"] = cls._deep_merge(
                            existing, agent_override["config_overrides"]
                        )

        return config

    @classmethod
    def _load_env_overrides(cls) -> Dict[str, Any]:
        """Load configuration overrides from CSRD_ENT_* environment variables."""
        overrides: Dict[str, Any] = {}
        prefix = cls.ENV_PREFIX

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().split("__")
                parsed_value = cls._parse_env_value(value)
                current = overrides
                for i, part in enumerate(config_path):
                    if i == len(config_path) - 1:
                        current[part] = parsed_value
                    else:
                        current = current.setdefault(part, {})

        return overrides

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse an environment variable value to its appropriate Python type."""
        lower_val = value.lower()
        if lower_val in ("true", "yes", "1"):
            return True
        if lower_val in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @classmethod
    def _deep_merge(
        cls, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries. Override values take precedence."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """Return string representation of PackConfig."""
        return (
            f"PackConfig("
            f"version={self.pack.metadata.version!r}, "
            f"tier={self.pack.metadata.tier!r}, "
            f"size={self.pack.presets.size_preset_id!r}, "
            f"sector={self.pack.presets.sector_preset_id!r}, "
            f"agents={len(self.active_agents)}, "
            f"frameworks={len(self.enabled_frameworks)}, "
            f"multi_tenant={self.multi_tenant_enabled}, "
            f"hash={self.config_hash[:12]})"
        )
