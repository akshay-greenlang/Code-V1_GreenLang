"""
PACK-002 CSRD Professional Pack - Configuration Manager

This module implements the PackConfig class that loads, merges, and validates
all configuration for the CSRD Professional Pack. It extends PACK-001's
configuration system with enterprise-grade features including multi-entity
consolidation, cross-framework alignment, scenario analysis, approval
workflows, quality gates, and regulatory change management.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest (PACK-001 inherited + PACK-002 additions)
    2. Size preset (enterprise_group / listed_company / financial_institution / multinational)
    3. Sector preset (manufacturing_pro / financial_services_pro / technology_pro / energy_pro / heavy_industry_pro)
    4. Environment overrides (CSRD_PRO_* environment variables)
    5. Explicit runtime overrides

Example:
    >>> config = PackConfig.load(
    ...     size_preset="enterprise_group",
    ...     sector_preset="manufacturing_pro",
    ... )
    >>> print(config.pack.metadata.display_name)
    'CSRD Professional Pack'
    >>> print(len(config.active_agents))
    93
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Pydantic Models - Core data structures (inherited from PACK-001)
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
    tier: str = Field("professional", description="Pack tier (starter, professional, enterprise)")
    author: str = Field("", description="Pack author or team")
    license: str = Field("Proprietary", description="License type")
    min_platform_version: str = Field("2.0.0", description="Minimum GreenLang platform version")
    release_date: str = Field("", description="Release date ISO string")
    support_tier: str = Field("enterprise", description="Support tier level")
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
    """All components included in the pack, including inherited and professional."""

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
    duration_days: int = Field(1, ge=1, description="Estimated duration in days")


class WorkflowConfig(BaseModel):
    """Configuration for a workflow orchestration."""

    display_name: str = Field(..., description="Human-readable workflow name")
    description: str = Field("", description="Workflow description")
    schedule: str = Field("on_demand", description="Schedule type: annual, quarterly, monthly, on_demand")
    estimated_duration_days: int = Field(1, ge=1, description="Total estimated duration")
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
    """Performance targets for the professional pack."""

    data_ingestion_rps: int = Field(10000, description="Records per second for data ingestion")
    data_ingestion_max_latency_ms: int = Field(150, description="Max latency for ingestion")
    ghg_single_scope_max_seconds: int = Field(45, description="Max seconds for single scope calc")
    ghg_full_inventory_max_seconds: int = Field(600, description="Max seconds for full inventory")
    ghg_multi_entity_max_minutes: int = Field(45, description="Max minutes for multi-entity calc")
    ghg_max_data_points: int = Field(50000, description="Max data points for multi-entity")
    report_esrs_max_seconds: int = Field(180, description="Max seconds for ESRS report")
    report_summary_max_seconds: int = Field(90, description="Max seconds for executive summary")
    report_xbrl_max_seconds: int = Field(300, description="Max seconds for XBRL tagging")
    report_consolidated_max_seconds: int = Field(600, description="Max seconds for consolidated report")
    report_cross_framework_max_seconds: int = Field(240, description="Max seconds for cross-framework")
    scenario_single_max_seconds: int = Field(120, description="Max seconds for single scenario")
    scenario_full_max_minutes: int = Field(30, description="Max minutes for full scenario analysis")
    scenario_monte_carlo_iterations: int = Field(10000, description="Monte Carlo iterations")
    api_p50_ms: int = Field(80, description="API p50 latency target")
    api_p95_ms: int = Field(400, description="API p95 latency target")
    api_p99_ms: int = Field(1500, description="API p99 latency target")
    quality_gate_max_seconds: int = Field(60, description="Max seconds for quality gate check")
    availability_percent: float = Field(99.95, description="Target uptime percentage")


class RequirementsConfig(BaseModel):
    """System requirements for the professional pack."""

    python_version: str = Field(">=3.11", description="Minimum Python version")
    postgresql_version: str = Field(">=14", description="Minimum PostgreSQL version")
    redis_version: str = Field(">=7", description="Minimum Redis version")
    min_cpu_cores: int = Field(8, description="Minimum CPU cores")
    min_memory_gb: int = Field(32, description="Minimum memory in GB")
    min_storage_gb: int = Field(250, description="Minimum storage in GB")
    recommended_cpu_cores: int = Field(16, description="Recommended CPU cores")
    recommended_memory_gb: int = Field(64, description="Recommended memory in GB")
    recommended_storage_gb: int = Field(1000, description="Recommended storage in GB")
    database_extensions: List[str] = Field(
        default_factory=lambda: ["pgvector", "timescaledb"],
        description="Required database extensions",
    )
    min_db_connections: int = Field(50, description="Minimum database connections")


# =============================================================================
# Pydantic Models - PACK-002 Professional Configuration Models
# =============================================================================


class EntityDefinition(BaseModel):
    """Definition of a subsidiary entity in a corporate group."""

    entity_id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Legal entity name")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    ownership_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Ownership percentage (0-100)"
    )
    consolidation_method: str = Field(
        "full",
        description="Consolidation method: full, proportional, equity"
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
            raise ValueError(f"Invalid consolidation method '{v}'. Must be one of: {sorted(valid)}")
        return v


class ConsolidationConfig(BaseModel):
    """Multi-entity consolidation configuration for corporate groups."""

    enabled: bool = Field(False, description="Whether multi-entity consolidation is active")
    max_subsidiaries: int = Field(100, ge=1, le=500, description="Maximum subsidiaries supported")
    consolidation_approaches: List[str] = Field(
        default_factory=lambda: ["operational_control", "financial_control", "equity_share"],
        description="Supported consolidation approaches"
    )
    default_approach: str = Field(
        "operational_control", description="Default consolidation approach"
    )
    intercompany_elimination: bool = Field(
        True, description="Eliminate intercompany transactions"
    )
    minority_interest_adjustment: bool = Field(
        True, description="Adjust for minority interests in non-wholly-owned subsidiaries"
    )
    entities: List[EntityDefinition] = Field(
        default_factory=list, description="Subsidiary entity definitions"
    )
    parallel_entity_processing: bool = Field(
        True, description="Process entities in parallel for performance"
    )
    entity_data_timeout_seconds: int = Field(
        300, description="Timeout for individual entity data collection"
    )

    @field_validator("default_approach")
    @classmethod
    def validate_default_approach(cls, v: str) -> str:
        """Validate default consolidation approach."""
        valid = {"operational_control", "financial_control", "equity_share"}
        if v not in valid:
            raise ValueError(f"Invalid consolidation approach '{v}'. Must be one of: {sorted(valid)}")
        return v


class ApprovalLevel(BaseModel):
    """Configuration for a single approval level."""

    role: str = Field(..., description="Approval role: preparer, reviewer, approver, board")
    required: bool = Field(True, description="Whether this approval level is required")
    auto_approve_threshold: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Quality score threshold for auto-approval (0-100, None=manual)"
    )
    escalation_timeout_hours: int = Field(
        48, ge=1, description="Hours before escalation to next level"
    )
    delegation_enabled: bool = Field(True, description="Allow delegation to another user")
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email"],
        description="Notification channels: email, slack, teams, webhook"
    )


class ApprovalConfig(BaseModel):
    """Approval workflow configuration with multi-level approval chain."""

    enabled: bool = Field(True, description="Whether approval workflows are active")
    levels: List[ApprovalLevel] = Field(
        default_factory=lambda: [
            ApprovalLevel(role="preparer", required=True, auto_approve_threshold=None,
                          escalation_timeout_hours=72),
            ApprovalLevel(role="reviewer", required=True, auto_approve_threshold=95.0,
                          escalation_timeout_hours=48),
            ApprovalLevel(role="approver", required=True, auto_approve_threshold=None,
                          escalation_timeout_hours=48),
            ApprovalLevel(role="board", required=True, auto_approve_threshold=None,
                          escalation_timeout_hours=168),
        ],
        description="Ordered approval levels from preparer to board"
    )
    require_comments_on_rejection: bool = Field(
        True, description="Require comments when rejecting an approval"
    )
    audit_trail_enabled: bool = Field(
        True, description="Track all approval actions in audit trail"
    )
    parallel_approval_allowed: bool = Field(
        False, description="Allow multiple approvers at same level"
    )


class QualityGate(BaseModel):
    """Configuration for a single quality gate."""

    id: str = Field(..., description="Quality gate identifier")
    name: str = Field(..., description="Human-readable gate name")
    enabled: bool = Field(True, description="Whether this gate is active")
    threshold: float = Field(
        ..., ge=0.0, le=100.0,
        description="Minimum score to pass this gate (0-100)"
    )
    weight: float = Field(
        1.0, ge=0.0, le=10.0,
        description="Weight of this gate in overall quality score"
    )
    blocking: bool = Field(
        True, description="Whether failing this gate blocks the pipeline"
    )
    checks: List[str] = Field(
        default_factory=list, description="List of check IDs in this gate"
    )


class QualityGateConfig(BaseModel):
    """Quality gate configuration with weighted scoring."""

    enabled: bool = Field(True, description="Whether quality gates are active")
    gates: List[QualityGate] = Field(
        default_factory=lambda: [
            QualityGate(
                id="data_completeness",
                name="Data Completeness Gate",
                enabled=True,
                threshold=90.0,
                weight=3.0,
                blocking=True,
                checks=[
                    "mandatory_fields_present",
                    "reference_data_linked",
                    "emission_factors_resolved",
                    "time_series_complete",
                    "entity_coverage_complete",
                ],
            ),
            QualityGate(
                id="calculation_integrity",
                name="Calculation Integrity Gate",
                enabled=True,
                threshold=95.0,
                weight=5.0,
                blocking=True,
                checks=[
                    "scope_totals_reconcile",
                    "consolidation_balances",
                    "year_on_year_variance_explained",
                    "unit_conversion_verified",
                    "emission_factor_sources_valid",
                    "provenance_hashes_verified",
                ],
            ),
            QualityGate(
                id="compliance_readiness",
                name="Compliance Readiness Gate",
                enabled=True,
                threshold=85.0,
                weight=4.0,
                blocking=False,
                checks=[
                    "esrs_disclosures_complete",
                    "xbrl_tags_valid",
                    "materiality_assessment_documented",
                    "audit_trail_complete",
                    "cross_framework_alignment_verified",
                ],
            ),
        ],
        description="Quality gates with weighted scoring"
    )
    overall_pass_threshold: float = Field(
        88.0, ge=0.0, le=100.0,
        description="Weighted average threshold for overall pass"
    )
    run_on_commit: bool = Field(
        False, description="Run quality gates on every data commit"
    )
    run_on_calculation: bool = Field(
        True, description="Run quality gates after every calculation run"
    )


class CrossFrameworkConfig(BaseModel):
    """Cross-framework alignment configuration."""

    enabled: bool = Field(True, description="Whether cross-framework alignment is active")
    enabled_frameworks: Dict[str, bool] = Field(
        default_factory=lambda: {
            "cdp": True,
            "tcfd": True,
            "sbti": True,
            "taxonomy": True,
            "gri": True,
            "sasb": True,
        },
        description="Frameworks enabled for alignment mapping"
    )
    mapping_version: str = Field(
        "2024.1", description="Version of the cross-framework mapping table"
    )
    auto_populate_from_esrs: bool = Field(
        True, description="Auto-populate framework responses from ESRS data"
    )
    gap_analysis_enabled: bool = Field(
        True, description="Enable gap analysis between frameworks"
    )
    reconciliation_checks: bool = Field(
        True, description="Run reconciliation checks across framework outputs"
    )
    primary_framework: str = Field(
        "esrs", description="Primary framework from which data flows to others"
    )


class ScenarioDefinition(BaseModel):
    """Definition of a single climate scenario."""

    id: str = Field(..., description="Scenario identifier")
    name: str = Field(..., description="Human-readable scenario name")
    source: str = Field(..., description="Scenario source (IEA, NGFS, custom)")
    temperature_outcome_c: Optional[float] = Field(
        None, description="Temperature outcome in degrees Celsius"
    )
    time_horizons: List[int] = Field(
        default_factory=lambda: [2030, 2040, 2050],
        description="Analysis time horizons (years)"
    )
    enabled: bool = Field(True, description="Whether this scenario is enabled")


class ScenarioConfig(BaseModel):
    """Climate scenario analysis configuration."""

    enabled: bool = Field(True, description="Whether scenario analysis is active")
    scenarios: List[ScenarioDefinition] = Field(
        default_factory=lambda: [
            ScenarioDefinition(
                id="iea_nze", name="IEA Net Zero by 2050",
                source="IEA", temperature_outcome_c=1.5
            ),
            ScenarioDefinition(
                id="iea_aps", name="IEA Announced Pledges",
                source="IEA", temperature_outcome_c=1.7
            ),
            ScenarioDefinition(
                id="iea_steps", name="IEA Stated Policies",
                source="IEA", temperature_outcome_c=2.5
            ),
            ScenarioDefinition(
                id="ngfs_orderly", name="NGFS Orderly Transition",
                source="NGFS", temperature_outcome_c=1.5
            ),
            ScenarioDefinition(
                id="ngfs_disorderly", name="NGFS Disorderly Transition",
                source="NGFS", temperature_outcome_c=1.5
            ),
            ScenarioDefinition(
                id="ngfs_hot_house", name="NGFS Hot House World",
                source="NGFS", temperature_outcome_c=3.0
            ),
            ScenarioDefinition(
                id="ngfs_too_little", name="NGFS Too Little Too Late",
                source="NGFS", temperature_outcome_c=2.5
            ),
            ScenarioDefinition(
                id="custom_bau", name="Custom Business-as-Usual",
                source="custom", temperature_outcome_c=3.5,
                enabled=False
            ),
        ],
        description="Climate scenarios for analysis"
    )
    monte_carlo_enabled: bool = Field(
        True, description="Enable Monte Carlo simulation for uncertainty analysis"
    )
    monte_carlo_iterations: int = Field(
        10000, ge=100, le=100000,
        description="Number of Monte Carlo iterations"
    )
    monte_carlo_confidence_level: float = Field(
        0.95, ge=0.80, le=0.99,
        description="Confidence level for Monte Carlo results"
    )
    physical_risk_enabled: bool = Field(
        True, description="Include physical risk assessment in scenarios"
    )
    transition_risk_enabled: bool = Field(
        True, description="Include transition risk assessment in scenarios"
    )
    financial_impact_quantification: bool = Field(
        True, description="Quantify financial impacts under each scenario"
    )
    time_horizons: List[int] = Field(
        default_factory=lambda: [2030, 2040, 2050],
        description="Default analysis time horizons"
    )


class BenchmarkingConfig(BaseModel):
    """Peer benchmarking and ESG rating comparison configuration."""

    enabled: bool = Field(True, description="Whether benchmarking is active")
    peer_comparison_enabled: bool = Field(
        True, description="Compare performance against sector peers"
    )
    peer_group_source: str = Field(
        "nace_code", description="Source for peer group definition: nace_code, gics, custom"
    )
    max_peer_count: int = Field(
        20, ge=5, le=100,
        description="Maximum peers in comparison group"
    )
    esg_rating_frameworks: List[str] = Field(
        default_factory=lambda: ["msci", "sustainalytics", "iss_esg", "cdp"],
        description="ESG rating frameworks for comparison"
    )
    percentile_targets: Dict[str, int] = Field(
        default_factory=lambda: {
            "ghg_intensity": 25,
            "renewable_energy": 75,
            "water_intensity": 25,
            "waste_diversion": 75,
            "safety_rate": 25,
        },
        description="Target percentile rankings (lower=better for intensity, higher=better for %)"
    )
    update_frequency: str = Field(
        "quarterly", description="How often to refresh benchmark data"
    )


class StakeholderConfig(BaseModel):
    """Stakeholder engagement configuration."""

    enabled: bool = Field(True, description="Whether stakeholder engagement features are active")
    stakeholder_categories: List[str] = Field(
        default_factory=lambda: [
            "investors", "employees", "customers", "suppliers",
            "communities", "regulators", "ngos", "media",
        ],
        description="Stakeholder categories for engagement"
    )
    survey_enabled: bool = Field(True, description="Enable stakeholder surveys")
    survey_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Survey languages"
    )
    response_target_per_category: int = Field(
        50, ge=10, description="Target response count per stakeholder category"
    )
    materiality_weight_in_scoring: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Weight of stakeholder input in materiality scoring (0-1)"
    )
    anonymous_responses_allowed: bool = Field(
        True, description="Allow anonymous survey responses"
    )


class RegulatoryConfig(BaseModel):
    """Regulatory change monitoring configuration."""

    enabled: bool = Field(True, description="Whether regulatory monitoring is active")
    monitored_jurisdictions: List[str] = Field(
        default_factory=lambda: ["EU", "DE", "FR", "NL", "ES", "IT"],
        description="Jurisdictions to monitor for regulatory changes (ISO codes)"
    )
    monitored_regulations: List[str] = Field(
        default_factory=lambda: [
            "CSRD", "ESRS", "EU_Taxonomy", "SFDR", "CBAM", "CSDDD", "ESEF",
        ],
        description="Regulations to monitor for changes"
    )
    scan_frequency: str = Field(
        "weekly", description="How often to scan for regulatory changes"
    )
    impact_assessment_auto: bool = Field(
        True, description="Automatically assess impact of detected changes"
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "dashboard"],
        description="Channels for regulatory change notifications"
    )
    filing_calendars: Dict[str, str] = Field(
        default_factory=lambda: {
            "EU": "Annual, April 30",
            "DE": "Annual, April 30",
            "FR": "Annual, April 30",
        },
        description="Filing calendar per jurisdiction"
    )


class DataGovernanceConfig(BaseModel):
    """Data governance configuration for professional deployments."""

    enabled: bool = Field(True, description="Whether data governance features are active")
    data_retention_years: int = Field(
        7, ge=1, le=25,
        description="Data retention period in years"
    )
    data_classification_enabled: bool = Field(
        True, description="Enable automatic data classification"
    )
    classification_levels: List[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential", "restricted"],
        description="Data classification levels"
    )
    gdpr_compliance: bool = Field(
        True, description="Enable GDPR compliance features"
    )
    gdpr_data_subject_access: bool = Field(
        True, description="Enable data subject access request handling"
    )
    gdpr_right_to_erasure: bool = Field(
        True, description="Enable right to erasure request handling"
    )
    data_lineage_tracking: bool = Field(
        True, description="Track full data lineage from source to report"
    )
    version_control_enabled: bool = Field(
        True, description="Enable version control for all data changes"
    )
    change_log_retention_years: int = Field(
        10, ge=1, description="Change log retention period"
    )


class WebhookConfig(BaseModel):
    """Webhook notification configuration."""

    enabled: bool = Field(False, description="Whether webhooks are active")
    endpoints: List[str] = Field(
        default_factory=list, description="Webhook endpoint URLs"
    )
    channels: List[str] = Field(
        default_factory=lambda: ["approval", "quality_gate", "regulatory_change"],
        description="Event channels to send via webhook"
    )
    hmac_secret: str = Field(
        "", description="HMAC secret for webhook payload signing"
    )
    retry_count: int = Field(3, ge=0, le=10, description="Retry count for failed deliveries")
    retry_delay_seconds: int = Field(30, ge=1, description="Delay between retries")
    timeout_seconds: int = Field(10, ge=1, le=60, description="Webhook request timeout")


class AssuranceConfig(BaseModel):
    """External assurance engagement configuration."""

    enabled: bool = Field(True, description="Whether assurance features are active")
    assurance_level: str = Field(
        "limited", description="Assurance level: limited, reasonable"
    )
    isae_standard: str = Field(
        "ISAE_3000", description="Applicable ISAE standard: ISAE_3000, ISAE_3410"
    )
    assurance_scope: List[str] = Field(
        default_factory=lambda: [
            "scope_1_emissions", "scope_2_emissions",
            "scope_3_material_categories",
            "esrs_e1_disclosures",
        ],
        description="Disclosures in scope for assurance"
    )
    evidence_format: str = Field(
        "structured", description="Evidence format: structured, narrative, hybrid"
    )
    auditor_access_portal: bool = Field(
        True, description="Enable read-only auditor access portal"
    )
    evidence_retention_years: int = Field(
        10, ge=5, description="Evidence retention period in years"
    )
    reasonable_assurance_roadmap_year: Optional[int] = Field(
        None, description="Target year for reasonable assurance (if currently limited)"
    )

    @field_validator("assurance_level")
    @classmethod
    def validate_assurance_level(cls, v: str) -> str:
        """Validate assurance level."""
        valid = {"limited", "reasonable"}
        if v not in valid:
            raise ValueError(f"Invalid assurance level '{v}'. Must be one of: {sorted(valid)}")
        return v


class IntensityMetricsConfig(BaseModel):
    """Intensity metrics configuration for normalized reporting."""

    enabled: bool = Field(True, description="Whether intensity metrics are active")
    metric_types: List[str] = Field(
        default_factory=lambda: [
            "per_revenue", "per_employee", "per_unit",
        ],
        description="Intensity metric types to calculate"
    )
    per_revenue_currency: str = Field(
        "EUR", description="Currency for per-revenue intensity (ISO 4217)"
    )
    per_revenue_unit: str = Field(
        "million", description="Revenue unit: thousand, million, billion"
    )
    per_unit_definitions: Dict[str, str] = Field(
        default_factory=lambda: {
            "manufacturing": "per_tonne_product",
            "energy": "per_mwh_generated",
            "financial": "per_million_aum",
            "technology": "per_server_rack",
            "transport": "per_tonne_km",
        },
        description="Sector-specific per-unit metric definitions"
    )
    baseline_year: Optional[int] = Field(
        None, description="Baseline year for intensity tracking"
    )
    reduction_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Intensity reduction targets by metric type (percentage)"
    )


# =============================================================================
# ESRS and Scope 3 Configuration (inherited from PACK-001, extended)
# =============================================================================


class ESRSStandardConfig(BaseModel):
    """Configuration for an individual ESRS standard."""

    id: str = Field(..., description="Standard ID (e.g., E1, S1, G1)")
    name: str = Field(..., description="Standard name")
    enabled: bool = Field(True, description="Whether this standard is active")
    mandatory: bool = Field(False, description="Whether this standard is always required")
    material: Optional[bool] = Field(None, description="Materiality assessment result")
    disclosure_requirements: List[str] = Field(
        default_factory=list, description="Active disclosure requirement IDs"
    )


class Scope3Config(BaseModel):
    """Configuration for Scope 3 category enablement."""

    enabled_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Enabled Scope 3 category numbers (1-15)",
    )
    priority_categories: List[int] = Field(
        default_factory=list,
        description="Priority categories for enhanced calculation",
    )
    screening_required: bool = Field(
        True, description="Whether Scope 3 screening is required"
    )
    materiality_threshold_percent: float = Field(
        5.0, description="Threshold below which a category may be excluded"
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
    """Configuration for guided/tutorial mode."""

    enabled: bool = Field(False, description="Whether guided mode is active")
    tutorial_mode: bool = Field(False, description="Show tutorial overlays")
    pre_populated_examples: bool = Field(False, description="Load example data")
    ai_assistance_level: str = Field(
        "standard", description="AI assistance level: minimal, standard, enhanced"
    )
    step_by_step: bool = Field(False, description="Enable step-by-step guidance")
    onboarding_extended: bool = Field(False, description="Extended onboarding flow")


class DemoConfig(BaseModel):
    """Demo mode configuration."""

    enabled: bool = Field(False, description="Whether demo mode is active")
    use_sample_data: bool = Field(False, description="Use bundled sample data")
    skip_external_apis: bool = Field(False, description="Skip calls to external APIs")
    mock_ai_responses: bool = Field(False, description="Use mocked AI responses")
    fast_execution: bool = Field(False, description="Skip delays and long operations")
    sample_company_profile: str = Field("", description="Path to sample company profile")
    sample_esg_data: str = Field("", description="Path to sample ESG dataset")
    sample_group_profile: str = Field("", description="Path to sample group profile (multi-entity)")
    sample_subsidiary_data: str = Field("", description="Path to sample subsidiary dataset")


# =============================================================================
# ProfessionalPackConfig - Combines all professional-tier configurations
# =============================================================================


class ProfessionalPackConfig(BaseModel):
    """
    Combined professional-tier configuration.

    This model groups all PACK-002-specific configuration that extends
    beyond what PACK-001 provides. It includes consolidation, approval
    workflows, quality gates, cross-framework alignment, scenario analysis,
    benchmarking, stakeholder engagement, regulatory monitoring, data
    governance, webhooks, assurance, and intensity metrics.
    """

    consolidation: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig,
        description="Multi-entity consolidation configuration"
    )
    approval: ApprovalConfig = Field(
        default_factory=ApprovalConfig,
        description="Approval workflow configuration"
    )
    quality_gates: QualityGateConfig = Field(
        default_factory=QualityGateConfig,
        description="Quality gate configuration"
    )
    cross_framework: CrossFrameworkConfig = Field(
        default_factory=CrossFrameworkConfig,
        description="Cross-framework alignment configuration"
    )
    scenarios: ScenarioConfig = Field(
        default_factory=ScenarioConfig,
        description="Scenario analysis configuration"
    )
    benchmarking: BenchmarkingConfig = Field(
        default_factory=BenchmarkingConfig,
        description="Benchmarking configuration"
    )
    stakeholder: StakeholderConfig = Field(
        default_factory=StakeholderConfig,
        description="Stakeholder engagement configuration"
    )
    regulatory: RegulatoryConfig = Field(
        default_factory=RegulatoryConfig,
        description="Regulatory change monitoring configuration"
    )
    data_governance: DataGovernanceConfig = Field(
        default_factory=DataGovernanceConfig,
        description="Data governance configuration"
    )
    webhooks: WebhookConfig = Field(
        default_factory=WebhookConfig,
        description="Webhook notification configuration"
    )
    assurance: AssuranceConfig = Field(
        default_factory=AssuranceConfig,
        description="External assurance configuration"
    )
    intensity_metrics: IntensityMetricsConfig = Field(
        default_factory=IntensityMetricsConfig,
        description="Intensity metrics configuration"
    )


# =============================================================================
# PresetConfig - Active preset state (inherited from PACK-001, extended)
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
        default_factory=list,
        description="Ordered list of emission factor sources to prioritize",
    )
    sector_specific: Dict[str, Any] = Field(
        default_factory=dict, description="Sector-specific configuration values"
    )


# =============================================================================
# CSRDProfessionalPackConfig - Top-level pack configuration
# =============================================================================


class CSRDProfessionalPackConfig(BaseModel):
    """
    Top-level CSRD Professional Pack configuration model.

    This model represents the fully merged and validated configuration
    for a CSRD Professional Pack deployment. It combines the base manifest,
    size preset, sector preset, professional configuration, and any
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
    professional: ProfessionalPackConfig = Field(
        default_factory=ProfessionalPackConfig,
        description="Professional-tier configuration"
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
            fw for fw, enabled in self.professional.cross_framework.enabled_frameworks.items()
            if enabled
        ]

    def get_enabled_scenarios(self) -> List[ScenarioDefinition]:
        """Return list of enabled scenario definitions."""
        return [
            s for s in self.professional.scenarios.scenarios
            if s.enabled
        ]


# =============================================================================
# PackConfig - Main configuration manager
# =============================================================================


class PackConfig:
    """
    Configuration manager for PACK-002 CSRD Professional Pack.

    Loads and merges configuration from multiple sources in the following
    priority order (later sources override earlier):

        1. Base pack.yaml manifest
        2. Size preset (enterprise_group, listed_company, financial_institution, multinational)
        3. Sector preset (manufacturing_pro, financial_services_pro, technology_pro, energy_pro, heavy_industry_pro)
        4. Environment variables (CSRD_PRO_* prefix)
        5. Runtime overrides

    Attributes:
        pack: The fully resolved CSRDProfessionalPackConfig instance.
        config_hash: SHA-256 hash of the resolved configuration for provenance.
        loaded_at: Timestamp when configuration was loaded.
        source_files: List of files loaded during configuration resolution.

    Example:
        >>> config = PackConfig.load(
        ...     size_preset="enterprise_group",
        ...     sector_preset="manufacturing_pro",
        ... )
        >>> print(config.pack.metadata.version)
        '1.0.0'
        >>> print(len(config.active_agents))
        93
    """

    VALID_SIZE_PRESETS = {
        "enterprise_group", "listed_company",
        "financial_institution", "multinational",
    }
    VALID_SECTOR_PRESETS = {
        "manufacturing_pro", "financial_services_pro",
        "technology_pro", "energy_pro", "heavy_industry_pro",
    }

    ENV_PREFIX = "CSRD_PRO_"

    def __init__(
        self,
        pack: CSRDProfessionalPackConfig,
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
            pack_dir: Path to the pack root directory. Defaults to PACK_BASE_DIR.
            size_preset: Size preset to apply.
            sector_preset: Sector preset to apply.
            overrides: Dictionary of runtime configuration overrides.
            demo_mode: If True, loads demo configuration.

        Returns:
            Fully resolved PackConfig instance.

        Raises:
            FileNotFoundError: If pack.yaml or preset files are missing.
            ValueError: If preset names are invalid or configuration is malformed.
        """
        start_time = datetime.now()
        pack_dir = Path(pack_dir) if pack_dir else PACK_BASE_DIR
        source_files: List[str] = []

        logger.info(
            "Loading CSRD Professional Pack configuration from %s",
            pack_dir,
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
            cls._validate_preset_name(sector_preset, cls.VALID_SECTOR_PRESETS, "sector")
            sector_path = pack_dir / "config" / "sectors" / f"{sector_preset}.yaml"
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
                "Applied %d environment variable overrides",
                len(env_overrides),
            )

        # Step 7: Apply runtime overrides
        if overrides:
            pack_config_dict = cls._deep_merge(pack_config_dict, overrides)
            logger.info("Applied runtime overrides")

        # Step 8: Validate and create typed config
        pack = CSRDProfessionalPackConfig(**pack_config_dict)

        # Step 9: Calculate configuration hash
        config_json = json.dumps(pack_config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        load_duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            "Professional Pack configuration loaded in %.1fms (hash: %s)",
            load_duration,
            config_hash[:12],
        )

        return cls(
            pack=pack,
            config_hash=config_hash,
            loaded_at=start_time,
            source_files=source_files,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def active_agents(self) -> List[str]:
        """Return list of all enabled agent IDs."""
        return self.pack.get_active_agent_ids()

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
        return self.pack.professional.consolidation.enabled

    @property
    def assurance_level(self) -> str:
        """Return current assurance level (limited or reasonable)."""
        return self.pack.professional.assurance.assurance_level

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_agent_config(self, agent_id: str) -> Optional[AgentComponentConfig]:
        """
        Retrieve configuration for a specific agent.

        Args:
            agent_id: The agent identifier (e.g., AGENT-MRV-001, GL-PRO-CONSOLIDATION).

        Returns:
            AgentComponentConfig if found, None otherwise.
        """
        return self.pack.components.find_agent(agent_id)

    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """
        Retrieve configuration for a specific workflow.

        Args:
            workflow_id: The workflow identifier (e.g., consolidated_reporting).

        Returns:
            WorkflowConfig if found, None otherwise.
        """
        return self.pack.workflows.get(workflow_id)

    def is_scope3_category_enabled(self, category: int) -> bool:
        """
        Check if a Scope 3 category is enabled.

        Args:
            category: Scope 3 category number (1-15).

        Returns:
            True if the category is enabled.
        """
        return category in self.pack.presets.scope3.enabled_categories

    def is_esrs_standard_enabled(self, standard_id: str) -> bool:
        """
        Check if an ESRS standard is enabled.

        Args:
            standard_id: ESRS standard identifier (e.g., E1, S1, G1).

        Returns:
            True if the standard is enabled.
        """
        for standard in self.pack.presets.esrs_standards:
            if standard.id == standard_id:
                return standard.enabled
        return False

    def is_framework_enabled(self, framework_id: str) -> bool:
        """
        Check if a cross-framework alignment is enabled.

        Args:
            framework_id: Framework identifier (e.g., cdp, tcfd, sbti, taxonomy).

        Returns:
            True if the framework is enabled.
        """
        return self.pack.professional.cross_framework.enabled_frameworks.get(
            framework_id, False
        )

    def is_scenario_enabled(self, scenario_id: str) -> bool:
        """
        Check if a climate scenario is enabled.

        Args:
            scenario_id: Scenario identifier (e.g., iea_nze, ngfs_orderly).

        Returns:
            True if the scenario is enabled.
        """
        for scenario in self.pack.professional.scenarios.scenarios:
            if scenario.id == scenario_id:
                return scenario.enabled
        return False

    def get_quality_gate(self, gate_id: str) -> Optional[QualityGate]:
        """
        Retrieve a quality gate by ID.

        Args:
            gate_id: Quality gate identifier.

        Returns:
            QualityGate if found, None otherwise.
        """
        for gate in self.pack.professional.quality_gates.gates:
            if gate.id == gate_id:
                return gate
        return None

    def get_entity(self, entity_id: str) -> Optional[EntityDefinition]:
        """
        Retrieve an entity definition by ID.

        Args:
            entity_id: Entity identifier.

        Returns:
            EntityDefinition if found, None otherwise.
        """
        for entity in self.pack.professional.consolidation.entities:
            if entity.id == entity_id:
                return entity
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full configuration to a dictionary."""
        return {
            "pack": self.pack.model_dump(),
            "config_hash": self.config_hash,
            "loaded_at": self.loaded_at.isoformat(),
            "source_files": self.source_files,
        }

    def get_provenance_hash(self) -> str:
        """
        Get SHA-256 hash of the configuration for audit provenance.

        Returns:
            SHA-256 hex digest of the configuration.
        """
        return self.config_hash

    # =========================================================================
    # Internal Methods
    # =========================================================================

    @staticmethod
    def _load_yaml_file(path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid YAML.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dictionary in {path}, got {type(data).__name__}")

        return data

    @staticmethod
    def _validate_preset_name(name: str, valid_names: set, preset_type: str) -> None:
        """
        Validate that a preset name is recognized.

        Args:
            name: The preset name to validate.
            valid_names: Set of valid preset names.
            preset_type: Type of preset (size, sector) for error messaging.

        Raises:
            ValueError: If the preset name is not valid.
        """
        if name not in valid_names:
            raise ValueError(
                f"Invalid {preset_type} preset '{name}'. "
                f"Valid options: {sorted(valid_names)}"
            )

    @classmethod
    def _parse_manifest(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw YAML manifest into the structure expected by CSRDProfessionalPackConfig.

        Args:
            raw: Raw parsed YAML dictionary from pack.yaml.

        Returns:
            Restructured dictionary for CSRDProfessionalPackConfig initialization.
        """
        result: Dict[str, Any] = {}

        # Metadata
        result["metadata"] = raw.get("metadata", {})

        # Components - restructure from YAML lists to Pydantic model format
        raw_components = raw.get("components", {})
        result["components"] = {}
        for group_key in [
            "apps", "data_agents", "quality_agents",
            "mrv_scope1", "mrv_scope2", "mrv_scope3", "foundation",
            "professional_apps", "cdp_engines", "tcfd_engines",
            "sbti_engines", "taxonomy_engines", "professional_engines",
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

        # Performance targets - flatten from nested YAML
        raw_perf = raw.get("performance", {})
        result["performance"] = {
            "data_ingestion_rps": raw_perf.get("data_ingestion", {}).get(
                "throughput_records_per_second", 10000
            ),
            "data_ingestion_max_latency_ms": raw_perf.get("data_ingestion", {}).get(
                "max_latency_ms", 150
            ),
            "ghg_single_scope_max_seconds": raw_perf.get("ghg_calculation", {}).get(
                "single_scope_max_seconds", 45
            ),
            "ghg_full_inventory_max_seconds": raw_perf.get("ghg_calculation", {}).get(
                "full_inventory_max_seconds", 600
            ),
            "ghg_multi_entity_max_minutes": raw_perf.get("ghg_calculation", {}).get(
                "multi_entity_max_minutes", 45
            ),
            "ghg_max_data_points": raw_perf.get("ghg_calculation", {}).get(
                "max_data_points", 50000
            ),
            "report_esrs_max_seconds": raw_perf.get("report_generation", {}).get(
                "esrs_disclosure_max_seconds", 180
            ),
            "report_summary_max_seconds": raw_perf.get("report_generation", {}).get(
                "executive_summary_max_seconds", 90
            ),
            "report_xbrl_max_seconds": raw_perf.get("report_generation", {}).get(
                "xbrl_tagging_max_seconds", 300
            ),
            "report_consolidated_max_seconds": raw_perf.get("report_generation", {}).get(
                "consolidated_report_max_seconds", 600
            ),
            "report_cross_framework_max_seconds": raw_perf.get("report_generation", {}).get(
                "cross_framework_max_seconds", 240
            ),
            "scenario_single_max_seconds": raw_perf.get("scenario_analysis", {}).get(
                "single_scenario_max_seconds", 120
            ),
            "scenario_full_max_minutes": raw_perf.get("scenario_analysis", {}).get(
                "full_analysis_max_minutes", 30
            ),
            "scenario_monte_carlo_iterations": raw_perf.get("scenario_analysis", {}).get(
                "monte_carlo_iterations", 10000
            ),
            "api_p50_ms": raw_perf.get("api_response", {}).get("p50_ms", 80),
            "api_p95_ms": raw_perf.get("api_response", {}).get("p95_ms", 400),
            "api_p99_ms": raw_perf.get("api_response", {}).get("p99_ms", 1500),
            "quality_gate_max_seconds": raw_perf.get("data_quality", {}).get(
                "quality_gate_max_seconds", 60
            ),
            "availability_percent": raw_perf.get("availability", {}).get(
                "target_uptime_percent", 99.95
            ),
        }

        # Requirements - flatten from nested YAML
        raw_req = raw.get("requirements", {})
        raw_runtime = raw_req.get("runtime", {})
        raw_infra = raw_req.get("infrastructure", {})
        raw_db = raw_req.get("database", {})
        result["requirements"] = {
            "python_version": raw_runtime.get("python", ">=3.11"),
            "postgresql_version": raw_runtime.get("postgresql", ">=14"),
            "redis_version": raw_runtime.get("redis", ">=7"),
            "min_cpu_cores": raw_infra.get("min_cpu_cores", 8),
            "min_memory_gb": raw_infra.get("min_memory_gb", 32),
            "min_storage_gb": raw_infra.get("min_storage_gb", 250),
            "recommended_cpu_cores": raw_infra.get("recommended_cpu_cores", 16),
            "recommended_memory_gb": raw_infra.get("recommended_memory_gb", 64),
            "recommended_storage_gb": raw_infra.get("recommended_storage_gb", 1000),
            "database_extensions": raw_db.get("extensions", ["pgvector", "timescaledb"]),
            "min_db_connections": raw_db.get("min_connections", 50),
        }

        # Presets - initialize empty, filled by preset loading
        result["presets"] = {}

        # Professional config - initialize with defaults
        result["professional"] = {}

        return result

    @classmethod
    def _merge_preset(
        cls,
        base: Dict[str, Any],
        preset_data: Dict[str, Any],
        preset_type: str,
        preset_id: str,
    ) -> Dict[str, Any]:
        """
        Merge a preset configuration into the base configuration.

        Args:
            base: Current configuration dictionary.
            preset_data: Preset configuration to merge.
            preset_type: Type of preset (size, sector).
            preset_id: Identifier of the preset.

        Returns:
            Merged configuration dictionary.
        """
        result = cls._deep_merge(base, {})

        # Initialize presets if missing
        if "presets" not in result:
            result["presets"] = {}

        # Set preset identifier
        if preset_type == "size":
            result["presets"]["size_preset_id"] = preset_id
        elif preset_type == "sector":
            result["presets"]["sector_preset_id"] = preset_id

        # Merge ESRS standards
        if "esrs_standards" in preset_data:
            result["presets"]["esrs_standards"] = preset_data["esrs_standards"]

        # Merge Scope 3 configuration
        if "scope3" in preset_data:
            result["presets"]["scope3"] = preset_data["scope3"]

        # Merge XBRL configuration
        if "xbrl" in preset_data:
            result["presets"]["xbrl"] = preset_data["xbrl"]

        # Merge consolidation configuration (presets level for backward compat)
        if "consolidation" in preset_data:
            result["presets"]["consolidation"] = preset_data["consolidation"]

        # Merge guided mode configuration
        if "guided_mode" in preset_data:
            result["presets"]["guided_mode"] = preset_data["guided_mode"]

        # Merge demo configuration
        if "demo" in preset_data:
            result["presets"]["demo"] = preset_data["demo"]

        # Merge emission factor priorities
        if "emission_factor_priorities" in preset_data:
            result["presets"]["emission_factor_priorities"] = preset_data[
                "emission_factor_priorities"
            ]

        # Merge sector-specific configuration
        if "sector_specific" in preset_data:
            existing_sector = result["presets"].get("sector_specific", {})
            result["presets"]["sector_specific"] = cls._deep_merge(
                existing_sector, preset_data["sector_specific"]
            )

        # Apply agent overrides (enable/disable specific agents)
        if "agent_overrides" in preset_data:
            result = cls._apply_agent_overrides(result, preset_data["agent_overrides"])

        # Merge performance overrides
        if "performance" in preset_data:
            result["performance"] = cls._deep_merge(
                result.get("performance", {}), preset_data["performance"]
            )

        # Merge professional configuration overrides
        if "professional" in preset_data:
            if "professional" not in result:
                result["professional"] = {}
            result["professional"] = cls._deep_merge(
                result["professional"], preset_data["professional"]
            )

        return result

    @classmethod
    def _merge_demo(
        cls, base: Dict[str, Any], demo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge demo configuration into base configuration.

        Args:
            base: Current configuration dictionary.
            demo_data: Demo configuration to merge.

        Returns:
            Merged configuration dictionary.
        """
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
        }

        # Apply demo-specific agent overrides if present
        if "agent_overrides" in demo_data:
            result = cls._apply_agent_overrides(result, demo_data["agent_overrides"])

        # Apply demo-specific professional overrides
        if "professional" in demo_data:
            if "professional" not in result:
                result["professional"] = {}
            result["professional"] = cls._deep_merge(
                result["professional"], demo_data["professional"]
            )

        return result

    @classmethod
    def _apply_agent_overrides(
        cls,
        config: Dict[str, Any],
        overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply agent-level overrides to component configuration.

        Args:
            config: Current configuration dictionary.
            overrides: Agent override specifications keyed by agent ID.

        Returns:
            Updated configuration dictionary.
        """
        components = config.get("components", {})

        for group_key in [
            "apps", "data_agents", "quality_agents",
            "mrv_scope1", "mrv_scope2", "mrv_scope3", "foundation",
            "professional_apps", "cdp_engines", "tcfd_engines",
            "sbti_engines", "taxonomy_engines", "professional_engines",
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
        """
        Load configuration overrides from environment variables.

        Environment variables with the prefix CSRD_PRO_ are parsed as
        configuration overrides. The variable name is converted to a
        nested dictionary path using double underscores as separators.

        Examples:
            CSRD_PRO_PERFORMANCE__API_P50_MS=50
            CSRD_PRO_PROFESSIONAL__ASSURANCE__ASSURANCE_LEVEL=reasonable

        Returns:
            Dictionary of environment-based overrides.
        """
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
        """
        Parse an environment variable value to its appropriate Python type.

        Args:
            value: Raw string value from environment.

        Returns:
            Parsed value (bool, int, float, or str).
        """
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
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries. Override values take precedence.

        Lists are replaced (not appended). Nested dicts are recursively merged.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            New merged dictionary.
        """
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
            f"hash={self.config_hash[:12]})"
        )
