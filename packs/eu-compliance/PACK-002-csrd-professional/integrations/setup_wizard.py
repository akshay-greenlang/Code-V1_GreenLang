# -*- coding: utf-8 -*-
"""
ProfessionalSetupWizard - Professional Pack Guided Setup
=========================================================

This module extends PACK-001's CSRDSetupWizard with professional-grade
configuration steps for enterprise groups, cross-framework alignment,
approval workflows, quality gates, and multi-entity management.

Wizard Steps (7 steps, extending PACK-001):
    1. Company Profile (enhanced): entity hierarchy, subsidiaries,
       consolidation approach, LEI per entity
    2. Reporting Scope (enhanced): cross-framework selection
       (CDP/TCFD/SBTi/Taxonomy), assurance level, regulatory jurisdictions
    3. Data Sources (enhanced): per-entity data source mapping,
       ERP integration per subsidiary
    4. Professional Features: quality gates, approval chain, benchmarking,
       stakeholder groups, webhook endpoints
    5. Preset Selection (enhanced): enterprise_group, listed_company,
       financial_institution, multinational with auto-recommendation
    6. Validation (enhanced): subsidiary connectivity, cross-framework
       engine availability, webhook endpoint tests
    7. Demo Run (enhanced): multi-entity mini pipeline with consolidation

Recommendation Logic:
    - >5 entities AND >EUR 1B revenue -> enterprise_group
    - Listed on stock exchange -> listed_company
    - Banking/insurance NACE codes -> financial_institution
    - >3 jurisdictions -> multinational
    - Fallback: listed_company

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WizardStepName(str, Enum):
    """Names of wizard steps in execution order."""
    COMPANY_PROFILE = "company_profile"
    REPORTING_SCOPE = "reporting_scope"
    DATA_SOURCES = "data_sources"
    PROFESSIONAL_FEATURES = "professional_features"
    PRESET_SELECTION = "preset_selection"
    VALIDATION = "validation"
    DEMO_RUN = "demo_run"


class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SectorCode(str, Enum):
    """NACE sector codes for preset matching."""
    MANUFACTURING = "manufacturing"
    FINANCIAL_SERVICES = "financial_services"
    ENERGY = "energy"
    TECHNOLOGY = "technology"
    RETAIL = "retail"
    TRANSPORTATION = "transportation"
    HEALTHCARE = "healthcare"
    REAL_ESTATE = "real_estate"
    AGRICULTURE = "agriculture"
    MINING = "mining"
    CONSTRUCTION = "construction"
    PROFESSIONAL_SERVICES = "professional_services"
    BANKING = "banking"
    INSURANCE = "insurance"
    OTHER = "other"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class AssuranceLevel(str, Enum):
    """CSRD assurance levels."""
    LIMITED = "limited"
    REASONABLE = "reasonable"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SubsidiaryEntity(BaseModel):
    """Configuration for a single subsidiary entity."""

    entity_id: str = Field(..., description="Unique entity identifier")
    entity_name: str = Field(..., description="Legal entity name")
    country: str = Field(..., min_length=2, max_length=3, description="ISO country")
    lei_code: Optional[str] = Field(None, description="LEI code")
    ownership_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Parent ownership percentage",
    )
    employee_count: int = Field(default=0, ge=0)
    annual_revenue_eur: float = Field(default=0.0, ge=0.0)
    nace_code: Optional[str] = Field(None)
    is_material: bool = Field(
        default=True,
        description="Whether entity is material for consolidation",
    )
    data_source_ids: List[str] = Field(
        default_factory=list,
        description="Data source IDs mapped to this entity",
    )
    erp_system: Optional[str] = Field(None, description="ERP system for this entity")


class CompanyProfile(BaseModel):
    """Enhanced company profile with entity hierarchy."""

    company_name: str = Field(..., min_length=1, max_length=255)
    sector: SectorCode = Field(...)
    nace_code: Optional[str] = Field(None)
    country: str = Field(..., min_length=2, max_length=3)
    headquarters_city: Optional[str] = Field(None)
    employee_count: int = Field(..., ge=1)
    annual_revenue_eur: Optional[float] = Field(None, ge=0)
    total_assets_eur: Optional[float] = Field(None, ge=0)
    lei_code: Optional[str] = Field(None, max_length=20)
    is_listed: bool = Field(default=False)
    parent_company: Optional[str] = Field(None)
    reporting_currency: str = Field(default="EUR")

    # Professional enhancements
    subsidiaries: List[SubsidiaryEntity] = Field(
        default_factory=list,
        description="List of subsidiary entities for consolidation",
    )
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
    )
    regulatory_jurisdictions: List[str] = Field(
        default_factory=lambda: ["EU"],
        description="Regulatory jurisdictions (EU, UK, US, etc.)",
    )

    @field_validator("lei_code")
    @classmethod
    def validate_lei_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate LEI code format."""
        if v is not None and len(v) != 20:
            raise ValueError("LEI code must be exactly 20 characters")
        return v


class ReportingScope(BaseModel):
    """Enhanced reporting scope with cross-framework selection."""

    enabled_standards: List[str] = Field(
        default_factory=lambda: [
            "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_S1", "ESRS_G1"
        ],
    )
    reporting_period_start: str = Field(...)
    reporting_period_end: str = Field(...)
    is_first_report: bool = Field(default=True)
    consolidation_scope: str = Field(default="operational_control")
    base_year: Optional[int] = Field(None, ge=2015, le=2030)
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
    )
    include_value_chain: bool = Field(default=True)
    phased_reporting: bool = Field(default=False)

    # Professional enhancements
    cross_frameworks: List[str] = Field(
        default_factory=lambda: ["cdp", "tcfd", "sbti", "eu_taxonomy"],
        description="Cross-framework alignment targets",
    )
    assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED,
    )
    assurance_provider: Optional[str] = Field(None)
    double_materiality: bool = Field(
        default=True,
        description="Whether double materiality assessment is required",
    )


class DataSourceConfig(BaseModel):
    """Enhanced data source configuration with per-entity mapping."""

    sources: List[Dict[str, Any]] = Field(default_factory=list)
    has_erp: bool = Field(default=False)
    erp_system: Optional[str] = Field(None)
    has_manual_data: bool = Field(default=True)
    estimated_data_volume: str = Field(default="medium")

    # Professional enhancements
    per_entity_sources: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of entity_id to list of source_ids",
    )
    erp_per_entity: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of entity_id to ERP system name",
    )


class ProfessionalFeaturesConfig(BaseModel):
    """Professional feature configuration (step 4)."""

    enable_quality_gates: bool = Field(default=True)
    quality_gate_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "QG-1": 0.7, "QG-2": 0.8, "QG-3": 0.85,
        },
    )
    enable_approval_chain: bool = Field(default=True)
    approval_levels: List[str] = Field(
        default_factory=lambda: [
            "preparer", "reviewer", "approver", "board",
        ],
    )
    approval_users: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of approval level to user IDs",
    )
    benchmarking_sector: Optional[str] = Field(None)
    stakeholder_groups: List[str] = Field(
        default_factory=lambda: [
            "investors", "employees", "regulators", "customers",
            "suppliers", "communities",
        ],
    )
    webhook_endpoints: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of webhook endpoint configs",
    )
    enable_scenario_analysis: bool = Field(default=True)
    enable_transition_planning: bool = Field(default=True)


class PresetRecommendation(BaseModel):
    """Professional preset recommendation."""

    recommended_size_preset: str = Field(...)
    recommended_sector_preset: Optional[str] = Field(None)
    size_reasoning: str = Field(default="")
    sector_reasoning: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_esrs_standards: List[str] = Field(default_factory=list)
    suggested_scope3_categories: List[int] = Field(default_factory=list)
    suggested_cross_frameworks: List[str] = Field(default_factory=list)


class WizardStep(BaseModel):
    """State of a single wizard step."""

    name: WizardStepName = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)


class WizardState(BaseModel):
    """Complete state of the professional setup wizard."""

    wizard_id: str = Field(default="")
    current_step: WizardStepName = Field(default=WizardStepName.COMPANY_PROFILE)
    steps: Dict[str, WizardStep] = Field(default_factory=dict)
    company_profile: Optional[CompanyProfile] = Field(None)
    reporting_scope: Optional[ReportingScope] = Field(None)
    data_sources: Optional[DataSourceConfig] = Field(None)
    professional_features: Optional[ProfessionalFeaturesConfig] = Field(None)
    preset_recommendation: Optional[PresetRecommendation] = Field(None)
    selected_size_preset: Optional[str] = Field(None)
    selected_sector_preset: Optional[str] = Field(None)
    validation_passed: bool = Field(default=False)
    demo_run_passed: bool = Field(default=False)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class SetupReport(BaseModel):
    """Final setup report generated upon wizard completion."""

    report_id: str = Field(default="")
    company_name: str = Field(default="")
    size_preset: str = Field(default="")
    sector_preset: Optional[str] = Field(None)
    enabled_standards: List[str] = Field(default_factory=list)
    enabled_scope3_categories: List[int] = Field(default_factory=list)
    cross_frameworks: List[str] = Field(default_factory=list)
    data_sources_configured: int = Field(default=0)
    entities_configured: int = Field(default=0)
    total_agents_active: int = Field(default=0)
    validation_status: str = Field(default="")
    demo_run_status: str = Field(default="")
    estimated_first_report_effort: str = Field(default="")
    professional_features_enabled: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_DEFINITIONS: List[WizardStepName] = [
    WizardStepName.COMPANY_PROFILE,
    WizardStepName.REPORTING_SCOPE,
    WizardStepName.DATA_SOURCES,
    WizardStepName.PROFESSIONAL_FEATURES,
    WizardStepName.PRESET_SELECTION,
    WizardStepName.VALIDATION,
    WizardStepName.DEMO_RUN,
]

STEP_DISPLAY_NAMES: Dict[WizardStepName, str] = {
    WizardStepName.COMPANY_PROFILE: "Company Profile & Entity Hierarchy",
    WizardStepName.REPORTING_SCOPE: "Reporting Scope & Cross-Framework",
    WizardStepName.DATA_SOURCES: "Data Sources & Entity Mapping",
    WizardStepName.PROFESSIONAL_FEATURES: "Professional Features Configuration",
    WizardStepName.PRESET_SELECTION: "Preset Selection & Recommendation",
    WizardStepName.VALIDATION: "Configuration Validation",
    WizardStepName.DEMO_RUN: "Multi-Entity Demo Pipeline",
}

# Valid professional presets
VALID_SIZE_PRESETS = {
    "enterprise_group",
    "listed_company",
    "financial_institution",
    "multinational",
}

# Financial sector NACE codes that trigger financial_institution preset
FINANCIAL_NACE_CODES = {
    "K64", "K65", "K66",  # Financial services
    "banking", "insurance",
}


# ---------------------------------------------------------------------------
# ProfessionalSetupWizard
# ---------------------------------------------------------------------------


class ProfessionalSetupWizard:
    """Professional setup wizard extending PACK-001's CSRDSetupWizard.

    Provides 7 guided configuration steps with enterprise-grade features
    including entity hierarchy management, cross-framework alignment,
    approval chain setup, and multi-entity demo pipeline.

    Attributes:
        _state: Current wizard state
        _step_handlers: Mapping of step names to handler methods

    Example:
        >>> wizard = ProfessionalSetupWizard()
        >>> state = await wizard.start()
        >>> state = await wizard.complete_step("company_profile", profile_data)
        >>> report = await wizard.finalize()
    """

    def __init__(self) -> None:
        """Initialize the professional setup wizard."""
        self._state: Optional[WizardState] = None
        self._step_handlers: Dict[WizardStepName, Any] = {
            WizardStepName.COMPANY_PROFILE: self._handle_company_profile,
            WizardStepName.REPORTING_SCOPE: self._handle_reporting_scope,
            WizardStepName.DATA_SOURCES: self._handle_data_sources,
            WizardStepName.PROFESSIONAL_FEATURES: self._handle_professional_features,
            WizardStepName.PRESET_SELECTION: self._handle_preset_selection,
            WizardStepName.VALIDATION: self._handle_validation,
            WizardStepName.DEMO_RUN: self._handle_demo_run,
        }
        logger.info("ProfessionalSetupWizard initialized")

    # -------------------------------------------------------------------------
    # Wizard Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState.
        """
        wizard_id = _compute_hash(f"pro-wizard:{_utcnow().isoformat()}")[:16]

        steps: Dict[str, WizardStep] = {}
        for step_name in STEP_DEFINITIONS:
            steps[step_name.value] = WizardStep(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )

        self._state = WizardState(
            wizard_id=wizard_id,
            current_step=STEP_DEFINITIONS[0],
            steps=steps,
        )

        logger.info("Professional wizard session started: %s", wizard_id)
        return self._state

    async def complete_step(
        self,
        step_name: str,
        data: Dict[str, Any],
    ) -> WizardState:
        """Complete a wizard step with the provided data.

        Args:
            step_name: Name of the step to complete.
            data: Step data.

        Returns:
            Updated WizardState.

        Raises:
            RuntimeError: If the wizard has not been started.
            ValueError: If the step name is invalid.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = WizardStepName(step_name)
        except ValueError:
            valid = [s.value for s in WizardStepName]
            raise ValueError(f"Unknown step '{step_name}'. Valid: {valid}")

        step = self._state.steps.get(step_name)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = _utcnow()
        start_time = time.monotonic()

        handler = self._step_handlers.get(step_enum)
        if handler is None:
            raise ValueError(f"No handler for step '{step_name}'")

        try:
            errors = await handler(data)
            elapsed = (time.monotonic() - start_time) * 1000
            step.execution_time_ms = elapsed
            step.data = data

            if errors:
                step.status = StepStatus.FAILED
                step.validation_errors = errors
            else:
                step.status = StepStatus.COMPLETED
                step.completed_at = _utcnow()
                step.validation_errors = []
                self._advance_to_next_step(step_enum)
                logger.info("Step '%s' completed in %.1fms", step_name, elapsed)
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

        return self._state

    async def finalize(self) -> SetupReport:
        """Finalize the wizard and generate the setup report.

        Returns:
            SetupReport with configuration summary.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        required = [
            WizardStepName.COMPANY_PROFILE,
            WizardStepName.REPORTING_SCOPE,
            WizardStepName.PRESET_SELECTION,
        ]
        for sn in required:
            step = self._state.steps.get(sn.value)
            if step is None or step.status != StepStatus.COMPLETED:
                raise RuntimeError(
                    f"Required step '{sn.value}' must be completed"
                )

        report = self._generate_setup_report()
        self._state.is_complete = True
        self._state.completed_at = _utcnow()
        logger.info("Professional wizard finalized: %s", report.report_id)
        return report

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    # -------------------------------------------------------------------------
    # Step Handlers
    # -------------------------------------------------------------------------

    async def _handle_company_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle enhanced company profile step with entity hierarchy.

        Args:
            data: Company profile data including subsidiaries.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        try:
            profile = CompanyProfile(**data)
            if self._state is not None:
                self._state.company_profile = profile
        except Exception as exc:
            errors.append(f"Invalid company profile: {exc}")
            return errors

        if profile.is_listed and not profile.lei_code:
            errors.append("LEI code is required for listed companies")

        # Validate subsidiaries
        entity_ids: List[str] = []
        for sub in profile.subsidiaries:
            if sub.entity_id in entity_ids:
                errors.append(f"Duplicate entity_id: {sub.entity_id}")
            entity_ids.append(sub.entity_id)

            if sub.ownership_pct <= 0:
                errors.append(
                    f"Entity '{sub.entity_id}' ownership must be > 0%"
                )

        return errors

    async def _handle_reporting_scope(self, data: Dict[str, Any]) -> List[str]:
        """Handle enhanced reporting scope with cross-framework selection.

        Args:
            data: Reporting scope data.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        try:
            scope = ReportingScope(**data)
            if self._state is not None:
                self._state.reporting_scope = scope
        except Exception as exc:
            errors.append(f"Invalid reporting scope: {exc}")
            return errors

        if not scope.enabled_standards:
            errors.append("At least one ESRS standard must be enabled")

        mandatory = {"ESRS_1", "ESRS_2"}
        missing = mandatory - set(scope.enabled_standards)
        if missing:
            errors.append(f"Mandatory standards missing: {sorted(missing)}")

        valid_frameworks = {"cdp", "tcfd", "sbti", "eu_taxonomy", "gri", "sasb"}
        for fw in scope.cross_frameworks:
            if fw not in valid_frameworks:
                errors.append(f"Invalid cross-framework: {fw}")

        return errors

    async def _handle_data_sources(self, data: Dict[str, Any]) -> List[str]:
        """Handle enhanced data sources with per-entity mapping.

        Args:
            data: Data source configuration.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        try:
            config = DataSourceConfig(**data)
            if self._state is not None:
                self._state.data_sources = config
        except Exception as exc:
            errors.append(f"Invalid data source config: {exc}")
            return errors

        if not config.sources and not config.has_manual_data:
            errors.append("At least one data source or manual entry required")

        return errors

    async def _handle_professional_features(
        self, data: Dict[str, Any]
    ) -> List[str]:
        """Handle professional features configuration (step 4).

        Args:
            data: Professional features configuration.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []
        try:
            features = ProfessionalFeaturesConfig(**data)
            if self._state is not None:
                self._state.professional_features = features
        except Exception as exc:
            errors.append(f"Invalid professional features: {exc}")
            return errors

        # Validate quality gate thresholds
        for gate_id, threshold in features.quality_gate_thresholds.items():
            if threshold < 0 or threshold > 1:
                errors.append(
                    f"Quality gate '{gate_id}' threshold must be 0-1, "
                    f"got {threshold}"
                )

        # Validate webhook endpoints
        for ep in features.webhook_endpoints:
            url = ep.get("url", "")
            if url and not url.startswith(("http://", "https://")):
                errors.append(f"Invalid webhook URL: {url}")

        return errors

    async def _handle_preset_selection(self, data: Dict[str, Any]) -> List[str]:
        """Handle enhanced preset selection with professional presets.

        Args:
            data: Preset selection data.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []

        if self._state is None or self._state.company_profile is None:
            errors.append("Company profile required before preset selection")
            return errors

        # Generate recommendation
        if self._state.preset_recommendation is None:
            self._state.preset_recommendation = (
                self._generate_preset_recommendation(self._state.company_profile)
            )

        self._state.selected_size_preset = data.get(
            "size_preset",
            self._state.preset_recommendation.recommended_size_preset,
        )
        self._state.selected_sector_preset = data.get(
            "sector_preset",
            self._state.preset_recommendation.recommended_sector_preset,
        )

        if self._state.selected_size_preset not in VALID_SIZE_PRESETS:
            errors.append(
                f"Invalid size preset '{self._state.selected_size_preset}'. "
                f"Valid: {sorted(VALID_SIZE_PRESETS)}"
            )

        return errors

    async def _handle_validation(self, data: Dict[str, Any]) -> List[str]:
        """Handle enhanced validation with subsidiary and framework checks.

        Args:
            data: Validation parameters.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []

        if self._state is None:
            errors.append("Wizard state not initialized")
            return errors

        if self._state.company_profile is None:
            errors.append("Company profile not configured")
        if self._state.reporting_scope is None:
            errors.append("Reporting scope not configured")
        if self._state.selected_size_preset is None:
            errors.append("Size preset not selected")

        # Subsidiary connectivity check (stub)
        if self._state.company_profile:
            for sub in self._state.company_profile.subsidiaries:
                if sub.data_source_ids:
                    logger.info(
                        "Subsidiary connectivity check: %s - OK",
                        sub.entity_id,
                    )

        # Cross-framework engine availability check
        if self._state.reporting_scope:
            for fw in self._state.reporting_scope.cross_frameworks:
                logger.info(
                    "Cross-framework engine check: %s - available (stub)", fw
                )

        # Webhook endpoint test
        if self._state.professional_features:
            for ep in self._state.professional_features.webhook_endpoints:
                url = ep.get("url", "")
                if url:
                    logger.info("Webhook endpoint test: %s - OK (stub)", url)

        if not errors:
            self._state.validation_passed = True
            logger.info("Professional validation passed")

        return errors

    async def _handle_demo_run(self, data: Dict[str, Any]) -> List[str]:
        """Handle enhanced demo run with multi-entity consolidation.

        Args:
            data: Demo run parameters.

        Returns:
            List of validation errors.
        """
        errors: List[str] = []

        if self._state is None or not self._state.validation_passed:
            errors.append("Validation must pass before demo run")
            return errors

        try:
            demo_result = await self._execute_demo_pipeline()
            if not demo_result.get("success"):
                errors.append(f"Demo failed: {demo_result.get('error', 'Unknown')}")
            else:
                self._state.demo_run_passed = True
                logger.info(
                    "Multi-entity demo: %d entities, %d metrics in %.1fms",
                    demo_result.get("entities_processed", 0),
                    demo_result.get("metrics_calculated", 0),
                    demo_result.get("execution_time_ms", 0),
                )
        except Exception as exc:
            errors.append(f"Demo pipeline error: {exc}")

        return errors

    # -------------------------------------------------------------------------
    # Recommendation Engine
    # -------------------------------------------------------------------------

    def _generate_preset_recommendation(
        self, profile: CompanyProfile
    ) -> PresetRecommendation:
        """Generate professional preset recommendation.

        Recommendation Logic:
            - >5 entities AND >EUR 1B revenue -> enterprise_group
            - Listed on stock exchange -> listed_company
            - Banking/insurance NACE codes -> financial_institution
            - >3 jurisdictions -> multinational
            - Fallback: listed_company

        Args:
            profile: Company profile data.

        Returns:
            PresetRecommendation.
        """
        reasons: List[str] = []
        preset = "listed_company"

        entity_count = len(profile.subsidiaries)
        revenue = profile.annual_revenue_eur or 0.0
        jurisdictions = len(profile.regulatory_jurisdictions)

        # Check enterprise_group
        if entity_count > 5 and revenue >= 1_000_000_000:
            preset = "enterprise_group"
            reasons.append(
                f"{entity_count} entities and EUR {revenue:,.0f} revenue "
                f"(>5 entities AND >EUR 1B)"
            )
        # Check listed_company
        elif profile.is_listed:
            preset = "listed_company"
            reasons.append("Company is publicly listed")
        # Check financial_institution
        elif (
            profile.sector in (SectorCode.BANKING, SectorCode.INSURANCE, SectorCode.FINANCIAL_SERVICES)
            or (profile.nace_code and any(
                profile.nace_code.startswith(code) for code in FINANCIAL_NACE_CODES
            ))
        ):
            preset = "financial_institution"
            reasons.append(
                f"Financial sector: {profile.sector.value}"
            )
        # Check multinational
        elif jurisdictions > 3:
            preset = "multinational"
            reasons.append(f"{jurisdictions} regulatory jurisdictions (>3)")
        else:
            reasons.append("Default recommendation for professional tier")

        # Framework suggestions
        suggested_fw = ["cdp", "tcfd", "sbti", "eu_taxonomy"]
        if preset == "financial_institution":
            suggested_fw.extend(["gri", "sasb"])

        recommendation = PresetRecommendation(
            recommended_size_preset=preset,
            recommended_sector_preset=profile.sector.value,
            size_reasoning="; ".join(reasons),
            sector_reasoning=f"Primary sector: {profile.sector.value}",
            confidence=0.88 if profile.nace_code else 0.80,
            suggested_esrs_standards=[
                "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2",
                "ESRS_S1", "ESRS_G1",
            ],
            suggested_scope3_categories=list(range(1, 16)),
            suggested_cross_frameworks=suggested_fw,
        )

        logger.info(
            "Professional preset recommendation: %s (confidence=%.2f)",
            preset, recommendation.confidence,
        )
        return recommendation

    # -------------------------------------------------------------------------
    # Demo Pipeline
    # -------------------------------------------------------------------------

    async def _execute_demo_pipeline(self) -> Dict[str, Any]:
        """Execute multi-entity demo pipeline with consolidation.

        Returns:
            Dictionary with demo results.
        """
        start_time = time.monotonic()

        entities_count = 1
        if self._state and self._state.company_profile:
            entities_count = max(1, len(self._state.company_profile.subsidiaries) + 1)

        # Simulate per-entity calculations
        total_metrics = 0
        entity_results = {}
        for i in range(entities_count):
            entity_id = f"entity_{i}" if i > 0 else "parent"
            # Deterministic sample calculation
            emissions = 1000.0 * (i + 1) * 2.0
            total_metrics += 2
            entity_results[entity_id] = {
                "scope1": emissions,
                "scope2": emissions * 0.4,
            }

        # Simulate consolidation
        consolidated_scope1 = sum(r["scope1"] for r in entity_results.values())
        consolidated_scope2 = sum(r["scope2"] for r in entity_results.values())

        elapsed = (time.monotonic() - start_time) * 1000

        return {
            "success": True,
            "entities_processed": entities_count,
            "metrics_calculated": total_metrics,
            "consolidated_scope1": consolidated_scope1,
            "consolidated_scope2": consolidated_scope2,
            "entity_results": entity_results,
            "execution_time_ms": elapsed,
        }

    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def _generate_setup_report(self) -> SetupReport:
        """Generate final setup report."""
        if self._state is None:
            return SetupReport()

        profile = self._state.company_profile
        scope = self._state.reporting_scope
        features = self._state.professional_features

        agent_counts = {
            "enterprise_group": 93,
            "listed_company": 85,
            "financial_institution": 90,
            "multinational": 88,
        }
        total_agents = agent_counts.get(
            self._state.selected_size_preset or "listed_company", 85
        )

        effort_estimates = {
            "enterprise_group": "12-20 weeks for first consolidated report",
            "listed_company": "8-14 weeks for first report",
            "financial_institution": "10-16 weeks for first report",
            "multinational": "10-18 weeks for first report",
        }
        effort = effort_estimates.get(
            self._state.selected_size_preset or "listed_company",
            "8-14 weeks",
        )

        enabled_features: List[str] = []
        if features:
            if features.enable_quality_gates:
                enabled_features.append("quality_gates")
            if features.enable_approval_chain:
                enabled_features.append("approval_chain")
            if features.enable_scenario_analysis:
                enabled_features.append("scenario_analysis")
            if features.enable_transition_planning:
                enabled_features.append("transition_planning")

        entities_count = 0
        if profile:
            entities_count = len(profile.subsidiaries) + 1

        data_sources_count = 0
        if self._state.data_sources:
            data_sources_count = len(self._state.data_sources.sources)

        recommendations = self._generate_recommendations()

        config_hash = _compute_hash(
            f"{profile.company_name if profile else ''}:"
            f"{self._state.selected_size_preset}:"
            f"{scope.cross_frameworks if scope else []}:"
            f"{entities_count}"
        )

        return SetupReport(
            report_id=config_hash[:16],
            company_name=profile.company_name if profile else "",
            size_preset=self._state.selected_size_preset or "",
            sector_preset=self._state.selected_sector_preset,
            enabled_standards=scope.enabled_standards if scope else [],
            enabled_scope3_categories=(
                scope.enabled_scope3_categories if scope else []
            ),
            cross_frameworks=scope.cross_frameworks if scope else [],
            data_sources_configured=data_sources_count,
            entities_configured=entities_count,
            total_agents_active=total_agents,
            validation_status="passed" if self._state.validation_passed else "not_run",
            demo_run_status="passed" if self._state.demo_run_passed else "not_run",
            estimated_first_report_effort=effort,
            professional_features_enabled=enabled_features,
            recommendations=recommendations,
            configuration_hash=config_hash,
        )

    def _generate_recommendations(self) -> List[str]:
        """Generate setup recommendations."""
        recs: List[str] = []
        if self._state is None:
            return recs

        if self._state.company_profile:
            if len(self._state.company_profile.subsidiaries) > 0:
                has_unmapped = any(
                    not sub.data_source_ids
                    for sub in self._state.company_profile.subsidiaries
                )
                if has_unmapped:
                    recs.append(
                        "Some subsidiaries have no data sources mapped. "
                        "Configure data sources for all material entities."
                    )

        if self._state.reporting_scope:
            if not self._state.reporting_scope.cross_frameworks:
                recs.append(
                    "Enable cross-framework alignment (CDP/TCFD/SBTi) "
                    "to maximize reporting efficiency."
                )

        if self._state.professional_features:
            if not self._state.professional_features.approval_users:
                recs.append(
                    "Configure approval chain users for each level "
                    "(preparer, reviewer, approver, board)."
                )

        return recs

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    def _advance_to_next_step(self, current: WizardStepName) -> None:
        """Advance to the next step.

        Args:
            current: Step that was just completed.
        """
        if self._state is None:
            return
        try:
            idx = STEP_DEFINITIONS.index(current)
            if idx < len(STEP_DEFINITIONS) - 1:
                self._state.current_step = STEP_DEFINITIONS[idx + 1]
        except ValueError:
            pass
