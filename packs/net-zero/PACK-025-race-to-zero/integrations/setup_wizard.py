# -*- coding: utf-8 -*-
"""
RaceToZeroSetupWizard - 8-Step Guided Configuration for PACK-025
===================================================================

This module implements an 8-step configuration wizard for organisations
setting up the Race to Zero Pack. Each step is tailored to UNFCCC Race
to Zero criteria, starting line requirements, credibility criteria,
and partner initiative requirements.

Wizard Steps (8):
    1. organization_profile     -- Name, type, sector, region, size
    2. partner_initiative       -- Select partner initiative, verify eligibility
    3. baseline_configuration   -- Base year, emissions data, scope coverage
    4. target_setting           -- Near-term (2030), net-zero, interim targets
    5. scope_configuration      -- Scope 1/2/3 inclusion and method selection
    6. data_source_setup        -- ERP type, file formats, API connections
    7. credibility_preferences  -- Fossil fuel policy, offset limits, just transition
    8. preset_selection          -- Auto-recommend based on profile

Sector Presets (8):
    manufacturing, services, technology, retail,
    financial_services, energy, real_estate, healthcare

Race to Zero Specific Configuration:
    - Partner initiative selection and eligibility check
    - Starting line criteria tracking setup
    - Credibility criteria configuration
    - UNFCCC portal connectivity setup
    - Sector pathway alignment

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-025 Race to Zero Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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

class RaceToZeroWizardStep(str, Enum):
    ORGANIZATION_PROFILE = "organization_profile"
    PARTNER_INITIATIVE = "partner_initiative"
    BASELINE_CONFIGURATION = "baseline_configuration"
    TARGET_SETTING = "target_setting"
    SCOPE_CONFIGURATION = "scope_configuration"
    DATA_SOURCE_SETUP = "data_source_setup"
    CREDIBILITY_PREFERENCES = "credibility_preferences"
    PRESET_SELECTION = "preset_selection"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class OrganizationType(str, Enum):
    BUSINESS = "business"
    CITY = "city"
    REGION = "region"
    INVESTOR = "investor"
    UNIVERSITY = "university"
    HEALTHCARE = "healthcare"
    FINANCIAL_INSTITUTION = "financial_institution"

class OrganizationSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class ConsolidationApproach(str, Enum):
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class Scope2Method(str, Enum):
    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"
    DUAL = "dual"

class AmbitionLevel(str, Enum):
    MINIMUM_COMPLIANCE = "minimum_compliance"
    ALIGNED = "aligned"
    AMBITIOUS = "ambitious"
    LEADING = "leading"

# ---------------------------------------------------------------------------
# Sector Presets
# ---------------------------------------------------------------------------

SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "name": "Manufacturing",
        "description": "Manufacturing and industrial operations",
        "org_type": "business",
        "scope2_method": "dual",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 3, 4, 5, 6, 7, 11, 12],
        "near_term_reduction_pct": 50.0,
        "net_zero_year": 2050,
        "partner_initiatives": ["science_based_targets", "business_ambition_1_5c"],
        "sector_pathway": "steel",
        "offset_limit_pct": 10.0,
        "ef_source": "ghg_protocol",
    },
    "services": {
        "name": "Professional Services",
        "description": "Office-based professional services",
        "org_type": "business",
        "scope2_method": "market_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 5, 6, 7],
        "near_term_reduction_pct": 50.0,
        "net_zero_year": 2045,
        "partner_initiatives": ["science_based_targets", "re100"],
        "sector_pathway": "buildings",
        "offset_limit_pct": 10.0,
        "ef_source": "defra",
    },
    "technology": {
        "name": "Technology",
        "description": "Technology and software companies",
        "org_type": "business",
        "scope2_method": "market_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 5, 6, 7, 11],
        "near_term_reduction_pct": 55.0,
        "net_zero_year": 2040,
        "partner_initiatives": ["science_based_targets", "re100", "ev100"],
        "sector_pathway": "technology",
        "offset_limit_pct": 5.0,
        "ef_source": "ghg_protocol",
    },
    "retail": {
        "name": "Retail & Consumer",
        "description": "Retail, consumer goods, and hospitality",
        "org_type": "business",
        "scope2_method": "dual",
        "include_scope3": True,
        "material_scope3_categories": [1, 4, 5, 7, 9, 11, 12],
        "near_term_reduction_pct": 50.0,
        "net_zero_year": 2050,
        "partner_initiatives": ["science_based_targets"],
        "sector_pathway": "buildings",
        "offset_limit_pct": 10.0,
        "ef_source": "defra",
    },
    "financial_services": {
        "name": "Financial Services",
        "description": "Banking, insurance, and investment",
        "org_type": "financial_institution",
        "scope2_method": "market_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 5, 6, 7, 15],
        "near_term_reduction_pct": 50.0,
        "net_zero_year": 2050,
        "partner_initiatives": ["gfanz", "nzaoa", "net_zero_banking"],
        "sector_pathway": "financial_services",
        "offset_limit_pct": 5.0,
        "ef_source": "ghg_protocol",
    },
    "energy": {
        "name": "Energy & Utilities",
        "description": "Energy generation, transmission, distribution",
        "org_type": "business",
        "scope2_method": "location_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 3, 4, 5, 11],
        "near_term_reduction_pct": 60.0,
        "net_zero_year": 2040,
        "partner_initiatives": ["science_based_targets", "re100"],
        "sector_pathway": "power_generation",
        "offset_limit_pct": 5.0,
        "ef_source": "epa",
    },
    "real_estate": {
        "name": "Real Estate",
        "description": "Property management and development",
        "org_type": "business",
        "scope2_method": "dual",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 5, 7, 13],
        "near_term_reduction_pct": 50.0,
        "net_zero_year": 2050,
        "partner_initiatives": ["science_based_targets"],
        "sector_pathway": "real_estate",
        "offset_limit_pct": 10.0,
        "ef_source": "ghg_protocol",
    },
    "healthcare": {
        "name": "Healthcare",
        "description": "Healthcare providers and systems",
        "org_type": "healthcare",
        "scope2_method": "dual",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 4, 5, 6, 7],
        "near_term_reduction_pct": 50.0,
        "net_zero_year": 2050,
        "partner_initiatives": ["health_care_climate_pledge"],
        "sector_pathway": "buildings",
        "offset_limit_pct": 10.0,
        "ef_source": "ghg_protocol",
    },
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class OrganizationProfile(BaseModel):
    """Step 1: Organization profile."""

    organization_name: str = Field(default="")
    organization_type: str = Field(default="business")
    sector: str = Field(default="")
    sub_sector: str = Field(default="")
    region: str = Field(default="")
    country: str = Field(default="")
    size: str = Field(default="medium")
    employee_count: int = Field(default=0, ge=0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    consolidation_approach: str = Field(default="operational_control")
    reporting_year: int = Field(default=2025, ge=2020, le=2035)

class PartnerInitiativeSelection(BaseModel):
    """Step 2: Partner initiative selection."""

    partner_initiative: str = Field(default="")
    initiative_eligible: bool = Field(default=False)
    commitment_signed: bool = Field(default=False)
    leadership_signoff: bool = Field(default=False)
    unfccc_portal_registered: bool = Field(default=False)
    unfccc_api_key: str = Field(default="")

class BaselineConfiguration(BaseModel):
    """Step 3: Baseline emissions configuration."""

    base_year: int = Field(default=2019, ge=2015, le=2025)
    base_year_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    current_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    current_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    current_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    data_quality_assessed: bool = Field(default=False)

class TargetSetting(BaseModel):
    """Step 4: Target setting configuration."""

    near_term_target_year: int = Field(default=2030, ge=2025, le=2035)
    near_term_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    net_zero_target_year: int = Field(default=2050, ge=2040, le=2060)
    net_zero_reduction_pct: float = Field(default=90.0, ge=0.0, le=100.0)
    interim_targets: Dict[int, float] = Field(default_factory=lambda: {2035: 60, 2040: 75, 2045: 85})
    targets_science_based: bool = Field(default=True)
    sbti_committed: bool = Field(default=False)
    ambition_level: str = Field(default="aligned")

class ScopeConfiguration(BaseModel):
    """Step 5: Scope configuration."""

    include_scope1: bool = Field(default=True)
    include_scope2: bool = Field(default=True)
    include_scope3: bool = Field(default=True)
    scope2_method: str = Field(default="dual")
    scope3_categories_included: List[int] = Field(default_factory=list)
    scope3_exclusion_justification: str = Field(default="")
    materiality_threshold_pct: float = Field(default=5.0, ge=0.0, le=100.0)

class DataSourceSetup(BaseModel):
    """Step 6: Data source configuration."""

    erp_system: str = Field(default="")
    erp_connected: bool = Field(default=False)
    file_formats: List[str] = Field(default_factory=lambda: ["excel", "csv"])
    api_connections: List[str] = Field(default_factory=list)
    ef_source: str = Field(default="ghg_protocol")
    cdp_connected: bool = Field(default=False)
    unfccc_connected: bool = Field(default=False)
    supplier_engagement_platform: str = Field(default="")

class CredibilityPreferences(BaseModel):
    """Step 7: Credibility criteria preferences."""

    no_fossil_expansion_committed: bool = Field(default=True)
    fossil_phase_out_plan: bool = Field(default=False)
    offset_limit_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    just_transition_policy: bool = Field(default=False)
    lobbying_alignment_committed: bool = Field(default=False)
    public_disclosure_committed: bool = Field(default=True)
    verification_planned: bool = Field(default=True)

class PresetSelection(BaseModel):
    """Step 8: Preset selection."""

    preset_name: str = Field(default="")
    preset_applied: bool = Field(default=False)
    customizations: Dict[str, Any] = Field(default_factory=dict)

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    step: RaceToZeroWizardStep = Field(...)
    status: StepStatus = Field(default=StepStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class WizardState(BaseModel):
    """Complete wizard state across all steps."""

    wizard_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    current_step: RaceToZeroWizardStep = Field(default=RaceToZeroWizardStep.ORGANIZATION_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)
    overall_progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class SetupResult(BaseModel):
    """Final wizard setup result."""

    setup_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-025")
    organization_name: str = Field(default="")
    preset_applied: str = Field(default="")
    steps_completed: int = Field(default=0)
    steps_total: int = Field(default=8)
    r2z_ready: bool = Field(default=False)
    starting_line_configured: bool = Field(default=False)
    credibility_configured: bool = Field(default=False)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# RaceToZeroSetupWizard
# ---------------------------------------------------------------------------

class RaceToZeroSetupWizard:
    """8-step guided configuration wizard for PACK-025 Race to Zero.

    Guides organizations through Race to Zero setup including partner
    initiative selection, baseline configuration, target setting,
    credibility preferences, and sector preset application.

    Example:
        >>> wizard = RaceToZeroSetupWizard()
        >>> wizard.set_organization_profile(OrganizationProfile(
        ...     organization_name="Acme Corp",
        ...     sector="technology",
        ... ))
        >>> result = wizard.complete_setup()
        >>> print(f"R2Z Ready: {result.r2z_ready}")
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._state = WizardState()
        self._init_steps()
        self.logger.info("RaceToZeroSetupWizard initialized: wizard_id=%s", self._state.wizard_id)

    def _init_steps(self) -> None:
        """Initialize all wizard steps."""
        for step in RaceToZeroWizardStep:
            self._state.steps[step.value] = WizardStepState(step=step)

    @property
    def state(self) -> WizardState:
        """Get current wizard state."""
        return self._state

    def get_available_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available sector presets.

        Returns:
            Dict of preset definitions.
        """
        return dict(SECTOR_PRESETS)

    def get_step_state(self, step: RaceToZeroWizardStep) -> WizardStepState:
        """Get state of a specific step.

        Args:
            step: The wizard step.

        Returns:
            WizardStepState for the step.
        """
        return self._state.steps.get(step.value, WizardStepState(step=step))

    def set_organization_profile(self, profile: OrganizationProfile) -> WizardStepState:
        """Set organization profile (Step 1).

        Args:
            profile: Organization profile data.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.ORGANIZATION_PROFILE
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = profile.model_dump()

        warnings = []
        if not profile.organization_name:
            state.errors.append("Organization name is required")
        if not profile.sector:
            warnings.append("Sector not specified, preset selection may be limited")

        state.warnings = warnings
        if not state.errors:
            state.status = StepStatus.COMPLETED
            state.completed_at = utcnow()

        self._update_progress()
        return state

    def set_partner_initiative(self, selection: PartnerInitiativeSelection) -> WizardStepState:
        """Set partner initiative (Step 2).

        Args:
            selection: Partner initiative selection.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.PARTNER_INITIATIVE
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = selection.model_dump()

        if not selection.partner_initiative:
            state.warnings.append("No partner initiative selected, consider joining one")
        if selection.partner_initiative and not selection.initiative_eligible:
            state.errors.append(f"Organization may not be eligible for {selection.partner_initiative}")
        if not selection.leadership_signoff:
            state.warnings.append("Leadership signoff recommended before submission")

        if not state.errors:
            state.status = StepStatus.COMPLETED
            state.completed_at = utcnow()

        self._update_progress()
        return state

    def set_baseline_configuration(self, config: BaselineConfiguration) -> WizardStepState:
        """Set baseline configuration (Step 3).

        Args:
            config: Baseline emissions configuration.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.BASELINE_CONFIGURATION
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = config.model_dump()

        if config.base_year < 2015:
            state.errors.append("Base year must be 2015 or later for Race to Zero")
        total_base = config.base_year_scope1_tco2e + config.base_year_scope2_tco2e + config.base_year_scope3_tco2e
        if total_base == 0:
            state.errors.append("Base year emissions data required")
        if config.base_year_scope3_tco2e == 0:
            state.warnings.append("Scope 3 base year emissions not provided, recommended for R2Z")

        if not state.errors:
            state.status = StepStatus.COMPLETED
            state.completed_at = utcnow()

        self._update_progress()
        return state

    def set_target_setting(self, targets: TargetSetting) -> WizardStepState:
        """Set target configuration (Step 4).

        Args:
            targets: Target setting configuration.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.TARGET_SETTING
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = targets.model_dump()

        if targets.near_term_reduction_pct < 50.0:
            state.warnings.append(
                f"R2Z requires 50% reduction by 2030, current: {targets.near_term_reduction_pct}%"
            )
        if targets.net_zero_target_year > 2050:
            state.errors.append("Net-zero target must be 2050 or sooner")
        if targets.net_zero_reduction_pct < 90.0:
            state.warnings.append("SBTi requires at least 90% reduction for net-zero")
        if not targets.targets_science_based:
            state.warnings.append("Science-based targets recommended for Race to Zero")

        if not state.errors:
            state.status = StepStatus.COMPLETED
            state.completed_at = utcnow()

        self._update_progress()
        return state

    def set_scope_configuration(self, config: ScopeConfiguration) -> WizardStepState:
        """Set scope configuration (Step 5).

        Args:
            config: Scope configuration.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.SCOPE_CONFIGURATION
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = config.model_dump()

        if not config.include_scope1 or not config.include_scope2:
            state.errors.append("Scope 1 and 2 are mandatory for Race to Zero")
        if not config.include_scope3:
            state.warnings.append("Scope 3 strongly recommended for Race to Zero credibility")
        if config.include_scope3 and len(config.scope3_categories_included) < 5:
            state.warnings.append("Include at least material Scope 3 categories")

        if not state.errors:
            state.status = StepStatus.COMPLETED
            state.completed_at = utcnow()

        self._update_progress()
        return state

    def set_data_source_setup(self, setup: DataSourceSetup) -> WizardStepState:
        """Set data source configuration (Step 6).

        Args:
            setup: Data source setup.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.DATA_SOURCE_SETUP
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = setup.model_dump()

        if not setup.file_formats and not setup.erp_connected:
            state.warnings.append("No data sources configured")
        if not setup.unfccc_connected:
            state.warnings.append("UNFCCC portal connection recommended for automated reporting")

        state.status = StepStatus.COMPLETED
        state.completed_at = utcnow()
        self._update_progress()
        return state

    def set_credibility_preferences(self, prefs: CredibilityPreferences) -> WizardStepState:
        """Set credibility preferences (Step 7).

        Args:
            prefs: Credibility criteria preferences.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.CREDIBILITY_PREFERENCES
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()
        state.data = prefs.model_dump()

        if not prefs.no_fossil_expansion_committed:
            state.warnings.append("No fossil expansion commitment required for R2Z credibility")
        if prefs.offset_limit_pct > 10.0:
            state.warnings.append("R2Z recommends offsetting limited to 10% of reductions")
        if not prefs.just_transition_policy:
            state.warnings.append("Just transition policy recommended for R2Z credibility")

        state.status = StepStatus.COMPLETED
        state.completed_at = utcnow()
        self._update_progress()
        return state

    def apply_preset(self, preset_name: str) -> WizardStepState:
        """Apply a sector preset (Step 8).

        Args:
            preset_name: Name of the preset to apply.

        Returns:
            Updated WizardStepState.
        """
        step = RaceToZeroWizardStep.PRESET_SELECTION
        state = self._state.steps[step.value]
        state.status = StepStatus.IN_PROGRESS
        state.started_at = utcnow()

        preset = SECTOR_PRESETS.get(preset_name)
        if not preset:
            state.errors.append(f"Unknown preset: {preset_name}")
            state.status = StepStatus.FAILED
            return state

        state.data = {"preset_name": preset_name, "preset": preset, "applied": True}
        state.status = StepStatus.COMPLETED
        state.completed_at = utcnow()
        self._update_progress()
        return state

    def recommend_preset(self, org_type: str = "", sector: str = "") -> str:
        """Recommend a preset based on organization profile.

        Args:
            org_type: Organization type.
            sector: Organization sector.

        Returns:
            Recommended preset name.
        """
        if sector in SECTOR_PRESETS:
            return sector

        type_mapping = {
            "financial_institution": "financial_services",
            "healthcare": "healthcare",
            "university": "services",
            "city": "real_estate",
            "region": "energy",
        }
        return type_mapping.get(org_type, "services")

    def complete_setup(self) -> SetupResult:
        """Complete the setup wizard and generate configuration.

        Returns:
            SetupResult with final configuration.
        """
        completed = sum(
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )

        all_data = {}
        for step_name, step_state in self._state.steps.items():
            all_data[step_name] = step_state.data

        org = all_data.get("organization_profile", {})
        targets = all_data.get("target_setting", {})
        cred = all_data.get("credibility_preferences", {})
        preset_data = all_data.get("preset_selection", {})

        starting_line = (
            "partner_initiative" in all_data
            and all_data.get("partner_initiative", {}).get("commitment_signed", False)
        )
        credibility = (
            cred.get("no_fossil_expansion_committed", False)
            and targets.get("near_term_reduction_pct", 0) >= 50.0
        )

        r2z_ready = completed >= 6 and starting_line

        warnings = []
        for step_state in self._state.steps.values():
            warnings.extend(step_state.warnings)

        result = SetupResult(
            organization_name=org.get("organization_name", ""),
            preset_applied=preset_data.get("preset_name", ""),
            steps_completed=completed,
            r2z_ready=r2z_ready,
            starting_line_configured=starting_line,
            credibility_configured=credibility,
            configuration=all_data,
            warnings=warnings,
        )

        if True:  # enable_provenance
            result.provenance_hash = _compute_hash(result)

        self._state.completed_at = utcnow()
        return result

    def _update_progress(self) -> None:
        """Update overall progress percentage."""
        total = len(RaceToZeroWizardStep)
        completed = sum(
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )
        self._state.overall_progress_pct = round(completed / total * 100, 1)
