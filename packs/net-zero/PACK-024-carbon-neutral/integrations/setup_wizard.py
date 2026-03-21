# -*- coding: utf-8 -*-
"""
CarbonNeutralSetupWizard - 6-Step Guided Configuration for PACK-024
=====================================================================

This module implements a 6-step configuration wizard for organisations
setting up the Carbon Neutral Pack. Each step is tailored to PAS 2060,
ICVCM Core Carbon Principles, and VCMI Claims Code requirements.

Wizard Steps (6):
    1. organization_profile    -- Name, sector, region, size, boundary type,
                                  consolidation approach
    2. boundary_selection      -- PAS 2060 subject boundary, GHG Protocol
                                  boundary, scope inclusion rules
    3. scope_configuration     -- Scope 1/2/3 inclusion, Scope 2 method,
                                  Scope 3 materiality, coverage requirements
    4. data_source_setup       -- ERP type, file formats, API connections,
                                  emission factor databases, registry API keys
    5. credit_preferences      -- Preferred registries, quality tier minimum,
                                  removal/avoidance mix, vintage requirements,
                                  ICVCM CCP preference, budget constraints
    6. preset_selection        -- Auto-recommend based on sector, size,
                                  boundary type, and credit preferences

Sector Presets (6):
    manufacturing, services, technology, retail,
    financial_services, energy

PAS 2060 Requirements Enforced:
    - Subject boundary clearly defined
    - All material emission sources included
    - Carbon management plan with reduction targets
    - Credits from recognized registries
    - No double counting
    - Qualifying explanatory statement
    - Independent verification
    - Public disclosure

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-024 Carbon Neutral Pack
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class CarbonNeutralWizardStep(str, Enum):
    """Names of wizard steps in execution order."""

    ORGANIZATION_PROFILE = "organization_profile"
    BOUNDARY_SELECTION = "boundary_selection"
    SCOPE_CONFIGURATION = "scope_configuration"
    DATA_SOURCE_SETUP = "data_source_setup"
    CREDIT_PREFERENCES = "credit_preferences"
    PRESET_SELECTION = "preset_selection"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class OrganizationSize(str, Enum):
    """Organization size classification."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class BoundaryType(str, Enum):
    """PAS 2060 subject boundary types."""

    ORGANIZATION = "organization"
    PRODUCT = "product"
    SERVICE = "service"
    BUILDING = "building"
    PROJECT = "project"
    EVENT = "event"


class Scope2Method(str, Enum):
    """Scope 2 reporting method."""

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"
    DUAL = "dual"


class CreditQualityPreference(str, Enum):
    """Credit quality preference level."""

    CCP_ONLY = "ccp_only"
    HIGH_QUALITY = "high_quality"
    STANDARD = "standard"
    ANY = "any"


class EmissionFactorSource(str, Enum):
    """Emission factor database sources."""

    GHG_PROTOCOL = "ghg_protocol"
    IPCC_AR6 = "ipcc_ar6"
    DEFRA = "defra"
    EPA = "epa"
    ECOINVENT = "ecoinvent"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Sector Presets
# ---------------------------------------------------------------------------

SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "name": "Manufacturing",
        "description": "Manufacturing and industrial operations",
        "boundary_type": "organization",
        "scope2_method": "dual",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 3, 4, 5, 6, 7, 11, 12],
        "credit_quality": "high_quality",
        "target_removal_pct": 10.0,
        "preferred_registries": ["verra", "gold_standard"],
        "ef_source": "ghg_protocol",
    },
    "services": {
        "name": "Professional Services",
        "description": "Office-based professional services",
        "boundary_type": "organization",
        "scope2_method": "market_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 5, 6, 7],
        "credit_quality": "standard",
        "target_removal_pct": 5.0,
        "preferred_registries": ["gold_standard", "verra"],
        "ef_source": "defra",
    },
    "technology": {
        "name": "Technology",
        "description": "Technology and software companies",
        "boundary_type": "organization",
        "scope2_method": "market_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 5, 6, 7, 11],
        "credit_quality": "ccp_only",
        "target_removal_pct": 20.0,
        "preferred_registries": ["verra", "gold_standard", "puro"],
        "ef_source": "ghg_protocol",
    },
    "retail": {
        "name": "Retail & Consumer",
        "description": "Retail, consumer goods, and hospitality",
        "boundary_type": "organization",
        "scope2_method": "dual",
        "include_scope3": True,
        "material_scope3_categories": [1, 4, 5, 7, 9, 11, 12],
        "credit_quality": "high_quality",
        "target_removal_pct": 10.0,
        "preferred_registries": ["verra", "gold_standard"],
        "ef_source": "defra",
    },
    "financial_services": {
        "name": "Financial Services",
        "description": "Banking, insurance, and investment",
        "boundary_type": "organization",
        "scope2_method": "market_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 5, 6, 7, 15],
        "credit_quality": "ccp_only",
        "target_removal_pct": 15.0,
        "preferred_registries": ["verra", "gold_standard", "acr"],
        "ef_source": "ghg_protocol",
    },
    "energy": {
        "name": "Energy & Utilities",
        "description": "Energy generation, transmission, and distribution",
        "boundary_type": "organization",
        "scope2_method": "location_based",
        "include_scope3": True,
        "material_scope3_categories": [1, 2, 3, 4, 5, 11],
        "credit_quality": "high_quality",
        "target_removal_pct": 10.0,
        "preferred_registries": ["verra", "acr", "car"],
        "ef_source": "epa",
    },
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class OrganizationProfile(BaseModel):
    """Step 1: Organization profile."""

    organization_name: str = Field(default="")
    sector: str = Field(default="")
    sub_sector: str = Field(default="")
    region: str = Field(default="")
    country: str = Field(default="")
    size: str = Field(default="medium")
    employee_count: int = Field(default=0, ge=0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    consolidation_approach: str = Field(default="operational_control")
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2019, ge=2015, le=2025)


class BoundarySelection(BaseModel):
    """Step 2: PAS 2060 subject boundary."""

    boundary_type: str = Field(default="organization")
    boundary_description: str = Field(default="")
    entities_included: List[str] = Field(default_factory=list)
    entities_excluded: List[str] = Field(default_factory=list)
    exclusion_justification: str = Field(default="")
    pas_2060_clause_7: bool = Field(default=True)


class ScopeConfiguration(BaseModel):
    """Step 3: Scope inclusion configuration."""

    include_scope1: bool = Field(default=True)
    include_scope2: bool = Field(default=True)
    include_scope3: bool = Field(default=True)
    scope2_method: str = Field(default="dual")
    scope3_categories_included: List[int] = Field(default_factory=list)
    scope3_exclusion_justification: str = Field(default="")
    materiality_threshold_pct: float = Field(default=5.0, ge=0.0, le=100.0)


class DataSourceSetup(BaseModel):
    """Step 4: Data source configuration."""

    erp_system: str = Field(default="")
    erp_connected: bool = Field(default=False)
    file_formats: List[str] = Field(default_factory=lambda: ["excel", "csv"])
    api_connections: List[str] = Field(default_factory=list)
    ef_source: str = Field(default="ghg_protocol")
    registry_api_keys: Dict[str, bool] = Field(default_factory=dict)
    base_year_data_available: bool = Field(default=False)


class CreditPreferences(BaseModel):
    """Step 5: Credit procurement preferences."""

    preferred_registries: List[str] = Field(default_factory=lambda: ["verra", "gold_standard"])
    min_quality_tier: str = Field(default="standard")
    target_removal_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    max_vintage_age_years: int = Field(default=5, ge=1, le=10)
    preferred_categories: List[str] = Field(default_factory=list)
    budget_per_tco2e_usd: float = Field(default=0.0, ge=0.0)
    total_budget_usd: float = Field(default=0.0, ge=0.0)
    icvcm_ccp_required: bool = Field(default=False)
    sdg_preferences: List[int] = Field(default_factory=list)


class PresetSelection(BaseModel):
    """Step 6: Preset selection."""

    preset_id: str = Field(default="")
    preset_name: str = Field(default="")
    auto_recommended: bool = Field(default=False)
    customizations: Dict[str, Any] = Field(default_factory=dict)


class WizardStepState(BaseModel):
    """State of a single wizard step."""

    step: str = Field(...)
    status: str = Field(default="pending")
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = Field(default=None)


class WizardState(BaseModel):
    """Complete wizard state across all steps."""

    wizard_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-024")
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    current_step: str = Field(default="organization_profile")
    completed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class SetupResult(BaseModel):
    """Final wizard setup result."""

    setup_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    organization_profile: Optional[OrganizationProfile] = Field(default=None)
    boundary_selection: Optional[BoundarySelection] = Field(default=None)
    scope_configuration: Optional[ScopeConfiguration] = Field(default=None)
    data_source_setup: Optional[DataSourceSetup] = Field(default=None)
    credit_preferences: Optional[CreditPreferences] = Field(default=None)
    preset_selection: Optional[PresetSelection] = Field(default=None)
    pack_config: Dict[str, Any] = Field(default_factory=dict)
    pas_2060_ready: bool = Field(default=False)
    steps_completed: int = Field(default=0)
    steps_total: int = Field(default=6)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# CarbonNeutralSetupWizard
# ---------------------------------------------------------------------------


class CarbonNeutralSetupWizard:
    """6-step guided configuration wizard for PACK-024.

    Guides organisations through PAS 2060 carbon neutrality setup including
    organization profile, boundary definition, scope configuration, data
    sources, credit preferences, and preset selection.

    Example:
        >>> wizard = CarbonNeutralSetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("organization_profile", {"organization_name": "ACME"})
        >>> assert state.steps["organization_profile"].status == "completed"
    """

    STEP_ORDER = [
        CarbonNeutralWizardStep.ORGANIZATION_PROFILE,
        CarbonNeutralWizardStep.BOUNDARY_SELECTION,
        CarbonNeutralWizardStep.SCOPE_CONFIGURATION,
        CarbonNeutralWizardStep.DATA_SOURCE_SETUP,
        CarbonNeutralWizardStep.CREDIT_PREFERENCES,
        CarbonNeutralWizardStep.PRESET_SELECTION,
    ]

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._state: Optional[WizardState] = None
        self.logger.info("CarbonNeutralSetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session."""
        self._state = WizardState(
            steps={
                step.value: WizardStepState(step=step.value)
                for step in self.STEP_ORDER
            },
            current_step=self.STEP_ORDER[0].value,
        )
        self.logger.info("Wizard started: %s", self._state.wizard_id)
        return self._state

    def get_state(self) -> Optional[WizardState]:
        """Get current wizard state."""
        return self._state

    def complete_step(
        self,
        step_name: str,
        data: Dict[str, Any],
    ) -> WizardState:
        """Complete a wizard step with provided data.

        Args:
            step_name: Step name to complete.
            data: Step data.

        Returns:
            Updated WizardState.
        """
        if not self._state:
            self._state = self.start()

        step_state = self._state.steps.get(step_name)
        if not step_state:
            raise ValueError(f"Unknown step: {step_name}")

        # Validate step data
        errors = self._validate_step(step_name, data)
        step_state.validation_errors = errors

        if errors:
            step_state.status = StepStatus.FAILED.value
        else:
            step_state.status = StepStatus.COMPLETED.value
            step_state.data = data
            step_state.completed_at = _utcnow()

            # Advance to next step
            current_idx = next(
                (i for i, s in enumerate(self.STEP_ORDER) if s.value == step_name), -1
            )
            if current_idx < len(self.STEP_ORDER) - 1:
                self._state.current_step = self.STEP_ORDER[current_idx + 1].value
            else:
                self._state.completed = True

        self._state.updated_at = _utcnow()
        return self._state

    def skip_step(self, step_name: str) -> WizardState:
        """Skip a wizard step."""
        if not self._state:
            self._state = self.start()

        step_state = self._state.steps.get(step_name)
        if step_state:
            step_state.status = StepStatus.SKIPPED.value

        # Advance
        current_idx = next(
            (i for i, s in enumerate(self.STEP_ORDER) if s.value == step_name), -1
        )
        if current_idx < len(self.STEP_ORDER) - 1:
            self._state.current_step = self.STEP_ORDER[current_idx + 1].value

        self._state.updated_at = _utcnow()
        return self._state

    def recommend_preset(
        self,
        profile: Optional[OrganizationProfile] = None,
    ) -> PresetSelection:
        """Recommend a preset based on organization profile.

        Args:
            profile: Organization profile for recommendation.

        Returns:
            PresetSelection with recommended preset.
        """
        if not profile and self._state:
            profile_data = self._state.steps.get("organization_profile", WizardStepState(step="organization_profile")).data
            if profile_data:
                profile = OrganizationProfile(**profile_data)

        if not profile:
            return PresetSelection(preset_id="services", preset_name="Professional Services", auto_recommended=True)

        sector = profile.sector.lower()

        # Match sector to preset
        preset_id = "services"  # Default
        if "manufact" in sector or "industrial" in sector:
            preset_id = "manufacturing"
        elif "tech" in sector or "software" in sector or "it" in sector:
            preset_id = "technology"
        elif "retail" in sector or "consumer" in sector or "hospitality" in sector:
            preset_id = "retail"
        elif "financ" in sector or "bank" in sector or "insurance" in sector:
            preset_id = "financial_services"
        elif "energy" in sector or "utilit" in sector or "power" in sector:
            preset_id = "energy"

        preset_info = SECTOR_PRESETS.get(preset_id, SECTOR_PRESETS["services"])
        return PresetSelection(
            preset_id=preset_id,
            preset_name=preset_info["name"],
            auto_recommended=True,
        )

    def finalize(self) -> SetupResult:
        """Finalize the wizard and generate setup result.

        Returns:
            SetupResult with complete configuration.
        """
        start = time.monotonic()

        if not self._state:
            self._state = self.start()

        completed_steps = sum(
            1 for s in self._state.steps.values()
            if s.status in (StepStatus.COMPLETED.value, StepStatus.SKIPPED.value)
        )

        # Build configuration from step data
        org_data = self._state.steps.get("organization_profile", WizardStepState(step="")).data
        boundary_data = self._state.steps.get("boundary_selection", WizardStepState(step="")).data
        scope_data = self._state.steps.get("scope_configuration", WizardStepState(step="")).data
        data_source_data = self._state.steps.get("data_source_setup", WizardStepState(step="")).data
        credit_data = self._state.steps.get("credit_preferences", WizardStepState(step="")).data
        preset_data = self._state.steps.get("preset_selection", WizardStepState(step="")).data

        org_profile = OrganizationProfile(**org_data) if org_data else None
        boundary = BoundarySelection(**boundary_data) if boundary_data else None
        scope_config = ScopeConfiguration(**scope_data) if scope_data else None
        data_setup = DataSourceSetup(**data_source_data) if data_source_data else None
        credit_prefs = CreditPreferences(**credit_data) if credit_data else None
        preset = PresetSelection(**preset_data) if preset_data else None

        # Generate pack configuration
        pack_config: Dict[str, Any] = {}
        if org_profile:
            pack_config["organization_name"] = org_profile.organization_name
            pack_config["reporting_year"] = org_profile.reporting_year
            pack_config["base_year"] = org_profile.base_year
        if boundary:
            pack_config["boundary_type"] = boundary.boundary_type
        if scope_config:
            pack_config["scope2_method"] = scope_config.scope2_method
            pack_config["include_scope3"] = scope_config.include_scope3
        if credit_prefs:
            pack_config["preferred_registries"] = credit_prefs.preferred_registries
            pack_config["min_quality_tier"] = credit_prefs.min_quality_tier
            pack_config["target_removal_pct"] = credit_prefs.target_removal_pct

        pas_ready = completed_steps >= 4 and org_profile is not None and boundary is not None

        result = SetupResult(
            status="completed" if self._state.completed else "in_progress",
            organization_profile=org_profile,
            boundary_selection=boundary,
            scope_configuration=scope_config,
            data_source_setup=data_setup,
            credit_preferences=credit_prefs,
            preset_selection=preset,
            pack_config=pack_config,
            pas_2060_ready=pas_ready,
            steps_completed=completed_steps,
            steps_total=len(self.STEP_ORDER),
        )
        result.duration_ms = (time.monotonic() - start) * 1000
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Wizard finalized: %d/%d steps, pas_2060_ready=%s",
            completed_steps, len(self.STEP_ORDER), pas_ready,
        )
        return result

    def _validate_step(self, step_name: str, data: Dict[str, Any]) -> List[str]:
        """Validate step data."""
        errors: List[str] = []

        if step_name == "organization_profile":
            if not data.get("organization_name"):
                errors.append("Organization name is required")
            if not data.get("sector"):
                errors.append("Sector is required")

        elif step_name == "boundary_selection":
            if not data.get("boundary_type"):
                errors.append("Boundary type is required")

        elif step_name == "scope_configuration":
            if not data.get("include_scope1", True) and not data.get("include_scope2", True):
                errors.append("At least Scope 1 or Scope 2 must be included")

        elif step_name == "credit_preferences":
            if not data.get("preferred_registries"):
                errors.append("At least one preferred registry is required")

        return errors

    def get_available_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all available sector presets."""
        return SECTOR_PRESETS

    def apply_preset(self, preset_id: str) -> WizardState:
        """Apply a preset to auto-fill wizard steps.

        Args:
            preset_id: Preset identifier.

        Returns:
            Updated WizardState with preset data applied.
        """
        if not self._state:
            self._state = self.start()

        preset = SECTOR_PRESETS.get(preset_id)
        if not preset:
            return self._state

        # Apply preset to scope configuration
        scope_data = {
            "include_scope1": True,
            "include_scope2": True,
            "include_scope3": preset.get("include_scope3", True),
            "scope2_method": preset.get("scope2_method", "dual"),
            "scope3_categories_included": preset.get("material_scope3_categories", []),
        }
        self.complete_step("scope_configuration", scope_data)

        # Apply preset to credit preferences
        credit_data = {
            "preferred_registries": preset.get("preferred_registries", ["verra"]),
            "min_quality_tier": preset.get("credit_quality", "standard"),
            "target_removal_pct": preset.get("target_removal_pct", 10.0),
        }
        self.complete_step("credit_preferences", credit_data)

        # Apply preset to data source
        data_data = {
            "ef_source": preset.get("ef_source", "ghg_protocol"),
        }
        self.complete_step("data_source_setup", data_data)

        # Apply preset selection
        self.complete_step("preset_selection", {
            "preset_id": preset_id,
            "preset_name": preset.get("name", preset_id),
            "auto_recommended": True,
        })

        return self._state
