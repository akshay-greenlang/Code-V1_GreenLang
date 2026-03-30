# -*- coding: utf-8 -*-
"""
NetZeroSetupWizard - 6-Step Guided Configuration for PACK-021
================================================================

This module implements a 6-step configuration wizard for organisations
setting up the Net Zero Starter Pack.

Wizard Steps (6):
    1. organization_profile  -- Name, sector, region, size, revenue
    2. boundary_selection    -- Operational/financial control or equity share
    3. scope_configuration   -- Which scopes, Scope 3 category selection
    4. data_source_setup     -- ERP type, file formats, API connections
    5. target_preferences    -- Ambition level, pathway, target year
    6. preset_selection      -- Auto-recommend based on sector + size

Sector Presets (6):
    manufacturing, services, technology, retail, financial_services, energy

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-021 Net Zero Starter Pack
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class NetZeroWizardStep(str, Enum):
    """Names of wizard steps in execution order."""

    ORGANIZATION_PROFILE = "organization_profile"
    BOUNDARY_SELECTION = "boundary_selection"
    SCOPE_CONFIGURATION = "scope_configuration"
    DATA_SOURCE_SETUP = "data_source_setup"
    TARGET_PREFERENCES = "target_preferences"
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

class AmbitionLevel(str, Enum):
    """Target ambition levels."""

    AMBITIOUS = "1.5C"
    MODERATE = "well_below_2C"
    STANDARD = "2C"

class OrganizationSize(str, Enum):
    """Organization size classification."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class OrganizationProfile(BaseModel):
    """Organization profile from step 1."""

    organization_name: str = Field(..., min_length=1, max_length=255)
    sector: str = Field(default="general")
    sub_sector: str = Field(default="")
    region: str = Field(default="EU")
    country: str = Field(default="DE")
    employee_count: int = Field(default=500, ge=1)
    annual_revenue_eur: float = Field(default=100_000_000.0, ge=0)
    size: OrganizationSize = Field(default=OrganizationSize.MEDIUM)
    is_listed: bool = Field(default=False)
    nace_code: str = Field(default="")
    fiscal_year_end: str = Field(default="12-31")

class BoundarySelection(BaseModel):
    """Boundary selection from step 2."""

    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
    )
    include_subsidiaries: bool = Field(default=True)
    subsidiary_count: int = Field(default=0, ge=0)
    joint_ventures: int = Field(default=0, ge=0)
    equity_share_threshold_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    countries_of_operation: List[str] = Field(default_factory=lambda: ["DE"])

class ScopeConfiguration(BaseModel):
    """Scope configuration from step 3."""

    include_scope_1: bool = Field(default=True)
    include_scope_2: bool = Field(default=True)
    include_scope_3: bool = Field(default=True)
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based", "market_based"],
    )
    scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7],
        description="Scope 3 category numbers (1-15)",
    )
    scope3_screening_done: bool = Field(default=False)
    scope1_sources: List[str] = Field(
        default_factory=lambda: ["stationary_combustion", "mobile_combustion"],
    )

class DataSourceSetup(BaseModel):
    """Data source setup from step 4."""

    erp_system: str = Field(default="none")
    erp_connected: bool = Field(default=False)
    file_formats: List[str] = Field(
        default_factory=lambda: ["excel", "csv"],
    )
    api_connections: List[str] = Field(default_factory=list)
    utility_provider_apis: bool = Field(default=False)
    travel_management_system: str = Field(default="none")
    procurement_system: str = Field(default="none")

class TargetPreferences(BaseModel):
    """Target preferences from step 5."""

    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.AMBITIOUS)
    pathway: str = Field(default="1.5C")
    base_year: int = Field(default=2019, ge=2015, le=2025)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2035)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2060)
    include_offset_strategy: bool = Field(default=False)
    sbti_submission_planned: bool = Field(default=True)
    net_zero_commitment_year: int = Field(default=2050, ge=2030, le=2060)

class PresetSelection(BaseModel):
    """Preset selection from step 6."""

    preset_name: str = Field(default="")
    preset_applied: bool = Field(default=False)
    engines_enabled: List[str] = Field(default_factory=list)
    scope3_priority: List[int] = Field(default_factory=list)
    recommended_levers: List[str] = Field(default_factory=list)

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: NetZeroWizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)

class WizardState(BaseModel):
    """Complete state of the setup wizard."""

    wizard_id: str = Field(default="")
    current_step: NetZeroWizardStep = Field(default=NetZeroWizardStep.ORGANIZATION_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    org_profile: Optional[OrganizationProfile] = Field(None)
    boundary: Optional[BoundarySelection] = Field(None)
    scope_config: Optional[ScopeConfiguration] = Field(None)
    data_sources: Optional[DataSourceSetup] = Field(None)
    target_prefs: Optional[TargetPreferences] = Field(None)
    preset: Optional[PresetSelection] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class SetupResult(BaseModel):
    """Final setup result with generated configuration."""

    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    sector: str = Field(default="")
    consolidation_approach: str = Field(default="")
    scopes_included: List[str] = Field(default_factory=list)
    scope3_categories: List[int] = Field(default_factory=list)
    pathway: str = Field(default="")
    base_year: int = Field(default=2019)
    target_year: int = Field(default=2030)
    engines_enabled: List[str] = Field(default_factory=list)
    recommended_levers: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=6)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[NetZeroWizardStep] = [
    NetZeroWizardStep.ORGANIZATION_PROFILE,
    NetZeroWizardStep.BOUNDARY_SELECTION,
    NetZeroWizardStep.SCOPE_CONFIGURATION,
    NetZeroWizardStep.DATA_SOURCE_SETUP,
    NetZeroWizardStep.TARGET_PREFERENCES,
    NetZeroWizardStep.PRESET_SELECTION,
]

STEP_DISPLAY_NAMES: Dict[NetZeroWizardStep, str] = {
    NetZeroWizardStep.ORGANIZATION_PROFILE: "Organization Profile",
    NetZeroWizardStep.BOUNDARY_SELECTION: "Boundary Selection",
    NetZeroWizardStep.SCOPE_CONFIGURATION: "Scope Configuration",
    NetZeroWizardStep.DATA_SOURCE_SETUP: "Data Source Setup",
    NetZeroWizardStep.TARGET_PREFERENCES: "Target Preferences",
    NetZeroWizardStep.PRESET_SELECTION: "Preset Selection",
}

# ---------------------------------------------------------------------------
# Sector Presets
# ---------------------------------------------------------------------------

SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "engines": [
            "baseline_calculation_engine", "target_setting_engine",
            "reduction_planning_engine", "progress_tracking_engine",
            "scenario_analysis_engine", "benchmark_engine",
        ],
        "scope3_priority": [1, 2, 3, 4, 5, 6, 7, 9, 12],
        "recommended_levers": [
            "energy_efficiency", "renewable_energy", "electrification",
            "fuel_switching", "process_innovation", "supplier_engagement",
        ],
        "scope1_dominant": True,
        "scope3_dominant_categories": [1, 4],
    },
    "services": {
        "engines": [
            "baseline_calculation_engine", "target_setting_engine",
            "reduction_planning_engine", "progress_tracking_engine",
            "benchmark_engine",
        ],
        "scope3_priority": [1, 3, 5, 6, 7, 8],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "building_decarbonisation",
            "fleet_decarbonisation", "green_procurement",
        ],
        "scope1_dominant": False,
        "scope3_dominant_categories": [1, 6, 7],
    },
    "technology": {
        "engines": [
            "baseline_calculation_engine", "target_setting_engine",
            "reduction_planning_engine", "progress_tracking_engine",
            "benchmark_engine",
        ],
        "scope3_priority": [1, 2, 3, 6, 7, 11],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "supplier_engagement",
            "green_procurement", "demand_reduction",
        ],
        "scope1_dominant": False,
        "scope3_dominant_categories": [1, 2, 11],
    },
    "retail": {
        "engines": [
            "baseline_calculation_engine", "target_setting_engine",
            "reduction_planning_engine", "progress_tracking_engine",
            "scenario_analysis_engine", "benchmark_engine",
        ],
        "scope3_priority": [1, 4, 5, 7, 9, 12],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "supplier_engagement",
            "fleet_decarbonisation", "green_procurement",
        ],
        "scope1_dominant": False,
        "scope3_dominant_categories": [1, 4, 9],
    },
    "financial_services": {
        "engines": [
            "baseline_calculation_engine", "target_setting_engine",
            "reduction_planning_engine", "progress_tracking_engine",
            "benchmark_engine",
        ],
        "scope3_priority": [1, 6, 7, 15],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "building_decarbonisation",
            "green_procurement",
        ],
        "scope1_dominant": False,
        "scope3_dominant_categories": [15],
    },
    "energy": {
        "engines": [
            "baseline_calculation_engine", "target_setting_engine",
            "reduction_planning_engine", "offset_strategy_engine",
            "progress_tracking_engine", "scenario_analysis_engine",
            "benchmark_engine",
        ],
        "scope3_priority": [1, 3, 4, 9, 10, 11],
        "recommended_levers": [
            "renewable_energy", "ccus", "fuel_switching",
            "electrification", "process_innovation", "energy_efficiency",
        ],
        "scope1_dominant": True,
        "scope3_dominant_categories": [3, 9, 11],
    },
}

# ---------------------------------------------------------------------------
# NetZeroSetupWizard
# ---------------------------------------------------------------------------

class NetZeroSetupWizard:
    """6-step guided configuration wizard for PACK-021.

    Guides organisations through net-zero setup with sector presets that
    auto-configure engines, Scope 3 priorities, and recommended levers.

    Example:
        >>> wizard = NetZeroSetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("organization_profile", {...})
        >>> result = wizard.generate_config()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Net Zero Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            NetZeroWizardStep.ORGANIZATION_PROFILE: self._handle_org_profile,
            NetZeroWizardStep.BOUNDARY_SELECTION: self._handle_boundary,
            NetZeroWizardStep.SCOPE_CONFIGURATION: self._handle_scope_config,
            NetZeroWizardStep.DATA_SOURCE_SETUP: self._handle_data_sources,
            NetZeroWizardStep.TARGET_PREFERENCES: self._handle_target_prefs,
            NetZeroWizardStep.PRESET_SELECTION: self._handle_preset,
        }
        self.logger.info("NetZeroSetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState.
        """
        wizard_id = _compute_hash(f"nz-wizard:{utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
        self.logger.info("Net Zero wizard started: %s", wizard_id)
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """Complete a wizard step with provided data.

        Args:
            step_name: Step name to complete.
            data: Step configuration data.

        Returns:
            Updated WizardState.

        Raises:
            RuntimeError: If wizard not started.
            ValueError: If step name invalid.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = NetZeroWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in NetZeroWizardStep]
            raise ValueError(f"Unknown step '{step_name}'. Valid: {valid}")

        step = self._state.steps.get(step_name)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = utcnow()
        start_time = time.monotonic()

        handler = self._step_handlers.get(step_enum)
        if handler is None:
            raise ValueError(f"No handler for step '{step_name}'")

        try:
            errors = handler(data)
            elapsed = (time.monotonic() - start_time) * 1000
            step.execution_time_ms = elapsed
            step.data = data

            if errors:
                step.status = StepStatus.FAILED
                step.validation_errors = errors
            else:
                step.status = StepStatus.COMPLETED
                step.completed_at = utcnow()
                step.validation_errors = []
                self._advance_step(step_enum)
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

        return self._state

    def generate_config(self) -> SetupResult:
        """Generate the final pack configuration from wizard state.

        Returns:
            SetupResult with complete configuration.
        """
        return self._generate_result()

    def run_demo(self) -> SetupResult:
        """Execute a pre-configured demo setup for a manufacturing company.

        Returns:
            SetupResult with demo configuration.
        """
        self.start()

        demo_steps = {
            "organization_profile": {
                "organization_name": "Demo Manufacturing Corp",
                "sector": "manufacturing",
                "region": "EU",
                "country": "DE",
                "employee_count": 5000,
                "annual_revenue_eur": 500_000_000.0,
                "size": "large",
                "is_listed": True,
                "nace_code": "C28.1",
            },
            "boundary_selection": {
                "consolidation_approach": "operational_control",
                "include_subsidiaries": True,
                "subsidiary_count": 3,
                "countries_of_operation": ["DE", "US", "CN"],
            },
            "scope_configuration": {
                "include_scope_1": True,
                "include_scope_2": True,
                "include_scope_3": True,
                "scope2_methods": ["location_based", "market_based"],
                "scope3_categories": [1, 2, 3, 4, 5, 6, 7, 9, 12],
                "scope1_sources": [
                    "stationary_combustion", "mobile_combustion",
                    "process_emissions", "fugitive_emissions",
                ],
            },
            "data_source_setup": {
                "erp_system": "sap",
                "erp_connected": True,
                "file_formats": ["excel", "csv"],
                "utility_provider_apis": False,
                "travel_management_system": "concur",
            },
            "target_preferences": {
                "ambition_level": "1.5C",
                "pathway": "1.5C",
                "base_year": 2019,
                "near_term_target_year": 2030,
                "long_term_target_year": 2050,
                "include_offset_strategy": False,
                "sbti_submission_planned": True,
                "net_zero_commitment_year": 2050,
            },
            "preset_selection": {
                "preset_name": "manufacturing",
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_sector_preset(self, sector: str) -> Optional[Dict[str, Any]]:
        """Get the preset configuration for a sector.

        Args:
            sector: Sector key (e.g., 'manufacturing', 'services').

        Returns:
            Preset configuration dict, or None if not found.
        """
        return SECTOR_PRESETS.get(sector)

    def recommend_preset(
        self,
        sector: str,
        size: str = "medium",
    ) -> Dict[str, Any]:
        """Recommend a preset based on sector and organization size.

        Args:
            sector: Organization sector.
            size: Organization size.

        Returns:
            Recommended preset with justification.
        """
        preset = SECTOR_PRESETS.get(sector)
        if preset is None:
            # Default to services
            preset = SECTOR_PRESETS["services"]
            sector = "services"

        recommendation = {
            "recommended_preset": sector,
            "preset": preset,
            "justification": f"Selected '{sector}' preset based on sector classification",
        }

        # Adjust for size
        if size in ("small", "medium"):
            # Reduce Scope 3 requirements for smaller orgs
            recommendation["scope3_adjustment"] = "Consider fewer Scope 3 categories for initial setup"
            recommendation["simplified_scope3"] = preset["scope3_priority"][:5]

        return recommendation

    # ---- Step Handlers ----

    def _handle_org_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle organization profile step."""
        errors: List[str] = []
        try:
            profile = OrganizationProfile(**data)
            if self._state:
                self._state.org_profile = profile
        except Exception as exc:
            errors.append(f"Invalid organization profile: {exc}")
        return errors

    def _handle_boundary(self, data: Dict[str, Any]) -> List[str]:
        """Handle boundary selection step."""
        errors: List[str] = []
        try:
            boundary = BoundarySelection(**data)
            if self._state:
                self._state.boundary = boundary
        except Exception as exc:
            errors.append(f"Invalid boundary selection: {exc}")
        return errors

    def _handle_scope_config(self, data: Dict[str, Any]) -> List[str]:
        """Handle scope configuration step."""
        errors: List[str] = []
        try:
            scope_config = ScopeConfiguration(**data)

            # Validate Scope 3 categories are in range
            for cat in scope_config.scope3_categories:
                if cat < 1 or cat > 15:
                    errors.append(f"Invalid Scope 3 category: {cat} (must be 1-15)")

            if not errors and self._state:
                self._state.scope_config = scope_config
        except Exception as exc:
            errors.append(f"Invalid scope configuration: {exc}")
        return errors

    def _handle_data_sources(self, data: Dict[str, Any]) -> List[str]:
        """Handle data source setup step."""
        errors: List[str] = []
        try:
            data_setup = DataSourceSetup(**data)
            if self._state:
                self._state.data_sources = data_setup
        except Exception as exc:
            errors.append(f"Invalid data source setup: {exc}")
        return errors

    def _handle_target_prefs(self, data: Dict[str, Any]) -> List[str]:
        """Handle target preferences step."""
        errors: List[str] = []
        try:
            prefs = TargetPreferences(**data)

            # Validate timeline
            if prefs.near_term_target_year <= prefs.base_year:
                errors.append("Near-term target year must be after base year")
            if prefs.long_term_target_year <= prefs.near_term_target_year:
                errors.append("Long-term target year must be after near-term target year")

            if not errors and self._state:
                self._state.target_prefs = prefs
        except Exception as exc:
            errors.append(f"Invalid target preferences: {exc}")
        return errors

    def _handle_preset(self, data: Dict[str, Any]) -> List[str]:
        """Handle preset selection step with auto-apply."""
        errors: List[str] = []
        preset_name = data.get("preset_name", "")

        if not preset_name:
            # Auto-detect from org profile
            if self._state and self._state.org_profile:
                preset_name = self._state.org_profile.sector

        preset = SECTOR_PRESETS.get(preset_name)
        if preset is None:
            errors.append(
                f"Unknown preset '{preset_name}'. Valid: {sorted(SECTOR_PRESETS.keys())}"
            )
            return errors

        selection = PresetSelection(
            preset_name=preset_name,
            preset_applied=True,
            engines_enabled=preset.get("engines", []),
            scope3_priority=preset.get("scope3_priority", []),
            recommended_levers=preset.get("recommended_levers", []),
        )

        if self._state:
            self._state.preset = selection

        self.logger.info("Sector preset applied: %s", preset_name)
        return errors

    # ---- Navigation ----

    def _advance_step(self, current: NetZeroWizardStep) -> None:
        """Advance to the next step."""
        if self._state is None:
            return
        try:
            idx = STEP_ORDER.index(current)
            if idx < len(STEP_ORDER) - 1:
                self._state.current_step = STEP_ORDER[idx + 1]
            else:
                self._state.is_complete = True
                self._state.completed_at = utcnow()
        except ValueError:
            pass

    # ---- Result Generation ----

    def _generate_result(self) -> SetupResult:
        """Generate the final setup result from wizard state."""
        if self._state is None:
            return SetupResult()

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        # Build scopes list
        scopes = []
        if self._state.scope_config:
            if self._state.scope_config.include_scope_1:
                scopes.append("scope_1")
            if self._state.scope_config.include_scope_2:
                scopes.append("scope_2")
            if self._state.scope_config.include_scope_3:
                scopes.append("scope_3")

        scope3_cats = []
        if self._state.scope_config:
            scope3_cats = list(self._state.scope_config.scope3_categories)

        engines = []
        levers = []
        if self._state.preset:
            engines = list(self._state.preset.engines_enabled)
            levers = list(self._state.preset.recommended_levers)

        config_hash = _compute_hash({
            "org": self._state.org_profile.organization_name if self._state.org_profile else "",
            "sector": self._state.org_profile.sector if self._state.org_profile else "",
            "scopes": scopes,
            "scope3": scope3_cats,
        })

        result = SetupResult(
            organization_name=(
                self._state.org_profile.organization_name if self._state.org_profile else ""
            ),
            sector=(
                self._state.org_profile.sector if self._state.org_profile else ""
            ),
            consolidation_approach=(
                self._state.boundary.consolidation_approach.value
                if self._state.boundary else ""
            ),
            scopes_included=scopes,
            scope3_categories=scope3_cats,
            pathway=(
                self._state.target_prefs.pathway if self._state.target_prefs else ""
            ),
            base_year=(
                self._state.target_prefs.base_year if self._state.target_prefs else 2019
            ),
            target_year=(
                self._state.target_prefs.near_term_target_year
                if self._state.target_prefs else 2030
            ),
            engines_enabled=engines,
            recommended_levers=levers,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
