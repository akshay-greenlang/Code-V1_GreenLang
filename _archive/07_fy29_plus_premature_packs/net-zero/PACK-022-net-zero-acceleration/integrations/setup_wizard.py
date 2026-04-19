# -*- coding: utf-8 -*-
"""
NetZeroAccelerationSetupWizard - 8-Step Guided Configuration for PACK-022
============================================================================

This module implements an 8-step configuration wizard for organisations
setting up the Net Zero Acceleration Pack, building on PACK-021 setup.

Wizard Steps (8):
    1. organization_profile    -- Name, sector, region, size, revenue
    2. pack021_status          -- Verify PACK-021 is configured and baseline exists
    3. scope3_strategy         -- Scope 3 deep-dive: priority categories, data methods
    4. sda_sector_selection    -- Select SDA sector if applicable
    5. supplier_programme      -- Supplier engagement programme setup
    6. finance_integration     -- CapEx planning, internal carbon price, taxonomy
    7. assurance_level         -- Target assurance level (limited/reasonable)
    8. preset_selection        -- Auto-recommend based on sector + acceleration needs

Sector Presets (8):
    manufacturing_sda, services_acceleration, technology_acceleration,
    retail_acceleration, financial_services_acceleration, energy_transition,
    heavy_industry_sda, transport_acceleration

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-022 Net Zero Acceleration Pack
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

class AccelerationWizardStep(str, Enum):
    """Names of wizard steps in execution order."""

    ORGANIZATION_PROFILE = "organization_profile"
    PACK021_STATUS = "pack021_status"
    SCOPE3_STRATEGY = "scope3_strategy"
    SDA_SECTOR_SELECTION = "sda_sector_selection"
    SUPPLIER_PROGRAMME = "supplier_programme"
    FINANCE_INTEGRATION = "finance_integration"
    ASSURANCE_LEVEL = "assurance_level"
    PRESET_SELECTION = "preset_selection"

class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class AssuranceLevelChoice(str, Enum):
    """Target assurance levels."""

    LIMITED = "limited"
    REASONABLE = "reasonable"
    NONE = "none"

class OrganizationSize(str, Enum):
    """Organization size classification."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class SupplierEngagementScope(str, Enum):
    """Scope of supplier engagement programme."""

    TOP_10 = "top_10"
    TOP_50 = "top_50"
    TOP_100 = "top_100"
    ALL_STRATEGIC = "all_strategic"

# ---------------------------------------------------------------------------
# Step Data Models
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
    multi_entity: bool = Field(default=False)
    entity_count: int = Field(default=1, ge=1)

class Pack021Status(BaseModel):
    """PACK-021 status from step 2."""

    pack021_configured: bool = Field(default=False)
    baseline_year: int = Field(default=2019, ge=2015, le=2025)
    baseline_total_tco2e: float = Field(default=0.0, ge=0.0)
    targets_set: bool = Field(default=False)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2035)
    near_term_reduction_pct: float = Field(default=42.0, ge=0.0, le=100.0)
    pathway: str = Field(default="1.5C")
    sbti_validated: bool = Field(default=False)
    gap_analysis_available: bool = Field(default=False)

class Scope3Strategy(BaseModel):
    """Scope 3 deep-dive strategy from step 3."""

    priority_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7],
        description="Scope 3 category numbers prioritized for deep dive",
    )
    activity_based_categories: List[int] = Field(
        default_factory=lambda: [1, 3, 6, 7],
        description="Categories with activity-based data available",
    )
    spend_based_categories: List[int] = Field(
        default_factory=lambda: [2, 4, 5],
        description="Categories using spend-based estimates only",
    )
    supplier_specific_data: bool = Field(default=False)
    screening_complete: bool = Field(default=True)

class SDASectorSelection(BaseModel):
    """SDA sector selection from step 4."""

    is_sda_sector: bool = Field(default=False)
    sda_sector: str = Field(default="")
    sda_convergence_year: int = Field(default=2050, ge=2030, le=2060)
    intensity_metric: str = Field(default="")
    current_intensity: float = Field(default=0.0, ge=0.0)
    benchmark_intensity: float = Field(default=0.0, ge=0.0)

class SupplierProgramme(BaseModel):
    """Supplier engagement programme from step 5."""

    engagement_scope: SupplierEngagementScope = Field(
        default=SupplierEngagementScope.TOP_50,
    )
    target_supplier_count: int = Field(default=50, ge=1)
    require_sbti_targets: bool = Field(default=False)
    require_cdp_disclosure: bool = Field(default=False)
    data_collection_method: str = Field(default="questionnaire")
    engagement_timeline_months: int = Field(default=12, ge=3, le=36)

class FinanceIntegration(BaseModel):
    """Climate finance integration from step 6."""

    internal_carbon_price_eur: float = Field(default=0.0, ge=0.0)
    capex_budget_eur: float = Field(default=0.0, ge=0.0)
    taxonomy_reporting_required: bool = Field(default=True)
    green_bond_eligible: bool = Field(default=False)
    transition_plan_required: bool = Field(default=True)

class AssuranceLevelConfig(BaseModel):
    """Assurance level configuration from step 7."""

    target_assurance_level: AssuranceLevelChoice = Field(
        default=AssuranceLevelChoice.LIMITED,
    )
    assurance_provider: str = Field(default="")
    scope1_scope2_assurance: bool = Field(default=True)
    scope3_assurance: bool = Field(default=False)
    target_assurance_year: int = Field(default=2026, ge=2025, le=2030)

class PresetSelection(BaseModel):
    """Preset selection from step 8."""

    preset_name: str = Field(default="")
    preset_applied: bool = Field(default=False)
    engines_enabled: List[str] = Field(default_factory=list)
    scope3_priority: List[int] = Field(default_factory=list)
    recommended_levers: List[str] = Field(default_factory=list)
    sda_enabled: bool = Field(default=False)
    supplier_engagement_enabled: bool = Field(default=True)
    assurance_preparation_enabled: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Wizard State Models
# ---------------------------------------------------------------------------

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: AccelerationWizardStep = Field(...)
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
    current_step: AccelerationWizardStep = Field(
        default=AccelerationWizardStep.ORGANIZATION_PROFILE,
    )
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    org_profile: Optional[OrganizationProfile] = Field(None)
    pack021_status: Optional[Pack021Status] = Field(None)
    scope3_strategy: Optional[Scope3Strategy] = Field(None)
    sda_selection: Optional[SDASectorSelection] = Field(None)
    supplier_programme: Optional[SupplierProgramme] = Field(None)
    finance_integration: Optional[FinanceIntegration] = Field(None)
    assurance_config: Optional[AssuranceLevelConfig] = Field(None)
    preset: Optional[PresetSelection] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class SetupResult(BaseModel):
    """Final setup result with generated configuration."""

    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    sector: str = Field(default="")
    multi_entity: bool = Field(default=False)
    entity_count: int = Field(default=1)
    baseline_tco2e: float = Field(default=0.0)
    pathway: str = Field(default="")
    is_sda_sector: bool = Field(default=False)
    sda_sector: str = Field(default="")
    scope3_priority_categories: List[int] = Field(default_factory=list)
    supplier_engagement_scope: str = Field(default="")
    supplier_target_count: int = Field(default=0)
    assurance_level: str = Field(default="")
    internal_carbon_price_eur: float = Field(default=0.0)
    engines_enabled: List[str] = Field(default_factory=list)
    recommended_levers: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[AccelerationWizardStep] = [
    AccelerationWizardStep.ORGANIZATION_PROFILE,
    AccelerationWizardStep.PACK021_STATUS,
    AccelerationWizardStep.SCOPE3_STRATEGY,
    AccelerationWizardStep.SDA_SECTOR_SELECTION,
    AccelerationWizardStep.SUPPLIER_PROGRAMME,
    AccelerationWizardStep.FINANCE_INTEGRATION,
    AccelerationWizardStep.ASSURANCE_LEVEL,
    AccelerationWizardStep.PRESET_SELECTION,
]

STEP_DISPLAY_NAMES: Dict[AccelerationWizardStep, str] = {
    AccelerationWizardStep.ORGANIZATION_PROFILE: "Organization Profile",
    AccelerationWizardStep.PACK021_STATUS: "PACK-021 Status",
    AccelerationWizardStep.SCOPE3_STRATEGY: "Scope 3 Strategy",
    AccelerationWizardStep.SDA_SECTOR_SELECTION: "SDA Sector Selection",
    AccelerationWizardStep.SUPPLIER_PROGRAMME: "Supplier Programme",
    AccelerationWizardStep.FINANCE_INTEGRATION: "Finance Integration",
    AccelerationWizardStep.ASSURANCE_LEVEL: "Assurance Level",
    AccelerationWizardStep.PRESET_SELECTION: "Preset Selection",
}

# ---------------------------------------------------------------------------
# Sector Presets (8)
# ---------------------------------------------------------------------------

SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing_sda": {
        "engines": [
            "scenario_analysis_engine", "sda_pathway_engine",
            "supplier_engagement_engine", "climate_finance_engine",
            "progress_analytics_engine", "temperature_scoring_engine",
            "variance_decomposition_engine", "monte_carlo_engine",
            "capex_planning_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 2, 3, 4, 5, 6, 7, 9, 12],
        "recommended_levers": [
            "energy_efficiency", "renewable_energy", "electrification",
            "fuel_switching", "process_innovation", "supplier_engagement",
            "waste_heat_recovery", "ccus",
        ],
        "sda_enabled": True,
        "sda_sectors": ["cement", "steel", "aluminium", "pulp_paper"],
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
    "services_acceleration": {
        "engines": [
            "scenario_analysis_engine", "supplier_engagement_engine",
            "climate_finance_engine", "progress_analytics_engine",
            "temperature_scoring_engine", "variance_decomposition_engine",
            "capex_planning_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 3, 5, 6, 7, 8],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "building_decarbonisation",
            "fleet_decarbonisation", "green_procurement", "remote_work",
        ],
        "sda_enabled": False,
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
    "technology_acceleration": {
        "engines": [
            "scenario_analysis_engine", "supplier_engagement_engine",
            "climate_finance_engine", "progress_analytics_engine",
            "temperature_scoring_engine", "variance_decomposition_engine",
            "capex_planning_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 2, 3, 6, 7, 11],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "supplier_engagement",
            "green_procurement", "data_center_efficiency", "product_lifecycle",
        ],
        "sda_enabled": False,
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
    "retail_acceleration": {
        "engines": [
            "scenario_analysis_engine", "supplier_engagement_engine",
            "climate_finance_engine", "progress_analytics_engine",
            "temperature_scoring_engine", "variance_decomposition_engine",
            "capex_planning_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 4, 5, 7, 9, 12],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "supplier_engagement",
            "fleet_decarbonisation", "green_procurement", "refrigerant_management",
        ],
        "sda_enabled": False,
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
    "financial_services_acceleration": {
        "engines": [
            "scenario_analysis_engine", "climate_finance_engine",
            "progress_analytics_engine", "temperature_scoring_engine",
            "variance_decomposition_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 6, 7, 15],
        "recommended_levers": [
            "renewable_energy", "energy_efficiency", "building_decarbonisation",
            "green_procurement", "portfolio_decarbonisation",
        ],
        "sda_enabled": False,
        "supplier_engagement_enabled": False,
        "assurance_preparation_enabled": True,
    },
    "energy_transition": {
        "engines": [
            "scenario_analysis_engine", "sda_pathway_engine",
            "supplier_engagement_engine", "climate_finance_engine",
            "progress_analytics_engine", "temperature_scoring_engine",
            "variance_decomposition_engine", "monte_carlo_engine",
            "capex_planning_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 3, 4, 9, 10, 11],
        "recommended_levers": [
            "renewable_energy", "ccus", "fuel_switching",
            "electrification", "process_innovation", "energy_efficiency",
            "methane_abatement", "hydrogen",
        ],
        "sda_enabled": True,
        "sda_sectors": ["power_generation"],
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
    "heavy_industry_sda": {
        "engines": [
            "scenario_analysis_engine", "sda_pathway_engine",
            "supplier_engagement_engine", "climate_finance_engine",
            "progress_analytics_engine", "temperature_scoring_engine",
            "variance_decomposition_engine", "monte_carlo_engine",
            "capex_planning_engine", "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 2, 3, 4, 5, 9],
        "recommended_levers": [
            "fuel_switching", "electrification", "ccus",
            "process_innovation", "waste_heat_recovery", "energy_efficiency",
            "hydrogen", "circular_economy",
        ],
        "sda_enabled": True,
        "sda_sectors": ["cement", "steel", "aluminium"],
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
    "transport_acceleration": {
        "engines": [
            "scenario_analysis_engine", "sda_pathway_engine",
            "supplier_engagement_engine", "climate_finance_engine",
            "progress_analytics_engine", "temperature_scoring_engine",
            "variance_decomposition_engine", "capex_planning_engine",
            "acceleration_reporting_engine",
        ],
        "scope3_priority": [1, 3, 4, 9, 11],
        "recommended_levers": [
            "fleet_decarbonisation", "electrification", "fuel_switching",
            "route_optimisation", "modal_shift", "energy_efficiency",
        ],
        "sda_enabled": True,
        "sda_sectors": ["transport"],
        "supplier_engagement_enabled": True,
        "assurance_preparation_enabled": True,
    },
}

# ---------------------------------------------------------------------------
# NetZeroAccelerationSetupWizard
# ---------------------------------------------------------------------------

class NetZeroAccelerationSetupWizard:
    """8-step guided configuration wizard for PACK-022.

    Guides organisations through net-zero acceleration setup with
    SDA sector selection, supplier engagement programme design,
    climate finance integration, assurance level targeting, and
    8 sector presets.

    Example:
        >>> wizard = NetZeroAccelerationSetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("organization_profile", {...})
        >>> result = wizard.generate_config()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Net Zero Acceleration Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            AccelerationWizardStep.ORGANIZATION_PROFILE: self._handle_org_profile,
            AccelerationWizardStep.PACK021_STATUS: self._handle_pack021_status,
            AccelerationWizardStep.SCOPE3_STRATEGY: self._handle_scope3_strategy,
            AccelerationWizardStep.SDA_SECTOR_SELECTION: self._handle_sda_selection,
            AccelerationWizardStep.SUPPLIER_PROGRAMME: self._handle_supplier_programme,
            AccelerationWizardStep.FINANCE_INTEGRATION: self._handle_finance_integration,
            AccelerationWizardStep.ASSURANCE_LEVEL: self._handle_assurance_level,
            AccelerationWizardStep.PRESET_SELECTION: self._handle_preset,
        }
        self.logger.info("NetZeroAccelerationSetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState.
        """
        wizard_id = _compute_hash(
            f"nz-accel-wizard:{utcnow().isoformat()}"
        )[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(
            wizard_id=wizard_id,
            current_step=STEP_ORDER[0],
            steps=steps,
        )
        self.logger.info("Acceleration wizard started: %s", wizard_id)
        return self._state

    def complete_step(
        self, step_name: str, data: Dict[str, Any],
    ) -> WizardState:
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
            step_enum = AccelerationWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in AccelerationWizardStep]
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
        """Execute a pre-configured demo setup for a manufacturing SDA company.

        Returns:
            SetupResult with demo configuration.
        """
        self.start()

        demo_steps = {
            "organization_profile": {
                "organization_name": "Demo Heavy Manufacturing Corp",
                "sector": "manufacturing",
                "sub_sector": "cement",
                "region": "EU",
                "country": "DE",
                "employee_count": 8000,
                "annual_revenue_eur": 2_000_000_000.0,
                "size": "enterprise",
                "is_listed": True,
                "nace_code": "C23.5",
                "multi_entity": True,
                "entity_count": 5,
            },
            "pack021_status": {
                "pack021_configured": True,
                "baseline_year": 2019,
                "baseline_total_tco2e": 250_000.0,
                "targets_set": True,
                "near_term_target_year": 2030,
                "near_term_reduction_pct": 42.0,
                "pathway": "1.5C",
                "sbti_validated": True,
                "gap_analysis_available": True,
            },
            "scope3_strategy": {
                "priority_categories": [1, 2, 3, 4, 5, 6, 7, 9],
                "activity_based_categories": [1, 3, 6, 7],
                "spend_based_categories": [2, 4, 5, 9],
                "supplier_specific_data": True,
                "screening_complete": True,
            },
            "sda_sector_selection": {
                "is_sda_sector": True,
                "sda_sector": "cement",
                "sda_convergence_year": 2050,
                "intensity_metric": "tCO2e_per_tonne_clinker",
                "current_intensity": 0.85,
                "benchmark_intensity": 0.52,
            },
            "supplier_programme": {
                "engagement_scope": "top_50",
                "target_supplier_count": 50,
                "require_sbti_targets": True,
                "require_cdp_disclosure": True,
                "data_collection_method": "questionnaire",
                "engagement_timeline_months": 18,
            },
            "finance_integration": {
                "internal_carbon_price_eur": 100.0,
                "capex_budget_eur": 50_000_000.0,
                "taxonomy_reporting_required": True,
                "green_bond_eligible": True,
                "transition_plan_required": True,
            },
            "assurance_level": {
                "target_assurance_level": "limited",
                "scope1_scope2_assurance": True,
                "scope3_assurance": False,
                "target_assurance_year": 2026,
            },
            "preset_selection": {
                "preset_name": "manufacturing_sda",
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_sector_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get the preset configuration for a sector.

        Args:
            preset_name: Preset key.

        Returns:
            Preset configuration dict, or None if not found.
        """
        return SECTOR_PRESETS.get(preset_name)

    def recommend_preset(
        self,
        sector: str,
        is_sda: bool = False,
        size: str = "medium",
    ) -> Dict[str, Any]:
        """Recommend a preset based on sector, SDA status, and size.

        Args:
            sector: Organization sector.
            is_sda: Whether organization operates in an SDA sector.
            size: Organization size.

        Returns:
            Recommended preset with justification.
        """
        # Match by sector and SDA
        if is_sda:
            if sector == "energy":
                preset_name = "energy_transition"
            elif sector in ("transport", "logistics"):
                preset_name = "transport_acceleration"
            elif sector in ("manufacturing", "industrial"):
                preset_name = "heavy_industry_sda"
            else:
                preset_name = "manufacturing_sda"
        else:
            sector_map = {
                "manufacturing": "manufacturing_sda",
                "services": "services_acceleration",
                "technology": "technology_acceleration",
                "retail": "retail_acceleration",
                "financial_services": "financial_services_acceleration",
                "energy": "energy_transition",
            }
            preset_name = sector_map.get(sector, "services_acceleration")

        preset = SECTOR_PRESETS.get(preset_name, SECTOR_PRESETS["services_acceleration"])

        recommendation = {
            "recommended_preset": preset_name,
            "preset": preset,
            "justification": (
                f"Selected '{preset_name}' preset based on sector='{sector}', "
                f"SDA={is_sda}, size='{size}'"
            ),
        }

        # Size-based adjustments
        if size in ("small", "medium"):
            recommendation["scope3_adjustment"] = (
                "Consider fewer Scope 3 categories for initial setup"
            )
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

    def _handle_pack021_status(self, data: Dict[str, Any]) -> List[str]:
        """Handle PACK-021 status verification step."""
        errors: List[str] = []
        try:
            status = Pack021Status(**data)
            if not status.pack021_configured:
                errors.append(
                    "PACK-021 must be configured before using PACK-022. "
                    "Run PACK-021 setup wizard first."
                )
            if not status.targets_set:
                errors.append(
                    "Science-based targets must be set in PACK-021 before acceleration."
                )
            if not errors and self._state:
                self._state.pack021_status = status
        except Exception as exc:
            errors.append(f"Invalid PACK-021 status: {exc}")
        return errors

    def _handle_scope3_strategy(self, data: Dict[str, Any]) -> List[str]:
        """Handle Scope 3 strategy step."""
        errors: List[str] = []
        try:
            strategy = Scope3Strategy(**data)
            for cat in strategy.priority_categories:
                if cat < 1 or cat > 15:
                    errors.append(
                        f"Invalid Scope 3 category: {cat} (must be 1-15)"
                    )
            if not strategy.screening_complete:
                errors.append("Scope 3 screening must be complete before deep-dive")
            if not errors and self._state:
                self._state.scope3_strategy = strategy
        except Exception as exc:
            errors.append(f"Invalid Scope 3 strategy: {exc}")
        return errors

    def _handle_sda_selection(self, data: Dict[str, Any]) -> List[str]:
        """Handle SDA sector selection step."""
        errors: List[str] = []
        try:
            selection = SDASectorSelection(**data)
            valid_sectors = [
                "power_generation", "cement", "steel", "aluminium",
                "pulp_paper", "transport", "buildings", "agriculture",
            ]
            if selection.is_sda_sector and selection.sda_sector not in valid_sectors:
                errors.append(
                    f"Invalid SDA sector '{selection.sda_sector}'. "
                    f"Valid: {valid_sectors}"
                )
            if not errors and self._state:
                self._state.sda_selection = selection
        except Exception as exc:
            errors.append(f"Invalid SDA selection: {exc}")
        return errors

    def _handle_supplier_programme(self, data: Dict[str, Any]) -> List[str]:
        """Handle supplier engagement programme step."""
        errors: List[str] = []
        try:
            programme = SupplierProgramme(**data)
            if self._state:
                self._state.supplier_programme = programme
        except Exception as exc:
            errors.append(f"Invalid supplier programme: {exc}")
        return errors

    def _handle_finance_integration(self, data: Dict[str, Any]) -> List[str]:
        """Handle climate finance integration step."""
        errors: List[str] = []
        try:
            finance = FinanceIntegration(**data)
            if self._state:
                self._state.finance_integration = finance
        except Exception as exc:
            errors.append(f"Invalid finance integration: {exc}")
        return errors

    def _handle_assurance_level(self, data: Dict[str, Any]) -> List[str]:
        """Handle assurance level step."""
        errors: List[str] = []
        try:
            assurance = AssuranceLevelConfig(**data)
            if self._state:
                self._state.assurance_config = assurance
        except Exception as exc:
            errors.append(f"Invalid assurance configuration: {exc}")
        return errors

    def _handle_preset(self, data: Dict[str, Any]) -> List[str]:
        """Handle preset selection step with auto-apply."""
        errors: List[str] = []
        preset_name = data.get("preset_name", "")

        if not preset_name:
            # Auto-detect from org profile and SDA selection
            if self._state and self._state.org_profile:
                is_sda = (
                    self._state.sda_selection.is_sda_sector
                    if self._state.sda_selection else False
                )
                rec = self.recommend_preset(
                    self._state.org_profile.sector, is_sda,
                    self._state.org_profile.size.value,
                )
                preset_name = rec["recommended_preset"]

        preset = SECTOR_PRESETS.get(preset_name)
        if preset is None:
            errors.append(
                f"Unknown preset '{preset_name}'. "
                f"Valid: {sorted(SECTOR_PRESETS.keys())}"
            )
            return errors

        selection = PresetSelection(
            preset_name=preset_name,
            preset_applied=True,
            engines_enabled=preset.get("engines", []),
            scope3_priority=preset.get("scope3_priority", []),
            recommended_levers=preset.get("recommended_levers", []),
            sda_enabled=preset.get("sda_enabled", False),
            supplier_engagement_enabled=preset.get("supplier_engagement_enabled", True),
            assurance_preparation_enabled=preset.get("assurance_preparation_enabled", True),
        )

        if self._state:
            self._state.preset = selection

        self.logger.info("Acceleration preset applied: %s", preset_name)
        return errors

    # ---- Navigation ----

    def _advance_step(self, current: AccelerationWizardStep) -> None:
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
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )

        config_hash = _compute_hash({
            "org": (
                self._state.org_profile.organization_name
                if self._state.org_profile else ""
            ),
            "sector": (
                self._state.org_profile.sector
                if self._state.org_profile else ""
            ),
            "sda": (
                self._state.sda_selection.sda_sector
                if self._state.sda_selection else ""
            ),
        })

        result = SetupResult(
            organization_name=(
                self._state.org_profile.organization_name
                if self._state.org_profile else ""
            ),
            sector=(
                self._state.org_profile.sector
                if self._state.org_profile else ""
            ),
            multi_entity=(
                self._state.org_profile.multi_entity
                if self._state.org_profile else False
            ),
            entity_count=(
                self._state.org_profile.entity_count
                if self._state.org_profile else 1
            ),
            baseline_tco2e=(
                self._state.pack021_status.baseline_total_tco2e
                if self._state.pack021_status else 0.0
            ),
            pathway=(
                self._state.pack021_status.pathway
                if self._state.pack021_status else ""
            ),
            is_sda_sector=(
                self._state.sda_selection.is_sda_sector
                if self._state.sda_selection else False
            ),
            sda_sector=(
                self._state.sda_selection.sda_sector
                if self._state.sda_selection else ""
            ),
            scope3_priority_categories=(
                list(self._state.scope3_strategy.priority_categories)
                if self._state.scope3_strategy else []
            ),
            supplier_engagement_scope=(
                self._state.supplier_programme.engagement_scope.value
                if self._state.supplier_programme else ""
            ),
            supplier_target_count=(
                self._state.supplier_programme.target_supplier_count
                if self._state.supplier_programme else 0
            ),
            assurance_level=(
                self._state.assurance_config.target_assurance_level.value
                if self._state.assurance_config else ""
            ),
            internal_carbon_price_eur=(
                self._state.finance_integration.internal_carbon_price_eur
                if self._state.finance_integration else 0.0
            ),
            engines_enabled=(
                list(self._state.preset.engines_enabled)
                if self._state.preset else []
            ),
            recommended_levers=(
                list(self._state.preset.recommended_levers)
                if self._state.preset else []
            ),
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
