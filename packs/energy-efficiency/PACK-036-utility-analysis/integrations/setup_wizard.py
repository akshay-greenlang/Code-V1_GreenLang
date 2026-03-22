# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Utility Account Configuration for PACK-036
========================================================================

This module implements an 8-step configuration wizard for facilities
setting up the Utility Analysis Pack.

Wizard Steps (8):
    1. WELCOME               -- Introduction, terms, scope overview
    2. FACILITY_PROFILE      -- Name, location, size, occupancy
    3. COMMODITY_ACCOUNTS    -- Utility accounts by commodity
    4. METER_MAPPING         -- Meter hierarchy and sub-metering
    5. RATE_SCHEDULES        -- Rate structures and TOU periods
    6. ALLOCATION_RULES      -- Cost allocation methods and entities
    7. BUDGET_PARAMETERS     -- Budget targets and variance thresholds
    8. REVIEW_CONFIRM        -- Review all inputs and confirm

8 Facility Presets with auto-configured analysis parameters.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-036 Utility Analysis
Status: Production Ready
"""

from __future__ import annotations

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


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SetupWizardStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    WELCOME = "welcome"
    FACILITY_PROFILE = "facility_profile"
    COMMODITY_ACCOUNTS = "commodity_accounts"
    METER_MAPPING = "meter_mapping"
    RATE_SCHEDULES = "rate_schedules"
    ALLOCATION_RULES = "allocation_rules"
    BUDGET_PARAMETERS = "budget_parameters"
    REVIEW_CONFIRM = "review_confirm"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class FacilitySetup(BaseModel):
    """Facility profile data from the wizard."""

    facility_name: str = Field(default="", max_length=255)
    facility_id: str = Field(default="")
    country: str = Field(default="DE")
    region: str = Field(default="")
    address: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0)
    employee_count: int = Field(default=0, ge=0)
    operating_hours_per_year: float = Field(default=2000.0, ge=0)
    building_type: str = Field(
        default="office",
        description="office|manufacturing|retail|warehouse|healthcare|education|data_center|portfolio",
    )
    year_built: int = Field(default=2000, ge=1900, le=2030)
    commodities: List[str] = Field(
        default_factory=lambda: ["electricity", "natural_gas"]
    )
    electricity_accounts: int = Field(default=1, ge=0)
    gas_accounts: int = Field(default=1, ge=0)
    water_accounts: int = Field(default=0, ge=0)
    meter_count: int = Field(default=2, ge=0)
    sub_meter_count: int = Field(default=0, ge=0)
    rate_structure: str = Field(
        default="tou", description="flat|tou|demand|tiered|blended"
    )
    allocation_method: str = Field(
        default="sub_metered",
        description="sub_metered|floor_area|headcount|fixed_split",
    )
    cost_centers: int = Field(default=1, ge=1)
    annual_budget_eur: float = Field(default=0.0, ge=0)
    variance_threshold_pct: float = Field(default=10.0, ge=0, le=100)
    benchmark_standard: str = Field(
        default="energy_star",
        description="energy_star|cibse_tm46|internal|none",
    )


class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: SetupWizardStep = Field(...)
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
    current_step: SetupWizardStep = Field(default=SetupWizardStep.WELCOME)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    facility_setup: Optional[FacilitySetup] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class PresetConfig(BaseModel):
    """Facility type preset configuration."""

    preset_name: str = Field(default="")
    building_type: str = Field(default="")
    preset_applied: bool = Field(default=False)
    commodities: List[str] = Field(default_factory=list)
    typical_meters: int = Field(default=0)
    rate_structure: str = Field(default="")
    allocation_method: str = Field(default="")
    benchmark_standard: str = Field(default="")
    engines_enabled: List[str] = Field(default_factory=list)
    benchmark_eui_kwh_m2: float = Field(default=0.0)
    typical_cost_eur_m2: float = Field(default=0.0)


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    country: str = Field(default="")
    floor_area_m2: float = Field(default=0.0)
    commodities: List[str] = Field(default_factory=list)
    meter_count: int = Field(default=0)
    rate_structure: str = Field(default="")
    allocation_method: str = Field(default="")
    annual_budget_eur: float = Field(default=0.0)
    benchmark_standard: str = Field(default="")
    preset_applied: str = Field(default="")
    engines_enabled: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupWizardStep] = [
    SetupWizardStep.WELCOME,
    SetupWizardStep.FACILITY_PROFILE,
    SetupWizardStep.COMMODITY_ACCOUNTS,
    SetupWizardStep.METER_MAPPING,
    SetupWizardStep.RATE_SCHEDULES,
    SetupWizardStep.ALLOCATION_RULES,
    SetupWizardStep.BUDGET_PARAMETERS,
    SetupWizardStep.REVIEW_CONFIRM,
]

STEP_DISPLAY_NAMES: Dict[SetupWizardStep, str] = {
    SetupWizardStep.WELCOME: "Welcome & Introduction",
    SetupWizardStep.FACILITY_PROFILE: "Facility Profile",
    SetupWizardStep.COMMODITY_ACCOUNTS: "Commodity Accounts",
    SetupWizardStep.METER_MAPPING: "Meter Mapping",
    SetupWizardStep.RATE_SCHEDULES: "Rate Schedules",
    SetupWizardStep.ALLOCATION_RULES: "Allocation Rules",
    SetupWizardStep.BUDGET_PARAMETERS: "Budget Parameters",
    SetupWizardStep.REVIEW_CONFIRM: "Review & Confirm",
}

# ---------------------------------------------------------------------------
# Facility Presets (8 presets)
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "office_building": {
        "building_type": "office",
        "commodities": ["electricity", "natural_gas"],
        "typical_meters": 4,
        "rate_structure": "tou",
        "allocation_method": "floor_area",
        "benchmark_standard": "energy_star",
        "engines": ["bill_parser", "rate_analyzer", "demand_analysis",
                     "cost_allocation", "budget_forecast", "benchmark",
                     "reporting"],
        "benchmark_eui": 200, "typical_cost_m2": 25.0,
    },
    "manufacturing": {
        "building_type": "manufacturing",
        "commodities": ["electricity", "natural_gas", "steam"],
        "typical_meters": 12,
        "rate_structure": "demand",
        "allocation_method": "sub_metered",
        "benchmark_standard": "internal",
        "engines": ["bill_parser", "rate_analyzer", "demand_analysis",
                     "cost_allocation", "budget_forecast", "benchmark",
                     "regulatory_optimizer", "procurement", "reporting"],
        "benchmark_eui": 400, "typical_cost_m2": 45.0,
    },
    "retail_store": {
        "building_type": "retail",
        "commodities": ["electricity", "natural_gas"],
        "typical_meters": 2,
        "rate_structure": "flat",
        "allocation_method": "fixed_split",
        "benchmark_standard": "energy_star",
        "engines": ["bill_parser", "rate_analyzer", "budget_forecast",
                     "benchmark", "reporting"],
        "benchmark_eui": 300, "typical_cost_m2": 35.0,
    },
    "warehouse": {
        "building_type": "warehouse",
        "commodities": ["electricity", "natural_gas"],
        "typical_meters": 3,
        "rate_structure": "demand",
        "allocation_method": "floor_area",
        "benchmark_standard": "cibse_tm46",
        "engines": ["bill_parser", "rate_analyzer", "demand_analysis",
                     "budget_forecast", "benchmark", "reporting"],
        "benchmark_eui": 120, "typical_cost_m2": 12.0,
    },
    "healthcare": {
        "building_type": "healthcare",
        "commodities": ["electricity", "natural_gas", "steam", "chilled_water"],
        "typical_meters": 15,
        "rate_structure": "demand",
        "allocation_method": "sub_metered",
        "benchmark_standard": "energy_star",
        "engines": ["bill_parser", "rate_analyzer", "demand_analysis",
                     "cost_allocation", "budget_forecast", "benchmark",
                     "regulatory_optimizer", "reporting"],
        "benchmark_eui": 350, "typical_cost_m2": 55.0,
    },
    "education": {
        "building_type": "education",
        "commodities": ["electricity", "natural_gas"],
        "typical_meters": 6,
        "rate_structure": "tou",
        "allocation_method": "floor_area",
        "benchmark_standard": "energy_star",
        "engines": ["bill_parser", "rate_analyzer", "cost_allocation",
                     "budget_forecast", "benchmark", "reporting"],
        "benchmark_eui": 180, "typical_cost_m2": 20.0,
    },
    "data_center": {
        "building_type": "data_center",
        "commodities": ["electricity"],
        "typical_meters": 8,
        "rate_structure": "demand",
        "allocation_method": "sub_metered",
        "benchmark_standard": "internal",
        "engines": ["bill_parser", "rate_analyzer", "demand_analysis",
                     "cost_allocation", "budget_forecast", "benchmark",
                     "procurement", "regulatory_optimizer", "reporting"],
        "benchmark_eui": 2000, "typical_cost_m2": 250.0,
    },
    "multi_site_portfolio": {
        "building_type": "portfolio",
        "commodities": ["electricity", "natural_gas"],
        "typical_meters": 50,
        "rate_structure": "mixed",
        "allocation_method": "sub_metered",
        "benchmark_standard": "energy_star",
        "engines": ["bill_parser", "rate_analyzer", "demand_analysis",
                     "cost_allocation", "budget_forecast", "benchmark",
                     "procurement", "regulatory_optimizer", "reporting"],
        "benchmark_eui": 250, "typical_cost_m2": 30.0,
    },
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """8-step guided utility account configuration wizard for PACK-036.

    Guides facilities through utility analysis setup including commodity
    accounts, meter mapping, rate schedules, allocation rules, and budget
    parameters with 8 facility presets.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.advance({"step": "welcome", "accepted": True})
        >>> result = wizard.complete()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            SetupWizardStep.WELCOME: self._handle_welcome,
            SetupWizardStep.FACILITY_PROFILE: self._handle_facility_profile,
            SetupWizardStep.COMMODITY_ACCOUNTS: self._handle_commodity_accounts,
            SetupWizardStep.METER_MAPPING: self._handle_meter_mapping,
            SetupWizardStep.RATE_SCHEDULES: self._handle_rate_schedules,
            SetupWizardStep.ALLOCATION_RULES: self._handle_allocation_rules,
            SetupWizardStep.BUDGET_PARAMETERS: self._handle_budget_parameters,
            SetupWizardStep.REVIEW_CONFIRM: self._handle_review_confirm,
        }
        self.logger.info("SetupWizard initialized: 8 steps, 8 presets")

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState with all steps in PENDING status.
        """
        wizard_id = _compute_hash(
            f"utility-wizard:{_utcnow().isoformat()}"
        )[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(
                    step_name, step_name.value
                ),
            )
        self._state = WizardState(
            wizard_id=wizard_id,
            current_step=STEP_ORDER[0],
            steps=steps,
        )
        self.logger.info("Setup wizard started: %s", wizard_id)
        return self._state

    def advance(self, step_data: Dict[str, Any]) -> WizardStepState:
        """Advance the wizard by completing the current step.

        Args:
            step_data: Data for the current step.

        Returns:
            Updated WizardStepState for the completed step.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        current = self._state.current_step
        step = self._state.steps.get(current.value)
        if step is None:
            raise ValueError(f"Step '{current.value}' not found")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = _utcnow()
        start_time = time.monotonic()

        handler = self._step_handlers.get(current)
        if handler is None:
            raise ValueError(f"No handler for step '{current.value}'")

        try:
            errors = handler(step_data)
            elapsed = (time.monotonic() - start_time) * 1000
            step.execution_time_ms = elapsed
            step.data = step_data

            if errors:
                step.status = StepStatus.PENDING
                step.validation_errors = errors
            else:
                step.status = StepStatus.COMPLETED
                step.completed_at = _utcnow()
                step.validation_errors = []
                self._advance_to_next(current)
        except Exception as exc:
            step.status = StepStatus.PENDING
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

        return step

    def complete(self) -> SetupResult:
        """Complete the wizard and generate the setup result.

        Returns:
            SetupResult with final configuration.
        """
        return self._generate_result()

    def load_preset(self, name: str) -> PresetConfig:
        """Load a facility type preset.

        Args:
            name: Preset name.

        Returns:
            PresetConfig with preset configuration.

        Raises:
            ValueError: If preset name is not found.
        """
        preset_data = FACILITY_PRESETS.get(name)
        if preset_data is None:
            valid = sorted(FACILITY_PRESETS.keys())
            raise ValueError(f"Unknown preset '{name}'. Valid: {valid}")

        return PresetConfig(
            preset_name=name,
            building_type=preset_data.get("building_type", ""),
            preset_applied=True,
            commodities=preset_data.get("commodities", []),
            typical_meters=preset_data.get("typical_meters", 0),
            rate_structure=preset_data.get("rate_structure", ""),
            allocation_method=preset_data.get("allocation_method", ""),
            benchmark_standard=preset_data.get("benchmark_standard", ""),
            engines_enabled=preset_data.get("engines", []),
            benchmark_eui_kwh_m2=preset_data.get("benchmark_eui", 0.0),
            typical_cost_eur_m2=preset_data.get("typical_cost_m2", 0.0),
        )

    def get_progress(self) -> Dict[str, Any]:
        """Get the current wizard progress.

        Returns:
            Dict with progress metrics.
        """
        if self._state is None:
            return {"started": False, "progress_pct": 0.0}

        completed = sum(
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )
        total = len(STEP_ORDER)

        return {
            "started": True,
            "wizard_id": self._state.wizard_id,
            "current_step": self._state.current_step.value,
            "steps_completed": completed,
            "total_steps": total,
            "progress_pct": round(completed / total * 100.0, 1),
            "is_complete": self._state.is_complete,
        }

    # ---- Step Handlers ----

    def _handle_welcome(self, data: Dict[str, Any]) -> List[str]:
        if not data.get("accepted", False):
            return ["Terms must be accepted to proceed"]
        return []

    def _handle_facility_profile(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not data.get("facility_name"):
            errors.append("Facility name is required")
        if data.get("floor_area_m2", 0) <= 0:
            errors.append("Floor area must be greater than 0")
        if not errors and self._state:
            if self._state.facility_setup is None:
                self._state.facility_setup = FacilitySetup()
            fs = self._state.facility_setup
            fs.facility_name = data.get("facility_name", "")
            fs.country = data.get("country", "DE")
            fs.floor_area_m2 = data.get("floor_area_m2", 0.0)
            fs.building_type = data.get("building_type", "office")
            fs.employee_count = data.get("employee_count", 0)
        return errors

    def _handle_commodity_accounts(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        commodities = data.get("commodities", [])
        if not commodities:
            errors.append("At least one commodity must be selected")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.commodities = commodities
            self._state.facility_setup.electricity_accounts = data.get(
                "electricity_accounts", 1
            )
            self._state.facility_setup.gas_accounts = data.get(
                "gas_accounts", 0
            )
        return errors

    def _handle_meter_mapping(self, data: Dict[str, Any]) -> List[str]:
        if self._state and self._state.facility_setup:
            self._state.facility_setup.meter_count = data.get("meter_count", 2)
            self._state.facility_setup.sub_meter_count = data.get(
                "sub_meter_count", 0
            )
        return []

    def _handle_rate_schedules(self, data: Dict[str, Any]) -> List[str]:
        if self._state and self._state.facility_setup:
            self._state.facility_setup.rate_structure = data.get(
                "rate_structure", "tou"
            )
        return []

    def _handle_allocation_rules(self, data: Dict[str, Any]) -> List[str]:
        if self._state and self._state.facility_setup:
            self._state.facility_setup.allocation_method = data.get(
                "allocation_method", "floor_area"
            )
            self._state.facility_setup.cost_centers = data.get(
                "cost_centers", 1
            )
        return []

    def _handle_budget_parameters(self, data: Dict[str, Any]) -> List[str]:
        if self._state and self._state.facility_setup:
            self._state.facility_setup.annual_budget_eur = data.get(
                "annual_budget_eur", 0.0
            )
            self._state.facility_setup.variance_threshold_pct = data.get(
                "variance_threshold_pct", 10.0
            )
            self._state.facility_setup.benchmark_standard = data.get(
                "benchmark_standard", "energy_star"
            )
        return []

    def _handle_review_confirm(self, data: Dict[str, Any]) -> List[str]:
        if not data.get("confirmed", False):
            return ["Configuration must be confirmed to complete"]
        return []

    # ---- Navigation ----

    def _advance_to_next(self, current: SetupWizardStep) -> None:
        if self._state is None:
            return
        try:
            idx = STEP_ORDER.index(current)
            if idx < len(STEP_ORDER) - 1:
                self._state.current_step = STEP_ORDER[idx + 1]
            else:
                self._state.is_complete = True
                self._state.completed_at = _utcnow()
        except ValueError:
            pass

    # ---- Result Generation ----

    def _generate_result(self) -> SetupResult:
        if self._state is None:
            return SetupResult()

        fs = self._state.facility_setup or FacilitySetup()

        completed_count = sum(
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )

        preset_data = FACILITY_PRESETS.get(fs.building_type, {})
        engines = preset_data.get("engines", [])

        config_hash = _compute_hash({
            "facility": fs.facility_name,
            "type": fs.building_type,
            "commodities": fs.commodities,
            "meters": fs.meter_count,
        })

        result = SetupResult(
            facility_name=fs.facility_name,
            building_type=fs.building_type,
            country=fs.country,
            floor_area_m2=fs.floor_area_m2,
            commodities=fs.commodities,
            meter_count=fs.meter_count,
            rate_structure=fs.rate_structure,
            allocation_method=fs.allocation_method,
            annual_budget_eur=fs.annual_budget_eur,
            benchmark_standard=fs.benchmark_standard,
            preset_applied=fs.building_type,
            engines_enabled=engines,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
