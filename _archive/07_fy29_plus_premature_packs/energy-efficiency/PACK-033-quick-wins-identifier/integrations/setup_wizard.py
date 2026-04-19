# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Facility Configuration for PACK-033
=================================================================

This module implements an 8-step configuration wizard for facilities setting
up the Quick Wins Identifier Pack.

Wizard Steps (8):
    1. WELCOME            -- Introduction, terms, scope overview
    2. FACILITY_PROFILE   -- Name, location, size, occupancy
    3. BUILDING_TYPE      -- Building/facility type and use pattern
    4. EQUIPMENT_SURVEY   -- Major equipment categories and counts
    5. ENERGY_DATA        -- Energy consumption and cost data
    6. UTILITY_INFO       -- Utility provider, rate structure
    7. GOALS_PRIORITIES   -- Savings targets, payback constraints
    8. REVIEW_CONFIRM     -- Review all inputs and confirm

8 Facility Presets with auto-configured scanning focus areas.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-033 Quick Wins Identifier
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

class SetupWizardStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    WELCOME = "welcome"
    FACILITY_PROFILE = "facility_profile"
    BUILDING_TYPE = "building_type"
    EQUIPMENT_SURVEY = "equipment_survey"
    ENERGY_DATA = "energy_data"
    UTILITY_INFO = "utility_info"
    GOALS_PRIORITIES = "goals_priorities"
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

    facility_name: str = Field(default="", min_length=0, max_length=255)
    facility_id: str = Field(default="")
    country: str = Field(default="DE")
    region: str = Field(default="")
    address: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0)
    employee_count: int = Field(default=0, ge=0)
    operating_hours_per_year: float = Field(default=2000.0, ge=0)
    building_type: str = Field(default="office", description="office|manufacturing|retail|warehouse|healthcare|education|data_center|sme")
    year_built: int = Field(default=2000, ge=1900, le=2030)
    annual_energy_kwh: float = Field(default=0.0, ge=0)
    annual_energy_cost_eur: float = Field(default=0.0, ge=0)
    electricity_pct: float = Field(default=70.0, ge=0, le=100)
    gas_pct: float = Field(default=30.0, ge=0, le=100)
    utility_provider: str = Field(default="")
    rate_structure: str = Field(default="flat", description="flat|tou|demand|blended")
    savings_target_pct: float = Field(default=10.0, ge=0, le=100)
    max_payback_months: int = Field(default=24, ge=1, le=120)
    max_investment_eur: float = Field(default=50_000.0, ge=0)
    priority_areas: List[str] = Field(default_factory=lambda: ["lighting", "hvac", "controls"])

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
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class PresetConfig(BaseModel):
    """Facility type preset configuration."""

    preset_name: str = Field(default="")
    building_type: str = Field(default="")
    preset_applied: bool = Field(default=False)
    scan_focus_areas: List[str] = Field(default_factory=list)
    engines_enabled: List[str] = Field(default_factory=list)
    typical_quick_wins: List[str] = Field(default_factory=list)
    benchmark_kwh_per_m2: float = Field(default=0.0)
    typical_savings_pct: float = Field(default=0.0)

class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    country: str = Field(default="")
    floor_area_m2: float = Field(default=0.0)
    annual_energy_kwh: float = Field(default=0.0)
    savings_target_pct: float = Field(default=0.0)
    max_payback_months: int = Field(default=24)
    preset_applied: str = Field(default="")
    engines_enabled: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupWizardStep] = [
    SetupWizardStep.WELCOME,
    SetupWizardStep.FACILITY_PROFILE,
    SetupWizardStep.BUILDING_TYPE,
    SetupWizardStep.EQUIPMENT_SURVEY,
    SetupWizardStep.ENERGY_DATA,
    SetupWizardStep.UTILITY_INFO,
    SetupWizardStep.GOALS_PRIORITIES,
    SetupWizardStep.REVIEW_CONFIRM,
]

STEP_DISPLAY_NAMES: Dict[SetupWizardStep, str] = {
    SetupWizardStep.WELCOME: "Welcome & Introduction",
    SetupWizardStep.FACILITY_PROFILE: "Facility Profile",
    SetupWizardStep.BUILDING_TYPE: "Building Type Selection",
    SetupWizardStep.EQUIPMENT_SURVEY: "Equipment Survey",
    SetupWizardStep.ENERGY_DATA: "Energy Data Input",
    SetupWizardStep.UTILITY_INFO: "Utility Information",
    SetupWizardStep.GOALS_PRIORITIES: "Goals & Priorities",
    SetupWizardStep.REVIEW_CONFIRM: "Review & Confirm",
}

# ---------------------------------------------------------------------------
# Facility Presets (8 presets)
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "office_building": {
        "building_type": "office",
        "scan_focus": ["lighting", "hvac", "controls", "plug_loads", "envelope"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "reporting"],
        "typical_wins": ["LED_retrofit", "occupancy_sensors", "setpoint_optimization", "plug_load_timers"],
        "benchmark_kwh_m2": 200,
        "typical_savings_pct": 15,
    },
    "manufacturing": {
        "building_type": "manufacturing",
        "scan_focus": ["compressed_air", "motors_drives", "lighting", "hvac", "process_heat"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "rebate", "reporting"],
        "typical_wins": ["compressed_air_leaks", "vfd_retrofit", "LED_retrofit", "insulation"],
        "benchmark_kwh_m2": 400,
        "typical_savings_pct": 12,
    },
    "retail_store": {
        "building_type": "retail",
        "scan_focus": ["lighting", "hvac", "refrigeration", "controls", "signage"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "reporting"],
        "typical_wins": ["LED_retrofit", "refrigeration_maintenance", "door_closers", "scheduling"],
        "benchmark_kwh_m2": 300,
        "typical_savings_pct": 18,
    },
    "warehouse": {
        "building_type": "warehouse",
        "scan_focus": ["lighting", "hvac", "envelope", "dock_doors", "forklift_charging"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "reporting"],
        "typical_wins": ["high_bay_LED", "destratification_fans", "dock_seals", "motion_sensors"],
        "benchmark_kwh_m2": 120,
        "typical_savings_pct": 20,
    },
    "healthcare": {
        "building_type": "healthcare",
        "scan_focus": ["hvac", "lighting", "controls", "steam", "medical_equipment"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "reporting"],
        "typical_wins": ["LED_retrofit", "vav_optimization", "steam_trap_repair", "scheduling"],
        "benchmark_kwh_m2": 350,
        "typical_savings_pct": 10,
    },
    "education": {
        "building_type": "education",
        "scan_focus": ["lighting", "hvac", "controls", "envelope", "it_equipment"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "reporting"],
        "typical_wins": ["LED_retrofit", "occupancy_sensors", "setback_scheduling", "envelope_sealing"],
        "benchmark_kwh_m2": 180,
        "typical_savings_pct": 15,
    },
    "data_center": {
        "building_type": "data_center",
        "scan_focus": ["cooling", "ups", "airflow", "lighting", "containment"],
        "engines": ["scanner", "financial", "carbon", "prioritization", "reporting"],
        "typical_wins": ["hot_cold_containment", "raise_setpoint", "blanking_panels", "LED_retrofit"],
        "benchmark_kwh_m2": 2000,
        "typical_savings_pct": 8,
    },
    "sme_simplified": {
        "building_type": "sme",
        "scan_focus": ["lighting", "hvac", "plug_loads", "controls"],
        "engines": ["scanner", "financial", "prioritization", "reporting"],
        "typical_wins": ["LED_retrofit", "smart_thermostat", "plug_load_strips", "timer_controls"],
        "benchmark_kwh_m2": 220,
        "typical_savings_pct": 20,
    },
}

# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------

class SetupWizard:
    """8-step guided facility configuration wizard for PACK-033.

    Guides facilities through quick win identification setup with facility
    presets that auto-configure scanning focus areas and engines.

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
            SetupWizardStep.BUILDING_TYPE: self._handle_building_type,
            SetupWizardStep.EQUIPMENT_SURVEY: self._handle_equipment_survey,
            SetupWizardStep.ENERGY_DATA: self._handle_energy_data,
            SetupWizardStep.UTILITY_INFO: self._handle_utility_info,
            SetupWizardStep.GOALS_PRIORITIES: self._handle_goals_priorities,
            SetupWizardStep.REVIEW_CONFIRM: self._handle_review_confirm,
        }
        self.logger.info("SetupWizard initialized: 8 steps, 8 presets")

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState with all steps in PENDING status.
        """
        wizard_id = _compute_hash(f"quick-wins-wizard:{utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
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
        step.started_at = utcnow()
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
                step.completed_at = utcnow()
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
            name: Preset name (e.g., 'office_building', 'manufacturing').

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
            scan_focus_areas=preset_data.get("scan_focus", []),
            engines_enabled=preset_data.get("engines", []),
            typical_quick_wins=preset_data.get("typical_wins", []),
            benchmark_kwh_per_m2=preset_data.get("benchmark_kwh_m2", 0.0),
            typical_savings_pct=preset_data.get("typical_savings_pct", 0.0),
        )

    def get_progress(self) -> Dict[str, Any]:
        """Get the current wizard progress.

        Returns:
            Dict with progress metrics.
        """
        if self._state is None:
            return {"started": False, "progress_pct": 0.0}

        completed = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
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
        """Handle welcome step."""
        if not data.get("accepted", False):
            return ["Terms must be accepted to proceed"]
        return []

    def _handle_facility_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle facility profile step."""
        errors: List[str] = []
        if not data.get("facility_name"):
            errors.append("Facility name is required")
        if data.get("floor_area_m2", 0) <= 0:
            errors.append("Floor area must be greater than 0")
        if not errors and self._state:
            if self._state.facility_setup is None:
                self._state.facility_setup = FacilitySetup()
            self._state.facility_setup.facility_name = data.get("facility_name", "")
            self._state.facility_setup.country = data.get("country", "DE")
            self._state.facility_setup.floor_area_m2 = data.get("floor_area_m2", 0.0)
            self._state.facility_setup.employee_count = data.get("employee_count", 0)
        return errors

    def _handle_building_type(self, data: Dict[str, Any]) -> List[str]:
        """Handle building type step."""
        errors: List[str] = []
        btype = data.get("building_type", "")
        valid_types = [p["building_type"] for p in FACILITY_PRESETS.values()]
        if btype and btype not in valid_types:
            errors.append(f"Unknown building type '{btype}'. Valid: {valid_types}")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.building_type = btype
        return errors

    def _handle_equipment_survey(self, data: Dict[str, Any]) -> List[str]:
        """Handle equipment survey step."""
        return []  # Optional step, always valid

    def _handle_energy_data(self, data: Dict[str, Any]) -> List[str]:
        """Handle energy data step."""
        errors: List[str] = []
        if data.get("annual_energy_kwh", 0) <= 0:
            errors.append("Annual energy consumption must be greater than 0")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.annual_energy_kwh = data.get("annual_energy_kwh", 0.0)
            self._state.facility_setup.annual_energy_cost_eur = data.get("annual_energy_cost_eur", 0.0)
            self._state.facility_setup.electricity_pct = data.get("electricity_pct", 70.0)
            self._state.facility_setup.gas_pct = data.get("gas_pct", 30.0)
        return errors

    def _handle_utility_info(self, data: Dict[str, Any]) -> List[str]:
        """Handle utility info step."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.utility_provider = data.get("utility_provider", "")
            self._state.facility_setup.rate_structure = data.get("rate_structure", "flat")
        return []

    def _handle_goals_priorities(self, data: Dict[str, Any]) -> List[str]:
        """Handle goals and priorities step."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.savings_target_pct = data.get("savings_target_pct", 10.0)
            self._state.facility_setup.max_payback_months = data.get("max_payback_months", 24)
            self._state.facility_setup.max_investment_eur = data.get("max_investment_eur", 50000.0)
            self._state.facility_setup.priority_areas = data.get("priority_areas", ["lighting", "hvac", "controls"])
        return []

    def _handle_review_confirm(self, data: Dict[str, Any]) -> List[str]:
        """Handle review and confirm step."""
        if not data.get("confirmed", False):
            return ["Configuration must be confirmed to complete"]
        return []

    # ---- Navigation ----

    def _advance_to_next(self, current: SetupWizardStep) -> None:
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
        """Generate the final setup result."""
        if self._state is None:
            return SetupResult()

        fs = self._state.facility_setup or FacilitySetup()

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        # Apply preset for engines
        preset_data = FACILITY_PRESETS.get(fs.building_type, {})
        engines = preset_data.get("engines", [])

        config_hash = _compute_hash({
            "facility": fs.facility_name,
            "type": fs.building_type,
            "energy": fs.annual_energy_kwh,
        })

        result = SetupResult(
            facility_name=fs.facility_name,
            building_type=fs.building_type,
            country=fs.country,
            floor_area_m2=fs.floor_area_m2,
            annual_energy_kwh=fs.annual_energy_kwh,
            savings_target_pct=fs.savings_target_pct,
            max_payback_months=fs.max_payback_months,
            preset_applied=fs.building_type,
            engines_enabled=engines,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
