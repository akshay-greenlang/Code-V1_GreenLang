# -*- coding: utf-8 -*-
"""
SetupWizard - 9-Step Demand Response Configuration Wizard for PACK-037
========================================================================

This module implements a 9-step configuration wizard for facilities setting
up the Demand Response Pack.

Wizard Steps (9):
    1. FACILITY_PROFILE       -- Facility name, type, size, location
    2. LOAD_INVENTORY         -- Controllable loads, equipment categories
    3. UTILITY_ACCOUNTS       -- Utility accounts, meter IDs, rate schedules
    4. DR_PROGRAM_PREFERENCES -- Program types, commitment levels, availability
    5. GRID_REGION            -- ISO/RTO region, zone, node
    6. BASELINE_DATA          -- Historical load data, baseline method
    7. DER_ASSETS             -- Battery, PV, EV, generator registration
    8. BMS_CONNECTIVITY       -- BMS protocol, endpoints, control points
    9. REPORTING              -- Report preferences, notification settings

8 Facility Presets with auto-configured DR parameters.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-037 Demand Response
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

class WizardStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    FACILITY_PROFILE = "facility_profile"
    LOAD_INVENTORY = "load_inventory"
    UTILITY_ACCOUNTS = "utility_accounts"
    DR_PROGRAM_PREFERENCES = "dr_program_preferences"
    GRID_REGION = "grid_region"
    BASELINE_DATA = "baseline_data"
    DER_ASSETS = "der_assets"
    BMS_CONNECTIVITY = "bms_connectivity"
    REPORTING = "reporting"

class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class FacilityDRProfile(BaseModel):
    """Facility profile data for DR configuration."""

    facility_name: str = Field(default="", min_length=0, max_length=255)
    facility_id: str = Field(default="")
    facility_type: str = Field(default="commercial_office")
    country: str = Field(default="US")
    state_province: str = Field(default="")
    address: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0)
    peak_demand_kw: float = Field(default=0.0, ge=0)
    annual_energy_kwh: float = Field(default=0.0, ge=0)
    controllable_load_kw: float = Field(default=0.0, ge=0)
    grid_region: str = Field(default="PJM")
    grid_zone: str = Field(default="")
    utility_name: str = Field(default="")
    meter_id: str = Field(default="")
    rate_schedule: str = Field(default="")
    dr_program_type: str = Field(default="capacity", description="capacity|energy|ancillary|emergency")
    commitment_kw: float = Field(default=0.0, ge=0)
    notification_preference_minutes: int = Field(default=30, ge=0)
    max_event_hours: int = Field(default=4, ge=1, le=12)
    max_events_per_year: int = Field(default=20, ge=0)
    baseline_method: str = Field(default="10_of_10")
    bms_protocol: str = Field(default="bacnet_ip")
    bms_host: str = Field(default="")
    der_battery_kw: float = Field(default=0.0, ge=0)
    der_solar_kw: float = Field(default=0.0, ge=0)
    der_ev_charger_kw: float = Field(default=0.0, ge=0)
    reporting_frequency: str = Field(default="monthly", description="weekly|monthly|quarterly")

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: WizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)

class WizardConfig(BaseModel):
    """Complete state of the setup wizard."""

    wizard_id: str = Field(default="")
    current_step: WizardStep = Field(default=WizardStep.FACILITY_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    facility_profile: Optional[FacilityDRProfile] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class PresetConfig(BaseModel):
    """Facility type preset for DR configuration."""

    preset_name: str = Field(default="")
    facility_type: str = Field(default="")
    preset_applied: bool = Field(default=False)
    typical_peak_kw: float = Field(default=0.0)
    typical_curtailable_pct: float = Field(default=0.0)
    recommended_program: str = Field(default="")
    controllable_categories: List[str] = Field(default_factory=list)
    typical_der_types: List[str] = Field(default_factory=list)
    baseline_method: str = Field(default="10_of_10")
    max_event_hours: int = Field(default=4)

class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    facility_type: str = Field(default="")
    grid_region: str = Field(default="")
    peak_demand_kw: float = Field(default=0.0)
    controllable_load_kw: float = Field(default=0.0)
    dr_program_type: str = Field(default="")
    commitment_kw: float = Field(default=0.0)
    baseline_method: str = Field(default="")
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=9)
    engines_enabled: List[str] = Field(default_factory=list)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[WizardStep] = [
    WizardStep.FACILITY_PROFILE,
    WizardStep.LOAD_INVENTORY,
    WizardStep.UTILITY_ACCOUNTS,
    WizardStep.DR_PROGRAM_PREFERENCES,
    WizardStep.GRID_REGION,
    WizardStep.BASELINE_DATA,
    WizardStep.DER_ASSETS,
    WizardStep.BMS_CONNECTIVITY,
    WizardStep.REPORTING,
]

STEP_DISPLAY_NAMES: Dict[WizardStep, str] = {
    WizardStep.FACILITY_PROFILE: "Facility Profile",
    WizardStep.LOAD_INVENTORY: "Load Inventory",
    WizardStep.UTILITY_ACCOUNTS: "Utility Accounts",
    WizardStep.DR_PROGRAM_PREFERENCES: "DR Program Preferences",
    WizardStep.GRID_REGION: "Grid Region Selection",
    WizardStep.BASELINE_DATA: "Baseline Data Configuration",
    WizardStep.DER_ASSETS: "DER Asset Registration",
    WizardStep.BMS_CONNECTIVITY: "BMS Connectivity Setup",
    WizardStep.REPORTING: "Reporting Preferences",
}

# ---------------------------------------------------------------------------
# Facility Presets (8 presets)
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "commercial_office": {
        "facility_type": "commercial_office",
        "typical_peak_kw": 1500,
        "typical_curtailable_pct": 25,
        "recommended_program": "capacity",
        "controllable": ["hvac", "lighting", "plug_loads", "ev_charging"],
        "der_types": ["battery", "solar_pv"],
        "baseline": "10_of_10",
        "max_event_hours": 4,
    },
    "manufacturing": {
        "facility_type": "manufacturing",
        "typical_peak_kw": 5000,
        "typical_curtailable_pct": 15,
        "recommended_program": "capacity",
        "controllable": ["hvac", "lighting", "process_scheduling", "compressed_air"],
        "der_types": ["backup_generator", "battery"],
        "baseline": "high_5_of_10",
        "max_event_hours": 2,
    },
    "retail_store": {
        "facility_type": "retail_store",
        "typical_peak_kw": 800,
        "typical_curtailable_pct": 20,
        "recommended_program": "energy",
        "controllable": ["hvac", "lighting", "signage"],
        "der_types": ["solar_pv"],
        "baseline": "10_of_10",
        "max_event_hours": 4,
    },
    "warehouse": {
        "facility_type": "warehouse",
        "typical_peak_kw": 600,
        "typical_curtailable_pct": 30,
        "recommended_program": "capacity",
        "controllable": ["hvac", "lighting", "ev_charging", "forklift_charging"],
        "der_types": ["solar_pv", "battery"],
        "baseline": "10_of_10",
        "max_event_hours": 6,
    },
    "healthcare": {
        "facility_type": "healthcare",
        "typical_peak_kw": 3000,
        "typical_curtailable_pct": 10,
        "recommended_program": "emergency",
        "controllable": ["hvac_non_critical", "lighting_common", "ev_charging"],
        "der_types": ["backup_generator", "battery"],
        "baseline": "weather_adjusted",
        "max_event_hours": 2,
    },
    "education": {
        "facility_type": "education",
        "typical_peak_kw": 1200,
        "typical_curtailable_pct": 25,
        "recommended_program": "capacity",
        "controllable": ["hvac", "lighting", "plug_loads"],
        "der_types": ["solar_pv"],
        "baseline": "10_of_10",
        "max_event_hours": 4,
    },
    "data_center": {
        "facility_type": "data_center",
        "typical_peak_kw": 10000,
        "typical_curtailable_pct": 5,
        "recommended_program": "ancillary",
        "controllable": ["cooling_optimization", "ups_load_shedding"],
        "der_types": ["backup_generator", "battery", "fuel_cell"],
        "baseline": "regression",
        "max_event_hours": 1,
    },
    "campus": {
        "facility_type": "campus",
        "typical_peak_kw": 8000,
        "typical_curtailable_pct": 20,
        "recommended_program": "capacity",
        "controllable": ["hvac", "lighting", "ev_charging", "thermal_storage"],
        "der_types": ["solar_pv", "battery", "thermal_storage"],
        "baseline": "weather_adjusted",
        "max_event_hours": 4,
    },
}

# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------

class SetupWizard:
    """9-step guided DR configuration wizard for PACK-037.

    Guides facilities through demand response setup with facility presets
    that auto-configure DR program parameters, DER assets, and BMS control.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.advance({"facility_name": "Campus A", "peak_demand_kw": 2500})
        >>> result = wizard.complete()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardConfig] = None
        self._step_handlers = {
            WizardStep.FACILITY_PROFILE: self._handle_facility_profile,
            WizardStep.LOAD_INVENTORY: self._handle_load_inventory,
            WizardStep.UTILITY_ACCOUNTS: self._handle_utility_accounts,
            WizardStep.DR_PROGRAM_PREFERENCES: self._handle_dr_preferences,
            WizardStep.GRID_REGION: self._handle_grid_region,
            WizardStep.BASELINE_DATA: self._handle_baseline_data,
            WizardStep.DER_ASSETS: self._handle_der_assets,
            WizardStep.BMS_CONNECTIVITY: self._handle_bms_connectivity,
            WizardStep.REPORTING: self._handle_reporting,
        }
        self.logger.info("SetupWizard initialized: 9 steps, 8 presets")

    def start(self) -> WizardConfig:
        """Start a new wizard session.

        Returns:
            Initial WizardConfig with all steps in PENDING status.
        """
        wizard_id = _compute_hash(f"dr-wizard:{utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardConfig(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
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
            name: Preset name (e.g., 'commercial_office', 'manufacturing').

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
            facility_type=preset_data.get("facility_type", ""),
            preset_applied=True,
            typical_peak_kw=preset_data.get("typical_peak_kw", 0.0),
            typical_curtailable_pct=preset_data.get("typical_curtailable_pct", 0.0),
            recommended_program=preset_data.get("recommended_program", "capacity"),
            controllable_categories=preset_data.get("controllable", []),
            typical_der_types=preset_data.get("der_types", []),
            baseline_method=preset_data.get("baseline", "10_of_10"),
            max_event_hours=preset_data.get("max_event_hours", 4),
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

    def _handle_facility_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle facility profile step."""
        errors: List[str] = []
        if not data.get("facility_name"):
            errors.append("Facility name is required")
        if data.get("peak_demand_kw", 0) <= 0:
            errors.append("Peak demand must be greater than 0")
        if not errors and self._state:
            if self._state.facility_profile is None:
                self._state.facility_profile = FacilityDRProfile()
            self._state.facility_profile.facility_name = data.get("facility_name", "")
            self._state.facility_profile.facility_type = data.get("facility_type", "commercial_office")
            self._state.facility_profile.peak_demand_kw = data.get("peak_demand_kw", 0.0)
            self._state.facility_profile.floor_area_m2 = data.get("floor_area_m2", 0.0)
        return errors

    def _handle_load_inventory(self, data: Dict[str, Any]) -> List[str]:
        """Handle load inventory step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.controllable_load_kw = data.get("controllable_load_kw", 0.0)
        return []

    def _handle_utility_accounts(self, data: Dict[str, Any]) -> List[str]:
        """Handle utility accounts step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.utility_name = data.get("utility_name", "")
            self._state.facility_profile.meter_id = data.get("meter_id", "")
            self._state.facility_profile.rate_schedule = data.get("rate_schedule", "")
        return []

    def _handle_dr_preferences(self, data: Dict[str, Any]) -> List[str]:
        """Handle DR program preferences step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.dr_program_type = data.get("dr_program_type", "capacity")
            self._state.facility_profile.commitment_kw = data.get("commitment_kw", 0.0)
            self._state.facility_profile.max_event_hours = data.get("max_event_hours", 4)
            self._state.facility_profile.max_events_per_year = data.get("max_events_per_year", 20)
        return []

    def _handle_grid_region(self, data: Dict[str, Any]) -> List[str]:
        """Handle grid region step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.grid_region = data.get("grid_region", "PJM")
            self._state.facility_profile.grid_zone = data.get("grid_zone", "")
        return []

    def _handle_baseline_data(self, data: Dict[str, Any]) -> List[str]:
        """Handle baseline data step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.baseline_method = data.get("baseline_method", "10_of_10")
        return []

    def _handle_der_assets(self, data: Dict[str, Any]) -> List[str]:
        """Handle DER assets step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.der_battery_kw = data.get("battery_kw", 0.0)
            self._state.facility_profile.der_solar_kw = data.get("solar_kw", 0.0)
            self._state.facility_profile.der_ev_charger_kw = data.get("ev_charger_kw", 0.0)
        return []

    def _handle_bms_connectivity(self, data: Dict[str, Any]) -> List[str]:
        """Handle BMS connectivity step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.bms_protocol = data.get("bms_protocol", "bacnet_ip")
            self._state.facility_profile.bms_host = data.get("bms_host", "")
        return []

    def _handle_reporting(self, data: Dict[str, Any]) -> List[str]:
        """Handle reporting preferences step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.reporting_frequency = data.get("reporting_frequency", "monthly")
        return []

    # ---- Navigation ----

    def _advance_to_next(self, current: WizardStep) -> None:
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

        fp = self._state.facility_profile or FacilityDRProfile()

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        preset_data = FACILITY_PRESETS.get(fp.facility_type, {})
        engines = [
            "dr_flexibility_engine", "baseline_calculation_engine",
            "dispatch_optimization_engine", "der_coordination_engine",
            "event_management_engine", "performance_tracking_engine",
            "revenue_reconciliation_engine", "program_matching_engine",
            "load_inventory_engine", "reporting_engine",
        ]

        config_hash = _compute_hash({
            "facility": fp.facility_name,
            "type": fp.facility_type,
            "region": fp.grid_region,
            "program": fp.dr_program_type,
        })

        result = SetupResult(
            facility_name=fp.facility_name,
            facility_type=fp.facility_type,
            grid_region=fp.grid_region,
            peak_demand_kw=fp.peak_demand_kw,
            controllable_load_kw=fp.controllable_load_kw,
            dr_program_type=fp.dr_program_type,
            commitment_kw=fp.commitment_kw,
            baseline_method=fp.baseline_method,
            total_steps_completed=completed_count,
            engines_enabled=engines,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
