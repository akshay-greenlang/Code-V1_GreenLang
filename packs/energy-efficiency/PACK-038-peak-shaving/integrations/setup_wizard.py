# -*- coding: utf-8 -*-
"""
SetupWizard - 9-Step Peak Shaving Configuration Wizard for PACK-038
======================================================================

This module implements a 9-step configuration wizard for facilities setting
up the Peak Shaving Pack. It guides users through facility profiling,
interval data import, tariff configuration, peak targets, BESS parameters,
load shift preferences, coincident peak settings, alert preferences, and
final review.

Wizard Steps (9):
    1. FACILITY_PROFILE       -- Facility name, type, size, location
    2. INTERVAL_DATA_IMPORT   -- Meter data source, interval length, date range
    3. TARIFF_CONFIGURATION   -- Utility tariff, demand charges, ratchet clause
    4. PEAK_TARGETS           -- Peak reduction targets, demand limit goals
    5. BESS_PARAMETERS        -- Battery size, chemistry, SOC limits, C-rate
    6. LOAD_SHIFT_PREFERENCES -- Controllable loads, comfort boundaries
    7. CP_SETTINGS            -- Coincident peak program, CP prediction
    8. ALERT_PREFERENCES      -- Alert channels, thresholds, escalation
    9. REVIEW_CONFIRM         -- Review configuration and confirm

8 Facility Presets with auto-configured peak shaving parameters.

Zero-Hallucination:
    All preset values and default configurations use deterministic
    lookup tables. No LLM calls in the configuration path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-038 Peak Shaving
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
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WizardStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    FACILITY_PROFILE = "facility_profile"
    INTERVAL_DATA_IMPORT = "interval_data_import"
    TARIFF_CONFIGURATION = "tariff_configuration"
    PEAK_TARGETS = "peak_targets"
    BESS_PARAMETERS = "bess_parameters"
    LOAD_SHIFT_PREFERENCES = "load_shift_preferences"
    CP_SETTINGS = "cp_settings"
    ALERT_PREFERENCES = "alert_preferences"
    REVIEW_CONFIRM = "review_confirm"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class BESSChemistryPreset(str, Enum):
    """Battery chemistry preset options."""

    LFP = "lfp"
    NMC = "nmc"
    FLOW = "flow"
    LEAD_ACID = "lead_acid"


class PeakShavingMode(str, Enum):
    """Peak shaving operating mode presets."""

    DEMAND_LIMITING = "demand_limiting"
    BESS_ONLY = "bess_only"
    LOAD_SHIFT_ONLY = "load_shift_only"
    COMBINED = "combined"


class CPProgramType(str, Enum):
    """Coincident peak program types."""

    PJM_5CP = "pjm_5cp"
    ERCOT_4CP = "ercot_4cp"
    ISO_NE_ICAP = "iso_ne_icap"
    NYISO_ICAP = "nyiso_icap"
    NONE = "none"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class FacilityPSProfile(BaseModel):
    """Facility profile data for peak shaving configuration."""

    facility_name: str = Field(default="", min_length=0, max_length=255)
    facility_id: str = Field(default="")
    facility_type: str = Field(default="commercial_office")
    country: str = Field(default="US")
    state_province: str = Field(default="")
    address: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0)
    peak_demand_kw: float = Field(default=0.0, ge=0)
    annual_energy_kwh: float = Field(default=0.0, ge=0)
    utility_name: str = Field(default="")
    meter_id: str = Field(default="")
    rate_schedule: str = Field(default="")
    demand_charge_usd_per_kw: float = Field(default=0.0, ge=0)
    ratchet_pct: float = Field(default=0.0, ge=0, le=100)
    peak_shaving_target_kw: float = Field(default=0.0, ge=0)
    peak_shaving_mode: str = Field(default="combined")
    bess_power_kw: float = Field(default=0.0, ge=0)
    bess_capacity_kwh: float = Field(default=0.0, ge=0)
    bess_chemistry: str = Field(default="lfp")
    bess_min_soc_pct: float = Field(default=10.0, ge=0, le=100)
    bess_max_soc_pct: float = Field(default=95.0, ge=0, le=100)
    controllable_load_kw: float = Field(default=0.0, ge=0)
    cp_program: str = Field(default="none")
    alert_email: str = Field(default="")
    reporting_frequency: str = Field(default="monthly")


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
    facility_profile: Optional[FacilityPSProfile] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class PresetConfig(BaseModel):
    """Facility type preset for peak shaving configuration."""

    preset_name: str = Field(default="")
    facility_type: str = Field(default="")
    preset_applied: bool = Field(default=False)
    typical_peak_kw: float = Field(default=0.0)
    typical_shaveable_pct: float = Field(default=0.0)
    recommended_bess_kw: float = Field(default=0.0)
    recommended_bess_kwh: float = Field(default=0.0)
    recommended_chemistry: str = Field(default="lfp")
    controllable_categories: List[str] = Field(default_factory=list)
    typical_demand_charge: float = Field(default=0.0)
    recommended_mode: str = Field(default="combined")


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    facility_type: str = Field(default="")
    peak_demand_kw: float = Field(default=0.0)
    peak_shaving_target_kw: float = Field(default=0.0)
    bess_power_kw: float = Field(default=0.0)
    bess_capacity_kwh: float = Field(default=0.0)
    controllable_load_kw: float = Field(default=0.0)
    demand_charge_usd_per_kw: float = Field(default=0.0)
    peak_shaving_mode: str = Field(default="")
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=9)
    engines_enabled: List[str] = Field(default_factory=list)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[WizardStep] = [
    WizardStep.FACILITY_PROFILE,
    WizardStep.INTERVAL_DATA_IMPORT,
    WizardStep.TARIFF_CONFIGURATION,
    WizardStep.PEAK_TARGETS,
    WizardStep.BESS_PARAMETERS,
    WizardStep.LOAD_SHIFT_PREFERENCES,
    WizardStep.CP_SETTINGS,
    WizardStep.ALERT_PREFERENCES,
    WizardStep.REVIEW_CONFIRM,
]

STEP_DISPLAY_NAMES: Dict[WizardStep, str] = {
    WizardStep.FACILITY_PROFILE: "Facility Profile",
    WizardStep.INTERVAL_DATA_IMPORT: "Interval Data Import",
    WizardStep.TARIFF_CONFIGURATION: "Tariff Configuration",
    WizardStep.PEAK_TARGETS: "Peak Reduction Targets",
    WizardStep.BESS_PARAMETERS: "BESS Parameters",
    WizardStep.LOAD_SHIFT_PREFERENCES: "Load Shift Preferences",
    WizardStep.CP_SETTINGS: "Coincident Peak Settings",
    WizardStep.ALERT_PREFERENCES: "Alert Preferences",
    WizardStep.REVIEW_CONFIRM: "Review & Confirm",
}

# ---------------------------------------------------------------------------
# Facility Presets (8 presets)
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "commercial_office": {
        "facility_type": "commercial_office",
        "typical_peak_kw": 1500,
        "typical_shaveable_pct": 20,
        "bess_kw": 300, "bess_kwh": 1200,
        "chemistry": "lfp",
        "controllable": ["hvac", "lighting", "ev_charging"],
        "demand_charge": 15.0, "mode": "combined",
    },
    "manufacturing": {
        "facility_type": "manufacturing",
        "typical_peak_kw": 5000,
        "typical_shaveable_pct": 15,
        "bess_kw": 750, "bess_kwh": 3000,
        "chemistry": "lfp",
        "controllable": ["hvac", "compressed_air", "process_scheduling"],
        "demand_charge": 20.0, "mode": "combined",
    },
    "retail_store": {
        "facility_type": "retail_store",
        "typical_peak_kw": 800,
        "typical_shaveable_pct": 18,
        "bess_kw": 150, "bess_kwh": 600,
        "chemistry": "lfp",
        "controllable": ["hvac", "lighting", "refrigeration"],
        "demand_charge": 12.0, "mode": "bess_only",
    },
    "warehouse": {
        "facility_type": "warehouse",
        "typical_peak_kw": 600,
        "typical_shaveable_pct": 25,
        "bess_kw": 150, "bess_kwh": 600,
        "chemistry": "lfp",
        "controllable": ["hvac", "lighting", "ev_charging", "forklift_charging"],
        "demand_charge": 10.0, "mode": "combined",
    },
    "healthcare": {
        "facility_type": "healthcare",
        "typical_peak_kw": 3000,
        "typical_shaveable_pct": 8,
        "bess_kw": 250, "bess_kwh": 1000,
        "chemistry": "lfp",
        "controllable": ["hvac_non_critical", "lighting_common"],
        "demand_charge": 18.0, "mode": "bess_only",
    },
    "education": {
        "facility_type": "education",
        "typical_peak_kw": 1200,
        "typical_shaveable_pct": 22,
        "bess_kw": 250, "bess_kwh": 1000,
        "chemistry": "lfp",
        "controllable": ["hvac", "lighting"],
        "demand_charge": 14.0, "mode": "combined",
    },
    "data_center": {
        "facility_type": "data_center",
        "typical_peak_kw": 10000,
        "typical_shaveable_pct": 5,
        "bess_kw": 500, "bess_kwh": 2000,
        "chemistry": "lfp",
        "controllable": ["cooling_optimization"],
        "demand_charge": 22.0, "mode": "bess_only",
    },
    "campus": {
        "facility_type": "campus",
        "typical_peak_kw": 8000,
        "typical_shaveable_pct": 18,
        "bess_kw": 1500, "bess_kwh": 6000,
        "chemistry": "lfp",
        "controllable": ["hvac", "lighting", "ev_charging", "thermal_storage"],
        "demand_charge": 16.0, "mode": "combined",
    },
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """9-step guided peak shaving configuration wizard for PACK-038.

    Guides facilities through peak shaving setup with facility presets
    that auto-configure BESS sizing, demand targets, and load control.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.advance({"facility_name": "Plant A", "peak_demand_kw": 5000})
        >>> result = wizard.complete()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardConfig] = None
        self._step_handlers = {
            WizardStep.FACILITY_PROFILE: self._handle_facility_profile,
            WizardStep.INTERVAL_DATA_IMPORT: self._handle_interval_data,
            WizardStep.TARIFF_CONFIGURATION: self._handle_tariff_config,
            WizardStep.PEAK_TARGETS: self._handle_peak_targets,
            WizardStep.BESS_PARAMETERS: self._handle_bess_params,
            WizardStep.LOAD_SHIFT_PREFERENCES: self._handle_load_shift,
            WizardStep.CP_SETTINGS: self._handle_cp_settings,
            WizardStep.ALERT_PREFERENCES: self._handle_alert_prefs,
            WizardStep.REVIEW_CONFIRM: self._handle_review,
        }
        self.logger.info("SetupWizard initialized: 9 steps, 8 presets")

    def start(self) -> WizardConfig:
        """Start a new wizard session.

        Returns:
            Initial WizardConfig with all steps in PENDING status.
        """
        wizard_id = _compute_hash(f"ps-wizard:{_utcnow().isoformat()}")[:16]
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
            typical_shaveable_pct=preset_data.get("typical_shaveable_pct", 0.0),
            recommended_bess_kw=preset_data.get("bess_kw", 0.0),
            recommended_bess_kwh=preset_data.get("bess_kwh", 0.0),
            recommended_chemistry=preset_data.get("chemistry", "lfp"),
            controllable_categories=preset_data.get("controllable", []),
            typical_demand_charge=preset_data.get("demand_charge", 0.0),
            recommended_mode=preset_data.get("mode", "combined"),
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
                self._state.facility_profile = FacilityPSProfile()
            self._state.facility_profile.facility_name = data.get("facility_name", "")
            self._state.facility_profile.facility_type = data.get("facility_type", "commercial_office")
            self._state.facility_profile.peak_demand_kw = data.get("peak_demand_kw", 0.0)
            self._state.facility_profile.floor_area_m2 = data.get("floor_area_m2", 0.0)
        return errors

    def _handle_interval_data(self, data: Dict[str, Any]) -> List[str]:
        """Handle interval data import step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.meter_id = data.get("meter_id", "")
        return []

    def _handle_tariff_config(self, data: Dict[str, Any]) -> List[str]:
        """Handle tariff configuration step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.utility_name = data.get("utility_name", "")
            self._state.facility_profile.rate_schedule = data.get("rate_schedule", "")
            self._state.facility_profile.demand_charge_usd_per_kw = data.get("demand_charge_usd_per_kw", 0.0)
            self._state.facility_profile.ratchet_pct = data.get("ratchet_pct", 0.0)
        return []

    def _handle_peak_targets(self, data: Dict[str, Any]) -> List[str]:
        """Handle peak targets step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.peak_shaving_target_kw = data.get("peak_shaving_target_kw", 0.0)
            self._state.facility_profile.peak_shaving_mode = data.get("peak_shaving_mode", "combined")
        return []

    def _handle_bess_params(self, data: Dict[str, Any]) -> List[str]:
        """Handle BESS parameters step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.bess_power_kw = data.get("bess_power_kw", 0.0)
            self._state.facility_profile.bess_capacity_kwh = data.get("bess_capacity_kwh", 0.0)
            self._state.facility_profile.bess_chemistry = data.get("bess_chemistry", "lfp")
            self._state.facility_profile.bess_min_soc_pct = data.get("bess_min_soc_pct", 10.0)
            self._state.facility_profile.bess_max_soc_pct = data.get("bess_max_soc_pct", 95.0)
        return []

    def _handle_load_shift(self, data: Dict[str, Any]) -> List[str]:
        """Handle load shift preferences step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.controllable_load_kw = data.get("controllable_load_kw", 0.0)
        return []

    def _handle_cp_settings(self, data: Dict[str, Any]) -> List[str]:
        """Handle coincident peak settings step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.cp_program = data.get("cp_program", "none")
        return []

    def _handle_alert_prefs(self, data: Dict[str, Any]) -> List[str]:
        """Handle alert preferences step."""
        if self._state and self._state.facility_profile:
            self._state.facility_profile.alert_email = data.get("alert_email", "")
            self._state.facility_profile.reporting_frequency = data.get("reporting_frequency", "monthly")
        return []

    def _handle_review(self, data: Dict[str, Any]) -> List[str]:
        """Handle review and confirm step."""
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
                self._state.completed_at = _utcnow()
        except ValueError:
            pass

    # ---- Result Generation ----

    def _generate_result(self) -> SetupResult:
        """Generate the final setup result."""
        if self._state is None:
            return SetupResult()

        fp = self._state.facility_profile or FacilityPSProfile()

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        engines = [
            "profile_analysis_engine", "peak_identification_engine",
            "demand_charge_engine", "bess_sizing_engine",
            "load_shifting_engine", "cp_management_engine",
            "ratchet_analysis_engine", "power_factor_engine",
            "financial_modeling_engine", "reporting_engine",
        ]

        config_hash = _compute_hash({
            "facility": fp.facility_name,
            "type": fp.facility_type,
            "peak": fp.peak_demand_kw,
            "target": fp.peak_shaving_target_kw,
            "mode": fp.peak_shaving_mode,
        })

        result = SetupResult(
            facility_name=fp.facility_name,
            facility_type=fp.facility_type,
            peak_demand_kw=fp.peak_demand_kw,
            peak_shaving_target_kw=fp.peak_shaving_target_kw,
            bess_power_kw=fp.bess_power_kw,
            bess_capacity_kwh=fp.bess_capacity_kwh,
            controllable_load_kw=fp.controllable_load_kw,
            demand_charge_usd_per_kw=fp.demand_charge_usd_per_kw,
            peak_shaving_mode=fp.peak_shaving_mode,
            total_steps_completed=completed_count,
            engines_enabled=engines,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
