# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Facility Configuration for PACK-035
=================================================================

This module implements an 8-step configuration wizard for facilities setting
up the Energy Benchmark Pack.

Wizard Steps (8):
    1. BUILDING_TYPE_SELECTION    -- Select building type and benchmark category
    2. FACILITY_REGISTRATION     -- Register facility name, location, ID
    3. FLOOR_AREA_DEFINITION     -- Define gross/conditioned/treated floor areas
    4. METERING_SETUP            -- Configure meter hierarchy and data sources
    5. WEATHER_STATION_SELECTION -- Select nearest weather station for normalisation
    6. BENCHMARK_SOURCE_SELECTION-- Choose benchmark database (CIBSE TM46, ENERGY STAR, etc.)
    7. REPORT_PREFERENCES        -- Configure report format and sections
    8. CONFIRMATION              -- Review all inputs and confirm

8 facility presets with auto-configured benchmark parameters.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-035 Energy Benchmark
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

    BUILDING_TYPE_SELECTION = "building_type_selection"
    FACILITY_REGISTRATION = "facility_registration"
    FLOOR_AREA_DEFINITION = "floor_area_definition"
    METERING_SETUP = "metering_setup"
    WEATHER_STATION_SELECTION = "weather_station_selection"
    BENCHMARK_SOURCE_SELECTION = "benchmark_source_selection"
    REPORT_PREFERENCES = "report_preferences"
    CONFIRMATION = "confirmation"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class FacilityType(str, Enum):
    """Facility types for benchmark context."""

    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    EDUCATION = "education"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    DATA_CENTRE = "data_centre"
    MIXED_USE = "mixed_use"
    RESIDENTIAL_MULTI = "residential_multi"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class SetupWizardConfig(BaseModel):
    """Configuration for the setup wizard."""

    pack_id: str = Field(default="PACK-035")
    pack_version: str = Field(default="1.0.0")
    enable_provenance: bool = Field(default=True)


class FacilitySetup(BaseModel):
    """Facility profile data from the wizard."""

    facility_name: str = Field(default="", min_length=0, max_length=255)
    facility_id: str = Field(default="")
    building_type: str = Field(default="office")
    country: str = Field(default="GB")
    region: str = Field(default="")
    address: str = Field(default="")
    postcode: str = Field(default="")
    latitude: float = Field(default=0.0)
    longitude: float = Field(default=0.0)
    gross_floor_area_m2: float = Field(default=0.0, ge=0)
    conditioned_area_m2: float = Field(default=0.0, ge=0)
    treated_floor_area_m2: float = Field(default=0.0, ge=0)
    year_built: int = Field(default=2000, ge=1800, le=2030)
    operating_hours_per_week: float = Field(default=50.0, ge=0)
    occupant_count: int = Field(default=0, ge=0)
    main_meter_id: str = Field(default="")
    sub_meter_count: int = Field(default=0, ge=0)
    weather_station_id: str = Field(default="")
    benchmark_source: str = Field(default="cibse_tm46")
    report_format: str = Field(default="pdf", description="pdf|html|excel")
    include_portfolio_view: bool = Field(default=False)


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


class WizardState(BaseModel):
    """Complete state of the setup wizard."""

    wizard_id: str = Field(default="")
    current_step: WizardStep = Field(default=WizardStep.BUILDING_TYPE_SELECTION)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    facility_setup: Optional[FacilitySetup] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class StepResult(BaseModel):
    """Result of processing a wizard step."""

    step: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    validation_errors: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    country: str = Field(default="")
    gross_floor_area_m2: float = Field(default=0.0)
    benchmark_source: str = Field(default="")
    weather_station_id: str = Field(default="")
    report_format: str = Field(default="pdf")
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

STEP_ORDER: List[WizardStep] = [
    WizardStep.BUILDING_TYPE_SELECTION,
    WizardStep.FACILITY_REGISTRATION,
    WizardStep.FLOOR_AREA_DEFINITION,
    WizardStep.METERING_SETUP,
    WizardStep.WEATHER_STATION_SELECTION,
    WizardStep.BENCHMARK_SOURCE_SELECTION,
    WizardStep.REPORT_PREFERENCES,
    WizardStep.CONFIRMATION,
]

STEP_DISPLAY_NAMES: Dict[WizardStep, str] = {
    WizardStep.BUILDING_TYPE_SELECTION: "Building Type Selection",
    WizardStep.FACILITY_REGISTRATION: "Facility Registration",
    WizardStep.FLOOR_AREA_DEFINITION: "Floor Area Definition",
    WizardStep.METERING_SETUP: "Metering Setup",
    WizardStep.WEATHER_STATION_SELECTION: "Weather Station Selection",
    WizardStep.BENCHMARK_SOURCE_SELECTION: "Benchmark Source Selection",
    WizardStep.REPORT_PREFERENCES: "Report Preferences",
    WizardStep.CONFIRMATION: "Review & Confirm",
}


# ---------------------------------------------------------------------------
# Facility Presets (8 presets)
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "office_standard": {
        "building_type": "office",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "general_office",
        "typical_eui_kwh_m2": 120,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "trend", "reporting"],
    },
    "office_premium": {
        "building_type": "office",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "general_office",
        "typical_eui_kwh_m2": 120,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "trend", "portfolio", "carbon", "reporting"],
    },
    "retail_standard": {
        "building_type": "retail",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "general_retail",
        "typical_eui_kwh_m2": 165,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "reporting"],
    },
    "hotel_standard": {
        "building_type": "hotel",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "hotel",
        "typical_eui_kwh_m2": 330,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "trend", "reporting"],
    },
    "hospital_standard": {
        "building_type": "hospital",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "general_hospital",
        "typical_eui_kwh_m2": 410,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "trend", "reporting"],
    },
    "education_standard": {
        "building_type": "education",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "secondary_school",
        "typical_eui_kwh_m2": 135,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "reporting"],
    },
    "warehouse_standard": {
        "building_type": "warehouse",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "distribution_warehouse",
        "typical_eui_kwh_m2": 55,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "reporting"],
    },
    "industrial_standard": {
        "building_type": "industrial",
        "benchmark_source": "cibse_tm46",
        "benchmark_classification": "general_industrial",
        "typical_eui_kwh_m2": 200,
        "engines": ["eui", "weather_norm", "peer_compare", "rating", "gap", "trend", "reporting"],
    },
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """8-step guided facility configuration wizard for PACK-035.

    Guides facilities through benchmark setup with presets that
    auto-configure benchmark sources, classification, and engines.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start_wizard()
        >>> result = wizard.process_step({"building_type": "office"})
        >>> final = wizard.complete_setup()
    """

    def __init__(self, config: Optional[SetupWizardConfig] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or SetupWizardConfig()
        self._state: Optional[WizardState] = None
        self._step_handlers: Dict[WizardStep, Any] = {
            WizardStep.BUILDING_TYPE_SELECTION: self._handle_building_type,
            WizardStep.FACILITY_REGISTRATION: self._handle_facility_registration,
            WizardStep.FLOOR_AREA_DEFINITION: self._handle_floor_area,
            WizardStep.METERING_SETUP: self._handle_metering,
            WizardStep.WEATHER_STATION_SELECTION: self._handle_weather_station,
            WizardStep.BENCHMARK_SOURCE_SELECTION: self._handle_benchmark_source,
            WizardStep.REPORT_PREFERENCES: self._handle_report_preferences,
            WizardStep.CONFIRMATION: self._handle_confirmation,
        }
        self.logger.info("SetupWizard initialized: 8 steps, 8 presets")

    def start_wizard(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState with all steps in PENDING status.
        """
        wizard_id = _compute_hash(f"benchmark-wizard:{_utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
        self.logger.info("Setup wizard started: %s", wizard_id)
        return self._state

    def process_step(self, step_data: Dict[str, Any]) -> StepResult:
        """Process the current wizard step with provided data.

        Args:
            step_data: Data for the current step.

        Returns:
            StepResult with validation status.
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

        return StepResult(
            step=current.value,
            status=step.status,
            validation_errors=step.validation_errors,
            execution_time_ms=step.execution_time_ms,
        )

    def validate_step(self, step_name: str, step_data: Dict[str, Any]) -> List[str]:
        """Validate step data without advancing the wizard.

        Args:
            step_name: Step name to validate.
            step_data: Data to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        try:
            step_enum = WizardStep(step_name)
        except ValueError:
            return [f"Unknown step: {step_name}"]

        handler = self._step_handlers.get(step_enum)
        if handler is None:
            return [f"No handler for step: {step_name}"]

        return handler(step_data)

    def complete_setup(self) -> SetupResult:
        """Complete the wizard and generate the setup result.

        Returns:
            SetupResult with final configuration.
        """
        return self._generate_result()

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

    def _handle_building_type(self, data: Dict[str, Any]) -> List[str]:
        """Handle building type selection step."""
        errors: List[str] = []
        btype = data.get("building_type", "")
        valid_types = [ft.value for ft in FacilityType]
        if btype and btype not in valid_types:
            errors.append(f"Unknown building type '{btype}'. Valid: {valid_types}")
        if not errors and self._state:
            if self._state.facility_setup is None:
                self._state.facility_setup = FacilitySetup()
            self._state.facility_setup.building_type = btype or "office"
        return errors

    def _handle_facility_registration(self, data: Dict[str, Any]) -> List[str]:
        """Handle facility registration step."""
        errors: List[str] = []
        if not data.get("facility_name"):
            errors.append("Facility name is required")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.facility_name = data.get("facility_name", "")
            self._state.facility_setup.country = data.get("country", "GB")
            self._state.facility_setup.address = data.get("address", "")
            self._state.facility_setup.postcode = data.get("postcode", "")
        return errors

    def _handle_floor_area(self, data: Dict[str, Any]) -> List[str]:
        """Handle floor area definition step."""
        errors: List[str] = []
        if data.get("gross_floor_area_m2", 0) <= 0:
            errors.append("Gross floor area must be greater than 0")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.gross_floor_area_m2 = data.get("gross_floor_area_m2", 0.0)
            self._state.facility_setup.conditioned_area_m2 = data.get("conditioned_area_m2", 0.0)
            self._state.facility_setup.treated_floor_area_m2 = data.get("treated_floor_area_m2", 0.0)
        return errors

    def _handle_metering(self, data: Dict[str, Any]) -> List[str]:
        """Handle metering setup step."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.main_meter_id = data.get("main_meter_id", "")
            self._state.facility_setup.sub_meter_count = data.get("sub_meter_count", 0)
        return []

    def _handle_weather_station(self, data: Dict[str, Any]) -> List[str]:
        """Handle weather station selection step."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.weather_station_id = data.get("weather_station_id", "")
            self._state.facility_setup.latitude = data.get("latitude", 0.0)
            self._state.facility_setup.longitude = data.get("longitude", 0.0)
        return []

    def _handle_benchmark_source(self, data: Dict[str, Any]) -> List[str]:
        """Handle benchmark source selection step."""
        valid_sources = ["cibse_tm46", "energy_star", "din_v_18599", "bpie", "ashrae_90_1", "custom"]
        source = data.get("benchmark_source", "")
        if source and source not in valid_sources:
            return [f"Unknown benchmark source '{source}'. Valid: {valid_sources}"]
        if self._state and self._state.facility_setup:
            self._state.facility_setup.benchmark_source = source or "cibse_tm46"
        return []

    def _handle_report_preferences(self, data: Dict[str, Any]) -> List[str]:
        """Handle report preferences step."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.report_format = data.get("report_format", "pdf")
            self._state.facility_setup.include_portfolio_view = data.get("include_portfolio_view", False)
        return []

    def _handle_confirmation(self, data: Dict[str, Any]) -> List[str]:
        """Handle confirmation step."""
        if not data.get("confirmed", False):
            return ["Configuration must be confirmed to complete"]
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

        fs = self._state.facility_setup or FacilitySetup()

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        # Find matching preset
        preset_name = ""
        engines: List[str] = []
        for pname, pdata in FACILITY_PRESETS.items():
            if pdata["building_type"] == fs.building_type:
                preset_name = pname
                engines = pdata.get("engines", [])
                break

        config_hash = _compute_hash({
            "facility": fs.facility_name,
            "type": fs.building_type,
            "area": fs.gross_floor_area_m2,
            "source": fs.benchmark_source,
        })

        result = SetupResult(
            facility_name=fs.facility_name,
            building_type=fs.building_type,
            country=fs.country,
            gross_floor_area_m2=fs.gross_floor_area_m2,
            benchmark_source=fs.benchmark_source,
            weather_station_id=fs.weather_station_id,
            report_format=fs.report_format,
            preset_applied=preset_name,
            engines_enabled=engines,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
