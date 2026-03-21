# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Building Configuration for PACK-032
=================================================================

This module implements an 8-step configuration wizard for buildings setting
up the Building Energy Assessment Pack.

Wizard Steps (8):
    1. building_type       -- Select building type and sub-type
    2. location_climate    -- Set location, country, ASHRAE climate zone
    3. building_geometry   -- GIA, floors, height, orientation
    4. envelope_chars      -- Wall/roof/floor/window construction details
    5. hvac_systems        -- Heating, cooling, ventilation system types
    6. lighting_dhw        -- Lighting type, DHW system configuration
    7. renewable_systems   -- PV, solar thermal, heat pump details
    8. regulatory_scope    -- EPBD, MEES, certification targets

8 Building Type Presets with auto-selected engines, benchmarks, and
assessment focus areas.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-032 Building Energy Assessment
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


class SetupWizardStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    BUILDING_TYPE = "building_type"
    LOCATION_CLIMATE = "location_climate"
    BUILDING_GEOMETRY = "building_geometry"
    ENVELOPE_CHARS = "envelope_chars"
    HVAC_SYSTEMS = "hvac_systems"
    LIGHTING_DHW = "lighting_dhw"
    RENEWABLE_SYSTEMS = "renewable_systems"
    REGULATORY_SCOPE = "regulatory_scope"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class BuildingTypeConfig(BaseModel):
    """Building type configuration from step 1."""

    building_type: str = Field(default="commercial_office")
    sub_type: str = Field(default="general")
    building_use: str = Field(default="office", description="Primary use")
    is_listed: bool = Field(default=False, description="Heritage/listed building")
    is_public: bool = Field(default=False, description="Public sector building")


class LocationClimateConfig(BaseModel):
    """Location and climate configuration from step 2."""

    building_name: str = Field(..., min_length=1, max_length=255)
    building_id: str = Field(default="")
    address: str = Field(default="")
    city: str = Field(default="London")
    postcode: str = Field(default="")
    country: str = Field(default="GB")
    latitude: float = Field(default=51.5, ge=-90, le=90)
    longitude: float = Field(default=-0.1, ge=-180, le=180)
    climate_zone: str = Field(default="4A")
    weather_station: str = Field(default="London")
    heating_degree_days: float = Field(default=2600, ge=0)
    cooling_degree_days: float = Field(default=100, ge=0)


class BuildingGeometryConfig(BaseModel):
    """Building geometry configuration from step 3."""

    gross_internal_area_m2: float = Field(default=1000.0, ge=0)
    net_internal_area_m2: float = Field(default=0.0, ge=0)
    number_of_floors: int = Field(default=3, ge=1, le=200)
    number_of_basements: int = Field(default=0, ge=0)
    floor_to_ceiling_height_m: float = Field(default=2.7, ge=2.0, le=20.0)
    building_orientation_deg: float = Field(default=0.0, ge=0, le=360)
    year_of_construction: int = Field(default=2000, ge=1800, le=2035)
    year_of_last_renovation: Optional[int] = Field(None, ge=1800, le=2035)
    window_to_wall_ratio: float = Field(default=0.30, ge=0, le=1.0)
    perimeter_m: float = Field(default=0.0, ge=0)


class EnvelopeConfig(BaseModel):
    """Building envelope configuration from step 4."""

    wall_construction: str = Field(default="cavity_insulated")
    wall_u_value: float = Field(default=0.30, ge=0, le=10)
    roof_construction: str = Field(default="flat_insulated")
    roof_u_value: float = Field(default=0.20, ge=0, le=10)
    floor_construction: str = Field(default="concrete_insulated")
    floor_u_value: float = Field(default=0.22, ge=0, le=10)
    window_type: str = Field(default="double_glazed")
    window_u_value: float = Field(default=1.8, ge=0, le=10)
    window_g_value: float = Field(default=0.50, ge=0, le=1)
    air_permeability_m3_m2_h: float = Field(default=7.0, ge=0, le=50)
    thermal_mass: str = Field(default="medium", description="low/medium/high")


class HVACSystemsConfig(BaseModel):
    """HVAC systems configuration from step 5."""

    heating_type: str = Field(default="gas_boiler")
    heating_fuel: str = Field(default="natural_gas")
    heating_efficiency: float = Field(default=0.88, ge=0, le=5)
    heating_distribution: str = Field(default="radiators")
    cooling_type: str = Field(default="split_system")
    cooling_cop: float = Field(default=3.0, ge=0, le=10)
    ventilation_type: str = Field(default="mechanical_extract")
    specific_fan_power_w_l_s: float = Field(default=2.0, ge=0, le=10)
    heat_recovery: bool = Field(default=False)
    heat_recovery_efficiency: float = Field(default=0.0, ge=0, le=1)
    bms_installed: bool = Field(default=False)
    refrigerant_type: str = Field(default="R410A")


class LightingDHWConfig(BaseModel):
    """Lighting and DHW configuration from step 6."""

    lighting_type: str = Field(default="fluorescent_t8")
    lighting_power_density_w_m2: float = Field(default=10.0, ge=0, le=50)
    lighting_controls: str = Field(default="manual_switching")
    occupancy_sensors: bool = Field(default=False)
    daylight_linking: bool = Field(default=False)
    dhw_system_type: str = Field(default="gas_instantaneous")
    dhw_fuel: str = Field(default="natural_gas")
    dhw_efficiency: float = Field(default=0.85, ge=0, le=5)
    dhw_demand_kwh_m2: float = Field(default=15.0, ge=0)
    external_lighting_w: float = Field(default=0.0, ge=0)


class RenewableSystemsConfig(BaseModel):
    """Renewable systems configuration from step 7."""

    pv_installed_kwp: float = Field(default=0.0, ge=0)
    pv_orientation_deg: float = Field(default=180.0, ge=0, le=360)
    pv_tilt_deg: float = Field(default=30.0, ge=0, le=90)
    solar_thermal_area_m2: float = Field(default=0.0, ge=0)
    heat_pump_installed: bool = Field(default=False)
    heat_pump_type: str = Field(default="", description="ashp/gshp/wshp")
    heat_pump_cop: float = Field(default=0.0, ge=0, le=10)
    battery_storage_kwh: float = Field(default=0.0, ge=0)
    wind_turbine_kw: float = Field(default=0.0, ge=0)
    biomass_boiler: bool = Field(default=False)
    chp_installed: bool = Field(default=False)


class RegulatoryConfig(BaseModel):
    """Regulatory scope configuration from step 8."""

    epbd_applicable: bool = Field(default=True)
    mees_applicable: bool = Field(default=True)
    target_epc_rating: str = Field(default="B")
    certification_target: str = Field(default="", description="LEED/BREEAM/ENERGY_STAR")
    certification_level: str = Field(default="")
    crrem_assessment: bool = Field(default=False)
    crrem_scenario: str = Field(default="1.5C")
    nzeb_target: bool = Field(default=False)
    include_whole_life_carbon: bool = Field(default=False)
    reporting_frameworks: List[str] = Field(default_factory=list)


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
    current_step: SetupWizardStep = Field(default=SetupWizardStep.BUILDING_TYPE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    building_type_config: Optional[BuildingTypeConfig] = Field(None)
    location_climate: Optional[LocationClimateConfig] = Field(None)
    building_geometry: Optional[BuildingGeometryConfig] = Field(None)
    envelope_config: Optional[EnvelopeConfig] = Field(None)
    hvac_systems: Optional[HVACSystemsConfig] = Field(None)
    lighting_dhw: Optional[LightingDHWConfig] = Field(None)
    renewable_systems: Optional[RenewableSystemsConfig] = Field(None)
    regulatory_config: Optional[RegulatoryConfig] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    building_name: str = Field(default="")
    building_type: str = Field(default="")
    country: str = Field(default="")
    city: str = Field(default="")
    gross_internal_area_m2: float = Field(default=0.0)
    year_of_construction: int = Field(default=0)
    heating_type: str = Field(default="")
    epc_target: str = Field(default="")
    engines_enabled: List[str] = Field(default_factory=list)
    regulations_applicable: List[str] = Field(default_factory=list)
    certifications_targeted: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupWizardStep] = [
    SetupWizardStep.BUILDING_TYPE,
    SetupWizardStep.LOCATION_CLIMATE,
    SetupWizardStep.BUILDING_GEOMETRY,
    SetupWizardStep.ENVELOPE_CHARS,
    SetupWizardStep.HVAC_SYSTEMS,
    SetupWizardStep.LIGHTING_DHW,
    SetupWizardStep.RENEWABLE_SYSTEMS,
    SetupWizardStep.REGULATORY_SCOPE,
]

STEP_DISPLAY_NAMES: Dict[SetupWizardStep, str] = {
    SetupWizardStep.BUILDING_TYPE: "Building Type Selection",
    SetupWizardStep.LOCATION_CLIMATE: "Location & Climate Zone",
    SetupWizardStep.BUILDING_GEOMETRY: "Building Geometry",
    SetupWizardStep.ENVELOPE_CHARS: "Envelope Characteristics",
    SetupWizardStep.HVAC_SYSTEMS: "HVAC Systems",
    SetupWizardStep.LIGHTING_DHW: "Lighting & Domestic Hot Water",
    SetupWizardStep.RENEWABLE_SYSTEMS: "Renewable Energy Systems",
    SetupWizardStep.REGULATORY_SCOPE: "Regulatory & Certification Scope",
}

# ---------------------------------------------------------------------------
# Building Type Presets (8 presets)
# ---------------------------------------------------------------------------

BUILDING_PRESETS: Dict[str, Dict[str, Any]] = {
    "commercial_office": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit", "iq", "wlc"],
        "benchmark_db": "cibse_tm46_office",
        "assessment_focus": ["hvac", "lighting", "envelope", "controls", "renewables"],
        "kpi": ["kwh_per_m2", "co2_kg_m2", "epc_rating", "display_energy_certificate"],
        "typical_kwh_m2": 200,
        "typical_hours": 2500,
    },
    "retail_building": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit"],
        "benchmark_db": "cibse_tm46_retail",
        "assessment_focus": ["lighting", "hvac", "refrigeration", "envelope", "renewables"],
        "kpi": ["kwh_per_m2", "kwh_per_m2_sales", "epc_rating"],
        "typical_kwh_m2": 270,
        "typical_hours": 3500,
    },
    "hotel_hospitality": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit", "iq"],
        "benchmark_db": "cibse_tm46_hotel",
        "assessment_focus": ["dhw", "hvac", "lighting", "laundry", "kitchen", "pool"],
        "kpi": ["kwh_per_m2", "kwh_per_bedroom", "epc_rating"],
        "typical_kwh_m2": 350,
        "typical_hours": 8760,
    },
    "healthcare_facility": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit", "iq"],
        "benchmark_db": "cibse_tm46_healthcare",
        "assessment_focus": ["hvac", "sterilization", "lighting", "dhw", "medical_gases"],
        "kpi": ["kwh_per_m2", "co2_kg_m2", "epc_rating"],
        "typical_kwh_m2": 400,
        "typical_hours": 8760,
    },
    "education_building": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit", "iq"],
        "benchmark_db": "cibse_tm46_education",
        "assessment_focus": ["envelope", "hvac", "lighting", "controls", "renewables"],
        "kpi": ["kwh_per_m2", "kwh_per_pupil", "epc_rating", "dec_rating"],
        "typical_kwh_m2": 170,
        "typical_hours": 1800,
    },
    "residential_multifamily": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit"],
        "benchmark_db": "sap_rdsap_residential",
        "assessment_focus": ["envelope", "heating", "dhw", "ventilation", "renewables"],
        "kpi": ["kwh_per_m2", "epc_rating", "sap_rating"],
        "typical_kwh_m2": 150,
        "typical_hours": 8760,
    },
    "mixed_use_development": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit", "iq", "wlc"],
        "benchmark_db": "mixed_use_composite",
        "assessment_focus": ["hvac", "lighting", "envelope", "shared_services", "renewables"],
        "kpi": ["kwh_per_m2", "co2_kg_m2", "epc_rating"],
        "typical_kwh_m2": 220,
        "typical_hours": 4000,
    },
    "public_sector_building": {
        "engines": ["envelope", "hvac", "lighting", "dhw", "renewable", "benchmark", "epc", "retrofit", "iq"],
        "benchmark_db": "cibse_tm46_public",
        "assessment_focus": ["envelope", "hvac", "lighting", "dec_compliance", "renewables"],
        "kpi": ["kwh_per_m2", "dec_rating", "epc_rating", "co2_kg_m2"],
        "typical_kwh_m2": 190,
        "typical_hours": 2400,
    },
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """8-step guided building configuration wizard for PACK-032.

    Guides building owners/assessors through energy assessment setup with
    building type presets that auto-configure engines, benchmarks, and
    assessment focus areas.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("building_type", {"building_type": "commercial_office"})
        >>> result = wizard.run_demo()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            SetupWizardStep.BUILDING_TYPE: self._handle_building_type,
            SetupWizardStep.LOCATION_CLIMATE: self._handle_location_climate,
            SetupWizardStep.BUILDING_GEOMETRY: self._handle_building_geometry,
            SetupWizardStep.ENVELOPE_CHARS: self._handle_envelope_chars,
            SetupWizardStep.HVAC_SYSTEMS: self._handle_hvac_systems,
            SetupWizardStep.LIGHTING_DHW: self._handle_lighting_dhw,
            SetupWizardStep.RENEWABLE_SYSTEMS: self._handle_renewable_systems,
            SetupWizardStep.REGULATORY_SCOPE: self._handle_regulatory_scope,
        }
        self.logger.info("SetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session."""
        wizard_id = _compute_hash(f"building-assessment-wizard:{_utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
        self.logger.info("Setup wizard started: %s", wizard_id)
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """Complete a wizard step with provided data.

        Args:
            step_name: Step name to complete.
            data: Step configuration data.

        Returns:
            Updated WizardState.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = SetupWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in SetupWizardStep]
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
            errors = handler(data)
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
                self._advance_step(step_enum)
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

        return self._state

    def run_demo(self) -> SetupResult:
        """Execute a pre-configured demo setup for a commercial office."""
        self.start()

        demo_steps = {
            "building_type": {
                "building_type": "commercial_office",
                "sub_type": "city_centre_office",
                "is_listed": False,
                "is_public": False,
            },
            "location_climate": {
                "building_name": "Demo Office Building",
                "city": "London",
                "country": "GB",
                "latitude": 51.51,
                "longitude": -0.13,
                "climate_zone": "4A",
                "weather_station": "London",
                "heating_degree_days": 2600,
                "cooling_degree_days": 100,
            },
            "building_geometry": {
                "gross_internal_area_m2": 5000.0,
                "net_internal_area_m2": 4250.0,
                "number_of_floors": 5,
                "floor_to_ceiling_height_m": 2.7,
                "year_of_construction": 2005,
                "window_to_wall_ratio": 0.35,
            },
            "envelope_chars": {
                "wall_construction": "cavity_insulated",
                "wall_u_value": 0.35,
                "roof_construction": "flat_insulated",
                "roof_u_value": 0.25,
                "floor_construction": "concrete_insulated",
                "floor_u_value": 0.25,
                "window_type": "double_glazed_low_e",
                "window_u_value": 1.6,
                "window_g_value": 0.40,
                "air_permeability_m3_m2_h": 7.0,
                "thermal_mass": "medium",
            },
            "hvac_systems": {
                "heating_type": "gas_boiler",
                "heating_fuel": "natural_gas",
                "heating_efficiency": 0.90,
                "cooling_type": "vrf_system",
                "cooling_cop": 3.5,
                "ventilation_type": "mechanical_supply_extract",
                "specific_fan_power_w_l_s": 1.8,
                "heat_recovery": True,
                "heat_recovery_efficiency": 0.70,
                "bms_installed": True,
            },
            "lighting_dhw": {
                "lighting_type": "led_with_t5",
                "lighting_power_density_w_m2": 8.0,
                "lighting_controls": "pir_with_daylight",
                "occupancy_sensors": True,
                "daylight_linking": True,
                "dhw_system_type": "gas_point_of_use",
                "dhw_efficiency": 0.90,
                "dhw_demand_kwh_m2": 12.0,
            },
            "renewable_systems": {
                "pv_installed_kwp": 50.0,
                "pv_orientation_deg": 180.0,
                "pv_tilt_deg": 15.0,
                "solar_thermal_area_m2": 0.0,
                "heat_pump_installed": False,
            },
            "regulatory_scope": {
                "epbd_applicable": False,
                "mees_applicable": True,
                "target_epc_rating": "B",
                "certification_target": "BREEAM",
                "certification_level": "very_good",
                "crrem_assessment": True,
                "crrem_scenario": "1.5C",
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_building_preset(self, building_type: str) -> Optional[Dict[str, Any]]:
        """Get the preset configuration for a building type.

        Args:
            building_type: Building type key.

        Returns:
            Preset configuration dict, or None if not found.
        """
        return BUILDING_PRESETS.get(building_type)

    # ---- Step Handlers ----

    def _handle_building_type(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = BuildingTypeConfig(**data)
            if self._state:
                self._state.building_type_config = config
        except Exception as exc:
            errors.append(f"Invalid building type config: {exc}")
        return errors

    def _handle_location_climate(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = LocationClimateConfig(**data)
            if self._state:
                self._state.location_climate = config
        except Exception as exc:
            errors.append(f"Invalid location/climate config: {exc}")
        return errors

    def _handle_building_geometry(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = BuildingGeometryConfig(**data)
            if self._state:
                self._state.building_geometry = config
        except Exception as exc:
            errors.append(f"Invalid building geometry: {exc}")
        return errors

    def _handle_envelope_chars(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = EnvelopeConfig(**data)
            if self._state:
                self._state.envelope_config = config
        except Exception as exc:
            errors.append(f"Invalid envelope config: {exc}")
        return errors

    def _handle_hvac_systems(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = HVACSystemsConfig(**data)
            if self._state:
                self._state.hvac_systems = config
        except Exception as exc:
            errors.append(f"Invalid HVAC config: {exc}")
        return errors

    def _handle_lighting_dhw(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = LightingDHWConfig(**data)
            if self._state:
                self._state.lighting_dhw = config
        except Exception as exc:
            errors.append(f"Invalid lighting/DHW config: {exc}")
        return errors

    def _handle_renewable_systems(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = RenewableSystemsConfig(**data)
            if self._state:
                self._state.renewable_systems = config
        except Exception as exc:
            errors.append(f"Invalid renewable systems config: {exc}")
        return errors

    def _handle_regulatory_scope(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = RegulatoryConfig(**data)
            if self._state:
                self._state.regulatory_config = config
        except Exception as exc:
            errors.append(f"Invalid regulatory config: {exc}")
        return errors

    # ---- Internal ----

    def _advance_step(self, completed_step: SetupWizardStep) -> None:
        """Advance to the next wizard step after completion."""
        if self._state is None:
            return
        idx = STEP_ORDER.index(completed_step)
        if idx < len(STEP_ORDER) - 1:
            self._state.current_step = STEP_ORDER[idx + 1]
        else:
            self._state.is_complete = True
            self._state.completed_at = _utcnow()

    def _generate_result(self) -> SetupResult:
        """Generate the final setup result from wizard state."""
        if self._state is None:
            return SetupResult()

        completed_count = sum(
            1 for s in self._state.steps.values()
            if s.status == StepStatus.COMPLETED
        )

        building_name = ""
        building_type = ""
        country = ""
        city = ""
        gia = 0.0
        year = 0
        heating = ""
        epc_target = ""
        engines: List[str] = []
        regulations: List[str] = []
        certifications: List[str] = []

        if self._state.location_climate:
            building_name = self._state.location_climate.building_name
            country = self._state.location_climate.country
            city = self._state.location_climate.city

        if self._state.building_type_config:
            building_type = self._state.building_type_config.building_type
            preset = BUILDING_PRESETS.get(building_type, {})
            engines = preset.get("engines", [])

        if self._state.building_geometry:
            gia = self._state.building_geometry.gross_internal_area_m2
            year = self._state.building_geometry.year_of_construction

        if self._state.hvac_systems:
            heating = self._state.hvac_systems.heating_type

        if self._state.regulatory_config:
            epc_target = self._state.regulatory_config.target_epc_rating
            if self._state.regulatory_config.epbd_applicable:
                regulations.append("EPBD")
            if self._state.regulatory_config.mees_applicable:
                regulations.append("MEES")
            if self._state.regulatory_config.certification_target:
                certifications.append(self._state.regulatory_config.certification_target)

        result = SetupResult(
            building_name=building_name,
            building_type=building_type,
            country=country,
            city=city,
            gross_internal_area_m2=gia,
            year_of_construction=year,
            heating_type=heating,
            epc_target=epc_target,
            engines_enabled=engines,
            regulations_applicable=regulations,
            certifications_targeted=certifications,
            total_steps_completed=completed_count,
        )
        result.configuration_hash = _compute_hash(result)
        result.provenance_hash = _compute_hash(result)

        return result
