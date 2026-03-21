# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Facility Configuration for PACK-031
=================================================================

This module implements an 8-step configuration wizard for industrial facilities
setting up the Industrial Energy Audit Pack.

Wizard Steps (8):
    1. industry_sector     -- Select industry sector and sub-sector
    2. facility_profile    -- Name, location, size, production type
    3. energy_carriers     -- Configure energy sources (electricity, gas, steam)
    4. meter_registration  -- Register main meters and sub-meters
    5. equipment_inventory -- Set up equipment inventory
    6. regulatory_scope    -- Assess regulatory obligations (EED, ISO 50001)
    7. preset_application  -- Apply industry-specific preset configuration
    8. baseline_setup      -- Configure baseline data collection period

6 Industry Presets with auto-selected engines, benchmarks, and audit focus.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
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

    INDUSTRY_SECTOR = "industry_sector"
    FACILITY_PROFILE = "facility_profile"
    ENERGY_CARRIERS = "energy_carriers"
    METER_REGISTRATION = "meter_registration"
    EQUIPMENT_INVENTORY = "equipment_inventory"
    REGULATORY_SCOPE = "regulatory_scope"
    PRESET_APPLICATION = "preset_application"
    BASELINE_SETUP = "baseline_setup"


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


class IndustrySectorConfig(BaseModel):
    """Industry sector configuration from step 1."""

    sector: str = Field(default="manufacturing")
    sub_sector: str = Field(default="general")
    nace_code: str = Field(default="C25", description="Primary NACE code")
    is_energy_intensive: bool = Field(default=False)
    eu_ets_installation: bool = Field(default=False)


class FacilityProfile(BaseModel):
    """Facility profile from step 2."""

    facility_name: str = Field(..., min_length=1, max_length=255)
    facility_id: str = Field(default="")
    country: str = Field(default="DE")
    region: str = Field(default="")
    address: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0)
    production_area_m2: float = Field(default=0.0, ge=0)
    employee_count: int = Field(default=0, ge=0)
    annual_production_units: float = Field(default=0.0, ge=0)
    production_unit_name: str = Field(default="tonnes", description="e.g., tonnes, units, m2")
    operating_hours_per_year: float = Field(default=8000.0, ge=0)
    shift_pattern: str = Field(default="3_shift", description="1_shift|2_shift|3_shift|continuous")


class EnergyCarrierConfig(BaseModel):
    """Energy carrier configuration from step 3."""

    carriers: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"carrier": "electricity", "annual_kwh": 0, "cost_eur_per_kwh": 0.15},
        {"carrier": "natural_gas", "annual_kwh": 0, "cost_eur_per_kwh": 0.05},
    ])
    total_annual_energy_kwh: float = Field(default=0.0, ge=0)
    total_annual_energy_cost_eur: float = Field(default=0.0, ge=0)
    has_onsite_generation: bool = Field(default=False)
    has_chp: bool = Field(default=False)
    has_solar_pv: bool = Field(default=False)


class MeterSetupConfig(BaseModel):
    """Meter registration configuration from step 4."""

    main_meters: List[Dict[str, Any]] = Field(default_factory=list)
    sub_meters: List[Dict[str, Any]] = Field(default_factory=list)
    total_meter_count: int = Field(default=0, ge=0)
    metering_coverage_pct: float = Field(default=0.0, ge=0, le=100)
    interval_data_available: bool = Field(default=False)
    interval_resolution: str = Field(default="15min")


class EquipmentSetupConfig(BaseModel):
    """Equipment inventory configuration from step 5."""

    equipment_categories: List[str] = Field(default_factory=list)
    total_equipment_count: int = Field(default=0, ge=0)
    total_installed_power_kw: float = Field(default=0.0, ge=0)
    has_compressed_air: bool = Field(default=False)
    has_steam_system: bool = Field(default=False)
    has_refrigeration: bool = Field(default=False)
    has_vfds: bool = Field(default=False)
    motor_inventory_complete: bool = Field(default=False)


class RegulatoryConfig(BaseModel):
    """Regulatory obligation configuration from step 6."""

    eed_applicable: bool = Field(default=True)
    eed_exempt_iso_50001: bool = Field(default=False)
    iso_50001_certified: bool = Field(default=False)
    iso_50002_audit_standard: bool = Field(default=True)
    en_16247_applicable: bool = Field(default=True)
    eu_ets_applicable: bool = Field(default=False)
    national_requirements: List[str] = Field(default_factory=list)


class PresetConfig(BaseModel):
    """Industry preset configuration from step 7."""

    preset_name: str = Field(default="")
    preset_applied: bool = Field(default=False)
    engines_enabled: List[str] = Field(default_factory=list)
    benchmark_database: str = Field(default="")
    audit_focus_areas: List[str] = Field(default_factory=list)
    enpi_templates: List[str] = Field(default_factory=list)


class BaselineConfig(BaseModel):
    """Baseline data collection configuration from step 8."""

    baseline_year: int = Field(default=2024, ge=2020, le=2035)
    baseline_start_date: str = Field(default="")
    baseline_end_date: str = Field(default="")
    data_collection_method: str = Field(default="meter_data")
    weather_normalization: bool = Field(default=True)
    production_normalization: bool = Field(default=True)
    weather_station_id: str = Field(default="")


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
    current_step: SetupWizardStep = Field(default=SetupWizardStep.INDUSTRY_SECTOR)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    sector_config: Optional[IndustrySectorConfig] = Field(None)
    facility_profile: Optional[FacilityProfile] = Field(None)
    energy_carriers: Optional[EnergyCarrierConfig] = Field(None)
    meter_setup: Optional[MeterSetupConfig] = Field(None)
    equipment_setup: Optional[EquipmentSetupConfig] = Field(None)
    regulatory_config: Optional[RegulatoryConfig] = Field(None)
    preset_config: Optional[PresetConfig] = Field(None)
    baseline_config: Optional[BaselineConfig] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    industry_sector: str = Field(default="")
    country: str = Field(default="")
    total_energy_kwh: float = Field(default=0.0)
    meter_count: int = Field(default=0)
    equipment_count: int = Field(default=0)
    regulations_applicable: List[str] = Field(default_factory=list)
    engines_enabled: List[str] = Field(default_factory=list)
    baseline_year: int = Field(default=0)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupWizardStep] = [
    SetupWizardStep.INDUSTRY_SECTOR,
    SetupWizardStep.FACILITY_PROFILE,
    SetupWizardStep.ENERGY_CARRIERS,
    SetupWizardStep.METER_REGISTRATION,
    SetupWizardStep.EQUIPMENT_INVENTORY,
    SetupWizardStep.REGULATORY_SCOPE,
    SetupWizardStep.PRESET_APPLICATION,
    SetupWizardStep.BASELINE_SETUP,
]

STEP_DISPLAY_NAMES: Dict[SetupWizardStep, str] = {
    SetupWizardStep.INDUSTRY_SECTOR: "Industry Sector Selection",
    SetupWizardStep.FACILITY_PROFILE: "Facility Registration",
    SetupWizardStep.ENERGY_CARRIERS: "Energy Carrier Configuration",
    SetupWizardStep.METER_REGISTRATION: "Meter Registration",
    SetupWizardStep.EQUIPMENT_INVENTORY: "Equipment Inventory Setup",
    SetupWizardStep.REGULATORY_SCOPE: "Regulatory Obligation Assessment",
    SetupWizardStep.PRESET_APPLICATION: "Preset Application",
    SetupWizardStep.BASELINE_SETUP: "Baseline Data Collection Period",
}

# ---------------------------------------------------------------------------
# Industry Presets (6 presets)
# ---------------------------------------------------------------------------

INDUSTRY_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "engines": ["baseline", "audit", "process_mapping", "equipment_assessment", "savings", "compressed_air", "benchmark", "report"],
        "benchmark_db": "eu_manufacturing_bref",
        "audit_focus": ["motors_drives", "compressed_air", "hvac", "lighting", "process_heat"],
        "enpi": ["kwh_per_unit", "kwh_per_m2", "specific_energy"],
    },
    "food_beverage": {
        "engines": ["baseline", "audit", "process_mapping", "equipment_assessment", "savings", "steam", "waste_heat", "benchmark", "report"],
        "benchmark_db": "eu_food_bref",
        "audit_focus": ["steam_systems", "refrigeration", "process_heat", "compressed_air", "cleaning"],
        "enpi": ["kwh_per_tonne", "steam_per_tonne", "water_per_tonne"],
    },
    "chemicals": {
        "engines": ["baseline", "audit", "process_mapping", "equipment_assessment", "savings", "steam", "waste_heat", "benchmark", "report"],
        "benchmark_db": "eu_chemicals_bref",
        "audit_focus": ["process_heat", "steam_systems", "pumps", "distillation", "waste_heat"],
        "enpi": ["kwh_per_tonne", "gj_per_tonne_product"],
    },
    "metals": {
        "engines": ["baseline", "audit", "process_mapping", "equipment_assessment", "savings", "waste_heat", "benchmark", "report"],
        "benchmark_db": "eu_metals_bref",
        "audit_focus": ["furnaces", "electric_arc", "rolling_mills", "compressed_air", "waste_heat"],
        "enpi": ["kwh_per_tonne_steel", "gj_per_tonne_casting"],
    },
    "data_centres": {
        "engines": ["baseline", "audit", "equipment_assessment", "savings", "benchmark", "report"],
        "benchmark_db": "eu_code_conduct_dc",
        "audit_focus": ["cooling_systems", "ups_efficiency", "it_load", "airflow_management", "lighting"],
        "enpi": ["pue", "kwh_per_rack", "cooling_efficiency"],
    },
    "commercial_buildings": {
        "engines": ["baseline", "audit", "equipment_assessment", "savings", "benchmark", "report"],
        "benchmark_db": "eu_buildings_epbd",
        "audit_focus": ["hvac", "lighting", "building_envelope", "hot_water", "controls"],
        "enpi": ["kwh_per_m2", "kwh_per_occupant", "energy_rating"],
    },
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """8-step guided facility configuration wizard for PACK-031.

    Guides industrial facilities through energy audit setup with industry
    presets that auto-configure engines, benchmarks, and audit focus areas.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("industry_sector", {"sector": "chemicals"})
        >>> result = wizard.run_demo()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            SetupWizardStep.INDUSTRY_SECTOR: self._handle_industry_sector,
            SetupWizardStep.FACILITY_PROFILE: self._handle_facility_profile,
            SetupWizardStep.ENERGY_CARRIERS: self._handle_energy_carriers,
            SetupWizardStep.METER_REGISTRATION: self._handle_meter_registration,
            SetupWizardStep.EQUIPMENT_INVENTORY: self._handle_equipment_inventory,
            SetupWizardStep.REGULATORY_SCOPE: self._handle_regulatory_scope,
            SetupWizardStep.PRESET_APPLICATION: self._handle_preset_application,
            SetupWizardStep.BASELINE_SETUP: self._handle_baseline_setup,
        }
        self.logger.info("SetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session."""
        wizard_id = _compute_hash(f"energy-audit-wizard:{_utcnow().isoformat()}")[:16]
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
        """Execute a pre-configured demo setup for a manufacturing facility."""
        self.start()

        demo_steps = {
            "industry_sector": {
                "sector": "manufacturing",
                "sub_sector": "automotive_parts",
                "nace_code": "C29.3",
                "is_energy_intensive": False,
                "eu_ets_installation": False,
            },
            "facility_profile": {
                "facility_name": "Demo Automotive Parts Plant",
                "country": "DE",
                "region": "Bavaria",
                "floor_area_m2": 25000.0,
                "production_area_m2": 18000.0,
                "employee_count": 450,
                "annual_production_units": 12000.0,
                "production_unit_name": "tonnes",
                "operating_hours_per_year": 6000.0,
                "shift_pattern": "2_shift",
            },
            "energy_carriers": {
                "carriers": [
                    {"carrier": "electricity", "annual_kwh": 8_000_000, "cost_eur_per_kwh": 0.18},
                    {"carrier": "natural_gas", "annual_kwh": 5_000_000, "cost_eur_per_kwh": 0.055},
                    {"carrier": "compressed_air", "annual_kwh": 1_200_000, "cost_eur_per_kwh": 0.03},
                ],
                "total_annual_energy_kwh": 14_200_000,
                "total_annual_energy_cost_eur": 1_795_000,
                "has_compressed_air_system": True,
            },
            "meter_registration": {
                "main_meters": [
                    {"name": "Main Electricity Incomer", "carrier": "electricity"},
                    {"name": "Main Gas Meter", "carrier": "natural_gas"},
                ],
                "sub_meters": [
                    {"name": "Production Hall A", "carrier": "electricity"},
                    {"name": "Production Hall B", "carrier": "electricity"},
                    {"name": "Office Block", "carrier": "electricity"},
                    {"name": "Compressor Room", "carrier": "electricity"},
                ],
                "total_meter_count": 6,
                "metering_coverage_pct": 85.0,
                "interval_data_available": True,
            },
            "equipment_inventory": {
                "equipment_categories": ["motor", "pump", "fan", "compressor", "hvac_ahu", "lighting"],
                "total_equipment_count": 120,
                "total_installed_power_kw": 3500.0,
                "has_compressed_air": True,
                "has_steam_system": False,
                "has_vfds": True,
            },
            "regulatory_scope": {
                "eed_applicable": True,
                "iso_50001_certified": False,
                "en_16247_applicable": True,
                "eu_ets_applicable": False,
            },
            "preset_application": {"preset_name": "manufacturing"},
            "baseline_setup": {
                "baseline_year": 2024,
                "baseline_start_date": "2024-01-01",
                "baseline_end_date": "2024-12-31",
                "weather_normalization": True,
                "production_normalization": True,
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_industry_preset(self, sector: str) -> Optional[Dict[str, Any]]:
        """Get the preset configuration for an industry sector.

        Args:
            sector: Industry sector key.

        Returns:
            Preset configuration dict, or None if not found.
        """
        return INDUSTRY_PRESETS.get(sector)

    # ---- Step Handlers ----

    def _handle_industry_sector(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = IndustrySectorConfig(**data)
            if self._state:
                self._state.sector_config = config
        except Exception as exc:
            errors.append(f"Invalid industry sector config: {exc}")
        return errors

    def _handle_facility_profile(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            profile = FacilityProfile(**data)
            if self._state:
                self._state.facility_profile = profile
        except Exception as exc:
            errors.append(f"Invalid facility profile: {exc}")
        return errors

    def _handle_energy_carriers(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = EnergyCarrierConfig(**data)
            if self._state:
                self._state.energy_carriers = config
        except Exception as exc:
            errors.append(f"Invalid energy carrier config: {exc}")
        return errors

    def _handle_meter_registration(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = MeterSetupConfig(**data)
            if self._state:
                self._state.meter_setup = config
        except Exception as exc:
            errors.append(f"Invalid meter config: {exc}")
        return errors

    def _handle_equipment_inventory(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = EquipmentSetupConfig(**data)
            if self._state:
                self._state.equipment_setup = config
        except Exception as exc:
            errors.append(f"Invalid equipment config: {exc}")
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

    def _handle_preset_application(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        preset_name = data.get("preset_name", "")
        preset = INDUSTRY_PRESETS.get(preset_name)

        if preset is None:
            errors.append(f"Unknown preset '{preset_name}'. Valid: {sorted(INDUSTRY_PRESETS.keys())}")
            return errors

        config = PresetConfig(
            preset_name=preset_name,
            preset_applied=True,
            engines_enabled=preset.get("engines", []),
            benchmark_database=preset.get("benchmark_db", ""),
            audit_focus_areas=preset.get("audit_focus", []),
            enpi_templates=preset.get("enpi", []),
        )
        if self._state:
            self._state.preset_config = config

        self.logger.info("Industry preset applied: %s", preset_name)
        return errors

    def _handle_baseline_setup(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            config = BaselineConfig(**data)
            if self._state:
                self._state.baseline_config = config
        except Exception as exc:
            errors.append(f"Invalid baseline config: {exc}")
        return errors

    # ---- Navigation ----

    def _advance_step(self, current: SetupWizardStep) -> None:
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

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        regulations: List[str] = []
        if self._state.regulatory_config:
            rc = self._state.regulatory_config
            if rc.eed_applicable:
                regulations.append("EED")
            if rc.en_16247_applicable:
                regulations.append("EN 16247")
            if rc.iso_50002_audit_standard:
                regulations.append("ISO 50002")
            if rc.iso_50001_certified:
                regulations.append("ISO 50001")
            if rc.eu_ets_applicable:
                regulations.append("EU ETS")

        engines: List[str] = []
        if self._state.preset_config:
            engines = list(self._state.preset_config.engines_enabled)

        config_hash = _compute_hash({
            "facility": self._state.facility_profile.facility_name if self._state.facility_profile else "",
            "sector": self._state.sector_config.sector if self._state.sector_config else "",
            "regulations": regulations,
        })

        result = SetupResult(
            facility_name=(self._state.facility_profile.facility_name if self._state.facility_profile else ""),
            industry_sector=(self._state.sector_config.sector if self._state.sector_config else ""),
            country=(self._state.facility_profile.country if self._state.facility_profile else ""),
            total_energy_kwh=(self._state.energy_carriers.total_annual_energy_kwh if self._state.energy_carriers else 0),
            meter_count=(self._state.meter_setup.total_meter_count if self._state.meter_setup else 0),
            equipment_count=(self._state.equipment_setup.total_equipment_count if self._state.equipment_setup else 0),
            regulations_applicable=regulations,
            engines_enabled=engines,
            baseline_year=(self._state.baseline_config.baseline_year if self._state.baseline_config else 0),
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
