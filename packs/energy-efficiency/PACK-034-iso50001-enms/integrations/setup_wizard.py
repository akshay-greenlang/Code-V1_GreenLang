# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided EnMS Configuration for PACK-034
==============================================================

This module implements an 8-step configuration wizard for organizations
setting up the ISO 50001 Energy Management System Pack.

Wizard Steps (8):
    1. ORGANIZATION_PROFILE   -- Organization context, scope, boundaries
    2. SCOPE_BOUNDARIES       -- EnMS scope definition and exclusions
    3. ENERGY_SOURCES         -- Energy sources and carriers inventory
    4. METERING_SETUP         -- Metering hierarchy configuration
    5. SEU_IDENTIFICATION     -- Significant Energy Use identification
    6. BASELINE_CONFIG        -- Baseline period and relevant variables
    7. ENPI_DEFINITION        -- Energy Performance Indicator definition
    8. REVIEW_FINALIZE        -- Review all inputs and finalize

8 Facility Presets with auto-configured EnMS parameters.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
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
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SetupStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    ORGANIZATION_PROFILE = "organization_profile"
    SCOPE_BOUNDARIES = "scope_boundaries"
    ENERGY_SOURCES = "energy_sources"
    METERING_SETUP = "metering_setup"
    SEU_IDENTIFICATION = "seu_identification"
    BASELINE_CONFIG = "baseline_config"
    ENPI_DEFINITION = "enpi_definition"
    REVIEW_FINALIZE = "review_finalize"


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
    """Facility and organization profile data from the wizard."""

    organization_name: str = Field(default="", max_length=255)
    facility_name: str = Field(default="", max_length=255)
    facility_id: str = Field(default="")
    country: str = Field(default="DE")
    region: str = Field(default="")
    address: str = Field(default="")
    industry_sector: str = Field(default="", description="NACE code or sector name")
    floor_area_m2: float = Field(default=0.0, ge=0)
    employee_count: int = Field(default=0, ge=0)
    operating_hours_per_year: float = Field(default=4000.0, ge=0)
    facility_type: str = Field(default="manufacturing")
    year_built: int = Field(default=2000, ge=1900, le=2030)
    annual_energy_kwh: float = Field(default=0.0, ge=0)
    annual_energy_cost_eur: float = Field(default=0.0, ge=0)
    energy_sources: List[str] = Field(default_factory=lambda: ["electricity", "natural_gas"])
    scope_boundaries: str = Field(default="", description="EnMS scope statement")
    scope_exclusions: List[str] = Field(default_factory=list)
    seus_identified: int = Field(default=0, ge=0)
    baseline_year: int = Field(default=2024, ge=2000, le=2030)
    enpis_defined: int = Field(default=0, ge=0)
    iso50001_version: str = Field(default="2018")
    certification_target: bool = Field(default=True)
    energy_team_size: int = Field(default=3, ge=1, le=50)
    management_representative: str = Field(default="")


class SetupWizardStep(BaseModel):
    """Definition of a wizard step."""

    name: SetupStep = Field(...)
    display_name: str = Field(default="")
    description: str = Field(default="")
    iso50001_clause: str = Field(default="")
    required: bool = Field(default=True)


class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: SetupStep = Field(...)
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
    current_step: SetupStep = Field(default=SetupStep.ORGANIZATION_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    facility_setup: Optional[FacilitySetup] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class PresetConfig(BaseModel):
    """Facility type preset configuration."""

    preset_name: str = Field(default="")
    facility_type: str = Field(default="")
    preset_applied: bool = Field(default=False)
    typical_energy_sources: List[str] = Field(default_factory=list)
    typical_seus: List[str] = Field(default_factory=list)
    typical_enpis: List[str] = Field(default_factory=list)
    engines_enabled: List[str] = Field(default_factory=list)
    benchmark_kwh_per_m2: float = Field(default=0.0)
    typical_savings_pct: float = Field(default=0.0)
    baseline_relevant_variables: List[str] = Field(default_factory=list)


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    facility_name: str = Field(default="")
    facility_type: str = Field(default="")
    country: str = Field(default="")
    floor_area_m2: float = Field(default=0.0)
    annual_energy_kwh: float = Field(default=0.0)
    energy_sources: List[str] = Field(default_factory=list)
    seus_count: int = Field(default=0)
    enpis_count: int = Field(default=0)
    baseline_year: int = Field(default=2024)
    certification_target: bool = Field(default=True)
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

STEP_ORDER: List[SetupStep] = [
    SetupStep.ORGANIZATION_PROFILE,
    SetupStep.SCOPE_BOUNDARIES,
    SetupStep.ENERGY_SOURCES,
    SetupStep.METERING_SETUP,
    SetupStep.SEU_IDENTIFICATION,
    SetupStep.BASELINE_CONFIG,
    SetupStep.ENPI_DEFINITION,
    SetupStep.REVIEW_FINALIZE,
]

STEP_DISPLAY_NAMES: Dict[SetupStep, str] = {
    SetupStep.ORGANIZATION_PROFILE: "Organization Profile (Clause 4.1)",
    SetupStep.SCOPE_BOUNDARIES: "Scope & Boundaries (Clause 4.3)",
    SetupStep.ENERGY_SOURCES: "Energy Sources (Clause 6.3)",
    SetupStep.METERING_SETUP: "Metering Setup (Clause 6.6)",
    SetupStep.SEU_IDENTIFICATION: "SEU Identification (Clause 6.3)",
    SetupStep.BASELINE_CONFIG: "Baseline Configuration (Clause 6.5)",
    SetupStep.ENPI_DEFINITION: "EnPI Definition (Clause 6.4)",
    SetupStep.REVIEW_FINALIZE: "Review & Finalize",
}

# ---------------------------------------------------------------------------
# Facility Presets (8 presets)
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "manufacturing_facility": {
        "facility_type": "manufacturing",
        "energy_sources": ["electricity", "natural_gas", "diesel"],
        "typical_seus": ["compressed_air", "process_heating", "hvac", "lighting", "motors_drives"],
        "typical_enpis": ["kwh_per_unit_produced", "kwh_per_m2", "energy_cost_per_unit"],
        "engines": ["energy_review", "baseline", "enpi", "seu_identification", "monitoring", "action_planning", "audit_compliance", "management_review"],
        "benchmark_kwh_m2": 400,
        "typical_savings_pct": 12,
        "relevant_variables": ["production_volume", "hdd", "cdd", "operating_hours"],
    },
    "commercial_office": {
        "facility_type": "commercial_office",
        "energy_sources": ["electricity", "natural_gas"],
        "typical_seus": ["hvac", "lighting", "plug_loads", "elevator"],
        "typical_enpis": ["kwh_per_m2", "kwh_per_employee", "kwh_per_hdd"],
        "engines": ["energy_review", "baseline", "enpi", "monitoring", "action_planning", "audit_compliance", "management_review"],
        "benchmark_kwh_m2": 200,
        "typical_savings_pct": 15,
        "relevant_variables": ["occupancy", "hdd", "cdd"],
    },
    "data_center": {
        "facility_type": "data_center",
        "energy_sources": ["electricity", "diesel_backup"],
        "typical_seus": ["it_load", "cooling", "ups", "lighting"],
        "typical_enpis": ["pue", "kwh_per_rack", "dcie"],
        "engines": ["energy_review", "baseline", "enpi", "seu_identification", "monitoring", "action_planning", "audit_compliance", "management_review"],
        "benchmark_kwh_m2": 2000,
        "typical_savings_pct": 8,
        "relevant_variables": ["it_load_kw", "outside_temperature"],
    },
    "healthcare_facility": {
        "facility_type": "healthcare",
        "energy_sources": ["electricity", "natural_gas", "steam"],
        "typical_seus": ["hvac", "steam_systems", "medical_equipment", "lighting", "hot_water"],
        "typical_enpis": ["kwh_per_bed", "kwh_per_m2", "kwh_per_patient_day"],
        "engines": ["energy_review", "baseline", "enpi", "monitoring", "action_planning", "audit_compliance", "management_review"],
        "benchmark_kwh_m2": 350,
        "typical_savings_pct": 10,
        "relevant_variables": ["bed_occupancy", "hdd", "cdd"],
    },
    "retail_chain": {
        "facility_type": "retail",
        "energy_sources": ["electricity", "natural_gas"],
        "typical_seus": ["refrigeration", "hvac", "lighting", "cooking"],
        "typical_enpis": ["kwh_per_m2_sales", "kwh_per_transaction", "kwh_per_hdd"],
        "engines": ["energy_review", "baseline", "enpi", "monitoring", "action_planning", "management_review"],
        "benchmark_kwh_m2": 300,
        "typical_savings_pct": 18,
        "relevant_variables": ["sales_volume", "hdd", "cdd", "trading_hours"],
    },
    "logistics_warehouse": {
        "facility_type": "warehouse",
        "energy_sources": ["electricity", "diesel"],
        "typical_seus": ["lighting", "mhe_charging", "hvac", "dock_doors"],
        "typical_enpis": ["kwh_per_m2", "kwh_per_pallet", "kwh_per_dispatch"],
        "engines": ["energy_review", "baseline", "enpi", "monitoring", "action_planning", "management_review"],
        "benchmark_kwh_m2": 120,
        "typical_savings_pct": 20,
        "relevant_variables": ["throughput_pallets", "hdd"],
    },
    "food_processing": {
        "facility_type": "food_processing",
        "energy_sources": ["electricity", "natural_gas", "steam", "diesel"],
        "typical_seus": ["refrigeration", "steam_generation", "process_heating", "compressed_air", "hvac"],
        "typical_enpis": ["kwh_per_tonne_product", "kwh_per_m2", "thermal_kwh_per_tonne"],
        "engines": ["energy_review", "baseline", "enpi", "seu_identification", "monitoring", "action_planning", "audit_compliance", "management_review"],
        "benchmark_kwh_m2": 500,
        "typical_savings_pct": 10,
        "relevant_variables": ["production_tonnes", "hdd", "cdd", "product_mix"],
    },
    "sme_multi_site": {
        "facility_type": "sme_multi_site",
        "energy_sources": ["electricity", "natural_gas"],
        "typical_seus": ["hvac", "lighting", "process_equipment"],
        "typical_enpis": ["kwh_per_m2", "kwh_per_employee", "energy_cost_per_revenue"],
        "engines": ["energy_review", "baseline", "enpi", "monitoring", "action_planning", "management_review"],
        "benchmark_kwh_m2": 220,
        "typical_savings_pct": 15,
        "relevant_variables": ["revenue", "employee_count", "hdd"],
    },
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """8-step guided EnMS configuration wizard for PACK-034.

    Guides organizations through ISO 50001 EnMS setup with facility
    presets that auto-configure energy sources, SEUs, EnPIs, and baselines.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start_wizard("manufacturing_facility")
        >>> state = wizard.advance_step({"organization_name": "Acme GmbH", ...})
        >>> result = wizard.finalize_setup()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            SetupStep.ORGANIZATION_PROFILE: self._handle_organization_profile,
            SetupStep.SCOPE_BOUNDARIES: self._handle_scope_boundaries,
            SetupStep.ENERGY_SOURCES: self._handle_energy_sources,
            SetupStep.METERING_SETUP: self._handle_metering_setup,
            SetupStep.SEU_IDENTIFICATION: self._handle_seu_identification,
            SetupStep.BASELINE_CONFIG: self._handle_baseline_config,
            SetupStep.ENPI_DEFINITION: self._handle_enpi_definition,
            SetupStep.REVIEW_FINALIZE: self._handle_review_finalize,
        }
        self.logger.info("SetupWizard initialized: 8 steps, 8 presets")

    def start_wizard(self, preset: Optional[str] = None) -> WizardState:
        """Start a new wizard session with optional preset.

        Args:
            preset: Preset name to apply (e.g., 'manufacturing_facility').

        Returns:
            Initial WizardState with all steps in PENDING status.
        """
        wizard_id = _compute_hash(f"enms-wizard:{_utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)

        if preset:
            preset_config = self.apply_preset(preset)
            if preset_config.preset_applied and self._state.facility_setup is None:
                self._state.facility_setup = FacilitySetup(
                    facility_type=preset_config.facility_type,
                    energy_sources=preset_config.typical_energy_sources,
                )

        self.logger.info("Setup wizard started: %s, preset=%s", wizard_id, preset or "none")
        return self._state

    def advance_step(self, step_data: Dict[str, Any]) -> WizardStepState:
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

    def go_back(self) -> Optional[SetupStep]:
        """Navigate back to the previous step.

        Returns:
            Previous SetupStep, or None if already at the first step.
        """
        if self._state is None:
            return None
        try:
            idx = STEP_ORDER.index(self._state.current_step)
            if idx > 0:
                self._state.current_step = STEP_ORDER[idx - 1]
                return self._state.current_step
        except ValueError:
            pass
        return None

    def get_current_step(self) -> Optional[SetupStep]:
        """Get the current wizard step.

        Returns:
            Current SetupStep, or None if wizard not started.
        """
        if self._state is None:
            return None
        return self._state.current_step

    def apply_preset(self, preset_name: str) -> PresetConfig:
        """Load and apply a facility type preset.

        Args:
            preset_name: Preset name (e.g., 'manufacturing_facility').

        Returns:
            PresetConfig with preset configuration.

        Raises:
            ValueError: If preset name is not found.
        """
        preset_data = FACILITY_PRESETS.get(preset_name)
        if preset_data is None:
            valid = sorted(FACILITY_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Valid: {valid}")

        return PresetConfig(
            preset_name=preset_name,
            facility_type=preset_data.get("facility_type", ""),
            preset_applied=True,
            typical_energy_sources=preset_data.get("energy_sources", []),
            typical_seus=preset_data.get("typical_seus", []),
            typical_enpis=preset_data.get("typical_enpis", []),
            engines_enabled=preset_data.get("engines", []),
            benchmark_kwh_per_m2=preset_data.get("benchmark_kwh_m2", 0.0),
            typical_savings_pct=preset_data.get("typical_savings_pct", 0.0),
            baseline_relevant_variables=preset_data.get("relevant_variables", []),
        )

    def validate_step(
        self, step: SetupStep, data: Dict[str, Any],
    ) -> List[str]:
        """Validate step data without advancing.

        Args:
            step: Step to validate.
            data: Step data to validate.

        Returns:
            List of validation error strings (empty if valid).
        """
        handler = self._step_handlers.get(step)
        if handler is None:
            return [f"No handler for step '{step.value}'"]
        return handler(data)

    def finalize_setup(self) -> SetupResult:
        """Complete the wizard and generate the setup result.

        Returns:
            SetupResult with final configuration.
        """
        return self._generate_result()

    def get_wizard_progress(self) -> Dict[str, Any]:
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

    def _handle_organization_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle organization profile step (Clause 4.1)."""
        errors: List[str] = []
        if not data.get("organization_name"):
            errors.append("Organization name is required")
        if not data.get("facility_name"):
            errors.append("Facility name is required")
        if not errors and self._state:
            if self._state.facility_setup is None:
                self._state.facility_setup = FacilitySetup()
            self._state.facility_setup.organization_name = data.get("organization_name", "")
            self._state.facility_setup.facility_name = data.get("facility_name", "")
            self._state.facility_setup.country = data.get("country", "DE")
            self._state.facility_setup.floor_area_m2 = data.get("floor_area_m2", 0.0)
            self._state.facility_setup.employee_count = data.get("employee_count", 0)
            self._state.facility_setup.management_representative = data.get("management_representative", "")
        return errors

    def _handle_scope_boundaries(self, data: Dict[str, Any]) -> List[str]:
        """Handle scope and boundaries step (Clause 4.3)."""
        errors: List[str] = []
        if not data.get("scope_statement"):
            errors.append("EnMS scope statement is required")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.scope_boundaries = data.get("scope_statement", "")
            self._state.facility_setup.scope_exclusions = data.get("exclusions", [])
        return errors

    def _handle_energy_sources(self, data: Dict[str, Any]) -> List[str]:
        """Handle energy sources step (Clause 6.3)."""
        errors: List[str] = []
        sources = data.get("energy_sources", [])
        if not sources:
            errors.append("At least one energy source is required")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.energy_sources = sources
            self._state.facility_setup.annual_energy_kwh = data.get("annual_energy_kwh", 0.0)
            self._state.facility_setup.annual_energy_cost_eur = data.get("annual_energy_cost_eur", 0.0)
        return errors

    def _handle_metering_setup(self, data: Dict[str, Any]) -> List[str]:
        """Handle metering setup step (Clause 6.6)."""
        return []  # Optional step, always valid

    def _handle_seu_identification(self, data: Dict[str, Any]) -> List[str]:
        """Handle SEU identification step (Clause 6.3)."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.seus_identified = data.get("seus_count", 0)
        return []

    def _handle_baseline_config(self, data: Dict[str, Any]) -> List[str]:
        """Handle baseline configuration step (Clause 6.5)."""
        errors: List[str] = []
        baseline_year = data.get("baseline_year", 0)
        if baseline_year < 2000 or baseline_year > 2030:
            errors.append("Baseline year must be between 2000 and 2030")
        if not errors and self._state and self._state.facility_setup:
            self._state.facility_setup.baseline_year = baseline_year
        return errors

    def _handle_enpi_definition(self, data: Dict[str, Any]) -> List[str]:
        """Handle EnPI definition step (Clause 6.4)."""
        if self._state and self._state.facility_setup:
            self._state.facility_setup.enpis_defined = data.get("enpis_count", 0)
        return []

    def _handle_review_finalize(self, data: Dict[str, Any]) -> List[str]:
        """Handle review and finalize step."""
        if not data.get("confirmed", False):
            return ["Configuration must be confirmed to complete"]
        return []

    # ---- Navigation ----

    def _advance_to_next(self, current: SetupStep) -> None:
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

        # Apply preset for engines
        preset_data = FACILITY_PRESETS.get(fs.facility_type, {})
        engines = preset_data.get("engines", [])

        config_hash = _compute_hash({
            "organization": fs.organization_name,
            "facility": fs.facility_name,
            "type": fs.facility_type,
            "energy": fs.annual_energy_kwh,
        })

        result = SetupResult(
            organization_name=fs.organization_name,
            facility_name=fs.facility_name,
            facility_type=fs.facility_type,
            country=fs.country,
            floor_area_m2=fs.floor_area_m2,
            annual_energy_kwh=fs.annual_energy_kwh,
            energy_sources=fs.energy_sources,
            seus_count=fs.seus_identified,
            enpis_count=fs.enpis_defined,
            baseline_year=fs.baseline_year,
            certification_target=fs.certification_target,
            preset_applied=fs.facility_type,
            engines_enabled=engines,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
