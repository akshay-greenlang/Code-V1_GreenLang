# -*- coding: utf-8 -*-
"""
SetupWizard - 9-Step M&V Project Configuration Wizard for PACK-040
=====================================================================

This module implements a 9-step configuration wizard for setting up
Measurement & Verification projects. It guides users through project
profiling, ECM inventory, baseline period definition, meter configuration,
IPMVP option selection, adjustment planning, reporting schedule,
compliance framework selection, and final review.

Wizard Steps (9):
    1. PROJECT_PROFILE       -- Project name, facility type, location
    2. ECM_INVENTORY         -- List ECMs, categories, expected savings
    3. BASELINE_PERIOD       -- Define baseline start/end, data sources
    4. METER_CONFIG          -- Metering plan, calibration, sampling
    5. OPTION_SELECTION      -- IPMVP Option A/B/C/D selection
    6. ADJUSTMENT_PLAN       -- Routine and non-routine adjustments
    7. REPORTING_SCHEDULE    -- Report frequency, recipients, deadlines
    8. COMPLIANCE_FRAMEWORK  -- ISO 50015, FEMP 4.0, ASHRAE 14, EU EED
    9. REVIEW_CONFIRM        -- Review configuration and confirm

8 Facility Presets with auto-configured M&V parameters:
    - Commercial Office, Manufacturing, Retail Portfolio, Hospital
    - University Campus, Government FEMP, ESCO Contract, Portfolio M&V

Zero-Hallucination:
    All preset values and default configurations use deterministic
    lookup tables. No LLM calls in the configuration path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-040 Measurement & Verification
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

class WizardStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    PROJECT_PROFILE = "project_profile"
    ECM_INVENTORY = "ecm_inventory"
    BASELINE_PERIOD = "baseline_period"
    METER_CONFIG = "meter_config"
    OPTION_SELECTION = "option_selection"
    ADJUSTMENT_PLAN = "adjustment_plan"
    REPORTING_SCHEDULE = "reporting_schedule"
    COMPLIANCE_FRAMEWORK = "compliance_framework"
    REVIEW_CONFIRM = "review_confirm"

class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"

class MVTier(str, Enum):
    """M&V complexity tier presets."""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    INVESTMENT_GRADE = "investment_grade"

class IPMVPOptionWZ(str, Enum):
    """IPMVP options for wizard selection."""

    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"
    OPTION_D = "option_d"

class ReportFrequency(str, Enum):
    """M&V report generation frequency."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"

class FacilityMVType(str, Enum):
    """Facility types for M&V wizard presets."""

    COMMERCIAL_OFFICE = "commercial_office"
    MANUFACTURING = "manufacturing"
    RETAIL_PORTFOLIO = "retail_portfolio"
    HOSPITAL = "hospital"
    UNIVERSITY_CAMPUS = "university_campus"
    GOVERNMENT_FEMP = "government_femp"
    ESCO_CONTRACT = "esco_performance_contract"
    PORTFOLIO_MV = "portfolio_mv"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    step: WizardStep = Field(...)
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = Field(None)

class FacilityMVProfile(BaseModel):
    """Facility M&V profile collected in step 1."""

    project_name: str = Field(default="")
    facility_id: str = Field(default="")
    facility_type: FacilityMVType = Field(default=FacilityMVType.COMMERCIAL_OFFICE)
    floor_area_sqft: float = Field(default=0.0, ge=0.0)
    location_city: str = Field(default="")
    location_state: str = Field(default="")
    latitude: float = Field(default=0.0, ge=-90.0, le=90.0)
    longitude: float = Field(default=0.0, ge=-180.0, le=180.0)
    climate_zone: str = Field(default="4A")
    annual_energy_kwh: float = Field(default=0.0, ge=0.0)
    annual_energy_cost_usd: float = Field(default=0.0, ge=0.0)
    esco_contract: bool = Field(default=False)
    mv_tier: MVTier = Field(default=MVTier.STANDARD)

class PresetConfig(BaseModel):
    """Preset configuration for a facility type."""

    preset_name: str = Field(default="")
    facility_type: FacilityMVType = Field(default=FacilityMVType.COMMERCIAL_OFFICE)
    default_ipmvp_option: str = Field(default="option_c")
    default_baseline_months: int = Field(default=12)
    default_model_type: str = Field(default="5p")
    default_report_frequency: str = Field(default="quarterly")
    default_compliance_framework: str = Field(default="ipmvp_2022")
    weather_normalization: bool = Field(default=True)
    occupancy_normalization: bool = Field(default=False)
    production_normalization: bool = Field(default=False)
    persistence_tracking: bool = Field(default=True)
    uncertainty_analysis: bool = Field(default=True)
    cvrmse_threshold_pct: float = Field(default=25.0)
    nmbe_threshold_pct: float = Field(default=0.5)
    min_r_squared: float = Field(default=0.75)
    confidence_level_pct: float = Field(default=90.0)

class WizardConfig(BaseModel):
    """Complete wizard configuration output."""

    config_id: str = Field(default_factory=_new_uuid)
    project_name: str = Field(default="")
    facility_profile: Optional[FacilityMVProfile] = Field(None)
    ecm_count: int = Field(default=0)
    ipmvp_option: str = Field(default="option_c")
    baseline_months: int = Field(default=12)
    baseline_start: str = Field(default="")
    baseline_end: str = Field(default="")
    model_type: str = Field(default="5p")
    meter_count: int = Field(default=0)
    report_frequency: str = Field(default="quarterly")
    compliance_framework: str = Field(default="ipmvp_2022")
    weather_normalization: bool = Field(default=True)
    persistence_tracking: bool = Field(default=True)
    preset_applied: str = Field(default="")

class SetupResult(BaseModel):
    """Result of completing the setup wizard."""

    result_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="completed")
    steps_completed: int = Field(default=0)
    steps_skipped: int = Field(default=0)
    steps_total: int = Field(default=9)
    wizard_config: Optional[WizardConfig] = Field(None)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# Facility Presets
# ---------------------------------------------------------------------------

FACILITY_PRESETS: Dict[FacilityMVType, PresetConfig] = {
    FacilityMVType.COMMERCIAL_OFFICE: PresetConfig(
        preset_name="Commercial Office",
        facility_type=FacilityMVType.COMMERCIAL_OFFICE,
        default_ipmvp_option="option_c",
        default_model_type="5p",
        default_report_frequency="quarterly",
        default_compliance_framework="ipmvp_2022",
        weather_normalization=True,
        occupancy_normalization=True,
        cvrmse_threshold_pct=25.0,
        confidence_level_pct=90.0,
    ),
    FacilityMVType.MANUFACTURING: PresetConfig(
        preset_name="Manufacturing",
        facility_type=FacilityMVType.MANUFACTURING,
        default_ipmvp_option="option_c",
        default_model_type="multivariate",
        default_report_frequency="monthly",
        default_compliance_framework="iso_50015",
        weather_normalization=True,
        production_normalization=True,
        cvrmse_threshold_pct=20.0,
        confidence_level_pct=90.0,
    ),
    FacilityMVType.RETAIL_PORTFOLIO: PresetConfig(
        preset_name="Retail Portfolio",
        facility_type=FacilityMVType.RETAIL_PORTFOLIO,
        default_ipmvp_option="option_c",
        default_model_type="3p",
        default_report_frequency="quarterly",
        default_compliance_framework="ipmvp_2022",
        weather_normalization=True,
        persistence_tracking=True,
        cvrmse_threshold_pct=25.0,
    ),
    FacilityMVType.HOSPITAL: PresetConfig(
        preset_name="Hospital",
        facility_type=FacilityMVType.HOSPITAL,
        default_ipmvp_option="option_c",
        default_model_type="5p",
        default_report_frequency="monthly",
        default_compliance_framework="ipmvp_2022",
        weather_normalization=True,
        occupancy_normalization=True,
        cvrmse_threshold_pct=20.0,
        confidence_level_pct=90.0,
    ),
    FacilityMVType.UNIVERSITY_CAMPUS: PresetConfig(
        preset_name="University Campus",
        facility_type=FacilityMVType.UNIVERSITY_CAMPUS,
        default_ipmvp_option="option_c",
        default_model_type="towt",
        default_report_frequency="semi_annual",
        default_compliance_framework="ipmvp_2022",
        weather_normalization=True,
        occupancy_normalization=True,
        cvrmse_threshold_pct=25.0,
    ),
    FacilityMVType.GOVERNMENT_FEMP: PresetConfig(
        preset_name="Government FEMP",
        facility_type=FacilityMVType.GOVERNMENT_FEMP,
        default_ipmvp_option="option_c",
        default_model_type="5p",
        default_report_frequency="annual",
        default_compliance_framework="femp_4_0",
        weather_normalization=True,
        persistence_tracking=True,
        confidence_level_pct=90.0,
        cvrmse_threshold_pct=20.0,
    ),
    FacilityMVType.ESCO_CONTRACT: PresetConfig(
        preset_name="ESCO Performance Contract",
        facility_type=FacilityMVType.ESCO_CONTRACT,
        default_ipmvp_option="option_c",
        default_model_type="5p",
        default_baseline_months=12,
        default_report_frequency="monthly",
        default_compliance_framework="ipmvp_2022",
        weather_normalization=True,
        persistence_tracking=True,
        uncertainty_analysis=True,
        confidence_level_pct=90.0,
        cvrmse_threshold_pct=15.0,
        nmbe_threshold_pct=0.5,
        min_r_squared=0.80,
    ),
    FacilityMVType.PORTFOLIO_MV: PresetConfig(
        preset_name="Portfolio M&V",
        facility_type=FacilityMVType.PORTFOLIO_MV,
        default_ipmvp_option="option_c",
        default_model_type="3p",
        default_report_frequency="quarterly",
        default_compliance_framework="ipmvp_2022",
        weather_normalization=True,
        persistence_tracking=True,
        cvrmse_threshold_pct=30.0,
        confidence_level_pct=80.0,
    ),
}

STEP_ORDER: List[WizardStep] = [
    WizardStep.PROJECT_PROFILE,
    WizardStep.ECM_INVENTORY,
    WizardStep.BASELINE_PERIOD,
    WizardStep.METER_CONFIG,
    WizardStep.OPTION_SELECTION,
    WizardStep.ADJUSTMENT_PLAN,
    WizardStep.REPORTING_SCHEDULE,
    WizardStep.COMPLIANCE_FRAMEWORK,
    WizardStep.REVIEW_CONFIRM,
]

# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------

class SetupWizard:
    """9-step M&V project configuration wizard.

    Guides users through complete M&V project setup with facility-specific
    presets that auto-configure IPMVP options, statistical thresholds,
    metering requirements, and reporting schedules.

    Attributes:
        _steps: Current step states.
        _preset: Applied facility preset.

    Example:
        >>> wizard = SetupWizard()
        >>> wizard.apply_preset(FacilityMVType.ESCO_CONTRACT)
        >>> wizard.complete_step(WizardStep.PROJECT_PROFILE, data)
        >>> result = wizard.finalize()
        >>> assert result.status == "completed"
    """

    def __init__(self) -> None:
        """Initialize SetupWizard with 9 pending steps."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._steps: Dict[WizardStep, WizardStepState] = {
            step: WizardStepState(step=step) for step in STEP_ORDER
        }
        self._preset: Optional[PresetConfig] = None
        self._config = WizardConfig()
        self.logger.info("SetupWizard initialized: %d steps", len(STEP_ORDER))

    def apply_preset(
        self,
        facility_type: FacilityMVType,
    ) -> PresetConfig:
        """Apply a facility preset configuration.

        Args:
            facility_type: Facility type to load preset for.

        Returns:
            Applied preset configuration.
        """
        self._preset = FACILITY_PRESETS.get(
            facility_type,
            FACILITY_PRESETS[FacilityMVType.COMMERCIAL_OFFICE],
        )
        self._config.ipmvp_option = self._preset.default_ipmvp_option
        self._config.model_type = self._preset.default_model_type
        self._config.baseline_months = self._preset.default_baseline_months
        self._config.report_frequency = self._preset.default_report_frequency
        self._config.compliance_framework = self._preset.default_compliance_framework
        self._config.weather_normalization = self._preset.weather_normalization
        self._config.persistence_tracking = self._preset.persistence_tracking
        self._config.preset_applied = self._preset.preset_name

        self.logger.info(
            "Applied preset: %s (option=%s, model=%s)",
            self._preset.preset_name,
            self._preset.default_ipmvp_option,
            self._preset.default_model_type,
        )
        return self._preset

    def complete_step(
        self,
        step: WizardStep,
        data: Dict[str, Any],
    ) -> WizardStepState:
        """Complete a wizard step with provided data.

        Args:
            step: Step to complete.
            data: Step data payload.

        Returns:
            Updated step state.
        """
        if step not in self._steps:
            raise ValueError(f"Unknown step: {step.value}")

        state = self._steps[step]
        errors = self._validate_step_data(step, data)

        if errors:
            state.status = StepStatus.IN_PROGRESS
            state.validation_errors = errors
            self.logger.warning(
                "Step %s validation failed: %d errors", step.value, len(errors)
            )
        else:
            state.status = StepStatus.COMPLETED
            state.data = data
            state.validation_errors = []
            state.completed_at = utcnow()
            self._apply_step_data(step, data)
            self.logger.info("Step %s completed", step.value)

        return state

    def skip_step(self, step: WizardStep) -> WizardStepState:
        """Skip a wizard step.

        Args:
            step: Step to skip.

        Returns:
            Updated step state.
        """
        state = self._steps[step]
        state.status = StepStatus.SKIPPED
        self.logger.info("Step %s skipped", step.value)
        return state

    def get_step_state(self, step: WizardStep) -> WizardStepState:
        """Get the current state of a wizard step.

        Args:
            step: Step to query.

        Returns:
            Current step state.
        """
        return self._steps.get(step, WizardStepState(step=step))

    def get_progress(self) -> Dict[str, Any]:
        """Get overall wizard progress.

        Returns:
            Dict with progress information.
        """
        completed = sum(
            1 for s in self._steps.values()
            if s.status == StepStatus.COMPLETED
        )
        skipped = sum(
            1 for s in self._steps.values()
            if s.status == StepStatus.SKIPPED
        )
        total = len(STEP_ORDER)
        pct = ((completed + skipped) / total * 100.0) if total > 0 else 0.0

        return {
            "completed": completed,
            "skipped": skipped,
            "pending": total - completed - skipped,
            "total": total,
            "progress_pct": round(pct, 1),
            "current_step": self._get_current_step(),
            "preset_applied": self._config.preset_applied,
        }

    def finalize(self) -> SetupResult:
        """Finalize the wizard and produce the configuration.

        Returns:
            SetupResult with final configuration.
        """
        start_time = time.monotonic()

        completed = sum(
            1 for s in self._steps.values()
            if s.status == StepStatus.COMPLETED
        )
        skipped = sum(
            1 for s in self._steps.values()
            if s.status == StepStatus.SKIPPED
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = SetupResult(
            status="completed" if completed + skipped == len(STEP_ORDER) else "incomplete",
            steps_completed=completed,
            steps_skipped=skipped,
            wizard_config=self._config,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Wizard finalized: status=%s, completed=%d, skipped=%d",
            result.status, completed, skipped,
        )
        return result

    def get_available_presets(self) -> List[Dict[str, Any]]:
        """Get all available facility presets.

        Returns:
            List of preset summaries.
        """
        return [
            {
                "facility_type": ft.value,
                "preset_name": pc.preset_name,
                "ipmvp_option": pc.default_ipmvp_option,
                "model_type": pc.default_model_type,
                "report_frequency": pc.default_report_frequency,
                "compliance_framework": pc.default_compliance_framework,
            }
            for ft, pc in FACILITY_PRESETS.items()
        ]

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _get_current_step(self) -> Optional[str]:
        """Get the next pending step."""
        for step in STEP_ORDER:
            state = self._steps[step]
            if state.status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                return step.value
        return None

    def _validate_step_data(
        self, step: WizardStep, data: Dict[str, Any]
    ) -> List[str]:
        """Validate step data.

        Args:
            step: Step being validated.
            data: Data to validate.

        Returns:
            List of validation error messages.
        """
        errors: List[str] = []

        if step == WizardStep.PROJECT_PROFILE:
            if not data.get("project_name"):
                errors.append("Project name is required")
            if not data.get("facility_type"):
                errors.append("Facility type is required")

        elif step == WizardStep.BASELINE_PERIOD:
            if not data.get("baseline_start"):
                errors.append("Baseline start date is required")
            if not data.get("baseline_end"):
                errors.append("Baseline end date is required")

        elif step == WizardStep.OPTION_SELECTION:
            option = data.get("ipmvp_option", "")
            if option and option not in [o.value for o in IPMVPOptionWZ]:
                errors.append(f"Invalid IPMVP option: {option}")

        return errors

    def _apply_step_data(
        self, step: WizardStep, data: Dict[str, Any]
    ) -> None:
        """Apply step data to the wizard configuration.

        Args:
            step: Completed step.
            data: Step data to apply.
        """
        if step == WizardStep.PROJECT_PROFILE:
            self._config.project_name = data.get("project_name", "")
            if data.get("facility_profile"):
                self._config.facility_profile = FacilityMVProfile(
                    **data["facility_profile"]
                )

        elif step == WizardStep.ECM_INVENTORY:
            self._config.ecm_count = data.get("ecm_count", 0)

        elif step == WizardStep.BASELINE_PERIOD:
            self._config.baseline_start = data.get("baseline_start", "")
            self._config.baseline_end = data.get("baseline_end", "")
            self._config.baseline_months = data.get("baseline_months", 12)

        elif step == WizardStep.METER_CONFIG:
            self._config.meter_count = data.get("meter_count", 0)

        elif step == WizardStep.OPTION_SELECTION:
            self._config.ipmvp_option = data.get("ipmvp_option", "option_c")
            self._config.model_type = data.get("model_type", "5p")

        elif step == WizardStep.REPORTING_SCHEDULE:
            self._config.report_frequency = data.get("report_frequency", "quarterly")

        elif step == WizardStep.COMPLIANCE_FRAMEWORK:
            self._config.compliance_framework = data.get(
                "compliance_framework", "ipmvp_2022"
            )
