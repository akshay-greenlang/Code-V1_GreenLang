# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Intensity Metrics Configuration Wizard for PACK-046
==========================================================================

Provides an 8-step guided configuration wizard for setting up intensity
metrics calculation including organisation profile, denominator selection,
scope inclusion, benchmark peer group, target parameters, reporting
framework mapping, time series configuration, and output preferences.

Steps (8 total):
    1. OrganisationProfile       - Company, sector, reporting boundary
    2. DenominatorSelection      - Choose denominator types and units
    3. ScopeInclusion            - Select scopes and Scope 2 method
    4. BenchmarkPeerGroup        - Define peer group for comparison
    5. TargetParameters          - Set intensity reduction targets
    6. ReportingFrameworks       - Map to CDP, GRI, CSRD, TCFD
    7. TimeSeriesConfig          - Historical periods and base year
    8. OutputPreferences         - Report formats, visualisation options

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-046 Intensity Metrics
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
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
# Enumerations
# ---------------------------------------------------------------------------


class SetupStep(str, Enum):
    """Wizard setup steps."""

    ORGANISATION_PROFILE = "organisation_profile"
    DENOMINATOR_SELECTION = "denominator_selection"
    SCOPE_INCLUSION = "scope_inclusion"
    BENCHMARK_PEER_GROUP = "benchmark_peer_group"
    TARGET_PARAMETERS = "target_parameters"
    REPORTING_FRAMEWORKS = "reporting_frameworks"
    TIME_SERIES_CONFIG = "time_series_config"
    OUTPUT_PREFERENCES = "output_preferences"


class StepStatus(str, Enum):
    """Step completion status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Step Ordering and Descriptions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupStep] = [
    SetupStep.ORGANISATION_PROFILE,
    SetupStep.DENOMINATOR_SELECTION,
    SetupStep.SCOPE_INCLUSION,
    SetupStep.BENCHMARK_PEER_GROUP,
    SetupStep.TARGET_PARAMETERS,
    SetupStep.REPORTING_FRAMEWORKS,
    SetupStep.TIME_SERIES_CONFIG,
    SetupStep.OUTPUT_PREFERENCES,
]

STEP_DESCRIPTIONS: Dict[SetupStep, str] = {
    SetupStep.ORGANISATION_PROFILE: (
        "Configure organisation name, sector classification (GICS/NACE), "
        "and reporting boundary (operational control, financial control, equity share)"
    ),
    SetupStep.DENOMINATOR_SELECTION: (
        "Select denominator types for intensity calculations "
        "(revenue, FTE, production volume, floor area, etc.) and units"
    ),
    SetupStep.SCOPE_INCLUSION: (
        "Define which emission scopes to include (Scope 1, 2, 3) and "
        "Scope 2 method (location-based, market-based, or both)"
    ),
    SetupStep.BENCHMARK_PEER_GROUP: (
        "Define peer group for benchmarking by sector, geography, "
        "company size, and benchmark data sources (CDP, TPI, GRESB)"
    ),
    SetupStep.TARGET_PARAMETERS: (
        "Set intensity reduction targets including base year, target year, "
        "reduction percentage, and SBTi alignment"
    ),
    SetupStep.REPORTING_FRAMEWORKS: (
        "Map intensity metrics to disclosure frameworks "
        "(CDP, GRI 305, CSRD ESRS E1, TCFD)"
    ),
    SetupStep.TIME_SERIES_CONFIG: (
        "Configure historical time series periods, base year reference, "
        "and recalculation handling"
    ),
    SetupStep.OUTPUT_PREFERENCES: (
        "Set report output formats (PDF, Excel, JSON), visualisation "
        "options, and distribution preferences"
    ),
}

STEP_RECOMMENDATIONS: Dict[SetupStep, List[str]] = {
    SetupStep.ORGANISATION_PROFILE: [
        "Use GICS sector classification for CDP alignment",
        "Select operational control for most accurate Scope 1-2 reporting",
    ],
    SetupStep.DENOMINATOR_SELECTION: [
        "Revenue is the most common denominator for cross-sector comparison",
        "Include sector-specific physical denominators for SBTi SDA alignment",
        "FTE is required for Scope 3 employee commuting intensity",
    ],
    SetupStep.SCOPE_INCLUSION: [
        "Scope 1+2 is minimum for CDP and CSRD reporting",
        "Include Scope 3 for comprehensive intensity analysis",
        "Report both location-based and market-based Scope 2",
    ],
    SetupStep.BENCHMARK_PEER_GROUP: [
        "Select peers with similar size and geographic footprint",
        "CDP provides the largest public benchmark dataset",
        "Use TPI for high-emitting sectors (steel, cement, power)",
    ],
    SetupStep.TARGET_PARAMETERS: [
        "SBTi recommends 4.2% annual linear reduction for 1.5C alignment",
        "Align base year with your GHG inventory base year (PACK-045)",
        "Set near-term (2030) and long-term (2050) targets",
    ],
    SetupStep.REPORTING_FRAMEWORKS: [
        "CDP Climate Change questionnaire requires Scope 1+2 intensity",
        "GRI 305-4 requires at least one intensity ratio",
        "CSRD ESRS E1-6 requires GHG intensity per net revenue",
    ],
    SetupStep.TIME_SERIES_CONFIG: [
        "Include at least 3 years of historical data for trend analysis",
        "Use recalculation-adjusted data from PACK-045 for consistency",
    ],
    SetupStep.OUTPUT_PREFERENCES: [
        "Excel output supports downstream analysis and audit",
        "PDF output is suitable for board presentations",
        "JSON output enables API integration and dashboards",
    ],
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class StepData(BaseModel):
    """Data collected for a wizard step."""

    step: str
    status: str = StepStatus.NOT_STARTED.value
    data: Dict[str, Any] = Field(default_factory=dict)
    completed_at: Optional[str] = None
    validated: bool = False
    errors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class WizardState(BaseModel):
    """Current state of the setup wizard."""

    wizard_id: str = ""
    current_step: str = SetupStep.ORGANISATION_PROFILE.value
    steps: List[StepData] = Field(default_factory=list)
    overall_progress_pct: float = 0.0
    started_at: str = ""
    last_updated: str = ""
    is_complete: bool = False


class WizardInput(BaseModel):
    """Input for completing a wizard step."""

    step_name: str = Field(..., description="Name of the step to complete")
    data: Dict[str, Any] = Field(..., description="Data for this step")


class WizardResult(BaseModel):
    """Result of completing a wizard step."""

    success: bool
    step_name: str = ""
    validated: bool = False
    errors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    state: Optional[WizardState] = None


class PackConfigOutput(BaseModel):
    """Final pack configuration produced by the wizard."""

    config_id: str = ""
    company_name: str = ""
    sector: str = ""
    consolidation_approach: str = ""
    denominator_types: List[str] = Field(default_factory=list)
    denominator_units: Dict[str, str] = Field(default_factory=dict)
    scopes_included: List[str] = Field(default_factory=list)
    scope2_method: str = "location_based"
    peer_group: Dict[str, Any] = Field(default_factory=dict)
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    reporting_frameworks: List[str] = Field(default_factory=list)
    base_year: str = ""
    historical_periods: List[str] = Field(default_factory=list)
    output_formats: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Wizard Implementation
# ---------------------------------------------------------------------------


class SetupWizard:
    """
    8-step intensity metrics configuration wizard.

    Guides users through the complete setup of intensity metrics
    including denominator selection, scope inclusion, benchmark
    peer group definition, and target parameters.

    Attributes:
        _state: Current wizard state.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> result = await wizard.complete_step(WizardInput(
        ...     step_name="organisation_profile",
        ...     data={"company_name": "ACME", "sector": "industrials"}
        ... ))
    """

    def __init__(self) -> None:
        """Initialize SetupWizard."""
        self._state: Optional[WizardState] = None
        logger.info("SetupWizard initialized: %d steps", len(STEP_ORDER))

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initial WizardState with all steps at NOT_STARTED.
        """
        wizard_id = f"wizard-{_new_uuid()[:8]}"
        steps = []
        for step in STEP_ORDER:
            step_enum = SetupStep(step.value)
            recommendations = STEP_RECOMMENDATIONS.get(step_enum, [])
            steps.append(StepData(
                step=step.value,
                recommendations=recommendations,
            ))

        self._state = WizardState(
            wizard_id=wizard_id,
            steps=steps,
            started_at=_utcnow().isoformat(),
            last_updated=_utcnow().isoformat(),
        )
        logger.info("Wizard session started: %s", wizard_id)
        return self._state

    async def complete_step(self, wizard_input: WizardInput) -> WizardResult:
        """
        Complete a wizard step with provided data.

        Args:
            wizard_input: WizardInput with step name and data.

        Returns:
            WizardResult with validation status and updated state.

        Raises:
            ValueError: If wizard not started or step invalid.
        """
        if self._state is None:
            raise ValueError("Wizard not started. Call start() first.")

        step_name = wizard_input.step_name
        data = wizard_input.data
        step_found = False

        for step_data in self._state.steps:
            if step_data.step == step_name:
                validation_errors = self._validate_step(step_name, data)
                step_data.data = data
                step_data.validated = len(validation_errors) == 0
                step_data.errors = validation_errors
                step_data.status = (
                    StepStatus.COMPLETED.value if step_data.validated
                    else StepStatus.ERROR.value
                )
                step_data.completed_at = _utcnow().isoformat()
                step_found = True
                break

        if not step_found:
            raise ValueError(f"Unknown step: {step_name}")

        # Update progress
        completed = sum(
            1 for s in self._state.steps
            if s.status == StepStatus.COMPLETED.value
        )
        self._state.overall_progress_pct = (completed / len(STEP_ORDER)) * 100
        self._state.last_updated = _utcnow().isoformat()

        # Advance current step
        for step in STEP_ORDER:
            sd = next(
                (s for s in self._state.steps if s.step == step.value), None
            )
            if sd and sd.status != StepStatus.COMPLETED.value:
                self._state.current_step = step.value
                break

        if completed == len(STEP_ORDER):
            self._state.is_complete = True

        # Build result
        step_data_obj = next(
            (s for s in self._state.steps if s.step == step_name), None
        )
        errors = step_data_obj.errors if step_data_obj else []
        recommendations = step_data_obj.recommendations if step_data_obj else []

        logger.info(
            "Step %s completed. Progress: %.0f%%",
            step_name, self._state.overall_progress_pct,
        )

        return WizardResult(
            success=len(errors) == 0,
            step_name=step_name,
            validated=len(errors) == 0,
            errors=errors,
            recommendations=recommendations,
            state=self._state,
        )

    def get_state(self) -> Optional[WizardState]:
        """Get current wizard state."""
        return self._state

    def get_step_info(self, step_name: str) -> Dict[str, Any]:
        """Get information about a specific step.

        Args:
            step_name: Step enum value.

        Returns:
            Dictionary with step details.
        """
        try:
            step = SetupStep(step_name)
        except ValueError:
            return {"error": f"Unknown step: {step_name}"}
        return {
            "name": step.value,
            "description": STEP_DESCRIPTIONS.get(step, ""),
            "order": str(STEP_ORDER.index(step) + 1),
            "recommendations": STEP_RECOMMENDATIONS.get(step, []),
        }

    async def generate_config(self) -> PackConfigOutput:
        """
        Generate final pack configuration from wizard data.

        Returns:
            PackConfigOutput with all configuration parameters.

        Raises:
            ValueError: If wizard not started or incomplete.
        """
        if self._state is None:
            raise ValueError("Wizard not started")

        step_data_map: Dict[str, Dict[str, Any]] = {}
        for s in self._state.steps:
            step_data_map[s.step] = s.data

        org = step_data_map.get(SetupStep.ORGANISATION_PROFILE.value, {})
        denom = step_data_map.get(SetupStep.DENOMINATOR_SELECTION.value, {})
        scope = step_data_map.get(SetupStep.SCOPE_INCLUSION.value, {})
        bench = step_data_map.get(SetupStep.BENCHMARK_PEER_GROUP.value, {})
        targets = step_data_map.get(SetupStep.TARGET_PARAMETERS.value, {})
        frameworks = step_data_map.get(SetupStep.REPORTING_FRAMEWORKS.value, {})
        ts = step_data_map.get(SetupStep.TIME_SERIES_CONFIG.value, {})
        output = step_data_map.get(SetupStep.OUTPUT_PREFERENCES.value, {})

        config = PackConfigOutput(
            config_id=f"config-{_new_uuid()[:8]}",
            company_name=org.get("company_name", ""),
            sector=org.get("sector", ""),
            consolidation_approach=org.get(
                "consolidation_approach", "operational_control"
            ),
            denominator_types=denom.get("types", []),
            denominator_units=denom.get("units", {}),
            scopes_included=scope.get("scopes", []),
            scope2_method=scope.get("scope2_method", "location_based"),
            peer_group=bench,
            targets=targets.get("targets", []),
            reporting_frameworks=frameworks.get("frameworks", []),
            base_year=ts.get("base_year", ""),
            historical_periods=ts.get("periods", []),
            output_formats=output.get("formats", ["pdf", "xlsx"]),
            generated_at=_utcnow().isoformat(),
        )
        config.provenance_hash = _compute_hash(config.model_dump())

        logger.info("Pack configuration generated: %s", config.config_id)
        return config

    def _validate_step(
        self, step_name: str, data: Dict[str, Any]
    ) -> List[str]:
        """Validate data for a specific step.

        Args:
            step_name: Step name to validate.
            data: Step data to validate.

        Returns:
            List of validation error messages (empty = valid).
        """
        errors: List[str] = []

        if step_name == SetupStep.ORGANISATION_PROFILE.value:
            if not data.get("company_name"):
                errors.append("Company name is required")
            if not data.get("sector"):
                errors.append("Sector classification is required")

        elif step_name == SetupStep.DENOMINATOR_SELECTION.value:
            types = data.get("types", [])
            if not types:
                errors.append("At least one denominator type must be selected")

        elif step_name == SetupStep.SCOPE_INCLUSION.value:
            scopes = data.get("scopes", [])
            if not scopes:
                errors.append("At least one scope must be selected")

        elif step_name == SetupStep.TARGET_PARAMETERS.value:
            targets = data.get("targets", [])
            for target in targets:
                if target.get("reduction_pct", 0) <= 0:
                    errors.append("Reduction percentage must be positive")
                if target.get("reduction_pct", 0) > 100:
                    errors.append("Reduction percentage cannot exceed 100%")

        elif step_name == SetupStep.TIME_SERIES_CONFIG.value:
            if not data.get("base_year"):
                errors.append("Base year is required")

        return errors

    def health_check(self) -> Dict[str, Any]:
        """Check wizard health status."""
        return {
            "bridge": "SetupWizard",
            "status": "healthy",
            "version": _MODULE_VERSION,
            "steps": len(STEP_ORDER),
            "active_session": self._state is not None,
        }
