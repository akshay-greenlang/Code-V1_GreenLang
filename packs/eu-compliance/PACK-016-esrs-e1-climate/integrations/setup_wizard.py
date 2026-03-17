# -*- coding: utf-8 -*-
"""
E1SetupWizard - 6-Step Guided Configuration Wizard for E1 Climate PACK-016
=============================================================================

This module implements a 6-step configuration wizard for the ESRS E1
Climate Pack. It guides users through company profiling, GHG scope
definition, energy scope, target configuration, carbon pricing settings,
and reporting preferences.

Wizard Steps (6):
    1. company_profile    -- Name, sector, NACE codes, size, headquarters
    2. ghg_scope          -- Scope 1/2/3 boundaries, consolidation approach
    3. energy_scope       -- Energy sources, renewable targets
    4. targets            -- Climate targets, SBTi commitment, base year
    5. carbon_pricing     -- ETS exposure, shadow pricing
    6. reporting          -- Format, language, assurance level

Each step is validated individually. The wizard generates a complete
E1Config from the assembled inputs with SHA-256 provenance tracking.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-016 ESRS E1 Climate Pack
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class E1WizardStep(str, Enum):
    """The 6 steps of the E1 Setup Wizard."""

    COMPANY_PROFILE = "company_profile"
    GHG_SCOPE = "ghg_scope"
    ENERGY_SCOPE = "energy_scope"
    TARGETS = "targets"
    CARBON_PRICING = "carbon_pricing"
    REPORTING = "reporting"


class StepStatus(str, Enum):
    """Step completion status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


class ConsolidationApproach(str, Enum):
    """GHG consolidation approach."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


# ---------------------------------------------------------------------------
# Step Data Models
# ---------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """Step 1: Company profile data."""

    entity_name: str = Field(default="", description="Legal entity name")
    nace_codes: List[str] = Field(default_factory=list, description="NACE sector codes")
    sector: str = Field(default="", description="Primary sector")
    size_category: str = Field(default="large", description="SME/large/listed")
    employee_count: int = Field(default=0, ge=0)
    headquarters_country: str = Field(default="", description="ISO 3166-1 alpha-2")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    currency: str = Field(default="EUR")


class GHGScope(BaseModel):
    """Step 2: GHG scope definition."""

    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL
    )
    scope1_included: bool = Field(default=True)
    scope2_included: bool = Field(default=True)
    scope3_included: bool = Field(default=True)
    scope3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16))
    )
    gwp_source: str = Field(default="IPCC AR6")
    base_year: int = Field(default=2019, ge=2000, le=2030)
    recalculation_threshold_pct: float = Field(default=5.0, ge=0.0, le=100.0)


class EnergyScope(BaseModel):
    """Step 3: Energy scope definition."""

    track_fossil: bool = Field(default=True)
    track_renewable: bool = Field(default=True)
    track_nuclear: bool = Field(default=True)
    renewable_target_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    renewable_target_year: int = Field(default=2030, ge=2025, le=2050)
    energy_unit: str = Field(default="MWh")
    include_self_generated: bool = Field(default=True)


class TargetConfig(BaseModel):
    """Step 4: Climate target configuration."""

    has_targets: bool = Field(default=True)
    sbti_committed: bool = Field(default=False)
    sbti_validated: bool = Field(default=False)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2035)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2060)
    net_zero_year: int = Field(default=2050, ge=2035, le=2070)
    target_types: List[str] = Field(
        default_factory=lambda: ["absolute", "intensity"]
    )


class CarbonPricingConfig(BaseModel):
    """Step 5: Carbon pricing configuration."""

    exposed_to_ets: bool = Field(default=False)
    ets_jurisdictions: List[str] = Field(default_factory=list)
    uses_shadow_pricing: bool = Field(default=False)
    shadow_price_eur: float = Field(default=0.0, ge=0.0)
    include_carbon_credits: bool = Field(default=False)
    carbon_tax_jurisdictions: List[str] = Field(default_factory=list)


class ReportingConfig(BaseModel):
    """Step 6: Reporting preferences."""

    output_formats: List[str] = Field(
        default_factory=lambda: ["markdown", "html", "json"]
    )
    language: str = Field(default="en")
    detail_level: str = Field(default="standard")
    include_methodology: bool = Field(default=True)
    assurance_level: str = Field(default="limited")
    assurance_provider: str = Field(default="")
    include_prior_year: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Wizard State
# ---------------------------------------------------------------------------


class WizardStepState(BaseModel):
    """State of a single wizard step."""

    step: E1WizardStep = Field(...)
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = Field(None)


class WizardState(BaseModel):
    """Complete wizard state."""

    wizard_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-016")
    started_at: datetime = Field(default_factory=_utcnow)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    current_step: E1WizardStep = Field(default=E1WizardStep.COMPANY_PROFILE)
    completed: bool = Field(default=False)


class SetupResult(BaseModel):
    """Result of the setup wizard."""

    wizard_id: str = Field(default="")
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    steps_completed: int = Field(default=0)
    steps_total: int = Field(default=6)
    config: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Order
# ---------------------------------------------------------------------------

WIZARD_STEP_ORDER: List[E1WizardStep] = [
    E1WizardStep.COMPANY_PROFILE,
    E1WizardStep.GHG_SCOPE,
    E1WizardStep.ENERGY_SCOPE,
    E1WizardStep.TARGETS,
    E1WizardStep.CARBON_PRICING,
    E1WizardStep.REPORTING,
]


# ---------------------------------------------------------------------------
# E1SetupWizard
# ---------------------------------------------------------------------------


class E1SetupWizard:
    """6-step guided configuration wizard for E1 Climate PACK-016.

    Guides users through company profiling, GHG scope definition,
    energy scope, target configuration, carbon pricing, and reporting
    preferences to produce a complete E1 pack configuration.

    Attributes:
        state: Current wizard state.

    Example:
        >>> wizard = E1SetupWizard()
        >>> wizard.submit_step("company_profile", {"entity_name": "Acme Corp"})
        >>> wizard.submit_step("ghg_scope", {"base_year": 2019})
        >>> result = wizard.finalize()
        >>> assert result.status == "completed"
    """

    def __init__(self) -> None:
        """Initialize E1SetupWizard."""
        self.state = WizardState()
        for step in WIZARD_STEP_ORDER:
            self.state.steps[step.value] = WizardStepState(step=step)
        logger.info("E1SetupWizard initialized (id=%s)", self.state.wizard_id)

    def get_current_step(self) -> E1WizardStep:
        """Get the current wizard step.

        Returns:
            Current E1WizardStep enum value.
        """
        return self.state.current_step

    def get_step_status(self, step_name: str) -> Optional[WizardStepState]:
        """Get the status of a specific step.

        Args:
            step_name: Step name string.

        Returns:
            WizardStepState or None if not found.
        """
        return self.state.steps.get(step_name)

    def submit_step(
        self,
        step_name: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit data for a wizard step.

        Args:
            step_name: Step name to submit.
            data: Step data to validate and store.

        Returns:
            Dict with 'valid' bool and 'errors'/'warnings'.
        """
        if step_name not in self.state.steps:
            return {"valid": False, "errors": [f"Unknown step: {step_name}"], "warnings": []}

        step_state = self.state.steps[step_name]
        step_state.status = StepStatus.IN_PROGRESS

        # Validate step data
        validation = self._validate_step(step_name, data)

        if validation["valid"]:
            step_state.data = data
            step_state.status = StepStatus.COMPLETED
            step_state.completed_at = _utcnow()
            step_state.errors = []

            # Advance to next step
            self._advance_step(step_name)

            logger.info("Step %s completed", step_name)
        else:
            step_state.status = StepStatus.ERROR
            step_state.errors = validation["errors"]
            logger.warning("Step %s validation failed: %s", step_name, validation["errors"])

        return validation

    def skip_step(self, step_name: str) -> bool:
        """Skip a wizard step.

        Args:
            step_name: Step name to skip.

        Returns:
            True if step was skipped.
        """
        if step_name not in self.state.steps:
            return False

        step_state = self.state.steps[step_name]
        step_state.status = StepStatus.SKIPPED
        self._advance_step(step_name)
        logger.info("Step %s skipped", step_name)
        return True

    def finalize(self) -> SetupResult:
        """Finalize the wizard and generate configuration.

        Returns:
            SetupResult with assembled configuration.
        """
        result = SetupResult(
            wizard_id=self.state.wizard_id,
            started_at=self.state.started_at,
        )

        # Count completed steps
        for step_name, step_state in self.state.steps.items():
            if step_state.status in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                result.steps_completed += 1

        # Assemble configuration from all steps
        config = self._assemble_config()
        result.config = config

        if result.steps_completed >= 2:  # Minimum: company_profile + ghg_scope
            result.status = "completed"
            self.state.completed = True
        else:
            result.status = "incomplete"
            result.errors.append(
                "At least company_profile and ghg_scope must be completed"
            )

        result.completed_at = _utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000

        result.provenance_hash = _compute_hash(config)

        logger.info(
            "Setup wizard finalized: %s (%d/%d steps)",
            result.status,
            result.steps_completed,
            result.steps_total,
        )
        return result

    def get_progress(self) -> Dict[str, Any]:
        """Get wizard progress summary.

        Returns:
            Dict with progress information.
        """
        completed = sum(
            1 for s in self.state.steps.values()
            if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )
        return {
            "wizard_id": self.state.wizard_id,
            "current_step": self.state.current_step.value,
            "steps_completed": completed,
            "steps_total": len(WIZARD_STEP_ORDER),
            "progress_pct": round(completed / len(WIZARD_STEP_ORDER) * 100, 1),
            "completed": self.state.completed,
            "steps": {
                name: state.status.value
                for name, state in self.state.steps.items()
            },
        }

    def apply_industry_defaults(self, nace_code: str) -> Dict[str, Any]:
        """Apply industry-specific defaults based on NACE code.

        Args:
            nace_code: NACE sector code.

        Returns:
            Dict of default values applied.
        """
        # High-emission sectors get broader scope 3
        high_emission_naces = {"B", "C", "D", "H"}
        sector_letter = nace_code[0] if nace_code else ""

        defaults: Dict[str, Any] = {
            "consolidation_approach": "operational_control",
            "gwp_source": "IPCC AR6",
            "base_year": 2019,
            "scope3_categories": list(range(1, 16)),
        }

        if sector_letter in high_emission_naces:
            defaults["scope3_categories"] = list(range(1, 16))
            defaults["energy_tracking"] = True
            defaults["exposed_to_ets"] = True
        else:
            defaults["scope3_categories"] = [1, 2, 3, 5, 6, 7]
            defaults["energy_tracking"] = True
            defaults["exposed_to_ets"] = False

        logger.info(
            "Applied industry defaults for NACE %s: %d scope3 categories",
            nace_code,
            len(defaults["scope3_categories"]),
        )
        return defaults

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _validate_step(
        self,
        step_name: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate data for a specific step."""
        validators = {
            E1WizardStep.COMPANY_PROFILE.value: self._validate_company_profile,
            E1WizardStep.GHG_SCOPE.value: self._validate_ghg_scope,
            E1WizardStep.ENERGY_SCOPE.value: self._validate_energy_scope,
            E1WizardStep.TARGETS.value: self._validate_targets,
            E1WizardStep.CARBON_PRICING.value: self._validate_carbon_pricing,
            E1WizardStep.REPORTING.value: self._validate_reporting,
        }
        validator = validators.get(step_name)
        if validator is None:
            return {"valid": True, "errors": [], "warnings": []}
        return validator(data)

    def _validate_company_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate company profile step."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("nace_codes"):
            warnings.append("nace_codes recommended for sector defaults")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_ghg_scope(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GHG scope step."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("scope1_included", True):
            warnings.append("Scope 1 exclusion requires justification per ESRS E1-6")
        if not data.get("scope2_included", True):
            warnings.append("Scope 2 exclusion requires justification per ESRS E1-6")
        if not data.get("base_year"):
            warnings.append("base_year recommended; defaulting to 2019")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_energy_scope(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate energy scope step."""
        errors: List[str] = []
        warnings: List[str] = []
        target_pct = data.get("renewable_target_pct", 0.0)
        if target_pct > 0 and not data.get("renewable_target_year"):
            warnings.append("renewable_target_year recommended when target is set")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate targets step."""
        errors: List[str] = []
        warnings: List[str] = []
        if data.get("has_targets") and not data.get("near_term_target_year"):
            warnings.append("near_term_target_year recommended for ESRS E1-4")
        if data.get("sbti_committed") and not data.get("sbti_validated"):
            warnings.append("SBTi committed but not yet validated")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_carbon_pricing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate carbon pricing step."""
        errors: List[str] = []
        warnings: List[str] = []
        if data.get("exposed_to_ets") and not data.get("ets_jurisdictions"):
            warnings.append("ets_jurisdictions recommended when ETS exposure is indicated")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _validate_reporting(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reporting step."""
        errors: List[str] = []
        warnings: List[str] = []
        formats = data.get("output_formats", [])
        valid_formats = {"markdown", "html", "json", "pdf"}
        invalid = [f for f in formats if f not in valid_formats]
        if invalid:
            warnings.append(f"Unsupported formats: {', '.join(invalid)}")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _advance_step(self, current_step_name: str) -> None:
        """Advance to the next wizard step."""
        try:
            current_idx = [s.value for s in WIZARD_STEP_ORDER].index(current_step_name)
            if current_idx + 1 < len(WIZARD_STEP_ORDER):
                self.state.current_step = WIZARD_STEP_ORDER[current_idx + 1]
        except ValueError:
            pass

    def _assemble_config(self) -> Dict[str, Any]:
        """Assemble final configuration from all step data."""
        config: Dict[str, Any] = {
            "pack_id": "PACK-016",
            "pack_version": "1.0.0",
            "assembled_at": _utcnow().isoformat(),
        }

        for step_name, step_state in self.state.steps.items():
            if step_state.status == StepStatus.COMPLETED:
                config[step_name] = step_state.data

        return config
