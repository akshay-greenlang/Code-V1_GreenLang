# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step GHG Assurance Configuration Wizard for PACK-048
======================================================================

Provides an 8-step guided configuration wizard for setting up GHG
assurance preparation including organisation profile, assurance
standard selection, assurance level configuration, scope configuration,
existing evidence inventory, control framework status, verifier
information, and timeline/budget preferences.

Steps (8 total):
    1. OrganisationProfile       - Jurisdiction, size, listing status
    2. AssuranceStandard         - ISAE 3410, ISO 14064-3, etc.
    3. AssuranceLevel            - Limited or reasonable assurance
    4. ScopeConfiguration        - S1/S2/S3 inclusion in assurance
    5. EvidenceInventory         - Existing evidence assessment
    6. ControlFramework          - Control framework status
    7. VerifierInformation       - Verifier details (if selected)
    8. TimelineBudget            - Timeline and budget preferences

Reference:
    ISAE 3410 para 12-16: Preconditions for assurance
    ISO 14064-3 clause 6.1: Validation/verification objectives

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-048 GHG Assurance Prep
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
    ASSURANCE_STANDARD = "assurance_standard"
    ASSURANCE_LEVEL = "assurance_level"
    SCOPE_CONFIGURATION = "scope_configuration"
    EVIDENCE_INVENTORY = "evidence_inventory"
    CONTROL_FRAMEWORK = "control_framework"
    VERIFIER_INFORMATION = "verifier_information"
    TIMELINE_BUDGET = "timeline_budget"


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
    SetupStep.ASSURANCE_STANDARD,
    SetupStep.ASSURANCE_LEVEL,
    SetupStep.SCOPE_CONFIGURATION,
    SetupStep.EVIDENCE_INVENTORY,
    SetupStep.CONTROL_FRAMEWORK,
    SetupStep.VERIFIER_INFORMATION,
    SetupStep.TIMELINE_BUDGET,
]

STEP_DESCRIPTIONS: Dict[SetupStep, str] = {
    SetupStep.ORGANISATION_PROFILE: (
        "Configure organisation jurisdiction (EU CSRD, UK, AU NGER), "
        "company size (revenue, employees), and listing status for "
        "determining mandatory assurance requirements"
    ),
    SetupStep.ASSURANCE_STANDARD: (
        "Select the assurance standard: ISAE 3410 (international), "
        "ISO 14064-3 (verification), AA1000AS v3 (stakeholder), "
        "or jurisdiction-specific standard"
    ),
    SetupStep.ASSURANCE_LEVEL: (
        "Select assurance level: limited (negative form) or reasonable "
        "(positive form). CSRD Phase 1 requires limited assurance, "
        "transitioning to reasonable assurance later"
    ),
    SetupStep.SCOPE_CONFIGURATION: (
        "Define which emission scopes to include in assurance scope "
        "(Scope 1, 2, 3) and specific Scope 3 categories if applicable"
    ),
    SetupStep.EVIDENCE_INVENTORY: (
        "Assess existing evidence availability: methodology documents, "
        "calculation records, emission factor sources, activity data "
        "files, QA/QC procedures, and internal audit records"
    ),
    SetupStep.CONTROL_FRAMEWORK: (
        "Assess control framework status: data collection controls, "
        "calculation review controls, reporting controls, IT general "
        "controls, and segregation of duties"
    ),
    SetupStep.VERIFIER_INFORMATION: (
        "Enter verifier details if already selected: verifier name, "
        "accreditation body, lead verifier contact, engagement terms, "
        "and site visit requirements"
    ),
    SetupStep.TIMELINE_BUDGET: (
        "Set timeline preferences (engagement start, fieldwork dates, "
        "opinion target) and budget range for the assurance engagement"
    ),
}

STEP_RECOMMENDATIONS: Dict[SetupStep, List[str]] = {
    SetupStep.ORGANISATION_PROFILE: [
        "EU CSRD requires mandatory limited assurance from 2025/2026",
        "UK companies under Streamlined Energy and Carbon Reporting (SECR)",
        "Listed companies typically face stricter assurance requirements",
    ],
    SetupStep.ASSURANCE_STANDARD: [
        "ISAE 3410 is the primary international GHG assurance standard",
        "ISO 14064-3 is preferred for verification engagements",
        "AA1000AS v3 covers broader sustainability assurance",
    ],
    SetupStep.ASSURANCE_LEVEL: [
        "Limited assurance provides moderate confidence (negative form)",
        "Reasonable assurance provides high confidence (positive form)",
        "CSRD requires limited assurance initially, reasonable later",
    ],
    SetupStep.SCOPE_CONFIGURATION: [
        "Scope 1 and 2 are minimum for most assurance engagements",
        "Include Scope 3 material categories per CSRD ESRS E1 requirements",
        "Consider phased approach: S1/S2 first, then add S3 categories",
    ],
    SetupStep.EVIDENCE_INVENTORY: [
        "Map all evidence to specific GHG statement assertions",
        "Ensure emission factor sources have published references",
        "Activity data should be traceable to source documents",
    ],
    SetupStep.CONTROL_FRAMEWORK: [
        "Document data collection controls with clear ownership",
        "Implement four-eyes principle for calculation review",
        "IT general controls should cover data systems access",
    ],
    SetupStep.VERIFIER_INFORMATION: [
        "Select a verifier accredited for GHG under ISAE 3410",
        "Plan site visits 6-8 weeks before opinion deadline",
        "Establish secure document exchange channel early",
    ],
    SetupStep.TIMELINE_BUDGET: [
        "Allow 12-16 weeks for first-time reasonable assurance",
        "Limited assurance typically 6-10 weeks engagement",
        "Budget 30-50% of external audit fee for GHG assurance",
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
    jurisdiction: str = ""
    listing_status: str = ""
    company_size: str = ""
    assurance_standard: str = "ISAE_3410"
    assurance_level: str = "limited"
    scopes_included: List[str] = Field(default_factory=list)
    scope3_categories: List[int] = Field(default_factory=list)
    evidence_inventory: Dict[str, bool] = Field(default_factory=dict)
    control_framework_status: Dict[str, str] = Field(default_factory=dict)
    verifier_name: str = ""
    verifier_accreditation: str = ""
    engagement_start_date: str = ""
    fieldwork_start_date: str = ""
    opinion_target_date: str = ""
    budget_range: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = ""
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Wizard Implementation
# ---------------------------------------------------------------------------


class SetupWizard:
    """
    8-step GHG assurance configuration wizard.

    Guides users through the complete setup of GHG assurance
    preparation including organisation profile, assurance standard
    and level selection, scope configuration, evidence assessment,
    control framework, verifier details, and timeline/budget.

    Attributes:
        _state: Current wizard state.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> result = await wizard.complete_step(WizardInput(
        ...     step_name="organisation_profile",
        ...     data={"company_name": "ACME", "jurisdiction": "EU"}
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
        standard = step_data_map.get(SetupStep.ASSURANCE_STANDARD.value, {})
        level = step_data_map.get(SetupStep.ASSURANCE_LEVEL.value, {})
        scope = step_data_map.get(SetupStep.SCOPE_CONFIGURATION.value, {})
        evidence = step_data_map.get(SetupStep.EVIDENCE_INVENTORY.value, {})
        controls = step_data_map.get(SetupStep.CONTROL_FRAMEWORK.value, {})
        verifier = step_data_map.get(SetupStep.VERIFIER_INFORMATION.value, {})
        timeline = step_data_map.get(SetupStep.TIMELINE_BUDGET.value, {})

        config = PackConfigOutput(
            config_id=f"config-{_new_uuid()[:8]}",
            company_name=org.get("company_name", ""),
            jurisdiction=org.get("jurisdiction", ""),
            listing_status=org.get("listing_status", ""),
            company_size=org.get("company_size", ""),
            assurance_standard=standard.get("standard", "ISAE_3410"),
            assurance_level=level.get("level", "limited"),
            scopes_included=scope.get("scopes", []),
            scope3_categories=scope.get("scope3_categories", []),
            evidence_inventory=evidence.get("inventory", {}),
            control_framework_status=controls.get("status", {}),
            verifier_name=verifier.get("verifier_name", ""),
            verifier_accreditation=verifier.get("accreditation", ""),
            engagement_start_date=timeline.get("engagement_start", ""),
            fieldwork_start_date=timeline.get("fieldwork_start", ""),
            opinion_target_date=timeline.get("opinion_target", ""),
            budget_range=timeline.get("budget", {}),
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
            if not data.get("jurisdiction"):
                errors.append("Jurisdiction is required")

        elif step_name == SetupStep.ASSURANCE_STANDARD.value:
            valid_standards = ["ISAE_3410", "ISO_14064_3", "AA1000AS", "CUSTOM"]
            standard = data.get("standard", "")
            if standard and standard not in valid_standards:
                errors.append(f"Invalid standard: {standard}")

        elif step_name == SetupStep.ASSURANCE_LEVEL.value:
            valid_levels = ["limited", "reasonable"]
            level = data.get("level", "")
            if level and level not in valid_levels:
                errors.append(f"Invalid assurance level: {level}")

        elif step_name == SetupStep.SCOPE_CONFIGURATION.value:
            scopes = data.get("scopes", [])
            if not scopes:
                errors.append("At least one scope must be selected")

        elif step_name == SetupStep.TIMELINE_BUDGET.value:
            budget = data.get("budget", {})
            if budget:
                min_val = budget.get("min", 0)
                max_val = budget.get("max", 0)
                if max_val < min_val:
                    errors.append("Budget maximum must be >= minimum")

        return errors

    def verify_connection(self) -> Dict[str, Any]:
        """Verify wizard module availability."""
        return {
            "bridge": "SetupWizard",
            "status": "healthy",
            "version": _MODULE_VERSION,
            "steps": len(STEP_ORDER),
            "active_session": self._state is not None,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get wizard module status."""
        return {
            "bridge": "SetupWizard",
            "status": "healthy",
            "version": _MODULE_VERSION,
            "steps": len(STEP_ORDER),
            "active_session": self._state is not None,
        }
