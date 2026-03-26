# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Base Year Configuration Wizard for PACK-045
==================================================================

Provides an 8-step guided configuration wizard for setting up base
year management including policy configuration, base year selection,
scope definition, threshold setting, target registration, notification
preferences, integration setup, and validation.

Steps (8 total):
    1. Organization Profile       - Company details, sector, reporting boundary
    2. Base Year Selection        - Candidate evaluation, year selection
    3. Scope Definition           - Include/exclude scopes and categories
    4. Policy Configuration       - Recalculation policy, thresholds
    5. Target Registration        - Register emission reduction targets
    6. Integration Setup          - Connect to PACK-041/042/043/044, ERP
    7. Notification Preferences   - Channels, recipients, frequencies
    8. Validation & Activation    - Validate config, activate base year

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-045 Base Year Management
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SetupStep(str, Enum):
    """Wizard setup steps."""
    ORGANIZATION_PROFILE = "organization_profile"
    BASE_YEAR_SELECTION = "base_year_selection"
    SCOPE_DEFINITION = "scope_definition"
    POLICY_CONFIGURATION = "policy_configuration"
    TARGET_REGISTRATION = "target_registration"
    INTEGRATION_SETUP = "integration_setup"
    NOTIFICATION_PREFERENCES = "notification_preferences"
    VALIDATION_ACTIVATION = "validation_activation"


class StepStatus(str, Enum):
    """Step completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERROR = "error"


STEP_ORDER: List[SetupStep] = [
    SetupStep.ORGANIZATION_PROFILE,
    SetupStep.BASE_YEAR_SELECTION,
    SetupStep.SCOPE_DEFINITION,
    SetupStep.POLICY_CONFIGURATION,
    SetupStep.TARGET_REGISTRATION,
    SetupStep.INTEGRATION_SETUP,
    SetupStep.NOTIFICATION_PREFERENCES,
    SetupStep.VALIDATION_ACTIVATION,
]

STEP_DESCRIPTIONS: Dict[SetupStep, str] = {
    SetupStep.ORGANIZATION_PROFILE: "Configure organization details, sector classification, and reporting boundary",
    SetupStep.BASE_YEAR_SELECTION: "Evaluate candidate years and select the base year",
    SetupStep.SCOPE_DEFINITION: "Define which scopes and categories to include",
    SetupStep.POLICY_CONFIGURATION: "Set recalculation policy, significance thresholds, and review frequency",
    SetupStep.TARGET_REGISTRATION: "Register emission reduction targets and SBTi commitments",
    SetupStep.INTEGRATION_SETUP: "Connect to GHG accounting packs and ERP systems",
    SetupStep.NOTIFICATION_PREFERENCES: "Configure notification channels and recipients",
    SetupStep.VALIDATION_ACTIVATION: "Validate configuration and activate base year management",
}


class StepData(BaseModel):
    """Data collected for a wizard step."""
    step: str
    status: str = StepStatus.NOT_STARTED.value
    data: Dict[str, Any] = Field(default_factory=dict)
    completed_at: Optional[str] = None
    validated: bool = False
    errors: List[str] = Field(default_factory=list)


class WizardState(BaseModel):
    """Current state of the setup wizard."""
    wizard_id: str = ""
    current_step: str = SetupStep.ORGANIZATION_PROFILE.value
    steps: List[StepData] = Field(default_factory=list)
    overall_progress_pct: float = 0.0
    started_at: str = ""
    last_updated: str = ""
    is_complete: bool = False


class PackConfig(BaseModel):
    """Final pack configuration produced by the wizard."""
    config_id: str = ""
    company_name: str = ""
    base_year: str = ""
    scopes_included: List[str] = Field(default_factory=list)
    significance_threshold_pct: float = 5.0
    recalculation_policy: str = "automatic"
    review_frequency: str = "annual"
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    integrations: List[str] = Field(default_factory=list)
    notification_channels: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    activated_at: str = ""


class SetupWizard:
    """
    8-step base year configuration wizard.

    Guides users through the complete setup of base year management
    including organization profile, year selection, policy configuration,
    target registration, and integration setup.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = await wizard.complete_step("organization_profile", org_data)
    """

    def __init__(self) -> None:
        """Initialize SetupWizard."""
        self._state: Optional[WizardState] = None
        logger.info("SetupWizard initialized: %d steps", len(STEP_ORDER))

    def start(self) -> WizardState:
        """Start a new wizard session."""
        wizard_id = f"wizard-{int(time.time())}"
        steps = [
            StepData(step=step.value)
            for step in STEP_ORDER
        ]
        self._state = WizardState(
            wizard_id=wizard_id,
            steps=steps,
            started_at=_utcnow().isoformat(),
            last_updated=_utcnow().isoformat(),
        )
        logger.info("Wizard session started: %s", wizard_id)
        return self._state

    async def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """
        Complete a wizard step with provided data.

        Args:
            step_name: Name of the step to complete.
            data: Data collected for this step.

        Returns:
            Updated WizardState.

        Raises:
            ValueError: If wizard not started or step invalid.
        """
        if self._state is None:
            raise ValueError("Wizard not started. Call start() first.")

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
        completed = sum(1 for s in self._state.steps if s.status == StepStatus.COMPLETED.value)
        self._state.overall_progress_pct = (completed / len(STEP_ORDER)) * 100
        self._state.last_updated = _utcnow().isoformat()

        # Advance current step
        for step in STEP_ORDER:
            step_data = next((s for s in self._state.steps if s.step == step.value), None)
            if step_data and step_data.status != StepStatus.COMPLETED.value:
                self._state.current_step = step.value
                break

        if completed == len(STEP_ORDER):
            self._state.is_complete = True

        logger.info("Step %s completed. Progress: %.0f%%", step_name, self._state.overall_progress_pct)
        return self._state

    def get_state(self) -> Optional[WizardState]:
        """Get current wizard state."""
        return self._state

    def get_step_info(self, step_name: str) -> Dict[str, str]:
        """Get information about a specific step."""
        try:
            step = SetupStep(step_name)
        except ValueError:
            return {"error": f"Unknown step: {step_name}"}
        return {
            "name": step.value,
            "description": STEP_DESCRIPTIONS.get(step, ""),
            "order": str(STEP_ORDER.index(step) + 1),
        }

    async def generate_config(self) -> PackConfig:
        """Generate final pack configuration from wizard data."""
        if self._state is None:
            raise ValueError("Wizard not started")

        step_data_map: Dict[str, Dict[str, Any]] = {}
        for s in self._state.steps:
            step_data_map[s.step] = s.data

        org = step_data_map.get(SetupStep.ORGANIZATION_PROFILE.value, {})
        by = step_data_map.get(SetupStep.BASE_YEAR_SELECTION.value, {})
        scope = step_data_map.get(SetupStep.SCOPE_DEFINITION.value, {})
        policy = step_data_map.get(SetupStep.POLICY_CONFIGURATION.value, {})
        targets = step_data_map.get(SetupStep.TARGET_REGISTRATION.value, {})
        integrations = step_data_map.get(SetupStep.INTEGRATION_SETUP.value, {})
        notif = step_data_map.get(SetupStep.NOTIFICATION_PREFERENCES.value, {})

        config = PackConfig(
            config_id=f"config-{int(time.time())}",
            company_name=org.get("company_name", ""),
            base_year=by.get("selected_year", ""),
            scopes_included=scope.get("scopes", []),
            significance_threshold_pct=policy.get("threshold_pct", 5.0),
            recalculation_policy=policy.get("policy", "automatic"),
            review_frequency=policy.get("frequency", "annual"),
            targets=targets.get("targets", []),
            integrations=integrations.get("enabled", []),
            notification_channels=notif.get("channels", []),
            activated_at=_utcnow().isoformat(),
        )
        config.provenance_hash = _compute_hash(config.model_dump())

        logger.info("Pack configuration generated: %s", config.config_id)
        return config

    def _validate_step(self, step_name: str, data: Dict[str, Any]) -> List[str]:
        """Validate data for a specific step."""
        errors: List[str] = []

        if step_name == SetupStep.ORGANIZATION_PROFILE.value:
            if not data.get("company_name"):
                errors.append("Company name is required")

        elif step_name == SetupStep.BASE_YEAR_SELECTION.value:
            if not data.get("selected_year"):
                errors.append("Base year selection is required")

        elif step_name == SetupStep.SCOPE_DEFINITION.value:
            if not data.get("scopes"):
                errors.append("At least one scope must be selected")

        elif step_name == SetupStep.POLICY_CONFIGURATION.value:
            threshold = data.get("threshold_pct", 0)
            if threshold <= 0 or threshold > 50:
                errors.append("Significance threshold must be between 0.1% and 50%")

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
