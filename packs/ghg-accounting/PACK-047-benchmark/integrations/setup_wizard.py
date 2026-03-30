# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step GHG Benchmark Configuration Wizard for PACK-047
======================================================================

Provides an 8-step guided configuration wizard for setting up GHG
emissions benchmarking including organisation profile, scope
configuration, peer group preferences, pathway selection, external
data sources, portfolio setup, disclosure frameworks, and alert
preferences.

Steps (8 total):
    1. OrganisationProfile       - Sector, size, geography
    2. ScopeConfiguration        - Which scopes to benchmark
    3. PeerGroupPreferences      - Matching criteria, min peers
    4. PathwaySelection          - Which science-based pathways
    5. ExternalDataSources       - CDP, TPI, GRESB, etc.
    6. PortfolioSetup            - If financial institution
    7. DisclosureFrameworks      - ESRS, CDP, SFDR, etc.
    8. AlertPreferences          - Thresholds, notification channels

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-047 GHG Emissions Benchmark
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
# Enumerations
# ---------------------------------------------------------------------------

class SetupStep(str, Enum):
    """Wizard setup steps."""

    ORGANISATION_PROFILE = "organisation_profile"
    SCOPE_CONFIGURATION = "scope_configuration"
    PEER_GROUP_PREFERENCES = "peer_group_preferences"
    PATHWAY_SELECTION = "pathway_selection"
    EXTERNAL_DATA_SOURCES = "external_data_sources"
    PORTFOLIO_SETUP = "portfolio_setup"
    DISCLOSURE_FRAMEWORKS = "disclosure_frameworks"
    ALERT_PREFERENCES = "alert_preferences"

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
    SetupStep.SCOPE_CONFIGURATION,
    SetupStep.PEER_GROUP_PREFERENCES,
    SetupStep.PATHWAY_SELECTION,
    SetupStep.EXTERNAL_DATA_SOURCES,
    SetupStep.PORTFOLIO_SETUP,
    SetupStep.DISCLOSURE_FRAMEWORKS,
    SetupStep.ALERT_PREFERENCES,
]

STEP_DESCRIPTIONS: Dict[SetupStep, str] = {
    SetupStep.ORGANISATION_PROFILE: (
        "Configure organisation sector (GICS/NACE), company size "
        "(revenue, employees), and geographic footprint for peer matching"
    ),
    SetupStep.SCOPE_CONFIGURATION: (
        "Define which emission scopes to benchmark (Scope 1, 2, 3) and "
        "Scope 2 method (location-based, market-based, or both)"
    ),
    SetupStep.PEER_GROUP_PREFERENCES: (
        "Set peer group matching criteria (sector, geography, size), "
        "minimum number of peers, and peer data quality thresholds"
    ),
    SetupStep.PATHWAY_SELECTION: (
        "Select science-based pathways for alignment analysis "
        "(SBTi SDA, IEA NZE, OECM, CRREM) and ambition level"
    ),
    SetupStep.EXTERNAL_DATA_SOURCES: (
        "Enable external benchmark data sources "
        "(CDP, TPI, GRESB, CRREM, ISS ESG) and configure API credentials"
    ),
    SetupStep.PORTFOLIO_SETUP: (
        "Configure portfolio alignment settings for financial institutions "
        "(PCAF, WACI, ITR) with asset class and weighting method"
    ),
    SetupStep.DISCLOSURE_FRAMEWORKS: (
        "Map benchmark results to disclosure frameworks "
        "(ESRS E1, CDP Climate, SFDR PAI, TCFD)"
    ),
    SetupStep.ALERT_PREFERENCES: (
        "Configure alert thresholds (percentile change, pathway deviation), "
        "notification channels (email, webhook, Slack), and cooldown periods"
    ),
}

STEP_RECOMMENDATIONS: Dict[SetupStep, List[str]] = {
    SetupStep.ORGANISATION_PROFILE: [
        "Use GICS sector classification for widest peer comparability",
        "Include sub-industry for more precise peer matching",
        "Specify geographic regions for location-adjusted benchmarks",
    ],
    SetupStep.SCOPE_CONFIGURATION: [
        "Scope 1+2 is the minimum for most benchmark comparisons",
        "Include Scope 3 for comprehensive value chain benchmarking",
        "Use both location-based and market-based Scope 2 for dual reporting",
    ],
    SetupStep.PEER_GROUP_PREFERENCES: [
        "Minimum 10 peers recommended for statistically valid percentiles",
        "Weight peers by revenue similarity for better comparability",
        "Use CDP as primary peer data source for widest coverage",
    ],
    SetupStep.PATHWAY_SELECTION: [
        "SBTi SDA pathways provide sector-specific convergence targets",
        "IEA NZE pathway provides global economy-wide reference",
        "CRREM pathways are specific to real estate building types",
    ],
    SetupStep.EXTERNAL_DATA_SOURCES: [
        "CDP provides the largest public benchmark dataset",
        "TPI covers high-emitting sectors with detailed pathway data",
        "GRESB is essential for real estate and infrastructure",
    ],
    SetupStep.PORTFOLIO_SETUP: [
        "PCAF is the standard for financial institution carbon accounting",
        "WACI (Weighted Average Carbon Intensity) is used by TCFD",
        "ITR (Implied Temperature Rise) maps portfolio to temperature outcome",
    ],
    SetupStep.DISCLOSURE_FRAMEWORKS: [
        "ESRS E1-6 requires GHG intensity per net revenue benchmarking",
        "CDP Climate Change questionnaire uses sector-relative scoring",
        "SFDR PAI requires portfolio carbon footprint benchmarks",
    ],
    SetupStep.ALERT_PREFERENCES: [
        "Set percentile rank change threshold to 10 for meaningful alerts",
        "Enable pathway deviation alerts for SBTi-committed companies",
        "Use webhook channel for automated dashboard integration",
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
    sub_industry: str = ""
    geography: str = ""
    company_size: str = ""
    scopes_included: List[str] = Field(default_factory=list)
    scope2_method: str = "location_based"
    peer_group: Dict[str, Any] = Field(default_factory=dict)
    min_peers: int = 10
    pathways: List[str] = Field(default_factory=list)
    ambition_level: str = "1.5c"
    external_sources: List[str] = Field(default_factory=list)
    is_financial_institution: bool = False
    portfolio_config: Dict[str, Any] = Field(default_factory=dict)
    disclosure_frameworks: List[str] = Field(default_factory=list)
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    alert_channels: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    generated_at: str = ""

# ---------------------------------------------------------------------------
# Wizard Implementation
# ---------------------------------------------------------------------------

class SetupWizard:
    """
    8-step GHG benchmark configuration wizard.

    Guides users through the complete setup of GHG emissions
    benchmarking including organisation profile, scope selection,
    peer group configuration, pathway selection, and alert setup.

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
            started_at=utcnow().isoformat(),
            last_updated=utcnow().isoformat(),
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
                step_data.completed_at = utcnow().isoformat()
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
        self._state.last_updated = utcnow().isoformat()

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
        scope = step_data_map.get(SetupStep.SCOPE_CONFIGURATION.value, {})
        peer = step_data_map.get(SetupStep.PEER_GROUP_PREFERENCES.value, {})
        pathway = step_data_map.get(SetupStep.PATHWAY_SELECTION.value, {})
        sources = step_data_map.get(SetupStep.EXTERNAL_DATA_SOURCES.value, {})
        portfolio = step_data_map.get(SetupStep.PORTFOLIO_SETUP.value, {})
        frameworks = step_data_map.get(SetupStep.DISCLOSURE_FRAMEWORKS.value, {})
        alerts = step_data_map.get(SetupStep.ALERT_PREFERENCES.value, {})

        config = PackConfigOutput(
            config_id=f"config-{_new_uuid()[:8]}",
            company_name=org.get("company_name", ""),
            sector=org.get("sector", ""),
            sub_industry=org.get("sub_industry", ""),
            geography=org.get("geography", ""),
            company_size=org.get("company_size", ""),
            scopes_included=scope.get("scopes", []),
            scope2_method=scope.get("scope2_method", "location_based"),
            peer_group=peer,
            min_peers=peer.get("min_peers", 10),
            pathways=pathway.get("pathways", []),
            ambition_level=pathway.get("ambition_level", "1.5c"),
            external_sources=sources.get("sources", []),
            is_financial_institution=portfolio.get("is_financial_institution", False),
            portfolio_config=portfolio,
            disclosure_frameworks=frameworks.get("frameworks", []),
            alert_thresholds=alerts.get("thresholds", {}),
            alert_channels=alerts.get("channels", []),
            generated_at=utcnow().isoformat(),
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

        elif step_name == SetupStep.SCOPE_CONFIGURATION.value:
            scopes = data.get("scopes", [])
            if not scopes:
                errors.append("At least one scope must be selected")

        elif step_name == SetupStep.PEER_GROUP_PREFERENCES.value:
            min_peers = data.get("min_peers", 0)
            if min_peers < 1:
                errors.append("Minimum peer count must be at least 1")

        elif step_name == SetupStep.PATHWAY_SELECTION.value:
            pathways = data.get("pathways", [])
            if not pathways:
                errors.append("At least one pathway must be selected")

        elif step_name == SetupStep.EXTERNAL_DATA_SOURCES.value:
            sources = data.get("sources", [])
            if not sources:
                errors.append("At least one external data source must be enabled")

        elif step_name == SetupStep.ALERT_PREFERENCES.value:
            thresholds = data.get("thresholds", {})
            for key, value in thresholds.items():
                if value < 0:
                    errors.append(f"Threshold '{key}' must be non-negative")

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
