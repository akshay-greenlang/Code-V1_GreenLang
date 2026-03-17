# -*- coding: utf-8 -*-
"""
DMASetupWizard - 6-Step Guided Configuration Wizard for DMA PACK-015
=======================================================================

This module implements a 6-step configuration wizard for the Double
Materiality Assessment Pack. It guides users through company profiling,
scope definition, stakeholder configuration, scoring methodology selection,
threshold configuration, and reporting preferences.

Wizard Steps (6):
    1. company_profile       -- Name, sector, size, NACE codes, headquarters
    2. scope_definition      -- ESRS topics to assess, value chain boundaries
    3. stakeholder_config    -- Stakeholder categories, engagement methods
    4. scoring_methodology   -- Geometric mean, weighted sum, maximum, etc.
    5. threshold_config      -- Industry defaults or custom thresholds
    6. reporting_preferences -- Format, language, detail level, assurance

Each step is validated individually. The wizard generates a complete DMAConfig
from the assembled inputs with SHA-256 provenance tracking.

Architecture:
    User Input --> DMASetupWizard --> Step Validation
                        |                  |
                        v                  v
    Step Handlers <-- WizardState    Error Collection
                        |                  |
                        v                  v
    DMAConfig <-- Provenance Hash <-- SetupResult

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-015 Double Materiality Assessment
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


class DMAWizardStep(str, Enum):
    """Names of DMA wizard steps in execution order."""

    COMPANY_PROFILE = "company_profile"
    SCOPE_DEFINITION = "scope_definition"
    STAKEHOLDER_CONFIG = "stakeholder_config"
    SCORING_METHODOLOGY = "scoring_methodology"
    THRESHOLD_CONFIG = "threshold_config"
    REPORTING_PREFERENCES = "reporting_preferences"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScoringMethod(str, Enum):
    """Available scoring methodologies."""

    GEOMETRIC_MEAN = "geometric_mean"
    WEIGHTED_SUM = "weighted_sum"
    MAXIMUM = "maximum"
    ARITHMETIC_MEAN = "arithmetic_mean"


class EngagementMethod(str, Enum):
    """Stakeholder engagement methods."""

    SURVEY = "survey"
    INTERVIEW = "interview"
    WORKSHOP = "workshop"
    FOCUS_GROUP = "focus_group"
    WRITTEN_CONSULTATION = "written_consultation"
    ADVISORY_PANEL = "advisory_panel"


class ValueChainBoundary(str, Enum):
    """Value chain boundary options for DMA scope."""

    OWN_OPERATIONS = "own_operations"
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    FULL_VALUE_CHAIN = "full_value_chain"


# ---------------------------------------------------------------------------
# Data Models - Step Inputs
# ---------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """Company profile from step 1."""

    company_name: str = Field(..., min_length=1, max_length=255)
    nace_codes: List[str] = Field(
        default_factory=lambda: ["C25.1"],
        description="NACE code(s) for sector classification",
    )
    employee_count: int = Field(default=500, ge=1)
    annual_revenue_eur: float = Field(default=100_000_000.0, ge=0)
    headquarters_country: str = Field(default="DE")
    is_listed: bool = Field(default=False)
    balance_sheet_total_eur: Optional[float] = Field(None, ge=0)
    subsidiary_count: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2025, ge=2024, le=2030)


class ScopeDefinition(BaseModel):
    """Scope definition from step 2."""

    esrs_topics_in_scope: List[str] = Field(
        default_factory=lambda: ["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"],
        description="ESRS topics to include in DMA",
    )
    value_chain_boundary: ValueChainBoundary = Field(
        default=ValueChainBoundary.FULL_VALUE_CHAIN,
    )
    include_upstream: bool = Field(default=True)
    include_downstream: bool = Field(default=True)
    geographic_scope: List[str] = Field(
        default_factory=lambda: ["EU"],
        description="Geographic regions in scope",
    )
    time_horizons: Dict[str, str] = Field(
        default_factory=lambda: {
            "short_term": "0-1 years",
            "medium_term": "1-5 years",
            "long_term": "5-10+ years",
        },
    )


class StakeholderConfig(BaseModel):
    """Stakeholder configuration from step 3."""

    stakeholder_categories: List[str] = Field(
        default_factory=lambda: [
            "investors",
            "employees",
            "customers",
            "suppliers",
            "regulators",
            "civil_society",
        ],
    )
    engagement_methods: List[EngagementMethod] = Field(
        default_factory=lambda: [EngagementMethod.SURVEY, EngagementMethod.WORKSHOP],
    )
    weighting_approach: str = Field(
        default="equal",
        description="equal, expertise_based, or impact_based",
    )
    min_stakeholders_per_category: int = Field(default=5, ge=1, le=100)
    target_response_rate_pct: float = Field(default=60.0, ge=10.0, le=100.0)


class ScoringMethodologyConfig(BaseModel):
    """Scoring methodology from step 4."""

    scoring_method: ScoringMethod = Field(default=ScoringMethod.GEOMETRIC_MEAN)
    impact_dimensions: List[str] = Field(
        default_factory=lambda: ["scale", "scope", "irremediability"],
        description="Dimensions for impact severity scoring",
    )
    financial_dimensions: List[str] = Field(
        default_factory=lambda: ["magnitude", "likelihood"],
        description="Dimensions for financial materiality scoring",
    )
    scale_min: float = Field(default=1.0, ge=1.0)
    scale_max: float = Field(default=5.0, le=10.0)
    use_likelihood_for_impact: bool = Field(default=True)


class ThresholdConfig(BaseModel):
    """Threshold configuration from step 5."""

    use_industry_defaults: bool = Field(default=True)
    impact_threshold: float = Field(
        default=3.0, ge=1.0, le=5.0,
        description="Minimum score for impact materiality",
    )
    financial_threshold: float = Field(
        default=3.0, ge=1.0, le=5.0,
        description="Minimum score for financial materiality",
    )
    topic_specific_thresholds: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Override thresholds per ESRS topic",
    )
    auto_material_topics: List[str] = Field(
        default_factory=list,
        description="Topics presumed material regardless of score",
    )


class ReportingPreferences(BaseModel):
    """Reporting preferences from step 6."""

    report_format: str = Field(default="PDF", description="PDF, XHTML, or DOCX")
    report_language: str = Field(default="en", description="ISO 639-1 language code")
    detail_level: str = Field(
        default="standard",
        description="summary, standard, or detailed",
    )
    include_methodology_appendix: bool = Field(default=True)
    include_stakeholder_details: bool = Field(default=True)
    include_scoring_matrices: bool = Field(default=True)
    assurance_level: str = Field(default="limited", description="limited or reasonable")
    first_reporting_year: int = Field(default=2025, ge=2024, le=2030)


# ---------------------------------------------------------------------------
# Wizard State Models
# ---------------------------------------------------------------------------


class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: DMAWizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)


class WizardState(BaseModel):
    """Complete state of the DMA setup wizard."""

    wizard_id: str = Field(default="")
    current_step: DMAWizardStep = Field(default=DMAWizardStep.COMPANY_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    company_profile: Optional[CompanyProfile] = Field(None)
    scope_definition: Optional[ScopeDefinition] = Field(None)
    stakeholder_config: Optional[StakeholderConfig] = Field(None)
    scoring_methodology: Optional[ScoringMethodologyConfig] = Field(None)
    threshold_config: Optional[ThresholdConfig] = Field(None)
    reporting_preferences: Optional[ReportingPreferences] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class SetupResult(BaseModel):
    """Final setup result with generated DMAConfig."""

    result_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field(default="")
    nace_codes: List[str] = Field(default_factory=list)
    esrs_topics_in_scope: List[str] = Field(default_factory=list)
    scoring_method: str = Field(default="geometric_mean")
    impact_threshold: float = Field(default=3.0)
    financial_threshold: float = Field(default=3.0)
    stakeholder_categories: List[str] = Field(default_factory=list)
    value_chain_boundary: str = Field(default="full_value_chain")
    report_format: str = Field(default="PDF")
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=6)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[DMAWizardStep] = [
    DMAWizardStep.COMPANY_PROFILE,
    DMAWizardStep.SCOPE_DEFINITION,
    DMAWizardStep.STAKEHOLDER_CONFIG,
    DMAWizardStep.SCORING_METHODOLOGY,
    DMAWizardStep.THRESHOLD_CONFIG,
    DMAWizardStep.REPORTING_PREFERENCES,
]

STEP_DISPLAY_NAMES: Dict[DMAWizardStep, str] = {
    DMAWizardStep.COMPANY_PROFILE: "Company Profile",
    DMAWizardStep.SCOPE_DEFINITION: "Scope Definition",
    DMAWizardStep.STAKEHOLDER_CONFIG: "Stakeholder Configuration",
    DMAWizardStep.SCORING_METHODOLOGY: "Scoring Methodology",
    DMAWizardStep.THRESHOLD_CONFIG: "Threshold Configuration",
    DMAWizardStep.REPORTING_PREFERENCES: "Reporting Preferences",
}

# Industry default thresholds per sector
INDUSTRY_DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "A": {"impact": 2.5, "financial": 3.0},
    "B": {"impact": 2.5, "financial": 2.5},
    "C": {"impact": 3.0, "financial": 3.0},
    "D": {"impact": 2.5, "financial": 2.5},
    "F": {"impact": 3.0, "financial": 3.0},
    "G": {"impact": 3.0, "financial": 3.0},
    "H": {"impact": 2.5, "financial": 3.0},
    "I": {"impact": 3.0, "financial": 3.0},
    "J": {"impact": 3.0, "financial": 3.0},
    "K": {"impact": 3.0, "financial": 2.5},
    "L": {"impact": 3.0, "financial": 3.0},
    "Q": {"impact": 2.5, "financial": 3.0},
}


# ---------------------------------------------------------------------------
# DMASetupWizard
# ---------------------------------------------------------------------------


class DMASetupWizard:
    """6-step guided configuration wizard for DMA PACK-015.

    Guides users through the full DMA setup process with step validation,
    industry defaults, and configurable scoring methodology.

    Example:
        >>> wizard = DMASetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("company_profile", {...})
        >>> result = wizard.run_demo()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DMA Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            DMAWizardStep.COMPANY_PROFILE: self._handle_company_profile,
            DMAWizardStep.SCOPE_DEFINITION: self._handle_scope_definition,
            DMAWizardStep.STAKEHOLDER_CONFIG: self._handle_stakeholder_config,
            DMAWizardStep.SCORING_METHODOLOGY: self._handle_scoring_methodology,
            DMAWizardStep.THRESHOLD_CONFIG: self._handle_threshold_config,
            DMAWizardStep.REPORTING_PREFERENCES: self._handle_reporting_preferences,
        }
        self.logger.info("DMASetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session.

        Returns:
            Initialized WizardState with all steps in PENDING status.
        """
        wizard_id = _compute_hash(f"dma-wizard:{_utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(
            wizard_id=wizard_id,
            current_step=STEP_ORDER[0],
            steps=steps,
        )
        self.logger.info("DMA wizard started: %s", wizard_id)
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """Complete a wizard step with provided data.

        Args:
            step_name: Step name to complete.
            data: Step configuration data.

        Returns:
            Updated WizardState.

        Raises:
            RuntimeError: If wizard has not been started.
            ValueError: If step name is unknown.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = DMAWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in DMAWizardStep]
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
        """Execute a pre-configured demo setup for a manufacturing company.

        Returns:
            SetupResult with complete DMA configuration.
        """
        self.start()

        demo_steps = {
            "company_profile": {
                "company_name": "Demo Manufacturing GmbH",
                "nace_codes": ["C25.1", "C28.2"],
                "employee_count": 2500,
                "annual_revenue_eur": 500_000_000.0,
                "headquarters_country": "DE",
                "is_listed": True,
                "reporting_year": 2025,
            },
            "scope_definition": {
                "esrs_topics_in_scope": ["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"],
                "value_chain_boundary": "full_value_chain",
                "include_upstream": True,
                "include_downstream": True,
                "geographic_scope": ["EU", "APAC"],
            },
            "stakeholder_config": {
                "stakeholder_categories": [
                    "investors", "employees", "customers",
                    "suppliers", "regulators", "civil_society",
                ],
                "engagement_methods": ["survey", "workshop", "interview"],
                "weighting_approach": "expertise_based",
                "min_stakeholders_per_category": 10,
                "target_response_rate_pct": 70.0,
            },
            "scoring_methodology": {
                "scoring_method": "geometric_mean",
                "impact_dimensions": ["scale", "scope", "irremediability"],
                "financial_dimensions": ["magnitude", "likelihood"],
                "scale_min": 1.0,
                "scale_max": 5.0,
                "use_likelihood_for_impact": True,
            },
            "threshold_config": {
                "use_industry_defaults": True,
                "impact_threshold": 3.0,
                "financial_threshold": 3.0,
                "auto_material_topics": ["E1"],
            },
            "reporting_preferences": {
                "report_format": "PDF",
                "report_language": "en",
                "detail_level": "standard",
                "include_methodology_appendix": True,
                "include_stakeholder_details": True,
                "include_scoring_matrices": True,
                "assurance_level": "limited",
                "first_reporting_year": 2025,
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_industry_defaults(self, nace_section: str) -> Optional[Dict[str, float]]:
        """Get industry default thresholds for a NACE section.

        Args:
            nace_section: NACE section letter (e.g., 'C', 'K').

        Returns:
            Dict with impact and financial thresholds, or None.
        """
        return INDUSTRY_DEFAULT_THRESHOLDS.get(nace_section)

    # ---- Step Handlers ----

    def _handle_company_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle company profile step."""
        errors: List[str] = []
        try:
            profile = CompanyProfile(**data)
            if self._state:
                self._state.company_profile = profile
        except Exception as exc:
            errors.append(f"Invalid company profile: {exc}")
        return errors

    def _handle_scope_definition(self, data: Dict[str, Any]) -> List[str]:
        """Handle scope definition step."""
        errors: List[str] = []
        try:
            # Convert string boundary to enum if needed
            if "value_chain_boundary" in data and isinstance(data["value_chain_boundary"], str):
                data["value_chain_boundary"] = ValueChainBoundary(data["value_chain_boundary"])
            scope = ScopeDefinition(**data)
            # Validate topic codes
            valid_topics = {"E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"}
            invalid = set(scope.esrs_topics_in_scope) - valid_topics
            if invalid:
                errors.append(f"Invalid ESRS topics: {sorted(invalid)}. Valid: {sorted(valid_topics)}")
            if self._state and not errors:
                self._state.scope_definition = scope
        except Exception as exc:
            errors.append(f"Invalid scope definition: {exc}")
        return errors

    def _handle_stakeholder_config(self, data: Dict[str, Any]) -> List[str]:
        """Handle stakeholder configuration step."""
        errors: List[str] = []
        try:
            # Convert string methods to enums if needed
            if "engagement_methods" in data:
                methods = []
                for m in data["engagement_methods"]:
                    if isinstance(m, str):
                        methods.append(EngagementMethod(m))
                    else:
                        methods.append(m)
                data["engagement_methods"] = methods
            config = StakeholderConfig(**data)
            if not config.stakeholder_categories:
                errors.append("At least one stakeholder category is required")
            if self._state and not errors:
                self._state.stakeholder_config = config
        except Exception as exc:
            errors.append(f"Invalid stakeholder config: {exc}")
        return errors

    def _handle_scoring_methodology(self, data: Dict[str, Any]) -> List[str]:
        """Handle scoring methodology step."""
        errors: List[str] = []
        try:
            if "scoring_method" in data and isinstance(data["scoring_method"], str):
                data["scoring_method"] = ScoringMethod(data["scoring_method"])
            config = ScoringMethodologyConfig(**data)
            if config.scale_min >= config.scale_max:
                errors.append("scale_min must be less than scale_max")
            if self._state and not errors:
                self._state.scoring_methodology = config
        except Exception as exc:
            errors.append(f"Invalid scoring methodology: {exc}")
        return errors

    def _handle_threshold_config(self, data: Dict[str, Any]) -> List[str]:
        """Handle threshold configuration step with industry defaults."""
        errors: List[str] = []
        try:
            config = ThresholdConfig(**data)

            # Apply industry defaults if requested
            if config.use_industry_defaults and self._state and self._state.company_profile:
                nace_codes = self._state.company_profile.nace_codes
                if nace_codes:
                    section = nace_codes[0][0] if nace_codes[0] else ""
                    defaults = INDUSTRY_DEFAULT_THRESHOLDS.get(section)
                    if defaults:
                        config.impact_threshold = defaults["impact"]
                        config.financial_threshold = defaults["financial"]

            if self._state and not errors:
                self._state.threshold_config = config
        except Exception as exc:
            errors.append(f"Invalid threshold config: {exc}")
        return errors

    def _handle_reporting_preferences(self, data: Dict[str, Any]) -> List[str]:
        """Handle reporting preferences step."""
        errors: List[str] = []
        try:
            prefs = ReportingPreferences(**data)
            valid_formats = {"PDF", "XHTML", "DOCX"}
            if prefs.report_format not in valid_formats:
                errors.append(f"Invalid format '{prefs.report_format}'. Valid: {sorted(valid_formats)}")
            if self._state and not errors:
                self._state.reporting_preferences = prefs
        except Exception as exc:
            errors.append(f"Invalid reporting preferences: {exc}")
        return errors

    # ---- Navigation ----

    def _advance_step(self, current: DMAWizardStep) -> None:
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

        config_hash = _compute_hash({
            "company": self._state.company_profile.company_name if self._state.company_profile else "",
            "nace_codes": self._state.company_profile.nace_codes if self._state.company_profile else [],
            "scoring_method": (
                self._state.scoring_methodology.scoring_method.value
                if self._state.scoring_methodology else "geometric_mean"
            ),
            "topics": (
                self._state.scope_definition.esrs_topics_in_scope
                if self._state.scope_definition else []
            ),
        })

        result = SetupResult(
            company_name=(
                self._state.company_profile.company_name
                if self._state.company_profile else ""
            ),
            nace_codes=(
                self._state.company_profile.nace_codes
                if self._state.company_profile else []
            ),
            esrs_topics_in_scope=(
                self._state.scope_definition.esrs_topics_in_scope
                if self._state.scope_definition else []
            ),
            scoring_method=(
                self._state.scoring_methodology.scoring_method.value
                if self._state.scoring_methodology else "geometric_mean"
            ),
            impact_threshold=(
                self._state.threshold_config.impact_threshold
                if self._state.threshold_config else 3.0
            ),
            financial_threshold=(
                self._state.threshold_config.financial_threshold
                if self._state.threshold_config else 3.0
            ),
            stakeholder_categories=(
                self._state.stakeholder_config.stakeholder_categories
                if self._state.stakeholder_config else []
            ),
            value_chain_boundary=(
                self._state.scope_definition.value_chain_boundary.value
                if self._state.scope_definition else "full_value_chain"
            ),
            report_format=(
                self._state.reporting_preferences.report_format
                if self._state.reporting_preferences else "PDF"
            ),
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
