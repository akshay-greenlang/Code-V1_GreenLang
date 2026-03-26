# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Scope 3 Inventory Configuration Wizard for PACK-042
==========================================================================

This module implements an 8-step configuration wizard for setting up
Scope 3 GHG inventory projects. It guides users through organizational
profile, sector selection, category relevance screening, methodology
tier selection, data source configuration, framework target selection,
supplier engagement preferences, and initial screening execution.

Wizard Steps (8):
    1. ORG_PROFILE              -- Organization name, sector, revenue, employees
    2. SECTOR_SELECTION         -- NAICS/ISIC sector, industry vertical
    3. CATEGORY_RELEVANCE       -- Which of 15 categories are relevant
    4. METHODOLOGY_TIER         -- Spend-based / average / supplier-specific
    5. DATA_SOURCES             -- ERP, surveys, questionnaires, estimates
    6. FRAMEWORK_TARGETS        -- ESRS E1, CDP, SBTi, GHG Protocol
    7. SUPPLIER_ENGAGEMENT      -- Engagement strategy and thresholds
    8. INITIAL_SCREENING        -- Run initial Scope 3 screening

Zero-Hallucination:
    All preset values, default configurations, and sector lookups use
    deterministic lookup tables. No LLM calls in the configuration path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-042 Scope 3 Starter
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


class SetupStep(str, Enum):
    """Names of setup wizard steps in execution order."""

    ORG_PROFILE = "org_profile"
    SECTOR_SELECTION = "sector_selection"
    CATEGORY_RELEVANCE = "category_relevance"
    METHODOLOGY_TIER = "methodology_tier"
    DATA_SOURCES = "data_sources"
    FRAMEWORK_TARGETS = "framework_targets"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    INITIAL_SCREENING = "initial_screening"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class MethodologyTier(str, Enum):
    """Scope 3 calculation methodology tiers."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    HYBRID = "hybrid"
    SUPPLIER_SPECIFIC = "supplier_specific"


class EngagementLevel(str, Enum):
    """Supplier engagement intensity levels."""

    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    COMPREHENSIVE = "comprehensive"


# ---------------------------------------------------------------------------
# Scope 3 Category Definitions
# ---------------------------------------------------------------------------

SCOPE3_CATEGORIES: Dict[int, Dict[str, str]] = {
    1: {"name": "Purchased Goods & Services", "direction": "upstream"},
    2: {"name": "Capital Goods", "direction": "upstream"},
    3: {"name": "Fuel & Energy Activities", "direction": "upstream"},
    4: {"name": "Upstream Transportation & Distribution", "direction": "upstream"},
    5: {"name": "Waste Generated in Operations", "direction": "upstream"},
    6: {"name": "Business Travel", "direction": "upstream"},
    7: {"name": "Employee Commuting", "direction": "upstream"},
    8: {"name": "Upstream Leased Assets", "direction": "upstream"},
    9: {"name": "Downstream Transportation & Distribution", "direction": "downstream"},
    10: {"name": "Processing of Sold Products", "direction": "downstream"},
    11: {"name": "Use of Sold Products", "direction": "downstream"},
    12: {"name": "End-of-Life Treatment of Sold Products", "direction": "downstream"},
    13: {"name": "Downstream Leased Assets", "direction": "downstream"},
    14: {"name": "Franchises", "direction": "downstream"},
    15: {"name": "Investments", "direction": "downstream"},
}

# Typical category relevance by sector (NAICS 2-digit)
SECTOR_CATEGORY_DEFAULTS: Dict[str, List[int]] = {
    "31-33": [1, 2, 3, 4, 5, 6, 7, 9, 11, 12],  # Manufacturing
    "44-45": [1, 4, 5, 6, 7, 9, 12],              # Retail
    "51": [1, 2, 5, 6, 7, 8],                      # Information
    "52": [1, 5, 6, 7, 8, 15],                     # Finance/Insurance
    "54": [1, 5, 6, 7],                             # Professional Services
    "72": [1, 3, 4, 5, 6, 7],                      # Accommodation/Food
    "48-49": [1, 3, 4, 5, 6, 7, 9],               # Transportation
    "22": [1, 2, 3, 5, 6, 7],                      # Utilities
    "23": [1, 2, 4, 5, 6, 7],                      # Construction
    "62": [1, 2, 5, 6, 7],                          # Healthcare
    "default": [1, 2, 3, 4, 5, 6, 7],              # Default: upstream categories
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class StepState(BaseModel):
    """State of a single wizard step."""

    step: SetupStep = Field(...)
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = Field(None)


class WizardState(BaseModel):
    """Complete wizard state."""

    wizard_id: str = Field(default_factory=_new_uuid)
    current_step: Optional[SetupStep] = Field(None)
    completed_steps: List[SetupStep] = Field(default_factory=list)
    skipped_steps: List[SetupStep] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class ValidationResult(BaseModel):
    """Validation result for a wizard step."""

    valid: bool = Field(default=True)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class Scope3PackConfig(BaseModel):
    """Final pack configuration generated by the wizard."""

    config_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    sector_naics: str = Field(default="")
    sector_name: str = Field(default="")
    annual_revenue_usd: float = Field(default=0.0)
    employee_count: int = Field(default=0)
    relevant_categories: List[int] = Field(default_factory=list)
    methodology_tier: str = Field(default="spend_based")
    data_sources: List[str] = Field(default_factory=list)
    target_frameworks: List[str] = Field(default_factory=list)
    supplier_engagement_level: str = Field(default="basic")
    supplier_engagement_threshold_pct: float = Field(default=80.0)
    reporting_year: int = Field(default=2025)
    base_year: int = Field(default=2019)
    screening_completed: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Step Order and Skippable Steps
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupStep] = [
    SetupStep.ORG_PROFILE,
    SetupStep.SECTOR_SELECTION,
    SetupStep.CATEGORY_RELEVANCE,
    SetupStep.METHODOLOGY_TIER,
    SetupStep.DATA_SOURCES,
    SetupStep.FRAMEWORK_TARGETS,
    SetupStep.SUPPLIER_ENGAGEMENT,
    SetupStep.INITIAL_SCREENING,
]

SKIPPABLE_STEPS: set = {
    SetupStep.SUPPLIER_ENGAGEMENT,
    SetupStep.INITIAL_SCREENING,
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """8-step Scope 3 inventory configuration wizard.

    Guides users through complete Scope 3 inventory setup including
    organizational profile, sector selection, category relevance screening,
    methodology tier selection, data source configuration, framework
    target selection, supplier engagement preferences, and initial
    screening execution.

    Attributes:
        _steps: Current step states.
        _config: Accumulated configuration.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.advance(SetupStep.ORG_PROFILE, org_data)
        >>> config = wizard.generate_config(state)
    """

    def __init__(self) -> None:
        """Initialize SetupWizard with 8 pending steps."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._steps: Dict[SetupStep, StepState] = {
            step: StepState(step=step) for step in STEP_ORDER
        }
        self._config: Dict[str, Any] = {}
        self.logger.info("SetupWizard initialized: %d steps", len(STEP_ORDER))

    # -------------------------------------------------------------------------
    # Wizard Flow
    # -------------------------------------------------------------------------

    def start(self) -> WizardState:
        """Start the setup wizard.

        Returns:
            WizardState with first step active.
        """
        self._steps = {step: StepState(step=step) for step in STEP_ORDER}
        self._config = {}

        state = WizardState(
            current_step=STEP_ORDER[0],
            configuration=self._config,
        )
        self.logger.info("Wizard started: first step=%s", STEP_ORDER[0].value)
        return state

    def advance(
        self,
        step: SetupStep,
        input_data: Dict[str, Any],
    ) -> WizardState:
        """Advance the wizard by completing a step.

        Args:
            step: Step to complete.
            input_data: Data for the step.

        Returns:
            Updated WizardState.
        """
        validation = self.validate_step(step, input_data)
        step_state = self._steps[step]

        if not validation.valid:
            step_state.status = StepStatus.IN_PROGRESS
            step_state.validation_errors = validation.errors
            self.logger.warning(
                "Step %s validation failed: %d errors", step.value, len(validation.errors)
            )
        else:
            step_state.status = StepStatus.COMPLETED
            step_state.data = input_data
            step_state.validation_errors = []
            step_state.completed_at = _utcnow()
            self._apply_step_data(step, input_data)
            self.logger.info("Step %s completed", step.value)

        return self._build_state()

    def validate_step(
        self,
        step: SetupStep,
        data: Dict[str, Any],
    ) -> ValidationResult:
        """Validate data for a wizard step.

        Args:
            step: Step being validated.
            data: Data to validate.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if step == SetupStep.ORG_PROFILE:
            if not data.get("organization_name"):
                errors.append("Organization name is required")
            if not data.get("reporting_year"):
                warnings.append("No reporting year specified, defaulting to 2025")

        elif step == SetupStep.SECTOR_SELECTION:
            sector = data.get("sector_naics", "")
            if not sector:
                errors.append("NAICS sector code is required")
            elif sector not in SECTOR_CATEGORY_DEFAULTS and sector != "default":
                warnings.append(f"Sector '{sector}' not in preset list, using defaults")

        elif step == SetupStep.CATEGORY_RELEVANCE:
            categories = data.get("relevant_categories", [])
            if not categories:
                errors.append("At least one Scope 3 category must be selected")
            else:
                invalid = [c for c in categories if c not in SCOPE3_CATEGORIES]
                if invalid:
                    errors.append(f"Invalid category numbers: {invalid}")

        elif step == SetupStep.METHODOLOGY_TIER:
            tier = data.get("methodology_tier", "")
            valid_tiers = [t.value for t in MethodologyTier]
            if tier and tier not in valid_tiers:
                errors.append(f"Invalid methodology tier: {tier}. Valid: {valid_tiers}")

        elif step == SetupStep.DATA_SOURCES:
            if not data.get("data_sources"):
                warnings.append("No data sources configured, manual entry will be required")

        elif step == SetupStep.FRAMEWORK_TARGETS:
            if not data.get("frameworks"):
                warnings.append("No compliance frameworks selected")

        elif step == SetupStep.SUPPLIER_ENGAGEMENT:
            level = data.get("engagement_level", "")
            valid_levels = [l.value for l in EngagementLevel]
            if level and level not in valid_levels:
                errors.append(f"Invalid engagement level: {level}")
            threshold = data.get("threshold_pct", 80.0)
            if threshold < 50.0 or threshold > 100.0:
                errors.append(f"Engagement threshold must be 50-100%, got {threshold}%")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def can_skip(self, step: SetupStep) -> bool:
        """Check if a step can be skipped.

        Args:
            step: Step to check.

        Returns:
            True if the step is skippable.
        """
        return step in SKIPPABLE_STEPS

    def skip_step(self, step: SetupStep) -> WizardState:
        """Skip a wizard step.

        Args:
            step: Step to skip.

        Returns:
            Updated WizardState.

        Raises:
            ValueError: If step is not skippable.
        """
        if not self.can_skip(step):
            raise ValueError(f"Step '{step.value}' cannot be skipped")

        self._steps[step].status = StepStatus.SKIPPED
        self.logger.info("Step %s skipped", step.value)
        return self._build_state()

    def get_category_defaults(self, sector_naics: str) -> List[int]:
        """Get default relevant categories for a sector.

        Args:
            sector_naics: NAICS 2-digit sector code.

        Returns:
            List of default relevant Scope 3 category numbers.
        """
        return SECTOR_CATEGORY_DEFAULTS.get(
            sector_naics,
            SECTOR_CATEGORY_DEFAULTS["default"],
        )

    def generate_config(
        self,
        state: Optional[WizardState] = None,
    ) -> Scope3PackConfig:
        """Generate final pack configuration from wizard state.

        Args:
            state: Wizard state. Uses internal state if None.

        Returns:
            Scope3PackConfig with complete configuration.
        """
        start_time = time.monotonic()

        # Determine relevant categories from config or defaults
        relevant_cats = self._config.get("relevant_categories", [])
        if not relevant_cats:
            sector = self._config.get("sector_naics", "default")
            relevant_cats = self.get_category_defaults(sector)

        config = Scope3PackConfig(
            organization_name=self._config.get("organization_name", ""),
            sector_naics=self._config.get("sector_naics", ""),
            sector_name=self._config.get("sector_name", ""),
            annual_revenue_usd=self._config.get("annual_revenue_usd", 0.0),
            employee_count=self._config.get("employee_count", 0),
            relevant_categories=relevant_cats,
            methodology_tier=self._config.get("methodology_tier", "spend_based"),
            data_sources=self._config.get("data_sources", []),
            target_frameworks=self._config.get("frameworks", ["ghg_protocol"]),
            supplier_engagement_level=self._config.get("engagement_level", "basic"),
            supplier_engagement_threshold_pct=self._config.get("threshold_pct", 80.0),
            reporting_year=self._config.get("reporting_year", 2025),
            base_year=self._config.get("base_year", 2019),
            screening_completed=self._config.get("screening_completed", False),
        )
        config.provenance_hash = _compute_hash(config)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self.logger.info(
            "Config generated: org=%s, sector=%s, categories=%d, tier=%s (%.1fms)",
            config.organization_name,
            config.sector_naics,
            len(config.relevant_categories),
            config.methodology_tier,
            elapsed_ms,
        )
        return config

    def get_progress(self) -> Dict[str, Any]:
        """Get overall wizard progress.

        Returns:
            Dict with progress information.
        """
        completed = sum(1 for s in self._steps.values() if s.status == StepStatus.COMPLETED)
        skipped = sum(1 for s in self._steps.values() if s.status == StepStatus.SKIPPED)
        total = len(STEP_ORDER)
        pct = ((completed + skipped) / total * 100.0) if total > 0 else 0.0

        return {
            "completed": completed,
            "skipped": skipped,
            "pending": total - completed - skipped,
            "total": total,
            "progress_pct": round(pct, 1),
            "current_step": self._get_current_step(),
        }

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

    def _build_state(self) -> WizardState:
        """Build current wizard state."""
        completed = [s.step for s in self._steps.values() if s.status == StepStatus.COMPLETED]
        skipped = [s.step for s in self._steps.values() if s.status == StepStatus.SKIPPED]
        total = len(STEP_ORDER)
        pct = ((len(completed) + len(skipped)) / total * 100.0) if total > 0 else 0.0

        current = None
        for step in STEP_ORDER:
            if self._steps[step].status in (StepStatus.PENDING, StepStatus.IN_PROGRESS):
                current = step
                break

        return WizardState(
            current_step=current,
            completed_steps=completed,
            skipped_steps=skipped,
            configuration=dict(self._config),
            progress_pct=round(pct, 1),
        )

    def _apply_step_data(
        self, step: SetupStep, data: Dict[str, Any]
    ) -> None:
        """Apply step data to accumulated configuration.

        Args:
            step: Completed step.
            data: Step data to apply.
        """
        if step == SetupStep.ORG_PROFILE:
            self._config["organization_name"] = data.get("organization_name", "")
            self._config["annual_revenue_usd"] = data.get("annual_revenue_usd", 0.0)
            self._config["employee_count"] = data.get("employee_count", 0)
            self._config["reporting_year"] = data.get("reporting_year", 2025)
            self._config["base_year"] = data.get("base_year", 2019)

        elif step == SetupStep.SECTOR_SELECTION:
            self._config["sector_naics"] = data.get("sector_naics", "")
            self._config["sector_name"] = data.get("sector_name", "")

        elif step == SetupStep.CATEGORY_RELEVANCE:
            self._config["relevant_categories"] = data.get("relevant_categories", [])

        elif step == SetupStep.METHODOLOGY_TIER:
            self._config["methodology_tier"] = data.get("methodology_tier", "spend_based")

        elif step == SetupStep.DATA_SOURCES:
            self._config["data_sources"] = data.get("data_sources", [])

        elif step == SetupStep.FRAMEWORK_TARGETS:
            self._config["frameworks"] = data.get("frameworks", ["ghg_protocol"])

        elif step == SetupStep.SUPPLIER_ENGAGEMENT:
            self._config["engagement_level"] = data.get("engagement_level", "basic")
            self._config["threshold_pct"] = data.get("threshold_pct", 80.0)

        elif step == SetupStep.INITIAL_SCREENING:
            self._config["screening_completed"] = data.get("screening_completed", False)
            self._config["screening_results"] = data.get("screening_results", {})
