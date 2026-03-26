# -*- coding: utf-8 -*-
"""
SetupWizard - 10-Step Enterprise Configuration Wizard for PACK-043
=====================================================================

This module implements a 10-step enterprise configuration wizard for
setting up Scope 3 Complete inventory projects including organization
profile with entity hierarchy, PACK-042 prerequisite verification,
boundary approach selection, maturity level assessment, LCA database
configuration, SBTi target configuration, supplier programme setup,
climate risk parameters, assurance level target, and initial enterprise
pipeline execution.

Wizard Steps (10):
    1.  ORG_PROFILE           -- Organization profile and entity hierarchy
    2.  PACK042_VERIFICATION  -- PACK-042 prerequisite verification
    3.  BOUNDARY_APPROACH     -- Equity/operational/financial control
    4.  MATURITY_ASSESSMENT   -- Current data maturity level
    5.  LCA_DATABASE          -- ecoinvent/GaBi database configuration
    6.  SBTI_TARGET           -- SBTi target scenario and sector
    7.  SUPPLIER_PROGRAMME    -- Supplier programme setup
    8.  CLIMATE_RISK_PARAMS   -- Carbon price scenario and risk parameters
    9.  ASSURANCE_LEVEL       -- Limited/reasonable assurance target
    10. INITIAL_EXECUTION     -- Run initial enterprise pipeline

Zero-Hallucination:
    All preset values, default configurations, and lookup tables use
    deterministic data. No LLM calls in the configuration path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-043 Scope 3 Complete
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

_MODULE_VERSION: str = "43.0.0"


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
    """Setup wizard steps in execution order."""

    ORG_PROFILE = "org_profile"
    PACK042_VERIFICATION = "pack042_verification"
    BOUNDARY_APPROACH = "boundary_approach"
    MATURITY_ASSESSMENT = "maturity_assessment"
    LCA_DATABASE = "lca_database"
    SBTI_TARGET = "sbti_target"
    SUPPLIER_PROGRAMME = "supplier_programme"
    CLIMATE_RISK_PARAMS = "climate_risk_params"
    ASSURANCE_LEVEL = "assurance_level"
    INITIAL_EXECUTION = "initial_execution"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class MaturityLevel(str, Enum):
    """Data maturity levels."""

    SCREENING = "screening"
    STARTER = "starter"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    LEADING = "leading"


class AssuranceTarget(str, Enum):
    """Target assurance level."""

    NONE = "none"
    LIMITED = "limited"
    REASONABLE = "reasonable"


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

BOUNDARY_APPROACH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "equity_share": {
        "name": "Equity Share",
        "description": "Account for emissions proportional to equity ownership",
        "common_for": "Listed companies, holding companies, JVs",
        "ghg_protocol_ref": "Chapter 3",
    },
    "operational_control": {
        "name": "Operational Control",
        "description": "Account for 100% of emissions from operations controlled",
        "common_for": "Most companies (recommended default)",
        "ghg_protocol_ref": "Chapter 3",
    },
    "financial_control": {
        "name": "Financial Control",
        "description": "Account for 100% of emissions from financially controlled operations",
        "common_for": "Companies aligning with financial reporting",
        "ghg_protocol_ref": "Chapter 3",
    },
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


class Scope3CompleteConfig(BaseModel):
    """Final pack configuration generated by the wizard."""

    config_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    entity_ids: List[str] = Field(default_factory=list)
    boundary_approach: str = Field(default="operational_control")
    maturity_level: str = Field(default="intermediate")
    lca_database: str = Field(default="ecoinvent_3.10")
    sbti_scenario: str = Field(default="1.5C")
    sbti_sector: str = Field(default="")
    supplier_programme_enabled: bool = Field(default=True)
    supplier_count: int = Field(default=0)
    carbon_price_scenario: str = Field(default="iea_nze")
    assurance_level: str = Field(default="limited")
    reporting_year: int = Field(default=2025)
    base_year: int = Field(default=2019)
    pack042_verified: bool = Field(default=False)
    initial_execution_completed: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Step Order
# ---------------------------------------------------------------------------

STEP_ORDER: List[SetupStep] = [
    SetupStep.ORG_PROFILE,
    SetupStep.PACK042_VERIFICATION,
    SetupStep.BOUNDARY_APPROACH,
    SetupStep.MATURITY_ASSESSMENT,
    SetupStep.LCA_DATABASE,
    SetupStep.SBTI_TARGET,
    SetupStep.SUPPLIER_PROGRAMME,
    SetupStep.CLIMATE_RISK_PARAMS,
    SetupStep.ASSURANCE_LEVEL,
    SetupStep.INITIAL_EXECUTION,
]

SKIPPABLE_STEPS: set = {
    SetupStep.SUPPLIER_PROGRAMME,
    SetupStep.CLIMATE_RISK_PARAMS,
    SetupStep.INITIAL_EXECUTION,
}


# ---------------------------------------------------------------------------
# SetupWizard
# ---------------------------------------------------------------------------


class SetupWizard:
    """10-step enterprise Scope 3 Complete configuration wizard.

    Guides users through complete enterprise Scope 3 setup including
    organization profile, PACK-042 verification, boundary approach,
    maturity assessment, LCA database, SBTi targets, supplier programme,
    climate risk parameters, assurance level, and initial execution.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.advance(SetupStep.ORG_PROFILE, org_data)
        >>> config = wizard.generate_config()
    """

    def __init__(self) -> None:
        """Initialize SetupWizard with 10 pending steps."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._steps: Dict[SetupStep, StepState] = {
            step: StepState(step=step) for step in STEP_ORDER
        }
        self._config: Dict[str, Any] = {}
        self.logger.info("SetupWizard initialized: %d steps", len(STEP_ORDER))

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
        self, step: SetupStep, input_data: Dict[str, Any]
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
                "Step %s validation failed: %d errors",
                step.value, len(validation.errors),
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
        self, step: SetupStep, data: Dict[str, Any]
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
                warnings.append("No reporting year, defaulting to 2025")

        elif step == SetupStep.PACK042_VERIFICATION:
            if not data.get("pack042_verified", False):
                errors.append("PACK-042 must be verified as available")

        elif step == SetupStep.BOUNDARY_APPROACH:
            approach = data.get("boundary_approach", "")
            if approach and approach not in BOUNDARY_APPROACH_DEFAULTS:
                errors.append(f"Invalid boundary approach: {approach}")

        elif step == SetupStep.MATURITY_ASSESSMENT:
            level = data.get("maturity_level", "")
            valid_levels = [m.value for m in MaturityLevel]
            if level and level not in valid_levels:
                errors.append(f"Invalid maturity level: {level}")

        elif step == SetupStep.LCA_DATABASE:
            db = data.get("lca_database", "")
            valid_dbs = ["ecoinvent_3.10", "gabi"]
            if db and db not in valid_dbs:
                warnings.append(f"Non-standard LCA database: {db}")

        elif step == SetupStep.SBTI_TARGET:
            scenario = data.get("sbti_scenario", "")
            if scenario and scenario not in ["1.5C", "well_below_2C"]:
                errors.append(f"Invalid SBTi scenario: {scenario}")

        elif step == SetupStep.ASSURANCE_LEVEL:
            level = data.get("assurance_level", "")
            valid_levels = [a.value for a in AssuranceTarget]
            if level and level not in valid_levels:
                errors.append(f"Invalid assurance level: {level}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def can_skip(self, step: SetupStep) -> bool:
        """Check if a step can be skipped."""
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

    def generate_config(self) -> Scope3CompleteConfig:
        """Generate final pack configuration from wizard state.

        Returns:
            Scope3CompleteConfig with complete configuration.
        """
        config = Scope3CompleteConfig(
            organization_name=self._config.get("organization_name", ""),
            entity_ids=self._config.get("entity_ids", []),
            boundary_approach=self._config.get("boundary_approach", "operational_control"),
            maturity_level=self._config.get("maturity_level", "intermediate"),
            lca_database=self._config.get("lca_database", "ecoinvent_3.10"),
            sbti_scenario=self._config.get("sbti_scenario", "1.5C"),
            sbti_sector=self._config.get("sbti_sector", ""),
            supplier_programme_enabled=self._config.get("supplier_programme_enabled", True),
            supplier_count=self._config.get("supplier_count", 0),
            carbon_price_scenario=self._config.get("carbon_price_scenario", "iea_nze"),
            assurance_level=self._config.get("assurance_level", "limited"),
            reporting_year=self._config.get("reporting_year", 2025),
            base_year=self._config.get("base_year", 2019),
            pack042_verified=self._config.get("pack042_verified", False),
            initial_execution_completed=self._config.get("initial_execution_completed", False),
        )
        config.provenance_hash = _compute_hash(config)

        self.logger.info(
            "Config generated: org=%s, boundary=%s, maturity=%s, sbti=%s, assurance=%s",
            config.organization_name,
            config.boundary_approach,
            config.maturity_level,
            config.sbti_scenario,
            config.assurance_level,
        )
        return config

    def get_progress(self) -> Dict[str, Any]:
        """Get overall wizard progress."""
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

    def _apply_step_data(self, step: SetupStep, data: Dict[str, Any]) -> None:
        """Apply step data to accumulated configuration."""
        if step == SetupStep.ORG_PROFILE:
            self._config["organization_name"] = data.get("organization_name", "")
            self._config["entity_ids"] = data.get("entity_ids", [])
            self._config["reporting_year"] = data.get("reporting_year", 2025)
            self._config["base_year"] = data.get("base_year", 2019)
        elif step == SetupStep.PACK042_VERIFICATION:
            self._config["pack042_verified"] = data.get("pack042_verified", False)
        elif step == SetupStep.BOUNDARY_APPROACH:
            self._config["boundary_approach"] = data.get("boundary_approach", "operational_control")
        elif step == SetupStep.MATURITY_ASSESSMENT:
            self._config["maturity_level"] = data.get("maturity_level", "intermediate")
        elif step == SetupStep.LCA_DATABASE:
            self._config["lca_database"] = data.get("lca_database", "ecoinvent_3.10")
        elif step == SetupStep.SBTI_TARGET:
            self._config["sbti_scenario"] = data.get("sbti_scenario", "1.5C")
            self._config["sbti_sector"] = data.get("sbti_sector", "")
        elif step == SetupStep.SUPPLIER_PROGRAMME:
            self._config["supplier_programme_enabled"] = data.get("supplier_programme_enabled", True)
            self._config["supplier_count"] = data.get("supplier_count", 0)
        elif step == SetupStep.CLIMATE_RISK_PARAMS:
            self._config["carbon_price_scenario"] = data.get("carbon_price_scenario", "iea_nze")
        elif step == SetupStep.ASSURANCE_LEVEL:
            self._config["assurance_level"] = data.get("assurance_level", "limited")
        elif step == SetupStep.INITIAL_EXECUTION:
            self._config["initial_execution_completed"] = data.get("execution_completed", False)
