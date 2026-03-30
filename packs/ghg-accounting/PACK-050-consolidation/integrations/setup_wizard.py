# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Consolidation Configuration Wizard for PACK-050
=======================================================================

Provides an 8-step guided configuration wizard for setting up corporate
GHG consolidation including corporate structure import, entity registry
setup, ownership chain configuration, boundary approach selection,
data collection setup, elimination rules, reporting framework selection,
and alert preferences.

Steps (8 total):
    1. CorporateStructureImport  - Import corporate hierarchy from CSV/ERP
    2. EntityRegistrySetup       - Define entity attributes and classifications
    3. OwnershipChainConfig      - Configure ownership chains and equity shares
    4. BoundaryApproachSelection - Select consolidation approach per GHG Protocol
    5. DataCollectionSetup       - Configure collection schedules and deadlines
    6. EliminationRules          - Define intercompany elimination rules
    7. ReportingFrameworkSelect  - Select reporting frameworks (GHG Protocol, CSRD)
    8. AlertPreferences          - Configure alert thresholds and channels

Reference:
    GHG Protocol Corporate Standard, Chapter 3: Setting Organisational
      Boundaries
    GHG Protocol Corporate Standard, Chapter 8: Reporting GHG Emissions
    IFRS S2: Climate-related Disclosures

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-050 GHG Consolidation
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

    CORPORATE_STRUCTURE_IMPORT = "corporate_structure_import"
    ENTITY_REGISTRY_SETUP = "entity_registry_setup"
    OWNERSHIP_CHAIN_CONFIG = "ownership_chain_config"
    BOUNDARY_APPROACH_SELECTION = "boundary_approach_selection"
    DATA_COLLECTION_SETUP = "data_collection_setup"
    ELIMINATION_RULES = "elimination_rules"
    REPORTING_FRAMEWORK_SELECT = "reporting_framework_select"
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
    SetupStep.CORPORATE_STRUCTURE_IMPORT,
    SetupStep.ENTITY_REGISTRY_SETUP,
    SetupStep.OWNERSHIP_CHAIN_CONFIG,
    SetupStep.BOUNDARY_APPROACH_SELECTION,
    SetupStep.DATA_COLLECTION_SETUP,
    SetupStep.ELIMINATION_RULES,
    SetupStep.REPORTING_FRAMEWORK_SELECT,
    SetupStep.ALERT_PREFERENCES,
]

STEP_DESCRIPTIONS: Dict[SetupStep, str] = {
    SetupStep.CORPORATE_STRUCTURE_IMPORT: (
        "Import corporate group hierarchy from CSV upload, ERP system, "
        "or manual entry. Defines the parent-subsidiary tree with legal "
        "entity identifiers, jurisdictions, and entity types"
    ),
    SetupStep.ENTITY_REGISTRY_SETUP: (
        "Configure entity attributes including legal name, registration "
        "number, jurisdiction, entity type (subsidiary, JV, associate), "
        "activity status, and reporting currency"
    ),
    SetupStep.OWNERSHIP_CHAIN_CONFIG: (
        "Define direct and indirect ownership chains with equity share "
        "percentages, control flags (operational, financial), and joint "
        "venture arrangements per GHG Protocol Chapter 3"
    ),
    SetupStep.BOUNDARY_APPROACH_SELECTION: (
        "Select the GHG Protocol consolidation approach: equity share, "
        "operational control, or financial control. Set materiality and "
        "de minimis thresholds for entity inclusion"
    ),
    SetupStep.DATA_COLLECTION_SETUP: (
        "Configure data collection schedules, submission deadlines, "
        "entity-level data contacts, automated reminders, and escalation "
        "rules for overdue entities"
    ),
    SetupStep.ELIMINATION_RULES: (
        "Define intercompany elimination rules for internal transfers, "
        "shared services, intra-group energy, and procurement to prevent "
        "double-counting in consolidated totals"
    ),
    SetupStep.REPORTING_FRAMEWORK_SELECT: (
        "Select target reporting frameworks: GHG Protocol Corporate "
        "Standard, CSRD/ESRS E1, CDP Climate Change, IFRS S2, "
        "ISO 14064-1. Configure report formats and distribution"
    ),
    SetupStep.ALERT_PREFERENCES: (
        "Configure alert thresholds for submission deadlines, entity "
        "completeness gaps, consolidation variance, boundary changes, "
        "M&A events, and assurance deadlines. Select delivery channels"
    ),
}

STEP_RECOMMENDATIONS: Dict[SetupStep, List[str]] = {
    SetupStep.CORPORATE_STRUCTURE_IMPORT: [
        "Use legal entity registry as the source of truth for hierarchy",
        "Include dormant entities for completeness documentation",
        "Map entity IDs to existing ERP or financial system codes",
    ],
    SetupStep.ENTITY_REGISTRY_SETUP: [
        "Use official legal names matching company registration",
        "Include jurisdiction for regional emission factor assignment",
        "Classify entity types for consolidation approach application",
    ],
    SetupStep.OWNERSHIP_CHAIN_CONFIG: [
        "Document both direct and indirect equity holdings",
        "Mark joint ventures and associates explicitly for boundary rules",
        "Validate ownership chain totals do not exceed 100% per entity",
    ],
    SetupStep.BOUNDARY_APPROACH_SELECTION: [
        "Equity share is required for some financial sector disclosures",
        "Operational control is most common for corporate reporting",
        "Approach must be consistent year-over-year per GHG Protocol",
    ],
    SetupStep.DATA_COLLECTION_SETUP: [
        "Set deadlines 4-6 weeks after period end for entity data",
        "Assign entity-level data contacts for accountability",
        "Enable automatic reminders at 14, 7, and 3 days before deadline",
    ],
    SetupStep.ELIMINATION_RULES: [
        "Focus eliminations on material intercompany transactions",
        "Document all elimination rules with policy references",
        "Shared services should use consistent allocation keys",
    ],
    SetupStep.REPORTING_FRAMEWORK_SELECT: [
        "GHG Protocol is the baseline for most frameworks",
        "CSRD requires double materiality assessment",
        "CDP scoring benefits from complete Scope 3 reporting",
    ],
    SetupStep.ALERT_PREFERENCES: [
        "Critical alerts for M&A events require immediate attention",
        "Set consolidation variance threshold at 5% for materiality",
        "Enable Slack/Teams integration for real-time notifications",
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
    current_step: str = SetupStep.CORPORATE_STRUCTURE_IMPORT.value
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
    group_name: str = ""
    parent_entity_name: str = ""
    total_entities: int = 0
    consolidation_approach: str = "operational_control"
    equity_threshold_pct: float = 20.0
    materiality_threshold_pct: float = 5.0
    de_minimis_threshold_pct: float = 1.0
    completeness_target_pct: float = 95.0
    collection_period: str = "annual"
    elimination_rules_count: int = 0
    reporting_frameworks: List[str] = Field(default_factory=list)
    report_formats: List[str] = Field(default_factory=list)
    alert_channels: List[str] = Field(default_factory=list)
    alert_recipients: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    generated_at: str = ""

# ---------------------------------------------------------------------------
# Wizard Implementation
# ---------------------------------------------------------------------------

class SetupWizard:
    """
    8-step consolidation configuration wizard.

    Guides users through complete setup of corporate GHG consolidation
    including corporate structure, entity registry, ownership chains,
    boundary approach, data collection, elimination rules, reporting
    frameworks, and alert preferences.

    Attributes:
        _state: Current wizard state.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> result = await wizard.complete_step(WizardInput(
        ...     step_name="corporate_structure_import",
        ...     data={"group_name": "ACME Corp", "source": "csv"}
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

    async def execute_step(self, wizard_input: WizardInput) -> WizardResult:
        """Alias for complete_step for interface consistency.

        Args:
            wizard_input: WizardInput with step name and data.

        Returns:
            WizardResult with validation status.
        """
        return await self.complete_step(wizard_input)

    def get_state(self) -> Optional[WizardState]:
        """Get current wizard state."""
        return self._state

    def get_step_info(self, step_name: str) -> Dict[str, Any]:
        """Get information about a specific step."""
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

        structure = step_data_map.get(SetupStep.CORPORATE_STRUCTURE_IMPORT.value, {})
        registry = step_data_map.get(SetupStep.ENTITY_REGISTRY_SETUP.value, {})
        ownership = step_data_map.get(SetupStep.OWNERSHIP_CHAIN_CONFIG.value, {})
        boundary = step_data_map.get(SetupStep.BOUNDARY_APPROACH_SELECTION.value, {})
        collection = step_data_map.get(SetupStep.DATA_COLLECTION_SETUP.value, {})
        elimination = step_data_map.get(SetupStep.ELIMINATION_RULES.value, {})
        reporting = step_data_map.get(SetupStep.REPORTING_FRAMEWORK_SELECT.value, {})
        alerts = step_data_map.get(SetupStep.ALERT_PREFERENCES.value, {})

        config = PackConfigOutput(
            config_id=f"config-{_new_uuid()[:8]}",
            group_name=structure.get("group_name", ""),
            parent_entity_name=structure.get("parent_entity_name", ""),
            total_entities=registry.get("total_entities", 0),
            consolidation_approach=boundary.get("approach", "operational_control"),
            equity_threshold_pct=boundary.get("equity_threshold_pct", 20.0),
            materiality_threshold_pct=boundary.get("materiality_threshold_pct", 5.0),
            de_minimis_threshold_pct=boundary.get("de_minimis_threshold_pct", 1.0),
            completeness_target_pct=boundary.get("completeness_target_pct", 95.0),
            collection_period=collection.get("frequency", "annual"),
            elimination_rules_count=elimination.get("rules_count", 0),
            reporting_frameworks=reporting.get("frameworks", []),
            report_formats=reporting.get("formats", ["pdf", "xlsx"]),
            alert_channels=alerts.get("channels", ["email", "in_app"]),
            alert_recipients=alerts.get("recipients", []),
            generated_at=utcnow().isoformat(),
        )
        config.provenance_hash = _compute_hash(config.model_dump())

        logger.info("Pack configuration generated: %s", config.config_id)
        return config

    def _validate_step(
        self, step_name: str, data: Dict[str, Any]
    ) -> List[str]:
        """Validate data for a specific step."""
        errors: List[str] = []

        if step_name == SetupStep.CORPORATE_STRUCTURE_IMPORT.value:
            if not data.get("group_name"):
                errors.append("Group name is required")

        elif step_name == SetupStep.BOUNDARY_APPROACH_SELECTION.value:
            valid = ["operational_control", "financial_control", "equity_share"]
            approach = data.get("approach", "")
            if approach and approach not in valid:
                errors.append(f"Invalid consolidation approach: {approach}")

        elif step_name == SetupStep.OWNERSHIP_CHAIN_CONFIG.value:
            chains = data.get("chains", [])
            for chain in chains:
                pct = chain.get("equity_pct", 0)
                if pct < 0 or pct > 100:
                    errors.append(f"Invalid equity percentage: {pct}")

        elif step_name == SetupStep.DATA_COLLECTION_SETUP.value:
            valid_freq = ["monthly", "quarterly", "semi_annual", "annual"]
            freq = data.get("frequency", "")
            if freq and freq not in valid_freq:
                errors.append(f"Invalid collection frequency: {freq}")

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
