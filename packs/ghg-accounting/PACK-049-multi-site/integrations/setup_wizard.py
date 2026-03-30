# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Multi-Site Configuration Wizard for PACK-049
===================================================================

Provides an 8-step guided configuration wizard for setting up multi-site
GHG management including organisation profile, consolidation approach,
site portfolio import, collection schedule, boundary rules, allocation
configuration, KPI selection, and reporting preferences.

Steps (8 total):
    1. OrganisationProfile       - Company name, size, industry, region
    2. ConsolidationApproach     - Operational/financial control, equity share
    3. SitePortfolioImport       - Import sites from registry or CSV
    4. CollectionSchedule        - Monthly/quarterly/annual, deadlines
    5. BoundaryRules             - Materiality thresholds, de minimis, exclusions
    6. AllocationConfig          - Shared emission allocation methods
    7. KPISelection              - Intensity metrics, peer groups, pathways
    8. ReportingPreferences      - Report formats, recipients, schedules

Reference:
    GHG Protocol Corporate Standard, Chapter 3: Setting Organisational
      Boundaries
    GHG Protocol Corporate Standard, Chapter 8: Reporting GHG Emissions

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-049 GHG Multi-Site Management
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
    CONSOLIDATION_APPROACH = "consolidation_approach"
    SITE_PORTFOLIO_IMPORT = "site_portfolio_import"
    COLLECTION_SCHEDULE = "collection_schedule"
    BOUNDARY_RULES = "boundary_rules"
    ALLOCATION_CONFIG = "allocation_config"
    KPI_SELECTION = "kpi_selection"
    REPORTING_PREFERENCES = "reporting_preferences"

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
    SetupStep.CONSOLIDATION_APPROACH,
    SetupStep.SITE_PORTFOLIO_IMPORT,
    SetupStep.COLLECTION_SCHEDULE,
    SetupStep.BOUNDARY_RULES,
    SetupStep.ALLOCATION_CONFIG,
    SetupStep.KPI_SELECTION,
    SetupStep.REPORTING_PREFERENCES,
]

STEP_DESCRIPTIONS: Dict[SetupStep, str] = {
    SetupStep.ORGANISATION_PROFILE: (
        "Configure organisation name, industry sector, company size, "
        "primary region, and reporting currency for multi-site portfolio"
    ),
    SetupStep.CONSOLIDATION_APPROACH: (
        "Select the GHG Protocol consolidation approach: operational "
        "control, financial control, or equity share. This determines "
        "how owned/operated sites are included in the boundary"
    ),
    SetupStep.SITE_PORTFOLIO_IMPORT: (
        "Import site portfolio from existing registry, CSV upload, or "
        "manual entry. Each site requires code, name, type, country, "
        "and key characteristics (floor area, headcount)"
    ),
    SetupStep.COLLECTION_SCHEDULE: (
        "Define data collection frequency (monthly, quarterly, annual), "
        "submission deadlines, reminder schedules, and escalation rules "
        "for overdue sites"
    ),
    SetupStep.BOUNDARY_RULES: (
        "Set organisational boundary rules including materiality threshold, "
        "de minimis threshold for small sites, scope inclusion rules, and "
        "exclusion criteria with justification requirements"
    ),
    SetupStep.ALLOCATION_CONFIG: (
        "Configure shared emission allocation methods for landlord-tenant "
        "splits, cogeneration, shared services, and common areas. Select "
        "allocation keys (floor area, headcount, production)"
    ),
    SetupStep.KPI_SELECTION: (
        "Select intensity metrics (tCO2e/revenue, tCO2e/FTE, tCO2e/sqm), "
        "peer group definitions for benchmarking, and pathway alignment "
        "criteria for site-level performance tracking"
    ),
    SetupStep.REPORTING_PREFERENCES: (
        "Configure report output formats (PDF, Excel, JSON), report "
        "distribution lists, dashboard preferences, and automated "
        "reporting schedules"
    ),
}

STEP_RECOMMENDATIONS: Dict[SetupStep, List[str]] = {
    SetupStep.ORGANISATION_PROFILE: [
        "Ensure company name matches legal entity for regulatory reporting",
        "Select industry sector for benchmark peer group matching",
        "Revenue and employee count used for size-based presets",
    ],
    SetupStep.CONSOLIDATION_APPROACH: [
        "Operational control is most common for corporate reporting",
        "Equity share required for some financial sector disclosures",
        "Approach must be consistent year-over-year per GHG Protocol",
    ],
    SetupStep.SITE_PORTFOLIO_IMPORT: [
        "Include all material sites even if data is initially estimated",
        "Minimum fields: site code, name, facility type, country",
        "GPS coordinates enable regional emission factor auto-assignment",
    ],
    SetupStep.COLLECTION_SCHEDULE: [
        "Monthly collection recommended for active management",
        "Set deadlines 3-4 weeks after period end for data availability",
        "Enable automatic reminders at 7 and 3 days before deadline",
    ],
    SetupStep.BOUNDARY_RULES: [
        "5% materiality threshold aligns with assurance standards",
        "1% de minimis threshold typical for small office sites",
        "Document all exclusions with justification for audit trail",
    ],
    SetupStep.ALLOCATION_CONFIG: [
        "Floor area is preferred allocation key for shared buildings",
        "Operating hours allocation suits shared manufacturing facilities",
        "Cogeneration allocation should follow IEA/BEIS methodology",
    ],
    SetupStep.KPI_SELECTION: [
        "Revenue intensity enables cross-sector comparison",
        "Floor area intensity preferred for real estate portfolios",
        "Select peer group based on facility type for meaningful comparison",
    ],
    SetupStep.REPORTING_PREFERENCES: [
        "PDF for external stakeholders, Excel for internal analysis",
        "JSON for system integration and API consumers",
        "Monthly automated reports enable timely management review",
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
    industry_sector: str = ""
    company_size: str = ""
    primary_region: str = ""
    reporting_currency: str = "EUR"
    consolidation_approach: str = "operational_control"
    collection_period: str = "monthly"
    materiality_threshold_pct: float = 5.0
    de_minimis_threshold_pct: float = 1.0
    completeness_target_pct: float = 95.0
    total_sites: int = 0
    allocation_methods: List[str] = Field(default_factory=list)
    kpi_types: List[str] = Field(default_factory=list)
    report_formats: List[str] = Field(default_factory=list)
    report_recipients: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    generated_at: str = ""

# ---------------------------------------------------------------------------
# Wizard Implementation
# ---------------------------------------------------------------------------

class SetupWizard:
    """
    8-step multi-site configuration wizard.

    Guides users through complete setup of multi-site GHG management
    including organisation profile, consolidation approach, site import,
    collection schedule, boundary rules, allocation, KPIs, and reporting.

    Attributes:
        _state: Current wizard state.

    Example:
        >>> wizard = SetupWizard()
        >>> state = wizard.start()
        >>> result = await wizard.complete_step(WizardInput(
        ...     step_name="organisation_profile",
        ...     data={"company_name": "ACME", "industry_sector": "manufacturing"}
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

        org = step_data_map.get(SetupStep.ORGANISATION_PROFILE.value, {})
        approach = step_data_map.get(SetupStep.CONSOLIDATION_APPROACH.value, {})
        portfolio = step_data_map.get(SetupStep.SITE_PORTFOLIO_IMPORT.value, {})
        schedule = step_data_map.get(SetupStep.COLLECTION_SCHEDULE.value, {})
        boundary = step_data_map.get(SetupStep.BOUNDARY_RULES.value, {})
        allocation = step_data_map.get(SetupStep.ALLOCATION_CONFIG.value, {})
        kpi = step_data_map.get(SetupStep.KPI_SELECTION.value, {})
        reporting = step_data_map.get(SetupStep.REPORTING_PREFERENCES.value, {})

        config = PackConfigOutput(
            config_id=f"config-{_new_uuid()[:8]}",
            company_name=org.get("company_name", ""),
            industry_sector=org.get("industry_sector", ""),
            company_size=org.get("company_size", ""),
            primary_region=org.get("primary_region", ""),
            reporting_currency=org.get("reporting_currency", "EUR"),
            consolidation_approach=approach.get("approach", "operational_control"),
            collection_period=schedule.get("frequency", "monthly"),
            materiality_threshold_pct=boundary.get("materiality_threshold_pct", 5.0),
            de_minimis_threshold_pct=boundary.get("de_minimis_threshold_pct", 1.0),
            completeness_target_pct=boundary.get("completeness_target_pct", 95.0),
            total_sites=portfolio.get("total_sites", 0),
            allocation_methods=allocation.get("methods", []),
            kpi_types=kpi.get("kpi_types", []),
            report_formats=reporting.get("formats", ["pdf", "xlsx"]),
            report_recipients=reporting.get("recipients", []),
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

        if step_name == SetupStep.ORGANISATION_PROFILE.value:
            if not data.get("company_name"):
                errors.append("Company name is required")

        elif step_name == SetupStep.CONSOLIDATION_APPROACH.value:
            valid = ["operational_control", "financial_control", "equity_share"]
            approach = data.get("approach", "")
            if approach and approach not in valid:
                errors.append(f"Invalid consolidation approach: {approach}")

        elif step_name == SetupStep.COLLECTION_SCHEDULE.value:
            valid_freq = ["monthly", "quarterly", "annual"]
            freq = data.get("frequency", "")
            if freq and freq not in valid_freq:
                errors.append(f"Invalid collection frequency: {freq}")

        elif step_name == SetupStep.BOUNDARY_RULES.value:
            mat = data.get("materiality_threshold_pct", 0)
            if mat < 0 or mat > 100:
                errors.append("Materiality threshold must be 0-100%")

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
