# -*- coding: utf-8 -*-
"""
SMESetupWizard - 5-Step Interactive Configuration for PACK-026
=================================================================

This module implements a 5-step configuration wizard tailored for
SMEs setting up the Net Zero Pack. Simplified compared to PACK-021's
6-step wizard, focusing on what matters most for small businesses.

Wizard Steps (5):
    1. organization_profile    -- Name, sector, country, size, revenue
    2. data_quality_tier       -- Bronze/Silver/Gold tier selection with
                                  explanation of trade-offs
    3. accounting_connection   -- Xero/QuickBooks/Sage connection setup
    4. grant_preferences       -- Country/region for grant matching
    5. certification_pathway   -- SME Climate Hub / B Corp / Carbon Trust

Database Schema:
    Initializes PACK-026 specific tables and configuration.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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

class SMEWizardStep(str, Enum):
    ORGANIZATION_PROFILE = "organization_profile"
    DATA_QUALITY_TIER = "data_quality_tier"
    ACCOUNTING_CONNECTION = "accounting_connection"
    GRANT_PREFERENCES = "grant_preferences"
    CERTIFICATION_PATHWAY = "certification_pathway"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class SMESize(str, Enum):
    MICRO = "micro"       # 1-9 employees
    SMALL = "small"       # 10-49 employees
    MEDIUM = "medium"     # 50-249 employees

class DataQualityTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

class AccountingSoftware(str, Enum):
    XERO = "xero"
    QUICKBOOKS = "quickbooks"
    SAGE = "sage"
    MANUAL = "manual"
    NONE = "none"

class CertificationPathway(str, Enum):
    SME_CLIMATE_HUB = "sme_climate_hub"
    B_CORP = "b_corp"
    CARBON_TRUST = "carbon_trust"
    ISO_14001 = "iso_14001"
    CLIMATE_ACTIVE = "climate_active"
    NONE = "none"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SMEOrganizationProfile(BaseModel):
    """Organization profile from step 1."""

    organization_name: str = Field(..., min_length=1, max_length=255)
    sector: str = Field(default="general")
    sub_sector: str = Field(default="")
    country: str = Field(default="GB")
    region: str = Field(default="")
    employee_count: int = Field(default=25, ge=1, le=250)
    annual_revenue_eur: float = Field(default=2_000_000.0, ge=0)
    size: SMESize = Field(default=SMESize.SMALL)
    fiscal_year_end: str = Field(default="03-31")
    website: str = Field(default="")

class DataQualitySelection(BaseModel):
    """Data quality tier selection from step 2."""

    tier: DataQualityTier = Field(default=DataQualityTier.BRONZE)
    scope3_approach: str = Field(default="spend_based")
    use_industry_defaults: bool = Field(default=True)
    estimated_accuracy_pct: float = Field(default=70.0)

class AccountingConnectionSetup(BaseModel):
    """Accounting software connection from step 3."""

    software: AccountingSoftware = Field(default=AccountingSoftware.NONE)
    connected: bool = Field(default=False)
    auto_import: bool = Field(default=False)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_currency: str = Field(default="GBP")

class GrantPreferences(BaseModel):
    """Grant preferences from step 4."""

    primary_region: str = Field(default="UK")
    country: str = Field(default="GB")
    interested_categories: List[str] = Field(
        default_factory=lambda: ["energy_efficiency", "renewable_energy"],
    )
    max_application_effort: str = Field(default="low")
    enable_deadline_alerts: bool = Field(default=True)

class CertificationSelection(BaseModel):
    """Certification pathway from step 5."""

    pathway: CertificationPathway = Field(default=CertificationPathway.SME_CLIMATE_HUB)
    target_year: int = Field(default=2026, ge=2026, le=2030)
    auto_submit_progress: bool = Field(default=True)

class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: SMEWizardStep = Field(...)
    display_name: str = Field(default="")
    description: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)

class WizardState(BaseModel):
    """Complete state of the SME setup wizard."""

    wizard_id: str = Field(default="")
    current_step: SMEWizardStep = Field(default=SMEWizardStep.ORGANIZATION_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    org_profile: Optional[SMEOrganizationProfile] = Field(None)
    data_quality: Optional[DataQualitySelection] = Field(None)
    accounting: Optional[AccountingConnectionSetup] = Field(None)
    grants: Optional[GrantPreferences] = Field(None)
    certification: Optional[CertificationSelection] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)

class SetupResult(BaseModel):
    """Final setup result with generated configuration."""

    result_id: str = Field(default_factory=_new_uuid)
    organization_name: str = Field(default="")
    sector: str = Field(default="")
    country: str = Field(default="")
    employee_count: int = Field(default=0)
    data_quality_tier: str = Field(default="bronze")
    accounting_software: str = Field(default="none")
    accounting_connected: bool = Field(default=False)
    grant_region: str = Field(default="UK")
    certification_pathway: str = Field(default="sme_climate_hub")
    path_type: str = Field(default="simplified")
    scopes_included: List[str] = Field(default_factory=list)
    scope3_categories: List[int] = Field(default_factory=list)
    base_year: int = Field(default=2023)
    target_year: int = Field(default=2030)
    engines_enabled: List[str] = Field(default_factory=list)
    recommended_quick_wins: List[str] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=5)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[SMEWizardStep] = [
    SMEWizardStep.ORGANIZATION_PROFILE,
    SMEWizardStep.DATA_QUALITY_TIER,
    SMEWizardStep.ACCOUNTING_CONNECTION,
    SMEWizardStep.GRANT_PREFERENCES,
    SMEWizardStep.CERTIFICATION_PATHWAY,
]

STEP_DISPLAY_NAMES: Dict[SMEWizardStep, str] = {
    SMEWizardStep.ORGANIZATION_PROFILE: "About Your Business",
    SMEWizardStep.DATA_QUALITY_TIER: "Data Quality Level",
    SMEWizardStep.ACCOUNTING_CONNECTION: "Connect Accounting Software",
    SMEWizardStep.GRANT_PREFERENCES: "Grant & Funding Preferences",
    SMEWizardStep.CERTIFICATION_PATHWAY: "Choose Your Certification",
}

STEP_DESCRIPTIONS: Dict[SMEWizardStep, str] = {
    SMEWizardStep.ORGANIZATION_PROFILE: (
        "Tell us about your business so we can tailor your net-zero journey."
    ),
    SMEWizardStep.DATA_QUALITY_TIER: (
        "Choose how detailed your carbon footprint should be. "
        "Bronze is quickest, Gold is most accurate."
    ),
    SMEWizardStep.ACCOUNTING_CONNECTION: (
        "Connect your accounting software to automatically import spend data. "
        "You can also enter data manually."
    ),
    SMEWizardStep.GRANT_PREFERENCES: (
        "Tell us your location so we can find relevant grants and funding."
    ),
    SMEWizardStep.CERTIFICATION_PATHWAY: (
        "Choose how you want to certify your commitment. "
        "The SME Climate Hub is free and widely recognised."
    ),
}

# ---------------------------------------------------------------------------
# Data Quality Tier Info
# ---------------------------------------------------------------------------

TIER_INFO: Dict[str, Dict[str, Any]] = {
    "bronze": {
        "name": "Bronze",
        "description": "Quick estimate using spend data and industry averages",
        "accuracy": "Indicative (+/- 30%)",
        "data_required": "Annual spend by category",
        "time_to_complete": "30 minutes",
        "scope3_approach": "spend_based",
        "use_industry_defaults": True,
        "recommended_for": "First-time users, micro businesses",
    },
    "silver": {
        "name": "Silver",
        "description": "Better accuracy using utility bills and activity data",
        "accuracy": "Reasonable (+/- 15%)",
        "data_required": "Utility bills, fuel receipts, travel records",
        "time_to_complete": "2-4 hours",
        "scope3_approach": "hybrid",
        "use_industry_defaults": True,
        "recommended_for": "Businesses wanting credible baselines",
    },
    "gold": {
        "name": "Gold",
        "description": "High accuracy using metered data and supplier data",
        "accuracy": "Robust (+/- 5%)",
        "data_required": "Meter readings, supplier-specific factors, full records",
        "time_to_complete": "1-2 weeks",
        "scope3_approach": "activity_based",
        "use_industry_defaults": False,
        "recommended_for": "Businesses seeking SBTi or Carbon Trust certification",
    },
}

# ---------------------------------------------------------------------------
# SMESetupWizard
# ---------------------------------------------------------------------------

class SMESetupWizard:
    """5-step guided configuration wizard for PACK-026 SME Net Zero.

    Guides SMEs through setup with clear explanations and sensible
    defaults. Simpler than the enterprise wizard (PACK-021).

    Example:
        >>> wizard = SMESetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("organization_profile", {...})
        >>> result = wizard.generate_config()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            SMEWizardStep.ORGANIZATION_PROFILE: self._handle_org_profile,
            SMEWizardStep.DATA_QUALITY_TIER: self._handle_data_quality,
            SMEWizardStep.ACCOUNTING_CONNECTION: self._handle_accounting,
            SMEWizardStep.GRANT_PREFERENCES: self._handle_grants,
            SMEWizardStep.CERTIFICATION_PATHWAY: self._handle_certification,
        }
        self.logger.info("SMESetupWizard initialized: 5 steps")

    def start(self) -> WizardState:
        """Start a new wizard session."""
        wizard_id = _compute_hash(f"sme-wizard:{utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
                description=STEP_DESCRIPTIONS.get(step_name, ""),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
        self.logger.info("SME wizard started: %s", wizard_id)
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """Complete a wizard step.

        Args:
            step_name: Step name to complete.
            data: Step configuration data.

        Returns:
            Updated WizardState.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = SMEWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in SMEWizardStep]
            raise ValueError(f"Unknown step '{step_name}'. Valid: {valid}")

        step = self._state.steps.get(step_name)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = utcnow()
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
                step.completed_at = utcnow()
                step.validation_errors = []
                self._advance_step(step_enum)
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000

        return self._state

    def generate_config(self) -> SetupResult:
        """Generate the final pack configuration from wizard state."""
        return self._generate_result()

    def run_demo(self) -> SetupResult:
        """Run a pre-configured demo for a small UK business."""
        self.start()

        demo_steps = {
            "organization_profile": {
                "organization_name": "Green Bakery Ltd",
                "sector": "hospitality",
                "country": "GB",
                "employee_count": 15,
                "annual_revenue_eur": 1_200_000.0,
                "size": "small",
                "fiscal_year_end": "03-31",
            },
            "data_quality_tier": {
                "tier": "bronze",
                "scope3_approach": "spend_based",
                "use_industry_defaults": True,
            },
            "accounting_connection": {
                "software": "xero",
                "connected": True,
                "auto_import": True,
                "reporting_year": 2025,
                "base_currency": "GBP",
            },
            "grant_preferences": {
                "primary_region": "UK",
                "country": "GB",
                "interested_categories": ["energy_efficiency", "renewable_energy"],
                "enable_deadline_alerts": True,
            },
            "certification_pathway": {
                "pathway": "sme_climate_hub",
                "target_year": 2026,
                "auto_submit_progress": True,
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_tier_info(self, tier: str = "") -> Dict[str, Any]:
        """Get information about data quality tiers.

        Args:
            tier: Specific tier, or empty for all tiers.

        Returns:
            Tier information dict.
        """
        if tier:
            return TIER_INFO.get(tier, {})
        return dict(TIER_INFO)

    def get_step_info(self) -> List[Dict[str, Any]]:
        """Get information about all wizard steps."""
        return [
            {
                "step": step.value,
                "display_name": STEP_DISPLAY_NAMES.get(step, ""),
                "description": STEP_DESCRIPTIONS.get(step, ""),
                "status": (
                    self._state.steps[step.value].status.value
                    if self._state and step.value in self._state.steps
                    else "pending"
                ),
            }
            for step in STEP_ORDER
        ]

    # ---- Step Handlers ----

    def _handle_org_profile(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            profile = SMEOrganizationProfile(**data)
            if profile.employee_count > 250:
                errors.append(
                    "This pack is designed for SMEs with up to 250 employees. "
                    "Consider PACK-021 (Net Zero Starter) for larger organisations."
                )
            if self._state:
                self._state.org_profile = profile
        except Exception as exc:
            errors.append(f"Please check your details: {exc}")
        return errors

    def _handle_data_quality(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            selection = DataQualitySelection(**data)
            tier_data = TIER_INFO.get(selection.tier.value)
            if tier_data:
                selection.estimated_accuracy_pct = {
                    "bronze": 70.0, "silver": 85.0, "gold": 95.0,
                }.get(selection.tier.value, 70.0)
                selection.scope3_approach = tier_data.get("scope3_approach", "spend_based")
                selection.use_industry_defaults = tier_data.get("use_industry_defaults", True)
            if self._state:
                self._state.data_quality = selection
        except Exception as exc:
            errors.append(f"Please select a data quality tier: {exc}")
        return errors

    def _handle_accounting(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            setup = AccountingConnectionSetup(**data)
            if self._state:
                self._state.accounting = setup
        except Exception as exc:
            errors.append(f"Please check your accounting setup: {exc}")
        return errors

    def _handle_grants(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            prefs = GrantPreferences(**data)
            if self._state:
                self._state.grants = prefs
        except Exception as exc:
            errors.append(f"Please check your grant preferences: {exc}")
        return errors

    def _handle_certification(self, data: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            selection = CertificationSelection(**data)
            if self._state:
                self._state.certification = selection
        except Exception as exc:
            errors.append(f"Please select a certification pathway: {exc}")
        return errors

    # ---- Navigation ----

    def _advance_step(self, current: SMEWizardStep) -> None:
        if self._state is None:
            return
        try:
            idx = STEP_ORDER.index(current)
            if idx < len(STEP_ORDER) - 1:
                self._state.current_step = STEP_ORDER[idx + 1]
            else:
                self._state.is_complete = True
                self._state.completed_at = utcnow()
        except ValueError:
            pass

    # ---- Result Generation ----

    def _generate_result(self) -> SetupResult:
        if self._state is None:
            return SetupResult()

        completed_count = sum(
            1 for s in self._state.steps.values() if s.status == StepStatus.COMPLETED
        )

        # Determine path type based on tier
        tier = "bronze"
        if self._state.data_quality:
            tier = self._state.data_quality.tier.value
        path_type = "standard" if tier in ("silver", "gold") else "simplified"

        # Scopes
        scopes = ["scope_1", "scope_2", "scope_3"]
        scope3_cats = [1, 6, 7]  # SME default
        if tier == "gold":
            scope3_cats = [1, 2, 3, 4, 5, 6, 7]

        # Engines
        engines = [
            "sme_baseline_engine",
            "sme_target_engine",
            "sme_quick_wins_engine",
            "sme_grant_search_engine",
            "sme_reporting_engine",
        ]
        if tier != "bronze":
            engines.append("sme_scenario_engine")

        # Quick wins
        quick_wins = [
            "Switch to renewable electricity",
            "Install LED lighting",
            "Optimize heating controls",
            "Reduce business travel",
        ]

        config_hash = _compute_hash({
            "org": self._state.org_profile.organization_name if self._state.org_profile else "",
            "tier": tier,
            "accounting": (
                self._state.accounting.software.value if self._state.accounting else "none"
            ),
        })

        result = SetupResult(
            organization_name=(
                self._state.org_profile.organization_name if self._state.org_profile else ""
            ),
            sector=(
                self._state.org_profile.sector if self._state.org_profile else ""
            ),
            country=(
                self._state.org_profile.country if self._state.org_profile else "GB"
            ),
            employee_count=(
                self._state.org_profile.employee_count if self._state.org_profile else 0
            ),
            data_quality_tier=tier,
            accounting_software=(
                self._state.accounting.software.value if self._state.accounting else "none"
            ),
            accounting_connected=(
                self._state.accounting.connected if self._state.accounting else False
            ),
            grant_region=(
                self._state.grants.primary_region if self._state.grants else "UK"
            ),
            certification_pathway=(
                self._state.certification.pathway.value
                if self._state.certification else "sme_climate_hub"
            ),
            path_type=path_type,
            scopes_included=scopes,
            scope3_categories=scope3_cats,
            base_year=2023,
            target_year=2030,
            engines_enabled=engines,
            recommended_quick_wins=quick_wins,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
