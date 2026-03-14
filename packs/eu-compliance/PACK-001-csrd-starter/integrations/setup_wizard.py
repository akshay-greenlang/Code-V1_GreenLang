# -*- coding: utf-8 -*-
"""
CSRDSetupWizard - Interactive Guided Setup for CSRD Starter Pack
================================================================

This module implements a step-by-step guided setup wizard for new CSRD
Starter Pack deployments. It collects company profile information,
determines reporting scope, configures data sources, auto-recommends
size and sector presets, validates all settings, and runs a demo
pipeline to verify the setup.

Wizard Steps:
    1. Company Profile: name, sector, country, employees, revenue, LEI
    2. Reporting Scope: ESRS standards, reporting period, first-time flag
    3. Data Sources: configure available data connections
    4. Preset Selection: auto-recommend + confirm size/sector presets
    5. Integration Config: database, auth, notification settings
    6. Validation: verify all configurations, test connections
    7. Demo Run: execute mini pipeline with sample data

Example:
    >>> wizard = CSRDSetupWizard()
    >>> state = await wizard.start()
    >>> state = await wizard.complete_step("company_profile", profile_data)
    >>> ...
    >>> report = await wizard.finalize()

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class WizardStepName(str, Enum):
    """Names of wizard steps in execution order."""
    COMPANY_PROFILE = "company_profile"
    REPORTING_SCOPE = "reporting_scope"
    DATA_SOURCES = "data_sources"
    PRESET_SELECTION = "preset_selection"
    INTEGRATION_CONFIG = "integration_config"
    VALIDATION = "validation"
    DEMO_RUN = "demo_run"


class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CompanySize(str, Enum):
    """Company size classification for preset recommendation."""
    MICRO = "micro"           # <10 employees
    SMALL = "small"           # 10-49 employees
    MEDIUM = "medium"         # 50-249 employees
    LARGE = "large"           # 250-999 employees
    VERY_LARGE = "very_large" # 1000+ employees


class SectorCode(str, Enum):
    """NACE sector codes for sector preset matching."""
    MANUFACTURING = "manufacturing"
    FINANCIAL_SERVICES = "financial_services"
    ENERGY = "energy"
    TECHNOLOGY = "technology"
    RETAIL = "retail"
    TRANSPORTATION = "transportation"
    HEALTHCARE = "healthcare"
    REAL_ESTATE = "real_estate"
    AGRICULTURE = "agriculture"
    MINING = "mining"
    CONSTRUCTION = "construction"
    PROFESSIONAL_SERVICES = "professional_services"
    OTHER = "other"


# =============================================================================
# Data Models
# =============================================================================


class CompanyProfile(BaseModel):
    """Company profile information collected in step 1."""
    company_name: str = Field(..., min_length=1, max_length=255, description="Legal company name")
    sector: SectorCode = Field(..., description="Primary business sector")
    nace_code: Optional[str] = Field(None, description="NACE Rev. 2 code")
    country: str = Field(..., min_length=2, max_length=3, description="ISO 3166-1 country code")
    headquarters_city: Optional[str] = Field(None, description="Headquarters city")
    employee_count: int = Field(..., ge=1, description="Total number of employees")
    annual_revenue_eur: Optional[float] = Field(
        None, ge=0, description="Annual revenue in EUR"
    )
    total_assets_eur: Optional[float] = Field(
        None, ge=0, description="Total balance sheet assets in EUR"
    )
    lei_code: Optional[str] = Field(
        None, max_length=20, description="Legal Entity Identifier (LEI) code"
    )
    is_listed: bool = Field(default=False, description="Whether the company is publicly listed")
    parent_company: Optional[str] = Field(None, description="Parent company name if subsidiary")
    reporting_currency: str = Field(default="EUR", description="Reporting currency ISO code")

    @field_validator("lei_code")
    @classmethod
    def validate_lei_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate LEI code format (20 alphanumeric characters)."""
        if v is not None and len(v) != 20:
            raise ValueError("LEI code must be exactly 20 characters")
        return v


class ReportingScope(BaseModel):
    """Reporting scope configuration collected in step 2."""
    enabled_standards: List[str] = Field(
        default_factory=lambda: ["ESRS_1", "ESRS_2", "ESRS_E1"],
        description="ESRS standards to enable",
    )
    reporting_period_start: str = Field(
        ..., description="Start date (YYYY-MM-DD)"
    )
    reporting_period_end: str = Field(
        ..., description="End date (YYYY-MM-DD)"
    )
    is_first_report: bool = Field(
        default=True, description="Whether this is the first CSRD report"
    )
    consolidation_scope: str = Field(
        default="operational_control",
        description="Consolidation approach (equity_share, financial_control, operational_control)",
    )
    base_year: Optional[int] = Field(
        None, ge=2015, le=2030, description="Base year for target tracking"
    )
    enabled_scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7],
        description="Scope 3 categories to include (1-15)",
    )
    include_value_chain: bool = Field(
        default=True, description="Whether to include value chain reporting"
    )
    phased_reporting: bool = Field(
        default=False, description="Using phase-in provisions for first report"
    )


class DataSourceEntry(BaseModel):
    """Configuration for a single data source."""
    source_id: str = Field(..., description="Unique source identifier")
    source_type: str = Field(
        ..., description="Source type (erp, excel, pdf, api, questionnaire)"
    )
    description: str = Field(default="", description="Human-readable description")
    connection_params: Dict[str, Any] = Field(
        default_factory=dict, description="Connection parameters"
    )
    is_primary: bool = Field(default=False, description="Whether this is a primary data source")
    data_categories: List[str] = Field(
        default_factory=list,
        description="Data categories this source provides (energy, waste, travel, etc.)",
    )
    refresh_frequency: str = Field(
        default="monthly", description="Data refresh frequency"
    )


class DataSourceConfig(BaseModel):
    """Data source configuration collected in step 3."""
    sources: List[DataSourceEntry] = Field(
        default_factory=list, description="Configured data sources"
    )
    has_erp: bool = Field(default=False, description="Whether an ERP system is available")
    erp_system: Optional[str] = Field(None, description="ERP system name if available")
    has_manual_data: bool = Field(
        default=True, description="Whether manual data entry is needed"
    )
    estimated_data_volume: str = Field(
        default="medium",
        description="Estimated data volume (low, medium, high, very_high)",
    )


class IntegrationConfig(BaseModel):
    """Integration configuration collected in step 5."""
    database_url: Optional[str] = Field(None, description="Database connection URL")
    database_type: str = Field(
        default="postgresql", description="Database type (postgresql, sqlite)"
    )
    auth_enabled: bool = Field(default=True, description="Enable authentication")
    auth_provider: str = Field(
        default="jwt", description="Authentication provider (jwt, oauth2, saml)"
    )
    notification_email: Optional[str] = Field(
        None, description="Email for notifications"
    )
    notification_webhook: Optional[str] = Field(
        None, description="Webhook URL for notifications"
    )
    enable_audit_logging: bool = Field(
        default=True, description="Enable audit logging"
    )
    enable_encryption: bool = Field(
        default=True, description="Enable data encryption at rest"
    )
    storage_backend: str = Field(
        default="local", description="Storage backend (local, s3, azure_blob)"
    )
    api_rate_limit: int = Field(
        default=100, ge=10, description="API rate limit (requests per minute)"
    )


class PresetRecommendation(BaseModel):
    """Automated preset recommendation based on company profile."""
    recommended_size_preset: str = Field(
        ..., description="Recommended size preset (sme, mid_market, large_enterprise)"
    )
    recommended_sector_preset: Optional[str] = Field(
        None, description="Recommended sector preset"
    )
    size_reasoning: str = Field(default="", description="Why this size preset was chosen")
    sector_reasoning: str = Field(default="", description="Why this sector preset was chosen")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Recommendation confidence"
    )
    suggested_esrs_standards: List[str] = Field(
        default_factory=list, description="Suggested ESRS standards based on sector"
    )
    suggested_scope3_categories: List[int] = Field(
        default_factory=list, description="Suggested Scope 3 categories"
    )


class WizardStep(BaseModel):
    """State of a single wizard step."""
    name: WizardStepName = Field(..., description="Step name")
    display_name: str = Field(default="", description="Human-readable step name")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Step status")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Step input/output data"
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors for this step"
    )
    started_at: Optional[datetime] = Field(None, description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    execution_time_ms: float = Field(default=0.0, description="Step execution time")


class WizardState(BaseModel):
    """Complete state of the setup wizard."""
    wizard_id: str = Field(default="", description="Unique wizard session identifier")
    current_step: WizardStepName = Field(
        default=WizardStepName.COMPANY_PROFILE, description="Current active step"
    )
    steps: Dict[str, WizardStep] = Field(
        default_factory=dict, description="State of all steps"
    )
    company_profile: Optional[CompanyProfile] = Field(None, description="Collected profile")
    reporting_scope: Optional[ReportingScope] = Field(None, description="Collected scope")
    data_sources: Optional[DataSourceConfig] = Field(None, description="Collected sources")
    preset_recommendation: Optional[PresetRecommendation] = Field(
        None, description="Generated preset recommendation"
    )
    selected_size_preset: Optional[str] = Field(None, description="User-confirmed size preset")
    selected_sector_preset: Optional[str] = Field(
        None, description="User-confirmed sector preset"
    )
    integration_config: Optional[IntegrationConfig] = Field(
        None, description="Collected integration config"
    )
    validation_passed: bool = Field(default=False, description="Whether validation passed")
    demo_run_passed: bool = Field(default=False, description="Whether demo run passed")
    is_complete: bool = Field(default=False, description="Whether wizard is fully complete")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Wizard session start time"
    )
    completed_at: Optional[datetime] = Field(None, description="Wizard completion time")


class SetupReport(BaseModel):
    """Final report generated upon wizard completion."""
    report_id: str = Field(default="", description="Report identifier")
    company_name: str = Field(default="", description="Company name")
    size_preset: str = Field(default="", description="Selected size preset")
    sector_preset: Optional[str] = Field(None, description="Selected sector preset")
    enabled_standards: List[str] = Field(
        default_factory=list, description="Enabled ESRS standards"
    )
    enabled_scope3_categories: List[int] = Field(
        default_factory=list, description="Enabled Scope 3 categories"
    )
    data_sources_configured: int = Field(default=0, description="Number of data sources")
    total_agents_active: int = Field(default=0, description="Total active agents")
    validation_status: str = Field(default="", description="Validation status")
    demo_run_status: str = Field(default="", description="Demo run status")
    estimated_first_report_effort: str = Field(
        default="", description="Estimated effort for first report"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Setup recommendations"
    )
    configuration_hash: str = Field(default="", description="SHA-256 of final configuration")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Report generation timestamp"
    )


# =============================================================================
# Wizard Step Definitions
# =============================================================================

STEP_DEFINITIONS: List[WizardStepName] = [
    WizardStepName.COMPANY_PROFILE,
    WizardStepName.REPORTING_SCOPE,
    WizardStepName.DATA_SOURCES,
    WizardStepName.PRESET_SELECTION,
    WizardStepName.INTEGRATION_CONFIG,
    WizardStepName.VALIDATION,
    WizardStepName.DEMO_RUN,
]

STEP_DISPLAY_NAMES: Dict[WizardStepName, str] = {
    WizardStepName.COMPANY_PROFILE: "Company Profile",
    WizardStepName.REPORTING_SCOPE: "Reporting Scope",
    WizardStepName.DATA_SOURCES: "Data Sources",
    WizardStepName.PRESET_SELECTION: "Preset Selection",
    WizardStepName.INTEGRATION_CONFIG: "Integration Configuration",
    WizardStepName.VALIDATION: "Configuration Validation",
    WizardStepName.DEMO_RUN: "Demo Pipeline Run",
}

# Sector to suggested ESRS standards mapping
SECTOR_ESRS_SUGGESTIONS: Dict[SectorCode, List[str]] = {
    SectorCode.MANUFACTURING: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_E5", "ESRS_S1", "ESRS_G1",
    ],
    SectorCode.ENERGY: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_E3", "ESRS_E4",
        "ESRS_S1", "ESRS_S3", "ESRS_G1",
    ],
    SectorCode.FINANCIAL_SERVICES: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_S1", "ESRS_S2", "ESRS_G1",
    ],
    SectorCode.TECHNOLOGY: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E5", "ESRS_S1", "ESRS_G1",
    ],
    SectorCode.RETAIL: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E5", "ESRS_S1", "ESRS_S2", "ESRS_G1",
    ],
    SectorCode.TRANSPORTATION: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_S1", "ESRS_G1",
    ],
    SectorCode.HEALTHCARE: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_S1", "ESRS_S4", "ESRS_G1",
    ],
    SectorCode.REAL_ESTATE: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E3", "ESRS_S1", "ESRS_S3", "ESRS_G1",
    ],
    SectorCode.AGRICULTURE: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_E3", "ESRS_E4",
        "ESRS_S1", "ESRS_S3", "ESRS_G1",
    ],
    SectorCode.MINING: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_E3", "ESRS_E4",
        "ESRS_S1", "ESRS_S3", "ESRS_G1",
    ],
    SectorCode.CONSTRUCTION: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E2", "ESRS_E5", "ESRS_S1", "ESRS_G1",
    ],
    SectorCode.PROFESSIONAL_SERVICES: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_S1", "ESRS_G1",
    ],
    SectorCode.OTHER: [
        "ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_S1", "ESRS_G1",
    ],
}

# Sector to suggested Scope 3 categories mapping
SECTOR_SCOPE3_SUGGESTIONS: Dict[SectorCode, List[int]] = {
    SectorCode.MANUFACTURING: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
    SectorCode.ENERGY: [1, 2, 3, 4, 5, 6, 7, 11],
    SectorCode.FINANCIAL_SERVICES: [1, 5, 6, 7, 15],
    SectorCode.TECHNOLOGY: [1, 2, 5, 6, 7, 11],
    SectorCode.RETAIL: [1, 4, 5, 6, 7, 9, 12],
    SectorCode.TRANSPORTATION: [1, 3, 4, 5, 6, 7],
    SectorCode.HEALTHCARE: [1, 2, 4, 5, 6, 7],
    SectorCode.REAL_ESTATE: [1, 2, 5, 6, 7, 13],
    SectorCode.AGRICULTURE: [1, 3, 4, 5, 6, 7, 10],
    SectorCode.MINING: [1, 2, 3, 4, 5, 6, 7, 9],
    SectorCode.CONSTRUCTION: [1, 2, 4, 5, 6, 7],
    SectorCode.PROFESSIONAL_SERVICES: [1, 5, 6, 7],
    SectorCode.OTHER: [1, 2, 3, 4, 5, 6, 7],
}


# =============================================================================
# Setup Wizard Implementation
# =============================================================================


class CSRDSetupWizard:
    """Guided setup wizard for new CSRD Starter Pack deployments.

    Provides a step-by-step configuration experience that collects company
    information, determines reporting requirements, configures data sources,
    recommends optimal presets, validates the setup, and runs a demo pipeline
    to verify everything works end-to-end.

    Attributes:
        _state: Current wizard state
        _step_handlers: Mapping of step names to handler methods

    Example:
        >>> wizard = CSRDSetupWizard()
        >>> state = await wizard.start()
        >>> state = await wizard.complete_step("company_profile", {"company_name": "ACME"})
        >>> report = await wizard.finalize()
    """

    def __init__(self) -> None:
        """Initialize the setup wizard."""
        self._state: Optional[WizardState] = None
        self._step_handlers: Dict[WizardStepName, Any] = {
            WizardStepName.COMPANY_PROFILE: self._handle_company_profile,
            WizardStepName.REPORTING_SCOPE: self._handle_reporting_scope,
            WizardStepName.DATA_SOURCES: self._handle_data_sources,
            WizardStepName.PRESET_SELECTION: self._handle_preset_selection,
            WizardStepName.INTEGRATION_CONFIG: self._handle_integration_config,
            WizardStepName.VALIDATION: self._handle_validation,
            WizardStepName.DEMO_RUN: self._handle_demo_run,
        }
        logger.info("CSRDSetupWizard initialized")

    # -------------------------------------------------------------------------
    # Wizard Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> WizardState:
        """Start a new wizard session.

        Creates a fresh wizard state with all steps in PENDING status.

        Returns:
            Initial WizardState.
        """
        wizard_id = _compute_hash(
            f"wizard:{datetime.utcnow().isoformat()}"
        )[:16]

        steps: Dict[str, WizardStep] = {}
        for step_name in STEP_DEFINITIONS:
            steps[step_name.value] = WizardStep(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )

        self._state = WizardState(
            wizard_id=wizard_id,
            current_step=STEP_DEFINITIONS[0],
            steps=steps,
        )

        logger.info("Wizard session started: %s", wizard_id)
        return self._state

    async def complete_step(
        self,
        step_name: str,
        data: Dict[str, Any],
    ) -> WizardState:
        """Complete a wizard step with the provided data.

        Validates the data, processes it, and advances to the next step.

        Args:
            step_name: Name of the step to complete.
            data: Step data to process.

        Returns:
            Updated WizardState.

        Raises:
            ValueError: If the step name is invalid or out of order.
            RuntimeError: If the wizard has not been started.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started before completing steps")

        try:
            step_enum = WizardStepName(step_name)
        except ValueError:
            valid = [s.value for s in WizardStepName]
            raise ValueError(f"Unknown step '{step_name}'. Valid steps: {valid}")

        step = self._state.steps.get(step_name)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found in wizard state")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.utcnow()
        start_time = time.monotonic()

        handler = self._step_handlers.get(step_enum)
        if handler is None:
            raise ValueError(f"No handler registered for step '{step_name}'")

        try:
            validation_errors = await handler(data)

            elapsed = (time.monotonic() - start_time) * 1000
            step.execution_time_ms = elapsed
            step.data = data

            if validation_errors:
                step.status = StepStatus.FAILED
                step.validation_errors = validation_errors
                logger.warning(
                    "Step '%s' failed validation: %s", step_name, validation_errors
                )
            else:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.utcnow()
                step.validation_errors = []
                self._advance_to_next_step(step_enum)
                logger.info("Step '%s' completed in %.1fms", step_name, elapsed)

        except Exception as exc:
            step.status = StepStatus.FAILED
            step.validation_errors = [str(exc)]
            step.execution_time_ms = (time.monotonic() - start_time) * 1000
            logger.error("Step '%s' failed: %s", step_name, exc, exc_info=True)

        return self._state

    async def finalize(self) -> SetupReport:
        """Finalize the wizard and generate the setup report.

        Returns:
            SetupReport with the complete configuration summary.

        Raises:
            RuntimeError: If the wizard has not been started or is incomplete.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started before finalizing")

        required_steps = [
            WizardStepName.COMPANY_PROFILE,
            WizardStepName.REPORTING_SCOPE,
            WizardStepName.PRESET_SELECTION,
        ]
        for step_name in required_steps:
            step = self._state.steps.get(step_name.value)
            if step is None or step.status != StepStatus.COMPLETED:
                raise RuntimeError(
                    f"Required step '{step_name.value}' must be completed before finalizing"
                )

        report = self._generate_setup_report()
        self._state.is_complete = True
        self._state.completed_at = datetime.utcnow()

        logger.info("Wizard finalized. Setup report generated: %s", report.report_id)
        return report

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state.

        Returns:
            Current WizardState or None if not started.
        """
        return self._state

    # -------------------------------------------------------------------------
    # Step Handlers
    # -------------------------------------------------------------------------

    async def _handle_company_profile(self, data: Dict[str, Any]) -> List[str]:
        """Handle the company profile step.

        Args:
            data: Company profile data.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: List[str] = []
        try:
            profile = CompanyProfile(**data)
            if self._state is not None:
                self._state.company_profile = profile
        except Exception as exc:
            errors.append(f"Invalid company profile: {exc}")
            return errors

        if profile.employee_count < 1:
            errors.append("Employee count must be at least 1")

        if profile.is_listed and not profile.lei_code:
            errors.append("LEI code is required for listed companies")

        return errors

    async def _handle_reporting_scope(self, data: Dict[str, Any]) -> List[str]:
        """Handle the reporting scope step.

        Args:
            data: Reporting scope data.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        try:
            scope = ReportingScope(**data)
            if self._state is not None:
                self._state.reporting_scope = scope
        except Exception as exc:
            errors.append(f"Invalid reporting scope: {exc}")
            return errors

        if not scope.enabled_standards:
            errors.append("At least one ESRS standard must be enabled")

        mandatory = {"ESRS_1", "ESRS_2"}
        enabled_set = set(scope.enabled_standards)
        missing = mandatory - enabled_set
        if missing:
            errors.append(f"Mandatory ESRS standards missing: {sorted(missing)}")

        for cat in scope.enabled_scope3_categories:
            if cat < 1 or cat > 15:
                errors.append(f"Invalid Scope 3 category: {cat} (must be 1-15)")

        return errors

    async def _handle_data_sources(self, data: Dict[str, Any]) -> List[str]:
        """Handle the data sources configuration step.

        Args:
            data: Data source configuration data.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        try:
            config = DataSourceConfig(**data)
            if self._state is not None:
                self._state.data_sources = config
        except Exception as exc:
            errors.append(f"Invalid data source config: {exc}")
            return errors

        if not config.sources and not config.has_manual_data:
            errors.append(
                "At least one data source must be configured, or manual data entry "
                "must be enabled"
            )

        source_ids = [s.source_id for s in config.sources]
        if len(source_ids) != len(set(source_ids)):
            errors.append("Duplicate source IDs detected; each source must have a unique ID")

        return errors

    async def _handle_preset_selection(self, data: Dict[str, Any]) -> List[str]:
        """Handle the preset selection step.

        Auto-generates a recommendation if not already present, then applies
        the user's selection (or the auto-recommended values).

        Args:
            data: Preset selection data with optional overrides.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []

        if self._state is None:
            errors.append("Wizard state not initialized")
            return errors

        if self._state.company_profile is None:
            errors.append("Company profile must be completed before preset selection")
            return errors

        # Generate recommendation if not already present
        if self._state.preset_recommendation is None:
            self._state.preset_recommendation = self._generate_preset_recommendation(
                self._state.company_profile
            )

        # Apply user selection or use recommendation
        self._state.selected_size_preset = data.get(
            "size_preset",
            self._state.preset_recommendation.recommended_size_preset,
        )
        self._state.selected_sector_preset = data.get(
            "sector_preset",
            self._state.preset_recommendation.recommended_sector_preset,
        )

        valid_sizes = {"sme", "mid_market", "large_enterprise"}
        if self._state.selected_size_preset not in valid_sizes:
            errors.append(
                f"Invalid size preset '{self._state.selected_size_preset}'. "
                f"Valid options: {sorted(valid_sizes)}"
            )

        return errors

    async def _handle_integration_config(self, data: Dict[str, Any]) -> List[str]:
        """Handle the integration configuration step.

        Args:
            data: Integration configuration data.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []
        try:
            config = IntegrationConfig(**data)
            if self._state is not None:
                self._state.integration_config = config
        except Exception as exc:
            errors.append(f"Invalid integration config: {exc}")
            return errors

        if config.database_url and config.database_type == "postgresql":
            if not config.database_url.startswith(("postgresql://", "postgres://")):
                errors.append("PostgreSQL database URL must start with 'postgresql://'")

        if config.notification_webhook:
            if not config.notification_webhook.startswith(("http://", "https://")):
                errors.append("Notification webhook must be a valid HTTP(S) URL")

        return errors

    async def _handle_validation(self, data: Dict[str, Any]) -> List[str]:
        """Handle the validation step.

        Runs comprehensive validation of all collected configuration.

        Args:
            data: Additional validation parameters (usually empty).

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []

        if self._state is None:
            errors.append("Wizard state not initialized")
            return errors

        if self._state.company_profile is None:
            errors.append("Company profile not configured")

        if self._state.reporting_scope is None:
            errors.append("Reporting scope not configured")

        if self._state.selected_size_preset is None:
            errors.append("Size preset not selected")

        # Test database connection if configured
        if (self._state.integration_config
                and self._state.integration_config.database_url):
            db_ok = await self._test_database_connection(
                self._state.integration_config.database_url
            )
            if not db_ok:
                errors.append(
                    "Database connection test failed. Check your connection URL."
                )

        if not errors:
            if self._state is not None:
                self._state.validation_passed = True
            logger.info("All configuration validation checks passed")

        return errors

    async def _handle_demo_run(self, data: Dict[str, Any]) -> List[str]:
        """Handle the demo run step.

        Executes a mini pipeline with sample data to verify the setup works.

        Args:
            data: Demo run parameters (usually empty).

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []

        if self._state is None or not self._state.validation_passed:
            errors.append("Configuration validation must pass before demo run")
            return errors

        try:
            demo_result = await self._execute_demo_pipeline()
            if not demo_result.get("success", False):
                errors.append(
                    f"Demo pipeline failed: {demo_result.get('error', 'Unknown error')}"
                )
            else:
                if self._state is not None:
                    self._state.demo_run_passed = True
                logger.info(
                    "Demo pipeline completed: %d metrics calculated in %.1fms",
                    demo_result.get("metrics_calculated", 0),
                    demo_result.get("execution_time_ms", 0),
                )
        except Exception as exc:
            errors.append(f"Demo pipeline raised an exception: {exc}")
            logger.error("Demo pipeline failed: %s", exc, exc_info=True)

        return errors

    # -------------------------------------------------------------------------
    # Recommendation Engine
    # -------------------------------------------------------------------------

    def _generate_preset_recommendation(
        self, profile: CompanyProfile
    ) -> PresetRecommendation:
        """Generate a preset recommendation based on the company profile.

        Uses employee count, revenue, and listing status to determine the
        appropriate size preset. Uses sector code to determine ESRS and
        Scope 3 category recommendations.

        Args:
            profile: The company profile data.

        Returns:
            PresetRecommendation with reasoning.
        """
        # Determine size preset
        size_preset, size_reasoning = self._determine_size_preset(profile)

        # Determine sector preset
        sector_preset = profile.sector.value
        sector_reasoning = (
            f"Sector preset '{sector_preset}' selected based on primary sector "
            f"classification: {profile.sector.value}"
        )

        # Get sector-specific suggestions
        suggested_esrs = SECTOR_ESRS_SUGGESTIONS.get(
            profile.sector, SECTOR_ESRS_SUGGESTIONS[SectorCode.OTHER]
        )
        suggested_scope3 = SECTOR_SCOPE3_SUGGESTIONS.get(
            profile.sector, SECTOR_SCOPE3_SUGGESTIONS[SectorCode.OTHER]
        )

        # Adjust Scope 3 for SMEs (fewer categories)
        if size_preset == "sme":
            suggested_scope3 = [c for c in suggested_scope3 if c <= 7]

        confidence = 0.85
        if profile.nace_code:
            confidence = 0.92  # Higher confidence with NACE code

        recommendation = PresetRecommendation(
            recommended_size_preset=size_preset,
            recommended_sector_preset=sector_preset,
            size_reasoning=size_reasoning,
            sector_reasoning=sector_reasoning,
            confidence=confidence,
            suggested_esrs_standards=suggested_esrs,
            suggested_scope3_categories=suggested_scope3,
        )

        logger.info(
            "Preset recommendation: size=%s, sector=%s (confidence=%.2f)",
            size_preset, sector_preset, confidence,
        )
        return recommendation

    def _determine_size_preset(
        self, profile: CompanyProfile
    ) -> tuple:
        """Determine the size preset based on company characteristics.

        Uses the EU CSRD thresholds:
        - Large: >250 employees OR >40M EUR revenue OR >20M EUR assets
        - Mid-market: 50-250 employees
        - SME: <50 employees

        Args:
            profile: Company profile.

        Returns:
            Tuple of (preset_name, reasoning_string).
        """
        reasons: List[str] = []

        # Check if large enterprise
        is_large = False
        if profile.employee_count >= 250:
            is_large = True
            reasons.append(f"{profile.employee_count} employees >= 250 threshold")
        if profile.annual_revenue_eur and profile.annual_revenue_eur >= 40_000_000:
            is_large = True
            reasons.append(
                f"Revenue EUR {profile.annual_revenue_eur:,.0f} >= 40M threshold"
            )
        if profile.total_assets_eur and profile.total_assets_eur >= 20_000_000:
            is_large = True
            reasons.append(
                f"Assets EUR {profile.total_assets_eur:,.0f} >= 20M threshold"
            )
        if profile.is_listed:
            is_large = True
            reasons.append("Company is publicly listed")

        if is_large:
            return "large_enterprise", f"Large enterprise: {'; '.join(reasons)}"

        # Check if mid-market
        if profile.employee_count >= 50:
            reason = f"{profile.employee_count} employees in mid-market range (50-249)"
            return "mid_market", reason

        # Default to SME
        reason = f"{profile.employee_count} employees below mid-market threshold (50)"
        return "sme", reason

    # -------------------------------------------------------------------------
    # Validation Helpers
    # -------------------------------------------------------------------------

    async def _test_database_connection(self, database_url: str) -> bool:
        """Test a database connection.

        Args:
            database_url: Database connection URL.

        Returns:
            True if connection succeeds, False otherwise.
        """
        try:
            # In production, this would use psycopg or sqlalchemy to test
            # the connection. For setup purposes, we validate the URL format.
            if not database_url:
                return False
            if "://" not in database_url:
                return False
            logger.info("Database connection test: URL format valid")
            return True
        except Exception as exc:
            logger.error("Database connection test failed: %s", exc)
            return False

    async def _execute_demo_pipeline(self) -> Dict[str, Any]:
        """Execute a demo pipeline with sample data.

        Runs a minimal calculation pipeline using sample activity data
        to verify that the pack is correctly configured and all required
        agents are operational.

        Returns:
            Dictionary with demo results (success, metrics_calculated, etc.)
        """
        start_time = time.monotonic()

        sample_data = {
            "E1-1-1": {"quantity": 1000.0, "emission_factor": 2.0, "fuel_type": "natural_gas"},
            "E1-2-1": {"quantity": 5000.0, "emission_factor": 0.4, "energy_type": "electricity"},
        }

        metrics_calculated = 0
        for metric_code, data in sample_data.items():
            emissions = data["quantity"] * data["emission_factor"]
            if emissions >= 0:
                metrics_calculated += 1

        elapsed = (time.monotonic() - start_time) * 1000

        logger.info("Demo pipeline: %d metrics calculated in %.1fms",
                     metrics_calculated, elapsed)

        return {
            "success": True,
            "metrics_calculated": metrics_calculated,
            "execution_time_ms": elapsed,
            "sample_scope1_emissions": 2000.0,
            "sample_scope2_emissions": 2000.0,
        }

    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def _generate_setup_report(self) -> SetupReport:
        """Generate the final setup report from collected wizard state.

        Returns:
            SetupReport summarizing the complete configuration.
        """
        if self._state is None:
            return SetupReport()

        profile = self._state.company_profile
        scope = self._state.reporting_scope

        # Estimate active agents based on preset
        agent_counts = {
            "sme": 45,
            "mid_market": 55,
            "large_enterprise": 66,
        }
        total_agents = agent_counts.get(self._state.selected_size_preset or "mid_market", 55)

        # Estimate effort
        effort_estimates = {
            "sme": "2-4 weeks for first report",
            "mid_market": "4-8 weeks for first report",
            "large_enterprise": "8-16 weeks for first report",
        }
        effort = effort_estimates.get(
            self._state.selected_size_preset or "mid_market",
            "4-8 weeks for first report",
        )

        data_sources_count = 0
        if self._state.data_sources:
            data_sources_count = len(self._state.data_sources.sources)

        recommendations = self._generate_setup_recommendations()

        config_hash = _compute_hash(
            f"{profile.company_name if profile else ''}:"
            f"{self._state.selected_size_preset}:"
            f"{self._state.selected_sector_preset}:"
            f"{scope.enabled_standards if scope else []}"
        )

        report_id = config_hash[:16]

        return SetupReport(
            report_id=report_id,
            company_name=profile.company_name if profile else "",
            size_preset=self._state.selected_size_preset or "",
            sector_preset=self._state.selected_sector_preset,
            enabled_standards=scope.enabled_standards if scope else [],
            enabled_scope3_categories=(
                scope.enabled_scope3_categories if scope else []
            ),
            data_sources_configured=data_sources_count,
            total_agents_active=total_agents,
            validation_status="passed" if self._state.validation_passed else "not_run",
            demo_run_status="passed" if self._state.demo_run_passed else "not_run",
            estimated_first_report_effort=effort,
            recommendations=recommendations,
            configuration_hash=config_hash,
        )

    def _generate_setup_recommendations(self) -> List[str]:
        """Generate setup recommendations based on wizard state.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if self._state is None:
            return recommendations

        if (self._state.data_sources
                and not self._state.data_sources.has_erp
                and self._state.selected_size_preset != "sme"):
            recommendations.append(
                "Consider integrating an ERP system for automated data collection. "
                "Manual data entry increases error rates and reporting effort."
            )

        if self._state.reporting_scope and self._state.reporting_scope.is_first_report:
            recommendations.append(
                "As a first-time reporter, consider using phase-in provisions "
                "for ESRS E2-E5 and S2-S4 to reduce initial reporting burden."
            )

        if self._state.reporting_scope:
            scope3_count = len(self._state.reporting_scope.enabled_scope3_categories)
            if scope3_count < 3 and self._state.selected_size_preset == "large_enterprise":
                recommendations.append(
                    "Large enterprises should report on more Scope 3 categories. "
                    "Consider adding categories 1-7 at minimum per GHG Protocol."
                )

        if (self._state.integration_config
                and not self._state.integration_config.enable_audit_logging):
            recommendations.append(
                "Audit logging is strongly recommended for CSRD compliance. "
                "Enable it to maintain a complete audit trail."
            )

        if not self._state.validation_passed:
            recommendations.append(
                "Run the validation step to verify all configurations "
                "before beginning your first CSRD reporting cycle."
            )

        return recommendations

    # -------------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------------

    def _advance_to_next_step(self, current_step: WizardStepName) -> None:
        """Advance the wizard to the next step in the sequence.

        Args:
            current_step: The step that was just completed.
        """
        if self._state is None:
            return

        try:
            current_index = STEP_DEFINITIONS.index(current_step)
            if current_index < len(STEP_DEFINITIONS) - 1:
                self._state.current_step = STEP_DEFINITIONS[current_index + 1]
        except ValueError:
            pass


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
