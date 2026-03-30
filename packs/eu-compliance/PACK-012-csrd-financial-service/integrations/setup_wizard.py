# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Configuration for FI CSRD Reporting
================================================================

This module implements an 8-step guided configuration wizard for setting
up CSRD reporting for financial institutions within PACK-012. The wizard
walks through institution type selection, regulatory scope determination,
data source configuration, engine configuration, workflow configuration,
template selection, integration setup, and review/validation.

Architecture:
    User --> SetupWizard (8 Steps) --> FSOrchestrationConfig
                  |
                  v
    Step 1: Institution Type Selection
    Step 2: Regulatory Scope
    Step 3: Data Source Configuration
    Step 4: Engine Configuration
    Step 5: Workflow Configuration
    Step 6: Template Selection
    Step 7: Integration Setup
    Step 8: Review & Validate

Regulatory Context:
    Financial institutions under CSRD have unique reporting obligations:
    - Banks: GAR, BTAR, Pillar 3 ESG, financed emissions (PCAF)
    - Insurers: Underwriting emissions, climate risk (Solvency II)
    - Asset Managers: SFDR cross-reference, financed emissions
    - Investment Firms: GAR (simplified), financed emissions
    - Pension Funds: SFDR, IORP II alignment, financed emissions
    - Development Banks: Impact reporting, financed emissions

    The wizard adapts its steps and defaults based on institution type.

Example:
    >>> config = SetupWizardConfig()
    >>> wizard = SetupWizard(config)
    >>> result = wizard.execute_all_steps(institution_data)
    >>> print(f"Valid: {result.is_valid}, config ready: {result.config_ready}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
Version: 1.0.0
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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Enums
# =============================================================================

class WizardStepId(str, Enum):
    """Wizard step identifiers."""
    INSTITUTION_TYPE = "institution_type_selection"
    REGULATORY_SCOPE = "regulatory_scope"
    DATA_SOURCE_CONFIG = "data_source_config"
    ENGINE_CONFIG = "engine_config"
    WORKFLOW_CONFIG = "workflow_config"
    TEMPLATE_SELECTION = "template_selection"
    INTEGRATION_SETUP = "integration_setup"
    REVIEW_VALIDATE = "review_validate"

class InstitutionType(str, Enum):
    """Financial institution types."""
    BANK = "bank"
    INSURER = "insurer"
    ASSET_MANAGER = "asset_manager"
    INVESTMENT_FIRM = "investment_firm"
    PENSION_FUND = "pension_fund"
    DEVELOPMENT_BANK = "development_bank"

class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

class RegulatoryFramework(str, Enum):
    """Applicable regulatory frameworks."""
    CSRD = "csrd"
    EBA_PILLAR3 = "eba_pillar3"
    SFDR = "sfdr"
    EU_TAXONOMY = "eu_taxonomy"
    SOLVENCY_II = "solvency_ii"
    IORP_II = "iorp_ii"
    CRR_CRD = "crr_crd"
    PCAF = "pcaf"

class PCAFAssetClass(str, Enum):
    """PCAF asset classes for financed emissions."""
    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    MOTOR_VEHICLE_LOANS = "motor_vehicle_loans"
    SOVEREIGN_DEBT = "sovereign_debt"

class CSRDTier(str, Enum):
    """CSRD reporting tier."""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

# =============================================================================
# Data Models
# =============================================================================

class SetupWizardConfig(BaseModel):
    """Configuration for the FI CSRD Setup Wizard."""
    allow_skip_optional: bool = Field(
        default=True,
        description="Allow skipping optional steps",
    )
    auto_populate_defaults: bool = Field(
        default=True,
        description="Auto-populate default values based on institution type",
    )
    validate_on_each_step: bool = Field(
        default=True,
        description="Validate data at each step",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking",
    )
    default_csrd_tier: str = Field(
        default="professional",
        description="Default CSRD pack tier",
    )
    default_pcaf_version: str = Field(
        default="2022",
        description="Default PCAF standard version",
    )
    default_reporting_currency: str = Field(
        default="EUR",
        description="Default reporting currency",
    )

class WizardStep(BaseModel):
    """Definition and state of a single wizard step."""
    step_id: str = Field(default="", description="Step identifier")
    step_number: int = Field(default=0, ge=1, le=8, description="Step number")
    step_name: str = Field(default="", description="Human-readable step name")
    description: str = Field(default="", description="Step description")
    status: str = Field(default="pending", description="Step status")
    is_optional: bool = Field(
        default=False, description="Whether step can be skipped",
    )
    is_conditional: bool = Field(
        default=False,
        description="Whether step is conditional on previous answers",
    )
    condition_met: bool = Field(
        default=True, description="Whether condition is met",
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Step input data",
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Step output data",
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors",
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings",
    )

class InstitutionInfo(BaseModel):
    """Institution information collected in Step 1."""
    institution_name: str = Field(
        default="", description="Institution legal name",
    )
    institution_type: str = Field(
        default="bank", description="Institution type",
    )
    lei_code: str = Field(default="", description="LEI code")
    domicile_country: str = Field(
        default="", description="Domicile country (ISO 3166)",
    )
    reporting_currency: str = Field(
        default="EUR", description="Reporting currency",
    )
    total_assets_eur: float = Field(
        default=0.0, description="Total assets in EUR (for size classification)",
    )
    employee_count: int = Field(
        default=0, description="Number of employees (for CSRD scope)",
    )
    is_listed: bool = Field(
        default=False, description="Whether institution is publicly listed",
    )
    fiscal_year_end: str = Field(
        default="12-31", description="Fiscal year end (MM-DD)",
    )
    parent_company: str = Field(
        default="", description="Parent company name (if subsidiary)",
    )
    consolidation_scope: str = Field(
        default="solo",
        description="Consolidation scope: solo, sub-consolidated, consolidated",
    )

class RegulatoryScope(BaseModel):
    """Regulatory scope determined in Step 2."""
    applicable_frameworks: List[str] = Field(
        default_factory=list,
        description="List of applicable regulatory frameworks",
    )
    csrd_tier: str = Field(
        default="professional", description="CSRD reporting tier",
    )
    gar_applicable: bool = Field(
        default=True, description="Green Asset Ratio applicable",
    )
    btar_applicable: bool = Field(
        default=True, description="BTAR applicable",
    )
    pillar3_applicable: bool = Field(
        default=True, description="EBA Pillar 3 ESG applicable",
    )
    sfdr_applicable: bool = Field(
        default=False, description="SFDR applicable",
    )
    pcaf_applicable: bool = Field(
        default=True, description="PCAF financed emissions applicable",
    )
    solvency_ii_applicable: bool = Field(
        default=False, description="Solvency II applicable (insurers)",
    )
    reporting_start_date: str = Field(
        default="", description="First CSRD reporting period start",
    )
    first_filing_date: str = Field(
        default="", description="First CSRD filing deadline",
    )

class DataSourceConfig(BaseModel):
    """Data source configuration from Step 3."""
    counterparty_data_source: str = Field(
        default="internal",
        description="Source: internal, external_provider, hybrid",
    )
    emissions_data_source: str = Field(
        default="estimated",
        description="Source: reported, estimated, hybrid",
    )
    taxonomy_data_source: str = Field(
        default="internal",
        description="Source: internal, eet_template, external_provider",
    )
    real_estate_data_source: str = Field(
        default="internal",
        description="Source: internal, epc_registry, external",
    )
    climate_scenario_provider: str = Field(
        default="ngfs",
        description="Provider: ngfs, internal, third_party",
    )
    pcaf_asset_classes: List[str] = Field(
        default_factory=lambda: [
            PCAFAssetClass.LISTED_EQUITY.value,
            PCAFAssetClass.CORPORATE_BONDS.value,
            PCAFAssetClass.BUSINESS_LOANS.value,
        ],
        description="PCAF asset classes in portfolio",
    )
    esg_data_providers: List[str] = Field(
        default_factory=list,
        description="External ESG data providers (e.g., MSCI, Sustainalytics)",
    )

class EngineConfig(BaseModel):
    """Engine configuration from Step 4."""
    financed_emissions_enabled: bool = Field(
        default=True, description="Enable financed emissions engine",
    )
    insurance_underwriting_enabled: bool = Field(
        default=False, description="Enable insurance underwriting engine",
    )
    gar_enabled: bool = Field(
        default=True, description="Enable GAR engine",
    )
    btar_enabled: bool = Field(
        default=True, description="Enable BTAR engine",
    )
    climate_risk_enabled: bool = Field(
        default=True, description="Enable climate risk scoring engine",
    )
    double_materiality_enabled: bool = Field(
        default=True, description="Enable FS double materiality engine",
    )
    transition_plan_enabled: bool = Field(
        default=True, description="Enable FS transition plan engine",
    )
    pillar3_enabled: bool = Field(
        default=True, description="Enable Pillar 3 ESG engine",
    )
    pcaf_version: str = Field(
        default="2022", description="PCAF standard version",
    )
    climate_scenarios: List[str] = Field(
        default_factory=lambda: [
            "ngfs_current_policies",
            "ngfs_net_zero_2050",
            "ngfs_delayed_transition",
        ],
        description="Climate scenarios to evaluate",
    )
    transition_horizons: List[str] = Field(
        default_factory=lambda: ["2030", "2040", "2050"],
        description="Transition plan target horizons",
    )
    quality_gate_threshold: float = Field(
        default=60.0, ge=0.0, le=100.0,
        description="Minimum quality gate pass score",
    )

class WorkflowConfig(BaseModel):
    """Workflow configuration from Step 5."""
    parallel_execution: bool = Field(
        default=True, description="Enable parallel workflow execution",
    )
    max_concurrent_workflows: int = Field(
        default=4, ge=1, le=16, description="Max concurrent workflows",
    )
    retry_on_failure: bool = Field(
        default=True, description="Auto-retry failed workflows",
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Max retry attempts",
    )
    approval_required: bool = Field(
        default=False,
        description="Require manual approval before disclosure generation",
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email"],
        description="Notification channels: email, slack, teams, webhook",
    )

class TemplateSelection(BaseModel):
    """Template selection from Step 6."""
    financed_emissions_template: bool = Field(
        default=True, description="Include financed emissions report",
    )
    gar_template: bool = Field(
        default=True, description="Include GAR disclosure template",
    )
    btar_template: bool = Field(
        default=True, description="Include BTAR disclosure template",
    )
    climate_risk_template: bool = Field(
        default=True, description="Include climate risk report",
    )
    double_materiality_template: bool = Field(
        default=True, description="Include DMA matrix template",
    )
    transition_plan_template: bool = Field(
        default=True, description="Include transition plan template",
    )
    pillar3_template_set: bool = Field(
        default=True, description="Include Pillar 3 ESG template set",
    )
    esrs_fi_supplement: bool = Field(
        default=True,
        description="Include FI-specific ESRS supplement",
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ["pdf", "xlsx", "xbrl"],
        description="Output format options: pdf, xlsx, xbrl, html, json",
    )

class IntegrationSetup(BaseModel):
    """Integration setup from Step 7."""
    csrd_pack_enabled: bool = Field(
        default=True, description="Enable CSRD pack bridge",
    )
    csrd_pack_tier: str = Field(
        default="professional", description="CSRD pack tier to bridge",
    )
    sfdr_pack_enabled: bool = Field(
        default=False, description="Enable SFDR pack bridge",
    )
    taxonomy_pack_enabled: bool = Field(
        default=True, description="Enable Taxonomy pack bridge",
    )
    mrv_bridge_enabled: bool = Field(
        default=True, description="Enable MRV investments bridge",
    )
    climate_risk_bridge_enabled: bool = Field(
        default=True, description="Enable climate risk bridge",
    )
    eba_pillar3_bridge_enabled: bool = Field(
        default=True, description="Enable EBA Pillar 3 bridge",
    )
    finance_agent_bridge_enabled: bool = Field(
        default=True, description="Enable finance agent bridge",
    )

class WizardResult(BaseModel):
    """Complete result of the setup wizard execution."""
    wizard_id: str = Field(default="", description="Wizard execution ID")
    started_at: str = Field(default="", description="Wizard start timestamp")
    completed_at: str = Field(
        default="", description="Wizard completion timestamp",
    )

    # Overall status
    is_valid: bool = Field(
        default=False,
        description="Whether all required steps passed validation",
    )
    config_ready: bool = Field(
        default=False,
        description="Whether configuration is ready for pipeline",
    )
    total_steps: int = Field(default=8, description="Total wizard steps")
    completed_steps: int = Field(default=0, description="Completed steps")
    skipped_steps: int = Field(default=0, description="Skipped steps")
    failed_steps: int = Field(default=0, description="Failed steps")

    # Per-step results
    steps: List[WizardStep] = Field(
        default_factory=list, description="Per-step state and results",
    )

    # Collected data
    institution_info: Optional[InstitutionInfo] = Field(
        default=None, description="Institution information (Step 1)",
    )
    regulatory_scope: Optional[RegulatoryScope] = Field(
        default=None, description="Regulatory scope (Step 2)",
    )
    data_source_config: Optional[DataSourceConfig] = Field(
        default=None, description="Data source configuration (Step 3)",
    )
    engine_config: Optional[EngineConfig] = Field(
        default=None, description="Engine configuration (Step 4)",
    )
    workflow_config: Optional[WorkflowConfig] = Field(
        default=None, description="Workflow configuration (Step 5)",
    )
    template_selection: Optional[TemplateSelection] = Field(
        default=None, description="Template selection (Step 6)",
    )
    integration_setup: Optional[IntegrationSetup] = Field(
        default=None, description="Integration setup (Step 7)",
    )

    # Generated pipeline config
    pipeline_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated FSOrchestrationConfig parameters",
    )

    # Metadata
    validation_errors: List[str] = Field(
        default_factory=list, description="All validation errors",
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="All validation warnings",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )
    execution_time_ms: float = Field(
        default=0.0, description="Total execution time",
    )

# =============================================================================
# Step Definitions
# =============================================================================

STEP_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "step_id": WizardStepId.INSTITUTION_TYPE.value,
        "step_number": 1,
        "step_name": "Institution Type Selection",
        "description": (
            "Select the financial institution type (bank, insurer, "
            "asset manager, etc.) which determines applicable regulations "
            "and default configurations"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.REGULATORY_SCOPE.value,
        "step_number": 2,
        "step_name": "Regulatory Scope",
        "description": (
            "Determine applicable regulatory frameworks based on "
            "institution type, size, and jurisdiction (CSRD, EBA Pillar 3, "
            "SFDR, EU Taxonomy, PCAF)"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.DATA_SOURCE_CONFIG.value,
        "step_number": 3,
        "step_name": "Data Source Configuration",
        "description": (
            "Configure data sources for counterparty data, emissions data, "
            "taxonomy alignment, real estate EPCs, and climate scenarios"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.ENGINE_CONFIG.value,
        "step_number": 4,
        "step_name": "Engine Configuration",
        "description": (
            "Configure which engines to enable and their parameters: "
            "financed emissions, GAR, BTAR, climate risk, double materiality, "
            "transition plan, Pillar 3 ESG"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.WORKFLOW_CONFIG.value,
        "step_number": 5,
        "step_name": "Workflow Configuration",
        "description": (
            "Configure workflow execution: parallelism, retries, "
            "approval gates, and notification channels"
        ),
        "is_optional": True,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.TEMPLATE_SELECTION.value,
        "step_number": 6,
        "step_name": "Template Selection",
        "description": (
            "Select disclosure templates and output formats: "
            "financed emissions, GAR, BTAR, climate risk, DMA, "
            "transition plan, Pillar 3 ESG"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.INTEGRATION_SETUP.value,
        "step_number": 7,
        "step_name": "Integration Setup",
        "description": (
            "Configure cross-pack bridges and external integrations: "
            "CSRD pack, SFDR pack, Taxonomy pack, MRV bridge, "
            "climate risk bridge, EBA Pillar 3 bridge"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.REVIEW_VALIDATE.value,
        "step_number": 8,
        "step_name": "Review & Validate",
        "description": (
            "Review all configuration, cross-validate settings, "
            "and generate the FSOrchestrationConfig for pipeline execution"
        ),
        "is_optional": False,
        "is_conditional": False,
    },
]

# Institution type guidance
INSTITUTION_TYPE_GUIDANCE: Dict[str, Dict[str, Any]] = {
    InstitutionType.BANK.value: {
        "name": "Credit Institution (Bank)",
        "description": (
            "Banks under CRR/CRD with full GAR, BTAR, Pillar 3 ESG, "
            "and financed emissions obligations"
        ),
        "default_frameworks": [
            RegulatoryFramework.CSRD.value,
            RegulatoryFramework.EBA_PILLAR3.value,
            RegulatoryFramework.EU_TAXONOMY.value,
            RegulatoryFramework.CRR_CRD.value,
            RegulatoryFramework.PCAF.value,
        ],
        "gar_applicable": True,
        "btar_applicable": True,
        "pillar3_applicable": True,
        "sfdr_applicable": False,
        "pcaf_applicable": True,
    },
    InstitutionType.INSURER.value: {
        "name": "Insurance / Reinsurance Undertaking",
        "description": (
            "Insurers under Solvency II with underwriting emissions, "
            "climate risk, and CSRD obligations"
        ),
        "default_frameworks": [
            RegulatoryFramework.CSRD.value,
            RegulatoryFramework.EU_TAXONOMY.value,
            RegulatoryFramework.SOLVENCY_II.value,
            RegulatoryFramework.PCAF.value,
        ],
        "gar_applicable": True,
        "btar_applicable": False,
        "pillar3_applicable": False,
        "sfdr_applicable": False,
        "pcaf_applicable": True,
    },
    InstitutionType.ASSET_MANAGER.value: {
        "name": "Asset Management Company",
        "description": (
            "Asset managers with SFDR, financed emissions, and CSRD obligations"
        ),
        "default_frameworks": [
            RegulatoryFramework.CSRD.value,
            RegulatoryFramework.SFDR.value,
            RegulatoryFramework.EU_TAXONOMY.value,
            RegulatoryFramework.PCAF.value,
        ],
        "gar_applicable": False,
        "btar_applicable": False,
        "pillar3_applicable": False,
        "sfdr_applicable": True,
        "pcaf_applicable": True,
    },
    InstitutionType.INVESTMENT_FIRM.value: {
        "name": "Investment Firm",
        "description": (
            "Investment firms under IFR/IFD with simplified GAR "
            "and financed emissions"
        ),
        "default_frameworks": [
            RegulatoryFramework.CSRD.value,
            RegulatoryFramework.EU_TAXONOMY.value,
            RegulatoryFramework.PCAF.value,
        ],
        "gar_applicable": True,
        "btar_applicable": False,
        "pillar3_applicable": False,
        "sfdr_applicable": False,
        "pcaf_applicable": True,
    },
    InstitutionType.PENSION_FUND.value: {
        "name": "Pension Fund / IORP",
        "description": (
            "Pension funds under IORP II with SFDR, financed emissions, "
            "and CSRD obligations"
        ),
        "default_frameworks": [
            RegulatoryFramework.CSRD.value,
            RegulatoryFramework.SFDR.value,
            RegulatoryFramework.EU_TAXONOMY.value,
            RegulatoryFramework.IORP_II.value,
            RegulatoryFramework.PCAF.value,
        ],
        "gar_applicable": False,
        "btar_applicable": False,
        "pillar3_applicable": False,
        "sfdr_applicable": True,
        "pcaf_applicable": True,
    },
    InstitutionType.DEVELOPMENT_BANK.value: {
        "name": "Development Bank / DFI",
        "description": (
            "Development finance institutions with impact reporting, "
            "financed emissions, and CSRD"
        ),
        "default_frameworks": [
            RegulatoryFramework.CSRD.value,
            RegulatoryFramework.EU_TAXONOMY.value,
            RegulatoryFramework.PCAF.value,
        ],
        "gar_applicable": True,
        "btar_applicable": True,
        "pillar3_applicable": False,
        "sfdr_applicable": False,
        "pcaf_applicable": True,
    },
}

# =============================================================================
# Setup Wizard
# =============================================================================

class SetupWizard:
    """8-step guided configuration wizard for FI CSRD reporting.

    Walks through institution setup, regulatory scope determination,
    data source configuration, engine and workflow setup, template
    selection, integration configuration, and final review/validation.

    The wizard adapts steps, defaults, and validation rules based on the
    institution type selected in Step 1.

    Attributes:
        config: Wizard configuration.
        _steps: Current step states.
        _collected: Collected data across steps.

    Example:
        >>> wizard = SetupWizard(SetupWizardConfig())
        >>> result = wizard.execute_all_steps({"institution_name": "GL Bank"})
        >>> print(f"Valid: {result.is_valid}")
    """

    def __init__(self, config: Optional[SetupWizardConfig] = None) -> None:
        """Initialize the Setup Wizard.

        Args:
            config: Wizard configuration. Uses defaults if not provided.
        """
        self.config = config or SetupWizardConfig()
        self.logger = logger
        self._steps: List[WizardStep] = self._initialize_steps()
        self._collected: Dict[str, Any] = {}
        self._institution_type: str = InstitutionType.BANK.value

        self.logger.info(
            "SetupWizard initialized: steps=%d, auto_populate=%s, "
            "validate_each=%s",
            len(self._steps),
            self.config.auto_populate_defaults,
            self.config.validate_on_each_step,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def execute_all_steps(
        self,
        institution_data: Dict[str, Any],
    ) -> WizardResult:
        """Execute all 8 wizard steps with provided institution data.

        Args:
            institution_data: Combined institution data for all steps.

        Returns:
            WizardResult with configuration and validation status.
        """
        start_time = time.time()
        self._collected = dict(institution_data)

        self._execute_step_1_institution_type(institution_data)
        self._execute_step_2_regulatory_scope(institution_data)
        self._execute_step_3_data_sources(institution_data)
        self._execute_step_4_engine_config(institution_data)
        self._execute_step_5_workflow_config(institution_data)
        self._execute_step_6_template_selection(institution_data)
        self._execute_step_7_integration_setup(institution_data)
        self._execute_step_8_review()

        elapsed_ms = (time.time() - start_time) * 1000
        return self._build_result(elapsed_ms)

    def execute_step(
        self,
        step_id: str,
        step_data: Dict[str, Any],
    ) -> WizardStep:
        """Execute a single wizard step.

        Args:
            step_id: Step identifier from WizardStepId enum.
            step_data: Data for this step.

        Returns:
            Updated WizardStep with status and output.
        """
        self._collected.update(step_data)

        step_handlers = {
            WizardStepId.INSTITUTION_TYPE.value: self._execute_step_1_institution_type,
            WizardStepId.REGULATORY_SCOPE.value: self._execute_step_2_regulatory_scope,
            WizardStepId.DATA_SOURCE_CONFIG.value: self._execute_step_3_data_sources,
            WizardStepId.ENGINE_CONFIG.value: self._execute_step_4_engine_config,
            WizardStepId.WORKFLOW_CONFIG.value: self._execute_step_5_workflow_config,
            WizardStepId.TEMPLATE_SELECTION.value: self._execute_step_6_template_selection,
            WizardStepId.INTEGRATION_SETUP.value: self._execute_step_7_integration_setup,
            WizardStepId.REVIEW_VALIDATE.value: self._execute_step_8_review,
        }

        handler = step_handlers.get(step_id)
        if handler:
            handler(step_data)

        step = self._get_step(step_id)
        return step if step else WizardStep(step_id=step_id)

    def get_step_guidance(self, step_id: str) -> Dict[str, Any]:
        """Get guidance information for a specific step.

        Args:
            step_id: Step identifier.

        Returns:
            Guidance data with descriptions, options, and defaults.
        """
        if step_id == WizardStepId.INSTITUTION_TYPE.value:
            return {
                "step_name": "Institution Type Selection",
                "options": INSTITUTION_TYPE_GUIDANCE,
                "required_fields": ["institution_name", "institution_type"],
            }
        elif step_id == WizardStepId.REGULATORY_SCOPE.value:
            return {
                "step_name": "Regulatory Scope",
                "frameworks": [f.value for f in RegulatoryFramework],
                "note": "Frameworks auto-selected based on institution type",
            }
        elif step_id == WizardStepId.DATA_SOURCE_CONFIG.value:
            return {
                "step_name": "Data Source Configuration",
                "pcaf_asset_classes": [a.value for a in PCAFAssetClass],
                "source_options": ["internal", "external_provider", "hybrid"],
            }
        elif step_id == WizardStepId.ENGINE_CONFIG.value:
            return {
                "step_name": "Engine Configuration",
                "engines": [
                    "financed_emissions", "insurance_underwriting",
                    "green_asset_ratio", "btar_calculator",
                    "climate_risk_scoring", "fs_double_materiality",
                    "fs_transition_plan", "pillar3_esg",
                ],
                "note": "Engines auto-enabled based on regulatory scope",
            }
        elif step_id == WizardStepId.WORKFLOW_CONFIG.value:
            return {
                "step_name": "Workflow Configuration",
                "notification_options": ["email", "slack", "teams", "webhook"],
            }
        elif step_id == WizardStepId.TEMPLATE_SELECTION.value:
            return {
                "step_name": "Template Selection",
                "output_formats": ["pdf", "xlsx", "xbrl", "html", "json"],
            }
        elif step_id == WizardStepId.INTEGRATION_SETUP.value:
            return {
                "step_name": "Integration Setup",
                "available_bridges": [
                    "csrd_pack", "sfdr_pack", "taxonomy_pack",
                    "mrv_bridge", "climate_risk_bridge",
                    "eba_pillar3_bridge", "finance_agent_bridge",
                ],
            }
        return {"step_name": step_id}

    def get_progress(self) -> Dict[str, Any]:
        """Get current wizard progress.

        Returns:
            Progress data with per-step status.
        """
        completed = sum(
            1 for s in self._steps
            if s.status == StepStatus.COMPLETED.value
        )
        skipped = sum(
            1 for s in self._steps
            if s.status == StepStatus.SKIPPED.value
        )
        failed = sum(
            1 for s in self._steps
            if s.status == StepStatus.FAILED.value
        )

        return {
            "total_steps": len(self._steps),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "pending": len(self._steps) - completed - skipped - failed,
            "progress_pct": (
                (completed / len(self._steps)) * 100.0
                if self._steps else 0.0
            ),
            "institution_type": self._institution_type,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_number": s.step_number,
                    "step_name": s.step_name,
                    "status": s.status,
                    "is_optional": s.is_optional,
                }
                for s in self._steps
            ],
        }

    def reset(self) -> None:
        """Reset the wizard to initial state."""
        self._steps = self._initialize_steps()
        self._collected = {}
        self._institution_type = InstitutionType.BANK.value
        self.logger.info("SetupWizard reset to initial state")

    # -------------------------------------------------------------------------
    # Step Execution Methods
    # -------------------------------------------------------------------------

    def _execute_step_1_institution_type(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 1: Institution type and basic information."""
        step = self._get_step(WizardStepId.INSTITUTION_TYPE.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []
        warnings: List[str] = []

        institution_name = str(data.get("institution_name", ""))
        institution_type = str(
            data.get("institution_type", InstitutionType.BANK.value)
        )

        if not institution_name and self.config.validate_on_each_step:
            errors.append("Institution name is required")

        valid_types = [t.value for t in InstitutionType]
        if institution_type not in valid_types:
            warnings.append(
                f"Unknown institution type: {institution_type}; "
                f"defaulting to bank"
            )
            institution_type = InstitutionType.BANK.value

        self._institution_type = institution_type

        info = InstitutionInfo(
            institution_name=institution_name,
            institution_type=institution_type,
            lei_code=str(data.get("lei_code", "")),
            domicile_country=str(data.get("domicile_country", "")),
            reporting_currency=str(
                data.get(
                    "reporting_currency",
                    self.config.default_reporting_currency,
                )
            ),
            total_assets_eur=float(data.get("total_assets_eur", 0.0)),
            employee_count=int(data.get("employee_count", 0)),
            is_listed=bool(data.get("is_listed", False)),
            fiscal_year_end=str(data.get("fiscal_year_end", "12-31")),
            parent_company=str(data.get("parent_company", "")),
            consolidation_scope=str(
                data.get("consolidation_scope", "solo")
            ),
        )

        step.output_data = {"institution_info": info.model_dump()}
        step.validation_errors = errors
        step.validation_warnings = warnings
        step.status = (
            StepStatus.FAILED.value if errors
            else StepStatus.COMPLETED.value
        )

    def _execute_step_2_regulatory_scope(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 2: Determine regulatory scope based on institution type."""
        step = self._get_step(WizardStepId.REGULATORY_SCOPE.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []

        guidance = INSTITUTION_TYPE_GUIDANCE.get(
            self._institution_type,
            INSTITUTION_TYPE_GUIDANCE[InstitutionType.BANK.value],
        )

        frameworks = data.get(
            "applicable_frameworks",
            guidance.get("default_frameworks", []),
        )

        now = utcnow()
        scope = RegulatoryScope(
            applicable_frameworks=frameworks,
            csrd_tier=str(data.get("csrd_tier", self.config.default_csrd_tier)),
            gar_applicable=bool(
                data.get("gar_applicable", guidance.get("gar_applicable", True))
            ),
            btar_applicable=bool(
                data.get("btar_applicable", guidance.get("btar_applicable", False))
            ),
            pillar3_applicable=bool(
                data.get(
                    "pillar3_applicable",
                    guidance.get("pillar3_applicable", False),
                )
            ),
            sfdr_applicable=bool(
                data.get(
                    "sfdr_applicable",
                    guidance.get("sfdr_applicable", False),
                )
            ),
            pcaf_applicable=bool(
                data.get(
                    "pcaf_applicable",
                    guidance.get("pcaf_applicable", True),
                )
            ),
            solvency_ii_applicable=bool(
                self._institution_type == InstitutionType.INSURER.value
            ),
            reporting_start_date=str(
                data.get("reporting_start_date", f"{now.year}-01-01")
            ),
            first_filing_date=str(
                data.get("first_filing_date", f"{now.year + 1}-04-30")
            ),
        )

        if not frameworks:
            errors.append("At least one regulatory framework must be selected")

        step.output_data = {"regulatory_scope": scope.model_dump()}
        step.validation_errors = errors
        step.status = (
            StepStatus.FAILED.value if errors
            else StepStatus.COMPLETED.value
        )

    def _execute_step_3_data_sources(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 3: Data source configuration."""
        step = self._get_step(WizardStepId.DATA_SOURCE_CONFIG.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        warnings: List[str] = []

        default_classes = [
            PCAFAssetClass.LISTED_EQUITY.value,
            PCAFAssetClass.CORPORATE_BONDS.value,
            PCAFAssetClass.BUSINESS_LOANS.value,
        ]

        ds_config = DataSourceConfig(
            counterparty_data_source=str(
                data.get("counterparty_data_source", "internal")
            ),
            emissions_data_source=str(
                data.get("emissions_data_source", "estimated")
            ),
            taxonomy_data_source=str(
                data.get("taxonomy_data_source", "internal")
            ),
            real_estate_data_source=str(
                data.get("real_estate_data_source", "internal")
            ),
            climate_scenario_provider=str(
                data.get("climate_scenario_provider", "ngfs")
            ),
            pcaf_asset_classes=data.get(
                "pcaf_asset_classes", default_classes
            ),
            esg_data_providers=data.get("esg_data_providers", []),
        )

        if ds_config.emissions_data_source == "estimated":
            warnings.append(
                "Using estimated emissions data; consider hybrid or reported "
                "sources for higher PCAF data quality scores"
            )

        if not ds_config.esg_data_providers:
            warnings.append(
                "No external ESG data providers configured; "
                "some data enrichment features will be limited"
            )

        step.output_data = {"data_source_config": ds_config.model_dump()}
        step.validation_warnings = warnings
        step.status = StepStatus.COMPLETED.value

    def _execute_step_4_engine_config(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 4: Engine configuration based on regulatory scope."""
        step = self._get_step(WizardStepId.ENGINE_CONFIG.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        warnings: List[str] = []

        # Get scope from step 2
        scope_step = self._get_step(WizardStepId.REGULATORY_SCOPE.value)
        scope_data = {}
        if scope_step and scope_step.output_data:
            scope_data = scope_step.output_data.get("regulatory_scope", {})

        gar_app = scope_data.get("gar_applicable", True)
        btar_app = scope_data.get("btar_applicable", True)
        p3_app = scope_data.get("pillar3_applicable", True)

        eng_config = EngineConfig(
            financed_emissions_enabled=bool(
                data.get("financed_emissions_enabled", True)
            ),
            insurance_underwriting_enabled=bool(
                data.get(
                    "insurance_underwriting_enabled",
                    self._institution_type == InstitutionType.INSURER.value,
                )
            ),
            gar_enabled=bool(data.get("gar_enabled", gar_app)),
            btar_enabled=bool(data.get("btar_enabled", btar_app)),
            climate_risk_enabled=bool(
                data.get("climate_risk_enabled", True)
            ),
            double_materiality_enabled=bool(
                data.get("double_materiality_enabled", True)
            ),
            transition_plan_enabled=bool(
                data.get("transition_plan_enabled", True)
            ),
            pillar3_enabled=bool(data.get("pillar3_enabled", p3_app)),
            pcaf_version=str(
                data.get("pcaf_version", self.config.default_pcaf_version)
            ),
            climate_scenarios=data.get("climate_scenarios", [
                "ngfs_current_policies",
                "ngfs_net_zero_2050",
                "ngfs_delayed_transition",
            ]),
            transition_horizons=data.get(
                "transition_horizons", ["2030", "2040", "2050"]
            ),
            quality_gate_threshold=float(
                data.get("quality_gate_threshold", 60.0)
            ),
        )

        if eng_config.quality_gate_threshold < 50.0:
            warnings.append(
                "Quality gate threshold below 50% may produce "
                "low-confidence disclosures"
            )

        step.output_data = {"engine_config": eng_config.model_dump()}
        step.validation_warnings = warnings
        step.status = StepStatus.COMPLETED.value

    def _execute_step_5_workflow_config(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 5: Workflow configuration."""
        step = self._get_step(WizardStepId.WORKFLOW_CONFIG.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value

        wf_config = WorkflowConfig(
            parallel_execution=bool(
                data.get("parallel_execution", True)
            ),
            max_concurrent_workflows=int(
                data.get("max_concurrent_workflows", 4)
            ),
            retry_on_failure=bool(data.get("retry_on_failure", True)),
            max_retries=int(data.get("max_retries", 3)),
            approval_required=bool(data.get("approval_required", False)),
            notification_channels=data.get(
                "notification_channels", ["email"]
            ),
        )

        step.output_data = {"workflow_config": wf_config.model_dump()}
        step.status = StepStatus.COMPLETED.value

    def _execute_step_6_template_selection(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 6: Template selection based on engine configuration."""
        step = self._get_step(WizardStepId.TEMPLATE_SELECTION.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value

        # Get engine config from step 4
        eng_step = self._get_step(WizardStepId.ENGINE_CONFIG.value)
        eng_data = {}
        if eng_step and eng_step.output_data:
            eng_data = eng_step.output_data.get("engine_config", {})

        tpl = TemplateSelection(
            financed_emissions_template=bool(
                data.get(
                    "financed_emissions_template",
                    eng_data.get("financed_emissions_enabled", True),
                )
            ),
            gar_template=bool(
                data.get("gar_template", eng_data.get("gar_enabled", True))
            ),
            btar_template=bool(
                data.get(
                    "btar_template", eng_data.get("btar_enabled", True)
                )
            ),
            climate_risk_template=bool(
                data.get(
                    "climate_risk_template",
                    eng_data.get("climate_risk_enabled", True),
                )
            ),
            double_materiality_template=bool(
                data.get(
                    "double_materiality_template",
                    eng_data.get("double_materiality_enabled", True),
                )
            ),
            transition_plan_template=bool(
                data.get(
                    "transition_plan_template",
                    eng_data.get("transition_plan_enabled", True),
                )
            ),
            pillar3_template_set=bool(
                data.get(
                    "pillar3_template_set",
                    eng_data.get("pillar3_enabled", True),
                )
            ),
            esrs_fi_supplement=bool(
                data.get("esrs_fi_supplement", True)
            ),
            output_formats=data.get(
                "output_formats", ["pdf", "xlsx", "xbrl"]
            ),
        )

        step.output_data = {"template_selection": tpl.model_dump()}
        step.status = StepStatus.COMPLETED.value

    def _execute_step_7_integration_setup(
        self, data: Dict[str, Any],
    ) -> None:
        """Step 7: Integration bridge setup."""
        step = self._get_step(WizardStepId.INTEGRATION_SETUP.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value

        scope_step = self._get_step(WizardStepId.REGULATORY_SCOPE.value)
        scope_data = {}
        if scope_step and scope_step.output_data:
            scope_data = scope_step.output_data.get("regulatory_scope", {})

        integration = IntegrationSetup(
            csrd_pack_enabled=bool(
                data.get("csrd_pack_enabled", True)
            ),
            csrd_pack_tier=str(
                data.get(
                    "csrd_pack_tier",
                    scope_data.get("csrd_tier", "professional"),
                )
            ),
            sfdr_pack_enabled=bool(
                data.get(
                    "sfdr_pack_enabled",
                    scope_data.get("sfdr_applicable", False),
                )
            ),
            taxonomy_pack_enabled=bool(
                data.get("taxonomy_pack_enabled", True)
            ),
            mrv_bridge_enabled=bool(
                data.get("mrv_bridge_enabled", True)
            ),
            climate_risk_bridge_enabled=bool(
                data.get("climate_risk_bridge_enabled", True)
            ),
            eba_pillar3_bridge_enabled=bool(
                data.get(
                    "eba_pillar3_bridge_enabled",
                    scope_data.get("pillar3_applicable", True),
                )
            ),
            finance_agent_bridge_enabled=bool(
                data.get("finance_agent_bridge_enabled", True)
            ),
        )

        step.output_data = {"integration_setup": integration.model_dump()}
        step.status = StepStatus.COMPLETED.value

    def _execute_step_8_review(self) -> None:
        """Step 8: Review and validate all configuration."""
        step = self._get_step(WizardStepId.REVIEW_VALIDATE.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []
        warnings: List[str] = []

        # Validate all required steps completed
        for s in self._steps:
            if s.step_id == WizardStepId.REVIEW_VALIDATE.value:
                continue
            if s.is_optional and s.status == StepStatus.SKIPPED.value:
                continue
            if s.status != StepStatus.COMPLETED.value:
                errors.append(
                    f"Step {s.step_number} ({s.step_name}) not completed: "
                    f"{s.status}"
                )
            errors.extend(s.validation_errors)
            warnings.extend(s.validation_warnings)

        # Cross-step validation
        cross_errors = self._cross_validate_steps()
        errors.extend(cross_errors)

        step.output_data = {
            "review_complete": True,
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "config_ready": len(errors) == 0,
        }
        step.validation_errors = errors
        step.validation_warnings = warnings
        step.status = (
            StepStatus.FAILED.value if errors
            else StepStatus.COMPLETED.value
        )

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _initialize_steps(self) -> List[WizardStep]:
        """Initialize wizard steps from definitions."""
        steps: List[WizardStep] = []
        for step_def in STEP_DEFINITIONS:
            steps.append(WizardStep(
                step_id=step_def["step_id"],
                step_number=step_def["step_number"],
                step_name=step_def["step_name"],
                description=step_def["description"],
                is_optional=step_def.get("is_optional", False),
                is_conditional=step_def.get("is_conditional", False),
            ))
        return steps

    def _get_step(self, step_id: str) -> Optional[WizardStep]:
        """Get a step by its identifier."""
        for step in self._steps:
            if step.step_id == step_id:
                return step
        return None

    def _cross_validate_steps(self) -> List[str]:
        """Cross-validate data across steps for consistency."""
        errors: List[str] = []

        # Check that engines match regulatory scope
        scope_step = self._get_step(WizardStepId.REGULATORY_SCOPE.value)
        eng_step = self._get_step(WizardStepId.ENGINE_CONFIG.value)

        if scope_step and eng_step:
            scope = scope_step.output_data.get("regulatory_scope", {})
            eng = eng_step.output_data.get("engine_config", {})

            if scope.get("pillar3_applicable") and not eng.get("pillar3_enabled"):
                errors.append(
                    "Pillar 3 is applicable but Pillar 3 ESG engine is disabled"
                )
            if scope.get("gar_applicable") and not eng.get("gar_enabled"):
                errors.append(
                    "GAR is applicable but GAR engine is disabled"
                )

        # Check integration bridges match engines
        int_step = self._get_step(WizardStepId.INTEGRATION_SETUP.value)
        if eng_step and int_step:
            eng = eng_step.output_data.get("engine_config", {})
            integ = int_step.output_data.get("integration_setup", {})

            if eng.get("pillar3_enabled") and not integ.get("eba_pillar3_bridge_enabled"):
                errors.append(
                    "Pillar 3 engine enabled but EBA Pillar 3 bridge disabled"
                )

        return errors

    def _build_result(self, elapsed_ms: float) -> WizardResult:
        """Build the complete wizard result."""
        all_errors: List[str] = []
        all_warnings: List[str] = []

        for step in self._steps:
            all_errors.extend(step.validation_errors)
            all_warnings.extend(step.validation_warnings)

        completed = sum(
            1 for s in self._steps
            if s.status == StepStatus.COMPLETED.value
        )
        skipped = sum(
            1 for s in self._steps
            if s.status == StepStatus.SKIPPED.value
        )
        failed = sum(
            1 for s in self._steps
            if s.status == StepStatus.FAILED.value
        )
        is_valid = failed == 0

        # Extract collected data
        institution_info = self._extract_model(
            WizardStepId.INSTITUTION_TYPE.value,
            "institution_info",
            InstitutionInfo,
        )
        regulatory_scope = self._extract_model(
            WizardStepId.REGULATORY_SCOPE.value,
            "regulatory_scope",
            RegulatoryScope,
        )
        data_source_config = self._extract_model(
            WizardStepId.DATA_SOURCE_CONFIG.value,
            "data_source_config",
            DataSourceConfig,
        )
        engine_config = self._extract_model(
            WizardStepId.ENGINE_CONFIG.value,
            "engine_config",
            EngineConfig,
        )
        workflow_config = self._extract_model(
            WizardStepId.WORKFLOW_CONFIG.value,
            "workflow_config",
            WorkflowConfig,
        )
        template_selection = self._extract_model(
            WizardStepId.TEMPLATE_SELECTION.value,
            "template_selection",
            TemplateSelection,
        )
        integration_setup = self._extract_model(
            WizardStepId.INTEGRATION_SETUP.value,
            "integration_setup",
            IntegrationSetup,
        )

        # Generate pipeline config
        pipeline_config = self._generate_pipeline_config(
            institution_info,
            regulatory_scope,
            data_source_config,
            engine_config,
            workflow_config,
            template_selection,
            integration_setup,
        )

        result = WizardResult(
            wizard_id=f"WIZ-FS-{utcnow().strftime('%Y%m%d%H%M%S')}",
            started_at=utcnow().isoformat(),
            completed_at=utcnow().isoformat(),
            is_valid=is_valid,
            config_ready=is_valid,
            total_steps=len(self._steps),
            completed_steps=completed,
            skipped_steps=skipped,
            failed_steps=failed,
            steps=list(self._steps),
            institution_info=institution_info,
            regulatory_scope=regulatory_scope,
            data_source_config=data_source_config,
            engine_config=engine_config,
            workflow_config=workflow_config,
            template_selection=template_selection,
            integration_setup=integration_setup,
            pipeline_config=pipeline_config,
            validation_errors=all_errors,
            validation_warnings=all_warnings,
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(exclude={"provenance_hash", "steps"})
            )

        self.logger.info(
            "SetupWizard complete: institution=%s, valid=%s, "
            "completed=%d/%d, errors=%d, warnings=%d, elapsed=%.1fms",
            self._institution_type, is_valid,
            completed, len(self._steps),
            len(all_errors), len(all_warnings), elapsed_ms,
        )
        return result

    def _extract_model(
        self,
        step_id: str,
        data_key: str,
        model_class: type,
    ) -> Optional[Any]:
        """Extract a Pydantic model from step output data."""
        step = self._get_step(step_id)
        if step is None or not step.output_data:
            return None
        raw = step.output_data.get(data_key)
        if raw is None or raw is True:
            return None
        try:
            return model_class(**raw)
        except Exception:
            return None

    def _generate_pipeline_config(
        self,
        institution_info: Optional[InstitutionInfo],
        regulatory_scope: Optional[RegulatoryScope],
        data_source_config: Optional[DataSourceConfig],
        engine_config: Optional[EngineConfig],
        workflow_config: Optional[WorkflowConfig],
        template_selection: Optional[TemplateSelection],
        integration_setup: Optional[IntegrationSetup],
    ) -> Dict[str, Any]:
        """Generate FSOrchestrationConfig parameters from wizard data."""
        config: Dict[str, Any] = {
            "pack_id": "PACK-012",
        }

        if institution_info:
            config["institution_name"] = institution_info.institution_name
            config["institution_type"] = institution_info.institution_type
            config["lei_code"] = institution_info.lei_code
            config["reporting_currency"] = institution_info.reporting_currency
            config["consolidation_scope"] = institution_info.consolidation_scope

        if regulatory_scope:
            config["gar_applicable"] = regulatory_scope.gar_applicable
            config["btar_applicable"] = regulatory_scope.btar_applicable
            config["pillar3_applicable"] = regulatory_scope.pillar3_applicable
            config["sfdr_applicable"] = regulatory_scope.sfdr_applicable
            config["pcaf_applicable"] = regulatory_scope.pcaf_applicable
            config["csrd_tier"] = regulatory_scope.csrd_tier

        if data_source_config:
            config["pcaf_asset_classes"] = data_source_config.pcaf_asset_classes
            config["emissions_data_source"] = data_source_config.emissions_data_source
            config["climate_scenario_provider"] = data_source_config.climate_scenario_provider

        if engine_config:
            config["pcaf_version"] = engine_config.pcaf_version
            config["climate_risk_scenarios"] = engine_config.climate_scenarios
            config["transition_plan_horizons"] = engine_config.transition_horizons
            config["quality_gate_threshold"] = engine_config.quality_gate_threshold

        if workflow_config:
            config["parallel_execution"] = workflow_config.parallel_execution
            config["max_retries"] = workflow_config.max_retries

        if template_selection:
            config["output_formats"] = template_selection.output_formats

        if integration_setup:
            config["csrd_pack_tier"] = integration_setup.csrd_pack_tier
            config["sfdr_bridge_enabled"] = integration_setup.sfdr_pack_enabled

        return config
