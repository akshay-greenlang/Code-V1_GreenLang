# -*- coding: utf-8 -*-
"""
SetupWizard - 8-Step Guided Configuration for SFDR Article 9 Products
======================================================================

This module implements an 8-step guided configuration wizard for setting
up SFDR Article 9 financial products within PACK-011. The wizard walks
through product type selection, Article 9 sub-type classification,
sustainable investment objective definition, benchmark designation,
PAI indicator configuration, impact KPI selection, disclosure calendar
setup, and review/validation.

Architecture:
    User --> SetupWizard (8 Steps) --> Article9OrchestrationConfig
                  |
                  v
    Step 1: Product Type
    Step 2: Article 9 Sub-Type (9(1)/9(2)/9(3))
    Step 3: Sustainable Investment Objective
    Step 4: Benchmark Designation (if 9(3))
    Step 5: PAI Configuration (18 mandatory + optional)
    Step 6: Impact KPIs
    Step 7: Disclosure Calendar
    Step 8: Review & Validate

Regulatory Context:
    Article 9 SFDR product setup requires careful classification:
    - Art 9(1): Environmental objective products
    - Art 9(2): Social objective or combined products
    - Art 9(3): Carbon emissions reduction with CTB/PAB benchmark
    Each sub-type has different disclosure requirements, benchmark
    obligations, and PAI reporting emphasis.

Example:
    >>> config = SetupWizardConfig()
    >>> wizard = SetupWizard(config)
    >>> result = wizard.execute_all_steps(product_data)
    >>> print(f"Valid: {result.is_valid}, config ready: {result.config_ready}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
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

from pydantic import BaseModel, Field, field_validator

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
    PRODUCT_TYPE = "product_type"
    ARTICLE_9_SUB_TYPE = "article_9_sub_type"
    SUSTAINABLE_OBJECTIVE = "sustainable_objective"
    BENCHMARK_DESIGNATION = "benchmark_designation"
    PAI_CONFIGURATION = "pai_configuration"
    IMPACT_KPIS = "impact_kpis"
    DISCLOSURE_CALENDAR = "disclosure_calendar"
    REVIEW_VALIDATE = "review_validate"

class ProductType(str, Enum):
    """Financial product types eligible for Article 9."""
    UCITS_FUND = "ucits_fund"
    AIF = "aif"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    INSURANCE_PRODUCT = "insurance_product"
    PENSION_PRODUCT = "pension_product"
    STRUCTURED_DEPOSIT = "structured_deposit"
    OTHER = "other"

class Article9SubType(str, Enum):
    """Article 9 product sub-types."""
    ART_9_1 = "art_9_1"
    ART_9_2 = "art_9_2"
    ART_9_3 = "art_9_3"

class ObjectiveCategory(str, Enum):
    """Sustainable investment objective categories."""
    CLIMATE_CHANGE_MITIGATION = "climate_change_mitigation"
    CLIMATE_CHANGE_ADAPTATION = "climate_change_adaptation"
    SUSTAINABLE_USE_WATER = "sustainable_use_water"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"
    CARBON_EMISSIONS_REDUCTION = "carbon_emissions_reduction"
    SOCIAL_OBJECTIVE = "social_objective"
    COMBINED = "combined"

class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"

class DisclosureFrequency(str, Enum):
    """Disclosure publication frequency."""
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"

# =============================================================================
# Data Models
# =============================================================================

class SetupWizardConfig(BaseModel):
    """Configuration for the Setup Wizard."""
    allow_skip_optional: bool = Field(
        default=True,
        description="Allow skipping optional steps (e.g., benchmark for non-9(3))",
    )
    auto_populate_defaults: bool = Field(
        default=True,
        description="Auto-populate default values where possible",
    )
    validate_on_each_step: bool = Field(
        default=True,
        description="Validate data at each step (not just review)",
    )
    require_benchmark_for_9_3: bool = Field(
        default=True,
        description="Require benchmark designation for Art 9(3)",
    )
    default_pai_mandatory: List[int] = Field(
        default_factory=lambda: list(range(1, 19)),
        description="Default mandatory PAI indicators (1-18)",
    )
    default_disclosure_frequency: DisclosureFrequency = Field(
        default=DisclosureFrequency.ANNUAL,
        description="Default disclosure frequency",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking"
    )

class WizardStep(BaseModel):
    """Definition and state of a single wizard step."""
    step_id: str = Field(default="", description="Step identifier")
    step_number: int = Field(default=0, ge=1, le=8, description="Step number")
    step_name: str = Field(default="", description="Human-readable step name")
    description: str = Field(default="", description="Step description")
    status: str = Field(
        default="pending", description="Step status"
    )
    is_optional: bool = Field(
        default=False, description="Whether step can be skipped"
    )
    is_conditional: bool = Field(
        default=False,
        description="Whether step is conditional on previous answers",
    )
    condition_met: bool = Field(
        default=True, description="Whether condition is met"
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Step input data"
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Step output data"
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )

class ProductInfo(BaseModel):
    """Product information collected in Step 1."""
    product_name: str = Field(default="", description="Product name")
    product_isin: str = Field(default="", description="Product ISIN")
    product_type: str = Field(
        default="ucits_fund", description="Product type"
    )
    management_company: str = Field(
        default="", description="Management company name"
    )
    lei_code: str = Field(default="", description="LEI code")
    domicile: str = Field(default="", description="Product domicile country")
    reporting_currency: str = Field(
        default="EUR", description="Reporting currency"
    )
    launch_date: str = Field(default="", description="Product launch date")
    total_aum_eur: float = Field(
        default=0.0, description="Total AUM in EUR"
    )

class SubTypeSelection(BaseModel):
    """Article 9 sub-type selection in Step 2."""
    sub_type: str = Field(
        default="art_9_1", description="Selected Article 9 sub-type"
    )
    sub_type_rationale: str = Field(
        default="", description="Rationale for sub-type selection"
    )
    has_carbon_reduction_objective: bool = Field(
        default=False,
        description="Whether product has carbon emissions reduction objective",
    )
    requires_benchmark: bool = Field(
        default=False,
        description="Whether benchmark designation is required (Art 9(3))",
    )

class ObjectiveDefinition(BaseModel):
    """Sustainable investment objective in Step 3."""
    primary_objective: str = Field(
        default="", description="Primary sustainable investment objective"
    )
    objective_category: str = Field(
        default="climate_change_mitigation",
        description="Objective category",
    )
    objective_description: str = Field(
        default="", description="Detailed objective description"
    )
    si_min_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Minimum sustainable investment percentage",
    )
    si_environmental_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage with environmental objective",
    )
    si_social_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage with social objective",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum taxonomy-aligned percentage",
    )
    sdg_targets: List[int] = Field(
        default_factory=list, description="Linked SDG targets"
    )

class BenchmarkDesignation(BaseModel):
    """Benchmark designation in Step 4 (Art 9(3) only)."""
    benchmark_type: str = Field(default="", description="CTB or PAB")
    benchmark_name: str = Field(default="", description="Benchmark index name")
    benchmark_provider: str = Field(
        default="", description="Benchmark provider"
    )
    benchmark_isin: str = Field(default="", description="Benchmark ISIN")
    base_year: int = Field(
        default=2019, ge=2015, le=2025, description="Decarbonization base year"
    )
    annual_reduction_target_pct: float = Field(
        default=7.0, ge=0.0, le=20.0,
        description="Annual carbon reduction target (%)",
    )
    tracking_error_limit_bps: float = Field(
        default=200.0, ge=0.0, description="Tracking error limit (bps)"
    )

class PAIConfiguration(BaseModel):
    """PAI indicator configuration in Step 5."""
    mandatory_indicators: List[int] = Field(
        default_factory=lambda: list(range(1, 19)),
        description="Mandatory PAI indicators (1-18, all required for Art 9)",
    )
    optional_environmental: List[int] = Field(
        default_factory=list,
        description="Optional environmental PAI indicators selected",
    )
    optional_social: List[int] = Field(
        default_factory=list,
        description="Optional social PAI indicators selected",
    )
    data_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Data source per PAI indicator",
    )
    reporting_period: str = Field(
        default="calendar_year", description="PAI reporting period"
    )

class ImpactKPISelection(BaseModel):
    """Impact KPI selection in Step 6."""
    selected_kpis: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Selected impact KPIs with name, unit, target",
    )
    measurement_methodology: str = Field(
        default="internal",
        description="Impact measurement methodology",
    )
    reporting_frequency: str = Field(
        default="annual", description="Impact reporting frequency"
    )
    verification_provider: str = Field(
        default="", description="Third-party verification provider"
    )

class DisclosureCalendar(BaseModel):
    """Disclosure calendar setup in Step 7."""
    reporting_period_start: str = Field(
        default="", description="Reporting period start date"
    )
    reporting_period_end: str = Field(
        default="", description="Reporting period end date"
    )
    pre_contractual_update_date: str = Field(
        default="", description="Annex III update date"
    )
    periodic_publication_date: str = Field(
        default="", description="Annex V publication date"
    )
    website_update_date: str = Field(
        default="", description="Website disclosure update date"
    )
    pai_statement_date: str = Field(
        default="", description="PAI statement publication date"
    )
    frequency: str = Field(
        default="annual", description="Publication frequency"
    )
    internal_review_dates: List[str] = Field(
        default_factory=list, description="Internal review milestone dates"
    )

class WizardResult(BaseModel):
    """Complete result of the setup wizard execution."""
    wizard_id: str = Field(default="", description="Wizard execution ID")
    started_at: str = Field(default="", description="Wizard start timestamp")
    completed_at: str = Field(default="", description="Wizard completion timestamp")

    # Overall status
    is_valid: bool = Field(
        default=False, description="Whether all required steps passed validation"
    )
    config_ready: bool = Field(
        default=False, description="Whether configuration is ready for pipeline"
    )
    total_steps: int = Field(default=8, description="Total wizard steps")
    completed_steps: int = Field(default=0, description="Completed steps")
    skipped_steps: int = Field(default=0, description="Skipped steps")
    failed_steps: int = Field(default=0, description="Failed steps")

    # Per-step results
    steps: List[WizardStep] = Field(
        default_factory=list, description="Per-step state and results"
    )

    # Collected data
    product_info: Optional[ProductInfo] = Field(
        default=None, description="Product information (Step 1)"
    )
    sub_type_selection: Optional[SubTypeSelection] = Field(
        default=None, description="Sub-type selection (Step 2)"
    )
    objective_definition: Optional[ObjectiveDefinition] = Field(
        default=None, description="Objective definition (Step 3)"
    )
    benchmark_designation: Optional[BenchmarkDesignation] = Field(
        default=None, description="Benchmark designation (Step 4)"
    )
    pai_configuration: Optional[PAIConfiguration] = Field(
        default=None, description="PAI configuration (Step 5)"
    )
    impact_kpi_selection: Optional[ImpactKPISelection] = Field(
        default=None, description="Impact KPI selection (Step 6)"
    )
    disclosure_calendar: Optional[DisclosureCalendar] = Field(
        default=None, description="Disclosure calendar (Step 7)"
    )

    # Generated pipeline config
    pipeline_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated Article9OrchestrationConfig parameters",
    )

    # Metadata
    validation_errors: List[str] = Field(
        default_factory=list, description="All validation errors"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="All validation warnings"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    execution_time_ms: float = Field(default=0.0, description="Total execution time")

# =============================================================================
# Step Definitions
# =============================================================================

STEP_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "step_id": WizardStepId.PRODUCT_TYPE.value,
        "step_number": 1,
        "step_name": "Product Type",
        "description": "Select the financial product type and enter basic product information",
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.ARTICLE_9_SUB_TYPE.value,
        "step_number": 2,
        "step_name": "Article 9 Sub-Type",
        "description": "Select the Article 9 sub-type: 9(1) environmental, 9(2) social/combined, or 9(3) carbon reduction with benchmark",
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.SUSTAINABLE_OBJECTIVE.value,
        "step_number": 3,
        "step_name": "Sustainable Investment Objective",
        "description": "Define the sustainable investment objective, SI breakdown, and SDG alignment",
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.BENCHMARK_DESIGNATION.value,
        "step_number": 4,
        "step_name": "Benchmark Designation",
        "description": "Designate a CTB or PAB benchmark (required for Art 9(3), optional otherwise)",
        "is_optional": True,
        "is_conditional": True,
    },
    {
        "step_id": WizardStepId.PAI_CONFIGURATION.value,
        "step_number": 5,
        "step_name": "PAI Configuration",
        "description": "Configure all 18 mandatory PAI indicators plus optional indicators",
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.IMPACT_KPIS.value,
        "step_number": 6,
        "step_name": "Impact KPIs",
        "description": "Select impact key performance indicators and measurement methodology",
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.DISCLOSURE_CALENDAR.value,
        "step_number": 7,
        "step_name": "Disclosure Calendar",
        "description": "Set up the disclosure publication calendar and review milestones",
        "is_optional": False,
        "is_conditional": False,
    },
    {
        "step_id": WizardStepId.REVIEW_VALIDATE.value,
        "step_number": 8,
        "step_name": "Review & Validate",
        "description": "Review all configuration and validate for pipeline readiness",
        "is_optional": False,
        "is_conditional": False,
    },
]

# Product type descriptions for guidance
PRODUCT_TYPE_GUIDANCE: Dict[str, str] = {
    ProductType.UCITS_FUND.value: "UCITS fund - most common for retail Art 9 products",
    ProductType.AIF.value: "Alternative Investment Fund - for professional/institutional investors",
    ProductType.PORTFOLIO_MANAGEMENT.value: "Discretionary portfolio management mandate",
    ProductType.INSURANCE_PRODUCT.value: "Insurance-based investment product (IBIP)",
    ProductType.PENSION_PRODUCT.value: "Pension product (PEPP or occupational)",
    ProductType.STRUCTURED_DEPOSIT.value: "Structured deposit with sustainability features",
    ProductType.OTHER.value: "Other financial product type",
}

# Article 9 sub-type descriptions
SUB_TYPE_GUIDANCE: Dict[str, Dict[str, str]] = {
    Article9SubType.ART_9_1.value: {
        "name": "Article 9(1) - Environmental Objective",
        "description": "Product with environmental sustainable investment objective aligned with EU Taxonomy environmental objectives",
        "benchmark_required": "No (optional)",
        "typical_use": "Green bond funds, renewable energy funds, climate funds",
    },
    Article9SubType.ART_9_2.value: {
        "name": "Article 9(2) - Social or Combined Objective",
        "description": "Product with social sustainable investment objective, or combined environmental and social",
        "benchmark_required": "No (optional)",
        "typical_use": "Social impact funds, SDG-aligned funds, microfinance funds",
    },
    Article9SubType.ART_9_3.value: {
        "name": "Article 9(3) - Carbon Emissions Reduction",
        "description": "Product with carbon emissions reduction objective requiring a designated CTB or PAB benchmark",
        "benchmark_required": "Yes (CTB or PAB mandatory)",
        "typical_use": "Low-carbon funds, Paris-aligned equity funds, net-zero strategies",
    },
}

# =============================================================================
# Setup Wizard
# =============================================================================

class SetupWizard:
    """8-step guided configuration wizard for SFDR Article 9 products.

    Walks through product setup, sub-type classification, objective
    definition, benchmark designation, PAI configuration, impact KPIs,
    disclosure calendar, and final review/validation.

    Attributes:
        config: Wizard configuration.
        _steps: Current step states.
        _collected: Collected data across steps.

    Example:
        >>> wizard = SetupWizard(SetupWizardConfig())
        >>> result = wizard.execute_all_steps({"product_name": "GL Deep Green"})
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
        product_data: Dict[str, Any],
    ) -> WizardResult:
        """Execute all 8 wizard steps with provided product data.

        Args:
            product_data: Combined product data for all steps.

        Returns:
            WizardResult with configuration and validation status.
        """
        start_time = time.time()
        self._collected = dict(product_data)

        # Step 1: Product Type
        self._execute_step_1_product_type(product_data)

        # Step 2: Article 9 Sub-Type
        self._execute_step_2_sub_type(product_data)

        # Step 3: Sustainable Objective
        self._execute_step_3_objective(product_data)

        # Step 4: Benchmark (conditional on 9(3))
        self._execute_step_4_benchmark(product_data)

        # Step 5: PAI Configuration
        self._execute_step_5_pai(product_data)

        # Step 6: Impact KPIs
        self._execute_step_6_impact_kpis(product_data)

        # Step 7: Disclosure Calendar
        self._execute_step_7_calendar(product_data)

        # Step 8: Review & Validate
        self._execute_step_8_review()

        # Build result
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
            WizardStepId.PRODUCT_TYPE.value: self._execute_step_1_product_type,
            WizardStepId.ARTICLE_9_SUB_TYPE.value: self._execute_step_2_sub_type,
            WizardStepId.SUSTAINABLE_OBJECTIVE.value: self._execute_step_3_objective,
            WizardStepId.BENCHMARK_DESIGNATION.value: self._execute_step_4_benchmark,
            WizardStepId.PAI_CONFIGURATION.value: self._execute_step_5_pai,
            WizardStepId.IMPACT_KPIS.value: self._execute_step_6_impact_kpis,
            WizardStepId.DISCLOSURE_CALENDAR.value: self._execute_step_7_calendar,
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
        if step_id == WizardStepId.PRODUCT_TYPE.value:
            return {
                "step_name": "Product Type",
                "options": PRODUCT_TYPE_GUIDANCE,
                "required_fields": ["product_name", "product_type"],
            }
        elif step_id == WizardStepId.ARTICLE_9_SUB_TYPE.value:
            return {
                "step_name": "Article 9 Sub-Type",
                "options": SUB_TYPE_GUIDANCE,
                "required_fields": ["sub_type"],
            }
        elif step_id == WizardStepId.SUSTAINABLE_OBJECTIVE.value:
            return {
                "step_name": "Sustainable Investment Objective",
                "categories": [c.value for c in ObjectiveCategory],
                "required_fields": ["primary_objective", "objective_category"],
            }
        elif step_id == WizardStepId.BENCHMARK_DESIGNATION.value:
            return {
                "step_name": "Benchmark Designation",
                "benchmark_types": ["CTB", "PAB"],
                "required_for": "Art 9(3) only",
                "required_fields": ["benchmark_type", "benchmark_name"],
            }
        elif step_id == WizardStepId.PAI_CONFIGURATION.value:
            return {
                "step_name": "PAI Configuration",
                "mandatory_count": 18,
                "note": "All 18 indicators are mandatory for Article 9 (no opt-out)",
            }
        elif step_id == WizardStepId.IMPACT_KPIS.value:
            return {
                "step_name": "Impact KPIs",
                "suggested_kpis": [
                    {"name": "GHG Avoided", "unit": "tCO2e"},
                    {"name": "Renewable Energy Generated", "unit": "MWh"},
                    {"name": "Jobs Created", "unit": "count"},
                    {"name": "People Reached", "unit": "count"},
                    {"name": "Water Saved", "unit": "m3"},
                ],
            }
        elif step_id == WizardStepId.DISCLOSURE_CALENDAR.value:
            return {
                "step_name": "Disclosure Calendar",
                "frequencies": [f.value for f in DisclosureFrequency],
                "key_dates": {
                    "annex_v_deadline": "April 30 (for previous calendar year)",
                    "pai_statement": "June 30 (annual entity-level)",
                    "website_update": "Continuous obligation",
                },
            }
        return {"step_name": step_id}

    def get_progress(self) -> Dict[str, Any]:
        """Get current wizard progress.

        Returns:
            Progress data with per-step status.
        """
        completed = sum(1 for s in self._steps if s.status == StepStatus.COMPLETED.value)
        skipped = sum(1 for s in self._steps if s.status == StepStatus.SKIPPED.value)
        failed = sum(1 for s in self._steps if s.status == StepStatus.FAILED.value)

        return {
            "total_steps": len(self._steps),
            "completed": completed,
            "skipped": skipped,
            "failed": failed,
            "pending": len(self._steps) - completed - skipped - failed,
            "progress_pct": (completed / len(self._steps)) * 100.0 if self._steps else 0.0,
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
        self.logger.info("SetupWizard reset to initial state")

    # -------------------------------------------------------------------------
    # Step Execution Methods
    # -------------------------------------------------------------------------

    def _execute_step_1_product_type(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 1: Product Type and basic information."""
        step = self._get_step(WizardStepId.PRODUCT_TYPE.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []
        warnings: List[str] = []

        product_name = str(data.get("product_name", ""))
        product_type = str(data.get("product_type", "ucits_fund"))

        if not product_name:
            if self.config.validate_on_each_step:
                errors.append("Product name is required")

        if product_type not in [pt.value for pt in ProductType]:
            warnings.append(f"Unknown product type: {product_type}")
            product_type = ProductType.OTHER.value

        product_info = ProductInfo(
            product_name=product_name,
            product_isin=str(data.get("product_isin", "")),
            product_type=product_type,
            management_company=str(data.get("management_company", "")),
            lei_code=str(data.get("lei_code", "")),
            domicile=str(data.get("domicile", "")),
            reporting_currency=str(data.get("reporting_currency", "EUR")),
            launch_date=str(data.get("launch_date", "")),
            total_aum_eur=float(data.get("total_aum_eur", 0.0)),
        )

        step.output_data = {"product_info": product_info.model_dump()}
        step.validation_errors = errors
        step.validation_warnings = warnings
        step.status = StepStatus.FAILED.value if errors else StepStatus.COMPLETED.value

    def _execute_step_2_sub_type(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 2: Article 9 sub-type selection."""
        step = self._get_step(WizardStepId.ARTICLE_9_SUB_TYPE.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []

        sub_type = str(data.get("sub_type", data.get("article_9_sub_type", "art_9_1")))
        valid_sub_types = [st.value for st in Article9SubType]
        if sub_type not in valid_sub_types:
            sub_type = Article9SubType.ART_9_1.value
            errors.append(f"Invalid sub-type; defaulting to {sub_type}")

        has_carbon = sub_type == Article9SubType.ART_9_3.value
        requires_benchmark = has_carbon and self.config.require_benchmark_for_9_3

        selection = SubTypeSelection(
            sub_type=sub_type,
            sub_type_rationale=str(data.get("sub_type_rationale", "")),
            has_carbon_reduction_objective=has_carbon,
            requires_benchmark=requires_benchmark,
        )

        # Update benchmark step conditionality
        benchmark_step = self._get_step(WizardStepId.BENCHMARK_DESIGNATION.value)
        if benchmark_step:
            benchmark_step.condition_met = has_carbon
            benchmark_step.is_optional = not requires_benchmark

        step.output_data = {"sub_type_selection": selection.model_dump()}
        step.validation_errors = errors
        step.status = StepStatus.FAILED.value if errors else StepStatus.COMPLETED.value

    def _execute_step_3_objective(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 3: Sustainable investment objective definition."""
        step = self._get_step(WizardStepId.SUSTAINABLE_OBJECTIVE.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []
        warnings: List[str] = []

        objective = ObjectiveDefinition(
            primary_objective=str(data.get("primary_objective", data.get("sustainable_objective", ""))),
            objective_category=str(data.get("objective_category", "climate_change_mitigation")),
            objective_description=str(data.get("objective_description", "")),
            si_min_pct=float(data.get("si_min_pct", data.get("sustainable_investment_min_pct", 100.0))),
            si_environmental_pct=float(data.get("si_environmental_pct", data.get("sustainable_env_objective_pct", 0.0))),
            si_social_pct=float(data.get("si_social_pct", data.get("sustainable_soc_objective_pct", 0.0))),
            taxonomy_aligned_pct=float(data.get("taxonomy_aligned_pct", 0.0)),
            sdg_targets=data.get("sdg_targets", data.get("impact_sdg_targets", [])),
        )

        # Validate SI total
        if objective.si_min_pct < 90.0:
            warnings.append(
                f"Art 9 typically requires ~100% SI; got {objective.si_min_pct:.1f}%"
            )

        # Validate env + social breakdown
        total_breakdown = objective.si_environmental_pct + objective.si_social_pct
        if total_breakdown > 100.0:
            errors.append(
                f"Environmental ({objective.si_environmental_pct}%) + "
                f"Social ({objective.si_social_pct}%) exceeds 100%"
            )

        if not objective.primary_objective and self.config.validate_on_each_step:
            errors.append("Primary sustainable investment objective is required")

        step.output_data = {"objective_definition": objective.model_dump()}
        step.validation_errors = errors
        step.validation_warnings = warnings
        step.status = StepStatus.FAILED.value if errors else StepStatus.COMPLETED.value

    def _execute_step_4_benchmark(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 4: Benchmark designation (conditional on Art 9(3))."""
        step = self._get_step(WizardStepId.BENCHMARK_DESIGNATION.value)
        if step is None:
            return

        # Check if step should be skipped
        sub_type_step = self._get_step(WizardStepId.ARTICLE_9_SUB_TYPE.value)
        is_9_3 = False
        if sub_type_step and sub_type_step.output_data:
            sel = sub_type_step.output_data.get("sub_type_selection", {})
            is_9_3 = sel.get("has_carbon_reduction_objective", False)

        if not is_9_3 and self.config.allow_skip_optional:
            step.status = StepStatus.SKIPPED.value
            step.output_data = {"skipped": True, "reason": "Not Art 9(3)"}
            return

        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []

        benchmark_type = str(data.get("benchmark_type", ""))
        benchmark_name = str(data.get("benchmark_name", data.get("benchmark_index_name", "")))

        if is_9_3 and not benchmark_type:
            errors.append("Benchmark type (CTB or PAB) required for Art 9(3)")
        if is_9_3 and not benchmark_name:
            errors.append("Benchmark name required for Art 9(3)")

        if benchmark_type and benchmark_type.upper() not in ("CTB", "PAB"):
            errors.append(f"Invalid benchmark type: {benchmark_type}. Must be CTB or PAB")

        designation = BenchmarkDesignation(
            benchmark_type=benchmark_type.upper() if benchmark_type else "",
            benchmark_name=benchmark_name,
            benchmark_provider=str(data.get("benchmark_provider", "")),
            benchmark_isin=str(data.get("benchmark_isin", "")),
            base_year=int(data.get("base_year", 2019)),
            annual_reduction_target_pct=float(data.get("annual_reduction_target_pct", 7.0)),
            tracking_error_limit_bps=float(data.get("tracking_error_limit_bps", 200.0)),
        )

        step.output_data = {"benchmark_designation": designation.model_dump()}
        step.validation_errors = errors
        step.status = StepStatus.FAILED.value if errors else StepStatus.COMPLETED.value

    def _execute_step_5_pai(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 5: PAI indicator configuration."""
        step = self._get_step(WizardStepId.PAI_CONFIGURATION.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        errors: List[str] = []
        warnings: List[str] = []

        mandatory = data.get(
            "pai_mandatory_indicators",
            self.config.default_pai_mandatory,
        )
        if not isinstance(mandatory, list):
            mandatory = list(range(1, 19))

        if len(mandatory) < 18:
            errors.append(
                f"Article 9 requires all 18 mandatory PAI indicators; "
                f"only {len(mandatory)} configured"
            )

        pai_config = PAIConfiguration(
            mandatory_indicators=mandatory,
            optional_environmental=data.get("pai_optional_environmental", []),
            optional_social=data.get("pai_optional_social", []),
            data_sources=data.get("pai_data_sources", {}),
            reporting_period=str(data.get("pai_reporting_period", "calendar_year")),
        )

        optional_count = len(pai_config.optional_environmental) + len(pai_config.optional_social)
        if optional_count == 0:
            warnings.append(
                "Consider selecting optional PAI indicators for enhanced disclosure"
            )

        step.output_data = {"pai_configuration": pai_config.model_dump()}
        step.validation_errors = errors
        step.validation_warnings = warnings
        step.status = StepStatus.FAILED.value if errors else StepStatus.COMPLETED.value

    def _execute_step_6_impact_kpis(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 6: Impact KPI selection."""
        step = self._get_step(WizardStepId.IMPACT_KPIS.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        warnings: List[str] = []

        selected_kpis = data.get("impact_kpis", data.get("selected_kpis", []))
        if isinstance(selected_kpis, list) and all(isinstance(k, str) for k in selected_kpis):
            selected_kpis = [{"name": k, "unit": "", "target": ""} for k in selected_kpis]

        if not selected_kpis and self.config.auto_populate_defaults:
            selected_kpis = [
                {"name": "GHG Avoided", "unit": "tCO2e", "target": ""},
                {"name": "Renewable Energy Generated", "unit": "MWh", "target": ""},
            ]
            warnings.append("No impact KPIs provided; using defaults")

        kpi_selection = ImpactKPISelection(
            selected_kpis=selected_kpis,
            measurement_methodology=str(data.get("measurement_methodology", "internal")),
            reporting_frequency=str(data.get("impact_reporting_frequency", "annual")),
            verification_provider=str(data.get("verification_provider", "")),
        )

        step.output_data = {"impact_kpi_selection": kpi_selection.model_dump()}
        step.validation_warnings = warnings
        step.status = StepStatus.COMPLETED.value

    def _execute_step_7_calendar(
        self, data: Dict[str, Any]
    ) -> None:
        """Step 7: Disclosure calendar setup."""
        step = self._get_step(WizardStepId.DISCLOSURE_CALENDAR.value)
        if step is None:
            return
        step.status = StepStatus.IN_PROGRESS.value
        warnings: List[str] = []

        now = utcnow()
        current_year = now.year

        calendar = DisclosureCalendar(
            reporting_period_start=str(data.get(
                "reporting_period_start",
                f"{current_year - 1}-01-01",
            )),
            reporting_period_end=str(data.get(
                "reporting_period_end",
                f"{current_year - 1}-12-31",
            )),
            pre_contractual_update_date=str(data.get(
                "pre_contractual_update_date",
                f"{current_year}-03-01",
            )),
            periodic_publication_date=str(data.get(
                "periodic_publication_date",
                f"{current_year}-04-30",
            )),
            website_update_date=str(data.get(
                "website_update_date",
                now.strftime("%Y-%m-%d"),
            )),
            pai_statement_date=str(data.get(
                "pai_statement_date",
                f"{current_year}-06-30",
            )),
            frequency=str(data.get(
                "disclosure_frequency",
                self.config.default_disclosure_frequency.value,
            )),
            internal_review_dates=data.get("internal_review_dates", []),
        )

        if self.config.auto_populate_defaults and not calendar.internal_review_dates:
            calendar.internal_review_dates = [
                f"{current_year}-02-15",
                f"{current_year}-03-15",
                f"{current_year}-04-15",
            ]
            warnings.append("Auto-populated internal review dates")

        step.output_data = {"disclosure_calendar": calendar.model_dump()}
        step.validation_warnings = warnings
        step.status = StepStatus.COMPLETED.value

    def _execute_step_8_review(self) -> None:
        """Step 8: Review & Validate all configuration."""
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
                errors.append(f"Step {s.step_number} ({s.step_name}) not completed: {s.status}")
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
        step.status = StepStatus.FAILED.value if errors else StepStatus.COMPLETED.value

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
        """Cross-validate data across steps."""
        errors: List[str] = []

        # Check Art 9(3) has benchmark
        sub_step = self._get_step(WizardStepId.ARTICLE_9_SUB_TYPE.value)
        bmk_step = self._get_step(WizardStepId.BENCHMARK_DESIGNATION.value)

        if sub_step and bmk_step:
            sub_data = sub_step.output_data.get("sub_type_selection", {})
            is_9_3 = sub_data.get("has_carbon_reduction_objective", False)

            if is_9_3 and bmk_step.status == StepStatus.SKIPPED.value:
                errors.append(
                    "Art 9(3) requires benchmark designation but Step 4 was skipped"
                )

        return errors

    def _build_result(self, elapsed_ms: float) -> WizardResult:
        """Build the complete wizard result."""
        all_errors: List[str] = []
        all_warnings: List[str] = []

        for step in self._steps:
            all_errors.extend(step.validation_errors)
            all_warnings.extend(step.validation_warnings)

        completed = sum(1 for s in self._steps if s.status == StepStatus.COMPLETED.value)
        skipped = sum(1 for s in self._steps if s.status == StepStatus.SKIPPED.value)
        failed = sum(1 for s in self._steps if s.status == StepStatus.FAILED.value)
        is_valid = failed == 0

        # Extract collected data
        product_info = self._extract_model(
            WizardStepId.PRODUCT_TYPE.value, "product_info", ProductInfo,
        )
        sub_type = self._extract_model(
            WizardStepId.ARTICLE_9_SUB_TYPE.value, "sub_type_selection", SubTypeSelection,
        )
        objective = self._extract_model(
            WizardStepId.SUSTAINABLE_OBJECTIVE.value, "objective_definition", ObjectiveDefinition,
        )
        benchmark = self._extract_model(
            WizardStepId.BENCHMARK_DESIGNATION.value, "benchmark_designation", BenchmarkDesignation,
        )
        pai_config = self._extract_model(
            WizardStepId.PAI_CONFIGURATION.value, "pai_configuration", PAIConfiguration,
        )
        impact_kpis = self._extract_model(
            WizardStepId.IMPACT_KPIS.value, "impact_kpi_selection", ImpactKPISelection,
        )
        calendar = self._extract_model(
            WizardStepId.DISCLOSURE_CALENDAR.value, "disclosure_calendar", DisclosureCalendar,
        )

        # Generate pipeline config
        pipeline_config = self._generate_pipeline_config(
            product_info, sub_type, objective, benchmark, pai_config, impact_kpis, calendar,
        )

        result = WizardResult(
            wizard_id=f"WIZ-{utcnow().strftime('%Y%m%d%H%M%S')}",
            started_at=utcnow().isoformat(),
            completed_at=utcnow().isoformat(),
            is_valid=is_valid,
            config_ready=is_valid,
            total_steps=len(self._steps),
            completed_steps=completed,
            skipped_steps=skipped,
            failed_steps=failed,
            steps=list(self._steps),
            product_info=product_info,
            sub_type_selection=sub_type,
            objective_definition=objective,
            benchmark_designation=benchmark,
            pai_configuration=pai_config,
            impact_kpi_selection=impact_kpis,
            disclosure_calendar=calendar,
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
            "SetupWizard complete: valid=%s, completed=%d/%d, "
            "errors=%d, warnings=%d, elapsed=%.1fms",
            is_valid, completed, len(self._steps),
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
        product_info: Optional[ProductInfo],
        sub_type: Optional[SubTypeSelection],
        objective: Optional[ObjectiveDefinition],
        benchmark: Optional[BenchmarkDesignation],
        pai_config: Optional[PAIConfiguration],
        impact_kpis: Optional[ImpactKPISelection],
        calendar: Optional[DisclosureCalendar],
    ) -> Dict[str, Any]:
        """Generate Article9OrchestrationConfig parameters."""
        config: Dict[str, Any] = {
            "pack_id": "PACK-011",
        }

        if product_info:
            config["product_name"] = product_info.product_name
            config["product_isin"] = product_info.product_isin
            config["management_company"] = product_info.management_company
            config["lei_code"] = product_info.lei_code
            config["reporting_currency"] = product_info.reporting_currency

        if sub_type:
            config["article_9_sub_type"] = sub_type.sub_type

        if objective:
            config["sustainable_objective"] = objective.primary_objective
            config["sustainable_investment_min_pct"] = objective.si_min_pct
            config["sustainable_env_objective_pct"] = objective.si_environmental_pct
            config["sustainable_soc_objective_pct"] = objective.si_social_pct
            config["impact_sdg_targets"] = objective.sdg_targets

        if benchmark:
            config["enable_benchmark_alignment"] = True
            config["benchmark_type"] = benchmark.benchmark_type
            config["benchmark_index_name"] = benchmark.benchmark_name
            config["benchmark_provider"] = benchmark.benchmark_provider

        if pai_config:
            config["pai_mandatory_indicators"] = pai_config.mandatory_indicators
            config["pai_optional_indicators"] = (
                pai_config.optional_environmental + pai_config.optional_social
            )

        if impact_kpis:
            config["impact_kpis"] = [
                k.get("name", "") for k in impact_kpis.selected_kpis
            ]

        if calendar:
            config["reporting_period_start"] = calendar.reporting_period_start
            config["reporting_period_end"] = calendar.reporting_period_end

        return config
