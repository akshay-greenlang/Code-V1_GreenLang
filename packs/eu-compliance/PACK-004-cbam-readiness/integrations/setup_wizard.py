# -*- coding: utf-8 -*-
"""
CBAMSetupWizard - 7-Step Guided Setup for CBAM Readiness Pack
===============================================================

This module implements a step-by-step guided setup wizard for new CBAM
Readiness Pack deployments. It collects company profile and EORI information,
configures applicable goods categories and CN codes, registers third-country
suppliers, sets up data sources, configures reporting preferences, and runs
a full health verification.

Wizard Steps:
    1. company_profile: Company name, EORI number, EU member state
    2. goods_categories: Select applicable CBAM goods categories
    3. cn_code_configuration: Map products to specific CN codes
    4. supplier_registry: Register third-country suppliers
    5. data_source_configuration: Configure customs/ERP/manual sources
    6. reporting_preferences: Format, frequency, language, notifications
    7. health_verification: Run full health check, verify configuration

Example:
    >>> wizard = CBAMSetupWizard()
    >>> result = wizard.run()
    >>> print(f"Score: {result.health_check_score}")
    >>> demo_result = wizard.run_demo()
    >>> assert demo_result.steps_completed == 7

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
"""

import hashlib
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class WizardStepName(str, Enum):
    """Names of wizard steps in execution order."""
    COMPANY_PROFILE = "company_profile"
    GOODS_CATEGORIES = "goods_categories"
    CN_CODE_CONFIGURATION = "cn_code_configuration"
    SUPPLIER_REGISTRY = "supplier_registry"
    DATA_SOURCE_CONFIGURATION = "data_source_configuration"
    REPORTING_PREFERENCES = "reporting_preferences"
    HEALTH_VERIFICATION = "health_verification"


class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GoodsCategory(str, Enum):
    """CBAM Annex I goods categories."""
    IRON_AND_STEEL = "IRON_AND_STEEL"
    ALUMINIUM = "ALUMINIUM"
    CEMENT = "CEMENT"
    FERTILISERS = "FERTILISERS"
    HYDROGEN = "HYDROGEN"
    ELECTRICITY = "ELECTRICITY"


class DataSourceType(str, Enum):
    """Types of data sources for CBAM."""
    CUSTOMS_FEED = "customs_feed"
    ERP_SYSTEM = "erp_system"
    MANUAL_UPLOAD = "manual_upload"
    SUPPLIER_PORTAL = "supplier_portal"
    API_INTEGRATION = "api_integration"


class ReportFormat(str, Enum):
    """Supported report output formats."""
    PDF = "pdf"
    EXCEL = "excel"
    XML = "xml"
    JSON = "json"


class ReportLanguage(str, Enum):
    """Supported report languages."""
    EN = "en"
    DE = "de"
    FR = "fr"
    ES = "es"
    IT = "it"
    NL = "nl"


# =============================================================================
# Data Models
# =============================================================================


class CompanyProfile(BaseModel):
    """Company profile collected in step 1."""
    company_name: str = Field(..., min_length=1, max_length=255, description="Legal company name")
    eori_number: str = Field(..., min_length=5, max_length=17, description="EORI number")
    member_state: str = Field(..., min_length=2, max_length=2, description="EU member state code")
    authorized_declarant_name: str = Field(
        default="", description="Authorized CBAM declarant name"
    )
    authorized_declarant_id: str = Field(
        default="", description="Authorized CBAM declarant identifier"
    )
    vat_number: Optional[str] = Field(None, description="VAT number")
    address: str = Field(default="", description="Registered company address")
    contact_email: str = Field(default="", description="Contact email for CBAM correspondence")
    contact_phone: str = Field(default="", description="Contact phone number")

    @field_validator("eori_number")
    @classmethod
    def validate_eori(cls, v: str) -> str:
        """Validate EORI format: 2-letter country + up to 15 alphanumeric."""
        if len(v) < 3:
            raise ValueError("EORI must be at least 3 characters (country code + identifier)")
        country = v[:2]
        if not country.isalpha():
            raise ValueError("EORI must start with 2-letter country code")
        return v.upper()


class GoodsCategorySelection(BaseModel):
    """Goods category selection from step 2."""
    selected_categories: List[GoodsCategory] = Field(
        default_factory=list, description="Selected CBAM goods categories"
    )
    notes: str = Field(default="", description="Notes on category selection")


class CNCodeMapping(BaseModel):
    """Mapping of a product to a specific CN code from step 3."""
    product_name: str = Field(..., description="Internal product name")
    cn_code: str = Field(..., description="8-digit CN code")
    goods_category: GoodsCategory = Field(..., description="CBAM goods category")
    typical_quantity_tonnes: float = Field(
        default=0.0, ge=0.0, description="Typical quarterly import quantity"
    )
    production_route: str = Field(
        default="default", description="Primary production route"
    )
    notes: str = Field(default="", description="Product-specific notes")


class CNCodeConfig(BaseModel):
    """CN code configuration from step 3."""
    mappings: List[CNCodeMapping] = Field(
        default_factory=list, description="Product to CN code mappings"
    )
    total_cn_codes: int = Field(default=0, description="Total unique CN codes configured")


class SupplierEntry(BaseModel):
    """A supplier entry from step 4."""
    supplier_id: str = Field(
        default_factory=lambda: str(uuid4())[:12], description="Supplier identifier"
    )
    supplier_name: str = Field(..., description="Supplier name")
    country: str = Field(..., min_length=2, max_length=2, description="Country code (ISO alpha-2)")
    installation_name: str = Field(default="", description="Production installation name")
    installation_id: str = Field(default="", description="Installation identifier")
    goods_category: GoodsCategory = Field(..., description="Primary goods category supplied")
    production_route: str = Field(default="default", description="Production route")
    has_verified_emissions: bool = Field(
        default=False, description="Whether supplier provides verified emission data"
    )
    contact_email: str = Field(default="", description="Supplier contact email")


class SupplierRegistry(BaseModel):
    """Supplier registry from step 4."""
    suppliers: List[SupplierEntry] = Field(
        default_factory=list, description="Registered suppliers"
    )
    total_suppliers: int = Field(default=0, description="Total suppliers registered")
    countries_represented: List[str] = Field(
        default_factory=list, description="Unique origin countries"
    )


class DataSourceEntry(BaseModel):
    """A data source configuration from step 5."""
    source_id: str = Field(
        default_factory=lambda: str(uuid4())[:8], description="Source identifier"
    )
    source_type: DataSourceType = Field(..., description="Type of data source")
    name: str = Field(default="", description="Source display name")
    description: str = Field(default="", description="Source description")
    connection_params: Dict[str, Any] = Field(
        default_factory=dict, description="Connection parameters"
    )
    is_primary: bool = Field(default=False, description="Whether this is the primary source")
    refresh_frequency: str = Field(
        default="per_quarter", description="Data refresh frequency"
    )


class DataSourceConfig(BaseModel):
    """Data source configuration from step 5."""
    sources: List[DataSourceEntry] = Field(
        default_factory=list, description="Configured data sources"
    )
    primary_source_type: Optional[DataSourceType] = Field(
        None, description="Primary data source type"
    )


class ReportingPreferences(BaseModel):
    """Reporting preferences from step 6."""
    report_format: ReportFormat = Field(
        default=ReportFormat.PDF, description="Default report format"
    )
    report_language: ReportLanguage = Field(
        default=ReportLanguage.EN, description="Report language"
    )
    quarterly_auto_generate: bool = Field(
        default=True, description="Auto-generate quarterly reports"
    )
    notification_email: str = Field(
        default="", description="Email for deadline notifications"
    )
    notification_days_before_deadline: int = Field(
        default=14, ge=1, le=60, description="Days before deadline to send notifications"
    )
    include_executive_summary: bool = Field(
        default=True, description="Include executive summary in reports"
    )
    include_supplier_annex: bool = Field(
        default=True, description="Include supplier data annex"
    )
    decimal_precision: int = Field(
        default=4, ge=0, le=8, description="Decimal precision for emission values"
    )


class WizardStep(BaseModel):
    """State of a single wizard step."""
    name: WizardStepName = Field(..., description="Step name")
    display_name: str = Field(default="", description="Human-readable step name")
    status: StepStatus = Field(default=StepStatus.PENDING, description="Step status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Step data")
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    execution_time_ms: float = Field(default=0.0, description="Step execution time")


class SetupResult(BaseModel):
    """Final result of the setup wizard."""
    wizard_id: str = Field(
        default_factory=lambda: str(uuid4())[:16], description="Wizard session ID"
    )
    steps_completed: int = Field(default=0, description="Number of steps completed")
    total_steps: int = Field(default=7, description="Total wizard steps")
    config: Dict[str, Any] = Field(default_factory=dict, description="Final configuration")
    warnings: List[str] = Field(default_factory=list, description="Setup warnings")
    health_check_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Health check score (0-100)"
    )
    company_name: str = Field(default="", description="Company name")
    eori_number: str = Field(default="", description="EORI number")
    goods_categories: List[str] = Field(
        default_factory=list, description="Selected goods categories"
    )
    cn_codes_configured: int = Field(default=0, description="Number of CN codes configured")
    suppliers_registered: int = Field(default=0, description="Number of suppliers registered")
    data_sources_configured: int = Field(default=0, description="Number of data sources")
    is_complete: bool = Field(default=False, description="Whether wizard completed fully")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Wizard start timestamp",
    )
    completed_at: Optional[str] = Field(None, description="Wizard completion timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# =============================================================================
# Step Definitions
# =============================================================================

STEP_ORDER: List[WizardStepName] = [
    WizardStepName.COMPANY_PROFILE,
    WizardStepName.GOODS_CATEGORIES,
    WizardStepName.CN_CODE_CONFIGURATION,
    WizardStepName.SUPPLIER_REGISTRY,
    WizardStepName.DATA_SOURCE_CONFIGURATION,
    WizardStepName.REPORTING_PREFERENCES,
    WizardStepName.HEALTH_VERIFICATION,
]

STEP_DISPLAY_NAMES: Dict[WizardStepName, str] = {
    WizardStepName.COMPANY_PROFILE: "Company Profile & EORI",
    WizardStepName.GOODS_CATEGORIES: "CBAM Goods Categories",
    WizardStepName.CN_CODE_CONFIGURATION: "CN Code Configuration",
    WizardStepName.SUPPLIER_REGISTRY: "Supplier Registry",
    WizardStepName.DATA_SOURCE_CONFIGURATION: "Data Source Configuration",
    WizardStepName.REPORTING_PREFERENCES: "Reporting Preferences",
    WizardStepName.HEALTH_VERIFICATION: "Health Verification",
}


# =============================================================================
# Setup Wizard Implementation
# =============================================================================


class CBAMSetupWizard:
    """7-step guided setup wizard for CBAM Readiness Pack.

    Collects company information, CBAM goods configuration, supplier
    registry data, data source setup, reporting preferences, and runs a
    health verification to ensure the pack is ready for operation.

    Supports both interactive mode (step by step) and demo mode
    (pre-configured for EuroSteel Imports GmbH).

    Attributes:
        config: Optional configuration dictionary
        _steps: Dictionary of step states
        _company_profile: Collected company profile
        _goods_selection: Selected goods categories
        _cn_config: CN code configuration
        _supplier_registry: Registered suppliers
        _data_sources: Data source configuration
        _preferences: Reporting preferences

    Example:
        >>> wizard = CBAMSetupWizard()
        >>> result = wizard.run_demo()
        >>> assert result.is_complete
        >>> assert result.health_check_score > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the setup wizard.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.logger = logger
        self._steps: Dict[str, WizardStep] = {}
        self._company_profile: Optional[CompanyProfile] = None
        self._goods_selection: Optional[GoodsCategorySelection] = None
        self._cn_config: Optional[CNCodeConfig] = None
        self._supplier_registry: Optional[SupplierRegistry] = None
        self._data_sources: Optional[DataSourceConfig] = None
        self._preferences: Optional[ReportingPreferences] = None
        self._current_step_index: int = 0

        # Initialize step state
        for step_name in STEP_ORDER:
            self._steps[step_name.value] = WizardStep(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )

        self.logger.info("CBAMSetupWizard initialized with %d steps", len(STEP_ORDER))

    # -------------------------------------------------------------------------
    # Main Entry Points
    # -------------------------------------------------------------------------

    def run(self) -> SetupResult:
        """Execute all wizard steps in order.

        Steps that fail validation will be marked as FAILED but the wizard
        continues to subsequent steps. The final health_check_score reflects
        the overall readiness.

        Returns:
            SetupResult summarizing the wizard outcome.
        """
        self.logger.info("Starting CBAM setup wizard (interactive mode)")
        start_time = time.monotonic()

        for step_name in STEP_ORDER:
            step = self._steps[step_name.value]
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.utcnow().isoformat()
            step_start = time.monotonic()

            try:
                errors = self._execute_step(step_name, {})
                step.execution_time_ms = (time.monotonic() - step_start) * 1000

                if errors:
                    step.status = StepStatus.FAILED
                    step.validation_errors = errors
                else:
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.utcnow().isoformat()

            except Exception as exc:
                step.status = StepStatus.FAILED
                step.validation_errors = [str(exc)]
                step.execution_time_ms = (time.monotonic() - step_start) * 1000
                self.logger.error("Step '%s' failed: %s", step_name.value, exc)

        elapsed = (time.monotonic() - start_time) * 1000
        result = self._build_result()
        self.logger.info(
            "Wizard completed in %.1fms: %d/%d steps, score=%.0f",
            elapsed, result.steps_completed, result.total_steps,
            result.health_check_score,
        )
        return result

    def run_demo(self) -> SetupResult:
        """Execute the wizard with pre-configured demo data.

        Uses the demo company "EuroSteel Imports GmbH" with a full set of
        steel and aluminium imports, three suppliers, customs feed + manual
        upload data sources, and standard reporting preferences.

        Returns:
            SetupResult with demo configuration.
        """
        self.logger.info("Starting CBAM setup wizard (demo mode: EuroSteel Imports GmbH)")
        start_time = time.monotonic()

        # Step 1: Company Profile
        self._company_profile = CompanyProfile(
            company_name="EuroSteel Imports GmbH",
            eori_number="DE123456789012345",
            member_state="DE",
            authorized_declarant_name="Hans Mueller",
            authorized_declarant_id="CBAM-DE-2025-00042",
            vat_number="DE301234567",
            address="Industriestrasse 15, 40213 Duesseldorf, Germany",
            contact_email="cbam@eurosteel-imports.de",
            contact_phone="+49 211 555 0123",
        )
        self._complete_step(WizardStepName.COMPANY_PROFILE)

        # Step 2: Goods Categories
        self._goods_selection = GoodsCategorySelection(
            selected_categories=[
                GoodsCategory.IRON_AND_STEEL,
                GoodsCategory.ALUMINIUM,
            ],
            notes="Primary import categories for Q1 2026",
        )
        self._complete_step(WizardStepName.GOODS_CATEGORIES)

        # Step 3: CN Code Configuration
        self._cn_config = CNCodeConfig(
            mappings=[
                CNCodeMapping(
                    product_name="Hot-rolled steel coils",
                    cn_code="7208 10 00",
                    goods_category=GoodsCategory.IRON_AND_STEEL,
                    typical_quantity_tonnes=500.0,
                    production_route="BF-BOF",
                ),
                CNCodeMapping(
                    product_name="Steel reinforcement bars",
                    cn_code="7213 10 00",
                    goods_category=GoodsCategory.IRON_AND_STEEL,
                    typical_quantity_tonnes=300.0,
                    production_route="EAF",
                ),
                CNCodeMapping(
                    product_name="Pig iron ingots",
                    cn_code="7201 10 11",
                    goods_category=GoodsCategory.IRON_AND_STEEL,
                    typical_quantity_tonnes=200.0,
                    production_route="BF-BOF",
                ),
                CNCodeMapping(
                    product_name="Unwrought aluminium",
                    cn_code="7601 10 00",
                    goods_category=GoodsCategory.ALUMINIUM,
                    typical_quantity_tonnes=150.0,
                    production_route="primary_smelting",
                ),
                CNCodeMapping(
                    product_name="Aluminium profiles",
                    cn_code="7604 21 00",
                    goods_category=GoodsCategory.ALUMINIUM,
                    typical_quantity_tonnes=75.0,
                    production_route="extrusion",
                ),
            ],
            total_cn_codes=5,
        )
        self._complete_step(WizardStepName.CN_CODE_CONFIGURATION)

        # Step 4: Supplier Registry
        self._supplier_registry = SupplierRegistry(
            suppliers=[
                SupplierEntry(
                    supplier_name="Turkiye Steel Corp",
                    country="TR",
                    installation_name="Iskenderun Steel Works",
                    installation_id="TR-ISK-001",
                    goods_category=GoodsCategory.IRON_AND_STEEL,
                    production_route="BF-BOF",
                    has_verified_emissions=True,
                    contact_email="cbam@turkiye-steel.com.tr",
                ),
                SupplierEntry(
                    supplier_name="India Iron & Steel Ltd",
                    country="IN",
                    installation_name="Jharkhand Plant",
                    installation_id="IN-JKH-002",
                    goods_category=GoodsCategory.IRON_AND_STEEL,
                    production_route="EAF",
                    has_verified_emissions=False,
                    contact_email="exports@indiairon.co.in",
                ),
                SupplierEntry(
                    supplier_name="Emirates Aluminium (EGA)",
                    country="AE",
                    installation_name="Jebel Ali Smelter",
                    installation_id="AE-JBL-001",
                    goods_category=GoodsCategory.ALUMINIUM,
                    production_route="primary_smelting",
                    has_verified_emissions=True,
                    contact_email="cbam@ega.ae",
                ),
            ],
            total_suppliers=3,
            countries_represented=["TR", "IN", "AE"],
        )
        self._complete_step(WizardStepName.SUPPLIER_REGISTRY)

        # Step 5: Data Source Configuration
        self._data_sources = DataSourceConfig(
            sources=[
                DataSourceEntry(
                    source_type=DataSourceType.CUSTOMS_FEED,
                    name="German Customs (ATLAS)",
                    description="Automated feed from German ATLAS customs system",
                    is_primary=True,
                    refresh_frequency="daily",
                ),
                DataSourceEntry(
                    source_type=DataSourceType.MANUAL_UPLOAD,
                    name="Supplier Emission Spreadsheets",
                    description="Excel uploads from supplier emission declarations",
                    is_primary=False,
                    refresh_frequency="per_quarter",
                ),
                DataSourceEntry(
                    source_type=DataSourceType.SUPPLIER_PORTAL,
                    name="CBAM Supplier Portal",
                    description="Direct supplier data submission portal",
                    is_primary=False,
                    refresh_frequency="per_quarter",
                ),
            ],
            primary_source_type=DataSourceType.CUSTOMS_FEED,
        )
        self._complete_step(WizardStepName.DATA_SOURCE_CONFIGURATION)

        # Step 6: Reporting Preferences
        self._preferences = ReportingPreferences(
            report_format=ReportFormat.PDF,
            report_language=ReportLanguage.EN,
            quarterly_auto_generate=True,
            notification_email="cbam@eurosteel-imports.de",
            notification_days_before_deadline=14,
            include_executive_summary=True,
            include_supplier_annex=True,
            decimal_precision=4,
        )
        self._complete_step(WizardStepName.REPORTING_PREFERENCES)

        # Step 7: Health Verification
        self._complete_step(WizardStepName.HEALTH_VERIFICATION)

        elapsed = (time.monotonic() - start_time) * 1000
        result = self._build_result()
        self.logger.info(
            "Demo wizard completed in %.1fms: %d/%d steps, score=%.0f",
            elapsed, result.steps_completed, result.total_steps,
            result.health_check_score,
        )
        return result

    def complete_step(
        self, step_name: str, data: Dict[str, Any]
    ) -> List[str]:
        """Complete a specific wizard step with provided data.

        Args:
            step_name: The step to complete.
            data: Step-specific data.

        Returns:
            List of validation errors (empty if step is valid).
        """
        try:
            step_enum = WizardStepName(step_name)
        except ValueError:
            return [f"Unknown step: {step_name}"]

        errors = self._execute_step(step_enum, data)
        step = self._steps.get(step_name)
        if step is not None:
            step.data = data
            if errors:
                step.status = StepStatus.FAILED
                step.validation_errors = errors
            else:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.utcnow().isoformat()
        return errors

    def get_progress(self) -> Dict[str, Any]:
        """Get current wizard progress.

        Returns:
            Dictionary with step statuses and completion percentage.
        """
        completed = sum(
            1 for s in self._steps.values() if s.status == StepStatus.COMPLETED
        )
        return {
            "total_steps": len(STEP_ORDER),
            "completed_steps": completed,
            "current_step": (
                STEP_ORDER[self._current_step_index].value
                if self._current_step_index < len(STEP_ORDER) else "done"
            ),
            "completion_percentage": round(completed / len(STEP_ORDER) * 100, 1),
            "steps": {
                name: {
                    "display_name": step.display_name,
                    "status": step.status.value,
                }
                for name, step in self._steps.items()
            },
        }

    # -------------------------------------------------------------------------
    # Step Execution
    # -------------------------------------------------------------------------

    def _execute_step(
        self, step_name: WizardStepName, data: Dict[str, Any]
    ) -> List[str]:
        """Execute and validate a wizard step.

        Args:
            step_name: Step to execute.
            data: Input data for the step.

        Returns:
            List of validation errors.
        """
        handlers = {
            WizardStepName.COMPANY_PROFILE: self._step_company_profile,
            WizardStepName.GOODS_CATEGORIES: self._step_goods_categories,
            WizardStepName.CN_CODE_CONFIGURATION: self._step_cn_codes,
            WizardStepName.SUPPLIER_REGISTRY: self._step_suppliers,
            WizardStepName.DATA_SOURCE_CONFIGURATION: self._step_data_sources,
            WizardStepName.REPORTING_PREFERENCES: self._step_preferences,
            WizardStepName.HEALTH_VERIFICATION: self._step_health,
        }

        handler = handlers.get(step_name)
        if handler is None:
            return [f"No handler for step: {step_name.value}"]

        return handler(data)

    def _step_company_profile(self, data: Dict[str, Any]) -> List[str]:
        """Validate and store company profile.

        Args:
            data: Company profile data.

        Returns:
            Validation errors.
        """
        if self._company_profile is not None:
            return []  # Already set (e.g. demo mode)

        errors: List[str] = []
        try:
            self._company_profile = CompanyProfile(**data)
        except Exception as exc:
            errors.append(f"Invalid company profile: {exc}")
            return errors

        if not self._company_profile.eori_number:
            errors.append("EORI number is required")

        return errors

    def _step_goods_categories(self, data: Dict[str, Any]) -> List[str]:
        """Validate and store goods category selection.

        Args:
            data: Goods category selection data.

        Returns:
            Validation errors.
        """
        if self._goods_selection is not None:
            return []

        errors: List[str] = []
        categories = data.get("selected_categories", [])
        if not categories:
            errors.append("At least one CBAM goods category must be selected")
            return errors

        valid_cats = set(c.value for c in GoodsCategory)
        for cat in categories:
            if cat not in valid_cats:
                errors.append(f"Unknown goods category: {cat}")

        if not errors:
            self._goods_selection = GoodsCategorySelection(
                selected_categories=[GoodsCategory(c) for c in categories],
                notes=data.get("notes", ""),
            )
        return errors

    def _step_cn_codes(self, data: Dict[str, Any]) -> List[str]:
        """Validate and store CN code configuration.

        Args:
            data: CN code mappings data.

        Returns:
            Validation errors.
        """
        if self._cn_config is not None:
            return []

        errors: List[str] = []
        mappings_data = data.get("mappings", [])
        if not mappings_data:
            # Allow empty for wizard.run() mode
            self._cn_config = CNCodeConfig(mappings=[], total_cn_codes=0)
            return []

        mappings: List[CNCodeMapping] = []
        for m in mappings_data:
            try:
                mappings.append(CNCodeMapping(**m))
            except Exception as exc:
                errors.append(f"Invalid CN code mapping: {exc}")

        if not errors:
            self._cn_config = CNCodeConfig(
                mappings=mappings,
                total_cn_codes=len(set(m.cn_code for m in mappings)),
            )
        return errors

    def _step_suppliers(self, data: Dict[str, Any]) -> List[str]:
        """Validate and store supplier registry.

        Args:
            data: Supplier registry data.

        Returns:
            Validation errors.
        """
        if self._supplier_registry is not None:
            return []

        suppliers_data = data.get("suppliers", [])
        suppliers: List[SupplierEntry] = []
        errors: List[str] = []

        for s in suppliers_data:
            try:
                suppliers.append(SupplierEntry(**s))
            except Exception as exc:
                errors.append(f"Invalid supplier entry: {exc}")

        if not errors:
            countries = list(set(s.country for s in suppliers))
            self._supplier_registry = SupplierRegistry(
                suppliers=suppliers,
                total_suppliers=len(suppliers),
                countries_represented=countries,
            )
        return errors

    def _step_data_sources(self, data: Dict[str, Any]) -> List[str]:
        """Validate and store data source configuration.

        Args:
            data: Data source configuration data.

        Returns:
            Validation errors.
        """
        if self._data_sources is not None:
            return []

        sources_data = data.get("sources", [])
        sources: List[DataSourceEntry] = []
        errors: List[str] = []

        for s in sources_data:
            try:
                sources.append(DataSourceEntry(**s))
            except Exception as exc:
                errors.append(f"Invalid data source: {exc}")

        if not errors:
            primary = next((s.source_type for s in sources if s.is_primary), None)
            self._data_sources = DataSourceConfig(
                sources=sources, primary_source_type=primary,
            )
        return errors

    def _step_preferences(self, data: Dict[str, Any]) -> List[str]:
        """Validate and store reporting preferences.

        Args:
            data: Reporting preferences data.

        Returns:
            Validation errors.
        """
        if self._preferences is not None:
            return []

        errors: List[str] = []
        try:
            self._preferences = ReportingPreferences(**data)
        except Exception as exc:
            errors.append(f"Invalid reporting preferences: {exc}")
        return errors

    def _step_health(self, data: Dict[str, Any]) -> List[str]:
        """Run health verification.

        Args:
            data: Additional health check parameters (usually empty).

        Returns:
            Validation errors (empty if health check passes).
        """
        # In a full setup, this would invoke CBAMHealthCheck.run()
        # For the wizard, we do a quick readiness assessment
        errors: List[str] = []

        if self._company_profile is None:
            errors.append("Company profile not configured")
        if self._goods_selection is None:
            errors.append("Goods categories not selected")

        return errors

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _complete_step(self, step_name: WizardStepName) -> None:
        """Mark a step as completed.

        Args:
            step_name: The step to mark complete.
        """
        step = self._steps.get(step_name.value)
        if step is not None:
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.utcnow().isoformat()

    def _build_result(self) -> SetupResult:
        """Build the final SetupResult from wizard state.

        Returns:
            SetupResult summarizing the configuration.
        """
        completed = sum(
            1 for s in self._steps.values() if s.status == StepStatus.COMPLETED
        )
        warnings: List[str] = []

        # Compute health check score
        score = 0.0
        checks = 0
        passed = 0

        # Company profile
        checks += 1
        if self._company_profile is not None:
            passed += 1
        else:
            warnings.append("Company profile not configured")

        # Goods categories
        checks += 1
        if self._goods_selection and self._goods_selection.selected_categories:
            passed += 1
        else:
            warnings.append("No goods categories selected")

        # CN codes
        checks += 1
        if self._cn_config and self._cn_config.total_cn_codes > 0:
            passed += 1
        else:
            warnings.append("No CN codes configured")

        # Suppliers
        checks += 1
        if self._supplier_registry and self._supplier_registry.total_suppliers > 0:
            passed += 1
        else:
            warnings.append("No suppliers registered")

        # Data sources
        checks += 1
        if self._data_sources and self._data_sources.sources:
            passed += 1
        else:
            warnings.append("No data sources configured")

        # Reporting preferences
        checks += 1
        if self._preferences is not None:
            passed += 1

        # Health step completed
        checks += 1
        health_step = self._steps.get(WizardStepName.HEALTH_VERIFICATION.value)
        if health_step and health_step.status == StepStatus.COMPLETED:
            passed += 1

        score = round((passed / max(checks, 1)) * 100, 1)

        # Build config dict
        config: Dict[str, Any] = {}
        if self._company_profile:
            config["company_profile"] = self._company_profile.model_dump()
        if self._goods_selection:
            config["goods_categories"] = [c.value for c in self._goods_selection.selected_categories]
        if self._cn_config:
            config["cn_codes"] = [m.model_dump() for m in self._cn_config.mappings]
        if self._supplier_registry:
            config["suppliers"] = [s.model_dump() for s in self._supplier_registry.suppliers]
        if self._data_sources:
            config["data_sources"] = [s.model_dump() for s in self._data_sources.sources]
        if self._preferences:
            config["reporting_preferences"] = self._preferences.model_dump()

        provenance = _compute_hash(
            f"wizard:{completed}:{score}:{datetime.utcnow().isoformat()}"
        )

        return SetupResult(
            steps_completed=completed,
            config=config,
            warnings=warnings,
            health_check_score=score,
            company_name=(
                self._company_profile.company_name if self._company_profile else ""
            ),
            eori_number=(
                self._company_profile.eori_number if self._company_profile else ""
            ),
            goods_categories=(
                [c.value for c in self._goods_selection.selected_categories]
                if self._goods_selection else []
            ),
            cn_codes_configured=(
                self._cn_config.total_cn_codes if self._cn_config else 0
            ),
            suppliers_registered=(
                self._supplier_registry.total_suppliers if self._supplier_registry else 0
            ),
            data_sources_configured=(
                len(self._data_sources.sources) if self._data_sources else 0
            ),
            is_complete=completed == len(STEP_ORDER),
            completed_at=datetime.utcnow().isoformat(),
            provenance_hash=provenance,
        )


# =============================================================================
# Module-Level Helper
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
