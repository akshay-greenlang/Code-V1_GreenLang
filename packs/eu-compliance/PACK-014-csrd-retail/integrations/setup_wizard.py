# -*- coding: utf-8 -*-
"""
RetailSetupWizard - 8-Step Retail-Specific Configuration Wizard for PACK-014
===============================================================================

This module implements an 8-step configuration wizard tailored for retail
and consumer goods companies setting up the CSRD Retail Pack.

Wizard Steps (8):
    1. company_profile     -- Name, NACE, employees, revenue, store count
    2. store_portfolio     -- Store types, locations, floor areas
    3. retail_sub_sector   -- Grocery/apparel/electronics/general/online/SME
    4. regulatory_scope    -- Which regulations apply (auto-detect from profile)
    5. emissions_sources   -- Scope 1/2/3 source prioritization
    6. product_categories  -- DPP products, EUDR commodities, packaging types
    7. supply_chain        -- Supplier tiers, CSDDD, forced labour screening
    8. reporting_setup     -- Frequency, format, disclosure requirements

13 Retail Sub-Sector Presets with auto-selected engines, MRV agents,
and ESRS focus areas.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-014 CSRD Retail & Consumer Goods
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


class RetailWizardStep(str, Enum):
    """Names of retail wizard steps in execution order."""

    COMPANY_PROFILE = "company_profile"
    STORE_PORTFOLIO = "store_portfolio"
    RETAIL_SUB_SECTOR = "retail_sub_sector"
    REGULATORY_SCOPE = "regulatory_scope"
    EMISSIONS_SOURCES = "emissions_sources"
    PRODUCT_CATEGORIES = "product_categories"
    SUPPLY_CHAIN = "supply_chain"
    REPORTING_SETUP = "reporting_setup"


class StepStatus(str, Enum):
    """Status of a wizard step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """Company profile from step 1."""

    company_name: str = Field(..., min_length=1, max_length=255)
    nace_code: str = Field(default="G47.19", description="Primary NACE code")
    employee_count: int = Field(default=500, ge=1)
    annual_revenue_eur: float = Field(default=100_000_000.0, ge=0)
    store_count: int = Field(default=10, ge=0)
    headquarters_country: str = Field(default="DE")
    is_listed: bool = Field(default=False)
    balance_sheet_total_eur: Optional[float] = Field(None, ge=0)


class StorePortfolio(BaseModel):
    """Store portfolio from step 2."""

    stores: List[Dict[str, Any]] = Field(default_factory=list)
    total_floor_area_m2: float = Field(default=0.0, ge=0)
    store_types: List[str] = Field(default_factory=list)
    countries: List[str] = Field(default_factory=list)
    distribution_centres: int = Field(default=0, ge=0)


class RetailSubSectorConfig(BaseModel):
    """Retail sub-sector configuration from step 3."""

    sub_sector: str = Field(default="general_merchandise")
    preset_applied: bool = Field(default=False)
    engines_enabled: List[str] = Field(default_factory=list)
    mrv_agents_priority: List[str] = Field(default_factory=list)
    esrs_focus_chapters: List[str] = Field(default_factory=list)


class RegulatoryScope(BaseModel):
    """Regulatory scope from step 4."""

    csrd_applicable: bool = Field(default=True)
    ppwr_applicable: bool = Field(default=True)
    eudr_applicable: bool = Field(default=False)
    csddd_applicable: bool = Field(default=False)
    eu_taxonomy_applicable: bool = Field(default=True)
    dpp_applicable: bool = Field(default=False)
    weee_applicable: bool = Field(default=False)
    battery_regulation_applicable: bool = Field(default=False)
    textile_epr_applicable: bool = Field(default=False)


class EmissionsSourceConfig(BaseModel):
    """Emissions source configuration from step 5."""

    scope1_sources: List[str] = Field(
        default_factory=lambda: ["store_heating", "refrigerant_leakage"],
    )
    scope2_sources: List[str] = Field(
        default_factory=lambda: ["store_electricity_location", "store_electricity_market"],
    )
    scope3_priorities: List[int] = Field(
        default_factory=lambda: [1, 4, 5, 9, 12],
        description="Scope 3 category numbers by priority",
    )
    has_owned_fleet: bool = Field(default=False)
    has_refrigeration: bool = Field(default=True)


class ProductCategoryConfig(BaseModel):
    """Product category configuration from step 6."""

    dpp_applicable_products: List[str] = Field(default_factory=list)
    eudr_commodities: List[str] = Field(default_factory=list)
    packaging_types: List[str] = Field(
        default_factory=lambda: ["cardboard", "plastic", "glass"],
    )
    total_sku_count: int = Field(default=0, ge=0)
    own_brand_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class SupplyChainConfig(BaseModel):
    """Supply chain configuration from step 7."""

    supplier_count_tier1: int = Field(default=100, ge=0)
    supplier_count_tier2: int = Field(default=500, ge=0)
    supplier_count_tier3: int = Field(default=0, ge=0)
    csddd_screening_enabled: bool = Field(default=False)
    forced_labour_screening_enabled: bool = Field(default=False)
    high_risk_countries: List[str] = Field(default_factory=list)


class ReportingSetup(BaseModel):
    """Reporting setup from step 8."""

    reporting_frequency: str = Field(default="annual")
    report_format: str = Field(default="XHTML")
    esrs_standards: List[str] = Field(
        default_factory=lambda: ["ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E5"],
    )
    first_reporting_year: int = Field(default=2025, ge=2024, le=2030)
    assurance_level: str = Field(default="limited")


class WizardStepState(BaseModel):
    """State of a single wizard step."""

    name: RetailWizardStep = Field(...)
    display_name: str = Field(default="")
    status: StepStatus = Field(default=StepStatus.PENDING)
    data: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    execution_time_ms: float = Field(default=0.0)


class WizardState(BaseModel):
    """Complete state of the retail setup wizard."""

    wizard_id: str = Field(default="")
    current_step: RetailWizardStep = Field(default=RetailWizardStep.COMPANY_PROFILE)
    steps: Dict[str, WizardStepState] = Field(default_factory=dict)
    company_profile: Optional[CompanyProfile] = Field(None)
    store_portfolio: Optional[StorePortfolio] = Field(None)
    sub_sector_config: Optional[RetailSubSectorConfig] = Field(None)
    regulatory_scope: Optional[RegulatoryScope] = Field(None)
    emissions_config: Optional[EmissionsSourceConfig] = Field(None)
    product_config: Optional[ProductCategoryConfig] = Field(None)
    supply_chain_config: Optional[SupplyChainConfig] = Field(None)
    reporting_setup: Optional[ReportingSetup] = Field(None)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = Field(None)


class SetupResult(BaseModel):
    """Final setup result."""

    result_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field(default="")
    sub_sector: str = Field(default="")
    store_count: int = Field(default=0)
    regulations_applicable: List[str] = Field(default_factory=list)
    engines_enabled: List[str] = Field(default_factory=list)
    scope3_categories: List[int] = Field(default_factory=list)
    total_steps_completed: int = Field(default=0)
    total_steps: int = Field(default=8)
    configuration_hash: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Step Definitions
# ---------------------------------------------------------------------------

STEP_ORDER: List[RetailWizardStep] = [
    RetailWizardStep.COMPANY_PROFILE,
    RetailWizardStep.STORE_PORTFOLIO,
    RetailWizardStep.RETAIL_SUB_SECTOR,
    RetailWizardStep.REGULATORY_SCOPE,
    RetailWizardStep.EMISSIONS_SOURCES,
    RetailWizardStep.PRODUCT_CATEGORIES,
    RetailWizardStep.SUPPLY_CHAIN,
    RetailWizardStep.REPORTING_SETUP,
]

STEP_DISPLAY_NAMES: Dict[RetailWizardStep, str] = {
    RetailWizardStep.COMPANY_PROFILE: "Company Profile",
    RetailWizardStep.STORE_PORTFOLIO: "Store Portfolio",
    RetailWizardStep.RETAIL_SUB_SECTOR: "Retail Sub-Sector",
    RetailWizardStep.REGULATORY_SCOPE: "Regulatory Scope",
    RetailWizardStep.EMISSIONS_SOURCES: "Emissions Sources",
    RetailWizardStep.PRODUCT_CATEGORIES: "Product Categories",
    RetailWizardStep.SUPPLY_CHAIN: "Supply Chain",
    RetailWizardStep.REPORTING_SETUP: "Reporting Setup",
}


# ---------------------------------------------------------------------------
# Sub-Sector Presets (13 presets)
# ---------------------------------------------------------------------------

SUB_SECTOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "grocery": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "food_waste", "circular_economy", "benchmark"],
        "mrv_priority": ["MRV-002", "MRV-001", "MRV-009", "MRV-014", "MRV-017", "MRV-018"],
        "esrs_focus": ["E1", "E5", "S2"],
        "scope3_priority": [1, 4, 5, 7, 9, 12],
        "ppwr": True, "food_waste_tracking": True, "eudr_likely": True,
    },
    "apparel": {
        "engines": ["store_emissions", "retail_scope3", "product_sustainability", "supply_chain_dd", "circular_economy", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-017", "MRV-022", "MRV-025"],
        "esrs_focus": ["E1", "E5", "S2", "S4"],
        "scope3_priority": [1, 4, 9, 11, 12],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": True,
    },
    "electronics": {
        "engines": ["store_emissions", "retail_scope3", "product_sustainability", "circular_economy", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-024", "MRV-025", "MRV-022"],
        "esrs_focus": ["E1", "E5", "S4"],
        "scope3_priority": [1, 4, 9, 11, 12],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": False,
    },
    "general_merchandise": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "circular_economy", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-001", "MRV-017", "MRV-018"],
        "esrs_focus": ["E1", "E5"],
        "scope3_priority": [1, 4, 5, 7, 9],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": False,
    },
    "online_only": {
        "engines": ["retail_scope3", "packaging_compliance", "product_sustainability", "benchmark"],
        "mrv_priority": ["MRV-014", "MRV-022", "MRV-018", "MRV-025"],
        "esrs_focus": ["E1", "E5", "S4"],
        "scope3_priority": [1, 4, 9, 12],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": False,
    },
    "luxury": {
        "engines": ["store_emissions", "retail_scope3", "product_sustainability", "supply_chain_dd", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-017", "MRV-022"],
        "esrs_focus": ["E1", "S2", "S4"],
        "scope3_priority": [1, 4, 9, 12],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": True,
    },
    "home_improvement": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "circular_economy", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-001", "MRV-014", "MRV-017", "MRV-025"],
        "esrs_focus": ["E1", "E5"],
        "scope3_priority": [1, 2, 4, 5, 12],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": True,
    },
    "pharmacy": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-018", "MRV-001"],
        "esrs_focus": ["E1", "E5", "S4"],
        "scope3_priority": [1, 4, 5],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": False,
    },
    "convenience": {
        "engines": ["store_emissions", "retail_scope3", "food_waste", "packaging_compliance", "benchmark"],
        "mrv_priority": ["MRV-002", "MRV-009", "MRV-014", "MRV-001"],
        "esrs_focus": ["E1", "E5"],
        "scope3_priority": [1, 4, 5, 7],
        "ppwr": True, "food_waste_tracking": True, "eudr_likely": False,
    },
    "department_store": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "product_sustainability", "circular_economy", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-001", "MRV-017", "MRV-022"],
        "esrs_focus": ["E1", "E5", "S2"],
        "scope3_priority": [1, 4, 5, 7, 9, 12],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": False,
    },
    "discount": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "food_waste", "benchmark"],
        "mrv_priority": ["MRV-002", "MRV-009", "MRV-014", "MRV-018"],
        "esrs_focus": ["E1", "E5"],
        "scope3_priority": [1, 4, 5],
        "ppwr": True, "food_waste_tracking": True, "eudr_likely": False,
    },
    "specialty_food": {
        "engines": ["store_emissions", "retail_scope3", "food_waste", "supply_chain_dd", "benchmark"],
        "mrv_priority": ["MRV-002", "MRV-009", "MRV-014", "MRV-017"],
        "esrs_focus": ["E1", "E5", "S2"],
        "scope3_priority": [1, 4, 5, 7],
        "ppwr": True, "food_waste_tracking": True, "eudr_likely": True,
    },
    "sme_retail": {
        "engines": ["store_emissions", "retail_scope3", "packaging_compliance", "benchmark"],
        "mrv_priority": ["MRV-009", "MRV-014", "MRV-001"],
        "esrs_focus": ["E1"],
        "scope3_priority": [1, 4],
        "ppwr": True, "food_waste_tracking": False, "eudr_likely": False,
    },
}


# ---------------------------------------------------------------------------
# RetailSetupWizard
# ---------------------------------------------------------------------------


class RetailSetupWizard:
    """8-step retail-specific configuration wizard for PACK-014.

    Guides retail companies through setup with sub-sector presets that
    auto-configure engines, MRV agent priorities, and ESRS focus areas.

    Example:
        >>> wizard = RetailSetupWizard()
        >>> state = wizard.start()
        >>> state = wizard.complete_step("company_profile", {...})
        >>> result = wizard.run_demo()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Retail Setup Wizard."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._state: Optional[WizardState] = None
        self._step_handlers = {
            RetailWizardStep.COMPANY_PROFILE: self._handle_company_profile,
            RetailWizardStep.STORE_PORTFOLIO: self._handle_store_portfolio,
            RetailWizardStep.RETAIL_SUB_SECTOR: self._handle_sub_sector,
            RetailWizardStep.REGULATORY_SCOPE: self._handle_regulatory_scope,
            RetailWizardStep.EMISSIONS_SOURCES: self._handle_emissions_sources,
            RetailWizardStep.PRODUCT_CATEGORIES: self._handle_product_categories,
            RetailWizardStep.SUPPLY_CHAIN: self._handle_supply_chain,
            RetailWizardStep.REPORTING_SETUP: self._handle_reporting_setup,
        }
        self.logger.info("RetailSetupWizard initialized")

    def start(self) -> WizardState:
        """Start a new wizard session."""
        wizard_id = _compute_hash(f"retail-wizard:{_utcnow().isoformat()}")[:16]
        steps: Dict[str, WizardStepState] = {}
        for step_name in STEP_ORDER:
            steps[step_name.value] = WizardStepState(
                name=step_name,
                display_name=STEP_DISPLAY_NAMES.get(step_name, step_name.value),
            )
        self._state = WizardState(wizard_id=wizard_id, current_step=STEP_ORDER[0], steps=steps)
        self.logger.info("Retail wizard started: %s", wizard_id)
        return self._state

    def complete_step(self, step_name: str, data: Dict[str, Any]) -> WizardState:
        """Complete a wizard step with provided data.

        Args:
            step_name: Step name to complete.
            data: Step configuration data.

        Returns:
            Updated WizardState.
        """
        if self._state is None:
            raise RuntimeError("Wizard must be started first")

        try:
            step_enum = RetailWizardStep(step_name)
        except ValueError:
            valid = [s.value for s in RetailWizardStep]
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
        """Execute a pre-configured demo setup for a grocery retailer."""
        self.start()

        demo_steps = {
            "company_profile": {
                "company_name": "Demo Grocery Chain",
                "nace_code": "G47.11",
                "employee_count": 15000,
                "annual_revenue_eur": 5_000_000_000.0,
                "store_count": 250,
                "headquarters_country": "DE",
                "is_listed": True,
            },
            "store_portfolio": {
                "stores": [{"store_id": f"S-{i:04d}", "type": "supermarket", "country": "DE", "area_m2": 2500.0} for i in range(1, 4)],
                "total_floor_area_m2": 625000.0,
                "store_types": ["supermarket", "hypermarket", "convenience"],
                "countries": ["DE", "AT", "NL", "PL"],
                "distribution_centres": 5,
            },
            "retail_sub_sector": {"sub_sector": "grocery"},
            "regulatory_scope": {
                "csrd_applicable": True, "ppwr_applicable": True,
                "eudr_applicable": True, "csddd_applicable": True,
                "eu_taxonomy_applicable": True,
            },
            "emissions_sources": {
                "scope1_sources": ["store_heating", "refrigerant_leakage", "delivery_fleet"],
                "scope2_sources": ["store_electricity_location", "store_electricity_market"],
                "scope3_priorities": [1, 4, 5, 7, 9, 12],
                "has_owned_fleet": True, "has_refrigeration": True,
            },
            "product_categories": {
                "dpp_applicable_products": [],
                "eudr_commodities": ["coffee", "cocoa", "palm_oil", "soy", "cattle"],
                "packaging_types": ["cardboard", "plastic", "glass", "metal"],
                "total_sku_count": 25000,
                "own_brand_pct": 35.0,
            },
            "supply_chain": {
                "supplier_count_tier1": 500,
                "supplier_count_tier2": 2000,
                "csddd_screening_enabled": True,
                "forced_labour_screening_enabled": True,
                "high_risk_countries": ["CN", "BD", "VN", "MM"],
            },
            "reporting_setup": {
                "reporting_frequency": "annual",
                "report_format": "XHTML",
                "esrs_standards": ["ESRS_1", "ESRS_2", "ESRS_E1", "ESRS_E5", "ESRS_S2"],
                "first_reporting_year": 2025,
                "assurance_level": "limited",
            },
        }

        for step_name, data in demo_steps.items():
            self.complete_step(step_name, data)

        return self._generate_result()

    def get_state(self) -> Optional[WizardState]:
        """Return the current wizard state."""
        return self._state

    def get_sub_sector_preset(self, sub_sector: str) -> Optional[Dict[str, Any]]:
        """Get the preset configuration for a retail sub-sector.

        Args:
            sub_sector: Sub-sector key (e.g., 'grocery', 'apparel').

        Returns:
            Preset configuration dict, or None if not found.
        """
        return SUB_SECTOR_PRESETS.get(sub_sector)

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

    def _handle_store_portfolio(self, data: Dict[str, Any]) -> List[str]:
        """Handle store portfolio step."""
        errors: List[str] = []
        try:
            portfolio = StorePortfolio(**data)
            if self._state:
                self._state.store_portfolio = portfolio
        except Exception as exc:
            errors.append(f"Invalid store portfolio: {exc}")
        return errors

    def _handle_sub_sector(self, data: Dict[str, Any]) -> List[str]:
        """Handle retail sub-sector step with preset auto-apply."""
        errors: List[str] = []
        sub_sector = data.get("sub_sector", "general_merchandise")
        preset = SUB_SECTOR_PRESETS.get(sub_sector)

        if preset is None:
            errors.append(f"Unknown sub-sector '{sub_sector}'. Valid: {sorted(SUB_SECTOR_PRESETS.keys())}")
            return errors

        config = RetailSubSectorConfig(
            sub_sector=sub_sector,
            preset_applied=True,
            engines_enabled=preset.get("engines", []),
            mrv_agents_priority=preset.get("mrv_priority", []),
            esrs_focus_chapters=preset.get("esrs_focus", []),
        )
        if self._state:
            self._state.sub_sector_config = config

        self.logger.info("Sub-sector preset applied: %s", sub_sector)
        return errors

    def _handle_regulatory_scope(self, data: Dict[str, Any]) -> List[str]:
        """Handle regulatory scope step."""
        errors: List[str] = []
        try:
            scope = RegulatoryScope(**data)
            if self._state:
                self._state.regulatory_scope = scope
        except Exception as exc:
            errors.append(f"Invalid regulatory scope: {exc}")
        return errors

    def _handle_emissions_sources(self, data: Dict[str, Any]) -> List[str]:
        """Handle emissions sources step."""
        errors: List[str] = []
        try:
            config = EmissionsSourceConfig(**data)
            if self._state:
                self._state.emissions_config = config
        except Exception as exc:
            errors.append(f"Invalid emissions config: {exc}")
        return errors

    def _handle_product_categories(self, data: Dict[str, Any]) -> List[str]:
        """Handle product categories step."""
        errors: List[str] = []
        try:
            config = ProductCategoryConfig(**data)
            if self._state:
                self._state.product_config = config
        except Exception as exc:
            errors.append(f"Invalid product config: {exc}")
        return errors

    def _handle_supply_chain(self, data: Dict[str, Any]) -> List[str]:
        """Handle supply chain step."""
        errors: List[str] = []
        try:
            config = SupplyChainConfig(**data)
            if self._state:
                self._state.supply_chain_config = config
        except Exception as exc:
            errors.append(f"Invalid supply chain config: {exc}")
        return errors

    def _handle_reporting_setup(self, data: Dict[str, Any]) -> List[str]:
        """Handle reporting setup step."""
        errors: List[str] = []
        try:
            setup = ReportingSetup(**data)
            if self._state:
                self._state.reporting_setup = setup
        except Exception as exc:
            errors.append(f"Invalid reporting setup: {exc}")
        return errors

    # ---- Navigation ----

    def _advance_step(self, current: RetailWizardStep) -> None:
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

        regulations = []
        if self._state.regulatory_scope:
            rs = self._state.regulatory_scope
            if rs.csrd_applicable:
                regulations.append("CSRD")
            if rs.ppwr_applicable:
                regulations.append("PPWR")
            if rs.eudr_applicable:
                regulations.append("EUDR")
            if rs.csddd_applicable:
                regulations.append("CSDDD")
            if rs.eu_taxonomy_applicable:
                regulations.append("EU Taxonomy")

        engines = []
        if self._state.sub_sector_config:
            engines = list(self._state.sub_sector_config.engines_enabled)

        scope3 = []
        if self._state.emissions_config:
            scope3 = list(self._state.emissions_config.scope3_priorities)

        config_hash = _compute_hash({
            "company": self._state.company_profile.company_name if self._state.company_profile else "",
            "sub_sector": self._state.sub_sector_config.sub_sector if self._state.sub_sector_config else "",
            "regulations": regulations,
        })

        result = SetupResult(
            company_name=(self._state.company_profile.company_name if self._state.company_profile else ""),
            sub_sector=(self._state.sub_sector_config.sub_sector if self._state.sub_sector_config else ""),
            store_count=(self._state.company_profile.store_count if self._state.company_profile else 0),
            regulations_applicable=regulations,
            engines_enabled=engines,
            scope3_categories=scope3,
            total_steps_completed=completed_count,
            configuration_hash=config_hash,
        )
        result.provenance_hash = _compute_hash(result)
        return result
