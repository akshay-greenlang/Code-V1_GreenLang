# -*- coding: utf-8 -*-
"""
Scope 3 Screening Workflow
================================

4-phase workflow for rapid screening-level assessment of all 15 GHG Protocol
Scope 3 categories within PACK-042 Scope 3 Starter Pack.

Phases:
    1. OrganizationProfile   -- Collect sector (NAICS/ISIC), revenue, employee
                                count, product types, facility count
    2. SpendDataIntake       -- Import procurement data, classify by EEIO sector,
                                validate totals
    3. ScreeningCalculation  -- Run spend-based estimates for all 15 categories
                                using EEIO emission factors
    4. RelevanceAssessment   -- Rank categories by magnitude (% of estimated
                                total), flag relevant categories (>1% threshold),
                                recommend methodology tiers

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic EEIO factors and arithmetic. SHA-256
provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Value Chain (Scope 3) Standard -- Chapter 7
    GHG Protocol Technical Guidance for Calculating Scope 3 Emissions (2013)
    ISO 14064-1:2018 Clause 5.2.4 (Indirect GHG emissions)

Schedule: on-demand (typically at the start of Scope 3 inventory)
Estimated duration: 2-4 hours

Author: GreenLang Platform Team
Version: 42.0.0
"""

_MODULE_VERSION: str = "42.0.0"

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""

    CAT_01_PURCHASED_GOODS = "cat_01_purchased_goods_services"
    CAT_02_CAPITAL_GOODS = "cat_02_capital_goods"
    CAT_03_FUEL_ENERGY = "cat_03_fuel_energy_related"
    CAT_04_UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    CAT_05_WASTE = "cat_05_waste_in_operations"
    CAT_06_BUSINESS_TRAVEL = "cat_06_business_travel"
    CAT_07_COMMUTING = "cat_07_employee_commuting"
    CAT_08_UPSTREAM_LEASED = "cat_08_upstream_leased_assets"
    CAT_09_DOWNSTREAM_TRANSPORT = "cat_09_downstream_transport"
    CAT_10_PROCESSING = "cat_10_processing_sold_products"
    CAT_11_USE_SOLD = "cat_11_use_of_sold_products"
    CAT_12_END_OF_LIFE = "cat_12_end_of_life_treatment"
    CAT_13_DOWNSTREAM_LEASED = "cat_13_downstream_leased_assets"
    CAT_14_FRANCHISES = "cat_14_franchises"
    CAT_15_INVESTMENTS = "cat_15_investments"


class MethodologyTier(str, Enum):
    """Methodology tier for Scope 3 calculation."""

    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    NOT_APPLICABLE = "not_applicable"


class RelevanceLevel(str, Enum):
    """Relevance classification for Scope 3 categories."""

    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    MARGINALLY_RELEVANT = "marginally_relevant"
    NOT_RELEVANT = "not_relevant"
    NOT_APPLICABLE = "not_applicable"


class SectorClassification(str, Enum):
    """Sector classification systems."""

    NAICS = "naics"
    ISIC = "isic"
    NACE = "nace"
    GICS = "gics"
    SIC = "sic"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Phase output data"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume capability."""

    workflow_id: str = Field(default="", description="Unique workflow execution ID")
    current_phase: int = Field(default=0, description="Last completed phase number")
    phase_statuses: Dict[str, str] = Field(
        default_factory=dict, description="Phase name -> status"
    )
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(
        default_factory=dict, description="Serialized intermediate data"
    )
    created_at: str = Field(default="", description="ISO-8601 timestamp")
    updated_at: str = Field(default="", description="ISO-8601 timestamp")


class OrganizationProfile(BaseModel):
    """Organization profile for Scope 3 screening."""

    organization_name: str = Field(default="", description="Legal entity name")
    sector_code: str = Field(
        default="", description="NAICS/ISIC/NACE sector code"
    )
    sector_classification: SectorClassification = Field(
        default=SectorClassification.NAICS
    )
    sector_description: str = Field(default="", description="Sector description")
    revenue_usd: float = Field(default=0.0, ge=0.0, description="Annual revenue USD")
    employee_count: int = Field(default=0, ge=0, description="Total employees")
    facility_count: int = Field(default=0, ge=0, description="Number of facilities")
    product_types: List[str] = Field(
        default_factory=list, description="Primary product/service types"
    )
    has_manufacturing: bool = Field(default=False)
    has_fleet: bool = Field(default=False)
    has_franchises: bool = Field(default=False)
    has_leased_assets_upstream: bool = Field(default=False)
    has_leased_assets_downstream: bool = Field(default=False)
    has_investments: bool = Field(default=False)
    country: str = Field(default="US", description="ISO 3166-1 alpha-2 HQ country")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)

    @field_validator("sector_code")
    @classmethod
    def validate_sector_code(cls, v: str) -> str:
        """Validate sector code is non-empty when provided."""
        return v.strip()


class SpendRecord(BaseModel):
    """Single procurement spend record."""

    record_id: str = Field(
        default_factory=lambda: f"spend-{uuid.uuid4().hex[:8]}"
    )
    supplier_name: str = Field(default="", description="Supplier name")
    eeio_sector: str = Field(
        default="", description="EEIO sector classification code"
    )
    spend_usd: float = Field(default=0.0, ge=0.0, description="Spend amount USD")
    scope3_category: Scope3Category = Field(
        default=Scope3Category.CAT_01_PURCHASED_GOODS
    )
    description: str = Field(default="", description="Description of spend")
    currency: str = Field(default="USD")
    year: int = Field(default=2025, ge=2020, le=2050)


class CategoryScreeningResult(BaseModel):
    """Screening result for a single Scope 3 category."""

    category: Scope3Category = Field(...)
    category_number: int = Field(default=0, ge=0, le=15)
    category_name: str = Field(default="")
    estimated_tco2e: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    spend_usd: float = Field(default=0.0, ge=0.0)
    emission_factor_kgco2e_per_usd: float = Field(
        default=0.0, ge=0.0, description="Applied EEIO factor"
    )
    relevance_level: RelevanceLevel = Field(default=RelevanceLevel.NOT_RELEVANT)
    recommended_tier: MethodologyTier = Field(default=MethodologyTier.SPEND_BASED)
    data_quality_score: float = Field(default=1.0, ge=1.0, le=5.0)
    is_applicable: bool = Field(default=True)
    notes: str = Field(default="")


class ScreeningOutput(BaseModel):
    """Complete Scope 3 screening output."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scope3_screening")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    total_scope3_estimated_tco2e: float = Field(default=0.0, ge=0.0)
    total_spend_usd: float = Field(default=0.0, ge=0.0)
    category_results: List[CategoryScreeningResult] = Field(default_factory=list)
    relevant_categories: List[str] = Field(
        default_factory=list, description="Categories exceeding relevance threshold"
    )
    relevance_threshold_pct: float = Field(default=1.0)
    methodology_recommendations: Dict[str, str] = Field(
        default_factory=dict, description="Category -> recommended tier"
    )
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT MODEL
# =============================================================================


class Scope3ScreeningInput(BaseModel):
    """Input data model for Scope3ScreeningWorkflow."""

    org_profile: OrganizationProfile = Field(
        default_factory=OrganizationProfile,
        description="Organization profile data",
    )
    spend_data: List[SpendRecord] = Field(
        default_factory=list, description="Procurement spend records"
    )
    relevance_threshold_pct: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Category relevance threshold as % of total",
    )
    include_non_applicable: bool = Field(
        default=True, description="Include non-applicable categories in output"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


# =============================================================================
# EEIO EMISSION FACTORS (Zero-Hallucination Reference Data)
# =============================================================================

# EEIO emission factors in kgCO2e per USD of spend, by EEIO sector
# Source: US EPA EEIO model (Supply Chain GHG Emission Factors v1.2)
EEIO_FACTORS_KGCO2E_PER_USD: Dict[str, float] = {
    "agriculture": 1.20,
    "mining": 1.80,
    "utilities": 2.50,
    "construction": 0.45,
    "manufacturing_food": 0.80,
    "manufacturing_textiles": 0.55,
    "manufacturing_wood": 0.60,
    "manufacturing_paper": 0.75,
    "manufacturing_chemicals": 1.10,
    "manufacturing_plastics": 0.90,
    "manufacturing_metals": 1.40,
    "manufacturing_electronics": 0.30,
    "manufacturing_machinery": 0.40,
    "manufacturing_vehicles": 0.50,
    "manufacturing_other": 0.65,
    "wholesale_trade": 0.15,
    "retail_trade": 0.12,
    "transport_road": 0.85,
    "transport_rail": 0.35,
    "transport_water": 0.55,
    "transport_air": 1.80,
    "transport_pipeline": 0.40,
    "warehousing": 0.20,
    "information": 0.08,
    "finance_insurance": 0.05,
    "real_estate": 0.10,
    "professional_services": 0.06,
    "management": 0.05,
    "admin_support": 0.07,
    "education": 0.08,
    "healthcare": 0.15,
    "arts_entertainment": 0.10,
    "accommodation_food": 0.25,
    "other_services": 0.12,
    "government": 0.10,
    "default": 0.40,
}

# Sector-average Scope 3 category distribution (% of total Scope 3)
# Source: CDP sector averages, GHG Protocol guidance
SECTOR_CATEGORY_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "manufacturing": {
        "cat_01_purchased_goods_services": 45.0,
        "cat_02_capital_goods": 5.0,
        "cat_03_fuel_energy_related": 3.0,
        "cat_04_upstream_transport": 8.0,
        "cat_05_waste_in_operations": 2.0,
        "cat_06_business_travel": 1.5,
        "cat_07_employee_commuting": 2.0,
        "cat_08_upstream_leased_assets": 0.5,
        "cat_09_downstream_transport": 6.0,
        "cat_10_processing_sold_products": 5.0,
        "cat_11_use_of_sold_products": 15.0,
        "cat_12_end_of_life_treatment": 4.0,
        "cat_13_downstream_leased_assets": 0.5,
        "cat_14_franchises": 0.0,
        "cat_15_investments": 2.5,
    },
    "services": {
        "cat_01_purchased_goods_services": 55.0,
        "cat_02_capital_goods": 8.0,
        "cat_03_fuel_energy_related": 4.0,
        "cat_04_upstream_transport": 3.0,
        "cat_05_waste_in_operations": 1.0,
        "cat_06_business_travel": 10.0,
        "cat_07_employee_commuting": 8.0,
        "cat_08_upstream_leased_assets": 3.0,
        "cat_09_downstream_transport": 1.0,
        "cat_10_processing_sold_products": 0.0,
        "cat_11_use_of_sold_products": 0.0,
        "cat_12_end_of_life_treatment": 0.5,
        "cat_13_downstream_leased_assets": 4.0,
        "cat_14_franchises": 0.0,
        "cat_15_investments": 2.5,
    },
    "retail": {
        "cat_01_purchased_goods_services": 60.0,
        "cat_02_capital_goods": 5.0,
        "cat_03_fuel_energy_related": 3.0,
        "cat_04_upstream_transport": 10.0,
        "cat_05_waste_in_operations": 1.5,
        "cat_06_business_travel": 1.0,
        "cat_07_employee_commuting": 3.0,
        "cat_08_upstream_leased_assets": 2.0,
        "cat_09_downstream_transport": 5.0,
        "cat_10_processing_sold_products": 0.0,
        "cat_11_use_of_sold_products": 4.0,
        "cat_12_end_of_life_treatment": 3.0,
        "cat_13_downstream_leased_assets": 1.0,
        "cat_14_franchises": 1.0,
        "cat_15_investments": 0.5,
    },
    "finance": {
        "cat_01_purchased_goods_services": 25.0,
        "cat_02_capital_goods": 5.0,
        "cat_03_fuel_energy_related": 2.0,
        "cat_04_upstream_transport": 1.0,
        "cat_05_waste_in_operations": 0.5,
        "cat_06_business_travel": 8.0,
        "cat_07_employee_commuting": 5.0,
        "cat_08_upstream_leased_assets": 3.0,
        "cat_09_downstream_transport": 0.0,
        "cat_10_processing_sold_products": 0.0,
        "cat_11_use_of_sold_products": 0.0,
        "cat_12_end_of_life_treatment": 0.0,
        "cat_13_downstream_leased_assets": 5.0,
        "cat_14_franchises": 0.0,
        "cat_15_investments": 45.5,
    },
    "default": {
        "cat_01_purchased_goods_services": 50.0,
        "cat_02_capital_goods": 5.0,
        "cat_03_fuel_energy_related": 3.0,
        "cat_04_upstream_transport": 5.0,
        "cat_05_waste_in_operations": 1.5,
        "cat_06_business_travel": 4.0,
        "cat_07_employee_commuting": 4.0,
        "cat_08_upstream_leased_assets": 1.5,
        "cat_09_downstream_transport": 3.0,
        "cat_10_processing_sold_products": 2.0,
        "cat_11_use_of_sold_products": 8.0,
        "cat_12_end_of_life_treatment": 2.5,
        "cat_13_downstream_leased_assets": 2.0,
        "cat_14_franchises": 0.5,
        "cat_15_investments": 8.5,
    },
}

# Category number to enum mapping
CATEGORY_NUMBER_MAP: Dict[int, Scope3Category] = {
    1: Scope3Category.CAT_01_PURCHASED_GOODS,
    2: Scope3Category.CAT_02_CAPITAL_GOODS,
    3: Scope3Category.CAT_03_FUEL_ENERGY,
    4: Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    5: Scope3Category.CAT_05_WASTE,
    6: Scope3Category.CAT_06_BUSINESS_TRAVEL,
    7: Scope3Category.CAT_07_COMMUTING,
    8: Scope3Category.CAT_08_UPSTREAM_LEASED,
    9: Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT,
    10: Scope3Category.CAT_10_PROCESSING,
    11: Scope3Category.CAT_11_USE_SOLD,
    12: Scope3Category.CAT_12_END_OF_LIFE,
    13: Scope3Category.CAT_13_DOWNSTREAM_LEASED,
    14: Scope3Category.CAT_14_FRANCHISES,
    15: Scope3Category.CAT_15_INVESTMENTS,
}

# Human-readable category names
CATEGORY_NAMES: Dict[Scope3Category, str] = {
    Scope3Category.CAT_01_PURCHASED_GOODS: "Purchased Goods & Services",
    Scope3Category.CAT_02_CAPITAL_GOODS: "Capital Goods",
    Scope3Category.CAT_03_FUEL_ENERGY: "Fuel- & Energy-Related Activities",
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT: "Upstream Transportation & Distribution",
    Scope3Category.CAT_05_WASTE: "Waste Generated in Operations",
    Scope3Category.CAT_06_BUSINESS_TRAVEL: "Business Travel",
    Scope3Category.CAT_07_COMMUTING: "Employee Commuting",
    Scope3Category.CAT_08_UPSTREAM_LEASED: "Upstream Leased Assets",
    Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT: "Downstream Transportation & Distribution",
    Scope3Category.CAT_10_PROCESSING: "Processing of Sold Products",
    Scope3Category.CAT_11_USE_SOLD: "Use of Sold Products",
    Scope3Category.CAT_12_END_OF_LIFE: "End-of-Life Treatment of Sold Products",
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: "Downstream Leased Assets",
    Scope3Category.CAT_14_FRANCHISES: "Franchises",
    Scope3Category.CAT_15_INVESTMENTS: "Investments",
}

# Sector-average emission intensity (kgCO2e per USD revenue) for fallback estimation
SECTOR_INTENSITY_KGCO2E_PER_USD_REVENUE: Dict[str, float] = {
    "manufacturing": 0.45,
    "services": 0.08,
    "retail": 0.25,
    "finance": 0.03,
    "energy": 1.80,
    "transport": 0.90,
    "agriculture": 1.50,
    "mining": 2.00,
    "construction": 0.55,
    "default": 0.30,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class Scope3ScreeningWorkflow:
    """
    4-phase Scope 3 screening workflow for rapid category-level assessment.

    Performs spend-based screening of all 15 Scope 3 categories using EEIO
    emission factors, ranks categories by magnitude, identifies relevant
    categories above the 1% threshold, and recommends methodology tiers
    for each category.

    Zero-hallucination: all emission estimates are derived from published EEIO
    factors and deterministic arithmetic. No LLM calls in numeric paths.

    Attributes:
        workflow_id: Unique execution identifier.
        _org_profile: Collected organization profile.
        _spend_records: Validated spend records.
        _category_results: Per-category screening results.
        _phase_results: Ordered phase outputs.
        _state: Checkpoint/resume state.

    Example:
        >>> wf = Scope3ScreeningWorkflow()
        >>> inp = Scope3ScreeningInput(
        ...     org_profile=OrganizationProfile(revenue_usd=100_000_000),
        ...     spend_data=[SpendRecord(spend_usd=50_000_000, eeio_sector="manufacturing_chemicals")],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "organization_profile",
        "spend_data_intake",
        "screening_calculation",
        "relevance_assessment",
    ]

    PHASE_WEIGHTS: Dict[str, float] = {
        "organization_profile": 10.0,
        "spend_data_intake": 25.0,
        "screening_calculation": 40.0,
        "relevance_assessment": 25.0,
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize Scope3ScreeningWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._org_profile: Optional[OrganizationProfile] = None
        self._spend_records: List[SpendRecord] = []
        self._category_results: List[CategoryScreeningResult] = []
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self._spend_by_category: Dict[str, float] = {}
        self._total_spend: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[Scope3ScreeningInput] = None,
        org_profile: Optional[OrganizationProfile] = None,
        spend_data: Optional[List[SpendRecord]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ScreeningOutput:
        """
        Execute the 4-phase Scope 3 screening workflow.

        Args:
            input_data: Full input model (preferred).
            org_profile: Organization profile (fallback).
            spend_data: Spend records (fallback).
            config: Optional configuration overrides.

        Returns:
            ScreeningOutput with per-category estimates and relevance assessment.

        Raises:
            ValueError: If required data is missing.
        """
        if input_data is None:
            input_data = Scope3ScreeningInput(
                org_profile=org_profile or OrganizationProfile(),
                spend_data=spend_data or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting Scope 3 screening workflow %s org=%s revenue=%.0f spend_records=%d",
            self.workflow_id,
            input_data.org_profile.organization_name,
            input_data.org_profile.revenue_usd,
            len(input_data.spend_data),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            # Phase 1: Organization Profile
            phase1 = await self._execute_with_retry(
                self._phase_organization_profile, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")
            self._update_progress(10.0)

            # Phase 2: Spend Data Intake
            phase2 = await self._execute_with_retry(
                self._phase_spend_data_intake, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")
            self._update_progress(35.0)

            # Phase 3: Screening Calculation
            phase3 = await self._execute_with_retry(
                self._phase_screening_calculation, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")
            self._update_progress(75.0)

            # Phase 4: Relevance Assessment
            phase4 = await self._execute_with_retry(
                self._phase_relevance_assessment, input_data, phase_number=4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")
            self._update_progress(100.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Scope 3 screening workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error",
                    phase_number=0,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Build relevant categories list
        relevant_cats = [
            cr.category.value
            for cr in self._category_results
            if cr.relevance_level
            in (RelevanceLevel.HIGHLY_RELEVANT, RelevanceLevel.RELEVANT)
        ]

        # Build methodology recommendations
        method_recs = {
            cr.category.value: cr.recommended_tier.value
            for cr in self._category_results
            if cr.is_applicable
        }

        total_estimated = sum(cr.estimated_tco2e for cr in self._category_results)

        result = ScreeningOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=(
                self._org_profile.organization_name if self._org_profile else ""
            ),
            reporting_year=input_data.reporting_year,
            total_scope3_estimated_tco2e=round(total_estimated, 2),
            total_spend_usd=round(self._total_spend, 2),
            category_results=self._category_results,
            relevant_categories=relevant_cats,
            relevance_threshold_pct=input_data.relevance_threshold_pct,
            methodology_recommendations=method_recs,
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Scope 3 screening workflow %s completed in %.2fs status=%s "
            "total=%.1f tCO2e relevant=%d/15 categories",
            self.workflow_id,
            elapsed,
            overall_status.value,
            total_estimated,
            len(relevant_cats),
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    async def resume(
        self,
        state: WorkflowState,
        input_data: Scope3ScreeningInput,
    ) -> ScreeningOutput:
        """
        Resume workflow from a saved checkpoint state.

        Args:
            state: Previously saved WorkflowState.
            input_data: Original input data.

        Returns:
            ScreeningOutput from the resumed execution.
        """
        self._state = state
        self.workflow_id = state.workflow_id
        self.logger.info(
            "Resuming workflow %s from phase %d",
            self.workflow_id,
            state.current_phase,
        )
        # Re-execute from the beginning; completed phases will be fast
        return await self.execute(input_data)

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self,
        phase_fn: Any,
        input_data: Scope3ScreeningInput,
        phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number,
                        attempt,
                        self.MAX_RETRIES,
                        exc,
                        delay,
                    )
                    import asyncio

                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Organization Profile
    # -------------------------------------------------------------------------

    async def _phase_organization_profile(
        self, input_data: Scope3ScreeningInput
    ) -> PhaseResult:
        """Collect and validate organization profile data."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        profile = input_data.org_profile
        self._org_profile = profile

        # Validate critical fields
        if not profile.organization_name:
            warnings.append("Organization name not provided; using 'Unknown'")

        if profile.revenue_usd <= 0:
            warnings.append(
                "Revenue not provided; spend-based estimates will rely "
                "solely on procurement data"
            )

        if profile.employee_count <= 0:
            warnings.append(
                "Employee count not provided; commuting and travel "
                "estimates will use sector averages"
            )

        # Determine sector category for distribution lookup
        sector_key = self._normalize_sector(profile.sector_description)

        # Check applicability flags
        applicability_flags = {
            "has_manufacturing": profile.has_manufacturing,
            "has_fleet": profile.has_fleet,
            "has_franchises": profile.has_franchises,
            "has_leased_assets_upstream": profile.has_leased_assets_upstream,
            "has_leased_assets_downstream": profile.has_leased_assets_downstream,
            "has_investments": profile.has_investments,
        }

        outputs["organization_name"] = profile.organization_name or "Unknown"
        outputs["sector_code"] = profile.sector_code
        outputs["sector_key"] = sector_key
        outputs["revenue_usd"] = profile.revenue_usd
        outputs["employee_count"] = profile.employee_count
        outputs["facility_count"] = profile.facility_count
        outputs["product_types"] = profile.product_types
        outputs["applicability_flags"] = applicability_flags
        outputs["country"] = profile.country

        self._state.phase_statuses["organization_profile"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 OrganizationProfile: org=%s sector=%s revenue=%.0f employees=%d",
            outputs["organization_name"],
            sector_key,
            profile.revenue_usd,
            profile.employee_count,
        )
        return PhaseResult(
            phase_name="organization_profile",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Spend Data Intake
    # -------------------------------------------------------------------------

    async def _phase_spend_data_intake(
        self, input_data: Scope3ScreeningInput
    ) -> PhaseResult:
        """Import procurement data, classify by EEIO sector, validate totals."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._spend_records = list(input_data.spend_data)
        total_spend = 0.0
        classified_count = 0
        unclassified_count = 0
        spend_by_eeio: Dict[str, float] = {}
        spend_by_category: Dict[str, float] = {}

        for record in self._spend_records:
            total_spend += record.spend_usd

            # Track EEIO sector classification
            eeio = record.eeio_sector.strip().lower() if record.eeio_sector else ""
            if eeio and eeio in EEIO_FACTORS_KGCO2E_PER_USD:
                classified_count += 1
                spend_by_eeio[eeio] = spend_by_eeio.get(eeio, 0.0) + record.spend_usd
            elif eeio:
                # Known sector but not in our factor table
                classified_count += 1
                spend_by_eeio[eeio] = spend_by_eeio.get(eeio, 0.0) + record.spend_usd
                warnings.append(
                    f"EEIO sector '{eeio}' not in factor table; "
                    f"will use default factor for record {record.record_id}"
                )
            else:
                unclassified_count += 1

            # Aggregate spend by Scope 3 category
            cat_key = record.scope3_category.value
            spend_by_category[cat_key] = (
                spend_by_category.get(cat_key, 0.0) + record.spend_usd
            )

        self._total_spend = total_spend
        self._spend_by_category = spend_by_category

        # Validate totals against revenue if available
        if self._org_profile and self._org_profile.revenue_usd > 0:
            spend_to_revenue_ratio = total_spend / self._org_profile.revenue_usd
            if spend_to_revenue_ratio > 1.5:
                warnings.append(
                    f"Total spend ({total_spend:.0f} USD) exceeds 150% of "
                    f"revenue ({self._org_profile.revenue_usd:.0f} USD); "
                    f"verify data completeness"
                )
            elif spend_to_revenue_ratio < 0.1:
                warnings.append(
                    f"Total spend ({total_spend:.0f} USD) is less than 10% of "
                    f"revenue; procurement data may be incomplete"
                )

        classification_rate = (
            (classified_count / len(self._spend_records) * 100.0)
            if self._spend_records
            else 0.0
        )

        if unclassified_count > 0:
            warnings.append(
                f"{unclassified_count} spend records have no EEIO sector "
                f"classification; default factor will be applied"
            )

        outputs["total_records"] = len(self._spend_records)
        outputs["total_spend_usd"] = round(total_spend, 2)
        outputs["classified_records"] = classified_count
        outputs["unclassified_records"] = unclassified_count
        outputs["classification_rate_pct"] = round(classification_rate, 1)
        outputs["unique_eeio_sectors"] = len(spend_by_eeio)
        outputs["spend_by_eeio_sector"] = {
            k: round(v, 2) for k, v in sorted(
                spend_by_eeio.items(), key=lambda x: x[1], reverse=True
            )
        }
        outputs["spend_by_scope3_category"] = {
            k: round(v, 2) for k, v in spend_by_category.items()
        }

        self._state.phase_statuses["spend_data_intake"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 SpendDataIntake: %d records, %.0f USD total, "
            "%.1f%% classified, %d EEIO sectors",
            len(self._spend_records),
            total_spend,
            classification_rate,
            len(spend_by_eeio),
        )
        return PhaseResult(
            phase_name="spend_data_intake",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Screening Calculation
    # -------------------------------------------------------------------------

    async def _phase_screening_calculation(
        self, input_data: Scope3ScreeningInput
    ) -> PhaseResult:
        """Run spend-based EEIO estimates for all 15 categories."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._category_results = []
        sector_key = self._normalize_sector(
            self._org_profile.sector_description if self._org_profile else ""
        )
        distribution = SECTOR_CATEGORY_DISTRIBUTION.get(
            sector_key, SECTOR_CATEGORY_DISTRIBUTION["default"]
        )

        # Strategy: use actual spend data where available, fall back to
        # revenue-based estimation using sector distribution
        has_spend_data = self._total_spend > 0

        for cat_num in range(1, 16):
            category = CATEGORY_NUMBER_MAP[cat_num]
            cat_key = category.value
            cat_name = CATEGORY_NAMES[category]

            # Check applicability based on org profile
            is_applicable = self._check_category_applicability(category)

            if not is_applicable:
                self._category_results.append(
                    CategoryScreeningResult(
                        category=category,
                        category_number=cat_num,
                        category_name=cat_name,
                        estimated_tco2e=0.0,
                        pct_of_total=0.0,
                        spend_usd=0.0,
                        emission_factor_kgco2e_per_usd=0.0,
                        relevance_level=RelevanceLevel.NOT_APPLICABLE,
                        recommended_tier=MethodologyTier.NOT_APPLICABLE,
                        data_quality_score=1.0,
                        is_applicable=False,
                        notes="Category not applicable based on organization profile",
                    )
                )
                continue

            # Calculate spend for this category
            cat_spend = self._spend_by_category.get(cat_key, 0.0)

            # If no direct spend data, estimate from revenue using sector distribution
            if cat_spend <= 0 and has_spend_data:
                # Try to allocate unclassified spend proportionally
                dist_pct = distribution.get(cat_key, 0.0)
                cat_spend = self._total_spend * (dist_pct / 100.0)
            elif cat_spend <= 0 and not has_spend_data:
                # Fall back to revenue-based estimation
                if self._org_profile and self._org_profile.revenue_usd > 0:
                    dist_pct = distribution.get(cat_key, 0.0)
                    sector_intensity = SECTOR_INTENSITY_KGCO2E_PER_USD_REVENUE.get(
                        sector_key, SECTOR_INTENSITY_KGCO2E_PER_USD_REVENUE["default"]
                    )
                    # Estimate: revenue * sector intensity * category distribution
                    cat_spend = (
                        self._org_profile.revenue_usd * (dist_pct / 100.0)
                    )
                    if cat_spend <= 0:
                        cat_spend = 0.0

            # Determine emission factor
            ef = self._get_category_emission_factor(category, sector_key)

            # Calculate estimated emissions: spend * factor / 1000 (kg to tonnes)
            estimated_tco2e = (cat_spend * ef) / 1000.0

            self._category_results.append(
                CategoryScreeningResult(
                    category=category,
                    category_number=cat_num,
                    category_name=cat_name,
                    estimated_tco2e=round(estimated_tco2e, 2),
                    spend_usd=round(cat_spend, 2),
                    emission_factor_kgco2e_per_usd=round(ef, 4),
                    is_applicable=True,
                    data_quality_score=self._assess_screening_data_quality(
                        cat_spend, has_spend_data
                    ),
                )
            )

        # Calculate percentages
        total_estimated = sum(cr.estimated_tco2e for cr in self._category_results)
        for cr in self._category_results:
            if total_estimated > 0 and cr.is_applicable:
                cr.pct_of_total = round(
                    (cr.estimated_tco2e / total_estimated) * 100.0, 2
                )

        if total_estimated <= 0:
            warnings.append(
                "Total estimated Scope 3 emissions are zero; "
                "check spend data and organization profile"
            )

        outputs["total_estimated_tco2e"] = round(total_estimated, 2)
        outputs["categories_calculated"] = sum(
            1 for cr in self._category_results if cr.is_applicable
        )
        outputs["categories_not_applicable"] = sum(
            1 for cr in self._category_results if not cr.is_applicable
        )
        outputs["estimation_method"] = (
            "spend_based" if has_spend_data else "revenue_based"
        )
        outputs["sector_key"] = sector_key
        outputs["top_5_categories"] = [
            {
                "category": cr.category_name,
                "tco2e": cr.estimated_tco2e,
                "pct": cr.pct_of_total,
            }
            for cr in sorted(
                self._category_results,
                key=lambda x: x.estimated_tco2e,
                reverse=True,
            )[:5]
        ]

        self._state.phase_statuses["screening_calculation"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ScreeningCalculation: total=%.1f tCO2e, "
            "%d categories calculated, method=%s",
            total_estimated,
            outputs["categories_calculated"],
            outputs["estimation_method"],
        )
        return PhaseResult(
            phase_name="screening_calculation",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Relevance Assessment
    # -------------------------------------------------------------------------

    async def _phase_relevance_assessment(
        self, input_data: Scope3ScreeningInput
    ) -> PhaseResult:
        """Rank categories, flag relevant ones, recommend methodology tiers."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        threshold = input_data.relevance_threshold_pct

        # Assign relevance levels and methodology recommendations
        for cr in self._category_results:
            if not cr.is_applicable:
                continue

            # Relevance classification
            if cr.pct_of_total >= threshold * 5:
                cr.relevance_level = RelevanceLevel.HIGHLY_RELEVANT
            elif cr.pct_of_total >= threshold:
                cr.relevance_level = RelevanceLevel.RELEVANT
            elif cr.pct_of_total >= threshold * 0.5:
                cr.relevance_level = RelevanceLevel.MARGINALLY_RELEVANT
            else:
                cr.relevance_level = RelevanceLevel.NOT_RELEVANT

            # Methodology tier recommendation based on relevance + data availability
            cr.recommended_tier = self._recommend_methodology_tier(cr)

            # Add notes for highly relevant categories
            if cr.relevance_level == RelevanceLevel.HIGHLY_RELEVANT:
                cr.notes = (
                    f"Major emission source ({cr.pct_of_total:.1f}% of total). "
                    f"Prioritize supplier-specific data collection."
                )
            elif cr.relevance_level == RelevanceLevel.RELEVANT:
                cr.notes = (
                    f"Significant emission source ({cr.pct_of_total:.1f}% of total). "
                    f"Consider upgrading from spend-based to average-data methodology."
                )

        # Sort by estimated emissions descending
        self._category_results.sort(
            key=lambda x: x.estimated_tco2e, reverse=True
        )

        # Pareto analysis: identify categories covering 80% of total
        total = sum(cr.estimated_tco2e for cr in self._category_results)
        cumulative = 0.0
        pareto_categories: List[str] = []
        for cr in self._category_results:
            if total > 0:
                cumulative += cr.estimated_tco2e
                pareto_categories.append(cr.category.value)
                if cumulative >= total * 0.8:
                    break

        highly_relevant = [
            cr.category_name
            for cr in self._category_results
            if cr.relevance_level == RelevanceLevel.HIGHLY_RELEVANT
        ]
        relevant = [
            cr.category_name
            for cr in self._category_results
            if cr.relevance_level == RelevanceLevel.RELEVANT
        ]
        not_relevant = [
            cr.category_name
            for cr in self._category_results
            if cr.relevance_level == RelevanceLevel.NOT_RELEVANT
        ]

        outputs["relevance_threshold_pct"] = threshold
        outputs["highly_relevant_count"] = len(highly_relevant)
        outputs["relevant_count"] = len(relevant)
        outputs["not_relevant_count"] = len(not_relevant)
        outputs["highly_relevant_categories"] = highly_relevant
        outputs["relevant_categories"] = relevant
        outputs["pareto_80_categories"] = pareto_categories
        outputs["pareto_80_count"] = len(pareto_categories)
        outputs["tier_recommendations"] = {
            cr.category.value: cr.recommended_tier.value
            for cr in self._category_results
            if cr.is_applicable
        }

        if len(highly_relevant) == 0 and len(relevant) == 0:
            warnings.append(
                "No categories above relevance threshold; review screening data quality"
            )

        self._state.phase_statuses["relevance_assessment"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 RelevanceAssessment: highly_relevant=%d, relevant=%d, "
            "not_relevant=%d, pareto_80=%d categories",
            len(highly_relevant),
            len(relevant),
            len(not_relevant),
            len(pareto_categories),
        )
        return PhaseResult(
            phase_name="relevance_assessment",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _check_category_applicability(self, category: Scope3Category) -> bool:
        """Check if a Scope 3 category is applicable to this organization."""
        if not self._org_profile:
            return True  # Default to applicable if no profile

        # Most categories are applicable to all organizations
        # These require specific business activities
        non_universal = {
            Scope3Category.CAT_10_PROCESSING: self._org_profile.has_manufacturing,
            Scope3Category.CAT_13_DOWNSTREAM_LEASED: (
                self._org_profile.has_leased_assets_downstream
            ),
            Scope3Category.CAT_14_FRANCHISES: self._org_profile.has_franchises,
            Scope3Category.CAT_15_INVESTMENTS: self._org_profile.has_investments,
        }

        if category in non_universal:
            return non_universal[category]

        return True

    def _get_category_emission_factor(
        self, category: Scope3Category, sector_key: str
    ) -> float:
        """Get the EEIO emission factor for a category in kgCO2e/USD."""
        # Map categories to typical EEIO sectors
        category_eeio_map: Dict[Scope3Category, str] = {
            Scope3Category.CAT_01_PURCHASED_GOODS: "manufacturing_other",
            Scope3Category.CAT_02_CAPITAL_GOODS: "manufacturing_machinery",
            Scope3Category.CAT_03_FUEL_ENERGY: "utilities",
            Scope3Category.CAT_04_UPSTREAM_TRANSPORT: "transport_road",
            Scope3Category.CAT_05_WASTE: "other_services",
            Scope3Category.CAT_06_BUSINESS_TRAVEL: "transport_air",
            Scope3Category.CAT_07_COMMUTING: "transport_road",
            Scope3Category.CAT_08_UPSTREAM_LEASED: "real_estate",
            Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT: "transport_road",
            Scope3Category.CAT_10_PROCESSING: "manufacturing_other",
            Scope3Category.CAT_11_USE_SOLD: "utilities",
            Scope3Category.CAT_12_END_OF_LIFE: "other_services",
            Scope3Category.CAT_13_DOWNSTREAM_LEASED: "real_estate",
            Scope3Category.CAT_14_FRANCHISES: "retail_trade",
            Scope3Category.CAT_15_INVESTMENTS: "finance_insurance",
        }

        eeio_sector = category_eeio_map.get(category, "default")
        return EEIO_FACTORS_KGCO2E_PER_USD.get(
            eeio_sector, EEIO_FACTORS_KGCO2E_PER_USD["default"]
        )

    def _assess_screening_data_quality(
        self, spend: float, has_actual_spend: bool
    ) -> float:
        """Assess data quality score (1-5) for screening estimates."""
        if spend <= 0:
            return 1.0
        if has_actual_spend and spend > 0:
            return 2.0  # Spend-based is quality level 2
        return 1.5  # Revenue-based estimation

    def _recommend_methodology_tier(
        self, result: CategoryScreeningResult
    ) -> MethodologyTier:
        """Recommend methodology tier based on relevance and data quality."""
        if result.relevance_level == RelevanceLevel.HIGHLY_RELEVANT:
            return MethodologyTier.SUPPLIER_SPECIFIC
        elif result.relevance_level == RelevanceLevel.RELEVANT:
            return MethodologyTier.AVERAGE_DATA
        elif result.relevance_level == RelevanceLevel.MARGINALLY_RELEVANT:
            return MethodologyTier.SPEND_BASED
        else:
            return MethodologyTier.SPEND_BASED

    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector string to a known distribution key."""
        if not sector:
            return "default"
        sector_lower = sector.lower().strip()
        mapping = {
            "manufacturing": "manufacturing",
            "industrial": "manufacturing",
            "factory": "manufacturing",
            "service": "services",
            "professional": "services",
            "consulting": "services",
            "technology": "services",
            "software": "services",
            "retail": "retail",
            "consumer": "retail",
            "wholesale": "retail",
            "finance": "finance",
            "banking": "finance",
            "insurance": "finance",
            "investment": "finance",
            "energy": "manufacturing",
            "mining": "manufacturing",
            "construction": "manufacturing",
            "transport": "services",
            "logistics": "services",
            "agriculture": "manufacturing",
        }
        for key, value in mapping.items():
            if key in sector_lower:
                return value
        return "default"

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._org_profile = None
        self._spend_records = []
        self._category_results = []
        self._phase_results = []
        self._spend_by_category = {}
        self._total_spend = 0.0
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _update_progress(self, pct: float) -> None:
        """Update progress percentage in state."""
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: ScreeningOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.total_scope3_estimated_tco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
