# -*- coding: utf-8 -*-
"""
Express Onboarding Workflow
================================

4-phase workflow for rapid SME net-zero onboarding within PACK-026
SME Net Zero Pack.  Designed for time-constrained small and medium
enterprises, completing full onboarding in 15-20 minutes.

Phases:
    1. OrganizationProfile   -- Collect basic company info (5 min)
    2. QuickBaseline         -- Bronze baseline from energy spend + headcount (5 min)
    3. AutoTarget            -- Auto-generate 1.5C-aligned targets (instant)
    4. QuickWins             -- Identify top 5 quick-win actions (5 min)

Total time: 15-20 minutes (vs. 2-4 hours for enterprise onboarding).

Uses: sme_baseline_engine (Bronze mode), simplified_target_engine,
      quick_wins_engine.

Zero-hallucination: all emission factors are deterministic DEFRA/IEA
2024 constants.  No LLM calls in numeric paths.  SHA-256 provenance
hashes guarantee end-to-end auditability.

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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

class BaselineTier(str, Enum):
    """SME baseline calculation tier."""

    BRONZE = "bronze"      # Spend-based only (~5 min)
    SILVER = "silver"      # Activity data (~30 min)
    GOLD = "gold"          # Metered data (~2 hrs)

class IndustrySector(str, Enum):
    """Simplified NACE-based industry sectors for SME."""

    OFFICE_SERVICES = "office_services"
    RETAIL_HOSPITALITY = "retail_hospitality"
    MANUFACTURING_LIGHT = "manufacturing_light"
    MANUFACTURING_HEAVY = "manufacturing_heavy"
    CONSTRUCTION = "construction"
    TRANSPORT_LOGISTICS = "transport_logistics"
    AGRICULTURE = "agriculture"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    TECHNOLOGY = "technology"
    OTHER = "other"

class CompanySizeBand(str, Enum):
    """SME size classification (EU definition)."""

    MICRO = "micro"            # <10 employees, <2M EUR
    SMALL = "small"            # <50 employees, <10M EUR
    MEDIUM = "medium"          # <250 employees, <50M EUR
    LARGE_SME = "large_sme"    # 250-500 employees

class QuickWinCategory(str, Enum):
    """Quick win action categories."""

    ENERGY_EFFICIENCY = "energy_efficiency"
    LIGHTING = "lighting"
    HEATING_COOLING = "heating_cooling"
    RENEWABLE_ENERGY = "renewable_energy"
    TRANSPORT = "transport"
    WASTE_REDUCTION = "waste_reduction"
    PROCUREMENT = "procurement"
    BEHAVIOUR_CHANGE = "behaviour_change"
    DIGITAL = "digital"
    WATER = "water"

# =============================================================================
# SME EMISSION FACTOR CONSTANTS (DEFRA 2024 / IEA 2024)
# =============================================================================

# Spend-to-emissions factors by sector (tCO2e per GBP 1000 spent)
# Source: DEFRA environmental reporting guidelines for SMEs
SECTOR_SPEND_EF_TCO2E_PER_1000GBP: Dict[str, float] = {
    "office_services": 0.062,
    "retail_hospitality": 0.095,
    "manufacturing_light": 0.148,
    "manufacturing_heavy": 0.285,
    "construction": 0.178,
    "transport_logistics": 0.215,
    "agriculture": 0.192,
    "healthcare": 0.087,
    "education": 0.068,
    "technology": 0.055,
    "other": 0.098,
}

# Per-employee emission factors by sector (tCO2e/employee/year)
# Source: UK BEIS SME benchmarks
SECTOR_PER_EMPLOYEE_EF_TCO2E: Dict[str, float] = {
    "office_services": 2.4,
    "retail_hospitality": 3.8,
    "manufacturing_light": 6.2,
    "manufacturing_heavy": 12.5,
    "construction": 7.8,
    "transport_logistics": 9.5,
    "agriculture": 8.1,
    "healthcare": 3.2,
    "education": 2.1,
    "technology": 1.9,
    "other": 4.0,
}

# Energy spend to kWh conversion (approximate GBP/kWh)
ENERGY_COST_PER_KWH_GBP: Dict[str, float] = {
    "electricity": 0.28,
    "gas": 0.08,
    "oil": 0.10,
}

# Grid electricity emission factor (kgCO2e/kWh)
GRID_EF_KGCO2E_PER_KWH: Dict[str, float] = {
    "UK": 0.2070,
    "EU-AVG": 0.2556,
    "US-AVG": 0.3710,
    "GLOBAL": 0.4940,
}

# Gas emission factor (kgCO2e/kWh)
GAS_EF_KGCO2E_PER_KWH = 0.18293

# Quick win savings database (tCO2e/year, GBP saved/year, implementation cost GBP)
QUICK_WIN_DATABASE: Dict[str, Dict[str, Any]] = {
    "led_lighting_upgrade": {
        "category": "lighting",
        "title": "Switch to LED lighting",
        "description": "Replace all fluorescent/halogen lighting with LED equivalents",
        "savings_pct_of_electricity": 0.12,
        "payback_months": 18,
        "difficulty": "easy",
        "cost_range_gbp": (500, 5000),
        "annual_saving_range_gbp": (200, 3000),
    },
    "smart_heating_controls": {
        "category": "heating_cooling",
        "title": "Install smart heating controls",
        "description": "Install programmable thermostats and zone controls",
        "savings_pct_of_gas": 0.15,
        "payback_months": 12,
        "difficulty": "easy",
        "cost_range_gbp": (300, 2000),
        "annual_saving_range_gbp": (150, 1500),
    },
    "green_energy_tariff": {
        "category": "renewable_energy",
        "title": "Switch to 100% renewable electricity tariff",
        "description": "Switch electricity supply to a certified 100% renewable tariff",
        "savings_pct_of_electricity": 0.80,
        "payback_months": 0,
        "difficulty": "easy",
        "cost_range_gbp": (0, 500),
        "annual_saving_range_gbp": (0, 0),
    },
    "ev_fleet_transition": {
        "category": "transport",
        "title": "Transition company vehicles to EVs",
        "description": "Replace diesel/petrol company cars with electric vehicles on next lease cycle",
        "savings_pct_of_transport": 0.60,
        "payback_months": 36,
        "difficulty": "medium",
        "cost_range_gbp": (2000, 25000),
        "annual_saving_range_gbp": (500, 5000),
    },
    "reduce_business_travel": {
        "category": "transport",
        "title": "Reduce business travel by 30%",
        "description": "Replace face-to-face meetings with video conferencing where possible",
        "savings_pct_of_travel": 0.30,
        "payback_months": 0,
        "difficulty": "easy",
        "cost_range_gbp": (0, 500),
        "annual_saving_range_gbp": (1000, 10000),
    },
    "waste_recycling_programme": {
        "category": "waste_reduction",
        "title": "Implement comprehensive recycling",
        "description": "Segregate waste streams and partner with recycling providers",
        "savings_pct_of_waste": 0.40,
        "payback_months": 6,
        "difficulty": "easy",
        "cost_range_gbp": (200, 2000),
        "annual_saving_range_gbp": (100, 1000),
    },
    "power_management": {
        "category": "energy_efficiency",
        "title": "Enable power management on all devices",
        "description": "Configure sleep/hibernate on PCs, turn off monitors, smart power strips",
        "savings_pct_of_electricity": 0.08,
        "payback_months": 3,
        "difficulty": "easy",
        "cost_range_gbp": (0, 500),
        "annual_saving_range_gbp": (100, 1500),
    },
    "insulation_improvements": {
        "category": "heating_cooling",
        "title": "Improve building insulation",
        "description": "Draft-proofing, loft insulation, cavity wall insulation",
        "savings_pct_of_gas": 0.20,
        "payback_months": 24,
        "difficulty": "medium",
        "cost_range_gbp": (1000, 10000),
        "annual_saving_range_gbp": (200, 3000),
    },
    "cycle_to_work_scheme": {
        "category": "behaviour_change",
        "title": "Introduce cycle-to-work scheme",
        "description": "Offer salary sacrifice cycle scheme and improve bike parking",
        "savings_pct_of_commuting": 0.10,
        "payback_months": 0,
        "difficulty": "easy",
        "cost_range_gbp": (0, 1000),
        "annual_saving_range_gbp": (0, 500),
    },
    "sustainable_procurement": {
        "category": "procurement",
        "title": "Green procurement policy",
        "description": "Prioritise suppliers with carbon targets and environmental certifications",
        "savings_pct_of_scope3": 0.05,
        "payback_months": 0,
        "difficulty": "easy",
        "cost_range_gbp": (0, 0),
        "annual_saving_range_gbp": (0, 0),
    },
}

# SBTi 1.5C-aligned annual reduction rates
SBTI_15C_ANNUAL_REDUCTION_PCT = 4.2
SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT = 2.5

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Phase progress %")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")
    mobile_summary: str = Field(default="", description="Short mobile-friendly summary")

class SMEOrganizationProfile(BaseModel):
    """SME organization profile - collected in Phase 1 (5 min)."""

    organization_name: str = Field(..., min_length=1, description="Company name")
    industry_sector: str = Field(default="other", description="NACE-based sector")
    nace_code: str = Field(default="", description="NACE Rev 2 code (e.g., 62.01)")
    employee_count: int = Field(default=1, ge=1, le=1000, description="Number of employees")
    annual_revenue_gbp: float = Field(default=0.0, ge=0.0, description="Annual revenue in GBP")
    company_size: str = Field(default="micro", description="micro|small|medium|large_sme")
    country: str = Field(default="UK", description="ISO 3166-1 alpha-2")
    region: str = Field(default="", description="Sub-national region")
    postcode: str = Field(default="", description="Postal code for grant matching")
    num_sites: int = Field(default=1, ge=1, description="Number of operational sites")
    primary_energy_source: str = Field(default="electricity", description="Main energy source")
    has_company_vehicles: bool = Field(default=False, description="Operates company vehicles")
    entity_id: str = Field(default="", description="GreenLang entity ID")
    tenant_id: str = Field(default="", description="GreenLang tenant ID")

    @field_validator("industry_sector")
    @classmethod
    def _validate_sector(cls, v: str) -> str:
        allowed = {s.value for s in IndustrySector}
        if v not in allowed:
            return "other"
        return v

    @field_validator("company_size")
    @classmethod
    def _validate_size(cls, v: str) -> str:
        allowed = {s.value for s in CompanySizeBand}
        if v not in allowed:
            return "micro"
        return v

class BronzeBaselineInput(BaseModel):
    """Minimal input for Bronze-tier baseline (spend + headcount)."""

    annual_electricity_spend_gbp: float = Field(default=0.0, ge=0.0)
    annual_gas_spend_gbp: float = Field(default=0.0, ge=0.0)
    annual_fuel_spend_gbp: float = Field(default=0.0, ge=0.0)
    annual_travel_spend_gbp: float = Field(default=0.0, ge=0.0)
    annual_waste_spend_gbp: float = Field(default=0.0, ge=0.0)
    annual_procurement_spend_gbp: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=1, ge=1)
    grid_region: str = Field(default="UK", description="Grid region for EF lookup")

class BronzeBaseline(BaseModel):
    """Bronze-tier baseline result (spend-based, +/-25% accuracy)."""

    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_gas_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_fuel_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_electricity_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_travel_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_waste_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_procurement_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_commuting_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    per_employee_tco2e: float = Field(default=0.0, ge=0.0)
    accuracy_band: str = Field(default="+/-25%", description="Estimated accuracy")
    tier: str = Field(default="bronze")
    benchmark_vs_sector: str = Field(default="", description="above_average|average|below_average")
    electricity_kwh_estimated: float = Field(default=0.0, ge=0.0)
    gas_kwh_estimated: float = Field(default=0.0, ge=0.0)

class AutoTarget(BaseModel):
    """Auto-generated emission reduction target."""

    target_name: str = Field(default="", description="Target label")
    base_year: int = Field(default=2025, ge=2020, le=2030)
    base_year_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_year: int = Field(default=2030, ge=2025, le=2040)
    near_term_target_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    long_term_year: int = Field(default=2050, ge=2040, le=2060)
    long_term_target_tco2e: float = Field(default=0.0, ge=0.0)
    long_term_reduction_pct: float = Field(default=90.0, ge=0.0, le=100.0)
    annual_reduction_rate_pct: float = Field(default=4.2, ge=0.0, le=20.0)
    pathway_type: str = Field(default="1.5C", description="1.5C|well_below_2C|2C")
    pathway_points: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_eligible: bool = Field(default=True)

class QuickWinAction(BaseModel):
    """A recommended quick-win action for SME."""

    action_id: str = Field(default="", description="Unique action identifier")
    rank: int = Field(default=0, ge=0, description="Priority rank (1 = best)")
    title: str = Field(default="", description="Action title")
    description: str = Field(default="", description="What to do")
    category: str = Field(default="", description="Quick win category")
    estimated_savings_tco2e: float = Field(default=0.0, ge=0.0, description="Annual CO2 savings")
    estimated_savings_gbp: float = Field(default=0.0, ge=0.0, description="Annual cost savings")
    implementation_cost_gbp: float = Field(default=0.0, ge=0.0)
    payback_months: int = Field(default=0, ge=0, description="Simple payback period")
    difficulty: str = Field(default="easy", description="easy|medium|hard")
    savings_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)

class ExpressOnboardingConfig(BaseModel):
    """Configuration for express onboarding workflow."""

    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2025, ge=2020, le=2035)
    target_pathway: str = Field(default="1.5C", description="1.5C|well_below_2C")
    max_quick_wins: int = Field(default=5, ge=1, le=10)
    currency: str = Field(default="GBP", description="GBP|EUR|USD")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("target_pathway")
    @classmethod
    def _validate_pathway(cls, v: str) -> str:
        if v not in {"1.5C", "well_below_2C", "2C"}:
            return "1.5C"
        return v

class ExpressOnboardingInput(BaseModel):
    """Complete input for express onboarding workflow."""

    profile: SMEOrganizationProfile = Field(..., description="Organization profile")
    baseline_data: BronzeBaselineInput = Field(
        default_factory=BronzeBaselineInput,
        description="Bronze baseline spend data",
    )
    config: ExpressOnboardingConfig = Field(
        default_factory=ExpressOnboardingConfig,
    )

class ExpressOnboardingResult(BaseModel):
    """Complete result from express onboarding workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sme_express_onboarding")
    pack_id: str = Field(default="PACK-026")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    profile: SMEOrganizationProfile = Field(
        default_factory=lambda: SMEOrganizationProfile(organization_name="Unknown"),
    )
    baseline: BronzeBaseline = Field(default_factory=BronzeBaseline)
    targets: List[AutoTarget] = Field(default_factory=list)
    quick_wins: List[QuickWinAction] = Field(default_factory=list)
    total_quick_win_savings_tco2e: float = Field(default=0.0, ge=0.0)
    total_quick_win_savings_gbp: float = Field(default=0.0, ge=0.0)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ExpressOnboardingWorkflow:
    """
    4-phase express onboarding workflow for SME net-zero journeys.

    Designed for time-constrained SMEs who need to establish a baseline,
    set targets, and identify quick wins in under 20 minutes.

    Phase 1: Organization Profile (5 min)
        Collect basic company info: name, sector, size, location.

    Phase 2: Quick Baseline (5 min)
        Bronze-tier baseline from energy spend and employee count.
        Uses spend-to-emissions factors for instant calculation.

    Phase 3: Auto-Target (instant)
        Auto-generate 1.5C-aligned near-term and long-term targets
        with annual pathway points.

    Phase 4: Quick Wins (5 min)
        Identify top 5 quick-win actions ranked by CO2 savings,
        cost savings, and ease of implementation.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Workflow configuration.

    Example:
        >>> wf = ExpressOnboardingWorkflow()
        >>> inp = ExpressOnboardingInput(
        ...     profile=SMEOrganizationProfile(
        ...         organization_name="Acme Ltd",
        ...         industry_sector="office_services",
        ...         employee_count=25,
        ...     ),
        ...     baseline_data=BronzeBaselineInput(
        ...         annual_electricity_spend_gbp=12000,
        ...         annual_gas_spend_gbp=4000,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
        >>> assert len(result.quick_wins) <= 5
    """

    def __init__(self, config: Optional[ExpressOnboardingConfig] = None) -> None:
        """Initialise ExpressOnboardingWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config = config or ExpressOnboardingConfig()
        self._phase_results: List[PhaseResult] = []
        self._profile: Optional[SMEOrganizationProfile] = None
        self._baseline: BronzeBaseline = BronzeBaseline()
        self._targets: List[AutoTarget] = []
        self._quick_wins: List[QuickWinAction] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: ExpressOnboardingInput) -> ExpressOnboardingResult:
        """
        Execute the 4-phase express onboarding workflow.

        Args:
            input_data: Validated input with profile, spend data, and config.

        Returns:
            ExpressOnboardingResult with baseline, targets, and quick wins.

        Raises:
            ValueError: If critical input data is missing.
        """
        started_at = utcnow()
        self.config = input_data.config
        self._profile = input_data.profile
        self.logger.info(
            "Starting express onboarding workflow %s for %s",
            self.workflow_id, input_data.profile.organization_name,
        )

        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Organization Profile
            phase1 = await self._phase_organization_profile(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"OrganizationProfile failed: {phase1.errors}")

            # Phase 2: Quick Baseline
            phase2 = await self._phase_quick_baseline(input_data)
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise ValueError(f"QuickBaseline failed: {phase2.errors}")

            # Phase 3: Auto-Target
            phase3 = await self._phase_auto_target(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Quick Wins
            phase4 = await self._phase_quick_wins(input_data)
            self._phase_results.append(phase4)

            failed_phases = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed_phases else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Express onboarding failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error",
                phase_number=99,
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                mobile_summary="Onboarding failed. Please try again.",
            ))

        elapsed = (utcnow() - started_at).total_seconds()

        # Generate next steps
        next_steps = self._generate_next_steps()

        # Calculate totals
        total_qw_savings = sum(qw.estimated_savings_tco2e for qw in self._quick_wins)
        total_qw_cost_savings = sum(qw.estimated_savings_gbp for qw in self._quick_wins)

        result = ExpressOnboardingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            profile=self._profile or SMEOrganizationProfile(organization_name="Unknown"),
            baseline=self._baseline,
            targets=self._targets,
            quick_wins=self._quick_wins,
            total_quick_win_savings_tco2e=round(total_qw_savings, 4),
            total_quick_win_savings_gbp=round(total_qw_cost_savings, 2),
            next_steps=next_steps,
        )
        result.provenance_hash = self._provenance_of_result(result)
        self.logger.info(
            "Express onboarding %s completed in %.2fs status=%s total=%.1f tCO2e",
            self.workflow_id, elapsed, overall_status.value, self._baseline.total_tco2e,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Organization Profile
    # -------------------------------------------------------------------------

    async def _phase_organization_profile(
        self, input_data: ExpressOnboardingInput,
    ) -> PhaseResult:
        """Validate and enrich organization profile."""
        started = utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        profile = input_data.profile

        # Validate organization name
        if not profile.organization_name or len(profile.organization_name.strip()) < 2:
            errors.append("Organization name is required (minimum 2 characters)")

        # Auto-classify company size if not set
        if profile.employee_count < 10:
            profile.company_size = CompanySizeBand.MICRO.value
        elif profile.employee_count < 50:
            profile.company_size = CompanySizeBand.SMALL.value
        elif profile.employee_count < 250:
            profile.company_size = CompanySizeBand.MEDIUM.value
        else:
            profile.company_size = CompanySizeBand.LARGE_SME.value

        # Validate industry sector
        if profile.industry_sector not in {s.value for s in IndustrySector}:
            warnings.append(
                f"Unknown sector '{profile.industry_sector}'; defaulting to 'other'"
            )
            profile.industry_sector = IndustrySector.OTHER.value

        # Validate revenue
        if profile.annual_revenue_gbp <= 0:
            warnings.append("Revenue not provided; intensity metrics will be unavailable")

        # Validate location
        if not profile.country:
            profile.country = "UK"
            warnings.append("Country not provided; defaulting to UK")

        outputs["organization_name"] = profile.organization_name
        outputs["industry_sector"] = profile.industry_sector
        outputs["company_size"] = profile.company_size
        outputs["employee_count"] = profile.employee_count
        outputs["country"] = profile.country
        outputs["num_sites"] = profile.num_sites

        self._profile = profile

        elapsed = (utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        return PhaseResult(
            phase_name="organization_profile",
            phase_number=1,
            status=status,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if not errors else 0.0,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Profile: {profile.organization_name} ({profile.company_size}, {profile.employee_count} employees)",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Quick Baseline (Bronze)
    # -------------------------------------------------------------------------

    async def _phase_quick_baseline(
        self, input_data: ExpressOnboardingInput,
    ) -> PhaseResult:
        """Calculate Bronze-tier baseline from energy spend and headcount."""
        started = utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        bd = input_data.baseline_data
        profile = self._profile or input_data.profile
        sector = profile.industry_sector

        # --- Scope 1: Gas ---
        gas_kwh = 0.0
        scope1_gas = 0.0
        if bd.annual_gas_spend_gbp > 0:
            gas_cost_per_kwh = ENERGY_COST_PER_KWH_GBP.get("gas", 0.08)
            gas_kwh = bd.annual_gas_spend_gbp / gas_cost_per_kwh
            scope1_gas = (gas_kwh * GAS_EF_KGCO2E_PER_KWH) / 1000.0
        else:
            warnings.append("No gas spend provided; Scope 1 gas emissions estimated from sector average")
            # Estimate from sector per-employee factor (gas is ~30% of total for most sectors)
            sector_ef = SECTOR_PER_EMPLOYEE_EF_TCO2E.get(sector, 4.0)
            scope1_gas = profile.employee_count * sector_ef * 0.15

        # --- Scope 1: Fuel (company vehicles) ---
        scope1_fuel = 0.0
        if bd.annual_fuel_spend_gbp > 0:
            fuel_cost_per_kwh = ENERGY_COST_PER_KWH_GBP.get("oil", 0.10)
            fuel_kwh = bd.annual_fuel_spend_gbp / fuel_cost_per_kwh
            scope1_fuel = (fuel_kwh * 0.25301) / 1000.0  # Diesel EF
        elif profile.has_company_vehicles:
            # Rough estimate: 5 tCO2e per vehicle per year
            estimated_vehicles = max(1, profile.employee_count // 10)
            scope1_fuel = estimated_vehicles * 5.0
            warnings.append(f"No fuel spend; estimated {estimated_vehicles} vehicles at 5 tCO2e/vehicle")

        scope1_total = scope1_gas + scope1_fuel

        # --- Scope 2: Electricity ---
        elec_kwh = 0.0
        scope2_electricity = 0.0
        grid_region = bd.grid_region or profile.country or "UK"
        grid_ef = GRID_EF_KGCO2E_PER_KWH.get(grid_region, GRID_EF_KGCO2E_PER_KWH["GLOBAL"])

        if bd.annual_electricity_spend_gbp > 0:
            elec_cost_per_kwh = ENERGY_COST_PER_KWH_GBP.get("electricity", 0.28)
            elec_kwh = bd.annual_electricity_spend_gbp / elec_cost_per_kwh
            scope2_electricity = (elec_kwh * grid_ef) / 1000.0
        else:
            warnings.append("No electricity spend provided; estimated from sector average")
            sector_ef = SECTOR_PER_EMPLOYEE_EF_TCO2E.get(sector, 4.0)
            scope2_electricity = profile.employee_count * sector_ef * 0.35

        # --- Scope 3: Travel ---
        scope3_travel = 0.0
        if bd.annual_travel_spend_gbp > 0:
            scope3_travel = bd.annual_travel_spend_gbp * 0.00026  # 0.26 kgCO2e/GBP
        else:
            # Estimate from employee count
            scope3_travel = profile.employee_count * 0.5  # 0.5 tCO2e/employee/year average
            if scope3_travel > 0:
                warnings.append("No travel spend; estimated from employee count")

        # --- Scope 3: Waste ---
        scope3_waste = 0.0
        if bd.annual_waste_spend_gbp > 0:
            scope3_waste = bd.annual_waste_spend_gbp * 0.00058  # 0.58 kgCO2e/GBP
        else:
            scope3_waste = profile.employee_count * 0.1  # 0.1 tCO2e/employee/year
            warnings.append("No waste spend; estimated from employee count")

        # --- Scope 3: Procurement ---
        scope3_procurement = 0.0
        if bd.annual_procurement_spend_gbp > 0:
            sector_ef_spend = SECTOR_SPEND_EF_TCO2E_PER_1000GBP.get(sector, 0.098)
            scope3_procurement = (bd.annual_procurement_spend_gbp / 1000.0) * sector_ef_spend
        else:
            # Estimate Scope 3 from per-employee sector benchmarks (50% of total is Scope 3)
            sector_ef = SECTOR_PER_EMPLOYEE_EF_TCO2E.get(sector, 4.0)
            scope3_procurement = profile.employee_count * sector_ef * 0.25
            warnings.append("No procurement spend; estimated from sector benchmarks")

        # --- Scope 3: Commuting ---
        scope3_commuting = profile.employee_count * 0.3  # 0.3 tCO2e/employee/year average

        scope3_total = scope3_travel + scope3_waste + scope3_procurement + scope3_commuting
        total = scope1_total + scope2_electricity + scope3_total
        per_employee = total / max(profile.employee_count, 1)

        # Benchmark against sector
        sector_avg = SECTOR_PER_EMPLOYEE_EF_TCO2E.get(sector, 4.0)
        if per_employee > sector_avg * 1.2:
            benchmark = "above_average"
        elif per_employee < sector_avg * 0.8:
            benchmark = "below_average"
        else:
            benchmark = "average"

        self._baseline = BronzeBaseline(
            scope1_tco2e=round(scope1_total, 4),
            scope1_gas_tco2e=round(scope1_gas, 4),
            scope1_fuel_tco2e=round(scope1_fuel, 4),
            scope2_electricity_tco2e=round(scope2_electricity, 4),
            scope3_travel_tco2e=round(scope3_travel, 4),
            scope3_waste_tco2e=round(scope3_waste, 4),
            scope3_procurement_tco2e=round(scope3_procurement, 4),
            scope3_commuting_tco2e=round(scope3_commuting, 4),
            scope3_total_tco2e=round(scope3_total, 4),
            total_tco2e=round(total, 4),
            per_employee_tco2e=round(per_employee, 4),
            accuracy_band="+/-25%",
            tier="bronze",
            benchmark_vs_sector=benchmark,
            electricity_kwh_estimated=round(elec_kwh, 2),
            gas_kwh_estimated=round(gas_kwh, 2),
        )

        outputs["scope1_tco2e"] = self._baseline.scope1_tco2e
        outputs["scope2_tco2e"] = self._baseline.scope2_electricity_tco2e
        outputs["scope3_tco2e"] = self._baseline.scope3_total_tco2e
        outputs["total_tco2e"] = self._baseline.total_tco2e
        outputs["per_employee_tco2e"] = self._baseline.per_employee_tco2e
        outputs["benchmark"] = benchmark
        outputs["accuracy"] = "+/-25%"

        if total <= 0:
            errors.append("Total emissions are zero; ensure at least one spend figure is provided")

        elapsed = (utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        self.logger.info(
            "Bronze baseline: S1=%.1f S2=%.1f S3=%.1f Total=%.1f tCO2e (%s vs sector)",
            scope1_total, scope2_electricity, scope3_total, total, benchmark,
        )
        return PhaseResult(
            phase_name="quick_baseline",
            phase_number=2,
            status=status,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if not errors else 0.0,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Baseline: {total:.1f} tCO2e ({per_employee:.1f}/employee, {benchmark})",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Auto-Target
    # -------------------------------------------------------------------------

    async def _phase_auto_target(
        self, input_data: ExpressOnboardingInput,
    ) -> PhaseResult:
        """Auto-generate 1.5C-aligned targets with pathway points."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        base_total = self._baseline.total_tco2e
        base_year = self.config.base_year
        pathway = self.config.target_pathway

        if pathway == "1.5C":
            annual_rate = SBTI_15C_ANNUAL_REDUCTION_PCT
            near_term_reduction = 50.0  # 50% by 2030 (SBTi 1.5C)
            long_term_reduction = 90.0  # 90% by 2050
        elif pathway == "well_below_2C":
            annual_rate = SBTI_WELL_BELOW_2C_ANNUAL_REDUCTION_PCT
            near_term_reduction = 30.0
            long_term_reduction = 90.0
        else:
            annual_rate = SBTI_15C_ANNUAL_REDUCTION_PCT
            near_term_reduction = 50.0
            long_term_reduction = 90.0

        near_term_year = 2030
        long_term_year = 2050

        # Near-term target
        near_term_tco2e = base_total * (1.0 - near_term_reduction / 100.0)

        # Long-term target
        long_term_tco2e = base_total * (1.0 - long_term_reduction / 100.0)

        # Build annual pathway points
        pathway_points: List[Dict[str, Any]] = []
        for year in range(base_year, long_term_year + 1):
            years_from_base = year - base_year
            if year <= near_term_year:
                # Linear interpolation to near-term target
                years_to_nt = near_term_year - base_year
                if years_to_nt > 0:
                    reduction = (near_term_reduction / years_to_nt) * years_from_base
                else:
                    reduction = 0.0
            else:
                # Linear interpolation from near-term to long-term
                years_post_nt = year - near_term_year
                years_nt_to_lt = long_term_year - near_term_year
                additional_reduction = (
                    (long_term_reduction - near_term_reduction) / years_nt_to_lt
                ) * years_post_nt if years_nt_to_lt > 0 else 0.0
                reduction = near_term_reduction + additional_reduction

            target_tco2e = max(base_total * (1.0 - reduction / 100.0), 0.0)
            pathway_points.append({
                "year": year,
                "target_tco2e": round(target_tco2e, 4),
                "reduction_pct": round(min(reduction, 100.0), 2),
            })

        target = AutoTarget(
            target_name=f"SME Net Zero Target ({pathway})",
            base_year=base_year,
            base_year_tco2e=round(base_total, 4),
            near_term_year=near_term_year,
            near_term_target_tco2e=round(near_term_tco2e, 4),
            near_term_reduction_pct=near_term_reduction,
            long_term_year=long_term_year,
            long_term_target_tco2e=round(long_term_tco2e, 4),
            long_term_reduction_pct=long_term_reduction,
            annual_reduction_rate_pct=annual_rate,
            pathway_type=pathway,
            pathway_points=pathway_points,
            sbti_eligible=True,
        )

        self._targets = [target]

        # Validate target ambition
        if base_total < 10:
            warnings.append(
                "Baseline is very low (<10 tCO2e); targets may be difficult to measure accurately"
            )

        outputs["target_name"] = target.target_name
        outputs["near_term_reduction_pct"] = near_term_reduction
        outputs["near_term_year"] = near_term_year
        outputs["near_term_target_tco2e"] = target.near_term_target_tco2e
        outputs["long_term_reduction_pct"] = long_term_reduction
        outputs["annual_rate_pct"] = annual_rate
        outputs["pathway_points_count"] = len(pathway_points)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Auto-target: %s - %.0f%% by %d, %.0f%% by %d (%.1f%%/yr)",
            pathway, near_term_reduction, near_term_year,
            long_term_reduction, long_term_year, annual_rate,
        )
        return PhaseResult(
            phase_name="auto_target",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Target: {near_term_reduction:.0f}% reduction by {near_term_year}",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Quick Wins
    # -------------------------------------------------------------------------

    async def _phase_quick_wins(
        self, input_data: ExpressOnboardingInput,
    ) -> PhaseResult:
        """Identify top quick-win actions ranked by impact and feasibility."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        profile = self._profile or input_data.profile
        baseline = self._baseline
        max_wins = self.config.max_quick_wins

        # Score each quick win for this SME
        scored_wins: List[Tuple[float, QuickWinAction]] = []

        for action_id, action_def in QUICK_WIN_DATABASE.items():
            estimated_savings = self._estimate_quick_win_savings(
                action_id, action_def, baseline, profile,
            )
            if estimated_savings <= 0:
                continue

            cost_low, cost_high = action_def.get("cost_range_gbp", (0, 0))
            save_low, save_high = action_def.get("annual_saving_range_gbp", (0, 0))

            # Scale cost and savings by company size
            size_factor = self._size_factor(profile)
            impl_cost = ((cost_low + cost_high) / 2.0) * size_factor
            annual_saving_gbp = ((save_low + save_high) / 2.0) * size_factor

            savings_pct = (estimated_savings / max(baseline.total_tco2e, 0.01)) * 100.0

            # Composite score: savings * ease / (cost + 1)
            difficulty = action_def.get("difficulty", "easy")
            ease_score = {"easy": 3.0, "medium": 2.0, "hard": 1.0}.get(difficulty, 1.0)
            composite = (estimated_savings * ease_score) / (impl_cost / 1000.0 + 1.0)

            qw = QuickWinAction(
                action_id=action_id,
                rank=0,
                title=action_def.get("title", ""),
                description=action_def.get("description", ""),
                category=action_def.get("category", ""),
                estimated_savings_tco2e=round(estimated_savings, 4),
                estimated_savings_gbp=round(annual_saving_gbp, 2),
                implementation_cost_gbp=round(impl_cost, 2),
                payback_months=action_def.get("payback_months", 0),
                difficulty=difficulty,
                savings_pct_of_total=round(savings_pct, 2),
            )
            scored_wins.append((composite, qw))

        # Sort by composite score descending, take top N
        scored_wins.sort(key=lambda x: x[0], reverse=True)
        top_wins = scored_wins[:max_wins]

        self._quick_wins = []
        for rank, (score, qw) in enumerate(top_wins, 1):
            qw.rank = rank
            self._quick_wins.append(qw)

        total_savings = sum(qw.estimated_savings_tco2e for qw in self._quick_wins)
        total_cost_savings = sum(qw.estimated_savings_gbp for qw in self._quick_wins)
        total_impl_cost = sum(qw.implementation_cost_gbp for qw in self._quick_wins)

        outputs["quick_win_count"] = len(self._quick_wins)
        outputs["total_savings_tco2e"] = round(total_savings, 4)
        outputs["total_cost_savings_gbp"] = round(total_cost_savings, 2)
        outputs["total_implementation_cost_gbp"] = round(total_impl_cost, 2)
        outputs["savings_pct_of_baseline"] = round(
            (total_savings / max(baseline.total_tco2e, 0.01)) * 100, 2,
        )

        if not self._quick_wins:
            warnings.append("No quick wins identified; baseline data may be insufficient")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Quick wins: %d actions, total savings %.1f tCO2e (%.1f%% of baseline)",
            len(self._quick_wins), total_savings,
            (total_savings / max(baseline.total_tco2e, 0.01)) * 100,
        )
        return PhaseResult(
            phase_name="quick_wins",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Quick wins: {len(self._quick_wins)} actions, save {total_savings:.1f} tCO2e/yr",
        )

    def _estimate_quick_win_savings(
        self,
        action_id: str,
        action_def: Dict[str, Any],
        baseline: BronzeBaseline,
        profile: SMEOrganizationProfile,
    ) -> float:
        """Estimate tCO2e savings for a specific quick win action."""
        savings = 0.0

        if "savings_pct_of_electricity" in action_def:
            savings = baseline.scope2_electricity_tco2e * action_def["savings_pct_of_electricity"]
        elif "savings_pct_of_gas" in action_def:
            savings = baseline.scope1_gas_tco2e * action_def["savings_pct_of_gas"]
        elif "savings_pct_of_transport" in action_def:
            savings = baseline.scope1_fuel_tco2e * action_def["savings_pct_of_transport"]
        elif "savings_pct_of_travel" in action_def:
            savings = baseline.scope3_travel_tco2e * action_def["savings_pct_of_travel"]
        elif "savings_pct_of_waste" in action_def:
            savings = baseline.scope3_waste_tco2e * action_def["savings_pct_of_waste"]
        elif "savings_pct_of_commuting" in action_def:
            savings = baseline.scope3_commuting_tco2e * action_def["savings_pct_of_commuting"]
        elif "savings_pct_of_scope3" in action_def:
            savings = baseline.scope3_total_tco2e * action_def["savings_pct_of_scope3"]

        return max(savings, 0.0)

    def _size_factor(self, profile: SMEOrganizationProfile) -> float:
        """Return scaling factor based on company size."""
        factors = {
            "micro": 0.3,
            "small": 0.6,
            "medium": 1.0,
            "large_sme": 1.5,
        }
        return factors.get(profile.company_size, 1.0)

    # -------------------------------------------------------------------------
    # Next Steps & Provenance
    # -------------------------------------------------------------------------

    def _generate_next_steps(self) -> List[str]:
        """Generate next-step recommendations after express onboarding."""
        steps: List[str] = []

        steps.append(
            "Review your baseline and quick wins in the SME dashboard."
        )

        if self._baseline.tier == "bronze":
            steps.append(
                "Upgrade to Silver baseline by collecting actual energy bills "
                "(Standard Setup Workflow, ~1-2 hours)."
            )

        if self._quick_wins:
            steps.append(
                f"Start implementing your top quick win: {self._quick_wins[0].title}."
            )

        steps.append(
            "Set up quarterly review reminders to track progress "
            "(Quarterly Review Workflow, ~15-30 min)."
        )

        steps.append(
            "Explore available grants for your sector and region "
            "(Grant Application Workflow)."
        )

        steps.append(
            "Consider certification pathways: SME Climate Hub, B Corp, "
            "or ISO 14001 (Certification Pathway Workflow)."
        )

        return steps

    def _provenance_of_result(self, result: ExpressOnboardingResult) -> str:
        """Compute SHA-256 provenance hash of the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)
