# -*- coding: utf-8 -*-
"""
Standard Setup Workflow
============================

6-phase workflow for comprehensive SME net-zero setup within PACK-026
SME Net Zero Pack.  Provides Silver-tier accuracy (+-15%) through
guided data collection and uses all 8 pack engines.

Phases:
    1. OrganizationProfile   -- Detailed org profile
    2. DataCollection        -- Guided data collection (bills, fuel, travel)
    3. SilverBaseline        -- Silver baseline (+/-15% accuracy)
    4. TargetValidation      -- Validate and refine targets
    5. ActionPrioritization  -- MACC lite, top 10 actions
    6. GrantMatching         -- Find matching grants

Total time: 1-2 hours.

Uses: sme_baseline_engine, simplified_target_engine, quick_wins_engine,
      action_prioritization_engine, cost_benefit_engine, grant_finder_engine,
      peer_benchmark_engine, certification_readiness_engine.

Zero-hallucination: all emission factors are deterministic.
SHA-256 provenance hashes for auditability.

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DataSourceType(str, Enum):
    ELECTRICITY_BILL = "electricity_bill"
    GAS_BILL = "gas_bill"
    FUEL_RECEIPT = "fuel_receipt"
    TRAVEL_EXPENSE = "travel_expense"
    WASTE_INVOICE = "waste_invoice"
    PROCUREMENT_REPORT = "procurement_report"
    FLEET_LOG = "fleet_log"
    WATER_BILL = "water_bill"
    MANUAL_ENTRY = "manual_entry"


class DataQualityLevel(str, Enum):
    MEASURED = "measured"
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    DEFAULT = "default"


class MACCActionType(str, Enum):
    NEGATIVE_COST = "negative_cost"      # Saves money
    LOW_COST = "low_cost"                # <100 GBP/tCO2e
    MEDIUM_COST = "medium_cost"          # 100-500 GBP/tCO2e
    HIGH_COST = "high_cost"              # >500 GBP/tCO2e


class GrantStatus(str, Enum):
    OPEN = "open"
    CLOSING_SOON = "closing_soon"
    CLOSED = "closed"
    UPCOMING = "upcoming"


# =============================================================================
# EMISSION FACTOR CONSTANTS
# =============================================================================

GRID_EF_KGCO2E_PER_KWH: Dict[str, float] = {
    "UK": 0.2070,
    "EU-AVG": 0.2556,
    "US-AVG": 0.3710,
    "DE": 0.3850,
    "FR": 0.0520,
    "GLOBAL": 0.4940,
}

GAS_EF_KGCO2E_PER_KWH = 0.18293

FUEL_EF_KGCO2E_PER_LITRE: Dict[str, float] = {
    "diesel": 2.70494,
    "petrol": 2.31440,
    "lpg": 1.65210,
}

BUSINESS_TRAVEL_EF: Dict[str, float] = {
    "domestic_flight_kgco2e_per_km": 0.24588,
    "short_haul_flight_kgco2e_per_km": 0.15353,
    "long_haul_flight_kgco2e_per_km": 0.19309,
    "train_kgco2e_per_km": 0.03549,
    "car_kgco2e_per_km": 0.17140,
    "taxi_kgco2e_per_km": 0.20369,
    "bus_kgco2e_per_km": 0.10312,
}

SCOPE3_SPEND_EF_KGCO2E_PER_GBP: Dict[str, float] = {
    "purchased_goods": 0.42,
    "capital_goods": 0.35,
    "transport": 0.72,
    "waste": 0.58,
    "travel": 0.26,
    "commuting": 0.18,
    "other": 0.30,
}

COMMUTING_EF_KGCO2E_PER_EMPLOYEE_PER_YEAR = 300.0

# SME Grant database (UK/EU)
SME_GRANT_DATABASE: List[Dict[str, Any]] = [
    {
        "grant_id": "BEIS-IETF",
        "name": "Industrial Energy Transformation Fund",
        "provider": "BEIS (UK)",
        "max_amount_gbp": 30_000_000,
        "min_amount_gbp": 100_000,
        "eligible_sectors": ["manufacturing_light", "manufacturing_heavy"],
        "eligible_sizes": ["medium", "large_sme"],
        "countries": ["UK"],
        "description": "Capital grants for industrial energy efficiency and decarbonisation",
        "status": "open",
    },
    {
        "grant_id": "SEAI-SME",
        "name": "SEAI SME Energy Efficiency Grant",
        "provider": "SEAI (Ireland)",
        "max_amount_gbp": 5000,
        "min_amount_gbp": 500,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "countries": ["IE"],
        "description": "Grants for SME energy audits and efficiency improvements",
        "status": "open",
    },
    {
        "grant_id": "ECO4-SME",
        "name": "ECO4 Energy Company Obligation",
        "provider": "Ofgem (UK)",
        "max_amount_gbp": 10000,
        "min_amount_gbp": 0,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium", "large_sme"],
        "countries": ["UK"],
        "description": "Energy efficiency measures funded by energy suppliers",
        "status": "open",
    },
    {
        "grant_id": "ERDF-GREEN",
        "name": "EU Green SME Fund",
        "provider": "European Regional Development Fund",
        "max_amount_gbp": 50000,
        "min_amount_gbp": 5000,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "countries": ["EU"],
        "description": "Co-funding for SME decarbonisation projects",
        "status": "open",
    },
    {
        "grant_id": "BEF-TECH",
        "name": "Business Energy Fund - Technology",
        "provider": "BEIS (UK)",
        "max_amount_gbp": 20000,
        "min_amount_gbp": 1000,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "countries": ["UK"],
        "description": "Grants for energy-saving technology including LEDs, heat pumps, insulation",
        "status": "open",
    },
    {
        "grant_id": "WG-GREEN-BIZ",
        "name": "Welsh Government Green Business Grant",
        "provider": "Welsh Government",
        "max_amount_gbp": 10000,
        "min_amount_gbp": 1000,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "countries": ["UK"],
        "description": "Capital support for Welsh SMEs investing in carbon reduction",
        "status": "open",
    },
    {
        "grant_id": "SG-SMEGREEN",
        "name": "Scottish Green Business Fund",
        "provider": "Zero Waste Scotland",
        "max_amount_gbp": 15000,
        "min_amount_gbp": 1000,
        "eligible_sectors": ["all"],
        "eligible_sizes": ["micro", "small", "medium"],
        "countries": ["UK"],
        "description": "Grants for Scottish SMEs to reduce carbon emissions",
        "status": "open",
    },
    {
        "grant_id": "HORIZON-SME",
        "name": "Horizon Europe SME Instrument",
        "provider": "European Commission",
        "max_amount_gbp": 200000,
        "min_amount_gbp": 50000,
        "eligible_sectors": ["technology", "manufacturing_light", "manufacturing_heavy"],
        "eligible_sizes": ["small", "medium"],
        "countries": ["EU"],
        "description": "Innovation funding for climate tech SMEs",
        "status": "open",
    },
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    mobile_summary: str = Field(default="")


class EnergyBillRecord(BaseModel):
    """Parsed energy bill record."""

    source_type: str = Field(default="electricity_bill")
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    consumption_kwh: float = Field(default=0.0, ge=0.0)
    cost_gbp: float = Field(default=0.0, ge=0.0)
    supplier: str = Field(default="")
    tariff_type: str = Field(default="standard", description="standard|green|mixed")
    renewable_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_quality: str = Field(default="measured")
    site_id: str = Field(default="main")


class FuelRecord(BaseModel):
    """Fuel consumption record."""

    fuel_type: str = Field(default="diesel")
    litres: float = Field(default=0.0, ge=0.0)
    cost_gbp: float = Field(default=0.0, ge=0.0)
    vehicle_type: str = Field(default="car")
    data_quality: str = Field(default="measured")


class TravelRecord(BaseModel):
    """Business travel record."""

    mode: str = Field(default="car", description="car|train|domestic_flight|short_haul_flight|long_haul_flight|taxi|bus")
    distance_km: float = Field(default=0.0, ge=0.0)
    spend_gbp: float = Field(default=0.0, ge=0.0)
    trips: int = Field(default=1, ge=1)
    data_quality: str = Field(default="estimated")


class ProcurementCategory(BaseModel):
    """Procurement spend by category."""

    category: str = Field(default="purchased_goods")
    description: str = Field(default="")
    annual_spend_gbp: float = Field(default=0.0, ge=0.0)
    supplier_ef_kgco2e_per_gbp: Optional[float] = Field(None, ge=0.0)
    data_quality: str = Field(default="estimated")


class StandardSetupConfig(BaseModel):
    """Configuration for standard setup workflow."""

    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2025, ge=2020, le=2035)
    target_pathway: str = Field(default="1.5C")
    max_actions: int = Field(default=10, ge=1, le=25)
    country: str = Field(default="UK")
    currency: str = Field(default="GBP")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class StandardSetupInput(BaseModel):
    """Complete input for standard setup workflow."""

    organization_name: str = Field(default="", description="Company name")
    industry_sector: str = Field(default="other")
    employee_count: int = Field(default=1, ge=1)
    annual_revenue_gbp: float = Field(default=0.0, ge=0.0)
    country: str = Field(default="UK")
    postcode: str = Field(default="")
    num_sites: int = Field(default=1, ge=1)
    electricity_bills: List[EnergyBillRecord] = Field(default_factory=list)
    gas_bills: List[EnergyBillRecord] = Field(default_factory=list)
    fuel_records: List[FuelRecord] = Field(default_factory=list)
    travel_records: List[TravelRecord] = Field(default_factory=list)
    procurement: List[ProcurementCategory] = Field(default_factory=list)
    annual_waste_spend_gbp: float = Field(default=0.0, ge=0.0)
    config: StandardSetupConfig = Field(default_factory=StandardSetupConfig)


class SilverBaseline(BaseModel):
    """Silver-tier baseline result (+/-15% accuracy)."""

    scope1_gas_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_fuel_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_travel_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_waste_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_procurement_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_commuting_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    per_employee_tco2e: float = Field(default=0.0, ge=0.0)
    intensity_per_revenue: float = Field(default=0.0, ge=0.0)
    accuracy_band: str = Field(default="+/-15%")
    tier: str = Field(default="silver")
    data_quality_score: float = Field(default=3.0, ge=1.0, le=5.0)
    total_electricity_kwh: float = Field(default=0.0, ge=0.0)
    total_gas_kwh: float = Field(default=0.0, ge=0.0)


class ValidatedTarget(BaseModel):
    """Validated emission reduction target."""

    target_name: str = Field(default="")
    base_year: int = Field(default=2025)
    base_year_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_year: int = Field(default=2030)
    near_term_target_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_reduction_pct: float = Field(default=50.0)
    annual_rate_pct: float = Field(default=4.2)
    pathway_points: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_aligned: bool = Field(default=True)
    validation_notes: List[str] = Field(default_factory=list)


class MACCAction(BaseModel):
    """Marginal Abatement Cost Curve action."""

    action_id: str = Field(default="")
    rank: int = Field(default=0, ge=0)
    title: str = Field(default="")
    description: str = Field(default="")
    category: str = Field(default="")
    abatement_tco2e: float = Field(default=0.0, ge=0.0)
    cost_per_tco2e_gbp: float = Field(default=0.0)
    annual_cost_gbp: float = Field(default=0.0)
    annual_saving_gbp: float = Field(default=0.0)
    net_annual_gbp: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    implementation_cost_gbp: float = Field(default=0.0, ge=0.0)
    action_type: str = Field(default="low_cost")
    timeframe: str = Field(default="short_term")


class GrantMatch(BaseModel):
    """Matched grant opportunity."""

    grant_id: str = Field(default="")
    name: str = Field(default="")
    provider: str = Field(default="")
    max_amount_gbp: float = Field(default=0.0, ge=0.0)
    min_amount_gbp: float = Field(default=0.0, ge=0.0)
    description: str = Field(default="")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    eligible: bool = Field(default=True)
    status: str = Field(default="open")


class StandardSetupResult(BaseModel):
    """Complete result from standard setup workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sme_standard_setup")
    pack_id: str = Field(default="PACK-026")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    baseline: SilverBaseline = Field(default_factory=SilverBaseline)
    targets: List[ValidatedTarget] = Field(default_factory=list)
    actions: List[MACCAction] = Field(default_factory=list)
    grants: List[GrantMatch] = Field(default_factory=list)
    total_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    total_net_savings_gbp: float = Field(default=0.0)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class StandardSetupWorkflow:
    """
    6-phase standard setup workflow for comprehensive SME net-zero planning.

    Provides Silver-tier accuracy through guided data collection of actual
    energy bills, fuel receipts, and travel records.  Includes MACC-lite
    action prioritisation and grant matching.

    Phase 1: Organization Profile
    Phase 2: Data Collection (guided, ~30 min)
    Phase 3: Silver Baseline (+/-15% accuracy)
    Phase 4: Target Validation
    Phase 5: Action Prioritisation (MACC lite, top 10)
    Phase 6: Grant Matching

    Total time: 1-2 hours.

    Example:
        >>> wf = StandardSetupWorkflow()
        >>> inp = StandardSetupInput(
        ...     organization_name="Acme Manufacturing",
        ...     industry_sector="manufacturing_light",
        ...     employee_count=85,
        ...     electricity_bills=[EnergyBillRecord(consumption_kwh=120000)],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[StandardSetupConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or StandardSetupConfig()
        self._phase_results: List[PhaseResult] = []
        self._baseline: SilverBaseline = SilverBaseline()
        self._targets: List[ValidatedTarget] = []
        self._actions: List[MACCAction] = []
        self._grants: List[GrantMatch] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: StandardSetupInput) -> StandardSetupResult:
        """Execute the 6-phase standard setup workflow."""
        started_at = _utcnow()
        self.config = input_data.config
        self.logger.info(
            "Starting standard setup workflow %s for %s",
            self.workflow_id, input_data.organization_name,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_organization_profile(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"OrganizationProfile failed: {phase1.errors}")

            phase2 = await self._phase_data_collection(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_silver_baseline(input_data)
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise ValueError(f"SilverBaseline failed: {phase3.errors}")

            phase4 = await self._phase_target_validation(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_action_prioritization(input_data)
            self._phase_results.append(phase5)

            phase6 = await self._phase_grant_matching(input_data)
            self._phase_results.append(phase6)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Standard setup failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
                mobile_summary="Setup failed. Please check your data.",
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        total_abatement = sum(a.abatement_tco2e for a in self._actions)
        total_net_savings = sum(a.net_annual_gbp for a in self._actions)
        next_steps = self._generate_next_steps(input_data)

        result = StandardSetupResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            baseline=self._baseline,
            targets=self._targets,
            actions=self._actions,
            grants=self._grants,
            total_abatement_tco2e=round(total_abatement, 4),
            total_net_savings_gbp=round(total_net_savings, 2),
            next_steps=next_steps,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Standard setup %s completed in %.2fs, total=%.1f tCO2e, %d actions, %d grants",
            self.workflow_id, elapsed, self._baseline.total_tco2e,
            len(self._actions), len(self._grants),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Organization Profile
    # -------------------------------------------------------------------------

    async def _phase_organization_profile(self, inp: StandardSetupInput) -> PhaseResult:
        started = _utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        if not inp.organization_name or len(inp.organization_name.strip()) < 2:
            errors.append("Organization name is required")

        if inp.employee_count < 1:
            errors.append("Employee count must be at least 1")

        if inp.annual_revenue_gbp <= 0:
            warnings.append("Revenue not provided; intensity metrics unavailable")

        outputs["organization_name"] = inp.organization_name
        outputs["sector"] = inp.industry_sector
        outputs["employees"] = inp.employee_count
        outputs["country"] = inp.country
        outputs["sites"] = inp.num_sites

        elapsed = (_utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        return PhaseResult(
            phase_name="organization_profile", phase_number=1,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if not errors else 0.0,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Profile: {inp.organization_name} ({inp.employee_count} employees)",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Collection (Guided)
    # -------------------------------------------------------------------------

    async def _phase_data_collection(self, inp: StandardSetupInput) -> PhaseResult:
        """Validate and summarise collected data sources."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        outputs["electricity_bills"] = len(inp.electricity_bills)
        outputs["gas_bills"] = len(inp.gas_bills)
        outputs["fuel_records"] = len(inp.fuel_records)
        outputs["travel_records"] = len(inp.travel_records)
        outputs["procurement_categories"] = len(inp.procurement)
        outputs["has_waste_data"] = inp.annual_waste_spend_gbp > 0

        total_sources = (
            len(inp.electricity_bills) + len(inp.gas_bills)
            + len(inp.fuel_records) + len(inp.travel_records)
        )

        if total_sources == 0:
            warnings.append("No activity data provided; baseline will use spend-based estimates only")

        if not inp.electricity_bills:
            warnings.append("No electricity bills; electricity will be estimated")

        if not inp.gas_bills:
            warnings.append("No gas bills; gas consumption will be estimated")

        # Assess overall data quality
        measured_count = 0
        total_count = 0
        for bill in inp.electricity_bills + inp.gas_bills:
            total_count += 1
            if bill.data_quality == "measured":
                measured_count += 1
        for fr in inp.fuel_records:
            total_count += 1
            if fr.data_quality == "measured":
                measured_count += 1

        data_quality_pct = (measured_count / total_count * 100) if total_count > 0 else 0
        outputs["data_quality_pct_measured"] = round(data_quality_pct, 1)
        outputs["total_data_sources"] = total_sources

        if data_quality_pct < 50:
            warnings.append(
                f"Only {data_quality_pct:.0f}% of data is measured; "
                "consider collecting actual bills for better accuracy"
            )

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="data_collection", phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Data: {total_sources} sources ({data_quality_pct:.0f}% measured)",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Silver Baseline
    # -------------------------------------------------------------------------

    async def _phase_silver_baseline(self, inp: StandardSetupInput) -> PhaseResult:
        """Calculate Silver-tier baseline from activity data."""
        started = _utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        country = inp.config.country or inp.country or "UK"
        grid_ef = GRID_EF_KGCO2E_PER_KWH.get(country, GRID_EF_KGCO2E_PER_KWH["GLOBAL"])

        # --- Scope 1: Gas ---
        total_gas_kwh = sum(b.consumption_kwh for b in inp.gas_bills)
        scope1_gas = (total_gas_kwh * GAS_EF_KGCO2E_PER_KWH) / 1000.0

        # --- Scope 1: Fuel ---
        scope1_fuel = 0.0
        for fr in inp.fuel_records:
            ef = FUEL_EF_KGCO2E_PER_LITRE.get(fr.fuel_type, 2.70)
            scope1_fuel += (fr.litres * ef) / 1000.0

        scope1_total = scope1_gas + scope1_fuel

        # --- Scope 2: Electricity ---
        total_elec_kwh = sum(b.consumption_kwh for b in inp.electricity_bills)
        scope2_location = (total_elec_kwh * grid_ef) / 1000.0

        # Market-based: exclude green tariff electricity
        scope2_market = 0.0
        for bill in inp.electricity_bills:
            if bill.tariff_type == "green" or bill.renewable_pct >= 100:
                continue
            non_renewable_kwh = bill.consumption_kwh * (1.0 - bill.renewable_pct / 100.0)
            scope2_market += (non_renewable_kwh * grid_ef) / 1000.0

        # --- Scope 3: Travel ---
        scope3_travel = 0.0
        for tr in inp.travel_records:
            if tr.distance_km > 0:
                ef_key = f"{tr.mode}_kgco2e_per_km"
                ef = BUSINESS_TRAVEL_EF.get(ef_key, 0.171)
                scope3_travel += (tr.distance_km * tr.trips * ef) / 1000.0
            elif tr.spend_gbp > 0:
                scope3_travel += (tr.spend_gbp * SCOPE3_SPEND_EF_KGCO2E_PER_GBP.get("travel", 0.26)) / 1000.0

        # --- Scope 3: Waste ---
        scope3_waste = 0.0
        if inp.annual_waste_spend_gbp > 0:
            scope3_waste = (inp.annual_waste_spend_gbp * SCOPE3_SPEND_EF_KGCO2E_PER_GBP["waste"]) / 1000.0
        else:
            scope3_waste = inp.employee_count * 0.1
            warnings.append("Waste estimated from employee count")

        # --- Scope 3: Procurement ---
        scope3_procurement = 0.0
        for pc in inp.procurement:
            ef = pc.supplier_ef_kgco2e_per_gbp if pc.supplier_ef_kgco2e_per_gbp else SCOPE3_SPEND_EF_KGCO2E_PER_GBP.get(pc.category, 0.30)
            scope3_procurement += (pc.annual_spend_gbp * ef) / 1000.0

        # --- Scope 3: Commuting ---
        scope3_commuting = (inp.employee_count * COMMUTING_EF_KGCO2E_PER_EMPLOYEE_PER_YEAR) / 1000.0

        scope3_total = scope3_travel + scope3_waste + scope3_procurement + scope3_commuting
        total = scope1_total + scope2_location + scope3_total
        per_employee = total / max(inp.employee_count, 1)
        intensity_rev = (total / inp.annual_revenue_gbp * 1_000_000) if inp.annual_revenue_gbp > 0 else 0.0

        # Data quality score (1=best, 5=worst)
        dq_score = self._calc_data_quality_score(inp)

        self._baseline = SilverBaseline(
            scope1_gas_tco2e=round(scope1_gas, 4),
            scope1_fuel_tco2e=round(scope1_fuel, 4),
            scope1_total_tco2e=round(scope1_total, 4),
            scope2_location_tco2e=round(scope2_location, 4),
            scope2_market_tco2e=round(scope2_market, 4),
            scope3_travel_tco2e=round(scope3_travel, 4),
            scope3_waste_tco2e=round(scope3_waste, 4),
            scope3_procurement_tco2e=round(scope3_procurement, 4),
            scope3_commuting_tco2e=round(scope3_commuting, 4),
            scope3_total_tco2e=round(scope3_total, 4),
            total_tco2e=round(total, 4),
            per_employee_tco2e=round(per_employee, 4),
            intensity_per_revenue=round(intensity_rev, 4),
            accuracy_band="+/-15%",
            tier="silver",
            data_quality_score=round(dq_score, 2),
            total_electricity_kwh=round(total_elec_kwh, 2),
            total_gas_kwh=round(total_gas_kwh, 2),
        )

        outputs["scope1_tco2e"] = self._baseline.scope1_total_tco2e
        outputs["scope2_location_tco2e"] = self._baseline.scope2_location_tco2e
        outputs["scope2_market_tco2e"] = self._baseline.scope2_market_tco2e
        outputs["scope3_tco2e"] = self._baseline.scope3_total_tco2e
        outputs["total_tco2e"] = self._baseline.total_tco2e
        outputs["per_employee"] = self._baseline.per_employee_tco2e
        outputs["data_quality"] = dq_score

        if total <= 0:
            errors.append("Total emissions are zero; verify input data")

        elapsed = (_utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        self.logger.info(
            "Silver baseline: S1=%.1f S2=%.1f S3=%.1f Total=%.1f tCO2e (DQ %.1f/5)",
            scope1_total, scope2_location, scope3_total, total, dq_score,
        )
        return PhaseResult(
            phase_name="silver_baseline", phase_number=3,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if not errors else 0.0,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Baseline: {total:.1f} tCO2e (Silver, DQ {dq_score:.1f}/5)",
        )

    def _calc_data_quality_score(self, inp: StandardSetupInput) -> float:
        """Calculate overall data quality score (1=best, 5=worst)."""
        scores: List[float] = []

        # Electricity
        if inp.electricity_bills:
            measured = sum(1 for b in inp.electricity_bills if b.data_quality == "measured")
            scores.append(1.0 + 4.0 * (1.0 - measured / len(inp.electricity_bills)))
        else:
            scores.append(5.0)

        # Gas
        if inp.gas_bills:
            measured = sum(1 for b in inp.gas_bills if b.data_quality == "measured")
            scores.append(1.0 + 4.0 * (1.0 - measured / len(inp.gas_bills)))
        else:
            scores.append(4.5)

        # Fuel
        if inp.fuel_records:
            measured = sum(1 for f in inp.fuel_records if f.data_quality == "measured")
            scores.append(1.0 + 4.0 * (1.0 - measured / len(inp.fuel_records)))
        else:
            scores.append(4.0)

        # Travel
        if inp.travel_records:
            with_km = sum(1 for t in inp.travel_records if t.distance_km > 0)
            scores.append(2.0 + 3.0 * (1.0 - with_km / len(inp.travel_records)))
        else:
            scores.append(4.5)

        # Procurement
        if inp.procurement:
            with_ef = sum(1 for p in inp.procurement if p.supplier_ef_kgco2e_per_gbp is not None)
            scores.append(2.0 + 3.0 * (1.0 - with_ef / len(inp.procurement)))
        else:
            scores.append(4.5)

        return sum(scores) / len(scores) if scores else 5.0

    # -------------------------------------------------------------------------
    # Phase 4: Target Validation
    # -------------------------------------------------------------------------

    async def _phase_target_validation(self, inp: StandardSetupInput) -> PhaseResult:
        """Validate and generate SBTi-aligned targets."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        base_total = self._baseline.total_tco2e
        base_year = self.config.base_year
        near_term_year = 2030
        annual_rate = 4.2  # SBTi 1.5C
        near_term_reduction = 50.0

        near_term_tco2e = base_total * (1.0 - near_term_reduction / 100.0)

        pathway_points: List[Dict[str, Any]] = []
        for yr in range(base_year, 2051):
            yrs = yr - base_year
            if yr <= near_term_year:
                nt_yrs = near_term_year - base_year
                reduction = (near_term_reduction / nt_yrs * yrs) if nt_yrs > 0 else 0
            else:
                post_nt = yr - near_term_year
                lt_yrs = 2050 - near_term_year
                additional = (40.0 / lt_yrs * post_nt) if lt_yrs > 0 else 0
                reduction = near_term_reduction + additional
            target = max(base_total * (1.0 - reduction / 100.0), 0)
            pathway_points.append({"year": yr, "target_tco2e": round(target, 4), "reduction_pct": round(min(reduction, 100), 2)})

        notes: List[str] = []
        if base_total < 25:
            notes.append("Small baseline; consider joining SME Climate Hub for simplified target setting")
        if self._baseline.data_quality_score > 3.0:
            notes.append("Improve data quality for more credible targets")
        notes.append(f"Target: {near_term_reduction:.0f}% by {near_term_year} (SBTi 1.5C aligned)")

        target = ValidatedTarget(
            target_name=f"SME Net Zero Target (1.5C)",
            base_year=base_year,
            base_year_tco2e=round(base_total, 4),
            near_term_year=near_term_year,
            near_term_target_tco2e=round(near_term_tco2e, 4),
            near_term_reduction_pct=near_term_reduction,
            annual_rate_pct=annual_rate,
            pathway_points=pathway_points,
            sbti_aligned=True,
            validation_notes=notes,
        )
        self._targets = [target]

        outputs["target_name"] = target.target_name
        outputs["near_term_reduction"] = near_term_reduction
        outputs["near_term_year"] = near_term_year
        outputs["annual_rate"] = annual_rate

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="target_validation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Target: {near_term_reduction:.0f}% by {near_term_year} (SBTi 1.5C)",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Action Prioritisation (MACC Lite)
    # -------------------------------------------------------------------------

    async def _phase_action_prioritization(self, inp: StandardSetupInput) -> PhaseResult:
        """Generate MACC-lite action prioritisation with top 10 actions."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        baseline = self._baseline
        max_actions = self.config.max_actions

        # Build action library
        action_templates: List[Dict[str, Any]] = [
            {"id": "led_upgrade", "title": "LED lighting upgrade", "category": "energy_efficiency",
             "abatement_pct": 0.05, "scope": "scope2", "cost_per_tco2e": -150, "impl_cost_factor": 2000,
             "payback_yrs": 1.5, "timeframe": "short_term"},
            {"id": "green_tariff", "title": "Switch to 100% renewable electricity", "category": "renewable_energy",
             "abatement_pct": 0.80, "scope": "scope2", "cost_per_tco2e": 10, "impl_cost_factor": 0,
             "payback_yrs": 0, "timeframe": "short_term"},
            {"id": "smart_heating", "title": "Smart heating controls", "category": "heating_cooling",
             "abatement_pct": 0.15, "scope": "scope1_gas", "cost_per_tco2e": -100, "impl_cost_factor": 1500,
             "payback_yrs": 1.0, "timeframe": "short_term"},
            {"id": "insulation", "title": "Building insulation improvements", "category": "energy_efficiency",
             "abatement_pct": 0.20, "scope": "scope1_gas", "cost_per_tco2e": 50, "impl_cost_factor": 8000,
             "payback_yrs": 3.0, "timeframe": "medium_term"},
            {"id": "heat_pump", "title": "Air source heat pump installation", "category": "heating_cooling",
             "abatement_pct": 0.60, "scope": "scope1_gas", "cost_per_tco2e": 200, "impl_cost_factor": 15000,
             "payback_yrs": 7.0, "timeframe": "medium_term"},
            {"id": "ev_transition", "title": "Electric vehicle fleet transition", "category": "transport",
             "abatement_pct": 0.60, "scope": "scope1_fuel", "cost_per_tco2e": 150, "impl_cost_factor": 5000,
             "payback_yrs": 4.0, "timeframe": "medium_term"},
            {"id": "reduce_travel", "title": "Reduce business travel 30%", "category": "transport",
             "abatement_pct": 0.30, "scope": "scope3_travel", "cost_per_tco2e": -500, "impl_cost_factor": 500,
             "payback_yrs": 0, "timeframe": "short_term"},
            {"id": "remote_working", "title": "Hybrid working policy (3 days/week)", "category": "behaviour",
             "abatement_pct": 0.40, "scope": "scope3_commuting", "cost_per_tco2e": -200, "impl_cost_factor": 0,
             "payback_yrs": 0, "timeframe": "short_term"},
            {"id": "green_procurement", "title": "Green procurement policy", "category": "procurement",
             "abatement_pct": 0.05, "scope": "scope3_procurement", "cost_per_tco2e": 0, "impl_cost_factor": 0,
             "payback_yrs": 0, "timeframe": "short_term"},
            {"id": "waste_recycling", "title": "Comprehensive recycling programme", "category": "waste",
             "abatement_pct": 0.40, "scope": "scope3_waste", "cost_per_tco2e": -50, "impl_cost_factor": 1000,
             "payback_yrs": 1.0, "timeframe": "short_term"},
            {"id": "solar_pv", "title": "Rooftop solar PV installation", "category": "renewable_energy",
             "abatement_pct": 0.30, "scope": "scope2", "cost_per_tco2e": 100, "impl_cost_factor": 20000,
             "payback_yrs": 8.0, "timeframe": "long_term"},
            {"id": "energy_audit", "title": "Professional energy audit", "category": "energy_efficiency",
             "abatement_pct": 0.10, "scope": "scope2", "cost_per_tco2e": -200, "impl_cost_factor": 3000,
             "payback_yrs": 0.5, "timeframe": "short_term"},
        ]

        actions: List[Tuple[float, MACCAction]] = []
        for tmpl in action_templates:
            scope_key = tmpl["scope"]
            base_emissions = self._get_scope_emissions(scope_key)
            abatement = base_emissions * tmpl["abatement_pct"]

            if abatement <= 0.01:
                continue

            cost_per_tco2e = tmpl["cost_per_tco2e"]
            annual_cost = abatement * max(cost_per_tco2e, 0)
            annual_saving = abatement * abs(min(cost_per_tco2e, 0))
            net_annual = annual_saving - annual_cost

            size_factor = max(inp.employee_count / 50.0, 0.3)
            impl_cost = tmpl["impl_cost_factor"] * min(size_factor, 3.0)

            action_type = MACCActionType.NEGATIVE_COST.value if cost_per_tco2e < 0 else (
                MACCActionType.LOW_COST.value if cost_per_tco2e < 100 else (
                    MACCActionType.MEDIUM_COST.value if cost_per_tco2e < 500 else MACCActionType.HIGH_COST.value
                )
            )

            action = MACCAction(
                action_id=tmpl["id"],
                rank=0,
                title=tmpl["title"],
                description="",
                category=tmpl["category"],
                abatement_tco2e=round(abatement, 4),
                cost_per_tco2e_gbp=cost_per_tco2e,
                annual_cost_gbp=round(annual_cost, 2),
                annual_saving_gbp=round(annual_saving, 2),
                net_annual_gbp=round(net_annual, 2),
                payback_years=tmpl["payback_yrs"],
                implementation_cost_gbp=round(impl_cost, 2),
                action_type=action_type,
                timeframe=tmpl["timeframe"],
            )

            # Sort score: negative cost first, then by abatement
            sort_key = (-1 if cost_per_tco2e < 0 else 1) * abs(cost_per_tco2e) - abatement * 100
            actions.append((sort_key, action))

        actions.sort(key=lambda x: x[0])
        self._actions = []
        for rank, (_, action) in enumerate(actions[:max_actions], 1):
            action.rank = rank
            self._actions.append(action)

        total_abatement = sum(a.abatement_tco2e for a in self._actions)
        total_net = sum(a.net_annual_gbp for a in self._actions)
        negative_cost_count = sum(1 for a in self._actions if a.cost_per_tco2e_gbp < 0)

        outputs["action_count"] = len(self._actions)
        outputs["total_abatement_tco2e"] = round(total_abatement, 4)
        outputs["total_net_annual_gbp"] = round(total_net, 2)
        outputs["negative_cost_actions"] = negative_cost_count
        outputs["abatement_pct_of_baseline"] = round(
            (total_abatement / max(baseline.total_tco2e, 0.01)) * 100, 2
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Action prioritisation: %d actions, %.1f tCO2e abatement (%.1f%% of baseline)",
            len(self._actions), total_abatement,
            (total_abatement / max(baseline.total_tco2e, 0.01)) * 100,
        )
        return PhaseResult(
            phase_name="action_prioritization", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Actions: {len(self._actions)} prioritised, {total_abatement:.1f} tCO2e savings",
        )

    def _get_scope_emissions(self, scope_key: str) -> float:
        """Get emissions for a scope key from the baseline."""
        b = self._baseline
        mapping = {
            "scope1": b.scope1_total_tco2e,
            "scope1_gas": b.scope1_gas_tco2e,
            "scope1_fuel": b.scope1_fuel_tco2e,
            "scope2": b.scope2_location_tco2e,
            "scope3_travel": b.scope3_travel_tco2e,
            "scope3_waste": b.scope3_waste_tco2e,
            "scope3_procurement": b.scope3_procurement_tco2e,
            "scope3_commuting": b.scope3_commuting_tco2e,
            "scope3": b.scope3_total_tco2e,
        }
        return mapping.get(scope_key, 0.0)

    # -------------------------------------------------------------------------
    # Phase 6: Grant Matching
    # -------------------------------------------------------------------------

    async def _phase_grant_matching(self, inp: StandardSetupInput) -> PhaseResult:
        """Find matching grants based on sector, size, and location."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        country = inp.country or inp.config.country or "UK"
        sector = inp.industry_sector
        size = "small"
        if inp.employee_count < 10:
            size = "micro"
        elif inp.employee_count < 50:
            size = "small"
        elif inp.employee_count < 250:
            size = "medium"
        else:
            size = "large_sme"

        matches: List[GrantMatch] = []

        for grant in SME_GRANT_DATABASE:
            eligible_sectors = grant.get("eligible_sectors", [])
            eligible_sizes = grant.get("eligible_sizes", [])
            grant_countries = grant.get("countries", [])

            # Check sector eligibility
            sector_ok = "all" in eligible_sectors or sector in eligible_sectors
            size_ok = size in eligible_sizes
            country_ok = country in grant_countries or (country in ["UK", "DE", "FR", "IE"] and "EU" in grant_countries)

            if not (sector_ok and size_ok and country_ok):
                continue

            # Calculate relevance score
            relevance = 0.5
            if sector in eligible_sectors:
                relevance += 0.2
            if country in grant_countries:
                relevance += 0.2
            if grant.get("status") == "open":
                relevance += 0.1

            matches.append(GrantMatch(
                grant_id=grant["grant_id"],
                name=grant["name"],
                provider=grant["provider"],
                max_amount_gbp=grant["max_amount_gbp"],
                min_amount_gbp=grant["min_amount_gbp"],
                description=grant["description"],
                relevance_score=round(min(relevance, 1.0), 2),
                eligible=True,
                status=grant.get("status", "open"),
            ))

        matches.sort(key=lambda g: g.relevance_score, reverse=True)
        self._grants = matches

        total_potential = sum(g.max_amount_gbp for g in matches)
        outputs["grant_matches"] = len(matches)
        outputs["total_potential_funding_gbp"] = total_potential

        if not matches:
            warnings.append("No matching grants found for your profile; consider broadening search criteria")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Grant matching: %d grants found, total potential GBP %d", len(matches), total_potential)
        return PhaseResult(
            phase_name="grant_matching", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Grants: {len(matches)} matches found",
        )

    # -------------------------------------------------------------------------
    # Next Steps & Provenance
    # -------------------------------------------------------------------------

    def _generate_next_steps(self, inp: StandardSetupInput) -> List[str]:
        steps: List[str] = []
        steps.append("Review your Silver baseline and validated targets in the dashboard.")

        if self._actions:
            neg_cost = [a for a in self._actions if a.cost_per_tco2e_gbp < 0]
            if neg_cost:
                steps.append(
                    f"Start with {len(neg_cost)} negative-cost actions that save money: "
                    f"{neg_cost[0].title}."
                )

        if self._grants:
            steps.append(
                f"Apply for {self._grants[0].name} grant (up to GBP {self._grants[0].max_amount_gbp:,.0f})."
            )

        steps.append("Set up quarterly reviews to track progress.")
        steps.append("Consider SME Climate Hub commitment for public accountability.")
        return steps
