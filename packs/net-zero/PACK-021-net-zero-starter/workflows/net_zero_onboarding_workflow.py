# -*- coding: utf-8 -*-
"""
Net-Zero Onboarding Workflow
================================

4-phase workflow for establishing a GHG emissions baseline as part of
PACK-021 Net-Zero Starter Pack.  The workflow collects organisation-level
energy, fuel, fleet, and procurement data, computes a Scope 1+2+3
baseline, evaluates data quality against GHG Protocol guidance, and
generates a baseline summary report with improvement recommendations.

Phases:
    1. DataCollection   -- Validate and normalise input data
    2. BaselineCalc     -- Calculate Scope 1+2+3 emissions
    3. DataQuality      -- Score data quality per GHG Protocol (1-5)
    4. BaselineReport   -- Generate baseline summary and recommendations

Zero-hallucination: every numeric result is derived from deterministic
emission factors and validated formulae.  SHA-256 provenance hashes
guarantee end-to-end auditability.

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"

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

class BoundaryMethod(str, Enum):
    """GHG Protocol organisational boundary approaches."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class DataQualityLevel(str, Enum):
    """GHG Protocol data quality levels (1=highest, 5=lowest)."""

    LEVEL_1 = "1"
    LEVEL_2 = "2"
    LEVEL_3 = "3"
    LEVEL_4 = "4"
    LEVEL_5 = "5"

class FuelType(str, Enum):
    """Stationary and mobile combustion fuel types."""

    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    PETROL = "petrol"
    LPG = "lpg"
    HEATING_OIL = "heating_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    PROPANE = "propane"

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories."""

    CAT1_PURCHASED_GOODS = "cat1_purchased_goods"
    CAT2_CAPITAL_GOODS = "cat2_capital_goods"
    CAT3_FUEL_ENERGY = "cat3_fuel_energy"
    CAT4_UPSTREAM_TRANSPORT = "cat4_upstream_transport"
    CAT5_WASTE = "cat5_waste"
    CAT6_BUSINESS_TRAVEL = "cat6_business_travel"
    CAT7_COMMUTING = "cat7_commuting"
    CAT8_UPSTREAM_LEASED = "cat8_upstream_leased"
    CAT9_DOWNSTREAM_TRANSPORT = "cat9_downstream_transport"
    CAT10_PROCESSING = "cat10_processing"
    CAT11_USE_OF_SOLD = "cat11_use_of_sold"
    CAT12_END_OF_LIFE = "cat12_end_of_life"
    CAT13_DOWNSTREAM_LEASED = "cat13_downstream_leased"
    CAT14_FRANCHISES = "cat14_franchises"
    CAT15_INVESTMENTS = "cat15_investments"

# =============================================================================
# EMISSION FACTOR CONSTANTS (Zero-Hallucination, DEFRA / IEA 2024)
# =============================================================================

FUEL_EF_KGCO2E_PER_KWH: Dict[str, float] = {
    "natural_gas": 0.18293,
    "diesel": 0.25301,
    "petrol": 0.23141,
    "lpg": 0.21448,
    "heating_oil": 0.24685,
    "coal": 0.32170,
    "biomass": 0.01538,
    "propane": 0.21448,
}

FUEL_LITRE_TO_KWH: Dict[str, float] = {
    "natural_gas": 10.55,
    "diesel": 10.00,
    "petrol": 9.06,
    "lpg": 7.08,
    "heating_oil": 10.35,
    "propane": 7.08,
}

VEHICLE_FUEL_EF_KGCO2E_PER_LITRE: Dict[str, float] = {
    "diesel": 2.70494,
    "petrol": 2.31440,
    "cng": 2.53990,
    "lpg": 1.65210,
    "electric": 0.0,
    "hybrid_petrol": 1.73160,
    "hybrid_diesel": 2.02870,
}

GRID_LOCATION_EF_KGCO2E_PER_KWH: Dict[str, float] = {
    "GLOBAL": 0.4940,
    "EU-AVG": 0.2556,
    "US-AVG": 0.3710,
    "CN": 0.5550,
    "IN": 0.7080,
    "DE": 0.3850,
    "FR": 0.0520,
    "UK": 0.2070,
    "JP": 0.4350,
    "AU": 0.6560,
}

GRID_MARKET_EF_KGCO2E_PER_KWH: Dict[str, float] = {
    "GLOBAL": 0.5500,
    "EU-AVG": 0.4200,
    "US-AVG": 0.4200,
    "DE": 0.5810,
    "FR": 0.0560,
    "UK": 0.3420,
}

# Spend-based Scope 3 EFs (kgCO2e per USD spent) - EEIO averages
SCOPE3_SPEND_EF_KGCO2E_PER_USD: Dict[str, float] = {
    "cat1_purchased_goods": 0.42,
    "cat2_capital_goods": 0.35,
    "cat4_upstream_transport": 0.72,
    "cat5_waste": 0.58,
    "cat6_business_travel": 0.26,
    "cat7_commuting": 0.18,
}

# WTT uplift factor for Scope 3 Category 3 (Fuel-and-energy-related)
WTT_UPLIFT_FACTOR = 0.20

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class FacilityRecord(BaseModel):
    """A single operational facility / site."""

    facility_id: str = Field(..., description="Unique facility identifier")
    facility_name: str = Field(default="", description="Human-readable name")
    country: str = Field(default="GLOBAL", description="ISO 3166-1 alpha-2 or keyword")
    region: str = Field(default="", description="Sub-national region")
    floor_area_sqm: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    facility_type: str = Field(default="office", description="office|factory|warehouse|retail|other")

class EnergyDataRecord(BaseModel):
    """Energy consumption record (stationary combustion / purchased electricity)."""

    facility_id: str = Field(..., description="Owning facility")
    fuel_type: str = Field(default="natural_gas", description="Fuel or energy type")
    consumption_kwh: float = Field(default=0.0, ge=0.0)
    consumption_litres: float = Field(default=0.0, ge=0.0)
    is_electricity: bool = Field(default=False, description="True if purchased electricity")
    grid_region: str = Field(default="GLOBAL", description="Grid region for EF lookup")
    renewable_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    has_guarantee_of_origin: bool = Field(default=False)
    supplier_ef_kgco2_kwh: Optional[float] = Field(None, ge=0.0)
    data_quality: str = Field(default="measured", description="measured|estimated|default")
    period_start: str = Field(default="", description="YYYY-MM-DD")
    period_end: str = Field(default="", description="YYYY-MM-DD")

class FuelRecord(BaseModel):
    """Fuel combustion record for Scope 1 stationary sources."""

    facility_id: str = Field(..., description="Owning facility")
    fuel_type: str = Field(default="natural_gas")
    consumption_kwh: float = Field(default=0.0, ge=0.0)
    consumption_litres: float = Field(default=0.0, ge=0.0)
    consumption_tonnes: float = Field(default=0.0, ge=0.0)
    data_quality: str = Field(default="measured")

class FleetRecord(BaseModel):
    """Fleet / mobile combustion record."""

    facility_id: str = Field(default="", description="Associated facility")
    vehicle_id: str = Field(default="")
    vehicle_type: str = Field(default="car", description="car|van|truck|other")
    fuel_type: str = Field(default="diesel")
    fuel_consumed_litres: float = Field(default=0.0, ge=0.0)
    distance_km: float = Field(default=0.0, ge=0.0)
    data_quality: str = Field(default="measured")

class ProcurementRecord(BaseModel):
    """Procurement spend record for Scope 3 spend-based estimation."""

    category: str = Field(default="cat1_purchased_goods", description="Scope 3 category key")
    description: str = Field(default="", description="Description of spend category")
    spend_usd: float = Field(default=0.0, ge=0.0)
    supplier_specific_ef: Optional[float] = Field(None, ge=0.0, description="kgCO2e per USD if known")
    data_quality: str = Field(default="estimated")

class OnboardingConfig(BaseModel):
    """Configuration for the net-zero onboarding workflow."""

    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2024, ge=2015, le=2050)
    boundary_method: str = Field(default="operational_control")
    include_scope3: bool = Field(default=True)
    scope3_categories_to_include: List[str] = Field(
        default_factory=lambda: [
            "cat1_purchased_goods",
            "cat3_fuel_energy",
            "cat5_waste",
            "cat6_business_travel",
            "cat7_commuting",
        ]
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("boundary_method")
    @classmethod
    def _validate_boundary(cls, v: str) -> str:
        allowed = {"operational_control", "financial_control", "equity_share"}
        if v not in allowed:
            raise ValueError(f"boundary_method must be one of {allowed}")
        return v

class OnboardingInput(BaseModel):
    """Input data for the net-zero onboarding workflow."""

    facilities: List[FacilityRecord] = Field(default_factory=list, description="Organisation facilities")
    energy_data: List[EnergyDataRecord] = Field(default_factory=list, description="Energy consumption records")
    fuel_data: List[FuelRecord] = Field(default_factory=list, description="Fuel combustion records")
    fleet_data: List[FleetRecord] = Field(default_factory=list, description="Fleet / mobile records")
    procurement_data: List[ProcurementRecord] = Field(default_factory=list, description="Procurement spend records")
    config: OnboardingConfig = Field(default_factory=OnboardingConfig)

class ScopeBreakdown(BaseModel):
    """Emissions breakdown by scope."""

    scope1_stationary_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_mobile_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_by_category: Dict[str, float] = Field(default_factory=dict)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)

class DataQualityItem(BaseModel):
    """Data quality assessment for a single data source."""

    source: str = Field(default="", description="Data source label")
    scope: str = Field(default="", description="scope1|scope2|scope3")
    quality_score: int = Field(default=5, ge=1, le=5, description="GHG Protocol 1-5")
    record_count: int = Field(default=0, ge=0)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    issues: List[str] = Field(default_factory=list)

class DataQualityReport(BaseModel):
    """Aggregate data quality report."""

    overall_score: float = Field(default=5.0, ge=1.0, le=5.0)
    items: List[DataQualityItem] = Field(default_factory=list)
    missing_scope3_categories: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)

class BaselineReportSummary(BaseModel):
    """Baseline summary report output."""

    organisation_name: str = Field(default="")
    base_year: int = Field(default=2024)
    reporting_year: int = Field(default=2025)
    boundary_method: str = Field(default="operational_control")
    facility_count: int = Field(default=0)
    emissions_breakdown: ScopeBreakdown = Field(default_factory=ScopeBreakdown)
    data_quality_matrix: DataQualityReport = Field(default_factory=DataQualityReport)
    top_emission_sources: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class OnboardingResult(BaseModel):
    """Complete result from the net-zero onboarding workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="net_zero_onboarding")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    baseline: ScopeBreakdown = Field(default_factory=ScopeBreakdown)
    data_quality_report: DataQualityReport = Field(default_factory=DataQualityReport)
    baseline_report: BaselineReportSummary = Field(default_factory=BaselineReportSummary)
    recommendations: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class NetZeroOnboardingWorkflow:
    """
    4-phase onboarding workflow establishing a GHG emissions baseline.

    Collects energy, fuel, fleet, and procurement data; calculates
    Scope 1+2+3 emissions; scores data quality per GHG Protocol; and
    generates a baseline summary report.

    Zero-hallucination: all calculations use deterministic emission
    factors.  No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Workflow configuration.

    Example:
        >>> wf = NetZeroOnboardingWorkflow()
        >>> inp = OnboardingInput(facilities=[...], energy_data=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[OnboardingConfig] = None) -> None:
        """Initialise NetZeroOnboardingWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config = config or OnboardingConfig()
        self._phase_results: List[PhaseResult] = []
        self._baseline: ScopeBreakdown = ScopeBreakdown()
        self._dq_report: DataQualityReport = DataQualityReport()
        self._report: BaselineReportSummary = BaselineReportSummary()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: OnboardingInput) -> OnboardingResult:
        """
        Execute the 4-phase net-zero onboarding workflow.

        Args:
            input_data: Validated onboarding input with facility, energy,
                fuel, fleet, and procurement data.

        Returns:
            OnboardingResult with baseline, data quality, and recommendations.

        Raises:
            ValueError: If critical input data is missing.
        """
        started_at = utcnow()
        cfg = input_data.config
        self.config = cfg
        self.logger.info(
            "Starting net-zero onboarding workflow %s, year=%d, base_year=%d",
            self.workflow_id, cfg.reporting_year, cfg.base_year,
        )

        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_collection(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"DataCollection failed: {phase1.errors}")

            phase2 = await self._phase_baseline_calc(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_data_quality(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_baseline_report(input_data)
            self._phase_results.append(phase4)

            failed_phases = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed_phases else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Onboarding workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = OnboardingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            baseline=self._baseline,
            data_quality_report=self._dq_report,
            baseline_report=self._report,
            recommendations=self._report.recommendations,
            reporting_year=cfg.reporting_year,
        )
        result.provenance_hash = self._provenance_of_result(result)
        self.logger.info(
            "Onboarding workflow %s completed in %.2fs status=%s",
            self.workflow_id, elapsed, overall_status.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(self, input_data: OnboardingInput) -> PhaseResult:
        """Validate and normalise input data from all sources."""
        started = utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        # Validate facilities
        facility_ids = {f.facility_id for f in input_data.facilities}
        outputs["facility_count"] = len(facility_ids)
        if not facility_ids:
            warnings.append("No facilities provided; results will be organisation-level only")

        # Validate energy data
        orphan_energy = [e for e in input_data.energy_data if e.facility_id not in facility_ids and facility_ids]
        if orphan_energy:
            warnings.append(f"{len(orphan_energy)} energy records reference unknown facilities")
        outputs["energy_record_count"] = len(input_data.energy_data)
        outputs["electricity_records"] = sum(1 for e in input_data.energy_data if e.is_electricity)
        outputs["fuel_records_from_energy"] = outputs["energy_record_count"] - outputs["electricity_records"]

        # Validate fuel data
        outputs["fuel_record_count"] = len(input_data.fuel_data)
        for fr in input_data.fuel_data:
            if fr.consumption_kwh <= 0 and fr.consumption_litres <= 0 and fr.consumption_tonnes <= 0:
                warnings.append(f"Fuel record for {fr.facility_id}/{fr.fuel_type}: all consumption fields are zero")

        # Validate fleet data
        outputs["fleet_record_count"] = len(input_data.fleet_data)
        for fl in input_data.fleet_data:
            if fl.fuel_consumed_litres <= 0 and fl.distance_km <= 0:
                warnings.append(f"Fleet record {fl.vehicle_id}: no fuel or distance data")

        # Validate procurement data
        outputs["procurement_record_count"] = len(input_data.procurement_data)
        total_spend = sum(p.spend_usd for p in input_data.procurement_data)
        outputs["total_procurement_spend_usd"] = round(total_spend, 2)
        if self.config.include_scope3 and not input_data.procurement_data:
            warnings.append("Scope 3 enabled but no procurement data provided")

        # Normalise units (convert litres to kWh where needed)
        normalised_count = self._normalise_energy_units(input_data)
        outputs["records_normalised"] = normalised_count

        elapsed = (utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        return PhaseResult(
            phase_name="data_collection",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _normalise_energy_units(self, input_data: OnboardingInput) -> int:
        """Convert litre-based records to kWh equivalents.  Returns count of normalised records."""
        count = 0
        for rec in input_data.energy_data:
            if rec.consumption_kwh <= 0 and rec.consumption_litres > 0 and not rec.is_electricity:
                conversion = FUEL_LITRE_TO_KWH.get(rec.fuel_type, 10.0)
                rec.consumption_kwh = rec.consumption_litres * conversion
                count += 1
        for rec in input_data.fuel_data:
            if rec.consumption_kwh <= 0 and rec.consumption_litres > 0:
                conversion = FUEL_LITRE_TO_KWH.get(rec.fuel_type, 10.0)
                rec.consumption_kwh = rec.consumption_litres * conversion
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Phase 2: Baseline Calculation
    # -------------------------------------------------------------------------

    async def _phase_baseline_calc(self, input_data: OnboardingInput) -> PhaseResult:
        """Calculate Scope 1+2+3 emissions baseline."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Scope 1 stationary combustion
        scope1_stationary = self._calc_scope1_stationary(input_data)

        # Scope 1 mobile combustion (fleet)
        scope1_mobile = self._calc_scope1_mobile(input_data)

        scope1_total = scope1_stationary + scope1_mobile

        # Scope 2 electricity (location-based and market-based)
        scope2_location = self._calc_scope2_location(input_data)
        scope2_market = self._calc_scope2_market(input_data)

        # Scope 3 (spend-based + category 3 WTT)
        scope3_by_cat: Dict[str, float] = {}
        scope3_total = 0.0
        if self.config.include_scope3:
            scope3_by_cat = self._calc_scope3(input_data, scope1_total, scope2_location)
            scope3_total = sum(scope3_by_cat.values())

        total = scope1_total + scope2_location + scope3_total

        self._baseline = ScopeBreakdown(
            scope1_stationary_tco2e=round(scope1_stationary, 4),
            scope1_mobile_tco2e=round(scope1_mobile, 4),
            scope1_total_tco2e=round(scope1_total, 4),
            scope2_location_tco2e=round(scope2_location, 4),
            scope2_market_tco2e=round(scope2_market, 4),
            scope3_by_category={k: round(v, 4) for k, v in scope3_by_cat.items()},
            scope3_total_tco2e=round(scope3_total, 4),
            total_tco2e=round(total, 4),
        )

        outputs["scope1_stationary_tco2e"] = self._baseline.scope1_stationary_tco2e
        outputs["scope1_mobile_tco2e"] = self._baseline.scope1_mobile_tco2e
        outputs["scope1_total_tco2e"] = self._baseline.scope1_total_tco2e
        outputs["scope2_location_tco2e"] = self._baseline.scope2_location_tco2e
        outputs["scope2_market_tco2e"] = self._baseline.scope2_market_tco2e
        outputs["scope3_total_tco2e"] = self._baseline.scope3_total_tco2e
        outputs["total_tco2e"] = self._baseline.total_tco2e

        if total <= 0:
            warnings.append("Total emissions are zero; check that input data has non-zero consumption")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Baseline calc: S1=%.2f S2=%.2f S3=%.2f Total=%.2f tCO2e",
            scope1_total, scope2_location, scope3_total, total,
        )
        return PhaseResult(
            phase_name="baseline_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calc_scope1_stationary(self, input_data: OnboardingInput) -> float:
        """Calculate Scope 1 stationary combustion emissions (tCO2e)."""
        total = 0.0
        # Energy records (non-electricity)
        for rec in input_data.energy_data:
            if rec.is_electricity:
                continue
            ef = FUEL_EF_KGCO2E_PER_KWH.get(rec.fuel_type, 0.20)
            kwh = rec.consumption_kwh
            total += (kwh * ef) / 1000.0

        # Dedicated fuel records
        for rec in input_data.fuel_data:
            ef = FUEL_EF_KGCO2E_PER_KWH.get(rec.fuel_type, 0.20)
            kwh = rec.consumption_kwh
            if kwh <= 0 and rec.consumption_tonnes > 0:
                # Approximate: 1 tonne natural gas ~ 13,900 kWh
                kwh = rec.consumption_tonnes * 13900.0
            total += (kwh * ef) / 1000.0
        return total

    def _calc_scope1_mobile(self, input_data: OnboardingInput) -> float:
        """Calculate Scope 1 mobile combustion emissions (tCO2e)."""
        total = 0.0
        for rec in input_data.fleet_data:
            ef = VEHICLE_FUEL_EF_KGCO2E_PER_LITRE.get(rec.fuel_type, 2.70)
            litres = rec.fuel_consumed_litres
            if litres <= 0 and rec.distance_km > 0:
                # Default fuel efficiency: 8 L/100km for diesel, 9 L/100km for petrol
                eff = 8.0 if rec.fuel_type == "diesel" else 9.0
                litres = rec.distance_km * (eff / 100.0)
            total += (litres * ef) / 1000.0
        return total

    def _calc_scope2_location(self, input_data: OnboardingInput) -> float:
        """Calculate Scope 2 location-based electricity emissions (tCO2e)."""
        total = 0.0
        for rec in input_data.energy_data:
            if not rec.is_electricity:
                continue
            region = rec.grid_region or "GLOBAL"
            ef = GRID_LOCATION_EF_KGCO2E_PER_KWH.get(region, GRID_LOCATION_EF_KGCO2E_PER_KWH["GLOBAL"])
            total += (rec.consumption_kwh * ef) / 1000.0
        return total

    def _calc_scope2_market(self, input_data: OnboardingInput) -> float:
        """Calculate Scope 2 market-based electricity emissions (tCO2e)."""
        total = 0.0
        for rec in input_data.energy_data:
            if not rec.is_electricity:
                continue
            if rec.has_guarantee_of_origin:
                continue
            if rec.supplier_ef_kgco2_kwh is not None:
                ef = rec.supplier_ef_kgco2_kwh
            else:
                region = rec.grid_region or "GLOBAL"
                ef = GRID_MARKET_EF_KGCO2E_PER_KWH.get(region, GRID_MARKET_EF_KGCO2E_PER_KWH.get("GLOBAL", 0.55))
            total += (rec.consumption_kwh * ef) / 1000.0
        return total

    def _calc_scope3(
        self,
        input_data: OnboardingInput,
        scope1_total: float,
        scope2_location: float,
    ) -> Dict[str, float]:
        """Calculate Scope 3 emissions by category (tCO2e)."""
        result: Dict[str, float] = {}
        included = set(self.config.scope3_categories_to_include)

        # Spend-based categories from procurement
        for rec in input_data.procurement_data:
            cat = rec.category
            if cat not in included:
                continue
            ef = rec.supplier_specific_ef if rec.supplier_specific_ef is not None else SCOPE3_SPEND_EF_KGCO2E_PER_USD.get(cat, 0.40)
            emissions_tco2e = (rec.spend_usd * ef) / 1000.0
            result[cat] = result.get(cat, 0.0) + emissions_tco2e

        # Category 3: Fuel-and-energy-related activities (WTT uplift)
        if "cat3_fuel_energy" in included:
            wtt = (scope1_total + scope2_location) * WTT_UPLIFT_FACTOR
            result["cat3_fuel_energy"] = result.get("cat3_fuel_energy", 0.0) + wtt

        return {k: round(v, 4) for k, v in result.items()}

    # -------------------------------------------------------------------------
    # Phase 3: Data Quality
    # -------------------------------------------------------------------------

    async def _phase_data_quality(self, input_data: OnboardingInput) -> PhaseResult:
        """Score data quality per GHG Protocol (1 = best, 5 = worst)."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        items: List[DataQualityItem] = []

        # Scope 1 stationary quality
        s1_stat_recs = [r for r in input_data.energy_data if not r.is_electricity] + [
            EnergyDataRecord(facility_id=f.facility_id, fuel_type=f.fuel_type,
                             consumption_kwh=f.consumption_kwh, data_quality=f.data_quality)
            for f in input_data.fuel_data
        ]
        s1_stat_item = self._score_data_source("scope1_stationary", "scope1", s1_stat_recs)
        items.append(s1_stat_item)

        # Scope 1 mobile quality
        s1_mob_item = self._score_fleet_quality("scope1_mobile", input_data.fleet_data)
        items.append(s1_mob_item)

        # Scope 2 quality
        s2_recs = [r for r in input_data.energy_data if r.is_electricity]
        s2_item = self._score_data_source("scope2_electricity", "scope2", s2_recs)
        items.append(s2_item)

        # Scope 3 quality
        if self.config.include_scope3:
            s3_item = self._score_procurement_quality("scope3_procurement", input_data.procurement_data)
            items.append(s3_item)

        # Missing Scope 3 categories
        all_scope3_cats = {e.value for e in Scope3Category}
        included_cats = set(self.config.scope3_categories_to_include)
        reported_cats = {p.category for p in input_data.procurement_data}
        # Combine what is included but has no data
        missing = []
        for cat in included_cats:
            if cat not in reported_cats and cat != "cat3_fuel_energy":
                missing.append(cat)
        # Categories not even included
        not_included = all_scope3_cats - included_cats
        if not_included:
            warnings.append(f"{len(not_included)} Scope 3 categories excluded from assessment")

        # Overall score (weighted average)
        if items:
            total_recs = sum(i.record_count for i in items) or 1
            weighted = sum(i.quality_score * i.record_count for i in items)
            overall = weighted / total_recs
        else:
            overall = 5.0

        # Gaps
        gaps: List[str] = []
        if not input_data.facilities:
            gaps.append("No facility records provided")
        if not input_data.energy_data:
            gaps.append("No energy consumption data")
        if not input_data.fleet_data:
            gaps.append("No fleet / mobile combustion data")
        if self.config.include_scope3 and not input_data.procurement_data:
            gaps.append("No procurement / Scope 3 data")

        # Improvement recommendations
        recommendations = self._generate_dq_recommendations(items, gaps, missing)

        self._dq_report = DataQualityReport(
            overall_score=round(min(max(overall, 1.0), 5.0), 2),
            items=items,
            missing_scope3_categories=missing,
            gaps=gaps,
            improvement_recommendations=recommendations,
        )

        outputs["overall_quality_score"] = self._dq_report.overall_score
        outputs["item_count"] = len(items)
        outputs["gap_count"] = len(gaps)
        outputs["missing_scope3_count"] = len(missing)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Data quality score: %.2f/5.0", self._dq_report.overall_score)
        return PhaseResult(
            phase_name="data_quality",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _score_data_source(
        self, source_label: str, scope: str, records: List[Any],
    ) -> DataQualityItem:
        """Score a data source based on measurement method and completeness."""
        if not records:
            return DataQualityItem(
                source=source_label, scope=scope, quality_score=5,
                record_count=0, completeness_pct=0.0,
                issues=["No data records provided"],
            )
        measured = sum(1 for r in records if getattr(r, "data_quality", "default") == "measured")
        estimated = sum(1 for r in records if getattr(r, "data_quality", "default") == "estimated")
        total = len(records)
        measured_pct = (measured / total) * 100.0 if total > 0 else 0.0

        if measured_pct >= 90:
            score = 1
        elif measured_pct >= 70:
            score = 2
        elif measured_pct >= 50:
            score = 3
        elif measured_pct >= 20:
            score = 4
        else:
            score = 5

        issues: List[str] = []
        if measured_pct < 50:
            issues.append(f"Only {measured_pct:.0f}% of records are measured (target >=50%)")
        if estimated > 0:
            issues.append(f"{estimated} records use estimated values")

        return DataQualityItem(
            source=source_label, scope=scope, quality_score=score,
            record_count=total, completeness_pct=round(measured_pct, 1),
            issues=issues,
        )

    def _score_fleet_quality(self, source_label: str, fleet_records: List[FleetRecord]) -> DataQualityItem:
        """Score fleet data quality."""
        if not fleet_records:
            return DataQualityItem(
                source=source_label, scope="scope1", quality_score=5,
                record_count=0, completeness_pct=0.0,
                issues=["No fleet records provided"],
            )
        with_fuel = sum(1 for f in fleet_records if f.fuel_consumed_litres > 0)
        with_distance = sum(1 for f in fleet_records if f.distance_km > 0)
        total = len(fleet_records)
        fuel_pct = (with_fuel / total) * 100.0

        if fuel_pct >= 90:
            score = 1
        elif fuel_pct >= 60:
            score = 2
        elif with_distance / total * 100 >= 80:
            score = 3
        elif fuel_pct >= 20 or with_distance > 0:
            score = 4
        else:
            score = 5

        issues: List[str] = []
        if fuel_pct < 80:
            issues.append(f"Only {fuel_pct:.0f}% of fleet records have actual fuel data")

        return DataQualityItem(
            source=source_label, scope="scope1", quality_score=score,
            record_count=total, completeness_pct=round(fuel_pct, 1),
            issues=issues,
        )

    def _score_procurement_quality(
        self, source_label: str, records: List[ProcurementRecord],
    ) -> DataQualityItem:
        """Score procurement / Scope 3 data quality."""
        if not records:
            return DataQualityItem(
                source=source_label, scope="scope3", quality_score=5,
                record_count=0, completeness_pct=0.0,
                issues=["No procurement records provided"],
            )
        with_specific_ef = sum(1 for r in records if r.supplier_specific_ef is not None)
        total = len(records)
        specific_pct = (with_specific_ef / total) * 100.0

        if specific_pct >= 80:
            score = 1
        elif specific_pct >= 50:
            score = 2
        elif specific_pct >= 20:
            score = 3
        else:
            # Spend-based with EEIO factors = quality 4
            score = 4

        issues: List[str] = []
        if specific_pct < 50:
            issues.append("Majority of Scope 3 uses spend-based EEIO factors (consider supplier-specific data)")

        return DataQualityItem(
            source=source_label, scope="scope3", quality_score=score,
            record_count=total, completeness_pct=round(specific_pct, 1),
            issues=issues,
        )

    def _generate_dq_recommendations(
        self,
        items: List[DataQualityItem],
        gaps: List[str],
        missing_cats: List[str],
    ) -> List[str]:
        """Generate data quality improvement recommendations."""
        recs: List[str] = []
        for item in items:
            if item.quality_score >= 4:
                recs.append(
                    f"Improve {item.source} data: current score {item.quality_score}/5. "
                    f"Collect measured data to replace estimates."
                )
        if missing_cats:
            recs.append(
                f"Collect data for {len(missing_cats)} missing Scope 3 categories: "
                f"{', '.join(missing_cats[:5])}"
            )
        for gap in gaps:
            recs.append(f"Address data gap: {gap}")
        if not recs:
            recs.append("Data quality is adequate. Continue monitoring annually.")
        return recs

    # -------------------------------------------------------------------------
    # Phase 4: Baseline Report
    # -------------------------------------------------------------------------

    async def _phase_baseline_report(self, input_data: OnboardingInput) -> PhaseResult:
        """Generate baseline summary report with emissions breakdown."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        top_sources = self._identify_top_sources()
        recommendations = self._generate_baseline_recommendations()

        self._report = BaselineReportSummary(
            base_year=self.config.base_year,
            reporting_year=self.config.reporting_year,
            boundary_method=self.config.boundary_method,
            facility_count=len(input_data.facilities),
            emissions_breakdown=self._baseline,
            data_quality_matrix=self._dq_report,
            top_emission_sources=top_sources,
            recommendations=recommendations,
        )

        outputs["facility_count"] = self._report.facility_count
        outputs["total_tco2e"] = self._baseline.total_tco2e
        outputs["top_source_count"] = len(top_sources)
        outputs["recommendation_count"] = len(recommendations)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Baseline report generated: %d recommendations", len(recommendations))
        return PhaseResult(
            phase_name="baseline_report",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _identify_top_sources(self) -> List[Dict[str, Any]]:
        """Identify top emission sources from the baseline."""
        sources: List[Dict[str, Any]] = []
        b = self._baseline
        total = b.total_tco2e or 1.0

        if b.scope1_stationary_tco2e > 0:
            sources.append({
                "source": "Scope 1 - Stationary Combustion",
                "tco2e": b.scope1_stationary_tco2e,
                "share_pct": round((b.scope1_stationary_tco2e / total) * 100, 1),
            })
        if b.scope1_mobile_tco2e > 0:
            sources.append({
                "source": "Scope 1 - Mobile Combustion",
                "tco2e": b.scope1_mobile_tco2e,
                "share_pct": round((b.scope1_mobile_tco2e / total) * 100, 1),
            })
        if b.scope2_location_tco2e > 0:
            sources.append({
                "source": "Scope 2 - Purchased Electricity",
                "tco2e": b.scope2_location_tco2e,
                "share_pct": round((b.scope2_location_tco2e / total) * 100, 1),
            })
        for cat, val in sorted(b.scope3_by_category.items(), key=lambda x: x[1], reverse=True):
            if val > 0:
                sources.append({
                    "source": f"Scope 3 - {cat}",
                    "tco2e": val,
                    "share_pct": round((val / total) * 100, 1),
                })

        sources.sort(key=lambda x: x["tco2e"], reverse=True)
        return sources[:10]

    def _generate_baseline_recommendations(self) -> List[str]:
        """Generate strategic recommendations based on baseline."""
        recs: List[str] = []
        b = self._baseline
        total = b.total_tco2e or 1.0

        # Scope distribution recommendations
        scope1_pct = (b.scope1_total_tco2e / total) * 100 if total > 0 else 0
        scope2_pct = (b.scope2_location_tco2e / total) * 100 if total > 0 else 0
        scope3_pct = (b.scope3_total_tco2e / total) * 100 if total > 0 else 0

        if scope1_pct > 30:
            recs.append(
                f"Scope 1 accounts for {scope1_pct:.0f}% of total emissions. "
                "Prioritise fuel switching, electrification, and energy efficiency."
            )
        if scope2_pct > 20:
            recs.append(
                f"Scope 2 accounts for {scope2_pct:.0f}% of total emissions. "
                "Consider renewable energy procurement (PPAs, GoOs) and on-site generation."
            )
        if scope3_pct > 50:
            recs.append(
                f"Scope 3 accounts for {scope3_pct:.0f}% of total emissions. "
                "Engage key suppliers on emission reduction targets and collect primary data."
            )
        if b.scope3_total_tco2e <= 0 and self.config.include_scope3:
            recs.append(
                "Scope 3 emissions are zero. Ensure procurement data is complete."
            )

        # General recommendations
        recs.append("Set science-based targets aligned with SBTi Net-Zero Standard.")
        recs.append("Develop a reduction roadmap with short, medium, and long-term actions.")
        if self._dq_report.overall_score > 3.0:
            recs.append(
                "Improve data quality (current score {:.1f}/5) by installing sub-metering "
                "and collecting supplier-specific emission factors.".format(self._dq_report.overall_score)
            )
        return recs

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    def _provenance_of_result(self, result: OnboardingResult) -> str:
        """Compute SHA-256 provenance hash of the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)
