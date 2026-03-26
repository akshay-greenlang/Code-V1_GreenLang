# -*- coding: utf-8 -*-
"""
Scope 2 Calculation Workflow
=================================

4-phase workflow for computing all Scope 2 indirect GHG emissions with
dual-method reporting within PACK-041 Scope 1-2 Complete Pack.

Phases:
    1. InstrumentCollection      -- Collect and validate contractual instruments
                                    (PPAs, RECs, GOs)
    2. DualMethodExecution       -- Execute location-based (MRV-009) and
                                    market-based (MRV-010) in parallel, plus
                                    steam/heat/cooling (MRV-011, MRV-012)
    3. AllocationReconciliation  -- Allocate instruments per GHG Protocol
                                    hierarchy, prevent double-counting
    4. DualReportGeneration      -- Reconcile location vs market, generate
                                    dual-report with variance analysis

The workflow follows GreenLang zero-hallucination principles: all emission
factors from published grid data (IEA, eGRID, AIB). SHA-256 provenance hashes
guarantee auditability.

Regulatory Basis:
    GHG Protocol Scope 2 Guidance (2015) - dual reporting requirement
    ISO 14064-1:2018 Clause 5.2.3 (Energy indirect GHG emissions)
    EU CSRD / ESRS E1-6 (Energy consumption and mix)

Schedule: on-demand (calculation cycle)
Estimated duration: 45 minutes

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
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


class Scope2Method(str, Enum):
    """Scope 2 calculation methods per GHG Protocol."""

    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"


class InstrumentType(str, Enum):
    """Types of contractual instruments for market-based method."""

    ENERGY_ATTRIBUTE_CERTIFICATE = "energy_attribute_certificate"
    REC = "renewable_energy_certificate"
    GO = "guarantee_of_origin"
    PPA = "power_purchase_agreement"
    GREEN_TARIFF = "green_tariff"
    DIRECT_CONTRACT = "direct_contract"
    UNBUNDLED_CERTIFICATE = "unbundled_certificate"
    RESIDUAL_MIX = "residual_mix"


class InstrumentQuality(str, Enum):
    """GHG Protocol Scope 2 instrument quality hierarchy."""

    TIER1_CERTIFICATE = "tier1_certificate"
    TIER2_CONTRACT = "tier2_contract"
    TIER3_SUPPLIER_MIX = "tier3_supplier_mix"
    TIER4_RESIDUAL_MIX = "tier4_residual_mix"
    TIER5_GRID_AVERAGE = "tier5_grid_average"


class EnergyType(str, Enum):
    """Types of purchased energy."""

    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEAT = "heat"
    COOLING = "cooling"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ContractualInstrument(BaseModel):
    """Contractual instrument for market-based Scope 2."""

    instrument_id: str = Field(default_factory=lambda: f"inst-{uuid.uuid4().hex[:8]}")
    instrument_type: InstrumentType = Field(default=InstrumentType.REC)
    quality_tier: InstrumentQuality = Field(default=InstrumentQuality.TIER1_CERTIFICATE)
    facility_id: str = Field(default="", description="Facility this instrument applies to")
    energy_type: EnergyType = Field(default=EnergyType.ELECTRICITY)
    mwh_covered: float = Field(default=0.0, ge=0.0, description="MWh covered by instrument")
    ef_tco2e_mwh: float = Field(default=0.0, ge=0.0, description="Emission factor tCO2e/MWh")
    issuer: str = Field(default="", description="Certificate issuing body")
    tracking_system: str = Field(default="", description="e.g. M-RETS, NAR, AIB")
    vintage_year: int = Field(default=2025, ge=2015)
    country: str = Field(default="")
    is_valid: bool = Field(default=True)
    validation_notes: str = Field(default="")


class FacilityConsumption(BaseModel):
    """Energy consumption data for a facility."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    entity_id: str = Field(default="")
    country: str = Field(default="")
    grid_region: str = Field(default="", description="Grid region code e.g. ERCOT, MISO, UK")
    electricity_mwh: float = Field(default=0.0, ge=0.0)
    steam_mwh: float = Field(default=0.0, ge=0.0)
    heat_mwh: float = Field(default=0.0, ge=0.0)
    cooling_mwh: float = Field(default=0.0, ge=0.0)
    total_energy_mwh: float = Field(default=0.0, ge=0.0)
    instruments: List[str] = Field(default_factory=list, description="Instrument IDs")


class FacilityDualResult(BaseModel):
    """Dual-method Scope 2 result for a single facility."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    location_based_tco2e: float = Field(default=0.0, ge=0.0)
    market_based_tco2e: float = Field(default=0.0, ge=0.0)
    delta_tco2e: float = Field(default=0.0, description="location - market (can be negative)")
    delta_pct: float = Field(default=0.0, description="Delta as % of location-based")
    electricity_location_tco2e: float = Field(default=0.0, ge=0.0)
    electricity_market_tco2e: float = Field(default=0.0, ge=0.0)
    steam_heat_tco2e: float = Field(default=0.0, ge=0.0)
    cooling_tco2e: float = Field(default=0.0, ge=0.0)
    instruments_applied: List[str] = Field(default_factory=list)
    residual_mwh: float = Field(default=0.0, ge=0.0, description="MWh covered by residual mix")


class InstrumentAllocation(BaseModel):
    """Allocation of an instrument to a facility."""

    instrument_id: str = Field(default="")
    facility_id: str = Field(default="")
    mwh_allocated: float = Field(default=0.0, ge=0.0)
    ef_applied_tco2e_mwh: float = Field(default=0.0, ge=0.0)
    tco2e_avoided: float = Field(default=0.0, ge=0.0)
    quality_tier: str = Field(default="")


class VarianceAnalysis(BaseModel):
    """Location vs market variance analysis."""

    total_location_tco2e: float = Field(default=0.0, ge=0.0)
    total_market_tco2e: float = Field(default=0.0, ge=0.0)
    absolute_delta_tco2e: float = Field(default=0.0)
    relative_delta_pct: float = Field(default=0.0)
    primary_driver: str = Field(default="")
    instrument_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    facilities_with_instruments: int = Field(default=0, ge=0)
    facilities_on_residual_mix: int = Field(default=0, ge=0)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class Scope2CalculationInput(BaseModel):
    """Input data model for Scope2CalculationWorkflow."""

    facilities: List[FacilityConsumption] = Field(
        default_factory=list, description="Facility consumption data"
    )
    consumption_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional consumption data"
    )
    instruments: List[ContractualInstrument] = Field(
        default_factory=list, description="Contractual instruments"
    )
    grid_factors: Dict[str, float] = Field(
        default_factory=dict, description="Grid emission factors tCO2e/MWh by region"
    )
    residual_factors: Dict[str, float] = Field(
        default_factory=dict, description="Residual mix factors tCO2e/MWh by region"
    )
    steam_factors: Dict[str, float] = Field(
        default_factory=dict, description="Steam/heat emission factors tCO2e/MWh by supplier"
    )
    cooling_factors: Dict[str, float] = Field(
        default_factory=dict, description="Cooling emission factors tCO2e/MWh by supplier"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facilities")
    @classmethod
    def validate_facilities(cls, v: List[FacilityConsumption]) -> List[FacilityConsumption]:
        """Ensure at least one facility is provided."""
        if not v:
            raise ValueError("At least one facility must be provided")
        return v


class Scope2CalculationResult(BaseModel):
    """Complete result from Scope 2 calculation workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scope2_calculation")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    location_based_total: float = Field(default=0.0, ge=0.0)
    market_based_total: float = Field(default=0.0, ge=0.0)
    per_facility_dual: List[FacilityDualResult] = Field(default_factory=list)
    instrument_allocation: List[InstrumentAllocation] = Field(default_factory=list)
    reconciliation: Optional[VarianceAnalysis] = Field(default=None)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# GRID EMISSION FACTORS (Zero-Hallucination, IEA/eGRID sourced)
# =============================================================================

# Default grid-average emission factors (tCO2e/MWh) by region
DEFAULT_GRID_FACTORS: Dict[str, float] = {
    "US_ERCOT": 0.3960,
    "US_MISO": 0.5290,
    "US_PJM": 0.3870,
    "US_WECC": 0.2850,
    "US_NPCC": 0.2100,
    "US_SERC": 0.4150,
    "US_SPP": 0.4530,
    "US_AVERAGE": 0.3860,
    "UK": 0.2070,
    "DE": 0.3380,
    "FR": 0.0520,
    "ES": 0.1510,
    "IT": 0.2580,
    "NL": 0.3280,
    "PL": 0.6320,
    "SE": 0.0080,
    "NO": 0.0080,
    "EU_AVERAGE": 0.2300,
    "CN": 0.5550,
    "IN": 0.7080,
    "JP": 0.4570,
    "AU": 0.6560,
    "BR": 0.0740,
    "CA": 0.1200,
    "WORLD_AVERAGE": 0.4360,
}

# Default residual mix factors (tCO2e/MWh)
DEFAULT_RESIDUAL_FACTORS: Dict[str, float] = {
    "UK": 0.2810,
    "DE": 0.4560,
    "FR": 0.0680,
    "ES": 0.2010,
    "IT": 0.3420,
    "NL": 0.4180,
    "PL": 0.7210,
    "SE": 0.0190,
    "NO": 0.0190,
    "EU_AVERAGE": 0.3400,
    "US_AVERAGE": 0.3860,
    "WORLD_AVERAGE": 0.4360,
}

# Default steam/heat emission factors (tCO2e/MWh)
DEFAULT_STEAM_FACTORS: Dict[str, float] = {
    "natural_gas_boiler": 0.2200,
    "coal_boiler": 0.3600,
    "biomass_boiler": 0.0150,
    "chp_natural_gas": 0.1800,
    "district_heating_eu": 0.1900,
    "default": 0.2100,
}

# Default cooling emission factors (tCO2e/MWh)
DEFAULT_COOLING_FACTORS: Dict[str, float] = {
    "electric_chiller": 0.1400,
    "absorption_chiller": 0.0800,
    "district_cooling": 0.1200,
    "default": 0.1300,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class Scope2CalculationWorkflow:
    """
    4-phase Scope 2 GHG emission calculation workflow with dual-method reporting.

    Collects and validates contractual instruments, executes location-based and
    market-based calculations in parallel using MRV agents 009-012, allocates
    instruments per the GHG Protocol quality hierarchy, and generates a
    dual-report with variance analysis.

    Zero-hallucination: all emission factors from published grid data.
    No LLM in numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _validated_instruments: Validated contractual instruments.
        _facility_dual_results: Per-facility dual-method results.
        _instrument_allocations: Instrument allocations.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = Scope2CalculationWorkflow()
        >>> inp = Scope2CalculationInput(facilities=[...], instruments=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_DEPENDENCIES: Dict[str, List[str]] = {
        "instrument_collection": [],
        "dual_method_execution": ["instrument_collection"],
        "allocation_reconciliation": ["dual_method_execution"],
        "dual_report_generation": ["allocation_reconciliation"],
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize Scope2CalculationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._validated_instruments: List[ContractualInstrument] = []
        self._facility_dual_results: List[FacilityDualResult] = []
        self._instrument_allocations: List[InstrumentAllocation] = []
        self._variance: Optional[VarianceAnalysis] = None
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[Scope2CalculationInput] = None,
        facilities: Optional[List[FacilityConsumption]] = None,
        instruments: Optional[List[ContractualInstrument]] = None,
    ) -> Scope2CalculationResult:
        """
        Execute the 4-phase Scope 2 calculation workflow.

        Args:
            input_data: Full input model (preferred).
            facilities: Facility list (fallback).
            instruments: Instruments list (fallback).

        Returns:
            Scope2CalculationResult with dual-method totals and variance analysis.

        Raises:
            ValueError: If no facilities are provided.
        """
        if input_data is None:
            if facilities is None or not facilities:
                raise ValueError("Either input_data or facilities must be provided")
            input_data = Scope2CalculationInput(
                facilities=facilities,
                instruments=instruments or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting Scope 2 calculation workflow %s facilities=%d instruments=%d",
            self.workflow_id,
            len(input_data.facilities),
            len(input_data.instruments),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._execute_with_retry(
                self._phase_instrument_collection, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            phase2 = await self._execute_with_retry(
                self._phase_dual_method_execution, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            phase3 = await self._execute_with_retry(
                self._phase_allocation_reconciliation, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            phase4 = await self._execute_with_retry(
                self._phase_dual_report_generation, input_data, phase_number=4
            )
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Scope 2 calculation workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        location_total = sum(f.location_based_tco2e for f in self._facility_dual_results)
        market_total = sum(f.market_based_tco2e for f in self._facility_dual_results)

        result = Scope2CalculationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            location_based_total=round(location_total, 4),
            market_based_total=round(market_total, 4),
            per_facility_dual=self._facility_dual_results,
            instrument_allocation=self._instrument_allocations,
            reconciliation=self._variance,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Scope 2 calculation workflow %s completed in %.2fs status=%s "
            "location=%.2f market=%.2f tCO2e",
            self.workflow_id, elapsed, overall_status.value,
            location_total, market_total,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: Scope2CalculationInput, phase_number: int
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
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
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
    # Phase 1: Instrument Collection
    # -------------------------------------------------------------------------

    async def _phase_instrument_collection(
        self, input_data: Scope2CalculationInput
    ) -> PhaseResult:
        """Collect and validate contractual instruments (PPAs, RECs, GOs)."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._validated_instruments = []

        for inst in input_data.instruments:
            validation_issues = self._validate_instrument(inst, input_data.reporting_year)

            if validation_issues:
                inst.is_valid = False
                inst.validation_notes = "; ".join(validation_issues)
                warnings.append(
                    f"Instrument {inst.instrument_id} ({inst.instrument_type.value}): "
                    f"{inst.validation_notes}"
                )
            else:
                inst.is_valid = True

            self._validated_instruments.append(inst)

        valid_count = sum(1 for i in self._validated_instruments if i.is_valid)
        total_mwh = sum(i.mwh_covered for i in self._validated_instruments if i.is_valid)

        # Categorize by quality tier
        tier_summary: Dict[str, int] = {}
        for inst in self._validated_instruments:
            if inst.is_valid:
                tier = inst.quality_tier.value
                tier_summary[tier] = tier_summary.get(tier, 0) + 1

        outputs["total_instruments"] = len(self._validated_instruments)
        outputs["valid_instruments"] = valid_count
        outputs["invalid_instruments"] = len(self._validated_instruments) - valid_count
        outputs["total_mwh_covered"] = round(total_mwh, 2)
        outputs["by_tier"] = tier_summary
        outputs["by_type"] = {}
        for inst in self._validated_instruments:
            if inst.is_valid:
                t = inst.instrument_type.value
                outputs["by_type"][t] = outputs["by_type"].get(t, 0) + 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 InstrumentCollection: %d/%d valid, %.0f MWh covered",
            valid_count, len(self._validated_instruments), total_mwh,
        )
        return PhaseResult(
            phase_name="instrument_collection",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _validate_instrument(
        self, inst: ContractualInstrument, reporting_year: int
    ) -> List[str]:
        """Validate a contractual instrument per GHG Protocol quality criteria."""
        issues: List[str] = []

        if inst.mwh_covered <= 0:
            issues.append("MWh covered must be positive")

        if inst.vintage_year > reporting_year:
            issues.append(f"Vintage year {inst.vintage_year} is after reporting year {reporting_year}")

        if inst.vintage_year < reporting_year - 1:
            issues.append(
                f"Vintage year {inst.vintage_year} is more than 1 year before "
                f"reporting year {reporting_year}"
            )

        if inst.instrument_type == InstrumentType.UNBUNDLED_CERTIFICATE:
            if inst.quality_tier not in (
                InstrumentQuality.TIER1_CERTIFICATE,
                InstrumentQuality.TIER2_CONTRACT,
            ):
                issues.append("Unbundled certificates should be Tier 1 or Tier 2")

        if not inst.tracking_system and inst.instrument_type in (
            InstrumentType.REC, InstrumentType.GO, InstrumentType.ENERGY_ATTRIBUTE_CERTIFICATE
        ):
            issues.append("Certificate instruments should specify a tracking system")

        return issues

    # -------------------------------------------------------------------------
    # Phase 2: Dual Method Execution
    # -------------------------------------------------------------------------

    async def _phase_dual_method_execution(
        self, input_data: Scope2CalculationInput
    ) -> PhaseResult:
        """Execute location-based and market-based calculations for all facilities."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._facility_dual_results = []

        for facility in input_data.facilities:
            # Location-based calculation (MRV-009)
            location_elec = self._calc_location_based_electricity(
                facility, input_data.grid_factors
            )

            # Steam/heat (MRV-011)
            steam_heat = self._calc_steam_heat(facility, input_data.steam_factors)

            # Cooling (MRV-012)
            cooling = self._calc_cooling(facility, input_data.cooling_factors)

            location_total = location_elec + steam_heat + cooling

            # Market-based: initially same as location (will be adjusted in Phase 3)
            market_elec = self._calc_market_based_electricity(
                facility, input_data.residual_factors, input_data.grid_factors
            )
            market_total = market_elec + steam_heat + cooling

            delta = location_total - market_total
            delta_pct = (delta / location_total * 100.0) if location_total > 0 else 0.0

            self._facility_dual_results.append(FacilityDualResult(
                facility_id=facility.facility_id,
                facility_name=facility.facility_name,
                location_based_tco2e=round(location_total, 6),
                market_based_tco2e=round(market_total, 6),
                delta_tco2e=round(delta, 6),
                delta_pct=round(delta_pct, 2),
                electricity_location_tco2e=round(location_elec, 6),
                electricity_market_tco2e=round(market_elec, 6),
                steam_heat_tco2e=round(steam_heat, 6),
                cooling_tco2e=round(cooling, 6),
                residual_mwh=facility.electricity_mwh,
            ))

        total_location = sum(f.location_based_tco2e for f in self._facility_dual_results)
        total_market = sum(f.market_based_tco2e for f in self._facility_dual_results)

        outputs["facilities_calculated"] = len(self._facility_dual_results)
        outputs["total_location_tco2e"] = round(total_location, 4)
        outputs["total_market_tco2e"] = round(total_market, 4)
        outputs["total_electricity_mwh"] = round(
            sum(f.electricity_mwh for f in input_data.facilities), 2
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DualMethodExecution: %d facilities, location=%.2f market=%.2f tCO2e",
            len(self._facility_dual_results), total_location, total_market,
        )
        return PhaseResult(
            phase_name="dual_method_execution",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calc_location_based_electricity(
        self, facility: FacilityConsumption, grid_factors: Dict[str, float]
    ) -> float:
        """Calculate location-based electricity emissions using grid-average factors."""
        region = facility.grid_region or facility.country or "WORLD_AVERAGE"

        # Look up grid factor: user override > default
        ef = grid_factors.get(region, DEFAULT_GRID_FACTORS.get(region, DEFAULT_GRID_FACTORS["WORLD_AVERAGE"]))

        return facility.electricity_mwh * ef

    def _calc_market_based_electricity(
        self, facility: FacilityConsumption,
        residual_factors: Dict[str, float],
        grid_factors: Dict[str, float],
    ) -> float:
        """Calculate market-based electricity emissions using residual mix (pre-instrument)."""
        region = facility.grid_region or facility.country or "WORLD_AVERAGE"

        # Use residual mix as default for market-based (instruments applied in Phase 3)
        ef = residual_factors.get(
            region,
            DEFAULT_RESIDUAL_FACTORS.get(
                region,
                grid_factors.get(region, DEFAULT_GRID_FACTORS.get(region, DEFAULT_GRID_FACTORS["WORLD_AVERAGE"]))
            )
        )
        return facility.electricity_mwh * ef

    def _calc_steam_heat(
        self, facility: FacilityConsumption, steam_factors: Dict[str, float]
    ) -> float:
        """Calculate steam and heat purchase emissions (MRV-011)."""
        total_steam_mwh = facility.steam_mwh + facility.heat_mwh
        if total_steam_mwh <= 0:
            return 0.0

        ef = steam_factors.get(
            facility.facility_id,
            DEFAULT_STEAM_FACTORS.get("default", 0.2100)
        )
        return total_steam_mwh * ef

    def _calc_cooling(
        self, facility: FacilityConsumption, cooling_factors: Dict[str, float]
    ) -> float:
        """Calculate cooling purchase emissions (MRV-012)."""
        if facility.cooling_mwh <= 0:
            return 0.0

        ef = cooling_factors.get(
            facility.facility_id,
            DEFAULT_COOLING_FACTORS.get("default", 0.1300)
        )
        return facility.cooling_mwh * ef

    # -------------------------------------------------------------------------
    # Phase 3: Allocation Reconciliation
    # -------------------------------------------------------------------------

    async def _phase_allocation_reconciliation(
        self, input_data: Scope2CalculationInput
    ) -> PhaseResult:
        """Allocate instruments per GHG Protocol hierarchy, prevent double-counting."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._instrument_allocations = []

        # Build facility consumption lookup
        facility_mwh: Dict[str, float] = {}
        for fac in input_data.facilities:
            facility_mwh[fac.facility_id] = fac.electricity_mwh

        # Sort instruments by quality tier (best first) per GHG Protocol hierarchy
        tier_priority = {
            InstrumentQuality.TIER1_CERTIFICATE: 1,
            InstrumentQuality.TIER2_CONTRACT: 2,
            InstrumentQuality.TIER3_SUPPLIER_MIX: 3,
            InstrumentQuality.TIER4_RESIDUAL_MIX: 4,
            InstrumentQuality.TIER5_GRID_AVERAGE: 5,
        }
        valid_instruments = [i for i in self._validated_instruments if i.is_valid]
        valid_instruments.sort(key=lambda i: tier_priority.get(i.quality_tier, 99))

        # Track remaining unmatched MWh per facility
        remaining_mwh: Dict[str, float] = dict(facility_mwh)
        allocated_instruments_per_facility: Dict[str, List[str]] = {}

        for inst in valid_instruments:
            target_fac = inst.facility_id
            if not target_fac or target_fac not in remaining_mwh:
                # Try to allocate to any facility with remaining MWh
                for fac_id, rem in remaining_mwh.items():
                    if rem > 0:
                        target_fac = fac_id
                        break

            if not target_fac or remaining_mwh.get(target_fac, 0) <= 0:
                warnings.append(
                    f"Instrument {inst.instrument_id} could not be allocated: "
                    f"no remaining MWh at target facility"
                )
                continue

            # Allocate MWh (capped at remaining)
            allocatable = min(inst.mwh_covered, remaining_mwh[target_fac])
            remaining_mwh[target_fac] -= allocatable

            # Calculate avoided emissions vs residual mix
            region = ""
            for fac in input_data.facilities:
                if fac.facility_id == target_fac:
                    region = fac.grid_region or fac.country or "WORLD_AVERAGE"
                    break

            residual_ef = input_data.residual_factors.get(
                region,
                DEFAULT_RESIDUAL_FACTORS.get(region, DEFAULT_RESIDUAL_FACTORS.get("WORLD_AVERAGE", 0.436))
            )
            instrument_ef = inst.ef_tco2e_mwh
            avoided = allocatable * (residual_ef - instrument_ef)

            self._instrument_allocations.append(InstrumentAllocation(
                instrument_id=inst.instrument_id,
                facility_id=target_fac,
                mwh_allocated=round(allocatable, 4),
                ef_applied_tco2e_mwh=round(instrument_ef, 6),
                tco2e_avoided=round(max(0.0, avoided), 6),
                quality_tier=inst.quality_tier.value,
            ))

            allocated_instruments_per_facility.setdefault(target_fac, []).append(inst.instrument_id)

        # Update market-based results with instrument allocations
        for fac_result in self._facility_dual_results:
            fac_id = fac_result.facility_id
            fac_allocations = [a for a in self._instrument_allocations if a.facility_id == fac_id]

            if fac_allocations:
                total_avoided = sum(a.tco2e_avoided for a in fac_allocations)
                fac_result.market_based_tco2e = round(
                    max(0.0, fac_result.market_based_tco2e - total_avoided), 6
                )
                fac_result.instruments_applied = [a.instrument_id for a in fac_allocations]
                fac_result.residual_mwh = remaining_mwh.get(fac_id, 0.0)

                # Recalculate delta
                fac_result.delta_tco2e = round(
                    fac_result.location_based_tco2e - fac_result.market_based_tco2e, 6
                )
                if fac_result.location_based_tco2e > 0:
                    fac_result.delta_pct = round(
                        fac_result.delta_tco2e / fac_result.location_based_tco2e * 100.0, 2
                    )

        total_avoided = sum(a.tco2e_avoided for a in self._instrument_allocations)
        total_allocated_mwh = sum(a.mwh_allocated for a in self._instrument_allocations)
        total_consumption_mwh = sum(facility_mwh.values())

        outputs["total_allocations"] = len(self._instrument_allocations)
        outputs["total_mwh_allocated"] = round(total_allocated_mwh, 2)
        outputs["total_tco2e_avoided"] = round(total_avoided, 4)
        outputs["coverage_pct"] = round(
            (total_allocated_mwh / total_consumption_mwh * 100.0)
            if total_consumption_mwh > 0 else 0.0, 2
        )
        outputs["facilities_with_instruments"] = len(allocated_instruments_per_facility)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 AllocationReconciliation: %d allocations, %.0f MWh, %.2f tCO2e avoided",
            len(self._instrument_allocations), total_allocated_mwh, total_avoided,
        )
        return PhaseResult(
            phase_name="allocation_reconciliation",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Dual Report Generation
    # -------------------------------------------------------------------------

    async def _phase_dual_report_generation(
        self, input_data: Scope2CalculationInput
    ) -> PhaseResult:
        """Reconcile location vs market, generate dual-report with variance analysis."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_location = sum(f.location_based_tco2e for f in self._facility_dual_results)
        total_market = sum(f.market_based_tco2e for f in self._facility_dual_results)

        absolute_delta = total_location - total_market
        relative_delta = (absolute_delta / total_location * 100.0) if total_location > 0 else 0.0

        # Determine primary driver
        if absolute_delta > 0:
            primary_driver = "Contractual instruments reduce market-based below location-based"
        elif absolute_delta < 0:
            primary_driver = "Residual mix factors exceed grid-average factors"
        else:
            primary_driver = "Location-based and market-based methods yield identical results"

        total_consumption = sum(f.electricity_mwh for f in input_data.facilities)
        total_instrument_mwh = sum(a.mwh_allocated for a in self._instrument_allocations)
        instrument_coverage = (
            (total_instrument_mwh / total_consumption * 100.0) if total_consumption > 0 else 0.0
        )

        fac_with_instruments = len({
            a.facility_id for a in self._instrument_allocations
        })
        fac_on_residual = len(self._facility_dual_results) - fac_with_instruments

        self._variance = VarianceAnalysis(
            total_location_tco2e=round(total_location, 4),
            total_market_tco2e=round(total_market, 4),
            absolute_delta_tco2e=round(absolute_delta, 4),
            relative_delta_pct=round(relative_delta, 2),
            primary_driver=primary_driver,
            instrument_coverage_pct=round(instrument_coverage, 2),
            facilities_with_instruments=fac_with_instruments,
            facilities_on_residual_mix=fac_on_residual,
        )

        # Compliance check: GHG Protocol requires reporting both methods
        if total_location == 0 and total_market == 0:
            warnings.append("Both location and market totals are zero; verify input data")

        outputs["location_total_tco2e"] = round(total_location, 4)
        outputs["market_total_tco2e"] = round(total_market, 4)
        outputs["absolute_delta_tco2e"] = round(absolute_delta, 4)
        outputs["relative_delta_pct"] = round(relative_delta, 2)
        outputs["instrument_coverage_pct"] = round(instrument_coverage, 2)
        outputs["primary_driver"] = primary_driver
        outputs["ghg_protocol_dual_report_complete"] = True

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 DualReportGeneration: location=%.2f market=%.2f delta=%.2f tCO2e (%.1f%%)",
            total_location, total_market, absolute_delta, relative_delta,
        )
        return PhaseResult(
            phase_name="dual_report_generation",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._validated_instruments = []
        self._facility_dual_results = []
        self._instrument_allocations = []
        self._variance = None
        self._phase_results = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: Scope2CalculationResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.location_based_total}|{result.market_based_total}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
