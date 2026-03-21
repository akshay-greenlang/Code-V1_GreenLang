# -*- coding: utf-8 -*-
"""
EUETSBridge - EU Emissions Trading System Integration for PACK-031
====================================================================

This module provides integration with the EU Emissions Trading System (EU ETS)
for energy-intensive industrial installations. It manages installation permit
tracking, MRV plan compliance, free allocation versus actual emissions
analysis, carbon price impact on energy savings ROI, benchmarking against
EU ETS product benchmarks, and compliance cycle tracking.

EU ETS Phase 4 (2021-2030) Key Elements:
    - Linear reduction factor: 2.2% per year
    - Free allocation based on product benchmarks
    - Carbon Leakage List (CLL) classification
    - Monitoring Plan (MP) and Annual Emissions Report (AER)
    - Compliance cycle: 28 February (verify), 30 April (surrender)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
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


class CarbonLeakageStatus(str, Enum):
    """Carbon leakage classification."""

    ON_CLL = "on_carbon_leakage_list"
    NOT_ON_CLL = "not_on_carbon_leakage_list"
    UNKNOWN = "unknown"


class AllocationMethod(str, Enum):
    """Free allocation method."""

    PRODUCT_BENCHMARK = "product_benchmark"
    HEAT_BENCHMARK = "heat_benchmark"
    FUEL_BENCHMARK = "fuel_benchmark"
    PROCESS_EMISSIONS = "process_emissions"


class ComplianceCycleStatus(str, Enum):
    """EU ETS compliance cycle status."""

    MONITORING = "monitoring"
    REPORT_SUBMITTED = "report_submitted"
    VERIFIED = "verified"
    SURRENDERED = "surrendered"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENALTY = "penalty"


class ETSPhase(str, Enum):
    """EU ETS trading phases."""

    PHASE_3 = "phase_3"  # 2013-2020
    PHASE_4 = "phase_4"  # 2021-2030


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class InstallationPermit(BaseModel):
    """EU ETS installation permit record."""

    permit_id: str = Field(default_factory=_new_uuid)
    installation_name: str = Field(default="")
    permit_number: str = Field(default="")
    competent_authority: str = Field(default="")
    member_state: str = Field(default="")
    activity_code: str = Field(default="", description="Annex I activity code")
    activity_description: str = Field(default="")
    rated_thermal_input_mw: float = Field(default=0.0, ge=0)
    carbon_leakage_status: CarbonLeakageStatus = Field(default=CarbonLeakageStatus.UNKNOWN)
    phase: ETSPhase = Field(default=ETSPhase.PHASE_4)
    permit_start_date: Optional[date] = Field(None)
    permit_expiry_date: Optional[date] = Field(None)
    monitoring_plan_approved: bool = Field(default=False)


class FreeAllocationRecord(BaseModel):
    """Free allocation record for an installation."""

    record_id: str = Field(default_factory=_new_uuid)
    installation_id: str = Field(default="")
    year: int = Field(default=2025, ge=2021, le=2030)
    allocation_method: AllocationMethod = Field(default=AllocationMethod.PRODUCT_BENCHMARK)
    benchmark_value: float = Field(default=0.0, ge=0, description="Benchmark in tCO2/unit")
    activity_level: float = Field(default=0.0, ge=0, description="Production or heat output")
    activity_unit: str = Field(default="tonnes")
    cross_sectoral_correction_factor: float = Field(default=1.0, ge=0, le=1)
    carbon_leakage_factor: float = Field(default=1.0, ge=0, le=1)
    free_allocation_eua: float = Field(default=0.0, ge=0, description="Free EUAs allocated")


class EmissionsRecord(BaseModel):
    """Annual emissions record for an installation."""

    record_id: str = Field(default_factory=_new_uuid)
    installation_id: str = Field(default="")
    year: int = Field(default=2025)
    verified_emissions_tco2: float = Field(default=0.0, ge=0)
    free_allocation_eua: float = Field(default=0.0, ge=0)
    surplus_deficit_eua: float = Field(default=0.0, description="Positive = surplus")
    eua_price_eur: float = Field(default=0.0, ge=0, description="Average EUA price")
    carbon_cost_eur: float = Field(default=0.0, description="Cost of deficit allowances")
    compliance_status: ComplianceCycleStatus = Field(default=ComplianceCycleStatus.MONITORING)
    provenance_hash: str = Field(default="")


class CarbonPriceImpact(BaseModel):
    """Carbon price impact analysis on energy savings ROI."""

    analysis_id: str = Field(default_factory=_new_uuid)
    opportunity_id: str = Field(default="")
    energy_savings_kwh: float = Field(default=0.0)
    avoided_emissions_tco2e: float = Field(default=0.0)
    eua_price_eur: float = Field(default=0.0)
    carbon_savings_eur: float = Field(default=0.0)
    energy_savings_eur: float = Field(default=0.0)
    total_savings_eur: float = Field(default=0.0)
    investment_cost_eur: float = Field(default=0.0)
    payback_without_carbon_years: float = Field(default=0.0)
    payback_with_carbon_years: float = Field(default=0.0)
    carbon_benefit_pct: float = Field(default=0.0, description="Pct of savings from carbon")
    provenance_hash: str = Field(default="")


class ETSBenchmarkComparison(BaseModel):
    """Comparison against EU ETS product benchmarks."""

    comparison_id: str = Field(default_factory=_new_uuid)
    installation_id: str = Field(default="")
    product: str = Field(default="")
    benchmark_value_tco2_per_unit: float = Field(default=0.0)
    actual_value_tco2_per_unit: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    performance_vs_benchmark: str = Field(default="", description="above|at|below benchmark")
    improvement_potential_tco2: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ComplianceCycle(BaseModel):
    """EU ETS compliance cycle tracking."""

    cycle_id: str = Field(default_factory=_new_uuid)
    installation_id: str = Field(default="")
    reporting_year: int = Field(default=2025)
    monitoring_plan_approved: bool = Field(default=False)
    aer_submitted: bool = Field(default=False)
    aer_submission_date: Optional[date] = Field(None)
    verification_complete: bool = Field(default=False)
    verification_date: Optional[date] = Field(None)
    verifier_name: str = Field(default="")
    allowances_surrendered: bool = Field(default=False)
    surrender_date: Optional[date] = Field(None)
    status: ComplianceCycleStatus = Field(default=ComplianceCycleStatus.MONITORING)


class EUETSBridgeConfig(BaseModel):
    """Configuration for the EU ETS Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    default_eua_price_eur: float = Field(default=80.0, ge=0, description="Default EUA price")
    phase: ETSPhase = Field(default=ETSPhase.PHASE_4)
    linear_reduction_factor_pct: float = Field(default=2.2, ge=0)


# ---------------------------------------------------------------------------
# EU ETS Product Benchmarks (selected, tCO2 per unit)
# ---------------------------------------------------------------------------

EU_ETS_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "hot_metal": {"benchmark": 1.328, "unit": "tonne", "sector": "metals"},
    "sintered_ore": {"benchmark": 0.171, "unit": "tonne", "sector": "metals"},
    "coke": {"benchmark": 0.286, "unit": "tonne", "sector": "metals"},
    "cement_clinker": {"benchmark": 0.766, "unit": "tonne", "sector": "cement"},
    "lime": {"benchmark": 0.954, "unit": "tonne", "sector": "cement"},
    "float_glass": {"benchmark": 0.453, "unit": "tonne", "sector": "glass"},
    "container_glass": {"benchmark": 0.382, "unit": "tonne", "sector": "glass"},
    "ammonia": {"benchmark": 1.619, "unit": "tonne", "sector": "chemicals"},
    "nitric_acid": {"benchmark": 0.302, "unit": "tonne", "sector": "chemicals"},
    "hydrogen": {"benchmark": 8.85, "unit": "tonne", "sector": "chemicals"},
    "newsprint": {"benchmark": 0.298, "unit": "tonne", "sector": "paper"},
    "fine_paper": {"benchmark": 0.318, "unit": "tonne", "sector": "paper"},
    "heat_benchmark": {"benchmark": 0.0622, "unit": "GJ", "sector": "cross_sectoral"},
    "fuel_benchmark": {"benchmark": 0.0562, "unit": "GJ", "sector": "cross_sectoral"},
}


# ---------------------------------------------------------------------------
# EUETSBridge
# ---------------------------------------------------------------------------


class EUETSBridge:
    """EU Emissions Trading System integration for energy-intensive industries.

    Manages installation permits, free allocation analysis, carbon price
    impact on energy savings ROI, EU ETS benchmark comparison, and
    compliance cycle tracking.

    Attributes:
        config: Bridge configuration.
        _permits: Installation permit records.
        _allocations: Free allocation records.
        _emissions: Annual emissions records.
        _cycles: Compliance cycle records.

    Example:
        >>> bridge = EUETSBridge()
        >>> impact = bridge.calculate_carbon_price_impact(
        ...     energy_savings_kwh=500_000, avoided_tco2e=183.0,
        ...     investment_cost_eur=200_000, energy_savings_eur=75_000
        ... )
        >>> print(f"Carbon benefit: {impact.carbon_savings_eur} EUR")
    """

    def __init__(self, config: Optional[EUETSBridgeConfig] = None) -> None:
        """Initialize the EU ETS Bridge."""
        self.config = config or EUETSBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._permits: Dict[str, InstallationPermit] = {}
        self._allocations: List[FreeAllocationRecord] = []
        self._emissions: List[EmissionsRecord] = []
        self._cycles: List[ComplianceCycle] = []
        self.logger.info(
            "EUETSBridge initialized: EUA price=%.0f EUR, phase=%s",
            self.config.default_eua_price_eur, self.config.phase.value,
        )

    # -------------------------------------------------------------------------
    # Installation Permit Management
    # -------------------------------------------------------------------------

    def register_installation(self, permit: InstallationPermit) -> InstallationPermit:
        """Register an EU ETS installation permit.

        Args:
            permit: Installation permit data.

        Returns:
            Registered InstallationPermit.
        """
        self._permits[permit.permit_id] = permit
        self.logger.info(
            "ETS installation registered: %s (activity: %s, %.1f MW)",
            permit.installation_name, permit.activity_code,
            permit.rated_thermal_input_mw,
        )
        return permit

    def get_installation(self, permit_id: str) -> Optional[InstallationPermit]:
        """Get installation permit by ID.

        Args:
            permit_id: Installation permit identifier.

        Returns:
            InstallationPermit or None.
        """
        return self._permits.get(permit_id)

    # -------------------------------------------------------------------------
    # Free Allocation Analysis
    # -------------------------------------------------------------------------

    def calculate_free_allocation(
        self,
        installation_id: str,
        year: int,
        method: AllocationMethod,
        benchmark_value: float,
        activity_level: float,
        activity_unit: str = "tonnes",
        on_carbon_leakage_list: bool = True,
    ) -> FreeAllocationRecord:
        """Calculate free allocation for an installation.

        Deterministic formula:
            free_eua = benchmark * activity * CSCF * CL_factor

        Args:
            installation_id: Installation identifier.
            year: Allocation year.
            method: Allocation method.
            benchmark_value: Product benchmark (tCO2/unit).
            activity_level: Production activity level.
            activity_unit: Unit of activity.
            on_carbon_leakage_list: Whether on carbon leakage list.

        Returns:
            FreeAllocationRecord.
        """
        # Cross-Sectoral Correction Factor (simplified)
        cscf = 1.0  # For CLL installations

        # Carbon leakage factor
        cl_factor = 1.0 if on_carbon_leakage_list else 0.3

        # Deterministic calculation
        free_eua = benchmark_value * activity_level * cscf * cl_factor

        record = FreeAllocationRecord(
            installation_id=installation_id,
            year=year,
            allocation_method=method,
            benchmark_value=benchmark_value,
            activity_level=activity_level,
            activity_unit=activity_unit,
            cross_sectoral_correction_factor=cscf,
            carbon_leakage_factor=cl_factor,
            free_allocation_eua=round(free_eua, 2),
        )
        self._allocations.append(record)
        return record

    def analyze_allocation_position(
        self,
        installation_id: str,
        year: int,
        verified_emissions_tco2: float,
        eua_price_eur: Optional[float] = None,
    ) -> EmissionsRecord:
        """Analyze free allocation vs actual emissions position.

        Deterministic calculation:
            surplus/deficit = free_allocation - verified_emissions
            carbon_cost = max(0, deficit) * eua_price

        Args:
            installation_id: Installation identifier.
            year: Reporting year.
            verified_emissions_tco2: Verified annual emissions.
            eua_price_eur: EUA price (default from config).

        Returns:
            EmissionsRecord with surplus/deficit analysis.
        """
        price = eua_price_eur or self.config.default_eua_price_eur

        # Find free allocation for this year
        allocation = next(
            (a for a in self._allocations
             if a.installation_id == installation_id and a.year == year),
            None,
        )
        free_eua = allocation.free_allocation_eua if allocation else 0.0

        # Deterministic surplus/deficit
        surplus_deficit = free_eua - verified_emissions_tco2
        carbon_cost = max(0.0, -surplus_deficit) * price

        record = EmissionsRecord(
            installation_id=installation_id,
            year=year,
            verified_emissions_tco2=verified_emissions_tco2,
            free_allocation_eua=free_eua,
            surplus_deficit_eua=round(surplus_deficit, 2),
            eua_price_eur=price,
            carbon_cost_eur=round(carbon_cost, 2),
            compliance_status=ComplianceCycleStatus.VERIFIED,
        )
        if self.config.enable_provenance:
            record.provenance_hash = _compute_hash(record)

        self._emissions.append(record)
        return record

    # -------------------------------------------------------------------------
    # Carbon Price Impact on Energy Savings ROI
    # -------------------------------------------------------------------------

    def calculate_carbon_price_impact(
        self,
        energy_savings_kwh: float,
        avoided_tco2e: float,
        investment_cost_eur: float,
        energy_savings_eur: float,
        opportunity_id: str = "",
        eua_price_eur: Optional[float] = None,
    ) -> CarbonPriceImpact:
        """Calculate carbon price impact on energy savings ROI.

        Deterministic formulas:
            carbon_savings = avoided_tco2e * eua_price
            total_savings = energy_savings + carbon_savings
            payback_without = investment / energy_savings
            payback_with = investment / total_savings

        Args:
            energy_savings_kwh: Annual energy savings in kWh.
            avoided_tco2e: Avoided emissions in tCO2e.
            investment_cost_eur: Investment cost.
            energy_savings_eur: Annual energy cost savings.
            opportunity_id: Savings opportunity identifier.
            eua_price_eur: EUA price override.

        Returns:
            CarbonPriceImpact with enhanced ROI analysis.
        """
        price = eua_price_eur or self.config.default_eua_price_eur

        # Deterministic calculations
        carbon_savings = avoided_tco2e * price
        total_savings = energy_savings_eur + carbon_savings

        payback_without = (
            investment_cost_eur / energy_savings_eur
            if energy_savings_eur > 0 else 999.0
        )
        payback_with = (
            investment_cost_eur / total_savings
            if total_savings > 0 else 999.0
        )
        carbon_benefit = (
            carbon_savings / total_savings * 100.0
            if total_savings > 0 else 0.0
        )

        impact = CarbonPriceImpact(
            opportunity_id=opportunity_id,
            energy_savings_kwh=energy_savings_kwh,
            avoided_emissions_tco2e=avoided_tco2e,
            eua_price_eur=price,
            carbon_savings_eur=round(carbon_savings, 2),
            energy_savings_eur=energy_savings_eur,
            total_savings_eur=round(total_savings, 2),
            investment_cost_eur=investment_cost_eur,
            payback_without_carbon_years=round(payback_without, 2),
            payback_with_carbon_years=round(payback_with, 2),
            carbon_benefit_pct=round(carbon_benefit, 1),
        )
        if self.config.enable_provenance:
            impact.provenance_hash = _compute_hash(impact)

        return impact

    # -------------------------------------------------------------------------
    # EU ETS Benchmark Comparison
    # -------------------------------------------------------------------------

    def compare_to_benchmark(
        self,
        installation_id: str,
        product: str,
        actual_tco2_per_unit: float,
        annual_production: float = 0.0,
    ) -> Optional[ETSBenchmarkComparison]:
        """Compare installation performance against EU ETS benchmark.

        Deterministic calculation:
            gap_pct = ((actual - benchmark) / benchmark) * 100

        Args:
            installation_id: Installation identifier.
            product: Product benchmark key.
            actual_tco2_per_unit: Actual emissions intensity.
            annual_production: Annual production volume.

        Returns:
            ETSBenchmarkComparison, or None if benchmark not found.
        """
        bm = EU_ETS_BENCHMARKS.get(product)
        if bm is None:
            return None

        benchmark_value = bm["benchmark"]

        # Deterministic gap calculation
        gap_pct = ((actual_tco2_per_unit - benchmark_value) / benchmark_value * 100.0
                   if benchmark_value > 0 else 0.0)

        if actual_tco2_per_unit <= benchmark_value:
            performance = "at_or_below"
        else:
            performance = "above"

        improvement_potential = max(
            0.0, (actual_tco2_per_unit - benchmark_value) * annual_production
        )

        comparison = ETSBenchmarkComparison(
            installation_id=installation_id,
            product=product,
            benchmark_value_tco2_per_unit=benchmark_value,
            actual_value_tco2_per_unit=actual_tco2_per_unit,
            gap_pct=round(gap_pct, 1),
            performance_vs_benchmark=performance,
            improvement_potential_tco2=round(improvement_potential, 2),
        )
        if self.config.enable_provenance:
            comparison.provenance_hash = _compute_hash(comparison)

        return comparison

    def get_available_benchmarks(self) -> List[Dict[str, Any]]:
        """Get all available EU ETS product benchmarks.

        Returns:
            List of benchmark entries.
        """
        return [
            {
                "product": key,
                "benchmark_tco2_per_unit": val["benchmark"],
                "unit": val["unit"],
                "sector": val["sector"],
            }
            for key, val in EU_ETS_BENCHMARKS.items()
        ]

    # -------------------------------------------------------------------------
    # Compliance Cycle Tracking
    # -------------------------------------------------------------------------

    def create_compliance_cycle(
        self, installation_id: str, reporting_year: int,
    ) -> ComplianceCycle:
        """Create a new compliance cycle for tracking.

        Args:
            installation_id: Installation identifier.
            reporting_year: Year being reported on.

        Returns:
            ComplianceCycle record.
        """
        cycle = ComplianceCycle(
            installation_id=installation_id,
            reporting_year=reporting_year,
        )
        self._cycles.append(cycle)
        return cycle

    def update_cycle_status(
        self,
        cycle_id: str,
        status: ComplianceCycleStatus,
        date_field: Optional[date] = None,
    ) -> Optional[ComplianceCycle]:
        """Update compliance cycle status.

        Args:
            cycle_id: Compliance cycle identifier.
            status: New status.
            date_field: Associated date.

        Returns:
            Updated ComplianceCycle, or None if not found.
        """
        for cycle in self._cycles:
            if cycle.cycle_id == cycle_id:
                cycle.status = status
                if status == ComplianceCycleStatus.REPORT_SUBMITTED:
                    cycle.aer_submitted = True
                    cycle.aer_submission_date = date_field
                elif status == ComplianceCycleStatus.VERIFIED:
                    cycle.verification_complete = True
                    cycle.verification_date = date_field
                elif status == ComplianceCycleStatus.SURRENDERED:
                    cycle.allowances_surrendered = True
                    cycle.surrender_date = date_field
                return cycle
        return None

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Check EU ETS bridge health.

        Returns:
            Dict with health metrics.
        """
        return {
            "installations_registered": len(self._permits),
            "allocation_records": len(self._allocations),
            "emissions_records": len(self._emissions),
            "compliance_cycles": len(self._cycles),
            "benchmarks_available": len(EU_ETS_BENCHMARKS),
            "default_eua_price": self.config.default_eua_price_eur,
            "status": "healthy",
        }
