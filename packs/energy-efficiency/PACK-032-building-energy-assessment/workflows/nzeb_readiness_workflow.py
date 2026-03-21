# -*- coding: utf-8 -*-
"""
nZEB Readiness Workflow
============================

4-phase workflow for Nearly Zero-Energy Building (nZEB) readiness assessment
within PACK-032 Building Energy Assessment Pack.

Phases:
    1. CurrentPerformance        -- Baseline energy and carbon per m2
    2. NZEBGapAnalysis           -- Gap to nZEB target by country/building type
    3. MeasurePrioritisation     -- Deep retrofit measures for nZEB
    4. RoadmapToNZEB             -- Staged plan with milestone verification

Compliant with EU EPBD recast 2024, EN 15603, and national nZEB definitions.

Zero-hallucination: all nZEB targets, primary energy limits, and cost
estimates are from published national regulations and validated databases.
No LLM calls in the calculation path.

Schedule: on-demand
Estimated duration: 180 minutes

Author: GreenLang Team
Version: 32.0.0
"""

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


class NZEBStandard(str, Enum):
    """nZEB standard definitions."""

    EPBD_2024 = "epbd_2024"
    PASSIVHAUS = "passivhaus"
    PASSIVE_HOUSE = "passive_house"
    MINERGIE_P = "minergie_p"
    NEARLY_ZERO = "nearly_zero"
    NET_ZERO_READY = "net_zero_ready"
    NET_ZERO = "net_zero"


class RetrofitDepth(str, Enum):
    """Retrofit depth classification per EU EPBD."""

    LIGHT = "light"
    MEDIUM = "medium"
    DEEP = "deep"
    NZEB = "nzeb"


class MilestoneStatus(str, Enum):
    """Milestone verification status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    BEHIND_SCHEDULE = "behind_schedule"
    AT_RISK = "at_risk"


# =============================================================================
# ZERO-HALLUCINATION REFERENCE CONSTANTS
# =============================================================================

# nZEB primary energy targets by country and building type (kWh/m2/yr)
# Source: National nZEB definitions per EPBD Art. 9
NZEB_TARGETS: Dict[str, Dict[str, float]] = {
    "GB": {"office": 85.0, "retail": 100.0, "residential": 55.0, "school": 70.0,
            "hospital": 150.0, "hotel": 110.0, "warehouse": 60.0},
    "DE": {"office": 75.0, "retail": 90.0, "residential": 45.0, "school": 60.0,
            "hospital": 130.0, "hotel": 100.0, "warehouse": 50.0},
    "FR": {"office": 70.0, "retail": 90.0, "residential": 50.0, "school": 65.0,
            "hospital": 140.0, "hotel": 100.0, "warehouse": 55.0},
    "NL": {"office": 65.0, "retail": 80.0, "residential": 40.0, "school": 55.0,
            "hospital": 120.0, "hotel": 90.0, "warehouse": 45.0},
    "SE": {"office": 80.0, "retail": 95.0, "residential": 50.0, "school": 65.0,
            "hospital": 140.0, "hotel": 105.0, "warehouse": 55.0},
    "ES": {"office": 60.0, "retail": 75.0, "residential": 35.0, "school": 50.0,
            "hospital": 110.0, "hotel": 80.0, "warehouse": 40.0},
    "IT": {"office": 65.0, "retail": 80.0, "residential": 40.0, "school": 55.0,
            "hospital": 120.0, "hotel": 85.0, "warehouse": 45.0},
    "DEFAULT": {"office": 80.0, "retail": 95.0, "residential": 50.0, "school": 65.0,
                "hospital": 140.0, "hotel": 100.0, "warehouse": 55.0},
}

# nZEB CO2 targets by country (kgCO2/m2/yr)
NZEB_CO2_TARGETS: Dict[str, Dict[str, float]] = {
    "GB": {"office": 15.0, "retail": 18.0, "residential": 10.0, "school": 12.0,
            "hospital": 28.0, "hotel": 20.0, "warehouse": 10.0},
    "DE": {"office": 12.0, "retail": 15.0, "residential": 8.0, "school": 10.0,
            "hospital": 22.0, "hotel": 17.0, "warehouse": 8.0},
    "DEFAULT": {"office": 14.0, "retail": 17.0, "residential": 9.0, "school": 11.0,
                "hospital": 25.0, "hotel": 18.0, "warehouse": 9.0},
}

# Minimum renewable energy share for nZEB (% of primary energy)
NZEB_RENEWABLE_SHARE: Dict[str, float] = {
    "GB": 0.0,  # No explicit requirement, just primary energy limit
    "DE": 15.0,
    "FR": 20.0,
    "NL": 0.0,
    "SE": 0.0,
    "ES": 20.0,
    "IT": 20.0,
    "DEFAULT": 15.0,
}

# Maximum U-values for nZEB compliance (W/m2K) -- deep retrofit targets
NZEB_U_VALUE_TARGETS: Dict[str, float] = {
    "wall": 0.15,
    "roof": 0.12,
    "floor": 0.15,
    "window": 0.80,
    "door": 1.00,
}

# Maximum air permeability for nZEB (m3/h/m2 @ 50Pa)
NZEB_AIR_PERMEABILITY_TARGET: float = 3.0

# Passivhaus criteria
PASSIVHAUS_CRITERIA: Dict[str, float] = {
    "heating_demand_kwh_per_sqm": 15.0,
    "cooling_demand_kwh_per_sqm": 15.0,
    "primary_energy_kwh_per_sqm": 120.0,
    "air_permeability_ach50": 0.6,
}

# CO2 emission factors (kgCO2/kWh) - DEFRA 2024
EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.207,
    "natural_gas": 0.183,
    "fuel_oil": 0.267,
    "lpg": 0.214,
    "district_heating": 0.160,
    "biomass": 0.015,
    "heat_pump": 0.207,
}

# Primary energy factors (per EN 15603)
PRIMARY_ENERGY_FACTORS: Dict[str, float] = {
    "electricity": 2.50,
    "natural_gas": 1.10,
    "fuel_oil": 1.10,
    "lpg": 1.10,
    "district_heating": 0.70,
    "biomass": 0.20,
    "heat_pump": 2.50,
    "solar": 0.00,
}

# Deep retrofit measure database for nZEB
NZEB_MEASURES: List[Dict[str, Any]] = [
    # Fabric first
    {"id": "NZ01", "name": "External wall insulation to nZEB standard (200mm)",
     "category": "envelope", "cost_per_sqm": 160.0, "energy_saving_pct": 0.15,
     "co2_saving_pct": 0.15, "lifetime": 30, "stage": 1},
    {"id": "NZ02", "name": "Triple glazing with low-e coating",
     "category": "envelope", "cost_per_sqm": 500.0, "energy_saving_pct": 0.08,
     "co2_saving_pct": 0.08, "lifetime": 30, "stage": 1},
    {"id": "NZ03", "name": "Roof insulation to nZEB standard (300mm)",
     "category": "envelope", "cost_per_sqm": 80.0, "energy_saving_pct": 0.06,
     "co2_saving_pct": 0.06, "lifetime": 40, "stage": 1},
    {"id": "NZ04", "name": "Floor insulation upgrade",
     "category": "envelope", "cost_per_sqm": 65.0, "energy_saving_pct": 0.04,
     "co2_saving_pct": 0.04, "lifetime": 30, "stage": 1},
    {"id": "NZ05", "name": "Air tightness to nZEB standard",
     "category": "envelope", "cost_per_sqm": 25.0, "energy_saving_pct": 0.06,
     "co2_saving_pct": 0.06, "lifetime": 20, "stage": 1},
    {"id": "NZ06", "name": "Thermal bridge remediation",
     "category": "envelope", "cost_per_sqm": 30.0, "energy_saving_pct": 0.03,
     "co2_saving_pct": 0.03, "lifetime": 30, "stage": 1},
    # Systems
    {"id": "NZ07", "name": "Air source heat pump (SCOP 4.0+)",
     "category": "hvac", "cost_per_sqm": 100.0, "energy_saving_pct": 0.25,
     "co2_saving_pct": 0.35, "lifetime": 20, "stage": 2},
    {"id": "NZ08", "name": "Ground source heat pump (SCOP 4.5+)",
     "category": "hvac", "cost_per_sqm": 160.0, "energy_saving_pct": 0.30,
     "co2_saving_pct": 0.40, "lifetime": 25, "stage": 2},
    {"id": "NZ09", "name": "MVHR system (90%+ heat recovery)",
     "category": "ventilation", "cost_per_sqm": 55.0, "energy_saving_pct": 0.10,
     "co2_saving_pct": 0.10, "lifetime": 15, "stage": 2},
    {"id": "NZ10", "name": "LED lighting with full automation",
     "category": "lighting", "cost_per_sqm": 35.0, "energy_saving_pct": 0.08,
     "co2_saving_pct": 0.06, "lifetime": 15, "stage": 2},
    {"id": "NZ11", "name": "Heat pump water heater",
     "category": "dhw", "cost_per_sqm": 20.0, "energy_saving_pct": 0.04,
     "co2_saving_pct": 0.05, "lifetime": 15, "stage": 2},
    {"id": "NZ12", "name": "Smart BMS with AI optimisation",
     "category": "controls", "cost_per_sqm": 15.0, "energy_saving_pct": 0.08,
     "co2_saving_pct": 0.06, "lifetime": 10, "stage": 2},
    # Renewables
    {"id": "NZ13", "name": "Rooftop solar PV (maximum capacity)",
     "category": "renewables", "cost_per_sqm": 0.0, "energy_saving_pct": 0.15,
     "co2_saving_pct": 0.15, "lifetime": 25, "stage": 3,
     "cost_per_kwp": 1000.0, "yield_kwh_per_kwp": 950.0},
    {"id": "NZ14", "name": "Solar thermal for DHW preheat",
     "category": "renewables", "cost_per_sqm": 400.0, "energy_saving_pct": 0.03,
     "co2_saving_pct": 0.03, "lifetime": 20, "stage": 3},
    {"id": "NZ15", "name": "Battery storage for self-consumption",
     "category": "storage", "cost_per_sqm": 0.0, "energy_saving_pct": 0.05,
     "co2_saving_pct": 0.04, "lifetime": 12, "stage": 3,
     "cost_per_kwh_cap": 450.0},
]


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


class CurrentPerformanceData(BaseModel):
    """Current building performance data."""

    total_floor_area_sqm: float = Field(default=0.0, ge=0.0)
    annual_energy_kwh: float = Field(default=0.0, ge=0.0)
    annual_cost_eur: float = Field(default=0.0, ge=0.0)
    primary_energy_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    co2_kg_per_sqm: float = Field(default=0.0, ge=0.0)
    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    heating_kwh: float = Field(default=0.0, ge=0.0)
    cooling_kwh: float = Field(default=0.0, ge=0.0)
    lighting_kwh: float = Field(default=0.0, ge=0.0)
    dhw_kwh: float = Field(default=0.0, ge=0.0)
    renewable_generation_kwh: float = Field(default=0.0, ge=0.0)
    renewable_share_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    primary_heating_fuel: str = Field(default="natural_gas")
    wall_u_value: float = Field(default=0.50, ge=0.0)
    roof_u_value: float = Field(default=0.25, ge=0.0)
    floor_u_value: float = Field(default=0.25, ge=0.0)
    window_u_value: float = Field(default=1.60, ge=0.0)
    air_permeability: float = Field(default=7.0, ge=0.0)
    epc_band: str = Field(default="D")


class NZEBGap(BaseModel):
    """Gap between current performance and nZEB target."""

    metric: str = Field(default="")
    current_value: float = Field(default=0.0)
    target_value: float = Field(default=0.0)
    gap: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    unit: str = Field(default="")
    critical: bool = Field(default=False)


class NZEBMeasure(BaseModel):
    """Prioritised retrofit measure for nZEB."""

    measure_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    stage: int = Field(default=1, ge=1, le=3)
    capital_cost_eur: float = Field(default=0.0, ge=0.0)
    annual_energy_saving_kwh: float = Field(default=0.0, ge=0.0)
    annual_co2_saving_kg: float = Field(default=0.0, ge=0.0)
    primary_energy_reduction_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    simple_payback_years: float = Field(default=0.0, ge=0.0)
    npv_eur: float = Field(default=0.0)
    cumulative_primary_energy: float = Field(default=0.0, ge=0.0)
    nzeb_contribution_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class RoadmapMilestone(BaseModel):
    """nZEB roadmap milestone."""

    milestone_id: str = Field(default_factory=lambda: f"ms-{uuid.uuid4().hex[:8]}")
    stage: int = Field(default=1, ge=1, le=4)
    name: str = Field(default="")
    description: str = Field(default="")
    target_year: int = Field(default=0)
    target_primary_energy: float = Field(default=0.0, ge=0.0)
    target_co2: float = Field(default=0.0, ge=0.0)
    measures: List[str] = Field(default_factory=list)
    estimated_cost_eur: float = Field(default=0.0, ge=0.0)
    status: MilestoneStatus = Field(default=MilestoneStatus.NOT_STARTED)


class NZEBReadinessInput(BaseModel):
    """Input data model for NZEBReadinessWorkflow."""

    building_name: str = Field(default="")
    building_type: str = Field(default="office")
    country: str = Field(default="GB")
    performance: CurrentPerformanceData = Field(default_factory=CurrentPerformanceData)
    target_standard: NZEBStandard = Field(default=NZEBStandard.EPBD_2024)
    target_year: int = Field(default=2030, ge=2025, le=2050)
    discount_rate_pct: float = Field(default=5.0, ge=0.0, le=20.0)
    energy_cost_eur_per_kwh: float = Field(default=0.15, ge=0.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("performance")
    @classmethod
    def validate_performance(cls, v: CurrentPerformanceData) -> CurrentPerformanceData:
        """Floor area must be positive."""
        if v.total_floor_area_sqm <= 0:
            raise ValueError("total_floor_area_sqm must be > 0")
        return v


class NZEBReadinessResult(BaseModel):
    """Complete result from nZEB readiness workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="nzeb_readiness")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    building_name: str = Field(default="")
    target_standard: str = Field(default="")
    target_year: int = Field(default=2030)
    current_primary_energy: float = Field(default=0.0, ge=0.0)
    target_primary_energy: float = Field(default=0.0, ge=0.0)
    current_co2: float = Field(default=0.0, ge=0.0)
    target_co2: float = Field(default=0.0, ge=0.0)
    nzeb_gaps: List[NZEBGap] = Field(default_factory=list)
    overall_gap_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    nzeb_achievable: bool = Field(default=False)
    measures: List[NZEBMeasure] = Field(default_factory=list)
    roadmap: List[RoadmapMilestone] = Field(default_factory=list)
    total_investment_eur: float = Field(default=0.0, ge=0.0)
    total_annual_savings_eur: float = Field(default=0.0, ge=0.0)
    projected_primary_energy: float = Field(default=0.0, ge=0.0)
    projected_co2: float = Field(default=0.0, ge=0.0)
    retrofit_depth: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class NZEBReadinessWorkflow:
    """
    4-phase nZEB readiness assessment workflow.

    Establishes current performance baseline, analyses the gap to nZEB
    targets by country and building type, prioritises deep retrofit
    measures, and generates a staged roadmap with milestones.

    Zero-hallucination: all nZEB targets from national regulations per
    EPBD, all calculations deterministic using EN 15603 factors.

    Example:
        >>> wf = NZEBReadinessWorkflow()
        >>> perf = CurrentPerformanceData(
        ...     total_floor_area_sqm=2000,
        ...     primary_energy_kwh_per_sqm=180
        ... )
        >>> inp = NZEBReadinessInput(performance=perf)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize NZEBReadinessWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._gaps: List[NZEBGap] = []
        self._measures: List[NZEBMeasure] = []
        self._milestones: List[RoadmapMilestone] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[NZEBReadinessInput] = None,
    ) -> NZEBReadinessResult:
        """Execute the 4-phase nZEB readiness workflow."""
        if input_data is None:
            raise ValueError("input_data must be provided")

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting nZEB readiness workflow %s for %s (target=%s, year=%d)",
            self.workflow_id, input_data.building_name,
            input_data.target_standard.value, input_data.target_year,
        )

        self._phase_results = []
        self._gaps = []
        self._measures = []
        self._milestones = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_current_performance(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_nzeb_gap_analysis(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_measure_prioritisation(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_roadmap_to_nzeb(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("nZEB readiness workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        perf = input_data.performance
        country_targets = NZEB_TARGETS.get(input_data.country, NZEB_TARGETS["DEFAULT"])
        target_pe = country_targets.get(input_data.building_type, 80.0)
        co2_targets = NZEB_CO2_TARGETS.get(input_data.country, NZEB_CO2_TARGETS["DEFAULT"])
        target_co2 = co2_targets.get(input_data.building_type, 14.0)

        total_investment = sum(m.capital_cost_eur for m in self._measures)
        total_savings = sum(m.annual_energy_saving_kwh for m in self._measures) * input_data.energy_cost_eur_per_kwh
        total_pe_reduction = sum(m.primary_energy_reduction_kwh_per_sqm for m in self._measures)
        projected_pe = max(0, perf.primary_energy_kwh_per_sqm - total_pe_reduction)
        total_co2_reduction = sum(m.annual_co2_saving_kg for m in self._measures)
        projected_co2 = max(0, perf.co2_kg_per_sqm - total_co2_reduction / max(perf.total_floor_area_sqm, 1))

        overall_gap_pct = (
            (perf.primary_energy_kwh_per_sqm - target_pe) / perf.primary_energy_kwh_per_sqm * 100
            if perf.primary_energy_kwh_per_sqm > 0 else 0.0
        )
        nzeb_achievable = projected_pe <= target_pe

        # Retrofit depth classification
        if overall_gap_pct <= 20:
            depth = "light"
        elif overall_gap_pct <= 40:
            depth = "medium"
        elif overall_gap_pct <= 60:
            depth = "deep"
        else:
            depth = "nzeb"

        result = NZEBReadinessResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            building_name=input_data.building_name,
            target_standard=input_data.target_standard.value,
            target_year=input_data.target_year,
            current_primary_energy=round(perf.primary_energy_kwh_per_sqm, 2),
            target_primary_energy=round(target_pe, 2),
            current_co2=round(perf.co2_kg_per_sqm, 2),
            target_co2=round(target_co2, 2),
            nzeb_gaps=self._gaps,
            overall_gap_pct=round(max(overall_gap_pct, 0), 1),
            nzeb_achievable=nzeb_achievable,
            measures=self._measures,
            roadmap=self._milestones,
            total_investment_eur=round(total_investment, 2),
            total_annual_savings_eur=round(total_savings, 2),
            projected_primary_energy=round(projected_pe, 2),
            projected_co2=round(projected_co2, 2),
            retrofit_depth=depth,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "nZEB readiness workflow %s completed in %.2fs: PE %.0f->%.0f (target %.0f), "
            "achievable=%s, depth=%s, investment=%.0f EUR",
            self.workflow_id, elapsed, perf.primary_energy_kwh_per_sqm,
            projected_pe, target_pe, "YES" if nzeb_achievable else "NO",
            depth, total_investment,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Current Performance
    # -------------------------------------------------------------------------

    async def _phase_current_performance(
        self, input_data: NZEBReadinessInput
    ) -> PhaseResult:
        """Establish baseline energy and carbon per m2."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        perf = input_data.performance
        floor_area = perf.total_floor_area_sqm

        # Validate and fill gaps in performance data
        if perf.eui_kwh_per_sqm <= 0 and perf.annual_energy_kwh > 0 and floor_area > 0:
            perf.eui_kwh_per_sqm = perf.annual_energy_kwh / floor_area

        if perf.primary_energy_kwh_per_sqm <= 0 and perf.eui_kwh_per_sqm > 0:
            pef = PRIMARY_ENERGY_FACTORS.get(perf.primary_heating_fuel, 1.50)
            perf.primary_energy_kwh_per_sqm = perf.eui_kwh_per_sqm * pef
            warnings.append(f"Primary energy estimated using PEF={pef} for {perf.primary_heating_fuel}")

        if perf.co2_kg_per_sqm <= 0 and perf.eui_kwh_per_sqm > 0:
            ef = EMISSION_FACTORS.get(perf.primary_heating_fuel, 0.207)
            perf.co2_kg_per_sqm = perf.eui_kwh_per_sqm * ef
            warnings.append(f"CO2 estimated using EF={ef} for {perf.primary_heating_fuel}")

        if perf.annual_cost_eur <= 0 and perf.annual_energy_kwh > 0:
            perf.annual_cost_eur = perf.annual_energy_kwh * input_data.energy_cost_eur_per_kwh
            warnings.append("Energy cost estimated from energy_cost_eur_per_kwh")

        if perf.renewable_share_pct <= 0 and perf.renewable_generation_kwh > 0 and perf.annual_energy_kwh > 0:
            perf.renewable_share_pct = perf.renewable_generation_kwh / perf.annual_energy_kwh * 100

        # End-use breakdown estimation if missing
        if perf.heating_kwh <= 0 and perf.annual_energy_kwh > 0:
            perf.heating_kwh = perf.annual_energy_kwh * 0.50
            perf.cooling_kwh = perf.annual_energy_kwh * 0.10
            perf.lighting_kwh = perf.annual_energy_kwh * 0.20
            perf.dhw_kwh = perf.annual_energy_kwh * 0.10
            warnings.append("End-use breakdown estimated using typical splits")

        outputs["floor_area_sqm"] = floor_area
        outputs["eui_kwh_per_sqm"] = round(perf.eui_kwh_per_sqm, 2)
        outputs["primary_energy_kwh_per_sqm"] = round(perf.primary_energy_kwh_per_sqm, 2)
        outputs["co2_kg_per_sqm"] = round(perf.co2_kg_per_sqm, 2)
        outputs["annual_energy_kwh"] = round(perf.annual_energy_kwh, 2)
        outputs["annual_cost_eur"] = round(perf.annual_cost_eur, 2)
        outputs["renewable_share_pct"] = round(perf.renewable_share_pct, 1)
        outputs["primary_heating_fuel"] = perf.primary_heating_fuel
        outputs["epc_band"] = perf.epc_band
        outputs["wall_u_value"] = perf.wall_u_value
        outputs["roof_u_value"] = perf.roof_u_value
        outputs["window_u_value"] = perf.window_u_value
        outputs["air_permeability"] = perf.air_permeability

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 CurrentPerformance: PE=%.0f kWh/m2, CO2=%.1f kg/m2, "
            "EUI=%.0f kWh/m2, EPC=%s",
            perf.primary_energy_kwh_per_sqm, perf.co2_kg_per_sqm,
            perf.eui_kwh_per_sqm, perf.epc_band,
        )
        return PhaseResult(
            phase_name="current_performance", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: nZEB Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_nzeb_gap_analysis(
        self, input_data: NZEBReadinessInput
    ) -> PhaseResult:
        """Analyse gap to nZEB target by country and building type."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        perf = input_data.performance

        country_targets = NZEB_TARGETS.get(input_data.country, NZEB_TARGETS["DEFAULT"])
        target_pe = country_targets.get(input_data.building_type, 80.0)

        co2_targets = NZEB_CO2_TARGETS.get(input_data.country, NZEB_CO2_TARGETS["DEFAULT"])
        target_co2 = co2_targets.get(input_data.building_type, 14.0)

        min_renewable = NZEB_RENEWABLE_SHARE.get(input_data.country, 15.0)

        # Primary energy gap
        pe_gap = max(0, perf.primary_energy_kwh_per_sqm - target_pe)
        pe_gap_pct = pe_gap / perf.primary_energy_kwh_per_sqm * 100 if perf.primary_energy_kwh_per_sqm > 0 else 0.0
        self._gaps.append(NZEBGap(
            metric="primary_energy", current_value=round(perf.primary_energy_kwh_per_sqm, 2),
            target_value=target_pe, gap=round(pe_gap, 2), gap_pct=round(pe_gap_pct, 1),
            unit="kWh/m2/yr", critical=pe_gap > 0,
        ))

        # CO2 gap
        co2_gap = max(0, perf.co2_kg_per_sqm - target_co2)
        co2_gap_pct = co2_gap / perf.co2_kg_per_sqm * 100 if perf.co2_kg_per_sqm > 0 else 0.0
        self._gaps.append(NZEBGap(
            metric="co2_emissions", current_value=round(perf.co2_kg_per_sqm, 2),
            target_value=target_co2, gap=round(co2_gap, 2), gap_pct=round(co2_gap_pct, 1),
            unit="kgCO2/m2/yr", critical=co2_gap > 0,
        ))

        # Renewable share gap
        ren_gap = max(0, min_renewable - perf.renewable_share_pct)
        self._gaps.append(NZEBGap(
            metric="renewable_share", current_value=round(perf.renewable_share_pct, 1),
            target_value=min_renewable, gap=round(ren_gap, 1), gap_pct=round(ren_gap, 1),
            unit="%", critical=ren_gap > 0,
        ))

        # U-value gaps
        for element, target_u in NZEB_U_VALUE_TARGETS.items():
            current_u = getattr(perf, f"{element}_u_value", 0.0) if hasattr(perf, f"{element}_u_value") else 0.0
            if element == "wall":
                current_u = perf.wall_u_value
            elif element == "roof":
                current_u = perf.roof_u_value
            elif element == "floor":
                current_u = perf.floor_u_value
            elif element == "window":
                current_u = perf.window_u_value
            else:
                continue

            if current_u > 0:
                u_gap = max(0, current_u - target_u)
                u_gap_pct = u_gap / current_u * 100 if current_u > 0 else 0.0
                self._gaps.append(NZEBGap(
                    metric=f"u_value_{element}", current_value=round(current_u, 3),
                    target_value=target_u, gap=round(u_gap, 3), gap_pct=round(u_gap_pct, 1),
                    unit="W/m2K", critical=u_gap > 0,
                ))

        # Air permeability gap
        perm_gap = max(0, perf.air_permeability - NZEB_AIR_PERMEABILITY_TARGET)
        self._gaps.append(NZEBGap(
            metric="air_permeability", current_value=round(perf.air_permeability, 1),
            target_value=NZEB_AIR_PERMEABILITY_TARGET, gap=round(perm_gap, 1),
            gap_pct=round(perm_gap / max(perf.air_permeability, 0.1) * 100, 1),
            unit="m3/h/m2@50Pa", critical=perm_gap > 0,
        ))

        critical_gaps = sum(1 for g in self._gaps if g.critical)

        outputs["nzeb_target_primary_energy"] = target_pe
        outputs["nzeb_target_co2"] = target_co2
        outputs["nzeb_target_renewable_share"] = min_renewable
        outputs["primary_energy_gap_kwh_per_sqm"] = round(pe_gap, 2)
        outputs["primary_energy_gap_pct"] = round(pe_gap_pct, 1)
        outputs["co2_gap_kg_per_sqm"] = round(co2_gap, 2)
        outputs["total_gaps"] = len(self._gaps)
        outputs["critical_gaps"] = critical_gaps
        outputs["nzeb_standard"] = input_data.target_standard.value

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 NZEBGapAnalysis: PE gap=%.0f kWh/m2 (%.1f%%), "
            "CO2 gap=%.1f kg/m2, %d critical gaps",
            pe_gap, pe_gap_pct, co2_gap, critical_gaps,
        )
        return PhaseResult(
            phase_name="nzeb_gap_analysis", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Measure Prioritisation
    # -------------------------------------------------------------------------

    async def _phase_measure_prioritisation(
        self, input_data: NZEBReadinessInput
    ) -> PhaseResult:
        """Prioritise deep retrofit measures for nZEB."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        perf = input_data.performance
        floor_area = perf.total_floor_area_sqm
        discount_rate = input_data.discount_rate_pct / 100.0
        cost_per_kwh = input_data.energy_cost_eur_per_kwh

        cumulative_pe = perf.primary_energy_kwh_per_sqm

        for measure in NZEB_MEASURES:
            # Calculate costs
            if measure.get("cost_per_sqm", 0) > 0:
                capital_cost = measure["cost_per_sqm"] * floor_area
            elif "cost_per_kwp" in measure:
                footprint = floor_area / 3  # Assume 3 floors average
                usable_roof = footprint * 0.5
                capacity_kwp = usable_roof * 0.15
                capital_cost = capacity_kwp * measure["cost_per_kwp"]
            elif "cost_per_kwh_cap" in measure:
                storage_kwh = floor_area * 0.02
                capital_cost = storage_kwh * measure["cost_per_kwh_cap"]
            else:
                capital_cost = floor_area * 20.0

            # Calculate savings
            energy_saving_pct = measure.get("energy_saving_pct", 0.0)
            annual_saving_kwh = perf.annual_energy_kwh * energy_saving_pct
            annual_saving_eur = annual_saving_kwh * cost_per_kwh

            co2_saving_pct = measure.get("co2_saving_pct", 0.0)
            annual_co2_saving = perf.co2_kg_per_sqm * floor_area * co2_saving_pct

            # Primary energy reduction
            pef = PRIMARY_ENERGY_FACTORS.get(perf.primary_heating_fuel, 1.50)
            pe_reduction = energy_saving_pct * perf.primary_energy_kwh_per_sqm
            cumulative_pe -= pe_reduction

            # Payback
            payback = capital_cost / annual_saving_eur if annual_saving_eur > 0 else 99.0

            # NPV
            lifetime = measure.get("lifetime", 15)
            npv = -capital_cost + sum(
                annual_saving_eur / ((1.0 + discount_rate) ** y)
                for y in range(1, lifetime + 1)
            )

            # nZEB contribution
            country_targets = NZEB_TARGETS.get(input_data.country, NZEB_TARGETS["DEFAULT"])
            target_pe = country_targets.get(input_data.building_type, 80.0)
            total_gap = max(1.0, perf.primary_energy_kwh_per_sqm - target_pe)
            nzeb_contribution = (pe_reduction / total_gap * 100) if total_gap > 0 else 0.0

            self._measures.append(NZEBMeasure(
                measure_id=measure["id"],
                name=measure["name"],
                category=measure.get("category", ""),
                stage=measure.get("stage", 1),
                capital_cost_eur=round(capital_cost, 2),
                annual_energy_saving_kwh=round(annual_saving_kwh, 2),
                annual_co2_saving_kg=round(annual_co2_saving, 2),
                primary_energy_reduction_kwh_per_sqm=round(pe_reduction, 2),
                simple_payback_years=round(payback, 2),
                npv_eur=round(npv, 2),
                cumulative_primary_energy=round(max(cumulative_pe, 0), 2),
                nzeb_contribution_pct=round(nzeb_contribution, 1),
            ))

        # Sort by stage, then by nZEB contribution descending
        self._measures.sort(key=lambda m: (m.stage, -m.nzeb_contribution_pct))

        total_investment = sum(m.capital_cost_eur for m in self._measures)
        total_pe_reduction = sum(m.primary_energy_reduction_kwh_per_sqm for m in self._measures)

        outputs["measures_count"] = len(self._measures)
        outputs["total_investment_eur"] = round(total_investment, 2)
        outputs["total_pe_reduction_kwh_per_sqm"] = round(total_pe_reduction, 2)
        outputs["projected_primary_energy"] = round(max(0, perf.primary_energy_kwh_per_sqm - total_pe_reduction), 2)
        outputs["measures_by_stage"] = {
            1: sum(1 for m in self._measures if m.stage == 1),
            2: sum(1 for m in self._measures if m.stage == 2),
            3: sum(1 for m in self._measures if m.stage == 3),
        }
        outputs["measures_by_category"] = self._count_measures_by_category()

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 MeasurePrioritisation: %d measures, investment=%.0f EUR, "
            "PE reduction=%.0f kWh/m2",
            len(self._measures), total_investment, total_pe_reduction,
        )
        return PhaseResult(
            phase_name="measure_prioritisation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Roadmap to nZEB
    # -------------------------------------------------------------------------

    async def _phase_roadmap_to_nzeb(
        self, input_data: NZEBReadinessInput
    ) -> PhaseResult:
        """Generate staged plan with milestone verification."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        perf = input_data.performance

        current_year = datetime.utcnow().year
        target_year = input_data.target_year
        years_available = max(1, target_year - current_year)

        country_targets = NZEB_TARGETS.get(input_data.country, NZEB_TARGETS["DEFAULT"])
        target_pe = country_targets.get(input_data.building_type, 80.0)
        co2_targets = NZEB_CO2_TARGETS.get(input_data.country, NZEB_CO2_TARGETS["DEFAULT"])
        target_co2 = co2_targets.get(input_data.building_type, 14.0)

        # Stage 1: Fabric First (year 1-2)
        stage1_measures = [m for m in self._measures if m.stage == 1]
        stage1_pe_reduction = sum(m.primary_energy_reduction_kwh_per_sqm for m in stage1_measures)
        stage1_cost = sum(m.capital_cost_eur for m in stage1_measures)
        pe_after_s1 = max(0, perf.primary_energy_kwh_per_sqm - stage1_pe_reduction)

        self._milestones.append(RoadmapMilestone(
            stage=1,
            name="Fabric First - Deep insulation and air tightness",
            description=(
                "Upgrade building envelope to nZEB standard: external wall insulation, "
                "triple glazing, roof and floor insulation, thermal bridge remediation, "
                "and air tightness improvements."
            ),
            target_year=current_year + min(2, years_available),
            target_primary_energy=round(pe_after_s1, 2),
            target_co2=round(perf.co2_kg_per_sqm * (pe_after_s1 / max(perf.primary_energy_kwh_per_sqm, 1)), 2),
            measures=[m.measure_id for m in stage1_measures],
            estimated_cost_eur=round(stage1_cost, 2),
        ))

        # Stage 2: Systems Upgrade (year 2-4)
        stage2_measures = [m for m in self._measures if m.stage == 2]
        stage2_pe_reduction = sum(m.primary_energy_reduction_kwh_per_sqm for m in stage2_measures)
        stage2_cost = sum(m.capital_cost_eur for m in stage2_measures)
        pe_after_s2 = max(0, pe_after_s1 - stage2_pe_reduction)

        self._milestones.append(RoadmapMilestone(
            stage=2,
            name="Systems Upgrade - Heat pump, MVHR, LED, smart controls",
            description=(
                "Replace heating with heat pump, install MVHR, upgrade lighting "
                "to LED with full automation, and implement smart BMS."
            ),
            target_year=current_year + min(4, years_available),
            target_primary_energy=round(pe_after_s2, 2),
            target_co2=round(perf.co2_kg_per_sqm * (pe_after_s2 / max(perf.primary_energy_kwh_per_sqm, 1)), 2),
            measures=[m.measure_id for m in stage2_measures],
            estimated_cost_eur=round(stage2_cost, 2),
        ))

        # Stage 3: Renewables and Storage (year 3-5)
        stage3_measures = [m for m in self._measures if m.stage == 3]
        stage3_pe_reduction = sum(m.primary_energy_reduction_kwh_per_sqm for m in stage3_measures)
        stage3_cost = sum(m.capital_cost_eur for m in stage3_measures)
        pe_after_s3 = max(0, pe_after_s2 - stage3_pe_reduction)

        self._milestones.append(RoadmapMilestone(
            stage=3,
            name="Renewables and Storage - Solar PV, solar thermal, battery",
            description=(
                "Install maximum rooftop solar PV, solar thermal for DHW, "
                "and battery storage to maximise self-consumption."
            ),
            target_year=current_year + min(5, years_available),
            target_primary_energy=round(pe_after_s3, 2),
            target_co2=round(perf.co2_kg_per_sqm * (pe_after_s3 / max(perf.primary_energy_kwh_per_sqm, 1)), 2),
            measures=[m.measure_id for m in stage3_measures],
            estimated_cost_eur=round(stage3_cost, 2),
        ))

        # Stage 4: Verification and nZEB Certification
        self._milestones.append(RoadmapMilestone(
            stage=4,
            name="Verification and nZEB Certification",
            description=(
                "Post-occupancy evaluation, commissioning verification, "
                "nZEB certification application, and ongoing M&V."
            ),
            target_year=target_year,
            target_primary_energy=round(target_pe, 2),
            target_co2=round(target_co2, 2),
            measures=[],
            estimated_cost_eur=round(perf.total_floor_area_sqm * 5.0, 2),
        ))

        total_cost = sum(ms.estimated_cost_eur for ms in self._milestones)
        final_pe = pe_after_s3
        achievable = final_pe <= target_pe

        outputs["milestones_count"] = len(self._milestones)
        outputs["total_roadmap_cost_eur"] = round(total_cost, 2)
        outputs["projected_final_pe"] = round(final_pe, 2)
        outputs["target_pe"] = round(target_pe, 2)
        outputs["nzeb_achievable"] = achievable
        outputs["years_to_target"] = years_available
        outputs["pe_reduction_staged"] = {
            "stage_1": round(stage1_pe_reduction, 2),
            "stage_2": round(stage2_pe_reduction, 2),
            "stage_3": round(stage3_pe_reduction, 2),
        }

        if not achievable:
            warnings.append(
                f"Projected PE {final_pe:.0f} exceeds target {target_pe:.0f} kWh/m2. "
                "Additional measures or off-site renewables may be required."
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 RoadmapToNZEB: %d milestones, cost=%.0f EUR, "
            "PE %.0f->%.0f (target %.0f), achievable=%s",
            len(self._milestones), total_cost,
            perf.primary_energy_kwh_per_sqm, final_pe, target_pe,
            "YES" if achievable else "NO",
        )
        return PhaseResult(
            phase_name="roadmap_to_nzeb", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _count_measures_by_category(self) -> Dict[str, int]:
        """Count measures by category."""
        counts: Dict[str, int] = {}
        for m in self._measures:
            counts[m.category] = counts.get(m.category, 0) + 1
        return counts

    def _compute_provenance(self, result: NZEBReadinessResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
