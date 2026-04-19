# -*- coding: utf-8 -*-
"""
Waste Heat Recovery Workflow
==================================

4-phase workflow for waste heat recovery assessment within
PACK-031 Industrial Energy Audit Pack.

Phases:
    1. HeatSourceIdentification  -- Inventory all waste heat sources with temps and flows
    2. PinchAnalysis             -- Composite curves, minimum utility calculation
    3. TechnologySelection       -- Match sources to technologies: economizers, heat pumps, ORC
    4. ROICalculation            -- Capital costs, operating savings, payback, NPV per option

The workflow follows GreenLang zero-hallucination principles: all
thermodynamic calculations, pinch analysis, and financial metrics use
deterministic formulas. No LLM calls in the numeric computation path.

Schedule: on-demand
Estimated duration: 180 minutes

Author: GreenLang Team
Version: 31.0.0
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


class HeatSourceType(str, Enum):
    """Classification of waste heat source."""

    FLUE_GAS = "flue_gas"
    EXHAUST_AIR = "exhaust_air"
    COOLING_WATER = "cooling_water"
    CONDENSATE = "condensate"
    PROCESS_STREAM = "process_stream"
    COMPRESSOR_HEAT = "compressor_heat"
    FURNACE_EXHAUST = "furnace_exhaust"
    DRYER_EXHAUST = "dryer_exhaust"
    ENGINE_COOLANT = "engine_coolant"
    RADIATION = "radiation"


class HeatSinkType(str, Enum):
    """Classification of heat demand (sink)."""

    SPACE_HEATING = "space_heating"
    PROCESS_HEATING = "process_heating"
    BOILER_FEEDWATER = "boiler_feedwater"
    DOMESTIC_HOT_WATER = "domestic_hot_water"
    DRYING = "drying"
    PREHEATING = "preheating"
    ABSORPTION_COOLING = "absorption_cooling"
    DISTRICT_HEATING = "district_heating"
    POWER_GENERATION = "power_generation"


class RecoveryTechnologyType(str, Enum):
    """Waste heat recovery technology."""

    ECONOMIZER = "economizer"
    HEAT_EXCHANGER = "heat_exchanger"
    HEAT_PUMP = "heat_pump"
    ORC = "orc"  # Organic Rankine Cycle
    THERMOELECTRIC = "thermoelectric"
    ABSORPTION_CHILLER = "absorption_chiller"
    RECUPERATOR = "recuperator"
    REGENERATOR = "regenerator"
    HEAT_PIPE = "heat_pipe"
    RUN_AROUND_COIL = "run_around_coil"


class TemperatureGrade(str, Enum):
    """Waste heat temperature grade classification."""

    HIGH = "high"        # > 400C
    MEDIUM_HIGH = "medium_high"  # 200-400C
    MEDIUM = "medium"    # 100-200C
    LOW = "low"          # 50-100C
    VERY_LOW = "very_low"  # < 50C


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


class HeatSource(BaseModel):
    """Waste heat source record."""

    source_id: str = Field(default_factory=lambda: f"hs-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="Source name")
    source_type: HeatSourceType = Field(default=HeatSourceType.FLUE_GAS)
    temperature_c: float = Field(default=0.0, description="Source temperature C")
    min_exit_temp_c: float = Field(default=0.0, ge=0.0, description="Minimum exit temp C (acid dew point)")
    flow_rate_kg_hr: float = Field(default=0.0, ge=0.0, description="Mass flow rate kg/hr")
    flow_rate_m3_hr: float = Field(default=0.0, ge=0.0, description="Volume flow rate m3/hr")
    specific_heat_kj_kgk: float = Field(default=1.0, ge=0.0, description="Specific heat kJ/(kg.K)")
    available_hours_per_year: float = Field(default=6000.0, ge=0.0, description="Operating hours/yr")
    medium: str = Field(default="air", description="air|water|gas|oil|steam")
    equipment_source: str = Field(default="", description="Originating equipment")
    is_corrosive: bool = Field(default=False, description="Corrosive species present")
    is_intermittent: bool = Field(default=False, description="Intermittent availability")


class HeatSink(BaseModel):
    """Heat demand (sink) record."""

    sink_id: str = Field(default_factory=lambda: f"hk-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="Sink name")
    sink_type: HeatSinkType = Field(default=HeatSinkType.PROCESS_HEATING)
    target_temperature_c: float = Field(default=0.0, ge=0.0, description="Required temp C")
    supply_temperature_c: float = Field(default=0.0, ge=0.0, description="Current supply temp C")
    flow_rate_kg_hr: float = Field(default=0.0, ge=0.0, description="Mass flow rate kg/hr")
    specific_heat_kj_kgk: float = Field(default=4.186, ge=0.0, description="Specific heat")
    demand_kw: float = Field(default=0.0, ge=0.0, description="Heat demand in kW")
    demand_hours_per_year: float = Field(default=6000.0, ge=0.0)
    current_energy_source: str = Field(default="natural_gas", description="Current fuel")
    current_cost_eur_per_kwh: float = Field(default=0.04, ge=0.0)


class PinchAnalysisResult(BaseModel):
    """Result from pinch analysis calculations."""

    pinch_temperature_c: float = Field(default=0.0, description="Pinch temperature C")
    minimum_hot_utility_kw: float = Field(default=0.0, ge=0.0, description="Min hot utility")
    minimum_cold_utility_kw: float = Field(default=0.0, ge=0.0, description="Min cold utility")
    maximum_recovery_kw: float = Field(default=0.0, ge=0.0, description="Max heat recovery potential")
    total_hot_available_kw: float = Field(default=0.0, ge=0.0, description="Total hot stream capacity")
    total_cold_required_kw: float = Field(default=0.0, ge=0.0, description="Total cold stream demand")
    delta_t_min_c: float = Field(default=10.0, ge=0.0, description="Minimum approach temp C")
    heat_exchange_area_m2: float = Field(default=0.0, ge=0.0, description="Estimated HX area")
    energy_saving_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class RecoveryTechnology(BaseModel):
    """Matched recovery technology for a source-sink pair."""

    match_id: str = Field(default_factory=lambda: f"mtc-{uuid.uuid4().hex[:8]}")
    source_id: str = Field(default="")
    sink_id: str = Field(default="")
    technology: RecoveryTechnologyType = Field(default=RecoveryTechnologyType.HEAT_EXCHANGER)
    recoverable_kw: float = Field(default=0.0, ge=0.0, description="Recoverable heat kW")
    temperature_grade: TemperatureGrade = Field(default=TemperatureGrade.MEDIUM)
    effectiveness_pct: float = Field(default=70.0, ge=0.0, le=100.0, description="Heat recovery %")
    cop: float = Field(default=0.0, ge=0.0, description="COP for heat pumps")
    electrical_output_kw: float = Field(default=0.0, ge=0.0, description="ORC electrical output")
    feasibility_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Technical feasibility")
    notes: str = Field(default="")


class ROICalculation(BaseModel):
    """ROI and financial analysis for a recovery option."""

    option_id: str = Field(default_factory=lambda: f"roi-{uuid.uuid4().hex[:8]}")
    match_id: str = Field(default="")
    technology: str = Field(default="")
    capital_cost_eur: float = Field(default=0.0, ge=0.0)
    annual_operating_cost_eur: float = Field(default=0.0, ge=0.0)
    annual_savings_eur: float = Field(default=0.0, ge=0.0)
    annual_savings_kwh: float = Field(default=0.0, ge=0.0)
    simple_payback_years: float = Field(default=0.0, ge=0.0)
    npv_eur: float = Field(default=0.0, description="Net present value")
    irr_pct: float = Field(default=0.0, description="Internal rate of return %")
    co2_reduction_tonnes: float = Field(default=0.0, ge=0.0)
    lifetime_years: int = Field(default=15, ge=1)


class WasteHeatRecoveryInput(BaseModel):
    """Input data model for WasteHeatRecoveryWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    heat_sources: List[HeatSource] = Field(default_factory=list)
    heat_sinks: List[HeatSink] = Field(default_factory=list)
    delta_t_min_c: float = Field(default=10.0, ge=1.0, le=50.0, description="Min approach temp")
    discount_rate_pct: float = Field(default=8.0, ge=0.0, le=30.0)
    project_lifetime_years: int = Field(default=15, ge=1, le=30)
    electricity_cost_eur_per_kwh: float = Field(default=0.12, ge=0.0)
    electricity_ef_kgco2_kwh: float = Field(default=0.385, ge=0.0)
    fuel_ef_kgco2_kwh: float = Field(default=0.184, ge=0.0)
    operating_hours_per_year: float = Field(default=6000.0, ge=0.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class WasteHeatRecoveryResult(BaseModel):
    """Complete result from waste heat recovery workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="waste_heat_recovery")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    facility_id: str = Field(default="")
    pinch_analysis: PinchAnalysisResult = Field(default_factory=PinchAnalysisResult)
    technology_matches: List[RecoveryTechnology] = Field(default_factory=list)
    roi_calculations: List[ROICalculation] = Field(default_factory=list)
    total_recoverable_kw: float = Field(default=0.0, ge=0.0)
    total_annual_savings_eur: float = Field(default=0.0, ge=0.0)
    total_annual_savings_kwh: float = Field(default=0.0, ge=0.0)
    total_co2_reduction_tonnes: float = Field(default=0.0, ge=0.0)
    total_capital_cost_eur: float = Field(default=0.0, ge=0.0)
    portfolio_npv_eur: float = Field(default=0.0)
    portfolio_payback_years: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# TECHNOLOGY CONSTANTS (Zero-Hallucination)
# =============================================================================

# Capital cost estimates (EUR/kW of recovery) by technology
CAPITAL_COST_EUR_PER_KW: Dict[str, float] = {
    "economizer": 80.0,
    "heat_exchanger": 120.0,
    "heat_pump": 500.0,
    "orc": 2500.0,
    "thermoelectric": 3000.0,
    "absorption_chiller": 600.0,
    "recuperator": 150.0,
    "regenerator": 200.0,
    "heat_pipe": 250.0,
    "run_around_coil": 100.0,
}

# Typical effectiveness by technology (%)
TECHNOLOGY_EFFECTIVENESS: Dict[str, float] = {
    "economizer": 75.0,
    "heat_exchanger": 70.0,
    "heat_pump": 85.0,  # Based on COP uplift
    "orc": 12.0,       # Thermal-to-electric efficiency
    "thermoelectric": 5.0,
    "absorption_chiller": 65.0,
    "recuperator": 70.0,
    "regenerator": 80.0,
    "heat_pipe": 65.0,
    "run_around_coil": 55.0,
}

# Temperature ranges for technology applicability (min_source_temp_C, max_source_temp_C)
TECHNOLOGY_TEMP_RANGE: Dict[str, Tuple[float, float]] = {
    "economizer": (100.0, 500.0),
    "heat_exchanger": (40.0, 600.0),
    "heat_pump": (20.0, 100.0),
    "orc": (80.0, 400.0),
    "thermoelectric": (150.0, 600.0),
    "absorption_chiller": (80.0, 250.0),
    "recuperator": (200.0, 1000.0),
    "regenerator": (400.0, 1500.0),
    "heat_pipe": (50.0, 300.0),
    "run_around_coil": (30.0, 200.0),
}

# Annual maintenance as % of capital
ANNUAL_MAINTENANCE_PCT: Dict[str, float] = {
    "economizer": 2.0, "heat_exchanger": 2.5, "heat_pump": 4.0,
    "orc": 3.0, "thermoelectric": 1.5, "absorption_chiller": 3.0,
    "recuperator": 2.0, "regenerator": 3.0, "heat_pipe": 1.0,
    "run_around_coil": 2.5,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class WasteHeatRecoveryWorkflow:
    """
    4-phase waste heat recovery assessment workflow.

    Identifies waste heat sources, performs pinch analysis, matches
    sources to recovery technologies, and calculates ROI with NPV.

    Zero-hallucination: all thermodynamic and financial calculations
    use deterministic formulas. No LLM calls in numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _pinch_result: Pinch analysis output.
        _tech_matches: Technology matches for source-sink pairs.
        _roi_calcs: ROI calculations per option.

    Example:
        >>> wf = WasteHeatRecoveryWorkflow()
        >>> inp = WasteHeatRecoveryInput(heat_sources=[...], heat_sinks=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize WasteHeatRecoveryWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._pinch_result: PinchAnalysisResult = PinchAnalysisResult()
        self._tech_matches: List[RecoveryTechnology] = []
        self._roi_calcs: List[ROICalculation] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[WasteHeatRecoveryInput] = None,
        heat_sources: Optional[List[HeatSource]] = None,
        heat_sinks: Optional[List[HeatSink]] = None,
    ) -> WasteHeatRecoveryResult:
        """
        Execute the 4-phase waste heat recovery workflow.

        Args:
            input_data: Full input model (preferred).
            heat_sources: Heat source list (fallback).
            heat_sinks: Heat sink list (fallback).

        Returns:
            WasteHeatRecoveryResult with pinch analysis, matches, and ROI.
        """
        if input_data is None:
            input_data = WasteHeatRecoveryInput(
                heat_sources=heat_sources or [],
                heat_sinks=heat_sinks or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting waste heat recovery workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._tech_matches = []
        self._roi_calcs = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_heat_source_identification(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_pinch_analysis(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_technology_selection(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_roi_calculation(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Waste heat recovery workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_recoverable = sum(m.recoverable_kw for m in self._tech_matches)
        total_savings_eur = sum(r.annual_savings_eur for r in self._roi_calcs)
        total_savings_kwh = sum(r.annual_savings_kwh for r in self._roi_calcs)
        total_co2 = sum(r.co2_reduction_tonnes for r in self._roi_calcs)
        total_capex = sum(r.capital_cost_eur for r in self._roi_calcs)
        portfolio_npv = sum(r.npv_eur for r in self._roi_calcs)
        portfolio_payback = total_capex / total_savings_eur if total_savings_eur > 0 else 0.0

        result = WasteHeatRecoveryResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            facility_id=input_data.facility_id,
            pinch_analysis=self._pinch_result,
            technology_matches=self._tech_matches,
            roi_calculations=self._roi_calcs,
            total_recoverable_kw=round(total_recoverable, 2),
            total_annual_savings_eur=round(total_savings_eur, 2),
            total_annual_savings_kwh=round(total_savings_kwh, 2),
            total_co2_reduction_tonnes=round(total_co2, 4),
            total_capital_cost_eur=round(total_capex, 2),
            portfolio_npv_eur=round(portfolio_npv, 2),
            portfolio_payback_years=round(portfolio_payback, 2),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Waste heat recovery workflow %s completed in %.2fs recovery=%.0f kW",
            self.workflow_id, elapsed, total_recoverable,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Heat Source Identification
    # -------------------------------------------------------------------------

    async def _phase_heat_source_identification(
        self, input_data: WasteHeatRecoveryInput
    ) -> PhaseResult:
        """Inventory all waste heat sources with temperatures and flows."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        source_summary: List[Dict[str, Any]] = []
        total_available_kw = 0.0
        grade_counts: Dict[str, int] = {}

        for source in input_data.heat_sources:
            # Calculate available heat (kW)
            exit_temp = source.min_exit_temp_c if source.min_exit_temp_c > 0 else 40.0
            delta_t = source.temperature_c - exit_temp
            if delta_t <= 0:
                warnings.append(f"Source {source.source_id}: temp {source.temperature_c}C <= exit temp {exit_temp}C")
                continue

            available_kw = (source.flow_rate_kg_hr * source.specific_heat_kj_kgk * delta_t) / 3600.0
            total_available_kw += available_kw

            grade = self._classify_temperature_grade(source.temperature_c)
            grade_counts[grade.value] = grade_counts.get(grade.value, 0) + 1

            source_summary.append({
                "source_id": source.source_id,
                "name": source.name,
                "temperature_c": source.temperature_c,
                "available_kw": round(available_kw, 2),
                "grade": grade.value,
                "type": source.source_type.value,
            })

        sink_demand_kw = sum(s.demand_kw for s in input_data.heat_sinks)

        outputs["sources_count"] = len(input_data.heat_sources)
        outputs["sinks_count"] = len(input_data.heat_sinks)
        outputs["total_available_kw"] = round(total_available_kw, 2)
        outputs["total_sink_demand_kw"] = round(sink_demand_kw, 2)
        outputs["grade_distribution"] = grade_counts
        outputs["source_summary"] = source_summary

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 HeatSourceIdentification: %d sources, %.0f kW available",
            len(input_data.heat_sources), total_available_kw,
        )
        return PhaseResult(
            phase_name="heat_source_identification", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _classify_temperature_grade(self, temp_c: float) -> TemperatureGrade:
        """Classify waste heat by temperature grade."""
        if temp_c > 400.0:
            return TemperatureGrade.HIGH
        elif temp_c > 200.0:
            return TemperatureGrade.MEDIUM_HIGH
        elif temp_c > 100.0:
            return TemperatureGrade.MEDIUM
        elif temp_c > 50.0:
            return TemperatureGrade.LOW
        else:
            return TemperatureGrade.VERY_LOW

    # -------------------------------------------------------------------------
    # Phase 2: Pinch Analysis
    # -------------------------------------------------------------------------

    async def _phase_pinch_analysis(
        self, input_data: WasteHeatRecoveryInput
    ) -> PhaseResult:
        """Perform pinch analysis: composite curves and minimum utility."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        dt_min = input_data.delta_t_min_c

        # Build hot composite (sources)
        hot_streams: List[Dict[str, float]] = []
        for source in input_data.heat_sources:
            exit_temp = source.min_exit_temp_c if source.min_exit_temp_c > 0 else 40.0
            if source.temperature_c > exit_temp:
                cp_flow = source.flow_rate_kg_hr * source.specific_heat_kj_kgk / 3600.0  # kW/K
                heat_kw = cp_flow * (source.temperature_c - exit_temp)
                hot_streams.append({
                    "t_start": source.temperature_c,
                    "t_end": exit_temp,
                    "cp_flow_kw_k": cp_flow,
                    "heat_kw": heat_kw,
                })

        # Build cold composite (sinks)
        cold_streams: List[Dict[str, float]] = []
        for sink in input_data.heat_sinks:
            if sink.target_temperature_c > sink.supply_temperature_c:
                if sink.demand_kw > 0:
                    cp_flow = sink.demand_kw / (sink.target_temperature_c - sink.supply_temperature_c)
                elif sink.flow_rate_kg_hr > 0:
                    cp_flow = sink.flow_rate_kg_hr * sink.specific_heat_kj_kgk / 3600.0
                else:
                    continue
                heat_kw = cp_flow * (sink.target_temperature_c - sink.supply_temperature_c)
                cold_streams.append({
                    "t_start": sink.supply_temperature_c,
                    "t_end": sink.target_temperature_c,
                    "cp_flow_kw_k": cp_flow,
                    "heat_kw": heat_kw,
                })

        total_hot = sum(h["heat_kw"] for h in hot_streams)
        total_cold = sum(c["heat_kw"] for c in cold_streams)

        # Simplified pinch calculation using problem table algorithm
        pinch_temp, min_hot_utility, min_cold_utility, max_recovery = self._problem_table_algorithm(
            hot_streams, cold_streams, dt_min
        )

        energy_saving_pct = (max_recovery / max(total_cold, 1)) * 100.0

        # Estimate heat exchange area: Q = U * A * LMTD
        u_value = 50.0  # W/(m2.K) typical for gas-liquid
        lmtd = dt_min * 1.5  # Simplified LMTD estimate
        hx_area = (max_recovery * 1000.0) / (u_value * max(lmtd, 1.0)) if max_recovery > 0 else 0.0

        self._pinch_result = PinchAnalysisResult(
            pinch_temperature_c=round(pinch_temp, 1),
            minimum_hot_utility_kw=round(min_hot_utility, 2),
            minimum_cold_utility_kw=round(min_cold_utility, 2),
            maximum_recovery_kw=round(max_recovery, 2),
            total_hot_available_kw=round(total_hot, 2),
            total_cold_required_kw=round(total_cold, 2),
            delta_t_min_c=dt_min,
            heat_exchange_area_m2=round(hx_area, 2),
            energy_saving_pct=round(min(energy_saving_pct, 100.0), 1),
        )

        outputs["pinch_temp_c"] = round(pinch_temp, 1)
        outputs["max_recovery_kw"] = round(max_recovery, 2)
        outputs["min_hot_utility_kw"] = round(min_hot_utility, 2)
        outputs["min_cold_utility_kw"] = round(min_cold_utility, 2)
        outputs["energy_saving_pct"] = round(min(energy_saving_pct, 100.0), 1)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 PinchAnalysis: pinch=%.1fC max_recovery=%.0f kW saving=%.1f%%",
            pinch_temp, max_recovery, energy_saving_pct,
        )
        return PhaseResult(
            phase_name="pinch_analysis", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _problem_table_algorithm(
        self,
        hot_streams: List[Dict[str, float]],
        cold_streams: List[Dict[str, float]],
        dt_min: float,
    ) -> Tuple[float, float, float, float]:
        """Simplified problem table algorithm for pinch point determination."""
        if not hot_streams or not cold_streams:
            total_hot = sum(h["heat_kw"] for h in hot_streams)
            total_cold = sum(c["heat_kw"] for c in cold_streams)
            return 0.0, total_cold, total_hot, 0.0

        # Collect all temperature intervals
        temps = set()
        for h in hot_streams:
            temps.add(h["t_start"])
            temps.add(h["t_end"])
        for c in cold_streams:
            temps.add(c["t_start"] + dt_min)
            temps.add(c["t_end"] + dt_min)

        sorted_temps = sorted(temps, reverse=True)

        # Calculate heat surplus/deficit in each interval
        cascaded_heat = 0.0
        min_cascade = 0.0
        pinch_temp = sorted_temps[-1] if sorted_temps else 0.0

        for i in range(len(sorted_temps) - 1):
            t_high = sorted_temps[i]
            t_low = sorted_temps[i + 1]
            dt = t_high - t_low

            # Hot stream contribution in this interval
            hot_cp = sum(
                h["cp_flow_kw_k"] for h in hot_streams
                if h["t_start"] >= t_high and h["t_end"] <= t_low
            )
            # Cold stream contribution
            cold_cp = sum(
                c["cp_flow_kw_k"] for c in cold_streams
                if (c["t_start"] + dt_min) <= t_high and (c["t_end"] + dt_min) >= t_low
            )

            interval_surplus = (hot_cp - cold_cp) * dt
            cascaded_heat += interval_surplus

            if cascaded_heat < min_cascade:
                min_cascade = cascaded_heat
                pinch_temp = t_low

        total_hot = sum(h["heat_kw"] for h in hot_streams)
        total_cold = sum(c["heat_kw"] for c in cold_streams)
        min_hot_utility = max(0.0, total_cold - total_hot + abs(min_cascade))
        min_cold_utility = max(0.0, total_hot - total_cold + abs(min_cascade))
        max_recovery = total_hot - min_cold_utility

        return pinch_temp, min_hot_utility, min_cold_utility, max(max_recovery, 0.0)

    # -------------------------------------------------------------------------
    # Phase 3: Technology Selection
    # -------------------------------------------------------------------------

    async def _phase_technology_selection(
        self, input_data: WasteHeatRecoveryInput
    ) -> PhaseResult:
        """Match heat sources to recovery technologies."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for source in input_data.heat_sources:
            best_matches = self._match_technologies(source, input_data)
            self._tech_matches.extend(best_matches)

        # Sort by recoverable kW descending
        self._tech_matches.sort(key=lambda m: m.recoverable_kw, reverse=True)

        outputs["matches_found"] = len(self._tech_matches)
        outputs["total_recoverable_kw"] = round(
            sum(m.recoverable_kw for m in self._tech_matches), 2
        )
        tech_types = {}
        for m in self._tech_matches:
            t = m.technology.value
            tech_types[t] = tech_types.get(t, 0) + 1
        outputs["technology_distribution"] = tech_types

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 TechnologySelection: %d matches, %.0f kW recoverable",
            len(self._tech_matches), outputs["total_recoverable_kw"],
        )
        return PhaseResult(
            phase_name="technology_selection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _match_technologies(
        self, source: HeatSource, input_data: WasteHeatRecoveryInput
    ) -> List[RecoveryTechnology]:
        """Find applicable technologies for a heat source."""
        matches: List[RecoveryTechnology] = []
        exit_temp = source.min_exit_temp_c if source.min_exit_temp_c > 0 else 40.0
        delta_t = source.temperature_c - exit_temp
        if delta_t <= 0:
            return matches

        available_kw = (source.flow_rate_kg_hr * source.specific_heat_kj_kgk * delta_t) / 3600.0
        grade = self._classify_temperature_grade(source.temperature_c)

        for tech_name, (min_t, max_t) in TECHNOLOGY_TEMP_RANGE.items():
            if min_t <= source.temperature_c <= max_t:
                effectiveness = TECHNOLOGY_EFFECTIVENESS.get(tech_name, 70.0) / 100.0
                tech_type = RecoveryTechnologyType(tech_name)

                # Calculate recoverable heat
                if tech_name == "orc":
                    recoverable = available_kw * effectiveness  # Electrical output
                    electrical_output = recoverable
                    recoverable_thermal = 0.0
                elif tech_name == "heat_pump":
                    cop = 3.5 if source.temperature_c < 60 else 2.5
                    recoverable = available_kw * cop / (cop - 1)  # Uplifted heat
                    electrical_output = 0.0
                else:
                    recoverable = available_kw * effectiveness
                    electrical_output = 0.0

                # Feasibility score
                feasibility = 80.0
                if source.is_corrosive:
                    feasibility -= 20.0
                if source.is_intermittent:
                    feasibility -= 10.0
                if grade == TemperatureGrade.HIGH and tech_name in ("heat_pump", "run_around_coil"):
                    feasibility -= 30.0

                # Find best matching sink
                sink_id = ""
                for sink in input_data.heat_sinks:
                    if source.temperature_c - input_data.delta_t_min_c >= sink.target_temperature_c:
                        sink_id = sink.sink_id
                        break

                matches.append(RecoveryTechnology(
                    source_id=source.source_id,
                    sink_id=sink_id,
                    technology=tech_type,
                    recoverable_kw=round(recoverable, 2),
                    temperature_grade=grade,
                    effectiveness_pct=round(effectiveness * 100.0, 1),
                    cop=cop if tech_name == "heat_pump" else 0.0,
                    electrical_output_kw=round(electrical_output, 2) if tech_name == "orc" else 0.0,
                    feasibility_score=round(max(feasibility, 0.0), 1),
                ))

        # Return top 2 most feasible options per source
        matches.sort(key=lambda m: m.feasibility_score, reverse=True)
        return matches[:2]

    # -------------------------------------------------------------------------
    # Phase 4: ROI Calculation
    # -------------------------------------------------------------------------

    async def _phase_roi_calculation(
        self, input_data: WasteHeatRecoveryInput
    ) -> PhaseResult:
        """Calculate capital costs, operating savings, payback, NPV per option."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        discount = input_data.discount_rate_pct / 100.0
        lifetime = input_data.project_lifetime_years

        for match in self._tech_matches:
            roi = self._calculate_roi(match, input_data, discount, lifetime)
            self._roi_calcs.append(roi)

        # Sort by NPV descending
        self._roi_calcs.sort(key=lambda r: r.npv_eur, reverse=True)

        outputs["options_evaluated"] = len(self._roi_calcs)
        outputs["positive_npv_count"] = sum(1 for r in self._roi_calcs if r.npv_eur > 0)
        outputs["total_capital_eur"] = round(sum(r.capital_cost_eur for r in self._roi_calcs), 2)
        outputs["total_annual_savings_eur"] = round(sum(r.annual_savings_eur for r in self._roi_calcs), 2)
        outputs["portfolio_npv_eur"] = round(sum(r.npv_eur for r in self._roi_calcs), 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ROICalculation: %d options, portfolio NPV=%.0f EUR",
            len(self._roi_calcs), outputs["portfolio_npv_eur"],
        )
        return PhaseResult(
            phase_name="roi_calculation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_roi(
        self,
        match: RecoveryTechnology,
        input_data: WasteHeatRecoveryInput,
        discount: float,
        lifetime: int,
    ) -> ROICalculation:
        """Calculate ROI for a single recovery technology match."""
        tech_name = match.technology.value
        capital = match.recoverable_kw * CAPITAL_COST_EUR_PER_KW.get(tech_name, 200.0)
        maint_pct = ANNUAL_MAINTENANCE_PCT.get(tech_name, 2.5) / 100.0
        annual_maint = capital * maint_pct

        # Annual energy savings
        if tech_name == "orc":
            savings_kwh = match.electrical_output_kw * input_data.operating_hours_per_year
            savings_eur = savings_kwh * input_data.electricity_cost_eur_per_kwh
            co2 = savings_kwh * input_data.electricity_ef_kgco2_kwh / 1000.0
        else:
            savings_kwh = match.recoverable_kw * input_data.operating_hours_per_year
            # Find sink cost rate
            sink = next(
                (s for s in input_data.heat_sinks if s.sink_id == match.sink_id), None
            )
            cost_per_kwh = sink.current_cost_eur_per_kwh if sink else 0.04
            savings_eur = savings_kwh * cost_per_kwh
            co2 = savings_kwh * input_data.fuel_ef_kgco2_kwh / 1000.0

        net_annual = savings_eur - annual_maint
        payback = capital / net_annual if net_annual > 0 else 99.0

        # NPV calculation
        npv = -capital
        for year in range(1, lifetime + 1):
            npv += net_annual / ((1.0 + discount) ** year)

        # IRR approximation
        irr = self._approximate_irr(capital, net_annual, lifetime)

        return ROICalculation(
            match_id=match.match_id,
            technology=tech_name,
            capital_cost_eur=round(capital, 2),
            annual_operating_cost_eur=round(annual_maint, 2),
            annual_savings_eur=round(net_annual, 2),
            annual_savings_kwh=round(savings_kwh, 2),
            simple_payback_years=round(payback, 2),
            npv_eur=round(npv, 2),
            irr_pct=round(irr, 2),
            co2_reduction_tonnes=round(co2, 4),
            lifetime_years=lifetime,
        )

    def _approximate_irr(
        self, investment: float, annual_cashflow: float, years: int
    ) -> float:
        """Approximate IRR using bisection method (zero-hallucination)."""
        if investment <= 0 or annual_cashflow <= 0:
            return 0.0
        low, high = 0.0, 5.0
        for _ in range(50):
            mid = (low + high) / 2.0
            npv = -investment + sum(
                annual_cashflow / ((1.0 + mid) ** y) for y in range(1, years + 1)
            )
            if npv > 0:
                low = mid
            else:
                high = mid
        return mid * 100.0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: WasteHeatRecoveryResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
