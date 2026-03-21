# -*- coding: utf-8 -*-
"""
Compressed Air Audit Workflow
===================================

4-phase workflow for compressed air system auditing within
PACK-031 Industrial Energy Audit Pack.

Phases:
    1. SystemMapping           -- Inventory compressors, dryers, receivers, distribution
    2. LeakSurvey              -- Ultrasonic detection, quantification, cost calculation
    3. PerformanceTesting      -- Specific power, FAD tests, load profiles
    4. OptimizationRecommendations -- VSD, pressure reduction, leak repair, receiver sizing

The workflow follows GreenLang zero-hallucination principles: all
calculations use deterministic engineering formulas per ISO 11011
and the Compressed Air & Gas Institute (CAGI) standards.

Schedule: annual
Estimated duration: 240 minutes

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

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


class CompressorType(str, Enum):
    """Compressor technology type."""

    ROTARY_SCREW = "rotary_screw"
    RECIPROCATING = "reciprocating"
    CENTRIFUGAL = "centrifugal"
    SCROLL = "scroll"
    ROTARY_VANE = "rotary_vane"


class CompressorControl(str, Enum):
    """Compressor capacity control method."""

    LOAD_UNLOAD = "load_unload"
    MODULATING = "modulating"
    VSD = "vsd"
    ON_OFF = "on_off"
    MULTI_STEP = "multi_step"


class DryerType(str, Enum):
    """Compressed air dryer type."""

    REFRIGERATED = "refrigerated"
    DESICCANT_HEATLESS = "desiccant_heatless"
    DESICCANT_HEATED = "desiccant_heated"
    MEMBRANE = "membrane"
    NONE = "none"


class LeakSeverity(str, Enum):
    """Leak severity classification."""

    CRITICAL = "critical"  # > 5 cfm
    MAJOR = "major"        # 2-5 cfm
    MINOR = "minor"        # 0.5-2 cfm
    TRACE = "trace"        # < 0.5 cfm


class RecommendationPriority(str, Enum):
    """Recommendation priority."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


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


class CompressorRecord(BaseModel):
    """Compressor inventory record."""

    compressor_id: str = Field(default_factory=lambda: f"cmp-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="Compressor name/tag")
    compressor_type: CompressorType = Field(default=CompressorType.ROTARY_SCREW)
    control_type: CompressorControl = Field(default=CompressorControl.LOAD_UNLOAD)
    manufacturer: str = Field(default="")
    model: str = Field(default="")
    year_installed: int = Field(default=0, ge=0)
    rated_power_kw: float = Field(default=0.0, ge=0.0, description="Motor nameplate kW")
    rated_flow_cfm: float = Field(default=0.0, ge=0.0, description="Rated FAD in cfm")
    rated_flow_m3_min: float = Field(default=0.0, ge=0.0, description="Rated FAD in m3/min")
    rated_pressure_bar: float = Field(default=7.0, ge=0.0, description="Rated discharge pressure")
    actual_pressure_bar: float = Field(default=7.0, ge=0.0, description="Actual discharge pressure")
    operating_hours_per_year: float = Field(default=0.0, ge=0.0)
    loaded_hours_per_year: float = Field(default=0.0, ge=0.0)
    unloaded_hours_per_year: float = Field(default=0.0, ge=0.0)
    measured_power_loaded_kw: float = Field(default=0.0, ge=0.0, description="Measured at load")
    measured_power_unloaded_kw: float = Field(default=0.0, ge=0.0, description="Measured unloaded")
    is_vsd: bool = Field(default=False, description="Has variable speed drive")


class DryerRecord(BaseModel):
    """Compressed air dryer record."""

    dryer_id: str = Field(default_factory=lambda: f"dry-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="")
    dryer_type: DryerType = Field(default=DryerType.REFRIGERATED)
    rated_flow_cfm: float = Field(default=0.0, ge=0.0)
    power_consumption_kw: float = Field(default=0.0, ge=0.0)
    purge_air_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Desiccant purge %")
    dewpoint_c: float = Field(default=3.0, description="Dewpoint in Celsius")


class ReceiverRecord(BaseModel):
    """Air receiver (storage tank) record."""

    receiver_id: str = Field(default_factory=lambda: f"rcv-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="")
    volume_litres: float = Field(default=0.0, ge=0.0, description="Tank volume in litres")
    location: str = Field(default="", description="primary|secondary|point_of_use")
    pressure_bar: float = Field(default=7.0, ge=0.0)


class LeakRecord(BaseModel):
    """Individual leak detection record."""

    leak_id: str = Field(default_factory=lambda: f"lk-{uuid.uuid4().hex[:8]}")
    location: str = Field(default="", description="Leak location description")
    equipment_tag: str = Field(default="", description="Associated equipment")
    severity: LeakSeverity = Field(default=LeakSeverity.MINOR)
    estimated_flow_cfm: float = Field(default=0.0, ge=0.0, description="Leak flow in cfm")
    estimated_flow_l_min: float = Field(default=0.0, ge=0.0, description="Leak flow l/min")
    db_level: float = Field(default=0.0, ge=0.0, description="Ultrasonic dB reading")
    component_type: str = Field(default="", description="coupling|valve|hose|fitting|pipe|frl")
    is_accessible: bool = Field(default=True, description="Can be repaired without shutdown")
    repair_cost_eur: float = Field(default=0.0, ge=0.0)


class CompressorPerformance(BaseModel):
    """Performance test result for a compressor."""

    compressor_id: str = Field(default="")
    specific_power_kw_per_m3_min: float = Field(default=0.0, ge=0.0, description="kW/(m3/min)")
    specific_power_kw_per_100cfm: float = Field(default=0.0, ge=0.0, description="kW/100cfm")
    fad_m3_min: float = Field(default=0.0, ge=0.0, description="Free Air Delivery m3/min")
    fad_cfm: float = Field(default=0.0, ge=0.0, description="Free Air Delivery cfm")
    load_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Average load %")
    efficiency_rating: str = Field(default="", description="excellent|good|average|poor")
    annual_energy_kwh: float = Field(default=0.0, ge=0.0, description="Annual energy kWh")
    annual_cost_eur: float = Field(default=0.0, ge=0.0)


class CompressedAirRecommendation(BaseModel):
    """Optimization recommendation for compressed air system."""

    recommendation_id: str = Field(default_factory=lambda: f"rec-{uuid.uuid4().hex[:8]}")
    title: str = Field(default="")
    description: str = Field(default="")
    category: str = Field(default="", description="leak_repair|pressure|vsd|receiver|controls|dryer")
    priority: RecommendationPriority = Field(default=RecommendationPriority.SHORT_TERM)
    annual_savings_kwh: float = Field(default=0.0, ge=0.0)
    annual_savings_eur: float = Field(default=0.0, ge=0.0)
    implementation_cost_eur: float = Field(default=0.0, ge=0.0)
    simple_payback_years: float = Field(default=0.0, ge=0.0)
    co2_reduction_tonnes: float = Field(default=0.0, ge=0.0)


class CompressedAirAuditInput(BaseModel):
    """Input data model for CompressedAirAuditWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    compressors: List[CompressorRecord] = Field(default_factory=list)
    dryers: List[DryerRecord] = Field(default_factory=list)
    receivers: List[ReceiverRecord] = Field(default_factory=list)
    leaks: List[LeakRecord] = Field(default_factory=list)
    system_pressure_bar: float = Field(default=7.0, ge=0.0, description="Target system pressure")
    total_demand_cfm: float = Field(default=0.0, ge=0.0, description="Measured total demand")
    total_demand_m3_min: float = Field(default=0.0, ge=0.0)
    operating_hours_per_year: float = Field(default=6000.0, ge=0.0)
    electricity_cost_eur_per_kwh: float = Field(default=0.10, ge=0.0)
    electricity_ef_kgco2_kwh: float = Field(default=0.385, ge=0.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class CompressedAirAuditResult(BaseModel):
    """Complete result from compressed air audit workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="compressed_air_audit")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    facility_id: str = Field(default="")
    compressor_performance: List[CompressorPerformance] = Field(default_factory=list)
    leaks_detected: int = Field(default=0)
    leak_flow_total_cfm: float = Field(default=0.0)
    leak_cost_annual_eur: float = Field(default=0.0)
    recommendations: List[CompressedAirRecommendation] = Field(default_factory=list)
    total_system_energy_kwh: float = Field(default=0.0)
    total_savings_potential_kwh: float = Field(default=0.0)
    total_savings_potential_eur: float = Field(default=0.0)
    total_co2_reduction_tonnes: float = Field(default=0.0)
    system_specific_power: float = Field(default=0.0, description="Overall kW/(m3/min)")
    system_leak_pct: float = Field(default=0.0, description="Leakage as % of supply")
    provenance_hash: str = Field(default="")


# =============================================================================
# COMPRESSED AIR CONSTANTS (Zero-Hallucination)
# =============================================================================

# Specific power benchmarks kW/(m3/min) at 7 bar (ISO 1217 / CAGI)
SPECIFIC_POWER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "rotary_screw": {"excellent": 5.5, "good": 6.5, "average": 7.5, "poor": 9.0},
    "reciprocating": {"excellent": 5.0, "good": 6.0, "average": 7.0, "poor": 8.5},
    "centrifugal": {"excellent": 5.0, "good": 5.8, "average": 6.8, "poor": 8.0},
    "scroll": {"excellent": 6.0, "good": 7.0, "average": 8.0, "poor": 9.5},
    "rotary_vane": {"excellent": 6.5, "good": 7.5, "average": 8.5, "poor": 10.0},
}

# Pressure correction: each 1 bar above 7 bar adds ~7% energy
PRESSURE_ENERGY_FACTOR_PER_BAR = 0.07

# cfm to m3/min conversion
CFM_TO_M3_MIN = 0.028317

# Leak flow estimation from dB level (simplified model)
# Based on ultrasonic leak detection correlations
DB_TO_CFM: Dict[str, float] = {
    "trace": 0.25, "minor": 1.0, "major": 3.5, "critical": 7.0,
}

# Receiver sizing rule: 3-5 litres per m3/min for primary storage
RECEIVER_SIZING_LITRES_PER_M3_MIN = 4.0


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CompressedAirAuditWorkflow:
    """
    4-phase compressed air system audit workflow.

    Performs system mapping, ultrasonic leak survey with quantification,
    compressor performance testing, and optimization recommendations
    per ISO 11011 and CAGI standards.

    Zero-hallucination: all calculations use deterministic engineering
    formulas. No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _performance_results: Per-compressor performance.
        _recommendations: Optimization recommendations.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = CompressedAirAuditWorkflow()
        >>> inp = CompressedAirAuditInput(compressors=[...], leaks=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CompressedAirAuditWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._performance_results: List[CompressorPerformance] = []
        self._recommendations: List[CompressedAirRecommendation] = []
        self._phase_results: List[PhaseResult] = []
        self._total_system_kwh: float = 0.0
        self._total_leak_cfm: float = 0.0
        self._total_supply_cfm: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[CompressedAirAuditInput] = None,
        compressors: Optional[List[CompressorRecord]] = None,
        leaks: Optional[List[LeakRecord]] = None,
    ) -> CompressedAirAuditResult:
        """
        Execute the 4-phase compressed air audit workflow.

        Args:
            input_data: Full input model (preferred).
            compressors: Compressor list (fallback).
            leaks: Leak list (fallback).

        Returns:
            CompressedAirAuditResult with performance and recommendations.
        """
        if input_data is None:
            input_data = CompressedAirAuditInput(
                compressors=compressors or [],
                leaks=leaks or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting compressed air audit workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._performance_results = []
        self._recommendations = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_system_mapping(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_leak_survey(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_performance_testing(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_optimization_recommendations(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Compressed air audit workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        leak_cost = self._total_leak_cfm * CFM_TO_M3_MIN * 7.5 * input_data.operating_hours_per_year * input_data.electricity_cost_eur_per_kwh
        leak_pct = (self._total_leak_cfm / self._total_supply_cfm * 100.0) if self._total_supply_cfm > 0 else 0.0
        total_savings_kwh = sum(r.annual_savings_kwh for r in self._recommendations)
        total_savings_eur = sum(r.annual_savings_eur for r in self._recommendations)
        total_co2 = sum(r.co2_reduction_tonnes for r in self._recommendations)

        system_sp = 0.0
        total_flow = sum(p.fad_m3_min for p in self._performance_results)
        total_power = sum(p.annual_energy_kwh / max(input_data.operating_hours_per_year, 1) for p in self._performance_results)
        if total_flow > 0:
            system_sp = total_power / total_flow

        result = CompressedAirAuditResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            facility_id=input_data.facility_id,
            compressor_performance=self._performance_results,
            leaks_detected=len(input_data.leaks),
            leak_flow_total_cfm=round(self._total_leak_cfm, 2),
            leak_cost_annual_eur=round(leak_cost, 2),
            recommendations=self._recommendations,
            total_system_energy_kwh=round(self._total_system_kwh, 2),
            total_savings_potential_kwh=round(total_savings_kwh, 2),
            total_savings_potential_eur=round(total_savings_eur, 2),
            total_co2_reduction_tonnes=round(total_co2, 4),
            system_specific_power=round(system_sp, 2),
            system_leak_pct=round(leak_pct, 1),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Compressed air audit workflow %s completed in %.2fs leaks=%d savings=%.0f kWh",
            self.workflow_id, elapsed, len(input_data.leaks), total_savings_kwh,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: System Mapping
    # -------------------------------------------------------------------------

    async def _phase_system_mapping(
        self, input_data: CompressedAirAuditInput
    ) -> PhaseResult:
        """Inventory compressors, dryers, receivers, and distribution network."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Compressor summary
        total_rated_kw = sum(c.rated_power_kw for c in input_data.compressors)
        total_rated_cfm = sum(c.rated_flow_cfm for c in input_data.compressors)
        total_rated_m3_min = sum(
            c.rated_flow_m3_min if c.rated_flow_m3_min > 0 else c.rated_flow_cfm * CFM_TO_M3_MIN
            for c in input_data.compressors
        )
        self._total_supply_cfm = total_rated_cfm

        vsd_count = sum(1 for c in input_data.compressors if c.is_vsd)
        avg_age = 0
        aged = [c.year_installed for c in input_data.compressors if c.year_installed > 0]
        if aged:
            avg_age = datetime.utcnow().year - int(sum(aged) / len(aged))

        # Receiver storage assessment
        total_receiver_litres = sum(r.volume_litres for r in input_data.receivers)
        recommended_litres = total_rated_m3_min * 1000 * RECEIVER_SIZING_LITRES_PER_M3_MIN
        storage_adequate = total_receiver_litres >= recommended_litres * 0.8

        if not storage_adequate:
            warnings.append(
                f"Receiver storage {total_receiver_litres:.0f}L is below recommended "
                f"{recommended_litres:.0f}L"
            )

        # Dryer summary
        dryer_types = [d.dryer_type.value for d in input_data.dryers]
        dryer_power = sum(d.power_consumption_kw for d in input_data.dryers)

        if not input_data.compressors:
            warnings.append("No compressors in inventory")

        outputs["compressor_count"] = len(input_data.compressors)
        outputs["total_rated_kw"] = round(total_rated_kw, 2)
        outputs["total_rated_cfm"] = round(total_rated_cfm, 2)
        outputs["total_rated_m3_min"] = round(total_rated_m3_min, 2)
        outputs["vsd_compressors"] = vsd_count
        outputs["avg_compressor_age_years"] = avg_age
        outputs["dryer_count"] = len(input_data.dryers)
        outputs["dryer_types"] = dryer_types
        outputs["dryer_total_power_kw"] = round(dryer_power, 2)
        outputs["receiver_count"] = len(input_data.receivers)
        outputs["total_storage_litres"] = round(total_receiver_litres, 0)
        outputs["recommended_storage_litres"] = round(recommended_litres, 0)
        outputs["storage_adequate"] = storage_adequate
        outputs["system_pressure_bar"] = input_data.system_pressure_bar

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 SystemMapping: %d compressors, %.0f kW, %d dryers, %d receivers",
            len(input_data.compressors), total_rated_kw,
            len(input_data.dryers), len(input_data.receivers),
        )
        return PhaseResult(
            phase_name="system_mapping", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Leak Survey
    # -------------------------------------------------------------------------

    async def _phase_leak_survey(
        self, input_data: CompressedAirAuditInput
    ) -> PhaseResult:
        """Ultrasonic leak detection, quantification, and cost calculation."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_leak_cfm = 0.0
        total_leak_cost = 0.0
        severity_counts: Dict[str, int] = {}

        for leak in input_data.leaks:
            # Quantify leak flow
            flow_cfm = leak.estimated_flow_cfm
            if flow_cfm <= 0:
                flow_cfm = DB_TO_CFM.get(leak.severity.value, 1.0)

            # Calculate annual cost of leak
            flow_m3_min = flow_cfm * CFM_TO_M3_MIN
            # Power to generate leaked air: flow * specific_power
            leak_power_kw = flow_m3_min * 7.5  # Approx kW at 7 bar
            leak_annual_kwh = leak_power_kw * input_data.operating_hours_per_year
            leak_annual_cost = leak_annual_kwh * input_data.electricity_cost_eur_per_kwh

            total_leak_cfm += flow_cfm
            total_leak_cost += leak_annual_cost

            sev = leak.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        self._total_leak_cfm = total_leak_cfm
        leak_pct = (total_leak_cfm / self._total_supply_cfm * 100.0) if self._total_supply_cfm > 0 else 0.0

        if leak_pct > 25.0:
            warnings.append(f"System leak rate {leak_pct:.1f}% exceeds 25% threshold")
        elif leak_pct > 10.0:
            warnings.append(f"System leak rate {leak_pct:.1f}% exceeds 10% best practice")

        outputs["leaks_detected"] = len(input_data.leaks)
        outputs["total_leak_cfm"] = round(total_leak_cfm, 2)
        outputs["total_leak_m3_min"] = round(total_leak_cfm * CFM_TO_M3_MIN, 3)
        outputs["leak_pct_of_supply"] = round(leak_pct, 1)
        outputs["annual_leak_cost_eur"] = round(total_leak_cost, 2)
        outputs["severity_breakdown"] = severity_counts
        outputs["accessible_leaks"] = sum(1 for l in input_data.leaks if l.is_accessible)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 LeakSurvey: %d leaks, %.1f cfm total, %.0f EUR/yr",
            len(input_data.leaks), total_leak_cfm, total_leak_cost,
        )
        return PhaseResult(
            phase_name="leak_survey", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Performance Testing
    # -------------------------------------------------------------------------

    async def _phase_performance_testing(
        self, input_data: CompressedAirAuditInput
    ) -> PhaseResult:
        """Calculate specific power, FAD, and load profiles per compressor."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for comp in input_data.compressors:
            perf = self._test_compressor_performance(comp, input_data)
            self._performance_results.append(perf)

        self._total_system_kwh = sum(p.annual_energy_kwh for p in self._performance_results)

        outputs["compressors_tested"] = len(self._performance_results)
        outputs["total_annual_energy_kwh"] = round(self._total_system_kwh, 2)
        outputs["total_annual_cost_eur"] = round(
            sum(p.annual_cost_eur for p in self._performance_results), 2
        )
        outputs["performance_ratings"] = {
            p.compressor_id: p.efficiency_rating for p in self._performance_results
        }

        poor_performers = [p for p in self._performance_results if p.efficiency_rating == "poor"]
        if poor_performers:
            warnings.append(
                f"{len(poor_performers)} compressor(s) rated 'poor' efficiency"
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PerformanceTesting: %d compressors, total=%.0f kWh/yr",
            len(self._performance_results), self._total_system_kwh,
        )
        return PhaseResult(
            phase_name="performance_testing", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _test_compressor_performance(
        self, comp: CompressorRecord, input_data: CompressedAirAuditInput
    ) -> CompressorPerformance:
        """Calculate performance metrics for a single compressor."""
        # Calculate FAD
        fad_m3_min = comp.rated_flow_m3_min if comp.rated_flow_m3_min > 0 else comp.rated_flow_cfm * CFM_TO_M3_MIN
        fad_cfm = comp.rated_flow_cfm if comp.rated_flow_cfm > 0 else fad_m3_min / CFM_TO_M3_MIN

        # Measured or calculated power
        loaded_kw = comp.measured_power_loaded_kw if comp.measured_power_loaded_kw > 0 else comp.rated_power_kw
        unloaded_kw = comp.measured_power_unloaded_kw if comp.measured_power_unloaded_kw > 0 else loaded_kw * 0.25

        # Load percentage
        total_hours = comp.operating_hours_per_year if comp.operating_hours_per_year > 0 else input_data.operating_hours_per_year
        loaded_hours = comp.loaded_hours_per_year if comp.loaded_hours_per_year > 0 else total_hours * 0.70
        unloaded_hours = comp.unloaded_hours_per_year if comp.unloaded_hours_per_year > 0 else total_hours - loaded_hours
        load_pct = (loaded_hours / max(total_hours, 1)) * 100.0

        # Annual energy
        annual_kwh = (loaded_kw * loaded_hours) + (unloaded_kw * unloaded_hours)
        annual_cost = annual_kwh * input_data.electricity_cost_eur_per_kwh

        # Specific power (kW per m3/min)
        sp = loaded_kw / fad_m3_min if fad_m3_min > 0 else 0.0

        # Pressure correction
        pressure_diff = comp.actual_pressure_bar - 7.0
        if pressure_diff > 0:
            sp_corrected = sp / (1.0 + pressure_diff * PRESSURE_ENERGY_FACTOR_PER_BAR)
        else:
            sp_corrected = sp

        # Rate against benchmarks
        benchmarks = SPECIFIC_POWER_BENCHMARKS.get(comp.compressor_type.value, SPECIFIC_POWER_BENCHMARKS["rotary_screw"])
        if sp_corrected <= benchmarks["excellent"]:
            rating = "excellent"
        elif sp_corrected <= benchmarks["good"]:
            rating = "good"
        elif sp_corrected <= benchmarks["average"]:
            rating = "average"
        else:
            rating = "poor"

        sp_per_100cfm = loaded_kw / (fad_cfm / 100.0) if fad_cfm > 0 else 0.0

        return CompressorPerformance(
            compressor_id=comp.compressor_id,
            specific_power_kw_per_m3_min=round(sp, 2),
            specific_power_kw_per_100cfm=round(sp_per_100cfm, 2),
            fad_m3_min=round(fad_m3_min, 2),
            fad_cfm=round(fad_cfm, 2),
            load_pct=round(load_pct, 1),
            efficiency_rating=rating,
            annual_energy_kwh=round(annual_kwh, 2),
            annual_cost_eur=round(annual_cost, 2),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Optimization Recommendations
    # -------------------------------------------------------------------------

    async def _phase_optimization_recommendations(
        self, input_data: CompressedAirAuditInput
    ) -> PhaseResult:
        """Generate VSD, pressure reduction, leak repair, receiver sizing recs."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # 1. Leak Repair
        if self._total_leak_cfm > 0:
            self._recommendations.append(
                self._recommend_leak_repair(input_data)
            )

        # 2. Pressure Reduction
        if input_data.system_pressure_bar > 7.0:
            self._recommendations.append(
                self._recommend_pressure_reduction(input_data)
            )

        # 3. VSD Retrofit for partially loaded compressors
        for perf in self._performance_results:
            comp = next((c for c in input_data.compressors if c.compressor_id == perf.compressor_id), None)
            if comp and not comp.is_vsd and perf.load_pct < 70.0 and perf.annual_energy_kwh > 10000:
                self._recommendations.append(
                    self._recommend_vsd_retrofit(comp, perf, input_data)
                )

        # 4. Receiver Sizing
        total_receiver_litres = sum(r.volume_litres for r in input_data.receivers)
        total_flow = sum(p.fad_m3_min for p in self._performance_results)
        recommended_litres = total_flow * 1000 * RECEIVER_SIZING_LITRES_PER_M3_MIN
        if total_receiver_litres < recommended_litres * 0.8:
            self._recommendations.append(
                self._recommend_receiver_upgrade(
                    total_receiver_litres, recommended_litres, input_data
                )
            )

        # Sort by payback
        self._recommendations.sort(key=lambda r: r.simple_payback_years)

        outputs["recommendations_count"] = len(self._recommendations)
        outputs["total_savings_kwh"] = round(
            sum(r.annual_savings_kwh for r in self._recommendations), 2
        )
        outputs["total_savings_eur"] = round(
            sum(r.annual_savings_eur for r in self._recommendations), 2
        )
        outputs["total_investment_eur"] = round(
            sum(r.implementation_cost_eur for r in self._recommendations), 2
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 Optimization: %d recommendations, savings=%.0f EUR/yr",
            len(self._recommendations), outputs["total_savings_eur"],
        )
        return PhaseResult(
            phase_name="optimization_recommendations", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _recommend_leak_repair(
        self, input_data: CompressedAirAuditInput
    ) -> CompressedAirRecommendation:
        """Generate leak repair recommendation."""
        flow_m3_min = self._total_leak_cfm * CFM_TO_M3_MIN
        power_saved_kw = flow_m3_min * 7.5  # approx kW at 7 bar
        savings_kwh = power_saved_kw * input_data.operating_hours_per_year * 0.80  # 80% repair rate
        savings_eur = savings_kwh * input_data.electricity_cost_eur_per_kwh
        co2 = savings_kwh * input_data.electricity_ef_kgco2_kwh / 1000.0
        repair_cost = sum(l.repair_cost_eur for l in input_data.leaks if l.repair_cost_eur > 0)
        if repair_cost == 0:
            repair_cost = len(input_data.leaks) * 50.0  # EUR 50 per leak average
        payback = repair_cost / savings_eur if savings_eur > 0 else 0.0

        return CompressedAirRecommendation(
            title="Compressed air leak repair program",
            description=(
                f"Repair {len(input_data.leaks)} leaks totaling {self._total_leak_cfm:.1f} cfm. "
                f"Assuming 80% repair success rate."
            ),
            category="leak_repair",
            priority=RecommendationPriority.IMMEDIATE,
            annual_savings_kwh=round(savings_kwh, 2),
            annual_savings_eur=round(savings_eur, 2),
            implementation_cost_eur=round(repair_cost, 2),
            simple_payback_years=round(payback, 2),
            co2_reduction_tonnes=round(co2, 4),
        )

    def _recommend_pressure_reduction(
        self, input_data: CompressedAirAuditInput
    ) -> CompressedAirRecommendation:
        """Generate pressure reduction recommendation."""
        current_bar = input_data.system_pressure_bar
        target_bar = max(6.5, current_bar - 1.0)
        reduction_bar = current_bar - target_bar
        savings_pct = reduction_bar * PRESSURE_ENERGY_FACTOR_PER_BAR
        savings_kwh = self._total_system_kwh * savings_pct
        savings_eur = savings_kwh * input_data.electricity_cost_eur_per_kwh
        co2 = savings_kwh * input_data.electricity_ef_kgco2_kwh / 1000.0
        impl_cost = 2000.0  # Controls adjustment / flow controller
        payback = impl_cost / savings_eur if savings_eur > 0 else 0.0

        return CompressedAirRecommendation(
            title=f"Reduce system pressure from {current_bar:.1f} to {target_bar:.1f} bar",
            description=(
                f"Each 1 bar reduction saves ~7% energy. Reduction of {reduction_bar:.1f} bar "
                f"saves {savings_pct * 100:.1f}% of compressor energy."
            ),
            category="pressure",
            priority=RecommendationPriority.SHORT_TERM,
            annual_savings_kwh=round(savings_kwh, 2),
            annual_savings_eur=round(savings_eur, 2),
            implementation_cost_eur=round(impl_cost, 2),
            simple_payback_years=round(payback, 2),
            co2_reduction_tonnes=round(co2, 4),
        )

    def _recommend_vsd_retrofit(
        self,
        comp: CompressorRecord,
        perf: CompressorPerformance,
        input_data: CompressedAirAuditInput,
    ) -> CompressedAirRecommendation:
        """Generate VSD retrofit recommendation for a partially loaded compressor."""
        # VSD savings: eliminates unloaded running and improves part-load efficiency
        loaded_frac = perf.load_pct / 100.0
        # Current energy with load/unload
        current_kwh = perf.annual_energy_kwh
        # VSD energy: proportional to actual load
        vsd_kwh = comp.rated_power_kw * loaded_frac * input_data.operating_hours_per_year
        savings_kwh = max(0.0, current_kwh - vsd_kwh)
        savings_eur = savings_kwh * input_data.electricity_cost_eur_per_kwh
        co2 = savings_kwh * input_data.electricity_ef_kgco2_kwh / 1000.0
        # VSD cost: approx EUR 200/kW
        impl_cost = comp.rated_power_kw * 200.0
        payback = impl_cost / savings_eur if savings_eur > 0 else 99.0

        return CompressedAirRecommendation(
            title=f"VSD retrofit for compressor {comp.name or comp.compressor_id}",
            description=(
                f"Compressor {comp.compressor_id} runs at {perf.load_pct:.0f}% load. "
                f"VSD retrofit would match speed to demand, saving {savings_kwh:.0f} kWh/yr."
            ),
            category="vsd",
            priority=RecommendationPriority.MEDIUM_TERM if payback > 3.0 else RecommendationPriority.SHORT_TERM,
            annual_savings_kwh=round(savings_kwh, 2),
            annual_savings_eur=round(savings_eur, 2),
            implementation_cost_eur=round(impl_cost, 2),
            simple_payback_years=round(payback, 2),
            co2_reduction_tonnes=round(co2, 4),
        )

    def _recommend_receiver_upgrade(
        self,
        current_litres: float,
        recommended_litres: float,
        input_data: CompressedAirAuditInput,
    ) -> CompressedAirRecommendation:
        """Generate receiver storage upgrade recommendation."""
        additional_litres = recommended_litres - current_litres
        # Additional storage reduces short-cycling, approx 3% energy saving
        savings_pct = 0.03
        savings_kwh = self._total_system_kwh * savings_pct
        savings_eur = savings_kwh * input_data.electricity_cost_eur_per_kwh
        co2 = savings_kwh * input_data.electricity_ef_kgco2_kwh / 1000.0
        impl_cost = additional_litres * 3.0  # approx EUR 3/litre
        payback = impl_cost / savings_eur if savings_eur > 0 else 99.0

        return CompressedAirRecommendation(
            title="Increase receiver storage capacity",
            description=(
                f"Current storage {current_litres:.0f}L, recommended {recommended_litres:.0f}L. "
                f"Additional {additional_litres:.0f}L reduces short-cycling."
            ),
            category="receiver",
            priority=RecommendationPriority.MEDIUM_TERM,
            annual_savings_kwh=round(savings_kwh, 2),
            annual_savings_eur=round(savings_eur, 2),
            implementation_cost_eur=round(impl_cost, 2),
            simple_payback_years=round(payback, 2),
            co2_reduction_tonnes=round(co2, 4),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: CompressedAirAuditResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
