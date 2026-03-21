# -*- coding: utf-8 -*-
"""
Target Setting Workflow
===================================

3-phase workflow for energy benchmark target setting within
PACK-035 Energy Benchmark Pack.

Phases:
    1. BaselineEstablishment  -- Establish baseline EUI with data quality assessment
    2. PeerContextAnalysis    -- Analyse peer distribution for context-aware targets
    3. TargetDefinition       -- Define targets (absolute, peer-relative, SBTi-aligned),
                                 generate trajectory with milestones

Supports absolute reduction targets, peer-relative targets (top-quartile
approach), and SBTi 1.5C-aligned trajectories using SDA pathways.

The workflow follows GreenLang zero-hallucination principles: baseline
calculations, peer distributions, and SBTi trajectories use deterministic
formulas and published decarbonisation pathways. No LLM calls in the
numeric computation path.

Schedule: on-demand / annual
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class AmbitionLevel(str, Enum):
    """Target ambition level."""

    MINIMUM_COMPLIANCE = "minimum_compliance"
    PEER_MEDIAN = "peer_median"
    TOP_QUARTILE = "top_quartile"
    BEST_IN_CLASS = "best_in_class"
    SBTi_1_5C = "sbti_1_5c"
    SBTi_WB2C = "sbti_well_below_2c"
    NZEB = "nearly_zero_energy"
    CUSTOM = "custom"


class BuildingType(str, Enum):
    """Building type classification."""

    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    DATA_CENTRE = "data_centre"
    MIXED_USE = "mixed_use"


class TargetType(str, Enum):
    """Type of target."""

    ABSOLUTE_EUI = "absolute_eui"
    REDUCTION_PCT = "reduction_pct"
    CARBON_INTENSITY = "carbon_intensity"
    ENERGY_STAR_SCORE = "energy_star_score"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# CIBSE TM46 typical/good benchmarks
CIBSE_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {"typical": 215.0, "good": 133.0, "best": 80.0},
    "retail": {"typical": 270.0, "good": 150.0, "best": 95.0},
    "hotel": {"typical": 305.0, "good": 180.0, "best": 120.0},
    "hospital": {"typical": 440.0, "good": 315.0, "best": 220.0},
    "school": {"typical": 150.0, "good": 87.0, "best": 55.0},
    "warehouse": {"typical": 65.0, "good": 40.0, "best": 25.0},
    "industrial": {"typical": 255.0, "good": 155.0, "best": 95.0},
    "data_centre": {"typical": 510.0, "good": 305.0, "best": 200.0},
    "mixed_use": {"typical": 230.0, "good": 140.0, "best": 90.0},
}

# SBTi SDA annual reduction rates by sector (% per year)
# Source: SBTi Buildings Sector Guidance v1.0
SBTI_SDA_RATES: Dict[str, Dict[str, float]] = {
    "office": {"1.5C": 4.2, "WB2C": 2.5},
    "retail": {"1.5C": 3.8, "WB2C": 2.2},
    "hotel": {"1.5C": 3.5, "WB2C": 2.0},
    "hospital": {"1.5C": 3.0, "WB2C": 1.8},
    "school": {"1.5C": 4.0, "WB2C": 2.3},
    "warehouse": {"1.5C": 3.5, "WB2C": 2.0},
    "industrial": {"1.5C": 3.2, "WB2C": 1.9},
    "data_centre": {"1.5C": 4.5, "WB2C": 2.7},
    "mixed_use": {"1.5C": 3.8, "WB2C": 2.2},
}

# CRREM 2050 target EUIs by building type (kWh/m2/yr)
CRREM_2050_TARGETS: Dict[str, float] = {
    "office": 40.0, "retail": 50.0, "hotel": 55.0, "hospital": 100.0,
    "school": 30.0, "warehouse": 15.0, "industrial": 45.0,
    "data_centre": 80.0, "mixed_use": 45.0,
}

# CO2 emission factors
EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.207,
    "natural_gas": 0.183,
    "fuel_oil": 0.267,
}

# Grid decarbonisation factor (annual grid EF reduction %)
GRID_DECARB_RATE = 3.5  # % per year (EU average trajectory)


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


class TargetMilestone(BaseModel):
    """Annual target milestone."""

    year: int = Field(default=0, ge=0)
    target_eui: float = Field(default=0.0, ge=0.0, description="Target EUI kWh/m2/yr")
    reduction_from_baseline_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    carbon_intensity_target: float = Field(default=0.0, ge=0.0, description="kgCO2e/m2/yr")
    cumulative_savings_kwh_m2: float = Field(default=0.0, ge=0.0)


class ProposedTarget(BaseModel):
    """A proposed energy target."""

    target_name: str = Field(default="", description="Target description")
    target_type: TargetType = Field(default=TargetType.ABSOLUTE_EUI)
    ambition_level: str = Field(default="")
    target_year: int = Field(default=0, ge=0)
    target_eui: float = Field(default=0.0, ge=0.0, description="Target EUI kWh/m2/yr")
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    annual_reduction_rate: float = Field(default=0.0, ge=0.0, description="% per year")
    milestones: List[TargetMilestone] = Field(default_factory=list)
    sbti_aligned: bool = Field(default=False)
    feasibility_score: float = Field(default=0.0, ge=0.0, le=100.0)


class TargetSettingInput(BaseModel):
    """Input data model for TargetSettingWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    building_type: BuildingType = Field(default=BuildingType.OFFICE)
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    baseline_year: int = Field(default=2024, ge=2015, le=2030)
    baseline_eui: float = Field(default=0.0, ge=0.0, description="Baseline EUI kWh/m2/yr")
    baseline_consumption_kwh: float = Field(default=0.0, ge=0.0)
    energy_mix: Dict[str, float] = Field(
        default_factory=lambda: {"electricity": 0.6, "natural_gas": 0.4},
        description="Fraction by energy source",
    )
    target_year: int = Field(default=2030, ge=2025, le=2060)
    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.TOP_QUARTILE)
    sbti_aligned: bool = Field(default=False, description="Require SBTi alignment check")
    custom_target_eui: float = Field(default=0.0, ge=0.0, description="Custom target if CUSTOM")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class TargetSettingResult(BaseModel):
    """Complete result from target setting workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="target_setting")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    baseline_eui: float = Field(default=0.0, ge=0.0)
    baseline_year: int = Field(default=0)
    peer_context: Dict[str, Any] = Field(default_factory=dict)
    proposed_targets: List[Dict[str, Any]] = Field(default_factory=list)
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_alignment: Dict[str, Any] = Field(default_factory=dict)
    recommended_target: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TargetSettingWorkflow:
    """
    3-phase energy benchmark target setting workflow.

    Performs baseline establishment, peer context analysis, and target
    definition with SBTi alignment checking and trajectory generation.

    Zero-hallucination: baseline calculations, peer benchmarks from
    CIBSE/ENERGY STAR, SBTi SDA pathway rates, and CRREM targets
    use published deterministic values. No LLM in numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _baseline_eui: Established baseline EUI.
        _peer_context: Peer distribution context.
        _proposed_targets: List of proposed targets.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = TargetSettingWorkflow()
        >>> inp = TargetSettingInput(
        ...     building_type=BuildingType.OFFICE, floor_area_m2=5000,
        ...     baseline_eui=200.0, target_year=2030,
        ... )
        >>> result = wf.run(inp)
        >>> assert len(result.proposed_targets) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TargetSettingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._baseline_eui: float = 0.0
        self._baseline_carbon: float = 0.0
        self._peer_context: Dict[str, Any] = {}
        self._proposed_targets: List[ProposedTarget] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: TargetSettingInput) -> TargetSettingResult:
        """
        Execute the 3-phase target setting workflow.

        Args:
            input_data: Validated target setting input.

        Returns:
            TargetSettingResult with baseline, peer context, proposed targets, trajectory.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting target setting workflow %s ambition=%s target_year=%d",
            self.workflow_id, input_data.ambition_level.value, input_data.target_year,
        )

        self._phase_results = []
        self._baseline_eui = 0.0
        self._baseline_carbon = 0.0
        self._peer_context = {}
        self._proposed_targets = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_baseline_establishment(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_peer_context_analysis(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_target_definition(input_data)
            self._phase_results.append(phase3)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Target setting workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        # Select recommended target
        recommended = {}
        if self._proposed_targets:
            best = max(self._proposed_targets, key=lambda t: t.feasibility_score)
            recommended = best.model_dump()

        # Build trajectory from recommended
        trajectory = []
        if self._proposed_targets:
            primary = self._proposed_targets[0]
            trajectory = [m.model_dump() for m in primary.milestones]

        # SBTi alignment
        sbti_data = self._check_sbti_alignment(input_data)

        result = TargetSettingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            baseline_eui=round(self._baseline_eui, 2),
            baseline_year=input_data.baseline_year,
            peer_context=self._peer_context,
            proposed_targets=[t.model_dump() for t in self._proposed_targets],
            trajectory=trajectory,
            sbti_alignment=sbti_data,
            recommended_target=recommended,
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Target setting workflow %s completed in %.2fs targets=%d",
            self.workflow_id, elapsed, len(self._proposed_targets),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Establishment
    # -------------------------------------------------------------------------

    def _phase_baseline_establishment(
        self, input_data: TargetSettingInput
    ) -> PhaseResult:
        """Establish baseline EUI with data quality assessment."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Determine baseline EUI
        if input_data.baseline_eui > 0:
            self._baseline_eui = input_data.baseline_eui
        elif input_data.baseline_consumption_kwh > 0 and input_data.floor_area_m2 > 0:
            self._baseline_eui = input_data.baseline_consumption_kwh / input_data.floor_area_m2
        else:
            bt = input_data.building_type.value
            benchmarks = CIBSE_BENCHMARKS.get(bt, CIBSE_BENCHMARKS["office"])
            self._baseline_eui = benchmarks["typical"]
            warnings.append("No baseline data provided; using typical benchmark as proxy")

        # Calculate baseline carbon intensity
        total_ef = 0.0
        for source, fraction in input_data.energy_mix.items():
            ef = EMISSION_FACTORS.get(source, 0.207)
            total_ef += ef * fraction
        self._baseline_carbon = self._baseline_eui * total_ef

        # Data quality score
        dq_score = 0.0
        if input_data.baseline_eui > 0:
            dq_score += 50.0
        if input_data.baseline_consumption_kwh > 0:
            dq_score += 30.0
        if input_data.floor_area_m2 > 0:
            dq_score += 20.0

        outputs["baseline_eui"] = round(self._baseline_eui, 2)
        outputs["baseline_carbon_kgco2_m2"] = round(self._baseline_carbon, 4)
        outputs["baseline_year"] = input_data.baseline_year
        outputs["data_quality_score"] = round(dq_score, 1)
        outputs["energy_mix"] = input_data.energy_mix

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 BaselineEstablishment: EUI=%.1f carbon=%.2f kgCO2/m2",
            self._baseline_eui, self._baseline_carbon,
        )
        return PhaseResult(
            phase_name="baseline_establishment", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Peer Context Analysis
    # -------------------------------------------------------------------------

    def _phase_peer_context_analysis(
        self, input_data: TargetSettingInput
    ) -> PhaseResult:
        """Analyse peer distribution for context-aware target setting."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bt = input_data.building_type.value

        benchmarks = CIBSE_BENCHMARKS.get(bt, CIBSE_BENCHMARKS["office"])
        typical = benchmarks["typical"]
        good = benchmarks["good"]
        best = benchmarks["best"]
        crrem_2050 = CRREM_2050_TARGETS.get(bt, 45.0)

        # Peer distribution estimates
        std_dev = (typical - good) * 0.80
        p10 = max(1.0, typical - 1.28 * std_dev)
        p25 = max(1.0, good)
        p50 = typical * 0.95
        p75 = typical + 0.67 * std_dev
        p90 = typical + 1.28 * std_dev

        # Position of facility in distribution
        z = (typical - self._baseline_eui) / std_dev if std_dev > 0 else 0.0
        percentile = 100.0 / (1.0 + math.exp(-1.7 * z))
        percentile = max(1.0, min(99.0, percentile))

        self._peer_context = {
            "building_type": bt,
            "typical_eui": round(typical, 2),
            "good_practice_eui": round(good, 2),
            "best_practice_eui": round(best, 2),
            "crrem_2050_target": round(crrem_2050, 2),
            "peer_p10": round(p10, 2),
            "peer_p25": round(p25, 2),
            "peer_p50": round(p50, 2),
            "peer_p75": round(p75, 2),
            "peer_p90": round(p90, 2),
            "facility_percentile": round(percentile, 1),
            "gap_to_good_pct": round((self._baseline_eui - good) / good * 100.0, 2) if good > 0 else 0.0,
            "gap_to_best_pct": round((self._baseline_eui - best) / best * 100.0, 2) if best > 0 else 0.0,
        }

        outputs.update(self._peer_context)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 PeerContextAnalysis: percentile=%.1f gap_to_good=%.1f%%",
            percentile, self._peer_context["gap_to_good_pct"],
        )
        return PhaseResult(
            phase_name="peer_context_analysis", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Target Definition
    # -------------------------------------------------------------------------

    def _phase_target_definition(
        self, input_data: TargetSettingInput
    ) -> PhaseResult:
        """Define targets with trajectories and milestones."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        bt = input_data.building_type.value
        years = input_data.target_year - input_data.baseline_year

        if years <= 0:
            warnings.append("Target year must be after baseline year")
            years = 5

        benchmarks = CIBSE_BENCHMARKS.get(bt, CIBSE_BENCHMARKS["office"])

        # Generate targets based on ambition level
        ambition = input_data.ambition_level

        if ambition == AmbitionLevel.CUSTOM and input_data.custom_target_eui > 0:
            target_eui = input_data.custom_target_eui
            self._proposed_targets.append(self._build_target(
                "Custom Target", target_eui, input_data, years, feasibility=70.0,
            ))
        elif ambition == AmbitionLevel.PEER_MEDIAN:
            target_eui = benchmarks["typical"] * 0.95
            self._proposed_targets.append(self._build_target(
                "Peer Median", target_eui, input_data, years, feasibility=90.0,
            ))
        elif ambition == AmbitionLevel.TOP_QUARTILE:
            target_eui = benchmarks["good"]
            self._proposed_targets.append(self._build_target(
                "Top Quartile (Good Practice)", target_eui, input_data, years, feasibility=80.0,
            ))
        elif ambition == AmbitionLevel.BEST_IN_CLASS:
            target_eui = benchmarks["best"]
            self._proposed_targets.append(self._build_target(
                "Best in Class", target_eui, input_data, years, feasibility=55.0,
            ))
        elif ambition == AmbitionLevel.SBTi_1_5C:
            sda_rate = SBTI_SDA_RATES.get(bt, {"1.5C": 4.0})["1.5C"]
            target_eui = self._baseline_eui * ((1.0 - sda_rate / 100.0) ** years)
            self._proposed_targets.append(self._build_target(
                f"SBTi 1.5C ({sda_rate}%/yr)", target_eui, input_data, years,
                feasibility=60.0, sbti=True, annual_rate=sda_rate,
            ))
        elif ambition == AmbitionLevel.SBTi_WB2C:
            sda_rate = SBTI_SDA_RATES.get(bt, {"WB2C": 2.5})["WB2C"]
            target_eui = self._baseline_eui * ((1.0 - sda_rate / 100.0) ** years)
            self._proposed_targets.append(self._build_target(
                f"SBTi WB2C ({sda_rate}%/yr)", target_eui, input_data, years,
                feasibility=75.0, sbti=True, annual_rate=sda_rate,
            ))
        elif ambition == AmbitionLevel.NZEB:
            target_eui = CRREM_2050_TARGETS.get(bt, 45.0)
            self._proposed_targets.append(self._build_target(
                "Near-Zero Energy Building", target_eui, input_data, years, feasibility=40.0,
            ))
        else:
            # MINIMUM_COMPLIANCE
            target_eui = benchmarks["typical"]
            self._proposed_targets.append(self._build_target(
                "Minimum Compliance", target_eui, input_data, years, feasibility=95.0,
            ))

        # Always add an alternative SBTi target for comparison
        if ambition != AmbitionLevel.SBTi_1_5C:
            sda_rate = SBTI_SDA_RATES.get(bt, {"1.5C": 4.0})["1.5C"]
            sbti_eui = self._baseline_eui * ((1.0 - sda_rate / 100.0) ** years)
            self._proposed_targets.append(self._build_target(
                f"SBTi 1.5C Reference ({sda_rate}%/yr)", sbti_eui, input_data, years,
                feasibility=60.0, sbti=True, annual_rate=sda_rate,
            ))

        outputs["targets_proposed"] = len(self._proposed_targets)
        outputs["primary_target_eui"] = round(self._proposed_targets[0].target_eui, 2) if self._proposed_targets else 0.0
        outputs["primary_reduction_pct"] = round(self._proposed_targets[0].reduction_pct, 2) if self._proposed_targets else 0.0
        outputs["ambition_level"] = ambition.value

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 TargetDefinition: %d targets, primary=%.1f kWh/m2 (%.1f%% reduction)",
            len(self._proposed_targets),
            outputs["primary_target_eui"],
            outputs["primary_reduction_pct"],
        )
        return PhaseResult(
            phase_name="target_definition", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _build_target(
        self,
        name: str,
        target_eui: float,
        input_data: TargetSettingInput,
        years: int,
        feasibility: float = 70.0,
        sbti: bool = False,
        annual_rate: float = 0.0,
    ) -> ProposedTarget:
        """Build a proposed target with milestones (zero-hallucination)."""
        target_eui = max(1.0, target_eui)
        reduction_pct = ((self._baseline_eui - target_eui) / self._baseline_eui * 100.0) if self._baseline_eui > 0 else 0.0
        reduction_pct = max(0.0, reduction_pct)

        if annual_rate <= 0 and years > 0:
            annual_rate = reduction_pct / years

        # Generate annual milestones
        milestones: List[TargetMilestone] = []
        for y in range(0, years + 1):
            if annual_rate > 0:
                ms_eui = self._baseline_eui * ((1.0 - annual_rate / 100.0) ** y)
            else:
                # Linear interpolation
                ms_eui = self._baseline_eui - (self._baseline_eui - target_eui) * y / years

            ms_eui = max(1.0, ms_eui)
            ms_reduction = ((self._baseline_eui - ms_eui) / self._baseline_eui * 100.0) if self._baseline_eui > 0 else 0.0
            ms_cumulative = max(0.0, (self._baseline_eui - ms_eui) * y)

            # Carbon intensity at milestone (accounting for grid decarbonisation)
            total_ef = 0.0
            for source, fraction in input_data.energy_mix.items():
                ef = EMISSION_FACTORS.get(source, 0.207)
                if source == "electricity":
                    ef = ef * ((1.0 - GRID_DECARB_RATE / 100.0) ** y)
                total_ef += ef * fraction
            carbon_target = ms_eui * total_ef

            milestones.append(TargetMilestone(
                year=input_data.baseline_year + y,
                target_eui=round(ms_eui, 2),
                reduction_from_baseline_pct=round(ms_reduction, 2),
                carbon_intensity_target=round(carbon_target, 4),
                cumulative_savings_kwh_m2=round(ms_cumulative, 2),
            ))

        # Feasibility adjustment based on gap size
        if reduction_pct > 50:
            feasibility = min(feasibility, 40.0)
        elif reduction_pct > 30:
            feasibility = min(feasibility, 60.0)

        return ProposedTarget(
            target_name=name,
            target_type=TargetType.ABSOLUTE_EUI,
            ambition_level=input_data.ambition_level.value,
            target_year=input_data.target_year,
            target_eui=round(target_eui, 2),
            reduction_pct=round(reduction_pct, 2),
            annual_reduction_rate=round(annual_rate, 2),
            milestones=milestones,
            sbti_aligned=sbti,
            feasibility_score=round(feasibility, 1),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _check_sbti_alignment(self, input_data: TargetSettingInput) -> Dict[str, Any]:
        """Check SBTi alignment of proposed targets (zero-hallucination)."""
        bt = input_data.building_type.value
        sda_rates = SBTI_SDA_RATES.get(bt, {"1.5C": 4.0, "WB2C": 2.5})
        years = input_data.target_year - input_data.baseline_year
        if years <= 0:
            years = 5

        sbti_1_5c_eui = self._baseline_eui * ((1.0 - sda_rates["1.5C"] / 100.0) ** years)
        sbti_wb2c_eui = self._baseline_eui * ((1.0 - sda_rates["WB2C"] / 100.0) ** years)

        alignment_results: List[Dict[str, Any]] = []
        for target in self._proposed_targets:
            aligned_1_5c = target.target_eui <= sbti_1_5c_eui * 1.05  # 5% tolerance
            aligned_wb2c = target.target_eui <= sbti_wb2c_eui * 1.05
            alignment_results.append({
                "target_name": target.target_name,
                "target_eui": target.target_eui,
                "sbti_1_5c_aligned": aligned_1_5c,
                "sbti_wb2c_aligned": aligned_wb2c,
            })

        return {
            "sbti_1_5c_target_eui": round(sbti_1_5c_eui, 2),
            "sbti_wb2c_target_eui": round(sbti_wb2c_eui, 2),
            "sda_rate_1_5c": sda_rates["1.5C"],
            "sda_rate_wb2c": sda_rates["WB2C"],
            "target_alignment": alignment_results,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: TargetSettingResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
