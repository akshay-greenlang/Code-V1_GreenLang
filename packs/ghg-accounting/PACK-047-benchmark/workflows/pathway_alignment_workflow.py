# -*- coding: utf-8 -*-
"""
Pathway Alignment Workflow
====================================

4-phase workflow for science-based pathway alignment assessment within
PACK-047 GHG Emissions Benchmark Pack.

Phases:
    1. PathwayLoading             -- Load applicable science-based
                                     decarbonisation pathways (IEA Net Zero,
                                     IPCC SR1.5, SBTi SDA, OECM 1.5C, TPI
                                     Carbon Performance, CRREM) for the
                                     entity's sector and geography.
    2. WaypointInterpolation      -- Interpolate annual waypoints for each
                                     loaded pathway from the base year through
                                     the target year, using linear, exponential,
                                     or convergence interpolation methods as
                                     specified by each pathway source.
    3. GapAnalysis                -- Calculate the gap between the entity's
                                     current emissions trajectory and each
                                     pathway: absolute gap (tCO2e), percentage
                                     gap, years to convergence, and cumulative
                                     budget overshoot.
    4. AlignmentScoring           -- Compute a composite alignment score
                                     across all applicable pathways, weighting
                                     by pathway authority and sector relevance,
                                     and assign a temperature alignment estimate.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    IEA Net Zero by 2050 Roadmap (2023 update)
    IPCC SR1.5 (2018) - 1.5C-consistent pathways
    SBTi SDA v2.0 - Sectoral Decarbonisation Approach
    OECM (2022) - One Earth Climate Model 1.5C pathways
    TPI (2024) - Transition Pathway Initiative benchmarks
    CRREM (2023) - Carbon Risk Real Estate Monitor
    ESRS E1-4 (2024) - Transition plan alignment
    TCFD Recommendations - Strategy scenario analysis

Schedule: Annually after emissions calculation
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 47.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

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

class AlignmentPhase(str, Enum):
    """Pathway alignment workflow phases."""

    PATHWAY_LOADING = "pathway_loading"
    WAYPOINT_INTERPOLATION = "waypoint_interpolation"
    GAP_ANALYSIS = "gap_analysis"
    ALIGNMENT_SCORING = "alignment_scoring"

class PathwaySource(str, Enum):
    """Science-based pathway source."""

    IEA_NZE = "iea_nze"
    IPCC_SR15 = "ipcc_sr15"
    SBTI_SDA = "sbti_sda"
    OECM_15C = "oecm_15c"
    TPI = "tpi"
    CRREM = "crrem"

class InterpolationMethod(str, Enum):
    """Method for waypoint interpolation."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CONVERGENCE = "convergence"

class TemperatureAlignment(str, Enum):
    """Temperature alignment classification."""

    BELOW_1_5C = "below_1_5c"
    AT_1_5C = "at_1_5c"
    BETWEEN_1_5C_2C = "between_1_5c_2c"
    AT_2C = "at_2c"
    ABOVE_2C = "above_2c"
    WELL_ABOVE_2C = "well_above_2c"

class AlignmentStatus(str, Enum):
    """Alignment status relative to a pathway."""

    ALIGNED = "aligned"
    CONVERGING = "converging"
    DIVERGING = "diverging"
    NOT_ALIGNED = "not_aligned"

# =============================================================================
# PATHWAY REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Reduction from 2020 base year by target year as percentage
PATHWAY_REDUCTION_TARGETS: Dict[str, Dict[str, Any]] = {
    "iea_nze": {
        "name": "IEA Net Zero by 2050",
        "base_year": 2020,
        "target_year": 2050,
        "waypoints": {2025: 12.0, 2030: 30.0, 2035: 50.0, 2040: 65.0, 2050: 100.0},
        "interpolation": "exponential",
        "authority_weight": 0.25,
        "temperature": "1.5c",
    },
    "ipcc_sr15": {
        "name": "IPCC SR1.5 Median",
        "base_year": 2020,
        "target_year": 2050,
        "waypoints": {2025: 10.0, 2030: 45.0, 2040: 70.0, 2050: 100.0},
        "interpolation": "linear",
        "authority_weight": 0.25,
        "temperature": "1.5c",
    },
    "sbti_sda": {
        "name": "SBTi SDA 1.5C",
        "base_year": 2020,
        "target_year": 2050,
        "waypoints": {2025: 8.0, 2030: 42.0, 2035: 58.0, 2040: 72.0, 2050: 100.0},
        "interpolation": "convergence",
        "authority_weight": 0.20,
        "temperature": "1.5c",
    },
    "oecm_15c": {
        "name": "OECM 1.5C",
        "base_year": 2020,
        "target_year": 2050,
        "waypoints": {2025: 15.0, 2030: 50.0, 2040: 80.0, 2050: 100.0},
        "interpolation": "exponential",
        "authority_weight": 0.10,
        "temperature": "1.5c",
    },
    "tpi": {
        "name": "TPI Carbon Performance 2C",
        "base_year": 2020,
        "target_year": 2050,
        "waypoints": {2025: 5.0, 2030: 20.0, 2035: 35.0, 2040: 50.0, 2050: 80.0},
        "interpolation": "linear",
        "authority_weight": 0.10,
        "temperature": "2c",
    },
    "crrem": {
        "name": "CRREM 1.5C (Real Estate)",
        "base_year": 2020,
        "target_year": 2050,
        "waypoints": {2025: 18.0, 2030: 40.0, 2035: 55.0, 2040: 68.0, 2050: 100.0},
        "interpolation": "linear",
        "authority_weight": 0.10,
        "temperature": "1.5c",
    },
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class PathwayDefinition(BaseModel):
    """A loaded science-based pathway."""

    source: PathwaySource = Field(...)
    name: str = Field(default="")
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2050)
    waypoints: Dict[int, float] = Field(
        default_factory=dict,
        description="Year -> cumulative reduction % from base",
    )
    interpolation_method: InterpolationMethod = Field(
        default=InterpolationMethod.LINEAR,
    )
    authority_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_target: str = Field(default="1.5c")
    applicable: bool = Field(default=True)
    provenance_hash: str = Field(default="")

class AnnualWaypoint(BaseModel):
    """An interpolated annual waypoint on a pathway."""

    source: PathwaySource = Field(...)
    year: int = Field(...)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    absolute_target_tco2e: float = Field(default=0.0, ge=0.0)
    provenance_hash: str = Field(default="")

class PathwayGap(BaseModel):
    """Gap between entity trajectory and a pathway for a given year."""

    source: PathwaySource = Field(...)
    year: int = Field(...)
    entity_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    pathway_target_tco2e: float = Field(default=0.0, ge=0.0)
    absolute_gap_tco2e: float = Field(default=0.0)
    percentage_gap: float = Field(default=0.0)
    years_to_convergence: Optional[int] = Field(default=None)
    cumulative_overshoot_tco2e: float = Field(default=0.0)
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.NOT_ALIGNED)
    provenance_hash: str = Field(default="")

class PathwayAlignmentScore(BaseModel):
    """Alignment score for a single pathway."""

    source: PathwaySource = Field(...)
    name: str = Field(default="")
    alignment_score: float = Field(default=0.0, ge=0.0, le=100.0)
    weighted_score: float = Field(default=0.0, ge=0.0, le=100.0)
    authority_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    current_gap_pct: float = Field(default=0.0)
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.NOT_ALIGNED)
    provenance_hash: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class PathwayAlignmentInput(BaseModel):
    """Input data model for PathwayAlignmentWorkflow."""

    organization_id: str = Field(..., min_length=1)
    base_year: int = Field(default=2020, ge=2010, le=2030)
    base_year_emissions_tco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total Scope 1+2 emissions in base year",
    )
    current_year: int = Field(default=2024, ge=2015, le=2035)
    current_emissions_tco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total Scope 1+2 emissions in current year",
    )
    projected_emissions: Dict[int, float] = Field(
        default_factory=dict,
        description="Year -> projected emissions tCO2e for future years",
    )
    applicable_pathways: List[PathwaySource] = Field(
        default_factory=lambda: [
            PathwaySource.IEA_NZE,
            PathwaySource.IPCC_SR15,
            PathwaySource.SBTI_SDA,
        ],
    )
    sector: str = Field(default="industrials")
    target_year: int = Field(default=2050, ge=2030, le=2060)
    include_scope3: bool = Field(default=False)
    scope3_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class PathwayAlignmentResult(BaseModel):
    """Complete result from pathway alignment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="pathway_alignment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    loaded_pathways: List[PathwayDefinition] = Field(default_factory=list)
    annual_waypoints: List[AnnualWaypoint] = Field(default_factory=list)
    pathway_gaps: List[PathwayGap] = Field(default_factory=list)
    alignment_scores: List[PathwayAlignmentScore] = Field(default_factory=list)
    composite_alignment_score: float = Field(default=0.0, ge=0.0, le=100.0)
    temperature_alignment: TemperatureAlignment = Field(
        default=TemperatureAlignment.ABOVE_2C,
    )
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class PathwayAlignmentWorkflow:
    """
    4-phase workflow for science-based pathway alignment assessment.

    Loads applicable decarbonisation pathways, interpolates annual waypoints,
    calculates gap-to-pathway metrics, and computes composite alignment
    scores with temperature alignment estimate.

    Zero-hallucination: pathway definitions use published reduction targets;
    interpolation uses deterministic mathematical functions; no LLM calls in
    calculation path; SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _pathways: Loaded pathway definitions.
        _waypoints: Interpolated annual waypoints.
        _gaps: Gap analysis results per pathway/year.
        _scores: Alignment scores per pathway.
        _composite_score: Weighted composite alignment.
        _temperature: Temperature alignment estimate.

    Example:
        >>> wf = PathwayAlignmentWorkflow()
        >>> inp = PathwayAlignmentInput(
        ...     organization_id="org-001",
        ...     base_year_emissions_tco2e=10000,
        ...     current_emissions_tco2e=8500,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[AlignmentPhase] = [
        AlignmentPhase.PATHWAY_LOADING,
        AlignmentPhase.WAYPOINT_INTERPOLATION,
        AlignmentPhase.GAP_ANALYSIS,
        AlignmentPhase.ALIGNMENT_SCORING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PathwayAlignmentWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._pathways: List[PathwayDefinition] = []
        self._waypoints: List[AnnualWaypoint] = []
        self._gaps: List[PathwayGap] = []
        self._scores: List[PathwayAlignmentScore] = []
        self._composite_score: float = 0.0
        self._temperature: TemperatureAlignment = TemperatureAlignment.ABOVE_2C
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: PathwayAlignmentInput,
    ) -> PathwayAlignmentResult:
        """
        Execute the 4-phase pathway alignment workflow.

        Args:
            input_data: Organisation emissions and pathway selection.

        Returns:
            PathwayAlignmentResult with alignment scores and temperature estimate.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting pathway alignment %s org=%s pathways=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.applicable_pathways),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_pathway_loading,
            self._phase_2_waypoint_interpolation,
            self._phase_3_gap_analysis,
            self._phase_4_alignment_scoring,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Pathway alignment failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = PathwayAlignmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            loaded_pathways=self._pathways,
            annual_waypoints=self._waypoints,
            pathway_gaps=self._gaps,
            alignment_scores=self._scores,
            composite_alignment_score=self._composite_score,
            temperature_alignment=self._temperature,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Pathway alignment %s completed in %.2fs status=%s score=%.1f temp=%s",
            self.workflow_id, elapsed, overall_status.value,
            self._composite_score, self._temperature.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Pathway Loading
    # -------------------------------------------------------------------------

    async def _phase_1_pathway_loading(
        self, input_data: PathwayAlignmentInput,
    ) -> PhaseResult:
        """Load applicable science-based decarbonisation pathways."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._pathways = []

        for source in input_data.applicable_pathways:
            ref = PATHWAY_REDUCTION_TARGETS.get(source.value)
            if not ref:
                warnings.append(f"Unknown pathway source: {source.value}")
                continue

            # Check sector applicability
            applicable = True
            if source == PathwaySource.CRREM and input_data.sector != "real_estate":
                applicable = False
                warnings.append(
                    f"CRREM pathway not applicable to sector {input_data.sector}"
                )

            try:
                interp_method = InterpolationMethod(ref.get("interpolation", "linear"))
            except ValueError:
                interp_method = InterpolationMethod.LINEAR

            pathway_data = {
                "source": source.value, "name": ref["name"],
                "base": ref["base_year"], "target": ref["target_year"],
            }

            self._pathways.append(PathwayDefinition(
                source=source,
                name=ref["name"],
                base_year=ref["base_year"],
                target_year=ref["target_year"],
                waypoints={int(k): float(v) for k, v in ref["waypoints"].items()},
                interpolation_method=interp_method,
                authority_weight=ref.get("authority_weight", 0.1),
                temperature_target=ref.get("temperature", "1.5c"),
                applicable=applicable,
                provenance_hash=_compute_hash(pathway_data),
            ))

        outputs["pathways_loaded"] = len(self._pathways)
        outputs["pathways_applicable"] = sum(1 for p in self._pathways if p.applicable)
        outputs["pathway_names"] = [p.name for p in self._pathways]

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 PathwayLoading: %d pathways loaded, %d applicable",
            len(self._pathways),
            sum(1 for p in self._pathways if p.applicable),
        )
        return PhaseResult(
            phase_name="pathway_loading", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Waypoint Interpolation
    # -------------------------------------------------------------------------

    async def _phase_2_waypoint_interpolation(
        self, input_data: PathwayAlignmentInput,
    ) -> PhaseResult:
        """Interpolate annual waypoints for each pathway."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._waypoints = []
        base_emissions = input_data.base_year_emissions_tco2e

        if base_emissions <= 0:
            warnings.append("Base year emissions is zero; waypoints will be zero")

        for pathway in self._pathways:
            if not pathway.applicable:
                continue

            for year in range(pathway.base_year, pathway.target_year + 1):
                reduction_pct = self._interpolate_reduction(
                    year, pathway.waypoints,
                    pathway.base_year, pathway.target_year,
                    pathway.interpolation_method,
                )

                target_emissions = base_emissions * (1.0 - reduction_pct / 100.0)

                wp_data = {
                    "source": pathway.source.value, "year": year,
                    "reduction": round(reduction_pct, 4),
                }
                self._waypoints.append(AnnualWaypoint(
                    source=pathway.source,
                    year=year,
                    reduction_pct=round(reduction_pct, 4),
                    absolute_target_tco2e=round(max(target_emissions, 0.0), 2),
                    provenance_hash=_compute_hash(wp_data),
                ))

        outputs["waypoints_generated"] = len(self._waypoints)
        outputs["pathways_interpolated"] = sum(
            1 for p in self._pathways if p.applicable
        )
        outputs["year_range"] = (
            f"{min(w.year for w in self._waypoints)}-"
            f"{max(w.year for w in self._waypoints)}"
            if self._waypoints else "none"
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 WaypointInterpolation: %d waypoints generated",
            len(self._waypoints),
        )
        return PhaseResult(
            phase_name="waypoint_interpolation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_3_gap_analysis(
        self, input_data: PathwayAlignmentInput,
    ) -> PhaseResult:
        """Calculate gap between entity trajectory and each pathway."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gaps = []
        current_emissions = input_data.current_emissions_tco2e
        if input_data.include_scope3:
            current_emissions += input_data.scope3_emissions_tco2e

        # Build entity trajectory: historical + projected
        entity_trajectory: Dict[int, float] = {
            input_data.base_year: input_data.base_year_emissions_tco2e,
            input_data.current_year: current_emissions,
        }
        entity_trajectory.update(input_data.projected_emissions)

        for pathway in self._pathways:
            if not pathway.applicable:
                continue

            pathway_waypoints = {
                w.year: w.absolute_target_tco2e
                for w in self._waypoints if w.source == pathway.source
            }

            # Compute gap for current year
            current_target = pathway_waypoints.get(input_data.current_year, 0.0)
            abs_gap = current_emissions - current_target
            pct_gap = (
                (abs_gap / max(current_target, 1e-12)) * 100.0
                if current_target > 0 else 0.0
            )

            # Estimate years to convergence
            years_to_converge: Optional[int] = None
            if abs_gap > 0 and input_data.projected_emissions:
                for future_year in sorted(input_data.projected_emissions.keys()):
                    future_entity = input_data.projected_emissions[future_year]
                    future_target = pathway_waypoints.get(future_year)
                    if future_target and future_entity <= future_target:
                        years_to_converge = future_year - input_data.current_year
                        break

            # Cumulative overshoot: sum of gaps over pathway years
            cum_overshoot = Decimal("0")
            for year in range(input_data.current_year, pathway.target_year + 1):
                entity_val = entity_trajectory.get(year, current_emissions)
                target_val = pathway_waypoints.get(year, 0.0)
                if entity_val > target_val:
                    cum_overshoot += Decimal(str(entity_val - target_val))

            # Alignment status
            if abs_gap <= 0:
                status = AlignmentStatus.ALIGNED
            elif pct_gap < 10.0:
                status = AlignmentStatus.CONVERGING
            elif years_to_converge and years_to_converge <= 5:
                status = AlignmentStatus.CONVERGING
            else:
                status = AlignmentStatus.DIVERGING

            gap_data = {
                "source": pathway.source.value,
                "year": input_data.current_year,
                "abs_gap": round(abs_gap, 2), "pct_gap": round(pct_gap, 4),
            }
            self._gaps.append(PathwayGap(
                source=pathway.source,
                year=input_data.current_year,
                entity_emissions_tco2e=round(current_emissions, 2),
                pathway_target_tco2e=round(current_target, 2),
                absolute_gap_tco2e=round(abs_gap, 2),
                percentage_gap=round(pct_gap, 4),
                years_to_convergence=years_to_converge,
                cumulative_overshoot_tco2e=float(
                    cum_overshoot.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ),
                alignment_status=status,
                provenance_hash=_compute_hash(gap_data),
            ))

        outputs["gaps_computed"] = len(self._gaps)
        outputs["aligned_pathways"] = sum(
            1 for g in self._gaps if g.alignment_status == AlignmentStatus.ALIGNED
        )
        outputs["converging_pathways"] = sum(
            1 for g in self._gaps if g.alignment_status == AlignmentStatus.CONVERGING
        )
        outputs["diverging_pathways"] = sum(
            1 for g in self._gaps if g.alignment_status == AlignmentStatus.DIVERGING
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 GapAnalysis: %d gaps, %d aligned, %d converging",
            len(self._gaps),
            outputs["aligned_pathways"],
            outputs["converging_pathways"],
        )
        return PhaseResult(
            phase_name="gap_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Alignment Scoring
    # -------------------------------------------------------------------------

    async def _phase_4_alignment_scoring(
        self, input_data: PathwayAlignmentInput,
    ) -> PhaseResult:
        """Compute composite alignment score and temperature alignment."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._scores = []
        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for pathway in self._pathways:
            if not pathway.applicable:
                continue

            # Find gap for this pathway
            gap = next(
                (g for g in self._gaps if g.source == pathway.source), None,
            )
            if not gap:
                continue

            # Alignment score: 100 if aligned, decreasing with gap
            if gap.absolute_gap_tco2e <= 0:
                alignment_score = 100.0
            elif gap.percentage_gap <= 0:
                alignment_score = 100.0
            else:
                # Score decreases with gap percentage: 100 - gap_pct, min 0
                alignment_score = max(0.0, 100.0 - gap.percentage_gap)

            weighted = alignment_score * pathway.authority_weight

            score_data = {
                "source": pathway.source.value,
                "score": round(alignment_score, 2),
                "weight": pathway.authority_weight,
            }
            self._scores.append(PathwayAlignmentScore(
                source=pathway.source,
                name=pathway.name,
                alignment_score=round(alignment_score, 2),
                weighted_score=round(weighted, 4),
                authority_weight=pathway.authority_weight,
                current_gap_pct=gap.percentage_gap,
                alignment_status=gap.alignment_status,
                provenance_hash=_compute_hash(score_data),
            ))

            total_weight += Decimal(str(pathway.authority_weight))
            weighted_sum += Decimal(str(weighted))

        # Composite score
        if total_weight > 0:
            self._composite_score = float(
                (weighted_sum / total_weight).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP,
                )
            )
        else:
            self._composite_score = 0.0

        # Temperature alignment estimate
        self._temperature = self._estimate_temperature(self._composite_score)

        outputs["alignment_scores_computed"] = len(self._scores)
        outputs["composite_alignment_score"] = self._composite_score
        outputs["temperature_alignment"] = self._temperature.value
        outputs["total_authority_weight"] = float(total_weight)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 AlignmentScoring: composite=%.1f temp=%s",
            self._composite_score, self._temperature.value,
        )
        return PhaseResult(
            phase_name="alignment_scoring", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: PathwayAlignmentInput,
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
    # Calculation Helpers
    # -------------------------------------------------------------------------

    def _interpolate_reduction(
        self, year: int, waypoints: Dict[int, float],
        base_year: int, target_year: int,
        method: InterpolationMethod,
    ) -> float:
        """Interpolate reduction percentage for a given year."""
        if year <= base_year:
            return 0.0
        if year >= target_year:
            return waypoints.get(target_year, 100.0)
        if year in waypoints:
            return waypoints[year]

        # Find bracketing waypoints
        sorted_years = sorted([base_year] + list(waypoints.keys()))
        lower_year = base_year
        lower_val = 0.0
        upper_year = target_year
        upper_val = waypoints.get(target_year, 100.0)

        for wy in sorted_years:
            if wy <= year:
                lower_year = wy
                lower_val = waypoints.get(wy, 0.0) if wy != base_year else 0.0
            elif wy > year:
                upper_year = wy
                upper_val = waypoints.get(wy, 100.0)
                break

        span = upper_year - lower_year
        if span <= 0:
            return lower_val

        t = (year - lower_year) / span

        if method == InterpolationMethod.LINEAR:
            return lower_val + t * (upper_val - lower_val)
        elif method == InterpolationMethod.EXPONENTIAL:
            if lower_val <= 0:
                lower_val = 0.1
            ratio = upper_val / max(lower_val, 0.1)
            return lower_val * (ratio ** t)
        elif method == InterpolationMethod.CONVERGENCE:
            # Faster initial, slower later (square root curve)
            return lower_val + math.sqrt(t) * (upper_val - lower_val)
        else:
            return lower_val + t * (upper_val - lower_val)

    def _estimate_temperature(self, composite_score: float) -> TemperatureAlignment:
        """Estimate temperature alignment from composite score."""
        if composite_score >= 90.0:
            return TemperatureAlignment.BELOW_1_5C
        elif composite_score >= 75.0:
            return TemperatureAlignment.AT_1_5C
        elif composite_score >= 55.0:
            return TemperatureAlignment.BETWEEN_1_5C_2C
        elif composite_score >= 35.0:
            return TemperatureAlignment.AT_2C
        elif composite_score >= 15.0:
            return TemperatureAlignment.ABOVE_2C
        else:
            return TemperatureAlignment.WELL_ABOVE_2C

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._pathways = []
        self._waypoints = []
        self._gaps = []
        self._scores = []
        self._composite_score = 0.0
        self._temperature = TemperatureAlignment.ABOVE_2C

    def _compute_provenance(self, result: PathwayAlignmentResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.composite_alignment_score}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
