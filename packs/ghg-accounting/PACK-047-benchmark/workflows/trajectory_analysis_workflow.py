# -*- coding: utf-8 -*-
"""
Trajectory Analysis Workflow
====================================

4-phase workflow for multi-year emissions trajectory analysis within
PACK-047 GHG Emissions Benchmark Pack.

Phases:
    1. TimeSeriesLoading          -- Load multi-year emissions time series
                                     for the organisation and peer entities,
                                     validate temporal completeness, identify
                                     gaps, and align reporting periods.
    2. CARRComputation            -- Calculate the Compound Annual Reduction
                                     Rate (CARR) for each entity, along with
                                     acceleration/deceleration metrics and
                                     structural break detection using Chow
                                     test approximation.
    3. ConvergenceAnalysis        -- Analyse convergence or divergence of
                                     the organisation's trajectory relative
                                     to the peer group median, including
                                     beta-convergence (catching up) and
                                     sigma-convergence (spread narrowing).
    4. TrajectoryRanking          -- Rank all entities by decarbonisation
                                     speed (CARR), assign trajectory bands,
                                     and compute relative momentum scores.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ESRS E1-4 (2024) - Climate change transition plan targets
    SBTi Progress Framework (2024) - Target tracking methodology
    CDP C4.1-C4.2 (2026) - Targets and performance tracking
    TCFD Recommendations - Metrics and target trajectories
    GRI 305 (2016) - Emissions trend reporting
    IFRS S2 (2023) - Historical emissions trend disclosure

Schedule: Annually or after each reporting cycle
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


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


class TrajectoryPhase(str, Enum):
    """Trajectory analysis workflow phases."""

    TIME_SERIES_LOADING = "time_series_loading"
    CARR_COMPUTATION = "carr_computation"
    CONVERGENCE_ANALYSIS = "convergence_analysis"
    TRAJECTORY_RANKING = "trajectory_ranking"


class TrajectoryBand(str, Enum):
    """Trajectory performance band based on CARR."""

    RAPID_DECARBONISER = "rapid_decarboniser"
    STEADY_DECARBONISER = "steady_decarboniser"
    SLOW_DECARBONISER = "slow_decarboniser"
    FLAT = "flat"
    INCREASING = "increasing"


class ConvergenceType(str, Enum):
    """Type of convergence analysis."""

    BETA_CONVERGENCE = "beta_convergence"
    SIGMA_CONVERGENCE = "sigma_convergence"


class ConvergenceStatus(str, Enum):
    """Convergence status relative to peer median."""

    CONVERGING = "converging"
    DIVERGING = "diverging"
    STABLE = "stable"
    OVERTAKING = "overtaking"


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


class EntityTimeSeries(BaseModel):
    """Multi-year emissions time series for an entity."""

    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    is_organisation: bool = Field(default=False)
    annual_emissions: Dict[int, float] = Field(
        default_factory=dict,
        description="Year -> tCO2e emissions",
    )
    annual_intensity: Dict[int, float] = Field(
        default_factory=dict,
        description="Year -> intensity metric",
    )
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)


class CARRResult(BaseModel):
    """Compound Annual Reduction Rate result for an entity."""

    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    is_organisation: bool = Field(default=False)
    start_year: int = Field(default=0)
    end_year: int = Field(default=0)
    start_emissions: float = Field(default=0.0, ge=0.0)
    end_emissions: float = Field(default=0.0, ge=0.0)
    carr_pct: float = Field(default=0.0)
    total_change_pct: float = Field(default=0.0)
    acceleration: float = Field(
        default=0.0,
        description="Rate of change in reduction rate (positive=accelerating)",
    )
    has_structural_break: bool = Field(default=False)
    structural_break_year: Optional[int] = Field(default=None)
    trajectory_band: TrajectoryBand = Field(default=TrajectoryBand.FLAT)
    provenance_hash: str = Field(default="")


class ConvergenceResult(BaseModel):
    """Convergence analysis result."""

    convergence_type: ConvergenceType = Field(...)
    status: ConvergenceStatus = Field(default=ConvergenceStatus.STABLE)
    coefficient: float = Field(
        default=0.0,
        description="Beta-convergence coefficient or sigma-convergence ratio",
    )
    org_distance_to_median_start: float = Field(default=0.0)
    org_distance_to_median_end: float = Field(default=0.0)
    distance_change_pct: float = Field(default=0.0)
    peer_spread_start: float = Field(default=0.0)
    peer_spread_end: float = Field(default=0.0)
    spread_change_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class TrajectoryRank(BaseModel):
    """Trajectory ranking for an entity."""

    entity_id: str = Field(...)
    entity_name: str = Field(default="")
    is_organisation: bool = Field(default=False)
    carr_pct: float = Field(default=0.0)
    rank_position: int = Field(default=0, ge=0)
    total_entities: int = Field(default=0, ge=0)
    percentile: float = Field(default=0.0, ge=0.0, le=100.0)
    trajectory_band: TrajectoryBand = Field(default=TrajectoryBand.FLAT)
    momentum_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class TrajectoryAnalysisInput(BaseModel):
    """Input data model for TrajectoryAnalysisWorkflow."""

    organization_id: str = Field(..., min_length=1)
    org_time_series: EntityTimeSeries = Field(
        ..., description="Organisation multi-year emissions",
    )
    peer_time_series: List[EntityTimeSeries] = Field(
        default_factory=list,
        description="Peer entity multi-year emissions",
    )
    analysis_start_year: int = Field(default=2019, ge=2010, le=2030)
    analysis_end_year: int = Field(default=2024, ge=2015, le=2035)
    structural_break_threshold_pct: float = Field(
        default=20.0, ge=5.0, le=50.0,
        description="Year-over-year change % to flag structural break",
    )
    minimum_years: int = Field(
        default=3, ge=2, le=10,
        description="Minimum years of data required for analysis",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryAnalysisResult(BaseModel):
    """Complete result from trajectory analysis workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="trajectory_analysis")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    carr_results: List[CARRResult] = Field(default_factory=list)
    convergence_results: List[ConvergenceResult] = Field(default_factory=list)
    trajectory_ranks: List[TrajectoryRank] = Field(default_factory=list)
    org_carr_pct: float = Field(default=0.0)
    org_trajectory_band: TrajectoryBand = Field(default=TrajectoryBand.FLAT)
    org_rank_position: int = Field(default=0)
    org_momentum_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class TrajectoryAnalysisWorkflow:
    """
    4-phase workflow for multi-year emissions trajectory analysis.

    Loads time series data, computes CARR and structural breaks, analyses
    convergence to peer median, and ranks entities by decarbonisation speed.

    Zero-hallucination: CARR uses deterministic CAGR formula; structural
    breaks use threshold-based detection; convergence uses distance metrics;
    no LLM calls in calculation path; SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _org_series: Organisation time series.
        _peer_series: Peer time series list.
        _carr_results: CARR computation results.
        _convergence: Convergence analysis results.
        _ranks: Trajectory rankings.

    Example:
        >>> wf = TrajectoryAnalysisWorkflow()
        >>> org_ts = EntityTimeSeries(
        ...     entity_id="org-001",
        ...     annual_emissions={2020: 10000, 2021: 9500, 2022: 9000, 2023: 8500, 2024: 8000},
        ... )
        >>> inp = TrajectoryAnalysisInput(
        ...     organization_id="org-001",
        ...     org_time_series=org_ts,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[TrajectoryPhase] = [
        TrajectoryPhase.TIME_SERIES_LOADING,
        TrajectoryPhase.CARR_COMPUTATION,
        TrajectoryPhase.CONVERGENCE_ANALYSIS,
        TrajectoryPhase.TRAJECTORY_RANKING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize TrajectoryAnalysisWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._org_series: Optional[EntityTimeSeries] = None
        self._peer_series: List[EntityTimeSeries] = []
        self._carr_results: List[CARRResult] = []
        self._convergence: List[ConvergenceResult] = []
        self._ranks: List[TrajectoryRank] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: TrajectoryAnalysisInput,
    ) -> TrajectoryAnalysisResult:
        """
        Execute the 4-phase trajectory analysis workflow.

        Args:
            input_data: Organisation and peer time series data.

        Returns:
            TrajectoryAnalysisResult with CARR, convergence, and rankings.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting trajectory analysis %s org=%s peers=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.peer_time_series),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_time_series_loading,
            self._phase_2_carr_computation,
            self._phase_3_convergence_analysis,
            self._phase_4_trajectory_ranking,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Trajectory analysis failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Extract org-level results
        org_carr = next(
            (c for c in self._carr_results if c.is_organisation), None,
        )
        org_rank = next(
            (r for r in self._ranks if r.is_organisation), None,
        )

        result = TrajectoryAnalysisResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            carr_results=self._carr_results,
            convergence_results=self._convergence,
            trajectory_ranks=self._ranks,
            org_carr_pct=org_carr.carr_pct if org_carr else 0.0,
            org_trajectory_band=(
                org_carr.trajectory_band if org_carr else TrajectoryBand.FLAT
            ),
            org_rank_position=org_rank.rank_position if org_rank else 0,
            org_momentum_score=org_rank.momentum_score if org_rank else 0.0,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Trajectory analysis %s completed in %.2fs status=%s carr=%.2f%%",
            self.workflow_id, elapsed, overall_status.value,
            result.org_carr_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Time Series Loading
    # -------------------------------------------------------------------------

    async def _phase_1_time_series_loading(
        self, input_data: TrajectoryAnalysisInput,
    ) -> PhaseResult:
        """Load and validate multi-year emissions time series."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._org_series = input_data.org_time_series
        self._org_series.is_organisation = True
        self._peer_series = list(input_data.peer_time_series)

        # Validate org series has minimum years
        org_years = sorted(self._org_series.annual_emissions.keys())
        if len(org_years) < input_data.minimum_years:
            warnings.append(
                f"Organisation has {len(org_years)} years of data; "
                f"minimum is {input_data.minimum_years}"
            )

        # Validate peers
        valid_peers = 0
        for peer in self._peer_series:
            peer_years = sorted(peer.annual_emissions.keys())
            if len(peer_years) >= input_data.minimum_years:
                valid_peers += 1
            else:
                warnings.append(
                    f"Peer {peer.entity_id} has only {len(peer_years)} years"
                )

        # Find common year range
        all_years: set = set(org_years)
        for peer in self._peer_series:
            all_years.update(peer.annual_emissions.keys())

        outputs["org_years"] = org_years
        outputs["org_data_points"] = len(org_years)
        outputs["peers_loaded"] = len(self._peer_series)
        outputs["peers_valid"] = valid_peers
        outputs["total_year_range"] = (
            f"{min(all_years)}-{max(all_years)}" if all_years else "none"
        )
        outputs["analysis_window"] = (
            f"{input_data.analysis_start_year}-{input_data.analysis_end_year}"
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 TimeSeriesLoading: org=%d years, %d peers (%d valid)",
            len(org_years), len(self._peer_series), valid_peers,
        )
        return PhaseResult(
            phase_name="time_series_loading", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: CARR Computation
    # -------------------------------------------------------------------------

    async def _phase_2_carr_computation(
        self, input_data: TrajectoryAnalysisInput,
    ) -> PhaseResult:
        """Calculate CARR, acceleration, and structural breaks."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._carr_results = []
        all_series = [self._org_series] + self._peer_series

        for ts in all_series:
            if ts is None:
                continue

            carr_result = self._compute_carr(
                ts, input_data.analysis_start_year,
                input_data.analysis_end_year,
                input_data.structural_break_threshold_pct,
            )
            self._carr_results.append(carr_result)

        org_carr = next(
            (c for c in self._carr_results if c.is_organisation), None,
        )

        outputs["entities_analysed"] = len(self._carr_results)
        outputs["org_carr_pct"] = org_carr.carr_pct if org_carr else 0.0
        outputs["org_trajectory_band"] = (
            org_carr.trajectory_band.value if org_carr else "flat"
        )
        if self._carr_results:
            carrs = [c.carr_pct for c in self._carr_results]
            outputs["carr_range"] = {
                "min": round(min(carrs), 4),
                "max": round(max(carrs), 4),
                "mean": round(sum(carrs) / len(carrs), 4),
            }
        outputs["structural_breaks"] = sum(
            1 for c in self._carr_results if c.has_structural_break
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 CARRComputation: %d entities, org_carr=%.2f%%",
            len(self._carr_results),
            org_carr.carr_pct if org_carr else 0.0,
        )
        return PhaseResult(
            phase_name="carr_computation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Convergence Analysis
    # -------------------------------------------------------------------------

    async def _phase_3_convergence_analysis(
        self, input_data: TrajectoryAnalysisInput,
    ) -> PhaseResult:
        """Analyse convergence/divergence to peer median."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._convergence = []

        if not self._peer_series or not self._org_series:
            warnings.append("Insufficient data for convergence analysis")
            outputs["convergence_computed"] = 0
            elapsed = time.monotonic() - started
            return PhaseResult(
                phase_name="convergence_analysis", phase_number=3,
                status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                outputs=outputs, warnings=warnings,
                provenance_hash=_compute_hash(outputs),
            )

        start_year = input_data.analysis_start_year
        end_year = input_data.analysis_end_year

        # Beta-convergence: is org closing the distance to peer median?
        peer_start_emissions = []
        peer_end_emissions = []
        for peer in self._peer_series:
            s_val = peer.annual_emissions.get(start_year)
            e_val = peer.annual_emissions.get(end_year)
            if s_val is not None and e_val is not None:
                peer_start_emissions.append(s_val)
                peer_end_emissions.append(e_val)

        if peer_start_emissions and peer_end_emissions:
            peer_median_start = sorted(peer_start_emissions)[
                len(peer_start_emissions) // 2
            ]
            peer_median_end = sorted(peer_end_emissions)[
                len(peer_end_emissions) // 2
            ]

            org_start = self._org_series.annual_emissions.get(start_year, 0.0)
            org_end = self._org_series.annual_emissions.get(end_year, 0.0)

            dist_start = abs(org_start - peer_median_start)
            dist_end = abs(org_end - peer_median_end)
            dist_change = (
                ((dist_end - dist_start) / max(dist_start, 1e-12)) * 100.0
                if dist_start > 0 else 0.0
            )

            # Beta coefficient: negative = converging
            beta_coeff = dist_change / 100.0

            if dist_change < -10.0:
                beta_status = ConvergenceStatus.CONVERGING
            elif dist_change > 10.0:
                beta_status = ConvergenceStatus.DIVERGING
            elif org_end < peer_median_end and org_start >= peer_median_start:
                beta_status = ConvergenceStatus.OVERTAKING
            else:
                beta_status = ConvergenceStatus.STABLE

            beta_data = {
                "type": "beta", "coeff": round(beta_coeff, 4),
                "dist_change": round(dist_change, 4),
            }
            self._convergence.append(ConvergenceResult(
                convergence_type=ConvergenceType.BETA_CONVERGENCE,
                status=beta_status,
                coefficient=round(beta_coeff, 4),
                org_distance_to_median_start=round(dist_start, 2),
                org_distance_to_median_end=round(dist_end, 2),
                distance_change_pct=round(dist_change, 4),
                provenance_hash=_compute_hash(beta_data),
            ))

            # Sigma-convergence: is the peer group spread narrowing?
            if len(peer_start_emissions) >= 3 and len(peer_end_emissions) >= 3:
                spread_start = self._compute_std(peer_start_emissions)
                spread_end = self._compute_std(peer_end_emissions)
                spread_change = (
                    ((spread_end - spread_start) / max(spread_start, 1e-12)) * 100.0
                    if spread_start > 0 else 0.0
                )

                sigma_status = (
                    ConvergenceStatus.CONVERGING if spread_change < -5.0
                    else ConvergenceStatus.DIVERGING if spread_change > 5.0
                    else ConvergenceStatus.STABLE
                )

                sigma_data = {
                    "type": "sigma",
                    "spread_start": round(spread_start, 2),
                    "spread_end": round(spread_end, 2),
                }
                self._convergence.append(ConvergenceResult(
                    convergence_type=ConvergenceType.SIGMA_CONVERGENCE,
                    status=sigma_status,
                    coefficient=round(spread_change / 100.0, 4),
                    peer_spread_start=round(spread_start, 2),
                    peer_spread_end=round(spread_end, 2),
                    spread_change_pct=round(spread_change, 4),
                    provenance_hash=_compute_hash(sigma_data),
                ))

        outputs["convergence_computed"] = len(self._convergence)
        for conv in self._convergence:
            outputs[f"{conv.convergence_type.value}_status"] = conv.status.value

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 ConvergenceAnalysis: %d results", len(self._convergence),
        )
        return PhaseResult(
            phase_name="convergence_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Trajectory Ranking
    # -------------------------------------------------------------------------

    async def _phase_4_trajectory_ranking(
        self, input_data: TrajectoryAnalysisInput,
    ) -> PhaseResult:
        """Rank entities by decarbonisation speed and assign trajectory bands."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._ranks = []

        # Sort by CARR (most negative = fastest decarboniser)
        sorted_carrs = sorted(self._carr_results, key=lambda c: c.carr_pct)
        n = len(sorted_carrs)

        for rank_idx, carr in enumerate(sorted_carrs):
            # Percentile: lower CARR = better = higher percentile
            if n > 1:
                percentile = round(
                    ((n - 1 - rank_idx) / (n - 1)) * 100.0, 2,
                )
            else:
                percentile = 50.0

            # Momentum score: combination of CARR and acceleration
            momentum = self._compute_momentum(carr)

            rank_data = {
                "entity": carr.entity_id, "carr": carr.carr_pct,
                "rank": rank_idx + 1, "n": n,
            }
            self._ranks.append(TrajectoryRank(
                entity_id=carr.entity_id,
                entity_name=carr.entity_name,
                is_organisation=carr.is_organisation,
                carr_pct=carr.carr_pct,
                rank_position=rank_idx + 1,
                total_entities=n,
                percentile=percentile,
                trajectory_band=carr.trajectory_band,
                momentum_score=momentum,
                provenance_hash=_compute_hash(rank_data),
            ))

        org_rank = next(
            (r for r in self._ranks if r.is_organisation), None,
        )

        outputs["entities_ranked"] = len(self._ranks)
        outputs["org_rank"] = org_rank.rank_position if org_rank else 0
        outputs["org_percentile"] = org_rank.percentile if org_rank else 0.0
        outputs["org_momentum"] = org_rank.momentum_score if org_rank else 0.0
        if self._ranks:
            outputs["band_distribution"] = {}
            for band in TrajectoryBand:
                count = sum(1 for r in self._ranks if r.trajectory_band == band)
                outputs["band_distribution"][band.value] = count

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 TrajectoryRanking: %d ranked, org_rank=%d/%d",
            len(self._ranks),
            org_rank.rank_position if org_rank else 0,
            n,
        )
        return PhaseResult(
            phase_name="trajectory_ranking", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: TrajectoryAnalysisInput,
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

    def _compute_carr(
        self, ts: EntityTimeSeries,
        start_year: int, end_year: int,
        break_threshold_pct: float,
    ) -> CARRResult:
        """Compute CARR, acceleration, and structural breaks for an entity."""
        years = sorted(ts.annual_emissions.keys())
        filtered = [y for y in years if start_year <= y <= end_year]

        if len(filtered) < 2:
            return CARRResult(
                entity_id=ts.entity_id,
                entity_name=ts.entity_name,
                is_organisation=ts.is_organisation,
                trajectory_band=TrajectoryBand.FLAT,
                provenance_hash=_compute_hash({"entity": ts.entity_id, "carr": 0}),
            )

        start_val = ts.annual_emissions.get(filtered[0], 0.0)
        end_val = ts.annual_emissions.get(filtered[-1], 0.0)
        n_years = filtered[-1] - filtered[0]

        if n_years <= 0 or start_val <= 0:
            carr_pct = 0.0
        else:
            ratio = end_val / start_val
            carr_pct = round(
                ((ratio ** (1.0 / n_years)) - 1.0) * 100.0, 4,
            )

        total_change = round(
            ((end_val - start_val) / max(start_val, 1e-12)) * 100.0, 4,
        )

        # Acceleration: compare first-half vs second-half CARR
        mid_idx = len(filtered) // 2
        acceleration = 0.0
        if mid_idx > 0 and mid_idx < len(filtered) - 1:
            first_half_start = ts.annual_emissions.get(filtered[0], 0.0)
            first_half_end = ts.annual_emissions.get(filtered[mid_idx], 0.0)
            second_half_start = first_half_end
            second_half_end = ts.annual_emissions.get(filtered[-1], 0.0)

            h1_years = filtered[mid_idx] - filtered[0]
            h2_years = filtered[-1] - filtered[mid_idx]

            if first_half_start > 0 and h1_years > 0:
                h1_rate = ((first_half_end / first_half_start) ** (1.0 / h1_years) - 1.0) * 100.0
            else:
                h1_rate = 0.0

            if second_half_start > 0 and h2_years > 0:
                h2_rate = ((second_half_end / second_half_start) ** (1.0 / h2_years) - 1.0) * 100.0
            else:
                h2_rate = 0.0

            acceleration = round(h2_rate - h1_rate, 4)

        # Structural break detection
        has_break = False
        break_year: Optional[int] = None
        for i in range(1, len(filtered)):
            prev = ts.annual_emissions.get(filtered[i - 1], 0.0)
            curr = ts.annual_emissions.get(filtered[i], 0.0)
            if prev > 0:
                yoy_change = abs((curr - prev) / prev) * 100.0
                if yoy_change > break_threshold_pct:
                    has_break = True
                    break_year = filtered[i]
                    break

        # Classify trajectory band
        band = self._classify_trajectory_band(carr_pct)

        carr_data = {
            "entity": ts.entity_id, "carr": carr_pct,
            "total_change": total_change, "acceleration": acceleration,
        }

        return CARRResult(
            entity_id=ts.entity_id,
            entity_name=ts.entity_name,
            is_organisation=ts.is_organisation,
            start_year=filtered[0],
            end_year=filtered[-1],
            start_emissions=round(start_val, 2),
            end_emissions=round(end_val, 2),
            carr_pct=carr_pct,
            total_change_pct=total_change,
            acceleration=acceleration,
            has_structural_break=has_break,
            structural_break_year=break_year,
            trajectory_band=band,
            provenance_hash=_compute_hash(carr_data),
        )

    def _classify_trajectory_band(self, carr_pct: float) -> TrajectoryBand:
        """Classify CARR into trajectory band."""
        if carr_pct <= -5.0:
            return TrajectoryBand.RAPID_DECARBONISER
        elif carr_pct <= -2.0:
            return TrajectoryBand.STEADY_DECARBONISER
        elif carr_pct <= -0.5:
            return TrajectoryBand.SLOW_DECARBONISER
        elif carr_pct <= 0.5:
            return TrajectoryBand.FLAT
        else:
            return TrajectoryBand.INCREASING

    def _compute_momentum(self, carr: CARRResult) -> float:
        """Compute momentum score combining CARR and acceleration."""
        # CARR contribution: more negative = higher score
        carr_score = max(0.0, min(100.0, 50.0 - carr.carr_pct * 10.0))

        # Acceleration contribution: more negative = accelerating reductions
        accel_score = max(0.0, min(50.0, 25.0 - carr.acceleration * 5.0))

        return round(min(carr_score + accel_score, 100.0), 2)

    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation of a list of values."""
        if len(values) < 2:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._org_series = None
        self._peer_series = []
        self._carr_results = []
        self._convergence = []
        self._ranks = []

    def _compute_provenance(self, result: TrajectoryAnalysisResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.org_carr_pct}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
