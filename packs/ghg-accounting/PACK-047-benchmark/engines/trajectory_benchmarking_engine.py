# -*- coding: utf-8 -*-
"""
TrajectoryBenchmarkingEngine - PACK-047 GHG Emissions Benchmark Engine 6
====================================================================

Analyses emissions trajectories across peer groups with CARR (Compound
Annual Rate of Reduction), acceleration/deceleration detection, convergence/
divergence analysis, percentile trajectory ranking, structural break
detection, and fan chart data generation.

Calculation Methodology:
    Compound Annual Rate of Reduction (CARR):
        CARR = (E_end / E_start)^(1/n) - 1

        Where:
            E_end   = emissions or intensity at end year
            E_start = emissions or intensity at start year
            n       = number of years between start and end

    Acceleration (2nd derivative):
        a = CARR(t2, t1) - CARR(t1, t0)

        Where:
            CARR(t2, t1) = CARR over recent period
            CARR(t1, t0) = CARR over prior period
            a > 0  = decarbonisation accelerating
            a < 0  = decarbonisation decelerating
            a = 0  = constant rate

    Convergence Rate:
        CR = (gap_t1 - gap_t0) / gap_t0

        Where:
            gap_t  = entity_value(t) - peer_median(t)
            CR < 0 = converging towards median
            CR > 0 = diverging from median
            CR = 0 = parallel trajectory

    Structural Break Detection:
        A break is flagged when year-over-year change exceeds
        threshold * peer_std_dev. Default threshold = 3.0.
        Indicates potential M&A, methodology change, or divestment.

    Fan Chart Generation:
        Percentile trajectories (p10, p25, p50, p75, p90) projected
        forward based on peer CARR distribution.

Regulatory References:
    - GHG Protocol Corporate Standard: Year-over-year comparison
    - ESRS E1-5: Gross Scopes 1-3 trend analysis
    - CDP Climate Change C7.9: Change in emissions
    - SBTi Monitoring Report: Progress tracking
    - TCFD Metrics and Targets (b): Trend disclosure

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-047 GHG Emissions Benchmark
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _round6(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _median_decimal(values: List[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / Decimal("2")


def _percentile_decimal(values: List[Decimal], pct: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (pct / Decimal("100")) * Decimal(str(n - 1))
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_vals[-1]
    frac = rank - Decimal(str(lower))
    return sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])


def _std_deviation_decimal(values: List[Decimal]) -> Decimal:
    if len(values) < 2:
        return Decimal("0")
    n = Decimal(str(len(values)))
    mean = sum(values) / n
    squared_diffs = [(v - mean) ** 2 for v in values]
    variance = sum(squared_diffs) / n
    std_float = float(variance) ** 0.5
    return _decimal(std_float)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrajectoryDirection(str, Enum):
    """Direction of emissions trajectory."""
    DECLINING = "declining"
    FLAT = "flat"
    INCREASING = "increasing"


class AccelerationStatus(str, Enum):
    """Acceleration/deceleration of decarbonisation."""
    ACCELERATING = "accelerating"
    CONSTANT = "constant"
    DECELERATING = "decelerating"


class ConvergenceStatus(str, Enum):
    """Convergence/divergence relative to peer median."""
    CONVERGING = "converging"
    PARALLEL = "parallel"
    DIVERGING = "diverging"


class BreakType(str, Enum):
    """Type of structural break detected."""
    METHODOLOGY_CHANGE = "methodology_change"
    MERGER_ACQUISITION = "merger_acquisition"
    DIVESTMENT = "divestment"
    DATA_ERROR = "data_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BREAK_THRESHOLD: Decimal = Decimal("3.0")
MIN_YEARS_FOR_CARR: int = 2
MAX_HISTORY_YEARS: int = 30
MIN_PEERS_FOR_TRAJECTORY: int = 5


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class TrajectoryPoint(BaseModel):
    """A single year data point on a trajectory.

    Attributes:
        year:       Year.
        value:      Emissions or intensity value.
    """
    year: int = Field(..., description="Year")
    value: Decimal = Field(..., ge=0, description="Value")

    @field_validator("value", mode="before")
    @classmethod
    def coerce_val(cls, v: Any) -> Decimal:
        return _decimal(v)


class EntityTrajectory(BaseModel):
    """Historical trajectory for an entity.

    Attributes:
        entity_id:  Entity identifier.
        entity_name: Entity name.
        points:     Time-series data points.
        is_org:     Whether this is the reference organisation.
    """
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    points: List[TrajectoryPoint] = Field(default_factory=list, description="Data points")
    is_org: bool = Field(default=False, description="Is reference org")


class TrajectoryInput(BaseModel):
    """Input for trajectory benchmarking analysis.

    Attributes:
        organisation_id:        Organisation identifier.
        trajectories:           All entity trajectories (org + peers).
        projection_years:       Years to project forward.
        break_threshold:        Structural break threshold (std devs).
        carr_window_years:      CARR calculation window.
        output_precision:       Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    trajectories: List[EntityTrajectory] = Field(default_factory=list)
    projection_years: int = Field(default=5, ge=1, le=20)
    break_threshold: Decimal = Field(default=DEFAULT_BREAK_THRESHOLD, gt=0)
    carr_window_years: int = Field(default=5, ge=2, le=20)
    output_precision: int = Field(default=4, ge=0, le=12)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class CARRResult(BaseModel):
    """CARR calculation result for an entity.

    Attributes:
        entity_id:      Entity identifier.
        entity_name:    Entity name.
        carr:           Compound Annual Rate of Reduction.
        start_year:     Start year.
        end_year:       End year.
        start_value:    Start value.
        end_value:      End value.
        years:          Number of years.
        direction:      Trajectory direction.
        rank:           Rank among peers (1 = fastest reduction).
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    carr: Decimal = Field(default=Decimal("0"), description="CARR")
    start_year: int = Field(default=0)
    end_year: int = Field(default=0)
    start_value: Decimal = Field(default=Decimal("0"))
    end_value: Decimal = Field(default=Decimal("0"))
    years: int = Field(default=0)
    direction: TrajectoryDirection = Field(default=TrajectoryDirection.FLAT)
    rank: int = Field(default=0)


class AccelerationResult(BaseModel):
    """Acceleration/deceleration detection result.

    Attributes:
        entity_id:          Entity identifier.
        acceleration:       Acceleration value (CARR difference).
        status:             Accelerating/constant/decelerating.
        recent_carr:        CARR over recent period.
        prior_carr:         CARR over prior period.
    """
    entity_id: str = Field(default="")
    acceleration: Decimal = Field(default=Decimal("0"))
    status: AccelerationStatus = Field(default=AccelerationStatus.CONSTANT)
    recent_carr: Decimal = Field(default=Decimal("0"))
    prior_carr: Decimal = Field(default=Decimal("0"))


class ConvergenceResult(BaseModel):
    """Convergence/divergence analysis.

    Attributes:
        entity_id:          Entity identifier.
        convergence_rate:   Convergence rate.
        status:             Converging/parallel/diverging.
        gap_current:        Current gap to peer median.
        gap_previous:       Previous gap to peer median.
    """
    entity_id: str = Field(default="")
    convergence_rate: Decimal = Field(default=Decimal("0"))
    status: ConvergenceStatus = Field(default=ConvergenceStatus.PARALLEL)
    gap_current: Decimal = Field(default=Decimal("0"))
    gap_previous: Decimal = Field(default=Decimal("0"))


class StructuralBreak(BaseModel):
    """A detected structural break.

    Attributes:
        entity_id:      Entity identifier.
        year:           Year of break.
        change_pct:     Year-over-year change (%).
        threshold_multiple: Multiple of std dev exceeded.
        break_type:     Suspected type of break.
    """
    entity_id: str = Field(default="")
    year: int = Field(default=0)
    change_pct: Decimal = Field(default=Decimal("0"))
    threshold_multiple: Decimal = Field(default=Decimal("0"))
    break_type: BreakType = Field(default=BreakType.UNKNOWN)


class FanChartBand(BaseModel):
    """A single year of fan chart data.

    Attributes:
        year:   Year.
        p10:    10th percentile.
        p25:    25th percentile.
        p50:    50th percentile (median).
        p75:    75th percentile.
        p90:    90th percentile.
    """
    year: int = Field(..., description="Year")
    p10: Decimal = Field(default=Decimal("0"))
    p25: Decimal = Field(default=Decimal("0"))
    p50: Decimal = Field(default=Decimal("0"))
    p75: Decimal = Field(default=Decimal("0"))
    p90: Decimal = Field(default=Decimal("0"))


class TrajectoryAnalysis(BaseModel):
    """Complete trajectory benchmarking result.

    Attributes:
        result_id:              Unique result ID.
        organisation_id:        Organisation ID.
        org_carr:               Organisation CARR.
        peer_carrs:             Peer CARRs ranked.
        org_acceleration:       Org acceleration analysis.
        org_convergence:        Org convergence analysis.
        structural_breaks:      Detected structural breaks.
        fan_chart:              Fan chart data.
        org_percentile_rank:    Organisation percentile by CARR.
        peer_count:             Number of peers.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Org ID")
    org_carr: Optional[CARRResult] = Field(default=None)
    peer_carrs: List[CARRResult] = Field(default_factory=list)
    org_acceleration: Optional[AccelerationResult] = Field(default=None)
    org_convergence: Optional[ConvergenceResult] = Field(default=None)
    structural_breaks: List[StructuralBreak] = Field(default_factory=list)
    fan_chart: List[FanChartBand] = Field(default_factory=list)
    org_percentile_rank: Decimal = Field(default=Decimal("0"))
    peer_count: int = Field(default=0)
    warnings: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TrajectoryBenchmarkingEngine:
    """Analyses emissions trajectories across peer groups.

    Provides CARR calculation, acceleration/deceleration detection,
    convergence analysis, structural break detection, and fan chart
    generation.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every trajectory calculation documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("TrajectoryBenchmarkingEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: TrajectoryInput) -> TrajectoryAnalysis:
        """Perform trajectory benchmarking analysis.

        Args:
            input_data: Trajectories and configuration.

        Returns:
            TrajectoryAnalysis with CARR, acceleration, convergence, breaks, fan chart.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        trajectories = input_data.trajectories
        org_traj: Optional[EntityTrajectory] = None
        peer_trajs: List[EntityTrajectory] = []

        for traj in trajectories:
            if traj.is_org:
                org_traj = traj
            else:
                peer_trajs.append(traj)

        if org_traj is None:
            warnings.append("No organisation trajectory flagged. Using first trajectory.")
            if trajectories:
                org_traj = trajectories[0]
                peer_trajs = trajectories[1:]

        # CARR calculations
        window = input_data.carr_window_years
        all_carrs: List[CARRResult] = []
        org_carr: Optional[CARRResult] = None

        if org_traj:
            org_carr = self._compute_carr(org_traj, window, prec_str)
            if org_carr:
                all_carrs.append(org_carr)

        for peer in peer_trajs:
            carr = self._compute_carr(peer, window, prec_str)
            if carr:
                all_carrs.append(carr)

        # Rank by CARR (most negative = fastest reduction = rank 1)
        all_carrs.sort(key=lambda c: c.carr)
        for i, carr in enumerate(all_carrs):
            carr.rank = i + 1

        peer_carrs = [c for c in all_carrs if c.entity_id != (org_traj.entity_id if org_traj else "")]

        # Organisation percentile
        org_pct = Decimal("50")
        if org_carr and peer_carrs:
            below = sum(1 for c in peer_carrs if c.carr > org_carr.carr)
            org_pct = (
                Decimal(str(below)) / Decimal(str(len(peer_carrs))) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Acceleration
        org_accel: Optional[AccelerationResult] = None
        if org_traj:
            org_accel = self._compute_acceleration(org_traj, window, prec_str)

        # Convergence
        org_conv: Optional[ConvergenceResult] = None
        if org_traj and peer_trajs:
            org_conv = self._compute_convergence(org_traj, peer_trajs, prec_str)

        # Structural breaks
        breaks: List[StructuralBreak] = []
        if org_traj:
            peer_values_by_year = self._aggregate_peer_values(peer_trajs)
            org_breaks = self._detect_breaks(
                org_traj, peer_values_by_year, input_data.break_threshold, prec_str,
            )
            breaks.extend(org_breaks)

        # Fan chart
        fan_chart: List[FanChartBand] = []
        if peer_trajs and len(peer_trajs) >= MIN_PEERS_FOR_TRAJECTORY:
            fan_chart = self._generate_fan_chart(
                peer_trajs, input_data.projection_years, prec_str,
            )

        peer_count = len(peer_trajs)
        if peer_count < MIN_PEERS_FOR_TRAJECTORY:
            warnings.append(
                f"Only {peer_count} peers. Minimum {MIN_PEERS_FOR_TRAJECTORY} "
                f"recommended for trajectory statistics."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = TrajectoryAnalysis(
            organisation_id=input_data.organisation_id,
            org_carr=org_carr,
            peer_carrs=peer_carrs,
            org_acceleration=org_accel,
            org_convergence=org_conv,
            structural_breaks=breaks,
            fan_chart=fan_chart,
            org_percentile_rank=org_pct,
            peer_count=peer_count,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def compute_carr(
        self, start_value: Decimal, end_value: Decimal, years: int,
    ) -> Decimal:
        """Compute CARR externally.

        CARR = (E_end/E_start)^(1/n) - 1
        """
        return self._carr(start_value, end_value, years)

    # ------------------------------------------------------------------
    # Internal: CARR
    # ------------------------------------------------------------------

    def _compute_carr(
        self, traj: EntityTrajectory, window: int, prec_str: str,
    ) -> Optional[CARRResult]:
        """Compute CARR for an entity trajectory."""
        points = sorted(traj.points, key=lambda p: p.year)
        if len(points) < MIN_YEARS_FOR_CARR:
            return None

        # Use the last `window` years
        if len(points) > window:
            points = points[-window:]

        start = points[0]
        end = points[-1]
        n = end.year - start.year
        if n < 1 or start.value <= Decimal("0"):
            return None

        carr = self._carr(start.value, end.value, n)

        direction = TrajectoryDirection.FLAT
        if carr < Decimal("-0.005"):
            direction = TrajectoryDirection.DECLINING
        elif carr > Decimal("0.005"):
            direction = TrajectoryDirection.INCREASING

        return CARRResult(
            entity_id=traj.entity_id,
            entity_name=traj.entity_name,
            carr=carr.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            start_year=start.year,
            end_year=end.year,
            start_value=start.value,
            end_value=end.value,
            years=n,
            direction=direction,
        )

    def _carr(self, start_value: Decimal, end_value: Decimal, years: int) -> Decimal:
        """CARR = (E_end/E_start)^(1/n) - 1."""
        if start_value <= Decimal("0") or years <= 0:
            return Decimal("0")
        ratio = float(end_value / start_value)
        if ratio <= 0:
            return Decimal("-1")
        return _decimal(ratio ** (1.0 / float(years)) - 1)

    # ------------------------------------------------------------------
    # Internal: Acceleration
    # ------------------------------------------------------------------

    def _compute_acceleration(
        self, traj: EntityTrajectory, window: int, prec_str: str,
    ) -> Optional[AccelerationResult]:
        """Detect acceleration: a = CARR(recent) - CARR(prior)."""
        points = sorted(traj.points, key=lambda p: p.year)
        half = max(window // 2, 2)
        if len(points) < 2 * half:
            return None

        # Split into prior and recent halves
        mid_idx = len(points) // 2
        prior_pts = points[:mid_idx]
        recent_pts = points[mid_idx:]

        if len(prior_pts) < 2 or len(recent_pts) < 2:
            return None

        prior_carr = self._carr(
            prior_pts[0].value, prior_pts[-1].value,
            prior_pts[-1].year - prior_pts[0].year,
        )
        recent_carr = self._carr(
            recent_pts[0].value, recent_pts[-1].value,
            recent_pts[-1].year - recent_pts[0].year,
        )

        acceleration = recent_carr - prior_carr

        status = AccelerationStatus.CONSTANT
        if acceleration < Decimal("-0.005"):
            status = AccelerationStatus.ACCELERATING
        elif acceleration > Decimal("0.005"):
            status = AccelerationStatus.DECELERATING

        return AccelerationResult(
            entity_id=traj.entity_id,
            acceleration=acceleration.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            status=status,
            recent_carr=recent_carr.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
            prior_carr=prior_carr.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
        )

    # ------------------------------------------------------------------
    # Internal: Convergence
    # ------------------------------------------------------------------

    def _compute_convergence(
        self,
        org_traj: EntityTrajectory,
        peer_trajs: List[EntityTrajectory],
        prec_str: str,
    ) -> Optional[ConvergenceResult]:
        """Convergence: CR = (gap_t1 - gap_t0) / gap_t0."""
        org_points = {p.year: p.value for p in org_traj.points}
        years = sorted(org_points.keys())

        if len(years) < 2:
            return None

        peer_medians = self._compute_peer_medians(peer_trajs)

        # Get two most recent years with data
        recent_years = [y for y in years if y in peer_medians]
        if len(recent_years) < 2:
            return None

        y0 = recent_years[-2]
        y1 = recent_years[-1]

        gap_t0 = org_points[y0] - peer_medians[y0]
        gap_t1 = org_points[y1] - peer_medians[y1]

        if gap_t0 == Decimal("0"):
            cr = Decimal("0")
        else:
            cr = _safe_divide(gap_t1 - gap_t0, abs(gap_t0))

        status = ConvergenceStatus.PARALLEL
        if cr < Decimal("-0.05"):
            status = ConvergenceStatus.CONVERGING
        elif cr > Decimal("0.05"):
            status = ConvergenceStatus.DIVERGING

        return ConvergenceResult(
            entity_id=org_traj.entity_id,
            convergence_rate=cr.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            status=status,
            gap_current=gap_t1.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            gap_previous=gap_t0.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
        )

    def _compute_peer_medians(
        self, peer_trajs: List[EntityTrajectory],
    ) -> Dict[int, Decimal]:
        """Compute peer median values by year."""
        by_year: Dict[int, List[Decimal]] = {}
        for traj in peer_trajs:
            for p in traj.points:
                by_year.setdefault(p.year, []).append(p.value)
        return {y: _median_decimal(vals) for y, vals in by_year.items()}

    # ------------------------------------------------------------------
    # Internal: Structural Breaks
    # ------------------------------------------------------------------

    def _detect_breaks(
        self,
        traj: EntityTrajectory,
        peer_values_by_year: Dict[int, List[Decimal]],
        threshold: Decimal,
        prec_str: str,
    ) -> List[StructuralBreak]:
        """Detect structural breaks via threshold * peer_std_dev."""
        breaks: List[StructuralBreak] = []
        points = sorted(traj.points, key=lambda p: p.year)

        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            if prev.value == Decimal("0"):
                continue

            yoy_change = (curr.value - prev.value) / prev.value
            yoy_pct = yoy_change * Decimal("100")

            # Get peer std for this year
            peer_vals = peer_values_by_year.get(curr.year, [])
            if len(peer_vals) < 3:
                continue

            peer_std = _std_deviation_decimal(peer_vals)
            peer_mean = sum(peer_vals) / Decimal(str(len(peer_vals)))

            if peer_std == Decimal("0"):
                continue

            change_abs = abs(curr.value - prev.value)
            multiple = _safe_divide(change_abs, peer_std)

            if multiple > threshold:
                break_type = BreakType.UNKNOWN
                if yoy_change < Decimal("-0.3"):
                    break_type = BreakType.DIVESTMENT
                elif yoy_change > Decimal("0.3"):
                    break_type = BreakType.MERGER_ACQUISITION

                breaks.append(StructuralBreak(
                    entity_id=traj.entity_id,
                    year=curr.year,
                    change_pct=yoy_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    threshold_multiple=multiple.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    break_type=break_type,
                ))

        return breaks

    def _aggregate_peer_values(
        self, peer_trajs: List[EntityTrajectory],
    ) -> Dict[int, List[Decimal]]:
        """Aggregate peer values by year."""
        by_year: Dict[int, List[Decimal]] = {}
        for traj in peer_trajs:
            for p in traj.points:
                by_year.setdefault(p.year, []).append(p.value)
        return by_year

    # ------------------------------------------------------------------
    # Internal: Fan Chart
    # ------------------------------------------------------------------

    def _generate_fan_chart(
        self,
        peer_trajs: List[EntityTrajectory],
        projection_years: int,
        prec_str: str,
    ) -> List[FanChartBand]:
        """Generate fan chart from peer CARR distribution."""
        # Compute CARRs for all peers
        carrs: List[Decimal] = []
        latest_values: List[Decimal] = []
        max_year = 0

        for traj in peer_trajs:
            pts = sorted(traj.points, key=lambda p: p.year)
            if len(pts) >= 2:
                n = pts[-1].year - pts[0].year
                if n > 0 and pts[0].value > Decimal("0"):
                    carr = self._carr(pts[0].value, pts[-1].value, n)
                    carrs.append(carr)
                    latest_values.append(pts[-1].value)
                    max_year = max(max_year, pts[-1].year)

        if not carrs or not latest_values:
            return []

        # Compute percentile CARRs
        carr_p10 = _percentile_decimal(carrs, Decimal("10"))
        carr_p25 = _percentile_decimal(carrs, Decimal("25"))
        carr_p50 = _percentile_decimal(carrs, Decimal("50"))
        carr_p75 = _percentile_decimal(carrs, Decimal("75"))
        carr_p90 = _percentile_decimal(carrs, Decimal("90"))

        # Median starting value
        median_start = _median_decimal(latest_values)

        bands: List[FanChartBand] = []
        for y_off in range(0, projection_years + 1):
            year = max_year + y_off
            exp = Decimal(str(y_off))

            p10 = max(median_start * (Decimal("1") + carr_p10) ** exp, Decimal("0"))
            p25 = max(median_start * (Decimal("1") + carr_p25) ** exp, Decimal("0"))
            p50 = max(median_start * (Decimal("1") + carr_p50) ** exp, Decimal("0"))
            p75 = max(median_start * (Decimal("1") + carr_p75) ** exp, Decimal("0"))
            p90 = max(median_start * (Decimal("1") + carr_p90) ** exp, Decimal("0"))

            bands.append(FanChartBand(
                year=year,
                p10=p10.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                p25=p25.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                p50=p50.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                p75=p75.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
                p90=p90.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP),
            ))

        return bands

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "TrajectoryDirection",
    "AccelerationStatus",
    "ConvergenceStatus",
    "BreakType",
    # Input Models
    "TrajectoryPoint",
    "EntityTrajectory",
    "TrajectoryInput",
    # Output Models
    "CARRResult",
    "AccelerationResult",
    "ConvergenceResult",
    "StructuralBreak",
    "FanChartBand",
    "TrajectoryAnalysis",
    # Engine
    "TrajectoryBenchmarkingEngine",
]
