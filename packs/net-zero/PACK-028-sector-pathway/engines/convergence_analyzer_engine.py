# -*- coding: utf-8 -*-
"""
ConvergenceAnalyzerEngine - PACK-028 Sector Pathway Engine 4
===============================================================

Sector intensity convergence analysis vs. SBTi/IEA benchmarks with
gap quantification, time-to-convergence calculation, required
acceleration rate assessment, risk level determination, and
catch-up scenario modeling.

Analysis Methodology:
    Gap Analysis:
        gap_absolute = current_intensity - pathway_target_intensity
        gap_pct = (current - target) / target * 100

    Time-to-Convergence:
        Given current intensity and trajectory slope,
        years until intensity meets sector pathway.

    Required Acceleration:
        Additional annual reduction (%) needed beyond current
        trajectory to converge with sector pathway by target year.

    Risk Assessment:
        Based on gap magnitude, trajectory direction, time remaining,
        and technology feasibility.

    Catch-up Scenarios:
        3 scenarios: gradual (10yr), moderate (7yr), aggressive (5yr)
        with required annual reduction rates.

Regulatory References:
    - SBTi SDA Methodology (convergence assessment)
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - IEA NZE 2050 Roadmap (milestone tracking)
    - TCFD Transition Risk Assessment (gap analysis)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Gap and convergence calculations are pure arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-028 Sector Pathway
Status:  Production Ready
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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


def _safe_divide(n: Decimal, d: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if d == Decimal("0"):
        return default
    return n / d


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    q = "0." + "0" * places
    return value.quantize(Decimal(q), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConvergenceStatus(str, Enum):
    """Convergence status relative to sector pathway."""
    ON_TRACK = "on_track"
    SLIGHTLY_OFF = "slightly_off"
    SIGNIFICANTLY_OFF = "significantly_off"
    DIVERGING = "diverging"
    CONVERGED = "converged"
    NOT_ASSESSED = "not_assessed"


class RiskLevel(str, Enum):
    """Risk level for convergence gap."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrajectoryDirection(str, Enum):
    """Direction of company intensity trajectory."""
    DECLINING = "declining"
    STABLE = "stable"
    INCREASING = "increasing"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class CatchUpScenarioType(str, Enum):
    """Catch-up scenario aggressiveness."""
    GRADUAL = "gradual"       # 10-year catch-up
    MODERATE = "moderate"     # 7-year catch-up
    AGGRESSIVE = "aggressive"  # 5-year catch-up


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class HistoricalIntensityPoint(BaseModel):
    """Historical intensity data point.

    Attributes:
        year: Reporting year.
        intensity: Emission intensity value.
        emissions_tco2e: Absolute emissions.
        activity_value: Activity level.
    """
    year: int = Field(..., ge=2010, le=2035)
    intensity: Decimal = Field(..., ge=Decimal("0"))
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    activity_value: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))


class PathwayTargetPoint(BaseModel):
    """Sector pathway target at a given year.

    Attributes:
        year: Target year.
        target_intensity: Required sector intensity.
        scenario: Climate scenario for this target.
    """
    year: int = Field(..., ge=2020, le=2070)
    target_intensity: Decimal = Field(..., ge=Decimal("0"))
    scenario: str = Field(default="nze")


class ConvergenceInput(BaseModel):
    """Input for convergence analysis.

    Attributes:
        entity_name: Entity name.
        sector: Sector classification.
        intensity_unit: Intensity metric unit string.
        historical_data: Historical intensity trajectory (2+ years).
        pathway_targets: Sector pathway targets (from Engine 3).
        base_year: Base year.
        target_year: Target convergence year.
        current_year: Most recent data year.
        include_catch_up_scenarios: Generate catch-up scenarios.
        include_milestone_analysis: Analyze milestone compliance.
        include_risk_assessment: Perform risk assessment.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    sector: str = Field(
        ..., min_length=1, max_length=100, description="Sector"
    )
    intensity_unit: str = Field(
        default="", max_length=50, description="Intensity unit"
    )
    historical_data: List[HistoricalIntensityPoint] = Field(
        ..., min_length=1, description="Historical intensity data"
    )
    pathway_targets: List[PathwayTargetPoint] = Field(
        ..., min_length=1, description="Pathway targets"
    )
    base_year: int = Field(
        default=2019, ge=2010, le=2030, description="Base year"
    )
    target_year: int = Field(
        default=2050, ge=2030, le=2070, description="Target year"
    )
    current_year: int = Field(
        default=2024, ge=2015, le=2035, description="Current year"
    )
    include_catch_up_scenarios: bool = Field(
        default=True, description="Generate catch-up scenarios"
    )
    include_milestone_analysis: bool = Field(
        default=True, description="Milestone analysis"
    )
    include_risk_assessment: bool = Field(
        default=True, description="Risk assessment"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class GapAnalysisPoint(BaseModel):
    """Gap analysis at a specific year.

    Attributes:
        year: Analysis year.
        company_intensity: Company's actual/projected intensity.
        pathway_target: Sector pathway target.
        gap_absolute: Absolute gap (company - target).
        gap_pct: Gap as percentage of target.
        status: Convergence status at this year.
    """
    year: int = Field(default=0)
    company_intensity: Decimal = Field(default=Decimal("0"))
    pathway_target: Decimal = Field(default=Decimal("0"))
    gap_absolute: Decimal = Field(default=Decimal("0"))
    gap_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=ConvergenceStatus.NOT_ASSESSED.value)


class TimeToConvergence(BaseModel):
    """Time-to-convergence calculation result.

    Attributes:
        years_to_converge: Estimated years until pathway alignment.
        convergence_year: Projected convergence year.
        achievable: Whether convergence is achievable by target year.
        current_annual_reduction_pct: Current annual reduction rate.
        required_annual_reduction_pct: Required rate for convergence.
        acceleration_needed_pct: Additional reduction rate needed.
    """
    years_to_converge: int = Field(default=0)
    convergence_year: int = Field(default=0)
    achievable: bool = Field(default=False)
    current_annual_reduction_pct: Decimal = Field(default=Decimal("0"))
    required_annual_reduction_pct: Decimal = Field(default=Decimal("0"))
    acceleration_needed_pct: Decimal = Field(default=Decimal("0"))


class CatchUpScenario(BaseModel):
    """A catch-up scenario for closing the convergence gap.

    Attributes:
        scenario_type: Catch-up aggressiveness.
        catch_up_years: Years to close the gap.
        start_year: When catch-up begins.
        convergence_year: When pathway alignment is achieved.
        required_annual_reduction_pct: Required annual reduction.
        front_loaded: Whether reductions are front-loaded.
        year_by_year_targets: Year-by-year intensity targets.
        feasibility: Feasibility assessment (low/medium/high).
        key_actions: Key actions required.
    """
    scenario_type: str = Field(default="")
    catch_up_years: int = Field(default=0)
    start_year: int = Field(default=0)
    convergence_year: int = Field(default=0)
    required_annual_reduction_pct: Decimal = Field(default=Decimal("0"))
    front_loaded: bool = Field(default=False)
    year_by_year_targets: Dict[int, Decimal] = Field(default_factory=dict)
    feasibility: str = Field(default="medium")
    key_actions: List[str] = Field(default_factory=list)


class MilestoneCheck(BaseModel):
    """Milestone compliance check.

    Attributes:
        milestone_year: Milestone year (2025, 2030, etc.).
        pathway_target: Required intensity at milestone.
        projected_intensity: Company's projected intensity.
        on_track: Whether company is on track.
        gap_pct: Gap as percentage of target.
    """
    milestone_year: int = Field(default=0)
    pathway_target: Decimal = Field(default=Decimal("0"))
    projected_intensity: Decimal = Field(default=Decimal("0"))
    on_track: bool = Field(default=False)
    gap_pct: Decimal = Field(default=Decimal("0"))


class RiskAssessment(BaseModel):
    """Convergence risk assessment.

    Attributes:
        overall_risk: Overall risk level.
        gap_risk: Risk from current gap magnitude.
        trajectory_risk: Risk from trajectory direction.
        time_risk: Risk from remaining time.
        technology_risk: Risk from technology feasibility.
        risk_factors: Detailed risk factors.
        mitigation_actions: Recommended mitigation actions.
    """
    overall_risk: str = Field(default=RiskLevel.MEDIUM.value)
    gap_risk: str = Field(default=RiskLevel.MEDIUM.value)
    trajectory_risk: str = Field(default=RiskLevel.MEDIUM.value)
    time_risk: str = Field(default=RiskLevel.MEDIUM.value)
    technology_risk: str = Field(default=RiskLevel.MEDIUM.value)
    risk_factors: List[str] = Field(default_factory=list)
    mitigation_actions: List[str] = Field(default_factory=list)


class ConvergenceResult(BaseModel):
    """Complete convergence analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        sector: Sector.
        intensity_unit: Intensity unit.
        convergence_status: Overall convergence status.
        current_intensity: Most recent intensity.
        current_pathway_target: Current year pathway target.
        current_gap_absolute: Current absolute gap.
        current_gap_pct: Current gap percentage.
        trajectory_direction: Historical trajectory direction.
        gap_analysis: Year-by-year gap analysis.
        time_to_convergence: Time-to-convergence calculation.
        catch_up_scenarios: Catch-up scenarios (if requested).
        milestone_checks: Milestone compliance checks.
        risk_assessment: Risk assessment (if requested).
        recommendations: Recommendations.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    sector: str = Field(default="")
    intensity_unit: str = Field(default="")
    convergence_status: str = Field(default=ConvergenceStatus.NOT_ASSESSED.value)
    current_intensity: Decimal = Field(default=Decimal("0"))
    current_pathway_target: Decimal = Field(default=Decimal("0"))
    current_gap_absolute: Decimal = Field(default=Decimal("0"))
    current_gap_pct: Decimal = Field(default=Decimal("0"))
    trajectory_direction: str = Field(
        default=TrajectoryDirection.INSUFFICIENT_DATA.value
    )
    gap_analysis: List[GapAnalysisPoint] = Field(default_factory=list)
    time_to_convergence: Optional[TimeToConvergence] = Field(default=None)
    catch_up_scenarios: List[CatchUpScenario] = Field(default_factory=list)
    milestone_checks: List[MilestoneCheck] = Field(default_factory=list)
    risk_assessment: Optional[RiskAssessment] = Field(default=None)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ConvergenceAnalyzerEngine:
    """Sector intensity convergence analysis engine.

    Analyzes the gap between a company's current intensity trajectory
    and the sector decarbonization pathway, calculates time-to-convergence,
    models catch-up scenarios, and assesses risk levels.

    All calculations use Decimal arithmetic. No LLM in any path.

    Usage::

        engine = ConvergenceAnalyzerEngine()
        result = engine.calculate(convergence_input)
        print(f"Status: {result.convergence_status}")
        print(f"Gap: {result.current_gap_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: ConvergenceInput) -> ConvergenceResult:
        """Run complete convergence analysis.

        Args:
            data: Validated convergence input.

        Returns:
            ConvergenceResult with gap analysis, convergence timing,
            catch-up scenarios, and risk assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Convergence analysis: entity=%s, sector=%s, years=%d",
            data.entity_name, data.sector,
            len(data.historical_data),
        )

        # Step 1: Determine trajectory direction
        sorted_hist = sorted(data.historical_data, key=lambda p: p.year)
        trajectory_dir = self._assess_trajectory(sorted_hist)

        # Step 2: Get current intensity
        current_intensity = sorted_hist[-1].intensity if sorted_hist else Decimal("0")
        current_year = sorted_hist[-1].year if sorted_hist else data.current_year

        # Step 3: Get current pathway target
        current_target = self._interpolate_target(
            data.pathway_targets, current_year
        )

        # Step 4: Current gap
        gap_abs = current_intensity - current_target
        gap_pct = _safe_pct(gap_abs, current_target) if current_target > Decimal("0") else Decimal("0")

        # Step 5: Convergence status
        conv_status = self._determine_status(gap_pct, trajectory_dir)

        # Step 6: Gap analysis across years
        gap_analysis = self._build_gap_analysis(
            data, sorted_hist, trajectory_dir
        )

        # Step 7: Time-to-convergence
        ttc = self._calculate_time_to_convergence(
            data, sorted_hist, current_intensity, current_year
        )

        # Step 8: Catch-up scenarios
        catch_up: List[CatchUpScenario] = []
        if data.include_catch_up_scenarios and gap_abs > Decimal("0"):
            catch_up = self._generate_catch_up_scenarios(
                data, current_intensity, current_year
            )

        # Step 9: Milestone checks
        milestone_checks: List[MilestoneCheck] = []
        if data.include_milestone_analysis:
            milestone_checks = self._check_milestones(
                data, sorted_hist, trajectory_dir
            )

        # Step 10: Risk assessment
        risk: Optional[RiskAssessment] = None
        if data.include_risk_assessment:
            risk = self._assess_risk(
                data, gap_pct, trajectory_dir, ttc, current_year
            )

        # Step 11: Recommendations
        recommendations = self._generate_recommendations(
            data, conv_status, gap_pct, trajectory_dir, ttc, risk
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ConvergenceResult(
            entity_name=data.entity_name,
            sector=data.sector,
            intensity_unit=data.intensity_unit,
            convergence_status=conv_status,
            current_intensity=_round_val(current_intensity),
            current_pathway_target=_round_val(current_target),
            current_gap_absolute=_round_val(gap_abs),
            current_gap_pct=_round_val(gap_pct, 2),
            trajectory_direction=trajectory_dir,
            gap_analysis=gap_analysis,
            time_to_convergence=ttc,
            catch_up_scenarios=catch_up,
            milestone_checks=milestone_checks,
            risk_assessment=risk,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Convergence complete: entity=%s, status=%s, gap=%.2f%%",
            data.entity_name, conv_status, float(gap_pct),
        )
        return result

    # ------------------------------------------------------------------ #
    # Trajectory Assessment                                                #
    # ------------------------------------------------------------------ #

    def _assess_trajectory(
        self,
        sorted_hist: List[HistoricalIntensityPoint],
    ) -> str:
        """Assess direction of historical intensity trajectory."""
        if len(sorted_hist) < 2:
            return TrajectoryDirection.INSUFFICIENT_DATA.value

        # Calculate year-over-year changes
        changes: List[Decimal] = []
        for i in range(1, len(sorted_hist)):
            prev = sorted_hist[i - 1].intensity
            curr = sorted_hist[i].intensity
            if prev > Decimal("0"):
                change = _safe_pct(curr - prev, prev)
                changes.append(change)

        if not changes:
            return TrajectoryDirection.INSUFFICIENT_DATA.value

        avg_change = sum(changes) / _decimal(len(changes))

        # Check for volatility
        if len(changes) >= 3:
            pos_count = sum(1 for c in changes if c > Decimal("2"))
            neg_count = sum(1 for c in changes if c < Decimal("-2"))
            if pos_count >= 1 and neg_count >= 1 and (pos_count + neg_count) / len(changes) > 0.5:
                return TrajectoryDirection.VOLATILE.value

        if avg_change < Decimal("-1"):
            return TrajectoryDirection.DECLINING.value
        elif avg_change > Decimal("1"):
            return TrajectoryDirection.INCREASING.value
        else:
            return TrajectoryDirection.STABLE.value

    # ------------------------------------------------------------------ #
    # Pathway Target Interpolation                                         #
    # ------------------------------------------------------------------ #

    def _interpolate_target(
        self,
        targets: List[PathwayTargetPoint],
        year: int,
    ) -> Decimal:
        """Interpolate pathway target for a given year."""
        sorted_t = sorted(targets, key=lambda t: t.year)
        if not sorted_t:
            return Decimal("0")

        # Exact match
        for t in sorted_t:
            if t.year == year:
                return t.target_intensity

        # Clamp
        if year <= sorted_t[0].year:
            return sorted_t[0].target_intensity
        if year >= sorted_t[-1].year:
            return sorted_t[-1].target_intensity

        # Interpolate
        for i in range(len(sorted_t) - 1):
            if sorted_t[i].year <= year <= sorted_t[i + 1].year:
                frac = _decimal(year - sorted_t[i].year) / _decimal(
                    sorted_t[i + 1].year - sorted_t[i].year
                )
                return sorted_t[i].target_intensity + (
                    sorted_t[i + 1].target_intensity
                    - sorted_t[i].target_intensity
                ) * frac

        return Decimal("0")

    # ------------------------------------------------------------------ #
    # Gap Analysis                                                         #
    # ------------------------------------------------------------------ #

    def _build_gap_analysis(
        self,
        data: ConvergenceInput,
        sorted_hist: List[HistoricalIntensityPoint],
        trajectory_dir: str,
    ) -> List[GapAnalysisPoint]:
        """Build year-by-year gap analysis."""
        gap_points: List[GapAnalysisPoint] = []

        # Project forward from current trajectory
        if len(sorted_hist) >= 2:
            current = sorted_hist[-1]
            prev = sorted_hist[-2]
            years_diff = current.year - prev.year
            if years_diff > 0 and prev.intensity > Decimal("0"):
                annual_change = (current.intensity - prev.intensity) / _decimal(years_diff)
            else:
                annual_change = Decimal("0")
        else:
            current = sorted_hist[-1] if sorted_hist else None
            annual_change = Decimal("0")

        # Analysis years
        analysis_years = set()
        for h in sorted_hist:
            analysis_years.add(h.year)
        for t in data.pathway_targets:
            analysis_years.add(t.year)
        for milestone in [2025, 2030, 2035, 2040, 2045, 2050]:
            if data.base_year <= milestone <= data.target_year:
                analysis_years.add(milestone)

        hist_map = {h.year: h.intensity for h in sorted_hist}

        for year in sorted(analysis_years):
            # Company intensity (actual or projected)
            if year in hist_map:
                company_int = hist_map[year]
            elif current is not None and year > current.year:
                # Simple linear projection
                years_ahead = year - current.year
                company_int = max(
                    current.intensity + annual_change * _decimal(years_ahead),
                    Decimal("0")
                )
            else:
                continue

            # Pathway target
            pathway_t = self._interpolate_target(data.pathway_targets, year)

            gap_abs = company_int - pathway_t
            gap_pct_val = _safe_pct(gap_abs, pathway_t) if pathway_t > Decimal("0") else Decimal("0")

            # Status
            if gap_pct_val <= Decimal("-5"):
                status = ConvergenceStatus.CONVERGED.value
            elif gap_pct_val <= Decimal("5"):
                status = ConvergenceStatus.ON_TRACK.value
            elif gap_pct_val <= Decimal("20"):
                status = ConvergenceStatus.SLIGHTLY_OFF.value
            elif gap_pct_val <= Decimal("50"):
                status = ConvergenceStatus.SIGNIFICANTLY_OFF.value
            else:
                status = ConvergenceStatus.DIVERGING.value

            gap_points.append(GapAnalysisPoint(
                year=year,
                company_intensity=_round_val(company_int),
                pathway_target=_round_val(pathway_t),
                gap_absolute=_round_val(gap_abs),
                gap_pct=_round_val(gap_pct_val, 2),
                status=status,
            ))

        return gap_points

    # ------------------------------------------------------------------ #
    # Time-to-Convergence                                                  #
    # ------------------------------------------------------------------ #

    def _calculate_time_to_convergence(
        self,
        data: ConvergenceInput,
        sorted_hist: List[HistoricalIntensityPoint],
        current_intensity: Decimal,
        current_year: int,
    ) -> TimeToConvergence:
        """Calculate time-to-convergence with current trajectory."""
        # Current annual reduction rate
        current_rate = Decimal("0")
        if len(sorted_hist) >= 2:
            first = sorted_hist[0]
            last = sorted_hist[-1]
            years_span = _decimal(last.year - first.year)
            if years_span > Decimal("0") and first.intensity > Decimal("0"):
                total_change = _safe_pct(
                    first.intensity - last.intensity,
                    first.intensity,
                )
                current_rate = _safe_divide(total_change, years_span)

        # Target at target year
        target_intensity = self._interpolate_target(
            data.pathway_targets, data.target_year
        )

        # Required reduction
        if current_intensity > target_intensity and current_intensity > Decimal("0"):
            required_total = _safe_pct(
                current_intensity - target_intensity,
                current_intensity,
            )
            years_remaining = _decimal(data.target_year - current_year)
            required_rate = _safe_divide(required_total, years_remaining)
        else:
            required_rate = Decimal("0")
            required_total = Decimal("0")

        # Time to converge at current rate
        years_to_converge = 0
        achievable = False
        convergence_year = 0

        if current_rate > Decimal("0") and current_intensity > target_intensity:
            # Simple estimate: gap / rate
            gap_pct_total = _safe_pct(
                current_intensity - target_intensity,
                current_intensity,
            )
            if current_rate > Decimal("0"):
                years_dec = _safe_divide(gap_pct_total, current_rate)
                years_to_converge = int(float(years_dec)) + 1
                convergence_year = current_year + years_to_converge
                achievable = convergence_year <= data.target_year
        elif current_intensity <= target_intensity:
            # Already converged
            years_to_converge = 0
            convergence_year = current_year
            achievable = True

        acceleration = max(required_rate - current_rate, Decimal("0"))

        return TimeToConvergence(
            years_to_converge=years_to_converge,
            convergence_year=convergence_year,
            achievable=achievable,
            current_annual_reduction_pct=_round_val(current_rate, 3),
            required_annual_reduction_pct=_round_val(required_rate, 3),
            acceleration_needed_pct=_round_val(acceleration, 3),
        )

    # ------------------------------------------------------------------ #
    # Catch-Up Scenarios                                                   #
    # ------------------------------------------------------------------ #

    def _generate_catch_up_scenarios(
        self,
        data: ConvergenceInput,
        current_intensity: Decimal,
        current_year: int,
    ) -> List[CatchUpScenario]:
        """Generate 3 catch-up scenarios (gradual, moderate, aggressive)."""
        scenarios: List[CatchUpScenario] = []

        configs = [
            (CatchUpScenarioType.GRADUAL, 10, False, "medium",
             ["Energy efficiency improvements", "Renewable procurement",
              "Gradual technology transition"]),
            (CatchUpScenarioType.MODERATE, 7, False, "medium",
             ["Accelerated renewable deployment", "Technology pilots",
              "Process optimisation", "Supply chain engagement"]),
            (CatchUpScenarioType.AGGRESSIVE, 5, True, "low",
             ["Rapid technology deployment", "Major CapEx investment",
              "Full electrification", "CCS deployment",
              "Hydrogen integration"]),
        ]

        for sc_type, years, front_loaded, feasibility, actions in configs:
            catch_up_year = min(current_year + years, data.target_year)
            target_at_catch_up = self._interpolate_target(
                data.pathway_targets, catch_up_year
            )

            if current_intensity <= target_at_catch_up:
                # Already meeting target
                scenarios.append(CatchUpScenario(
                    scenario_type=sc_type.value,
                    catch_up_years=0,
                    start_year=current_year,
                    convergence_year=current_year,
                    required_annual_reduction_pct=Decimal("0"),
                    front_loaded=False,
                    year_by_year_targets={},
                    feasibility="high",
                    key_actions=["Maintain current trajectory."],
                ))
                continue

            total_reduction = _safe_pct(
                current_intensity - target_at_catch_up,
                current_intensity,
            )
            annual_rate = _safe_divide(total_reduction, _decimal(years))

            # Year-by-year targets
            yby: Dict[int, Decimal] = {}
            for y in range(current_year, catch_up_year + 1):
                elapsed = y - current_year
                if front_loaded and years > 0:
                    # Front-loaded: more reduction in early years
                    frac = _decimal(elapsed) / _decimal(years)
                    # Use sqrt for front-loading
                    try:
                        adj_frac = _decimal(math.sqrt(float(frac)))
                    except (ValueError, OverflowError):
                        adj_frac = frac
                    intensity = current_intensity - (
                        current_intensity - target_at_catch_up
                    ) * adj_frac
                else:
                    # Linear
                    frac = _safe_divide(
                        _decimal(elapsed), _decimal(years)
                    )
                    intensity = current_intensity - (
                        current_intensity - target_at_catch_up
                    ) * frac
                yby[y] = _round_val(max(intensity, Decimal("0")))

            scenarios.append(CatchUpScenario(
                scenario_type=sc_type.value,
                catch_up_years=years,
                start_year=current_year,
                convergence_year=catch_up_year,
                required_annual_reduction_pct=_round_val(annual_rate, 3),
                front_loaded=front_loaded,
                year_by_year_targets=yby,
                feasibility=feasibility,
                key_actions=actions,
            ))

        return scenarios

    # ------------------------------------------------------------------ #
    # Milestone Checks                                                     #
    # ------------------------------------------------------------------ #

    def _check_milestones(
        self,
        data: ConvergenceInput,
        sorted_hist: List[HistoricalIntensityPoint],
        trajectory_dir: str,
    ) -> List[MilestoneCheck]:
        """Check compliance with milestone years."""
        checks: List[MilestoneCheck] = []
        milestones = [2025, 2030, 2035, 2040, 2045, 2050]

        # Project intensity forward
        current = sorted_hist[-1] if sorted_hist else None
        if current and len(sorted_hist) >= 2:
            prev = sorted_hist[-2]
            years_d = current.year - prev.year
            if years_d > 0 and prev.intensity > Decimal("0"):
                annual_change = (current.intensity - prev.intensity) / _decimal(years_d)
            else:
                annual_change = Decimal("0")
        else:
            annual_change = Decimal("0")

        hist_map = {h.year: h.intensity for h in sorted_hist}

        for milestone in milestones:
            if milestone < data.base_year or milestone > data.target_year:
                continue

            # Get target
            target = self._interpolate_target(data.pathway_targets, milestone)

            # Get actual/projected intensity
            if milestone in hist_map:
                projected = hist_map[milestone]
            elif current and milestone > current.year:
                years_ahead = milestone - current.year
                projected = max(
                    current.intensity + annual_change * _decimal(years_ahead),
                    Decimal("0")
                )
            else:
                continue

            on_track = projected <= target * Decimal("1.10")  # 10% tolerance
            gap = _safe_pct(projected - target, target) if target > Decimal("0") else Decimal("0")

            checks.append(MilestoneCheck(
                milestone_year=milestone,
                pathway_target=_round_val(target),
                projected_intensity=_round_val(projected),
                on_track=on_track,
                gap_pct=_round_val(gap, 2),
            ))

        return checks

    # ------------------------------------------------------------------ #
    # Risk Assessment                                                      #
    # ------------------------------------------------------------------ #

    def _assess_risk(
        self,
        data: ConvergenceInput,
        gap_pct: Decimal,
        trajectory_dir: str,
        ttc: TimeToConvergence,
        current_year: int,
    ) -> RiskAssessment:
        """Assess convergence risk across multiple dimensions."""
        risk_factors: List[str] = []
        mitigation: List[str] = []

        # Gap risk
        if abs(float(gap_pct)) <= 10:
            gap_risk = RiskLevel.LOW.value
        elif abs(float(gap_pct)) <= 30:
            gap_risk = RiskLevel.MEDIUM.value
            risk_factors.append(
                f"Intensity gap of {gap_pct}% vs. sector pathway."
            )
        elif abs(float(gap_pct)) <= 50:
            gap_risk = RiskLevel.HIGH.value
            risk_factors.append(
                f"Significant intensity gap of {gap_pct}% vs. pathway."
            )
            mitigation.append("Deploy transformational technologies.")
        else:
            gap_risk = RiskLevel.CRITICAL.value
            risk_factors.append(
                f"Critical intensity gap of {gap_pct}% vs. pathway."
            )
            mitigation.append("Fundamental strategy overhaul required.")

        # Trajectory risk
        if trajectory_dir == TrajectoryDirection.DECLINING.value:
            trajectory_risk = RiskLevel.LOW.value
        elif trajectory_dir == TrajectoryDirection.STABLE.value:
            trajectory_risk = RiskLevel.MEDIUM.value
            risk_factors.append("Intensity trend is flat -- no progress.")
            mitigation.append("Accelerate emission reduction initiatives.")
        elif trajectory_dir == TrajectoryDirection.INCREASING.value:
            trajectory_risk = RiskLevel.HIGH.value
            risk_factors.append("Intensity is increasing -- diverging from pathway.")
            mitigation.append("Immediately address emission growth drivers.")
        elif trajectory_dir == TrajectoryDirection.VOLATILE.value:
            trajectory_risk = RiskLevel.MEDIUM.value
            risk_factors.append("Volatile intensity trajectory -- inconsistent progress.")
            mitigation.append("Stabilize operations and improve data consistency.")
        else:
            trajectory_risk = RiskLevel.MEDIUM.value

        # Time risk
        years_remaining = data.target_year - current_year
        if years_remaining >= 25:
            time_risk = RiskLevel.LOW.value
        elif years_remaining >= 15:
            time_risk = RiskLevel.MEDIUM.value
        elif years_remaining >= 8:
            time_risk = RiskLevel.HIGH.value
            risk_factors.append(f"Only {years_remaining} years remaining to target.")
            mitigation.append("Accelerate technology deployment timeline.")
        else:
            time_risk = RiskLevel.CRITICAL.value
            risk_factors.append(
                f"Only {years_remaining} years remaining -- "
                f"extremely tight timeline."
            )
            mitigation.append(
                "Emergency decarbonization plan with maximum CapEx allocation."
            )

        # Technology risk (based on acceleration needed)
        if ttc.acceleration_needed_pct <= Decimal("1"):
            tech_risk = RiskLevel.LOW.value
        elif ttc.acceleration_needed_pct <= Decimal("3"):
            tech_risk = RiskLevel.MEDIUM.value
        elif ttc.acceleration_needed_pct <= Decimal("6"):
            tech_risk = RiskLevel.HIGH.value
            risk_factors.append(
                f"Requires {ttc.acceleration_needed_pct}%/yr additional "
                f"reduction -- significant technology acceleration."
            )
            mitigation.append("Invest in breakthrough technologies (hydrogen, CCS).")
        else:
            tech_risk = RiskLevel.CRITICAL.value
            risk_factors.append(
                f"Requires {ttc.acceleration_needed_pct}%/yr additional "
                f"reduction -- may not be technically feasible."
            )
            mitigation.append("Explore sector transition or divestiture options.")

        # Overall risk (worst of individual risks)
        risk_order = [
            RiskLevel.LOW.value, RiskLevel.MEDIUM.value,
            RiskLevel.HIGH.value, RiskLevel.CRITICAL.value,
        ]
        individual_risks = [gap_risk, trajectory_risk, time_risk, tech_risk]
        overall_idx = max(risk_order.index(r) for r in individual_risks)
        overall_risk = risk_order[overall_idx]

        return RiskAssessment(
            overall_risk=overall_risk,
            gap_risk=gap_risk,
            trajectory_risk=trajectory_risk,
            time_risk=time_risk,
            technology_risk=tech_risk,
            risk_factors=risk_factors,
            mitigation_actions=mitigation,
        )

    # ------------------------------------------------------------------ #
    # Status Determination                                                 #
    # ------------------------------------------------------------------ #

    def _determine_status(
        self,
        gap_pct: Decimal,
        trajectory_dir: str,
    ) -> str:
        """Determine overall convergence status."""
        if gap_pct <= Decimal("-5"):
            return ConvergenceStatus.CONVERGED.value
        elif gap_pct <= Decimal("5"):
            return ConvergenceStatus.ON_TRACK.value
        elif gap_pct <= Decimal("20"):
            return ConvergenceStatus.SLIGHTLY_OFF.value
        elif gap_pct <= Decimal("50"):
            if trajectory_dir == TrajectoryDirection.INCREASING.value:
                return ConvergenceStatus.DIVERGING.value
            return ConvergenceStatus.SIGNIFICANTLY_OFF.value
        else:
            return ConvergenceStatus.DIVERGING.value

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: ConvergenceInput,
        status: str,
        gap_pct: Decimal,
        trajectory_dir: str,
        ttc: TimeToConvergence,
        risk: Optional[RiskAssessment],
    ) -> List[str]:
        """Generate convergence improvement recommendations."""
        recs: List[str] = []

        if status == ConvergenceStatus.CONVERGED.value:
            recs.append(
                "On track or ahead of sector pathway. Maintain current "
                "trajectory and consider increasing ambition."
            )
        elif status == ConvergenceStatus.ON_TRACK.value:
            recs.append(
                "Intensity is approximately aligned with sector pathway. "
                "Continue current reduction initiatives."
            )
        elif status == ConvergenceStatus.SLIGHTLY_OFF.value:
            recs.append(
                f"Intensity is {gap_pct}% above pathway. Identify "
                f"quick-win efficiency improvements to close the gap."
            )
        elif status == ConvergenceStatus.SIGNIFICANTLY_OFF.value:
            recs.append(
                f"Intensity is {gap_pct}% above pathway. Requires "
                f"significant technology deployment and investment "
                f"to close the gap."
            )
        elif status == ConvergenceStatus.DIVERGING.value:
            recs.append(
                f"Intensity is diverging from pathway ({gap_pct}% gap). "
                f"Immediate strategic intervention required."
            )

        if not ttc.achievable and ttc.years_to_converge > 0:
            recs.append(
                f"At current trajectory, convergence would take "
                f"{ttc.years_to_converge} years (by {ttc.convergence_year}), "
                f"missing the {data.target_year} target. "
                f"Accelerate by {ttc.acceleration_needed_pct}%/yr."
            )

        if trajectory_dir == TrajectoryDirection.INCREASING.value:
            recs.append(
                "Reverse the increasing intensity trend before "
                "pursuing pathway alignment."
            )

        if len(data.historical_data) < 3:
            recs.append(
                "Provide at least 3 years of historical data for "
                "reliable convergence analysis."
            )

        return recs
