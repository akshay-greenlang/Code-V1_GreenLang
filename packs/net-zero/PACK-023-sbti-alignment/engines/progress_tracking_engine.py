# -*- coding: utf-8 -*-
"""
ProgressTrackingEngine - PACK-023 SBTi Alignment Engine 7
============================================================

Tracks annual progress against validated SBTi targets, providing
on-track / off-track / critical assessment with RAG status, gap
analysis, corrective action identification, trajectory projection,
and carbon budget analysis.

Progress Assessment Framework:
    Organizations with validated SBTi targets must demonstrate annual
    progress toward their near-term and long-term emissions reduction
    commitments.  This engine compares actual emissions trajectories
    against the required linear reduction pathway and flags deviations
    using a three-colour RAG (Red/Amber/Green) system.

RAG Thresholds:
    GREEN  - actual reduction within 5% of required pathway (on track)
    AMBER  - deviation between 5% and 15% of required pathway
    RED    - deviation exceeds 15% of required pathway (critical)

Tracking Statuses:
    ON_TRACK        - GREEN; entity is meeting its reduction commitments
    MINOR_DEVIATION - AMBER; minor corrective actions recommended
    MAJOR_DEVIATION - RED; significant remediation required
    CRITICAL        - RED; target at risk of non-achievement
    NOT_STARTED     - no current-year data available

Key Calculations:
    - Required annual reduction rate (linear pathway)
    - Actual annual reduction rate (observed emissions decline)
    - Gap in tCO2e and as percentage of required
    - Trajectory projection (linear extrapolation to target year)
    - Carbon budget remaining from base year through target year
    - Corrective action identification based on gap severity

Regulatory References:
    - SBTi Corporate Manual V5.3 (2024) - Progress reporting
    - SBTi Corporate Net-Zero Standard V1.3 (2024) - Tracking
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - CDP Climate Change Questionnaire C4 (2024) - Progress
    - CSRD ESRS E1-4 (2023) - GHG reduction progress

Zero-Hallucination:
    - All reductions computed as Decimal arithmetic
    - Trajectory projection uses deterministic linear extrapolation
    - RAG classification uses fixed numeric thresholds
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
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
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely, returning 0 on zero denominator."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RAG threshold constants (percentage deviation from required pathway)
RAG_GREEN_THRESHOLD: Decimal = Decimal("5.00")    # <= 5% deviation
RAG_AMBER_THRESHOLD: Decimal = Decimal("15.00")   # <= 15% deviation
# > 15% deviation -> RED

# Critical threshold: if projected trajectory overshoots target by > 25%
CRITICAL_OVERSHOOT_THRESHOLD: Decimal = Decimal("25.00")

# Budget exhaustion warning threshold
BUDGET_WARNING_THRESHOLD: Decimal = Decimal("80.00")  # 80% of budget consumed


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrackingStatus(str, Enum):
    """Progress tracking status."""
    ON_TRACK = "on_track"
    MINOR_DEVIATION = "minor_deviation"
    MAJOR_DEVIATION = "major_deviation"
    CRITICAL = "critical"
    NOT_STARTED = "not_started"


class RAGStatus(str, Enum):
    """Red/Amber/Green status indicator."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class ActionPriority(str, Enum):
    """Priority level for corrective actions."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class TargetScope(str, Enum):
    """Target scope classification."""
    S1S2 = "s1s2"
    S3 = "s3"
    S1S2S3 = "s1s2s3"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AnnualDataPoint(BaseModel):
    """A single year's emissions data point."""
    year: int = Field(description="Reporting year")
    emissions_tco2e: Decimal = Field(description="Total emissions (tCO2e)")
    scope: TargetScope = Field(default=TargetScope.S1S2, description="Scope")
    is_verified: bool = Field(default=False, description="Third-party verified")
    data_quality: str = Field(default="reported", description="Data quality level")
    notes: str = Field(default="", description="Notes")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emissions_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class TrajectoryPoint(BaseModel):
    """A single point on a projected trajectory."""
    year: int = Field(description="Year")
    required_emissions_tco2e: Decimal = Field(
        description="Required emissions per linear pathway (tCO2e)"
    )
    projected_emissions_tco2e: Decimal = Field(
        description="Projected emissions based on current trend (tCO2e)"
    )
    gap_tco2e: Decimal = Field(
        description="Gap between projected and required (tCO2e)"
    )
    gap_pct: Decimal = Field(
        description="Gap as percentage of required"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("required_emissions_tco2e", "projected_emissions_tco2e",
                     "gap_tco2e", "gap_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class CorrectiveAction(BaseModel):
    """A recommended corrective action."""
    action_id: str = Field(default_factory=_new_uuid, description="Action ID")
    priority: ActionPriority = Field(description="Priority level")
    category: str = Field(description="Action category (e.g. energy, procurement)")
    description: str = Field(description="Action description")
    estimated_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), description="Estimated reduction potential (tCO2e)"
    )
    estimated_timeline_months: int = Field(
        default=0, description="Estimated implementation timeline (months)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("estimated_reduction_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class BudgetAnalysis(BaseModel):
    """Carbon budget analysis for remaining target period."""
    total_budget_tco2e: Decimal = Field(
        description="Total carbon budget from base year to target year (tCO2e)"
    )
    consumed_budget_tco2e: Decimal = Field(
        description="Budget consumed so far (tCO2e)"
    )
    remaining_budget_tco2e: Decimal = Field(
        description="Remaining carbon budget (tCO2e)"
    )
    consumed_pct: Decimal = Field(description="Percentage of budget consumed")
    remaining_years: int = Field(description="Years remaining to target year")
    required_annual_reduction_tco2e: Decimal = Field(
        description="Required annual reduction from now (tCO2e/yr)"
    )
    budget_on_track: bool = Field(description="Whether budget consumption is on track")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_budget_tco2e", "consumed_budget_tco2e",
                     "remaining_budget_tco2e", "consumed_pct",
                     "required_annual_reduction_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ProgressInput(BaseModel):
    """Input data for progress tracking assessment."""
    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment ID")
    entity_name: str = Field(description="Organization name")
    entity_id: str = Field(default_factory=_new_uuid, description="Entity ID")
    scope: TargetScope = Field(default=TargetScope.S1S2, description="Target scope")
    base_year: int = Field(description="Target base year")
    base_year_emissions: Decimal = Field(description="Base year emissions (tCO2e)")
    target_year: int = Field(description="Target year")
    target_year_emissions: Decimal = Field(description="Target year emissions (tCO2e)")
    current_year: int = Field(description="Current reporting year")
    current_year_emissions: Decimal = Field(description="Current year emissions (tCO2e)")
    annual_data: List[AnnualDataPoint] = Field(
        default_factory=list, description="Historical annual data points"
    )
    reduction_target_pct: Decimal = Field(
        default=Decimal("0"), description="Overall reduction target (%)"
    )
    requested_at: datetime = Field(default_factory=_utcnow, description="Request timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("base_year_emissions", "target_year_emissions",
                     "current_year_emissions", "reduction_target_pct",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ProgressResult(BaseModel):
    """Complete progress tracking result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    entity_name: str = Field(default="", description="Organization name")
    entity_id: str = Field(default="", description="Entity ID")
    scope: TargetScope = Field(default=TargetScope.S1S2, description="Scope")
    status: TrackingStatus = Field(description="Overall tracking status")
    rag: RAGStatus = Field(description="RAG status")
    actual_reduction_pct: Decimal = Field(
        description="Actual reduction from base year (%)"
    )
    required_reduction_pct: Decimal = Field(
        description="Required reduction to date per linear pathway (%)"
    )
    gap_tco2e: Decimal = Field(
        description="Gap between actual and required (tCO2e, positive = behind)"
    )
    gap_pct: Decimal = Field(
        description="Gap as percentage of required reduction"
    )
    actual_arr_pct: Decimal = Field(
        description="Actual annualized reduction rate (%/yr)"
    )
    required_arr_pct: Decimal = Field(
        description="Required annualized reduction rate (%/yr)"
    )
    trajectory_projection: List[TrajectoryPoint] = Field(
        default_factory=list, description="Projected trajectory to target year"
    )
    corrective_actions: List[CorrectiveAction] = Field(
        default_factory=list, description="Recommended corrective actions"
    )
    budget_analysis: Optional[BudgetAnalysis] = Field(
        default=None, description="Carbon budget analysis"
    )
    annual_comparison: List[Dict[str, Any]] = Field(
        default_factory=list, description="Year-over-year comparison"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("actual_reduction_pct", "required_reduction_pct",
                     "gap_tco2e", "gap_pct", "actual_arr_pct",
                     "required_arr_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class ProgressTrackingConfig(BaseModel):
    """Configuration for the ProgressTrackingEngine."""
    green_threshold_pct: Decimal = Field(
        default=RAG_GREEN_THRESHOLD,
        description="Maximum deviation (%) for GREEN status",
    )
    amber_threshold_pct: Decimal = Field(
        default=RAG_AMBER_THRESHOLD,
        description="Maximum deviation (%) for AMBER status",
    )
    critical_overshoot_pct: Decimal = Field(
        default=CRITICAL_OVERSHOOT_THRESHOLD,
        description="Overshoot (%) that triggers CRITICAL status",
    )
    budget_warning_pct: Decimal = Field(
        default=BUDGET_WARNING_THRESHOLD,
        description="Budget consumption (%) that triggers warning",
    )
    score_precision: int = Field(
        default=4, description="Decimal places for calculated values"
    )
    projection_years: int = Field(
        default=0, description="Years to project (0 = project to target year)"
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

AnnualDataPoint.model_rebuild()
TrajectoryPoint.model_rebuild()
CorrectiveAction.model_rebuild()
BudgetAnalysis.model_rebuild()
ProgressInput.model_rebuild()
ProgressResult.model_rebuild()
ProgressTrackingConfig.model_rebuild()


# ---------------------------------------------------------------------------
# ProgressTrackingEngine
# ---------------------------------------------------------------------------


class ProgressTrackingEngine:
    """
    Annual progress tracking engine for SBTi validated targets.

    Evaluates whether an organization is on track to meet its emissions
    reduction commitments by comparing actual emissions against the
    required linear reduction pathway and identifying corrective actions.

    Attributes:
        config: Engine configuration.

    Example:
        >>> engine = ProgressTrackingEngine()
        >>> inp = ProgressInput(
        ...     entity_name="Acme Corp",
        ...     base_year=2019, base_year_emissions=100000,
        ...     target_year=2030, target_year_emissions=58000,
        ...     current_year=2025, current_year_emissions=82000,
        ... )
        >>> result = engine.assess_progress(inp)
        >>> assert result.rag in (RAGStatus.GREEN, RAGStatus.AMBER, RAGStatus.RED)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProgressTrackingEngine.

        Args:
            config: Optional configuration dictionary or ProgressTrackingConfig.
        """
        if config and isinstance(config, dict):
            self.config = ProgressTrackingConfig(**config)
        elif config and isinstance(config, ProgressTrackingConfig):
            self.config = config
        else:
            self.config = ProgressTrackingConfig()

        logger.info("ProgressTrackingEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Core Progress Assessment
    # -------------------------------------------------------------------

    def assess_progress(self, inp: ProgressInput) -> ProgressResult:
        """Assess progress against a validated target.

        This is the primary entry point.  It calculates the gap between
        actual and required reductions, assigns a RAG status, projects
        the trajectory, identifies corrective actions, and performs
        carbon budget analysis.

        Args:
            inp: ProgressInput with base year, target, and current data.

        Returns:
            ProgressResult with complete assessment.

        Raises:
            ValueError: If input data is invalid.
        """
        self._validate_input(inp)

        # Step 1: Calculate required and actual reductions
        required_reduction_pct = self._calculate_required_reduction_pct(inp)
        actual_reduction_pct = self._calculate_actual_reduction_pct(inp)
        required_arr = self._calculate_required_arr(inp)
        actual_arr = self._calculate_actual_arr(inp)

        # Step 2: Calculate gap
        gap_tco2e, gap_pct = self.calculate_gap(inp)

        # Step 3: Determine status and RAG
        rag = self._determine_rag(gap_pct)
        status = self._determine_status(gap_pct, inp)

        # Step 4: Project trajectory
        trajectory = self.project_trajectory(inp)

        # Step 5: Identify corrective actions
        corrective_actions = self.identify_actions(status, gap_tco2e, inp)

        # Step 6: Budget analysis
        budget = self.calculate_budget(inp)

        # Step 7: Annual comparison
        annual_comparison = self.compare_annual_data(inp)

        result = ProgressResult(
            entity_name=inp.entity_name,
            entity_id=inp.entity_id,
            scope=inp.scope,
            status=status,
            rag=rag,
            actual_reduction_pct=actual_reduction_pct,
            required_reduction_pct=required_reduction_pct,
            gap_tco2e=gap_tco2e,
            gap_pct=gap_pct,
            actual_arr_pct=actual_arr,
            required_arr_pct=required_arr,
            trajectory_projection=trajectory,
            corrective_actions=corrective_actions,
            budget_analysis=budget,
            annual_comparison=annual_comparison,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Progress assessment for %s: status=%s, rag=%s, gap=%.2f%%",
            inp.entity_name, status.value, rag.value, float(gap_pct),
        )
        return result

    # -------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------

    def _validate_input(self, inp: ProgressInput) -> None:
        """Validate input data for consistency.

        Args:
            inp: ProgressInput to validate.

        Raises:
            ValueError: If input data is invalid.
        """
        if inp.base_year >= inp.target_year:
            raise ValueError(
                f"Base year ({inp.base_year}) must be before target year ({inp.target_year})"
            )
        if inp.current_year < inp.base_year:
            raise ValueError(
                f"Current year ({inp.current_year}) must be >= base year ({inp.base_year})"
            )
        if inp.base_year_emissions <= Decimal("0"):
            raise ValueError("Base year emissions must be positive")

    # -------------------------------------------------------------------
    # Reduction Calculations
    # -------------------------------------------------------------------

    def _calculate_required_reduction_pct(self, inp: ProgressInput) -> Decimal:
        """Calculate the required reduction percentage to date.

        Uses linear interpolation between base year and target year
        to determine what reduction should have been achieved by now.

        Args:
            inp: Progress input.

        Returns:
            Required reduction percentage.
        """
        total_duration = _decimal(inp.target_year - inp.base_year)
        elapsed = _decimal(inp.current_year - inp.base_year)

        total_reduction_pct = _safe_pct(
            inp.base_year_emissions - inp.target_year_emissions,
            inp.base_year_emissions,
        )

        required = _safe_divide(
            total_reduction_pct * elapsed, total_duration, Decimal("0")
        )
        return _round_val(required, self.config.score_precision)

    def _calculate_actual_reduction_pct(self, inp: ProgressInput) -> Decimal:
        """Calculate the actual reduction percentage achieved.

        Args:
            inp: Progress input.

        Returns:
            Actual reduction percentage from base year.
        """
        if inp.base_year_emissions <= Decimal("0"):
            return Decimal("0")
        actual = _safe_pct(
            inp.base_year_emissions - inp.current_year_emissions,
            inp.base_year_emissions,
        )
        return _round_val(actual, self.config.score_precision)

    def _calculate_required_arr(self, inp: ProgressInput) -> Decimal:
        """Calculate the overall required annual reduction rate.

        Args:
            inp: Progress input.

        Returns:
            Required ARR (%/yr).
        """
        total_duration = _decimal(inp.target_year - inp.base_year)
        if total_duration <= Decimal("0"):
            return Decimal("0")

        total_reduction_pct = _safe_pct(
            inp.base_year_emissions - inp.target_year_emissions,
            inp.base_year_emissions,
        )
        return _round_val(
            _safe_divide(total_reduction_pct, total_duration, Decimal("0")),
            self.config.score_precision,
        )

    def _calculate_actual_arr(self, inp: ProgressInput) -> Decimal:
        """Calculate the actual annual reduction rate observed.

        Args:
            inp: Progress input.

        Returns:
            Actual ARR (%/yr).
        """
        elapsed = _decimal(inp.current_year - inp.base_year)
        if elapsed <= Decimal("0"):
            return Decimal("0")

        actual_reduction_pct = self._calculate_actual_reduction_pct(inp)
        return _round_val(
            _safe_divide(actual_reduction_pct, elapsed, Decimal("0")),
            self.config.score_precision,
        )

    # -------------------------------------------------------------------
    # Gap Calculation
    # -------------------------------------------------------------------

    def calculate_gap(self, inp: ProgressInput) -> Tuple[Decimal, Decimal]:
        """Calculate the gap between actual and required emissions.

        A positive gap means actual emissions are ABOVE the required
        pathway (behind schedule).

        Args:
            inp: Progress input.

        Returns:
            Tuple of (gap_tco2e, gap_pct).
        """
        required_emissions = self._interpolate_required_emissions(inp)
        gap_tco2e = inp.current_year_emissions - required_emissions
        gap_tco2e = _round_val(gap_tco2e, self.config.score_precision)

        gap_pct = Decimal("0")
        if required_emissions > Decimal("0"):
            gap_pct = _safe_pct(gap_tco2e, required_emissions)

        return gap_tco2e, gap_pct

    def _interpolate_required_emissions(self, inp: ProgressInput) -> Decimal:
        """Interpolate required emissions at the current year on linear pathway.

        Args:
            inp: Progress input.

        Returns:
            Required emissions at current year (tCO2e).
        """
        total_duration = _decimal(inp.target_year - inp.base_year)
        elapsed = _decimal(inp.current_year - inp.base_year)

        if total_duration <= Decimal("0"):
            return inp.base_year_emissions

        fraction = _safe_divide(elapsed, total_duration, Decimal("0"))
        total_reduction = inp.base_year_emissions - inp.target_year_emissions
        required_reduction = total_reduction * fraction
        required_emissions = inp.base_year_emissions - required_reduction
        return _round_val(required_emissions, self.config.score_precision)

    # -------------------------------------------------------------------
    # RAG and Status Determination
    # -------------------------------------------------------------------

    def _determine_rag(self, gap_pct: Decimal) -> RAGStatus:
        """Determine RAG status from gap percentage.

        Args:
            gap_pct: Gap as percentage (positive = behind).

        Returns:
            RAGStatus.
        """
        abs_gap = abs(gap_pct)
        if abs_gap <= self.config.green_threshold_pct and gap_pct <= self.config.green_threshold_pct:
            return RAGStatus.GREEN
        elif abs_gap <= self.config.amber_threshold_pct and gap_pct <= self.config.amber_threshold_pct:
            return RAGStatus.AMBER
        else:
            return RAGStatus.RED

    def _determine_status(
        self, gap_pct: Decimal, inp: ProgressInput
    ) -> TrackingStatus:
        """Determine tracking status from gap and context.

        Args:
            gap_pct: Gap as percentage.
            inp: Progress input.

        Returns:
            TrackingStatus.
        """
        if inp.current_year <= inp.base_year:
            return TrackingStatus.NOT_STARTED

        if gap_pct <= self.config.green_threshold_pct:
            return TrackingStatus.ON_TRACK
        elif gap_pct <= self.config.amber_threshold_pct:
            return TrackingStatus.MINOR_DEVIATION
        elif gap_pct <= self.config.critical_overshoot_pct:
            return TrackingStatus.MAJOR_DEVIATION
        else:
            return TrackingStatus.CRITICAL

    # -------------------------------------------------------------------
    # Trajectory Projection
    # -------------------------------------------------------------------

    def project_trajectory(self, inp: ProgressInput) -> List[TrajectoryPoint]:
        """Project emissions trajectory from current year to target year.

        Uses linear extrapolation of the observed trend (base year to
        current year) and compares against the required linear pathway.

        Args:
            inp: Progress input.

        Returns:
            List of TrajectoryPoint for each future year.
        """
        if inp.current_year >= inp.target_year:
            return []

        elapsed = _decimal(inp.current_year - inp.base_year)
        if elapsed <= Decimal("0"):
            return []

        # Actual annual change
        actual_annual_change = _safe_divide(
            inp.current_year_emissions - inp.base_year_emissions,
            elapsed,
            Decimal("0"),
        )

        # Required annual change
        total_duration = _decimal(inp.target_year - inp.base_year)
        required_annual_change = _safe_divide(
            inp.target_year_emissions - inp.base_year_emissions,
            total_duration,
            Decimal("0"),
        )

        points: List[TrajectoryPoint] = []
        for year in range(inp.current_year + 1, inp.target_year + 1):
            years_from_base = _decimal(year - inp.base_year)

            # Required emissions on linear pathway
            required = inp.base_year_emissions + required_annual_change * years_from_base
            required = _round_val(max(required, Decimal("0")), self.config.score_precision)

            # Projected emissions on current trend
            projected = inp.base_year_emissions + actual_annual_change * years_from_base
            projected = _round_val(max(projected, Decimal("0")), self.config.score_precision)

            gap_tco2e = projected - required
            gap_tco2e = _round_val(gap_tco2e, self.config.score_precision)
            gap_pct = _safe_pct(gap_tco2e, required) if required > Decimal("0") else Decimal("0")

            point = TrajectoryPoint(
                year=year,
                required_emissions_tco2e=required,
                projected_emissions_tco2e=projected,
                gap_tco2e=gap_tco2e,
                gap_pct=gap_pct,
            )
            point.provenance_hash = _compute_hash(point)
            points.append(point)

        logger.info(
            "Projected trajectory: %d points from %d to %d",
            len(points), inp.current_year + 1, inp.target_year,
        )
        return points

    # -------------------------------------------------------------------
    # Corrective Actions
    # -------------------------------------------------------------------

    def identify_actions(
        self,
        status: TrackingStatus,
        gap_tco2e: Decimal,
        inp: ProgressInput,
    ) -> List[CorrectiveAction]:
        """Identify corrective actions based on progress status.

        Generates a prioritized list of recommended corrective actions
        based on the severity of the gap and the remaining time.

        Args:
            status: Current tracking status.
            gap_tco2e: Gap in tCO2e (positive = behind).
            inp: Progress input.

        Returns:
            List of CorrectiveAction sorted by priority.
        """
        actions: List[CorrectiveAction] = []

        if status == TrackingStatus.ON_TRACK:
            actions.append(CorrectiveAction(
                priority=ActionPriority.LONG_TERM,
                category="continuous_improvement",
                description="Continue current reduction trajectory. Monitor for deviations.",
                estimated_reduction_tco2e=Decimal("0"),
                estimated_timeline_months=12,
            ))
            return self._finalize_actions(actions)

        remaining_years = max(inp.target_year - inp.current_year, 1)

        if status == TrackingStatus.CRITICAL:
            actions.extend(self._generate_critical_actions(gap_tco2e, remaining_years))
        elif status == TrackingStatus.MAJOR_DEVIATION:
            actions.extend(self._generate_major_actions(gap_tco2e, remaining_years))
        elif status == TrackingStatus.MINOR_DEVIATION:
            actions.extend(self._generate_minor_actions(gap_tco2e, remaining_years))

        return self._finalize_actions(actions)

    def _generate_critical_actions(
        self, gap_tco2e: Decimal, remaining_years: int
    ) -> List[CorrectiveAction]:
        """Generate actions for CRITICAL status.

        Args:
            gap_tco2e: Gap in tCO2e.
            remaining_years: Years remaining.

        Returns:
            List of CorrectiveAction.
        """
        annual_gap = _safe_divide(gap_tco2e, _decimal(remaining_years))
        return [
            CorrectiveAction(
                priority=ActionPriority.IMMEDIATE,
                category="energy_transition",
                description="Accelerate renewable energy procurement to close emissions gap.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.30"), 2),
                estimated_timeline_months=6,
            ),
            CorrectiveAction(
                priority=ActionPriority.IMMEDIATE,
                category="operational_efficiency",
                description="Implement emergency energy efficiency measures across all operations.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.20"), 2),
                estimated_timeline_months=3,
            ),
            CorrectiveAction(
                priority=ActionPriority.SHORT_TERM,
                category="supply_chain",
                description="Engage top suppliers for emissions reduction commitments.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.25"), 2),
                estimated_timeline_months=12,
            ),
            CorrectiveAction(
                priority=ActionPriority.SHORT_TERM,
                category="target_review",
                description="Review target feasibility and consider base year recalculation.",
                estimated_reduction_tco2e=Decimal("0"),
                estimated_timeline_months=6,
            ),
            CorrectiveAction(
                priority=ActionPriority.MEDIUM_TERM,
                category="capital_investment",
                description="Fast-track capital investments in low-carbon technologies.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.25"), 2),
                estimated_timeline_months=18,
            ),
        ]

    def _generate_major_actions(
        self, gap_tco2e: Decimal, remaining_years: int
    ) -> List[CorrectiveAction]:
        """Generate actions for MAJOR_DEVIATION status.

        Args:
            gap_tco2e: Gap in tCO2e.
            remaining_years: Years remaining.

        Returns:
            List of CorrectiveAction.
        """
        annual_gap = _safe_divide(gap_tco2e, _decimal(remaining_years))
        return [
            CorrectiveAction(
                priority=ActionPriority.SHORT_TERM,
                category="energy_transition",
                description="Increase renewable energy share through PPAs or on-site generation.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.35"), 2),
                estimated_timeline_months=12,
            ),
            CorrectiveAction(
                priority=ActionPriority.SHORT_TERM,
                category="operational_efficiency",
                description="Enhance energy management systems and implement efficiency upgrades.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.25"), 2),
                estimated_timeline_months=9,
            ),
            CorrectiveAction(
                priority=ActionPriority.MEDIUM_TERM,
                category="supply_chain",
                description="Implement supplier engagement program with reduction targets.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.25"), 2),
                estimated_timeline_months=18,
            ),
            CorrectiveAction(
                priority=ActionPriority.MEDIUM_TERM,
                category="process_improvement",
                description="Evaluate process electrification and fuel switching opportunities.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.15"), 2),
                estimated_timeline_months=24,
            ),
        ]

    def _generate_minor_actions(
        self, gap_tco2e: Decimal, remaining_years: int
    ) -> List[CorrectiveAction]:
        """Generate actions for MINOR_DEVIATION status.

        Args:
            gap_tco2e: Gap in tCO2e.
            remaining_years: Years remaining.

        Returns:
            List of CorrectiveAction.
        """
        annual_gap = _safe_divide(gap_tco2e, _decimal(remaining_years))
        return [
            CorrectiveAction(
                priority=ActionPriority.MEDIUM_TERM,
                category="energy_transition",
                description="Expand renewable energy procurement to cover gap.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.40"), 2),
                estimated_timeline_months=12,
            ),
            CorrectiveAction(
                priority=ActionPriority.MEDIUM_TERM,
                category="operational_efficiency",
                description="Implement targeted efficiency measures in high-impact areas.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.35"), 2),
                estimated_timeline_months=12,
            ),
            CorrectiveAction(
                priority=ActionPriority.LONG_TERM,
                category="monitoring",
                description="Strengthen monitoring and increase reporting frequency.",
                estimated_reduction_tco2e=_round_val(annual_gap * Decimal("0.25"), 2),
                estimated_timeline_months=6,
            ),
        ]

    def _finalize_actions(
        self, actions: List[CorrectiveAction]
    ) -> List[CorrectiveAction]:
        """Add provenance hashes and sort by priority.

        Args:
            actions: List of corrective actions.

        Returns:
            Finalized list.
        """
        priority_order = {
            ActionPriority.IMMEDIATE: 0,
            ActionPriority.SHORT_TERM: 1,
            ActionPriority.MEDIUM_TERM: 2,
            ActionPriority.LONG_TERM: 3,
        }
        for action in actions:
            action.provenance_hash = _compute_hash(action)
        actions.sort(key=lambda a: priority_order.get(a.priority, 99))
        return actions

    # -------------------------------------------------------------------
    # Carbon Budget Analysis
    # -------------------------------------------------------------------

    def calculate_budget(self, inp: ProgressInput) -> BudgetAnalysis:
        """Calculate carbon budget analysis.

        The carbon budget is the total allowable emissions from base year
        through target year under the linear reduction pathway.  This is
        the area under the linear pathway curve.

        Args:
            inp: Progress input.

        Returns:
            BudgetAnalysis.
        """
        total_duration = _decimal(inp.target_year - inp.base_year)
        elapsed = _decimal(inp.current_year - inp.base_year)
        remaining_years = max(inp.target_year - inp.current_year, 0)

        # Total budget = area under linear pathway (trapezoid)
        total_budget = (
            (inp.base_year_emissions + inp.target_year_emissions)
            * total_duration / Decimal("2")
        )
        total_budget = _round_val(total_budget, self.config.score_precision)

        # Consumed budget = sum of actual annual emissions
        consumed_budget = self._calculate_consumed_budget(inp)

        remaining_budget = total_budget - consumed_budget
        remaining_budget = _round_val(remaining_budget, self.config.score_precision)

        consumed_pct = _safe_pct(consumed_budget, total_budget)

        # Required annual reduction from now
        required_annual = Decimal("0")
        if remaining_years > 0 and remaining_budget > Decimal("0"):
            required_annual = _safe_divide(
                remaining_budget, _decimal(remaining_years)
            )
            # This represents how much emissions per year are allowed
            # Convert to required reduction rate
            current_to_target_reduction = (
                inp.current_year_emissions - inp.target_year_emissions
            )
            required_annual_reduction = _safe_divide(
                current_to_target_reduction,
                _decimal(remaining_years),
            )
            required_annual = _round_val(required_annual_reduction, self.config.score_precision)

        budget_on_track = consumed_pct <= self.config.budget_warning_pct

        result = BudgetAnalysis(
            total_budget_tco2e=total_budget,
            consumed_budget_tco2e=consumed_budget,
            remaining_budget_tco2e=remaining_budget,
            consumed_pct=consumed_pct,
            remaining_years=remaining_years,
            required_annual_reduction_tco2e=required_annual,
            budget_on_track=budget_on_track,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _calculate_consumed_budget(self, inp: ProgressInput) -> Decimal:
        """Calculate consumed carbon budget from annual data.

        If annual data is available, sum the actual emissions.
        Otherwise, estimate using the trapezoidal rule from base year
        to current year.

        Args:
            inp: Progress input.

        Returns:
            Consumed budget in tCO2e.
        """
        if inp.annual_data:
            total = Decimal("0")
            for dp in inp.annual_data:
                total += dp.emissions_tco2e
            return _round_val(total, self.config.score_precision)

        # Estimate: trapezoidal area from base to current
        elapsed = _decimal(inp.current_year - inp.base_year)
        consumed = (
            (inp.base_year_emissions + inp.current_year_emissions)
            * elapsed / Decimal("2")
        )
        return _round_val(consumed, self.config.score_precision)

    # -------------------------------------------------------------------
    # Annual Comparison
    # -------------------------------------------------------------------

    def compare_annual_data(self, inp: ProgressInput) -> List[Dict[str, Any]]:
        """Compare annual data points against the required pathway.

        Args:
            inp: Progress input.

        Returns:
            List of dicts with year, actual, required, gap, yoy_change.
        """
        if not inp.annual_data:
            return []

        total_duration = _decimal(inp.target_year - inp.base_year)
        total_reduction = inp.base_year_emissions - inp.target_year_emissions

        sorted_data = sorted(inp.annual_data, key=lambda d: d.year)
        comparisons: List[Dict[str, Any]] = []

        for i, dp in enumerate(sorted_data):
            years_from_base = _decimal(dp.year - inp.base_year)
            fraction = _safe_divide(years_from_base, total_duration)
            required = inp.base_year_emissions - total_reduction * fraction
            required = _round_val(required, self.config.score_precision)

            gap = dp.emissions_tco2e - required
            gap = _round_val(gap, self.config.score_precision)

            yoy_change = Decimal("0")
            if i > 0:
                prev = sorted_data[i - 1].emissions_tco2e
                yoy_change = dp.emissions_tco2e - prev
                yoy_change = _round_val(yoy_change, self.config.score_precision)

            comparisons.append({
                "year": dp.year,
                "actual_tco2e": str(dp.emissions_tco2e),
                "required_tco2e": str(required),
                "gap_tco2e": str(gap),
                "yoy_change_tco2e": str(yoy_change),
                "on_track": gap <= Decimal("0"),
                "provenance_hash": _compute_hash({
                    "year": dp.year,
                    "actual": str(dp.emissions_tco2e),
                    "required": str(required),
                }),
            })

        return comparisons

    # -------------------------------------------------------------------
    # Batch Assessment
    # -------------------------------------------------------------------

    def assess_multiple(
        self, inputs: List[ProgressInput]
    ) -> List[ProgressResult]:
        """Assess progress for multiple targets.

        Args:
            inputs: List of ProgressInput.

        Returns:
            List of ProgressResult.
        """
        results: List[ProgressResult] = []
        for inp in inputs:
            try:
                result = self.assess_progress(inp)
                results.append(result)
            except ValueError as e:
                logger.error("Failed to assess %s: %s", inp.entity_name, str(e))
        logger.info("Assessed progress for %d targets", len(results))
        return results

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    def get_progress_summary(
        self, results: List[ProgressResult]
    ) -> Dict[str, Any]:
        """Generate a summary across multiple progress assessments.

        Args:
            results: List of ProgressResult.

        Returns:
            Dict with counts by status and RAG, average gap, etc.
        """
        status_counts: Dict[str, int] = {s.value: 0 for s in TrackingStatus}
        rag_counts: Dict[str, int] = {r.value: 0 for r in RAGStatus}
        total_gap = Decimal("0")

        for r in results:
            status_counts[r.status.value] += 1
            rag_counts[r.rag.value] += 1
            total_gap += r.gap_tco2e

        avg_gap = _safe_divide(total_gap, _decimal(len(results))) if results else Decimal("0")

        summary = {
            "total_targets": len(results),
            "status_counts": status_counts,
            "rag_counts": rag_counts,
            "total_gap_tco2e": str(_round_val(total_gap, 2)),
            "average_gap_tco2e": str(_round_val(avg_gap, 2)),
            "provenance_hash": _compute_hash(status_counts),
        }
        return summary
