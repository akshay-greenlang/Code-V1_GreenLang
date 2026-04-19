"""
KPI Calculation Engine - PACK-008 EU Taxonomy Alignment

This module calculates the three mandatory EU Taxonomy KPIs required under
Article 8 of the Disclosures Delegated Act (EU) 2021/2178:

- Turnover alignment ratio  = taxonomy-aligned turnover  / total turnover
- CapEx alignment ratio     = taxonomy-aligned CapEx     / total CapEx
- OpEx alignment ratio      = taxonomy-aligned OpEx      / total OpEx

All financial arithmetic uses ``Decimal`` to avoid floating-point issues.
The engine also produces eligible-vs-aligned breakdowns, prevents double-
counting across environmental objectives, supports CapEx plan recognition,
and generates year-over-year comparison metrics.

Example:
    >>> from decimal import Decimal
    >>> engine = KPICalculationEngine()
    >>> result = engine.calculate_kpis(
    ...     activities=[
    ...         {"activity_id": "CCM-4.1", "turnover": "500000",
    ...          "capex": "120000", "opex": "30000",
    ...          "is_eligible": True, "is_aligned": True,
    ...          "sc_objectives": ["CCM"]},
    ...     ],
    ...     financials={"total_turnover": "2000000",
    ...                 "total_capex": "400000", "total_opex": "100000"},
    ... )
    >>> print(f"Turnover alignment: {result.turnover_ratio}")
"""

import hashlib
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
PRECISION = Decimal("0.0001")  # 4 decimal places for ratios


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""
    CCM = "CCM"
    CCA = "CCA"
    WTR = "WTR"
    CE = "CE"
    PPC = "PPC"
    BIO = "BIO"


class KPIType(str, Enum):
    """KPI types."""
    TURNOVER = "TURNOVER"
    CAPEX = "CAPEX"
    OPEX = "OPEX"


class CapExPlanStatus(str, Enum):
    """Status of a CapEx plan under Article 8, para 1.1.2.2."""
    NO_PLAN = "NO_PLAN"
    PLAN_IN_PROGRESS = "PLAN_IN_PROGRESS"
    PLAN_APPROVED = "PLAN_APPROVED"
    PLAN_COMPLETED = "PLAN_COMPLETED"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ActivityFinancialData(BaseModel):
    """Financial data for one economic activity."""

    activity_id: str = Field(..., description="Taxonomy activity ID or NACE code")
    activity_name: str = Field(default="", description="Activity display name")
    turnover: Decimal = Field(default=ZERO, ge=ZERO, description="Activity turnover (EUR)")
    capex: Decimal = Field(default=ZERO, ge=ZERO, description="Activity CapEx (EUR)")
    opex: Decimal = Field(default=ZERO, ge=ZERO, description="Activity OpEx (EUR)")
    is_eligible: bool = Field(default=False, description="Taxonomy eligible")
    is_aligned: bool = Field(default=False, description="Taxonomy aligned (SC+DNSH+MS)")
    sc_objectives: List[str] = Field(
        default_factory=list,
        description="Environmental objectives with SC (for double-counting prevention)"
    )
    capex_plan_status: CapExPlanStatus = Field(
        default=CapExPlanStatus.NO_PLAN,
        description="CapEx plan status"
    )
    capex_plan_amount: Decimal = Field(
        default=ZERO, ge=ZERO,
        description="CapEx attributable to an approved CapEx plan (EUR)"
    )

    class Config:
        json_encoders = {Decimal: str}


class CompanyFinancials(BaseModel):
    """Company-level total financial figures."""

    total_turnover: Decimal = Field(..., gt=ZERO, description="Total company turnover (EUR)")
    total_capex: Decimal = Field(..., gt=ZERO, description="Total company CapEx (EUR)")
    total_opex: Decimal = Field(..., gt=ZERO, description="Total company OpEx (EUR)")
    reporting_period_start: Optional[str] = Field(
        None, description="ISO date string for period start"
    )
    reporting_period_end: Optional[str] = Field(
        None, description="ISO date string for period end"
    )

    class Config:
        json_encoders = {Decimal: str}


class ObjectiveBreakdown(BaseModel):
    """KPI breakdown for a single environmental objective."""

    objective: str = Field(..., description="Environmental objective code")
    eligible_turnover: Decimal = Field(default=ZERO, description="Eligible turnover")
    aligned_turnover: Decimal = Field(default=ZERO, description="Aligned turnover")
    eligible_capex: Decimal = Field(default=ZERO, description="Eligible CapEx")
    aligned_capex: Decimal = Field(default=ZERO, description="Aligned CapEx")
    eligible_opex: Decimal = Field(default=ZERO, description="Eligible OpEx")
    aligned_opex: Decimal = Field(default=ZERO, description="Aligned OpEx")

    class Config:
        json_encoders = {Decimal: str}


class YoYComparison(BaseModel):
    """Year-over-year KPI comparison."""

    kpi_type: KPIType = Field(..., description="KPI type")
    current_ratio: Decimal = Field(..., description="Current period ratio")
    previous_ratio: Decimal = Field(..., description="Previous period ratio")
    change_absolute: Decimal = Field(..., description="Absolute change (pp)")
    change_relative_pct: Optional[Decimal] = Field(
        None, description="Relative change percentage"
    )
    trend: str = Field(..., description="IMPROVED / DECLINED / STABLE")

    class Config:
        json_encoders = {Decimal: str}


class KPIResult(BaseModel):
    """Complete KPI calculation result."""

    # Eligible amounts
    eligible_turnover: Decimal = Field(default=ZERO, description="Total eligible turnover")
    eligible_capex: Decimal = Field(default=ZERO, description="Total eligible CapEx")
    eligible_opex: Decimal = Field(default=ZERO, description="Total eligible OpEx")

    # Aligned amounts
    aligned_turnover: Decimal = Field(default=ZERO, description="Total aligned turnover")
    aligned_capex: Decimal = Field(default=ZERO, description="Total aligned CapEx")
    aligned_opex: Decimal = Field(default=ZERO, description="Total aligned OpEx")

    # Total denominators
    total_turnover: Decimal = Field(..., description="Total company turnover")
    total_capex: Decimal = Field(..., description="Total company CapEx")
    total_opex: Decimal = Field(..., description="Total company OpEx")

    # Eligible ratios
    eligible_turnover_ratio: Decimal = Field(
        default=ZERO, description="Eligible turnover / total turnover"
    )
    eligible_capex_ratio: Decimal = Field(
        default=ZERO, description="Eligible CapEx / total CapEx"
    )
    eligible_opex_ratio: Decimal = Field(
        default=ZERO, description="Eligible OpEx / total OpEx"
    )

    # Aligned ratios (primary KPIs)
    turnover_ratio: Decimal = Field(
        default=ZERO, description="Aligned turnover / total turnover"
    )
    capex_ratio: Decimal = Field(
        default=ZERO, description="Aligned CapEx / total CapEx"
    )
    opex_ratio: Decimal = Field(
        default=ZERO, description="Aligned OpEx / total OpEx"
    )

    # CapEx plan
    capex_plan_amount: Decimal = Field(
        default=ZERO, description="CapEx attributable to approved plans"
    )

    # Breakdowns
    activities_total: int = Field(default=0, description="Total activities assessed")
    activities_eligible: int = Field(default=0, description="Eligible activities count")
    activities_aligned: int = Field(default=0, description="Aligned activities count")
    objective_breakdown: Dict[str, ObjectiveBreakdown] = Field(
        default_factory=dict,
        description="Per-objective eligible/aligned breakdown"
    )
    activity_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-activity financial mapping detail"
    )

    # Year-over-year
    yoy_comparisons: List[YoYComparison] = Field(
        default_factory=list,
        description="Year-over-year comparison (if prior period provided)"
    )

    # Meta
    double_counting_adjustments: Decimal = Field(
        default=ZERO,
        description="Amount deducted to prevent double-counting across objectives"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Calculation timestamp"
    )

    class Config:
        json_encoders = {Decimal: str}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class KPICalculationEngine:
    """
    KPI Calculation Engine for PACK-008 EU Taxonomy Alignment.

    This engine calculates the three mandatory Taxonomy KPIs (Turnover,
    CapEx, OpEx alignment ratios) required under the Disclosures Delegated
    Act.  It uses ``Decimal`` for all financial arithmetic, prevents
    double-counting across environmental objectives, and supports CapEx
    plan recognition.

    It follows GreenLang's zero-hallucination principle by performing only
    deterministic arithmetic -- no LLM inference in the calculation path.

    Example:
        >>> engine = KPICalculationEngine()
        >>> result = engine.calculate_kpis(
        ...     activities=[...], financials={...},
        ... )
        >>> assert result.turnover_ratio >= ZERO
    """

    def __init__(self) -> None:
        """Initialize the KPI Calculation Engine."""
        logger.info("Initialized KPICalculationEngine")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_kpis(
        self,
        activities: List[Dict[str, Any]],
        financials: Dict[str, Any],
        prior_period: Optional[Dict[str, Any]] = None,
    ) -> KPIResult:
        """
        Calculate all three Taxonomy KPIs.

        Args:
            activities: List of activity financial data dicts.  Each must
                contain at least ``activity_id``; optional keys include
                ``turnover``, ``capex``, ``opex``, ``is_eligible``,
                ``is_aligned``, ``sc_objectives``, ``capex_plan_status``,
                ``capex_plan_amount``.
            financials: Company financials dict with keys
                ``total_turnover``, ``total_capex``, ``total_opex``.
            prior_period: Optional dict with keys ``turnover_ratio``,
                ``capex_ratio``, ``opex_ratio`` from the previous period
                for year-over-year comparison.

        Returns:
            KPIResult with all ratios, breakdowns, and provenance.

        Raises:
            ValueError: If activities list is empty or financials are missing.
        """
        if not activities:
            raise ValueError("activities list cannot be empty")

        start = datetime.utcnow()

        # Parse company financials
        company = self._parse_financials(financials)

        # Parse activity-level data
        parsed_activities = [self._parse_activity(a) for a in activities]

        logger.info(
            "Calculating KPIs for %d activities; totals: turnover=%s capex=%s opex=%s",
            len(parsed_activities),
            company.total_turnover, company.total_capex, company.total_opex,
        )

        # Aggregate eligible and aligned amounts (with double-counting prevention)
        agg = self._aggregate(parsed_activities)

        # Calculate ratios
        turnover_ratio = self._safe_divide(agg["aligned_turnover"], company.total_turnover)
        capex_ratio = self._safe_divide(agg["aligned_capex"], company.total_capex)
        opex_ratio = self._safe_divide(agg["aligned_opex"], company.total_opex)

        eligible_turnover_ratio = self._safe_divide(agg["eligible_turnover"], company.total_turnover)
        eligible_capex_ratio = self._safe_divide(agg["eligible_capex"], company.total_capex)
        eligible_opex_ratio = self._safe_divide(agg["eligible_opex"], company.total_opex)

        # Year-over-year comparisons
        yoy: List[YoYComparison] = []
        if prior_period:
            yoy = self._build_yoy(
                turnover_ratio, capex_ratio, opex_ratio, prior_period
            )

        # Activity-level detail for audit
        activity_details = self._build_activity_details(parsed_activities)

        # Provenance
        provenance_hash = self._hash(
            f"KPI|{turnover_ratio}|{capex_ratio}|{opex_ratio}|{len(parsed_activities)}"
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = KPIResult(
            eligible_turnover=agg["eligible_turnover"],
            eligible_capex=agg["eligible_capex"],
            eligible_opex=agg["eligible_opex"],
            aligned_turnover=agg["aligned_turnover"],
            aligned_capex=agg["aligned_capex"],
            aligned_opex=agg["aligned_opex"],
            total_turnover=company.total_turnover,
            total_capex=company.total_capex,
            total_opex=company.total_opex,
            eligible_turnover_ratio=eligible_turnover_ratio,
            eligible_capex_ratio=eligible_capex_ratio,
            eligible_opex_ratio=eligible_opex_ratio,
            turnover_ratio=turnover_ratio,
            capex_ratio=capex_ratio,
            opex_ratio=opex_ratio,
            capex_plan_amount=agg["capex_plan_amount"],
            activities_total=len(parsed_activities),
            activities_eligible=agg["count_eligible"],
            activities_aligned=agg["count_aligned"],
            objective_breakdown=agg["objective_breakdown"],
            activity_details=activity_details,
            yoy_comparisons=yoy,
            double_counting_adjustments=agg["double_counting_adj"],
            provenance_hash=provenance_hash,
        )

        logger.info(
            "KPI calculation complete in %.1fms: turnover=%.4f capex=%.4f opex=%.4f",
            elapsed_ms, turnover_ratio, capex_ratio, opex_ratio,
        )

        return result

    def calculate_turnover_ratio(
        self,
        activities: List[Dict[str, Any]],
        total_turnover: str,
    ) -> Decimal:
        """
        Calculate only the Turnover alignment ratio.

        Args:
            activities: Activity dicts with ``turnover`` and ``is_aligned``.
            total_turnover: Total company turnover as string.

        Returns:
            Decimal turnover alignment ratio.
        """
        total = self._to_decimal(total_turnover)
        aligned = ZERO
        seen_activities: set = set()

        for a in activities:
            aid = a.get("activity_id", "")
            if not a.get("is_aligned", False):
                continue
            if aid in seen_activities:
                continue
            seen_activities.add(aid)
            aligned += self._to_decimal(a.get("turnover", "0"))

        return self._safe_divide(aligned, total)

    def calculate_capex_ratio(
        self,
        activities: List[Dict[str, Any]],
        total_capex: str,
    ) -> Decimal:
        """
        Calculate only the CapEx alignment ratio.

        Includes CapEx plan amounts for activities with approved plans.

        Args:
            activities: Activity dicts with ``capex``, ``is_aligned``, and
                optional ``capex_plan_status``/``capex_plan_amount``.
            total_capex: Total company CapEx as string.

        Returns:
            Decimal CapEx alignment ratio.
        """
        total = self._to_decimal(total_capex)
        aligned = ZERO
        seen_activities: set = set()

        for a in activities:
            aid = a.get("activity_id", "")
            if aid in seen_activities:
                continue
            seen_activities.add(aid)

            if a.get("is_aligned", False):
                aligned += self._to_decimal(a.get("capex", "0"))

            # CapEx plan recognition (Article 8, para 1.1.2.2)
            plan_status = a.get("capex_plan_status", CapExPlanStatus.NO_PLAN.value)
            if plan_status in (
                CapExPlanStatus.PLAN_APPROVED.value,
                CapExPlanStatus.PLAN_IN_PROGRESS.value,
            ):
                aligned += self._to_decimal(a.get("capex_plan_amount", "0"))

        return self._safe_divide(aligned, total)

    def calculate_opex_ratio(
        self,
        activities: List[Dict[str, Any]],
        total_opex: str,
    ) -> Decimal:
        """
        Calculate only the OpEx alignment ratio.

        Args:
            activities: Activity dicts with ``opex`` and ``is_aligned``.
            total_opex: Total company OpEx as string.

        Returns:
            Decimal OpEx alignment ratio.
        """
        total = self._to_decimal(total_opex)
        aligned = ZERO
        seen_activities: set = set()

        for a in activities:
            aid = a.get("activity_id", "")
            if not a.get("is_aligned", False):
                continue
            if aid in seen_activities:
                continue
            seen_activities.add(aid)
            aligned += self._to_decimal(a.get("opex", "0"))

        return self._safe_divide(aligned, total)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_financials(
        self,
        data: Dict[str, Any],
    ) -> CompanyFinancials:
        """Parse and validate company-level financials."""
        return CompanyFinancials(
            total_turnover=self._to_decimal(data["total_turnover"]),
            total_capex=self._to_decimal(data["total_capex"]),
            total_opex=self._to_decimal(data["total_opex"]),
            reporting_period_start=data.get("reporting_period_start"),
            reporting_period_end=data.get("reporting_period_end"),
        )

    def _parse_activity(
        self,
        data: Dict[str, Any],
    ) -> ActivityFinancialData:
        """Parse a single activity financial data dict into a model."""
        plan_status_raw = data.get("capex_plan_status", CapExPlanStatus.NO_PLAN.value)
        try:
            plan_status = CapExPlanStatus(plan_status_raw)
        except ValueError:
            plan_status = CapExPlanStatus.NO_PLAN

        return ActivityFinancialData(
            activity_id=data.get("activity_id", "UNKNOWN"),
            activity_name=data.get("activity_name", ""),
            turnover=self._to_decimal(data.get("turnover", "0")),
            capex=self._to_decimal(data.get("capex", "0")),
            opex=self._to_decimal(data.get("opex", "0")),
            is_eligible=bool(data.get("is_eligible", False)),
            is_aligned=bool(data.get("is_aligned", False)),
            sc_objectives=data.get("sc_objectives", []),
            capex_plan_status=plan_status,
            capex_plan_amount=self._to_decimal(data.get("capex_plan_amount", "0")),
        )

    def _aggregate(
        self,
        activities: List[ActivityFinancialData],
    ) -> Dict[str, Any]:
        """
        Aggregate eligible and aligned amounts with double-counting prevention.

        Double-counting prevention: if the same activity contributes to
        multiple objectives, its financial amounts are counted only once in
        the total numerators.  The per-objective breakdown records the
        full amount for each objective, but the totals are de-duplicated.
        """
        eligible_turnover = ZERO
        eligible_capex = ZERO
        eligible_opex = ZERO
        aligned_turnover = ZERO
        aligned_capex = ZERO
        aligned_opex = ZERO
        capex_plan_total = ZERO
        count_eligible = 0
        count_aligned = 0
        double_counting_adj = ZERO

        # Per-objective breakdown
        obj_map: Dict[str, ObjectiveBreakdown] = {}
        for obj in EnvironmentalObjective:
            obj_map[obj.value] = ObjectiveBreakdown(objective=obj.value)

        # Track seen activity IDs for double-counting prevention
        seen_eligible: set = set()
        seen_aligned: set = set()

        for act in activities:
            # Eligible aggregation
            if act.is_eligible:
                if act.activity_id not in seen_eligible:
                    eligible_turnover += act.turnover
                    eligible_capex += act.capex
                    eligible_opex += act.opex
                    count_eligible += 1
                    seen_eligible.add(act.activity_id)
                else:
                    double_counting_adj += act.turnover + act.capex + act.opex

                # Per-objective breakdown (records each objective)
                for obj_str in act.sc_objectives:
                    if obj_str in obj_map:
                        ob = obj_map[obj_str]
                        ob.eligible_turnover += act.turnover
                        ob.eligible_capex += act.capex
                        ob.eligible_opex += act.opex

            # Aligned aggregation
            if act.is_aligned:
                if act.activity_id not in seen_aligned:
                    aligned_turnover += act.turnover
                    aligned_capex += act.capex
                    aligned_opex += act.opex
                    count_aligned += 1
                    seen_aligned.add(act.activity_id)
                else:
                    double_counting_adj += act.turnover + act.capex + act.opex

                for obj_str in act.sc_objectives:
                    if obj_str in obj_map:
                        ob = obj_map[obj_str]
                        ob.aligned_turnover += act.turnover
                        ob.aligned_capex += act.capex
                        ob.aligned_opex += act.opex

            # CapEx plan (Article 8, para 1.1.2.2)
            if act.capex_plan_status in (
                CapExPlanStatus.PLAN_APPROVED,
                CapExPlanStatus.PLAN_IN_PROGRESS,
            ):
                capex_plan_total += act.capex_plan_amount

        return {
            "eligible_turnover": eligible_turnover,
            "eligible_capex": eligible_capex,
            "eligible_opex": eligible_opex,
            "aligned_turnover": aligned_turnover,
            "aligned_capex": aligned_capex + capex_plan_total,
            "aligned_opex": aligned_opex,
            "capex_plan_amount": capex_plan_total,
            "count_eligible": count_eligible,
            "count_aligned": count_aligned,
            "objective_breakdown": obj_map,
            "double_counting_adj": double_counting_adj,
        }

    @staticmethod
    def _build_activity_details(
        activities: List[ActivityFinancialData],
    ) -> List[Dict[str, Any]]:
        """Build per-activity detail records for audit trail."""
        details: List[Dict[str, Any]] = []
        for act in activities:
            details.append({
                "activity_id": act.activity_id,
                "activity_name": act.activity_name,
                "turnover": str(act.turnover),
                "capex": str(act.capex),
                "opex": str(act.opex),
                "is_eligible": act.is_eligible,
                "is_aligned": act.is_aligned,
                "sc_objectives": act.sc_objectives,
                "capex_plan_status": act.capex_plan_status.value,
                "capex_plan_amount": str(act.capex_plan_amount),
            })
        return details

    def _build_yoy(
        self,
        turnover_ratio: Decimal,
        capex_ratio: Decimal,
        opex_ratio: Decimal,
        prior: Dict[str, Any],
    ) -> List[YoYComparison]:
        """Build year-over-year comparison records."""
        comparisons: List[YoYComparison] = []

        for kpi_type, current, prior_key in [
            (KPIType.TURNOVER, turnover_ratio, "turnover_ratio"),
            (KPIType.CAPEX, capex_ratio, "capex_ratio"),
            (KPIType.OPEX, opex_ratio, "opex_ratio"),
        ]:
            previous = self._to_decimal(prior.get(prior_key, "0"))
            change_abs = current - previous
            change_rel = (
                (change_abs / previous * HUNDRED).quantize(PRECISION, ROUND_HALF_UP)
                if previous != ZERO else None
            )
            trend = self._determine_trend(change_abs)

            comparisons.append(YoYComparison(
                kpi_type=kpi_type,
                current_ratio=current,
                previous_ratio=previous,
                change_absolute=change_abs,
                change_relative_pct=change_rel,
                trend=trend,
            ))

        return comparisons

    @staticmethod
    def _determine_trend(change: Decimal) -> str:
        """Classify a change as IMPROVED, DECLINED, or STABLE."""
        threshold = Decimal("0.005")  # 0.5 pp
        if change > threshold:
            return "IMPROVED"
        elif change < -threshold:
            return "DECLINED"
        return "STABLE"

    @staticmethod
    def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
        """Divide with zero-denominator protection, rounded to 4 dp."""
        if denominator == ZERO:
            return ZERO
        return (numerator / denominator).quantize(PRECISION, ROUND_HALF_UP)

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        """Convert a value to Decimal, defaulting to ZERO on failure."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return ZERO

    @staticmethod
    def _hash(data: str) -> str:
        """Return SHA-256 hex digest."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
