"""
KPI Calculation Engine -- Turnover/CapEx/OpEx with Double-Counting Prevention

Implements the EU Taxonomy Article 8 KPI calculation pipeline for non-financial
undertakings: computing the Turnover KPI, CapEx KPI, and OpEx KPI as the ratio
of taxonomy-aligned amounts to total amounts, with double-counting prevention,
CapEx plan support, and objective disaggregation.

Key capabilities:
  - Turnover KPI: aligned revenue / total net revenue (IAS 1, IFRS 15)
  - CapEx KPI: aligned CapEx / total CapEx (IAS 16, IAS 38, IFRS 16, IAS 40)
  - OpEx KPI: aligned narrow OpEx / total narrow OpEx
  - Double-counting prevention (each EUR assigned to one objective only)
  - CapEx plan registration (up to 10-year plans)
  - Objective-level disaggregation of KPI numerators
  - Multi-period comparison
  - KPI dashboard with enabling/transitional breakdowns

All calculations are deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852, Article 8
    - Delegated Regulation (EU) 2021/2178 (Article 8 Disclosures)
    - IAS 1 (Revenue), IFRS 15 (Revenue from Contracts with Customers)
    - IAS 16 (Property, Plant and Equipment), IAS 38 (Intangible Assets)
    - IFRS 16 (Leases), IAS 40 (Investment Property)
    - EBA Pillar 3 ESG ITS (for financial institutions GAR/BTAR)

Example:
    >>> engine = KPICalculationEngine(config)
    >>> result = engine.calculate_turnover_kpi("org-1", "2025", activity_financials)
    >>> result.aligned_pct
    45.2
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    KPIType,
    TaxonomyAppConfig,
)
from .models import (
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class TurnoverResult(BaseModel):
    """Turnover KPI calculation result."""

    kpi_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    period: str = Field(...)
    total_turnover_eur: Decimal = Field(default=Decimal("0"))
    eligible_turnover_eur: Decimal = Field(default=Decimal("0"))
    aligned_turnover_eur: Decimal = Field(default=Decimal("0"))
    eligible_pct: Decimal = Field(default=Decimal("0"))
    aligned_pct: Decimal = Field(default=Decimal("0"))
    enabling_eur: Decimal = Field(default=Decimal("0"))
    transitional_eur: Decimal = Field(default=Decimal("0"))
    objective_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    activity_count: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class CapExResult(BaseModel):
    """CapEx KPI calculation result."""

    kpi_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    period: str = Field(...)
    total_capex_eur: Decimal = Field(default=Decimal("0"))
    eligible_capex_eur: Decimal = Field(default=Decimal("0"))
    aligned_capex_eur: Decimal = Field(default=Decimal("0"))
    eligible_pct: Decimal = Field(default=Decimal("0"))
    aligned_pct: Decimal = Field(default=Decimal("0"))
    enabling_eur: Decimal = Field(default=Decimal("0"))
    transitional_eur: Decimal = Field(default=Decimal("0"))
    capex_plan_eur: Decimal = Field(default=Decimal("0"), description="Amounts from CapEx plans")
    objective_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    activity_count: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class OpExResult(BaseModel):
    """OpEx KPI calculation result."""

    kpi_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    period: str = Field(...)
    total_opex_eur: Decimal = Field(default=Decimal("0"))
    eligible_opex_eur: Decimal = Field(default=Decimal("0"))
    aligned_opex_eur: Decimal = Field(default=Decimal("0"))
    eligible_pct: Decimal = Field(default=Decimal("0"))
    aligned_pct: Decimal = Field(default=Decimal("0"))
    enabling_eur: Decimal = Field(default=Decimal("0"))
    transitional_eur: Decimal = Field(default=Decimal("0"))
    objective_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    activity_count: int = Field(default=0)
    narrow_opex_categories: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class KPIResult(BaseModel):
    """Combined Turnover + CapEx + OpEx KPI result."""

    kpi_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    period: str = Field(...)
    turnover: TurnoverResult = Field(...)
    capex: CapExResult = Field(...)
    opex: OpExResult = Field(...)
    calculated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class CapExPlanResult(BaseModel):
    """Registered CapEx plan details."""

    plan_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    activity_code: str = Field(...)
    plan_start_year: int = Field(...)
    plan_end_year: int = Field(...)
    total_capex_eur: Decimal = Field(default=Decimal("0"))
    annual_capex_eur: Decimal = Field(default=Decimal("0"))
    objective: str = Field(default="climate_mitigation")
    description: str = Field(default="")
    status: str = Field(default="approved")
    registered_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class ObjectiveBreakdown(BaseModel):
    """KPI disaggregation by environmental objective."""

    org_id: str = Field(...)
    period: str = Field(...)
    kpi_type: str = Field(...)
    total_aligned_eur: Decimal = Field(default=Decimal("0"))
    objectives: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class KPIDashboard(BaseModel):
    """Executive KPI dashboard for an organization-period."""

    org_id: str = Field(...)
    period: str = Field(...)
    turnover_aligned_pct: Decimal = Field(default=Decimal("0"))
    capex_aligned_pct: Decimal = Field(default=Decimal("0"))
    opex_aligned_pct: Decimal = Field(default=Decimal("0"))
    turnover_eligible_pct: Decimal = Field(default=Decimal("0"))
    capex_eligible_pct: Decimal = Field(default=Decimal("0"))
    opex_eligible_pct: Decimal = Field(default=Decimal("0"))
    total_enabling_eur: Decimal = Field(default=Decimal("0"))
    total_transitional_eur: Decimal = Field(default=Decimal("0"))
    capex_plan_count: int = Field(default=0)
    activity_count: int = Field(default=0)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Narrow OpEx Categories (Article 8 Disclosures Delegated Act)
# ---------------------------------------------------------------------------

_NARROW_OPEX_CATEGORIES: List[str] = [
    "research_and_development",
    "building_renovation_measures",
    "short_term_lease",
    "maintenance_and_repair",
    "other_direct_expenditures",
]


# ---------------------------------------------------------------------------
# KPICalculationEngine
# ---------------------------------------------------------------------------

class KPICalculationEngine:
    """
    KPI Calculation Engine for EU Taxonomy Article 8 disclosures.

    Computes the three mandatory KPIs (Turnover, CapEx, OpEx) as the ratio
    of taxonomy-aligned amounts to total amounts, with double-counting
    prevention ensuring each EUR is assigned to exactly one environmental
    objective.  Supports CapEx plans and objective-level disaggregation.

    Attributes:
        config: Application configuration.
        _kpi_results: In-memory store keyed by (org_id, period, kpi_type).
        _capex_plans: CapEx plans keyed by plan_id.
        _org_capex_plans: Plans indexed by org_id.
        _activities: Activity financials keyed by org_id.

    Example:
        >>> engine = KPICalculationEngine(config)
        >>> result = engine.calculate_all_kpis("org-1", "2025", financials)
        >>> result.turnover.aligned_pct
        Decimal('45.20')
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """
        Initialize KPICalculationEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or TaxonomyAppConfig()
        self._kpi_results: Dict[str, KPIResult] = {}
        self._capex_plans: Dict[str, CapExPlanResult] = {}
        self._org_capex_plans: Dict[str, List[str]] = {}
        self._activities: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("KPICalculationEngine initialized")

    # ------------------------------------------------------------------
    # Turnover KPI
    # ------------------------------------------------------------------

    def calculate_turnover_kpi(
        self,
        org_id: str,
        period: str,
        activity_financials: List[Dict[str, Any]],
    ) -> TurnoverResult:
        """
        Calculate the Turnover KPI (Article 8 denominator: IAS 1 / IFRS 15).

        Numerator is the sum of net revenue from taxonomy-aligned activities.
        Denominator is total net revenue. Each EUR counted only once.

        Args:
            org_id: Organization identifier.
            period: Reporting period (e.g. '2025').
            activity_financials: List of dicts with keys:
                activity_code, objective, activity_type, turnover_eur,
                is_eligible, is_aligned.

        Returns:
            TurnoverResult with aligned/eligible percentages.
        """
        start = datetime.utcnow()

        # Prevent double-counting
        cleaned = self.prevent_double_counting(activity_financials)
        self._activities[org_id] = cleaned

        total_turnover = Decimal("0")
        eligible_turnover = Decimal("0")
        aligned_turnover = Decimal("0")
        enabling_eur = Decimal("0")
        transitional_eur = Decimal("0")
        obj_breakdown: Dict[str, Decimal] = {}
        activity_count = 0

        for act in cleaned:
            amt = Decimal(str(act.get("turnover_eur", 0)))
            total_turnover += amt

            if act.get("is_eligible", False):
                eligible_turnover += amt

            if act.get("is_aligned", False):
                aligned_turnover += amt
                activity_count += 1

                objective = act.get("objective", "climate_mitigation")
                obj_breakdown[objective] = obj_breakdown.get(objective, Decimal("0")) + amt

                act_type = act.get("activity_type", "own_performance")
                if act_type == "enabling":
                    enabling_eur += amt
                elif act_type == "transitional":
                    transitional_eur += amt

        eligible_pct = self._safe_pct(eligible_turnover, total_turnover)
        aligned_pct = self._safe_pct(aligned_turnover, total_turnover)

        provenance = _sha256(
            f"turnover_kpi:{org_id}:{period}:{float(total_turnover)}:{float(aligned_turnover)}"
        )

        result = TurnoverResult(
            org_id=org_id,
            period=period,
            total_turnover_eur=total_turnover,
            eligible_turnover_eur=eligible_turnover,
            aligned_turnover_eur=aligned_turnover,
            eligible_pct=eligible_pct,
            aligned_pct=aligned_pct,
            enabling_eur=enabling_eur,
            transitional_eur=transitional_eur,
            objective_breakdown=obj_breakdown,
            activity_count=activity_count,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Turnover KPI for %s/%s: aligned=%.2f%% (EUR %s / EUR %s) in %.1f ms",
            org_id, period, float(aligned_pct),
            float(aligned_turnover), float(total_turnover), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # CapEx KPI
    # ------------------------------------------------------------------

    def calculate_capex_kpi(
        self,
        org_id: str,
        period: str,
        activity_financials: List[Dict[str, Any]],
    ) -> CapExResult:
        """
        Calculate the CapEx KPI (IAS 16/38, IFRS 16, IAS 40).

        Numerator includes CapEx from aligned activities plus amounts
        from registered CapEx plans.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            activity_financials: Activity financial data.

        Returns:
            CapExResult with aligned/eligible percentages.
        """
        start = datetime.utcnow()

        cleaned = self.prevent_double_counting(activity_financials)

        total_capex = Decimal("0")
        eligible_capex = Decimal("0")
        aligned_capex = Decimal("0")
        enabling_eur = Decimal("0")
        transitional_eur = Decimal("0")
        obj_breakdown: Dict[str, Decimal] = {}
        activity_count = 0

        for act in cleaned:
            amt = Decimal(str(act.get("capex_eur", 0)))
            total_capex += amt

            if act.get("is_eligible", False):
                eligible_capex += amt

            if act.get("is_aligned", False):
                aligned_capex += amt
                activity_count += 1

                objective = act.get("objective", "climate_mitigation")
                obj_breakdown[objective] = obj_breakdown.get(objective, Decimal("0")) + amt

                act_type = act.get("activity_type", "own_performance")
                if act_type == "enabling":
                    enabling_eur += amt
                elif act_type == "transitional":
                    transitional_eur += amt

        # Add CapEx plan amounts (if enabled)
        capex_plan_eur = Decimal("0")
        if self.config.include_capex_plans if hasattr(self.config, 'include_capex_plans') else True:
            plan_ids = self._org_capex_plans.get(org_id, [])
            for plan_id in plan_ids:
                plan = self._capex_plans.get(plan_id)
                if plan and str(plan.plan_start_year) <= period <= str(plan.plan_end_year):
                    capex_plan_eur += plan.annual_capex_eur
                    aligned_capex += plan.annual_capex_eur
                    obj = plan.objective
                    obj_breakdown[obj] = obj_breakdown.get(obj, Decimal("0")) + plan.annual_capex_eur

        eligible_pct = self._safe_pct(eligible_capex, total_capex)
        aligned_pct = self._safe_pct(aligned_capex, total_capex)

        provenance = _sha256(
            f"capex_kpi:{org_id}:{period}:{float(total_capex)}:{float(aligned_capex)}"
        )

        result = CapExResult(
            org_id=org_id,
            period=period,
            total_capex_eur=total_capex,
            eligible_capex_eur=eligible_capex,
            aligned_capex_eur=aligned_capex,
            eligible_pct=eligible_pct,
            aligned_pct=aligned_pct,
            enabling_eur=enabling_eur,
            transitional_eur=transitional_eur,
            capex_plan_eur=capex_plan_eur,
            objective_breakdown=obj_breakdown,
            activity_count=activity_count,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "CapEx KPI for %s/%s: aligned=%.2f%% (plans=EUR %s) in %.1f ms",
            org_id, period, float(aligned_pct), float(capex_plan_eur), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # OpEx KPI
    # ------------------------------------------------------------------

    def calculate_opex_kpi(
        self,
        org_id: str,
        period: str,
        activity_financials: List[Dict[str, Any]],
    ) -> OpExResult:
        """
        Calculate the OpEx KPI (narrow OpEx categories only).

        Narrow OpEx includes: R&D, building renovation measures,
        short-term lease, maintenance and repair, and other direct
        expenditures related to day-to-day asset servicing.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            activity_financials: Activity financial data.

        Returns:
            OpExResult with aligned/eligible percentages.
        """
        start = datetime.utcnow()

        cleaned = self.prevent_double_counting(activity_financials)

        total_opex = Decimal("0")
        eligible_opex = Decimal("0")
        aligned_opex = Decimal("0")
        enabling_eur = Decimal("0")
        transitional_eur = Decimal("0")
        obj_breakdown: Dict[str, Decimal] = {}
        activity_count = 0

        for act in cleaned:
            amt = Decimal(str(act.get("opex_eur", 0)))
            total_opex += amt

            if act.get("is_eligible", False):
                eligible_opex += amt

            if act.get("is_aligned", False):
                aligned_opex += amt
                activity_count += 1

                objective = act.get("objective", "climate_mitigation")
                obj_breakdown[objective] = obj_breakdown.get(objective, Decimal("0")) + amt

                act_type = act.get("activity_type", "own_performance")
                if act_type == "enabling":
                    enabling_eur += amt
                elif act_type == "transitional":
                    transitional_eur += amt

        eligible_pct = self._safe_pct(eligible_opex, total_opex)
        aligned_pct = self._safe_pct(aligned_opex, total_opex)

        provenance = _sha256(
            f"opex_kpi:{org_id}:{period}:{float(total_opex)}:{float(aligned_opex)}"
        )

        result = OpExResult(
            org_id=org_id,
            period=period,
            total_opex_eur=total_opex,
            eligible_opex_eur=eligible_opex,
            aligned_opex_eur=aligned_opex,
            eligible_pct=eligible_pct,
            aligned_pct=aligned_pct,
            enabling_eur=enabling_eur,
            transitional_eur=transitional_eur,
            objective_breakdown=obj_breakdown,
            activity_count=activity_count,
            narrow_opex_categories=_NARROW_OPEX_CATEGORIES.copy(),
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "OpEx KPI for %s/%s: aligned=%.2f%% in %.1f ms",
            org_id, period, float(aligned_pct), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Combined KPI
    # ------------------------------------------------------------------

    def calculate_all_kpis(
        self,
        org_id: str,
        period: str,
        activity_financials: List[Dict[str, Any]],
    ) -> KPIResult:
        """
        Calculate all three KPIs (Turnover, CapEx, OpEx) in one call.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            activity_financials: Activity financial data.

        Returns:
            KPIResult containing all three KPIs.
        """
        start = datetime.utcnow()

        turnover = self.calculate_turnover_kpi(org_id, period, activity_financials)
        capex = self.calculate_capex_kpi(org_id, period, activity_financials)
        opex = self.calculate_opex_kpi(org_id, period, activity_financials)

        provenance = _sha256(
            f"all_kpi:{org_id}:{period}:"
            f"{turnover.provenance_hash}:{capex.provenance_hash}:{opex.provenance_hash}"
        )

        result = KPIResult(
            org_id=org_id,
            period=period,
            turnover=turnover,
            capex=capex,
            opex=opex,
            provenance_hash=provenance,
        )

        cache_key = f"{org_id}:{period}"
        self._kpi_results[cache_key] = result

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "All KPIs for %s/%s: turnover=%.2f%%, capex=%.2f%%, opex=%.2f%% in %.1f ms",
            org_id, period,
            float(turnover.aligned_pct), float(capex.aligned_pct),
            float(opex.aligned_pct), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Double-Counting Prevention
    # ------------------------------------------------------------------

    def prevent_double_counting(
        self,
        activities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ensure no EUR is counted in more than one objective's numerator.

        Uses the configured strategy (first_claimed or company_choice).
        When an activity contributes to multiple objectives, only the
        first-claimed objective receives the financial amounts.

        Args:
            activities: List of activity financial records.

        Returns:
            Cleaned list with double-counting resolved.
        """
        # Track which activity+objective combinations have been claimed
        claimed: Dict[str, str] = {}  # activity_code -> claimed_objective
        cleaned: List[Dict[str, Any]] = []

        for act in activities:
            act_code = act.get("activity_code", "")
            objective = act.get("objective", "climate_mitigation")

            if act_code in claimed:
                # Already claimed for a different objective -- zero out aligned amounts
                existing_obj = claimed[act_code]
                if existing_obj != objective:
                    act_copy = dict(act)
                    act_copy["is_aligned"] = False
                    act_copy["double_counting_note"] = (
                        f"Activity {act_code} already counted under {existing_obj}; "
                        f"prevented double-counting for {objective}"
                    )
                    cleaned.append(act_copy)
                    logger.debug(
                        "Double-counting prevented: %s already claimed for %s, "
                        "not counting for %s",
                        act_code, existing_obj, objective,
                    )
                    continue

            # Claim this activity for the objective
            if act.get("is_aligned", False):
                claimed[act_code] = objective

            cleaned.append(dict(act))

        dc_prevented = len(activities) - len(
            [a for a in cleaned if a.get("is_aligned", False)]
        )
        if dc_prevented > 0:
            logger.info(
                "Double-counting prevention: %d activities adjusted", dc_prevented,
            )
        return cleaned

    # ------------------------------------------------------------------
    # CapEx Plans
    # ------------------------------------------------------------------

    def register_capex_plan(
        self,
        org_id: str,
        activity_code: str,
        plan_data: Dict[str, Any],
    ) -> CapExPlanResult:
        """
        Register a CapEx plan for taxonomy alignment.

        CapEx plans allow inclusion of CapEx amounts that are not yet
        taxonomy-aligned but are part of a credible plan to become
        aligned within a specified period (max 10 years per DA).

        Args:
            org_id: Organization identifier.
            activity_code: Taxonomy activity code.
            plan_data: Dict with keys: plan_start_year, plan_end_year,
                total_capex_eur, objective, description.

        Returns:
            CapExPlanResult with plan registration details.
        """
        plan_start = plan_data.get("plan_start_year", 2025)
        plan_end = plan_data.get("plan_end_year", 2030)
        total = Decimal(str(plan_data.get("total_capex_eur", 0)))
        objective = plan_data.get("objective", "climate_mitigation")
        description = plan_data.get("description", "")

        # Validate plan duration
        max_years = self.config.capex_plan_max_years if hasattr(self.config, 'capex_plan_max_years') else 10
        duration = plan_end - plan_start
        if duration > max_years:
            logger.warning(
                "CapEx plan duration %d years exceeds maximum %d years",
                duration, max_years,
            )
            plan_end = plan_start + max_years
            duration = max_years

        # Annual CapEx
        annual = total / Decimal(str(max(duration, 1)))

        provenance = _sha256(
            f"capex_plan:{org_id}:{activity_code}:{plan_start}:{plan_end}:{float(total)}"
        )

        plan = CapExPlanResult(
            org_id=org_id,
            activity_code=activity_code,
            plan_start_year=plan_start,
            plan_end_year=plan_end,
            total_capex_eur=total,
            annual_capex_eur=annual.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            objective=objective,
            description=description,
            status="approved",
            provenance_hash=provenance,
        )

        self._capex_plans[plan.plan_id] = plan
        self._org_capex_plans.setdefault(org_id, []).append(plan.plan_id)

        logger.info(
            "Registered CapEx plan %s for %s/%s: EUR %s over %d years (EUR %s/yr)",
            plan.plan_id, org_id, activity_code,
            float(total), duration, float(annual),
        )
        return plan

    # ------------------------------------------------------------------
    # Objective Breakdown
    # ------------------------------------------------------------------

    def get_objective_breakdown(
        self,
        org_id: str,
        period: str,
        kpi_type: str = "turnover",
    ) -> ObjectiveBreakdown:
        """
        Get KPI numerator disaggregated by environmental objective.

        Args:
            org_id: Organization identifier.
            period: Reporting period.
            kpi_type: KPI type (turnover, capex, opex).

        Returns:
            ObjectiveBreakdown with per-objective amounts and percentages.
        """
        cache_key = f"{org_id}:{period}"
        kpi_result = self._kpi_results.get(cache_key)

        objectives: Dict[str, Dict[str, Any]] = {}
        total_aligned = Decimal("0")

        if kpi_result:
            if kpi_type == "turnover":
                breakdown = kpi_result.turnover.objective_breakdown
                total_aligned = kpi_result.turnover.aligned_turnover_eur
            elif kpi_type == "capex":
                breakdown = kpi_result.capex.objective_breakdown
                total_aligned = kpi_result.capex.aligned_capex_eur
            elif kpi_type == "opex":
                breakdown = kpi_result.opex.objective_breakdown
                total_aligned = kpi_result.opex.aligned_opex_eur
            else:
                breakdown = {}

            for obj, amount in breakdown.items():
                obj_pct = self._safe_pct(amount, total_aligned)
                obj_name = self._objective_display_name(obj)
                objectives[obj] = {
                    "name": obj_name,
                    "amount_eur": float(amount),
                    "pct_of_aligned": float(obj_pct),
                }

        provenance = _sha256(
            f"obj_breakdown:{org_id}:{period}:{kpi_type}:{float(total_aligned)}"
        )

        return ObjectiveBreakdown(
            org_id=org_id,
            period=period,
            kpi_type=kpi_type,
            total_aligned_eur=total_aligned,
            objectives=objectives,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_kpi_dashboard(
        self,
        org_id: str,
        period: str,
    ) -> KPIDashboard:
        """
        Get executive KPI dashboard for an organization-period.

        Args:
            org_id: Organization identifier.
            period: Reporting period.

        Returns:
            KPIDashboard with all three KPI percentages and breakdowns.
        """
        cache_key = f"{org_id}:{period}"
        kpi_result = self._kpi_results.get(cache_key)

        plan_count = len(self._org_capex_plans.get(org_id, []))

        if kpi_result:
            enabling_total = (
                kpi_result.turnover.enabling_eur
                + kpi_result.capex.enabling_eur
                + kpi_result.opex.enabling_eur
            )
            transitional_total = (
                kpi_result.turnover.transitional_eur
                + kpi_result.capex.transitional_eur
                + kpi_result.opex.transitional_eur
            )
            activity_count = max(
                kpi_result.turnover.activity_count,
                kpi_result.capex.activity_count,
                kpi_result.opex.activity_count,
            )

            provenance = _sha256(
                f"dashboard:{org_id}:{period}:{kpi_result.provenance_hash}"
            )

            return KPIDashboard(
                org_id=org_id,
                period=period,
                turnover_aligned_pct=kpi_result.turnover.aligned_pct,
                capex_aligned_pct=kpi_result.capex.aligned_pct,
                opex_aligned_pct=kpi_result.opex.aligned_pct,
                turnover_eligible_pct=kpi_result.turnover.eligible_pct,
                capex_eligible_pct=kpi_result.capex.eligible_pct,
                opex_eligible_pct=kpi_result.opex.eligible_pct,
                total_enabling_eur=enabling_total,
                total_transitional_eur=transitional_total,
                capex_plan_count=plan_count,
                activity_count=activity_count,
                provenance_hash=provenance,
            )

        provenance = _sha256(f"dashboard:{org_id}:{period}:empty")
        return KPIDashboard(
            org_id=org_id,
            period=period,
            capex_plan_count=plan_count,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Period Comparison
    # ------------------------------------------------------------------

    def compare_periods(
        self,
        org_id: str,
        period1: str,
        period2: str,
    ) -> Dict[str, Any]:
        """
        Compare KPI results across two reporting periods.

        Args:
            org_id: Organization identifier.
            period1: First period (earlier).
            period2: Second period (later).

        Returns:
            Dict with period-over-period changes for each KPI.
        """
        result1 = self._kpi_results.get(f"{org_id}:{period1}")
        result2 = self._kpi_results.get(f"{org_id}:{period2}")

        comparison: Dict[str, Any] = {
            "org_id": org_id,
            "period_1": period1,
            "period_2": period2,
            "data_available": result1 is not None and result2 is not None,
        }

        if result1 and result2:
            for kpi_name in ("turnover", "capex", "opex"):
                kpi1 = getattr(result1, kpi_name)
                kpi2 = getattr(result2, kpi_name)

                aligned_pct_1 = float(kpi1.aligned_pct)
                aligned_pct_2 = float(kpi2.aligned_pct)
                delta = aligned_pct_2 - aligned_pct_1

                comparison[kpi_name] = {
                    f"{period1}_aligned_pct": aligned_pct_1,
                    f"{period2}_aligned_pct": aligned_pct_2,
                    "delta_pct": round(delta, 2),
                    "direction": "improved" if delta > 0 else (
                        "declined" if delta < 0 else "unchanged"
                    ),
                    f"{period1}_eligible_pct": float(kpi1.eligible_pct),
                    f"{period2}_eligible_pct": float(kpi2.eligible_pct),
                }

        return comparison

    # ------------------------------------------------------------------
    # Denominator Validation
    # ------------------------------------------------------------------

    def validate_denominators(
        self,
        turnover_total: float,
        capex_total: float,
        opex_total: float,
    ) -> Dict[str, Any]:
        """
        Validate KPI denominator values for completeness and plausibility.

        Args:
            turnover_total: Total net revenue (IAS 1 / IFRS 15).
            capex_total: Total CapEx (IAS 16 + IAS 38 + IFRS 16 + IAS 40).
            opex_total: Total narrow OpEx.

        Returns:
            Validation result with warnings for each KPI denominator.
        """
        warnings: List[str] = []
        valid = True

        if turnover_total <= 0:
            warnings.append("Turnover denominator is zero or negative")
            valid = False
        if capex_total < 0:
            warnings.append("CapEx denominator is negative")
            valid = False
        if opex_total < 0:
            warnings.append("OpEx denominator is negative")
            valid = False

        # Plausibility checks
        if capex_total > 0 and turnover_total > 0:
            capex_intensity = capex_total / turnover_total
            if capex_intensity > 1.0:
                warnings.append(
                    f"CapEx intensity ({capex_intensity:.2%}) exceeds 100% of turnover"
                )
            if capex_intensity < 0.01:
                warnings.append(
                    f"CapEx intensity ({capex_intensity:.2%}) is unusually low"
                )

        if opex_total > 0 and turnover_total > 0:
            opex_intensity = opex_total / turnover_total
            if opex_intensity > 0.5:
                warnings.append(
                    f"Narrow OpEx intensity ({opex_intensity:.2%}) is unusually high"
                )

        return {
            "valid": valid,
            "turnover_total_eur": turnover_total,
            "capex_total_eur": capex_total,
            "opex_total_eur": opex_total,
            "warnings": warnings,
            "accounting_standards": {
                "turnover": "IAS 1 (Revenue per IFRS 15)",
                "capex": "IAS 16 + IAS 38 + IFRS 16 + IAS 40 additions",
                "opex": "Narrow OpEx: R&D, renovation, short-term lease, maintenance, other direct",
            },
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _safe_pct(self, numerator: Decimal, denominator: Decimal) -> Decimal:
        """
        Calculate percentage with safe division.

        Returns percentage (0-100) with 2 decimal places.
        Returns Decimal('0') if denominator is zero.
        """
        if denominator <= 0:
            return Decimal("0")
        pct = (numerator / denominator * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        return min(pct, Decimal("100"))

    def _objective_display_name(self, objective: str) -> str:
        """Get human-readable display name for an objective."""
        names: Dict[str, str] = {
            "climate_mitigation": "Climate Change Mitigation",
            "climate_adaptation": "Climate Change Adaptation",
            "water": "Water and Marine Resources",
            "circular_economy": "Circular Economy",
            "pollution": "Pollution Prevention and Control",
            "biodiversity": "Biodiversity and Ecosystems",
        }
        return names.get(objective, objective)
