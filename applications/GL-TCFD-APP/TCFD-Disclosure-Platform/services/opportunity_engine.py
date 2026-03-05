"""
Opportunity Engine -- Climate Opportunity Assessment and Sizing

Implements climate opportunity assessment across the five TCFD opportunity
categories: resource efficiency, energy source, products/services, markets,
and resilience.

Provides:
  - Opportunity creation and lifecycle management
  - Revenue opportunity sizing (TAM/SAM/SOM methodology)
  - Cost savings estimation with payback calculation
  - Investment ROI calculation (NPV, IRR, payback)
  - Impact vs feasibility prioritization matrix
  - Pipeline tracking (identified/evaluated/approved/in_progress/realized)
  - Green financing assessment and eligibility
  - Green revenue share tracking
  - Sector-specific opportunity templates
  - Opportunity disclosure generation
  - Portfolio-level NPV analysis

Reference:
    - TCFD Final Report, Appendix 1: Climate-Related Opportunities
    - IFRS S2 Paragraphs 10-12 (Strategy: Opportunities)
    - ICMA Green Bond Principles (2021)

Example:
    >>> engine = OpportunityEngine(config)
    >>> opp = await engine.create_opportunity("org-1", "tenant-1", opp_data)
    >>> prioritized = await engine.prioritize_opportunities("org-1")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    OpportunityCategory,
    SectorType,
    TCFDAppConfig,
    TimeHorizon,
)
from .models import (
    ClimateOpportunity,
    CreateClimateOpportunityRequest,
    InvestmentAnalysis,
    OpportunityAssessment,
    OpportunityPipeline,
    RevenueSizing,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector-Specific Opportunity Templates
# ---------------------------------------------------------------------------

OPPORTUNITY_TEMPLATES: Dict[SectorType, List[Dict[str, Any]]] = {
    SectorType.ENERGY: [
        {"name": "Renewable energy generation", "category": "energy_source", "typical_roi_pct": 15},
        {"name": "Battery storage deployment", "category": "energy_source", "typical_roi_pct": 12},
        {"name": "Hydrogen production", "category": "products_services", "typical_roi_pct": 8},
        {"name": "Grid modernization services", "category": "markets", "typical_roi_pct": 18},
        {"name": "Carbon capture utilization", "category": "products_services", "typical_roi_pct": 5},
    ],
    SectorType.TRANSPORTATION: [
        {"name": "Electric vehicle fleet", "category": "resource_efficiency", "typical_roi_pct": 20},
        {"name": "Sustainable aviation fuel", "category": "energy_source", "typical_roi_pct": 10},
        {"name": "Mobility-as-a-service", "category": "markets", "typical_roi_pct": 22},
        {"name": "Route optimization AI", "category": "resource_efficiency", "typical_roi_pct": 30},
        {"name": "Last-mile electrification", "category": "products_services", "typical_roi_pct": 16},
    ],
    SectorType.MATERIALS_BUILDINGS: [
        {"name": "Green building certification", "category": "products_services", "typical_roi_pct": 14},
        {"name": "Circular materials recovery", "category": "resource_efficiency", "typical_roi_pct": 18},
        {"name": "Energy efficiency retrofits", "category": "resource_efficiency", "typical_roi_pct": 25},
        {"name": "Heat pump deployment", "category": "energy_source", "typical_roi_pct": 20},
        {"name": "Low-carbon cement/steel", "category": "products_services", "typical_roi_pct": 8},
    ],
    SectorType.AGRICULTURE_FOOD_FOREST: [
        {"name": "Precision agriculture", "category": "resource_efficiency", "typical_roi_pct": 22},
        {"name": "Alternative proteins", "category": "products_services", "typical_roi_pct": 15},
        {"name": "Carbon farming credits", "category": "markets", "typical_roi_pct": 12},
        {"name": "Sustainable packaging", "category": "products_services", "typical_roi_pct": 10},
        {"name": "Water efficiency systems", "category": "resource_efficiency", "typical_roi_pct": 18},
    ],
    SectorType.BANKING: [
        {"name": "Green bond issuance", "category": "markets", "typical_roi_pct": 5},
        {"name": "Sustainability-linked loans", "category": "products_services", "typical_roi_pct": 8},
        {"name": "Climate analytics platform", "category": "products_services", "typical_roi_pct": 30},
        {"name": "Transition finance", "category": "markets", "typical_roi_pct": 10},
        {"name": "Carbon trading desk", "category": "markets", "typical_roi_pct": 25},
    ],
    SectorType.INSURANCE: [
        {"name": "Parametric insurance products", "category": "products_services", "typical_roi_pct": 15},
        {"name": "Climate risk analytics", "category": "products_services", "typical_roi_pct": 25},
        {"name": "Resilience advisory services", "category": "markets", "typical_roi_pct": 20},
        {"name": "Renewable energy insurance", "category": "markets", "typical_roi_pct": 12},
        {"name": "Nature-based solutions coverage", "category": "products_services", "typical_roi_pct": 10},
    ],
    SectorType.TECHNOLOGY_MEDIA: [
        {"name": "Data center renewable PPAs", "category": "energy_source", "typical_roi_pct": 18},
        {"name": "Climate SaaS products", "category": "products_services", "typical_roi_pct": 35},
        {"name": "E-waste circular economy", "category": "resource_efficiency", "typical_roi_pct": 12},
        {"name": "Carbon accounting software", "category": "products_services", "typical_roi_pct": 40},
        {"name": "Energy optimization AI", "category": "resource_efficiency", "typical_roi_pct": 28},
    ],
}

# ---------------------------------------------------------------------------
# Cost Savings Category Benchmarks (% of baseline cost)
# ---------------------------------------------------------------------------

_SAVINGS_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "energy": {"low": Decimal("5"), "mid": Decimal("15"), "high": Decimal("30")},
    "water": {"low": Decimal("3"), "mid": Decimal("10"), "high": Decimal("20")},
    "waste": {"low": Decimal("5"), "mid": Decimal("12"), "high": Decimal("25")},
    "materials": {"low": Decimal("2"), "mid": Decimal("8"), "high": Decimal("18")},
    "carbon": {"low": Decimal("3"), "mid": Decimal("10"), "high": Decimal("25")},
}


class OpportunityEngine:
    """
    Climate opportunity assessment and sizing engine.

    Manages the full lifecycle of climate opportunities from identification
    through evaluation, approval, implementation, and realization. Provides
    financial analysis (ROI, NPV, IRR, payback) and pipeline management.

    Attributes:
        config: Application configuration.
        _opportunities: In-memory store keyed by org_id.
        _assessments: Full opportunity assessments by opp_id.
        _green_revenue: Green revenue tracking by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        self.config = config or TCFDAppConfig()
        self._opportunities: Dict[str, List[ClimateOpportunity]] = {}
        self._assessments: Dict[str, OpportunityAssessment] = {}
        self._green_revenue: Dict[str, Dict[int, Decimal]] = {}
        logger.info("OpportunityEngine initialized")

    # ------------------------------------------------------------------
    # Opportunity CRUD
    # ------------------------------------------------------------------

    async def create_opportunity(
        self, org_id: str, tenant_id: str,
        opp_data: CreateClimateOpportunityRequest,
    ) -> ClimateOpportunity:
        """
        Create and register a climate opportunity.

        Calculates ROI from revenue potential and cost savings against
        investment required.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            opp_data: Opportunity creation request.

        Returns:
            Created ClimateOpportunity with computed ROI.
        """
        roi = Decimal("0")
        if opp_data.investment_required_usd > 0:
            net = opp_data.revenue_potential_usd + opp_data.cost_savings_usd
            roi = (net / opp_data.investment_required_usd).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )

        opp = ClimateOpportunity(
            tenant_id=tenant_id,
            org_id=org_id,
            category=opp_data.category,
            name=opp_data.name,
            description=opp_data.description,
            revenue_potential_usd=opp_data.revenue_potential_usd,
            cost_savings_usd=opp_data.cost_savings_usd,
            investment_required_usd=opp_data.investment_required_usd,
            roi_estimate_pct=roi,
            timeline=opp_data.timeline,
            feasibility_score=opp_data.feasibility_score,
            priority_score=opp_data.priority_score,
        )
        if org_id not in self._opportunities:
            self._opportunities[org_id] = []
        self._opportunities[org_id].append(opp)
        logger.info("Created opportunity '%s' for org %s, ROI=%.2f", opp.name, org_id, roi)
        return opp

    async def update_opportunity(
        self, opp_id: str, updates: Dict[str, Any],
    ) -> ClimateOpportunity:
        """
        Update an existing opportunity.

        Args:
            opp_id: Opportunity identifier.
            updates: Fields to update.

        Returns:
            Updated ClimateOpportunity.

        Raises:
            ValueError: If opportunity not found.
        """
        for org_id, opps in self._opportunities.items():
            for i, opp in enumerate(opps):
                if opp.id == opp_id:
                    data = opp.model_dump()
                    data.update(updates)
                    data["updated_at"] = _now()
                    data["provenance_hash"] = ""
                    updated = ClimateOpportunity(**data)
                    self._opportunities[org_id][i] = updated
                    logger.info("Updated opportunity %s", opp_id)
                    return updated
        raise ValueError(f"Opportunity {opp_id} not found")

    async def list_opportunities(
        self, org_id: str, filters: Optional[Dict[str, Any]] = None,
    ) -> List[ClimateOpportunity]:
        """
        List opportunities with optional filters.

        Args:
            org_id: Organization identifier.
            filters: Optional filters (category, status, timeline).

        Returns:
            Filtered list of ClimateOpportunity objects.
        """
        opps = list(self._opportunities.get(org_id, []))
        if filters:
            if "category" in filters:
                cat = OpportunityCategory(filters["category"])
                opps = [o for o in opps if o.category == cat]
            if "status" in filters:
                opps = [o for o in opps if o.status == filters["status"]]
            if "timeline" in filters:
                th = TimeHorizon(filters["timeline"])
                opps = [o for o in opps if o.timeline == th]
        return opps

    # ------------------------------------------------------------------
    # Revenue Sizing (TAM/SAM/SOM)
    # ------------------------------------------------------------------

    async def size_revenue_opportunity(
        self, org_id: str, opp_id: str,
        tam_usd: Optional[Decimal] = None,
        market_share_pct: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Size the revenue opportunity with TAM/SAM/SOM methodology.

        Args:
            org_id: Organization identifier.
            opp_id: Opportunity identifier.
            tam_usd: Total Addressable Market (optional override).
            market_share_pct: Target market share percentage.

        Returns:
            Dict with revenue sizing (TAM/SAM/SOM), ramp-up profile,
            confidence level, and RevenueSizing record.
        """
        opp = self._find_opportunity(org_id, opp_id)
        if opp is None:
            raise ValueError(f"Opportunity {opp_id} not found")

        # TAM defaults to 10x revenue potential; SAM = 30% of TAM; SOM = market_share of SAM
        tam = tam_usd or (opp.revenue_potential_usd * Decimal("10"))
        sam = tam * Decimal("0.3")
        share_pct = market_share_pct or Decimal("10")
        som = (sam * share_pct / 100).quantize(Decimal("0.01"))

        # Revenue ramp-up profile: S-curve over 5 years
        year_1 = som * Decimal("0.1")
        year_2 = som * Decimal("0.25")
        year_3 = som * Decimal("0.55")
        year_4 = som * Decimal("0.80")
        year_5 = som

        # Growth rate (CAGR over 5 years)
        growth_rate = Decimal("0")
        if year_1 > 0:
            ratio = float(year_5 / year_1)
            if ratio > 0:
                growth_rate = Decimal(str(round((ratio ** 0.2 - 1) * 100, 1)))

        # Create revenue sizing model
        sizing = RevenueSizing(
            tenant_id="default",
            opportunity_id=opp_id,
            addressable_market_usd=tam,
            market_share_pct=share_pct,
            revenue_year_1_usd=year_1,
            revenue_year_3_usd=year_3,
            revenue_year_5_usd=year_5,
            growth_rate_pct=growth_rate,
        )

        # Confidence based on feasibility score
        if opp.feasibility_score >= 4:
            confidence = "high"
        elif opp.feasibility_score >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "opportunity_id": opp_id,
            "tam_usd": str(tam),
            "sam_usd": str(sam),
            "som_usd": str(som),
            "market_share_pct": str(share_pct),
            "revenue_ramp": {
                "year_1": str(year_1.quantize(Decimal("0.01"))),
                "year_2": str(year_2.quantize(Decimal("0.01"))),
                "year_3": str(year_3.quantize(Decimal("0.01"))),
                "year_4": str(year_4.quantize(Decimal("0.01"))),
                "year_5": str(year_5.quantize(Decimal("0.01"))),
            },
            "cagr_pct": str(growth_rate),
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Cost Savings Estimation
    # ------------------------------------------------------------------

    async def estimate_cost_savings(
        self, org_id: str, category: OpportunityCategory,
    ) -> Dict[str, Any]:
        """
        Estimate cost savings for a given opportunity category.

        Aggregates all opportunities in the category, calculates total
        savings, investment, and simple payback period.

        Args:
            org_id: Organization identifier.
            category: Opportunity category to filter.

        Returns:
            Dict with total savings, investment, payback, and per-opportunity detail.
        """
        opps = [o for o in self._opportunities.get(org_id, []) if o.category == category]
        total_savings = sum(o.cost_savings_usd for o in opps)
        total_investment = sum(o.investment_required_usd for o in opps)
        payback_years = Decimal("0")
        if total_savings > 0:
            payback_years = (total_investment / total_savings).quantize(Decimal("0.1"))

        per_opp: List[Dict[str, str]] = []
        for opp in opps:
            opp_payback = Decimal("0")
            if opp.cost_savings_usd > 0:
                opp_payback = (opp.investment_required_usd / opp.cost_savings_usd).quantize(
                    Decimal("0.1")
                )
            per_opp.append({
                "opportunity_id": opp.id,
                "name": opp.name,
                "savings": str(opp.cost_savings_usd),
                "investment": str(opp.investment_required_usd),
                "payback_years": str(opp_payback),
            })

        return {
            "org_id": org_id,
            "category": category.value,
            "opportunity_count": len(opps),
            "total_cost_savings": str(total_savings),
            "total_investment": str(total_investment),
            "payback_years": str(payback_years),
            "per_opportunity": per_opp,
        }

    async def estimate_savings_by_resource(
        self, org_id: str,
        resource_category: str,
        baseline_cost_usd: Decimal,
        scenario: str = "mid",
    ) -> Dict[str, Any]:
        """
        Estimate savings for a specific resource category using benchmarks.

        Args:
            org_id: Organization identifier.
            resource_category: One of energy, water, waste, materials, carbon.
            baseline_cost_usd: Current annual cost for this resource.
            scenario: Savings scenario (low, mid, high).

        Returns:
            Dict with estimated savings, range, and payback.
        """
        benchmarks = _SAVINGS_BENCHMARKS.get(resource_category.lower(), {})
        if not benchmarks:
            return {
                "org_id": org_id,
                "resource_category": resource_category,
                "message": "No benchmarks available for this resource category",
            }

        savings_pct = benchmarks.get(scenario, benchmarks.get("mid", Decimal("10")))
        estimated_savings = (baseline_cost_usd * savings_pct / 100).quantize(Decimal("0.01"))

        # Typical implementation cost is 1.5-3x annual savings
        implementation_cost = (estimated_savings * Decimal("2.5")).quantize(Decimal("0.01"))
        payback = Decimal("0")
        if estimated_savings > 0:
            payback = (implementation_cost / estimated_savings).quantize(Decimal("0.1"))

        return {
            "org_id": org_id,
            "resource_category": resource_category,
            "baseline_cost_usd": str(baseline_cost_usd),
            "savings_pct": str(savings_pct),
            "estimated_annual_savings_usd": str(estimated_savings),
            "estimated_implementation_cost_usd": str(implementation_cost),
            "payback_years": str(payback),
            "savings_range": {
                "low": str(
                    (baseline_cost_usd * benchmarks.get("low", Decimal("5")) / 100).quantize(
                        Decimal("0.01")
                    )
                ),
                "mid": str(
                    (baseline_cost_usd * benchmarks.get("mid", Decimal("10")) / 100).quantize(
                        Decimal("0.01")
                    )
                ),
                "high": str(
                    (baseline_cost_usd * benchmarks.get("high", Decimal("20")) / 100).quantize(
                        Decimal("0.01")
                    )
                ),
            },
        }

    # ------------------------------------------------------------------
    # Investment ROI Calculation
    # ------------------------------------------------------------------

    async def calculate_investment_roi(
        self, org_id: str, opp_id: str,
        discount_rate: Optional[Decimal] = None,
        projection_years: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate detailed ROI analysis for an opportunity.

        Computes ROI percentage, simple payback, NPV, and approximate IRR
        over the specified projection horizon.

        Args:
            org_id: Organization identifier.
            opp_id: Opportunity identifier.
            discount_rate: Discount rate for NPV (defaults to config).
            projection_years: Number of years for NPV projection.

        Returns:
            Dict with ROI, NPV, IRR, payback, and year-by-year cash flows.
        """
        opp = self._find_opportunity(org_id, opp_id)
        if opp is None:
            raise ValueError(f"Opportunity {opp_id} not found")

        rate = discount_rate or self.config.default_discount_rate
        total_benefit = opp.revenue_potential_usd + opp.cost_savings_usd

        # ROI calculation
        roi = Decimal("0")
        if opp.investment_required_usd > 0:
            roi = (
                (total_benefit - opp.investment_required_usd) / opp.investment_required_usd * 100
            ).quantize(Decimal("0.1"))

        # Simple payback
        annual_benefit = total_benefit / Decimal(str(max(projection_years, 1)))
        payback = Decimal("0")
        if annual_benefit > 0:
            payback = (opp.investment_required_usd / annual_benefit).quantize(Decimal("0.1"))

        # NPV calculation (initial investment at year 0, then equal annual benefits)
        annual_cf = annual_benefit
        npv = -opp.investment_required_usd
        yearly_cash_flows: Dict[str, str] = {"0": str(-opp.investment_required_usd)}

        for t in range(1, projection_years + 1):
            discount_factor = (Decimal("1") + rate) ** t
            pv = annual_cf / discount_factor
            npv += pv
            yearly_cash_flows[str(t)] = str(annual_cf.quantize(Decimal("0.01")))

        npv = npv.quantize(Decimal("0.01"))

        # Approximate IRR using bisection
        irr = self._approximate_irr(opp.investment_required_usd, annual_cf, projection_years)

        # Build investment analysis model
        investment = InvestmentAnalysis(
            tenant_id="default",
            opportunity_id=opp_id,
            total_investment_usd=opp.investment_required_usd,
            npv_usd=npv,
            irr_pct=irr or Decimal("0"),
            payback_period_years=payback,
            risk_adjusted_return_pct=roi * Decimal("0.8"),
            discount_rate_used=rate,
        )

        # Store full assessment
        self._assessments[opp_id] = OpportunityAssessment(
            tenant_id="default",
            org_id=org_id,
            opportunity_id=opp_id,
            investment_analysis=investment,
            total_npv_usd=npv,
            recommendation="proceed" if npv > 0 and roi > 15 else (
                "defer" if npv > 0 else "reject"
            ),
        )

        recommendation = "Pursue" if roi > Decimal("15") and npv > 0 else (
            "Evaluate further" if npv > 0 else "Reconsider"
        )

        return {
            "opportunity_id": opp_id,
            "opportunity_name": opp.name,
            "investment_required": str(opp.investment_required_usd),
            "total_benefit": str(total_benefit),
            "roi_pct": str(roi),
            "payback_period_years": str(payback),
            "npv_usd": str(npv),
            "irr_pct": str(irr) if irr else "N/A",
            "discount_rate": str(rate),
            "projection_years": projection_years,
            "yearly_cash_flows": yearly_cash_flows,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Prioritization
    # ------------------------------------------------------------------

    async def prioritize_opportunities(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Prioritize opportunities by impact vs feasibility matrix.

        Uses a weighted composite score: impact (40%) + feasibility (60%).
        Impact is derived from total economic value normalized to 1-5 scale.

        Args:
            org_id: Organization identifier.

        Returns:
            Ranked list of opportunities with composite scores and quadrant.
        """
        opps = self._opportunities.get(org_id, [])
        scored: List[Dict[str, Any]] = []

        for opp in opps:
            total_value = opp.revenue_potential_usd + opp.cost_savings_usd
            impact_score = min(int(total_value / Decimal("100000")) + 1, 5)
            composite = (impact_score * 2 + opp.feasibility_score * 3) / 5

            # Determine quadrant
            quadrant = self._determine_quadrant(impact_score, opp.feasibility_score)

            scored.append({
                "opportunity_id": opp.id,
                "name": opp.name,
                "category": opp.category.value,
                "impact_score": impact_score,
                "feasibility_score": opp.feasibility_score,
                "composite_priority": round(composite, 1),
                "total_value": str(total_value),
                "investment_required": str(opp.investment_required_usd),
                "timeline": opp.timeline.value if hasattr(opp.timeline, "value") else str(opp.timeline),
                "status": opp.status,
                "quadrant": quadrant,
            })

        scored.sort(key=lambda x: x["composite_priority"], reverse=True)
        for i, item in enumerate(scored):
            item["rank"] = i + 1
        return scored

    # ------------------------------------------------------------------
    # Pipeline Tracking
    # ------------------------------------------------------------------

    async def track_pipeline(self, org_id: str) -> Dict[str, Any]:
        """
        Track opportunity pipeline with stage counts and values.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with pipeline stages, conversion rates, and velocity.
        """
        opps = self._opportunities.get(org_id, [])
        stages = {
            "identified": 0,
            "evaluated": 0,
            "approved": 0,
            "in_progress": 0,
            "realized": 0,
        }
        value_by_stage: Dict[str, Decimal] = {k: Decimal("0") for k in stages}
        investment_by_stage: Dict[str, Decimal] = {k: Decimal("0") for k in stages}

        for opp in opps:
            stage = opp.status if opp.status in stages else "identified"
            stages[stage] += 1
            value_by_stage[stage] += opp.revenue_potential_usd + opp.cost_savings_usd
            investment_by_stage[stage] += opp.investment_required_usd

        total = len(opps)
        realized = stages["realized"]
        conversion_rate = Decimal("0")
        if total > 0:
            conversion_rate = (Decimal(str(realized)) / Decimal(str(total)) * 100).quantize(
                Decimal("0.1")
            )

        # Pipeline velocity (avg days in each stage)
        in_progress_pct = Decimal("0")
        if total > 0:
            in_progress_pct = (
                Decimal(str(stages["in_progress"] + stages["realized"]))
                / Decimal(str(total))
                * 100
            ).quantize(Decimal("0.1"))

        return {
            "org_id": org_id,
            "total_opportunities": total,
            "by_stage_count": stages,
            "by_stage_value": {k: str(v) for k, v in value_by_stage.items()},
            "by_stage_investment": {k: str(v) for k, v in investment_by_stage.items()},
            "conversion_rate_pct": str(conversion_rate),
            "execution_rate_pct": str(in_progress_pct),
            "total_pipeline_value": str(sum(value_by_stage.values())),
            "total_pipeline_investment": str(sum(investment_by_stage.values())),
        }

    async def build_pipeline_model(
        self, org_id: str, tenant_id: str,
    ) -> OpportunityPipeline:
        """
        Build a complete OpportunityPipeline model for the organization.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.

        Returns:
            OpportunityPipeline model with aggregate totals and NPV.
        """
        opps = self._opportunities.get(org_id, [])
        total_revenue = sum(o.revenue_potential_usd for o in opps)
        total_savings = sum(o.cost_savings_usd for o in opps)
        total_investment = sum(o.investment_required_usd for o in opps)

        # Calculate pipeline NPV (simple: benefits - investment, discounted)
        rate = self.config.default_discount_rate
        total_benefit = total_revenue + total_savings
        annual_benefit = total_benefit / Decimal("5")
        npv = -total_investment
        for t in range(1, 11):
            npv += annual_benefit / ((Decimal("1") + rate) ** t)

        pipeline = OpportunityPipeline(
            tenant_id=tenant_id,
            org_id=org_id,
            opportunities=opps,
            total_revenue_potential_usd=total_revenue,
            total_cost_savings_usd=total_savings,
            total_investment_required_usd=total_investment,
            pipeline_npv_usd=npv.quantize(Decimal("0.01")),
        )

        logger.info(
            "Pipeline model for org %s: %d opps, NPV=$%.0f",
            org_id, len(opps), npv,
        )
        return pipeline

    # ------------------------------------------------------------------
    # Green Revenue Tracking
    # ------------------------------------------------------------------

    async def record_green_revenue(
        self, org_id: str, year: int, green_revenue_usd: Decimal,
    ) -> None:
        """Record green revenue for a given year."""
        if org_id not in self._green_revenue:
            self._green_revenue[org_id] = {}
        self._green_revenue[org_id][year] = green_revenue_usd
        logger.info("Recorded green revenue $%.0f for org %s year %d", green_revenue_usd, org_id, year)

    async def get_green_revenue_share(
        self, org_id: str, total_revenue_usd: Decimal,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate green revenue as a percentage of total revenue.

        This is ISSB cross-industry metric #4 (climate opportunity revenue).

        Args:
            org_id: Organization identifier.
            total_revenue_usd: Total organizational revenue.
            year: Reporting year (defaults to config year).

        Returns:
            Dict with green revenue share and time series.
        """
        report_year = year or self.config.reporting_year
        org_data = self._green_revenue.get(org_id, {})
        green_rev = org_data.get(report_year, Decimal("0"))

        share_pct = Decimal("0")
        if total_revenue_usd > 0:
            share_pct = (green_rev / total_revenue_usd * 100).quantize(Decimal("0.1"))

        # Build time series
        time_series: Dict[str, str] = {}
        for yr, rev in sorted(org_data.items()):
            time_series[str(yr)] = str(rev)

        return {
            "org_id": org_id,
            "reporting_year": report_year,
            "green_revenue_usd": str(green_rev),
            "total_revenue_usd": str(total_revenue_usd),
            "green_revenue_share_pct": str(share_pct),
            "issb_metric": "ISSB-CI-04",
            "time_series": time_series,
        }

    # ------------------------------------------------------------------
    # Green Financing Assessment
    # ------------------------------------------------------------------

    async def assess_green_financing(self, org_id: str) -> Dict[str, Any]:
        """
        Assess green financing eligibility for opportunities.

        Evaluates which opportunities qualify for green bonds,
        sustainability-linked loans, and other green finance instruments
        based on ICMA Green Bond Principles.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with eligible opportunities, instruments, and rate benefit.
        """
        opps = self._opportunities.get(org_id, [])

        # Green bond eligible categories
        eligible_categories = {
            OpportunityCategory.ENERGY_SOURCE,
            OpportunityCategory.RESOURCE_EFFICIENCY,
            OpportunityCategory.RESILIENCE,
        }
        eligible = [o for o in opps if o.category in eligible_categories]
        total_eligible_investment = sum(o.investment_required_usd for o in eligible)
        total_eligible_value = sum(o.revenue_potential_usd + o.cost_savings_usd for o in eligible)

        # Sustainability-linked loan eligible (broader scope)
        sll_eligible = [
            o for o in opps
            if o.category in eligible_categories or o.category == OpportunityCategory.PRODUCTS_SERVICES
        ]
        sll_investment = sum(o.investment_required_usd for o in sll_eligible)

        # Rate benefit estimate (bps)
        if total_eligible_investment >= Decimal("50000000"):
            rate_benefit = "30-50"
        elif total_eligible_investment >= Decimal("10000000"):
            rate_benefit = "15-35"
        else:
            rate_benefit = "10-25"

        return {
            "org_id": org_id,
            "green_bond_eligible": {
                "opportunity_count": len(eligible),
                "total_investment": str(total_eligible_investment),
                "total_value": str(total_eligible_value),
            },
            "sustainability_linked_loan_eligible": {
                "opportunity_count": len(sll_eligible),
                "total_investment": str(sll_investment),
            },
            "financing_instruments": [
                {"instrument": "Green bonds (ICMA GBP)", "eligibility": "eligible" if eligible else "not_eligible"},
                {"instrument": "Sustainability-linked loans", "eligibility": "eligible" if sll_eligible else "not_eligible"},
                {"instrument": "Green project finance", "eligibility": "eligible" if eligible else "not_eligible"},
                {"instrument": "Climate transition finance", "eligibility": "eligible"},
                {"instrument": "EU Green Bond Standard", "eligibility": "pending_assessment"},
            ],
            "estimated_rate_benefit_bps": rate_benefit,
        }

    # ------------------------------------------------------------------
    # Sector Templates
    # ------------------------------------------------------------------

    async def get_opportunity_templates(
        self, sector: SectorType,
    ) -> List[Dict[str, Any]]:
        """
        Get sector-specific opportunity templates.

        Args:
            sector: TCFD sector classification.

        Returns:
            List of opportunity templates with name, category, and typical ROI.
        """
        templates = OPPORTUNITY_TEMPLATES.get(sector, [])
        if not templates:
            return [{
                "name": "General climate opportunity",
                "category": "resource_efficiency",
                "typical_roi_pct": 15,
                "message": "No sector-specific templates available",
            }]
        return templates

    # ------------------------------------------------------------------
    # Disclosure Generation
    # ------------------------------------------------------------------

    async def generate_opportunity_disclosure(self, org_id: str) -> Dict[str, Any]:
        """
        Generate TCFD-aligned opportunity disclosure content.

        Produces structured narrative covering identified opportunities,
        categorization, financial sizing, and strategic response.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with disclosure content, breakdown, and compliance score.
        """
        opps = self._opportunities.get(org_id, [])
        total_value = sum(o.revenue_potential_usd + o.cost_savings_usd for o in opps)
        total_investment = sum(o.investment_required_usd for o in opps)

        # Category breakdown
        by_category: Dict[str, Dict[str, Any]] = {}
        for o in opps:
            cat = o.category.value
            if cat not in by_category:
                by_category[cat] = {"count": 0, "value": Decimal("0"), "investment": Decimal("0")}
            by_category[cat]["count"] += 1
            by_category[cat]["value"] += o.revenue_potential_usd + o.cost_savings_usd
            by_category[cat]["investment"] += o.investment_required_usd

        # Serialize values
        by_category_str: Dict[str, Dict[str, str]] = {}
        for cat, data in by_category.items():
            by_category_str[cat] = {
                "count": str(data["count"]),
                "value": str(data["value"]),
                "investment": str(data["investment"]),
            }

        # Timeline distribution
        by_timeline: Dict[str, int] = {}
        for o in opps:
            th_val = o.timeline.value if hasattr(o.timeline, "value") else str(o.timeline)
            by_timeline[th_val] = by_timeline.get(th_val, 0) + 1

        # Build narrative
        narrative_parts = [
            f"The organization has identified {len(opps)} climate-related opportunity/ies "
            f"with total potential value of ${total_value:,.0f} "
            f"requiring ${total_investment:,.0f} in investment. ",
        ]
        if by_category:
            cat_summary = ", ".join(
                f"{cat.replace('_', ' ').title()} ({data['count']})"
                for cat, data in by_category.items()
            )
            narrative_parts.append(f"Opportunities span: {cat_summary}. ")

        # ROI summary
        avg_roi = Decimal("0")
        roi_count = 0
        for o in opps:
            if o.roi_estimate_pct > 0:
                avg_roi += o.roi_estimate_pct
                roi_count += 1
        if roi_count > 0:
            avg_roi = (avg_roi / roi_count).quantize(Decimal("0.01"))
            narrative_parts.append(f"Average estimated ROI is {avg_roi:.0f}%. ")

        # Compliance score
        compliance_score = self._score_opportunity_disclosure(opps, by_category)

        return {
            "org_id": org_id,
            "content": "".join(narrative_parts),
            "opportunity_count": len(opps),
            "total_value": str(total_value),
            "total_investment": str(total_investment),
            "average_roi_pct": str(avg_roi),
            "by_category": by_category_str,
            "by_timeline": by_timeline,
            "compliance_score": compliance_score,
        }

    async def get_opportunity_summary(self, org_id: str) -> Dict[str, Any]:
        """
        Get a high-level summary of all opportunities for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with aggregate metrics and category coverage.
        """
        opps = self._opportunities.get(org_id, [])
        if not opps:
            return {
                "org_id": org_id,
                "opportunity_count": 0,
                "message": "No opportunities identified",
            }

        total_revenue = sum(o.revenue_potential_usd for o in opps)
        total_savings = sum(o.cost_savings_usd for o in opps)
        total_investment = sum(o.investment_required_usd for o in opps)
        total_value = total_revenue + total_savings

        # Category coverage
        categories_covered = list(set(o.category.value for o in opps))
        all_categories = [c.value for c in OpportunityCategory]
        coverage_pct = len(categories_covered) / len(all_categories) * 100

        # Top 5 by value
        sorted_opps = sorted(
            opps,
            key=lambda o: o.revenue_potential_usd + o.cost_savings_usd,
            reverse=True,
        )
        top_5 = [
            {"name": o.name, "category": o.category.value, "value": str(o.revenue_potential_usd + o.cost_savings_usd)}
            for o in sorted_opps[:5]
        ]

        return {
            "org_id": org_id,
            "opportunity_count": len(opps),
            "total_revenue_potential": str(total_revenue),
            "total_cost_savings": str(total_savings),
            "total_value": str(total_value),
            "total_investment_required": str(total_investment),
            "categories_covered": categories_covered,
            "category_coverage_pct": round(coverage_pct, 1),
            "top_5_opportunities": top_5,
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _find_opportunity(
        self, org_id: str, opp_id: str,
    ) -> Optional[ClimateOpportunity]:
        """Find an opportunity by org_id and opp_id."""
        for opp in self._opportunities.get(org_id, []):
            if opp.id == opp_id:
                return opp
        return None

    @staticmethod
    def _determine_quadrant(impact: int, feasibility: int) -> str:
        """
        Determine the opportunity quadrant from impact/feasibility scores.

        Quadrant mapping:
            - Quick Wins: high feasibility, high impact
            - Strategic Bets: low feasibility, high impact
            - Low Hanging Fruit: high feasibility, low impact
            - Deprioritize: low feasibility, low impact
        """
        if impact >= 3 and feasibility >= 3:
            return "quick_win"
        elif impact >= 3 and feasibility < 3:
            return "strategic_bet"
        elif impact < 3 and feasibility >= 3:
            return "low_hanging_fruit"
        else:
            return "deprioritize"

    @staticmethod
    def _approximate_irr(
        investment: Decimal,
        annual_cf: Decimal,
        years: int,
    ) -> Optional[Decimal]:
        """
        Approximate IRR using bisection method.

        Args:
            investment: Initial investment (positive value).
            annual_cf: Annual cash flow (positive value).
            years: Number of years of cash flows.

        Returns:
            Approximate IRR as a percentage, or None if not calculable.
        """
        if investment <= 0 or annual_cf <= 0:
            return None

        low, high = Decimal("-0.5"), Decimal("3.0")
        for _ in range(100):
            mid = (low + high) / 2
            npv = -investment
            for t in range(1, years + 1):
                denom = (Decimal("1") + mid) ** t
                if denom != 0:
                    npv += annual_cf / denom

            if abs(npv) < Decimal("0.01"):
                return (mid * 100).quantize(Decimal("0.1"))
            if npv > 0:
                low = mid
            else:
                high = mid

        return ((low + high) / 2 * 100).quantize(Decimal("0.1"))

    @staticmethod
    def _score_opportunity_disclosure(
        opps: List[ClimateOpportunity],
        by_category: Dict[str, Dict[str, Any]],
    ) -> int:
        """
        Score opportunity disclosure completeness (0-100).

        Scoring criteria:
        - Opportunities exist: 15 pts
        - Multiple categories covered: up to 20 pts (4 pts each)
        - Financial quantification (revenue + savings): 15 pts
        - Investment analysis present: 10 pts
        - Time horizons covered: 15 pts (5 each)
        - ROI estimated: 10 pts
        - Status tracking active: 15 pts
        """
        if not opps:
            return 0

        score = 15  # Base for having opportunities

        # Category coverage (4 pts each, max 20)
        score += min(len(by_category) * 4, 20)

        # Financial quantification (15 pts)
        if all(o.revenue_potential_usd > 0 or o.cost_savings_usd > 0 for o in opps):
            score += 15
        elif any(o.revenue_potential_usd > 0 or o.cost_savings_usd > 0 for o in opps):
            score += 8

        # Investment analysis (10 pts)
        if all(o.investment_required_usd > 0 for o in opps):
            score += 10
        elif any(o.investment_required_usd > 0 for o in opps):
            score += 5

        # Time horizon coverage (5 pts each, max 15)
        timelines = set()
        for o in opps:
            if hasattr(o.timeline, "value"):
                timelines.add(o.timeline.value)
            else:
                timelines.add(str(o.timeline))
        score += min(len(timelines) * 5, 15)

        # ROI estimated (10 pts)
        if any(o.roi_estimate_pct > 0 for o in opps):
            score += 10

        # Status tracking (15 pts)
        statuses = set(o.status for o in opps)
        if len(statuses) >= 2:
            score += 15
        elif len(statuses) == 1 and "identified" not in statuses:
            score += 10
        else:
            score += 5

        return min(score, 100)
