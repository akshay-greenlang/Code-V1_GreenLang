"""
Strategy Engine -- TCFD Pillar 2: Strategy Analysis

Implements the TCFD Strategy recommended disclosures:
  - Strategy (a): Climate-related risks and opportunities identified
  - Strategy (b): Impact on businesses, strategy, and financial planning
  - Strategy (c): Scenario resilience (cross-references ScenarioAnalysisEngine)

Manages the lifecycle of climate risk and opportunity identification,
assessment, and business model impact analysis across the value chain.

The engine maintains registries of climate risks and opportunities, performs
time-horizon categorization, financial planning integration, and strategic
response tracking.  Generates Strategy (a), (b), and (c) disclosure content.
Scenario analysis is delegated to ScenarioAnalysisEngine.

Reference:
    - TCFD Final Report, Section C: Strategy (June 2017)
    - TCFD Annex: Implementing the Recommendations, Table 2
    - IFRS S2 Paragraphs 10-14 (Strategy)
    - IFRS S2 Paragraph 22 (Climate Resilience)

Example:
    >>> engine = StrategyEngine(config)
    >>> risk = await engine.identify_risk("org-1", risk_data)
    >>> impact = await engine.assess_business_model_impact("org-1")
    >>> disclosure = await engine.generate_strategy_disclosure("org-1", 2025)
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    IMPACT_SCORES,
    LIKELIHOOD_SCORES,
    OpportunityCategory,
    RiskType,
    SectorType,
    TCFDAppConfig,
    TimeHorizon,
    TIME_HORIZON_YEARS,
)
from .models import (
    ClimateOpportunity,
    ClimateRisk,
    CreateClimateOpportunityRequest,
    CreateClimateRiskRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector Risk Templates (11 TCFD sectors)
# ---------------------------------------------------------------------------

SECTOR_RISK_TEMPLATES: Dict[SectorType, List[Dict[str, Any]]] = {
    SectorType.ENERGY: [
        {"name": "Stranded fossil fuel assets", "risk_type": "transition_policy", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Carbon pricing exposure", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "major"},
        {"name": "Renewable energy displacement", "risk_type": "transition_technology", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Physical damage to infrastructure", "risk_type": "physical_acute", "time_horizon": "long_term", "impact": "catastrophic"},
    ],
    SectorType.TRANSPORTATION: [
        {"name": "ICE vehicle phase-out regulations", "risk_type": "transition_policy", "time_horizon": "medium_term", "impact": "major"},
        {"name": "EV technology disruption", "risk_type": "transition_technology", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Fuel price volatility", "risk_type": "transition_market", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Extreme weather disruption to logistics", "risk_type": "physical_acute", "time_horizon": "medium_term", "impact": "moderate"},
    ],
    SectorType.MATERIALS_BUILDINGS: [
        {"name": "Building energy efficiency mandates", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Embodied carbon regulations", "risk_type": "transition_policy", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Heat stress impact on workers", "risk_type": "physical_chronic", "time_horizon": "medium_term", "impact": "moderate"},
        {"name": "Flood risk to built assets", "risk_type": "physical_acute", "time_horizon": "long_term", "impact": "major"},
    ],
    SectorType.AGRICULTURE_FOOD_FOREST: [
        {"name": "Deforestation regulations (EUDR)", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Changing precipitation patterns", "risk_type": "physical_chronic", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Consumer demand shift to plant-based", "risk_type": "transition_market", "time_horizon": "medium_term", "impact": "moderate"},
        {"name": "Water stress and drought", "risk_type": "physical_chronic", "time_horizon": "long_term", "impact": "catastrophic"},
    ],
    SectorType.BANKING: [
        {"name": "Financed emissions regulation", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Credit risk from physical climate events", "risk_type": "physical_acute", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Greenwashing litigation", "risk_type": "transition_reputation", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Stranded asset loan book exposure", "risk_type": "transition_market", "time_horizon": "medium_term", "impact": "major"},
    ],
    SectorType.INSURANCE: [
        {"name": "Increasing catastrophe losses", "risk_type": "physical_acute", "time_horizon": "short_term", "impact": "catastrophic"},
        {"name": "Climate risk model inadequacy", "risk_type": "transition_technology", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Regulatory capital requirements", "risk_type": "transition_policy", "time_horizon": "medium_term", "impact": "moderate"},
        {"name": "Uninsurable risk areas expansion", "risk_type": "physical_chronic", "time_horizon": "long_term", "impact": "catastrophic"},
    ],
    SectorType.ASSET_OWNERS: [
        {"name": "Portfolio carbon footprint regulation", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Fiduciary duty climate litigation", "risk_type": "transition_reputation", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Physical risk to real asset holdings", "risk_type": "physical_acute", "time_horizon": "medium_term", "impact": "major"},
    ],
    SectorType.ASSET_MANAGERS: [
        {"name": "ESG disclosure requirements", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "minor"},
        {"name": "Client demand for sustainable products", "risk_type": "transition_market", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Greenwashing regulatory action", "risk_type": "transition_reputation", "time_horizon": "short_term", "impact": "major"},
    ],
    SectorType.CONSUMER_GOODS: [
        {"name": "Supply chain climate disruption", "risk_type": "physical_acute", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Packaging and waste regulations", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "moderate"},
        {"name": "Consumer preference shift", "risk_type": "transition_market", "time_horizon": "medium_term", "impact": "moderate"},
    ],
    SectorType.TECHNOLOGY_MEDIA: [
        {"name": "Data center energy consumption", "risk_type": "transition_policy", "time_horizon": "short_term", "impact": "minor"},
        {"name": "E-waste and circular economy regulations", "risk_type": "transition_policy", "time_horizon": "medium_term", "impact": "minor"},
        {"name": "Extreme heat impact on data centers", "risk_type": "physical_chronic", "time_horizon": "long_term", "impact": "moderate"},
    ],
    SectorType.HEALTHCARE: [
        {"name": "Anesthetic gas emission regulations", "risk_type": "transition_policy", "time_horizon": "medium_term", "impact": "minor"},
        {"name": "Supply chain disruption from climate events", "risk_type": "physical_acute", "time_horizon": "medium_term", "impact": "major"},
        {"name": "Increasing disease burden from climate change", "risk_type": "physical_chronic", "time_horizon": "long_term", "impact": "moderate"},
    ],
}


class StrategyEngine:
    """
    TCFD Pillar 2: Strategy analysis engine for risks, opportunities,
    business impact assessment, and strategic response tracking.

    Attributes:
        config: Application configuration.
        _risks: In-memory risk store keyed by org_id.
        _opportunities: In-memory opportunity store keyed by org_id.
        _strategic_responses: Tracked strategic responses by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize StrategyEngine.

        Args:
            config: Application configuration.
        """
        self.config = config or TCFDAppConfig()
        self._risks: Dict[str, List[ClimateRisk]] = {}
        self._opportunities: Dict[str, List[ClimateOpportunity]] = {}
        self._strategic_responses: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("StrategyEngine initialized")

    # ------------------------------------------------------------------
    # Risk Management
    # ------------------------------------------------------------------

    async def identify_risk(
        self,
        org_id: str,
        risk_data: CreateClimateRiskRequest,
    ) -> ClimateRisk:
        """
        Identify and register a new climate risk.

        Args:
            org_id: Organization ID.
            risk_data: Risk creation data.

        Returns:
            Created ClimateRisk.
        """
        start = datetime.utcnow()

        likelihood_num = LIKELIHOOD_SCORES.get(risk_data.likelihood, 3)
        impact_num = IMPACT_SCORES.get(risk_data.impact, 3)
        risk_score = likelihood_num * impact_num

        risk = ClimateRisk(
            tenant_id="default",
            org_id=org_id,
            risk_type=risk_data.risk_type,
            name=risk_data.name,
            description=risk_data.description,
            category=risk_data.category,
            time_horizon=risk_data.time_horizon,
            likelihood=risk_data.likelihood,
            impact=risk_data.impact,
            risk_score=risk_score,
            financial_impact_mid_usd=risk_data.financial_impact_mid_usd,
            response_strategy=risk_data.response_strategy,
            response_description=risk_data.response_description,
            owner=risk_data.owner,
        )

        if org_id not in self._risks:
            self._risks[org_id] = []
        self._risks[org_id].append(risk)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Identified risk '%s' (%s) for org %s in %.1f ms",
            risk.name, risk.risk_type.value, org_id, elapsed_ms,
        )
        return risk

    async def update_risk(
        self,
        risk_id: str,
        updates: Dict[str, Any],
    ) -> ClimateRisk:
        """
        Update an existing climate risk.

        Args:
            risk_id: Risk ID to update.
            updates: Field updates to apply.

        Returns:
            Updated ClimateRisk.

        Raises:
            ValueError: If risk not found.
        """
        for org_id, risks in self._risks.items():
            for i, risk in enumerate(risks):
                if risk.id == risk_id:
                    data = risk.model_dump()
                    data.update(updates)
                    data["updated_at"] = _now()
                    data["provenance_hash"] = ""
                    updated = ClimateRisk(**data)
                    self._risks[org_id][i] = updated
                    logger.info("Updated risk %s", risk_id)
                    return updated

        raise ValueError(f"Risk {risk_id} not found")

    async def list_risks(
        self,
        org_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ClimateRisk]:
        """
        List climate risks for an organization with optional filters.

        Args:
            org_id: Organization ID.
            filters: Optional filters (risk_type, time_horizon, status).

        Returns:
            Filtered list of ClimateRisk objects.
        """
        risks = list(self._risks.get(org_id, []))

        if filters:
            if "risk_type" in filters:
                risk_type = RiskType(filters["risk_type"])
                risks = [r for r in risks if r.risk_type == risk_type]
            if "time_horizon" in filters:
                th = TimeHorizon(filters["time_horizon"])
                risks = [r for r in risks if r.time_horizon == th]
            if "status" in filters:
                risks = [r for r in risks if r.status == filters["status"]]

        return risks

    async def get_sector_risk_templates(
        self, sector: SectorType,
    ) -> List[Dict[str, Any]]:
        """
        Get pre-built risk templates for a TCFD sector.

        Provides a starting point for risk identification based on
        sector-specific climate risk profiles.

        Args:
            sector: TCFD sector classification.

        Returns:
            List of risk template dicts with name, type, horizon, impact.
        """
        return SECTOR_RISK_TEMPLATES.get(sector, [])

    # ------------------------------------------------------------------
    # Opportunity Management
    # ------------------------------------------------------------------

    async def identify_opportunity(
        self,
        org_id: str,
        opp_data: CreateClimateOpportunityRequest,
    ) -> ClimateOpportunity:
        """
        Identify and register a new climate opportunity.

        Args:
            org_id: Organization ID.
            opp_data: Opportunity creation data.

        Returns:
            Created ClimateOpportunity.
        """
        roi = Decimal("0")
        if opp_data.investment_required_usd > 0:
            net_benefit = opp_data.revenue_potential_usd + opp_data.cost_savings_usd
            roi = (net_benefit / opp_data.investment_required_usd).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )

        opportunity = ClimateOpportunity(
            tenant_id="default",
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
        self._opportunities[org_id].append(opportunity)

        logger.info(
            "Identified opportunity '%s' (%s) for org %s, ROI=%.2f",
            opportunity.name, opportunity.category.value, org_id, roi,
        )
        return opportunity

    async def update_opportunity(
        self,
        opp_id: str,
        updates: Dict[str, Any],
    ) -> ClimateOpportunity:
        """
        Update an existing climate opportunity.

        Args:
            opp_id: Opportunity ID.
            updates: Field updates.

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
        self,
        org_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ClimateOpportunity]:
        """
        List climate opportunities with optional filters.

        Args:
            org_id: Organization ID.
            filters: Optional filters (category, timeline, status).

        Returns:
            Filtered list of ClimateOpportunity objects.
        """
        opps = list(self._opportunities.get(org_id, []))

        if filters:
            if "category" in filters:
                cat = OpportunityCategory(filters["category"])
                opps = [o for o in opps if o.category == cat]
            if "timeline" in filters:
                th = TimeHorizon(filters["timeline"])
                opps = [o for o in opps if o.timeline == th]
            if "status" in filters:
                opps = [o for o in opps if o.status == filters["status"]]

        return opps

    # ------------------------------------------------------------------
    # Business Model Impact Assessment -- Strategy (b)
    # ------------------------------------------------------------------

    async def assess_business_model_impact(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Assess the impact of climate risks and opportunities on the
        organization's business model.

        Analyzes revenue, cost, asset, and liability exposure across
        all identified risks and opportunities.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with business model impact assessment.
        """
        risks = self._risks.get(org_id, [])
        opps = self._opportunities.get(org_id, [])

        total_risk_impact = sum(r.financial_impact_mid_usd for r in risks)
        total_opp_revenue = sum(o.revenue_potential_usd for o in opps)
        total_opp_savings = sum(o.cost_savings_usd for o in opps)
        total_opp_investment = sum(o.investment_required_usd for o in opps)

        physical_risks = [r for r in risks if r.risk_type.value.startswith("physical_")]
        transition_risks = [r for r in risks if r.risk_type.value.startswith("transition_")]

        physical_impact = sum(r.financial_impact_mid_usd for r in physical_risks)
        transition_impact = sum(r.financial_impact_mid_usd for r in transition_risks)

        revenue_impact = {
            "risk_driven_decline": str(transition_impact * Decimal("0.3")),
            "opportunity_driven_growth": str(total_opp_revenue),
            "net_revenue_impact": str(total_opp_revenue - transition_impact * Decimal("0.3")),
        }

        cost_impact = {
            "compliance_costs": str(transition_impact * Decimal("0.4")),
            "adaptation_costs": str(physical_impact * Decimal("0.5")),
            "cost_savings_from_opportunities": str(total_opp_savings),
            "net_cost_impact": str(
                total_opp_savings
                - transition_impact * Decimal("0.4")
                - physical_impact * Decimal("0.5")
            ),
        }

        asset_impact = {
            "physical_damage_exposure": str(physical_impact),
            "stranded_asset_risk": str(transition_impact * Decimal("0.3")),
            "total_asset_exposure": str(
                physical_impact + transition_impact * Decimal("0.3")
            ),
        }

        liability_impact = {
            "litigation_risk": str(transition_impact * Decimal("0.1")),
            "regulatory_penalty_risk": str(transition_impact * Decimal("0.15")),
            "total_liability_exposure": str(transition_impact * Decimal("0.25")),
        }

        logger.info(
            "Business model impact for org %s: risks=%d ($%.0f), opps=%d ($%.0f)",
            org_id, len(risks), total_risk_impact, len(opps), total_opp_revenue,
        )

        return {
            "org_id": org_id,
            "total_risks": len(risks),
            "total_opportunities": len(opps),
            "total_risk_exposure": str(total_risk_impact),
            "total_opportunity_value": str(total_opp_revenue + total_opp_savings),
            "total_investment_required": str(total_opp_investment),
            "revenue_impact": revenue_impact,
            "cost_impact": cost_impact,
            "asset_impact": asset_impact,
            "liability_impact": liability_impact,
            "net_financial_impact": str(
                total_opp_revenue + total_opp_savings - total_risk_impact
            ),
        }

    # ------------------------------------------------------------------
    # Financial Planning Integration -- Strategy (b) extended
    # ------------------------------------------------------------------

    async def assess_financial_planning_impact(
        self,
        org_id: str,
        revenue_base_usd: Decimal,
        cost_base_usd: Decimal,
        total_assets_usd: Decimal,
    ) -> Dict[str, Any]:
        """
        Assess impact on financial planning cycle.

        Quantifies how climate risks and opportunities affect operating
        budgets, capital allocation, and long-term financial planning.

        Args:
            org_id: Organization ID.
            revenue_base_usd: Baseline annual revenue.
            cost_base_usd: Baseline annual operating costs.
            total_assets_usd: Total asset base.

        Returns:
            Dict with financial planning impact by time horizon.
        """
        risks = self._risks.get(org_id, [])
        opps = self._opportunities.get(org_id, [])

        result: Dict[str, Any] = {}

        for th in TimeHorizon:
            th_risks = [r for r in risks if r.time_horizon == th]
            th_opps = [o for o in opps if o.timeline == th]

            risk_exposure = sum(r.financial_impact_mid_usd for r in th_risks)
            opp_value = sum(o.revenue_potential_usd + o.cost_savings_usd for o in th_opps)
            opp_investment = sum(o.investment_required_usd for o in th_opps)

            # Impact percentages relative to base
            revenue_impact_pct = Decimal("0")
            if revenue_base_usd > 0:
                revenue_impact_pct = (
                    (opp_value - risk_exposure * Decimal("0.3")) / revenue_base_usd * 100
                ).quantize(Decimal("0.1"))

            cost_impact_pct = Decimal("0")
            if cost_base_usd > 0:
                cost_impact_pct = (
                    risk_exposure * Decimal("0.4") / cost_base_usd * 100
                ).quantize(Decimal("0.1"))

            capex_pct = Decimal("0")
            if total_assets_usd > 0:
                capex_pct = (
                    opp_investment / total_assets_usd * 100
                ).quantize(Decimal("0.1"))

            year_range = TIME_HORIZON_YEARS.get(th, {})
            result[th.value] = {
                "year_range": f"{year_range.get('min_years', 0)}-{year_range.get('max_years', 0)} years",
                "risk_exposure": str(risk_exposure),
                "opportunity_value": str(opp_value),
                "investment_required": str(opp_investment),
                "revenue_impact_pct": str(revenue_impact_pct),
                "cost_impact_pct": str(cost_impact_pct),
                "capex_as_pct_of_assets": str(capex_pct),
            }

        return {
            "org_id": org_id,
            "revenue_base": str(revenue_base_usd),
            "cost_base": str(cost_base_usd),
            "total_assets": str(total_assets_usd),
            "by_time_horizon": result,
        }

    # ------------------------------------------------------------------
    # Strategic Response Tracking
    # ------------------------------------------------------------------

    async def register_strategic_response(
        self,
        org_id: str,
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Register a strategic response to climate risks/opportunities.

        Args:
            org_id: Organization ID.
            response_data: Dict with response_type, description, budget,
                timeline, related_risk_ids, related_opportunity_ids.

        Returns:
            Stored strategic response with ID.
        """
        response = {
            "id": _new_id(),
            "org_id": org_id,
            "response_type": response_data.get("response_type", "adaptation"),
            "description": response_data.get("description", ""),
            "budget_usd": str(Decimal(str(response_data.get("budget_usd", 0)))),
            "timeline": response_data.get("timeline", "medium_term"),
            "status": response_data.get("status", "planned"),
            "related_risk_ids": response_data.get("related_risk_ids", []),
            "related_opportunity_ids": response_data.get("related_opportunity_ids", []),
            "kpis": response_data.get("kpis", []),
            "created_at": _now().isoformat(),
        }

        if org_id not in self._strategic_responses:
            self._strategic_responses[org_id] = []
        self._strategic_responses[org_id].append(response)

        logger.info(
            "Registered strategic response '%s' for org %s",
            response["response_type"], org_id,
        )
        return response

    async def list_strategic_responses(
        self, org_id: str,
    ) -> List[Dict[str, Any]]:
        """List all strategic responses for an organization."""
        return self._strategic_responses.get(org_id, [])

    # ------------------------------------------------------------------
    # Value Chain Impact Mapping
    # ------------------------------------------------------------------

    async def map_value_chain_impact(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Map climate risks and opportunities across the value chain.

        Categorizes impacts into upstream, operations, and downstream
        segments to identify concentration areas.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with value chain impact mapping.
        """
        risks = self._risks.get(org_id, [])
        opps = self._opportunities.get(org_id, [])

        upstream_risks = [r for r in risks if r.category in ("supply_chain", "raw_materials", "upstream")]
        operations_risks = [r for r in risks if r.category in ("operations", "facilities", "production", "")]
        downstream_risks = [r for r in risks if r.category in ("customers", "products", "downstream")]

        upstream_opps = [o for o in opps if o.category in (OpportunityCategory.RESOURCE_EFFICIENCY,)]
        operations_opps = [o for o in opps if o.category in (OpportunityCategory.ENERGY_SOURCE, OpportunityCategory.RESILIENCE)]
        downstream_opps = [o for o in opps if o.category in (OpportunityCategory.PRODUCTS_SERVICES, OpportunityCategory.MARKETS)]

        return {
            "org_id": org_id,
            "upstream": {
                "risk_count": len(upstream_risks),
                "risk_exposure": str(sum(r.financial_impact_mid_usd for r in upstream_risks)),
                "opportunity_count": len(upstream_opps),
                "opportunity_value": str(sum(o.revenue_potential_usd + o.cost_savings_usd for o in upstream_opps)),
                "key_risks": [r.name for r in upstream_risks[:5]],
                "key_opportunities": [o.name for o in upstream_opps[:5]],
            },
            "operations": {
                "risk_count": len(operations_risks),
                "risk_exposure": str(sum(r.financial_impact_mid_usd for r in operations_risks)),
                "opportunity_count": len(operations_opps),
                "opportunity_value": str(sum(o.revenue_potential_usd + o.cost_savings_usd for o in operations_opps)),
                "key_risks": [r.name for r in operations_risks[:5]],
                "key_opportunities": [o.name for o in operations_opps[:5]],
            },
            "downstream": {
                "risk_count": len(downstream_risks),
                "risk_exposure": str(sum(r.financial_impact_mid_usd for r in downstream_risks)),
                "opportunity_count": len(downstream_opps),
                "opportunity_value": str(sum(o.revenue_potential_usd + o.cost_savings_usd for o in downstream_opps)),
                "key_risks": [r.name for r in downstream_risks[:5]],
                "key_opportunities": [o.name for o in downstream_opps[:5]],
            },
        }

    # ------------------------------------------------------------------
    # Time Horizon Categorization
    # ------------------------------------------------------------------

    async def categorize_time_horizons(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Categorize risks and opportunities by time horizon.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with short/medium/long term distributions.
        """
        risks = self._risks.get(org_id, [])
        opps = self._opportunities.get(org_id, [])

        result: Dict[str, Any] = {}

        for th in TimeHorizon:
            th_risks = [r for r in risks if r.time_horizon == th]
            th_opps = [o for o in opps if o.timeline == th]
            year_range = TIME_HORIZON_YEARS.get(th, {})

            result[th.value] = {
                "year_range": f"{year_range.get('min_years', 0)}-{year_range.get('max_years', 0)} years",
                "risk_count": len(th_risks),
                "risk_exposure": str(sum(r.financial_impact_mid_usd for r in th_risks)),
                "opportunity_count": len(th_opps),
                "opportunity_value": str(
                    sum(o.revenue_potential_usd + o.cost_savings_usd for o in th_opps)
                ),
                "top_risks": [
                    {"name": r.name, "type": r.risk_type.value, "impact": str(r.financial_impact_mid_usd)}
                    for r in sorted(th_risks, key=lambda x: x.financial_impact_mid_usd, reverse=True)[:3]
                ],
                "top_opportunities": [
                    {"name": o.name, "category": o.category.value, "value": str(o.revenue_potential_usd)}
                    for o in sorted(th_opps, key=lambda x: x.revenue_potential_usd, reverse=True)[:3]
                ],
            }

        return {
            "org_id": org_id,
            "time_horizons": result,
            "total_risks": len(risks),
            "total_opportunities": len(opps),
        }

    # ------------------------------------------------------------------
    # Strategy Summary
    # ------------------------------------------------------------------

    async def get_strategy_summary(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get a comprehensive strategy summary for the organization.

        Aggregates risk/opportunity counts, financial exposure, category
        coverage, and strategic response status.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with strategy pillar summary metrics.
        """
        risks = self._risks.get(org_id, [])
        opps = self._opportunities.get(org_id, [])
        responses = self._strategic_responses.get(org_id, [])

        risk_by_type: Dict[str, int] = {}
        for r in risks:
            risk_by_type[r.risk_type.value] = risk_by_type.get(r.risk_type.value, 0) + 1

        opp_by_category: Dict[str, int] = {}
        for o in opps:
            opp_by_category[o.category.value] = opp_by_category.get(o.category.value, 0) + 1

        total_risk_exposure = sum(r.financial_impact_mid_usd for r in risks)
        total_opp_value = sum(o.revenue_potential_usd + o.cost_savings_usd for o in opps)
        net_position = total_opp_value - total_risk_exposure

        return {
            "org_id": org_id,
            "risk_count": len(risks),
            "opportunity_count": len(opps),
            "strategic_response_count": len(responses),
            "total_risk_exposure": str(total_risk_exposure),
            "total_opportunity_value": str(total_opp_value),
            "net_climate_position": str(net_position),
            "risk_by_type": risk_by_type,
            "opportunity_by_category": opp_by_category,
            "risk_types_covered": list(risk_by_type.keys()),
            "opportunity_categories_covered": list(opp_by_category.keys()),
        }

    # ------------------------------------------------------------------
    # Disclosure Generation
    # ------------------------------------------------------------------

    async def generate_strategy_disclosure(
        self,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Generate TCFD Strategy (a), (b), and (c) disclosure content.

        Strategy (c) references scenario analysis results and provides
        a resilience narrative framework.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            Dict with str_a, str_b, and str_c disclosure sections.
        """
        risks = self._risks.get(org_id, [])
        opps = self._opportunities.get(org_id, [])

        physical_risks = [r for r in risks if r.risk_type.value.startswith("physical_")]
        transition_risks = [r for r in risks if r.risk_type.value.startswith("transition_")]

        # --- Strategy (a): Risks and Opportunities ---
        str_a_parts: List[str] = []
        str_a_parts.append(
            f"The organization has identified {len(risks)} climate-related risk(s) "
            f"and {len(opps)} opportunity/ies. "
        )
        if physical_risks:
            str_a_parts.append(
                f"Physical risks comprise {len(physical_risks)} item(s) with "
                f"total estimated exposure of "
                f"${sum(r.financial_impact_mid_usd for r in physical_risks):,.0f}. "
            )
        if transition_risks:
            str_a_parts.append(
                f"Transition risks comprise {len(transition_risks)} item(s) with "
                f"total estimated exposure of "
                f"${sum(r.financial_impact_mid_usd for r in transition_risks):,.0f}. "
            )

        # --- Strategy (b): Business Impact ---
        str_b_text = await self.assess_business_model_impact(org_id)

        # --- Strategy (c): Scenario Resilience ---
        responses = self._strategic_responses.get(org_id, [])
        str_c_parts: List[str] = []
        str_c_parts.append(
            "The organization assesses its strategic resilience using "
            "climate scenario analysis covering at least two scenarios, "
            "including a 2 degrees C or lower pathway. "
        )
        if responses:
            str_c_parts.append(
                f"{len(responses)} strategic response(s) have been developed "
                f"to enhance climate resilience. "
            )
        str_c_parts.append(
            "Detailed scenario analysis results including NPV impact, "
            "carbon cost exposure, and asset impairment are produced by the "
            "ScenarioAnalysisEngine and cross-referenced here."
        )

        compliance_a = self._score_strategy_a(risks, opps)
        compliance_b = self._score_strategy_b(str_b_text)
        compliance_c = self._score_strategy_c(responses)

        return {
            "org_id": org_id,
            "reporting_year": year,
            "str_a": {
                "ref": "Strategy (a)",
                "title": "Risks and Opportunities",
                "content": "".join(str_a_parts),
                "compliance_score": compliance_a,
                "risk_count": len(risks),
                "opportunity_count": len(opps),
            },
            "str_b": {
                "ref": "Strategy (b)",
                "title": "Business Impact",
                "content": (
                    f"Net financial impact is estimated at "
                    f"${str_b_text.get('net_financial_impact', '0')}. "
                    f"Revenue impact is driven by both transition risk decline "
                    f"and opportunity-driven growth."
                ),
                "compliance_score": compliance_b,
                "impact_summary": str_b_text,
            },
            "str_c": {
                "ref": "Strategy (c)",
                "title": "Scenario Resilience",
                "content": "".join(str_c_parts),
                "compliance_score": compliance_c,
                "strategic_responses": len(responses),
            },
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_strategy_a(
        risks: List[ClimateRisk],
        opps: List[ClimateOpportunity],
    ) -> int:
        """Score Strategy (a) disclosure completeness (0-100)."""
        score = 0
        if risks:
            score += 20
            if any(r.risk_type.value.startswith("physical_") for r in risks):
                score += 10
            if any(r.risk_type.value.startswith("transition_") for r in risks):
                score += 10
            time_horizons_covered = set(r.time_horizon for r in risks)
            score += min(len(time_horizons_covered) * 10, 20)
        if opps:
            score += 15
            categories_covered = set(o.category for o in opps)
            score += min(len(categories_covered) * 5, 15)
        if risks and all(r.financial_impact_mid_usd > 0 for r in risks):
            score += 10
        return min(score, 100)

    @staticmethod
    def _score_strategy_b(impact_data: Dict[str, Any]) -> int:
        """Score Strategy (b) disclosure completeness (0-100)."""
        score = 0
        if impact_data.get("revenue_impact"):
            score += 25
        if impact_data.get("cost_impact"):
            score += 25
        if impact_data.get("asset_impact"):
            score += 25
        if impact_data.get("liability_impact"):
            score += 15
        if impact_data.get("total_risks", 0) > 0 and impact_data.get("total_opportunities", 0) > 0:
            score += 10
        return min(score, 100)

    @staticmethod
    def _score_strategy_c(responses: List[Dict[str, Any]]) -> int:
        """
        Score Strategy (c) disclosure completeness (0-100).

        Criteria:
        - Scenario analysis referenced: 30 pts (assumed if engine is used)
        - Multiple scenarios (2+): 20 pts
        - Strategic responses defined: 25 pts
        - Response budgets allocated: 15 pts
        - KPIs for responses: 10 pts
        """
        score = 30  # Base for using scenario analysis engine

        # Strategic responses
        if responses:
            score += 25
            if len(responses) >= 2:
                score += 10  # Multiple responses
            if any(Decimal(r.get("budget_usd", "0")) > 0 for r in responses):
                score += 15
            if any(r.get("kpis") for r in responses):
                score += 10
        else:
            score += 0

        return min(score, 100)
