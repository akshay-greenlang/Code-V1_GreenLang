"""
Transition Risk Engine -- TCFD Transition Risk Assessment

Implements transition risk assessment across the four TCFD transition risk
categories: policy and legal, technology, market, and reputation.

Provides:
  - Individual assessment for each transition risk category
  - Carbon cost exposure calculation with trajectory interpolation
  - Asset stranding probability and timeline assessment
  - Regulatory compliance cost estimation with breakdown
  - Litigation risk assessment
  - Composite transition risk scoring with weighting
  - Sector-specific transition profiles (11 TCFD sectors)
  - Regulation timeline tracking
  - Multi-scenario transition risk comparison
  - Technology disruption probability modeling
  - Market demand shift analysis
  - Reputation scoring framework
  - Stranded asset timeline projection
  - Transition risk disclosure generation

Reference:
    - TCFD Final Report, Appendix 1: Transition Risks
    - IFRS S2 Paragraphs 10-12 (Strategy: Risks)
    - IEA WEO 2023 (carbon price trajectories)
    - NGFS Phase IV (sector-specific transition profiles)

Example:
    >>> engine = TransitionRiskEngine(config)
    >>> result = await engine.assess_policy_risk("org-1", "tenant-1", policy_data)
    >>> composite = await engine.calculate_composite_transition_risk("org-1")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    RiskType,
    SCENARIO_CARBON_PRICES,
    SCENARIO_LIBRARY,
    SECTOR_TRANSITION_PROFILES,
    SectorType,
    ScenarioType,
    TCFDAppConfig,
    TimeHorizon,
    TransitionRiskSubType,
)
from .models import (
    PolicyRisk,
    TechnologyRisk,
    MarketRisk,
    ReputationRisk,
    StrandedAssetAnalysis,
    TransitionRiskAssessment,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carbon Price Regulation Reference Database
# ---------------------------------------------------------------------------

CARBON_PRICE_REGULATIONS: List[Dict[str, Any]] = [
    {
        "regulation": "EU ETS Phase IV",
        "jurisdiction": "EU",
        "effective": "2021-01-01",
        "impact": "very_high",
        "current_price_usd": 60,
        "projected_2030_usd": 130,
        "covered_emissions_pct": 40,
    },
    {
        "regulation": "EU CBAM",
        "jurisdiction": "EU",
        "effective": "2026-01-01",
        "impact": "high",
        "current_price_usd": 0,
        "projected_2030_usd": 130,
        "covered_emissions_pct": 15,
    },
    {
        "regulation": "UK ETS",
        "jurisdiction": "UK",
        "effective": "2021-01-01",
        "impact": "high",
        "current_price_usd": 55,
        "projected_2030_usd": 120,
        "covered_emissions_pct": 30,
    },
    {
        "regulation": "California Cap-and-Trade",
        "jurisdiction": "US-CA",
        "effective": "2013-01-01",
        "impact": "high",
        "current_price_usd": 35,
        "projected_2030_usd": 80,
        "covered_emissions_pct": 85,
    },
    {
        "regulation": "Canada Federal Carbon Price",
        "jurisdiction": "CA",
        "effective": "2019-04-01",
        "impact": "high",
        "current_price_usd": 50,
        "projected_2030_usd": 170,
        "covered_emissions_pct": 70,
    },
    {
        "regulation": "Japan GX ETS",
        "jurisdiction": "JP",
        "effective": "2026-04-01",
        "impact": "medium",
        "current_price_usd": 0,
        "projected_2030_usd": 50,
        "covered_emissions_pct": 45,
    },
    {
        "regulation": "SEC Climate Disclosure Rule",
        "jurisdiction": "US",
        "effective": "2026-01-01",
        "impact": "high",
        "current_price_usd": 0,
        "projected_2030_usd": 0,
        "covered_emissions_pct": 0,
    },
    {
        "regulation": "ISSB/IFRS S2",
        "jurisdiction": "Global",
        "effective": "2025-01-01",
        "impact": "high",
        "current_price_usd": 0,
        "projected_2030_usd": 0,
        "covered_emissions_pct": 0,
    },
    {
        "regulation": "UK Transition Plans",
        "jurisdiction": "UK",
        "effective": "2025-10-01",
        "impact": "medium",
        "current_price_usd": 0,
        "projected_2030_usd": 0,
        "covered_emissions_pct": 0,
    },
    {
        "regulation": "CSRD",
        "jurisdiction": "EU",
        "effective": "2025-01-01",
        "impact": "high",
        "current_price_usd": 0,
        "projected_2030_usd": 0,
        "covered_emissions_pct": 0,
    },
]

# ---------------------------------------------------------------------------
# Technology Disruption Probability by Sector
# ---------------------------------------------------------------------------

_TECH_DISRUPTION_PROBABILITY: Dict[SectorType, Dict[str, float]] = {
    SectorType.ENERGY: {
        "renewables_displacement": 0.85,
        "ccs_adoption": 0.40,
        "hydrogen_scale": 0.35,
        "battery_storage": 0.75,
    },
    SectorType.TRANSPORTATION: {
        "ev_adoption": 0.80,
        "sustainable_aviation_fuel": 0.30,
        "autonomous_logistics": 0.25,
        "hydrogen_heavy_transport": 0.20,
    },
    SectorType.MATERIALS_BUILDINGS: {
        "green_steel": 0.35,
        "green_cement": 0.25,
        "circular_economy": 0.50,
        "heat_pump_adoption": 0.70,
    },
    SectorType.AGRICULTURE_FOOD_FOREST: {
        "precision_agriculture": 0.55,
        "alternative_proteins": 0.40,
        "methane_capture": 0.30,
        "vertical_farming": 0.20,
    },
    SectorType.BANKING: {
        "green_fintech": 0.45,
        "climate_analytics": 0.60,
        "carbon_markets": 0.50,
        "digital_mrvs": 0.40,
    },
    SectorType.INSURANCE: {
        "parametric_insurance": 0.55,
        "catastrophe_modeling": 0.70,
        "climate_risk_pricing": 0.65,
        "resilience_bonds": 0.25,
    },
}

# ---------------------------------------------------------------------------
# Transition Risk Weighting by Category
# ---------------------------------------------------------------------------

_RISK_CATEGORY_WEIGHTS: Dict[RiskType, Decimal] = {
    RiskType.TRANSITION_POLICY: Decimal("0.35"),
    RiskType.TRANSITION_TECHNOLOGY: Decimal("0.25"),
    RiskType.TRANSITION_MARKET: Decimal("0.25"),
    RiskType.TRANSITION_REPUTATION: Decimal("0.15"),
}

# ---------------------------------------------------------------------------
# Sector Exposure Multipliers
# ---------------------------------------------------------------------------

_SECTOR_EXPOSURE_MULTIPLIERS: Dict[str, Decimal] = {
    "very_high": Decimal("1.5"),
    "high": Decimal("1.2"),
    "medium": Decimal("1.0"),
    "low": Decimal("0.7"),
    "unknown": Decimal("1.0"),
}


class TransitionRiskEngine:
    """
    Transition risk assessment across 4 TCFD categories.

    Covers policy/legal, technology, market, and reputation transition risks
    with multi-scenario comparison, carbon cost modeling, stranded asset
    analysis, and TCFD-aligned disclosure generation.

    Attributes:
        config: Application configuration.
        _assessments: In-memory store keyed by org_id.
        _regulations: In-memory regulation timeline store.
        _stranded_analyses: In-memory stranded asset analyses.
        _policy_risks: Detailed policy risk records by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        self.config = config or TCFDAppConfig()
        self._assessments: Dict[str, List[TransitionRiskAssessment]] = {}
        self._regulations: Dict[str, List[Dict[str, Any]]] = {}
        self._stranded_analyses: Dict[str, List[StrandedAssetAnalysis]] = {}
        self._policy_risks: Dict[str, List[PolicyRisk]] = {}
        logger.info("TransitionRiskEngine initialized")

    # ------------------------------------------------------------------
    # Category-Level Assessment Methods
    # ------------------------------------------------------------------

    async def assess_policy_risk(
        self, org_id: str, tenant_id: str,
        policy_data: Dict[str, Any],
    ) -> TransitionRiskAssessment:
        """
        Assess policy and legal transition risk.

        Calculates carbon cost exposure across current and projected carbon
        prices for Scope 1 and Scope 2 emissions, applying sector-specific
        decarbonization pathways for 2030 and 2050 projections.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            policy_data: Dict with emissions_scope1, emissions_scope2,
                carbon_price_2030, carbon_price_2050, sector, current_carbon_price,
                mitigation_actions, and optional jurisdiction details.

        Returns:
            TransitionRiskAssessment with policy risk quantification.
        """
        start = datetime.utcnow()
        emissions_scope1 = Decimal(str(policy_data.get("emissions_scope1", 1000)))
        emissions_scope2 = Decimal(str(policy_data.get("emissions_scope2", 500)))
        carbon_price = Decimal(str(policy_data.get("carbon_price_2030", 130)))
        sector = SectorType(policy_data.get("sector", "energy"))

        current_carbon_price = Decimal(str(policy_data.get("current_carbon_price", 25)))
        current_cost = (emissions_scope1 + emissions_scope2) * current_carbon_price
        projected_2030 = (emissions_scope1 + emissions_scope2) * carbon_price

        # Apply decarbonization pathway reduction for 2050 projection
        carbon_price_2050 = Decimal(str(policy_data.get("carbon_price_2050", 250)))
        projected_2050 = (
            emissions_scope1 * Decimal("0.3") + emissions_scope2 * Decimal("0.2")
        ) * carbon_price_2050

        # Build detailed policy risk records for audit trail
        jurisdiction = policy_data.get("jurisdiction", "Global")
        policy_risk_detail = PolicyRisk(
            tenant_id=tenant_id,
            org_id=org_id,
            policy_name=f"Carbon pricing - {jurisdiction}",
            jurisdiction=jurisdiction,
            compliance_cost_usd=projected_2030 * Decimal("0.05"),
            penalty_risk_usd=projected_2030 * Decimal("0.02"),
            carbon_price_exposure_usd=projected_2030 - current_cost,
            description=f"Carbon price exposure for {sector.value} sector in {jurisdiction}",
        )

        # Apply sector exposure multiplier
        profile = SECTOR_TRANSITION_PROFILES.get(sector, {})
        exposure_level = profile.get("transition_exposure", "medium")
        multiplier = _SECTOR_EXPOSURE_MULTIPLIERS.get(exposure_level, Decimal("1.0"))
        financial_impact = (projected_2030 - current_cost) * multiplier

        assessment = TransitionRiskAssessment(
            tenant_id=tenant_id,
            org_id=org_id,
            risk_type=RiskType.TRANSITION_POLICY,
            sub_type=TransitionRiskSubType.CARBON_PRICING,
            sector=sector,
            current_exposure_usd=current_cost,
            projected_exposure_2030_usd=projected_2030,
            projected_exposure_2050_usd=projected_2050,
            financial_impact_usd=financial_impact,
            policy_risks=[policy_risk_detail],
            mitigation_actions=policy_data.get("mitigation_actions", []),
        )

        self._store_assessment(org_id, assessment)
        self._store_policy_risk(org_id, policy_risk_detail)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Policy risk for org %s: impact=$%.0f in %.1f ms",
            org_id, assessment.financial_impact_usd, elapsed_ms,
        )
        return assessment

    async def assess_technology_risk(
        self, org_id: str, tenant_id: str,
        tech_data: Dict[str, Any],
    ) -> TransitionRiskAssessment:
        """
        Assess technology transition risk.

        Evaluates disruption probability, write-off risk, and replacement
        cost for technology assets under transition scenarios.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            tech_data: Dict with current_technology_value, disruption_probability,
                replacement_cost, sector, technology_area, and mitigation_actions.

        Returns:
            TransitionRiskAssessment with technology risk quantification.
        """
        start = datetime.utcnow()
        sector = SectorType(tech_data.get("sector", "energy"))
        current_tech_value = Decimal(str(tech_data.get("current_technology_value", 5000000)))
        disruption_probability = Decimal(str(tech_data.get("disruption_probability", 30))) / 100
        replacement_cost = Decimal(str(tech_data.get("replacement_cost", 2000000)))
        technology_area = tech_data.get("technology_area", "legacy_systems")

        # Expected loss = value * probability + replacement cost
        write_off_risk = current_tech_value * disruption_probability
        financial_impact = write_off_risk + replacement_cost

        # Build detailed technology risk record
        tech_risk_detail = TechnologyRisk(
            tenant_id=tenant_id,
            org_id=org_id,
            technology_area=technology_area,
            disruption_timeline=TimeHorizon(tech_data.get("disruption_timeline", "medium_term")),
            current_technology_value_usd=current_tech_value,
            write_off_risk_usd=write_off_risk,
            replacement_cost_usd=replacement_cost,
            description=tech_data.get("description", f"Technology risk: {technology_area}"),
        )

        # Residual value decay: 2030 = 1-p, 2050 = 20% remaining
        projected_2030 = current_tech_value * (Decimal("1") - disruption_probability)
        projected_2050 = current_tech_value * Decimal("0.2")

        assessment = TransitionRiskAssessment(
            tenant_id=tenant_id,
            org_id=org_id,
            risk_type=RiskType.TRANSITION_TECHNOLOGY,
            sub_type=TransitionRiskSubType.TECHNOLOGY_SUBSTITUTION,
            sector=sector,
            current_exposure_usd=current_tech_value,
            projected_exposure_2030_usd=projected_2030,
            projected_exposure_2050_usd=projected_2050,
            financial_impact_usd=financial_impact,
            technology_risks=[tech_risk_detail],
            mitigation_actions=tech_data.get("mitigation_actions", []),
        )
        self._store_assessment(org_id, assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Technology risk for org %s (%s): impact=$%.0f in %.1f ms",
            org_id, technology_area, financial_impact, elapsed_ms,
        )
        return assessment

    async def assess_market_risk(
        self, org_id: str, tenant_id: str,
        market_data: Dict[str, Any],
    ) -> TransitionRiskAssessment:
        """
        Assess market transition risk.

        Models revenue-at-risk from demand shifts, commodity price changes,
        and supply chain disruptions under transition scenarios.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            market_data: Dict with revenue_at_risk, demand_shift_pct, sector,
                market_segment, commodity_price_impact, supply_chain_cost.

        Returns:
            TransitionRiskAssessment with market risk quantification.
        """
        start = datetime.utcnow()
        sector = SectorType(market_data.get("sector", "energy"))
        revenue_at_risk = Decimal(str(market_data.get("revenue_at_risk", 1000000)))
        demand_shift_pct = Decimal(str(market_data.get("demand_shift_pct", 15))) / 100
        market_segment = market_data.get("market_segment", "core_products")
        commodity_impact = Decimal(str(market_data.get("commodity_price_impact", 0)))
        supply_chain_cost = Decimal(str(market_data.get("supply_chain_cost", 0)))

        demand_impact = revenue_at_risk * demand_shift_pct
        financial_impact = demand_impact + commodity_impact + supply_chain_cost

        # Build detailed market risk record
        market_risk_detail = MarketRisk(
            tenant_id=tenant_id,
            org_id=org_id,
            market_segment=market_segment,
            demand_shift_pct=demand_shift_pct * 100,
            revenue_at_risk_usd=revenue_at_risk,
            commodity_price_impact_usd=commodity_impact,
            supply_chain_disruption_cost_usd=supply_chain_cost,
            description=market_data.get("description", f"Market risk: {market_segment}"),
        )

        # Progressive demand erosion over time
        projected_2030 = revenue_at_risk * (Decimal("1") - demand_shift_pct)
        projected_2050 = revenue_at_risk * (Decimal("1") - demand_shift_pct * 2)
        # Floor at zero -- revenue cannot go negative
        projected_2050 = max(projected_2050, Decimal("0"))

        assessment = TransitionRiskAssessment(
            tenant_id=tenant_id,
            org_id=org_id,
            risk_type=RiskType.TRANSITION_MARKET,
            sub_type=TransitionRiskSubType.DEMAND_SHIFT,
            sector=sector,
            current_exposure_usd=revenue_at_risk,
            projected_exposure_2030_usd=projected_2030,
            projected_exposure_2050_usd=projected_2050,
            financial_impact_usd=financial_impact,
            market_risks=[market_risk_detail],
            mitigation_actions=market_data.get("mitigation_actions", []),
        )
        self._store_assessment(org_id, assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Market risk for org %s (%s): impact=$%.0f in %.1f ms",
            org_id, market_segment, financial_impact, elapsed_ms,
        )
        return assessment

    async def assess_reputation_risk(
        self, org_id: str, tenant_id: str,
        rep_data: Dict[str, Any],
    ) -> TransitionRiskAssessment:
        """
        Assess reputational transition risk.

        Quantifies brand value erosion, customer attrition, and talent
        retention risks from climate-related reputational factors.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            rep_data: Dict with brand_value, sentiment_risk_pct, sector,
                risk_driver, customer_attrition_pct, talent_attrition_pct.

        Returns:
            TransitionRiskAssessment with reputation risk quantification.
        """
        start = datetime.utcnow()
        sector = SectorType(rep_data.get("sector", "energy"))
        brand_value = Decimal(str(rep_data.get("brand_value", 10000000)))
        sentiment_risk_pct = Decimal(str(rep_data.get("sentiment_risk_pct", 5))) / 100
        risk_driver = rep_data.get("risk_driver", "stakeholder_sentiment")
        customer_attrition_pct = Decimal(str(rep_data.get("customer_attrition_pct", 3))) / 100
        talent_attrition_pct = Decimal(str(rep_data.get("talent_attrition_pct", 2))) / 100
        revenue_base = Decimal(str(rep_data.get("revenue_base", 50000000)))

        brand_impact = brand_value * sentiment_risk_pct
        customer_impact = revenue_base * customer_attrition_pct
        talent_cost = revenue_base * Decimal("0.01") * (talent_attrition_pct * 100)
        financial_impact = brand_impact + customer_impact + talent_cost

        # Build detailed reputation risk record
        rep_risk_detail = ReputationRisk(
            tenant_id=tenant_id,
            org_id=org_id,
            risk_driver=risk_driver,
            brand_value_at_risk_pct=sentiment_risk_pct * 100,
            customer_attrition_risk_pct=customer_attrition_pct * 100,
            talent_attrition_risk_pct=talent_attrition_pct * 100,
            estimated_revenue_impact_usd=customer_impact,
            description=rep_data.get("description", f"Reputation risk: {risk_driver}"),
        )

        projected_2030 = brand_value * (Decimal("1") - sentiment_risk_pct)
        projected_2050 = brand_value * (Decimal("1") - sentiment_risk_pct * 2)
        projected_2050 = max(projected_2050, Decimal("0"))

        assessment = TransitionRiskAssessment(
            tenant_id=tenant_id,
            org_id=org_id,
            risk_type=RiskType.TRANSITION_REPUTATION,
            sub_type=TransitionRiskSubType.STAKEHOLDER_SENTIMENT,
            sector=sector,
            current_exposure_usd=brand_value,
            projected_exposure_2030_usd=projected_2030,
            projected_exposure_2050_usd=projected_2050,
            financial_impact_usd=financial_impact,
            reputation_risks=[rep_risk_detail],
            mitigation_actions=rep_data.get("mitigation_actions", []),
        )
        self._store_assessment(org_id, assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Reputation risk for org %s (%s): impact=$%.0f in %.1f ms",
            org_id, risk_driver, financial_impact, elapsed_ms,
        )
        return assessment

    # ------------------------------------------------------------------
    # Carbon Cost Exposure (Trajectory-Based)
    # ------------------------------------------------------------------

    async def calculate_carbon_cost_exposure(
        self, org_id: str,
        carbon_price_trajectory: Dict[int, Decimal],
    ) -> Dict[str, Any]:
        """
        Calculate carbon cost exposure over time with trajectory interpolation.

        Interpolates carbon prices between provided years and projects total
        cost based on current policy risk assessments.

        Args:
            org_id: Organization identifier.
            carbon_price_trajectory: Year-to-price mapping (e.g. {2025: 75, 2030: 130}).

        Returns:
            Dict with yearly costs, total exposure, cumulative NPV, and breakeven year.
        """
        assessments = self._assessments.get(org_id, [])
        policy_assessments = [
            a for a in assessments if a.risk_type == RiskType.TRANSITION_POLICY
        ]
        total_current = sum(a.current_exposure_usd for a in policy_assessments)

        if not carbon_price_trajectory:
            return {
                "org_id": org_id,
                "yearly_carbon_costs": {},
                "current_exposure": str(total_current),
                "message": "No carbon price trajectory provided",
            }

        # Get the baseline price (earliest year in trajectory)
        sorted_years = sorted(carbon_price_trajectory.items())
        baseline_price = sorted_years[0][1] if sorted_years else Decimal("25")

        yearly_costs: Dict[str, str] = {}
        cumulative = Decimal("0")

        # Interpolate prices for every year in range
        min_year = min(carbon_price_trajectory.keys())
        max_year = max(carbon_price_trajectory.keys())

        for year in range(min_year, max_year + 1):
            price = self._interpolate_price(year, carbon_price_trajectory)
            factor = price / max(baseline_price, Decimal("1"))
            annual_cost = (total_current * factor).quantize(Decimal("0.01"))
            yearly_costs[str(year)] = str(annual_cost)
            cumulative += annual_cost

        # Calculate NPV of cumulative carbon costs
        discount_rate = self.config.default_discount_rate
        npv = Decimal("0")
        base_year = min_year
        for year in range(min_year, max_year + 1):
            annual = Decimal(yearly_costs[str(year)])
            t = year - base_year
            if t >= 0:
                discount_factor = (Decimal("1") + discount_rate) ** t
                npv += annual / discount_factor

        return {
            "org_id": org_id,
            "yearly_carbon_costs": yearly_costs,
            "current_exposure": str(total_current),
            "cumulative_cost": str(cumulative),
            "npv_carbon_costs": str(npv.quantize(Decimal("0.01"))),
            "trajectory_years": f"{min_year}-{max_year}",
        }

    async def calculate_carbon_cost_by_scenario(
        self, org_id: str,
        total_emissions_tco2e: Decimal,
        scenarios: Optional[List[ScenarioType]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate carbon costs under multiple IEA/NGFS scenarios.

        Args:
            org_id: Organization identifier.
            total_emissions_tco2e: Total Scope 1+2 emissions (tCO2e).
            scenarios: List of scenario types (defaults to all IEA scenarios).

        Returns:
            Dict with scenario comparison of carbon costs at 2030 and 2050.
        """
        if scenarios is None:
            scenarios = [ScenarioType.IEA_NZE, ScenarioType.IEA_APS, ScenarioType.IEA_STEPS]

        results: List[Dict[str, Any]] = []
        for scenario_type in scenarios:
            prices = SCENARIO_CARBON_PRICES.get(scenario_type, {})
            price_2030 = Decimal(str(prices.get(2030, 0)))
            price_2050 = Decimal(str(prices.get(2050, 0)))

            cost_2030 = total_emissions_tco2e * price_2030
            cost_2050 = total_emissions_tco2e * Decimal("0.3") * price_2050

            results.append({
                "scenario": scenario_type.value,
                "carbon_price_2030": str(price_2030),
                "carbon_price_2050": str(price_2050),
                "annual_cost_2030": str(cost_2030.quantize(Decimal("0.01"))),
                "annual_cost_2050": str(cost_2050.quantize(Decimal("0.01"))),
                "cost_as_pct_of_emissions": str(
                    (cost_2030 / max(total_emissions_tco2e, Decimal("1"))).quantize(Decimal("0.01"))
                ),
            })

        # Sort by 2030 cost (highest exposure first)
        results.sort(key=lambda x: Decimal(x["annual_cost_2030"]), reverse=True)

        return {
            "org_id": org_id,
            "total_emissions_tco2e": str(total_emissions_tco2e),
            "scenario_count": len(results),
            "scenario_costs": results,
            "worst_case_2030": results[0]["annual_cost_2030"] if results else "0",
            "best_case_2030": results[-1]["annual_cost_2030"] if results else "0",
        }

    # ------------------------------------------------------------------
    # Stranded Asset Analysis
    # ------------------------------------------------------------------

    async def assess_stranding_probability(
        self, org_id: str, asset_type: str, sector: SectorType,
    ) -> Dict[str, Any]:
        """
        Assess asset stranding probability based on sector and type.

        Uses sector transition profiles to derive base stranding risk and
        applies asset-type adjustments.

        Args:
            org_id: Organization identifier.
            asset_type: Type of asset (e.g. "fossil_fuel_reserves", "ICE_fleet").
            sector: TCFD sector classification.

        Returns:
            Dict with stranding probability, timeline, and mitigation guidance.
        """
        profile = SECTOR_TRANSITION_PROFILES.get(sector, {})
        base_risk = profile.get("stranding_risk", "medium")

        probability_map = {"low": 10, "medium": 30, "high": 55, "very_high": 80}
        base_probability = probability_map.get(base_risk, 30)

        # Asset-type adjustments
        high_risk_assets = {"fossil_fuel_reserves", "coal_plant", "oil_refinery", "gas_pipeline"}
        medium_risk_assets = {"ice_fleet", "gas_turbine", "diesel_generator"}
        if asset_type.lower() in high_risk_assets:
            base_probability = min(base_probability + 25, 95)
        elif asset_type.lower() in medium_risk_assets:
            base_probability = min(base_probability + 10, 85)

        # Estimate stranding timeline based on probability
        if base_probability >= 70:
            stranding_year = 2030
            urgency = "immediate"
        elif base_probability >= 40:
            stranding_year = 2035
            urgency = "high"
        else:
            stranding_year = 2045
            urgency = "moderate"

        pathway = profile.get("decarbonization_pathway", "transition planning")

        return {
            "org_id": org_id,
            "asset_type": asset_type,
            "sector": sector.value,
            "stranding_probability_pct": base_probability,
            "sector_stranding_risk": base_risk,
            "estimated_stranding_year": stranding_year,
            "urgency": urgency,
            "mitigation_recommendation": f"Consider diversification away from {sector.value} sector assets",
            "decarbonization_pathway": pathway,
        }

    async def assess_stranded_asset_portfolio(
        self, org_id: str, tenant_id: str,
        assets: List[Dict[str, Any]],
        scenario_type: ScenarioType = ScenarioType.IEA_NZE,
    ) -> Dict[str, Any]:
        """
        Assess stranding risk for an entire portfolio of assets.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            assets: List of dicts with asset_id, book_value, sector, asset_type.
            scenario_type: Climate scenario for valuation.

        Returns:
            Dict with total portfolio stranding exposure and per-asset breakdown.
        """
        total_book_value = Decimal("0")
        total_impairment = Decimal("0")
        asset_results: List[Dict[str, Any]] = []
        stranded_analyses: List[StrandedAssetAnalysis] = []

        for asset in assets:
            book_value = Decimal(str(asset.get("book_value", 0)))
            sector = SectorType(asset.get("sector", "energy"))
            asset_type_str = asset.get("asset_type", "general")
            asset_id = asset.get("asset_id", _new_id())

            stranding_info = await self.assess_stranding_probability(
                org_id, asset_type_str, sector,
            )
            prob = Decimal(str(stranding_info["stranding_probability_pct"])) / 100
            impairment = (book_value * prob).quantize(Decimal("0.01"))
            projected_value = book_value - impairment

            total_book_value += book_value
            total_impairment += impairment

            analysis = StrandedAssetAnalysis(
                tenant_id=tenant_id,
                org_id=org_id,
                asset_id=asset_id,
                scenario_id=scenario_type.value,
                current_book_value_usd=book_value,
                projected_value_usd=projected_value,
                impairment_usd=impairment,
                stranding_probability_pct=prob * 100,
                stranding_year=stranding_info.get("estimated_stranding_year"),
            )
            stranded_analyses.append(analysis)

            asset_results.append({
                "asset_id": asset_id,
                "asset_type": asset_type_str,
                "sector": sector.value,
                "book_value": str(book_value),
                "stranding_probability_pct": stranding_info["stranding_probability_pct"],
                "impairment": str(impairment),
                "projected_value": str(projected_value),
            })

        # Store analyses
        if org_id not in self._stranded_analyses:
            self._stranded_analyses[org_id] = []
        self._stranded_analyses[org_id].extend(stranded_analyses)

        return {
            "org_id": org_id,
            "scenario": scenario_type.value,
            "total_book_value": str(total_book_value),
            "total_impairment": str(total_impairment),
            "impairment_pct": str(
                (total_impairment / max(total_book_value, Decimal("1")) * 100).quantize(
                    Decimal("0.1")
                )
            ),
            "asset_count": len(asset_results),
            "asset_breakdown": asset_results,
        }

    # ------------------------------------------------------------------
    # Regulatory Compliance Costs
    # ------------------------------------------------------------------

    async def calculate_regulatory_compliance_cost(
        self, org_id: str,
        regulations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate total regulatory compliance cost with detailed breakdown.

        Args:
            org_id: Organization identifier.
            regulations: List of dicts with name, jurisdiction,
                effective_date, estimated_cost, and category.

        Returns:
            Dict with total cost, yearly breakdown, and per-regulation detail.
        """
        total = Decimal("0")
        one_time = Decimal("0")
        recurring = Decimal("0")
        breakdown: List[Dict[str, str]] = []

        for reg in regulations:
            cost = Decimal(str(reg.get("estimated_cost", 0)))
            cost_type = reg.get("cost_type", "recurring")
            total += cost
            if cost_type == "one_time":
                one_time += cost
            else:
                recurring += cost

            breakdown.append({
                "regulation": reg.get("name", "Unknown"),
                "jurisdiction": reg.get("jurisdiction", ""),
                "effective_date": reg.get("effective_date", ""),
                "cost": str(cost),
                "cost_type": cost_type,
                "category": reg.get("category", "compliance"),
            })

        # Sort by cost descending
        breakdown.sort(key=lambda x: Decimal(x["cost"]), reverse=True)

        return {
            "org_id": org_id,
            "total_compliance_cost": str(total),
            "one_time_costs": str(one_time),
            "recurring_annual_costs": str(recurring),
            "regulation_count": len(breakdown),
            "breakdown": breakdown,
        }

    # ------------------------------------------------------------------
    # Litigation Risk Assessment
    # ------------------------------------------------------------------

    async def assess_litigation_risk(
        self, org_id: str,
        sector: Optional[SectorType] = None,
    ) -> Dict[str, Any]:
        """
        Assess climate litigation risk exposure.

        Calculates litigation exposure based on total transition risk
        assessments, sector profile, and common litigation risk factors.

        Args:
            org_id: Organization identifier.
            sector: Optional sector for sector-specific litigation profile.

        Returns:
            Dict with litigation exposure, risk factors, and mitigations.
        """
        assessments = self._assessments.get(org_id, [])
        total_exposure = sum(a.financial_impact_usd for a in assessments)

        # Base litigation factor varies by sector
        base_factor = Decimal("0.05")
        if sector:
            profile = SECTOR_TRANSITION_PROFILES.get(sector, {})
            exposure_level = profile.get("transition_exposure", "medium")
            if exposure_level in ("very_high", "high"):
                base_factor = Decimal("0.08")
            elif exposure_level == "medium":
                base_factor = Decimal("0.05")
            else:
                base_factor = Decimal("0.02")

        litigation_exposure = (total_exposure * base_factor).quantize(Decimal("0.01"))

        # Categorize risk factors with probability weights
        risk_factors = [
            {"factor": "Greenwashing claims", "probability": "medium", "trend": "increasing"},
            {"factor": "Failure to disclose material risks", "probability": "high", "trend": "increasing"},
            {"factor": "Breach of fiduciary duty", "probability": "low", "trend": "stable"},
            {"factor": "Climate attribution liability", "probability": "low", "trend": "increasing"},
            {"factor": "Shareholder derivative actions", "probability": "medium", "trend": "increasing"},
            {"factor": "Regulatory enforcement actions", "probability": "medium", "trend": "increasing"},
        ]

        return {
            "org_id": org_id,
            "litigation_exposure": str(litigation_exposure),
            "total_transition_exposure": str(total_exposure),
            "litigation_factor": str(base_factor),
            "risk_factors": risk_factors,
            "mitigation": [
                "Ensure TCFD-aligned disclosure and substantiate climate claims",
                "Implement robust internal controls for climate data",
                "Engage external assurance for emissions disclosures",
                "Maintain alignment between public commitments and actions",
            ],
        }

    # ------------------------------------------------------------------
    # Technology Disruption Assessment
    # ------------------------------------------------------------------

    async def assess_technology_disruption(
        self, org_id: str,
        sector: SectorType,
    ) -> Dict[str, Any]:
        """
        Assess sector-specific technology disruption probabilities.

        Args:
            org_id: Organization identifier.
            sector: TCFD sector classification.

        Returns:
            Dict with technology disruption probabilities and timelines.
        """
        disruption_data = _TECH_DISRUPTION_PROBABILITY.get(sector, {})
        if not disruption_data:
            return {
                "org_id": org_id,
                "sector": sector.value,
                "message": "No technology disruption data available for this sector",
                "technologies": [],
            }

        technologies: List[Dict[str, Any]] = []
        avg_probability = 0.0

        for tech, probability in disruption_data.items():
            # Timeline estimation based on probability
            if probability >= 0.7:
                timeline = "short_term"
                readiness = "mature"
            elif probability >= 0.4:
                timeline = "medium_term"
                readiness = "emerging"
            else:
                timeline = "long_term"
                readiness = "early_stage"

            technologies.append({
                "technology": tech,
                "disruption_probability": round(probability * 100, 1),
                "expected_timeline": timeline,
                "technology_readiness": readiness,
            })
            avg_probability += probability

        avg_probability /= max(len(disruption_data), 1)

        # Sort by probability descending
        technologies.sort(key=lambda x: x["disruption_probability"], reverse=True)

        return {
            "org_id": org_id,
            "sector": sector.value,
            "average_disruption_probability": round(avg_probability * 100, 1),
            "technology_count": len(technologies),
            "technologies": technologies,
            "sector_recommendation": SECTOR_TRANSITION_PROFILES.get(sector, {}).get(
                "decarbonization_pathway", ""
            ),
        }

    # ------------------------------------------------------------------
    # Composite Risk Scoring
    # ------------------------------------------------------------------

    async def calculate_composite_transition_risk(
        self, org_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate weighted composite transition risk score across all categories.

        Applies category weights (policy 35%, technology 25%, market 25%,
        reputation 15%) and sector exposure multipliers.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with composite score, category breakdown, and risk rating.
        """
        assessments = self._assessments.get(org_id, [])
        if not assessments:
            return {"org_id": org_id, "composite_score": 0, "message": "No assessments"}

        total_impact = sum(a.financial_impact_usd for a in assessments)

        # Group by risk type
        by_type: Dict[str, List[TransitionRiskAssessment]] = {}
        for a in assessments:
            key = a.risk_type.value
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(a)

        # Calculate weighted scores
        weighted_impact = Decimal("0")
        category_breakdown: Dict[str, Dict[str, str]] = {}

        for risk_type, weight in _RISK_CATEGORY_WEIGHTS.items():
            type_assessments = by_type.get(risk_type.value, [])
            type_impact = sum(a.financial_impact_usd for a in type_assessments)
            weighted_contribution = type_impact * weight
            weighted_impact += weighted_contribution

            category_breakdown[risk_type.value] = {
                "assessment_count": str(len(type_assessments)),
                "total_impact": str(type_impact),
                "weight": str(weight),
                "weighted_impact": str(weighted_contribution.quantize(Decimal("0.01"))),
            }

        # Determine risk rating based on weighted exposure
        risk_rating = self._impact_to_rating(weighted_impact)

        return {
            "org_id": org_id,
            "total_transition_risk_exposure": str(total_impact),
            "weighted_transition_risk": str(weighted_impact.quantize(Decimal("0.01"))),
            "risk_rating": risk_rating,
            "assessment_count": len(assessments),
            "category_breakdown": category_breakdown,
        }

    async def compare_transition_risk_scenarios(
        self, org_id: str, tenant_id: str,
        base_data: Dict[str, Any],
        scenarios: Optional[List[ScenarioType]] = None,
    ) -> Dict[str, Any]:
        """
        Compare transition risk exposure across multiple scenarios.

        Runs policy risk assessment under each scenario's carbon prices
        and compares financial impact.

        Args:
            org_id: Organization identifier.
            tenant_id: Tenant identifier.
            base_data: Organization data (emissions, sector, etc.).
            scenarios: Scenario types to compare.

        Returns:
            Dict with per-scenario comparison and range analysis.
        """
        if scenarios is None:
            scenarios = [ScenarioType.IEA_NZE, ScenarioType.IEA_APS, ScenarioType.IEA_STEPS]

        scenario_results: List[Dict[str, Any]] = []

        for scenario_type in scenarios:
            prices = SCENARIO_CARBON_PRICES.get(scenario_type, {})
            lib_data = SCENARIO_LIBRARY.get(scenario_type, {})

            policy_data = dict(base_data)
            policy_data["carbon_price_2030"] = prices.get(2030, 50)
            policy_data["carbon_price_2050"] = prices.get(2050, 100)

            assessment = await self.assess_policy_risk(org_id, tenant_id, policy_data)

            scenario_results.append({
                "scenario": scenario_type.value,
                "scenario_name": lib_data.get("name", scenario_type.value),
                "carbon_price_2030": str(prices.get(2030, 0)),
                "carbon_price_2050": str(prices.get(2050, 0)),
                "financial_impact": str(assessment.financial_impact_usd),
                "current_exposure": str(assessment.current_exposure_usd),
                "projected_2030": str(assessment.projected_exposure_2030_usd),
                "projected_2050": str(assessment.projected_exposure_2050_usd),
            })

        # Calculate range
        impacts = [Decimal(r["financial_impact"]) for r in scenario_results]
        min_impact = min(impacts) if impacts else Decimal("0")
        max_impact = max(impacts) if impacts else Decimal("0")

        return {
            "org_id": org_id,
            "scenario_count": len(scenario_results),
            "scenario_results": scenario_results,
            "min_impact": str(min_impact),
            "max_impact": str(max_impact),
            "impact_range": str(max_impact - min_impact),
        }

    # ------------------------------------------------------------------
    # Sector & Regulation Lookups
    # ------------------------------------------------------------------

    async def get_sector_transition_profile(
        self, sector: SectorType,
    ) -> Dict[str, str]:
        """Get transition risk profile for a TCFD sector."""
        return SECTOR_TRANSITION_PROFILES.get(sector, {
            "transition_exposure": "unknown",
            "stranding_risk": "unknown",
            "key_drivers": "",
            "decarbonization_pathway": "",
        })

    async def track_regulation_timeline(
        self, org_id: str,
        jurisdictions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get upcoming regulation timeline, optionally filtered by jurisdiction.

        Args:
            org_id: Organization identifier.
            jurisdictions: Optional list of jurisdiction codes to filter.

        Returns:
            List of regulation timeline entries sorted by effective date.
        """
        custom = self._regulations.get(org_id, [])
        combined = list(custom) + list(CARBON_PRICE_REGULATIONS)

        if jurisdictions:
            jurisdictions_upper = [j.upper() for j in jurisdictions]
            combined = [
                r for r in combined
                if r.get("jurisdiction", "").upper() in jurisdictions_upper
            ]

        # Sort by effective date
        combined.sort(key=lambda x: x.get("effective", "9999-12-31"))

        return combined

    async def add_custom_regulation(
        self, org_id: str, regulation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Add a custom regulation to the organization's tracking timeline.

        Args:
            org_id: Organization identifier.
            regulation: Dict with regulation, jurisdiction, effective, impact.

        Returns:
            The stored regulation with ID.
        """
        regulation["id"] = _new_id()
        regulation["added_at"] = _now().isoformat()

        if org_id not in self._regulations:
            self._regulations[org_id] = []
        self._regulations[org_id].append(regulation)

        logger.info(
            "Added custom regulation '%s' for org %s",
            regulation.get("regulation", "Unknown"), org_id,
        )
        return regulation

    # ------------------------------------------------------------------
    # Disclosure Generation
    # ------------------------------------------------------------------

    async def generate_transition_risk_disclosure(
        self, org_id: str,
    ) -> Dict[str, Any]:
        """
        Generate TCFD-aligned transition risk disclosure content.

        Produces structured disclosure covering all four transition risk
        categories with financial impact quantification, sector context,
        and mitigation strategies.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with disclosure content, risk breakdown, and compliance score.
        """
        assessments = self._assessments.get(org_id, [])
        total = sum(a.financial_impact_usd for a in assessments)

        # Group by category
        by_category: Dict[str, List[TransitionRiskAssessment]] = {}
        for a in assessments:
            key = a.risk_type.value
            if key not in by_category:
                by_category[key] = []
            by_category[key].append(a)

        # Build category summaries
        category_summaries: List[str] = []
        for cat_name, cat_assessments in by_category.items():
            cat_total = sum(a.financial_impact_usd for a in cat_assessments)
            display_name = cat_name.replace("transition_", "").replace("_", " ").title()
            category_summaries.append(
                f"{display_name}: {len(cat_assessments)} risk(s) with "
                f"total exposure of ${cat_total:,.0f}"
            )

        # Sectors represented
        sectors = list(set(a.sector.value for a in assessments))

        # Mitigation actions across all assessments
        all_mitigations: List[str] = []
        for a in assessments:
            all_mitigations.extend(a.mitigation_actions)
        unique_mitigations = list(set(all_mitigations))

        # Build narrative
        narrative_parts = [
            f"The organization has assessed {len(assessments)} transition risk(s) "
            f"with total financial exposure of ${total:,.0f}. ",
        ]
        if category_summaries:
            narrative_parts.append(
                "Transition risks span " + "; ".join(category_summaries) + ". "
            )
        if sectors:
            narrative_parts.append(
                f"Sector(s) assessed: {', '.join(sectors)}. "
            )
        if unique_mitigations:
            narrative_parts.append(
                f"Key mitigation actions include: {'; '.join(unique_mitigations[:5])}."
            )

        # Calculate compliance score (0-100)
        compliance_score = self._score_transition_disclosure(assessments, by_category)

        return {
            "org_id": org_id,
            "content": "".join(narrative_parts),
            "risk_count": len(assessments),
            "total_exposure": str(total),
            "categories_covered": list(by_category.keys()),
            "sectors_assessed": sectors,
            "mitigation_actions": unique_mitigations[:10],
            "compliance_score": compliance_score,
            "category_breakdown": {
                k: {
                    "count": len(v),
                    "total_impact": str(sum(a.financial_impact_usd for a in v)),
                }
                for k, v in by_category.items()
            },
        }

    async def get_transition_risk_summary(
        self, org_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary view of all transition risk assessments for an org.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with counts, totals, ratings, and category breakdown.
        """
        assessments = self._assessments.get(org_id, [])
        if not assessments:
            return {
                "org_id": org_id,
                "assessment_count": 0,
                "total_exposure": "0",
                "message": "No transition risk assessments found",
            }

        total = sum(a.financial_impact_usd for a in assessments)
        current_total = sum(a.current_exposure_usd for a in assessments)
        projected_2030 = sum(a.projected_exposure_2030_usd for a in assessments)
        projected_2050 = sum(a.projected_exposure_2050_usd for a in assessments)

        # Category counts
        by_type: Dict[str, int] = {}
        for a in assessments:
            by_type[a.risk_type.value] = by_type.get(a.risk_type.value, 0) + 1

        # Sector distribution
        by_sector: Dict[str, int] = {}
        for a in assessments:
            by_sector[a.sector.value] = by_sector.get(a.sector.value, 0) + 1

        return {
            "org_id": org_id,
            "assessment_count": len(assessments),
            "total_financial_impact": str(total),
            "current_total_exposure": str(current_total),
            "projected_2030_total": str(projected_2030),
            "projected_2050_total": str(projected_2050),
            "risk_rating": self._impact_to_rating(total),
            "by_risk_type": by_type,
            "by_sector": by_sector,
            "coverage": {
                "policy": "transition_policy" in by_type,
                "technology": "transition_technology" in by_type,
                "market": "transition_market" in by_type,
                "reputation": "transition_reputation" in by_type,
            },
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _store_assessment(
        self, org_id: str, assessment: TransitionRiskAssessment,
    ) -> None:
        """Store a transition risk assessment."""
        if org_id not in self._assessments:
            self._assessments[org_id] = []
        self._assessments[org_id].append(assessment)

    def _store_policy_risk(self, org_id: str, policy_risk: PolicyRisk) -> None:
        """Store a detailed policy risk record."""
        if org_id not in self._policy_risks:
            self._policy_risks[org_id] = []
        self._policy_risks[org_id].append(policy_risk)

    @staticmethod
    def _interpolate_price(
        year: int, trajectory: Dict[int, Decimal],
    ) -> Decimal:
        """
        Linearly interpolate a carbon price for a given year.

        If the year falls between two known data points, a linear
        interpolation is used. If before the first point, the first
        value is returned. If after the last, the last value is returned.
        """
        sorted_points = sorted(trajectory.items())
        if not sorted_points:
            return Decimal("0")

        # Before first point
        if year <= sorted_points[0][0]:
            return sorted_points[0][1]
        # After last point
        if year >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        # Find bounding points and interpolate
        for i in range(len(sorted_points) - 1):
            y0, p0 = sorted_points[i]
            y1, p1 = sorted_points[i + 1]
            if y0 <= year <= y1:
                fraction = Decimal(str(year - y0)) / Decimal(str(max(y1 - y0, 1)))
                return (p0 + (p1 - p0) * fraction).quantize(Decimal("0.01"))

        return sorted_points[-1][1]

    @staticmethod
    def _impact_to_rating(impact: Decimal) -> str:
        """Convert financial impact to a qualitative risk rating."""
        impact_float = float(impact)
        if impact_float >= 10_000_000:
            return "critical"
        elif impact_float >= 5_000_000:
            return "high"
        elif impact_float >= 1_000_000:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _score_transition_disclosure(
        assessments: List[TransitionRiskAssessment],
        by_category: Dict[str, List[TransitionRiskAssessment]],
    ) -> int:
        """
        Score transition risk disclosure completeness (0-100).

        Scoring criteria:
        - Assessments exist: 20 pts
        - All 4 categories covered: 20 pts (5 each)
        - Financial quantification present: 15 pts
        - Multiple sectors assessed: 10 pts
        - 2030 projections present: 10 pts
        - 2050 projections present: 10 pts
        - Mitigation actions documented: 15 pts
        """
        if not assessments:
            return 0

        score = 20  # Base for having assessments

        # Category coverage (5 pts each, max 20)
        for cat in ["transition_policy", "transition_technology",
                     "transition_market", "transition_reputation"]:
            if cat in by_category:
                score += 5

        # Financial quantification (15 pts)
        if all(a.financial_impact_usd > 0 for a in assessments):
            score += 15
        elif any(a.financial_impact_usd > 0 for a in assessments):
            score += 8

        # Multi-sector (10 pts)
        sectors = set(a.sector.value for a in assessments)
        if len(sectors) >= 2:
            score += 10
        elif len(sectors) == 1:
            score += 5

        # 2030 projections (10 pts)
        if all(a.projected_exposure_2030_usd > 0 for a in assessments):
            score += 10

        # 2050 projections (10 pts)
        if all(a.projected_exposure_2050_usd > 0 for a in assessments):
            score += 10

        # Mitigation actions (15 pts)
        if all(len(a.mitigation_actions) > 0 for a in assessments):
            score += 15
        elif any(len(a.mitigation_actions) > 0 for a in assessments):
            score += 8

        return min(score, 100)
