# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-009: Grid Modernization Agent

Plans grid infrastructure upgrades for renewable integration, reliability,
and smart grid capabilities.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class GridModernizationAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-009: Grid Modernization Agent"""

    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-009",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Grid infrastructure modernization planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-009", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        smart_meter_pct = inputs.get("smart_meter_penetration_pct", 30)
        renewable_target = inputs.get("renewable_penetration_target_pct", 50)
        ev_target = inputs.get("ev_adoption_target_pct", 30)
        grid_chars = inputs.get("grid_characteristics", {})

        # Transmission investments
        transmission_investments = []
        transmission_cost = 0

        if renewable_target > 40:
            transmission_investments.append({
                "project": "Renewable integration lines",
                "cost_billion": 2.0,
                "timeline_years": 5,
            })
            transmission_cost += 2.0

        if renewable_target > 60:
            transmission_investments.append({
                "project": "HVDC backbone",
                "cost_billion": 5.0,
                "timeline_years": 8,
            })
            transmission_cost += 5.0

        # Distribution investments
        distribution_investments = []
        distribution_cost = 0

        if ev_target > 20:
            distribution_investments.append({
                "project": "EV charging infrastructure",
                "cost_billion": 1.5,
                "timeline_years": 5,
            })
            distribution_cost += 1.5

        distribution_investments.append({
            "project": "Substation automation",
            "cost_billion": 0.8,
            "timeline_years": 3,
        })
        distribution_cost += 0.8

        # Smart grid investments
        smart_grid_investments = []
        smart_cost = 0

        smart_meter_gap = 100 - smart_meter_pct
        if smart_meter_gap > 0:
            meter_cost = smart_meter_gap / 100 * 2.0  # $2B for full deployment
            smart_grid_investments.append({
                "project": "Smart meter rollout",
                "cost_billion": round(meter_cost, 2),
                "timeline_years": 5,
            })
            smart_cost += meter_cost

        smart_grid_investments.append({
            "project": "Grid management systems (DERMS/ADMS)",
            "cost_billion": 0.5,
            "timeline_years": 3,
        })
        smart_cost += 0.5

        total_investment = transmission_cost + distribution_cost + smart_cost

        # Benefits
        hosting_capacity_increase = min(renewable_target, 80)
        reliability_improvement = 15 + smart_meter_pct * 0.2

        annual_benefits = total_investment * 0.15  # 15% annual benefit

        return {
            "organization_id": inputs.get("organization_id", ""),
            "transmission_investments": transmission_investments,
            "distribution_investments": distribution_investments,
            "smart_grid_investments": smart_grid_investments,
            "hosting_capacity_increase_pct": round(hosting_capacity_increase, 0),
            "reliability_improvement_pct": round(reliability_improvement, 1),
            "total_investment_billion_usd": round(total_investment, 2),
            "annual_benefits_billion_usd": round(annual_benefits, 2),
            "recommended_pathways": ["grid_modernization"],
            "total_abatement_mtco2e": 0,  # Enabling investment
            "total_investment_million_usd": round(total_investment * 1000, 0),
            "levelized_abatement_cost_usd_tco2e": 0,
            "confidence_level": 0.70,
            "key_risks": [
                "Regulatory cost recovery",
                "Technology interoperability",
                "Cybersecurity requirements",
            ],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
