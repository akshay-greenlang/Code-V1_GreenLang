# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-004: Demand Flexibility Agent

Plans demand-side management and flexibility programs for grid support
and emissions reduction.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class DemandFlexibilityAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-004: Demand Flexibility Agent"""

    AGENT_ID = "GL-DECARB-ENE-004"
    AGENT_NAME = "Demand Flexibility Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-004",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Demand-side flexibility and management planning"
    )

    # Flexibility potential by sector (% of load)
    FLEXIBILITY_POTENTIAL = {
        "industrial": 0.15,
        "commercial": 0.12,
        "residential": 0.08,
    }

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-004", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sector = inputs.get("sector", "commercial")
        peak_mw = inputs.get("current_peak_mw", 100)
        annual_mwh = inputs.get("annual_consumption_mwh", 876000)

        flexibility_pct = self.FLEXIBILITY_POTENTIAL.get(sector, 0.10)
        flexible_load = peak_mw * flexibility_pct
        peak_reduction = flexible_load * 0.8

        # Value calculations
        peak_reduction_value = peak_reduction * 150 * 1000  # $150/kW-year
        energy_savings = annual_mwh * 0.02  # 2% efficiency gains
        savings_value = energy_savings * 50  # $50/MWh

        # Emissions from peak reduction (avoiding peaker plants)
        emissions_reduction = peak_reduction * 4 * 365 * 0.8 / 1000  # tonnes

        programs = [
            {"name": "Critical Peak Pricing", "potential_mw": flexible_load * 0.3},
            {"name": "Direct Load Control", "potential_mw": flexible_load * 0.4},
            {"name": "Behavioral DR", "potential_mw": flexible_load * 0.3},
        ]

        return {
            "organization_id": inputs.get("organization_id", ""),
            "total_flexible_load_mw": round(flexible_load, 2),
            "flexibility_percentage": round(flexibility_pct * 100, 1),
            "peak_reduction_mw": round(peak_reduction, 2),
            "recommended_programs": programs,
            "annual_savings_usd": round(savings_value, 0),
            "grid_benefit_usd": round(peak_reduction_value, 0),
            "emissions_reduction_tonnes": round(emissions_reduction, 0),
            "recommended_pathways": ["demand_flexibility", "efficiency"],
            "total_abatement_mtco2e": round(emissions_reduction / 1000, 4),
            "total_investment_million_usd": round(flexible_load * 50, 2),  # $50k/MW
            "levelized_abatement_cost_usd_tco2e": 25,
            "confidence_level": 0.70,
            "key_risks": ["Customer participation rates", "Technology adoption"],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
