# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-007: CCUS for Power Agent

Plans carbon capture, utilization, and storage for power sector applications.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class CCUSPowerAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-007: CCUS for Power Agent"""

    AGENT_ID = "GL-DECARB-ENE-007"
    AGENT_NAME = "CCUS for Power Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-007",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Carbon capture for power sector"
    )

    CCUS_TECHNOLOGIES = {
        "post_combustion": {
            "capture_rate": 0.90,
            "capital_per_tco2_year": 80,
            "operating_per_tco2": 30,
            "energy_penalty": 0.25,
        },
        "pre_combustion": {
            "capture_rate": 0.85,
            "capital_per_tco2_year": 70,
            "operating_per_tco2": 25,
            "energy_penalty": 0.20,
        },
        "oxy_combustion": {
            "capture_rate": 0.95,
            "capital_per_tco2_year": 90,
            "operating_per_tco2": 35,
            "energy_penalty": 0.30,
        },
    }

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-007", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        facilities = inputs.get("target_facilities", [])
        tech_pref = inputs.get("technology_preference", "post_combustion")
        transport_km = inputs.get("co2_transport_distance_km", 50)

        # Calculate total emissions from facilities
        total_emissions = sum(f.get("annual_emissions_mt", 1.0) for f in facilities)
        total_emissions_mt = total_emissions

        # Technology selection
        tech = tech_pref if tech_pref in self.CCUS_TECHNOLOGIES else "post_combustion"
        specs = self.CCUS_TECHNOLOGIES[tech]

        # Capture capacity
        capture_mt = total_emissions_mt * specs["capture_rate"]

        # Costs
        capital = capture_mt * 1e6 * specs["capital_per_tco2_year"]
        transport_cost = transport_km * 0.02 * capture_mt * 1e6  # $0.02/tCO2/km
        storage_cost = capture_mt * 1e6 * 10  # $10/tCO2 storage

        total_capital = capital + transport_cost + storage_cost

        # Operating costs
        annual_operating = capture_mt * 1e6 * specs["operating_per_tco2"]

        # Cost per tonne
        capture_cost = (total_capital / 25 + annual_operating) / (capture_mt * 1e6)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "capture_technology": tech,
            "capture_capacity_mtco2_year": round(capture_mt, 2),
            "capture_rate_pct": round(specs["capture_rate"] * 100, 0),
            "storage_solution": "geological_saline_aquifer",
            "storage_capacity_mt": round(capture_mt * 25, 0),  # 25-year capacity
            "capture_cost_usd_tco2": round(capture_cost, 0),
            "capital_investment_million_usd": round(total_capital / 1e6, 0),
            "energy_penalty_pct": round(specs["energy_penalty"] * 100, 0),
            "net_emissions_reduction_mtco2_year": round(capture_mt, 2),
            "recommended_pathways": ["ccus"],
            "total_abatement_mtco2e": round(capture_mt, 2),
            "total_investment_million_usd": round(total_capital / 1e6, 0),
            "levelized_abatement_cost_usd_tco2e": round(capture_cost, 0),
            "confidence_level": 0.65,
            "key_risks": [
                "Storage site availability",
                "Capture cost reduction trajectory",
                "CO2 transport infrastructure",
                "Long-term storage liability",
            ],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
