# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-005: Hydrogen Strategy Agent

Plans hydrogen transition strategy including production pathway selection,
infrastructure requirements, and economic analysis.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class HydrogenStrategyAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-005: Hydrogen Strategy Agent"""

    AGENT_ID = "GL-DECARB-ENE-005"
    AGENT_NAME = "Hydrogen Strategy Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-005",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Hydrogen transition strategy and planning"
    )

    # Hydrogen production costs ($/kg H2)
    PRODUCTION_COSTS = {
        "smr_grey": 1.50,
        "smr_blue": 2.20,
        "electrolysis_grid": 5.00,
        "electrolysis_renewable": 4.00,
    }

    # Carbon intensity (kg CO2/kg H2)
    CARBON_INTENSITY = {
        "smr_grey": 10.0,
        "smr_blue": 2.0,
        "electrolysis_grid": 25.0,
        "electrolysis_renewable": 0.5,
    }

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-005", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        target_demand = inputs.get("target_hydrogen_demand_tonnes_year", 10000)
        renewable_price = inputs.get("renewable_electricity_price_usd_mwh", 30)
        gas_price = inputs.get("natural_gas_price_usd_mmbtu", 4)

        # Determine best production method
        if renewable_price < 25:
            method = "electrolysis_renewable"
            electrolyzer_mw = target_demand * 55 / 8760 * 1.2  # 55 kWh/kg, 80% CF
        elif renewable_price < 40:
            method = "electrolysis_renewable"
            electrolyzer_mw = target_demand * 55 / 8760 * 1.2
        elif gas_price < 5:
            method = "smr_blue"
            electrolyzer_mw = None
        else:
            method = "electrolysis_renewable"
            electrolyzer_mw = target_demand * 55 / 8760 * 1.2

        prod_cost = self.PRODUCTION_COSTS.get(method, 4.0)
        carbon_intensity = self.CARBON_INTENSITY.get(method, 5.0)

        # Adjust for electricity price
        if "electrolysis" in method:
            prod_cost = renewable_price * 0.055 + 1.5  # $1.5/kg fixed costs

        # Capital costs
        if electrolyzer_mw:
            capital = electrolyzer_mw * 1200  # $1200/kW electrolyzer
        else:
            capital = target_demand * 150  # SMR capital

        # Storage
        storage_tonnes = target_demand / 365 * 3  # 3 days storage
        storage_capital = storage_tonnes * 500  # $500/kg storage

        # Operating costs
        annual_operating = target_demand * prod_cost

        # Emissions avoided vs grey hydrogen
        baseline_intensity = 10.0
        avoided_emissions = target_demand * (baseline_intensity - carbon_intensity)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "recommended_production_method": method,
            "production_capacity_tonnes_year": target_demand,
            "electrolyzer_capacity_mw": round(electrolyzer_mw, 0) if electrolyzer_mw else None,
            "storage_capacity_tonnes": round(storage_tonnes, 0),
            "pipeline_km": 0,
            "hydrogen_production_cost_usd_kg": round(prod_cost, 2),
            "capital_investment_million_usd": round((capital + storage_capital) / 1e6, 2),
            "annual_operating_cost_million_usd": round(annual_operating / 1e6, 2),
            "hydrogen_carbon_intensity_kg_co2_kg_h2": carbon_intensity,
            "annual_emissions_avoided_tonnes": round(avoided_emissions, 0),
            "recommended_pathways": ["hydrogen"],
            "total_abatement_mtco2e": round(avoided_emissions / 1000, 2),
            "total_investment_million_usd": round((capital + storage_capital) / 1e6, 2),
            "levelized_abatement_cost_usd_tco2e": round(
                (capital + storage_capital) / avoided_emissions / 20, 2
            ) if avoided_emissions > 0 else 0,
            "confidence_level": 0.65,
            "key_risks": [
                "Electrolyzer cost trajectory",
                "Renewable electricity availability",
                "Offtake certainty",
            ],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
