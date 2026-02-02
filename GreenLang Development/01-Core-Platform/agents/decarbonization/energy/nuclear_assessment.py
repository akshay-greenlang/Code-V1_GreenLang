# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-006: Nuclear Assessment Agent

Assesses nuclear energy options including large reactors, SMRs,
and advanced technologies.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class NuclearAssessmentAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-006: Nuclear Assessment Agent"""

    AGENT_ID = "GL-DECARB-ENE-006"
    AGENT_NAME = "Nuclear Assessment Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-006",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Nuclear energy technology assessment"
    )

    NUCLEAR_TECHNOLOGIES = {
        "large_lwr": {
            "capacity_gw": 1.0,
            "overnight_cost_per_kw": 6500,
            "construction_years": 10,
            "capacity_factor": 0.92,
            "lifetime": 60,
        },
        "smr": {
            "capacity_gw": 0.3,
            "overnight_cost_per_kw": 6000,
            "construction_years": 5,
            "capacity_factor": 0.90,
            "lifetime": 40,
        },
        "gen_iv": {
            "capacity_gw": 0.5,
            "overnight_cost_per_kw": 8000,
            "construction_years": 8,
            "capacity_factor": 0.85,
            "lifetime": 40,
        },
    }

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-006", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        capacity_req = inputs.get("capacity_requirement_gw", 1.0)
        tech_pref = inputs.get("technology_preference", "smr")
        baseline_year = inputs.get("baseline_year", 2024)

        # Select technology
        if tech_pref in self.NUCLEAR_TECHNOLOGIES:
            tech = tech_pref
        else:
            tech = "smr" if capacity_req < 1.0 else "large_lwr"

        specs = self.NUCLEAR_TECHNOLOGIES[tech]

        # Calculate number of units needed
        num_units = max(1, round(capacity_req / specs["capacity_gw"]))
        total_capacity = num_units * specs["capacity_gw"]

        # Costs
        overnight_cost = total_capacity * 1e6 * specs["overnight_cost_per_kw"] / 1e9

        # Timeline
        construction_start = baseline_year + 3  # Licensing period
        commercial_operation = construction_start + specs["construction_years"]

        # Generation
        hours_per_year = 8760
        annual_generation = total_capacity * specs["capacity_factor"] * hours_per_year

        # LCOE
        lcoe = self.calculate_lcoe(
            capital_cost_per_kw=specs["overnight_cost_per_kw"],
            fixed_om_per_kw_year=100,
            variable_om_per_mwh=2,
            capacity_factor=specs["capacity_factor"],
            lifetime_years=specs["lifetime"],
            fuel_cost_per_mwh=7,  # Nuclear fuel
        )

        # Avoided emissions
        avoided_mt = annual_generation * 0.4 / 1000  # 400 kg/MWh displaced

        return {
            "organization_id": inputs.get("organization_id", ""),
            "recommended_technology": tech,
            "number_of_units": num_units,
            "total_capacity_gw": round(total_capacity, 2),
            "development_timeline_years": specs["construction_years"] + 3,
            "construction_start_year": construction_start,
            "commercial_operation_year": commercial_operation,
            "overnight_cost_billion_usd": round(overnight_cost, 2),
            "lcoe_usd_mwh": lcoe,
            "annual_generation_twh": round(annual_generation / 1000, 2),
            "annual_emissions_avoided_mt": round(avoided_mt, 2),
            "recommended_pathways": ["nuclear"],
            "total_abatement_mtco2e": round(avoided_mt, 2),
            "total_investment_million_usd": round(overnight_cost * 1000, 0),
            "levelized_abatement_cost_usd_tco2e": round(
                overnight_cost * 1e9 / (avoided_mt * 1e6) / specs["lifetime"], 2
            ),
            "confidence_level": 0.60,
            "key_risks": [
                "Construction cost overruns",
                "Regulatory approval timeline",
                "Public acceptance",
                "Waste management",
            ],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
