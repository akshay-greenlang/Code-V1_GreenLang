# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-008: Distributed Generation Agent

Plans distributed energy resource (DER) deployment including rooftop solar,
behind-the-meter storage, and virtual power plants.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class DistributedGenerationAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-008: Distributed Generation Agent"""

    AGENT_ID = "GL-DECARB-ENE-008"
    AGENT_NAME = "Distributed Generation Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-008",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Distributed energy resource planning"
    )

    # Sector characteristics
    SECTOR_POTENTIAL = {
        "residential": {"participation": 0.15, "avg_system_kw": 7, "storage_ratio": 0.3},
        "commercial": {"participation": 0.25, "avg_system_kw": 100, "storage_ratio": 0.2},
        "industrial": {"participation": 0.10, "avg_system_kw": 500, "storage_ratio": 0.15},
    }

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-008", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sector = inputs.get("sector", "commercial")
        num_buildings = inputs.get("number_of_buildings", 1000)
        avg_consumption = inputs.get("average_building_consumption_kwh", 50000)
        rooftop_km2 = inputs.get("rooftop_area_km2", 1.0)
        net_metering = inputs.get("net_metering_available", True)

        specs = self.SECTOR_POTENTIAL.get(sector, self.SECTOR_POTENTIAL["commercial"])

        # Calculate participation
        participating = num_buildings * specs["participation"]
        avg_system = specs["avg_system_kw"]

        # Solar capacity
        solar_mw = participating * avg_system / 1000

        # Storage capacity (paired with solar)
        storage_mwh = solar_mw * specs["storage_ratio"] * 4  # 4-hour duration

        # Generation
        capacity_factor = 0.18  # Rooftop lower than utility
        annual_generation = solar_mw * capacity_factor * 8760 / 1000  # GWh

        # Grid impact
        peak_reduction = solar_mw * 0.5  # 50% of capacity at peak

        # Grid export
        self_consumption = 0.70
        grid_export = annual_generation * (1 - self_consumption) * 1000  # MWh

        # Economics
        capital_per_kw = 1500  # Rooftop costs more
        total_capital = solar_mw * 1000 * capital_per_kw

        # Savings per building
        if net_metering:
            savings_per_kwh = 0.12
        else:
            savings_per_kwh = 0.08

        annual_savings = annual_generation * 1e6 * savings_per_kwh
        payback = total_capital / annual_savings

        # LCOE
        lcoe = self.calculate_lcoe(
            capital_cost_per_kw=capital_per_kw,
            fixed_om_per_kw_year=20,
            variable_om_per_mwh=0,
            capacity_factor=capacity_factor,
            lifetime_years=25,
        )

        # Emissions
        avoided_tonnes = self.calculate_avoided_emissions(annual_generation * 1000, "grid_average_us")

        return {
            "organization_id": inputs.get("organization_id", ""),
            "solar_pv_capacity_mw": round(solar_mw, 2),
            "battery_storage_mwh": round(storage_mwh, 2),
            "participation_rate_pct": round(specs["participation"] * 100, 0),
            "peak_reduction_mw": round(peak_reduction, 2),
            "grid_export_mwh_year": round(grid_export, 0),
            "average_payback_years": round(payback, 1),
            "lcoe_usd_kwh": round(lcoe / 1000, 3),
            "annual_emissions_avoided_tonnes": round(avoided_tonnes, 0),
            "recommended_pathways": ["renewable_integration", "demand_flexibility"],
            "total_abatement_mtco2e": round(avoided_tonnes / 1000, 3),
            "total_investment_million_usd": round(total_capital / 1e6, 2),
            "levelized_abatement_cost_usd_tco2e": round(
                total_capital / avoided_tonnes / 25, 2
            ),
            "confidence_level": 0.75,
            "key_risks": [
                "Customer adoption rates",
                "Net metering policy changes",
                "Installation workforce availability",
            ],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
