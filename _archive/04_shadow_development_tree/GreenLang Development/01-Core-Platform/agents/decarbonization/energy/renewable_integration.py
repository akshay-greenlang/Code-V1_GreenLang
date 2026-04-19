# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-002: Renewable Integration Agent

Plans renewable energy project integration including site assessment,
grid interconnection, and economic analysis.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent
from greenlang.agents.decarbonization.energy.schemas import (
    RenewableIntegrationInput,
    RenewableIntegrationOutput,
    RenewableTechnology,
)


logger = logging.getLogger(__name__)


class RenewableIntegrationAgent(DecarbonizationEnergyBaseAgent):
    """
    GL-DECARB-ENE-002: Renewable Integration Agent

    Plans renewable energy project integration including:
    - Site-specific generation estimates
    - Grid interconnection requirements
    - LCOE and financial analysis
    - Curtailment risk assessment
    """

    AGENT_ID = "GL-DECARB-ENE-002"
    AGENT_NAME = "Renewable Integration Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-002",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Renewable energy project integration planning"
    )

    # Capacity factors by technology
    TYPICAL_CAPACITY_FACTORS = {
        "solar_pv": 0.22,
        "solar_csp": 0.28,
        "wind_onshore": 0.35,
        "wind_offshore": 0.45,
        "hydro": 0.40,
        "geothermal": 0.85,
        "biomass": 0.70,
    }

    # Land use (km2/MW)
    LAND_USE_FACTORS = {
        "solar_pv": 0.02,  # 20 hectares/MW
        "wind_onshore": 0.04,  # 40 hectares/MW (with spacing)
        "wind_offshore": 0.0,
    }

    def __init__(self):
        """Initialize Renewable Integration Agent."""
        super().__init__(
            agent_id="GL-DECARB-ENE-002",
            version="1.0.0"
        )

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate renewable project economics."""
        try:
            validated = RenewableIntegrationInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        tech = validated.technology.value
        capacity_mw = validated.target_capacity_mw

        # Get capacity factor
        site_cf = validated.site_characteristics.get("capacity_factor")
        capacity_factor = site_cf if site_cf else self.TYPICAL_CAPACITY_FACTORS.get(tech, 0.25)

        # Calculate generation
        hours_per_year = 8760
        annual_generation_gwh = capacity_mw * capacity_factor * hours_per_year / 1000

        # Get technology costs
        tech_key = "solar_pv" if "solar" in tech else "wind_onshore" if "onshore" in tech else "wind_offshore"
        costs = self.TECHNOLOGY_COSTS.get(tech_key, self.TECHNOLOGY_COSTS["solar_pv"])

        # Capital cost
        capital_cost_million = capacity_mw * costs["capital"] / 1000

        # Calculate LCOE
        lcoe = self.calculate_lcoe(
            capital_cost_per_kw=costs["capital"],
            fixed_om_per_kw_year=costs["om_fixed"],
            variable_om_per_mwh=0,
            capacity_factor=capacity_factor,
            lifetime_years=costs["lifetime"],
        )

        # Curtailment risk based on grid conditions
        curtailment_risk = 0.0
        if capacity_mw > 100:
            curtailment_risk = min(capacity_mw / 1000 * 5, 15)  # Up to 15%

        # Storage requirement for firming
        storage_mwh = capacity_mw * 0.5 * 4  # 50% power, 4 hours

        # Grid upgrades
        grid_upgrades = []
        if capacity_mw > 50:
            grid_upgrades.append(f"Substation upgrade at {validated.grid_connection_point}")
        if capacity_mw > 200:
            grid_upgrades.append("New transmission line segment")

        # Land use
        land_factor = self.LAND_USE_FACTORS.get(tech, 0.02)
        land_use = capacity_mw * land_factor

        # Check land availability
        if validated.land_availability_km2 and land_use > validated.land_availability_km2:
            land_use = validated.land_availability_km2
            capacity_mw = land_use / land_factor

        # Payback calculation
        # Assume $40/MWh PPA price
        ppa_price = 40
        annual_revenue = annual_generation_gwh * 1000 * ppa_price / 1e6
        annual_cost = capacity_mw * costs["om_fixed"] / 1000
        annual_savings = annual_revenue - annual_cost
        payback = self.calculate_payback_period(capital_cost_million, annual_savings)

        # Avoided emissions
        avoided_emissions = self.calculate_avoided_emissions(
            annual_generation_gwh * 1000,
            "grid_average_us"
        )

        return {
            "organization_id": validated.organization_id,
            "technology": tech,
            "installed_capacity_mw": round(capacity_mw, 2),
            "expected_generation_gwh_year": round(annual_generation_gwh, 2),
            "capacity_factor": round(capacity_factor, 3),
            "curtailment_risk_pct": round(curtailment_risk, 1),
            "storage_requirement_mwh": round(storage_mwh, 0),
            "grid_upgrade_requirements": grid_upgrades,
            "lcoe_usd_mwh": lcoe,
            "capital_cost_million_usd": round(capital_cost_million, 2),
            "payback_years": payback,
            "annual_emissions_avoided_tonnes": round(avoided_emissions, 0),
            "land_use_km2": round(land_use, 2),
            "recommended_pathways": ["renewable_integration"],
            "total_abatement_mtco2e": round(avoided_emissions / 1000, 3),
            "total_investment_million_usd": round(capital_cost_million, 2),
            "levelized_abatement_cost_usd_tco2e": round(capital_cost_million * 1e6 / avoided_emissions / 25, 2),
            "confidence_level": 0.80,
            "key_risks": [
                "Interconnection timeline uncertainty",
                "Resource variability",
                "Policy/subsidy changes",
            ],
        }

    async def reason(
        self,
        context: Dict[str, Any],
        session,
        rag_engine,
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """AI-powered renewable integration analysis."""
        economics = self._calculate_economics(context)

        rag_result = await self._rag_retrieve(
            query=f"Renewable energy {context.get('technology', '')} integration best practices",
            rag_engine=rag_engine,
            collections=["renewable_projects", "grid_integration"],
            top_k=5
        )

        return economics
