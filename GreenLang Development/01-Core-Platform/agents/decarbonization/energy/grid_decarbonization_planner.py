# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-001: Grid Decarbonization Planner Agent

Plans grid-level decarbonization pathways including generation mix
optimization, renewable deployment, and fossil retirement scheduling.

Example:
    >>> agent = GridDecarbonizationPlannerAgent()
    >>> result = agent.process({
    ...     "organization_id": "UTILITY-001",
    ...     "region": "western_us",
    ...     "baseline_year": 2024,
    ...     "target_year": 2035,
    ...     "current_emissions_mtco2e": 50,
    ...     "target_emissions_mtco2e": 10,
    ...     "current_generation_mix": {"coal": 30, "gas": 40, "solar": 15, "wind": 10, "hydro": 5},
    ...     "peak_demand_gw": 25,
    ...     "annual_demand_twh": 150,
    ... })
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent
from greenlang.agents.decarbonization.energy.schemas import (
    GridDecarbonizationInput,
    GridDecarbonizationOutput,
)


logger = logging.getLogger(__name__)


class GridDecarbonizationPlannerAgent(DecarbonizationEnergyBaseAgent):
    """
    GL-DECARB-ENE-001: Grid Decarbonization Planner

    Plans comprehensive grid decarbonization including:
    - Optimal renewable capacity additions
    - Fossil plant retirement schedules
    - Storage requirements
    - Transmission upgrades

    Uses RECOMMENDATION PATH pattern with AI for strategic analysis
    and deterministic calculations for financials.
    """

    AGENT_ID = "GL-DECARB-ENE-001"
    AGENT_NAME = "Grid Decarbonization Planner Agent"
    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-001",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Grid-level decarbonization pathway planning"
    )

    def __init__(self):
        """Initialize Grid Decarbonization Planner."""
        super().__init__(
            agent_id="GL-DECARB-ENE-001",
            version="1.0.0"
        )

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate grid decarbonization economics.

        Deterministic calculations for capacity planning and costs.
        """
        # Parse inputs
        try:
            validated = GridDecarbonizationInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        current_mix = validated.current_generation_mix
        peak_demand = validated.peak_demand_gw
        annual_demand = validated.annual_demand_twh
        years_to_target = validated.target_year - validated.baseline_year

        # Calculate required emission reduction
        emission_reduction_required = (
            validated.current_emissions_mtco2e -
            validated.target_emissions_mtco2e
        )

        # Estimate fossil generation to retire (approximate)
        fossil_share = current_mix.get("coal", 0) + current_mix.get("gas", 0)
        fossil_generation_twh = annual_demand * fossil_share / 100

        # Calculate renewable capacity needed
        # Assume 25% capacity factor average for solar/wind mix
        avg_capacity_factor = 0.28
        hours_per_year = 8760
        renewable_generation_needed_twh = fossil_generation_twh * 0.8  # 80% replacement

        renewable_capacity_gw = (
            renewable_generation_needed_twh * 1000 /
            (avg_capacity_factor * hours_per_year)
        )

        # Split between solar and wind (60/40)
        solar_additions = renewable_capacity_gw * 0.6
        wind_additions = renewable_capacity_gw * 0.4

        # Storage requirement (4-hour duration, ~20% of renewable capacity)
        storage_power_gw = renewable_capacity_gw * 0.20
        storage_energy_gwh = storage_power_gw * 4

        # Transmission additions (~10% of new renewable capacity)
        transmission_additions = renewable_capacity_gw * 0.10

        # Calculate capital investment
        solar_cost = solar_additions * 1000 * self.TECHNOLOGY_COSTS["solar_pv"]["capital"] / 1e9
        wind_cost = wind_additions * 1000 * self.TECHNOLOGY_COSTS["wind_onshore"]["capital"] / 1e9
        storage_cost = storage_energy_gwh * 1000 * self.TECHNOLOGY_COSTS["battery_4h"]["capital"] / 1e9
        transmission_cost = transmission_additions * 1000 * 500 / 1e9  # $500/kW for transmission

        total_capital = solar_cost + wind_cost + storage_cost + transmission_cost

        # Calculate LCOE for new build
        solar_lcoe = self.calculate_lcoe(
            capital_cost_per_kw=self.TECHNOLOGY_COSTS["solar_pv"]["capital"],
            fixed_om_per_kw_year=self.TECHNOLOGY_COSTS["solar_pv"]["om_fixed"],
            variable_om_per_mwh=0,
            capacity_factor=0.25,
            lifetime_years=30,
        )

        wind_lcoe = self.calculate_lcoe(
            capital_cost_per_kw=self.TECHNOLOGY_COSTS["wind_onshore"]["capital"],
            fixed_om_per_kw_year=self.TECHNOLOGY_COSTS["wind_onshore"]["om_fixed"],
            variable_om_per_mwh=0,
            capacity_factor=0.35,
            lifetime_years=25,
        )

        # Operating savings from fuel displacement
        # Assume $50/MWh average fuel + O&M for displaced fossil
        annual_fuel_savings = renewable_generation_needed_twh * 1e6 * 50 / 1e6

        # Target generation mix
        renewable_pct = (
            current_mix.get("solar", 0) + current_mix.get("wind", 0) +
            current_mix.get("hydro", 0) + solar_additions/peak_demand*100 +
            wind_additions/peak_demand*100
        )
        clean_energy_pct = renewable_pct + current_mix.get("nuclear", 0)

        # Fossil retirements
        coal_retirement = current_mix.get("coal", 0) * peak_demand / 100
        gas_retirement = current_mix.get("gas", 0) * peak_demand / 100 * 0.5  # Partial gas

        # Abatement cost
        abatement_cost = self.calculate_abatement_cost(
            capital_cost=total_capital * 1e9,
            annual_operating_cost=0,  # Net zero with fuel savings
            annual_emissions_reduced_tonnes=emission_reduction_required * 1e6,
            lifetime_years=25,
        )

        # Build output
        target_mix = {
            "solar": min(current_mix.get("solar", 0) + solar_additions/peak_demand*100, 50),
            "wind": min(current_mix.get("wind", 0) + wind_additions/peak_demand*100, 30),
            "hydro": current_mix.get("hydro", 0),
            "nuclear": current_mix.get("nuclear", 0),
            "gas": max(current_mix.get("gas", 0) - 20, 5),  # Reduced gas
            "coal": 0,  # Coal retired
            "storage": storage_power_gw/peak_demand*100,
        }

        return {
            "organization_id": validated.organization_id,
            "target_generation_mix": target_mix,
            "renewable_percentage": min(renewable_pct, 80),
            "clean_energy_percentage": min(clean_energy_pct, 90),
            "renewable_additions_gw": {
                "solar": round(solar_additions, 2),
                "wind": round(wind_additions, 2),
            },
            "storage_additions_gwh": round(storage_energy_gwh, 2),
            "transmission_additions_gw": round(transmission_additions, 2),
            "fossil_retirements_gw": {
                "coal": round(coal_retirement, 2),
                "gas": round(gas_retirement, 2),
            },
            "capital_investment_billion_usd": round(total_capital, 2),
            "annual_operating_savings_million_usd": round(annual_fuel_savings, 0),
            "recommended_pathways": ["renewable_integration", "energy_storage", "grid_modernization"],
            "total_abatement_mtco2e": round(emission_reduction_required, 2),
            "total_investment_million_usd": round(total_capital * 1000, 0),
            "levelized_abatement_cost_usd_tco2e": round(abatement_cost, 2),
            "milestones": [
                {"year": validated.baseline_year + 3, "action": "Coal phase-out 50%"},
                {"year": validated.baseline_year + 5, "action": "Coal phase-out complete"},
                {"year": validated.target_year, "action": "Target achieved"},
            ],
            "confidence_level": 0.75,
            "key_risks": [
                "Renewable interconnection delays",
                "Supply chain constraints",
                "Regulatory uncertainty",
            ],
        }

    async def reason(
        self,
        context: Dict[str, Any],
        session,
        rag_engine,
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        AI-powered grid decarbonization reasoning.

        Uses RAG for case studies and LLM for scenario analysis.
        """
        # Get deterministic economics first
        economics = self._calculate_economics(context)

        # RAG retrieval for case studies
        rag_result = await self._rag_retrieve(
            query=f"Grid decarbonization {context.get('region', '')} renewable integration",
            rag_engine=rag_engine,
            collections=["case_studies", "grid_planning"],
            top_k=5
        )

        rag_context = self._format_rag_results(rag_result)

        # LLM reasoning for strategic recommendations
        prompt = f"""Analyze this grid decarbonization plan and provide strategic recommendations:

Current Situation:
- Region: {context.get('region', 'Unknown')}
- Current emissions: {context.get('current_emissions_mtco2e', 0)} Mt CO2e
- Target emissions: {context.get('target_emissions_mtco2e', 0)} Mt CO2e
- Timeline: {context.get('baseline_year', 2024)} to {context.get('target_year', 2035)}

Economic Analysis:
- Capital investment required: ${economics['capital_investment_billion_usd']} billion
- Annual savings: ${economics['annual_operating_savings_million_usd']} million
- Renewable additions: {economics['renewable_additions_gw']}

Relevant Case Studies:
{rag_context}

Provide:
1. Key success factors from similar projects
2. Technology sequencing recommendations
3. Policy enablers needed
4. Risk mitigation strategies
"""

        response = await session.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        # Combine deterministic economics with AI insights
        economics["ai_recommendations"] = response.text
        economics["rag_sources_consulted"] = len(rag_result.chunks) if rag_result else 0

        return economics
