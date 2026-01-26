# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-010: Just Transition Planner Agent

Plans workforce transition and community support for energy sector
decarbonization with social equity considerations.
"""

from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent


logger = logging.getLogger(__name__)


class JustTransitionPlannerAgent(DecarbonizationEnergyBaseAgent):
    """GL-DECARB-ENE-010: Just Transition Planner Agent"""

    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-010",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Just transition and workforce planning"
    )

    # Job ratios (new jobs per displaced job by sector)
    JOB_MULTIPLIERS = {
        "coal_mining": 0.8,
        "coal_power": 1.5,
        "oil_gas": 1.2,
        "gas_power": 1.8,
    }

    # Retraining costs per worker
    RETRAINING_COST = 25000  # $25k average

    def __init__(self):
        super().__init__(agent_id="GL-DECARB-ENE-010", version="1.0.0")

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        communities = inputs.get("affected_communities", [])
        fossil_employment = inputs.get("fossil_fuel_employment", 5000)
        retirements = inputs.get("planned_retirements", [])
        retraining_budget = inputs.get("retraining_budget_million_usd")

        # Analyze affected communities
        total_jobs_at_risk = fossil_employment

        # Calculate new clean energy jobs
        # Assume 1.5 clean energy jobs per fossil job
        new_jobs = int(total_jobs_at_risk * 1.5)
        net_job_impact = new_jobs - total_jobs_at_risk

        # Retraining programs
        retraining_programs = [
            {
                "program": "Solar installation certification",
                "duration_months": 6,
                "capacity": int(total_jobs_at_risk * 0.3),
                "cost_per_participant": 15000,
            },
            {
                "program": "Wind turbine technician",
                "duration_months": 12,
                "capacity": int(total_jobs_at_risk * 0.2),
                "cost_per_participant": 25000,
            },
            {
                "program": "Battery storage specialist",
                "duration_months": 9,
                "capacity": int(total_jobs_at_risk * 0.15),
                "cost_per_participant": 20000,
            },
            {
                "program": "Grid modernization electrician",
                "duration_months": 18,
                "capacity": int(total_jobs_at_risk * 0.15),
                "cost_per_participant": 30000,
            },
        ]

        # Calculate total retraining cost
        retraining_cost = sum(
            p["capacity"] * p["cost_per_participant"]
            for p in retraining_programs
        )

        # Economic diversification
        diversification = [
            {
                "initiative": "Clean energy manufacturing hub",
                "jobs_created": int(total_jobs_at_risk * 0.2),
                "investment_million": 150,
            },
            {
                "initiative": "Renewable energy services cluster",
                "jobs_created": int(total_jobs_at_risk * 0.15),
                "investment_million": 50,
            },
            {
                "initiative": "Green technology R&D center",
                "jobs_created": int(total_jobs_at_risk * 0.05),
                "investment_million": 100,
            },
        ]

        diversification_cost = sum(d["investment_million"] for d in diversification)

        # Community investments
        community_investments = [
            {
                "investment": "Community transition fund",
                "amount_million": 50,
                "purpose": "Economic development grants",
            },
            {
                "investment": "Healthcare continuity",
                "amount_million": 25,
                "purpose": "Bridge benefits for displaced workers",
            },
            {
                "investment": "Education and training infrastructure",
                "amount_million": 30,
                "purpose": "Community college expansion",
            },
        ]

        community_cost = sum(c["amount_million"] for c in community_investments)

        # Timeline
        transition_years = 10

        # Total investment
        total_investment = (
            retraining_cost / 1e6 +
            diversification_cost +
            community_cost
        )

        return {
            "organization_id": inputs.get("organization_id", ""),
            "jobs_at_risk": total_jobs_at_risk,
            "new_clean_energy_jobs": new_jobs,
            "net_job_impact": net_job_impact,
            "retraining_programs": retraining_programs,
            "economic_diversification": diversification,
            "community_investments": community_investments,
            "transition_timeline_years": transition_years,
            "total_transition_investment_million_usd": round(total_investment, 2),
            "recommended_pathways": ["just_transition"],
            "total_abatement_mtco2e": 0,  # Social investment
            "total_investment_million_usd": round(total_investment, 2),
            "levelized_abatement_cost_usd_tco2e": 0,
            "confidence_level": 0.70,
            "key_risks": [
                "Worker participation in retraining",
                "New job location vs affected communities",
                "Political continuity of transition programs",
                "Skills mismatch between old and new jobs",
            ],
        }

    async def reason(self, context, session, rag_engine, tools=None):
        return self._calculate_economics(context)
