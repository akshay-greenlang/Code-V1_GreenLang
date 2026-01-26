# -*- coding: utf-8 -*-
"""
GL-DECARB-ENE-003: Storage Optimization Agent

Optimizes energy storage sizing, technology selection, and value stacking
for grid-scale and behind-the-meter applications.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.decarbonization.energy.base import DecarbonizationEnergyBaseAgent
from greenlang.agents.decarbonization.energy.schemas import (
    StorageOptimizationInput,
    StorageOptimizationOutput,
    StorageApplication,
)


logger = logging.getLogger(__name__)


class StorageOptimizationAgent(DecarbonizationEnergyBaseAgent):
    """
    GL-DECARB-ENE-003: Storage Optimization Agent

    Optimizes energy storage deployment including:
    - Optimal power and energy sizing
    - Technology selection
    - Revenue stream stacking
    - Economic analysis
    """

    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="GL-DECARB-ENE-003",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        description="Energy storage optimization and sizing"
    )

    # Technology characteristics
    STORAGE_TECHNOLOGIES = {
        "li_ion_lfp": {
            "capital_per_kwh": 180,
            "power_per_kw": 150,
            "efficiency": 0.88,
            "cycles": 6000,
            "lifetime": 15,
            "best_for": ["frequency_regulation", "peak_shaving"],
        },
        "li_ion_nmc": {
            "capital_per_kwh": 200,
            "power_per_kw": 140,
            "efficiency": 0.90,
            "cycles": 4000,
            "lifetime": 12,
            "best_for": ["peak_shaving", "load_shifting"],
        },
        "flow_vanadium": {
            "capital_per_kwh": 400,
            "power_per_kw": 600,
            "efficiency": 0.75,
            "cycles": 15000,
            "lifetime": 25,
            "best_for": ["renewable_firming", "load_shifting"],
        },
    }

    # Revenue potential by application ($/kW-year)
    REVENUE_POTENTIAL = {
        "frequency_regulation": 100,
        "spinning_reserve": 50,
        "peak_shaving": 80,
        "load_shifting": 60,
        "renewable_firming": 40,
        "transmission_deferral": 120,
    }

    def __init__(self):
        """Initialize Storage Optimization Agent."""
        super().__init__(
            agent_id="GL-DECARB-ENE-003",
            version="1.0.0"
        )

    def _calculate_economics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate storage optimization economics."""
        try:
            validated = StorageOptimizationInput(**inputs)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")

        peak_demand = validated.peak_demand_mw
        renewable_pct = validated.renewable_penetration_pct
        applications = [a.value for a in validated.applications]

        # Size storage based on applications
        power_mw = 0
        duration_hours = 4

        if "frequency_regulation" in applications:
            power_mw += peak_demand * 0.02  # 2% of peak
            duration_hours = max(duration_hours, 1)

        if "peak_shaving" in applications:
            power_mw += peak_demand * 0.10  # 10% of peak
            duration_hours = max(duration_hours, 4)

        if "renewable_firming" in applications:
            power_mw += peak_demand * renewable_pct / 100 * 0.20
            duration_hours = max(duration_hours, 6)

        if "load_shifting" in applications:
            power_mw += peak_demand * 0.05
            duration_hours = max(duration_hours, 4)

        energy_mwh = power_mw * duration_hours

        # Select technology based on duration
        if duration_hours <= 2:
            tech = "li_ion_nmc"
            tech_rationale = "Short duration favors high power density Li-ion NMC"
        elif duration_hours <= 4:
            tech = "li_ion_lfp"
            tech_rationale = "Medium duration favors LFP for better cycle life"
        else:
            tech = "flow_vanadium"
            tech_rationale = "Long duration favors flow batteries for lower energy cost"

        tech_specs = self.STORAGE_TECHNOLOGIES[tech]

        # Calculate capital cost
        capital_cost = (
            power_mw * 1000 * tech_specs["power_per_kw"] +
            energy_mwh * 1000 * tech_specs["capital_per_kwh"]
        )

        # Calculate revenues
        annual_revenues = {}
        total_revenue = 0
        for app in applications:
            if app in self.REVENUE_POTENTIAL:
                rev = power_mw * self.REVENUE_POTENTIAL[app]
                annual_revenues[app] = round(rev, 0)
                total_revenue += rev

        # LCOE of storage
        annual_cycles = 365  # One cycle per day
        annual_discharge = energy_mwh * annual_cycles * tech_specs["efficiency"]
        annual_fixed_om = power_mw * 20 * 1000  # $20/kW-year

        lcoe = (capital_cost / tech_specs["lifetime"] + annual_fixed_om) / annual_discharge

        # NPV calculation
        annual_cash_flows = [total_revenue * 1000] * tech_specs["lifetime"]
        npv = self.calculate_npv(capital_cost, annual_cash_flows)
        irr = self.calculate_irr(capital_cost, annual_cash_flows)

        return {
            "organization_id": validated.organization_id,
            "recommended_power_mw": round(power_mw, 2),
            "recommended_energy_mwh": round(energy_mwh, 2),
            "recommended_duration_hours": duration_hours,
            "recommended_technology": tech,
            "technology_rationale": tech_rationale,
            "annual_revenue_streams": annual_revenues,
            "total_annual_value_usd": round(total_revenue * 1000, 0),
            "capital_cost_usd": round(capital_cost, 0),
            "lcoe_usd_mwh": round(lcoe, 2),
            "npv_million_usd": round(npv / 1e6, 2),
            "irr_pct": round(irr * 100, 1) if irr else 0,
            "recommended_pathways": ["energy_storage"],
            "total_abatement_mtco2e": round(annual_discharge * 0.4 / 1000, 3),
            "total_investment_million_usd": round(capital_cost / 1e6, 2),
            "levelized_abatement_cost_usd_tco2e": 0,  # Storage enables other abatement
            "confidence_level": 0.75,
            "key_risks": [
                "Technology cost decline uncertainty",
                "Market revenue volatility",
                "Regulatory changes for storage",
            ],
        }

    async def reason(
        self,
        context: Dict[str, Any],
        session,
        rag_engine,
        tools: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """AI-powered storage optimization analysis."""
        return self._calculate_economics(context)
