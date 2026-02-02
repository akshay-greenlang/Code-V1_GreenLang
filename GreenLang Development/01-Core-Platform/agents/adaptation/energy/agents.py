# -*- coding: utf-8 -*-
"""
Adaptation Energy Sector Agents

GL-ADAPT-ENE-001 to GL-ADAPT-ENE-008: Climate adaptation agents for
energy infrastructure resilience.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.agents.adaptation.energy.base import AdaptationEnergyBaseAgent
from greenlang.agents.adaptation.energy.schemas import (
    ClimateHazard,
    ClimateScenario,
    InfrastructureType,
    VulnerabilityLevel,
)


logger = logging.getLogger(__name__)


class ExtremeHeatResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-001: Extreme Heat Resilience Agent

    Assesses and plans for extreme heat impacts on power infrastructure
    including generation derating, transmission losses, and cooling needs.
    """

    AGENT_ID = "GL-ADAPT-ENE-001"
    AGENT_NAME = "Extreme Heat Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-001",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        description="Extreme heat resilience for power infrastructure"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-001", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scenario = inputs.get("climate_scenario", "ssp2_45")
        assets = inputs.get("infrastructure_assets", [])

        total_risk_score = 0
        total_eal = 0
        measures = []

        for asset in assets:
            asset_type = asset.get("type", "power_plant")
            value = asset.get("value_million", 100)

            risk = self.calculate_risk_score(
                "extreme_heat", asset_type, scenario, value
            )
            total_risk_score += risk["risk_score"]
            total_eal += risk["expected_annual_loss_million"]

        avg_risk = total_risk_score / len(assets) if assets else 50

        # Adaptation measures
        if avg_risk > 30:
            measures.append({
                "measure": "Enhanced cooling systems for power plants",
                "cost_million": 25,
                "risk_reduction_pct": 40,
            })
        if avg_risk > 50:
            measures.append({
                "measure": "Transmission line thermal uprating",
                "cost_million": 50,
                "risk_reduction_pct": 30,
            })
            measures.append({
                "measure": "Underground cable conversion (critical segments)",
                "cost_million": 100,
                "risk_reduction_pct": 60,
            })

        total_cost = sum(m["cost_million"] for m in measures)
        avg_reduction = sum(m["risk_reduction_pct"] for m in measures) / len(measures) if measures else 0

        # Calculate benefits
        if measures:
            benefit = self.calculate_adaptation_benefit(
                total_cost, avg_reduction, total_eal
            )
            bcr = benefit["benefit_cost_ratio"]
            avoided = benefit["npv_avoided_losses_million"]
        else:
            bcr = 0
            avoided = 0

        vuln = "critical" if avg_risk > 70 else "high" if avg_risk > 50 else "moderate" if avg_risk > 30 else "low"

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["extreme_heat"],
            "vulnerability_level": vuln,
            "risk_score": round(avg_risk, 1),
            "expected_annual_loss_million": round(total_eal, 4),
            "recommended_measures": measures,
            "total_adaptation_cost_million_usd": round(total_cost, 2),
            "risk_reduction_pct": round(avg_reduction, 1),
            "benefit_cost_ratio": bcr,
            "avoided_losses_million_usd": avoided,
            "confidence_level": 0.75,
            "key_uncertainties": [
                "Regional heat wave frequency projections",
                "Thermal derating curve accuracy",
            ],
        }


class FloodResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-002: Flood Resilience Agent

    Assesses flood risk to energy infrastructure including substations,
    power plants, and underground systems.
    """

    AGENT_ID = "GL-ADAPT-ENE-002"
    AGENT_NAME = "Flood Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-002",
        category=AgentCategory.INSIGHT,
        description="Flood resilience for energy infrastructure"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-002", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scenario = inputs.get("climate_scenario", "ssp2_45")
        assets = inputs.get("infrastructure_assets", [])

        total_eal = 0
        for asset in assets:
            risk = self.calculate_risk_score(
                "flooding", asset.get("type", "substation"),
                scenario, asset.get("value_million", 50)
            )
            total_eal += risk["expected_annual_loss_million"]

        measures = [
            {"measure": "Substation elevation/waterproofing", "cost_million": 15, "risk_reduction_pct": 70},
            {"measure": "Flood barriers and pumping systems", "cost_million": 8, "risk_reduction_pct": 50},
            {"measure": "Underground cable sealing", "cost_million": 5, "risk_reduction_pct": 40},
        ]

        total_cost = sum(m["cost_million"] for m in measures)
        benefit = self.calculate_adaptation_benefit(total_cost, 60, total_eal)

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["flooding"],
            "vulnerability_level": "high" if total_eal > 1 else "moderate",
            "risk_score": min(80, total_eal * 20),
            "expected_annual_loss_million": round(total_eal, 4),
            "recommended_measures": measures,
            "total_adaptation_cost_million_usd": total_cost,
            "risk_reduction_pct": 60,
            "benefit_cost_ratio": benefit["benefit_cost_ratio"],
            "avoided_losses_million_usd": benefit["npv_avoided_losses_million"],
            "confidence_level": 0.70,
            "key_uncertainties": ["Flood frequency projections", "Precipitation intensity changes"],
        }


class WildfireResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-003: Wildfire Resilience Agent

    Assesses wildfire risk to transmission lines, substations, and
    renewable generation facilities.
    """

    AGENT_ID = "GL-ADAPT-ENE-003"
    AGENT_NAME = "Wildfire Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-003",
        category=AgentCategory.INSIGHT,
        description="Wildfire resilience for energy infrastructure"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-003", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scenario = inputs.get("climate_scenario", "ssp2_45")

        measures = [
            {"measure": "Vegetation management intensification", "cost_million": 20, "risk_reduction_pct": 35},
            {"measure": "Covered conductor installation", "cost_million": 80, "risk_reduction_pct": 70},
            {"measure": "Sectionalizing switches for PSPS", "cost_million": 15, "risk_reduction_pct": 25},
            {"measure": "Undergrounding high-risk segments", "cost_million": 200, "risk_reduction_pct": 95},
        ]

        total_cost = sum(m["cost_million"] for m in measures[:3])  # Exclude undergrounding by default

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["wildfire"],
            "vulnerability_level": "critical" if scenario == "ssp5_85" else "high",
            "risk_score": 75 if scenario == "ssp5_85" else 55,
            "expected_annual_loss_million": 5.0,
            "recommended_measures": measures[:3],
            "total_adaptation_cost_million_usd": total_cost,
            "risk_reduction_pct": 50,
            "benefit_cost_ratio": 2.5,
            "avoided_losses_million_usd": 75,
            "confidence_level": 0.65,
            "key_uncertainties": ["Fire weather day projections", "PSPS impact modeling"],
        }


class DroughtResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-004: Drought Resilience Agent

    Assesses drought impacts on hydropower and thermal cooling.
    """

    AGENT_ID = "GL-ADAPT-ENE-004"
    AGENT_NAME = "Drought Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-004",
        category=AgentCategory.INSIGHT,
        description="Drought resilience for hydropower and cooling"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-004", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        measures = [
            {"measure": "Dry/hybrid cooling conversion", "cost_million": 50, "risk_reduction_pct": 60},
            {"measure": "Alternative water supply development", "cost_million": 30, "risk_reduction_pct": 40},
            {"measure": "Reservoir optimization and storage", "cost_million": 100, "risk_reduction_pct": 30},
        ]

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["drought"],
            "vulnerability_level": "high",
            "risk_score": 60,
            "expected_annual_loss_million": 3.0,
            "recommended_measures": measures,
            "total_adaptation_cost_million_usd": 180,
            "risk_reduction_pct": 45,
            "benefit_cost_ratio": 1.8,
            "avoided_losses_million_usd": 40,
            "confidence_level": 0.70,
            "key_uncertainties": ["Precipitation pattern changes", "Competing water uses"],
        }


class HurricaneResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-005: Hurricane Resilience Agent

    Assesses hurricane and severe storm risk to coastal infrastructure.
    """

    AGENT_ID = "GL-ADAPT-ENE-005"
    AGENT_NAME = "Hurricane Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-005",
        category=AgentCategory.INSIGHT,
        description="Hurricane resilience for coastal energy infrastructure"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-005", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        measures = [
            {"measure": "Distribution hardening (steel poles)", "cost_million": 150, "risk_reduction_pct": 50},
            {"measure": "Underground conversion (critical areas)", "cost_million": 300, "risk_reduction_pct": 85},
            {"measure": "Substation storm hardening", "cost_million": 40, "risk_reduction_pct": 60},
            {"measure": "Mobile response assets", "cost_million": 25, "risk_reduction_pct": 20},
        ]

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["hurricane", "storm_surge"],
            "vulnerability_level": "critical",
            "risk_score": 80,
            "expected_annual_loss_million": 25.0,
            "recommended_measures": measures,
            "total_adaptation_cost_million_usd": 515,
            "risk_reduction_pct": 55,
            "benefit_cost_ratio": 3.2,
            "avoided_losses_million_usd": 410,
            "confidence_level": 0.75,
            "key_uncertainties": ["Hurricane intensity projections", "Storm surge modeling"],
        }


class SeaLevelRiseResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-006: Sea Level Rise Resilience Agent

    Assesses sea level rise risk to coastal energy facilities.
    """

    AGENT_ID = "GL-ADAPT-ENE-006"
    AGENT_NAME = "Sea Level Rise Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-006",
        category=AgentCategory.INSIGHT,
        description="Sea level rise resilience planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-006", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        planning_years = inputs.get("planning_horizon_years", 30)

        measures = [
            {"measure": "Facility elevation above future flood level", "cost_million": 100, "risk_reduction_pct": 80},
            {"measure": "Seawall/barrier construction", "cost_million": 200, "risk_reduction_pct": 70},
            {"measure": "Planned asset relocation", "cost_million": 500, "risk_reduction_pct": 95},
        ]

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["sea_level_rise"],
            "vulnerability_level": "high",
            "risk_score": 65,
            "expected_annual_loss_million": 8.0,
            "recommended_measures": measures[:2],  # Relocation is long-term option
            "total_adaptation_cost_million_usd": 300,
            "risk_reduction_pct": 75,
            "benefit_cost_ratio": 2.0,
            "avoided_losses_million_usd": 180,
            "confidence_level": 0.80,
            "key_uncertainties": ["Ice sheet dynamics", "Regional sea level variation"],
        }


class IceStormResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-007: Ice Storm Resilience Agent

    Assesses ice storm risk to transmission and distribution systems.
    """

    AGENT_ID = "GL-ADAPT-ENE-007"
    AGENT_NAME = "Ice Storm Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-007",
        category=AgentCategory.INSIGHT,
        description="Ice storm resilience for power systems"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-007", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        measures = [
            {"measure": "De-icing conductor installation", "cost_million": 40, "risk_reduction_pct": 50},
            {"measure": "Increased design ice loading standards", "cost_million": 80, "risk_reduction_pct": 40},
            {"measure": "Strategic tree trimming", "cost_million": 15, "risk_reduction_pct": 30},
        ]

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": ["ice_storm", "extreme_cold"],
            "vulnerability_level": "moderate",
            "risk_score": 45,
            "expected_annual_loss_million": 2.0,
            "recommended_measures": measures,
            "total_adaptation_cost_million_usd": 135,
            "risk_reduction_pct": 50,
            "benefit_cost_ratio": 1.5,
            "avoided_losses_million_usd": 30,
            "confidence_level": 0.70,
            "key_uncertainties": ["Freezing rain frequency", "Mixed precipitation events"],
        }


class ComprehensiveClimateResilienceAgent(AdaptationEnergyBaseAgent):
    """
    GL-ADAPT-ENE-008: Comprehensive Climate Resilience Agent

    Integrates all hazard assessments for portfolio-level resilience planning.
    """

    AGENT_ID = "GL-ADAPT-ENE-008"
    AGENT_NAME = "Comprehensive Climate Resilience Agent"
    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="GL-ADAPT-ENE-008",
        category=AgentCategory.INSIGHT,
        description="Comprehensive climate resilience portfolio planning"
    )

    def __init__(self):
        super().__init__(agent_id="GL-ADAPT-ENE-008", version="1.0.0")

    def _calculate_risk_assessment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        scenario = inputs.get("climate_scenario", "ssp2_45")
        assets = inputs.get("infrastructure_assets", [])

        # Assess all hazards
        hazards = ["extreme_heat", "flooding", "wildfire", "drought", "hurricane"]
        total_risk = 0
        total_eal = 0
        hazard_risks = {}

        for hazard in hazards:
            hazard_score = 0
            hazard_eal = 0
            for asset in assets if assets else [{"type": "power_plant", "value_million": 100}]:
                risk = self.calculate_risk_score(
                    hazard, asset.get("type", "power_plant"),
                    scenario, asset.get("value_million", 100)
                )
                hazard_score += risk["risk_score"]
                hazard_eal += risk["expected_annual_loss_million"]

            hazard_risks[hazard] = {
                "risk_score": round(hazard_score / max(len(assets), 1), 1),
                "eal_million": round(hazard_eal, 4),
            }
            total_risk += hazard_score
            total_eal += hazard_eal

        avg_risk = total_risk / (len(hazards) * max(len(assets), 1))

        # Priority measures across hazards
        measures = [
            {"measure": "Comprehensive risk monitoring system", "cost_million": 10, "risk_reduction_pct": 15},
            {"measure": "Emergency response enhancement", "cost_million": 20, "risk_reduction_pct": 20},
            {"measure": "Asset condition monitoring", "cost_million": 15, "risk_reduction_pct": 10},
            {"measure": "Grid resilience investments", "cost_million": 100, "risk_reduction_pct": 40},
        ]

        total_cost = sum(m["cost_million"] for m in measures)
        benefit = self.calculate_adaptation_benefit(total_cost, 40, total_eal)

        vuln = "critical" if avg_risk > 60 else "high" if avg_risk > 40 else "moderate"

        return {
            "organization_id": inputs.get("organization_id", ""),
            "hazards_assessed": hazards,
            "hazard_risk_breakdown": hazard_risks,
            "vulnerability_level": vuln,
            "risk_score": round(avg_risk, 1),
            "expected_annual_loss_million": round(total_eal, 4),
            "recommended_measures": measures,
            "total_adaptation_cost_million_usd": total_cost,
            "risk_reduction_pct": 40,
            "benefit_cost_ratio": benefit["benefit_cost_ratio"],
            "avoided_losses_million_usd": benefit["npv_avoided_losses_million"],
            "confidence_level": 0.70,
            "key_uncertainties": [
                "Climate model ensemble spread",
                "Compound event interactions",
                "Cascading failure dynamics",
            ],
        }
