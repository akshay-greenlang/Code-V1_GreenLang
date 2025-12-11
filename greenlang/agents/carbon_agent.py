# -*- coding: utf-8 -*-
"""
CarbonAgent - AI-Native Carbon Footprint Calculator

This agent aggregates emissions and calculates total carbon footprint with
FULL LLM INTELLIGENCE capabilities via IntelligenceMixin.

Intelligence Level: STANDARD
Capabilities: Explanations, Recommendations, Anomaly Detection

Regulatory Context: GHG Protocol, CSRD ESRS E1
"""
from typing import Any, Dict, List
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.intelligence_mixin import IntelligenceMixin
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel
from templates.agent_monitoring import OperationalMonitoringMixin


class CarbonAgent(IntelligenceMixin, OperationalMonitoringMixin, BaseAgent):
    """
    AI-Native Carbon Footprint Calculator Agent.

    This agent aggregates emissions from multiple sources and calculates
    total carbon footprint with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC CALCULATIONS (Zero-Hallucination):
       - Total CO2e aggregation
       - Percentage breakdown by source
       - Intensity calculations

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of footprint
       - Reduction recommendations with ROI
       - Anomaly detection for unusual patterns
    """

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="CarbonAgent",
                description="AI-native carbon footprint calculator with LLM intelligence",
            )
        super().__init__(config)
        self.setup_monitoring(agent_name="carbon_agent")
        # Intelligence auto-initializes on first use

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return IntelligenceLevel.STANDARD

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=False,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "emissions" in input_data and isinstance(input_data["emissions"], list)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        with self.track_execution(input_data) as tracker:
            emissions_list = input_data["emissions"]

            if not emissions_list:
                return AgentResult(
                    success=True,
                    data={
                        "total_co2e_kg": 0,
                        "total_co2e_tons": 0,
                        "emissions_breakdown": [],
                        "summary": "No emissions data provided",
                        "explanation": "No emissions data was provided for analysis.",
                        "recommendations": [],
                    },
                )

            # =================================================================
            # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
            # =================================================================
            total_co2e_kg = 0
            emissions_breakdown = []

            for emission in emissions_list:
                if isinstance(emission, dict):
                    co2e = emission.get("co2e_emissions_kg", 0)
                    total_co2e_kg += co2e

                    breakdown_item = {
                        "source": emission.get("fuel_type", "Unknown"),
                        "co2e_kg": co2e,
                        "co2e_tons": co2e / 1000,
                        "percentage": 0,
                    }
                    emissions_breakdown.append(breakdown_item)

            for item in emissions_breakdown:
                if total_co2e_kg > 0:
                    item["percentage"] = round((item["co2e_kg"] / total_co2e_kg) * 100, 2)

            total_co2e_tons = total_co2e_kg / 1000
            summary = self._generate_summary(total_co2e_tons, emissions_breakdown)

            # Build deterministic result
            result_data = {
                "total_co2e_kg": round(total_co2e_kg, 2),
                "total_co2e_tons": round(total_co2e_tons, 3),
                "emissions_breakdown": emissions_breakdown,
                "summary": summary,
                "carbon_intensity": self._calculate_intensity(input_data, total_co2e_kg),
            }

            # =================================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =================================================================
            calculation_steps = [
                f"Received {len(emissions_list)} emission sources",
                f"Aggregated total: {total_co2e_kg:.2f} kg CO2e",
                f"Converted to tons: {total_co2e_tons:.3f} metric tons",
                "Calculated percentage breakdown by source"
            ]

            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=result_data,
                calculation_steps=calculation_steps
            )
            result_data["explanation"] = explanation

            # =================================================================
            # STEP 3: AI-POWERED RECOMMENDATIONS
            # =================================================================
            top_sources = [item["source"] for item in sorted(
                emissions_breakdown, key=lambda x: x["co2e_kg"], reverse=True
            )[:3]]

            recommendations = self.generate_recommendations(
                analysis={
                    **result_data,
                    "top_sources": top_sources,
                    "reduction_potential": f"{total_co2e_tons * 0.2:.1f}-{total_co2e_tons * 0.3:.1f} tons"
                },
                max_recommendations=5,
                focus_areas=["efficiency", "fuel_switching", "renewable_energy"]
            )
            result_data["recommendations"] = recommendations

            # =================================================================
            # STEP 4: ADD INTELLIGENCE METADATA
            # =================================================================
            result_data["intelligence_level"] = self.get_intelligence_level().value

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "CarbonAgent",
                    "num_sources": len(emissions_breakdown),
                    "intelligence_metrics": self.get_intelligence_metrics(),
                    "regulatory_context": "GHG Protocol, CSRD ESRS E1"
                },
            )

    def _generate_summary(self, total_tons: float, breakdown: List[Dict]) -> str:
        """Generate deterministic summary (not AI-powered)."""
        if total_tons == 0:
            return "No carbon emissions"

        summary = f"Total carbon footprint: {total_tons:.3f} metric tons CO2e\n"

        if breakdown:
            summary += "Breakdown by source:\n"
            for item in sorted(breakdown, key=lambda x: x["co2e_kg"], reverse=True):
                summary += f"  - {item['source']}: {item['co2e_tons']:.3f} tons ({item['percentage']}%)\n"

        return summary.strip()

    def _calculate_intensity(
        self, input_data: Dict[str, Any], total_co2e: float
    ) -> Dict[str, float]:
        """Calculate carbon intensity metrics."""
        intensity = {}

        if "building_area" in input_data:
            area = input_data["building_area"]
            intensity["per_sqft"] = round(total_co2e / area, 4) if area > 0 else 0

        if "occupancy" in input_data:
            occupancy = input_data["occupancy"]
            intensity["per_person"] = round(total_co2e / occupancy, 4) if occupancy > 0 else 0

        return intensity
