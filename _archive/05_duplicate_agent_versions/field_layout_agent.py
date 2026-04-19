# -*- coding: utf-8 -*-
"""
FieldLayoutAgent - AI-Native Solar Collector Field Sizing

This agent performs high-level sizing of solar collector fields with
FULL LLM INTELLIGENCE capabilities via IntelligenceMixin.

Intelligence Level: STANDARD
Capabilities: Explanations, Recommendations

Regulatory Context: IEC 62862, ISO 9806
"""

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.intelligence_mixin import IntelligenceMixin
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

# Constants - these are typical values for a good solar location
# In a real model, this would be calculated from TMY data.
ANNUAL_DNI_KWH_PER_M2 = 2000  # Annual Direct Normal Irradiance in kWh/m^2/year
SYSTEM_EFFICIENCY_HEURISTIC = 0.50  # Overall system efficiency (optical, thermal, etc.)


class FieldLayoutAgent(IntelligenceMixin, BaseAgent):
    """
    AI-Native Solar Collector Field Sizing Agent.

    This agent sizes solar collector fields with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC CALCULATIONS (Zero-Hallucination):
       - Required aperture area calculation
       - Collector count determination
       - Land area estimation

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of sizing decisions
       - Layout optimization recommendations
       - Technology selection guidance
    """

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="FieldLayoutAgent",
                description="AI-native solar collector field sizing with LLM intelligence",
            )
        super().__init__(config)
        # Intelligence auto-initializes on first use

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return IntelligenceLevel.STANDARD

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=False,
            uses_rag=False,
            uses_tools=False
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return (
            input_data.get("total_annual_demand_gwh") is not None
            and input_data.get("solar_config") is not None
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Size solar collector field with AI-powered insights."""
        try:
            total_annual_demand_gwh = input_data.get("total_annual_demand_gwh")
            solar_config = input_data.get("solar_config")

            if total_annual_demand_gwh is None or not solar_config:
                return AgentResult(
                    success=False,
                    error="total_annual_demand_gwh and solar_config must be provided",
                )

            # =================================================================
            # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
            # =================================================================
            target_solar_fraction = 0.50  # For v1, we hardcode a 50% target

            # Calculate the total solar energy we need to generate in a year
            required_solar_energy_kwh = (
                total_annual_demand_gwh * 1e6 * target_solar_fraction
            )

            # Calculate the energy generated per square meter of collector
            annual_yield_per_m2 = ANNUAL_DNI_KWH_PER_M2 * SYSTEM_EFFICIENCY_HEURISTIC

            # Calculate the required aperture area
            required_aperture_area_m2 = required_solar_energy_kwh / annual_yield_per_m2

            # Get collector-specific data (this would come from a config file)
            collector_aperture_area = 50  # m^2 per collector for a typical model

            num_collectors = round(required_aperture_area_m2 / collector_aperture_area)

            # Recalculate the actual area based on the integer number of collectors
            actual_aperture_area_m2 = num_collectors * collector_aperture_area

            # Estimate land area
            land_area_per_aperture_area = solar_config["row_spacing_factor"]
            required_land_area_m2 = (
                actual_aperture_area_m2 * land_area_per_aperture_area
            )
            required_land_area_acres = required_land_area_m2 / 4047

            # Build deterministic result
            result_data = {
                "required_aperture_area_m2": actual_aperture_area_m2,
                "num_collectors": num_collectors,
                "required_land_area_m2": round(required_land_area_m2, 0),
                "required_land_area_acres": round(required_land_area_acres, 2),
                "target_solar_fraction": target_solar_fraction,
                "annual_yield_per_m2": annual_yield_per_m2,
            }

            # =================================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =================================================================
            calculation_steps = [
                f"Annual demand: {total_annual_demand_gwh} GWh",
                f"Target solar fraction: {target_solar_fraction * 100}%",
                f"Required solar energy: {required_solar_energy_kwh:,.0f} kWh/year",
                f"Annual yield per m2: {annual_yield_per_m2} kWh (at {SYSTEM_EFFICIENCY_HEURISTIC * 100}% efficiency)",
                f"Required aperture area: {required_aperture_area_m2:,.0f} m2",
                f"Number of collectors (50 m2 each): {num_collectors}",
                f"Actual aperture area: {actual_aperture_area_m2:,.0f} m2",
                f"Row spacing factor: {land_area_per_aperture_area}",
                f"Required land area: {required_land_area_m2:,.0f} m2 ({required_land_area_acres:.1f} acres)"
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
            recommendations = self.generate_recommendations(
                analysis={
                    "aperture_area_m2": actual_aperture_area_m2,
                    "num_collectors": num_collectors,
                    "land_area_m2": required_land_area_m2,
                    "land_area_acres": required_land_area_acres,
                    "target_solar_fraction": target_solar_fraction,
                },
                max_recommendations=5,
                focus_areas=["field_optimization", "land_use", "collector_selection", "economics"]
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
                    "agent": "FieldLayoutAgent",
                    "intelligence_metrics": self.get_intelligence_metrics(),
                    "regulatory_context": "IEC 62862, ISO 9806"
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
