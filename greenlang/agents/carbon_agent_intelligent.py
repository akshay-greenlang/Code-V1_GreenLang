# -*- coding: utf-8 -*-
"""
Intelligent Carbon Agent - AI-Native Carbon Footprint Calculator

This is the RETROFITTED version of CarbonAgent with full LLM intelligence.
It demonstrates the pattern for upgrading any existing agent to be AI-native.

BEFORE (carbon_agent.py):
    - Aggregates emissions deterministically
    - Returns static summary strings
    - No AI insights or recommendations

AFTER (this file):
    - Same deterministic calculations (zero-hallucination)
    - AI-generated explanations of carbon footprint
    - AI-powered recommendations for reduction
    - Anomaly detection for unusual emission patterns
    - Regulatory context awareness

Migration Pattern:
    The original CarbonAgent calculation logic is UNCHANGED.
    Intelligence is ADDED on top via IntelligenceMixin.

Usage:
    from greenlang.agents.carbon_agent_intelligent import IntelligentCarbonAgent

    agent = IntelligentCarbonAgent()
    result = agent.run({
        "emissions": [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
            {"fuel_type": "electricity", "co2e_emissions_kg": 3000}
        ]
    })

    # Result now includes AI-generated content:
    print(result.data["explanation"])  # "Your total carbon footprint is..."
    print(result.data["recommendations"])  # [{"title": "Switch to renewable...", ...}]

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.intelligent_base import (
    IntelligentAgentBase,
    IntelligentAgentConfig,
    IntelligenceLevel,
    Recommendation,
    Anomaly,
)
from greenlang.agents.intelligence_interface import (
    IntelligenceCapabilities,
    require_intelligence,
)
from greenlang.determinism import DeterministicClock
from templates.agent_monitoring import OperationalMonitoringMixin

logger = logging.getLogger(__name__)


class IntelligentCarbonAgentConfig(IntelligentAgentConfig):
    """Configuration for Intelligent Carbon Agent."""

    # Inherits all IntelligentAgentConfig fields:
    # - intelligence_level, llm_model, max_budget_per_call_usd, etc.

    class Config:
        extra = "allow"


@require_intelligence
class IntelligentCarbonAgent(OperationalMonitoringMixin, IntelligentAgentBase):
    """
    AI-Native Carbon Footprint Calculator.

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
       - Regulatory context (CSRD, SB253, etc.)

    Intelligence Level: STANDARD (explanations + recommendations)
    """

    def __init__(self, config: Optional[IntelligentCarbonAgentConfig] = None):
        """Initialize the intelligent carbon agent."""
        if config is None:
            config = IntelligentCarbonAgentConfig(
                name="IntelligentCarbonAgent",
                description="AI-native carbon footprint calculator with LLM intelligence",
                intelligence_level=IntelligenceLevel.STANDARD,
                enable_explanations=True,
                enable_recommendations=True,
                enable_anomaly_detection=True,
                regulatory_context="GHG Protocol, CSRD ESRS E1",
                domain_context="greenhouse gas emissions and carbon footprint"
            )

        super().__init__(config)
        self.setup_monitoring(agent_name="intelligent_carbon_agent")

        logger.info(
            f"Initialized IntelligentCarbonAgent with intelligence level: "
            f"{config.intelligence_level.value}"
        )

    # =========================================================================
    # MANDATORY: Intelligence Interface Implementation
    # =========================================================================

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return self.intelligent_config.intelligence_level

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure."""
        if "emissions" not in input_data:
            logger.error("Missing required field: emissions")
            return False

        if not isinstance(input_data["emissions"], list):
            logger.error("emissions must be a list")
            return False

        return True

    # =========================================================================
    # CORE EXECUTION
    # =========================================================================

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute carbon footprint calculation with AI intelligence.

        This method:
        1. Performs deterministic emission aggregation
        2. Generates AI explanation of the footprint
        3. Generates AI recommendations for reduction
        4. Detects anomalies in emission patterns

        Args:
            input_data: Dictionary with "emissions" list

        Returns:
            AgentResult with calculation data and AI intelligence
        """
        with self.track_execution(input_data) as tracker:
            emissions_list = input_data.get("emissions", [])

            # =========================================================
            # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
            # =========================================================
            calculation_result = self._calculate_footprint(emissions_list)
            calculation_steps = self._get_calculation_steps(emissions_list, calculation_result)

            # Handle empty case
            if not emissions_list:
                return AgentResult(
                    success=True,
                    data={
                        **calculation_result,
                        "explanation": "No emissions data provided for analysis.",
                        "recommendations": [],
                        "anomalies": [],
                    },
                    timestamp=DeterministicClock.now()
                )

            # =========================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =========================================================
            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=calculation_result,
                calculation_steps=calculation_steps
            )

            # =========================================================
            # STEP 3: AI-POWERED RECOMMENDATIONS
            # =========================================================
            recommendations = []
            if self.intelligent_config.enable_recommendations:
                recommendations = self.generate_recommendations(
                    analysis={
                        **calculation_result,
                        "top_sources": self._get_top_emission_sources(calculation_result),
                        "reduction_potential": self._estimate_reduction_potential(calculation_result)
                    },
                    max_recommendations=5,
                    focus_areas=["efficiency", "fuel_switching", "renewable_energy", "compliance"]
                )

            # =========================================================
            # STEP 4: ANOMALY DETECTION
            # =========================================================
            anomalies = []
            if self.intelligent_config.enable_anomaly_detection:
                anomalies = self.detect_anomalies(
                    data=input_data,
                    expected_ranges=self._get_expected_emission_ranges()
                )

            # =========================================================
            # STEP 5: BUILD INTELLIGENT RESULT
            # =========================================================
            result_data = {
                **calculation_result,
                "explanation": explanation,
                "recommendations": [r.dict() if hasattr(r, 'dict') else r for r in recommendations],
                "anomalies": [a.dict() if hasattr(a, 'dict') else a for a in anomalies],
                "intelligence_level": self.get_intelligence_level().value,
            }

            # Add intensity metrics if additional context provided
            if "building_area" in input_data or "occupancy" in input_data:
                result_data["carbon_intensity"] = self._calculate_intensity(
                    input_data, calculation_result["total_co2e_kg"]
                )

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "IntelligentCarbonAgent",
                    "num_sources": len(calculation_result.get("emissions_breakdown", [])),
                    "intelligence_metrics": self.get_intelligence_metrics().dict(),
                    "regulatory_context": self.intelligent_config.regulatory_context
                },
                timestamp=DeterministicClock.now()
            )

    # =========================================================================
    # DETERMINISTIC CALCULATION METHODS (Zero-Hallucination)
    # =========================================================================

    def _calculate_footprint(self, emissions_list: List[Dict]) -> Dict[str, Any]:
        """
        Calculate total carbon footprint - FULLY DETERMINISTIC.

        This method contains ZERO LLM calls. All calculations are
        reproducible and auditable.
        """
        if not emissions_list:
            return {
                "total_co2e_kg": 0,
                "total_co2e_tons": 0,
                "emissions_breakdown": [],
                "summary": "No emissions data provided",
            }

        total_co2e_kg = 0
        emissions_breakdown = []

        # Aggregate emissions from all sources
        for emission in emissions_list:
            if isinstance(emission, dict):
                co2e = emission.get("co2e_emissions_kg", 0)
                total_co2e_kg += co2e

                breakdown_item = {
                    "source": emission.get("fuel_type", "Unknown"),
                    "co2e_kg": co2e,
                    "co2e_tons": co2e / 1000,
                    "percentage": 0,  # Will be calculated below
                }
                emissions_breakdown.append(breakdown_item)

        # Calculate percentages
        for item in emissions_breakdown:
            if total_co2e_kg > 0:
                item["percentage"] = round((item["co2e_kg"] / total_co2e_kg) * 100, 2)

        total_co2e_tons = total_co2e_kg / 1000

        # Generate deterministic summary (not AI)
        summary = self._generate_deterministic_summary(total_co2e_tons, emissions_breakdown)

        return {
            "total_co2e_kg": round(total_co2e_kg, 2),
            "total_co2e_tons": round(total_co2e_tons, 3),
            "emissions_breakdown": emissions_breakdown,
            "summary": summary,
        }

    def _generate_deterministic_summary(
        self,
        total_tons: float,
        breakdown: List[Dict]
    ) -> str:
        """Generate a deterministic summary string (not AI-powered)."""
        if total_tons == 0:
            return "No carbon emissions"

        summary = f"Total carbon footprint: {total_tons:.3f} metric tons CO2e\n"

        if breakdown:
            summary += "Breakdown by source:\n"
            for item in sorted(breakdown, key=lambda x: x["co2e_kg"], reverse=True):
                summary += f"  - {item['source']}: {item['co2e_tons']:.3f} tons ({item['percentage']}%)\n"

        return summary.strip()

    def _get_calculation_steps(
        self,
        emissions_list: List[Dict],
        result: Dict[str, Any]
    ) -> List[str]:
        """Document calculation steps for explainability."""
        steps = [
            f"Received {len(emissions_list)} emission sources for aggregation",
        ]

        for emission in emissions_list[:5]:  # Show first 5
            fuel_type = emission.get("fuel_type", "Unknown")
            co2e = emission.get("co2e_emissions_kg", 0)
            steps.append(f"Added {fuel_type}: {co2e} kg CO2e")

        if len(emissions_list) > 5:
            steps.append(f"... and {len(emissions_list) - 5} more sources")

        steps.append(f"Calculated total: {result['total_co2e_kg']} kg CO2e")
        steps.append(f"Converted to tons: {result['total_co2e_tons']} metric tons CO2e")
        steps.append("Calculated percentage breakdown by source")

        return steps

    def _calculate_intensity(
        self,
        input_data: Dict[str, Any],
        total_co2e: float
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

    def _get_top_emission_sources(self, result: Dict[str, Any]) -> List[str]:
        """Get the top emission sources by percentage."""
        breakdown = result.get("emissions_breakdown", [])
        sorted_sources = sorted(breakdown, key=lambda x: x["co2e_kg"], reverse=True)
        return [item["source"] for item in sorted_sources[:3]]

    def _estimate_reduction_potential(self, result: Dict[str, Any]) -> str:
        """Estimate potential reduction based on source mix."""
        total = result.get("total_co2e_tons", 0)
        if total == 0:
            return "N/A"

        # Simple heuristic - 20-30% reduction typically achievable
        low_estimate = total * 0.20
        high_estimate = total * 0.30

        return f"{low_estimate:.1f}-{high_estimate:.1f} tons CO2e (20-30%)"

    def _get_expected_emission_ranges(self) -> Dict[str, tuple]:
        """Define expected ranges for anomaly detection."""
        return {
            "co2e_emissions_kg": (0, 1000000),  # 0 to 1000 tons per source
            "percentage": (0, 100),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_intelligent_carbon_agent(
    intelligence_level: IntelligenceLevel = IntelligenceLevel.STANDARD,
    regulatory_context: str = "GHG Protocol, CSRD ESRS E1",
    **kwargs
) -> IntelligentCarbonAgent:
    """
    Factory function to create IntelligentCarbonAgent.

    Args:
        intelligence_level: Level of LLM intelligence
        regulatory_context: Regulatory context for prompts
        **kwargs: Additional configuration

    Returns:
        Configured IntelligentCarbonAgent
    """
    config = IntelligentCarbonAgentConfig(
        name="IntelligentCarbonAgent",
        description="AI-native carbon footprint calculator",
        intelligence_level=intelligence_level,
        regulatory_context=regulatory_context,
        **kwargs
    )

    return IntelligentCarbonAgent(config)


# =============================================================================
# BACKWARD COMPATIBILITY: Retrofit existing CarbonAgent
# =============================================================================

def retrofit_carbon_agent():
    """
    Retrofit the existing CarbonAgent with intelligence.

    This function demonstrates how to add intelligence to any
    existing agent without modifying its original code.
    """
    from greenlang.agents.carbon_agent import CarbonAgent
    from greenlang.agents.intelligence_mixin import retrofit_agent_class

    # One-line retrofit!
    IntelligentCarbonAgentRetrofit = retrofit_agent_class(CarbonAgent)

    return IntelligentCarbonAgentRetrofit


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of IntelligentCarbonAgent."""

    # Create agent
    agent = create_intelligent_carbon_agent()

    # Example input
    example_input = {
        "emissions": [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 5310.0},
            {"fuel_type": "electricity", "co2e_emissions_kg": 3200.0},
            {"fuel_type": "diesel", "co2e_emissions_kg": 1024.5},
            {"fuel_type": "company_vehicles", "co2e_emissions_kg": 890.0}
        ],
        "building_area": 50000,
        "occupancy": 200
    }

    # Run agent
    result = agent.run(example_input)

    # Print results
    print("=" * 60)
    print("INTELLIGENT CARBON AGENT RESULTS")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\n--- DETERMINISTIC DATA ---")
    print(f"Total CO2e: {result.data['total_co2e_kg']} kg")
    print(f"Total CO2e: {result.data['total_co2e_tons']} tons")
    print(f"\n--- AI-GENERATED EXPLANATION ---")
    print(result.data.get("explanation", "N/A"))
    print(f"\n--- AI-GENERATED RECOMMENDATIONS ---")
    for rec in result.data.get("recommendations", []):
        title = rec.get("title", "Unknown") if isinstance(rec, dict) else "Unknown"
        print(f"  - {title}")
    print(f"\n--- INTELLIGENCE METRICS ---")
    print(agent.get_intelligence_metrics())
