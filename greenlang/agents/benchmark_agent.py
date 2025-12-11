# -*- coding: utf-8 -*-
"""
BenchmarkAgent - AI-Native Emissions Benchmarking Agent

This agent compares emissions against industry benchmarks with
FULL LLM INTELLIGENCE capabilities via IntelligenceMixin.

Intelligence Level: STANDARD
Capabilities: Explanations, Recommendations

Regulatory Context: ENERGY STAR, GRESB, CRREM
"""
from typing import Any, Dict
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.intelligence_mixin import IntelligenceMixin
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel


class BenchmarkAgent(IntelligenceMixin, BaseAgent):
    """
    AI-Native Emissions Benchmarking Agent.

    This agent compares emissions against industry benchmarks with
    FULL LLM INTELLIGENCE:

    1. DETERMINISTIC CALCULATIONS (Zero-Hallucination):
       - Carbon intensity calculation
       - Percentile ranking
       - Benchmark comparison

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of performance
       - Prioritized improvement recommendations
       - Peer comparison narratives
    """

    BENCHMARKS = {
        "commercial_office": {
            "excellent": 20,
            "good": 35,
            "average": 50,
            "poor": 70,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "retail": {
            "excellent": 25,
            "good": 40,
            "average": 55,
            "poor": 75,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "warehouse": {
            "excellent": 15,
            "good": 25,
            "average": 35,
            "poor": 50,
            "unit": "kg_co2e_per_sqft_per_year",
        },
        "residential": {
            "excellent": 15,
            "good": 25,
            "average": 35,
            "poor": 45,
            "unit": "kg_co2e_per_sqft_per_year",
        },
    }

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="BenchmarkAgent",
                description="AI-native emissions benchmarking with LLM intelligence",
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
            can_reason=True,
            can_validate=False,
            uses_rag=False,
            uses_tools=False
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return "building_area" in input_data and input_data.get("building_area", 0) > 0

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        building_type = input_data.get("building_type", "commercial_office")
        total_emissions_kg = input_data.get("total_emissions_kg", 0)
        building_area = input_data.get("building_area", 0)
        period_months = input_data.get("period_months", 12)

        if building_area <= 0:
            return AgentResult(
                success=False, error="Building area must be greater than 0"
            )

        # =================================================================
        # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
        # =================================================================
        annualized_emissions = (total_emissions_kg / period_months) * 12
        intensity = annualized_emissions / building_area

        benchmarks = self.BENCHMARKS.get(
            building_type, self.BENCHMARKS["commercial_office"]
        )

        rating = self._get_rating(intensity, benchmarks)
        percentile = self._estimate_percentile(intensity, benchmarks)
        static_recommendations = self._generate_recommendations(rating, intensity, benchmarks)

        # Build deterministic result
        result_data = {
            "carbon_intensity": round(intensity, 2),
            "unit": "kg_co2e_per_sqft_per_year",
            "rating": rating,
            "percentile": percentile,
            "benchmarks": benchmarks,
            "comparison": {
                "vs_excellent": round(intensity - benchmarks["excellent"], 2),
                "vs_average": round(intensity - benchmarks["average"], 2),
                "improvement_to_good": max(
                    0, round(intensity - benchmarks["good"], 2)
                ),
            },
            "static_recommendations": static_recommendations,
        }

        # =================================================================
        # STEP 2: AI-POWERED EXPLANATION
        # =================================================================
        calculation_steps = [
            f"Annualized emissions: {annualized_emissions:.2f} kg CO2e/year",
            f"Building area: {building_area} sqft",
            f"Carbon intensity: {intensity:.2f} kg CO2e/sqft/year",
            f"Compared against {building_type} benchmarks",
            f"Rating determined: {rating} ({percentile}th percentile)"
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
        ai_recommendations = self.generate_recommendations(
            analysis={
                **result_data,
                "building_type": building_type,
                "gap_to_good": max(0, intensity - benchmarks["good"]),
                "gap_to_excellent": max(0, intensity - benchmarks["excellent"]),
            },
            max_recommendations=5,
            focus_areas=["efficiency", "equipment_upgrades", "renewable_energy", "building_envelope"]
        )
        result_data["recommendations"] = ai_recommendations

        # =================================================================
        # STEP 4: ADD INTELLIGENCE METADATA
        # =================================================================
        result_data["intelligence_level"] = self.get_intelligence_level().value

        return AgentResult(
            success=True,
            data=result_data,
            metadata={
                "agent": "BenchmarkAgent",
                "building_type": building_type,
                "intelligence_metrics": self.get_intelligence_metrics(),
                "regulatory_context": "ENERGY STAR, GRESB, CRREM"
            },
        )

    def _get_rating(self, intensity: float, benchmarks: Dict) -> str:
        if intensity <= benchmarks["excellent"]:
            return "Excellent"
        elif intensity <= benchmarks["good"]:
            return "Good"
        elif intensity <= benchmarks["average"]:
            return "Average"
        elif intensity <= benchmarks["poor"]:
            return "Below Average"
        else:
            return "Poor"

    def _estimate_percentile(self, intensity: float, benchmarks: Dict) -> int:
        if intensity <= benchmarks["excellent"]:
            return 90
        elif intensity <= benchmarks["good"]:
            return 70
        elif intensity <= benchmarks["average"]:
            return 50
        elif intensity <= benchmarks["poor"]:
            return 30
        else:
            return 10

    def _generate_recommendations(
        self, rating: str, intensity: float, benchmarks: Dict
    ) -> list:
        """Generate static recommendations (fallback)."""
        recommendations = []

        if rating == "Excellent":
            recommendations.append("Maintain current excellent performance")
            recommendations.append("Consider pursuing green building certifications")
            recommendations.append("Share best practices with industry peers")
        elif rating == "Good":
            recommendations.append("Focus on energy efficiency improvements")
            recommendations.append("Consider renewable energy sources")
            recommendations.append("Implement smart building technologies")
        elif rating == "Average":
            recommendations.append("Conduct energy audit to identify improvement areas")
            recommendations.append("Upgrade to energy-efficient lighting and HVAC")
            recommendations.append("Implement energy management system")
            recommendations.append("Consider on-site renewable energy generation")
        else:
            recommendations.append("Urgent: Conduct comprehensive energy audit")
            recommendations.append("Replace inefficient equipment immediately")
            recommendations.append("Implement aggressive energy reduction program")
            recommendations.append("Consider building envelope improvements")
            recommendations.append("Evaluate renewable energy options")

        reduction_needed = max(0, intensity - benchmarks["good"])
        if reduction_needed > 0:
            percent_reduction = (reduction_needed / intensity) * 100
            recommendations.insert(
                0,
                f"Reduce emissions by {percent_reduction:.1f}% to achieve 'Good' rating",
            )

        return recommendations
