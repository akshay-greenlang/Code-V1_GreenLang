from typing import Any, Dict
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig


class BenchmarkAgent(BaseAgent):
    """Agent for comparing emissions against industry benchmarks"""

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
                description="Compares emissions against industry benchmarks",
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        building_type = input_data.get("building_type", "commercial_office")
        total_emissions_kg = input_data.get("total_emissions_kg", 0)
        building_area = input_data.get("building_area", 0)
        period_months = input_data.get("period_months", 12)

        if building_area <= 0:
            return AgentResult(
                success=False, error="Building area must be greater than 0"
            )

        annualized_emissions = (total_emissions_kg / period_months) * 12
        intensity = annualized_emissions / building_area

        benchmarks = self.BENCHMARKS.get(
            building_type, self.BENCHMARKS["commercial_office"]
        )

        rating = self._get_rating(intensity, benchmarks)
        percentile = self._estimate_percentile(intensity, benchmarks)
        recommendations = self._generate_recommendations(rating, intensity, benchmarks)

        return AgentResult(
            success=True,
            data={
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
                "recommendations": recommendations,
            },
            metadata={"agent": "BenchmarkAgent", "building_type": building_type},
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
