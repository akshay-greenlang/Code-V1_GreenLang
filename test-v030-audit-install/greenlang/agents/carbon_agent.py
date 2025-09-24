from typing import Any, Dict, List
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig


class CarbonAgent(BaseAgent):
    """Agent for aggregating emissions and calculating total carbon footprint"""

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="CarbonAgent",
                description="Aggregates emissions and provides carbon footprint",
            )
        super().__init__(config)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return "emissions" in input_data and isinstance(input_data["emissions"], list)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        emissions_list = input_data["emissions"]

        if not emissions_list:
            return AgentResult(
                success=True,
                data={
                    "total_co2e_kg": 0,
                    "total_co2e_tons": 0,
                    "emissions_breakdown": [],
                    "summary": "No emissions data provided",
                },
            )

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

        return AgentResult(
            success=True,
            data={
                "total_co2e_kg": round(total_co2e_kg, 2),
                "total_co2e_tons": round(total_co2e_tons, 3),
                "emissions_breakdown": emissions_breakdown,
                "summary": summary,
                "carbon_intensity": self._calculate_intensity(
                    input_data, total_co2e_kg
                ),
            },
            metadata={"agent": "CarbonAgent", "num_sources": len(emissions_breakdown)},
        )

    def _generate_summary(self, total_tons: float, breakdown: List[Dict]) -> str:
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
        intensity = {}

        if "building_area" in input_data:
            area = input_data["building_area"]
            intensity["per_sqft"] = total_co2e / area if area > 0 else 0

        if "occupancy" in input_data:
            occupancy = input_data["occupancy"]
            intensity["per_person"] = total_co2e / occupancy if occupancy > 0 else 0

        return intensity
