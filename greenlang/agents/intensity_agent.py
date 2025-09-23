from typing import Dict, Any, Optional
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
import logging

logger = logging.getLogger(__name__)


class IntensityAgent(BaseAgent):
    """Agent for calculating various emission intensity metrics"""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="IntensityAgent",
                description="Calculates emission intensity metrics per area, occupancy, and other factors",
            )
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        required = ["total_emissions_kg"]
        return all(field in input_data for field in required)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Calculate various intensity metrics"""
        try:
            total_emissions_kg = input_data.get("total_emissions_kg", 0)
            area = input_data.get("area", 0)
            area_unit = input_data.get("area_unit", "sqft")
            occupancy = input_data.get("occupancy", 0)
            floor_count = input_data.get("floor_count", 1)
            period_months = input_data.get("period_months", 12)
            operating_hours = input_data.get("operating_hours_per_day", 10)
            operating_days = input_data.get("operating_days_per_year", 260)
            revenue = input_data.get("revenue", 0)
            production_units = input_data.get("production_units", 0)

            # Convert to annual emissions if not already
            annual_emissions_kg = total_emissions_kg * (12 / period_months)

            # Convert area to consistent units (sqft)
            if area_unit == "sqm":
                area_sqft = area * 10.764
                area_sqm = area
            else:
                area_sqft = area
                area_sqm = area / 10.764

            intensities = {}

            # Area-based intensities
            if area_sqft > 0:
                intensities["per_sqft_year"] = round(annual_emissions_kg / area_sqft, 3)
                intensities["per_sqm_year"] = round(annual_emissions_kg / area_sqm, 3)
                intensities["per_sqft_month"] = round(
                    annual_emissions_kg / area_sqft / 12, 4
                )

            # Occupancy-based intensities
            if occupancy > 0:
                intensities["per_person_year"] = round(
                    annual_emissions_kg / occupancy, 1
                )
                intensities["per_person_day"] = round(
                    annual_emissions_kg / occupancy / operating_days, 3
                )

            # Floor-based intensity
            if floor_count > 0:
                intensities["per_floor_year"] = round(
                    annual_emissions_kg / floor_count, 1
                )

            # Operating hours intensity
            total_operating_hours = operating_hours * operating_days
            if total_operating_hours > 0:
                intensities["per_operating_hour"] = round(
                    annual_emissions_kg / total_operating_hours, 2
                )

            # Economic intensity
            if revenue > 0:
                intensities["per_revenue_dollar"] = round(
                    annual_emissions_kg / revenue, 6
                )
                intensities["per_million_revenue"] = round(
                    annual_emissions_kg / (revenue / 1000000), 1
                )

            # Production intensity (for industrial)
            if production_units > 0:
                intensities["per_production_unit"] = round(
                    annual_emissions_kg / production_units, 3
                )

            # Calculate energy use intensity if energy data provided
            if "total_energy_kwh" in input_data:
                total_energy_kwh = input_data["total_energy_kwh"]
                if area_sqft > 0:
                    intensities["energy_use_intensity_kwh_sqft"] = round(
                        total_energy_kwh / area_sqft, 1
                    )
                if total_energy_kwh > 0:
                    intensities["carbon_per_kwh"] = round(
                        total_emissions_kg / total_energy_kwh, 4
                    )

            # Determine performance rating based on intensity
            rating = self._determine_rating(
                intensities.get("per_sqft_year", 0),
                input_data.get("building_type", "commercial_office"),
            )

            # Calculate metrics vs benchmarks
            benchmark_comparison = self._compare_to_benchmark(
                intensities.get("per_sqft_year", 0),
                input_data.get("building_type", "commercial_office"),
                input_data.get("country", "US"),
            )

            return AgentResult(
                success=True,
                data={
                    "intensities": intensities,
                    "annual_emissions_kg": round(annual_emissions_kg, 1),
                    "annual_emissions_tons": round(annual_emissions_kg / 1000, 2),
                    "performance_rating": rating,
                    "benchmark_comparison": benchmark_comparison,
                    "primary_metric": {
                        "value": intensities.get("per_sqft_year", 0),
                        "unit": "kgCO2e/sqft/year",
                        "description": "Primary intensity metric for benchmarking",
                    },
                },
                metadata={
                    "calculation_period": f"{period_months} months",
                    "area_used": f"{area_sqft:.0f} sqft",
                    "metrics_calculated": len(intensities),
                },
            )

        except Exception as e:
            logger.error(f"Error calculating intensities: {e}")
            return AgentResult(success=False, error=str(e))

    def _determine_rating(self, intensity: float, building_type: str) -> str:
        """Determine performance rating based on intensity"""
        # Intensity thresholds (kgCO2e/sqft/year)
        thresholds = {
            "commercial_office": {
                "excellent": 10,
                "good": 20,
                "average": 35,
                "poor": 50,
            },
            "hospital": {"excellent": 30, "good": 50, "average": 75, "poor": 100},
            "data_center": {"excellent": 200, "good": 400, "average": 600, "poor": 800},
            "retail": {"excellent": 15, "good": 30, "average": 45, "poor": 60},
            "warehouse": {"excellent": 5, "good": 10, "average": 20, "poor": 30},
            "hotel": {"excellent": 20, "good": 35, "average": 50, "poor": 70},
            "education": {"excellent": 12, "good": 25, "average": 40, "poor": 55},
        }

        building_thresholds = thresholds.get(
            building_type, thresholds["commercial_office"]
        )

        if intensity <= building_thresholds["excellent"]:
            return "Excellent"
        elif intensity <= building_thresholds["good"]:
            return "Good"
        elif intensity <= building_thresholds["average"]:
            return "Average"
        elif intensity <= building_thresholds["poor"]:
            return "Below Average"
        else:
            return "Poor"

    def _compare_to_benchmark(
        self, intensity: float, building_type: str, country: str
    ) -> Dict:
        """Compare intensity to regional benchmarks"""
        # Regional benchmark adjustments
        regional_factors = {
            "US": 1.0,
            "IN": 0.8,  # Lower benchmarks due to lower grid intensity expectations
            "EU": 0.7,
            "CN": 1.1,
            "JP": 0.9,
            "BR": 0.5,  # Very low due to clean grid
            "KR": 1.0,
            "UK": 0.7,
            "CA": 0.6,
            "AU": 1.2,
        }

        factor = regional_factors.get(country, 1.0)

        # Get base thresholds
        base_thresholds = {
            "commercial_office": 35,  # Average kgCO2e/sqft/year
            "hospital": 75,
            "data_center": 600,
            "retail": 45,
            "warehouse": 20,
            "hotel": 50,
            "education": 40,
        }

        benchmark = base_thresholds.get(building_type, 35) * factor

        difference = intensity - benchmark
        percentage = ((intensity - benchmark) / benchmark * 100) if benchmark > 0 else 0

        return {
            "regional_benchmark": round(benchmark, 1),
            "difference": round(difference, 1),
            "percentage_difference": round(percentage, 1),
            "performance": (
                "Better than benchmark" if difference < 0 else "Worse than benchmark"
            ),
            "benchmark_source": f"{country} typical {building_type}",
        }
