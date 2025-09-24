from typing import Dict, Any, Optional
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
import logging

logger = logging.getLogger(__name__)


class BuildingProfileAgent(BaseAgent):
    """Agent for categorizing buildings and determining appropriate benchmarks"""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="BuildingProfileAgent",
                description="Categorizes buildings and maps to appropriate benchmarks and standards",
            )
        )
        self.building_profiles = self._load_building_profiles()

    def _load_building_profiles(self) -> Dict:
        """Load building profile data"""
        return {
            "commercial_office": {
                "typical_eui": {  # Energy Use Intensity kWh/sqft/year
                    "excellent": 30,
                    "good": 50,
                    "average": 75,
                    "poor": 100,
                },
                "typical_occupancy_density": 200,  # sqft per person
                "typical_operating_hours": 10,
                "hvac_load_factor": 0.45,
                "lighting_load_factor": 0.25,
                "plug_load_factor": 0.20,
            },
            "hospital": {
                "typical_eui": {
                    "excellent": 100,
                    "good": 150,
                    "average": 200,
                    "poor": 250,
                },
                "typical_occupancy_density": 150,
                "typical_operating_hours": 24,
                "hvac_load_factor": 0.40,
                "lighting_load_factor": 0.20,
                "equipment_load_factor": 0.35,
            },
            "data_center": {
                "typical_eui": {
                    "excellent": 500,
                    "good": 750,
                    "average": 1000,
                    "poor": 1500,
                },
                "typical_pue": {  # Power Usage Effectiveness
                    "excellent": 1.2,
                    "good": 1.5,
                    "average": 1.8,
                    "poor": 2.2,
                },
                "typical_operating_hours": 24,
                "it_load_factor": 0.60,
                "cooling_load_factor": 0.30,
            },
            "retail": {
                "typical_eui": {
                    "excellent": 40,
                    "good": 65,
                    "average": 90,
                    "poor": 120,
                },
                "typical_occupancy_density": 100,
                "typical_operating_hours": 12,
                "hvac_load_factor": 0.40,
                "lighting_load_factor": 0.35,
                "plug_load_factor": 0.15,
            },
            "warehouse": {
                "typical_eui": {"excellent": 15, "good": 25, "average": 35, "poor": 50},
                "typical_occupancy_density": 1000,
                "typical_operating_hours": 16,
                "hvac_load_factor": 0.25,
                "lighting_load_factor": 0.40,
                "equipment_load_factor": 0.25,
            },
            "hotel": {
                "typical_eui": {
                    "excellent": 60,
                    "good": 85,
                    "average": 110,
                    "poor": 140,
                },
                "typical_occupancy_rate": 0.70,
                "typical_operating_hours": 24,
                "hvac_load_factor": 0.45,
                "lighting_load_factor": 0.20,
                "water_heating_factor": 0.20,
            },
            "education": {
                "typical_eui": {"excellent": 35, "good": 55, "average": 75, "poor": 95},
                "typical_occupancy_density": 150,
                "typical_operating_hours": 10,
                "typical_operating_days": 180,
                "hvac_load_factor": 0.50,
                "lighting_load_factor": 0.25,
            },
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        return "building_type" in input_data

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Analyze building and provide profile information"""
        try:
            building_type = input_data.get("building_type", "commercial_office")
            area = input_data.get("area", 0)
            area_unit = input_data.get("area_unit", "sqft")
            occupancy = input_data.get("occupancy")
            floor_count = input_data.get("floor_count", 1)
            building_age = input_data.get("building_age", 0)
            climate_zone = input_data.get("climate_zone")
            country = input_data.get("country", "US")

            # Convert area to sqft if needed
            if area_unit == "sqm":
                area = area * 10.764

            # Get building profile
            profile = self.building_profiles.get(
                building_type, self.building_profiles["commercial_office"]
            )

            # Calculate expected metrics
            expected_occupancy = area / profile.get("typical_occupancy_density", 200)

            # Determine efficiency category based on age
            if building_age < 5:
                age_category = "new"
                efficiency_modifier = 0.85
            elif building_age < 15:
                age_category = "modern"
                efficiency_modifier = 1.0
            elif building_age < 30:
                age_category = "mature"
                efficiency_modifier = 1.15
            else:
                age_category = "old"
                efficiency_modifier = 1.30

            # Climate zone adjustments
            climate_adjustments = self._get_climate_adjustments(climate_zone)

            # Calculate expected energy use
            base_eui = profile["typical_eui"]["average"]
            adjusted_eui = (
                base_eui
                * efficiency_modifier
                * climate_adjustments.get("hvac_modifier", 1.0)
            )

            # Provide load breakdown
            load_breakdown = {}
            total_factor = 0
            for load_type in [
                "hvac_load_factor",
                "lighting_load_factor",
                "plug_load_factor",
                "equipment_load_factor",
                "it_load_factor",
                "water_heating_factor",
            ]:
                if load_type in profile:
                    load_breakdown[load_type.replace("_factor", "")] = profile[
                        load_type
                    ]
                    total_factor += profile[load_type]

            # Normalize if factors don't sum to ~1
            if total_factor > 0 and abs(total_factor - 1.0) > 0.1:
                for key in load_breakdown:
                    load_breakdown[key] = load_breakdown[key] / total_factor

            result_data = {
                "building_category": building_type,
                "age_category": age_category,
                "typical_eui_range": profile["typical_eui"],
                "expected_eui": round(adjusted_eui, 1),
                "expected_annual_energy_kwh": round(adjusted_eui * area),
                "expected_occupancy": round(expected_occupancy),
                "typical_operating_hours": profile.get("typical_operating_hours", 10),
                "load_breakdown": load_breakdown,
                "efficiency_modifier": efficiency_modifier,
                "climate_adjustment": climate_adjustments,
                "benchmark_standards": self._get_benchmark_standards(
                    building_type, country
                ),
            }

            # Add specific metrics for data centers
            if building_type == "data_center":
                result_data["typical_pue"] = profile.get("typical_pue", {})

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "analysis_type": "building_profile",
                    "area_sqft": area,
                    "floors": floor_count,
                },
            )

        except Exception as e:
            logger.error(f"Error profiling building: {e}")
            return AgentResult(success=False, error=str(e))

    def _get_climate_adjustments(self, climate_zone: Optional[str]) -> Dict:
        """Get climate-based adjustment factors"""
        if not climate_zone:
            return {"hvac_modifier": 1.0, "description": "No climate zone specified"}

        climate_factors = {
            "1A": {"hvac_modifier": 1.3, "description": "Very Hot-Humid"},
            "2A": {"hvac_modifier": 1.2, "description": "Hot-Humid"},
            "3A": {"hvac_modifier": 1.1, "description": "Warm-Humid"},
            "4A": {"hvac_modifier": 1.0, "description": "Mixed-Humid"},
            "5A": {"hvac_modifier": 1.05, "description": "Cool-Humid"},
            "6A": {"hvac_modifier": 1.15, "description": "Cold-Humid"},
            "7": {"hvac_modifier": 1.25, "description": "Very Cold"},
            "8": {"hvac_modifier": 1.35, "description": "Subarctic"},
            "tropical": {"hvac_modifier": 1.25, "description": "Tropical climate"},
            "dry": {"hvac_modifier": 1.1, "description": "Dry climate"},
            "temperate": {"hvac_modifier": 1.0, "description": "Temperate climate"},
            "continental": {
                "hvac_modifier": 1.15,
                "description": "Continental climate",
            },
            "polar": {"hvac_modifier": 1.4, "description": "Polar climate"},
        }

        return climate_factors.get(
            climate_zone, {"hvac_modifier": 1.0, "description": "Unknown zone"}
        )

    def _get_benchmark_standards(self, building_type: str, country: str) -> Dict:
        """Get relevant benchmark standards for the building type and country"""
        standards = {
            "US": {
                "program": "ENERGY STAR",
                "certification_levels": ["Certified (75+)", "Not Certified (<75)"],
                "scoring_range": "1-100",
            },
            "IN": {
                "program": "BEE Star Rating",
                "certification_levels": [
                    "5 Star",
                    "4 Star",
                    "3 Star",
                    "2 Star",
                    "1 Star",
                ],
                "scoring_range": "1-5 stars",
            },
            "EU": {
                "program": "EU Energy Performance Certificate",
                "certification_levels": ["A", "B", "C", "D", "E", "F", "G"],
                "scoring_range": "A-G",
            },
            "UK": {
                "program": "UK EPC",
                "certification_levels": ["A", "B", "C", "D", "E", "F", "G"],
                "scoring_range": "A-G",
            },
            "JP": {
                "program": "Top Runner / CASBEE",
                "certification_levels": ["S", "A", "B+", "B-", "C"],
                "scoring_range": "S-C",
            },
            "CN": {
                "program": "China Green Building Label",
                "certification_levels": ["3 Star", "2 Star", "1 Star"],
                "scoring_range": "1-3 stars",
            },
        }

        return standards.get(country, standards["US"])
