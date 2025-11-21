# -*- coding: utf-8 -*-
"""
Scope 3 Category 6: Business Travel

Calculates emissions from transportation of employees for business-related activities
in vehicles not owned or operated by the reporting company.

GHG Protocol Definition:
Emissions from the transportation of employees for business-related activities
in vehicles owned or operated by third parties, including aircraft, trains,
buses, and passenger cars.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator

from greenlang.agents.scope3.base import (
    Scope3BaseAgent,
    Scope3InputData,
    Scope3Result,
    AgentConfig
)


class BusinessTravelInput(Scope3InputData):
    """Input data model for Business Travel emissions calculation."""

    travel_data: List[Dict[str, Any]] = Field(
        ...,
        description="List of business travel activities"
    )

    class TravelActivity(BaseModel):
        """Individual travel activity."""
        mode: str = Field(..., description="Travel mode: air, rail, road, other")
        distance_km: Optional[float] = Field(None, gt=0, description="Distance traveled in km")
        passenger_km: Optional[float] = Field(None, gt=0, description="Passenger-kilometers")
        travel_class: Optional[str] = Field(
            "economy",
            description="Travel class: economy, premium_economy, business, first"
        )
        num_travelers: int = Field(1, gt=0, description="Number of travelers")
        hotel_nights: Optional[int] = Field(None, ge=0, description="Hotel nights stayed")
        hotel_country: Optional[str] = Field(None, description="Country of hotel stay")


class BusinessTravelAgent(Scope3BaseAgent):
    """
    Agent for calculating Scope 3 Category 6: Business Travel emissions.

    Includes:
    - Air travel (with class multipliers)
    - Rail travel
    - Road travel (taxi, rental car, personal vehicle reimbursement)
    - Hotel stays
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Business Travel Agent."""
        if config is None:
            config = AgentConfig(
                name="BusinessTravelAgent",
                description="Calculates Scope 3 Category 6: Business Travel emissions",
                version="1.0.0"
            )
        super().__init__(config)

        self.travel_factors = self._load_travel_factors()

    def _load_travel_factors(self) -> Dict[str, Any]:
        """Load business travel emission factors."""
        return {
            "air": {
                # kg CO2e per passenger-km by distance and class
                "short_haul": {  # <500km
                    "economy": Decimal("0.255"),
                    "premium_economy": Decimal("0.408"),
                    "business": Decimal("0.510"),
                    "first": Decimal("0.765")
                },
                "medium_haul": {  # 500-3700km
                    "economy": Decimal("0.156"),
                    "premium_economy": Decimal("0.250"),
                    "business": Decimal("0.312"),
                    "first": Decimal("0.468")
                },
                "long_haul": {  # >3700km
                    "economy": Decimal("0.147"),
                    "premium_economy": Decimal("0.235"),
                    "business": Decimal("0.426"),
                    "first": Decimal("0.588")
                }
            },
            "rail": {
                "high_speed": Decimal("0.014"),  # kg CO2e per passenger-km
                "intercity": Decimal("0.041"),
                "regional": Decimal("0.065"),
                "subway": Decimal("0.031"),
                "default": Decimal("0.041")
            },
            "road": {
                "taxi": Decimal("0.175"),  # kg CO2e per km
                "rental_car_small": Decimal("0.145"),
                "rental_car_medium": Decimal("0.172"),
                "rental_car_large": Decimal("0.213"),
                "bus": Decimal("0.089"),
                "default": Decimal("0.171")
            },
            "hotel": {
                # kg CO2e per room-night by country/region
                "us": Decimal("19.47"),
                "uk": Decimal("11.91"),
                "eu": Decimal("10.52"),
                "china": Decimal("27.43"),
                "india": Decimal("22.81"),
                "global": Decimal("15.68")
            }
        }

    def _get_air_travel_category(self, distance_km: float) -> str:
        """Determine air travel category based on distance."""
        if distance_km < 500:
            return "short_haul"
        elif distance_km <= 3700:
            return "medium_haul"
        else:
            return "long_haul"

    def _parse_input(self, input_data: Dict[str, Any]) -> BusinessTravelInput:
        """Parse and validate input data."""
        return BusinessTravelInput(**input_data)

    async def calculate_emissions(
        self,
        input_data: BusinessTravelInput
    ) -> Scope3Result:
        """
        Calculate Business Travel emissions.

        Formulas:
        - Air/Rail/Road: distance_km × emission_factor × num_travelers
        - Hotel: hotel_nights × emission_factor
        """
        total_emissions = Decimal("0")

        for idx, travel in enumerate(input_data.travel_data):
            mode = travel['mode'].lower()
            emissions = Decimal("0")

            if mode == "air":
                distance = Decimal(str(travel.get('distance_km', 0)))
                travel_class = travel.get('travel_class', 'economy')
                num_travelers = travel.get('num_travelers', 1)

                # Determine haul category
                category = self._get_air_travel_category(float(distance))
                factor = self.travel_factors['air'][category][travel_class]

                emissions = distance * factor * Decimal(str(num_travelers))

                self._record_calculation_step(
                    description=f"Calculate air travel emissions ({category}, {travel_class})",
                    operation="multiply",
                    inputs={
                        "distance_km": distance,
                        "emission_factor": factor,
                        "num_travelers": num_travelers,
                        "travel_class": travel_class
                    },
                    output_value=emissions,
                    output_name=f"travel_{idx}_air_emissions",
                    formula="distance × factor × travelers",
                    unit="kg CO2e"
                )

            elif mode == "rail":
                distance = Decimal(str(travel.get('distance_km', 0)))
                num_travelers = travel.get('num_travelers', 1)
                rail_type = travel.get('vehicle_type', 'default')

                factor = self.travel_factors['rail'].get(
                    rail_type,
                    self.travel_factors['rail']['default']
                )

                emissions = distance * factor * Decimal(str(num_travelers))

                self._record_calculation_step(
                    description=f"Calculate rail travel emissions",
                    operation="multiply",
                    inputs={
                        "distance_km": distance,
                        "emission_factor": factor,
                        "num_travelers": num_travelers
                    },
                    output_value=emissions,
                    output_name=f"travel_{idx}_rail_emissions",
                    formula="distance × factor × travelers",
                    unit="kg CO2e"
                )

            elif mode == "road":
                distance = Decimal(str(travel.get('distance_km', 0)))
                vehicle_type = travel.get('vehicle_type', 'default')

                factor = self.travel_factors['road'].get(
                    vehicle_type,
                    self.travel_factors['road']['default']
                )

                emissions = distance * factor

                self._record_calculation_step(
                    description=f"Calculate road travel emissions",
                    operation="multiply",
                    inputs={
                        "distance_km": distance,
                        "emission_factor": factor,
                        "vehicle_type": vehicle_type
                    },
                    output_value=emissions,
                    output_name=f"travel_{idx}_road_emissions",
                    formula="distance × emission_factor",
                    unit="kg CO2e"
                )

            # Add hotel emissions if present
            if travel.get('hotel_nights'):
                nights = Decimal(str(travel['hotel_nights']))
                country = travel.get('hotel_country', 'global').lower()

                hotel_factor = self.travel_factors['hotel'].get(
                    country,
                    self.travel_factors['hotel']['global']
                )

                hotel_emissions = nights * hotel_factor

                self._record_calculation_step(
                    description=f"Calculate hotel stay emissions",
                    operation="multiply",
                    inputs={
                        "nights": nights,
                        "emission_factor": hotel_factor,
                        "country": country
                    },
                    output_value=hotel_emissions,
                    output_name=f"travel_{idx}_hotel_emissions",
                    formula="nights × emission_factor",
                    unit="kg CO2e"
                )

                emissions += hotel_emissions

            total_emissions += emissions
            self.factors_used[f"travel_{idx}"] = {
                "mode": mode,
                "emissions": float(emissions)
            }

        # Convert to tonnes
        total_emissions_tonnes = total_emissions / Decimal("1000")

        self._record_calculation_step(
            description="Convert total emissions to tonnes CO2e",
            operation="divide",
            inputs={
                "total_kg_co2e": total_emissions,
                "conversion_factor": Decimal("1000")
            },
            output_value=total_emissions_tonnes,
            output_name="total_emissions_t_co2e",
            formula="total_kg_co2e / 1000",
            unit="t CO2e"
        )

        # Data quality and uncertainty
        data_quality = 2.0  # Good quality for distance-based
        uncertainty = self._estimate_uncertainty(data_quality, "distance-based")

        # Generate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            self.calculation_steps,
            total_emissions
        )

        return Scope3Result(
            category="Category 6: Business Travel",
            category_number=6,
            total_emissions_kg_co2e=self._apply_precision(total_emissions),
            total_emissions_t_co2e=self._apply_precision(total_emissions_tonnes),
            calculation_methodology="distance-based",
            data_quality_score=data_quality,
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=provenance_hash,
            calculation_timestamp=self.clock.now().isoformat(),
            ghg_protocol_compliance=True,
            uncertainty_range=uncertainty
        )

    def get_example_calculation(self) -> Dict[str, Any]:
        """Provide example calculation."""
        return {
            "example_input": {
                "reporting_year": 2024,
                "reporting_entity": "Example Corp",
                "travel_data": [
                    {
                        "mode": "air",
                        "distance_km": 5000,
                        "travel_class": "business",
                        "num_travelers": 2,
                        "hotel_nights": 3,
                        "hotel_country": "us"
                    },
                    {
                        "mode": "rail",
                        "distance_km": 200,
                        "vehicle_type": "high_speed",
                        "num_travelers": 5
                    }
                ]
            },
            "calculation": """
            1. Air travel (long-haul business):
               5,000 km × 0.426 kg CO2e/km × 2 travelers = 4,260 kg CO2e

            2. Hotel stay:
               3 nights × 19.47 kg CO2e/night = 58.41 kg CO2e

            3. Rail travel:
               200 km × 0.014 kg CO2e/km × 5 travelers = 14 kg CO2e

            Total: 4,332.41 kg CO2e = 4.33 t CO2e
            """,
            "result": "4.33 t CO2e"
        }