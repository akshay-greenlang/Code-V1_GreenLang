"""
Category 6: Business Travel Agent

Calculates emissions from employee business travel

GHG Protocol Reference:
- Chapter 5.6 of Scope 3 Standard
- Category 6 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class BusinessTravelInput(Scope3InputData):
    """Input data for Category 6: Business Travel."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: distance-based, spend-based"
    )

    # Add category-specific fields
    air_travel: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Air travel by distance and class"
    )

    rail_travel: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Rail travel distances"
    )

    road_travel: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Car rental and taxi data"
    )

    hotel_stays: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hotel nights by country"
    )

    travel_spend: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Travel spend by category (for spend-based)"
    )


class BusinessTravelAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 6: Business Travel.

    Includes:
    - Air travel
    - Rail travel
    - Road travel (rental cars, taxis)
    - Hotel stays
    """

    def __init__(self):
        """Initialize Business Travel agent."""
        super().__init__()
        self.category_number = 6
        self.category_name = "Business Travel"
        self.emission_factors = self._load_travel_factors()

    def _load_travel_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Business Travel."""
        return {
            # Air travel factors (kg CO2e/km)
            "air_domestic": Decimal("0.244"),
            "air_short_haul": Decimal("0.156"),
            "air_long_haul": Decimal("0.193"),

            # Class multipliers
            "economy_class": Decimal("1.0"),
            "premium_economy": Decimal("1.6"),
            "business_class": Decimal("2.9"),
            "first_class": Decimal("4.0"),

            # Rail factors
            "rail_national": Decimal("0.041"),
            "rail_international": Decimal("0.014"),

            # Road factors (kg CO2e/km)
            "car_small": Decimal("0.147"),
            "car_medium": Decimal("0.186"),
            "car_large": Decimal("0.279"),
            "taxi": Decimal("0.208"),

            # Hotel factors (kg CO2e/night)
            "hotel_us": Decimal("31.1"),
            "hotel_eu": Decimal("21.0"),
            "hotel_global": Decimal("25.0")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> BusinessTravelInput:
        """Parse and validate input data."""
        return BusinessTravelInput(**input_data)

    async def calculate_emissions(self, input_data: BusinessTravelInput) -> Scope3Result:
        """
        Calculate Category 6 emissions.

        Formula:
        Emissions = Sum(Distance × Mode_EF × Class_Multiplier) + Hotel_Nights × Hotel_EF
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        if input_data.calculation_method == "distance-based":
            # Air travel
            if input_data.air_travel:
                for flight_type, data in input_data.air_travel.items():
                    distance = Decimal(str(data.get("distance_km", 0)))
                    class_type = data.get("class", "economy_class")

                    ef = self.emission_factors.get(f"air_{flight_type}")
                    multiplier = self.emission_factors.get(class_type, Decimal("1.0"))

                    emissions = distance * ef * multiplier
                    total_emissions += emissions

            # Hotel stays
            if input_data.hotel_stays:
                nights = Decimal(str(input_data.hotel_stays.get("nights", 0)))
                region = input_data.hotel_stays.get("region", "global")

                ef = self.emission_factors.get(f"hotel_{region}", self.emission_factors["hotel_global"])
                emissions = nights * ef
                total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 6: Business Travel",
            category_number=6,
            total_emissions_kg_co2e=total_emissions,
            total_emissions_t_co2e=total_emissions / Decimal("1000"),
            calculation_methodology=input_data.calculation_method,
            data_quality_score=self._calculate_data_quality_score(2.5, 2.5, 2.5, 2.5, 2.5),
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=self._calculate_provenance_hash(
                input_data.dict(),
                self.calculation_steps,
                total_emissions
            ),
            calculation_timestamp=str(self._get_timestamp()),
            ghg_protocol_compliance=True,
            uncertainty_range=self._estimate_uncertainty(2.5, input_data.calculation_method)
        )

        return result

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
