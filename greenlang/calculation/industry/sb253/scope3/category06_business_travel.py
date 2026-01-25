# -*- coding: utf-8 -*-
"""
Category 6: Business Travel Calculator

Calculates emissions from transportation of employees for business-related
activities in vehicles not owned or operated by the reporting organization.

Includes:
1. Air travel (short-haul, medium-haul, long-haul)
2. Rail travel
3. Car rental and taxi
4. Hotel stays
5. Other ground transportation

Supported Methods:
1. Distance-based method (passenger-km)
2. Spend-based method (travel expenses)
3. Fuel-based method (actual fuel consumption)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category06BusinessTravelCalculator()
    >>> input_data = BusinessTravelInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     trips=[
    ...         Trip(mode="air", distance_km=5000, passengers=2, cabin_class="economy"),
    ...         Trip(mode="rail", distance_km=300, passengers=1),
    ...         Trip(mode="hotel", nights=5, rooms=2),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator

from greenlang.calculators.sb253.scope3.base import (
    Scope3CategoryCalculator,
    Scope3CalculationInput,
    Scope3CalculationResult,
    CalculationMethod,
    CalculationStep,
    EmissionFactorRecord,
    EmissionFactorSource,
    DataQualityTier,
)

logger = logging.getLogger(__name__)


class Trip(BaseModel):
    """Individual business trip record."""

    # Basic trip info
    mode: str = Field(..., description="Travel mode (air, rail, car, hotel)")
    description: Optional[str] = Field(None, description="Trip description")

    # Distance-based inputs
    distance_km: Optional[Decimal] = Field(None, ge=0, description="Distance in km")
    passengers: int = Field(1, ge=1, description="Number of passengers")

    # Air travel specific
    cabin_class: Optional[str] = Field(None, description="Cabin class (economy, business, first)")
    with_radiative_forcing: bool = Field(
        True, description="Include radiative forcing for air travel"
    )

    # Hotel specific
    nights: Optional[int] = Field(None, ge=0, description="Number of hotel nights")
    rooms: int = Field(1, ge=1, description="Number of rooms")
    hotel_country: Optional[str] = Field(None, description="Hotel country")

    # Car specific
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    fuel_type: Optional[str] = Field(None, description="Fuel type")

    # Spend-based
    spend_usd: Optional[Decimal] = Field(None, ge=0, description="Trip spend in USD")

    # Origin/destination
    origin: Optional[str] = Field(None, description="Origin")
    destination: Optional[str] = Field(None, description="Destination")

    @validator("mode")
    def normalize_mode(cls, v: str) -> str:
        """Normalize travel mode."""
        mode_map = {
            "air": "air",
            "flight": "air",
            "plane": "air",
            "rail": "rail",
            "train": "rail",
            "car": "car",
            "rental": "car",
            "taxi": "taxi",
            "uber": "taxi",
            "lyft": "taxi",
            "rideshare": "taxi",
            "hotel": "hotel",
            "accommodation": "hotel",
            "bus": "bus",
            "coach": "bus",
        }
        normalized = v.lower().strip()
        return mode_map.get(normalized, normalized)

    @validator("cabin_class")
    def normalize_cabin_class(cls, v: Optional[str]) -> Optional[str]:
        """Normalize cabin class."""
        if v is None:
            return None
        class_map = {
            "economy": "economy",
            "coach": "economy",
            "premium_economy": "premium_economy",
            "premium": "premium_economy",
            "business": "business",
            "first": "first",
            "first_class": "first",
        }
        return class_map.get(v.lower().strip(), "economy")


class BusinessTravelInput(Scope3CalculationInput):
    """Input model for Category 6: Business Travel."""

    # Trip data
    trips: List[Trip] = Field(
        default_factory=list, description="List of business trips"
    )

    # Aggregated inputs (alternative)
    total_air_km: Optional[Decimal] = Field(None, ge=0, description="Total air km")
    total_rail_km: Optional[Decimal] = Field(None, ge=0, description="Total rail km")
    total_car_km: Optional[Decimal] = Field(None, ge=0, description="Total car km")
    total_hotel_nights: Optional[int] = Field(None, ge=0, description="Total hotel nights")
    total_travel_spend_usd: Optional[Decimal] = Field(
        None, ge=0, description="Total travel spend"
    )

    # Configuration
    default_cabin_class: str = Field("economy", description="Default cabin class")
    include_radiative_forcing: bool = Field(
        True, description="Include radiative forcing for flights"
    )


# Air travel emission factors (kg CO2e per passenger-km)
# Source: DEFRA 2024
AIR_TRAVEL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "economy": {
        "short_haul": Decimal("0.151"),  # <500km
        "medium_haul": Decimal("0.101"),  # 500-3700km
        "long_haul": Decimal("0.095"),  # >3700km
        "average": Decimal("0.102"),
    },
    "premium_economy": {
        "short_haul": Decimal("0.242"),
        "medium_haul": Decimal("0.162"),
        "long_haul": Decimal("0.152"),
        "average": Decimal("0.163"),
    },
    "business": {
        "short_haul": Decimal("0.439"),
        "medium_haul": Decimal("0.294"),
        "long_haul": Decimal("0.276"),
        "average": Decimal("0.296"),
    },
    "first": {
        "short_haul": Decimal("0.604"),
        "medium_haul": Decimal("0.404"),
        "long_haul": Decimal("0.380"),
        "average": Decimal("0.407"),
    },
}

# Radiative forcing multiplier for aviation
RADIATIVE_FORCING_MULTIPLIER = Decimal("1.9")

# Other travel emission factors
OTHER_TRAVEL_FACTORS: Dict[str, Dict[str, Any]] = {
    "rail": {
        "factor": Decimal("0.035"),  # kg CO2e/passenger-km
        "unit": "passenger-km",
        "description": "Average rail",
    },
    "rail_electric": {
        "factor": Decimal("0.028"),
        "unit": "passenger-km",
    },
    "rail_diesel": {
        "factor": Decimal("0.061"),
        "unit": "passenger-km",
    },
    "car": {
        "factor": Decimal("0.171"),  # Average car
        "unit": "passenger-km",
    },
    "car_small": {
        "factor": Decimal("0.139"),
        "unit": "passenger-km",
    },
    "car_medium": {
        "factor": Decimal("0.171"),
        "unit": "passenger-km",
    },
    "car_large": {
        "factor": Decimal("0.209"),
        "unit": "passenger-km",
    },
    "taxi": {
        "factor": Decimal("0.208"),
        "unit": "passenger-km",
    },
    "bus": {
        "factor": Decimal("0.089"),
        "unit": "passenger-km",
    },
}

# Hotel emission factors (kg CO2e per room-night)
HOTEL_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("31.1"),
    "UK": Decimal("12.0"),
    "EU": Decimal("15.0"),
    "APAC": Decimal("25.0"),
    "default": Decimal("20.0"),
}

# Spend-based travel factors (kg CO2e per USD)
TRAVEL_SPEND_FACTORS: Dict[str, Decimal] = {
    "air": Decimal("0.38"),
    "rail": Decimal("0.15"),
    "car_rental": Decimal("0.25"),
    "hotel": Decimal("0.12"),
    "taxi": Decimal("0.20"),
    "default": Decimal("0.25"),
}


class Category06BusinessTravelCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 6: Business Travel.

    Calculates emissions from employee business travel.

    Attributes:
        CATEGORY_NUMBER: 6
        CATEGORY_NAME: "Business Travel"

    Example:
        >>> calculator = Category06BusinessTravelCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 6
    CATEGORY_NAME = "Business Travel"
    SUPPORTED_METHODS = [
        CalculationMethod.DISTANCE_BASED,
        CalculationMethod.SPEND_BASED,
    ]

    def __init__(self):
        """Initialize the Category 6 calculator."""
        super().__init__()
        self._air_factors = AIR_TRAVEL_FACTORS
        self._travel_factors = OTHER_TRAVEL_FACTORS
        self._hotel_factors = HOTEL_FACTORS
        self._spend_factors = TRAVEL_SPEND_FACTORS

    def calculate(
        self, input_data: BusinessTravelInput
    ) -> Scope3CalculationResult:
        """
        Calculate Category 6 emissions.

        Args:
            input_data: Business travel input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        if input_data.calculation_method == CalculationMethod.SPEND_BASED:
            return self._calculate_spend_based(input_data, start_time)
        else:
            return self._calculate_distance_based(input_data, start_time)

    def _calculate_distance_based(
        self,
        input_data: BusinessTravelInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using distance-based method.

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize distance-based business travel calculation",
            inputs={
                "num_trips": len(input_data.trips),
                "include_radiative_forcing": input_data.include_radiative_forcing,
            },
        ))

        total_air_km = Decimal("0")
        total_rail_km = Decimal("0")
        total_car_km = Decimal("0")
        total_hotel_nights = 0

        if input_data.trips:
            for trip in input_data.trips:
                if trip.mode == "air" and trip.distance_km:
                    emissions = self._calculate_air_emissions(
                        trip, input_data.include_radiative_forcing
                    )
                    total_emissions_kg += emissions
                    total_air_km += trip.distance_km * trip.passengers

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate air travel: {trip.origin or ''}-{trip.destination or 'trip'}",
                        formula="emissions = distance x passengers x factor x RF_multiplier",
                        inputs={
                            "distance_km": str(trip.distance_km),
                            "passengers": trip.passengers,
                            "cabin_class": trip.cabin_class or "economy",
                            "with_rf": trip.with_radiative_forcing,
                        },
                        output=str(emissions),
                    ))

                elif trip.mode == "rail" and trip.distance_km:
                    emissions = self._calculate_rail_emissions(trip)
                    total_emissions_kg += emissions
                    total_rail_km += trip.distance_km * trip.passengers

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate rail travel",
                        formula="emissions = distance x passengers x factor",
                        inputs={
                            "distance_km": str(trip.distance_km),
                            "passengers": trip.passengers,
                        },
                        output=str(emissions),
                    ))

                elif trip.mode in ["car", "taxi"] and trip.distance_km:
                    emissions = self._calculate_car_emissions(trip)
                    total_emissions_kg += emissions
                    total_car_km += trip.distance_km * trip.passengers

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate {trip.mode} travel",
                        inputs={
                            "distance_km": str(trip.distance_km),
                            "passengers": trip.passengers,
                        },
                        output=str(emissions),
                    ))

                elif trip.mode == "hotel" and trip.nights:
                    emissions = self._calculate_hotel_emissions(trip)
                    total_emissions_kg += emissions
                    total_hotel_nights += trip.nights * trip.rooms

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description="Calculate hotel stay emissions",
                        inputs={
                            "nights": trip.nights,
                            "rooms": trip.rooms,
                            "country": trip.hotel_country or "default",
                        },
                        output=str(emissions),
                    ))

        else:
            # Use aggregated inputs
            if input_data.total_air_km:
                factor = self._air_factors["economy"]["average"]
                rf = RADIATIVE_FORCING_MULTIPLIER if input_data.include_radiative_forcing else Decimal("1")
                air_emissions = (input_data.total_air_km * factor * rf).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_emissions_kg += air_emissions
                total_air_km = input_data.total_air_km

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description="Calculate from total air km",
                    inputs={"total_air_km": str(input_data.total_air_km)},
                    output=str(air_emissions),
                ))

            if input_data.total_rail_km:
                factor = self._travel_factors["rail"]["factor"]
                rail_emissions = (input_data.total_rail_km * factor).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_emissions_kg += rail_emissions
                total_rail_km = input_data.total_rail_km

            if input_data.total_car_km:
                factor = self._travel_factors["car"]["factor"]
                car_emissions = (input_data.total_car_km * factor).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_emissions_kg += car_emissions
                total_car_km = input_data.total_car_km

            if input_data.total_hotel_nights:
                factor = self._hotel_factors["default"]
                hotel_emissions = (
                    Decimal(str(input_data.total_hotel_nights)) * factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                total_emissions_kg += hotel_emissions
                total_hotel_nights = input_data.total_hotel_nights

        emission_factor = EmissionFactorRecord(
            factor_id="business_travel_composite",
            factor_value=Decimal("0.102"),  # Economy air average
            factor_unit="kg CO2e/passenger-km",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_air_km": str(total_air_km),
            "total_rail_km": str(total_rail_km),
            "total_car_km": str(total_car_km),
            "total_hotel_nights": total_hotel_nights,
            "num_trips": len(input_data.trips),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.DISTANCE_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_spend_based(
        self,
        input_data: BusinessTravelInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using spend-based method.

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_spend = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize spend-based travel calculation",
        ))

        if input_data.trips:
            for trip in input_data.trips:
                if trip.spend_usd:
                    factor = self._spend_factors.get(
                        trip.mode, self._spend_factors["default"]
                    )
                    trip_emissions = (trip.spend_usd * factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    total_emissions_kg += trip_emissions
                    total_spend += trip.spend_usd

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate from {trip.mode} spend",
                        formula="emissions = spend x factor",
                        inputs={
                            "spend_usd": str(trip.spend_usd),
                            "mode": trip.mode,
                            "factor": str(factor),
                        },
                        output=str(trip_emissions),
                    ))
        elif input_data.total_travel_spend_usd:
            factor = self._spend_factors["default"]
            total_emissions_kg = (
                input_data.total_travel_spend_usd * factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_spend = input_data.total_travel_spend_usd

            steps.append(CalculationStep(
                step_number=2,
                description="Calculate from total travel spend",
                inputs={
                    "total_spend_usd": str(total_spend),
                    "factor": str(factor),
                },
                output=str(total_emissions_kg),
            ))

        emission_factor = EmissionFactorRecord(
            factor_id="business_travel_spend",
            factor_value=Decimal("0.25"),
            factor_unit="kg CO2e/USD",
            source=EmissionFactorSource.EPA_EEIO,
            source_uri="https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=349324",
            version="2023",
            last_updated="2023-01-01",
            data_quality_tier=DataQualityTier.TIER_3,
        )

        activity_data = {
            "total_travel_spend_usd": str(total_spend),
            "num_trips": len(input_data.trips),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.SPEND_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_air_emissions(
        self,
        trip: Trip,
        include_rf: bool,
    ) -> Decimal:
        """Calculate air travel emissions."""
        cabin = trip.cabin_class or "economy"
        distance = trip.distance_km or Decimal("0")

        # Determine haul category
        if distance < 500:
            haul = "short_haul"
        elif distance < 3700:
            haul = "medium_haul"
        else:
            haul = "long_haul"

        factor = self._air_factors.get(cabin, self._air_factors["economy"])[haul]

        # Calculate passenger-km
        pax_km = distance * trip.passengers

        # Apply radiative forcing if requested
        rf_multiplier = RADIATIVE_FORCING_MULTIPLIER if (include_rf and trip.with_radiative_forcing) else Decimal("1")

        emissions = (pax_km * factor * rf_multiplier).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        return emissions

    def _calculate_rail_emissions(self, trip: Trip) -> Decimal:
        """Calculate rail travel emissions."""
        factor = self._travel_factors["rail"]["factor"]
        pax_km = (trip.distance_km or Decimal("0")) * trip.passengers

        return (pax_km * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_car_emissions(self, trip: Trip) -> Decimal:
        """Calculate car/taxi travel emissions."""
        mode_key = trip.mode if trip.mode in self._travel_factors else "car"
        factor = self._travel_factors[mode_key]["factor"]
        pax_km = (trip.distance_km or Decimal("0")) * trip.passengers

        return (pax_km * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_hotel_emissions(self, trip: Trip) -> Decimal:
        """Calculate hotel stay emissions."""
        country = trip.hotel_country or "default"
        factor = self._hotel_factors.get(country, self._hotel_factors["default"])
        room_nights = (trip.nights or 0) * trip.rooms

        return (Decimal(str(room_nights)) * factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
