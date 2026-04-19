# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-008: Business Travel MRV Agent
=========================================

This module implements the Business Travel MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from employee business travel.

Supported Features:
- Air travel (multiple cabin classes)
- Rail travel (national/international)
- Road travel (car rental, taxi, personal vehicle)
- Hotel stays
- Employee commuting (optional)
- Travel booking integration

Reference Standards:
- GHG Protocol Scope 3, Category 6 (Business Travel)
- GHG Protocol Scope 3, Category 7 (Employee Commuting)
- DEFRA Conversion Factors 2024
- ICAO Carbon Emissions Calculator

Example:
    >>> agent = BusinessTravelMRVAgent()
    >>> input_data = BusinessTravelInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     travel_records=[
    ...         TravelRecord(
    ...             travel_type=TravelType.AIR,
    ...             origin="LHR",
    ...             destination="JFK",
    ...             cabin_class=CabinClass.BUSINESS,
    ...             passengers=1,
    ...         )
    ...     ]
    ... )
    >>> result = agent.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field

from greenlang.agents.mrv.transport.base import (
    BaseTransportMRVAgent,
    TransportMRVInput,
    TransportMRVOutput,
    TransportMode,
    EmissionScope,
    DataQualityTier,
    EmissionFactor,
    EmissionFactorSource,
    CalculationStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Business Travel Enums
# =============================================================================

class TravelType(str, Enum):
    """Types of business travel."""
    AIR = "air"
    RAIL = "rail"
    CAR_RENTAL = "car_rental"
    TAXI = "taxi"
    PERSONAL_VEHICLE = "personal_vehicle"
    BUS = "bus"
    FERRY = "ferry"
    HOTEL = "hotel"


class CabinClass(str, Enum):
    """Cabin class for air travel."""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class RailClass(str, Enum):
    """Rail travel types."""
    NATIONAL = "national"
    INTERNATIONAL = "international"
    HIGH_SPEED = "high_speed"
    METRO = "metro"


class CarType(str, Enum):
    """Car types for road travel."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    SUV = "suv"
    ELECTRIC = "electric"
    HYBRID = "hybrid"


class HotelClass(str, Enum):
    """Hotel class for accommodation."""
    BUDGET = "budget"
    STANDARD = "standard"
    UPSCALE = "upscale"
    LUXURY = "luxury"


# =============================================================================
# DEFRA 2024 Business Travel Emission Factors
# =============================================================================

# Air travel (kg CO2e per passenger-km) - with radiative forcing
AIR_TRAVEL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "domestic": {
        CabinClass.ECONOMY.value: Decimal("0.24587"),
        CabinClass.BUSINESS.value: Decimal("0.24587"),
    },
    "short_haul": {
        CabinClass.ECONOMY.value: Decimal("0.15102"),
        CabinClass.PREMIUM_ECONOMY.value: Decimal("0.24164"),
        CabinClass.BUSINESS.value: Decimal("0.45306"),
        CabinClass.FIRST.value: Decimal("0.60408"),
    },
    "long_haul": {
        CabinClass.ECONOMY.value: Decimal("0.14787"),
        CabinClass.PREMIUM_ECONOMY.value: Decimal("0.23660"),
        CabinClass.BUSINESS.value: Decimal("0.42883"),
        CabinClass.FIRST.value: Decimal("0.59148"),
    },
}

# Rail travel (kg CO2e per passenger-km)
RAIL_TRAVEL_FACTORS: Dict[str, Decimal] = {
    RailClass.NATIONAL.value: Decimal("0.03549"),
    RailClass.INTERNATIONAL.value: Decimal("0.00446"),
    RailClass.HIGH_SPEED.value: Decimal("0.00446"),
    RailClass.METRO.value: Decimal("0.02781"),
}

# Car travel (kg CO2e per km)
CAR_TRAVEL_FACTORS: Dict[str, Decimal] = {
    CarType.SMALL.value: Decimal("0.14901"),
    CarType.MEDIUM.value: Decimal("0.17049"),
    CarType.LARGE.value: Decimal("0.21016"),
    CarType.SUV.value: Decimal("0.21016"),
    CarType.ELECTRIC.value: Decimal("0.05297"),
    CarType.HYBRID.value: Decimal("0.11716"),
}

# Taxi (kg CO2e per km)
TAXI_FACTOR = Decimal("0.14826")

# Bus (kg CO2e per passenger-km)
BUS_FACTOR = Decimal("0.10231")

# Hotel (kg CO2e per night)
HOTEL_FACTORS: Dict[str, Decimal] = {
    HotelClass.BUDGET.value: Decimal("15.2"),
    HotelClass.STANDARD.value: Decimal("25.4"),
    HotelClass.UPSCALE.value: Decimal("38.1"),
    HotelClass.LUXURY.value: Decimal("52.6"),
}


# =============================================================================
# Input Models
# =============================================================================

class TravelRecord(BaseModel):
    """Individual business travel record."""

    # Travel identification
    trip_id: Optional[str] = Field(None, description="Trip identifier")
    employee_id: Optional[str] = Field(None, description="Employee ID")
    date: Optional[str] = Field(None, description="Travel date")

    # Travel type
    travel_type: TravelType = Field(..., description="Type of travel")

    # Route information
    origin: Optional[str] = Field(None, description="Origin")
    destination: Optional[str] = Field(None, description="Destination")
    distance_km: Optional[Decimal] = Field(None, ge=0, description="Distance in km")

    # Air travel specific
    cabin_class: CabinClass = Field(
        CabinClass.ECONOMY, description="Cabin class for air travel"
    )
    include_radiative_forcing: bool = Field(
        True, description="Include RF multiplier for air"
    )

    # Rail travel specific
    rail_class: RailClass = Field(
        RailClass.NATIONAL, description="Rail travel type"
    )

    # Car travel specific
    car_type: CarType = Field(
        CarType.MEDIUM, description="Car type for road travel"
    )

    # Hotel specific
    hotel_nights: int = Field(0, ge=0, description="Number of hotel nights")
    hotel_class: HotelClass = Field(
        HotelClass.STANDARD, description="Hotel class"
    )

    # Passengers/trips
    passengers: int = Field(1, ge=1, description="Number of travelers")
    is_return_trip: bool = Field(False, description="Is return trip")

    class Config:
        use_enum_values = True


class BusinessTravelInput(TransportMRVInput):
    """Input model for Business Travel MRV Agent."""

    # Travel records
    travel_records: List[TravelRecord] = Field(
        default_factory=list, description="Individual travel records"
    )

    # Aggregated data
    total_air_passenger_km: Optional[Decimal] = Field(
        None, ge=0, description="Total air passenger-km"
    )
    total_rail_passenger_km: Optional[Decimal] = Field(
        None, ge=0, description="Total rail passenger-km"
    )
    total_car_km: Optional[Decimal] = Field(
        None, ge=0, description="Total car km"
    )
    total_hotel_nights: Optional[int] = Field(
        None, ge=0, description="Total hotel nights"
    )

    # Defaults for aggregated data
    default_cabin_class: CabinClass = Field(
        CabinClass.ECONOMY, description="Default cabin class"
    )
    default_car_type: CarType = Field(
        CarType.MEDIUM, description="Default car type"
    )
    default_hotel_class: HotelClass = Field(
        HotelClass.STANDARD, description="Default hotel class"
    )

    # Options
    include_radiative_forcing: bool = Field(
        True, description="Include radiative forcing for air travel"
    )
    include_hotel: bool = Field(
        True, description="Include hotel emissions"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Output Model
# =============================================================================

class BusinessTravelOutput(TransportMRVOutput):
    """Output model for Business Travel MRV Agent."""

    # Travel metrics
    total_trips: int = Field(0, ge=0, description="Total number of trips")
    total_employees: int = Field(0, ge=0, description="Unique employees")

    # Distance metrics
    total_air_passenger_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total air passenger-km"
    )
    total_rail_passenger_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total rail passenger-km"
    )
    total_car_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total car km"
    )
    total_hotel_nights: int = Field(0, ge=0, description="Total hotel nights")

    # Emission breakdown
    air_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Air travel emissions"
    )
    rail_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Rail travel emissions"
    )
    car_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Car travel emissions"
    )
    hotel_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Hotel emissions"
    )
    other_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Other transport emissions"
    )

    # Efficiency
    emissions_per_trip: Optional[Decimal] = Field(
        None, description="Average kg CO2e per trip"
    )
    emissions_per_employee: Optional[Decimal] = Field(
        None, description="Average kg CO2e per employee"
    )

    # Breakdown by travel type
    emissions_by_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by travel type"
    )


# =============================================================================
# Business Travel MRV Agent
# =============================================================================

class BusinessTravelMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-008: Business Travel MRV Agent

    Calculates greenhouse gas emissions from employee business travel.

    Key Features:
    - Air travel with cabin class weighting
    - Rail travel (national/international)
    - Car rental and taxi
    - Hotel accommodation
    - Radiative forcing for aviation

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance
    """

    AGENT_ID = "GL-MRV-TRN-008"
    AGENT_NAME = "Business Travel MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.INTERMODAL
    DEFAULT_SCOPE = EmissionScope.SCOPE_3

    def calculate(self, input_data: BusinessTravelInput) -> BusinessTravelOutput:
        """
        Calculate business travel emissions.

        Args:
            input_data: Business travel input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Initialize totals
        total_emissions_kg = Decimal("0")
        total_air_pax_km = Decimal("0")
        total_rail_pax_km = Decimal("0")
        total_car_km = Decimal("0")
        total_hotel_nights = 0
        air_emissions = Decimal("0")
        rail_emissions = Decimal("0")
        car_emissions = Decimal("0")
        hotel_emissions = Decimal("0")
        other_emissions = Decimal("0")
        emissions_by_type: Dict[str, Decimal] = {}
        employee_ids: set = set()

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize business travel emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "num_records": len(input_data.travel_records),
                "include_rf": input_data.include_radiative_forcing,
            },
        ))

        # Process individual travel records
        for record in input_data.travel_records:
            result = self._calculate_travel_emissions(
                record=record,
                include_rf=input_data.include_radiative_forcing,
                include_hotel=input_data.include_hotel,
                step_offset=len(steps),
            )

            steps.extend(result["steps"])
            emission_factors.extend(result["factors"])

            total_emissions_kg += result["total_kg"]

            # Track by travel type
            ttype = record.travel_type.value if hasattr(record.travel_type, 'value') else str(record.travel_type)
            emissions_by_type[ttype] = emissions_by_type.get(
                ttype, Decimal("0")
            ) + result["travel_kg"]

            # Accumulate by category
            if record.travel_type == TravelType.AIR:
                air_emissions += result["travel_kg"]
                total_air_pax_km += result["passenger_km"]
            elif record.travel_type == TravelType.RAIL:
                rail_emissions += result["travel_kg"]
                total_rail_pax_km += result["passenger_km"]
            elif record.travel_type in [TravelType.CAR_RENTAL, TravelType.PERSONAL_VEHICLE, TravelType.TAXI]:
                car_emissions += result["travel_kg"]
                total_car_km += result["distance_km"]
            else:
                other_emissions += result["travel_kg"]

            hotel_emissions += result["hotel_kg"]
            total_hotel_nights += record.hotel_nights

            if record.employee_id:
                employee_ids.add(record.employee_id)

        # Process aggregated data
        if input_data.total_air_passenger_km and not input_data.travel_records:
            cabin = input_data.default_cabin_class.value if hasattr(input_data.default_cabin_class, 'value') else str(input_data.default_cabin_class)
            # Use short-haul as default
            factor = AIR_TRAVEL_FACTORS["short_haul"].get(
                cabin, AIR_TRAVEL_FACTORS["short_haul"][CabinClass.ECONOMY.value]
            )
            agg_air = (input_data.total_air_passenger_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += agg_air
            air_emissions += agg_air
            total_air_pax_km += input_data.total_air_passenger_km

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate aggregated air travel emissions",
                inputs={
                    "total_passenger_km": str(input_data.total_air_passenger_km),
                    "cabin_class": cabin,
                },
                output=str(agg_air),
            ))

        if input_data.total_rail_passenger_km and not input_data.travel_records:
            factor = RAIL_TRAVEL_FACTORS[RailClass.NATIONAL.value]
            agg_rail = (input_data.total_rail_passenger_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += agg_rail
            rail_emissions += agg_rail
            total_rail_pax_km += input_data.total_rail_passenger_km

        if input_data.total_car_km and not input_data.travel_records:
            ctype = input_data.default_car_type.value if hasattr(input_data.default_car_type, 'value') else str(input_data.default_car_type)
            factor = CAR_TRAVEL_FACTORS.get(
                ctype, CAR_TRAVEL_FACTORS[CarType.MEDIUM.value]
            )
            agg_car = (input_data.total_car_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += agg_car
            car_emissions += agg_car
            total_car_km += input_data.total_car_km

        if input_data.total_hotel_nights and not input_data.travel_records:
            hclass = input_data.default_hotel_class.value if hasattr(input_data.default_hotel_class, 'value') else str(input_data.default_hotel_class)
            factor = HOTEL_FACTORS.get(
                hclass, HOTEL_FACTORS[HotelClass.STANDARD.value]
            )
            agg_hotel = (Decimal(str(input_data.total_hotel_nights)) * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += agg_hotel
            hotel_emissions += agg_hotel
            total_hotel_nights += input_data.total_hotel_nights

        # Calculate efficiency metrics
        total_trips = len(input_data.travel_records)
        total_employees = len(employee_ids) if employee_ids else 0

        emissions_per_trip = None
        emissions_per_employee = None

        if total_trips > 0:
            emissions_per_trip = (total_emissions_kg / Decimal(str(total_trips))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        if total_employees > 0:
            emissions_per_employee = (total_emissions_kg / Decimal(str(total_employees))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Final summary
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total business travel emissions",
            formula="total = air + rail + car + hotel + other",
            inputs={
                "air_emissions_kg": str(air_emissions),
                "rail_emissions_kg": str(rail_emissions),
                "car_emissions_kg": str(car_emissions),
                "hotel_emissions_kg": str(hotel_emissions),
            },
            output=str(total_emissions_kg),
        ))

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "total_trips": total_trips,
            "total_employees": total_employees,
            "total_air_pax_km": str(total_air_pax_km),
            "total_rail_pax_km": str(total_rail_pax_km),
            "total_car_km": str(total_car_km),
            "total_hotel_nights": total_hotel_nights,
        }

        # Create base output
        base_output = self._create_output(
            total_emissions_kg=total_emissions_kg,
            co2_kg=total_emissions_kg,
            ch4_kg=Decimal("0"),
            n2o_kg=Decimal("0"),
            steps=steps,
            emission_factors=emission_factors,
            activity_summary=activity_summary,
            start_time=start_time,
            scope=EmissionScope.SCOPE_3,
            warnings=warnings,
        )

        return BusinessTravelOutput(
            **base_output.dict(),
            total_trips=total_trips,
            total_employees=total_employees,
            total_air_passenger_km=total_air_pax_km,
            total_rail_passenger_km=total_rail_pax_km,
            total_car_km=total_car_km,
            total_hotel_nights=total_hotel_nights,
            air_emissions_kg=air_emissions,
            rail_emissions_kg=rail_emissions,
            car_emissions_kg=car_emissions,
            hotel_emissions_kg=hotel_emissions,
            other_emissions_kg=other_emissions,
            emissions_per_trip=emissions_per_trip,
            emissions_per_employee=emissions_per_employee,
            emissions_by_type=emissions_by_type,
        )

    def _calculate_travel_emissions(
        self,
        record: TravelRecord,
        include_rf: bool,
        include_hotel: bool,
        step_offset: int,
    ) -> Dict[str, Any]:
        """Calculate emissions for a single travel record."""
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        travel_kg = Decimal("0")
        hotel_kg = Decimal("0")
        passenger_km = Decimal("0")
        distance_km = record.distance_km or Decimal("0")

        # Return trip multiplier
        multiplier = Decimal("2") if record.is_return_trip else Decimal("1")

        ttype = record.travel_type.value if hasattr(record.travel_type, 'value') else str(record.travel_type)

        if record.travel_type == TravelType.AIR:
            # Determine flight type by distance
            if distance_km and distance_km < 500:
                flight_type = "domestic"
            elif distance_km and distance_km < 3700:
                flight_type = "short_haul"
            else:
                flight_type = "long_haul"

            cabin = record.cabin_class.value if hasattr(record.cabin_class, 'value') else str(record.cabin_class)
            factor = AIR_TRAVEL_FACTORS.get(flight_type, {}).get(
                cabin, AIR_TRAVEL_FACTORS["short_haul"][CabinClass.ECONOMY.value]
            )

            passenger_km = distance_km * Decimal(str(record.passengers)) * multiplier
            travel_kg = (passenger_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate air travel: {record.origin or 'Origin'} to {record.destination or 'Destination'}",
                formula="emissions = passenger_km x EF",
                inputs={
                    "flight_type": flight_type,
                    "cabin_class": cabin,
                    "distance_km": str(distance_km),
                    "passengers": record.passengers,
                    "passenger_km": str(passenger_km),
                    "emission_factor": str(factor),
                },
                output=str(travel_kg),
            ))

        elif record.travel_type == TravelType.RAIL:
            rclass = record.rail_class.value if hasattr(record.rail_class, 'value') else str(record.rail_class)
            factor = RAIL_TRAVEL_FACTORS.get(
                rclass, RAIL_TRAVEL_FACTORS[RailClass.NATIONAL.value]
            )

            passenger_km = distance_km * Decimal(str(record.passengers)) * multiplier
            travel_kg = (passenger_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate rail travel: {record.origin or 'Origin'} to {record.destination or 'Destination'}",
                inputs={
                    "rail_class": rclass,
                    "passenger_km": str(passenger_km),
                    "emission_factor": str(factor),
                },
                output=str(travel_kg),
            ))

        elif record.travel_type in [TravelType.CAR_RENTAL, TravelType.PERSONAL_VEHICLE]:
            ctype = record.car_type.value if hasattr(record.car_type, 'value') else str(record.car_type)
            factor = CAR_TRAVEL_FACTORS.get(
                ctype, CAR_TRAVEL_FACTORS[CarType.MEDIUM.value]
            )

            adjusted_distance = distance_km * multiplier
            travel_kg = (adjusted_distance * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate car travel: {record.origin or 'Origin'} to {record.destination or 'Destination'}",
                inputs={
                    "car_type": ctype,
                    "distance_km": str(adjusted_distance),
                    "emission_factor": str(factor),
                },
                output=str(travel_kg),
            ))

        elif record.travel_type == TravelType.TAXI:
            adjusted_distance = distance_km * multiplier
            travel_kg = (adjusted_distance * TAXI_FACTOR).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        elif record.travel_type == TravelType.BUS:
            passenger_km = distance_km * Decimal(str(record.passengers)) * multiplier
            travel_kg = (passenger_km * BUS_FACTOR).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Calculate hotel emissions
        if include_hotel and record.hotel_nights > 0:
            hclass = record.hotel_class.value if hasattr(record.hotel_class, 'value') else str(record.hotel_class)
            hotel_factor = HOTEL_FACTORS.get(
                hclass, HOTEL_FACTORS[HotelClass.STANDARD.value]
            )
            hotel_kg = (Decimal(str(record.hotel_nights)) * hotel_factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            steps.append(CalculationStep(
                step_number=step_offset + 2,
                description="Calculate hotel emissions",
                inputs={
                    "hotel_nights": record.hotel_nights,
                    "hotel_class": hclass,
                    "factor_per_night": str(hotel_factor),
                },
                output=str(hotel_kg),
            ))

        total_kg = travel_kg + hotel_kg

        return {
            "total_kg": total_kg,
            "travel_kg": travel_kg,
            "hotel_kg": hotel_kg,
            "passenger_km": passenger_km,
            "distance_km": distance_km * multiplier,
            "steps": steps,
            "factors": factors,
        }
