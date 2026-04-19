# -*- coding: utf-8 -*-
"""
Travel Calculator - Business Travel Emissions Calculations
GL-VCCI Scope 3 Platform

Provides specialized calculations for Category 6 (Business Travel):
- Flight emissions (with radiative forcing)
- Hotel stays
- Ground transportation
- Complete trip aggregation

Features:
- Class-based flight emission factors
- DEFRA radiative forcing (1.9x for flights)
- Regional hotel emission factors
- Multi-modal trip support

Version: 1.0.0
Date: 2025-11-08
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..config import CabinClass, FLIGHT_EMISSION_FACTORS, HOTEL_EMISSION_FACTORS
from ..models import CalculationResult, DataQualityInfo
from ..exceptions import DataValidationError

logger = logging.getLogger(__name__)


class TravelCalculator:
    """
    Business travel emissions calculator.

    Implements comprehensive travel emission calculations including:
    1. Flight emissions with cabin class differentiation
    2. Hotel stay emissions by region
    3. Ground transportation emissions
    4. Multi-leg trip aggregation

    Features:
    - Radiative forcing for flights (DEFRA: 1.9x)
    - Class-based emission factors (economy, business, first)
    - Regional hotel factors
    - Ground transport mode variety
    """

    # Default flight emission factors (kgCO2e/passenger-km) by cabin class
    DEFAULT_FLIGHT_FACTORS = {
        CabinClass.ECONOMY: 0.115,
        CabinClass.PREMIUM_ECONOMY: 0.165,
        CabinClass.BUSINESS: 0.230,
        CabinClass.FIRST: 0.345,
    }

    # Default hotel emission factors (kgCO2e/night)
    DEFAULT_HOTEL_FACTORS = {
        "Global": 20.0,
        "North America": 22.0,
        "Europe": 18.0,
        "Asia": 25.0,
        "Middle East": 30.0,
        "Africa": 15.0,
        "Latin America": 17.0,
        "Oceania": 19.0,
    }

    # Ground transport emission factors (kgCO2e/km)
    GROUND_TRANSPORT_FACTORS = {
        "car_small": 0.105,
        "car_medium": 0.145,
        "car_large": 0.198,
        "car_electric": 0.045,
        "taxi": 0.150,
        "rental_car": 0.145,
        "bus": 0.028,
        "train": 0.041,
        "subway": 0.030,
        "tram": 0.035,
        "rideshare": 0.140,
    }

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize travel calculator.

        Args:
            config: Calculator configuration
        """
        self.config = config
        self.radiative_forcing_factor = getattr(
            config, 'category_6_radiative_forcing_factor', 1.9
        ) if config else 1.9
        logger.info(
            f"Initialized TravelCalculator (RF factor: {self.radiative_forcing_factor})"
        )

    async def calculate_flight_emissions(
        self,
        distance_km: float,
        num_passengers: int = 1,
        cabin_class: CabinClass = CabinClass.ECONOMY,
        emission_factor: Optional[float] = None,
        apply_radiative_forcing: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate flight emissions with radiative forcing.

        Formula:
            emissions = distance × passengers × EF × radiative_forcing

        Args:
            distance_km: Flight distance in kilometers
            num_passengers: Number of passengers
            cabin_class: Cabin class (affects emission factor)
            emission_factor: Optional custom emission factor (kgCO2e/passenger-km)
            apply_radiative_forcing: Apply radiative forcing multiplier (default True)

        Returns:
            Dict with emissions and calculation details
        """
        # Validate inputs
        if distance_km <= 0:
            raise DataValidationError(
                field="distance_km",
                value=distance_km,
                reason="Flight distance must be positive",
                category=6
            )

        if num_passengers < 1:
            raise DataValidationError(
                field="num_passengers",
                value=num_passengers,
                reason="Number of passengers must be at least 1",
                category=6
            )

        # Get emission factor
        ef = emission_factor or self._get_flight_emission_factor(cabin_class)

        # Calculate base emissions
        base_emissions = distance_km * num_passengers * ef

        # Apply radiative forcing if requested
        rf_factor = 1.0
        if apply_radiative_forcing:
            rf_factor = self.radiative_forcing_factor
            total_emissions = base_emissions * rf_factor
        else:
            total_emissions = base_emissions

        logger.info(
            f"Flight: {distance_km} km × {num_passengers} pax × {ef} kgCO2e/pax-km "
            f"× RF{rf_factor} = {total_emissions:.2f} kgCO2e"
        )

        return {
            "emissions_kgco2e": total_emissions,
            "emissions_tco2e": total_emissions / 1000,
            "base_emissions_kgco2e": base_emissions,
            "distance_km": distance_km,
            "num_passengers": num_passengers,
            "cabin_class": cabin_class.value,
            "emission_factor": ef,
            "radiative_forcing_applied": apply_radiative_forcing,
            "radiative_forcing_factor": rf_factor if apply_radiative_forcing else None,
            "passenger_km": distance_km * num_passengers,
        }

    async def calculate_hotel_emissions(
        self,
        nights: int,
        region: str = "Global",
        emission_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate hotel stay emissions.

        Formula:
            emissions = nights × EF

        Args:
            nights: Number of nights
            region: Geographic region (affects emission factor)
            emission_factor: Optional custom emission factor (kgCO2e/night)

        Returns:
            Dict with emissions and calculation details
        """
        # Validate inputs
        if nights < 1:
            raise DataValidationError(
                field="nights",
                value=nights,
                reason="Number of nights must be at least 1",
                category=6
            )

        # Get emission factor
        ef = emission_factor or self._get_hotel_emission_factor(region)

        # Calculate emissions
        emissions = nights * ef

        logger.info(
            f"Hotel: {nights} nights × {ef} kgCO2e/night = {emissions:.2f} kgCO2e"
        )

        return {
            "emissions_kgco2e": emissions,
            "emissions_tco2e": emissions / 1000,
            "nights": nights,
            "region": region,
            "emission_factor": ef,
        }

    async def calculate_ground_transport_emissions(
        self,
        distance_km: float,
        vehicle_type: str = "car_medium",
        emission_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate ground transportation emissions.

        Formula:
            emissions = distance × EF

        Args:
            distance_km: Travel distance in kilometers
            vehicle_type: Type of vehicle/transport mode
            emission_factor: Optional custom emission factor (kgCO2e/km)

        Returns:
            Dict with emissions and calculation details
        """
        # Validate inputs
        if distance_km < 0:
            raise DataValidationError(
                field="distance_km",
                value=distance_km,
                reason="Distance cannot be negative",
                category=6
            )

        # Get emission factor
        ef = emission_factor or self._get_ground_transport_factor(vehicle_type)

        # Calculate emissions
        emissions = distance_km * ef

        logger.info(
            f"Ground transport: {distance_km} km × {ef} kgCO2e/km = {emissions:.2f} kgCO2e"
        )

        return {
            "emissions_kgco2e": emissions,
            "emissions_tco2e": emissions / 1000,
            "distance_km": distance_km,
            "vehicle_type": vehicle_type,
            "emission_factor": ef,
        }

    async def calculate_complete_trip(
        self,
        flights: Optional[List[Dict[str, Any]]] = None,
        hotels: Optional[List[Dict[str, Any]]] = None,
        ground_transports: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate total emissions for complete business trip.

        Args:
            flights: List of flight dictionaries with keys:
                - distance_km
                - num_passengers (optional, default 1)
                - cabin_class (optional, default ECONOMY)
                - emission_factor (optional)
                - apply_radiative_forcing (optional, default True)
            hotels: List of hotel dictionaries with keys:
                - nights
                - region (optional, default "Global")
                - emission_factor (optional)
            ground_transports: List of ground transport dictionaries with keys:
                - distance_km
                - vehicle_type (optional, default "car_medium")
                - emission_factor (optional)

        Returns:
            Dict with total emissions and component breakdown
        """
        total_emissions = 0.0
        components = []

        # Calculate flight emissions
        if flights:
            flight_emissions, flight_details = await self._calculate_all_flights(flights)
            total_emissions += flight_emissions
            components.append({
                "type": "flights",
                "emissions_kgco2e": flight_emissions,
                "count": len(flights),
                "details": flight_details,
            })

        # Calculate hotel emissions
        if hotels:
            hotel_emissions, hotel_details = await self._calculate_all_hotels(hotels)
            total_emissions += hotel_emissions
            components.append({
                "type": "hotels",
                "emissions_kgco2e": hotel_emissions,
                "count": len(hotels),
                "details": hotel_details,
            })

        # Calculate ground transport emissions
        if ground_transports:
            ground_emissions, ground_details = await self._calculate_all_ground_transport(
                ground_transports
            )
            total_emissions += ground_emissions
            components.append({
                "type": "ground_transport",
                "emissions_kgco2e": ground_emissions,
                "count": len(ground_transports),
                "details": ground_details,
            })

        return {
            "total_emissions_kgco2e": total_emissions,
            "total_emissions_tco2e": total_emissions / 1000,
            "components": components,
            "num_flights": len(flights) if flights else 0,
            "num_hotel_nights": sum(h.get('nights', 0) for h in hotels) if hotels else 0,
            "num_ground_trips": len(ground_transports) if ground_transports else 0,
        }

    async def calculate_round_trip_flight(
        self,
        distance_km: float,
        num_passengers: int = 1,
        cabin_class: CabinClass = CabinClass.ECONOMY,
        apply_radiative_forcing: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate round-trip flight emissions.

        Args:
            distance_km: One-way distance
            num_passengers: Number of passengers
            cabin_class: Cabin class
            apply_radiative_forcing: Apply radiative forcing

        Returns:
            Dict with round-trip emissions
        """
        one_way = await self.calculate_flight_emissions(
            distance_km=distance_km,
            num_passengers=num_passengers,
            cabin_class=cabin_class,
            apply_radiative_forcing=apply_radiative_forcing,
        )

        total_emissions = one_way['emissions_kgco2e'] * 2

        return {
            "total_emissions_kgco2e": total_emissions,
            "total_emissions_tco2e": total_emissions / 1000,
            "one_way_emissions_kgco2e": one_way['emissions_kgco2e'],
            "total_distance_km": distance_km * 2,
            "num_passengers": num_passengers,
            "cabin_class": cabin_class.value,
            "is_round_trip": True,
            "radiative_forcing_applied": apply_radiative_forcing,
        }

    async def _calculate_all_flights(
        self, flights: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate emissions from all flights."""
        total = 0.0
        details = []

        for flight in flights:
            result = await self.calculate_flight_emissions(
                distance_km=flight['distance_km'],
                num_passengers=flight.get('num_passengers', 1),
                cabin_class=flight.get('cabin_class', CabinClass.ECONOMY),
                emission_factor=flight.get('emission_factor'),
                apply_radiative_forcing=flight.get('apply_radiative_forcing', True),
            )
            total += result['emissions_kgco2e']
            details.append(result)

        return total, details

    async def _calculate_all_hotels(
        self, hotels: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate emissions from all hotel stays."""
        total = 0.0
        details = []

        for hotel in hotels:
            result = await self.calculate_hotel_emissions(
                nights=hotel['nights'],
                region=hotel.get('region', 'Global'),
                emission_factor=hotel.get('emission_factor'),
            )
            total += result['emissions_kgco2e']
            details.append(result)

        return total, details

    async def _calculate_all_ground_transport(
        self, transports: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculate emissions from all ground transportation."""
        total = 0.0
        details = []

        for transport in transports:
            result = await self.calculate_ground_transport_emissions(
                distance_km=transport['distance_km'],
                vehicle_type=transport.get('vehicle_type', 'car_medium'),
                emission_factor=transport.get('emission_factor'),
            )
            total += result['emissions_kgco2e']
            details.append(result)

        return total, details

    def _get_flight_emission_factor(self, cabin_class: CabinClass) -> float:
        """Get flight emission factor for cabin class."""
        # Try config first if available
        if self.config and hasattr(self.config, 'FLIGHT_EMISSION_FACTORS'):
            config_factors = getattr(self.config, 'FLIGHT_EMISSION_FACTORS')
            if cabin_class in config_factors:
                return config_factors[cabin_class]

        # Try global constant
        if cabin_class in FLIGHT_EMISSION_FACTORS:
            return FLIGHT_EMISSION_FACTORS[cabin_class]

        # Fall back to default
        return self.DEFAULT_FLIGHT_FACTORS.get(cabin_class, 0.115)

    def _get_hotel_emission_factor(self, region: str) -> float:
        """Get hotel emission factor for region."""
        # Try config first if available
        if self.config and hasattr(self.config, 'HOTEL_EMISSION_FACTORS'):
            config_factors = getattr(self.config, 'HOTEL_EMISSION_FACTORS')
            if region in config_factors:
                return config_factors[region]

        # Try global constant
        if region in HOTEL_EMISSION_FACTORS:
            return HOTEL_EMISSION_FACTORS[region]

        # Fall back to default
        return self.DEFAULT_HOTEL_FACTORS.get(region, 20.0)

    def _get_ground_transport_factor(self, vehicle_type: str) -> float:
        """Get ground transport emission factor."""
        return self.GROUND_TRANSPORT_FACTORS.get(vehicle_type, 0.145)

    def get_cabin_class_multiplier(self, cabin_class: CabinClass) -> float:
        """
        Get multiplier showing how much more emissions a cabin class produces
        compared to economy.

        Args:
            cabin_class: Cabin class

        Returns:
            Multiplier relative to economy
        """
        economy_ef = self._get_flight_emission_factor(CabinClass.ECONOMY)
        class_ef = self._get_flight_emission_factor(cabin_class)
        return class_ef / economy_ef

    def estimate_flight_distance(
        self, origin: str, destination: str
    ) -> Optional[float]:
        """
        Estimate flight distance between airports (placeholder).

        In production, this would use airport code database or API.

        Args:
            origin: Origin airport code
            destination: Destination airport code

        Returns:
            Estimated distance in km, or None if cannot estimate
        """
        # Placeholder - in production would use airport database
        logger.warning(
            "Flight distance estimation not implemented. "
            "Please provide distance_km directly."
        )
        return None


__all__ = ["TravelCalculator"]
