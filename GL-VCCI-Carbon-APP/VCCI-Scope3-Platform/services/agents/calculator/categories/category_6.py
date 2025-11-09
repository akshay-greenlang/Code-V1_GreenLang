"""
Category 6: Business Travel Calculator
GL-VCCI Scope 3 Platform

Covers three main types of business travel emissions:
1. Flights (with radiative forcing)
2. Hotels
3. Ground transport

Version: 1.0.0
Date: 2025-10-30
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models import (
    Category6Input,
    Category6FlightInput,
    Category6HotelInput,
    Category6GroundTransportInput,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
)
from ..config import (
    TierType,
    CabinClass,
    get_config,
    FLIGHT_EMISSION_FACTORS,
    HOTEL_EMISSION_FACTORS,
)
from ..exceptions import DataValidationError, CalculationError

logger = logging.getLogger(__name__)


class Category6Calculator:
    """
    Category 6 (Business Travel) calculator.

    Components:
    - Flights: distance × passengers × EF × radiative_forcing
    - Hotels: nights × EF
    - Ground transport: distance × EF

    Features:
    - Radiative forcing for flights (DEFRA: 1.9)
    - Multi-leg trip support
    - Regional hotel factors
    - Complete provenance tracking
    """

    def __init__(
        self,
        factor_broker: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        config: Optional[Any] = None
    ):
        """Initialize Category 6 calculator."""
        self.factor_broker = factor_broker
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()
        logger.info("Initialized Category6Calculator")

    async def calculate(self, input_data: Category6Input) -> CalculationResult:
        """
        Calculate Category 6 emissions for complete trip.

        Args:
            input_data: Category 6 input with flights, hotels, ground transport

        Returns:
            CalculationResult with total emissions and uncertainty
        """
        total_emissions = 0.0
        components = []
        warnings = []

        # Calculate flight emissions
        if input_data.flights:
            flight_emissions, flight_details = await self._calculate_flights(
                input_data.flights
            )
            total_emissions += flight_emissions
            components.append({
                "type": "flights",
                "emissions_kgco2e": flight_emissions,
                "count": len(input_data.flights),
                "details": flight_details
            })

        # Calculate hotel emissions
        if input_data.hotels and self.config.category_6_include_hotel_emissions:
            hotel_emissions, hotel_details = await self._calculate_hotels(
                input_data.hotels
            )
            total_emissions += hotel_emissions
            components.append({
                "type": "hotels",
                "emissions_kgco2e": hotel_emissions,
                "count": len(input_data.hotels),
                "details": hotel_details
            })

        # Calculate ground transport emissions
        if input_data.ground_transport and self.config.category_6_include_ground_transport:
            ground_emissions, ground_details = await self._calculate_ground_transport(
                input_data.ground_transport
            )
            total_emissions += ground_emissions
            components.append({
                "type": "ground_transport",
                "emissions_kgco2e": ground_emissions,
                "count": len(input_data.ground_transport),
                "details": ground_details
            })

        # Monte Carlo uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo and total_emissions > 0:
            # Combined uncertainty for business travel
            # Flight distances: ±8%, Hotel nights: exact, Ground transport: ±10%
            # Emission factors: ±15% typical for business travel
            combined_uncertainty = 0.12  # Combined relative uncertainty for business travel

            uncertainty = await self.uncertainty_engine.propagate(
                quantity=total_emissions,
                quantity_uncertainty=combined_uncertainty,
                emission_factor=1.0,  # Already baked into total
                factor_uncertainty=0.15,
                iterations=self.config.monte_carlo_iterations
            )

            logger.debug(
                f"Category 6 uncertainty: mean={uncertainty.mean:.2f}, "
                f"P5={uncertainty.p5:.2f}, P95={uncertainty.p95:.2f}"
            )

        # Data quality (business travel is typically Tier 2-3)
        avg_dqi = 65.0  # Typical for business travel data

        data_quality = DataQualityInfo(
            dqi_score=avg_dqi,
            tier=TierType.TIER_2,
            rating="good",
            pedigree_score=3.2,
            warnings=warnings
        )

        # Provenance
        provenance = await self.provenance_builder.build(
            category=6,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=None,
            calculation={
                "formula": "sum(flights + hotels + ground_transport)",
                "components": components,
                "result_kgco2e": total_emissions,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions,
            emissions_tco2e=total_emissions / 1000,
            category=6,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="business_travel_combined",
            warnings=warnings,
            metadata={
                "trip_id": input_data.trip_id,
                "employee_id": input_data.employee_id,
                "components": components,
                "num_flights": len(input_data.flights),
                "num_hotel_nights": sum(h.nights for h in input_data.hotels),
            }
        )

    async def _calculate_flights(
        self, flights: List[Category6FlightInput]
    ) -> tuple[float, List[Dict]]:
        """Calculate emissions from all flights."""
        total = 0.0
        details = []

        for flight in flights:
            # Get emission factor
            ef = flight.emission_factor or FLIGHT_EMISSION_FACTORS.get(
                flight.cabin_class, 0.115
            )

            # Calculate base emissions
            emissions = flight.distance_km * flight.num_passengers * ef

            # Apply radiative forcing if enabled
            rf_factor = 1.0
            if flight.apply_radiative_forcing:
                rf_factor = self.config.category_6_radiative_forcing_factor
                emissions *= rf_factor

            total += emissions

            details.append({
                "distance_km": flight.distance_km,
                "cabin_class": flight.cabin_class.value,
                "passengers": flight.num_passengers,
                "emission_factor": ef,
                "radiative_forcing": rf_factor,
                "emissions_kgco2e": emissions,
                "origin": flight.origin,
                "destination": flight.destination,
            })

            logger.debug(
                f"Flight: {flight.distance_km} km × {flight.num_passengers} pax × "
                f"{ef} × RF{rf_factor} = {emissions:.2f} kgCO2e"
            )

        return total, details

    async def _calculate_hotels(
        self, hotels: List[Category6HotelInput]
    ) -> tuple[float, List[Dict]]:
        """Calculate emissions from hotel stays."""
        total = 0.0
        details = []

        for hotel in hotels:
            # Get emission factor
            ef = hotel.emission_factor or HOTEL_EMISSION_FACTORS.get(
                hotel.region, HOTEL_EMISSION_FACTORS["Global"]
            )

            # Calculate emissions
            emissions = hotel.nights * ef

            total += emissions

            details.append({
                "nights": hotel.nights,
                "region": hotel.region,
                "emission_factor": ef,
                "emissions_kgco2e": emissions,
                "hotel_name": hotel.hotel_name,
            })

            logger.debug(
                f"Hotel: {hotel.nights} nights × {ef} = {emissions:.2f} kgCO2e"
            )

        return total, details

    async def _calculate_ground_transport(
        self, transports: List[Category6GroundTransportInput]
    ) -> tuple[float, List[Dict]]:
        """Calculate emissions from ground transport."""
        total = 0.0
        details = []

        # Default emission factors for ground transport (kgCO2e/km)
        ground_efs = {
            "car_small": 0.105,
            "car_medium": 0.145,
            "car_large": 0.198,
            "taxi": 0.150,
            "rental_car": 0.145,
            "bus": 0.028,
            "train": 0.041,
        }

        for transport in transports:
            # Get emission factor
            ef = transport.emission_factor or ground_efs.get(
                transport.vehicle_type, 0.145
            )

            # Calculate emissions
            emissions = transport.distance_km * ef

            total += emissions

            details.append({
                "distance_km": transport.distance_km,
                "vehicle_type": transport.vehicle_type,
                "emission_factor": ef,
                "emissions_kgco2e": emissions,
            })

            logger.debug(
                f"Ground: {transport.distance_km} km × {ef} = {emissions:.2f} kgCO2e"
            )

        return total, details


__all__ = ["Category6Calculator"]
