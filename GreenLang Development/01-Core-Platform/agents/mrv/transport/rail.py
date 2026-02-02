# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-004: Rail MRV Agent
==============================

This module implements the Rail MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from rail transport activities.

Supported Features:
- Freight rail emissions
- Passenger rail emissions
- Electric and diesel rail
- Tonne-km and passenger-km calculations
- Multiple traction types

Reference Standards:
- GHG Protocol Scope 3, Categories 4, 6, 7, 9
- DEFRA Conversion Factors 2024
- UIC Rail Transport and Environment Facts

Example:
    >>> agent = RailMRVAgent()
    >>> input_data = RailInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     shipments=[
    ...         RailShipmentRecord(
    ...             rail_type=RailType.FREIGHT_DIESEL,
    ...             distance_km=Decimal("500"),
    ...             cargo_tonnes=Decimal("100"),
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
    FuelType,
    EmissionScope,
    CalculationMethod,
    DataQualityTier,
    EmissionFactor,
    EmissionFactorSource,
    CalculationStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Rail-Specific Enums
# =============================================================================

class RailType(str, Enum):
    """Types of rail transport."""
    FREIGHT_DIESEL = "freight_diesel"
    FREIGHT_ELECTRIC = "freight_electric"
    FREIGHT_AVERAGE = "freight_average"
    PASSENGER_NATIONAL = "passenger_national"
    PASSENGER_INTERNATIONAL = "passenger_international"
    PASSENGER_LIGHT_RAIL = "passenger_light_rail"
    PASSENGER_METRO = "passenger_metro"
    PASSENGER_TRAM = "passenger_tram"
    HIGH_SPEED = "high_speed"


class TractionType(str, Enum):
    """Rail traction types."""
    DIESEL = "diesel"
    ELECTRIC = "electric"
    DUAL_MODE = "dual_mode"
    HYDROGEN = "hydrogen"
    BATTERY = "battery"


# =============================================================================
# DEFRA 2024 Rail Emission Factors
# =============================================================================

# Freight rail factors (kg CO2e per tonne-km)
RAIL_FREIGHT_FACTORS: Dict[str, Decimal] = {
    RailType.FREIGHT_DIESEL.value: Decimal("0.03214"),
    RailType.FREIGHT_ELECTRIC.value: Decimal("0.01763"),
    RailType.FREIGHT_AVERAGE.value: Decimal("0.02781"),
}

# Passenger rail factors (kg CO2e per passenger-km)
RAIL_PASSENGER_FACTORS: Dict[str, Decimal] = {
    RailType.PASSENGER_NATIONAL.value: Decimal("0.03549"),
    RailType.PASSENGER_INTERNATIONAL.value: Decimal("0.00446"),  # Eurostar
    RailType.PASSENGER_LIGHT_RAIL.value: Decimal("0.02899"),
    RailType.PASSENGER_METRO.value: Decimal("0.02781"),
    RailType.PASSENGER_TRAM.value: Decimal("0.02899"),
    RailType.HIGH_SPEED.value: Decimal("0.00446"),
}


# =============================================================================
# Input Models
# =============================================================================

class RailShipmentRecord(BaseModel):
    """Individual rail shipment or journey record."""

    # Shipment identification
    shipment_id: Optional[str] = Field(None, description="Unique shipment identifier")

    # Rail type and traction
    rail_type: RailType = Field(..., description="Type of rail service")
    traction_type: TractionType = Field(
        TractionType.DIESEL, description="Traction type"
    )

    # Route information
    origin: Optional[str] = Field(None, description="Origin station/terminal")
    destination: Optional[str] = Field(None, description="Destination station/terminal")
    distance_km: Decimal = Field(..., ge=0, description="Distance in kilometers")

    # Freight details
    cargo_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Cargo weight in tonnes"
    )
    wagon_count: Optional[int] = Field(
        None, ge=0, description="Number of wagons"
    )

    # Passenger details
    passengers: Optional[int] = Field(
        None, ge=0, description="Number of passengers"
    )

    # Direct fuel/energy data (optional)
    fuel_liters: Optional[Decimal] = Field(
        None, ge=0, description="Diesel fuel consumed in liters"
    )
    electricity_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Electricity consumed in kWh"
    )

    class Config:
        use_enum_values = True


class RailInput(TransportMRVInput):
    """Input model for Rail MRV Agent."""

    # Rail records
    shipments: List[RailShipmentRecord] = Field(
        default_factory=list, description="List of rail shipment/journey records"
    )

    # Aggregated freight data
    total_freight_tonne_km: Optional[Decimal] = Field(
        None, ge=0, description="Total freight tonne-km"
    )
    default_freight_type: RailType = Field(
        RailType.FREIGHT_AVERAGE, description="Default freight rail type"
    )

    # Aggregated passenger data
    total_passenger_km: Optional[Decimal] = Field(
        None, ge=0, description="Total passenger-km"
    )
    default_passenger_type: RailType = Field(
        RailType.PASSENGER_NATIONAL, description="Default passenger rail type"
    )

    # Fleet ownership
    is_owned_fleet: bool = Field(
        False, description="Whether rail fleet is owned (Scope 1)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Output Model
# =============================================================================

class RailOutput(TransportMRVOutput):
    """Output model for Rail MRV Agent."""

    # Rail-specific metrics
    total_shipments: int = Field(0, ge=0, description="Total number of shipments")
    total_distance_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total distance"
    )
    total_freight_tonne_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total freight tonne-km"
    )
    total_passenger_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total passenger-km"
    )

    # Breakdown
    freight_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Freight emissions"
    )
    passenger_emissions_kg: Decimal = Field(
        Decimal("0"), ge=0, description="Passenger emissions"
    )

    # Efficiency metrics
    emissions_per_freight_tonne_km: Optional[Decimal] = Field(
        None, description="kg CO2e per freight tonne-km"
    )
    emissions_per_passenger_km: Optional[Decimal] = Field(
        None, description="kg CO2e per passenger-km"
    )

    # Breakdown by rail type
    emissions_by_rail_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by rail type"
    )


# =============================================================================
# Rail MRV Agent
# =============================================================================

class RailMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-004: Rail MRV Agent

    Calculates greenhouse gas emissions from rail transport activities
    including freight and passenger rail.

    Key Features:
    - Freight rail (diesel and electric)
    - Passenger rail (national, international, metro)
    - Tonne-km and passenger-km calculations
    - DEFRA 2024 emission factors

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance
    """

    AGENT_ID = "GL-MRV-TRN-004"
    AGENT_NAME = "Rail MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.RAIL
    DEFAULT_SCOPE = EmissionScope.SCOPE_3

    def __init__(self):
        """Initialize Rail MRV Agent."""
        super().__init__()
        self._freight_factors = RAIL_FREIGHT_FACTORS
        self._passenger_factors = RAIL_PASSENGER_FACTORS

    def calculate(self, input_data: RailInput) -> RailOutput:
        """
        Calculate rail transport emissions.

        Args:
            input_data: Rail input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Initialize totals
        total_emissions_kg = Decimal("0")
        total_distance_km = Decimal("0")
        total_freight_tonne_km = Decimal("0")
        total_passenger_km = Decimal("0")
        freight_emissions_kg = Decimal("0")
        passenger_emissions_kg = Decimal("0")
        emissions_by_type: Dict[str, Decimal] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize rail emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "num_shipments": len(input_data.shipments),
            },
        ))

        # Process individual records
        for idx, shipment in enumerate(input_data.shipments):
            shipment_result = self._calculate_shipment_emissions(
                shipment=shipment,
                step_offset=len(steps),
            )

            steps.extend(shipment_result["steps"])
            emission_factors.extend(shipment_result["factors"])

            total_emissions_kg += shipment_result["total_kg"]
            total_distance_km += shipment.distance_km
            total_freight_tonne_km += shipment_result["freight_tonne_km"]
            total_passenger_km += shipment_result["passenger_km"]
            freight_emissions_kg += shipment_result["freight_emissions_kg"]
            passenger_emissions_kg += shipment_result["passenger_emissions_kg"]

            # Track by rail type
            rtype = shipment.rail_type.value if hasattr(shipment.rail_type, 'value') else str(shipment.rail_type)
            emissions_by_type[rtype] = emissions_by_type.get(
                rtype, Decimal("0")
            ) + shipment_result["total_kg"]

        # Process aggregated freight data
        if input_data.total_freight_tonne_km and not input_data.shipments:
            ftype = input_data.default_freight_type.value if hasattr(input_data.default_freight_type, 'value') else str(input_data.default_freight_type)
            factor = self._freight_factors.get(
                ftype, self._freight_factors[RailType.FREIGHT_AVERAGE.value]
            )
            agg_freight_emissions = (input_data.total_freight_tonne_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += agg_freight_emissions
            freight_emissions_kg += agg_freight_emissions
            total_freight_tonne_km += input_data.total_freight_tonne_km

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate aggregated freight emissions",
                formula="emissions = tonne_km x EF",
                inputs={
                    "total_freight_tonne_km": str(input_data.total_freight_tonne_km),
                    "rail_type": ftype,
                    "emission_factor": str(factor),
                },
                output=str(agg_freight_emissions),
            ))

        # Process aggregated passenger data
        if input_data.total_passenger_km and not input_data.shipments:
            ptype = input_data.default_passenger_type.value if hasattr(input_data.default_passenger_type, 'value') else str(input_data.default_passenger_type)
            factor = self._passenger_factors.get(
                ptype, self._passenger_factors[RailType.PASSENGER_NATIONAL.value]
            )
            agg_passenger_emissions = (input_data.total_passenger_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += agg_passenger_emissions
            passenger_emissions_kg += agg_passenger_emissions
            total_passenger_km += input_data.total_passenger_km

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate aggregated passenger emissions",
                formula="emissions = passenger_km x EF",
                inputs={
                    "total_passenger_km": str(input_data.total_passenger_km),
                    "rail_type": ptype,
                    "emission_factor": str(factor),
                },
                output=str(agg_passenger_emissions),
            ))

        # Calculate efficiency metrics
        emissions_per_freight_tonne_km = None
        emissions_per_passenger_km = None

        if total_freight_tonne_km > 0:
            emissions_per_freight_tonne_km = (freight_emissions_kg / total_freight_tonne_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
        if total_passenger_km > 0:
            emissions_per_passenger_km = (passenger_emissions_kg / total_passenger_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )

        # Final summary
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total rail emissions",
            inputs={
                "freight_emissions_kg": str(freight_emissions_kg),
                "passenger_emissions_kg": str(passenger_emissions_kg),
            },
            output=str(total_emissions_kg),
        ))

        # Determine scope
        scope = EmissionScope.SCOPE_1 if input_data.is_owned_fleet else EmissionScope.SCOPE_3

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "transport_mode": "rail",
            "total_shipments": len(input_data.shipments),
            "total_freight_tonne_km": str(total_freight_tonne_km),
            "total_passenger_km": str(total_passenger_km),
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
            scope=scope,
            warnings=warnings,
        )

        return RailOutput(
            **base_output.dict(),
            total_shipments=len(input_data.shipments),
            total_distance_km=total_distance_km,
            total_freight_tonne_km=total_freight_tonne_km,
            total_passenger_km=total_passenger_km,
            freight_emissions_kg=freight_emissions_kg,
            passenger_emissions_kg=passenger_emissions_kg,
            emissions_per_freight_tonne_km=emissions_per_freight_tonne_km,
            emissions_per_passenger_km=emissions_per_passenger_km,
            emissions_by_rail_type=emissions_by_type,
        )

    def _calculate_shipment_emissions(
        self,
        shipment: RailShipmentRecord,
        step_offset: int,
    ) -> Dict[str, Any]:
        """
        Calculate emissions for a single rail shipment/journey.

        Args:
            shipment: Rail shipment record
            step_offset: Step number offset

        Returns:
            Dictionary with emissions and calculation details
        """
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        total_kg = Decimal("0")
        freight_tonne_km = Decimal("0")
        passenger_km = Decimal("0")
        freight_emissions_kg = Decimal("0")
        passenger_emissions_kg = Decimal("0")

        rtype = shipment.rail_type.value if hasattr(shipment.rail_type, 'value') else str(shipment.rail_type)

        # Calculate freight emissions
        if shipment.cargo_tonnes and shipment.cargo_tonnes > 0:
            freight_tonne_km = shipment.distance_km * shipment.cargo_tonnes
            factor = self._freight_factors.get(
                rtype, self._freight_factors[RailType.FREIGHT_AVERAGE.value]
            )
            freight_emissions_kg = (freight_tonne_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_kg += freight_emissions_kg

            ef_record = EmissionFactor(
                factor_id=f"defra_2024_rail_{rtype}",
                factor_value=factor,
                factor_unit="kg CO2e/tonne-km",
                source=EmissionFactorSource.DEFRA,
                source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
                version="2024",
                last_updated="2024-06-01",
                data_quality_tier=DataQualityTier.TIER_2,
            )
            factors.append(ef_record)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate freight emissions: {shipment.origin or 'Origin'} to {shipment.destination or 'Destination'}",
                formula="emissions = tonne_km x EF",
                inputs={
                    "rail_type": rtype,
                    "distance_km": str(shipment.distance_km),
                    "cargo_tonnes": str(shipment.cargo_tonnes),
                    "tonne_km": str(freight_tonne_km),
                    "emission_factor": str(factor),
                },
                output=str(freight_emissions_kg),
                emission_factor=ef_record,
            ))

        # Calculate passenger emissions
        if shipment.passengers and shipment.passengers > 0:
            passenger_km = shipment.distance_km * Decimal(str(shipment.passengers))
            factor = self._passenger_factors.get(
                rtype, self._passenger_factors[RailType.PASSENGER_NATIONAL.value]
            )
            passenger_emissions_kg = (passenger_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_kg += passenger_emissions_kg

            steps.append(CalculationStep(
                step_number=step_offset + 2,
                description=f"Calculate passenger emissions: {shipment.origin or 'Origin'} to {shipment.destination or 'Destination'}",
                formula="emissions = passenger_km x EF",
                inputs={
                    "rail_type": rtype,
                    "distance_km": str(shipment.distance_km),
                    "passengers": shipment.passengers,
                    "passenger_km": str(passenger_km),
                    "emission_factor": str(factor),
                },
                output=str(passenger_emissions_kg),
            ))

        return {
            "total_kg": total_kg,
            "freight_tonne_km": freight_tonne_km,
            "passenger_km": passenger_km,
            "freight_emissions_kg": freight_emissions_kg,
            "passenger_emissions_kg": passenger_emissions_kg,
            "steps": steps,
            "factors": factors,
        }
