# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-005: Last-Mile MRV Agent
===================================

This module implements the Last-Mile MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from last-mile delivery activities.

Supported Features:
- Parcel delivery emissions
- Multiple delivery vehicle types (vans, bikes, e-cargo bikes)
- Urban vs suburban delivery
- Failed delivery calculations
- Per-parcel emission intensity

Reference Standards:
- GHG Protocol Scope 3, Categories 4, 9
- DEFRA Conversion Factors 2024
- GLEC Framework for Logistics

Example:
    >>> agent = LastMileMRVAgent()
    >>> input_data = LastMileInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     deliveries=[
    ...         DeliveryRecord(
    ...             vehicle_type=LastMileVehicle.VAN_DIESEL,
    ...             parcels_delivered=100,
    ...             distance_km=Decimal("50"),
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
    DataQualityTier,
    EmissionFactor,
    EmissionFactorSource,
    CalculationStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Last-Mile Specific Enums
# =============================================================================

class LastMileVehicle(str, Enum):
    """Last-mile delivery vehicle types."""
    VAN_DIESEL = "van_diesel"
    VAN_PETROL = "van_petrol"
    VAN_ELECTRIC = "van_electric"
    VAN_CNG = "van_cng"
    CARGO_BIKE = "cargo_bike"
    E_CARGO_BIKE = "e_cargo_bike"
    MOTORCYCLE = "motorcycle"
    E_SCOOTER = "e_scooter"
    CAR_SMALL = "car_small"
    CAR_ELECTRIC = "car_electric"
    WALKING = "walking"
    DRONE = "drone"


class DeliveryArea(str, Enum):
    """Delivery area types."""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    MIXED = "mixed"


# =============================================================================
# DEFRA 2024 Last-Mile Emission Factors
# =============================================================================

# Vehicle emission factors (kg CO2e per km)
LAST_MILE_FACTORS: Dict[str, Decimal] = {
    LastMileVehicle.VAN_DIESEL.value: Decimal("0.24102"),
    LastMileVehicle.VAN_PETROL.value: Decimal("0.21556"),
    LastMileVehicle.VAN_ELECTRIC.value: Decimal("0.05297"),
    LastMileVehicle.VAN_CNG.value: Decimal("0.19234"),
    LastMileVehicle.CARGO_BIKE.value: Decimal("0"),
    LastMileVehicle.E_CARGO_BIKE.value: Decimal("0.00621"),  # Based on electricity
    LastMileVehicle.MOTORCYCLE.value: Decimal("0.11374"),
    LastMileVehicle.E_SCOOTER.value: Decimal("0.01053"),
    LastMileVehicle.CAR_SMALL.value: Decimal("0.14901"),
    LastMileVehicle.CAR_ELECTRIC.value: Decimal("0.05297"),
    LastMileVehicle.WALKING.value: Decimal("0"),
    LastMileVehicle.DRONE.value: Decimal("0.00312"),  # Estimated
}

# Average parcels per delivery stop by vehicle type
PARCELS_PER_STOP: Dict[str, Decimal] = {
    LastMileVehicle.VAN_DIESEL.value: Decimal("1.3"),
    LastMileVehicle.VAN_ELECTRIC.value: Decimal("1.3"),
    LastMileVehicle.CARGO_BIKE.value: Decimal("1.1"),
    LastMileVehicle.E_CARGO_BIKE.value: Decimal("1.2"),
    "default": Decimal("1.2"),
}

# Average km per delivery stop by area
KM_PER_STOP: Dict[str, Decimal] = {
    DeliveryArea.URBAN.value: Decimal("0.5"),
    DeliveryArea.SUBURBAN.value: Decimal("1.2"),
    DeliveryArea.RURAL.value: Decimal("3.5"),
    DeliveryArea.MIXED.value: Decimal("1.0"),
}


# =============================================================================
# Input Models
# =============================================================================

class DeliveryRecord(BaseModel):
    """Individual delivery route/batch record."""

    # Identification
    route_id: Optional[str] = Field(None, description="Route identifier")
    date: Optional[str] = Field(None, description="Delivery date")

    # Vehicle details
    vehicle_type: LastMileVehicle = Field(..., description="Delivery vehicle type")

    # Delivery metrics
    parcels_delivered: int = Field(..., ge=0, description="Number of parcels delivered")
    parcels_attempted: Optional[int] = Field(
        None, ge=0, description="Total delivery attempts"
    )
    stops_made: Optional[int] = Field(None, ge=0, description="Number of stops")

    # Distance data
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Total distance traveled"
    )

    # Fuel/energy data
    fuel_liters: Optional[Decimal] = Field(
        None, ge=0, description="Fuel consumed in liters"
    )
    electricity_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Electricity consumed in kWh"
    )

    # Area type
    delivery_area: DeliveryArea = Field(
        DeliveryArea.MIXED, description="Delivery area type"
    )

    # Weight data
    total_weight_kg: Optional[Decimal] = Field(
        None, ge=0, description="Total parcel weight"
    )

    class Config:
        use_enum_values = True


class LastMileInput(TransportMRVInput):
    """Input model for Last-Mile MRV Agent."""

    # Delivery records
    deliveries: List[DeliveryRecord] = Field(
        default_factory=list, description="List of delivery records"
    )

    # Aggregated data
    total_parcels: Optional[int] = Field(
        None, ge=0, description="Total parcels delivered"
    )
    total_distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Total delivery distance"
    )
    default_vehicle: LastMileVehicle = Field(
        LastMileVehicle.VAN_DIESEL, description="Default vehicle type"
    )
    default_area: DeliveryArea = Field(
        DeliveryArea.MIXED, description="Default delivery area"
    )

    # Failed delivery factor
    failed_delivery_rate: Decimal = Field(
        Decimal("0.05"), ge=0, le=1, description="Failed delivery rate (0-1)"
    )
    include_failed_attempts: bool = Field(
        True, description="Include emissions from failed deliveries"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# Output Model
# =============================================================================

class LastMileOutput(TransportMRVOutput):
    """Output model for Last-Mile MRV Agent."""

    # Last-mile specific metrics
    total_routes: int = Field(0, ge=0, description="Total delivery routes")
    total_parcels_delivered: int = Field(0, ge=0, description="Parcels delivered")
    total_parcels_attempted: int = Field(0, ge=0, description="Delivery attempts")
    total_stops: int = Field(0, ge=0, description="Total delivery stops")
    total_distance_km: Decimal = Field(Decimal("0"), ge=0, description="Total distance")

    # Efficiency metrics
    emissions_per_parcel_kg: Optional[Decimal] = Field(
        None, description="kg CO2e per parcel"
    )
    emissions_per_km: Optional[Decimal] = Field(
        None, description="kg CO2e per km"
    )
    parcels_per_km: Optional[Decimal] = Field(
        None, description="Delivery density"
    )

    # Breakdown by vehicle type
    emissions_by_vehicle: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by vehicle type"
    )
    parcels_by_vehicle: Dict[str, int] = Field(
        default_factory=dict, description="Parcels by vehicle type"
    )


# =============================================================================
# Last-Mile MRV Agent
# =============================================================================

class LastMileMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-005: Last-Mile MRV Agent

    Calculates greenhouse gas emissions from last-mile delivery operations.

    Key Features:
    - Multiple vehicle types (vans, bikes, drones)
    - Per-parcel emission intensity
    - Failed delivery emissions
    - Urban/suburban/rural differentiation

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance
    """

    AGENT_ID = "GL-MRV-TRN-005"
    AGENT_NAME = "Last-Mile MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.ROAD
    DEFAULT_SCOPE = EmissionScope.SCOPE_3

    def calculate(self, input_data: LastMileInput) -> LastMileOutput:
        """
        Calculate last-mile delivery emissions.

        Args:
            input_data: Last-mile input data

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
        total_parcels_delivered = 0
        total_parcels_attempted = 0
        total_stops = 0
        emissions_by_vehicle: Dict[str, Decimal] = {}
        parcels_by_vehicle: Dict[str, int] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize last-mile emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "num_routes": len(input_data.deliveries),
                "failed_delivery_rate": str(input_data.failed_delivery_rate),
            },
        ))

        # Process individual delivery records
        for delivery in input_data.deliveries:
            result = self._calculate_delivery_emissions(
                delivery=delivery,
                include_failed=input_data.include_failed_attempts,
                failed_rate=input_data.failed_delivery_rate,
                step_offset=len(steps),
            )

            steps.extend(result["steps"])
            emission_factors.extend(result["factors"])

            total_emissions_kg += result["total_kg"]
            total_distance_km += result["distance_km"]
            total_parcels_delivered += delivery.parcels_delivered
            total_parcels_attempted += result["parcels_attempted"]
            total_stops += result["stops"]

            # Track by vehicle
            vtype = delivery.vehicle_type.value if hasattr(delivery.vehicle_type, 'value') else str(delivery.vehicle_type)
            emissions_by_vehicle[vtype] = emissions_by_vehicle.get(
                vtype, Decimal("0")
            ) + result["total_kg"]
            parcels_by_vehicle[vtype] = parcels_by_vehicle.get(
                vtype, 0
            ) + delivery.parcels_delivered

        # Process aggregated data
        if input_data.total_parcels and not input_data.deliveries:
            agg_result = self._calculate_aggregated_emissions(
                parcels=input_data.total_parcels,
                distance_km=input_data.total_distance_km,
                vehicle_type=input_data.default_vehicle,
                area=input_data.default_area,
                failed_rate=input_data.failed_delivery_rate,
                include_failed=input_data.include_failed_attempts,
                step_offset=len(steps),
            )

            steps.extend(agg_result["steps"])
            total_emissions_kg += agg_result["total_kg"]
            total_distance_km += agg_result["distance_km"]
            total_parcels_delivered += input_data.total_parcels
            total_parcels_attempted += agg_result["parcels_attempted"]

        # Calculate efficiency metrics
        emissions_per_parcel = None
        emissions_per_km = None
        parcels_per_km = None

        if total_parcels_delivered > 0:
            emissions_per_parcel = (total_emissions_kg / Decimal(str(total_parcels_delivered))).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
        if total_distance_km > 0:
            emissions_per_km = (total_emissions_kg / total_distance_km).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            parcels_per_km = (Decimal(str(total_parcels_delivered)) / total_distance_km).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Final summary
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total last-mile emissions",
            inputs={
                "total_parcels": total_parcels_delivered,
                "total_distance_km": str(total_distance_km),
            },
            output=str(total_emissions_kg),
        ))

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "transport_mode": "last_mile",
            "total_routes": len(input_data.deliveries),
            "total_parcels": total_parcels_delivered,
            "total_distance_km": str(total_distance_km),
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

        return LastMileOutput(
            **base_output.dict(),
            total_routes=len(input_data.deliveries),
            total_parcels_delivered=total_parcels_delivered,
            total_parcels_attempted=total_parcels_attempted,
            total_stops=total_stops,
            total_distance_km=total_distance_km,
            emissions_per_parcel_kg=emissions_per_parcel,
            emissions_per_km=emissions_per_km,
            parcels_per_km=parcels_per_km,
            emissions_by_vehicle=emissions_by_vehicle,
            parcels_by_vehicle=parcels_by_vehicle,
        )

    def _calculate_delivery_emissions(
        self,
        delivery: DeliveryRecord,
        include_failed: bool,
        failed_rate: Decimal,
        step_offset: int,
    ) -> Dict[str, Any]:
        """Calculate emissions for a single delivery route."""
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        vtype = delivery.vehicle_type.value if hasattr(delivery.vehicle_type, 'value') else str(delivery.vehicle_type)

        # Get emission factor
        factor = LAST_MILE_FACTORS.get(vtype, LAST_MILE_FACTORS[LastMileVehicle.VAN_DIESEL.value])

        # Calculate or estimate distance
        if delivery.distance_km:
            distance_km = delivery.distance_km
        else:
            # Estimate from stops and area
            area = delivery.delivery_area.value if hasattr(delivery.delivery_area, 'value') else str(delivery.delivery_area)
            km_per_stop = KM_PER_STOP.get(area, KM_PER_STOP[DeliveryArea.MIXED.value])

            if delivery.stops_made:
                stops = delivery.stops_made
            else:
                parcels_per_stop = PARCELS_PER_STOP.get(vtype, PARCELS_PER_STOP["default"])
                stops = int((Decimal(str(delivery.parcels_delivered)) / parcels_per_stop).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                ))

            distance_km = Decimal(str(stops)) * km_per_stop

        # Calculate attempts including failed
        if delivery.parcels_attempted:
            parcels_attempted = delivery.parcels_attempted
        elif include_failed:
            parcels_attempted = int(Decimal(str(delivery.parcels_delivered)) / (Decimal("1") - failed_rate))
        else:
            parcels_attempted = delivery.parcels_delivered

        # Adjust distance for failed deliveries
        if include_failed and parcels_attempted > delivery.parcels_delivered:
            failed_fraction = Decimal(str(parcels_attempted - delivery.parcels_delivered)) / Decimal(str(parcels_attempted))
            distance_km = distance_km * (Decimal("1") + failed_fraction * Decimal("0.5"))

        # Calculate emissions
        total_kg = (distance_km * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        ef_record = EmissionFactor(
            factor_id=f"defra_2024_{vtype}",
            factor_value=factor,
            factor_unit="kg CO2e/km",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-06-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )
        factors.append(ef_record)

        steps.append(CalculationStep(
            step_number=step_offset + 1,
            description=f"Calculate delivery route emissions: {delivery.route_id or 'Route'}",
            formula="emissions = distance_km x EF",
            inputs={
                "vehicle_type": vtype,
                "parcels_delivered": delivery.parcels_delivered,
                "distance_km": str(distance_km),
                "emission_factor": str(factor),
            },
            output=str(total_kg),
            emission_factor=ef_record,
        ))

        return {
            "total_kg": total_kg,
            "distance_km": distance_km,
            "parcels_attempted": parcels_attempted,
            "stops": delivery.stops_made or int(distance_km / Decimal("1.0")),
            "steps": steps,
            "factors": factors,
        }

    def _calculate_aggregated_emissions(
        self,
        parcels: int,
        distance_km: Optional[Decimal],
        vehicle_type: LastMileVehicle,
        area: DeliveryArea,
        failed_rate: Decimal,
        include_failed: bool,
        step_offset: int,
    ) -> Dict[str, Any]:
        """Calculate emissions from aggregated data."""
        steps: List[CalculationStep] = []

        vtype = vehicle_type.value if hasattr(vehicle_type, 'value') else str(vehicle_type)
        factor = LAST_MILE_FACTORS.get(vtype, LAST_MILE_FACTORS[LastMileVehicle.VAN_DIESEL.value])

        # Calculate attempts
        if include_failed:
            parcels_attempted = int(Decimal(str(parcels)) / (Decimal("1") - failed_rate))
        else:
            parcels_attempted = parcels

        # Estimate or use provided distance
        if distance_km:
            total_distance = distance_km
        else:
            area_val = area.value if hasattr(area, 'value') else str(area)
            km_per_stop = KM_PER_STOP.get(area_val, KM_PER_STOP[DeliveryArea.MIXED.value])
            parcels_per_stop = PARCELS_PER_STOP.get(vtype, PARCELS_PER_STOP["default"])
            stops = Decimal(str(parcels_attempted)) / parcels_per_stop
            total_distance = (stops * km_per_stop).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        total_kg = (total_distance * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=step_offset + 1,
            description="Calculate aggregated last-mile emissions",
            formula="emissions = distance_km x EF",
            inputs={
                "total_parcels": parcels,
                "parcels_attempted": parcels_attempted,
                "distance_km": str(total_distance),
                "vehicle_type": vtype,
                "emission_factor": str(factor),
            },
            output=str(total_kg),
        ))

        return {
            "total_kg": total_kg,
            "distance_km": total_distance,
            "parcels_attempted": parcels_attempted,
            "steps": steps,
        }
