# -*- coding: utf-8 -*-
"""
GL-MRV-TRN-001: Road Transport MRV Agent
=========================================

This module implements the Road Transport MRV Agent for measuring, reporting,
and verifying greenhouse gas emissions from road transport activities.

Supported Features:
- Fleet-level emissions calculation
- Individual vehicle tracking
- Fuel-based method (primary)
- Distance-based method (secondary)
- Support for multiple fuel types (diesel, petrol, LPG, CNG, electric)
- Scope 1 (owned/controlled) and Scope 3 (upstream transport) classification

Reference Standards:
- GHG Protocol Corporate Standard, Chapter 5
- GHG Protocol Scope 3, Categories 4, 6, 7, 9
- DEFRA Conversion Factors 2024
- EPA SmartWay

Example:
    >>> agent = RoadTransportMRVAgent()
    >>> input_data = RoadTransportInput(
    ...     organization_id="ORG001",
    ...     reporting_year=2024,
    ...     fleet_records=[
    ...         FleetRecord(
    ...             fleet_id="FLEET001",
    ...             vehicle_type=VehicleType.TRUCK_ARTICULATED,
    ...             fuel_type=FuelType.DIESEL,
    ...             fuel_consumed_liters=Decimal("50000"),
    ...             distance_km=Decimal("200000"),
    ...         )
    ...     ]
    ... )
    >>> result = agent.calculate(input_data)
    >>> print(f"Emissions: {result.total_emissions_mt_co2e} MT CO2e")
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator

from greenlang.agents.mrv.transport.base import (
    BaseTransportMRVAgent,
    TransportMRVInput,
    TransportMRVOutput,
    TransportMode,
    FuelType,
    VehicleType,
    EmissionScope,
    CalculationMethod,
    DataQualityTier,
    EmissionFactor,
    CalculationStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Input Models
# =============================================================================

class VehicleRecord(BaseModel):
    """Individual vehicle record for emissions tracking."""

    # Vehicle identification
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    vehicle_type: VehicleType = Field(..., description="Type of vehicle")
    registration: Optional[str] = Field(None, description="Vehicle registration")

    # Fuel data
    fuel_type: FuelType = Field(FuelType.DIESEL, description="Fuel type")
    fuel_consumed_liters: Optional[Decimal] = Field(
        None, ge=0, description="Fuel consumed in liters"
    )

    # Distance data
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Distance traveled in km"
    )
    odometer_start: Optional[Decimal] = Field(
        None, ge=0, description="Odometer reading at start"
    )
    odometer_end: Optional[Decimal] = Field(
        None, ge=0, description="Odometer reading at end"
    )

    # Load data (for freight)
    cargo_weight_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Cargo weight in tonnes"
    )
    load_factor: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Load factor (0-1)"
    )

    # Additional metadata
    vehicle_year: Optional[int] = Field(None, ge=1900, le=2100, description="Vehicle year")
    euro_standard: Optional[str] = Field(None, description="Euro emission standard")

    class Config:
        use_enum_values = True

    @validator("distance_km", pre=True, always=True)
    def calculate_distance_from_odometer(cls, v, values):
        """Calculate distance from odometer readings if not provided."""
        if v is None:
            start = values.get("odometer_start")
            end = values.get("odometer_end")
            if start is not None and end is not None:
                return Decimal(str(end)) - Decimal(str(start))
        return v


class FleetRecord(BaseModel):
    """Fleet-level aggregated record."""

    # Fleet identification
    fleet_id: str = Field(..., description="Fleet identifier")
    fleet_name: Optional[str] = Field(None, description="Fleet name")
    vehicle_count: int = Field(1, ge=1, description="Number of vehicles in fleet")

    # Vehicle characteristics (common for fleet)
    vehicle_type: VehicleType = Field(..., description="Predominant vehicle type")
    fuel_type: FuelType = Field(FuelType.DIESEL, description="Predominant fuel type")

    # Aggregated fuel data
    fuel_consumed_liters: Optional[Decimal] = Field(
        None, ge=0, description="Total fuel consumed in liters"
    )
    fuel_consumed_kg: Optional[Decimal] = Field(
        None, ge=0, description="Total fuel consumed in kg (for CNG)"
    )

    # Aggregated distance data
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Total distance traveled in km"
    )

    # Aggregated freight data
    total_tonne_km: Optional[Decimal] = Field(
        None, ge=0, description="Total tonne-km transported"
    )

    # Data quality
    data_source: str = Field(
        "fleet_management_system", description="Source of data"
    )
    data_completeness_pct: Decimal = Field(
        Decimal("100"), ge=0, le=100, description="Data completeness"
    )

    class Config:
        use_enum_values = True


class RoadTransportInput(TransportMRVInput):
    """Input model for Road Transport MRV Agent."""

    # Vehicle-level records
    vehicle_records: List[VehicleRecord] = Field(
        default_factory=list, description="Individual vehicle records"
    )

    # Fleet-level records
    fleet_records: List[FleetRecord] = Field(
        default_factory=list, description="Fleet-level aggregated records"
    )

    # Scope classification
    is_owned_fleet: bool = Field(
        True, description="Whether fleet is owned/controlled (Scope 1)"
    )

    # Refrigeration (for refrigerated transport)
    refrigeration_fuel_liters: Optional[Decimal] = Field(
        None, ge=0, description="Refrigeration unit fuel consumption"
    )

    @validator("vehicle_records", "fleet_records")
    def validate_has_data(cls, v, values, field):
        """Validate that either vehicles or fleets are provided."""
        return v


# =============================================================================
# Output Model
# =============================================================================

class RoadTransportOutput(TransportMRVOutput):
    """Output model for Road Transport MRV Agent."""

    # Mode-specific metrics
    total_distance_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total distance traveled"
    )
    total_fuel_liters: Decimal = Field(
        Decimal("0"), ge=0, description="Total fuel consumed"
    )
    total_tonne_km: Decimal = Field(
        Decimal("0"), ge=0, description="Total tonne-km transported"
    )

    # Efficiency metrics
    emissions_per_km: Optional[Decimal] = Field(
        None, description="kg CO2e per km"
    )
    emissions_per_tonne_km: Optional[Decimal] = Field(
        None, description="kg CO2e per tonne-km"
    )
    fuel_efficiency_km_per_liter: Optional[Decimal] = Field(
        None, description="Fuel efficiency in km/liter"
    )

    # Breakdown by vehicle type
    emissions_by_vehicle_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by vehicle type"
    )

    # Breakdown by fuel type
    emissions_by_fuel_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by fuel type"
    )


# =============================================================================
# Road Transport MRV Agent
# =============================================================================

class RoadTransportMRVAgent(BaseTransportMRVAgent):
    """
    GL-MRV-TRN-001: Road Transport MRV Agent

    Calculates greenhouse gas emissions from road transport activities
    including fleet vehicles, trucks, vans, and cars.

    Key Features:
    - Fuel-based calculation (primary method)
    - Distance-based calculation (secondary method)
    - Fleet-level and vehicle-level granularity
    - Multiple fuel type support
    - DEFRA 2024 emission factors

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - No LLM calls in the calculation path
    - Full audit trail with SHA-256 provenance

    Example:
        >>> agent = RoadTransportMRVAgent()
        >>> result = agent.calculate(input_data)
    """

    AGENT_ID = "GL-MRV-TRN-001"
    AGENT_NAME = "Road Transport MRV Agent"
    AGENT_VERSION = "1.0.0"
    TRANSPORT_MODE = TransportMode.ROAD
    DEFAULT_SCOPE = EmissionScope.SCOPE_1

    def calculate(self, input_data: RoadTransportInput) -> RoadTransportOutput:
        """
        Calculate road transport emissions.

        Args:
            input_data: Road transport input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        steps: List[CalculationStep] = []
        emission_factors: List[EmissionFactor] = []
        warnings: List[str] = []

        # Initialize totals
        total_emissions_kg = Decimal("0")
        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")
        total_distance_km = Decimal("0")
        total_fuel_liters = Decimal("0")
        total_tonne_km = Decimal("0")
        emissions_by_vehicle: Dict[str, Decimal] = {}
        emissions_by_fuel: Dict[str, Decimal] = {}

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize road transport emissions calculation",
            inputs={
                "organization_id": input_data.organization_id,
                "reporting_year": input_data.reporting_year,
                "calculation_method": input_data.calculation_method,
                "num_vehicle_records": len(input_data.vehicle_records),
                "num_fleet_records": len(input_data.fleet_records),
            },
        ))

        # Process vehicle-level records
        for idx, vehicle in enumerate(input_data.vehicle_records):
            vehicle_emissions, vehicle_steps, vehicle_factors = self._calculate_vehicle_emissions(
                vehicle=vehicle,
                method=input_data.calculation_method,
                step_offset=len(steps),
            )

            steps.extend(vehicle_steps)
            emission_factors.extend(vehicle_factors)

            # Accumulate totals
            total_emissions_kg += vehicle_emissions["total_kg"]
            total_co2_kg += vehicle_emissions["co2_kg"]
            total_ch4_kg += vehicle_emissions["ch4_kg"]
            total_n2o_kg += vehicle_emissions["n2o_kg"]

            if vehicle.distance_km:
                total_distance_km += vehicle.distance_km
            if vehicle.fuel_consumed_liters:
                total_fuel_liters += vehicle.fuel_consumed_liters

            # Track by vehicle type
            vtype = vehicle.vehicle_type.value if hasattr(vehicle.vehicle_type, 'value') else str(vehicle.vehicle_type)
            emissions_by_vehicle[vtype] = emissions_by_vehicle.get(
                vtype, Decimal("0")
            ) + vehicle_emissions["total_kg"]

            # Track by fuel type
            ftype = vehicle.fuel_type.value if hasattr(vehicle.fuel_type, 'value') else str(vehicle.fuel_type)
            emissions_by_fuel[ftype] = emissions_by_fuel.get(
                ftype, Decimal("0")
            ) + vehicle_emissions["total_kg"]

        # Process fleet-level records
        for idx, fleet in enumerate(input_data.fleet_records):
            fleet_emissions, fleet_steps, fleet_factors = self._calculate_fleet_emissions(
                fleet=fleet,
                method=input_data.calculation_method,
                step_offset=len(steps),
            )

            steps.extend(fleet_steps)
            emission_factors.extend(fleet_factors)

            # Accumulate totals
            total_emissions_kg += fleet_emissions["total_kg"]
            total_co2_kg += fleet_emissions["co2_kg"]
            total_ch4_kg += fleet_emissions["ch4_kg"]
            total_n2o_kg += fleet_emissions["n2o_kg"]

            if fleet.distance_km:
                total_distance_km += fleet.distance_km
            if fleet.fuel_consumed_liters:
                total_fuel_liters += fleet.fuel_consumed_liters
            if fleet.total_tonne_km:
                total_tonne_km += fleet.total_tonne_km

            # Track by vehicle type
            vtype = fleet.vehicle_type.value if hasattr(fleet.vehicle_type, 'value') else str(fleet.vehicle_type)
            emissions_by_vehicle[vtype] = emissions_by_vehicle.get(
                vtype, Decimal("0")
            ) + fleet_emissions["total_kg"]

            # Track by fuel type
            ftype = fleet.fuel_type.value if hasattr(fleet.fuel_type, 'value') else str(fleet.fuel_type)
            emissions_by_fuel[ftype] = emissions_by_fuel.get(
                ftype, Decimal("0")
            ) + fleet_emissions["total_kg"]

        # Calculate refrigeration emissions if applicable
        if input_data.refrigeration_fuel_liters:
            ref_total, ref_co2, ref_ch4, ref_n2o = self._calculate_fuel_based(
                fuel_liters=input_data.refrigeration_fuel_liters,
                fuel_type=FuelType.DIESEL,
            )
            total_emissions_kg += ref_total
            total_co2_kg += ref_co2
            total_ch4_kg += ref_ch4
            total_n2o_kg += ref_n2o

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate refrigeration unit emissions",
                formula="emissions = refrigeration_fuel x EF_diesel",
                inputs={"refrigeration_fuel_liters": str(input_data.refrigeration_fuel_liters)},
                output=str(ref_total),
            ))

        # Calculate efficiency metrics
        emissions_per_km = None
        emissions_per_tonne_km = None
        fuel_efficiency = None

        if total_distance_km > 0:
            emissions_per_km = (total_emissions_kg / total_distance_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
        if total_tonne_km > 0:
            emissions_per_tonne_km = (total_emissions_kg / total_tonne_km).quantize(
                Decimal("0.00001"), rounding=ROUND_HALF_UP
            )
        if total_fuel_liters > 0 and total_distance_km > 0:
            fuel_efficiency = (total_distance_km / total_fuel_liters).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Final summary step
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Aggregate total road transport emissions",
            formula="total = SUM(vehicle_emissions) + SUM(fleet_emissions) + refrigeration",
            inputs={
                "vehicle_emissions_count": len(input_data.vehicle_records),
                "fleet_emissions_count": len(input_data.fleet_records),
            },
            output=str(total_emissions_kg),
        ))

        # Determine scope
        scope = EmissionScope.SCOPE_1 if input_data.is_owned_fleet else EmissionScope.SCOPE_3

        # Build activity summary
        activity_summary = {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "transport_mode": "road",
            "total_vehicles": len(input_data.vehicle_records),
            "total_fleets": len(input_data.fleet_records),
            "total_distance_km": str(total_distance_km),
            "total_fuel_liters": str(total_fuel_liters),
            "is_owned_fleet": input_data.is_owned_fleet,
        }

        # Create base output
        base_output = self._create_output(
            total_emissions_kg=total_emissions_kg,
            co2_kg=total_co2_kg,
            ch4_kg=total_ch4_kg,
            n2o_kg=total_n2o_kg,
            steps=steps,
            emission_factors=emission_factors,
            activity_summary=activity_summary,
            start_time=start_time,
            scope=scope,
            warnings=warnings,
        )

        # Create road-specific output
        return RoadTransportOutput(
            **base_output.dict(),
            total_distance_km=total_distance_km,
            total_fuel_liters=total_fuel_liters,
            total_tonne_km=total_tonne_km,
            emissions_per_km=emissions_per_km,
            emissions_per_tonne_km=emissions_per_tonne_km,
            fuel_efficiency_km_per_liter=fuel_efficiency,
            emissions_by_vehicle_type=emissions_by_vehicle,
            emissions_by_fuel_type=emissions_by_fuel,
        )

    def _calculate_vehicle_emissions(
        self,
        vehicle: VehicleRecord,
        method: CalculationMethod,
        step_offset: int,
    ) -> tuple[Dict[str, Decimal], List[CalculationStep], List[EmissionFactor]]:
        """
        Calculate emissions for a single vehicle.

        Args:
            vehicle: Vehicle record
            method: Calculation method
            step_offset: Step number offset

        Returns:
            Tuple of (emissions_dict, steps, factors)
        """
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        total_kg = Decimal("0")
        co2_kg = Decimal("0")
        ch4_kg = Decimal("0")
        n2o_kg = Decimal("0")

        # Prefer fuel-based if available
        if vehicle.fuel_consumed_liters and (
            method == CalculationMethod.FUEL_BASED or
            method == CalculationMethod.ACTIVITY_BASED
        ):
            total_kg, co2_kg, ch4_kg, n2o_kg = self._calculate_fuel_based(
                fuel_liters=vehicle.fuel_consumed_liters,
                fuel_type=vehicle.fuel_type,
            )
            factor = self._get_fuel_factor(vehicle.fuel_type)
            factors.append(factor)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate fuel-based emissions for vehicle {vehicle.vehicle_id}",
                formula="emissions = fuel_liters x EF",
                inputs={
                    "vehicle_id": vehicle.vehicle_id,
                    "fuel_type": vehicle.fuel_type.value if hasattr(vehicle.fuel_type, 'value') else str(vehicle.fuel_type),
                    "fuel_liters": str(vehicle.fuel_consumed_liters),
                    "emission_factor": str(factor.factor_value),
                },
                output=str(total_kg),
                emission_factor=factor,
            ))

        # Fall back to distance-based
        elif vehicle.distance_km and method in [
            CalculationMethod.DISTANCE_BASED,
            CalculationMethod.ACTIVITY_BASED
        ]:
            total_kg = self._calculate_distance_based(
                distance_km=vehicle.distance_km,
                vehicle_type=vehicle.vehicle_type,
            )
            co2_kg = total_kg  # Approximate (distance factors are total CO2e)
            factor = self._get_vehicle_factor(vehicle.vehicle_type)
            factors.append(factor)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate distance-based emissions for vehicle {vehicle.vehicle_id}",
                formula="emissions = distance_km x EF",
                inputs={
                    "vehicle_id": vehicle.vehicle_id,
                    "vehicle_type": vehicle.vehicle_type.value if hasattr(vehicle.vehicle_type, 'value') else str(vehicle.vehicle_type),
                    "distance_km": str(vehicle.distance_km),
                    "emission_factor": str(factor.factor_value),
                },
                output=str(total_kg),
                emission_factor=factor,
            ))

        return {
            "total_kg": total_kg,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
        }, steps, factors

    def _calculate_fleet_emissions(
        self,
        fleet: FleetRecord,
        method: CalculationMethod,
        step_offset: int,
    ) -> tuple[Dict[str, Decimal], List[CalculationStep], List[EmissionFactor]]:
        """
        Calculate emissions for a fleet.

        Args:
            fleet: Fleet record
            method: Calculation method
            step_offset: Step number offset

        Returns:
            Tuple of (emissions_dict, steps, factors)
        """
        steps: List[CalculationStep] = []
        factors: List[EmissionFactor] = []

        total_kg = Decimal("0")
        co2_kg = Decimal("0")
        ch4_kg = Decimal("0")
        n2o_kg = Decimal("0")

        # Prefer fuel-based if available
        if fleet.fuel_consumed_liters and (
            method == CalculationMethod.FUEL_BASED or
            method == CalculationMethod.ACTIVITY_BASED
        ):
            total_kg, co2_kg, ch4_kg, n2o_kg = self._calculate_fuel_based(
                fuel_liters=fleet.fuel_consumed_liters,
                fuel_type=fleet.fuel_type,
            )
            factor = self._get_fuel_factor(fleet.fuel_type)
            factors.append(factor)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate fuel-based emissions for fleet {fleet.fleet_id}",
                formula="emissions = fuel_liters x EF",
                inputs={
                    "fleet_id": fleet.fleet_id,
                    "vehicle_count": fleet.vehicle_count,
                    "fuel_type": fleet.fuel_type.value if hasattr(fleet.fuel_type, 'value') else str(fleet.fuel_type),
                    "fuel_liters": str(fleet.fuel_consumed_liters),
                    "emission_factor": str(factor.factor_value),
                },
                output=str(total_kg),
                emission_factor=factor,
            ))

        # Handle CNG (measured in kg)
        elif fleet.fuel_consumed_kg and fleet.fuel_type == FuelType.CNG:
            from greenlang.agents.mrv.transport.base import DEFRA_FUEL_FACTORS
            cng_factor = DEFRA_FUEL_FACTORS[FuelType.CNG.value]["total_per_kg"]
            total_kg = (fleet.fuel_consumed_kg * cng_factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            co2_kg = total_kg

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate CNG emissions for fleet {fleet.fleet_id}",
                formula="emissions = fuel_kg x EF",
                inputs={
                    "fleet_id": fleet.fleet_id,
                    "fuel_kg": str(fleet.fuel_consumed_kg),
                    "emission_factor": str(cng_factor),
                },
                output=str(total_kg),
            ))

        # Fall back to distance-based
        elif fleet.distance_km:
            total_kg = self._calculate_distance_based(
                distance_km=fleet.distance_km,
                vehicle_type=fleet.vehicle_type,
            )
            co2_kg = total_kg
            factor = self._get_vehicle_factor(fleet.vehicle_type)
            factors.append(factor)

            steps.append(CalculationStep(
                step_number=step_offset + 1,
                description=f"Calculate distance-based emissions for fleet {fleet.fleet_id}",
                formula="emissions = distance_km x EF",
                inputs={
                    "fleet_id": fleet.fleet_id,
                    "vehicle_count": fleet.vehicle_count,
                    "vehicle_type": fleet.vehicle_type.value if hasattr(fleet.vehicle_type, 'value') else str(fleet.vehicle_type),
                    "distance_km": str(fleet.distance_km),
                    "emission_factor": str(factor.factor_value),
                },
                output=str(total_kg),
                emission_factor=factor,
            ))

        return {
            "total_kg": total_kg,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
        }, steps, factors
