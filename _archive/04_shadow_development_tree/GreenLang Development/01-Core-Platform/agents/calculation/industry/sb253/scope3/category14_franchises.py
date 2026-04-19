# -*- coding: utf-8 -*-
"""
Category 14: Franchises Calculator

Calculates emissions from the operation of franchises not included in
Scope 1 or 2. This category applies to franchisors.

For franchisees, franchise emissions are typically Scope 1 and 2.

Supported Methods:
1. Franchise-specific method (detailed franchise data)
2. Average data method (industry averages)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category14FranchisesCalculator()
    >>> input_data = FranchisesInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     franchises=[
    ...         Franchise(franchise_type="restaurant", locations=500, avg_floor_area_sqm=250),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .base import (
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


class Franchise(BaseModel):
    """Individual franchise or franchise group."""

    franchise_type: str = Field(..., description="Type of franchise")
    franchise_name: Optional[str] = Field(None, description="Franchise name")
    locations: int = Field(..., ge=0, description="Number of locations")

    # Building data
    avg_floor_area_sqm: Optional[Decimal] = Field(
        None, ge=0, description="Average floor area per location (sqm)"
    )
    total_floor_area_sqm: Optional[Decimal] = Field(
        None, ge=0, description="Total floor area all locations (sqm)"
    )

    # Energy data (if available)
    total_electricity_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Total electricity all locations (kWh)"
    )
    total_gas_therms: Optional[Decimal] = Field(
        None, ge=0, description="Total gas all locations (therms)"
    )

    # Franchise-reported data
    franchise_reported_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Franchise-reported emissions"
    )

    # Vehicle fleet (if applicable)
    fleet_vehicles: Optional[int] = Field(None, ge=0, description="Fleet vehicles")
    fleet_annual_km: Optional[Decimal] = Field(None, ge=0, description="Fleet annual km")

    @validator("franchise_type")
    def normalize_franchise_type(cls, v: str) -> str:
        """Normalize franchise type."""
        return v.lower().strip().replace(" ", "_")


class FranchisesInput(Scope3CalculationInput):
    """Input model for Category 14: Franchises."""

    franchises: List[Franchise] = Field(
        default_factory=list, description="List of franchises"
    )

    # Aggregated inputs
    total_franchise_locations: Optional[int] = Field(
        None, ge=0, description="Total franchise locations"
    )
    default_franchise_type: str = Field(
        "retail", description="Default franchise type"
    )


# Franchise emission factors by type (kg CO2e per location per year)
# Based on typical energy consumption patterns
FRANCHISE_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "restaurant": {
        "per_location": Decimal("85000"),  # High energy for cooking, HVAC
        "per_sqm": Decimal("350"),
    },
    "fast_food": {
        "per_location": Decimal("95000"),
        "per_sqm": Decimal("380"),
    },
    "coffee_shop": {
        "per_location": Decimal("45000"),
        "per_sqm": Decimal("320"),
    },
    "convenience_store": {
        "per_location": Decimal("65000"),
        "per_sqm": Decimal("280"),  # Refrigeration
    },
    "retail": {
        "per_location": Decimal("55000"),
        "per_sqm": Decimal("125"),
    },
    "hotel": {
        "per_location": Decimal("320000"),
        "per_sqm": Decimal("85"),
    },
    "gym": {
        "per_location": Decimal("120000"),
        "per_sqm": Decimal("150"),
    },
    "gas_station": {
        "per_location": Decimal("75000"),
        "per_sqm": Decimal("200"),
    },
    "car_rental": {
        "per_location": Decimal("45000"),
        "per_sqm": Decimal("100"),
    },
    "default": {
        "per_location": Decimal("60000"),
        "per_sqm": Decimal("125"),
    },
}


class Category14FranchisesCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 14: Franchises.

    Calculates emissions from franchise operations (franchisor perspective).
    """

    CATEGORY_NUMBER = 14
    CATEGORY_NAME = "Franchises"
    SUPPORTED_METHODS = [
        CalculationMethod.ACTIVITY_BASED,
        CalculationMethod.AVERAGE_DATA,
        CalculationMethod.SUPPLIER_SPECIFIC,  # Franchise-specific
    ]

    def __init__(self):
        """Initialize the Category 14 calculator."""
        super().__init__()
        self._franchise_factors = FRANCHISE_FACTORS

    def calculate(self, input_data: FranchisesInput) -> Scope3CalculationResult:
        """Calculate Category 14 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_locations = 0

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize franchises calculation",
            inputs={"num_franchise_groups": len(input_data.franchises)},
        ))

        if input_data.franchises:
            for franchise in input_data.franchises:
                total_locations += franchise.locations

                # Use franchise-reported data if available
                if franchise.franchise_reported_emissions_kg:
                    franchise_emissions = franchise.franchise_reported_emissions_kg
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Use franchise-reported: {franchise.franchise_name or franchise.franchise_type}",
                        inputs={"reported_emissions_kg": str(franchise_emissions)},
                        output=str(franchise_emissions),
                    ))

                # Calculate from energy data
                elif franchise.total_electricity_kwh or franchise.total_gas_therms:
                    franchise_emissions = Decimal("0")

                    if franchise.total_electricity_kwh:
                        grid_factor = Decimal("0.42")
                        elec_emissions = franchise.total_electricity_kwh * grid_factor
                        franchise_emissions += elec_emissions

                    if franchise.total_gas_therms:
                        gas_factor = Decimal("5.3")  # kg CO2e per therm
                        gas_emissions = franchise.total_gas_therms * gas_factor
                        franchise_emissions += gas_emissions

                    franchise_emissions = franchise_emissions.quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate from energy: {franchise.franchise_name or franchise.franchise_type}",
                        inputs={
                            "electricity_kwh": str(franchise.total_electricity_kwh or 0),
                            "gas_therms": str(franchise.total_gas_therms or 0),
                        },
                        output=str(franchise_emissions),
                    ))

                # Calculate from floor area
                elif franchise.total_floor_area_sqm or franchise.avg_floor_area_sqm:
                    factors = self._franchise_factors.get(
                        franchise.franchise_type,
                        self._franchise_factors["default"]
                    )

                    if franchise.total_floor_area_sqm:
                        floor_area = franchise.total_floor_area_sqm
                    else:
                        floor_area = (
                            franchise.avg_floor_area_sqm * franchise.locations
                        )

                    franchise_emissions = (
                        floor_area * factors["per_sqm"]
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate from floor area: {franchise.franchise_name or franchise.franchise_type}",
                        inputs={
                            "franchise_type": franchise.franchise_type,
                            "floor_area_sqm": str(floor_area),
                            "factor_per_sqm": str(factors["per_sqm"]),
                        },
                        output=str(franchise_emissions),
                    ))

                # Calculate from location count
                else:
                    factors = self._franchise_factors.get(
                        franchise.franchise_type,
                        self._franchise_factors["default"]
                    )
                    franchise_emissions = (
                        Decimal(str(franchise.locations)) * factors["per_location"]
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate from locations: {franchise.franchise_name or franchise.franchise_type}",
                        inputs={
                            "franchise_type": franchise.franchise_type,
                            "locations": franchise.locations,
                            "factor_per_location": str(factors["per_location"]),
                        },
                        output=str(franchise_emissions),
                    ))

                # Add fleet emissions if applicable
                if franchise.fleet_vehicles and franchise.fleet_annual_km:
                    vehicle_factor = Decimal("0.23")  # kg CO2e per km
                    fleet_emissions = (
                        franchise.fleet_annual_km * vehicle_factor
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                    franchise_emissions += fleet_emissions

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Add fleet emissions",
                        inputs={
                            "vehicles": franchise.fleet_vehicles,
                            "annual_km": str(franchise.fleet_annual_km),
                        },
                        output=str(fleet_emissions),
                    ))

                total_emissions_kg += franchise_emissions
        else:
            # Use aggregated data
            if input_data.total_franchise_locations:
                factors = self._franchise_factors.get(
                    input_data.default_franchise_type,
                    self._franchise_factors["default"]
                )
                total_emissions_kg = (
                    Decimal(str(input_data.total_franchise_locations)) *
                    factors["per_location"]
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                total_locations = input_data.total_franchise_locations

                steps.append(CalculationStep(
                    step_number=2,
                    description="Calculate from total locations",
                    inputs={
                        "total_locations": total_locations,
                        "franchise_type": input_data.default_franchise_type,
                        "factor_per_location": str(factors["per_location"]),
                    },
                    output=str(total_emissions_kg),
                ))

        emission_factor = EmissionFactorRecord(
            factor_id="franchises_composite",
            factor_value=Decimal("60000"),
            factor_unit="kg CO2e/location/year",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_franchise_locations": total_locations,
            "num_franchise_groups": len(input_data.franchises),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=input_data.calculation_method,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )
