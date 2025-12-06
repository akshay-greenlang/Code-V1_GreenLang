# -*- coding: utf-8 -*-
"""
Category 13: Downstream Leased Assets Calculator

Calculates emissions from the operation of assets owned by the reporting
organization (lessor) and leased to other entities. This is the inverse
of Category 8 (Upstream Leased Assets).

This category applies when the reporting organization acts as lessor.

Supported Methods:
1. Asset-specific method (detailed asset data)
2. Lessee-specific method (data from lessees)
3. Average data method (industry averages)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category13DownstreamLeasedCalculator()
    >>> input_data = DownstreamLeasedInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     leased_out_assets=[
    ...         LeasedOutAsset(asset_type="office_building", floor_area_sqm=10000),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

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


class LeasedOutAsset(BaseModel):
    """Asset leased out to other entities (organization is lessor)."""

    asset_type: str = Field(..., description="Type of leased asset")
    description: Optional[str] = Field(None, description="Asset description")

    # Building-specific
    floor_area_sqm: Optional[Decimal] = Field(None, ge=0, description="Floor area (sqm)")
    building_type: Optional[str] = Field(None, description="Building type")

    # Vehicle-specific
    quantity: int = Field(1, ge=1, description="Number of units")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    annual_km_estimate: Optional[Decimal] = Field(None, ge=0, description="Estimated annual km")

    # Equipment-specific
    power_rating_kw: Optional[Decimal] = Field(None, ge=0, description="Power rating (kW)")
    estimated_operating_hours: Optional[Decimal] = Field(None, ge=0, description="Est. operating hours")

    # Lessee data
    lessee_reported_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Lessee-reported emissions"
    )
    lessee_name: Optional[str] = Field(None, description="Lessee name")

    @validator("asset_type")
    def normalize_asset_type(cls, v: str) -> str:
        """Normalize asset type."""
        return v.lower().strip().replace(" ", "_")


class DownstreamLeasedInput(Scope3CalculationInput):
    """Input model for Category 13: Downstream Leased Assets."""

    leased_out_assets: List[LeasedOutAsset] = Field(
        default_factory=list, description="List of assets leased out"
    )

    # Aggregated inputs
    total_leased_floor_area_sqm: Optional[Decimal] = Field(
        None, ge=0, description="Total floor area leased out (sqm)"
    )
    total_leased_vehicles: Optional[int] = Field(
        None, ge=0, description="Total vehicles leased out"
    )


# Same factors as Category 8 for consistency
DOWNSTREAM_BUILDING_FACTORS: Dict[str, Decimal] = {
    "office_building": Decimal("85"),
    "retail_building": Decimal("125"),
    "warehouse_building": Decimal("45"),
    "industrial_building": Decimal("120"),
    "residential_building": Decimal("55"),
    "data_center": Decimal("850"),
    "default": Decimal("85"),
}

DOWNSTREAM_VEHICLE_FACTORS: Dict[str, Decimal] = {
    "sedan": Decimal("3500"),
    "suv": Decimal("5200"),
    "truck": Decimal("12000"),
    "van": Decimal("6500"),
    "default": Decimal("4500"),
}


class Category13DownstreamLeasedCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 13: Downstream Leased Assets.

    Calculates emissions from assets leased out to third parties.
    """

    CATEGORY_NUMBER = 13
    CATEGORY_NAME = "Downstream Leased Assets"
    SUPPORTED_METHODS = [
        CalculationMethod.ASSET_SPECIFIC,
        CalculationMethod.AVERAGE_DATA,
        CalculationMethod.SUPPLIER_SPECIFIC,  # Lessee-specific
    ]

    def __init__(self):
        """Initialize the Category 13 calculator."""
        super().__init__()
        self._building_factors = DOWNSTREAM_BUILDING_FACTORS
        self._vehicle_factors = DOWNSTREAM_VEHICLE_FACTORS

    def calculate(
        self, input_data: DownstreamLeasedInput
    ) -> Scope3CalculationResult:
        """Calculate Category 13 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize downstream leased assets calculation",
            inputs={"num_assets": len(input_data.leased_out_assets)},
        ))

        if input_data.leased_out_assets:
            for asset in input_data.leased_out_assets:
                # Use lessee-reported data if available
                if asset.lessee_reported_emissions_kg:
                    asset_emissions = asset.lessee_reported_emissions_kg
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Use lessee-reported emissions: {asset.description or asset.asset_type}",
                        inputs={
                            "lessee": asset.lessee_name,
                            "emissions_kg": str(asset_emissions),
                        },
                        output=str(asset_emissions),
                    ))
                elif asset.asset_type.endswith("_building") or asset.floor_area_sqm:
                    # Calculate building emissions
                    floor_area = asset.floor_area_sqm or Decimal("0")
                    factor = self._building_factors.get(
                        asset.asset_type, self._building_factors["default"]
                    )
                    asset_emissions = (floor_area * factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate building emissions: {asset.description or asset.asset_type}",
                        formula="emissions = floor_area x factor",
                        inputs={
                            "asset_type": asset.asset_type,
                            "floor_area_sqm": str(floor_area),
                            "factor": str(factor),
                        },
                        output=str(asset_emissions),
                    ))
                elif asset.asset_type == "vehicle" or asset.vehicle_type:
                    # Calculate vehicle emissions
                    v_type = asset.vehicle_type or "default"
                    factor = self._vehicle_factors.get(
                        v_type.lower(), self._vehicle_factors["default"]
                    )
                    asset_emissions = (Decimal(str(asset.quantity)) * factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate vehicle emissions",
                        inputs={
                            "vehicle_type": v_type,
                            "quantity": asset.quantity,
                            "annual_factor": str(factor),
                        },
                        output=str(asset_emissions),
                    ))
                else:
                    warnings.append(f"Unable to calculate emissions for: {asset.asset_type}")
                    continue

                total_emissions_kg += asset_emissions
        else:
            # Use aggregated inputs
            if input_data.total_leased_floor_area_sqm:
                factor = self._building_factors["default"]
                building_emissions = (
                    input_data.total_leased_floor_area_sqm * factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                total_emissions_kg += building_emissions

                steps.append(CalculationStep(
                    step_number=2,
                    description="Calculate from total leased floor area",
                    inputs={
                        "floor_area_sqm": str(input_data.total_leased_floor_area_sqm),
                        "factor": str(factor),
                    },
                    output=str(building_emissions),
                ))

            if input_data.total_leased_vehicles:
                factor = self._vehicle_factors["default"]
                vehicle_emissions = (
                    Decimal(str(input_data.total_leased_vehicles)) * factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                total_emissions_kg += vehicle_emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description="Calculate from total leased vehicles",
                    inputs={
                        "num_vehicles": input_data.total_leased_vehicles,
                        "annual_factor": str(factor),
                    },
                    output=str(vehicle_emissions),
                ))

        emission_factor = EmissionFactorRecord(
            factor_id="downstream_leased_composite",
            factor_value=Decimal("85"),
            factor_unit="kg CO2e/sqm/year",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "num_leased_assets": len(input_data.leased_out_assets),
            "total_floor_area_sqm": str(input_data.total_leased_floor_area_sqm or 0),
            "total_vehicles": input_data.total_leased_vehicles or 0,
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
