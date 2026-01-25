# -*- coding: utf-8 -*-
"""
Category 8: Upstream Leased Assets Calculator

Calculates emissions from the operation of assets leased by the reporting
organization (lessee) that are not included in Scope 1 or 2.

This category applies when the reporting organization is the lessee.
For downstream leased assets (where org is lessor), see Category 13.

Supported Methods:
1. Asset-specific method (detailed asset data)
2. Lessor-specific method (data from lessor)
3. Average data method (industry averages)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category08UpstreamLeasedAssetsCalculator()
    >>> input_data = UpstreamLeasedAssetsInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     leased_assets=[
    ...         LeasedAsset(asset_type="office_building", floor_area_sqm=5000),
    ...         LeasedAsset(asset_type="vehicle", quantity=20, vehicle_type="sedan"),
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


class LeasedAsset(BaseModel):
    """Individual leased asset."""

    asset_type: str = Field(..., description="Type of leased asset")
    description: Optional[str] = Field(None, description="Asset description")

    # Building-specific
    floor_area_sqm: Optional[Decimal] = Field(None, ge=0, description="Floor area (sqm)")
    floor_area_sqft: Optional[Decimal] = Field(None, ge=0, description="Floor area (sqft)")
    building_type: Optional[str] = Field(None, description="Building type")

    # Vehicle-specific
    quantity: int = Field(1, ge=1, description="Number of units")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    annual_km: Optional[Decimal] = Field(None, ge=0, description="Annual km per vehicle")
    fuel_type: Optional[str] = Field(None, description="Fuel type")

    # Equipment-specific
    power_rating_kw: Optional[Decimal] = Field(None, ge=0, description="Power rating (kW)")
    operating_hours: Optional[Decimal] = Field(None, ge=0, description="Annual operating hours")

    # Energy consumption (if known)
    electricity_kwh: Optional[Decimal] = Field(None, ge=0, description="Annual electricity (kWh)")
    natural_gas_therms: Optional[Decimal] = Field(None, ge=0, description="Annual gas (therms)")

    # Lessor data
    lessor_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Lessor-provided emissions"
    )

    @validator("asset_type")
    def normalize_asset_type(cls, v: str) -> str:
        """Normalize asset type."""
        type_map = {
            "office": "office_building",
            "office_building": "office_building",
            "retail": "retail_building",
            "warehouse": "warehouse_building",
            "industrial": "industrial_building",
            "vehicle": "vehicle",
            "car": "vehicle",
            "truck": "vehicle",
            "van": "vehicle",
            "equipment": "equipment",
            "data_center": "data_center",
            "it_equipment": "it_equipment",
        }
        normalized = v.lower().strip().replace(" ", "_")
        return type_map.get(normalized, normalized)

    def get_floor_area_sqm(self) -> Decimal:
        """Get floor area in sqm."""
        if self.floor_area_sqm:
            return self.floor_area_sqm
        if self.floor_area_sqft:
            return (self.floor_area_sqft * Decimal("0.0929")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        return Decimal("0")


class UpstreamLeasedAssetsInput(Scope3CalculationInput):
    """Input model for Category 8: Upstream Leased Assets."""

    # Leased asset data
    leased_assets: List[LeasedAsset] = Field(
        default_factory=list, description="List of leased assets"
    )

    # Aggregated inputs
    total_leased_floor_area_sqm: Optional[Decimal] = Field(
        None, ge=0, description="Total leased floor area (sqm)"
    )
    total_leased_vehicles: Optional[int] = Field(
        None, ge=0, description="Total leased vehicles"
    )

    # Configuration
    default_building_type: str = Field(
        "office", description="Default building type"
    )


# Building emission factors (kg CO2e per sqm per year)
# Based on typical building energy consumption and grid factors
BUILDING_FACTORS: Dict[str, Decimal] = {
    "office_building": Decimal("85"),  # ~90 kWh/sqm electricity
    "retail_building": Decimal("125"),  # Higher lighting/HVAC
    "warehouse_building": Decimal("45"),  # Lower intensity
    "industrial_building": Decimal("120"),  # Process loads
    "data_center": Decimal("850"),  # Very high energy intensity
    "default": Decimal("85"),
}

# Vehicle emission factors (kg CO2e per year)
VEHICLE_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "sedan": {
        "annual_emissions": Decimal("3500"),  # ~15,000 km/year
        "km_factor": Decimal("0.23"),
    },
    "suv": {
        "annual_emissions": Decimal("5200"),
        "km_factor": Decimal("0.32"),
    },
    "pickup": {
        "annual_emissions": Decimal("5800"),
        "km_factor": Decimal("0.36"),
    },
    "van": {
        "annual_emissions": Decimal("6500"),
        "km_factor": Decimal("0.40"),
    },
    "truck": {
        "annual_emissions": Decimal("12000"),
        "km_factor": Decimal("0.75"),
    },
    "default": {
        "annual_emissions": Decimal("4500"),
        "km_factor": Decimal("0.28"),
    },
}

# Equipment emission factors (kg CO2e per kWh)
EQUIPMENT_FACTORS: Dict[str, Decimal] = {
    "equipment": Decimal("0.42"),  # Grid electricity factor
    "it_equipment": Decimal("0.42"),
}


class Category08UpstreamLeasedAssetsCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 8: Upstream Leased Assets.

    Calculates emissions from leased assets where the organization is the lessee.

    Attributes:
        CATEGORY_NUMBER: 8
        CATEGORY_NAME: "Upstream Leased Assets"

    Example:
        >>> calculator = Category08UpstreamLeasedAssetsCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 8
    CATEGORY_NAME = "Upstream Leased Assets"
    SUPPORTED_METHODS = [
        CalculationMethod.ASSET_SPECIFIC,
        CalculationMethod.AVERAGE_DATA,
        CalculationMethod.SUPPLIER_SPECIFIC,
    ]

    def __init__(self):
        """Initialize the Category 8 calculator."""
        super().__init__()
        self._building_factors = BUILDING_FACTORS
        self._vehicle_factors = VEHICLE_FACTORS
        self._equipment_factors = EQUIPMENT_FACTORS

    def calculate(
        self, input_data: UpstreamLeasedAssetsInput
    ) -> Scope3CalculationResult:
        """
        Calculate Category 8 emissions.

        Args:
            input_data: Upstream leased assets input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        if input_data.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC:
            return self._calculate_lessor_specific(input_data, start_time)
        elif input_data.calculation_method == CalculationMethod.AVERAGE_DATA:
            return self._calculate_average_data(input_data, start_time)
        else:
            return self._calculate_asset_specific(input_data, start_time)

    def _calculate_asset_specific(
        self,
        input_data: UpstreamLeasedAssetsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using asset-specific method.

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        building_emissions = Decimal("0")
        vehicle_emissions = Decimal("0")
        equipment_emissions = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize upstream leased assets calculation",
            inputs={"num_assets": len(input_data.leased_assets)},
        ))

        for asset in input_data.leased_assets:
            if asset.asset_type in ["office_building", "retail_building",
                                    "warehouse_building", "industrial_building",
                                    "data_center"]:
                emissions = self._calculate_building_emissions(asset)
                building_emissions += emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate emissions for {asset.asset_type}",
                    formula="emissions = floor_area x factor",
                    inputs={
                        "asset_type": asset.asset_type,
                        "floor_area_sqm": str(asset.get_floor_area_sqm()),
                    },
                    output=str(emissions),
                ))

            elif asset.asset_type == "vehicle":
                emissions = self._calculate_vehicle_emissions(asset)
                vehicle_emissions += emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate emissions for leased vehicles",
                    inputs={
                        "vehicle_type": asset.vehicle_type or "default",
                        "quantity": asset.quantity,
                        "annual_km": str(asset.annual_km) if asset.annual_km else "default",
                    },
                    output=str(emissions),
                ))

            elif asset.asset_type in ["equipment", "it_equipment"]:
                emissions = self._calculate_equipment_emissions(asset)
                equipment_emissions += emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate emissions for {asset.asset_type}",
                    inputs={
                        "power_rating_kw": str(asset.power_rating_kw) if asset.power_rating_kw else "N/A",
                        "operating_hours": str(asset.operating_hours) if asset.operating_hours else "N/A",
                        "electricity_kwh": str(asset.electricity_kwh) if asset.electricity_kwh else "N/A",
                    },
                    output=str(emissions),
                ))

        total_emissions_kg = building_emissions + vehicle_emissions + equipment_emissions

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum all leased asset emissions",
            inputs={
                "building_emissions_kg": str(building_emissions),
                "vehicle_emissions_kg": str(vehicle_emissions),
                "equipment_emissions_kg": str(equipment_emissions),
            },
            output=str(total_emissions_kg),
        ))

        emission_factor = EmissionFactorRecord(
            factor_id="leased_assets_composite",
            factor_value=Decimal("85"),  # Office building default
            factor_unit="kg CO2e/sqm/year",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "num_leased_assets": len(input_data.leased_assets),
            "building_emissions_kg": str(building_emissions),
            "vehicle_emissions_kg": str(vehicle_emissions),
            "equipment_emissions_kg": str(equipment_emissions),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.ASSET_SPECIFIC,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_lessor_specific(
        self,
        input_data: UpstreamLeasedAssetsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using lessor-specific data.

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Collect lessor-specific emissions data",
        ))

        for asset in input_data.leased_assets:
            if asset.lessor_emissions_kg:
                total_emissions_kg += asset.lessor_emissions_kg

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Add lessor emissions for {asset.description or asset.asset_type}",
                    inputs={"lessor_emissions_kg": str(asset.lessor_emissions_kg)},
                    output=str(asset.lessor_emissions_kg),
                ))

        emission_factor = EmissionFactorRecord(
            factor_id="lessor_specific",
            factor_value=Decimal("1.0"),
            factor_unit="kg CO2e",
            source=EmissionFactorSource.SUPPLIER_SPECIFIC,
            source_uri="",
            version="2024",
            last_updated=datetime.utcnow().strftime("%Y-%m-%d"),
            data_quality_tier=DataQualityTier.TIER_1,
        )

        activity_data = {
            "num_leased_assets": len(input_data.leased_assets),
            "lessor_provided_emissions_kg": str(total_emissions_kg),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.SUPPLIER_SPECIFIC,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
        )

    def _calculate_average_data(
        self,
        input_data: UpstreamLeasedAssetsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using average data method.

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
            description="Initialize average data calculation",
        ))

        # Calculate from aggregated floor area
        if input_data.total_leased_floor_area_sqm:
            factor = self._building_factors.get(
                input_data.default_building_type + "_building",
                self._building_factors["default"]
            )
            building_emissions = (
                input_data.total_leased_floor_area_sqm * factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_emissions_kg += building_emissions

            steps.append(CalculationStep(
                step_number=2,
                description="Calculate from total leased floor area",
                formula="emissions = floor_area x factor",
                inputs={
                    "floor_area_sqm": str(input_data.total_leased_floor_area_sqm),
                    "building_type": input_data.default_building_type,
                    "factor": str(factor),
                },
                output=str(building_emissions),
            ))

        # Calculate from aggregated vehicles
        if input_data.total_leased_vehicles:
            vehicle_factor = self._vehicle_factors["default"]["annual_emissions"]
            vehicle_emissions = (
                Decimal(str(input_data.total_leased_vehicles)) * vehicle_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_emissions_kg += vehicle_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate from total leased vehicles",
                inputs={
                    "num_vehicles": input_data.total_leased_vehicles,
                    "annual_factor": str(vehicle_factor),
                },
                output=str(vehicle_emissions),
            ))

        emission_factor = EmissionFactorRecord(
            factor_id="leased_assets_average",
            factor_value=Decimal("85"),
            factor_unit="kg CO2e/sqm/year",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_3,
        )

        activity_data = {
            "total_leased_floor_area_sqm": str(input_data.total_leased_floor_area_sqm or 0),
            "total_leased_vehicles": input_data.total_leased_vehicles or 0,
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.AVERAGE_DATA,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_building_emissions(self, asset: LeasedAsset) -> Decimal:
        """Calculate building emissions."""
        floor_area = asset.get_floor_area_sqm()
        if floor_area == 0:
            return Decimal("0")

        factor = self._building_factors.get(
            asset.asset_type, self._building_factors["default"]
        )

        return (floor_area * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_vehicle_emissions(self, asset: LeasedAsset) -> Decimal:
        """Calculate vehicle emissions."""
        vehicle_type = asset.vehicle_type or "default"
        vehicle_data = self._vehicle_factors.get(
            vehicle_type.lower(), self._vehicle_factors["default"]
        )

        if asset.annual_km:
            # Use actual km
            emissions_per_vehicle = asset.annual_km * vehicle_data["km_factor"]
        else:
            # Use default annual emissions
            emissions_per_vehicle = vehicle_data["annual_emissions"]

        total_emissions = emissions_per_vehicle * asset.quantity
        return total_emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _calculate_equipment_emissions(self, asset: LeasedAsset) -> Decimal:
        """Calculate equipment emissions."""
        # Use actual electricity if known
        if asset.electricity_kwh:
            factor = self._equipment_factors.get(
                asset.asset_type, self._equipment_factors["equipment"]
            )
            return (asset.electricity_kwh * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

        # Estimate from power rating and operating hours
        if asset.power_rating_kw and asset.operating_hours:
            kwh = asset.power_rating_kw * asset.operating_hours
            factor = self._equipment_factors.get(
                asset.asset_type, self._equipment_factors["equipment"]
            )
            return (kwh * factor).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        return Decimal("0")
