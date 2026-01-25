# -*- coding: utf-8 -*-
"""
Category 11: Use of Sold Products Calculator

Calculates emissions from the use of goods and services sold by the
reporting organization. This category is often significant for companies
that sell energy-consuming products.

Includes:
1. Direct use-phase emissions (combustion of fuels sold)
2. Indirect use-phase emissions (electricity consumption during use)

Supported Methods:
1. Product-specific method (detailed product data)
2. Average data method (industry averages)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category11ProductUseCalculator()
    >>> input_data = ProductUseInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     products_sold=[
    ...         ProductSold(product_type="appliance", units_sold=10000, lifetime_years=10, annual_kwh=500),
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


class ProductSold(BaseModel):
    """Product sold with use-phase emission data."""

    product_type: str = Field(..., description="Type of product")
    product_name: Optional[str] = Field(None, description="Product name")
    units_sold: int = Field(..., ge=0, description="Units sold")

    # Lifetime
    lifetime_years: Optional[Decimal] = Field(None, ge=0, description="Product lifetime (years)")
    lifetime_uses: Optional[Decimal] = Field(None, ge=0, description="Lifetime uses/cycles")

    # Direct emissions (for fuel-using products)
    fuel_consumption_per_use: Optional[Decimal] = Field(
        None, ge=0, description="Fuel per use (liters/units)"
    )
    fuel_type: Optional[str] = Field(None, description="Fuel type")
    uses_per_year: Optional[Decimal] = Field(None, ge=0, description="Uses per year")

    # Indirect emissions (electricity-using products)
    annual_kwh: Optional[Decimal] = Field(None, ge=0, description="Annual kWh consumption")
    power_rating_watts: Optional[Decimal] = Field(None, ge=0, description="Power rating (watts)")
    daily_use_hours: Optional[Decimal] = Field(None, ge=0, description="Daily use hours")

    # Refrigerant emissions
    refrigerant_type: Optional[str] = Field(None, description="Refrigerant type")
    refrigerant_charge_kg: Optional[Decimal] = Field(None, ge=0, description="Refrigerant charge (kg)")
    annual_leak_rate: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Annual leak rate (0-1)"
    )

    @validator("product_type")
    def normalize_product_type(cls, v: str) -> str:
        """Normalize product type."""
        return v.lower().strip().replace(" ", "_")


class ProductUseInput(Scope3CalculationInput):
    """Input model for Category 11: Use of Sold Products."""

    products_sold: List[ProductSold] = Field(
        default_factory=list, description="List of products sold"
    )

    # For fuel products
    fuels_sold: Optional[Dict[str, Decimal]] = Field(
        None, description="Fuels sold {fuel_type: quantity}"
    )

    # Configuration
    grid_emission_factor: Decimal = Field(
        Decimal("0.42"), ge=0, description="Grid emission factor (kg CO2e/kWh)"
    )


# Default product lifetimes (years)
DEFAULT_LIFETIMES: Dict[str, Decimal] = {
    "vehicle": Decimal("12"),
    "appliance": Decimal("12"),
    "hvac": Decimal("15"),
    "refrigerator": Decimal("14"),
    "electronics": Decimal("5"),
    "computer": Decimal("4"),
    "phone": Decimal("3"),
    "tv": Decimal("8"),
    "lighting": Decimal("10"),
    "industrial_equipment": Decimal("20"),
    "default": Decimal("10"),
}

# Fuel emission factors (kg CO2e per liter)
FUEL_EMISSION_FACTORS: Dict[str, Decimal] = {
    "gasoline": Decimal("2.31"),
    "diesel": Decimal("2.68"),
    "natural_gas": Decimal("2.02"),  # per m3
    "propane": Decimal("1.51"),
    "lpg": Decimal("1.51"),
}

# Refrigerant GWP values (100-year)
REFRIGERANT_GWP: Dict[str, Decimal] = {
    "r134a": Decimal("1430"),
    "r410a": Decimal("2088"),
    "r32": Decimal("675"),
    "r404a": Decimal("3922"),
    "r407c": Decimal("1774"),
    "r290": Decimal("3"),  # Propane
    "r600a": Decimal("3"),  # Isobutane
    "co2": Decimal("1"),
    "ammonia": Decimal("0"),
}


class Category11ProductUseCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 11: Use of Sold Products.

    Calculates lifetime use-phase emissions of products sold.
    """

    CATEGORY_NUMBER = 11
    CATEGORY_NAME = "Use of Sold Products"
    SUPPORTED_METHODS = [
        CalculationMethod.ACTIVITY_BASED,
        CalculationMethod.AVERAGE_DATA,
    ]

    def __init__(self):
        """Initialize the Category 11 calculator."""
        super().__init__()
        self._lifetimes = DEFAULT_LIFETIMES
        self._fuel_factors = FUEL_EMISSION_FACTORS
        self._refrigerant_gwp = REFRIGERANT_GWP

    def calculate(self, input_data: ProductUseInput) -> Scope3CalculationResult:
        """Calculate Category 11 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        direct_emissions = Decimal("0")
        indirect_emissions = Decimal("0")
        refrigerant_emissions = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize use of sold products calculation",
            inputs={"num_products": len(input_data.products_sold)},
        ))

        # Calculate for each product
        for product in input_data.products_sold:
            lifetime = product.lifetime_years or self._lifetimes.get(
                product.product_type, self._lifetimes["default"]
            )

            # Direct emissions (fuel-using products)
            if product.fuel_consumption_per_use and product.uses_per_year:
                fuel_factor = self._fuel_factors.get(
                    product.fuel_type or "gasoline",
                    Decimal("2.31")
                )
                annual_fuel = product.fuel_consumption_per_use * product.uses_per_year
                lifetime_fuel = annual_fuel * lifetime
                product_direct = (
                    Decimal(str(product.units_sold)) * lifetime_fuel * fuel_factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                direct_emissions += product_direct

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate direct emissions: {product.product_name or product.product_type}",
                    formula="emissions = units x lifetime_fuel x fuel_factor",
                    inputs={
                        "units_sold": product.units_sold,
                        "lifetime_years": str(lifetime),
                        "fuel_type": product.fuel_type or "gasoline",
                        "annual_fuel": str(annual_fuel),
                    },
                    output=str(product_direct),
                ))

            # Indirect emissions (electricity-using products)
            if product.annual_kwh:
                annual_energy = product.annual_kwh
            elif product.power_rating_watts and product.daily_use_hours:
                # Calculate from power rating
                # Annual kWh = (watts/1000) x daily_hours x 365
                annual_energy = (
                    product.power_rating_watts / 1000 *
                    product.daily_use_hours * 365
                )
            else:
                annual_energy = None

            if annual_energy:
                lifetime_kwh = annual_energy * lifetime
                product_indirect = (
                    Decimal(str(product.units_sold)) *
                    lifetime_kwh *
                    input_data.grid_emission_factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                indirect_emissions += product_indirect

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate indirect emissions: {product.product_name or product.product_type}",
                    formula="emissions = units x lifetime_kwh x grid_factor",
                    inputs={
                        "units_sold": product.units_sold,
                        "lifetime_years": str(lifetime),
                        "annual_kwh": str(annual_energy),
                        "grid_factor": str(input_data.grid_emission_factor),
                    },
                    output=str(product_indirect),
                ))

            # Refrigerant emissions
            if product.refrigerant_type and product.refrigerant_charge_kg:
                gwp = self._refrigerant_gwp.get(
                    product.refrigerant_type.lower(),
                    Decimal("1500")  # Default mid-range GWP
                )
                leak_rate = product.annual_leak_rate or Decimal("0.02")  # 2% default
                lifetime_leakage = product.refrigerant_charge_kg * leak_rate * lifetime
                product_refrigerant = (
                    Decimal(str(product.units_sold)) * lifetime_leakage * gwp
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                refrigerant_emissions += product_refrigerant

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate refrigerant emissions: {product.product_name or product.product_type}",
                    inputs={
                        "refrigerant": product.refrigerant_type,
                        "charge_kg": str(product.refrigerant_charge_kg),
                        "leak_rate": str(leak_rate),
                        "gwp": str(gwp),
                    },
                    output=str(product_refrigerant),
                ))

        # Handle direct fuel sales
        if input_data.fuels_sold:
            for fuel_type, quantity in input_data.fuels_sold.items():
                factor = self._fuel_factors.get(fuel_type.lower(), Decimal("2.31"))
                fuel_emissions = (quantity * factor * Decimal("1000")).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )  # Convert to kg
                direct_emissions += fuel_emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate combustion of fuel sold: {fuel_type}",
                    inputs={
                        "fuel_type": fuel_type,
                        "quantity": str(quantity),
                        "factor": str(factor),
                    },
                    output=str(fuel_emissions),
                ))

        total_emissions_kg = direct_emissions + indirect_emissions + refrigerant_emissions

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum all use-phase emissions",
            inputs={
                "direct_emissions_kg": str(direct_emissions),
                "indirect_emissions_kg": str(indirect_emissions),
                "refrigerant_emissions_kg": str(refrigerant_emissions),
            },
            output=str(total_emissions_kg),
        ))

        emission_factor = EmissionFactorRecord(
            factor_id="product_use_composite",
            factor_value=input_data.grid_emission_factor,
            factor_unit="kg CO2e/kWh",
            source=EmissionFactorSource.EPA_GHG,
            source_uri="https://www.epa.gov/ghgemissions/emission-factors-hub",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "num_products": len(input_data.products_sold),
            "total_units_sold": sum(p.units_sold for p in input_data.products_sold),
            "direct_emissions_kg": str(direct_emissions),
            "indirect_emissions_kg": str(indirect_emissions),
            "refrigerant_emissions_kg": str(refrigerant_emissions),
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
