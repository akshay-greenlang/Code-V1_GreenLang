# -*- coding: utf-8 -*-
"""
Category 12: End-of-Life Treatment of Sold Products Calculator

Calculates emissions from the waste disposal and treatment of products
sold by the reporting organization at the end of their useful life.

Includes:
1. Landfill emissions
2. Incineration emissions
3. Recycling processing
4. Other treatment methods

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category12EndOfLifeCalculator()
    >>> input_data = EndOfLifeInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     products_sold=[
    ...         ProductEOL(product="electronics", units_sold=50000, weight_kg=2.5),
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


class ProductEOL(BaseModel):
    """Product end-of-life data."""

    product: str = Field(..., description="Product type")
    product_name: Optional[str] = Field(None, description="Product name")
    units_sold: int = Field(..., ge=0, description="Units sold")
    weight_kg: Optional[Decimal] = Field(None, ge=0, description="Weight per unit (kg)")
    total_weight_kg: Optional[Decimal] = Field(None, ge=0, description="Total weight sold (kg)")

    # Material composition (percentages)
    material_composition: Optional[Dict[str, Decimal]] = Field(
        None, description="Material composition {material: percentage}"
    )

    # EOL treatment splits
    landfill_pct: Decimal = Field(Decimal("60"), ge=0, le=100, description="% to landfill")
    incineration_pct: Decimal = Field(Decimal("10"), ge=0, le=100, description="% incinerated")
    recycling_pct: Decimal = Field(Decimal("30"), ge=0, le=100, description="% recycled")
    composting_pct: Decimal = Field(Decimal("0"), ge=0, le=100, description="% composted")

    @validator("product")
    def normalize_product(cls, v: str) -> str:
        """Normalize product name."""
        return v.lower().strip().replace(" ", "_")

    def get_total_weight_kg(self) -> Decimal:
        """Get total weight sold."""
        if self.total_weight_kg:
            return self.total_weight_kg
        if self.weight_kg:
            return self.weight_kg * self.units_sold
        return Decimal("0")


class EndOfLifeInput(Scope3CalculationInput):
    """Input model for Category 12: End-of-Life Treatment."""

    products_sold: List[ProductEOL] = Field(
        default_factory=list, description="List of products sold"
    )

    # Aggregated inputs
    total_products_weight_kg: Optional[Decimal] = Field(
        None, ge=0, description="Total weight of products sold (kg)"
    )

    # Include recycling credits
    include_recycling_credit: bool = Field(
        False, description="Include emission credits for recycling"
    )


# End-of-life emission factors by material and treatment (kg CO2e per kg)
EOL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "mixed_products": {
        "landfill": Decimal("0.587"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-0.182"),
        "composting": Decimal("0.036"),
    },
    "plastic": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("2.760"),
        "recycling": Decimal("-1.440"),
    },
    "metal": {
        "landfill": Decimal("0.021"),
        "recycling": Decimal("-2.500"),
    },
    "aluminum": {
        "landfill": Decimal("0.021"),
        "recycling": Decimal("-9.120"),
    },
    "steel": {
        "landfill": Decimal("0.021"),
        "recycling": Decimal("-1.820"),
    },
    "glass": {
        "landfill": Decimal("0.021"),
        "recycling": Decimal("-0.315"),
    },
    "paper": {
        "landfill": Decimal("1.095"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-0.680"),
        "composting": Decimal("0.036"),
    },
    "electronics": {
        "landfill": Decimal("0.021"),
        "recycling": Decimal("-2.500"),
    },
    "textile": {
        "landfill": Decimal("0.021"),
        "incineration": Decimal("2.760"),
        "recycling": Decimal("-2.130"),
    },
    "wood": {
        "landfill": Decimal("0.729"),
        "incineration": Decimal("0.021"),
        "recycling": Decimal("-0.516"),
        "composting": Decimal("0.036"),
    },
    "organic": {
        "landfill": Decimal("1.824"),
        "composting": Decimal("0.036"),
    },
}

# Default material composition by product type
DEFAULT_COMPOSITIONS: Dict[str, Dict[str, Decimal]] = {
    "electronics": {
        "plastic": Decimal("40"),
        "metal": Decimal("35"),
        "glass": Decimal("15"),
        "other": Decimal("10"),
    },
    "appliance": {
        "metal": Decimal("60"),
        "plastic": Decimal("30"),
        "other": Decimal("10"),
    },
    "packaging": {
        "paper": Decimal("50"),
        "plastic": Decimal("30"),
        "other": Decimal("20"),
    },
    "furniture": {
        "wood": Decimal("60"),
        "metal": Decimal("20"),
        "textile": Decimal("15"),
        "other": Decimal("5"),
    },
    "clothing": {
        "textile": Decimal("90"),
        "other": Decimal("10"),
    },
    "default": {
        "mixed_products": Decimal("100"),
    },
}


class Category12EndOfLifeCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 12: End-of-Life Treatment of Sold Products.

    Calculates emissions from disposal of products at end of life.
    """

    CATEGORY_NUMBER = 12
    CATEGORY_NAME = "End-of-Life Treatment of Sold Products"
    SUPPORTED_METHODS = [
        CalculationMethod.ACTIVITY_BASED,
        CalculationMethod.AVERAGE_DATA,
    ]

    def __init__(self):
        """Initialize the Category 12 calculator."""
        super().__init__()
        self._eol_factors = EOL_FACTORS
        self._default_compositions = DEFAULT_COMPOSITIONS

    def calculate(self, input_data: EndOfLifeInput) -> Scope3CalculationResult:
        """Calculate Category 12 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_weight_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize end-of-life treatment calculation",
            inputs={
                "num_products": len(input_data.products_sold),
                "include_recycling_credit": input_data.include_recycling_credit,
            },
        ))

        for product in input_data.products_sold:
            product_weight = product.get_total_weight_kg()
            if product_weight == 0:
                warnings.append(f"No weight data for product: {product.product}")
                continue

            total_weight_kg += product_weight

            # Get material composition
            composition = product.material_composition
            if not composition:
                composition = self._default_compositions.get(
                    product.product,
                    self._default_compositions["default"]
                )

            # Calculate emissions by material and treatment
            product_emissions = Decimal("0")

            for material, pct in composition.items():
                material_weight = product_weight * (pct / 100)
                material_factors = self._eol_factors.get(
                    material, self._eol_factors["mixed_products"]
                )

                # Landfill
                if product.landfill_pct > 0:
                    landfill_weight = material_weight * (product.landfill_pct / 100)
                    landfill_factor = material_factors.get("landfill", Decimal("0.587"))
                    landfill_emissions = landfill_weight * landfill_factor
                    product_emissions += landfill_emissions

                # Incineration
                if product.incineration_pct > 0:
                    incin_weight = material_weight * (product.incineration_pct / 100)
                    incin_factor = material_factors.get("incineration", Decimal("0.021"))
                    incin_emissions = incin_weight * incin_factor
                    product_emissions += incin_emissions

                # Recycling
                if product.recycling_pct > 0:
                    recyc_weight = material_weight * (product.recycling_pct / 100)
                    recyc_factor = material_factors.get("recycling", Decimal("-0.182"))
                    if input_data.include_recycling_credit:
                        product_emissions += recyc_weight * recyc_factor
                    else:
                        # Only count processing emissions, not credit
                        product_emissions += recyc_weight * Decimal("0.021")

                # Composting
                if product.composting_pct > 0:
                    comp_weight = material_weight * (product.composting_pct / 100)
                    comp_factor = material_factors.get("composting", Decimal("0.036"))
                    product_emissions += comp_weight * comp_factor

            product_emissions = product_emissions.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += product_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description=f"Calculate EOL emissions: {product.product_name or product.product}",
                inputs={
                    "product": product.product,
                    "total_weight_kg": str(product_weight),
                    "landfill_pct": str(product.landfill_pct),
                    "incineration_pct": str(product.incineration_pct),
                    "recycling_pct": str(product.recycling_pct),
                },
                output=str(product_emissions),
            ))

        # Handle negative emissions (recycling credits)
        if total_emissions_kg < 0 and not input_data.include_recycling_credit:
            total_emissions_kg = Decimal("0")
            warnings.append("Recycling credits exceed emissions; set to zero")

        emission_factor = EmissionFactorRecord(
            factor_id="eol_composite",
            factor_value=Decimal("0.587"),
            factor_unit="kg CO2e/kg",
            source=EmissionFactorSource.EPA_GHG,
            source_uri="https://www.epa.gov/warm",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_weight_kg": str(total_weight_kg),
            "num_products": len(input_data.products_sold),
            "include_recycling_credit": input_data.include_recycling_credit,
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
