# -*- coding: utf-8 -*-
"""
Category 10: Processing of Sold Products Calculator

Calculates emissions from processing of sold intermediate products
by third parties (e.g., manufacturers) after sale by the reporting
organization and before end use.

This category is relevant for companies that sell intermediate products
that require further processing, transformation, or inclusion in another
product before use by the end consumer.

Supported Methods:
1. Site-specific method (actual processing data)
2. Average data method (industry average processing factors)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category10ProcessingCalculator()
    >>> input_data = ProcessingInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     intermediate_products=[
    ...         IntermediateProduct(product="steel_coil", quantity_tonnes=1000),
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


class IntermediateProduct(BaseModel):
    """Intermediate product sold for further processing."""

    product: str = Field(..., description="Product type")
    description: Optional[str] = Field(None, description="Product description")
    quantity_tonnes: Optional[Decimal] = Field(None, ge=0, description="Quantity (tonnes)")
    quantity_units: Optional[Decimal] = Field(None, ge=0, description="Quantity (units)")
    unit: Optional[str] = Field(None, description="Unit if not tonnes")

    # Processing details
    processing_type: Optional[str] = Field(None, description="Type of processing")
    processor_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Processor-provided emissions"
    )
    processing_energy_kwh: Optional[Decimal] = Field(
        None, ge=0, description="Processing energy consumption"
    )

    @validator("product")
    def normalize_product(cls, v: str) -> str:
        """Normalize product name."""
        return v.lower().strip().replace(" ", "_")


class ProcessingInput(Scope3CalculationInput):
    """Input model for Category 10: Processing of Sold Products."""

    intermediate_products: List[IntermediateProduct] = Field(
        default_factory=list, description="List of intermediate products"
    )

    # Aggregated inputs
    total_intermediate_products_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Total intermediate products (tonnes)"
    )
    default_processing_type: str = Field(
        "manufacturing", description="Default processing type"
    )


# Processing emission factors (kg CO2e per tonne of product processed)
PROCESSING_FACTORS: Dict[str, Decimal] = {
    # Metals processing
    "steel_coil": Decimal("180"),  # Stamping, forming
    "steel_sheet": Decimal("150"),
    "aluminum_sheet": Decimal("250"),
    "aluminum_extrusion": Decimal("220"),

    # Plastics processing
    "plastic_pellets": Decimal("450"),  # Injection molding
    "plastic_resin": Decimal("380"),
    "polymer": Decimal("400"),

    # Chemicals
    "chemical_intermediate": Decimal("320"),
    "petrochemical": Decimal("450"),

    # Paper/pulp
    "pulp": Decimal("280"),  # Paper manufacturing
    "paper_stock": Decimal("220"),

    # Textiles
    "yarn": Decimal("350"),  # Weaving, finishing
    "fabric_roll": Decimal("280"),

    # Electronics
    "semiconductor": Decimal("850"),
    "electronic_component": Decimal("620"),

    # Food
    "food_ingredient": Decimal("150"),
    "agricultural_commodity": Decimal("120"),

    # Default
    "default": Decimal("250"),
}


class Category10ProcessingCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 10: Processing of Sold Products.

    Calculates emissions from downstream processing of intermediate products.
    """

    CATEGORY_NUMBER = 10
    CATEGORY_NAME = "Processing of Sold Products"
    SUPPORTED_METHODS = [
        CalculationMethod.ACTIVITY_BASED,
        CalculationMethod.AVERAGE_DATA,
        CalculationMethod.SUPPLIER_SPECIFIC,
    ]

    def __init__(self):
        """Initialize the Category 10 calculator."""
        super().__init__()
        self._processing_factors = PROCESSING_FACTORS

    def calculate(self, input_data: ProcessingInput) -> Scope3CalculationResult:
        """Calculate Category 10 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_quantity_tonnes = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize processing of sold products calculation",
            inputs={"num_products": len(input_data.intermediate_products)},
        ))

        if input_data.intermediate_products:
            for product in input_data.intermediate_products:
                # Use processor-provided data if available
                if product.processor_emissions_kg:
                    product_emissions = product.processor_emissions_kg
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Use processor data for {product.product}",
                        inputs={"processor_emissions_kg": str(product_emissions)},
                        output=str(product_emissions),
                    ))
                elif product.processing_energy_kwh:
                    # Estimate from energy consumption
                    grid_factor = Decimal("0.42")  # kg CO2e/kWh
                    product_emissions = (
                        product.processing_energy_kwh * grid_factor
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate from energy: {product.product}",
                        formula="emissions = energy_kwh x grid_factor",
                        inputs={
                            "energy_kwh": str(product.processing_energy_kwh),
                            "grid_factor": str(grid_factor),
                        },
                        output=str(product_emissions),
                    ))
                elif product.quantity_tonnes:
                    # Use average factor
                    factor = self._processing_factors.get(
                        product.product, self._processing_factors["default"]
                    )
                    product_emissions = (product.quantity_tonnes * factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    total_quantity_tonnes += product.quantity_tonnes
                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate processing emissions: {product.product}",
                        formula="emissions = quantity_tonnes x processing_factor",
                        inputs={
                            "product": product.product,
                            "quantity_tonnes": str(product.quantity_tonnes),
                            "factor": str(factor),
                        },
                        output=str(product_emissions),
                    ))
                else:
                    warnings.append(f"No quantity data for product: {product.product}")
                    continue

                total_emissions_kg += product_emissions
        else:
            # Use aggregated data
            if input_data.total_intermediate_products_tonnes:
                factor = self._processing_factors["default"]
                total_emissions_kg = (
                    input_data.total_intermediate_products_tonnes * factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                total_quantity_tonnes = input_data.total_intermediate_products_tonnes

                steps.append(CalculationStep(
                    step_number=2,
                    description="Calculate from total intermediate products",
                    inputs={
                        "total_tonnes": str(input_data.total_intermediate_products_tonnes),
                        "factor": str(factor),
                    },
                    output=str(total_emissions_kg),
                ))

        emission_factor = EmissionFactorRecord(
            factor_id="processing_composite",
            factor_value=Decimal("250"),
            factor_unit="kg CO2e/tonne",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_quantity_tonnes": str(total_quantity_tonnes),
            "num_products": len(input_data.intermediate_products),
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
