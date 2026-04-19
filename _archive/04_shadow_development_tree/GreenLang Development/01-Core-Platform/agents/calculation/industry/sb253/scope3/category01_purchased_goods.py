# -*- coding: utf-8 -*-
"""
Category 1: Purchased Goods and Services Calculator

Calculates emissions from the cradle-to-gate production of goods and services
purchased by the reporting organization. This is typically one of the largest
Scope 3 categories, often representing 40-60% of total Scope 3 emissions.

Supported Methods:
1. Spend-based method (EPA EEIO factors)
2. Supplier-specific method (primary data from suppliers)
3. Hybrid method (combination of spend and supplier data)
4. Industry-average method (physical quantities x average factors)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category01PurchasedGoodsCalculator()
    >>> input_data = PurchasedGoodsInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     spend_data=[
    ...         SpendItem(description="Steel", spend_usd=500000, naics_code="331"),
    ...         SpendItem(description="Electronics", spend_usd=200000, naics_code="334"),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any

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


class SpendItem(BaseModel):
    """Individual spend line item for purchased goods/services."""

    description: str = Field(..., description="Description of good/service")
    spend_usd: Decimal = Field(..., ge=0, description="Spend amount in USD")
    naics_code: Optional[str] = Field(None, description="NAICS code (2-6 digits)")
    sic_code: Optional[str] = Field(None, description="SIC code")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    supplier_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Supplier-provided emissions (kg CO2e)"
    )
    quantity: Optional[Decimal] = Field(None, ge=0, description="Physical quantity")
    quantity_unit: Optional[str] = Field(None, description="Unit for quantity")
    category: Optional[str] = Field(None, description="Spend category")

    @validator("naics_code")
    def validate_naics(cls, v: Optional[str]) -> Optional[str]:
        """Validate NAICS code format."""
        if v is not None:
            v = v.strip()
            if not v.isdigit() or len(v) < 2 or len(v) > 6:
                raise ValueError("NAICS code must be 2-6 digits")
        return v


class PurchasedGoodsInput(Scope3CalculationInput):
    """Input model for Category 1: Purchased Goods and Services."""

    # Spend data
    spend_data: List[SpendItem] = Field(
        default_factory=list, description="List of spend items"
    )

    # Total spend (alternative to line items)
    total_spend_usd: Optional[Decimal] = Field(
        None, ge=0, description="Total spend in USD (if not using line items)"
    )
    default_naics_code: Optional[str] = Field(
        None, description="Default NAICS for total spend"
    )

    # Supplier-specific data
    supplier_specific_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Total supplier-specific emissions (kg CO2e)"
    )

    # Industry-average data
    physical_quantities: Optional[List[Dict[str, Any]]] = Field(
        None, description="Physical quantities for activity-based calculation"
    )

    @validator("spend_data", "total_spend_usd", pre=True, always=True)
    def validate_spend_input(cls, v, values):
        """Ensure at least one form of spend data is provided."""
        return v

    def get_total_spend(self) -> Decimal:
        """Calculate total spend from line items or total."""
        if self.spend_data:
            return sum(item.spend_usd for item in self.spend_data)
        return self.total_spend_usd or Decimal("0")


# Industry-average emission factors (kg CO2e per unit)
# Source: Various LCA databases, GHG Protocol guidance
INDUSTRY_AVERAGE_FACTORS: Dict[str, Dict[str, Any]] = {
    "steel_primary": {
        "factor": Decimal("1.85"),
        "unit": "kg",
        "description": "Primary steel (blast furnace)",
    },
    "steel_recycled": {
        "factor": Decimal("0.42"),
        "unit": "kg",
        "description": "Recycled steel (EAF)",
    },
    "aluminum_primary": {
        "factor": Decimal("11.5"),
        "unit": "kg",
        "description": "Primary aluminum",
    },
    "aluminum_recycled": {
        "factor": Decimal("0.7"),
        "unit": "kg",
        "description": "Recycled aluminum",
    },
    "plastic_pet": {
        "factor": Decimal("2.73"),
        "unit": "kg",
        "description": "PET plastic",
    },
    "plastic_hdpe": {
        "factor": Decimal("1.93"),
        "unit": "kg",
        "description": "HDPE plastic",
    },
    "paper_virgin": {
        "factor": Decimal("1.09"),
        "unit": "kg",
        "description": "Virgin paper",
    },
    "paper_recycled": {
        "factor": Decimal("0.67"),
        "unit": "kg",
        "description": "Recycled paper",
    },
    "cement": {
        "factor": Decimal("0.91"),
        "unit": "kg",
        "description": "Portland cement",
    },
    "concrete": {
        "factor": Decimal("0.13"),
        "unit": "kg",
        "description": "Ready-mix concrete",
    },
    "glass": {
        "factor": Decimal("0.85"),
        "unit": "kg",
        "description": "Container glass",
    },
}


class Category01PurchasedGoodsCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 1: Purchased Goods and Services.

    Calculates cradle-to-gate emissions from purchased goods and services.
    Supports multiple calculation methods per GHG Protocol guidance.

    Attributes:
        CATEGORY_NUMBER: 1
        CATEGORY_NAME: "Purchased Goods and Services"

    Example:
        >>> calculator = Category01PurchasedGoodsCalculator()
        >>> result = calculator.calculate_spend_based(
        ...     spend_usd=Decimal("1000000"),
        ...     naics_code="331"
        ... )
        >>> print(f"Emissions: {result.emissions_mt_co2e} MT CO2e")
    """

    CATEGORY_NUMBER = 1
    CATEGORY_NAME = "Purchased Goods and Services"
    SUPPORTED_METHODS = [
        CalculationMethod.SPEND_BASED,
        CalculationMethod.SUPPLIER_SPECIFIC,
        CalculationMethod.HYBRID,
        CalculationMethod.INDUSTRY_AVERAGE,
    ]

    def __init__(self):
        """Initialize the Category 1 calculator."""
        super().__init__()
        self._industry_factors = INDUSTRY_AVERAGE_FACTORS

    def calculate(self, input_data: PurchasedGoodsInput) -> Scope3CalculationResult:
        """
        Calculate Category 1 emissions using the specified method.

        Args:
            input_data: Purchased goods input data

        Returns:
            Complete calculation result with audit trail

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        # Route to appropriate calculation method
        if input_data.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC:
            return self._calculate_supplier_specific(input_data, start_time)
        elif input_data.calculation_method == CalculationMethod.HYBRID:
            return self._calculate_hybrid(input_data, start_time)
        elif input_data.calculation_method == CalculationMethod.INDUSTRY_AVERAGE:
            return self._calculate_industry_average(input_data, start_time)
        else:
            return self._calculate_spend_based(input_data, start_time)

    def _calculate_spend_based(
        self,
        input_data: PurchasedGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using spend-based method.

        Formula: Emissions = SUM(Spend_i x EF_i)
        where EF_i is the EPA EEIO factor for sector i

        Args:
            input_data: Input data with spend information
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        emission_factor = None

        # Step 1: Validate and summarize input
        steps.append(CalculationStep(
            step_number=1,
            description="Validate input data and extract spend information",
            inputs={"num_items": len(input_data.spend_data)},
            output="Input validated successfully",
        ))

        # Step 2: Calculate emissions for each spend item
        if input_data.spend_data:
            for idx, item in enumerate(input_data.spend_data):
                factor = self._get_eeio_factor(naics_code=item.naics_code)
                item_emissions = self._calculate_spend_based_emissions(
                    item.spend_usd, factor
                )
                total_emissions_kg += item_emissions

                steps.append(CalculationStep(
                    step_number=2 + idx,
                    description=f"Calculate emissions for: {item.description}",
                    formula="emissions_kg = spend_usd x emission_factor",
                    inputs={
                        "spend_usd": str(item.spend_usd),
                        "naics_code": item.naics_code,
                        "emission_factor": str(factor.factor_value),
                    },
                    output=str(item_emissions),
                ))

                if emission_factor is None:
                    emission_factor = factor
        else:
            # Use total spend with default NAICS
            total_spend = input_data.total_spend_usd or Decimal("0")
            emission_factor = self._get_eeio_factor(
                naics_code=input_data.default_naics_code
            )
            total_emissions_kg = self._calculate_spend_based_emissions(
                total_spend, emission_factor
            )

            steps.append(CalculationStep(
                step_number=2,
                description="Calculate emissions from total spend",
                formula="emissions_kg = total_spend_usd x emission_factor",
                inputs={
                    "total_spend_usd": str(total_spend),
                    "emission_factor": str(emission_factor.factor_value),
                },
                output=str(total_emissions_kg),
            ))

            if total_spend == 0:
                warnings.append("Total spend is zero - emissions will be zero")

        # Final step: Summarize
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum all emissions",
            formula="total_emissions = SUM(item_emissions)",
            inputs={},
            output=str(total_emissions_kg),
        ))

        activity_data = {
            "total_spend_usd": str(input_data.get_total_spend()),
            "num_items": len(input_data.spend_data),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.SPEND_BASED,
            emission_factor=emission_factor or self._get_eeio_factor(),
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_supplier_specific(
        self,
        input_data: PurchasedGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using supplier-specific method.

        Uses actual emissions data provided by suppliers.
        Highest data quality but requires supplier engagement.

        Args:
            input_data: Input data with supplier emissions
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Collect supplier-specific emissions data",
            inputs={"method": "supplier_specific"},
        ))

        # Sum supplier-provided emissions
        for idx, item in enumerate(input_data.spend_data):
            if item.supplier_emissions_kg is not None:
                total_emissions_kg += item.supplier_emissions_kg
                steps.append(CalculationStep(
                    step_number=2 + idx,
                    description=f"Add supplier emissions: {item.supplier_name or item.description}",
                    inputs={"supplier_emissions_kg": str(item.supplier_emissions_kg)},
                    output=str(total_emissions_kg),
                ))

        # Add any bulk supplier emissions
        if input_data.supplier_specific_emissions_kg:
            total_emissions_kg += input_data.supplier_specific_emissions_kg
            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Add bulk supplier-specific emissions",
                inputs={
                    "bulk_emissions_kg": str(input_data.supplier_specific_emissions_kg)
                },
                output=str(total_emissions_kg),
            ))

        # Create a supplier-specific emission factor record
        emission_factor = EmissionFactorRecord(
            factor_id="supplier_specific",
            factor_value=Decimal("1.0"),
            factor_unit="kg CO2e/kg CO2e",
            source=EmissionFactorSource.SUPPLIER_SPECIFIC,
            source_uri="",
            version="2024",
            last_updated=datetime.utcnow().strftime("%Y-%m-%d"),
            data_quality_tier=DataQualityTier.TIER_1,
            geographic_scope="supplier-specific",
        )

        activity_data = {
            "total_supplier_emissions_kg": str(total_emissions_kg),
            "num_suppliers": len([i for i in input_data.spend_data if i.supplier_emissions_kg]),
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

    def _calculate_hybrid(
        self,
        input_data: PurchasedGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using hybrid method.

        Combines supplier-specific data where available with
        spend-based estimates for remaining items.

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        supplier_emissions_kg = Decimal("0")
        spend_based_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize hybrid calculation",
            inputs={"method": "hybrid"},
        ))

        # Process items with supplier data first
        for item in input_data.spend_data:
            if item.supplier_emissions_kg is not None:
                supplier_emissions_kg += item.supplier_emissions_kg
                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Use supplier data: {item.description}",
                    inputs={"supplier_emissions_kg": str(item.supplier_emissions_kg)},
                ))
            else:
                # Fall back to spend-based
                factor = self._get_eeio_factor(naics_code=item.naics_code)
                item_emissions = self._calculate_spend_based_emissions(
                    item.spend_usd, factor
                )
                spend_based_emissions_kg += item_emissions
                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Use spend-based estimate: {item.description}",
                    formula="emissions = spend x factor",
                    inputs={
                        "spend_usd": str(item.spend_usd),
                        "factor": str(factor.factor_value),
                    },
                    output=str(item_emissions),
                ))

        total_emissions_kg = supplier_emissions_kg + spend_based_emissions_kg

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum supplier and spend-based emissions",
            formula="total = supplier_emissions + spend_based_emissions",
            inputs={
                "supplier_emissions_kg": str(supplier_emissions_kg),
                "spend_based_emissions_kg": str(spend_based_emissions_kg),
            },
            output=str(total_emissions_kg),
        ))

        # Use average factor for reporting
        emission_factor = self._get_eeio_factor()

        activity_data = {
            "supplier_emissions_kg": str(supplier_emissions_kg),
            "spend_based_emissions_kg": str(spend_based_emissions_kg),
            "total_spend_usd": str(input_data.get_total_spend()),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.HYBRID,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
        )

    def _calculate_industry_average(
        self,
        input_data: PurchasedGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using industry-average method.

        Uses physical quantities and industry-average emission factors.

        Formula: Emissions = SUM(Quantity_i x Industry_EF_i)

        Args:
            input_data: Input data with physical quantities
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize industry-average calculation",
            inputs={"method": "industry_average"},
        ))

        if input_data.physical_quantities:
            for idx, pq in enumerate(input_data.physical_quantities):
                material = pq.get("material", "").lower()
                quantity = Decimal(str(pq.get("quantity", 0)))
                unit = pq.get("unit", "kg")

                if material in self._industry_factors:
                    factor_data = self._industry_factors[material]
                    factor = factor_data["factor"]

                    # Convert quantity to kg if needed
                    if unit != "kg":
                        # Simple conversions
                        if unit == "tonnes" or unit == "t":
                            quantity = quantity * 1000
                        elif unit == "lbs":
                            quantity = quantity * Decimal("0.453592")

                    item_emissions = quantity * factor
                    total_emissions_kg += item_emissions

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate emissions for {material}",
                        formula="emissions = quantity x factor",
                        inputs={
                            "material": material,
                            "quantity_kg": str(quantity),
                            "factor_kg_co2e_per_kg": str(factor),
                        },
                        output=str(item_emissions),
                    ))
                else:
                    warnings.append(f"No industry factor for material: {material}")

        # Create industry-average factor record
        emission_factor = EmissionFactorRecord(
            factor_id="industry_average_mix",
            factor_value=Decimal("1.0"),
            factor_unit="kg CO2e/kg",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
            geographic_scope="global",
        )

        activity_data = {
            "num_materials": len(input_data.physical_quantities or []),
            "total_emissions_kg": str(total_emissions_kg),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.INDUSTRY_AVERAGE,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_spend_based_emissions(
        self,
        spend_usd: Decimal,
        emission_factor: EmissionFactorRecord,
    ) -> Decimal:
        """
        Calculate emissions from spend using EEIO factor.

        Args:
            spend_usd: Spend amount in USD
            emission_factor: EEIO emission factor

        Returns:
            Emissions in kg CO2e
        """
        emissions = spend_usd * emission_factor.factor_value
        return emissions.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
