# -*- coding: utf-8 -*-
"""
Category 2: Capital Goods Calculator

Calculates emissions from the cradle-to-gate production of capital goods
purchased by the reporting organization. Capital goods are final products
with an extended life that are used to manufacture products, provide services,
or sell, store, and deliver merchandise.

Supported Methods:
1. Spend-based method (EPA EEIO factors)
2. Asset-specific method (detailed asset data)
3. Depreciation-weighted method (emissions allocated by depreciation)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category02CapitalGoodsCalculator()
    >>> input_data = CapitalGoodsInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     capital_expenditures=[
    ...         CapitalAsset(description="Manufacturing Equipment", cost_usd=2000000, naics_code="333"),
    ...         CapitalAsset(description="Building", cost_usd=5000000, naics_code="236"),
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


class CapitalAsset(BaseModel):
    """Individual capital asset purchase."""

    description: str = Field(..., description="Asset description")
    cost_usd: Decimal = Field(..., ge=0, description="Purchase cost in USD")
    naics_code: Optional[str] = Field(None, description="NAICS code for asset category")
    asset_type: Optional[str] = Field(None, description="Asset type classification")
    useful_life_years: Optional[int] = Field(
        None, ge=1, le=100, description="Useful life in years"
    )
    purchase_date: Optional[str] = Field(None, description="Purchase date (YYYY-MM-DD)")
    supplier_emissions_kg: Optional[Decimal] = Field(
        None, ge=0, description="Supplier-provided emissions"
    )
    depreciation_method: Optional[str] = Field(
        "straight_line", description="Depreciation method"
    )
    quantity: Optional[int] = Field(1, ge=1, description="Number of units")

    @validator("asset_type")
    def validate_asset_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate asset type."""
        valid_types = [
            "machinery", "equipment", "building", "vehicle",
            "furniture", "computer", "software", "other"
        ]
        if v and v.lower() not in valid_types:
            logger.warning(f"Asset type '{v}' not in standard list: {valid_types}")
        return v


class CapitalGoodsInput(Scope3CalculationInput):
    """Input model for Category 2: Capital Goods."""

    # Capital asset data
    capital_expenditures: List[CapitalAsset] = Field(
        default_factory=list, description="List of capital assets purchased"
    )

    # Total capex (alternative to line items)
    total_capex_usd: Optional[Decimal] = Field(
        None, ge=0, description="Total capital expenditure in USD"
    )
    default_naics_code: Optional[str] = Field(
        None, description="Default NAICS for total capex"
    )

    # Depreciation-weighted allocation
    use_depreciation_weighting: bool = Field(
        False, description="Allocate emissions based on depreciation"
    )

    def get_total_capex(self) -> Decimal:
        """Calculate total capital expenditure."""
        if self.capital_expenditures:
            return sum(
                asset.cost_usd * (asset.quantity or 1)
                for asset in self.capital_expenditures
            )
        return self.total_capex_usd or Decimal("0")


# Default useful life by asset type (years)
DEFAULT_USEFUL_LIFE: Dict[str, int] = {
    "machinery": 10,
    "equipment": 7,
    "building": 40,
    "vehicle": 5,
    "furniture": 7,
    "computer": 3,
    "software": 5,
    "other": 7,
}


class Category02CapitalGoodsCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 2: Capital Goods.

    Calculates cradle-to-gate emissions from capital goods purchases.
    Emissions are typically recognized in the year of acquisition,
    though depreciation-weighted allocation is also supported.

    Attributes:
        CATEGORY_NUMBER: 2
        CATEGORY_NAME: "Capital Goods"

    Example:
        >>> calculator = Category02CapitalGoodsCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 2
    CATEGORY_NAME = "Capital Goods"
    SUPPORTED_METHODS = [
        CalculationMethod.SPEND_BASED,
        CalculationMethod.ASSET_SPECIFIC,
        CalculationMethod.SUPPLIER_SPECIFIC,
    ]

    def __init__(self):
        """Initialize the Category 2 calculator."""
        super().__init__()
        self._default_useful_life = DEFAULT_USEFUL_LIFE

    def calculate(self, input_data: CapitalGoodsInput) -> Scope3CalculationResult:
        """
        Calculate Category 2 emissions.

        Args:
            input_data: Capital goods input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        if input_data.use_depreciation_weighting:
            return self._calculate_depreciation_weighted(input_data, start_time)
        elif input_data.calculation_method == CalculationMethod.ASSET_SPECIFIC:
            return self._calculate_asset_specific(input_data, start_time)
        elif input_data.calculation_method == CalculationMethod.SUPPLIER_SPECIFIC:
            return self._calculate_supplier_specific(input_data, start_time)
        else:
            return self._calculate_spend_based(input_data, start_time)

    def _calculate_spend_based(
        self,
        input_data: CapitalGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using spend-based method.

        Formula: Emissions = SUM(CapEx_i x EF_i)

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        emission_factor = None

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize spend-based capital goods calculation",
            inputs={
                "num_assets": len(input_data.capital_expenditures),
                "total_capex": str(input_data.get_total_capex()),
            },
        ))

        if input_data.capital_expenditures:
            for idx, asset in enumerate(input_data.capital_expenditures):
                factor = self._get_eeio_factor(naics_code=asset.naics_code)
                asset_cost = asset.cost_usd * (asset.quantity or 1)
                asset_emissions = asset_cost * factor.factor_value
                asset_emissions = asset_emissions.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_emissions_kg += asset_emissions

                steps.append(CalculationStep(
                    step_number=2 + idx,
                    description=f"Calculate emissions for: {asset.description}",
                    formula="emissions = cost_usd x quantity x emission_factor",
                    inputs={
                        "cost_usd": str(asset.cost_usd),
                        "quantity": asset.quantity or 1,
                        "emission_factor": str(factor.factor_value),
                    },
                    output=str(asset_emissions),
                ))

                if emission_factor is None:
                    emission_factor = factor
        else:
            total_capex = input_data.total_capex_usd or Decimal("0")
            emission_factor = self._get_eeio_factor(
                naics_code=input_data.default_naics_code
            )
            total_emissions_kg = (total_capex * emission_factor.factor_value).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )

            steps.append(CalculationStep(
                step_number=2,
                description="Calculate emissions from total capex",
                formula="emissions = total_capex x emission_factor",
                inputs={
                    "total_capex_usd": str(total_capex),
                    "emission_factor": str(emission_factor.factor_value),
                },
                output=str(total_emissions_kg),
            ))

        activity_data = {
            "total_capex_usd": str(input_data.get_total_capex()),
            "num_assets": len(input_data.capital_expenditures),
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

    def _calculate_asset_specific(
        self,
        input_data: CapitalGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using asset-specific method.

        Uses detailed asset information and category-specific factors.

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
            description="Initialize asset-specific calculation",
            inputs={"num_assets": len(input_data.capital_expenditures)},
        ))

        for idx, asset in enumerate(input_data.capital_expenditures):
            # Get appropriate factor based on asset type
            factor = self._get_asset_specific_factor(asset)
            asset_cost = asset.cost_usd * (asset.quantity or 1)
            asset_emissions = (asset_cost * factor.factor_value).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_emissions_kg += asset_emissions

            steps.append(CalculationStep(
                step_number=2 + idx,
                description=f"Calculate asset-specific emissions: {asset.description}",
                formula="emissions = cost x factor",
                inputs={
                    "asset_type": asset.asset_type,
                    "cost_usd": str(asset_cost),
                    "factor": str(factor.factor_value),
                },
                output=str(asset_emissions),
            ))

        emission_factor = self._get_eeio_factor()

        activity_data = {
            "total_capex_usd": str(input_data.get_total_capex()),
            "num_assets": len(input_data.capital_expenditures),
            "asset_types": list(set(
                a.asset_type for a in input_data.capital_expenditures if a.asset_type
            )),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.ASSET_SPECIFIC,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
        )

    def _calculate_supplier_specific(
        self,
        input_data: CapitalGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using supplier-specific data.

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
            description="Collect supplier-specific emissions for capital goods",
        ))

        for idx, asset in enumerate(input_data.capital_expenditures):
            if asset.supplier_emissions_kg:
                emissions = asset.supplier_emissions_kg * (asset.quantity or 1)
                total_emissions_kg += emissions

                steps.append(CalculationStep(
                    step_number=2 + idx,
                    description=f"Add supplier emissions: {asset.description}",
                    inputs={
                        "supplier_emissions_kg": str(asset.supplier_emissions_kg),
                        "quantity": asset.quantity or 1,
                    },
                    output=str(emissions),
                ))

        emission_factor = EmissionFactorRecord(
            factor_id="supplier_specific_capital",
            factor_value=Decimal("1.0"),
            factor_unit="kg CO2e/kg CO2e",
            source=EmissionFactorSource.SUPPLIER_SPECIFIC,
            source_uri="",
            version="2024",
            last_updated=datetime.utcnow().strftime("%Y-%m-%d"),
            data_quality_tier=DataQualityTier.TIER_1,
        )

        activity_data = {
            "total_supplier_emissions_kg": str(total_emissions_kg),
            "num_assets": len(input_data.capital_expenditures),
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

    def _calculate_depreciation_weighted(
        self,
        input_data: CapitalGoodsInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions with depreciation weighting.

        Allocates total asset emissions across useful life.
        Reports only the current year's depreciated portion.

        Formula: Annual_Emissions = Total_Emissions / Useful_Life

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_annual_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize depreciation-weighted calculation",
            inputs={"method": "depreciation_weighted"},
        ))

        for idx, asset in enumerate(input_data.capital_expenditures):
            # Get useful life
            useful_life = asset.useful_life_years
            if not useful_life:
                asset_type = (asset.asset_type or "other").lower()
                useful_life = self._default_useful_life.get(asset_type, 7)
                warnings.append(
                    f"Using default useful life ({useful_life} years) for: {asset.description}"
                )

            # Calculate total lifetime emissions
            factor = self._get_eeio_factor(naics_code=asset.naics_code)
            asset_cost = asset.cost_usd * (asset.quantity or 1)
            total_emissions = asset_cost * factor.factor_value

            # Calculate annual depreciated emissions (straight-line)
            annual_emissions = (total_emissions / Decimal(str(useful_life))).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_annual_emissions_kg += annual_emissions

            steps.append(CalculationStep(
                step_number=2 + idx,
                description=f"Calculate depreciated emissions: {asset.description}",
                formula="annual_emissions = (cost x factor) / useful_life",
                inputs={
                    "cost_usd": str(asset_cost),
                    "factor": str(factor.factor_value),
                    "useful_life_years": useful_life,
                },
                output=str(annual_emissions),
            ))

        emission_factor = self._get_eeio_factor()

        activity_data = {
            "total_capex_usd": str(input_data.get_total_capex()),
            "num_assets": len(input_data.capital_expenditures),
            "depreciation_method": "straight_line",
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_annual_emissions_kg,
            method=CalculationMethod.SPEND_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _get_asset_specific_factor(
        self,
        asset: CapitalAsset
    ) -> EmissionFactorRecord:
        """Get emission factor based on asset characteristics."""
        # First try NAICS
        if asset.naics_code:
            return self._get_eeio_factor(naics_code=asset.naics_code)

        # Map asset type to NAICS
        asset_type_to_naics = {
            "machinery": "333",
            "equipment": "333",
            "building": "236",
            "vehicle": "336",
            "furniture": "337",
            "computer": "334",
            "software": "51",
        }

        naics = asset_type_to_naics.get(
            (asset.asset_type or "").lower(),
            "default"
        )
        return self._get_eeio_factor(naics_code=naics)
