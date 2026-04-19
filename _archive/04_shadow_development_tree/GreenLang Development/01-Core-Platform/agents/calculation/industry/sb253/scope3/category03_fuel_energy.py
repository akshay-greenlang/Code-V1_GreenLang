# -*- coding: utf-8 -*-
"""
Category 3: Fuel and Energy Related Activities Calculator

Calculates emissions from fuel and energy-related activities not included
in Scope 1 or Scope 2. This includes:

1. Upstream emissions of purchased fuels (extraction, production, transportation)
2. Upstream emissions of purchased electricity (generation fuel extraction)
3. Transmission and distribution (T&D) losses
4. Generation of purchased electricity sold to end users (for utilities)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category03FuelEnergyCalculator()
    >>> input_data = FuelEnergyInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     fuel_purchases=[
    ...         FuelPurchase(fuel_type="natural_gas", quantity=10000, unit="therms"),
    ...         FuelPurchase(fuel_type="diesel", quantity=5000, unit="gallons"),
    ...     ],
    ...     electricity_kwh=1000000,
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


class FuelPurchase(BaseModel):
    """Individual fuel purchase record."""

    fuel_type: str = Field(..., description="Type of fuel")
    quantity: Decimal = Field(..., ge=0, description="Quantity purchased")
    unit: str = Field(..., description="Unit (gallons, liters, therms, MMBtu, etc.)")
    supplier: Optional[str] = Field(None, description="Fuel supplier")
    upstream_factor_override: Optional[Decimal] = Field(
        None, ge=0, description="Custom upstream emission factor"
    )

    @validator("fuel_type")
    def normalize_fuel_type(cls, v: str) -> str:
        """Normalize fuel type name."""
        return v.lower().strip().replace(" ", "_")


class FuelEnergyInput(Scope3CalculationInput):
    """Input model for Category 3: Fuel and Energy Related Activities."""

    # Fuel purchase data
    fuel_purchases: List[FuelPurchase] = Field(
        default_factory=list, description="List of fuel purchases"
    )

    # Electricity data
    electricity_kwh: Decimal = Field(
        Decimal("0"), ge=0, description="Total electricity purchased (kWh)"
    )
    electricity_grid_region: Optional[str] = Field(
        None, description="Grid region for electricity"
    )

    # T&D loss calculation
    include_td_losses: bool = Field(
        True, description="Include transmission & distribution losses"
    )
    custom_td_loss_rate: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Custom T&D loss rate (0-1)"
    )

    # Steam/Heat data
    steam_purchased_mmbtu: Decimal = Field(
        Decimal("0"), ge=0, description="Steam purchased (MMBtu)"
    )
    heat_purchased_mmbtu: Decimal = Field(
        Decimal("0"), ge=0, description="Heat purchased (MMBtu)"
    )
    cooling_purchased_mmbtu: Decimal = Field(
        Decimal("0"), ge=0, description="Cooling purchased (MMBtu)"
    )


# Well-to-Tank (WTT) emission factors for upstream fuel emissions
# Source: DEFRA 2024, EPA GHG Emission Factors Hub
WTT_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # Liquid fuels (kg CO2e per gallon)
    "gasoline": {"wtt_factor": Decimal("2.21"), "unit": "gallon"},
    "diesel": {"wtt_factor": Decimal("2.50"), "unit": "gallon"},
    "biodiesel": {"wtt_factor": Decimal("0.82"), "unit": "gallon"},
    "jet_fuel": {"wtt_factor": Decimal("2.43"), "unit": "gallon"},
    "fuel_oil": {"wtt_factor": Decimal("2.67"), "unit": "gallon"},
    "propane": {"wtt_factor": Decimal("1.53"), "unit": "gallon"},
    "lpg": {"wtt_factor": Decimal("1.53"), "unit": "gallon"},

    # Gaseous fuels (kg CO2e per therm/MMBtu)
    "natural_gas": {"wtt_factor": Decimal("1.18"), "unit": "therm"},
    "natural_gas_mmbtu": {"wtt_factor": Decimal("11.8"), "unit": "MMBtu"},

    # Solid fuels (kg CO2e per short ton)
    "coal_bituminous": {"wtt_factor": Decimal("178.0"), "unit": "short_ton"},
    "coal_sub_bituminous": {"wtt_factor": Decimal("156.0"), "unit": "short_ton"},
    "coal_lignite": {"wtt_factor": Decimal("132.0"), "unit": "short_ton"},
}

# Upstream electricity emission factors (kg CO2e per kWh)
# Represents emissions from fuel extraction/production for power generation
UPSTREAM_ELECTRICITY_FACTORS: Dict[str, Decimal] = {
    "US_national": Decimal("0.042"),  # US national average
    "WECC": Decimal("0.038"),  # Western grid
    "ERCOT": Decimal("0.035"),  # Texas grid
    "SERC": Decimal("0.048"),  # Southeast grid
    "MRO": Decimal("0.052"),  # Midwest grid
    "NPCC": Decimal("0.034"),  # Northeast grid
    "RFC": Decimal("0.044"),  # Mid-Atlantic grid
    "default": Decimal("0.042"),
}

# T&D loss rates by region
TD_LOSS_RATES: Dict[str, Decimal] = {
    "US_national": Decimal("0.052"),  # 5.2% average loss
    "WECC": Decimal("0.048"),
    "ERCOT": Decimal("0.051"),
    "SERC": Decimal("0.054"),
    "default": Decimal("0.052"),
}


class Category03FuelEnergyCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 3: Fuel and Energy Related Activities.

    Calculates upstream emissions from:
    - Purchased fuels (well-to-tank)
    - Purchased electricity (upstream of generation)
    - T&D losses for electricity
    - Purchased steam, heat, and cooling

    Attributes:
        CATEGORY_NUMBER: 3
        CATEGORY_NAME: "Fuel and Energy Related Activities"

    Example:
        >>> calculator = Category03FuelEnergyCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 3
    CATEGORY_NAME = "Fuel and Energy Related Activities"
    SUPPORTED_METHODS = [
        CalculationMethod.ACTIVITY_BASED,
        CalculationMethod.SUPPLIER_SPECIFIC,
    ]

    def __init__(self):
        """Initialize the Category 3 calculator."""
        super().__init__()
        self._wtt_factors = WTT_FACTORS
        self._upstream_elec_factors = UPSTREAM_ELECTRICITY_FACTORS
        self._td_loss_rates = TD_LOSS_RATES

    def calculate(self, input_data: FuelEnergyInput) -> Scope3CalculationResult:
        """
        Calculate Category 3 emissions.

        Args:
            input_data: Fuel and energy input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize Category 3 calculation",
            inputs={
                "num_fuel_purchases": len(input_data.fuel_purchases),
                "electricity_kwh": str(input_data.electricity_kwh),
                "include_td_losses": input_data.include_td_losses,
            },
        ))

        # Step 2: Calculate upstream fuel emissions (WTT)
        fuel_emissions = self._calculate_wtt_emissions(input_data, steps, warnings)
        total_emissions_kg += fuel_emissions

        # Step 3: Calculate upstream electricity emissions
        elec_upstream = self._calculate_upstream_electricity(input_data, steps, warnings)
        total_emissions_kg += elec_upstream

        # Step 4: Calculate T&D loss emissions
        if input_data.include_td_losses and input_data.electricity_kwh > 0:
            td_emissions = self._calculate_td_losses(input_data, steps, warnings)
            total_emissions_kg += td_emissions

        # Step 5: Calculate upstream steam/heat/cooling emissions
        thermal_emissions = self._calculate_thermal_upstream(input_data, steps, warnings)
        total_emissions_kg += thermal_emissions

        # Final summary step
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum all Category 3 emissions",
            formula="total = fuel_wtt + elec_upstream + td_losses + thermal_upstream",
            output=str(total_emissions_kg),
        ))

        emission_factor = EmissionFactorRecord(
            factor_id="category_3_composite",
            factor_value=Decimal("1.0"),
            factor_unit="kg CO2e",
            source=EmissionFactorSource.EPA_GHG,
            source_uri="https://www.epa.gov/ghgemissions/emission-factors-hub",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_fuel_purchases": len(input_data.fuel_purchases),
            "electricity_kwh": str(input_data.electricity_kwh),
            "include_td_losses": input_data.include_td_losses,
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.ACTIVITY_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_wtt_emissions(
        self,
        input_data: FuelEnergyInput,
        steps: List[CalculationStep],
        warnings: List[str],
    ) -> Decimal:
        """
        Calculate well-to-tank upstream emissions for purchased fuels.

        Args:
            input_data: Input data
            steps: Calculation steps list (modified in place)
            warnings: Warnings list (modified in place)

        Returns:
            WTT emissions in kg CO2e
        """
        total_wtt = Decimal("0")

        for fuel in input_data.fuel_purchases:
            wtt_factor = self._get_wtt_factor(fuel.fuel_type, fuel.unit)

            if wtt_factor is None:
                warnings.append(
                    f"No WTT factor found for fuel type: {fuel.fuel_type}. Using default."
                )
                wtt_factor = Decimal("0.20")  # Default ~20% of combustion emissions

            # Use override if provided
            if fuel.upstream_factor_override is not None:
                wtt_factor = fuel.upstream_factor_override

            fuel_wtt = (fuel.quantity * wtt_factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_wtt += fuel_wtt

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description=f"Calculate WTT emissions for {fuel.fuel_type}",
                formula="wtt_emissions = quantity x wtt_factor",
                inputs={
                    "fuel_type": fuel.fuel_type,
                    "quantity": str(fuel.quantity),
                    "unit": fuel.unit,
                    "wtt_factor": str(wtt_factor),
                },
                output=str(fuel_wtt),
            ))

        return total_wtt

    def _calculate_upstream_electricity(
        self,
        input_data: FuelEnergyInput,
        steps: List[CalculationStep],
        warnings: List[str],
    ) -> Decimal:
        """
        Calculate upstream emissions from purchased electricity.

        This represents emissions from fuel extraction and processing
        for the fuels used to generate the electricity (not the generation itself).

        Args:
            input_data: Input data
            steps: Calculation steps list
            warnings: Warnings list

        Returns:
            Upstream electricity emissions in kg CO2e
        """
        if input_data.electricity_kwh == 0:
            return Decimal("0")

        # Get upstream factor for grid region
        grid_region = input_data.electricity_grid_region or "default"
        upstream_factor = self._upstream_elec_factors.get(
            grid_region, self._upstream_elec_factors["default"]
        )

        upstream_emissions = (input_data.electricity_kwh * upstream_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Calculate upstream electricity emissions",
            formula="upstream_elec = electricity_kwh x upstream_factor",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh),
                "grid_region": grid_region,
                "upstream_factor": str(upstream_factor),
            },
            output=str(upstream_emissions),
        ))

        return upstream_emissions

    def _calculate_td_losses(
        self,
        input_data: FuelEnergyInput,
        steps: List[CalculationStep],
        warnings: List[str],
    ) -> Decimal:
        """
        Calculate emissions from transmission and distribution losses.

        T&D losses represent electricity lost in transmission that must
        be generated but is not delivered to the end user.

        Formula: T&D Emissions = Electricity x T&D Loss Rate x Grid Factor

        Args:
            input_data: Input data
            steps: Calculation steps list
            warnings: Warnings list

        Returns:
            T&D loss emissions in kg CO2e
        """
        # Get T&D loss rate
        if input_data.custom_td_loss_rate is not None:
            td_rate = input_data.custom_td_loss_rate
        else:
            grid_region = input_data.electricity_grid_region or "default"
            td_rate = self._td_loss_rates.get(
                grid_region, self._td_loss_rates["default"]
            )

        # Calculate lost electricity
        lost_kwh = input_data.electricity_kwh * td_rate

        # Get grid emission factor (simplified - would normally use regional factors)
        # Using US average grid factor
        grid_factor = Decimal("0.42")  # kg CO2e/kWh US average

        td_emissions = (lost_kwh * grid_factor).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Calculate T&D loss emissions",
            formula="td_emissions = electricity_kwh x td_rate x grid_factor",
            inputs={
                "electricity_kwh": str(input_data.electricity_kwh),
                "td_loss_rate": str(td_rate),
                "lost_kwh": str(lost_kwh),
                "grid_factor": str(grid_factor),
            },
            output=str(td_emissions),
        ))

        return td_emissions

    def _calculate_thermal_upstream(
        self,
        input_data: FuelEnergyInput,
        steps: List[CalculationStep],
        warnings: List[str],
    ) -> Decimal:
        """
        Calculate upstream emissions from purchased steam, heat, and cooling.

        Args:
            input_data: Input data
            steps: Calculation steps list
            warnings: Warnings list

        Returns:
            Thermal upstream emissions in kg CO2e
        """
        total_thermal = Decimal("0")

        # Steam upstream factor (kg CO2e per MMBtu)
        steam_factor = Decimal("7.5")  # Approximate upstream for natural gas
        if input_data.steam_purchased_mmbtu > 0:
            steam_emissions = (
                input_data.steam_purchased_mmbtu * steam_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_thermal += steam_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate upstream emissions for purchased steam",
                formula="steam_upstream = mmbtu x factor",
                inputs={
                    "steam_mmbtu": str(input_data.steam_purchased_mmbtu),
                    "factor": str(steam_factor),
                },
                output=str(steam_emissions),
            ))

        # Heat upstream factor
        heat_factor = Decimal("7.5")
        if input_data.heat_purchased_mmbtu > 0:
            heat_emissions = (
                input_data.heat_purchased_mmbtu * heat_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_thermal += heat_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate upstream emissions for purchased heat",
                inputs={
                    "heat_mmbtu": str(input_data.heat_purchased_mmbtu),
                    "factor": str(heat_factor),
                },
                output=str(heat_emissions),
            ))

        # Cooling upstream factor (electricity-based)
        cooling_factor = Decimal("12.0")  # Higher due to electricity for chillers
        if input_data.cooling_purchased_mmbtu > 0:
            cooling_emissions = (
                input_data.cooling_purchased_mmbtu * cooling_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_thermal += cooling_emissions

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate upstream emissions for purchased cooling",
                inputs={
                    "cooling_mmbtu": str(input_data.cooling_purchased_mmbtu),
                    "factor": str(cooling_factor),
                },
                output=str(cooling_emissions),
            ))

        return total_thermal

    def _get_wtt_factor(
        self,
        fuel_type: str,
        unit: str,
    ) -> Optional[Decimal]:
        """
        Get well-to-tank emission factor for a fuel type.

        Args:
            fuel_type: Type of fuel
            unit: Unit of measurement

        Returns:
            WTT factor in kg CO2e per unit, or None if not found
        """
        fuel_key = fuel_type.lower().strip().replace(" ", "_")

        if fuel_key in self._wtt_factors:
            factor_data = self._wtt_factors[fuel_key]
            base_factor = factor_data["wtt_factor"]
            base_unit = factor_data["unit"]

            # Apply unit conversion if needed
            if unit.lower() != base_unit.lower():
                conversion = self._get_unit_conversion(unit.lower(), base_unit.lower())
                return base_factor * conversion

            return base_factor

        return None

    def _get_unit_conversion(
        self,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Get conversion factor between units.

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Conversion multiplier
        """
        # Volume conversions
        conversions = {
            ("liters", "gallon"): Decimal("0.264172"),
            ("gallon", "liters"): Decimal("3.78541"),
            ("mmbtu", "therm"): Decimal("10"),
            ("therm", "mmbtu"): Decimal("0.1"),
            ("mcf", "therm"): Decimal("10.37"),
            ("ccf", "therm"): Decimal("1.037"),
        }

        key = (from_unit.lower(), to_unit.lower())
        return conversions.get(key, Decimal("1.0"))
