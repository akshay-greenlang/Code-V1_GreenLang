"""
GL-023 HEATLOADBALANCER - Fuel Cost Calculator

This module provides zero-hallucination fuel cost calculations for heat load
balancing optimization. All calculations are deterministic with complete
provenance tracking.

Key Features:
    - Fuel cost calculation with efficiency adjustment
    - Multi-fuel cost optimization
    - Emissions cost (carbon pricing) integration
    - Total operating cost calculation
    - Spot price integration for real-time pricing

Key Formulas:
    - Fuel cost: C = fuel_flow * price_per_unit
    - Carbon cost: C_carbon = emissions * carbon_price
    - Total cost: C_total = fuel + maintenance + emissions

Standards Reference:
    - EPA Method 19 (emissions calculations)
    - 40 CFR Part 98 (GHG reporting)
    - GHG Protocol (Scope 1 emissions)

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators import (
    ...     FuelCostCalculator,
    ... )
    >>>
    >>> calc = FuelCostCalculator(fuel_type="natural_gas")
    >>> result = calc.calculate(
    ...     fuel_flow=1000.0,
    ...     fuel_price=3.50,
    ... )
    >>> print(f"Fuel cost: ${result.fuel_cost_usd:.2f}/hr")
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Reference Values
# =============================================================================

class FuelCostConstants:
    """Fuel cost calculation constants."""

    # CO2 Emission Factors (kg CO2/MMBTU) - 40 CFR Part 98 Table C-1
    CO2_EMISSION_FACTORS = {
        "natural_gas": 53.06,
        "no2_fuel_oil": 73.16,
        "no6_fuel_oil": 75.10,
        "lpg_propane": 62.87,
        "lpg_butane": 64.77,
        "coal_bituminous": 93.28,
        "coal_sub_bituminous": 97.17,
        "biomass_wood": 0.0,  # Biogenic (carbon neutral)
        "biogas": 0.0,  # Biogenic
        "hydrogen_green": 0.0,  # Zero carbon if green
        "hydrogen_grey": 89.0,  # SMR-produced
        "rng": 0.0,  # Renewable natural gas
    }

    # Higher Heating Values (HHV) in BTU/unit
    HHV = {
        "natural_gas": 1028.0,  # BTU/SCF
        "no2_fuel_oil": 140000.0,  # BTU/gal
        "no6_fuel_oil": 150000.0,  # BTU/gal
        "lpg_propane": 91500.0,  # BTU/gal
        "coal_bituminous": 25000000.0,  # BTU/ton
        "biomass_wood": 17000000.0,  # BTU/ton
        "hydrogen": 325.0,  # BTU/SCF
    }

    # Unit conversions
    MMBTU_TO_THERM = 10.0
    MMBTU_TO_GJ = 1.055


class MaintenanceCostFactors:
    """Maintenance cost factors relative to natural gas = 1.0."""

    FACTORS = {
        "natural_gas": 1.0,
        "no2_fuel_oil": 1.3,
        "no6_fuel_oil": 1.5,
        "lpg_propane": 1.1,
        "coal_bituminous": 2.0,
        "biomass_wood": 1.8,
        "biogas": 1.2,
        "hydrogen": 1.25,
    }


# =============================================================================
# DATA MODELS
# =============================================================================

class FuelCostResult(BaseModel):
    """Result from fuel cost calculation."""

    fuel_cost_usd: float = Field(..., ge=0, description="Fuel cost (USD)")
    fuel_flow: float = Field(..., ge=0, description="Fuel flow rate")
    fuel_flow_unit: str = Field(..., description="Fuel flow unit")
    fuel_price: float = Field(..., ge=0, description="Fuel price per unit")
    fuel_price_unit: str = Field(..., description="Fuel price unit")
    fuel_type: str = Field(..., description="Fuel type")

    # Energy basis
    energy_mmbtu: float = Field(..., ge=0, description="Energy in MMBTU")
    cost_per_mmbtu: float = Field(..., ge=0, description="Cost per MMBTU")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )


class EmissionsCostResult(BaseModel):
    """Result from emissions cost calculation."""

    carbon_cost_usd: float = Field(..., ge=0, description="Carbon cost (USD)")
    co2_emissions_kg: float = Field(..., ge=0, description="CO2 emissions (kg)")
    co2_emissions_tons: float = Field(..., ge=0, description="CO2 emissions (metric tons)")
    carbon_price_per_ton: float = Field(..., ge=0, description="Carbon price (USD/ton)")
    emission_factor: float = Field(..., ge=0, description="Emission factor (kg CO2/MMBTU)")
    fuel_type: str = Field(..., description="Fuel type")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")
    standard_reference: str = Field(
        default="40 CFR Part 98 Table C-1",
        description="Standard reference"
    )


class TotalOperatingCostResult(BaseModel):
    """Result from total operating cost calculation."""

    # Cost components
    fuel_cost_usd: float = Field(..., ge=0, description="Fuel cost (USD)")
    maintenance_cost_usd: float = Field(..., ge=0, description="Maintenance cost (USD)")
    emissions_cost_usd: float = Field(..., ge=0, description="Emissions/carbon cost (USD)")
    total_cost_usd: float = Field(..., ge=0, description="Total operating cost (USD)")

    # Per-unit costs
    cost_per_mmbtu: float = Field(..., ge=0, description="Total cost per MMBTU")
    cost_per_hour: float = Field(..., ge=0, description="Total cost per hour")

    # Breakdown percentages
    fuel_cost_pct: float = Field(..., ge=0, le=100, description="Fuel cost percentage")
    maintenance_cost_pct: float = Field(..., ge=0, le=100, description="Maintenance percentage")
    emissions_cost_pct: float = Field(..., ge=0, le=100, description="Emissions cost percentage")

    # Operating parameters
    heat_output_mmbtu_hr: float = Field(..., ge=0, description="Heat output (MMBTU/hr)")
    operating_hours: float = Field(..., ge=0, description="Operating hours")
    fuel_type: str = Field(..., description="Fuel type")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )


class MultiFuelOptimizationResult(BaseModel):
    """Result from multi-fuel cost optimization."""

    optimal_fuel: str = Field(..., description="Optimal fuel type")
    optimal_blend: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optimal fuel blend (if blending allowed)"
    )
    optimal_total_cost: float = Field(..., ge=0, description="Optimal total cost")
    optimal_cost_per_mmbtu: float = Field(..., ge=0, description="Optimal cost per MMBTU")

    # Rankings
    fuel_rankings: List[Tuple[str, float]] = Field(
        ...,
        description="Fuel types ranked by total cost"
    )
    cost_comparison: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Cost breakdown by fuel type"
    )

    # Savings
    savings_vs_current_usd: float = Field(
        default=0.0,
        description="Savings vs current fuel"
    )
    savings_vs_current_pct: float = Field(
        default=0.0,
        description="Savings percentage"
    )

    # Constraints satisfied
    emissions_cap_satisfied: bool = Field(
        default=True,
        description="Emissions cap constraint satisfied"
    )

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )


class SpotPriceData(BaseModel):
    """Real-time spot price data."""

    fuel_type: str = Field(..., description="Fuel type")
    price: float = Field(..., ge=0, description="Spot price")
    price_unit: str = Field(..., description="Price unit (e.g., USD/MMBTU)")
    source: str = Field(..., description="Price source")
    timestamp: datetime = Field(..., description="Price timestamp")
    valid_until: datetime = Field(..., description="Price validity end")
    region: str = Field(default="", description="Pricing region")
    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Price confidence level"
    )


# =============================================================================
# FUEL COST CALCULATOR
# =============================================================================

class FuelCostCalculator:
    """
    Calculate fuel cost from fuel flow and price.

    Formula: C = fuel_flow * price_per_unit

    ZERO-HALLUCINATION: Deterministic calculation with provenance.

    Example:
        >>> calc = FuelCostCalculator(fuel_type="natural_gas")
        >>> result = calc.calculate(
        ...     fuel_flow=1000.0,  # SCF/hr
        ...     fuel_price=3.50,   # $/MMBTU
        ... )
    """

    def __init__(
        self,
        fuel_type: str = "natural_gas",
        unit_id: str = "UNIT-001",
    ) -> None:
        """
        Initialize fuel cost calculator.

        Args:
            fuel_type: Fuel type identifier
            unit_id: Equipment unit identifier
        """
        self.fuel_type = fuel_type.lower().replace(" ", "_")
        self.unit_id = unit_id
        self._calculation_count = 0

        # Get HHV for this fuel
        self.hhv = FuelCostConstants.HHV.get(self.fuel_type, 1000.0)

        logger.info(
            f"FuelCostCalculator initialized for {unit_id} "
            f"(fuel: {fuel_type}, HHV: {self.hhv})"
        )

    def calculate(
        self,
        fuel_flow: float,
        fuel_price: float,
        fuel_flow_unit: str = "SCF/hr",
        fuel_price_unit: str = "USD/MMBTU",
    ) -> FuelCostResult:
        """
        Calculate fuel cost from flow and price.

        DETERMINISTIC: Same inputs always produce same output.

        Args:
            fuel_flow: Fuel flow rate
            fuel_price: Fuel price per unit
            fuel_flow_unit: Unit of fuel flow
            fuel_price_unit: Unit of fuel price

        Returns:
            FuelCostResult with cost and provenance
        """
        self._calculation_count += 1

        # Validate inputs
        if fuel_flow < 0:
            raise ValueError(f"Fuel flow cannot be negative: {fuel_flow}")
        if fuel_price < 0:
            raise ValueError(f"Fuel price cannot be negative: {fuel_price}")

        # Convert fuel flow to energy (MMBTU)
        energy_mmbtu = self._convert_to_mmbtu(fuel_flow, fuel_flow_unit)

        # Calculate fuel cost based on price unit
        if fuel_price_unit == "USD/MMBTU":
            fuel_cost = energy_mmbtu * fuel_price
            cost_per_mmbtu = fuel_price
        elif fuel_price_unit == "USD/therm":
            # 1 MMBTU = 10 therms
            fuel_cost = energy_mmbtu * fuel_price * 10
            cost_per_mmbtu = fuel_price * 10
        elif fuel_price_unit == "USD/SCF":
            fuel_cost = fuel_flow * fuel_price
            cost_per_mmbtu = fuel_cost / energy_mmbtu if energy_mmbtu > 0 else 0
        elif fuel_price_unit == "USD/gal":
            # For liquid fuels, need to convert via HHV
            gal_per_mmbtu = 1_000_000 / self.hhv
            fuel_cost = energy_mmbtu * gal_per_mmbtu * fuel_price
            cost_per_mmbtu = gal_per_mmbtu * fuel_price
        else:
            # Default: assume price is per MMBTU
            fuel_cost = energy_mmbtu * fuel_price
            cost_per_mmbtu = fuel_price

        # Calculate provenance hash
        calculation_hash = self._calculate_hash(
            fuel_flow=fuel_flow,
            fuel_price=fuel_price,
            fuel_cost=fuel_cost,
        )

        return FuelCostResult(
            fuel_cost_usd=round(fuel_cost, 2),
            fuel_flow=fuel_flow,
            fuel_flow_unit=fuel_flow_unit,
            fuel_price=fuel_price,
            fuel_price_unit=fuel_price_unit,
            fuel_type=self.fuel_type,
            energy_mmbtu=round(energy_mmbtu, 4),
            cost_per_mmbtu=round(cost_per_mmbtu, 4),
            calculation_hash=calculation_hash,
        )

    def calculate_from_heat_output(
        self,
        heat_output_mmbtu_hr: float,
        efficiency_pct: float,
        fuel_price: float,
        fuel_price_unit: str = "USD/MMBTU",
    ) -> FuelCostResult:
        """
        Calculate fuel cost from heat output and efficiency.

        Args:
            heat_output_mmbtu_hr: Heat output (MMBTU/hr)
            efficiency_pct: Equipment efficiency (%)
            fuel_price: Fuel price
            fuel_price_unit: Fuel price unit

        Returns:
            FuelCostResult with cost and provenance
        """
        if efficiency_pct <= 0 or efficiency_pct > 100:
            raise ValueError(f"Invalid efficiency: {efficiency_pct}%")

        # Calculate required fuel energy input
        fuel_input_mmbtu = heat_output_mmbtu_hr / (efficiency_pct / 100)

        # Convert to fuel flow
        fuel_flow = self._convert_from_mmbtu(fuel_input_mmbtu)

        return self.calculate(
            fuel_flow=fuel_flow,
            fuel_price=fuel_price,
            fuel_flow_unit=self._get_default_flow_unit(),
            fuel_price_unit=fuel_price_unit,
        )

    def _convert_to_mmbtu(self, fuel_flow: float, unit: str) -> float:
        """Convert fuel flow to MMBTU."""
        if unit == "SCF/hr" or unit == "SCF":
            # Natural gas: BTU/SCF from HHV
            return fuel_flow * self.hhv / 1_000_000
        elif unit == "MMBTU/hr" or unit == "MMBTU":
            return fuel_flow
        elif unit == "gal/hr" or unit == "gal":
            # Liquid fuel: BTU/gal from HHV
            return fuel_flow * self.hhv / 1_000_000
        elif unit == "lb/hr" or unit == "lb":
            # Solid fuel: need density conversion
            # Approximate for coal: ~12,500 BTU/lb
            return fuel_flow * 12500 / 1_000_000
        elif unit == "ton/hr" or unit == "ton":
            return fuel_flow * self.hhv / 1_000_000
        else:
            logger.warning(f"Unknown fuel flow unit: {unit}, assuming MMBTU/hr")
            return fuel_flow

    def _convert_from_mmbtu(self, energy_mmbtu: float) -> float:
        """Convert MMBTU to default fuel flow unit."""
        return energy_mmbtu * 1_000_000 / self.hhv

    def _get_default_flow_unit(self) -> str:
        """Get default flow unit for fuel type."""
        if self.fuel_type in ["natural_gas", "biogas", "hydrogen"]:
            return "SCF/hr"
        elif self.fuel_type in ["no2_fuel_oil", "no6_fuel_oil", "lpg_propane"]:
            return "gal/hr"
        elif self.fuel_type in ["coal_bituminous", "biomass_wood"]:
            return "ton/hr"
        else:
            return "MMBTU/hr"

    def _calculate_hash(
        self,
        fuel_flow: float,
        fuel_price: float,
        fuel_cost: float,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "calculator": "FuelCostCalculator",
            "unit_id": self.unit_id,
            "fuel_type": self.fuel_type,
            "fuel_flow": fuel_flow,
            "fuel_price": fuel_price,
            "fuel_cost": fuel_cost,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# EMISSIONS COST CALCULATOR
# =============================================================================

class EmissionsCostCalculator:
    """
    Calculate emissions (carbon) cost from energy consumption.

    Formula: C_carbon = emissions_kg * (carbon_price / 1000)

    Based on 40 CFR Part 98 emission factors.

    ZERO-HALLUCINATION: Deterministic calculation with provenance.

    Example:
        >>> calc = EmissionsCostCalculator(carbon_price_per_ton=50.0)
        >>> result = calc.calculate(
        ...     energy_mmbtu=100.0,
        ...     fuel_type="natural_gas",
        ... )
    """

    def __init__(
        self,
        carbon_price_per_ton: float = 50.0,
        unit_id: str = "UNIT-001",
    ) -> None:
        """
        Initialize emissions cost calculator.

        Args:
            carbon_price_per_ton: Carbon price (USD/metric ton CO2)
            unit_id: Equipment unit identifier
        """
        self.carbon_price_per_ton = carbon_price_per_ton
        self.unit_id = unit_id
        self._calculation_count = 0

        logger.info(
            f"EmissionsCostCalculator initialized for {unit_id} "
            f"(carbon price: ${carbon_price_per_ton}/ton)"
        )

    def calculate(
        self,
        energy_mmbtu: float,
        fuel_type: str = "natural_gas",
        custom_emission_factor: Optional[float] = None,
    ) -> EmissionsCostResult:
        """
        Calculate emissions cost from energy consumption.

        DETERMINISTIC: Same inputs always produce same output.

        Args:
            energy_mmbtu: Energy consumption (MMBTU)
            fuel_type: Fuel type identifier
            custom_emission_factor: Override emission factor (kg CO2/MMBTU)

        Returns:
            EmissionsCostResult with emissions cost and provenance
        """
        self._calculation_count += 1

        # Validate inputs
        if energy_mmbtu < 0:
            raise ValueError(f"Energy cannot be negative: {energy_mmbtu}")

        # Get emission factor
        fuel_key = fuel_type.lower().replace(" ", "_")
        if custom_emission_factor is not None:
            emission_factor = custom_emission_factor
        else:
            emission_factor = FuelCostConstants.CO2_EMISSION_FACTORS.get(
                fuel_key, 53.06  # Default to natural gas
            )

        # Calculate emissions
        # kg CO2 = MMBTU * emission_factor (kg CO2/MMBTU)
        co2_emissions_kg = energy_mmbtu * emission_factor
        co2_emissions_tons = co2_emissions_kg / 1000.0

        # Calculate carbon cost
        # Cost = tons * price_per_ton
        carbon_cost = co2_emissions_tons * self.carbon_price_per_ton

        # Calculate provenance hash
        calculation_hash = self._calculate_hash(
            energy_mmbtu=energy_mmbtu,
            fuel_type=fuel_type,
            co2_emissions_kg=co2_emissions_kg,
            carbon_cost=carbon_cost,
        )

        return EmissionsCostResult(
            carbon_cost_usd=round(carbon_cost, 2),
            co2_emissions_kg=round(co2_emissions_kg, 2),
            co2_emissions_tons=round(co2_emissions_tons, 4),
            carbon_price_per_ton=self.carbon_price_per_ton,
            emission_factor=emission_factor,
            fuel_type=fuel_type,
            calculation_hash=calculation_hash,
        )

    def calculate_from_fuel_flow(
        self,
        fuel_flow: float,
        fuel_type: str,
        fuel_flow_unit: str = "SCF/hr",
        hours: float = 1.0,
    ) -> EmissionsCostResult:
        """
        Calculate emissions cost from fuel flow.

        Args:
            fuel_flow: Fuel flow rate
            fuel_type: Fuel type identifier
            fuel_flow_unit: Unit of fuel flow
            hours: Operating hours

        Returns:
            EmissionsCostResult with emissions cost
        """
        # Convert to energy
        fuel_key = fuel_type.lower().replace(" ", "_")
        hhv = FuelCostConstants.HHV.get(fuel_key, 1000.0)

        if fuel_flow_unit in ["SCF/hr", "SCF"]:
            energy_mmbtu = fuel_flow * hours * hhv / 1_000_000
        elif fuel_flow_unit in ["gal/hr", "gal"]:
            energy_mmbtu = fuel_flow * hours * hhv / 1_000_000
        elif fuel_flow_unit in ["MMBTU/hr", "MMBTU"]:
            energy_mmbtu = fuel_flow * hours
        else:
            energy_mmbtu = fuel_flow * hours

        return self.calculate(
            energy_mmbtu=energy_mmbtu,
            fuel_type=fuel_type,
        )

    def set_carbon_price(self, price_per_ton: float) -> None:
        """Update carbon price."""
        self.carbon_price_per_ton = price_per_ton
        logger.info(f"Carbon price updated to ${price_per_ton}/ton")

    def _calculate_hash(
        self,
        energy_mmbtu: float,
        fuel_type: str,
        co2_emissions_kg: float,
        carbon_cost: float,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "calculator": "EmissionsCostCalculator",
            "unit_id": self.unit_id,
            "energy_mmbtu": energy_mmbtu,
            "fuel_type": fuel_type,
            "co2_emissions_kg": co2_emissions_kg,
            "carbon_cost": carbon_cost,
            "carbon_price_per_ton": self.carbon_price_per_ton,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# TOTAL OPERATING COST CALCULATOR
# =============================================================================

class TotalOperatingCostCalculator:
    """
    Calculate total operating cost including fuel, maintenance, and emissions.

    Formula: C_total = fuel + maintenance + emissions

    ZERO-HALLUCINATION: Deterministic calculation with provenance.

    Example:
        >>> calc = TotalOperatingCostCalculator(carbon_price_per_ton=50.0)
        >>> result = calc.calculate(
        ...     heat_output_mmbtu_hr=100.0,
        ...     efficiency_pct=85.0,
        ...     fuel_type="natural_gas",
        ...     fuel_price=3.50,
        ...     operating_hours=8000.0,
        ... )
    """

    def __init__(
        self,
        carbon_price_per_ton: float = 50.0,
        base_maintenance_cost_per_year: float = 50000.0,
        unit_id: str = "UNIT-001",
    ) -> None:
        """
        Initialize total operating cost calculator.

        Args:
            carbon_price_per_ton: Carbon price (USD/metric ton CO2)
            base_maintenance_cost_per_year: Base annual maintenance cost
            unit_id: Equipment unit identifier
        """
        self.carbon_price_per_ton = carbon_price_per_ton
        self.base_maintenance_cost = base_maintenance_cost_per_year
        self.unit_id = unit_id

        # Initialize sub-calculators
        self._emissions_calc = EmissionsCostCalculator(
            carbon_price_per_ton=carbon_price_per_ton,
            unit_id=unit_id,
        )

        self._calculation_count = 0

        logger.info(
            f"TotalOperatingCostCalculator initialized for {unit_id}"
        )

    def calculate(
        self,
        heat_output_mmbtu_hr: float,
        efficiency_pct: float,
        fuel_type: str,
        fuel_price: float,
        operating_hours: float = 8000.0,
        fuel_price_unit: str = "USD/MMBTU",
        custom_maintenance_factor: Optional[float] = None,
    ) -> TotalOperatingCostResult:
        """
        Calculate total operating cost.

        DETERMINISTIC: Same inputs always produce same output.

        Args:
            heat_output_mmbtu_hr: Heat output (MMBTU/hr)
            efficiency_pct: Equipment efficiency (%)
            fuel_type: Fuel type identifier
            fuel_price: Fuel price
            operating_hours: Annual operating hours
            fuel_price_unit: Fuel price unit
            custom_maintenance_factor: Override maintenance factor

        Returns:
            TotalOperatingCostResult with cost breakdown
        """
        self._calculation_count += 1

        # Validate inputs
        if heat_output_mmbtu_hr < 0:
            raise ValueError("Heat output cannot be negative")
        if efficiency_pct <= 0 or efficiency_pct > 100:
            raise ValueError(f"Invalid efficiency: {efficiency_pct}%")
        if operating_hours < 0:
            raise ValueError("Operating hours cannot be negative")

        fuel_key = fuel_type.lower().replace(" ", "_")

        # Calculate fuel input (accounting for efficiency)
        fuel_input_mmbtu_hr = heat_output_mmbtu_hr / (efficiency_pct / 100)
        total_fuel_mmbtu = fuel_input_mmbtu_hr * operating_hours

        # Calculate fuel cost
        if fuel_price_unit == "USD/MMBTU":
            fuel_cost = total_fuel_mmbtu * fuel_price
        elif fuel_price_unit == "USD/therm":
            fuel_cost = total_fuel_mmbtu * fuel_price * 10
        else:
            fuel_cost = total_fuel_mmbtu * fuel_price

        # Calculate emissions cost
        emissions_result = self._emissions_calc.calculate(
            energy_mmbtu=total_fuel_mmbtu,
            fuel_type=fuel_type,
        )
        emissions_cost = emissions_result.carbon_cost_usd

        # Calculate maintenance cost
        if custom_maintenance_factor is not None:
            maintenance_factor = custom_maintenance_factor
        else:
            maintenance_factor = MaintenanceCostFactors.FACTORS.get(fuel_key, 1.0)

        maintenance_cost = self.base_maintenance_cost * maintenance_factor

        # Total cost
        total_cost = fuel_cost + maintenance_cost + emissions_cost

        # Per-unit costs
        total_energy_mmbtu = heat_output_mmbtu_hr * operating_hours
        cost_per_mmbtu = total_cost / total_energy_mmbtu if total_energy_mmbtu > 0 else 0
        cost_per_hour = total_cost / operating_hours if operating_hours > 0 else 0

        # Breakdown percentages
        if total_cost > 0:
            fuel_cost_pct = (fuel_cost / total_cost) * 100
            maintenance_cost_pct = (maintenance_cost / total_cost) * 100
            emissions_cost_pct = (emissions_cost / total_cost) * 100
        else:
            fuel_cost_pct = 0.0
            maintenance_cost_pct = 0.0
            emissions_cost_pct = 0.0

        # Calculate provenance hash
        calculation_hash = self._calculate_hash(
            heat_output_mmbtu_hr=heat_output_mmbtu_hr,
            fuel_type=fuel_type,
            total_cost=total_cost,
        )

        return TotalOperatingCostResult(
            fuel_cost_usd=round(fuel_cost, 2),
            maintenance_cost_usd=round(maintenance_cost, 2),
            emissions_cost_usd=round(emissions_cost, 2),
            total_cost_usd=round(total_cost, 2),
            cost_per_mmbtu=round(cost_per_mmbtu, 4),
            cost_per_hour=round(cost_per_hour, 2),
            fuel_cost_pct=round(fuel_cost_pct, 1),
            maintenance_cost_pct=round(maintenance_cost_pct, 1),
            emissions_cost_pct=round(emissions_cost_pct, 1),
            heat_output_mmbtu_hr=heat_output_mmbtu_hr,
            operating_hours=operating_hours,
            fuel_type=fuel_type,
            calculation_hash=calculation_hash,
        )

    def _calculate_hash(
        self,
        heat_output_mmbtu_hr: float,
        fuel_type: str,
        total_cost: float,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "calculator": "TotalOperatingCostCalculator",
            "unit_id": self.unit_id,
            "heat_output_mmbtu_hr": heat_output_mmbtu_hr,
            "fuel_type": fuel_type,
            "total_cost": total_cost,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# MULTI-FUEL COST OPTIMIZER
# =============================================================================

class MultiFuelCostOptimizer:
    """
    Optimize fuel selection across multiple fuel options.

    Finds the fuel (or blend) that minimizes total operating cost
    while satisfying emissions constraints.

    ZERO-HALLUCINATION: Deterministic optimization with provenance.

    Example:
        >>> optimizer = MultiFuelCostOptimizer(carbon_price_per_ton=50.0)
        >>> fuel_prices = {
        ...     "natural_gas": 3.50,
        ...     "no2_fuel_oil": 12.00,
        ...     "biomass_wood": 5.00,
        ... }
        >>> result = optimizer.optimize(
        ...     heat_demand_mmbtu_hr=100.0,
        ...     fuel_prices=fuel_prices,
        ...     efficiencies={"natural_gas": 85.0, "no2_fuel_oil": 82.0},
        ... )
    """

    def __init__(
        self,
        carbon_price_per_ton: float = 50.0,
        base_maintenance_cost_per_year: float = 50000.0,
        unit_id: str = "UNIT-001",
    ) -> None:
        """
        Initialize multi-fuel cost optimizer.

        Args:
            carbon_price_per_ton: Carbon price (USD/metric ton CO2)
            base_maintenance_cost_per_year: Base annual maintenance cost
            unit_id: Equipment unit identifier
        """
        self.carbon_price_per_ton = carbon_price_per_ton
        self.base_maintenance_cost = base_maintenance_cost_per_year
        self.unit_id = unit_id

        self._cost_calc = TotalOperatingCostCalculator(
            carbon_price_per_ton=carbon_price_per_ton,
            base_maintenance_cost_per_year=base_maintenance_cost_per_year,
            unit_id=unit_id,
        )

        self._optimization_count = 0

        logger.info(f"MultiFuelCostOptimizer initialized for {unit_id}")

    def optimize(
        self,
        heat_demand_mmbtu_hr: float,
        fuel_prices: Dict[str, float],
        efficiencies: Optional[Dict[str, float]] = None,
        operating_hours: float = 8000.0,
        current_fuel: Optional[str] = None,
        emissions_cap_tons: Optional[float] = None,
        allow_blending: bool = False,
    ) -> MultiFuelOptimizationResult:
        """
        Optimize fuel selection for minimum total cost.

        DETERMINISTIC: Same inputs always produce same result.

        Args:
            heat_demand_mmbtu_hr: Heat demand (MMBTU/hr)
            fuel_prices: Dictionary of fuel type to price (USD/MMBTU)
            efficiencies: Dictionary of fuel type to efficiency (%)
            operating_hours: Annual operating hours
            current_fuel: Current fuel type (for savings calculation)
            emissions_cap_tons: Optional emissions cap (metric tons CO2/year)
            allow_blending: Allow fuel blending optimization

        Returns:
            MultiFuelOptimizationResult with optimal selection
        """
        self._optimization_count += 1

        if not fuel_prices:
            raise ValueError("At least one fuel type required")

        # Default efficiencies if not provided
        if efficiencies is None:
            efficiencies = {fuel: 82.0 for fuel in fuel_prices}

        # Calculate costs for each fuel
        cost_comparison: Dict[str, Dict[str, float]] = {}
        emissions_by_fuel: Dict[str, float] = {}

        for fuel_type, price in fuel_prices.items():
            efficiency = efficiencies.get(fuel_type, 82.0)

            result = self._cost_calc.calculate(
                heat_output_mmbtu_hr=heat_demand_mmbtu_hr,
                efficiency_pct=efficiency,
                fuel_type=fuel_type,
                fuel_price=price,
                operating_hours=operating_hours,
            )

            cost_comparison[fuel_type] = {
                "fuel_cost": result.fuel_cost_usd,
                "maintenance_cost": result.maintenance_cost_usd,
                "emissions_cost": result.emissions_cost_usd,
                "total_cost": result.total_cost_usd,
                "cost_per_mmbtu": result.cost_per_mmbtu,
            }

            # Calculate emissions for constraint checking
            fuel_input = heat_demand_mmbtu_hr / (efficiency / 100) * operating_hours
            emission_factor = FuelCostConstants.CO2_EMISSION_FACTORS.get(
                fuel_type.lower().replace(" ", "_"), 53.0
            )
            emissions_by_fuel[fuel_type] = fuel_input * emission_factor / 1000

        # Rank fuels by total cost
        fuel_rankings = sorted(
            [(fuel, data["total_cost"]) for fuel, data in cost_comparison.items()],
            key=lambda x: x[1]
        )

        # Check emissions constraint
        optimal_fuel = fuel_rankings[0][0]
        emissions_cap_satisfied = True

        if emissions_cap_tons is not None:
            # Find cheapest fuel that meets emissions cap
            for fuel, cost in fuel_rankings:
                if emissions_by_fuel[fuel] <= emissions_cap_tons:
                    optimal_fuel = fuel
                    break
            else:
                # No fuel meets constraint - use lowest emission fuel
                optimal_fuel = min(emissions_by_fuel.items(), key=lambda x: x[1])[0]
                emissions_cap_satisfied = False

        optimal_total_cost = cost_comparison[optimal_fuel]["total_cost"]
        optimal_cost_per_mmbtu = cost_comparison[optimal_fuel]["cost_per_mmbtu"]

        # Calculate savings vs current
        savings_usd = 0.0
        savings_pct = 0.0
        if current_fuel and current_fuel in cost_comparison:
            current_cost = cost_comparison[current_fuel]["total_cost"]
            savings_usd = current_cost - optimal_total_cost
            if current_cost > 0:
                savings_pct = (savings_usd / current_cost) * 100

        # Calculate provenance hash
        calculation_hash = self._calculate_hash(
            heat_demand_mmbtu_hr=heat_demand_mmbtu_hr,
            fuel_prices=fuel_prices,
            optimal_fuel=optimal_fuel,
            optimal_total_cost=optimal_total_cost,
        )

        return MultiFuelOptimizationResult(
            optimal_fuel=optimal_fuel,
            optimal_blend=None,  # Blending not implemented in this version
            optimal_total_cost=round(optimal_total_cost, 2),
            optimal_cost_per_mmbtu=round(optimal_cost_per_mmbtu, 4),
            fuel_rankings=fuel_rankings,
            cost_comparison=cost_comparison,
            savings_vs_current_usd=round(savings_usd, 2),
            savings_vs_current_pct=round(savings_pct, 2),
            emissions_cap_satisfied=emissions_cap_satisfied,
            calculation_hash=calculation_hash,
        )

    def _calculate_hash(
        self,
        heat_demand_mmbtu_hr: float,
        fuel_prices: Dict[str, float],
        optimal_fuel: str,
        optimal_total_cost: float,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "calculator": "MultiFuelCostOptimizer",
            "unit_id": self.unit_id,
            "heat_demand_mmbtu_hr": heat_demand_mmbtu_hr,
            "fuel_prices_hash": hashlib.sha256(
                json.dumps(fuel_prices, sort_keys=True).encode()
            ).hexdigest()[:16],
            "optimal_fuel": optimal_fuel,
            "optimal_total_cost": optimal_total_cost,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def optimization_count(self) -> int:
        """Get total optimization count."""
        return self._optimization_count


# =============================================================================
# SPOT PRICE INTEGRATION
# =============================================================================

class SpotPriceIntegration:
    """
    Handle real-time fuel spot price integration.

    Provides caching, fallback, and price validation for
    real-time fuel pricing.

    Example:
        >>> spot = SpotPriceIntegration()
        >>> spot.set_price("natural_gas", 3.50, source="Henry Hub")
        >>> price = spot.get_price("natural_gas")
    """

    def __init__(
        self,
        cache_ttl_minutes: int = 15,
        unit_id: str = "UNIT-001",
    ) -> None:
        """
        Initialize spot price integration.

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
            unit_id: Unit identifier
        """
        self.cache_ttl_minutes = cache_ttl_minutes
        self.unit_id = unit_id

        self._prices: Dict[str, SpotPriceData] = {}
        self._fallback_prices: Dict[str, float] = {
            "natural_gas": 3.00,
            "no2_fuel_oil": 15.00,
            "no6_fuel_oil": 12.00,
            "lpg_propane": 8.00,
            "coal_bituminous": 2.50,
            "biomass_wood": 4.00,
            "hydrogen": 15.00,
        }

        logger.info(
            f"SpotPriceIntegration initialized (TTL: {cache_ttl_minutes}min)"
        )

    def set_price(
        self,
        fuel_type: str,
        price: float,
        source: str = "manual",
        price_unit: str = "USD/MMBTU",
        region: str = "",
        valid_minutes: Optional[int] = None,
    ) -> SpotPriceData:
        """
        Set spot price for a fuel type.

        Args:
            fuel_type: Fuel type identifier
            price: Spot price
            source: Price source
            price_unit: Price unit
            region: Pricing region
            valid_minutes: Price validity period

        Returns:
            SpotPriceData with price and metadata
        """
        if price < 0:
            raise ValueError(f"Price cannot be negative: {price}")

        now = datetime.now(timezone.utc)
        valid_period = valid_minutes or self.cache_ttl_minutes

        price_data = SpotPriceData(
            fuel_type=fuel_type,
            price=price,
            price_unit=price_unit,
            source=source,
            timestamp=now,
            valid_until=now + timedelta(minutes=valid_period),
            region=region,
        )

        self._prices[fuel_type.lower()] = price_data

        logger.debug(
            f"Spot price set: {fuel_type} = ${price:.4f}/{price_unit} "
            f"(source: {source})"
        )

        return price_data

    def get_price(
        self,
        fuel_type: str,
        use_fallback: bool = True,
    ) -> Optional[SpotPriceData]:
        """
        Get current spot price for a fuel type.

        Args:
            fuel_type: Fuel type identifier
            use_fallback: Use fallback if no valid price

        Returns:
            SpotPriceData or None if not available
        """
        fuel_key = fuel_type.lower()
        price_data = self._prices.get(fuel_key)

        now = datetime.now(timezone.utc)

        # Check if price is valid
        if price_data and now <= price_data.valid_until:
            return price_data

        # Price expired or not found
        if use_fallback and fuel_key in self._fallback_prices:
            return SpotPriceData(
                fuel_type=fuel_type,
                price=self._fallback_prices[fuel_key],
                price_unit="USD/MMBTU",
                source="fallback",
                timestamp=now,
                valid_until=now + timedelta(hours=24),
                confidence=0.7,
            )

        return None

    def get_all_prices(
        self,
        fuel_types: Optional[List[str]] = None,
    ) -> Dict[str, SpotPriceData]:
        """
        Get prices for multiple fuel types.

        Args:
            fuel_types: List of fuel types (all if None)

        Returns:
            Dictionary of fuel type to SpotPriceData
        """
        if fuel_types is None:
            fuel_types = list(self._prices.keys()) + list(self._fallback_prices.keys())

        result = {}
        for fuel_type in fuel_types:
            price_data = self.get_price(fuel_type)
            if price_data:
                result[fuel_type] = price_data

        return result

    def is_price_valid(self, fuel_type: str) -> bool:
        """Check if price is valid (not expired)."""
        fuel_key = fuel_type.lower()
        price_data = self._prices.get(fuel_key)

        if not price_data:
            return False

        return datetime.now(timezone.utc) <= price_data.valid_until

    def set_fallback_price(self, fuel_type: str, price: float) -> None:
        """Set fallback price for a fuel type."""
        self._fallback_prices[fuel_type.lower()] = price

    def clear_prices(self) -> None:
        """Clear all cached prices."""
        self._prices.clear()
        logger.info("Spot prices cleared")

    def get_price_for_calculation(
        self,
        fuel_type: str,
    ) -> float:
        """
        Get price value for calculation.

        Returns numeric price suitable for calculations.

        Args:
            fuel_type: Fuel type identifier

        Returns:
            Price in USD/MMBTU
        """
        price_data = self.get_price(fuel_type, use_fallback=True)

        if price_data is None:
            raise ValueError(f"No price available for {fuel_type}")

        return price_data.price
