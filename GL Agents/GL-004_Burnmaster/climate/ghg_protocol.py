"""
GL-004 BURNMASTER - GHG Protocol Calculator

GHG Protocol Corporate Standard Scope 1 direct emissions calculations.
Implements stationary combustion methodology for industrial facilities.

References:
    - GHG Protocol Corporate Standard (Revised Edition)
    - GHG Protocol Calculation Tools
    - IPCC Guidelines for National Greenhouse Gas Inventories

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FuelCategory(str, Enum):
    """GHG Protocol fuel categories for stationary combustion."""
    GASEOUS = "gaseous"
    LIQUID = "liquid"
    SOLID = "solid"
    BIOMASS = "biomass"


class CalculationApproach(str, Enum):
    """GHG Protocol calculation approaches."""
    FUEL_BASED = "fuel_based"           # Default - based on fuel consumption
    DIRECT_MEASUREMENT = "direct_measurement"  # CEMS
    MASS_BALANCE = "mass_balance"       # For certain processes


class EmissionFactor(BaseModel):
    """Emission factor with source and uncertainty."""
    factor_id: str = Field(default_factory=lambda: str(uuid4()))
    fuel_type: str = Field(..., description="Fuel type identifier")
    co2_kg_per_unit: Decimal = Field(..., ge=0, description="CO2 factor")
    ch4_kg_per_unit: Decimal = Field(default=Decimal("0"), ge=0)
    n2o_kg_per_unit: Decimal = Field(default=Decimal("0"), ge=0)
    unit: str = Field(default="kg", description="Unit of fuel (kg, m3, mmbtu)")
    source: str = Field(default="IPCC 2006", description="Source of factor")
    uncertainty_percent: float = Field(default=5.0, ge=0, le=100)
    year: int = Field(default=2024, description="Year factor is valid for")


class Scope1Emissions(BaseModel):
    """Scope 1 direct emissions result."""
    calculation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    facility_id: str = Field(..., description="Facility identifier")
    reporting_year: int = Field(..., description="Reporting year")

    # Total emissions (tonnes CO2e)
    total_co2_tonnes: Decimal = Field(..., ge=0)
    total_ch4_tonnes: Decimal = Field(..., ge=0)
    total_n2o_tonnes: Decimal = Field(..., ge=0)
    total_co2e_tonnes: Decimal = Field(..., ge=0)

    # By source category
    stationary_combustion_co2e: Decimal = Field(default=Decimal("0"), ge=0)
    mobile_combustion_co2e: Decimal = Field(default=Decimal("0"), ge=0)
    process_emissions_co2e: Decimal = Field(default=Decimal("0"), ge=0)
    fugitive_emissions_co2e: Decimal = Field(default=Decimal("0"), ge=0)

    # By fuel category
    emissions_by_fuel_category: Dict[str, Decimal] = Field(default_factory=dict)

    # Biogenic emissions (reported separately per GHG Protocol)
    biogenic_co2_tonnes: Decimal = Field(default=Decimal("0"), ge=0)

    # Calculation details
    approach: CalculationApproach = Field(default=CalculationApproach.FUEL_BASED)
    gwp_source: str = Field(default="IPCC AR5", description="GWP source")
    base_year: Optional[int] = Field(None, description="Base year for tracking")
    change_from_base_year_percent: Optional[float] = Field(None)

    # Quality and uncertainty
    overall_uncertainty_percent: float = Field(default=5.0, ge=0)
    data_quality_indicator: str = Field(default="tier_1")

    provenance_hash: str = Field(default="", description="SHA-256 hash")


# GWP values from IPCC AR5 (100-year horizon)
GWP_AR5 = {
    "co2": 1,
    "ch4": 28,
    "n2o": 265,
    "sf6": 23500,
    "hfc_134a": 1300,
}

# Default emission factors (kg CO2/unit) from IPCC 2006, EPA, and UK DEFRA
# These are representative values - actual implementations should use
# jurisdiction-specific factors
DEFAULT_EMISSION_FACTORS: Dict[str, EmissionFactor] = {
    "natural_gas": EmissionFactor(
        fuel_type="natural_gas",
        co2_kg_per_unit=Decimal("2.02"),  # kg CO2 per Nm3
        ch4_kg_per_unit=Decimal("0.000039"),
        n2o_kg_per_unit=Decimal("0.000039"),
        unit="nm3",
        source="IPCC 2006"
    ),
    "natural_gas_kg": EmissionFactor(
        fuel_type="natural_gas_kg",
        co2_kg_per_unit=Decimal("2.75"),  # kg CO2 per kg
        ch4_kg_per_unit=Decimal("0.001"),
        n2o_kg_per_unit=Decimal("0.0001"),
        unit="kg",
        source="IPCC 2006"
    ),
    "diesel": EmissionFactor(
        fuel_type="diesel",
        co2_kg_per_unit=Decimal("2.68"),  # kg CO2 per liter
        ch4_kg_per_unit=Decimal("0.0001"),
        n2o_kg_per_unit=Decimal("0.00022"),
        unit="liter",
        source="IPCC 2006"
    ),
    "fuel_oil_no2": EmissionFactor(
        fuel_type="fuel_oil_no2",
        co2_kg_per_unit=Decimal("2.75"),  # kg CO2 per liter
        ch4_kg_per_unit=Decimal("0.0001"),
        n2o_kg_per_unit=Decimal("0.00022"),
        unit="liter",
        source="EPA"
    ),
    "fuel_oil_no6": EmissionFactor(
        fuel_type="fuel_oil_no6",
        co2_kg_per_unit=Decimal("3.11"),  # kg CO2 per liter
        ch4_kg_per_unit=Decimal("0.0003"),
        n2o_kg_per_unit=Decimal("0.0006"),
        unit="liter",
        source="EPA"
    ),
    "propane": EmissionFactor(
        fuel_type="propane",
        co2_kg_per_unit=Decimal("1.51"),  # kg CO2 per liter
        ch4_kg_per_unit=Decimal("0.0001"),
        n2o_kg_per_unit=Decimal("0.0001"),
        unit="liter",
        source="IPCC 2006"
    ),
    "coal_bituminous": EmissionFactor(
        fuel_type="coal_bituminous",
        co2_kg_per_unit=Decimal("2.42"),  # kg CO2 per kg
        ch4_kg_per_unit=Decimal("0.011"),
        n2o_kg_per_unit=Decimal("0.0016"),
        unit="kg",
        source="IPCC 2006"
    ),
    "biomass_wood": EmissionFactor(
        fuel_type="biomass_wood",
        co2_kg_per_unit=Decimal("1.83"),  # Biogenic - reported separately
        ch4_kg_per_unit=Decimal("0.0072"),
        n2o_kg_per_unit=Decimal("0.0036"),
        unit="kg",
        source="IPCC 2006"
    ),
}


class GHGProtocolCalculator:
    """
    GHG Protocol Scope 1 emissions calculator for stationary combustion.

    Implements the GHG Protocol Corporate Standard methodology for
    calculating direct emissions from stationary combustion sources.

    Example:
        >>> calc = GHGProtocolCalculator()
        >>> result = calc.calculate_scope1(
        ...     facility_id="FAC-001",
        ...     reporting_year=2024,
        ...     fuel_data=[
        ...         {"fuel_type": "natural_gas_kg", "quantity": Decimal("100000")}
        ...     ]
        ... )
        >>> print(f"Total Scope 1: {result.total_co2e_tonnes} tonnes CO2e")
    """

    def __init__(self, precision: int = 4):
        """
        Initialize GHG Protocol calculator.

        Args:
            precision: Decimal places for calculations
        """
        self.precision = precision
        self._quantize_str = "0." + "0" * precision
        self._custom_factors: Dict[str, EmissionFactor] = {}
        self._base_year_emissions: Dict[str, Decimal] = {}
        logger.info("GHGProtocolCalculator initialized")

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def set_emission_factor(self, fuel_type: str, factor: EmissionFactor) -> None:
        """Set custom emission factor for a fuel type."""
        self._custom_factors[fuel_type] = factor
        logger.info(f"Custom emission factor set for {fuel_type}")

    def get_emission_factor(self, fuel_type: str) -> Optional[EmissionFactor]:
        """Get emission factor for a fuel type."""
        if fuel_type in self._custom_factors:
            return self._custom_factors[fuel_type]
        return DEFAULT_EMISSION_FACTORS.get(fuel_type)

    def set_base_year_emissions(
        self,
        facility_id: str,
        base_year: int,
        co2e_tonnes: Decimal
    ) -> None:
        """Set base year emissions for tracking changes over time."""
        key = f"{facility_id}_{base_year}"
        self._base_year_emissions[key] = co2e_tonnes
        logger.info(f"Base year emissions set: {facility_id} {base_year} = {co2e_tonnes} tCO2e")

    def calculate_fuel_emissions(
        self,
        fuel_type: str,
        quantity: Decimal,
        quantity_unit: Optional[str] = None,
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal, bool]:
        """
        Calculate emissions from a single fuel source.

        Args:
            fuel_type: Fuel type identifier
            quantity: Fuel quantity consumed
            quantity_unit: Unit override (if different from factor default)

        Returns:
            Tuple of (co2_kg, ch4_kg, n2o_kg, co2e_kg, is_biogenic)
        """
        factor = self.get_emission_factor(fuel_type)
        if not factor:
            logger.warning(f"No emission factor for {fuel_type}, using zero")
            return Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), False

        # Calculate emissions (DETERMINISTIC)
        co2_kg = self._quantize(quantity * factor.co2_kg_per_unit)
        ch4_kg = self._quantize(quantity * factor.ch4_kg_per_unit)
        n2o_kg = self._quantize(quantity * factor.n2o_kg_per_unit)

        # Calculate CO2e using GWP
        co2e_kg = self._quantize(
            co2_kg * GWP_AR5["co2"] +
            ch4_kg * GWP_AR5["ch4"] +
            n2o_kg * GWP_AR5["n2o"]
        )

        # Check if biogenic (biomass fuels)
        is_biogenic = "biomass" in fuel_type.lower() or "bio" in fuel_type.lower()

        return co2_kg, ch4_kg, n2o_kg, co2e_kg, is_biogenic

    def calculate_scope1(
        self,
        facility_id: str,
        reporting_year: int,
        fuel_data: List[Dict[str, Any]],
        mobile_emissions_co2e_kg: Decimal = Decimal("0"),
        process_emissions_co2e_kg: Decimal = Decimal("0"),
        fugitive_emissions_co2e_kg: Decimal = Decimal("0"),
        base_year: Optional[int] = None,
    ) -> Scope1Emissions:
        """
        Calculate total Scope 1 emissions for a facility.

        Args:
            facility_id: Facility identifier
            reporting_year: Year being reported
            fuel_data: List of fuel consumption records, each with:
                - fuel_type: str
                - quantity: Decimal
                - unit: str (optional)
            mobile_emissions_co2e_kg: Mobile combustion emissions (if any)
            process_emissions_co2e_kg: Process emissions (if any)
            fugitive_emissions_co2e_kg: Fugitive emissions (if any)
            base_year: Base year for change tracking

        Returns:
            Scope1Emissions result object
        """
        # Initialize accumulators
        total_co2_kg = Decimal("0")
        total_ch4_kg = Decimal("0")
        total_n2o_kg = Decimal("0")
        total_co2e_kg = Decimal("0")
        total_biogenic_co2_kg = Decimal("0")
        emissions_by_category: Dict[str, Decimal] = {
            FuelCategory.GASEOUS.value: Decimal("0"),
            FuelCategory.LIQUID.value: Decimal("0"),
            FuelCategory.SOLID.value: Decimal("0"),
            FuelCategory.BIOMASS.value: Decimal("0"),
        }

        # Process each fuel source
        for fd in fuel_data:
            fuel_type = fd.get("fuel_type", "")
            quantity = Decimal(str(fd.get("quantity", 0)))

            co2_kg, ch4_kg, n2o_kg, co2e_kg, is_biogenic = self.calculate_fuel_emissions(
                fuel_type, quantity
            )

            if is_biogenic:
                # Biogenic CO2 reported separately
                total_biogenic_co2_kg += co2_kg
                # Only CH4 and N2O count for Scope 1
                co2e_kg_adjusted = self._quantize(
                    ch4_kg * GWP_AR5["ch4"] +
                    n2o_kg * GWP_AR5["n2o"]
                )
                total_co2e_kg += co2e_kg_adjusted
                emissions_by_category[FuelCategory.BIOMASS.value] += co2e_kg_adjusted
            else:
                total_co2_kg += co2_kg
                total_co2e_kg += co2e_kg

                # Categorize by fuel type
                category = self._get_fuel_category(fuel_type)
                emissions_by_category[category.value] += co2e_kg

            total_ch4_kg += ch4_kg
            total_n2o_kg += n2o_kg

        # Add mobile, process, fugitive emissions
        stationary_combustion_co2e_kg = total_co2e_kg
        total_co2e_kg += mobile_emissions_co2e_kg
        total_co2e_kg += process_emissions_co2e_kg
        total_co2e_kg += fugitive_emissions_co2e_kg

        # Convert to tonnes
        total_co2_tonnes = self._quantize(total_co2_kg / Decimal("1000"))
        total_ch4_tonnes = self._quantize(total_ch4_kg / Decimal("1000"))
        total_n2o_tonnes = self._quantize(total_n2o_kg / Decimal("1000"))
        total_co2e_tonnes = self._quantize(total_co2e_kg / Decimal("1000"))
        biogenic_co2_tonnes = self._quantize(total_biogenic_co2_kg / Decimal("1000"))

        # Calculate change from base year if applicable
        change_from_base = None
        if base_year:
            key = f"{facility_id}_{base_year}"
            if key in self._base_year_emissions:
                base_emissions = self._base_year_emissions[key]
                if base_emissions > 0:
                    change_from_base = float(
                        (total_co2e_tonnes - base_emissions) / base_emissions * 100
                    )

        # Compute provenance hash
        provenance = self._compute_hash({
            "facility_id": facility_id,
            "reporting_year": reporting_year,
            "fuel_data_count": len(fuel_data),
            "total_co2e_tonnes": str(total_co2e_tonnes),
        })

        return Scope1Emissions(
            facility_id=facility_id,
            reporting_year=reporting_year,
            total_co2_tonnes=total_co2_tonnes,
            total_ch4_tonnes=total_ch4_tonnes,
            total_n2o_tonnes=total_n2o_tonnes,
            total_co2e_tonnes=total_co2e_tonnes,
            stationary_combustion_co2e=self._quantize(stationary_combustion_co2e_kg / Decimal("1000")),
            mobile_combustion_co2e=self._quantize(mobile_emissions_co2e_kg / Decimal("1000")),
            process_emissions_co2e=self._quantize(process_emissions_co2e_kg / Decimal("1000")),
            fugitive_emissions_co2e=self._quantize(fugitive_emissions_co2e_kg / Decimal("1000")),
            emissions_by_fuel_category={
                k: self._quantize(v / Decimal("1000"))
                for k, v in emissions_by_category.items()
            },
            biogenic_co2_tonnes=biogenic_co2_tonnes,
            base_year=base_year,
            change_from_base_year_percent=change_from_base,
            provenance_hash=provenance,
        )

    def _get_fuel_category(self, fuel_type: str) -> FuelCategory:
        """Determine fuel category from fuel type."""
        fuel_lower = fuel_type.lower()

        if any(g in fuel_lower for g in ["gas", "propane", "methane", "lpg", "cng"]):
            return FuelCategory.GASEOUS
        elif any(l in fuel_lower for l in ["oil", "diesel", "petrol", "gasoline", "kerosene"]):
            return FuelCategory.LIQUID
        elif any(s in fuel_lower for s in ["coal", "coke", "anthracite"]):
            return FuelCategory.SOLID
        elif any(b in fuel_lower for b in ["bio", "wood", "pellet", "biomass"]):
            return FuelCategory.BIOMASS
        else:
            return FuelCategory.GASEOUS  # Default

    def get_supported_fuels(self) -> List[str]:
        """Get list of supported fuel types."""
        all_fuels = set(DEFAULT_EMISSION_FACTORS.keys())
        all_fuels.update(self._custom_factors.keys())
        return sorted(list(all_fuels))
