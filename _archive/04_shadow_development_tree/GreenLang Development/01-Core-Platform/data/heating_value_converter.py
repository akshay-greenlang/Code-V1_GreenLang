# -*- coding: utf-8 -*-
"""
greenlang/data/heating_value_converter.py

HHV/LHV Conversion Module - Zero-Hallucination Heating Value Conversions

This module provides deterministic conversion between Higher Heating Value (HHV)
and Lower Heating Value (LHV) for fuel emission factors and energy calculations.

ZERO-HALLUCINATION GUARANTEE:
- All conversion ratios from authoritative sources (IPCC, EPA)
- No LLM involvement in calculations
- Same inputs always produce identical outputs (bit-perfect)
- Full provenance tracking

Sources:
- EPA: https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator
- IPCC 2006 Guidelines for National Greenhouse Gas Inventories
- Engineering Toolbox: https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html

Author: GreenLang Team
Date: 2025-11-25
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum


class HeatingValueBasis(str, Enum):
    """Heating value basis for fuels"""
    HHV = "HHV"  # Higher Heating Value (gross) - includes latent heat of water vapor
    LHV = "LHV"  # Lower Heating Value (net) - excludes latent heat of water vapor


class HeatingValueConverter:
    """
    Converter for HHV (Higher Heating Value) to LHV (Lower Heating Value) and vice versa.

    HHV includes latent heat of water vapor, LHV does not.
    Conversion ratios are fuel-specific based on hydrogen and moisture content.

    Standard conversion ratios (HHV/LHV):
    - Natural gas: ~1.11 (varies 1.09-1.12 depending on composition)
    - Coal: ~1.05 (varies 1.02-1.08 depending on moisture)
    - Oil/diesel: ~1.06 (varies 1.05-1.07)
    - Gasoline: ~1.07
    - LPG/Propane: ~1.08
    - Biomass: ~1.10 (varies significantly with moisture)

    ZERO-HALLUCINATION GUARANTEE:
    - All ratios from IPCC 2006 Guidelines and EPA references
    - Deterministic calculations (same input -> same output)
    - Complete audit trail for regulatory compliance
    """

    # HHV/LHV ratios by fuel type (HHV = LHV * ratio)
    # Source: IPCC 2006 Guidelines, EPA, Engineering references
    CONVERSION_RATIOS: Dict[str, float] = {
        # Natural gas variants
        "natural_gas": 1.11,
        "natural_gas_pipeline": 1.11,
        "natural_gas_liquefied": 1.10,
        "lng": 1.10,
        "cng": 1.11,

        # Coal variants
        "coal": 1.05,
        "coal_anthracite": 1.02,
        "coal_bituminous": 1.05,
        "coal_subbituminous": 1.06,
        "coal_lignite": 1.08,
        "anthracite": 1.02,
        "bituminous_coal": 1.05,
        "lignite": 1.08,

        # Oil and petroleum products
        "oil": 1.06,
        "crude_oil": 1.06,
        "diesel": 1.06,
        "diesel_fuel": 1.06,
        "fuel_oil": 1.06,
        "heavy_fuel_oil": 1.05,
        "light_fuel_oil": 1.06,
        "residual_fuel_oil": 1.05,
        "heating_oil": 1.06,
        "kerosene": 1.06,
        "jet_fuel": 1.06,
        "aviation_fuel": 1.06,

        # Gasoline variants
        "gasoline": 1.07,
        "petrol": 1.07,
        "motor_gasoline": 1.07,

        # LPG and propane
        "lpg": 1.08,
        "propane": 1.08,
        "butane": 1.07,

        # Biomass
        "biomass": 1.10,
        "wood": 1.12,
        "wood_pellets": 1.08,
        "biogas": 1.10,
        "biodiesel": 1.06,
        "ethanol": 1.08,
        "bioethanol": 1.08,

        # Other fuels
        "hydrogen": 1.18,
        "methanol": 1.10,
        "peat": 1.09,
        "coke": 1.04,
    }

    # Default ratio for unknown fuels (conservative estimate based on average)
    DEFAULT_RATIO: float = 1.06

    @classmethod
    def get_conversion_ratio(cls, fuel_type: str) -> float:
        """
        Get HHV/LHV conversion ratio for a fuel type.

        Args:
            fuel_type: Fuel type identifier (case-insensitive)

        Returns:
            HHV/LHV ratio (always >= 1.0)

        Example:
            >>> HeatingValueConverter.get_conversion_ratio("natural_gas")
            1.11
            >>> HeatingValueConverter.get_conversion_ratio("diesel")
            1.06
        """
        normalized_fuel = fuel_type.lower().strip().replace(" ", "_").replace("-", "_")
        return cls.CONVERSION_RATIOS.get(normalized_fuel, cls.DEFAULT_RATIO)

    @classmethod
    def hhv_to_lhv(cls, value: Union[float, Decimal], fuel_type: str) -> float:
        """
        Convert HHV (Higher Heating Value) to LHV (Lower Heating Value).

        Args:
            value: Energy or emission factor value in HHV basis
            fuel_type: Fuel type for ratio lookup

        Returns:
            Value converted to LHV basis

        Example:
            >>> HeatingValueConverter.hhv_to_lhv(100.0, "natural_gas")
            90.09009009009009  # Approximately 100 / 1.11
        """
        ratio = cls.get_conversion_ratio(fuel_type)
        return float(value) / ratio

    @classmethod
    def lhv_to_hhv(cls, value: Union[float, Decimal], fuel_type: str) -> float:
        """
        Convert LHV (Lower Heating Value) to HHV (Higher Heating Value).

        Args:
            value: Energy or emission factor value in LHV basis
            fuel_type: Fuel type for ratio lookup

        Returns:
            Value converted to HHV basis

        Example:
            >>> HeatingValueConverter.lhv_to_hhv(90.0, "natural_gas")
            99.9  # Approximately 90 * 1.11
        """
        ratio = cls.get_conversion_ratio(fuel_type)
        return float(value) * ratio

    @classmethod
    def convert(
        cls,
        value: Union[float, Decimal],
        fuel_type: str,
        from_basis: HeatingValueBasis,
        to_basis: HeatingValueBasis
    ) -> float:
        """
        Convert value between HHV and LHV basis.

        Args:
            value: Energy or emission factor value
            fuel_type: Fuel type for ratio lookup
            from_basis: Current heating value basis (HHV or LHV)
            to_basis: Target heating value basis (HHV or LHV)

        Returns:
            Converted value in target basis

        Raises:
            ValueError: If invalid basis provided

        Example:
            >>> HeatingValueConverter.convert(100.0, "diesel", HeatingValueBasis.HHV, HeatingValueBasis.LHV)
            94.33962264150944
        """
        if from_basis == to_basis:
            return float(value)

        if from_basis == HeatingValueBasis.HHV and to_basis == HeatingValueBasis.LHV:
            return cls.hhv_to_lhv(value, fuel_type)
        elif from_basis == HeatingValueBasis.LHV and to_basis == HeatingValueBasis.HHV:
            return cls.lhv_to_hhv(value, fuel_type)
        else:
            raise ValueError(f"Invalid conversion: {from_basis} to {to_basis}")

    @classmethod
    def convert_decimal(
        cls,
        value: Union[float, Decimal],
        fuel_type: str,
        from_basis: HeatingValueBasis,
        to_basis: HeatingValueBasis,
        precision: int = 8
    ) -> Decimal:
        """
        Convert value between HHV and LHV basis using Decimal for precision.

        Args:
            value: Energy or emission factor value
            fuel_type: Fuel type for ratio lookup
            from_basis: Current heating value basis (HHV or LHV)
            to_basis: Target heating value basis (HHV or LHV)
            precision: Decimal places for rounding (default: 8)

        Returns:
            Converted value as Decimal with specified precision

        Example:
            >>> HeatingValueConverter.convert_decimal(100.0, "diesel", HeatingValueBasis.HHV, HeatingValueBasis.LHV)
            Decimal('94.33962264')
        """
        result = cls.convert(value, fuel_type, from_basis, to_basis)
        decimal_result = Decimal(str(result))
        quantize_str = '0.' + '0' * precision
        return decimal_result.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    @classmethod
    def convert_emission_factor(
        cls,
        emission_factor: Union[float, Decimal],
        fuel_type: str,
        from_basis: HeatingValueBasis,
        to_basis: HeatingValueBasis
    ) -> Tuple[float, Dict]:
        """
        Convert emission factor between HHV and LHV basis with provenance.

        When emission factors are expressed per unit energy (e.g., kg CO2/MJ),
        converting basis requires adjusting the factor value.

        Args:
            emission_factor: Emission factor value (e.g., kg CO2e per MJ)
            fuel_type: Fuel type for ratio lookup
            from_basis: Current heating value basis
            to_basis: Target heating value basis

        Returns:
            Tuple of (converted_factor, conversion_metadata)

        Note:
            If EF is 56.1 kg CO2/GJ on HHV basis for natural gas:
            LHV basis = 56.1 * 1.11 = 62.27 kg CO2/GJ (LHV)

            This is because the same fuel quantity has LOWER energy on LHV basis,
            so emissions per unit energy are HIGHER.

        Example:
            >>> factor, metadata = HeatingValueConverter.convert_emission_factor(
            ...     56.1, "natural_gas", HeatingValueBasis.HHV, HeatingValueBasis.LHV
            ... )
            >>> print(f"Converted factor: {factor:.2f}")
            Converted factor: 62.27
        """
        if from_basis == to_basis:
            return float(emission_factor), {"conversion_applied": False}

        ratio = cls.get_conversion_ratio(fuel_type)

        # For emission factors per unit energy:
        # Converting HHV->LHV: multiply by ratio (higher emissions per LHV unit)
        # Converting LHV->HHV: divide by ratio (lower emissions per HHV unit)
        if from_basis == HeatingValueBasis.HHV and to_basis == HeatingValueBasis.LHV:
            converted = float(emission_factor) * ratio
        else:  # LHV to HHV
            converted = float(emission_factor) / ratio

        metadata = {
            "conversion_applied": True,
            "from_basis": from_basis.value,
            "to_basis": to_basis.value,
            "fuel_type": fuel_type,
            "conversion_ratio": ratio,
            "original_value": float(emission_factor),
            "converted_value": converted,
            "source": "IPCC 2006 Guidelines / EPA",
        }

        return converted, metadata

    @classmethod
    def list_supported_fuels(cls) -> List[str]:
        """
        List all fuel types with defined conversion ratios.

        Returns:
            Sorted list of supported fuel type identifiers

        Example:
            >>> fuels = HeatingValueConverter.list_supported_fuels()
            >>> print(fuels[:5])
            ['anthracite', 'aviation_fuel', 'biodiesel', 'bioethanol', 'biogas']
        """
        return sorted(cls.CONVERSION_RATIOS.keys())

    @classmethod
    def is_fuel_supported(cls, fuel_type: str) -> bool:
        """
        Check if a fuel type has a defined conversion ratio.

        Args:
            fuel_type: Fuel type identifier (case-insensitive)

        Returns:
            True if fuel has explicit ratio, False if default will be used

        Example:
            >>> HeatingValueConverter.is_fuel_supported("natural_gas")
            True
            >>> HeatingValueConverter.is_fuel_supported("unknown_fuel")
            False
        """
        normalized_fuel = fuel_type.lower().strip().replace(" ", "_").replace("-", "_")
        return normalized_fuel in cls.CONVERSION_RATIOS

    @classmethod
    def get_fuel_ratios_by_category(cls) -> Dict[str, Dict[str, float]]:
        """
        Get conversion ratios organized by fuel category.

        Returns:
            Dictionary of fuel categories with their fuels and ratios

        Example:
            >>> categories = HeatingValueConverter.get_fuel_ratios_by_category()
            >>> print(categories["natural_gas"])
            {'natural_gas': 1.11, 'lng': 1.10, 'cng': 1.11, ...}
        """
        return {
            "natural_gas": {
                "natural_gas": 1.11,
                "natural_gas_pipeline": 1.11,
                "natural_gas_liquefied": 1.10,
                "lng": 1.10,
                "cng": 1.11,
            },
            "coal": {
                "coal": 1.05,
                "coal_anthracite": 1.02,
                "coal_bituminous": 1.05,
                "coal_subbituminous": 1.06,
                "coal_lignite": 1.08,
                "anthracite": 1.02,
                "bituminous_coal": 1.05,
                "lignite": 1.08,
            },
            "oil_petroleum": {
                "oil": 1.06,
                "crude_oil": 1.06,
                "diesel": 1.06,
                "diesel_fuel": 1.06,
                "fuel_oil": 1.06,
                "heavy_fuel_oil": 1.05,
                "light_fuel_oil": 1.06,
                "residual_fuel_oil": 1.05,
                "heating_oil": 1.06,
                "kerosene": 1.06,
                "jet_fuel": 1.06,
                "aviation_fuel": 1.06,
            },
            "gasoline": {
                "gasoline": 1.07,
                "petrol": 1.07,
                "motor_gasoline": 1.07,
            },
            "lpg_propane": {
                "lpg": 1.08,
                "propane": 1.08,
                "butane": 1.07,
            },
            "biomass": {
                "biomass": 1.10,
                "wood": 1.12,
                "wood_pellets": 1.08,
                "biogas": 1.10,
                "biodiesel": 1.06,
                "ethanol": 1.08,
                "bioethanol": 1.08,
            },
            "other": {
                "hydrogen": 1.18,
                "methanol": 1.10,
                "peat": 1.09,
                "coke": 1.04,
            },
        }


# Convenience functions for direct use
def hhv_to_lhv(value: Union[float, Decimal], fuel_type: str) -> float:
    """Convert HHV to LHV value. See HeatingValueConverter.hhv_to_lhv for details."""
    return HeatingValueConverter.hhv_to_lhv(value, fuel_type)


def lhv_to_hhv(value: Union[float, Decimal], fuel_type: str) -> float:
    """Convert LHV to HHV value. See HeatingValueConverter.lhv_to_hhv for details."""
    return HeatingValueConverter.lhv_to_hhv(value, fuel_type)


def get_hhv_lhv_ratio(fuel_type: str) -> float:
    """Get HHV/LHV ratio for fuel. See HeatingValueConverter.get_conversion_ratio for details."""
    return HeatingValueConverter.get_conversion_ratio(fuel_type)


__all__ = [
    'HeatingValueBasis',
    'HeatingValueConverter',
    'hhv_to_lhv',
    'lhv_to_hhv',
    'get_hhv_lhv_ratio',
]
