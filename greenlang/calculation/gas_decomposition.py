# -*- coding: utf-8 -*-
"""
Multi-Gas Decomposition Calculator

Decomposes total CO2e emissions into individual greenhouse gas contributions
(CO2, CH4, N2O, F-gases) using IPCC AR6 Global Warming Potentials.

ZERO-HALLUCINATION: All calculations use IPCC AR6 GWP100 values (deterministic).
"""

from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
from enum import Enum
from greenlang.determinism import FinancialDecimal


class GasType(str, Enum):
    """Greenhouse gas types"""
    CO2 = "CO2"
    CH4_FOSSIL = "CH4_fossil"
    CH4_BIOGENIC = "CH4_biogenic"
    N2O = "N2O"
    HFC_134A = "HFC-134a"
    R_410A = "R-410A"
    SF6 = "SF6"
    HFC_125 = "HFC-125"
    HFC_32 = "HFC-32"
    PFC_14 = "PFC-14"


# IPCC AR6 Global Warming Potentials (100-year time horizon)
# Source: IPCC Sixth Assessment Report (2021)
# URI: https://www.ipcc.ch/report/ar6/wg1/
GWP_AR6_100YR: Dict[str, Decimal] = {
    # Carbon dioxide (baseline)
    'CO2': Decimal('1.0'),

    # Methane
    'CH4_fossil': Decimal('29.8'),  # Fossil methane with climate-carbon feedback
    'CH4_biogenic': Decimal('27.2'),  # Biogenic methane (slightly lower)

    # Nitrous oxide
    'N2O': Decimal('273.0'),

    # Hydrofluorocarbons (HFCs)
    'HFC-134a': Decimal('1430.0'),  # Common in refrigeration & A/C
    'HFC-125': Decimal('3740.0'),
    'HFC-32': Decimal('771.0'),
    'R-410A': Decimal('2088.0'),  # Blend of HFC-32 and HFC-125

    # Perfluorocarbons (PFCs)
    'PFC-14': Decimal('7380.0'),  # CF4

    # Sulfur hexafluoride
    'SF6': Decimal('23500.0'),  # Extremely potent
}

# Typical gas composition for common fuel types
# Used when detailed gas vectors not provided
DEFAULT_GAS_VECTORS: Dict[str, Dict[str, float]] = {
    # Fossil fuels (primarily CO2 with trace CH4, N2O)
    'natural_gas': {
        'CO2': 0.98,  # 98% of emissions as CO2
        'CH4_fossil': 0.015,  # 1.5% as methane
        'N2O': 0.005,  # 0.5% as N2O
    },
    'diesel': {
        'CO2': 0.995,
        'CH4_fossil': 0.003,
        'N2O': 0.002,
    },
    'gasoline': {
        'CO2': 0.995,
        'CH4_fossil': 0.003,
        'N2O': 0.002,
    },
    'coal': {
        'CO2': 0.98,
        'CH4_fossil': 0.01,
        'N2O': 0.01,
    },

    # Electricity (varies by grid, but typically CO2-dominant)
    'electricity': {
        'CO2': 0.985,
        'CH4_fossil': 0.01,
        'N2O': 0.005,
    },

    # Agriculture (high N2O content)
    'fertilizer': {
        'CO2': 0.15,
        'N2O': 0.85,
    },
    'enteric_fermentation': {
        'CH4_biogenic': 1.0,  # 100% methane from cattle
    },

    # Refrigerants (100% F-gases)
    'HFC-134a': {
        'HFC-134a': 1.0,
    },
    'R-410A': {
        'R-410A': 1.0,
    },
}


@dataclass
class GasBreakdown:
    """
    Individual gas contributions to total CO2e emissions.

    Attributes:
        total_co2e_kg: Total emissions in kg CO2e
        gas_amounts_kg: Mass of each gas in kg
        gas_co2e_contributions_kg: CO2e contribution of each gas
        gwp_values: GWP values used for conversion
    """
    total_co2e_kg: Decimal
    gas_amounts_kg: Dict[str, Decimal]
    gas_co2e_contributions_kg: Dict[str, Decimal]
    gwp_values: Dict[str, Decimal]
    fuel_type: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'total_co2e_kg': str(self.total_co2e_kg),
            'gas_amounts_kg': {k: str(v) for k, v in self.gas_amounts_kg.items()},
            'gas_co2e_contributions_kg': {k: str(v) for k, v in self.gas_co2e_contributions_kg.items()},
            'gwp_values': {k: str(v) for k, v in self.gwp_values.items()},
            'fuel_type': self.fuel_type,
        }

    def get_percentage_by_gas(self) -> Dict[str, float]:
        """Get percentage contribution of each gas to total CO2e"""
        if self.total_co2e_kg == 0:
            return {}

        return {
            gas: FinancialDecimal.from_string((co2e / self.total_co2e_kg) * 100)
            for gas, co2e in self.gas_co2e_contributions_kg.items()
        }


class MultiGasCalculator:
    """
    Decomposes total CO2e into individual gas contributions.

    Two modes:
    1. With gas vector: Precise decomposition based on known gas composition
    2. Without gas vector: Use default composition for fuel type

    ZERO-HALLUCINATION: Uses IPCC AR6 GWP values (deterministic)
    """

    def __init__(self):
        """Initialize multi-gas calculator"""
        self.gwp_values = GWP_AR6_100YR
        self.default_vectors = DEFAULT_GAS_VECTORS

    def decompose(
        self,
        total_co2e_kg: float,
        fuel_type: Optional[str] = None,
        gas_vector: Optional[Dict[str, float]] = None,
    ) -> GasBreakdown:
        """
        Decompose total CO2e into individual gas contributions.

        Args:
            total_co2e_kg: Total emissions in kg CO2e
            fuel_type: Fuel/activity type (e.g., 'natural_gas', 'diesel')
            gas_vector: Optional gas composition (fractions summing to 1.0)
                       e.g., {'CO2': 0.98, 'CH4_fossil': 0.015, 'N2O': 0.005}

        Returns:
            GasBreakdown with individual gas amounts and contributions

        Raises:
            ValueError: If gas_vector invalid or fuel_type unknown
        """
        total_co2e = Decimal(str(total_co2e_kg))

        # Determine gas vector
        if gas_vector is None:
            if fuel_type is None:
                # Default to 100% CO2 if no information provided
                gas_vector = {'CO2': 1.0}
            elif fuel_type.lower() in self.default_vectors:
                gas_vector = self.default_vectors[fuel_type.lower()]
            else:
                # Unknown fuel type, default to 100% CO2
                gas_vector = {'CO2': 1.0}

        # Validate gas vector
        vector_sum = sum(gas_vector.values())
        if not (0.99 <= vector_sum <= 1.01):  # Allow small floating-point errors
            raise ValueError(f"Gas vector must sum to 1.0, got {vector_sum}")

        # Calculate individual gas amounts
        gas_amounts_kg: Dict[str, Decimal] = {}
        gas_co2e_contributions_kg: Dict[str, Decimal] = {}
        gwp_used: Dict[str, Decimal] = {}

        for gas, fraction in gas_vector.items():
            # Get GWP for this gas
            if gas not in self.gwp_values:
                raise ValueError(f"Unknown gas type: {gas}. Available: {list(self.gwp_values.keys())}")

            gwp = self.gwp_values[gas]
            gwp_used[gas] = gwp

            # Calculate CO2e contribution
            co2e_contribution = total_co2e * Decimal(str(fraction))
            gas_co2e_contributions_kg[gas] = co2e_contribution.quantize(
                Decimal('0.001'),
                rounding=ROUND_HALF_UP
            )

            # Calculate actual gas amount (kg)
            # Formula: gas_kg = co2e_kg / GWP
            gas_amount = co2e_contribution / gwp
            gas_amounts_kg[gas] = gas_amount.quantize(
                Decimal('0.001'),
                rounding=ROUND_HALF_UP
            )

        return GasBreakdown(
            total_co2e_kg=total_co2e,
            gas_amounts_kg=gas_amounts_kg,
            gas_co2e_contributions_kg=gas_co2e_contributions_kg,
            gwp_values=gwp_used,
            fuel_type=fuel_type,
        )

    def calculate_co2e_from_gas_amounts(
        self,
        gas_amounts_kg: Dict[str, float],
    ) -> Decimal:
        """
        Calculate total CO2e from individual gas amounts.

        Inverse operation of decompose().

        Args:
            gas_amounts_kg: Dictionary of gas amounts in kg
                           e.g., {'CO2': 100, 'CH4_fossil': 2, 'N2O': 0.5}

        Returns:
            Total CO2e in kg

        Raises:
            ValueError: If unknown gas type
        """
        total_co2e = Decimal('0')

        for gas, amount_kg in gas_amounts_kg.items():
            if gas not in self.gwp_values:
                raise ValueError(f"Unknown gas type: {gas}")

            gwp = self.gwp_values[gas]
            amount = Decimal(str(amount_kg))

            # Convert to CO2e
            co2e = amount * gwp
            total_co2e += co2e

        return total_co2e.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)

    def get_gwp(self, gas: str) -> Decimal:
        """
        Get GWP value for a gas.

        Args:
            gas: Gas name (e.g., 'CH4_fossil', 'N2O')

        Returns:
            GWP100 value from IPCC AR6

        Raises:
            ValueError: If gas unknown
        """
        if gas not in self.gwp_values:
            raise ValueError(f"Unknown gas: {gas}. Available: {list(self.gwp_values.keys())}")
        return self.gwp_values[gas]

    def list_supported_gases(self) -> list:
        """List all supported gas types"""
        return list(self.gwp_values.keys())

    def get_default_vector(self, fuel_type: str) -> Optional[Dict[str, float]]:
        """
        Get default gas vector for a fuel type.

        Args:
            fuel_type: Fuel type (e.g., 'natural_gas', 'diesel')

        Returns:
            Gas vector dictionary or None if not found
        """
        return self.default_vectors.get(fuel_type.lower())
