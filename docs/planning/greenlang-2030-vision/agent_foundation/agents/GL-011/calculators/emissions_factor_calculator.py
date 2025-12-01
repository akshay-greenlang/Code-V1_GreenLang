# -*- coding: utf-8 -*-
"""
Emissions Factor Calculator for GL-011 FUELCRAFT.

Provides deterministic emission factor lookups and calculations
for NOx, SOx, CO2, CH4, N2O, and particulate matter.

Standards: IPCC 2006 Guidelines, GHG Protocol, EPA AP-42
Zero-hallucination: All calculations are deterministic.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# IPCC 2006 Default Emission Factors (stationary combustion)
IPCC_EMISSION_FACTORS = {
    'coal': {
        'co2_kg_gj': 94.6,
        'ch4_kg_gj': 0.001,
        'n2o_kg_gj': 0.0015,
        'source': 'IPCC 2006 Vol 2 Table 2.2',
        'tier': 1
    },
    'natural_gas': {
        'co2_kg_gj': 56.1,
        'ch4_kg_gj': 0.001,
        'n2o_kg_gj': 0.0001,
        'source': 'IPCC 2006 Vol 2 Table 2.2',
        'tier': 1
    },
    'fuel_oil': {
        'co2_kg_gj': 77.4,
        'ch4_kg_gj': 0.003,
        'n2o_kg_gj': 0.0006,
        'source': 'IPCC 2006 Vol 2 Table 2.2',
        'tier': 1
    },
    'diesel': {
        'co2_kg_gj': 74.1,
        'ch4_kg_gj': 0.003,
        'n2o_kg_gj': 0.0006,
        'source': 'IPCC 2006 Vol 2 Table 2.2',
        'tier': 1
    },
    'biomass': {
        'co2_kg_gj': 0.0,  # Biogenic - carbon neutral
        'ch4_kg_gj': 0.030,
        'n2o_kg_gj': 0.004,
        'source': 'IPCC 2006 Vol 2 Table 2.2',
        'tier': 1
    },
    'hydrogen': {
        'co2_kg_gj': 0.0,
        'ch4_kg_gj': 0.0,
        'n2o_kg_gj': 0.0,
        'source': 'GreenLang',
        'tier': 1
    },
    'propane': {
        'co2_kg_gj': 63.1,
        'ch4_kg_gj': 0.001,
        'n2o_kg_gj': 0.0001,
        'source': 'IPCC 2006 Vol 2 Table 2.2',
        'tier': 1
    },
    'biogas': {
        'co2_kg_gj': 0.0,  # Biogenic
        'ch4_kg_gj': 0.005,
        'n2o_kg_gj': 0.0001,
        'source': 'IPCC 2006 Vol 2',
        'tier': 1
    }
}

# Criteria pollutant emission factors (g/GJ)
CRITERIA_EMISSION_FACTORS = {
    'coal': {
        'nox_g_gj': 250,
        'sox_g_gj': 500,
        'pm_g_gj': 50,
        'co_g_gj': 100
    },
    'natural_gas': {
        'nox_g_gj': 50,
        'sox_g_gj': 0.3,
        'pm_g_gj': 1,
        'co_g_gj': 25
    },
    'fuel_oil': {
        'nox_g_gj': 200,
        'sox_g_gj': 600,
        'pm_g_gj': 30,
        'co_g_gj': 50
    },
    'diesel': {
        'nox_g_gj': 180,
        'sox_g_gj': 30,
        'pm_g_gj': 20,
        'co_g_gj': 40
    },
    'biomass': {
        'nox_g_gj': 150,
        'sox_g_gj': 20,
        'pm_g_gj': 30,
        'co_g_gj': 200
    },
    'hydrogen': {
        'nox_g_gj': 10,
        'sox_g_gj': 0,
        'pm_g_gj': 0,
        'co_g_gj': 0
    },
    'propane': {
        'nox_g_gj': 60,
        'sox_g_gj': 0.1,
        'pm_g_gj': 2,
        'co_g_gj': 30
    },
    'biogas': {
        'nox_g_gj': 40,
        'sox_g_gj': 15,
        'pm_g_gj': 5,
        'co_g_gj': 50
    }
}


@dataclass
class EmissionFactorInput:
    """Input for emission factor lookup."""
    fuel_type: str
    combustion_technology: str = 'boiler'
    emission_control: str = 'uncontrolled'
    region: str = 'default'


@dataclass
class EmissionFactorOutput:
    """Output of emission factor lookup."""
    fuel_type: str
    co2_kg_gj: float
    ch4_kg_gj: float
    n2o_kg_gj: float
    nox_g_gj: float
    sox_g_gj: float
    pm_g_gj: float
    co_g_gj: float
    co2e_kg_gj: float  # Total CO2 equivalent
    source: str
    tier: int
    uncertainty_percent: float
    provenance_hash: str


class EmissionsFactorCalculator:
    """
    Deterministic emissions factor calculator.

    Provides emission factor lookups and calculations for GHG
    and criteria pollutants using IPCC, GHG Protocol, and EPA factors.
    """

    # GWP values (AR6)
    GWP_CH4 = 29.8
    GWP_N2O = 273.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        self.config = config or {}
        self.ghg_factors = IPCC_EMISSION_FACTORS.copy()
        self.criteria_factors = CRITERIA_EMISSION_FACTORS.copy()
        self.calculation_count = 0

    def get_emission_factors(
        self,
        input_data: EmissionFactorInput
    ) -> EmissionFactorOutput:
        """
        Get emission factors for a fuel type.

        Args:
            input_data: Fuel type and technology parameters

        Returns:
            Complete emission factors
        """
        self.calculation_count += 1

        fuel = input_data.fuel_type
        ghg = self.ghg_factors.get(fuel, self.ghg_factors.get('coal'))
        criteria = self.criteria_factors.get(fuel, self.criteria_factors.get('coal'))

        # Apply control technology adjustments
        control_reduction = self._get_control_reduction(
            input_data.emission_control
        )

        nox_adjusted = criteria['nox_g_gj'] * (1 - control_reduction.get('nox', 0))
        sox_adjusted = criteria['sox_g_gj'] * (1 - control_reduction.get('sox', 0))
        pm_adjusted = criteria['pm_g_gj'] * (1 - control_reduction.get('pm', 0))

        # Calculate CO2e
        co2e = (
            ghg['co2_kg_gj'] +
            ghg['ch4_kg_gj'] * self.GWP_CH4 +
            ghg['n2o_kg_gj'] * self.GWP_N2O
        )

        # Uncertainty
        uncertainty = 5.0 if ghg.get('tier', 1) == 1 else 15.0

        provenance = self._calculate_provenance(input_data, co2e)

        return EmissionFactorOutput(
            fuel_type=fuel,
            co2_kg_gj=round(ghg['co2_kg_gj'], 2),
            ch4_kg_gj=round(ghg['ch4_kg_gj'], 6),
            n2o_kg_gj=round(ghg['n2o_kg_gj'], 6),
            nox_g_gj=round(nox_adjusted, 1),
            sox_g_gj=round(sox_adjusted, 1),
            pm_g_gj=round(pm_adjusted, 1),
            co_g_gj=round(criteria['co_g_gj'], 1),
            co2e_kg_gj=round(co2e, 2),
            source=ghg.get('source', 'IPCC 2006'),
            tier=ghg.get('tier', 1),
            uncertainty_percent=uncertainty,
            provenance_hash=provenance
        )

    def calculate_emissions(
        self,
        fuel_type: str,
        energy_gj: float,
        emission_control: str = 'uncontrolled'
    ) -> Dict[str, float]:
        """
        Calculate emissions for given energy consumption.

        Args:
            fuel_type: Type of fuel
            energy_gj: Energy consumption in GJ
            emission_control: Control technology

        Returns:
            Emissions by pollutant in kg
        """
        factors = self.get_emission_factors(EmissionFactorInput(
            fuel_type=fuel_type,
            emission_control=emission_control
        ))

        return {
            'co2_kg': factors.co2_kg_gj * energy_gj,
            'ch4_kg': factors.ch4_kg_gj * energy_gj,
            'n2o_kg': factors.n2o_kg_gj * energy_gj,
            'nox_kg': factors.nox_g_gj * energy_gj / 1000,
            'sox_kg': factors.sox_g_gj * energy_gj / 1000,
            'pm_kg': factors.pm_g_gj * energy_gj / 1000,
            'co2e_kg': factors.co2e_kg_gj * energy_gj
        }

    def _get_control_reduction(
        self,
        control_type: str
    ) -> Dict[str, float]:
        """Get emission reduction fractions for control technology."""
        reductions = {
            'uncontrolled': {'nox': 0, 'sox': 0, 'pm': 0},
            'low_nox_burner': {'nox': 0.4, 'sox': 0, 'pm': 0},
            'scr': {'nox': 0.9, 'sox': 0, 'pm': 0},
            'fgd': {'nox': 0, 'sox': 0.95, 'pm': 0.5},
            'esp': {'nox': 0, 'sox': 0, 'pm': 0.99},
            'baghouse': {'nox': 0, 'sox': 0, 'pm': 0.999},
            'combined': {'nox': 0.9, 'sox': 0.95, 'pm': 0.99}
        }
        return reductions.get(control_type, reductions['uncontrolled'])

    def get_available_fuels(self) -> List[str]:
        """Get list of available fuel types."""
        return list(self.ghg_factors.keys())

    def _calculate_provenance(
        self,
        input_data: EmissionFactorInput,
        result: float
    ) -> str:
        """Calculate provenance hash."""
        data = {
            'fuel_type': input_data.fuel_type,
            'technology': input_data.combustion_technology,
            'control': input_data.emission_control,
            'result': result
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
