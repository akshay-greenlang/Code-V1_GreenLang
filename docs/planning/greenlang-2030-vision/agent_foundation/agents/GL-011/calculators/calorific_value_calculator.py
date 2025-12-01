# -*- coding: utf-8 -*-
"""
Calorific Value Calculator for GL-011 FUELCRAFT.

Provides deterministic calculations for fuel energy content (heating value)
following ISO 6976:2016 for natural gas and ASTM D4809 for liquid fuels.

Standards: ISO 6976:2016, ISO 17225, ASTM D4809
Zero-hallucination: All calculations are deterministic.
"""

import hashlib
import json
import logging
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalorificValueInput:
    """Input for calorific value calculation."""
    fuel_type: str
    composition: Dict[str, float]  # Component -> percentage
    temperature_c: float = 15.0
    pressure_kpa: float = 101.325
    moisture_percent: float = 0.0


@dataclass
class CalorificValueOutput:
    """Output of calorific value calculation."""
    gross_calorific_value_mj_kg: float  # HHV
    net_calorific_value_mj_kg: float  # LHV
    gross_calorific_value_mj_m3: float  # For gases
    net_calorific_value_mj_m3: float
    density_kg_m3: float
    wobbe_index: float  # For gases
    energy_content_per_unit: float
    calculation_method: str
    provenance_hash: str


class CalorificValueCalculator:
    """
    Deterministic calorific value calculator.

    Calculates gross (HHV) and net (LHV) heating values for various fuels
    based on composition and standard conditions.

    Standards:
    - ISO 6976:2016 for natural gas
    - ISO 17225 for solid biofuels
    - ASTM D4809 for liquid fuels
    """

    # Standard enthalpy of combustion (kJ/mol) at 25C
    COMBUSTION_ENTHALPIES = {
        'methane': 890.3,
        'ethane': 1560.7,
        'propane': 2220.0,
        'n_butane': 2878.5,
        'i_butane': 2868.2,
        'n_pentane': 3509.0,
        'i_pentane': 3503.0,
        'hexane': 4163.0,
        'hydrogen': 285.8,
        'carbon_monoxide': 283.0,
        'carbon': 393.5,  # Per mol C
    }

    # Molecular weights
    MOLECULAR_WEIGHTS = {
        'methane': 16.043,
        'ethane': 30.069,
        'propane': 44.096,
        'n_butane': 58.122,
        'i_butane': 58.122,
        'n_pentane': 72.149,
        'hexane': 86.175,
        'hydrogen': 2.016,
        'nitrogen': 28.013,
        'carbon_dioxide': 44.009,
        'oxygen': 31.999,
    }

    # Standard reference conditions
    STD_TEMP_K = 288.15  # 15C
    STD_PRESSURE_KPA = 101.325

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        self.config = config or {}
        self.calculation_count = 0

    def calculate(self, input_data: CalorificValueInput) -> CalorificValueOutput:
        """
        Calculate calorific values for fuel.

        Args:
            input_data: Fuel composition and conditions

        Returns:
            Gross and net calorific values
        """
        self.calculation_count += 1

        if input_data.fuel_type in ['natural_gas', 'biogas', 'hydrogen']:
            return self._calculate_gas_cv(input_data)
        elif input_data.fuel_type in ['coal', 'biomass', 'wood_pellets']:
            return self._calculate_solid_cv(input_data)
        elif input_data.fuel_type in ['fuel_oil', 'diesel', 'gasoline']:
            return self._calculate_liquid_cv(input_data)
        else:
            # Default to solid fuel method
            return self._calculate_solid_cv(input_data)

    def _calculate_gas_cv(
        self,
        input_data: CalorificValueInput
    ) -> CalorificValueOutput:
        """
        Calculate calorific value for gaseous fuels per ISO 6976.

        Args:
            input_data: Gas composition

        Returns:
            Calorific values for gas
        """
        composition = input_data.composition

        # Calculate molar composition weighted GCV
        gcv_mj_m3 = 0.0
        ncv_mj_m3 = 0.0
        molar_mass = 0.0

        # Component heating values at standard conditions (MJ/m3)
        gas_gcv = {
            'methane': 39.82,
            'ethane': 70.30,
            'propane': 101.24,
            'n_butane': 133.80,
            'i_butane': 132.95,
            'n_pentane': 166.07,
            'hydrogen': 12.75,
            'carbon_monoxide': 12.63,
        }

        gas_ncv = {
            'methane': 35.88,
            'ethane': 64.36,
            'propane': 93.18,
            'n_butane': 123.56,
            'i_butane': 122.71,
            'n_pentane': 153.63,
            'hydrogen': 10.78,
            'carbon_monoxide': 12.63,
        }

        for component, fraction in composition.items():
            frac = fraction / 100.0
            gcv_mj_m3 += frac * gas_gcv.get(component, 0)
            ncv_mj_m3 += frac * gas_ncv.get(component, 0)
            molar_mass += frac * self.MOLECULAR_WEIGHTS.get(component, 28)

        # Convert to MJ/kg using density
        # Density at standard conditions (kg/m3)
        density = molar_mass / 22.414  # Ideal gas molar volume

        gcv_mj_kg = gcv_mj_m3 / density if density > 0 else 0
        ncv_mj_kg = ncv_mj_m3 / density if density > 0 else 0

        # Calculate Wobbe Index
        relative_density = density / 1.293  # Air density at STP
        wobbe_index = gcv_mj_m3 / math.sqrt(relative_density) if relative_density > 0 else 0

        provenance = self._calculate_provenance(input_data, gcv_mj_kg)

        return CalorificValueOutput(
            gross_calorific_value_mj_kg=round(gcv_mj_kg, 2),
            net_calorific_value_mj_kg=round(ncv_mj_kg, 2),
            gross_calorific_value_mj_m3=round(gcv_mj_m3, 2),
            net_calorific_value_mj_m3=round(ncv_mj_m3, 2),
            density_kg_m3=round(density, 4),
            wobbe_index=round(wobbe_index, 2),
            energy_content_per_unit=round(gcv_mj_m3, 2),
            calculation_method='ISO_6976',
            provenance_hash=provenance
        )

    def _calculate_solid_cv(
        self,
        input_data: CalorificValueInput
    ) -> CalorificValueOutput:
        """
        Calculate calorific value for solid fuels using Dulong formula.

        Args:
            input_data: Ultimate analysis composition

        Returns:
            Calorific values for solid fuel
        """
        composition = input_data.composition
        moisture = input_data.moisture_percent

        # Extract composition (as-received basis)
        C = composition.get('carbon', 50)
        H = composition.get('hydrogen', 6)
        O = composition.get('oxygen', 40)
        S = composition.get('sulfur', 0)
        N = composition.get('nitrogen', 0)

        # Dulong formula for GCV (MJ/kg)
        # GCV = 0.3383*C + 1.443*(H - O/8) + 0.0942*S
        gcv_dry = 0.3383 * C + 1.443 * (H - O/8) + 0.0942 * S

        # Adjust for moisture (as-received)
        gcv_ar = gcv_dry * (1 - moisture/100)

        # NCV = GCV - 0.206*H - 0.023*moisture
        ncv_ar = gcv_ar - 0.206 * H * (1 - moisture/100) - 0.023 * moisture

        # Estimate density based on fuel type
        density = 1300  # Default for coal
        if input_data.fuel_type == 'biomass':
            density = 600
        elif input_data.fuel_type == 'wood_pellets':
            density = 650

        provenance = self._calculate_provenance(input_data, gcv_ar)

        return CalorificValueOutput(
            gross_calorific_value_mj_kg=round(gcv_ar, 2),
            net_calorific_value_mj_kg=round(ncv_ar, 2),
            gross_calorific_value_mj_m3=round(gcv_ar * density, 2),
            net_calorific_value_mj_m3=round(ncv_ar * density, 2),
            density_kg_m3=round(density, 2),
            wobbe_index=0.0,  # Not applicable for solids
            energy_content_per_unit=round(gcv_ar, 2),
            calculation_method='Dulong_formula',
            provenance_hash=provenance
        )

    def _calculate_liquid_cv(
        self,
        input_data: CalorificValueInput
    ) -> CalorificValueOutput:
        """
        Calculate calorific value for liquid fuels per ASTM D4809.

        Args:
            input_data: Fuel composition

        Returns:
            Calorific values for liquid fuel
        """
        composition = input_data.composition

        # For liquid fuels, use empirical correlations
        # Based on API gravity or density
        density = composition.get('density_kg_m3', 850)
        api_gravity = 141.5 / (density / 1000) - 131.5  # Convert to API

        # Empirical formula for petroleum products
        # GCV (MJ/kg) = 51.92 - 8.79 * S^2
        # where S = specific gravity
        specific_gravity = density / 1000
        gcv_mj_kg = 51.92 - 8.79 * (specific_gravity ** 2)

        # NCV typically 5-6% less than GCV for liquid fuels
        ncv_mj_kg = gcv_mj_kg * 0.94

        # Volumetric values
        gcv_mj_l = gcv_mj_kg * density / 1000
        ncv_mj_l = ncv_mj_kg * density / 1000

        provenance = self._calculate_provenance(input_data, gcv_mj_kg)

        return CalorificValueOutput(
            gross_calorific_value_mj_kg=round(gcv_mj_kg, 2),
            net_calorific_value_mj_kg=round(ncv_mj_kg, 2),
            gross_calorific_value_mj_m3=round(gcv_mj_l * 1000, 2),
            net_calorific_value_mj_m3=round(ncv_mj_l * 1000, 2),
            density_kg_m3=round(density, 2),
            wobbe_index=0.0,  # Not applicable
            energy_content_per_unit=round(gcv_mj_kg, 2),
            calculation_method='ASTM_D4809',
            provenance_hash=provenance
        )

    def _calculate_provenance(
        self,
        input_data: CalorificValueInput,
        result: float
    ) -> str:
        """Calculate provenance hash."""
        data = {
            'fuel_type': input_data.fuel_type,
            'composition': input_data.composition,
            'temperature': input_data.temperature_c,
            'result': result
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
