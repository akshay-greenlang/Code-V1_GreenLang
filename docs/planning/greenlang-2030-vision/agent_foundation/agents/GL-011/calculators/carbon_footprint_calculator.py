# -*- coding: utf-8 -*-
"""
Carbon Footprint Calculator for GL-011 FUELCRAFT.

Provides deterministic calculations for GHG emissions from fuel combustion
using IPCC Guidelines and GHG Protocol methodologies.

Standards: GHG Protocol, IPCC 2006 Guidelines
Zero-hallucination: All calculations are deterministic.
"""

import hashlib
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# IPCC AR6 Global Warming Potentials (100-year)
GWP_AR6 = {
    'CO2': 1.0,
    'CH4': 29.8,
    'N2O': 273.0
}


@dataclass
class CarbonFootprintInput:
    """Input for carbon footprint calculation."""
    fuel_quantities: Dict[str, float]  # kg per fuel
    fuel_properties: Dict[str, Dict[str, Any]]
    include_upstream: bool = False
    carbon_price_usd_per_tonne: float = 50.0


@dataclass
class CarbonFootprintOutput:
    """Output of carbon footprint calculation."""
    total_co2e_kg: float
    co2_kg: float
    ch4_kg: float
    n2o_kg: float
    biogenic_carbon_kg: float
    fossil_carbon_kg: float
    carbon_intensity_kg_mwh: float
    scope1_emissions_kg: float
    scope3_upstream_kg: float
    carbon_cost_usd: float
    emission_breakdown: Dict[str, Dict[str, float]]
    provenance_hash: str


class CarbonFootprintCalculator:
    """
    Deterministic carbon footprint calculator.

    Calculates GHG emissions (CO2, CH4, N2O) from fuel combustion
    using IPCC emission factors and AR6 GWPs.

    Example:
        >>> calc = CarbonFootprintCalculator()
        >>> result = calc.calculate(input_data)
        >>> print(f"Total CO2e: {result.total_co2e_kg} kg")
    """

    MJ_TO_GJ = 0.001

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calculator."""
        self.config = config or {}
        self.gwp = GWP_AR6.copy()
        self.calculation_count = 0

    def calculate(self, input_data: CarbonFootprintInput) -> CarbonFootprintOutput:
        """
        Calculate carbon footprint for fuel mix.

        Args:
            input_data: Carbon footprint calculation parameters

        Returns:
            Comprehensive carbon footprint analysis
        """
        self.calculation_count += 1

        total_co2 = 0.0
        total_ch4 = 0.0
        total_n2o = 0.0
        biogenic_carbon = 0.0
        fossil_carbon = 0.0
        total_energy_gj = 0.0
        scope3_upstream = 0.0
        breakdown = {}

        for fuel, qty_kg in input_data.fuel_quantities.items():
            if qty_kg <= 0:
                continue

            props = input_data.fuel_properties.get(fuel, {})
            heating_value_gj = props.get('heating_value_mj_kg', 30) * self.MJ_TO_GJ
            energy_gj = qty_kg * heating_value_gj
            total_energy_gj += energy_gj

            # Emission factors (kg or g per GJ)
            co2_factor = props.get('emission_factor_co2_kg_gj', 60)
            ch4_factor = props.get('emission_factor_ch4_g_gj', 1)
            n2o_factor = props.get('emission_factor_n2o_g_gj', 0.1)
            biogenic_pct = props.get('biogenic_carbon_percent', 0)

            # Calculate emissions
            co2_kg = co2_factor * energy_gj
            ch4_kg = ch4_factor * energy_gj / 1000  # g to kg
            n2o_kg = n2o_factor * energy_gj / 1000  # g to kg

            # Separate biogenic and fossil
            biogenic_kg = co2_kg * biogenic_pct / 100
            fossil_kg = co2_kg - biogenic_kg

            # Upstream emissions (scope 3)
            if input_data.include_upstream:
                upstream_factor = props.get('upstream_emission_factor_kg_gj', 5)
                scope3_upstream += upstream_factor * energy_gj

            # Accumulate totals
            total_co2 += co2_kg
            total_ch4 += ch4_kg
            total_n2o += n2o_kg
            biogenic_carbon += biogenic_kg
            fossil_carbon += fossil_kg

            breakdown[fuel] = {
                'quantity_kg': round(qty_kg, 2),
                'energy_gj': round(energy_gj, 2),
                'co2_kg': round(co2_kg, 2),
                'ch4_kg': round(ch4_kg, 4),
                'n2o_kg': round(n2o_kg, 4),
                'biogenic_kg': round(biogenic_kg, 2),
                'fossil_kg': round(fossil_kg, 2)
            }

        # Calculate CO2e
        total_co2e = (
            total_co2 * self.gwp['CO2'] +
            total_ch4 * self.gwp['CH4'] +
            total_n2o * self.gwp['N2O']
        )

        # Scope 1 = direct combustion emissions (fossil only)
        scope1 = fossil_carbon + total_ch4 * self.gwp['CH4'] + total_n2o * self.gwp['N2O']

        # Carbon intensity
        energy_mwh = total_energy_gj / 3.6
        carbon_intensity = total_co2e / energy_mwh if energy_mwh > 0 else 0

        # Carbon cost
        carbon_cost = total_co2e / 1000 * input_data.carbon_price_usd_per_tonne

        provenance_hash = self._calculate_provenance(input_data, total_co2e)

        return CarbonFootprintOutput(
            total_co2e_kg=round(total_co2e, 2),
            co2_kg=round(total_co2, 2),
            ch4_kg=round(total_ch4, 4),
            n2o_kg=round(total_n2o, 4),
            biogenic_carbon_kg=round(biogenic_carbon, 2),
            fossil_carbon_kg=round(fossil_carbon, 2),
            carbon_intensity_kg_mwh=round(carbon_intensity, 2),
            scope1_emissions_kg=round(scope1, 2),
            scope3_upstream_kg=round(scope3_upstream, 2),
            carbon_cost_usd=round(carbon_cost, 2),
            emission_breakdown=breakdown,
            provenance_hash=provenance_hash
        )

    def calculate_reduction_potential(
        self,
        current_mix: Dict[str, float],
        target_mix: Dict[str, float],
        properties: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate emission reduction potential from fuel switching.

        Args:
            current_mix: Current fuel quantities (kg)
            target_mix: Target fuel quantities (kg)
            properties: Fuel properties

        Returns:
            Reduction analysis
        """
        current_input = CarbonFootprintInput(
            fuel_quantities=current_mix,
            fuel_properties=properties
        )
        target_input = CarbonFootprintInput(
            fuel_quantities=target_mix,
            fuel_properties=properties
        )

        current_result = self.calculate(current_input)
        target_result = self.calculate(target_input)

        reduction_kg = current_result.total_co2e_kg - target_result.total_co2e_kg
        reduction_pct = (
            reduction_kg / current_result.total_co2e_kg * 100
            if current_result.total_co2e_kg > 0 else 0
        )

        return {
            'current_co2e_kg': current_result.total_co2e_kg,
            'target_co2e_kg': target_result.total_co2e_kg,
            'reduction_kg': round(reduction_kg, 2),
            'reduction_percent': round(reduction_pct, 2)
        }

    def _calculate_provenance(
        self,
        input_data: CarbonFootprintInput,
        total_co2e: float
    ) -> str:
        """Calculate provenance hash."""
        data = {
            'fuels': sorted(input_data.fuel_quantities.keys()),
            'quantities': input_data.fuel_quantities,
            'result': total_co2e
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
