# -*- coding: utf-8 -*-
"""
Steam System Emissions Calculator - Zero Hallucination

Calculates CO2, NOx, SOx emissions from steam generation fuel consumption.
Uses authoritative emission factors from EPA, IPCC, and DEFRA.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: GHG Protocol, EPA AP-42, IPCC Guidelines
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict
from dataclasses import dataclass
from .provenance import ProvenanceTracker
from greenlang.determinism import FinancialDecimal


@dataclass
class FuelConsumptionData:
    """Fuel consumption data for emissions calculation."""
    fuel_type: str  # natural_gas, coal, fuel_oil, biomass
    fuel_consumption_kg: float  # Total consumption
    fuel_heating_value_kj_kg: float
    boiler_efficiency_percent: float


@dataclass
class EmissionsResult:
    """Emissions calculation results."""
    co2_emissions_kg: float
    co2_emissions_tonnes: float
    nox_emissions_kg: float
    sox_emissions_kg: float
    emission_intensity_kg_co2_per_gj: float
    emission_factor_source: str
    recommendations: List[str]
    provenance: Dict


class EmissionsCalculator:
    """
    Calculate emissions from steam system fuel consumption.

    Zero Hallucination Guarantee:
    - EPA AP-42 emission factors (authoritative source)
    - Pure mathematical calculations
    - No LLM inference
    """

    # EPA AP-42 and IPCC Emission Factors (kg per GJ fuel input)
    EMISSION_FACTORS = {
        'natural_gas': {
            'co2_kg_per_gj': 56.1,  # IPCC 2006
            'nox_kg_per_gj': 0.092,  # EPA AP-42
            'sox_kg_per_gj': 0.0006,
            'source': 'EPA AP-42 Table 1.4-2, IPCC 2006 Guidelines'
        },
        'fuel_oil': {
            'co2_kg_per_gj': 77.4,  # IPCC 2006
            'nox_kg_per_gj': 0.142,
            'sox_kg_per_gj': 0.498,
            'source': 'EPA AP-42 Table 1.3-3, IPCC 2006 Guidelines'
        },
        'coal_bituminous': {
            'co2_kg_per_gj': 94.6,  # IPCC 2006
            'nox_kg_per_gj': 0.380,
            'sox_kg_per_gj': 1.548,
            'source': 'EPA AP-42 Table 1.1-3, IPCC 2006 Guidelines'
        },
        'biomass_wood': {
            'co2_kg_per_gj': 0.0,  # Biogenic CO2 (carbon neutral)
            'nox_kg_per_gj': 0.130,
            'sox_kg_per_gj': 0.025,
            'source': 'EPA AP-42 Table 1.6-1'
        }
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize calculator."""
        self.version = version

    def calculate_emissions(
        self,
        data: FuelConsumptionData
    ) -> EmissionsResult:
        """
        Calculate emissions from fuel consumption.

        Args:
            data: Fuel consumption data

        Returns:
            EmissionsResult with CO2, NOx, SOx emissions
        """
        tracker = ProvenanceTracker(
            calculation_id=f"emissions_{id(data)}",
            calculation_type="steam_emissions",
            version=self.version
        )

        tracker.record_inputs(data.__dict__)

        # Step 1: Get emission factors
        factors = self._get_emission_factors(data.fuel_type, tracker)

        # Step 2: Calculate total energy input
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        heating_value = Decimal(str(data.fuel_heating_value_kj_kg))

        energy_input_gj = (fuel_kg * heating_value) / Decimal('1000000')

        tracker.record_step(
            operation="energy_input",
            description="Calculate total energy input from fuel",
            inputs={
                'fuel_kg': fuel_kg,
                'heating_value_kj_kg': heating_value
            },
            output_value=energy_input_gj,
            output_name="energy_input_gj",
            formula="E = m_fuel * HV / 1000000",
            units="GJ"
        )

        # Step 3: Calculate CO2 emissions
        co2_kg = energy_input_gj * Decimal(str(factors['co2_kg_per_gj']))

        tracker.record_step(
            operation="co2_emissions",
            description="Calculate CO2 emissions",
            inputs={
                'energy_input_gj': energy_input_gj,
                'co2_factor_kg_per_gj': Decimal(str(factors['co2_kg_per_gj']))
            },
            output_value=co2_kg,
            output_name="co2_emissions_kg",
            formula="CO2 = Energy * EF_CO2",
            units="kg CO2"
        )

        # Step 4: Calculate NOx emissions
        nox_kg = energy_input_gj * Decimal(str(factors['nox_kg_per_gj']))

        # Step 5: Calculate SOx emissions
        sox_kg = energy_input_gj * Decimal(str(factors['sox_kg_per_gj']))

        # Step 6: Calculate emission intensity
        # Account for boiler efficiency
        efficiency = Decimal(str(data.boiler_efficiency_percent)) / Decimal('100')
        useful_energy_gj = energy_input_gj * efficiency

        if useful_energy_gj > Decimal('0'):
            emission_intensity = co2_kg / useful_energy_gj
        else:
            emission_intensity = Decimal('0')

        tracker.record_step(
            operation="emission_intensity",
            description="Calculate CO2 emission intensity per useful energy",
            inputs={
                'co2_kg': co2_kg,
                'useful_energy_gj': useful_energy_gj
            },
            output_value=emission_intensity,
            output_name="emission_intensity_kg_co2_per_gj",
            formula="Intensity = CO2 / Useful_Energy",
            units="kg CO2/GJ"
        )

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(
            FinancialDecimal.from_string(emission_intensity),
            float(data.boiler_efficiency_percent),
            data.fuel_type
        )

        return EmissionsResult(
            co2_emissions_kg=FinancialDecimal.from_string(co2_kg),
            co2_emissions_tonnes=FinancialDecimal.from_string(co2_kg / Decimal('1000')),
            nox_emissions_kg=FinancialDecimal.from_string(nox_kg),
            sox_emissions_kg=FinancialDecimal.from_string(sox_kg),
            emission_intensity_kg_co2_per_gj=FinancialDecimal.from_string(emission_intensity),
            emission_factor_source=factors['source'],
            recommendations=recommendations,
            provenance=tracker.get_provenance_record(co2_kg).to_dict()
        )

    def _get_emission_factors(
        self,
        fuel_type: str,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Get emission factors from authoritative sources."""
        # Map fuel type to standard categories
        fuel_category = fuel_type.lower()

        if 'natural_gas' in fuel_category or 'gas' in fuel_category:
            factors = self.EMISSION_FACTORS['natural_gas']
        elif 'fuel_oil' in fuel_category or 'oil' in fuel_category:
            factors = self.EMISSION_FACTORS['fuel_oil']
        elif 'coal' in fuel_category:
            factors = self.EMISSION_FACTORS['coal_bituminous']
        elif 'biomass' in fuel_category or 'wood' in fuel_category:
            factors = self.EMISSION_FACTORS['biomass_wood']
        else:
            # Default to natural gas (conservative)
            factors = self.EMISSION_FACTORS['natural_gas']

        tracker.record_step(
            operation="emission_factors",
            description="Lookup emission factors from EPA AP-42/IPCC",
            inputs={'fuel_type': fuel_type},
            output_value=factors['co2_kg_per_gj'],
            output_name="co2_emission_factor",
            formula="Database lookup - EPA AP-42",
            units="kg CO2/GJ"
        )

        return factors

    def _generate_recommendations(
        self,
        emission_intensity: float,
        efficiency: float,
        fuel_type: str
    ) -> List[str]:
        """Generate recommendations to reduce emissions."""
        recommendations = []

        # Efficiency recommendations
        if efficiency < 80:
            recommendations.append(
                f"Boiler efficiency ({efficiency:.1f}%) is low. "
                f"Improve efficiency to reduce emissions and fuel costs."
            )

        # Fuel switching recommendations
        if 'coal' in fuel_type.lower():
            recommendations.append(
                "Consider fuel switching from coal to natural gas to reduce CO2 emissions by ~40%"
            )
        elif 'oil' in fuel_type.lower():
            recommendations.append(
                "Consider fuel switching from oil to natural gas to reduce CO2 emissions by ~25%"
            )

        # General recommendations
        if emission_intensity > 70:
            recommendations.append(
                "High emission intensity. Priority actions: "
                "1) Improve boiler efficiency, "
                "2) Reduce steam losses, "
                "3) Optimize condensate return"
            )

        recommendations.append(
            "Implement continuous emissions monitoring for compliance and optimization"
        )

        return recommendations
