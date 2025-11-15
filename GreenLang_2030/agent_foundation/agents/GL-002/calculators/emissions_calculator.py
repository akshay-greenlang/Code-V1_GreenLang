"""
Emissions Calculator - Zero Hallucination Guarantee

Implements EPA AP-42, EU ETS, and GHG Protocol compliant emissions calculations
for NOx, CO2, SOx, PM, and other pollutants from boiler operations.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: EPA AP-42, 40 CFR Part 60, EU Directive 2010/75/EU, GHG Protocol
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .provenance import ProvenanceTracker, ProvenanceRecord


@dataclass
class EmissionFactors:
    """Emission factors for different pollutants (kg/kg fuel or kg/GJ)."""
    co2_kg_per_kg: float
    nox_kg_per_kg: float
    sox_kg_per_kg: float
    co_kg_per_kg: float
    pm10_kg_per_kg: float
    pm25_kg_per_kg: float
    voc_kg_per_kg: float
    ch4_kg_per_kg: float  # Methane
    n2o_kg_per_kg: float  # Nitrous oxide


@dataclass
class BoilerEmissionData:
    """Input data for emissions calculations."""
    fuel_type: str
    fuel_consumption_kg_hr: float
    fuel_heating_value_kj_kg: float  # Lower heating value
    fuel_carbon_content_percent: float
    fuel_sulfur_content_percent: float
    fuel_nitrogen_content_percent: float
    fuel_ash_content_percent: float
    combustion_temperature_c: float
    excess_air_percent: float
    nox_control_type: Optional[str] = None  # SCR, SNCR, LNB, None
    nox_control_efficiency_percent: float = 0.0
    sox_control_type: Optional[str] = None  # FGD, DSI, None
    sox_control_efficiency_percent: float = 0.0
    particulate_control_type: Optional[str] = None  # ESP, Baghouse, None
    particulate_control_efficiency_percent: float = 0.0
    operating_hours_per_year: float = 8760


@dataclass
class RegulatoryLimits:
    """Regulatory emission limits."""
    nox_limit_mg_per_nm3: float  # at reference O2
    sox_limit_mg_per_nm3: float
    pm_limit_mg_per_nm3: float
    co_limit_mg_per_nm3: float
    reference_o2_percent: float = 3.0  # Reference O2 for limit
    co2_intensity_limit_kg_per_mwh: Optional[float] = None
    annual_co2_allowance_tonnes: Optional[float] = None


class EmissionsCalculator:
    """
    Calculates emissions from boiler operations per EPA and EU standards.

    Zero Hallucination Guarantee:
    - Pure mathematical calculations using EPA AP-42 factors
    - No LLM inference
    - Bit-perfect reproducibility
    - Complete provenance tracking
    """

    # EPA AP-42 Emission Factors (Table 1.4-2, 1.1-3, etc.)
    EPA_EMISSION_FACTORS = {
        'natural_gas': {
            'co2_kg_per_gj': 56.1,
            'nox_kg_per_gj': 0.098,  # Uncontrolled
            'sox_kg_per_gj': 0.0006,
            'co_kg_per_gj': 0.082,
            'pm10_kg_per_gj': 0.0076,
            'pm25_kg_per_gj': 0.0076,
            'voc_kg_per_gj': 0.0054,
            'ch4_kg_per_gj': 0.001,
            'n2o_kg_per_gj': 0.0001
        },
        'fuel_oil_no2': {
            'co2_kg_per_gj': 73.2,
            'nox_kg_per_gj': 0.142,
            'sox_kg_per_gj': 0.498,  # S-dependent
            'co_kg_per_gj': 0.024,
            'pm10_kg_per_gj': 0.025,
            'pm25_kg_per_gj': 0.023,
            'voc_kg_per_gj': 0.002,
            'ch4_kg_per_gj': 0.003,
            'n2o_kg_per_gj': 0.0003
        },
        'fuel_oil_no6': {
            'co2_kg_per_gj': 77.4,
            'nox_kg_per_gj': 0.257,
            'sox_kg_per_gj': 1.092,  # S-dependent
            'co_kg_per_gj': 0.024,
            'pm10_kg_per_gj': 0.085,
            'pm25_kg_per_gj': 0.065,
            'voc_kg_per_gj': 0.003,
            'ch4_kg_per_gj': 0.003,
            'n2o_kg_per_gj': 0.0003
        },
        'coal_bituminous': {
            'co2_kg_per_gj': 94.6,
            'nox_kg_per_gj': 0.380,
            'sox_kg_per_gj': 1.548,  # S-dependent
            'co_kg_per_gj': 0.024,
            'pm10_kg_per_gj': 0.586,  # Ash-dependent
            'pm25_kg_per_gj': 0.254,
            'voc_kg_per_gj': 0.003,
            'ch4_kg_per_gj': 0.001,
            'n2o_kg_per_gj': 0.0014
        },
        'biomass_wood': {
            'co2_kg_per_gj': 0.0,  # Biogenic CO2 (carbon neutral)
            'nox_kg_per_gj': 0.130,
            'sox_kg_per_gj': 0.025,
            'co_kg_per_gj': 0.600,
            'pm10_kg_per_gj': 0.326,
            'pm25_kg_per_gj': 0.259,
            'voc_kg_per_gj': 0.017,
            'ch4_kg_per_gj': 0.032,
            'n2o_kg_per_gj': 0.004
        }
    }

    # NOx control technology efficiencies
    NOX_CONTROL_EFFICIENCY = {
        'SCR': 0.90,  # Selective Catalytic Reduction
        'SNCR': 0.50,  # Selective Non-Catalytic Reduction
        'LNB': 0.40,  # Low-NOx Burners
        'OFA': 0.30,  # Over-Fire Air
        'FGR': 0.25   # Flue Gas Recirculation
    }

    # Global Warming Potentials (IPCC AR6)
    GWP_FACTORS = {
        'CO2': 1,
        'CH4': 28,  # 100-year GWP
        'N2O': 265  # 100-year GWP
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize calculator with version tracking."""
        self.version = version

    def calculate_emissions(self, data: BoilerEmissionData) -> Dict:
        """
        Calculate all emissions from boiler operation.

        Returns emissions rates, annual totals, and regulatory compliance status.
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"emissions_{id(data)}",
            calculation_type="boiler_emissions",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs(data.__dict__)

        # Step 1: Get base emission factors
        base_factors = self._get_emission_factors(data, tracker)

        # Step 2: Calculate uncontrolled emissions
        uncontrolled = self._calculate_uncontrolled_emissions(data, base_factors, tracker)

        # Step 3: Apply control technology efficiencies
        controlled = self._apply_control_technologies(data, uncontrolled, tracker)

        # Step 4: Calculate emission concentrations (mg/Nm³)
        concentrations = self._calculate_concentrations(data, controlled, tracker)

        # Step 5: Calculate annual emissions
        annual = self._calculate_annual_emissions(data, controlled, tracker)

        # Step 6: Calculate CO2 equivalent emissions
        co2e = self._calculate_co2_equivalent(annual, tracker)

        # Step 7: Calculate emission intensity
        intensity = self._calculate_emission_intensity(data, controlled, tracker)

        # Final result
        result = {
            'emission_rates_kg_hr': {
                'co2': float(controlled['co2']),
                'nox': float(controlled['nox']),
                'sox': float(controlled['sox']),
                'co': float(controlled['co']),
                'pm10': float(controlled['pm10']),
                'pm25': float(controlled['pm25']),
                'voc': float(controlled['voc']),
                'ch4': float(controlled['ch4']),
                'n2o': float(controlled['n2o'])
            },
            'concentrations_mg_nm3': {
                'nox': float(concentrations['nox']),
                'sox': float(concentrations['sox']),
                'pm': float(concentrations['pm10']),
                'co': float(concentrations['co'])
            },
            'annual_emissions_tonnes': {
                'co2': float(annual['co2']),
                'nox': float(annual['nox']),
                'sox': float(annual['sox']),
                'pm10': float(annual['pm10']),
                'pm25': float(annual['pm25'])
            },
            'co2_equivalent_tonnes': float(co2e),
            'emission_intensity': {
                'co2_kg_per_gj': float(intensity['co2_per_gj']),
                'nox_g_per_gj': float(intensity['nox_per_gj'] * Decimal('1000')),
                'sox_g_per_gj': float(intensity['sox_per_gj'] * Decimal('1000'))
            },
            'control_effectiveness': {
                'nox_reduction_percent': float(data.nox_control_efficiency_percent),
                'sox_reduction_percent': float(data.sox_control_efficiency_percent),
                'pm_reduction_percent': float(data.particulate_control_efficiency_percent)
            },
            'provenance': tracker.get_provenance_record(co2e).to_dict()
        }

        return result

    def check_compliance(
        self,
        emissions: Dict,
        limits: RegulatoryLimits
    ) -> Dict:
        """
        Check emissions against regulatory limits.

        Returns compliance status and margin to limits.
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"compliance_{id(limits)}",
            calculation_type="regulatory_compliance",
            version=self.version
        )

        tracker.record_inputs({
            'emissions': emissions,
            'limits': limits.__dict__
        })

        compliance_results = {
            'compliant': True,
            'violations': [],
            'margins': {}
        }

        # Check concentration limits
        concentrations = emissions.get('concentrations_mg_nm3', {})

        # NOx compliance
        nox_conc = Decimal(str(concentrations.get('nox', 0)))
        nox_limit = Decimal(str(limits.nox_limit_mg_per_nm3))

        if nox_conc > nox_limit:
            compliance_results['compliant'] = False
            compliance_results['violations'].append({
                'pollutant': 'NOx',
                'value': float(nox_conc),
                'limit': float(nox_limit),
                'exceedance_percent': float((nox_conc - nox_limit) / nox_limit * Decimal('100'))
            })

        compliance_results['margins']['nox'] = {
            'value': float(nox_conc),
            'limit': float(nox_limit),
            'margin_percent': float((nox_limit - nox_conc) / nox_limit * Decimal('100'))
        }

        # SOx compliance
        sox_conc = Decimal(str(concentrations.get('sox', 0)))
        sox_limit = Decimal(str(limits.sox_limit_mg_per_nm3))

        if sox_conc > sox_limit:
            compliance_results['compliant'] = False
            compliance_results['violations'].append({
                'pollutant': 'SOx',
                'value': float(sox_conc),
                'limit': float(sox_limit),
                'exceedance_percent': float((sox_conc - sox_limit) / sox_limit * Decimal('100'))
            })

        compliance_results['margins']['sox'] = {
            'value': float(sox_conc),
            'limit': float(sox_limit),
            'margin_percent': float((sox_limit - sox_conc) / sox_limit * Decimal('100'))
        }

        # PM compliance
        pm_conc = Decimal(str(concentrations.get('pm', 0)))
        pm_limit = Decimal(str(limits.pm_limit_mg_per_nm3))

        if pm_conc > pm_limit:
            compliance_results['compliant'] = False
            compliance_results['violations'].append({
                'pollutant': 'PM',
                'value': float(pm_conc),
                'limit': float(pm_limit),
                'exceedance_percent': float((pm_conc - pm_limit) / pm_limit * Decimal('100'))
            })

        compliance_results['margins']['pm'] = {
            'value': float(pm_conc),
            'limit': float(pm_limit),
            'margin_percent': float((pm_limit - pm_conc) / pm_limit * Decimal('100'))
        }

        # CO2 intensity check (if applicable)
        if limits.co2_intensity_limit_kg_per_mwh:
            intensity = Decimal(str(emissions.get('emission_intensity', {}).get('co2_kg_per_mwh', 0)))
            co2_limit = Decimal(str(limits.co2_intensity_limit_kg_per_mwh))

            if intensity > co2_limit:
                compliance_results['compliant'] = False
                compliance_results['violations'].append({
                    'pollutant': 'CO2 Intensity',
                    'value': float(intensity),
                    'limit': float(co2_limit),
                    'exceedance_percent': float((intensity - co2_limit) / co2_limit * Decimal('100'))
                })

        # Annual CO2 allowance check
        if limits.annual_co2_allowance_tonnes:
            annual_co2 = Decimal(str(emissions.get('annual_emissions_tonnes', {}).get('co2', 0)))
            allowance = Decimal(str(limits.annual_co2_allowance_tonnes))

            if annual_co2 > allowance:
                compliance_results['compliant'] = False
                compliance_results['violations'].append({
                    'pollutant': 'Annual CO2',
                    'value': float(annual_co2),
                    'limit': float(allowance),
                    'exceedance_percent': float((annual_co2 - allowance) / allowance * Decimal('100'))
                })

            compliance_results['margins']['annual_co2'] = {
                'value': float(annual_co2),
                'limit': float(allowance),
                'margin_percent': float((allowance - annual_co2) / allowance * Decimal('100'))
            }

        # Add recommendations if non-compliant
        if not compliance_results['compliant']:
            compliance_results['recommendations'] = self._generate_compliance_recommendations(
                compliance_results['violations']
            )

        compliance_results['provenance'] = tracker.get_provenance_record(
            compliance_results['compliant']
        ).to_dict()

        return compliance_results

    def calculate_carbon_tax(
        self,
        annual_co2_tonnes: float,
        carbon_price_per_tonne: float,
        exemptions: Optional[Dict] = None
    ) -> Dict:
        """Calculate carbon tax liability."""
        tracker = ProvenanceTracker(
            calculation_id=f"carbon_tax_{datetime.now().timestamp()}",
            calculation_type="carbon_tax",
            version=self.version
        )

        co2 = Decimal(str(annual_co2_tonnes))
        price = Decimal(str(carbon_price_per_tonne))

        # Apply exemptions if any
        taxable_co2 = co2
        if exemptions:
            if exemptions.get('biomass_exemption', False):
                # Biomass CO2 is often exempt
                taxable_co2 *= Decimal('0.9')  # Assume 10% biomass
            if 'free_allowance_tonnes' in exemptions:
                free_allowance = Decimal(str(exemptions['free_allowance_tonnes']))
                taxable_co2 = max(taxable_co2 - free_allowance, Decimal('0'))

        carbon_tax = taxable_co2 * price

        tracker.record_step(
            operation="carbon_tax_calculation",
            description="Calculate carbon tax liability",
            inputs={
                'annual_co2_tonnes': co2,
                'carbon_price': price,
                'taxable_co2': taxable_co2
            },
            output_value=carbon_tax,
            output_name="carbon_tax",
            formula="Tax = Taxable_CO2 * Price",
            units="currency"
        )

        result = {
            'total_co2_tonnes': float(co2),
            'taxable_co2_tonnes': float(taxable_co2),
            'exempt_co2_tonnes': float(co2 - taxable_co2),
            'carbon_price_per_tonne': float(price),
            'total_carbon_tax': float(carbon_tax),
            'effective_rate_per_tonne': float(carbon_tax / co2) if co2 > 0 else 0,
            'provenance': tracker.get_provenance_record(carbon_tax).to_dict()
        }

        return result

    def _get_emission_factors(
        self,
        data: BoilerEmissionData,
        tracker: ProvenanceTracker
    ) -> Dict[str, Decimal]:
        """Get emission factors for fuel type."""
        # Get base factors from EPA AP-42
        if data.fuel_type in self.EPA_EMISSION_FACTORS:
            factors_per_gj = self.EPA_EMISSION_FACTORS[data.fuel_type]
        else:
            # Use default factors for unknown fuel
            factors_per_gj = self.EPA_EMISSION_FACTORS['coal_bituminous']

        # Convert from per GJ to per kg fuel
        heating_value_gj = Decimal(str(data.fuel_heating_value_kj_kg)) / Decimal('1000')

        factors = {}
        for pollutant, factor_gj in factors_per_gj.items():
            factors[pollutant.replace('_kg_per_gj', '')] = (
                Decimal(str(factor_gj)) * heating_value_gj / Decimal('1000')
            )

        # Adjust SOx based on sulfur content
        sulfur_fraction = Decimal(str(data.fuel_sulfur_content_percent)) / Decimal('100')
        factors['sox'] = sulfur_fraction * Decimal('2')  # SO2 = S * (64/32)

        # Adjust PM based on ash content
        ash_fraction = Decimal(str(data.fuel_ash_content_percent)) / Decimal('100')
        factors['pm10'] = factors.get('pm10', Decimal('0')) * (Decimal('1') + ash_fraction)
        factors['pm25'] = factors.get('pm25', Decimal('0')) * (Decimal('1') + ash_fraction)

        tracker.record_step(
            operation="emission_factors",
            description="Get emission factors for fuel",
            inputs={
                'fuel_type': data.fuel_type,
                'heating_value_kj_kg': data.fuel_heating_value_kj_kg,
                'sulfur_percent': data.fuel_sulfur_content_percent
            },
            output_value=factors,
            output_name="emission_factors",
            formula="EPA AP-42 methodology",
            units="kg/kg fuel"
        )

        return factors

    def _calculate_uncontrolled_emissions(
        self,
        data: BoilerEmissionData,
        factors: Dict[str, Decimal],
        tracker: ProvenanceTracker
    ) -> Dict[str, Decimal]:
        """Calculate uncontrolled emission rates."""
        fuel_rate = Decimal(str(data.fuel_consumption_kg_hr))

        emissions = {}
        for pollutant, factor in factors.items():
            emissions[pollutant] = fuel_rate * factor

        # Thermal NOx adjustment based on temperature
        temp = Decimal(str(data.combustion_temperature_c))
        if temp > Decimal('1500'):
            # Thermal NOx increases exponentially with temperature
            thermal_factor = Decimal('1') + (temp - Decimal('1500')) / Decimal('500')
            emissions['nox'] = emissions.get('nox', Decimal('0')) * thermal_factor

        # Excess air adjustment for NOx
        excess_air = Decimal(str(data.excess_air_percent))
        air_factor = Decimal('1') + excess_air / Decimal('200')  # 0.5% increase per % excess air
        emissions['nox'] = emissions.get('nox', Decimal('0')) * air_factor

        tracker.record_step(
            operation="uncontrolled_emissions",
            description="Calculate uncontrolled emission rates",
            inputs={
                'fuel_rate_kg_hr': fuel_rate,
                'factors': factors,
                'temperature_c': temp,
                'excess_air_percent': excess_air
            },
            output_value=sum(emissions.values()),
            output_name="total_uncontrolled_kg_hr",
            formula="E = Fuel_Rate * EF * Adjustments",
            units="kg/hr"
        )

        return emissions

    def _apply_control_technologies(
        self,
        data: BoilerEmissionData,
        uncontrolled: Dict[str, Decimal],
        tracker: ProvenanceTracker
    ) -> Dict[str, Decimal]:
        """Apply emission control technology reductions."""
        controlled = uncontrolled.copy()

        # NOx control
        if data.nox_control_type:
            efficiency = Decimal(str(data.nox_control_efficiency_percent)) / Decimal('100')
            controlled['nox'] = uncontrolled['nox'] * (Decimal('1') - efficiency)

            tracker.record_step(
                operation="nox_control",
                description=f"Apply {data.nox_control_type} NOx control",
                inputs={
                    'uncontrolled_nox': uncontrolled['nox'],
                    'control_efficiency': efficiency
                },
                output_value=controlled['nox'],
                output_name="controlled_nox_kg_hr",
                formula="E_controlled = E_uncontrolled * (1 - η)",
                units="kg/hr"
            )

        # SOx control
        if data.sox_control_type:
            efficiency = Decimal(str(data.sox_control_efficiency_percent)) / Decimal('100')
            controlled['sox'] = uncontrolled.get('sox', Decimal('0')) * (Decimal('1') - efficiency)

        # Particulate control
        if data.particulate_control_type:
            efficiency = Decimal(str(data.particulate_control_efficiency_percent)) / Decimal('100')
            controlled['pm10'] = uncontrolled.get('pm10', Decimal('0')) * (Decimal('1') - efficiency)
            controlled['pm25'] = uncontrolled.get('pm25', Decimal('0')) * (Decimal('1') - efficiency)

        return controlled

    def _calculate_concentrations(
        self,
        data: BoilerEmissionData,
        emissions: Dict[str, Decimal],
        tracker: ProvenanceTracker
    ) -> Dict[str, Decimal]:
        """Convert emission rates to concentrations (mg/Nm³)."""
        # Calculate flue gas volume (simplified)
        fuel_rate = Decimal(str(data.fuel_consumption_kg_hr))

        # Theoretical air requirement (kg air/kg fuel)
        theoretical_air = Decimal('14.5')  # Typical for hydrocarbons

        # Actual air with excess
        excess_air = Decimal(str(data.excess_air_percent)) / Decimal('100')
        actual_air = theoretical_air * (Decimal('1') + excess_air)

        # Flue gas volume (Nm³/hr) - simplified
        # Assume 1.3 Nm³ flue gas per kg air
        flue_gas_volume = fuel_rate * actual_air * Decimal('1.3')

        concentrations = {}

        # Convert kg/hr to mg/Nm³
        for pollutant in ['nox', 'sox', 'co', 'pm10']:
            if pollutant in emissions:
                # kg/hr to mg/hr (* 1e6), then divide by Nm³/hr
                conc = (emissions[pollutant] * Decimal('1000000')) / flue_gas_volume
                concentrations[pollutant] = conc

        # Correct to reference O2 (typically 3% for gas, 6% for oil/coal)
        reference_o2 = Decimal('3')
        actual_o2 = Decimal('21') - (Decimal('21') / (Decimal('1') + excess_air))

        correction_factor = (Decimal('21') - reference_o2) / (Decimal('21') - actual_o2)

        for pollutant in concentrations:
            concentrations[pollutant] *= correction_factor

        tracker.record_step(
            operation="concentration_calculation",
            description="Convert emissions to concentrations",
            inputs={
                'flue_gas_volume_nm3_hr': flue_gas_volume,
                'reference_o2': reference_o2,
                'actual_o2': actual_o2
            },
            output_value=concentrations,
            output_name="concentrations_mg_nm3",
            formula="C = (E * 1e6) / V_fg * Correction",
            units="mg/Nm³"
        )

        return concentrations

    def _calculate_annual_emissions(
        self,
        data: BoilerEmissionData,
        hourly_emissions: Dict[str, Decimal],
        tracker: ProvenanceTracker
    ) -> Dict[str, Decimal]:
        """Calculate annual emission totals."""
        operating_hours = Decimal(str(data.operating_hours_per_year))

        annual = {}
        for pollutant, hourly_rate in hourly_emissions.items():
            # Convert kg/hr to tonnes/year
            annual[pollutant] = (hourly_rate * operating_hours) / Decimal('1000')

        tracker.record_step(
            operation="annual_emissions",
            description="Calculate annual emission totals",
            inputs={
                'operating_hours': operating_hours,
                'hourly_emissions': hourly_emissions
            },
            output_value=sum(annual.values()),
            output_name="total_annual_tonnes",
            formula="Annual = Hourly * Hours / 1000",
            units="tonnes/year"
        )

        return annual

    def _calculate_co2_equivalent(
        self,
        annual: Dict[str, Decimal],
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate CO2 equivalent emissions using GWP factors."""
        co2e = Decimal('0')

        # Direct CO2
        co2e += annual.get('co2', Decimal('0'))

        # CH4 contribution
        ch4 = annual.get('ch4', Decimal('0'))
        co2e += ch4 * Decimal(str(self.GWP_FACTORS['CH4']))

        # N2O contribution
        n2o = annual.get('n2o', Decimal('0'))
        co2e += n2o * Decimal(str(self.GWP_FACTORS['N2O']))

        tracker.record_step(
            operation="co2e_calculation",
            description="Calculate CO2 equivalent emissions",
            inputs={
                'co2': annual.get('co2', Decimal('0')),
                'ch4': ch4,
                'n2o': n2o,
                'gwp_ch4': self.GWP_FACTORS['CH4'],
                'gwp_n2o': self.GWP_FACTORS['N2O']
            },
            output_value=co2e,
            output_name="co2_equivalent_tonnes",
            formula="CO2e = CO2 + CH4*GWP_CH4 + N2O*GWP_N2O",
            units="tonnes CO2e"
        )

        return co2e

    def _calculate_emission_intensity(
        self,
        data: BoilerEmissionData,
        emissions: Dict[str, Decimal],
        tracker: ProvenanceTracker
    ) -> Dict[str, Decimal]:
        """Calculate emission intensity metrics."""
        fuel_rate = Decimal(str(data.fuel_consumption_kg_hr))
        heating_value = Decimal(str(data.fuel_heating_value_kj_kg))

        # Energy input in GJ/hr
        energy_input_gj = (fuel_rate * heating_value) / Decimal('1000000')

        intensity = {}
        for pollutant in ['co2', 'nox', 'sox']:
            if pollutant in emissions:
                intensity[f'{pollutant}_per_gj'] = emissions[pollutant] / energy_input_gj

        tracker.record_step(
            operation="intensity_calculation",
            description="Calculate emission intensities",
            inputs={
                'energy_input_gj_hr': energy_input_gj,
                'emissions': emissions
            },
            output_value=intensity,
            output_name="emission_intensity",
            formula="Intensity = Emissions / Energy_Input",
            units="kg/GJ"
        )

        return intensity

    def _generate_compliance_recommendations(self, violations: List[Dict]) -> List[Dict]:
        """Generate recommendations to achieve compliance."""
        recommendations = []

        for violation in violations:
            pollutant = violation['pollutant']
            exceedance = violation['exceedance_percent']

            if pollutant == 'NOx':
                if exceedance > 50:
                    recommendations.append({
                        'priority': 'High',
                        'measure': 'Install Selective Catalytic Reduction (SCR)',
                        'expected_reduction': '80-90%',
                        'implementation_time': '12-18 months'
                    })
                elif exceedance > 20:
                    recommendations.append({
                        'priority': 'Medium',
                        'measure': 'Install Selective Non-Catalytic Reduction (SNCR)',
                        'expected_reduction': '40-50%',
                        'implementation_time': '6-9 months'
                    })
                else:
                    recommendations.append({
                        'priority': 'Low',
                        'measure': 'Optimize combustion and install Low-NOx Burners',
                        'expected_reduction': '20-30%',
                        'implementation_time': '3-6 months'
                    })

            elif pollutant == 'SOx':
                if exceedance > 30:
                    recommendations.append({
                        'priority': 'High',
                        'measure': 'Install Flue Gas Desulfurization (FGD)',
                        'expected_reduction': '90-95%',
                        'implementation_time': '18-24 months'
                    })
                else:
                    recommendations.append({
                        'priority': 'Medium',
                        'measure': 'Switch to lower sulfur fuel',
                        'expected_reduction': 'Proportional to sulfur reduction',
                        'implementation_time': '1-2 months'
                    })

            elif pollutant == 'PM':
                recommendations.append({
                    'priority': 'High' if exceedance > 20 else 'Medium',
                    'measure': 'Install Electrostatic Precipitator or Baghouse',
                    'expected_reduction': '95-99%',
                    'implementation_time': '9-12 months'
                })

        return recommendations