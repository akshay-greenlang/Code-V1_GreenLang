# -*- coding: utf-8 -*-
"""
Emissions Calculator for GL-005 CombustionControlAgent

Calculates combustion emissions (NOx, CO, CO2, SOx, PM) with regulatory compliance checking.
Zero-hallucination design using emission factors and combustion chemistry.

Reference Standards:
- EPA 40 CFR Part 60: Standards of Performance for New Stationary Sources
- EPA AP-42: Compilation of Air Pollutant Emission Factors
- EU Industrial Emissions Directive (IED) 2010/75/EU
- ISO 14064-1: Greenhouse Gases - Specification with guidance for quantification
- GHG Protocol: Corporate Standard for GHG Accounting

Mathematical Formulas:
- CO2 from Carbon: CO2 = C_mass * (44/12) kg CO2 per kg C
- Thermal NOx (Zeldovich): d[NO]/dt = k*[O2]^0.5*[N2]*exp(-E/RT)
- CO from incomplete combustion: CO = f(excess_air, temperature, mixing)
- Emission Concentration: C(mg/Nm³) = (mass_flow * 1e6) / (volume_flow * conditions)
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EmissionType(str, Enum):
    """Types of combustion emissions"""
    NOX = "nox"
    CO = "co"
    CO2 = "co2"
    SOX = "sox"
    PM = "pm"  # Particulate matter


class RegulatoryStandard(str, Enum):
    """Regulatory standards for compliance"""
    EPA_NSPS = "epa_nsps"  # EPA New Source Performance Standards
    EU_IED = "eu_ied"      # EU Industrial Emissions Directive
    NESHAP = "neshap"      # National Emission Standards for Hazardous Air Pollutants
    STATE_LOCAL = "state_local"


class ComplianceStatus(str, Enum):
    """Emission compliance status"""
    COMPLIANT = "compliant"
    NEAR_LIMIT = "near_limit"
    EXCEEDED = "exceeded"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class EmissionLimit:
    """Regulatory emission limit"""
    emission_type: EmissionType
    limit_value: float
    unit: str  # mg/Nm³, ppm, kg/hr, tCO2e/yr
    averaging_period: str  # hourly, daily, annual
    standard: RegulatoryStandard


@dataclass
class FuelProperties:
    """Fuel properties for emission calculations"""
    carbon_content_percent: float
    hydrogen_content_percent: float
    sulfur_content_percent: float
    nitrogen_content_percent: float
    ash_content_percent: float
    moisture_content_percent: float
    lower_heating_value_mj_per_kg: float


class EmissionsInput(BaseModel):
    """Input parameters for emissions calculation"""

    # Fuel properties
    fuel_type: str = Field(..., description="Type of fuel")
    fuel_flow_rate_kg_per_hr: float = Field(..., gt=0, description="Fuel flow rate")
    fuel_properties: Dict[str, float] = Field(
        ...,
        description="Fuel properties (C, H, S, N, ash, moisture percentages)"
    )
    fuel_heating_value_mj_per_kg: float = Field(..., gt=0)

    # Air flow
    air_flow_rate_kg_per_hr: float = Field(..., ge=0)

    # Combustion conditions
    combustion_temperature_c: float = Field(..., ge=0, le=2000)
    excess_air_percent: float = Field(..., ge=0, le=100)

    # Flue gas measurements
    flue_gas_o2_percent: float = Field(..., ge=0, le=21)
    flue_gas_co_ppm: float = Field(default=0, ge=0)
    flue_gas_nox_ppm: Optional[float] = Field(None, ge=0)
    flue_gas_temperature_c: float = Field(..., ge=0)

    # Operating conditions
    operating_hours_per_year: float = Field(
        default=8760,
        ge=0,
        le=8760,
        description="Annual operating hours"
    )

    # Regulatory requirements
    applicable_standards: List[str] = Field(
        default_factory=list,
        description="List of applicable regulatory standards"
    )
    emission_limits: Optional[Dict[str, float]] = Field(
        None,
        description="Emission limits by type"
    )


class EmissionsResult(BaseModel):
    """Emissions calculation results"""

    # Emission rates (mass basis)
    nox_kg_per_hr: float
    co_kg_per_hr: float
    co2_kg_per_hr: float
    sox_kg_per_hr: float
    pm_kg_per_hr: Optional[float] = None

    # Emission concentrations (volume basis, dry, reference O2)
    nox_mg_per_nm3: float = Field(..., description="NOx at reference O2 (3% for most combustion)")
    co_mg_per_nm3: float
    co2_percent: float
    sox_mg_per_nm3: float

    # Annual emissions
    nox_tonnes_per_year: float
    co_tonnes_per_year: float
    co2_tonnes_per_year: float
    co2e_tonnes_per_year: float  # CO2 equivalent

    # Emission factors
    nox_emission_factor_kg_per_gj: float
    co2_emission_factor_kg_per_gj: float

    # Compliance status
    compliance_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Compliance status by emission type"
    )
    overall_compliance: ComplianceStatus

    # Performance metrics
    specific_co2_kg_per_kwh: float = Field(
        ...,
        description="Specific CO2 emissions per kWh output"
    )
    emission_intensity: float = Field(
        ...,
        description="Emission intensity (kg CO2e per unit output)"
    )

    # Regulatory reporting
    requires_reporting: bool
    exceeds_any_limit: bool
    violations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class EmissionsCalculator:
    """
    Calculate combustion emissions using emission factors and combustion chemistry.

    This calculator uses deterministic emission factor methods approved by EPA and EU:

    1. Carbon Balance Method (for CO2)
    2. Emission Factor Method (AP-42)
    3. Continuous Emission Monitoring (when available)

    All calculations are based on:
        - Complete combustion chemistry
        - EPA AP-42 emission factors
        - IPCC emission factors
        - Measured concentrations (when available)

    Reference O2 for concentration reporting:
        - Boilers/furnaces: 3% O2
        - Gas turbines: 15% O2
        - Diesel engines: 15% O2
    """

    # Molecular weights (kg/kmol)
    MW = {
        'C': 12.011,
        'H': 1.008,
        'O': 15.999,
        'N': 14.007,
        'S': 32.065,
        'O2': 31.998,
        'N2': 28.014,
        'CO2': 44.01,
        'H2O': 18.015,
        'SO2': 64.064,
        'NO': 30.006,
        'NO2': 46.006,
        'CO': 28.01
    }

    # Reference O2 levels by equipment type
    REFERENCE_O2 = {
        'boiler': 3.0,
        'furnace': 3.0,
        'gas_turbine': 15.0,
        'engine': 15.0
    }

    # Default emission factors (kg/GJ fuel input) - EPA AP-42
    DEFAULT_EMISSION_FACTORS = {
        'natural_gas': {
            'nox': 0.09,  # kg NOx/GJ (uncontrolled)
            'co': 0.04,   # kg CO/GJ
            'pm': 0.007,  # kg PM/GJ
        },
        'fuel_oil': {
            'nox': 0.14,
            'co': 0.005,
            'pm': 0.024,
        },
        'diesel': {
            'nox': 0.20,
            'co': 0.006,
            'pm': 0.016,
        },
        'coal': {
            'nox': 0.30,
            'co': 0.020,
            'pm': 0.300,
        }
    }

    # GWP (Global Warming Potential) values - IPCC AR6
    GWP = {
        'CO2': 1.0,
        'CH4': 29.8,  # 100-year GWP
        'N2O': 273,   # 100-year GWP
    }

    def __init__(self):
        """Initialize emissions calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate_nox_emissions(
        self,
        emissions_input: EmissionsInput,
        use_measurement: bool = False
    ) -> Tuple[float, float]:
        """
        Calculate NOx emissions.

        NOx formation mechanisms:
        1. Thermal NOx (Zeldovich mechanism) - high temperature
        2. Fuel NOx - nitrogen in fuel
        3. Prompt NOx - hydrocarbon radicals

        Args:
            emissions_input: Emission calculation inputs
            use_measurement: Use measured NOx if available

        Returns:
            Tuple of (nox_kg_per_hr, nox_mg_per_nm3)
        """
        # If measurement available, use it
        if use_measurement and emissions_input.flue_gas_nox_ppm is not None:
            return self._calculate_nox_from_measurement(emissions_input)

        # Otherwise, use emission factor method
        fuel_type = emissions_input.fuel_type.lower()
        fuel_input_gj_per_hr = (
            emissions_input.fuel_flow_rate_kg_per_hr *
            emissions_input.fuel_heating_value_mj_per_kg / 1000
        )

        # Get base emission factor
        base_ef = self.DEFAULT_EMISSION_FACTORS.get(fuel_type, {}).get('nox', 0.15)

        # Adjust for excess air (more O2 → more thermal NOx)
        excess_air_factor = 1 + (emissions_input.excess_air_percent / 100) * 0.2

        # Adjust for temperature (exponential relationship)
        temp_factor = math.exp((emissions_input.combustion_temperature_c - 1000) / 500)

        # Adjusted emission factor
        adjusted_ef = base_ef * excess_air_factor * temp_factor

        # NOx mass flow rate (kg/hr)
        nox_kg_per_hr = adjusted_ef * fuel_input_gj_per_hr

        # Convert to concentration (mg/Nm³ at reference O2)
        flue_gas_volume = self._calculate_flue_gas_volume(emissions_input)
        nox_mg_per_nm3 = (nox_kg_per_hr * 1e6) / flue_gas_volume

        # Correct to reference O2
        nox_mg_per_nm3_corrected = self._correct_to_reference_o2(
            nox_mg_per_nm3,
            emissions_input.flue_gas_o2_percent,
            3.0  # Reference O2 for combustion sources
        )

        return nox_kg_per_hr, nox_mg_per_nm3_corrected

    def calculate_co_emissions(
        self,
        emissions_input: EmissionsInput
    ) -> Tuple[float, float]:
        """
        Calculate CO emissions from incomplete combustion.

        CO formation depends on:
        - Excess air (low air → high CO)
        - Temperature (low temp → high CO)
        - Mixing quality

        Args:
            emissions_input: Emission calculation inputs

        Returns:
            Tuple of (co_kg_per_hr, co_mg_per_nm3)
        """
        # Use measured CO if available
        if emissions_input.flue_gas_co_ppm > 0:
            co_ppm = emissions_input.flue_gas_co_ppm
        else:
            # Estimate CO from excess air and temperature
            co_ppm = self._estimate_co_from_conditions(
                emissions_input.excess_air_percent,
                emissions_input.combustion_temperature_c
            )

        # Convert ppm to mg/Nm³
        # At STP: ppm * MW / 22.4 = mg/Nm³
        co_mg_per_nm3 = co_ppm * (self.MW['CO'] / 22.4)

        # Calculate mass flow rate
        flue_gas_volume = self._calculate_flue_gas_volume(emissions_input)
        co_kg_per_hr = (co_mg_per_nm3 * flue_gas_volume) / 1e6

        return co_kg_per_hr, co_mg_per_nm3

    def calculate_co2_emissions(
        self,
        emissions_input: EmissionsInput
    ) -> Tuple[float, float]:
        """
        Calculate CO2 emissions using carbon balance method.

        This is the most accurate method for CO2, based on fuel carbon content.

        Formula:
            CO2 = C_in_fuel * (44/12) kg CO2 per kg C

        Args:
            emissions_input: Emission calculation inputs

        Returns:
            Tuple of (co2_kg_per_hr, co2_percent_in_flue_gas)
        """
        # Extract carbon content
        carbon_percent = emissions_input.fuel_properties.get('C', 85) / 100

        # CO2 from complete combustion (kg/hr)
        carbon_flow = emissions_input.fuel_flow_rate_kg_per_hr * carbon_percent
        co2_kg_per_hr = carbon_flow * (self.MW['CO2'] / self.MW['C'])

        # CO2 percentage in flue gas
        flue_gas_mass = (
            emissions_input.fuel_flow_rate_kg_per_hr +
            emissions_input.air_flow_rate_kg_per_hr
        )
        co2_percent = (co2_kg_per_hr / flue_gas_mass * 100) if flue_gas_mass > 0 else 0

        return co2_kg_per_hr, co2_percent

    def check_regulatory_compliance(
        self,
        emissions_result: EmissionsResult,
        emission_limits: Dict[str, float],
        applicable_standards: List[str]
    ) -> Dict[str, ComplianceStatus]:
        """
        Check emissions against regulatory limits.

        Args:
            emissions_result: Calculated emissions
            emission_limits: Dictionary of limits by emission type
            applicable_standards: List of applicable standards

        Returns:
            Dictionary of compliance status by emission type
        """
        compliance = {}

        # Check NOx
        if 'nox_mg_per_nm3' in emission_limits:
            limit = emission_limits['nox_mg_per_nm3']
            actual = emissions_result.nox_mg_per_nm3

            if actual <= limit * 0.9:
                compliance['nox'] = ComplianceStatus.COMPLIANT
            elif actual <= limit:
                compliance['nox'] = ComplianceStatus.NEAR_LIMIT
            else:
                compliance['nox'] = ComplianceStatus.EXCEEDED

        # Check CO
        if 'co_mg_per_nm3' in emission_limits:
            limit = emission_limits['co_mg_per_nm3']
            actual = emissions_result.co_mg_per_nm3

            if actual <= limit * 0.9:
                compliance['co'] = ComplianceStatus.COMPLIANT
            elif actual <= limit:
                compliance['co'] = ComplianceStatus.NEAR_LIMIT
            else:
                compliance['co'] = ComplianceStatus.EXCEEDED

        # Check CO2 (if limit exists)
        if 'co2_tonnes_per_year' in emission_limits:
            limit = emission_limits['co2_tonnes_per_year']
            actual = emissions_result.co2_tonnes_per_year

            if actual <= limit * 0.9:
                compliance['co2'] = ComplianceStatus.COMPLIANT
            elif actual <= limit:
                compliance['co2'] = ComplianceStatus.NEAR_LIMIT
            else:
                compliance['co2'] = ComplianceStatus.EXCEEDED

        return compliance

    def calculate_all_emissions(
        self,
        emissions_input: EmissionsInput
    ) -> EmissionsResult:
        """
        Calculate all combustion emissions and check compliance.

        Args:
            emissions_input: Emission calculation inputs

        Returns:
            EmissionsResult with all emission calculations
        """
        self.logger.info("Calculating combustion emissions")

        # Calculate individual emissions
        nox_kg_hr, nox_mg_nm3 = self.calculate_nox_emissions(emissions_input)
        co_kg_hr, co_mg_nm3 = self.calculate_co_emissions(emissions_input)
        co2_kg_hr, co2_percent = self.calculate_co2_emissions(emissions_input)

        # Calculate SOx (from fuel sulfur)
        sox_kg_hr, sox_mg_nm3 = self._calculate_sox_emissions(emissions_input)

        # Calculate PM (if applicable)
        pm_kg_hr = self._calculate_pm_emissions(emissions_input)

        # Annual emissions
        hours_per_year = emissions_input.operating_hours_per_year
        nox_tonnes_yr = nox_kg_hr * hours_per_year / 1000
        co_tonnes_yr = co_kg_hr * hours_per_year / 1000
        co2_tonnes_yr = co2_kg_hr * hours_per_year / 1000

        # CO2 equivalent (including other GHGs if applicable)
        co2e_tonnes_yr = co2_tonnes_yr  # Simplified, could include CH4, N2O

        # Emission factors
        fuel_input_gj_hr = (
            emissions_input.fuel_flow_rate_kg_per_hr *
            emissions_input.fuel_heating_value_mj_per_kg / 1000
        )
        nox_ef = nox_kg_hr / fuel_input_gj_hr if fuel_input_gj_hr > 0 else 0
        co2_ef = co2_kg_hr / fuel_input_gj_hr if fuel_input_gj_hr > 0 else 0

        # Specific CO2 (kg/kWh)
        # Assuming 85% efficiency
        heat_output_kw = fuel_input_gj_hr * 1000 / 3.6 * 0.85
        specific_co2 = co2_kg_hr / heat_output_kw if heat_output_kw > 0 else 0

        # Emission intensity
        emission_intensity = specific_co2  # Simplified

        # Check compliance
        compliance_status = {}
        overall_compliance = ComplianceStatus.COMPLIANT
        exceeds_any_limit = False
        violations = []

        if emissions_input.emission_limits:
            compliance_status = self.check_regulatory_compliance(
                EmissionsResult(
                    nox_kg_per_hr=nox_kg_hr,
                    co_kg_per_hr=co_kg_hr,
                    co2_kg_per_hr=co2_kg_hr,
                    sox_kg_per_hr=sox_kg_hr,
                    pm_kg_per_hr=pm_kg_hr,
                    nox_mg_per_nm3=nox_mg_nm3,
                    co_mg_per_nm3=co_mg_nm3,
                    co2_percent=co2_percent,
                    sox_mg_per_nm3=sox_mg_nm3,
                    nox_tonnes_per_year=nox_tonnes_yr,
                    co_tonnes_per_year=co_tonnes_yr,
                    co2_tonnes_per_year=co2_tonnes_yr,
                    co2e_tonnes_per_year=co2e_tonnes_yr,
                    nox_emission_factor_kg_per_gj=nox_ef,
                    co2_emission_factor_kg_per_gj=co2_ef,
                    specific_co2_kg_per_kwh=specific_co2,
                    emission_intensity=emission_intensity,
                    compliance_status={},
                    overall_compliance=ComplianceStatus.COMPLIANT,
                    requires_reporting=False,
                    exceeds_any_limit=False
                ),
                emissions_input.emission_limits,
                emissions_input.applicable_standards
            )

            # Check for violations
            if ComplianceStatus.EXCEEDED in compliance_status.values():
                exceeds_any_limit = True
                overall_compliance = ComplianceStatus.EXCEEDED
                violations = [
                    f"{k.upper()} exceeds limit"
                    for k, v in compliance_status.items()
                    if v == ComplianceStatus.EXCEEDED
                ]
            elif ComplianceStatus.NEAR_LIMIT in compliance_status.values():
                overall_compliance = ComplianceStatus.NEAR_LIMIT

        # Reporting requirement (simplified - typically based on thresholds)
        requires_reporting = co2_tonnes_yr > 25000  # EU ETS threshold

        # Generate recommendations
        recommendations = self._generate_emission_recommendations(
            nox_mg_nm3, co_mg_nm3, co2_tonnes_yr, compliance_status
        )

        return EmissionsResult(
            nox_kg_per_hr=self._round_decimal(nox_kg_hr, 4),
            co_kg_per_hr=self._round_decimal(co_kg_hr, 4),
            co2_kg_per_hr=self._round_decimal(co2_kg_hr, 2),
            sox_kg_per_hr=self._round_decimal(sox_kg_hr, 4),
            pm_kg_per_hr=self._round_decimal(pm_kg_hr, 4) if pm_kg_hr else None,
            nox_mg_per_nm3=self._round_decimal(nox_mg_nm3, 2),
            co_mg_per_nm3=self._round_decimal(co_mg_nm3, 2),
            co2_percent=self._round_decimal(co2_percent, 2),
            sox_mg_per_nm3=self._round_decimal(sox_mg_nm3, 2),
            nox_tonnes_per_year=self._round_decimal(nox_tonnes_yr, 3),
            co_tonnes_per_year=self._round_decimal(co_tonnes_yr, 3),
            co2_tonnes_per_year=self._round_decimal(co2_tonnes_yr, 2),
            co2e_tonnes_per_year=self._round_decimal(co2e_tonnes_yr, 2),
            nox_emission_factor_kg_per_gj=self._round_decimal(nox_ef, 4),
            co2_emission_factor_kg_per_gj=self._round_decimal(co2_ef, 3),
            compliance_status={k: v.value for k, v in compliance_status.items()},
            overall_compliance=overall_compliance,
            specific_co2_kg_per_kwh=self._round_decimal(specific_co2, 4),
            emission_intensity=self._round_decimal(emission_intensity, 4),
            requires_reporting=requires_reporting,
            exceeds_any_limit=exceeds_any_limit,
            violations=violations,
            recommendations=recommendations
        )

    def _calculate_nox_from_measurement(
        self,
        emissions_input: EmissionsInput
    ) -> Tuple[float, float]:
        """Calculate NOx from measured concentration"""
        nox_ppm = emissions_input.flue_gas_nox_ppm
        nox_mg_per_nm3 = nox_ppm * (self.MW['NO2'] / 22.4)  # Assume NO2 equivalent

        flue_gas_volume = self._calculate_flue_gas_volume(emissions_input)
        nox_kg_per_hr = (nox_mg_per_nm3 * flue_gas_volume) / 1e6

        return nox_kg_per_hr, nox_mg_per_nm3

    def _calculate_sox_emissions(
        self,
        emissions_input: EmissionsInput
    ) -> Tuple[float, float]:
        """
        Calculate SOx emissions from fuel sulfur.

        Formula:
            SO2 = S_in_fuel * (64/32) kg SO2 per kg S
        """
        sulfur_percent = emissions_input.fuel_properties.get('S', 0) / 100
        sulfur_flow = emissions_input.fuel_flow_rate_kg_per_hr * sulfur_percent

        # Assume 100% conversion to SO2
        sox_kg_per_hr = sulfur_flow * (self.MW['SO2'] / self.MW['S'])

        # Convert to concentration
        flue_gas_volume = self._calculate_flue_gas_volume(emissions_input)
        sox_mg_per_nm3 = (sox_kg_per_hr * 1e6) / flue_gas_volume if flue_gas_volume > 0 else 0

        return sox_kg_per_hr, sox_mg_per_nm3

    def _calculate_pm_emissions(
        self,
        emissions_input: EmissionsInput
    ) -> float:
        """Calculate particulate matter emissions"""
        fuel_type = emissions_input.fuel_type.lower()
        fuel_input_gj_per_hr = (
            emissions_input.fuel_flow_rate_kg_per_hr *
            emissions_input.fuel_heating_value_mj_per_kg / 1000
        )

        # Get base emission factor
        pm_ef = self.DEFAULT_EMISSION_FACTORS.get(fuel_type, {}).get('pm', 0.01)

        # PM depends on ash content
        ash_percent = emissions_input.fuel_properties.get('ash', 0)
        ash_factor = 1 + (ash_percent / 10)  # Higher ash → more PM

        pm_kg_per_hr = pm_ef * fuel_input_gj_per_hr * ash_factor

        return pm_kg_per_hr

    def _calculate_flue_gas_volume(
        self,
        emissions_input: EmissionsInput
    ) -> float:
        """
        Calculate flue gas volume flow rate (Nm³/hr at STP).

        Simplified: Flue gas mass → volume using density
        Density at STP ≈ 1.3 kg/Nm³
        """
        flue_gas_mass = (
            emissions_input.fuel_flow_rate_kg_per_hr +
            emissions_input.air_flow_rate_kg_per_hr
        )

        # Convert to Nm³/hr (density ≈ 1.3 kg/Nm³)
        flue_gas_volume = flue_gas_mass / 1.3

        return flue_gas_volume

    def _correct_to_reference_o2(
        self,
        concentration: float,
        measured_o2: float,
        reference_o2: float
    ) -> float:
        """
        Correct emission concentration to reference O2 level.

        Formula:
            C_ref = C_measured * (21 - O2_ref) / (21 - O2_measured)
        """
        if measured_o2 >= 21:
            return concentration

        correction_factor = (21 - reference_o2) / (21 - measured_o2)
        return concentration * correction_factor

    def _estimate_co_from_conditions(
        self,
        excess_air_percent: float,
        temperature_c: float
    ) -> float:
        """
        Estimate CO concentration from combustion conditions.

        CO increases with:
        - Low excess air (<10%)
        - Low temperature (<800°C)
        - Poor mixing
        """
        # Base CO (good conditions)
        base_co = 50  # ppm

        # Excess air factor (exponential increase at low excess air)
        if excess_air_percent < 10:
            ea_factor = math.exp((10 - excess_air_percent) / 5)
        else:
            ea_factor = 1.0

        # Temperature factor
        if temperature_c < 800:
            temp_factor = math.exp((800 - temperature_c) / 200)
        else:
            temp_factor = 1.0

        co_ppm = base_co * ea_factor * temp_factor

        return min(co_ppm, 10000)  # Cap at 10000 ppm

    def _generate_emission_recommendations(
        self,
        nox_mg_nm3: float,
        co_mg_nm3: float,
        co2_tonnes_yr: float,
        compliance_status: Dict[str, ComplianceStatus]
    ) -> List[str]:
        """Generate recommendations for emission control"""
        recommendations = []

        # NOx recommendations
        if nox_mg_nm3 > 200:
            recommendations.append(
                "High NOx emissions - consider low-NOx burner or SCR/SNCR"
            )

        # CO recommendations
        if co_mg_nm3 > 100:
            recommendations.append(
                "High CO emissions - improve combustion air mixing or increase excess air"
            )

        # CO2 recommendations
        if co2_tonnes_yr > 25000:
            recommendations.append(
                "CO2 emissions exceed EU ETS threshold - reporting required"
            )

        # Compliance recommendations
        for emission_type, status in compliance_status.items():
            if status == ComplianceStatus.EXCEEDED:
                recommendations.append(
                    f"CRITICAL: {emission_type.upper()} exceeds regulatory limit - immediate action required"
                )
            elif status == ComplianceStatus.NEAR_LIMIT:
                recommendations.append(
                    f"WARNING: {emission_type.upper()} approaching limit - monitor closely"
                )

        if not recommendations:
            recommendations.append("Emissions within acceptable limits")

        return recommendations

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        if value is None:
            return None
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
