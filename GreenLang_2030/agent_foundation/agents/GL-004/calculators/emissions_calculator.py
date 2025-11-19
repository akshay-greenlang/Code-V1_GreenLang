"""
Emissions Calculator for GL-004 BurnerOptimizationAgent

Calculates combustion emissions (NOx, CO, CO2, SOx, PM) with regulatory compliance checking.
Zero-hallucination design using emission correlations and combustion chemistry.

Reference Standards:
- EPA 40 CFR Part 60: Standards of Performance for New Stationary Sources
- EPA AP-42: Compilation of Air Pollutant Emission Factors
- EU Industrial Emissions Directive (IED) 2010/75/EU
- Zeldovich Mechanism for Thermal NOx Formation
- ASME Research Committee on Industrial and Municipal Wastes

Mathematical Models:
- Thermal NOx: Extended Zeldovich Mechanism
- Prompt NOx: Fenimore Mechanism
- Fuel NOx: Conversion of fuel-bound nitrogen
- CO Formation: Chemical equilibrium and quenching
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EmissionType(str, Enum):
    """Types of combustion emissions"""
    NOX = "nox"
    CO = "co"
    CO2 = "co2"
    SOX = "sox"
    PM = "pm"
    VOC = "voc"


class RegulatoryStandard(str, Enum):
    """Regulatory standards for compliance"""
    EPA_NSPS = "epa_nsps"  # EPA New Source Performance Standards
    EU_IED = "eu_ied"      # EU Industrial Emissions Directive
    EPA_NESHAP = "epa_neshap"  # National Emission Standards for Hazardous Air Pollutants
    STATE_IMPLEMENTATION = "state_sip"  # State Implementation Plans


class ComplianceLevel(str, Enum):
    """Emission compliance levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"  # 80-90% of limit
    CRITICAL = "critical"  # 90-100% of limit
    VIOLATION = "violation"  # Exceeds limit


@dataclass
class EmissionLimit:
    """Regulatory emission limit definition"""
    pollutant: EmissionType
    limit_value: float
    units: str  # mg/Nm³, ppm, lb/MMBtu, g/GJ
    o2_reference: float  # Reference O2 level (%)
    averaging_time: str  # 1-hour, 3-hour, daily, annual
    standard: RegulatoryStandard


class EmissionsInput(BaseModel):
    """Input parameters for emissions calculation"""

    # Fuel composition (weight %)
    fuel_carbon: float = Field(..., ge=0, le=100, description="Carbon content %")
    fuel_hydrogen: float = Field(..., ge=0, le=100, description="Hydrogen content %")
    fuel_oxygen: float = Field(0, ge=0, le=100, description="Oxygen content %")
    fuel_nitrogen: float = Field(0, ge=0, le=100, description="Nitrogen content %")
    fuel_sulfur: float = Field(0, ge=0, le=10, description="Sulfur content %")
    fuel_moisture: float = Field(0, ge=0, le=50, description="Moisture content %")
    fuel_ash: float = Field(0, ge=0, le=50, description="Ash content %")

    # Fuel properties
    fuel_flow_kg_hr: float = Field(..., gt=0, description="Fuel flow rate kg/hr")
    fuel_hhv_mj_kg: float = Field(..., gt=0, description="Higher heating value MJ/kg")
    fuel_lhv_mj_kg: Optional[float] = Field(None, gt=0, description="Lower heating value MJ/kg")

    # Combustion conditions
    combustion_temp_c: float = Field(..., ge=500, le=2500, description="Combustion temperature °C")
    residence_time_s: float = Field(2.0, gt=0, le=10, description="Residence time seconds")
    excess_air_percent: float = Field(..., ge=-10, le=200, description="Excess air %")

    # Measured values
    flue_gas_o2_percent: float = Field(..., ge=0, le=21, description="O2 in flue gas %")
    flue_gas_co_ppm: Optional[float] = Field(None, ge=0, description="CO in flue gas ppm")
    flue_gas_nox_ppm: Optional[float] = Field(None, ge=0, description="NOx in flue gas ppm")
    flue_gas_temp_c: float = Field(..., ge=50, le=1000, description="Flue gas temperature °C")
    ambient_temp_c: float = Field(25, ge=-40, le=50, description="Ambient temperature °C")

    # Control technologies
    low_nox_burner: bool = Field(False, description="Low NOx burner installed")
    flue_gas_recirculation: bool = Field(False, description="FGR active")
    staged_combustion: bool = Field(False, description="Staged combustion active")
    scr_efficiency: Optional[float] = Field(None, ge=0, le=100, description="SCR efficiency %")

    # Operating parameters
    load_percent: float = Field(100, gt=0, le=110, description="Load %")
    operating_hours_year: float = Field(8000, ge=0, le=8760, description="Annual hours")

    @validator('fuel_lhv_mj_kg', always=True)
    def calculate_lhv(cls, v, values):
        """Calculate LHV if not provided"""
        if v is None and 'fuel_hhv_mj_kg' in values and 'fuel_hydrogen' in values:
            # LHV = HHV - 2.442 * 9 * H/100 (MJ/kg)
            hhv = values['fuel_hhv_mj_kg']
            h_fraction = values['fuel_hydrogen'] / 100
            v = hhv - 2.442 * 9 * h_fraction
        return v


class EmissionsOutput(BaseModel):
    """Output from emissions calculation"""

    # Mass emissions (kg/hr)
    nox_kg_hr: float = Field(..., description="NOx emissions kg/hr as NO2")
    co_kg_hr: float = Field(..., description="CO emissions kg/hr")
    co2_kg_hr: float = Field(..., description="CO2 emissions kg/hr")
    sox_kg_hr: float = Field(..., description="SOx emissions kg/hr as SO2")
    pm_kg_hr: float = Field(..., description="PM emissions kg/hr")
    voc_kg_hr: float = Field(..., description="VOC emissions kg/hr")

    # Concentrations (mg/Nm³ @ reference O2)
    nox_mg_nm3: float = Field(..., description="NOx concentration mg/Nm³")
    co_mg_nm3: float = Field(..., description="CO concentration mg/Nm³")
    sox_mg_nm3: float = Field(..., description="SOx concentration mg/Nm³")
    pm_mg_nm3: float = Field(..., description="PM concentration mg/Nm³")

    # Concentrations (ppm dry @ reference O2)
    nox_ppm_dry: float = Field(..., description="NOx ppm dry")
    co_ppm_dry: float = Field(..., description="CO ppm dry")
    o2_percent_dry: float = Field(..., description="O2 % dry")
    co2_percent_dry: float = Field(..., description="CO2 % dry")

    # Annual emissions (tonnes/year)
    nox_tonnes_year: float = Field(..., description="Annual NOx tonnes")
    co2_tonnes_year: float = Field(..., description="Annual CO2 tonnes")
    co2e_tonnes_year: float = Field(..., description="Annual CO2e tonnes")

    # Emission factors
    nox_g_gj: float = Field(..., description="NOx emission factor g/GJ")
    co_g_gj: float = Field(..., description="CO emission factor g/GJ")
    co2_kg_gj: float = Field(..., description="CO2 emission factor kg/GJ")

    # NOx components
    thermal_nox_ppm: float = Field(..., description="Thermal NOx ppm")
    prompt_nox_ppm: float = Field(..., description="Prompt NOx ppm")
    fuel_nox_ppm: float = Field(..., description="Fuel NOx ppm")

    # Compliance
    compliance_status: Dict[str, str] = Field(..., description="Compliance by pollutant")
    violations: List[str] = Field(default_factory=list, description="Limit violations")
    margin_to_limit: Dict[str, float] = Field(..., description="% margin to limit")

    # Provenance
    calculation_timestamp: str = Field(..., description="ISO timestamp")
    provenance_hash: str = Field(..., description="SHA-256 hash")


class EmissionsCalculator:
    """
    Calculate combustion emissions using deterministic correlations.

    Zero-hallucination approach:
    - Zeldovich mechanism for thermal NOx
    - Fenimore mechanism for prompt NOx
    - Fuel nitrogen conversion for fuel NOx
    - Chemical equilibrium for CO
    - Stoichiometry for CO2 and SOx
    - Emission factors for PM and VOC

    No AI/ML models used for regulatory calculations.
    """

    # Molecular weights (g/mol)
    MW = {
        'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007, 'S': 32.065,
        'O2': 31.998, 'N2': 28.014, 'CO2': 44.01, 'H2O': 18.015,
        'SO2': 64.064, 'NO': 30.006, 'NO2': 46.006, 'CO': 28.01
    }

    # Universal gas constant
    R = 8.314  # J/mol·K

    # EPA emission limits (example values)
    EPA_LIMITS = {
        'gas_turbine': {
            'nox_ppm': 42,  # @ 15% O2
            'co_ppm': 10,   # @ 15% O2
            'sox_ppm': 10,  # Fuel sulfur limit
        },
        'boiler': {
            'nox_mg_nm3': 200,  # @ 3% O2
            'co_mg_nm3': 100,   # @ 3% O2
            'pm_mg_nm3': 20,    # @ 3% O2
        }
    }

    # EU IED limits (mg/Nm³ @ 3% O2 for combustion plants 50-100 MW)
    EU_IED_LIMITS = {
        'nox': 200,  # Natural gas
        'co': 100,
        'sox': 35,   # Natural gas
        'pm': 5,     # Natural gas
    }

    def __init__(self):
        """Initialize emissions calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate(self, inputs: EmissionsInput) -> EmissionsOutput:
        """
        Main calculation method for all emissions.

        Args:
            inputs: Validated input parameters

        Returns:
            EmissionsOutput with all emission calculations
        """
        start_time = datetime.now()
        self.logger.info("Starting emissions calculation")

        # Calculate NOx components
        thermal_nox = self._calculate_thermal_nox(inputs)
        prompt_nox = self._calculate_prompt_nox(inputs)
        fuel_nox = self._calculate_fuel_nox(inputs)
        total_nox = thermal_nox + prompt_nox + fuel_nox

        # Apply NOx reduction technologies
        if inputs.low_nox_burner:
            total_nox *= 0.5  # 50% reduction
        if inputs.flue_gas_recirculation:
            total_nox *= 0.7  # Additional 30% reduction
        if inputs.scr_efficiency:
            total_nox *= (1 - inputs.scr_efficiency / 100)

        # Calculate CO
        co_ppm = self._calculate_co_formation(inputs)

        # Calculate CO2 (stoichiometric)
        co2_kg_hr = self._calculate_co2_emissions(inputs)

        # Calculate SOx (fuel sulfur conversion)
        sox_kg_hr = self._calculate_sox_emissions(inputs)

        # Calculate PM
        pm_kg_hr = self._calculate_pm_emissions(inputs)

        # Calculate VOC
        voc_kg_hr = self._calculate_voc_emissions(inputs)

        # Convert to mass flow rates
        flue_gas_flow = self._calculate_flue_gas_flow(inputs)
        nox_kg_hr = self._ppm_to_kg_hr(total_nox, self.MW['NO2'], flue_gas_flow)
        co_kg_hr = self._ppm_to_kg_hr(co_ppm, self.MW['CO'], flue_gas_flow)

        # Convert to concentrations at reference O2
        ref_o2 = 3.0 if inputs.fuel_flow_kg_hr < 1000 else 3.0  # Boiler reference
        nox_mg_nm3 = self._correct_to_reference_o2(
            self._ppm_to_mg_nm3(total_nox, self.MW['NO2']),
            inputs.flue_gas_o2_percent,
            ref_o2
        )
        co_mg_nm3 = self._correct_to_reference_o2(
            self._ppm_to_mg_nm3(co_ppm, self.MW['CO']),
            inputs.flue_gas_o2_percent,
            ref_o2
        )
        sox_mg_nm3 = (sox_kg_hr * 1e6) / (flue_gas_flow * 3600) if flue_gas_flow > 0 else 0
        pm_mg_nm3 = (pm_kg_hr * 1e6) / (flue_gas_flow * 3600) if flue_gas_flow > 0 else 0

        # Calculate CO2 concentration
        co2_percent = (co2_kg_hr / (inputs.fuel_flow_kg_hr + inputs.fuel_flow_kg_hr * 15)) * 100

        # Annual emissions
        nox_tonnes_yr = nox_kg_hr * inputs.operating_hours_year / 1000
        co2_tonnes_yr = co2_kg_hr * inputs.operating_hours_year / 1000
        co2e_tonnes_yr = co2_tonnes_yr  # Simplified (could add CH4, N2O)

        # Emission factors
        fuel_input_gj = inputs.fuel_flow_kg_hr * inputs.fuel_lhv_mj_kg / 1000
        nox_g_gj = (nox_kg_hr * 1000) / fuel_input_gj if fuel_input_gj > 0 else 0
        co_g_gj = (co_kg_hr * 1000) / fuel_input_gj if fuel_input_gj > 0 else 0
        co2_kg_gj = co2_kg_hr / fuel_input_gj if fuel_input_gj > 0 else 0

        # Check compliance
        compliance, violations, margins = self._check_compliance(
            nox_mg_nm3, co_mg_nm3, sox_mg_nm3, pm_mg_nm3
        )

        # Calculate provenance hash
        input_str = inputs.json()
        timestamp = datetime.now().isoformat()
        provenance_data = f"{input_str}|{timestamp}|{total_nox}|{co_ppm}|{co2_kg_hr}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return EmissionsOutput(
            nox_kg_hr=round(nox_kg_hr, 4),
            co_kg_hr=round(co_kg_hr, 4),
            co2_kg_hr=round(co2_kg_hr, 2),
            sox_kg_hr=round(sox_kg_hr, 4),
            pm_kg_hr=round(pm_kg_hr, 4),
            voc_kg_hr=round(voc_kg_hr, 4),
            nox_mg_nm3=round(nox_mg_nm3, 2),
            co_mg_nm3=round(co_mg_nm3, 2),
            sox_mg_nm3=round(sox_mg_nm3, 2),
            pm_mg_nm3=round(pm_mg_nm3, 2),
            nox_ppm_dry=round(total_nox, 2),
            co_ppm_dry=round(co_ppm, 2),
            o2_percent_dry=round(inputs.flue_gas_o2_percent, 2),
            co2_percent_dry=round(co2_percent, 2),
            nox_tonnes_year=round(nox_tonnes_yr, 2),
            co2_tonnes_year=round(co2_tonnes_yr, 0),
            co2e_tonnes_year=round(co2e_tonnes_yr, 0),
            nox_g_gj=round(nox_g_gj, 3),
            co_g_gj=round(co_g_gj, 3),
            co2_kg_gj=round(co2_kg_gj, 2),
            thermal_nox_ppm=round(thermal_nox, 2),
            prompt_nox_ppm=round(prompt_nox, 2),
            fuel_nox_ppm=round(fuel_nox, 2),
            compliance_status=compliance,
            violations=violations,
            margin_to_limit=margins,
            calculation_timestamp=timestamp,
            provenance_hash=provenance_hash
        )

    def _calculate_thermal_nox(self, inputs: EmissionsInput) -> float:
        """
        Calculate thermal NOx using extended Zeldovich mechanism.

        Rate equations:
        O + N2 -> NO + N
        N + O2 -> NO + O
        N + OH -> NO + H

        Exponential temperature dependence above 1500°C.
        """
        T = inputs.combustion_temp_c + 273.15  # Kelvin

        if T < 1773:  # Below 1500°C, minimal thermal NOx
            return 0.0

        # Zeldovich rate constant (simplified)
        # k = A * T^n * exp(-E/RT)
        E_activation = 315000  # J/mol activation energy
        k = 7.6e10 * math.exp(-E_activation / (self.R * T))

        # O2 and N2 concentrations (simplified)
        o2_mole_fraction = inputs.flue_gas_o2_percent / 100
        n2_mole_fraction = 0.79 * (1 - inputs.excess_air_percent / 100)

        # Residence time effect
        tau = inputs.residence_time_s

        # Thermal NOx formation (ppm)
        # Simplified correlation based on temperature and residence time
        thermal_nox = (
            k * o2_mole_fraction ** 0.5 * n2_mole_fraction * tau * 1e6
        )

        # Temperature factor (exponential above 1500°C)
        if T > 2073:  # Above 1800°C
            thermal_nox *= math.exp((T - 2073) / 200)

        # Practical limit
        thermal_nox = min(thermal_nox, 2000)  # Cap at 2000 ppm

        return thermal_nox

    def _calculate_prompt_nox(self, inputs: EmissionsInput) -> float:
        """
        Calculate prompt NOx (Fenimore mechanism).

        HC radicals react with N2 in flame front:
        CH + N2 -> HCN + N -> NO

        Significant in fuel-rich zones.
        """
        # Prompt NOx depends on fuel type and stoichiometry
        # Higher with hydrocarbon fuels, peaks at phi~1.2

        # Equivalence ratio
        theoretical_air = self._calculate_theoretical_air(inputs)
        actual_air = theoretical_air * (1 + inputs.excess_air_percent / 100)
        phi = theoretical_air / actual_air if actual_air > 0 else 1.0

        # Base prompt NOx (ppm) - peaks at slightly rich
        if phi < 0.7 or phi > 1.5:
            base_prompt = 5  # Low outside optimal range
        else:
            # Gaussian-like peak at phi = 1.2
            base_prompt = 20 * math.exp(-((phi - 1.2) ** 2) / 0.1)

        # Temperature factor (lower than thermal NOx)
        T = inputs.combustion_temp_c + 273.15
        if T < 1473:  # Below 1200°C
            temp_factor = 0.5
        elif T < 1773:  # 1200-1500°C
            temp_factor = 1.0
        else:  # Above 1500°C
            temp_factor = 1.2

        prompt_nox = base_prompt * temp_factor

        # HC content factor (more HC -> more prompt NOx)
        hc_factor = 1 + inputs.fuel_hydrogen / 20
        prompt_nox *= hc_factor

        return min(prompt_nox, 50)  # Cap at 50 ppm

    def _calculate_fuel_nox(self, inputs: EmissionsInput) -> float:
        """
        Calculate fuel NOx from nitrogen in fuel.

        Conversion efficiency depends on:
        - Fuel nitrogen content
        - Excess air
        - Temperature
        """
        if inputs.fuel_nitrogen <= 0:
            return 0.0

        # Fuel nitrogen to NOx conversion (20-80% typical)
        # Higher conversion at higher O2
        base_conversion = 0.4  # 40% base conversion

        # Excess air effect
        if inputs.excess_air_percent < 10:
            conversion = base_conversion * 0.7
        elif inputs.excess_air_percent > 50:
            conversion = base_conversion * 1.2
        else:
            conversion = base_conversion

        # Temperature effect (higher T -> higher conversion)
        T = inputs.combustion_temp_c
        if T < 1000:
            conversion *= 0.8
        elif T > 1400:
            conversion *= 1.1

        conversion = min(conversion, 0.8)  # Cap at 80%

        # Calculate fuel NOx (ppm)
        # N in fuel -> NO (30/14 mass ratio)
        n_kg_hr = inputs.fuel_flow_kg_hr * inputs.fuel_nitrogen / 100
        no_kg_hr = n_kg_hr * (self.MW['NO'] / self.MW['N']) * conversion

        # Convert to ppm
        flue_gas_flow = self._calculate_flue_gas_flow(inputs)
        fuel_nox = self._kg_hr_to_ppm(no_kg_hr, self.MW['NO'], flue_gas_flow)

        return fuel_nox

    def _calculate_co_formation(self, inputs: EmissionsInput) -> float:
        """
        Calculate CO from incomplete combustion.

        CO forms when:
        - Insufficient oxygen (fuel-rich)
        - Poor mixing
        - Flame quenching
        - Low temperature
        """
        # Use measured CO if available
        if inputs.flue_gas_co_ppm is not None:
            return inputs.flue_gas_co_ppm

        # Otherwise calculate from conditions
        base_co = 10  # ppm at good conditions

        # Excess air effect (exponential increase at low EA)
        if inputs.excess_air_percent < 5:
            ea_factor = math.exp((5 - inputs.excess_air_percent) / 2)
        elif inputs.excess_air_percent > 30:
            ea_factor = 0.5  # Very low CO at high excess air
        else:
            ea_factor = 1.0

        # Temperature effect
        T = inputs.combustion_temp_c
        if T < 800:
            temp_factor = math.exp((800 - T) / 100)
        elif T > 1200:
            temp_factor = 0.7  # Good combustion at high temp
        else:
            temp_factor = 1.0

        # Load effect (poor mixing at low load)
        if inputs.load_percent < 50:
            load_factor = 2.0
        elif inputs.load_percent < 70:
            load_factor = 1.5
        else:
            load_factor = 1.0

        co_ppm = base_co * ea_factor * temp_factor * load_factor

        # Practical limits
        co_ppm = max(1, min(co_ppm, 5000))

        return co_ppm

    def _calculate_co2_emissions(self, inputs: EmissionsInput) -> float:
        """
        Calculate CO2 from fuel carbon content (stoichiometric).

        C + O2 -> CO2
        Mass ratio: 44/12 = 3.67
        """
        carbon_kg_hr = inputs.fuel_flow_kg_hr * inputs.fuel_carbon / 100
        co2_kg_hr = carbon_kg_hr * (self.MW['CO2'] / self.MW['C'])

        return co2_kg_hr

    def _calculate_sox_emissions(self, inputs: EmissionsInput) -> float:
        """
        Calculate SOx from fuel sulfur (assume 100% conversion to SO2).

        S + O2 -> SO2
        Mass ratio: 64/32 = 2.0
        """
        sulfur_kg_hr = inputs.fuel_flow_kg_hr * inputs.fuel_sulfur / 100
        sox_kg_hr = sulfur_kg_hr * (self.MW['SO2'] / self.MW['S'])

        return sox_kg_hr

    def _calculate_pm_emissions(self, inputs: EmissionsInput) -> float:
        """
        Calculate particulate matter emissions.

        PM sources:
        - Fuel ash
        - Incomplete combustion (soot)
        - Sulfates
        """
        # Base PM from fuel ash (assume 80% of ash becomes PM)
        ash_pm = inputs.fuel_flow_kg_hr * inputs.fuel_ash / 100 * 0.8

        # Soot from incomplete combustion (correlates with CO)
        co_ppm = self._calculate_co_formation(inputs)
        soot_factor = co_ppm / 1000  # Rough correlation
        soot_pm = inputs.fuel_flow_kg_hr * 0.001 * soot_factor

        # Sulfate PM (small fraction of SOx)
        sox_kg_hr = self._calculate_sox_emissions(inputs)
        sulfate_pm = sox_kg_hr * 0.02  # 2% of SOx as sulfate PM

        total_pm = ash_pm + soot_pm + sulfate_pm

        return total_pm

    def _calculate_voc_emissions(self, inputs: EmissionsInput) -> float:
        """
        Calculate VOC (volatile organic compounds) emissions.

        VOCs from incomplete combustion of hydrocarbons.
        Correlates with CO emissions.
        """
        # Simple correlation with CO
        co_ppm = self._calculate_co_formation(inputs)
        voc_ppm = co_ppm * 0.1  # Typical VOC/CO ratio

        # Convert to kg/hr (use methane MW as proxy)
        flue_gas_flow = self._calculate_flue_gas_flow(inputs)
        voc_kg_hr = self._ppm_to_kg_hr(voc_ppm, 16, flue_gas_flow)

        return voc_kg_hr

    def _calculate_theoretical_air(self, inputs: EmissionsInput) -> float:
        """Calculate theoretical air requirement (kg air/kg fuel)"""
        # Simplified: Based on C, H, S oxidation
        c = inputs.fuel_carbon / 100
        h = inputs.fuel_hydrogen / 100
        s = inputs.fuel_sulfur / 100
        o = inputs.fuel_oxygen / 100

        # O2 required (kg O2/kg fuel)
        o2_required = c * (32/12) + h * (8/1) + s * (32/32) - o

        # Air required (23.15% O2 by mass in air)
        air_required = o2_required / 0.2315

        return air_required

    def _calculate_flue_gas_flow(self, inputs: EmissionsInput) -> float:
        """Calculate flue gas volumetric flow (Nm³/s)"""
        # Mass flow (kg/s)
        theoretical_air = self._calculate_theoretical_air(inputs)
        actual_air = theoretical_air * (1 + inputs.excess_air_percent / 100)

        flue_gas_mass = (inputs.fuel_flow_kg_hr +
                        inputs.fuel_flow_kg_hr * actual_air) / 3600

        # Volume flow at STP (assume density 1.3 kg/Nm³)
        flue_gas_flow = flue_gas_mass / 1.3

        return flue_gas_flow

    def _ppm_to_kg_hr(self, ppm: float, mw: float, flow_nm3_s: float) -> float:
        """Convert ppm to kg/hr"""
        # ppm to mg/Nm³: ppm * MW / 22.4
        mg_nm3 = ppm * mw / 22.4

        # mg/Nm³ to kg/hr
        kg_hr = mg_nm3 * flow_nm3_s * 3600 / 1e6

        return kg_hr

    def _kg_hr_to_ppm(self, kg_hr: float, mw: float, flow_nm3_s: float) -> float:
        """Convert kg/hr to ppm"""
        # kg/hr to mg/Nm³
        mg_nm3 = (kg_hr * 1e6) / (flow_nm3_s * 3600)

        # mg/Nm³ to ppm
        ppm = mg_nm3 * 22.4 / mw

        return ppm

    def _ppm_to_mg_nm3(self, ppm: float, mw: float) -> float:
        """Convert ppm to mg/Nm³ at STP"""
        return ppm * mw / 22.4

    def _correct_to_reference_o2(self, conc: float, measured_o2: float, ref_o2: float) -> float:
        """
        Correct concentration to reference O2 level.

        Formula: C_ref = C_measured * (21 - O2_ref) / (21 - O2_measured)
        """
        if measured_o2 >= 21:
            return conc

        correction = (21 - ref_o2) / (21 - measured_o2)
        return conc * correction

    def _check_compliance(
        self,
        nox_mg_nm3: float,
        co_mg_nm3: float,
        sox_mg_nm3: float,
        pm_mg_nm3: float
    ) -> Tuple[Dict[str, str], List[str], Dict[str, float]]:
        """Check compliance with EU IED limits"""
        compliance = {}
        violations = []
        margins = {}

        # Check NOx
        limit = self.EU_IED_LIMITS['nox']
        margin = ((limit - nox_mg_nm3) / limit * 100) if limit > 0 else 100
        margins['nox'] = round(margin, 1)

        if nox_mg_nm3 > limit:
            compliance['nox'] = ComplianceLevel.VIOLATION.value
            violations.append(f"NOx exceeds limit: {nox_mg_nm3:.1f} > {limit} mg/Nm³")
        elif nox_mg_nm3 > limit * 0.9:
            compliance['nox'] = ComplianceLevel.CRITICAL.value
        elif nox_mg_nm3 > limit * 0.8:
            compliance['nox'] = ComplianceLevel.WARNING.value
        else:
            compliance['nox'] = ComplianceLevel.COMPLIANT.value

        # Check CO
        limit = self.EU_IED_LIMITS['co']
        margin = ((limit - co_mg_nm3) / limit * 100) if limit > 0 else 100
        margins['co'] = round(margin, 1)

        if co_mg_nm3 > limit:
            compliance['co'] = ComplianceLevel.VIOLATION.value
            violations.append(f"CO exceeds limit: {co_mg_nm3:.1f} > {limit} mg/Nm³")
        elif co_mg_nm3 > limit * 0.9:
            compliance['co'] = ComplianceLevel.CRITICAL.value
        elif co_mg_nm3 > limit * 0.8:
            compliance['co'] = ComplianceLevel.WARNING.value
        else:
            compliance['co'] = ComplianceLevel.COMPLIANT.value

        # Check SOx
        limit = self.EU_IED_LIMITS['sox']
        margin = ((limit - sox_mg_nm3) / limit * 100) if limit > 0 else 100
        margins['sox'] = round(margin, 1)

        if sox_mg_nm3 > limit:
            compliance['sox'] = ComplianceLevel.VIOLATION.value
            violations.append(f"SOx exceeds limit: {sox_mg_nm3:.1f} > {limit} mg/Nm³")
        else:
            compliance['sox'] = ComplianceLevel.COMPLIANT.value

        # Check PM
        limit = self.EU_IED_LIMITS['pm']
        margin = ((limit - pm_mg_nm3) / limit * 100) if limit > 0 else 100
        margins['pm'] = round(margin, 1)

        if pm_mg_nm3 > limit:
            compliance['pm'] = ComplianceLevel.VIOLATION.value
            violations.append(f"PM exceeds limit: {pm_mg_nm3:.1f} > {limit} mg/Nm³")
        else:
            compliance['pm'] = ComplianceLevel.COMPLIANT.value

        return compliance, violations, margins