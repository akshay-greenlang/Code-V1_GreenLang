# -*- coding: utf-8 -*-
"""
GL-018 FLUEFLOW - Flue Gas Composition Calculator

Zero-hallucination, deterministic calculations for flue gas composition
analysis following ASME PTC 4.1 and EPA Method 19 standards.

This module provides:
- Complete combustion stoichiometry for all fuel types
- Excess air calculation from O2 and CO2 measurements
- Wet and dry basis conversions
- Dew point calculation (water, acid dew points)
- Molecular weight of flue gas mixture
- Specific heat of flue gas (temperature dependent)
- JANAF thermochemical tables integration

Standards Reference:
- ASME PTC 4.1 - Fired Steam Generators Performance Test Code
- EPA Method 19 - Determination of SO2 Removal Efficiency
- JANAF Thermochemical Tables (4th Edition)
- ISO 10396 - Stationary Source Emissions - Sampling

Formula Derivations:
    Combustion Stoichiometry (per mole of fuel component):
        C + O2 -> CO2                           (Carbon combustion)
        H2 + 0.5 O2 -> H2O                      (Hydrogen combustion)
        S + O2 -> SO2                           (Sulfur combustion)

    Theoretical Air Requirement (mass basis, kg air/kg fuel):
        A_th = 11.5*C + 34.5*(H - O/8) + 4.3*S   (Approximate formula)

    Excess Air from O2 measurement:
        EA% = O2_dry / (21 - O2_dry) * 100

    Excess Air from CO2 measurement:
        EA% = (CO2_max - CO2_meas) / CO2_meas * 100

    Wet/Dry Conversion:
        C_dry = C_wet / (1 - H2O/100)
        C_wet = C_dry * (1 - H2O/100)

    Dew Point (water vapor):
        T_dp = (B * ln(Pw/A)) / (C - ln(Pw/A))   (Magnus formula)

    Acid Dew Point (sulfuric acid):
        T_adp = 1000 / (2.276 - 0.0294*ln(pH2O) - 0.0858*ln(pSO3) + 0.0062*ln(pH2O*pSO3))
        (Pierce correlation, K)

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Universal gas constant
R_UNIVERSAL = 8.314462  # J/(mol*K)
R_SPECIFIC_AIR = 287.058  # J/(kg*K)

# Standard conditions
STANDARD_TEMP_K = 273.15  # 0 deg C
STANDARD_TEMP_C = 0.0
STANDARD_PRESSURE_PA = 101325.0  # 1 atm
STANDARD_PRESSURE_KPA = 101.325

# Reference O2 in dry air
O2_IN_AIR_VOL_PCT = 20.95  # Volume percent
N2_IN_AIR_VOL_PCT = 78.09  # Volume percent
AR_IN_AIR_VOL_PCT = 0.93   # Argon volume percent
CO2_IN_AIR_VOL_PCT = 0.03  # CO2 volume percent

# Molecular weights (g/mol)
MW_C = 12.011
MW_H = 1.008
MW_O = 15.999
MW_N = 14.007
MW_S = 32.065
MW_H2 = 2.016
MW_O2 = 31.998
MW_N2 = 28.014
MW_CO = 28.010
MW_CO2 = 44.009
MW_H2O = 18.015
MW_SO2 = 64.064
MW_SO3 = 80.063
MW_NO = 30.006
MW_NO2 = 46.005
MW_NOX = 46.005  # Reported as NO2 equivalent
MW_AIR = 28.964
MW_AR = 39.948

# Molar volume at STP (Nm3/kmol)
MOLAR_VOLUME_STP = 22.414

# Latent heat of water vaporization (kJ/kg)
LATENT_HEAT_H2O = 2257.0

# Water vapor constants for dew point (Magnus formula)
MAGNUS_A = 6.112  # hPa
MAGNUS_B = 17.67
MAGNUS_C = 243.5  # deg C


# =============================================================================
# JANAF THERMOCHEMICAL DATA
# =============================================================================

class JANAFData:
    """
    JANAF Thermochemical Tables data for common flue gas species.

    Reference: JANAF Thermochemical Tables, 4th Edition (1998)
    Data format: Cp = a + b*T + c*T^2 + d*T^3 + e/T^2 (J/mol-K)
    Temperature range: 300-1500 K
    """

    # Specific heat coefficients (Shomate equation): Cp = A + B*t + C*t^2 + D*t^3 + E/t^2
    # Where t = T(K)/1000
    # Units: J/(mol*K)
    SHOMATE_COEFFS = {
        'N2': {
            'range': (300, 1500),
            'A': 28.98641,
            'B': 1.853978,
            'C': -9.647459,
            'D': 16.63537,
            'E': 0.000117,
        },
        'O2': {
            'range': (300, 1500),
            'A': 31.32234,
            'B': -20.23531,
            'C': 57.86644,
            'D': -36.50624,
            'E': -0.007374,
        },
        'CO2': {
            'range': (300, 1500),
            'A': 24.99735,
            'B': 55.18696,
            'C': -33.69137,
            'D': 7.948387,
            'E': -0.136638,
        },
        'H2O': {
            'range': (300, 1500),
            'A': 30.09200,
            'B': 6.832514,
            'C': 6.793435,
            'D': -2.534480,
            'E': 0.082139,
        },
        'CO': {
            'range': (300, 1500),
            'A': 25.56759,
            'B': 6.096130,
            'C': 4.054656,
            'D': -2.671301,
            'E': 0.131021,
        },
        'SO2': {
            'range': (300, 1500),
            'A': 21.43049,
            'B': 74.35094,
            'C': -57.75217,
            'D': 16.35534,
            'E': 0.086731,
        },
        'NO': {
            'range': (300, 1500),
            'A': 23.83491,
            'B': 12.58878,
            'C': -1.139011,
            'D': -1.497459,
            'E': 0.214194,
        },
        'NO2': {
            'range': (300, 1500),
            'A': 16.10857,
            'B': 75.89525,
            'C': -54.38740,
            'D': 14.30777,
            'E': 0.239423,
        },
        'AR': {
            'range': (300, 1500),
            'A': 20.78600,
            'B': 0.0,
            'C': 0.0,
            'D': 0.0,
            'E': 0.0,
        },
    }

    @classmethod
    def get_cp_molar(cls, species: str, temp_k: float) -> float:
        """
        Calculate molar specific heat at constant pressure.

        Args:
            species: Chemical species name
            temp_k: Temperature in Kelvin

        Returns:
            Molar specific heat Cp in J/(mol*K)
        """
        if species not in cls.SHOMATE_COEFFS:
            raise ValueError(f"Unknown species: {species}")

        coeffs = cls.SHOMATE_COEFFS[species]
        t = temp_k / 1000.0

        cp = (coeffs['A'] +
              coeffs['B'] * t +
              coeffs['C'] * t**2 +
              coeffs['D'] * t**3 +
              coeffs['E'] / (t**2))

        return cp

    @classmethod
    def get_cp_mass(cls, species: str, temp_k: float, mw: float) -> float:
        """
        Calculate mass-based specific heat.

        Args:
            species: Chemical species name
            temp_k: Temperature in Kelvin
            mw: Molecular weight (g/mol)

        Returns:
            Mass-based specific heat Cp in kJ/(kg*K)
        """
        cp_molar = cls.get_cp_molar(species, temp_k)
        return cp_molar / mw  # J/(g*K) = kJ/(kg*K)


# =============================================================================
# FUEL COMPOSITION DATABASE
# =============================================================================

@dataclass(frozen=True)
class FuelComposition:
    """
    Ultimate analysis of fuel (mass percent, as-received basis).

    Attributes:
        carbon: Carbon content (% by mass)
        hydrogen: Hydrogen content (% by mass)
        oxygen: Oxygen content (% by mass)
        nitrogen: Nitrogen content (% by mass)
        sulfur: Sulfur content (% by mass)
        moisture: Moisture content (% by mass)
        ash: Ash content (% by mass)
        hhv_kj_kg: Higher heating value (kJ/kg)
        lhv_kj_kg: Lower heating value (kJ/kg)
    """
    carbon: float
    hydrogen: float
    oxygen: float
    nitrogen: float
    sulfur: float
    moisture: float
    ash: float
    hhv_kj_kg: float
    lhv_kj_kg: float

    def validate(self) -> bool:
        """Validate that composition sums to approximately 100%."""
        total = (self.carbon + self.hydrogen + self.oxygen +
                 self.nitrogen + self.sulfur + self.moisture + self.ash)
        return 99.0 <= total <= 101.0


# Standard fuel compositions (typical values)
FUEL_COMPOSITIONS = {
    'natural_gas': FuelComposition(
        carbon=74.0, hydrogen=24.0, oxygen=0.5, nitrogen=1.0,
        sulfur=0.01, moisture=0.0, ash=0.0,
        hhv_kj_kg=55500.0, lhv_kj_kg=50000.0
    ),
    'fuel_oil_no2': FuelComposition(
        carbon=87.2, hydrogen=12.5, oxygen=0.1, nitrogen=0.0,
        sulfur=0.2, moisture=0.0, ash=0.0,
        hhv_kj_kg=45500.0, lhv_kj_kg=42700.0
    ),
    'fuel_oil_no6': FuelComposition(
        carbon=87.0, hydrogen=10.5, oxygen=0.5, nitrogen=0.3,
        sulfur=1.5, moisture=0.0, ash=0.1,
        hhv_kj_kg=43000.0, lhv_kj_kg=40500.0
    ),
    'diesel': FuelComposition(
        carbon=86.5, hydrogen=13.2, oxygen=0.0, nitrogen=0.0,
        sulfur=0.3, moisture=0.0, ash=0.0,
        hhv_kj_kg=45800.0, lhv_kj_kg=43100.0
    ),
    'propane': FuelComposition(
        carbon=81.8, hydrogen=18.2, oxygen=0.0, nitrogen=0.0,
        sulfur=0.0, moisture=0.0, ash=0.0,
        hhv_kj_kg=50300.0, lhv_kj_kg=46400.0
    ),
    'bituminous_coal': FuelComposition(
        carbon=75.5, hydrogen=5.0, oxygen=6.5, nitrogen=1.5,
        sulfur=1.5, moisture=3.5, ash=6.5,
        hhv_kj_kg=31400.0, lhv_kj_kg=30200.0
    ),
    'subbituminous_coal': FuelComposition(
        carbon=52.0, hydrogen=3.5, oxygen=11.0, nitrogen=0.8,
        sulfur=0.4, moisture=25.0, ash=7.3,
        hhv_kj_kg=21500.0, lhv_kj_kg=20300.0
    ),
    'lignite': FuelComposition(
        carbon=40.0, hydrogen=2.8, oxygen=12.0, nitrogen=0.6,
        sulfur=0.8, moisture=35.0, ash=8.8,
        hhv_kj_kg=16300.0, lhv_kj_kg=15100.0
    ),
    'wood_biomass': FuelComposition(
        carbon=50.0, hydrogen=6.0, oxygen=42.0, nitrogen=0.5,
        sulfur=0.1, moisture=20.0, ash=1.4,
        hhv_kj_kg=18500.0, lhv_kj_kg=17000.0
    ),
}


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class FlueGasCompositionInput:
    """
    Input parameters for flue gas composition calculations.

    Attributes:
        o2_pct: Oxygen concentration (vol %, dry or wet basis)
        co2_pct: Carbon dioxide concentration (vol %, dry or wet basis)
        co_ppm: Carbon monoxide concentration (ppm, dry basis)
        so2_ppm: Sulfur dioxide concentration (ppm, dry basis)
        nox_ppm: Nitrogen oxides concentration (ppm, as NO2)
        flue_gas_temp_c: Flue gas temperature (deg C)
        ambient_temp_c: Ambient air temperature (deg C)
        ambient_pressure_kpa: Ambient pressure (kPa)
        relative_humidity_pct: Ambient relative humidity (%)
        fuel_type: Fuel type identifier
        fuel_composition: Optional custom fuel composition
        measurement_basis: 'wet' or 'dry'
        h2o_pct_measured: Water vapor content if measured (vol %)
    """
    o2_pct: float
    co2_pct: float
    flue_gas_temp_c: float
    ambient_temp_c: float
    fuel_type: str
    co_ppm: float = 0.0
    so2_ppm: float = 0.0
    nox_ppm: float = 0.0
    ambient_pressure_kpa: float = STANDARD_PRESSURE_KPA
    relative_humidity_pct: float = 50.0
    fuel_composition: Optional[FuelComposition] = None
    measurement_basis: str = 'dry'
    h2o_pct_measured: Optional[float] = None


@dataclass(frozen=True)
class StoichiometryResult:
    """
    Combustion stoichiometry calculation results.

    Attributes:
        theoretical_air_kg_per_kg_fuel: Stoichiometric air requirement (kg/kg fuel)
        theoretical_air_mol_per_mol_fuel: Molar air requirement
        theoretical_o2_kg_per_kg_fuel: Theoretical O2 requirement (kg/kg fuel)
        co2_produced_kg_per_kg_fuel: CO2 produced (kg/kg fuel)
        h2o_produced_kg_per_kg_fuel: H2O produced (kg/kg fuel)
        so2_produced_kg_per_kg_fuel: SO2 produced (kg/kg fuel)
        n2_in_flue_gas_kg_per_kg_fuel: N2 in flue gas (kg/kg fuel)
        theoretical_co2_max_pct: Maximum theoretical CO2 (vol % dry)
    """
    theoretical_air_kg_per_kg_fuel: float
    theoretical_air_mol_per_mol_fuel: float
    theoretical_o2_kg_per_kg_fuel: float
    co2_produced_kg_per_kg_fuel: float
    h2o_produced_kg_per_kg_fuel: float
    so2_produced_kg_per_kg_fuel: float
    n2_in_flue_gas_kg_per_kg_fuel: float
    theoretical_co2_max_pct: float


@dataclass(frozen=True)
class FlueGasCompositionOutput:
    """
    Complete flue gas composition analysis results.

    Attributes:
        o2_pct_dry: O2 concentration (vol %, dry basis)
        o2_pct_wet: O2 concentration (vol %, wet basis)
        co2_pct_dry: CO2 concentration (vol %, dry basis)
        co2_pct_wet: CO2 concentration (vol %, wet basis)
        h2o_pct: Water vapor content (vol %)
        n2_pct_dry: N2 concentration (vol %, dry basis)
        n2_pct_wet: N2 concentration (vol %, wet basis)
        co_ppm_dry: CO concentration (ppm, dry basis)
        so2_ppm_dry: SO2 concentration (ppm, dry basis)
        nox_ppm_dry: NOx concentration (ppm, dry basis)

        excess_air_pct: Excess air from O2 measurement (%)
        excess_air_from_co2_pct: Excess air from CO2 measurement (%)
        lambda_ratio: Air-fuel equivalence ratio (lambda)
        actual_air_kg_per_kg_fuel: Actual air used (kg/kg fuel)

        molecular_weight_dry: Molecular weight of dry flue gas (g/mol)
        molecular_weight_wet: Molecular weight of wet flue gas (g/mol)
        density_dry_kg_m3: Dry flue gas density at STP (kg/Nm3)
        density_wet_kg_m3: Wet flue gas density at STP (kg/Nm3)
        density_actual_kg_m3: Flue gas density at actual conditions (kg/m3)

        cp_dry_kj_kg_k: Specific heat of dry flue gas (kJ/kg-K)
        cp_wet_kj_kg_k: Specific heat of wet flue gas (kJ/kg-K)

        water_dew_point_c: Water vapor dew point (deg C)
        acid_dew_point_c: Sulfuric acid dew point (deg C)

        flue_gas_volume_dry_nm3_per_kg_fuel: Dry flue gas volume (Nm3/kg fuel)
        flue_gas_volume_wet_nm3_per_kg_fuel: Wet flue gas volume (Nm3/kg fuel)

        stoichiometry: Combustion stoichiometry results
    """
    # Compositions
    o2_pct_dry: float
    o2_pct_wet: float
    co2_pct_dry: float
    co2_pct_wet: float
    h2o_pct: float
    n2_pct_dry: float
    n2_pct_wet: float
    co_ppm_dry: float
    so2_ppm_dry: float
    nox_ppm_dry: float

    # Excess air and ratios
    excess_air_pct: float
    excess_air_from_co2_pct: float
    lambda_ratio: float
    actual_air_kg_per_kg_fuel: float

    # Physical properties
    molecular_weight_dry: float
    molecular_weight_wet: float
    density_dry_kg_m3: float
    density_wet_kg_m3: float
    density_actual_kg_m3: float

    # Thermal properties
    cp_dry_kj_kg_k: float
    cp_wet_kj_kg_k: float

    # Dew points
    water_dew_point_c: float
    acid_dew_point_c: float

    # Volumes
    flue_gas_volume_dry_nm3_per_kg_fuel: float
    flue_gas_volume_wet_nm3_per_kg_fuel: float

    # Stoichiometry
    stoichiometry: StoichiometryResult


# =============================================================================
# FLUE GAS COMPOSITION CALCULATOR
# =============================================================================

class FlueGasCompositionCalculator:
    """
    Zero-hallucination calculator for flue gas composition analysis.

    Implements deterministic calculations following ASME PTC 4.1 and
    EPA Method 19 for comprehensive flue gas characterization.

    All calculations are:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = FlueGasCompositionCalculator()
        >>> inputs = FlueGasCompositionInput(
        ...     o2_pct=3.5,
        ...     co2_pct=11.5,
        ...     flue_gas_temp_c=180.0,
        ...     ambient_temp_c=25.0,
        ...     fuel_type='natural_gas'
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Excess Air: {result.excess_air_pct:.1f}%")
        >>> print(f"Dew Point: {result.water_dew_point_c:.1f} deg C")
    """

    VERSION = "1.0.0"
    NAME = "FlueGasCompositionCalculator"

    def __init__(self):
        """Initialize the flue gas composition calculator."""
        self._tracker: Optional[ProvenanceTracker] = None
        self._step_counter: int = 0

    def calculate(
        self,
        inputs: FlueGasCompositionInput
    ) -> Tuple[FlueGasCompositionOutput, ProvenanceRecord]:
        """
        Perform complete flue gas composition analysis.

        Args:
            inputs: FlueGasCompositionInput with measurement data

        Returns:
            Tuple of (FlueGasCompositionOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ASME PTC 4.1", "EPA Method 19", "JANAF Tables"],
                "domain": "Flue Gas Composition Analysis"
            }
        )
        self._step_counter = 0

        # Set inputs for provenance
        input_dict = {
            "o2_pct": inputs.o2_pct,
            "co2_pct": inputs.co2_pct,
            "co_ppm": inputs.co_ppm,
            "so2_ppm": inputs.so2_ppm,
            "nox_ppm": inputs.nox_ppm,
            "flue_gas_temp_c": inputs.flue_gas_temp_c,
            "ambient_temp_c": inputs.ambient_temp_c,
            "ambient_pressure_kpa": inputs.ambient_pressure_kpa,
            "relative_humidity_pct": inputs.relative_humidity_pct,
            "fuel_type": inputs.fuel_type,
            "measurement_basis": inputs.measurement_basis,
            "h2o_pct_measured": inputs.h2o_pct_measured
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Get fuel composition
        fuel = self._get_fuel_composition(inputs)

        # Step 1: Calculate combustion stoichiometry
        stoichiometry = self._calculate_stoichiometry(fuel)

        # Step 2: Convert measurements to dry basis
        o2_dry, co2_dry = self._convert_to_dry_basis(
            inputs.o2_pct,
            inputs.co2_pct,
            inputs.measurement_basis,
            inputs.h2o_pct_measured
        )

        # Step 3: Calculate excess air from O2
        excess_air_o2 = self._calculate_excess_air_from_o2(o2_dry)

        # Step 4: Calculate excess air from CO2
        excess_air_co2 = self._calculate_excess_air_from_co2(
            co2_dry, stoichiometry.theoretical_co2_max_pct
        )

        # Step 5: Calculate lambda (air-fuel equivalence ratio)
        lambda_ratio = self._calculate_lambda_ratio(excess_air_o2)

        # Step 6: Calculate actual air requirement
        actual_air = self._calculate_actual_air(
            stoichiometry.theoretical_air_kg_per_kg_fuel,
            excess_air_o2
        )

        # Step 7: Calculate water vapor content in flue gas
        h2o_pct = self._calculate_water_vapor_content(
            fuel, excess_air_o2, inputs.ambient_temp_c,
            inputs.relative_humidity_pct, inputs.h2o_pct_measured
        )

        # Step 8: Calculate N2 content
        n2_dry = self._calculate_n2_content_dry(o2_dry, co2_dry)

        # Step 9: Convert all to wet basis
        o2_wet, co2_wet, n2_wet = self._convert_to_wet_basis(
            o2_dry, co2_dry, n2_dry, h2o_pct
        )

        # Step 10: Calculate molecular weight
        mw_dry, mw_wet = self._calculate_molecular_weight(
            o2_dry, co2_dry, n2_dry, h2o_pct
        )

        # Step 11: Calculate density
        density_dry, density_wet, density_actual = self._calculate_density(
            mw_dry, mw_wet, inputs.flue_gas_temp_c, inputs.ambient_pressure_kpa
        )

        # Step 12: Calculate specific heat (using JANAF data)
        cp_dry, cp_wet = self._calculate_specific_heat(
            o2_dry, co2_dry, n2_dry, h2o_pct,
            inputs.flue_gas_temp_c
        )

        # Step 13: Calculate water dew point
        water_dp = self._calculate_water_dew_point(
            h2o_pct, inputs.ambient_pressure_kpa
        )

        # Step 14: Calculate acid dew point
        acid_dp = self._calculate_acid_dew_point(
            h2o_pct, inputs.so2_ppm, inputs.flue_gas_temp_c
        )

        # Step 15: Calculate flue gas volumes
        vol_dry, vol_wet = self._calculate_flue_gas_volume(
            stoichiometry, excess_air_o2
        )

        # Create output
        output = FlueGasCompositionOutput(
            o2_pct_dry=round(o2_dry, 3),
            o2_pct_wet=round(o2_wet, 3),
            co2_pct_dry=round(co2_dry, 3),
            co2_pct_wet=round(co2_wet, 3),
            h2o_pct=round(h2o_pct, 3),
            n2_pct_dry=round(n2_dry, 3),
            n2_pct_wet=round(n2_wet, 3),
            co_ppm_dry=round(inputs.co_ppm, 1),
            so2_ppm_dry=round(inputs.so2_ppm, 1),
            nox_ppm_dry=round(inputs.nox_ppm, 1),
            excess_air_pct=round(excess_air_o2, 2),
            excess_air_from_co2_pct=round(excess_air_co2, 2),
            lambda_ratio=round(lambda_ratio, 4),
            actual_air_kg_per_kg_fuel=round(actual_air, 3),
            molecular_weight_dry=round(mw_dry, 3),
            molecular_weight_wet=round(mw_wet, 3),
            density_dry_kg_m3=round(density_dry, 4),
            density_wet_kg_m3=round(density_wet, 4),
            density_actual_kg_m3=round(density_actual, 4),
            cp_dry_kj_kg_k=round(cp_dry, 4),
            cp_wet_kj_kg_k=round(cp_wet, 4),
            water_dew_point_c=round(water_dp, 1),
            acid_dew_point_c=round(acid_dp, 1),
            flue_gas_volume_dry_nm3_per_kg_fuel=round(vol_dry, 4),
            flue_gas_volume_wet_nm3_per_kg_fuel=round(vol_wet, 4),
            stoichiometry=stoichiometry
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "o2_pct_dry": output.o2_pct_dry,
            "o2_pct_wet": output.o2_pct_wet,
            "co2_pct_dry": output.co2_pct_dry,
            "co2_pct_wet": output.co2_pct_wet,
            "h2o_pct": output.h2o_pct,
            "n2_pct_dry": output.n2_pct_dry,
            "excess_air_pct": output.excess_air_pct,
            "excess_air_from_co2_pct": output.excess_air_from_co2_pct,
            "lambda_ratio": output.lambda_ratio,
            "molecular_weight_dry": output.molecular_weight_dry,
            "molecular_weight_wet": output.molecular_weight_wet,
            "cp_dry_kj_kg_k": output.cp_dry_kj_kg_k,
            "cp_wet_kj_kg_k": output.cp_wet_kj_kg_k,
            "water_dew_point_c": output.water_dew_point_c,
            "acid_dew_point_c": output.acid_dew_point_c,
            "flue_gas_volume_dry_nm3_per_kg_fuel": output.flue_gas_volume_dry_nm3_per_kg_fuel,
            "flue_gas_volume_wet_nm3_per_kg_fuel": output.flue_gas_volume_wet_nm3_per_kg_fuel
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _next_step(self) -> int:
        """Get next step number for provenance tracking."""
        self._step_counter += 1
        return self._step_counter

    def _validate_inputs(self, inputs: FlueGasCompositionInput) -> None:
        """Validate input parameters."""
        if inputs.o2_pct < 0 or inputs.o2_pct > 21:
            raise ValueError(f"O2 concentration {inputs.o2_pct}% out of range (0-21%)")

        if inputs.co2_pct < 0 or inputs.co2_pct > 25:
            raise ValueError(f"CO2 concentration {inputs.co2_pct}% out of range (0-25%)")

        if inputs.co_ppm < 0:
            raise ValueError("CO concentration cannot be negative")

        if inputs.flue_gas_temp_c < 50 or inputs.flue_gas_temp_c > 1200:
            raise ValueError(
                f"Flue gas temperature {inputs.flue_gas_temp_c} deg C out of range (50-1200 deg C)"
            )

        if inputs.measurement_basis not in ('wet', 'dry'):
            raise ValueError(f"Invalid measurement basis: {inputs.measurement_basis}")

    def _get_fuel_composition(
        self, inputs: FlueGasCompositionInput
    ) -> FuelComposition:
        """Get fuel composition from input or database."""
        if inputs.fuel_composition is not None:
            return inputs.fuel_composition

        fuel_key = inputs.fuel_type.lower().replace(' ', '_').replace('-', '_')
        if fuel_key not in FUEL_COMPOSITIONS:
            raise ValueError(f"Unknown fuel type: {inputs.fuel_type}")

        return FUEL_COMPOSITIONS[fuel_key]

    def _calculate_stoichiometry(
        self, fuel: FuelComposition
    ) -> StoichiometryResult:
        """
        Calculate combustion stoichiometry from fuel ultimate analysis.

        Based on complete combustion reactions:
        - C + O2 -> CO2
        - H2 + 0.5*O2 -> H2O
        - S + O2 -> SO2

        Theoretical air formula (kg air/kg fuel):
            A_th = (1/0.232) * [2.667*C + 8*H + S - O] / 100

        where 0.232 is the mass fraction of O2 in air.
        """
        step = self._next_step()

        # Mass fractions (convert from %)
        c = fuel.carbon / 100.0
        h = fuel.hydrogen / 100.0
        o = fuel.oxygen / 100.0
        s = fuel.sulfur / 100.0
        n = fuel.nitrogen / 100.0
        m = fuel.moisture / 100.0

        # O2 required for complete combustion (kg O2/kg fuel)
        # C + O2 -> CO2:  32/12 = 2.667 kg O2/kg C
        # H2 + 0.5*O2 -> H2O:  16/2 = 8 kg O2/kg H2
        # S + O2 -> SO2:  32/32 = 1 kg O2/kg S
        o2_required = 2.667 * c + 8.0 * h + 1.0 * s - o

        # Theoretical air (kg air/kg fuel)
        # Air is 23.2% O2 by mass
        theoretical_air = o2_required / 0.232

        # Molar air requirement (approximation)
        # Average molecular weight of fuel approximation
        theoretical_air_mol = theoretical_air * 1000 / MW_AIR

        # Products of combustion per kg fuel
        co2_produced = (44.0 / 12.0) * c  # 3.667 kg CO2/kg C
        h2o_combustion = (18.0 / 2.0) * h  # 9 kg H2O/kg H2
        h2o_from_moisture = m
        h2o_produced = h2o_combustion + h2o_from_moisture
        so2_produced = (64.0 / 32.0) * s  # 2 kg SO2/kg S
        n2_in_flue = n + 0.768 * theoretical_air  # N2 from fuel + air

        # Maximum theoretical CO2 (volume %, dry basis)
        # At stoichiometric conditions with no excess air
        # CO2_max = (CO2 volume) / (CO2 + N2 + SO2 volumes) * 100
        co2_moles = co2_produced / MW_CO2
        n2_moles = n2_in_flue / MW_N2
        so2_moles = so2_produced / MW_SO2

        total_dry_moles = co2_moles + n2_moles + so2_moles
        co2_max = (co2_moles / total_dry_moles) * 100.0 if total_dry_moles > 0 else 11.8

        result = StoichiometryResult(
            theoretical_air_kg_per_kg_fuel=round(theoretical_air, 4),
            theoretical_air_mol_per_mol_fuel=round(theoretical_air_mol, 4),
            theoretical_o2_kg_per_kg_fuel=round(o2_required, 4),
            co2_produced_kg_per_kg_fuel=round(co2_produced, 4),
            h2o_produced_kg_per_kg_fuel=round(h2o_produced, 4),
            so2_produced_kg_per_kg_fuel=round(so2_produced, 4),
            n2_in_flue_gas_kg_per_kg_fuel=round(n2_in_flue, 4),
            theoretical_co2_max_pct=round(co2_max, 2)
        )

        self._tracker.add_step(
            step_number=step,
            description="Calculate combustion stoichiometry from fuel analysis",
            operation="stoichiometry_calculation",
            inputs={
                "carbon_pct": fuel.carbon,
                "hydrogen_pct": fuel.hydrogen,
                "oxygen_pct": fuel.oxygen,
                "sulfur_pct": fuel.sulfur,
                "nitrogen_pct": fuel.nitrogen,
                "moisture_pct": fuel.moisture
            },
            output_value=theoretical_air,
            output_name="theoretical_air_kg_per_kg_fuel",
            formula="A_th = (2.667*C + 8*H + S - O) / 0.232"
        )

        return result

    def _convert_to_dry_basis(
        self,
        o2_measured: float,
        co2_measured: float,
        basis: str,
        h2o_pct: Optional[float]
    ) -> Tuple[float, float]:
        """
        Convert gas concentrations to dry basis if needed.

        Formula (wet to dry):
            C_dry = C_wet / (1 - H2O/100)
        """
        step = self._next_step()

        if basis == 'dry':
            o2_dry = o2_measured
            co2_dry = co2_measured
            conversion_applied = False
        else:
            # Use measured H2O or estimate
            h2o = h2o_pct if h2o_pct is not None else 10.0
            dry_fraction = 1.0 - (h2o / 100.0)

            o2_dry = o2_measured / dry_fraction
            co2_dry = co2_measured / dry_fraction
            conversion_applied = True

        self._tracker.add_step(
            step_number=step,
            description="Convert gas concentrations to dry basis",
            operation="wet_to_dry_conversion",
            inputs={
                "o2_measured": o2_measured,
                "co2_measured": co2_measured,
                "measurement_basis": basis,
                "h2o_pct": h2o_pct,
                "conversion_applied": conversion_applied
            },
            output_value=o2_dry,
            output_name="o2_pct_dry",
            formula="C_dry = C_wet / (1 - H2O/100)"
        )

        return o2_dry, co2_dry

    def _calculate_excess_air_from_o2(self, o2_dry: float) -> float:
        """
        Calculate excess air percentage from O2 measurement.

        Formula (ASME PTC 4.1):
            EA% = (O2_dry / (20.95 - O2_dry)) * 100

        Derivation:
            At excess air EA, the O2 remaining in flue gas is:
            O2 = 20.95 * EA / (100 + EA)

            Solving for EA:
            EA = O2 * 100 / (20.95 - O2)
        """
        step = self._next_step()

        # Prevent division by zero
        if o2_dry >= O2_IN_AIR_VOL_PCT:
            raise ValueError(
                f"O2 {o2_dry}% cannot exceed atmospheric level {O2_IN_AIR_VOL_PCT}%"
            )

        excess_air = (o2_dry / (O2_IN_AIR_VOL_PCT - o2_dry)) * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate excess air from O2 measurement",
            operation="excess_air_from_o2",
            inputs={
                "o2_pct_dry": o2_dry,
                "o2_in_air_pct": O2_IN_AIR_VOL_PCT
            },
            output_value=excess_air,
            output_name="excess_air_pct",
            formula="EA% = (O2 / (20.95 - O2)) * 100"
        )

        return excess_air

    def _calculate_excess_air_from_co2(
        self,
        co2_dry: float,
        co2_max: float
    ) -> float:
        """
        Calculate excess air percentage from CO2 measurement.

        Formula:
            EA% = ((CO2_max - CO2_meas) / CO2_meas) * 100

        This method is less accurate but provides a cross-check.
        """
        step = self._next_step()

        if co2_dry <= 0:
            excess_air = 0.0
        else:
            excess_air = ((co2_max - co2_dry) / co2_dry) * 100.0

        # Ensure non-negative
        excess_air = max(0.0, excess_air)

        self._tracker.add_step(
            step_number=step,
            description="Calculate excess air from CO2 measurement",
            operation="excess_air_from_co2",
            inputs={
                "co2_pct_dry": co2_dry,
                "co2_max_pct": co2_max
            },
            output_value=excess_air,
            output_name="excess_air_from_co2_pct",
            formula="EA% = ((CO2_max - CO2) / CO2) * 100"
        )

        return excess_air

    def _calculate_lambda_ratio(self, excess_air: float) -> float:
        """
        Calculate air-fuel equivalence ratio (lambda).

        Formula:
            lambda = (100 + EA) / 100 = 1 + EA/100

        lambda = 1.0: Stoichiometric
        lambda > 1.0: Lean (excess air)
        lambda < 1.0: Rich (excess fuel)
        """
        step = self._next_step()

        lambda_ratio = 1.0 + (excess_air / 100.0)

        self._tracker.add_step(
            step_number=step,
            description="Calculate air-fuel equivalence ratio (lambda)",
            operation="lambda_calculation",
            inputs={"excess_air_pct": excess_air},
            output_value=lambda_ratio,
            output_name="lambda_ratio",
            formula="lambda = 1 + EA/100"
        )

        return lambda_ratio

    def _calculate_actual_air(
        self,
        theoretical_air: float,
        excess_air: float
    ) -> float:
        """
        Calculate actual air requirement.

        Formula:
            A_actual = A_theoretical * (1 + EA/100)
        """
        step = self._next_step()

        actual_air = theoretical_air * (1.0 + excess_air / 100.0)

        self._tracker.add_step(
            step_number=step,
            description="Calculate actual air requirement",
            operation="actual_air_calculation",
            inputs={
                "theoretical_air_kg_per_kg_fuel": theoretical_air,
                "excess_air_pct": excess_air
            },
            output_value=actual_air,
            output_name="actual_air_kg_per_kg_fuel",
            formula="A_actual = A_th * (1 + EA/100)"
        )

        return actual_air

    def _calculate_water_vapor_content(
        self,
        fuel: FuelComposition,
        excess_air: float,
        ambient_temp_c: float,
        relative_humidity: float,
        h2o_measured: Optional[float]
    ) -> float:
        """
        Calculate water vapor content in flue gas.

        Sources:
        1. Combustion of hydrogen in fuel: 9 kg H2O per kg H2
        2. Moisture in fuel
        3. Humidity in combustion air

        If measured value provided, use it directly.
        """
        step = self._next_step()

        if h2o_measured is not None:
            h2o_pct = h2o_measured
            calculation_method = "measured"
        else:
            # Estimate water vapor from fuel analysis
            # Typical range: 8-15% for hydrocarbon fuels

            # Hydrogen contribution (dominant for natural gas)
            h2o_from_h2 = 9.0 * (fuel.hydrogen / 100.0)

            # Moisture in fuel
            moisture_fraction = fuel.moisture / 100.0

            # Humidity in air (simplified)
            # At 25 deg C and 50% RH, ~0.01 kg H2O/kg dry air
            humidity_contribution = 0.01 * (relative_humidity / 50.0)

            # Total water (kg per kg fuel)
            total_h2o_kg = h2o_from_h2 + moisture_fraction + humidity_contribution * 17.0

            # Convert to volume percent (approximate)
            # Natural gas: ~10-12% H2O
            # Oil: ~8-10% H2O
            # Coal: ~6-8% H2O
            if fuel.hydrogen > 15:  # Gas
                h2o_pct = 10.0 + (total_h2o_kg - 2.0) * 2.0
            elif fuel.hydrogen > 10:  # Oil
                h2o_pct = 8.0 + (total_h2o_kg - 1.5) * 2.0
            else:  # Coal/Biomass
                h2o_pct = 6.0 + (total_h2o_kg - 1.0) * 2.0

            h2o_pct = max(5.0, min(18.0, h2o_pct))  # Clamp to reasonable range
            calculation_method = "estimated"

        self._tracker.add_step(
            step_number=step,
            description="Calculate water vapor content in flue gas",
            operation="h2o_calculation",
            inputs={
                "fuel_hydrogen_pct": fuel.hydrogen,
                "fuel_moisture_pct": fuel.moisture,
                "relative_humidity_pct": relative_humidity,
                "h2o_measured": h2o_measured,
                "calculation_method": calculation_method
            },
            output_value=h2o_pct,
            output_name="h2o_pct",
            formula="H2O from hydrogen combustion + moisture + humidity"
        )

        return h2o_pct

    def _calculate_n2_content_dry(
        self,
        o2_dry: float,
        co2_dry: float
    ) -> float:
        """
        Calculate nitrogen content in dry flue gas.

        For simplified analysis, assume flue gas is mostly CO2, O2, and N2:
            N2% = 100 - O2% - CO2% - trace gases
        """
        step = self._next_step()

        # Assume trace gases (Ar, SO2, NOx, CO) are < 1%
        n2_dry = 100.0 - o2_dry - co2_dry - 1.0
        n2_dry = max(70.0, min(85.0, n2_dry))  # Clamp to reasonable range

        self._tracker.add_step(
            step_number=step,
            description="Calculate N2 content (by difference)",
            operation="n2_by_difference",
            inputs={
                "o2_pct_dry": o2_dry,
                "co2_pct_dry": co2_dry,
                "trace_gases_pct": 1.0
            },
            output_value=n2_dry,
            output_name="n2_pct_dry",
            formula="N2% = 100 - O2% - CO2% - traces"
        )

        return n2_dry

    def _convert_to_wet_basis(
        self,
        o2_dry: float,
        co2_dry: float,
        n2_dry: float,
        h2o_pct: float
    ) -> Tuple[float, float, float]:
        """
        Convert dry basis concentrations to wet basis.

        Formula:
            C_wet = C_dry * (1 - H2O/100)
        """
        step = self._next_step()

        wet_factor = 1.0 - (h2o_pct / 100.0)

        o2_wet = o2_dry * wet_factor
        co2_wet = co2_dry * wet_factor
        n2_wet = n2_dry * wet_factor

        self._tracker.add_step(
            step_number=step,
            description="Convert to wet basis concentrations",
            operation="dry_to_wet_conversion",
            inputs={
                "o2_pct_dry": o2_dry,
                "co2_pct_dry": co2_dry,
                "n2_pct_dry": n2_dry,
                "h2o_pct": h2o_pct,
                "wet_factor": wet_factor
            },
            output_value=o2_wet,
            output_name="o2_pct_wet",
            formula="C_wet = C_dry * (1 - H2O/100)"
        )

        return o2_wet, co2_wet, n2_wet

    def _calculate_molecular_weight(
        self,
        o2_dry: float,
        co2_dry: float,
        n2_dry: float,
        h2o_pct: float
    ) -> Tuple[float, float]:
        """
        Calculate molecular weight of flue gas mixture.

        Formula (mixing rule):
            MW_mix = sum(y_i * MW_i)

        where y_i is the mole fraction of component i.
        """
        step = self._next_step()

        # Dry basis molecular weight
        # Assume remainder is Ar (0.93%) for accuracy
        ar_dry = 0.93
        other_dry = 100.0 - o2_dry - co2_dry - n2_dry - ar_dry

        mw_dry = (
            (o2_dry / 100.0) * MW_O2 +
            (co2_dry / 100.0) * MW_CO2 +
            (n2_dry / 100.0) * MW_N2 +
            (ar_dry / 100.0) * MW_AR +
            (other_dry / 100.0) * MW_N2  # Approximate remainder as N2
        )

        # Wet basis molecular weight
        wet_factor = 1.0 - (h2o_pct / 100.0)
        mw_wet = (
            mw_dry * wet_factor +
            (h2o_pct / 100.0) * MW_H2O
        )

        self._tracker.add_step(
            step_number=step,
            description="Calculate molecular weight of flue gas mixture",
            operation="molecular_weight_mixing",
            inputs={
                "o2_pct_dry": o2_dry,
                "co2_pct_dry": co2_dry,
                "n2_pct_dry": n2_dry,
                "h2o_pct": h2o_pct,
                "MW_O2": MW_O2,
                "MW_CO2": MW_CO2,
                "MW_N2": MW_N2,
                "MW_H2O": MW_H2O
            },
            output_value=mw_dry,
            output_name="molecular_weight_dry",
            formula="MW_mix = sum(y_i * MW_i)"
        )

        return mw_dry, mw_wet

    def _calculate_density(
        self,
        mw_dry: float,
        mw_wet: float,
        temp_c: float,
        pressure_kpa: float
    ) -> Tuple[float, float, float]:
        """
        Calculate flue gas density.

        Formula (ideal gas law):
            rho = P * MW / (R * T)

        At STP (0 deg C, 101.325 kPa):
            rho_stp = MW / 22.414 (kg/Nm3)
        """
        step = self._next_step()

        # Density at STP (kg/Nm3)
        density_dry_stp = mw_dry / (MOLAR_VOLUME_STP * 1000.0)  # g/mol to kg/mol
        density_wet_stp = mw_wet / (MOLAR_VOLUME_STP * 1000.0)

        # Density at actual conditions
        temp_k = temp_c + 273.15

        # rho_actual = rho_stp * (T_stp/T) * (P/P_stp)
        density_actual = density_wet_stp * (STANDARD_TEMP_K / temp_k) * (pressure_kpa / STANDARD_PRESSURE_KPA)

        self._tracker.add_step(
            step_number=step,
            description="Calculate flue gas density",
            operation="density_ideal_gas",
            inputs={
                "molecular_weight_dry": mw_dry,
                "molecular_weight_wet": mw_wet,
                "temp_c": temp_c,
                "pressure_kpa": pressure_kpa
            },
            output_value=density_dry_stp,
            output_name="density_dry_kg_m3",
            formula="rho = MW / 22.414 (at STP)"
        )

        return density_dry_stp, density_wet_stp, density_actual

    def _calculate_specific_heat(
        self,
        o2_dry: float,
        co2_dry: float,
        n2_dry: float,
        h2o_pct: float,
        temp_c: float
    ) -> Tuple[float, float]:
        """
        Calculate specific heat of flue gas using JANAF data.

        Uses Shomate equation coefficients for temperature-dependent Cp.

        Formula (mixing rule):
            Cp_mix = sum(y_i * Cp_i)
        """
        step = self._next_step()

        temp_k = temp_c + 273.15

        # Get Cp for each component at temperature
        cp_o2 = JANAFData.get_cp_molar('O2', temp_k)
        cp_co2 = JANAFData.get_cp_molar('CO2', temp_k)
        cp_n2 = JANAFData.get_cp_molar('N2', temp_k)
        cp_h2o = JANAFData.get_cp_molar('H2O', temp_k)

        # Dry basis molar Cp
        cp_dry_molar = (
            (o2_dry / 100.0) * cp_o2 +
            (co2_dry / 100.0) * cp_co2 +
            (n2_dry / 100.0) * cp_n2
        )

        # Wet basis molar Cp
        wet_factor = 1.0 - (h2o_pct / 100.0)
        cp_wet_molar = (
            cp_dry_molar * wet_factor +
            (h2o_pct / 100.0) * cp_h2o
        )

        # Convert to mass basis using molecular weights
        mw_dry = (
            (o2_dry / 100.0) * MW_O2 +
            (co2_dry / 100.0) * MW_CO2 +
            (n2_dry / 100.0) * MW_N2
        )
        mw_wet = mw_dry * wet_factor + (h2o_pct / 100.0) * MW_H2O

        # kJ/(kg*K)
        cp_dry_mass = cp_dry_molar / mw_dry
        cp_wet_mass = cp_wet_molar / mw_wet

        self._tracker.add_step(
            step_number=step,
            description="Calculate specific heat using JANAF thermochemical data",
            operation="specific_heat_janaf",
            inputs={
                "temp_k": temp_k,
                "o2_pct_dry": o2_dry,
                "co2_pct_dry": co2_dry,
                "n2_pct_dry": n2_dry,
                "h2o_pct": h2o_pct,
                "cp_o2_j_mol_k": cp_o2,
                "cp_co2_j_mol_k": cp_co2,
                "cp_n2_j_mol_k": cp_n2,
                "cp_h2o_j_mol_k": cp_h2o
            },
            output_value=cp_dry_mass,
            output_name="cp_dry_kj_kg_k",
            formula="Cp = sum(y_i * Cp_i) / MW_mix (JANAF data)"
        )

        return cp_dry_mass, cp_wet_mass

    def _calculate_water_dew_point(
        self,
        h2o_pct: float,
        pressure_kpa: float
    ) -> float:
        """
        Calculate water vapor dew point temperature.

        Uses Magnus formula for saturation vapor pressure:
            P_sat = A * exp(B * T / (C + T))

        Inversion for dew point:
            T_dp = C * ln(P_w/A) / (B - ln(P_w/A))

        where P_w is the partial pressure of water vapor.
        """
        step = self._next_step()

        # Partial pressure of water vapor (hPa)
        p_water = (h2o_pct / 100.0) * pressure_kpa * 10.0  # kPa to hPa

        # Apply Magnus formula (inversion)
        if p_water <= 0:
            dew_point = -40.0  # Very low dew point
        else:
            ln_ratio = math.log(p_water / MAGNUS_A)
            dew_point = MAGNUS_C * ln_ratio / (MAGNUS_B - ln_ratio)

        # Clamp to physical range
        dew_point = max(-40.0, min(100.0, dew_point))

        self._tracker.add_step(
            step_number=step,
            description="Calculate water vapor dew point (Magnus formula)",
            operation="water_dew_point",
            inputs={
                "h2o_pct": h2o_pct,
                "pressure_kpa": pressure_kpa,
                "p_water_hpa": p_water,
                "magnus_a": MAGNUS_A,
                "magnus_b": MAGNUS_B,
                "magnus_c": MAGNUS_C
            },
            output_value=dew_point,
            output_name="water_dew_point_c",
            formula="T_dp = C * ln(Pw/A) / (B - ln(Pw/A))"
        )

        return dew_point

    def _calculate_acid_dew_point(
        self,
        h2o_pct: float,
        so2_ppm: float,
        flue_gas_temp_c: float
    ) -> float:
        """
        Calculate sulfuric acid dew point temperature.

        Uses Pierce correlation (1977):
            T_adp(K) = 1000 / (2.276 - 0.0294*ln(pH2O) - 0.0858*ln(pSO3) + 0.0062*ln(pH2O*pSO3))

        where pH2O and pSO3 are in mmHg.

        Simplified estimation when SO2 is measured instead of SO3:
        - Assume 1-5% SO2 to SO3 conversion
        """
        step = self._next_step()

        if so2_ppm <= 0 or h2o_pct <= 0:
            # No sulfur, no acid dew point concern
            acid_dp = 0.0
        else:
            # Estimate SO3 from SO2 (typical 2% conversion)
            so3_ppm = so2_ppm * 0.02

            # Convert to partial pressures (mmHg)
            # At 101.325 kPa = 760 mmHg
            p_h2o_mmhg = (h2o_pct / 100.0) * 760.0
            p_so3_mmhg = (so3_ppm / 1e6) * 760.0

            # Ensure minimum values to avoid log(0)
            p_h2o_mmhg = max(p_h2o_mmhg, 0.1)
            p_so3_mmhg = max(p_so3_mmhg, 1e-6)

            # Pierce correlation
            ln_h2o = math.log(p_h2o_mmhg)
            ln_so3 = math.log(p_so3_mmhg)
            ln_product = math.log(p_h2o_mmhg * p_so3_mmhg)

            denominator = 2.276 - 0.0294 * ln_h2o - 0.0858 * ln_so3 + 0.0062 * ln_product

            if denominator <= 0:
                acid_dp = 150.0  # Default high value
            else:
                acid_dp_k = 1000.0 / denominator
                acid_dp = acid_dp_k - 273.15

        # Clamp to physical range (typical 100-160 deg C)
        acid_dp = max(80.0, min(180.0, acid_dp))

        self._tracker.add_step(
            step_number=step,
            description="Calculate sulfuric acid dew point (Pierce correlation)",
            operation="acid_dew_point",
            inputs={
                "h2o_pct": h2o_pct,
                "so2_ppm": so2_ppm,
                "so3_ppm_estimated": so2_ppm * 0.02,
                "so2_to_so3_conversion": 0.02
            },
            output_value=acid_dp,
            output_name="acid_dew_point_c",
            formula="T_adp(K) = 1000 / (2.276 - 0.0294*ln(pH2O) - 0.0858*ln(pSO3) + 0.0062*ln(pH2O*pSO3))"
        )

        return acid_dp

    def _calculate_flue_gas_volume(
        self,
        stoichiometry: StoichiometryResult,
        excess_air: float
    ) -> Tuple[float, float]:
        """
        Calculate flue gas volume per kg of fuel.

        Formula:
            V_fg_dry = V_stoich * (1 + EA/100)
            V_fg_wet = V_fg_dry + V_H2O
        """
        step = self._next_step()

        # Estimate stoichiometric flue gas volume from products
        # CO2 + N2 from combustion and air
        co2_moles = stoichiometry.co2_produced_kg_per_kg_fuel / MW_CO2 * 1000
        n2_moles = stoichiometry.n2_in_flue_gas_kg_per_kg_fuel / MW_N2 * 1000
        h2o_moles = stoichiometry.h2o_produced_kg_per_kg_fuel / MW_H2O * 1000

        # Volume at STP (Nm3)
        vol_stoich_dry = (co2_moles + n2_moles) * MOLAR_VOLUME_STP / 1000
        vol_h2o = h2o_moles * MOLAR_VOLUME_STP / 1000

        # Add excess air contribution
        excess_air_vol = stoichiometry.theoretical_air_kg_per_kg_fuel * (excess_air / 100) / MW_AIR * 1000 * MOLAR_VOLUME_STP / 1000

        vol_dry = vol_stoich_dry + excess_air_vol
        vol_wet = vol_dry + vol_h2o

        self._tracker.add_step(
            step_number=step,
            description="Calculate flue gas volume per kg fuel",
            operation="flue_gas_volume",
            inputs={
                "co2_moles_per_kg_fuel": co2_moles,
                "n2_moles_per_kg_fuel": n2_moles,
                "h2o_moles_per_kg_fuel": h2o_moles,
                "excess_air_pct": excess_air,
                "molar_volume_stp": MOLAR_VOLUME_STP
            },
            output_value=vol_dry,
            output_name="flue_gas_volume_dry_nm3_per_kg_fuel",
            formula="V_fg = sum(n_i * Vm) * (1 + EA/100)"
        )

        return vol_dry, vol_wet


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_excess_air_from_o2(o2_pct_dry: float) -> float:
    """
    Calculate excess air from O2 measurement (standalone function).

    Formula (ASME PTC 4.1):
        EA% = (O2 / (20.95 - O2)) * 100

    Args:
        o2_pct_dry: O2 concentration on dry basis (vol %)

    Returns:
        Excess air percentage (%)

    Example:
        >>> excess_air = calculate_excess_air_from_o2(3.5)
        >>> print(f"Excess Air: {excess_air:.1f}%")  # ~20.1%
    """
    if o2_pct_dry < 0 or o2_pct_dry >= O2_IN_AIR_VOL_PCT:
        raise ValueError(f"O2 must be in range 0-{O2_IN_AIR_VOL_PCT}%, got {o2_pct_dry}%")

    return (o2_pct_dry / (O2_IN_AIR_VOL_PCT - o2_pct_dry)) * 100.0


def calculate_excess_air_from_co2(
    co2_pct_dry: float,
    co2_max_pct: float = 11.8
) -> float:
    """
    Calculate excess air from CO2 measurement (standalone function).

    Formula:
        EA% = ((CO2_max - CO2_meas) / CO2_meas) * 100

    Args:
        co2_pct_dry: CO2 concentration on dry basis (vol %)
        co2_max_pct: Maximum theoretical CO2 (default 11.8% for natural gas)

    Returns:
        Excess air percentage (%)
    """
    if co2_pct_dry <= 0:
        return 0.0

    return max(0.0, ((co2_max_pct - co2_pct_dry) / co2_pct_dry) * 100.0)


def convert_wet_to_dry(concentration_wet: float, h2o_pct: float) -> float:
    """
    Convert gas concentration from wet to dry basis.

    Formula:
        C_dry = C_wet / (1 - H2O/100)

    Args:
        concentration_wet: Concentration on wet basis (%)
        h2o_pct: Water vapor content (vol %)

    Returns:
        Concentration on dry basis (%)
    """
    if h2o_pct < 0 or h2o_pct >= 100:
        raise ValueError(f"H2O must be in range 0-100%, got {h2o_pct}%")

    return concentration_wet / (1.0 - h2o_pct / 100.0)


def convert_dry_to_wet(concentration_dry: float, h2o_pct: float) -> float:
    """
    Convert gas concentration from dry to wet basis.

    Formula:
        C_wet = C_dry * (1 - H2O/100)

    Args:
        concentration_dry: Concentration on dry basis (%)
        h2o_pct: Water vapor content (vol %)

    Returns:
        Concentration on wet basis (%)
    """
    if h2o_pct < 0 or h2o_pct >= 100:
        raise ValueError(f"H2O must be in range 0-100%, got {h2o_pct}%")

    return concentration_dry * (1.0 - h2o_pct / 100.0)


def calculate_water_dew_point(h2o_pct: float, pressure_kpa: float = 101.325) -> float:
    """
    Calculate water vapor dew point using Magnus formula.

    Args:
        h2o_pct: Water vapor content (vol %)
        pressure_kpa: Total pressure (kPa)

    Returns:
        Dew point temperature (deg C)
    """
    p_water = (h2o_pct / 100.0) * pressure_kpa * 10.0  # kPa to hPa

    if p_water <= 0:
        return -40.0

    ln_ratio = math.log(p_water / MAGNUS_A)
    return MAGNUS_C * ln_ratio / (MAGNUS_B - ln_ratio)


def calculate_molecular_weight(
    o2_pct: float,
    co2_pct: float,
    n2_pct: float,
    h2o_pct: float = 0.0
) -> float:
    """
    Calculate molecular weight of flue gas mixture.

    Args:
        o2_pct: O2 concentration (vol %)
        co2_pct: CO2 concentration (vol %)
        n2_pct: N2 concentration (vol %)
        h2o_pct: H2O concentration (vol %, default 0 for dry)

    Returns:
        Molecular weight (g/mol)
    """
    total = o2_pct + co2_pct + n2_pct + h2o_pct

    mw = (
        (o2_pct / total) * MW_O2 +
        (co2_pct / total) * MW_CO2 +
        (n2_pct / total) * MW_N2 +
        (h2o_pct / total) * MW_H2O
    )

    return mw


def get_specific_heat(species: str, temp_c: float) -> float:
    """
    Get specific heat of a flue gas species at temperature (JANAF data).

    Args:
        species: Species name ('N2', 'O2', 'CO2', 'H2O', etc.)
        temp_c: Temperature (deg C)

    Returns:
        Specific heat Cp (kJ/kg-K)
    """
    temp_k = temp_c + 273.15
    mw_map = {
        'N2': MW_N2, 'O2': MW_O2, 'CO2': MW_CO2,
        'H2O': MW_H2O, 'CO': MW_CO, 'SO2': MW_SO2
    }

    mw = mw_map.get(species, MW_N2)
    return JANAFData.get_cp_mass(species, temp_k, mw)
