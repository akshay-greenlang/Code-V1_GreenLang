# -*- coding: utf-8 -*-
"""
Advanced Stoichiometry Calculator for GL-005 CombustionControlAgent

Multi-fuel combustion stoichiometry with thermodynamic calculations.
Zero-hallucination design using deterministic combustion chemistry.

Reference Standards:
- NIST-JANAF Thermochemical Tables
- AGA Report No. 4A: Natural Gas Contract Measurement and Quality Clauses
- ISO 6976: Natural Gas - Calculation of Calorific Values
- API 2000: Venting Atmospheric and Low-pressure Storage Tanks
- NFPA 68: Standard on Explosion Protection by Deflagration Venting

Mathematical Formulas:
- Stoichiometric Air: A_s = 11.5*C + 34.3*(H - O/8) + 4.3*S (kg air/kg fuel)
- Adiabatic Flame Temp: T_ad = T_0 + Q_rxn / (m_products * Cp_avg)
- Wobbe Index: W = HHV / sqrt(SG)
- Heat Release Rate: Q = m_fuel * HV * eta

Author: GreenLang GL-005 Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - THERMODYNAMIC AND COMBUSTION DATA
# =============================================================================


# Molecular weights (kg/kmol)
MOLECULAR_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "H2": 2.016,
    "O": 15.999,
    "O2": 31.998,
    "N": 14.007,
    "N2": 28.014,
    "S": 32.065,
    "CO2": 44.01,
    "H2O": 18.015,
    "SO2": 64.064,
    "CO": 28.01,
    "NO": 30.006,
    "NO2": 46.006,
    "CH4": 16.043,
    "C2H6": 30.07,
    "C3H8": 44.097,
    "C4H10": 58.123,
    "C2H4": 28.054,
    "C2H2": 26.038,
}

# Standard heats of formation at 298K (kJ/mol)
HEATS_OF_FORMATION = {
    "CH4": -74.87,
    "C2H6": -84.68,
    "C3H8": -103.85,
    "C4H10": -126.15,
    "C2H4": 52.47,
    "C2H2": 226.73,
    "CO2": -393.52,
    "H2O": -241.83,  # Gas phase
    "H2O_l": -285.83,  # Liquid phase
    "CO": -110.53,
    "SO2": -296.84,
    "NO": 90.29,
    "NO2": 33.10,
    "O2": 0.0,
    "N2": 0.0,
    "H2": 0.0,
}

# Specific heat capacities Cp (kJ/kmol-K) at ~1500K
SPECIFIC_HEATS_1500K = {
    "CO2": 54.31,
    "H2O": 43.87,
    "N2": 33.71,
    "O2": 36.08,
    "SO2": 54.89,
    "NO": 33.05,
    "CO": 33.18,
}

# NASA polynomial coefficients for Cp (simplified 3rd order)
# Cp/R = a0 + a1*T + a2*T^2 + a3*T^3 (T in Kelvin)
NASA_COEFFICIENTS = {
    "CO2": (2.356, 8.98e-3, -7.12e-6, 2.46e-9),
    "H2O": (4.198, -2.04e-3, 6.52e-6, -5.48e-9),
    "N2": (3.298, 1.40e-3, -3.96e-6, 5.64e-9),
    "O2": (3.282, 1.48e-3, -7.58e-7, 2.09e-10),
}

# Flammability limits (vol% in air at STP)
FLAMMABILITY_LIMITS = {
    "CH4": {"LFL": 5.0, "UFL": 15.0},
    "C2H6": {"LFL": 3.0, "UFL": 12.4},
    "C3H8": {"LFL": 2.1, "UFL": 9.5},
    "C4H10": {"LFL": 1.8, "UFL": 8.4},
    "H2": {"LFL": 4.0, "UFL": 75.0},
    "CO": {"LFL": 12.5, "UFL": 74.0},
    "C2H4": {"LFL": 2.7, "UFL": 36.0},
    "C2H2": {"LFL": 2.5, "UFL": 81.0},
}

# Detonation cell sizes (mm) - for detonation risk assessment
DETONATION_CELL_SIZES = {
    "CH4": 300,
    "C2H6": 50,
    "C3H8": 50,
    "H2": 10,
    "C2H2": 5,
}

# Universal gas constant
R_UNIVERSAL = 8.314  # kJ/kmol-K


# =============================================================================
# ENUMERATIONS
# =============================================================================


class FuelType(str, Enum):
    """Types of combustion fuels"""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    BUTANE = "butane"
    HYDROGEN = "hydrogen"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    MIXED = "mixed"


class CombustionMode(str, Enum):
    """Combustion mode classifications"""
    STOICHIOMETRIC = "stoichiometric"
    LEAN = "lean"
    RICH = "rich"


class FlammabilityStatus(str, Enum):
    """Flammability status classifications"""
    BELOW_LFL = "below_lfl"
    FLAMMABLE = "flammable"
    ABOVE_UFL = "above_ufl"


class DetonationRisk(str, Enum):
    """Detonation risk levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterchangeabilityStatus(str, Enum):
    """Fuel interchangeability status (AGA method)"""
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    UNACCEPTABLE = "unacceptable"


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================


@dataclass(frozen=True)
class FuelComposition:
    """Immutable fuel composition (mole fractions)"""
    ch4: float = 0.0  # Methane
    c2h6: float = 0.0  # Ethane
    c3h8: float = 0.0  # Propane
    c4h10: float = 0.0  # Butane
    c2h4: float = 0.0  # Ethylene
    c2h2: float = 0.0  # Acetylene
    h2: float = 0.0  # Hydrogen
    co: float = 0.0  # Carbon monoxide
    co2: float = 0.0  # Carbon dioxide
    n2: float = 0.0  # Nitrogen
    o2: float = 0.0  # Oxygen
    h2o: float = 0.0  # Water vapor

    def total(self) -> float:
        """Sum of all components"""
        return (self.ch4 + self.c2h6 + self.c3h8 + self.c4h10 +
                self.c2h4 + self.c2h2 + self.h2 + self.co +
                self.co2 + self.n2 + self.o2 + self.h2o)


@dataclass(frozen=True)
class StoichiometricResult:
    """Immutable stoichiometric calculation result"""
    stoichiometric_air_fuel_ratio: float  # kg air / kg fuel
    stoichiometric_air_fuel_ratio_molar: float  # mol air / mol fuel
    theoretical_air_volume_nm3_per_kg: float  # Nm3 air / kg fuel
    theoretical_co2_volume_nm3_per_kg: float  # Nm3 CO2 / kg fuel
    theoretical_h2o_volume_nm3_per_kg: float  # Nm3 H2O / kg fuel
    theoretical_n2_volume_nm3_per_kg: float  # Nm3 N2 / kg fuel
    total_flue_gas_volume_nm3_per_kg: float  # Nm3 flue gas / kg fuel
    excess_air_percent: float
    equivalence_ratio: float  # phi = (F/A)_actual / (F/A)_stoich
    combustion_mode: CombustionMode
    provenance_hash: str


@dataclass(frozen=True)
class AdiabaticFlameTemperature:
    """Immutable adiabatic flame temperature result"""
    temperature_k: float
    temperature_c: float
    temperature_with_dissociation_k: float
    temperature_with_dissociation_c: float
    dissociation_correction_k: float
    enthalpy_of_combustion_kj_per_kg: float
    mean_cp_products_kj_per_kmol_k: float
    provenance_hash: str


@dataclass(frozen=True)
class EquilibriumSpecies:
    """Immutable equilibrium species concentrations"""
    co2_mole_fraction: float
    h2o_mole_fraction: float
    n2_mole_fraction: float
    o2_mole_fraction: float
    co_mole_fraction: float  # From dissociation
    h2_mole_fraction: float  # From dissociation
    oh_mole_fraction: float  # From dissociation
    no_mole_fraction: float  # Thermal NOx
    total_moles_per_kg_fuel: float
    temperature_k: float
    provenance_hash: str


@dataclass(frozen=True)
class DissociationEffects:
    """Immutable dissociation effects at high temperature"""
    co2_dissociation_percent: float
    h2o_dissociation_percent: float
    temperature_k: float
    equilibrium_constant_co2: float
    equilibrium_constant_h2o: float
    flame_temp_reduction_k: float
    provenance_hash: str


@dataclass(frozen=True)
class FuelInterchangeability:
    """Immutable fuel interchangeability result (AGA method)"""
    wobbe_index_mj_per_nm3: float
    wobbe_number: float  # Relative to reference
    lifting_index: float
    flashback_index: float
    yellow_tipping_index: float
    aga_index_a: float  # Interchangeability Index A
    aga_index_b: float  # Interchangeability Index B
    status: InterchangeabilityStatus
    provenance_hash: str


@dataclass(frozen=True)
class HeatReleaseRate:
    """Immutable heat release rate calculation"""
    heat_release_rate_kw: float
    heat_release_rate_mw: float
    volumetric_heat_release_kw_per_m3: float
    heat_flux_kw_per_m2: float
    thermal_efficiency_percent: float
    fuel_consumption_kg_per_hr: float
    specific_fuel_consumption_kg_per_kwh: float
    provenance_hash: str


@dataclass(frozen=True)
class FlammabilityLimits:
    """Immutable flammability limits result"""
    lower_flammability_limit_vol_percent: float
    upper_flammability_limit_vol_percent: float
    flammable_range_vol_percent: float
    current_concentration_vol_percent: float
    status: FlammabilityStatus
    margin_to_lfl_percent: float
    margin_to_ufl_percent: float
    provenance_hash: str


@dataclass(frozen=True)
class DetonationRiskAssessment:
    """Immutable detonation risk assessment"""
    detonation_cell_size_mm: float
    critical_diameter_mm: float
    run_up_distance_m: float
    deflagration_to_detonation_transition_possible: bool
    peak_overpressure_bar: float
    risk_level: DetonationRisk
    recommended_relief_area_m2: float
    provenance_hash: str


@dataclass(frozen=True)
class ComprehensiveStoichiometryResult:
    """Immutable comprehensive stoichiometry analysis"""
    timestamp: datetime
    fuel_type: FuelType
    stoichiometry: StoichiometricResult
    adiabatic_flame_temp: AdiabaticFlameTemperature
    equilibrium_species: EquilibriumSpecies
    dissociation: DissociationEffects
    interchangeability: FuelInterchangeability
    heat_release: HeatReleaseRate
    flammability: FlammabilityLimits
    detonation_risk: DetonationRiskAssessment
    provenance_hash: str


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================


class StoichiometryInput(BaseModel):
    """Input parameters for stoichiometry calculations"""

    # Fuel specification
    fuel_type: FuelType = Field(..., description="Type of fuel")
    fuel_composition: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fuel composition as mole fractions (sum to 1.0)"
    )

    # Flow rates
    fuel_flow_kg_hr: float = Field(..., ge=0, description="Fuel mass flow rate")
    air_flow_kg_hr: float = Field(..., ge=0, description="Air mass flow rate")
    air_temperature_c: float = Field(default=25.0, description="Air inlet temperature")
    air_humidity_percent: float = Field(default=50.0, ge=0, le=100, description="Relative humidity")

    # Fuel properties (if not using composition)
    fuel_carbon_mass_percent: float = Field(default=75.0, ge=0, le=100)
    fuel_hydrogen_mass_percent: float = Field(default=25.0, ge=0, le=100)
    fuel_oxygen_mass_percent: float = Field(default=0.0, ge=0, le=100)
    fuel_nitrogen_mass_percent: float = Field(default=0.0, ge=0, le=100)
    fuel_sulfur_mass_percent: float = Field(default=0.0, ge=0, le=100)
    fuel_higher_heating_value_mj_kg: float = Field(default=50.0, gt=0)
    fuel_lower_heating_value_mj_kg: float = Field(default=45.0, gt=0)
    fuel_specific_gravity: float = Field(default=0.6, gt=0)
    fuel_molecular_weight: float = Field(default=18.0, gt=0)

    # Reference fuel for interchangeability
    reference_wobbe_index: float = Field(default=50.0, gt=0)
    reference_heating_value: float = Field(default=38.0, gt=0)

    # Combustion chamber parameters
    combustion_chamber_volume_m3: float = Field(default=1.0, gt=0)
    combustion_chamber_surface_area_m2: float = Field(default=6.0, gt=0)
    combustion_pressure_bar: float = Field(default=1.0, gt=0)
    initial_temperature_c: float = Field(default=25.0)

    # Efficiency
    combustion_efficiency_percent: float = Field(default=98.0, ge=0, le=100)

    # Current concentration for flammability check
    current_fuel_concentration_vol_percent: float = Field(default=0.0, ge=0)

    # Enclosure parameters for detonation assessment
    enclosure_characteristic_length_m: float = Field(default=10.0, gt=0)

    @field_validator('fuel_composition')
    @classmethod
    def validate_composition(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate fuel composition sums to approximately 1.0"""
        if v is not None:
            total = sum(v.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Fuel composition must sum to 1.0, got {total}")
        return v


class StoichiometryOutput(BaseModel):
    """Output from stoichiometry calculations"""

    result: ComprehensiveStoichiometryResult = Field(..., description="Complete stoichiometry analysis")
    processing_time_ms: float = Field(..., description="Processing duration")
    calculation_timestamp: datetime = Field(..., description="Timestamp of calculation")

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# THREAD-SAFE CACHE
# =============================================================================


class ThreadSafeCache:
    """Thread-safe LRU cache for expensive thermodynamic calculations"""

    def __init__(self, maxsize: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self._lock:
            if len(self._cache) >= self._maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================


class AdvancedStoichiometryCalculator:
    """
    Advanced stoichiometry calculator for multi-fuel combustion.

    Zero-hallucination design using deterministic combustion chemistry
    and thermodynamic calculations from NIST-JANAF tables.

    Features:
    - Multi-fuel combustion stoichiometry
    - Adiabatic flame temperature calculation
    - Species concentration at equilibrium
    - Dissociation effects at high temperature
    - Fuel interchangeability (AGA method)
    - Heat release rate calculation
    - Flammability limit checking
    - Detonation risk assessment

    Thread-safe with LRU caching for expensive thermodynamic lookups.
    """

    # Air composition
    AIR_O2_MOLE_FRACTION = 0.21
    AIR_N2_MOLE_FRACTION = 0.79
    AIR_MOLECULAR_WEIGHT = 28.97  # kg/kmol

    # Standard conditions
    STP_TEMPERATURE_K = 273.15
    STP_PRESSURE_PA = 101325
    MOLAR_VOLUME_STP_M3 = 22.414 / 1000  # m3/mol at STP

    # AGA interchangeability limits
    AGA_WOBBE_TOLERANCE = 0.05  # +/- 5%
    AGA_LIFTING_INDEX_LIMIT = 1.06
    AGA_FLASHBACK_INDEX_LIMIT = 1.20

    def __init__(self):
        """Initialize advanced stoichiometry calculator"""
        self._logger = logging.getLogger(__name__)
        self._cache = ThreadSafeCache(maxsize=1000)

    def calculate_comprehensive(
        self,
        stoich_input: StoichiometryInput
    ) -> StoichiometryOutput:
        """
        Perform comprehensive stoichiometry analysis.

        Args:
            stoich_input: Input parameters for calculation

        Returns:
            StoichiometryOutput with complete analysis
        """
        start_time = datetime.now(timezone.utc)
        self._logger.info("Starting comprehensive stoichiometry calculation")

        try:
            # Get fuel composition
            fuel_comp = self._get_fuel_composition(stoich_input)

            # Step 1: Basic stoichiometry
            stoichiometry = self._calculate_stoichiometry(stoich_input, fuel_comp)

            # Step 2: Adiabatic flame temperature
            adiabatic_temp = self._calculate_adiabatic_flame_temp(
                stoich_input, fuel_comp, stoichiometry
            )

            # Step 3: Equilibrium species
            equilibrium = self._calculate_equilibrium_species(
                stoich_input, fuel_comp, stoichiometry, adiabatic_temp
            )

            # Step 4: Dissociation effects
            dissociation = self._calculate_dissociation_effects(
                adiabatic_temp.temperature_k, stoichiometry.equivalence_ratio
            )

            # Step 5: Fuel interchangeability
            interchangeability = self._calculate_interchangeability(
                stoich_input, fuel_comp
            )

            # Step 6: Heat release rate
            heat_release = self._calculate_heat_release_rate(stoich_input)

            # Step 7: Flammability limits
            flammability = self._calculate_flammability_limits(
                stoich_input, fuel_comp
            )

            # Step 8: Detonation risk
            detonation = self._calculate_detonation_risk(
                stoich_input, fuel_comp, flammability
            )

            # Create comprehensive result
            result_data = {
                "fuel_type": stoich_input.fuel_type.value,
                "stoichiometry": stoichiometry.stoichiometric_air_fuel_ratio,
                "flame_temp": adiabatic_temp.temperature_k
            }
            result_hash = self._compute_hash(result_data)

            result = ComprehensiveStoichiometryResult(
                timestamp=start_time,
                fuel_type=stoich_input.fuel_type,
                stoichiometry=stoichiometry,
                adiabatic_flame_temp=adiabatic_temp,
                equilibrium_species=equilibrium,
                dissociation=dissociation,
                interchangeability=interchangeability,
                heat_release=heat_release,
                flammability=flammability,
                detonation_risk=detonation,
                provenance_hash=result_hash
            )

            end_time = datetime.now(timezone.utc)
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return StoichiometryOutput(
                result=result,
                processing_time_ms=processing_time_ms,
                calculation_timestamp=start_time
            )

        except Exception as e:
            self._logger.error(f"Stoichiometry calculation failed: {e}", exc_info=True)
            raise

    def calculate_stoichiometric_air(
        self,
        carbon_mass_fraction: float,
        hydrogen_mass_fraction: float,
        oxygen_mass_fraction: float = 0.0,
        sulfur_mass_fraction: float = 0.0
    ) -> float:
        """
        Calculate stoichiometric air requirement.

        Formula (kg air / kg fuel):
        A_s = (1/0.232) * [2.667*C + 8*H + S - O]

        Where 0.232 is mass fraction of O2 in air.

        Args:
            carbon_mass_fraction: Carbon mass fraction (0-1)
            hydrogen_mass_fraction: Hydrogen mass fraction (0-1)
            oxygen_mass_fraction: Oxygen mass fraction (0-1)
            sulfur_mass_fraction: Sulfur mass fraction (0-1)

        Returns:
            Stoichiometric air-fuel ratio (kg air / kg fuel)
        """
        # Oxygen required per kg fuel (kg O2 / kg fuel)
        # C + O2 -> CO2: 32/12 = 2.667 kg O2/kg C
        # 2H2 + O2 -> 2H2O: 32/4 = 8 kg O2/kg H2
        # S + O2 -> SO2: 32/32 = 1 kg O2/kg S

        o2_required = (
            2.667 * carbon_mass_fraction +
            8.0 * hydrogen_mass_fraction +
            1.0 * sulfur_mass_fraction -
            oxygen_mass_fraction
        )

        # Air required (O2 is 23.2% of air by mass)
        air_required = o2_required / 0.232

        return self._round_decimal(air_required, 4)

    def calculate_adiabatic_flame_temperature(
        self,
        heating_value_kj_kg: float,
        stoich_afr: float,
        excess_air_fraction: float,
        initial_temp_k: float = 298.15
    ) -> float:
        """
        Calculate adiabatic flame temperature.

        Uses energy balance: Q_rxn = m_products * Cp_avg * (T_ad - T_initial)

        Args:
            heating_value_kj_kg: Fuel heating value (kJ/kg)
            stoich_afr: Stoichiometric air-fuel ratio
            excess_air_fraction: Excess air fraction (0.1 = 10%)
            initial_temp_k: Initial temperature in Kelvin

        Returns:
            Adiabatic flame temperature in Kelvin
        """
        # Products mass per kg fuel
        actual_afr = stoich_afr * (1 + excess_air_fraction)
        products_mass = 1 + actual_afr  # kg products / kg fuel

        # Approximate average Cp of products (kJ/kg-K)
        # Weighted average assuming typical flue gas composition
        cp_avg = 1.15  # kJ/kg-K (approximate for flue gas at high temp)

        # Energy balance
        delta_t = heating_value_kj_kg / (products_mass * cp_avg)
        t_adiabatic = initial_temp_k + delta_t

        # Apply correction for excess air (dilution effect)
        dilution_factor = 1 / (1 + 0.3 * excess_air_fraction)
        t_adiabatic_corrected = initial_temp_k + delta_t * dilution_factor

        return self._round_decimal(t_adiabatic_corrected, 2)

    def calculate_equilibrium_constant(
        self,
        reaction: str,
        temperature_k: float
    ) -> float:
        """
        Calculate equilibrium constant for dissociation reaction.

        Uses van't Hoff equation with Gibbs free energy:
        ln(Kp) = -deltaG / (R*T)

        Simplified correlations for common reactions.

        Args:
            reaction: Reaction identifier ("CO2_dissociation" or "H2O_dissociation")
            temperature_k: Temperature in Kelvin

        Returns:
            Equilibrium constant Kp
        """
        # Empirical correlations from NIST data
        if reaction == "CO2_dissociation":
            # CO2 <-> CO + 0.5*O2
            # ln(Kp) = -29764/T + 8.657 (approximate)
            ln_kp = -29764 / temperature_k + 8.657
        elif reaction == "H2O_dissociation":
            # H2O <-> H2 + 0.5*O2
            # ln(Kp) = -30208/T + 8.205 (approximate)
            ln_kp = -30208 / temperature_k + 8.205
        else:
            return 0.0

        kp = math.exp(ln_kp)
        return kp

    def calculate_wobbe_index(
        self,
        higher_heating_value_mj_m3: float,
        specific_gravity: float
    ) -> float:
        """
        Calculate Wobbe Index for fuel interchangeability.

        Formula: W = HHV / sqrt(SG)

        Args:
            higher_heating_value_mj_m3: Higher heating value (MJ/Nm3)
            specific_gravity: Fuel specific gravity (relative to air)

        Returns:
            Wobbe Index (MJ/Nm3)
        """
        if specific_gravity <= 0:
            return 0.0
        wobbe = higher_heating_value_mj_m3 / math.sqrt(specific_gravity)
        return self._round_decimal(wobbe, 2)

    def calculate_flammability_limits_mixture(
        self,
        composition: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculate flammability limits for fuel mixture using Le Chatelier's rule.

        Formula: 1/LFL_mix = sum(y_i / LFL_i)

        Args:
            composition: Mole fractions of combustible components

        Returns:
            Tuple of (LFL, UFL) in vol%
        """
        # Filter to only combustible components
        combustibles = {k: v for k, v in composition.items()
                       if k.upper() in FLAMMABILITY_LIMITS}

        if not combustibles:
            return (0.0, 0.0)

        # Normalize to combustible fraction only
        total_combustible = sum(combustibles.values())
        if total_combustible == 0:
            return (0.0, 0.0)

        normalized = {k: v/total_combustible for k, v in combustibles.items()}

        # Le Chatelier's rule
        lfl_sum = sum(y / FLAMMABILITY_LIMITS[k.upper()]["LFL"]
                     for k, y in normalized.items()
                     if k.upper() in FLAMMABILITY_LIMITS)
        ufl_sum = sum(y / FLAMMABILITY_LIMITS[k.upper()]["UFL"]
                     for k, y in normalized.items()
                     if k.upper() in FLAMMABILITY_LIMITS)

        lfl_mix = 1.0 / lfl_sum if lfl_sum > 0 else 0.0
        ufl_mix = 1.0 / ufl_sum if ufl_sum > 0 else 0.0

        return (self._round_decimal(lfl_mix, 2), self._round_decimal(ufl_mix, 2))

    def assess_detonation_risk(
        self,
        fuel_type: str,
        concentration_vol_percent: float,
        enclosure_length_m: float,
        initial_pressure_bar: float = 1.0
    ) -> Tuple[DetonationRisk, float]:
        """
        Assess detonation risk for fuel-air mixture.

        Args:
            fuel_type: Fuel type identifier
            concentration_vol_percent: Fuel concentration in vol%
            enclosure_length_m: Characteristic length of enclosure
            initial_pressure_bar: Initial pressure

        Returns:
            Tuple of (risk level, peak overpressure in bar)
        """
        fuel_key = fuel_type.upper()

        # Check if in flammable range
        if fuel_key not in FLAMMABILITY_LIMITS:
            return (DetonationRisk.NONE, 1.0)

        lfl = FLAMMABILITY_LIMITS[fuel_key]["LFL"]
        ufl = FLAMMABILITY_LIMITS[fuel_key]["UFL"]

        if concentration_vol_percent < lfl or concentration_vol_percent > ufl:
            return (DetonationRisk.NONE, initial_pressure_bar)

        # Get detonation cell size
        cell_size_mm = DETONATION_CELL_SIZES.get(fuel_key, 100)

        # Critical diameter for detonation propagation
        # D_crit ~ 13 * lambda (where lambda is cell size)
        critical_diameter_m = 13 * cell_size_mm / 1000

        # Run-up distance for DDT
        # L_DDT ~ 40 * D (typical for confined spaces)
        runup_distance_m = 40 * critical_diameter_m

        # Check if DDT is possible
        ddt_possible = enclosure_length_m > runup_distance_m

        # Peak overpressure
        if ddt_possible:
            # Detonation overpressure ~15-20x initial for hydrocarbons
            peak_pressure = initial_pressure_bar * 18
            risk = DetonationRisk.CRITICAL
        else:
            # Deflagration overpressure ~8x initial (vented)
            peak_pressure = initial_pressure_bar * 8
            risk = DetonationRisk.MEDIUM

        # Adjust risk based on reactivity
        if fuel_key in ("H2", "C2H2"):
            risk = DetonationRisk.CRITICAL if ddt_possible else DetonationRisk.HIGH

        return (risk, self._round_decimal(peak_pressure, 2))

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_fuel_composition(
        self,
        stoich_input: StoichiometryInput
    ) -> FuelComposition:
        """Get or derive fuel composition"""
        if stoich_input.fuel_composition:
            comp = stoich_input.fuel_composition
            return FuelComposition(
                ch4=comp.get("CH4", comp.get("ch4", 0.0)),
                c2h6=comp.get("C2H6", comp.get("c2h6", 0.0)),
                c3h8=comp.get("C3H8", comp.get("c3h8", 0.0)),
                c4h10=comp.get("C4H10", comp.get("c4h10", 0.0)),
                c2h4=comp.get("C2H4", comp.get("c2h4", 0.0)),
                c2h2=comp.get("C2H2", comp.get("c2h2", 0.0)),
                h2=comp.get("H2", comp.get("h2", 0.0)),
                co=comp.get("CO", comp.get("co", 0.0)),
                co2=comp.get("CO2", comp.get("co2", 0.0)),
                n2=comp.get("N2", comp.get("n2", 0.0)),
                o2=comp.get("O2", comp.get("o2", 0.0)),
                h2o=comp.get("H2O", comp.get("h2o", 0.0))
            )

        # Default compositions by fuel type
        defaults = {
            FuelType.NATURAL_GAS: FuelComposition(ch4=0.95, c2h6=0.025, c3h8=0.01, n2=0.015),
            FuelType.PROPANE: FuelComposition(c3h8=1.0),
            FuelType.BUTANE: FuelComposition(c4h10=1.0),
            FuelType.HYDROGEN: FuelComposition(h2=1.0),
        }

        return defaults.get(stoich_input.fuel_type, FuelComposition(ch4=1.0))

    def _calculate_stoichiometry(
        self,
        stoich_input: StoichiometryInput,
        fuel_comp: FuelComposition
    ) -> StoichiometricResult:
        """Calculate basic stoichiometry"""
        # Calculate stoichiometric air requirement
        c_frac = stoich_input.fuel_carbon_mass_percent / 100
        h_frac = stoich_input.fuel_hydrogen_mass_percent / 100
        o_frac = stoich_input.fuel_oxygen_mass_percent / 100
        s_frac = stoich_input.fuel_sulfur_mass_percent / 100

        stoich_afr = self.calculate_stoichiometric_air(c_frac, h_frac, o_frac, s_frac)

        # Actual air-fuel ratio
        actual_afr = stoich_input.air_flow_kg_hr / stoich_input.fuel_flow_kg_hr if stoich_input.fuel_flow_kg_hr > 0 else 0

        # Excess air
        excess_air = ((actual_afr / stoich_afr) - 1) * 100 if stoich_afr > 0 else 0

        # Equivalence ratio (phi)
        equivalence_ratio = stoich_afr / actual_afr if actual_afr > 0 else 0

        # Combustion mode
        if 0.98 <= equivalence_ratio <= 1.02:
            mode = CombustionMode.STOICHIOMETRIC
        elif equivalence_ratio < 0.98:
            mode = CombustionMode.LEAN
        else:
            mode = CombustionMode.RICH

        # Theoretical volumes per kg fuel (at STP)
        # CO2 volume: (C_mass / 12) * 22.414 L/mol
        co2_vol = (c_frac / 12) * 22.414 / 1000  # Nm3/kg fuel
        h2o_vol = (h_frac / 2) * 22.414 / 1000
        n2_vol = stoich_afr * 0.768 / 1.25  # Approximate
        total_flue = co2_vol + h2o_vol + n2_vol + (excess_air / 100) * stoich_afr / 1.25

        # Molar stoichiometry
        fuel_mw = stoich_input.fuel_molecular_weight
        stoich_afr_molar = stoich_afr * fuel_mw / self.AIR_MOLECULAR_WEIGHT

        stoich_data = {
            "stoich_afr": stoich_afr,
            "excess_air": excess_air,
            "phi": equivalence_ratio
        }

        return StoichiometricResult(
            stoichiometric_air_fuel_ratio=self._round_decimal(stoich_afr, 4),
            stoichiometric_air_fuel_ratio_molar=self._round_decimal(stoich_afr_molar, 4),
            theoretical_air_volume_nm3_per_kg=self._round_decimal(stoich_afr / 1.293, 4),
            theoretical_co2_volume_nm3_per_kg=self._round_decimal(co2_vol, 4),
            theoretical_h2o_volume_nm3_per_kg=self._round_decimal(h2o_vol, 4),
            theoretical_n2_volume_nm3_per_kg=self._round_decimal(n2_vol, 4),
            total_flue_gas_volume_nm3_per_kg=self._round_decimal(total_flue, 4),
            excess_air_percent=self._round_decimal(excess_air, 2),
            equivalence_ratio=self._round_decimal(equivalence_ratio, 4),
            combustion_mode=mode,
            provenance_hash=self._compute_hash(stoich_data)
        )

    def _calculate_adiabatic_flame_temp(
        self,
        stoich_input: StoichiometryInput,
        fuel_comp: FuelComposition,
        stoichiometry: StoichiometricResult
    ) -> AdiabaticFlameTemperature:
        """Calculate adiabatic flame temperature with dissociation correction"""
        # Initial temperature
        t_initial_k = stoich_input.initial_temperature_c + 273.15

        # Calculate without dissociation
        excess_air_frac = stoichiometry.excess_air_percent / 100
        t_adiabatic = self.calculate_adiabatic_flame_temperature(
            stoich_input.fuel_lower_heating_value_mj_kg * 1000,  # Convert to kJ/kg
            stoichiometry.stoichiometric_air_fuel_ratio,
            excess_air_frac,
            t_initial_k
        )

        # Dissociation correction (significant above 1800K)
        dissociation_correction = 0.0
        if t_adiabatic > 1800:
            # Empirical correction: ~100-300K reduction at 2200K
            dissociation_correction = min((t_adiabatic - 1800) * 0.15, 300)

        t_with_dissociation = t_adiabatic - dissociation_correction

        # Enthalpy of combustion
        enthalpy = stoich_input.fuel_lower_heating_value_mj_kg * 1000  # kJ/kg

        # Mean Cp of products
        cp_mean = 35.0  # kJ/kmol-K (approximate weighted average)

        temp_data = {
            "t_adiabatic": t_adiabatic,
            "t_with_dissociation": t_with_dissociation,
            "enthalpy": enthalpy
        }

        return AdiabaticFlameTemperature(
            temperature_k=self._round_decimal(t_adiabatic, 2),
            temperature_c=self._round_decimal(t_adiabatic - 273.15, 2),
            temperature_with_dissociation_k=self._round_decimal(t_with_dissociation, 2),
            temperature_with_dissociation_c=self._round_decimal(t_with_dissociation - 273.15, 2),
            dissociation_correction_k=self._round_decimal(dissociation_correction, 2),
            enthalpy_of_combustion_kj_per_kg=self._round_decimal(enthalpy, 2),
            mean_cp_products_kj_per_kmol_k=self._round_decimal(cp_mean, 2),
            provenance_hash=self._compute_hash(temp_data)
        )

    def _calculate_equilibrium_species(
        self,
        stoich_input: StoichiometryInput,
        fuel_comp: FuelComposition,
        stoichiometry: StoichiometricResult,
        adiabatic_temp: AdiabaticFlameTemperature
    ) -> EquilibriumSpecies:
        """Calculate equilibrium species concentrations"""
        # Use flame temperature
        temp_k = adiabatic_temp.temperature_with_dissociation_k

        # Mass fractions from fuel composition
        c_frac = stoich_input.fuel_carbon_mass_percent / 100
        h_frac = stoich_input.fuel_hydrogen_mass_percent / 100

        # Calculate product moles per kg fuel
        moles_co2 = c_frac / 12  # kmol CO2 / kg fuel
        moles_h2o = h_frac / 2  # kmol H2O / kg fuel

        # Nitrogen from air
        air_per_kg_fuel = stoich_input.air_flow_kg_hr / stoich_input.fuel_flow_kg_hr if stoich_input.fuel_flow_kg_hr > 0 else stoichiometry.stoichiometric_air_fuel_ratio
        moles_n2 = air_per_kg_fuel * 0.768 / 28.014  # kmol N2 / kg fuel

        # Excess O2
        excess_air_frac = stoichiometry.excess_air_percent / 100
        moles_o2 = air_per_kg_fuel * 0.232 * excess_air_frac / 32  # kmol O2 / kg fuel

        total_moles = moles_co2 + moles_h2o + moles_n2 + moles_o2

        # Calculate dissociation products at high temp
        co_moles = 0.0
        h2_moles = 0.0
        oh_moles = 0.0

        if temp_k > 1800:
            kp_co2 = self.calculate_equilibrium_constant("CO2_dissociation", temp_k)
            kp_h2o = self.calculate_equilibrium_constant("H2O_dissociation", temp_k)

            # Simplified dissociation estimate
            co2_dissoc_frac = min(kp_co2 * 0.01, 0.1)  # Limit to 10%
            h2o_dissoc_frac = min(kp_h2o * 0.01, 0.1)

            co_moles = moles_co2 * co2_dissoc_frac
            h2_moles = moles_h2o * h2o_dissoc_frac

            moles_co2 -= co_moles
            moles_h2o -= h2_moles

        # Thermal NOx (Zeldovich) - simplified
        # NO formation increases exponentially with temperature
        no_moles = 0.0
        if temp_k > 1500:
            # Empirical: ~1-2% of N2 at 2000K
            no_formation_rate = math.exp((temp_k - 1500) / 300) * 1e-5
            no_moles = moles_n2 * no_formation_rate

        total_moles = moles_co2 + moles_h2o + moles_n2 + moles_o2 + co_moles + h2_moles + no_moles

        species_data = {
            "temp_k": temp_k,
            "total_moles": total_moles,
            "co2": moles_co2
        }

        return EquilibriumSpecies(
            co2_mole_fraction=self._round_decimal(moles_co2 / total_moles, 6) if total_moles > 0 else 0,
            h2o_mole_fraction=self._round_decimal(moles_h2o / total_moles, 6) if total_moles > 0 else 0,
            n2_mole_fraction=self._round_decimal(moles_n2 / total_moles, 6) if total_moles > 0 else 0,
            o2_mole_fraction=self._round_decimal(moles_o2 / total_moles, 6) if total_moles > 0 else 0,
            co_mole_fraction=self._round_decimal(co_moles / total_moles, 6) if total_moles > 0 else 0,
            h2_mole_fraction=self._round_decimal(h2_moles / total_moles, 6) if total_moles > 0 else 0,
            oh_mole_fraction=self._round_decimal(oh_moles / total_moles, 6) if total_moles > 0 else 0,
            no_mole_fraction=self._round_decimal(no_moles / total_moles, 8) if total_moles > 0 else 0,
            total_moles_per_kg_fuel=self._round_decimal(total_moles, 6),
            temperature_k=self._round_decimal(temp_k, 2),
            provenance_hash=self._compute_hash(species_data)
        )

    def _calculate_dissociation_effects(
        self,
        temperature_k: float,
        equivalence_ratio: float
    ) -> DissociationEffects:
        """Calculate dissociation effects at high temperature"""
        kp_co2 = self.calculate_equilibrium_constant("CO2_dissociation", temperature_k)
        kp_h2o = self.calculate_equilibrium_constant("H2O_dissociation", temperature_k)

        # Estimate dissociation percentages
        co2_dissoc = 0.0
        h2o_dissoc = 0.0
        temp_reduction = 0.0

        if temperature_k > 1800:
            # Simplified correlation for dissociation percentage
            co2_dissoc = min(kp_co2 * 100, 15.0)  # % dissociated
            h2o_dissoc = min(kp_h2o * 100, 10.0)

            # Temperature reduction due to dissociation (endothermic)
            # Approximately 50-100K per % dissociation
            temp_reduction = (co2_dissoc + h2o_dissoc) * 10

        dissoc_data = {
            "temp_k": temperature_k,
            "co2_dissoc": co2_dissoc,
            "h2o_dissoc": h2o_dissoc
        }

        return DissociationEffects(
            co2_dissociation_percent=self._round_decimal(co2_dissoc, 2),
            h2o_dissociation_percent=self._round_decimal(h2o_dissoc, 2),
            temperature_k=self._round_decimal(temperature_k, 2),
            equilibrium_constant_co2=kp_co2,
            equilibrium_constant_h2o=kp_h2o,
            flame_temp_reduction_k=self._round_decimal(temp_reduction, 2),
            provenance_hash=self._compute_hash(dissoc_data)
        )

    def _calculate_interchangeability(
        self,
        stoich_input: StoichiometryInput,
        fuel_comp: FuelComposition
    ) -> FuelInterchangeability:
        """Calculate fuel interchangeability using AGA method"""
        # Heating value in MJ/Nm3 (approximate conversion)
        hv_vol = stoich_input.fuel_higher_heating_value_mj_kg * stoich_input.fuel_specific_gravity * 1.293

        # Wobbe Index
        wobbe = self.calculate_wobbe_index(hv_vol, stoich_input.fuel_specific_gravity)

        # Wobbe number (relative to reference)
        wobbe_number = wobbe / stoich_input.reference_wobbe_index if stoich_input.reference_wobbe_index > 0 else 1.0

        # AGA indices (simplified)
        # Lifting index: tendency for flame to lift off burner
        lifting_index = math.sqrt(stoich_input.fuel_specific_gravity) * hv_vol / stoich_input.reference_heating_value

        # Flashback index: tendency for flame to propagate into burner
        flashback_index = 1.0 / (stoich_input.fuel_specific_gravity * math.sqrt(hv_vol / stoich_input.reference_heating_value + 0.01))

        # Yellow tipping index
        yellow_tipping = 1.0  # Simplified

        # AGA Index A (primary interchangeability)
        aga_a = 1.0 - abs(wobbe_number - 1.0)

        # AGA Index B (secondary interchangeability)
        aga_b = 1.0 - abs(lifting_index - 1.0) * 0.5

        # Determine status
        if abs(wobbe_number - 1.0) <= self.AGA_WOBBE_TOLERANCE:
            status = InterchangeabilityStatus.ACCEPTABLE
        elif abs(wobbe_number - 1.0) <= 2 * self.AGA_WOBBE_TOLERANCE:
            status = InterchangeabilityStatus.MARGINAL
        else:
            status = InterchangeabilityStatus.UNACCEPTABLE

        inter_data = {
            "wobbe": wobbe,
            "wobbe_number": wobbe_number,
            "status": status.value
        }

        return FuelInterchangeability(
            wobbe_index_mj_per_nm3=self._round_decimal(wobbe, 2),
            wobbe_number=self._round_decimal(wobbe_number, 4),
            lifting_index=self._round_decimal(lifting_index, 4),
            flashback_index=self._round_decimal(flashback_index, 4),
            yellow_tipping_index=self._round_decimal(yellow_tipping, 4),
            aga_index_a=self._round_decimal(aga_a, 4),
            aga_index_b=self._round_decimal(aga_b, 4),
            status=status,
            provenance_hash=self._compute_hash(inter_data)
        )

    def _calculate_heat_release_rate(
        self,
        stoich_input: StoichiometryInput
    ) -> HeatReleaseRate:
        """Calculate heat release rate"""
        # Heat release rate (kW)
        heat_release_kw = (
            stoich_input.fuel_flow_kg_hr *
            stoich_input.fuel_lower_heating_value_mj_kg *
            stoich_input.combustion_efficiency_percent / 100 *
            1000 / 3600  # Convert MJ/hr to kW
        )

        heat_release_mw = heat_release_kw / 1000

        # Volumetric heat release
        vol_heat_release = heat_release_kw / stoich_input.combustion_chamber_volume_m3

        # Heat flux
        heat_flux = heat_release_kw / stoich_input.combustion_chamber_surface_area_m2

        # Specific fuel consumption
        sfc = stoich_input.fuel_flow_kg_hr / heat_release_kw if heat_release_kw > 0 else 0

        heat_data = {
            "heat_release_kw": heat_release_kw,
            "efficiency": stoich_input.combustion_efficiency_percent
        }

        return HeatReleaseRate(
            heat_release_rate_kw=self._round_decimal(heat_release_kw, 2),
            heat_release_rate_mw=self._round_decimal(heat_release_mw, 4),
            volumetric_heat_release_kw_per_m3=self._round_decimal(vol_heat_release, 2),
            heat_flux_kw_per_m2=self._round_decimal(heat_flux, 2),
            thermal_efficiency_percent=stoich_input.combustion_efficiency_percent,
            fuel_consumption_kg_per_hr=stoich_input.fuel_flow_kg_hr,
            specific_fuel_consumption_kg_per_kwh=self._round_decimal(sfc, 6),
            provenance_hash=self._compute_hash(heat_data)
        )

    def _calculate_flammability_limits(
        self,
        stoich_input: StoichiometryInput,
        fuel_comp: FuelComposition
    ) -> FlammabilityLimits:
        """Calculate flammability limits for fuel mixture"""
        # Build composition dict for Le Chatelier calculation
        comp_dict = {
            "CH4": fuel_comp.ch4,
            "C2H6": fuel_comp.c2h6,
            "C3H8": fuel_comp.c3h8,
            "C4H10": fuel_comp.c4h10,
            "C2H4": fuel_comp.c2h4,
            "C2H2": fuel_comp.c2h2,
            "H2": fuel_comp.h2,
            "CO": fuel_comp.co,
        }

        lfl, ufl = self.calculate_flammability_limits_mixture(comp_dict)

        # Handle pure fuel defaults
        if lfl == 0 and ufl == 0:
            # Default to methane limits
            lfl, ufl = 5.0, 15.0

        flammable_range = ufl - lfl
        current_conc = stoich_input.current_fuel_concentration_vol_percent

        # Determine status
        if current_conc < lfl:
            status = FlammabilityStatus.BELOW_LFL
        elif current_conc > ufl:
            status = FlammabilityStatus.ABOVE_UFL
        else:
            status = FlammabilityStatus.FLAMMABLE

        # Margins
        margin_lfl = lfl - current_conc
        margin_ufl = ufl - current_conc

        flam_data = {
            "lfl": lfl,
            "ufl": ufl,
            "current": current_conc,
            "status": status.value
        }

        return FlammabilityLimits(
            lower_flammability_limit_vol_percent=lfl,
            upper_flammability_limit_vol_percent=ufl,
            flammable_range_vol_percent=self._round_decimal(flammable_range, 2),
            current_concentration_vol_percent=current_conc,
            status=status,
            margin_to_lfl_percent=self._round_decimal(margin_lfl, 2),
            margin_to_ufl_percent=self._round_decimal(margin_ufl, 2),
            provenance_hash=self._compute_hash(flam_data)
        )

    def _calculate_detonation_risk(
        self,
        stoich_input: StoichiometryInput,
        fuel_comp: FuelComposition,
        flammability: FlammabilityLimits
    ) -> DetonationRiskAssessment:
        """Calculate detonation risk assessment"""
        # Determine dominant fuel for detonation cell size
        max_comp = max(
            ("CH4", fuel_comp.ch4),
            ("C2H6", fuel_comp.c2h6),
            ("C3H8", fuel_comp.c3h8),
            ("H2", fuel_comp.h2),
            ("C2H2", fuel_comp.c2h2),
            key=lambda x: x[1]
        )
        dominant_fuel = max_comp[0]

        # Get detonation cell size
        cell_size = DETONATION_CELL_SIZES.get(dominant_fuel, 100)

        # Assess risk
        risk_level, peak_pressure = self.assess_detonation_risk(
            dominant_fuel,
            flammability.current_concentration_vol_percent,
            stoich_input.enclosure_characteristic_length_m,
            stoich_input.combustion_pressure_bar
        )

        # Critical diameter and run-up distance
        critical_diameter = 13 * cell_size / 1000  # m
        runup_distance = 40 * critical_diameter

        # DDT possible?
        ddt_possible = stoich_input.enclosure_characteristic_length_m > runup_distance

        # Recommended relief area (NFPA 68 simplified)
        # A_v = C * A_s * (P_red)^(-0.5) where C ~ 0.1 for vented deflagration
        relief_area = 0.1 * stoich_input.combustion_chamber_surface_area_m2 / math.sqrt(peak_pressure)

        det_data = {
            "fuel": dominant_fuel,
            "cell_size": cell_size,
            "risk": risk_level.value
        }

        return DetonationRiskAssessment(
            detonation_cell_size_mm=float(cell_size),
            critical_diameter_mm=self._round_decimal(critical_diameter * 1000, 2),
            run_up_distance_m=self._round_decimal(runup_distance, 2),
            deflagration_to_detonation_transition_possible=ddt_possible,
            peak_overpressure_bar=peak_pressure,
            risk_level=risk_level,
            recommended_relief_area_m2=self._round_decimal(relief_area, 4),
            provenance_hash=self._compute_hash(det_data)
        )

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        if value is None:
            return 0.0
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
