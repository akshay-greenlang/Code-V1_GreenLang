"""
IAPWS-IF97 Wrapper Module for GL-012_SteamQual

This module provides a wrapper around IAPWS-IF97 steam property calculations,
implementing the Industrial Formulation 1997 for water and steam properties.
All calculations are DETERMINISTIC with complete provenance tracking.

The module can either import from GL-003_UnifiedSteam or use a standalone
minimal implementation for core functions needed by the SteamQual agent.

Regions:
- Region 1: Compressed liquid (subcooled water)
- Region 2: Superheated vapor (superheated steam)
- Region 4: Two-phase mixture (wet steam / saturation line)

Reference: IAPWS-IF97
           Wagner, W., et al. (2000). The IAPWS Industrial Formulation 1997

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import hashlib
import json
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# IAPWS-IF97 CONSTANTS
# =============================================================================

IF97_CONSTANTS: Dict[str, float] = {
    # Specific gas constant for water [kJ/(kg*K)]
    "R": 0.461526,

    # Critical point properties
    "T_CRIT": 647.096,      # Critical temperature [K]
    "P_CRIT": 22.064,       # Critical pressure [MPa]
    "RHO_CRIT": 322.0,      # Critical density [kg/m^3]

    # Triple point properties
    "T_TRIPLE": 273.16,     # Triple point temperature [K]
    "P_TRIPLE": 0.000611657,  # Triple point pressure [MPa]

    # Reference values for Region 1
    "P_STAR_1": 16.53,      # Reference pressure [MPa]
    "T_STAR_1": 1386.0,     # Reference temperature [K]

    # Reference values for Region 2
    "P_STAR_2": 1.0,        # Reference pressure [MPa]
    "T_STAR_2": 540.0,      # Reference temperature [K]

    # Conversion factors
    "CELSIUS_TO_KELVIN": 273.15,
    "KPA_TO_MPA": 0.001,
    "MPA_TO_KPA": 1000.0,
}

REGION_BOUNDARIES: Dict[str, float] = {
    # Temperature boundaries [K]
    "T_MIN": 273.15,        # 0 C - minimum temperature
    "T_MAX_1_3": 623.15,    # 350 C - boundary between regions 1/3
    "T_MAX_2": 1073.15,     # 800 C - maximum for region 2

    # Pressure boundaries [MPa]
    "P_MIN": 0.000611657,   # Triple point pressure
    "P_MAX_1_2": 100.0,     # Maximum pressure for regions 1 and 2

    # Boundary 2-3 coefficients
    "B23_N3": 348.05185628969,
    "B23_N4": -1.1671859879975,
    "B23_N5": 1.0192970039326e-3,
}


# =============================================================================
# REGION 4 COEFFICIENTS (Saturation Line)
# =============================================================================

REGION4_COEFFICIENTS: Dict[str, list] = {
    "n": [
        0.11670521452767e4,
        -0.72421316703206e6,
        -0.17073846940092e2,
        0.12020824702470e5,
        -0.32325550322333e7,
        0.14915108613530e2,
        -0.48232657361591e4,
        0.40511340542057e6,
        -0.23855557567849,
        0.65017534844798e3,
    ],
}


# =============================================================================
# REGION 1 COEFFICIENTS (Compressed Liquid)
# =============================================================================

REGION1_COEFFICIENTS: Dict[str, list] = {
    "I": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
          3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32],
    "J": [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4,
          0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41],
    "n": [
        0.14632971213167e1, -0.84548187169114, -0.37563603672040e1,
        0.33855169168385e1, -0.95791963387872, 0.15772038513228,
        -0.16616417199501e-1, 0.81214629983568e-3, 0.28319080123804e-3,
        -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1,
        -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3,
        -0.30001780793026e-3, 0.47661393906987e-4, -0.44141845330846e-5,
        -0.72694996297594e-15, -0.31679644845054e-4, -0.28270797985312e-5,
        -0.85205128120103e-9, -0.22425281908000e-5, -0.65171222895601e-6,
        -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8,
        -0.17424871230634e-9, -0.68762131295531e-18, 0.14478307828521e-19,
        0.26335781662795e-22, -0.11947622640071e-22, 0.18228094581404e-23,
        -0.93537087292458e-25,
    ],
}


# =============================================================================
# REGION 2 COEFFICIENTS (Superheated Vapor)
# =============================================================================

REGION2_IDEAL_COEFFICIENTS: Dict[str, list] = {
    "J0": [0, 1, -5, -4, -3, -2, -1, 2, 3],
    "n0": [
        -0.96927686500217e1, 0.10086655968018e2, -0.56087911283020e-2,
        0.71452738081455e-1, -0.40710498223928, 0.14240819171444e1,
        -0.43839511319450e1, -0.28408632460772, 0.21268463753307e-1,
    ],
}

REGION2_RESIDUAL_COEFFICIENTS: Dict[str, list] = {
    "I": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6,
          6, 6, 7, 7, 7, 8, 8, 9, 10, 10, 10, 16, 16, 18, 20, 20, 20, 21, 22, 23, 24, 24, 24],
    "J": [0, 1, 2, 3, 6, 1, 2, 4, 7, 36, 0, 1, 3, 6, 35, 1, 2, 3, 7, 3,
          16, 35, 0, 11, 25, 8, 36, 13, 4, 10, 14, 29, 50, 57, 20, 35, 48, 21, 53, 39, 26, 40, 58],
    "n": [
        -0.17731742473213e-2, -0.17834862292358e-1, -0.45996013696365e-1,
        -0.57581259083432e-1, -0.50325278727930e-1, -0.33032641670203e-4,
        -0.18948987516315e-3, -0.39392777243355e-2, -0.43797295650573e-1,
        -0.26674547914087e-4, 0.20481737692309e-7, 0.43870667284435e-6,
        -0.32277677238570e-4, -0.15033924542148e-2, -0.40668253562649e-1,
        -0.78847309559367e-9, 0.12790717852285e-7, 0.48225372718507e-6,
        0.22922076337661e-5, -0.16714766451061e-10, -0.21171472321355e-2,
        -0.23895741934104e2, -0.59059564324270e-17, -0.12621808899101e-5,
        -0.38946842435739e-1, 0.11256211360459e-10, -0.82311340897998e1,
        0.19809712802088e-7, 0.10406965210174e-18, -0.10234747095929e-12,
        -0.10018179379511e-8, -0.80882908646985e-10, 0.10693031879409,
        -0.33662250574171, 0.89185845355421e-24, 0.30629316876232e-12,
        -0.42002467698208e-5, -0.59056029685639e-25, 0.37826947613457e-5,
        -0.12768608934681e-14, 0.73087610595061e-28, 0.55414715350778e-16,
        -0.94369707241210e-6,
    ],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SaturationData:
    """Saturation properties at given pressure."""
    pressure_mpa: float
    temperature_k: float
    hf: float    # Saturated liquid enthalpy [kJ/kg]
    hg: float    # Saturated vapor enthalpy [kJ/kg]
    hfg: float   # Latent heat of vaporization [kJ/kg]
    sf: float    # Saturated liquid entropy [kJ/(kg*K)]
    sg: float    # Saturated vapor entropy [kJ/(kg*K)]
    sfg: float   # Entropy of vaporization [kJ/(kg*K)]
    vf: float    # Saturated liquid specific volume [m^3/kg]
    vg: float    # Saturated vapor specific volume [m^3/kg]


# =============================================================================
# UNIT CONVERSION FUNCTIONS
# =============================================================================

def celsius_to_kelvin(temperature_c: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.

    DETERMINISTIC: Same input always produces same output.

    Args:
        temperature_c: Temperature in Celsius

    Returns:
        Temperature in Kelvin
    """
    return temperature_c + IF97_CONSTANTS["CELSIUS_TO_KELVIN"]


def kelvin_to_celsius(temperature_k: float) -> float:
    """
    Convert temperature from Kelvin to Celsius.

    DETERMINISTIC: Same input always produces same output.

    Args:
        temperature_k: Temperature in Kelvin

    Returns:
        Temperature in Celsius
    """
    return temperature_k - IF97_CONSTANTS["CELSIUS_TO_KELVIN"]


def kpa_to_mpa(pressure_kpa: float) -> float:
    """
    Convert pressure from kPa to MPa.

    DETERMINISTIC: Same input always produces same output.

    Args:
        pressure_kpa: Pressure in kPa

    Returns:
        Pressure in MPa
    """
    return pressure_kpa * IF97_CONSTANTS["KPA_TO_MPA"]


def mpa_to_kpa(pressure_mpa: float) -> float:
    """
    Convert pressure from MPa to kPa.

    DETERMINISTIC: Same input always produces same output.

    Args:
        pressure_mpa: Pressure in MPa

    Returns:
        Pressure in kPa
    """
    return pressure_mpa * IF97_CONSTANTS["MPA_TO_KPA"]


def compute_density(specific_volume_m3_kg: float) -> float:
    """
    Calculate density from specific volume.

    DETERMINISTIC: Same input always produces same output.

    Args:
        specific_volume_m3_kg: Specific volume in m^3/kg

    Returns:
        Density in kg/m^3

    Raises:
        ValueError: If specific volume is not positive
    """
    if specific_volume_m3_kg <= 0:
        raise ValueError("Specific volume must be positive")
    return 1.0 / specific_volume_m3_kg


# =============================================================================
# SATURATION FUNCTIONS (REGION 4)
# =============================================================================

def get_saturation_pressure(temperature_k: float) -> float:
    """
    Calculate saturation pressure at given temperature using IAPWS-IF97.

    DETERMINISTIC: Same input always produces same output.

    Args:
        temperature_k: Temperature in Kelvin (273.15 to 647.096 K)

    Returns:
        Saturation pressure in MPa

    Raises:
        ValueError: If temperature is outside valid range
    """
    T = temperature_k
    T_MIN = REGION_BOUNDARIES["T_MIN"]
    T_CRIT = IF97_CONSTANTS["T_CRIT"]

    if T < T_MIN or T > T_CRIT:
        raise ValueError(
            f"Temperature {T} K is outside saturation range [{T_MIN}, {T_CRIT}] K"
        )

    n = REGION4_COEFFICIENTS["n"]

    theta = T + n[8] / (T - n[9])

    A = theta**2 + n[0] * theta + n[1]
    B = n[2] * theta**2 + n[3] * theta + n[4]
    C = n[5] * theta**2 + n[6] * theta + n[7]

    p_sat = (2 * C / (-B + math.sqrt(B**2 - 4 * A * C)))**4

    return p_sat


def get_saturation_temperature(pressure_mpa: float) -> float:
    """
    Calculate saturation temperature at given pressure using IAPWS-IF97.

    DETERMINISTIC: Same input always produces same output.

    Args:
        pressure_mpa: Pressure in MPa (0.000611657 to 22.064 MPa)

    Returns:
        Saturation temperature in Kelvin

    Raises:
        ValueError: If pressure is outside valid range
    """
    P = pressure_mpa
    P_MIN = REGION_BOUNDARIES["P_MIN"]
    P_CRIT = IF97_CONSTANTS["P_CRIT"]

    if P < P_MIN or P > P_CRIT:
        raise ValueError(
            f"Pressure {P} MPa is outside saturation range [{P_MIN}, {P_CRIT}] MPa"
        )

    n = REGION4_COEFFICIENTS["n"]

    beta = P**0.25

    E = beta**2 + n[2] * beta + n[5]
    F = n[0] * beta**2 + n[3] * beta + n[6]
    G = n[1] * beta**2 + n[4] * beta + n[7]

    D = 2 * G / (-F - math.sqrt(F**2 - 4 * E * G))

    T_sat = (n[9] + D - math.sqrt((n[9] + D)**2 - 4 * (n[8] + n[9] * D))) / 2

    return T_sat


# =============================================================================
# REGION 1 PROPERTY FUNCTIONS (Compressed Liquid)
# =============================================================================

def _region1_gamma(pressure_mpa: float, temperature_k: float) -> Dict[str, float]:
    """
    Calculate dimensionless Gibbs free energy and derivatives for Region 1.

    DETERMINISTIC: Same inputs always produce same outputs.
    """
    P_star = IF97_CONSTANTS["P_STAR_1"]
    T_star = IF97_CONSTANTS["T_STAR_1"]

    pi = pressure_mpa / P_star
    tau = T_star / temperature_k

    I = REGION1_COEFFICIENTS["I"]
    J = REGION1_COEFFICIENTS["J"]
    n = REGION1_COEFFICIENTS["n"]

    gamma = 0.0
    gamma_pi = 0.0
    gamma_tau = 0.0
    gamma_tautau = 0.0

    for i in range(len(n)):
        term = n[i] * (7.1 - pi)**I[i] * (tau - 1.222)**J[i]
        gamma += term

        if I[i] != 0:
            gamma_pi -= n[i] * I[i] * (7.1 - pi)**(I[i] - 1) * (tau - 1.222)**J[i]

        if J[i] != 0:
            gamma_tau += n[i] * (7.1 - pi)**I[i] * J[i] * (tau - 1.222)**(J[i] - 1)
            gamma_tautau += n[i] * (7.1 - pi)**I[i] * J[i] * (J[i] - 1) * (tau - 1.222)**(J[i] - 2)

    return {
        "gamma": gamma,
        "gamma_pi": gamma_pi,
        "gamma_tau": gamma_tau,
        "gamma_tautau": gamma_tautau,
        "pi": pi,
        "tau": tau,
    }


def region1_specific_volume(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific volume in Region 1 (compressed liquid).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific volume in m^3/kg
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region1_gamma(pressure_mpa, temperature_k)
    v = R * temperature_k / (pressure_mpa * 1000) * gamma_data["pi"] * gamma_data["gamma_pi"]
    return v


def region1_specific_enthalpy(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific enthalpy in Region 1 (compressed liquid).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific enthalpy in kJ/kg
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region1_gamma(pressure_mpa, temperature_k)
    h = R * temperature_k * gamma_data["tau"] * gamma_data["gamma_tau"]
    return h


def region1_specific_entropy(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific entropy in Region 1 (compressed liquid).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific entropy in kJ/(kg*K)
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region1_gamma(pressure_mpa, temperature_k)
    s = R * (gamma_data["tau"] * gamma_data["gamma_tau"] - gamma_data["gamma"])
    return s


def region1_specific_isobaric_heat_capacity(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific isobaric heat capacity in Region 1.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific heat capacity Cp in kJ/(kg*K)
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region1_gamma(pressure_mpa, temperature_k)
    cp = -R * gamma_data["tau"]**2 * gamma_data["gamma_tautau"]
    return cp


# =============================================================================
# REGION 2 PROPERTY FUNCTIONS (Superheated Vapor)
# =============================================================================

def _region2_gamma(pressure_mpa: float, temperature_k: float) -> Dict[str, float]:
    """
    Calculate dimensionless Gibbs free energy and derivatives for Region 2.

    DETERMINISTIC: Same inputs always produce same outputs.
    """
    P_star = IF97_CONSTANTS["P_STAR_2"]
    T_star = IF97_CONSTANTS["T_STAR_2"]

    pi = pressure_mpa / P_star
    tau = T_star / temperature_k

    # Ideal gas part
    J0 = REGION2_IDEAL_COEFFICIENTS["J0"]
    n0 = REGION2_IDEAL_COEFFICIENTS["n0"]

    gamma0 = math.log(pi)
    gamma0_pi = 1.0 / pi
    gamma0_tau = 0.0
    gamma0_tautau = 0.0

    for i in range(len(n0)):
        gamma0 += n0[i] * tau**J0[i]
        if J0[i] != 0:
            gamma0_tau += n0[i] * J0[i] * tau**(J0[i] - 1)
            gamma0_tautau += n0[i] * J0[i] * (J0[i] - 1) * tau**(J0[i] - 2)

    # Residual part
    I = REGION2_RESIDUAL_COEFFICIENTS["I"]
    J = REGION2_RESIDUAL_COEFFICIENTS["J"]
    n = REGION2_RESIDUAL_COEFFICIENTS["n"]

    gammar = 0.0
    gammar_pi = 0.0
    gammar_tau = 0.0
    gammar_tautau = 0.0

    for i in range(len(n)):
        term = n[i] * pi**I[i] * (tau - 0.5)**J[i]
        gammar += term
        gammar_pi += n[i] * I[i] * pi**(I[i] - 1) * (tau - 0.5)**J[i]

        if J[i] != 0:
            gammar_tau += n[i] * pi**I[i] * J[i] * (tau - 0.5)**(J[i] - 1)
            gammar_tautau += n[i] * pi**I[i] * J[i] * (J[i] - 1) * (tau - 0.5)**(J[i] - 2)

    return {
        "gamma0": gamma0,
        "gamma0_pi": gamma0_pi,
        "gamma0_tau": gamma0_tau,
        "gamma0_tautau": gamma0_tautau,
        "gammar": gammar,
        "gammar_pi": gammar_pi,
        "gammar_tau": gammar_tau,
        "gammar_tautau": gammar_tautau,
        "pi": pi,
        "tau": tau,
    }


def region2_specific_volume(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific volume in Region 2 (superheated vapor).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific volume in m^3/kg
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region2_gamma(pressure_mpa, temperature_k)
    gamma_pi = gamma_data["gamma0_pi"] + gamma_data["gammar_pi"]
    v = R * temperature_k / (pressure_mpa * 1000) * gamma_data["pi"] * gamma_pi
    return v


def region2_specific_enthalpy(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific enthalpy in Region 2 (superheated vapor).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific enthalpy in kJ/kg
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region2_gamma(pressure_mpa, temperature_k)
    gamma_tau = gamma_data["gamma0_tau"] + gamma_data["gammar_tau"]
    h = R * temperature_k * gamma_data["tau"] * gamma_tau
    return h


def region2_specific_entropy(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific entropy in Region 2 (superheated vapor).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific entropy in kJ/(kg*K)
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region2_gamma(pressure_mpa, temperature_k)
    gamma = gamma_data["gamma0"] + gamma_data["gammar"]
    gamma_tau = gamma_data["gamma0_tau"] + gamma_data["gammar_tau"]
    s = R * (gamma_data["tau"] * gamma_tau - gamma)
    return s


def region2_specific_isobaric_heat_capacity(pressure_mpa: float, temperature_k: float) -> float:
    """
    Calculate specific isobaric heat capacity in Region 2.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Specific heat capacity Cp in kJ/(kg*K)
    """
    R = IF97_CONSTANTS["R"]
    gamma_data = _region2_gamma(pressure_mpa, temperature_k)
    gamma_tautau = gamma_data["gamma0_tautau"] + gamma_data["gammar_tautau"]
    cp = -R * gamma_data["tau"]**2 * gamma_tautau
    return cp


# =============================================================================
# REGION 4 SATURATION PROPERTIES
# =============================================================================

def get_saturation_properties(pressure_mpa: float) -> SaturationData:
    """
    Calculate saturation properties at given pressure.

    DETERMINISTIC: Same input always produces same output.

    Args:
        pressure_mpa: Pressure in MPa

    Returns:
        SaturationData with all saturation properties
    """
    P = pressure_mpa
    T_sat = get_saturation_temperature(P)

    # Saturated liquid properties (Region 1 at saturation)
    hf = region1_specific_enthalpy(P, T_sat)
    sf = region1_specific_entropy(P, T_sat)
    vf = region1_specific_volume(P, T_sat)

    # Saturated vapor properties (Region 2 at saturation)
    hg = region2_specific_enthalpy(P, T_sat)
    sg = region2_specific_entropy(P, T_sat)
    vg = region2_specific_volume(P, T_sat)

    # Derived properties
    hfg = hg - hf
    sfg = sg - sf

    return SaturationData(
        pressure_mpa=P,
        temperature_k=T_sat,
        hf=hf,
        hg=hg,
        hfg=hfg,
        sf=sf,
        sg=sg,
        sfg=sfg,
        vf=vf,
        vg=vg,
    )


def region4_mixture_enthalpy(pressure_mpa: float, quality_x: float) -> float:
    """
    Calculate mixture enthalpy for wet steam.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        quality_x: Steam quality (dryness fraction), 0 <= x <= 1

    Returns:
        Specific enthalpy in kJ/kg
    """
    if quality_x < 0 or quality_x > 1:
        raise ValueError(f"Quality must be between 0 and 1, got {quality_x}")

    sat = get_saturation_properties(pressure_mpa)
    h = sat.hf + quality_x * sat.hfg
    return h


def region4_mixture_entropy(pressure_mpa: float, quality_x: float) -> float:
    """
    Calculate mixture entropy for wet steam.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        quality_x: Steam quality (dryness fraction), 0 <= x <= 1

    Returns:
        Specific entropy in kJ/(kg*K)
    """
    if quality_x < 0 or quality_x > 1:
        raise ValueError(f"Quality must be between 0 and 1, got {quality_x}")

    sat = get_saturation_properties(pressure_mpa)
    s = sat.sf + quality_x * sat.sfg
    return s


def region4_mixture_specific_volume(pressure_mpa: float, quality_x: float) -> float:
    """
    Calculate mixture specific volume for wet steam.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        quality_x: Steam quality (dryness fraction), 0 <= x <= 1

    Returns:
        Specific volume in m^3/kg
    """
    if quality_x < 0 or quality_x > 1:
        raise ValueError(f"Quality must be between 0 and 1, got {quality_x}")

    sat = get_saturation_properties(pressure_mpa)
    v = sat.vf + quality_x * (sat.vg - sat.vf)
    return v


# =============================================================================
# REGION DETECTION
# =============================================================================

def detect_region(pressure_mpa: float, temperature_k: float) -> int:
    """
    Detect the IAPWS-IF97 region for given pressure and temperature.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in Kelvin

    Returns:
        Region number (1, 2, or 4)

    Raises:
        ValueError: If inputs are outside valid range
    """
    T = temperature_k
    P = pressure_mpa

    T_MIN = REGION_BOUNDARIES["T_MIN"]
    T_MAX_1_3 = REGION_BOUNDARIES["T_MAX_1_3"]
    T_MAX_2 = REGION_BOUNDARIES["T_MAX_2"]
    P_MIN = REGION_BOUNDARIES["P_MIN"]
    P_MAX = REGION_BOUNDARIES["P_MAX_1_2"]

    if T < T_MIN:
        raise ValueError(f"Temperature {T} K is below minimum {T_MIN} K")
    if T > T_MAX_2:
        raise ValueError(f"Temperature {T} K is above maximum {T_MAX_2} K")
    if P < P_MIN:
        raise ValueError(f"Pressure {P} MPa is below minimum {P_MIN} MPa")
    if P > P_MAX:
        raise ValueError(f"Pressure {P} MPa is above maximum {P_MAX} MPa")

    try:
        T_sat = get_saturation_temperature(P)
    except ValueError:
        T_sat = None

    if T_sat is not None:
        if abs(T - T_sat) < 0.001:
            return 4
        if T < T_sat and T <= T_MAX_1_3:
            return 1
        if T > T_sat and T <= T_MAX_2:
            return 2

    if T <= T_MAX_1_3:
        return 1

    return 2


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

def compute_provenance_hash(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    method: str = "IAPWS-IF97",
) -> str:
    """
    Compute SHA-256 hash for calculation provenance and audit trail.

    DETERMINISTIC: Same inputs always produce same hash.

    Args:
        inputs: Input parameters
        outputs: Calculated outputs
        method: Calculation method identifier

    Returns:
        SHA-256 hex digest string
    """
    provenance_data = {
        "inputs": inputs,
        "outputs": outputs,
        "method": method,
        "version": "GL-012-SteamQual-v1.0.0",
    }

    provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
    return hashlib.sha256(provenance_str.encode()).hexdigest()
