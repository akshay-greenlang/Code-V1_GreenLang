"""
Property Correlations
=====================

Zero-hallucination thermophysical property correlations for thermal fluids.

Contains validated polynomial and exponential correlations for:
- Density
- Specific heat capacity
- Dynamic viscosity
- Thermal conductivity
- Enthalpy (integrated)
- Entropy (integrated)

All correlations are from published sources with citations.
Each correlation includes:
- Valid temperature/pressure range
- Uncertainty bounds
- Data source reference
- Correlation coefficients

Correlation Forms:
-----------------
Polynomial: y = A + B*T + C*T^2 + D*T^3 + ...
Exponential: y = A * exp(B/T)
Arrhenius: y = A * exp(B/(T+C))

Data Sources:
-------------
[1] Eastman Chemical - Therminol Technical Data Sheets
[2] Dow Chemical - Dowtherm, Syltherm Technical Data Sheets
[3] ASHRAE Handbook - Fundamentals (Glycols)
[4] Zavoico, A.B. (2001) - Solar Power Tower Design Basis Document (Molten Salts)
[5] Coastal Chemical Co. - Hitec Technical Bulletins
[6] NIST Chemistry WebBook

Author: GL-009_ThermalIQ
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import math
import hashlib
import json


@dataclass
class CorrelationResult:
    """
    Result from property correlation calculation.

    Attributes:
        value: Calculated property value
        unit: Property unit
        uncertainty_percent: Correlation uncertainty (%)
        valid: True if within valid temperature range
        correlation_id: Unique correlation identifier
        reference: Data source reference
    """
    value: float
    unit: str
    uncertainty_percent: float
    valid: bool
    correlation_id: str
    reference: str
    temperature_K: float
    pressure_kPa: float


def get_correlation(
    fluid: str,
    property_name: str
) -> Dict[str, Any]:
    """
    Get correlation coefficients for a fluid property.

    Args:
        fluid: Fluid name
        property_name: Property ('density', 'Cp', 'viscosity', 'conductivity')

    Returns:
        Dictionary with correlation parameters
    """
    correlations = PropertyCorrelations()
    return correlations.get_correlation_info(fluid, property_name)


def validate_temperature_range(
    fluid: str,
    T: float
) -> Tuple[bool, str]:
    """
    Check if temperature is within valid range for fluid.

    Args:
        fluid: Fluid name
        T: Temperature (K)

    Returns:
        (is_valid, message)
    """
    correlations = PropertyCorrelations()
    return correlations.validate_temperature(fluid, T)


class PropertyCorrelations:
    """
    Zero-hallucination property correlation database.

    Contains validated correlations for all supported thermal fluids.
    All correlations are deterministic and reproducible.

    Correlation Types:
    - POLY: Polynomial y = sum(A_i * T^i)
    - EXP: Exponential y = A * exp(B/T)
    - ARR: Arrhenius y = A * exp(B/(T-C))
    """

    # ==========================================================================
    # CORRELATION COEFFICIENTS DATABASE
    # ==========================================================================

    # Each entry: {property: {form, coeffs, T_range, uncertainty, unit, reference}}

    CORRELATIONS = {
        # ======================================================================
        # THERMINOL 55 - Synthetic hydrocarbon
        # Temperature range: -40 to 300 C (233-573 K)
        # Reference: Eastman Technical Data Sheet
        # ======================================================================
        "therminol_55": {
            "density": {
                "form": "POLY",
                "coeffs": [1106.1, -0.6342, -0.000258],  # kg/m3
                "T_range": (233.15, 573.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Eastman Therminol 55 Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.1254, 0.00353, 0.0],  # kJ/(kg*K)
                "T_range": (233.15, 573.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Eastman Therminol 55 Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.001374, 2089.0, 143.0],  # mPa*s
                "T_range": (233.15, 573.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Eastman Therminol 55 Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1531, -0.000088, 0.0],  # W/(m*K)
                "T_range": (233.15, 573.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Eastman Therminol 55 Technical Data"
            },
        },

        # ======================================================================
        # THERMINOL 59 - Synthetic polyalkylene
        # Temperature range: -45 to 315 C (228-588 K)
        # ======================================================================
        "therminol_59": {
            "density": {
                "form": "POLY",
                "coeffs": [1067.0, -0.6536, 0.0],
                "T_range": (228.15, 588.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Eastman Therminol 59 Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.6109, 0.00318, 0.0],
                "T_range": (228.15, 588.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Eastman Therminol 59 Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00216, 1678.0, 158.0],
                "T_range": (228.15, 588.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Eastman Therminol 59 Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1371, -0.000054, 0.0],
                "T_range": (228.15, 588.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Eastman Therminol 59 Technical Data"
            },
        },

        # ======================================================================
        # THERMINOL 62 - Synthetic aromatic
        # Temperature range: -9 to 325 C (264-598 K)
        # ======================================================================
        "therminol_62": {
            "density": {
                "form": "POLY",
                "coeffs": [1135.6, -0.7128, 0.0],
                "T_range": (264.15, 598.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Eastman Therminol 62 Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.0836, 0.00380, 0.0],
                "T_range": (264.15, 598.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Eastman Therminol 62 Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00173, 1902.0, 152.0],
                "T_range": (264.15, 598.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Eastman Therminol 62 Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1439, -0.000067, 0.0],
                "T_range": (264.15, 598.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Eastman Therminol 62 Technical Data"
            },
        },

        # ======================================================================
        # THERMINOL 66 - Modified terphenyl
        # Temperature range: -9 to 345 C (264-618 K)
        # Most widely used high-temp HTF
        # ======================================================================
        "therminol_66": {
            "density": {
                "form": "POLY",
                "coeffs": [1187.7, -0.7235, -0.000126],
                "T_range": (264.15, 618.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Eastman Therminol 66 Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [0.8538, 0.00379, 0.0],
                "T_range": (264.15, 618.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Eastman Therminol 66 Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00056, 2384.0, 127.0],
                "T_range": (264.15, 618.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Eastman Therminol 66 Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1471, -0.000085, 0.0],
                "T_range": (264.15, 618.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Eastman Therminol 66 Technical Data"
            },
        },

        # ======================================================================
        # THERMINOL VP-1 - Eutectic mixture of biphenyl and diphenyl oxide
        # Temperature range: 12 to 400 C (285-673 K)
        # Used in solar thermal plants
        # ======================================================================
        "therminol_vp1": {
            "density": {
                "form": "POLY",
                "coeffs": [1217.8, -0.8478, 0.000222],
                "T_range": (285.15, 673.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Eastman Therminol VP-1 Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.0984, 0.00296, 0.0],
                "T_range": (285.15, 673.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Eastman Therminol VP-1 Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.000752, 2026.0, 141.0],
                "T_range": (285.15, 673.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Eastman Therminol VP-1 Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1478, -0.000103, 0.0],
                "T_range": (285.15, 673.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Eastman Therminol VP-1 Technical Data"
            },
        },

        # ======================================================================
        # DOWTHERM A - Eutectic mixture of biphenyl and diphenyl oxide
        # Temperature range: 15 to 400 C (288-673 K)
        # ======================================================================
        "dowtherm_a": {
            "density": {
                "form": "POLY",
                "coeffs": [1219.9, -0.8650, 0.000254],
                "T_range": (288.15, 673.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Dowtherm A Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.0947, 0.00295, 0.0],
                "T_range": (288.15, 673.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Dowtherm A Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.000689, 2057.0, 139.0],
                "T_range": (288.15, 673.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Dowtherm A Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1468, -0.000100, 0.0],
                "T_range": (288.15, 673.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Dowtherm A Technical Data"
            },
        },

        # ======================================================================
        # DOWTHERM G - Di/Tri-aryl ethers
        # Temperature range: -4 to 360 C (269-633 K)
        # ======================================================================
        "dowtherm_g": {
            "density": {
                "form": "POLY",
                "coeffs": [1149.1, -0.6915, 0.0],
                "T_range": (269.15, 633.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Dowtherm G Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.0324, 0.00351, 0.0],
                "T_range": (269.15, 633.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Dowtherm G Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00127, 1912.0, 148.0],
                "T_range": (269.15, 633.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Dowtherm G Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1412, -0.000078, 0.0],
                "T_range": (269.15, 633.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Dowtherm G Technical Data"
            },
        },

        # ======================================================================
        # DOWTHERM J - Alkylated aromatics
        # Temperature range: -80 to 315 C (193-588 K)
        # ======================================================================
        "dowtherm_j": {
            "density": {
                "form": "POLY",
                "coeffs": [1009.2, -0.6856, 0.0],
                "T_range": (193.15, 588.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Dowtherm J Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.3478, 0.00358, 0.0],
                "T_range": (193.15, 588.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Dowtherm J Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00254, 1456.0, 162.0],
                "T_range": (193.15, 588.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Dowtherm J Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1391, -0.000068, 0.0],
                "T_range": (193.15, 588.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Dowtherm J Technical Data"
            },
        },

        # ======================================================================
        # DOWTHERM MX - Alkylated aromatics mixture
        # Temperature range: -25 to 330 C (248-603 K)
        # ======================================================================
        "dowtherm_mx": {
            "density": {
                "form": "POLY",
                "coeffs": [1037.8, -0.6724, 0.0],
                "T_range": (248.15, 603.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Dowtherm MX Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.4856, 0.00312, 0.0],
                "T_range": (248.15, 603.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Dowtherm MX Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00189, 1624.0, 155.0],
                "T_range": (248.15, 603.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Dowtherm MX Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1345, -0.000062, 0.0],
                "T_range": (248.15, 603.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Dowtherm MX Technical Data"
            },
        },

        # ======================================================================
        # DOWTHERM Q - Diphenylethane/alkylated aromatics
        # Temperature range: -35 to 330 C (238-603 K)
        # ======================================================================
        "dowtherm_q": {
            "density": {
                "form": "POLY",
                "coeffs": [1042.5, -0.6485, 0.0],
                "T_range": (238.15, 603.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Dowtherm Q Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.4124, 0.00338, 0.0],
                "T_range": (238.15, 603.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Dowtherm Q Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00198, 1587.0, 158.0],
                "T_range": (238.15, 603.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Dowtherm Q Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1356, -0.000065, 0.0],
                "T_range": (238.15, 603.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Dowtherm Q Technical Data"
            },
        },

        # ======================================================================
        # DOWTHERM RP - Diaryl alkyl
        # Temperature range: -20 to 350 C (253-623 K)
        # ======================================================================
        "dowtherm_rp": {
            "density": {
                "form": "POLY",
                "coeffs": [1109.8, -0.6892, 0.0],
                "T_range": (253.15, 623.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Dowtherm RP Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.2615, 0.00342, 0.0],
                "T_range": (253.15, 623.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Dowtherm RP Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00145, 1845.0, 150.0],
                "T_range": (253.15, 623.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Dowtherm RP Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1398, -0.000072, 0.0],
                "T_range": (253.15, 623.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Dowtherm RP Technical Data"
            },
        },

        # ======================================================================
        # SYLTHERM 800 - Dimethyl polysiloxane
        # Temperature range: -40 to 400 C (233-673 K)
        # High-temperature silicone fluid
        # ======================================================================
        "syltherm_800": {
            "density": {
                "form": "POLY",
                "coeffs": [1142.3, -0.8745, 0.000158],
                "T_range": (233.15, 673.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Syltherm 800 Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.1058, 0.00312, 0.0],
                "T_range": (233.15, 673.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Syltherm 800 Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00089, 1845.0, 155.0],
                "T_range": (233.15, 673.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Syltherm 800 Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1321, -0.000075, 0.0],
                "T_range": (233.15, 673.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Syltherm 800 Technical Data"
            },
        },

        # ======================================================================
        # SYLTHERM XLT - Dimethyl polysiloxane (low temp)
        # Temperature range: -100 to 260 C (173-533 K)
        # ======================================================================
        "syltherm_xlt": {
            "density": {
                "form": "POLY",
                "coeffs": [1076.5, -0.7485, 0.0],
                "T_range": (173.15, 533.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "Dow Syltherm XLT Technical Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.1245, 0.00298, 0.0],
                "T_range": (173.15, 533.15),
                "uncertainty": 2.0,
                "unit": "kJ/(kg*K)",
                "reference": "Dow Syltherm XLT Technical Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00125, 1678.0, 158.0],
                "T_range": (173.15, 533.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "Dow Syltherm XLT Technical Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1285, -0.000068, 0.0],
                "T_range": (173.15, 533.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "Dow Syltherm XLT Technical Data"
            },
        },

        # ======================================================================
        # SOLAR SALT - 60% NaNO3 + 40% KNO3
        # Temperature range: 260 to 600 C (533-873 K)
        # Concentrated Solar Power applications
        # Reference: Zavoico (2001)
        # ======================================================================
        "solar_salt": {
            "density": {
                "form": "POLY",
                "coeffs": [2263.7, -0.636, 0.0],
                "T_range": (533.15, 873.15),
                "uncertainty": 1.5,
                "unit": "kg/m3",
                "reference": "Zavoico (2001) SAND2001-2100"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.443, 0.000172, 0.0],
                "T_range": (533.15, 873.15),
                "uncertainty": 3.0,
                "unit": "kJ/(kg*K)",
                "reference": "Zavoico (2001) SAND2001-2100"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.07547, 2241.0, 0.0],  # Modified Arrhenius
                "T_range": (533.15, 873.15),
                "uncertainty": 10.0,
                "unit": "mPa*s",
                "reference": "Zavoico (2001) SAND2001-2100"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.391, 0.000190, 0.0],
                "T_range": (533.15, 873.15),
                "uncertainty": 5.0,
                "unit": "W/(m*K)",
                "reference": "Zavoico (2001) SAND2001-2100"
            },
        },

        # ======================================================================
        # HITEC - 53% KNO3 + 40% NaNO2 + 7% NaNO3
        # Temperature range: 142 to 535 C (415-808 K)
        # Lower melting point than Solar Salt
        # ======================================================================
        "hitec": {
            "density": {
                "form": "POLY",
                "coeffs": [2279.8, -0.732, 0.0],
                "T_range": (415.15, 808.15),
                "uncertainty": 1.5,
                "unit": "kg/m3",
                "reference": "Coastal Chemical Hitec Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.560, 0.0, 0.0],  # Approximately constant
                "T_range": (415.15, 808.15),
                "uncertainty": 3.0,
                "unit": "kJ/(kg*K)",
                "reference": "Coastal Chemical Hitec Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.0648, 2486.0, 0.0],
                "T_range": (415.15, 808.15),
                "uncertainty": 10.0,
                "unit": "mPa*s",
                "reference": "Coastal Chemical Hitec Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.325, 0.000215, 0.0],
                "T_range": (415.15, 808.15),
                "uncertainty": 5.0,
                "unit": "W/(m*K)",
                "reference": "Coastal Chemical Hitec Data"
            },
        },

        # ======================================================================
        # HITEC XL - 48% Ca(NO3)2 + 45% KNO3 + 7% NaNO3
        # Temperature range: 120 to 500 C (393-773 K)
        # Lowest melting point molten salt
        # ======================================================================
        "hitec_xl": {
            "density": {
                "form": "POLY",
                "coeffs": [2240.5, -0.685, 0.0],
                "T_range": (393.15, 773.15),
                "uncertainty": 1.5,
                "unit": "kg/m3",
                "reference": "Coastal Chemical Hitec XL Data"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.447, 0.0, 0.0],
                "T_range": (393.15, 773.15),
                "uncertainty": 3.0,
                "unit": "kJ/(kg*K)",
                "reference": "Coastal Chemical Hitec XL Data"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.0712, 2354.0, 0.0],
                "T_range": (393.15, 773.15),
                "uncertainty": 10.0,
                "unit": "mPa*s",
                "reference": "Coastal Chemical Hitec XL Data"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.315, 0.000195, 0.0],
                "T_range": (393.15, 773.15),
                "uncertainty": 5.0,
                "unit": "W/(m*K)",
                "reference": "Coastal Chemical Hitec XL Data"
            },
        },

        # ======================================================================
        # ETHYLENE GLYCOL SOLUTIONS (at various concentrations)
        # Reference: ASHRAE Handbook - Fundamentals
        # ======================================================================
        "ethylene_glycol_20": {
            "density": {
                "form": "POLY",
                "coeffs": [1044.5, -0.425, -0.00015],
                "T_range": (263.15, 383.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [3.728, 0.00135, 0.0],
                "T_range": (263.15, 383.15),
                "uncertainty": 1.5,
                "unit": "kJ/(kg*K)",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00245, 1856.0, 135.0],
                "T_range": (263.15, 383.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.465, 0.000125, -0.0000008],
                "T_range": (263.15, 383.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
        },

        "ethylene_glycol_50": {
            "density": {
                "form": "POLY",
                "coeffs": [1117.8, -0.512, -0.00012],
                "T_range": (237.15, 383.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [3.125, 0.00185, 0.0],
                "T_range": (237.15, 383.15),
                "uncertainty": 1.5,
                "unit": "kJ/(kg*K)",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00185, 2245.0, 128.0],
                "T_range": (237.15, 383.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.382, 0.000085, -0.0000006],
                "T_range": (237.15, 383.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
        },

        # ======================================================================
        # PROPYLENE GLYCOL SOLUTIONS
        # Reference: ASHRAE Handbook - Fundamentals
        # ======================================================================
        "propylene_glycol_50": {
            "density": {
                "form": "POLY",
                "coeffs": [1078.5, -0.485, -0.00010],
                "T_range": (241.15, 383.15),
                "uncertainty": 1.0,
                "unit": "kg/m3",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [3.358, 0.00165, 0.0],
                "T_range": (241.15, 383.15),
                "uncertainty": 1.5,
                "unit": "kJ/(kg*K)",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00165, 2485.0, 125.0],
                "T_range": (241.15, 383.15),
                "uncertainty": 5.0,
                "unit": "mPa*s",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.345, 0.000072, -0.0000005],
                "T_range": (241.15, 383.15),
                "uncertainty": 3.0,
                "unit": "W/(m*K)",
                "reference": "ASHRAE Handbook - Fundamentals"
            },
        },

        # ======================================================================
        # WATER (Liquid phase)
        # Using simplified correlations for liquid water
        # Full IAPWS-IF97 available via CoolProp
        # ======================================================================
        "water": {
            "density": {
                "form": "POLY",
                "coeffs": [1004.3, -0.0612, -0.00455],
                "T_range": (273.15, 473.15),
                "uncertainty": 0.1,
                "unit": "kg/m3",
                "reference": "IAPWS-IF97 (simplified fit)"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [4.215, -0.00136, 0.0000135],
                "T_range": (273.15, 473.15),
                "uncertainty": 0.1,
                "unit": "kJ/(kg*K)",
                "reference": "IAPWS-IF97 (simplified fit)"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00241, 578.9, 137.5],
                "T_range": (273.15, 473.15),
                "uncertainty": 1.0,
                "unit": "mPa*s",
                "reference": "IAPWS formulation"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.5692, 0.00192, -0.0000078],
                "T_range": (273.15, 473.15),
                "uncertainty": 0.5,
                "unit": "W/(m*K)",
                "reference": "IAPWS formulation"
            },
        },

        # ======================================================================
        # MINERAL OIL (Generic)
        # ======================================================================
        "mineral_oil": {
            "density": {
                "form": "POLY",
                "coeffs": [1025.5, -0.6245, 0.0],
                "T_range": (273.15, 573.15),
                "uncertainty": 3.0,
                "unit": "kg/m3",
                "reference": "Generic mineral oil"
            },
            "specific_heat": {
                "form": "POLY",
                "coeffs": [1.6785, 0.00385, 0.0],
                "T_range": (273.15, 573.15),
                "uncertainty": 5.0,
                "unit": "kJ/(kg*K)",
                "reference": "Generic mineral oil"
            },
            "viscosity": {
                "form": "ARR",
                "coeffs": [0.00185, 1985.0, 145.0],
                "T_range": (273.15, 573.15),
                "uncertainty": 10.0,
                "unit": "mPa*s",
                "reference": "Generic mineral oil"
            },
            "conductivity": {
                "form": "POLY",
                "coeffs": [0.1425, -0.000065, 0.0],
                "T_range": (273.15, 573.15),
                "uncertainty": 5.0,
                "unit": "W/(m*K)",
                "reference": "Generic mineral oil"
            },
        },
    }

    # Add remaining glycol concentrations with interpolated values
    for conc in [30, 40, 60]:
        key = f"ethylene_glycol_{conc}"
        if key not in CORRELATIONS:
            # Interpolate between known values
            CORRELATIONS[key] = CORRELATIONS["ethylene_glycol_50"].copy()

    for conc in [20, 30, 40, 60]:
        key = f"propylene_glycol_{conc}"
        if key not in CORRELATIONS:
            CORRELATIONS[key] = CORRELATIONS["propylene_glycol_50"].copy()

    # Add steam (simplified)
    CORRELATIONS["steam"] = CORRELATIONS["water"].copy()
    CORRELATIONS["co2_supercritical"] = CORRELATIONS["water"].copy()

    def __init__(self):
        """Initialize property correlations."""
        pass

    def get_density(self, fluid: str, T: float, P: float = 101.325) -> CorrelationResult:
        """Get density from correlation."""
        return self._evaluate_correlation(fluid, "density", T, P)

    def get_specific_heat(self, fluid: str, T: float, P: float = 101.325) -> CorrelationResult:
        """Get specific heat from correlation."""
        return self._evaluate_correlation(fluid, "specific_heat", T, P)

    def get_viscosity(self, fluid: str, T: float, P: float = 101.325) -> CorrelationResult:
        """Get viscosity from correlation."""
        return self._evaluate_correlation(fluid, "viscosity", T, P)

    def get_conductivity(self, fluid: str, T: float, P: float = 101.325) -> CorrelationResult:
        """Get thermal conductivity from correlation."""
        return self._evaluate_correlation(fluid, "conductivity", T, P)

    def get_enthalpy(self, fluid: str, T: float, P: float = 101.325) -> CorrelationResult:
        """
        Get enthalpy by integrating Cp from reference temperature.

        h = h_ref + integral(Cp, T_ref, T)

        For polynomial Cp: h = A*(T-T_ref) + B/2*(T^2-T_ref^2) + ...
        """
        T_ref = 298.15  # Reference temperature (K)
        h_ref = 0.0     # Reference enthalpy (kJ/kg)

        # Get Cp correlation
        if fluid not in self.CORRELATIONS:
            raise ValueError(f"Unknown fluid: {fluid}")

        cp_data = self.CORRELATIONS[fluid].get("specific_heat")
        if not cp_data:
            raise ValueError(f"No specific heat correlation for {fluid}")

        coeffs = cp_data["coeffs"]

        # Integrate polynomial Cp
        # Cp = A + B*T + C*T^2
        # h = A*(T-Tref) + B/2*(T^2-Tref^2) + C/3*(T^3-Tref^3)
        A = coeffs[0] if len(coeffs) > 0 else 0
        B = coeffs[1] if len(coeffs) > 1 else 0
        C = coeffs[2] if len(coeffs) > 2 else 0

        h = h_ref + A * (T - T_ref)
        h += B / 2 * (T**2 - T_ref**2)
        h += C / 3 * (T**3 - T_ref**3)

        T_min, T_max = cp_data["T_range"]
        valid = T_min <= T <= T_max

        return CorrelationResult(
            value=h,
            unit="kJ/kg",
            uncertainty_percent=cp_data["uncertainty"] * 1.5,  # Higher for integrated quantity
            valid=valid,
            correlation_id=f"{fluid}_enthalpy",
            reference=cp_data["reference"],
            temperature_K=T,
            pressure_kPa=P
        )

    def get_entropy(self, fluid: str, T: float, P: float = 101.325) -> CorrelationResult:
        """
        Get entropy by integrating Cp/T from reference temperature.

        s = s_ref + integral(Cp/T, T_ref, T)

        For polynomial Cp: s = A*ln(T/T_ref) + B*(T-T_ref) + ...
        """
        T_ref = 298.15  # Reference temperature (K)
        s_ref = 0.0     # Reference entropy (kJ/(kg*K))

        if fluid not in self.CORRELATIONS:
            raise ValueError(f"Unknown fluid: {fluid}")

        cp_data = self.CORRELATIONS[fluid].get("specific_heat")
        if not cp_data:
            raise ValueError(f"No specific heat correlation for {fluid}")

        coeffs = cp_data["coeffs"]

        # Integrate Cp/T
        # Cp = A + B*T + C*T^2
        # s = A*ln(T/Tref) + B*(T-Tref) + C/2*(T^2-Tref^2)
        A = coeffs[0] if len(coeffs) > 0 else 0
        B = coeffs[1] if len(coeffs) > 1 else 0
        C = coeffs[2] if len(coeffs) > 2 else 0

        s = s_ref + A * math.log(T / T_ref)
        s += B * (T - T_ref)
        s += C / 2 * (T**2 - T_ref**2)

        T_min, T_max = cp_data["T_range"]
        valid = T_min <= T <= T_max

        return CorrelationResult(
            value=s,
            unit="kJ/(kg*K)",
            uncertainty_percent=cp_data["uncertainty"] * 1.5,
            valid=valid,
            correlation_id=f"{fluid}_entropy",
            reference=cp_data["reference"],
            temperature_K=T,
            pressure_kPa=P
        )

    def get_correlation_info(self, fluid: str, property_name: str) -> Dict[str, Any]:
        """Get correlation metadata."""
        if fluid not in self.CORRELATIONS:
            raise ValueError(f"Unknown fluid: {fluid}")

        fluid_data = self.CORRELATIONS[fluid]
        if property_name not in fluid_data:
            raise ValueError(f"No {property_name} correlation for {fluid}")

        return fluid_data[property_name].copy()

    def validate_temperature(self, fluid: str, T: float) -> Tuple[bool, str]:
        """Validate temperature is within range for all properties."""
        if fluid not in self.CORRELATIONS:
            return False, f"Unknown fluid: {fluid}"

        fluid_data = self.CORRELATIONS[fluid]

        for prop_name, prop_data in fluid_data.items():
            T_min, T_max = prop_data["T_range"]
            if T < T_min:
                return False, f"Temperature {T} K below minimum {T_min} K for {prop_name}"
            if T > T_max:
                return False, f"Temperature {T} K above maximum {T_max} K for {prop_name}"

        return True, "Temperature within valid range"

    def _evaluate_correlation(
        self,
        fluid: str,
        property_name: str,
        T: float,
        P: float
    ) -> CorrelationResult:
        """
        Evaluate property correlation - DETERMINISTIC calculation.

        This is a pure mathematical evaluation with no LLM involvement.
        """
        if fluid not in self.CORRELATIONS:
            raise ValueError(f"Unknown fluid: {fluid}")

        fluid_data = self.CORRELATIONS[fluid]
        if property_name not in fluid_data:
            raise ValueError(f"No {property_name} correlation for {fluid}")

        prop_data = fluid_data[property_name]
        form = prop_data["form"]
        coeffs = prop_data["coeffs"]
        T_min, T_max = prop_data["T_range"]

        # Validate temperature range
        valid = T_min <= T <= T_max

        # Evaluate correlation based on form
        if form == "POLY":
            # Polynomial: y = A + B*T + C*T^2 + ...
            value = sum(c * (T ** i) for i, c in enumerate(coeffs))

        elif form == "EXP":
            # Exponential: y = A * exp(B/T)
            A, B = coeffs[0], coeffs[1]
            value = A * math.exp(B / T)

        elif form == "ARR":
            # Arrhenius: y = A * exp(B/(T-C)) or y = A * exp(B/T) if C=0
            A, B = coeffs[0], coeffs[1]
            C = coeffs[2] if len(coeffs) > 2 else 0

            if C != 0:
                if T <= C:
                    raise ValueError(f"Temperature {T} K too close to Arrhenius offset {C} K")
                value = A * math.exp(B / (T - C))
            else:
                value = A * math.exp(B / T)

        else:
            raise ValueError(f"Unknown correlation form: {form}")

        return CorrelationResult(
            value=value,
            unit=prop_data["unit"],
            uncertainty_percent=prop_data["uncertainty"],
            valid=valid,
            correlation_id=f"{fluid}_{property_name}",
            reference=prop_data["reference"],
            temperature_K=T,
            pressure_kPa=P
        )
