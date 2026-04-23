"""
GL-009_ThermalIQ Fluids Module
==============================

Zero-hallucination thermal fluid property library.

Supports 25+ thermal fluids with validated property correlations for:
- Water/Steam (IAPWS-IF97)
- Heat transfer fluids (Therminol, Dowtherm, Syltherm)
- Glycol solutions (Ethylene Glycol, Propylene Glycol)
- Molten salts (Solar Salt, Hitec, Hitec XL)
- Thermal oils and supercritical CO2

All property calculations are:
- DETERMINISTIC: Same input produces identical output (bit-perfect)
- REPRODUCIBLE: Full provenance tracking with SHA-256 hashes
- AUDITABLE: Complete calculation trails for regulatory compliance
- STANDARDS-BASED: Correlations from published sources with citations

NO LLM in calculation path - guarantees zero hallucination risk.

References:
-----------
- IAPWS-IF97: Industrial Formulation for Water and Steam
- NIST Chemistry WebBook
- Solutia (now Eastman) Therminol Technical Data
- Dow Chemical Dowtherm Technical Data
- Coastal Chemical Hitec Technical Data
"""

from .fluid_library import (
    ThermalFluidLibrary,
    FluidProperties,
    FluidRecommendation,
    FluidCategory,
)

from .property_correlations import (
    PropertyCorrelations,
    CorrelationResult,
    get_correlation,
    validate_temperature_range,
)

__all__ = [
    # Fluid Library
    "ThermalFluidLibrary",
    "FluidProperties",
    "FluidRecommendation",
    "FluidCategory",
    # Property Correlations
    "PropertyCorrelations",
    "CorrelationResult",
    "get_correlation",
    "validate_temperature_range",
]

__version__ = "1.0.0"
__author__ = "GL-009_ThermalIQ"

# Supported fluids summary
SUPPORTED_FLUIDS = {
    "water_steam": [
        "water",
        "steam",
    ],
    "therminol": [
        "therminol_55",
        "therminol_59",
        "therminol_62",
        "therminol_66",
        "therminol_vp1",
    ],
    "dowtherm": [
        "dowtherm_a",
        "dowtherm_g",
        "dowtherm_j",
        "dowtherm_mx",
        "dowtherm_q",
        "dowtherm_rp",
    ],
    "syltherm": [
        "syltherm_800",
        "syltherm_xlt",
    ],
    "glycols": [
        "ethylene_glycol_20",
        "ethylene_glycol_30",
        "ethylene_glycol_40",
        "ethylene_glycol_50",
        "ethylene_glycol_60",
        "propylene_glycol_20",
        "propylene_glycol_30",
        "propylene_glycol_40",
        "propylene_glycol_50",
        "propylene_glycol_60",
    ],
    "molten_salts": [
        "solar_salt",  # 60% NaNO3 + 40% KNO3
        "hitec",       # 53% KNO3 + 40% NaNO2 + 7% NaNO3
        "hitec_xl",    # 48% Ca(NO3)2 + 45% KNO3 + 7% NaNO3
    ],
    "other": [
        "mineral_oil",
        "co2_supercritical",
    ],
}
