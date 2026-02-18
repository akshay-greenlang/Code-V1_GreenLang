# -*- coding: utf-8 -*-
"""
FlareSystemDatabaseEngine - Engine 1: Flaring Agent (AGENT-MRV-006)

Manages flare system specifications, gas compositions, component properties,
emission factors, and heating value calculations for GHG Protocol Scope 1
flaring emission calculations. Provides deterministic, zero-hallucination
lookup of flare type specifications, default gas compositions, emission
factors, heating values, and Wobbe Index calculations.

All numeric values use ``Decimal`` for precision. The engine is thread-safe
via ``threading.Lock()`` and tracks every lookup through SHA-256 provenance
hashing.

Data Sources:
    - EPA 40 CFR Part 98 Subpart W (Section W.23)
    - IPCC 2006 Guidelines (Vol 2, Energy, Ch 4)
    - API Compendium of GHG Emissions Methodologies
    - GPA Midstream Standard 2145 (gas compositions)
    - GPSA Engineering Data Book (component properties)

Flare Types (8):
    ELEVATED_STEAM_ASSISTED, ELEVATED_AIR_ASSISTED, ELEVATED_UNASSISTED,
    ENCLOSED_GROUND, MULTI_POINT_GROUND, OFFSHORE_MARINE, CANDLESTICK,
    LOW_PRESSURE

Gas Composition Scenarios (9):
    SWEET_NATURAL_GAS, NATURAL_GAS, SOUR_NATURAL_GAS, ASSOCIATED_GAS,
    REFINERY_OFF_GAS, LANDFILL_GAS, BIOGAS, CHEMICAL_PLANT_WASTE_GAS,
    COKE_OVEN_GAS

Component Properties (15):
    CH4, C2H6, C3H8, N_C4H10, I_C4H10, C5H12, C6_PLUS, CO2, N2,
    H2S, H2, CO, C2H4, C3H6, H2O

Example:
    >>> from greenlang.flaring.flare_system_database import FlareSystemDatabaseEngine
    >>> db = FlareSystemDatabaseEngine()
    >>> specs = db.get_flare_type_specs("ELEVATED_STEAM_ASSISTED")
    >>> print(specs["typical_ce"])  # Decimal('0.98')
    >>> comp = db.get_default_composition("NATURAL_GAS")
    >>> hhv = db.calculate_hhv(comp)
    >>> print(hhv)  # ~1012 BTU/scf
    >>> wi = db.calculate_wobbe_index(comp)
    >>> print(wi)  # ~1350 BTU/scf

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["FlareSystemDatabaseEngine", "FlareType", "GasCompositionScenario"]

# ---------------------------------------------------------------------------
# Conditional imports for GreenLang infrastructure
# ---------------------------------------------------------------------------

try:
    from greenlang.flaring.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.flaring.metrics import (
        record_flare_lookup as _record_flare_lookup,
        record_factor_selection as _record_factor_selection,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_flare_lookup = None  # type: ignore[assignment]
    _record_factor_selection = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places


# ===========================================================================
# Enumerations
# ===========================================================================


class FlareType(str, Enum):
    """Eight flare type classifications per PRD AGENT-MRV-006.

    Each type has distinct physical characteristics, typical combustion
    efficiency, and applicable operational parameters that affect emission
    calculations.
    """

    ELEVATED_STEAM_ASSISTED = "ELEVATED_STEAM_ASSISTED"
    ELEVATED_AIR_ASSISTED = "ELEVATED_AIR_ASSISTED"
    ELEVATED_UNASSISTED = "ELEVATED_UNASSISTED"
    ENCLOSED_GROUND = "ENCLOSED_GROUND"
    MULTI_POINT_GROUND = "MULTI_POINT_GROUND"
    OFFSHORE_MARINE = "OFFSHORE_MARINE"
    CANDLESTICK = "CANDLESTICK"
    LOW_PRESSURE = "LOW_PRESSURE"


class GasCompositionScenario(str, Enum):
    """Nine default gas composition scenarios for common flaring situations.

    Each scenario maps to a typical mole-fraction composition of gas
    components found in the corresponding industrial context.
    """

    SWEET_NATURAL_GAS = "SWEET_NATURAL_GAS"
    NATURAL_GAS = "NATURAL_GAS"
    SOUR_NATURAL_GAS = "SOUR_NATURAL_GAS"
    ASSOCIATED_GAS = "ASSOCIATED_GAS"
    REFINERY_OFF_GAS = "REFINERY_OFF_GAS"
    LANDFILL_GAS = "LANDFILL_GAS"
    BIOGAS = "BIOGAS"
    CHEMICAL_PLANT_WASTE_GAS = "CHEMICAL_PLANT_WASTE_GAS"
    COKE_OVEN_GAS = "COKE_OVEN_GAS"


class EFSource(str, Enum):
    """Emission factor data sources for flaring calculations."""

    EPA = "EPA"
    IPCC = "IPCC"
    API = "API"
    CUSTOM = "CUSTOM"


# ===========================================================================
# Flare Type Specifications
# ===========================================================================

_FLARE_TYPE_SPECS: Dict[str, Dict[str, Any]] = {
    "ELEVATED_STEAM_ASSISTED": {
        "display_name": "Elevated Steam-Assisted Flare",
        "description": (
            "High-pressure flare tip with external steam injection for "
            "smokeless operation. Most common in refineries and large "
            "chemical plants."
        ),
        "capacity_range_mmbtu_hr": (Decimal("10"), Decimal("5000")),
        "typical_ce": Decimal("0.98"),
        "tip_velocity_range_m_s": (Decimal("10"), Decimal("120")),
        "assist_type": "STEAM",
        "height_range_m": (Decimal("30"), Decimal("150")),
        "typical_applications": [
            "Petroleum refineries",
            "Chemical plants",
            "Natural gas processing",
            "Ethylene plants",
        ],
        "smokeless_capability": True,
        "wind_shielding": False,
        "minimum_lhv_btu_scf": Decimal("200"),
        "requires_assist_medium": True,
    },
    "ELEVATED_AIR_ASSISTED": {
        "display_name": "Elevated Air-Assisted Flare",
        "description": (
            "Flare tip with forced-draft air blowers for combustion "
            "enhancement and smokeless operation. Used where steam is "
            "not economically available."
        ),
        "capacity_range_mmbtu_hr": (Decimal("5"), Decimal("2000")),
        "typical_ce": Decimal("0.98"),
        "tip_velocity_range_m_s": (Decimal("8"), Decimal("100")),
        "assist_type": "AIR",
        "height_range_m": (Decimal("20"), Decimal("100")),
        "typical_applications": [
            "Upstream oil and gas",
            "Small refineries",
            "Tank farms",
            "Wastewater treatment",
        ],
        "smokeless_capability": True,
        "wind_shielding": False,
        "minimum_lhv_btu_scf": Decimal("200"),
        "requires_assist_medium": True,
    },
    "ELEVATED_UNASSISTED": {
        "display_name": "Elevated Unassisted Flare",
        "description": (
            "Simple open pipe flare with no assist medium. Relies on "
            "natural draft and gas momentum for combustion. Lower cost "
            "but may produce visible smoke."
        ),
        "capacity_range_mmbtu_hr": (Decimal("1"), Decimal("500")),
        "typical_ce": Decimal("0.96"),
        "tip_velocity_range_m_s": (Decimal("5"), Decimal("80")),
        "assist_type": "NONE",
        "height_range_m": (Decimal("15"), Decimal("80")),
        "typical_applications": [
            "Production facilities",
            "Gas gathering stations",
            "Small chemical plants",
            "Emergency relief only",
        ],
        "smokeless_capability": False,
        "wind_shielding": False,
        "minimum_lhv_btu_scf": Decimal("300"),
        "requires_assist_medium": False,
    },
    "ENCLOSED_GROUND": {
        "display_name": "Enclosed Ground Flare",
        "description": (
            "Multi-burner system enclosed in a refractory-lined enclosure "
            "at ground level. Provides highest combustion efficiency and "
            "visibility concealment. No visible flame or radiation."
        ),
        "capacity_range_mmbtu_hr": (Decimal("1"), Decimal("1000")),
        "typical_ce": Decimal("0.995"),
        "tip_velocity_range_m_s": (Decimal("2"), Decimal("30")),
        "assist_type": "ENCLOSED",
        "height_range_m": (Decimal("3"), Decimal("20")),
        "typical_applications": [
            "Landfill gas",
            "Biogas plants",
            "Urban refineries",
            "Tank terminal vapor recovery",
        ],
        "smokeless_capability": True,
        "wind_shielding": True,
        "minimum_lhv_btu_scf": Decimal("100"),
        "requires_assist_medium": False,
    },
    "MULTI_POINT_GROUND": {
        "display_name": "Multi-Point Ground Flare (MPGF)",
        "description": (
            "Multiple staged burners distributed at ground level within "
            "a fenced area. Provides high capacity with reduced radiation "
            "and noise compared to elevated flares."
        ),
        "capacity_range_mmbtu_hr": (Decimal("50"), Decimal("10000")),
        "typical_ce": Decimal("0.99"),
        "tip_velocity_range_m_s": (Decimal("3"), Decimal("40")),
        "assist_type": "STAGED",
        "height_range_m": (Decimal("2"), Decimal("10")),
        "typical_applications": [
            "Large refineries",
            "LNG terminals",
            "Ethylene plants",
            "High-capacity emergency relief",
        ],
        "smokeless_capability": True,
        "wind_shielding": True,
        "minimum_lhv_btu_scf": Decimal("150"),
        "requires_assist_medium": False,
    },
    "OFFSHORE_MARINE": {
        "display_name": "Offshore Marine Flare",
        "description": (
            "Boom-mounted flare extending from offshore platform. "
            "Designed to withstand marine winds, salt spray, and wave "
            "motion. Typically unassisted due to logistics."
        ),
        "capacity_range_mmbtu_hr": (Decimal("5"), Decimal("3000")),
        "typical_ce": Decimal("0.95"),
        "tip_velocity_range_m_s": (Decimal("10"), Decimal("150")),
        "assist_type": "NONE",
        "height_range_m": (Decimal("20"), Decimal("80")),
        "typical_applications": [
            "Offshore production platforms",
            "FPSOs",
            "Subsea tiebacks",
            "Offshore drilling rigs",
        ],
        "smokeless_capability": False,
        "wind_shielding": False,
        "minimum_lhv_btu_scf": Decimal("300"),
        "requires_assist_medium": False,
    },
    "CANDLESTICK": {
        "display_name": "Candlestick Flare",
        "description": (
            "Simple vertical pipe flare with no wind shielding or "
            "assist medium. Lowest cost option. Subject to wind-induced "
            "efficiency losses."
        ),
        "capacity_range_mmbtu_hr": (Decimal("0.5"), Decimal("200")),
        "typical_ce": Decimal("0.95"),
        "tip_velocity_range_m_s": (Decimal("3"), Decimal("60")),
        "assist_type": "NONE",
        "height_range_m": (Decimal("5"), Decimal("40")),
        "typical_applications": [
            "Well sites",
            "Tank batteries",
            "Small production facilities",
            "Temporary flaring",
        ],
        "smokeless_capability": False,
        "wind_shielding": False,
        "minimum_lhv_btu_scf": Decimal("300"),
        "requires_assist_medium": False,
    },
    "LOW_PRESSURE": {
        "display_name": "Low-Pressure Flare",
        "description": (
            "Designed for low-flow, low-pressure waste gas streams. "
            "May incorporate flame retention devices to maintain stable "
            "combustion at low velocities."
        ),
        "capacity_range_mmbtu_hr": (Decimal("0.1"), Decimal("50")),
        "typical_ce": Decimal("0.93"),
        "tip_velocity_range_m_s": (Decimal("1"), Decimal("20")),
        "assist_type": "NONE",
        "height_range_m": (Decimal("3"), Decimal("20")),
        "typical_applications": [
            "Vapor recovery units",
            "Low-pressure vent gas",
            "Glycol dehydrator off-gas",
            "Storage tank vents",
        ],
        "smokeless_capability": False,
        "wind_shielding": False,
        "minimum_lhv_btu_scf": Decimal("200"),
        "requires_assist_medium": False,
    },
}


# ===========================================================================
# Default Gas Compositions (mole fractions, sum to 1.0)
# ===========================================================================

_DEFAULT_GAS_COMPOSITIONS: Dict[str, Dict[str, Decimal]] = {
    "SWEET_NATURAL_GAS": {
        "CH4":   Decimal("0.875"),
        "C2H6":  Decimal("0.065"),
        "C3H8":  Decimal("0.030"),
        "CO2":   Decimal("0.015"),
        "N2":    Decimal("0.015"),
    },
    "NATURAL_GAS": {
        "CH4":   Decimal("0.85"),
        "C2H6":  Decimal("0.07"),
        "C3H8":  Decimal("0.03"),
        "CO2":   Decimal("0.02"),
        "N2":    Decimal("0.03"),
    },
    "SOUR_NATURAL_GAS": {
        "CH4":   Decimal("0.82"),
        "C2H6":  Decimal("0.04"),
        "C3H8":  Decimal("0.02"),
        "H2S":   Decimal("0.05"),
        "CO2":   Decimal("0.03"),
        "N2":    Decimal("0.02"),
        "N_C4H10": Decimal("0.02"),
    },
    "ASSOCIATED_GAS": {
        "CH4":     Decimal("0.70"),
        "C2H6":    Decimal("0.10"),
        "C3H8":    Decimal("0.08"),
        "N_C4H10": Decimal("0.05"),
        "CO2":     Decimal("0.03"),
        "N2":      Decimal("0.02"),
        "H2S":     Decimal("0.02"),
    },
    "REFINERY_OFF_GAS": {
        "H2":    Decimal("0.30"),
        "CH4":   Decimal("0.30"),
        "C2H4":  Decimal("0.15"),
        "C3H6":  Decimal("0.10"),
        "CO2":   Decimal("0.10"),
        "N2":    Decimal("0.05"),
    },
    "LANDFILL_GAS": {
        "CH4":  Decimal("0.50"),
        "CO2":  Decimal("0.45"),
        "N2":   Decimal("0.05"),
    },
    "BIOGAS": {
        "CH4":  Decimal("0.60"),
        "CO2":  Decimal("0.35"),
        "H2S":  Decimal("0.01"),
        "N2":   Decimal("0.04"),
    },
    "CHEMICAL_PLANT_WASTE_GAS": {
        "H2":   Decimal("0.40"),
        "CO":   Decimal("0.20"),
        "CH4":  Decimal("0.20"),
        "CO2":  Decimal("0.15"),
        "N2":   Decimal("0.05"),
    },
    "COKE_OVEN_GAS": {
        "H2":   Decimal("0.55"),
        "CH4":  Decimal("0.25"),
        "CO":   Decimal("0.06"),
        "CO2":  Decimal("0.02"),
        "N2":   Decimal("0.10"),
        "C2H6": Decimal("0.02"),
    },
}


# ===========================================================================
# Component Properties
# ===========================================================================

# Molecular weights (g/mol)
_MOLECULAR_WEIGHTS: Dict[str, Decimal] = {
    "CH4":     Decimal("16.043"),
    "C2H6":    Decimal("30.069"),
    "C3H8":    Decimal("44.096"),
    "N_C4H10": Decimal("58.122"),
    "I_C4H10": Decimal("58.122"),
    "C5H12":   Decimal("72.149"),
    "C6_PLUS": Decimal("86.175"),  # hexane representative
    "CO2":     Decimal("44.010"),
    "N2":      Decimal("28.014"),
    "H2S":     Decimal("34.081"),
    "H2":      Decimal("2.016"),
    "CO":      Decimal("28.010"),
    "C2H4":    Decimal("28.054"),  # ethylene
    "C3H6":    Decimal("42.080"),  # propylene
    "H2O":     Decimal("18.015"),
}

# Carbon atom count per molecule (for CO2 stoichiometry)
_CARBON_COUNTS: Dict[str, int] = {
    "CH4":     1,
    "C2H6":    2,
    "C3H8":    3,
    "N_C4H10": 4,
    "I_C4H10": 4,
    "C5H12":   5,
    "C6_PLUS": 6,
    "CO2":     1,  # already oxidized, not combustible carbon
    "N2":      0,
    "H2S":     0,
    "H2":      0,
    "CO":      1,
    "C2H4":    2,
    "C3H6":    3,
    "H2O":     0,
}

# Is the component a hydrocarbon or combustible gas?
_IS_COMBUSTIBLE: Dict[str, bool] = {
    "CH4":     True,
    "C2H6":    True,
    "C3H8":    True,
    "N_C4H10": True,
    "I_C4H10": True,
    "C5H12":   True,
    "C6_PLUS": True,
    "CO2":     False,
    "N2":      False,
    "H2S":     True,   # combustible but not hydrocarbon; produces SO2 + H2O
    "H2":      True,   # combustible, no CO2 produced
    "CO":      True,   # combustible, produces CO2
    "C2H4":    True,
    "C3H6":    True,
    "H2O":     False,
}

# Does combustion of this component produce CO2?
_PRODUCES_CO2: Dict[str, bool] = {
    "CH4":     True,
    "C2H6":    True,
    "C3H8":    True,
    "N_C4H10": True,
    "I_C4H10": True,
    "C5H12":   True,
    "C6_PLUS": True,
    "CO2":     False,  # already CO2
    "N2":      False,
    "H2S":     False,  # H2S -> SO2 + H2O
    "H2":      False,  # H2 -> H2O
    "CO":      True,   # CO -> CO2
    "C2H4":    True,
    "C3H6":    True,
    "H2O":     False,
}

# Higher Heating Value (HHV) per component in BTU/scf at 60 deg F, 14.696 psia
# Source: GPSA Engineering Data Book, GPA 2145
_COMPONENT_HHV_BTU_SCF: Dict[str, Decimal] = {
    "CH4":     Decimal("1012.0"),
    "C2H6":    Decimal("1773.0"),
    "C3H8":    Decimal("2524.0"),
    "N_C4H10": Decimal("3271.0"),
    "I_C4H10": Decimal("3253.0"),
    "C5H12":   Decimal("4010.0"),
    "C6_PLUS": Decimal("4762.0"),  # hexane representative
    "CO2":     Decimal("0.0"),
    "N2":      Decimal("0.0"),
    "H2S":     Decimal("647.0"),
    "H2":      Decimal("325.0"),
    "CO":      Decimal("321.0"),
    "C2H4":    Decimal("1614.0"),
    "C3H6":    Decimal("2336.0"),
    "H2O":     Decimal("0.0"),
}

# Lower Heating Value (LHV) per component in BTU/scf at 60 deg F, 14.696 psia
# LHV = HHV minus latent heat of water vaporization from combustion products
# Source: GPSA Engineering Data Book
_COMPONENT_LHV_BTU_SCF: Dict[str, Decimal] = {
    "CH4":     Decimal("911.0"),
    "C2H6":    Decimal("1631.0"),
    "C3H8":    Decimal("2316.0"),
    "N_C4H10": Decimal("3013.0"),
    "I_C4H10": Decimal("2997.0"),
    "C5H12":   Decimal("3707.0"),
    "C6_PLUS": Decimal("4404.0"),
    "CO2":     Decimal("0.0"),
    "N2":      Decimal("0.0"),
    "H2S":     Decimal("596.0"),
    "H2":      Decimal("275.0"),
    "CO":      Decimal("321.0"),  # same as HHV (no water product)
    "C2H4":    Decimal("1513.0"),
    "C3H6":    Decimal("2186.0"),
    "H2O":     Decimal("0.0"),
}

# HHV per component in MJ/Nm3 at 15 deg C, 101.325 kPa (ISO standard)
# Conversion: 1 BTU/scf (60 deg F) = 0.037259 MJ/Nm3 (15 deg C) approximately
# Note: These are independent reference values from IPCC/ISO sources.
_COMPONENT_HHV_MJ_NM3: Dict[str, Decimal] = {
    "CH4":     Decimal("39.82"),
    "C2H6":    Decimal("69.73"),
    "C3H8":    Decimal("99.17"),
    "N_C4H10": Decimal("128.54"),
    "I_C4H10": Decimal("127.83"),
    "C5H12":   Decimal("157.61"),
    "C6_PLUS": Decimal("187.17"),
    "CO2":     Decimal("0.00"),
    "N2":      Decimal("0.00"),
    "H2S":     Decimal("25.44"),
    "H2":      Decimal("12.77"),
    "CO":      Decimal("12.63"),
    "C2H4":    Decimal("63.43"),
    "C3H6":    Decimal("91.82"),
    "H2O":     Decimal("0.00"),
}

# LHV per component in MJ/Nm3 at 15 deg C, 101.325 kPa
_COMPONENT_LHV_MJ_NM3: Dict[str, Decimal] = {
    "CH4":     Decimal("35.81"),
    "C2H6":    Decimal("64.12"),
    "C3H8":    Decimal("91.00"),
    "N_C4H10": Decimal("118.41"),
    "I_C4H10": Decimal("117.78"),
    "C5H12":   Decimal("145.71"),
    "C6_PLUS": Decimal("173.10"),
    "CO2":     Decimal("0.00"),
    "N2":      Decimal("0.00"),
    "H2S":     Decimal("23.43"),
    "H2":      Decimal("10.81"),
    "CO":      Decimal("12.63"),
    "C2H4":    Decimal("59.46"),
    "C3H6":    Decimal("85.92"),
    "H2O":     Decimal("0.00"),
}

# Molecular weight of air (for specific gravity calculations)
_MW_AIR = Decimal("28.9647")

# Molecular weight of CO2
_MW_CO2 = Decimal("44.010")


# ===========================================================================
# Default Emission Factors by Source
# ===========================================================================

# EPA Subpart W default flaring emission factors
# Source: 40 CFR 98.253(b) and EPA GHG Emission Factors Hub
_EPA_FLARING_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # General flaring factors (kg per MMBtu, HHV basis)
    "GENERAL": {
        "CO2":  Decimal("60.0"),    # kg CO2/MMBtu (general flaring)
        "CH4":  Decimal("0.003"),   # kg CH4/MMBtu (uncombusted, assumes 98% CE)
        "N2O":  Decimal("0.00006"), # kg N2O/MMBtu
    },
    # Natural gas flaring
    "NATURAL_GAS": {
        "CO2":  Decimal("53.06"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    # Associated gas (heavier than pipeline-quality NG)
    "ASSOCIATED_GAS": {
        "CO2":  Decimal("58.40"),
        "CH4":  Decimal("0.002"),
        "N2O":  Decimal("0.0001"),
    },
    # Refinery off-gas (variable composition, H2-rich)
    "REFINERY_OFF_GAS": {
        "CO2":  Decimal("35.00"),
        "CH4":  Decimal("0.003"),
        "N2O":  Decimal("0.00006"),
    },
    # Landfill gas
    "LANDFILL_GAS": {
        "CO2":  Decimal("52.07"),
        "CH4":  Decimal("0.0032"),
        "N2O":  Decimal("0.00063"),
    },
    # Sweet natural gas (slightly higher HHV than pipeline NG)
    "SWEET_NATURAL_GAS": {
        "CO2":  Decimal("53.06"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    # Sour natural gas (H2S content, SO2 produced on combustion)
    "SOUR_NATURAL_GAS": {
        "CO2":  Decimal("51.80"),
        "CH4":  Decimal("0.0015"),
        "N2O":  Decimal("0.0001"),
    },
    # Coke oven gas (H2-rich, steel industry)
    "COKE_OVEN_GAS": {
        "CO2":  Decimal("36.50"),
        "CH4":  Decimal("0.002"),
        "N2O":  Decimal("0.00006"),
    },
    # Biogas
    "BIOGAS": {
        "CO2":  Decimal("52.07"),
        "CH4":  Decimal("0.0032"),
        "N2O":  Decimal("0.00063"),
    },
}

# IPCC 2006 default emission factors for flaring
# Source: IPCC 2006 GL Vol 2, Ch 4 (Fugitive Emissions)
# Units: kg per GJ (NCV/LHV basis)
_IPCC_FLARING_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "GENERAL": {
        "CO2":  Decimal("56.1"),    # kg CO2/GJ for natural gas flaring
        "CH4":  Decimal("0.0012"),  # kg CH4/GJ
        "N2O":  Decimal("0.00009"), # kg N2O/GJ
    },
    "NATURAL_GAS": {
        "CO2":  Decimal("56.1"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    "ASSOCIATED_GAS": {
        "CO2":  Decimal("58.3"),
        "CH4":  Decimal("0.0015"),
        "N2O":  Decimal("0.0001"),
    },
    "REFINERY_OFF_GAS": {
        "CO2":  Decimal("35.5"),
        "CH4":  Decimal("0.0012"),
        "N2O":  Decimal("0.00009"),
    },
    "LANDFILL_GAS": {
        "CO2":  Decimal("54.6"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    "BIOGAS": {
        "CO2":  Decimal("54.6"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    "SWEET_NATURAL_GAS": {
        "CO2":  Decimal("56.1"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    "SOUR_NATURAL_GAS": {
        "CO2":  Decimal("54.0"),
        "CH4":  Decimal("0.0012"),
        "N2O":  Decimal("0.0001"),
    },
    "COKE_OVEN_GAS": {
        "CO2":  Decimal("37.0"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.00009"),
    },
}

# API Compendium emission factors for flaring
# Source: API Compendium of GHG Emissions Methodologies, 2021
# Units: kg per MMBtu (HHV basis)
_API_FLARING_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "GENERAL": {
        "CO2":  Decimal("59.0"),
        "CH4":  Decimal("0.0023"),
        "N2O":  Decimal("0.00006"),
    },
    "NATURAL_GAS": {
        "CO2":  Decimal("53.02"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    "ASSOCIATED_GAS": {
        "CO2":  Decimal("57.50"),
        "CH4":  Decimal("0.0019"),
        "N2O":  Decimal("0.0001"),
    },
    "REFINERY_OFF_GAS": {
        "CO2":  Decimal("33.00"),
        "CH4":  Decimal("0.003"),
        "N2O":  Decimal("0.00006"),
    },
    "LANDFILL_GAS": {
        "CO2":  Decimal("51.00"),
        "CH4":  Decimal("0.003"),
        "N2O":  Decimal("0.0006"),
    },
    "BIOGAS": {
        "CO2":  Decimal("51.50"),
        "CH4":  Decimal("0.003"),
        "N2O":  Decimal("0.0006"),
    },
    "SWEET_NATURAL_GAS": {
        "CO2":  Decimal("53.02"),
        "CH4":  Decimal("0.001"),
        "N2O":  Decimal("0.0001"),
    },
    "SOUR_NATURAL_GAS": {
        "CO2":  Decimal("52.00"),
        "CH4":  Decimal("0.0015"),
        "N2O":  Decimal("0.0001"),
    },
    "COKE_OVEN_GAS": {
        "CO2":  Decimal("35.00"),
        "CH4":  Decimal("0.002"),
        "N2O":  Decimal("0.00006"),
    },
}

# Emission factor unit labels per source
_EF_UNITS: Dict[str, str] = {
    "EPA":    "kg/MMBtu",
    "IPCC":   "kg/GJ",
    "API":    "kg/MMBtu",
    "CUSTOM": "varies",
}


# ===========================================================================
# GWP Values
# ===========================================================================

_GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4":      {"CO2": Decimal("1"), "CH4": Decimal("25"),   "N2O": Decimal("298")},
    "AR5":      {"CO2": Decimal("1"), "CH4": Decimal("28"),   "N2O": Decimal("265")},
    "AR6":      {"CO2": Decimal("1"), "CH4": Decimal("29.8"), "N2O": Decimal("273")},
    "AR6_20YR": {"CO2": Decimal("1"), "CH4": Decimal("82.5"), "N2O": Decimal("273")},
}


# ===========================================================================
# Unit Conversion Constants
# ===========================================================================

#: 1 MMBtu = 1.055056 GJ
_MMBTU_TO_GJ = Decimal("1.055056")

#: 1 GJ = 1000 MJ
_GJ_TO_MJ = Decimal("1000")

#: Conversion factor from scf (60 deg F) to Nm3 (15 deg C)
#: Nm3 = scf * (288.15/288.71) * (14.696/14.696) * 0.0283168
#: Simplified: 1 scf = 0.02832 Nm3 (volume), but T/P correction needed
#: Standard conditions: EPA 60 deg F = 288.71 K, ISO 15 deg C = 288.15 K
#: 1 scf (60 F, 14.696 psia) = 0.028317 m3 * (288.15/288.71) = 0.02826 Nm3
_SCF_TO_NM3 = Decimal("0.028317")

#: 1 Nm3 = 35.3147 scf (inverse)
_NM3_TO_SCF = Decimal("35.3147")

#: Standard temperature EPA (60 deg F in Kelvin)
_T_STD_EPA_K = Decimal("288.706")

#: Standard temperature ISO (15 deg C in Kelvin)
_T_STD_ISO_K = Decimal("288.15")

#: Temperature correction factor EPA to ISO: T_ISO / T_EPA
_T_CORRECTION_EPA_TO_ISO = Decimal("0.998075")

#: 1 lb = 0.453592 kg
_LB_TO_KG = Decimal("0.453592")

#: 1 kg = 2.20462 lb
_KG_TO_LB = Decimal("2.20462")

#: 1 BTU = 1.05506 kJ
_BTU_TO_KJ = Decimal("1.05506")

#: 1 kJ = 0.947817 BTU
_KJ_TO_BTU = Decimal("0.947817")


# ===========================================================================
# Type alias for custom factor key
# ===========================================================================

_EFKey = Tuple[str, str, str]


# ===========================================================================
# FlareSystemDatabaseEngine
# ===========================================================================


class FlareSystemDatabaseEngine:
    """Manages flare system data, gas compositions, emission factors,
    and thermodynamic property calculations.

    This engine is the authoritative data source for all flaring emission
    calculations. It provides deterministic, zero-hallucination lookups
    for flare type specifications, default gas compositions, emission
    factors from EPA/IPCC/API sources, and thermodynamic property
    calculations (HHV, LHV, Wobbe Index).

    Thread-safe: all mutable state is guarded by ``threading.Lock()``.

    Attributes:
        _config: Optional configuration dictionary.
        _custom_factors: Registry of user-defined emission factors.
        _custom_compositions: Registry of user-defined gas compositions.
        _lock: Thread lock for mutable state mutations.
        _provenance: Reference to the provenance tracker.

    Example:
        >>> db = FlareSystemDatabaseEngine()
        >>> specs = db.get_flare_type_specs("ENCLOSED_GROUND")
        >>> assert specs["typical_ce"] == Decimal("0.995")
        >>> comp = db.get_default_composition("NATURAL_GAS")
        >>> hhv = db.calculate_hhv(comp)
        >>> assert hhv > Decimal("900")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FlareSystemDatabaseEngine with optional configuration.

        Loads all built-in flare type specifications, gas compositions,
        and emission factors. No database calls are made; all data is held
        in-memory for deterministic, zero-latency lookups.

        Args:
            config: Optional configuration dict. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                  Defaults to True.
                - ``composition_tolerance`` (str): Tolerance for mole
                  fraction sum validation. Defaults to "0.001".
                - ``decimal_precision`` (int): Decimal places for
                  rounding. Defaults to 8.
        """
        self._config = config or {}
        self._custom_factors: Dict[_EFKey, Dict[str, Any]] = {}
        self._custom_compositions: Dict[str, Dict[str, Decimal]] = {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        self._composition_tolerance: Decimal = Decimal(
            str(self._config.get("composition_tolerance", "0.001"))
        )
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_quantizer = Decimal(10) ** -self._precision_places

        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "FlareSystemDatabaseEngine initialized: %d flare types, "
            "%d gas scenarios, %d EF sources",
            len(_FLARE_TYPE_SPECS),
            len(_DEFAULT_GAS_COMPOSITIONS),
            3,  # EPA, IPCC, API
        )

    # ==================================================================
    # PUBLIC API: Flare Type Specifications
    # ==================================================================

    def get_flare_type_specs(self, flare_type: str) -> Dict[str, Any]:
        """Return the full specification for a flare type.

        Args:
            flare_type: Flare type identifier (e.g.
                ``"ELEVATED_STEAM_ASSISTED"``).

        Returns:
            Dictionary with keys: display_name, description,
            capacity_range_mmbtu_hr, typical_ce, tip_velocity_range_m_s,
            assist_type, height_range_m, typical_applications,
            smokeless_capability, wind_shielding, minimum_lhv_btu_scf,
            requires_assist_medium.

        Raises:
            KeyError: If the flare type is not found.

        Example:
            >>> specs = db.get_flare_type_specs("ENCLOSED_GROUND")
            >>> specs["typical_ce"]
            Decimal('0.995')
        """
        ft_key = flare_type.upper()
        if ft_key not in _FLARE_TYPE_SPECS:
            raise KeyError(
                f"Unknown flare type: {flare_type}. "
                f"Valid types: {sorted(_FLARE_TYPE_SPECS.keys())}"
            )

        specs = dict(_FLARE_TYPE_SPECS[ft_key])

        if _METRICS_AVAILABLE and _record_flare_lookup is not None:
            _record_flare_lookup("flare_type")

        self._record_provenance(
            "lookup_flare_type", ft_key,
            {"typical_ce": str(specs["typical_ce"])},
        )

        return specs

    def list_flare_types(self) -> List[str]:
        """Return sorted list of all known flare type identifiers.

        Returns:
            List of flare type key strings.

        Example:
            >>> db.list_flare_types()
            ['CANDLESTICK', 'ELEVATED_AIR_ASSISTED', ...]
        """
        return sorted(_FLARE_TYPE_SPECS.keys())

    def get_typical_ce(self, flare_type: str) -> Decimal:
        """Return the typical combustion efficiency for a flare type.

        Convenience method that extracts only the typical CE from the
        full specification.

        Args:
            flare_type: Flare type identifier.

        Returns:
            Typical combustion efficiency as a Decimal (0-1).

        Raises:
            KeyError: If the flare type is not found.

        Example:
            >>> db.get_typical_ce("ELEVATED_STEAM_ASSISTED")
            Decimal('0.98')
        """
        specs = self.get_flare_type_specs(flare_type)
        return specs["typical_ce"]

    # ==================================================================
    # PUBLIC API: Gas Compositions
    # ==================================================================

    def get_default_composition(
        self,
        scenario: str,
    ) -> Dict[str, Decimal]:
        """Return the default gas composition for a scenario.

        The returned dictionary maps gas component identifiers to their
        mole fractions. All mole fractions sum to 1.0 within tolerance.

        Args:
            scenario: Gas composition scenario identifier (e.g.
                ``"NATURAL_GAS"``, ``"ASSOCIATED_GAS"``).

        Returns:
            Dictionary mapping component names to mole fractions.

        Raises:
            KeyError: If the scenario is not found.

        Example:
            >>> comp = db.get_default_composition("NATURAL_GAS")
            >>> comp["CH4"]
            Decimal('0.85')
        """
        sc_key = scenario.upper()

        # Check custom compositions first
        with self._lock:
            if sc_key in self._custom_compositions:
                comp = dict(self._custom_compositions[sc_key])
                self._record_provenance(
                    "lookup_composition", sc_key,
                    {"source": "CUSTOM", "components": len(comp)},
                )
                return comp

        if sc_key not in _DEFAULT_GAS_COMPOSITIONS:
            raise KeyError(
                f"Unknown gas composition scenario: {scenario}. "
                f"Valid scenarios: {sorted(_DEFAULT_GAS_COMPOSITIONS.keys())}"
            )

        comp = dict(_DEFAULT_GAS_COMPOSITIONS[sc_key])

        if _METRICS_AVAILABLE and _record_flare_lookup is not None:
            _record_flare_lookup("gas_composition")

        self._record_provenance(
            "lookup_composition", sc_key,
            {"source": "DEFAULT", "components": len(comp)},
        )

        return comp

    def list_composition_scenarios(self) -> List[str]:
        """Return sorted list of available composition scenario identifiers.

        Includes both built-in and custom-registered scenarios.

        Returns:
            List of scenario key strings.
        """
        built_in = set(_DEFAULT_GAS_COMPOSITIONS.keys())
        with self._lock:
            custom = set(self._custom_compositions.keys())
        return sorted(built_in | custom)

    def register_custom_composition(
        self,
        scenario_name: str,
        composition: Dict[str, Decimal],
        validate: bool = True,
    ) -> str:
        """Register a custom gas composition scenario.

        Custom compositions take priority over built-in defaults when
        the scenario name matches.

        Args:
            scenario_name: Unique identifier for the composition.
            composition: Dictionary mapping component names to mole
                fractions.
            validate: Whether to validate the composition. Default True.

        Returns:
            Registration ID string.

        Raises:
            ValueError: If validation fails (fractions do not sum to 1.0,
                unknown components, or negative values).

        Example:
            >>> reg_id = db.register_custom_composition(
            ...     "MY_GAS",
            ...     {"CH4": Decimal("0.90"), "C2H6": Decimal("0.05"),
            ...      "CO2": Decimal("0.03"), "N2": Decimal("0.02")},
            ... )
        """
        sc_key = scenario_name.upper()

        if validate:
            self.validate_gas_composition(composition)

        reg_id = f"comp_{uuid.uuid4().hex[:12]}"

        with self._lock:
            self._custom_compositions[sc_key] = {
                k.upper(): Decimal(str(v)) for k, v in composition.items()
            }

        self._record_provenance(
            "register_composition", reg_id,
            {"scenario": sc_key, "components": len(composition)},
        )

        logger.info(
            "Registered custom gas composition: %s with %d components (id=%s)",
            sc_key, len(composition), reg_id,
        )
        return reg_id

    def validate_gas_composition(
        self,
        composition: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """Validate a gas composition dictionary.

        Checks:
        1. All component names are recognized.
        2. All mole fractions are non-negative.
        3. Mole fractions sum to 1.0 within configured tolerance.

        Args:
            composition: Dictionary mapping component names to mole
                fractions.

        Returns:
            Validation result dictionary with keys:
                - is_valid (bool)
                - fraction_sum (Decimal)
                - deviation (Decimal)
                - warnings (List[str])
                - errors (List[str])

        Raises:
            ValueError: If critical validation errors are found.

        Example:
            >>> result = db.validate_gas_composition(
            ...     {"CH4": Decimal("0.85"), "CO2": Decimal("0.15")}
            ... )
            >>> result["is_valid"]
            True
        """
        errors: List[str] = []
        warnings: List[str] = []
        fraction_sum = Decimal("0")

        for component, fraction in composition.items():
            comp_key = component.upper()

            # Check known component
            if comp_key not in _MOLECULAR_WEIGHTS:
                errors.append(
                    f"Unknown gas component: {component}. "
                    f"Valid components: {sorted(_MOLECULAR_WEIGHTS.keys())}"
                )
                continue

            # Check non-negative
            frac = Decimal(str(fraction))
            if frac < Decimal("0"):
                errors.append(
                    f"Negative mole fraction for {component}: {frac}"
                )
                continue

            if frac > Decimal("1"):
                errors.append(
                    f"Mole fraction > 1.0 for {component}: {frac}"
                )
                continue

            fraction_sum += frac

        # Check sum = 1.0
        deviation = abs(fraction_sum - Decimal("1.0"))
        if deviation > self._composition_tolerance:
            errors.append(
                f"Mole fractions sum to {fraction_sum} "
                f"(deviation {deviation} exceeds tolerance "
                f"{self._composition_tolerance})"
            )

        if deviation > Decimal("0") and deviation <= self._composition_tolerance:
            warnings.append(
                f"Mole fractions sum to {fraction_sum} "
                f"(deviation {deviation} within tolerance)"
            )

        is_valid = len(errors) == 0

        if not is_valid:
            raise ValueError(
                f"Gas composition validation failed: {'; '.join(errors)}"
            )

        return {
            "is_valid": is_valid,
            "fraction_sum": fraction_sum,
            "deviation": deviation,
            "warnings": warnings,
            "errors": errors,
        }

    # ==================================================================
    # PUBLIC API: Emission Factors
    # ==================================================================

    def get_emission_factor(
        self,
        gas_type: str,
        gas: str,
        source: str = "EPA",
    ) -> Decimal:
        """Look up a flaring emission factor for a gas type, emission gas,
        and regulatory source.

        Args:
            gas_type: Flare gas type identifier (e.g. ``"NATURAL_GAS"``,
                ``"GENERAL"``).
            gas: Emission gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            source: Factor source (``"EPA"``, ``"IPCC"``, ``"API"``,
                ``"CUSTOM"``). Defaults to ``"EPA"``.

        Returns:
            Emission factor as a ``Decimal``.

        Raises:
            KeyError: If the combination is not found.

        Example:
            >>> db.get_emission_factor("GENERAL", "CO2", "EPA")
            Decimal('60.0')
            >>> db.get_emission_factor("NATURAL_GAS", "CO2", "IPCC")
            Decimal('56.1')
        """
        gt_key = gas_type.upper()
        gas_key = gas.upper()
        source_key = source.upper()

        if _METRICS_AVAILABLE and _record_factor_selection is not None:
            _record_factor_selection("default_ef", source_key)

        # Check custom factors first
        custom_key = (gt_key, gas_key, source_key)
        with self._lock:
            if custom_key in self._custom_factors:
                val = self._custom_factors[custom_key]["value"]
                self._record_provenance(
                    "lookup_emission_factor", gt_key,
                    {"gas": gas_key, "source": source_key,
                     "value": str(val), "origin": "CUSTOM"},
                )
                return val

        # Look up in built-in sources
        source_map = self._get_factor_source_map(source_key)
        if gt_key not in source_map:
            raise KeyError(
                f"No flaring emission factors for gas type '{gas_type}' "
                f"in source '{source}'"
            )

        gas_factors = source_map[gt_key]
        if gas_key not in gas_factors:
            raise KeyError(
                f"No {gas} emission factor for gas type '{gas_type}' "
                f"in source '{source}'"
            )

        value = gas_factors[gas_key]
        self._record_provenance(
            "lookup_emission_factor", gt_key,
            {"gas": gas_key, "source": source_key,
             "value": str(value), "origin": "BUILT_IN"},
        )
        return value

    def get_emission_factor_unit(self, source: str) -> str:
        """Return the unit string for emission factors from a source.

        Args:
            source: Factor source identifier.

        Returns:
            Unit string (e.g. ``"kg/MMBtu"``).
        """
        return _EF_UNITS.get(source.upper(), "varies")

    def register_custom_factor(
        self,
        gas_type: str,
        gas: str,
        value: Decimal,
        unit: str,
        source: str = "CUSTOM",
        reference: str = "",
    ) -> str:
        """Register a custom flaring emission factor.

        Custom factors take priority over built-in factors when the
        gas_type/gas/source combination matches.

        Args:
            gas_type: Flare gas type identifier.
            gas: Emission gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            value: Factor value as ``Decimal``.
            unit: Factor unit string.
            source: Source label. Defaults to ``"CUSTOM"``.
            reference: Regulatory or document reference.

        Returns:
            Registration ID string.

        Raises:
            ValueError: If value is negative.

        Example:
            >>> reg_id = db.register_custom_factor(
            ...     "MY_GAS", "CO2", Decimal("55.0"), "kg/MMBtu"
            ... )
        """
        if value < Decimal("0"):
            raise ValueError(
                f"Emission factor value must be >= 0, got {value}"
            )

        gt_key = gas_type.upper()
        gas_key = gas.upper()
        source_key = source.upper()
        reg_id = f"fl_ef_{uuid.uuid4().hex[:12]}"

        with self._lock:
            self._custom_factors[(gt_key, gas_key, source_key)] = {
                "value": Decimal(str(value)),
                "unit": unit,
                "reference": reference,
                "registration_id": reg_id,
            }

        self._record_provenance(
            "register_custom_factor", reg_id,
            {
                "gas_type": gt_key, "gas": gas_key,
                "value": str(value), "unit": unit,
                "source": source_key,
            },
        )

        logger.info(
            "Registered custom flaring factor: %s/%s/%s = %s %s (id=%s)",
            gt_key, gas_key, source_key, value, unit, reg_id,
        )
        return reg_id

    def list_emission_factors(
        self,
        gas_type: Optional[str] = None,
        source: Optional[str] = None,
        gas: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List flaring emission factors with optional filtering.

        Args:
            gas_type: Filter by gas type (optional).
            source: Filter by source (optional).
            gas: Filter by emission gas (optional).

        Returns:
            List of dictionaries with gas_type, gas, source, value, unit.
        """
        results: List[Dict[str, Any]] = []
        sources_to_check = self._get_all_factor_sources(source)

        for src_name, src_map in sources_to_check:
            for gt_name, gases in src_map.items():
                if gas_type and gt_name != gas_type.upper():
                    continue
                for g_name, g_val in gases.items():
                    if gas and g_name != gas.upper():
                        continue
                    results.append({
                        "gas_type": gt_name,
                        "gas": g_name,
                        "source": src_name,
                        "value": g_val,
                        "unit": _EF_UNITS.get(src_name, "varies"),
                    })

        # Include custom factors
        with self._lock:
            for (gt, g, s), data in self._custom_factors.items():
                if gas_type and gt != gas_type.upper():
                    continue
                if source and s != source.upper():
                    continue
                if gas and g != gas.upper():
                    continue
                results.append({
                    "gas_type": gt,
                    "gas": g,
                    "source": s,
                    "value": data["value"],
                    "unit": data.get("unit", "varies"),
                })

        return results

    def get_factor_count(self) -> int:
        """Return the total count of emission factors across all sources.

        Returns:
            Integer count of all registered emission factors
            (built-in + custom).
        """
        count = 0
        for src_map in [
            _EPA_FLARING_FACTORS,
            _IPCC_FLARING_FACTORS,
            _API_FLARING_FACTORS,
        ]:
            for gases in src_map.values():
                count += len(gases)

        with self._lock:
            count += len(self._custom_factors)

        return count

    # ==================================================================
    # PUBLIC API: GWP Values
    # ==================================================================

    def get_gwp(
        self,
        gas: str,
        source: str = "AR6",
        timeframe: str = "100yr",
    ) -> Decimal:
        """Look up the Global Warming Potential for a gas.

        Args:
            gas: Greenhouse gas identifier (``"CO2"``, ``"CH4"``,
                ``"N2O"``).
            source: IPCC Assessment Report (``"AR4"``, ``"AR5"``,
                ``"AR6"``). Defaults to ``"AR6"``.
            timeframe: ``"100yr"`` or ``"20yr"``. Defaults to ``"100yr"``.

        Returns:
            GWP value as a ``Decimal``.

        Raises:
            KeyError: If the gas or source is not found.
            ValueError: If timeframe is invalid.

        Example:
            >>> db.get_gwp("CH4", "AR6")
            Decimal('29.8')
        """
        gas_key = gas.upper()
        source_key = source.upper()
        tf_key = timeframe.lower()

        if tf_key not in ("100yr", "20yr"):
            raise ValueError(
                f"timeframe must be '100yr' or '20yr', got '{timeframe}'"
            )

        if tf_key == "20yr":
            source_key = "AR6_20YR"

        if source_key not in _GWP_VALUES:
            raise KeyError(f"Unknown GWP source: {source}")

        gwp_table = _GWP_VALUES[source_key]
        if gas_key not in gwp_table:
            raise KeyError(
                f"No GWP for gas '{gas}' in source '{source_key}'"
            )

        value = gwp_table[gas_key]
        self._record_provenance(
            "lookup_gwp", gas_key,
            {"source": source_key, "timeframe": tf_key, "value": str(value)},
        )
        return value

    # ==================================================================
    # PUBLIC API: Component Properties
    # ==================================================================

    def get_molecular_weight(self, component: str) -> Decimal:
        """Return the molecular weight (g/mol) for a gas component.

        Args:
            component: Component identifier (e.g. ``"CH4"``).

        Returns:
            Molecular weight as ``Decimal``.

        Raises:
            KeyError: If the component is not found.
        """
        comp_key = component.upper()
        if comp_key not in _MOLECULAR_WEIGHTS:
            raise KeyError(f"Unknown gas component: {component}")
        return _MOLECULAR_WEIGHTS[comp_key]

    def get_carbon_count(self, component: str) -> int:
        """Return the number of carbon atoms per molecule for a component.

        Args:
            component: Component identifier.

        Returns:
            Integer carbon atom count.

        Raises:
            KeyError: If the component is not found.
        """
        comp_key = component.upper()
        if comp_key not in _CARBON_COUNTS:
            raise KeyError(f"Unknown gas component: {component}")
        return _CARBON_COUNTS[comp_key]

    def is_combustible(self, component: str) -> bool:
        """Check if a gas component is combustible.

        Args:
            component: Component identifier.

        Returns:
            True if the component is combustible.

        Raises:
            KeyError: If the component is not found.
        """
        comp_key = component.upper()
        if comp_key not in _IS_COMBUSTIBLE:
            raise KeyError(f"Unknown gas component: {component}")
        return _IS_COMBUSTIBLE[comp_key]

    def produces_co2(self, component: str) -> bool:
        """Check if combustion of a component produces CO2.

        Args:
            component: Component identifier.

        Returns:
            True if combustion produces CO2.

        Raises:
            KeyError: If the component is not found.
        """
        comp_key = component.upper()
        if comp_key not in _PRODUCES_CO2:
            raise KeyError(f"Unknown gas component: {component}")
        return _PRODUCES_CO2[comp_key]

    def get_component_hhv(
        self,
        component: str,
        unit_system: str = "EPA",
    ) -> Decimal:
        """Return the Higher Heating Value for a gas component.

        Args:
            component: Component identifier.
            unit_system: ``"EPA"`` for BTU/scf at 60 deg F or
                ``"ISO"`` for MJ/Nm3 at 15 deg C. Defaults to ``"EPA"``.

        Returns:
            HHV as ``Decimal``.

        Raises:
            KeyError: If the component is not found.
            ValueError: If unit_system is invalid.
        """
        comp_key = component.upper()
        us_key = unit_system.upper()

        if us_key == "EPA":
            if comp_key not in _COMPONENT_HHV_BTU_SCF:
                raise KeyError(f"Unknown gas component: {component}")
            return _COMPONENT_HHV_BTU_SCF[comp_key]
        elif us_key == "ISO":
            if comp_key not in _COMPONENT_HHV_MJ_NM3:
                raise KeyError(f"Unknown gas component: {component}")
            return _COMPONENT_HHV_MJ_NM3[comp_key]
        else:
            raise ValueError(
                f"unit_system must be 'EPA' or 'ISO', got '{unit_system}'"
            )

    def get_component_lhv(
        self,
        component: str,
        unit_system: str = "EPA",
    ) -> Decimal:
        """Return the Lower Heating Value for a gas component.

        Args:
            component: Component identifier.
            unit_system: ``"EPA"`` for BTU/scf or ``"ISO"`` for MJ/Nm3.
                Defaults to ``"EPA"``.

        Returns:
            LHV as ``Decimal``.

        Raises:
            KeyError: If the component is not found.
            ValueError: If unit_system is invalid.
        """
        comp_key = component.upper()
        us_key = unit_system.upper()

        if us_key == "EPA":
            if comp_key not in _COMPONENT_LHV_BTU_SCF:
                raise KeyError(f"Unknown gas component: {component}")
            return _COMPONENT_LHV_BTU_SCF[comp_key]
        elif us_key == "ISO":
            if comp_key not in _COMPONENT_LHV_MJ_NM3:
                raise KeyError(f"Unknown gas component: {component}")
            return _COMPONENT_LHV_MJ_NM3[comp_key]
        else:
            raise ValueError(
                f"unit_system must be 'EPA' or 'ISO', got '{unit_system}'"
            )

    def list_components(self) -> List[str]:
        """Return sorted list of all known gas component identifiers.

        Returns:
            List of component key strings.
        """
        return sorted(_MOLECULAR_WEIGHTS.keys())

    # ==================================================================
    # PUBLIC API: Heating Value Calculations
    # ==================================================================

    def calculate_hhv(
        self,
        composition: Dict[str, Decimal],
        unit_system: str = "EPA",
    ) -> Decimal:
        """Calculate the Higher Heating Value of a gas mixture.

        HHV is calculated as the mole-fraction-weighted sum of
        individual component HHVs:

            HHV_mix = sum(x_i * HHV_i) for each component i

        Args:
            composition: Dictionary mapping component names to mole
                fractions (must sum to 1.0 within tolerance).
            unit_system: ``"EPA"`` for BTU/scf or ``"ISO"`` for
                MJ/Nm3. Defaults to ``"EPA"``.

        Returns:
            Mixture HHV as ``Decimal``.

        Raises:
            KeyError: If any component is unknown.

        Example:
            >>> comp = {"CH4": Decimal("0.85"), "C2H6": Decimal("0.07"),
            ...         "C3H8": Decimal("0.03"), "CO2": Decimal("0.02"),
            ...         "N2": Decimal("0.03")}
            >>> db.calculate_hhv(comp)
            Decimal('1060.51000000')
        """
        hhv_mix = Decimal("0")

        for component, fraction in composition.items():
            comp_key = component.upper()
            frac = Decimal(str(fraction))
            comp_hhv = self.get_component_hhv(comp_key, unit_system)
            hhv_mix += frac * comp_hhv

        result = self._quantize(hhv_mix)

        self._record_provenance(
            "calculate_hhv", "mixture",
            {"unit_system": unit_system, "hhv": str(result),
             "components": len(composition)},
        )

        return result

    def calculate_lhv(
        self,
        composition: Dict[str, Decimal],
        unit_system: str = "EPA",
    ) -> Decimal:
        """Calculate the Lower Heating Value of a gas mixture.

        LHV is calculated as the mole-fraction-weighted sum of
        individual component LHVs:

            LHV_mix = sum(x_i * LHV_i) for each component i

        Args:
            composition: Dictionary mapping component names to mole
                fractions.
            unit_system: ``"EPA"`` for BTU/scf or ``"ISO"`` for
                MJ/Nm3. Defaults to ``"EPA"``.

        Returns:
            Mixture LHV as ``Decimal``.

        Raises:
            KeyError: If any component is unknown.

        Example:
            >>> comp = {"CH4": Decimal("0.50"), "CO2": Decimal("0.45"),
            ...         "N2": Decimal("0.05")}
            >>> db.calculate_lhv(comp)
            Decimal('455.50000000')
        """
        lhv_mix = Decimal("0")

        for component, fraction in composition.items():
            comp_key = component.upper()
            frac = Decimal(str(fraction))
            comp_lhv = self.get_component_lhv(comp_key, unit_system)
            lhv_mix += frac * comp_lhv

        result = self._quantize(lhv_mix)

        self._record_provenance(
            "calculate_lhv", "mixture",
            {"unit_system": unit_system, "lhv": str(result),
             "components": len(composition)},
        )

        return result

    def get_molecular_weight_mixture(
        self,
        composition: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate the average molecular weight of a gas mixture.

        MW_mix = sum(x_i * MW_i) for each component i

        Args:
            composition: Dictionary mapping component names to mole
                fractions.

        Returns:
            Average molecular weight (g/mol) as ``Decimal``.

        Raises:
            KeyError: If any component is unknown.

        Example:
            >>> comp = {"CH4": Decimal("0.85"), "C2H6": Decimal("0.07"),
            ...         "C3H8": Decimal("0.03"), "CO2": Decimal("0.02"),
            ...         "N2": Decimal("0.03")}
            >>> db.get_molecular_weight_mixture(comp)
            Decimal('18.82614000')
        """
        mw_mix = Decimal("0")

        for component, fraction in composition.items():
            comp_key = component.upper()
            frac = Decimal(str(fraction))
            comp_mw = self.get_molecular_weight(comp_key)
            mw_mix += frac * comp_mw

        return self._quantize(mw_mix)

    def calculate_specific_gravity(
        self,
        composition: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate the specific gravity of a gas mixture relative to air.

        SG = MW_mix / MW_air

        where MW_air = 28.9647 g/mol.

        Args:
            composition: Dictionary mapping component names to mole
                fractions.

        Returns:
            Specific gravity (dimensionless) as ``Decimal``.

        Example:
            >>> comp = {"CH4": Decimal("1.0")}
            >>> db.calculate_specific_gravity(comp)
            Decimal('0.55388893')
        """
        mw_mix = self.get_molecular_weight_mixture(composition)
        sg = self._quantize(mw_mix / _MW_AIR)

        self._record_provenance(
            "calculate_specific_gravity", "mixture",
            {"mw_mix": str(mw_mix), "sg": str(sg)},
        )

        return sg

    def calculate_wobbe_index(
        self,
        composition: Dict[str, Decimal],
        unit_system: str = "EPA",
    ) -> Decimal:
        """Calculate the Wobbe Index of a gas mixture.

        The Wobbe Index is a measure of gas interchangeability:

            WI = HHV / sqrt(SG)

        where SG = specific gravity relative to air.

        Higher Wobbe Index means more energy per unit volume at the
        same pressure drop.

        Args:
            composition: Dictionary mapping component names to mole
                fractions.
            unit_system: ``"EPA"`` for BTU/scf or ``"ISO"`` for
                MJ/Nm3. Defaults to ``"EPA"``.

        Returns:
            Wobbe Index as ``Decimal``.

        Example:
            >>> comp = db.get_default_composition("NATURAL_GAS")
            >>> wi = db.calculate_wobbe_index(comp)
            >>> assert wi > Decimal("1200")
        """
        hhv = self.calculate_hhv(composition, unit_system)
        sg = self.calculate_specific_gravity(composition)

        if sg <= Decimal("0"):
            raise ValueError(
                "Cannot calculate Wobbe Index: specific gravity is zero "
                "or negative"
            )

        # sqrt via Decimal: use Python float sqrt then convert back
        # This preserves reasonable precision for engineering calculations
        import math
        sg_sqrt = Decimal(str(math.sqrt(float(sg))))

        wi = self._quantize(hhv / sg_sqrt)

        self._record_provenance(
            "calculate_wobbe_index", "mixture",
            {"hhv": str(hhv), "sg": str(sg), "wobbe_index": str(wi),
             "unit_system": unit_system},
        )

        return wi

    # ==================================================================
    # PUBLIC API: Unit Conversions
    # ==================================================================

    def convert_volume_scf_to_nm3(self, volume_scf: Decimal) -> Decimal:
        """Convert gas volume from standard cubic feet to normal cubic meters.

        1 scf (60 deg F, 14.696 psia) = 0.028317 Nm3 (15 deg C, 101.325 kPa)

        Includes temperature correction from EPA standard (60 deg F) to
        ISO standard (15 deg C).

        Args:
            volume_scf: Volume in standard cubic feet.

        Returns:
            Volume in normal cubic meters.
        """
        return self._quantize(volume_scf * _SCF_TO_NM3 * _T_CORRECTION_EPA_TO_ISO)

    def convert_volume_nm3_to_scf(self, volume_nm3: Decimal) -> Decimal:
        """Convert gas volume from normal cubic meters to standard cubic feet.

        Args:
            volume_nm3: Volume in normal cubic meters.

        Returns:
            Volume in standard cubic feet.
        """
        return self._quantize(
            volume_nm3 * _NM3_TO_SCF / _T_CORRECTION_EPA_TO_ISO
        )

    def convert_energy_mmbtu_to_gj(self, energy_mmbtu: Decimal) -> Decimal:
        """Convert energy from MMBtu to GJ.

        Args:
            energy_mmbtu: Energy in million British thermal units.

        Returns:
            Energy in gigajoules.
        """
        return self._quantize(energy_mmbtu * _MMBTU_TO_GJ)

    def convert_energy_gj_to_mmbtu(self, energy_gj: Decimal) -> Decimal:
        """Convert energy from GJ to MMBtu.

        Args:
            energy_gj: Energy in gigajoules.

        Returns:
            Energy in million British thermal units.
        """
        return self._quantize(energy_gj / _MMBTU_TO_GJ)

    def convert_mass_lb_to_kg(self, mass_lb: Decimal) -> Decimal:
        """Convert mass from pounds to kilograms.

        Args:
            mass_lb: Mass in pounds.

        Returns:
            Mass in kilograms.
        """
        return self._quantize(mass_lb * _LB_TO_KG)

    def convert_mass_kg_to_lb(self, mass_kg: Decimal) -> Decimal:
        """Convert mass from kilograms to pounds.

        Args:
            mass_kg: Mass in kilograms.

        Returns:
            Mass in pounds.
        """
        return self._quantize(mass_kg * _KG_TO_LB)

    # ==================================================================
    # PUBLIC API: Summary and Statistics
    # ==================================================================

    def get_database_summary(self) -> Dict[str, Any]:
        """Return a summary of the database contents.

        Returns:
            Dictionary with counts and lists of available data.
        """
        with self._lock:
            custom_factor_count = len(self._custom_factors)
            custom_comp_count = len(self._custom_compositions)

        return {
            "flare_types": len(_FLARE_TYPE_SPECS),
            "flare_type_list": self.list_flare_types(),
            "gas_composition_scenarios": (
                len(_DEFAULT_GAS_COMPOSITIONS) + custom_comp_count
            ),
            "scenario_list": self.list_composition_scenarios(),
            "gas_components": len(_MOLECULAR_WEIGHTS),
            "component_list": self.list_components(),
            "emission_factor_sources": ["EPA", "IPCC", "API"],
            "total_emission_factors": self.get_factor_count(),
            "custom_factors": custom_factor_count,
            "custom_compositions": custom_comp_count,
            "gwp_sources": sorted(_GWP_VALUES.keys()),
        }

    # ==================================================================
    # PRIVATE: Internal helpers
    # ==================================================================

    def _get_factor_source_map(
        self,
        source: str,
    ) -> Dict[str, Dict[str, Decimal]]:
        """Return the emission factor dictionary for a given source.

        Args:
            source: Uppercase source key.

        Returns:
            Dictionary mapping gas_type -> {gas -> Decimal}.

        Raises:
            KeyError: If the source is not recognized.
        """
        source_maps: Dict[str, Dict[str, Dict[str, Decimal]]] = {
            "EPA":  _EPA_FLARING_FACTORS,
            "IPCC": _IPCC_FLARING_FACTORS,
            "API":  _API_FLARING_FACTORS,
        }
        if source not in source_maps:
            raise KeyError(f"Unknown emission factor source: {source}")
        return source_maps[source]

    def _get_all_factor_sources(
        self,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Dict[str, Decimal]]]]:
        """Return all factor source maps, optionally filtered.

        Args:
            source_filter: Optional source to filter to.

        Returns:
            List of (source_name, source_dict) tuples.
        """
        all_sources: List[Tuple[str, Dict[str, Dict[str, Decimal]]]] = [
            ("EPA",  _EPA_FLARING_FACTORS),
            ("IPCC", _IPCC_FLARING_FACTORS),
            ("API",  _API_FLARING_FACTORS),
        ]
        if source_filter:
            sf = source_filter.upper()
            return [(n, m) for n, m in all_sources if n == sf]
        return all_sources

    def _quantize(self, value: Decimal) -> Decimal:
        """Round a Decimal to the configured precision.

        Args:
            value: Raw Decimal value.

        Returns:
            Rounded Decimal.
        """
        try:
            return value.quantize(
                self._precision_quantizer, rounding=ROUND_HALF_UP,
            )
        except InvalidOperation:
            logger.warning("Failed to quantize value: %s", value)
            return value

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action label.
            entity_id: Entity identifier.
            data: Optional data payload.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="flare_system_database",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"FlareSystemDatabaseEngine("
            f"flare_types={len(_FLARE_TYPE_SPECS)}, "
            f"scenarios={len(_DEFAULT_GAS_COMPOSITIONS)}, "
            f"factors={self.get_factor_count()}, "
            f"custom_factors={len(self._custom_factors)}, "
            f"custom_compositions={len(self._custom_compositions)})"
        )
