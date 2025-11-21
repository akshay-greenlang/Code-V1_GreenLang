# -*- coding: utf-8 -*-
"""
greenlang/data/wtt_emission_factors.py

Well-to-Tank (WTT) Upstream Emission Factors

WTT includes:
- Fuel extraction (e.g., crude oil drilling, natural gas extraction)
- Processing/refining (e.g., crude → diesel, natural gas processing)
- Transportation (pipeline, truck, rail)
- Distribution

Sources:
- GREET Model (Argonne National Laboratory)
- EU JRC Well-to-Wheels Analysis
- UK BEIS Conversion Factors 2024
- IPCC Guidelines for National GHG Inventories

Boundaries:
- combustion: Direct combustion only (tank-to-wheel)
- WTT: Upstream only (well-to-tank)
- WTW: Full lifecycle (WTT + combustion)

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Dict, Tuple

# WTT Emission Factors (kgCO2e per unit of fuel delivered)
# Format: (fuel_type, unit, country) -> (WTT_factor, source)

WTT_FACTORS: Dict[Tuple[str, str, str], Tuple[float, str]] = {
    # ==================== DIESEL ====================
    # Diesel WTT ~20% of combustion emissions
    ("diesel", "gallons", "US"): (
        2.04,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Petroleum diesel upstream"
    ),
    ("diesel", "liters", "UK"): (
        0.54,  # kgCO2e/liter (WTT)
        "UK BEIS 2024 - Diesel WTT"
    ),
    ("diesel", "liters", "EU"): (
        0.56,  # kgCO2e/liter (WTT)
        "JRC WTW 2024 - Diesel upstream EU average"
    ),

    # ==================== GASOLINE ====================
    # Gasoline WTT ~18% of combustion emissions
    ("gasoline", "gallons", "US"): (
        1.58,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Gasoline upstream"
    ),
    ("gasoline", "liters", "UK"): (
        0.42,  # kgCO2e/liter (WTT)
        "UK BEIS 2024 - Gasoline WTT"
    ),
    ("gasoline", "liters", "EU"): (
        0.44,  # kgCO2e/liter (WTT)
        "JRC WTW 2024 - Gasoline upstream EU average"
    ),

    # ==================== NATURAL GAS ====================
    # Natural gas WTT ~15-20% of combustion (methane leakage is key)
    ("natural_gas", "therms", "US"): (
        0.95,  # kgCO2e/therm (WTT)
        "GREET 2024 - Natural gas upstream (includes methane leakage)"
    ),
    ("natural_gas", "kWh", "UK"): (
        0.031,  # kgCO2e/kWh (WTT)
        "UK BEIS 2024 - Natural gas WTT"
    ),
    ("natural_gas", "m3", "EU"): (
        2.10,  # kgCO2e/m3 (WTT)
        "JRC WTW 2024 - Natural gas upstream EU"
    ),

    # ==================== COAL ====================
    # Coal WTT ~5-10% of combustion (lower than oil/gas)
    ("coal", "tons", "US"): (
        180.0,  # kgCO2e/ton (WTT)
        "GREET 2024 - Coal mining and transport"
    ),
    ("coal", "tonnes", "UK"): (
        180.0,  # kgCO2e/tonne (WTT)
        "UK BEIS 2024 - Coal WTT"
    ),

    # ==================== ELECTRICITY ====================
    # Electricity WTT = transmission losses (typically 5-8%)
    ("electricity", "kWh", "US"): (
        0.029,  # kgCO2e/kWh (WTT - transmission losses)
        "EPA eGRID 2024 - Transmission and distribution losses"
    ),
    ("electricity", "kWh", "UK"): (
        0.016,  # kgCO2e/kWh (WTT - T&D losses)
        "UK BEIS 2024 - Electricity WTT"
    ),

    # ==================== BIOFUELS ====================
    # Biodiesel WTT includes farming, processing
    ("biodiesel", "gallons", "US"): (
        0.85,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Biodiesel (soy) upstream"
    ),

    # Ethanol WTT
    ("ethanol", "gallons", "US"): (
        0.62,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Corn ethanol upstream"
    ),

    # ==================== LPG/PROPANE ====================
    ("propane", "gallons", "US"): (
        0.92,  # kgCO2e/gallon (WTT)
        "GREET 2024 - LPG upstream"
    ),
    ("lpg", "liters", "UK"): (
        0.23,  # kgCO2e/liter (WTT)
        "UK BEIS 2024 - LPG WTT"
    ),

    # ==================== FUEL OIL ====================
    ("fuel_oil", "gallons", "US"): (
        2.15,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Fuel oil upstream"
    ),

    # ==================== JET FUEL ====================
    ("jet_fuel", "gallons", "US"): (
        1.95,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Jet fuel upstream"
    ),

    # ==================== KEROSENE ====================
    ("kerosene", "gallons", "US"): (
        1.88,  # kgCO2e/gallon (WTT)
        "GREET 2024 - Kerosene upstream"
    ),

    # ==================== LNG ====================
    ("lng", "gallons", "US"): (
        1.12,  # kgCO2e/gallon (WTT)
        "GREET 2024 - LNG upstream (liquefaction + transport)"
    ),
}


def get_wtt_factor(
    fuel_type: str,
    unit: str,
    country: str = "US"
) -> Tuple[float, str]:
    """
    Get WTT (upstream) emission factor.

    Args:
        fuel_type: Fuel type
        unit: Unit
        country: Country code

    Returns:
        Tuple of (wtt_factor, source)
        Returns (0.0, "No WTT data") if not found
    """
    key = (fuel_type.lower(), unit.lower(), country.upper())

    if key in WTT_FACTORS:
        return WTT_FACTORS[key]

    # Try without country (use US as default)
    key_default = (fuel_type.lower(), unit.lower(), "US")
    if key_default in WTT_FACTORS:
        return WTT_FACTORS[key_default]

    # No WTT data available
    return (0.0, "No WTT data available")


def calculate_wtw_factor(
    combustion_factor: float,
    wtt_factor: float
) -> float:
    """
    Calculate Well-to-Wheel (WTW) factor.

    WTW = WTT + Combustion

    Args:
        combustion_factor: Combustion (tank-to-wheel) emission factor
        wtt_factor: WTT (well-to-tank) emission factor

    Returns:
        WTW emission factor
    """
    return combustion_factor + wtt_factor


def get_wtt_percentage(
    combustion_factor: float,
    wtt_factor: float
) -> float:
    """
    Calculate WTT as percentage of total lifecycle.

    Args:
        combustion_factor: Combustion emission factor
        wtt_factor: WTT emission factor

    Returns:
        WTT percentage (0-100)
    """
    total = combustion_factor + wtt_factor
    if total == 0:
        return 0.0
    return (wtt_factor / total) * 100


# ==================== TYPICAL WTT RATIOS ====================

TYPICAL_WTT_RATIOS = {
    "diesel": 0.20,          # WTT is ~20% of combustion
    "gasoline": 0.18,        # WTT is ~18% of combustion
    "natural_gas": 0.18,     # WTT is ~18% of combustion (higher if methane leakage)
    "coal": 0.08,            # WTT is ~8% of combustion
    "electricity": 0.08,     # WTT is ~8% of combustion (T&D losses)
    "biodiesel": 0.15,       # WTT is ~15% of combustion
    "ethanol": 0.12,         # WTT is ~12% of combustion
    "propane": 0.16,         # WTT is ~16% of combustion
    "fuel_oil": 0.20,        # WTT is ~20% of combustion
    "jet_fuel": 0.19,        # WTT is ~19% of combustion
}


def estimate_wtt_factor(
    fuel_type: str,
    combustion_factor: float
) -> float:
    """
    Estimate WTT factor using typical ratios.

    Used as fallback when specific WTT data not available.

    Args:
        fuel_type: Fuel type
        combustion_factor: Combustion emission factor

    Returns:
        Estimated WTT factor
    """
    ratio = TYPICAL_WTT_RATIOS.get(fuel_type.lower(), 0.15)  # Default 15%
    return combustion_factor * ratio
