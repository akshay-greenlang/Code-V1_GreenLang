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
    ("electricity", "kwh", "US"): (
        0.029,  # kgCO2e/kWh (WTT - transmission losses)
        "EPA eGRID 2024 - Transmission and distribution losses"
    ),
    ("electricity", "kwh", "UK"): (
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

    # ====================================================================
    # REGIONAL WTT FACTORS - MAJOR ECONOMIES
    # ====================================================================

    # ==================== CHINA ====================
    # China is the world's largest energy consumer
    # Sources: GREET-China, IEA China Energy Outlook, Tsinghua LCIA Database

    ("diesel", "liters", "CN"): (
        0.62,  # kgCO2e/liter (WTT) - Higher due to longer supply chains
        "GREET-China 2024 / IEA China - Diesel upstream (uncertainty: +/-12%)"
    ),
    ("gasoline", "liters", "CN"): (
        0.51,  # kgCO2e/liter (WTT)
        "GREET-China 2024 / IEA China - Gasoline upstream (uncertainty: +/-10%)"
    ),
    ("natural_gas", "m3", "CN"): (
        2.45,  # kgCO2e/m3 (WTT) - Higher methane leakage in pipeline network
        "IEA China Energy Outlook 2024 - Natural gas upstream (uncertainty: +/-15%)"
    ),
    ("coal", "tonnes", "CN"): (
        165.0,  # kgCO2e/tonne (WTT) - Domestic mining, shorter transport
        "China Coal Industry Association / GREET-China 2024 (uncertainty: +/-8%)"
    ),
    ("electricity", "kwh", "CN"): (
        0.038,  # kgCO2e/kWh (WTT) - Higher T&D losses (~6.5%)
        "China Electricity Council 2024 - T&D losses (uncertainty: +/-5%)"
    ),

    # ==================== INDIA ====================
    # India is the 3rd largest energy consumer
    # Sources: India GHG Platform, IEA India Energy Outlook, CEA Reports

    ("diesel", "liters", "IN"): (
        0.65,  # kgCO2e/liter (WTT) - Import-heavy, longer supply chains
        "India GHG Platform 2024 / IEA India - Diesel upstream (uncertainty: +/-15%)"
    ),
    ("gasoline", "liters", "IN"): (
        0.53,  # kgCO2e/liter (WTT)
        "India GHG Platform 2024 / IEA India - Gasoline upstream (uncertainty: +/-12%)"
    ),
    ("natural_gas", "m3", "IN"): (
        2.65,  # kgCO2e/m3 (WTT) - LNG imports dominate, high liquefaction overhead
        "IEA India Energy Outlook 2024 - Natural gas upstream (uncertainty: +/-18%)"
    ),
    ("coal", "tonnes", "IN"): (
        195.0,  # kgCO2e/tonne (WTT) - Domestic mining with higher extraction intensity
        "India Coal Ministry / CEA 2024 - Coal mining upstream (uncertainty: +/-10%)"
    ),
    ("electricity", "kwh", "IN"): (
        0.052,  # kgCO2e/kWh (WTT) - Higher T&D losses (~20% in some regions)
        "CEA India 2024 - Transmission and distribution losses (uncertainty: +/-8%)"
    ),

    # ==================== INDONESIA ====================
    # Indonesia - 4th most populous, significant coal/palm oil producer
    # Sources: Indonesia Ministry of Energy, IEA Southeast Asia, GREET adapted

    ("diesel", "liters", "ID"): (
        0.58,  # kgCO2e/liter (WTT) - Domestic refining with imports
        "Indonesia Ministry of Energy 2024 / IEA SEA - Diesel upstream (uncertainty: +/-14%)"
    ),
    ("gasoline", "liters", "ID"): (
        0.48,  # kgCO2e/liter (WTT)
        "Indonesia Ministry of Energy 2024 / IEA SEA - Gasoline upstream (uncertainty: +/-12%)"
    ),
    ("natural_gas", "m3", "ID"): (
        2.20,  # kgCO2e/m3 (WTT) - Domestic production, lower transport distances
        "IEA Southeast Asia Energy Outlook 2024 - Natural gas upstream (uncertainty: +/-15%)"
    ),
    ("coal", "tonnes", "ID"): (
        145.0,  # kgCO2e/tonne (WTT) - Major exporter, efficient surface mining
        "Indonesia Coal Mining Association 2024 - Coal upstream (uncertainty: +/-10%)"
    ),
    ("electricity", "kwh", "ID"): (
        0.045,  # kgCO2e/kWh (WTT) - Higher T&D losses (~11%), island grid challenges
        "PLN Indonesia 2024 / IEA SEA - T&D losses (uncertainty: +/-10%)"
    ),

    # ==================== BRAZIL ====================
    # Brazil - Major biofuel producer, large hydropower share
    # Sources: CETESB, Brazilian GHG Protocol, IEA Brazil

    ("diesel", "liters", "BR"): (
        0.52,  # kgCO2e/liter (WTT) - Pre-salt oil, efficient extraction
        "CETESB 2024 / Brazilian GHG Protocol - Diesel upstream (uncertainty: +/-10%)"
    ),
    ("gasoline", "liters", "BR"): (
        0.43,  # kgCO2e/liter (WTT) - Domestic production
        "CETESB 2024 / Brazilian GHG Protocol - Gasoline upstream (uncertainty: +/-10%)"
    ),
    ("ethanol", "liters", "BR"): (
        0.12,  # kgCO2e/liter (WTT) - Sugarcane ethanol, highly efficient
        "CETESB 2024 / RenovaBio - Sugarcane ethanol upstream (uncertainty: +/-8%)"
    ),
    ("electricity", "kwh", "BR"): (
        0.008,  # kgCO2e/kWh (WTT) - Low T&D losses (~8%), clean grid
        "ANEEL Brazil 2024 - Transmission and distribution losses (uncertainty: +/-5%)"
    ),

    # ==================== ADDITIONAL REGIONAL FACTORS ====================

    # Japan - High LNG imports
    ("natural_gas", "m3", "JP"): (
        2.85,  # kgCO2e/m3 (WTT) - 100% LNG imports, long shipping distances
        "Japan METI 2024 / IEA Japan - LNG upstream (uncertainty: +/-12%)"
    ),
    ("electricity", "kwh", "JP"): (
        0.022,  # kgCO2e/kWh (WTT) - Efficient T&D system
        "Japan METI 2024 - T&D losses (uncertainty: +/-5%)"
    ),

    # South Korea - Similar to Japan
    ("natural_gas", "m3", "KR"): (
        2.80,  # kgCO2e/m3 (WTT) - LNG imports
        "Korea Energy Agency 2024 / IEA Korea - LNG upstream (uncertainty: +/-12%)"
    ),
    ("electricity", "kwh", "KR"): (
        0.020,  # kgCO2e/kWh (WTT) - Efficient T&D
        "Korea Energy Agency 2024 - T&D losses (uncertainty: +/-5%)"
    ),

    # Australia - Major coal/LNG exporter
    ("coal", "tonnes", "AU"): (
        135.0,  # kgCO2e/tonne (WTT) - Efficient mining operations
        "Australian Government DISER 2024 - Coal mining upstream (uncertainty: +/-8%)"
    ),
    ("natural_gas", "m3", "AU"): (
        1.95,  # kgCO2e/m3 (WTT) - Domestic production, short supply chains
        "Australian Government DISER 2024 - Natural gas upstream (uncertainty: +/-10%)"
    ),

    # Canada - Oil sands have higher upstream emissions
    ("diesel", "liters", "CA"): (
        0.68,  # kgCO2e/liter (WTT) - Oil sands extraction intensive
        "Environment Canada 2024 / GREET - Diesel upstream (uncertainty: +/-15%)"
    ),
    ("gasoline", "liters", "CA"): (
        0.55,  # kgCO2e/liter (WTT) - Oil sands blend
        "Environment Canada 2024 / GREET - Gasoline upstream (uncertainty: +/-15%)"
    ),
    ("natural_gas", "m3", "CA"): (
        1.85,  # kgCO2e/m3 (WTT) - Domestic production
        "Environment Canada 2024 - Natural gas upstream (uncertainty: +/-10%)"
    ),

    # Mexico
    ("diesel", "liters", "MX"): (
        0.60,  # kgCO2e/liter (WTT)
        "SEMARNAT Mexico 2024 / IEA - Diesel upstream (uncertainty: +/-12%)"
    ),
    ("gasoline", "liters", "MX"): (
        0.49,  # kgCO2e/liter (WTT)
        "SEMARNAT Mexico 2024 / IEA - Gasoline upstream (uncertainty: +/-12%)"
    ),

    # Russia - Major oil/gas producer
    ("diesel", "liters", "RU"): (
        0.58,  # kgCO2e/liter (WTT)
        "IEA Russia 2024 - Diesel upstream (uncertainty: +/-18%)"
    ),
    ("natural_gas", "m3", "RU"): (
        2.35,  # kgCO2e/m3 (WTT) - Long pipeline distances, methane leakage
        "IEA Russia 2024 - Natural gas upstream (uncertainty: +/-20%)"
    ),

    # Saudi Arabia - Efficient extraction
    ("diesel", "liters", "SA"): (
        0.48,  # kgCO2e/liter (WTT) - Low extraction energy
        "Saudi Aramco Sustainability Report 2024 / IEA (uncertainty: +/-10%)"
    ),
    ("gasoline", "liters", "SA"): (
        0.40,  # kgCO2e/liter (WTT)
        "Saudi Aramco Sustainability Report 2024 / IEA (uncertainty: +/-10%)"
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
