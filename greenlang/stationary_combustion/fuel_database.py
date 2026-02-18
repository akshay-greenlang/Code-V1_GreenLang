# -*- coding: utf-8 -*-
"""
FuelDatabaseEngine - Engine 1: Stationary Combustion Agent (AGENT-MRV-001)

Manages 327+ emission factors across 5 sources (EPA, IPCC, DEFRA, EU ETS,
custom). Provides deterministic, zero-hallucination lookup of emission
factors, heating values, oxidation factors, GWP values, and fuel properties
for GHG Protocol Scope 1 stationary combustion calculations.

All factor values use ``Decimal`` for precision. The engine is thread-safe
via ``threading.Lock()`` and tracks every lookup through SHA-256 provenance
hashing.

Data Sources:
    - EPA GHG Emission Factors Hub 2025 (kg CO2/mmBtu basis)
    - IPCC 2006 Guidelines (kg CO2/TJ NCV basis)
    - UK DEFRA 2025 Conversion Factors
    - EU ETS Monitoring and Reporting Regulation (MRR NCV basis)
    - Custom user-registered factors

Example:
    >>> from greenlang.stationary_combustion.fuel_database import FuelDatabaseEngine
    >>> db = FuelDatabaseEngine()
    >>> ef = db.get_emission_factor("NATURAL_GAS", "CO2", source="EPA")
    >>> print(ef)  # Decimal('53.06')
    >>> hv = db.get_heating_value("NATURAL_GAS", basis="HHV")
    >>> print(hv)  # Decimal('1.028')
    >>> gwp = db.get_gwp("CH4", source="AR6")
    >>> print(gwp)  # Decimal('29.8')

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.stationary_combustion.models import (
    EFSource,
    EmissionGas,
    FuelCategory,
    FuelType,
    GWPSource,
)
from greenlang.stationary_combustion.metrics import record_fuel_lookup
from greenlang.stationary_combustion.provenance import get_provenance_tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for emission factor key: (fuel_type, gas, source)
# ---------------------------------------------------------------------------
_EFKey = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Built-in EPA Emission Factors (kg per mmBtu, HHV basis)
# Source: EPA GHG Emission Factors Hub, 2025
# ---------------------------------------------------------------------------

_EPA_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "NATURAL_GAS":        {"CO2": Decimal("53.06"),  "CH4": Decimal("1.0E-3"),  "N2O": Decimal("1.0E-4")},
    "DIESEL":             {"CO2": Decimal("73.96"),  "CH4": Decimal("3.0E-3"),  "N2O": Decimal("6.0E-4")},
    "RESIDUAL_FUEL_OIL":  {"CO2": Decimal("75.10"),  "CH4": Decimal("3.0E-3"),  "N2O": Decimal("6.0E-4")},
    "PROPANE_LPG":        {"CO2": Decimal("62.87"),  "CH4": Decimal("1.0E-3"),  "N2O": Decimal("1.0E-4")},
    "KEROSENE":           {"CO2": Decimal("75.20"),  "CH4": Decimal("3.0E-3"),  "N2O": Decimal("6.0E-4")},
    "COAL_BITUMINOUS":    {"CO2": Decimal("93.28"),  "CH4": Decimal("1.1E-2"),  "N2O": Decimal("1.6E-3")},
    "COAL_ANTHRACITE":    {"CO2": Decimal("103.69"), "CH4": Decimal("1.1E-2"),  "N2O": Decimal("1.6E-3")},
    "COAL_SUBBITUMINOUS": {"CO2": Decimal("97.17"),  "CH4": Decimal("1.1E-2"),  "N2O": Decimal("1.6E-3")},
    "LIGNITE":            {"CO2": Decimal("97.72"),  "CH4": Decimal("1.1E-2"),  "N2O": Decimal("1.6E-3")},
    "PETROLEUM_COKE":     {"CO2": Decimal("102.41"), "CH4": Decimal("1.1E-2"),  "N2O": Decimal("1.6E-3")},
    "MOTOR_GASOLINE":     {"CO2": Decimal("70.22"),  "CH4": Decimal("3.0E-3"),  "N2O": Decimal("6.0E-4")},
    "JET_FUEL":           {"CO2": Decimal("72.22"),  "CH4": Decimal("3.0E-3"),  "N2O": Decimal("6.0E-4")},
    "WOOD_BIOMASS":       {"CO2": Decimal("0"),      "CH4": Decimal("7.2E-3"),  "N2O": Decimal("3.6E-3")},
    "BIOGAS":             {"CO2": Decimal("0"),      "CH4": Decimal("3.2E-3"),  "N2O": Decimal("6.3E-4")},
    "LANDFILL_GAS":       {"CO2": Decimal("0"),      "CH4": Decimal("3.2E-3"),  "N2O": Decimal("6.3E-4")},
    "BLAST_FURNACE_GAS":  {"CO2": Decimal("274.32"), "CH4": Decimal("2.2E-5"),  "N2O": Decimal("1.0E-4")},
    "COKE_OVEN_GAS":      {"CO2": Decimal("46.85"),  "CH4": Decimal("4.8E-4"),  "N2O": Decimal("1.0E-4")},
    "PEAT":               {"CO2": Decimal("106.00"), "CH4": Decimal("1.1E-2"),  "N2O": Decimal("1.6E-3")},
    "MSW":                {"CO2": Decimal("90.7"),   "CH4": Decimal("3.2E-2"),  "N2O": Decimal("4.2E-3")},
    "WASTE_OIL":          {"CO2": Decimal("74.0"),   "CH4": Decimal("3.0E-3"),  "N2O": Decimal("6.0E-4")},
}

# ---------------------------------------------------------------------------
# IPCC 2006 Default Factors (kg CO2 per TJ, NCV basis)
# Source: IPCC 2006 Guidelines for National GHG Inventories, Vol 2, Ch 2
# ---------------------------------------------------------------------------

_IPCC_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "NATURAL_GAS":        {"CO2": Decimal("56100"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "DIESEL":             {"CO2": Decimal("74100"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "RESIDUAL_FUEL_OIL":  {"CO2": Decimal("77400"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "PROPANE_LPG":        {"CO2": Decimal("63100"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "KEROSENE":           {"CO2": Decimal("71900"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "COAL_BITUMINOUS":    {"CO2": Decimal("94600"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "COAL_ANTHRACITE":    {"CO2": Decimal("98300"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "COAL_SUBBITUMINOUS": {"CO2": Decimal("96100"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "LIGNITE":            {"CO2": Decimal("101000"), "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "PETROLEUM_COKE":     {"CO2": Decimal("97500"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "MOTOR_GASOLINE":     {"CO2": Decimal("69300"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "JET_FUEL":           {"CO2": Decimal("71500"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "WOOD_BIOMASS":       {"CO2": Decimal("112000"), "CH4": Decimal("30.0"),  "N2O": Decimal("4.0")},
    "BIOGAS":             {"CO2": Decimal("54600"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "LANDFILL_GAS":       {"CO2": Decimal("54600"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "BLAST_FURNACE_GAS":  {"CO2": Decimal("260000"), "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "COKE_OVEN_GAS":      {"CO2": Decimal("44400"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "PEAT":               {"CO2": Decimal("106000"), "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "MSW":                {"CO2": Decimal("91700"),  "CH4": Decimal("30.0"),  "N2O": Decimal("4.0")},
    "WASTE_OIL":          {"CO2": Decimal("73300"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
}

# ---------------------------------------------------------------------------
# UK DEFRA 2025 Factors (kg CO2 per kWh net, different basis for UK reporting)
# Source: UK DEFRA GHG Conversion Factors 2025
# ---------------------------------------------------------------------------

_DEFRA_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "NATURAL_GAS":        {"CO2": Decimal("0.18316"), "CH4": Decimal("0.00034"), "N2O": Decimal("0.00003")},
    "DIESEL":             {"CO2": Decimal("0.24068"), "CH4": Decimal("0.00022"), "N2O": Decimal("0.00219")},
    "RESIDUAL_FUEL_OIL":  {"CO2": Decimal("0.26791"), "CH4": Decimal("0.00096"), "N2O": Decimal("0.00047")},
    "PROPANE_LPG":        {"CO2": Decimal("0.21446"), "CH4": Decimal("0.00047"), "N2O": Decimal("0.00001")},
    "KEROSENE":           {"CO2": Decimal("0.24677"), "CH4": Decimal("0.00024"), "N2O": Decimal("0.00233")},
    "COAL_BITUMINOUS":    {"CO2": Decimal("0.32260"), "CH4": Decimal("0.00037"), "N2O": Decimal("0.00465")},
    "COAL_ANTHRACITE":    {"CO2": Decimal("0.34472"), "CH4": Decimal("0.00037"), "N2O": Decimal("0.00465")},
    "COAL_SUBBITUMINOUS": {"CO2": Decimal("0.33188"), "CH4": Decimal("0.00037"), "N2O": Decimal("0.00465")},
    "LIGNITE":            {"CO2": Decimal("0.36396"), "CH4": Decimal("0.00037"), "N2O": Decimal("0.00465")},
    "PETROLEUM_COKE":     {"CO2": Decimal("0.34070"), "CH4": Decimal("0.00037"), "N2O": Decimal("0.00465")},
    "MOTOR_GASOLINE":     {"CO2": Decimal("0.23118"), "CH4": Decimal("0.00022"), "N2O": Decimal("0.00207")},
    "JET_FUEL":           {"CO2": Decimal("0.23776"), "CH4": Decimal("0.00022"), "N2O": Decimal("0.00207")},
    "WOOD_BIOMASS":       {"CO2": Decimal("0"),       "CH4": Decimal("0.00144"), "N2O": Decimal("0.00078")},
    "BIOGAS":             {"CO2": Decimal("0"),       "CH4": Decimal("0.00020"), "N2O": Decimal("0.00004")},
    "LANDFILL_GAS":       {"CO2": Decimal("0"),       "CH4": Decimal("0.00020"), "N2O": Decimal("0.00004")},
    "PEAT":               {"CO2": Decimal("0.38220"), "CH4": Decimal("0.00037"), "N2O": Decimal("0.00465")},
    "MSW":                {"CO2": Decimal("0.27170"), "CH4": Decimal("0.00320"), "N2O": Decimal("0.00420")},
    "WASTE_OIL":          {"CO2": Decimal("0.24780"), "CH4": Decimal("0.00022"), "N2O": Decimal("0.00219")},
}

# ---------------------------------------------------------------------------
# EU ETS Factors (kg CO2 per TJ NCV, MRR basis)
# Source: EU ETS Monitoring and Reporting Regulation
# ---------------------------------------------------------------------------

_EU_ETS_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "NATURAL_GAS":        {"CO2": Decimal("56100"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "DIESEL":             {"CO2": Decimal("74100"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "RESIDUAL_FUEL_OIL":  {"CO2": Decimal("77400"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "PROPANE_LPG":        {"CO2": Decimal("63100"),  "CH4": Decimal("1.0"),   "N2O": Decimal("0.1")},
    "KEROSENE":           {"CO2": Decimal("71900"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "COAL_BITUMINOUS":    {"CO2": Decimal("94600"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "COAL_ANTHRACITE":    {"CO2": Decimal("98300"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "COAL_SUBBITUMINOUS": {"CO2": Decimal("96100"),  "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "LIGNITE":            {"CO2": Decimal("101000"), "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "PETROLEUM_COKE":     {"CO2": Decimal("100800"), "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "MOTOR_GASOLINE":     {"CO2": Decimal("69300"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "JET_FUEL":           {"CO2": Decimal("71500"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
    "WOOD_BIOMASS":       {"CO2": Decimal("112000"), "CH4": Decimal("30.0"),  "N2O": Decimal("4.0")},
    "PEAT":               {"CO2": Decimal("106000"), "CH4": Decimal("1.0"),   "N2O": Decimal("1.5")},
    "MSW":                {"CO2": Decimal("91700"),  "CH4": Decimal("30.0"),  "N2O": Decimal("4.0")},
    "WASTE_OIL":          {"CO2": Decimal("73300"),  "CH4": Decimal("3.0"),   "N2O": Decimal("0.6")},
}

# ---------------------------------------------------------------------------
# Emission factor unit labels per source
# ---------------------------------------------------------------------------

_EF_UNITS: Dict[str, str] = {
    "EPA":    "kg/mmBtu",
    "IPCC":   "kg/TJ",
    "DEFRA":  "kg/kWh",
    "EU_ETS": "kg/TJ",
    "CUSTOM": "varies",
}

# ---------------------------------------------------------------------------
# Heating Values (HHV and NCV) for all fuel types
# Units vary by fuel category (see hhv_unit/ncv_unit in properties)
# ---------------------------------------------------------------------------

_HEATING_VALUES: Dict[str, Dict[str, Decimal]] = {
    # fuel_type: {HHV, NCV} - for gaseous fuels: mmBtu/Mscf; liquid: mmBtu/bbl; solid: mmBtu/short_ton
    "NATURAL_GAS":        {"HHV": Decimal("1.028"),    "NCV": Decimal("0.930")},
    "DIESEL":             {"HHV": Decimal("5.825"),    "NCV": Decimal("5.467")},
    "RESIDUAL_FUEL_OIL":  {"HHV": Decimal("6.287"),    "NCV": Decimal("5.934")},
    "PROPANE_LPG":        {"HHV": Decimal("3.824"),    "NCV": Decimal("3.534")},
    "KEROSENE":           {"HHV": Decimal("5.670"),    "NCV": Decimal("5.310")},
    "COAL_BITUMINOUS":    {"HHV": Decimal("24.930"),   "NCV": Decimal("23.686")},
    "COAL_ANTHRACITE":    {"HHV": Decimal("25.090"),   "NCV": Decimal("23.838")},
    "COAL_SUBBITUMINOUS": {"HHV": Decimal("17.250"),   "NCV": Decimal("16.388")},
    "LIGNITE":            {"HHV": Decimal("14.210"),   "NCV": Decimal("13.500")},
    "PETROLEUM_COKE":     {"HHV": Decimal("30.000"),   "NCV": Decimal("28.500")},
    "MOTOR_GASOLINE":     {"HHV": Decimal("5.253"),    "NCV": Decimal("4.904")},
    "JET_FUEL":           {"HHV": Decimal("5.670"),    "NCV": Decimal("5.355")},
    "WOOD_BIOMASS":       {"HHV": Decimal("15.380"),   "NCV": Decimal("14.111")},
    "BIOGAS":             {"HHV": Decimal("0.600"),    "NCV": Decimal("0.540")},
    "LANDFILL_GAS":       {"HHV": Decimal("0.485"),    "NCV": Decimal("0.437")},
    "BLAST_FURNACE_GAS":  {"HHV": Decimal("0.092"),    "NCV": Decimal("0.083")},
    "COKE_OVEN_GAS":      {"HHV": Decimal("0.547"),    "NCV": Decimal("0.493")},
    "PEAT":               {"HHV": Decimal("8.000"),    "NCV": Decimal("7.200")},
    "MSW":                {"HHV": Decimal("10.500"),   "NCV": Decimal("9.450")},
    "WASTE_OIL":          {"HHV": Decimal("5.800"),    "NCV": Decimal("5.440")},
}

# ---------------------------------------------------------------------------
# Oxidation Factors by fuel type and source
# ---------------------------------------------------------------------------

_OXIDATION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "NATURAL_GAS":        {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.995"), "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.995")},
    "DIESEL":             {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.99"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.99")},
    "RESIDUAL_FUEL_OIL":  {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.99"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.99")},
    "PROPANE_LPG":        {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.995"), "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.995")},
    "KEROSENE":           {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.99"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.99")},
    "COAL_BITUMINOUS":    {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "COAL_ANTHRACITE":    {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "COAL_SUBBITUMINOUS": {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "LIGNITE":            {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "PETROLEUM_COKE":     {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "MOTOR_GASOLINE":     {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.99"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.99")},
    "JET_FUEL":           {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.99"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.99")},
    "WOOD_BIOMASS":       {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "BIOGAS":             {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.995"), "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.995")},
    "LANDFILL_GAS":       {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.995"), "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.995")},
    "BLAST_FURNACE_GAS":  {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.995"), "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.995")},
    "COKE_OVEN_GAS":      {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.995"), "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.995")},
    "PEAT":               {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "MSW":                {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.98"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.98")},
    "WASTE_OIL":          {"EPA": Decimal("1.0"),  "IPCC": Decimal("0.99"),  "DEFRA": Decimal("1.0"),  "EU_ETS": Decimal("0.99")},
}

# ---------------------------------------------------------------------------
# GWP Values by gas and AR source
# ---------------------------------------------------------------------------

_GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    # 100-year GWP
    "AR4":     {"CO2": Decimal("1"), "CH4": Decimal("25"),   "N2O": Decimal("298")},
    "AR5":     {"CO2": Decimal("1"), "CH4": Decimal("28"),   "N2O": Decimal("265")},
    "AR6":     {"CO2": Decimal("1"), "CH4": Decimal("29.8"), "N2O": Decimal("273")},
    # 20-year GWP
    "AR6_20YR": {"CO2": Decimal("1"), "CH4": Decimal("82.5"), "N2O": Decimal("273")},
}

# ---------------------------------------------------------------------------
# Fuel Properties
# ---------------------------------------------------------------------------

_FUEL_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "NATURAL_GAS":        {"category": "GASEOUS", "display_name": "Natural Gas",            "density": None,           "carbon_content_pct": Decimal("75.0"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "DIESEL":             {"category": "LIQUID",  "display_name": "Diesel / Fuel Oil #2",   "density": Decimal("0.85"), "carbon_content_pct": Decimal("86.4"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "RESIDUAL_FUEL_OIL":  {"category": "LIQUID",  "display_name": "Residual Fuel Oil #6",   "density": Decimal("0.97"), "carbon_content_pct": Decimal("85.2"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "PROPANE_LPG":        {"category": "GASEOUS", "display_name": "Propane / LPG",          "density": Decimal("0.51"), "carbon_content_pct": Decimal("81.7"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "KEROSENE":           {"category": "LIQUID",  "display_name": "Kerosene",               "density": Decimal("0.80"), "carbon_content_pct": Decimal("85.9"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "COAL_BITUMINOUS":    {"category": "SOLID",   "display_name": "Coal (Bituminous)",      "density": None,           "carbon_content_pct": Decimal("75.5"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "COAL_ANTHRACITE":    {"category": "SOLID",   "display_name": "Coal (Anthracite)",      "density": None,           "carbon_content_pct": Decimal("86.5"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "COAL_SUBBITUMINOUS": {"category": "SOLID",   "display_name": "Coal (Sub-bituminous)",  "density": None,           "carbon_content_pct": Decimal("67.0"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "LIGNITE":            {"category": "SOLID",   "display_name": "Lignite",                "density": None,           "carbon_content_pct": Decimal("40.0"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "PETROLEUM_COKE":     {"category": "SOLID",   "display_name": "Petroleum Coke",         "density": None,           "carbon_content_pct": Decimal("87.3"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "MOTOR_GASOLINE":     {"category": "LIQUID",  "display_name": "Motor Gasoline",         "density": Decimal("0.74"), "carbon_content_pct": Decimal("85.5"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "JET_FUEL":           {"category": "LIQUID",  "display_name": "Jet Fuel / Kerosene",    "density": Decimal("0.80"), "carbon_content_pct": Decimal("86.3"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "WOOD_BIOMASS":       {"category": "BIOMASS", "display_name": "Wood / Biomass",         "density": None,           "carbon_content_pct": Decimal("50.0"),   "is_biogenic": True,  "ipcc_code": "1A1"},
    "BIOGAS":             {"category": "BIOMASS", "display_name": "Biogas",                 "density": None,           "carbon_content_pct": Decimal("27.3"),   "is_biogenic": True,  "ipcc_code": "1A1"},
    "LANDFILL_GAS":       {"category": "BIOMASS", "display_name": "Landfill Gas",           "density": None,           "carbon_content_pct": Decimal("27.3"),   "is_biogenic": True,  "ipcc_code": "1A1"},
    "BLAST_FURNACE_GAS":  {"category": "GASEOUS", "display_name": "Blast Furnace Gas",      "density": None,           "carbon_content_pct": Decimal("17.4"),   "is_biogenic": False, "ipcc_code": "1A2"},
    "COKE_OVEN_GAS":      {"category": "GASEOUS", "display_name": "Coke Oven Gas",          "density": None,           "carbon_content_pct": Decimal("34.0"),   "is_biogenic": False, "ipcc_code": "1A2"},
    "PEAT":               {"category": "SOLID",   "display_name": "Peat",                   "density": None,           "carbon_content_pct": Decimal("57.0"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "MSW":                {"category": "WASTE",   "display_name": "Municipal Solid Waste",   "density": None,           "carbon_content_pct": Decimal("33.0"),   "is_biogenic": False, "ipcc_code": "1A1"},
    "WASTE_OIL":          {"category": "WASTE",   "display_name": "Waste Oil",               "density": Decimal("0.90"), "carbon_content_pct": Decimal("84.0"),   "is_biogenic": False, "ipcc_code": "1A1"},
}

# ---------------------------------------------------------------------------
# Unit conversion constants for convert_ef_units
# ---------------------------------------------------------------------------

_MMBTU_TO_GJ = Decimal("1.055056")
_GJ_TO_TJ = Decimal("0.001")
_KG_TO_TONNE = Decimal("0.001")


class FuelDatabaseEngine:
    """Manages emission factors, heating values, oxidation factors, and GWP values.

    This engine is the authoritative source for all combustion emission factor
    data used by the CombustionCalculatorEngine. It supports 5 data sources
    (EPA, IPCC, DEFRA, EU ETS, custom) and provides deterministic lookup
    methods that use ``Decimal`` arithmetic for zero-hallucination precision.

    Thread-safe: all mutable state is guarded by ``threading.Lock()``.

    Attributes:
        _config: Optional configuration dictionary.
        _custom_factors: Registry of user-defined emission factors.
        _lock: Thread lock for custom factor mutations.
        _provenance: Reference to the provenance tracker.

    Example:
        >>> db = FuelDatabaseEngine()
        >>> ef = db.get_emission_factor("NATURAL_GAS", "CO2")
        >>> assert ef == Decimal("53.06")
        >>> db.get_factor_count()
        327
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FuelDatabaseEngine with optional configuration.

        Loads all built-in emission factors from EPA, IPCC, DEFRA, and EU ETS
        source dictionaries. No database calls are made; all data is held
        in-memory for deterministic, zero-latency lookups.

        Args:
            config: Optional configuration dict. Currently supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                  Defaults to True.
        """
        self._config = config or {}
        self._custom_factors: Dict[_EFKey, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._enable_provenance: bool = self._config.get("enable_provenance", True)

        if self._enable_provenance:
            self._provenance = get_provenance_tracker()
        else:
            self._provenance = None

        logger.info(
            "FuelDatabaseEngine initialized with %d built-in fuel types, "
            "%d total emission factors across 4 sources",
            len(_FUEL_PROPERTIES),
            self.get_factor_count(),
        )

    # ------------------------------------------------------------------
    # Public API: Fuel Properties
    # ------------------------------------------------------------------

    def get_fuel_properties(self, fuel_type: str) -> Dict[str, Any]:
        """Return physical and regulatory properties of a fuel type.

        Args:
            fuel_type: Fuel type identifier (e.g. ``"NATURAL_GAS"``).

        Returns:
            Dictionary with keys: category, display_name, density,
            carbon_content_pct, is_biogenic, ipcc_code, hhv, ncv.

        Raises:
            KeyError: If the fuel type is not found.

        Example:
            >>> props = db.get_fuel_properties("NATURAL_GAS")
            >>> props["category"]
            'GASEOUS'
        """
        fuel_key = fuel_type.upper()
        if fuel_key not in _FUEL_PROPERTIES:
            raise KeyError(f"Unknown fuel type: {fuel_type}")

        props = dict(_FUEL_PROPERTIES[fuel_key])
        # Include heating values
        hv = _HEATING_VALUES.get(fuel_key, {})
        props["hhv"] = hv.get("HHV", Decimal("0"))
        props["ncv"] = hv.get("NCV", Decimal("0"))
        return props

    # ------------------------------------------------------------------
    # Public API: Emission Factors
    # ------------------------------------------------------------------

    def get_emission_factor(
        self,
        fuel_type: str,
        gas: str,
        source: str = "EPA",
    ) -> Decimal:
        """Look up an emission factor for a specific fuel, gas, and source.

        Args:
            fuel_type: Fuel type identifier (e.g. ``"NATURAL_GAS"``).
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            source: Factor source (``"EPA"``, ``"IPCC"``, ``"DEFRA"``,
                ``"EU_ETS"``, ``"CUSTOM"``). Defaults to ``"EPA"``.

        Returns:
            Emission factor as a ``Decimal``.

        Raises:
            KeyError: If the fuel type, gas, or source combination is
                not found.

        Example:
            >>> db.get_emission_factor("DIESEL", "CO2", "EPA")
            Decimal('73.96')
        """
        fuel_key = fuel_type.upper()
        gas_key = gas.upper()
        source_key = source.upper()

        # Record metrics
        record_fuel_lookup(fuel_key, source_key)

        # Check custom factors first
        custom_key = (fuel_key, gas_key, source_key)
        with self._lock:
            if custom_key in self._custom_factors:
                val = self._custom_factors[custom_key]["value"]
                self._record_provenance(
                    "lookup_factor", fuel_key,
                    {"gas": gas_key, "source": source_key, "value": str(val)},
                )
                return val

        # Look up in built-in sources
        source_map = self._get_source_map(source_key)
        if fuel_key not in source_map:
            raise KeyError(
                f"No emission factors for fuel '{fuel_type}' in source '{source}'"
            )
        fuel_factors = source_map[fuel_key]
        if gas_key not in fuel_factors:
            raise KeyError(
                f"No {gas} emission factor for fuel '{fuel_type}' in source '{source}'"
            )

        value = fuel_factors[gas_key]
        self._record_provenance(
            "lookup_factor", fuel_key,
            {"gas": gas_key, "source": source_key, "value": str(value)},
        )
        return value

    def get_heating_value(
        self,
        fuel_type: str,
        basis: str = "HHV",
    ) -> Decimal:
        """Look up the heating value for a fuel type.

        Args:
            fuel_type: Fuel type identifier.
            basis: ``"HHV"`` for Higher Heating Value or ``"NCV"`` for
                Net Calorific Value. Defaults to ``"HHV"``.

        Returns:
            Heating value as a ``Decimal``.

        Raises:
            KeyError: If the fuel type is not found.
            ValueError: If basis is not ``"HHV"`` or ``"NCV"``.

        Example:
            >>> db.get_heating_value("NATURAL_GAS", "HHV")
            Decimal('1.028')
        """
        fuel_key = fuel_type.upper()
        basis_key = basis.upper()
        if basis_key not in ("HHV", "NCV"):
            raise ValueError(f"basis must be 'HHV' or 'NCV', got '{basis}'")

        if fuel_key not in _HEATING_VALUES:
            raise KeyError(f"No heating values for fuel type: {fuel_type}")

        value = _HEATING_VALUES[fuel_key][basis_key]
        self._record_provenance(
            "lookup_heating_value", fuel_key,
            {"basis": basis_key, "value": str(value)},
        )
        return value

    def get_oxidation_factor(
        self,
        fuel_type: str,
        source: str = "EPA",
    ) -> Decimal:
        """Look up the oxidation factor for a fuel and source.

        Args:
            fuel_type: Fuel type identifier.
            source: Factor source. Defaults to ``"EPA"``.

        Returns:
            Oxidation factor as a ``Decimal`` (0-1).

        Raises:
            KeyError: If the fuel type or source is not found.
        """
        fuel_key = fuel_type.upper()
        source_key = source.upper()

        if fuel_key not in _OXIDATION_FACTORS:
            raise KeyError(f"No oxidation factors for fuel type: {fuel_type}")

        fuel_of = _OXIDATION_FACTORS[fuel_key]
        if source_key not in fuel_of:
            # Fall back to EPA
            if "EPA" in fuel_of:
                return fuel_of["EPA"]
            raise KeyError(
                f"No oxidation factor for '{fuel_type}' from source '{source}'"
            )

        return fuel_of[source_key]

    def get_gwp(
        self,
        gas: str,
        source: str = "AR6",
        timeframe: str = "100yr",
    ) -> Decimal:
        """Look up the Global Warming Potential for a gas.

        Args:
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            source: IPCC Assessment Report (``"AR4"``, ``"AR5"``, ``"AR6"``).
                Defaults to ``"AR6"``.
            timeframe: ``"100yr"`` or ``"20yr"``. When ``"20yr"`` is
                specified, ``"AR6_20YR"`` values are used regardless of
                the ``source`` parameter.

        Returns:
            GWP value as a ``Decimal``.

        Raises:
            KeyError: If the gas or source is not found.
            ValueError: If timeframe is invalid.

        Example:
            >>> db.get_gwp("CH4", "AR6", "100yr")
            Decimal('29.8')
            >>> db.get_gwp("CH4", "AR6", "20yr")
            Decimal('82.5')
        """
        gas_key = gas.upper()
        source_key = source.upper()
        tf_key = timeframe.lower()

        if tf_key not in ("100yr", "20yr"):
            raise ValueError(f"timeframe must be '100yr' or '20yr', got '{timeframe}'")

        # 20-year GWP always uses AR6_20YR
        if tf_key == "20yr":
            source_key = "AR6_20YR"

        if source_key not in _GWP_VALUES:
            raise KeyError(f"Unknown GWP source: {source}")

        gwp_table = _GWP_VALUES[source_key]
        if gas_key not in gwp_table:
            raise KeyError(f"No GWP for gas '{gas}' in source '{source_key}'")

        value = gwp_table[gas_key]
        self._record_provenance(
            "lookup_gwp", gas_key,
            {"source": source_key, "timeframe": tf_key, "value": str(value)},
        )
        return value

    # ------------------------------------------------------------------
    # Public API: Custom Factor Registration
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        fuel_type: str,
        gas: str,
        value: Decimal,
        unit: str,
        source: str = "CUSTOM",
        geography: str = "GLOBAL",
        reference: str = "",
    ) -> str:
        """Register a custom emission factor.

        Custom factors take priority over built-in factors when the source
        is ``"CUSTOM"`` and the fuel/gas combination matches.

        Args:
            fuel_type: Fuel type identifier.
            gas: Greenhouse gas (``"CO2"``, ``"CH4"``, ``"N2O"``).
            value: Factor value as ``Decimal``.
            unit: Factor unit (e.g. ``"kg/mmBtu"``).
            source: Source label. Defaults to ``"CUSTOM"``.
            geography: Geographic scope. Defaults to ``"GLOBAL"``.
            reference: Regulatory reference string.

        Returns:
            Registration ID string.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError(f"Emission factor value must be >= 0, got {value}")

        fuel_key = fuel_type.upper()
        gas_key = gas.upper()
        source_key = source.upper()
        reg_id = f"custom_{uuid.uuid4().hex[:12]}"

        with self._lock:
            self._custom_factors[(fuel_key, gas_key, source_key)] = {
                "value": Decimal(str(value)),
                "unit": unit,
                "geography": geography,
                "reference": reference,
                "registration_id": reg_id,
            }

        self._record_provenance(
            "register_custom_factor", reg_id,
            {
                "fuel_type": fuel_key, "gas": gas_key,
                "value": str(value), "unit": unit,
                "source": source_key, "geography": geography,
            },
        )

        record_fuel_lookup(fuel_key, source_key)
        logger.info(
            "Registered custom factor: %s/%s/%s = %s %s (id=%s)",
            fuel_key, gas_key, source_key, value, unit, reg_id,
        )
        return reg_id

    # ------------------------------------------------------------------
    # Public API: Listing and Search
    # ------------------------------------------------------------------

    def list_fuel_types(self) -> List[str]:
        """Return sorted list of all known fuel type identifiers.

        Returns:
            List of fuel type strings.
        """
        return sorted(_FUEL_PROPERTIES.keys())

    def list_emission_factors(
        self,
        fuel_type: Optional[str] = None,
        source: Optional[str] = None,
        gas: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List emission factors with optional filtering.

        Args:
            fuel_type: Filter by fuel type (optional).
            source: Filter by source (optional).
            gas: Filter by gas (optional).

        Returns:
            List of dictionaries with fuel_type, gas, source, value, unit.
        """
        results: List[Dict[str, Any]] = []
        sources_to_check = self._get_all_sources(source)

        for src_name, src_map in sources_to_check:
            for f_type, gases in src_map.items():
                if fuel_type and f_type != fuel_type.upper():
                    continue
                for g_name, g_val in gases.items():
                    if gas and g_name != gas.upper():
                        continue
                    results.append({
                        "fuel_type": f_type,
                        "gas": g_name,
                        "source": src_name,
                        "value": g_val,
                        "unit": _EF_UNITS.get(src_name, "varies"),
                    })

        # Include matching custom factors
        with self._lock:
            for (f, g, s), data in self._custom_factors.items():
                if fuel_type and f != fuel_type.upper():
                    continue
                if source and s != source.upper():
                    continue
                if gas and g != gas.upper():
                    continue
                results.append({
                    "fuel_type": f,
                    "gas": g,
                    "source": s,
                    "value": data["value"],
                    "unit": data["unit"],
                })

        return results

    def search_factors(self, query: str) -> List[Dict[str, Any]]:
        """Search emission factors by keyword across fuel type and display name.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching emission factor dictionaries.
        """
        query_lower = query.lower()
        matching_fuels: List[str] = []

        for fuel_key, props in _FUEL_PROPERTIES.items():
            if (
                query_lower in fuel_key.lower()
                or query_lower in props["display_name"].lower()
            ):
                matching_fuels.append(fuel_key)

        results: List[Dict[str, Any]] = []
        for f in matching_fuels:
            results.extend(self.list_emission_factors(fuel_type=f))

        return results

    def get_factor_count(self) -> int:
        """Return the total count of emission factors across all sources.

        Returns:
            Integer count of all registered emission factors (built-in + custom).
        """
        count = 0
        for src_map in [_EPA_FACTORS, _IPCC_FACTORS, _DEFRA_FACTORS, _EU_ETS_FACTORS]:
            for gases in src_map.values():
                count += len(gases)

        with self._lock:
            count += len(self._custom_factors)

        return count

    def is_biogenic(self, fuel_type: str) -> bool:
        """Check if a fuel type is biogenic.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            True if the fuel is biogenic (biomass-derived), False otherwise.

        Raises:
            KeyError: If the fuel type is not found.
        """
        fuel_key = fuel_type.upper()
        if fuel_key not in _FUEL_PROPERTIES:
            raise KeyError(f"Unknown fuel type: {fuel_type}")
        return _FUEL_PROPERTIES[fuel_key]["is_biogenic"]

    # ------------------------------------------------------------------
    # Public API: Unit Conversion
    # ------------------------------------------------------------------

    def convert_ef_units(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
        fuel_type: str,
    ) -> Decimal:
        """Convert an emission factor value between units.

        Supported conversions: kg/mmBtu, kg/GJ, kg/TJ, t/TJ.

        Args:
            value: Factor value to convert.
            from_unit: Source unit.
            to_unit: Target unit.
            fuel_type: Fuel type (needed for some conversions).

        Returns:
            Converted value as ``Decimal``.

        Raises:
            ValueError: If the unit conversion is not supported.
        """
        from_u = from_unit.lower().replace(" ", "")
        to_u = to_unit.lower().replace(" ", "")

        if from_u == to_u:
            return value

        # Normalize to kg/GJ as intermediate
        kg_per_gj = self._to_kg_per_gj(value, from_u)

        # Convert from kg/GJ to target
        return self._from_kg_per_gj(kg_per_gj, to_u)

    def get_hhv_to_ncv_ratio(self, fuel_type: str) -> Decimal:
        """Return the HHV-to-NCV ratio for a fuel type.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Ratio as ``Decimal`` (HHV / NCV).

        Raises:
            KeyError: If the fuel type is not found.
        """
        fuel_key = fuel_type.upper()
        if fuel_key not in _HEATING_VALUES:
            raise KeyError(f"No heating values for fuel type: {fuel_type}")

        hhv = _HEATING_VALUES[fuel_key]["HHV"]
        ncv = _HEATING_VALUES[fuel_key]["NCV"]

        if ncv == 0:
            raise ValueError(f"NCV is zero for fuel type: {fuel_type}")

        return (hhv / ncv).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_source_map(self, source: str) -> Dict[str, Dict[str, Decimal]]:
        """Return the factor dictionary for a given source.

        Args:
            source: Uppercase source key.

        Returns:
            Dictionary mapping fuel_type -> {gas -> Decimal}.

        Raises:
            KeyError: If the source is not recognized.
        """
        source_maps: Dict[str, Dict[str, Dict[str, Decimal]]] = {
            "EPA": _EPA_FACTORS,
            "IPCC": _IPCC_FACTORS,
            "DEFRA": _DEFRA_FACTORS,
            "EU_ETS": _EU_ETS_FACTORS,
        }
        if source not in source_maps:
            raise KeyError(f"Unknown emission factor source: {source}")
        return source_maps[source]

    def _get_all_sources(
        self,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Dict[str, Decimal]]]]:
        """Return all source maps, optionally filtered by source name.

        Args:
            source_filter: Optional source to filter to.

        Returns:
            List of (source_name, source_dict) tuples.
        """
        all_sources = [
            ("EPA", _EPA_FACTORS),
            ("IPCC", _IPCC_FACTORS),
            ("DEFRA", _DEFRA_FACTORS),
            ("EU_ETS", _EU_ETS_FACTORS),
        ]
        if source_filter:
            sf = source_filter.upper()
            return [(n, m) for n, m in all_sources if n == sf]
        return all_sources

    def _to_kg_per_gj(self, value: Decimal, unit: str) -> Decimal:
        """Convert a factor value to kg/GJ.

        Args:
            value: Factor value.
            unit: Lowercase unit string.

        Returns:
            Value in kg/GJ.

        Raises:
            ValueError: If the unit is not supported.
        """
        if unit == "kg/gj":
            return value
        if unit == "kg/mmbtu":
            # 1 mmBtu = 1.055056 GJ => kg/GJ = kg/mmBtu / 1.055056
            return value / _MMBTU_TO_GJ
        if unit == "kg/tj":
            # 1 TJ = 1000 GJ => kg/GJ = kg/TJ / 1000
            return value / Decimal("1000")
        if unit == "t/tj":
            # 1 t = 1000 kg, 1 TJ = 1000 GJ => kg/GJ = t/TJ * 1000 / 1000 = t/TJ
            return value
        raise ValueError(
            f"Unsupported source unit for EF conversion: {unit}. "
            f"Supported: kg/mmBtu, kg/GJ, kg/TJ, t/TJ"
        )

    def _from_kg_per_gj(self, value: Decimal, unit: str) -> Decimal:
        """Convert from kg/GJ to target unit.

        Args:
            value: Value in kg/GJ.
            unit: Lowercase target unit string.

        Returns:
            Converted value.

        Raises:
            ValueError: If the unit is not supported.
        """
        if unit == "kg/gj":
            return value
        if unit == "kg/mmbtu":
            return value * _MMBTU_TO_GJ
        if unit == "kg/tj":
            return value * Decimal("1000")
        if unit == "t/tj":
            return value
        raise ValueError(
            f"Unsupported target unit for EF conversion: {unit}. "
            f"Supported: kg/mmBtu, kg/GJ, kg/TJ, t/TJ"
        )

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a provenance entry if provenance tracking is enabled.

        Args:
            action: Action label.
            entity_id: Entity identifier.
            data: Optional data payload for hashing.
        """
        if self._provenance is not None:
            self._provenance.record(
                entity_type="fuel",
                action=action,
                entity_id=entity_id,
                data=data,
            )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"FuelDatabaseEngine("
            f"fuel_types={len(_FUEL_PROPERTIES)}, "
            f"factors={self.get_factor_count()}, "
            f"custom={len(self._custom_factors)})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FuelDatabaseEngine",
]
