# -*- coding: utf-8 -*-
"""
greenlang/data/unit_normalizer.py

Unit Normalization Module - Consistent Case-Insensitive Unit Handling

This module provides deterministic unit normalization to fix the kWh vs kwh
case sensitivity bug and ensure consistent unit handling across GreenLang.

ZERO-HALLUCINATION GUARANTEE:
- All normalizations are deterministic (same input -> same output)
- Uses canonical forms from authoritative sources (SI, EPA, IPCC)
- No LLM involvement in normalization
- Full mapping table with provenance

Sources:
- SI (International System of Units)
- EPA Emission Factor Hub
- IPCC 2006 Guidelines
- IEA Energy Statistics Manual

Author: GreenLang Team
Date: 2025-11-25
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal


class UnitNormalizer:
    """
    Unit normalizer for consistent case handling across GreenLang.

    ZERO-HALLUCINATION GUARANTEE:
    - All normalizations use fixed mapping tables
    - Same input always produces identical output
    - Unknown units are passed through (no guessing)
    - Full provenance for all canonical forms
    """

    # Canonical unit forms with all variations mapped
    # Key: lowercase variation -> Value: canonical form
    # Canonical forms preserve proper capitalization (e.g., kWh, not kwh)
    CANONICAL_UNITS: Dict[str, str] = {
        # Energy units (canonical: kWh, MWh, GWh, MJ, GJ, MMBtu, therm)
        "kwh": "kWh",
        "kilowatt-hour": "kWh",
        "kilowatt_hour": "kWh",
        "kilowatthour": "kWh",
        "kw-h": "kWh",
        "kw_h": "kWh",
        "mwh": "MWh",
        "megawatt-hour": "MWh",
        "megawatt_hour": "MWh",
        "megawatthour": "MWh",
        "mw-h": "MWh",
        "gwh": "GWh",
        "gigawatt-hour": "GWh",
        "gigawatt_hour": "GWh",
        "twh": "TWh",
        "terawatt-hour": "TWh",
        "mj": "MJ",
        "megajoule": "MJ",
        "megajoules": "MJ",
        "gj": "GJ",
        "gigajoule": "GJ",
        "gigajoules": "GJ",
        "tj": "TJ",
        "terajoule": "TJ",
        "terajoules": "TJ",
        "btu": "Btu",
        "british_thermal_unit": "Btu",
        "british thermal unit": "Btu",
        "mmbtu": "MMBtu",
        "million_btu": "MMBtu",
        "million btu": "MMBtu",
        "therm": "therm",
        "therms": "therm",
        "us_therm": "therm",
        "uk_therm": "therm_UK",

        # Volume units (canonical: L, gal, m3, cf)
        "l": "L",
        "liter": "L",
        "liters": "L",
        "litre": "L",
        "litres": "L",
        "ml": "mL",
        "milliliter": "mL",
        "milliliters": "mL",
        "gal": "gal",
        "gallon": "gal",
        "gallons": "gal",
        "us_gallon": "gal",
        "us gallon": "gal",
        "uk_gallon": "gal_UK",
        "imperial_gallon": "gal_UK",
        "m3": "m3",
        "m^3": "m3",
        "cubic_meter": "m3",
        "cubic meter": "m3",
        "cubic_meters": "m3",
        "cubic_metre": "m3",
        "cbm": "m3",
        "cf": "cf",
        "ft3": "cf",
        "cubic_foot": "cf",
        "cubic_feet": "cf",
        "ccf": "ccf",  # 100 cubic feet
        "mcf": "Mcf",  # 1000 cubic feet
        "scf": "scf",  # Standard cubic feet
        "nm3": "Nm3",  # Normal cubic meter
        "bbl": "bbl",  # Barrel (oil)
        "barrel": "bbl",
        "barrels": "bbl",

        # Mass units (canonical: kg, t, lb)
        "kg": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "kilo": "kg",
        "g": "g",
        "gram": "g",
        "grams": "g",
        "mg": "mg",
        "milligram": "mg",
        "t": "t",
        "tonne": "t",
        "tonnes": "t",
        "metric_ton": "t",
        "metric ton": "t",
        "metric_tonne": "t",
        "mt": "t",
        "ton": "ton_US",  # US short ton (2000 lb)
        "tons": "ton_US",
        "short_ton": "ton_US",
        "short ton": "ton_US",
        "us_ton": "ton_US",
        "long_ton": "ton_UK",  # UK long ton (2240 lb)
        "lb": "lb",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
        "oz": "oz",
        "ounce": "oz",
        "ounces": "oz",

        # Distance units (canonical: km, mi, m)
        "km": "km",
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",
        "m": "m",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "mi": "mi",
        "mile": "mi",
        "miles": "mi",
        "nmi": "nmi",  # Nautical mile
        "nautical_mile": "nmi",
        "nautical mile": "nmi",
        "ft": "ft",
        "foot": "ft",
        "feet": "ft",

        # Area units (canonical: m2, ha, acre)
        "m2": "m2",
        "m^2": "m2",
        "sq_m": "m2",
        "sqm": "m2",
        "square_meter": "m2",
        "square_metre": "m2",
        "ha": "ha",
        "hectare": "ha",
        "hectares": "ha",
        "acre": "acre",
        "acres": "acre",
        "km2": "km2",
        "km^2": "km2",
        "sq_km": "km2",
        "ft2": "ft2",
        "ft^2": "ft2",
        "sq_ft": "ft2",
        "sqft": "ft2",
        "square_foot": "ft2",
        "square_feet": "ft2",

        # Time units (canonical: h, d, yr)
        "h": "h",
        "hr": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
        "wk": "wk",
        "week": "wk",
        "weeks": "wk",
        "mo": "mo",
        "month": "mo",
        "months": "mo",
        "yr": "yr",
        "year": "yr",
        "years": "yr",
        "a": "yr",  # "a" for annum (year) in scientific notation

        # Emission units (canonical: kgCO2e, tCO2e)
        "kgco2e": "kgCO2e",
        "kg_co2e": "kgCO2e",
        "kg co2e": "kgCO2e",
        "kg-co2e": "kgCO2e",
        "kgco2eq": "kgCO2e",
        "kg_co2_eq": "kgCO2e",
        "kilogram_co2e": "kgCO2e",
        "tco2e": "tCO2e",
        "t_co2e": "tCO2e",
        "t co2e": "tCO2e",
        "t-co2e": "tCO2e",
        "tco2eq": "tCO2e",
        "tonne_co2e": "tCO2e",
        "tonnes_co2e": "tCO2e",
        "mtco2e": "MtCO2e",  # Million tonnes CO2e
        "million_tonnes_co2e": "MtCO2e",
        "gtco2e": "GtCO2e",  # Gigatonnes CO2e

        # Currency units (canonical: USD, EUR, GBP)
        "usd": "USD",
        "$": "USD",
        "dollar": "USD",
        "dollars": "USD",
        "eur": "EUR",
        "euro": "EUR",
        "euros": "EUR",
        "gbp": "GBP",
        "pound_sterling": "GBP",
    }

    # Unit aliases for common variations (bidirectional lookup)
    # Maps from canonical to list of acceptable variations
    UNIT_ALIASES: Dict[str, List[str]] = {
        "kWh": ["kwh", "kilowatt-hour", "kilowatt_hour", "kw-h"],
        "MWh": ["mwh", "megawatt-hour", "megawatt_hour"],
        "GWh": ["gwh", "gigawatt-hour"],
        "L": ["l", "liter", "liters", "litre", "litres"],
        "gal": ["gallon", "gallons", "us_gallon"],
        "kg": ["kilogram", "kilograms", "kilo"],
        "t": ["tonne", "tonnes", "metric_ton", "metric ton", "mt"],
        "mi": ["mile", "miles"],
        "km": ["kilometer", "kilometers", "kilometre"],
        "m2": ["sqm", "square_meter", "square_metre"],
        "kgCO2e": ["kgco2e", "kg_co2e", "kg co2e", "kgco2eq"],
        "tCO2e": ["tco2e", "t_co2e", "t co2e", "tco2eq"],
    }

    @classmethod
    def normalize(cls, unit: str) -> str:
        """
        Normalize unit to canonical form.

        ZERO-HALLUCINATION GUARANTEE:
        - Uses fixed mapping table
        - Same input always produces identical output
        - Unknown units are passed through unchanged

        Args:
            unit: Unit string in any case or format

        Returns:
            Canonical unit form (e.g., "kWh" for "kwh", "KWH", "kilowatt-hour")
            If unknown, returns stripped input

        Example:
            >>> UnitNormalizer.normalize("kwh")
            'kWh'
            >>> UnitNormalizer.normalize("KWH")
            'kWh'
            >>> UnitNormalizer.normalize("kilowatt-hour")
            'kWh'
            >>> UnitNormalizer.normalize("unknown_unit")
            'unknown_unit'
        """
        if not unit:
            return unit

        # Clean and lowercase for lookup
        clean = unit.strip().lower().replace(" ", "_").replace("-", "_")

        # Look up canonical form
        return cls.CANONICAL_UNITS.get(clean, unit.strip())

    @classmethod
    def normalize_for_storage(cls, unit: str) -> str:
        """
        Normalize unit for database storage (lowercase canonical).

        Use this for database keys and lookups to ensure consistent matching.

        Args:
            unit: Unit string

        Returns:
            Lowercase normalized unit suitable for storage/lookup

        Example:
            >>> UnitNormalizer.normalize_for_storage("kWh")
            'kwh'
            >>> UnitNormalizer.normalize_for_storage("KWH")
            'kwh'
        """
        canonical = cls.normalize(unit)
        return canonical.lower()

    @classmethod
    def normalize_for_display(cls, unit: str) -> str:
        """
        Normalize unit for user display (proper capitalization).

        Use this for user-facing output and reports.

        Args:
            unit: Unit string

        Returns:
            Properly capitalized canonical form

        Example:
            >>> UnitNormalizer.normalize_for_display("kwh")
            'kWh'
            >>> UnitNormalizer.normalize_for_display("tco2e")
            'tCO2e'
        """
        return cls.normalize(unit)

    @classmethod
    def are_equivalent(cls, unit1: str, unit2: str) -> bool:
        """
        Check if two unit strings represent the same unit.

        Performs case-insensitive comparison after normalization.

        Args:
            unit1: First unit string
            unit2: Second unit string

        Returns:
            True if units are equivalent, False otherwise

        Example:
            >>> UnitNormalizer.are_equivalent("kWh", "kwh")
            True
            >>> UnitNormalizer.are_equivalent("kWh", "MWh")
            False
            >>> UnitNormalizer.are_equivalent("kilowatt-hour", "kWh")
            True
        """
        return cls.normalize(unit1).lower() == cls.normalize(unit2).lower()

    @classmethod
    def get_canonical_form(cls, unit: str) -> Tuple[str, bool]:
        """
        Get canonical form with known/unknown indicator.

        Args:
            unit: Unit string

        Returns:
            Tuple of (canonical_form, is_known)

        Example:
            >>> UnitNormalizer.get_canonical_form("kwh")
            ('kWh', True)
            >>> UnitNormalizer.get_canonical_form("unknown")
            ('unknown', False)
        """
        clean = unit.strip().lower().replace(" ", "_").replace("-", "_")
        is_known = clean in cls.CANONICAL_UNITS
        canonical = cls.CANONICAL_UNITS.get(clean, unit.strip())
        return canonical, is_known

    @classmethod
    def list_known_units(cls, category: Optional[str] = None) -> List[str]:
        """
        List all known canonical units.

        Args:
            category: Optional filter by category ('energy', 'volume', 'mass', etc.)

        Returns:
            List of canonical unit strings
        """
        # Get unique canonical forms
        canonicals = sorted(set(cls.CANONICAL_UNITS.values()))

        if category is None:
            return canonicals

        # Filter by category
        category_prefixes = {
            "energy": ["kWh", "MWh", "GWh", "TWh", "MJ", "GJ", "TJ", "Btu", "MMBtu", "therm"],
            "volume": ["L", "mL", "gal", "m3", "cf", "ccf", "Mcf", "scf", "Nm3", "bbl"],
            "mass": ["kg", "g", "mg", "t", "ton_US", "ton_UK", "lb", "oz"],
            "distance": ["km", "m", "mi", "nmi", "ft"],
            "area": ["m2", "ha", "acre", "km2", "ft2"],
            "time": ["h", "d", "wk", "mo", "yr"],
            "emissions": ["kgCO2e", "tCO2e", "MtCO2e", "GtCO2e"],
            "currency": ["USD", "EUR", "GBP"],
        }

        if category.lower() in category_prefixes:
            allowed = category_prefixes[category.lower()]
            return [u for u in canonicals if u in allowed]

        return canonicals


# Convenience functions for direct use
def normalize_unit(unit: str) -> str:
    """Normalize unit to canonical form. See UnitNormalizer.normalize for details."""
    return UnitNormalizer.normalize(unit)


def units_are_equivalent(unit1: str, unit2: str) -> bool:
    """Check if units are equivalent. See UnitNormalizer.are_equivalent for details."""
    return UnitNormalizer.are_equivalent(unit1, unit2)


def normalize_unit_for_storage(unit: str) -> str:
    """Normalize unit for storage. See UnitNormalizer.normalize_for_storage for details."""
    return UnitNormalizer.normalize_for_storage(unit)


__all__ = [
    'UnitNormalizer',
    'normalize_unit',
    'units_are_equivalent',
    'normalize_unit_for_storage',
]
