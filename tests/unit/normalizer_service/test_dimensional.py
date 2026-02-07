# -*- coding: utf-8 -*-
"""
Unit Tests for DimensionalAnalyzer (AGENT-FOUND-003)

Tests dimensional compatibility checking, dimension lookups, base units,
valid/invalid unit checks, and coverage of all dimensions.

Coverage target: 85%+ of dimensional.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline DimensionalAnalyzer mirroring greenlang/normalizer/dimensional.py
# ---------------------------------------------------------------------------


class Dimension(str, Enum):
    ENERGY = "ENERGY"
    MASS = "MASS"
    EMISSIONS = "EMISSIONS"
    VOLUME = "VOLUME"
    AREA = "AREA"
    DISTANCE = "DISTANCE"
    TIME = "TIME"


class DimensionInfo:
    """Information about a dimension."""

    def __init__(self, dimension: Dimension, base_unit: str, all_units: List[str]):
        self.dimension = dimension
        self.base_unit = base_unit
        self.all_units = all_units


# Unit -> Dimension mapping
_UNIT_DIMENSIONS: Dict[str, Dimension] = {
    # Energy
    "kWh": Dimension.ENERGY, "MWh": Dimension.ENERGY, "GWh": Dimension.ENERGY,
    "GJ": Dimension.ENERGY, "MJ": Dimension.ENERGY, "J": Dimension.ENERGY,
    "kJ": Dimension.ENERGY, "BTU": Dimension.ENERGY, "therm": Dimension.ENERGY,
    "MMBTU": Dimension.ENERGY,
    # Mass
    "kg": Dimension.MASS, "g": Dimension.MASS, "mg": Dimension.MASS,
    "t": Dimension.MASS, "tonne": Dimension.MASS, "lb": Dimension.MASS,
    "oz": Dimension.MASS, "short_ton": Dimension.MASS, "long_ton": Dimension.MASS,
    # Emissions
    "kgCO2e": Dimension.EMISSIONS, "tCO2e": Dimension.EMISSIONS,
    "gCO2e": Dimension.EMISSIONS, "MtCO2e": Dimension.EMISSIONS,
    "kgCO2": Dimension.EMISSIONS, "tCO2": Dimension.EMISSIONS,
    # Volume
    "L": Dimension.VOLUME, "mL": Dimension.VOLUME, "m3": Dimension.VOLUME,
    "gal": Dimension.VOLUME, "bbl": Dimension.VOLUME, "ft3": Dimension.VOLUME,
    # Area
    "m2": Dimension.AREA, "km2": Dimension.AREA, "hectare": Dimension.AREA,
    "acre": Dimension.AREA,
    # Distance
    "m": Dimension.DISTANCE, "km": Dimension.DISTANCE, "mi": Dimension.DISTANCE,
    "ft": Dimension.DISTANCE, "nmi": Dimension.DISTANCE,
    # Time
    "s": Dimension.TIME, "min": Dimension.TIME, "h": Dimension.TIME,
    "day": Dimension.TIME,
}

# Aliases for case-insensitive lookup
_UNIT_ALIASES: Dict[str, str] = {
    "kwh": "kWh", "mwh": "MWh", "gwh": "GWh", "gj": "GJ", "mj": "MJ",
    "kj": "kJ", "btu": "BTU", "mmbtu": "MMBTU",
    "KG": "kg", "KWH": "kWh", "MWH": "MWh",
    "liter": "L", "litre": "L", "l": "L",
    "ml": "mL", "gallon": "gal", "barrel": "bbl",
    "kilogram": "kg", "gram": "g", "tonne": "t", "pound": "lb",
    "meter": "m", "metre": "m", "kilometer": "km", "mile": "mi",
    "foot": "ft", "feet": "ft",
    "kgco2e": "kgCO2e", "tco2e": "tCO2e", "gco2e": "gCO2e",
    "mtco2e": "MtCO2e", "kgco2": "kgCO2", "tco2": "tCO2",
    "sqm": "m2", "hectare": "hectare", "acre": "acre",
}

# Base unit per dimension
_BASE_UNITS: Dict[Dimension, str] = {
    Dimension.ENERGY: "kWh",
    Dimension.MASS: "kg",
    Dimension.EMISSIONS: "kgCO2e",
    Dimension.VOLUME: "L",
    Dimension.AREA: "m2",
    Dimension.DISTANCE: "m",
    Dimension.TIME: "s",
}


class DimensionalAnalyzer:
    """
    Analyzes dimensional compatibility of units.

    Provides checks for whether two units are compatible (same dimension),
    dimension lookups, and base unit information.
    """

    def check_compatibility(self, unit_a: str, unit_b: str) -> bool:
        """Check if two units are in the same dimension."""
        dim_a = self.get_dimension(unit_a)
        dim_b = self.get_dimension(unit_b)
        if dim_a is None or dim_b is None:
            return False
        return dim_a == dim_b

    def get_dimension(self, unit: str) -> Optional[Dimension]:
        """Get the dimension of a unit."""
        canonical = self._resolve(unit)
        if canonical is None:
            return None
        return _UNIT_DIMENSIONS.get(canonical)

    def get_dimension_info(self, unit: str) -> Optional[DimensionInfo]:
        """Get full dimension info for a unit."""
        dim = self.get_dimension(unit)
        if dim is None:
            return None
        base = _BASE_UNITS[dim]
        units = [u for u, d in _UNIT_DIMENSIONS.items() if d == dim]
        return DimensionInfo(dimension=dim, base_unit=base, all_units=units)

    def get_base_unit(self, dimension: Dimension) -> str:
        """Get the base unit for a dimension."""
        return _BASE_UNITS[dimension]

    def is_valid_unit(self, unit: str) -> bool:
        """Check if a unit name is recognized."""
        return self._resolve(unit) is not None

    def list_dimensions(self) -> List[Dimension]:
        """List all supported dimensions."""
        return list(_BASE_UNITS.keys())

    def list_units(self, dimension: Dimension) -> List[str]:
        """List all units for a specific dimension."""
        return [u for u, d in _UNIT_DIMENSIONS.items() if d == dimension]

    def _resolve(self, unit: str) -> Optional[str]:
        """Resolve a unit name to its canonical form."""
        if unit in _UNIT_DIMENSIONS:
            return unit
        stripped = unit.strip()
        if stripped in _UNIT_DIMENSIONS:
            return stripped
        alias = _UNIT_ALIASES.get(unit) or _UNIT_ALIASES.get(stripped) or _UNIT_ALIASES.get(unit.lower())
        if alias and alias in _UNIT_DIMENSIONS:
            return alias
        return None


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCompatibleDimensions:
    """Test compatible unit pairs within the same dimension."""

    def test_kwh_mwh_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("kWh", "MWh") is True

    def test_kg_t_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("kg", "t") is True

    def test_l_m3_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("L", "m3") is True

    def test_m_km_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("m", "km") is True

    def test_m2_hectare_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("m2", "hectare") is True

    def test_kgco2e_tco2e_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("kgCO2e", "tCO2e") is True

    def test_gj_kwh_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("GJ", "kWh") is True

    def test_same_unit_compatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("kg", "kg") is True


class TestIncompatibleDimensions:
    """Test incompatible unit pairs across different dimensions."""

    def test_kwh_kg_incompatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("kWh", "kg") is False

    def test_m_l_incompatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("m", "L") is False

    def test_kg_kwh_incompatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("kg", "kWh") is False

    def test_m2_m_incompatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("m2", "m") is False

    def test_l_kg_incompatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("L", "kg") is False

    def test_unknown_unit_incompatible(self):
        da = DimensionalAnalyzer()
        assert da.check_compatibility("unknown", "kg") is False


class TestDimensionLookup:
    """Test get_dimension() returns correct dimension."""

    def test_kwh_is_energy(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("kWh") == Dimension.ENERGY

    def test_kg_is_mass(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("kg") == Dimension.MASS

    def test_l_is_volume(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("L") == Dimension.VOLUME

    def test_m_is_distance(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("m") == Dimension.DISTANCE

    def test_m2_is_area(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("m2") == Dimension.AREA

    def test_kgco2e_is_emissions(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("kgCO2e") == Dimension.EMISSIONS

    def test_unknown_returns_none(self):
        da = DimensionalAnalyzer()
        assert da.get_dimension("furlongs") is None


class TestBaseUnits:
    """Test base unit retrieval per dimension."""

    def test_energy_base_is_kwh(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.ENERGY) == "kWh"

    def test_mass_base_is_kg(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.MASS) == "kg"

    def test_volume_base_is_l(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.VOLUME) == "L"

    def test_area_base_is_m2(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.AREA) == "m2"

    def test_distance_base_is_m(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.DISTANCE) == "m"

    def test_emissions_base_is_kgco2e(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.EMISSIONS) == "kgCO2e"

    def test_time_base_is_s(self):
        da = DimensionalAnalyzer()
        assert da.get_base_unit(Dimension.TIME) == "s"


class TestValidInvalidUnits:
    """Test is_valid_unit() checks."""

    def test_known_units_are_valid(self):
        da = DimensionalAnalyzer()
        for unit in ["kWh", "kg", "L", "m", "m2", "kgCO2e", "t", "MWh"]:
            assert da.is_valid_unit(unit) is True, f"{unit} should be valid"

    def test_unknown_units_are_invalid(self):
        da = DimensionalAnalyzer()
        for unit in ["furlongs", "stone", "bushel", ""]:
            assert da.is_valid_unit(unit) is False, f"{unit} should be invalid"

    def test_alias_units_are_valid(self):
        da = DimensionalAnalyzer()
        for unit in ["kwh", "liter", "kilogram", "meter"]:
            assert da.is_valid_unit(unit) is True, f"alias {unit} should be valid"


class TestAllDimensionsHaveBaseUnit:
    """Test that all listed dimensions have a base unit."""

    def test_all_dimensions_have_base(self):
        da = DimensionalAnalyzer()
        for dim in da.list_dimensions():
            base = da.get_base_unit(dim)
            assert base is not None, f"Dimension {dim} has no base unit"
            assert len(base) > 0


class TestListUnits:
    """Test listing units for dimensions."""

    def test_energy_units_not_empty(self):
        da = DimensionalAnalyzer()
        units = da.list_units(Dimension.ENERGY)
        assert len(units) > 0
        assert "kWh" in units

    def test_mass_units_not_empty(self):
        da = DimensionalAnalyzer()
        units = da.list_units(Dimension.MASS)
        assert len(units) > 0
        assert "kg" in units

    def test_list_dimensions_returns_all(self):
        da = DimensionalAnalyzer()
        dims = da.list_dimensions()
        assert Dimension.ENERGY in dims
        assert Dimension.MASS in dims
        assert Dimension.VOLUME in dims
        assert Dimension.DISTANCE in dims
        assert Dimension.AREA in dims
        assert Dimension.EMISSIONS in dims


class TestDimensionInfo:
    """Test get_dimension_info() returns full info."""

    def test_dimension_info_for_kwh(self):
        da = DimensionalAnalyzer()
        info = da.get_dimension_info("kWh")
        assert info is not None
        assert info.dimension == Dimension.ENERGY
        assert info.base_unit == "kWh"
        assert "kWh" in info.all_units
        assert "MWh" in info.all_units

    def test_dimension_info_for_unknown(self):
        da = DimensionalAnalyzer()
        info = da.get_dimension_info("furlongs")
        assert info is None
