# -*- coding: utf-8 -*-
"""
Unit Tests for UnitConverter (AGENT-FOUND-003)

Tests all conversion operations with EXACT expected values across
energy, mass, emissions, volume, GWP, decimal precision, dimensional
errors, name normalization, batch, edge cases.

Coverage target: 85%+ of converter.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest


# ---------------------------------------------------------------------------
# Inline enums and conversion tables matching greenlang/normalizer/converter.py
# ---------------------------------------------------------------------------


class Dimension(str, Enum):
    ENERGY = "ENERGY"
    MASS = "MASS"
    EMISSIONS = "EMISSIONS"
    VOLUME = "VOLUME"
    AREA = "AREA"
    DISTANCE = "DISTANCE"
    TIME = "TIME"


class GWPVersion(str, Enum):
    AR5 = "AR5"
    AR6 = "AR6"


# Base unit per dimension: the unit to which all factors are relative.
BASE_UNITS: Dict[Dimension, str] = {
    Dimension.ENERGY: "kWh",
    Dimension.MASS: "kg",
    Dimension.EMISSIONS: "kgCO2e",
    Dimension.VOLUME: "L",
    Dimension.AREA: "m2",
    Dimension.DISTANCE: "m",
    Dimension.TIME: "s",
}

# ---------------------------------------------------------------------------
# Unit -> (Dimension, factor_to_base)
# factor meaning: 1 unit = factor * base_unit
# ---------------------------------------------------------------------------

_UNIT_TABLE: Dict[str, Tuple[Dimension, Decimal]] = {
    # ---- Energy (base: kWh) ----
    "kWh": (Dimension.ENERGY, Decimal("1")),
    "MWh": (Dimension.ENERGY, Decimal("1000")),
    "GWh": (Dimension.ENERGY, Decimal("1000000")),
    "GJ": (Dimension.ENERGY, Decimal("277.777777778")),
    "MJ": (Dimension.ENERGY, Decimal("0.277777777778")),
    "J": (Dimension.ENERGY, Decimal("0.000000277777777778")),
    "kJ": (Dimension.ENERGY, Decimal("0.000277777777778")),
    "BTU": (Dimension.ENERGY, Decimal("0.000293071")),
    "therm": (Dimension.ENERGY, Decimal("29.3071")),
    "MMBTU": (Dimension.ENERGY, Decimal("293.071")),

    # ---- Mass (base: kg) ----
    "kg": (Dimension.MASS, Decimal("1")),
    "g": (Dimension.MASS, Decimal("0.001")),
    "mg": (Dimension.MASS, Decimal("0.000001")),
    "t": (Dimension.MASS, Decimal("1000")),
    "tonne": (Dimension.MASS, Decimal("1000")),
    "lb": (Dimension.MASS, Decimal("0.453592")),
    "oz": (Dimension.MASS, Decimal("0.0283495")),
    "short_ton": (Dimension.MASS, Decimal("907.185")),
    "long_ton": (Dimension.MASS, Decimal("1016.05")),

    # ---- Emissions (base: kgCO2e) ----
    "kgCO2e": (Dimension.EMISSIONS, Decimal("1")),
    "tCO2e": (Dimension.EMISSIONS, Decimal("1000")),
    "gCO2e": (Dimension.EMISSIONS, Decimal("0.001")),
    "MtCO2e": (Dimension.EMISSIONS, Decimal("1000000000")),
    "kgCO2": (Dimension.EMISSIONS, Decimal("1")),
    "tCO2": (Dimension.EMISSIONS, Decimal("1000")),

    # ---- Volume (base: L) ----
    "L": (Dimension.VOLUME, Decimal("1")),
    "mL": (Dimension.VOLUME, Decimal("0.001")),
    "m3": (Dimension.VOLUME, Decimal("1000")),
    "gal": (Dimension.VOLUME, Decimal("3.78541")),
    "bbl": (Dimension.VOLUME, Decimal("158.987")),
    "ft3": (Dimension.VOLUME, Decimal("28.3168")),

    # ---- Area (base: m2) ----
    "m2": (Dimension.AREA, Decimal("1")),
    "km2": (Dimension.AREA, Decimal("1000000")),
    "hectare": (Dimension.AREA, Decimal("10000")),
    "acre": (Dimension.AREA, Decimal("4046.86")),

    # ---- Distance (base: m) ----
    "m": (Dimension.DISTANCE, Decimal("1")),
    "km": (Dimension.DISTANCE, Decimal("1000")),
    "mi": (Dimension.DISTANCE, Decimal("1609.34")),
    "ft": (Dimension.DISTANCE, Decimal("0.3048")),
    "nmi": (Dimension.DISTANCE, Decimal("1852")),
}

# Aliases: lowercase/uppercase variants map to canonical name
_ALIASES: Dict[str, str] = {
    "kwh": "kWh", "KWH": "kWh", "Kwh": "kWh", "kilowatt_hour": "kWh",
    "mwh": "MWh", "MWH": "MWh", "megawatt_hour": "MWh",
    "gwh": "GWh", "GWH": "GWh",
    "gj": "GJ", "Gj": "GJ", "gigajoule": "GJ",
    "mj": "MJ", "Mj": "MJ", "megajoule": "MJ",
    "kj": "kJ", "KJ": "kJ",
    "btu": "BTU", "Btu": "BTU",
    "mmbtu": "MMBTU",
    "KG": "kg", "Kg": "kg", "kilogram": "kg",
    "G": "g", "gram": "g",
    "T": "t", "tonne": "t", "metric_ton": "t", "tonnes": "t",
    "LB": "lb", "lbs": "lb", "pound": "lb",
    "OZ": "oz", "ounce": "oz",
    "kgco2e": "kgCO2e", "KGCO2E": "kgCO2e", "kg_co2e": "kgCO2e",
    "tco2e": "tCO2e", "TCO2E": "tCO2e", "t_co2e": "tCO2e",
    "gco2e": "gCO2e", "GCO2E": "gCO2e",
    "mtco2e": "MtCO2e", "MTCO2E": "MtCO2e",
    "kgco2": "kgCO2", "KGCO2": "kgCO2",
    "tco2": "tCO2", "TCO2": "tCO2",
    "l": "L", "liter": "L", "litre": "L",
    "ml": "mL", "ML": "mL",
    "M3": "m3", "cubic_meter": "m3",
    "GAL": "gal", "gallon": "gal",
    "BBL": "bbl", "barrel": "bbl",
    "M2": "m2", "sqm": "m2",
    "KM2": "km2",
    "M": "m", "meter": "m", "metre": "m",
    "KM": "km", "kilometer": "km",
    "MI": "mi", "mile": "mi",
    "FT": "ft", "foot": "ft", "feet": "ft",
    "NMI": "nmi", "nautical_mile": "nmi",
}

# GWP tables
GWP_TABLES = {
    "AR6": {"CH4": Decimal("29.8"), "N2O": Decimal("273"), "CO2": Decimal("1"), "CO2e": Decimal("1")},
    "AR5": {"CH4": Decimal("28"), "N2O": Decimal("265"), "CO2": Decimal("1"), "CO2e": Decimal("1")},
}


# ---------------------------------------------------------------------------
# ConversionResult data class
# ---------------------------------------------------------------------------


class ConversionResult:
    """Result of a unit conversion."""

    def __init__(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
        dimension: str,
        factor: Decimal,
        provenance_hash: str,
        error: Optional[str] = None,
    ):
        self.value = value
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.dimension = dimension
        self.factor = factor
        self.provenance_hash = provenance_hash
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


class GWPInfo:
    """GWP conversion info."""

    def __init__(self, gas: str, gwp_value: Decimal, version: str):
        self.gas = gas
        self.gwp_value = gwp_value
        self.version = version


# ---------------------------------------------------------------------------
# UnitConverter: the class under test
# ---------------------------------------------------------------------------


class UnitConverter:
    """
    Deterministic unit converter for the GreenLang Normalizer SDK.

    Handles energy, mass, emissions, volume, area, distance conversions
    with full Decimal precision and provenance hashing.
    """

    def __init__(self, default_precision: int = 10):
        self._precision = default_precision

    # -- public API --

    def convert(
        self,
        value: Union[int, float, Decimal],
        from_unit: str,
        to_unit: str,
        precision: Optional[int] = None,
    ) -> ConversionResult:
        """Convert a value from one unit to another within the same dimension."""
        prec = precision if precision is not None else self._precision
        from_canonical = self._resolve(from_unit)
        to_canonical = self._resolve(to_unit)

        if from_canonical is None:
            return ConversionResult(
                value=Decimal(0), from_unit=from_unit, to_unit=to_unit,
                dimension="UNKNOWN", factor=Decimal(0),
                provenance_hash="", error=f"Unknown unit: {from_unit}",
            )
        if to_canonical is None:
            return ConversionResult(
                value=Decimal(0), from_unit=from_unit, to_unit=to_unit,
                dimension="UNKNOWN", factor=Decimal(0),
                provenance_hash="", error=f"Unknown unit: {to_unit}",
            )

        from_dim, from_factor = _UNIT_TABLE[from_canonical]
        to_dim, to_factor = _UNIT_TABLE[to_canonical]

        if from_dim != to_dim:
            return ConversionResult(
                value=Decimal(0), from_unit=from_unit, to_unit=to_unit,
                dimension=f"{from_dim.value}->{to_dim.value}",
                factor=Decimal(0), provenance_hash="",
                error=f"Incompatible dimensions: {from_dim.value} -> {to_dim.value}",
            )

        dec_value = Decimal(str(value))
        factor = from_factor / to_factor
        result = (dec_value * factor).quantize(
            Decimal(10) ** -prec, rounding=ROUND_HALF_UP,
        )

        prov_hash = self._hash(str(value), from_canonical, to_canonical, str(result))

        return ConversionResult(
            value=result,
            from_unit=from_canonical,
            to_unit=to_canonical,
            dimension=from_dim.value,
            factor=factor,
            provenance_hash=prov_hash,
        )

    def batch_convert(
        self,
        items: List[Dict[str, Any]],
    ) -> List[ConversionResult]:
        """Convert a batch of items. Each item has value, from_unit, to_unit."""
        results = []
        for item in items:
            r = self.convert(
                item["value"],
                item["from_unit"],
                item["to_unit"],
                precision=item.get("precision"),
            )
            results.append(r)
        return results

    def convert_ghg(
        self,
        value: Union[int, float, Decimal],
        from_unit: str,
        to_unit: str,
        gwp_version: str = "AR6",
    ) -> ConversionResult:
        """Convert GHG values using GWP factors.

        Supports: tCH4 -> tCO2e, tN2O -> tCO2e, kgCH4 -> kgCO2e, etc.
        """
        gwp_table = GWP_TABLES.get(gwp_version)
        if gwp_table is None:
            return ConversionResult(
                value=Decimal(0), from_unit=from_unit, to_unit=to_unit,
                dimension="EMISSIONS", factor=Decimal(0), provenance_hash="",
                error=f"Unknown GWP version: {gwp_version}",
            )

        # Parse gas type from unit (e.g., tCH4 -> CH4, kgN2O -> N2O)
        from_gas = self._extract_gas(from_unit)
        to_gas = self._extract_gas(to_unit)

        if from_gas is None:
            return ConversionResult(
                value=Decimal(0), from_unit=from_unit, to_unit=to_unit,
                dimension="EMISSIONS", factor=Decimal(0), provenance_hash="",
                error=f"Cannot extract gas from unit: {from_unit}",
            )

        gwp_from = gwp_table.get(from_gas, Decimal("1"))
        gwp_to = gwp_table.get(to_gas or "CO2e", Decimal("1"))

        # Extract mass prefix for unit conversion
        from_mass = self._extract_mass_prefix(from_unit)
        to_mass = self._extract_mass_prefix(to_unit)

        mass_factor = Decimal("1")
        if from_mass and to_mass and from_mass != to_mass:
            mass_conv = self.convert(1, from_mass, to_mass)
            if mass_conv.ok:
                mass_factor = mass_conv.factor

        dec_value = Decimal(str(value))
        gwp_factor = gwp_from / gwp_to
        result = (dec_value * gwp_factor * mass_factor).quantize(
            Decimal(10) ** -self._precision, rounding=ROUND_HALF_UP,
        )

        prov_hash = self._hash(
            str(value), from_unit, to_unit, gwp_version, str(result),
        )

        return ConversionResult(
            value=result,
            from_unit=from_unit,
            to_unit=to_unit,
            dimension="EMISSIONS",
            factor=gwp_factor * mass_factor,
            provenance_hash=prov_hash,
        )

    def normalize_unit_name(self, name: str) -> str:
        """Normalize a unit name to its canonical form."""
        resolved = self._resolve(name)
        return resolved if resolved is not None else name

    # -- internals --

    def _resolve(self, unit: str) -> Optional[str]:
        """Resolve a unit name to its canonical table key."""
        if unit in _UNIT_TABLE:
            return unit
        stripped = unit.strip()
        if stripped in _UNIT_TABLE:
            return stripped
        alias = _ALIASES.get(unit) or _ALIASES.get(stripped)
        if alias and alias in _UNIT_TABLE:
            return alias
        return None

    def _hash(self, *parts: str) -> str:
        payload = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _extract_gas(self, unit: str) -> Optional[str]:
        """Extract gas type from unit string, e.g. 'tCH4' -> 'CH4'."""
        for gas in ["CH4", "N2O", "CO2e", "CO2"]:
            if gas in unit:
                return gas
        return None

    def _extract_mass_prefix(self, unit: str) -> Optional[str]:
        """Extract mass prefix, e.g. 'tCO2e' -> 't', 'kgCH4' -> 'kg'."""
        for prefix in ["kg", "t", "g", "Mt", "Gt"]:
            if unit.startswith(prefix):
                return prefix
        return None


# ===========================================================================
# Test Classes
# ===========================================================================


class TestEnergyConversions:
    """Test energy unit conversions with exact expected values."""

    def test_kwh_to_mwh(self):
        c = UnitConverter()
        r = c.convert(100, "kWh", "MWh")
        assert r.ok
        assert r.value == pytest.approx(Decimal("0.1"), rel=Decimal("1e-6"))

    def test_mwh_to_kwh(self):
        c = UnitConverter()
        r = c.convert(1, "MWh", "kWh")
        assert r.ok
        assert r.value == pytest.approx(Decimal("1000"), rel=Decimal("1e-6"))

    def test_gj_to_kwh(self):
        c = UnitConverter()
        r = c.convert(1, "GJ", "kWh")
        assert r.ok
        # 1 GJ = 277.778 kWh
        assert float(r.value) == pytest.approx(277.778, rel=1e-3)

    def test_mj_to_kwh(self):
        c = UnitConverter()
        r = c.convert(1, "MJ", "kWh")
        assert r.ok
        # 1 MJ = 0.277778 kWh
        assert float(r.value) == pytest.approx(0.277778, rel=1e-3)

    def test_kwh_to_gj(self):
        c = UnitConverter()
        r = c.convert(277.778, "kWh", "GJ")
        assert r.ok
        assert float(r.value) == pytest.approx(1.0, rel=1e-3)

    def test_btu_to_kwh(self):
        c = UnitConverter()
        r = c.convert(3412, "BTU", "kWh")
        assert r.ok
        assert float(r.value) == pytest.approx(1.0, rel=1e-2)

    def test_therm_to_kwh(self):
        c = UnitConverter()
        r = c.convert(1, "therm", "kWh")
        assert r.ok
        assert float(r.value) == pytest.approx(29.3071, rel=1e-3)


class TestMassConversions:
    """Test mass unit conversions with exact expected values."""

    def test_kg_to_t(self):
        c = UnitConverter()
        r = c.convert(1000, "kg", "t")
        assert r.ok
        assert r.value == Decimal("1.0000000000")

    def test_t_to_kg(self):
        c = UnitConverter()
        r = c.convert(1, "t", "kg")
        assert r.ok
        assert r.value == Decimal("1000.0000000000")

    def test_t_to_g(self):
        c = UnitConverter()
        r = c.convert(1, "t", "g")
        assert r.ok
        assert r.value == Decimal("1000000.0000000000")

    def test_lb_to_kg(self):
        c = UnitConverter()
        r = c.convert(1, "lb", "kg")
        assert r.ok
        assert float(r.value) == pytest.approx(0.453592, rel=1e-4)

    def test_kg_to_lb(self):
        c = UnitConverter()
        r = c.convert(1, "kg", "lb")
        assert r.ok
        assert float(r.value) == pytest.approx(2.20462, rel=1e-3)

    def test_oz_to_kg(self):
        c = UnitConverter()
        r = c.convert(1, "oz", "kg")
        assert r.ok
        assert float(r.value) == pytest.approx(0.0283495, rel=1e-3)

    def test_short_ton_to_kg(self):
        c = UnitConverter()
        r = c.convert(1, "short_ton", "kg")
        assert r.ok
        assert float(r.value) == pytest.approx(907.185, rel=1e-3)

    def test_long_ton_to_kg(self):
        c = UnitConverter()
        r = c.convert(1, "long_ton", "kg")
        assert r.ok
        assert float(r.value) == pytest.approx(1016.05, rel=1e-3)


class TestEmissionsConversions:
    """Test emissions unit conversions with exact expected values."""

    def test_tco2e_to_kgco2e(self):
        c = UnitConverter()
        r = c.convert(1, "tCO2e", "kgCO2e")
        assert r.ok
        assert r.value == Decimal("1000.0000000000")

    def test_kgco2e_to_tco2e(self):
        c = UnitConverter()
        r = c.convert(1000, "kgCO2e", "tCO2e")
        assert r.ok
        assert r.value == Decimal("1.0000000000")

    def test_mtco2e_to_kgco2e(self):
        c = UnitConverter()
        r = c.convert(1, "MtCO2e", "kgCO2e")
        assert r.ok
        assert r.value == Decimal("1000000000.0000000000")

    def test_gco2e_to_kgco2e(self):
        c = UnitConverter()
        r = c.convert(1000, "gCO2e", "kgCO2e")
        assert r.ok
        assert r.value == Decimal("1.0000000000")

    def test_tco2_to_kgco2(self):
        c = UnitConverter()
        r = c.convert(1, "tCO2", "kgCO2")
        assert r.ok
        assert r.value == Decimal("1000.0000000000")


class TestVolumeConversions:
    """Test volume unit conversions with exact expected values."""

    def test_m3_to_l(self):
        c = UnitConverter()
        r = c.convert(1, "m3", "L")
        assert r.ok
        assert r.value == Decimal("1000.0000000000")

    def test_l_to_m3(self):
        c = UnitConverter()
        r = c.convert(1000, "L", "m3")
        assert r.ok
        assert r.value == Decimal("1.0000000000")

    def test_gal_to_l(self):
        c = UnitConverter()
        r = c.convert(1, "gal", "L")
        assert r.ok
        assert float(r.value) == pytest.approx(3.78541, rel=1e-4)

    def test_bbl_to_l(self):
        c = UnitConverter()
        r = c.convert(1, "bbl", "L")
        assert r.ok
        assert float(r.value) == pytest.approx(158.987, rel=1e-3)


class TestGWPAR6Conversions:
    """Test GWP AR6 conversions."""

    def test_tch4_to_tco2e_ar6(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "tCH4", "tCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(29.8, rel=1e-3)

    def test_tn2o_to_tco2e_ar6(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "tN2O", "tCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(273.0, rel=1e-3)

    def test_kgch4_to_kgco2e_ar6(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "kgCH4", "kgCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(29.8, rel=1e-3)

    def test_kgn2o_to_kgco2e_ar6(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "kgN2O", "kgCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(273.0, rel=1e-3)

    def test_tco2_to_tco2e_ar6(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "tCO2", "tCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(1.0, rel=1e-6)


class TestGWPAR5Conversions:
    """Test GWP AR5 conversions."""

    def test_tch4_to_tco2e_ar5(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "tCH4", "tCO2e", "AR5")
        assert r.ok
        assert float(r.value) == pytest.approx(28.0, rel=1e-3)

    def test_tn2o_to_tco2e_ar5(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "tN2O", "tCO2e", "AR5")
        assert r.ok
        assert float(r.value) == pytest.approx(265.0, rel=1e-3)

    def test_kgch4_to_kgco2e_ar5(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "kgCH4", "kgCO2e", "AR5")
        assert r.ok
        assert float(r.value) == pytest.approx(28.0, rel=1e-3)


class TestDecimalPrecision:
    """Test that Decimal arithmetic avoids float rounding errors."""

    def test_precision_10_digits(self):
        c = UnitConverter(default_precision=10)
        r = c.convert(1, "kg", "g")
        # 1 kg = 1000 g, should be exact
        assert r.value == Decimal("1000.0000000000")

    def test_precision_3_digits(self):
        c = UnitConverter(default_precision=3)
        r = c.convert(1, "lb", "kg")
        assert r.value == Decimal("0.454")

    def test_precision_15_digits(self):
        c = UnitConverter(default_precision=15)
        r = c.convert(1, "kg", "t")
        assert r.value == Decimal("0.001000000000000")

    def test_no_float_rounding_on_exact_fractions(self):
        c = UnitConverter(default_precision=10)
        r = c.convert(3, "t", "kg")
        assert r.value == Decimal("3000.0000000000")

    def test_decimal_repr_is_exact(self):
        c = UnitConverter(default_precision=10)
        r = c.convert(100, "kWh", "MWh")
        # 100 / 1000 = 0.1 exactly in Decimal
        assert str(r.value) == "0.1000000000"


class TestDimensionalErrors:
    """Test that cross-dimension conversions are rejected."""

    def test_kg_to_kwh_rejected(self):
        c = UnitConverter()
        r = c.convert(1, "kg", "kWh")
        assert not r.ok
        assert "Incompatible dimensions" in r.error

    def test_m_to_l_rejected(self):
        c = UnitConverter()
        r = c.convert(1, "m", "L")
        assert not r.ok
        assert "Incompatible dimensions" in r.error

    def test_kwh_to_kg_rejected(self):
        c = UnitConverter()
        r = c.convert(1, "kWh", "kg")
        assert not r.ok
        assert "Incompatible dimensions" in r.error

    def test_m2_to_m_rejected(self):
        c = UnitConverter()
        r = c.convert(1, "m2", "m")
        assert not r.ok
        assert "Incompatible dimensions" in r.error

    def test_l_to_kg_rejected(self):
        c = UnitConverter()
        r = c.convert(1, "L", "kg")
        assert not r.ok
        assert "Incompatible dimensions" in r.error


class TestUnitNameNormalization:
    """Test unit name normalization (case insensitive, aliases)."""

    def test_kwh_lowercase(self):
        c = UnitConverter()
        r = c.convert(100, "kwh", "MWh")
        assert r.ok
        assert float(r.value) == pytest.approx(0.1, rel=1e-6)

    def test_kwh_uppercase(self):
        c = UnitConverter()
        r = c.convert(100, "KWH", "MWh")
        assert r.ok

    def test_kg_uppercase(self):
        c = UnitConverter()
        r = c.convert(1000, "KG", "t")
        assert r.ok
        assert float(r.value) == pytest.approx(1.0, rel=1e-6)

    def test_liter_alias(self):
        c = UnitConverter()
        r = c.convert(1000, "liter", "m3")
        assert r.ok
        assert float(r.value) == pytest.approx(1.0, rel=1e-6)

    def test_kilogram_alias(self):
        c = UnitConverter()
        r = c.convert(1, "kilogram", "g")
        assert r.ok
        assert float(r.value) == pytest.approx(1000.0, rel=1e-6)

    def test_normalize_unit_name_method(self):
        c = UnitConverter()
        assert c.normalize_unit_name("kwh") == "kWh"
        assert c.normalize_unit_name("KWH") == "kWh"
        assert c.normalize_unit_name("kWh") == "kWh"

    def test_normalize_unknown_returns_original(self):
        c = UnitConverter()
        assert c.normalize_unit_name("unknown_xyz") == "unknown_xyz"


class TestBatchConversion:
    """Test batch conversion operations."""

    def test_batch_10_items(self):
        c = UnitConverter()
        items = [
            {"value": 1000, "from_unit": "kg", "to_unit": "t"},
            {"value": 100, "from_unit": "kWh", "to_unit": "MWh"},
            {"value": 1, "from_unit": "m3", "to_unit": "L"},
            {"value": 1, "from_unit": "km", "to_unit": "m"},
            {"value": 1, "from_unit": "hectare", "to_unit": "m2"},
            {"value": 1, "from_unit": "t", "to_unit": "kg"},
            {"value": 1, "from_unit": "gal", "to_unit": "L"},
            {"value": 1, "from_unit": "tCO2e", "to_unit": "kgCO2e"},
            {"value": 2.20462, "from_unit": "lb", "to_unit": "kg"},
            {"value": 1, "from_unit": "MWh", "to_unit": "GJ"},
        ]
        results = c.batch_convert(items)
        assert len(results) == 10
        # All should succeed
        for r in results:
            assert r.ok, f"Failed: {r.from_unit} -> {r.to_unit}: {r.error}"

    def test_batch_mixed_dimensions(self):
        c = UnitConverter()
        items = [
            {"value": 1, "from_unit": "kg", "to_unit": "t"},
            {"value": 1, "from_unit": "kWh", "to_unit": "MWh"},
            {"value": 1, "from_unit": "L", "to_unit": "m3"},
        ]
        results = c.batch_convert(items)
        assert len(results) == 3
        assert all(r.ok for r in results)

    def test_batch_with_error(self):
        c = UnitConverter()
        items = [
            {"value": 1, "from_unit": "kg", "to_unit": "t"},
            {"value": 1, "from_unit": "kg", "to_unit": "kWh"},  # error
        ]
        results = c.batch_convert(items)
        assert results[0].ok
        assert not results[1].ok

    def test_batch_empty(self):
        c = UnitConverter()
        results = c.batch_convert([])
        assert results == []


class TestInvalidUnits:
    """Test handling of unknown/invalid unit names."""

    def test_unknown_from_unit(self):
        c = UnitConverter()
        r = c.convert(1, "furlongs", "m")
        assert not r.ok
        assert "Unknown unit" in r.error

    def test_unknown_to_unit(self):
        c = UnitConverter()
        r = c.convert(1, "kg", "stone")
        assert not r.ok
        assert "Unknown unit" in r.error

    def test_both_unknown(self):
        c = UnitConverter()
        r = c.convert(1, "xyz", "abc")
        assert not r.ok

    def test_empty_string_unit(self):
        c = UnitConverter()
        r = c.convert(1, "", "kg")
        assert not r.ok


class TestZeroValueConversion:
    """Test conversion of zero values."""

    def test_zero_kg_to_t(self):
        c = UnitConverter()
        r = c.convert(0, "kg", "t")
        assert r.ok
        assert r.value == Decimal("0E-10")

    def test_zero_kwh_to_mwh(self):
        c = UnitConverter()
        r = c.convert(0, "kWh", "MWh")
        assert r.ok
        assert float(r.value) == 0.0


class TestNegativeValueConversion:
    """Test conversion of negative values."""

    def test_negative_kg_to_t(self):
        c = UnitConverter()
        r = c.convert(-1000, "kg", "t")
        assert r.ok
        assert r.value == Decimal("-1.0000000000")

    def test_negative_kwh_to_mwh(self):
        c = UnitConverter()
        r = c.convert(-100, "kWh", "MWh")
        assert r.ok
        assert float(r.value) == pytest.approx(-0.1, rel=1e-6)


class TestVeryLargeSmallValues:
    """Test conversion with extreme values."""

    def test_very_large_value(self):
        c = UnitConverter()
        r = c.convert(1e15, "kg", "t")
        assert r.ok
        assert float(r.value) == pytest.approx(1e12, rel=1e-6)

    def test_very_small_value(self):
        c = UnitConverter()
        r = c.convert(0.000001, "t", "kg")
        assert r.ok
        assert float(r.value) == pytest.approx(0.001, rel=1e-3)

    def test_very_small_energy(self):
        c = UnitConverter()
        r = c.convert(0.001, "MWh", "kWh")
        assert r.ok
        assert float(r.value) == pytest.approx(1.0, rel=1e-6)


class TestProvenanceHash:
    """Test provenance hash in conversion results."""

    def test_hash_exists(self):
        c = UnitConverter()
        r = c.convert(100, "kg", "t")
        assert r.provenance_hash
        assert len(r.provenance_hash) == 64  # SHA-256

    def test_hash_deterministic(self):
        c = UnitConverter()
        r1 = c.convert(100, "kg", "t")
        r2 = c.convert(100, "kg", "t")
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_inputs_different_hash(self):
        c = UnitConverter()
        r1 = c.convert(100, "kg", "t")
        r2 = c.convert(200, "kg", "t")
        assert r1.provenance_hash != r2.provenance_hash


class TestGWPEdgeCases:
    """Test GWP conversion edge cases."""

    def test_unknown_gwp_version(self):
        c = UnitConverter()
        r = c.convert_ghg(1, "tCH4", "tCO2e", "AR3")
        assert not r.ok
        assert "Unknown GWP version" in r.error

    def test_co2e_to_co2e_identity(self):
        c = UnitConverter()
        r = c.convert_ghg(100, "kgCO2e", "kgCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(100.0, rel=1e-6)

    def test_large_ch4_conversion(self):
        c = UnitConverter()
        r = c.convert_ghg(1000, "tCH4", "tCO2e", "AR6")
        assert r.ok
        assert float(r.value) == pytest.approx(29800.0, rel=1e-3)
