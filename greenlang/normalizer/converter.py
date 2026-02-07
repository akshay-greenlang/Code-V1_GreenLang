# -*- coding: utf-8 -*-
"""
Unit Converter - AGENT-FOUND-003: Unit & Reference Normalizer

Core unit conversion engine for the Normalizer SDK. Provides
deterministic, auditable conversions using Decimal precision
across all supported physical dimensions.

Zero-Hallucination Guarantees:
    - All conversions are deterministic Decimal arithmetic
    - NO LLM involvement in any calculation path
    - All conversion factors traceable to SI/IPCC sources
    - Complete provenance hash for every operation

Example:
    >>> from greenlang.normalizer.converter import UnitConverter
    >>> c = UnitConverter()
    >>> r = c.convert(100, "kWh", "MWh")
    >>> print(r.converted_value)  # Decimal('0.1')

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.normalizer.config import get_config
from greenlang.normalizer.models import (
    BatchConversionResult,
    ConversionResult,
    DimensionInfo,
    GHGGas,
    GWPVersion,
    UnitDimension,
    UnitInfo,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONVERSION FACTOR TABLES (from foundation agent, all Decimal)
# =============================================================================

# Mass units - base unit: kg
MASS_UNITS: Dict[str, Decimal] = {
    "g": Decimal("0.001"),
    "gram": Decimal("0.001"),
    "grams": Decimal("0.001"),
    "kg": Decimal("1"),
    "kilogram": Decimal("1"),
    "kilograms": Decimal("1"),
    "tonne": Decimal("1000"),
    "tonnes": Decimal("1000"),
    "metric_ton": Decimal("1000"),
    "t": Decimal("1000"),
    "lb": Decimal("0.45359237"),
    "lbs": Decimal("0.45359237"),
    "pound": Decimal("0.45359237"),
    "pounds": Decimal("0.45359237"),
    "short_ton": Decimal("907.18474"),
    "us_ton": Decimal("907.18474"),
    "long_ton": Decimal("1016.0469088"),
    "uk_ton": Decimal("1016.0469088"),
    "oz": Decimal("0.028349523125"),
    "ounce": Decimal("0.028349523125"),
    "mg": Decimal("0.000001"),
    "milligram": Decimal("0.000001"),
}

# Energy units - base unit: J (Joules)
ENERGY_UNITS: Dict[str, Decimal] = {
    "j": Decimal("1"),
    "joule": Decimal("1"),
    "joules": Decimal("1"),
    "kj": Decimal("1000"),
    "kilojoule": Decimal("1000"),
    "mj": Decimal("1000000"),
    "megajoule": Decimal("1000000"),
    "gj": Decimal("1000000000"),
    "gigajoule": Decimal("1000000000"),
    "kwh": Decimal("3600000"),
    "kilowatt_hour": Decimal("3600000"),
    "mwh": Decimal("3600000000"),
    "megawatt_hour": Decimal("3600000000"),
    "gwh": Decimal("3600000000000"),
    "gigawatt_hour": Decimal("3600000000000"),
    "btu": Decimal("1055.05585262"),
    "therm": Decimal("105505585.262"),
    "therms": Decimal("105505585.262"),
    "mmbtu": Decimal("1055055852.62"),
    "cal": Decimal("4.184"),
    "calorie": Decimal("4.184"),
    "kcal": Decimal("4184"),
    "kilocalorie": Decimal("4184"),
}

# Volume units - base unit: L (liters)
VOLUME_UNITS: Dict[str, Decimal] = {
    "l": Decimal("1"),
    "liter": Decimal("1"),
    "liters": Decimal("1"),
    "litre": Decimal("1"),
    "litres": Decimal("1"),
    "ml": Decimal("0.001"),
    "milliliter": Decimal("0.001"),
    "m3": Decimal("1000"),
    "cubic_meter": Decimal("1000"),
    "cubic_metre": Decimal("1000"),
    "gallon": Decimal("3.785411784"),
    "gallon_us": Decimal("3.785411784"),
    "gal": Decimal("3.785411784"),
    "gallon_uk": Decimal("4.54609"),
    "gallon_imperial": Decimal("4.54609"),
    "barrel": Decimal("158.987294928"),
    "bbl": Decimal("158.987294928"),
    "oil_barrel": Decimal("158.987294928"),
    "ft3": Decimal("28.316846592"),
    "cubic_foot": Decimal("28.316846592"),
    "cubic_feet": Decimal("28.316846592"),
    "ccf": Decimal("2831.6846592"),
    "mcf": Decimal("28316.846592"),
}

# Area units - base unit: m2
AREA_UNITS: Dict[str, Decimal] = {
    "m2": Decimal("1"),
    "sqm": Decimal("1"),
    "square_meter": Decimal("1"),
    "square_metre": Decimal("1"),
    "hectare": Decimal("10000"),
    "ha": Decimal("10000"),
    "acre": Decimal("4046.8564224"),
    "acres": Decimal("4046.8564224"),
    "km2": Decimal("1000000"),
    "sqkm": Decimal("1000000"),
    "square_kilometer": Decimal("1000000"),
    "ft2": Decimal("0.09290304"),
    "sqft": Decimal("0.09290304"),
    "square_foot": Decimal("0.09290304"),
    "square_feet": Decimal("0.09290304"),
    "mi2": Decimal("2589988.110336"),
    "square_mile": Decimal("2589988.110336"),
}

# Distance units - base unit: m
DISTANCE_UNITS: Dict[str, Decimal] = {
    "m": Decimal("1"),
    "meter": Decimal("1"),
    "metre": Decimal("1"),
    "meters": Decimal("1"),
    "km": Decimal("1000"),
    "kilometer": Decimal("1000"),
    "kilometre": Decimal("1000"),
    "mi": Decimal("1609.344"),
    "mile": Decimal("1609.344"),
    "miles": Decimal("1609.344"),
    "nmi": Decimal("1852"),
    "nautical_mile": Decimal("1852"),
    "ft": Decimal("0.3048"),
    "foot": Decimal("0.3048"),
    "feet": Decimal("0.3048"),
    "yd": Decimal("0.9144"),
    "yard": Decimal("0.9144"),
    "yards": Decimal("0.9144"),
    "cm": Decimal("0.01"),
    "centimeter": Decimal("0.01"),
    "mm": Decimal("0.001"),
    "millimeter": Decimal("0.001"),
}

# Emissions units - base unit: kgCO2e
EMISSIONS_UNITS: Dict[str, Decimal] = {
    "kgco2e": Decimal("1"),
    "kgco2": Decimal("1"),
    "kg_co2e": Decimal("1"),
    "kg_co2": Decimal("1"),
    "tco2e": Decimal("1000"),
    "tco2": Decimal("1000"),
    "t_co2e": Decimal("1000"),
    "t_co2": Decimal("1000"),
    "tonneco2e": Decimal("1000"),
    "tonneco2": Decimal("1000"),
    "mtco2e": Decimal("1000"),
    "gco2e": Decimal("0.001"),
    "gco2": Decimal("0.001"),
    "g_co2e": Decimal("0.001"),
    "lbco2e": Decimal("0.45359237"),
    "lb_co2e": Decimal("0.45359237"),
}

# Time units - base unit: seconds
TIME_UNITS: Dict[str, Decimal] = {
    "s": Decimal("1"),
    "second": Decimal("1"),
    "seconds": Decimal("1"),
    "min": Decimal("60"),
    "minute": Decimal("60"),
    "minutes": Decimal("60"),
    "h": Decimal("3600"),
    "hr": Decimal("3600"),
    "hour": Decimal("3600"),
    "hours": Decimal("3600"),
    "day": Decimal("86400"),
    "days": Decimal("86400"),
    "week": Decimal("604800"),
    "weeks": Decimal("604800"),
    "month": Decimal("2629746"),
    "months": Decimal("2629746"),
    "year": Decimal("31556952"),
    "years": Decimal("31556952"),
}

# Dimension to units mapping
DIMENSION_UNITS: Dict[UnitDimension, Dict[str, Decimal]] = {
    UnitDimension.MASS: MASS_UNITS,
    UnitDimension.ENERGY: ENERGY_UNITS,
    UnitDimension.VOLUME: VOLUME_UNITS,
    UnitDimension.AREA: AREA_UNITS,
    UnitDimension.DISTANCE: DISTANCE_UNITS,
    UnitDimension.EMISSIONS: EMISSIONS_UNITS,
    UnitDimension.TIME: TIME_UNITS,
}

# Base units per dimension
BASE_UNITS: Dict[UnitDimension, str] = {
    UnitDimension.MASS: "kg",
    UnitDimension.ENERGY: "j",
    UnitDimension.VOLUME: "l",
    UnitDimension.AREA: "m2",
    UnitDimension.DISTANCE: "m",
    UnitDimension.EMISSIONS: "kgco2e",
    UnitDimension.TIME: "s",
}


# =============================================================================
# GWP TABLES (IPCC AR5 and AR6)
# =============================================================================

GWP_AR6_100: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("29.8"),
    "N2O": Decimal("273"),
    "SF6": Decimal("25200"),
    "NF3": Decimal("17400"),
    "CO2e": Decimal("1"),
}

GWP_AR6_20: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("82.5"),
    "N2O": Decimal("273"),
    "SF6": Decimal("18300"),
    "NF3": Decimal("13400"),
    "CO2e": Decimal("1"),
}

GWP_AR5_100: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("28"),
    "N2O": Decimal("265"),
    "SF6": Decimal("23500"),
    "NF3": Decimal("16100"),
    "CO2e": Decimal("1"),
}

GWP_AR5_20: Dict[str, Decimal] = {
    "CO2": Decimal("1"),
    "CH4": Decimal("84"),
    "N2O": Decimal("264"),
    "SF6": Decimal("17500"),
    "NF3": Decimal("12800"),
    "CO2e": Decimal("1"),
}

GWP_TABLES: Dict[str, Dict[str, Decimal]] = {
    "AR6_100": GWP_AR6_100,
    "AR6_20": GWP_AR6_20,
    "AR5_100": GWP_AR5_100,
    "AR5_20": GWP_AR5_20,
}


# =============================================================================
# UNIT CONVERTER
# =============================================================================


class UnitConverter:
    """Core unit conversion engine with Decimal precision.

    This class provides deterministic, auditable unit conversions
    across all supported physical dimensions. It uses Decimal
    arithmetic exclusively for conversion factor math to prevent
    floating-point errors in regulatory reporting.

    Zero-Hallucination Guarantees:
        - All factors are static Decimal lookup tables
        - No LLM calls in any conversion path
        - SHA-256 provenance hash for every result

    Attributes:
        config: NormalizerConfig controlling precision and GWP version.

    Example:
        >>> converter = UnitConverter()
        >>> result = converter.convert(100, "kWh", "MWh")
        >>> assert result.converted_value == Decimal("0.1")
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize UnitConverter.

        Args:
            config: Optional NormalizerConfig. Uses global config if None.
        """
        self.config = config or get_config()
        logger.info("UnitConverter initialized (precision=%d)", self.config.default_precision)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        value: Any,
        from_unit: str,
        to_unit: str,
        precision: Optional[int] = None,
    ) -> ConversionResult:
        """Convert a value from one unit to another.

        Args:
            value: Numeric value to convert (int, float, str, or Decimal).
            from_unit: Source unit name.
            to_unit: Target unit name.
            precision: Decimal places (overrides config default).

        Returns:
            ConversionResult with Decimal precision values.

        Raises:
            ValueError: If units are unknown or incompatible.
        """
        prec = precision if precision is not None else self.config.default_precision

        from_norm = self.normalize_unit_name(from_unit)
        to_norm = self.normalize_unit_name(to_unit)

        from_dim = self._get_dimension(from_norm)
        to_dim = self._get_dimension(to_norm)

        if from_dim is None:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_dim is None:
            raise ValueError(f"Unknown unit: {to_unit}")
        if from_dim != to_dim:
            raise ValueError(
                f"Cannot convert between dimensions: "
                f"{from_dim.value} ({from_unit}) -> {to_dim.value} ({to_unit})"
            )

        factor = self.get_conversion_factor(from_unit, to_unit)
        value_dec = Decimal(str(value))
        converted = value_dec * factor

        precision_str = f"1e-{prec}"
        converted_rounded = converted.quantize(
            Decimal(precision_str), rounding=ROUND_HALF_UP,
        )

        provenance_hash = self._compute_hash(
            {"value": str(value), "from": from_norm, "to": to_norm},
            {"converted": str(converted_rounded)},
        )

        return ConversionResult(
            value=converted_rounded,
            from_unit=from_unit,
            to_unit=to_unit,
            from_value=value_dec,
            converted_value=converted_rounded,
            dimension=from_dim,
            conversion_factor=factor,
            provenance_hash=provenance_hash,
        )

    def batch_convert(
        self,
        items: List[Dict[str, Any]],
    ) -> BatchConversionResult:
        """Convert a batch of items.

        Each item dict must contain keys: value, from_unit, to_unit.
        Optional key: precision.

        Args:
            items: List of conversion request dicts.

        Returns:
            BatchConversionResult with all results and summary.

        Raises:
            ValueError: If batch exceeds max_batch_size.
        """
        if len(items) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(items)} exceeds max {self.config.max_batch_size}"
            )

        start = time.monotonic()
        results: List[ConversionResult] = []
        failed = 0

        for item in items:
            try:
                result = self.convert(
                    value=item["value"],
                    from_unit=item["from_unit"],
                    to_unit=item["to_unit"],
                    precision=item.get("precision"),
                )
                results.append(result)
            except (ValueError, KeyError) as exc:
                logger.warning("Batch item conversion failed: %s", exc)
                failed += 1

        duration_ms = (time.monotonic() - start) * 1000

        return BatchConversionResult(
            results=results,
            total=len(items),
            succeeded=len(results),
            failed=failed,
            duration_ms=round(duration_ms, 2),
        )

    def convert_ghg(
        self,
        value: Any,
        from_gas: str,
        to_gas: str = "CO2e",
        gwp_version: Optional[str] = None,
        gwp_timeframe: Optional[int] = None,
    ) -> ConversionResult:
        """Convert between GHG gases using GWP factors.

        Converts a mass of one greenhouse gas to its CO2-equivalent
        (or other gas) using IPCC GWP factors.

        Args:
            value: Mass value to convert.
            from_gas: Source gas name (CO2, CH4, N2O, SF6, NF3).
            to_gas: Target gas name (default CO2e).
            gwp_version: IPCC version (AR5/AR6). Uses config default if None.
            gwp_timeframe: GWP timeframe in years (20/100). Uses config default if None.

        Returns:
            ConversionResult with GWP-adjusted value.

        Raises:
            ValueError: If gas type is unknown or no GWP available.
        """
        version = gwp_version or self.config.gwp_version
        timeframe = gwp_timeframe or self.config.gwp_timeframe
        table_key = f"{version}_{timeframe}"

        gwp_table = GWP_TABLES.get(table_key)
        if gwp_table is None:
            raise ValueError(f"No GWP table for {table_key}")

        from_gwp = gwp_table.get(from_gas)
        to_gwp = gwp_table.get(to_gas)

        if from_gwp is None:
            raise ValueError(f"No GWP value for gas: {from_gas} in {table_key}")
        if to_gwp is None:
            raise ValueError(f"No GWP value for gas: {to_gas} in {table_key}")

        gwp_factor = from_gwp / to_gwp
        value_dec = Decimal(str(value))
        converted = value_dec * gwp_factor

        prec = self.config.default_precision
        precision_str = f"1e-{prec}"
        converted_rounded = converted.quantize(
            Decimal(precision_str), rounding=ROUND_HALF_UP,
        )

        provenance_hash = self._compute_hash(
            {"value": str(value), "from_gas": from_gas, "to_gas": to_gas,
             "gwp_version": version, "timeframe": timeframe},
            {"converted": str(converted_rounded), "gwp_factor": str(gwp_factor)},
        )

        return ConversionResult(
            value=converted_rounded,
            from_unit=from_gas,
            to_unit=to_gas,
            from_value=value_dec,
            converted_value=converted_rounded,
            dimension=UnitDimension.EMISSIONS,
            conversion_factor=gwp_factor,
            provenance_hash=provenance_hash,
        )

    def supported_units(
        self, dimension: Optional[UnitDimension] = None,
    ) -> List[UnitInfo]:
        """List supported units, optionally filtered by dimension.

        Args:
            dimension: Optional filter by dimension.

        Returns:
            List of UnitInfo objects.
        """
        result: List[UnitInfo] = []
        dims = [dimension] if dimension else list(DIMENSION_UNITS.keys())

        for dim in dims:
            units_table = DIMENSION_UNITS.get(dim, {})
            base = BASE_UNITS.get(dim, "")
            for symbol, factor in units_table.items():
                result.append(UnitInfo(
                    symbol=symbol,
                    dimension=dim,
                    to_base_factor=str(factor),
                    base_unit=base,
                ))

        return result

    def supported_dimensions(self) -> List[DimensionInfo]:
        """List all supported dimensions with their units.

        Returns:
            List of DimensionInfo objects.
        """
        result: List[DimensionInfo] = []
        for dim, units_table in DIMENSION_UNITS.items():
            result.append(DimensionInfo(
                dimension=dim,
                base_unit=BASE_UNITS.get(dim, ""),
                supported_units=list(units_table.keys()),
            ))
        return result

    def get_conversion_factor(
        self, from_unit: str, to_unit: str,
    ) -> Decimal:
        """Get the Decimal conversion factor between two units.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            Conversion factor as Decimal (from_base / to_base).

        Raises:
            ValueError: If units are unknown or incompatible.
        """
        from_norm = self.normalize_unit_name(from_unit)
        to_norm = self.normalize_unit_name(to_unit)

        from_dim = self._get_dimension(from_norm)
        to_dim = self._get_dimension(to_norm)

        if from_dim is None:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_dim is None:
            raise ValueError(f"Unknown unit: {to_unit}")
        if from_dim != to_dim:
            raise ValueError(
                f"Cannot get factor between dimensions: "
                f"{from_dim.value} ({from_unit}) -> {to_dim.value} ({to_unit})"
            )

        units_table = DIMENSION_UNITS[from_dim]
        from_factor = units_table[from_norm]
        to_factor = units_table[to_norm]

        return from_factor / to_factor

    def normalize_unit_name(self, unit_str: str) -> str:
        """Normalize a unit name to canonical lowercase form.

        Handles case normalization, whitespace/hyphen/caret
        standardization, and subscript equivalents.

        Args:
            unit_str: Raw unit string.

        Returns:
            Normalized unit string (lowercase, underscored).
        """
        normalized = unit_str.lower().strip()
        normalized = normalized.replace("-", "_").replace(" ", "_")
        normalized = normalized.replace("^2", "2").replace("^3", "3")
        normalized = normalized.replace("\u2082", "2")  # subscript 2
        return normalized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_dimension(self, unit_normalized: str) -> Optional[UnitDimension]:
        """Determine the dimension for a normalised unit name.

        Args:
            unit_normalized: Already-normalised unit string.

        Returns:
            UnitDimension or None if not found.
        """
        for dimension, units_table in DIMENSION_UNITS.items():
            if unit_normalized in units_table:
                return dimension
        return None

    def _compute_hash(self, input_data: Any, output_data: Any) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            input_data: Input data dict.
            output_data: Output data dict.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


__all__ = [
    "UnitConverter",
    # Tables exported for downstream use
    "MASS_UNITS",
    "ENERGY_UNITS",
    "VOLUME_UNITS",
    "AREA_UNITS",
    "DISTANCE_UNITS",
    "EMISSIONS_UNITS",
    "TIME_UNITS",
    "DIMENSION_UNITS",
    "BASE_UNITS",
    "GWP_AR6_100",
    "GWP_AR6_20",
    "GWP_AR5_100",
    "GWP_AR5_20",
    "GWP_TABLES",
]
