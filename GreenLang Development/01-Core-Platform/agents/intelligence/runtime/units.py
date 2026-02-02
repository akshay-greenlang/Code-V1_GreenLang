# -*- coding: utf-8 -*-
"""
Unit System & Normalization (INTL-103)

Wraps pint for unit handling with GreenLang-specific requirements:
- Canonical unit conversions (e.g., 1 tCO2e → 1000 kgCO2e)
- Dimension validation (mass vs energy vs power)
- Decimal arithmetic for reproducibility
- Allowlist enforcement

CTO Specification: All numeric values carry units; dimensions are validated.
"""

from __future__ import annotations
from decimal import Decimal, getcontext
from typing import Optional, Tuple, Set
import pint

from .schemas import Quantity
from .errors import GLValidationError

# Set decimal precision for reproducibility
getcontext().prec = 28


class UnitRegistry:
    """
    Unit registry with canonical conversions

    Wraps pint for:
    - Unit parsing and validation
    - Canonical unit normalization (e.g., all mass → kg)
    - Dimension checking (energy vs power vs mass)
    - Allowlist enforcement

    Example:
        ureg = UnitRegistry()

        # Normalize
        value, unit = ureg.normalize(Quantity(value=1.0, unit="tCO2e"))
        assert value == Decimal("1000")
        assert unit == "kgCO2e"

        # Check if allowed
        assert ureg.is_allowed("kWh")
        assert not ureg.is_allowed("parsecs")

        # Compare quantities
        q1 = Quantity(value=1000, unit="g")
        q2 = Quantity(value=1, unit="kg")
        assert ureg.same_quantity(q1, q2)
    """

    def __init__(self):
        """Initialize with pint and define custom units"""
        # Create pint registry
        self.ureg = pint.UnitRegistry(
            force_ndarray=False,  # Use scalars, not arrays
            autoconvert_offset_to_baseunit=True,  # Handle °C → K
        )

        # Define custom climate/energy units
        self._define_custom_units()

        # Canonical units by dimension
        self.canonical_units = {
            "mass": "kg",
            "energy": "kWh",
            "power": "kW",
            "volume": "m3",
            "length": "m",
            "temperature": "K",  # Store in K, render in °C
            "dimensionless": "",  # For percentages
        }

        # Allowlist of valid units
        self.allowlist: Set[str] = {
            # Energy
            "Wh",
            "kWh",
            "MWh",
            "GWh",
            "MJ",
            "GJ",
            # Power
            "W",
            "kW",
            "MW",
            # Emissions
            "gCO2e",
            "kgCO2e",
            "tCO2e",
            # Dimensionless
            "%",
            "percent",
            # Currency (ISO 4217) - treated as tagged, non-convertible
            "USD",
            "EUR",
            "GBP",
            "INR",
            "CNY",
            "JPY",
            # Volume
            "m3",
            "L",
            "gal",
            # Mass
            "g",
            "kg",
            "t",
            "ton",
            # Length
            "m",
            "km",
            "mi",
            "ft",
            # Temperature
            "K",
            "C",
            "degC",
            "F",
            "degF",
            # Area
            "m2",
            "km2",
            "ft2",
            # Intensity units
            "kWh/m2",
            "kgCO2e/m2",
            "kWh/m2/year",
        }

    def _define_custom_units(self):
        """Define GreenLang-specific units"""
        # CO2e units
        self.ureg.define("kgCO2e = kg")
        self.ureg.define("gCO2e = 0.001 * kgCO2e")
        self.ureg.define("tCO2e = 1000 * kgCO2e")

        # Alias for percent
        self.ureg.define("percent = 0.01 * dimensionless")

        # Currency units (tagged, non-convertible)
        # Each currency is its own dimension - they cannot be converted to each other
        # This ensures 100 USD ≠ 100 EUR
        self.ureg.define("USD = [currency_USD]")
        self.ureg.define("EUR = [currency_EUR]")
        self.ureg.define("GBP = [currency_GBP]")
        self.ureg.define("INR = [currency_INR]")
        self.ureg.define("CNY = [currency_CNY]")
        self.ureg.define("JPY = [currency_JPY]")

        # Energy units (already in pint, but ensure consistency)
        # pint has: Wh, kWh, MWh, GWh, MJ, GJ

        # Enable parsing of "/" in compound units (e.g., "kWh/m2")
        self.ureg.default_format = "~P"  # Use pretty formatting

    def _preprocess_unit(self, unit: str) -> str:
        """
        Preprocess unit string for pint compatibility

        Converts shorthand notation to pint-compatible format:
        - m2 → m**2
        - m3 → m**3
        - ft2 → ft**2
        - km2 → km**2

        Args:
            unit: Original unit string

        Returns:
            Pint-compatible unit string
        """
        # Replace common area/volume shortcuts
        unit = unit.replace("m2", "m**2")
        unit = unit.replace("m3", "m**3")
        unit = unit.replace("ft2", "ft**2")
        unit = unit.replace("km2", "km**2")
        return unit

    def normalize(self, quantity: Quantity) -> Tuple[Decimal, str]:
        """
        Normalize quantity to canonical unit

        Converts to canonical unit for the dimension and returns Decimal value.

        Args:
            quantity: Quantity to normalize

        Returns:
            (normalized_value, canonical_unit) tuple

        Raises:
            GLValidationError.UNIT_UNKNOWN: If unit not recognized

        Example:
            >>> ureg = UnitRegistry()
            >>> value, unit = ureg.normalize(Quantity(value=1.0, unit="tCO2e"))
            >>> value
            Decimal('1000')
            >>> unit
            'kgCO2e'
        """
        try:
            # Preprocess unit for pint compatibility
            pint_unit = self._preprocess_unit(quantity.unit)

            # Parse quantity with pint
            pint_qty = self.ureg.Quantity(quantity.value, pint_unit)

            # Get dimensionality
            dim = pint_qty.dimensionality

            # Determine canonical unit based on dimension
            canonical_unit = self._get_canonical_unit(quantity.unit, dim)

            # Convert to canonical
            if canonical_unit:
                converted = pint_qty.to(canonical_unit)
            else:
                # Already in canonical or dimensionless
                converted = pint_qty

            # Convert to Decimal for precision
            value_decimal = Decimal(str(converted.magnitude))

            return (value_decimal, canonical_unit or quantity.unit)

        except pint.UndefinedUnitError:
            raise GLValidationError(
                code="UNIT_UNKNOWN",
                message=f"Unit '{quantity.unit}' not recognized",
                hint=f"Add '{quantity.unit}' to allowlist or define custom unit",
            )
        except Exception as e:
            raise GLValidationError(
                code="UNIT_UNKNOWN", message=f"Failed to normalize {quantity}: {e}"
            )

    def _get_canonical_unit(self, unit: str, dimensionality) -> Optional[str]:
        """
        Get canonical unit for given dimensionality

        Args:
            unit: Original unit string
            dimensionality: pint dimensionality object

        Returns:
            Canonical unit string or None if dimensionless or complex
        """
        # Special case: CO2e units stay as kgCO2e (not just kg)
        if "CO2e" in unit or "co2e" in unit.lower():
            return "kgCO2e"

        # Map dimensionality to canonical unit
        dim_str = str(dimensionality)

        # Check for exact matches first (not partial!)
        if dim_str == "[mass]":
            return "kg"
        elif dim_str == "[energy]":
            return "kWh"
        elif dim_str == "[power]":
            return "kW"
        elif dim_str == "[length] ** 3":  # volume
            return "m**3"
        elif dim_str == "[length] ** 2":  # area
            return "m**2"
        elif dim_str == "[length]":
            return "m"
        elif dim_str == "[temperature]":
            return "K"
        elif dimensionality == self.ureg.dimensionless.dimensionality:
            return ""  # dimensionless

        # Complex/compound units (like kWh/m2, kgCO2e/m2) - keep original
        return None

    def is_allowed(self, unit: str) -> bool:
        """
        Check if unit is in allowlist

        Args:
            unit: Unit string to check

        Returns:
            True if unit is allowed

        Example:
            >>> ureg = UnitRegistry()
            >>> ureg.is_allowed("kWh")
            True
            >>> ureg.is_allowed("parsecs")
            False
        """
        # Case-insensitive check
        return unit in self.allowlist or unit.lower() in {
            u.lower() for u in self.allowlist
        }

    def same_quantity(self, a: Quantity, b: Quantity, tolerance: float = 1e-9) -> bool:
        """
        Check if two quantities are equal (after normalization)

        Compares quantities after converting to canonical units.
        Uses tolerance for float comparison.

        Args:
            a: First quantity
            b: Second quantity
            tolerance: Relative tolerance for comparison

        Returns:
            True if quantities are equal within tolerance

        Example:
            >>> ureg = UnitRegistry()
            >>> q1 = Quantity(value=1000, unit="g")
            >>> q2 = Quantity(value=1, unit="kg")
            >>> ureg.same_quantity(q1, q2)
            True
        """
        try:
            # Normalize both
            val_a, unit_a = self.normalize(a)
            val_b, unit_b = self.normalize(b)

            # Must have same canonical unit
            if unit_a != unit_b:
                return False

            # Compare values with tolerance
            if val_a == 0 and val_b == 0:
                return True

            max_val = max(abs(val_a), abs(val_b))
            diff = abs(val_a - val_b)
            tol = Decimal(str(tolerance))

            return diff <= tol * max(Decimal("1"), max_val)

        except Exception:
            return False

    def validate_dimension(self, quantity: Quantity, expected_dimension: str) -> None:
        """
        Validate that quantity has expected dimension

        Args:
            quantity: Quantity to validate
            expected_dimension: Expected dimension name (e.g., "mass", "energy")

        Raises:
            GLValidationError: If dimension doesn't match

        Example:
            >>> ureg = UnitRegistry()
            >>> ureg.validate_dimension(Quantity(value=100, unit="kWh"), "energy")  # OK
            >>> ureg.validate_dimension(Quantity(value=100, unit="kg"), "energy")  # Raises
        """
        try:
            pint_unit = self._preprocess_unit(quantity.unit)
            pint_qty = self.ureg.Quantity(quantity.value, pint_unit)
            dim = pint_qty.dimensionality
            dim_str = str(dim)

            dimension_map = {
                "mass": "[mass]",
                "energy": "[energy]",
                "power": "[power]",
                "volume": "[length] ** 3",
                "area": "[length] ** 2",
                "length": "[length]",
                "temperature": "[temperature]",
                "dimensionless": "",
            }

            expected_dim_str = dimension_map.get(expected_dimension)

            if expected_dim_str is None:
                raise GLValidationError(
                    code="UNIT_UNKNOWN",
                    message=f"Unknown dimension: {expected_dimension}",
                )

            # Must be exact match, not partial
            if dim_str != expected_dim_str:
                raise GLValidationError(
                    code="UNIT_UNKNOWN",
                    message=f"Dimension mismatch: expected {expected_dimension} ({expected_dim_str}), got {dim_str}",
                    hint=f"Quantity has wrong dimension. Expected {expected_dimension} but got {quantity.unit}",
                )

        except GLValidationError:
            raise
        except Exception as e:
            raise GLValidationError(
                code="UNIT_UNKNOWN", message=f"Failed to validate dimension: {e}"
            )

    def convert_to(self, quantity: Quantity, target_unit: str) -> Quantity:
        """
        Convert quantity to target unit

        Args:
            quantity: Quantity to convert
            target_unit: Target unit

        Returns:
            New Quantity in target unit

        Raises:
            GLValidationError: If conversion fails

        Example:
            >>> ureg = UnitRegistry()
            >>> q = Quantity(value=1, unit="tCO2e")
            >>> result = ureg.convert_to(q, "kgCO2e")
            >>> result.value
            1000.0
        """
        try:
            pint_unit = self._preprocess_unit(quantity.unit)
            pint_target = self._preprocess_unit(target_unit)
            pint_qty = self.ureg.Quantity(quantity.value, pint_unit)
            converted = pint_qty.to(pint_target)

            return Quantity(value=float(converted.magnitude), unit=target_unit)

        except Exception as e:
            raise GLValidationError(
                code="UNIT_UNKNOWN",
                message=f"Failed to convert {quantity.unit} to {target_unit}: {e}",
            )
