"""
Unit Converter module for the GreenLang Normalizer.

This module provides unit conversion capabilities built on Pint,
with extensions for sustainability-specific units and GHG emissions.

Example:
    >>> from gl_normalizer_core.converter import UnitConverter
    >>> from gl_normalizer_core.parser import Quantity
    >>> converter = UnitConverter()
    >>> quantity = Quantity(magnitude=1000, unit="kilogram")
    >>> result = converter.convert(quantity, "metric_ton")
    >>> print(result.converted_quantity.magnitude)
    1.0
"""

from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
from enum import Enum
import hashlib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
import structlog

try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

from gl_normalizer_core.errors import ConversionError, DimensionMismatchError
from gl_normalizer_core.parser import Quantity

logger = structlog.get_logger(__name__)


class ConversionMethod(str, Enum):
    """Methods used for unit conversion."""

    DIRECT = "direct"  # Direct conversion factor
    CHAIN = "chain"  # Chain of conversions
    CUSTOM = "custom"  # Custom conversion function


class ConversionResult(BaseModel):
    """
    Result of a unit conversion operation.

    Attributes:
        success: Whether conversion succeeded
        original_quantity: The input quantity
        converted_quantity: The converted quantity
        conversion_factor: Factor applied
        conversion_method: Method used for conversion
        precision_loss: Whether precision was lost
        warnings: Any warnings generated
        provenance_hash: SHA-256 hash for audit trail
        conversion_time_ms: Time taken for conversion
    """

    success: bool = Field(..., description="Whether conversion succeeded")
    original_quantity: Quantity = Field(..., description="Original quantity")
    converted_quantity: Optional[Quantity] = Field(None, description="Converted quantity")
    conversion_factor: Optional[float] = Field(None, description="Conversion factor applied")
    conversion_method: ConversionMethod = Field(
        default=ConversionMethod.DIRECT,
        description="Method used for conversion"
    )
    precision_loss: bool = Field(default=False, description="Whether precision was lost")
    warnings: List[str] = Field(default_factory=list, description="Conversion warnings")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    conversion_time_ms: float = Field(..., description="Conversion time in milliseconds")

    @classmethod
    def create_success(
        cls,
        original: Quantity,
        converted: Quantity,
        factor: float,
        method: ConversionMethod,
        conversion_time_ms: float,
        warnings: Optional[List[str]] = None,
    ) -> "ConversionResult":
        """Create a successful conversion result."""
        provenance_str = (
            f"{original.magnitude}|{original.unit}|"
            f"{converted.magnitude}|{converted.unit}|{factor}"
        )
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        # Check for precision loss
        precision_loss = False
        if converted.magnitude != 0:
            reverse = converted.magnitude / factor if factor != 0 else 0
            if abs(reverse - original.magnitude) / abs(original.magnitude) > 1e-10:
                precision_loss = True

        return cls(
            success=True,
            original_quantity=original,
            converted_quantity=converted,
            conversion_factor=factor,
            conversion_method=method,
            precision_loss=precision_loss,
            warnings=warnings or [],
            provenance_hash=provenance_hash,
            conversion_time_ms=conversion_time_ms,
        )

    @classmethod
    def create_failure(
        cls,
        original: Quantity,
        conversion_time_ms: float,
        warnings: Optional[List[str]] = None,
    ) -> "ConversionResult":
        """Create a failed conversion result."""
        provenance_hash = hashlib.sha256(
            f"FAILED|{original.magnitude}|{original.unit}".encode()
        ).hexdigest()
        return cls(
            success=False,
            original_quantity=original,
            converted_quantity=None,
            conversion_factor=None,
            warnings=warnings or [],
            provenance_hash=provenance_hash,
            conversion_time_ms=conversion_time_ms,
        )


class UnitConverter:
    """
    Unit converter for sustainability quantities.

    This class provides unit conversion capabilities using Pint,
    with extensions for GHG emissions and sustainability-specific units.

    Attributes:
        ureg: Pint UnitRegistry
        custom_conversions: Custom conversion factors

    Example:
        >>> converter = UnitConverter()
        >>> quantity = Quantity(magnitude=1000, unit="kilogram")
        >>> result = converter.convert(quantity, "metric_ton")
        >>> print(result.converted_quantity)
        Quantity(magnitude=1.0, unit='metric_ton')
    """

    # GreenLang-specific unit definitions for Pint
    GREENLANG_UNITS = """
        # GHG Emissions
        CO2_equivalent = [emissions]
        kg_CO2e = kilogram * CO2_equivalent
        t_CO2e = metric_ton * CO2_equivalent
        g_CO2e = gram * CO2_equivalent
        lb_CO2e = pound * CO2_equivalent

        # Energy intensity
        kWh_per_unit = kilowatt_hour / unit
        MJ_per_unit = megajoule / unit

        # Common sustainability units
        metric_ton = 1000 * kilogram = t = tonne
        barrel_oil_equivalent = 6.1178632 * gigajoule = boe
        therm = 105.4804 * megajoule
    """

    # Pre-defined conversion factors for common sustainability conversions
    CONVERSION_FACTORS: Dict[tuple, float] = {
        ("kilogram", "metric_ton"): 0.001,
        ("metric_ton", "kilogram"): 1000.0,
        ("gram", "kilogram"): 0.001,
        ("kilogram", "gram"): 1000.0,
        ("kilogram", "pound"): 2.20462,
        ("pound", "kilogram"): 0.453592,
        ("kilowatt_hour", "megajoule"): 3.6,
        ("megajoule", "kilowatt_hour"): 0.277778,
        ("megawatt_hour", "kilowatt_hour"): 1000.0,
        ("kilowatt_hour", "megawatt_hour"): 0.001,
        ("gigajoule", "megajoule"): 1000.0,
        ("megajoule", "gigajoule"): 0.001,
        ("liter", "gallon"): 0.264172,
        ("gallon", "liter"): 3.78541,
        ("cubic_meter", "liter"): 1000.0,
        ("liter", "cubic_meter"): 0.001,
        ("kilometer", "mile"): 0.621371,
        ("mile", "kilometer"): 1.60934,
    }

    def __init__(
        self,
        custom_conversions: Optional[Dict[tuple, float]] = None,
        use_pint: bool = True,
    ) -> None:
        """
        Initialize UnitConverter.

        Args:
            custom_conversions: Additional conversion factors
            use_pint: Whether to use Pint for conversions
        """
        self.use_pint = use_pint and PINT_AVAILABLE
        self.custom_conversions = {**self.CONVERSION_FACTORS}
        if custom_conversions:
            self.custom_conversions.update(custom_conversions)

        self.ureg: Optional[Any] = None
        if self.use_pint:
            self._init_pint_registry()

        logger.info(
            "UnitConverter initialized",
            use_pint=self.use_pint,
            custom_conversion_count=len(self.custom_conversions),
        )

    def _init_pint_registry(self) -> None:
        """Initialize Pint unit registry with GreenLang units."""
        if not PINT_AVAILABLE:
            logger.warning("Pint not available, using fallback conversions")
            return

        self.ureg = pint.UnitRegistry()

        # Add GreenLang-specific units
        try:
            # Add metric_ton if not present
            if "metric_ton" not in self.ureg:
                self.ureg.define("metric_ton = 1000 * kilogram = t = tonne")

            # Add emissions units
            self.ureg.define("[emissions] = [mass]")
            self.ureg.define("CO2_equivalent = kilogram")
            self.ureg.define("kg_CO2e = kilogram")
            self.ureg.define("t_CO2e = metric_ton")

            logger.debug("Pint registry initialized with GreenLang units")

        except pint.errors.RedefinitionError as e:
            logger.warning(f"Unit redefinition warning: {e}")

    def convert(
        self,
        quantity: Quantity,
        target_unit: str,
        policy_id: Optional[str] = None,
    ) -> ConversionResult:
        """
        Convert a quantity to a target unit.

        Args:
            quantity: Source quantity to convert
            target_unit: Target unit string
            policy_id: Optional policy ID for compliance checking

        Returns:
            ConversionResult containing converted quantity or error

        Raises:
            ConversionError: If conversion fails and no fallback available
            DimensionMismatchError: If units have incompatible dimensions
        """
        start_time = datetime.now()
        warnings: List[str] = []

        try:
            # Normalize target unit
            target_normalized = self._normalize_unit(target_unit)

            # Check if same unit (no conversion needed)
            if quantity.unit == target_normalized:
                conversion_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                return ConversionResult.create_success(
                    original=quantity,
                    converted=quantity,
                    factor=1.0,
                    method=ConversionMethod.DIRECT,
                    conversion_time_ms=conversion_time_ms,
                )

            # Try pre-defined conversions first
            factor = self._get_conversion_factor(quantity.unit, target_normalized)

            if factor is not None:
                converted_magnitude = quantity.magnitude * factor
                converted = Quantity(
                    magnitude=converted_magnitude,
                    unit=target_normalized,
                    original_unit=target_unit,
                    unit_system=quantity.unit_system,
                )
                conversion_time_ms = (datetime.now() - start_time).total_seconds() * 1000

                logger.debug(
                    "Converted quantity",
                    from_unit=quantity.unit,
                    to_unit=target_normalized,
                    factor=factor,
                    method="predefined",
                )

                return ConversionResult.create_success(
                    original=quantity,
                    converted=converted,
                    factor=factor,
                    method=ConversionMethod.DIRECT,
                    conversion_time_ms=conversion_time_ms,
                    warnings=warnings,
                )

            # Try Pint conversion
            if self.use_pint and self.ureg:
                try:
                    result = self._convert_with_pint(quantity, target_normalized)
                    if result:
                        conversion_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                        return ConversionResult.create_success(
                            original=quantity,
                            converted=result[0],
                            factor=result[1],
                            method=ConversionMethod.CHAIN,
                            conversion_time_ms=conversion_time_ms,
                            warnings=warnings,
                        )
                except Exception as e:
                    warnings.append(f"Pint conversion failed: {str(e)}")

            # Conversion not possible
            raise ConversionError(
                f"Cannot convert from '{quantity.unit}' to '{target_normalized}'",
                source_unit=quantity.unit,
                target_unit=target_normalized,
                hint="Ensure units are compatible and conversion factor is defined.",
            )

        except (ConversionError, DimensionMismatchError):
            raise
        except Exception as e:
            conversion_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error("Conversion failed", error=str(e), exc_info=True)
            return ConversionResult.create_failure(
                original=quantity,
                conversion_time_ms=conversion_time_ms,
                warnings=warnings + [str(e)],
            )

    def _normalize_unit(self, unit: str) -> str:
        """Normalize a unit string."""
        # Replace spaces with underscores and handle common variants
        return unit.strip().replace(" ", "_").replace("-", "_")

    def _get_conversion_factor(
        self,
        source_unit: str,
        target_unit: str,
    ) -> Optional[float]:
        """
        Get conversion factor between two units.

        Args:
            source_unit: Source unit
            target_unit: Target unit

        Returns:
            Conversion factor or None if not found
        """
        key = (source_unit, target_unit)
        if key in self.custom_conversions:
            return self.custom_conversions[key]

        # Try case-insensitive match
        source_lower = source_unit.lower()
        target_lower = target_unit.lower()
        for (s, t), factor in self.custom_conversions.items():
            if s.lower() == source_lower and t.lower() == target_lower:
                return factor

        return None

    def _convert_with_pint(
        self,
        quantity: Quantity,
        target_unit: str,
    ) -> Optional[tuple]:
        """
        Convert using Pint library.

        Args:
            quantity: Source quantity
            target_unit: Target unit

        Returns:
            Tuple of (converted Quantity, factor) or None
        """
        if not self.ureg:
            return None

        try:
            # Create Pint quantity
            pint_qty = self.ureg.Quantity(quantity.magnitude, quantity.unit)

            # Convert to target
            converted = pint_qty.to(target_unit)

            # Calculate factor
            factor = converted.magnitude / quantity.magnitude if quantity.magnitude != 0 else 0

            result_quantity = Quantity(
                magnitude=float(converted.magnitude),
                unit=target_unit,
                original_unit=target_unit,
                unit_system=quantity.unit_system,
            )

            return (result_quantity, factor)

        except Exception as e:
            logger.debug(f"Pint conversion failed: {e}")
            return None

    def add_conversion(
        self,
        source_unit: str,
        target_unit: str,
        factor: float,
    ) -> None:
        """
        Add a custom conversion factor.

        Args:
            source_unit: Source unit
            target_unit: Target unit
            factor: Conversion factor
        """
        self.custom_conversions[(source_unit, target_unit)] = factor
        # Also add reverse conversion
        if factor != 0:
            self.custom_conversions[(target_unit, source_unit)] = 1.0 / factor
        logger.debug(
            "Added conversion",
            source=source_unit,
            target=target_unit,
            factor=factor,
        )

    def get_compatible_units(self, unit: str) -> List[str]:
        """
        Get list of units compatible with the given unit.

        Args:
            unit: Unit to find compatible units for

        Returns:
            List of compatible unit strings
        """
        compatible = set()
        for (source, target) in self.custom_conversions.keys():
            if source == unit:
                compatible.add(target)
            elif target == unit:
                compatible.add(source)
        return sorted(compatible)


__all__ = [
    "UnitConverter",
    "ConversionResult",
    "ConversionMethod",
]
