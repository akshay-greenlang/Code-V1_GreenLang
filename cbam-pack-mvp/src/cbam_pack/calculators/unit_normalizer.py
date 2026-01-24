"""
Unit Normalizer

Handles unit conversions with full audit trail for CBAM calculations.
Uses Decimal for precision.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from cbam_pack.errors import UnitConversionError
from cbam_pack.models import Unit


@dataclass
class ConversionRecord:
    """Record of a unit conversion for audit trail."""
    original_value: Decimal
    original_unit: str
    converted_value: Decimal
    converted_unit: str
    conversion_factor: Decimal


class UnitNormalizer:
    """
    Handles unit conversions for CBAM calculations.

    All quantities are normalized to tonnes for emissions calculations.
    """

    # Conversion factors to tonnes
    CONVERSION_TO_TONNES: dict[str, Decimal] = {
        "kg": Decimal("0.001"),  # 1 kg = 0.001 tonnes
        "tonnes": Decimal("1"),  # 1 tonne = 1 tonne
    }

    def __init__(self):
        self._conversion_log: list[ConversionRecord] = []

    @property
    def conversion_log(self) -> list[ConversionRecord]:
        """Get the log of all conversions performed."""
        return self._conversion_log.copy()

    def clear_log(self) -> None:
        """Clear the conversion log."""
        self._conversion_log.clear()

    def normalize_to_tonnes(
        self,
        value: Decimal,
        from_unit: str,
        line_id: Optional[str] = None,
    ) -> Decimal:
        """
        Convert a quantity to tonnes.

        Args:
            value: The quantity to convert
            from_unit: The source unit (kg or tonnes)
            line_id: Optional line ID for error reporting

        Returns:
            Quantity in tonnes

        Raises:
            UnitConversionError: If unit is not supported
        """
        from_unit_lower = from_unit.lower().strip()

        if from_unit_lower not in self.CONVERSION_TO_TONNES:
            raise UnitConversionError(from_unit, "tonnes")

        conversion_factor = self.CONVERSION_TO_TONNES[from_unit_lower]
        converted_value = value * conversion_factor

        # Round to 6 decimal places for intermediate values
        converted_value = converted_value.quantize(Decimal("0.000001"))

        # Log the conversion
        record = ConversionRecord(
            original_value=value,
            original_unit=from_unit_lower,
            converted_value=converted_value,
            converted_unit="tonnes",
            conversion_factor=conversion_factor,
        )
        self._conversion_log.append(record)

        return converted_value

    def convert(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert between supported units.

        Args:
            value: The quantity to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted quantity

        Raises:
            UnitConversionError: If conversion is not supported
        """
        from_unit_lower = from_unit.lower().strip()
        to_unit_lower = to_unit.lower().strip()

        if from_unit_lower not in self.CONVERSION_TO_TONNES:
            raise UnitConversionError(from_unit, to_unit)

        if to_unit_lower not in self.CONVERSION_TO_TONNES:
            raise UnitConversionError(from_unit, to_unit)

        # Convert to tonnes first, then to target unit
        to_tonnes = self.CONVERSION_TO_TONNES[from_unit_lower]
        from_tonnes = Decimal("1") / self.CONVERSION_TO_TONNES[to_unit_lower]

        conversion_factor = to_tonnes * from_tonnes
        converted_value = value * conversion_factor

        # Round appropriately
        converted_value = converted_value.quantize(Decimal("0.000001"))

        # Log the conversion
        record = ConversionRecord(
            original_value=value,
            original_unit=from_unit_lower,
            converted_value=converted_value,
            converted_unit=to_unit_lower,
            conversion_factor=conversion_factor,
        )
        self._conversion_log.append(record)

        return converted_value

    @staticmethod
    def validate_unit(unit: str) -> bool:
        """
        Check if a unit is supported.

        Args:
            unit: Unit string to validate

        Returns:
            True if supported, False otherwise
        """
        return unit.lower().strip() in UnitNormalizer.CONVERSION_TO_TONNES
