"""
Unit Conversion Engine for GL-FOUND-X-003 GreenLang Normalizer.

This module implements the core ConversionEngine class that provides
production-grade unit conversion capabilities for sustainability reporting.

Key features:
- Support for GL Canonical Units (MJ, kg, kgCO2e, m3, kPa_abs, degC, K)
- GWP version handling (AR4, AR5, AR6) for CO2e conversions
- Basis handling (LHV/HHV for energy, wet/dry for mass)
- Reference conditions for gas volumes (temperature, pressure)
- Gauge-to-absolute pressure conversion with atmospheric reference
- Pint library integration for base unit conversions
- Complete audit trail with provenance tracking

All conversion functions are pure and deterministic with no side effects.

Example:
    >>> from gl_normalizer_core.conversion.engine import ConversionEngine
    >>> from gl_normalizer_core.conversion.contexts import ConversionContext
    >>> engine = ConversionEngine()
    >>> context = ConversionContext(gwp_version="AR6", basis="LHV")
    >>> result = engine.convert(100.0, "kWh", "MJ", context)
    >>> print(result.converted_value)
    360.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.errors import ConversionError, DimensionMismatchError

from gl_normalizer_core.conversion.contexts import (
    ConversionContext,
    GWPVersion,
    EnergyBasis,
    PressureMode,
    gauge_to_absolute,
    celsius_to_kelvin,
    kelvin_to_celsius,
    DEFAULT_ATMOSPHERIC_PRESSURE_KPA,
)
from gl_normalizer_core.conversion.factors import (
    ConversionFactor,
    ConversionFactorRegistry,
    ConversionType,
    GWPFactor,
    create_inverse_factor,
)
from gl_normalizer_core.conversion.validators import (
    ValidationResult,
    validate_conversion_path,
    validate_context_for_conversion,
    validate_numeric_value,
    validate_reference_conditions,
)

# Try to import Pint for advanced unit handling
try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False


# GL Canonical Units by dimension
GL_CANONICAL_UNITS: Dict[str, str] = {
    "energy": "MJ",
    "mass": "kg",
    "emissions": "kgCO2e",
    "volume": "m3",
    "pressure": "kPa_abs",
    "temperature": "degC",
    "temperature_absolute": "K",
}

# Unit dimension mapping
UNIT_DIMENSIONS: Dict[str, str] = {
    # Energy units
    "MJ": "energy", "GJ": "energy", "TJ": "energy", "J": "energy", "kJ": "energy",
    "kWh": "energy", "MWh": "energy", "GWh": "energy",
    "BTU": "energy", "MMBTU": "energy", "therm": "energy",
    # Mass units
    "kg": "mass", "g": "mass", "mg": "mass", "t": "mass", "tonne": "mass",
    "metric_ton": "mass", "Mt": "mass", "kt": "mass",
    "lb": "mass", "oz": "mass", "short_ton": "mass", "long_ton": "mass",
    # Emissions units
    "kgCO2e": "emissions", "tCO2e": "emissions", "gCO2e": "emissions",
    "kg_CO2e": "emissions", "t_CO2e": "emissions", "g_CO2e": "emissions",
    # Volume units
    "m3": "volume", "L": "volume", "mL": "volume", "kL": "volume",
    "gal": "volume", "gal_US": "volume", "gal_UK": "volume",
    "bbl": "volume", "ft3": "volume", "liter": "volume", "litre": "volume",
    "Nm3": "volume", "Sm3": "volume",
    # Pressure units
    "kPa_abs": "pressure", "kPa": "pressure", "Pa": "pressure", "MPa": "pressure",
    "bar": "pressure", "mbar": "pressure", "atm": "pressure",
    "psi": "pressure", "psia": "pressure", "psig": "pressure",
    "mmHg": "pressure", "Torr": "pressure", "inHg": "pressure",
    # Temperature units
    "degC": "temperature", "degF": "temperature", "K": "temperature_absolute",
    "C": "temperature", "F": "temperature",
}


@dataclass(frozen=True)
class ConversionStep:
    """
    Immutable record of a single conversion step.

    Attributes:
        from_unit: Source unit for this step.
        to_unit: Target unit for this step.
        factor: Conversion factor applied.
        method: Conversion method used.
        input_value: Value before this step.
        output_value: Value after this step.
        factor_version: Version of the factor used.
    """

    from_unit: str
    to_unit: str
    factor: float
    method: str
    input_value: float
    output_value: float
    factor_version: str = "2026.01.0"
    offset: Optional[float] = None


@dataclass
class ConversionResult:
    """
    Result of a unit conversion operation.

    Attributes:
        success: Whether conversion succeeded.
        original_value: Input value.
        original_unit: Input unit.
        converted_value: Output value (None if failed).
        converted_unit: Output unit.
        conversion_steps: List of conversion steps applied.
        total_factor: Combined conversion factor.
        provenance_hash: SHA-256 hash for audit trail.
        conversion_time_ms: Processing time in milliseconds.
        warnings: List of warning messages.
        error_code: Error code if conversion failed.
        error_message: Error message if conversion failed.
    """

    success: bool
    original_value: float
    original_unit: str
    converted_value: Optional[float]
    converted_unit: str
    conversion_steps: List[ConversionStep] = field(default_factory=list)
    total_factor: Optional[float] = None
    provenance_hash: str = ""
    conversion_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.original_value}|{self.original_unit}|"
                f"{self.converted_value}|{self.converted_unit}|"
                f"{self.total_factor}"
            )
            object.__setattr__(
                self,
                "provenance_hash",
                hashlib.sha256(provenance_str.encode()).hexdigest(),
            )

    @classmethod
    def create_success(
        cls,
        original_value: float,
        original_unit: str,
        converted_value: float,
        converted_unit: str,
        steps: List[ConversionStep],
        total_factor: float,
        conversion_time_ms: float,
        warnings: Optional[List[str]] = None,
    ) -> "ConversionResult":
        """Create a successful conversion result."""
        return cls(
            success=True,
            original_value=original_value,
            original_unit=original_unit,
            converted_value=converted_value,
            converted_unit=converted_unit,
            conversion_steps=steps,
            total_factor=total_factor,
            conversion_time_ms=conversion_time_ms,
            warnings=warnings or [],
        )

    @classmethod
    def create_failure(
        cls,
        original_value: float,
        original_unit: str,
        target_unit: str,
        error_code: str,
        error_message: str,
        conversion_time_ms: float,
        warnings: Optional[List[str]] = None,
    ) -> "ConversionResult":
        """Create a failed conversion result."""
        return cls(
            success=False,
            original_value=original_value,
            original_unit=original_unit,
            converted_value=None,
            converted_unit=target_unit,
            error_code=error_code,
            error_message=error_message,
            conversion_time_ms=conversion_time_ms,
            warnings=warnings or [],
        )


class ConversionEngine:
    """
    Production-grade unit conversion engine for sustainability reporting.

    This class provides comprehensive unit conversion capabilities with:
    - Support for all GL Canonical Units
    - Context-aware conversions (GWP, basis, reference conditions)
    - Complete audit trail with provenance tracking
    - Pint library integration for extensibility
    - Zero-hallucination deterministic calculations

    Example:
        >>> engine = ConversionEngine()
        >>> context = ConversionContext(gwp_version="AR6")
        >>> result = engine.convert(100.0, "kWh", "MJ", context)
        >>> print(result.converted_value)
        360.0
    """

    def __init__(
        self,
        use_pint: bool = True,
        custom_factors: Optional[Dict[Tuple[str, str], ConversionFactor]] = None,
    ) -> None:
        """
        Initialize the ConversionEngine.

        Args:
            use_pint: Whether to use Pint library for advanced conversions.
            custom_factors: Optional dictionary of custom conversion factors.
        """
        self._registry = ConversionFactorRegistry()
        self._use_pint = use_pint and PINT_AVAILABLE
        self._ureg: Optional[Any] = None

        # Register custom factors if provided
        if custom_factors:
            for factor in custom_factors.values():
                self._registry.register_factor(factor)

        # Initialize Pint if available and requested
        if self._use_pint:
            self._init_pint()

    def _init_pint(self) -> None:
        """Initialize Pint unit registry with GreenLang extensions."""
        if not PINT_AVAILABLE:
            return

        self._ureg = pint.UnitRegistry()

        # Define custom units if not present
        try:
            if "metric_ton" not in self._ureg:
                self._ureg.define("metric_ton = 1000 * kilogram = t = tonne")
        except pint.errors.RedefinitionError:
            pass

    @property
    def version(self) -> str:
        """Get the conversion engine version."""
        return self._registry.version

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        context: Optional[ConversionContext] = None,
    ) -> ConversionResult:
        """
        Convert a value from one unit to another.

        This is the main conversion method that handles all unit conversion
        scenarios, including context-dependent conversions.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit string.
            to_unit: Target unit string.
            context: Optional conversion context with GWP, basis, etc.

        Returns:
            ConversionResult with converted value and audit trail.

        Example:
            >>> engine = ConversionEngine()
            >>> result = engine.convert(1000.0, "kWh", "MJ")
            >>> print(result.converted_value)
            3600.0
        """
        start_time = datetime.now()
        warnings: List[str] = []
        steps: List[ConversionStep] = []

        try:
            # Validate numeric value
            value_validation = validate_numeric_value(value)
            if not value_validation.is_valid:
                return ConversionResult.create_failure(
                    original_value=value,
                    original_unit=from_unit,
                    target_unit=to_unit,
                    error_code=value_validation.error_code or GLNORMErrorCode.E304_PRECISION_OVERFLOW.value,
                    error_message=value_validation.error_message or "Invalid numeric value",
                    conversion_time_ms=self._elapsed_ms(start_time),
                )

            # Trivial case: same unit
            if from_unit == to_unit:
                return ConversionResult.create_success(
                    original_value=value,
                    original_unit=from_unit,
                    converted_value=value,
                    converted_unit=to_unit,
                    steps=[],
                    total_factor=1.0,
                    conversion_time_ms=self._elapsed_ms(start_time),
                )

            # Validate conversion path
            path_validation = validate_conversion_path(from_unit, to_unit, context)
            if not path_validation.is_valid:
                return ConversionResult.create_failure(
                    original_value=value,
                    original_unit=from_unit,
                    target_unit=to_unit,
                    error_code=path_validation.error_code or GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED.value,
                    error_message=path_validation.error_message or "Conversion not supported",
                    conversion_time_ms=self._elapsed_ms(start_time),
                    warnings=list(path_validation.warnings),
                )
            if path_validation.warnings:
                warnings.extend(path_validation.warnings)

            # Normalize units
            from_unit_normalized = self._normalize_unit(from_unit)
            to_unit_normalized = self._normalize_unit(to_unit)

            # Check for pressure mode conversion (gauge -> absolute)
            if context and context.pressure_mode == PressureMode.GAUGE.value:
                if self._is_pressure_unit(from_unit_normalized):
                    # Convert gauge to absolute first
                    value = gauge_to_absolute(value, context.atmospheric_pressure)
                    steps.append(ConversionStep(
                        from_unit=f"{from_unit}_gauge",
                        to_unit=f"{from_unit}_abs",
                        factor=1.0,
                        method="gauge_to_absolute",
                        input_value=value - context.atmospheric_pressure,
                        output_value=value,
                    ))
                    warnings.append(
                        f"Converted gauge pressure to absolute using atmospheric pressure "
                        f"{context.atmospheric_pressure} kPa"
                    )

            # Try registry conversion first
            result = self._convert_with_registry(
                value, from_unit_normalized, to_unit_normalized, context, steps, warnings
            )
            if result is not None:
                return ConversionResult.create_success(
                    original_value=value,
                    original_unit=from_unit,
                    converted_value=result[0],
                    converted_unit=to_unit,
                    steps=result[1],
                    total_factor=result[2],
                    conversion_time_ms=self._elapsed_ms(start_time),
                    warnings=warnings,
                )

            # Try Pint conversion
            if self._use_pint and self._ureg:
                result = self._convert_with_pint(
                    value, from_unit_normalized, to_unit_normalized, steps, warnings
                )
                if result is not None:
                    return ConversionResult.create_success(
                        original_value=value,
                        original_unit=from_unit,
                        converted_value=result[0],
                        converted_unit=to_unit,
                        steps=result[1],
                        total_factor=result[2],
                        conversion_time_ms=self._elapsed_ms(start_time),
                        warnings=warnings,
                    )

            # Conversion not supported
            return ConversionResult.create_failure(
                original_value=value,
                original_unit=from_unit,
                target_unit=to_unit,
                error_code=GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED.value,
                error_message=f"No conversion path found from '{from_unit}' to '{to_unit}'",
                conversion_time_ms=self._elapsed_ms(start_time),
                warnings=warnings,
            )

        except Exception as e:
            return ConversionResult.create_failure(
                original_value=value,
                original_unit=from_unit,
                target_unit=to_unit,
                error_code=GLNORMErrorCode.E904_INTERNAL_ERROR.value,
                error_message=f"Unexpected error during conversion: {str(e)}",
                conversion_time_ms=self._elapsed_ms(start_time),
                warnings=warnings,
            )

    def convert_to_canonical(
        self,
        value: float,
        from_unit: str,
        dimension: str,
        context: Optional[ConversionContext] = None,
    ) -> ConversionResult:
        """
        Convert a value to the GL canonical unit for a dimension.

        Args:
            value: Numeric value to convert.
            from_unit: Source unit string.
            dimension: Target dimension (energy, mass, emissions, etc.).
            context: Optional conversion context.

        Returns:
            ConversionResult with value in canonical unit.

        Example:
            >>> engine = ConversionEngine()
            >>> result = engine.convert_to_canonical(1000.0, "kWh", "energy")
            >>> print(result.converted_unit)
            MJ
        """
        canonical_unit = GL_CANONICAL_UNITS.get(dimension)
        if canonical_unit is None:
            return ConversionResult.create_failure(
                original_value=value,
                original_unit=from_unit,
                target_unit="unknown",
                error_code=GLNORMErrorCode.E201_DIMENSION_UNKNOWN.value,
                error_message=f"Unknown dimension '{dimension}'. Valid dimensions: {list(GL_CANONICAL_UNITS.keys())}",
                conversion_time_ms=0.0,
            )

        return self.convert(value, from_unit, canonical_unit, context)

    def convert_ghg_to_co2e(
        self,
        value: float,
        gas: str,
        mass_unit: str,
        gwp_version: str,
    ) -> ConversionResult:
        """
        Convert a greenhouse gas mass to CO2 equivalent.

        Args:
            value: Mass of the gas.
            gas: Gas identifier (e.g., CH4, N2O, SF6).
            mass_unit: Unit of the input mass.
            gwp_version: IPCC assessment report version (AR4, AR5, AR6).

        Returns:
            ConversionResult with value in kgCO2e.

        Example:
            >>> engine = ConversionEngine()
            >>> result = engine.convert_ghg_to_co2e(1.0, "CH4", "kg", "AR6")
            >>> print(result.converted_value)
            27.9
        """
        start_time = datetime.now()
        warnings: List[str] = []
        steps: List[ConversionStep] = []

        try:
            # Get GWP factor
            gwp_factor = self._registry.get_gwp_factor(gas, gwp_version)
            if gwp_factor is None:
                return ConversionResult.create_failure(
                    original_value=value,
                    original_unit=f"{mass_unit} {gas}",
                    target_unit="kgCO2e",
                    error_code=GLNORMErrorCode.E303_CONVERSION_FACTOR_MISSING.value,
                    error_message=f"No GWP factor found for gas '{gas}' with version '{gwp_version}'",
                    conversion_time_ms=self._elapsed_ms(start_time),
                )

            # Convert mass to kg first if needed
            mass_in_kg = value
            if mass_unit != "kg":
                mass_conversion = self.convert(value, mass_unit, "kg")
                if not mass_conversion.success:
                    return ConversionResult.create_failure(
                        original_value=value,
                        original_unit=f"{mass_unit} {gas}",
                        target_unit="kgCO2e",
                        error_code=mass_conversion.error_code or GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED.value,
                        error_message=f"Failed to convert {mass_unit} to kg: {mass_conversion.error_message}",
                        conversion_time_ms=self._elapsed_ms(start_time),
                    )
                mass_in_kg = mass_conversion.converted_value or 0.0
                steps.extend(mass_conversion.conversion_steps)

            # Apply GWP factor
            co2e_value = mass_in_kg * gwp_factor.value
            steps.append(ConversionStep(
                from_unit=f"kg {gas}",
                to_unit="kgCO2e",
                factor=gwp_factor.value,
                method="gwp_multiply",
                input_value=mass_in_kg,
                output_value=co2e_value,
                factor_version=f"IPCC {gwp_version}",
            ))

            return ConversionResult.create_success(
                original_value=value,
                original_unit=f"{mass_unit} {gas}",
                converted_value=co2e_value,
                converted_unit="kgCO2e",
                steps=steps,
                total_factor=gwp_factor.value * (mass_in_kg / value if value != 0 else 1.0),
                conversion_time_ms=self._elapsed_ms(start_time),
                warnings=warnings,
            )

        except Exception as e:
            return ConversionResult.create_failure(
                original_value=value,
                original_unit=f"{mass_unit} {gas}",
                target_unit="kgCO2e",
                error_code=GLNORMErrorCode.E904_INTERNAL_ERROR.value,
                error_message=f"Unexpected error during GHG conversion: {str(e)}",
                conversion_time_ms=self._elapsed_ms(start_time),
            )

    def convert_with_basis(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        from_basis: str,
        to_basis: str,
        fuel: str,
    ) -> ConversionResult:
        """
        Convert energy values between different bases (LHV/HHV).

        Args:
            value: Energy value to convert.
            from_unit: Source energy unit.
            to_unit: Target energy unit.
            from_basis: Source basis (LHV or HHV).
            to_basis: Target basis (LHV or HHV).
            fuel: Fuel type for LHV/HHV ratio.

        Returns:
            ConversionResult with basis-adjusted value.

        Example:
            >>> engine = ConversionEngine()
            >>> result = engine.convert_with_basis(100.0, "MJ", "MJ", "LHV", "HHV", "natural_gas")
            >>> print(result.converted_value)
            110.9
        """
        start_time = datetime.now()
        warnings: List[str] = []
        steps: List[ConversionStep] = []

        try:
            # Get LHV/HHV ratio for the fuel
            ratio = self._registry.get_lhv_hhv_ratio(fuel)
            if ratio is None:
                return ConversionResult.create_failure(
                    original_value=value,
                    original_unit=f"{from_unit}_{from_basis}",
                    target_unit=f"{to_unit}_{to_basis}",
                    error_code=GLNORMErrorCode.E303_CONVERSION_FACTOR_MISSING.value,
                    error_message=f"No LHV/HHV ratio found for fuel '{fuel}'",
                    conversion_time_ms=self._elapsed_ms(start_time),
                )

            current_value = value

            # Apply basis conversion if needed
            if from_basis != to_basis:
                if from_basis == EnergyBasis.LHV.value and to_basis == EnergyBasis.HHV.value:
                    # LHV to HHV: multiply by ratio
                    basis_factor = ratio
                elif from_basis == EnergyBasis.HHV.value and to_basis == EnergyBasis.LHV.value:
                    # HHV to LHV: divide by ratio
                    basis_factor = 1.0 / ratio
                else:
                    return ConversionResult.create_failure(
                        original_value=value,
                        original_unit=f"{from_unit}_{from_basis}",
                        target_unit=f"{to_unit}_{to_basis}",
                        error_code=GLNORMErrorCode.E306_BASIS_MISSING.value,
                        error_message=f"Invalid basis conversion: {from_basis} to {to_basis}",
                        conversion_time_ms=self._elapsed_ms(start_time),
                    )

                new_value = current_value * basis_factor
                steps.append(ConversionStep(
                    from_unit=f"{from_unit}_{from_basis}",
                    to_unit=f"{from_unit}_{to_basis}",
                    factor=basis_factor,
                    method="basis_conversion",
                    input_value=current_value,
                    output_value=new_value,
                ))
                current_value = new_value

            # Now convert units if needed
            if from_unit != to_unit:
                unit_conversion = self.convert(current_value, from_unit, to_unit)
                if not unit_conversion.success:
                    return ConversionResult.create_failure(
                        original_value=value,
                        original_unit=f"{from_unit}_{from_basis}",
                        target_unit=f"{to_unit}_{to_basis}",
                        error_code=unit_conversion.error_code or GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED.value,
                        error_message=unit_conversion.error_message,
                        conversion_time_ms=self._elapsed_ms(start_time),
                    )
                steps.extend(unit_conversion.conversion_steps)
                current_value = unit_conversion.converted_value or current_value

            # Calculate total factor
            total_factor = current_value / value if value != 0 else 1.0

            return ConversionResult.create_success(
                original_value=value,
                original_unit=f"{from_unit}_{from_basis}",
                converted_value=current_value,
                converted_unit=f"{to_unit}_{to_basis}",
                steps=steps,
                total_factor=total_factor,
                conversion_time_ms=self._elapsed_ms(start_time),
                warnings=warnings,
            )

        except Exception as e:
            return ConversionResult.create_failure(
                original_value=value,
                original_unit=f"{from_unit}_{from_basis}",
                target_unit=f"{to_unit}_{to_basis}",
                error_code=GLNORMErrorCode.E904_INTERNAL_ERROR.value,
                error_message=f"Unexpected error during basis conversion: {str(e)}",
                conversion_time_ms=self._elapsed_ms(start_time),
            )

    def convert_gas_volume(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        from_conditions: Tuple[float, float],
        to_conditions: Tuple[float, float],
    ) -> ConversionResult:
        """
        Convert gas volume between different reference conditions.

        Uses the ideal gas law to adjust for temperature and pressure differences.

        Args:
            value: Volume value to convert.
            from_unit: Source volume unit.
            to_unit: Target volume unit.
            from_conditions: Source (temperature_C, pressure_kPa).
            to_conditions: Target (temperature_C, pressure_kPa).

        Returns:
            ConversionResult with adjusted volume.

        Example:
            >>> engine = ConversionEngine()
            >>> # Convert 100 m3 at STP to NTP
            >>> result = engine.convert_gas_volume(
            ...     100.0, "m3", "m3",
            ...     (0.0, 101.325),    # STP
            ...     (20.0, 101.325)    # NTP
            ... )
            >>> print(result.converted_value)  # Higher due to expansion
        """
        start_time = datetime.now()
        warnings: List[str] = []
        steps: List[ConversionStep] = []

        try:
            # Validate reference conditions
            from_temp_c, from_press_kpa = from_conditions
            to_temp_c, to_press_kpa = to_conditions

            from_validation = validate_reference_conditions(from_temp_c, from_press_kpa)
            if not from_validation.is_valid:
                return ConversionResult.create_failure(
                    original_value=value,
                    original_unit=from_unit,
                    target_unit=to_unit,
                    error_code=from_validation.error_code or GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS.value,
                    error_message=f"Invalid source conditions: {from_validation.error_message}",
                    conversion_time_ms=self._elapsed_ms(start_time),
                )

            to_validation = validate_reference_conditions(to_temp_c, to_press_kpa)
            if not to_validation.is_valid:
                return ConversionResult.create_failure(
                    original_value=value,
                    original_unit=from_unit,
                    target_unit=to_unit,
                    error_code=to_validation.error_code or GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS.value,
                    error_message=f"Invalid target conditions: {to_validation.error_message}",
                    conversion_time_ms=self._elapsed_ms(start_time),
                )

            # Convert temperatures to Kelvin for ideal gas law
            from_temp_k = celsius_to_kelvin(from_temp_c)
            to_temp_k = celsius_to_kelvin(to_temp_c)

            # Apply ideal gas law: V2/V1 = (T2/T1) * (P1/P2)
            # V2 = V1 * (T2/T1) * (P1/P2)
            condition_factor = (to_temp_k / from_temp_k) * (from_press_kpa / to_press_kpa)
            adjusted_value = value * condition_factor

            steps.append(ConversionStep(
                from_unit=f"{from_unit}@{from_temp_c}C,{from_press_kpa}kPa",
                to_unit=f"{from_unit}@{to_temp_c}C,{to_press_kpa}kPa",
                factor=condition_factor,
                method="ideal_gas_law",
                input_value=value,
                output_value=adjusted_value,
            ))

            current_value = adjusted_value

            # Convert units if needed
            if from_unit != to_unit:
                unit_conversion = self.convert(current_value, from_unit, to_unit)
                if not unit_conversion.success:
                    return ConversionResult.create_failure(
                        original_value=value,
                        original_unit=from_unit,
                        target_unit=to_unit,
                        error_code=unit_conversion.error_code or GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED.value,
                        error_message=unit_conversion.error_message,
                        conversion_time_ms=self._elapsed_ms(start_time),
                    )
                steps.extend(unit_conversion.conversion_steps)
                current_value = unit_conversion.converted_value or current_value

            total_factor = current_value / value if value != 0 else 1.0

            return ConversionResult.create_success(
                original_value=value,
                original_unit=from_unit,
                converted_value=current_value,
                converted_unit=to_unit,
                steps=steps,
                total_factor=total_factor,
                conversion_time_ms=self._elapsed_ms(start_time),
                warnings=warnings,
            )

        except Exception as e:
            return ConversionResult.create_failure(
                original_value=value,
                original_unit=from_unit,
                target_unit=to_unit,
                error_code=GLNORMErrorCode.E904_INTERNAL_ERROR.value,
                error_message=f"Unexpected error during gas volume conversion: {str(e)}",
                conversion_time_ms=self._elapsed_ms(start_time),
            )

    def get_dimension(self, unit: str) -> Optional[str]:
        """
        Get the physical dimension for a unit.

        Args:
            unit: Unit string.

        Returns:
            Dimension string or None if unknown.
        """
        normalized = self._normalize_unit(unit)
        return UNIT_DIMENSIONS.get(normalized)

    def get_canonical_unit(self, dimension: str) -> Optional[str]:
        """
        Get the GL canonical unit for a dimension.

        Args:
            dimension: Physical dimension.

        Returns:
            Canonical unit string or None if unknown.
        """
        return GL_CANONICAL_UNITS.get(dimension)

    def get_supported_dimensions(self) -> List[str]:
        """Get list of supported dimensions."""
        return list(GL_CANONICAL_UNITS.keys())

    def is_conversion_supported(
        self,
        from_unit: str,
        to_unit: str,
        context: Optional[ConversionContext] = None,
    ) -> bool:
        """
        Check if a conversion is supported.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.
            context: Optional conversion context.

        Returns:
            True if conversion is supported.
        """
        if from_unit == to_unit:
            return True

        from_normalized = self._normalize_unit(from_unit)
        to_normalized = self._normalize_unit(to_unit)

        # Check registry
        if self._registry.has_factor(from_normalized, to_normalized):
            return True

        # Check Pint if available
        if self._use_pint and self._ureg:
            try:
                pint_from = self._ureg.Quantity(1.0, from_normalized)
                pint_from.to(to_normalized)
                return True
            except Exception:
                pass

        return False

    def _convert_with_registry(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        context: Optional[ConversionContext],
        steps: List[ConversionStep],
        warnings: List[str],
    ) -> Optional[Tuple[float, List[ConversionStep], float]]:
        """Convert using the factor registry."""
        # Try direct factor lookup
        factor = self._registry.get_factor_with_aliases(from_unit, to_unit)

        if factor is not None:
            # Check for deprecation
            if factor.is_deprecated:
                warning = self._registry.get_deprecation_warning(from_unit, to_unit)
                if warning:
                    warnings.append(warning)

            # Apply the factor
            converted_value = factor.apply(value)

            steps.append(ConversionStep(
                from_unit=from_unit,
                to_unit=to_unit,
                factor=factor.value,
                method=factor.conversion_type.value,
                input_value=value,
                output_value=converted_value,
                factor_version=factor.version,
                offset=factor.offset if factor.offset != 0 else None,
            ))

            return (converted_value, steps, factor.value)

        # Try finding a conversion path through an intermediate unit
        from_dimension = self.get_dimension(from_unit)
        to_dimension = self.get_dimension(to_unit)

        if from_dimension and from_dimension == to_dimension:
            canonical = GL_CANONICAL_UNITS.get(from_dimension)
            if canonical and canonical != from_unit and canonical != to_unit:
                # Try: from_unit -> canonical -> to_unit
                factor1 = self._registry.get_factor_with_aliases(from_unit, canonical)
                factor2 = self._registry.get_factor_with_aliases(canonical, to_unit)

                if factor1 and factor2:
                    intermediate = factor1.apply(value)
                    final_value = factor2.apply(intermediate)

                    steps.append(ConversionStep(
                        from_unit=from_unit,
                        to_unit=canonical,
                        factor=factor1.value,
                        method=factor1.conversion_type.value,
                        input_value=value,
                        output_value=intermediate,
                        factor_version=factor1.version,
                    ))
                    steps.append(ConversionStep(
                        from_unit=canonical,
                        to_unit=to_unit,
                        factor=factor2.value,
                        method=factor2.conversion_type.value,
                        input_value=intermediate,
                        output_value=final_value,
                        factor_version=factor2.version,
                    ))

                    total_factor = final_value / value if value != 0 else 1.0
                    return (final_value, steps, total_factor)

        return None

    def _convert_with_pint(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        steps: List[ConversionStep],
        warnings: List[str],
    ) -> Optional[Tuple[float, List[ConversionStep], float]]:
        """Convert using Pint library."""
        if not self._ureg:
            return None

        try:
            # Map some common unit names to Pint equivalents
            pint_from = self._map_to_pint_unit(from_unit)
            pint_to = self._map_to_pint_unit(to_unit)

            pint_qty = self._ureg.Quantity(value, pint_from)
            converted = pint_qty.to(pint_to)
            converted_value = float(converted.magnitude)
            factor = converted_value / value if value != 0 else 1.0

            steps.append(ConversionStep(
                from_unit=from_unit,
                to_unit=to_unit,
                factor=factor,
                method="pint",
                input_value=value,
                output_value=converted_value,
                factor_version="pint",
            ))

            return (converted_value, steps, factor)

        except Exception:
            return None

    def _map_to_pint_unit(self, unit: str) -> str:
        """Map GreenLang unit names to Pint unit names."""
        pint_mapping = {
            "degC": "degC",
            "degF": "degF",
            "K": "kelvin",
            "kPa_abs": "kilopascal",
            "kPa": "kilopascal",
            "m3": "meter**3",
            "L": "liter",
            "t": "metric_ton",
            "tonne": "metric_ton",
        }
        return pint_mapping.get(unit, unit)

    def _normalize_unit(self, unit: str) -> str:
        """Normalize a unit string."""
        # Handle common variations
        replacements = {
            " ": "_",
            "-": "_",
            "^": "",
            "**": "",
            "celsius": "degC",
            "fahrenheit": "degF",
            "kelvin": "K",
        }

        result = unit.strip()
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result

    def _is_pressure_unit(self, unit: str) -> bool:
        """Check if a unit is a pressure unit."""
        dimension = self.get_dimension(unit)
        return dimension == "pressure"

    def _elapsed_ms(self, start_time: datetime) -> float:
        """Calculate elapsed time in milliseconds."""
        return (datetime.now() - start_time).total_seconds() * 1000


__all__ = [
    "ConversionEngine",
    "ConversionResult",
    "ConversionStep",
    "GL_CANONICAL_UNITS",
    "UNIT_DIMENSIONS",
]
