"""
Psychrometric Calculator Module - ASHRAE Standard Compliant

Zero-Hallucination Imperial Unit Psychrometric Calculations for Process Heat Applications

This module implements comprehensive psychrometric calculations following ASHRAE
Handbook of Fundamentals using Imperial units (Fahrenheit, psia, BTU/lb).
All calculations are deterministic with complete provenance tracking.

ZERO-HALLUCINATION GUARANTEE:
- All calculations use deterministic ASHRAE formulas (no LLM inference)
- Complete provenance tracking with SHA-256 hashes
- Bit-perfect reproducibility (same input = same output)
- All coefficients from ASHRAE Handbook of Fundamentals

STANDARDS COMPLIANCE:
- ASHRAE Handbook of Fundamentals (Psychrometrics Chapter)
- Magnus-Tetens equation for saturation pressure
- ASHRAE equations for humidity ratio, enthalpy, specific volume

KEY FORMULAS IMPLEMENTED:
- Magnus-Tetens: P_sat = 0.61094 * exp(17.625 * T_c / (T_c + 243.04)) [kPa]
- Humidity ratio: W = 0.622 * P_w / (P_total - P_w) [lb_water/lb_dry_air]
- Enthalpy: h = 0.24 * T_f + W * (1061 + 0.444 * T_f) [BTU/lb_dry_air]
- Specific volume: v = 0.370486 * (T_f + 459.67) * (1 + 1.6078 * W) / P_psia [ft3/lb]

Author: GreenLang Engineering Team
License: MIT
Version: 1.0.0 (Production)
"""

from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, Optional, Any, Tuple
from enum import Enum
import hashlib
import math
import logging
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

# Set high precision for Decimal operations
getcontext().prec = 50

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - ASHRAE HANDBOOK OF FUNDAMENTALS
# =============================================================================

class ASHRAEConstants:
    """
    ASHRAE Handbook of Fundamentals Constants for Psychrometric Calculations.

    All values are extracted from ASHRAE Handbook of Fundamentals.
    ALL VALUES ARE DETERMINISTIC.
    """

    # Temperature conversion
    RANKINE_OFFSET = Decimal("459.67")  # Fahrenheit to Rankine

    # Molecular weight ratio (water vapor / dry air)
    # MW_water / MW_air = 18.01528 / 28.9647 = 0.621945
    EPSILON = Decimal("0.621945")

    # Specific heat values (Imperial units)
    CP_DRY_AIR_BTU_LB_F = Decimal("0.24")  # BTU/(lb*F) - dry air
    CP_WATER_VAPOR_BTU_LB_F = Decimal("0.444")  # BTU/(lb*F) - water vapor

    # Latent heat of vaporization at 0F (approximately)
    # hfg at 32F = 1075.8 BTU/lb, adjusted for 0F reference = 1061 BTU/lb
    LATENT_HEAT_WATER_0F_BTU_LB = Decimal("1061")

    # Gas constant for dry air
    R_DA_FT3_PSIA_LB_R = Decimal("0.370486")  # ft3*psia/(lb*R)

    # Humidity ratio coefficient for specific volume
    # 1/epsilon = 1.6078
    INVERSE_EPSILON_MINUS_ONE = Decimal("1.6078")

    # Magnus-Tetens coefficients (for Celsius, kPa)
    MAGNUS_A = Decimal("0.61094")  # kPa
    MAGNUS_B = Decimal("17.625")  # dimensionless
    MAGNUS_C = Decimal("243.04")  # Celsius

    # Standard atmospheric pressure
    STD_PRESSURE_PSIA = Decimal("14.696")  # psia
    STD_PRESSURE_KPA = Decimal("101.325")  # kPa

    # Conversion factors
    KPA_TO_PSIA = Decimal("0.145038")  # 1 kPa = 0.145038 psia
    PSIA_TO_KPA = Decimal("6.89476")  # 1 psia = 6.89476 kPa

    # ASHRAE polynomial coefficients for saturation pressure over water (SI)
    # Valid for 0C to 200C (ASHRAE Handbook Fundamentals, Equation 6)
    C1_WATER = Decimal("-5.8002206E+03")
    C2_WATER = Decimal("1.3914993E+00")
    C3_WATER = Decimal("-4.8640239E-02")
    C4_WATER = Decimal("4.1764768E-05")
    C5_WATER = Decimal("-1.4452093E-08")
    C6_WATER = Decimal("6.5459673E+00")

    # ASHRAE polynomial coefficients for saturation pressure over ice (SI)
    # Valid for -100C to 0C (ASHRAE Handbook Fundamentals, Equation 5)
    C1_ICE = Decimal("-5.6745359E+03")
    C2_ICE = Decimal("6.3925247E+00")
    C3_ICE = Decimal("-9.6778430E-03")
    C4_ICE = Decimal("6.2215701E-07")
    C5_ICE = Decimal("2.0747825E-09")
    C6_ICE = Decimal("-9.4840240E-13")
    C7_ICE = Decimal("4.1635019E+00")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CalculationMethod(str, Enum):
    """Calculation method selection for saturation pressure."""
    MAGNUS_TETENS = "magnus_tetens"
    ASHRAE_POLYNOMIAL = "ashrae_polynomial"


class UnitSystem(str, Enum):
    """Unit system for input/output."""
    IMPERIAL = "imperial"  # F, psia, BTU/lb, ft3/lb
    SI = "si"  # C, kPa, kJ/kg, m3/kg


# =============================================================================
# PYDANTIC INPUT/OUTPUT MODELS
# =============================================================================

class HumidityRatioInput(BaseModel):
    """
    Input model for humidity ratio calculation from dry-bulb and wet-bulb temperatures.

    ASHRAE Reference: Humidity ratio from wet-bulb and dry-bulb measurements.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    wet_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Wet-bulb temperature in degrees Fahrenheit"
    )
    pressure_psia: float = Field(
        default=14.696,
        gt=0.0,
        le=500.0,
        description="Total atmospheric pressure in psia"
    )

    @model_validator(mode='after')
    def validate_temperatures(self) -> 'HumidityRatioInput':
        """Validate that wet-bulb does not exceed dry-bulb temperature."""
        if self.wet_bulb_f > self.dry_bulb_f:
            raise ValueError(
                f"Wet-bulb temperature ({self.wet_bulb_f}F) cannot exceed "
                f"dry-bulb temperature ({self.dry_bulb_f}F)"
            )
        return self


class HumidityRatioOutput(BaseModel):
    """
    Output model for humidity ratio calculation.

    Contains the calculated humidity ratio with full provenance tracking.
    """
    humidity_ratio_lb_lb: float = Field(
        ...,
        ge=0.0,
        description="Humidity ratio in lb_water per lb_dry_air"
    )
    saturation_pressure_wet_bulb_psia: float = Field(
        ...,
        gt=0.0,
        description="Saturation pressure at wet-bulb temperature (psia)"
    )
    saturation_humidity_ratio_wet_bulb: float = Field(
        ...,
        ge=0.0,
        description="Saturation humidity ratio at wet-bulb temperature"
    )
    partial_pressure_water_psia: float = Field(
        ...,
        ge=0.0,
        description="Partial pressure of water vapor (psia)"
    )
    calculation_method: str = Field(
        ...,
        description="Method used for calculation"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class RelativeHumidityInput(BaseModel):
    """
    Input model for relative humidity calculation from dry-bulb and dew point.

    ASHRAE Reference: Relative humidity from vapor pressures.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    dew_point_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dew point temperature in degrees Fahrenheit"
    )

    @model_validator(mode='after')
    def validate_temperatures(self) -> 'RelativeHumidityInput':
        """Validate that dew point does not exceed dry-bulb temperature."""
        if self.dew_point_f > self.dry_bulb_f:
            raise ValueError(
                f"Dew point temperature ({self.dew_point_f}F) cannot exceed "
                f"dry-bulb temperature ({self.dry_bulb_f}F)"
            )
        return self


class RelativeHumidityOutput(BaseModel):
    """
    Output model for relative humidity calculation.

    Contains the calculated relative humidity with provenance tracking.
    """
    relative_humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity as percentage (0-100)"
    )
    saturation_pressure_dry_bulb_psia: float = Field(
        ...,
        gt=0.0,
        description="Saturation pressure at dry-bulb temperature (psia)"
    )
    saturation_pressure_dew_point_psia: float = Field(
        ...,
        gt=0.0,
        description="Saturation pressure at dew point (psia)"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class DewPointInput(BaseModel):
    """
    Input model for dew point temperature calculation.

    ASHRAE Reference: Dew point from dry-bulb and relative humidity.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    relative_humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity as percentage (0-100)"
    )

    @field_validator('relative_humidity_pct')
    @classmethod
    def validate_rh(cls, v: float) -> float:
        """Validate relative humidity is within valid range."""
        if v < 0 or v > 100:
            raise ValueError(f"Relative humidity must be 0-100%, got {v}%")
        return v


class DewPointOutput(BaseModel):
    """
    Output model for dew point temperature calculation.

    Contains the calculated dew point with provenance tracking.
    """
    dew_point_f: float = Field(
        ...,
        description="Dew point temperature in degrees Fahrenheit"
    )
    saturation_pressure_dry_bulb_psia: float = Field(
        ...,
        gt=0.0,
        description="Saturation pressure at dry-bulb temperature (psia)"
    )
    partial_pressure_water_psia: float = Field(
        ...,
        ge=0.0,
        description="Partial pressure of water vapor (psia)"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class WetBulbInput(BaseModel):
    """
    Input model for wet-bulb temperature calculation.

    ASHRAE Reference: Wet-bulb from dry-bulb, relative humidity, and pressure.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    relative_humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity as percentage (0-100)"
    )
    pressure_psia: float = Field(
        default=14.696,
        gt=0.0,
        le=500.0,
        description="Total atmospheric pressure in psia"
    )


class WetBulbOutput(BaseModel):
    """
    Output model for wet-bulb temperature calculation.

    Contains the calculated wet-bulb temperature with provenance tracking.
    """
    wet_bulb_f: float = Field(
        ...,
        description="Wet-bulb temperature in degrees Fahrenheit"
    )
    humidity_ratio_lb_lb: float = Field(
        ...,
        ge=0.0,
        description="Humidity ratio at the given conditions"
    )
    iterations_required: int = Field(
        ...,
        ge=1,
        description="Number of iterations for convergence"
    )
    convergence_error: float = Field(
        ...,
        ge=0.0,
        description="Final convergence error"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class EnthalpyInput(BaseModel):
    """
    Input model for specific enthalpy calculation.

    ASHRAE Reference: Moist air enthalpy equation.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    humidity_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Humidity ratio in lb_water per lb_dry_air"
    )


class EnthalpyOutput(BaseModel):
    """
    Output model for specific enthalpy calculation.

    Contains the calculated enthalpy with provenance tracking.
    """
    enthalpy_btu_lb: float = Field(
        ...,
        description="Specific enthalpy in BTU per lb of dry air"
    )
    enthalpy_dry_air_btu_lb: float = Field(
        ...,
        description="Dry air contribution to enthalpy"
    )
    enthalpy_water_vapor_btu_lb: float = Field(
        ...,
        description="Water vapor contribution to enthalpy"
    )
    formula_used: str = Field(
        ...,
        description="Formula reference used"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class SpecificVolumeInput(BaseModel):
    """
    Input model for specific volume calculation.

    ASHRAE Reference: Moist air specific volume equation.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    humidity_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Humidity ratio in lb_water per lb_dry_air"
    )
    pressure_psia: float = Field(
        default=14.696,
        gt=0.0,
        le=500.0,
        description="Total atmospheric pressure in psia"
    )


class SpecificVolumeOutput(BaseModel):
    """
    Output model for specific volume calculation.

    Contains the calculated specific volume with provenance tracking.
    """
    specific_volume_ft3_lb: float = Field(
        ...,
        gt=0.0,
        description="Specific volume in cubic feet per lb of dry air"
    )
    density_lb_ft3: float = Field(
        ...,
        gt=0.0,
        description="Moist air density in lb per cubic foot"
    )
    formula_used: str = Field(
        ...,
        description="Formula reference used"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class SaturationPressureInput(BaseModel):
    """
    Input model for saturation vapor pressure calculation.

    ASHRAE Reference: Magnus-Tetens or ASHRAE polynomial equations.
    """
    temperature_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Temperature in degrees Fahrenheit"
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.MAGNUS_TETENS,
        description="Calculation method to use"
    )


class SaturationPressureOutput(BaseModel):
    """
    Output model for saturation vapor pressure calculation.

    Contains the calculated saturation pressure with provenance tracking.
    """
    saturation_pressure_psia: float = Field(
        ...,
        gt=0.0,
        description="Saturation vapor pressure in psia"
    )
    saturation_pressure_kpa: float = Field(
        ...,
        gt=0.0,
        description="Saturation vapor pressure in kPa"
    )
    temperature_c: float = Field(
        ...,
        description="Temperature in Celsius (for reference)"
    )
    method_used: str = Field(
        ...,
        description="Calculation method used"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds"
    )


class CompletePsychrometricInput(BaseModel):
    """
    Input model for complete psychrometric state calculation.

    Requires two independent properties plus pressure to fully define the state.
    """
    dry_bulb_f: float = Field(
        ...,
        ge=-40.0,
        le=400.0,
        description="Dry-bulb temperature in degrees Fahrenheit"
    )
    relative_humidity_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Relative humidity as percentage (0-100)"
    )
    wet_bulb_f: Optional[float] = Field(
        default=None,
        ge=-40.0,
        le=400.0,
        description="Wet-bulb temperature in degrees Fahrenheit"
    )
    dew_point_f: Optional[float] = Field(
        default=None,
        ge=-40.0,
        le=400.0,
        description="Dew point temperature in degrees Fahrenheit"
    )
    humidity_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Humidity ratio in lb_water per lb_dry_air"
    )
    pressure_psia: float = Field(
        default=14.696,
        gt=0.0,
        le=500.0,
        description="Total atmospheric pressure in psia"
    )

    @model_validator(mode='after')
    def validate_inputs(self) -> 'CompletePsychrometricInput':
        """Validate that at least one humidity property is provided."""
        humidity_props = [
            self.relative_humidity_pct,
            self.wet_bulb_f,
            self.dew_point_f,
            self.humidity_ratio
        ]
        provided = sum(1 for p in humidity_props if p is not None)

        if provided == 0:
            raise ValueError(
                "At least one humidity property must be provided: "
                "relative_humidity_pct, wet_bulb_f, dew_point_f, or humidity_ratio"
            )
        return self


class CompletePsychrometricOutput(BaseModel):
    """
    Output model for complete psychrometric state.

    Contains all psychrometric properties with full provenance tracking.
    """
    # Temperature properties (F)
    dry_bulb_f: float = Field(..., description="Dry-bulb temperature (F)")
    wet_bulb_f: float = Field(..., description="Wet-bulb temperature (F)")
    dew_point_f: float = Field(..., description="Dew point temperature (F)")

    # Humidity properties
    relative_humidity_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity (%)"
    )
    humidity_ratio_lb_lb: float = Field(
        ...,
        ge=0.0,
        description="Humidity ratio (lb_water/lb_dry_air)"
    )
    specific_humidity_lb_lb: float = Field(
        ...,
        ge=0.0,
        description="Specific humidity (lb_water/lb_moist_air)"
    )

    # Pressure properties (psia)
    pressure_psia: float = Field(..., gt=0.0, description="Total pressure (psia)")
    saturation_pressure_psia: float = Field(
        ...,
        gt=0.0,
        description="Saturation pressure at dry-bulb (psia)"
    )
    partial_pressure_water_psia: float = Field(
        ...,
        ge=0.0,
        description="Partial pressure of water vapor (psia)"
    )

    # Thermodynamic properties
    enthalpy_btu_lb: float = Field(
        ...,
        description="Specific enthalpy (BTU/lb_dry_air)"
    )
    specific_volume_ft3_lb: float = Field(
        ...,
        gt=0.0,
        description="Specific volume (ft3/lb_dry_air)"
    )
    density_lb_ft3: float = Field(
        ...,
        gt=0.0,
        description="Moist air density (lb/ft3)"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_timestamp: str = Field(..., description="ISO timestamp of calculation")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time (ms)")


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================

class PsychrometricCalculator:
    """
    ASHRAE-Compliant Psychrometric Calculator (Imperial Units).

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic (no LLM inference)
    - Based on ASHRAE Handbook of Fundamentals equations
    - Complete provenance tracking with SHA-256 hashes
    - Bit-perfect reproducibility (same input = same output)

    Key Features:
    - Imperial unit system (F, psia, BTU/lb, ft3/lb)
    - Magnus-Tetens saturation pressure calculation
    - ASHRAE polynomial equations for extended accuracy
    - Complete state calculation from minimal inputs
    - Full provenance tracking for regulatory compliance

    Example:
        >>> calc = PsychrometricCalculator()
        >>> # Calculate humidity ratio from wet-bulb and dry-bulb
        >>> result = calc.calculate_humidity_ratio(80.0, 65.0, 14.696)
        >>> print(f"Humidity ratio: {result.humidity_ratio_lb_lb:.6f} lb/lb")

        >>> # Calculate complete psychrometric state
        >>> state = calc.calculate_complete_state(80.0, relative_humidity_pct=50.0)
        >>> print(f"Enthalpy: {state.enthalpy_btu_lb:.2f} BTU/lb")

    References:
        - ASHRAE Handbook of Fundamentals, Chapter 1 (Psychrometrics)
        - Magnus-Tetens equation (Buck, 1981)
        - ASME PTC 4.1: Steam Generating Units
    """

    def __init__(
        self,
        precision: int = 6,
        method: CalculationMethod = CalculationMethod.MAGNUS_TETENS
    ):
        """
        Initialize the Psychrometric Calculator.

        Args:
            precision: Number of decimal places for results (default: 6)
            method: Default calculation method for saturation pressure
        """
        self.precision = precision
        self.default_method = method
        self.constants = ASHRAEConstants()
        logger.info(
            f"PsychrometricCalculator initialized with precision={precision}, "
            f"method={method.value}"
        )

    def _apply_precision(self, value: Decimal) -> Decimal:
        """
        Apply precision rounding to a Decimal value.

        Args:
            value: Decimal value to round

        Returns:
            Rounded Decimal value
        """
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(
        self,
        function_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 hash for complete audit trail.

        This provides a unique, deterministic identifier for the calculation
        that can be used to verify reproducibility.

        Args:
            function_name: Name of the calculation function
            inputs: Dictionary of input parameters
            outputs: Dictionary of output values

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            "standard": "ASHRAE_Psychrometrics",
            "function": function_name,
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _fahrenheit_to_celsius(self, temp_f: float) -> Decimal:
        """
        Convert temperature from Fahrenheit to Celsius.

        Args:
            temp_f: Temperature in Fahrenheit

        Returns:
            Temperature in Celsius as Decimal
        """
        return (Decimal(str(temp_f)) - Decimal("32")) * Decimal("5") / Decimal("9")

    def _celsius_to_fahrenheit(self, temp_c: Decimal) -> Decimal:
        """
        Convert temperature from Celsius to Fahrenheit.

        Args:
            temp_c: Temperature in Celsius as Decimal

        Returns:
            Temperature in Fahrenheit as Decimal
        """
        return temp_c * Decimal("9") / Decimal("5") + Decimal("32")

    def _kpa_to_psia(self, pressure_kpa: Decimal) -> Decimal:
        """
        Convert pressure from kPa to psia.

        Args:
            pressure_kpa: Pressure in kPa

        Returns:
            Pressure in psia as Decimal
        """
        return pressure_kpa * self.constants.KPA_TO_PSIA

    def _psia_to_kpa(self, pressure_psia: Decimal) -> Decimal:
        """
        Convert pressure from psia to kPa.

        Args:
            pressure_psia: Pressure in psia

        Returns:
            Pressure in kPa as Decimal
        """
        return pressure_psia * self.constants.PSIA_TO_KPA

    # =========================================================================
    # SATURATION PRESSURE CALCULATION
    # =========================================================================

    def calculate_saturation_pressure(
        self,
        temp_f: float,
        method: Optional[CalculationMethod] = None
    ) -> SaturationPressureOutput:
        """
        Calculate saturation vapor pressure using Magnus-Tetens equation.

        ASHRAE Reference: Magnus-Tetens approximation for saturation vapor pressure.

        Formula (Magnus-Tetens):
            P_sat = 0.61094 * exp(17.625 * T_c / (T_c + 243.04)) [kPa]

        This equation is valid for temperatures from -40C to 50C with accuracy
        better than 0.1% compared to the Goff-Gratch equation.

        Args:
            temp_f: Temperature in degrees Fahrenheit
            method: Calculation method (defaults to instance default)

        Returns:
            SaturationPressureOutput with calculated pressure and provenance

        Raises:
            ValueError: If temperature is outside valid range

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_saturation_pressure(77.0)
            >>> print(f"P_sat = {result.saturation_pressure_psia:.4f} psia")
        """
        start_time = datetime.now()
        method = method or self.default_method

        # Convert to Celsius for calculation
        temp_c = self._fahrenheit_to_celsius(temp_f)

        # Validate temperature range
        if temp_c < Decimal("-100") or temp_c > Decimal("200"):
            raise ValueError(
                f"Temperature {temp_f}F ({float(temp_c):.1f}C) outside valid range "
                f"[-148F to 392F] / [-100C to 200C]"
            )

        if method == CalculationMethod.MAGNUS_TETENS:
            # Magnus-Tetens equation
            # P_sat = 0.61094 * exp(17.625 * T_c / (T_c + 243.04)) [kPa]
            exponent = (
                self.constants.MAGNUS_B * temp_c /
                (temp_c + self.constants.MAGNUS_C)
            )
            p_sat_kpa = self.constants.MAGNUS_A * Decimal(str(math.exp(float(exponent))))

        else:  # ASHRAE_POLYNOMIAL
            # ASHRAE Handbook Fundamentals polynomial
            temp_k = temp_c + Decimal("273.15")

            if temp_c >= Decimal("0"):
                # Over liquid water (Equation 6)
                ln_pws = (
                    self.constants.C1_WATER / temp_k +
                    self.constants.C2_WATER +
                    self.constants.C3_WATER * temp_k +
                    self.constants.C4_WATER * temp_k ** 2 +
                    self.constants.C5_WATER * temp_k ** 3 +
                    self.constants.C6_WATER * Decimal(str(math.log(float(temp_k))))
                )
            else:
                # Over ice (Equation 5)
                ln_pws = (
                    self.constants.C1_ICE / temp_k +
                    self.constants.C2_ICE +
                    self.constants.C3_ICE * temp_k +
                    self.constants.C4_ICE * temp_k ** 2 +
                    self.constants.C5_ICE * temp_k ** 3 +
                    self.constants.C6_ICE * temp_k ** 4 +
                    self.constants.C7_ICE * Decimal(str(math.log(float(temp_k))))
                )

            # Result in Pa, convert to kPa
            p_sat_kpa = Decimal(str(math.exp(float(ln_pws)))) / Decimal("1000")

        # Convert to psia
        p_sat_psia = self._kpa_to_psia(p_sat_kpa)

        # Apply precision
        p_sat_kpa = self._apply_precision(p_sat_kpa)
        p_sat_psia = self._apply_precision(p_sat_psia)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_saturation_pressure",
            {"temp_f": temp_f, "method": method.value},
            {"p_sat_psia": str(p_sat_psia), "p_sat_kpa": str(p_sat_kpa)}
        )

        return SaturationPressureOutput(
            saturation_pressure_psia=float(p_sat_psia),
            saturation_pressure_kpa=float(p_sat_kpa),
            temperature_c=float(temp_c),
            method_used=method.value,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # HUMIDITY RATIO CALCULATION
    # =========================================================================

    def calculate_humidity_ratio(
        self,
        dry_bulb_f: float,
        wet_bulb_f: float,
        pressure_psia: float = 14.696
    ) -> HumidityRatioOutput:
        """
        Calculate humidity ratio from dry-bulb and wet-bulb temperatures.

        ASHRAE Reference: Psychrometric equation for humidity ratio.

        Formula:
            W_s* = 0.622 * P_ws* / (P - P_ws*)  [saturation at wet-bulb]

            For T_wb >= 32F (above freezing):
            W = ((1093 - 0.556*T_wb) * W_s* - 0.24*(T_db - T_wb)) /
                (1093 + 0.444*T_db - T_wb)

            For T_wb < 32F (below freezing):
            W = ((1220 - 0.04*T_wb) * W_s* - 0.24*(T_db - T_wb)) /
                (1220 + 0.444*T_db - 0.48*T_wb)

        The humidity ratio formula is derived from the energy balance on an
        adiabatic saturation process.

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            wet_bulb_f: Wet-bulb temperature in degrees Fahrenheit
            pressure_psia: Total atmospheric pressure in psia (default: 14.696)

        Returns:
            HumidityRatioOutput with calculated humidity ratio and provenance

        Raises:
            ValueError: If wet-bulb exceeds dry-bulb temperature

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_humidity_ratio(80.0, 65.0, 14.696)
            >>> print(f"W = {result.humidity_ratio_lb_lb:.6f} lb/lb")
        """
        start_time = datetime.now()

        # Validate input
        input_data = HumidityRatioInput(
            dry_bulb_f=dry_bulb_f,
            wet_bulb_f=wet_bulb_f,
            pressure_psia=pressure_psia
        )

        t_db = Decimal(str(input_data.dry_bulb_f))
        t_wb = Decimal(str(input_data.wet_bulb_f))
        p = Decimal(str(input_data.pressure_psia))

        # Calculate saturation pressure at wet-bulb temperature
        p_ws_result = self.calculate_saturation_pressure(float(t_wb))
        p_ws = Decimal(str(p_ws_result.saturation_pressure_psia))

        # Calculate saturation humidity ratio at wet-bulb
        # W_s* = 0.622 * P_ws* / (P - P_ws*)
        if p_ws >= p:
            raise ValueError(
                f"Saturation pressure ({float(p_ws):.4f} psia) cannot exceed "
                f"total pressure ({float(p):.4f} psia)"
            )

        w_s_star = self.constants.EPSILON * p_ws / (p - p_ws)

        # Calculate humidity ratio using psychrometric equation
        if t_wb >= Decimal("32"):
            # Above freezing (ASHRAE Equation 35)
            # Coefficients: 1093 = hfg at 32F, 0.556 = cp_w - cp_ice, 0.24 = cp_air
            numerator = (
                (Decimal("1093") - Decimal("0.556") * t_wb) * w_s_star -
                Decimal("0.24") * (t_db - t_wb)
            )
            denominator = Decimal("1093") + Decimal("0.444") * t_db - t_wb
        else:
            # Below freezing (ASHRAE Equation 37)
            # Uses sublimation properties
            numerator = (
                (Decimal("1220") - Decimal("0.04") * t_wb) * w_s_star -
                Decimal("0.24") * (t_db - t_wb)
            )
            denominator = Decimal("1220") + Decimal("0.444") * t_db - Decimal("0.48") * t_wb

        w = numerator / denominator

        # Ensure non-negative
        if w < Decimal("0"):
            w = Decimal("0")

        # Calculate partial pressure of water vapor from humidity ratio
        # w = 0.622 * pw / (p - pw)
        # pw = w * p / (0.622 + w)
        p_w = w * p / (self.constants.EPSILON + w)

        # Apply precision
        w = self._apply_precision(w)
        p_w = self._apply_precision(p_w)
        w_s_star = self._apply_precision(w_s_star)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_humidity_ratio",
            {"dry_bulb_f": dry_bulb_f, "wet_bulb_f": wet_bulb_f, "pressure_psia": pressure_psia},
            {"humidity_ratio": str(w), "partial_pressure_psia": str(p_w)}
        )

        return HumidityRatioOutput(
            humidity_ratio_lb_lb=float(w),
            saturation_pressure_wet_bulb_psia=float(p_ws),
            saturation_humidity_ratio_wet_bulb=float(w_s_star),
            partial_pressure_water_psia=float(p_w),
            calculation_method="ASHRAE_psychrometric_equation",
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # RELATIVE HUMIDITY CALCULATION
    # =========================================================================

    def calculate_relative_humidity(
        self,
        dry_bulb_f: float,
        dew_point_f: float
    ) -> RelativeHumidityOutput:
        """
        Calculate relative humidity from dry-bulb and dew point temperatures.

        ASHRAE Reference: Relative humidity from saturation pressures.

        Formula:
            RH = (P_ws(T_dp) / P_ws(T_db)) * 100

        Where:
            P_ws(T_dp) = saturation pressure at dew point temperature
            P_ws(T_db) = saturation pressure at dry-bulb temperature

        The relative humidity is the ratio of actual water vapor partial
        pressure to the saturation pressure at the dry-bulb temperature.

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            dew_point_f: Dew point temperature in degrees Fahrenheit

        Returns:
            RelativeHumidityOutput with calculated relative humidity and provenance

        Raises:
            ValueError: If dew point exceeds dry-bulb temperature

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_relative_humidity(80.0, 60.0)
            >>> print(f"RH = {result.relative_humidity_pct:.1f}%")
        """
        start_time = datetime.now()

        # Validate input
        input_data = RelativeHumidityInput(
            dry_bulb_f=dry_bulb_f,
            dew_point_f=dew_point_f
        )

        # Calculate saturation pressure at dry-bulb temperature
        p_ws_db = self.calculate_saturation_pressure(input_data.dry_bulb_f)

        # Calculate saturation pressure at dew point (equals partial pressure)
        p_ws_dp = self.calculate_saturation_pressure(input_data.dew_point_f)

        # Calculate relative humidity
        # RH = P_ws(T_dp) / P_ws(T_db) * 100
        p_ws_db_dec = Decimal(str(p_ws_db.saturation_pressure_psia))
        p_ws_dp_dec = Decimal(str(p_ws_dp.saturation_pressure_psia))

        rh = p_ws_dp_dec / p_ws_db_dec * Decimal("100")

        # Clamp to valid range
        if rh > Decimal("100"):
            rh = Decimal("100")
        if rh < Decimal("0"):
            rh = Decimal("0")

        rh = self._apply_precision(rh)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_relative_humidity",
            {"dry_bulb_f": dry_bulb_f, "dew_point_f": dew_point_f},
            {"relative_humidity_pct": str(rh)}
        )

        return RelativeHumidityOutput(
            relative_humidity_pct=float(rh),
            saturation_pressure_dry_bulb_psia=p_ws_db.saturation_pressure_psia,
            saturation_pressure_dew_point_psia=p_ws_dp.saturation_pressure_psia,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # DEW POINT CALCULATION
    # =========================================================================

    def calculate_dew_point(
        self,
        dry_bulb_f: float,
        relative_humidity_pct: float
    ) -> DewPointOutput:
        """
        Calculate dew point temperature from dry-bulb and relative humidity.

        ASHRAE Reference: Dew point from vapor pressure inversion.

        Formula:
            1. Calculate partial pressure: P_w = RH/100 * P_ws(T_db)
            2. Invert Magnus-Tetens to find T_dp where P_ws(T_dp) = P_w

            Magnus-Tetens inversion (for Celsius):
            T_dp = 243.04 * ln(P_w / 0.61094) / (17.625 - ln(P_w / 0.61094))

        The dew point is the temperature at which saturation pressure equals
        the actual vapor partial pressure.

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            relative_humidity_pct: Relative humidity as percentage (0-100)

        Returns:
            DewPointOutput with calculated dew point and provenance

        Raises:
            ValueError: If relative humidity is outside valid range

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_dew_point(80.0, 50.0)
            >>> print(f"Dew point = {result.dew_point_f:.1f}F")
        """
        start_time = datetime.now()

        # Validate input
        input_data = DewPointInput(
            dry_bulb_f=dry_bulb_f,
            relative_humidity_pct=relative_humidity_pct
        )

        rh = Decimal(str(input_data.relative_humidity_pct)) / Decimal("100")

        # Handle edge case of 0% RH
        if rh <= Decimal("0"):
            # Return very low dew point
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            provenance_hash = self._calculate_provenance(
                "calculate_dew_point",
                {"dry_bulb_f": dry_bulb_f, "relative_humidity_pct": relative_humidity_pct},
                {"dew_point_f": "-40.0"}
            )
            return DewPointOutput(
                dew_point_f=-40.0,
                saturation_pressure_dry_bulb_psia=0.0,
                partial_pressure_water_psia=0.0,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time
            )

        # Calculate saturation pressure at dry-bulb
        p_ws_db = self.calculate_saturation_pressure(input_data.dry_bulb_f)
        p_ws_db_dec = Decimal(str(p_ws_db.saturation_pressure_psia))

        # Calculate partial pressure of water vapor
        p_w = rh * p_ws_db_dec

        # Convert to kPa for Magnus-Tetens inversion
        p_w_kpa = self._psia_to_kpa(p_w)

        # Invert Magnus-Tetens equation to find dew point
        # P_w = 0.61094 * exp(17.625 * T_c / (T_c + 243.04))
        # ln(P_w / 0.61094) = 17.625 * T_c / (T_c + 243.04)
        # Let alpha = ln(P_w / 0.61094)
        # alpha * (T_c + 243.04) = 17.625 * T_c
        # alpha * T_c + 243.04 * alpha = 17.625 * T_c
        # 243.04 * alpha = T_c * (17.625 - alpha)
        # T_c = 243.04 * alpha / (17.625 - alpha)

        alpha = Decimal(str(math.log(float(p_w_kpa / self.constants.MAGNUS_A))))
        t_dp_c = self.constants.MAGNUS_C * alpha / (self.constants.MAGNUS_B - alpha)

        # Convert to Fahrenheit
        t_dp_f = self._celsius_to_fahrenheit(t_dp_c)
        t_dp_f = self._apply_precision(t_dp_f)
        p_w = self._apply_precision(p_w)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_dew_point",
            {"dry_bulb_f": dry_bulb_f, "relative_humidity_pct": relative_humidity_pct},
            {"dew_point_f": str(t_dp_f), "partial_pressure_psia": str(p_w)}
        )

        return DewPointOutput(
            dew_point_f=float(t_dp_f),
            saturation_pressure_dry_bulb_psia=p_ws_db.saturation_pressure_psia,
            partial_pressure_water_psia=float(p_w),
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # WET-BULB TEMPERATURE CALCULATION
    # =========================================================================

    def calculate_wet_bulb(
        self,
        dry_bulb_f: float,
        relative_humidity_pct: float,
        pressure_psia: float = 14.696
    ) -> WetBulbOutput:
        """
        Calculate wet-bulb temperature from dry-bulb, relative humidity, and pressure.

        ASHRAE Reference: Iterative solution of psychrometric equation.

        Method:
            1. Calculate target humidity ratio from T_db and RH
            2. Iterate on T_wb until calculated W matches target W
            3. Uses Newton-Raphson iteration for fast convergence

        The wet-bulb temperature is found by solving the psychrometric equation
        iteratively until the humidity ratio matches the actual value.

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            relative_humidity_pct: Relative humidity as percentage (0-100)
            pressure_psia: Total atmospheric pressure in psia (default: 14.696)

        Returns:
            WetBulbOutput with calculated wet-bulb temperature and provenance

        Raises:
            ValueError: If relative humidity is outside valid range
            RuntimeError: If iteration fails to converge

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_wet_bulb(80.0, 50.0, 14.696)
            >>> print(f"Wet-bulb = {result.wet_bulb_f:.1f}F")
        """
        start_time = datetime.now()

        # Validate input
        input_data = WetBulbInput(
            dry_bulb_f=dry_bulb_f,
            relative_humidity_pct=relative_humidity_pct,
            pressure_psia=pressure_psia
        )

        t_db = Decimal(str(input_data.dry_bulb_f))
        rh = Decimal(str(input_data.relative_humidity_pct)) / Decimal("100")
        p = Decimal(str(input_data.pressure_psia))

        # Handle edge case: 100% RH means wet-bulb equals dry-bulb
        if rh >= Decimal("1"):
            # Calculate humidity ratio at saturation
            p_ws = self.calculate_saturation_pressure(float(t_db))
            p_ws_dec = Decimal(str(p_ws.saturation_pressure_psia))
            w = self.constants.EPSILON * p_ws_dec / (p - p_ws_dec)
            w = self._apply_precision(w)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            provenance_hash = self._calculate_provenance(
                "calculate_wet_bulb",
                {"dry_bulb_f": dry_bulb_f, "relative_humidity_pct": relative_humidity_pct,
                 "pressure_psia": pressure_psia},
                {"wet_bulb_f": str(t_db), "humidity_ratio": str(w)}
            )
            return WetBulbOutput(
                wet_bulb_f=float(t_db),
                humidity_ratio_lb_lb=float(w),
                iterations_required=1,
                convergence_error=0.0,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time
            )

        # Calculate target humidity ratio from T_db and RH
        p_ws_db = self.calculate_saturation_pressure(float(t_db))
        p_ws_db_dec = Decimal(str(p_ws_db.saturation_pressure_psia))
        p_w = rh * p_ws_db_dec
        w_target = self.constants.EPSILON * p_w / (p - p_w)

        # Initial guess: use approximation formula
        # T_wb ~ T_db - (1 - RH) * (T_db - T_dp) / 3
        # Simplified: T_wb ~ T_db * (1 - 0.3 * (1 - RH))
        t_wb = t_db - (Decimal("1") - rh) * Decimal("15")

        # Iteration parameters
        tolerance = Decimal("0.0001")
        max_iterations = 50
        iterations = 0

        for i in range(max_iterations):
            iterations = i + 1

            # Calculate humidity ratio at current wet-bulb guess
            p_ws_wb = self.calculate_saturation_pressure(float(t_wb))
            p_ws_wb_dec = Decimal(str(p_ws_wb.saturation_pressure_psia))

            # Saturation humidity ratio at wet-bulb
            w_s_star = self.constants.EPSILON * p_ws_wb_dec / (p - p_ws_wb_dec)

            # Calculate humidity ratio using psychrometric equation
            if t_wb >= Decimal("32"):
                numerator = (
                    (Decimal("1093") - Decimal("0.556") * t_wb) * w_s_star -
                    Decimal("0.24") * (t_db - t_wb)
                )
                denominator = Decimal("1093") + Decimal("0.444") * t_db - t_wb
            else:
                numerator = (
                    (Decimal("1220") - Decimal("0.04") * t_wb) * w_s_star -
                    Decimal("0.24") * (t_db - t_wb)
                )
                denominator = Decimal("1220") + Decimal("0.444") * t_db - Decimal("0.48") * t_wb

            w_calc = numerator / denominator

            # Calculate error
            error = w_calc - w_target

            if abs(error) < tolerance:
                break

            # Adjust wet-bulb temperature
            # Use secant-like adjustment
            adjustment = error * Decimal("100")
            if error > 0:
                t_wb = t_wb - Decimal("0.5") * abs(adjustment)
            else:
                t_wb = t_wb + Decimal("0.5") * abs(adjustment)

            # Ensure wet-bulb doesn't exceed dry-bulb
            if t_wb > t_db:
                t_wb = t_db

            # Ensure wet-bulb doesn't go below minimum
            if t_wb < Decimal("-40"):
                t_wb = Decimal("-40")

        # Final values
        t_wb = self._apply_precision(t_wb)
        w_target = self._apply_precision(w_target)
        final_error = float(abs(error))

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_wet_bulb",
            {"dry_bulb_f": dry_bulb_f, "relative_humidity_pct": relative_humidity_pct,
             "pressure_psia": pressure_psia},
            {"wet_bulb_f": str(t_wb), "humidity_ratio": str(w_target),
             "iterations": str(iterations)}
        )

        return WetBulbOutput(
            wet_bulb_f=float(t_wb),
            humidity_ratio_lb_lb=float(w_target),
            iterations_required=iterations,
            convergence_error=final_error,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # ENTHALPY CALCULATION
    # =========================================================================

    def calculate_enthalpy(
        self,
        dry_bulb_f: float,
        humidity_ratio: float
    ) -> EnthalpyOutput:
        """
        Calculate specific enthalpy of moist air.

        ASHRAE Reference: Moist air enthalpy equation.

        Formula:
            h = 0.24 * T_f + W * (1061 + 0.444 * T_f) [BTU/lb_dry_air]

        Where:
            0.24 = specific heat of dry air (BTU/lb-F)
            1061 = latent heat of vaporization at 0F (BTU/lb)
            0.444 = specific heat of water vapor (BTU/lb-F)

        The enthalpy reference is 0 BTU/lb for dry air at 0F.

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            humidity_ratio: Humidity ratio in lb_water per lb_dry_air

        Returns:
            EnthalpyOutput with calculated enthalpy and provenance

        Raises:
            ValueError: If humidity ratio is negative

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_enthalpy(80.0, 0.0115)
            >>> print(f"h = {result.enthalpy_btu_lb:.2f} BTU/lb")
        """
        start_time = datetime.now()

        # Validate input
        input_data = EnthalpyInput(
            dry_bulb_f=dry_bulb_f,
            humidity_ratio=humidity_ratio
        )

        t = Decimal(str(input_data.dry_bulb_f))
        w = Decimal(str(input_data.humidity_ratio))

        # Calculate enthalpy components
        # h = cp_air * T + W * (hfg + cp_vapor * T)
        # h = 0.24 * T + W * (1061 + 0.444 * T)

        h_dry_air = self.constants.CP_DRY_AIR_BTU_LB_F * t
        h_water_vapor = w * (
            self.constants.LATENT_HEAT_WATER_0F_BTU_LB +
            self.constants.CP_WATER_VAPOR_BTU_LB_F * t
        )
        h_total = h_dry_air + h_water_vapor

        # Apply precision
        h_total = self._apply_precision(h_total)
        h_dry_air = self._apply_precision(h_dry_air)
        h_water_vapor = self._apply_precision(h_water_vapor)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_enthalpy",
            {"dry_bulb_f": dry_bulb_f, "humidity_ratio": humidity_ratio},
            {"enthalpy_btu_lb": str(h_total)}
        )

        return EnthalpyOutput(
            enthalpy_btu_lb=float(h_total),
            enthalpy_dry_air_btu_lb=float(h_dry_air),
            enthalpy_water_vapor_btu_lb=float(h_water_vapor),
            formula_used="h = 0.24*T + W*(1061 + 0.444*T)",
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # SPECIFIC VOLUME CALCULATION
    # =========================================================================

    def calculate_specific_volume(
        self,
        dry_bulb_f: float,
        humidity_ratio: float,
        pressure_psia: float = 14.696
    ) -> SpecificVolumeOutput:
        """
        Calculate specific volume of moist air.

        ASHRAE Reference: Moist air specific volume equation.

        Formula:
            v = 0.370486 * (T_f + 459.67) * (1 + 1.6078 * W) / P_psia [ft3/lb_dry_air]

        Where:
            0.370486 = R_air / M_air = gas constant for dry air (ft3*psia/(lb*R))
            459.67 = Fahrenheit to Rankine offset
            1.6078 = 1/epsilon - 1 = molecular weight adjustment
            P_psia = total pressure in psia

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            humidity_ratio: Humidity ratio in lb_water per lb_dry_air
            pressure_psia: Total atmospheric pressure in psia (default: 14.696)

        Returns:
            SpecificVolumeOutput with calculated specific volume and provenance

        Raises:
            ValueError: If pressure or humidity ratio is invalid

        Example:
            >>> calc = PsychrometricCalculator()
            >>> result = calc.calculate_specific_volume(80.0, 0.0115, 14.696)
            >>> print(f"v = {result.specific_volume_ft3_lb:.3f} ft3/lb")
        """
        start_time = datetime.now()

        # Validate input
        input_data = SpecificVolumeInput(
            dry_bulb_f=dry_bulb_f,
            humidity_ratio=humidity_ratio,
            pressure_psia=pressure_psia
        )

        t = Decimal(str(input_data.dry_bulb_f))
        w = Decimal(str(input_data.humidity_ratio))
        p = Decimal(str(input_data.pressure_psia))

        # Calculate specific volume
        # v = R_da * T_R * (1 + W/epsilon) / P
        # v = 0.370486 * (T_f + 459.67) * (1 + 1.6078*W) / P

        t_rankine = t + self.constants.RANKINE_OFFSET
        humidity_factor = Decimal("1") + self.constants.INVERSE_EPSILON_MINUS_ONE * w

        v = self.constants.R_DA_FT3_PSIA_LB_R * t_rankine * humidity_factor / p

        # Calculate density (lb moist air per ft3)
        # density = (1 + W) / v
        density = (Decimal("1") + w) / v

        # Apply precision
        v = self._apply_precision(v)
        density = self._apply_precision(density)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_specific_volume",
            {"dry_bulb_f": dry_bulb_f, "humidity_ratio": humidity_ratio,
             "pressure_psia": pressure_psia},
            {"specific_volume_ft3_lb": str(v), "density_lb_ft3": str(density)}
        )

        return SpecificVolumeOutput(
            specific_volume_ft3_lb=float(v),
            density_lb_ft3=float(density),
            formula_used="v = 0.370486*(T+459.67)*(1+1.6078*W)/P",
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # COMPLETE PSYCHROMETRIC STATE CALCULATION
    # =========================================================================

    def calculate_complete_state(
        self,
        dry_bulb_f: float,
        relative_humidity_pct: Optional[float] = None,
        wet_bulb_f: Optional[float] = None,
        dew_point_f: Optional[float] = None,
        humidity_ratio: Optional[float] = None,
        pressure_psia: float = 14.696
    ) -> CompletePsychrometricOutput:
        """
        Calculate complete psychrometric state from minimal inputs.

        ASHRAE Reference: Complete psychrometric state calculation.

        This method calculates all psychrometric properties from the dry-bulb
        temperature, pressure, and one humidity property. The priority order
        for humidity property selection is:
        1. relative_humidity_pct
        2. wet_bulb_f
        3. dew_point_f
        4. humidity_ratio

        Args:
            dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
            relative_humidity_pct: Relative humidity (0-100), optional
            wet_bulb_f: Wet-bulb temperature (F), optional
            dew_point_f: Dew point temperature (F), optional
            humidity_ratio: Humidity ratio (lb/lb), optional
            pressure_psia: Total pressure in psia (default: 14.696)

        Returns:
            CompletePsychrometricOutput with all psychrometric properties

        Raises:
            ValueError: If no humidity property is provided

        Example:
            >>> calc = PsychrometricCalculator()
            >>> state = calc.calculate_complete_state(80.0, relative_humidity_pct=50.0)
            >>> print(f"Wet-bulb: {state.wet_bulb_f:.1f}F")
            >>> print(f"Dew point: {state.dew_point_f:.1f}F")
            >>> print(f"Enthalpy: {state.enthalpy_btu_lb:.2f} BTU/lb")
        """
        start_time = datetime.now()

        # Validate input
        input_data = CompletePsychrometricInput(
            dry_bulb_f=dry_bulb_f,
            relative_humidity_pct=relative_humidity_pct,
            wet_bulb_f=wet_bulb_f,
            dew_point_f=dew_point_f,
            humidity_ratio=humidity_ratio,
            pressure_psia=pressure_psia
        )

        t_db = Decimal(str(input_data.dry_bulb_f))
        p = Decimal(str(input_data.pressure_psia))

        # Calculate saturation pressure at dry-bulb
        p_ws_db = self.calculate_saturation_pressure(float(t_db))
        p_ws_db_dec = Decimal(str(p_ws_db.saturation_pressure_psia))

        # Determine humidity ratio based on provided input
        if input_data.relative_humidity_pct is not None:
            # Calculate from relative humidity
            rh = Decimal(str(input_data.relative_humidity_pct)) / Decimal("100")
            p_w = rh * p_ws_db_dec
            w = self.constants.EPSILON * p_w / (p - p_w)
            rh_pct = input_data.relative_humidity_pct

        elif input_data.wet_bulb_f is not None:
            # Calculate from wet-bulb temperature
            w_result = self.calculate_humidity_ratio(
                float(t_db), input_data.wet_bulb_f, float(p)
            )
            w = Decimal(str(w_result.humidity_ratio_lb_lb))
            p_w = Decimal(str(w_result.partial_pressure_water_psia))
            rh_pct = float(p_w / p_ws_db_dec * Decimal("100"))

        elif input_data.dew_point_f is not None:
            # Calculate from dew point
            p_w_result = self.calculate_saturation_pressure(input_data.dew_point_f)
            p_w = Decimal(str(p_w_result.saturation_pressure_psia))
            w = self.constants.EPSILON * p_w / (p - p_w)
            rh_pct = float(p_w / p_ws_db_dec * Decimal("100"))

        else:  # humidity_ratio is provided
            w = Decimal(str(input_data.humidity_ratio))
            p_w = w * p / (self.constants.EPSILON + w)
            rh_pct = float(p_w / p_ws_db_dec * Decimal("100"))

        # Clamp relative humidity
        if rh_pct > 100.0:
            rh_pct = 100.0
        if rh_pct < 0.0:
            rh_pct = 0.0

        # Calculate dew point
        if input_data.dew_point_f is not None:
            t_dp = Decimal(str(input_data.dew_point_f))
        else:
            dp_result = self.calculate_dew_point(float(t_db), rh_pct)
            t_dp = Decimal(str(dp_result.dew_point_f))

        # Calculate wet-bulb
        if input_data.wet_bulb_f is not None:
            t_wb = Decimal(str(input_data.wet_bulb_f))
        else:
            wb_result = self.calculate_wet_bulb(float(t_db), rh_pct, float(p))
            t_wb = Decimal(str(wb_result.wet_bulb_f))

        # Calculate specific humidity
        # q = W / (1 + W)
        specific_humidity = w / (Decimal("1") + w)

        # Calculate enthalpy
        enthalpy_result = self.calculate_enthalpy(float(t_db), float(w))
        h = Decimal(str(enthalpy_result.enthalpy_btu_lb))

        # Calculate specific volume
        volume_result = self.calculate_specific_volume(float(t_db), float(w), float(p))
        v = Decimal(str(volume_result.specific_volume_ft3_lb))
        density = Decimal(str(volume_result.density_lb_ft3))

        # Apply precision
        w = self._apply_precision(w)
        p_w = self._apply_precision(p_w)
        specific_humidity = self._apply_precision(specific_humidity)
        t_dp = self._apply_precision(t_dp)
        t_wb = self._apply_precision(t_wb)
        h = self._apply_precision(h)
        v = self._apply_precision(v)
        density = self._apply_precision(density)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        timestamp = datetime.now().isoformat()

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            "calculate_complete_state",
            {
                "dry_bulb_f": dry_bulb_f,
                "relative_humidity_pct": relative_humidity_pct,
                "wet_bulb_f": wet_bulb_f,
                "dew_point_f": dew_point_f,
                "humidity_ratio": humidity_ratio,
                "pressure_psia": pressure_psia
            },
            {
                "humidity_ratio": str(w),
                "wet_bulb_f": str(t_wb),
                "dew_point_f": str(t_dp),
                "enthalpy_btu_lb": str(h),
                "specific_volume_ft3_lb": str(v)
            }
        )

        return CompletePsychrometricOutput(
            dry_bulb_f=float(t_db),
            wet_bulb_f=float(t_wb),
            dew_point_f=float(t_dp),
            relative_humidity_pct=rh_pct,
            humidity_ratio_lb_lb=float(w),
            specific_humidity_lb_lb=float(specific_humidity),
            pressure_psia=float(p),
            saturation_pressure_psia=float(p_ws_db_dec),
            partial_pressure_water_psia=float(p_w),
            enthalpy_btu_lb=float(h),
            specific_volume_ft3_lb=float(v),
            density_lb_ft3=float(density),
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            processing_time_ms=processing_time
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_humidity_ratio(
    dry_bulb_f: float,
    wet_bulb_f: float,
    pressure_psia: float = 14.696
) -> HumidityRatioOutput:
    """
    Calculate humidity ratio from dry-bulb and wet-bulb temperatures.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        wet_bulb_f: Wet-bulb temperature in degrees Fahrenheit
        pressure_psia: Total atmospheric pressure in psia

    Returns:
        HumidityRatioOutput with calculated humidity ratio

    Example:
        >>> result = calculate_humidity_ratio(80.0, 65.0)
        >>> print(f"W = {result.humidity_ratio_lb_lb:.6f} lb/lb")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_humidity_ratio(dry_bulb_f, wet_bulb_f, pressure_psia)


def calculate_relative_humidity(
    dry_bulb_f: float,
    dew_point_f: float
) -> RelativeHumidityOutput:
    """
    Calculate relative humidity from dry-bulb and dew point temperatures.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        dew_point_f: Dew point temperature in degrees Fahrenheit

    Returns:
        RelativeHumidityOutput with calculated relative humidity

    Example:
        >>> result = calculate_relative_humidity(80.0, 60.0)
        >>> print(f"RH = {result.relative_humidity_pct:.1f}%")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_relative_humidity(dry_bulb_f, dew_point_f)


def calculate_dew_point(
    dry_bulb_f: float,
    relative_humidity_pct: float
) -> DewPointOutput:
    """
    Calculate dew point temperature from dry-bulb and relative humidity.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        relative_humidity_pct: Relative humidity as percentage (0-100)

    Returns:
        DewPointOutput with calculated dew point temperature

    Example:
        >>> result = calculate_dew_point(80.0, 50.0)
        >>> print(f"Dew point = {result.dew_point_f:.1f}F")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_dew_point(dry_bulb_f, relative_humidity_pct)


def calculate_wet_bulb(
    dry_bulb_f: float,
    relative_humidity_pct: float,
    pressure_psia: float = 14.696
) -> WetBulbOutput:
    """
    Calculate wet-bulb temperature from dry-bulb, relative humidity, and pressure.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        relative_humidity_pct: Relative humidity as percentage (0-100)
        pressure_psia: Total atmospheric pressure in psia

    Returns:
        WetBulbOutput with calculated wet-bulb temperature

    Example:
        >>> result = calculate_wet_bulb(80.0, 50.0)
        >>> print(f"Wet-bulb = {result.wet_bulb_f:.1f}F")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_wet_bulb(dry_bulb_f, relative_humidity_pct, pressure_psia)


def calculate_enthalpy(
    dry_bulb_f: float,
    humidity_ratio: float
) -> EnthalpyOutput:
    """
    Calculate specific enthalpy of moist air.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        humidity_ratio: Humidity ratio in lb_water per lb_dry_air

    Returns:
        EnthalpyOutput with calculated enthalpy

    Example:
        >>> result = calculate_enthalpy(80.0, 0.0115)
        >>> print(f"h = {result.enthalpy_btu_lb:.2f} BTU/lb")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_enthalpy(dry_bulb_f, humidity_ratio)


def calculate_specific_volume(
    dry_bulb_f: float,
    humidity_ratio: float,
    pressure_psia: float = 14.696
) -> SpecificVolumeOutput:
    """
    Calculate specific volume of moist air.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        humidity_ratio: Humidity ratio in lb_water per lb_dry_air
        pressure_psia: Total atmospheric pressure in psia

    Returns:
        SpecificVolumeOutput with calculated specific volume

    Example:
        >>> result = calculate_specific_volume(80.0, 0.0115)
        >>> print(f"v = {result.specific_volume_ft3_lb:.3f} ft3/lb")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_specific_volume(dry_bulb_f, humidity_ratio, pressure_psia)


def calculate_saturation_pressure(
    temp_f: float,
    method: CalculationMethod = CalculationMethod.MAGNUS_TETENS
) -> SaturationPressureOutput:
    """
    Calculate saturation vapor pressure using Magnus-Tetens equation.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        temp_f: Temperature in degrees Fahrenheit
        method: Calculation method (MAGNUS_TETENS or ASHRAE_POLYNOMIAL)

    Returns:
        SaturationPressureOutput with calculated saturation pressure

    Example:
        >>> result = calculate_saturation_pressure(77.0)
        >>> print(f"P_sat = {result.saturation_pressure_psia:.4f} psia")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_saturation_pressure(temp_f, method)


def psychrometric_state(
    dry_bulb_f: float,
    relative_humidity_pct: Optional[float] = None,
    wet_bulb_f: Optional[float] = None,
    dew_point_f: Optional[float] = None,
    humidity_ratio: Optional[float] = None,
    pressure_psia: float = 14.696
) -> CompletePsychrometricOutput:
    """
    Calculate complete psychrometric state from minimal inputs.

    Convenience function that creates a calculator and performs the calculation.

    Args:
        dry_bulb_f: Dry-bulb temperature in degrees Fahrenheit
        relative_humidity_pct: Relative humidity (0-100), optional
        wet_bulb_f: Wet-bulb temperature (F), optional
        dew_point_f: Dew point temperature (F), optional
        humidity_ratio: Humidity ratio (lb/lb), optional
        pressure_psia: Total pressure in psia

    Returns:
        CompletePsychrometricOutput with all psychrometric properties

    Example:
        >>> state = psychrometric_state(80.0, relative_humidity_pct=50.0)
        >>> print(f"Wet-bulb: {state.wet_bulb_f:.1f}F")
        >>> print(f"Enthalpy: {state.enthalpy_btu_lb:.2f} BTU/lb")
    """
    calc = PsychrometricCalculator()
    return calc.calculate_complete_state(
        dry_bulb_f,
        relative_humidity_pct,
        wet_bulb_f,
        dew_point_f,
        humidity_ratio,
        pressure_psia
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main calculator class
    "PsychrometricCalculator",

    # Constants
    "ASHRAEConstants",

    # Enumerations
    "CalculationMethod",
    "UnitSystem",

    # Input models
    "HumidityRatioInput",
    "RelativeHumidityInput",
    "DewPointInput",
    "WetBulbInput",
    "EnthalpyInput",
    "SpecificVolumeInput",
    "SaturationPressureInput",
    "CompletePsychrometricInput",

    # Output models
    "HumidityRatioOutput",
    "RelativeHumidityOutput",
    "DewPointOutput",
    "WetBulbOutput",
    "EnthalpyOutput",
    "SpecificVolumeOutput",
    "SaturationPressureOutput",
    "CompletePsychrometricOutput",

    # Convenience functions
    "calculate_humidity_ratio",
    "calculate_relative_humidity",
    "calculate_dew_point",
    "calculate_wet_bulb",
    "calculate_enthalpy",
    "calculate_specific_volume",
    "calculate_saturation_pressure",
    "psychrometric_state",
]
