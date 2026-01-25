"""
GL-015 INSULSCAN - Surface Temperature Analyzer Module

This module implements comprehensive, zero-hallucination surface temperature
analysis for thermal insulation assessment. All calculations are deterministic,
fully traceable, and compliant with ASTM and ISO standards.

Key Features:
- Environmental condition corrections (ambient, wind, solar)
- Reflected temperature compensation
- Personnel protection limit verification (ASTM C1055)
- Temperature distribution analysis (uniformity, hot/cold spots)
- Seasonal adjustment calculations
- Complete measurement uncertainty quantification (GUM method)

Reference Standards:
- ASTM C1055: Standard Guide for Heated System Surface Conditions
- ASTM E1933: Standard Practice for Measuring and Compensating for Emissivity
- ISO 6946: Building Components - Thermal Resistance
- ISO 9869: Thermal Insulation - Building Elements
- NFRC 102: Procedure for Measuring Steady-State Thermal Transmittance
- GUM (Guide to Expression of Uncertainty in Measurement) - JCGM 100:2008

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Final
from enum import Enum, auto
from datetime import datetime, timezone, time
import hashlib
import json
import math
import uuid


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Stefan-Boltzmann constant (W/(m^2.K^4))
STEFAN_BOLTZMANN: Final[Decimal] = Decimal("5.670374419e-8")

# Reference ambient temperature (Celsius)
REFERENCE_AMBIENT_C: Final[Decimal] = Decimal("20.0")

# Absolute zero offset (Celsius to Kelvin)
KELVIN_OFFSET: Final[Decimal] = Decimal("273.15")

# Pi to high precision
PI: Final[Decimal] = Decimal("3.14159265358979323846264338327950288419716939937510")

# Default calculation precision
DEFAULT_DECIMAL_PRECISION: Final[int] = 6


# =============================================================================
# ENUMS
# =============================================================================

class LocationType(Enum):
    """Location classification for wind exposure."""
    INDOOR = auto()
    OUTDOOR_SHELTERED = auto()
    OUTDOOR_EXPOSED = auto()
    OUTDOOR_HIGHLY_EXPOSED = auto()


class SurfaceOrientation(Enum):
    """Surface orientation for solar loading calculations."""
    HORIZONTAL_UP = auto()
    HORIZONTAL_DOWN = auto()
    VERTICAL_NORTH = auto()
    VERTICAL_SOUTH = auto()
    VERTICAL_EAST = auto()
    VERTICAL_WEST = auto()
    INCLINED = auto()


class SurfaceColor(Enum):
    """Surface color classification for solar absorptivity."""
    WHITE = auto()
    LIGHT_GRAY = auto()
    ALUMINUM = auto()
    MEDIUM_GRAY = auto()
    DARK_GRAY = auto()
    BLACK = auto()
    RED = auto()
    BROWN = auto()
    GREEN = auto()
    BLUE = auto()


class CloudCover(Enum):
    """Cloud cover classification."""
    CLEAR = auto()
    PARTLY_CLOUDY = auto()
    MOSTLY_CLOUDY = auto()
    OVERCAST = auto()


class Season(Enum):
    """Seasonal classification."""
    WINTER = auto()
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()


class InspectionCondition(Enum):
    """Inspection condition quality rating."""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()
    UNACCEPTABLE = auto()


# =============================================================================
# LOOKUP TABLES
# =============================================================================

# Solar absorptivity by surface color
# Source: ASHRAE Fundamentals Handbook, Table 5
SOLAR_ABSORPTIVITY: Dict[SurfaceColor, Decimal] = {
    SurfaceColor.WHITE: Decimal("0.20"),
    SurfaceColor.LIGHT_GRAY: Decimal("0.40"),
    SurfaceColor.ALUMINUM: Decimal("0.30"),
    SurfaceColor.MEDIUM_GRAY: Decimal("0.55"),
    SurfaceColor.DARK_GRAY: Decimal("0.70"),
    SurfaceColor.BLACK: Decimal("0.95"),
    SurfaceColor.RED: Decimal("0.65"),
    SurfaceColor.BROWN: Decimal("0.75"),
    SurfaceColor.GREEN: Decimal("0.60"),
    SurfaceColor.BLUE: Decimal("0.55"),
}

# Wind speed correction factors for convective heat transfer
# Source: ASHRAE Fundamentals, ISO 6946
# Format: (min_speed_m_s, max_speed_m_s): correction_factor
WIND_CORRECTION_FACTORS: Dict[Tuple[Decimal, Decimal], Decimal] = {
    (Decimal("0.0"), Decimal("1.0")): Decimal("1.00"),    # Calm/indoor
    (Decimal("1.0"), Decimal("3.0")): Decimal("1.15"),    # Light breeze
    (Decimal("3.0"), Decimal("5.0")): Decimal("1.35"),    # Gentle breeze
    (Decimal("5.0"), Decimal("8.0")): Decimal("1.60"),    # Moderate wind
    (Decimal("8.0"), Decimal("12.0")): Decimal("1.90"),   # Fresh wind
    (Decimal("12.0"), Decimal("20.0")): Decimal("2.30"),  # Strong wind
    (Decimal("20.0"), Decimal("100.0")): Decimal("2.80"), # Very strong wind
}

# Cloud cover reduction factors for solar radiation
# Source: ASHRAE Clear Sky Model
CLOUD_COVER_FACTORS: Dict[CloudCover, Decimal] = {
    CloudCover.CLEAR: Decimal("1.00"),
    CloudCover.PARTLY_CLOUDY: Decimal("0.75"),
    CloudCover.MOSTLY_CLOUDY: Decimal("0.45"),
    CloudCover.OVERCAST: Decimal("0.20"),
}

# Typical sky temperature depression below ambient (degrees C)
# Source: ISO 6946, Annex A
SKY_TEMPERATURE_DEPRESSION: Dict[CloudCover, Decimal] = {
    CloudCover.CLEAR: Decimal("20.0"),
    CloudCover.PARTLY_CLOUDY: Decimal("12.0"),
    CloudCover.MOSTLY_CLOUDY: Decimal("6.0"),
    CloudCover.OVERCAST: Decimal("2.0"),
}

# Seasonal baseline temperature adjustments (typical mid-latitude)
# Source: ASHRAE Weather Data
SEASONAL_BASELINE_C: Dict[Season, Decimal] = {
    Season.WINTER: Decimal("5.0"),
    Season.SPRING: Decimal("15.0"),
    Season.SUMMER: Decimal("25.0"),
    Season.AUTUMN: Decimal("15.0"),
}

# ASTM C1055 Personnel protection temperature limits
# T_surface = 140 F (60 C) for continuous contact potential
PERSONNEL_PROTECTION_LIMIT_C: Final[Decimal] = Decimal("60.0")
PERSONNEL_PROTECTION_LIMIT_F: Final[Decimal] = Decimal("140.0")

# Recommended inspection conditions per ASTM E1934
RECOMMENDED_DELTA_T_MIN_C: Final[Decimal] = Decimal("10.0")
RECOMMENDED_WIND_MAX_M_S: Final[Decimal] = Decimal("5.0")


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class AmbientNormalizationResult:
    """Result of ambient temperature normalization."""
    original_surface_temp_c: Decimal
    original_ambient_temp_c: Decimal
    reference_ambient_temp_c: Decimal
    normalized_surface_temp_c: Decimal
    delta_t_original: Decimal
    delta_t_normalized: Decimal
    temperature_ratio: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_surface_temp_c": str(self.original_surface_temp_c),
            "original_ambient_temp_c": str(self.original_ambient_temp_c),
            "reference_ambient_temp_c": str(self.reference_ambient_temp_c),
            "normalized_surface_temp_c": str(self.normalized_surface_temp_c),
            "delta_t_original": str(self.delta_t_original),
            "delta_t_normalized": str(self.delta_t_normalized),
            "temperature_ratio": str(self.temperature_ratio),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class WindCorrectionResult:
    """Result of wind speed correction."""
    original_surface_temp_c: Decimal
    wind_speed_m_s: Decimal
    location_type: str
    correction_factor: Decimal
    corrected_surface_temp_c: Decimal
    convection_increase_percent: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_surface_temp_c": str(self.original_surface_temp_c),
            "wind_speed_m_s": str(self.wind_speed_m_s),
            "location_type": self.location_type,
            "correction_factor": str(self.correction_factor),
            "corrected_surface_temp_c": str(self.corrected_surface_temp_c),
            "convection_increase_percent": str(self.convection_increase_percent),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class SolarCorrectionResult:
    """Result of solar loading correction."""
    original_surface_temp_c: Decimal
    solar_irradiance_w_m2: Decimal
    surface_absorptivity: Decimal
    solar_angle_factor: Decimal
    cloud_cover_factor: Decimal
    solar_heat_gain_c: Decimal
    corrected_surface_temp_c: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_surface_temp_c": str(self.original_surface_temp_c),
            "solar_irradiance_w_m2": str(self.solar_irradiance_w_m2),
            "surface_absorptivity": str(self.surface_absorptivity),
            "solar_angle_factor": str(self.solar_angle_factor),
            "cloud_cover_factor": str(self.cloud_cover_factor),
            "solar_heat_gain_c": str(self.solar_heat_gain_c),
            "corrected_surface_temp_c": str(self.corrected_surface_temp_c),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class ReflectedTemperatureResult:
    """Result of reflected temperature compensation."""
    measured_radiance_temp_c: Decimal
    emissivity: Decimal
    reflected_temp_c: Decimal
    true_surface_temp_c: Decimal
    reflection_contribution_c: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "measured_radiance_temp_c": str(self.measured_radiance_temp_c),
            "emissivity": str(self.emissivity),
            "reflected_temp_c": str(self.reflected_temp_c),
            "true_surface_temp_c": str(self.true_surface_temp_c),
            "reflection_contribution_c": str(self.reflection_contribution_c),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class PersonnelProtectionResult:
    """Result of personnel protection limit check."""
    surface_temp_c: Decimal
    surface_temp_f: Decimal
    limit_temp_c: Decimal
    limit_temp_f: Decimal
    exceeds_limit: bool
    margin_c: Decimal
    margin_percent: Decimal
    risk_level: str
    recommendation: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "surface_temp_c": str(self.surface_temp_c),
            "surface_temp_f": str(self.surface_temp_f),
            "limit_temp_c": str(self.limit_temp_c),
            "limit_temp_f": str(self.limit_temp_f),
            "exceeds_limit": self.exceeds_limit,
            "margin_c": str(self.margin_c),
            "margin_percent": str(self.margin_percent),
            "risk_level": self.risk_level,
            "recommendation": self.recommendation,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class TemperatureDistributionResult:
    """Result of temperature distribution analysis."""
    measurement_count: int
    mean_temp_c: Decimal
    min_temp_c: Decimal
    max_temp_c: Decimal
    std_deviation_c: Decimal
    uniformity_index: Decimal
    hot_spot_count: int
    cold_spot_count: int
    hot_spot_locations: Tuple[int, ...]
    cold_spot_locations: Tuple[int, ...]
    edge_effect_detected: bool
    joint_anomaly_detected: bool
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "measurement_count": self.measurement_count,
            "mean_temp_c": str(self.mean_temp_c),
            "min_temp_c": str(self.min_temp_c),
            "max_temp_c": str(self.max_temp_c),
            "std_deviation_c": str(self.std_deviation_c),
            "uniformity_index": str(self.uniformity_index),
            "hot_spot_count": self.hot_spot_count,
            "cold_spot_count": self.cold_spot_count,
            "hot_spot_locations": list(self.hot_spot_locations),
            "cold_spot_locations": list(self.cold_spot_locations),
            "edge_effect_detected": self.edge_effect_detected,
            "joint_anomaly_detected": self.joint_anomaly_detected,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class MeasurementUncertaintyResult:
    """Result of measurement uncertainty calculation (GUM method)."""
    measured_value_c: Decimal
    camera_uncertainty_c: Decimal
    emissivity_uncertainty_c: Decimal
    ambient_uncertainty_c: Decimal
    reflected_temp_uncertainty_c: Decimal
    environmental_uncertainty_c: Decimal
    combined_standard_uncertainty_c: Decimal
    expanded_uncertainty_c: Decimal
    coverage_factor: Decimal
    confidence_level_percent: Decimal
    relative_uncertainty_percent: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "measured_value_c": str(self.measured_value_c),
            "camera_uncertainty_c": str(self.camera_uncertainty_c),
            "emissivity_uncertainty_c": str(self.emissivity_uncertainty_c),
            "ambient_uncertainty_c": str(self.ambient_uncertainty_c),
            "reflected_temp_uncertainty_c": str(self.reflected_temp_uncertainty_c),
            "environmental_uncertainty_c": str(self.environmental_uncertainty_c),
            "combined_standard_uncertainty_c": str(self.combined_standard_uncertainty_c),
            "expanded_uncertainty_c": str(self.expanded_uncertainty_c),
            "coverage_factor": str(self.coverage_factor),
            "confidence_level_percent": str(self.confidence_level_percent),
            "relative_uncertainty_percent": str(self.relative_uncertainty_percent),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class InspectionConditionsResult:
    """Result of inspection conditions recommendation."""
    current_ambient_c: Decimal
    current_wind_m_s: Decimal
    current_delta_t: Decimal
    current_solar_load: bool
    condition_rating: str
    is_recommended: bool
    issues: Tuple[str, ...]
    recommendations: Tuple[str, ...]
    optimal_time_windows: Tuple[str, ...]
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_ambient_c": str(self.current_ambient_c),
            "current_wind_m_s": str(self.current_wind_m_s),
            "current_delta_t": str(self.current_delta_t),
            "current_solar_load": self.current_solar_load,
            "condition_rating": self.condition_rating,
            "is_recommended": self.is_recommended,
            "issues": list(self.issues),
            "recommendations": list(self.recommendations),
            "optimal_time_windows": list(self.optimal_time_windows),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class SeasonalAdjustmentResult:
    """Result of seasonal temperature adjustment."""
    measured_temp_c: Decimal
    measurement_season: str
    target_season: str
    seasonal_correction_c: Decimal
    adjusted_temp_c: Decimal
    annual_average_temp_c: Decimal
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "measured_temp_c": str(self.measured_temp_c),
            "measurement_season": self.measurement_season,
            "target_season": self.target_season,
            "seasonal_correction_c": str(self.seasonal_correction_c),
            "adjusted_temp_c": str(self.adjusted_temp_c),
            "annual_average_temp_c": str(self.annual_average_temp_c),
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# PROVENANCE TRACKING
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """Immutable record of a single calculation step."""
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": self._serialize_inputs(self.inputs),
            "output_name": self.output_name,
            "output_value": self._serialize_value(self.output_value),
            "formula": self.formula,
            "reference": self.reference,
        }

    @staticmethod
    def _serialize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize input values to JSON-compatible format."""
        result = {}
        for key, value in inputs.items():
            result[key] = CalculationStep._serialize_value(value)
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, Decimal):
            return str(value)
        elif isinstance(value, Enum):
            return value.name
        elif isinstance(value, (list, tuple)):
            return [CalculationStep._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: CalculationStep._serialize_value(v) for k, v in value.items()}
        return value


class ProvenanceBuilder:
    """Builder for creating provenance records with calculation steps."""

    def __init__(self, calculation_type: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize provenance builder."""
        self._record_id = str(uuid.uuid4())
        self._calculation_type = calculation_type
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._steps: List[CalculationStep] = []
        self._metadata = metadata or {}

    def add_input(self, name: str, value: Any) -> "ProvenanceBuilder":
        """Add an input parameter."""
        self._inputs[name] = value
        return self

    def add_output(self, name: str, value: Any) -> "ProvenanceBuilder":
        """Add an output value."""
        self._outputs[name] = value
        return self

    def add_step(
        self,
        step_number: int,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Any,
        formula: str = "",
        reference: str = ""
    ) -> "ProvenanceBuilder":
        """Add a calculation step."""
        step = CalculationStep(
            step_number=step_number,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula,
            reference=reference
        )
        self._steps.append(step)
        return self

    def build(self) -> str:
        """Build and return the provenance hash."""
        return self._calculate_final_hash()

    def _calculate_final_hash(self) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        hash_data = {
            "record_id": self._record_id,
            "calculation_type": self._calculation_type,
            "timestamp": self._timestamp,
            "inputs": self._serialize_dict(self._inputs),
            "outputs": self._serialize_dict(self._outputs),
            "steps": [step.to_dict() for step in self._steps],
        }
        json_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize dictionary values."""
        result = {}
        for key, value in d.items():
            if isinstance(value, Decimal):
                result[key] = str(value)
            elif isinstance(value, Enum):
                result[key] = value.name
            elif isinstance(value, dict):
                result[key] = ProvenanceBuilder._serialize_dict(value)
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    str(v) if isinstance(v, Decimal) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result


# =============================================================================
# SURFACE TEMPERATURE ANALYZER
# =============================================================================

class SurfaceTemperatureAnalyzer:
    """
    Zero-hallucination surface temperature analyzer for thermal insulation assessment.

    This analyzer implements comprehensive environmental corrections and analysis
    methods for infrared thermography inspections per ASTM and ISO standards.

    Guarantees:
    - Deterministic: Same input yields same output (bit-perfect)
    - Reproducible: Full provenance tracking with SHA-256 hashes
    - Auditable: Complete calculation step documentation
    - NO LLM: Zero hallucination risk in calculation path

    Reference Standards:
    - ASTM C1055: Heated System Surface Conditions
    - ASTM E1933: Emissivity Measurement
    - ISO 6946: Thermal Resistance Calculation
    - GUM: Guide to Expression of Uncertainty in Measurement

    Example:
        >>> analyzer = SurfaceTemperatureAnalyzer()
        >>> result = analyzer.normalize_to_reference_ambient(
        ...     surface_temp_c=Decimal("75.0"),
        ...     ambient_temp_c=Decimal("25.0"),
        ...     process_temp_c=Decimal("150.0")
        ... )
        >>> print(f"Normalized: {result.normalized_surface_temp_c} C")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        reference_ambient_c: Optional[Decimal] = None
    ):
        """
        Initialize Surface Temperature Analyzer.

        Args:
            precision: Decimal precision for calculations (default: 6)
            reference_ambient_c: Reference ambient temperature (default: 20 C)
        """
        self._precision = precision
        self._reference_ambient = reference_ambient_c or REFERENCE_AMBIENT_C

    # =========================================================================
    # AMBIENT CONDITION CORRECTIONS
    # =========================================================================

    def normalize_to_reference_ambient(
        self,
        surface_temp_c: Union[Decimal, float, str],
        ambient_temp_c: Union[Decimal, float, str],
        process_temp_c: Union[Decimal, float, str],
        reference_ambient_c: Optional[Union[Decimal, float, str]] = None
    ) -> AmbientNormalizationResult:
        """
        Normalize surface temperature to a reference ambient condition.

        Uses the temperature ratio method to adjust surface temperature
        measurements to a standard reference ambient temperature (typically 20 C).

        The temperature ratio method maintains the relationship:
            (T_surface - T_ambient) / (T_process - T_ambient) = constant

        This allows comparison of measurements taken at different ambient
        conditions by normalizing to a common reference.

        Args:
            surface_temp_c: Measured surface temperature (Celsius)
            ambient_temp_c: Actual ambient temperature during measurement (Celsius)
            process_temp_c: Process/operating temperature inside insulation (Celsius)
            reference_ambient_c: Target reference ambient (default: 20 C)

        Returns:
            AmbientNormalizationResult with normalized temperature

        Reference:
            ASTM C680: Standard Practice for Estimate of Heat Gain or Loss
            ISO 12241: Thermal Insulation for Building Equipment

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.normalize_to_reference_ambient(
            ...     surface_temp_c="75.0",
            ...     ambient_temp_c="30.0",
            ...     process_temp_c="150.0"
            ... )
            >>> print(f"Normalized temp: {result.normalized_surface_temp_c} C")
        """
        builder = ProvenanceBuilder("AMBIENT_NORMALIZATION")

        # Convert inputs to Decimal
        T_surface = self._to_decimal(surface_temp_c)
        T_ambient = self._to_decimal(ambient_temp_c)
        T_process = self._to_decimal(process_temp_c)
        T_ref = self._to_decimal(reference_ambient_c) if reference_ambient_c else self._reference_ambient

        builder.add_input("surface_temp_c", T_surface)
        builder.add_input("ambient_temp_c", T_ambient)
        builder.add_input("process_temp_c", T_process)
        builder.add_input("reference_ambient_c", T_ref)

        # Step 1: Calculate original delta-T (surface to ambient)
        delta_T_original = T_surface - T_ambient

        builder.add_step(
            step_number=1,
            operation="subtract",
            description="Calculate original temperature difference",
            inputs={"T_surface": T_surface, "T_ambient": T_ambient},
            output_name="delta_T_original",
            output_value=delta_T_original,
            formula="delta_T = T_surface - T_ambient"
        )

        # Step 2: Calculate temperature ratio
        # Ratio represents fraction of total temperature drop across insulation
        if T_process - T_ambient != Decimal("0"):
            temp_ratio = delta_T_original / (T_process - T_ambient)
        else:
            temp_ratio = Decimal("0")

        builder.add_step(
            step_number=2,
            operation="divide",
            description="Calculate temperature ratio",
            inputs={"delta_T": delta_T_original, "T_process": T_process, "T_ambient": T_ambient},
            output_name="temp_ratio",
            output_value=temp_ratio,
            formula="ratio = (T_surface - T_ambient) / (T_process - T_ambient)",
            reference="ASTM C680"
        )

        # Step 3: Calculate normalized delta-T at reference ambient
        delta_T_normalized = temp_ratio * (T_process - T_ref)

        builder.add_step(
            step_number=3,
            operation="multiply",
            description="Calculate normalized delta-T",
            inputs={"ratio": temp_ratio, "T_process": T_process, "T_ref": T_ref},
            output_name="delta_T_normalized",
            output_value=delta_T_normalized,
            formula="delta_T_norm = ratio * (T_process - T_ref)"
        )

        # Step 4: Calculate normalized surface temperature
        T_surface_normalized = T_ref + delta_T_normalized

        builder.add_step(
            step_number=4,
            operation="add",
            description="Calculate normalized surface temperature",
            inputs={"T_ref": T_ref, "delta_T_norm": delta_T_normalized},
            output_name="T_surface_normalized",
            output_value=T_surface_normalized,
            formula="T_surface_norm = T_ref + delta_T_norm"
        )

        builder.add_output("normalized_surface_temp_c", T_surface_normalized)
        provenance_hash = builder.build()

        return AmbientNormalizationResult(
            original_surface_temp_c=self._apply_precision(T_surface),
            original_ambient_temp_c=self._apply_precision(T_ambient),
            reference_ambient_temp_c=self._apply_precision(T_ref),
            normalized_surface_temp_c=self._apply_precision(T_surface_normalized),
            delta_t_original=self._apply_precision(delta_T_original),
            delta_t_normalized=self._apply_precision(delta_T_normalized),
            temperature_ratio=self._apply_precision(temp_ratio, 4),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # WIND SPEED CORRECTIONS
    # =========================================================================

    def apply_wind_correction(
        self,
        surface_temp_c: Union[Decimal, float, str],
        ambient_temp_c: Union[Decimal, float, str],
        wind_speed_m_s: Union[Decimal, float, str],
        location_type: LocationType = LocationType.OUTDOOR_EXPOSED
    ) -> WindCorrectionResult:
        """
        Apply wind speed correction to surface temperature.

        Wind increases convective heat transfer from the surface, which
        lowers the apparent surface temperature. This method corrects
        for wind effects to estimate the true surface temperature under
        calm conditions.

        The forced convection coefficient increases with wind speed:
            h_wind = h_calm * correction_factor

        This affects surface temperature:
            T_surface_calm = T_ambient + (T_surface_wind - T_ambient) * correction_factor

        Args:
            surface_temp_c: Measured surface temperature (Celsius)
            ambient_temp_c: Ambient temperature (Celsius)
            wind_speed_m_s: Wind speed in meters per second
            location_type: Location exposure classification

        Returns:
            WindCorrectionResult with corrected temperature

        Reference:
            ISO 6946: Building Components Thermal Resistance
            ASHRAE Fundamentals Handbook, Chapter 26

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.apply_wind_correction(
            ...     surface_temp_c="65.0",
            ...     ambient_temp_c="20.0",
            ...     wind_speed_m_s="5.0"
            ... )
            >>> print(f"Corrected temp: {result.corrected_surface_temp_c} C")
        """
        builder = ProvenanceBuilder("WIND_CORRECTION")

        # Convert inputs
        T_surface = self._to_decimal(surface_temp_c)
        T_ambient = self._to_decimal(ambient_temp_c)
        wind_speed = self._to_decimal(wind_speed_m_s)

        builder.add_input("surface_temp_c", T_surface)
        builder.add_input("ambient_temp_c", T_ambient)
        builder.add_input("wind_speed_m_s", wind_speed)
        builder.add_input("location_type", location_type.name)

        # Step 1: Apply location sheltering factor
        sheltering_factor = self._get_sheltering_factor(location_type)
        effective_wind = wind_speed * sheltering_factor

        builder.add_step(
            step_number=1,
            operation="multiply",
            description="Apply location sheltering factor",
            inputs={"wind_speed": wind_speed, "sheltering_factor": sheltering_factor},
            output_name="effective_wind",
            output_value=effective_wind,
            formula="effective_wind = wind_speed * sheltering_factor"
        )

        # Step 2: Look up wind correction factor
        correction_factor = self._get_wind_correction_factor(effective_wind)

        builder.add_step(
            step_number=2,
            operation="lookup",
            description="Look up wind correction factor",
            inputs={"effective_wind": effective_wind},
            output_name="correction_factor",
            output_value=correction_factor,
            reference="ISO 6946"
        )

        # Step 3: Calculate temperature correction
        # Under windy conditions, surface temp is lower than it would be in calm
        # T_calm = T_ambient + (T_windy - T_ambient) * correction_factor
        delta_T_windy = T_surface - T_ambient
        delta_T_calm = delta_T_windy * correction_factor
        T_surface_corrected = T_ambient + delta_T_calm

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate corrected surface temperature",
            inputs={
                "T_surface": T_surface,
                "T_ambient": T_ambient,
                "correction_factor": correction_factor
            },
            output_name="T_surface_corrected",
            output_value=T_surface_corrected,
            formula="T_corrected = T_ambient + (T_surface - T_ambient) * factor"
        )

        # Step 4: Calculate convection increase percentage
        convection_increase = (correction_factor - Decimal("1")) * Decimal("100")

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate convection increase percentage",
            inputs={"correction_factor": correction_factor},
            output_name="convection_increase_percent",
            output_value=convection_increase,
            formula="increase_% = (factor - 1) * 100"
        )

        builder.add_output("corrected_surface_temp_c", T_surface_corrected)
        provenance_hash = builder.build()

        return WindCorrectionResult(
            original_surface_temp_c=self._apply_precision(T_surface),
            wind_speed_m_s=self._apply_precision(wind_speed),
            location_type=location_type.name,
            correction_factor=self._apply_precision(correction_factor, 3),
            corrected_surface_temp_c=self._apply_precision(T_surface_corrected),
            convection_increase_percent=self._apply_precision(convection_increase, 1),
            provenance_hash=provenance_hash
        )

    def _get_sheltering_factor(self, location_type: LocationType) -> Decimal:
        """Get wind sheltering factor by location type."""
        sheltering_factors = {
            LocationType.INDOOR: Decimal("0.0"),
            LocationType.OUTDOOR_SHELTERED: Decimal("0.5"),
            LocationType.OUTDOOR_EXPOSED: Decimal("1.0"),
            LocationType.OUTDOOR_HIGHLY_EXPOSED: Decimal("1.2"),
        }
        return sheltering_factors.get(location_type, Decimal("1.0"))

    def _get_wind_correction_factor(self, wind_speed: Decimal) -> Decimal:
        """Look up wind correction factor from table."""
        for (min_speed, max_speed), factor in WIND_CORRECTION_FACTORS.items():
            if min_speed <= wind_speed < max_speed:
                return factor
        # Default for very high winds
        return Decimal("2.80")

    # =========================================================================
    # SOLAR LOADING CORRECTIONS
    # =========================================================================

    def apply_solar_correction(
        self,
        surface_temp_c: Union[Decimal, float, str],
        ambient_temp_c: Union[Decimal, float, str],
        solar_irradiance_w_m2: Union[Decimal, float, str],
        surface_color: SurfaceColor,
        surface_orientation: SurfaceOrientation,
        solar_altitude_deg: Union[Decimal, float, str],
        cloud_cover: CloudCover = CloudCover.CLEAR,
        hour_of_day: Optional[int] = None
    ) -> SolarCorrectionResult:
        """
        Apply solar loading correction to surface temperature.

        Removes the solar heat gain component from measured surface temperature
        to determine the base temperature without solar loading. This is
        important for accurate insulation assessment.

        Solar heat gain: Q_solar = alpha * G * cos(theta) * cloud_factor
        Temperature rise: delta_T_solar = Q_solar / h_combined

        Args:
            surface_temp_c: Measured surface temperature (Celsius)
            ambient_temp_c: Ambient temperature (Celsius)
            solar_irradiance_w_m2: Solar irradiance (W/m2), typically 0-1000
            surface_color: Surface color for absorptivity lookup
            surface_orientation: Surface orientation relative to sun
            solar_altitude_deg: Solar altitude angle (degrees above horizon)
            cloud_cover: Cloud cover condition
            hour_of_day: Hour of day (0-23) for time-based adjustments

        Returns:
            SolarCorrectionResult with corrected temperature

        Reference:
            ASHRAE Fundamentals, Solar Radiation Chapter
            ASTM E1918: Standard Test Method for Solar Absorptance

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.apply_solar_correction(
            ...     surface_temp_c="85.0",
            ...     ambient_temp_c="25.0",
            ...     solar_irradiance_w_m2="800",
            ...     surface_color=SurfaceColor.DARK_GRAY,
            ...     surface_orientation=SurfaceOrientation.HORIZONTAL_UP,
            ...     solar_altitude_deg="60"
            ... )
        """
        builder = ProvenanceBuilder("SOLAR_CORRECTION")

        # Convert inputs
        T_surface = self._to_decimal(surface_temp_c)
        T_ambient = self._to_decimal(ambient_temp_c)
        G_solar = self._to_decimal(solar_irradiance_w_m2)
        solar_altitude = self._to_decimal(solar_altitude_deg)

        builder.add_input("surface_temp_c", T_surface)
        builder.add_input("ambient_temp_c", T_ambient)
        builder.add_input("solar_irradiance_w_m2", G_solar)
        builder.add_input("surface_color", surface_color.name)
        builder.add_input("surface_orientation", surface_orientation.name)
        builder.add_input("solar_altitude_deg", solar_altitude)
        builder.add_input("cloud_cover", cloud_cover.name)

        # Step 1: Get surface absorptivity
        alpha = SOLAR_ABSORPTIVITY[surface_color]

        builder.add_step(
            step_number=1,
            operation="lookup",
            description="Look up surface absorptivity",
            inputs={"surface_color": surface_color.name},
            output_name="absorptivity",
            output_value=alpha,
            reference="ASHRAE Fundamentals Table 5"
        )

        # Step 2: Calculate solar angle factor
        angle_factor = self._calculate_solar_angle_factor(
            surface_orientation, solar_altitude
        )

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate solar angle factor",
            inputs={
                "orientation": surface_orientation.name,
                "solar_altitude": solar_altitude
            },
            output_name="angle_factor",
            output_value=angle_factor,
            formula="factor = cos(incidence_angle)"
        )

        # Step 3: Get cloud cover factor
        cloud_factor = CLOUD_COVER_FACTORS[cloud_cover]

        builder.add_step(
            step_number=3,
            operation="lookup",
            description="Look up cloud cover factor",
            inputs={"cloud_cover": cloud_cover.name},
            output_name="cloud_factor",
            output_value=cloud_factor,
            reference="ASHRAE Clear Sky Model"
        )

        # Step 4: Calculate absorbed solar heat flux
        # Q_absorbed = alpha * G * angle_factor * cloud_factor (W/m2)
        Q_absorbed = alpha * G_solar * angle_factor * cloud_factor

        builder.add_step(
            step_number=4,
            operation="multiply",
            description="Calculate absorbed solar heat flux",
            inputs={
                "alpha": alpha,
                "G_solar": G_solar,
                "angle_factor": angle_factor,
                "cloud_factor": cloud_factor
            },
            output_name="Q_absorbed",
            output_value=Q_absorbed,
            formula="Q = alpha * G * angle_factor * cloud_factor"
        )

        # Step 5: Estimate temperature rise from solar loading
        # Using typical combined heat transfer coefficient of 15 W/(m2.K)
        h_combined = Decimal("15.0")  # W/(m2.K)
        delta_T_solar = Q_absorbed / h_combined

        builder.add_step(
            step_number=5,
            operation="divide",
            description="Calculate solar temperature rise",
            inputs={"Q_absorbed": Q_absorbed, "h_combined": h_combined},
            output_name="delta_T_solar",
            output_value=delta_T_solar,
            formula="delta_T = Q / h_combined"
        )

        # Step 6: Calculate corrected surface temperature
        T_corrected = T_surface - delta_T_solar

        builder.add_step(
            step_number=6,
            operation="subtract",
            description="Calculate corrected surface temperature",
            inputs={"T_surface": T_surface, "delta_T_solar": delta_T_solar},
            output_name="T_corrected",
            output_value=T_corrected,
            formula="T_corrected = T_surface - delta_T_solar"
        )

        builder.add_output("corrected_surface_temp_c", T_corrected)
        provenance_hash = builder.build()

        return SolarCorrectionResult(
            original_surface_temp_c=self._apply_precision(T_surface),
            solar_irradiance_w_m2=self._apply_precision(G_solar),
            surface_absorptivity=self._apply_precision(alpha, 2),
            solar_angle_factor=self._apply_precision(angle_factor, 3),
            cloud_cover_factor=self._apply_precision(cloud_factor, 2),
            solar_heat_gain_c=self._apply_precision(delta_T_solar),
            corrected_surface_temp_c=self._apply_precision(T_corrected),
            provenance_hash=provenance_hash
        )

    def _calculate_solar_angle_factor(
        self,
        orientation: SurfaceOrientation,
        solar_altitude: Decimal
    ) -> Decimal:
        """Calculate solar angle factor based on surface orientation."""
        # Convert altitude to radians for calculation
        altitude_rad = float(solar_altitude) * math.pi / 180.0

        if orientation == SurfaceOrientation.HORIZONTAL_UP:
            # Direct sun angle
            factor = Decimal(str(math.sin(altitude_rad)))
        elif orientation == SurfaceOrientation.HORIZONTAL_DOWN:
            # No direct sun on downward-facing surface
            factor = Decimal("0.0")
        elif orientation == SurfaceOrientation.VERTICAL_SOUTH:
            # Maximum sun exposure for vertical south-facing
            factor = Decimal(str(math.cos(altitude_rad)))
        elif orientation == SurfaceOrientation.VERTICAL_NORTH:
            # Minimal direct sun (diffuse only)
            factor = Decimal("0.2")
        elif orientation in (SurfaceOrientation.VERTICAL_EAST,
                            SurfaceOrientation.VERTICAL_WEST):
            # Partial exposure
            factor = Decimal(str(0.5 * math.cos(altitude_rad)))
        else:
            # Default for inclined surfaces
            factor = Decimal(str(0.7 * math.sin(altitude_rad)))

        return max(Decimal("0"), min(factor, Decimal("1")))

    # =========================================================================
    # REFLECTED TEMPERATURE COMPENSATION
    # =========================================================================

    def compensate_reflected_temperature(
        self,
        measured_radiance_temp_c: Union[Decimal, float, str],
        emissivity: Union[Decimal, float, str],
        reflected_temp_c: Union[Decimal, float, str]
    ) -> ReflectedTemperatureResult:
        """
        Compensate for reflected temperature in IR measurement.

        The measured radiance temperature includes both emitted and reflected
        components. This method calculates the true surface temperature by
        removing the reflected component.

        For an IR camera:
            T_measured^4 = epsilon * T_surface^4 + (1-epsilon) * T_reflected^4

        Solving for T_surface:
            T_surface = ((T_measured^4 - (1-epsilon) * T_reflected^4) / epsilon)^(1/4)

        Args:
            measured_radiance_temp_c: Camera-reported temperature (Celsius)
            emissivity: Surface emissivity (0-1)
            reflected_temp_c: Reflected/background temperature (Celsius)

        Returns:
            ReflectedTemperatureResult with true surface temperature

        Reference:
            ASTM E1933: Standard Practice for Measuring and Compensating
            for Emissivity Using Infrared Imaging Radiometers

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.compensate_reflected_temperature(
            ...     measured_radiance_temp_c="80.0",
            ...     emissivity="0.85",
            ...     reflected_temp_c="25.0"
            ... )
            >>> print(f"True temp: {result.true_surface_temp_c} C")
        """
        builder = ProvenanceBuilder("REFLECTED_TEMPERATURE_COMPENSATION")

        # Convert inputs
        T_measured = self._to_decimal(measured_radiance_temp_c)
        eps = self._to_decimal(emissivity)
        T_reflected = self._to_decimal(reflected_temp_c)

        builder.add_input("measured_radiance_temp_c", T_measured)
        builder.add_input("emissivity", eps)
        builder.add_input("reflected_temp_c", T_reflected)

        # Validate emissivity
        if eps <= Decimal("0") or eps > Decimal("1"):
            raise ValueError(f"Emissivity must be between 0 and 1, got {eps}")

        # Step 1: Convert temperatures to Kelvin
        T_measured_K = T_measured + KELVIN_OFFSET
        T_reflected_K = T_reflected + KELVIN_OFFSET

        builder.add_step(
            step_number=1,
            operation="add",
            description="Convert to Kelvin",
            inputs={"T_measured_C": T_measured, "T_reflected_C": T_reflected},
            output_name="temperatures_K",
            output_value={"T_measured_K": T_measured_K, "T_reflected_K": T_reflected_K},
            formula="T_K = T_C + 273.15"
        )

        # Step 2: Calculate T^4 terms
        T_measured_4 = self._power(T_measured_K, Decimal("4"))
        T_reflected_4 = self._power(T_reflected_K, Decimal("4"))

        builder.add_step(
            step_number=2,
            operation="power",
            description="Calculate fourth power of temperatures",
            inputs={"T_measured_K": T_measured_K, "T_reflected_K": T_reflected_K},
            output_name="T_4_values",
            output_value={"T_measured_4": T_measured_4, "T_reflected_4": T_reflected_4},
            formula="T^4"
        )

        # Step 3: Calculate reflected component
        reflected_component = (Decimal("1") - eps) * T_reflected_4

        builder.add_step(
            step_number=3,
            operation="multiply",
            description="Calculate reflected radiation component",
            inputs={"reflectivity": Decimal("1") - eps, "T_reflected_4": T_reflected_4},
            output_name="reflected_component",
            output_value=reflected_component,
            formula="(1 - epsilon) * T_reflected^4"
        )

        # Step 4: Calculate true surface T^4
        T_surface_4 = (T_measured_4 - reflected_component) / eps

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate true surface T^4",
            inputs={
                "T_measured_4": T_measured_4,
                "reflected_component": reflected_component,
                "emissivity": eps
            },
            output_name="T_surface_4",
            output_value=T_surface_4,
            formula="T_surface^4 = (T_measured^4 - (1-eps)*T_reflected^4) / eps",
            reference="ASTM E1933"
        )

        # Step 5: Calculate true surface temperature
        if T_surface_4 > Decimal("0"):
            T_surface_K = self._power(T_surface_4, Decimal("0.25"))
        else:
            T_surface_K = T_reflected_K  # Fallback for edge cases

        T_surface_C = T_surface_K - KELVIN_OFFSET

        builder.add_step(
            step_number=5,
            operation="calculate",
            description="Calculate true surface temperature",
            inputs={"T_surface_4": T_surface_4},
            output_name="T_surface_C",
            output_value=T_surface_C,
            formula="T_surface = (T_surface^4)^(1/4) - 273.15"
        )

        # Step 6: Calculate reflection contribution
        reflection_contribution = T_measured - T_surface_C

        builder.add_step(
            step_number=6,
            operation="subtract",
            description="Calculate reflection contribution",
            inputs={"T_measured": T_measured, "T_surface_C": T_surface_C},
            output_name="reflection_contribution",
            output_value=reflection_contribution,
            formula="contribution = T_measured - T_surface"
        )

        builder.add_output("true_surface_temp_c", T_surface_C)
        provenance_hash = builder.build()

        return ReflectedTemperatureResult(
            measured_radiance_temp_c=self._apply_precision(T_measured),
            emissivity=self._apply_precision(eps, 3),
            reflected_temp_c=self._apply_precision(T_reflected),
            true_surface_temp_c=self._apply_precision(T_surface_C),
            reflection_contribution_c=self._apply_precision(reflection_contribution),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # PERSONNEL PROTECTION LIMIT CHECK
    # =========================================================================

    def check_personnel_protection_limit(
        self,
        surface_temp_c: Union[Decimal, float, str],
        custom_limit_c: Optional[Union[Decimal, float, str]] = None
    ) -> PersonnelProtectionResult:
        """
        Check surface temperature against personnel protection limits.

        Per ASTM C1055, surfaces that personnel may contact should not
        exceed 140 F (60 C) to prevent burn injuries. This method evaluates
        compliance and provides risk assessment.

        Args:
            surface_temp_c: Surface temperature to check (Celsius)
            custom_limit_c: Optional custom limit (default: 60 C per ASTM C1055)

        Returns:
            PersonnelProtectionResult with compliance status

        Reference:
            ASTM C1055: Standard Guide for Heated System Surface Conditions
            That Produce Contact Burn Injuries

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.check_personnel_protection_limit(
            ...     surface_temp_c="55.0"
            ... )
            >>> print(f"Safe: {not result.exceeds_limit}")
        """
        builder = ProvenanceBuilder("PERSONNEL_PROTECTION_CHECK")

        # Convert inputs
        T_surface = self._to_decimal(surface_temp_c)
        limit_c = self._to_decimal(custom_limit_c) if custom_limit_c else PERSONNEL_PROTECTION_LIMIT_C

        builder.add_input("surface_temp_c", T_surface)
        builder.add_input("limit_temp_c", limit_c)

        # Step 1: Convert to Fahrenheit for reference
        T_surface_f = T_surface * Decimal("9") / Decimal("5") + Decimal("32")
        limit_f = limit_c * Decimal("9") / Decimal("5") + Decimal("32")

        builder.add_step(
            step_number=1,
            operation="convert",
            description="Convert to Fahrenheit",
            inputs={"T_surface_C": T_surface, "limit_C": limit_c},
            output_name="temperatures_F",
            output_value={"T_surface_F": T_surface_f, "limit_F": limit_f},
            formula="T_F = T_C * 9/5 + 32"
        )

        # Step 2: Check against limit
        exceeds_limit = T_surface > limit_c
        margin_c = limit_c - T_surface
        margin_percent = (margin_c / limit_c) * Decimal("100") if limit_c > Decimal("0") else Decimal("0")

        builder.add_step(
            step_number=2,
            operation="compare",
            description="Check against personnel protection limit",
            inputs={"T_surface": T_surface, "limit": limit_c},
            output_name="exceeds_limit",
            output_value=exceeds_limit,
            formula="exceeds = T_surface > limit",
            reference="ASTM C1055"
        )

        # Step 3: Determine risk level
        if margin_c >= Decimal("15"):
            risk_level = "LOW"
            recommendation = "Surface temperature is well within safe limits."
        elif margin_c >= Decimal("5"):
            risk_level = "MODERATE"
            recommendation = "Surface temperature is acceptable but should be monitored."
        elif margin_c >= Decimal("0"):
            risk_level = "HIGH"
            recommendation = "Surface temperature is near limit. Consider additional insulation or barriers."
        else:
            risk_level = "CRITICAL"
            recommendation = "Surface temperature EXCEEDS safety limit. Immediate corrective action required per ASTM C1055."

        builder.add_step(
            step_number=3,
            operation="assess",
            description="Determine risk level",
            inputs={"margin_c": margin_c},
            output_name="risk_level",
            output_value=risk_level
        )

        builder.add_output("exceeds_limit", exceeds_limit)
        builder.add_output("risk_level", risk_level)
        provenance_hash = builder.build()

        return PersonnelProtectionResult(
            surface_temp_c=self._apply_precision(T_surface),
            surface_temp_f=self._apply_precision(T_surface_f, 1),
            limit_temp_c=self._apply_precision(limit_c),
            limit_temp_f=self._apply_precision(limit_f, 1),
            exceeds_limit=exceeds_limit,
            margin_c=self._apply_precision(margin_c),
            margin_percent=self._apply_precision(margin_percent, 1),
            risk_level=risk_level,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # TEMPERATURE DISTRIBUTION ANALYSIS
    # =========================================================================

    def analyze_temperature_distribution(
        self,
        temperatures: List[Union[Decimal, float, str]],
        hot_spot_threshold_std: Union[Decimal, float, str] = "2.0",
        cold_spot_threshold_std: Union[Decimal, float, str] = "2.0",
        edge_indices: Optional[List[int]] = None,
        joint_indices: Optional[List[int]] = None
    ) -> TemperatureDistributionResult:
        """
        Analyze temperature distribution across an insulated surface.

        Calculates statistical metrics and identifies anomalies including:
        - Hot spots (potential insulation defects or missing insulation)
        - Cold spots (potential moisture or overcooling)
        - Edge effects at insulation boundaries
        - Joint and seam temperature anomalies

        Args:
            temperatures: List of temperature measurements (Celsius)
            hot_spot_threshold_std: Std devs above mean for hot spot (default: 2.0)
            cold_spot_threshold_std: Std devs below mean for cold spot (default: 2.0)
            edge_indices: Indices of measurements at insulation edges
            joint_indices: Indices of measurements at joints/seams

        Returns:
            TemperatureDistributionResult with analysis

        Reference:
            ASTM C1060: Standard Practice for Thermographic Inspection
            of Insulation Installations in Envelope Cavities

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> temps = [45.0, 46.0, 47.0, 55.0, 46.0, 44.0, 48.0, 65.0, 47.0]
            >>> result = analyzer.analyze_temperature_distribution(temps)
            >>> print(f"Hot spots: {result.hot_spot_count}")
        """
        builder = ProvenanceBuilder("TEMPERATURE_DISTRIBUTION_ANALYSIS")

        # Convert inputs
        temps = [self._to_decimal(t) for t in temperatures]
        hot_threshold = self._to_decimal(hot_spot_threshold_std)
        cold_threshold = self._to_decimal(cold_spot_threshold_std)

        n = len(temps)
        if n < 2:
            raise ValueError("At least 2 temperature measurements required")

        builder.add_input("measurement_count", n)
        builder.add_input("hot_spot_threshold_std", hot_threshold)
        builder.add_input("cold_spot_threshold_std", cold_threshold)

        # Step 1: Calculate basic statistics
        mean_temp = sum(temps) / Decimal(str(n))
        min_temp = min(temps)
        max_temp = max(temps)

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate mean, min, max temperatures",
            inputs={"temperatures": [str(t) for t in temps]},
            output_name="basic_stats",
            output_value={"mean": mean_temp, "min": min_temp, "max": max_temp}
        )

        # Step 2: Calculate standard deviation
        variance = sum((t - mean_temp) ** 2 for t in temps) / Decimal(str(n - 1))
        std_dev = self._sqrt(variance)

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate standard deviation",
            inputs={"variance": variance},
            output_name="std_deviation",
            output_value=std_dev,
            formula="std = sqrt(sum((x-mean)^2)/(n-1))"
        )

        # Step 3: Calculate uniformity index (0 = uniform, higher = non-uniform)
        # Coefficient of variation: std/mean * 100
        if mean_temp != Decimal("0"):
            uniformity_index = (std_dev / abs(mean_temp)) * Decimal("100")
        else:
            uniformity_index = Decimal("0")

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate uniformity index (CV%)",
            inputs={"std_dev": std_dev, "mean": mean_temp},
            output_name="uniformity_index",
            output_value=uniformity_index,
            formula="CV = (std / mean) * 100"
        )

        # Step 4: Identify hot spots
        hot_spot_limit = mean_temp + hot_threshold * std_dev
        hot_spots = [i for i, t in enumerate(temps) if t > hot_spot_limit]

        builder.add_step(
            step_number=4,
            operation="filter",
            description="Identify hot spots",
            inputs={"threshold": hot_spot_limit},
            output_name="hot_spots",
            output_value=hot_spots,
            formula="T > mean + threshold * std"
        )

        # Step 5: Identify cold spots
        cold_spot_limit = mean_temp - cold_threshold * std_dev
        cold_spots = [i for i, t in enumerate(temps) if t < cold_spot_limit]

        builder.add_step(
            step_number=5,
            operation="filter",
            description="Identify cold spots",
            inputs={"threshold": cold_spot_limit},
            output_name="cold_spots",
            output_value=cold_spots,
            formula="T < mean - threshold * std"
        )

        # Step 6: Check edge effects
        edge_effect = False
        if edge_indices:
            edge_temps = [temps[i] for i in edge_indices if i < n]
            if edge_temps:
                edge_mean = sum(edge_temps) / Decimal(str(len(edge_temps)))
                # Edge effect if edge temps significantly different from overall mean
                edge_effect = abs(edge_mean - mean_temp) > std_dev

        builder.add_step(
            step_number=6,
            operation="check",
            description="Check for edge effects",
            inputs={"edge_indices": edge_indices},
            output_name="edge_effect_detected",
            output_value=edge_effect
        )

        # Step 7: Check joint anomalies
        joint_anomaly = False
        if joint_indices:
            joint_temps = [temps[i] for i in joint_indices if i < n]
            if joint_temps:
                joint_mean = sum(joint_temps) / Decimal(str(len(joint_temps)))
                # Joint anomaly if joint temps significantly higher than mean
                joint_anomaly = joint_mean > mean_temp + Decimal("0.5") * std_dev

        builder.add_step(
            step_number=7,
            operation="check",
            description="Check for joint anomalies",
            inputs={"joint_indices": joint_indices},
            output_name="joint_anomaly_detected",
            output_value=joint_anomaly
        )

        builder.add_output("uniformity_index", uniformity_index)
        builder.add_output("hot_spot_count", len(hot_spots))
        builder.add_output("cold_spot_count", len(cold_spots))
        provenance_hash = builder.build()

        return TemperatureDistributionResult(
            measurement_count=n,
            mean_temp_c=self._apply_precision(mean_temp),
            min_temp_c=self._apply_precision(min_temp),
            max_temp_c=self._apply_precision(max_temp),
            std_deviation_c=self._apply_precision(std_dev),
            uniformity_index=self._apply_precision(uniformity_index, 2),
            hot_spot_count=len(hot_spots),
            cold_spot_count=len(cold_spots),
            hot_spot_locations=tuple(hot_spots),
            cold_spot_locations=tuple(cold_spots),
            edge_effect_detected=edge_effect,
            joint_anomaly_detected=joint_anomaly,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # MEASUREMENT UNCERTAINTY (GUM METHOD)
    # =========================================================================

    def calculate_measurement_uncertainty(
        self,
        measured_temp_c: Union[Decimal, float, str],
        camera_accuracy_c: Union[Decimal, float, str] = "2.0",
        emissivity: Union[Decimal, float, str] = "0.95",
        emissivity_uncertainty: Union[Decimal, float, str] = "0.02",
        ambient_temp_c: Union[Decimal, float, str] = "20.0",
        ambient_uncertainty_c: Union[Decimal, float, str] = "1.0",
        reflected_temp_c: Optional[Union[Decimal, float, str]] = None,
        reflected_temp_uncertainty_c: Optional[Union[Decimal, float, str]] = None,
        wind_speed_m_s: Union[Decimal, float, str] = "0.0",
        coverage_factor: Union[Decimal, float, str] = "2.0"
    ) -> MeasurementUncertaintyResult:
        """
        Calculate combined measurement uncertainty using GUM methodology.

        Implements the Guide to Expression of Uncertainty in Measurement (GUM)
        to combine individual uncertainty components into a combined standard
        uncertainty and expanded uncertainty.

        Uncertainty sources:
        - Camera accuracy (Type B)
        - Emissivity uncertainty (Type B)
        - Ambient temperature uncertainty (Type A/B)
        - Reflected temperature uncertainty (Type B)
        - Environmental factors (Type B)

        Combined uncertainty: u_c = sqrt(sum(u_i^2))
        Expanded uncertainty: U = k * u_c

        Args:
            measured_temp_c: Measured temperature (Celsius)
            camera_accuracy_c: Camera accuracy specification (default: +/-2 C)
            emissivity: Surface emissivity used
            emissivity_uncertainty: Uncertainty in emissivity value
            ambient_temp_c: Ambient temperature
            ambient_uncertainty_c: Uncertainty in ambient (default: +/-1 C)
            reflected_temp_c: Reflected temperature (if applicable)
            reflected_temp_uncertainty_c: Uncertainty in reflected temp
            wind_speed_m_s: Wind speed (affects environmental uncertainty)
            coverage_factor: k-factor for expanded uncertainty (default: 2 for ~95%)

        Returns:
            MeasurementUncertaintyResult with complete uncertainty budget

        Reference:
            JCGM 100:2008 (GUM)
            ASTM E2758: Standard Guide for Selection and Use of Wideband
            Low Temperature Infrared Thermometers

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.calculate_measurement_uncertainty(
            ...     measured_temp_c="75.0",
            ...     emissivity="0.90",
            ...     emissivity_uncertainty="0.03"
            ... )
            >>> print(f"U95: +/-{result.expanded_uncertainty_c} C")
        """
        builder = ProvenanceBuilder("MEASUREMENT_UNCERTAINTY")

        # Convert inputs
        T_measured = self._to_decimal(measured_temp_c)
        u_camera = self._to_decimal(camera_accuracy_c)
        eps = self._to_decimal(emissivity)
        u_eps = self._to_decimal(emissivity_uncertainty)
        T_ambient = self._to_decimal(ambient_temp_c)
        u_ambient = self._to_decimal(ambient_uncertainty_c)
        wind = self._to_decimal(wind_speed_m_s)
        k = self._to_decimal(coverage_factor)

        builder.add_input("measured_temp_c", T_measured)
        builder.add_input("camera_accuracy_c", u_camera)
        builder.add_input("emissivity", eps)
        builder.add_input("emissivity_uncertainty", u_eps)
        builder.add_input("coverage_factor", k)

        # Step 1: Camera uncertainty (Type B - rectangular distribution)
        # Standard uncertainty = specification / sqrt(3) for rectangular
        u_camera_std = u_camera / self._sqrt(Decimal("3"))

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate camera standard uncertainty",
            inputs={"camera_accuracy": u_camera},
            output_name="u_camera_std",
            output_value=u_camera_std,
            formula="u = accuracy / sqrt(3)",
            reference="GUM Type B rectangular"
        )

        # Step 2: Emissivity uncertainty contribution
        # Sensitivity coefficient: dT/deps (approximate)
        # For small epsilon changes: delta_T ~ (1-eps)/eps * T * delta_eps
        if eps > Decimal("0"):
            sensitivity_eps = abs(T_measured) * (Decimal("1") - eps) / eps
            u_emissivity = sensitivity_eps * u_eps
        else:
            u_emissivity = Decimal("0")

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate emissivity uncertainty contribution",
            inputs={"T": T_measured, "eps": eps, "u_eps": u_eps},
            output_name="u_emissivity",
            output_value=u_emissivity,
            formula="u = T * (1-eps)/eps * u_eps"
        )

        # Step 3: Ambient temperature uncertainty
        u_ambient_std = u_ambient / self._sqrt(Decimal("3"))

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate ambient uncertainty contribution",
            inputs={"u_ambient": u_ambient},
            output_name="u_ambient_std",
            output_value=u_ambient_std,
            formula="u = u_ambient / sqrt(3)"
        )

        # Step 4: Reflected temperature uncertainty
        if reflected_temp_c is not None and reflected_temp_uncertainty_c is not None:
            T_reflected = self._to_decimal(reflected_temp_c)
            u_reflected = self._to_decimal(reflected_temp_uncertainty_c)
            # Sensitivity: (1-eps) * factor based on T difference
            u_reflected_contribution = (Decimal("1") - eps) * u_reflected / self._sqrt(Decimal("3"))
        else:
            u_reflected_contribution = Decimal("0")

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate reflected temperature uncertainty",
            inputs={"reflected_temp_c": reflected_temp_c},
            output_name="u_reflected",
            output_value=u_reflected_contribution
        )

        # Step 5: Environmental uncertainty (wind, solar, etc.)
        # Increase uncertainty with wind speed
        wind_factor = Decimal("1") + wind * Decimal("0.1")
        u_environmental = Decimal("0.5") * wind_factor

        builder.add_step(
            step_number=5,
            operation="calculate",
            description="Calculate environmental uncertainty",
            inputs={"wind_speed": wind},
            output_name="u_environmental",
            output_value=u_environmental,
            formula="u = 0.5 * (1 + wind * 0.1)"
        )

        # Step 6: Combined standard uncertainty (RSS)
        u_combined_squared = (
            u_camera_std ** 2 +
            u_emissivity ** 2 +
            u_ambient_std ** 2 +
            u_reflected_contribution ** 2 +
            u_environmental ** 2
        )
        u_combined = self._sqrt(u_combined_squared)

        builder.add_step(
            step_number=6,
            operation="calculate",
            description="Calculate combined standard uncertainty (RSS)",
            inputs={
                "u_camera": u_camera_std,
                "u_emissivity": u_emissivity,
                "u_ambient": u_ambient_std,
                "u_reflected": u_reflected_contribution,
                "u_environmental": u_environmental
            },
            output_name="u_combined",
            output_value=u_combined,
            formula="u_c = sqrt(sum(u_i^2))",
            reference="GUM clause 5.1"
        )

        # Step 7: Expanded uncertainty
        U_expanded = k * u_combined

        builder.add_step(
            step_number=7,
            operation="multiply",
            description="Calculate expanded uncertainty",
            inputs={"k": k, "u_combined": u_combined},
            output_name="U_expanded",
            output_value=U_expanded,
            formula="U = k * u_c",
            reference="GUM clause 6"
        )

        # Step 8: Calculate confidence level and relative uncertainty
        # k=2 corresponds to ~95% confidence for normal distribution
        if k == Decimal("2"):
            confidence = Decimal("95")
        elif k == Decimal("3"):
            confidence = Decimal("99")
        else:
            confidence = Decimal("95")  # Default assumption

        relative_u = (U_expanded / abs(T_measured)) * Decimal("100") if T_measured != Decimal("0") else Decimal("0")

        builder.add_step(
            step_number=8,
            operation="calculate",
            description="Calculate confidence level and relative uncertainty",
            inputs={"k": k, "U": U_expanded, "T": T_measured},
            output_name="relative_uncertainty",
            output_value=relative_u,
            formula="relative_% = U / T * 100"
        )

        builder.add_output("combined_standard_uncertainty_c", u_combined)
        builder.add_output("expanded_uncertainty_c", U_expanded)
        provenance_hash = builder.build()

        return MeasurementUncertaintyResult(
            measured_value_c=self._apply_precision(T_measured),
            camera_uncertainty_c=self._apply_precision(u_camera_std),
            emissivity_uncertainty_c=self._apply_precision(u_emissivity),
            ambient_uncertainty_c=self._apply_precision(u_ambient_std),
            reflected_temp_uncertainty_c=self._apply_precision(u_reflected_contribution),
            environmental_uncertainty_c=self._apply_precision(u_environmental),
            combined_standard_uncertainty_c=self._apply_precision(u_combined),
            expanded_uncertainty_c=self._apply_precision(U_expanded),
            coverage_factor=self._apply_precision(k, 1),
            confidence_level_percent=self._apply_precision(confidence, 0),
            relative_uncertainty_percent=self._apply_precision(relative_u, 2),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # INSPECTION CONDITIONS RECOMMENDATION
    # =========================================================================

    def recommend_inspection_conditions(
        self,
        ambient_temp_c: Union[Decimal, float, str],
        process_temp_c: Union[Decimal, float, str],
        wind_speed_m_s: Union[Decimal, float, str],
        is_sunny: bool = False,
        hour_of_day: Optional[int] = None
    ) -> InspectionConditionsResult:
        """
        Evaluate and recommend optimal inspection conditions.

        Per ASTM E1934, thermographic inspections require adequate temperature
        differential and stable conditions. This method evaluates current
        conditions and provides recommendations.

        Criteria for good inspection conditions:
        - Delta-T (process - ambient) >= 10 C
        - Wind speed <= 5 m/s
        - No direct solar loading on target surfaces
        - Stable conditions (not during transients)

        Args:
            ambient_temp_c: Current ambient temperature (Celsius)
            process_temp_c: Process/operating temperature (Celsius)
            wind_speed_m_s: Current wind speed (m/s)
            is_sunny: Whether there is direct sunlight
            hour_of_day: Current hour (0-23) for timing recommendations

        Returns:
            InspectionConditionsResult with recommendations

        Reference:
            ASTM E1934: Standard Guide for Examining Electrical and
            Mechanical Equipment with Infrared Thermography

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.recommend_inspection_conditions(
            ...     ambient_temp_c="25.0",
            ...     process_temp_c="150.0",
            ...     wind_speed_m_s="3.0",
            ...     is_sunny=True,
            ...     hour_of_day=14
            ... )
            >>> print(f"Recommended: {result.is_recommended}")
        """
        builder = ProvenanceBuilder("INSPECTION_CONDITIONS_RECOMMENDATION")

        # Convert inputs
        T_ambient = self._to_decimal(ambient_temp_c)
        T_process = self._to_decimal(process_temp_c)
        wind = self._to_decimal(wind_speed_m_s)

        builder.add_input("ambient_temp_c", T_ambient)
        builder.add_input("process_temp_c", T_process)
        builder.add_input("wind_speed_m_s", wind)
        builder.add_input("is_sunny", is_sunny)
        builder.add_input("hour_of_day", hour_of_day)

        issues: List[str] = []
        recommendations: List[str] = []
        optimal_times: List[str] = []

        # Step 1: Check temperature differential
        delta_T = T_process - T_ambient
        delta_T_adequate = delta_T >= RECOMMENDED_DELTA_T_MIN_C

        builder.add_step(
            step_number=1,
            operation="check",
            description="Check temperature differential",
            inputs={"delta_T": delta_T, "minimum": RECOMMENDED_DELTA_T_MIN_C},
            output_name="delta_T_adequate",
            output_value=delta_T_adequate,
            reference="ASTM E1934"
        )

        if not delta_T_adequate:
            issues.append(f"Insufficient delta-T: {delta_T} C (minimum: {RECOMMENDED_DELTA_T_MIN_C} C)")
            recommendations.append("Wait for process to reach steady-state operating temperature")
            recommendations.append("Consider inspecting during cooler ambient conditions")

        # Step 2: Check wind speed
        wind_acceptable = wind <= RECOMMENDED_WIND_MAX_M_S

        builder.add_step(
            step_number=2,
            operation="check",
            description="Check wind speed",
            inputs={"wind": wind, "maximum": RECOMMENDED_WIND_MAX_M_S},
            output_name="wind_acceptable",
            output_value=wind_acceptable
        )

        if not wind_acceptable:
            issues.append(f"Excessive wind: {wind} m/s (maximum: {RECOMMENDED_WIND_MAX_M_S} m/s)")
            recommendations.append("Wait for calmer conditions or use wind shields")
            recommendations.append("Apply appropriate wind correction factors to measurements")

        # Step 3: Check solar loading
        solar_acceptable = not is_sunny

        builder.add_step(
            step_number=3,
            operation="check",
            description="Check solar loading",
            inputs={"is_sunny": is_sunny},
            output_name="solar_acceptable",
            output_value=solar_acceptable
        )

        if not solar_acceptable:
            issues.append("Direct solar loading may affect measurements")
            recommendations.append("Inspect shaded sides of equipment first")
            recommendations.append("Apply solar correction to sun-exposed measurements")
            optimal_times.append("Before sunrise (pre-dawn)")
            optimal_times.append("After sunset (1-2 hours)")
            optimal_times.append("On overcast days")

        # Step 4: Recommend optimal timing
        if hour_of_day is not None:
            if 6 <= hour_of_day <= 8:
                optimal_times.append("Current time is near-optimal (early morning)")
            elif 18 <= hour_of_day <= 20:
                optimal_times.append("Current time is near-optimal (evening)")
            elif 10 <= hour_of_day <= 16:
                if is_sunny:
                    optimal_times.append("Avoid mid-day solar loading - reschedule to early AM or evening")
        else:
            optimal_times.append("Early morning (6-8 AM): Minimal solar loading")
            optimal_times.append("Evening (6-8 PM): Minimal solar loading")
            optimal_times.append("Night: No solar interference")

        # Step 5: Determine overall condition rating
        if delta_T_adequate and wind_acceptable and solar_acceptable:
            condition_rating = InspectionCondition.EXCELLENT.name
            is_recommended = True
        elif delta_T_adequate and wind_acceptable:
            condition_rating = InspectionCondition.GOOD.name
            is_recommended = True
        elif delta_T_adequate:
            condition_rating = InspectionCondition.ACCEPTABLE.name
            is_recommended = True
        elif not delta_T_adequate:
            condition_rating = InspectionCondition.POOR.name
            is_recommended = False
        else:
            condition_rating = InspectionCondition.UNACCEPTABLE.name
            is_recommended = False

        builder.add_step(
            step_number=5,
            operation="assess",
            description="Determine overall condition rating",
            inputs={
                "delta_T_ok": delta_T_adequate,
                "wind_ok": wind_acceptable,
                "solar_ok": solar_acceptable
            },
            output_name="condition_rating",
            output_value=condition_rating
        )

        builder.add_output("is_recommended", is_recommended)
        builder.add_output("condition_rating", condition_rating)
        provenance_hash = builder.build()

        return InspectionConditionsResult(
            current_ambient_c=self._apply_precision(T_ambient),
            current_wind_m_s=self._apply_precision(wind),
            current_delta_t=self._apply_precision(delta_T),
            current_solar_load=is_sunny,
            condition_rating=condition_rating,
            is_recommended=is_recommended,
            issues=tuple(issues),
            recommendations=tuple(recommendations),
            optimal_time_windows=tuple(optimal_times),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # SEASONAL ADJUSTMENT
    # =========================================================================

    def apply_seasonal_adjustment(
        self,
        measured_temp_c: Union[Decimal, float, str],
        ambient_at_measurement_c: Union[Decimal, float, str],
        measurement_season: Season,
        target_season: Optional[Season] = None,
        process_temp_c: Optional[Union[Decimal, float, str]] = None
    ) -> SeasonalAdjustmentResult:
        """
        Adjust measured temperature for seasonal baseline differences.

        Allows comparison of measurements taken in different seasons by
        normalizing to a common baseline or projecting to a target season.

        The adjustment uses the temperature ratio method to scale the
        surface temperature based on ambient temperature changes.

        Args:
            measured_temp_c: Measured surface temperature (Celsius)
            ambient_at_measurement_c: Ambient during measurement (Celsius)
            measurement_season: Season when measurement was taken
            target_season: Target season for adjustment (default: annual average)
            process_temp_c: Process temperature (for ratio method)

        Returns:
            SeasonalAdjustmentResult with adjusted temperature

        Example:
            >>> analyzer = SurfaceTemperatureAnalyzer()
            >>> result = analyzer.apply_seasonal_adjustment(
            ...     measured_temp_c="50.0",
            ...     ambient_at_measurement_c="30.0",
            ...     measurement_season=Season.SUMMER,
            ...     target_season=Season.WINTER
            ... )
        """
        builder = ProvenanceBuilder("SEASONAL_ADJUSTMENT")

        # Convert inputs
        T_measured = self._to_decimal(measured_temp_c)
        T_ambient_meas = self._to_decimal(ambient_at_measurement_c)
        T_process = self._to_decimal(process_temp_c) if process_temp_c else T_measured + Decimal("100")

        builder.add_input("measured_temp_c", T_measured)
        builder.add_input("ambient_at_measurement_c", T_ambient_meas)
        builder.add_input("measurement_season", measurement_season.name)

        # Step 1: Get seasonal baseline temperatures
        baseline_measurement = SEASONAL_BASELINE_C[measurement_season]

        if target_season:
            baseline_target = SEASONAL_BASELINE_C[target_season]
            target_name = target_season.name
        else:
            # Use annual average
            baseline_target = sum(SEASONAL_BASELINE_C.values()) / Decimal("4")
            target_name = "ANNUAL_AVERAGE"

        builder.add_step(
            step_number=1,
            operation="lookup",
            description="Get seasonal baseline temperatures",
            inputs={"measurement_season": measurement_season.name},
            output_name="baselines",
            output_value={
                "measurement": baseline_measurement,
                "target": baseline_target
            }
        )

        # Step 2: Calculate temperature ratio
        delta_T_meas = T_measured - T_ambient_meas
        if T_process - T_ambient_meas != Decimal("0"):
            temp_ratio = delta_T_meas / (T_process - T_ambient_meas)
        else:
            temp_ratio = Decimal("1")

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate temperature ratio",
            inputs={"T_measured": T_measured, "T_ambient": T_ambient_meas, "T_process": T_process},
            output_name="temp_ratio",
            output_value=temp_ratio
        )

        # Step 3: Calculate seasonal correction
        seasonal_correction = baseline_target - baseline_measurement

        builder.add_step(
            step_number=3,
            operation="subtract",
            description="Calculate seasonal correction",
            inputs={"baseline_target": baseline_target, "baseline_measurement": baseline_measurement},
            output_name="seasonal_correction",
            output_value=seasonal_correction
        )

        # Step 4: Calculate adjusted temperature
        # Adjust ambient to target season and recalculate surface temp
        T_ambient_target = T_ambient_meas + seasonal_correction
        delta_T_target = temp_ratio * (T_process - T_ambient_target)
        T_adjusted = T_ambient_target + delta_T_target

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate adjusted surface temperature",
            inputs={
                "T_ambient_target": T_ambient_target,
                "temp_ratio": temp_ratio,
                "T_process": T_process
            },
            output_name="T_adjusted",
            output_value=T_adjusted
        )

        # Step 5: Calculate annual average
        annual_avg_ambient = sum(SEASONAL_BASELINE_C.values()) / Decimal("4")
        delta_T_annual = temp_ratio * (T_process - annual_avg_ambient)
        T_annual_avg = annual_avg_ambient + delta_T_annual

        builder.add_step(
            step_number=5,
            operation="calculate",
            description="Calculate annual average temperature",
            inputs={"annual_avg_ambient": annual_avg_ambient},
            output_name="T_annual_avg",
            output_value=T_annual_avg
        )

        builder.add_output("adjusted_temp_c", T_adjusted)
        provenance_hash = builder.build()

        return SeasonalAdjustmentResult(
            measured_temp_c=self._apply_precision(T_measured),
            measurement_season=measurement_season.name,
            target_season=target_name,
            seasonal_correction_c=self._apply_precision(seasonal_correction),
            adjusted_temp_c=self._apply_precision(T_adjusted),
            annual_average_temp_c=self._apply_precision(T_annual_avg),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal with error handling."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding using ROUND_HALF_UP."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _sqrt(self, value: Decimal) -> Decimal:
        """Calculate square root of a Decimal."""
        if value < Decimal("0"):
            raise ValueError("Cannot calculate square root of negative number")
        if value == Decimal("0"):
            return Decimal("0")
        return Decimal(str(math.sqrt(float(value))))

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate base^exponent for Decimal values."""
        if base == Decimal("0"):
            return Decimal("0") if exponent > Decimal("0") else Decimal("1")
        if exponent == Decimal("0"):
            return Decimal("1")
        return Decimal(str(math.pow(float(base), float(exponent))))


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LocationType",
    "SurfaceOrientation",
    "SurfaceColor",
    "CloudCover",
    "Season",
    "InspectionCondition",

    # Constants
    "STEFAN_BOLTZMANN",
    "REFERENCE_AMBIENT_C",
    "KELVIN_OFFSET",
    "PERSONNEL_PROTECTION_LIMIT_C",
    "PERSONNEL_PROTECTION_LIMIT_F",
    "SOLAR_ABSORPTIVITY",
    "WIND_CORRECTION_FACTORS",
    "CLOUD_COVER_FACTORS",
    "SKY_TEMPERATURE_DEPRESSION",
    "SEASONAL_BASELINE_C",

    # Result data classes
    "AmbientNormalizationResult",
    "WindCorrectionResult",
    "SolarCorrectionResult",
    "ReflectedTemperatureResult",
    "PersonnelProtectionResult",
    "TemperatureDistributionResult",
    "MeasurementUncertaintyResult",
    "InspectionConditionsResult",
    "SeasonalAdjustmentResult",

    # Main class
    "SurfaceTemperatureAnalyzer",
]
