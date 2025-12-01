"""
Thermal Image Analyzer Module for GL-015 INSULSCAN

This module provides comprehensive thermal image analysis capabilities with
zero-hallucination guarantees for insulation assessment and building thermography.

All calculations are deterministic and include complete provenance tracking
for regulatory compliance and audit requirements.

Features:
- Temperature matrix processing from IR camera data formats
- Emissivity corrections and atmospheric compensation
- Hotspot detection with clustering and boundary analysis
- Temperature mapping and statistical analysis
- Region of Interest (ROI) analysis
- Thermal anomaly classification per ASTM E1934
- Image quality assessment

Standards Compliance:
- ASTM E1934: Standard Practice for Examining Electrical and Mechanical
  Equipment with Infrared Thermography
- ISO 6781: Thermal Insulation - Qualitative detection of thermal
  irregularities in building envelopes
- EN 13187: Thermal performance of buildings - Qualitative detection
  of thermal irregularities in building envelopes

Example:
    >>> from thermal_image_analyzer import ThermalImageAnalyzer
    >>> analyzer = ThermalImageAnalyzer()
    >>> matrix = analyzer.process_temperature_matrix(raw_data, emissivity=0.95)
    >>> hotspots = analyzer.detect_hotspots(matrix, delta_t_threshold=5.0)
    >>> anomalies = analyzer.classify_thermal_anomaly(hotspots)
"""

from typing import Dict, List, Optional, Any, Tuple, Union, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
import math
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class IRCameraResolution(Enum):
    """Standard IR camera resolutions."""
    RES_160x120 = (160, 120)
    RES_320x240 = (320, 240)
    RES_640x480 = (640, 480)
    RES_1024x768 = (1024, 768)
    RES_1280x960 = (1280, 960)


class ThermalAnomalyType(Enum):
    """Classification of thermal anomalies in insulation systems."""
    MISSING_INSULATION = "missing_insulation"
    WET_INSULATION = "wet_insulation"
    DAMAGED_INSULATION = "damaged_insulation"
    COMPRESSED_INSULATION = "compressed_insulation"
    THERMAL_BRIDGING = "thermal_bridging"
    JOINT_LEAK = "joint_leak"
    FLANGE_LEAK = "flange_leak"
    VALVE_EXPOSURE = "valve_exposure"
    FITTING_EXPOSURE = "fitting_exposure"
    AIR_INFILTRATION = "air_infiltration"
    MOISTURE_INTRUSION = "moisture_intrusion"
    NORMAL = "normal"
    INDETERMINATE = "indeterminate"


class ROIShape(Enum):
    """Region of Interest shape types."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    LINE = "line"
    POINT = "point"


class ImageQualityLevel(Enum):
    """Image quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"
    UNUSABLE = "unusable"


# Physical constants
STEFAN_BOLTZMANN = Decimal("5.670374419E-8")  # W/(m^2*K^4)
ABSOLUTE_ZERO_C = Decimal("-273.15")


# =============================================================================
# IMMUTABLE DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class TemperatureMatrixConfig:
    """Configuration for temperature matrix processing."""
    width: int
    height: int
    emissivity: Decimal
    reflected_temperature_c: Decimal
    atmospheric_temperature_c: Decimal
    distance_m: Decimal
    relative_humidity: Decimal
    atmospheric_transmission: Decimal

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        if not Decimal("0.01") <= self.emissivity <= Decimal("1.0"):
            raise ValueError("Emissivity must be between 0.01 and 1.0")
        if self.distance_m <= 0:
            raise ValueError("Distance must be positive")
        if not Decimal("0") <= self.relative_humidity <= Decimal("100"):
            raise ValueError("Relative humidity must be between 0 and 100")


@dataclass(frozen=True)
class TemperaturePoint:
    """Single temperature measurement point."""
    row: int
    col: int
    temperature_c: Decimal
    raw_value: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "row": self.row,
            "col": self.col,
            "temperature_c": str(self.temperature_c),
            "raw_value": self.raw_value
        }


@dataclass(frozen=True)
class TemperatureStatistics:
    """Statistical summary of temperature distribution."""
    min_c: Decimal
    max_c: Decimal
    mean_c: Decimal
    std_dev_c: Decimal
    median_c: Decimal
    percentile_5_c: Decimal
    percentile_25_c: Decimal
    percentile_75_c: Decimal
    percentile_95_c: Decimal
    pixel_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "min_c": str(self.min_c),
            "max_c": str(self.max_c),
            "mean_c": str(self.mean_c),
            "std_dev_c": str(self.std_dev_c),
            "median_c": str(self.median_c),
            "percentile_5_c": str(self.percentile_5_c),
            "percentile_25_c": str(self.percentile_25_c),
            "percentile_75_c": str(self.percentile_75_c),
            "percentile_95_c": str(self.percentile_95_c),
            "pixel_count": self.pixel_count
        }


@dataclass(frozen=True)
class Hotspot:
    """Detected thermal hotspot."""
    hotspot_id: str
    centroid_row: Decimal
    centroid_col: Decimal
    peak_temperature_c: Decimal
    peak_row: int
    peak_col: int
    mean_temperature_c: Decimal
    area_pixels: int
    area_m2: Optional[Decimal]
    delta_t_from_ambient: Decimal
    boundary_pixels: Tuple[Tuple[int, int], ...]
    severity_score: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hotspot_id": self.hotspot_id,
            "centroid_row": str(self.centroid_row),
            "centroid_col": str(self.centroid_col),
            "peak_temperature_c": str(self.peak_temperature_c),
            "peak_row": self.peak_row,
            "peak_col": self.peak_col,
            "mean_temperature_c": str(self.mean_temperature_c),
            "area_pixels": self.area_pixels,
            "area_m2": str(self.area_m2) if self.area_m2 else None,
            "delta_t_from_ambient": str(self.delta_t_from_ambient),
            "boundary_pixels": self.boundary_pixels,
            "severity_score": str(self.severity_score)
        }


@dataclass(frozen=True)
class TemperatureGradient:
    """Temperature gradient calculation result."""
    magnitude_c_per_m: Decimal
    direction_degrees: Decimal
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    distance_m: Decimal
    temperature_difference_c: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "magnitude_c_per_m": str(self.magnitude_c_per_m),
            "direction_degrees": str(self.direction_degrees),
            "start_point": self.start_point,
            "end_point": self.end_point,
            "distance_m": str(self.distance_m),
            "temperature_difference_c": str(self.temperature_difference_c)
        }


@dataclass(frozen=True)
class IsothermalContour:
    """Isothermal contour line at a specific temperature."""
    temperature_c: Decimal
    contour_id: str
    pixels: Tuple[Tuple[int, int], ...]
    length_pixels: int
    is_closed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "temperature_c": str(self.temperature_c),
            "contour_id": self.contour_id,
            "pixels": self.pixels,
            "length_pixels": self.length_pixels,
            "is_closed": self.is_closed
        }


@dataclass(frozen=True)
class ROIDefinition:
    """Region of Interest definition."""
    roi_id: str
    shape: ROIShape
    name: str
    vertices: Tuple[Tuple[int, int], ...]  # For polygon/rectangle
    center: Optional[Tuple[int, int]] = None  # For circle/ellipse
    radius: Optional[int] = None  # For circle
    semi_axes: Optional[Tuple[int, int]] = None  # For ellipse (a, b)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "roi_id": self.roi_id,
            "shape": self.shape.value,
            "name": self.name,
            "vertices": self.vertices,
            "center": self.center,
            "radius": self.radius,
            "semi_axes": self.semi_axes
        }


@dataclass(frozen=True)
class ROIAnalysisResult:
    """Result of ROI temperature analysis."""
    roi: ROIDefinition
    statistics: TemperatureStatistics
    hotspots: Tuple[Hotspot, ...]
    reference_points: Tuple[TemperaturePoint, ...]
    analysis_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "roi": self.roi.to_dict(),
            "statistics": self.statistics.to_dict(),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "reference_points": [p.to_dict() for p in self.reference_points],
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


@dataclass(frozen=True)
class ThermalAnomalyClassification:
    """Classification result for thermal anomaly."""
    anomaly_type: ThermalAnomalyType
    confidence: Decimal
    severity: str  # low, medium, high, critical
    description: str
    recommended_action: str
    reference_standard: str
    supporting_evidence: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "confidence": str(self.confidence),
            "severity": self.severity,
            "description": self.description,
            "recommended_action": self.recommended_action,
            "reference_standard": self.reference_standard,
            "supporting_evidence": self.supporting_evidence
        }


@dataclass(frozen=True)
class ImageQualityAssessment:
    """Assessment of thermal image quality."""
    overall_quality: ImageQualityLevel
    focus_quality_score: Decimal
    thermal_contrast_score: Decimal
    spatial_resolution_score: Decimal
    noise_level_score: Decimal
    environmental_suitability_score: Decimal
    issues_detected: Tuple[str, ...]
    recommendations: Tuple[str, ...]
    is_usable_for_analysis: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_quality": self.overall_quality.value,
            "focus_quality_score": str(self.focus_quality_score),
            "thermal_contrast_score": str(self.thermal_contrast_score),
            "spatial_resolution_score": str(self.spatial_resolution_score),
            "noise_level_score": str(self.noise_level_score),
            "environmental_suitability_score": str(self.environmental_suitability_score),
            "issues_detected": self.issues_detected,
            "recommendations": self.recommendations,
            "is_usable_for_analysis": self.is_usable_for_analysis
        }


@dataclass(frozen=True)
class TemperatureMapResult:
    """Result of temperature mapping analysis."""
    statistics: TemperatureStatistics
    contours: Tuple[IsothermalContour, ...]
    histogram_bins: Tuple[Decimal, ...]
    histogram_counts: Tuple[int, ...]
    gradient_map_available: bool
    max_gradient: Optional[TemperatureGradient]
    analysis_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "statistics": self.statistics.to_dict(),
            "contours": [c.to_dict() for c in self.contours],
            "histogram_bins": [str(b) for b in self.histogram_bins],
            "histogram_counts": list(self.histogram_counts),
            "gradient_map_available": self.gradient_map_available,
            "max_gradient": self.max_gradient.to_dict() if self.max_gradient else None,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


@dataclass(frozen=True)
class ThermalAnalysisResult:
    """Complete thermal image analysis result with provenance."""
    analysis_id: str
    config: TemperatureMatrixConfig
    temperature_matrix_shape: Tuple[int, int]
    statistics: TemperatureStatistics
    hotspots: Tuple[Hotspot, ...]
    anomaly_classifications: Tuple[ThermalAnomalyClassification, ...]
    image_quality: ImageQualityAssessment
    temperature_map: Optional[TemperatureMapResult]
    roi_results: Tuple[ROIAnalysisResult, ...]
    provenance_hash: str
    calculation_steps: Tuple[Dict[str, Any], ...]
    analysis_timestamp: datetime
    processing_time_ms: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analysis_id": self.analysis_id,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "emissivity": str(self.config.emissivity),
                "reflected_temperature_c": str(self.config.reflected_temperature_c),
                "atmospheric_temperature_c": str(self.config.atmospheric_temperature_c),
                "distance_m": str(self.config.distance_m),
                "relative_humidity": str(self.config.relative_humidity),
                "atmospheric_transmission": str(self.config.atmospheric_transmission)
            },
            "temperature_matrix_shape": self.temperature_matrix_shape,
            "statistics": self.statistics.to_dict(),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "anomaly_classifications": [a.to_dict() for a in self.anomaly_classifications],
            "image_quality": self.image_quality.to_dict(),
            "temperature_map": self.temperature_map.to_dict() if self.temperature_map else None,
            "roi_results": [r.to_dict() for r in self.roi_results],
            "provenance_hash": self.provenance_hash,
            "calculation_steps": list(self.calculation_steps),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "processing_time_ms": str(self.processing_time_ms)
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# ANOMALY CLASSIFICATION THRESHOLDS (ASTM E1934 BASED)
# =============================================================================

@dataclass(frozen=True)
class AnomalyThresholds:
    """Thresholds for thermal anomaly classification."""
    # Delta T thresholds for anomaly severity (from ambient)
    MINOR_DELTA_T_C: Decimal = Decimal("3.0")
    MODERATE_DELTA_T_C: Decimal = Decimal("5.0")
    SERIOUS_DELTA_T_C: Decimal = Decimal("10.0")
    CRITICAL_DELTA_T_C: Decimal = Decimal("20.0")

    # Gradient thresholds
    HIGH_GRADIENT_C_PER_M: Decimal = Decimal("50.0")
    VERY_HIGH_GRADIENT_C_PER_M: Decimal = Decimal("100.0")

    # Area thresholds (as fraction of total image)
    SMALL_AREA_FRACTION: Decimal = Decimal("0.01")
    MEDIUM_AREA_FRACTION: Decimal = Decimal("0.05")
    LARGE_AREA_FRACTION: Decimal = Decimal("0.15")

    # Wet insulation detection
    WET_INSULATION_TEMP_DEPRESSION_C: Decimal = Decimal("2.0")

    # Missing insulation (hot pipe)
    MISSING_INSULATION_DELTA_T_C: Decimal = Decimal("15.0")


ANOMALY_THRESHOLDS = AnomalyThresholds()


# =============================================================================
# MAIN THERMAL IMAGE ANALYZER CLASS
# =============================================================================

class ThermalImageAnalyzer:
    """
    Zero-hallucination thermal image analyzer.

    Provides comprehensive thermal image analysis with complete provenance
    tracking and deterministic calculations for regulatory compliance.

    All calculations use Decimal precision and are fully auditable.
    """

    def __init__(
        self,
        pixel_size_m: Optional[Decimal] = None,
        default_emissivity: Decimal = Decimal("0.95"),
        default_ambient_c: Decimal = Decimal("20.0")
    ):
        """
        Initialize the thermal image analyzer.

        Args:
            pixel_size_m: Physical size of each pixel in meters (for area calculations)
            default_emissivity: Default surface emissivity (0.01 to 1.0)
            default_ambient_c: Default ambient temperature in Celsius
        """
        self.pixel_size_m = pixel_size_m
        self.default_emissivity = default_emissivity
        self.default_ambient_c = default_ambient_c
        self._calculation_steps: List[Dict[str, Any]] = []

        logger.debug(
            f"ThermalImageAnalyzer initialized: pixel_size={pixel_size_m}m, "
            f"emissivity={default_emissivity}, ambient={default_ambient_c}C"
        )

    def _add_calculation_step(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula: Optional[str] = None,
        reference: Optional[str] = None
    ) -> None:
        """Record a calculation step for provenance tracking."""
        step = {
            "step_id": f"step_{len(self._calculation_steps) + 1:04d}",
            "step_name": step_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": self._serialize_for_json(inputs),
            "outputs": self._serialize_for_json(outputs),
            "formula": formula,
            "reference": reference
        }
        self._calculation_steps.append(step)

    def _serialize_for_json(self, data: Any) -> Any:
        """Serialize data for JSON output."""
        if isinstance(data, Decimal):
            return str(data)
        elif isinstance(data, dict):
            return {k: self._serialize_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_json(v) for v in data]
        elif isinstance(data, Enum):
            return data.value
        elif isinstance(data, datetime):
            return data.isoformat()
        return data

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _decimal_sqrt(self, value: Decimal) -> Decimal:
        """Calculate square root of Decimal with high precision."""
        if value < 0:
            raise ValueError("Cannot calculate square root of negative number")
        if value == 0:
            return Decimal("0")

        # Newton-Raphson method
        precision = Decimal("0.0000000001")
        x = value
        while True:
            x_new = (x + value / x) / 2
            if abs(x - x_new) < precision:
                return x_new.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
            x = x_new

    def _decimal_atan2(self, y: Decimal, x: Decimal) -> Decimal:
        """Calculate atan2 for Decimal values, return degrees."""
        # Convert to float for math operations, then back to Decimal
        result = math.atan2(float(y), float(x))
        degrees = math.degrees(result)
        return Decimal(str(degrees)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # =========================================================================
    # TEMPERATURE MATRIX PROCESSING
    # =========================================================================

    def process_temperature_matrix(
        self,
        raw_data: List[List[Union[int, float, Decimal]]],
        emissivity: Optional[Decimal] = None,
        reflected_temperature_c: Decimal = Decimal("20.0"),
        atmospheric_temperature_c: Decimal = Decimal("20.0"),
        distance_m: Decimal = Decimal("1.0"),
        relative_humidity: Decimal = Decimal("50.0"),
        raw_to_temp_slope: Decimal = Decimal("0.04"),
        raw_to_temp_offset: Decimal = Decimal("-273.15")
    ) -> Tuple[List[List[Decimal]], TemperatureMatrixConfig]:
        """
        Process raw IR camera data into corrected temperature matrix.

        Applies emissivity correction, reflected temperature compensation,
        and atmospheric transmission correction.

        Args:
            raw_data: Raw radiometric values from IR camera
            emissivity: Surface emissivity (0.01 to 1.0)
            reflected_temperature_c: Reflected apparent temperature
            atmospheric_temperature_c: Atmospheric temperature
            distance_m: Distance from camera to target
            relative_humidity: Relative humidity percentage
            raw_to_temp_slope: Calibration slope for raw-to-temp conversion
            raw_to_temp_offset: Calibration offset for raw-to-temp conversion

        Returns:
            Tuple of (corrected temperature matrix, configuration)
        """
        self._calculation_steps = []

        emissivity = emissivity or self.default_emissivity
        height = len(raw_data)
        width = len(raw_data[0]) if height > 0 else 0

        # Calculate atmospheric transmission
        atmospheric_transmission = self._calculate_atmospheric_transmission(
            distance_m, atmospheric_temperature_c, relative_humidity
        )

        config = TemperatureMatrixConfig(
            width=width,
            height=height,
            emissivity=emissivity,
            reflected_temperature_c=reflected_temperature_c,
            atmospheric_temperature_c=atmospheric_temperature_c,
            distance_m=distance_m,
            relative_humidity=relative_humidity,
            atmospheric_transmission=atmospheric_transmission
        )

        self._add_calculation_step(
            "initialize_processing",
            {
                "width": width,
                "height": height,
                "emissivity": emissivity,
                "distance_m": distance_m
            },
            {"config_created": True},
            reference="ASTM E1934"
        )

        # Convert raw to apparent temperature
        apparent_matrix: List[List[Decimal]] = []
        for row_idx, row in enumerate(raw_data):
            apparent_row: List[Decimal] = []
            for col_idx, raw_val in enumerate(row):
                raw_decimal = Decimal(str(raw_val))
                apparent_temp = raw_decimal * raw_to_temp_slope + raw_to_temp_offset
                apparent_row.append(apparent_temp)
            apparent_matrix.append(apparent_row)

        self._add_calculation_step(
            "raw_to_apparent_temperature",
            {"raw_to_temp_slope": raw_to_temp_slope, "raw_to_temp_offset": raw_to_temp_offset},
            {"matrix_converted": True},
            formula="T_apparent = raw_value * slope + offset"
        )

        # Apply emissivity correction
        corrected_matrix = self.apply_emissivity_correction(
            apparent_matrix,
            emissivity,
            reflected_temperature_c,
            atmospheric_temperature_c,
            atmospheric_transmission
        )

        return corrected_matrix, config

    def apply_emissivity_correction(
        self,
        apparent_matrix: List[List[Decimal]],
        emissivity: Decimal,
        reflected_temperature_c: Decimal,
        atmospheric_temperature_c: Decimal,
        atmospheric_transmission: Decimal
    ) -> List[List[Decimal]]:
        """
        Apply emissivity and atmospheric corrections to temperature matrix.

        Formula: T_true = ((T_apparent^4 - (1-e)*T_reflected^4 -
                          (1-tau)*T_atm^4) / (e * tau))^0.25

        Simplified for typical conditions: T_true = T_apparent / e^0.25

        Args:
            apparent_matrix: Apparent temperature matrix in Celsius
            emissivity: Surface emissivity
            reflected_temperature_c: Reflected temperature
            atmospheric_temperature_c: Atmospheric temperature
            atmospheric_transmission: Atmospheric transmission factor

        Returns:
            Corrected temperature matrix
        """
        corrected_matrix: List[List[Decimal]] = []

        # Convert temperatures to Kelvin for radiometric calculations
        t_reflected_k = reflected_temperature_c - ABSOLUTE_ZERO_C
        t_atm_k = atmospheric_temperature_c - ABSOLUTE_ZERO_C

        e = emissivity
        tau = atmospheric_transmission

        for row in apparent_matrix:
            corrected_row: List[Decimal] = []
            for t_apparent_c in row:
                t_apparent_k = t_apparent_c - ABSOLUTE_ZERO_C

                # Full radiometric correction
                # T_true^4 = (T_apparent^4 - (1-e)*T_reflected^4 - (1-tau)*T_atm^4) / (e * tau)
                t_apparent_4 = t_apparent_k ** 4
                t_reflected_4 = t_reflected_k ** 4
                t_atm_4 = t_atm_k ** 4

                numerator = t_apparent_4 - (1 - e) * t_reflected_4 - (1 - tau) * t_atm_4
                denominator = e * tau

                if denominator > 0 and numerator > 0:
                    t_true_4 = numerator / denominator
                    t_true_k = self._decimal_pow_quarter(t_true_4)
                    t_true_c = t_true_k + ABSOLUTE_ZERO_C
                else:
                    # Fallback to simplified correction
                    t_true_c = t_apparent_c / self._decimal_pow_quarter(e)

                corrected_row.append(
                    t_true_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                )
            corrected_matrix.append(corrected_row)

        self._add_calculation_step(
            "emissivity_correction",
            {
                "emissivity": emissivity,
                "reflected_temperature_c": reflected_temperature_c,
                "atmospheric_transmission": atmospheric_transmission
            },
            {"correction_applied": True},
            formula="T_true^4 = (T_apparent^4 - (1-e)*T_reflected^4 - (1-tau)*T_atm^4) / (e*tau)",
            reference="Radiometric temperature measurement theory"
        )

        return corrected_matrix

    def _decimal_pow_quarter(self, value: Decimal) -> Decimal:
        """Calculate fourth root (x^0.25) of a Decimal."""
        if value < 0:
            raise ValueError("Cannot calculate fourth root of negative number")
        if value == 0:
            return Decimal("0")

        # x^0.25 = sqrt(sqrt(x))
        sqrt1 = self._decimal_sqrt(value)
        return self._decimal_sqrt(sqrt1)

    def _calculate_atmospheric_transmission(
        self,
        distance_m: Decimal,
        temperature_c: Decimal,
        relative_humidity: Decimal
    ) -> Decimal:
        """
        Calculate atmospheric transmission factor.

        Based on the Beer-Lambert law for atmospheric absorption.

        Args:
            distance_m: Distance in meters
            temperature_c: Atmospheric temperature in Celsius
            relative_humidity: Relative humidity percentage

        Returns:
            Atmospheric transmission factor (0 to 1)
        """
        # Simplified atmospheric transmission model
        # tau = exp(-alpha * distance)
        # alpha depends on temperature and humidity

        # Calculate water vapor content
        t_k = temperature_c - ABSOLUTE_ZERO_C

        # Saturation vapor pressure (simplified)
        e_sat = Decimal("6.1078") * Decimal(str(math.exp(
            float(Decimal("17.27") * temperature_c / (temperature_c + Decimal("237.3")))
        )))

        # Actual vapor pressure
        e_act = e_sat * relative_humidity / Decimal("100")

        # Water vapor absorption coefficient (simplified)
        alpha = Decimal("0.006") + Decimal("0.001") * e_act / Decimal("10")

        # Transmission
        transmission = Decimal(str(math.exp(-float(alpha * distance_m))))

        # Clamp to valid range
        transmission = max(Decimal("0.1"), min(Decimal("1.0"), transmission))

        self._add_calculation_step(
            "atmospheric_transmission",
            {
                "distance_m": distance_m,
                "temperature_c": temperature_c,
                "relative_humidity": relative_humidity
            },
            {"transmission": transmission},
            formula="tau = exp(-alpha * distance)",
            reference="Beer-Lambert atmospheric absorption"
        )

        return transmission.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    # =========================================================================
    # HOTSPOT DETECTION
    # =========================================================================

    def detect_hotspots(
        self,
        temperature_matrix: List[List[Decimal]],
        delta_t_threshold: Decimal = Decimal("5.0"),
        ambient_temperature_c: Optional[Decimal] = None,
        min_hotspot_pixels: int = 4,
        merge_distance_pixels: int = 3
    ) -> List[Hotspot]:
        """
        Detect thermal hotspots in temperature matrix.

        Uses threshold-based detection with gradient-based edge detection
        and clustering for multiple hotspot identification.

        Args:
            temperature_matrix: Corrected temperature matrix
            delta_t_threshold: Temperature difference threshold from ambient
            ambient_temperature_c: Ambient reference temperature
            min_hotspot_pixels: Minimum pixels for valid hotspot
            merge_distance_pixels: Distance for merging nearby hotspots

        Returns:
            List of detected hotspots
        """
        if not temperature_matrix or not temperature_matrix[0]:
            return []

        height = len(temperature_matrix)
        width = len(temperature_matrix[0])

        # Calculate ambient if not provided
        if ambient_temperature_c is None:
            ambient_temperature_c = self._calculate_ambient_reference(temperature_matrix)

        threshold_temp = ambient_temperature_c + delta_t_threshold

        self._add_calculation_step(
            "hotspot_detection_init",
            {
                "delta_t_threshold": delta_t_threshold,
                "ambient_temperature_c": ambient_temperature_c,
                "threshold_temp": threshold_temp
            },
            {"detection_started": True}
        )

        # Binary mask of above-threshold pixels
        mask: List[List[bool]] = []
        for row in temperature_matrix:
            mask_row = [temp >= threshold_temp for temp in row]
            mask.append(mask_row)

        # Find connected components (hotspot clustering)
        visited = [[False] * width for _ in range(height)]
        hotspots: List[Hotspot] = []

        for row_idx in range(height):
            for col_idx in range(width):
                if mask[row_idx][col_idx] and not visited[row_idx][col_idx]:
                    # Flood fill to find connected component
                    component_pixels = self._flood_fill(
                        mask, visited, row_idx, col_idx, height, width
                    )

                    if len(component_pixels) >= min_hotspot_pixels:
                        hotspot = self._create_hotspot_from_pixels(
                            temperature_matrix,
                            component_pixels,
                            ambient_temperature_c,
                            len(hotspots) + 1
                        )
                        hotspots.append(hotspot)

        # Merge nearby hotspots
        hotspots = self._merge_nearby_hotspots(hotspots, merge_distance_pixels)

        self._add_calculation_step(
            "hotspot_detection_complete",
            {"min_hotspot_pixels": min_hotspot_pixels},
            {"hotspots_detected": len(hotspots)},
            reference="ASTM E1934 - Thermographic inspection"
        )

        return hotspots

    def _flood_fill(
        self,
        mask: List[List[bool]],
        visited: List[List[bool]],
        start_row: int,
        start_col: int,
        height: int,
        width: int
    ) -> List[Tuple[int, int]]:
        """Flood fill algorithm to find connected components."""
        pixels: List[Tuple[int, int]] = []
        stack = [(start_row, start_col)]

        while stack:
            row, col = stack.pop()

            if (row < 0 or row >= height or col < 0 or col >= width or
                visited[row][col] or not mask[row][col]):
                continue

            visited[row][col] = True
            pixels.append((row, col))

            # 8-connectivity
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr != 0 or dc != 0:
                        stack.append((row + dr, col + dc))

        return pixels

    def _create_hotspot_from_pixels(
        self,
        temperature_matrix: List[List[Decimal]],
        pixels: List[Tuple[int, int]],
        ambient_temperature_c: Decimal,
        hotspot_number: int
    ) -> Hotspot:
        """Create hotspot object from pixel list."""
        temperatures = [temperature_matrix[r][c] for r, c in pixels]

        # Find peak temperature and location
        peak_temp = max(temperatures)
        peak_idx = temperatures.index(peak_temp)
        peak_row, peak_col = pixels[peak_idx]

        # Calculate mean temperature
        mean_temp = sum(temperatures) / len(temperatures)

        # Calculate centroid
        centroid_row = Decimal(sum(r for r, c in pixels)) / len(pixels)
        centroid_col = Decimal(sum(c for r, c in pixels)) / len(pixels)

        # Find boundary pixels
        boundary = self._find_boundary_pixels(pixels)

        # Calculate area in m^2 if pixel size known
        area_m2 = None
        if self.pixel_size_m:
            area_m2 = Decimal(len(pixels)) * self.pixel_size_m ** 2

        # Calculate delta T from ambient
        delta_t = peak_temp - ambient_temperature_c

        # Calculate severity score (0-100)
        severity_score = self._calculate_severity_score(delta_t, len(pixels))

        return Hotspot(
            hotspot_id=f"HS-{hotspot_number:04d}",
            centroid_row=centroid_row.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            centroid_col=centroid_col.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            peak_temperature_c=peak_temp,
            peak_row=peak_row,
            peak_col=peak_col,
            mean_temperature_c=mean_temp.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            area_pixels=len(pixels),
            area_m2=area_m2.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) if area_m2 else None,
            delta_t_from_ambient=delta_t.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            boundary_pixels=tuple(boundary),
            severity_score=severity_score
        )

    def _find_boundary_pixels(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find boundary pixels of a region."""
        pixel_set = set(pixels)
        boundary = []

        for row, col in pixels:
            is_boundary = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (row + dr, col + dc)
                if neighbor not in pixel_set:
                    is_boundary = True
                    break
            if is_boundary:
                boundary.append((row, col))

        return boundary

    def _calculate_severity_score(self, delta_t: Decimal, area_pixels: int) -> Decimal:
        """
        Calculate severity score (0-100) based on temperature difference and area.

        Based on ASTM E1934 severity classification guidelines.
        """
        # Temperature component (0-70 points)
        if delta_t >= ANOMALY_THRESHOLDS.CRITICAL_DELTA_T_C:
            temp_score = Decimal("70")
        elif delta_t >= ANOMALY_THRESHOLDS.SERIOUS_DELTA_T_C:
            temp_score = Decimal("50") + (delta_t - ANOMALY_THRESHOLDS.SERIOUS_DELTA_T_C) * Decimal("2")
        elif delta_t >= ANOMALY_THRESHOLDS.MODERATE_DELTA_T_C:
            temp_score = Decimal("30") + (delta_t - ANOMALY_THRESHOLDS.MODERATE_DELTA_T_C) * Decimal("4")
        elif delta_t >= ANOMALY_THRESHOLDS.MINOR_DELTA_T_C:
            temp_score = Decimal("10") + (delta_t - ANOMALY_THRESHOLDS.MINOR_DELTA_T_C) * Decimal("10")
        else:
            temp_score = delta_t * Decimal("3.33")

        temp_score = min(Decimal("70"), temp_score)

        # Area component (0-30 points) - logarithmic scaling
        area_score = Decimal(str(min(30, math.log10(area_pixels + 1) * 15)))

        total_score = temp_score + area_score
        return min(Decimal("100"), total_score).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def _merge_nearby_hotspots(
        self,
        hotspots: List[Hotspot],
        merge_distance: int
    ) -> List[Hotspot]:
        """Merge hotspots that are within merge_distance of each other."""
        if len(hotspots) <= 1:
            return hotspots

        # Simple distance-based merging (could be improved with union-find)
        merged = list(hotspots)
        changed = True

        while changed:
            changed = False
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if i >= len(merged) or j >= len(merged):
                        break

                    dist = self._decimal_sqrt(
                        (merged[i].centroid_row - merged[j].centroid_row) ** 2 +
                        (merged[i].centroid_col - merged[j].centroid_col) ** 2
                    )

                    if dist <= merge_distance:
                        # Merge j into i
                        # Keep the one with higher severity
                        if merged[j].severity_score > merged[i].severity_score:
                            merged[i] = merged[j]
                        merged.pop(j)
                        changed = True
                        break
                if changed:
                    break

        return merged

    def _calculate_ambient_reference(
        self,
        temperature_matrix: List[List[Decimal]]
    ) -> Decimal:
        """Calculate ambient reference temperature from matrix."""
        # Use 10th percentile as ambient reference
        all_temps = [temp for row in temperature_matrix for temp in row]
        all_temps.sort()

        idx = int(len(all_temps) * 0.1)
        return all_temps[idx] if all_temps else self.default_ambient_c

    # =========================================================================
    # TEMPERATURE MAPPING
    # =========================================================================

    def generate_temperature_map(
        self,
        temperature_matrix: List[List[Decimal]],
        contour_interval_c: Decimal = Decimal("2.0"),
        calculate_gradients: bool = True,
        histogram_bins: int = 50
    ) -> TemperatureMapResult:
        """
        Generate temperature map with isothermal contours and statistics.

        Args:
            temperature_matrix: Corrected temperature matrix
            contour_interval_c: Interval between isothermal contours
            calculate_gradients: Whether to calculate temperature gradients
            histogram_bins: Number of bins for temperature histogram

        Returns:
            Temperature map analysis result
        """
        if not temperature_matrix or not temperature_matrix[0]:
            raise ValueError("Empty temperature matrix")

        # Calculate statistics
        statistics = self._calculate_statistics(temperature_matrix)

        # Generate isothermal contours
        contours = self._generate_isothermal_contours(
            temperature_matrix, contour_interval_c, statistics.min_c, statistics.max_c
        )

        # Calculate histogram
        hist_bins, hist_counts = self._calculate_histogram(
            temperature_matrix, histogram_bins, statistics.min_c, statistics.max_c
        )

        # Calculate maximum gradient if requested
        max_gradient = None
        if calculate_gradients:
            max_gradient = self._find_maximum_gradient(temperature_matrix)

        self._add_calculation_step(
            "temperature_map_generation",
            {
                "contour_interval_c": contour_interval_c,
                "histogram_bins": histogram_bins
            },
            {
                "contours_generated": len(contours),
                "statistics_calculated": True
            }
        )

        return TemperatureMapResult(
            statistics=statistics,
            contours=tuple(contours),
            histogram_bins=tuple(hist_bins),
            histogram_counts=tuple(hist_counts),
            gradient_map_available=calculate_gradients,
            max_gradient=max_gradient,
            analysis_timestamp=datetime.now(timezone.utc)
        )

    def _calculate_statistics(
        self,
        temperature_matrix: List[List[Decimal]]
    ) -> TemperatureStatistics:
        """Calculate comprehensive temperature statistics."""
        all_temps = [temp for row in temperature_matrix for temp in row]
        all_temps.sort()
        n = len(all_temps)

        if n == 0:
            raise ValueError("Empty temperature matrix")

        min_c = all_temps[0]
        max_c = all_temps[-1]

        # Mean
        mean_c = sum(all_temps) / n

        # Standard deviation
        variance = sum((t - mean_c) ** 2 for t in all_temps) / n
        std_dev_c = self._decimal_sqrt(variance)

        # Median
        median_c = all_temps[n // 2] if n % 2 == 1 else (all_temps[n // 2 - 1] + all_temps[n // 2]) / 2

        # Percentiles
        def percentile(p: int) -> Decimal:
            idx = int(n * p / 100)
            return all_temps[min(idx, n - 1)]

        return TemperatureStatistics(
            min_c=min_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            max_c=max_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            mean_c=mean_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            std_dev_c=std_dev_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            median_c=median_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_5_c=percentile(5).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_25_c=percentile(25).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_75_c=percentile(75).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_95_c=percentile(95).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            pixel_count=n
        )

    def _generate_isothermal_contours(
        self,
        temperature_matrix: List[List[Decimal]],
        interval_c: Decimal,
        min_c: Decimal,
        max_c: Decimal
    ) -> List[IsothermalContour]:
        """Generate isothermal contour lines."""
        contours: List[IsothermalContour] = []
        height = len(temperature_matrix)
        width = len(temperature_matrix[0])

        # Generate contour at each interval
        current_temp = min_c + interval_c
        contour_num = 0

        while current_temp < max_c:
            contour_pixels: List[Tuple[int, int]] = []

            # Find pixels at this temperature level (marching squares simplified)
            for row in range(height - 1):
                for col in range(width - 1):
                    # Check if contour crosses this cell
                    t00 = temperature_matrix[row][col]
                    t01 = temperature_matrix[row][col + 1]
                    t10 = temperature_matrix[row + 1][col]
                    t11 = temperature_matrix[row + 1][col + 1]

                    temps = [t00, t01, t10, t11]
                    above = sum(1 for t in temps if t >= current_temp)

                    # Contour crosses if some above and some below
                    if 0 < above < 4:
                        contour_pixels.append((row, col))

            if contour_pixels:
                # Determine if contour is closed
                is_closed = self._is_contour_closed(contour_pixels)

                contour_num += 1
                contours.append(IsothermalContour(
                    temperature_c=current_temp.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
                    contour_id=f"ISO-{contour_num:04d}",
                    pixels=tuple(contour_pixels),
                    length_pixels=len(contour_pixels),
                    is_closed=is_closed
                ))

            current_temp += interval_c

        return contours

    def _is_contour_closed(self, pixels: List[Tuple[int, int]]) -> bool:
        """Check if contour forms a closed loop."""
        if len(pixels) < 4:
            return False

        # Simple check: if first and last pixels are neighbors
        first = pixels[0]
        last = pixels[-1]

        dist = abs(first[0] - last[0]) + abs(first[1] - last[1])
        return dist <= 2

    def _calculate_histogram(
        self,
        temperature_matrix: List[List[Decimal]],
        num_bins: int,
        min_c: Decimal,
        max_c: Decimal
    ) -> Tuple[List[Decimal], List[int]]:
        """Calculate temperature histogram."""
        bin_width = (max_c - min_c) / num_bins
        bins: List[Decimal] = []
        counts: List[int] = [0] * num_bins

        # Create bin edges
        for i in range(num_bins + 1):
            bin_edge = min_c + bin_width * i
            bins.append(bin_edge.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        # Count temperatures in each bin
        for row in temperature_matrix:
            for temp in row:
                bin_idx = int((temp - min_c) / bin_width)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                counts[bin_idx] += 1

        return bins, counts

    def _find_maximum_gradient(
        self,
        temperature_matrix: List[List[Decimal]]
    ) -> Optional[TemperatureGradient]:
        """Find the maximum temperature gradient in the matrix."""
        height = len(temperature_matrix)
        width = len(temperature_matrix[0])

        if height < 2 or width < 2:
            return None

        max_gradient_magnitude = Decimal("0")
        max_gradient: Optional[TemperatureGradient] = None

        for row in range(height - 1):
            for col in range(width - 1):
                # Calculate gradient using central differences
                dt_dx = (temperature_matrix[row][col + 1] - temperature_matrix[row][col])
                dt_dy = (temperature_matrix[row + 1][col] - temperature_matrix[row][col])

                magnitude = self._decimal_sqrt(dt_dx ** 2 + dt_dy ** 2)

                if magnitude > max_gradient_magnitude:
                    max_gradient_magnitude = magnitude

                    # Calculate physical gradient if pixel size known
                    if self.pixel_size_m:
                        magnitude_per_m = magnitude / self.pixel_size_m
                        distance_m = self.pixel_size_m
                    else:
                        magnitude_per_m = magnitude
                        distance_m = Decimal("1")

                    direction = self._decimal_atan2(dt_dy, dt_dx)

                    max_gradient = TemperatureGradient(
                        magnitude_c_per_m=magnitude_per_m.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                        direction_degrees=direction,
                        start_point=(row, col),
                        end_point=(row + 1, col + 1),
                        distance_m=distance_m,
                        temperature_difference_c=magnitude.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    )

        return max_gradient

    # =========================================================================
    # REGION OF INTEREST ANALYSIS
    # =========================================================================

    def analyze_roi(
        self,
        temperature_matrix: List[List[Decimal]],
        roi: ROIDefinition,
        detect_hotspots: bool = True,
        hotspot_threshold_c: Decimal = Decimal("5.0")
    ) -> ROIAnalysisResult:
        """
        Analyze temperature distribution within a Region of Interest.

        Args:
            temperature_matrix: Corrected temperature matrix
            roi: Region of Interest definition
            detect_hotspots: Whether to detect hotspots within ROI
            hotspot_threshold_c: Threshold for hotspot detection

        Returns:
            ROI analysis result
        """
        # Extract pixels within ROI
        roi_pixels = self._get_roi_pixels(temperature_matrix, roi)

        if not roi_pixels:
            raise ValueError(f"ROI {roi.name} contains no valid pixels")

        # Extract temperatures
        roi_temps: List[List[Decimal]] = [[p.temperature_c] for p in roi_pixels]

        # Calculate statistics
        all_temps = [p.temperature_c for p in roi_pixels]
        all_temps.sort()
        n = len(all_temps)

        min_c = all_temps[0]
        max_c = all_temps[-1]
        mean_c = sum(all_temps) / n
        variance = sum((t - mean_c) ** 2 for t in all_temps) / n
        std_dev_c = self._decimal_sqrt(variance)
        median_c = all_temps[n // 2] if n % 2 == 1 else (all_temps[n // 2 - 1] + all_temps[n // 2]) / 2

        def percentile(p: int) -> Decimal:
            idx = int(n * p / 100)
            return all_temps[min(idx, n - 1)]

        statistics = TemperatureStatistics(
            min_c=min_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            max_c=max_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            mean_c=mean_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            std_dev_c=std_dev_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            median_c=median_c.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_5_c=percentile(5).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_25_c=percentile(25).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_75_c=percentile(75).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            percentile_95_c=percentile(95).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            pixel_count=n
        )

        # Detect hotspots within ROI
        hotspots: List[Hotspot] = []
        if detect_hotspots:
            roi_matrix = self._create_roi_matrix(temperature_matrix, roi)
            if roi_matrix:
                hotspots = self.detect_hotspots(
                    roi_matrix,
                    delta_t_threshold=hotspot_threshold_c,
                    min_hotspot_pixels=2
                )

        # Reference points (corners and center)
        reference_points = self._get_roi_reference_points(temperature_matrix, roi)

        self._add_calculation_step(
            "roi_analysis",
            {"roi_name": roi.name, "roi_shape": roi.shape.value},
            {
                "pixel_count": n,
                "hotspots_found": len(hotspots),
                "mean_temperature_c": str(mean_c)
            }
        )

        return ROIAnalysisResult(
            roi=roi,
            statistics=statistics,
            hotspots=tuple(hotspots),
            reference_points=tuple(reference_points),
            analysis_timestamp=datetime.now(timezone.utc)
        )

    def _get_roi_pixels(
        self,
        temperature_matrix: List[List[Decimal]],
        roi: ROIDefinition
    ) -> List[TemperaturePoint]:
        """Get all pixels within a Region of Interest."""
        height = len(temperature_matrix)
        width = len(temperature_matrix[0]) if height > 0 else 0
        pixels: List[TemperaturePoint] = []

        if roi.shape == ROIShape.RECTANGLE:
            if len(roi.vertices) >= 2:
                min_row = min(v[0] for v in roi.vertices)
                max_row = max(v[0] for v in roi.vertices)
                min_col = min(v[1] for v in roi.vertices)
                max_col = max(v[1] for v in roi.vertices)

                for row in range(max(0, min_row), min(height, max_row + 1)):
                    for col in range(max(0, min_col), min(width, max_col + 1)):
                        pixels.append(TemperaturePoint(
                            row=row,
                            col=col,
                            temperature_c=temperature_matrix[row][col]
                        ))

        elif roi.shape == ROIShape.CIRCLE:
            if roi.center and roi.radius:
                cr, cc = roi.center
                r = roi.radius

                for row in range(max(0, cr - r), min(height, cr + r + 1)):
                    for col in range(max(0, cc - r), min(width, cc + r + 1)):
                        dist_sq = (row - cr) ** 2 + (col - cc) ** 2
                        if dist_sq <= r ** 2:
                            pixels.append(TemperaturePoint(
                                row=row,
                                col=col,
                                temperature_c=temperature_matrix[row][col]
                            ))

        elif roi.shape == ROIShape.ELLIPSE:
            if roi.center and roi.semi_axes:
                cr, cc = roi.center
                a, b = roi.semi_axes

                for row in range(max(0, cr - b), min(height, cr + b + 1)):
                    for col in range(max(0, cc - a), min(width, cc + a + 1)):
                        # Ellipse equation: (x-cx)^2/a^2 + (y-cy)^2/b^2 <= 1
                        if a > 0 and b > 0:
                            val = ((col - cc) ** 2) / (a ** 2) + ((row - cr) ** 2) / (b ** 2)
                            if val <= 1:
                                pixels.append(TemperaturePoint(
                                    row=row,
                                    col=col,
                                    temperature_c=temperature_matrix[row][col]
                                ))

        elif roi.shape == ROIShape.POLYGON:
            # Point-in-polygon test for each pixel
            for row in range(height):
                for col in range(width):
                    if self._point_in_polygon(row, col, roi.vertices):
                        pixels.append(TemperaturePoint(
                            row=row,
                            col=col,
                            temperature_c=temperature_matrix[row][col]
                        ))

        elif roi.shape == ROIShape.POINT:
            if roi.vertices:
                row, col = roi.vertices[0]
                if 0 <= row < height and 0 <= col < width:
                    pixels.append(TemperaturePoint(
                        row=row,
                        col=col,
                        temperature_c=temperature_matrix[row][col]
                    ))

        return pixels

    def _point_in_polygon(
        self,
        row: int,
        col: int,
        vertices: Tuple[Tuple[int, int], ...]
    ) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(vertices)
        if n < 3:
            return False

        inside = False
        j = n - 1

        for i in range(n):
            yi, xi = vertices[i]
            yj, xj = vertices[j]

            if ((yi > row) != (yj > row)) and \
               (col < (xj - xi) * (row - yi) / (yj - yi + 0.0001) + xi):
                inside = not inside
            j = i

        return inside

    def _create_roi_matrix(
        self,
        temperature_matrix: List[List[Decimal]],
        roi: ROIDefinition
    ) -> List[List[Decimal]]:
        """Create temperature matrix for ROI only."""
        if roi.shape == ROIShape.RECTANGLE and len(roi.vertices) >= 2:
            min_row = min(v[0] for v in roi.vertices)
            max_row = max(v[0] for v in roi.vertices)
            min_col = min(v[1] for v in roi.vertices)
            max_col = max(v[1] for v in roi.vertices)

            height = len(temperature_matrix)
            width = len(temperature_matrix[0]) if height > 0 else 0

            roi_matrix: List[List[Decimal]] = []
            for row in range(max(0, min_row), min(height, max_row + 1)):
                roi_row: List[Decimal] = []
                for col in range(max(0, min_col), min(width, max_col + 1)):
                    roi_row.append(temperature_matrix[row][col])
                if roi_row:
                    roi_matrix.append(roi_row)

            return roi_matrix

        return []

    def _get_roi_reference_points(
        self,
        temperature_matrix: List[List[Decimal]],
        roi: ROIDefinition
    ) -> List[TemperaturePoint]:
        """Get reference temperature points for ROI."""
        height = len(temperature_matrix)
        width = len(temperature_matrix[0]) if height > 0 else 0
        points: List[TemperaturePoint] = []

        # Add center point
        if roi.center:
            cr, cc = roi.center
            if 0 <= cr < height and 0 <= cc < width:
                points.append(TemperaturePoint(
                    row=cr,
                    col=cc,
                    temperature_c=temperature_matrix[cr][cc]
                ))
        elif roi.vertices:
            # Calculate centroid of vertices
            avg_row = sum(v[0] for v in roi.vertices) // len(roi.vertices)
            avg_col = sum(v[1] for v in roi.vertices) // len(roi.vertices)
            if 0 <= avg_row < height and 0 <= avg_col < width:
                points.append(TemperaturePoint(
                    row=avg_row,
                    col=avg_col,
                    temperature_c=temperature_matrix[avg_row][avg_col]
                ))

        # Add vertex points
        for vertex in roi.vertices[:4]:  # Limit to 4 vertices
            row, col = vertex
            if 0 <= row < height and 0 <= col < width:
                points.append(TemperaturePoint(
                    row=row,
                    col=col,
                    temperature_c=temperature_matrix[row][col]
                ))

        return points

    # =========================================================================
    # THERMAL ANOMALY CLASSIFICATION
    # =========================================================================

    def classify_thermal_anomaly(
        self,
        hotspot: Hotspot,
        ambient_temperature_c: Decimal,
        expected_surface_temp_c: Optional[Decimal] = None,
        pipe_process_temp_c: Optional[Decimal] = None,
        insulation_r_value: Optional[Decimal] = None
    ) -> ThermalAnomalyClassification:
        """
        Classify thermal anomaly based on hotspot characteristics.

        Uses ASTM E1934 guidelines and thermal analysis principles.

        Args:
            hotspot: Detected hotspot to classify
            ambient_temperature_c: Ambient temperature reference
            expected_surface_temp_c: Expected insulated surface temperature
            pipe_process_temp_c: Internal process temperature (if known)
            insulation_r_value: Insulation R-value (if known)

        Returns:
            Thermal anomaly classification
        """
        delta_t = hotspot.delta_t_from_ambient
        severity_score = hotspot.severity_score

        # Initialize classification parameters
        anomaly_type = ThermalAnomalyType.INDETERMINATE
        confidence = Decimal("0.5")
        severity = "low"
        description = ""
        recommended_action = ""
        evidence: List[str] = []

        # Determine severity level
        if delta_t >= ANOMALY_THRESHOLDS.CRITICAL_DELTA_T_C:
            severity = "critical"
            evidence.append(f"Delta-T of {delta_t}C exceeds critical threshold of {ANOMALY_THRESHOLDS.CRITICAL_DELTA_T_C}C")
        elif delta_t >= ANOMALY_THRESHOLDS.SERIOUS_DELTA_T_C:
            severity = "high"
            evidence.append(f"Delta-T of {delta_t}C exceeds serious threshold of {ANOMALY_THRESHOLDS.SERIOUS_DELTA_T_C}C")
        elif delta_t >= ANOMALY_THRESHOLDS.MODERATE_DELTA_T_C:
            severity = "medium"
            evidence.append(f"Delta-T of {delta_t}C exceeds moderate threshold of {ANOMALY_THRESHOLDS.MODERATE_DELTA_T_C}C")
        else:
            severity = "low"
            evidence.append(f"Delta-T of {delta_t}C below moderate threshold")

        # Classification logic based on thermal signature
        if pipe_process_temp_c:
            # Hot pipe analysis
            expected_ratio = (hotspot.peak_temperature_c - ambient_temperature_c) / \
                           (pipe_process_temp_c - ambient_temperature_c)

            if expected_ratio > Decimal("0.5"):
                anomaly_type = ThermalAnomalyType.MISSING_INSULATION
                confidence = Decimal("0.85")
                description = "Surface temperature indicates missing or severely degraded insulation"
                recommended_action = "Inspect and replace insulation immediately"
                evidence.append(f"Surface/process temperature ratio: {expected_ratio:.2f}")
            elif expected_ratio > Decimal("0.3"):
                anomaly_type = ThermalAnomalyType.DAMAGED_INSULATION
                confidence = Decimal("0.75")
                description = "Elevated surface temperature suggests damaged insulation"
                recommended_action = "Schedule inspection and repair within 30 days"
                evidence.append(f"Surface/process temperature ratio: {expected_ratio:.2f}")

        # Pattern-based classification
        area_ratio = Decimal(hotspot.area_pixels) / Decimal("10000")  # Normalize

        # Wet insulation signature: temperature depression relative to surroundings
        if expected_surface_temp_c:
            if hotspot.mean_temperature_c < expected_surface_temp_c - ANOMALY_THRESHOLDS.WET_INSULATION_TEMP_DEPRESSION_C:
                anomaly_type = ThermalAnomalyType.WET_INSULATION
                confidence = Decimal("0.70")
                description = "Temperature depression indicates possible moisture infiltration"
                recommended_action = "Conduct moisture probe test to confirm"
                evidence.append("Cold spot pattern consistent with wet insulation")

        # Joint/flange leak signature: localized high temperature at connection
        if severity_score > Decimal("60") and hotspot.area_pixels < 500:
            anomaly_type = ThermalAnomalyType.JOINT_LEAK
            confidence = Decimal("0.65")
            description = "Localized high temperature at connection point suggests joint leak"
            recommended_action = "Inspect joint/flange sealing and insulation"
            evidence.append("Small area with high temperature differential")

        # Large area missing insulation
        if delta_t >= ANOMALY_THRESHOLDS.MISSING_INSULATION_DELTA_T_C and hotspot.area_pixels > 1000:
            anomaly_type = ThermalAnomalyType.MISSING_INSULATION
            confidence = Decimal("0.80")
            description = "Large area with extreme temperature differential indicates missing insulation"
            recommended_action = "Emergency insulation replacement required"
            evidence.append("Large hotspot area with significant delta-T")

        # Thermal bridging: linear pattern with moderate temperature
        if Decimal("3") <= delta_t < Decimal("10") and hotspot.area_pixels > 200:
            # Could be thermal bridging - would need aspect ratio analysis
            if anomaly_type == ThermalAnomalyType.INDETERMINATE:
                anomaly_type = ThermalAnomalyType.THERMAL_BRIDGING
                confidence = Decimal("0.55")
                description = "Temperature pattern may indicate thermal bridging"
                recommended_action = "Further investigation recommended"
                evidence.append("Moderate temperature elevation over extended area")

        # Default to normal if no anomaly detected
        if anomaly_type == ThermalAnomalyType.INDETERMINATE and severity == "low":
            anomaly_type = ThermalAnomalyType.NORMAL
            confidence = Decimal("0.60")
            description = "Temperature within acceptable range for insulated surface"
            recommended_action = "Continue routine monitoring"
            evidence.append("No significant thermal anomaly detected")

        self._add_calculation_step(
            "anomaly_classification",
            {
                "hotspot_id": hotspot.hotspot_id,
                "delta_t": delta_t,
                "severity_score": severity_score
            },
            {
                "anomaly_type": anomaly_type.value,
                "confidence": confidence,
                "severity": severity
            },
            reference="ASTM E1934"
        )

        return ThermalAnomalyClassification(
            anomaly_type=anomaly_type,
            confidence=confidence,
            severity=severity,
            description=description,
            recommended_action=recommended_action,
            reference_standard="ASTM E1934",
            supporting_evidence=tuple(evidence)
        )

    # =========================================================================
    # IMAGE QUALITY ASSESSMENT
    # =========================================================================

    def assess_image_quality(
        self,
        temperature_matrix: List[List[Decimal]],
        ambient_temperature_c: Decimal,
        expected_resolution: IRCameraResolution = IRCameraResolution.RES_640x480,
        min_delta_t_required: Decimal = Decimal("5.0"),
        wind_speed_m_s: Decimal = Decimal("0.0"),
        solar_loading: bool = False
    ) -> ImageQualityAssessment:
        """
        Assess thermal image quality for analysis suitability.

        Based on ASTM E1934 and ISO 6781 guidelines.

        Args:
            temperature_matrix: Temperature matrix to assess
            ambient_temperature_c: Ambient temperature during capture
            expected_resolution: Expected camera resolution
            min_delta_t_required: Minimum temperature differential needed
            wind_speed_m_s: Wind speed during capture
            solar_loading: Whether direct solar loading was present

        Returns:
            Image quality assessment
        """
        height = len(temperature_matrix)
        width = len(temperature_matrix[0]) if height > 0 else 0

        issues: List[str] = []
        recommendations: List[str] = []

        # Resolution check
        expected_w, expected_h = expected_resolution.value
        resolution_match = (width >= expected_w * 0.9 and height >= expected_h * 0.9)

        if not resolution_match:
            resolution_score = Decimal("0.5")
            issues.append(f"Resolution {width}x{height} below expected {expected_w}x{expected_h}")
            recommendations.append("Use higher resolution IR camera if available")
        else:
            resolution_score = Decimal("1.0")

        # Calculate thermal contrast
        all_temps = [temp for row in temperature_matrix for temp in row]
        if all_temps:
            temp_range = max(all_temps) - min(all_temps)

            if temp_range >= min_delta_t_required:
                contrast_score = min(Decimal("1.0"), temp_range / (min_delta_t_required * 2))
            else:
                contrast_score = temp_range / min_delta_t_required
                issues.append(f"Thermal contrast {temp_range}C below required {min_delta_t_required}C")
                recommendations.append("Ensure adequate temperature differential before capture")
        else:
            contrast_score = Decimal("0")
            issues.append("No temperature data available")

        # Focus quality estimation (based on gradient sharpness)
        focus_score = self._estimate_focus_quality(temperature_matrix)

        if focus_score < Decimal("0.6"):
            issues.append("Image appears out of focus")
            recommendations.append("Ensure proper focus before capture")

        # Noise level estimation
        noise_score = self._estimate_noise_level(temperature_matrix)

        if noise_score < Decimal("0.7"):
            issues.append("High noise level detected")
            recommendations.append("Use thermal averaging or lower NETD camera")

        # Environmental suitability
        env_score = Decimal("1.0")

        if wind_speed_m_s > Decimal("5.0"):
            env_score -= Decimal("0.3")
            issues.append(f"Wind speed {wind_speed_m_s} m/s affects surface temperatures")
            recommendations.append("Conduct survey in calm conditions (<5 m/s wind)")

        if solar_loading:
            env_score -= Decimal("0.4")
            issues.append("Solar loading may mask thermal anomalies")
            recommendations.append("Conduct survey during overcast conditions or at night")

        env_score = max(Decimal("0.1"), env_score)

        # Calculate overall quality
        overall_score = (
            resolution_score * Decimal("0.2") +
            contrast_score * Decimal("0.3") +
            focus_score * Decimal("0.25") +
            noise_score * Decimal("0.15") +
            env_score * Decimal("0.1")
        )

        # Determine quality level
        if overall_score >= Decimal("0.9"):
            quality_level = ImageQualityLevel.EXCELLENT
        elif overall_score >= Decimal("0.75"):
            quality_level = ImageQualityLevel.GOOD
        elif overall_score >= Decimal("0.6"):
            quality_level = ImageQualityLevel.ACCEPTABLE
        elif overall_score >= Decimal("0.4"):
            quality_level = ImageQualityLevel.MARGINAL
        elif overall_score >= Decimal("0.2"):
            quality_level = ImageQualityLevel.POOR
        else:
            quality_level = ImageQualityLevel.UNUSABLE

        is_usable = quality_level in [
            ImageQualityLevel.EXCELLENT,
            ImageQualityLevel.GOOD,
            ImageQualityLevel.ACCEPTABLE
        ]

        self._add_calculation_step(
            "image_quality_assessment",
            {
                "width": width,
                "height": height,
                "wind_speed_m_s": wind_speed_m_s,
                "solar_loading": solar_loading
            },
            {
                "overall_quality": quality_level.value,
                "is_usable": is_usable
            },
            reference="ASTM E1934, ISO 6781"
        )

        return ImageQualityAssessment(
            overall_quality=quality_level,
            focus_quality_score=focus_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            thermal_contrast_score=contrast_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            spatial_resolution_score=resolution_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            noise_level_score=noise_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            environmental_suitability_score=env_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            issues_detected=tuple(issues),
            recommendations=tuple(recommendations),
            is_usable_for_analysis=is_usable
        )

    def _estimate_focus_quality(self, temperature_matrix: List[List[Decimal]]) -> Decimal:
        """
        Estimate focus quality based on gradient sharpness.

        Uses Laplacian-based focus measure.
        """
        height = len(temperature_matrix)
        width = len(temperature_matrix[0]) if height > 0 else 0

        if height < 3 or width < 3:
            return Decimal("0.5")

        # Calculate Laplacian variance
        laplacian_sum = Decimal("0")
        count = 0

        for row in range(1, height - 1):
            for col in range(1, width - 1):
                # Laplacian: center pixel - average of neighbors
                center = temperature_matrix[row][col]
                neighbors_sum = (
                    temperature_matrix[row-1][col] +
                    temperature_matrix[row+1][col] +
                    temperature_matrix[row][col-1] +
                    temperature_matrix[row][col+1]
                )
                laplacian = abs(center * 4 - neighbors_sum)
                laplacian_sum += laplacian ** 2
                count += 1

        if count == 0:
            return Decimal("0.5")

        variance = laplacian_sum / count

        # Normalize to 0-1 range (empirical thresholds)
        focus_score = min(Decimal("1.0"), self._decimal_sqrt(variance) / Decimal("10"))

        return focus_score

    def _estimate_noise_level(self, temperature_matrix: List[List[Decimal]]) -> Decimal:
        """
        Estimate noise level in temperature matrix.

        Uses local variance method in uniform regions.
        """
        height = len(temperature_matrix)
        width = len(temperature_matrix[0]) if height > 0 else 0

        if height < 5 or width < 5:
            return Decimal("0.7")

        # Calculate local variance in 3x3 windows
        local_variances: List[Decimal] = []

        for row in range(1, height - 1, 3):
            for col in range(1, width - 1, 3):
                # Get 3x3 window
                window_temps: List[Decimal] = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        window_temps.append(temperature_matrix[row + dr][col + dc])

                # Calculate variance
                mean_temp = sum(window_temps) / len(window_temps)
                variance = sum((t - mean_temp) ** 2 for t in window_temps) / len(window_temps)
                local_variances.append(variance)

        if not local_variances:
            return Decimal("0.7")

        # Use median of local variances as noise estimate
        local_variances.sort()
        median_variance = local_variances[len(local_variances) // 2]

        # Convert to noise score (lower variance = better score)
        noise_std = self._decimal_sqrt(median_variance)

        # Empirical scaling: 0.1C noise = score 1.0, 1.0C noise = score 0.5
        noise_score = max(Decimal("0.1"), Decimal("1.0") - noise_std / Decimal("2"))

        return noise_score

    # =========================================================================
    # TEMPERATURE GRADIENT CALCULATION
    # =========================================================================

    def calculate_temperature_gradient(
        self,
        temperature_matrix: List[List[Decimal]],
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        physical_distance_m: Optional[Decimal] = None
    ) -> TemperatureGradient:
        """
        Calculate temperature gradient between two points.

        Args:
            temperature_matrix: Temperature matrix
            start_point: Starting (row, col) coordinates
            end_point: Ending (row, col) coordinates
            physical_distance_m: Physical distance between points (optional)

        Returns:
            Temperature gradient result
        """
        height = len(temperature_matrix)
        width = len(temperature_matrix[0]) if height > 0 else 0

        sr, sc = start_point
        er, ec = end_point

        # Validate coordinates
        if not (0 <= sr < height and 0 <= sc < width):
            raise ValueError(f"Start point {start_point} out of bounds")
        if not (0 <= er < height and 0 <= ec < width):
            raise ValueError(f"End point {end_point} out of bounds")

        # Get temperatures
        t_start = temperature_matrix[sr][sc]
        t_end = temperature_matrix[er][ec]

        temp_diff = t_end - t_start

        # Calculate pixel distance
        pixel_distance = self._decimal_sqrt(
            Decimal((er - sr) ** 2 + (ec - sc) ** 2)
        )

        # Determine physical distance
        if physical_distance_m:
            distance_m = physical_distance_m
        elif self.pixel_size_m:
            distance_m = pixel_distance * self.pixel_size_m
        else:
            distance_m = pixel_distance  # Use pixel units

        # Calculate gradient magnitude
        if distance_m > 0:
            gradient_magnitude = abs(temp_diff) / distance_m
        else:
            gradient_magnitude = Decimal("0")

        # Calculate direction (degrees from positive x-axis)
        direction = self._decimal_atan2(
            Decimal(er - sr),
            Decimal(ec - sc)
        )

        self._add_calculation_step(
            "temperature_gradient",
            {
                "start_point": start_point,
                "end_point": end_point,
                "t_start": t_start,
                "t_end": t_end
            },
            {
                "gradient_c_per_m": gradient_magnitude,
                "direction_degrees": direction
            },
            formula="gradient = |delta_T| / distance"
        )

        return TemperatureGradient(
            magnitude_c_per_m=gradient_magnitude.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            direction_degrees=direction,
            start_point=start_point,
            end_point=end_point,
            distance_m=distance_m.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            temperature_difference_c=temp_diff.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    # =========================================================================
    # COMPLETE ANALYSIS
    # =========================================================================

    def analyze_thermal_image(
        self,
        raw_data: List[List[Union[int, float, Decimal]]],
        emissivity: Decimal = Decimal("0.95"),
        ambient_temperature_c: Optional[Decimal] = None,
        reflected_temperature_c: Decimal = Decimal("20.0"),
        distance_m: Decimal = Decimal("1.0"),
        relative_humidity: Decimal = Decimal("50.0"),
        rois: Optional[List[ROIDefinition]] = None,
        generate_map: bool = True,
        assess_quality: bool = True
    ) -> ThermalAnalysisResult:
        """
        Perform complete thermal image analysis with provenance tracking.

        Args:
            raw_data: Raw radiometric data from IR camera
            emissivity: Surface emissivity
            ambient_temperature_c: Ambient temperature (auto-calculated if None)
            reflected_temperature_c: Reflected apparent temperature
            distance_m: Camera-to-target distance
            relative_humidity: Relative humidity percentage
            rois: List of Regions of Interest to analyze
            generate_map: Whether to generate temperature map
            assess_quality: Whether to assess image quality

        Returns:
            Complete thermal analysis result with provenance
        """
        import time
        start_time = time.perf_counter()

        analysis_id = str(uuid.uuid4())
        self._calculation_steps = []

        # Process temperature matrix
        temperature_matrix, config = self.process_temperature_matrix(
            raw_data,
            emissivity=emissivity,
            reflected_temperature_c=reflected_temperature_c,
            distance_m=distance_m,
            relative_humidity=relative_humidity
        )

        # Calculate ambient if not provided
        if ambient_temperature_c is None:
            ambient_temperature_c = self._calculate_ambient_reference(temperature_matrix)

        # Calculate statistics
        statistics = self._calculate_statistics(temperature_matrix)

        # Detect hotspots
        hotspots = self.detect_hotspots(
            temperature_matrix,
            ambient_temperature_c=ambient_temperature_c
        )

        # Classify anomalies
        anomaly_classifications: List[ThermalAnomalyClassification] = []
        for hotspot in hotspots:
            classification = self.classify_thermal_anomaly(
                hotspot,
                ambient_temperature_c=ambient_temperature_c
            )
            anomaly_classifications.append(classification)

        # Assess image quality
        if assess_quality:
            image_quality = self.assess_image_quality(
                temperature_matrix,
                ambient_temperature_c=ambient_temperature_c
            )
        else:
            image_quality = ImageQualityAssessment(
                overall_quality=ImageQualityLevel.ACCEPTABLE,
                focus_quality_score=Decimal("0.8"),
                thermal_contrast_score=Decimal("0.8"),
                spatial_resolution_score=Decimal("0.8"),
                noise_level_score=Decimal("0.8"),
                environmental_suitability_score=Decimal("0.8"),
                issues_detected=(),
                recommendations=(),
                is_usable_for_analysis=True
            )

        # Generate temperature map
        temperature_map = None
        if generate_map:
            temperature_map = self.generate_temperature_map(temperature_matrix)

        # Analyze ROIs
        roi_results: List[ROIAnalysisResult] = []
        if rois:
            for roi in rois:
                try:
                    roi_result = self.analyze_roi(temperature_matrix, roi)
                    roi_results.append(roi_result)
                except ValueError as e:
                    logger.warning(f"ROI analysis failed for {roi.name}: {e}")

        # Calculate processing time
        end_time = time.perf_counter()
        processing_time_ms = Decimal(str((end_time - start_time) * 1000))

        # Calculate provenance hash
        provenance_data = {
            "analysis_id": analysis_id,
            "config": {
                "width": config.width,
                "height": config.height,
                "emissivity": str(config.emissivity),
                "distance_m": str(config.distance_m)
            },
            "statistics": statistics.to_dict(),
            "hotspots_count": len(hotspots),
            "calculation_steps": self._calculation_steps
        }
        provenance_hash = self._calculate_provenance_hash(provenance_data)

        result = ThermalAnalysisResult(
            analysis_id=analysis_id,
            config=config,
            temperature_matrix_shape=(len(temperature_matrix), len(temperature_matrix[0]) if temperature_matrix else 0),
            statistics=statistics,
            hotspots=tuple(hotspots),
            anomaly_classifications=tuple(anomaly_classifications),
            image_quality=image_quality,
            temperature_map=temperature_map,
            roi_results=tuple(roi_results),
            provenance_hash=provenance_hash,
            calculation_steps=tuple(self._calculation_steps),
            analysis_timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

        logger.info(
            f"Thermal analysis complete: {analysis_id}, "
            f"{len(hotspots)} hotspots detected, "
            f"processing time: {processing_time_ms:.2f}ms"
        )

        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_roi_rectangle(
    name: str,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> ROIDefinition:
    """Create a rectangular ROI definition."""
    return ROIDefinition(
        roi_id=str(uuid.uuid4()),
        shape=ROIShape.RECTANGLE,
        name=name,
        vertices=(top_left, bottom_right)
    )


def create_roi_circle(
    name: str,
    center: Tuple[int, int],
    radius: int
) -> ROIDefinition:
    """Create a circular ROI definition."""
    return ROIDefinition(
        roi_id=str(uuid.uuid4()),
        shape=ROIShape.CIRCLE,
        name=name,
        vertices=(),
        center=center,
        radius=radius
    )


def create_roi_polygon(
    name: str,
    vertices: List[Tuple[int, int]]
) -> ROIDefinition:
    """Create a polygonal ROI definition."""
    return ROIDefinition(
        roi_id=str(uuid.uuid4()),
        shape=ROIShape.POLYGON,
        name=name,
        vertices=tuple(vertices)
    )


def verify_analysis_provenance(result: ThermalAnalysisResult) -> bool:
    """
    Verify the provenance hash of an analysis result.

    Args:
        result: Analysis result to verify

    Returns:
        True if provenance hash matches, False otherwise
    """
    provenance_data = {
        "analysis_id": result.analysis_id,
        "config": {
            "width": result.config.width,
            "height": result.config.height,
            "emissivity": str(result.config.emissivity),
            "distance_m": str(result.config.distance_m)
        },
        "statistics": result.statistics.to_dict(),
        "hotspots_count": len(result.hotspots),
        "calculation_steps": list(result.calculation_steps)
    }

    content = json.dumps(provenance_data, sort_keys=True, default=str)
    calculated_hash = hashlib.sha256(content.encode()).hexdigest()

    return calculated_hash == result.provenance_hash


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "IRCameraResolution",
    "ThermalAnomalyType",
    "ROIShape",
    "ImageQualityLevel",
    # Data structures
    "TemperatureMatrixConfig",
    "TemperaturePoint",
    "TemperatureStatistics",
    "Hotspot",
    "TemperatureGradient",
    "IsothermalContour",
    "ROIDefinition",
    "ROIAnalysisResult",
    "ThermalAnomalyClassification",
    "ImageQualityAssessment",
    "TemperatureMapResult",
    "ThermalAnalysisResult",
    "AnomalyThresholds",
    "ANOMALY_THRESHOLDS",
    # Main class
    "ThermalImageAnalyzer",
    # Utility functions
    "create_roi_rectangle",
    "create_roi_circle",
    "create_roi_polygon",
    "verify_analysis_provenance",
]
