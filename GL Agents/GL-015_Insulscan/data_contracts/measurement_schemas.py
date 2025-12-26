# -*- coding: utf-8 -*-
"""
GL-015 Insulscan: Measurement Schemas - Version 1.0

Provides validated data schemas for thermal imaging, surface temperature
measurements, ambient conditions, and inspection records.

This module defines Pydantic v2 models for:
- ThermalImageData: Infrared camera image with temperature matrix
- SurfaceTemperatureMeasurement: Point temperature measurements
- AmbientConditions: Environmental conditions affecting heat loss
- InspectionRecord: Visual and thermal inspection findings

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class MeasurementQuality(str, Enum):
    """Quality codes for measurement validity."""
    GOOD = "good"
    GOOD_ESTIMATED = "good_estimated"
    UNCERTAIN = "uncertain"
    UNCERTAIN_SUBSTITUTED = "uncertain_substituted"
    BAD = "bad"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_OUT_OF_RANGE = "bad_out_of_range"
    MISSING = "missing"


class DataSource(str, Enum):
    """Source of measurement data."""
    THERMAL_CAMERA = "thermal_camera"
    CONTACT_SENSOR = "contact_sensor"
    WEATHER_STATION = "weather_station"
    MANUAL_ENTRY = "manual_entry"
    HISTORIAN = "historian"
    CALCULATED = "calculated"
    EXTERNAL_API = "external_api"


class CameraType(str, Enum):
    """Type of thermal imaging camera."""
    HANDHELD = "handheld"
    FIXED_MOUNT = "fixed_mount"
    DRONE_MOUNTED = "drone_mounted"
    ROBOTIC = "robotic"
    SMARTPHONE = "smartphone"


class CalibrationStatus(str, Enum):
    """Calibration status of measurement device."""
    CALIBRATED = "calibrated"
    CALIBRATION_DUE = "calibration_due"
    CALIBRATION_OVERDUE = "calibration_overdue"
    UNCALIBRATED = "uncalibrated"


class InspectionMethod(str, Enum):
    """Method of inspection performed."""
    THERMAL_IMAGING = "thermal_imaging"
    VISUAL = "visual"
    TACTILE = "tactile"
    THICKNESS_GAUGE = "thickness_gauge"
    MOISTURE_METER = "moisture_meter"
    COMBINED = "combined"


class FindingSeverity(str, Enum):
    """Severity level of inspection findings."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class DamageType(str, Enum):
    """Types of insulation damage."""
    MISSING_INSULATION = "missing_insulation"
    CRUSHED = "crushed"
    WET_DAMAGED = "wet_damaged"
    THERMAL_DEGRADATION = "thermal_degradation"
    MECHANICAL_DAMAGE = "mechanical_damage"
    JACKET_DAMAGE = "jacket_damage"
    SEALANT_FAILURE = "sealant_failure"
    CORROSION_UNDER_INSULATION = "corrosion_under_insulation"
    VAPOR_BARRIER_BREACH = "vapor_barrier_breach"
    SETTLING = "settling"
    GAP = "gap"
    OTHER = "other"


# =============================================================================
# CALIBRATION DATA
# =============================================================================

class CalibrationData(BaseModel):
    """Calibration data for thermal imaging equipment."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "emissivity_setting": 0.95,
                    "reflected_temp_c": 25.0,
                    "distance_m": 3.0,
                    "atmospheric_temp_c": 22.0,
                    "relative_humidity_percent": 50.0,
                    "calibration_date": "2024-01-01T00:00:00Z"
                }
            ]
        }
    )

    # Emissivity settings
    emissivity_setting: float = Field(
        ...,
        ge=0.01,
        le=1.0,
        description="Emissivity value used for measurement"
    )
    emissivity_source: Optional[Literal[
        "material_table", "measured", "estimated", "default"
    ]] = Field(
        None,
        description="Source of emissivity value"
    )

    # Reflected temperature
    reflected_temp_c: float = Field(
        ...,
        ge=-50,
        le=200,
        description="Reflected apparent temperature in Celsius"
    )

    # Distance and atmospheric corrections
    distance_m: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Distance from camera to target in meters"
    )
    atmospheric_temp_c: float = Field(
        ...,
        ge=-50,
        le=60,
        description="Atmospheric temperature in Celsius"
    )
    relative_humidity_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relative humidity for atmospheric transmission"
    )

    # Calibration status
    calibration_date: Optional[datetime] = Field(
        None,
        description="Date of last camera calibration"
    )
    calibration_certificate: Optional[str] = Field(
        None,
        max_length=100,
        description="Calibration certificate reference"
    )
    calibration_status: CalibrationStatus = Field(
        default=CalibrationStatus.CALIBRATED,
        description="Current calibration status"
    )

    # Accuracy specifications
    accuracy_k: Optional[float] = Field(
        None,
        gt=0,
        le=10,
        description="Camera accuracy specification in Kelvin"
    )
    accuracy_percent: Optional[float] = Field(
        None,
        gt=0,
        le=20,
        description="Camera accuracy specification as percentage"
    )


# =============================================================================
# THERMAL IMAGE DATA
# =============================================================================

class ThermalImageData(BaseModel):
    """
    Infrared thermal image with temperature matrix.

    Captures complete thermal image data including temperature values,
    calibration settings, and camera metadata for heat loss analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "image_id": "TI-2024-00001234",
                    "asset_id": "INS-1001",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "resolution": [640, 480],
                    "min_temp_c": 25.5,
                    "max_temp_c": 85.3,
                    "avg_temp_c": 42.1,
                    "camera_model": "FLIR E96",
                    "data_quality": "good"
                }
            ]
        }
    )

    # Identifiers
    image_id: str = Field(
        default_factory=lambda: f"TI-{uuid.uuid4().hex[:12].upper()}",
        description="Unique thermal image identifier"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )
    timestamp: datetime = Field(
        ...,
        description="Image capture timestamp in UTC"
    )

    # Image specifications
    resolution: Tuple[int, int] = Field(
        ...,
        description="Image resolution (width, height) in pixels"
    )
    pixel_size_mm: Optional[float] = Field(
        None,
        gt=0,
        description="Size of each pixel in mm at target distance"
    )

    # Temperature matrix - stored as reference, not inline for large images
    temperature_matrix_ref: Optional[str] = Field(
        None,
        max_length=500,
        description="Reference to stored temperature matrix (e.g., S3 path)"
    )
    temperature_matrix_format: Optional[Literal[
        "numpy", "csv", "radiometric_jpeg", "raw"
    ]] = Field(
        None,
        description="Format of stored temperature matrix"
    )

    # Temperature statistics (always available even without full matrix)
    min_temp_c: float = Field(
        ...,
        ge=-50,
        le=1500,
        description="Minimum temperature in image in Celsius"
    )
    max_temp_c: float = Field(
        ...,
        ge=-50,
        le=1500,
        description="Maximum temperature in image in Celsius"
    )
    avg_temp_c: float = Field(
        ...,
        ge=-50,
        le=1500,
        description="Average temperature in image in Celsius"
    )
    std_temp_c: Optional[float] = Field(
        None,
        ge=0,
        description="Standard deviation of temperatures in Celsius"
    )

    # Regions of interest
    roi_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of regions of interest defined"
    )
    roi_temperatures: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Temperature stats per ROI: {roi_name: {min, max, avg}}"
    )

    # Hotspot detection
    hotspot_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of detected hotspots"
    )
    hotspot_locations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of hotspot locations and temperatures"
    )

    # Calibration data
    calibration_data: CalibrationData = Field(
        ...,
        description="Calibration parameters for temperature calculation"
    )

    # Camera information
    camera_type: CameraType = Field(
        default=CameraType.HANDHELD,
        description="Type of thermal camera"
    )
    camera_model: Optional[str] = Field(
        None,
        max_length=100,
        description="Camera model name"
    )
    camera_serial: Optional[str] = Field(
        None,
        max_length=50,
        description="Camera serial number"
    )
    lens_info: Optional[str] = Field(
        None,
        max_length=100,
        description="Lens specifications"
    )
    spectral_range: Optional[str] = Field(
        None,
        max_length=50,
        description="Spectral range (e.g., '7.5-14 um')"
    )

    # Visual image reference
    visual_image_ref: Optional[str] = Field(
        None,
        max_length=500,
        description="Reference to corresponding visual image"
    )

    # Capture metadata
    operator_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Thermographer/operator ID"
    )
    capture_location: Optional[str] = Field(
        None,
        max_length=500,
        description="Description of capture location/angle"
    )
    gps_coordinates: Optional[Tuple[float, float]] = Field(
        None,
        description="GPS coordinates (latitude, longitude)"
    )

    # Data quality
    data_quality: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Overall data quality assessment"
    )
    quality_notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Notes on image quality issues"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode="after")
    def validate_temperatures(self) -> "ThermalImageData":
        """Validate temperature consistency."""
        if self.min_temp_c > self.max_temp_c:
            raise ValueError("min_temp_c cannot be greater than max_temp_c")
        if self.avg_temp_c < self.min_temp_c or self.avg_temp_c > self.max_temp_c:
            raise ValueError("avg_temp_c must be between min_temp_c and max_temp_c")
        return self

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = (
            f"{self.image_id}"
            f"{self.asset_id}"
            f"{self.timestamp.isoformat()}"
            f"{self.min_temp_c:.2f}"
            f"{self.max_temp_c:.2f}"
            f"{self.avg_temp_c:.2f}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# SURFACE TEMPERATURE MEASUREMENT
# =============================================================================

class SurfaceTemperatureMeasurement(BaseModel):
    """
    Point surface temperature measurement.

    Captures discrete temperature readings from contact sensors
    or spot measurements from thermal cameras.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "sensor_id": "TC-1001-01",
                    "asset_id": "INS-1001",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "temp_c": 45.2,
                    "location": "Pipe section 3, top",
                    "data_quality": "good"
                }
            ]
        }
    )

    # Identifiers
    measurement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique measurement identifier"
    )
    sensor_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Sensor or measurement point identifier"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC"
    )

    # Temperature value
    temp_c: float = Field(
        ...,
        ge=-50,
        le=1500,
        description="Measured surface temperature in Celsius"
    )
    temp_uncertainty_c: Optional[float] = Field(
        None,
        ge=0,
        le=50,
        description="Temperature measurement uncertainty in Celsius"
    )

    # Location on asset
    location: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Measurement location description on asset"
    )
    location_x_m: Optional[float] = Field(
        None,
        description="X coordinate in local asset reference frame (meters)"
    )
    location_y_m: Optional[float] = Field(
        None,
        description="Y coordinate in local asset reference frame (meters)"
    )
    location_angle_deg: Optional[float] = Field(
        None,
        ge=0,
        le=360,
        description="Angular position for cylindrical surfaces (degrees)"
    )

    # Measurement method
    source: DataSource = Field(
        default=DataSource.CONTACT_SENSOR,
        description="Source of measurement"
    )
    sensor_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Type of sensor (thermocouple, RTD, IR spot, etc.)"
    )

    # Data quality
    data_quality: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality code for measurement"
    )
    quality_notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Notes on measurement quality"
    )

    # For thermal camera spot measurements
    emissivity_used: Optional[float] = Field(
        None,
        ge=0.01,
        le=1.0,
        description="Emissivity value used (for IR spot measurements)"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Ensure timestamp is UTC."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v


# =============================================================================
# AMBIENT CONDITIONS
# =============================================================================

class AmbientConditions(BaseModel):
    """
    Environmental conditions affecting heat loss calculations.

    Captures all ambient parameters needed for accurate heat loss
    calculations including temperature, wind, and solar radiation.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "temp_c": 22.5,
                    "humidity_percent": 55.0,
                    "wind_speed_ms": 2.5,
                    "solar_radiation_w_m2": 350.0,
                    "source": "weather_station"
                }
            ]
        }
    )

    # Timestamp
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC"
    )

    # Temperature
    temp_c: float = Field(
        ...,
        ge=-60,
        le=60,
        description="Ambient air temperature in Celsius"
    )
    temp_source: Optional[str] = Field(
        None,
        max_length=100,
        description="Source of temperature measurement"
    )

    # Humidity
    humidity_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relative humidity in percent"
    )
    dew_point_c: Optional[float] = Field(
        None,
        ge=-80,
        le=50,
        description="Dew point temperature in Celsius"
    )

    # Wind conditions
    wind_speed_ms: float = Field(
        ...,
        ge=0,
        le=100,
        description="Wind speed in meters per second"
    )
    wind_direction_deg: Optional[float] = Field(
        None,
        ge=0,
        le=360,
        description="Wind direction in degrees from north"
    )
    wind_gust_ms: Optional[float] = Field(
        None,
        ge=0,
        le=150,
        description="Maximum wind gust speed in m/s"
    )

    # Solar radiation
    solar_radiation_w_m2: float = Field(
        default=0.0,
        ge=0,
        le=1500,
        description="Solar radiation intensity in W/m^2"
    )
    cloud_cover_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Cloud cover percentage"
    )
    is_daytime: Optional[bool] = Field(
        None,
        description="Whether measurement was during daytime"
    )

    # Atmospheric pressure
    pressure_mbar: Optional[float] = Field(
        None,
        ge=800,
        le=1100,
        description="Atmospheric pressure in millibars"
    )

    # Precipitation
    precipitation: Optional[Literal[
        "none", "light_rain", "rain", "heavy_rain", "snow", "sleet"
    ]] = Field(
        None,
        description="Current precipitation type"
    )
    precipitation_rate_mm_h: Optional[float] = Field(
        None,
        ge=0,
        description="Precipitation rate in mm/hour"
    )

    # Indoor/outdoor
    is_indoor: bool = Field(
        default=False,
        description="Whether conditions are for indoor environment"
    )
    enclosure_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Type of enclosure if applicable"
    )

    # Data source
    source: DataSource = Field(
        default=DataSource.WEATHER_STATION,
        description="Source of ambient data"
    )
    weather_station_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Weather station identifier"
    )
    location: Optional[str] = Field(
        None,
        max_length=200,
        description="Location description"
    )

    # Data quality
    data_quality: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality code for ambient data"
    )


# =============================================================================
# INSPECTION FINDING
# =============================================================================

class InspectionFinding(BaseModel):
    """Individual finding from inspection."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "finding_id": "FND-001",
                    "damage_type": "jacket_damage",
                    "severity": "moderate",
                    "location": "Pipe section 3, bottom",
                    "description": "Dented aluminum jacket, insulation visible",
                    "affected_area_m2": 0.5
                }
            ]
        }
    )

    finding_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique finding identifier"
    )
    damage_type: DamageType = Field(
        ...,
        description="Type of damage/issue found"
    )
    severity: FindingSeverity = Field(
        ...,
        description="Severity of the finding"
    )
    location: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Location of finding on asset"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Detailed description of finding"
    )

    # Extent of damage
    affected_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Affected area in square meters"
    )
    affected_length_m: Optional[float] = Field(
        None,
        ge=0,
        description="Affected length in meters"
    )
    affected_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage of asset affected"
    )

    # Thermal impact
    surface_temp_c: Optional[float] = Field(
        None,
        description="Surface temperature at finding location"
    )
    temp_anomaly_c: Optional[float] = Field(
        None,
        description="Temperature difference from expected"
    )
    estimated_heat_loss_w: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated heat loss from this defect in Watts"
    )

    # Moisture detection
    moisture_detected: bool = Field(
        default=False,
        description="Whether moisture was detected"
    )
    moisture_level: Optional[Literal["dry", "damp", "wet", "saturated"]] = Field(
        None,
        description="Moisture level if detected"
    )

    # CUI (Corrosion Under Insulation) risk
    cui_risk: Optional[Literal["none", "low", "medium", "high", "confirmed"]] = Field(
        None,
        description="Risk of corrosion under insulation"
    )

    # Recommendations
    recommended_action: Optional[str] = Field(
        None,
        max_length=500,
        description="Recommended corrective action"
    )
    priority: Optional[Literal["immediate", "urgent", "scheduled", "monitor"]] = Field(
        None,
        description="Priority for corrective action"
    )
    estimated_repair_cost: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated repair cost in local currency"
    )

    # Photo references
    photo_refs: List[str] = Field(
        default_factory=list,
        description="References to finding photos"
    )
    thermal_image_ref: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to thermal image of finding"
    )


# =============================================================================
# INSPECTION RECORD
# =============================================================================

class InspectionRecord(BaseModel):
    """
    Complete inspection record for insulation asset.

    Captures visual and thermal inspection findings with
    condition assessment and recommendations.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "inspection_id": "INSP-2024-00042",
                    "asset_id": "INS-1001",
                    "inspector_id": "TECH-001",
                    "date": "2024-01-15T10:00:00Z",
                    "inspection_method": "combined",
                    "overall_condition": "fair",
                    "visual_findings": ["Minor jacket damage at support"],
                    "thermal_findings": ["2 hotspots detected"]
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    inspection_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique inspection identifier"
    )
    asset_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to insulation asset"
    )

    # Inspector information
    inspector_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Inspector identifier"
    )
    inspector_name: Optional[str] = Field(
        None,
        max_length=200,
        description="Inspector name"
    )
    inspector_certification: Optional[str] = Field(
        None,
        max_length=200,
        description="Inspector certification/qualification"
    )
    inspector_company: Optional[str] = Field(
        None,
        max_length=200,
        description="Inspection company"
    )

    # Inspection details
    date: datetime = Field(
        ...,
        description="Inspection date and time"
    )
    duration_minutes: Optional[int] = Field(
        None,
        ge=1,
        description="Inspection duration in minutes"
    )
    inspection_method: InspectionMethod = Field(
        ...,
        description="Primary inspection method used"
    )
    methods_used: List[InspectionMethod] = Field(
        default_factory=list,
        description="All inspection methods used"
    )

    # Equipment used
    equipment_used: Optional[List[str]] = Field(
        None,
        description="List of equipment/instruments used"
    )
    thermal_camera_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Thermal camera identifier if used"
    )

    # Ambient conditions during inspection
    ambient_conditions: Optional[AmbientConditions] = Field(
        None,
        description="Ambient conditions during inspection"
    )

    # Overall assessment
    overall_condition: Literal[
        "excellent", "good", "fair", "poor", "critical"
    ] = Field(
        ...,
        description="Overall condition rating"
    )
    condition_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Numeric condition score (0-100)"
    )
    condition_trend: Optional[Literal[
        "improving", "stable", "degrading", "unknown"
    ]] = Field(
        None,
        description="Condition trend since last inspection"
    )

    # Findings summary
    visual_findings: List[str] = Field(
        default_factory=list,
        description="Summary of visual inspection findings"
    )
    thermal_findings: List[str] = Field(
        default_factory=list,
        description="Summary of thermal inspection findings"
    )

    # Detailed findings
    findings: List[InspectionFinding] = Field(
        default_factory=list,
        description="List of detailed findings"
    )
    hotspot_count: int = Field(
        default=0,
        ge=0,
        description="Number of thermal hotspots detected"
    )
    defect_count: int = Field(
        default=0,
        ge=0,
        description="Total number of defects found"
    )

    # Heat loss assessment
    estimated_total_heat_loss_w: Optional[float] = Field(
        None,
        ge=0,
        description="Total estimated heat loss from defects in Watts"
    )
    heat_loss_percentage_increase: Optional[float] = Field(
        None,
        ge=0,
        description="Heat loss increase vs design due to defects (%)"
    )

    # Photos and thermal images
    photos: List[str] = Field(
        default_factory=list,
        description="References to inspection photos"
    )
    thermal_images: List[str] = Field(
        default_factory=list,
        description="References to thermal images"
    )

    # Recommendations
    recommendations: Optional[str] = Field(
        None,
        max_length=2000,
        description="Overall recommendations"
    )
    next_inspection_date: Optional[datetime] = Field(
        None,
        description="Recommended next inspection date"
    )
    repair_required: bool = Field(
        default=False,
        description="Whether repair is required"
    )
    repair_urgency: Optional[Literal[
        "immediate", "urgent", "scheduled", "planned"
    ]] = Field(
        None,
        description="Urgency of required repairs"
    )

    # Work order linkage
    work_order_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Associated work order ID"
    )

    # Notes
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional inspection notes"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record last update timestamp"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    @property
    def critical_findings(self) -> List[InspectionFinding]:
        """Return list of critical findings."""
        return [f for f in self.findings if f.severity == FindingSeverity.CRITICAL]

    @property
    def major_findings(self) -> List[InspectionFinding]:
        """Return list of major findings."""
        return [f for f in self.findings if f.severity == FindingSeverity.MAJOR]

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for audit trail."""
        content = (
            f"{self.inspection_id}"
            f"{self.asset_id}"
            f"{self.date.isoformat()}"
            f"{self.overall_condition}"
            f"{self.defect_count}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# EXPORTS
# =============================================================================

MEASUREMENT_SCHEMAS = {
    "MeasurementQuality": MeasurementQuality,
    "DataSource": DataSource,
    "CameraType": CameraType,
    "CalibrationStatus": CalibrationStatus,
    "InspectionMethod": InspectionMethod,
    "FindingSeverity": FindingSeverity,
    "DamageType": DamageType,
    "CalibrationData": CalibrationData,
    "ThermalImageData": ThermalImageData,
    "SurfaceTemperatureMeasurement": SurfaceTemperatureMeasurement,
    "AmbientConditions": AmbientConditions,
    "InspectionFinding": InspectionFinding,
    "InspectionRecord": InspectionRecord,
}

__all__ = [
    # Enumerations
    "MeasurementQuality",
    "DataSource",
    "CameraType",
    "CalibrationStatus",
    "InspectionMethod",
    "FindingSeverity",
    "DamageType",
    # Supporting models
    "CalibrationData",
    "InspectionFinding",
    # Main schemas
    "ThermalImageData",
    "SurfaceTemperatureMeasurement",
    "AmbientConditions",
    "InspectionRecord",
    # Export dictionary
    "MEASUREMENT_SCHEMAS",
]
