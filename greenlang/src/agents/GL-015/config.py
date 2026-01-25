# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN Insulation Inspection Agent - Configuration Management

This module provides comprehensive configuration for the Insulation Inspection
agent, including all enumerations, Pydantic models for inputs/outputs, validation
rules, and settings management.

Agent ID: GL-015
Codename: INSULSCAN
Version: 1.0.0
Category: Energy Conservation
Type: Monitor

The configuration follows GreenLang's zero-hallucination principle with
deterministic heat transfer calculations and complete data validation.

Example:
    >>> from config import Settings, InsulationConfig
    >>> from config import ThermalImageData, AmbientConditions, EquipmentParameters
    >>>
    >>> settings = Settings()
    >>> config = InsulationConfig()
    >>> ambient = AmbientConditions(
    ...     ambient_temp_c=25.0,
    ...     wind_speed_m_s=2.0,
    ...     relative_humidity_percent=60.0
    ... )

Author: GreenLang AI Agent Factory
License: Apache-2.0
"""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure module logger
logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMERATIONS
# ==============================================================================


class InsulationType(str, Enum):
    """
    Insulation material type classification.

    Defines the physical insulation material, which determines thermal
    conductivity, temperature limits, and moisture resistance.
    """

    MINERAL_WOOL = "mineral_wool"
    """Rock wool or slag wool insulation - versatile, high-temp capable."""

    CALCIUM_SILICATE = "calcium_silicate"
    """Calcium silicate insulation - high-temp, fire resistant."""

    CELLULAR_GLASS = "cellular_glass"
    """Closed-cell glass foam - excellent moisture resistance."""

    PERLITE = "perlite"
    """Expanded perlite insulation - high-temp, fire resistant."""

    AEROGEL = "aerogel"
    """Aerogel-based insulation - lowest thermal conductivity."""

    POLYURETHANE = "polyurethane"
    """Polyurethane foam - low-temp, excellent k-value."""

    POLYSTYRENE = "polystyrene"
    """Expanded or extruded polystyrene - low-temp only."""

    PHENOLIC_FOAM = "phenolic_foam"
    """Phenolic foam insulation - good fire properties."""

    FIBERGLASS = "fiberglass"
    """Glass fiber insulation - economical, moderate temp."""

    MICROPOROUS = "microporous"
    """Microporous silica insulation - very low k-value."""

    CERAMIC_FIBER = "ceramic_fiber"
    """Ceramic fiber insulation - very high temp applications."""

    UNKNOWN = "unknown"
    """Unknown or unspecified insulation type."""


class InsulationMaterial(str, Enum):
    """
    Detailed insulation material specification.

    Provides finer granularity for specific product types.
    """

    # Mineral Wool Types
    ROCKWOOL_INDUSTRIAL = "rockwool_industrial"
    ROCKWOOL_MARINE = "rockwool_marine"
    SLAGWOOL = "slagwool"

    # Calcium Silicate Types
    CALSIL_STANDARD = "calsil_standard"
    CALSIL_HIGH_DENSITY = "calsil_high_density"

    # Cellular Glass Types
    FOAMGLAS_STANDARD = "foamglas_standard"
    FOAMGLAS_HIGH_LOAD = "foamglas_high_load"

    # Perlite Types
    PERLITE_STANDARD = "perlite_standard"
    PERLITE_HIGH_TEMP = "perlite_high_temp"

    # Aerogel Types
    AEROGEL_BLANKET = "aerogel_blanket"
    AEROGEL_BOARD = "aerogel_board"

    # Foam Types
    PIR_FOAM = "pir_foam"
    PUR_FOAM = "pur_foam"
    XPS_FOAM = "xps_foam"
    EPS_FOAM = "eps_foam"

    # Fiber Types
    FIBERGLASS_STANDARD = "fiberglass_standard"
    FIBERGLASS_HIGH_TEMP = "fiberglass_high_temp"
    CERAMIC_FIBER_BLANKET = "ceramic_fiber_blanket"
    CERAMIC_FIBER_BOARD = "ceramic_fiber_board"


class JacketMaterial(str, Enum):
    """
    Weather barrier/jacketing material classification.

    Defines the outer protective cladding material.
    """

    ALUMINUM = "aluminum"
    """Aluminum jacketing - most common, good weather protection."""

    STAINLESS_STEEL = "stainless_steel"
    """Stainless steel jacketing - corrosive environments."""

    GALVANIZED_STEEL = "galvanized_steel"
    """Galvanized steel jacketing - economical option."""

    PVC = "pvc"
    """PVC jacketing - cold service, indoor use."""

    NONE = "none"
    """No jacketing - indoor or temporary installations."""

    PAINTED_METAL = "painted_metal"
    """Painted carbon steel or aluminum."""

    FIBERGLASS_REINFORCED = "fiberglass_reinforced"
    """Fiberglass reinforced plastic jacketing."""

    COMPOSITE = "composite"
    """Composite or multi-layer jacketing."""


class DegradationSeverity(str, Enum):
    """
    Insulation degradation severity classification.

    Categorizes degradation level based on performance loss.
    """

    NONE = "none"
    """No degradation detected - performing as designed."""

    MINOR = "minor"
    """0-15% performance loss - monitor situation."""

    MODERATE = "moderate"
    """15-30% performance loss - schedule repair."""

    SEVERE = "severe"
    """30-50% performance loss - urgent repair needed."""

    FAILED = "failed"
    """>50% performance loss or missing - immediate action."""


class RepairPriority(str, Enum):
    """
    Repair urgency classification.

    Defines urgency levels for repair recommendations.
    """

    ROUTINE = "routine"
    """Schedule for next turnaround - not urgent."""

    PLANNED = "planned"
    """Schedule within 6 months - moderate priority."""

    URGENT = "urgent"
    """Schedule within 30 days - high priority."""

    CRITICAL = "critical"
    """Immediate action required - safety/cost critical."""


class SurfaceType(str, Enum):
    """
    Surface orientation classification.

    Affects convective heat transfer coefficient calculations.
    """

    HORIZONTAL_TOP = "horizontal_top"
    """Horizontal surface, heat flow upward."""

    HORIZONTAL_BOTTOM = "horizontal_bottom"
    """Horizontal surface, heat flow downward."""

    VERTICAL = "vertical"
    """Vertical surface."""

    ANGLED_UP = "angled_up"
    """Angled surface, heat flow predominantly upward."""

    ANGLED_DOWN = "angled_down"
    """Angled surface, heat flow predominantly downward."""

    CYLINDRICAL_HORIZONTAL = "cylindrical_horizontal"
    """Horizontal cylinder (pipe)."""

    CYLINDRICAL_VERTICAL = "cylindrical_vertical"
    """Vertical cylinder."""

    SPHERICAL = "spherical"
    """Spherical surface."""


class EquipmentType(str, Enum):
    """
    Equipment type classification.

    Defines the type of insulated equipment.
    """

    PIPE = "pipe"
    """Piping - straight runs, elbows, tees."""

    VESSEL = "vessel"
    """Pressure vessel - drums, reactors."""

    TANK = "tank"
    """Storage tank - atmospheric or pressurized."""

    EXCHANGER = "exchanger"
    """Heat exchanger - shell, heads, channels."""

    COLUMN = "column"
    """Distillation or absorption column."""

    DUCT = "duct"
    """Air or gas ductwork."""

    VALVE = "valve"
    """Valves - gate, globe, ball, etc."""

    FLANGE = "flange"
    """Flanged connections."""

    FITTING = "fitting"
    """Pipe fittings - elbows, tees, reducers."""

    TURBINE = "turbine"
    """Steam or gas turbine."""

    PUMP = "pump"
    """Pump - casing and connections."""

    BOILER = "boiler"
    """Boiler or steam generator."""

    REACTOR = "reactor"
    """Chemical reactor."""

    FURNACE = "furnace"
    """Process furnace or heater."""


class WeatherCondition(str, Enum):
    """
    Weather/sky condition classification.

    Affects solar loading and radiative heat transfer.
    """

    CLEAR = "clear"
    """Clear sky - maximum solar loading."""

    PARTLY_CLOUDY = "partly_cloudy"
    """Partially cloudy - moderate solar loading."""

    OVERCAST = "overcast"
    """Overcast sky - minimal solar loading."""

    RAIN = "rain"
    """Raining - affects surface conditions."""

    NIGHT = "night"
    """Nighttime - no solar loading."""

    FOG = "fog"
    """Foggy conditions - affects visibility."""

    SNOW = "snow"
    """Snow or ice conditions."""


class CameraType(str, Enum):
    """
    Thermal camera manufacturer classification.

    Defines supported thermal imaging camera brands.
    """

    FLIR = "FLIR"
    """FLIR Systems thermal cameras."""

    FLUKE = "Fluke"
    """Fluke Corporation thermal cameras."""

    TESTO = "Testo"
    """Testo thermal cameras."""

    OPTRIS = "Optris"
    """Optris thermal cameras."""

    INFRATEC = "InfraTec"
    """InfraTec thermal cameras."""

    SEEK = "Seek"
    """Seek Thermal cameras."""

    HIKMICRO = "HIKMICRO"
    """HIKMICRO thermal cameras."""

    OTHER = "Other"
    """Other/unknown camera type."""


class EmissivityClass(str, Enum):
    """
    Surface emissivity classification.

    Categorizes surfaces by typical emissivity range.
    """

    HIGH = "high"
    """High emissivity (0.85-0.98) - painted, oxidized surfaces."""

    MEDIUM = "medium"
    """Medium emissivity (0.50-0.85) - weathered metals."""

    LOW = "low"
    """Low emissivity (0.20-0.50) - lightly oxidized metals."""

    VERY_LOW = "very_low"
    """Very low emissivity (0.05-0.20) - polished metals."""


class FailureMode(str, Enum):
    """
    Insulation failure mode classification.

    Identifies the primary failure mechanism.
    """

    MOISTURE_INGRESS = "moisture_ingress"
    """Water infiltration degrading insulation."""

    MECHANICAL_DAMAGE = "mechanical_damage"
    """Physical damage from impacts, foot traffic."""

    THERMAL_DEGRADATION = "thermal_degradation"
    """Degradation from thermal cycling or over-temperature."""

    COMPRESSION = "compression"
    """Loss of thickness due to compression."""

    MISSING_INSULATION = "missing_insulation"
    """Sections of insulation completely missing."""

    JACKET_DAMAGE = "jacket_damage"
    """Weather barrier damage allowing water entry."""

    SEAL_FAILURE = "seal_failure"
    """Failed caulk, mastic, or sealant."""

    CUI_CORROSION = "cui_corrosion"
    """Corrosion Under Insulation detected."""

    SETTLING = "settling"
    """Insulation settling/sagging in vertical runs."""

    WEATHERING = "weathering"
    """General deterioration from weather exposure."""

    AGE_DEGRADATION = "age_degradation"
    """Normal aging and material degradation."""

    INSTALLATION_DEFECT = "installation_defect"
    """Original installation deficiency."""


class DamageType(str, Enum):
    """
    Damage type classification.

    Detailed damage categorization for assessment.
    """

    # Moisture-related
    MOISTURE_INFILTRATION = "moisture_infiltration"
    WATERLOGGED = "waterlogged"
    CONDENSATION = "condensation"

    # Physical damage
    IMPACT_DAMAGE = "impact_damage"
    ABRASION = "abrasion"
    PUNCTURE = "puncture"
    CRUSHING = "crushing"

    # Thermal damage
    SINTERING = "sintering"
    DELAMINATION = "delamination"
    SHRINKAGE = "shrinkage"
    MELTING = "melting"

    # Structural
    GAPS = "gaps"
    VOIDS = "voids"
    MISSING_SECTIONS = "missing_sections"
    SAGGING = "sagging"

    # Surface/Jacket
    JACKET_CORROSION = "jacket_corrosion"
    JACKET_DENTS = "jacket_dents"
    JACKET_HOLES = "jacket_holes"
    SEALANT_FAILURE = "sealant_failure"
    BAND_LOOSENING = "band_loosening"


class InspectionMethod(str, Enum):
    """
    Inspection method classification.

    Defines inspection techniques used.
    """

    INFRARED_THERMOGRAPHY = "infrared_thermography"
    """IR thermal imaging inspection."""

    VISUAL = "visual"
    """Visual inspection only."""

    ULTRASONIC = "ultrasonic"
    """Ultrasonic thickness measurement."""

    MOISTURE_PROBE = "moisture_probe"
    """Moisture probe testing."""

    DESTRUCTIVE = "destructive"
    """Destructive sampling/testing."""

    NEUTRON_BACKSCATTER = "neutron_backscatter"
    """Neutron backscatter moisture detection."""

    PULSED_EDDY_CURRENT = "pulsed_eddy_current"
    """Pulsed eddy current for CUI detection."""


class MoistureState(str, Enum):
    """
    Moisture content classification.

    Categorizes moisture presence severity.
    """

    DRY = "dry"
    """No moisture detected."""

    TRACE = "trace"
    """Trace moisture - minor concern."""

    MODERATE = "moderate"
    """Moderate moisture - needs attention."""

    SATURATED = "saturated"
    """Saturated - severe condition."""


class CorrosionRisk(str, Enum):
    """
    Corrosion Under Insulation (CUI) risk classification.

    Categorizes CUI risk level.
    """

    LOW = "low"
    """Low CUI risk - operating temp outside CUI range."""

    MEDIUM = "medium"
    """Medium CUI risk - some risk factors present."""

    HIGH = "high"
    """High CUI risk - multiple risk factors."""

    CRITICAL = "critical"
    """Critical CUI risk - immediate inspection needed."""


class OverallCondition(str, Enum):
    """
    Overall insulation condition classification.

    High-level condition assessment.
    """

    EXCELLENT = "excellent"
    """>95% efficiency, no defects detected."""

    GOOD = "good"
    """85-95% efficiency, minor issues only."""

    FAIR = "fair"
    """70-85% efficiency, some degradation."""

    POOR = "poor"
    """50-70% efficiency, significant issues."""

    CRITICAL = "critical"
    """<50% efficiency, failed or missing."""


class RecommendedAction(str, Enum):
    """
    Recommended repair action classification.

    Defines the type of repair recommended.
    """

    MONITOR = "monitor"
    """Continue monitoring - no immediate action."""

    SPOT_REPAIR = "spot_repair"
    """Localized repair of specific areas."""

    PARTIAL_REPLACEMENT = "partial_replacement"
    """Replace damaged sections only."""

    FULL_REPLACEMENT = "full_replacement"
    """Complete replacement of insulation system."""

    EMERGENCY_REPAIR = "emergency_repair"
    """Immediate temporary repair required."""

    REINSULATE = "reinsulate"
    """Remove and reinsulate entire system."""


class ProcessFluid(str, Enum):
    """
    Process fluid type classification.

    Affects heat content and economic calculations.
    """

    STEAM = "steam"
    """Steam (saturated or superheated)."""

    HOT_WATER = "hot_water"
    """Hot water."""

    THERMAL_OIL = "thermal_oil"
    """Thermal oil/heat transfer fluid."""

    PROCESS_GAS = "process_gas"
    """Process gas (various)."""

    PROCESS_LIQUID = "process_liquid"
    """Process liquid (various)."""

    CONDENSATE = "condensate"
    """Steam condensate."""

    AIR = "air"
    """Hot air or flue gas."""

    NITROGEN = "nitrogen"
    """Hot nitrogen."""

    HYDROGEN = "hydrogen"
    """Hot hydrogen."""

    OTHER = "other"
    """Other process fluid."""


# ==============================================================================
# THERMAL IMAGING INPUT MODELS
# ==============================================================================


class Hotspot(BaseModel):
    """
    Thermal hotspot identified in IR image.

    Represents a localized area of elevated temperature indicating
    potential insulation degradation.

    Attributes:
        hotspot_id: Unique identifier for this hotspot.
        center_x: X-coordinate of hotspot center (pixels).
        center_y: Y-coordinate of hotspot center (pixels).
        radius_pixels: Approximate radius of hotspot.
        max_temp_c: Maximum temperature within hotspot.
        avg_temp_c: Average temperature within hotspot.
        min_temp_c: Minimum temperature within hotspot.
        area_pixels: Area of hotspot in pixels.
        area_m2: Estimated area in square meters.
        temp_elevation_c: Temperature elevation above ambient.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    hotspot_id: str = Field(
        ...,
        max_length=50,
        description="Unique identifier for this hotspot",
    )

    center_x: int = Field(
        ...,
        ge=0,
        description="X-coordinate of hotspot center (pixels)",
    )

    center_y: int = Field(
        ...,
        ge=0,
        description="Y-coordinate of hotspot center (pixels)",
    )

    radius_pixels: int = Field(
        ...,
        ge=1,
        description="Approximate radius of hotspot (pixels)",
    )

    max_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=1000.0,
        description="Maximum temperature within hotspot (Celsius)",
    )

    avg_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=1000.0,
        description="Average temperature within hotspot (Celsius)",
    )

    min_temp_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=1000.0,
        description="Minimum temperature within hotspot (Celsius)",
    )

    area_pixels: int = Field(
        ...,
        ge=1,
        description="Area of hotspot in pixels",
    )

    area_m2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated area in square meters",
    )

    temp_elevation_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Temperature elevation above ambient",
    )


class RegionOfInterest(BaseModel):
    """
    User-defined region of interest in thermal image.

    Represents a specific area to analyze within the thermal image.

    Attributes:
        roi_id: Unique identifier for this ROI.
        roi_type: Shape type (rectangle, ellipse, polygon).
        x_min: Minimum X coordinate.
        y_min: Minimum Y coordinate.
        x_max: Maximum X coordinate.
        y_max: Maximum Y coordinate.
        coordinates: List of coordinates for polygon.
        max_temp_c: Maximum temperature in ROI.
        min_temp_c: Minimum temperature in ROI.
        avg_temp_c: Average temperature in ROI.
        std_temp_c: Temperature standard deviation.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    roi_id: str = Field(
        ...,
        max_length=50,
        description="Unique identifier for this ROI",
    )

    roi_type: str = Field(
        ...,
        pattern="^(rectangle|ellipse|polygon)$",
        description="Shape type",
    )

    x_min: int = Field(
        ...,
        ge=0,
        description="Minimum X coordinate",
    )

    y_min: int = Field(
        ...,
        ge=0,
        description="Minimum Y coordinate",
    )

    x_max: int = Field(
        ...,
        ge=0,
        description="Maximum X coordinate",
    )

    y_max: int = Field(
        ...,
        ge=0,
        description="Maximum Y coordinate",
    )

    coordinates: Optional[List[Tuple[int, int]]] = Field(
        default=None,
        description="List of coordinates for polygon",
    )

    max_temp_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=1000.0,
        description="Maximum temperature in ROI (Celsius)",
    )

    min_temp_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=1000.0,
        description="Minimum temperature in ROI (Celsius)",
    )

    avg_temp_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=1000.0,
        description="Average temperature in ROI (Celsius)",
    )

    std_temp_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Temperature standard deviation (Celsius)",
    )


class TemperatureMatrix(BaseModel):
    """
    2D temperature matrix from thermal image.

    Represents the temperature values for each pixel in the thermal image.

    Attributes:
        data: 2D array of temperature values in Celsius.
        rows: Number of rows (height).
        cols: Number of columns (width).
        dtype: Data type of matrix values.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    data: List[List[float]] = Field(
        ...,
        description="2D array of temperature values in Celsius",
    )

    rows: int = Field(
        ...,
        ge=1,
        description="Number of rows (height)",
    )

    cols: int = Field(
        ...,
        ge=1,
        description="Number of columns (width)",
    )

    dtype: str = Field(
        default="float32",
        description="Data type of matrix values",
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "TemperatureMatrix":
        """Validate matrix dimensions match specified rows/cols."""
        if len(self.data) != self.rows:
            raise ValueError(
                f"Matrix has {len(self.data)} rows but rows={self.rows} specified"
            )
        if self.data and len(self.data[0]) != self.cols:
            raise ValueError(
                f"Matrix has {len(self.data[0])} cols but cols={self.cols} specified"
            )
        return self


class ThermalImageData(BaseModel):
    """
    Thermal image data from IR camera.

    Contains the complete thermal image data including temperature matrix,
    camera settings, and identified anomalies.

    Attributes:
        image_id: Unique image identifier.
        camera_type: Thermal camera manufacturer.
        camera_model: Camera model identifier.
        temperature_matrix: 2D array of temperatures (optional - can be large).
        min_temp_c: Minimum temperature in image.
        max_temp_c: Maximum temperature in image.
        avg_temp_c: Average temperature in image.
        resolution_width: Image width in pixels.
        resolution_height: Image height in pixels.
        emissivity_setting: Camera emissivity setting.
        reflected_temp_c: Reflected apparent temperature.
        distance_m: Distance from camera to target.
        capture_timestamp: Image capture timestamp.
        hotspots: Identified hotspot regions.
        regions_of_interest: User-defined ROIs.

    Example:
        >>> image_data = ThermalImageData(
        ...     image_id="IMG-2024-001",
        ...     camera_type=CameraType.FLIR,
        ...     min_temp_c=25.0,
        ...     max_temp_c=85.0,
        ...     avg_temp_c=45.0,
        ...     resolution_width=640,
        ...     resolution_height=480,
        ...     emissivity_setting=0.95,
        ...     capture_timestamp=datetime.now()
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    image_id: str = Field(
        ...,
        max_length=50,
        description="Unique image identifier",
    )

    camera_type: CameraType = Field(
        ...,
        description="Thermal camera manufacturer",
    )

    camera_model: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Camera model identifier",
    )

    temperature_matrix: Optional[TemperatureMatrix] = Field(
        default=None,
        description="2D array of temperature values in Celsius",
    )

    min_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=1000.0,
        description="Minimum temperature in image (Celsius)",
    )

    max_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=1000.0,
        description="Maximum temperature in image (Celsius)",
    )

    avg_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=1000.0,
        description="Average temperature in image (Celsius)",
    )

    resolution_width: int = Field(
        ...,
        ge=80,
        le=4096,
        description="Image width in pixels",
    )

    resolution_height: int = Field(
        ...,
        ge=60,
        le=4096,
        description="Image height in pixels",
    )

    emissivity_setting: float = Field(
        ...,
        ge=0.10,
        le=0.98,
        description="Camera emissivity setting",
    )

    reflected_temp_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=100.0,
        description="Reflected apparent temperature (Celsius)",
    )

    distance_m: float = Field(
        default=3.0,
        ge=0.1,
        le=100.0,
        description="Distance from camera to target (m)",
    )

    atmospheric_temp_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=60.0,
        description="Atmospheric temperature (Celsius)",
    )

    relative_humidity_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Relative humidity for transmission correction (%)",
    )

    capture_timestamp: datetime = Field(
        ...,
        description="Image capture timestamp",
    )

    hotspots: Optional[List[Hotspot]] = Field(
        default=None,
        description="Identified hotspot regions",
    )

    regions_of_interest: Optional[List[RegionOfInterest]] = Field(
        default=None,
        description="User-defined regions of interest",
    )

    @field_validator("max_temp_c")
    @classmethod
    def validate_max_temp(cls, v: float, info) -> float:
        """Validate max temp is greater than min temp."""
        if "min_temp_c" in info.data:
            min_temp = info.data["min_temp_c"]
            if v < min_temp:
                raise ValueError(
                    f"Max temp ({v}) must be >= min temp ({min_temp})"
                )
        return v

    @computed_field
    @property
    def temp_range_c(self) -> float:
        """Calculate temperature range in image."""
        return self.max_temp_c - self.min_temp_c

    @computed_field
    @property
    def pixel_count(self) -> int:
        """Calculate total pixel count."""
        return self.resolution_width * self.resolution_height


class AmbientConditions(BaseModel):
    """
    Environmental conditions during inspection.

    Contains ambient temperature, wind, humidity, and solar loading data
    required for heat transfer calculations.

    Attributes:
        ambient_temp_c: Ambient air temperature (Celsius).
        wind_speed_m_s: Wind speed at inspection location (m/s).
        wind_direction_deg: Wind direction from North (degrees).
        relative_humidity_percent: Relative humidity (%).
        solar_loading_w_m2: Solar radiation intensity (W/m2).
        sky_condition: Sky/weather conditions.
        precipitation: Whether precipitation is occurring.
        measurement_timestamp: Ambient measurement timestamp.

    Example:
        >>> ambient = AmbientConditions(
        ...     ambient_temp_c=25.0,
        ...     wind_speed_m_s=2.0,
        ...     relative_humidity_percent=60.0,
        ...     measurement_timestamp=datetime.now()
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    ambient_temp_c: float = Field(
        ...,
        ge=-50.0,
        le=60.0,
        description="Ambient air temperature (Celsius)",
    )

    wind_speed_m_s: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Wind speed at inspection location (m/s)",
    )

    wind_direction_deg: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=360.0,
        description="Wind direction from North (degrees)",
    )

    relative_humidity_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative humidity (%)",
    )

    solar_loading_w_m2: float = Field(
        default=0.0,
        ge=0.0,
        le=1400.0,
        description="Solar radiation intensity (W/m2)",
    )

    sky_condition: WeatherCondition = Field(
        default=WeatherCondition.OVERCAST,
        description="Sky/weather conditions",
    )

    precipitation: bool = Field(
        default=False,
        description="Whether precipitation is occurring",
    )

    barometric_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=80.0,
        le=110.0,
        description="Barometric pressure (kPa)",
    )

    measurement_timestamp: datetime = Field(
        ...,
        description="Ambient measurement timestamp",
    )

    @computed_field
    @property
    def is_calm_conditions(self) -> bool:
        """Check if wind conditions are calm (<0.5 m/s)."""
        return self.wind_speed_m_s < 0.5

    @computed_field
    @property
    def is_suitable_for_ir(self) -> bool:
        """Check if conditions suitable for IR inspection."""
        # Not suitable during rain or high solar loading
        if self.precipitation:
            return False
        if self.solar_loading_w_m2 > 800:
            return False
        return True


class EquipmentParameters(BaseModel):
    """
    Equipment specifications for insulated asset.

    Contains all design specifications required for thermal performance
    calculations and heat loss quantification.

    Attributes:
        equipment_id: Unique equipment identifier.
        equipment_name: Equipment name/description.
        equipment_type: Type of equipment.
        process_temp_c: Internal process temperature (Celsius).
        design_temp_c: Design operating temperature (Celsius).
        process_fluid: Type of process fluid.
        pipe_diameter_mm: Pipe outer diameter (mm).
        pipe_length_m: Inspected pipe length (m).
        surface_area_m2: Total insulated surface area (m2).
        surface_orientation: Surface orientation.
        operating_hours_per_year: Annual operating hours.
        location: Physical location identifier.
        criticality: Equipment criticality level.

    Example:
        >>> params = EquipmentParameters(
        ...     equipment_id="PIPE-101",
        ...     equipment_type=EquipmentType.PIPE,
        ...     process_temp_c=200.0,
        ...     pipe_diameter_mm=150.0,
        ...     pipe_length_m=50.0
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    equipment_id: str = Field(
        ...,
        max_length=50,
        pattern=r"^[A-Za-z0-9\-_]+$",
        description="Unique equipment identifier",
    )

    equipment_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Equipment name/description",
    )

    equipment_type: EquipmentType = Field(
        ...,
        description="Type of equipment",
    )

    process_temp_c: float = Field(
        ...,
        ge=-200.0,
        le=800.0,
        description="Internal process temperature (Celsius)",
    )

    design_temp_c: Optional[float] = Field(
        default=None,
        ge=-200.0,
        le=800.0,
        description="Design operating temperature (Celsius)",
    )

    process_fluid: ProcessFluid = Field(
        default=ProcessFluid.STEAM,
        description="Type of process fluid",
    )

    # Pipe-specific parameters
    pipe_diameter_mm: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=3000.0,
        description="Pipe outer diameter (mm)",
    )

    pipe_length_m: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=1000.0,
        description="Inspected pipe length (m)",
    )

    # Vessel-specific parameters
    vessel_diameter_m: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=50.0,
        description="Vessel diameter (m)",
    )

    vessel_height_m: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=100.0,
        description="Vessel height (m)",
    )

    # General parameters
    surface_area_m2: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=10000.0,
        description="Total insulated surface area (m2)",
    )

    surface_orientation: SurfaceType = Field(
        default=SurfaceType.CYLINDRICAL_HORIZONTAL,
        description="Surface orientation",
    )

    operating_hours_per_year: int = Field(
        default=8760,
        ge=0,
        le=8760,
        description="Annual operating hours",
    )

    location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Physical location identifier",
    )

    unit: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process unit name",
    )

    elevation_m: Optional[float] = Field(
        default=None,
        description="Equipment elevation (m)",
    )

    criticality: str = Field(
        default="medium",
        pattern="^(critical|high|medium|low)$",
        description="Equipment criticality level",
    )

    @computed_field
    @property
    def calculated_surface_area_m2(self) -> Optional[float]:
        """Calculate surface area from dimensions if not provided."""
        if self.surface_area_m2 is not None:
            return self.surface_area_m2

        if self.equipment_type == EquipmentType.PIPE:
            if self.pipe_diameter_mm and self.pipe_length_m:
                diameter_m = self.pipe_diameter_mm / 1000.0
                return math.pi * diameter_m * self.pipe_length_m

        if self.equipment_type in [EquipmentType.VESSEL, EquipmentType.TANK]:
            if self.vessel_diameter_m and self.vessel_height_m:
                # Cylinder surface area (sides + ends)
                r = self.vessel_diameter_m / 2.0
                return (2 * math.pi * r * self.vessel_height_m +
                        2 * math.pi * r * r)

        return None

    @computed_field
    @property
    def is_cui_susceptible_temp(self) -> bool:
        """Check if operating in CUI-susceptible temperature range."""
        # CUI typically occurs between 0C and 175C
        return 0.0 <= self.process_temp_c <= 175.0


class InsulationSpecifications(BaseModel):
    """
    Insulation design specifications.

    Contains insulation material properties, dimensions, and installation
    information required for thermal calculations.

    Attributes:
        insulation_type: Insulation material type.
        insulation_thickness_mm: Design insulation thickness (mm).
        actual_thickness_mm: Measured/estimated actual thickness (mm).
        thermal_conductivity_w_mk: Insulation thermal conductivity (W/m-K).
        jacket_type: Weather barrier/jacketing material.
        jacket_condition: Visual jacket condition.
        vapor_barrier: Whether vapor barrier is present.
        installation_date: Insulation installation date.
        age_years: Current insulation age (years).
        last_inspection_date: Date of last inspection.

    Example:
        >>> specs = InsulationSpecifications(
        ...     insulation_type=InsulationType.MINERAL_WOOL,
        ...     insulation_thickness_mm=75.0,
        ...     jacket_type=JacketMaterial.ALUMINUM
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    insulation_type: InsulationType = Field(
        ...,
        description="Insulation material type",
    )

    insulation_material: Optional[InsulationMaterial] = Field(
        default=None,
        description="Specific insulation material/product",
    )

    insulation_thickness_mm: float = Field(
        ...,
        ge=10.0,
        le=500.0,
        description="Design insulation thickness (mm)",
    )

    actual_thickness_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=500.0,
        description="Measured/estimated actual thickness (mm)",
    )

    thermal_conductivity_w_mk: Optional[float] = Field(
        default=None,
        ge=0.010,
        le=0.200,
        description="Insulation thermal conductivity (W/m-K)",
    )

    density_kg_m3: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=500.0,
        description="Insulation density (kg/m3)",
    )

    jacket_type: JacketMaterial = Field(
        ...,
        description="Weather barrier/jacketing material",
    )

    jacket_thickness_mm: float = Field(
        default=0.8,
        ge=0.3,
        le=5.0,
        description="Jacket thickness (mm)",
    )

    jacket_condition: OverallCondition = Field(
        default=OverallCondition.GOOD,
        description="Visual jacket condition",
    )

    vapor_barrier: bool = Field(
        default=True,
        description="Whether vapor barrier is present",
    )

    installation_date: Optional[date] = Field(
        default=None,
        description="Insulation installation date",
    )

    expected_life_years: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Expected insulation service life (years)",
    )

    age_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Current insulation age (years)",
    )

    manufacturer: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Insulation manufacturer",
    )

    product_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Insulation product name",
    )

    last_inspection_date: Optional[date] = Field(
        default=None,
        description="Date of last inspection",
    )

    last_repair_date: Optional[date] = Field(
        default=None,
        description="Date of last repair",
    )

    @computed_field
    @property
    def calculated_age_years(self) -> Optional[float]:
        """Calculate age from installation date if not provided."""
        if self.age_years is not None:
            return self.age_years
        if self.installation_date:
            days = (date.today() - self.installation_date).days
            return days / 365.25
        return None

    @computed_field
    @property
    def remaining_life_percent(self) -> Optional[float]:
        """Calculate percentage of remaining useful life."""
        age = self.calculated_age_years
        if age is not None and self.expected_life_years > 0:
            remaining = max(0, 100.0 * (1.0 - age / self.expected_life_years))
            return round(remaining, 1)
        return None


class InspectionRecord(BaseModel):
    """
    Historical inspection record.

    Attributes:
        inspection_date: Date of inspection.
        inspector: Inspector name/ID.
        condition_rating: Condition rating assigned.
        heat_loss_w_m2: Measured heat loss.
        findings: Summary of findings.
        recommendations: Recommendations made.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    inspection_date: date = Field(..., description="Date of inspection")
    inspector: Optional[str] = Field(default=None, description="Inspector name/ID")
    condition_rating: Optional[OverallCondition] = Field(
        default=None, description="Condition rating"
    )
    heat_loss_w_m2: Optional[float] = Field(
        default=None, ge=0.0, description="Measured heat loss (W/m2)"
    )
    findings: Optional[str] = Field(default=None, description="Summary of findings")
    recommendations: Optional[str] = Field(default=None, description="Recommendations")


class RepairRecord(BaseModel):
    """
    Historical repair record.

    Attributes:
        repair_date: Date of repair.
        repair_type: Type of repair performed.
        area_repaired_m2: Area repaired.
        cost_usd: Repair cost.
        contractor: Contractor name.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    repair_date: date = Field(..., description="Date of repair")
    repair_type: RecommendedAction = Field(..., description="Type of repair")
    area_repaired_m2: Optional[float] = Field(
        default=None, ge=0.0, description="Area repaired (m2)"
    )
    cost_usd: Optional[float] = Field(
        default=None, ge=0.0, description="Repair cost (USD)"
    )
    contractor: Optional[str] = Field(default=None, description="Contractor name")


class HistoricalData(BaseModel):
    """
    Historical inspection and performance data.

    Contains historical information used for trend analysis
    and performance comparison.

    Attributes:
        previous_inspections: Array of previous inspection records.
        repair_history: History of repairs and replacements.
        energy_baseline: Baseline energy performance data.
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    previous_inspections: Optional[List[InspectionRecord]] = Field(
        default=None,
        description="Array of previous inspection records",
    )

    repair_history: Optional[List[RepairRecord]] = Field(
        default=None,
        description="History of repairs and replacements",
    )

    baseline_heat_loss_w_m2: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Baseline heat loss for comparison (W/m2)",
    )

    baseline_date: Optional[date] = Field(
        default=None,
        description="Date of baseline measurement",
    )


class InsulationInspectionInput(BaseModel):
    """
    Complete input data model for InsulationInspectionAgent.

    Aggregates all input data required for comprehensive insulation
    inspection including thermal images, ambient conditions, equipment
    parameters, and specifications.

    Attributes:
        thermal_image_data: IR camera thermal image data.
        ambient_conditions: Environmental conditions during inspection.
        equipment_parameters: Equipment specifications.
        insulation_specifications: Insulation design specifications.
        historical_data: Historical inspection data (optional).
        analysis_timestamp: Timestamp for this analysis request.

    Example:
        >>> input_data = InsulationInspectionInput(
        ...     thermal_image_data=image_data,
        ...     ambient_conditions=ambient,
        ...     equipment_parameters=params,
        ...     insulation_specifications=specs
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    thermal_image_data: ThermalImageData = Field(
        ...,
        description="IR camera thermal image data",
    )

    ambient_conditions: AmbientConditions = Field(
        ...,
        description="Environmental conditions during inspection",
    )

    equipment_parameters: EquipmentParameters = Field(
        ...,
        description="Equipment specifications",
    )

    insulation_specifications: InsulationSpecifications = Field(
        ...,
        description="Insulation design specifications",
    )

    historical_data: Optional[HistoricalData] = Field(
        default=None,
        description="Historical inspection data",
    )

    analysis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp for this analysis request",
    )

    def calculate_input_hash(self) -> str:
        """
        Calculate SHA-256 hash of input data for provenance tracking.

        Returns:
            str: SHA-256 hash of serialized input data.
        """
        # Serialize input data to JSON (exclude timestamp and large matrix)
        input_json = self.model_dump_json(
            exclude={"analysis_timestamp", "thermal_image_data": {"temperature_matrix"}}
        )
        return hashlib.sha256(input_json.encode()).hexdigest()


# ==============================================================================
# OUTPUT MODELS
# ==============================================================================


class HeatLossAnalysis(BaseModel):
    """
    Heat loss analysis output.

    Contains quantified heat losses with energy and cost impact.

    Attributes:
        equipment_id: Equipment identifier.
        analysis_timestamp: Timestamp of analysis.
        surface_temp_c: Measured surface temperature.
        total_heat_loss_w: Total heat loss rate (W).
        heat_loss_per_area_w_m2: Heat flux per unit area (W/m2).
        baseline_heat_loss_w: Expected heat loss with proper insulation.
        excess_heat_loss_w: Excess heat loss above baseline.
        annual_energy_loss_mwh: Annual energy loss (MWh).
        annual_energy_cost_usd: Annual energy cost (USD).
        thermal_efficiency_percent: Current thermal efficiency.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    equipment_id: str = Field(..., description="Equipment identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    # Temperature measurements
    surface_temp_c: float = Field(
        ..., description="Measured surface temperature (Celsius)"
    )
    surface_temp_elevation_c: float = Field(
        ..., ge=0.0, description="Surface temp above ambient (Celsius)"
    )

    # Heat loss measurements
    total_heat_loss_w: float = Field(
        ..., ge=0.0, description="Total heat loss rate (W)"
    )
    heat_loss_per_area_w_m2: float = Field(
        ..., ge=0.0, description="Heat flux per unit area (W/m2)"
    )
    radiative_heat_loss_w_m2: float = Field(
        ..., ge=0.0, description="Radiative heat flux (W/m2)"
    )
    convective_heat_loss_w_m2: float = Field(
        ..., ge=0.0, description="Convective heat flux (W/m2)"
    )

    # Baseline comparison
    baseline_heat_loss_w: float = Field(
        ..., ge=0.0, description="Expected heat loss with proper insulation (W)"
    )
    baseline_heat_loss_w_m2: float = Field(
        ..., ge=0.0, description="Expected heat flux (W/m2)"
    )
    excess_heat_loss_w: float = Field(
        ..., ge=0.0, description="Excess heat loss above baseline (W)"
    )
    excess_heat_loss_percent: float = Field(
        ..., ge=0.0, description="Percentage excess heat loss"
    )

    # Energy impact
    hourly_energy_loss_kwh: float = Field(
        ..., ge=0.0, description="Hourly energy loss (kWh)"
    )
    daily_energy_loss_kwh: float = Field(
        ..., ge=0.0, description="Daily energy loss (kWh)"
    )
    annual_energy_loss_mwh: float = Field(
        ..., ge=0.0, description="Annual energy loss (MWh)"
    )
    annual_excess_energy_mwh: float = Field(
        ..., ge=0.0, description="Annual excess energy loss (MWh)"
    )

    # Cost impact
    hourly_energy_cost_usd: float = Field(
        ..., ge=0.0, description="Hourly energy cost (USD)"
    )
    daily_energy_cost_usd: float = Field(
        ..., ge=0.0, description="Daily energy cost (USD)"
    )
    annual_energy_cost_usd: float = Field(
        ..., ge=0.0, description="Annual energy cost (USD)"
    )
    annual_excess_cost_usd: float = Field(
        ..., ge=0.0, description="Annual excess cost vs baseline (USD)"
    )

    # Carbon impact
    annual_co2_emissions_kg: float = Field(
        ..., ge=0.0, description="Annual CO2 emissions (kg)"
    )
    annual_excess_co2_kg: float = Field(
        ..., ge=0.0, description="Annual excess CO2 (kg)"
    )

    # Efficiency metrics
    thermal_efficiency_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Current thermal efficiency (%)"
    )
    design_efficiency_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Design efficiency (%)"
    )
    efficiency_loss_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Efficiency degradation (%)"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class DamageDetail(BaseModel):
    """
    Detail of identified damage.

    Attributes:
        damage_type: Type of damage.
        severity: Damage severity.
        location: Damage location.
        area_affected_m2: Affected area.
        confidence: Detection confidence.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    damage_type: DamageType = Field(..., description="Type of damage")
    severity: DegradationSeverity = Field(..., description="Damage severity")
    location: Optional[str] = Field(default=None, description="Damage location")
    area_affected_m2: Optional[float] = Field(
        default=None, ge=0.0, description="Affected area (m2)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )


class CUIRiskFactor(BaseModel):
    """
    CUI risk factor assessment.

    Attributes:
        factor: Risk factor name.
        weight: Factor weight in risk score.
        assessment: Assessment description.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    factor: str = Field(..., description="Risk factor name")
    weight: float = Field(ge=0.0, le=1.0, description="Factor weight")
    assessment: str = Field(..., description="Assessment description")
    contributes_risk: bool = Field(default=True, description="Whether factor adds risk")


class DegradationAssessment(BaseModel):
    """
    Insulation condition and degradation assessment output.

    Contains comprehensive degradation analysis including condition
    classification, damage identification, and CUI risk.

    Attributes:
        equipment_id: Equipment identifier.
        analysis_timestamp: Timestamp of analysis.
        overall_condition: Overall insulation condition.
        degradation_severity: Degradation severity classification.
        condition_score: Numeric condition score (0-100).
        damage_types: Identified damage types.
        moisture_detected: Whether moisture is detected.
        cui_risk_level: Corrosion Under Insulation risk.
        remaining_life_years: Estimated remaining useful life.
        confidence_score: Assessment confidence.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    equipment_id: str = Field(..., description="Equipment identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    # Overall condition
    overall_condition: OverallCondition = Field(
        ..., description="Overall insulation condition"
    )
    degradation_severity: DegradationSeverity = Field(
        ..., description="Degradation severity classification"
    )
    condition_score: float = Field(
        ..., ge=0.0, le=100.0, description="Numeric condition score (0-100)"
    )

    # Damage assessment
    damage_types: Optional[List[DamageDetail]] = Field(
        default=None, description="Identified damage types"
    )
    primary_damage_type: Optional[DamageType] = Field(
        default=None, description="Primary damage mechanism"
    )
    primary_failure_mode: Optional[FailureMode] = Field(
        default=None, description="Primary failure mode"
    )

    # Moisture assessment
    moisture_detected: bool = Field(
        default=False, description="Whether moisture is detected"
    )
    moisture_severity: MoistureState = Field(
        default=MoistureState.DRY, description="Moisture severity level"
    )
    moisture_indicators: Optional[List[str]] = Field(
        default=None, description="Indicators of moisture presence"
    )

    # CUI risk assessment
    cui_risk_level: CorrosionRisk = Field(
        ..., description="Corrosion Under Insulation risk"
    )
    cui_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="CUI risk score (0-100)"
    )
    cui_risk_factors: Optional[List[CUIRiskFactor]] = Field(
        default=None, description="Contributing CUI risk factors"
    )

    # Life assessment
    estimated_age_years: Optional[float] = Field(
        default=None, ge=0.0, description="Estimated insulation age"
    )
    remaining_life_years: Optional[float] = Field(
        default=None, ge=0.0, description="Estimated remaining useful life"
    )
    life_consumed_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Percentage of life consumed"
    )

    # Confidence
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Assessment confidence (0-1)"
    )
    confidence_factors: Optional[List[str]] = Field(
        default=None, description="Factors affecting confidence"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class RepairScope(BaseModel):
    """
    Scope of repair work.

    Attributes:
        area_m2: Area to be repaired.
        linear_m: Linear meters (for pipes).
        sections: Number of sections.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    area_m2: Optional[float] = Field(default=None, ge=0.0, description="Area (m2)")
    linear_m: Optional[float] = Field(
        default=None, ge=0.0, description="Linear meters"
    )
    sections: Optional[int] = Field(default=None, ge=1, description="Number of sections")


class AlternativeAction(BaseModel):
    """
    Alternative repair action option.

    Attributes:
        action: Action description.
        cost_usd: Estimated cost.
        savings_usd: Annual savings.
        payback_months: Payback period.
        pros: Advantages.
        cons: Disadvantages.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    action: RecommendedAction = Field(..., description="Action type")
    description: str = Field(..., description="Action description")
    cost_usd: float = Field(ge=0.0, description="Estimated cost (USD)")
    savings_usd: float = Field(ge=0.0, description="Annual savings (USD)")
    payback_months: float = Field(ge=0.0, description="Payback period (months)")
    pros: List[str] = Field(default_factory=list, description="Advantages")
    cons: List[str] = Field(default_factory=list, description="Disadvantages")


class RepairRecommendation(BaseModel):
    """
    Single repair recommendation with economic analysis.

    Attributes:
        equipment_id: Equipment identifier.
        priority_ranking: Priority rank (1 = highest).
        recommended_action: Recommended repair action.
        estimated_repair_cost_usd: Total estimated repair cost.
        estimated_savings_annual_usd: Annual energy savings.
        payback_period_months: Simple payback period.
        roi_percent: Return on investment.
        urgency_level: Repair urgency.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    equipment_id: str = Field(..., description="Equipment identifier")
    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")

    # Priority
    priority_ranking: int = Field(..., ge=1, description="Priority rank (1 = highest)")
    priority_score: float = Field(
        ..., ge=0.0, le=100.0, description="Priority score (0-100)"
    )
    urgency_level: RepairPriority = Field(..., description="Repair urgency")

    # Recommended action
    recommended_action: RecommendedAction = Field(
        ..., description="Recommended repair action"
    )
    action_description: str = Field(..., description="Detailed action description")
    repair_scope: Optional[RepairScope] = Field(
        default=None, description="Scope of repair work"
    )

    # Cost estimates
    material_cost_usd: float = Field(ge=0.0, description="Estimated material cost")
    labor_cost_usd: float = Field(ge=0.0, description="Estimated labor cost")
    scaffold_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Scaffolding/access cost"
    )
    equipment_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Equipment rental cost"
    )
    total_repair_cost_usd: float = Field(
        ..., ge=0.0, description="Total estimated repair cost"
    )
    cost_confidence: str = Field(
        default="estimate", description="Cost estimate confidence level"
    )

    # Economic analysis
    annual_savings_usd: float = Field(
        ..., ge=0.0, description="Annual energy savings after repair"
    )
    payback_period_months: float = Field(
        ..., ge=0.0, description="Simple payback period"
    )
    roi_percent: float = Field(..., description="Return on investment (%)")
    npv_10yr_usd: Optional[float] = Field(
        default=None, description="10-year Net Present Value"
    )
    irr_percent: Optional[float] = Field(
        default=None, description="Internal Rate of Return (%)"
    )

    # Schedule
    recommended_timing: str = Field(..., description="Recommended repair timing")
    estimated_duration_days: float = Field(
        ..., ge=0.0, description="Estimated repair duration"
    )
    shutdown_required: bool = Field(
        default=False, description="Whether process shutdown is required"
    )

    # Alternatives
    alternative_actions: Optional[List[AlternativeAction]] = Field(
        default=None, description="Alternative repair options"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class RepairPriorities(BaseModel):
    """
    Prioritized repair recommendations output.

    Contains all recommendations sorted by priority.

    Attributes:
        analysis_timestamp: Timestamp of analysis.
        total_items: Total number of items assessed.
        items_requiring_action: Items needing repair.
        recommendations: Prioritized repair recommendations.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")
    total_items: int = Field(..., ge=0, description="Total items assessed")
    items_requiring_action: int = Field(..., ge=0, description="Items needing repair")

    recommendations: List[RepairRecommendation] = Field(
        ..., description="Prioritized repair recommendations"
    )

    # Summary statistics
    total_repair_cost_usd: float = Field(
        ..., ge=0.0, description="Total repair cost for all items"
    )
    total_annual_savings_usd: float = Field(
        ..., ge=0.0, description="Total annual savings if all repaired"
    )
    portfolio_payback_months: float = Field(
        ..., ge=0.0, description="Portfolio payback period"
    )
    portfolio_roi_percent: float = Field(..., description="Portfolio ROI (%)")

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class Finding(BaseModel):
    """
    Inspection finding.

    Attributes:
        finding_id: Unique finding identifier.
        location: Finding location.
        finding_type: Type of finding.
        severity: Finding severity.
        description: Detailed description.
        evidence: Supporting evidence.
        image_reference: Reference to supporting image.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    finding_id: str = Field(..., description="Unique finding identifier")
    location: str = Field(..., description="Finding location")
    finding_type: str = Field(..., description="Type of finding")
    severity: DegradationSeverity = Field(..., description="Finding severity")
    description: str = Field(..., description="Detailed description")
    evidence: Optional[str] = Field(default=None, description="Supporting evidence")
    image_reference: Optional[str] = Field(
        default=None, description="Reference to image"
    )


class Recommendation(BaseModel):
    """
    Inspection recommendation.

    Attributes:
        recommendation_id: Unique recommendation identifier.
        priority: Recommendation priority.
        action: Recommended action.
        description: Detailed description.
        estimated_cost_usd: Estimated cost.
        estimated_savings_usd: Estimated savings.
        target_date: Target completion date.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    priority: RepairPriority = Field(..., description="Priority level")
    action: RecommendedAction = Field(..., description="Recommended action")
    description: str = Field(..., description="Detailed description")
    estimated_cost_usd: Optional[float] = Field(
        default=None, ge=0.0, description="Estimated cost"
    )
    estimated_savings_usd: Optional[float] = Field(
        default=None, ge=0.0, description="Estimated savings"
    )
    target_date: Optional[date] = Field(
        default=None, description="Target completion date"
    )


class ImageReference(BaseModel):
    """
    Reference to thermal image.

    Attributes:
        image_id: Image identifier.
        image_type: Type of image.
        caption: Image caption.
        location: Location in report.
        url: Image URL/path.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    image_id: str = Field(..., description="Image identifier")
    image_type: str = Field(..., description="Type of image")
    caption: Optional[str] = Field(default=None, description="Image caption")
    location: Optional[str] = Field(default=None, description="Location in report")
    url: Optional[str] = Field(default=None, description="Image URL/path")


class ComplianceStatus(BaseModel):
    """
    Compliance assessment status.

    Attributes:
        astm_c1055_compliant: ASTM C1055 personnel protection compliance.
        personnel_protection_ok: Surface temp within safe limits.
        energy_code_compliant: Energy code compliance.
        violations: List of violations found.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    astm_c1055_compliant: bool = Field(
        ..., description="ASTM C1055 compliance (surface temp)"
    )
    personnel_protection_ok: bool = Field(
        ..., description="Surface temp within safe limits"
    )
    max_surface_temp_c: float = Field(..., description="Maximum surface temperature")
    safe_touch_temp_c: float = Field(
        default=60.0, description="Safe touch temperature threshold"
    )
    energy_code_compliant: Optional[bool] = Field(
        default=None, description="Energy code compliance"
    )
    violations: List[str] = Field(
        default_factory=list, description="List of violations"
    )


class InspectionReport(BaseModel):
    """
    Comprehensive inspection report output.

    Contains complete inspection findings, assessments, and recommendations.

    Attributes:
        report_id: Unique report identifier.
        inspection_date: Inspection date.
        equipment_summary: Equipment summary information.
        findings: Inspection findings.
        recommendations: Action recommendations.
        compliance_status: Compliance assessment.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    report_id: str = Field(..., description="Unique report identifier")
    inspection_date: date = Field(..., description="Inspection date")
    inspection_timestamp: datetime = Field(..., description="Inspection timestamp")
    inspector: Optional[str] = Field(default=None, description="Inspector name/ID")

    # Equipment summary
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    location: Optional[str] = Field(default=None, description="Location")
    unit: Optional[str] = Field(default=None, description="Process unit")

    # Inspection conditions
    ambient_temp_c: float = Field(..., description="Ambient temperature")
    wind_speed_m_s: float = Field(..., description="Wind speed")
    sky_condition: WeatherCondition = Field(..., description="Sky condition")
    camera_used: CameraType = Field(..., description="Camera used")

    # Insulation details
    insulation_type: InsulationType = Field(..., description="Insulation type")
    insulation_thickness_mm: float = Field(..., description="Insulation thickness")
    process_temp_c: float = Field(..., description="Process temperature")
    surface_area_m2: Optional[float] = Field(default=None, description="Surface area")

    # Findings
    findings: List[Finding] = Field(..., description="Inspection findings")
    finding_count: int = Field(..., ge=0, description="Number of findings")

    # Overall assessment
    overall_condition: OverallCondition = Field(..., description="Overall condition")
    heat_loss_w_m2: float = Field(..., ge=0.0, description="Measured heat loss")
    annual_energy_cost_usd: float = Field(..., ge=0.0, description="Annual energy cost")
    cui_risk_level: CorrosionRisk = Field(..., description="CUI risk level")
    remaining_life_years: Optional[float] = Field(
        default=None, description="Remaining life estimate"
    )

    # Recommendations
    recommendations: List[Recommendation] = Field(
        ..., description="Action recommendations"
    )
    recommendation_count: int = Field(..., ge=0, description="Number of recommendations")

    # Images
    images: Optional[List[ImageReference]] = Field(
        default=None, description="Associated thermal images"
    )

    # Compliance
    compliance_status: ComplianceStatus = Field(..., description="Compliance status")

    # Approval
    approval_status: str = Field(
        default="draft", description="Report approval status"
    )
    approved_by: Optional[str] = Field(default=None, description="Approver name")
    approval_date: Optional[date] = Field(default=None, description="Approval date")

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class PriorityBreakdown(BaseModel):
    """
    Breakdown by priority level.

    Attributes:
        count: Number of items.
        total_cost_usd: Total cost.
        total_savings_usd: Total savings.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    count: int = Field(ge=0, description="Number of items")
    total_cost_usd: float = Field(ge=0.0, description="Total cost (USD)")
    total_savings_usd: float = Field(ge=0.0, description="Total savings (USD)")


class EconomicImpact(BaseModel):
    """
    Economic impact analysis output.

    Contains comprehensive economic analysis including energy waste costs,
    repair ROI, and portfolio-level metrics.

    Attributes:
        analysis_timestamp: Timestamp of analysis.
        currency: Currency for monetary values.
        total_annual_waste_usd: Total annual energy waste.
        total_repair_cost_usd: Total repair investment needed.
        total_annual_savings_usd: Total annual savings if repaired.
        portfolio_roi_percent: Portfolio ROI.
        carbon_emissions_avoided_kg: CO2 reduction potential.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    analysis_timestamp: datetime = Field(..., description="Timestamp of analysis")
    currency: str = Field(default="USD", description="Currency for monetary values")

    # Current state costs
    current_annual_energy_cost_usd: float = Field(
        ..., ge=0.0, description="Current annual energy cost"
    )
    baseline_annual_energy_cost_usd: float = Field(
        ..., ge=0.0, description="Baseline annual energy cost"
    )
    annual_waste_usd: float = Field(
        ..., ge=0.0, description="Annual energy waste cost"
    )

    # Portfolio summary
    total_equipment_count: int = Field(..., ge=0, description="Total equipment inspected")
    equipment_requiring_action: int = Field(..., ge=0, description="Equipment needing repair")
    total_area_inspected_m2: float = Field(
        ..., ge=0.0, description="Total area inspected"
    )
    total_degraded_area_m2: float = Field(
        ..., ge=0.0, description="Total degraded area"
    )

    # Cost summary
    total_annual_waste_usd: float = Field(
        ..., ge=0.0, description="Total annual energy waste"
    )
    total_repair_cost_usd: float = Field(
        ..., ge=0.0, description="Total repair investment needed"
    )
    total_annual_savings_usd: float = Field(
        ..., ge=0.0, description="Total annual savings if repaired"
    )

    # ROI analysis
    portfolio_simple_payback_years: float = Field(
        ..., ge=0.0, description="Portfolio simple payback"
    )
    portfolio_roi_percent: float = Field(..., description="Portfolio ROI (%)")
    portfolio_npv_10yr_usd: Optional[float] = Field(
        default=None, description="10-year portfolio NPV"
    )
    portfolio_irr_percent: Optional[float] = Field(
        default=None, description="Portfolio IRR (%)"
    )

    # Carbon impact
    current_annual_co2_tons: float = Field(
        ..., ge=0.0, description="Current annual CO2 emissions"
    )
    baseline_annual_co2_tons: float = Field(
        ..., ge=0.0, description="Baseline annual CO2 emissions"
    )
    annual_co2_reduction_potential_tons: float = Field(
        ..., ge=0.0, description="Potential annual CO2 reduction"
    )
    carbon_cost_savings_usd: Optional[float] = Field(
        default=None, ge=0.0, description="Carbon cost savings"
    )

    # Priority breakdown
    critical_repairs: Optional[PriorityBreakdown] = Field(
        default=None, description="Critical priority repairs"
    )
    urgent_repairs: Optional[PriorityBreakdown] = Field(
        default=None, description="Urgent priority repairs"
    )
    planned_repairs: Optional[PriorityBreakdown] = Field(
        default=None, description="Planned priority repairs"
    )
    routine_repairs: Optional[PriorityBreakdown] = Field(
        default=None, description="Routine priority repairs"
    )

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class InsulationInspectionOutput(BaseModel):
    """
    Complete output data model for InsulationInspectionAgent.

    Aggregates all output data from comprehensive insulation inspection.

    Attributes:
        heat_loss_analysis: Quantified heat losses.
        degradation_assessment: Insulation condition assessment.
        repair_priorities: Prioritized repair recommendations.
        inspection_report: Comprehensive inspection report.
        economic_impact: Economic impact analysis.
        processing_time_ms: Processing duration in milliseconds.
        validation_status: PASS or FAIL.
        master_provenance_hash: Combined SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    heat_loss_analysis: HeatLossAnalysis = Field(
        ..., description="Quantified heat losses"
    )
    degradation_assessment: DegradationAssessment = Field(
        ..., description="Insulation condition assessment"
    )
    repair_priorities: RepairRecommendation = Field(
        ..., description="Repair recommendation"
    )
    inspection_report: InspectionReport = Field(
        ..., description="Comprehensive inspection report"
    )
    economic_impact: EconomicImpact = Field(
        ..., description="Economic impact analysis"
    )

    processing_time_ms: float = Field(
        ..., ge=0.0, description="Processing duration (milliseconds)"
    )
    validation_status: str = Field(
        ..., pattern="^(PASS|FAIL)$", description="Validation status"
    )
    master_provenance_hash: str = Field(
        ..., description="Combined SHA-256 hash for audit trail"
    )


# ==============================================================================
# VALIDATION MODELS
# ==============================================================================


class ValidationError(BaseModel):
    """
    Single validation error.

    Attributes:
        field: Field name with error.
        error: Error description.
        value: Invalid value (if applicable).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    field: str = Field(..., description="Field name with error")
    error: str = Field(..., description="Error description")
    value: Optional[Any] = Field(default=None, description="Invalid value")


class ValidationResult(BaseModel):
    """
    Validation result for input or output data.

    Attributes:
        is_valid: Whether validation passed.
        errors: List of validation errors.
        warnings: List of validation warnings.
        validated_at: Timestamp of validation.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[ValidationError] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    validated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of validation"
    )


# ==============================================================================
# CONFIGURATION CLASSES
# ==============================================================================


class HeatFluxThresholds(BaseModel):
    """
    Heat flux thresholds for severity classification.

    Attributes:
        minor: Minor heat loss threshold (W/m2).
        moderate: Moderate heat loss threshold (W/m2).
        severe: Severe heat loss threshold (W/m2).
        critical: Critical heat loss threshold (W/m2).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    minor: float = Field(default=50.0, ge=0.0, description="Minor threshold (W/m2)")
    moderate: float = Field(default=150.0, ge=0.0, description="Moderate threshold")
    severe: float = Field(default=300.0, ge=0.0, description="Severe threshold")
    critical: float = Field(default=500.0, ge=0.0, description="Critical threshold")


class TempDeltaThresholds(BaseModel):
    """
    Temperature delta thresholds for classification.

    Attributes:
        minor: Minor elevation threshold (K).
        moderate: Moderate elevation threshold (K).
        severe: Severe elevation threshold (K).
        critical: Critical elevation threshold (K).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    minor: float = Field(default=10.0, ge=0.0, description="Minor threshold (K)")
    moderate: float = Field(default=25.0, ge=0.0, description="Moderate threshold")
    severe: float = Field(default=50.0, ge=0.0, description="Severe threshold")
    critical: float = Field(default=100.0, ge=0.0, description="Critical threshold")


class CUIRiskThresholds(BaseModel):
    """
    CUI risk temperature thresholds.

    Attributes:
        risk_min_c: Lower bound for CUI risk zone.
        risk_max_c: Upper bound for CUI risk zone.
        optimal_min_c: Start of high-risk zone.
        optimal_max_c: End of high-risk zone.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    risk_min_c: float = Field(default=0.0, description="CUI risk lower bound (C)")
    risk_max_c: float = Field(default=175.0, description="CUI risk upper bound (C)")
    optimal_min_c: float = Field(default=50.0, description="High-risk zone start (C)")
    optimal_max_c: float = Field(default=150.0, description="High-risk zone end (C)")


class EfficiencyThresholds(BaseModel):
    """
    Insulation efficiency thresholds.

    Attributes:
        excellent: Excellent efficiency threshold (%).
        good: Good efficiency threshold (%).
        fair: Fair efficiency threshold (%).
        poor: Poor efficiency threshold (%).
        critical: Critical efficiency threshold (%).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    excellent: float = Field(default=95.0, ge=0.0, le=100.0)
    good: float = Field(default=85.0, ge=0.0, le=100.0)
    fair: float = Field(default=70.0, ge=0.0, le=100.0)
    poor: float = Field(default=50.0, ge=0.0, le=100.0)
    critical: float = Field(default=30.0, ge=0.0, le=100.0)


class ROIThresholds(BaseModel):
    """
    ROI thresholds for recommendations.

    Attributes:
        min_roi_for_repair: Minimum ROI to recommend repair (%).
        max_payback_months: Maximum acceptable payback (months).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    min_roi_for_repair: float = Field(default=25.0, ge=0.0)
    max_payback_months: int = Field(default=36, ge=1)


class EconomicParameters(BaseModel):
    """
    Economic parameters for cost calculations.

    Attributes:
        natural_gas_cost_per_mmbtu: Natural gas cost (USD/MMBtu).
        electricity_cost_per_kwh: Electricity cost (USD/kWh).
        steam_cost_per_klb: Steam cost (USD/klb).
        labor_cost_per_hour: Labor rate (USD/hour).
        carbon_cost_per_ton: Carbon cost (USD/ton CO2).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    natural_gas_cost_per_mmbtu: float = Field(
        default=4.50, ge=0.0, description="Natural gas cost (USD/MMBtu)"
    )
    electricity_cost_per_kwh: float = Field(
        default=0.10, ge=0.0, description="Electricity cost (USD/kWh)"
    )
    steam_cost_per_klb: float = Field(
        default=12.00, ge=0.0, description="Steam cost (USD/klb)"
    )
    thermal_oil_cost_per_kwh: float = Field(
        default=0.08, ge=0.0, description="Thermal oil cost (USD/kWh)"
    )
    labor_cost_per_hour: float = Field(
        default=85.00, ge=0.0, description="Labor rate (USD/hour)"
    )
    scaffold_cost_per_day: float = Field(
        default=500.00, ge=0.0, description="Scaffold cost (USD/day)"
    )
    carbon_cost_per_ton: float = Field(
        default=50.00, ge=0.0, description="Carbon cost (USD/ton CO2)"
    )


class RepairCostEstimates(BaseModel):
    """
    Default repair cost estimates.

    Attributes:
        spot_repair_per_m2: Spot repair cost (USD/m2).
        partial_replacement_per_m2: Partial replacement cost (USD/m2).
        full_replacement_per_m2: Full replacement cost (USD/m2).
        cui_remediation_per_m2: CUI remediation cost (USD/m2).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    spot_repair_per_m2: float = Field(default=150.0, ge=0.0)
    partial_replacement_per_m2: float = Field(default=350.0, ge=0.0)
    full_replacement_per_m2: float = Field(default=550.0, ge=0.0)
    cui_remediation_per_m2: float = Field(default=800.0, ge=0.0)
    scaffold_setup: float = Field(default=2500.0, ge=0.0)
    equipment_rental_per_day: float = Field(default=350.0, ge=0.0)


class InsulationPropertySet(BaseModel):
    """
    Thermal properties for a single insulation type.

    Attributes:
        k_value: Thermal conductivity (W/m-K).
        max_temp_c: Maximum service temperature (C).
        density_kg_m3: Density (kg/m3).
        moisture_resistance: Moisture resistance rating.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    k_value: float = Field(ge=0.010, le=0.200, description="k-value (W/m-K)")
    max_temp_c: float = Field(ge=50.0, le=1200.0, description="Max temp (C)")
    density_kg_m3: float = Field(ge=10.0, le=500.0, description="Density (kg/m3)")
    moisture_resistance: str = Field(description="Moisture resistance rating")


class EmissivityValues(BaseModel):
    """
    Surface emissivity reference values.

    Attributes:
        aluminum_weathered: Weathered aluminum emissivity.
        aluminum_oxidized: Oxidized aluminum emissivity.
        stainless_steel_weathered: Weathered stainless emissivity.
        pvc_jacket: PVC jacket emissivity.
        painted_surface: Painted surface emissivity.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    aluminum_weathered: float = Field(default=0.20, ge=0.05, le=1.0)
    aluminum_oxidized: float = Field(default=0.30, ge=0.05, le=1.0)
    stainless_steel_weathered: float = Field(default=0.60, ge=0.05, le=1.0)
    galvanized_steel_weathered: float = Field(default=0.30, ge=0.05, le=1.0)
    galvanized_steel_oxidized: float = Field(default=0.50, ge=0.05, le=1.0)
    pvc_jacket: float = Field(default=0.90, ge=0.05, le=1.0)
    painted_surface: float = Field(default=0.90, ge=0.05, le=1.0)
    bare_steel_oxidized: float = Field(default=0.80, ge=0.05, le=1.0)
    bare_steel_polished: float = Field(default=0.15, ge=0.05, le=1.0)


class SafetyLimits(BaseModel):
    """
    Safety limits for operation.

    Attributes:
        max_safe_surface_temp_c: ASTM C1055 safe touch temperature.
        max_process_temp_c: Maximum process temperature.
        min_ambient_temp_c: Minimum ambient temperature.
        max_wind_speed_m_s: Maximum wind speed for inspection.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    max_safe_surface_temp_c: float = Field(
        default=60.0, ge=40.0, le=100.0, description="Safe touch temp (C)"
    )
    max_process_temp_c: float = Field(default=800.0, ge=100.0, le=1200.0)
    min_ambient_temp_c: float = Field(default=-50.0, ge=-100.0, le=0.0)
    max_ambient_temp_c: float = Field(default=60.0, ge=30.0, le=80.0)
    max_wind_speed_m_s: float = Field(default=20.0, ge=5.0, le=50.0)
    min_emissivity: float = Field(default=0.10, ge=0.01, le=0.50)
    max_emissivity: float = Field(default=0.98, ge=0.90, le=1.0)


class InsulationConfig(BaseModel):
    """
    Comprehensive insulation inspection configuration.

    Contains all configurable parameters for the Insulation Inspection
    agent including thresholds, economic parameters, and material properties.

    Attributes:
        heat_flux_thresholds: Heat flux classification thresholds.
        temp_delta_thresholds: Temperature delta thresholds.
        cui_risk_thresholds: CUI risk temperature thresholds.
        efficiency_thresholds: Efficiency classification thresholds.
        roi_thresholds: ROI thresholds for recommendations.
        economic_parameters: Economic parameters.
        repair_cost_estimates: Default repair costs.
        emissivity_values: Reference emissivity values.
        safety_limits: Safety limits.
        cache_ttl_seconds: Cache time-to-live.
        max_batch_size: Maximum batch size.
        max_retries: Maximum retry attempts.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    heat_flux_thresholds: HeatFluxThresholds = Field(
        default_factory=HeatFluxThresholds,
        description="Heat flux classification thresholds",
    )

    temp_delta_thresholds: TempDeltaThresholds = Field(
        default_factory=TempDeltaThresholds,
        description="Temperature delta thresholds",
    )

    cui_risk_thresholds: CUIRiskThresholds = Field(
        default_factory=CUIRiskThresholds,
        description="CUI risk temperature thresholds",
    )

    efficiency_thresholds: EfficiencyThresholds = Field(
        default_factory=EfficiencyThresholds,
        description="Efficiency classification thresholds",
    )

    roi_thresholds: ROIThresholds = Field(
        default_factory=ROIThresholds,
        description="ROI thresholds for recommendations",
    )

    economic_parameters: EconomicParameters = Field(
        default_factory=EconomicParameters,
        description="Economic parameters for cost calculations",
    )

    repair_cost_estimates: RepairCostEstimates = Field(
        default_factory=RepairCostEstimates,
        description="Default repair cost estimates",
    )

    emissivity_values: EmissivityValues = Field(
        default_factory=EmissivityValues,
        description="Reference emissivity values",
    )

    safety_limits: SafetyLimits = Field(
        default_factory=SafetyLimits,
        description="Safety limits for operation",
    )

    cache_ttl_seconds: int = Field(
        default=600,
        ge=0,
        description="Cache time-to-live in seconds",
    )

    max_batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum batch processing size",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts",
    )


# ==============================================================================
# SETTINGS (Environment-based Configuration)
# ==============================================================================


class Settings(BaseSettings):
    """
    Application configuration settings for GL-015 INSULSCAN.

    Loads settings from environment variables and .env file, with
    validation and sensible defaults for all parameters.

    Attributes:
        GREENLANG_ENV: Environment (development, staging, production).
        APP_NAME: Application name.
        APP_VERSION: Application version.
        LOG_LEVEL: Logging level.
        DEBUG: Debug mode flag.
        DATABASE_URL: PostgreSQL connection string.
        REDIS_URL: Redis connection string.

    Example:
        >>> settings = Settings()
        >>> print(settings.APP_NAME)
        'GL-015-InsulationInspection'
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Application Configuration
    # ==========================================================================
    GREENLANG_ENV: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )
    APP_NAME: str = Field(
        default="GL-015-InsulationInspection",
        description="Application name",
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode",
    )

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/greenlang",
        description="PostgreSQL connection string",
    )
    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size",
    )
    DB_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum overflow connections",
    )

    # ==========================================================================
    # Cache Configuration
    # ==========================================================================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )
    REDIS_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Redis connection pool size",
    )
    CACHE_TTL: int = Field(
        default=600,
        ge=0,
        description="Cache TTL in seconds",
    )

    # ==========================================================================
    # Monitoring & Observability
    # ==========================================================================
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    PROMETHEUS_PORT: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="Prometheus metrics port",
    )
    TRACING_ENABLED: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(
        default="http://localhost:4317",
        description="OTLP exporter endpoint",
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format: json or text",
    )

    # ==========================================================================
    # Economic Parameters
    # ==========================================================================
    NATURAL_GAS_COST_PER_MMBTU: float = Field(
        default=4.50,
        ge=0.0,
        description="Natural gas cost (USD/MMBtu)",
    )
    ELECTRICITY_COST_PER_KWH: float = Field(
        default=0.10,
        ge=0.0,
        description="Electricity cost (USD/kWh)",
    )
    STEAM_COST_PER_KLB: float = Field(
        default=12.00,
        ge=0.0,
        description="Steam cost (USD/klb)",
    )
    LABOR_COST_PER_HOUR: float = Field(
        default=85.00,
        ge=0.0,
        description="Labor rate (USD/hour)",
    )
    CARBON_COST_PER_TON: float = Field(
        default=50.00,
        ge=0.0,
        description="Carbon cost (USD/ton CO2)",
    )

    # ==========================================================================
    # Heat Loss Thresholds
    # ==========================================================================
    HEAT_FLUX_THRESHOLD_MINOR: float = Field(
        default=50.0,
        ge=0.0,
        description="Minor heat loss threshold (W/m2)",
    )
    HEAT_FLUX_THRESHOLD_MODERATE: float = Field(
        default=150.0,
        ge=0.0,
        description="Moderate heat loss threshold (W/m2)",
    )
    HEAT_FLUX_THRESHOLD_SEVERE: float = Field(
        default=300.0,
        ge=0.0,
        description="Severe heat loss threshold (W/m2)",
    )
    HEAT_FLUX_THRESHOLD_CRITICAL: float = Field(
        default=500.0,
        ge=0.0,
        description="Critical heat loss threshold (W/m2)",
    )

    # ==========================================================================
    # Safety Limits
    # ==========================================================================
    MAX_SAFE_SURFACE_TEMP_C: float = Field(
        default=60.0,
        ge=40.0,
        le=100.0,
        description="ASTM C1055 safe touch temperature (C)",
    )

    # ==========================================================================
    # Analysis Configuration
    # ==========================================================================
    MIN_ROI_FOR_REPAIR: float = Field(
        default=25.0,
        ge=0.0,
        description="Minimum ROI to recommend repair (%)",
    )
    MAX_PAYBACK_MONTHS: int = Field(
        default=36,
        ge=1,
        description="Maximum acceptable payback period (months)",
    )

    # ==========================================================================
    # Batch Processing
    # ==========================================================================
    BATCH_PROCESSING_ENABLED: bool = Field(
        default=True,
        description="Enable batch processing",
    )
    MAX_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum batch size",
    )
    PARALLEL_WORKERS: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of parallel workers",
    )

    # ==========================================================================
    # Performance & Scaling
    # ==========================================================================
    WORKER_COUNT: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker processes",
    )
    TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        description="Request timeout in seconds",
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=100,
        ge=1,
        description="API rate limit per minute",
    )

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    ENABLE_CUI_DETECTION: bool = Field(
        default=True,
        description="Enable CUI risk assessment",
    )
    ENABLE_CMMS_INTEGRATION: bool = Field(
        default=False,
        description="Enable CMMS integration",
    )
    ENABLE_WEATHER_SERVICE: bool = Field(
        default=False,
        description="Enable weather service integration",
    )

    @field_validator("GREENLANG_ENV")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format setting."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of {valid_formats}")
        return v.lower()

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.GREENLANG_ENV == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.GREENLANG_ENV == "development"

    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.GREENLANG_ENV == "staging"

    def get_config(self) -> InsulationConfig:
        """
        Create InsulationConfig from settings.

        Returns:
            InsulationConfig: Configuration object populated from settings.
        """
        return InsulationConfig(
            heat_flux_thresholds=HeatFluxThresholds(
                minor=self.HEAT_FLUX_THRESHOLD_MINOR,
                moderate=self.HEAT_FLUX_THRESHOLD_MODERATE,
                severe=self.HEAT_FLUX_THRESHOLD_SEVERE,
                critical=self.HEAT_FLUX_THRESHOLD_CRITICAL,
            ),
            roi_thresholds=ROIThresholds(
                min_roi_for_repair=self.MIN_ROI_FOR_REPAIR,
                max_payback_months=self.MAX_PAYBACK_MONTHS,
            ),
            economic_parameters=EconomicParameters(
                natural_gas_cost_per_mmbtu=self.NATURAL_GAS_COST_PER_MMBTU,
                electricity_cost_per_kwh=self.ELECTRICITY_COST_PER_KWH,
                steam_cost_per_klb=self.STEAM_COST_PER_KLB,
                labor_cost_per_hour=self.LABOR_COST_PER_HOUR,
                carbon_cost_per_ton=self.CARBON_COST_PER_TON,
            ),
            safety_limits=SafetyLimits(
                max_safe_surface_temp_c=self.MAX_SAFE_SURFACE_TEMP_C,
            ),
            cache_ttl_seconds=self.CACHE_TTL,
            max_batch_size=self.MAX_BATCH_SIZE,
            max_retries=3,
        )


# ==============================================================================
# INSULATION MATERIAL PROPERTY TABLES
# ==============================================================================

# Default thermal conductivity values for common insulation types (W/m-K at mean temp)
INSULATION_K_VALUES: Dict[InsulationType, float] = {
    InsulationType.MINERAL_WOOL: 0.040,
    InsulationType.CALCIUM_SILICATE: 0.055,
    InsulationType.CELLULAR_GLASS: 0.050,
    InsulationType.PERLITE: 0.052,
    InsulationType.AEROGEL: 0.015,
    InsulationType.POLYURETHANE: 0.025,
    InsulationType.POLYSTYRENE: 0.035,
    InsulationType.PHENOLIC_FOAM: 0.022,
    InsulationType.FIBERGLASS: 0.038,
    InsulationType.MICROPOROUS: 0.020,
    InsulationType.CERAMIC_FIBER: 0.080,
    InsulationType.UNKNOWN: 0.045,
}

# Maximum service temperatures for insulation types (Celsius)
INSULATION_MAX_TEMPS: Dict[InsulationType, float] = {
    InsulationType.MINERAL_WOOL: 650.0,
    InsulationType.CALCIUM_SILICATE: 650.0,
    InsulationType.CELLULAR_GLASS: 430.0,
    InsulationType.PERLITE: 650.0,
    InsulationType.AEROGEL: 650.0,
    InsulationType.POLYURETHANE: 120.0,
    InsulationType.POLYSTYRENE: 75.0,
    InsulationType.PHENOLIC_FOAM: 120.0,
    InsulationType.FIBERGLASS: 450.0,
    InsulationType.MICROPOROUS: 1000.0,
    InsulationType.CERAMIC_FIBER: 1260.0,
    InsulationType.UNKNOWN: 200.0,
}

# Default emissivity values for jacket materials
JACKET_EMISSIVITY: Dict[JacketMaterial, float] = {
    JacketMaterial.ALUMINUM: 0.20,
    JacketMaterial.STAINLESS_STEEL: 0.60,
    JacketMaterial.GALVANIZED_STEEL: 0.30,
    JacketMaterial.PVC: 0.90,
    JacketMaterial.NONE: 0.90,
    JacketMaterial.PAINTED_METAL: 0.90,
    JacketMaterial.FIBERGLASS_REINFORCED: 0.85,
    JacketMaterial.COMPOSITE: 0.85,
}

# Stefan-Boltzmann constant (W/m2-K4)
STEFAN_BOLTZMANN_CONSTANT: float = 5.67e-8

# CO2 emission factors (kg CO2 per unit energy)
CO2_EMISSION_FACTORS: Dict[str, float] = {
    "natural_gas_per_mmbtu": 53.07,  # kg CO2/MMBtu
    "electricity_per_kwh": 0.42,      # kg CO2/kWh (US average)
    "steam_per_klb": 74.0,            # kg CO2/klb (typical boiler)
}


# Global settings instance
settings = Settings()
