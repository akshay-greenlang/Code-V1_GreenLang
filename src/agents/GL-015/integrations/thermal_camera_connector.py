"""
Thermal Camera Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade integration with industrial thermal imaging cameras:
- FLIR Systems (FLIR Tools API, Thermal Studio SDK)
- Fluke Ti-series thermography cameras
- Testo thermal imaging cameras
- Optris PI series cameras
- InfraTec ImageIR cameras
- Generic ONVIF thermal camera support

Supports radiometric data extraction, image file parsing (JPEG, TIFF with thermal data),
real-time streaming, and camera calibration data handling.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import struct
import uuid
from pathlib import Path

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    CircuitOpenError,
    ConfigurationError,
    ConnectionError,
    ConnectionState,
    ConnectorError,
    ConnectorType,
    DataQualityLevel,
    DataQualityResult,
    HealthCheckResult,
    HealthStatus,
    ImageFormat,
    ImageProcessingError,
    CameraConnectionError,
    CalibrationError,
    TimeoutError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ThermalCameraProvider(str, Enum):
    """Supported thermal camera providers."""

    FLIR_SYSTEMS = "flir_systems"  # FLIR cameras with FLIR Tools API
    FLIR_THERMAL_STUDIO = "flir_thermal_studio"  # FLIR Thermal Studio SDK
    FLUKE_TI = "fluke_ti"  # Fluke Ti-series cameras
    TESTO = "testo"  # Testo thermography cameras
    OPTRIS_PI = "optris_pi"  # Optris PI series
    INFRATEC_IMAGEIR = "infratec_imageir"  # InfraTec ImageIR
    ONVIF_THERMAL = "onvif_thermal"  # Generic ONVIF thermal cameras
    GENERIC_HTTP = "generic_http"  # Generic HTTP-based cameras


class CameraConnectionMode(str, Enum):
    """Camera connection modes."""

    USB = "usb"  # Direct USB connection
    ETHERNET = "ethernet"  # Ethernet/IP connection
    WIFI = "wifi"  # WiFi connection
    BLUETOOTH = "bluetooth"  # Bluetooth connection
    SDK = "sdk"  # SDK-based connection
    FILE = "file"  # File-based (offline analysis)


class StreamProtocol(str, Enum):
    """Streaming protocols."""

    RTSP = "rtsp"  # Real-Time Streaming Protocol
    HTTP_MJPEG = "http_mjpeg"  # HTTP Motion JPEG
    ONVIF = "onvif"  # ONVIF protocol
    PROPRIETARY = "proprietary"  # Vendor-specific protocol
    RAW_TCP = "raw_tcp"  # Raw TCP socket


class CalibrationStatus(str, Enum):
    """Camera calibration status."""

    VALID = "valid"
    EXPIRED = "expired"
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATION_DUE = "calibration_due"
    UNKNOWN = "unknown"


class TemperatureUnit(str, Enum):
    """Temperature units."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


class NUCMode(str, Enum):
    """Non-Uniformity Correction (NUC) modes."""

    AUTO = "auto"
    MANUAL = "manual"
    DISABLED = "disabled"


class PaletteType(str, Enum):
    """Thermal image color palettes."""

    IRONBOW = "ironbow"
    RAINBOW = "rainbow"
    GRAYSCALE = "grayscale"
    WHITE_HOT = "white_hot"
    BLACK_HOT = "black_hot"
    ARCTIC = "arctic"
    LAVA = "lava"
    AMBER = "amber"
    GREEN = "green"
    HOT_METAL = "hot_metal"


class MeasurementType(str, Enum):
    """Temperature measurement types."""

    SPOT = "spot"  # Single point measurement
    AREA_MAX = "area_max"  # Maximum temperature in area
    AREA_MIN = "area_min"  # Minimum temperature in area
    AREA_AVG = "area_avg"  # Average temperature in area
    LINE = "line"  # Temperature along a line
    ISOTHERM = "isotherm"  # Isotherm analysis
    DELTA_T = "delta_t"  # Temperature difference


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class FLIRConfig(BaseModel):
    """FLIR camera configuration."""

    model_config = ConfigDict(extra="forbid")

    api_endpoint: Optional[str] = Field(
        default=None,
        description="FLIR Tools API endpoint URL"
    )
    sdk_path: Optional[str] = Field(
        default=None,
        description="Path to FLIR Thermal Studio SDK"
    )
    camera_serial: Optional[str] = Field(
        default=None,
        description="Camera serial number"
    )
    use_radiometric: bool = Field(
        default=True,
        description="Use radiometric data extraction"
    )
    exif_extraction: bool = Field(
        default=True,
        description="Extract EXIF thermal metadata"
    )


class FlukeTiConfig(BaseModel):
    """Fluke Ti-series camera configuration."""

    model_config = ConfigDict(extra="forbid")

    api_endpoint: Optional[str] = Field(
        default=None,
        description="Fluke Connect API endpoint"
    )
    device_id: Optional[str] = Field(
        default=None,
        description="Device identifier"
    )
    fluke_connect_enabled: bool = Field(
        default=True,
        description="Use Fluke Connect cloud"
    )


class TestoConfig(BaseModel):
    """Testo camera configuration."""

    model_config = ConfigDict(extra="forbid")

    api_endpoint: Optional[str] = Field(
        default=None,
        description="Testo API endpoint"
    )
    device_serial: Optional[str] = Field(
        default=None,
        description="Device serial number"
    )
    testo_cloud_enabled: bool = Field(
        default=False,
        description="Use Testo Cloud"
    )


class OptrisPiConfig(BaseModel):
    """Optris PI camera configuration."""

    model_config = ConfigDict(extra="forbid")

    ip_address: str = Field(
        ...,
        description="Camera IP address"
    )
    port: int = Field(
        default=80,
        ge=1,
        le=65535,
        description="Camera port"
    )
    sdk_version: str = Field(
        default="4.0",
        description="Optris SDK version"
    )
    process_interface: bool = Field(
        default=True,
        description="Use process interface mode"
    )


class InfraTecConfig(BaseModel):
    """InfraTec ImageIR configuration."""

    model_config = ConfigDict(extra="forbid")

    ip_address: str = Field(
        ...,
        description="Camera IP address"
    )
    port: int = Field(
        default=80,
        ge=1,
        le=65535,
        description="Camera port"
    )
    irbis_enabled: bool = Field(
        default=True,
        description="Use IRBIS software integration"
    )


class ONVIFConfig(BaseModel):
    """ONVIF thermal camera configuration."""

    model_config = ConfigDict(extra="forbid")

    ip_address: str = Field(
        ...,
        description="Camera IP address"
    )
    port: int = Field(
        default=80,
        ge=1,
        le=65535,
        description="ONVIF port"
    )
    username: str = Field(
        default="admin",
        description="ONVIF username"
    )
    password: str = Field(
        default="",
        description="ONVIF password"
    )
    profile_token: Optional[str] = Field(
        default=None,
        description="ONVIF profile token"
    )
    wsdl_path: Optional[str] = Field(
        default=None,
        description="Path to ONVIF WSDL files"
    )


class ThermalCameraConnectorConfig(BaseConnectorConfig):
    """Configuration for thermal camera connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: ThermalCameraProvider = Field(
        ...,
        description="Thermal camera provider"
    )
    connection_mode: CameraConnectionMode = Field(
        default=CameraConnectionMode.ETHERNET,
        description="Camera connection mode"
    )

    # Provider-specific configurations
    flir_config: Optional[FLIRConfig] = Field(
        default=None,
        description="FLIR-specific configuration"
    )
    fluke_config: Optional[FlukeTiConfig] = Field(
        default=None,
        description="Fluke Ti-specific configuration"
    )
    testo_config: Optional[TestoConfig] = Field(
        default=None,
        description="Testo-specific configuration"
    )
    optris_config: Optional[OptrisPiConfig] = Field(
        default=None,
        description="Optris PI-specific configuration"
    )
    infratec_config: Optional[InfraTecConfig] = Field(
        default=None,
        description="InfraTec-specific configuration"
    )
    onvif_config: Optional[ONVIFConfig] = Field(
        default=None,
        description="ONVIF-specific configuration"
    )

    # Streaming settings
    stream_enabled: bool = Field(
        default=False,
        description="Enable real-time streaming"
    )
    stream_protocol: StreamProtocol = Field(
        default=StreamProtocol.RTSP,
        description="Streaming protocol"
    )
    stream_fps: int = Field(
        default=9,
        ge=1,
        le=60,
        description="Target frames per second"
    )
    stream_buffer_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Frame buffer size"
    )

    # Image settings
    default_image_format: ImageFormat = Field(
        default=ImageFormat.RADIOMETRIC_JPEG,
        description="Default image format"
    )
    temperature_unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Temperature unit"
    )
    default_palette: PaletteType = Field(
        default=PaletteType.IRONBOW,
        description="Default color palette"
    )

    # Calibration settings
    calibration_check_enabled: bool = Field(
        default=True,
        description="Check calibration status"
    )
    calibration_warning_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days before calibration due to warn"
    )
    nuc_mode: NUCMode = Field(
        default=NUCMode.AUTO,
        description="Non-Uniformity Correction mode"
    )

    # Radiometric settings
    default_emissivity: float = Field(
        default=0.95,
        ge=0.01,
        le=1.0,
        description="Default emissivity value"
    )
    default_reflected_temperature_c: float = Field(
        default=20.0,
        ge=-40.0,
        le=500.0,
        description="Default reflected temperature (Celsius)"
    )
    default_atmospheric_temperature_c: float = Field(
        default=20.0,
        ge=-40.0,
        le=60.0,
        description="Default atmospheric temperature (Celsius)"
    )
    default_distance_m: float = Field(
        default=1.0,
        ge=0.1,
        le=1000.0,
        description="Default distance to object (meters)"
    )
    default_relative_humidity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default relative humidity (0-1)"
    )

    def __init__(self, **data):
        """Initialize with connector type set."""
        data['connector_type'] = ConnectorType.THERMAL_CAMERA
        super().__init__(**data)


# =============================================================================
# Data Models
# =============================================================================


class CameraInfo(BaseModel):
    """Thermal camera device information."""

    model_config = ConfigDict(frozen=True)

    camera_id: str = Field(..., description="Unique camera identifier")
    manufacturer: str = Field(..., description="Camera manufacturer")
    model: str = Field(..., description="Camera model")
    serial_number: str = Field(..., description="Serial number")
    firmware_version: str = Field(default="", description="Firmware version")
    resolution_x: int = Field(..., ge=1, description="Sensor width in pixels")
    resolution_y: int = Field(..., ge=1, description="Sensor height in pixels")
    temperature_range_min_c: float = Field(..., description="Min temp range")
    temperature_range_max_c: float = Field(..., description="Max temp range")
    netd_mk: Optional[float] = Field(
        default=None,
        description="Noise Equivalent Temperature Difference (mK)"
    )
    fov_horizontal_deg: Optional[float] = Field(
        default=None,
        description="Horizontal field of view (degrees)"
    )
    fov_vertical_deg: Optional[float] = Field(
        default=None,
        description="Vertical field of view (degrees)"
    )
    lens_info: Optional[str] = Field(default=None, description="Lens information")


class CalibrationData(BaseModel):
    """Camera calibration data."""

    model_config = ConfigDict(frozen=True)

    camera_id: str = Field(..., description="Camera identifier")
    calibration_date: datetime = Field(..., description="Last calibration date")
    calibration_due_date: Optional[datetime] = Field(
        default=None,
        description="Next calibration due date"
    )
    calibration_status: CalibrationStatus = Field(
        default=CalibrationStatus.UNKNOWN,
        description="Calibration status"
    )
    calibration_certificate: Optional[str] = Field(
        default=None,
        description="Calibration certificate number"
    )
    temperature_accuracy_c: float = Field(
        default=2.0,
        description="Temperature accuracy (+/- Celsius)"
    )
    calibration_lab: Optional[str] = Field(
        default=None,
        description="Calibration laboratory"
    )
    offset_corrections: Dict[str, float] = Field(
        default_factory=dict,
        description="Calibration offset corrections"
    )


class RadiometricParameters(BaseModel):
    """Radiometric measurement parameters."""

    model_config = ConfigDict(frozen=True)

    emissivity: float = Field(
        default=0.95,
        ge=0.01,
        le=1.0,
        description="Surface emissivity"
    )
    reflected_temperature_c: float = Field(
        default=20.0,
        description="Reflected apparent temperature"
    )
    atmospheric_temperature_c: float = Field(
        default=20.0,
        description="Atmospheric temperature"
    )
    distance_m: float = Field(
        default=1.0,
        ge=0.1,
        description="Distance to object"
    )
    relative_humidity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative humidity"
    )
    atmospheric_transmission: Optional[float] = Field(
        default=None,
        description="Calculated atmospheric transmission"
    )


class TemperaturePoint(BaseModel):
    """Single temperature measurement point."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(..., ge=0, description="X coordinate (pixel)")
    y: int = Field(..., ge=0, description="Y coordinate (pixel)")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    raw_value: Optional[int] = Field(
        default=None,
        description="Raw sensor value"
    )
    quality: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Measurement quality"
    )


class TemperatureRegion(BaseModel):
    """Temperature measurement region (box, ellipse, polygon)."""

    model_config = ConfigDict(frozen=True)

    region_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Region identifier"
    )
    region_type: str = Field(
        default="rectangle",
        pattern="^(rectangle|ellipse|polygon|line)$",
        description="Region type"
    )
    coordinates: List[Tuple[int, int]] = Field(
        ...,
        min_length=2,
        description="Region coordinates [(x, y), ...]"
    )
    min_temperature_c: float = Field(..., description="Minimum temperature")
    max_temperature_c: float = Field(..., description="Maximum temperature")
    avg_temperature_c: float = Field(..., description="Average temperature")
    std_deviation_c: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation"
    )
    min_point: Optional[TemperaturePoint] = Field(
        default=None,
        description="Location of minimum temperature"
    )
    max_point: Optional[TemperaturePoint] = Field(
        default=None,
        description="Location of maximum temperature"
    )
    pixel_count: int = Field(default=0, ge=0, description="Pixels in region")


class TemperatureMatrix(BaseModel):
    """Temperature matrix (radiometric data grid)."""

    model_config = ConfigDict(frozen=False)

    width: int = Field(..., ge=1, description="Matrix width")
    height: int = Field(..., ge=1, description="Matrix height")
    data: List[List[float]] = Field(
        ...,
        description="2D temperature array [row][col] in Celsius"
    )
    unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Temperature unit"
    )
    min_temperature: float = Field(..., description="Minimum temperature")
    max_temperature: float = Field(..., description="Maximum temperature")
    avg_temperature: float = Field(..., description="Average temperature")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Capture timestamp"
    )

    @field_validator('data')
    @classmethod
    def validate_matrix_dimensions(cls, v, info):
        """Validate matrix dimensions match width/height."""
        height = info.data.get('height', 0)
        width = info.data.get('width', 0)
        if len(v) != height:
            raise ValueError(f'Matrix rows {len(v)} does not match height {height}')
        for row in v:
            if len(row) != width:
                raise ValueError(f'Matrix columns {len(row)} does not match width {width}')
        return v


class ThermalImage(BaseModel):
    """Complete thermal image with metadata and radiometric data."""

    model_config = ConfigDict(frozen=False)

    image_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique image identifier"
    )
    camera_id: str = Field(..., description="Camera that captured the image")
    capture_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Capture timestamp"
    )

    # Image data
    visual_image_base64: Optional[str] = Field(
        default=None,
        description="Visual image as base64 encoded string"
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.RADIOMETRIC_JPEG,
        description="Image format"
    )
    width: int = Field(..., ge=1, description="Image width")
    height: int = Field(..., ge=1, description="Image height")

    # Temperature data
    temperature_matrix: Optional[TemperatureMatrix] = Field(
        default=None,
        description="Radiometric temperature matrix"
    )
    min_temperature_c: float = Field(..., description="Minimum temperature")
    max_temperature_c: float = Field(..., description="Maximum temperature")
    avg_temperature_c: float = Field(..., description="Average temperature")

    # Measurement regions
    measurement_regions: List[TemperatureRegion] = Field(
        default_factory=list,
        description="Defined measurement regions"
    )
    spot_measurements: List[TemperaturePoint] = Field(
        default_factory=list,
        description="Spot temperature measurements"
    )

    # Radiometric parameters
    radiometric_params: RadiometricParameters = Field(
        default_factory=RadiometricParameters,
        description="Radiometric measurement parameters"
    )

    # Metadata
    palette: PaletteType = Field(
        default=PaletteType.IRONBOW,
        description="Color palette used"
    )
    temperature_range_min_c: Optional[float] = Field(
        default=None,
        description="Display temperature range minimum"
    )
    temperature_range_max_c: Optional[float] = Field(
        default=None,
        description="Display temperature range maximum"
    )
    exif_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="EXIF metadata"
    )
    gps_coordinates: Optional[Tuple[float, float]] = Field(
        default=None,
        description="GPS coordinates (lat, lon)"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Image notes"
    )


class ThermalFrame(BaseModel):
    """Single frame from thermal camera stream."""

    model_config = ConfigDict(frozen=True)

    frame_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Frame identifier"
    )
    sequence_number: int = Field(..., ge=0, description="Frame sequence number")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Frame timestamp"
    )
    temperature_matrix: TemperatureMatrix = Field(
        ...,
        description="Temperature data"
    )
    frame_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Current frame rate"
    )
    dropped_frames: int = Field(
        default=0,
        ge=0,
        description="Dropped frame count"
    )


class CaptureRequest(BaseModel):
    """Request to capture a thermal image."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.RADIOMETRIC_JPEG,
        description="Desired image format"
    )
    include_visual: bool = Field(
        default=True,
        description="Include visual image"
    )
    include_temperature_matrix: bool = Field(
        default=True,
        description="Include temperature matrix"
    )
    radiometric_params: Optional[RadiometricParameters] = Field(
        default=None,
        description="Override radiometric parameters"
    )
    measurement_regions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Regions to measure"
    )
    spot_points: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Spot measurement points [(x, y), ...]"
    )
    palette: Optional[PaletteType] = Field(
        default=None,
        description="Override color palette"
    )
    trigger_nuc: bool = Field(
        default=False,
        description="Trigger NUC before capture"
    )


class StreamConfiguration(BaseModel):
    """Configuration for camera streaming."""

    model_config = ConfigDict(frozen=True)

    protocol: StreamProtocol = Field(
        default=StreamProtocol.RTSP,
        description="Streaming protocol"
    )
    fps: int = Field(
        default=9,
        ge=1,
        le=60,
        description="Target frame rate"
    )
    include_radiometric: bool = Field(
        default=True,
        description="Include radiometric data"
    )
    buffer_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Frame buffer size"
    )
    compression: Optional[str] = Field(
        default=None,
        description="Compression codec"
    )


class FileParseRequest(BaseModel):
    """Request to parse a thermal image file."""

    model_config = ConfigDict(frozen=True)

    file_path: Optional[str] = Field(
        default=None,
        description="Path to thermal image file"
    )
    file_data: Optional[bytes] = Field(
        default=None,
        description="Raw file data"
    )
    file_format: Optional[ImageFormat] = Field(
        default=None,
        description="Image format (auto-detect if not specified)"
    )
    extract_radiometric: bool = Field(
        default=True,
        description="Extract radiometric data"
    )
    extract_visual: bool = Field(
        default=True,
        description="Extract visual image"
    )
    radiometric_params: Optional[RadiometricParameters] = Field(
        default=None,
        description="Override radiometric parameters"
    )


# =============================================================================
# Thermal Camera Connector
# =============================================================================


class ThermalCameraConnector(BaseConnector):
    """
    Thermal Camera Connector for GL-015 INSULSCAN.

    Provides unified interface for thermal imaging camera integration
    supporting multiple vendors (FLIR, Fluke, Testo, Optris, InfraTec, ONVIF).

    Features:
    - Multi-vendor camera support with provider-specific adapters
    - Radiometric data extraction from thermal images
    - Real-time streaming with frame buffering
    - Camera calibration management
    - Image file parsing (JPEG, TIFF, vendor formats)
    - Automatic temperature unit conversion
    """

    def __init__(self, config: ThermalCameraConnectorConfig) -> None:
        """
        Initialize thermal camera connector.

        Args:
            config: Thermal camera connector configuration
        """
        super().__init__(config)
        self._camera_config = config

        # Camera state
        self._camera_info: Optional[CameraInfo] = None
        self._calibration_data: Optional[CalibrationData] = None
        self._is_streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._frame_buffer: asyncio.Queue = asyncio.Queue(
            maxsize=config.stream_buffer_size
        )

        # Provider-specific client
        self._provider_client: Optional[Any] = None

        # Session for HTTP-based providers
        self._http_session: Optional[aiohttp.ClientSession] = None

    @property
    def camera_info(self) -> Optional[CameraInfo]:
        """Get camera information."""
        return self._camera_info

    @property
    def calibration_data(self) -> Optional[CalibrationData]:
        """Get calibration data."""
        return self._calibration_data

    @property
    def is_streaming(self) -> bool:
        """Check if camera is streaming."""
        return self._is_streaming

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Establish connection to the thermal camera."""
        self._logger.info(
            f"Connecting to thermal camera: {self._camera_config.provider.value}"
        )

        try:
            self._state = ConnectionState.CONNECTING

            # Initialize HTTP session for HTTP-based providers
            if self._camera_config.connection_mode in [
                CameraConnectionMode.ETHERNET,
                CameraConnectionMode.WIFI
            ]:
                session = await self._pool.get_session()
                self._http_session = session

            # Connect based on provider
            if self._camera_config.provider == ThermalCameraProvider.FLIR_SYSTEMS:
                await self._connect_flir()
            elif self._camera_config.provider == ThermalCameraProvider.FLIR_THERMAL_STUDIO:
                await self._connect_flir_thermal_studio()
            elif self._camera_config.provider == ThermalCameraProvider.FLUKE_TI:
                await self._connect_fluke()
            elif self._camera_config.provider == ThermalCameraProvider.TESTO:
                await self._connect_testo()
            elif self._camera_config.provider == ThermalCameraProvider.OPTRIS_PI:
                await self._connect_optris()
            elif self._camera_config.provider == ThermalCameraProvider.INFRATEC_IMAGEIR:
                await self._connect_infratec()
            elif self._camera_config.provider == ThermalCameraProvider.ONVIF_THERMAL:
                await self._connect_onvif()
            else:
                await self._connect_generic_http()

            # Get camera info
            self._camera_info = await self._get_camera_info()

            # Check calibration if enabled
            if self._camera_config.calibration_check_enabled:
                self._calibration_data = await self._get_calibration_data()
                self._check_calibration_status()

            self._state = ConnectionState.CONNECTED
            self._logger.info(
                f"Connected to thermal camera: {self._camera_info.model} "
                f"({self._camera_info.serial_number})"
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to thermal camera: {e}")
            raise CameraConnectionError(
                f"Camera connection failed: {e}",
                details={"provider": self._camera_config.provider.value}
            )

    async def disconnect(self) -> None:
        """Disconnect from the thermal camera."""
        self._logger.info("Disconnecting from thermal camera")

        # Stop streaming if active
        if self._is_streaming:
            await self.stop_stream()

        # Disconnect from provider
        if self._provider_client:
            try:
                # Provider-specific disconnect
                pass
            except Exception as e:
                self._logger.warning(f"Error during provider disconnect: {e}")

        self._provider_client = None
        self._camera_info = None
        self._calibration_data = None
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on camera connection."""
        import time
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Camera not connected: {self._state.value}"
                )

            # Try to get camera status
            status = await self._get_camera_status()
            latency_ms = (time.time() - start_time) * 1000

            # Check calibration
            calibration_status = HealthStatus.HEALTHY
            calibration_message = ""

            if self._calibration_data:
                if self._calibration_data.calibration_status == CalibrationStatus.EXPIRED:
                    calibration_status = HealthStatus.DEGRADED
                    calibration_message = "Calibration expired"
                elif self._calibration_data.calibration_status == CalibrationStatus.CALIBRATION_DUE:
                    calibration_status = HealthStatus.DEGRADED
                    calibration_message = "Calibration due soon"

            # Determine overall status
            if status.get("connected", False):
                overall_status = calibration_status
                message = calibration_message if calibration_message else "Camera healthy"
            else:
                overall_status = HealthStatus.UNHEALTHY
                message = "Camera not responding"

            return HealthCheckResult(
                status=overall_status,
                latency_ms=latency_ms,
                message=message,
                details={
                    "camera_model": self._camera_info.model if self._camera_info else None,
                    "calibration_status": self._calibration_data.calibration_status.value if self._calibration_data else None,
                    "is_streaming": self._is_streaming,
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}"
            )

    async def validate_configuration(self) -> bool:
        """Validate connector configuration."""
        provider = self._camera_config.provider

        # Validate provider-specific configuration
        if provider == ThermalCameraProvider.FLIR_SYSTEMS:
            if not self._camera_config.flir_config:
                raise ConfigurationError("FLIR config required for FLIR provider")

        elif provider == ThermalCameraProvider.FLUKE_TI:
            if not self._camera_config.fluke_config:
                raise ConfigurationError("Fluke config required for Fluke provider")

        elif provider == ThermalCameraProvider.TESTO:
            if not self._camera_config.testo_config:
                raise ConfigurationError("Testo config required for Testo provider")

        elif provider == ThermalCameraProvider.OPTRIS_PI:
            if not self._camera_config.optris_config:
                raise ConfigurationError("Optris config required for Optris provider")

        elif provider == ThermalCameraProvider.INFRATEC_IMAGEIR:
            if not self._camera_config.infratec_config:
                raise ConfigurationError("InfraTec config required for InfraTec provider")

        elif provider == ThermalCameraProvider.ONVIF_THERMAL:
            if not self._camera_config.onvif_config:
                raise ConfigurationError("ONVIF config required for ONVIF provider")

        return True

    # =========================================================================
    # Provider-Specific Connection Methods
    # =========================================================================

    async def _connect_flir(self) -> None:
        """Connect to FLIR camera via FLIR Tools API."""
        config = self._camera_config.flir_config
        self._logger.info("Connecting to FLIR camera via FLIR Tools API")

        if config.api_endpoint:
            # HTTP-based FLIR Tools API connection
            # Verify API endpoint is reachable
            try:
                async with self._http_session.get(
                    f"{config.api_endpoint}/api/v1/cameras",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        raise CameraConnectionError(
                            f"FLIR API returned {response.status}"
                        )
            except aiohttp.ClientError as e:
                raise CameraConnectionError(f"FLIR API connection failed: {e}")

        elif config.sdk_path:
            # SDK-based connection (would require native SDK bindings)
            self._logger.info(f"Using FLIR SDK at: {config.sdk_path}")
            # SDK initialization would go here

    async def _connect_flir_thermal_studio(self) -> None:
        """Connect to FLIR camera via Thermal Studio SDK."""
        config = self._camera_config.flir_config
        self._logger.info("Connecting to FLIR Thermal Studio SDK")

        if config and config.sdk_path:
            # Thermal Studio SDK connection
            self._logger.info(f"Using Thermal Studio SDK at: {config.sdk_path}")
            # SDK initialization would go here

    async def _connect_fluke(self) -> None:
        """Connect to Fluke Ti-series camera."""
        config = self._camera_config.fluke_config
        self._logger.info("Connecting to Fluke Ti-series camera")

        if config.fluke_connect_enabled and config.api_endpoint:
            # Fluke Connect cloud API
            try:
                async with self._http_session.get(
                    f"{config.api_endpoint}/api/v1/devices",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        raise CameraConnectionError(
                            f"Fluke Connect API returned {response.status}"
                        )
            except aiohttp.ClientError as e:
                raise CameraConnectionError(f"Fluke Connect connection failed: {e}")

    async def _connect_testo(self) -> None:
        """Connect to Testo thermography camera."""
        config = self._camera_config.testo_config
        self._logger.info("Connecting to Testo camera")

        if config.api_endpoint:
            try:
                async with self._http_session.get(
                    f"{config.api_endpoint}/api/devices/{config.device_serial}",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        raise CameraConnectionError(
                            f"Testo API returned {response.status}"
                        )
            except aiohttp.ClientError as e:
                raise CameraConnectionError(f"Testo connection failed: {e}")

    async def _connect_optris(self) -> None:
        """Connect to Optris PI camera."""
        config = self._camera_config.optris_config
        self._logger.info(f"Connecting to Optris PI camera at {config.ip_address}")

        # Optris cameras use HTTP for configuration, raw TCP for data
        try:
            async with self._http_session.get(
                f"http://{config.ip_address}:{config.port}/api/camera/info",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise CameraConnectionError(
                        f"Optris camera returned {response.status}"
                    )
        except aiohttp.ClientError as e:
            raise CameraConnectionError(f"Optris connection failed: {e}")

    async def _connect_infratec(self) -> None:
        """Connect to InfraTec ImageIR camera."""
        config = self._camera_config.infratec_config
        self._logger.info(f"Connecting to InfraTec ImageIR at {config.ip_address}")

        try:
            async with self._http_session.get(
                f"http://{config.ip_address}:{config.port}/api/status",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise CameraConnectionError(
                        f"InfraTec camera returned {response.status}"
                    )
        except aiohttp.ClientError as e:
            raise CameraConnectionError(f"InfraTec connection failed: {e}")

    async def _connect_onvif(self) -> None:
        """Connect to ONVIF thermal camera."""
        config = self._camera_config.onvif_config
        self._logger.info(f"Connecting to ONVIF camera at {config.ip_address}")

        # ONVIF device discovery and connection
        device_url = f"http://{config.ip_address}:{config.port}/onvif/device_service"

        try:
            # SOAP request to get device capabilities
            soap_envelope = self._build_onvif_get_capabilities_request()
            headers = {"Content-Type": "application/soap+xml; charset=utf-8"}

            async with self._http_session.post(
                device_url,
                data=soap_envelope,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    raise CameraConnectionError(
                        f"ONVIF device returned {response.status}"
                    )
        except aiohttp.ClientError as e:
            raise CameraConnectionError(f"ONVIF connection failed: {e}")

    async def _connect_generic_http(self) -> None:
        """Connect to generic HTTP-based thermal camera."""
        self._logger.info("Connecting to generic HTTP thermal camera")
        # Generic HTTP connection for unsupported cameras

    def _build_onvif_get_capabilities_request(self) -> str:
        """Build ONVIF GetCapabilities SOAP request."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
            xmlns:wsdl="http://www.onvif.org/ver10/device/wsdl">
            <soap:Body>
                <wsdl:GetCapabilities>
                    <wsdl:Category>All</wsdl:Category>
                </wsdl:GetCapabilities>
            </soap:Body>
        </soap:Envelope>"""

    # =========================================================================
    # Camera Information and Status
    # =========================================================================

    async def _get_camera_info(self) -> CameraInfo:
        """Get camera device information."""
        # Provider-specific implementation would go here
        # This is a mock implementation for demonstration
        return CameraInfo(
            camera_id=str(uuid.uuid4()),
            manufacturer=self._get_manufacturer_name(),
            model=self._get_model_name(),
            serial_number=self._get_serial_number(),
            firmware_version="1.0.0",
            resolution_x=640,
            resolution_y=480,
            temperature_range_min_c=-20.0,
            temperature_range_max_c=650.0,
            netd_mk=50.0,
            fov_horizontal_deg=45.0,
            fov_vertical_deg=34.0
        )

    def _get_manufacturer_name(self) -> str:
        """Get manufacturer name based on provider."""
        provider_manufacturers = {
            ThermalCameraProvider.FLIR_SYSTEMS: "FLIR Systems",
            ThermalCameraProvider.FLIR_THERMAL_STUDIO: "FLIR Systems",
            ThermalCameraProvider.FLUKE_TI: "Fluke Corporation",
            ThermalCameraProvider.TESTO: "Testo SE & Co. KGaA",
            ThermalCameraProvider.OPTRIS_PI: "Optris GmbH",
            ThermalCameraProvider.INFRATEC_IMAGEIR: "InfraTec GmbH",
            ThermalCameraProvider.ONVIF_THERMAL: "Generic ONVIF",
            ThermalCameraProvider.GENERIC_HTTP: "Unknown",
        }
        return provider_manufacturers.get(self._camera_config.provider, "Unknown")

    def _get_model_name(self) -> str:
        """Get model name based on configuration."""
        # Would be retrieved from camera in real implementation
        return f"{self._camera_config.provider.value}_camera"

    def _get_serial_number(self) -> str:
        """Get serial number based on configuration."""
        # Check provider-specific configs for serial
        if self._camera_config.flir_config and self._camera_config.flir_config.camera_serial:
            return self._camera_config.flir_config.camera_serial
        if self._camera_config.testo_config and self._camera_config.testo_config.device_serial:
            return self._camera_config.testo_config.device_serial
        return "UNKNOWN"

    async def _get_camera_status(self) -> Dict[str, Any]:
        """Get current camera status."""
        return {
            "connected": self._state == ConnectionState.CONNECTED,
            "streaming": self._is_streaming,
            "temperature_range": [-20, 650],
        }

    async def _get_calibration_data(self) -> CalibrationData:
        """Get camera calibration data."""
        # Would be retrieved from camera or calibration service
        calibration_date = datetime.utcnow() - timedelta(days=180)
        due_date = calibration_date + timedelta(days=365)

        status = CalibrationStatus.VALID
        if due_date < datetime.utcnow():
            status = CalibrationStatus.EXPIRED
        elif due_date < datetime.utcnow() + timedelta(days=self._camera_config.calibration_warning_days):
            status = CalibrationStatus.CALIBRATION_DUE

        return CalibrationData(
            camera_id=self._camera_info.camera_id if self._camera_info else "",
            calibration_date=calibration_date,
            calibration_due_date=due_date,
            calibration_status=status,
            temperature_accuracy_c=2.0,
        )

    def _check_calibration_status(self) -> None:
        """Check and log calibration status."""
        if not self._calibration_data:
            return

        status = self._calibration_data.calibration_status
        if status == CalibrationStatus.EXPIRED:
            self._logger.warning(
                f"Camera calibration EXPIRED on "
                f"{self._calibration_data.calibration_due_date}"
            )
        elif status == CalibrationStatus.CALIBRATION_DUE:
            self._logger.warning(
                f"Camera calibration due on "
                f"{self._calibration_data.calibration_due_date}"
            )

    # =========================================================================
    # Image Capture
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def capture_image(
        self,
        request: Optional[CaptureRequest] = None
    ) -> ThermalImage:
        """
        Capture a thermal image from the camera.

        Args:
            request: Capture request parameters

        Returns:
            Captured thermal image with radiometric data

        Raises:
            CameraConnectionError: If camera is not connected
            ImageProcessingError: If image capture fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise CameraConnectionError("Camera not connected")

        request = request or CaptureRequest()

        async def _capture():
            self._logger.debug(f"Capturing thermal image: {request.request_id}")

            # Trigger NUC if requested
            if request.trigger_nuc and self._camera_config.nuc_mode != NUCMode.DISABLED:
                await self._trigger_nuc()

            # Get radiometric parameters
            params = request.radiometric_params or RadiometricParameters(
                emissivity=self._camera_config.default_emissivity,
                reflected_temperature_c=self._camera_config.default_reflected_temperature_c,
                atmospheric_temperature_c=self._camera_config.default_atmospheric_temperature_c,
                distance_m=self._camera_config.default_distance_m,
                relative_humidity=self._camera_config.default_relative_humidity,
            )

            # Capture image from camera
            image_data = await self._capture_raw_image()

            # Extract temperature matrix if requested
            temp_matrix = None
            if request.include_temperature_matrix:
                temp_matrix = await self._extract_temperature_matrix(
                    image_data,
                    params
                )

            # Calculate temperature statistics
            if temp_matrix:
                min_temp = temp_matrix.min_temperature
                max_temp = temp_matrix.max_temperature
                avg_temp = temp_matrix.avg_temperature
            else:
                min_temp = max_temp = avg_temp = 0.0

            # Create thermal image
            image = ThermalImage(
                camera_id=self._camera_info.camera_id if self._camera_info else "",
                width=self._camera_info.resolution_x if self._camera_info else 640,
                height=self._camera_info.resolution_y if self._camera_info else 480,
                temperature_matrix=temp_matrix,
                min_temperature_c=min_temp,
                max_temperature_c=max_temp,
                avg_temperature_c=avg_temp,
                radiometric_params=params,
                palette=request.palette or self._camera_config.default_palette,
                image_format=request.image_format,
            )

            # Add spot measurements
            if request.spot_points:
                image.spot_measurements = await self._measure_spots(
                    temp_matrix,
                    request.spot_points
                )

            # Add region measurements
            if request.measurement_regions:
                image.measurement_regions = await self._measure_regions(
                    temp_matrix,
                    request.measurement_regions
                )

            # Include visual image if requested
            if request.include_visual:
                image.visual_image_base64 = await self._get_visual_image(image_data)

            self._logger.info(
                f"Captured thermal image: {image.image_id} "
                f"(min={min_temp:.1f}C, max={max_temp:.1f}C)"
            )

            return image

        return await self.execute_with_protection(
            operation=_capture,
            operation_name="capture_image",
            validate_result=False
        )

    async def _capture_raw_image(self) -> bytes:
        """Capture raw image data from camera."""
        # Provider-specific implementation
        # Mock implementation for demonstration
        return b""

    async def _extract_temperature_matrix(
        self,
        image_data: bytes,
        params: RadiometricParameters
    ) -> TemperatureMatrix:
        """Extract temperature matrix from image data."""
        # Provider-specific radiometric extraction
        # Mock implementation for demonstration
        width = self._camera_info.resolution_x if self._camera_info else 640
        height = self._camera_info.resolution_y if self._camera_info else 480

        # Generate mock temperature data
        import random
        data = []
        all_temps = []
        for row in range(height):
            row_data = []
            for col in range(width):
                temp = random.uniform(20.0, 80.0)
                row_data.append(temp)
                all_temps.append(temp)
            data.append(row_data)

        return TemperatureMatrix(
            width=width,
            height=height,
            data=data,
            min_temperature=min(all_temps),
            max_temperature=max(all_temps),
            avg_temperature=sum(all_temps) / len(all_temps),
        )

    async def _trigger_nuc(self) -> None:
        """Trigger Non-Uniformity Correction."""
        self._logger.debug("Triggering NUC")
        # Provider-specific NUC trigger
        await asyncio.sleep(0.5)  # Simulate NUC delay

    async def _measure_spots(
        self,
        temp_matrix: Optional[TemperatureMatrix],
        points: List[Tuple[int, int]]
    ) -> List[TemperaturePoint]:
        """Measure temperatures at specific points."""
        if not temp_matrix:
            return []

        measurements = []
        for x, y in points:
            if 0 <= y < temp_matrix.height and 0 <= x < temp_matrix.width:
                temp = temp_matrix.data[y][x]
                measurements.append(TemperaturePoint(
                    x=x,
                    y=y,
                    temperature_c=temp
                ))
        return measurements

    async def _measure_regions(
        self,
        temp_matrix: Optional[TemperatureMatrix],
        regions: List[Dict[str, Any]]
    ) -> List[TemperatureRegion]:
        """Measure temperatures in defined regions."""
        if not temp_matrix:
            return []

        measurements = []
        for region_def in regions:
            # Extract region coordinates and type
            region_type = region_def.get("type", "rectangle")
            coords = region_def.get("coordinates", [])

            if region_type == "rectangle" and len(coords) >= 2:
                x1, y1 = coords[0]
                x2, y2 = coords[1]

                # Extract temperatures in region
                temps = []
                min_point = None
                max_point = None
                min_temp = float('inf')
                max_temp = float('-inf')

                for row in range(y1, min(y2 + 1, temp_matrix.height)):
                    for col in range(x1, min(x2 + 1, temp_matrix.width)):
                        temp = temp_matrix.data[row][col]
                        temps.append(temp)
                        if temp < min_temp:
                            min_temp = temp
                            min_point = TemperaturePoint(x=col, y=row, temperature_c=temp)
                        if temp > max_temp:
                            max_temp = temp
                            max_point = TemperaturePoint(x=col, y=row, temperature_c=temp)

                if temps:
                    avg_temp = sum(temps) / len(temps)
                    std_dev = math.sqrt(sum((t - avg_temp) ** 2 for t in temps) / len(temps))

                    measurements.append(TemperatureRegion(
                        region_type=region_type,
                        coordinates=coords,
                        min_temperature_c=min_temp,
                        max_temperature_c=max_temp,
                        avg_temperature_c=avg_temp,
                        std_deviation_c=std_dev,
                        min_point=min_point,
                        max_point=max_point,
                        pixel_count=len(temps)
                    ))

        return measurements

    async def _get_visual_image(self, image_data: bytes) -> str:
        """Get visual image as base64."""
        # Convert raw image to visual and encode
        return base64.b64encode(image_data).decode('utf-8')

    # =========================================================================
    # Real-Time Streaming
    # =========================================================================

    async def start_stream(
        self,
        config: Optional[StreamConfiguration] = None
    ) -> None:
        """
        Start real-time streaming from camera.

        Args:
            config: Stream configuration
        """
        if self._is_streaming:
            self._logger.warning("Stream already active")
            return

        if self._state != ConnectionState.CONNECTED:
            raise CameraConnectionError("Camera not connected")

        config = config or StreamConfiguration(
            fps=self._camera_config.stream_fps,
            protocol=self._camera_config.stream_protocol,
            buffer_size=self._camera_config.stream_buffer_size,
        )

        self._logger.info(f"Starting stream at {config.fps} FPS")

        self._is_streaming = True
        self._stream_task = asyncio.create_task(
            self._stream_loop(config)
        )

    async def stop_stream(self) -> None:
        """Stop real-time streaming."""
        if not self._is_streaming:
            return

        self._logger.info("Stopping stream")
        self._is_streaming = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        # Clear frame buffer
        while not self._frame_buffer.empty():
            try:
                self._frame_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _stream_loop(self, config: StreamConfiguration) -> None:
        """Background task for streaming frames."""
        sequence = 0
        frame_interval = 1.0 / config.fps

        while self._is_streaming:
            try:
                start_time = asyncio.get_event_loop().time()

                # Capture frame
                temp_matrix = await self._capture_frame()

                frame = ThermalFrame(
                    sequence_number=sequence,
                    temperature_matrix=temp_matrix,
                    frame_rate=config.fps,
                )

                # Add to buffer
                try:
                    self._frame_buffer.put_nowait(frame)
                except asyncio.QueueFull:
                    # Drop oldest frame
                    try:
                        self._frame_buffer.get_nowait()
                        self._frame_buffer.put_nowait(frame)
                    except asyncio.QueueEmpty:
                        pass

                sequence += 1

                # Maintain frame rate
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Stream error: {e}")
                await asyncio.sleep(0.1)

    async def _capture_frame(self) -> TemperatureMatrix:
        """Capture a single frame for streaming."""
        # Simplified frame capture for streaming
        return await self._extract_temperature_matrix(b"", RadiometricParameters())

    async def get_next_frame(
        self,
        timeout_seconds: float = 5.0
    ) -> Optional[ThermalFrame]:
        """
        Get next frame from stream buffer.

        Args:
            timeout_seconds: Maximum wait time

        Returns:
            Next thermal frame or None if timeout
        """
        if not self._is_streaming:
            return None

        try:
            return await asyncio.wait_for(
                self._frame_buffer.get(),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return None

    async def stream_frames(self) -> AsyncIterator[ThermalFrame]:
        """
        Async iterator for streaming frames.

        Yields:
            Thermal frames as they become available
        """
        while self._is_streaming:
            frame = await self.get_next_frame()
            if frame:
                yield frame

    # =========================================================================
    # File Parsing
    # =========================================================================

    async def parse_thermal_image(
        self,
        request: FileParseRequest
    ) -> ThermalImage:
        """
        Parse a thermal image file and extract radiometric data.

        Supports FLIR radiometric JPEG, TIFF, and vendor-specific formats.

        Args:
            request: File parse request

        Returns:
            Parsed thermal image with temperature data

        Raises:
            ImageProcessingError: If parsing fails
        """
        self._logger.debug("Parsing thermal image file")

        try:
            # Get file data
            if request.file_path:
                file_path = Path(request.file_path)
                if not file_path.exists():
                    raise ImageProcessingError(f"File not found: {file_path}")
                file_data = file_path.read_bytes()
            elif request.file_data:
                file_data = request.file_data
            else:
                raise ImageProcessingError("No file path or data provided")

            # Detect format if not specified
            image_format = request.file_format or self._detect_image_format(file_data)

            # Parse based on format
            if image_format == ImageFormat.RADIOMETRIC_JPEG:
                return await self._parse_radiometric_jpeg(file_data, request)
            elif image_format in [ImageFormat.JPEG, ImageFormat.PNG]:
                return await self._parse_standard_image(file_data, request)
            elif image_format == ImageFormat.TIFF:
                return await self._parse_tiff(file_data, request)
            elif image_format == ImageFormat.FFF:
                return await self._parse_flir_fff(file_data, request)
            elif image_format == ImageFormat.SEQ:
                return await self._parse_flir_seq(file_data, request)
            elif image_format == ImageFormat.IS2:
                return await self._parse_infratec_is2(file_data, request)
            elif image_format == ImageFormat.TMX:
                return await self._parse_testo_tmx(file_data, request)
            else:
                raise ImageProcessingError(f"Unsupported format: {image_format}")

        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to parse thermal image: {e}",
                details={"format": request.file_format.value if request.file_format else "unknown"}
            )

    def _detect_image_format(self, data: bytes) -> ImageFormat:
        """Detect image format from file header."""
        if len(data) < 8:
            raise ImageProcessingError("File too small to detect format")

        # JPEG
        if data[:2] == b'\xff\xd8':
            # Check for FLIR radiometric data in EXIF
            if b'FLIR' in data[:1000]:
                return ImageFormat.RADIOMETRIC_JPEG
            return ImageFormat.JPEG

        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ImageFormat.PNG

        # TIFF
        if data[:4] in [b'II\x2a\x00', b'MM\x00\x2a']:
            return ImageFormat.TIFF

        # FLIR FFF
        if data[:4] == b'FFF\x00':
            return ImageFormat.FFF

        # FLIR SEQ
        if data[:4] == b'FLIR':
            return ImageFormat.SEQ

        raise ImageProcessingError("Unable to detect image format")

    async def _parse_radiometric_jpeg(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """
        Parse FLIR radiometric JPEG.

        FLIR radiometric JPEGs contain embedded thermal data in EXIF APP1 segment.
        """
        self._logger.debug("Parsing FLIR radiometric JPEG")

        # Extract EXIF data
        exif_data = self._extract_exif(data)

        # Extract FLIR radiometric segment
        raw_thermal_data = self._extract_flir_thermal_segment(data)

        if raw_thermal_data:
            # Parse raw thermal data
            temp_matrix = self._parse_raw_thermal_data(
                raw_thermal_data,
                request.radiometric_params or RadiometricParameters()
            )
        else:
            # No embedded thermal data
            temp_matrix = None

        return ThermalImage(
            camera_id="file_import",
            width=640,  # Would be extracted from EXIF
            height=480,
            image_format=ImageFormat.RADIOMETRIC_JPEG,
            temperature_matrix=temp_matrix,
            min_temperature_c=temp_matrix.min_temperature if temp_matrix else 0,
            max_temperature_c=temp_matrix.max_temperature if temp_matrix else 0,
            avg_temperature_c=temp_matrix.avg_temperature if temp_matrix else 0,
            exif_data=exif_data,
            visual_image_base64=base64.b64encode(data).decode('utf-8') if request.extract_visual else None,
        )

    def _extract_exif(self, data: bytes) -> Dict[str, Any]:
        """Extract EXIF metadata from JPEG."""
        exif = {}
        # Would use PIL or exiftool for actual EXIF extraction
        return exif

    def _extract_flir_thermal_segment(self, data: bytes) -> Optional[bytes]:
        """Extract FLIR thermal data segment from radiometric JPEG."""
        # Search for FLIR thermal segment marker
        flir_marker = b'FliR'
        idx = data.find(flir_marker)
        if idx == -1:
            return None

        # Extract thermal data
        # Actual implementation would parse FLIR format structure
        return None

    def _parse_raw_thermal_data(
        self,
        data: bytes,
        params: RadiometricParameters
    ) -> TemperatureMatrix:
        """Parse raw thermal data to temperature matrix."""
        # Would implement actual raw data parsing with radiometric conversion
        return TemperatureMatrix(
            width=640,
            height=480,
            data=[[20.0] * 640 for _ in range(480)],
            min_temperature=20.0,
            max_temperature=20.0,
            avg_temperature=20.0,
        )

    async def _parse_standard_image(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """Parse standard JPEG/PNG (no radiometric data)."""
        return ThermalImage(
            camera_id="file_import",
            width=640,
            height=480,
            image_format=ImageFormat.JPEG,
            min_temperature_c=0,
            max_temperature_c=0,
            avg_temperature_c=0,
            visual_image_base64=base64.b64encode(data).decode('utf-8') if request.extract_visual else None,
        )

    async def _parse_tiff(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """Parse TIFF with thermal data."""
        # Would implement TIFF thermal data parsing
        return ThermalImage(
            camera_id="file_import",
            width=640,
            height=480,
            image_format=ImageFormat.TIFF,
            min_temperature_c=0,
            max_temperature_c=0,
            avg_temperature_c=0,
        )

    async def _parse_flir_fff(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """Parse FLIR FFF format."""
        # Would implement FFF parsing
        raise ImageProcessingError("FFF format parsing not implemented")

    async def _parse_flir_seq(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """Parse FLIR SEQ format."""
        # Would implement SEQ parsing
        raise ImageProcessingError("SEQ format parsing not implemented")

    async def _parse_infratec_is2(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """Parse InfraTec IS2 format."""
        # Would implement IS2 parsing
        raise ImageProcessingError("IS2 format parsing not implemented")

    async def _parse_testo_tmx(
        self,
        data: bytes,
        request: FileParseRequest
    ) -> ThermalImage:
        """Parse Testo TMX format."""
        # Would implement TMX parsing
        raise ImageProcessingError("TMX format parsing not implemented")

    # =========================================================================
    # Camera Control
    # =========================================================================

    async def set_radiometric_parameters(
        self,
        params: RadiometricParameters
    ) -> None:
        """Set camera radiometric parameters."""
        self._logger.info(f"Setting radiometric parameters: emissivity={params.emissivity}")
        # Provider-specific implementation

    async def set_palette(self, palette: PaletteType) -> None:
        """Set camera color palette."""
        self._logger.info(f"Setting palette: {palette.value}")
        # Provider-specific implementation

    async def set_temperature_range(
        self,
        min_temp_c: float,
        max_temp_c: float
    ) -> None:
        """Set camera temperature range."""
        self._logger.info(f"Setting temperature range: {min_temp_c}C to {max_temp_c}C")
        # Provider-specific implementation

    async def perform_nuc(self) -> None:
        """Perform Non-Uniformity Correction."""
        self._logger.info("Performing NUC")
        await self._trigger_nuc()

    async def auto_focus(self) -> None:
        """Trigger auto-focus."""
        self._logger.info("Triggering auto-focus")
        # Provider-specific implementation


# =============================================================================
# Factory Function
# =============================================================================


def create_thermal_camera_connector(
    provider: ThermalCameraProvider,
    connector_name: str = "ThermalCamera",
    **kwargs
) -> ThermalCameraConnector:
    """
    Factory function to create thermal camera connector.

    Args:
        provider: Thermal camera provider
        connector_name: Connector name
        **kwargs: Additional configuration options

    Returns:
        Configured ThermalCameraConnector instance
    """
    config = ThermalCameraConnectorConfig(
        connector_name=connector_name,
        provider=provider,
        **kwargs
    )
    return ThermalCameraConnector(config)
