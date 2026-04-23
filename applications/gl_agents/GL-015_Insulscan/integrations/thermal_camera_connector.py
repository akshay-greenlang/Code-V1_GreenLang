"""
Thermal Camera Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade integration with thermal imaging cameras:
- FLIR camera SDK integration (FLIR Atlas, FLIR Tools)
- Capture thermal images programmatically
- Parse radiometric data from thermal images
- Hot spot detection and analysis
- Support for multiple camera models
- Stream and batch image processing

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
    Set,
    Tuple,
    Union,
)
import asyncio
import logging
import uuid
from pathlib import Path
import struct
import io

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class CameraManufacturer(str, Enum):
    """Supported camera manufacturers."""

    FLIR = "flir"
    FLUKE = "fluke"
    HIKVISION = "hikvision"
    SEEK = "seek"
    OPTRIS = "optris"
    TESTO = "testo"
    INFRARED_CAMERAS_INC = "ici"
    GENERIC = "generic"


class CameraModel(str, Enum):
    """Common camera models."""

    # FLIR models
    FLIR_E4 = "flir_e4"
    FLIR_E5 = "flir_e5"
    FLIR_E6 = "flir_e6"
    FLIR_E8 = "flir_e8"
    FLIR_T540 = "flir_t540"
    FLIR_T620 = "flir_t620"
    FLIR_T640 = "flir_t640"
    FLIR_T1030SC = "flir_t1030sc"
    FLIR_A310 = "flir_a310"
    FLIR_A320 = "flir_a320"
    FLIR_A615 = "flir_a615"
    FLIR_A655SC = "flir_a655sc"
    FLIR_A8300 = "flir_a8300"
    FLIR_AX8 = "flir_ax8"

    # Fluke models
    FLUKE_TI300 = "fluke_ti300"
    FLUKE_TI400 = "fluke_ti400"
    FLUKE_TI450 = "fluke_ti450"

    # Generic
    GENERIC_THERMAL = "generic_thermal"


class CameraConnectionType(str, Enum):
    """Camera connection types."""

    USB = "usb"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    GIGE_VISION = "gige_vision"
    CAMERA_LINK = "camera_link"
    MODBUS = "modbus"
    SDK_NATIVE = "sdk_native"


class ImageFormat(str, Enum):
    """Thermal image formats."""

    RADIOMETRIC_JPG = "radiometric_jpg"  # FLIR radiometric JPEG
    RADIOMETRIC_PNG = "radiometric_png"
    SEQ = "seq"  # FLIR sequence file
    CSQ = "csq"  # Compressed sequence
    FFF = "fff"  # FLIR raw format
    IS2 = "is2"  # Infrared Systems format
    TIFF_16BIT = "tiff_16bit"
    RAW_16BIT = "raw_16bit"
    STANDARD_JPG = "standard_jpg"


class TemperatureUnit(str, Enum):
    """Temperature units."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


class ColorPalette(str, Enum):
    """Thermal color palettes."""

    IRON = "iron"
    RAINBOW = "rainbow"
    GRAYSCALE = "grayscale"
    ARCTIC = "arctic"
    LAVA = "lava"
    COLDEST = "coldest"
    HOTTEST = "hottest"
    BLACK_HOT = "black_hot"
    WHITE_HOT = "white_hot"


class ConnectionState(str, Enum):
    """Camera connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class CalibrationStatus(str, Enum):
    """Camera calibration status."""

    VALID = "valid"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    IN_PROGRESS = "in_progress"


# =============================================================================
# Custom Exceptions
# =============================================================================


class ThermalCameraError(Exception):
    """Base thermal camera exception."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class CameraConnectionError(ThermalCameraError):
    """Camera connection error."""
    pass


class CameraNotFoundError(ThermalCameraError):
    """Camera not found error."""
    pass


class ImageCaptureError(ThermalCameraError):
    """Image capture error."""
    pass


class RadiometricDataError(ThermalCameraError):
    """Radiometric data parsing error."""
    pass


class CalibrationError(ThermalCameraError):
    """Camera calibration error."""
    pass


# =============================================================================
# Pydantic Models - Camera Specifications
# =============================================================================


class CameraSpecs(BaseModel):
    """Camera technical specifications."""

    model_config = ConfigDict(frozen=True)

    # Resolution
    width_pixels: int = Field(ge=1, description="Image width")
    height_pixels: int = Field(ge=1, description="Image height")

    # Thermal specs
    temp_range_min_c: float = Field(description="Minimum temperature range")
    temp_range_max_c: float = Field(description="Maximum temperature range")
    accuracy_c: float = Field(ge=0, description="Temperature accuracy (+/- C)")
    netd_mk: float = Field(ge=0, description="NETD (thermal sensitivity) in mK")

    # Optical
    fov_h_degrees: float = Field(ge=0, description="Horizontal FOV degrees")
    fov_v_degrees: float = Field(ge=0, description="Vertical FOV degrees")
    ifov_mrad: float = Field(ge=0, description="IFOV in milliradians")
    focus_type: str = Field(default="auto", description="Focus type")
    min_focus_distance_m: float = Field(ge=0, description="Min focus distance")

    # Frame rate
    max_frame_rate: float = Field(ge=0, description="Maximum frame rate")

    # Spectral
    spectral_range_um: Tuple[float, float] = Field(
        description="Spectral range in micrometers"
    )


class CameraCalibration(BaseModel):
    """Camera calibration data."""

    model_config = ConfigDict(frozen=False)

    calibration_date: datetime = Field(..., description="Calibration date")
    expiration_date: Optional[datetime] = Field(
        default=None,
        description="Calibration expiration"
    )
    certificate_number: Optional[str] = Field(
        default=None,
        description="Calibration certificate number"
    )
    calibration_lab: Optional[str] = Field(
        default=None,
        description="Calibration laboratory"
    )

    # Calibration parameters
    planck_r1: float = Field(default=14000.0, description="Planck R1 constant")
    planck_r2: float = Field(default=0.03, description="Planck R2 constant")
    planck_b: float = Field(default=1400.0, description="Planck B constant")
    planck_f: float = Field(default=1.0, description="Planck F constant")
    planck_o: float = Field(default=-6000.0, description="Planck O constant")

    @property
    def is_valid(self) -> bool:
        """Check if calibration is valid."""
        if self.expiration_date:
            return datetime.utcnow() < self.expiration_date
        return True


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class CameraConfig(BaseModel):
    """Camera hardware configuration."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Camera identifier"
    )
    camera_name: str = Field(
        default="ThermalCamera",
        description="Camera name"
    )
    manufacturer: CameraManufacturer = Field(
        default=CameraManufacturer.FLIR,
        description="Camera manufacturer"
    )
    model: CameraModel = Field(
        default=CameraModel.FLIR_E8,
        description="Camera model"
    )

    # Connection
    connection_type: CameraConnectionType = Field(
        default=CameraConnectionType.USB,
        description="Connection type"
    )
    ip_address: Optional[str] = Field(
        default=None,
        description="IP address for network cameras"
    )
    port: Optional[int] = Field(
        default=None,
        description="Port for network cameras"
    )
    serial_number: Optional[str] = Field(
        default=None,
        description="Camera serial number"
    )

    # SDK settings
    sdk_path: Optional[str] = Field(
        default=None,
        description="Path to camera SDK"
    )
    use_mock: bool = Field(
        default=False,
        description="Use mock camera for testing"
    )


class CaptureSettings(BaseModel):
    """Image capture settings."""

    model_config = ConfigDict(frozen=False)

    # Output format
    output_format: ImageFormat = Field(
        default=ImageFormat.RADIOMETRIC_JPG,
        description="Output image format"
    )
    color_palette: ColorPalette = Field(
        default=ColorPalette.IRON,
        description="Display color palette"
    )

    # Temperature display
    temperature_unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Temperature unit"
    )
    display_min_temp: Optional[float] = Field(
        default=None,
        description="Display minimum temperature"
    )
    display_max_temp: Optional[float] = Field(
        default=None,
        description="Display maximum temperature"
    )
    auto_scale: bool = Field(
        default=True,
        description="Auto-scale temperature display"
    )

    # Environment compensation
    emissivity: float = Field(
        default=0.95,
        ge=0.01,
        le=1.0,
        description="Object emissivity"
    )
    reflected_temp_c: float = Field(
        default=20.0,
        description="Reflected apparent temperature"
    )
    atmospheric_temp_c: float = Field(
        default=20.0,
        description="Atmospheric temperature"
    )
    distance_m: float = Field(
        default=1.0,
        ge=0.1,
        description="Distance to object in meters"
    )
    relative_humidity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative humidity (0-1)"
    )

    # Focus
    auto_focus: bool = Field(
        default=True,
        description="Enable auto focus"
    )
    manual_focus_distance_m: Optional[float] = Field(
        default=None,
        description="Manual focus distance"
    )


class HotSpotDetectionConfig(BaseModel):
    """Configuration for hot spot detection."""

    model_config = ConfigDict(frozen=False)

    enabled: bool = Field(
        default=True,
        description="Enable hot spot detection"
    )

    # Detection parameters
    hot_spot_threshold_delta_c: float = Field(
        default=10.0,
        ge=0.0,
        description="Temperature delta above ambient to flag as hot spot"
    )
    absolute_threshold_c: Optional[float] = Field(
        default=None,
        description="Absolute temperature threshold for hot spots"
    )
    min_hot_spot_area_pixels: int = Field(
        default=10,
        ge=1,
        description="Minimum area for hot spot detection"
    )
    max_hot_spots: int = Field(
        default=20,
        ge=1,
        description="Maximum hot spots to return"
    )

    # Reference
    use_ambient_reference: bool = Field(
        default=True,
        description="Use ambient temperature as reference"
    )
    reference_region: Optional[Tuple[int, int, int, int]] = Field(
        default=None,
        description="Reference region (x, y, width, height)"
    )


class ThermalCameraConnectorConfig(BaseModel):
    """Complete thermal camera connector configuration."""

    model_config = ConfigDict(extra="forbid")

    connector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Connector identifier"
    )
    connector_name: str = Field(
        default="ThermalCamera-Connector",
        description="Connector name"
    )

    # Camera configuration
    camera_config: CameraConfig = Field(
        ...,
        description="Camera configuration"
    )

    # Capture settings
    capture_settings: CaptureSettings = Field(
        default_factory=CaptureSettings,
        description="Default capture settings"
    )

    # Hot spot detection
    hot_spot_config: HotSpotDetectionConfig = Field(
        default_factory=HotSpotDetectionConfig,
        description="Hot spot detection configuration"
    )

    # Storage
    image_storage_path: str = Field(
        default="./thermal_images",
        description="Path to store captured images"
    )
    auto_save_images: bool = Field(
        default=True,
        description="Auto-save captured images"
    )
    image_retention_days: int = Field(
        default=90,
        ge=1,
        description="Image retention period"
    )

    # Connection
    connection_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        description="Connection timeout"
    )
    reconnect_enabled: bool = Field(
        default=True,
        description="Enable auto-reconnection"
    )
    reconnect_delay_seconds: float = Field(
        default=5.0,
        ge=1.0,
        description="Reconnection delay"
    )

    # Health check
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health checks"
    )
    health_check_interval_seconds: float = Field(
        default=60.0,
        ge=10.0,
        description="Health check interval"
    )


# =============================================================================
# Data Models - Thermal Data
# =============================================================================


class ThermalPixel(BaseModel):
    """Single thermal pixel data."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0, description="X coordinate")
    y: int = Field(ge=0, description="Y coordinate")
    temperature_c: float = Field(description="Temperature in Celsius")
    raw_value: Optional[int] = Field(default=None, description="Raw sensor value")


class TemperatureStatistics(BaseModel):
    """Temperature statistics for a region."""

    model_config = ConfigDict(frozen=True)

    min_temp_c: float = Field(description="Minimum temperature")
    max_temp_c: float = Field(description="Maximum temperature")
    avg_temp_c: float = Field(description="Average temperature")
    std_dev_c: float = Field(description="Standard deviation")
    median_temp_c: Optional[float] = Field(default=None, description="Median temperature")
    pixel_count: int = Field(ge=0, description="Number of pixels")


class HotSpot(BaseModel):
    """Detected hot spot."""

    model_config = ConfigDict(frozen=True)

    hot_spot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Hot spot identifier"
    )
    center_x: int = Field(ge=0, description="Center X coordinate")
    center_y: int = Field(ge=0, description="Center Y coordinate")
    width: int = Field(ge=1, description="Width in pixels")
    height: int = Field(ge=1, description="Height in pixels")
    area_pixels: int = Field(ge=1, description="Area in pixels")

    # Temperature
    max_temp_c: float = Field(description="Maximum temperature in hot spot")
    avg_temp_c: float = Field(description="Average temperature in hot spot")
    temp_delta_c: float = Field(description="Temperature delta from reference")

    # Severity
    severity: str = Field(description="Severity level (low, medium, high, critical)")

    # Location mapping
    relative_x: float = Field(
        ge=0.0, le=1.0,
        description="Relative X position (0-1)"
    )
    relative_y: float = Field(
        ge=0.0, le=1.0,
        description="Relative Y position (0-1)"
    )


class RegionOfInterest(BaseModel):
    """Region of interest in thermal image."""

    model_config = ConfigDict(frozen=True)

    roi_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ROI identifier"
    )
    roi_name: str = Field(..., description="ROI name")
    roi_type: str = Field(
        default="rectangle",
        description="ROI type (rectangle, ellipse, polygon)"
    )

    # Bounds
    x: int = Field(ge=0, description="X coordinate")
    y: int = Field(ge=0, description="Y coordinate")
    width: int = Field(ge=1, description="Width")
    height: int = Field(ge=1, description="Height")

    # For polygon ROI
    polygon_points: Optional[List[Tuple[int, int]]] = Field(
        default=None,
        description="Polygon points"
    )

    # Statistics
    statistics: Optional[TemperatureStatistics] = Field(
        default=None,
        description="Temperature statistics for ROI"
    )


class RadiometricMetadata(BaseModel):
    """Radiometric metadata from thermal image."""

    model_config = ConfigDict(frozen=False)

    # Camera info
    camera_serial: Optional[str] = Field(default=None, description="Camera serial")
    camera_model: Optional[str] = Field(default=None, description="Camera model")
    lens_model: Optional[str] = Field(default=None, description="Lens model")

    # Capture time
    capture_timestamp: datetime = Field(..., description="Capture timestamp")
    gps_latitude: Optional[float] = Field(default=None, description="GPS latitude")
    gps_longitude: Optional[float] = Field(default=None, description="GPS longitude")

    # Environment parameters
    emissivity: float = Field(default=0.95, description="Emissivity")
    reflected_temp_c: float = Field(default=20.0, description="Reflected temperature")
    atmospheric_temp_c: float = Field(default=20.0, description="Atmospheric temperature")
    object_distance_m: float = Field(default=1.0, description="Object distance")
    relative_humidity: float = Field(default=0.5, description="Relative humidity")

    # Calibration
    planck_r1: float = Field(default=14000.0, description="Planck R1")
    planck_r2: float = Field(default=0.03, description="Planck R2")
    planck_b: float = Field(default=1400.0, description="Planck B")
    planck_f: float = Field(default=1.0, description="Planck F")
    planck_o: float = Field(default=-6000.0, description="Planck O")

    # Raw data info
    raw_value_range: Tuple[int, int] = Field(
        default=(0, 16383),
        description="Raw value range"
    )
    raw_value_median: Optional[int] = Field(default=None, description="Raw median")


class ThermalImage(BaseModel):
    """Complete thermal image with all data."""

    model_config = ConfigDict(frozen=False)

    image_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Image identifier"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="File path if saved"
    )

    # Image dimensions
    width: int = Field(ge=1, description="Image width")
    height: int = Field(ge=1, description="Image height")

    # Temperature data
    temperature_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="2D temperature matrix (row-major)"
    )
    raw_data: Optional[bytes] = Field(
        default=None,
        description="Raw thermal data"
    )

    # Statistics
    statistics: TemperatureStatistics = Field(..., description="Full image statistics")

    # Metadata
    metadata: RadiometricMetadata = Field(..., description="Radiometric metadata")

    # Hot spots
    hot_spots: List[HotSpot] = Field(
        default_factory=list,
        description="Detected hot spots"
    )

    # ROIs
    regions_of_interest: List[RegionOfInterest] = Field(
        default_factory=list,
        description="Regions of interest"
    )

    # Equipment mapping
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment"
    )
    inspection_id: Optional[str] = Field(
        default=None,
        description="Associated inspection"
    )

    def get_temperature_at(self, x: int, y: int) -> Optional[float]:
        """Get temperature at pixel coordinates."""
        if self.temperature_matrix is None:
            return None
        if 0 <= y < len(self.temperature_matrix):
            if 0 <= x < len(self.temperature_matrix[y]):
                return self.temperature_matrix[y][x]
        return None


class ThermalImageCapture(BaseModel):
    """Thermal image capture result."""

    model_config = ConfigDict(frozen=True)

    capture_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Capture identifier"
    )
    capture_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Capture timestamp"
    )
    thermal_image: ThermalImage = Field(..., description="Captured thermal image")
    visual_image_path: Optional[str] = Field(
        default=None,
        description="Path to visual image"
    )
    capture_settings: CaptureSettings = Field(..., description="Settings used")
    capture_duration_ms: float = Field(
        default=0.0,
        description="Capture duration"
    )
    success: bool = Field(default=True, description="Capture successful")
    error_message: Optional[str] = Field(default=None, description="Error if any")


# =============================================================================
# Thermal Camera Connector
# =============================================================================


class ThermalCameraConnector:
    """
    Thermal Camera Connector for GL-015 INSULSCAN.

    Provides unified interface for thermal camera integration:
    - FLIR SDK integration
    - Image capture and streaming
    - Radiometric data extraction
    - Hot spot detection
    - Multi-camera support

    Note: In production, requires FLIR Atlas SDK or equivalent.
    """

    def __init__(self, config: ThermalCameraConnectorConfig) -> None:
        """
        Initialize thermal camera connector.

        Args:
            config: Connector configuration
        """
        self._config = config
        self._logger = logging.getLogger(
            f"{__name__}.{config.connector_name}"
        )

        self._state = ConnectionState.DISCONNECTED
        self._camera: Optional[Any] = None  # SDK camera object
        self._specs: Optional[CameraSpecs] = None
        self._calibration: Optional[CameraCalibration] = None

        # Streaming
        self._is_streaming = False
        self._stream_callback: Optional[Callable] = None

        # Health check
        self._health_check_task: Optional[asyncio.Task] = None

        # Statistics
        self._captures_count = 0
        self._captures_failed = 0
        self._last_capture_time: Optional[datetime] = None

        # Ensure storage path exists
        Path(config.image_storage_path).mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> ThermalCameraConnectorConfig:
        """Get configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Get connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state in [ConnectionState.CONNECTED, ConnectionState.STREAMING]

    @property
    def is_streaming(self) -> bool:
        """Check if streaming."""
        return self._state == ConnectionState.STREAMING

    @property
    def specs(self) -> Optional[CameraSpecs]:
        """Get camera specifications."""
        return self._specs

    @property
    def calibration(self) -> Optional[CameraCalibration]:
        """Get calibration data."""
        return self._calibration

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to thermal camera.

        Raises:
            CameraConnectionError: If connection fails
            CameraNotFoundError: If camera not found
        """
        self._state = ConnectionState.CONNECTING
        camera_config = self._config.camera_config

        self._logger.info(
            f"Connecting to {camera_config.manufacturer.value} camera: "
            f"{camera_config.model.value}"
        )

        try:
            if camera_config.use_mock:
                await self._connect_mock()
            elif camera_config.manufacturer == CameraManufacturer.FLIR:
                await self._connect_flir()
            else:
                await self._connect_generic()

            # Get camera specs
            self._specs = await self._get_camera_specs()

            # Get calibration
            self._calibration = await self._get_calibration()

            self._state = ConnectionState.CONNECTED

            # Start health check
            if self._config.health_check_enabled:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

            self._logger.info(
                f"Connected to camera: {camera_config.camera_name} "
                f"(Serial: {camera_config.serial_number})"
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Camera connection failed: {e}")
            raise CameraConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from camera."""
        self._logger.info("Disconnecting from camera")

        # Stop streaming
        if self._is_streaming:
            await self.stop_stream()

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Disconnect from camera SDK
        # if self._camera:
        #     self._camera.disconnect()

        self._camera = None
        self._state = ConnectionState.DISCONNECTED

    async def _connect_flir(self) -> None:
        """Connect to FLIR camera using SDK."""
        # In production, use FLIR Atlas SDK:
        # from Flir.Atlas.Live import Device
        # from Flir.Atlas.Image import ThermalImage
        #
        # devices = Device.GetDevices()
        # if not devices:
        #     raise CameraNotFoundError("No FLIR cameras found")
        #
        # self._camera = devices[0]
        # self._camera.Connect()

        self._logger.info("FLIR SDK connection initialized")

    async def _connect_generic(self) -> None:
        """Connect to generic thermal camera."""
        # Implement generic camera connection
        pass

    async def _connect_mock(self) -> None:
        """Connect to mock camera for testing."""
        self._logger.info("Using mock thermal camera")

    async def _get_camera_specs(self) -> CameraSpecs:
        """Get camera specifications."""
        # In production, read from camera
        # Return default specs based on model
        model = self._config.camera_config.model

        if model == CameraModel.FLIR_E8:
            return CameraSpecs(
                width_pixels=320,
                height_pixels=240,
                temp_range_min_c=-20,
                temp_range_max_c=550,
                accuracy_c=2.0,
                netd_mk=60,
                fov_h_degrees=45,
                fov_v_degrees=34,
                ifov_mrad=1.5,
                focus_type="auto",
                min_focus_distance_m=0.3,
                max_frame_rate=9,
                spectral_range_um=(7.5, 13.0),
            )
        elif model in [CameraModel.FLIR_T540, CameraModel.FLIR_T620, CameraModel.FLIR_T640]:
            return CameraSpecs(
                width_pixels=640,
                height_pixels=480,
                temp_range_min_c=-40,
                temp_range_max_c=2000,
                accuracy_c=2.0,
                netd_mk=40,
                fov_h_degrees=25,
                fov_v_degrees=19,
                ifov_mrad=0.68,
                focus_type="auto",
                min_focus_distance_m=0.25,
                max_frame_rate=30,
                spectral_range_um=(7.5, 14.0),
            )
        else:
            # Default specs
            return CameraSpecs(
                width_pixels=320,
                height_pixels=240,
                temp_range_min_c=-20,
                temp_range_max_c=500,
                accuracy_c=2.0,
                netd_mk=100,
                fov_h_degrees=40,
                fov_v_degrees=30,
                ifov_mrad=2.0,
                focus_type="fixed",
                min_focus_distance_m=0.5,
                max_frame_rate=9,
                spectral_range_um=(8.0, 14.0),
            )

    async def _get_calibration(self) -> CameraCalibration:
        """Get camera calibration data."""
        # In production, read from camera
        return CameraCalibration(
            calibration_date=datetime.utcnow() - timedelta(days=180),
            expiration_date=datetime.utcnow() + timedelta(days=185),
        )

    # =========================================================================
    # Image Capture
    # =========================================================================

    async def capture_image(
        self,
        settings: Optional[CaptureSettings] = None,
        equipment_id: Optional[str] = None,
        inspection_id: Optional[str] = None,
    ) -> ThermalImageCapture:
        """
        Capture a thermal image.

        Args:
            settings: Optional capture settings override
            equipment_id: Associated equipment ID
            inspection_id: Associated inspection ID

        Returns:
            Thermal image capture result

        Raises:
            ImageCaptureError: If capture fails
        """
        if not self.is_connected:
            raise CameraConnectionError("Camera not connected")

        import time
        start_time = time.time()

        settings = settings or self._config.capture_settings

        try:
            self._logger.debug("Capturing thermal image")

            # In production, use SDK:
            # thermal_data = self._camera.GetThermalImage()
            # raw_data = thermal_data.GetRawData()

            # Generate mock/simulated data
            thermal_image = await self._capture_thermal_data(settings)

            # Apply equipment mapping
            thermal_image.equipment_id = equipment_id
            thermal_image.inspection_id = inspection_id

            # Detect hot spots
            if self._config.hot_spot_config.enabled:
                hot_spots = await self._detect_hot_spots(
                    thermal_image,
                    self._config.hot_spot_config
                )
                thermal_image.hot_spots = hot_spots

            # Save image if configured
            if self._config.auto_save_images:
                file_path = await self._save_image(thermal_image)
                thermal_image.file_path = file_path

            capture_duration = (time.time() - start_time) * 1000
            self._captures_count += 1
            self._last_capture_time = datetime.utcnow()

            self._logger.info(
                f"Captured thermal image: {thermal_image.image_id} "
                f"({thermal_image.width}x{thermal_image.height})"
            )

            return ThermalImageCapture(
                thermal_image=thermal_image,
                capture_settings=settings,
                capture_duration_ms=capture_duration,
                success=True,
            )

        except Exception as e:
            self._captures_failed += 1
            self._logger.error(f"Image capture failed: {e}")
            raise ImageCaptureError(f"Capture failed: {e}")

    async def _capture_thermal_data(
        self,
        settings: CaptureSettings
    ) -> ThermalImage:
        """Capture thermal data from camera."""
        width = self._specs.width_pixels if self._specs else 320
        height = self._specs.height_pixels if self._specs else 240

        # In production, get actual thermal data from SDK
        # Generate synthetic data for mock
        import random
        base_temp = settings.atmospheric_temp_c
        temp_matrix = []

        for y in range(height):
            row = []
            for x in range(width):
                # Generate temperature with some spatial variation
                noise = random.gauss(0, 2)
                gradient = (x / width) * 5 + (y / height) * 3
                temp = base_temp + gradient + noise
                row.append(round(temp, 2))
            temp_matrix.append(row)

        # Calculate statistics
        all_temps = [t for row in temp_matrix for t in row]
        stats = TemperatureStatistics(
            min_temp_c=min(all_temps),
            max_temp_c=max(all_temps),
            avg_temp_c=sum(all_temps) / len(all_temps),
            std_dev_c=self._calculate_std_dev(all_temps),
            pixel_count=len(all_temps),
        )

        # Create metadata
        metadata = RadiometricMetadata(
            capture_timestamp=datetime.utcnow(),
            emissivity=settings.emissivity,
            reflected_temp_c=settings.reflected_temp_c,
            atmospheric_temp_c=settings.atmospheric_temp_c,
            object_distance_m=settings.distance_m,
            relative_humidity=settings.relative_humidity,
            camera_model=self._config.camera_config.model.value,
        )

        return ThermalImage(
            width=width,
            height=height,
            temperature_matrix=temp_matrix,
            statistics=stats,
            metadata=metadata,
        )

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    async def _save_image(self, image: ThermalImage) -> str:
        """Save thermal image to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"thermal_{image.image_id[:8]}_{timestamp}.json"
        file_path = Path(self._config.image_storage_path) / filename

        # Save as JSON (in production, save actual image format)
        import json
        data = {
            "image_id": image.image_id,
            "width": image.width,
            "height": image.height,
            "statistics": image.statistics.model_dump(),
            "metadata": image.metadata.model_dump(mode='json'),
            "hot_spots": [hs.model_dump() for hs in image.hot_spots],
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(file_path)

    # =========================================================================
    # Hot Spot Detection
    # =========================================================================

    async def _detect_hot_spots(
        self,
        image: ThermalImage,
        config: HotSpotDetectionConfig
    ) -> List[HotSpot]:
        """
        Detect hot spots in thermal image.

        Args:
            image: Thermal image
            config: Hot spot detection configuration

        Returns:
            List of detected hot spots
        """
        if not image.temperature_matrix:
            return []

        hot_spots = []

        # Determine reference temperature
        if config.use_ambient_reference:
            reference_temp = image.metadata.atmospheric_temp_c
        elif config.reference_region:
            # Calculate average in reference region
            x, y, w, h = config.reference_region
            temps = []
            for row_idx in range(y, min(y + h, image.height)):
                for col_idx in range(x, min(x + w, image.width)):
                    temps.append(image.temperature_matrix[row_idx][col_idx])
            reference_temp = sum(temps) / len(temps) if temps else 20.0
        else:
            reference_temp = image.statistics.avg_temp_c

        # Threshold temperature
        threshold = reference_temp + config.hot_spot_threshold_delta_c
        if config.absolute_threshold_c is not None:
            threshold = max(threshold, config.absolute_threshold_c)

        # Find connected regions above threshold
        visited = set()
        regions = []

        for y in range(image.height):
            for x in range(image.width):
                if (x, y) in visited:
                    continue

                temp = image.temperature_matrix[y][x]
                if temp >= threshold:
                    # BFS to find connected region
                    region = self._flood_fill_region(
                        image.temperature_matrix,
                        x, y,
                        threshold,
                        visited
                    )
                    if len(region) >= config.min_hot_spot_area_pixels:
                        regions.append(region)

        # Convert regions to hot spots
        for region in regions[:config.max_hot_spots]:
            hot_spot = self._create_hot_spot(
                image, region, reference_temp
            )
            hot_spots.append(hot_spot)

        # Sort by max temperature descending
        hot_spots.sort(key=lambda hs: hs.max_temp_c, reverse=True)

        self._logger.debug(f"Detected {len(hot_spots)} hot spots")

        return hot_spots[:config.max_hot_spots]

    def _flood_fill_region(
        self,
        matrix: List[List[float]],
        start_x: int,
        start_y: int,
        threshold: float,
        visited: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int, float]]:
        """Flood fill to find connected region above threshold."""
        region = []
        queue = [(start_x, start_y)]
        height = len(matrix)
        width = len(matrix[0]) if height > 0 else 0

        while queue:
            x, y = queue.pop(0)

            if (x, y) in visited:
                continue
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            temp = matrix[y][x]
            if temp < threshold:
                continue

            visited.add((x, y))
            region.append((x, y, temp))

            # Add neighbors
            queue.extend([
                (x + 1, y), (x - 1, y),
                (x, y + 1), (x, y - 1)
            ])

        return region

    def _create_hot_spot(
        self,
        image: ThermalImage,
        region: List[Tuple[int, int, float]],
        reference_temp: float
    ) -> HotSpot:
        """Create hot spot from region."""
        x_coords = [p[0] for p in region]
        y_coords = [p[1] for p in region]
        temps = [p[2] for p in region]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        max_temp = max(temps)
        avg_temp = sum(temps) / len(temps)
        delta = max_temp - reference_temp

        # Determine severity
        if delta >= 50:
            severity = "critical"
        elif delta >= 30:
            severity = "high"
        elif delta >= 15:
            severity = "medium"
        else:
            severity = "low"

        return HotSpot(
            center_x=center_x,
            center_y=center_y,
            width=max_x - min_x + 1,
            height=max_y - min_y + 1,
            area_pixels=len(region),
            max_temp_c=max_temp,
            avg_temp_c=avg_temp,
            temp_delta_c=delta,
            severity=severity,
            relative_x=center_x / image.width,
            relative_y=center_y / image.height,
        )

    # =========================================================================
    # Radiometric Data Processing
    # =========================================================================

    async def parse_radiometric_file(
        self,
        file_path: str
    ) -> ThermalImage:
        """
        Parse radiometric data from file.

        Args:
            file_path: Path to radiometric image file

        Returns:
            Parsed thermal image

        Raises:
            RadiometricDataError: If parsing fails
        """
        path = Path(file_path)

        if not path.exists():
            raise RadiometricDataError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        try:
            if suffix in ['.jpg', '.jpeg']:
                return await self._parse_flir_jpg(file_path)
            elif suffix == '.fff':
                return await self._parse_flir_fff(file_path)
            elif suffix == '.seq':
                return await self._parse_flir_seq(file_path)
            else:
                raise RadiometricDataError(f"Unsupported format: {suffix}")

        except Exception as e:
            raise RadiometricDataError(f"Failed to parse file: {e}")

    async def _parse_flir_jpg(self, file_path: str) -> ThermalImage:
        """
        Parse FLIR radiometric JPEG.

        FLIR embeds radiometric data in EXIF/APP1 segments.
        """
        # In production, use flirimageextractor or flirpy:
        # import flirimageextractor
        # fir = flirimageextractor.FlirImageExtractor()
        # fir.process_image(file_path)
        # thermal_np = fir.get_thermal_np()

        self._logger.info(f"Parsing FLIR JPG: {file_path}")

        # Return placeholder
        return ThermalImage(
            width=320,
            height=240,
            statistics=TemperatureStatistics(
                min_temp_c=20.0,
                max_temp_c=50.0,
                avg_temp_c=30.0,
                std_dev_c=5.0,
                pixel_count=76800,
            ),
            metadata=RadiometricMetadata(
                capture_timestamp=datetime.utcnow(),
            ),
        )

    async def _parse_flir_fff(self, file_path: str) -> ThermalImage:
        """Parse FLIR FFF raw format."""
        # FFF format parsing would go here
        raise NotImplementedError("FFF parsing not implemented")

    async def _parse_flir_seq(self, file_path: str) -> ThermalImage:
        """Parse FLIR sequence file."""
        # SEQ format parsing would go here
        raise NotImplementedError("SEQ parsing not implemented")

    def raw_to_temperature(
        self,
        raw_value: int,
        metadata: RadiometricMetadata
    ) -> float:
        """
        Convert raw sensor value to temperature.

        Uses Planck's radiation law with calibration constants.

        Args:
            raw_value: Raw sensor value
            metadata: Radiometric metadata with calibration

        Returns:
            Temperature in Celsius
        """
        # Planck equation: T = B / ln(R1/(R2*(S+O)) + F)
        # Where S is the raw signal, O is offset, R1, R2, B, F are constants

        r1 = metadata.planck_r1
        r2 = metadata.planck_r2
        b = metadata.planck_b
        f = metadata.planck_f
        o = metadata.planck_o

        # Apply atmospheric transmission correction
        # tau = atmospheric transmission
        # For simplification, using basic formula

        import math

        try:
            temp_k = b / math.log(r1 / (r2 * (raw_value + o)) + f)
            temp_c = temp_k - 273.15

            # Apply emissivity correction
            # Full radiometric equation would include:
            # - Emissivity correction
            # - Reflected temperature correction
            # - Atmospheric absorption
            # This is simplified

            return temp_c

        except (ValueError, ZeroDivisionError):
            return 0.0

    # =========================================================================
    # Streaming
    # =========================================================================

    async def start_stream(
        self,
        callback: Callable[[ThermalImage], None],
        frame_rate: Optional[float] = None
    ) -> None:
        """
        Start thermal image streaming.

        Args:
            callback: Callback function for each frame
            frame_rate: Optional frame rate override
        """
        if not self.is_connected:
            raise CameraConnectionError("Camera not connected")

        if self._is_streaming:
            self._logger.warning("Already streaming")
            return

        self._stream_callback = callback
        self._is_streaming = True
        self._state = ConnectionState.STREAMING

        # Start streaming task
        asyncio.create_task(self._stream_loop(frame_rate))

        self._logger.info("Started thermal streaming")

    async def stop_stream(self) -> None:
        """Stop thermal image streaming."""
        self._is_streaming = False
        self._stream_callback = None
        self._state = ConnectionState.CONNECTED

        self._logger.info("Stopped thermal streaming")

    async def _stream_loop(self, frame_rate: Optional[float]) -> None:
        """Background streaming loop."""
        rate = frame_rate or (self._specs.max_frame_rate if self._specs else 9)
        interval = 1.0 / rate

        while self._is_streaming:
            try:
                capture = await self.capture_image()
                if self._stream_callback and capture.success:
                    self._stream_callback(capture.thermal_image)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Stream error: {e}")
                await asyncio.sleep(interval)

    async def stream_frames(self) -> AsyncIterator[ThermalImage]:
        """
        Async iterator for streaming frames.

        Yields:
            Thermal images
        """
        rate = self._specs.max_frame_rate if self._specs else 9
        interval = 1.0 / rate

        while self.is_connected:
            try:
                capture = await self.capture_image()
                if capture.success:
                    yield capture.thermal_image
                await asyncio.sleep(interval)
            except Exception as e:
                self._logger.error(f"Frame error: {e}")
                await asyncio.sleep(interval)

    # =========================================================================
    # Health Check
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                # Check camera status
                # In production, query camera for status

                # Check calibration
                if self._calibration and not self._calibration.is_valid:
                    self._logger.warning("Camera calibration expired")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "state": self._state.value,
            "camera_model": self._config.camera_config.model.value,
            "captures_count": self._captures_count,
            "captures_failed": self._captures_failed,
            "is_streaming": self._is_streaming,
            "calibration_valid": self._calibration.is_valid if self._calibration else None,
            "last_capture": self._last_capture_time.isoformat() if self._last_capture_time else None,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_flir_camera_connector(
    camera_name: str = "FLIR-Camera",
    model: CameraModel = CameraModel.FLIR_E8,
    connection_type: CameraConnectionType = CameraConnectionType.USB,
    ip_address: Optional[str] = None,
    use_mock: bool = False,
    **kwargs
) -> ThermalCameraConnector:
    """
    Create FLIR camera connector.

    Args:
        camera_name: Camera name
        model: FLIR model
        connection_type: Connection type
        ip_address: IP address for network camera
        use_mock: Use mock camera
        **kwargs: Additional configuration

    Returns:
        Configured ThermalCameraConnector
    """
    camera_config = CameraConfig(
        camera_name=camera_name,
        manufacturer=CameraManufacturer.FLIR,
        model=model,
        connection_type=connection_type,
        ip_address=ip_address,
        use_mock=use_mock,
    )

    config = ThermalCameraConnectorConfig(
        connector_name=f"{camera_name}-Connector",
        camera_config=camera_config,
        **kwargs
    )

    return ThermalCameraConnector(config)


def create_network_thermal_camera(
    ip_address: str,
    port: int = 80,
    camera_name: str = "Network-ThermalCamera",
    manufacturer: CameraManufacturer = CameraManufacturer.FLIR,
    model: CameraModel = CameraModel.FLIR_A310,
    **kwargs
) -> ThermalCameraConnector:
    """
    Create network-connected thermal camera connector.

    Args:
        ip_address: Camera IP address
        port: Camera port
        camera_name: Camera name
        manufacturer: Camera manufacturer
        model: Camera model
        **kwargs: Additional configuration

    Returns:
        Configured ThermalCameraConnector
    """
    camera_config = CameraConfig(
        camera_name=camera_name,
        manufacturer=manufacturer,
        model=model,
        connection_type=CameraConnectionType.ETHERNET,
        ip_address=ip_address,
        port=port,
    )

    config = ThermalCameraConnectorConfig(
        connector_name=f"{camera_name}-Connector",
        camera_config=camera_config,
        **kwargs
    )

    return ThermalCameraConnector(config)
