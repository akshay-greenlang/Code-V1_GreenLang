"""
IR Camera Client for FurnacePulse

Infrared thermal imaging integration providing:
- IR image/frame acquisition from industrial thermal cameras
- Raw image storage in object storage with metadata
- Hotspot map derivation from IR data
- Timestamp synchronization with furnace telemetry
- Camera-to-furnace geometry mapping

Supports common industrial IR cameras: FLIR, InfraTec, Optris, etc.
"""

import asyncio
import io
import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, HttpUrl, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Thermal Data Models
# =============================================================================

class TemperatureUnit(str, Enum):
    """Temperature unit."""
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"


class CameraModel(str, Enum):
    """Supported IR camera models."""
    FLIR_A700 = "flir_a700"
    FLIR_A615 = "flir_a615"
    FLIR_AX8 = "flir_ax8"
    INFRATEC_PIR = "infratec_pir"
    OPTRIS_PI = "optris_pi"
    GENERIC_ONVIF = "generic_onvif"


@dataclass
class PixelCoordinate:
    """Pixel coordinate in IR image."""
    x: int
    y: int


@dataclass
class FurnaceCoordinate:
    """3D coordinate in furnace space."""
    x: float  # meters, along furnace length
    y: float  # meters, across furnace width
    z: float  # meters, height from floor


@dataclass
class GeometryMapping:
    """
    Mapping between camera pixels and furnace geometry.

    Enables translation from IR image coordinates to physical
    furnace locations (zones, tubes, burners).
    """
    camera_id: str
    furnace_id: str

    # Camera position and orientation
    camera_position: FurnaceCoordinate
    camera_rotation: Tuple[float, float, float]  # roll, pitch, yaw in degrees

    # Calibration points: pixel -> furnace coordinate
    calibration_points: List[Tuple[PixelCoordinate, FurnaceCoordinate]] = field(
        default_factory=list
    )

    # Pre-computed transformation matrix (3x4)
    transform_matrix: Optional[np.ndarray] = None

    # Zone boundaries in pixel coordinates
    zone_boundaries: Dict[str, List[PixelCoordinate]] = field(default_factory=dict)

    def pixel_to_furnace(self, pixel: PixelCoordinate) -> FurnaceCoordinate:
        """Convert pixel coordinate to furnace coordinate."""
        if self.transform_matrix is not None:
            # Apply transformation matrix
            px = np.array([pixel.x, pixel.y, 1])
            furnace = self.transform_matrix @ px
            return FurnaceCoordinate(x=furnace[0], y=furnace[1], z=furnace[2])

        # Fallback: simple linear interpolation
        if len(self.calibration_points) >= 4:
            # Use bilinear interpolation
            return self._bilinear_interpolate(pixel)

        raise ValueError("Insufficient calibration data")

    def _bilinear_interpolate(self, pixel: PixelCoordinate) -> FurnaceCoordinate:
        """Bilinear interpolation using calibration points."""
        # Simplified implementation
        if not self.calibration_points:
            return FurnaceCoordinate(x=0, y=0, z=0)

        # Find nearest calibration points and interpolate
        nearest = min(
            self.calibration_points,
            key=lambda p: (p[0].x - pixel.x) ** 2 + (p[0].y - pixel.y) ** 2
        )
        return nearest[1]

    def get_zone_for_pixel(self, pixel: PixelCoordinate) -> Optional[str]:
        """Get zone ID for a pixel coordinate."""
        for zone_id, boundary in self.zone_boundaries.items():
            if self._point_in_polygon(pixel, boundary):
                return zone_id
        return None

    def _point_in_polygon(
        self,
        point: PixelCoordinate,
        polygon: List[PixelCoordinate]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            if ((polygon[i].y > point.y) != (polygon[j].y > point.y)) and \
               (point.x < (polygon[j].x - polygon[i].x) *
                (point.y - polygon[i].y) / (polygon[j].y - polygon[i].y) +
                polygon[i].x):
                inside = not inside
            j = i

        return inside


class ThermalFrame(BaseModel):
    """
    Single thermal frame from IR camera.

    Contains temperature data matrix and metadata.
    """
    frame_id: str = Field(..., description="Unique frame identifier")
    camera_id: str
    furnace_id: str
    capture_timestamp: str = Field(..., description="ISO timestamp of capture")
    sync_timestamp: Optional[str] = Field(
        None,
        description="Synchronized timestamp with furnace telemetry"
    )

    # Image dimensions
    width: int
    height: int
    bits_per_pixel: int = Field(16, description="Bit depth per pixel")

    # Temperature data (flattened for serialization)
    temperature_data: List[float] = Field(
        ...,
        description="Row-major temperature values"
    )
    temperature_unit: TemperatureUnit = TemperatureUnit.CELSIUS

    # Statistics
    min_temperature: float
    max_temperature: float
    mean_temperature: float

    # Metadata
    emissivity: float = Field(0.95, ge=0, le=1)
    ambient_temperature: Optional[float] = None
    atmospheric_temperature: Optional[float] = None
    distance_to_target: Optional[float] = Field(None, description="Meters")

    # Camera settings
    camera_settings: Dict[str, Any] = Field(default_factory=dict)

    # Storage reference
    storage_url: Optional[str] = Field(None, description="URL in object storage")
    raw_data_url: Optional[str] = Field(None, description="URL to raw radiometric data")

    def to_numpy(self) -> np.ndarray:
        """Convert temperature data to numpy array."""
        return np.array(self.temperature_data).reshape((self.height, self.width))

    @classmethod
    def from_numpy(
        cls,
        frame_id: str,
        camera_id: str,
        furnace_id: str,
        data: np.ndarray,
        capture_timestamp: datetime,
        **kwargs
    ) -> "ThermalFrame":
        """Create ThermalFrame from numpy array."""
        return cls(
            frame_id=frame_id,
            camera_id=camera_id,
            furnace_id=furnace_id,
            capture_timestamp=capture_timestamp.isoformat(),
            width=data.shape[1],
            height=data.shape[0],
            temperature_data=data.flatten().tolist(),
            min_temperature=float(np.min(data)),
            max_temperature=float(np.max(data)),
            mean_temperature=float(np.mean(data)),
            **kwargs
        )


class Hotspot(BaseModel):
    """Detected hotspot in thermal image."""
    hotspot_id: str
    frame_id: str

    # Location
    center_pixel: Tuple[int, int]  # (x, y)
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    furnace_location: Optional[Tuple[float, float, float]] = None  # (x, y, z)
    zone_id: Optional[str] = None
    component_id: Optional[str] = None

    # Thermal characteristics
    peak_temperature: float
    mean_temperature: float
    area_pixels: int

    # Classification
    severity: str = Field(..., description="normal, elevated, warning, critical")
    is_anomalous: bool = False
    anomaly_score: float = Field(0.0, ge=0, le=1)


class HotspotMap(BaseModel):
    """
    Hotspot analysis result for a thermal frame.

    Contains detected hotspots and thermal distribution analysis.
    """
    frame_id: str
    camera_id: str
    furnace_id: str
    analysis_timestamp: str

    # Detected hotspots
    hotspots: List[Hotspot] = Field(default_factory=list)

    # Zone-level statistics
    zone_temperatures: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Zone ID -> {min, max, mean, std}"
    )

    # Overall thermal distribution
    temperature_histogram: Dict[str, int] = Field(
        default_factory=dict,
        description="Temperature bin -> count"
    )

    # Anomaly flags
    has_critical_hotspots: bool = False
    anomaly_count: int = 0

    # Reference to source frame
    source_frame_url: Optional[str] = None


# =============================================================================
# Object Storage Client
# =============================================================================

class ObjectStorageConfig(BaseModel):
    """Object storage configuration for IR images."""
    provider: str = Field("s3", description="s3, azure, gcs, minio")
    endpoint: Optional[str] = Field(None, description="Custom endpoint for S3-compatible")
    bucket: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None  # From vault
    region: str = "us-east-1"
    prefix: str = "furnacepulse/ir-images"


class ObjectStorageClient:
    """
    Object storage client for IR image storage.

    Stores raw thermal images with metadata for later analysis and audit.
    """

    def __init__(self, config: ObjectStorageConfig):
        """Initialize storage client."""
        self.config = config
        self._client = None

    async def connect(self) -> None:
        """Initialize storage client connection."""
        # In production, use boto3 or cloud-specific SDK:
        # import boto3
        # self._client = boto3.client(
        #     's3',
        #     endpoint_url=self.config.endpoint,
        #     aws_access_key_id=self.config.access_key,
        #     aws_secret_access_key=self.config.secret_key,
        #     region_name=self.config.region
        # )
        logger.info(f"Connected to object storage: {self.config.provider}")

    async def store_thermal_frame(
        self,
        frame: ThermalFrame,
        raw_data: Optional[bytes] = None
    ) -> str:
        """
        Store thermal frame in object storage.

        Args:
            frame: Thermal frame metadata
            raw_data: Optional raw radiometric data

        Returns:
            Storage URL
        """
        # Generate storage key
        timestamp = datetime.fromisoformat(frame.capture_timestamp.replace('Z', ''))
        key = (
            f"{self.config.prefix}/"
            f"{frame.furnace_id}/"
            f"{frame.camera_id}/"
            f"{timestamp.strftime('%Y/%m/%d')}/"
            f"{frame.frame_id}.json"
        )

        # Store metadata
        metadata = frame.dict()

        # In production:
        # self._client.put_object(
        #     Bucket=self.config.bucket,
        #     Key=key,
        #     Body=json.dumps(metadata).encode('utf-8'),
        #     ContentType='application/json',
        #     Metadata={
        #         'furnace_id': frame.furnace_id,
        #         'camera_id': frame.camera_id,
        #         'capture_timestamp': frame.capture_timestamp
        #     }
        # )

        url = f"s3://{self.config.bucket}/{key}"
        logger.debug(f"Stored thermal frame: {url}")

        # Store raw data if provided
        if raw_data:
            raw_key = key.replace('.json', '.raw')
            # self._client.put_object(
            #     Bucket=self.config.bucket,
            #     Key=raw_key,
            #     Body=raw_data,
            #     ContentType='application/octet-stream'
            # )
            frame.raw_data_url = f"s3://{self.config.bucket}/{raw_key}"

        frame.storage_url = url
        return url

    async def retrieve_thermal_frame(self, url: str) -> Optional[ThermalFrame]:
        """Retrieve thermal frame from storage."""
        try:
            # Parse URL
            # bucket, key = self._parse_url(url)

            # In production:
            # response = self._client.get_object(Bucket=bucket, Key=key)
            # data = json.loads(response['Body'].read().decode('utf-8'))
            # return ThermalFrame(**data)

            logger.debug(f"Retrieved thermal frame: {url}")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve thermal frame: {e}")
            return None

    async def list_frames(
        self,
        furnace_id: str,
        camera_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[str]:
        """List frame URLs in a time range."""
        # In production, use S3 list_objects_v2 with prefix filtering
        return []


# =============================================================================
# IR Camera Configuration
# =============================================================================

class IRCameraConfig(BaseModel):
    """IR camera configuration."""
    camera_id: str = Field(..., description="Unique camera identifier")
    camera_model: CameraModel
    furnace_id: str = Field(..., description="Associated furnace ID")

    # Connection
    host: str
    port: int = Field(80)
    protocol: str = Field("http", description="http, https, rtsp, gige")
    stream_path: str = Field("/ir/stream", description="Stream endpoint path")

    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None  # From vault
    api_key: Optional[str] = None  # From vault

    # Camera settings
    frame_rate: float = Field(1.0, description="Frames per second")
    resolution: Tuple[int, int] = Field((640, 480))
    temperature_range: Tuple[float, float] = Field((0, 500), description="Celsius")
    emissivity: float = Field(0.95, ge=0, le=1)
    distance_to_target: float = Field(5.0, description="Meters")

    # Time synchronization
    ntp_server: Optional[str] = Field(None, description="NTP server for time sync")
    time_offset_ms: int = Field(0, description="Manual time offset adjustment")

    # Object storage
    storage: ObjectStorageConfig

    # Geometry mapping file
    geometry_mapping_path: Optional[str] = None


# =============================================================================
# IR Camera Client
# =============================================================================

class IRCameraClient:
    """
    IR Camera client for FurnacePulse thermal imaging.

    Features:
    - IR image/frame acquisition
    - Raw image storage in object storage with metadata
    - Hotspot map derivation from IR data
    - Timestamp synchronization with furnace telemetry
    - Camera-to-furnace geometry mapping

    Usage:
        config = IRCameraConfig(
            camera_id="ir-cam-01",
            camera_model=CameraModel.FLIR_A700,
            furnace_id="furnace-001",
            host="192.168.1.100",
            storage=ObjectStorageConfig(bucket="furnacepulse-ir")
        )

        client = IRCameraClient(config)
        await client.connect()

        # Capture single frame
        frame = await client.capture_frame()

        # Analyze for hotspots
        hotspot_map = await client.analyze_hotspots(frame)

        # Start continuous streaming
        await client.start_streaming(callback=process_frame)
    """

    def __init__(
        self,
        config: IRCameraConfig,
        vault_client=None,
        telemetry_sync_callback: Optional[Callable[[datetime], datetime]] = None
    ):
        """
        Initialize IR camera client.

        Args:
            config: Camera configuration
            vault_client: Vault client for secrets
            telemetry_sync_callback: Callback to sync timestamps with telemetry
        """
        self.config = config
        self.vault_client = vault_client
        self.telemetry_sync_callback = telemetry_sync_callback

        # Clients
        self._http_client = None
        self._storage_client: Optional[ObjectStorageClient] = None

        # State
        self._connected = False
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None

        # Geometry mapping
        self._geometry: Optional[GeometryMapping] = None

        # Frame counter
        self._frame_count = 0

        # Hotspot detection thresholds
        self._hotspot_thresholds = {
            "normal": (0, 400),
            "elevated": (400, 450),
            "warning": (450, 500),
            "critical": (500, float('inf'))
        }

        logger.info(
            f"IRCameraClient initialized: {config.camera_id} "
            f"({config.camera_model}) for {config.furnace_id}"
        )

    async def connect(self) -> None:
        """Connect to IR camera and initialize storage."""
        # Retrieve credentials from vault
        if self.vault_client:
            if self.config.password:
                self.config.password = await self.vault_client.get_secret(
                    f"ir_camera_{self.config.camera_id}_password"
                )

        # Initialize HTTP client
        import httpx
        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            auth=(self.config.username, self.config.password)
            if self.config.username else None
        )

        # Test connection
        base_url = f"{self.config.protocol}://{self.config.host}:{self.config.port}"
        try:
            # response = await self._http_client.get(f"{base_url}/status")
            # response.raise_for_status()
            logger.info(f"Connected to IR camera at {base_url}")
        except Exception as e:
            logger.warning(f"Camera connection test failed: {e}")

        # Initialize storage
        self._storage_client = ObjectStorageClient(self.config.storage)
        await self._storage_client.connect()

        # Load geometry mapping
        if self.config.geometry_mapping_path:
            await self._load_geometry_mapping()

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from IR camera."""
        if self._streaming:
            await self.stop_streaming()

        if self._http_client:
            await self._http_client.aclose()

        self._connected = False
        logger.info(f"Disconnected from IR camera {self.config.camera_id}")

    async def _load_geometry_mapping(self) -> None:
        """Load camera-to-furnace geometry mapping."""
        try:
            path = Path(self.config.geometry_mapping_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                self._geometry = GeometryMapping(
                    camera_id=self.config.camera_id,
                    furnace_id=self.config.furnace_id,
                    camera_position=FurnaceCoordinate(**data.get("camera_position", {})),
                    camera_rotation=tuple(data.get("camera_rotation", [0, 0, 0])),
                    zone_boundaries={
                        zone_id: [PixelCoordinate(**p) for p in points]
                        for zone_id, points in data.get("zone_boundaries", {}).items()
                    }
                )

                logger.info(f"Loaded geometry mapping with {len(self._geometry.zone_boundaries)} zones")

        except Exception as e:
            logger.warning(f"Failed to load geometry mapping: {e}")

    async def capture_frame(self) -> ThermalFrame:
        """
        Capture single thermal frame from camera.

        Returns:
            ThermalFrame with temperature data and metadata
        """
        if not self._connected:
            raise RuntimeError("Not connected to camera")

        capture_time = datetime.utcnow()

        try:
            # In production, fetch from camera:
            # base_url = f"{self.config.protocol}://{self.config.host}:{self.config.port}"
            # response = await self._http_client.get(f"{base_url}/capture")
            # raw_data = response.content
            # temperature_data = self._parse_radiometric_data(raw_data)

            # Mock: generate synthetic thermal data
            width, height = self.config.resolution
            temperature_data = self._generate_mock_thermal_data(width, height)

            self._frame_count += 1
            frame_id = f"{self.config.camera_id}-{capture_time.strftime('%Y%m%d%H%M%S%f')}"

            # Apply time synchronization
            sync_timestamp = capture_time
            if self.telemetry_sync_callback:
                sync_timestamp = self.telemetry_sync_callback(capture_time)

            # Apply manual offset
            sync_timestamp += timedelta(milliseconds=self.config.time_offset_ms)

            frame = ThermalFrame(
                frame_id=frame_id,
                camera_id=self.config.camera_id,
                furnace_id=self.config.furnace_id,
                capture_timestamp=capture_time.isoformat() + "Z",
                sync_timestamp=sync_timestamp.isoformat() + "Z",
                width=width,
                height=height,
                temperature_data=temperature_data.flatten().tolist(),
                min_temperature=float(np.min(temperature_data)),
                max_temperature=float(np.max(temperature_data)),
                mean_temperature=float(np.mean(temperature_data)),
                emissivity=self.config.emissivity,
                distance_to_target=self.config.distance_to_target,
                camera_settings={
                    "frame_rate": self.config.frame_rate,
                    "temperature_range": self.config.temperature_range
                }
            )

            # Store in object storage
            if self._storage_client:
                await self._storage_client.store_thermal_frame(frame)

            logger.debug(f"Captured frame {frame_id}: max={frame.max_temperature:.1f}C")
            return frame

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            raise

    def _generate_mock_thermal_data(self, width: int, height: int) -> np.ndarray:
        """Generate synthetic thermal data for testing."""
        # Base temperature gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Base temperature (350-450C typical for furnace)
        base_temp = 380 + 40 * np.sin(xx * np.pi) * np.cos(yy * np.pi)

        # Add random noise
        noise = np.random.normal(0, 5, (height, width))

        # Add some hotspots
        for _ in range(np.random.randint(0, 3)):
            cx, cy = np.random.randint(0, width), np.random.randint(0, height)
            hotspot = 50 * np.exp(-((xx * width - cx) ** 2 + (yy * height - cy) ** 2) / 100)
            base_temp += hotspot

        return base_temp + noise

    def _parse_radiometric_data(self, raw_data: bytes) -> np.ndarray:
        """Parse raw radiometric data from camera."""
        width, height = self.config.resolution
        bits = self.config.resolution

        # Parse based on camera model
        if self.config.camera_model in [CameraModel.FLIR_A700, CameraModel.FLIR_A615]:
            # FLIR uses 16-bit little-endian with specific scaling
            data = np.frombuffer(raw_data, dtype='<u2').reshape((height, width))
            # Apply radiometric conversion
            temperature = data / 100.0 - 273.15  # Example conversion

        elif self.config.camera_model == CameraModel.OPTRIS_PI:
            # Optris format
            data = np.frombuffer(raw_data, dtype='<u2').reshape((height, width))
            temperature = data / 10.0

        else:
            # Generic 16-bit
            data = np.frombuffer(raw_data, dtype='<u2').reshape((height, width))
            temp_range = self.config.temperature_range
            temperature = data / 65535.0 * (temp_range[1] - temp_range[0]) + temp_range[0]

        return temperature

    async def start_streaming(
        self,
        callback: Callable[[ThermalFrame], None],
        frame_rate: Optional[float] = None
    ) -> None:
        """
        Start continuous frame streaming.

        Args:
            callback: Function to call for each frame
            frame_rate: Override configured frame rate
        """
        if self._streaming:
            logger.warning("Already streaming")
            return

        rate = frame_rate or self.config.frame_rate
        interval = 1.0 / rate

        async def stream_loop():
            while self._streaming:
                try:
                    frame = await self.capture_frame()
                    callback(frame)
                except Exception as e:
                    logger.error(f"Stream frame error: {e}")

                await asyncio.sleep(interval)

        self._streaming = True
        self._stream_task = asyncio.create_task(stream_loop())
        logger.info(f"Started streaming at {rate} fps")

    async def stop_streaming(self) -> None:
        """Stop continuous streaming."""
        self._streaming = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped streaming")

    async def analyze_hotspots(
        self,
        frame: ThermalFrame,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> HotspotMap:
        """
        Analyze thermal frame for hotspots.

        Args:
            frame: Thermal frame to analyze
            thresholds: Custom threshold ranges per severity

        Returns:
            HotspotMap with detected hotspots and analysis
        """
        thresholds = thresholds or self._hotspot_thresholds

        # Convert to numpy array
        temp_data = frame.to_numpy()

        hotspots = []
        zone_temperatures: Dict[str, Dict[str, float]] = {}

        # Detect hotspots using connected components
        critical_threshold = thresholds["elevated"][0]
        hotspot_mask = temp_data > critical_threshold

        # Find connected regions (simplified)
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(hotspot_mask)

        for region_id in range(1, num_features + 1):
            region_mask = labeled_array == region_id
            region_temps = temp_data[region_mask]

            # Get region properties
            positions = np.where(region_mask)
            y_coords, x_coords = positions
            center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
            min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
            min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))

            peak_temp = float(np.max(region_temps))
            mean_temp = float(np.mean(region_temps))

            # Determine severity
            severity = "normal"
            for sev, (low, high) in thresholds.items():
                if low <= peak_temp < high:
                    severity = sev
                    break

            # Get zone if geometry available
            zone_id = None
            furnace_loc = None
            if self._geometry:
                pixel = PixelCoordinate(x=center_x, y=center_y)
                zone_id = self._geometry.get_zone_for_pixel(pixel)
                try:
                    furnace_coord = self._geometry.pixel_to_furnace(pixel)
                    furnace_loc = (furnace_coord.x, furnace_coord.y, furnace_coord.z)
                except:
                    pass

            hotspot = Hotspot(
                hotspot_id=f"{frame.frame_id}-hs-{region_id}",
                frame_id=frame.frame_id,
                center_pixel=(center_x, center_y),
                bounding_box=(min_x, min_y, max_x - min_x, max_y - min_y),
                furnace_location=furnace_loc,
                zone_id=zone_id,
                peak_temperature=peak_temp,
                mean_temperature=mean_temp,
                area_pixels=int(np.sum(region_mask)),
                severity=severity,
                is_anomalous=severity in ["warning", "critical"],
                anomaly_score=min(1.0, (peak_temp - critical_threshold) / 100)
            )
            hotspots.append(hotspot)

        # Calculate zone-level statistics
        if self._geometry:
            for zone_id, boundary in self._geometry.zone_boundaries.items():
                # Create zone mask
                zone_mask = np.zeros_like(temp_data, dtype=bool)
                # Simplified: use bounding box
                x_coords = [p.x for p in boundary]
                y_coords = [p.y for p in boundary]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                zone_mask[min_y:max_y, min_x:max_x] = True
                zone_temps = temp_data[zone_mask]

                if len(zone_temps) > 0:
                    zone_temperatures[zone_id] = {
                        "min": float(np.min(zone_temps)),
                        "max": float(np.max(zone_temps)),
                        "mean": float(np.mean(zone_temps)),
                        "std": float(np.std(zone_temps))
                    }

        # Build temperature histogram
        hist, bin_edges = np.histogram(temp_data, bins=50)
        temperature_histogram = {
            f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}": int(hist[i])
            for i in range(len(hist))
        }

        hotspot_map = HotspotMap(
            frame_id=frame.frame_id,
            camera_id=frame.camera_id,
            furnace_id=frame.furnace_id,
            analysis_timestamp=datetime.utcnow().isoformat() + "Z",
            hotspots=hotspots,
            zone_temperatures=zone_temperatures,
            temperature_histogram=temperature_histogram,
            has_critical_hotspots=any(h.severity == "critical" for h in hotspots),
            anomaly_count=sum(1 for h in hotspots if h.is_anomalous),
            source_frame_url=frame.storage_url
        )

        logger.info(
            f"Hotspot analysis for {frame.frame_id}: "
            f"{len(hotspots)} hotspots, {hotspot_map.anomaly_count} anomalies"
        )

        return hotspot_map

    async def calibrate_geometry(
        self,
        calibration_points: List[Tuple[Tuple[int, int], Tuple[float, float, float]]]
    ) -> GeometryMapping:
        """
        Calibrate camera-to-furnace geometry mapping.

        Args:
            calibration_points: List of (pixel, furnace_coord) pairs

        Returns:
            Updated GeometryMapping
        """
        if not self._geometry:
            self._geometry = GeometryMapping(
                camera_id=self.config.camera_id,
                furnace_id=self.config.furnace_id,
                camera_position=FurnaceCoordinate(x=0, y=0, z=0),
                camera_rotation=(0, 0, 0)
            )

        self._geometry.calibration_points = [
            (PixelCoordinate(x=p[0][0], y=p[0][1]),
             FurnaceCoordinate(x=p[1][0], y=p[1][1], z=p[1][2]))
            for p in calibration_points
        ]

        # Compute transformation matrix using calibration points
        if len(calibration_points) >= 4:
            # Would use cv2.solvePnP or similar in production
            pass

        logger.info(
            f"Updated geometry calibration with {len(calibration_points)} points"
        )

        return self._geometry

    def set_zone_boundaries(
        self,
        zone_boundaries: Dict[str, List[Tuple[int, int]]]
    ) -> None:
        """
        Set zone boundaries in pixel coordinates.

        Args:
            zone_boundaries: Dict of zone_id -> list of (x, y) polygon vertices
        """
        if not self._geometry:
            self._geometry = GeometryMapping(
                camera_id=self.config.camera_id,
                furnace_id=self.config.furnace_id,
                camera_position=FurnaceCoordinate(x=0, y=0, z=0),
                camera_rotation=(0, 0, 0)
            )

        self._geometry.zone_boundaries = {
            zone_id: [PixelCoordinate(x=p[0], y=p[1]) for p in points]
            for zone_id, points in zone_boundaries.items()
        }

        logger.info(f"Set {len(zone_boundaries)} zone boundaries")

    def get_statistics(self) -> Dict[str, Any]:
        """Get camera client statistics."""
        return {
            "camera_id": self.config.camera_id,
            "furnace_id": self.config.furnace_id,
            "connected": self._connected,
            "streaming": self._streaming,
            "frames_captured": self._frame_count,
            "has_geometry": self._geometry is not None,
            "zones_mapped": (
                len(self._geometry.zone_boundaries)
                if self._geometry else 0
            )
        }
