"""
Landsat Client for USGS Earth Explorer API Integration.

Provides access to Landsat 8/9 OLI imagery as a fallback when
Sentinel-2 imagery is unavailable.

Landsat 8/9 OLI Bands:
- Band 2 (Blue): 452-512nm, 30m resolution
- Band 3 (Green): 533-590nm, 30m resolution
- Band 4 (Red): 636-673nm, 30m resolution
- Band 5 (NIR): 851-879nm, 30m resolution
- Band 6 (SWIR1): 1566-1651nm, 30m resolution
- Band 7 (SWIR2): 2107-2294nm, 30m resolution
- QA_PIXEL: Quality assessment band
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntFlag
from pathlib import Path
from typing import Any, Optional
import hashlib
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class LandsatQAFlags(IntFlag):
    """Landsat QA_PIXEL bit flags."""
    FILL = 1 << 0
    DILATED_CLOUD = 1 << 1
    CIRRUS = 1 << 2
    CLOUD = 1 << 3
    CLOUD_SHADOW = 1 << 4
    SNOW = 1 << 5
    CLEAR = 1 << 6
    WATER = 1 << 7


@dataclass
class BoundingBox:
    """Geographic bounding box in WGS84 coordinates."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if not (-180 <= self.min_lon <= 180):
            raise ValueError(f"min_lon must be between -180 and 180, got {self.min_lon}")
        if not (-180 <= self.max_lon <= 180):
            raise ValueError(f"max_lon must be between -180 and 180, got {self.max_lon}")
        if not (-90 <= self.min_lat <= 90):
            raise ValueError(f"min_lat must be between -90 and 90, got {self.min_lat}")
        if not (-90 <= self.max_lat <= 90):
            raise ValueError(f"max_lat must be between -90 and 90, got {self.max_lat}")
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be less than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be less than max_lat")

    def to_wkt(self) -> str:
        """Convert to WKT POLYGON string."""
        return (
            f"POLYGON(({self.min_lon} {self.min_lat}, "
            f"{self.max_lon} {self.min_lat}, "
            f"{self.max_lon} {self.max_lat}, "
            f"{self.min_lon} {self.max_lat}, "
            f"{self.min_lon} {self.min_lat}))"
        )


@dataclass
class LandsatBand:
    """Represents a single Landsat band."""
    name: str
    landsat_band: int
    sentinel2_equivalent: Optional[str]
    wavelength_nm: tuple[int, int]
    resolution_m: int = 30
    data: Optional[np.ndarray] = None

    @property
    def is_loaded(self) -> bool:
        """Check if band data is loaded."""
        return self.data is not None


@dataclass
class LandsatImage:
    """Container for Landsat imagery data."""
    scene_id: str
    acquisition_date: datetime
    cloud_cover_percentage: float
    bbox: BoundingBox
    satellite: str  # "LANDSAT_8" or "LANDSAT_9"
    bands: dict[str, LandsatBand] = field(default_factory=dict)
    qa_pixel: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_band(self, band_name: str) -> Optional[np.ndarray]:
        """Get band data by name (supports both Landsat and Sentinel-2 naming)."""
        # Direct match
        if band_name in self.bands:
            return self.bands[band_name].data

        # Try Sentinel-2 equivalent mapping
        for landsat_band in self.bands.values():
            if landsat_band.sentinel2_equivalent == band_name:
                return landsat_band.data

        return None

    def get_cloud_mask(self) -> Optional[np.ndarray]:
        """Generate cloud mask from QA_PIXEL band.

        Returns boolean array where True = valid (cloud-free) pixel.
        """
        if self.qa_pixel is None:
            return None

        # Check for cloud, cloud shadow, cirrus, and fill
        invalid_mask = (
            (self.qa_pixel & LandsatQAFlags.CLOUD) |
            (self.qa_pixel & LandsatQAFlags.CLOUD_SHADOW) |
            (self.qa_pixel & LandsatQAFlags.CIRRUS) |
            (self.qa_pixel & LandsatQAFlags.FILL)
        )

        return ~invalid_mask.astype(bool)

    def apply_cloud_mask(self, band_data: np.ndarray) -> np.ndarray:
        """Apply cloud mask to band data, setting masked pixels to NaN."""
        mask = self.get_cloud_mask()
        if mask is None:
            return band_data

        result = band_data.astype(float)
        result[~mask] = np.nan
        return result

    def to_sentinel2_bands(self) -> dict[str, np.ndarray]:
        """Convert Landsat bands to Sentinel-2 equivalent naming.

        Returns dict mapping Sentinel-2 band names to data arrays.
        This enables using the same analysis code for both sensors.
        """
        result = {}
        for band in self.bands.values():
            if band.sentinel2_equivalent and band.data is not None:
                result[band.sentinel2_equivalent] = band.data
        return result


@dataclass
class SearchResult:
    """Result from image search query."""
    scene_id: str
    acquisition_date: datetime
    cloud_cover_percentage: float
    bbox: BoundingBox
    satellite: str
    path: int
    row: int
    download_url: str
    file_size_mb: float


class LandsatClientError(Exception):
    """Base exception for Landsat client errors."""
    pass


class AuthenticationError(LandsatClientError):
    """Authentication failed."""
    pass


class QueryError(LandsatClientError):
    """Query failed."""
    pass


class DownloadError(LandsatClientError):
    """Download failed."""
    pass


class LandsatClient:
    """
    Client for accessing Landsat 8/9 imagery via USGS Earth Explorer API.

    Provides fallback imagery when Sentinel-2 is unavailable and
    band mapping to enable consistent analysis across both sensors.
    """

    USGS_API_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"
    USGS_AUTH_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/login"

    # Band mapping from Landsat to Sentinel-2 equivalents
    BAND_DEFINITIONS = {
        "B2": LandsatBand("B2", 2, "B2", (452, 512), 30),   # Blue
        "B3": LandsatBand("B3", 3, "B3", (533, 590), 30),   # Green
        "B4": LandsatBand("B4", 4, "B4", (636, 673), 30),   # Red
        "B5": LandsatBand("B5", 5, "B8", (851, 879), 30),   # NIR -> S2 B8
        "B6": LandsatBand("B6", 6, "B11", (1566, 1651), 30),  # SWIR1 -> S2 B11
        "B7": LandsatBand("B7", 7, "B12", (2107, 2294), 30),  # SWIR2 -> S2 B12
    }

    # Reverse mapping for Sentinel-2 band name requests
    SENTINEL2_TO_LANDSAT = {
        "B2": "B2",
        "B3": "B3",
        "B4": "B4",
        "B8": "B5",
        "B11": "B6",
        "B12": "B7",
    }

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_mock: bool = False,
    ):
        """
        Initialize Landsat client.

        Args:
            username: USGS Earth Explorer username
            password: USGS Earth Explorer password
            cache_dir: Directory for caching downloaded tiles
            use_mock: Use mock data instead of real API calls
        """
        self.username = username or os.getenv("USGS_USERNAME")
        self.password = password or os.getenv("USGS_PASSWORD")
        self.cache_dir = cache_dir or Path.home() / ".greenlang" / "satellite_cache" / "landsat"
        self.use_mock = use_mock
        self._api_key: Optional[str] = None

        if not self.use_mock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_bands(self, bands: list[str]) -> list[str]:
        """Resolve band names, converting Sentinel-2 names to Landsat names."""
        resolved = []
        for band in bands:
            if band in self.BAND_DEFINITIONS:
                resolved.append(band)
            elif band in self.SENTINEL2_TO_LANDSAT:
                resolved.append(self.SENTINEL2_TO_LANDSAT[band])
            else:
                raise ValueError(f"Unknown band: {band}")
        return resolved

    def authenticate(self) -> str:
        """
        Authenticate with USGS Earth Explorer API.

        Returns:
            API key string

        Raises:
            AuthenticationError: If authentication fails
        """
        if self.use_mock:
            self._api_key = "mock_api_key_67890"
            return self._api_key

        if self._api_key:
            return self._api_key

        if not self.username or not self.password:
            raise AuthenticationError(
                "Missing credentials. Set USGS_USERNAME and USGS_PASSWORD "
                "environment variables or pass them to the constructor."
            )

        logger.info("Authenticating with USGS Earth Explorer API...")

        # Stub: In real implementation, use requests to get API key
        # response = requests.post(
        #     self.USGS_AUTH_URL,
        #     json={"username": self.username, "password": self.password}
        # )

        raise AuthenticationError(
            "Real API authentication not implemented. Use use_mock=True for testing."
        )

    def search(
        self,
        bbox: BoundingBox,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float = 20.0,
        max_results: int = 10,
        satellites: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Search for Landsat images matching criteria.

        Args:
            bbox: Bounding box for search area
            start_date: Start of date range
            end_date: End of date range
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            max_results: Maximum number of results to return
            satellites: List of satellites to search ("LANDSAT_8", "LANDSAT_9")

        Returns:
            List of SearchResult objects

        Raises:
            QueryError: If search fails
        """
        if satellites is None:
            satellites = ["LANDSAT_8", "LANDSAT_9"]

        if self.use_mock:
            return self._mock_search(bbox, start_date, end_date, max_cloud_cover, max_results, satellites)

        self.authenticate()

        logger.info(
            f"Searching Landsat imagery: bbox={bbox}, "
            f"dates={start_date.date()} to {end_date.date()}, "
            f"satellites={satellites}"
        )

        raise QueryError(
            "Real API search not implemented. Use use_mock=True for testing."
        )

    def _mock_search(
        self,
        bbox: BoundingBox,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float,
        max_results: int,
        satellites: list[str],
    ) -> list[SearchResult]:
        """Generate mock search results for testing."""
        results = []

        # Generate deterministic mock results
        seed = int(bbox.min_lon * 100 + bbox.min_lat * 100 + 42)
        rng = np.random.default_rng(seed)

        # Calculate WRS-2 path/row (simplified approximation)
        center_lon = (bbox.min_lon + bbox.max_lon) / 2
        center_lat = (bbox.min_lat + bbox.max_lat) / 2
        path = int((180 - center_lon) / 360 * 233) % 233 + 1
        row = int((90 - center_lat) / 180 * 122) + 1

        current_date = start_date
        while current_date <= end_date and len(results) < max_results:
            # Landsat revisit time is 16 days (8 days with both L8 and L9)
            for satellite in satellites:
                cloud_cover = rng.uniform(0, 60)

                if cloud_cover <= max_cloud_cover and len(results) < max_results:
                    sat_code = "LC08" if satellite == "LANDSAT_8" else "LC09"
                    scene_id = f"{sat_code}_L2SP_{path:03d}{row:03d}_{current_date.strftime('%Y%m%d')}_02_T1"

                    results.append(SearchResult(
                        scene_id=scene_id,
                        acquisition_date=current_date,
                        cloud_cover_percentage=round(cloud_cover, 2),
                        bbox=bbox,
                        satellite=satellite,
                        path=path,
                        row=row,
                        download_url=f"https://earthexplorer.usgs.gov/download/{scene_id}",
                        file_size_mb=round(rng.uniform(800, 2000), 1),
                    ))

                current_date += timedelta(days=8)

        logger.info(f"Found {len(results)} Landsat images")
        return results

    def download_image(
        self,
        search_result: SearchResult,
        bands: Optional[list[str]] = None,
        apply_cloud_mask: bool = True,
    ) -> LandsatImage:
        """
        Download Landsat image with specified bands.

        Args:
            search_result: SearchResult from search() method
            bands: List of band names (Landsat or Sentinel-2 equivalent names)
            apply_cloud_mask: Whether to download QA_PIXEL band for cloud masking

        Returns:
            LandsatImage with downloaded band data

        Raises:
            DownloadError: If download fails
        """
        if bands is None:
            bands = list(self.BAND_DEFINITIONS.keys())
        else:
            bands = self._resolve_bands(bands)

        if self.use_mock:
            return self._mock_download(search_result, bands, apply_cloud_mask)

        self.authenticate()

        logger.info(f"Downloading image {search_result.scene_id}, bands: {bands}")

        raise DownloadError(
            "Real API download not implemented. Use use_mock=True for testing."
        )

    def _mock_download(
        self,
        search_result: SearchResult,
        bands: list[str],
        apply_cloud_mask: bool,
    ) -> LandsatImage:
        """Generate mock image data for testing."""
        seed = int(hashlib.md5(search_result.scene_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # Calculate image dimensions (30m resolution)
        bbox = search_result.bbox
        lat_km = (bbox.max_lat - bbox.min_lat) * 111.0
        lon_km = (bbox.max_lon - bbox.min_lon) * 111.0 * np.cos(np.radians((bbox.min_lat + bbox.max_lat) / 2))

        # 30m resolution = ~33 pixels per km
        height = max(10, int(lat_km * 33))
        width = max(10, int(lon_km * 33))

        # Limit size for mock data
        height = min(height, 500)
        width = min(width, 500)

        image = LandsatImage(
            scene_id=search_result.scene_id,
            acquisition_date=search_result.acquisition_date,
            cloud_cover_percentage=search_result.cloud_cover_percentage,
            bbox=bbox,
            satellite=search_result.satellite,
            metadata={
                "path": search_result.path,
                "row": search_result.row,
                "file_size_mb": search_result.file_size_mb,
            }
        )

        # Generate band data
        for band_name in bands:
            band_def = self.BAND_DEFINITIONS[band_name]

            # Create spatial patterns
            x = np.linspace(0, 4 * np.pi, width)
            y = np.linspace(0, 4 * np.pi, height)
            xx, yy = np.meshgrid(x, y)

            pattern = np.sin(xx * rng.uniform(0.5, 2)) * np.cos(yy * rng.uniform(0.5, 2))
            pattern = (pattern + 1) / 2
            noise = rng.normal(0, 0.1, (height, width))

            # Scale to Landsat L2 surface reflectance values (0-10000)
            if band_name == "B5":  # NIR
                base_value = 3500
                scale = 2500
            elif band_name == "B4":  # Red
                base_value = 800
                scale = 1200
            elif band_name == "B3":  # Green
                base_value = 1000
                scale = 1000
            elif band_name == "B2":  # Blue
                base_value = 900
                scale = 800
            elif band_name in ["B6", "B7"]:  # SWIR
                base_value = 1800
                scale = 1800
            else:
                base_value = 1500
                scale = 1500

            data = base_value + scale * pattern + scale * 0.2 * noise
            data = np.clip(data, 0, 10000).astype(np.uint16)

            image.bands[band_name] = LandsatBand(
                name=band_name,
                landsat_band=band_def.landsat_band,
                sentinel2_equivalent=band_def.sentinel2_equivalent,
                wavelength_nm=band_def.wavelength_nm,
                resolution_m=30,
                data=data,
            )

        # Generate QA_PIXEL band
        if apply_cloud_mask:
            qa_data = np.zeros((height, width), dtype=np.uint16)

            # Set CLEAR flag for most pixels
            qa_data[:] = LandsatQAFlags.CLEAR

            # Add cloud pixels
            cloud_fraction = search_result.cloud_cover_percentage / 100.0
            cloud_mask = rng.random((height, width)) < cloud_fraction
            qa_data[cloud_mask] = LandsatQAFlags.CLOUD

            # Add cloud shadows
            shadow_mask = rng.random((height, width)) < (cloud_fraction * 0.3)
            qa_data[shadow_mask & ~cloud_mask] = LandsatQAFlags.CLOUD_SHADOW

            # Add water
            water_mask = rng.random((height, width)) < 0.05
            qa_data[water_mask & ~cloud_mask & ~shadow_mask] = LandsatQAFlags.WATER | LandsatQAFlags.CLEAR

            image.qa_pixel = qa_data

        logger.info(f"Downloaded mock image: {image.scene_id}, shape: {height}x{width}")
        return image

    def get_time_series(
        self,
        bbox: BoundingBox,
        start_date: datetime,
        end_date: datetime,
        bands: Optional[list[str]] = None,
        max_cloud_cover: float = 20.0,
    ) -> list[LandsatImage]:
        """
        Get time series of images for an area.

        Args:
            bbox: Bounding box for area of interest
            start_date: Start of date range
            end_date: End of date range
            bands: Bands to download
            max_cloud_cover: Maximum cloud cover percentage

        Returns:
            List of LandsatImage objects sorted by date
        """
        results = self.search(bbox, start_date, end_date, max_cloud_cover)

        images = []
        for result in results:
            try:
                image = self.download_image(result, bands)
                images.append(image)
            except DownloadError as e:
                logger.warning(f"Failed to download {result.scene_id}: {e}")

        images.sort(key=lambda x: x.acquisition_date)
        return images


class HarmonizedSatelliteClient:
    """
    Unified client for accessing Sentinel-2 and Landsat imagery.

    Provides a consistent interface and automatic fallback to Landsat
    when Sentinel-2 imagery is unavailable.
    """

    def __init__(
        self,
        sentinel2_client: Optional["Sentinel2Client"] = None,
        landsat_client: Optional[LandsatClient] = None,
        prefer_sentinel2: bool = True,
    ):
        """
        Initialize harmonized client.

        Args:
            sentinel2_client: Sentinel2Client instance
            landsat_client: LandsatClient instance
            prefer_sentinel2: Prefer Sentinel-2 when both available
        """
        # Import here to avoid circular imports
        from greenlang.satellite.clients.sentinel2_client import Sentinel2Client

        self.sentinel2_client = sentinel2_client or Sentinel2Client(use_mock=True)
        self.landsat_client = landsat_client or LandsatClient(use_mock=True)
        self.prefer_sentinel2 = prefer_sentinel2

    def get_best_image(
        self,
        bbox: BoundingBox,
        target_date: datetime,
        max_cloud_cover: float = 20.0,
        date_tolerance_days: int = 30,
    ) -> tuple[Any, str]:
        """
        Get the best available image for a target date.

        Args:
            bbox: Bounding box for area
            target_date: Target acquisition date
            max_cloud_cover: Maximum cloud cover percentage
            date_tolerance_days: Days before/after target to search

        Returns:
            Tuple of (image, source) where source is "sentinel2" or "landsat"
        """
        start_date = target_date - timedelta(days=date_tolerance_days)
        end_date = target_date + timedelta(days=date_tolerance_days)

        # Import here to avoid circular imports
        from greenlang.satellite.clients.sentinel2_client import BoundingBox as S2BoundingBox

        s2_bbox = S2BoundingBox(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)

        if self.prefer_sentinel2:
            # Try Sentinel-2 first
            try:
                results = self.sentinel2_client.search(s2_bbox, start_date, end_date, max_cloud_cover, max_results=1)
                if results:
                    image = self.sentinel2_client.download_image(results[0])
                    return image, "sentinel2"
            except Exception as e:
                logger.warning(f"Sentinel-2 search failed: {e}")

        # Fall back to Landsat
        try:
            results = self.landsat_client.search(bbox, start_date, end_date, max_cloud_cover, max_results=1)
            if results:
                image = self.landsat_client.download_image(results[0])
                return image, "landsat"
        except Exception as e:
            logger.warning(f"Landsat search failed: {e}")

        raise QueryError(f"No imagery found for {target_date} within {date_tolerance_days} days")

    def get_harmonized_bands(
        self,
        image: Any,
        source: str,
    ) -> dict[str, np.ndarray]:
        """
        Get harmonized band data with consistent naming.

        Args:
            image: Sentinel2Image or LandsatImage
            source: "sentinel2" or "landsat"

        Returns:
            Dict mapping Sentinel-2 band names to data arrays
        """
        if source == "sentinel2":
            return {name: band.data for name, band in image.bands.items() if band.data is not None}
        elif source == "landsat":
            return image.to_sentinel2_bands()
        else:
            raise ValueError(f"Unknown source: {source}")
