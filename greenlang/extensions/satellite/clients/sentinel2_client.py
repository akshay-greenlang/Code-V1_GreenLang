"""
Sentinel-2 Client for Copernicus Data Space API Integration.

Provides access to Sentinel-2 MSI imagery for deforestation detection
and EUDR compliance verification.

Bands used:
- B2 (Blue): 490nm, 10m resolution
- B3 (Green): 560nm, 10m resolution
- B4 (Red): 665nm, 10m resolution
- B8 (NIR): 842nm, 10m resolution
- B11 (SWIR1): 1610nm, 20m resolution
- B12 (SWIR2): 2190nm, 20m resolution
- SCL (Scene Classification Layer): 20m resolution
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional
import hashlib
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class SceneClassification(IntEnum):
    """Sentinel-2 Scene Classification Layer (SCL) values."""
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    DARK_AREA_PIXELS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW = 11


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
            raise ValueError(f"min_lon must be less than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError(f"min_lat must be less than max_lat")

    def to_wkt(self) -> str:
        """Convert to WKT POLYGON string."""
        return (
            f"POLYGON(({self.min_lon} {self.min_lat}, "
            f"{self.max_lon} {self.min_lat}, "
            f"{self.max_lon} {self.max_lat}, "
            f"{self.min_lon} {self.max_lat}, "
            f"{self.min_lon} {self.min_lat}))"
        )

    def area_km2(self) -> float:
        """Approximate area in square kilometers using haversine."""
        lat_avg = (self.min_lat + self.max_lat) / 2
        lat_rad = np.radians(lat_avg)

        # Approximate degrees to km at this latitude
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(lat_rad)

        width_km = (self.max_lon - self.min_lon) * km_per_deg_lon
        height_km = (self.max_lat - self.min_lat) * km_per_deg_lat

        return width_km * height_km


@dataclass
class Sentinel2Band:
    """Represents a single Sentinel-2 band."""
    name: str
    wavelength_nm: int
    resolution_m: int
    data: Optional[np.ndarray] = None

    @property
    def is_loaded(self) -> bool:
        """Check if band data is loaded."""
        return self.data is not None


@dataclass
class Sentinel2Image:
    """Container for Sentinel-2 imagery data."""
    product_id: str
    acquisition_date: datetime
    cloud_cover_percentage: float
    bbox: BoundingBox
    bands: dict[str, Sentinel2Band] = field(default_factory=dict)
    scl: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_band(self, band_name: str) -> Optional[np.ndarray]:
        """Get band data by name."""
        band = self.bands.get(band_name)
        return band.data if band else None

    def get_cloud_mask(self) -> Optional[np.ndarray]:
        """Generate cloud mask from SCL band.

        Returns boolean array where True = valid (cloud-free) pixel.
        """
        if self.scl is None:
            return None

        # Valid pixels are vegetation, not vegetated, water, unclassified
        valid_classes = [
            SceneClassification.VEGETATION,
            SceneClassification.NOT_VEGETATED,
            SceneClassification.WATER,
            SceneClassification.UNCLASSIFIED,
        ]

        mask = np.isin(self.scl, valid_classes)
        return mask

    def apply_cloud_mask(self, band_data: np.ndarray) -> np.ndarray:
        """Apply cloud mask to band data, setting masked pixels to NaN."""
        mask = self.get_cloud_mask()
        if mask is None:
            return band_data

        # Resample mask if needed
        if mask.shape != band_data.shape:
            # Simple nearest neighbor resampling
            scale_y = band_data.shape[0] / mask.shape[0]
            scale_x = band_data.shape[1] / mask.shape[1]
            y_indices = (np.arange(band_data.shape[0]) / scale_y).astype(int)
            x_indices = (np.arange(band_data.shape[1]) / scale_x).astype(int)
            y_indices = np.clip(y_indices, 0, mask.shape[0] - 1)
            x_indices = np.clip(x_indices, 0, mask.shape[1] - 1)
            mask = mask[np.ix_(y_indices, x_indices)]

        result = band_data.astype(float)
        result[~mask] = np.nan
        return result


@dataclass
class SearchResult:
    """Result from image search query."""
    product_id: str
    acquisition_date: datetime
    cloud_cover_percentage: float
    bbox: BoundingBox
    download_url: str
    file_size_mb: float
    processing_level: str = "L2A"


class Sentinel2ClientError(Exception):
    """Base exception for Sentinel-2 client errors."""
    pass


class AuthenticationError(Sentinel2ClientError):
    """Authentication failed."""
    pass


class QueryError(Sentinel2ClientError):
    """Query failed."""
    pass


class DownloadError(Sentinel2ClientError):
    """Download failed."""
    pass


class Sentinel2Client:
    """
    Client for accessing Sentinel-2 imagery via Copernicus Data Space API.

    Supports:
    - Querying imagery by bounding box and date range
    - Downloading specific bands
    - Cloud masking using SCL band
    - Local tile caching
    """

    COPERNICUS_API_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    COPERNICUS_AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # Standard band definitions
    BAND_DEFINITIONS = {
        "B2": Sentinel2Band("B2", 490, 10),
        "B3": Sentinel2Band("B3", 560, 10),
        "B4": Sentinel2Band("B4", 665, 10),
        "B8": Sentinel2Band("B8", 842, 10),
        "B11": Sentinel2Band("B11", 1610, 20),
        "B12": Sentinel2Band("B12", 2190, 20),
    }

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_mock: bool = False,
    ):
        """
        Initialize Sentinel-2 client.

        Args:
            client_id: Copernicus Data Space client ID
            client_secret: Copernicus Data Space client secret
            cache_dir: Directory for caching downloaded tiles
            use_mock: Use mock data instead of real API calls
        """
        self.client_id = client_id or os.getenv("COPERNICUS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("COPERNICUS_CLIENT_SECRET")
        self.cache_dir = cache_dir or Path.home() / ".greenlang" / "satellite_cache" / "sentinel2"
        self.use_mock = use_mock
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        if not self.use_mock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, bbox: BoundingBox, date: datetime, bands: list[str]) -> str:
        """Generate cache key for tile data."""
        key_data = f"{bbox.min_lon}_{bbox.min_lat}_{bbox.max_lon}_{bbox.max_lat}_{date.isoformat()}_{','.join(sorted(bands))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[dict]:
        """Check if cached data exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: dict) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    def authenticate(self) -> str:
        """
        Authenticate with Copernicus Data Space API.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If authentication fails
        """
        if self.use_mock:
            self._access_token = "mock_token_12345"
            self._token_expiry = datetime.now() + timedelta(hours=1)
            return self._access_token

        if self._access_token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._access_token

        if not self.client_id or not self.client_secret:
            raise AuthenticationError(
                "Missing credentials. Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET "
                "environment variables or pass them to the constructor."
            )

        # In production, this would make an HTTP request
        # For now, we simulate the authentication
        logger.info("Authenticating with Copernicus Data Space API...")

        # Stub: In real implementation, use requests to get token
        # response = requests.post(
        #     self.COPERNICUS_AUTH_URL,
        #     data={
        #         "grant_type": "client_credentials",
        #         "client_id": self.client_id,
        #         "client_secret": self.client_secret,
        #     }
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
    ) -> list[SearchResult]:
        """
        Search for Sentinel-2 images matching criteria.

        Args:
            bbox: Bounding box for search area
            start_date: Start of date range
            end_date: End of date range
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            max_results: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            QueryError: If search fails
        """
        if self.use_mock:
            return self._mock_search(bbox, start_date, end_date, max_cloud_cover, max_results)

        # Ensure authenticated
        self.authenticate()

        # Build OData query
        # In production, this would construct and execute an HTTP request
        logger.info(
            f"Searching Sentinel-2 imagery: bbox={bbox}, "
            f"dates={start_date.date()} to {end_date.date()}, "
            f"max_cloud={max_cloud_cover}%"
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
    ) -> list[SearchResult]:
        """Generate mock search results for testing."""
        results = []

        # Generate deterministic mock results based on input
        seed = int(bbox.min_lon * 100 + bbox.min_lat * 100)
        rng = np.random.default_rng(seed)

        current_date = start_date
        while current_date <= end_date and len(results) < max_results:
            # Sentinel-2 revisit time is ~5 days
            cloud_cover = rng.uniform(0, 50)

            if cloud_cover <= max_cloud_cover:
                product_id = f"S2A_MSIL2A_{current_date.strftime('%Y%m%d')}T{rng.integers(0, 24):02d}{rng.integers(0, 60):02d}{rng.integers(0, 60):02d}_N0500_R{rng.integers(1, 150):03d}_T{rng.integers(10, 60):02d}ABC_{current_date.strftime('%Y%m%d')}T{rng.integers(0, 24):02d}{rng.integers(0, 60):02d}{rng.integers(0, 60):02d}"

                results.append(SearchResult(
                    product_id=product_id,
                    acquisition_date=current_date,
                    cloud_cover_percentage=round(cloud_cover, 2),
                    bbox=bbox,
                    download_url=f"https://download.dataspace.copernicus.eu/odata/v1/Products('{product_id}')/$value",
                    file_size_mb=round(rng.uniform(500, 1500), 1),
                    processing_level="L2A",
                ))

            current_date += timedelta(days=5)

        logger.info(f"Found {len(results)} Sentinel-2 images")
        return results

    def download_image(
        self,
        search_result: SearchResult,
        bands: Optional[list[str]] = None,
        apply_cloud_mask: bool = True,
    ) -> Sentinel2Image:
        """
        Download Sentinel-2 image with specified bands.

        Args:
            search_result: SearchResult from search() method
            bands: List of band names to download (default: all standard bands)
            apply_cloud_mask: Whether to download SCL band for cloud masking

        Returns:
            Sentinel2Image with downloaded band data

        Raises:
            DownloadError: If download fails
        """
        if bands is None:
            bands = list(self.BAND_DEFINITIONS.keys())

        # Validate bands
        invalid_bands = set(bands) - set(self.BAND_DEFINITIONS.keys())
        if invalid_bands:
            raise ValueError(f"Invalid bands: {invalid_bands}")

        if self.use_mock:
            return self._mock_download(search_result, bands, apply_cloud_mask)

        # Ensure authenticated
        self.authenticate()

        logger.info(f"Downloading image {search_result.product_id}, bands: {bands}")

        raise DownloadError(
            "Real API download not implemented. Use use_mock=True for testing."
        )

    def _mock_download(
        self,
        search_result: SearchResult,
        bands: list[str],
        apply_cloud_mask: bool,
    ) -> Sentinel2Image:
        """Generate mock image data for testing."""
        # Deterministic seed based on product ID
        seed = int(hashlib.md5(search_result.product_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # Calculate image dimensions based on bbox and resolution
        bbox = search_result.bbox
        lat_km = (bbox.max_lat - bbox.min_lat) * 111.0
        lon_km = (bbox.max_lon - bbox.min_lon) * 111.0 * np.cos(np.radians((bbox.min_lat + bbox.max_lat) / 2))

        # Base dimensions at 10m resolution
        base_height = max(10, int(lat_km * 100))  # 100 pixels per km at 10m
        base_width = max(10, int(lon_km * 100))

        # Limit size for mock data
        base_height = min(base_height, 1000)
        base_width = min(base_width, 1000)

        image = Sentinel2Image(
            product_id=search_result.product_id,
            acquisition_date=search_result.acquisition_date,
            cloud_cover_percentage=search_result.cloud_cover_percentage,
            bbox=bbox,
            metadata={
                "processing_level": search_result.processing_level,
                "file_size_mb": search_result.file_size_mb,
            }
        )

        # Generate band data
        for band_name in bands:
            band_def = self.BAND_DEFINITIONS[band_name]

            # Adjust resolution
            if band_def.resolution_m == 20:
                height = base_height // 2
                width = base_width // 2
            else:
                height = base_height
                width = base_width

            # Generate realistic reflectance values (0-10000 for L2A)
            # Create spatial patterns that look like real imagery
            x = np.linspace(0, 4 * np.pi, width)
            y = np.linspace(0, 4 * np.pi, height)
            xx, yy = np.meshgrid(x, y)

            # Base pattern
            pattern = np.sin(xx * rng.uniform(0.5, 2)) * np.cos(yy * rng.uniform(0.5, 2))
            pattern = (pattern + 1) / 2  # Normalize to 0-1

            # Add noise
            noise = rng.normal(0, 0.1, (height, width))

            # Scale to band-appropriate values
            if band_name in ["B8"]:  # NIR - high for vegetation
                base_value = 4000
                scale = 3000
            elif band_name in ["B4"]:  # Red - low for vegetation
                base_value = 1000
                scale = 1500
            elif band_name in ["B3"]:  # Green
                base_value = 1200
                scale = 1200
            elif band_name in ["B2"]:  # Blue
                base_value = 1100
                scale = 1000
            elif band_name in ["B11", "B12"]:  # SWIR
                base_value = 2000
                scale = 2000
            else:
                base_value = 2000
                scale = 2000

            data = base_value + scale * pattern + scale * 0.2 * noise
            data = np.clip(data, 0, 10000).astype(np.uint16)

            image.bands[band_name] = Sentinel2Band(
                name=band_name,
                wavelength_nm=band_def.wavelength_nm,
                resolution_m=band_def.resolution_m,
                data=data,
            )

        # Generate SCL band for cloud masking
        if apply_cloud_mask:
            scl_height = base_height // 2
            scl_width = base_width // 2

            # Generate realistic SCL classification
            scl_data = np.full((scl_height, scl_width), SceneClassification.VEGETATION, dtype=np.uint8)

            # Add some non-vegetated areas
            non_veg_mask = rng.random((scl_height, scl_width)) < 0.2
            scl_data[non_veg_mask] = SceneClassification.NOT_VEGETATED

            # Add cloud pixels based on reported cloud cover
            cloud_fraction = search_result.cloud_cover_percentage / 100.0
            cloud_mask = rng.random((scl_height, scl_width)) < cloud_fraction
            scl_data[cloud_mask] = rng.choice(
                [SceneClassification.CLOUD_MEDIUM_PROBABILITY, SceneClassification.CLOUD_HIGH_PROBABILITY],
                size=cloud_mask.sum()
            )

            # Add some cloud shadows near clouds
            shadow_mask = rng.random((scl_height, scl_width)) < (cloud_fraction * 0.3)
            scl_data[shadow_mask & ~cloud_mask] = SceneClassification.CLOUD_SHADOWS

            image.scl = scl_data

        logger.info(f"Downloaded mock image: {image.product_id}, shape: {base_height}x{base_width}")
        return image

    def get_time_series(
        self,
        bbox: BoundingBox,
        start_date: datetime,
        end_date: datetime,
        bands: Optional[list[str]] = None,
        max_cloud_cover: float = 20.0,
    ) -> list[Sentinel2Image]:
        """
        Get time series of images for an area.

        Args:
            bbox: Bounding box for area of interest
            start_date: Start of date range
            end_date: End of date range
            bands: Bands to download
            max_cloud_cover: Maximum cloud cover percentage

        Returns:
            List of Sentinel2Image objects sorted by date
        """
        results = self.search(bbox, start_date, end_date, max_cloud_cover)

        images = []
        for result in results:
            try:
                image = self.download_image(result, bands)
                images.append(image)
            except DownloadError as e:
                logger.warning(f"Failed to download {result.product_id}: {e}")

        # Sort by acquisition date
        images.sort(key=lambda x: x.acquisition_date)

        return images

    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached tile data.

        Args:
            older_than_days: Only clear files older than this many days

        Returns:
            Number of files deleted
        """
        if not self.cache_dir.exists():
            return 0

        deleted = 0
        cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None

        for cache_file in self.cache_dir.glob("*.json"):
            if cutoff:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if mtime > cutoff:
                    continue

            try:
                cache_file.unlink()
                deleted += 1
            except OSError as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")

        logger.info(f"Cleared {deleted} cached files")
        return deleted
