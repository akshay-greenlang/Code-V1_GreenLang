"""
Satellite Data Clients.

Provides access to satellite imagery from multiple sources:
- Sentinel-2 via Copernicus Data Space
- Landsat 8/9 via USGS Earth Explorer
- Harmonized access across both platforms
"""

from greenlang.satellite.clients.sentinel2_client import (
    Sentinel2Client,
    Sentinel2Image,
    Sentinel2Band,
    BoundingBox,
    SearchResult as Sentinel2SearchResult,
    SceneClassification,
    Sentinel2ClientError,
    AuthenticationError as Sentinel2AuthError,
    QueryError as Sentinel2QueryError,
    DownloadError as Sentinel2DownloadError,
)

from greenlang.satellite.clients.landsat_client import (
    LandsatClient,
    LandsatImage,
    LandsatBand,
    HarmonizedSatelliteClient,
    SearchResult as LandsatSearchResult,
    LandsatQAFlags,
    LandsatClientError,
    AuthenticationError as LandsatAuthError,
    QueryError as LandsatQueryError,
    DownloadError as LandsatDownloadError,
)

__all__ = [
    # Sentinel-2
    "Sentinel2Client",
    "Sentinel2Image",
    "Sentinel2Band",
    "BoundingBox",
    "Sentinel2SearchResult",
    "SceneClassification",
    "Sentinel2ClientError",
    "Sentinel2AuthError",
    "Sentinel2QueryError",
    "Sentinel2DownloadError",

    # Landsat
    "LandsatClient",
    "LandsatImage",
    "LandsatBand",
    "LandsatSearchResult",
    "LandsatQAFlags",
    "LandsatClientError",
    "LandsatAuthError",
    "LandsatQueryError",
    "LandsatDownloadError",

    # Harmonized
    "HarmonizedSatelliteClient",
]
