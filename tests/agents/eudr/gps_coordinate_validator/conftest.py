# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-007 GPS Coordinate Validator test suite.

Provides reusable fixtures for sample coordinates, configuration, engine
instances, helper functions, and deterministic test data used across all
test modules in this package.

Sample Coordinates (20+ predefined):
    COCOA_FARM_GHANA, PALM_PLANTATION_INDONESIA, COFFEE_FARM_COLOMBIA,
    SOYA_FIELD_BRAZIL, RUBBER_FARM_THAILAND, CATTLE_RANCH_BRAZIL,
    TIMBER_FOREST_CONGO, OCEAN_POINT, NULL_ISLAND, SWAPPED_COORDS,
    SIGN_ERROR, LOW_PRECISION, TRUNCATED, ARCTIC_POINT, URBAN_POINT,
    DESERT_POINT, HIGH_PRECISION, BOUNDARY_LATITUDE, BOUNDARY_LONGITUDE,
    ANTIMERIDIAN_EAST

DMS/DDM/UTM format strings for parsing tests.
Fixture factories for all seven engine components.
Helper assertions and builder functions.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.config import (
    GPSCoordinateValidatorConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.agents.eudr.gps_coordinate_validator.models import (
    VERSION,
    CoordinateFormat,
    GeodeticDatum,
    PrecisionLevel,
    SourceType,
    ValidationErrorType,
    ValidationSeverity,
    RawCoordinate,
    ParsedCoordinate,
    NormalizedCoordinate,
    PrecisionResult,
    ValidationError,
    ValidationResult,
    BatchValidationResult,
)


# ---------------------------------------------------------------------------
# Sample Coordinates (lat, lon) - 20+ predefined points
# ---------------------------------------------------------------------------

# Valid production coordinates for EUDR commodities
COCOA_FARM_GHANA: Tuple[float, float] = (5.603716, -0.186964)
PALM_PLANTATION_INDONESIA: Tuple[float, float] = (-2.524000, 111.876000)
COFFEE_FARM_COLOMBIA: Tuple[float, float] = (4.570868, -75.678000)
SOYA_FIELD_BRAZIL: Tuple[float, float] = (-12.970000, -55.320000)
RUBBER_FARM_THAILAND: Tuple[float, float] = (7.880400, 98.392300)
CATTLE_RANCH_BRAZIL: Tuple[float, float] = (-15.780000, -47.930000)
TIMBER_FOREST_CONGO: Tuple[float, float] = (-4.322000, 15.313000)
COFFEE_FARM_ETHIOPIA: Tuple[float, float] = (7.045000, 38.477000)
PALM_PLANTATION_MALAYSIA: Tuple[float, float] = (2.950000, 101.700000)
COCOA_FARM_IVORY_COAST: Tuple[float, float] = (6.827000, -5.289000)

# Invalid / suspicious coordinates
OCEAN_POINT: Tuple[float, float] = (0.0, -30.0)
NULL_ISLAND: Tuple[float, float] = (0.0, 0.0)
SWAPPED_COORDS: Tuple[float, float] = (-0.186964, 5.603716)
SIGN_ERROR: Tuple[float, float] = (5.603716, 0.186964)
LOW_PRECISION: Tuple[float, float] = (5.6, -0.2)
TRUNCATED: Tuple[float, float] = (6.0, -1.0)
ARCTIC_POINT: Tuple[float, float] = (78.0, 15.0)
URBAN_POINT: Tuple[float, float] = (51.5074, -0.1278)
DESERT_POINT: Tuple[float, float] = (23.4162, 25.6628)
ANTARCTIC_POINT: Tuple[float, float] = (-75.0, 0.0)

# Boundary / edge-case coordinates
HIGH_PRECISION: Tuple[float, float] = (5.60371589, -0.18696423)
BOUNDARY_LATITUDE: Tuple[float, float] = (90.0, 0.0)
BOUNDARY_LONGITUDE: Tuple[float, float] = (0.0, 180.0)
ANTIMERIDIAN_EAST: Tuple[float, float] = (0.0, -179.999999)
SOUTH_POLE: Tuple[float, float] = (-90.0, 0.0)


# ---------------------------------------------------------------------------
# DMS / DDM / UTM format strings for parsing tests
# ---------------------------------------------------------------------------

DMS_GHANA: str = "5\u00b036'13.4\"N 0\u00b011'13.1\"W"
DMS_BRAZIL: str = "12\u00b058'12.0\"S 55\u00b019'12.0\"W"
DMS_INDONESIA: str = "2\u00b031'26.4\"S 111\u00b052'33.6\"E"

DDM_GHANA: str = "5\u00b036.2233'N 0\u00b011.2183'W"
DDM_COLOMBIA: str = "4\u00b034.2521'N 75\u00b040.6800'W"

UTM_GHANA: str = "30N 808820 620350"
UTM_BRAZIL: str = "21L 693845 8565152"
UTM_INDONESIA: str = "49S 702834 9720714"

MGRS_GHANA: str = "30NUN0882020350"

# Alternative format representations
DD_SUFFIX_GHANA: str = "5.603716N 0.186964W"
DD_COMMA_GHANA: str = "5.603716, -0.186964"
DD_SPACE_GHANA: str = "5.603716 -0.186964"
DD_SEMICOLON_GHANA: str = "5.603716; -0.186964"

# DMS with different separator styles
DMS_WITH_DEG_SYMBOL: str = "5\u00b036'13.4\"N 0\u00b011'13.1\"W"
DMS_WITH_D_SEPARATOR: str = "5d36'13.4\"N 0d11'13.1\"W"
DMS_WITH_DEG_TEXT: str = "5deg36'13.4\"N 0deg11'13.1\"W"
DMS_UNICODE_SYMBOLS: str = "5\u00b036\u203213.4\u2033N 0\u00b011\u203213.1\u2033W"

# Garbage / invalid input strings
GARBAGE_INPUT: str = "not a coordinate"
EMPTY_INPUT: str = ""
PARTIAL_INPUT: str = "5.603716"


# ---------------------------------------------------------------------------
# Commodity-country-coordinate mappings for plausibility tests
# ---------------------------------------------------------------------------

PLAUSIBLE_COMMODITY_COORDINATES: Dict[str, List[Dict[str, Any]]] = {
    "cocoa": [
        {"lat": 5.603716, "lon": -0.186964, "country": "GH", "plausible": True},
        {"lat": 6.827, "lon": -5.289, "country": "CI", "plausible": True},
        {"lat": 78.0, "lon": 15.0, "country": "NO", "plausible": False},
    ],
    "oil_palm": [
        {"lat": -2.524, "lon": 111.876, "country": "ID", "plausible": True},
        {"lat": 2.95, "lon": 101.7, "country": "MY", "plausible": True},
        {"lat": 51.5074, "lon": -0.1278, "country": "GB", "plausible": False},
    ],
    "coffee": [
        {"lat": 4.570868, "lon": -75.678, "country": "CO", "plausible": True},
        {"lat": 7.045, "lon": 38.477, "country": "ET", "plausible": True},
        {"lat": -75.0, "lon": 0.0, "country": "AQ", "plausible": False},
    ],
    "soya": [
        {"lat": -12.97, "lon": -55.32, "country": "BR", "plausible": True},
        {"lat": 78.0, "lon": 15.0, "country": "NO", "plausible": False},
    ],
    "rubber": [
        {"lat": 7.8804, "lon": 98.3923, "country": "TH", "plausible": True},
        {"lat": 23.4162, "lon": 25.6628, "country": "EG", "plausible": False},
    ],
    "cattle": [
        {"lat": -15.78, "lon": -47.93, "country": "BR", "plausible": True},
        {"lat": -75.0, "lon": 0.0, "country": "AQ", "plausible": False},
    ],
    "wood": [
        {"lat": -4.322, "lon": 15.313, "country": "CD", "plausible": True},
        {"lat": 0.0, "lon": -30.0, "country": "XX", "plausible": False},
    ],
}


# ---------------------------------------------------------------------------
# Datum transformation reference data
# ---------------------------------------------------------------------------

# Known datum transformation vectors: (source_datum, lat, lon, expected_wgs84_lat, expected_wgs84_lon, tolerance_m)
DATUM_TRANSFORM_REFERENCE: List[Dict[str, Any]] = [
    {
        "name": "NAD27 to WGS84 - North America",
        "source_datum": GeodeticDatum.NAD27,
        "lat": 40.0, "lon": -75.0,
        "expected_lat": 40.0002, "expected_lon": -74.9998,
        "tolerance_m": 15.0,
    },
    {
        "name": "NAD83 to WGS84 - near identity",
        "source_datum": GeodeticDatum.NAD83,
        "lat": 40.0, "lon": -75.0,
        "expected_lat": 40.0, "expected_lon": -75.0,
        "tolerance_m": 2.0,
    },
    {
        "name": "ED50 to WGS84 - Europe",
        "source_datum": GeodeticDatum.ED50,
        "lat": 48.0, "lon": 2.0,
        "expected_lat": 47.9999, "expected_lon": 1.9998,
        "tolerance_m": 15.0,
    },
    {
        "name": "SIRGAS2000 to WGS84 - South America",
        "source_datum": GeodeticDatum.SIRGAS2000,
        "lat": -12.97, "lon": -55.32,
        "expected_lat": -12.97, "expected_lon": -55.32,
        "tolerance_m": 2.0,
    },
    {
        "name": "INDIAN_1975 to WGS84 - SE Asia",
        "source_datum": GeodeticDatum.INDIAN_1975,
        "lat": 7.88, "lon": 98.39,
        "expected_lat": 7.88, "expected_lon": 98.39,
        "tolerance_m": 25.0,
    },
    {
        "name": "PULKOVO_1942 to WGS84 - Russia",
        "source_datum": GeodeticDatum.PULKOVO_1942,
        "lat": 55.75, "lon": 37.62,
        "expected_lat": 55.75, "expected_lon": 37.62,
        "tolerance_m": 10.0,
    },
    {
        "name": "ARC1960 to WGS84 - East Africa",
        "source_datum": GeodeticDatum.ARC1960,
        "lat": -1.29, "lon": 36.82,
        "expected_lat": -1.29, "expected_lon": 36.82,
        "tolerance_m": 25.0,
    },
    {
        "name": "TOKYO to WGS84 - Japan",
        "source_datum": GeodeticDatum.TOKYO,
        "lat": 35.68, "lon": 139.77,
        "expected_lat": 35.68, "expected_lon": 139.77,
        "tolerance_m": 15.0,
    },
    {
        "name": "GDA94 to WGS84 - Australia",
        "source_datum": GeodeticDatum.GDA94,
        "lat": -33.87, "lon": 151.21,
        "expected_lat": -33.87, "expected_lon": 151.21,
        "tolerance_m": 2.0,
    },
]


# ---------------------------------------------------------------------------
# Precision reference data
# ---------------------------------------------------------------------------

# (decimal_places, expected_ground_resolution_m_at_equator)
PRECISION_GROUND_RESOLUTION: List[Tuple[int, float]] = [
    (0, 111_320.0),
    (1, 11_132.0),
    (2, 1_113.2),
    (3, 111.32),
    (4, 11.132),
    (5, 1.1132),
    (6, 0.11132),
    (7, 0.011132),
    (8, 0.0011132),
    (9, 0.00011132),
    (10, 0.000011132),
]


# ---------------------------------------------------------------------------
# Quality tier thresholds
# ---------------------------------------------------------------------------

GOLD_THRESHOLD: float = 90.0
SILVER_THRESHOLD: float = 70.0
BRONZE_THRESHOLD: float = 50.0
UNVERIFIED_THRESHOLD: float = 0.0

SHA256_HEX_LENGTH: int = 64


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> GPSCoordinateValidatorConfig:
    """Create a GPSCoordinateValidatorConfig with test defaults."""
    return GPSCoordinateValidatorConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        format_detection_min_confidence=0.7,
        eudr_min_decimal_places=5,
        datum_convergence_tolerance=1e-12,
        datum_max_iterations=20,
        null_island_threshold_m=1000.0,
        near_duplicate_threshold_m=1.0,
        spatial_hash_cell_size_deg=0.01,
        max_batch_size=10_000,
        enable_provenance=True,
        genesis_hash="GL-EUDR-GPS-007-TEST-GENESIS",
        enable_metrics=False,
        pool_size=5,
        rate_limit=500,
    )


@pytest.fixture(autouse=True)
def reset_singleton_config():
    """Reset the singleton config after each test to avoid cross-test leaks."""
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Engine Fixtures (stubs for engines not yet implemented)
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinate_parser(config):
    """Create a CoordinateParser instance for testing.

    If the engine is not yet implemented, this fixture will skip the test
    with an informative message.
    """
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.coordinate_parser import (
            CoordinateParser,
        )
        return CoordinateParser(config=config)
    except ImportError:
        pytest.skip("CoordinateParser not yet implemented")


@pytest.fixture
def datum_transformer(config):
    """Create a DatumTransformer instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.datum_transformer import (
            DatumTransformer,
        )
        return DatumTransformer(config=config)
    except ImportError:
        pytest.skip("DatumTransformer not yet implemented")


@pytest.fixture
def precision_analyzer(config):
    """Create a PrecisionAnalyzer instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.precision_analyzer import (
            PrecisionAnalyzer,
        )
        return PrecisionAnalyzer(config=config)
    except ImportError:
        pytest.skip("PrecisionAnalyzer not yet implemented")


@pytest.fixture
def format_validator(config):
    """Create a FormatValidator instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.format_validator import (
            FormatValidator,
        )
        return FormatValidator(config=config)
    except ImportError:
        pytest.skip("FormatValidator not yet implemented")


@pytest.fixture
def spatial_plausibility_checker(config):
    """Create a SpatialPlausibilityChecker instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.spatial_plausibility import (
            SpatialPlausibilityChecker,
        )
        return SpatialPlausibilityChecker(config=config)
    except ImportError:
        pytest.skip("SpatialPlausibilityChecker not yet implemented")


@pytest.fixture
def reverse_geocoder(config):
    """Create a ReverseGeocoder instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.reverse_geocoder import (
            ReverseGeocoder,
        )
        return ReverseGeocoder(config=config)
    except ImportError:
        pytest.skip("ReverseGeocoder not yet implemented")


@pytest.fixture
def accuracy_assessor(config):
    """Create an AccuracyAssessor instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.accuracy_assessor import (
            AccuracyAssessor,
        )
        return AccuracyAssessor(config=config)
    except ImportError:
        pytest.skip("AccuracyAssessor not yet implemented")


@pytest.fixture
def compliance_reporter(config):
    """Create a ComplianceReporter instance for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.compliance_reporter import (
            ComplianceReporter,
        )
        return ComplianceReporter(config=config)
    except ImportError:
        pytest.skip("ComplianceReporter not yet implemented")


@pytest.fixture
def service(config):
    """Create the top-level GPSCoordinateValidatorService for testing."""
    try:
        from greenlang.agents.eudr.gps_coordinate_validator.service import (
            GPSCoordinateValidatorService,
        )
        return GPSCoordinateValidatorService(config=config)
    except ImportError:
        pytest.skip("GPSCoordinateValidatorService not yet implemented")


# ---------------------------------------------------------------------------
# Coordinate Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_coordinate() -> Tuple[float, float]:
    """A valid high-precision cocoa farm coordinate in Ghana."""
    return COCOA_FARM_GHANA


@pytest.fixture
def invalid_ocean_coordinate() -> Tuple[float, float]:
    """An invalid coordinate in the Atlantic Ocean."""
    return OCEAN_POINT


@pytest.fixture
def swapped_coordinate() -> Tuple[float, float]:
    """A coordinate with lat/lon swapped (Ghana origin)."""
    return SWAPPED_COORDS


@pytest.fixture
def low_precision_coordinate() -> Tuple[float, float]:
    """A coordinate with only 1 decimal place (inadequate precision)."""
    return LOW_PRECISION


@pytest.fixture
def null_island_coordinate() -> Tuple[float, float]:
    """A coordinate at Null Island (0, 0)."""
    return NULL_ISLAND


@pytest.fixture
def high_precision_coordinate() -> Tuple[float, float]:
    """A coordinate with 8 decimal places (survey grade)."""
    return HIGH_PRECISION


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def make_raw(
    input_str: str,
    commodity: str = "cocoa",
    country: str = "GH",
    source_datum: Optional[GeodeticDatum] = None,
    source_type: Optional[SourceType] = None,
) -> RawCoordinate:
    """Build a RawCoordinate instance for testing.

    Args:
        input_str: The raw coordinate string.
        commodity: EUDR commodity identifier.
        country: ISO 3166-1 alpha-2 country code.
        source_datum: Optional declared source datum.
        source_type: Optional GPS source type.

    Returns:
        RawCoordinate instance ready for parsing tests.
    """
    return RawCoordinate(
        input_string=input_str,
        source_datum=source_datum,
        country_iso=country,
        source_type=source_type,
        metadata={"commodity": commodity},
    )


def make_normalized(
    lat: float,
    lon: float,
    decimal_places: int = 6,
    source_datum: GeodeticDatum = GeodeticDatum.WGS84,
    displacement_m: float = 0.0,
) -> NormalizedCoordinate:
    """Build a NormalizedCoordinate instance for testing.

    Args:
        lat: WGS84 latitude.
        lon: WGS84 longitude.
        decimal_places: Number of decimal places (informational).
        source_datum: Original datum.
        displacement_m: Displacement from datum transformation.

    Returns:
        NormalizedCoordinate instance.
    """
    return NormalizedCoordinate(
        latitude=lat,
        longitude=lon,
        source_datum=source_datum,
        target_datum=GeodeticDatum.WGS84,
        displacement_m=displacement_m,
        original_latitude=lat,
        original_longitude=lon,
        transformation_method="identity" if source_datum == GeodeticDatum.WGS84 else "helmert_7param",
    )


def make_parsed(
    lat: float,
    lon: float,
    fmt: CoordinateFormat = CoordinateFormat.DECIMAL_DEGREES,
    confidence: float = 0.95,
    original_input: str = "",
) -> ParsedCoordinate:
    """Build a ParsedCoordinate instance for testing.

    Args:
        lat: Parsed latitude.
        lon: Parsed longitude.
        fmt: Detected coordinate format.
        confidence: Format detection confidence.
        original_input: Original input string.

    Returns:
        ParsedCoordinate instance.
    """
    return ParsedCoordinate(
        latitude=lat,
        longitude=lon,
        detected_format=fmt,
        format_confidence=confidence,
        original_input=original_input,
        parse_successful=True,
    )


def assert_valid(result: Any) -> None:
    """Assert that a validation-like result indicates validity.

    Works with any object that has an ``is_valid`` boolean attribute.

    Args:
        result: A result object with an is_valid attribute.

    Raises:
        AssertionError: If result is not valid.
    """
    assert hasattr(result, "is_valid"), "Result must have is_valid attribute"
    assert result.is_valid is True, (
        f"Expected result to be valid, but is_valid={result.is_valid}. "
        f"Errors: {getattr(result, 'errors', [])}"
    )


def assert_invalid(result: Any) -> None:
    """Assert that a validation-like result indicates invalidity.

    Args:
        result: A result object with an is_valid attribute.

    Raises:
        AssertionError: If result is unexpectedly valid.
    """
    assert hasattr(result, "is_valid"), "Result must have is_valid attribute"
    assert result.is_valid is False, "Expected result to be invalid"


def assert_close(
    actual: float,
    expected: float,
    tolerance: float = 0.001,
) -> None:
    """Assert two float values are within tolerance.

    Args:
        actual: The actual value.
        expected: The expected value.
        tolerance: Maximum allowed absolute difference.

    Raises:
        AssertionError: If values differ by more than tolerance.
    """
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Values differ by {diff:.8f} (tolerance={tolerance}): "
        f"actual={actual}, expected={expected}"
    )


def haversine_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate Haversine distance in metres between two WGS84 points.

    Args:
        lat1: Latitude of point 1 (degrees).
        lon1: Longitude of point 1 (degrees).
        lat2: Latitude of point 2 (degrees).
        lon2: Longitude of point 2 (degrees).

    Returns:
        Distance in metres.
    """
    earth_radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of a JSON-serializable object.

    Args:
        data: JSON-serializable data.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ground_resolution_at_latitude(decimal_places: int, latitude_deg: float) -> float:
    """Compute approximate ground resolution for given decimal places and latitude.

    One degree of latitude is approximately 111,320 metres everywhere.
    One degree of longitude is approximately 111,320 * cos(latitude) metres.

    Args:
        decimal_places: Number of decimal places in the coordinate.
        latitude_deg: Latitude in degrees.

    Returns:
        Ground resolution in metres (for the longitude direction, which shrinks
        at higher latitudes).
    """
    one_degree_lat_m = 111_320.0
    one_degree_lon_m = 111_320.0 * math.cos(math.radians(latitude_deg))
    resolution_lat = one_degree_lat_m / (10 ** decimal_places)
    resolution_lon = one_degree_lon_m / (10 ** decimal_places)
    return max(resolution_lat, resolution_lon)


# ---------------------------------------------------------------------------
# Country Bounding Boxes (simplified) for swap/sign detection tests
# ---------------------------------------------------------------------------

COUNTRY_BOUNDING_BOXES: Dict[str, Dict[str, float]] = {
    "GH": {"lat_min": 4.5, "lat_max": 11.2, "lon_min": -3.3, "lon_max": 1.2},
    "BR": {"lat_min": -33.9, "lat_max": 5.3, "lon_min": -73.9, "lon_max": -28.6},
    "ID": {"lat_min": -11.0, "lat_max": 6.1, "lon_min": 95.0, "lon_max": 141.0},
    "CO": {"lat_min": -4.2, "lat_max": 13.4, "lon_min": -81.7, "lon_max": -66.9},
    "TH": {"lat_min": 5.6, "lat_max": 20.5, "lon_min": 97.3, "lon_max": 105.6},
    "CI": {"lat_min": 4.3, "lat_max": 10.7, "lon_min": -8.6, "lon_max": -2.5},
    "MY": {"lat_min": 0.8, "lat_max": 7.4, "lon_min": 99.6, "lon_max": 119.3},
    "ET": {"lat_min": 3.4, "lat_max": 14.9, "lon_min": 33.0, "lon_max": 48.0},
    "CD": {"lat_min": -13.5, "lat_max": 5.4, "lon_min": 12.2, "lon_max": 31.3},
    "KE": {"lat_min": -4.7, "lat_max": 5.1, "lon_min": 33.9, "lon_max": 41.9},
}


# ---------------------------------------------------------------------------
# Elevation reference data for commodity plausibility
# ---------------------------------------------------------------------------

COMMODITY_ELEVATION_LIMITS: Dict[str, Tuple[float, float]] = {
    "cocoa": (0.0, 1500.0),
    "oil_palm": (0.0, 1000.0),
    "coffee": (400.0, 2500.0),
    "soya": (0.0, 1500.0),
    "rubber": (0.0, 700.0),
    "cattle": (0.0, 4000.0),
    "wood": (0.0, 3500.0),
}


# ---------------------------------------------------------------------------
# Additional format strings for comprehensive parsing tests
# ---------------------------------------------------------------------------

# Comma-decimal European-style DD (e.g. "5,603716; -0,186964")
DD_COMMA_DECIMAL_EUROPEAN: str = "5,603716; -0,186964"

# DMS with compact notation (no spaces between parts)
DMS_COMPACT: str = "5\u00b036\u203213.4\u2033N0\u00b011\u203213.1\u2033W"

# UTM with lowercase letter designator
UTM_LOWERCASE_GHANA: str = "30n 808820 620350"

# Coordinate with excessive whitespace
DD_EXCESSIVE_WHITESPACE: str = "   5.603716   ,   -0.186964   "

# Coordinates with trailing zeros
DD_TRAILING_ZEROS: str = "5.60371600, -0.18696400"

# Coordinate with plus sign
DD_PLUS_SIGN: str = "+5.603716, -0.186964"


# ---------------------------------------------------------------------------
# Additional Batch Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_coordinates_10() -> List[Tuple[float, float]]:
    """Batch of 10 diverse coordinates for performance/batch tests."""
    return [
        COCOA_FARM_GHANA,
        PALM_PLANTATION_INDONESIA,
        COFFEE_FARM_COLOMBIA,
        SOYA_FIELD_BRAZIL,
        RUBBER_FARM_THAILAND,
        CATTLE_RANCH_BRAZIL,
        TIMBER_FOREST_CONGO,
        COFFEE_FARM_ETHIOPIA,
        PALM_PLANTATION_MALAYSIA,
        COCOA_FARM_IVORY_COAST,
    ]


@pytest.fixture
def batch_coordinates_mixed() -> List[Tuple[float, float]]:
    """Batch of coordinates mixing valid, invalid, and edge cases."""
    return [
        COCOA_FARM_GHANA,
        OCEAN_POINT,
        NULL_ISLAND,
        HIGH_PRECISION,
        LOW_PRECISION,
        TRUNCATED,
        ARCTIC_POINT,
        URBAN_POINT,
        DESERT_POINT,
        BOUNDARY_LATITUDE,
    ]


@pytest.fixture
def all_eudr_commodities() -> List[str]:
    """All 7 EUDR commodity identifiers."""
    return ["cocoa", "oil_palm", "coffee", "soya", "rubber", "cattle", "wood"]


# ---------------------------------------------------------------------------
# DMS to Decimal Conversion Helper
# ---------------------------------------------------------------------------


def dms_to_decimal(
    degrees: int,
    minutes: int,
    seconds: float,
    hemisphere: str,
) -> float:
    """Convert DMS components to decimal degrees.

    Args:
        degrees: Degree component (non-negative integer).
        minutes: Minute component (0-59).
        seconds: Second component (0-59.999...).
        hemisphere: One of 'N', 'S', 'E', 'W'.

    Returns:
        Decimal degree value (negative for S/W).
    """
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if hemisphere.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def is_within_bounding_box(
    lat: float,
    lon: float,
    country_iso: str,
) -> bool:
    """Check if a coordinate falls within a country's bounding box.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        country_iso: ISO 3166-1 alpha-2 country code.

    Returns:
        True if the coordinate is within the bounding box, False otherwise.
    """
    bbox = COUNTRY_BOUNDING_BOXES.get(country_iso)
    if bbox is None:
        return False
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )
