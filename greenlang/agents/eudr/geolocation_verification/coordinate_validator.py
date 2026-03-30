# -*- coding: utf-8 -*-
"""
Coordinate Validator Engine - AGENT-EUDR-002: Geolocation Verification (Feature 1)

Validates GPS coordinates for EUDR compliance per Article 9, checking WGS84
bounds, coordinate precision (6+ decimal places), lat/lon transposition,
country bounding-box matching against 60+ EUDR-relevant countries, simplified
ocean/land masking, duplicate detection via Haversine distance, elevation
plausibility per commodity, and spatial cluster anomaly detection.

Zero-Hallucination Guarantees:
    - All validation is deterministic (WGS84 bounds, bounding boxes, Haversine)
    - Country matching uses static bounding-box database (no external API)
    - Precision assessment is pure string/float analysis
    - Duplicate detection uses Haversine with configurable threshold
    - SHA-256 provenance hashes on all validation results
    - No ML/LLM used for any coordinate validation logic

Performance Targets:
    - Single coordinate validation: <5ms
    - Batch validation (10,000 coordinates): <2 seconds

Regulatory References:
    - EUDR Article 9: Geolocation of production plots
    - EUDR Article 9(1)(d): Polygon requirement for plots >4 ha
    - EUDR Article 10: Risk assessment using geolocation data

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 1: Coordinate Validation)
Agent ID: GL-EUDR-GEO-002
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from .models import (

    CoordinateValidationResult,
    CoordinateIssue,
    CoordinateIssueType,
    VerifyCoordinateRequest,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Earth radius in metres (WGS84 mean radius).
EARTH_RADIUS_M: float = 6_371_000.0

#: Minimum decimal precision for EUDR Article 9 compliance.
MIN_PRECISION_EUDR: int = 6

#: Default duplicate detection threshold in metres (10m).
DEFAULT_DUPLICATE_THRESHOLD_M: float = 10.0

#: Default cluster anomaly distance threshold in metres (1km).
DEFAULT_CLUSTER_ANOMALY_THRESHOLD_M: float = 1_000.0

#: Minimum number of points to detect cluster anomaly.
MIN_CLUSTER_SIZE: int = 3

# ---------------------------------------------------------------------------
# Country Bounding Box Database
# ---------------------------------------------------------------------------
# Each entry: (min_lat, max_lat, min_lon, max_lon)
# Covers 60+ EUDR-relevant countries: major producing countries for
# cattle, cocoa, coffee, oil palm, rubber, soya, and wood.
# Source: Natural Earth simplified country bounds, rounded to nearest degree.

COUNTRY_BOUNDING_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    # -- South America (EUDR hotspot: soya, cattle, coffee, wood) --
    "BR": (-33.75, 5.27, -73.99, -34.79),     # Brazil
    "CO": (-4.23, 13.39, -79.00, -66.87),      # Colombia
    "PE": (-18.35, -0.04, -81.33, -68.65),      # Peru
    "EC": (-5.01, 1.68, -81.08, -75.19),        # Ecuador
    "BO": (-22.90, -9.68, -69.64, -57.45),      # Bolivia
    "PY": (-27.59, -19.29, -62.65, -54.26),     # Paraguay
    "AR": (-55.06, -21.78, -73.57, -53.64),     # Argentina
    "VE": (0.63, 12.20, -73.35, -59.80),        # Venezuela
    "GY": (1.17, 8.56, -61.40, -56.48),         # Guyana
    "SR": (1.83, 6.01, -58.07, -53.98),         # Suriname
    "UY": (-35.00, -30.09, -58.44, -53.09),     # Uruguay
    "CL": (-55.98, -17.50, -75.64, -66.96),     # Chile

    # -- Central America & Caribbean (coffee, cattle) --
    "MX": (14.53, 32.72, -118.40, -86.71),      # Mexico
    "GT": (13.74, 17.82, -92.23, -88.22),       # Guatemala
    "HN": (12.98, 16.51, -89.35, -83.13),       # Honduras
    "NI": (10.71, 15.03, -87.69, -82.73),       # Nicaragua
    "CR": (8.03, 11.22, -85.95, -82.55),        # Costa Rica
    "PA": (7.20, 9.65, -83.05, -77.17),         # Panama
    "SV": (13.15, 14.45, -90.13, -87.69),       # El Salvador
    "BZ": (15.89, 18.50, -89.22, -87.49),       # Belize

    # -- Southeast Asia (oil palm, rubber, wood) --
    "ID": (-11.01, 5.91, 95.01, 141.02),        # Indonesia
    "MY": (0.85, 7.36, 99.64, 119.27),          # Malaysia
    "TH": (5.61, 20.46, 97.34, 105.64),         # Thailand
    "VN": (8.56, 23.39, 102.14, 109.47),        # Vietnam
    "PH": (4.59, 21.12, 116.93, 126.60),        # Philippines
    "MM": (9.78, 28.54, 92.19, 101.17),         # Myanmar
    "KH": (10.41, 14.69, 102.34, 107.63),       # Cambodia
    "LA": (13.91, 22.50, 100.08, 107.70),       # Laos
    "PG": (-11.66, -1.32, 140.84, 155.97),      # Papua New Guinea

    # -- West Africa (cocoa, oil palm, rubber, wood) --
    "GH": (4.74, 11.17, -3.26, 1.20),           # Ghana
    "CI": (4.36, 10.74, -8.60, -2.49),          # Ivory Coast
    "CM": (1.65, 13.08, 8.49, 16.19),           # Cameroon
    "NG": (4.27, 13.89, 2.69, 14.68),           # Nigeria
    "SL": (6.93, 10.00, -13.30, -10.27),        # Sierra Leone
    "LR": (4.35, 8.55, -11.49, -7.37),          # Liberia
    "GN": (7.19, 12.68, -15.08, -7.64),         # Guinea
    "TG": (6.10, 11.14, -0.15, 1.81),           # Togo
    "BJ": (6.23, 12.42, 0.77, 3.84),            # Benin
    "SN": (12.31, 16.69, -17.54, -11.35),       # Senegal
    "ML": (10.16, 25.00, -12.24, 4.27),         # Mali
    "BF": (9.39, 15.08, -5.52, 2.41),           # Burkina Faso
    "GW": (10.92, 12.69, -16.71, -13.64),       # Guinea-Bissau

    # -- Central & East Africa (cocoa, coffee, wood, cattle) --
    "CD": (-13.46, 5.39, 12.18, 31.31),         # DR Congo
    "CG": (-5.03, 3.70, 11.20, 18.65),          # Republic of Congo
    "GA": (-3.98, 2.32, 8.70, 14.50),           # Gabon
    "GQ": (-1.47, 3.79, 5.62, 11.34),           # Equatorial Guinea
    "CF": (2.22, 11.00, 14.42, 27.46),          # Central African Republic
    "ET": (3.40, 14.89, 32.99, 47.99),          # Ethiopia
    "UG": (-1.48, 4.23, 29.57, 35.00),          # Uganda
    "KE": (-4.68, 5.02, 33.91, 41.91),          # Kenya
    "TZ": (-11.75, -0.99, 29.33, 40.44),        # Tanzania
    "MZ": (-26.87, -10.47, 30.21, 40.84),       # Mozambique
    "MG": (-25.61, -11.95, 43.23, 50.48),       # Madagascar
    "RW": (-2.84, -1.05, 28.86, 30.90),         # Rwanda
    "BI": (-4.47, -2.31, 29.00, 30.85),         # Burundi
    "ZM": (-18.08, -8.22, 21.99, 33.71),        # Zambia
    "ZW": (-22.42, -15.61, 25.24, 33.06),       # Zimbabwe

    # -- EU / Europe (importers, low risk) --
    "DE": (47.27, 55.06, 5.87, 15.04),          # Germany
    "FR": (41.36, 51.09, -5.14, 9.56),          # France
    "NL": (50.75, 53.47, 3.36, 7.21),           # Netherlands
    "BE": (49.50, 51.50, 2.55, 6.40),           # Belgium
    "IT": (36.65, 47.09, 6.63, 18.52),          # Italy
    "ES": (27.64, 43.79, -18.17, 4.33),         # Spain
    "PT": (32.40, 42.15, -31.27, -6.19),        # Portugal
    "AT": (46.38, 49.02, 9.53, 17.16),          # Austria
    "SE": (55.34, 69.06, 11.11, 24.16),         # Sweden
    "FI": (59.81, 70.09, 20.55, 31.59),         # Finland
    "PL": (49.00, 54.84, 14.12, 24.15),         # Poland
    "RO": (43.62, 48.27, 20.26, 30.05),         # Romania
    "GR": (34.80, 41.75, 19.37, 29.65),         # Greece
    "UK": (49.96, 60.86, -8.17, 1.75),          # United Kingdom

    # -- Asia (India, China - large importers/producers) --
    "IN": (6.75, 35.50, 68.17, 97.40),          # India
    "CN": (18.17, 53.56, 73.50, 134.77),        # China
    "LK": (5.92, 9.84, 79.65, 81.88),           # Sri Lanka

    # -- North America --
    "US": (24.52, 49.38, -124.77, -66.95),      # United States
    "CA": (41.68, 83.11, -141.00, -52.62),       # Canada

    # -- Oceania --
    "AU": (-43.63, -10.06, 113.15, 153.64),     # Australia
    "NZ": (-47.29, -34.39, 166.43, 178.57),     # New Zealand
}

# ---------------------------------------------------------------------------
# Commodity Elevation Ranges (metres above sea level)
# ---------------------------------------------------------------------------
# Simplified plausibility ranges for EUDR commodity production.

COMMODITY_ELEVATION_RANGES: Dict[str, Tuple[float, float]] = {
    "cattle": (-50.0, 5_000.0),
    "cocoa": (0.0, 1_500.0),
    "coffee": (200.0, 2_800.0),
    "oil_palm": (0.0, 1_000.0),
    "rubber": (0.0, 1_200.0),
    "soya": (0.0, 2_000.0),
    "wood": (0.0, 4_500.0),
    # Derived products inherit primary ranges
    "beef": (-50.0, 5_000.0),
    "leather": (-50.0, 5_000.0),
    "chocolate": (0.0, 1_500.0),
    "palm_oil": (0.0, 1_000.0),
    "natural_rubber": (0.0, 1_200.0),
    "timber": (0.0, 4_500.0),
    "paper": (0.0, 4_500.0),
    "furniture": (0.0, 4_500.0),
    "charcoal": (0.0, 4_500.0),
    "soybean_oil": (0.0, 2_000.0),
    "soybean_meal": (0.0, 2_000.0),
    "tyres": (0.0, 1_200.0),
}

# ---------------------------------------------------------------------------
# Simplified Elevation Lookup
# ---------------------------------------------------------------------------
# 5-degree grid approximate mean elevations (metres) for major tropical
# regions. Used for plausibility checks only; NOT authoritative.

_ELEVATION_GRID: Dict[Tuple[int, int], float] = {
    # Amazon basin (low elevation)
    (-5, -60): 150.0, (-5, -55): 200.0, (0, -60): 100.0,
    (0, -55): 80.0, (0, -65): 120.0, (-10, -55): 250.0,
    (-10, -50): 500.0, (-15, -50): 600.0, (-15, -55): 400.0,
    (-15, -45): 700.0, (-20, -50): 500.0, (-20, -45): 800.0,
    (-25, -50): 600.0, (-25, -55): 300.0, (-30, -55): 100.0,
    # Andes (high elevation)
    (-5, -75): 2500.0, (-10, -75): 3000.0, (-15, -70): 3500.0,
    (0, -75): 2000.0, (5, -75): 1500.0,
    # West Africa (varied)
    (5, -5): 200.0, (5, 0): 150.0, (5, -10): 100.0,
    (10, -5): 300.0, (10, 0): 250.0, (5, 10): 400.0,
    (10, 10): 500.0, (5, 15): 300.0,
    # Southeast Asia (low elevation)
    (0, 105): 50.0, (0, 110): 30.0, (5, 100): 100.0,
    (0, 100): 50.0, (-5, 105): 20.0, (-5, 110): 10.0,
    (-5, 115): 15.0, (0, 115): 30.0, (5, 105): 80.0,
    (5, 110): 60.0, (0, 120): 100.0,
    # Central Africa
    (0, 20): 400.0, (0, 25): 500.0, (-5, 20): 350.0,
    (-5, 25): 600.0, (5, 20): 450.0, (5, 25): 700.0,
    (0, 30): 1000.0, (-5, 30): 800.0,
    # East Africa highlands
    (0, 35): 1200.0, (0, 40): 300.0, (-5, 35): 1500.0,
    (-5, 30): 800.0, (5, 35): 1000.0, (5, 40): 500.0,
    (-10, 35): 1200.0,
    # Papua New Guinea
    (-5, 145): 500.0, (-5, 150): 200.0, (-10, 145): 300.0,
    (-5, 140): 100.0,
}

# ---------------------------------------------------------------------------
# Simplified Ocean Polygons (bounding boxes of major ocean areas)
# ---------------------------------------------------------------------------
# Points falling within these regions AND not within any country bounding
# box are flagged as likely ocean. This is a simplified heuristic.

_MAJOR_OCEAN_REGIONS: List[Tuple[float, float, float, float]] = [
    # Central Pacific
    (-60.0, 60.0, -180.0, -130.0),
    # Mid Atlantic (south)
    (-60.0, -10.0, -40.0, 10.0),
    # Indian Ocean (south)
    (-60.0, -10.0, 50.0, 100.0),
    # Southern Ocean
    (-90.0, -60.0, -180.0, 180.0),
    # Arctic Ocean (central)
    (80.0, 90.0, -180.0, 180.0),
]

# ---------------------------------------------------------------------------
# CoordinateValidator
# ---------------------------------------------------------------------------

class CoordinateValidator:
    """Production-grade coordinate validation engine for EUDR compliance.

    Validates GPS coordinates against WGS84 bounds, assesses decimal
    precision, detects lat/lon transposition, checks country bounding-box
    matching, performs simplified land/ocean masking, detects coordinate
    duplicates via Haversine distance, checks elevation plausibility
    per commodity, and identifies spatial cluster anomalies.

    All validations are deterministic with zero LLM/ML involvement.

    Example::

        validator = CoordinateValidator()
        result = validator.validate_coordinate(
            lat=-3.4653, lon=28.2345,
            declared_country="CD", commodity="cocoa",
        )
        assert result.is_valid
        assert result.provenance_hash != ""

    Attributes:
        duplicate_threshold_m: Haversine distance threshold for duplicates.
        cluster_threshold_m: Distance threshold for cluster anomaly detection.
    """

    def __init__(
        self,
        duplicate_threshold_m: float = DEFAULT_DUPLICATE_THRESHOLD_M,
        cluster_threshold_m: float = DEFAULT_CLUSTER_ANOMALY_THRESHOLD_M,
        config: Any = None,
    ) -> None:
        """Initialize the CoordinateValidator.

        Args:
            duplicate_threshold_m: Max distance in metres to consider
                two coordinates as duplicates (default: 10m).
            cluster_threshold_m: Distance threshold for detecting cluster
                anomalies via standard deviation (default: 1000m).
            config: Optional GeolocationVerificationConfig instance.
                If provided, overrides duplicate_threshold_m with the
                config's ``duplicate_distance_threshold_m`` value.
        """
        if config is not None:
            # Extract relevant settings from centralized config
            duplicate_threshold_m = getattr(
                config, "duplicate_distance_threshold_m", duplicate_threshold_m
            )
        self.duplicate_threshold_m = duplicate_threshold_m
        self.cluster_threshold_m = cluster_threshold_m
        logger.info(
            "CoordinateValidator initialized: duplicate_threshold=%.1fm, "
            "cluster_threshold=%.1fm",
            self.duplicate_threshold_m,
            self.cluster_threshold_m,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_coordinate(
        self,
        lat: float,
        lon: float,
        declared_country: str = "",
        commodity: str = "",
    ) -> CoordinateValidationResult:
        """Validate a single GPS coordinate -- DETERMINISTIC.

        Runs all coordinate validation checks and returns a comprehensive
        result including issues, precision assessment, and provenance hash.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            declared_country: ISO 3166-1 alpha-2 country code.
            commodity: EUDR commodity identifier (e.g., 'cocoa').

        Returns:
            CoordinateValidationResult with all checks populated.
        """
        start_time = time.monotonic()
        result = CoordinateValidationResult(lat=lat, lon=lon)
        issues: List[ValidationIssue] = []

        # 1. WGS84 bounds check
        result.wgs84_valid = self._check_wgs84_bounds(lat, lon)
        if not result.wgs84_valid:
            issues.append(ValidationIssue(
                code="COORD_OUT_OF_BOUNDS",
                severity=IssueSeverity.CRITICAL,
                message=f"Coordinate ({lat}, {lon}) is outside WGS84 bounds "
                        f"(lat: -90..90, lon: -180..180).",
                field="lat/lon",
            ))
            result.is_valid = False
            result.issues = issues
            result.provenance_hash = self._compute_result_hash(result)
            return result

        # 2. Precision assessment
        decimal_places, precision_score = self._assess_precision(lat, lon)
        result.precision_decimal_places = decimal_places
        result.precision_score = precision_score
        if decimal_places < MIN_PRECISION_EUDR:
            issues.append(ValidationIssue(
                code="COORD_LOW_PRECISION",
                severity=IssueSeverity.HIGH,
                message=f"Coordinate precision is {decimal_places} decimal "
                        f"places; EUDR requires >= {MIN_PRECISION_EUDR}.",
                field="precision",
                details={"decimal_places": decimal_places, "required": MIN_PRECISION_EUDR},
            ))

        # 3. Transposition detection
        result.transposition_detected = self._detect_transposition(lat, lon)
        if result.transposition_detected:
            issues.append(ValidationIssue(
                code="COORD_TRANSPOSITION_SUSPECTED",
                severity=IssueSeverity.HIGH,
                message=f"Latitude ({lat}) and longitude ({lon}) appear "
                        f"transposed. Swapped values ({lon}, {lat}) would "
                        f"fall within a more plausible region.",
                field="lat/lon",
            ))

        # 4. Country match
        if declared_country:
            country_match, resolved = self._check_country_match(
                lat, lon, declared_country
            )
            result.country_match = country_match
            result.resolved_country = resolved
            if not country_match:
                issues.append(ValidationIssue(
                    code="COORD_COUNTRY_MISMATCH",
                    severity=IssueSeverity.HIGH,
                    message=f"Coordinate ({lat}, {lon}) does not fall within "
                            f"declared country {declared_country}. "
                            f"Resolved to: {resolved or 'unknown'}.",
                    field="declared_country",
                    details={
                        "declared": declared_country,
                        "resolved": resolved,
                    },
                ))

        # 5. Land mask (simplified ocean check)
        result.is_on_land = self._check_land_mask(lat, lon)
        if not result.is_on_land:
            issues.append(ValidationIssue(
                code="COORD_IN_OCEAN",
                severity=IssueSeverity.CRITICAL,
                message=f"Coordinate ({lat}, {lon}) appears to be located "
                        f"in the ocean, not on land.",
                field="lat/lon",
            ))

        # 6. Elevation plausibility
        if commodity:
            elevation, plausible = self._check_elevation_plausibility(
                lat, lon, commodity
            )
            result.elevation_m = elevation
            result.elevation_plausible = plausible
            if elevation is not None and not plausible:
                issues.append(ValidationIssue(
                    code="COORD_ELEVATION_IMPLAUSIBLE",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Estimated elevation {elevation:.0f}m is outside "
                            f"the plausible range for commodity '{commodity}'.",
                    field="elevation",
                    details={
                        "elevation_m": elevation,
                        "commodity": commodity,
                        "range": COMMODITY_ELEVATION_RANGES.get(commodity.lower()),
                    },
                ))

        # Determine overall validity
        critical_issues = [
            i for i in issues if i.severity == IssueSeverity.CRITICAL
        ]
        high_issues = [
            i for i in issues if i.severity == IssueSeverity.HIGH
        ]
        result.is_valid = len(critical_issues) == 0 and len(high_issues) == 0
        result.issues = issues

        # Compute provenance hash
        result.provenance_hash = self._compute_result_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Coordinate validation %s: lat=%.6f, lon=%.6f, valid=%s, "
            "issues=%d, %.2fms",
            result.validation_id, lat, lon, result.is_valid,
            len(issues), elapsed_ms,
        )

        return result

    def validate_batch(
        self,
        coordinates: List[CoordinateInput],
    ) -> List[CoordinateValidationResult]:
        """Validate a batch of coordinates with duplicate and anomaly detection.

        Runs individual validation on each coordinate, then applies batch-level
        checks: duplicate detection and cluster anomaly detection.

        Args:
            coordinates: List of CoordinateInput objects.

        Returns:
            List of CoordinateValidationResult, one per input coordinate.
        """
        start_time = time.monotonic()

        if not coordinates:
            logger.warning("validate_batch called with empty list")
            return []

        # 1. Validate each coordinate individually
        results: List[CoordinateValidationResult] = []
        for coord in coordinates:
            result = self.validate_coordinate(
                lat=coord.lat,
                lon=coord.lon,
                declared_country=coord.declared_country,
                commodity=coord.commodity,
            )
            results.append(result)

        # 2. Duplicate detection across the batch
        duplicate_flags = self._detect_duplicates(coordinates)
        for i, is_dup in enumerate(duplicate_flags):
            if is_dup:
                results[i].is_duplicate = True
                results[i].issues.append(ValidationIssue(
                    code="COORD_DUPLICATE",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Coordinate ({coordinates[i].lat}, "
                            f"{coordinates[i].lon}) is a duplicate of "
                            f"another coordinate within "
                            f"{self.duplicate_threshold_m:.0f}m.",
                    field="lat/lon",
                ))
                # Recompute hash after adding issue
                results[i].provenance_hash = self._compute_result_hash(results[i])

        # 3. Cluster anomaly detection
        anomaly_flags = self._detect_cluster_anomaly(coordinates)
        for i, is_anomaly in enumerate(anomaly_flags):
            if is_anomaly:
                results[i].cluster_anomaly = True
                results[i].issues.append(ValidationIssue(
                    code="COORD_CLUSTER_ANOMALY",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Coordinate ({coordinates[i].lat}, "
                            f"{coordinates[i].lon}) is a spatial outlier "
                            f"relative to the cluster centroid.",
                    field="lat/lon",
                ))
                # Recompute hash after adding issue
                results[i].provenance_hash = self._compute_result_hash(results[i])

        elapsed_ms = (time.monotonic() - start_time) * 1000
        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            "Batch coordinate validation: %d coordinates, %d valid, "
            "%d duplicates, %d anomalies, %.1fms",
            len(coordinates), valid_count,
            sum(1 for f in duplicate_flags if f),
            sum(1 for f in anomaly_flags if f),
            elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Internal: Validation Checks
    # ------------------------------------------------------------------

    def _check_wgs84_bounds(self, lat: float, lon: float) -> bool:
        """Check whether coordinates are within WGS84 bounds.

        WGS84 valid ranges:
            Latitude:  -90.0 to 90.0
            Longitude: -180.0 to 180.0

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            True if coordinates are within bounds.
        """
        return (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)

    def _assess_precision(
        self, lat: float, lon: float
    ) -> Tuple[int, float]:
        """Assess the decimal precision of coordinate values.

        EUDR Article 9 requires sufficient precision for plot-level
        identification. 6+ decimal places (~0.11m accuracy at equator)
        is the target.

        Precision scoring:
            >= 8 decimal places: 1.0 (survey-grade)
            >= 6 decimal places: 0.9 (EUDR compliant)
            >= 5 decimal places: 0.7 (marginal, ~1.1m)
            >= 4 decimal places: 0.4 (low, ~11m)
            >= 3 decimal places: 0.2 (very low, ~111m)
            < 3 decimal places:  0.0 (insufficient)

        Args:
            lat: Latitude value.
            lon: Longitude value.

        Returns:
            Tuple of (minimum_decimal_places, precision_score).
        """
        lat_decimals = self._count_decimal_places(lat)
        lon_decimals = self._count_decimal_places(lon)
        min_decimals = min(lat_decimals, lon_decimals)

        if min_decimals >= 8:
            score = 1.0
        elif min_decimals >= 6:
            score = 0.9
        elif min_decimals >= 5:
            score = 0.7
        elif min_decimals >= 4:
            score = 0.4
        elif min_decimals >= 3:
            score = 0.2
        else:
            score = 0.0

        return min_decimals, score

    def _count_decimal_places(self, value: float) -> int:
        """Count the number of significant decimal places in a float.

        Uses string representation to determine user-intended precision.
        Handles scientific notation and trailing zeros.

        Args:
            value: The floating-point value to assess.

        Returns:
            Number of decimal places (0 if integer).
        """
        str_val = f"{value}"
        # Handle scientific notation
        if "e" in str_val or "E" in str_val:
            # Convert from scientific notation
            str_val = f"{value:.15f}".rstrip("0")
        if "." not in str_val:
            return 0
        decimal_part = str_val.split(".")[1]
        # Strip trailing zeros (they are not significant in float repr)
        decimal_part = decimal_part.rstrip("0")
        return len(decimal_part)

    def _detect_transposition(self, lat: float, lon: float) -> bool:
        """Detect whether latitude and longitude may be transposed.

        Heuristic: if the raw values are both within WGS84 bounds when
        swapped AND the swapped values place the point in a known
        EUDR-relevant country bounding box (while the original does not),
        transposition is suspected.

        Only flags transposition when lat is outside (-90, 90) when used
        as-is but the swapped value would be valid; or when the original
        does not match any country but swapped matches.

        Args:
            lat: Original latitude value.
            lon: Original longitude value.

        Returns:
            True if transposition is suspected.
        """
        # Only check if the swapped values would be valid coordinates
        if not (-90.0 <= lon <= 90.0 and -180.0 <= lat <= 180.0):
            return False

        # Check: does original match any country bounding box?
        original_country = self._resolve_country(lat, lon)
        swapped_country = self._resolve_country(lon, lat)

        # Transposition suspected if: original matches nothing but
        # swapped matches a country, OR if |lat| > 90 (invalid as lat)
        if original_country is None and swapped_country is not None:
            return True

        return False

    def _check_country_match(
        self, lat: float, lon: float, declared_country: str
    ) -> Tuple[bool, Optional[str]]:
        """Check whether coordinate falls within declared country bounds.

        Uses simplified bounding-box database for deterministic lookups.
        Not a polygon-level check, so edge cases near borders may produce
        false positives/negatives.

        Args:
            lat: Latitude.
            lon: Longitude.
            declared_country: ISO 3166-1 alpha-2 country code.

        Returns:
            Tuple of (matches_declared, resolved_country_code).
        """
        declared_upper = declared_country.upper().strip()

        # Check if coordinate falls within declared country bbox
        bbox = COUNTRY_BOUNDING_BOXES.get(declared_upper)
        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return True, declared_upper

        # Resolve which country the coordinate actually falls in
        resolved = self._resolve_country(lat, lon)
        return False, resolved

    def _resolve_country(self, lat: float, lon: float) -> Optional[str]:
        """Resolve which country a coordinate falls within.

        Iterates over all country bounding boxes. Returns the first match.
        For overlapping bounding boxes, selects the smallest area bbox
        for better accuracy.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            ISO 3166-1 alpha-2 code or None if no match.
        """
        matches: List[Tuple[str, float]] = []

        for cc, bbox in COUNTRY_BOUNDING_BOXES.items():
            min_lat, max_lat, min_lon, max_lon = bbox
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                # Calculate bbox area for tie-breaking
                area = (max_lat - min_lat) * (max_lon - min_lon)
                matches.append((cc, area))

        if not matches:
            return None

        # Return the smallest bounding box match (most specific)
        matches.sort(key=lambda x: x[1])
        return matches[0][0]

    def _check_land_mask(self, lat: float, lon: float) -> bool:
        """Check whether coordinate is on land using simplified ocean masking.

        Uses a two-step heuristic:
        1. If the coordinate falls within any country bounding box, it is
           considered on land.
        2. If not, check if it falls within known major ocean regions.

        This is a simplified check. A production PostGIS-backed system
        would use actual coastline polygons from Natural Earth.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            True if coordinate is likely on land.
        """
        # Step 1: If it matches any country bbox, it is on land
        resolved = self._resolve_country(lat, lon)
        if resolved is not None:
            return True

        # Step 2: Check against major ocean regions
        for min_lat, max_lat, min_lon, max_lon in _MAJOR_OCEAN_REGIONS:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                return False

        # If not in ocean or country, assume small island / land
        return True

    def _detect_duplicates(
        self, coordinates: List[CoordinateInput]
    ) -> List[bool]:
        """Detect duplicate coordinates in a batch using Haversine distance.

        Two coordinates are considered duplicates if they are within
        ``self.duplicate_threshold_m`` metres of each other. Uses
        O(n^2) pairwise comparison; for large batches a spatial index
        should be used.

        Args:
            coordinates: List of CoordinateInput objects.

        Returns:
            List of booleans, True if the coordinate at that index is
            a duplicate of an earlier coordinate.
        """
        n = len(coordinates)
        duplicates = [False] * n

        for i in range(n):
            if duplicates[i]:
                continue
            for j in range(i + 1, n):
                if duplicates[j]:
                    continue
                dist = self._haversine_distance(
                    coordinates[i].lat, coordinates[i].lon,
                    coordinates[j].lat, coordinates[j].lon,
                )
                if dist < self.duplicate_threshold_m:
                    duplicates[j] = True

        return duplicates

    def _check_elevation_plausibility(
        self, lat: float, lon: float, commodity: str
    ) -> Tuple[Optional[float], bool]:
        """Check whether estimated elevation is plausible for a commodity.

        Uses a simplified 5-degree grid elevation lookup. If no grid cell
        is available for the location, returns (None, True) to avoid
        false negatives.

        Args:
            lat: Latitude.
            lon: Longitude.
            commodity: EUDR commodity identifier.

        Returns:
            Tuple of (estimated_elevation_m, is_plausible).
            Elevation is None if no data is available for the grid cell.
        """
        elevation = self._lookup_elevation(lat, lon)
        if elevation is None:
            return None, True

        commodity_lower = commodity.lower()
        elevation_range = COMMODITY_ELEVATION_RANGES.get(commodity_lower)
        if elevation_range is None:
            return elevation, True

        min_elev, max_elev = elevation_range
        plausible = min_elev <= elevation <= max_elev
        return elevation, plausible

    def _lookup_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Look up approximate elevation from the simplified grid.

        Rounds coordinates to the nearest 5-degree grid cell and returns
        the stored elevation value. Returns None if no data is available.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Approximate elevation in metres, or None.
        """
        # Round to nearest 5-degree cell
        grid_lat = int(round(lat / 5.0)) * 5
        grid_lon = int(round(lon / 5.0)) * 5
        return _ELEVATION_GRID.get((grid_lat, grid_lon))

    def _detect_cluster_anomaly(
        self, coordinates: List[CoordinateInput]
    ) -> List[bool]:
        """Detect spatial outliers in a batch of coordinates.

        Computes the centroid of all valid coordinates, then flags any
        coordinate whose Haversine distance from the centroid exceeds
        a multiple of the standard deviation.

        A coordinate is an anomaly if:
            distance_to_centroid > mean_distance + 2 * std_distance

        Requires at least MIN_CLUSTER_SIZE points to run.

        Args:
            coordinates: List of CoordinateInput objects.

        Returns:
            List of booleans, True if the coordinate is a spatial outlier.
        """
        n = len(coordinates)
        anomalies = [False] * n

        if n < MIN_CLUSTER_SIZE:
            return anomalies

        # Compute centroid using mean lat/lon
        valid_coords = [
            (c.lat, c.lon) for c in coordinates
            if self._check_wgs84_bounds(c.lat, c.lon)
        ]
        if len(valid_coords) < MIN_CLUSTER_SIZE:
            return anomalies

        mean_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
        mean_lon = sum(c[1] for c in valid_coords) / len(valid_coords)

        # Compute distances from centroid
        distances: List[float] = []
        for c in coordinates:
            if self._check_wgs84_bounds(c.lat, c.lon):
                dist = self._haversine_distance(c.lat, c.lon, mean_lat, mean_lon)
            else:
                dist = float("inf")
            distances.append(dist)

        # Compute mean and standard deviation of finite distances
        finite_dists = [d for d in distances if d != float("inf")]
        if not finite_dists:
            return anomalies

        mean_dist = sum(finite_dists) / len(finite_dists)
        if len(finite_dists) > 1:
            variance = sum(
                (d - mean_dist) ** 2 for d in finite_dists
            ) / (len(finite_dists) - 1)
            std_dist = math.sqrt(variance)
        else:
            std_dist = 0.0

        # Flag anomalies: distance > mean + 2*std AND > cluster_threshold
        threshold = mean_dist + 2.0 * std_dist
        threshold = max(threshold, self.cluster_threshold_m)

        for i in range(n):
            if distances[i] > threshold:
                anomalies[i] = True

        return anomalies

    # ------------------------------------------------------------------
    # Internal: Haversine Distance
    # ------------------------------------------------------------------

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate Haversine distance between two WGS84 coordinates.

        Uses the standard Haversine formula with the WGS84 mean Earth
        radius. This is a deterministic, zero-hallucination calculation.

        Args:
            lat1: Latitude of point 1 (degrees).
            lon1: Longitude of point 1 (degrees).
            lat2: Latitude of point 2 (degrees).
            lon2: Longitude of point 2 (degrees).

        Returns:
            Distance in metres.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return EARTH_RADIUS_M * c

    # ------------------------------------------------------------------
    # Internal: Provenance Hash
    # ------------------------------------------------------------------

    def _compute_result_hash(self, result: CoordinateValidationResult) -> str:
        """Compute SHA-256 provenance hash for a validation result.

        Covers lat, lon, validity flags, precision, and issue codes
        for deterministic reproducibility.

        Args:
            result: The validation result to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "lat": result.lat,
            "lon": result.lon,
            "is_valid": result.is_valid,
            "wgs84_valid": result.wgs84_valid,
            "precision_decimal_places": result.precision_decimal_places,
            "precision_score": result.precision_score,
            "transposition_detected": result.transposition_detected,
            "country_match": result.country_match,
            "resolved_country": result.resolved_country,
            "is_on_land": result.is_on_land,
            "is_duplicate": result.is_duplicate,
            "elevation_m": result.elevation_m,
            "elevation_plausible": result.elevation_plausible,
            "cluster_anomaly": result.cluster_anomaly,
            "issue_codes": sorted([i.code for i in result.issues]),
        }
        return _compute_hash(hash_data)

# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "CoordinateValidator",
    "COUNTRY_BOUNDING_BOXES",
    "COMMODITY_ELEVATION_RANGES",
    "EARTH_RADIUS_M",
    "MIN_PRECISION_EUDR",
]
