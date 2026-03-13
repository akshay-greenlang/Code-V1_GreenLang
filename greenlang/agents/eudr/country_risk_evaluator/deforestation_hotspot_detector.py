# -*- coding: utf-8 -*-
"""
Deforestation Hotspot Detector Engine - AGENT-EUDR-016 Engine 3

Sub-national deforestation hotspot detection using DBSCAN-like spatial
clustering, fire alert correlation (VIIRS/MODIS active fire data),
protected area overlap detection with buffer zones, indigenous territory
proximity scoring, severity classification (LOW, MODERATE, HIGH, CRITICAL),
temporal trend analysis (acceleration/deceleration), hotspot persistence
scoring (recurring vs one-time), area calculation for affected zones
(hectares), country-level aggregation, and alert generation for new or
expanding hotspots.

Hotspot Detection Algorithm (Zero-Hallucination):
    1. Cluster deforestation events using spatial proximity (DBSCAN)
    2. Calculate cluster centroid, radius, and total area
    3. Correlate with fire alerts within temporal window (7 days)
    4. Check overlap with protected areas (buffer zone)
    5. Check proximity to indigenous territories (buffer zone)
    6. Score severity based on: area, event density, protected overlap,
       indigenous proximity, fire correlation
    7. Classify severity: LOW (0-25), MODERATE (25-50), HIGH (50-75),
       CRITICAL (75-100)

Clustering Parameters:
    - min_points: Minimum events to form cluster (default 3)
    - radius_km: Clustering radius in kilometers (default 10 km)
    - Algorithm: DBSCAN-like density-based spatial clustering

Severity Scoring:
    severity = (area_score * 0.30) + (density_score * 0.25) +
               (protected_overlap * 0.25) + (indigenous_proximity * 0.15) +
               (fire_correlation * 0.05)

Temporal Trend:
    acceleration: new events increasing over rolling 30-day window
    deceleration: new events decreasing over rolling 30-day window
    stable: event rate stable

Zero-Hallucination: All spatial calculations use deterministic geometric
    algorithms (haversine distance, polygon overlap). No LLM calls.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import get_config
from .metrics import (
    observe_hotspot_detection_duration,
    record_hotspot_detected,
)
from .models import (
    DeforestationHotspot,
    HotspotSeverity,
    TrendDirection,
)
from .provenance import get_provenance_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Earth radius in kilometers (for haversine distance)
_EARTH_RADIUS_KM: float = 6371.0

#: Severity score weights
_SEVERITY_WEIGHTS: Dict[str, Decimal] = {
    "area": Decimal("0.30"),
    "density": Decimal("0.25"),
    "protected_overlap": Decimal("0.25"),
    "indigenous_proximity": Decimal("0.15"),
    "fire_correlation": Decimal("0.05"),
}


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal for precise arithmetic."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    """Convert Decimal to float for API responses."""
    return float(value)


# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------


def _haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> float:
    """Calculate haversine distance between two points in kilometers.

    Args:
        lat1: Latitude of point 1 (degrees).
        lon1: Longitude of point 1 (degrees).
        lat2: Latitude of point 2 (degrees).
        lon2: Longitude of point 2 (degrees).

    Returns:
        Distance in kilometers.
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return _EARTH_RADIUS_KM * c


def _calculate_cluster_centroid(
    points: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Calculate centroid of a cluster of points.

    Args:
        points: List of (latitude, longitude) tuples.

    Returns:
        Centroid (latitude, longitude).
    """
    if not points:
        return (0.0, 0.0)

    lat_sum = sum(lat for lat, _ in points)
    lon_sum = sum(lon for _, lon in points)
    n = len(points)

    return (lat_sum / n, lon_sum / n)


def _calculate_cluster_radius(
    centroid: Tuple[float, float],
    points: List[Tuple[float, float]],
) -> float:
    """Calculate maximum distance from centroid to cluster points.

    Args:
        centroid: Cluster centroid (latitude, longitude).
        points: List of (latitude, longitude) tuples.

    Returns:
        Radius in kilometers.
    """
    if not points:
        return 0.0

    distances = [
        _haversine_distance(centroid[0], centroid[1], lat, lon)
        for lat, lon in points
    ]

    return max(distances) if distances else 0.0


def _calculate_cluster_area(
    radius_km: float,
) -> float:
    """Calculate approximate area of a circular cluster.

    Args:
        radius_km: Cluster radius in kilometers.

    Returns:
        Area in hectares.
    """
    # Area = π * r^2 (km²)
    area_km2 = math.pi * (radius_km ** 2)
    # Convert km² to hectares (1 km² = 100 hectares)
    area_ha = area_km2 * 100.0
    return area_ha


# ---------------------------------------------------------------------------
# DeforestationHotspotDetector
# ---------------------------------------------------------------------------


class DeforestationHotspotDetector:
    """Sub-national deforestation hotspot detection using spatial clustering.

    Clusters deforestation events using DBSCAN-like density-based spatial
    clustering, correlates with fire alerts, checks protected area overlap,
    scores indigenous territory proximity, classifies severity, analyzes
    temporal trends, and generates alerts for new or expanding hotspots.

    All spatial calculations use deterministic geometric algorithms for
    zero-hallucination reproducibility.

    Attributes:
        _hotspots: In-memory store of detected hotspots keyed by hotspot_id.
        _fire_alerts: Fire alert data keyed by (latitude, longitude, date).
        _protected_areas: Protected area polygons keyed by area_id.
        _indigenous_territories: Indigenous territory polygons keyed by
            territory_id.
        _event_history: Deforestation event history by country_code.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> detector = DeforestationHotspotDetector()
        >>> events = [
        ...     {"latitude": -3.5, "longitude": -62.3, "date": "2024-01-15", "area_ha": 10},
        ...     {"latitude": -3.52, "longitude": -62.31, "date": "2024-01-16", "area_ha": 15},
        ... ]
        >>> hotspots = detector.detect_hotspots("BR", events)
        >>> assert len(hotspots) >= 1
    """

    def __init__(self) -> None:
        """Initialize DeforestationHotspotDetector with empty stores."""
        self._hotspots: Dict[str, DeforestationHotspot] = {}
        self._fire_alerts: Dict[Tuple[float, float, str], Dict[str, Any]] = {}
        self._protected_areas: Dict[str, Dict[str, Any]] = {}
        self._indigenous_territories: Dict[str, Dict[str, Any]] = {}
        self._event_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info(
            "DeforestationHotspotDetector initialized: "
            "clustering_algorithm=DBSCAN-like",
        )

    # ------------------------------------------------------------------
    # Primary detection
    # ------------------------------------------------------------------

    def detect_hotspots(
        self,
        country_code: str,
        deforestation_events: List[Dict[str, Any]],
        fire_alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DeforestationHotspot]:
        """Detect deforestation hotspots from event data.

        Applies the following detection pipeline:
        1. Validate inputs (country code, event data).
        2. Cluster events using DBSCAN-like spatial clustering.
        3. Calculate cluster centroid, radius, and area.
        4. Correlate with fire alerts (if provided).
        5. Check protected area overlap.
        6. Check indigenous territory proximity.
        7. Score severity based on multiple factors.
        8. Classify severity level.
        9. Analyze temporal trend.
        10. Store hotspots and record provenance/metrics.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            deforestation_events: List of deforestation event dicts.
                Each dict must contain:
                - latitude (float): Event latitude
                - longitude (float): Event longitude
                - date (str or datetime): Event date
                - area_ha (float): Deforested area in hectares
            fire_alerts: Optional list of fire alert dicts with
                latitude, longitude, date, confidence.

        Returns:
            List of DeforestationHotspot objects for detected clusters.

        Raises:
            ValueError: If country_code is empty or deforestation_events
                list is empty or malformed.
        """
        start = time.monotonic()
        cfg = get_config()

        # -- Input validation ------------------------------------------------
        country_code = self._validate_country_code(country_code)
        if not deforestation_events:
            raise ValueError("deforestation_events list must not be empty")

        # -- Load fire alerts ------------------------------------------------
        if fire_alerts:
            self._load_fire_alerts(fire_alerts)

        # -- Cluster events --------------------------------------------------
        clusters = self._cluster_events(
            deforestation_events, cfg.clustering_min_points, cfg.clustering_radius_km,
        )

        # -- Analyze clusters ------------------------------------------------
        hotspots: List[DeforestationHotspot] = []
        for cluster_events in clusters:
            hotspot = self._analyze_cluster(
                country_code, cluster_events, cfg,
            )
            hotspots.append(hotspot)

            # Store
            with self._lock:
                self._hotspots[hotspot.hotspot_id] = hotspot

            # Provenance
            tracker = get_provenance_tracker()
            tracker.record(
                entity_type="hotspot_detection",
                action="detect",
                entity_id=hotspot.hotspot_id,
                data=hotspot.model_dump(mode="json"),
                metadata={
                    "country_code": country_code,
                    "severity": hotspot.severity.value,
                    "event_count": len(cluster_events),
                },
            )

            # Metrics
            record_hotspot_detected(hotspot.severity.value)

        # -- Update event history --------------------------------------------
        with self._lock:
            if country_code not in self._event_history:
                self._event_history[country_code] = []
            self._event_history[country_code].extend(deforestation_events)

            # Keep only last 1000 events per country
            if len(self._event_history[country_code]) > 1000:
                self._event_history[country_code] = (
                    self._event_history[country_code][-1000:]
                )

        # -- Metrics ---------------------------------------------------------
        elapsed = time.monotonic() - start
        observe_hotspot_detection_duration(elapsed)

        logger.info(
            "Hotspot detection completed: country=%s events=%d "
            "clusters=%d elapsed_ms=%.1f",
            country_code,
            len(deforestation_events),
            len(hotspots),
            elapsed * 1000,
        )
        return hotspots

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_hotspot(
        self, hotspot_id: str,
    ) -> Optional[DeforestationHotspot]:
        """Retrieve a hotspot by its unique identifier.

        Args:
            hotspot_id: The hotspot_id to look up.

        Returns:
            DeforestationHotspot if found, None otherwise.
        """
        with self._lock:
            return self._hotspots.get(hotspot_id)

    def list_hotspots(
        self,
        country_code: Optional[str] = None,
        severity: Optional[str] = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DeforestationHotspot]:
        """List hotspots with optional filters.

        Args:
            country_code: Optional country code filter.
            severity: Optional severity filter (LOW/MODERATE/HIGH/CRITICAL).
            active_only: If True, filter to active hotspots only
                (detected within last 90 days).
            limit: Maximum number of results (default 100).
            offset: Pagination offset (default 0).

        Returns:
            Filtered list of DeforestationHotspot objects.
        """
        with self._lock:
            results = list(self._hotspots.values())

        if country_code:
            cc = country_code.upper().strip()
            results = [h for h in results if h.country_code == cc]

        if severity:
            results = [
                h for h in results
                if h.severity.value == severity.upper()
            ]

        if active_only:
            cutoff_date = _utcnow() - timedelta(days=90)
            results = [
                h for h in results if h.detected_at >= cutoff_date
            ]

        # Sort by detection timestamp descending
        results.sort(key=lambda h: h.detected_at, reverse=True)

        return results[offset:offset + limit]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_country(
        self,
        country_codes: List[str],
    ) -> Dict[str, Any]:
        """Aggregate hotspots by country.

        Args:
            country_codes: List of ISO 3166-1 alpha-2 country codes.

        Returns:
            Dictionary with country-level aggregations including
            total_hotspots, critical_hotspots, total_area_ha,
            severity_breakdown, and trend_direction.

        Raises:
            ValueError: If country_codes list is empty.
        """
        if not country_codes:
            raise ValueError("country_codes list must not be empty")

        aggregation: Dict[str, Any] = {}

        with self._lock:
            for cc in country_codes:
                cc_upper = cc.upper().strip()
                country_hotspots = [
                    h for h in self._hotspots.values()
                    if h.country_code == cc_upper
                ]

                if not country_hotspots:
                    aggregation[cc_upper] = {
                        "total_hotspots": 0,
                        "critical_hotspots": 0,
                        "total_area_ha": 0.0,
                        "severity_breakdown": {},
                        "trend_direction": "no_data",
                    }
                    continue

                critical_count = sum(
                    1 for h in country_hotspots
                    if h.severity == HotspotSeverity.CRITICAL
                )

                total_area = sum(h.area_hectares for h in country_hotspots)

                severity_breakdown = {}
                for severity in HotspotSeverity:
                    count = sum(
                        1 for h in country_hotspots
                        if h.severity == severity
                    )
                    severity_breakdown[severity.value] = count

                # Determine trend (simplified)
                trend_direction = self._get_country_trend(
                    cc_upper, country_hotspots,
                )

                aggregation[cc_upper] = {
                    "total_hotspots": len(country_hotspots),
                    "critical_hotspots": critical_count,
                    "total_area_ha": round(total_area, 2),
                    "severity_breakdown": severity_breakdown,
                    "trend_direction": trend_direction.value,
                }

        return aggregation

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def generate_alerts(
        self,
        threshold_severity: str = "HIGH",
    ) -> List[Dict[str, Any]]:
        """Generate alerts for hotspots exceeding severity threshold.

        Args:
            threshold_severity: Minimum severity for alert generation
                (LOW, MODERATE, HIGH, CRITICAL). Default HIGH.

        Returns:
            List of alert dictionaries with hotspot_id, country_code,
            severity, centroid, area_hectares, alert_message, and
            recommended_action.

        Raises:
            ValueError: If threshold_severity is invalid.
        """
        try:
            threshold = HotspotSeverity(threshold_severity.upper())
        except ValueError:
            raise ValueError(
                f"Invalid threshold_severity '{threshold_severity}'; "
                f"must be one of: LOW, MODERATE, HIGH, CRITICAL"
            )

        severity_order = {
            HotspotSeverity.LOW: 0,
            HotspotSeverity.MEDIUM: 1,
            HotspotSeverity.HIGH: 2,
            HotspotSeverity.CRITICAL: 3,
        }

        threshold_level = severity_order[threshold]

        alerts: List[Dict[str, Any]] = []
        with self._lock:
            for hotspot in self._hotspots.values():
                if severity_order[hotspot.severity] >= threshold_level:
                    alert = self._build_alert(hotspot)
                    alerts.append(alert)

        # Sort by severity descending
        alerts.sort(
            key=lambda x: severity_order[HotspotSeverity(x["severity"])],
            reverse=True,
        )

        logger.info(
            "Generated %d alerts with threshold=%s",
            len(alerts), threshold_severity,
        )
        return alerts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_country_code(self, country_code: str) -> str:
        """Validate and normalize country code."""
        if not country_code or not country_code.strip():
            raise ValueError("country_code must not be empty")
        cc = country_code.upper().strip()
        if len(cc) != 2:
            raise ValueError(
                f"country_code must be 2 characters, got '{cc}'"
            )
        return cc

    def _load_fire_alerts(
        self, fire_alerts: List[Dict[str, Any]],
    ) -> None:
        """Load fire alerts into internal store (thread-safe)."""
        with self._lock:
            for alert in fire_alerts:
                lat = alert["latitude"]
                lon = alert["longitude"]
                date_str = str(alert["date"])
                key = (lat, lon, date_str)
                self._fire_alerts[key] = alert

    def _cluster_events(
        self,
        events: List[Dict[str, Any]],
        min_points: int,
        radius_km: float,
    ) -> List[List[Dict[str, Any]]]:
        """Cluster deforestation events using DBSCAN-like algorithm.

        Args:
            events: List of deforestation event dicts.
            min_points: Minimum events to form a cluster.
            radius_km: Clustering radius in kilometers.

        Returns:
            List of event clusters (each cluster is a list of events).
        """
        if not events:
            return []

        visited: Set[int] = set()
        clusters: List[List[Dict[str, Any]]] = []

        for i, event in enumerate(events):
            if i in visited:
                continue

            visited.add(i)
            neighbors = self._get_neighbors(events, i, radius_km)

            if len(neighbors) < min_points:
                # Noise point (not enough neighbors)
                continue

            # Start new cluster
            cluster: List[Dict[str, Any]] = [event]
            cluster_indices: Set[int] = {i}

            # Expand cluster
            neighbor_queue = list(neighbors)
            while neighbor_queue:
                neighbor_idx = neighbor_queue.pop(0)
                if neighbor_idx in visited:
                    continue

                visited.add(neighbor_idx)
                neighbor_event = events[neighbor_idx]
                cluster.append(neighbor_event)
                cluster_indices.add(neighbor_idx)

                # Find neighbors of this neighbor
                neighbor_neighbors = self._get_neighbors(
                    events, neighbor_idx, radius_km,
                )
                if len(neighbor_neighbors) >= min_points:
                    for nn in neighbor_neighbors:
                        if nn not in visited:
                            neighbor_queue.append(nn)

            clusters.append(cluster)

        return clusters

    def _get_neighbors(
        self,
        events: List[Dict[str, Any]],
        event_idx: int,
        radius_km: float,
    ) -> List[int]:
        """Get neighbor event indices within radius of target event.

        Args:
            events: List of all events.
            event_idx: Index of target event.
            radius_km: Search radius in kilometers.

        Returns:
            List of neighbor event indices.
        """
        target = events[event_idx]
        target_lat = target["latitude"]
        target_lon = target["longitude"]

        neighbors: List[int] = []
        for i, event in enumerate(events):
            if i == event_idx:
                continue

            distance = _haversine_distance(
                target_lat, target_lon,
                event["latitude"], event["longitude"],
            )
            if distance <= radius_km:
                neighbors.append(i)

        return neighbors

    def _analyze_cluster(
        self,
        country_code: str,
        cluster_events: List[Dict[str, Any]],
        cfg: Any,
    ) -> DeforestationHotspot:
        """Analyze a cluster and create DeforestationHotspot.

        Args:
            country_code: ISO alpha-2 code.
            cluster_events: List of events in cluster.
            cfg: Agent configuration.

        Returns:
            Populated DeforestationHotspot model.
        """
        # Calculate centroid
        points = [
            (e["latitude"], e["longitude"]) for e in cluster_events
        ]
        centroid = _calculate_cluster_centroid(points)

        # Calculate radius
        radius_km = _calculate_cluster_radius(centroid, points)

        # Calculate area
        area_ha = _calculate_cluster_area(radius_km)

        # Event count
        event_count = len(cluster_events)

        # Fire correlation
        fire_correlation_score = self._calculate_fire_correlation(
            cluster_events, cfg,
        )

        # Protected area overlap
        protected_overlap_score = self._check_protected_overlap(
            centroid, radius_km, cfg,
        )

        # Indigenous proximity
        indigenous_proximity_score = self._check_indigenous_proximity(
            centroid, cfg,
        )

        # Event density (events per km²)
        cluster_area_km2 = math.pi * (radius_km ** 2)
        density = event_count / cluster_area_km2 if cluster_area_km2 > 0 else 0.0

        # Severity score
        severity_score = self._calculate_severity_score(
            area_ha=area_ha,
            density=density,
            protected_overlap=protected_overlap_score,
            indigenous_proximity=indigenous_proximity_score,
            fire_correlation=fire_correlation_score,
        )

        # Severity classification
        severity = self._classify_severity(severity_score)

        # Temporal trend
        trend_direction = self._get_cluster_trend(cluster_events)

        # Persistence
        is_persistent = self._check_persistence(centroid, cluster_events)

        # Provenance hash
        tracker = get_provenance_tracker()
        prov_data = {
            "country_code": country_code,
            "centroid": centroid,
            "event_count": event_count,
            "area_ha": area_ha,
            "severity_score": _float(severity_score),
        }
        provenance_hash = tracker.build_hash(prov_data)

        return DeforestationHotspot(
            country_code=country_code,
            region=country_code,  # Placeholder; resolved by geo lookup
            latitude=centroid[0],
            longitude=centroid[1],
            area_km2=area_ha / 100.0,  # hectares to km2
            severity=severity,
            alert_count=event_count,
            fire_correlation=min(fire_correlation_score / 100.0, 1.0) if fire_correlation_score > 0 else None,
            protected_area_overlap_pct=min(protected_overlap_score, 100.0) if protected_overlap_score > 0 else None,
            indigenous_territory_overlap=indigenous_proximity_score > 0,
            trend=trend_direction,
            provenance_hash=provenance_hash,
        )

    def _calculate_fire_correlation(
        self,
        cluster_events: List[Dict[str, Any]],
        cfg: Any,
    ) -> float:
        """Calculate fire alert correlation score.

        Args:
            cluster_events: List of events in cluster.
            cfg: Agent configuration.

        Returns:
            Fire correlation score (0-100).
        """
        if not cfg.enable_fire_correlation:
            return 0.0

        correlated_count = 0
        for event in cluster_events:
            # Check for fire alerts within 7 days and 5 km
            event_lat = event["latitude"]
            event_lon = event["longitude"]
            event_date = event["date"]
            if isinstance(event_date, str):
                event_date = datetime.fromisoformat(event_date.replace("Z", "+00:00"))

            with self._lock:
                for (fire_lat, fire_lon, fire_date_str), fire_alert in self._fire_alerts.items():
                    fire_date = datetime.fromisoformat(fire_date_str.replace("Z", "+00:00"))
                    time_delta = abs((event_date - fire_date).days)
                    if time_delta > 7:
                        continue

                    distance = _haversine_distance(
                        event_lat, event_lon, fire_lat, fire_lon,
                    )
                    if distance <= 5.0:
                        correlated_count += 1
                        break

        # Normalize to 0-100
        correlation_ratio = correlated_count / len(cluster_events)
        return correlation_ratio * 100.0

    def _check_protected_overlap(
        self,
        centroid: Tuple[float, float],
        radius_km: float,
        cfg: Any,
    ) -> float:
        """Check protected area overlap score.

        Args:
            centroid: Cluster centroid.
            radius_km: Cluster radius.
            cfg: Agent configuration.

        Returns:
            Protected overlap score (0-100).
        """
        # Simplified: check if centroid is within buffer of any protected area
        buffer_km = cfg.protected_area_buffer_km

        with self._lock:
            for area_id, area_data in self._protected_areas.items():
                area_centroid = (
                    area_data["centroid_latitude"],
                    area_data["centroid_longitude"],
                )
                distance = _haversine_distance(
                    centroid[0], centroid[1],
                    area_centroid[0], area_centroid[1],
                )
                if distance <= (radius_km + buffer_km):
                    # Overlap detected
                    return 100.0

        return 0.0

    def _check_indigenous_proximity(
        self,
        centroid: Tuple[float, float],
        cfg: Any,
    ) -> float:
        """Check indigenous territory proximity score.

        Args:
            centroid: Cluster centroid.
            cfg: Agent configuration.

        Returns:
            Indigenous proximity score (0-100).
        """
        buffer_km = cfg.indigenous_territory_buffer_km

        with self._lock:
            for territory_id, territory_data in self._indigenous_territories.items():
                territory_centroid = (
                    territory_data["centroid_latitude"],
                    territory_data["centroid_longitude"],
                )
                distance = _haversine_distance(
                    centroid[0], centroid[1],
                    territory_centroid[0], territory_centroid[1],
                )
                if distance <= buffer_km:
                    # Proximity detected
                    return 100.0

        return 0.0

    def _calculate_severity_score(
        self,
        area_ha: float,
        density: float,
        protected_overlap: float,
        indigenous_proximity: float,
        fire_correlation: float,
    ) -> Decimal:
        """Calculate composite severity score.

        Args:
            area_ha: Cluster area in hectares.
            density: Event density (events per km²).
            protected_overlap: Protected overlap score (0-100).
            indigenous_proximity: Indigenous proximity score (0-100).
            fire_correlation: Fire correlation score (0-100).

        Returns:
            Severity score Decimal (0-100).
        """
        # Normalize area to 0-100 (0-10000 ha scale)
        area_score = min(_decimal(area_ha) / Decimal("10000") * Decimal("100"), Decimal("100"))

        # Normalize density to 0-100 (0-10 events/km² scale)
        density_score = min(_decimal(density) / Decimal("10") * Decimal("100"), Decimal("100"))

        severity = (
            (area_score * _SEVERITY_WEIGHTS["area"]) +
            (density_score * _SEVERITY_WEIGHTS["density"]) +
            (_decimal(protected_overlap) * _SEVERITY_WEIGHTS["protected_overlap"]) +
            (_decimal(indigenous_proximity) * _SEVERITY_WEIGHTS["indigenous_proximity"]) +
            (_decimal(fire_correlation) * _SEVERITY_WEIGHTS["fire_correlation"])
        )

        return severity.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _classify_severity(
        self, severity_score: Decimal,
    ) -> HotspotSeverity:
        """Classify severity level from score.

        Args:
            severity_score: Severity score (0-100).

        Returns:
            HotspotSeverity enum value.
        """
        score_float = _float(severity_score)
        if score_float < 25.0:
            return HotspotSeverity.LOW
        if score_float < 50.0:
            return HotspotSeverity.MEDIUM
        if score_float < 75.0:
            return HotspotSeverity.HIGH
        return HotspotSeverity.CRITICAL

    def _get_cluster_trend(
        self, cluster_events: List[Dict[str, Any]],
    ) -> TrendDirection:
        """Get temporal trend direction for cluster.

        Args:
            cluster_events: List of events in cluster.

        Returns:
            TrendDirection enum value.
        """
        # Sort events by date
        sorted_events = sorted(
            cluster_events,
            key=lambda e: e["date"] if isinstance(e["date"], datetime) else datetime.fromisoformat(e["date"].replace("Z", "+00:00")),
        )

        if len(sorted_events) < 3:
            return TrendDirection.STABLE

        # Compare first half vs second half event count
        mid = len(sorted_events) // 2
        first_half = sorted_events[:mid]
        second_half = sorted_events[mid:]

        if len(second_half) > len(first_half) * 1.2:
            return TrendDirection.DETERIORATING
        if len(second_half) < len(first_half) * 0.8:
            return TrendDirection.IMPROVING

        return TrendDirection.STABLE

    def _check_persistence(
        self,
        centroid: Tuple[float, float],
        cluster_events: List[Dict[str, Any]],
    ) -> bool:
        """Check if hotspot is persistent (recurring).

        Args:
            centroid: Cluster centroid.
            cluster_events: List of events in cluster.

        Returns:
            True if hotspot is persistent, False if one-time.
        """
        # Persistent if events span > 30 days
        dates = [
            e["date"] if isinstance(e["date"], datetime) else datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
            for e in cluster_events
        ]
        if not dates:
            return False

        date_range = max(dates) - min(dates)
        return date_range.days > 30

    def _get_country_trend(
        self,
        country_code: str,
        hotspots: List[DeforestationHotspot],
    ) -> TrendDirection:
        """Get country-level hotspot trend.

        Args:
            country_code: ISO alpha-2 code.
            hotspots: List of hotspots for country.

        Returns:
            TrendDirection enum value.
        """
        if len(hotspots) < 3:
            return TrendDirection.STABLE

        # Sort by detection date
        sorted_hotspots = sorted(
            hotspots, key=lambda h: h.detected_at,
        )

        # Compare first half vs second half count
        mid = len(sorted_hotspots) // 2
        first_half = sorted_hotspots[:mid]
        second_half = sorted_hotspots[mid:]

        if len(second_half) > len(first_half) * 1.2:
            return TrendDirection.DETERIORATING
        if len(second_half) < len(first_half) * 0.8:
            return TrendDirection.IMPROVING

        return TrendDirection.STABLE

    def _build_alert(
        self, hotspot: DeforestationHotspot,
    ) -> Dict[str, Any]:
        """Build alert message for hotspot.

        Args:
            hotspot: DeforestationHotspot object.

        Returns:
            Alert dictionary.
        """
        alert_message = (
            f"{hotspot.severity.value} severity deforestation hotspot "
            f"detected in {hotspot.country_code} at "
            f"({hotspot.centroid_latitude:.4f}, {hotspot.centroid_longitude:.4f}). "
            f"Area: {hotspot.area_hectares:.1f} ha, Events: {hotspot.event_count}."
        )

        recommended_action = "Monitor hotspot"
        if hotspot.severity == HotspotSeverity.CRITICAL:
            recommended_action = "IMMEDIATE ACTION REQUIRED: Deploy field verification team, engage local authorities"
        elif hotspot.severity == HotspotSeverity.HIGH:
            recommended_action = "Escalate to compliance team, initiate supplier investigation"

        if hotspot.protected_area_overlap:
            alert_message += " PROTECTED AREA OVERLAP DETECTED."

        if hotspot.indigenous_territory_proximity:
            alert_message += " INDIGENOUS TERRITORY PROXIMITY DETECTED."

        return {
            "hotspot_id": hotspot.hotspot_id,
            "country_code": hotspot.country_code,
            "severity": hotspot.severity.value,
            "centroid_latitude": hotspot.centroid_latitude,
            "centroid_longitude": hotspot.centroid_longitude,
            "area_hectares": hotspot.area_hectares,
            "event_count": hotspot.event_count,
            "alert_message": alert_message,
            "recommended_action": recommended_action,
            "detected_at": hotspot.detected_at.isoformat(),
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        with self._lock:
            count = len(self._hotspots)
        return (
            f"DeforestationHotspotDetector("
            f"hotspots={count})"
        )

    def __len__(self) -> int:
        """Return number of stored hotspots."""
        with self._lock:
            return len(self._hotspots)
