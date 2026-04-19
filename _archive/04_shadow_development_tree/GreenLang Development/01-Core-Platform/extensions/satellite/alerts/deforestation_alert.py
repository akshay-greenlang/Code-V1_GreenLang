"""
Deforestation Alert System for EUDR Compliance.

Integrates with external alert systems:
- Global Forest Watch (GFW) alerts
- GLAD (Global Land Analysis & Discovery) alerts
- RADD (Radar for Detecting Deforestation) alerts for tropics

Provides:
- Alert retrieval and aggregation
- Polygon-based alert filtering
- Alert severity classification
- EUDR compliance status assessment
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
import hashlib
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


class AlertSource(Enum):
    """External alert data sources."""
    GFW = "global_forest_watch"
    GLAD = "glad"
    RADD = "radd"
    INTERNAL = "internal"


class AlertSeverity(Enum):
    """Alert severity classification."""
    LOW = "low"           # < 0.5 ha
    MEDIUM = "medium"     # 0.5 - 5 ha
    HIGH = "high"         # 5 - 50 ha
    CRITICAL = "critical" # > 50 ha


class AlertConfidence(Enum):
    """Alert confidence level."""
    LOW = "low"
    NOMINAL = "nominal"
    HIGH = "high"


@dataclass
class GeoPolygon:
    """Geographic polygon for area of interest."""
    coordinates: list[tuple[float, float]]  # List of (lon, lat) tuples
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        """Validate polygon."""
        if len(self.coordinates) < 3:
            raise ValueError("Polygon must have at least 3 coordinates")
        # Ensure polygon is closed
        if self.coordinates[0] != self.coordinates[-1]:
            self.coordinates.append(self.coordinates[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Get bounding box (min_lon, min_lat, max_lon, max_lat)."""
        lons = [c[0] for c in self.coordinates]
        lats = [c[1] for c in self.coordinates]
        return (min(lons), min(lats), max(lons), max(lats))

    def to_wkt(self) -> str:
        """Convert to WKT string."""
        coords_str = ", ".join(f"{lon} {lat}" for lon, lat in self.coordinates)
        return f"POLYGON(({coords_str}))"

    def to_geojson(self) -> dict:
        """Convert to GeoJSON geometry."""
        return {
            "type": "Polygon",
            "coordinates": [self.coordinates]
        }

    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if polygon contains a point (ray casting algorithm)."""
        n = len(self.coordinates) - 1  # Don't count closing point
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.coordinates[i]
            xj, yj = self.coordinates[j]

            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def area_ha(self) -> float:
        """Calculate approximate area in hectares using shoelace formula."""
        n = len(self.coordinates) - 1
        if n < 3:
            return 0.0

        # Calculate centroid latitude for projection
        lat_avg = sum(c[1] for c in self.coordinates[:-1]) / n
        lat_rad = np.radians(lat_avg)

        # Convert to approximate meters
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(lat_rad)

        # Shoelace formula
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            xi = self.coordinates[i][0] * km_per_deg_lon
            yi = self.coordinates[i][1] * km_per_deg_lat
            xj = self.coordinates[j][0] * km_per_deg_lon
            yj = self.coordinates[j][1] * km_per_deg_lat
            area += xi * yj - xj * yi

        area_km2 = abs(area) / 2
        return area_km2 * 100  # Convert km2 to hectares


@dataclass
class DeforestationAlert:
    """Individual deforestation alert record."""
    alert_id: str
    source: AlertSource
    detection_date: datetime
    latitude: float
    longitude: float
    area_ha: float
    confidence: AlertConfidence
    severity: AlertSeverity
    alert_type: str  # "deforestation", "degradation", "fire", etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def calculate_severity(cls, area_ha: float) -> AlertSeverity:
        """Calculate severity based on area."""
        if area_ha < 0.5:
            return AlertSeverity.LOW
        elif area_ha < 5:
            return AlertSeverity.MEDIUM
        elif area_ha < 50:
            return AlertSeverity.HIGH
        else:
            return AlertSeverity.CRITICAL


@dataclass
class AlertAggregation:
    """Aggregated alerts for an area of interest."""
    polygon: GeoPolygon
    alerts: list[DeforestationAlert]
    aggregation_date: datetime
    date_range_start: datetime
    date_range_end: datetime

    @property
    def total_alerts(self) -> int:
        """Total number of alerts."""
        return len(self.alerts)

    @property
    def total_affected_area_ha(self) -> float:
        """Total affected area in hectares."""
        return sum(a.area_ha for a in self.alerts)

    @property
    def alerts_by_severity(self) -> dict[AlertSeverity, list[DeforestationAlert]]:
        """Group alerts by severity."""
        result: dict[AlertSeverity, list[DeforestationAlert]] = {
            sev: [] for sev in AlertSeverity
        }
        for alert in self.alerts:
            result[alert.severity].append(alert)
        return result

    @property
    def alerts_by_source(self) -> dict[AlertSource, list[DeforestationAlert]]:
        """Group alerts by source."""
        result: dict[AlertSource, list[DeforestationAlert]] = {}
        for alert in self.alerts:
            if alert.source not in result:
                result[alert.source] = []
            result[alert.source].append(alert)
        return result

    @property
    def has_critical_alerts(self) -> bool:
        """Check if any critical severity alerts exist."""
        return any(a.severity == AlertSeverity.CRITICAL for a in self.alerts)

    @property
    def high_confidence_alerts(self) -> list[DeforestationAlert]:
        """Get high confidence alerts only."""
        return [a for a in self.alerts if a.confidence == AlertConfidence.HIGH]


class AlertSystemError(Exception):
    """Base exception for alert system errors."""
    pass


class APIError(AlertSystemError):
    """External API error."""
    pass


class GlobalForestWatchClient:
    """
    Client for Global Forest Watch API.

    GFW provides near-real-time forest change monitoring data
    including GLAD and RADD alerts.
    """

    GFW_API_URL = "https://data-api.globalforestwatch.org"

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_mock: bool = False,
    ):
        """
        Initialize GFW client.

        Args:
            api_key: GFW API key
            use_mock: Use mock data instead of real API
        """
        self.api_key = api_key
        self.use_mock = use_mock

    def get_glad_alerts(
        self,
        polygon: GeoPolygon,
        start_date: datetime,
        end_date: datetime,
        min_confidence: AlertConfidence = AlertConfidence.NOMINAL,
    ) -> list[DeforestationAlert]:
        """
        Get GLAD alerts for a polygon.

        GLAD (Global Land Analysis & Discovery) provides weekly
        updated forest disturbance alerts at 30m resolution.

        Args:
            polygon: Area of interest
            start_date: Start of date range
            end_date: End of date range
            min_confidence: Minimum confidence level

        Returns:
            List of DeforestationAlert objects
        """
        if self.use_mock:
            return self._mock_glad_alerts(polygon, start_date, end_date, min_confidence)

        logger.info(f"Fetching GLAD alerts for {start_date.date()} to {end_date.date()}")

        # Stub for real API call
        # response = requests.post(
        #     f"{self.GFW_API_URL}/analysis/glad-alerts",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     json={
        #         "geometry": polygon.to_geojson(),
        #         "start_date": start_date.isoformat(),
        #         "end_date": end_date.isoformat(),
        #     }
        # )

        raise APIError(
            "Real GFW API not implemented. Use use_mock=True for testing."
        )

    def _mock_glad_alerts(
        self,
        polygon: GeoPolygon,
        start_date: datetime,
        end_date: datetime,
        min_confidence: AlertConfidence,
    ) -> list[DeforestationAlert]:
        """Generate mock GLAD alerts for testing."""
        bounds = polygon.bounds
        seed = int(hashlib.md5(polygon.to_wkt().encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        alerts = []
        current_date = start_date

        # Generate ~1-5 alerts per month
        while current_date <= end_date:
            n_alerts = rng.integers(0, 6)

            for _ in range(n_alerts):
                # Random location within bounds
                lon = rng.uniform(bounds[0], bounds[2])
                lat = rng.uniform(bounds[1], bounds[3])

                # Only include if within polygon
                if not polygon.contains_point(lon, lat):
                    continue

                # Random area (log-normal distribution)
                area_ha = float(rng.lognormal(0, 1.5))
                area_ha = min(area_ha, 100)  # Cap at 100 ha

                # Confidence
                conf_val = rng.random()
                if conf_val > 0.7:
                    confidence = AlertConfidence.HIGH
                elif conf_val > 0.3:
                    confidence = AlertConfidence.NOMINAL
                else:
                    confidence = AlertConfidence.LOW

                # Filter by minimum confidence
                conf_order = [AlertConfidence.LOW, AlertConfidence.NOMINAL, AlertConfidence.HIGH]
                if conf_order.index(confidence) < conf_order.index(min_confidence):
                    continue

                alert_id = f"GLAD_{current_date.strftime('%Y%m%d')}_{rng.integers(10000, 99999)}"

                alerts.append(DeforestationAlert(
                    alert_id=alert_id,
                    source=AlertSource.GLAD,
                    detection_date=current_date + timedelta(days=rng.integers(0, 7)),
                    latitude=lat,
                    longitude=lon,
                    area_ha=round(area_ha, 3),
                    confidence=confidence,
                    severity=DeforestationAlert.calculate_severity(area_ha),
                    alert_type="deforestation",
                    metadata={
                        "data_source": "GLAD",
                        "resolution_m": 30,
                        "satellite": "Landsat",
                    }
                ))

            current_date += timedelta(days=7)  # Weekly updates

        logger.info(f"Generated {len(alerts)} mock GLAD alerts")
        return alerts

    def get_radd_alerts(
        self,
        polygon: GeoPolygon,
        start_date: datetime,
        end_date: datetime,
        min_confidence: AlertConfidence = AlertConfidence.NOMINAL,
    ) -> list[DeforestationAlert]:
        """
        Get RADD alerts for a polygon.

        RADD (Radar for Detecting Deforestation) uses Sentinel-1 SAR
        for deforestation detection in tropical regions, working
        through cloud cover.

        Args:
            polygon: Area of interest
            start_date: Start of date range
            end_date: End of date range
            min_confidence: Minimum confidence level

        Returns:
            List of DeforestationAlert objects
        """
        if self.use_mock:
            return self._mock_radd_alerts(polygon, start_date, end_date, min_confidence)

        raise APIError(
            "Real GFW API not implemented. Use use_mock=True for testing."
        )

    def _mock_radd_alerts(
        self,
        polygon: GeoPolygon,
        start_date: datetime,
        end_date: datetime,
        min_confidence: AlertConfidence,
    ) -> list[DeforestationAlert]:
        """Generate mock RADD alerts for testing."""
        bounds = polygon.bounds

        # RADD only covers tropics (-23.5 to 23.5 latitude)
        center_lat = (bounds[1] + bounds[3]) / 2
        if abs(center_lat) > 23.5:
            logger.info("Area outside RADD coverage (tropics only)")
            return []

        seed = int(hashlib.md5(f"RADD_{polygon.to_wkt()}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        alerts = []
        current_date = start_date

        # RADD provides more frequent updates (~daily) but similar alert density
        while current_date <= end_date:
            n_alerts = rng.integers(0, 3)

            for _ in range(n_alerts):
                lon = rng.uniform(bounds[0], bounds[2])
                lat = rng.uniform(bounds[1], bounds[3])

                if not polygon.contains_point(lon, lat):
                    continue

                area_ha = float(rng.lognormal(-0.5, 1.2))
                area_ha = min(area_ha, 50)

                conf_val = rng.random()
                if conf_val > 0.6:
                    confidence = AlertConfidence.HIGH
                elif conf_val > 0.2:
                    confidence = AlertConfidence.NOMINAL
                else:
                    confidence = AlertConfidence.LOW

                conf_order = [AlertConfidence.LOW, AlertConfidence.NOMINAL, AlertConfidence.HIGH]
                if conf_order.index(confidence) < conf_order.index(min_confidence):
                    continue

                alert_id = f"RADD_{current_date.strftime('%Y%m%d')}_{rng.integers(10000, 99999)}"

                alerts.append(DeforestationAlert(
                    alert_id=alert_id,
                    source=AlertSource.RADD,
                    detection_date=current_date,
                    latitude=lat,
                    longitude=lon,
                    area_ha=round(area_ha, 3),
                    confidence=confidence,
                    severity=DeforestationAlert.calculate_severity(area_ha),
                    alert_type="deforestation",
                    metadata={
                        "data_source": "RADD",
                        "resolution_m": 10,
                        "satellite": "Sentinel-1",
                        "method": "SAR",
                    }
                ))

            current_date += timedelta(days=1)

        logger.info(f"Generated {len(alerts)} mock RADD alerts")
        return alerts


class DeforestationAlertSystem:
    """
    Unified alert system for deforestation monitoring.

    Aggregates alerts from multiple sources and provides
    EUDR compliance assessment.
    """

    def __init__(
        self,
        gfw_client: Optional[GlobalForestWatchClient] = None,
        use_mock: bool = False,
    ):
        """
        Initialize alert system.

        Args:
            gfw_client: GFW API client
            use_mock: Use mock data for all sources
        """
        self.gfw_client = gfw_client or GlobalForestWatchClient(use_mock=use_mock)
        self.use_mock = use_mock

    def get_alerts_for_polygon(
        self,
        polygon: GeoPolygon,
        start_date: datetime,
        end_date: datetime,
        sources: Optional[list[AlertSource]] = None,
        min_confidence: AlertConfidence = AlertConfidence.NOMINAL,
    ) -> AlertAggregation:
        """
        Get all alerts for a polygon from specified sources.

        Args:
            polygon: Area of interest
            start_date: Start of date range
            end_date: End of date range
            sources: Alert sources to query (default: all)
            min_confidence: Minimum confidence level

        Returns:
            AlertAggregation with all alerts
        """
        if sources is None:
            sources = [AlertSource.GLAD, AlertSource.RADD]

        logger.info(
            f"Fetching alerts for polygon ({polygon.area_ha():.2f} ha), "
            f"sources: {[s.value for s in sources]}"
        )

        all_alerts = []

        if AlertSource.GLAD in sources:
            try:
                glad_alerts = self.gfw_client.get_glad_alerts(
                    polygon, start_date, end_date, min_confidence
                )
                all_alerts.extend(glad_alerts)
            except APIError as e:
                logger.warning(f"Failed to fetch GLAD alerts: {e}")

        if AlertSource.RADD in sources:
            try:
                radd_alerts = self.gfw_client.get_radd_alerts(
                    polygon, start_date, end_date, min_confidence
                )
                all_alerts.extend(radd_alerts)
            except APIError as e:
                logger.warning(f"Failed to fetch RADD alerts: {e}")

        # Sort by date
        all_alerts.sort(key=lambda a: a.detection_date)

        return AlertAggregation(
            polygon=polygon,
            alerts=all_alerts,
            aggregation_date=datetime.now(),
            date_range_start=start_date,
            date_range_end=end_date,
        )

    def add_internal_alert(
        self,
        aggregation: AlertAggregation,
        alert: DeforestationAlert,
    ) -> None:
        """
        Add an internally-detected alert to aggregation.

        Args:
            aggregation: Existing alert aggregation
            alert: New alert to add
        """
        alert.source = AlertSource.INTERNAL
        aggregation.alerts.append(alert)
        aggregation.alerts.sort(key=lambda a: a.detection_date)

    def assess_eudr_compliance(
        self,
        aggregation: AlertAggregation,
        cutoff_date: datetime = datetime(2020, 12, 31),
    ) -> dict[str, Any]:
        """
        Assess EUDR compliance based on alerts.

        EUDR requires that products are not from land deforested
        after December 31, 2020.

        Args:
            aggregation: Alert aggregation to assess
            cutoff_date: EUDR cutoff date (default: Dec 31, 2020)

        Returns:
            Dict with compliance assessment
        """
        post_cutoff_alerts = [
            a for a in aggregation.alerts
            if a.detection_date > cutoff_date
        ]

        high_conf_post_cutoff = [
            a for a in post_cutoff_alerts
            if a.confidence == AlertConfidence.HIGH
        ]

        # Calculate affected area
        total_affected_ha = sum(a.area_ha for a in post_cutoff_alerts)
        high_conf_affected_ha = sum(a.area_ha for a in high_conf_post_cutoff)

        # Compliance status
        if len(post_cutoff_alerts) == 0:
            status = "COMPLIANT"
            risk_level = "LOW"
        elif len(high_conf_post_cutoff) == 0:
            status = "REVIEW_REQUIRED"
            risk_level = "MEDIUM"
        else:
            status = "NON_COMPLIANT"
            risk_level = "HIGH" if high_conf_affected_ha > 1.0 else "MEDIUM"

        assessment = {
            "compliance_status": status,
            "risk_level": risk_level,
            "assessment_date": datetime.now().isoformat(),
            "cutoff_date": cutoff_date.isoformat(),
            "polygon_area_ha": round(aggregation.polygon.area_ha(), 2),
            "analysis_period": {
                "start": aggregation.date_range_start.isoformat(),
                "end": aggregation.date_range_end.isoformat(),
            },
            "alerts_summary": {
                "total_alerts": aggregation.total_alerts,
                "post_cutoff_alerts": len(post_cutoff_alerts),
                "high_confidence_post_cutoff": len(high_conf_post_cutoff),
            },
            "affected_area": {
                "total_post_cutoff_ha": round(total_affected_ha, 3),
                "high_confidence_ha": round(high_conf_affected_ha, 3),
            },
            "alerts_by_severity": {
                sev.value: len(alerts)
                for sev, alerts in aggregation.alerts_by_severity.items()
            },
            "recommendations": [],
        }

        # Add recommendations
        if status == "NON_COMPLIANT":
            assessment["recommendations"].extend([
                "Detailed satellite imagery review required",
                "Ground verification recommended for affected areas",
                "Consider alternative sourcing from compliant areas",
            ])
        elif status == "REVIEW_REQUIRED":
            assessment["recommendations"].extend([
                "Review low/medium confidence alerts with additional data",
                "Consider historical imagery analysis for verification",
            ])

        return assessment

    def generate_alert_report(
        self,
        aggregation: AlertAggregation,
    ) -> dict[str, Any]:
        """
        Generate comprehensive alert report.

        Args:
            aggregation: Alert aggregation to report on

        Returns:
            Dict with full report data
        """
        report = {
            "report_date": datetime.now().isoformat(),
            "polygon": {
                "wkt": aggregation.polygon.to_wkt(),
                "area_ha": round(aggregation.polygon.area_ha(), 2),
                "bounds": aggregation.polygon.bounds,
            },
            "date_range": {
                "start": aggregation.date_range_start.isoformat(),
                "end": aggregation.date_range_end.isoformat(),
                "days": (aggregation.date_range_end - aggregation.date_range_start).days,
            },
            "summary": {
                "total_alerts": aggregation.total_alerts,
                "total_affected_area_ha": round(aggregation.total_affected_area_ha, 3),
                "has_critical_alerts": aggregation.has_critical_alerts,
                "high_confidence_count": len(aggregation.high_confidence_alerts),
            },
            "alerts_by_source": {
                source.value: {
                    "count": len(alerts),
                    "area_ha": round(sum(a.area_ha for a in alerts), 3),
                }
                for source, alerts in aggregation.alerts_by_source.items()
            },
            "alerts_by_severity": {
                sev.value: {
                    "count": len(alerts),
                    "area_ha": round(sum(a.area_ha for a in alerts), 3),
                }
                for sev, alerts in aggregation.alerts_by_severity.items()
            },
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "source": a.source.value,
                    "date": a.detection_date.isoformat(),
                    "location": {"lat": a.latitude, "lon": a.longitude},
                    "area_ha": a.area_ha,
                    "severity": a.severity.value,
                    "confidence": a.confidence.value,
                    "type": a.alert_type,
                }
                for a in aggregation.alerts
            ],
        }

        return report


def create_polygon_from_coordinates(
    coordinates: list[tuple[float, float]],
) -> GeoPolygon:
    """
    Create a GeoPolygon from coordinate list.

    Args:
        coordinates: List of (longitude, latitude) tuples

    Returns:
        GeoPolygon object
    """
    return GeoPolygon(coordinates=coordinates)


def create_polygon_from_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> GeoPolygon:
    """
    Create a rectangular GeoPolygon from bounding box.

    Args:
        min_lon: Minimum longitude
        min_lat: Minimum latitude
        max_lon: Maximum longitude
        max_lat: Maximum latitude

    Returns:
        GeoPolygon object
    """
    coordinates = [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),  # Close polygon
    ]
    return GeoPolygon(coordinates=coordinates)
