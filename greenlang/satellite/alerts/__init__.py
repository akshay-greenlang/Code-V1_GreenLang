"""
Deforestation Alert System.

Integrates with external forest monitoring systems:
- Global Forest Watch (GFW)
- GLAD (Global Land Analysis & Discovery) alerts
- RADD (Radar for Detecting Deforestation) alerts
- EUDR compliance assessment
"""

from greenlang.satellite.alerts.deforestation_alert import (
    DeforestationAlertSystem,
    GlobalForestWatchClient,
    DeforestationAlert,
    AlertAggregation,
    AlertSource,
    AlertSeverity,
    AlertConfidence,
    GeoPolygon,
    create_polygon_from_bbox,
    create_polygon_from_coordinates,
    AlertSystemError,
    APIError,
)

__all__ = [
    "DeforestationAlertSystem",
    "GlobalForestWatchClient",
    "DeforestationAlert",
    "AlertAggregation",
    "AlertSource",
    "AlertSeverity",
    "AlertConfidence",
    "GeoPolygon",
    "create_polygon_from_bbox",
    "create_polygon_from_coordinates",
    "AlertSystemError",
    "APIError",
]
