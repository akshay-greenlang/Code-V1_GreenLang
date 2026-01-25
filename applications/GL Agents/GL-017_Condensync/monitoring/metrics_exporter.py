# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Metrics Exporter Module
============================================

Exports Prometheus metrics via HTTP endpoint and supports JSON export
for Grafana dashboard generation and external monitoring systems.

Features:
- Prometheus /metrics HTTP endpoint
- JSON metrics export for debugging
- Grafana dashboard JSON generation
- Metrics snapshot capability
- Custom registry support for testing

Standards Compliance:
- OpenMetrics specification
- Prometheus exposition format
- Grafana dashboard schema

Example:
    >>> from monitoring.metrics_exporter import MetricsExporter
    >>> exporter = MetricsExporter()
    >>> exporter.start_server(port=9090)
    >>>
    >>> # Get JSON metrics
    >>> json_metrics = exporter.get_metrics_json()
    >>>
    >>> # Generate Grafana dashboard
    >>> dashboard = exporter.generate_grafana_dashboard()

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from prometheus_client import (
    REGISTRY,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    generate_latest,
    start_http_server,
)

from .metrics import CondenserMetrics, get_metrics_instance

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP REQUEST HANDLER
# =============================================================================

class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for metrics endpoint.

    Supports:
    - GET /metrics - Prometheus format
    - GET /metrics/json - JSON format
    - GET /health - Simple health check
    """

    def __init__(self, *args, exporter: "MetricsExporter" = None, **kwargs):
        """Initialize handler with exporter reference."""
        self.exporter = exporter
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Override to use Python logging."""
        logger.debug(f"MetricsHTTP: {format % args}")

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/metrics":
            self._serve_prometheus_metrics()
        elif self.path == "/metrics/json":
            self._serve_json_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self._serve_not_found()

    def _serve_prometheus_metrics(self) -> None:
        """Serve Prometheus format metrics."""
        try:
            metrics = generate_latest(REGISTRY)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(metrics)))
            self.end_headers()
            self.wfile.write(metrics)
        except Exception as e:
            logger.error(f"Error serving Prometheus metrics: {e}")
            self._serve_error(500, str(e))

    def _serve_json_metrics(self) -> None:
        """Serve JSON format metrics."""
        try:
            if self.exporter:
                json_data = self.exporter.get_metrics_json()
            else:
                json_data = {"error": "Exporter not available"}

            content = json.dumps(json_data, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving JSON metrics: {e}")
            self._serve_error(500, str(e))

    def _serve_health(self) -> None:
        """Serve simple health check."""
        content = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_not_found(self) -> None:
        """Serve 404 response."""
        content = json.dumps({"error": "Not found"}).encode("utf-8")
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_error(self, status: int, message: str) -> None:
        """Serve error response."""
        content = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


# =============================================================================
# METRICS EXPORTER
# =============================================================================

class MetricsExporter:
    """
    Prometheus metrics exporter with JSON and Grafana support.

    Provides HTTP endpoint for Prometheus scraping and utilities
    for JSON export and Grafana dashboard generation.

    Example:
        >>> exporter = MetricsExporter()
        >>> exporter.initialize("1.0.0", "production")
        >>> exporter.start_server(port=9090)
        >>>
        >>> # Later, generate dashboard
        >>> dashboard = exporter.generate_grafana_dashboard()
        >>> with open("dashboard.json", "w") as f:
        ...     json.dump(dashboard, f)

    Attributes:
        metrics: CondenserMetrics instance
        registry: Prometheus registry
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        metrics: Optional[CondenserMetrics] = None,
    ):
        """
        Initialize metrics exporter.

        Args:
            registry: Optional custom Prometheus registry
            metrics: Optional CondenserMetrics instance
        """
        self.registry = registry or REGISTRY
        self.metrics = metrics or get_metrics_instance()

        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._port: int = 9090

        self.version: str = "1.0.0"
        self.environment: str = "development"
        self._initialized: bool = False

        logger.debug("MetricsExporter instance created")

    def initialize(
        self,
        version: str,
        environment: str,
        instance_id: Optional[str] = None,
    ) -> None:
        """
        Initialize exporter with metadata.

        Args:
            version: Agent version
            environment: Deployment environment
            instance_id: Optional instance ID
        """
        self.version = version
        self.environment = environment

        # Also initialize metrics if not done
        if not self.metrics._initialized:
            self.metrics.initialize(version, environment, instance_id)

        self._initialized = True
        logger.info(
            f"MetricsExporter initialized: version={version}, "
            f"environment={environment}"
        )

    def start_server(
        self,
        port: int = 9090,
        address: str = "0.0.0.0",
    ) -> None:
        """
        Start HTTP server for metrics endpoint.

        Args:
            port: Port to listen on
            address: Address to bind to

        Raises:
            RuntimeError: If server is already running
        """
        if self._server is not None:
            raise RuntimeError("Metrics server is already running")

        try:
            # Use prometheus_client's built-in server for simplicity
            start_http_server(port=port, addr=address, registry=self.registry)
            self._port = port

            logger.info(
                f"Prometheus metrics server started on {address}:{port}/metrics"
            )

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def stop_server(self) -> None:
        """Stop the HTTP server if running."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            logger.info("Metrics server stopped")

    def get_metrics_prometheus(self) -> bytes:
        """
        Get metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics as bytes
        """
        return generate_latest(self.registry)

    def get_metrics_json(self) -> Dict[str, Any]:
        """
        Get metrics in JSON format.

        Returns:
            Dictionary with metrics data
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Parse Prometheus output to extract metrics
        prometheus_output = generate_latest(self.registry).decode("utf-8")
        metrics_data = self._parse_prometheus_output(prometheus_output)

        return {
            "timestamp": timestamp,
            "version": self.version,
            "environment": self.environment,
            "metrics": metrics_data,
            "summary": self.metrics.get_summary(),
        }

    def _parse_prometheus_output(
        self,
        output: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse Prometheus text format to structured data.

        Args:
            output: Prometheus text format output

        Returns:
            Dictionary of metric families with values
        """
        metrics: Dict[str, List[Dict[str, Any]]] = {}
        current_metric: Optional[str] = None

        for line in output.split("\n"):
            line = line.strip()

            if not line:
                continue

            # Skip comments except HELP and TYPE
            if line.startswith("# HELP"):
                parts = line.split(" ", 3)
                if len(parts) >= 3:
                    current_metric = parts[2]
                    if current_metric not in metrics:
                        metrics[current_metric] = []
                continue

            if line.startswith("# TYPE"):
                continue

            if line.startswith("#"):
                continue

            # Parse metric line
            try:
                # Handle metrics with labels
                if "{" in line:
                    name_labels, value = line.rsplit(" ", 1)
                    name, labels_str = name_labels.split("{", 1)
                    labels_str = labels_str.rstrip("}")

                    # Parse labels
                    labels = {}
                    for label in labels_str.split(","):
                        if "=" in label:
                            key, val = label.split("=", 1)
                            labels[key.strip()] = val.strip().strip('"')

                    if name not in metrics:
                        metrics[name] = []

                    metrics[name].append({
                        "labels": labels,
                        "value": float(value),
                    })
                else:
                    # Metric without labels
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        name = parts[0]
                        value = float(parts[1])

                        if name not in metrics:
                            metrics[name] = []

                        metrics[name].append({
                            "labels": {},
                            "value": value,
                        })

            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse metric line: {line} - {e}")
                continue

        return metrics

    def generate_grafana_dashboard(
        self,
        title: str = "GL-017 CONDENSYNC - Condenser Optimization",
        uid: str = "gl017-condensync",
    ) -> Dict[str, Any]:
        """
        Generate Grafana dashboard JSON.

        Args:
            title: Dashboard title
            uid: Dashboard UID

        Returns:
            Grafana dashboard JSON structure
        """
        return {
            "annotations": {"list": []},
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "liveNow": False,
            "panels": self._generate_dashboard_panels(),
            "refresh": "30s",
            "schemaVersion": 38,
            "tags": ["condensync", "gl-017", "greenlang", "condenser"],
            "templating": {
                "list": [
                    {
                        "current": {"text": "All", "value": "$__all"},
                        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                        "definition": "label_values(condensync_cleanliness_factor, condenser_id)",
                        "hide": 0,
                        "includeAll": True,
                        "label": "Condenser",
                        "multi": True,
                        "name": "condenser_id",
                        "options": [],
                        "query": {
                            "query": "label_values(condensync_cleanliness_factor, condenser_id)",
                            "refId": "StandardVariableQuery",
                        },
                        "refresh": 1,
                        "regex": "",
                        "skipUrlSync": False,
                        "sort": 1,
                        "type": "query",
                    },
                    {
                        "current": {"text": "All", "value": "$__all"},
                        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                        "definition": "label_values(condensync_cleanliness_factor, unit)",
                        "hide": 0,
                        "includeAll": True,
                        "label": "Unit",
                        "multi": True,
                        "name": "unit",
                        "options": [],
                        "query": {
                            "query": "label_values(condensync_cleanliness_factor, unit)",
                            "refId": "StandardVariableQuery",
                        },
                        "refresh": 1,
                        "regex": "",
                        "skipUrlSync": False,
                        "sort": 1,
                        "type": "query",
                    },
                ],
            },
            "time": {"from": "now-6h", "to": "now"},
            "timepicker": {},
            "timezone": "browser",
            "title": title,
            "uid": uid,
            "version": 1,
            "weekStart": "",
        }

    def _generate_dashboard_panels(self) -> List[Dict[str, Any]]:
        """
        Generate Grafana dashboard panels.

        Returns:
            List of panel configurations
        """
        panels = []
        y_pos = 0

        # Row: KPIs Overview
        panels.append(self._create_row_panel("Condenser KPIs", y_pos))
        y_pos += 1

        # Cleanliness Factor Gauge
        panels.append(self._create_gauge_panel(
            title="Cleanliness Factor",
            metric="condensync_cleanliness_factor",
            x=0, y=y_pos, w=6, h=6,
            thresholds={"steps": [
                {"color": "red", "value": None},
                {"color": "orange", "value": 0.6},
                {"color": "yellow", "value": 0.75},
                {"color": "green", "value": 0.85},
            ]},
            max_value=1.0,
            unit="percentunit",
        ))

        # TTD Gauge
        panels.append(self._create_gauge_panel(
            title="Terminal Temperature Difference",
            metric="condensync_ttd_celsius",
            x=6, y=y_pos, w=6, h=6,
            thresholds={"steps": [
                {"color": "green", "value": None},
                {"color": "yellow", "value": 3},
                {"color": "orange", "value": 5},
                {"color": "red", "value": 8},
            ]},
            max_value=15.0,
            unit="celsius",
        ))

        # Vacuum Pressure Gauge
        panels.append(self._create_gauge_panel(
            title="Vacuum Pressure",
            metric="condensync_vacuum_pressure_bar_abs",
            x=12, y=y_pos, w=6, h=6,
            thresholds={"steps": [
                {"color": "green", "value": None},
                {"color": "yellow", "value": 0.04},
                {"color": "orange", "value": 0.05},
                {"color": "red", "value": 0.06},
            ]},
            max_value=0.1,
            unit="pressurembar",
        ))

        # Heat Duty Gauge
        panels.append(self._create_gauge_panel(
            title="Heat Duty",
            metric="condensync_heat_duty_kw",
            x=18, y=y_pos, w=6, h=6,
            thresholds={"steps": [
                {"color": "blue", "value": None},
            ]},
            max_value=200000,
            unit="kwatt",
        ))

        y_pos += 6

        # Row: Trends
        panels.append(self._create_row_panel("Performance Trends", y_pos))
        y_pos += 1

        # CF Trend Graph
        panels.append(self._create_timeseries_panel(
            title="Cleanliness Factor Trend",
            metric="condensync_cleanliness_factor",
            x=0, y=y_pos, w=12, h=8,
            unit="percentunit",
        ))

        # TTD Trend Graph
        panels.append(self._create_timeseries_panel(
            title="TTD Trend",
            metric="condensync_ttd_celsius",
            x=12, y=y_pos, w=12, h=8,
            unit="celsius",
        ))

        y_pos += 8

        # Row: Calculation Metrics
        panels.append(self._create_row_panel("Calculation Performance", y_pos))
        y_pos += 1

        # Calculation Latency Histogram
        panels.append(self._create_heatmap_panel(
            title="Calculation Latency Distribution",
            metric="condensync_calculation_latency_seconds_bucket",
            x=0, y=y_pos, w=12, h=8,
        ))

        # Calculation Rate
        panels.append(self._create_timeseries_panel(
            title="Calculations per Second",
            metric="rate(condensync_calculation_latency_seconds_count[5m])",
            x=12, y=y_pos, w=12, h=8,
            unit="ops",
        ))

        y_pos += 8

        # Row: Recommendations
        panels.append(self._create_row_panel("Recommendations", y_pos))
        y_pos += 1

        # Recommendations Generated
        panels.append(self._create_stat_panel(
            title="Recommendations Generated (24h)",
            metric="increase(condensync_recommendations_generated_total[24h])",
            x=0, y=y_pos, w=6, h=4,
        ))

        # Active Recommendations
        panels.append(self._create_stat_panel(
            title="Active Recommendations",
            metric="sum(condensync_active_recommendations)",
            x=6, y=y_pos, w=6, h=4,
        ))

        # Estimated Savings
        panels.append(self._create_stat_panel(
            title="Estimated Savings (USD/hr)",
            metric="sum(condensync_estimated_savings_usd_hr)",
            x=12, y=y_pos, w=6, h=4,
            unit="currencyUSD",
        ))

        # Implementation Rate
        panels.append(self._create_stat_panel(
            title="Implementation Rate",
            metric="sum(increase(condensync_recommendations_implemented_total[24h])) / sum(increase(condensync_recommendations_generated_total[24h]))",
            x=18, y=y_pos, w=6, h=4,
            unit="percentunit",
        ))

        y_pos += 4

        # Row: Data Quality
        panels.append(self._create_row_panel("Data Quality", y_pos))
        y_pos += 1

        # Data Quality Score
        panels.append(self._create_gauge_panel(
            title="Data Quality Score",
            metric="avg(condensync_data_quality_score)",
            x=0, y=y_pos, w=6, h=6,
            thresholds={"steps": [
                {"color": "red", "value": None},
                {"color": "orange", "value": 0.7},
                {"color": "yellow", "value": 0.85},
                {"color": "green", "value": 0.95},
            ]},
            max_value=1.0,
            unit="percentunit",
        ))

        # Data Freshness
        panels.append(self._create_timeseries_panel(
            title="Data Freshness (seconds)",
            metric="condensync_data_freshness_seconds",
            x=6, y=y_pos, w=9, h=6,
            unit="s",
        ))

        # Validation Failures
        panels.append(self._create_timeseries_panel(
            title="Validation Failures",
            metric="rate(condensync_data_validation_failures_total[5m])",
            x=15, y=y_pos, w=9, h=6,
            unit="short",
        ))

        y_pos += 6

        # Row: Alerts
        panels.append(self._create_row_panel("Alerts", y_pos))
        y_pos += 1

        # Active Alerts by Severity
        panels.append(self._create_piechart_panel(
            title="Active Alerts by Severity",
            metric="condensync_active_alerts",
            x=0, y=y_pos, w=8, h=6,
        ))

        # Alert Rate
        panels.append(self._create_timeseries_panel(
            title="Alert Rate",
            metric="rate(condensync_alerts_raised_total[5m])",
            x=8, y=y_pos, w=8, h=6,
            unit="short",
        ))

        # Alert Response Time
        panels.append(self._create_timeseries_panel(
            title="Alert Response Time (p95)",
            metric="histogram_quantile(0.95, rate(condensync_alert_response_time_seconds_bucket[1h]))",
            x=16, y=y_pos, w=8, h=6,
            unit="s",
        ))

        return panels

    def _create_row_panel(
        self,
        title: str,
        y: int,
    ) -> Dict[str, Any]:
        """Create a row panel."""
        return {
            "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "id": hash(title) % 10000,
            "panels": [],
            "title": title,
            "type": "row",
        }

    def _create_gauge_panel(
        self,
        title: str,
        metric: str,
        x: int,
        y: int,
        w: int,
        h: int,
        thresholds: Dict[str, Any],
        max_value: float,
        unit: str = "short",
    ) -> Dict[str, Any]:
        """Create a gauge panel."""
        return {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "max": max_value,
                    "min": 0,
                    "thresholds": thresholds,
                    "unit": unit,
                },
                "overrides": [],
            },
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "id": hash(f"{title}-{x}-{y}") % 10000,
            "options": {
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                    "expr": f'{metric}{{condenser_id=~"$condenser_id", unit=~"$unit"}}',
                    "refId": "A",
                }
            ],
            "title": title,
            "type": "gauge",
        }

    def _create_timeseries_panel(
        self,
        title: str,
        metric: str,
        x: int,
        y: int,
        w: int,
        h: int,
        unit: str = "short",
    ) -> Dict[str, Any]:
        """Create a timeseries panel."""
        return {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisBorderShow": False,
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "never",
                        "spanNulls": False,
                        "stacking": {"group": "A", "mode": "none"},
                        "thresholdsStyle": {"mode": "off"},
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}],
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "id": hash(f"{title}-{x}-{y}") % 10000,
            "options": {
                "legend": {"calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": True},
                "tooltip": {"mode": "single", "sort": "none"},
            },
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                    "expr": f'{metric}{{condenser_id=~"$condenser_id", unit=~"$unit"}}' if "{{" not in metric else metric,
                    "legendFormat": "{{condenser_id}}",
                    "refId": "A",
                }
            ],
            "title": title,
            "type": "timeseries",
        }

    def _create_stat_panel(
        self,
        title: str,
        metric: str,
        x: int,
        y: int,
        w: int,
        h: int,
        unit: str = "short",
    ) -> Dict[str, Any]:
        """Create a stat panel."""
        return {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}],
                    },
                    "unit": unit,
                },
                "overrides": [],
            },
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "id": hash(f"{title}-{x}-{y}") % 10000,
            "options": {
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "textMode": "auto",
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                    "expr": metric,
                    "refId": "A",
                }
            ],
            "title": title,
            "type": "stat",
        }

    def _create_heatmap_panel(
        self,
        title: str,
        metric: str,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Create a heatmap panel."""
        return {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "id": hash(f"{title}-{x}-{y}") % 10000,
            "options": {
                "calculate": False,
                "cellGap": 1,
                "color": {
                    "mode": "scheme",
                    "scale": "linear",
                    "scheme": "Oranges",
                    "steps": 64,
                },
                "exemplars": {"color": "rgba(255,0,255,0.7)"},
                "filterValues": {"le": 1e-9},
                "legend": {"show": True},
                "rowsFrame": {"layout": "auto"},
                "tooltip": {
                    "mode": "single",
                    "showColorScale": True,
                    "yHistogram": False,
                },
                "yAxis": {"axisPlacement": "left", "reverse": False},
            },
            "pluginVersion": "10.0.0",
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                    "expr": f"sum(increase({metric}[5m])) by (le)",
                    "format": "heatmap",
                    "refId": "A",
                }
            ],
            "title": title,
            "type": "heatmap",
        }

    def _create_piechart_panel(
        self,
        title: str,
        metric: str,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """Create a pie chart panel."""
        return {
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {"hideFrom": {"legend": False, "tooltip": False, "viz": False}},
                    "mappings": [],
                },
                "overrides": [],
            },
            "gridPos": {"h": h, "w": w, "x": x, "y": y},
            "id": hash(f"{title}-{x}-{y}") % 10000,
            "options": {
                "legend": {"displayMode": "list", "placement": "right", "showLegend": True},
                "pieType": "pie",
                "reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": False,
                },
                "tooltip": {"mode": "single", "sort": "none"},
            },
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                    "expr": f"sum({metric}) by (severity)",
                    "legendFormat": "{{severity}}",
                    "refId": "A",
                }
            ],
            "title": title,
            "type": "piechart",
        }

    def export_dashboard_json(
        self,
        filepath: str,
        title: str = "GL-017 CONDENSYNC - Condenser Optimization",
    ) -> None:
        """
        Export Grafana dashboard to JSON file.

        Args:
            filepath: Output file path
            title: Dashboard title
        """
        dashboard = self.generate_grafana_dashboard(title=title)

        with open(filepath, "w") as f:
            json.dump(dashboard, f, indent=2)

        logger.info(f"Exported Grafana dashboard to {filepath}")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_exporter: Optional[MetricsExporter] = None


def get_exporter_instance() -> MetricsExporter:
    """
    Get or create global exporter instance.

    Returns:
        MetricsExporter instance
    """
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = MetricsExporter()
    return _global_exporter


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "MetricsExporter",
    "MetricsHTTPHandler",
    "get_exporter_instance",
]
