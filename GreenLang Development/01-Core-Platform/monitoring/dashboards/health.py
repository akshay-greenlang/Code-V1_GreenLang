# -*- coding: utf-8 -*-
"""
Infrastructure Health Dashboard
================================

Monitor service health, resource utilization, and system availability.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class HealthDashboard:
    """
    Infrastructure health and availability monitoring.
    """

    def __init__(self):
        self.dashboard_uid = "greenlang-health"
        self.dashboard_title = "GreenLang Infrastructure Health"

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate infrastructure health dashboard"""
        return {
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "tags": ["greenlang", "health", "infrastructure"],
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    self._service_health_grid(),
                    self._uptime_percentage(),
                    self._cache_service_health(),
                    self._database_pool_status(),
                    self._llm_provider_availability(),
                    self._api_rate_limits(),
                    self._resource_usage_cpu(),
                    self._resource_usage_memory(),
                    self._resource_usage_disk(),
                    self._network_throughput()
                ]
            },
            "overwrite": True
        }

    def _service_health_grid(self) -> Dict[str, Any]:
        """Service health status grid"""
        return {
            "id": 1,
            "title": "Service Health Status",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 24, "h": 6},
            "targets": [
                {
                    "expr": "up{job=~\"greenlang-factor-broker|greenlang-entity-mdm|greenlang-form-builder|greenlang-api-gateway\"}",
                    "refId": "A",
                    "legendFormat": "{{job}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "0": {
                                    "text": "DOWN",
                                    "color": "red"
                                },
                                "1": {
                                    "text": "UP",
                                    "color": "green"
                                }
                            }
                        }
                    ],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 1, "color": "green"}
                        ]
                    }
                }
            },
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "textMode": "value_and_name",
                "orientation": "horizontal"
            },
            "datasource": "Prometheus"
        }

    def _uptime_percentage(self) -> Dict[str, Any]:
        """Service uptime percentage"""
        return {
            "id": 2,
            "title": "Service Uptime (24h)",
            "type": "gauge",
            "gridPos": {"x": 0, "y": 6, "w": 8, "h": 6},
            "targets": [
                {
                    "expr": "avg_over_time(up{job=~\"greenlang-.*\"}[24h]) * 100",
                    "refId": "A",
                    "legendFormat": "{{job}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "decimals": 2,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 99, "color": "yellow"},
                            {"value": 99.9, "color": "green"}
                        ]
                    }
                }
            },
            "options": {
                "showThresholdLabels": True,
                "showThresholdMarkers": True
            },
            "datasource": "Prometheus"
        }

    def _cache_service_health(self) -> Dict[str, Any]:
        """Cache service health (Redis)"""
        return {
            "id": 3,
            "title": "Cache Service Health",
            "type": "timeseries",
            "gridPos": {"x": 8, "y": 6, "w": 8, "h": 6},
            "targets": [
                {
                    "expr": "greenlang_cache_service_available{service=\"redis\"}",
                    "refId": "A",
                    "legendFormat": "Redis Availability"
                },
                {
                    "expr": "greenlang_cache_connections_active{service=\"redis\"}",
                    "refId": "B",
                    "legendFormat": "Active Connections"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Redis Availability"},
                        "properties": [
                            {"id": "custom.fillOpacity", "value": 50},
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byName", "options": "Active Connections"},
                        "properties": [
                            {"id": "custom.axisPlacement", "value": "right"}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _database_pool_status(self) -> Dict[str, Any]:
        """Database connection pool status"""
        return {
            "id": 4,
            "title": "Database Connection Pool",
            "type": "timeseries",
            "gridPos": {"x": 16, "y": 6, "w": 8, "h": 6},
            "targets": [
                {
                    "expr": "greenlang_db_pool_active_connections",
                    "refId": "A",
                    "legendFormat": "Active Connections"
                },
                {
                    "expr": "greenlang_db_pool_idle_connections",
                    "refId": "B",
                    "legendFormat": "Idle Connections"
                },
                {
                    "expr": "greenlang_db_pool_max_connections",
                    "refId": "C",
                    "legendFormat": "Max Connections"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 10
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Max Connections"},
                        "properties": [
                            {"id": "custom.lineStyle", "value": {"dash": [10, 10]}},
                            {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "last", "max"]
                }
            },
            "datasource": "Prometheus"
        }

    def _llm_provider_availability(self) -> Dict[str, Any]:
        """LLM provider availability multi-stat"""
        return {
            "id": 5,
            "title": "LLM Provider Availability",
            "type": "stat",
            "gridPos": {"x": 0, "y": 12, "w": 12, "h": 6},
            "targets": [
                {
                    "expr": "greenlang_llm_provider_available{provider=\"openai\"}",
                    "refId": "A"
                },
                {
                    "expr": "greenlang_llm_provider_available{provider=\"anthropic\"}",
                    "refId": "B"
                },
                {
                    "expr": "greenlang_llm_provider_available{provider=\"google\"}",
                    "refId": "C"
                },
                {
                    "expr": "greenlang_llm_provider_available{provider=\"azure\"}",
                    "refId": "D"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "0": {"text": "Unavailable", "color": "red"},
                                "1": {"text": "Available", "color": "green"}
                            }
                        }
                    ]
                },
                "overrides": [
                    {
                        "matcher": {"id": "byFrameRefID", "options": "A"},
                        "properties": [{"id": "displayName", "value": "OpenAI"}]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "B"},
                        "properties": [{"id": "displayName", "value": "Anthropic"}]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "C"},
                        "properties": [{"id": "displayName", "value": "Google"}]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "D"},
                        "properties": [{"id": "displayName", "value": "Azure"}]
                    }
                ]
            },
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "textMode": "value",
                "orientation": "horizontal"
            },
            "datasource": "Prometheus"
        }

    def _api_rate_limits(self) -> Dict[str, Any]:
        """API rate limit utilization"""
        return {
            "id": 6,
            "title": "API Rate Limit Utilization",
            "type": "gauge",
            "gridPos": {"x": 12, "y": 12, "w": 12, "h": 6},
            "targets": [
                {
                    "expr": "(greenlang_api_requests_current / greenlang_api_rate_limit_max) * 100",
                    "refId": "A",
                    "legendFormat": "{{provider}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 60, "color": "yellow"},
                            {"value": 80, "color": "red"}
                        ]
                    }
                }
            },
            "options": {
                "showThresholdLabels": False,
                "showThresholdMarkers": True
            },
            "datasource": "Prometheus"
        }

    def _resource_usage_cpu(self) -> Dict[str, Any]:
        """CPU usage across instances"""
        return {
            "id": 7,
            "title": "CPU Usage",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 18, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                    "refId": "A",
                    "legendFormat": "{{instance}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "max"]
                }
            },
            "datasource": "Prometheus"
        }

    def _resource_usage_memory(self) -> Dict[str, Any]:
        """Memory usage across instances"""
        return {
            "id": 8,
            "title": "Memory Usage",
            "type": "timeseries",
            "gridPos": {"x": 8, "y": 18, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "((node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes) * 100",
                    "refId": "A",
                    "legendFormat": "{{instance}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "max"]
                }
            },
            "datasource": "Prometheus"
        }

    def _resource_usage_disk(self) -> Dict[str, Any]:
        """Disk usage across instances"""
        return {
            "id": 9,
            "title": "Disk Usage",
            "type": "timeseries",
            "gridPos": {"x": 16, "y": 18, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "((node_filesystem_size_bytes{mountpoint=\"/\"} - node_filesystem_avail_bytes{mountpoint=\"/\"}) / node_filesystem_size_bytes{mountpoint=\"/\"}) * 100",
                    "refId": "A",
                    "legendFormat": "{{instance}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "last"]
                }
            },
            "datasource": "Prometheus"
        }

    def _network_throughput(self) -> Dict[str, Any]:
        """Network throughput"""
        return {
            "id": 10,
            "title": "Network Throughput",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 26, "w": 24, "h": 8},
            "targets": [
                {
                    "expr": "rate(node_network_receive_bytes_total{device!=\"lo\"}[5m]) * 8",
                    "refId": "A",
                    "legendFormat": "{{instance}} - Receive"
                },
                {
                    "expr": "rate(node_network_transmit_bytes_total{device!=\"lo\"}[5m]) * 8",
                    "refId": "B",
                    "legendFormat": "{{instance}} - Transmit"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "bps",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 10
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "max"]
                }
            },
            "datasource": "Prometheus"
        }

    def export_to_file(self, output_path: str) -> None:
        """Export dashboard to JSON"""
        dashboard = self.generate_dashboard()
        try:
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Health dashboard exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise


def main():
    """Main entry point"""
    dashboard = HealthDashboard()
    output_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\dashboards\\health.json"
    dashboard.export_to_file(output_path)
    print(f"Infrastructure Health Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
