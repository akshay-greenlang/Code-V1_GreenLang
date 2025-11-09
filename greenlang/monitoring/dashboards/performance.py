"""
Performance Monitoring Dashboard
=================================

Real-time performance monitoring with automated alerting.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """
    Comprehensive performance monitoring with SLA tracking and alerting.
    Alerts on: P95 > 1s, Error rate > 1%
    """

    def __init__(self):
        self.dashboard_uid = "greenlang-performance"
        self.dashboard_title = "GreenLang Performance Monitoring"

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete performance monitoring dashboard"""
        return {
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "tags": ["greenlang", "performance", "sla"],
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    self._p95_latency_stat(),
                    self._error_rate_stat(),
                    self._throughput_gauge(),
                    self._agent_execution_histogram(),
                    self._cache_latency_breakdown(),
                    self._database_performance(),
                    self._llm_response_distribution(),
                    self._error_timeline(),
                    self._slow_queries_table(),
                    self._performance_heatmap()
                ],
                "templating": {
                    "list": [
                        {
                            "name": "service",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_request_duration_seconds, service)",
                            "multi": True,
                            "includeAll": True
                        },
                        {
                            "name": "agent",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_agent_execution_seconds, agent)",
                            "multi": True,
                            "includeAll": True
                        }
                    ]
                },
                "annotations": {
                    "list": [
                        {
                            "name": "SLA Violations",
                            "datasource": "Prometheus",
                            "enable": True,
                            "iconColor": "red",
                            "expr": "histogram_quantile(0.95, rate(greenlang_request_duration_seconds_bucket[5m])) > 1"
                        }
                    ]
                }
            },
            "overwrite": True
        }

    def _p95_latency_stat(self) -> Dict[str, Any]:
        """P95 latency stat with alert threshold"""
        return {
            "id": 1,
            "title": "P95 Latency",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(greenlang_request_duration_seconds_bucket{service=~\"$service\"}[5m]))",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s",
                    "decimals": 3,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 0.5, "color": "yellow"},
                            {"value": 1.0, "color": "red"}
                        ]
                    }
                }
            },
            "options": {
                "graphMode": "area",
                "colorMode": "background",
                "textMode": "value_and_name"
            },
            "datasource": "Prometheus"
        }

    def _error_rate_stat(self) -> Dict[str, Any]:
        """Error rate with 1% threshold"""
        return {
            "id": 2,
            "title": "Error Rate",
            "type": "stat",
            "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
            "targets": [
                {
                    "expr": "sum(rate(greenlang_request_errors_total{service=~\"$service\"}[5m])) / sum(rate(greenlang_request_total{service=~\"$service\"}[5m])) * 100",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "decimals": 2,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 0.5, "color": "yellow"},
                            {"value": 1.0, "color": "red"}
                        ]
                    }
                }
            },
            "options": {
                "graphMode": "area",
                "colorMode": "background",
                "textMode": "value_and_name"
            },
            "datasource": "Prometheus"
        }

    def _throughput_gauge(self) -> Dict[str, Any]:
        """Requests per second gauge"""
        return {
            "id": 3,
            "title": "Throughput (req/s)",
            "type": "gauge",
            "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
            "targets": [
                {
                    "expr": "sum(rate(greenlang_request_total{service=~\"$service\"}[1m]))",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "reqps",
                    "min": 0,
                    "max": 1000,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "blue"},
                            {"value": 100, "color": "green"},
                            {"value": 500, "color": "yellow"},
                            {"value": 800, "color": "red"}
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

    def _agent_execution_histogram(self) -> Dict[str, Any]:
        """Agent execution times - P50, P95, P99"""
        return {
            "id": 4,
            "title": "Agent Execution Times (Histogram)",
            "type": "timeseries",
            "gridPos": {"x": 18, "y": 0, "w": 6, "h": 8},
            "targets": [
                {
                    "expr": "histogram_quantile(0.50, rate(greenlang_agent_execution_seconds_bucket{agent=~\"$agent\"}[5m]))",
                    "refId": "A",
                    "legendFormat": "P50"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(greenlang_agent_execution_seconds_bucket{agent=~\"$agent\"}[5m]))",
                    "refId": "B",
                    "legendFormat": "P95"
                },
                {
                    "expr": "histogram_quantile(0.99, rate(greenlang_agent_execution_seconds_bucket{agent=~\"$agent\"}[5m]))",
                    "refId": "C",
                    "legendFormat": "P99"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 0,
                        "showPoints": "never"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "P99"},
                        "properties": [
                            {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "max", "last"]
                }
            },
            "datasource": "Prometheus"
        }

    def _cache_latency_breakdown(self) -> Dict[str, Any]:
        """Cache latency by layer - L1/L2/L3"""
        return {
            "id": 5,
            "title": "Cache Latency by Layer",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_cache_latency_seconds{layer=\"L1\"}) by (service)",
                    "refId": "A",
                    "legendFormat": "L1 (Memory) - {{service}}"
                },
                {
                    "expr": "avg(greenlang_cache_latency_seconds{layer=\"L2\"}) by (service)",
                    "refId": "B",
                    "legendFormat": "L2 (Redis) - {{service}}"
                },
                {
                    "expr": "avg(greenlang_cache_latency_seconds{layer=\"L3\"}) by (service)",
                    "refId": "C",
                    "legendFormat": "L3 (Database) - {{service}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "ms",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20,
                        "showPoints": "never"
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

    def _database_performance(self) -> Dict[str, Any]:
        """Database query performance"""
        return {
            "id": 6,
            "title": "Database Query Performance",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 4, "w": 6, "h": 8},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(greenlang_db_query_duration_seconds_bucket[5m]))",
                    "refId": "A",
                    "legendFormat": "P95 Query Time"
                },
                {
                    "expr": "rate(greenlang_db_slow_queries_total[5m])",
                    "refId": "B",
                    "legendFormat": "Slow Queries/s"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "axisPlacement": "left"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "P95 Query Time"},
                        "properties": [
                            {"id": "unit", "value": "s"}
                        ]
                    },
                    {
                        "matcher": {"id": "byName", "options": "Slow Queries/s"},
                        "properties": [
                            {"id": "unit", "value": "qps"},
                            {"id": "custom.axisPlacement", "value": "right"}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _llm_response_distribution(self) -> Dict[str, Any]:
        """LLM response time distribution"""
        return {
            "id": 7,
            "title": "LLM Response Time Distribution",
            "type": "histogram",
            "gridPos": {"x": 0, "y": 12, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(rate(greenlang_llm_request_duration_seconds_bucket{service=~\"$service\"}[5m])) by (le)",
                    "refId": "A",
                    "format": "heatmap"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "s"
                }
            },
            "options": {
                "calculate": True,
                "cellGap": 2,
                "color": {
                    "mode": "scheme",
                    "scheme": "Spectral",
                    "fill": "dark-red"
                },
                "yAxis": {
                    "axisLabel": "Response Time",
                    "decimals": 2
                }
            },
            "datasource": "Prometheus"
        }

    def _error_timeline(self) -> Dict[str, Any]:
        """Error timeline with breakdown by type"""
        return {
            "id": 8,
            "title": "Error Rate Timeline",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 12, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(rate(greenlang_request_errors_total{service=~\"$service\"}[5m])) by (error_type)",
                    "refId": "A",
                    "legendFormat": "{{error_type}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "reqps",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 50,
                        "showPoints": "never",
                        "stacking": {
                            "mode": "normal"
                        }
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["sum"]
                }
            },
            "datasource": "Prometheus"
        }

    def _slow_queries_table(self) -> Dict[str, Any]:
        """Slow query table with details"""
        return {
            "id": 9,
            "title": "Slow Queries (P95 > 100ms)",
            "type": "table",
            "gridPos": {"x": 0, "y": 20, "w": 16, "h": 8},
            "targets": [
                {
                    "expr": "topk(20, histogram_quantile(0.95, rate(greenlang_db_query_duration_seconds_bucket[5m])) > 0.1)",
                    "refId": "A",
                    "format": "table",
                    "instant": True
                }
            ],
            "transformations": [
                {
                    "id": "organize",
                    "options": {
                        "excludeByName": {
                            "Time": True,
                            "__name__": True
                        },
                        "renameByName": {
                            "query_name": "Query",
                            "table": "Table",
                            "operation": "Operation",
                            "Value": "P95 Latency (s)"
                        }
                    }
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "left"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "P95 Latency (s)"},
                        "properties": [
                            {"id": "unit", "value": "s"},
                            {"id": "custom.displayMode", "value": "gradient-gauge"},
                            {"id": "decimals", "value": 3}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _performance_heatmap(self) -> Dict[str, Any]:
        """Performance heatmap across services"""
        return {
            "id": 10,
            "title": "Service Performance Heatmap",
            "type": "heatmap",
            "gridPos": {"x": 16, "y": 20, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(greenlang_request_duration_seconds_bucket[5m])) by (service)",
                    "refId": "A"
                }
            ],
            "options": {
                "calculate": True,
                "cellGap": 2,
                "color": {
                    "mode": "scheme",
                    "scheme": "RdYlGn",
                    "reverse": True
                },
                "yAxis": {
                    "axisLabel": "Service"
                }
            },
            "datasource": "Prometheus"
        }

    def export_to_file(self, output_path: str) -> None:
        """Export dashboard to JSON file"""
        dashboard = self.generate_dashboard()
        try:
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Performance dashboard exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise

    def deploy_to_grafana(self, grafana_url: str, api_key: str) -> None:
        """Deploy dashboard to Grafana"""
        import requests

        dashboard = self.generate_dashboard()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = f"{grafana_url}/api/dashboards/db"

        try:
            response = requests.post(url, json=dashboard, headers=headers)
            response.raise_for_status()
            logger.info(f"Performance dashboard deployed: {response.json()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to deploy dashboard: {e}")
            raise


def main():
    """Main entry point"""
    dashboard = PerformanceDashboard()
    output_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\dashboards\\performance.json"
    dashboard.export_to_file(output_path)
    print(f"Performance Monitoring Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
