# -*- coding: utf-8 -*-
"""
Cost Savings & ROI Dashboard
=============================

Track LLM cost savings, cache efficiency, and infrastructure ROI.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CostSavingsDashboard:
    """
    Real-time cost savings and ROI tracking dashboard.
    Updates every 5 minutes.
    """

    def __init__(self):
        self.dashboard_uid = "greenlang-cost-savings"
        self.dashboard_title = "GreenLang Cost Savings & ROI"

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate complete Grafana dashboard for cost tracking"""
        return {
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "tags": ["greenlang", "cost", "roi", "savings"],
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": "5m",
                "time": {
                    "from": "now-30d",
                    "to": "now"
                },
                "panels": [
                    self._total_savings_stat(),
                    self._roi_gauge(),
                    self._llm_cost_savings_trend(),
                    self._cache_hit_rates(),
                    self._developer_time_savings(),
                    self._infrastructure_vs_custom_time(),
                    self._savings_by_optimization(),
                    self._monthly_comparison(),
                    self._cost_breakdown_table()
                ],
                "templating": {
                    "list": [
                        {
                            "name": "application",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_cost_savings, application)",
                            "multi": True,
                            "includeAll": True
                        },
                        {
                            "name": "period",
                            "type": "custom",
                            "options": [
                                {"text": "Last 7 Days", "value": "7d"},
                                {"text": "Last 30 Days", "value": "30d"},
                                {"text": "Last 90 Days", "value": "90d"},
                                {"text": "Year to Date", "value": "ytd"}
                            ],
                            "current": {
                                "text": "Last 30 Days",
                                "value": "30d"
                            }
                        }
                    ]
                }
            },
            "overwrite": True
        }

    def _total_savings_stat(self) -> Dict[str, Any]:
        """Total cost savings - big number"""
        return {
            "id": 1,
            "title": "Total Cost Savings (30 Days)",
            "type": "stat",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 6},
            "targets": [
                {
                    "expr": "sum(increase(greenlang_cost_savings_usd{application=~\"$application\"}[30d]))",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "decimals": 2,
                    "color": {
                        "mode": "thresholds"
                    },
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "blue"},
                            {"value": 10000, "color": "green"},
                            {"value": 50000, "color": "dark-green"}
                        ]
                    }
                }
            },
            "options": {
                "graphMode": "area",
                "colorMode": "background",
                "justifyMode": "center",
                "textMode": "value_and_name",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            "datasource": "Prometheus"
        }

    def _roi_gauge(self) -> Dict[str, Any]:
        """ROI percentage gauge"""
        return {
            "id": 2,
            "title": "Return on Investment (ROI)",
            "type": "gauge",
            "gridPos": {"x": 6, "y": 0, "w": 6, "h": 6},
            "targets": [
                {
                    "expr": "(sum(greenlang_cost_savings_usd) / sum(greenlang_infrastructure_cost_usd)) * 100",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1000,
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 100, "color": "yellow"},
                            {"value": 300, "color": "green"},
                            {"value": 500, "color": "dark-green"}
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

    def _llm_cost_savings_trend(self) -> Dict[str, Any]:
        """LLM cost savings trend with semantic caching"""
        return {
            "id": 3,
            "title": "LLM Cost Savings (Semantic Caching)",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 0, "w": 12, "h": 6},
            "targets": [
                {
                    "expr": "sum(rate(greenlang_llm_cost_savings_usd{application=~\"$application\"}[5m])) * 86400",
                    "refId": "A",
                    "legendFormat": "Daily Savings"
                },
                {
                    "expr": "sum(rate(greenlang_llm_cost_without_cache_usd{application=~\"$application\"}[5m])) * 86400",
                    "refId": "B",
                    "legendFormat": "Cost Without Cache"
                },
                {
                    "expr": "sum(rate(greenlang_llm_actual_cost_usd{application=~\"$application\"}[5m])) * 86400",
                    "refId": "C",
                    "legendFormat": "Actual Cost"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20,
                        "showPoints": "never"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Daily Savings"},
                        "properties": [
                            {"id": "custom.fillOpacity", "value": 50},
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "last", "sum"]
                }
            },
            "datasource": "Prometheus"
        }

    def _cache_hit_rates(self) -> Dict[str, Any]:
        """Cache hit rates by service - pie chart"""
        return {
            "id": 4,
            "title": "Cache Hit Rates by Service",
            "type": "piechart",
            "gridPos": {"x": 0, "y": 6, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_cache_hit_rate{application=~\"$application\"}) by (service)",
                    "refId": "A",
                    "legendFormat": "{{service}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percentunit",
                    "custom": {
                        "hideFrom": {
                            "tooltip": False,
                            "viz": False,
                            "legend": False
                        }
                    }
                }
            },
            "options": {
                "pieType": "donut",
                "displayLabels": ["name", "percent"],
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "values": ["value", "percent"]
                }
            },
            "datasource": "Prometheus"
        }

    def _developer_time_savings(self) -> Dict[str, Any]:
        """Developer time savings - cumulative"""
        return {
            "id": 5,
            "title": "Developer Time Savings (Cumulative)",
            "type": "timeseries",
            "gridPos": {"x": 8, "y": 6, "w": 16, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_developer_hours_saved{application=~\"$application\"})",
                    "refId": "A",
                    "legendFormat": "Total Hours Saved"
                },
                {
                    "expr": "sum(greenlang_developer_hours_saved{application=~\"$application\"}) * 100",
                    "refId": "B",
                    "legendFormat": "Value (USD @ $100/hr)"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 30,
                        "showPoints": "never",
                        "axisPlacement": "left"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Total Hours Saved"},
                        "properties": [
                            {"id": "unit", "value": "h"},
                            {"id": "color", "value": {"fixedColor": "blue", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byName", "options": "Value (USD @ $100/hr)"},
                        "properties": [
                            {"id": "unit", "value": "currencyUSD"},
                            {"id": "custom.axisPlacement", "value": "right"},
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["last"]
                }
            },
            "datasource": "Prometheus"
        }

    def _infrastructure_vs_custom_time(self) -> Dict[str, Any]:
        """Infrastructure vs custom code development time"""
        return {
            "id": 6,
            "title": "Development Time: Infrastructure vs Custom",
            "type": "barchart",
            "gridPos": {"x": 0, "y": 14, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_dev_time_infrastructure_hours{application=~\"$application\"}) by (application)",
                    "refId": "A",
                    "legendFormat": "{{application}} - Infrastructure"
                },
                {
                    "expr": "sum(greenlang_dev_time_custom_hours{application=~\"$application\"}) by (application)",
                    "refId": "B",
                    "legendFormat": "{{application}} - Custom"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "h"
                },
                "overrides": [
                    {
                        "matcher": {"id": "byFrameRefID", "options": "A"},
                        "properties": [
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "B"},
                        "properties": [
                            {"id": "color", "value": {"fixedColor": "orange", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "orientation": "horizontal",
                "stacking": "normal",
                "showValue": "always",
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom"
                }
            },
            "datasource": "Prometheus"
        }

    def _savings_by_optimization(self) -> Dict[str, Any]:
        """Cost savings by optimization type - stacked bar"""
        return {
            "id": 7,
            "title": "Cost Savings by Optimization Type",
            "type": "barchart",
            "gridPos": {"x": 12, "y": 14, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_cost_savings_usd{application=~\"$application\"}) by (optimization_type)",
                    "refId": "A",
                    "legendFormat": "{{optimization_type}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD"
                }
            },
            "options": {
                "orientation": "vertical",
                "stacking": "normal",
                "showValue": "auto",
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "values": ["value", "percent"],
                    "calcs": ["sum"]
                }
            },
            "datasource": "Prometheus"
        }

    def _monthly_comparison(self) -> Dict[str, Any]:
        """Month-over-month comparison"""
        return {
            "id": 8,
            "title": "Monthly Cost Savings Trend",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 22, "w": 16, "h": 8},
            "targets": [
                {
                    "expr": "sum(increase(greenlang_cost_savings_usd{application=~\"$application\"}[30d]))",
                    "refId": "A",
                    "legendFormat": "Monthly Savings"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "currencyUSD",
                    "custom": {
                        "lineWidth": 3,
                        "fillOpacity": 40,
                        "showPoints": "always",
                        "pointSize": 8
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "max", "sum"]
                }
            },
            "datasource": "Prometheus"
        }

    def _cost_breakdown_table(self) -> Dict[str, Any]:
        """Detailed cost breakdown table"""
        return {
            "id": 9,
            "title": "Cost Savings Breakdown",
            "type": "table",
            "gridPos": {"x": 16, "y": 22, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_cost_savings_usd{application=~\"$application\"}) by (application, optimization_type, service)",
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
                            "Time": True
                        },
                        "renameByName": {
                            "application": "Application",
                            "optimization_type": "Optimization",
                            "service": "Service",
                            "Value": "Savings (USD)"
                        }
                    }
                },
                {
                    "id": "sortBy",
                    "options": {
                        "fields": {},
                        "sort": [{"field": "Savings (USD)", "desc": True}]
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
                        "matcher": {"id": "byName", "options": "Savings (USD)"},
                        "properties": [
                            {"id": "unit", "value": "currencyUSD"},
                            {"id": "custom.displayMode", "value": "gradient-gauge"},
                            {"id": "custom.align", "value": "right"}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def export_to_file(self, output_path: str) -> None:
        """Export dashboard to JSON file"""
        dashboard = self.generate_dashboard()
        try:
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Cost Savings dashboard exported to {output_path}")
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
            logger.info(f"Cost Savings dashboard deployed: {response.json()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to deploy dashboard: {e}")
            raise


def main():
    """Main entry point"""
    dashboard = CostSavingsDashboard()
    output_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\dashboards\\cost_savings.json"
    dashboard.export_to_file(output_path)
    print(f"Cost Savings Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
