"""
Infrastructure Usage Metrics (IUM) Dashboard
==============================================

Production-grade Grafana dashboard for tracking infrastructure usage across GreenLang applications.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class InfrastructureUsageDashboard:
    """
    Comprehensive IUM dashboard with real-time metrics visualization.
    """

    def __init__(self):
        self.dashboard_uid = "greenlang-ium"
        self.dashboard_title = "GreenLang Infrastructure Usage Metrics"

    def generate_dashboard(self) -> Dict[str, Any]:
        """
        Generate complete Grafana dashboard JSON definition.

        Returns:
            Complete dashboard configuration for Grafana import
        """
        dashboard = {
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "tags": ["greenlang", "infrastructure", "ium"],
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": "5m",
                "time": {
                    "from": "now-7d",
                    "to": "now"
                },
                "panels": [
                    self._overall_ium_gauge(),
                    self._ium_trend_chart(),
                    self._ium_by_application(),
                    self._ium_by_team(),
                    self._top_files_table(),
                    self._bottom_files_table(),
                    self._component_adoption_heatmap(),
                    self._custom_code_hotspots()
                ],
                "templating": {
                    "list": [
                        {
                            "name": "application",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_ium_score, application)",
                            "multi": True,
                            "includeAll": True
                        },
                        {
                            "name": "team",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_ium_score, team)",
                            "multi": True,
                            "includeAll": True
                        }
                    ]
                },
                "annotations": {
                    "list": [
                        {
                            "name": "Deployments",
                            "datasource": "Prometheus",
                            "enable": True,
                            "iconColor": "blue",
                            "expr": "changes(greenlang_deployment_timestamp[5m]) > 0"
                        },
                        {
                            "name": "IUM Violations",
                            "datasource": "Prometheus",
                            "enable": True,
                            "iconColor": "red",
                            "expr": "greenlang_ium_score < 90"
                        }
                    ]
                }
            },
            "overwrite": True
        }

        return dashboard

    def _overall_ium_gauge(self) -> Dict[str, Any]:
        """Overall IUM gauge panel (0-100%)"""
        return {
            "id": 1,
            "title": "Overall Infrastructure Usage Metric",
            "type": "gauge",
            "gridPos": {"x": 0, "y": 0, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_ium_score{application=~\"$application\", team=~\"$team\"})",
                    "refId": "A",
                    "legendFormat": "Overall IUM"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "red"},
                            {"value": 80, "color": "yellow"},
                            {"value": 95, "color": "green"}
                        ]
                    },
                    "unit": "percent"
                }
            },
            "options": {
                "showThresholdLabels": True,
                "showThresholdMarkers": True
            },
            "datasource": "Prometheus"
        }

    def _ium_trend_chart(self) -> Dict[str, Any]:
        """IUM trend over time - line chart"""
        return {
            "id": 2,
            "title": "IUM Trend (7 Days)",
            "type": "timeseries",
            "gridPos": {"x": 8, "y": 0, "w": 16, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_ium_score{application=~\"$application\", team=~\"$team\"}) by (application)",
                    "refId": "A",
                    "legendFormat": "{{application}}"
                },
                {
                    "expr": "avg(greenlang_ium_score)",
                    "refId": "B",
                    "legendFormat": "Platform Average"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 100,
                    "unit": "percent",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 10,
                        "showPoints": "never",
                        "lineInterpolation": "smooth"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Platform Average"},
                        "properties": [
                            {"id": "custom.lineStyle", "value": {"dash": [10, 10]}},
                            {"id": "color", "value": {"mode": "fixed", "fixedColor": "blue"}}
                        ]
                    }
                ]
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "calcs": ["mean", "last", "max"]
                },
                "tooltip": {
                    "mode": "multi"
                }
            },
            "datasource": "Prometheus"
        }

    def _ium_by_application(self) -> Dict[str, Any]:
        """IUM by application - bar chart"""
        return {
            "id": 3,
            "title": "IUM by Application",
            "type": "barchart",
            "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_ium_score{team=~\"$team\"}) by (application)",
                    "refId": "A",
                    "legendFormat": "{{application}}"
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
                            {"value": 0, "color": "red"},
                            {"value": 80, "color": "yellow"},
                            {"value": 95, "color": "green"}
                        ]
                    },
                    "color": {
                        "mode": "thresholds"
                    }
                }
            },
            "options": {
                "orientation": "horizontal",
                "xTickLabelRotation": 0,
                "showValue": "always",
                "legend": {
                    "displayMode": "hidden"
                }
            },
            "datasource": "Prometheus"
        }

    def _ium_by_team(self) -> Dict[str, Any]:
        """IUM by team - bar chart"""
        return {
            "id": 4,
            "title": "IUM by Team",
            "type": "barchart",
            "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_ium_score{application=~\"$application\"}) by (team)",
                    "refId": "A",
                    "legendFormat": "{{team}}"
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
                            {"value": 0, "color": "red"},
                            {"value": 80, "color": "yellow"},
                            {"value": 95, "color": "green"}
                        ]
                    },
                    "color": {
                        "mode": "thresholds"
                    }
                }
            },
            "options": {
                "orientation": "horizontal",
                "xTickLabelRotation": 0,
                "showValue": "always",
                "legend": {
                    "displayMode": "hidden"
                }
            },
            "datasource": "Prometheus"
        }

    def _top_files_table(self) -> Dict[str, Any]:
        """Top 10 files by IUM - table"""
        return {
            "id": 5,
            "title": "Top 10 Files by IUM (Best Practices)",
            "type": "table",
            "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "topk(10, greenlang_file_ium_score{application=~\"$application\", team=~\"$team\"})",
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
                            "file_path": "File Path",
                            "Value": "IUM Score",
                            "application": "Application",
                            "team": "Team",
                            "infrastructure_lines": "Infrastructure LOC",
                            "custom_lines": "Custom LOC"
                        }
                    }
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "left",
                        "filterable": True
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "IUM Score"},
                        "properties": [
                            {"id": "unit", "value": "percent"},
                            {"id": "custom.displayMode", "value": "gradient-gauge"},
                            {"id": "thresholds", "value": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 80, "color": "yellow"},
                                    {"value": 95, "color": "green"}
                                ]
                            }}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _bottom_files_table(self) -> Dict[str, Any]:
        """Bottom 10 files by IUM - need attention"""
        return {
            "id": 6,
            "title": "Bottom 10 Files by IUM (Need Attention)",
            "type": "table",
            "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "bottomk(10, greenlang_file_ium_score{application=~\"$application\", team=~\"$team\"})",
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
                            "file_path": "File Path",
                            "Value": "IUM Score",
                            "application": "Application",
                            "team": "Team",
                            "infrastructure_lines": "Infrastructure LOC",
                            "custom_lines": "Custom LOC",
                            "custom_percentage": "Custom %"
                        }
                    }
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "left",
                        "filterable": True
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "IUM Score"},
                        "properties": [
                            {"id": "unit", "value": "percent"},
                            {"id": "custom.displayMode", "value": "gradient-gauge"},
                            {"id": "thresholds", "value": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 80, "color": "yellow"},
                                    {"value": 95, "color": "green"}
                                ]
                            }}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _component_adoption_heatmap(self) -> Dict[str, Any]:
        """Infrastructure component adoption - heatmap"""
        return {
            "id": 7,
            "title": "Infrastructure Component Adoption Heatmap",
            "type": "heatmap",
            "gridPos": {"x": 0, "y": 24, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_component_usage{application=~\"$application\", team=~\"$team\"}) by (component, application)",
                    "refId": "A",
                    "format": "time_series"
                }
            ],
            "options": {
                "calculate": True,
                "cellGap": 2,
                "cellRadius": 0,
                "color": {
                    "mode": "scheme",
                    "scheme": "Spectral",
                    "fill": "dark-green",
                    "reverse": False
                },
                "yAxis": {
                    "axisLabel": "Component",
                    "reverse": False
                },
                "legend": {
                    "show": True
                },
                "tooltip": {
                    "show": True,
                    "yHistogram": False
                }
            },
            "datasource": "Prometheus"
        }

    def _custom_code_hotspots(self) -> Dict[str, Any]:
        """Custom code hotspots - treemap"""
        return {
            "id": 8,
            "title": "Custom Code Hotspots (Areas for Optimization)",
            "type": "piechart",
            "gridPos": {"x": 12, "y": 24, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_custom_code_lines{application=~\"$application\", team=~\"$team\"}) by (module)",
                    "refId": "A",
                    "legendFormat": "{{module}}"
                }
            ],
            "options": {
                "pieType": "pie",
                "displayLabels": ["name", "percent"],
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "values": ["value", "percent"],
                    "calcs": ["sum"]
                },
                "tooltip": {
                    "mode": "multi"
                }
            },
            "fieldConfig": {
                "defaults": {
                    "unit": "lines",
                    "custom": {
                        "hideFrom": {
                            "tooltip": False,
                            "viz": False,
                            "legend": False
                        }
                    }
                }
            },
            "datasource": "Prometheus"
        }

    def export_to_file(self, output_path: str) -> None:
        """
        Export dashboard to JSON file for Grafana import.

        Args:
            output_path: Path to save dashboard JSON
        """
        dashboard = self.generate_dashboard()

        try:
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Dashboard exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise

    def deploy_to_grafana(self, grafana_url: str, api_key: str) -> None:
        """
        Deploy dashboard directly to Grafana instance.

        Args:
            grafana_url: Grafana instance URL
            api_key: Grafana API key with dashboard write permissions
        """
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
            logger.info(f"Dashboard deployed successfully: {response.json()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to deploy dashboard: {e}")
            raise


def main():
    """Main entry point for dashboard generation"""
    dashboard = InfrastructureUsageDashboard()

    # Export to JSON file
    output_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\dashboards\\infrastructure_usage.json"
    dashboard.export_to_file(output_path)

    print(f"Infrastructure Usage Dashboard generated: {output_path}")
    print(f"Import this dashboard into Grafana to start monitoring IUM metrics.")

    # Optional: Deploy directly to Grafana
    # dashboard.deploy_to_grafana("http://localhost:3000", "your-api-key")


if __name__ == "__main__":
    main()
