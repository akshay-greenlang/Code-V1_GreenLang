"""
Developer Productivity Dashboard
=================================

Track developer metrics, time savings, and gamification leaderboards.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ProductivityDashboard:
    """
    Developer productivity metrics with gamification elements.
    """

    def __init__(self):
        self.dashboard_uid = "greenlang-productivity"
        self.dashboard_title = "GreenLang Developer Productivity"

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate productivity dashboard with leaderboards"""
        return {
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "tags": ["greenlang", "productivity", "developers"],
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": "10m",
                "time": {
                    "from": "now-30d",
                    "to": "now"
                },
                "panels": [
                    self._time_to_first_pr(),
                    self._code_reuse_percentage(),
                    self._loc_comparison(),
                    self._infrastructure_leaderboard(),
                    self._time_savings_comparison(),
                    self._bug_fix_time(),
                    self._contributions_heatmap(),
                    self._achievements_table(),
                    self._team_velocity(),
                    self._onboarding_progress()
                ],
                "templating": {
                    "list": [
                        {
                            "name": "team",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_developer_metrics, team)",
                            "multi": True,
                            "includeAll": True
                        },
                        {
                            "name": "developer",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_developer_metrics, developer)",
                            "multi": True,
                            "includeAll": True
                        }
                    ]
                }
            },
            "overwrite": True
        }

    def _time_to_first_pr(self) -> Dict[str, Any]:
        """Average time to first PR for new developers"""
        return {
            "id": 1,
            "title": "Time to First PR (New Developers)",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_time_to_first_pr_hours{team=~\"$team\"}) by (team)",
                    "refId": "A",
                    "legendFormat": "{{team}}"
                },
                {
                    "expr": "avg(greenlang_time_to_first_pr_hours)",
                    "refId": "B",
                    "legendFormat": "Platform Average"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "h",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 20,
                        "showPoints": "always"
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Platform Average"},
                        "properties": [
                            {"id": "custom.lineStyle", "value": {"dash": [10, 10]}},
                            {"id": "color", "value": {"fixedColor": "blue", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "calcs": ["mean", "last"]
                }
            },
            "datasource": "Prometheus"
        }

    def _code_reuse_percentage(self) -> Dict[str, Any]:
        """Code reuse percentage gauge"""
        return {
            "id": 2,
            "title": "Code Reuse Percentage",
            "type": "gauge",
            "gridPos": {"x": 12, "y": 0, "w": 6, "h": 8},
            "targets": [
                {
                    "expr": "(sum(greenlang_infrastructure_loc{team=~\"$team\"}) / sum(greenlang_total_loc{team=~\"$team\"})) * 100",
                    "refId": "A"
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
                            {"value": 70, "color": "yellow"},
                            {"value": 90, "color": "green"}
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

    def _loc_comparison(self) -> Dict[str, Any]:
        """Lines of code: infrastructure vs custom"""
        return {
            "id": 3,
            "title": "LOC per Developer (Infrastructure vs Custom)",
            "type": "barchart",
            "gridPos": {"x": 18, "y": 0, "w": 6, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_infrastructure_loc{developer=~\"$developer\"}) by (developer)",
                    "refId": "A",
                    "legendFormat": "{{developer}} - Infrastructure"
                },
                {
                    "expr": "sum(greenlang_custom_loc{developer=~\"$developer\"}) by (developer)",
                    "refId": "B",
                    "legendFormat": "{{developer}} - Custom"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "lines"
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
                "showValue": "auto",
                "legend": {
                    "displayMode": "list",
                    "placement": "bottom"
                }
            },
            "datasource": "Prometheus"
        }

    def _infrastructure_leaderboard(self) -> Dict[str, Any]:
        """Infrastructure contributions leaderboard"""
        return {
            "id": 4,
            "title": "Infrastructure Contributors Leaderboard",
            "type": "table",
            "gridPos": {"x": 0, "y": 8, "w": 12, "h": 10},
            "targets": [
                {
                    "expr": "topk(20, sum(greenlang_infrastructure_contributions{team=~\"$team\"}) by (developer, team))",
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
                            "developer": "Developer",
                            "team": "Team",
                            "Value": "Contributions"
                        }
                    }
                },
                {
                    "id": "sortBy",
                    "options": {
                        "sort": [{"field": "Contributions", "desc": True}]
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
                        "matcher": {"id": "byName", "options": "Contributions"},
                        "properties": [
                            {"id": "custom.displayMode", "value": "gradient-gauge"},
                            {"id": "custom.align", "value": "right"},
                            {"id": "color", "value": {
                                "mode": "continuous-GrYlRd"
                            }}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _time_savings_comparison(self) -> Dict[str, Any]:
        """Time saved vs manual implementation"""
        return {
            "id": 5,
            "title": "Time Saved vs Manual Implementation",
            "type": "bargauge",
            "gridPos": {"x": 12, "y": 8, "w": 12, "h": 10},
            "targets": [
                {
                    "expr": "sum(greenlang_time_saved_hours{team=~\"$team\"}) by (feature)",
                    "refId": "A",
                    "legendFormat": "{{feature}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "h",
                    "min": 0,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "blue"},
                            {"value": 10, "color": "green"},
                            {"value": 50, "color": "dark-green"}
                        ]
                    },
                    "color": {
                        "mode": "thresholds"
                    }
                }
            },
            "options": {
                "orientation": "horizontal",
                "displayMode": "gradient",
                "showUnfilled": True
            },
            "datasource": "Prometheus"
        }

    def _bug_fix_time(self) -> Dict[str, Any]:
        """Bug fix time: infrastructure vs custom"""
        return {
            "id": 6,
            "title": "Bug Fix Time: Infrastructure vs Custom",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 18, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, rate(greenlang_bug_fix_duration_hours_bucket{code_type=\"infrastructure\", team=~\"$team\"}[1d]))",
                    "refId": "A",
                    "legendFormat": "Infrastructure (P95)"
                },
                {
                    "expr": "histogram_quantile(0.95, rate(greenlang_bug_fix_duration_hours_bucket{code_type=\"custom\", team=~\"$team\"}[1d]))",
                    "refId": "B",
                    "legendFormat": "Custom (P95)"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "h",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 30
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Infrastructure (P95)"},
                        "properties": [
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byName", "options": "Custom (P95)"},
                        "properties": [
                            {"id": "color", "value": {"fixedColor": "orange", "mode": "fixed"}}
                        ]
                    }
                ]
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

    def _contributions_heatmap(self) -> Dict[str, Any]:
        """Contributions heatmap"""
        return {
            "id": 7,
            "title": "Weekly Contribution Activity",
            "type": "heatmap",
            "gridPos": {"x": 12, "y": 18, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(increase(greenlang_commits_total{developer=~\"$developer\"}[1d])) by (developer)",
                    "refId": "A"
                }
            ],
            "options": {
                "calculate": True,
                "cellGap": 2,
                "color": {
                    "mode": "scheme",
                    "scheme": "Greens",
                    "fill": "dark-green"
                },
                "yAxis": {
                    "axisLabel": "Developer"
                }
            },
            "datasource": "Prometheus"
        }

    def _achievements_table(self) -> Dict[str, Any]:
        """Achievements and badges"""
        return {
            "id": 8,
            "title": "Developer Achievements",
            "type": "table",
            "gridPos": {"x": 0, "y": 26, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "greenlang_developer_achievements{developer=~\"$developer\"}",
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
                            "developer": "Developer",
                            "achievement": "Achievement",
                            "level": "Level",
                            "earned_date": "Earned"
                        }
                    }
                }
            ],
            "fieldConfig": {
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Level"},
                        "properties": [
                            {"id": "custom.displayMode", "value": "color-background"},
                            {"id": "mappings", "value": [
                                {"type": "value", "options": {"bronze": {"color": "#CD7F32", "index": 0}}},
                                {"type": "value", "options": {"silver": {"color": "#C0C0C0", "index": 1}}},
                                {"type": "value", "options": {"gold": {"color": "#FFD700", "index": 2}}},
                                {"type": "value", "options": {"platinum": {"color": "#E5E4E2", "index": 3}}}
                            ]}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _team_velocity(self) -> Dict[str, Any]:
        """Team velocity metrics"""
        return {
            "id": 9,
            "title": "Team Velocity (Story Points per Sprint)",
            "type": "timeseries",
            "gridPos": {"x": 12, "y": 26, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(greenlang_story_points_completed{team=~\"$team\"}) by (team)",
                    "refId": "A",
                    "legendFormat": "{{team}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "custom": {
                        "lineWidth": 2,
                        "fillOpacity": 30,
                        "showPoints": "always",
                        "pointSize": 8
                    }
                }
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

    def _onboarding_progress(self) -> Dict[str, Any]:
        """New developer onboarding progress"""
        return {
            "id": 10,
            "title": "New Developer Onboarding Progress",
            "type": "stat",
            "gridPos": {"x": 0, "y": 34, "w": 24, "h": 4},
            "targets": [
                {
                    "expr": "count(greenlang_developer_onboarding_status{status=\"in_progress\"})",
                    "refId": "A"
                },
                {
                    "expr": "count(greenlang_developer_onboarding_status{status=\"completed\", completion_date > (time() - 2592000)})",
                    "refId": "B"
                },
                {
                    "expr": "avg(greenlang_onboarding_completion_percent{status=\"in_progress\"})",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "overrides": [
                    {
                        "matcher": {"id": "byFrameRefID", "options": "A"},
                        "properties": [
                            {"id": "displayName", "value": "In Progress"},
                            {"id": "color", "value": {"fixedColor": "yellow", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "B"},
                        "properties": [
                            {"id": "displayName", "value": "Completed (30d)"},
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "C"},
                        "properties": [
                            {"id": "displayName", "value": "Avg Completion %"},
                            {"id": "unit", "value": "percent"},
                            {"id": "color", "value": {"fixedColor": "blue", "mode": "fixed"}}
                        ]
                    }
                ]
            },
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "textMode": "value_and_name",
                "orientation": "horizontal"
            },
            "datasource": "Prometheus"
        }

    def export_to_file(self, output_path: str) -> None:
        """Export dashboard to JSON"""
        dashboard = self.generate_dashboard()
        try:
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Productivity dashboard exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise


def main():
    """Main entry point"""
    dashboard = ProductivityDashboard()
    output_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\dashboards\\productivity.json"
    dashboard.export_to_file(output_path)
    print(f"Developer Productivity Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
