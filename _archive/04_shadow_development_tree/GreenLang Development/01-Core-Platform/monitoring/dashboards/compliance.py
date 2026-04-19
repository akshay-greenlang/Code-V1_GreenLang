# -*- coding: utf-8 -*-
"""
Compliance & Quality Dashboard
===============================

Track enforcement violations, ADR coverage, code review metrics.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ComplianceDashboard:
    """
    Compliance and quality metrics tracking.
    Alerts on: IUM < 95% on PRs, missing ADRs
    """

    def __init__(self):
        self.dashboard_uid = "greenlang-compliance"
        self.dashboard_title = "GreenLang Compliance & Quality"

    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard"""
        return {
            "dashboard": {
                "uid": self.dashboard_uid,
                "title": self.dashboard_title,
                "tags": ["greenlang", "compliance", "quality"],
                "timezone": "browser",
                "schemaVersion": 38,
                "version": 1,
                "refresh": "5m",
                "time": {
                    "from": "now-7d",
                    "to": "now"
                },
                "panels": [
                    self._ium_compliance_gauge(),
                    self._adr_coverage_gauge(),
                    self._test_coverage_gauge(),
                    self._enforcement_violations(),
                    self._precommit_success_rate(),
                    self._pr_approval_time(),
                    self._review_feedback_categories(),
                    self._security_scan_results(),
                    self._violations_table(),
                    self._compliance_trend()
                ],
                "templating": {
                    "list": [
                        {
                            "name": "team",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_compliance_score, team)",
                            "multi": True,
                            "includeAll": True
                        },
                        {
                            "name": "application",
                            "type": "query",
                            "datasource": "Prometheus",
                            "query": "label_values(greenlang_compliance_score, application)",
                            "multi": True,
                            "includeAll": True
                        }
                    ]
                },
                "annotations": {
                    "list": [
                        {
                            "name": "Policy Violations",
                            "datasource": "Prometheus",
                            "enable": True,
                            "iconColor": "red",
                            "expr": "changes(greenlang_policy_violations_total[5m]) > 0"
                        }
                    ]
                }
            },
            "overwrite": True
        }

    def _ium_compliance_gauge(self) -> Dict[str, Any]:
        """IUM compliance for new PRs"""
        return {
            "id": 1,
            "title": "IUM Compliance (New PRs)",
            "type": "gauge",
            "gridPos": {"x": 0, "y": 0, "w": 6, "h": 6},
            "targets": [
                {
                    "expr": "avg(greenlang_pr_ium_score{team=~\"$team\", application=~\"$application\", pr_state=\"open\"})",
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
                            {"value": 90, "color": "yellow"},
                            {"value": 95, "color": "green"}
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

    def _adr_coverage_gauge(self) -> Dict[str, Any]:
        """ADR coverage percentage"""
        return {
            "id": 2,
            "title": "ADR Coverage",
            "type": "gauge",
            "gridPos": {"x": 6, "y": 0, "w": 6, "h": 6},
            "targets": [
                {
                    "expr": "(sum(greenlang_custom_code_with_adr{team=~\"$team\"}) / sum(greenlang_custom_code_total{team=~\"$team\"})) * 100",
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
                            {"value": 50, "color": "yellow"},
                            {"value": 80, "color": "green"}
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

    def _test_coverage_gauge(self) -> Dict[str, Any]:
        """Test coverage by application"""
        return {
            "id": 3,
            "title": "Test Coverage",
            "type": "gauge",
            "gridPos": {"x": 12, "y": 0, "w": 6, "h": 6},
            "targets": [
                {
                    "expr": "avg(greenlang_test_coverage_percent{application=~\"$application\"})",
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
                            {"value": 85, "color": "green"}
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

    def _enforcement_violations(self) -> Dict[str, Any]:
        """Enforcement violations by type"""
        return {
            "id": 4,
            "title": "Enforcement Violations by Type",
            "type": "barchart",
            "gridPos": {"x": 18, "y": 0, "w": 6, "h": 6},
            "targets": [
                {
                    "expr": "sum(increase(greenlang_policy_violations_total{team=~\"$team\"}[7d])) by (violation_type)",
                    "refId": "A",
                    "legendFormat": "{{violation_type}}"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "color": {
                        "mode": "palette-classic"
                    }
                }
            },
            "options": {
                "orientation": "horizontal",
                "showValue": "always",
                "legend": {
                    "displayMode": "hidden"
                }
            },
            "datasource": "Prometheus"
        }

    def _precommit_success_rate(self) -> Dict[str, Any]:
        """Pre-commit hook success rate"""
        return {
            "id": 5,
            "title": "Pre-commit Hook Success Rate",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 6, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "(sum(rate(greenlang_precommit_success_total{team=~\"$team\"}[5m])) / sum(rate(greenlang_precommit_attempts_total{team=~\"$team\"}[5m]))) * 100",
                    "refId": "A",
                    "legendFormat": "Success Rate"
                },
                {
                    "expr": "sum(rate(greenlang_precommit_disabled_total{team=~\"$team\"}[5m]))",
                    "refId": "B",
                    "legendFormat": "Disabled (violations)"
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
                        "matcher": {"id": "byName", "options": "Success Rate"},
                        "properties": [
                            {"id": "unit", "value": "percent"},
                            {"id": "color", "value": {"fixedColor": "green", "mode": "fixed"}}
                        ]
                    },
                    {
                        "matcher": {"id": "byName", "options": "Disabled (violations)"},
                        "properties": [
                            {"id": "unit", "value": "short"},
                            {"id": "color", "value": {"fixedColor": "red", "mode": "fixed"}},
                            {"id": "custom.axisPlacement", "value": "right"}
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

    def _pr_approval_time(self) -> Dict[str, Any]:
        """PR approval time distribution"""
        return {
            "id": 6,
            "title": "PR Approval Time Distribution",
            "type": "histogram",
            "gridPos": {"x": 12, "y": 6, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": "sum(rate(greenlang_pr_approval_duration_seconds_bucket{team=~\"$team\"}[1h])) by (le)",
                    "refId": "A",
                    "format": "heatmap"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "h"
                }
            },
            "options": {
                "calculate": True,
                "cellGap": 2,
                "color": {
                    "mode": "scheme",
                    "scheme": "Blues"
                },
                "yAxis": {
                    "axisLabel": "Hours to Approval",
                    "decimals": 1
                }
            },
            "datasource": "Prometheus"
        }

    def _review_feedback_categories(self) -> Dict[str, Any]:
        """Code review feedback categories"""
        return {
            "id": 7,
            "title": "Code Review Feedback Categories",
            "type": "piechart",
            "gridPos": {"x": 0, "y": 14, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "sum(increase(greenlang_review_comments_total{team=~\"$team\"}[7d])) by (category)",
                    "refId": "A",
                    "legendFormat": "{{category}}"
                }
            ],
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

    def _security_scan_results(self) -> Dict[str, Any]:
        """Security scan status indicators"""
        return {
            "id": 8,
            "title": "Security Scan Results",
            "type": "stat",
            "gridPos": {"x": 8, "y": 14, "w": 8, "h": 4},
            "targets": [
                {
                    "expr": "sum(greenlang_security_vulnerabilities{severity=\"critical\", application=~\"$application\"})",
                    "refId": "A"
                },
                {
                    "expr": "sum(greenlang_security_vulnerabilities{severity=\"high\", application=~\"$application\"})",
                    "refId": "B"
                },
                {
                    "expr": "sum(greenlang_security_vulnerabilities{severity=\"medium\", application=~\"$application\"})",
                    "refId": "C"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"value": 0, "color": "green"},
                            {"value": 1, "color": "red"}
                        ]
                    }
                },
                "overrides": [
                    {
                        "matcher": {"id": "byFrameRefID", "options": "A"},
                        "properties": [
                            {"id": "displayName", "value": "Critical"}
                        ]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "B"},
                        "properties": [
                            {"id": "displayName", "value": "High"},
                            {"id": "thresholds", "value": {
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 3, "color": "orange"}
                                ]
                            }}
                        ]
                    },
                    {
                        "matcher": {"id": "byFrameRefID", "options": "C"},
                        "properties": [
                            {"id": "displayName", "value": "Medium"},
                            {"id": "thresholds", "value": {
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 10, "color": "yellow"}
                                ]
                            }}
                        ]
                    }
                ]
            },
            "options": {
                "colorMode": "background",
                "graphMode": "none",
                "textMode": "value_and_name"
            },
            "datasource": "Prometheus"
        }

    def _violations_table(self) -> Dict[str, Any]:
        """Detailed violations table"""
        return {
            "id": 9,
            "title": "Recent Violations (Last 7 Days)",
            "type": "table",
            "gridPos": {"x": 16, "y": 14, "w": 8, "h": 8},
            "targets": [
                {
                    "expr": "topk(20, increase(greenlang_policy_violations_total{team=~\"$team\"}[7d]) > 0)",
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
                            "violation_type": "Violation Type",
                            "team": "Team",
                            "application": "Application",
                            "file_path": "File",
                            "Value": "Count"
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
                        "matcher": {"id": "byName", "options": "Count"},
                        "properties": [
                            {"id": "custom.displayMode", "value": "color-background"},
                            {"id": "color", "value": {
                                "mode": "thresholds"
                            }},
                            {"id": "thresholds", "value": {
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 5, "color": "yellow"},
                                    {"value": 10, "color": "red"}
                                ]
                            }}
                        ]
                    }
                ]
            },
            "datasource": "Prometheus"
        }

    def _compliance_trend(self) -> Dict[str, Any]:
        """Overall compliance trend"""
        return {
            "id": 10,
            "title": "Compliance Score Trend",
            "type": "timeseries",
            "gridPos": {"x": 0, "y": 22, "w": 24, "h": 8},
            "targets": [
                {
                    "expr": "avg(greenlang_compliance_score{team=~\"$team\", application=~\"$application\"}) by (application)",
                    "refId": "A",
                    "legendFormat": "{{application}}"
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
                        "showPoints": "never"
                    }
                }
            },
            "options": {
                "legend": {
                    "displayMode": "table",
                    "placement": "bottom",
                    "calcs": ["mean", "last", "min"]
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
            logger.info(f"Compliance dashboard exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export dashboard: {e}")
            raise


def main():
    """Main entry point"""
    dashboard = ComplianceDashboard()
    output_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\dashboards\\compliance.json"
    dashboard.export_to_file(output_path)
    print(f"Compliance & Quality Dashboard generated: {output_path}")


if __name__ == "__main__":
    main()
