"""
Real-Time Alerting System
=========================

Alert rules and notification handlers for GreenLang infrastructure monitoring.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels"""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"


class AlertRuleEngine:
    """
    Production-grade alerting system with multi-channel notifications.
    """

    def __init__(self):
        self.rules = self._define_alert_rules()

    def _define_alert_rules(self) -> List[Dict[str, Any]]:
        """Define all alert rules"""
        return [
            # IUM Alerts
            {
                "name": "IUM Below 90%",
                "expr": "avg(greenlang_ium_score) by (application) < 90",
                "for": "5m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.SLACK, AlertChannel.EMAIL],
                "annotations": {
                    "summary": "IUM dropped below 90% for {{ $labels.application }}",
                    "description": "Infrastructure Usage Metric is {{ $value }}% for {{ $labels.application }}. Target is >=95%.",
                    "runbook": "https://docs.greenlang.io/runbooks/ium-violation"
                },
                "labels": {
                    "category": "compliance",
                    "team": "platform"
                }
            },
            {
                "name": "New PR IUM Below 95%",
                "expr": "greenlang_pr_ium_score{pr_state=\"open\"} < 95",
                "for": "1m",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.SLACK],
                "annotations": {
                    "summary": "PR #{{ $labels.pr_number }} has IUM below 95%",
                    "description": "Pull request {{ $labels.pr_number }} in {{ $labels.application }} has IUM of {{ $value }}%.",
                    "action_required": "Review custom code and migrate to infrastructure components"
                },
                "labels": {
                    "category": "code_review",
                    "team": "{{ $labels.team }}"
                }
            },

            # Cache Performance Alerts
            {
                "name": "Semantic Cache Hit Rate Below 25%",
                "expr": "avg(greenlang_cache_hit_rate{cache_type=\"semantic\"}) < 0.25",
                "for": "10m",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.SLACK, AlertChannel.EMAIL],
                "annotations": {
                    "summary": "Semantic cache hit rate is {{ $value | humanizePercentage }}",
                    "description": "Cache efficiency is below target. This may increase LLM costs.",
                    "investigation": "Check cache configuration, TTL settings, and query patterns"
                },
                "labels": {
                    "category": "performance",
                    "component": "cache"
                }
            },

            # Performance Alerts
            {
                "name": "Agent Execution Time Above Baseline",
                "expr": "histogram_quantile(0.95, rate(greenlang_agent_execution_seconds_bucket[5m])) > (2 * histogram_quantile(0.95, rate(greenlang_agent_execution_seconds_bucket[1h] offset 1d)))",
                "for": "5m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.PAGERDUTY, AlertChannel.SLACK],
                "annotations": {
                    "summary": "Agent {{ $labels.agent }} execution time is 2x baseline",
                    "description": "P95 latency is {{ $value }}s, which is 2x the daily baseline.",
                    "runbook": "https://docs.greenlang.io/runbooks/agent-performance"
                },
                "labels": {
                    "category": "performance",
                    "severity": "high"
                }
            },
            {
                "name": "P95 Latency Above 1 Second",
                "expr": "histogram_quantile(0.95, rate(greenlang_request_duration_seconds_bucket[5m])) > 1",
                "for": "5m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.PAGERDUTY, AlertChannel.SLACK],
                "annotations": {
                    "summary": "P95 latency is {{ $value }}s for {{ $labels.service }}",
                    "description": "SLA violation: P95 latency exceeded 1 second threshold.",
                    "dashboard": "https://grafana.greenlang.io/d/greenlang-performance"
                },
                "labels": {
                    "category": "sla",
                    "severity": "high"
                }
            },
            {
                "name": "Error Rate Above 1%",
                "expr": "(sum(rate(greenlang_request_errors_total[5m])) by (service) / sum(rate(greenlang_request_total[5m])) by (service)) > 0.01",
                "for": "5m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.PAGERDUTY, AlertChannel.SLACK],
                "annotations": {
                    "summary": "Error rate is {{ $value | humanizePercentage }} for {{ $labels.service }}",
                    "description": "Service {{ $labels.service }} error rate exceeded 1% threshold.",
                    "dashboard": "https://grafana.greenlang.io/d/greenlang-performance"
                },
                "labels": {
                    "category": "reliability",
                    "severity": "high"
                }
            },

            # Security Alerts
            {
                "name": "Pre-commit Hook Disabled",
                "expr": "increase(greenlang_precommit_disabled_total[5m]) > 0",
                "for": "1m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.SLACK, AlertChannel.EMAIL],
                "annotations": {
                    "summary": "Pre-commit hook disabled by {{ $labels.developer }}",
                    "description": "Developer {{ $labels.developer }} in team {{ $labels.team }} disabled pre-commit hooks.",
                    "action_required": "Immediate review required. This violates security policy."
                },
                "labels": {
                    "category": "security",
                    "severity": "critical"
                }
            },
            {
                "name": "Custom LLM Wrapper Detected",
                "expr": "increase(greenlang_custom_llm_wrapper_violations[5m]) > 0",
                "for": "1m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.SLACK, AlertChannel.EMAIL],
                "annotations": {
                    "summary": "Custom LLM wrapper detected in {{ $labels.file_path }}",
                    "description": "Forbidden custom LLM implementation found. Must use infrastructure components.",
                    "action_required": "Block PR and require migration to LLMClient"
                },
                "labels": {
                    "category": "compliance",
                    "violation_type": "forbidden_pattern"
                }
            },
            {
                "name": "Critical Security Vulnerability",
                "expr": "greenlang_security_vulnerabilities{severity=\"critical\"} > 0",
                "for": "1m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
                "annotations": {
                    "summary": "{{ $value }} critical vulnerabilities in {{ $labels.application }}",
                    "description": "Critical security vulnerabilities detected. Immediate remediation required.",
                    "runbook": "https://docs.greenlang.io/runbooks/security-vulnerability"
                },
                "labels": {
                    "category": "security",
                    "severity": "critical"
                }
            },

            # Compliance Alerts
            {
                "name": "Missing ADR for Large Custom Code",
                "expr": "greenlang_custom_code_without_adr_lines > 100",
                "for": "5m",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.SLACK],
                "annotations": {
                    "summary": "{{ $labels.file_path }} has {{ $value }} lines of custom code without ADR",
                    "description": "File {{ $labels.file_path }} requires an Architecture Decision Record.",
                    "action_required": "Create ADR explaining why custom code is necessary"
                },
                "labels": {
                    "category": "compliance",
                    "team": "{{ $labels.team }}"
                }
            },
            {
                "name": "Test Coverage Below 85%",
                "expr": "greenlang_test_coverage_percent < 85",
                "for": "10m",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.SLACK],
                "annotations": {
                    "summary": "Test coverage is {{ $value }}% for {{ $labels.application }}",
                    "description": "Test coverage dropped below 85% target.",
                    "action_required": "Add tests to restore coverage above 85%"
                },
                "labels": {
                    "category": "quality",
                    "team": "{{ $labels.team }}"
                }
            },

            # Infrastructure Health Alerts
            {
                "name": "Service Down",
                "expr": "up{job=~\"greenlang-.*\"} == 0",
                "for": "1m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.PAGERDUTY, AlertChannel.SLACK],
                "annotations": {
                    "summary": "Service {{ $labels.job }} is down",
                    "description": "{{ $labels.instance }} has been down for more than 1 minute.",
                    "runbook": "https://docs.greenlang.io/runbooks/service-down"
                },
                "labels": {
                    "category": "availability",
                    "severity": "critical"
                }
            },
            {
                "name": "High Resource Utilization",
                "expr": "(greenlang_resource_usage_percent{resource=~\"cpu|memory|disk\"} > 80)",
                "for": "10m",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.SLACK],
                "annotations": {
                    "summary": "{{ $labels.resource }} utilization is {{ $value }}% on {{ $labels.instance }}",
                    "description": "Resource usage is above 80%. Consider scaling.",
                    "investigation": "Check for resource leaks or increase capacity"
                },
                "labels": {
                    "category": "capacity",
                    "resource": "{{ $labels.resource }}"
                }
            },
            {
                "name": "Redis Cache Service Unavailable",
                "expr": "greenlang_cache_service_available{service=\"redis\"} == 0",
                "for": "1m",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.PAGERDUTY, AlertChannel.SLACK],
                "annotations": {
                    "summary": "Redis cache service is unavailable",
                    "description": "L2 cache unavailable. Falling back to database queries.",
                    "runbook": "https://docs.greenlang.io/runbooks/redis-down"
                },
                "labels": {
                    "category": "availability",
                    "component": "cache"
                }
            },
            {
                "name": "LLM Provider Rate Limit",
                "expr": "rate(greenlang_llm_rate_limit_errors_total[5m]) > 0.1",
                "for": "5m",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.SLACK],
                "annotations": {
                    "summary": "LLM provider {{ $labels.provider }} rate limiting",
                    "description": "Experiencing rate limits from {{ $labels.provider }}. Consider increasing quota or implementing backoff.",
                    "investigation": "Check current usage against quota limits"
                },
                "labels": {
                    "category": "integration",
                    "provider": "{{ $labels.provider }}"
                }
            }
        ]

    def export_prometheus_rules(self, output_path: str) -> None:
        """
        Export alert rules in Prometheus format.

        Args:
            output_path: Path to save rules file
        """
        prometheus_rules = {
            "groups": [
                {
                    "name": "greenlang_alerts",
                    "interval": "30s",
                    "rules": [
                        {
                            "alert": rule["name"],
                            "expr": rule["expr"],
                            "for": rule["for"],
                            "labels": {
                                **rule["labels"],
                                "severity": rule["severity"].value
                            },
                            "annotations": rule["annotations"]
                        }
                        for rule in self.rules
                    ]
                }
            ]
        }

        try:
            with open(output_path, 'w') as f:
                json.dump(prometheus_rules, f, indent=2)
            logger.info(f"Prometheus alert rules exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export alert rules: {e}")
            raise

    def export_grafana_alerts(self, output_path: str) -> None:
        """
        Export alert rules in Grafana format.

        Args:
            output_path: Path to save Grafana alert configuration
        """
        grafana_alerts = []

        for rule in self.rules:
            alert = {
                "uid": rule["name"].lower().replace(" ", "-"),
                "title": rule["name"],
                "condition": rule["expr"],
                "data": [
                    {
                        "refId": "A",
                        "queryType": "",
                        "relativeTimeRange": {
                            "from": 600,
                            "to": 0
                        },
                        "datasourceUid": "prometheus",
                        "model": {
                            "expr": rule["expr"],
                            "refId": "A"
                        }
                    }
                ],
                "noDataState": "NoData",
                "execErrState": "Alerting",
                "for": rule["for"],
                "annotations": rule["annotations"],
                "labels": {
                    **rule["labels"],
                    "severity": rule["severity"].value
                },
                "isPaused": False
            }
            grafana_alerts.append(alert)

        try:
            with open(output_path, 'w') as f:
                json.dump(grafana_alerts, f, indent=2)
            logger.info(f"Grafana alert rules exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export Grafana alerts: {e}")
            raise


class NotificationHandler:
    """
    Multi-channel notification handler for alerts.
    """

    def __init__(self, config: Dict[str, Any]):
        self.slack_webhook = config.get("slack_webhook_url")
        self.email_config = config.get("email")
        self.pagerduty_key = config.get("pagerduty_integration_key")

    def send_slack_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert to Slack"""
        if not self.slack_webhook:
            logger.warning("Slack webhook not configured")
            return

        severity_colors = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "critical": "#ff0000"
        }

        payload = {
            "attachments": [
                {
                    "color": severity_colors.get(alert["severity"], "#808080"),
                    "title": f":rotating_light: {alert['name']}",
                    "text": alert["annotations"]["summary"],
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert["severity"].upper(),
                            "short": True
                        },
                        {
                            "title": "Category",
                            "value": alert["labels"].get("category", "N/A"),
                            "short": True
                        },
                        {
                            "title": "Description",
                            "value": alert["annotations"]["description"],
                            "short": False
                        }
                    ],
                    "footer": "GreenLang Monitoring",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }

        if "runbook" in alert["annotations"]:
            payload["attachments"][0]["fields"].append({
                "title": "Runbook",
                "value": f"<{alert['annotations']['runbook']}|View Runbook>",
                "short": False
            })

        try:
            response = requests.post(self.slack_webhook, json=payload)
            response.raise_for_status()
            logger.info(f"Slack notification sent for alert: {alert['name']}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def send_email_notification(self, alert: Dict[str, Any]) -> None:
        """Send alert via email"""
        if not self.email_config:
            logger.warning("Email not configured")
            return

        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert['severity'].upper()}] {alert['name']}"
        msg['From'] = self.email_config['from']
        msg['To'] = ", ".join(self.email_config['to'])

        html = f"""
        <html>
        <body>
            <h2 style="color: {'#ff0000' if alert['severity'] == 'critical' else '#ff9900'};">
                {alert['name']}
            </h2>
            <p><strong>Summary:</strong> {alert['annotations']['summary']}</p>
            <p><strong>Description:</strong> {alert['annotations']['description']}</p>
            <p><strong>Severity:</strong> {alert['severity'].upper()}</p>
            <p><strong>Category:</strong> {alert['labels'].get('category', 'N/A')}</p>
            {"<p><a href='" + alert['annotations']['runbook'] + "'>View Runbook</a></p>" if 'runbook' in alert['annotations'] else ""}
            <hr>
            <p style="color: #666; font-size: 12px;">
                GreenLang Monitoring System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        try:
            with smtplib.SMTP(self.email_config['smtp_host'], self.email_config['smtp_port']) as server:
                if self.email_config.get('use_tls'):
                    server.starttls()
                if self.email_config.get('username'):
                    server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            logger.info(f"Email notification sent for alert: {alert['name']}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def send_pagerduty_alert(self, alert: Dict[str, Any]) -> None:
        """Send critical alert to PagerDuty"""
        if not self.pagerduty_key:
            logger.warning("PagerDuty not configured")
            return

        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert["annotations"]["summary"],
                "severity": alert["severity"],
                "source": "greenlang-monitoring",
                "custom_details": {
                    "description": alert["annotations"]["description"],
                    "category": alert["labels"].get("category"),
                    "runbook": alert["annotations"].get("runbook")
                }
            }
        }

        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"PagerDuty alert sent for: {alert['name']}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")


def main():
    """Main entry point"""
    engine = AlertRuleEngine()

    # Export Prometheus rules
    prometheus_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\alerts\\prometheus_rules.json"
    engine.export_prometheus_rules(prometheus_path)
    print(f"Prometheus alert rules exported: {prometheus_path}")

    # Export Grafana alerts
    grafana_path = "C:\\Users\\aksha\\Code-V1_GreenLang\\greenlang\\monitoring\\alerts\\grafana_alerts.json"
    engine.export_grafana_alerts(grafana_path)
    print(f"Grafana alert rules exported: {grafana_path}")

    print(f"\nTotal alert rules configured: {len(engine.rules)}")
    print("\nAlert categories:")
    categories = {}
    for rule in engine.rules:
        cat = rule["labels"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in categories.items():
        print(f"  - {cat}: {count} rules")


if __name__ == "__main__":
    main()
