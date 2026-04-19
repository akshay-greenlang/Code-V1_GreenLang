# -*- coding: utf-8 -*-
"""
Monitoring service with alerting and dashboards for GreenLang
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import yaml

from .metrics import get_metrics_collector
from .tracing import get_tracing_manager
from .health import get_health_checker
from .logging import get_log_aggregator
from .performance import get_performance_monitor
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""

    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Alert:
    """Alert definition"""

    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fired_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["severity"] = self.severity.value
        data["status"] = self.status.value
        data["fired_at"] = self.fired_at.isoformat()
        if self.resolved_at:
            data["resolved_at"] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            data["acknowledged_at"] = self.acknowledged_at.isoformat()
        return data


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    expression: str  # Metric query expression
    condition: str  # Condition (e.g., "> 0.9")
    duration: int  # Duration in seconds before firing
    severity: AlertSeverity
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def evaluate(self, value: float) -> bool:
        """
        Evaluate alert condition

        Args:
            value: Metric value

        Returns:
            True if condition met
        """
        try:
            # Simple evaluation (in production, use safe expression parser)
            if self.condition.startswith(">"):
                threshold = float(self.condition[1:].strip())
                return value > threshold
            elif self.condition.startswith("<"):
                threshold = float(self.condition[1:].strip())
                return value < threshold
            elif self.condition.startswith("=="):
                threshold = float(self.condition[2:].strip())
                return value == threshold
            elif self.condition.startswith("!="):
                threshold = float(self.condition[2:].strip())
                return value != threshold
            else:
                return False
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return False


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self):
        """Initialize alert manager"""
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self._evaluation_thread = None
        self._stop_evaluation = threading.Event()
        self.evaluation_interval = 30  # seconds

        # Track when conditions first met
        self._pending_alerts: Dict[str, datetime] = {}

        logger.info("AlertManager initialized")

    def add_rule(self, rule: AlertRule):
        """
        Add alert rule

        Args:
            rule: Alert rule
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str):
        """
        Remove alert rule

        Args:
            name: Rule name
        """
        if name in self.rules:
            del self.rules[name]
            logger.info(f"Removed alert rule: {name}")

    def add_handler(self, handler: Callable[[Alert], None]):
        """
        Add alert handler

        Args:
            handler: Function to handle alerts
        """
        self.handlers.append(handler)

    def fire_alert(self, rule: AlertRule, message: str):
        """
        Fire an alert

        Args:
            rule: Alert rule that triggered
            message: Alert message
        """
        with self._lock:
            alert_id = f"{rule.name}_{int(time.time())}"

            alert = Alert(
                alert_id=alert_id,
                name=rule.name,
                severity=rule.severity,
                status=AlertStatus.FIRING,
                message=message,
                labels=rule.labels,
                annotations=rule.annotations,
            )

            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Notify handlers
            for handler in self.handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

            logger.warning(f"Alert fired: {rule.name} - {message}")

    def resolve_alert(self, alert_id: str):
        """
        Resolve an alert

        Args:
            alert_id: Alert ID
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = DeterministicClock.utcnow()

                del self.active_alerts[alert_id]

                logger.info(f"Alert resolved: {alert.name}")

    def acknowledge_alert(self, alert_id: str, user: str):
        """
        Acknowledge an alert

        Args:
            alert_id: Alert ID
            user: User acknowledging
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = DeterministicClock.utcnow()
                alert.acknowledged_by = user

                logger.info(f"Alert acknowledged: {alert.name} by {user}")

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff = DeterministicClock.utcnow() - timedelta(hours=hours)

        with self._lock:
            return [alert for alert in self.alert_history if alert.fired_at >= cutoff]

    def start_evaluation(self):
        """Start alert evaluation"""
        self._evaluation_thread = threading.Thread(
            target=self._evaluation_loop, daemon=True
        )
        self._evaluation_thread.start()
        logger.info("Started alert evaluation")

    def stop_evaluation(self):
        """Stop alert evaluation"""
        self._stop_evaluation.set()
        if self._evaluation_thread:
            self._evaluation_thread.join(timeout=5)
        logger.info("Stopped alert evaluation")

    def _evaluation_loop(self):
        """Alert evaluation loop"""
        while not self._stop_evaluation.is_set():
            try:
                self._evaluate_rules()
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")

            self._stop_evaluation.wait(self.evaluation_interval)

    def _evaluate_rules(self):
        """Evaluate all alert rules"""
        metrics_collector = get_metrics_collector()

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            try:
                # Get metric value (simplified - in production use proper query engine)
                # For now, just check system metrics
                value = self._get_metric_value(rule.expression)

                if value is not None and rule.evaluate(value):
                    # Check duration requirement
                    if rule.name not in self._pending_alerts:
                        self._pending_alerts[rule.name] = DeterministicClock.utcnow()
                    elif (
                        DeterministicClock.utcnow() - self._pending_alerts[rule.name]
                    ).total_seconds() >= rule.duration:
                        # Fire alert
                        self.fire_alert(
                            rule, f"{rule.expression} {rule.condition} (value: {value})"
                        )
                        del self._pending_alerts[rule.name]
                else:
                    # Condition not met, clear pending
                    if rule.name in self._pending_alerts:
                        del self._pending_alerts[rule.name]

                    # Check if we should resolve any active alerts for this rule
                    for alert_id, alert in list(self.active_alerts.items()):
                        if alert.name == rule.name:
                            self.resolve_alert(alert_id)

            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

    def _get_metric_value(self, expression: str) -> Optional[float]:
        """Get metric value for expression"""
        # Simplified metric retrieval
        # In production, implement proper metric query language

        if "cpu_usage" in expression:
            import psutil

            return psutil.cpu_percent()
        elif "memory_usage" in expression:
            import psutil

            return psutil.virtual_memory().percent
        elif "disk_usage" in expression:
            import psutil

            return psutil.disk_usage("/").percent

        return None


@dataclass
class Dashboard:
    """Dashboard configuration"""

    name: str
    title: str
    description: str = ""
    refresh_interval: int = 60  # seconds
    panels: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_grafana_json(self) -> Dict[str, Any]:
        """Convert to Grafana dashboard JSON"""
        return {
            "dashboard": {
                "title": self.title,
                "description": self.description,
                "refresh": f"{self.refresh_interval}s",
                "panels": self._convert_panels_to_grafana(),
                "templating": {"list": self._convert_variables_to_grafana()},
                "time": {"from": "now-6h", "to": "now"},
                "timezone": "browser",
            }
        }

    def _convert_panels_to_grafana(self) -> List[Dict[str, Any]]:
        """Convert panels to Grafana format"""
        grafana_panels = []

        for i, panel in enumerate(self.panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel.get("title", f"Panel {i+1}"),
                "type": panel.get("type", "graph"),
                "gridPos": panel.get("gridPos", {"x": 0, "y": i * 8, "w": 12, "h": 8}),
                "targets": panel.get("targets", []),
                "options": panel.get("options", {}),
                "fieldConfig": panel.get("fieldConfig", {}),
            }
            grafana_panels.append(grafana_panel)

        return grafana_panels

    def _convert_variables_to_grafana(self) -> List[Dict[str, Any]]:
        """Convert variables to Grafana format"""
        grafana_vars = []

        for name, config in self.variables.items():
            grafana_var = {
                "name": name,
                "label": config.get("label", name),
                "type": config.get("type", "query"),
                "query": config.get("query", ""),
                "current": config.get("default", {}),
            }
            grafana_vars.append(grafana_var)

        return grafana_vars


class MonitoringService:
    """Central monitoring service"""

    def __init__(self):
        """Initialize monitoring service"""
        self.metrics_collector = get_metrics_collector()
        self.tracing_manager = get_tracing_manager()
        self.health_checker = get_health_checker()
        self.log_aggregator = get_log_aggregator()
        self.performance_monitor = get_performance_monitor()
        self.alert_manager = AlertManager()

        self.dashboards: Dict[str, Dashboard] = {}

        # Initialize default alert rules
        self._init_default_alerts()

        # Initialize default dashboards
        self._init_default_dashboards()

        logger.info("MonitoringService initialized")

    def _init_default_alerts(self):
        """Initialize default alert rules"""
        # High CPU usage
        self.alert_manager.add_rule(
            AlertRule(
                name="high_cpu_usage",
                expression="cpu_usage",
                condition="> 80",
                duration=300,  # 5 minutes
                severity=AlertSeverity.WARNING,
                annotations={"description": "CPU usage is above 80%"},
            )
        )

        # High memory usage
        self.alert_manager.add_rule(
            AlertRule(
                name="high_memory_usage",
                expression="memory_usage",
                condition="> 90",
                duration=300,
                severity=AlertSeverity.WARNING,
                annotations={"description": "Memory usage is above 90%"},
            )
        )

        # Low disk space
        self.alert_manager.add_rule(
            AlertRule(
                name="low_disk_space",
                expression="disk_usage",
                condition="> 85",
                duration=600,
                severity=AlertSeverity.ERROR,
                annotations={"description": "Disk usage is above 85%"},
            )
        )

    def _init_default_dashboards(self):
        """Initialize default dashboards"""
        # System overview dashboard
        system_dashboard = Dashboard(
            name="system_overview",
            title="System Overview",
            description="System metrics overview",
            refresh_interval=30,
            panels=[
                {
                    "title": "CPU Usage",
                    "type": "graph",
                    "targets": [{"expr": "gl_cpu_usage_percent"}],
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                },
                {
                    "title": "Memory Usage",
                    "type": "graph",
                    "targets": [{"expr": "gl_memory_usage_bytes"}],
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                },
                {
                    "title": "Active Pipelines",
                    "type": "stat",
                    "targets": [{"expr": "gl_active_executions"}],
                    "gridPos": {"x": 0, "y": 8, "w": 6, "h": 4},
                },
                {
                    "title": "Pipeline Success Rate",
                    "type": "gauge",
                    "targets": [
                        {"expr": "rate(gl_pipeline_runs_total{status='success'}[5m])"}
                    ],
                    "gridPos": {"x": 6, "y": 8, "w": 6, "h": 4},
                },
            ],
        )
        self.dashboards["system_overview"] = system_dashboard

        # Pipeline metrics dashboard
        pipeline_dashboard = Dashboard(
            name="pipeline_metrics",
            title="Pipeline Metrics",
            description="Pipeline execution metrics",
            refresh_interval=60,
            panels=[
                {
                    "title": "Pipeline Runs",
                    "type": "graph",
                    "targets": [{"expr": "rate(gl_pipeline_runs_total[5m])"}],
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                },
                {
                    "title": "Pipeline Duration",
                    "type": "heatmap",
                    "targets": [{"expr": "gl_pipeline_duration_seconds"}],
                    "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
                },
                {
                    "title": "Error Rate",
                    "type": "graph",
                    "targets": [{"expr": "rate(gl_errors_total[5m])"}],
                    "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8},
                },
            ],
            variables={
                "pipeline": {
                    "label": "Pipeline",
                    "type": "query",
                    "query": "label_values(gl_pipeline_runs_total, pipeline)",
                }
            },
        )
        self.dashboards["pipeline_metrics"] = pipeline_dashboard

    def start(self, metrics_port: int = 8000):
        """
        Start monitoring service

        Args:
            metrics_port: Port for metrics endpoint
        """
        # Start metrics collection
        self.metrics_collector.start_collection(metrics_port)

        # Start health checks
        self.health_checker.start_background_checks()

        # Start performance monitoring
        self.performance_monitor.start_monitoring()

        # Start alert evaluation
        self.alert_manager.start_evaluation()

        logger.info(f"Monitoring service started (metrics port: {metrics_port})")

    def stop(self):
        """Stop monitoring service"""
        self.metrics_collector.stop_collection()
        self.health_checker.stop_background_checks()
        self.performance_monitor.stop_monitoring()
        self.alert_manager.stop_evaluation()

        logger.info("Monitoring service stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            "health": self.health_checker.get_status().value,
            "active_alerts": len(self.alert_manager.active_alerts),
            "metrics_count": len(self.metrics_collector.custom_metrics),
            "log_count": self.log_aggregator.get_statistics()["total_logs"],
            "dashboards": list(self.dashboards.keys()),
        }

    def export_dashboard(self, name: str, format: str = "json") -> str:
        """
        Export dashboard configuration

        Args:
            name: Dashboard name
            format: Export format (json, yaml, grafana)

        Returns:
            Dashboard configuration
        """
        dashboard = self.dashboards.get(name)
        if not dashboard:
            raise ValueError(f"Dashboard {name} not found")

        if format == "json":
            return json.dumps(dashboard.to_dict(), indent=2)
        elif format == "yaml":
            return yaml.dump(dashboard.to_dict())
        elif format == "grafana":
            return json.dumps(dashboard.to_grafana_json(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

    def import_dashboard(self, config: Dict[str, Any]):
        """
        Import dashboard configuration

        Args:
            config: Dashboard configuration
        """
        dashboard = Dashboard(**config)
        self.dashboards[dashboard.name] = dashboard
        logger.info(f"Imported dashboard: {dashboard.name}")


# Global monitoring service
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get global monitoring service"""
    global _monitoring_service
    if not _monitoring_service:
        _monitoring_service = MonitoringService()
    return _monitoring_service
