# -*- coding: utf-8 -*-
"""
GL-ECO-X-007: Agent Performance Monitor
========================================

Monitors agent performance, resource usage, and health metrics.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ResourceUsage(BaseModel):
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    cpu_percent: float = Field(default=0.0, ge=0, le=100)
    memory_mb: float = Field(default=0.0, ge=0)
    memory_percent: float = Field(default=0.0, ge=0, le=100)
    disk_io_mb: float = Field(default=0.0, ge=0)
    network_io_mb: float = Field(default=0.0, ge=0)


class AgentPerformanceMetrics(BaseModel):
    agent_id: str = Field(..., description="Agent identifier")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)

    # Execution metrics
    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    success_rate: float = Field(default=100.0, ge=0, le=100)

    # Timing metrics
    avg_execution_time_ms: float = Field(default=0.0, ge=0)
    min_execution_time_ms: float = Field(default=0.0, ge=0)
    max_execution_time_ms: float = Field(default=0.0, ge=0)
    p95_execution_time_ms: float = Field(default=0.0, ge=0)

    # Resource usage
    resource_usage: Optional[ResourceUsage] = Field(None)

    # Health
    health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)


class PerformanceThreshold(BaseModel):
    threshold_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = Field(..., description="Agent to monitor")
    metric_name: str = Field(..., description="Metric name")
    operator: str = Field(..., description="gt/lt/eq")
    threshold_value: float = Field(..., description="Threshold value")
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    enabled: bool = Field(default=True)


class PerformanceAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = Field(..., description="Agent with issue")
    threshold_id: str = Field(..., description="Threshold that triggered")
    severity: AlertSeverity = Field(...)
    message: str = Field(..., description="Alert message")
    metric_value: float = Field(..., description="Actual metric value")
    threshold_value: float = Field(..., description="Threshold value")
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = Field(None)


class PerformanceMonitorInput(BaseModel):
    operation: str = Field(..., description="Operation to perform")
    agent_id: Optional[str] = Field(None)
    metrics: Optional[AgentPerformanceMetrics] = Field(None)
    threshold: Optional[PerformanceThreshold] = Field(None)
    alert_id: Optional[str] = Field(None)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {
            'record_metrics', 'get_metrics', 'get_trends',
            'add_threshold', 'remove_threshold', 'check_thresholds',
            'get_alerts', 'resolve_alert', 'get_health_status',
            'get_summary', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class PerformanceMonitorOutput(BaseModel):
    success: bool = Field(...)
    operation: str = Field(...)
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class AgentPerformanceMonitor(BaseAgent):
    """GL-ECO-X-007: Agent Performance Monitor"""

    AGENT_ID = "GL-ECO-X-007"
    AGENT_NAME = "Agent Performance Monitor"
    VERSION = "1.0.0"
    MAX_HISTORY = 1000

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Monitors agent performance metrics",
                version=self.VERSION,
            )
        super().__init__(config)

        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_HISTORY))
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        self._alerts: Dict[str, PerformanceAlert] = {}
        self._health_status: Dict[str, HealthStatus] = {}
        self._total_metrics_recorded = 0

        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()
        try:
            pm_input = PerformanceMonitorInput(**input_data)
            result_data = self._route_operation(pm_input)
            provenance_hash = hashlib.sha256(
                json.dumps({"in": input_data, "out": result_data}, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            output = PerformanceMonitorOutput(
                success=True, operation=pm_input.operation, data=result_data,
                provenance_hash=provenance_hash, processing_time_ms=(time.time() - start_time) * 1000,
            )
            return AgentResult(success=True, data=output.model_dump())
        except Exception as e:
            self.logger.error(f"Operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, pm_input: PerformanceMonitorInput) -> Dict[str, Any]:
        op = pm_input.operation
        if op == "record_metrics":
            return self._record_metrics(pm_input.metrics)
        elif op == "get_metrics":
            return self._get_metrics(pm_input.agent_id, pm_input.start_time, pm_input.end_time)
        elif op == "get_trends":
            return self._get_trends(pm_input.agent_id)
        elif op == "add_threshold":
            return self._add_threshold(pm_input.threshold)
        elif op == "remove_threshold":
            return self._remove_threshold(pm_input.threshold.threshold_id if pm_input.threshold else None)
        elif op == "check_thresholds":
            return self._check_thresholds(pm_input.agent_id)
        elif op == "get_alerts":
            return self._get_alerts(pm_input.agent_id)
        elif op == "resolve_alert":
            return self._resolve_alert(pm_input.alert_id)
        elif op == "get_health_status":
            return self._get_health_status(pm_input.agent_id)
        elif op == "get_summary":
            return self._get_summary()
        elif op == "get_statistics":
            return self._get_statistics()
        raise ValueError(f"Unknown operation: {op}")

    def _record_metrics(self, metrics: Optional[AgentPerformanceMetrics]) -> Dict[str, Any]:
        if not metrics:
            return {"error": "metrics required"}

        self._metrics[metrics.agent_id].append(metrics)
        self._total_metrics_recorded += 1

        # Update health status
        self._update_health_status(metrics)

        # Check thresholds
        alerts = self._check_agent_thresholds(metrics)

        return {"agent_id": metrics.agent_id, "recorded": True, "alerts_triggered": len(alerts)}

    def _update_health_status(self, metrics: AgentPerformanceMetrics) -> None:
        """Update health status based on metrics."""
        if metrics.success_rate >= 99:
            status = HealthStatus.HEALTHY
        elif metrics.success_rate >= 95:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        self._health_status[metrics.agent_id] = status

    def _check_agent_thresholds(self, metrics: AgentPerformanceMetrics) -> List[PerformanceAlert]:
        """Check thresholds for an agent."""
        alerts = []

        for threshold in self._thresholds.values():
            if threshold.agent_id != metrics.agent_id or not threshold.enabled:
                continue

            value = self._get_metric_value(metrics, threshold.metric_name)
            if value is None:
                continue

            violated = False
            if threshold.operator == "gt" and value > threshold.threshold_value:
                violated = True
            elif threshold.operator == "lt" and value < threshold.threshold_value:
                violated = True
            elif threshold.operator == "eq" and value == threshold.threshold_value:
                violated = True

            if violated:
                alert = PerformanceAlert(
                    agent_id=metrics.agent_id,
                    threshold_id=threshold.threshold_id,
                    severity=threshold.severity,
                    message=f"{threshold.metric_name} {threshold.operator} {threshold.threshold_value}",
                    metric_value=value,
                    threshold_value=threshold.threshold_value,
                )
                self._alerts[alert.alert_id] = alert
                alerts.append(alert)

        return alerts

    def _get_metric_value(self, metrics: AgentPerformanceMetrics, metric_name: str) -> Optional[float]:
        """Get metric value by name."""
        mapping = {
            "success_rate": metrics.success_rate,
            "avg_execution_time_ms": metrics.avg_execution_time_ms,
            "max_execution_time_ms": metrics.max_execution_time_ms,
            "p95_execution_time_ms": metrics.p95_execution_time_ms,
            "total_executions": metrics.total_executions,
            "failed_executions": metrics.failed_executions,
        }
        return mapping.get(metric_name)

    def _get_metrics(
        self, agent_id: Optional[str], start_time: Optional[datetime], end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        if not agent_id:
            return {"error": "agent_id required"}

        metrics_list = list(self._metrics.get(agent_id, []))

        if start_time:
            metrics_list = [m for m in metrics_list if m.timestamp >= start_time]
        if end_time:
            metrics_list = [m for m in metrics_list if m.timestamp <= end_time]

        return {"agent_id": agent_id, "metrics": [m.model_dump() for m in metrics_list], "count": len(metrics_list)}

    def _get_trends(self, agent_id: Optional[str]) -> Dict[str, Any]:
        if not agent_id or agent_id not in self._metrics:
            return {"error": f"No metrics for agent: {agent_id}"}

        metrics_list = list(self._metrics[agent_id])
        if len(metrics_list) < 2:
            return {"agent_id": agent_id, "message": "Insufficient data for trends"}

        first = metrics_list[0]
        last = metrics_list[-1]

        return {
            "agent_id": agent_id,
            "trends": {
                "success_rate_change": last.success_rate - first.success_rate,
                "avg_time_change_ms": last.avg_execution_time_ms - first.avg_execution_time_ms,
                "executions_total": last.total_executions,
            },
            "data_points": len(metrics_list),
        }

    def _add_threshold(self, threshold: Optional[PerformanceThreshold]) -> Dict[str, Any]:
        if not threshold:
            return {"error": "threshold required"}
        self._thresholds[threshold.threshold_id] = threshold
        return {"threshold_id": threshold.threshold_id, "added": True}

    def _remove_threshold(self, threshold_id: Optional[str]) -> Dict[str, Any]:
        if threshold_id and threshold_id in self._thresholds:
            del self._thresholds[threshold_id]
            return {"threshold_id": threshold_id, "removed": True}
        return {"error": f"Threshold not found: {threshold_id}"}

    def _check_thresholds(self, agent_id: Optional[str]) -> Dict[str, Any]:
        thresholds = list(self._thresholds.values())
        if agent_id:
            thresholds = [t for t in thresholds if t.agent_id == agent_id]

        return {"thresholds": [t.model_dump() for t in thresholds], "count": len(thresholds)}

    def _get_alerts(self, agent_id: Optional[str]) -> Dict[str, Any]:
        alerts = list(self._alerts.values())
        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]

        active_alerts = [a for a in alerts if not a.resolved]
        return {"alerts": [a.model_dump() for a in active_alerts], "count": len(active_alerts)}

    def _resolve_alert(self, alert_id: Optional[str]) -> Dict[str, Any]:
        if not alert_id or alert_id not in self._alerts:
            return {"error": f"Alert not found: {alert_id}"}

        alert = self._alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = DeterministicClock.now()

        return {"alert_id": alert_id, "resolved": True}

    def _get_health_status(self, agent_id: Optional[str]) -> Dict[str, Any]:
        if agent_id:
            status = self._health_status.get(agent_id, HealthStatus.UNKNOWN)
            return {"agent_id": agent_id, "health_status": status.value}

        return {
            "health_by_agent": {aid: status.value for aid, status in self._health_status.items()},
            "summary": {
                "healthy": sum(1 for s in self._health_status.values() if s == HealthStatus.HEALTHY),
                "degraded": sum(1 for s in self._health_status.values() if s == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for s in self._health_status.values() if s == HealthStatus.UNHEALTHY),
            },
        }

    def _get_summary(self) -> Dict[str, Any]:
        return {
            "agents_monitored": len(self._metrics),
            "total_metrics_recorded": self._total_metrics_recorded,
            "active_thresholds": len(self._thresholds),
            "active_alerts": sum(1 for a in self._alerts.values() if not a.resolved),
            "health_summary": self._get_health_status(None),
        }

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "total_metrics_recorded": self._total_metrics_recorded,
            "agents_monitored": len(self._metrics),
            "total_thresholds": len(self._thresholds),
            "total_alerts": len(self._alerts),
            "resolved_alerts": sum(1 for a in self._alerts.values() if a.resolved),
        }
