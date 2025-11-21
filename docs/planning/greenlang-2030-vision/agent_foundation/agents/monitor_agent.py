# -*- coding: utf-8 -*-
"""
MonitorAgent - System monitoring and health check agent.

This module implements the MonitorAgent for tracking system health, performance
metrics, agent status, and generating alerts for anomalies.

Example:
    >>> agent = MonitorAgent(config)
    >>> result = await agent.execute(MonitorInput(
    ...     monitor_type="system_health",
    ...     targets=["agent_swarm", "database", "api"],
    ...     interval_seconds=60
    ... ))
"""

import asyncio
import hashlib
import logging
import statistics
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

import sys
import os
from greenlang.determinism import deterministic_random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, AgentConfig, ExecutionContext

logger = logging.getLogger(__name__)


class MonitorType(str, Enum):
    """Types of monitoring."""

    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    AGENT_STATUS = "agent_status"
    ERROR_TRACKING = "error_tracking"
    RESOURCE_USAGE = "resource_usage"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    BUSINESS_METRICS = "business_metrics"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MonitorInput(BaseModel):
    """Input data model for MonitorAgent."""

    monitor_type: MonitorType = Field(..., description="Type of monitoring")
    targets: List[str] = Field(..., description="Targets to monitor")
    interval_seconds: int = Field(60, ge=1, le=3600, description="Monitoring interval")
    duration_seconds: Optional[int] = Field(
        None, ge=1, le=86400,
        description="Total monitoring duration (None = continuous)"
    )
    metrics: List[str] = Field(
        default_factory=list,
        description="Specific metrics to collect"
    )
    thresholds: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Alert thresholds for metrics"
    )
    alert_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Alert configuration"
    )
    aggregation: str = Field("average", description="Metric aggregation method")
    history_size: int = Field(100, ge=10, le=1000, description="History buffer size")

    @validator('targets')
    def validate_targets(cls, v):
        """Validate at least one target specified."""
        if not v:
            raise ValueError("At least one monitoring target required")
        return v


class MonitorOutput(BaseModel):
    """Output data model for MonitorAgent."""

    success: bool = Field(..., description="Monitoring execution success")
    monitor_type: MonitorType = Field(..., description="Type of monitoring performed")
    health_status: HealthStatus = Field(..., description="Overall health status")
    targets_monitored: List[str] = Field(..., description="Targets that were monitored")
    metrics_collected: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Collected metrics by target"
    )
    alerts_triggered: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alerts that were triggered"
    )
    anomalies_detected: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected anomalies"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="System recommendations"
    )
    summary_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    monitoring_duration_ms: float = Field(..., description="Monitoring duration")
    samples_collected: int = Field(0, ge=0, description="Number of samples collected")


class MonitorAgent(BaseAgent):
    """
    MonitorAgent implementation for system health and performance monitoring.

    This agent continuously monitors system components, collects metrics,
    detects anomalies, and triggers alerts based on configured thresholds.

    Attributes:
        config: Agent configuration
        metric_collectors: Registry of metric collectors
        alert_manager: Alert management system
        anomaly_detector: Anomaly detection engine
        metric_history: Historical metric storage

    Example:
        >>> config = AgentConfig(name="system_monitor", version="1.0.0")
        >>> agent = MonitorAgent(config)
        >>> await agent.initialize()
        >>> result = await agent.execute(monitor_input)
        >>> print(f"Health status: {result.result.health_status}")
    """

    def __init__(self, config: AgentConfig):
        """Initialize MonitorAgent."""
        super().__init__(config)
        self.metric_collectors: Dict[MonitorType, MetricCollector] = {}
        self.alert_manager = AlertManager()
        self.anomaly_detector = AnomalyDetector()
        self.metric_history: Dict[str, Deque] = {}
        self.monitoring_active = False
        self.monitoring_history: List[MonitorOutput] = []

    async def _initialize_core(self) -> None:
        """Initialize monitoring resources."""
        self._logger.info("Initializing MonitorAgent resources")

        # Initialize metric collectors
        self._initialize_collectors()

        # Initialize alert manager
        await self.alert_manager.initialize()

        # Initialize anomaly detector
        self.anomaly_detector.initialize()

        self._logger.info(f"Monitor initialized with {len(self.metric_collectors)} collectors")

    def _initialize_collectors(self) -> None:
        """Initialize metric collectors for each monitor type."""
        self.metric_collectors = {
            MonitorType.SYSTEM_HEALTH: SystemHealthCollector(),
            MonitorType.PERFORMANCE: PerformanceCollector(),
            MonitorType.AGENT_STATUS: AgentStatusCollector(),
            MonitorType.ERROR_TRACKING: ErrorTrackingCollector(),
            MonitorType.RESOURCE_USAGE: ResourceUsageCollector(),
            MonitorType.COMPLIANCE: ComplianceCollector(),
            MonitorType.SECURITY: SecurityCollector(),
            MonitorType.BUSINESS_METRICS: BusinessMetricsCollector()
        }

    async def _execute_core(self, input_data: MonitorInput, context: ExecutionContext) -> MonitorOutput:
        """
        Core execution logic for monitoring.

        This method performs continuous or time-bound monitoring with alerting.
        """
        start_time = datetime.now(timezone.utc)
        metrics_collected = {}
        alerts_triggered = []
        anomalies_detected = []
        samples = 0

        try:
            # Step 1: Get appropriate collector
            collector = self.metric_collectors.get(input_data.monitor_type)
            if not collector:
                raise ValueError(f"No collector for monitor type: {input_data.monitor_type}")

            # Step 2: Initialize metric history buffers
            for target in input_data.targets:
                if target not in self.metric_history:
                    self.metric_history[target] = deque(maxlen=input_data.history_size)

            # Step 3: Start monitoring loop
            self.monitoring_active = True
            end_time = None
            if input_data.duration_seconds:
                end_time = start_time + timedelta(seconds=input_data.duration_seconds)

            self._logger.info(f"Starting {input_data.monitor_type} monitoring for {input_data.targets}")

            while self.monitoring_active:
                loop_start = datetime.now(timezone.utc)

                # Check if duration exceeded
                if end_time and loop_start >= end_time:
                    break

                # Step 4: Collect metrics for each target
                for target in input_data.targets:
                    try:
                        # Collect metrics
                        target_metrics = await collector.collect(
                            target,
                            input_data.metrics
                        )

                        # Store metrics
                        if target not in metrics_collected:
                            metrics_collected[target] = []
                        metrics_collected[target].append(target_metrics)

                        # Add to history
                        self.metric_history[target].append(target_metrics)

                        # Step 5: Check thresholds
                        if input_data.thresholds:
                            threshold_alerts = self._check_thresholds(
                                target,
                                target_metrics,
                                input_data.thresholds
                            )
                            alerts_triggered.extend(threshold_alerts)

                        # Step 6: Detect anomalies
                        if len(self.metric_history[target]) >= 10:  # Need history for detection
                            anomalies = self.anomaly_detector.detect(
                                list(self.metric_history[target]),
                                target
                            )
                            anomalies_detected.extend(anomalies)

                        samples += 1

                    except Exception as e:
                        self._logger.error(f"Error collecting metrics for {target}: {e}")

                # Step 7: Process alerts
                if alerts_triggered:
                    await self.alert_manager.process_alerts(alerts_triggered)

                # Step 8: Sleep until next interval (or break if single run)
                if input_data.duration_seconds is None:
                    # Single run for deterministic testing
                    break

                elapsed = (datetime.now(timezone.utc) - loop_start).total_seconds()
                sleep_time = max(0, input_data.interval_seconds - elapsed)
                await asyncio.sleep(sleep_time)

            # Step 9: Calculate overall health status
            health_status = self._calculate_health_status(
                metrics_collected,
                alerts_triggered,
                anomalies_detected
            )

            # Step 10: Generate recommendations
            recommendations = self._generate_recommendations(
                health_status,
                metrics_collected,
                alerts_triggered
            )

            # Step 11: Calculate summary statistics
            summary_stats = self._calculate_summary_stats(
                metrics_collected,
                input_data.aggregation
            )

            # Step 12: Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data.dict(),
                metrics_collected,
                context.execution_id
            )

            # Step 13: Calculate monitoring duration
            monitoring_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Step 14: Create output
            output = MonitorOutput(
                success=True,
                monitor_type=input_data.monitor_type,
                health_status=health_status,
                targets_monitored=input_data.targets,
                metrics_collected=self._aggregate_metrics(metrics_collected, input_data.aggregation),
                alerts_triggered=alerts_triggered,
                anomalies_detected=anomalies_detected,
                recommendations=recommendations,
                summary_stats=summary_stats,
                provenance_hash=provenance_hash,
                monitoring_duration_ms=monitoring_duration,
                samples_collected=samples
            )

            # Store in history
            self.monitoring_history.append(output)
            if len(self.monitoring_history) > 50:
                self.monitoring_history.pop(0)

            return output

        except Exception as e:
            self._logger.error(f"Monitoring failed: {str(e)}", exc_info=True)
            raise

        finally:
            self.monitoring_active = False

    def _check_thresholds(self, target: str, metrics: Dict, thresholds: Dict) -> List[Dict]:
        """Check metrics against configured thresholds."""
        alerts = []

        for metric_name, metric_value in metrics.items():
            if metric_name in thresholds:
                threshold_config = thresholds[metric_name]

                # Check upper threshold
                if "max" in threshold_config and metric_value > threshold_config["max"]:
                    alerts.append({
                        "target": target,
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold_config["max"],
                        "type": "exceeded_max",
                        "severity": AlertSeverity.HIGH,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                # Check lower threshold
                if "min" in threshold_config and metric_value < threshold_config["min"]:
                    alerts.append({
                        "target": target,
                        "metric": metric_name,
                        "value": metric_value,
                        "threshold": threshold_config["min"],
                        "type": "below_min",
                        "severity": AlertSeverity.HIGH,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

        return alerts

    def _calculate_health_status(
        self,
        metrics: Dict,
        alerts: List,
        anomalies: List
    ) -> HealthStatus:
        """Calculate overall health status."""
        # Count critical issues
        critical_alerts = [a for a in alerts if a.get("severity") == AlertSeverity.CRITICAL]
        high_alerts = [a for a in alerts if a.get("severity") == AlertSeverity.HIGH]

        if critical_alerts:
            return HealthStatus.CRITICAL
        elif high_alerts or len(anomalies) > 5:
            return HealthStatus.UNHEALTHY
        elif alerts or anomalies:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _generate_recommendations(
        self,
        health_status: HealthStatus,
        metrics: Dict,
        alerts: List
    ) -> List[str]:
        """Generate system recommendations based on monitoring results."""
        recommendations = []

        if health_status == HealthStatus.CRITICAL:
            recommendations.append("URGENT: System in critical state, immediate intervention required")

        if health_status == HealthStatus.UNHEALTHY:
            recommendations.append("System health degraded, investigate alerts and anomalies")

        # Analyze specific patterns
        for alert in alerts:
            if alert.get("type") == "exceeded_max" and alert.get("metric") == "cpu_usage":
                recommendations.append("Consider scaling up resources due to high CPU usage")
            elif alert.get("type") == "exceeded_max" and alert.get("metric") == "memory_usage":
                recommendations.append("Memory usage high, check for memory leaks")
            elif alert.get("metric") == "error_rate" and alert.get("value", 0) > 0.05:
                recommendations.append("Error rate elevated, review error logs")

        # Check for resource patterns
        for target, target_metrics in metrics.items():
            if target_metrics and isinstance(target_metrics, list):
                # Analyze trends
                if len(target_metrics) > 5:
                    if self._is_increasing_trend([m.get("cpu_usage", 0) for m in target_metrics]):
                        recommendations.append(f"CPU usage trending up for {target}")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _is_increasing_trend(self, values: List[float]) -> bool:
        """Check if values show increasing trend."""
        if len(values) < 3:
            return False

        # Simple trend detection
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

        return second_half > first_half * 1.1  # 10% increase threshold

    def _calculate_summary_stats(self, metrics: Dict, aggregation: str) -> Dict:
        """Calculate summary statistics for collected metrics."""
        summary = {}

        for target, target_metrics in metrics.items():
            if target_metrics and isinstance(target_metrics, list):
                # Extract numeric values
                all_values = {}
                for metric_dict in target_metrics:
                    for key, value in metric_dict.items():
                        if isinstance(value, (int, float)):
                            if key not in all_values:
                                all_values[key] = []
                            all_values[key].append(value)

                # Calculate aggregations
                target_summary = {}
                for key, values in all_values.items():
                    if values:
                        if aggregation == "average":
                            target_summary[f"{key}_avg"] = sum(values) / len(values)
                        elif aggregation == "max":
                            target_summary[f"{key}_max"] = max(values)
                        elif aggregation == "min":
                            target_summary[f"{key}_min"] = min(values)
                        elif aggregation == "sum":
                            target_summary[f"{key}_sum"] = sum(values)

                        # Always include basic stats
                        if len(values) > 1:
                            target_summary[f"{key}_stddev"] = statistics.stdev(values)

                summary[target] = target_summary

        return summary

    def _aggregate_metrics(self, metrics: Dict, aggregation: str) -> Dict:
        """Aggregate collected metrics."""
        aggregated = {}

        for target, target_metrics in metrics.items():
            if target_metrics and isinstance(target_metrics, list):
                if aggregation == "latest":
                    aggregated[target] = target_metrics[-1]
                else:
                    # Aggregate all samples
                    agg_values = {}
                    for metric_dict in target_metrics:
                        for key, value in metric_dict.items():
                            if isinstance(value, (int, float)):
                                if key not in agg_values:
                                    agg_values[key] = []
                                agg_values[key].append(value)

                    # Apply aggregation
                    final_values = {}
                    for key, values in agg_values.items():
                        if values:
                            if aggregation == "average":
                                final_values[key] = sum(values) / len(values)
                            elif aggregation == "max":
                                final_values[key] = max(values)
                            elif aggregation == "min":
                                final_values[key] = min(values)
                            elif aggregation == "sum":
                                final_values[key] = sum(values)

                    aggregated[target] = final_values

        return aggregated

    def _calculate_provenance_hash(self, inputs: Dict, metrics: Dict, execution_id: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "agent": self.config.name,
            "version": self.config.version,
            "execution_id": execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monitor_type": inputs.get("monitor_type"),
            "targets": inputs.get("targets"),
            "samples": len(metrics)
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def _terminate_core(self) -> None:
        """Cleanup monitoring resources."""
        self._logger.info("Cleaning up MonitorAgent resources")
        self.monitoring_active = False
        self.metric_collectors.clear()
        self.metric_history.clear()
        self.monitoring_history.clear()

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect monitor-specific metrics."""
        if not self.monitoring_history:
            return {}

        recent = self.monitoring_history[-50:]
        return {
            "total_monitoring_sessions": len(self.monitoring_history),
            "monitor_types": list(set(m.monitor_type for m in recent)),
            "total_alerts": sum(len(m.alerts_triggered) for m in recent),
            "total_anomalies": sum(len(m.anomalies_detected) for m in recent),
            "health_distribution": {
                status: sum(1 for m in recent if m.health_status == status)
                for status in HealthStatus
            },
            "average_samples": sum(m.samples_collected for m in recent) / len(recent)
        }


class MetricCollector:
    """Base class for metric collectors."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect metrics for target."""
        raise NotImplementedError


class SystemHealthCollector(MetricCollector):
    """Collect system health metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect system health metrics."""
        # Simulated metrics collection
        import random

        collected = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "status": "healthy" if deterministic_random().random() > 0.1 else "degraded",
            "uptime_seconds": deterministic_random().randint(86400, 864000),
            "response_time_ms": random.uniform(10, 100),
            "error_rate": random.uniform(0, 0.05),
            "availability": random.uniform(0.95, 1.0)
        }

        # Filter to requested metrics if specified
        if metrics:
            collected = {k: v for k, v in collected.items() if k in metrics or k in ["timestamp", "target"]}

        return collected


class PerformanceCollector(MetricCollector):
    """Collect performance metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect performance metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "cpu_usage": random.uniform(10, 80),
            "memory_usage": random.uniform(30, 70),
            "disk_io": random.uniform(0, 100),
            "network_io": random.uniform(0, 1000),
            "requests_per_second": random.uniform(100, 1000),
            "latency_p50": random.uniform(10, 50),
            "latency_p95": random.uniform(50, 200),
            "latency_p99": random.uniform(100, 500)
        }


class AgentStatusCollector(MetricCollector):
    """Collect agent status metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect agent status metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "active_agents": deterministic_random().randint(10, 100),
            "idle_agents": deterministic_random().randint(0, 20),
            "failed_agents": deterministic_random().randint(0, 5),
            "tasks_queued": deterministic_random().randint(0, 50),
            "tasks_processing": deterministic_random().randint(0, 30),
            "tasks_completed": deterministic_random().randint(100, 1000),
            "average_task_time": random.uniform(100, 5000)
        }


class ErrorTrackingCollector(MetricCollector):
    """Collect error tracking metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect error metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "error_count": deterministic_random().randint(0, 10),
            "warning_count": deterministic_random().randint(0, 50),
            "critical_errors": deterministic_random().randint(0, 2),
            "error_rate": random.uniform(0, 0.05),
            "mttr_minutes": random.uniform(5, 60)  # Mean time to recovery
        }


class ResourceUsageCollector(MetricCollector):
    """Collect resource usage metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect resource usage metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "cpu_cores": deterministic_random().randint(1, 16),
            "memory_gb": random.uniform(1, 64),
            "disk_gb": random.uniform(10, 1000),
            "network_mbps": random.uniform(10, 10000),
            "cost_per_hour": random.uniform(0.1, 10)
        }


class ComplianceCollector(MetricCollector):
    """Collect compliance metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect compliance metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "compliance_score": random.uniform(85, 100),
            "violations": deterministic_random().randint(0, 5),
            "audits_passed": deterministic_random().randint(8, 10),
            "audits_total": 10,
            "last_audit": datetime.now(timezone.utc).isoformat()
        }


class SecurityCollector(MetricCollector):
    """Collect security metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect security metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "security_score": random.uniform(80, 100),
            "vulnerabilities_critical": deterministic_random().randint(0, 2),
            "vulnerabilities_high": deterministic_random().randint(0, 5),
            "vulnerabilities_medium": deterministic_random().randint(0, 10),
            "failed_auth_attempts": deterministic_random().randint(0, 10),
            "suspicious_activities": deterministic_random().randint(0, 3)
        }


class BusinessMetricsCollector(MetricCollector):
    """Collect business metrics."""

    async def collect(self, target: str, metrics: List[str]) -> Dict[str, Any]:
        """Collect business metrics."""
        import random

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "reports_generated": deterministic_random().randint(10, 100),
            "calculations_performed": deterministic_random().randint(100, 10000),
            "data_processed_gb": random.uniform(1, 100),
            "api_calls": deterministic_random().randint(1000, 100000),
            "user_sessions": deterministic_random().randint(10, 1000)
        }


class AlertManager:
    """Manage system alerts."""

    def __init__(self):
        """Initialize alert manager."""
        self.alert_history = []
        self.alert_rules = {}

    async def initialize(self) -> None:
        """Initialize alert manager."""
        # Load alert rules
        self.alert_rules = {
            "cpu_critical": {"threshold": 90, "severity": AlertSeverity.CRITICAL},
            "memory_critical": {"threshold": 90, "severity": AlertSeverity.CRITICAL},
            "error_rate_high": {"threshold": 0.1, "severity": AlertSeverity.HIGH}
        }

    async def process_alerts(self, alerts: List[Dict]) -> None:
        """Process and route alerts."""
        for alert in alerts:
            # Store in history
            self.alert_history.append(alert)

            # Log based on severity
            if alert.get("severity") == AlertSeverity.CRITICAL:
                logger.critical(f"CRITICAL ALERT: {alert}")
            elif alert.get("severity") == AlertSeverity.HIGH:
                logger.error(f"HIGH ALERT: {alert}")
            else:
                logger.warning(f"ALERT: {alert}")

            # In production, would send notifications


class AnomalyDetector:
    """Detect anomalies in metrics."""

    def __init__(self):
        """Initialize anomaly detector."""
        self.sensitivity = 2.0  # Standard deviations for anomaly

    def initialize(self) -> None:
        """Initialize detector."""
        pass

    def detect(self, history: List[Dict], target: str) -> List[Dict]:
        """Detect anomalies in metric history."""
        anomalies = []

        # Extract numeric values
        numeric_metrics = {}
        for entry in history:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in ["timestamp"]:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)

        # Check each metric for anomalies
        for metric_name, values in numeric_metrics.items():
            if len(values) < 5:
                continue

            mean = statistics.mean(values)
            stdev = statistics.stdev(values)

            # Check latest value for anomaly
            latest = values[-1]
            z_score = abs((latest - mean) / stdev) if stdev > 0 else 0

            if z_score > self.sensitivity:
                anomalies.append({
                    "target": target,
                    "metric": metric_name,
                    "value": latest,
                    "expected_range": (mean - self.sensitivity * stdev, mean + self.sensitivity * stdev),
                    "z_score": z_score,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        return anomalies