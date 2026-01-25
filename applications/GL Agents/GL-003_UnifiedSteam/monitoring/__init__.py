"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Monitoring Module

This module provides comprehensive monitoring capabilities for the UNIFIEDSTEAM
steam system optimization agent, including alerting, health checks, metrics
collection, dashboard data providers, and SLA tracking.

Components:
    - alerting: Alert management with steam-specific alert types
    - health: Health monitoring with liveness/readiness probes
    - metrics: Prometheus-compatible metrics collection
    - dashboards: Dashboard data aggregation for UI consumption
    - sla_tracker: SLA monitoring and compliance reporting

The monitoring module ensures operational visibility and compliance with
service level agreements for steam system optimization operations.

Example:
    >>> from gl_003_unifiedsteam.monitoring import (
    ...     AlertManager,
    ...     HealthMonitor,
    ...     MetricsCollector,
    ...     DashboardProvider,
    ...     SLATracker,
    ... )
    >>> alert_manager = AlertManager()
    >>> health_monitor = HealthMonitor()
    >>> metrics = MetricsCollector(namespace="unifiedsteam")
"""

from .alerting import (
    # Enums
    AlertType,
    AlertSeverity,
    AlertState,
    EscalationLevel,
    # Data Classes
    Alert,
    AlertContext,
    AlertFilter,
    AcknowledgmentResult,
    EscalationResult,
    # Main Class
    AlertManager,
)

from .health import (
    # Enums
    HealthStatus,
    # Data Classes
    ServiceHealthStatus,
    IntegrationHealthStatus,
    ModelHealthStatus,
    DataFreshnessStatus,
    OverallHealthStatus,
    ComponentHealth,
    # Main Class
    HealthMonitor,
)

from .metrics import (
    # Data Classes
    MetricValue,
    SteamMetrics,
    OptimizationMetrics,
    TrapMetrics,
    DesuperheaterMetrics,
    CondensateMetrics,
    MetricsSummary,
    # Main Class
    MetricsCollector,
)

from .dashboards import (
    # Data Classes
    RealTimeDashboard,
    KPIDashboard,
    OptimizationDashboard,
    TrapHealthDashboard,
    ClimateImpactDashboard,
    DashboardWidget,
    TimeSeriesData,
    # Main Class
    DashboardProvider,
)

from .sla_tracker import (
    # Data Classes
    SLADefinition,
    SLAComplianceResult,
    SLAReport,
    SLAViolation,
    SLAMetricType,
    # Main Class
    SLATracker,
)

__all__ = [
    # Alerting
    "AlertType",
    "AlertSeverity",
    "AlertState",
    "EscalationLevel",
    "Alert",
    "AlertContext",
    "AlertFilter",
    "AcknowledgmentResult",
    "EscalationResult",
    "AlertManager",
    # Health
    "HealthStatus",
    "ServiceHealthStatus",
    "IntegrationHealthStatus",
    "ModelHealthStatus",
    "DataFreshnessStatus",
    "OverallHealthStatus",
    "ComponentHealth",
    "HealthMonitor",
    # Metrics
    "MetricValue",
    "SteamMetrics",
    "OptimizationMetrics",
    "TrapMetrics",
    "DesuperheaterMetrics",
    "CondensateMetrics",
    "MetricsSummary",
    "MetricsCollector",
    # Dashboards
    "RealTimeDashboard",
    "KPIDashboard",
    "OptimizationDashboard",
    "TrapHealthDashboard",
    "ClimateImpactDashboard",
    "DashboardWidget",
    "TimeSeriesData",
    "DashboardProvider",
    # SLA Tracker
    "SLADefinition",
    "SLAComplianceResult",
    "SLAReport",
    "SLAViolation",
    "SLAMetricType",
    "SLATracker",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-003"
__codename__ = "UNIFIEDSTEAM"
