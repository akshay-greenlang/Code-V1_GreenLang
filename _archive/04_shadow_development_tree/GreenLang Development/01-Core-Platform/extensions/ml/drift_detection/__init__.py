"""
Drift Detection Module - Evidently AI Integration for GreenLang Process Heat Agents.

This module provides comprehensive drift detection capabilities using Evidently AI
for monitoring data drift, prediction drift, and concept drift in production
ML models powering the GreenLang Process Heat agent pipeline.

Components:
    - ProcessHeatDriftMonitor: Main monitoring class with Evidently integration
    - DriftProfile: Agent-specific drift profile configurations
    - DriftAlertManager: Alert management with severity levels and integrations

Example:
    >>> from greenlang.ml.drift_detection import ProcessHeatDriftMonitor
    >>> monitor = ProcessHeatDriftMonitor()
    >>> report = monitor.detect_data_drift(reference_data, current_data, agent_id="GL-001")
    >>> if report.drift_detected:
    ...     print(f"Drift severity: {report.severity}")
"""

from .evidently_monitor import (
    ProcessHeatDriftMonitor,
    EvidentlyDriftConfig,
    DriftAnalysisResult,
)
from .drift_profiles import (
    BaseDriftProfile,
    GL001CarbonEmissionsDriftProfile,
    GL003CSRDReportingDriftProfile,
    GL006Scope3DriftProfile,
    GL010EmissionsGuardianDriftProfile,
    get_drift_profile,
)
from .alert_manager import (
    DriftAlertManager,
    DriftAlert,
    AlertSeverity,
    AlertChannel,
)

__all__ = [
    # Main monitor
    "ProcessHeatDriftMonitor",
    "EvidentlyDriftConfig",
    "DriftAnalysisResult",
    # Drift profiles
    "BaseDriftProfile",
    "GL001CarbonEmissionsDriftProfile",
    "GL003CSRDReportingDriftProfile",
    "GL006Scope3DriftProfile",
    "GL010EmissionsGuardianDriftProfile",
    "get_drift_profile",
    # Alert management
    "DriftAlertManager",
    "DriftAlert",
    "AlertSeverity",
    "AlertChannel",
]
