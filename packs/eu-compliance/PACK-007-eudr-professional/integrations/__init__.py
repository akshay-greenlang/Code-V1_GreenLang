# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Integration Layer
==========================================================

12 integration bridges for the EUDR Professional Pack providing
pipeline orchestration, application bridging, traceability integration,
risk assessment routing, due diligence workflow connectivity, satellite
monitoring feeds, GIS analytics, EU Information System connectivity,
CSRD cross-regulation mapping, system health monitoring, and guided
setup configuration.

Components:
    - PackOrchestrator                 - Multi-phase pipeline orchestration
    - EUDRProfessionalAppBridge        - GL-EUDR-APP integration bridge
    - FullTraceabilityBridge           - 15 supply chain traceability agent proxies
    - RiskAssessmentBridge             - 5 risk assessment agent routing
    - DueDiligenceBridge               - Due diligence core agent integration
    - DueDiligenceWorkflowBridge       - Due diligence workflow agent routing
    - SatelliteMonitoringBridge        - Satellite imagery and deforestation feeds
    - GISAnalyticsBridge               - GIS/mapping connector integration
    - EnhancedEUISBridge               - EU Information System enhanced connectivity
    - CSRDCrossRegulationBridge        - CSRD/CSDDD cross-regulation mapping
    - HealthCheck                      - System health monitoring
    - SetupWizard                      - Guided pack configuration wizard

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-007 EUDR Professional Pack
Status: Production Ready
"""

__version__ = "1.0.0"
__pack_id__ = "PACK-007"
__pack_name__ = "EUDR Professional Pack"

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
from .pack_orchestrator import PackOrchestrator, OrchestratorConfig, PipelineResult

# ---------------------------------------------------------------------------
# Application Bridge
# ---------------------------------------------------------------------------
from .eudr_app_bridge import EUDRProfessionalAppBridge, EUDRAppBridgeConfig

# ---------------------------------------------------------------------------
# Traceability Bridge
# ---------------------------------------------------------------------------
from .full_traceability_bridge import FullTraceabilityBridge, TraceabilityBridgeConfig

# ---------------------------------------------------------------------------
# Risk Assessment Bridge
# ---------------------------------------------------------------------------
from .risk_assessment_bridge import RiskAssessmentBridge, RiskAssessmentBridgeConfig

# ---------------------------------------------------------------------------
# Due Diligence Bridge
# ---------------------------------------------------------------------------
from .due_diligence_bridge import DueDiligenceBridge, DDCoreBridgeConfig

# ---------------------------------------------------------------------------
# Due Diligence Workflow Bridge
# ---------------------------------------------------------------------------
from .due_diligence_workflow_bridge import DueDiligenceWorkflowBridge, DDWorkflowBridgeConfig

# ---------------------------------------------------------------------------
# Satellite Monitoring Bridge
# ---------------------------------------------------------------------------
from .satellite_monitoring_bridge import SatelliteMonitoringBridge, SatelliteMonitoringConfig

# ---------------------------------------------------------------------------
# GIS Analytics Bridge
# ---------------------------------------------------------------------------
from .gis_analytics_bridge import GISAnalyticsBridge, GISAnalyticsConfig

# ---------------------------------------------------------------------------
# EU Information System Bridge
# ---------------------------------------------------------------------------
from .eu_information_system_bridge import EnhancedEUISBridge, EnhancedEUISConfig

# ---------------------------------------------------------------------------
# CSRD Cross-Regulation Bridge
# ---------------------------------------------------------------------------
from .csrd_cross_regulation_bridge import CSRDCrossRegulationBridge, CrossRegulationConfig

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
from .health_check import HealthCheck, HealthCheckConfig, HealthCheckResult

# ---------------------------------------------------------------------------
# Setup Wizard
# ---------------------------------------------------------------------------
from .setup_wizard import SetupWizard, SetupConfig

__all__ = [
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # --- Pipeline Orchestrator ---
    "PackOrchestrator",
    "OrchestratorConfig",
    "PipelineResult",
    # --- Application Bridge ---
    "EUDRProfessionalAppBridge",
    "EUDRAppBridgeConfig",
    # --- Traceability Bridge ---
    "FullTraceabilityBridge",
    "TraceabilityBridgeConfig",
    # --- Risk Assessment Bridge ---
    "RiskAssessmentBridge",
    "RiskAssessmentBridgeConfig",
    # --- Due Diligence Bridge ---
    "DueDiligenceBridge",
    "DDCoreBridgeConfig",
    # --- Due Diligence Workflow Bridge ---
    "DueDiligenceWorkflowBridge",
    "DDWorkflowBridgeConfig",
    # --- Satellite Monitoring Bridge ---
    "SatelliteMonitoringBridge",
    "SatelliteMonitoringConfig",
    # --- GIS Analytics Bridge ---
    "GISAnalyticsBridge",
    "GISAnalyticsConfig",
    # --- EU Information System Bridge ---
    "EnhancedEUISBridge",
    "EnhancedEUISConfig",
    # --- CSRD Cross-Regulation Bridge ---
    "CSRDCrossRegulationBridge",
    "CrossRegulationConfig",
    # --- Health Check ---
    "HealthCheck",
    "HealthCheckConfig",
    "HealthCheckResult",
    # --- Setup Wizard ---
    "SetupWizard",
    "SetupConfig",
]
