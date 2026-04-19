"""
GL-007 FurnacePulse Core Module

This module provides the core functionality for the FurnacePulse
Furnace Performance Monitoring Agent. It implements real-time TMT
monitoring, hotspot detection, efficiency KPI tracking, RUL predictions,
and NFPA 86 compliance verification.

The FurnacePulse agent follows GreenLang's zero-hallucination principle
by using deterministic calculations for all performance metrics and
safety-critical evaluations.

Components:
    - config: Configuration classes and enums for furnace monitoring
    - schemas: Pydantic data models for telemetry and analytics
    - orchestrator: Main agent orchestration logic
    - calculators: Zero-hallucination calculation engines

Example:
    >>> from core import FurnacePulseConfig, FurnaceState
    >>> from core.orchestrator import FurnacePulseOrchestrator
    >>>
    >>> config = FurnacePulseConfig.from_yaml("config.yaml")
    >>> orchestrator = FurnacePulseOrchestrator(config)
    >>> result = await orchestrator.process(furnace_state)

Standards Compliance:
    - NFPA 86: Standard for Ovens and Furnaces
    - API 560: Fired Heaters for General Refinery Service
    - API 530: Calculation of Heater-Tube Thickness
    - ISO 13705: Petroleum and Petrochemical Industries Heaters

Author: GreenLang Team
Version: 1.0.0
"""

from core.config import (
    # Enums
    AlertTier,
    SignalQuality,
    FurnaceZone,
    OperatingMode,
    FlameQuality,
    TrendDirection,
    ComplianceStatus,
    HotspotSeverity,
    # Dataclasses
    FurnaceConstraints,
    NFPA86ComplianceConfig,
    FurnacePulseConfig,
)

from core.schemas import (
    # Base schemas
    TelemetrySignal,
    # State schemas
    FurnaceState,
    TMTReading,
    FlameStatus,
    DraftReading,
    FuelConsumption,
    # Alert schemas
    HotspotAlert,
    MaintenanceAlert,
    # KPI schemas
    EfficiencyKPI,
    EfficiencyTrend,
    # Prediction schemas
    RULPrediction,
    ConfidenceBounds,
    # Compliance schemas
    ComplianceEvidence,
    ChecklistItem,
    SafetyEvent,
    # Output schemas
    FurnacePulseOutput,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-007"
__agent_name__ = "FURNACEPULSE"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__agent_name__",
    # Enums
    "AlertTier",
    "SignalQuality",
    "FurnaceZone",
    "OperatingMode",
    "FlameQuality",
    "TrendDirection",
    "ComplianceStatus",
    "HotspotSeverity",
    # Config classes
    "FurnaceConstraints",
    "NFPA86ComplianceConfig",
    "FurnacePulseConfig",
    # Schemas
    "TelemetrySignal",
    "FurnaceState",
    "TMTReading",
    "FlameStatus",
    "DraftReading",
    "FuelConsumption",
    "HotspotAlert",
    "MaintenanceAlert",
    "EfficiencyKPI",
    "EfficiencyTrend",
    "RULPrediction",
    "ConfidenceBounds",
    "ComplianceEvidence",
    "ChecklistItem",
    "SafetyEvent",
    "FurnacePulseOutput",
]
