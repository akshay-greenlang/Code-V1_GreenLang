# -*- coding: utf-8 -*-
"""
GL-020 ECONOPULSE - Economizer Performance Monitoring Agent.

This package implements a comprehensive economizer performance monitoring
system for industrial heat recovery applications. It provides real-time
monitoring of economizer fouling, efficiency degradation, and cleaning
optimization.

Key Features:
    - Real-time heat transfer performance monitoring
    - Fouling resistance (Rf) calculation and trending
    - Cleaning alert generation with predictive scheduling
    - Efficiency loss quantification in MMBtu and cost
    - Soot blower effectiveness tracking
    - Historical performance trend analysis
    - Zero-hallucination calculations using deterministic formulas
    - Data provenance tracking with SHA-256 hashing

Components:
    - EconomizerPerformanceAgent: Main orchestrator
    - AlertManager: Alert generation and routing
    - FoulingPredictor: ML-based fouling prediction
    - AnomalyDetector: Abnormal reading detection

Example:
    >>> from greenlang.GL_020 import EconomizerPerformanceAgent
    >>> from greenlang.GL_020.config import AgentConfiguration, EconomizerConfiguration
    >>>
    >>> # Create economizer configuration
    >>> econ_config = EconomizerConfiguration(
    ...     economizer_id="ECON-001",
    ...     economizer_type="finned_tube",
    ...     tube_count=200,
    ...     total_heat_transfer_area_sqft=2500.0,
    ...     design_water_flow_gpm=500.0,
    ...     design_heat_duty_mmbtu_hr=8.0,
    ... )
    >>>
    >>> # Create agent configuration
    >>> config = AgentConfiguration(
    ...     economizers=[econ_config],
    ...     fuel_cost_per_mmbtu=4.0,
    ... )
    >>>
    >>> # Initialize and run agent
    >>> agent = EconomizerPerformanceAgent(config)
    >>> result = await agent.execute()
    >>> print(f"Fouling Resistance: {result.performance_metrics.fouling_resistance}")

Author: GreenLang Team
Date: December 2025
Status: Production Ready
Version: 1.0.0
"""

from greenlang.GL_020.config import (
    # Enums
    EconomizerType,
    FoulingType,
    AlertSeverity,
    CleaningMethod,
    SootBlowerMediaType,
    TubeMaterial,
    SensorType,
    PerformanceStatus,
    # Configuration Models
    SensorConfiguration,
    EconomizerConfiguration,
    AlertThreshold,
    AlertConfiguration,
    SootBlowerZone,
    SootBlowerConfiguration,
    BaselineConfiguration,
    SCADAIntegration,
    AgentConfiguration,
)

from greenlang.GL_020.economizer_performance_agent import (
    # Data Models
    TemperatureReading,
    FlowReading,
    EconomizerState,
    PerformanceMetrics,
    FoulingAnalysis,
    CleaningAlert,
    PerformanceTrend,
    EfficiencyLossReport,
    EconomizerPerformanceResult,
    # Main Agent
    EconomizerPerformanceAgent,
)

from greenlang.GL_020.alerts import (
    AlertManager,
    Alert,
    AlertHistory,
)

from greenlang.GL_020.models import (
    FoulingPredictor,
    CleaningEffectivenessModel,
    AnomalyDetector,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-020"
__codename__ = "ECONOPULSE"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__codename__",
    # Enums
    "EconomizerType",
    "FoulingType",
    "AlertSeverity",
    "CleaningMethod",
    "SootBlowerMediaType",
    "TubeMaterial",
    "SensorType",
    "PerformanceStatus",
    # Configuration Models
    "SensorConfiguration",
    "EconomizerConfiguration",
    "AlertThreshold",
    "AlertConfiguration",
    "SootBlowerZone",
    "SootBlowerConfiguration",
    "BaselineConfiguration",
    "SCADAIntegration",
    "AgentConfiguration",
    # Data Models
    "TemperatureReading",
    "FlowReading",
    "EconomizerState",
    "PerformanceMetrics",
    "FoulingAnalysis",
    "CleaningAlert",
    "PerformanceTrend",
    "EfficiencyLossReport",
    "EconomizerPerformanceResult",
    # Main Agent
    "EconomizerPerformanceAgent",
    # Alert System
    "AlertManager",
    "Alert",
    "AlertHistory",
    # ML Models
    "FoulingPredictor",
    "CleaningEffectivenessModel",
    "AnomalyDetector",
]
