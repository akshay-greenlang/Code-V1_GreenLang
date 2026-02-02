# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Steam Trap Monitoring Agent

This module provides comprehensive steam trap monitoring, diagnostics, and
optimization for industrial steam systems. It implements DOE Best Practices,
Spirax Sarco guidelines, and ASME B16.34 compliance for steam trap management.

Features:
    - Steam trap type classification and selection guidance
    - Condensate load calculations (startup vs operating, safety factors)
    - Failure mode diagnostics (failed open, failed closed, leaking)
    - TSP (Trap Survey Program) route optimization
    - Wireless sensor network integration (ultrasonic, temperature)
    - Steam loss calculations and economic analysis
    - ASME B16.34 compliance for trap ratings

Standards Compliance:
    - DOE Steam System Best Practices
    - Spirax Sarco Steam Trap Selection and Sizing
    - ASME B16.34 Valves - Flanged, Threaded, and Welding End
    - ISO 6552 Automatic Steam Traps

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor import (
    ...     SteamTrapMonitorAgent,
    ...     SteamTrapMonitorConfig,
    ...     TrapDiagnosticInput,
    ... )
    >>>
    >>> config = SteamTrapMonitorConfig(
    ...     plant_id="PLANT-001",
    ...     steam_pressure_psig=150.0,
    ...     steam_cost_per_mlb=12.50,
    ... )
    >>> agent = SteamTrapMonitorAgent(config)
    >>> result = agent.process(diagnostic_input)
    >>> print(f"Trap status: {result.trap_status}")

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

# Configuration
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    TrapTypeConfig,
    SensorConfig,
    EconomicsConfig,
    SurveyConfig,
    WirelessSensorConfig,
    DiagnosticThresholds,
    TrapType,
    TrapApplication,
    FailureMode,
    DiagnosticMethod,
    SensorType,
    AlertSeverity,
)

# Schemas
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    # Input models
    TrapDiagnosticInput,
    SensorReading,
    UltrasonicReading,
    TemperatureReading,
    TrapSurveyInput,
    CondensateLoadInput,
    # Output models
    TrapDiagnosticOutput,
    CondensateLoadOutput,
    EconomicAnalysisOutput,
    SurveyRouteOutput,
    TrapStatusSummary,
    # Supporting models
    TrapCondition,
    TrapHealthScore,
    FailureModeProbability,
    MaintenanceRecommendation,
    SteamLossEstimate,
)

# Trap Types
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.trap_types import (
    TrapTypeClassifier,
    TrapSelectionCriteria,
    TrapApplicationGuide,
    TrapCharacteristics,
)

# Condensate Load
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.condensate_load import (
    CondensateLoadCalculator,
    StartupLoadCalculator,
    OperatingLoadCalculator,
    SafetyFactorCalculator,
)

# Failure Diagnostics
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.failure_diagnostics import (
    TrapDiagnosticsEngine,
    FailureModeDetector,
    DiagnosticDecisionTree,
    UltrasonicAnalyzer,
    TemperatureDifferentialAnalyzer,
)

# Survey Management
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.survey_management import (
    TrapSurveyManager,
    TSPRouteOptimizer,
    TrapPopulationManager,
    SurveyScheduler,
)

# Wireless Sensors
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.wireless_sensors import (
    WirelessSensorNetwork,
    SensorDataCollector,
    SensorHealthMonitor,
    DataAggregator,
)

# Economics
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.economics import (
    SteamLossCalculator,
    EconomicAnalyzer,
    ROICalculator,
    CostBenefitAnalyzer,
)

# Main Agent
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.monitor import (
    SteamTrapMonitorAgent,
    SteamTrapAgentConfig,
    create_steam_trap_monitor,
)

__all__ = [
    # Main Agent
    "SteamTrapMonitorAgent",
    "SteamTrapAgentConfig",
    "create_steam_trap_monitor",
    # Configuration
    "SteamTrapMonitorConfig",
    "TrapTypeConfig",
    "SensorConfig",
    "EconomicsConfig",
    "SurveyConfig",
    "WirelessSensorConfig",
    "DiagnosticThresholds",
    "TrapType",
    "TrapApplication",
    "FailureMode",
    "DiagnosticMethod",
    "SensorType",
    "AlertSeverity",
    # Schemas - Input
    "TrapDiagnosticInput",
    "SensorReading",
    "UltrasonicReading",
    "TemperatureReading",
    "TrapSurveyInput",
    "CondensateLoadInput",
    # Schemas - Output
    "TrapDiagnosticOutput",
    "CondensateLoadOutput",
    "EconomicAnalysisOutput",
    "SurveyRouteOutput",
    "TrapStatusSummary",
    # Schemas - Supporting
    "TrapCondition",
    "TrapHealthScore",
    "FailureModeProbability",
    "MaintenanceRecommendation",
    "SteamLossEstimate",
    # Trap Types
    "TrapTypeClassifier",
    "TrapSelectionCriteria",
    "TrapApplicationGuide",
    "TrapCharacteristics",
    # Condensate Load
    "CondensateLoadCalculator",
    "StartupLoadCalculator",
    "OperatingLoadCalculator",
    "SafetyFactorCalculator",
    # Failure Diagnostics
    "TrapDiagnosticsEngine",
    "FailureModeDetector",
    "DiagnosticDecisionTree",
    "UltrasonicAnalyzer",
    "TemperatureDifferentialAnalyzer",
    # Survey Management
    "TrapSurveyManager",
    "TSPRouteOptimizer",
    "TrapPopulationManager",
    "SurveyScheduler",
    # Wireless Sensors
    "WirelessSensorNetwork",
    "SensorDataCollector",
    "SensorHealthMonitor",
    "DataAggregator",
    # Economics
    "SteamLossCalculator",
    "EconomicAnalyzer",
    "ROICalculator",
    "CostBenefitAnalyzer",
]

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"
__agent_id__ = "GL-008"
__agent_name__ = "TRAPCATCHER"
