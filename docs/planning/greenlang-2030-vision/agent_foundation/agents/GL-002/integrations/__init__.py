# -*- coding: utf-8 -*-
"""
GL-002 BoilerEfficiencyOptimizer Integration Modules

This package provides comprehensive integration connectors for boiler control systems,
fuel management, SCADA, emissions monitoring, and inter-agent coordination.

Modules:
- boiler_control_connector: DCS/PLC integration for boiler control
- fuel_management_connector: Fuel system integration and optimization
- scada_connector: SCADA system real-time data integration
- emissions_monitoring_connector: CEMS integration for emissions compliance
- data_transformers: Data normalization and quality management
- agent_coordinator: Inter-agent communication and coordination
"""

from .boiler_control_connector import (
    BoilerControlManager,
    BoilerControlConfig,
    BoilerParameter,
    ParameterType,
    BoilerProtocol,
    ModbusBoilerConnector,
    OPCUABoilerConnector,
    SafetyInterlock
)

from .fuel_management_connector import (
    FuelManagementConnector,
    FuelSupplyConfig,
    FuelType,
    FuelSpecification,
    FuelTank,
    FuelFlowMeter,
    FuelQualityAnalyzer,
    FuelCostOptimizer
)

from .scada_connector import (
    SCADAConnector,
    SCADAConnectionConfig,
    SCADAProtocol,
    SCADATag,
    SCADAAlarm,
    AlarmPriority,
    DataQuality,
    AlarmManager,
    SCADADataBuffer
)

from .emissions_monitoring_connector import (
    EmissionsMonitoringConnector,
    CEMSConfig,
    EmissionType,
    ComplianceStandard,
    EmissionReading,
    EmissionLimit,
    CEMSAnalyzer,
    ComplianceMonitor,
    EmissionsCalculator,
    PredictiveEmissionsModel
)

from .data_transformers import (
    DataTransformationPipeline,
    UnitConverter,
    DataValidator,
    OutlierDetector,
    DataImputer,
    TimeSeriesAligner,
    SensorFusion,
    DataPoint,
    SensorConfig,
    UnitSystem,
    DataQualityIssue
)

from .agent_coordinator import (
    AgentCoordinator,
    MessageBus,
    TaskScheduler,
    StateManager,
    CollaborativeOptimizer,
    AgentMessage,
    AgentTask,
    AgentProfile,
    AgentCapability,
    MessageType,
    MessagePriority,
    AgentRole,
    TaskStatus
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"

# Define public API
__all__ = [
    # Boiler Control
    "BoilerControlManager",
    "BoilerControlConfig",
    "BoilerParameter",
    "ParameterType",
    "BoilerProtocol",
    "ModbusBoilerConnector",
    "OPCUABoilerConnector",
    "SafetyInterlock",

    # Fuel Management
    "FuelManagementConnector",
    "FuelSupplyConfig",
    "FuelType",
    "FuelSpecification",
    "FuelTank",
    "FuelFlowMeter",
    "FuelQualityAnalyzer",
    "FuelCostOptimizer",

    # SCADA
    "SCADAConnector",
    "SCADAConnectionConfig",
    "SCADAProtocol",
    "SCADATag",
    "SCADAAlarm",
    "AlarmPriority",
    "DataQuality",
    "AlarmManager",
    "SCADADataBuffer",

    # Emissions Monitoring
    "EmissionsMonitoringConnector",
    "CEMSConfig",
    "EmissionType",
    "ComplianceStandard",
    "EmissionReading",
    "EmissionLimit",
    "CEMSAnalyzer",
    "ComplianceMonitor",
    "EmissionsCalculator",
    "PredictiveEmissionsModel",

    # Data Transformation
    "DataTransformationPipeline",
    "UnitConverter",
    "DataValidator",
    "OutlierDetector",
    "DataImputer",
    "TimeSeriesAligner",
    "SensorFusion",
    "DataPoint",
    "SensorConfig",
    "UnitSystem",
    "DataQualityIssue",

    # Agent Coordination
    "AgentCoordinator",
    "MessageBus",
    "TaskScheduler",
    "StateManager",
    "CollaborativeOptimizer",
    "AgentMessage",
    "AgentTask",
    "AgentProfile",
    "AgentCapability",
    "MessageType",
    "MessagePriority",
    "AgentRole",
    "TaskStatus"
]