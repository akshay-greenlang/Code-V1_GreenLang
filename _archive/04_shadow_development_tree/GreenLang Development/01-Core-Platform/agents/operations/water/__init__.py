# -*- coding: utf-8 -*-
"""
GreenLang Water Operations Agents
=================================

This package provides operations agents for water sector optimization,
monitoring, and efficiency improvement.

Agents:
    GL-OPS-WAT-001: WaterNetworkOptimizationAgent - Distribution network optimization
    GL-OPS-WAT-002: PumpSchedulingAgent - Pump scheduling optimization
    GL-OPS-WAT-003: LeakDetectionAgent - Water loss and leak detection
    GL-OPS-WAT-004: WaterQualityMonitorAgent - Water quality monitoring
    GL-OPS-WAT-005: DemandForecastingAgent - Water demand forecasting
    GL-OPS-WAT-006: EnergyRecoveryAgent - Energy recovery from water systems
    GL-OPS-WAT-007: ReservoirManagementAgent - Reservoir optimization
    GL-OPS-WAT-008: StormwaterManagementAgent - Stormwater system management

All agents follow the GreenLang standard patterns with:
    - Zero-hallucination calculations
    - Complete provenance tracking
    - Deterministic outputs
"""

from greenlang.agents.operations.water.network_optimization import (
    WaterNetworkOptimizationAgent,
    NetworkOptimizationInput,
    NetworkOptimizationOutput,
    PipeSegment,
    OptimizationResult,
)
from greenlang.agents.operations.water.pump_scheduling import (
    PumpSchedulingAgent,
    PumpSchedulingInput,
    PumpSchedulingOutput,
    PumpStation,
    PumpSchedule,
)
from greenlang.agents.operations.water.leak_detection import (
    LeakDetectionAgent,
    LeakDetectionInput,
    LeakDetectionOutput,
    LeakCandidate,
    DMASummary,
)
from greenlang.agents.operations.water.water_quality import (
    WaterQualityMonitorAgent,
    WaterQualityInput,
    WaterQualityOutput,
    WaterQualitySample,
    QualityAlert,
)
from greenlang.agents.operations.water.demand_forecasting import (
    DemandForecastingAgent,
    DemandForecastInput,
    DemandForecastOutput,
    DemandForecast,
    ForecastAccuracy,
)
from greenlang.agents.operations.water.energy_recovery import (
    EnergyRecoveryAgent,
    EnergyRecoveryInput,
    EnergyRecoveryOutput,
    RecoveryOpportunity,
    RecoveryResult,
)
from greenlang.agents.operations.water.reservoir_management import (
    ReservoirManagementAgent,
    ReservoirManagementInput,
    ReservoirManagementOutput,
    ReservoirState,
    ReleaseSchedule,
)
from greenlang.agents.operations.water.stormwater_management import (
    StormwaterManagementAgent,
    StormwaterInput,
    StormwaterOutput,
    StormEvent,
    InfrastructureStatus,
)

__all__ = [
    # Network Optimization (GL-OPS-WAT-001)
    "WaterNetworkOptimizationAgent",
    "NetworkOptimizationInput",
    "NetworkOptimizationOutput",
    "PipeSegment",
    "OptimizationResult",
    # Pump Scheduling (GL-OPS-WAT-002)
    "PumpSchedulingAgent",
    "PumpSchedulingInput",
    "PumpSchedulingOutput",
    "PumpStation",
    "PumpSchedule",
    # Leak Detection (GL-OPS-WAT-003)
    "LeakDetectionAgent",
    "LeakDetectionInput",
    "LeakDetectionOutput",
    "LeakCandidate",
    "DMASummary",
    # Water Quality (GL-OPS-WAT-004)
    "WaterQualityMonitorAgent",
    "WaterQualityInput",
    "WaterQualityOutput",
    "WaterQualitySample",
    "QualityAlert",
    # Demand Forecasting (GL-OPS-WAT-005)
    "DemandForecastingAgent",
    "DemandForecastInput",
    "DemandForecastOutput",
    "DemandForecast",
    "ForecastAccuracy",
    # Energy Recovery (GL-OPS-WAT-006)
    "EnergyRecoveryAgent",
    "EnergyRecoveryInput",
    "EnergyRecoveryOutput",
    "RecoveryOpportunity",
    "RecoveryResult",
    # Reservoir Management (GL-OPS-WAT-007)
    "ReservoirManagementAgent",
    "ReservoirManagementInput",
    "ReservoirManagementOutput",
    "ReservoirState",
    "ReleaseSchedule",
    # Stormwater Management (GL-OPS-WAT-008)
    "StormwaterManagementAgent",
    "StormwaterInput",
    "StormwaterOutput",
    "StormEvent",
    "InfrastructureStatus",
]
