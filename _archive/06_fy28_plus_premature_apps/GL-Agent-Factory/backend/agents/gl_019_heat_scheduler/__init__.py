"""
GL-019 HEATSCHEDULER - Process Heating Scheduler Agent

This module provides the ProcessHeatingSchedulerAgent for ML-based demand
forecasting and heating schedule optimization with TOU tariff arbitrage.

The agent implements demand forecasting, MILP optimization, thermal storage
dispatch, and SSE streaming following zero-hallucination principles:
- ML for predictions only (not regulatory calculations)
- Deterministic MILP for optimization
- Exact tariff rate lookups for cost calculations

Agent ID: GL-019
Agent Name: HEATSCHEDULER (ProcessHeatingScheduler)
Version: 1.0.0
Priority: P1
Market Potential: $7B

Features:
- ML-based demand forecasting with uncertainty quantification
- SHAP/LIME explainability for forecasts
- Thermal storage optimization for TOU arbitrage
- Time-of-Use tariff arbitrage
- Production schedule integration
- SSE streaming schedule updates
- Zero-hallucination deterministic optimization core

Example:
    >>> from gl_019_heat_scheduler import ProcessHeatingSchedulerAgent, SchedulerInput
    >>> agent = ProcessHeatingSchedulerAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Cost savings: ${result.cost_savings:.2f}")
"""

from .agent import ProcessHeatingSchedulerAgent

from .schemas import (
    # Main input/output
    SchedulerInput,
    SchedulerOutput,
    SSEScheduleUpdate,
    AgentConfig,
    # Supporting models
    TimeSlot,
    ProductionOrder,
    TariffPeriod,
    EnergyTariff,
    Equipment,
    ThermalStorage,
    WeatherForecast,
    HistoricalDemand,
    GridSignal,
    UncertaintyBounds,
    DemandPrediction,
    ScheduledOperation,
    StorageDispatchPlan,
    ExplainabilityReport,
    # Enums
    TariffType,
    EquipmentStatus,
    StorageMode,
    SchedulePriority,
    ForecastConfidence,
    GridSignalType,
)

# Agent metadata
AGENT_ID = "GL-019"
AGENT_NAME = "HEATSCHEDULER"
VERSION = "1.0.0"
DESCRIPTION = "Process Heating Scheduler with ML-based demand forecasting and TOU optimization"
PRIORITY = "P1"
MARKET_POTENTIAL = "$7B"
CATEGORY = "Planning"
AGENT_TYPE = "Coordinator"

__all__ = [
    # Main agent class
    "ProcessHeatingSchedulerAgent",
    # Input/Output models
    "SchedulerInput",
    "SchedulerOutput",
    "SSEScheduleUpdate",
    "AgentConfig",
    # Supporting models
    "TimeSlot",
    "ProductionOrder",
    "TariffPeriod",
    "EnergyTariff",
    "Equipment",
    "ThermalStorage",
    "WeatherForecast",
    "HistoricalDemand",
    "GridSignal",
    "UncertaintyBounds",
    "DemandPrediction",
    "ScheduledOperation",
    "StorageDispatchPlan",
    "ExplainabilityReport",
    # Enums
    "TariffType",
    "EquipmentStatus",
    "StorageMode",
    "SchedulePriority",
    "ForecastConfidence",
    "GridSignalType",
    # Metadata
    "AGENT_ID",
    "AGENT_NAME",
    "VERSION",
    "DESCRIPTION",
    "PRIORITY",
    "MARKET_POTENTIAL",
    "CATEGORY",
    "AGENT_TYPE",
]
