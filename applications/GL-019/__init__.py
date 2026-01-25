# -*- coding: utf-8 -*-
"""
GL-019 HEATSCHEDULER - Process Heating Scheduler Agent Package.

This package provides production-grade process heating schedule optimization,
energy cost minimization, and demand response integration for industrial
heating operations.

Main Components:
    - ProcessHeatingSchedulerAgent: Main orchestrator for heating schedule optimization
    - Data Models: Production batches, heating tasks, energy tariffs, equipment
    - Configuration: Agent, tariff, equipment, and optimization configuration models
    - Tools: FastAPI REST API for schedule management and optimization

Example Usage:
    >>> from greenlang.GL_019 import ProcessHeatingSchedulerAgent, AgentConfiguration
    >>> from greenlang.GL_019.config import (
    ...     TariffConfiguration,
    ...     EquipmentConfiguration,
    ...     ProductionScheduleConfiguration,
    ...     OptimizationParameters,
    ... )
    >>>
    >>> # Create tariff configuration
    >>> tariff_config = TariffConfiguration(
    ...     tariff_id="TOU-001",
    ...     tariff_type="time_of_use",
    ...     peak_rate_per_kwh=0.15,
    ...     off_peak_rate_per_kwh=0.06,
    ...     peak_hours_start=14,
    ...     peak_hours_end=20,
    ... )
    >>>
    >>> # Create equipment configuration
    >>> equipment_config = EquipmentConfiguration(
    ...     equipment_id="FURN-001",
    ...     equipment_type="electric_furnace",
    ...     capacity_kw=500.0,
    ...     efficiency=0.92,
    ...     max_temperature_c=1200.0,
    ... )
    >>>
    >>> # Create agent configuration
    >>> agent_config = AgentConfiguration(
    ...     tariffs=[tariff_config],
    ...     equipment=[equipment_config],
    ...     optimization_parameters=OptimizationParameters(),
    ... )
    >>>
    >>> # Initialize agent
    >>> agent = ProcessHeatingSchedulerAgent(agent_config)
    >>>
    >>> # Execute optimization
    >>> import asyncio
    >>> result = asyncio.run(agent.execute())
    >>>
    >>> # Check results
    >>> print(f"Total Cost: ${result.optimized_schedule.total_cost:.2f}")
    >>> print(f"Savings vs Baseline: ${result.optimized_schedule.savings_vs_baseline:.2f}")

Author: GreenLang Team
Date: December 2025
Version: 1.0.0
Status: Production Ready
"""

from greenlang.GL_019.process_heating_scheduler_agent import (
    ProcessHeatingSchedulerAgent,
    ProductionBatch,
    HeatingTask,
    EnergyTariff,
    Equipment,
    OptimizedSchedule,
    ScheduleOptimizationResult,
    DemandResponseEvent,
    CostForecast,
)

from greenlang.GL_019.config import (
    TariffType,
    EquipmentType,
    EquipmentStatus,
    OptimizationObjective,
    SchedulePriority,
    TariffConfiguration,
    EquipmentConfiguration,
    ProductionScheduleConfiguration,
    OptimizationParameters,
    ERPIntegration,
    ControlSystemIntegration,
    AgentConfiguration,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__status__ = "Production Ready"

__all__ = [
    # Main Agent
    "ProcessHeatingSchedulerAgent",
    # Data Models
    "ProductionBatch",
    "HeatingTask",
    "EnergyTariff",
    "Equipment",
    "OptimizedSchedule",
    "ScheduleOptimizationResult",
    "DemandResponseEvent",
    "CostForecast",
    # Configuration Enums
    "TariffType",
    "EquipmentType",
    "EquipmentStatus",
    "OptimizationObjective",
    "SchedulePriority",
    # Configuration Models
    "TariffConfiguration",
    "EquipmentConfiguration",
    "ProductionScheduleConfiguration",
    "OptimizationParameters",
    "ERPIntegration",
    "ControlSystemIntegration",
    "AgentConfiguration",
]