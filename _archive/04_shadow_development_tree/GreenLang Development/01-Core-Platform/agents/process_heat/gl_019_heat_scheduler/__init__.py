"""
GL-019 HEATSCHEDULER - Industrial Process Heat Scheduling Agent

Comprehensive heat load scheduling optimization for industrial
process heat applications with ML-based forecasting, thermal
storage optimization, and demand charge management.

Key Components:
    - HeatSchedulerAgent: Main agent class
    - EnsembleForecaster: ML-based load forecasting
    - ThermalStorageOptimizer: Storage dispatch optimization
    - DemandChargeOptimizer: Peak demand reduction
    - ProductionPlanner: Production schedule integration
    - WeatherService: Weather forecast integration

Features:
    - ML-based 24-48 hour load forecasting using ensemble methods
    - Thermal storage optimization (hot water tanks, PCM)
    - Demand charge optimization (peak reduction, load shifting)
    - Real-time pricing response
    - Production schedule integration
    - Weather-based load adjustment
    - Zero-hallucination: All optimizations are deterministic
    - Complete provenance tracking with SHA-256 hashing

Example:
    >>> from greenlang.agents.process_heat.gl_019_heat_scheduler import (
    ...     HeatSchedulerAgent,
    ...     HeatSchedulerConfig,
    ...     HeatSchedulerInput,
    ... )
    >>>
    >>> # Configure agent
    >>> config = HeatSchedulerConfig(
    ...     agent_name="Plant-A Scheduler",
    ...     tariffs=[TariffConfiguration(
    ...         tariff_id="TOU-001",
    ...         tariff_type="time_of_use",
    ...         peak_rate_per_kwh=0.15,
    ...         off_peak_rate_per_kwh=0.06,
    ...     )],
    ...     thermal_storage=[ThermalStorageConfiguration(
    ...         storage_id="TES-001",
    ...         storage_type="hot_water_tank",
    ...         capacity_kwh=5000.0,
    ...         max_charge_rate_kw=500.0,
    ...         max_discharge_rate_kw=500.0,
    ...     )],
    ... )
    >>>
    >>> # Create agent
    >>> agent = HeatSchedulerAgent(config)
    >>>
    >>> # Process scheduling request
    >>> input_data = HeatSchedulerInput(
    ...     facility_id="PLANT-A",
    ...     optimization_horizon_hours=24,
    ...     current_load_kw=1500.0,
    ... )
    >>> result = agent.process(input_data)
    >>>
    >>> # Access results
    >>> print(f"Peak demand: {result.peak_demand_kw} kW")
    >>> print(f"Total savings: ${result.total_savings_usd:.2f}")
    >>> print(f"Schedule actions: {len(result.schedule_actions)}")

Score Target: 95+/100

Author: GreenLang Team
Date: December 2025
Version: 1.0.0
"""

# Configuration imports
from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
    # Enums
    TariffType,
    EquipmentType,
    EquipmentStatus,
    OptimizationObjective,
    SchedulePriority,
    StorageType,
    ForecastModel,
    # Configurations
    TariffConfiguration,
    EquipmentConfiguration,
    ThermalStorageConfiguration,
    LoadForecastingConfiguration,
    DemandChargeConfiguration,
    WeatherConfiguration,
    OptimizationParameters,
    ERPIntegration,
    ControlSystemIntegration,
    HeatSchedulerConfig,
)

# Schema imports
from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
    # Enums
    ScheduleStatus,
    LoadForecastStatus,
    StorageMode,
    DemandAlertLevel,
    ScheduleAction,
    ProductionStatus,
    # Load Forecasting
    LoadForecastPoint,
    LoadForecastResult,
    # Thermal Storage
    StorageStatePoint,
    StorageDispatchSchedule,
    ThermalStorageResult,
    # Demand Charge
    DemandPeriod,
    DemandChargeResult,
    # Production
    ProductionOrder,
    ProductionScheduleResult,
    # Weather
    WeatherForecastPoint,
    WeatherForecastResult,
    # Schedule Actions
    ScheduleActionItem,
    # Main I/O
    HeatSchedulerInput,
    HeatSchedulerOutput,
)

# Main agent
from greenlang.agents.process_heat.gl_019_heat_scheduler.scheduler import (
    HeatSchedulerAgent,
)

# Load forecasting
from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
    HistoricalDataPoint,
    ForecastFeatures,
    ModelPerformance,
    FeatureEngineer,
    BaseForecastModel,
    GradientBoostingModel,
    RandomForestModel,
    ARIMAModel,
    EnsembleForecaster,
)

# Thermal storage
from greenlang.agents.process_heat.gl_019_heat_scheduler.thermal_storage import (
    ThermalStorageUnit,
    ThermalStorageOptimizer,
    PCMStorageCalculator,
)

# Demand charge
from greenlang.agents.process_heat.gl_019_heat_scheduler.demand_charge import (
    DemandPeriodAnalyzer,
    LoadShifter,
    DemandResponseHandler,
    DemandChargeOptimizer,
)

# Production planning
from greenlang.agents.process_heat.gl_019_heat_scheduler.production_planning import (
    ProductionShift,
    ShiftScheduleManager,
    ProductionOrderScheduler,
    ERPConnector,
    ProductionPlanner,
)

# Weather integration
from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
    CurrentWeather,
    HistoricalWeather,
    WeatherProvider,
    OpenWeatherMapProvider,
    ManualWeatherProvider,
    DegreeDayCalculator,
    WeatherImpactCalculator,
    WeatherService,
)


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-019"
__agent_name__ = "HEATSCHEDULER"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__agent_name__",
    # Main agent
    "HeatSchedulerAgent",
    # Configuration - Enums
    "TariffType",
    "EquipmentType",
    "EquipmentStatus",
    "OptimizationObjective",
    "SchedulePriority",
    "StorageType",
    "ForecastModel",
    # Configuration - Classes
    "TariffConfiguration",
    "EquipmentConfiguration",
    "ThermalStorageConfiguration",
    "LoadForecastingConfiguration",
    "DemandChargeConfiguration",
    "WeatherConfiguration",
    "OptimizationParameters",
    "ERPIntegration",
    "ControlSystemIntegration",
    "HeatSchedulerConfig",
    # Schemas - Enums
    "ScheduleStatus",
    "LoadForecastStatus",
    "StorageMode",
    "DemandAlertLevel",
    "ScheduleAction",
    "ProductionStatus",
    # Schemas - Load Forecasting
    "LoadForecastPoint",
    "LoadForecastResult",
    # Schemas - Thermal Storage
    "StorageStatePoint",
    "StorageDispatchSchedule",
    "ThermalStorageResult",
    # Schemas - Demand Charge
    "DemandPeriod",
    "DemandChargeResult",
    # Schemas - Production
    "ProductionOrder",
    "ProductionScheduleResult",
    # Schemas - Weather
    "WeatherForecastPoint",
    "WeatherForecastResult",
    # Schemas - Schedule
    "ScheduleActionItem",
    # Schemas - I/O
    "HeatSchedulerInput",
    "HeatSchedulerOutput",
    # Load Forecasting
    "HistoricalDataPoint",
    "ForecastFeatures",
    "ModelPerformance",
    "FeatureEngineer",
    "BaseForecastModel",
    "GradientBoostingModel",
    "RandomForestModel",
    "ARIMAModel",
    "EnsembleForecaster",
    # Thermal Storage
    "ThermalStorageUnit",
    "ThermalStorageOptimizer",
    "PCMStorageCalculator",
    # Demand Charge
    "DemandPeriodAnalyzer",
    "LoadShifter",
    "DemandResponseHandler",
    "DemandChargeOptimizer",
    # Production Planning
    "ProductionShift",
    "ShiftScheduleManager",
    "ProductionOrderScheduler",
    "ERPConnector",
    "ProductionPlanner",
    # Weather Integration
    "CurrentWeather",
    "HistoricalWeather",
    "WeatherProvider",
    "OpenWeatherMapProvider",
    "ManualWeatherProvider",
    "DegreeDayCalculator",
    "WeatherImpactCalculator",
    "WeatherService",
]
