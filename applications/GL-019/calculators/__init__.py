"""
GL-019 HEATSCHEDULER - Process Heating Schedule Optimization Calculators

Zero-hallucination, deterministic calculations for heating schedule optimization
to minimize energy costs while meeting production constraints.

Available Calculators:
- EnergyCostCalculator: Calculate energy costs under various tariff structures
- ScheduleOptimizer: MILP optimization for heating schedules
- LoadForecaster: Predict heating loads and energy costs
- SavingsCalculator: Analyze cost savings from schedule optimization

Key Features:
- Time-of-Use (ToU) tariff support
- Demand charge optimization
- Load shifting to off-peak hours
- Equipment capacity constraints
- Production deadline enforcement
- Thermal storage optimization
- IPMVP-compliant savings verification
- Complete provenance tracking (SHA-256)

Standards Reference:
- ISO 50001 - Energy Management Systems
- ISO 50006 - Measuring Energy Performance Using Baselines
- ASHRAE Guideline 14 - Measurement of Energy and Demand Savings
- IPMVP - International Performance Measurement and Verification Protocol

Guarantees:
- DETERMINISTIC: Same input always produces same output
- REPRODUCIBLE: SHA-256 verified calculation chain
- AUDITABLE: Complete step-by-step provenance trail
- ZERO HALLUCINATION: No LLM in calculation path

Example:
    >>> from calculators import EnergyCostCalculator, TariffStructure, TariffType
    >>> calculator = EnergyCostCalculator()
    >>> tariff = TariffStructure(
    ...     tariff_type=TariffType.TIME_OF_USE,
    ...     rates=[...],
    ...     peak_hours=[14, 15, 16, 17, 18, 19]
    ... )
    >>> result, provenance = calculator.calculate(inputs)
    >>> print(f"Total Cost: ${result.total_cost:.2f}")

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

# Provenance tracking
from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    compute_input_fingerprint,
    compute_output_fingerprint,
    verify_provenance,
    format_provenance_report,
    get_utc_timestamp
)

# Energy Cost Calculator
from .energy_cost_calculator import (
    EnergyCostCalculator,
    EnergyCostInput,
    EnergyCostOutput,
    TariffStructure,
    TariffRate,
    TariffType,
    TimePeriod,
    HourlyLoad,
    # Standalone functions
    calculate_simple_energy_cost,
    calculate_demand_charge,
    calculate_tou_cost,
    classify_hour,
    calculate_average_rate
)

# Schedule Optimizer
from .schedule_optimizer import (
    ScheduleOptimizer,
    ScheduleOptimizerInput,
    ScheduleOptimizerOutput,
    HeatingJob,
    EquipmentConstraint,
    ThermalStorage,
    TimeSlotCost,
    ScheduledOperation,
    OptimizationObjective,
    ConstraintType,
    # Standalone functions
    calculate_optimal_start_time,
    calculate_load_factor,
    calculate_peak_shaving_potential,
    create_shifted_profile,
    calculate_scheduling_flexibility
)

# Load Forecaster
from .load_forecaster import (
    LoadForecaster,
    LoadForecastInput,
    LoadForecastOutput,
    HistoricalLoad,
    ProductionSchedule,
    HourlyForecast,
    ForecastMethod,
    SeasonType,
    # Standalone functions
    calculate_heating_degree_days,
    simple_moving_average,
    exponential_smoothing,
    calculate_forecast_confidence_interval,
    calculate_mape,
    calculate_rmse,
    forecast_with_seasonality,
    get_season
)

# Savings Calculator
from .savings_calculator import (
    SavingsCalculator,
    SavingsCalculatorInput,
    SavingsCalculatorOutput,
    ScheduleComparison,
    HourlyScheduleData,
    ProjectCosts,
    SavingsByCategory,
    ROIMetrics,
    SavingsCategory,
    VerificationMethod,
    # Standalone functions
    calculate_simple_payback,
    calculate_npv,
    calculate_levelized_cost,
    calculate_savings_percentage,
    annualize_savings,
    calculate_demand_charge_savings,
    calculate_load_shift_savings,
    estimate_annual_savings_range
)

__all__ = [
    # Provenance tracking
    "ProvenanceTracker",
    "ProvenanceRecord",
    "CalculationStep",
    "compute_input_fingerprint",
    "compute_output_fingerprint",
    "verify_provenance",
    "format_provenance_report",
    "get_utc_timestamp",

    # Energy Cost Calculator
    "EnergyCostCalculator",
    "EnergyCostInput",
    "EnergyCostOutput",
    "TariffStructure",
    "TariffRate",
    "TariffType",
    "TimePeriod",
    "HourlyLoad",
    "calculate_simple_energy_cost",
    "calculate_demand_charge",
    "calculate_tou_cost",
    "classify_hour",
    "calculate_average_rate",

    # Schedule Optimizer
    "ScheduleOptimizer",
    "ScheduleOptimizerInput",
    "ScheduleOptimizerOutput",
    "HeatingJob",
    "EquipmentConstraint",
    "ThermalStorage",
    "TimeSlotCost",
    "ScheduledOperation",
    "OptimizationObjective",
    "ConstraintType",
    "calculate_optimal_start_time",
    "calculate_load_factor",
    "calculate_peak_shaving_potential",
    "create_shifted_profile",
    "calculate_scheduling_flexibility",

    # Load Forecaster
    "LoadForecaster",
    "LoadForecastInput",
    "LoadForecastOutput",
    "HistoricalLoad",
    "ProductionSchedule",
    "HourlyForecast",
    "ForecastMethod",
    "SeasonType",
    "calculate_heating_degree_days",
    "simple_moving_average",
    "exponential_smoothing",
    "calculate_forecast_confidence_interval",
    "calculate_mape",
    "calculate_rmse",
    "forecast_with_seasonality",
    "get_season",

    # Savings Calculator
    "SavingsCalculator",
    "SavingsCalculatorInput",
    "SavingsCalculatorOutput",
    "ScheduleComparison",
    "HourlyScheduleData",
    "ProjectCosts",
    "SavingsByCategory",
    "ROIMetrics",
    "SavingsCategory",
    "VerificationMethod",
    "calculate_simple_payback",
    "calculate_npv",
    "calculate_levelized_cost",
    "calculate_savings_percentage",
    "annualize_savings",
    "calculate_demand_charge_savings",
    "calculate_load_shift_savings",
    "estimate_annual_savings_range",
]

__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"
__agent_id__ = "GL-019"
__codename__ = "HEATSCHEDULER"
