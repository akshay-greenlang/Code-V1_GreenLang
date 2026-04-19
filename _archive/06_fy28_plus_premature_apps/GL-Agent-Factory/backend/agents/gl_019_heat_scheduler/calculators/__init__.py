"""
Calculators for GL-019 HEATSCHEDULER Agent

This module exports all calculator functions for process heating schedule
optimization including demand forecasting, MILP optimization, and tariff analysis.

All calculations follow zero-hallucination principles:
- ML is used ONLY for predictions (forecasting), NOT regulatory calculations
- Optimization uses deterministic mathematical programming (MILP)
- Tariff calculations use exact rate lookups
- All results include uncertainty quantification and provenance

Calculator Modules:
    - forecasting: ML-based demand forecasting with SHAP explainability
    - optimization: MILP schedule optimization with thermal storage
    - tariff: TOU tariff analysis and cost calculations

Example:
    >>> from calculators import DemandForecaster, ScheduleOptimizer, TariffAnalyzer
    >>> forecaster = DemandForecaster()
    >>> optimizer = ScheduleOptimizer()
    >>> analyzer = TariffAnalyzer(periods)
"""

# Forecasting module
from .forecasting import (
    # Feature extraction
    extract_time_features,
    extract_weather_features,
    extract_lag_features,
    # Uncertainty quantification
    calculate_prediction_intervals,
    quantile_regression_intervals,
    # Explainability
    calculate_feature_contributions,
    generate_explanation_text,
    FeatureContribution,
    # Main forecaster class
    DemandForecaster,
    # Ensemble forecasting
    ensemble_forecast,
)

# Optimization module
from .optimization import (
    # Data structures
    OptimizationStatus,
    TimeSlotData,
    EquipmentData,
    StorageData,
    ScheduleSlot,
    OptimizationResult,
    # Helper functions
    calculate_baseline_cost,
    calculate_equipment_efficiency_cost,
    calculate_storage_arbitrage_value,
    # Main optimizer class
    ScheduleOptimizer,
    # Storage optimization
    optimize_storage_dispatch,
    # Robustness analysis
    calculate_schedule_robustness,
    calculate_schedule_feasibility,
)

# Tariff module
from .tariff import (
    # Enums and data structures
    RatePeriod,
    TariffPeriodDef,
    DemandChargeStructure,
    CostBreakdown,
    # Rate lookup functions
    get_period_type,
    create_standard_tou_periods,
    # Cost calculation functions
    calculate_energy_cost,
    calculate_demand_charge,
    calculate_carbon_cost,
    # Main analyzer class
    TariffAnalyzer,
    # Real-time pricing support
    interpolate_rtp_rates,
    apply_critical_peak_event,
)

__all__ = [
    # =========================================================================
    # Forecasting
    # =========================================================================
    # Feature extraction
    "extract_time_features",
    "extract_weather_features",
    "extract_lag_features",
    # Uncertainty
    "calculate_prediction_intervals",
    "quantile_regression_intervals",
    # Explainability
    "calculate_feature_contributions",
    "generate_explanation_text",
    "FeatureContribution",
    # Main class
    "DemandForecaster",
    "ensemble_forecast",

    # =========================================================================
    # Optimization
    # =========================================================================
    # Data structures
    "OptimizationStatus",
    "TimeSlotData",
    "EquipmentData",
    "StorageData",
    "ScheduleSlot",
    "OptimizationResult",
    # Functions
    "calculate_baseline_cost",
    "calculate_equipment_efficiency_cost",
    "calculate_storage_arbitrage_value",
    # Main class
    "ScheduleOptimizer",
    "optimize_storage_dispatch",
    # Analysis
    "calculate_schedule_robustness",
    "calculate_schedule_feasibility",

    # =========================================================================
    # Tariff
    # =========================================================================
    # Data structures
    "RatePeriod",
    "TariffPeriodDef",
    "DemandChargeStructure",
    "CostBreakdown",
    # Functions
    "get_period_type",
    "create_standard_tou_periods",
    "calculate_energy_cost",
    "calculate_demand_charge",
    "calculate_carbon_cost",
    # Main class
    "TariffAnalyzer",
    # RTP support
    "interpolate_rtp_rates",
    "apply_critical_peak_event",
]
