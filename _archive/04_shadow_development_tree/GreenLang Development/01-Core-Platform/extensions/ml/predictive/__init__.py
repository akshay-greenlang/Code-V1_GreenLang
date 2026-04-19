# -*- coding: utf-8 -*-
"""
GreenLang Predictive Models Module

Zero-Hallucination Predictive Analytics for Process Heat Systems

This module provides production-ready predictive models for GreenLang agents,
supporting operational optimization and financial planning for industrial
process heat systems.

Modules:
    fuel_price: Time series forecasting for fuel prices (TASK-106)
    spare_parts: Inventory optimization for maintenance parts (TASK-109)
    production_impact: Production impact and cost modeling (TASK-110)

Integration Points:
    - GL-011 Fuel Module: Fuel cost projections
    - GL-013 PredictiveMaint: Maintenance planning and parts demand
    - Process Heat Agents (GL-003 to GL-018): Operational impact

Zero-Hallucination Guarantee:
    All predictions use deterministic statistical models with:
    - Complete provenance tracking (SHA-256 hashes)
    - No LLM involvement in numeric calculations
    - Reproducible results with seeded random generators
    - Auditable calculation trails

Example:
    >>> from greenlang.ml.predictive import FuelPricePredictor, SparePartsOptimizer
    >>> from greenlang.ml.predictive import ProductionImpactModeler
    >>>
    >>> # Fuel price forecasting
    >>> fuel_predictor = FuelPricePredictor()
    >>> forecast = fuel_predictor.predict(fuel_input)
    >>> print(f"30-day forecast: ${forecast.forecast_30d.predicted_price:.2f}")
    >>>
    >>> # Spare parts optimization
    >>> parts_optimizer = SparePartsOptimizer()
    >>> optimization = parts_optimizer.optimize(part_input)
    >>> print(f"EOQ: {optimization.eoq_result.eoq} units")
    >>>
    >>> # Production impact modeling
    >>> impact_modeler = ProductionImpactModeler()
    >>> impact = impact_modeler.analyze(impact_input)
    >>> print(f"Expected annual cost: ${impact.expected_annual_cost:,.2f}")

Author: GreenLang Engineering Team
License: MIT
"""

__version__ = "1.0.0"

# Fuel Price Prediction (TASK-106)
from greenlang.ml.predictive.fuel_price import (
    # Core classes
    FuelPricePredictor,
    # Enums
    FuelType,
    ForecastHorizon,
    ModelType,
    VolatilityRegime,
    MarketFactor,
    # Input/Output models
    FuelPriceDataPoint,
    FuelPriceInput,
    FuelPriceOutput,
    ForecastPoint,
    VolatilityMetrics,
    MarketFactorAnalysis,
    # Convenience functions
    forecast_natural_gas_price,
    forecast_fuel_oil_price,
    forecast_coal_price,
)

# Spare Parts Optimization (TASK-109)
from greenlang.ml.predictive.spare_parts import (
    # Core classes
    SparePartsOptimizer,
    # Enums
    PartCriticality,
    EquipmentType,
    StockingStrategy,
    DemandPattern,
    # Input/Output models
    PartUsageRecord,
    SparePartInput,
    SparePartOptimizationOutput,
    EOQResult,
    SafetyStockResult,
    DemandForecastResult,
    # Convenience functions
    calculate_eoq,
    calculate_safety_stock,
    recommend_stocking_strategy,
)

# Production Impact Modeling (TASK-110)
from greenlang.ml.predictive.production_impact import (
    # Core classes
    ProductionImpactModeler,
    # Enums
    FailureMode,
    ImpactSeverity,
    ProductionState,
    CostCategory,
    # Input/Output models
    EquipmentProfile,
    ProductionParameters,
    FailureScenario,
    ProductionImpactInput,
    ProductionImpactOutput,
    DowntimeCostResult,
    ScenarioResult,
    MonteCarloResult,
    # Convenience functions
    estimate_downtime_cost,
    calculate_availability,
    calculate_risk_priority_number,
)

__all__ = [
    # Version
    "__version__",

    # === Fuel Price Prediction (TASK-106) ===
    "FuelPricePredictor",
    # Enums
    "FuelType",
    "ForecastHorizon",
    "ModelType",
    "VolatilityRegime",
    "MarketFactor",
    # Models
    "FuelPriceDataPoint",
    "FuelPriceInput",
    "FuelPriceOutput",
    "ForecastPoint",
    "VolatilityMetrics",
    "MarketFactorAnalysis",
    # Functions
    "forecast_natural_gas_price",
    "forecast_fuel_oil_price",
    "forecast_coal_price",

    # === Spare Parts Optimization (TASK-109) ===
    "SparePartsOptimizer",
    # Enums
    "PartCriticality",
    "EquipmentType",
    "StockingStrategy",
    "DemandPattern",
    # Models
    "PartUsageRecord",
    "SparePartInput",
    "SparePartOptimizationOutput",
    "EOQResult",
    "SafetyStockResult",
    "DemandForecastResult",
    # Functions
    "calculate_eoq",
    "calculate_safety_stock",
    "recommend_stocking_strategy",

    # === Production Impact Modeling (TASK-110) ===
    "ProductionImpactModeler",
    # Enums
    "FailureMode",
    "ImpactSeverity",
    "ProductionState",
    "CostCategory",
    # Models
    "EquipmentProfile",
    "ProductionParameters",
    "FailureScenario",
    "ProductionImpactInput",
    "ProductionImpactOutput",
    "DowntimeCostResult",
    "ScenarioResult",
    "MonteCarloResult",
    # Functions
    "estimate_downtime_cost",
    "calculate_availability",
    "calculate_risk_priority_number",
]
