# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Optimization Module

Cleaning schedule optimization with deterministic cost models, multi-asset
scheduling, and what-if scenario analysis for heat exchanger networks.

This module provides:
- CleaningCostModel: 5-component cost model (energy, production, cleaning, downtime, risk)
- CleaningScheduleOptimizer: Deterministic grid-search optimizer for single exchangers
- MultiAssetScheduler: MILP-based fleet scheduling with shared resource constraints
- ScenarioAnalyzer: What-if and Monte Carlo sensitivity analysis

Zero-Hallucination Principle:
    All optimization computations use deterministic algorithms (grid search, MILP).
    Cost calculations use explicit formulas with traceable inputs.
    Uncertainty is quantified through Monte Carlo simulation, not LLM estimates.

Author: GreenLang AI Team
Version: 1.0.0
"""

from .cost_model import (
    CleaningCostModel,
    CostModelConfig,
    EnergyLossCost,
    ProductionLossCost,
    CleaningCost,
    DowntimeCost,
    RiskPenalty,
    TotalCostBreakdown,
    CostCurve,
    CostProjection,
)
from .cleaning_optimizer import (
    CleaningScheduleOptimizer,
    OptimizerConfig,
    CleaningWindow,
    CleaningMethod,
    OptimizationResult,
    CleaningRecommendation,
    WhatIfScenario,
    WhatIfResult,
    ScheduleRanking,
)
from .multi_asset_scheduler import (
    MultiAssetScheduler,
    FleetSchedulerConfig,
    AssetSchedule,
    FleetSchedule,
    ResourceConstraint,
    OutageWindow,
    MILPSolution,
    SchedulingObjective,
)
from .scenario_analyzer import (
    ScenarioAnalyzer,
    ScenarioConfig,
    ScenarioDefinition,
    ScenarioComparison,
    SensitivityResult,
    MonteCarloResult,
    TornadoChart,
    BreakevenAnalysis,
)

__all__ = [
    # Cost model
    "CleaningCostModel",
    "CostModelConfig",
    "EnergyLossCost",
    "ProductionLossCost",
    "CleaningCost",
    "DowntimeCost",
    "RiskPenalty",
    "TotalCostBreakdown",
    "CostCurve",
    "CostProjection",
    # Cleaning optimizer
    "CleaningScheduleOptimizer",
    "OptimizerConfig",
    "CleaningWindow",
    "CleaningMethod",
    "OptimizationResult",
    "CleaningRecommendation",
    "WhatIfScenario",
    "WhatIfResult",
    "ScheduleRanking",
    # Multi-asset scheduler
    "MultiAssetScheduler",
    "FleetSchedulerConfig",
    "AssetSchedule",
    "FleetSchedule",
    "ResourceConstraint",
    "OutageWindow",
    "MILPSolution",
    "SchedulingObjective",
    # Scenario analyzer
    "ScenarioAnalyzer",
    "ScenarioConfig",
    "ScenarioDefinition",
    "ScenarioComparison",
    "SensitivityResult",
    "MonteCarloResult",
    "TornadoChart",
    "BreakevenAnalysis",
]

__version__ = "1.0.0"
