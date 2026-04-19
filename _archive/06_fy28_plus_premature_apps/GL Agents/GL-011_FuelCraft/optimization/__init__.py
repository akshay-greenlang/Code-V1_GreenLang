"""
GL-011 FuelCraft - Optimization Package

LP/MILP optimization for multi-fuel procurement, storage, and blending.

Components:
- FuelOptimizationModel: Model construction with proper indexing
- Solver: Interface for HiGHS, CBC, CPLEX, Gurobi
- ScenarioManager: Deterministic scenario analysis
- FuelMixOptimizer: Complete optimization workflow orchestrator
- CostModel: Deterministic cost calculation engine

Decision Variables:
- x_{i,t}: Procure quantity of fuel i in period t (MJ or kg)
- y_{i,t}: Consumed/withdrawn fuel i in period t
- s_{k,t}: Inventory in tank k at period t
- b_{i,t}: Blend fraction of fuel i in period t (sum = 1)
- z_{c,t}: Contract commitment decisions (binary/integer)

Objective:
Minimize: Purchase cost + Logistics cost + Penalties + Carbon cost + Risk term

Constraints:
1. Demand satisfaction (energy basis)
2. Inventory balance: s_{k,t} = s_{k,t-1} + inflow - outflow - losses
3. Tank/flow limits (min/max)
4. Blend quality limits (sulfur, ash, water, viscosity)
5. Contract constraints (take-or-pay, min/max)
6. Safety constraints (flash point, vapor pressure)

Zero-Hallucination Approach:
- All optimizations use deterministic solvers (no ML in objective)
- All calculations traceable to governed data and formulas
- Full provenance tracking with SHA-256 run bundle hashing
"""

from optimization.model_builder import (
    FuelOptimizationModel,
    ModelConfig,
    FuelData,
    TankData,
    ContractData,
    DemandData,
    OptimizationVariable,
    Constraint,
)
from optimization.solver import (
    Solver,
    SolverConfig,
    SolverType,
    SolverStatus,
    Solution,
)
from optimization.scenario_manager import (
    ScenarioManager,
    Scenario,
    ScenarioType,
    ScenarioResult,
    ScenarioComparison,
)
from optimization.cost_model import (
    CostModel,
    CostBreakdown,
    CostComponent,
    CostCategory,
    PurchaseCostParams,
    LogisticsCostParams,
    StorageCostParams,
    ContractPenaltyParams,
    CarbonCostParams,
    RiskCostParams,
    PricingType,
    LogisticsMode,
    CarbonScheme,
)
from optimization.fuel_mix_optimizer import (
    FuelMixOptimizer,
    OptimizerConfig,
    OptimizationResult,
    OptimizationStatus,
    FuelMixEntry,
    ProcurementSchedule,
    InventoryProjection,
    BlendQuality,
    BlendQualityStatus,
)


__all__ = [
    # Model Builder
    "FuelOptimizationModel",
    "ModelConfig",
    "FuelData",
    "TankData",
    "ContractData",
    "DemandData",
    "OptimizationVariable",
    "Constraint",
    # Solver
    "Solver",
    "SolverConfig",
    "SolverType",
    "SolverStatus",
    "Solution",
    # Scenario Manager
    "ScenarioManager",
    "Scenario",
    "ScenarioType",
    "ScenarioResult",
    "ScenarioComparison",
    # Cost Model
    "CostModel",
    "CostBreakdown",
    "CostComponent",
    "CostCategory",
    "PurchaseCostParams",
    "LogisticsCostParams",
    "StorageCostParams",
    "ContractPenaltyParams",
    "CarbonCostParams",
    "RiskCostParams",
    "PricingType",
    "LogisticsMode",
    "CarbonScheme",
    # Fuel Mix Optimizer
    "FuelMixOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "OptimizationStatus",
    "FuelMixEntry",
    "ProcurementSchedule",
    "InventoryProjection",
    "BlendQuality",
    "BlendQualityStatus",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-011"
