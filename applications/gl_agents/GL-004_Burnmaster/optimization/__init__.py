"""
GL-004 Burnmaster - Optimization Module

Multi-objective constrained optimization for combustion control.

This module implements the optimization objective:
    Minimize: (fuel_cost * fuel_rate) + (emissions_cost * NOx)
              + (CO_penalty * max(0, CO - CO_limit))
              + (stability_penalty * instability_risk)
              + (move_penalty * actuator_moves)
    Subject to: duty constraint, hard safety limits, actuator bounds, rate limits

Components:
    - objective_functions: Multi-objective optimization functions
    - constraint_handler: Constraint management system
    - combustion_optimizer: Main optimization engine
    - setpoint_optimizer: Individual setpoint optimization
    - recommendation_engine: Recommendation generation and tracking
    - scenario_evaluator: Scenario analysis and sensitivity studies

Author: GreenLang Optimization Team
Version: 1.0.0
"""

# Objective Functions
from .objective_functions import (
    # Enums
    ObjectiveType,
    OptimizationDirection,
    # Data models
    BurnerState,
    SetpointVector,
    ObjectiveEvaluation,
    MultiObjectiveResult,
    # Base class
    BaseObjectiveFunction,
    # Objective implementations
    FuelCostObjective,
    EmissionsObjective,
    StabilityObjective,
    ActuatorMoveObjective,
    # Multi-objective
    MultiObjectiveFunction,
    # Factory functions
    create_fuel_focused_objective,
    create_emissions_focused_objective,
    create_balanced_objective,
)

# Constraint Handler
from .constraint_handler import (
    # Enums
    ConstraintType,
    ConstraintStatus,
    ViolationSeverity,
    # Data models
    ConstraintEvaluation,
    ConstraintSetResult,
    # Base class
    BaseConstraint,
    # Constraint implementations
    HardConstraint,
    SoftConstraint,
    RateLimitConstraint,
    DeadbandConstraint,
    # Constraint set
    ConstraintSet,
    # Factory functions
    create_combustion_constraint_set,
)

# Combustion Optimizer
from .combustion_optimizer import (
    # Enums
    OptimizerStatus,
    OptimizerMethod,
    # Data models
    TrajectoryPoint,
    TrajectoryPlan,
    OptimizationResult,
    # Main class
    CombustionOptimizer,
)

# Setpoint Optimizer
from .setpoint_optimizer import (
    # Enums
    SetpointPriority,
    SetpointDirection,
    # Data models
    SetpointRecommendation,
    CoordinatedPlan,
    # Main class
    SetpointOptimizer,
)

# Recommendation Engine
from .recommendation_engine import (
    # Enums
    RecommendationType,
    RecommendationStatus,
    ImpactLevel,
    # Data models
    Recommendation,
    ExplanationPayload,
    RecommendationOutcome,
    # Main class
    RecommendationEngine,
)

# Scenario Evaluator
from .scenario_evaluator import (
    # Enums
    ScenarioType,
    # Data models
    ScenarioOutcome,
    SensitivityResult,
    SensitivityResults,
    RobustSetpoints,
    FuelSwitchAnalysis,
    # Main class
    ScenarioEvaluator,
)

__all__ = [
    # Objective Functions
    "ObjectiveType",
    "OptimizationDirection",
    "BurnerState",
    "SetpointVector",
    "ObjectiveEvaluation",
    "MultiObjectiveResult",
    "BaseObjectiveFunction",
    "FuelCostObjective",
    "EmissionsObjective",
    "StabilityObjective",
    "ActuatorMoveObjective",
    "MultiObjectiveFunction",
    "create_fuel_focused_objective",
    "create_emissions_focused_objective",
    "create_balanced_objective",
    # Constraint Handler
    "ConstraintType",
    "ConstraintStatus",
    "ViolationSeverity",
    "ConstraintEvaluation",
    "ConstraintSetResult",
    "BaseConstraint",
    "HardConstraint",
    "SoftConstraint",
    "RateLimitConstraint",
    "DeadbandConstraint",
    "ConstraintSet",
    "create_combustion_constraint_set",
    # Combustion Optimizer
    "OptimizerStatus",
    "OptimizerMethod",
    "TrajectoryPoint",
    "TrajectoryPlan",
    "OptimizationResult",
    "CombustionOptimizer",
    # Setpoint Optimizer
    "SetpointPriority",
    "SetpointDirection",
    "SetpointRecommendation",
    "CoordinatedPlan",
    "SetpointOptimizer",
    # Recommendation Engine
    "RecommendationType",
    "RecommendationStatus",
    "ImpactLevel",
    "Recommendation",
    "ExplanationPayload",
    "RecommendationOutcome",
    "RecommendationEngine",
    # Scenario Evaluator
    "ScenarioType",
    "ScenarioOutcome",
    "SensitivityResult",
    "SensitivityResults",
    "RobustSetpoints",
    "FuelSwitchAnalysis",
    "ScenarioEvaluator",
]

__version__ = "1.0.0"
__author__ = "GreenLang Optimization Team"
