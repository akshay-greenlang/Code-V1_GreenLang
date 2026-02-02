# -*- coding: utf-8 -*-
"""
GL-023 HeatLoadBalancer Optimizers Module
=========================================

This module provides Mixed Integer Linear Programming (MILP) optimization
capabilities for industrial heat load balancing, including:

- MILPLoadBalancer: Core MILP solver for optimal equipment allocation
- ConstraintManager: Constraint validation and relaxation
- ParetoOptimizer: Multi-objective optimization with Pareto frontier
- HeuristicLoadBalancer: Fast heuristic fallback methods

All optimizers follow GreenLang's zero-hallucination principle:
- Deterministic calculations only
- SHA-256 provenance tracking
- Complete audit trails

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.optimizers import (
    ...     MILPLoadBalancer,
    ...     ConstraintManager,
    ...     ParetoOptimizer,
    ...     HeuristicLoadBalancer
    ... )
    >>> solver = MILPLoadBalancer()
    >>> result = solver.solve(time_limit_s=60)

Author: GreenLang Framework Team
Agent: GL-023 HeatLoadBalancer
Status: Production Ready
"""

from .milp_solver import (
    MILPLoadBalancer,
    MILPConfig,
    MILPSolverStatus,
    EquipmentSetpoint,
    OptimizationResult,
    ObjectiveType,
)

from .constraint_handler import (
    ConstraintManager,
    ConstraintViolation,
    ConstraintType,
    FeasibilityResult,
    RelaxationResult,
)

from .multi_objective import (
    ParetoOptimizer,
    ParetoPoint,
    ParetoFrontier,
    WeightedSumResult,
    KneePointResult,
)

from .heuristic_fallback import (
    HeuristicLoadBalancer,
    HeuristicMethod,
    HeuristicResult,
    MeritOrderConfig,
)


__all__ = [
    # MILP Solver
    "MILPLoadBalancer",
    "MILPConfig",
    "MILPSolverStatus",
    "EquipmentSetpoint",
    "OptimizationResult",
    "ObjectiveType",
    # Constraint Handler
    "ConstraintManager",
    "ConstraintViolation",
    "ConstraintType",
    "FeasibilityResult",
    "RelaxationResult",
    # Multi-Objective
    "ParetoOptimizer",
    "ParetoPoint",
    "ParetoFrontier",
    "WeightedSumResult",
    "KneePointResult",
    # Heuristic Fallback
    "HeuristicLoadBalancer",
    "HeuristicMethod",
    "HeuristicResult",
    "MeritOrderConfig",
]

__version__ = "1.0.0"
__agent__ = "GL-023"
