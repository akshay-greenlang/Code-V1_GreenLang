"""
GL-016 Waterguard Optimization Module

Optimization service for cooling tower water treatment systems.
Implements Mixed Integer Linear Programming (MILP), scenario-based
optimization, and uncertainty quantification for blowdown and
chemical dosing control.

Key Components:
    - WaterguardOptimizer: MILP optimizer for constrained optimization
    - ScenarioOptimizer: What-if analysis and robust optimization
    - UncertaintyModel: Sensor accuracy and model uncertainty
    - ObjectiveFunctions: Multi-objective optimization components
    - ConstraintHandler: Hard/soft constraint management
    - SolverMonitor: Optimization solver monitoring
    - CalibrationTracker: Valve/pump calibration tracking

Optimization Objective:
    Minimize(Water_Loss + Energy_Loss + Chemical_Cost + Risk_Penalty)
    Subject to: Chemistry_Constraints AND Equipment_Constraints AND Ramp_Rate_Limits

Reference Standards:
    - CTI STD-201 (Cooling Tower Water Treatment)
    - ASHRAE 188 (Legionella Risk Management)
    - IEC 61131-3 (Control Systems)

Author: GreenLang Water Treatment Team
Version: 1.0.0
"""

__version__ = "1.0.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .milp_optimizer import (
        WaterguardOptimizer,
        OptimizationResult,
        BlowdownSetpoint,
        DosingSetpoint,
        OptimizationConfig,
        OptimizationStatus,
    )
    from .scenario_optimizer import (
        ScenarioOptimizer,
        ScenarioResult,
        RobustSolution,
        UncertaintyRange,
    )
    from .uncertainty_models import (
        UncertaintyModel,
        SensorUncertainty,
        UQResult,
        ConfidenceInterval,
        PredictionInterval,
    )
    from .objective_functions import (
        WaterLossObjective,
        EnergyLossObjective,
        ChemicalCostObjective,
        RiskPenaltyObjective,
        WeightedSumObjective,
        ParetoFrontier,
    )
    from .constraint_handler import (
        ConstraintHandler,
        HardConstraint,
        SoftConstraint,
        ConstraintViolation,
        ConstraintRelaxation,
    )
    from .solver_monitor import (
        SolverMonitor,
        SolverMetrics,
        SolverStatus,
        SolverDiagnostics,
    )
    from .calibration_tracker import (
        CalibrationTracker,
        ValveCalibration,
        PumpCalibration,
        CalibrationDrift,
        CalibrationAlert,
    )

__all__ = [
    # MILP Optimizer
    "WaterguardOptimizer",
    "OptimizationResult",
    "BlowdownSetpoint",
    "DosingSetpoint",
    "OptimizationConfig",
    "OptimizationStatus",
    # Scenario Optimizer
    "ScenarioOptimizer",
    "ScenarioResult",
    "RobustSolution",
    "UncertaintyRange",
    # Uncertainty Models
    "UncertaintyModel",
    "SensorUncertainty",
    "UQResult",
    "ConfidenceInterval",
    "PredictionInterval",
    # Objective Functions
    "WaterLossObjective",
    "EnergyLossObjective",
    "ChemicalCostObjective",
    "RiskPenaltyObjective",
    "WeightedSumObjective",
    "ParetoFrontier",
    # Constraint Handler
    "ConstraintHandler",
    "HardConstraint",
    "SoftConstraint",
    "ConstraintViolation",
    "ConstraintRelaxation",
    # Solver Monitor
    "SolverMonitor",
    "SolverMetrics",
    "SolverStatus",
    "SolverDiagnostics",
    # Calibration Tracker
    "CalibrationTracker",
    "ValveCalibration",
    "PumpCalibration",
    "CalibrationDrift",
    "CalibrationAlert",
]
