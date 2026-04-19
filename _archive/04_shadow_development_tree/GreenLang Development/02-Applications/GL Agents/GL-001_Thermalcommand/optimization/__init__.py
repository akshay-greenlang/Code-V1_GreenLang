"""
GL-001 ThermalCommand Optimization Module

Uncertainty Quantification (UQ) Engine for robust planning and optimization.
Implements deterministic, bit-perfect, reproducible calculations with
SHA-256 provenance tracking for regulatory compliance.

Key Components:
    - UQ Schemas: Data models for uncertainty quantification
    - Uncertainty Models: Prediction intervals, scenarios, variability
    - Scenario Optimizer: Robust optimization under uncertainty
    - Calibration Tracker: Interval coverage and drift detection
    - UQ Display: Visualization data generation

Reference Standards:
    - GHG Protocol (emission calculations)
    - ISO 50001:2018 (Energy Management)
    - IEC 61131-3 (Control Systems)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

__version__ = "1.0.0"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .uq_schemas import (
        PredictionInterval,
        Scenario,
        ScenarioSet,
        CalibrationMetrics,
        UncertaintyBand,
        RobustSolution,
        UncertaintySource,
        QuantileSet,
        ProvenanceRecord,
    )
    from .uncertainty_models import (
        UncertaintyModelEngine,
        WeatherUncertaintyModel,
        PriceUncertaintyModel,
        DemandUncertaintyModel,
    )
    from .scenario_optimizer import (
        ScenarioOptimizer,
        RobustConstraintEngine,
        StochasticProgrammingEngine,
    )
    from .calibration_tracker import (
        CalibrationTracker,
        DriftDetector,
        ReliabilityDiagramGenerator,
    )
    from .uq_display import (
        UQDisplayEngine,
        FanChartGenerator,
        RiskAssessmentEngine,
    )

__all__ = [
    # Schemas
    "PredictionInterval",
    "Scenario",
    "ScenarioSet",
    "CalibrationMetrics",
    "UncertaintyBand",
    "RobustSolution",
    "UncertaintySource",
    "QuantileSet",
    "ProvenanceRecord",
    # Uncertainty Models
    "UncertaintyModelEngine",
    "WeatherUncertaintyModel",
    "PriceUncertaintyModel",
    "DemandUncertaintyModel",
    # Scenario Optimizer
    "ScenarioOptimizer",
    "RobustConstraintEngine",
    "StochasticProgrammingEngine",
    # Calibration Tracker
    "CalibrationTracker",
    "DriftDetector",
    "ReliabilityDiagramGenerator",
    # UQ Display
    "UQDisplayEngine",
    "FanChartGenerator",
    "RiskAssessmentEngine",
]
