# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Uncertainty Module

Comprehensive uncertainty quantification for combustion measurements
and predictions. Implements GUM (Guide to Expression of Uncertainty
in Measurement) principles for both Type A (statistical) and Type B
(systematic) uncertainty analysis.

Key Components:
    - SensorUncertaintyManager: Sensor-level uncertainty management
    - UncertaintyPropagator: Uncertainty propagation through calculations
    - UncertaintyGate: Gating recommendations based on uncertainty
    - AnalyzerUncertaintyModel: Gas analyzer specific uncertainty
    - DerivedVariableUncertainty: Uncertainty for derived KPIs
    - UncertaintyReporter: Report generation and visualization

Design Principles:
    - ZERO HALLUCINATION: All calculations are deterministic
    - GUM Compliant: Follows ISO GUM principles
    - Audit Ready: SHA-256 provenance hashes for all calculations
    - Complete Traceability: Full uncertainty budgets

Typical Sensor Uncertainties:
    - O2 analyzers: +/-0.1-0.5% accuracy
    - CO analyzers: +/-1-5 ppm or 2-5% of reading
    - Flow meters: +/-0.5-2% accuracy
    - Thermocouples: +/-0.75% or +/-2.2C (whichever is greater)
    - Pressure transmitters: +/-0.1-0.25% of span

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

# Sensor Uncertainty Management
from .sensor_uncertainty import (
    SensorUncertaintyManager,
    SensorSpecs,
    SensorUncertainty,
    MeasurementUncertainty,
    PropagatedUncertainty,
    CalibrationResult,
    SensorType,
    DistributionType,
    UncertaintyType,
)

# Uncertainty Propagation
from .uncertainty_propagator import (
    UncertaintyPropagator,
    MonteCarloResult,
    ConfidenceInterval,
    PredictionUncertainty,
    CombinedUncertainty,
    PropagationMethod,
)

# Uncertainty Gating
from .uncertainty_gate import (
    UncertaintyGate,
    GateResult,
    BlockResult,
    ActionRecommendation,
    GateDecision,
    UncertaintyReport as GateUncertaintyReport,
    GateStatus,
    ActionType,
    VariableCategory,
)

# Analyzer-Specific Uncertainty
from .analyzer_uncertainty import (
    AnalyzerUncertaintyModel,
    LagModel,
    DriftModel,
    BiasEstimate,
    CalibrationData,
    AnalyzerType,
)

# Derived Variable Uncertainty
from .derived_uncertainty import (
    DerivedVariableUncertainty,
    KPIUncertainty,
    DerivedVariableType,
)

# Uncertainty Reporting
from .uncertainty_reporter import (
    UncertaintyReporter,
    UncertaintyBudget,
    UncertaintyReport,
    Contributor,
    CalculationRecord,
    Figure,
    ReportFormat,
)


__all__ = [
    # Sensor Uncertainty
    "SensorUncertaintyManager",
    "SensorSpecs",
    "SensorUncertainty",
    "MeasurementUncertainty",
    "PropagatedUncertainty",
    "CalibrationResult",
    "SensorType",
    "DistributionType",
    "UncertaintyType",
    # Propagation
    "UncertaintyPropagator",
    "MonteCarloResult",
    "ConfidenceInterval",
    "PredictionUncertainty",
    "CombinedUncertainty",
    "PropagationMethod",
    # Gating
    "UncertaintyGate",
    "GateResult",
    "BlockResult",
    "ActionRecommendation",
    "GateDecision",
    "GateUncertaintyReport",
    "GateStatus",
    "ActionType",
    "VariableCategory",
    # Analyzer
    "AnalyzerUncertaintyModel",
    "LagModel",
    "DriftModel",
    "BiasEstimate",
    "CalibrationData",
    "AnalyzerType",
    # Derived
    "DerivedVariableUncertainty",
    "KPIUncertainty",
    "DerivedVariableType",
    # Reporting
    "UncertaintyReporter",
    "UncertaintyBudget",
    "UncertaintyReport",
    "Contributor",
    "CalculationRecord",
    "Figure",
    "ReportFormat",
]

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"
