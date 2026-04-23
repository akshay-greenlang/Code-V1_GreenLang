"""
Uncertainty Quantification Module for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module provides comprehensive uncertainty quantification capabilities:
- Sensor uncertainty tracking with calibration and drift modeling
- Uncertainty propagation (linear, Jacobian, Monte Carlo methods)
- Quality gates for uncertainty-based decision making
- User-friendly uncertainty reporting

Zero-Hallucination Guarantee:
All calculations in this module are deterministic mathematical operations.
No LLM inference is used in any calculation path. Every result is:
- Reproducible (same input -> same output)
- Auditable (complete provenance tracking)
- Verifiable (SHA-256 hashes for audit trails)

Example Usage:
    from uncertainty import (
        SensorUncertaintyManager,
        UncertaintyPropagator,
        UncertaintyGate,
        UncertaintyReporter,
        UncertainValue,
        Distribution,
        DriftClass
    )

    # Register sensors with uncertainty metadata
    manager = SensorUncertaintyManager()
    manager.register_sensor(
        sensor_id="TT-101",
        accuracy_percent=0.5,
        calibration_date=datetime(2024, 1, 15),
        drift_class=DriftClass.CLASS_B
    )

    # Create uncertain values from measurements
    temp_uncertain = manager.get_measurement_uncertainty("TT-101", 450.0)

    # Propagate uncertainty through calculations
    propagator = UncertaintyPropagator()
    result = propagator.propagate_nonlinear(
        inputs={"temperature": temp_uncertain, "pressure": pressure_uncertain},
        function=compute_steam_property,
        jacobian=compute_steam_property_jacobian
    )

    # Check quality gates before recommendations
    gate = UncertaintyGate()
    gate_result = gate.check_recommendation_confidence(recommendation, min_confidence=0.90)

    # Format for display
    reporter = UncertaintyReporter()
    formatted = reporter.format_with_bounds(result.value, result.uncertainty)
    # Output: "1234.5 +/- 12.3 (95% CI)"
"""

# Version and metadata
__version__ = "1.0.0"
__author__ = "GreenLang GL-003 Team"
__description__ = "Zero-hallucination uncertainty quantification for steam system optimization"

# Core data models
from .uncertainty_models import (
    # Enums
    DistributionType,
    DriftClass,
    ConfidenceLevel,

    # Core uncertain value
    UncertainValue,

    # Sensor-specific models
    SensorUncertainty,
    SensorRegistration,
    SensorFlag,

    # Property and propagation models
    PropertyUncertainty,
    PropagatedUncertainty,
    MonteCarloResult,
    Distribution,

    # Reporting models
    UncertaintyBreakdown,
    UncertaintySource,
)

# Sensor uncertainty management
from .sensor_uncertainty import (
    SensorUncertaintyManager,
    CalibrationRecord,
    DEFAULT_DRIFT_RATES,
    RECOMMENDED_CALIBRATION_INTERVALS,
    UNCERTAINTY_THRESHOLDS,
)

# Uncertainty propagation
from .propagation import (
    UncertaintyPropagator,
    CorrelationMatrix,
    combine_uncertainties,
    DEFAULT_FINITE_DIFF_STEP,
    DEFAULT_MC_SAMPLES,
    MC_CONVERGENCE_THRESHOLD,
)

# Quality gates
from .quality_gates import (
    UncertaintyGate,
    UncertaintyThresholds,
    GateResult,
    GateStatus,
    QualityCheckResult,
    Warning,
    Recommendation,
    RiskLevel,
    WarningPriority,
)

# Uncertainty reporting
from .uncertainty_reporter import (
    UncertaintyReporter,
    FormatStyle,
    InstrumentationRecommendation,
)

# Advanced Monte Carlo Engine
from .mc_engine import (
    MonteCarloEngine,
    MonteCarloConfig,
    ConvergenceDiagnostic,
    ExtendedMonteCarloResult,
)

# Steam-specific Monte Carlo
from .monte_carlo import (
    SteamMonteCarloEngine,
    SteamPropertyUncertainty,
    EnthalpyBalanceResult,
    VisualizationData,
    propagate_steam_property,
    analyze_enthalpy_balance,
)

# Extended quality gates
from .quality_gates_extended import (
    QualityGate,
    QualityGateConfig,
    QualityGateResult,
    UncertaintyQualityChecker,
    UncertaintyCheckResult,
    DataQualityValidator,
    DataValidationResult,
    ComplianceChecker,
    ComplianceCheckResult,
    QualityGateReport,
    generate_quality_gate_report,
    format_quality_gate_report,
)

# Extended reporting with audit features
from .reporter_extended import (
    AuditReporter,
    AuditDocumentation,
    ReductionRoadmap,
    ReductionAction,
)

# Public API - all exported symbols
__all__ = [
    # Version
    "__version__",
    "__author__",
    "__description__",

    # Enums
    "DistributionType",
    "DriftClass",
    "ConfidenceLevel",
    "GateStatus",
    "RiskLevel",
    "WarningPriority",
    "FormatStyle",

    # Core data models
    "UncertainValue",
    "SensorUncertainty",
    "SensorRegistration",
    "SensorFlag",
    "PropertyUncertainty",
    "PropagatedUncertainty",
    "MonteCarloResult",
    "Distribution",
    "UncertaintyBreakdown",
    "UncertaintySource",

    # Sensor management
    "SensorUncertaintyManager",
    "CalibrationRecord",
    "DEFAULT_DRIFT_RATES",
    "RECOMMENDED_CALIBRATION_INTERVALS",
    "UNCERTAINTY_THRESHOLDS",

    # Propagation
    "UncertaintyPropagator",
    "CorrelationMatrix",
    "combine_uncertainties",
    "DEFAULT_FINITE_DIFF_STEP",
    "DEFAULT_MC_SAMPLES",
    "MC_CONVERGENCE_THRESHOLD",

    # Quality gates
    "UncertaintyGate",
    "UncertaintyThresholds",
    "GateResult",
    "QualityCheckResult",
    "Warning",
    "Recommendation",

    # Reporting
    "UncertaintyReporter",
    "InstrumentationRecommendation",

    # Advanced Monte Carlo
    "MonteCarloEngine",
    "MonteCarloConfig",
    "ConvergenceDiagnostic",
    "ExtendedMonteCarloResult",

    # Steam Monte Carlo
    "SteamMonteCarloEngine",
    "SteamPropertyUncertainty",
    "EnthalpyBalanceResult",
    "VisualizationData",
    "propagate_steam_property",
    "analyze_enthalpy_balance",

    # Extended quality gates
    "QualityGate",
    "QualityGateConfig",
    "QualityGateResult",
    "UncertaintyQualityChecker",
    "UncertaintyCheckResult",
    "DataQualityValidator",
    "DataValidationResult",
    "ComplianceChecker",
    "ComplianceCheckResult",
    "QualityGateReport",
    "generate_quality_gate_report",
    "format_quality_gate_report",

    # Extended reporting
    "AuditReporter",
    "AuditDocumentation",
    "ReductionRoadmap",
    "ReductionAction",
]


def get_module_info() -> dict:
    """
    Get module information and capabilities.

    Returns:
        Dictionary with module metadata and feature list
    """
    return {
        "name": "GL-003 Uncertainty Quantification Module",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "features": [
            "Sensor uncertainty tracking with calibration history",
            "Time-degraded uncertainty based on drift class",
            "Linear uncertainty propagation",
            "Jacobian-based nonlinear propagation",
            "Monte Carlo uncertainty propagation",
            "Latin Hypercube Sampling (LHS)",
            "Cholesky decomposition for correlated inputs",
            "Gelman-Rubin convergence diagnostics",
            "Correlated input handling",
            "Sobol sensitivity analysis",
            "Quality gates for uncertainty-based decisions",
            "Operator confirmation requirements",
            "User-friendly uncertainty formatting",
            "Instrumentation improvement recommendations",
            "Audit-ready documentation generation",
            "Uncertainty reduction roadmaps"
        ],
        "guarantees": [
            "Zero hallucination - no LLM in calculation path",
            "Bit-perfect reproducibility",
            "Complete provenance tracking",
            "SHA-256 audit trail hashes"
        ],
        "supported_distributions": [d.value for d in DistributionType],
        "supported_drift_classes": [d.value for d in DriftClass],
        "supported_confidence_levels": [cl.value for cl in ConfidenceLevel]
    }
