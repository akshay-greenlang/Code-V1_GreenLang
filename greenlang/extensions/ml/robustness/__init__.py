# -*- coding: utf-8 -*-
"""
GreenLang ML Robustness Framework

This module provides comprehensive robustness, safety, and reliability
capabilities for Process Heat ML models, including:

- Adversarial Testing: FGSM, PGD attacks, boundary testing, robustness scoring
- Distribution Shift Detection: PSI, KS test, MMD, Chi-squared
- Model Validation Pipeline: Cross-validation, out-of-time validation, nested CV
- Output Constraint Enforcement: Physical bounds, safety limits, conservation laws
- Prediction Anomaly Detection: Isolation Forest, One-class SVM, DBSCAN, Z-score
- Graceful Degradation: Fallback models, ensemble voting, rule-based backup

All calculations are deterministic with SHA-256 provenance hashing for
regulatory compliance and audit trail.

Example:
    >>> from greenlang.ml.robustness import (
    ...     AdversarialTestingFramework,
    ...     DistributionShiftDetector,
    ...     ValidationPipeline,
    ...     OutputConstraintEnforcer,
    ...     PredictionAnomalyDetector,
    ...     GracefulDegradationManager
    ... )

    # Adversarial testing
    >>> framework = AdversarialTestingFramework(model)
    >>> result = framework.comprehensive_test(X_test, y_test)
    >>> print(f"Robustness grade: {result.robustness_grade}")

    # Distribution shift detection
    >>> detector = DistributionShiftDetector(reference_data=X_train)
    >>> shift_result = detector.detect_shift(X_production)
    >>> if shift_result.shift_severity == ShiftSeverity.CRITICAL:
    ...     trigger_retraining()

    # Model validation
    >>> pipeline = ValidationPipeline(model)
    >>> validation = pipeline.validate(X, y)
    >>> print(f"Status: {validation.status}")

    # Output constraints
    >>> enforcer = OutputConstraintEnforcer()
    >>> constrained, result = enforcer.enforce(predictions)

    # Anomaly detection
    >>> anomaly_detector = PredictionAnomalyDetector()
    >>> anomaly_detector.fit(normal_predictions)
    >>> anomalies = anomaly_detector.detect(new_predictions)

    # Graceful degradation
    >>> manager = GracefulDegradationManager(primary_model, fallback_models)
    >>> safe_result = manager.predict_safe(X)
"""

# TASK-061: Adversarial Testing Framework
from greenlang.ml.robustness.adversarial_testing import (
    AdversarialTestingFramework,
    AdversarialTestingConfig,
    AdversarialTestingResult,
    AdversarialAttackMethod,
    ProcessHeatBounds,
    RobustnessScore,
    AdversarialSample,
    BoundaryTestResult,
    DetectionResult as AdversarialDetectionResult,
)

# TASK-062: Distribution Shift Detection
from greenlang.ml.robustness.distribution_shift_detection import (
    DistributionShiftDetector,
    DistributionShiftConfig,
    ShiftDetectionResult,
    ShiftType,
    ShiftSeverity,
    DetectionMethod as ShiftDetectionMethod,
    FeatureShiftResult,
)

# TASK-066: Model Validation Pipeline
from greenlang.ml.robustness.validation_pipeline import (
    ValidationPipeline,
    ValidationConfig,
    ValidationReport,
    ValidationStatus,
    CVStrategy,
    MetricType,
    FoldResult,
    OutOfTimeResult,
    NestedCVResult,
    SignificanceTestResult,
)

# TASK-068: Output Constraint Enforcement
from greenlang.ml.robustness.output_constraints import (
    OutputConstraintEnforcer,
    ProcessHeatConstraintConfig,
    EnforcementResult,
    ConstraintViolation,
    ConstraintType,
    ViolationSeverity,
    EnforcementAction,
    PhysicalBound,
    SafetyLimit,
    ConservationConstraint,
    MonotonicityConstraint,
    create_process_heat_enforcer,
)

# TASK-069: Prediction Anomaly Detection
from greenlang.ml.robustness.prediction_anomaly import (
    PredictionAnomalyDetector,
    AnomalyDetectionConfig,
    AnomalyDetectionResult,
    AnomalyMethod,
    AnomalySeverity,
    AlertType,
    AnomalyAlert,
    MethodResult,
    IsolationForestDetector,
    OneClassSVMDetector,
    DBSCANDetector,
    ZScoreDetector,
    MADDetector,
)

# TASK-070: Graceful Degradation
from greenlang.ml.robustness.graceful_degradation import (
    GracefulDegradationManager,
    DegradationConfig,
    PredictionResult,
    DegradationStatus,
    DegradationLevel,
    FallbackReason,
    VotingStrategy,
    DegradationEvent,
    ModelWrapper,
    RuleBasedModel,
    FeatureSubsetModel,
    create_process_heat_degradation_manager,
)

# Legacy exports (existing modules)
from greenlang.ml.robustness.adversarial_tester import AdversarialTester
from greenlang.ml.robustness.distribution_shift import DistributionShift

__all__ = [
    # TASK-061: Adversarial Testing
    "AdversarialTestingFramework",
    "AdversarialTestingConfig",
    "AdversarialTestingResult",
    "AdversarialAttackMethod",
    "ProcessHeatBounds",
    "RobustnessScore",
    "AdversarialSample",
    "BoundaryTestResult",
    "AdversarialDetectionResult",

    # TASK-062: Distribution Shift
    "DistributionShiftDetector",
    "DistributionShiftConfig",
    "ShiftDetectionResult",
    "ShiftType",
    "ShiftSeverity",
    "ShiftDetectionMethod",
    "FeatureShiftResult",

    # TASK-066: Validation Pipeline
    "ValidationPipeline",
    "ValidationConfig",
    "ValidationReport",
    "ValidationStatus",
    "CVStrategy",
    "MetricType",
    "FoldResult",
    "OutOfTimeResult",
    "NestedCVResult",
    "SignificanceTestResult",

    # TASK-068: Output Constraints
    "OutputConstraintEnforcer",
    "ProcessHeatConstraintConfig",
    "EnforcementResult",
    "ConstraintViolation",
    "ConstraintType",
    "ViolationSeverity",
    "EnforcementAction",
    "PhysicalBound",
    "SafetyLimit",
    "ConservationConstraint",
    "MonotonicityConstraint",
    "create_process_heat_enforcer",

    # TASK-069: Anomaly Detection
    "PredictionAnomalyDetector",
    "AnomalyDetectionConfig",
    "AnomalyDetectionResult",
    "AnomalyMethod",
    "AnomalySeverity",
    "AlertType",
    "AnomalyAlert",
    "MethodResult",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "DBSCANDetector",
    "ZScoreDetector",
    "MADDetector",

    # TASK-070: Graceful Degradation
    "GracefulDegradationManager",
    "DegradationConfig",
    "PredictionResult",
    "DegradationStatus",
    "DegradationLevel",
    "FallbackReason",
    "VotingStrategy",
    "DegradationEvent",
    "ModelWrapper",
    "RuleBasedModel",
    "FeatureSubsetModel",
    "create_process_heat_degradation_manager",

    # Legacy exports
    "AdversarialTester",
    "DistributionShift",
]
