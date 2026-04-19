# -*- coding: utf-8 -*-
"""
GreenLang Uncertainty Quantification

Provides uncertainty quantification capabilities for ML predictions,
including ensemble methods, conformal prediction, calibration, Bayesian
neural networks, visualization, and uncertainty-aware decision making.

Classes:
    EnsemblePredictor: 10-model ensemble for uncertainty estimation
    ConformalPredictor: Conformal prediction for valid confidence intervals
    AdaptiveConformalPredictor: Adaptive conformal for distribution shift
    ConformalClassifier: Conformal prediction sets for classification
    CoverageValidator: Coverage guarantee validation
    Calibrator: Temperature scaling and calibration methods
    CalibrationDiagnostics: Comprehensive calibration analysis
    BinaryCalibrator: Platt/isotonic calibration for binary classification
    ConfidenceReporter: Confidence interval API
    BayesianNeuralNetwork: MC Dropout and variational inference
    UncertaintyAPIService: FastAPI endpoints for uncertainty
    UncertaintyVisualizer: Visualization data generators
    DecisionEngine: Uncertainty-aware decision making

Example:
    >>> from greenlang.ml.uncertainty import EnsemblePredictor, ConformalPredictor
    >>> ensemble = EnsemblePredictor(n_models=10)
    >>> prediction, uncertainty = ensemble.predict_with_uncertainty(X)
    >>> conformal = ConformalPredictor(model, calibration_data)
    >>> intervals = conformal.predict_interval(X, confidence=0.95)

    # Bayesian Neural Network
    >>> from greenlang.ml.uncertainty import BayesianNeuralNetwork
    >>> bnn = BayesianNeuralNetwork(config=BayesianNNConfig(input_dim=10))
    >>> bnn.fit(X_train, y_train)
    >>> result = bnn.predict_with_uncertainty(X_test)
    >>> print(f"Epistemic: {result.epistemic_uncertainty[0]:.4f}")

    # Uncertainty-Aware Decision Making
    >>> from greenlang.ml.uncertainty import DecisionEngine
    >>> engine = DecisionEngine(config=DecisionConfig(confidence_threshold=0.90))
    >>> decision = engine.make_decision(prediction, uncertainty)
"""

from greenlang.ml.uncertainty.ensemble_predictor import EnsemblePredictor
from greenlang.ml.uncertainty.conformal_prediction import (
    ConformalPredictor,
    AdaptiveConformalPredictor,
    ConformalClassifier,
    CoverageValidator,
)
from greenlang.ml.uncertainty.calibration import (
    Calibrator,
    CalibrationDiagnostics,
    BinaryCalibrator,
    ReliabilityDiagramData,
)
from greenlang.ml.uncertainty.confidence_reporter import ConfidenceReporter
from greenlang.ml.uncertainty.bayesian_nn import (
    BayesianNeuralNetwork,
    BayesianNNConfig,
    BayesianPrediction,
    MCDropoutWrapper,
    ProcessHeatFeatures,
)
from greenlang.ml.uncertainty.api import (
    UncertaintyAPIService,
    create_uncertainty_router,
    PredictionRequest,
    PredictionResponse,
)
from greenlang.ml.uncertainty.visualization import (
    UncertaintyVisualizer,
    HeatmapData,
    ConfidenceBandData,
    CalibrationPlotData,
    FeatureCorrelationData,
    UncertaintyDistributionData,
    UncertaintyDecompositionData,
)
from greenlang.ml.uncertainty.decision_making import (
    DecisionEngine,
    DecisionConfig,
    Decision,
    RiskAssessment,
    HumanReviewRequest,
    OptimizationResult,
)

__all__ = [
    # Ensemble
    "EnsemblePredictor",
    # Conformal
    "ConformalPredictor",
    "AdaptiveConformalPredictor",
    "ConformalClassifier",
    "CoverageValidator",
    # Calibration
    "Calibrator",
    "CalibrationDiagnostics",
    "BinaryCalibrator",
    "ReliabilityDiagramData",
    # Confidence Reporter
    "ConfidenceReporter",
    # Bayesian NN
    "BayesianNeuralNetwork",
    "BayesianNNConfig",
    "BayesianPrediction",
    "MCDropoutWrapper",
    "ProcessHeatFeatures",
    # API
    "UncertaintyAPIService",
    "create_uncertainty_router",
    "PredictionRequest",
    "PredictionResponse",
    # Visualization
    "UncertaintyVisualizer",
    "HeatmapData",
    "ConfidenceBandData",
    "CalibrationPlotData",
    "FeatureCorrelationData",
    "UncertaintyDistributionData",
    "UncertaintyDecompositionData",
    # Decision Making
    "DecisionEngine",
    "DecisionConfig",
    "Decision",
    "RiskAssessment",
    "HumanReviewRequest",
    "OptimizationResult",
]
