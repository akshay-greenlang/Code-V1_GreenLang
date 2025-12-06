# -*- coding: utf-8 -*-
"""
GreenLang Uncertainty Quantification

Provides uncertainty quantification capabilities for ML predictions,
including ensemble methods, conformal prediction, and calibration.

Classes:
    EnsemblePredictor: 10-model ensemble for uncertainty estimation
    ConformalPredictor: Conformal prediction for valid confidence intervals
    Calibrator: Temperature scaling and calibration methods
    ConfidenceReporter: Confidence interval API

Example:
    >>> from greenlang.ml.uncertainty import EnsemblePredictor, ConformalPredictor
    >>> ensemble = EnsemblePredictor(n_models=10)
    >>> prediction, uncertainty = ensemble.predict_with_uncertainty(X)
    >>> conformal = ConformalPredictor(model, calibration_data)
    >>> intervals = conformal.predict_interval(X, confidence=0.95)
"""

from greenlang.ml.uncertainty.ensemble_predictor import EnsemblePredictor
from greenlang.ml.uncertainty.conformal_prediction import ConformalPredictor
from greenlang.ml.uncertainty.calibration import Calibrator
from greenlang.ml.uncertainty.confidence_reporter import ConfidenceReporter

__all__ = [
    "EnsemblePredictor",
    "ConformalPredictor",
    "Calibrator",
    "ConfidenceReporter",
]
