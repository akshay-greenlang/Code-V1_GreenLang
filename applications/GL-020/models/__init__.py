# -*- coding: utf-8 -*-
"""
GL-020 ECONOPULSE Performance Models.

This module provides machine learning models for economizer performance
analysis, including fouling prediction, cleaning effectiveness estimation,
and anomaly detection. All models include deterministic fallbacks for
zero-hallucination operation.

Components:
    - FoulingPredictor: Predicts fouling rate from operating conditions
    - CleaningEffectivenessModel: Estimates cleaning impact
    - AnomalyDetector: Detects abnormal readings (leaks, sensor faults)

Note:
    All models follow zero-hallucination principles. ML predictions are
    used for trend analysis and early warning only. Critical calculations
    always use deterministic formulas.

Example:
    >>> from greenlang.GL_020.models import FoulingPredictor
    >>> predictor = FoulingPredictor(baseline_config)
    >>> rate = predictor.predict_fouling_rate(operating_data)

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from greenlang.GL_020.models.performance_models import (
    FoulingPredictor,
    CleaningEffectivenessModel,
    AnomalyDetector,
    AnomalyType,
    AnomalyResult,
)

__all__ = [
    "FoulingPredictor",
    "CleaningEffectivenessModel",
    "AnomalyDetector",
    "AnomalyType",
    "AnomalyResult",
]
