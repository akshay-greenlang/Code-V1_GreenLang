"""
ML Pipelines Module - Automated machine learning workflows for GreenLang.

This module provides production-grade ML pipeline implementations including
automated retraining, feature engineering, model validation, and deployment.

Modules:
    auto_retrain: Automated retraining pipeline with drift detection and
                  safe deployment controls

Example:
    >>> from greenlang.ml.pipelines import AutoRetrainPipeline
    >>> pipeline = AutoRetrainPipeline(config)
    >>> pipeline.configure_trigger(metric_threshold=0.92)
    >>> if pipeline.check_retrain_needed("heat_predictor"):
    ...     job_id = pipeline.start_retrain_job("heat_predictor", config)
"""

from greenlang.ml.pipelines.auto_retrain import (
    AutoRetrainPipeline,
    TriggerConfig,
    TriggerType,
    RetrainingStatus,
    ValidationResult,
    PerformanceDegradationTrigger,
    DataDriftTrigger,
    ScheduledTrigger,
)

__all__ = [
    "AutoRetrainPipeline",
    "TriggerConfig",
    "TriggerType",
    "RetrainingStatus",
    "ValidationResult",
    "PerformanceDegradationTrigger",
    "DataDriftTrigger",
    "ScheduledTrigger",
]
