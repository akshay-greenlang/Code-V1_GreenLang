# -*- coding: utf-8 -*-
"""
GreenLang MLOps Pipeline

Provides production-ready MLOps capabilities including model registry,
experiment tracking, A/B testing, drift detection, model rollback,
and Weights & Biases integration.

Classes:
    ModelRegistry: MLflow-based model versioning and deployment
    ExperimentTracker: Experiment tracking and logging
    ABTesting: A/B testing framework for model comparison
    DriftDetector: Data and concept drift detection
    AutoRetrainer: Automatic retraining pipeline
    RollbackManager: Model rollback mechanisms with audit logging

W&B Integration Classes:
    WandBConfig: Configuration for Weights & Biases integration
    WandBExperimentTracker: W&B experiment tracking
    WandBSweepManager: Hyperparameter sweep management
    WandBAlerting: Metric-based alerting
    WandBReportGenerator: Automated report generation
    WandBMLflowBridge: Bridge between W&B and MLflow
    ProcessHeatRunConfig: Agent-specific run configurations
    AgentType: Enum of supported Process Heat agent types
    SweepMethod: Enum of sweep optimization methods

Example:
    >>> from greenlang.ml.mlops import ModelRegistry, DriftDetector, RollbackManager
    >>> registry = ModelRegistry()
    >>> registry.register_model(model, "emission_predictor", version="1.0.0")
    >>> detector = DriftDetector()
    >>> if detector.detect_drift(new_data):
    ...     trigger_retraining()
    >>> rollback = RollbackManager(registry)
    >>> if rollback.should_rollback("model_name", "v2"):
    ...     rollback.execute_rollback("model_name", "v2", "v1")

W&B Example:
    >>> from greenlang.ml.mlops import WandBExperimentTracker, AgentType
    >>> tracker = WandBExperimentTracker()
    >>> with tracker.init_run("fuel_training", agent_type=AgentType.GL_008_FUEL):
    ...     tracker.log_hyperparameters({"learning_rate": 0.01})
    ...     tracker.log_metrics({"rmse": 0.05, "r2": 0.95})
    ...     tracker.log_model(model, "fuel_emission_model")
"""

from greenlang.ml.mlops.model_registry import ModelRegistry
from greenlang.ml.mlops.experiment_tracker import ExperimentTracker
from greenlang.ml.mlops.ab_testing import ABTesting
from greenlang.ml.mlops.drift_detector import DriftDetector
from greenlang.ml.mlops.auto_retrainer import AutoRetrainer
from greenlang.ml.mlops.rollback import RollbackManager

# W&B Integration imports
from greenlang.ml.mlops.wandb_integration import (
    WandBConfig,
    WandBExperimentTracker,
    WandBSweepManager,
    WandBAlerting,
    WandBReportGenerator,
    WandBMLflowBridge,
    WandBCacheManager,
    ProcessHeatRunConfig,
    AgentType,
    SweepMethod,
    AlertLevel,
    RunStatus,
    RunInfo,
    SweepInfo,
    AlertConfig,
    create_experiment_tracker,
    create_sweep,
)

__all__ = [
    # Core MLOps
    "ModelRegistry",
    "ExperimentTracker",
    "ABTesting",
    "DriftDetector",
    "AutoRetrainer",
    "RollbackManager",
    # W&B Integration - Main Classes
    "WandBConfig",
    "WandBExperimentTracker",
    "WandBSweepManager",
    "WandBAlerting",
    "WandBReportGenerator",
    "WandBMLflowBridge",
    "WandBCacheManager",
    # W&B Integration - Configuration
    "ProcessHeatRunConfig",
    # W&B Integration - Enums
    "AgentType",
    "SweepMethod",
    "AlertLevel",
    "RunStatus",
    # W&B Integration - Data Classes
    "RunInfo",
    "SweepInfo",
    "AlertConfig",
    # W&B Integration - Convenience Functions
    "create_experiment_tracker",
    "create_sweep",
]
