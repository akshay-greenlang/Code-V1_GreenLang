# -*- coding: utf-8 -*-
"""
GreenLang Self-Learning Architecture

Provides adaptive learning capabilities for ML-powered agents,
including online learning, continual learning, meta-learning,
transfer learning, and comprehensive metrics tracking.

Modules:
    OnlineLearner: River/scikit-multiflow integration for streaming data
    ContinualLearner: EWC-based catastrophic forgetting prevention
    MetaLearner: MAML implementation for few-shot adaptation
    AdaptationTrigger: Model adaptation logic and triggers
    TransferLearningPipeline: Transfer learning for Process Heat domains
    IncrementalUpdater: Warm-start training and model updates
    LRSchedulers: Advanced learning rate scheduling
    ExperienceReplayBuffer: Prioritized experience replay
    MetricsDashboard: Self-learning metrics and monitoring

Example:
    >>> from greenlang.ml.self_learning import OnlineLearner, AdaptationTrigger
    >>> learner = OnlineLearner(model_type="hoeffding_tree")
    >>> trigger = AdaptationTrigger(performance_threshold=0.85)
    >>> for X, y in data_stream:
    ...     if trigger.should_adapt(metrics):
    ...         learner.partial_fit(X, y)

    >>> from greenlang.ml.self_learning import TransferLearningPipeline
    >>> pipeline = TransferLearningPipeline.load_pretrained("process_heat_base_v1")
    >>> pipeline.freeze_backbone()
    >>> pipeline.replace_head(num_classes=5)
    >>> result = pipeline.fine_tune(boiler_X, boiler_y)

    >>> from greenlang.ml.self_learning import MetricsDashboard
    >>> dashboard = MetricsDashboard()
    >>> dashboard.record_training_step(loss=0.5, accuracy=0.85)
    >>> summary = dashboard.get_summary()
"""

# Core self-learning modules
from greenlang.ml.self_learning.online_learner import (
    OnlineLearner,
    OnlineLearnerConfig,
    OnlineLearnerMetrics,
    OnlineLearnerResult,
    OnlineModelType,
)

from greenlang.ml.self_learning.continual_learning import (
    ContinualLearner,
    ContinualLearnerConfig,
    ContinualLearnerResult,
    ContinualMethod,
    TaskInfo,
    ForgettingMetrics,
)

from greenlang.ml.self_learning.meta_learner import (
    MetaLearner,
    MetaLearnerConfig,
    MetaLearningMethod,
    TaskData,
    AdaptationResult,
    MetaTrainingResult,
)

from greenlang.ml.self_learning.adaptation_triggers import (
    AdaptationTrigger,
    AdaptationTriggerConfig,
    TriggerType,
    TriggerAction,
    TriggerCondition,
    TriggerEvent,
    TriggerStatus,
    TriggerResult,
    create_performance_trigger,
    create_drift_trigger,
    create_scheduled_trigger,
)

# Transfer learning module
from greenlang.ml.self_learning.transfer_learning import (
    TransferLearningPipeline,
    TransferLearningConfig,
    TransferStrategy,
    ProcessHeatDomain,
    PretrainedModelInfo,
    TransferMetrics,
    FineTuningResult,
    create_boiler_transfer_pipeline,
    create_furnace_transfer_pipeline,
    create_steam_transfer_pipeline,
)

# Incremental updates module
from greenlang.ml.self_learning.incremental_updates import (
    IncrementalUpdater,
    IncrementalUpdateConfig,
    UpdateFrequency,
    UpdateMode,
    UpdateResult,
    UpdateSchedule,
    CheckpointInfo,
    create_daily_updater,
    create_continuous_updater,
    create_sliding_window_updater,
)

# Learning rate schedulers
from greenlang.ml.self_learning.lr_schedulers import (
    BaseLRScheduler,
    CyclicLRScheduler,
    CyclicLRConfig,
    CosineAnnealingScheduler,
    CosineAnnealingConfig,
    OneCycleScheduler,
    OneCycleConfig,
    ReduceOnPlateauScheduler,
    ReduceOnPlateauConfig,
    WarmupScheduler,
    SchedulerMode,
    SchedulerConfig,
    SchedulerState,
    create_cyclic_scheduler,
    create_cosine_annealing_scheduler,
    create_one_cycle_scheduler,
    create_reduce_on_plateau_scheduler,
)

# Experience replay buffer
from greenlang.ml.self_learning.experience_replay import (
    ExperienceReplayBuffer,
    ExperienceReplayConfig,
    Experience,
    SampleBatch,
    SamplingStrategy,
    BufferStatistics,
    ProcessHeatReplayBuffer,
    SumTree,
    create_prioritized_buffer,
    create_reservoir_buffer,
    create_importance_sampling_buffer,
)

# Metrics dashboard
from greenlang.ml.self_learning.metrics_dashboard import (
    MetricsDashboard,
    DashboardConfig,
    DashboardSummary,
    MetricType,
    MetricAggregation,
    MetricDataPoint,
    LearningCurve,
    DriftStatus,
    PlasticityMetrics,
    RetentionMetrics,
    MetricAlert,
    create_standard_dashboard,
    create_production_dashboard,
)

__all__ = [
    # Online Learner
    "OnlineLearner",
    "OnlineLearnerConfig",
    "OnlineLearnerMetrics",
    "OnlineLearnerResult",
    "OnlineModelType",
    # Continual Learning
    "ContinualLearner",
    "ContinualLearnerConfig",
    "ContinualLearnerResult",
    "ContinualMethod",
    "TaskInfo",
    "ForgettingMetrics",
    # Meta Learning
    "MetaLearner",
    "MetaLearnerConfig",
    "MetaLearningMethod",
    "TaskData",
    "AdaptationResult",
    "MetaTrainingResult",
    # Adaptation Triggers
    "AdaptationTrigger",
    "AdaptationTriggerConfig",
    "TriggerType",
    "TriggerAction",
    "TriggerCondition",
    "TriggerEvent",
    "TriggerStatus",
    "TriggerResult",
    "create_performance_trigger",
    "create_drift_trigger",
    "create_scheduled_trigger",
    # Transfer Learning
    "TransferLearningPipeline",
    "TransferLearningConfig",
    "TransferStrategy",
    "ProcessHeatDomain",
    "PretrainedModelInfo",
    "TransferMetrics",
    "FineTuningResult",
    "create_boiler_transfer_pipeline",
    "create_furnace_transfer_pipeline",
    "create_steam_transfer_pipeline",
    # Incremental Updates
    "IncrementalUpdater",
    "IncrementalUpdateConfig",
    "UpdateFrequency",
    "UpdateMode",
    "UpdateResult",
    "UpdateSchedule",
    "CheckpointInfo",
    "create_daily_updater",
    "create_continuous_updater",
    "create_sliding_window_updater",
    # Learning Rate Schedulers
    "BaseLRScheduler",
    "CyclicLRScheduler",
    "CyclicLRConfig",
    "CosineAnnealingScheduler",
    "CosineAnnealingConfig",
    "OneCycleScheduler",
    "OneCycleConfig",
    "ReduceOnPlateauScheduler",
    "ReduceOnPlateauConfig",
    "WarmupScheduler",
    "SchedulerMode",
    "SchedulerConfig",
    "SchedulerState",
    "create_cyclic_scheduler",
    "create_cosine_annealing_scheduler",
    "create_one_cycle_scheduler",
    "create_reduce_on_plateau_scheduler",
    # Experience Replay
    "ExperienceReplayBuffer",
    "ExperienceReplayConfig",
    "Experience",
    "SampleBatch",
    "SamplingStrategy",
    "BufferStatistics",
    "ProcessHeatReplayBuffer",
    "SumTree",
    "create_prioritized_buffer",
    "create_reservoir_buffer",
    "create_importance_sampling_buffer",
    # Metrics Dashboard
    "MetricsDashboard",
    "DashboardConfig",
    "DashboardSummary",
    "MetricType",
    "MetricAggregation",
    "MetricDataPoint",
    "LearningCurve",
    "DriftStatus",
    "PlasticityMetrics",
    "RetentionMetrics",
    "MetricAlert",
    "create_standard_dashboard",
    "create_production_dashboard",
]
