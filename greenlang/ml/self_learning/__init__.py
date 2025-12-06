# -*- coding: utf-8 -*-
"""
GreenLang Self-Learning Architecture

Provides adaptive learning capabilities for ML-powered agents,
including online learning, continual learning, and meta-learning.

Classes:
    OnlineLearner: River/scikit-multiflow integration for streaming data
    ContinualLearner: EWC-based catastrophic forgetting prevention
    MetaLearner: MAML implementation for few-shot adaptation
    AdaptationTrigger: Model adaptation logic and triggers

Example:
    >>> from greenlang.ml.self_learning import OnlineLearner, AdaptationTrigger
    >>> learner = OnlineLearner(model_type="hoeffding_tree")
    >>> trigger = AdaptationTrigger(performance_threshold=0.85)
    >>> for X, y in data_stream:
    ...     if trigger.should_adapt(metrics):
    ...         learner.partial_fit(X, y)
"""

from greenlang.ml.self_learning.online_learner import OnlineLearner
from greenlang.ml.self_learning.continual_learning import ContinualLearner
from greenlang.ml.self_learning.meta_learner import MetaLearner
from greenlang.ml.self_learning.adaptation_triggers import AdaptationTrigger

__all__ = [
    "OnlineLearner",
    "ContinualLearner",
    "MetaLearner",
    "AdaptationTrigger",
]
