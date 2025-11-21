# -*- coding: utf-8 -*-
"""
A/B Testing Framework for GL-002 BoilerEfficiencyOptimizer

This module provides comprehensive A/B testing capabilities for continuous
improvement through controlled experimentation.

Components:
    - ExperimentManager: Creates and manages A/B/C/n experiments
    - TrafficRouter: Routes users to experiment variants
    - StatisticalAnalyzer: Analyzes experiment results
    - ExperimentModels: Pydantic models for experiments
"""

from .experiment_models import (
    Experiment,
    ExperimentVariant,
    ExperimentResult,
    ExperimentMetrics,
    StatisticalSignificance
)
from .experiment_manager import ExperimentManager
from .traffic_router import TrafficRouter
from .statistical_analyzer import StatisticalAnalyzer

__all__ = [
    "Experiment",
    "ExperimentVariant",
    "ExperimentResult",
    "ExperimentMetrics",
    "StatisticalSignificance",
    "ExperimentManager",
    "TrafficRouter",
    "StatisticalAnalyzer"
]

__version__ = "1.0.0"
