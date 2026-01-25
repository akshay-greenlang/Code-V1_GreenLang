# -*- coding: utf-8 -*-
"""
GreenLang Experimentation Framework

Provides A/B testing and experimentation tools specifically designed for
Process Heat agents, enabling rigorous comparison of model variants,
optimization strategies, and operational improvements.

Classes:
    ABTestManager: Main A/B testing orchestrator
    ExperimentMetrics: Metric collection and storage
    StatisticalAnalyzer: Statistical significance testing

Example:
    >>> from greenlang.ml.experimentation import ABTestManager
    >>> manager = ABTestManager()
    >>> exp_id = manager.create_experiment(
    ...     name="boiler_efficiency_v2",
    ...     variants=["baseline", "optimized"],
    ...     traffic_split={"baseline": 0.5, "optimized": 0.5}
    ... )
    >>> result = manager.analyze_results(exp_id)
    >>> print(f"Winner: {result.winner}")
"""

from greenlang.ml.experimentation.ab_testing import (
    ABTestManager,
    ExperimentMetrics,
    StatisticalAnalyzer,
    ExperimentResult,
    VariantMetrics,
    MetricType,
    TestType,
)

__all__ = [
    "ABTestManager",
    "ExperimentMetrics",
    "StatisticalAnalyzer",
    "ExperimentResult",
    "VariantMetrics",
    "MetricType",
    "TestType",
]
