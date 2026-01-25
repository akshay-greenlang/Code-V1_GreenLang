"""
GreenLang ML Platform

Provides model API, evaluation harness, and routing capabilities for
production-grade LLM integration with zero-hallucination guarantees.

Components:
- model_api: FastAPI endpoints for model invocation and evaluation
- evaluation: Golden test execution and metrics collection
- router: Intelligent model routing with cost optimization
"""

from greenlang.ml_platform.model_api import app, ModelInvokeRequest, ModelInvokeResponse
from greenlang.ml_platform.evaluation import (
    GoldenTestExecutor,
    DeterminismValidator,
    MetricsCollector,
    EvaluationReport
)
from greenlang.ml_platform.router import ModelRouter, RoutingCriteria

__all__ = [
    "app",
    "ModelInvokeRequest",
    "ModelInvokeResponse",
    "GoldenTestExecutor",
    "DeterminismValidator",
    "MetricsCollector",
    "EvaluationReport",
    "ModelRouter",
    "RoutingCriteria"
]
