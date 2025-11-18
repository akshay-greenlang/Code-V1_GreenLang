"""
Feedback System for GL-002 BoilerEfficiencyOptimizer

This module provides user feedback collection, analysis, and continuous improvement
mechanisms for the boiler efficiency optimization agent.

Components:
    - FeedbackCollector: Collects and stores user feedback
    - FeedbackModels: Pydantic models for feedback data
    - FeedbackAPI: FastAPI endpoints for feedback operations
    - FeedbackAnalyzer: Analyzes feedback trends and patterns
"""

from .feedback_collector import FeedbackCollector
from .feedback_models import (
    OptimizationFeedback,
    FeedbackStats,
    SatisfactionTrend,
    FeedbackSummary
)
from .feedback_api import create_feedback_router

__all__ = [
    "FeedbackCollector",
    "OptimizationFeedback",
    "FeedbackStats",
    "SatisfactionTrend",
    "FeedbackSummary",
    "create_feedback_router"
]

__version__ = "1.0.0"
