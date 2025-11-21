# -*- coding: utf-8 -*-
"""
Automated Analysis Module for GL-002 Continuous Improvement

This module provides automated analysis of feedback and experiment data
to identify opportunities for improvement.

Components:
    - FeedbackAnalyzer: Analyzes feedback patterns and trends
    - PerformanceAnalyzer: Identifies underperforming optimizations
    - RecommendationEngine: Generates improvement recommendations
    - ReportGenerator: Creates automated weekly/monthly reports
"""

from .feedback_analyzer import FeedbackAnalyzer
from .performance_analyzer import PerformanceAnalyzer
from .recommendation_engine import RecommendationEngine
from .report_generator import ReportGenerator

__all__ = [
    "FeedbackAnalyzer",
    "PerformanceAnalyzer",
    "RecommendationEngine",
    "ReportGenerator"
]

__version__ = "1.0.0"
