# -*- coding: utf-8 -*-
"""
Insights Generation Modules
GL-VCCI Scope 3 Platform

Hotspot detection and actionable recommendation generation.

Version: 1.0.0
"""

from .hotspot_detector import HotspotDetector
from .recommendation_engine import RecommendationEngine

__all__ = [
    "HotspotDetector",
    "RecommendationEngine",
]
