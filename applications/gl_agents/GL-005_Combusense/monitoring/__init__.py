# -*- coding: utf-8 -*-
"""
Real-time Monitoring Module for GL-005 CombustionSense
======================================================

This module provides:
    - streaming_validator: Real-time data validation
    - alarm_manager: Alarm threshold management (ISA-18.2)
    - trend_analyzer: Combustion quality trend analysis

All components are designed for:
    - Low-latency processing
    - Complete data provenance
    - Deterministic behavior
"""

from .streaming_validator import StreamingDataValidator, DataPoint, ValidationReport
from .alarm_manager import AlarmManager, AlarmThreshold, AlarmPriority
from .trend_analyzer import TrendAnalyzer, TrendResult, CombustionQualityReport

__all__ = [
    "StreamingDataValidator",
    "DataPoint",
    "ValidationReport",
    "AlarmManager",
    "AlarmThreshold",
    "AlarmPriority",
    "TrendAnalyzer",
    "TrendResult",
    "CombustionQualityReport",
]
