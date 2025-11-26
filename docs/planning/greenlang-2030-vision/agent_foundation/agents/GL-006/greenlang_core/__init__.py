# -*- coding: utf-8 -*-
"""
GreenLang Core Framework Stubs for GL-006 HeatRecoveryMaximizer.

This module provides the core framework components required for the GL-006 agent
including base agent classes, configuration management, validation, and provenance tracking.
"""

from .base_agent import BaseAgent, AgentConfig, AgentState, AgentStatus, AgentCapability
from .validation import ValidationResult, ValidationError, Validator, ValidationContext
from .provenance import ProvenanceTracker, ProvenanceRecord, DataLineage

__version__ = "1.0.0"
__author__ = "GreenLang Team"

__all__ = [
    # Base Agent
    'BaseAgent',
    'AgentConfig',
    'AgentState',
    'AgentStatus',
    'AgentCapability',
    # Validation
    'ValidationResult',
    'ValidationError',
    'Validator',
    'ValidationContext',
    # Provenance
    'ProvenanceTracker',
    'ProvenanceRecord',
    'DataLineage',
]
