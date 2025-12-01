# -*- coding: utf-8 -*-
"""
Calculators module for GL-012 STEAMQUAL SteamQualityController agent.

This module provides deterministic calculators for steam quality control operations
including steam quality analysis, desuperheater control, pressure management,
moisture analysis, and provenance tracking.

All calculators follow zero-hallucination principles with deterministic algorithms
and complete audit trail tracking via SHA-256 provenance hashing.

Standards:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ASME PTC 19.11: Steam and Water Sampling
- ISO 3046: Reciprocating internal combustion engines

Zero-hallucination: All calculations are deterministic with bit-perfect reproducibility.

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from .steam_quality_calculator import (
    SteamQualityCalculator,
    SteamState,
    SteamQualityInput,
    SteamQualityOutput,
)
from .desuperheater_calculator import (
    DesuperheaterCalculator,
    DesuperheaterInput,
    DesuperheaterOutput,
    ControlSignal,
)
from .pressure_control_calculator import (
    PressureControlCalculator,
    PressureControlInput,
    PressureControlOutput,
    PIDGains,
)
from .moisture_analyzer import (
    MoistureAnalyzer,
    MoistureAnalysisInput,
    MoistureAnalysisOutput,
    RiskLevel,
    MoistureSource,
)
from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    ProvenanceValidator,
    create_calculation_hash,
)

__all__ = [
    # Steam Quality Calculator
    'SteamQualityCalculator',
    'SteamState',
    'SteamQualityInput',
    'SteamQualityOutput',
    # Desuperheater Calculator
    'DesuperheaterCalculator',
    'DesuperheaterInput',
    'DesuperheaterOutput',
    'ControlSignal',
    # Pressure Control Calculator
    'PressureControlCalculator',
    'PressureControlInput',
    'PressureControlOutput',
    'PIDGains',
    # Moisture Analyzer
    'MoistureAnalyzer',
    'MoistureAnalysisInput',
    'MoistureAnalysisOutput',
    'RiskLevel',
    'MoistureSource',
    # Provenance Tracking
    'ProvenanceTracker',
    'ProvenanceRecord',
    'CalculationStep',
    'ProvenanceValidator',
    'create_calculation_hash',
]

__version__ = '1.0.0'
__author__ = 'GL-CalculatorEngineer'
