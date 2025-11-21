# -*- coding: utf-8 -*-
"""
GreenLang Zero-Hallucination Calculator Module

This package provides production-grade calculation capabilities with:
- Deterministic formula engine
- Emission factor databases
- Complete provenance tracking
- Regulatory compliance calculations
"""

from .formula_engine import FormulaEngine, Formula, FormulaLibrary
from .emission_factors import EmissionFactorDatabase, EmissionFactor
from .calculation_engine import CalculationEngine, CalculationStep, CalculationResult
from .unit_converter import UnitConverter, Unit
from .validators import CalculationValidator, ValidationResult

__all__ = [
    'FormulaEngine',
    'Formula',
    'FormulaLibrary',
    'EmissionFactorDatabase',
    'EmissionFactor',
    'CalculationEngine',
    'CalculationStep',
    'CalculationResult',
    'UnitConverter',
    'Unit',
    'CalculationValidator',
    'ValidationResult'
]
