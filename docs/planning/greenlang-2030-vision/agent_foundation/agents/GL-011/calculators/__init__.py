# -*- coding: utf-8 -*-
"""
Calculators module for GL-011 FUELCRAFT agent.

This module provides deterministic calculators for fuel management operations
including multi-fuel optimization, cost optimization, blending, carbon footprint,
calorific value, emissions factors, procurement, and provenance tracking.

All calculators follow zero-hallucination principles with deterministic algorithms.
"""

from .multi_fuel_optimizer import MultiFuelOptimizer
from .cost_optimization_calculator import CostOptimizationCalculator
from .fuel_blending_calculator import FuelBlendingCalculator
from .carbon_footprint_calculator import CarbonFootprintCalculator
from .calorific_value_calculator import CalorificValueCalculator
from .emissions_factor_calculator import EmissionsFactorCalculator
from .procurement_optimizer import ProcurementOptimizer
from .provenance_tracker import ProvenanceTracker

__all__ = [
    'MultiFuelOptimizer',
    'CostOptimizationCalculator',
    'FuelBlendingCalculator',
    'CarbonFootprintCalculator',
    'CalorificValueCalculator',
    'EmissionsFactorCalculator',
    'ProcurementOptimizer',
    'ProvenanceTracker'
]
