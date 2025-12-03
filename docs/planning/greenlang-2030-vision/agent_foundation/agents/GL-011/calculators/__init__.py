# -*- coding: utf-8 -*-
"""
Calculators module for GL-011 FUELCRAFT agent.

This module provides deterministic calculators for fuel management operations
including multi-fuel optimization, cost optimization, blending, carbon footprint,
calorific value, emissions factors, procurement, and provenance tracking.

Advanced calculators include:
- FuelBlendingOptimizer: LP-based multi-fuel optimization with Refutas viscosity
- FuelQualityAnalyzer: Complete fuel quality analysis with ASTM standards

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

# Advanced calculators
from .fuel_blending_optimizer import (
    FuelBlendingOptimizer,
    FuelComponent,
    BlendSpecification,
    OptimizationObjective,
    BlendOptimizationInput,
    BlendOptimizationResult
)
from .fuel_quality_analyzer import (
    FuelQualityAnalyzer,
    ProximateAnalysis,
    UltimateAnalysis,
    AshComposition,
    FuelSampleInput,
    FuelQualityResult,
    HeatingValueResult,
    SlaggingFoulingResult,
    EmissionFactorResult,
    QualityDeviationAlert,
    SamplingRecommendation,
    FuelType,
    AnalysisBasis,
    QualityGrade,
    SlaggingRisk
)

__all__ = [
    # Core calculators
    'MultiFuelOptimizer',
    'CostOptimizationCalculator',
    'FuelBlendingCalculator',
    'CarbonFootprintCalculator',
    'CalorificValueCalculator',
    'EmissionsFactorCalculator',
    'ProcurementOptimizer',
    'ProvenanceTracker',
    # Advanced blending optimizer
    'FuelBlendingOptimizer',
    'FuelComponent',
    'BlendSpecification',
    'OptimizationObjective',
    'BlendOptimizationInput',
    'BlendOptimizationResult',
    # Advanced quality analyzer
    'FuelQualityAnalyzer',
    'ProximateAnalysis',
    'UltimateAnalysis',
    'AshComposition',
    'FuelSampleInput',
    'FuelQualityResult',
    'HeatingValueResult',
    'SlaggingFoulingResult',
    'EmissionFactorResult',
    'QualityDeviationAlert',
    'SamplingRecommendation',
    'FuelType',
    'AnalysisBasis',
    'QualityGrade',
    'SlaggingRisk'
]
