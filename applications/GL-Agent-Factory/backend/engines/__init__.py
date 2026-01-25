"""
GreenLang Zero-Hallucination Calculation Engines

This package provides deterministic, bit-perfect calculation engines for
regulatory compliance and climate intelligence. All calculations are:

- Deterministic: Same input always produces same output
- Reproducible: Complete provenance tracking with SHA-256 hashes
- Auditable: Full audit trail for regulatory compliance
- Zero-Hallucination: NO LLM in calculation path

NEW in v1.1.0: SHAP/LIME Explainability Engine
- Feature importance analysis (SHAP values)
- Local explanations (LIME)
- Natural language summary generation
- All explanations are for TRANSPARENCY - calculations remain deterministic

Modules:
    base_calculator: Abstract base class and provenance tracking
    unit_converter: Energy, mass, volume, distance, area conversions
    emission_factors: Emission factor database with 100,000+ factors
    explainability: SHAP/LIME explainability for AI/ML transparency
    scope1_calculator: Scope 1 emissions (stationary, mobile, fugitive, process)
    scope2_calculator: Scope 2 emissions (location-based, market-based)
    scope3_calculator: Scope 3 emissions (all 15 categories)
    cbam_calculator: CBAM embedded emissions calculations
    building_calculator: Building energy and CRREM pathway calculations

Example:
    >>> from engines import Scope1Calculator, EmissionFactorDB
    >>> ef_db = EmissionFactorDB()
    >>> calc = Scope1Calculator(ef_db)
    >>> result = calc.stationary_combustion(
    ...     fuel_type="natural_gas",
    ...     quantity=1000,
    ...     unit="m3",
    ...     region="US"
    ... )
    >>> print(f"Emissions: {result.value} {result.unit}")
    >>> print(f"Provenance: {result.provenance_hash}")

Explainability Example:
    >>> from engines import ExplainabilityEngine
    >>> engine = ExplainabilityEngine()
    >>> report = engine.explain_health_score(
    ...     operating_hours=15000,
    ...     design_life=50000,
    ...     flame_quality=85,
    ...     cycles_factor=0.9,
    ...     age_factor=0.8,
    ...     calculated_health_score=72.5
    ... )
    >>> print(report.natural_language_summary)
"""

from .base_calculator import (
    BaseCalculator,
    CalculationResult,
    CalculationStep,
    ProvenanceMixin,
    RoundingRule,
)
from .unit_converter import UnitConverter, UnitCategory
from .emission_factors import EmissionFactorDB, EmissionFactor, EmissionFactorSource

# Import explainability components
from .explainability import (
    ExplainabilityEngine,
    ExplainabilityMixin,
    ExplainabilityReport,
    SHAPValues,
    LIMEExplanation,
    FeatureContribution,
    UncertaintyQuantification,
    NaturalLanguageGenerator,
    KernelSHAPExplainer,
    LIMEExplainer,
    ExplanationType,
    ConfidenceLevel,
    MINIMUM_CONFIDENCE_THRESHOLD,
)

__all__ = [
    # Base calculator components
    "BaseCalculator",
    "CalculationResult",
    "CalculationStep",
    "ProvenanceMixin",
    "RoundingRule",
    # Unit conversion
    "UnitConverter",
    "UnitCategory",
    # Emission factors
    "EmissionFactorDB",
    "EmissionFactor",
    "EmissionFactorSource",
    # Explainability components
    "ExplainabilityEngine",
    "ExplainabilityMixin",
    "ExplainabilityReport",
    "SHAPValues",
    "LIMEExplanation",
    "FeatureContribution",
    "UncertaintyQuantification",
    "NaturalLanguageGenerator",
    "KernelSHAPExplainer",
    "LIMEExplainer",
    "ExplanationType",
    "ConfidenceLevel",
    "MINIMUM_CONFIDENCE_THRESHOLD",
]

__version__ = "1.1.0"
