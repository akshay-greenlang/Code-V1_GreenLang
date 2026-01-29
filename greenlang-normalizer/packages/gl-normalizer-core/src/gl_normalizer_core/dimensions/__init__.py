"""
Dimensional Analysis Module for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides comprehensive dimensional analysis capabilities for the
GreenLang Normalizer. It enables:

- Determination of physical dimensions for unit strings
- Compatibility checking between units for conversion
- Simplification of complex dimension expressions
- Context requirement identification for conditional conversions

The module follows GreenLang's zero-hallucination principle by using
deterministic lookups and pure functions for all analysis operations.

Key Components:
    - DimensionalAnalyzer: Main analyzer class for dimensional analysis
    - Dimension: Immutable representation of a physical dimension
    - SimplifiedDimension: Lightweight dimension for API responses
    - Compatibility functions: Pure functions for compatibility checking

Example:
    >>> from gl_normalizer_core.dimensions import (
    ...     DimensionalAnalyzer,
    ...     Dimension,
    ...     are_compatible,
    ... )
    >>> # Using the analyzer
    >>> analyzer = DimensionalAnalyzer()
    >>> dim = analyzer.get_dimension("kWh")
    >>> print(dim)
    [M][L]^2[T]^-2
    >>> # Check compatibility
    >>> analyzer.are_compatible("kWh", "MJ")
    True
    >>> # Direct compatibility check
    >>> are_compatible("kWh", "kg")
    False
    >>> # Create dimensions programmatically
    >>> mass = Dimension.from_name("mass")
    >>> length = Dimension.from_name("length")
    >>> velocity = length / Dimension.from_name("time")
    >>> print(velocity)
    [L][T]^-1
"""

# Core classes
from gl_normalizer_core.dimensions.dimension import (
    Dimension,
    SimplifiedDimension,
    # Pre-instantiated common dimensions
    DIMENSIONLESS,
    LENGTH,
    MASS,
    TIME,
    TEMPERATURE,
    AMOUNT,
    CURRENT,
    LUMINOSITY,
    ENERGY,
    POWER,
    VOLUME,
    AREA,
    PRESSURE,
    DENSITY,
    VELOCITY,
    EMISSIONS,
    VOLUME_FLOW,
    MASS_FLOW,
)

# Analyzer
from gl_normalizer_core.dimensions.analyzer import (
    DimensionalAnalyzer,
    DimensionAnalysisResult,
    # Module-level convenience functions
    get_dimension,
    simplify_dimensions,
)

# Compatibility checking
from gl_normalizer_core.dimensions.compatibility import (
    # Core functions
    check_compatibility,
    are_compatible,
    get_missing_context,
    get_unit_dimension,
    validate_expected_dimension,
    create_compatibility_error,
    # Result types
    CompatibilityResult,
    CompatibilityError,
    MissingContextInfo,
    ContextRequirement,
)

# Constants
from gl_normalizer_core.dimensions.constants import (
    BASE_DIMENSION_SYMBOLS,
    DIMENSION_SYMBOLS,
    DERIVED_DIMENSION_DEFINITIONS,
    GL_CANONICAL_DIMENSIONS,
    DIMENSION_ALIASES,
    CANONICAL_UNIT_PER_DIMENSION,
    UNIT_CATEGORIES,
    CONTEXT_DEPENDENT_UNITS,
)


__all__ = [
    # ==========================================================================
    # Core Classes
    # ==========================================================================
    "Dimension",
    "SimplifiedDimension",
    "DimensionalAnalyzer",
    "DimensionAnalysisResult",
    # ==========================================================================
    # Compatibility Types
    # ==========================================================================
    "CompatibilityResult",
    "CompatibilityError",
    "MissingContextInfo",
    "ContextRequirement",
    # ==========================================================================
    # Functions
    # ==========================================================================
    # Analyzer functions
    "get_dimension",
    "simplify_dimensions",
    # Compatibility functions
    "check_compatibility",
    "are_compatible",
    "get_missing_context",
    "get_unit_dimension",
    "validate_expected_dimension",
    "create_compatibility_error",
    # ==========================================================================
    # Pre-instantiated Dimensions
    # ==========================================================================
    # Base dimensions
    "DIMENSIONLESS",
    "LENGTH",
    "MASS",
    "TIME",
    "TEMPERATURE",
    "AMOUNT",
    "CURRENT",
    "LUMINOSITY",
    # Derived dimensions
    "ENERGY",
    "POWER",
    "VOLUME",
    "AREA",
    "PRESSURE",
    "DENSITY",
    "VELOCITY",
    # GreenLang-specific dimensions
    "EMISSIONS",
    "VOLUME_FLOW",
    "MASS_FLOW",
    # ==========================================================================
    # Constants
    # ==========================================================================
    "BASE_DIMENSION_SYMBOLS",
    "DIMENSION_SYMBOLS",
    "DERIVED_DIMENSION_DEFINITIONS",
    "GL_CANONICAL_DIMENSIONS",
    "DIMENSION_ALIASES",
    "CANONICAL_UNIT_PER_DIMENSION",
    "UNIT_CATEGORIES",
    "CONTEXT_DEPENDENT_UNITS",
]


# Module version
__version__ = "0.1.0"
