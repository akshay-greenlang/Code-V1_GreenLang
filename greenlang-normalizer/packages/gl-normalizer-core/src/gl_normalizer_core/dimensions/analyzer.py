"""
Dimensional Analyzer for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides the DimensionalAnalyzer class for performing dimensional
analysis on unit strings. It leverages the Pint library where possible while
maintaining deterministic behavior and full audit trail support.

Key Features:
    - Pure function design for deterministic analysis
    - Integration with Pint for standard unit parsing
    - GreenLang-specific unit handling (emissions, intensities)
    - Simplified dimension output for API responses
    - Thread-safe caching for performance

Example:
    >>> from gl_normalizer_core.dimensions.analyzer import DimensionalAnalyzer
    >>> analyzer = DimensionalAnalyzer()
    >>> dim = analyzer.get_dimension("kWh")
    >>> print(dim)
    [M][L]^2[T]^-2
    >>> analyzer.are_compatible("kWh", "MJ")
    True
    >>> simplified = analyzer.simplify_dimensions("kg/m3")
    >>> print(simplified.name)
    'density'
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
import pint

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.dimensions.constants import (
    GL_CANONICAL_DIMENSIONS,
    DERIVED_DIMENSION_DEFINITIONS,
    BASE_DIMENSION_SYMBOLS,
    CONTEXT_DEPENDENT_UNITS,
)
from gl_normalizer_core.dimensions.dimension import (
    Dimension,
    SimplifiedDimension,
)
from gl_normalizer_core.dimensions.compatibility import (
    check_compatibility,
    are_compatible as compat_are_compatible,
    get_missing_context,
    CompatibilityResult,
    MissingContextInfo,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Pint Unit Registry Configuration
# =============================================================================

# Create a Pint unit registry for standard unit parsing
# This is module-level to ensure thread-safe singleton behavior
_PINT_REGISTRY: Optional[pint.UnitRegistry] = None


def _get_pint_registry() -> pint.UnitRegistry:
    """
    Get or create the Pint unit registry.

    Returns:
        Configured Pint UnitRegistry instance
    """
    global _PINT_REGISTRY
    if _PINT_REGISTRY is None:
        _PINT_REGISTRY = pint.UnitRegistry()
        # Register GreenLang-specific units
        _register_greenlang_units(_PINT_REGISTRY)
    return _PINT_REGISTRY


def _register_greenlang_units(ureg: pint.UnitRegistry) -> None:
    """
    Register GreenLang-specific units with the Pint registry.

    Args:
        ureg: Pint UnitRegistry to configure
    """
    try:
        # Emissions units (mass-based)
        ureg.define("kgCO2e = kg")
        ureg.define("kgCO2eq = kg")
        ureg.define("tCO2e = metric_ton")
        ureg.define("tCO2eq = metric_ton")
        ureg.define("gCO2e = gram")
        ureg.define("lbCO2e = pound")
        ureg.define("kgCO2 = kg")
        ureg.define("tCO2 = metric_ton")
        ureg.define("kgCH4 = kg")
        ureg.define("kgN2O = kg")

        # Normal/standard volume units
        ureg.define("Nm3 = m^3")
        ureg.define("scf = foot^3")

        # Energy aliases
        ureg.define("therm = 105506000 * joule")
        ureg.define("MMBtu = 1055055852.62 * joule")

        logger.debug("Registered GreenLang-specific units with Pint")
    except pint.errors.RedefinitionError:
        # Units already defined, ignore
        pass


# =============================================================================
# Analysis Result Models
# =============================================================================


class DimensionAnalysisResult(BaseModel):
    """
    Complete result of a dimensional analysis operation.

    Attributes:
        success: Whether the analysis succeeded
        unit_str: The input unit string
        dimension: The analyzed Dimension (if successful)
        simplified: Simplified dimension for API responses
        pint_compatible: Whether Pint was able to parse the unit
        error_code: GLNORM error code if analysis failed
        error_message: Error message if analysis failed
        provenance_hash: SHA-256 hash for audit trail
    """

    success: bool = Field(..., description="Whether the analysis succeeded")
    unit_str: str = Field(..., description="The input unit string")
    dimension: Optional[Dimension] = Field(
        default=None,
        description="The analyzed Dimension object",
    )
    simplified: Optional[SimplifiedDimension] = Field(
        default=None,
        description="Simplified dimension for API responses",
    )
    pint_compatible: bool = Field(
        default=False,
        description="Whether Pint was able to parse the unit",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="GLNORM error code if analysis failed",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if analysis failed",
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")

    model_config = {"frozen": True}


# =============================================================================
# DimensionalAnalyzer Class
# =============================================================================


class DimensionalAnalyzer:
    """
    Analyzer for determining dimensions of unit strings.

    This class provides methods for analyzing unit strings to determine
    their physical dimensions, checking compatibility between units,
    and simplifying complex dimension expressions.

    The analyzer uses Pint for standard unit parsing where possible,
    with fallback to the GreenLang canonical dimension registry for
    sustainability-specific units.

    Attributes:
        use_pint: Whether to use Pint for unit parsing
        cache_enabled: Whether to cache analysis results
        registry_version: Version string for the dimension registry

    Example:
        >>> analyzer = DimensionalAnalyzer()
        >>> dim = analyzer.get_dimension("kWh")
        >>> print(dim)
        [M][L]^2[T]^-2
        >>> analyzer.are_compatible("kWh", "MJ")
        True
    """

    # Registry version for audit trails
    REGISTRY_VERSION = "2026.01.0"

    def __init__(
        self,
        use_pint: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        """
        Initialize the DimensionalAnalyzer.

        Args:
            use_pint: Whether to use Pint for unit parsing (default True)
            cache_enabled: Whether to cache analysis results (default True)
        """
        self.use_pint = use_pint
        self.cache_enabled = cache_enabled
        self._ureg: Optional[pint.UnitRegistry] = None

        logger.info(
            "DimensionalAnalyzer initialized",
            extra={
                "use_pint": use_pint,
                "cache_enabled": cache_enabled,
                "registry_version": self.REGISTRY_VERSION,
            },
        )

    @property
    def ureg(self) -> pint.UnitRegistry:
        """Get the Pint unit registry."""
        if self._ureg is None:
            self._ureg = _get_pint_registry()
        return self._ureg

    # =========================================================================
    # Primary API Methods
    # =========================================================================

    def get_dimension(self, unit_str: str) -> Dimension:
        """
        Get the Dimension for a unit string.

        This is the primary method for dimensional analysis. It attempts
        to determine the dimension using:
        1. GreenLang canonical registry lookup
        2. Pint parsing (if enabled)

        Args:
            unit_str: Unit string to analyze (e.g., "kWh", "kg/m3")

        Returns:
            Dimension object representing the unit's dimension

        Raises:
            ValueError: If dimension cannot be determined

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> dim = analyzer.get_dimension("kWh")
            >>> dim.name
            'energy'
            >>> str(dim)
            '[M][L]^2[T]^-2'
        """
        result = self.analyze(unit_str)
        if not result.success or result.dimension is None:
            raise ValueError(
                f"Cannot determine dimension for unit '{unit_str}': "
                f"{result.error_message}"
            )
        return result.dimension

    def are_compatible(self, unit1: str, unit2: str) -> bool:
        """
        Check if two units are dimensionally compatible.

        Two units are compatible if they have the same dimension,
        meaning they can be converted to each other.

        Args:
            unit1: First unit string
            unit2: Second unit string

        Returns:
            True if units are compatible, False otherwise

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> analyzer.are_compatible("kWh", "MJ")
            True
            >>> analyzer.are_compatible("kWh", "kg")
            False
        """
        return compat_are_compatible(unit1, unit2)

    def simplify_dimensions(self, complex_unit: str) -> SimplifiedDimension:
        """
        Simplify a complex unit string to its dimension representation.

        This method analyzes a potentially complex unit string and
        returns a simplified dimension object suitable for API responses.

        Args:
            complex_unit: Complex unit string (e.g., "kg*m^2/s^2")

        Returns:
            SimplifiedDimension with canonical representation

        Raises:
            ValueError: If dimension cannot be determined

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> simple = analyzer.simplify_dimensions("kg/m3")
            >>> simple.name
            'density'
            >>> simple.display
            '[M][L]^-3'
        """
        result = self.analyze(complex_unit)
        if not result.success or result.simplified is None:
            raise ValueError(
                f"Cannot simplify dimension for unit '{complex_unit}': "
                f"{result.error_message}"
            )
        return result.simplified

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze(self, unit_str: str) -> DimensionAnalysisResult:
        """
        Perform complete dimensional analysis on a unit string.

        This method returns a full analysis result including the dimension,
        simplified representation, and any errors.

        Args:
            unit_str: Unit string to analyze

        Returns:
            DimensionAnalysisResult with complete analysis

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> result = analyzer.analyze("kWh")
            >>> result.success
            True
            >>> result.dimension.name
            'energy'
        """
        if self.cache_enabled:
            return self._analyze_cached(unit_str)
        return self._analyze_impl(unit_str)

    @lru_cache(maxsize=10000)
    def _analyze_cached(self, unit_str: str) -> DimensionAnalysisResult:
        """Cached version of analysis."""
        return self._analyze_impl(unit_str)

    def _analyze_impl(self, unit_str: str) -> DimensionAnalysisResult:
        """
        Implementation of dimensional analysis.

        Args:
            unit_str: Unit string to analyze

        Returns:
            DimensionAnalysisResult
        """
        # Compute provenance hash
        provenance_str = f"{unit_str}|{self.REGISTRY_VERSION}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        # Normalize input
        normalized = unit_str.strip()
        if not normalized:
            return DimensionAnalysisResult(
                success=False,
                unit_str=unit_str,
                error_code=GLNORMErrorCode.E100_UNIT_PARSE_FAILED.value,
                error_message="Empty unit string",
                provenance_hash=provenance_hash,
            )

        # Try GreenLang registry first
        gl_result = self._analyze_from_registry(normalized)
        if gl_result is not None:
            dimension, name = gl_result
            simplified = SimplifiedDimension.from_dimension(dimension)
            return DimensionAnalysisResult(
                success=True,
                unit_str=unit_str,
                dimension=dimension,
                simplified=simplified,
                pint_compatible=False,
                provenance_hash=provenance_hash,
            )

        # Try Pint parsing
        if self.use_pint:
            pint_result = self._analyze_from_pint(normalized)
            if pint_result is not None:
                dimension = pint_result
                simplified = SimplifiedDimension.from_dimension(dimension)
                return DimensionAnalysisResult(
                    success=True,
                    unit_str=unit_str,
                    dimension=dimension,
                    simplified=simplified,
                    pint_compatible=True,
                    provenance_hash=provenance_hash,
                )

        # Analysis failed
        return DimensionAnalysisResult(
            success=False,
            unit_str=unit_str,
            error_code=GLNORMErrorCode.E201_DIMENSION_UNKNOWN.value,
            error_message=f"Cannot determine dimension for unit '{unit_str}'",
            provenance_hash=provenance_hash,
        )

    def _analyze_from_registry(
        self,
        unit_str: str,
    ) -> Optional[Tuple[Dimension, Optional[str]]]:
        """
        Try to analyze dimension from GreenLang registry.

        Args:
            unit_str: Normalized unit string

        Returns:
            Tuple of (Dimension, name) if found, None otherwise
        """
        # Direct lookup
        if unit_str in GL_CANONICAL_DIMENSIONS:
            exponents = dict(GL_CANONICAL_DIMENSIONS[unit_str])
            name = self._find_dimension_name(exponents)
            return Dimension(exponents=exponents, name=name), name

        # Try common variations
        variations = [
            unit_str.replace("^", ""),
            unit_str.replace(" ", ""),
            unit_str.replace("-", ""),
            unit_str.replace("_", ""),
        ]

        for var in variations:
            if var in GL_CANONICAL_DIMENSIONS:
                exponents = dict(GL_CANONICAL_DIMENSIONS[var])
                name = self._find_dimension_name(exponents)
                return Dimension(exponents=exponents, name=name), name

        return None

    def _analyze_from_pint(self, unit_str: str) -> Optional[Dimension]:
        """
        Try to analyze dimension using Pint.

        Args:
            unit_str: Unit string to analyze

        Returns:
            Dimension if successful, None otherwise
        """
        try:
            unit = self.ureg.parse_expression(unit_str)
            pint_dim = unit.dimensionality

            # Convert Pint dimensionality to our format
            exponents = self._convert_pint_dimensionality(pint_dim)
            if exponents is not None:
                name = self._find_dimension_name(exponents)
                return Dimension(exponents=exponents, name=name)

        except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError) as e:
            logger.debug(f"Pint parsing failed for '{unit_str}': {e}")
        except Exception as e:
            logger.warning(f"Unexpected Pint error for '{unit_str}': {e}")

        return None

    def _convert_pint_dimensionality(
        self,
        pint_dim: pint.util.UnitsContainer,
    ) -> Optional[Dict[str, int]]:
        """
        Convert Pint dimensionality to GreenLang format.

        Args:
            pint_dim: Pint UnitsContainer with dimensionality

        Returns:
            Dictionary mapping base dimension names to exponents
        """
        # Mapping from Pint dimension names to GreenLang names
        pint_to_gl = {
            "[length]": "length",
            "[mass]": "mass",
            "[time]": "time",
            "[temperature]": "temperature",
            "[substance]": "amount",
            "[current]": "current",
            "[luminosity]": "luminosity",
        }

        exponents: Dict[str, int] = {}
        for pint_name, exp in pint_dim.items():
            gl_name = pint_to_gl.get(pint_name)
            if gl_name is None:
                # Unknown dimension - cannot convert
                return None
            if exp != 0:
                exponents[gl_name] = int(exp)

        return exponents

    def _find_dimension_name(self, exponents: Dict[str, int]) -> Optional[str]:
        """
        Find the name for a set of dimension exponents.

        Args:
            exponents: Dictionary of base dimensions to exponents

        Returns:
            Dimension name if found, None otherwise
        """
        # Check derived dimensions
        for name, derived_exp in DERIVED_DIMENSION_DEFINITIONS.items():
            if exponents == derived_exp:
                return name

        # Check if it's a single base dimension
        if len(exponents) == 1:
            dim_name = list(exponents.keys())[0]
            if exponents[dim_name] == 1:
                return dim_name

        return None

    # =========================================================================
    # Compatibility Checking Methods
    # =========================================================================

    def check_compatibility(
        self,
        source_unit: str,
        target_unit: str,
    ) -> CompatibilityResult:
        """
        Perform detailed compatibility check between two units.

        This method returns a full compatibility result including
        any missing context requirements.

        Args:
            source_unit: Source unit string
            target_unit: Target unit string

        Returns:
            CompatibilityResult with full details

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> result = analyzer.check_compatibility("kWh", "MJ")
            >>> result.is_compatible
            True
        """
        return check_compatibility(source_unit, target_unit)

    def get_missing_context(
        self,
        source_unit: str,
        target_unit: str,
    ) -> List[MissingContextInfo]:
        """
        Get missing context requirements for a conversion.

        Args:
            source_unit: Source unit string
            target_unit: Target unit string

        Returns:
            List of missing context requirements

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> missing = analyzer.get_missing_context("kgCH4", "kgCO2e")
            >>> len(missing) > 0
            True
        """
        return get_missing_context(source_unit, target_unit)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_registry_version(self) -> str:
        """
        Get the version of the dimension registry.

        Returns:
            Version string (e.g., "2026.01.0")
        """
        return self.REGISTRY_VERSION

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        if hasattr(self._analyze_cached, "cache_clear"):
            self._analyze_cached.cache_clear()
        logger.info("DimensionalAnalyzer cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit/miss statistics
        """
        if hasattr(self._analyze_cached, "cache_info"):
            info = self._analyze_cached.cache_info()
            return {
                "hits": info.hits,
                "misses": info.misses,
                "maxsize": info.maxsize,
                "currsize": info.currsize,
            }
        return {"cache_enabled": False}

    def list_supported_dimensions(self) -> List[str]:
        """
        List all supported dimension names.

        Returns:
            Sorted list of dimension names

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> dims = analyzer.list_supported_dimensions()
            >>> 'energy' in dims
            True
        """
        all_dims = set(BASE_DIMENSION_SYMBOLS)
        all_dims.update(DERIVED_DIMENSION_DEFINITIONS.keys())
        return sorted(all_dims)

    def list_context_dependent_units(self) -> Dict[str, str]:
        """
        List units that require additional context.

        Returns:
            Dictionary mapping unit strings to required context type

        Example:
            >>> analyzer = DimensionalAnalyzer()
            >>> ctx_units = analyzer.list_context_dependent_units()
            >>> ctx_units.get("kgCO2e")
            'gwp_version'
        """
        return dict(CONTEXT_DEPENDENT_UNITS)


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


def get_dimension(unit_str: str) -> Dimension:
    """
    Get the dimension for a unit string.

    This is a module-level convenience function that uses a default
    DimensionalAnalyzer instance.

    Args:
        unit_str: Unit string to analyze

    Returns:
        Dimension object

    Raises:
        ValueError: If dimension cannot be determined

    Example:
        >>> from gl_normalizer_core.dimensions.analyzer import get_dimension
        >>> dim = get_dimension("kWh")
        >>> str(dim)
        '[M][L]^2[T]^-2'
    """
    analyzer = DimensionalAnalyzer()
    return analyzer.get_dimension(unit_str)


def simplify_dimensions(complex_unit: str) -> SimplifiedDimension:
    """
    Simplify a complex unit to its dimension.

    This is a module-level convenience function.

    Args:
        complex_unit: Complex unit string

    Returns:
        SimplifiedDimension

    Example:
        >>> from gl_normalizer_core.dimensions.analyzer import simplify_dimensions
        >>> simple = simplify_dimensions("kg/m3")
        >>> simple.name
        'density'
    """
    analyzer = DimensionalAnalyzer()
    return analyzer.simplify_dimensions(complex_unit)
