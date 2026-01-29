"""
Dimension Class for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides the core Dimension class for representing and manipulating
physical dimensions. Dimensions are immutable value objects that support
arithmetic operations (multiplication, division, power) while maintaining
full determinism for audit trails.

The Dimension class follows the SI base dimension model with extensions
for GreenLang-specific sustainability dimensions (emissions, energy intensity).

Key Design Principles:
    - Immutability: All Dimension instances are frozen after creation
    - Pure functions: All operations return new instances
    - Deterministic: Identical inputs always produce identical outputs
    - Human-readable: String representations follow standard notation

Example:
    >>> from gl_normalizer_core.dimensions.dimension import Dimension
    >>> mass = Dimension.from_name("mass")
    >>> length = Dimension.from_name("length")
    >>> time = Dimension.from_name("time")
    >>> # Create derived dimension: velocity = length / time
    >>> velocity = length / time
    >>> print(velocity)
    [L][T]^-1
    >>> # Create energy = mass * length^2 / time^2
    >>> energy = mass * (length ** 2) / (time ** 2)
    >>> print(energy)
    [M][L]^2[T]^-2
"""

from __future__ import annotations

import hashlib
from functools import cached_property
from typing import Any, Dict, FrozenSet, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from gl_normalizer_core.dimensions.constants import (
    BASE_DIMENSION_SYMBOLS,
    DERIVED_DIMENSION_DEFINITIONS,
    DIMENSION_SYMBOLS,
    DIMENSION_ALIASES,
)


class Dimension(BaseModel):
    """
    Immutable representation of a physical dimension.

    A dimension is represented as a dictionary mapping base dimension names
    to their exponents. For example:
    - Mass: {"mass": 1}
    - Energy: {"mass": 1, "length": 2, "time": -2}
    - Dimensionless: {}

    Attributes:
        exponents: Frozen dictionary mapping base dimension names to exponents.
        name: Optional human-readable name for the dimension.

    Example:
        >>> mass = Dimension(exponents={"mass": 1})
        >>> print(mass)
        [M]
        >>> energy = Dimension(exponents={"mass": 1, "length": 2, "time": -2})
        >>> print(energy)
        [M][L]^2[T]^-2
    """

    exponents: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of base dimension names to their exponents",
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional human-readable name for the dimension",
    )

    model_config = {"frozen": True}

    @field_validator("exponents", mode="before")
    @classmethod
    def normalize_exponents(cls, v: Dict[str, int]) -> Dict[str, int]:
        """
        Normalize exponents by removing zero entries and validating keys.

        Args:
            v: Raw exponents dictionary

        Returns:
            Normalized dictionary with zero exponents removed

        Raises:
            ValueError: If any key is not a valid base dimension
        """
        if not isinstance(v, dict):
            raise ValueError(f"exponents must be a dict, got {type(v)}")

        normalized = {}
        for key, exp in v.items():
            # Validate dimension name
            if key not in BASE_DIMENSION_SYMBOLS:
                raise ValueError(
                    f"Invalid base dimension '{key}'. "
                    f"Valid dimensions: {sorted(BASE_DIMENSION_SYMBOLS)}"
                )
            # Only keep non-zero exponents
            if exp != 0:
                normalized[key] = exp

        return normalized

    @model_validator(mode="after")
    def validate_dimension(self) -> "Dimension":
        """Validate the complete dimension object."""
        # Ensure exponents are integers
        for key, exp in self.exponents.items():
            if not isinstance(exp, int):
                raise ValueError(
                    f"Exponent for dimension '{key}' must be an integer, "
                    f"got {type(exp).__name__}: {exp}"
                )
        return self

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_name(cls, name: str) -> "Dimension":
        """
        Create a Dimension from a dimension name.

        Supports both base dimensions (mass, length, etc.) and derived
        dimensions (energy, power, etc.) from the predefined registry.

        Args:
            name: Dimension name (e.g., "mass", "energy", "pressure")

        Returns:
            Dimension instance

        Raises:
            ValueError: If dimension name is not recognized

        Example:
            >>> energy = Dimension.from_name("energy")
            >>> print(energy.exponents)
            {'mass': 1, 'length': 2, 'time': -2}
        """
        # Resolve aliases
        canonical_name = DIMENSION_ALIASES.get(name.lower(), name.lower())

        # Check base dimensions
        if canonical_name in BASE_DIMENSION_SYMBOLS:
            return cls(exponents={canonical_name: 1}, name=canonical_name)

        # Check derived dimensions
        if canonical_name in DERIVED_DIMENSION_DEFINITIONS:
            exponents = dict(DERIVED_DIMENSION_DEFINITIONS[canonical_name])
            return cls(exponents=exponents, name=canonical_name)

        raise ValueError(
            f"Unknown dimension name '{name}'. "
            f"Valid base dimensions: {sorted(BASE_DIMENSION_SYMBOLS)}. "
            f"Valid derived dimensions: {sorted(DERIVED_DIMENSION_DEFINITIONS.keys())}."
        )

    @classmethod
    def from_exponents(
        cls,
        exponents: Dict[str, int],
        name: Optional[str] = None,
    ) -> "Dimension":
        """
        Create a Dimension from an exponents dictionary.

        Args:
            exponents: Dictionary mapping base dimensions to exponents
            name: Optional name for the dimension

        Returns:
            Dimension instance

        Example:
            >>> velocity = Dimension.from_exponents({"length": 1, "time": -1})
            >>> print(velocity)
            [L][T]^-1
        """
        return cls(exponents=exponents, name=name)

    @classmethod
    def dimensionless(cls) -> "Dimension":
        """
        Create a dimensionless Dimension (empty exponents).

        Returns:
            Dimensionless Dimension instance

        Example:
            >>> d = Dimension.dimensionless()
            >>> d.is_dimensionless
            True
        """
        return cls(exponents={}, name="dimensionless")

    # =========================================================================
    # Properties
    # =========================================================================

    @cached_property
    def is_dimensionless(self) -> bool:
        """
        Check if this dimension is dimensionless.

        Returns:
            True if all exponents are zero (or empty)

        Example:
            >>> Dimension(exponents={}).is_dimensionless
            True
            >>> Dimension(exponents={"mass": 1}).is_dimensionless
            False
        """
        return len(self.exponents) == 0

    @cached_property
    def is_base(self) -> bool:
        """
        Check if this is a base dimension (single dimension with exponent 1).

        Returns:
            True if this is a base dimension

        Example:
            >>> Dimension(exponents={"mass": 1}).is_base
            True
            >>> Dimension(exponents={"mass": 1, "length": 2}).is_base
            False
        """
        if len(self.exponents) != 1:
            return False
        return list(self.exponents.values())[0] == 1

    @cached_property
    def base_dimensions(self) -> FrozenSet[str]:
        """
        Get the set of base dimensions with non-zero exponents.

        Returns:
            Frozen set of base dimension names

        Example:
            >>> energy = Dimension.from_name("energy")
            >>> energy.base_dimensions
            frozenset({'mass', 'length', 'time'})
        """
        return frozenset(self.exponents.keys())

    @cached_property
    def dimension_signature(self) -> str:
        """
        Get a canonical string signature for this dimension.

        The signature is deterministic and can be used for equality
        comparison and hashing. Format: "dim1:exp1,dim2:exp2,..."
        with dimensions sorted alphabetically.

        Returns:
            Canonical dimension signature string

        Example:
            >>> energy = Dimension.from_name("energy")
            >>> energy.dimension_signature
            'length:2,mass:1,time:-2'
        """
        if self.is_dimensionless:
            return "dimensionless"

        parts = [f"{dim}:{exp}" for dim, exp in sorted(self.exponents.items())]
        return ",".join(parts)

    @cached_property
    def provenance_hash(self) -> str:
        """
        Get SHA-256 hash for audit trail provenance.

        Returns:
            Hex-encoded SHA-256 hash of the dimension signature
        """
        return hashlib.sha256(self.dimension_signature.encode()).hexdigest()

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def __mul__(self, other: "Dimension") -> "Dimension":
        """
        Multiply two dimensions (add exponents).

        Args:
            other: Another Dimension instance

        Returns:
            New Dimension with combined exponents

        Example:
            >>> mass = Dimension.from_name("mass")
            >>> velocity = Dimension.from_exponents({"length": 1, "time": -1})
            >>> momentum = mass * velocity
            >>> print(momentum)
            [M][L][T]^-1
        """
        if not isinstance(other, Dimension):
            return NotImplemented

        new_exponents: Dict[str, int] = dict(self.exponents)
        for dim, exp in other.exponents.items():
            new_exponents[dim] = new_exponents.get(dim, 0) + exp
            if new_exponents[dim] == 0:
                del new_exponents[dim]

        return Dimension(exponents=new_exponents)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        """
        Divide two dimensions (subtract exponents).

        Args:
            other: Another Dimension instance

        Returns:
            New Dimension with divided exponents

        Example:
            >>> length = Dimension.from_name("length")
            >>> time = Dimension.from_name("time")
            >>> velocity = length / time
            >>> print(velocity)
            [L][T]^-1
        """
        if not isinstance(other, Dimension):
            return NotImplemented

        new_exponents: Dict[str, int] = dict(self.exponents)
        for dim, exp in other.exponents.items():
            new_exponents[dim] = new_exponents.get(dim, 0) - exp
            if new_exponents[dim] == 0:
                del new_exponents[dim]

        return Dimension(exponents=new_exponents)

    def __pow__(self, power: int) -> "Dimension":
        """
        Raise dimension to an integer power (multiply all exponents).

        Args:
            power: Integer power to raise to

        Returns:
            New Dimension with scaled exponents

        Raises:
            TypeError: If power is not an integer

        Example:
            >>> length = Dimension.from_name("length")
            >>> area = length ** 2
            >>> print(area)
            [L]^2
        """
        if not isinstance(power, int):
            raise TypeError(f"Power must be an integer, got {type(power).__name__}")

        if power == 0:
            return Dimension.dimensionless()

        new_exponents = {dim: exp * power for dim, exp in self.exponents.items()}
        return Dimension(exponents=new_exponents)

    def inverse(self) -> "Dimension":
        """
        Get the multiplicative inverse (negate all exponents).

        Returns:
            New Dimension with negated exponents

        Example:
            >>> time = Dimension.from_name("time")
            >>> frequency = time.inverse()
            >>> print(frequency)
            [T]^-1
        """
        new_exponents = {dim: -exp for dim, exp in self.exponents.items()}
        return Dimension(exponents=new_exponents)

    # =========================================================================
    # Comparison Operations
    # =========================================================================

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on exponents.

        Two dimensions are equal if they have identical exponents.
        """
        if not isinstance(other, Dimension):
            return NotImplemented
        return self.exponents == other.exponents

    def __hash__(self) -> int:
        """Hash based on frozen exponents."""
        return hash(self.dimension_signature)

    def is_compatible_with(self, other: "Dimension") -> bool:
        """
        Check if this dimension is compatible with another for conversion.

        Dimensions are compatible if they have identical exponents.

        Args:
            other: Another Dimension instance

        Returns:
            True if dimensions are compatible

        Example:
            >>> energy1 = Dimension.from_name("energy")
            >>> energy2 = Dimension.from_exponents({"mass": 1, "length": 2, "time": -2})
            >>> energy1.is_compatible_with(energy2)
            True
        """
        return self.exponents == other.exponents

    # =========================================================================
    # String Representations
    # =========================================================================

    def __str__(self) -> str:
        """
        Human-readable string representation.

        Format: [M]^n[L]^m[T]^p... with sorted dimensions.
        Exponent 1 is omitted.

        Example:
            >>> energy = Dimension.from_name("energy")
            >>> str(energy)
            '[M][L]^2[T]^-2'
        """
        if self.is_dimensionless:
            return "dimensionless"

        parts = []
        for dim in sorted(self.exponents.keys()):
            symbol = DIMENSION_SYMBOLS.get(dim, dim[0].upper())
            exp = self.exponents[dim]
            if exp == 1:
                parts.append(f"[{symbol}]")
            else:
                parts.append(f"[{symbol}]^{exp}")

        return "".join(parts)

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Dimension(exponents={self.exponents}, name={self.name!r})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            >>> energy = Dimension.from_name("energy")
            >>> energy.to_dict()
            {'exponents': {'mass': 1, 'length': 2, 'time': -2}, 'name': 'energy', ...}
        """
        return {
            "exponents": dict(self.exponents),
            "name": self.name,
            "signature": self.dimension_signature,
            "display": str(self),
            "is_dimensionless": self.is_dimensionless,
            "is_base": self.is_base,
        }


class SimplifiedDimension(BaseModel):
    """
    Simplified dimension representation for API responses.

    This is a lighter-weight representation of a dimension suitable
    for API responses and audit trails.

    Attributes:
        signature: Canonical dimension signature string
        display: Human-readable display string
        name: Optional dimension name
        exponents: Dictionary of base dimensions to exponents
        is_dimensionless: Whether the dimension is dimensionless
    """

    signature: str = Field(..., description="Canonical dimension signature")
    display: str = Field(..., description="Human-readable display string")
    name: Optional[str] = Field(default=None, description="Optional dimension name")
    exponents: Dict[str, int] = Field(..., description="Base dimension exponents")
    is_dimensionless: bool = Field(..., description="Whether dimensionless")

    model_config = {"frozen": True}

    @classmethod
    def from_dimension(cls, dim: Dimension) -> "SimplifiedDimension":
        """
        Create a SimplifiedDimension from a Dimension.

        Args:
            dim: Source Dimension instance

        Returns:
            SimplifiedDimension instance
        """
        return cls(
            signature=dim.dimension_signature,
            display=str(dim),
            name=dim.name,
            exponents=dict(dim.exponents),
            is_dimensionless=dim.is_dimensionless,
        )

    def to_dimension(self) -> Dimension:
        """
        Convert back to a full Dimension instance.

        Returns:
            Dimension instance
        """
        return Dimension(exponents=self.exponents, name=self.name)


# =============================================================================
# Module-Level Constants for Common Dimensions
# =============================================================================
# Pre-instantiated dimension objects for common use cases

DIMENSIONLESS = Dimension.dimensionless()
LENGTH = Dimension.from_name("length")
MASS = Dimension.from_name("mass")
TIME = Dimension.from_name("time")
TEMPERATURE = Dimension.from_name("temperature")
AMOUNT = Dimension.from_name("amount")
CURRENT = Dimension.from_name("current")
LUMINOSITY = Dimension.from_name("luminosity")

# Derived dimensions
ENERGY = Dimension.from_name("energy")
POWER = Dimension.from_name("power")
VOLUME = Dimension.from_name("volume")
AREA = Dimension.from_name("area")
PRESSURE = Dimension.from_name("pressure")
DENSITY = Dimension.from_name("density")
VELOCITY = Dimension.from_name("velocity")

# GreenLang-specific dimensions
EMISSIONS = Dimension.from_name("emissions")
VOLUME_FLOW = Dimension.from_name("volume_flow")
MASS_FLOW = Dimension.from_name("mass_flow")
