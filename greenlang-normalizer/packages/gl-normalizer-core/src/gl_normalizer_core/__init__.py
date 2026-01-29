"""
GL-FOUND-X-003: GreenLang Unit & Reference Normalizer - Core Library

This module provides the foundational components for unit conversion,
reference data resolution, and vocabulary management in the GreenLang
sustainability platform.

Example:
    >>> from gl_normalizer_core import UnitParser, UnitConverter
    >>> parser = UnitParser()
    >>> quantity = parser.parse("100 kg CO2e")
    >>> converter = UnitConverter()
    >>> result = converter.convert(quantity, "t CO2e")
    >>> print(result.magnitude)
    0.1
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gl-normalizer-core")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Core types
from gl_normalizer_core.parser import UnitParser, Quantity, ParseResult
from gl_normalizer_core.converter import UnitConverter, ConversionResult
from gl_normalizer_core.resolver import ReferenceResolver, ResolvedReference, MatchConfidence
from gl_normalizer_core.dimension import DimensionAnalyzer, Dimension
from gl_normalizer_core.policy import ConversionPolicy, PolicyEngine, ComplianceProfile
from gl_normalizer_core.audit import AuditTrail, ProvenanceRecord, AuditLogger
from gl_normalizer_core.vocab import (
    VocabularyManager,
    Vocabulary,
    VocabEntry,
    VocabVersion,
)
from gl_normalizer_core.errors import (
    NormalizerError,
    ParseError,
    ConversionError,
    ResolutionError,
    VocabularyError,
    PolicyViolationError,
    DimensionMismatchError,
)

__all__ = [
    # Version
    "__version__",
    # Parser
    "UnitParser",
    "Quantity",
    "ParseResult",
    # Converter
    "UnitConverter",
    "ConversionResult",
    # Resolver
    "ReferenceResolver",
    "ResolvedReference",
    "MatchConfidence",
    # Dimension
    "DimensionAnalyzer",
    "Dimension",
    # Policy
    "ConversionPolicy",
    "PolicyEngine",
    "ComplianceProfile",
    # Audit
    "AuditTrail",
    "ProvenanceRecord",
    "AuditLogger",
    # Vocabulary
    "VocabularyManager",
    "Vocabulary",
    "VocabEntry",
    "VocabVersion",
    # Errors
    "NormalizerError",
    "ParseError",
    "ConversionError",
    "ResolutionError",
    "VocabularyError",
    "PolicyViolationError",
    "DimensionMismatchError",
]
