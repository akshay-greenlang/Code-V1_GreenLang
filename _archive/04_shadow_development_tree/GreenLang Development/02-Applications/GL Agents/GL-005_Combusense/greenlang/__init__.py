"""
GreenLang Core Module - Determinism Utilities

This module provides deterministic utilities for GreenLang agents to ensure
zero-hallucination, reproducible operations in regulatory compliance systems.

Modules:
    determinism: Deterministic timestamp, UUID generation, and provenance hashing
"""

from .determinism import (
    DeterministicClock,
    deterministic_uuid,
    calculate_provenance_hash,
    DeterminismValidator,
)

__all__ = [
    "DeterministicClock",
    "deterministic_uuid",
    "calculate_provenance_hash",
    "DeterminismValidator",
]

__version__ = "1.0.0"
