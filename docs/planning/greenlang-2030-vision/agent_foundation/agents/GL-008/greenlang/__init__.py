# -*- coding: utf-8 -*-
"""
GreenLang Core Package - Zero-Hallucination Utilities

This package provides deterministic operations for GreenLang agents to ensure
reproducible, auditable calculations in regulatory compliance systems.

Modules:
    determinism: Deterministic clock, UUIDs, and provenance hashing

Example:
    >>> from greenlang.determinism import DeterministicClock, deterministic_uuid
    >>> clock = DeterministicClock(test_mode=True)
    >>> clock.set_time("2024-01-01T00:00:00Z")
    >>> timestamp = clock.now()
    >>> uuid = deterministic_uuid("steam_trap_inspection_123")
"""

from greenlang.determinism import (
    DeterministicClock,
    deterministic_uuid,
    calculate_provenance_hash,
    DeterminismValidator,
)

__version__ = "1.0.0"
__all__ = [
    "DeterministicClock",
    "deterministic_uuid",
    "calculate_provenance_hash",
    "DeterminismValidator",
]
