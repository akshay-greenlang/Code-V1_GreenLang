# -*- coding: utf-8 -*-
"""
GreenLang Determinism Package for GL-010 EMISSIONWATCH.

This package provides deterministic utilities for zero-hallucination
calculations including deterministic clocks, UUID generation, and
provenance tracking for emissions compliance monitoring.

Example:
    >>> from greenlang import DeterministicClock, deterministic_uuid
    >>> clock = DeterministicClock(test_mode=True)
    >>> clock.set_time("2024-01-01T00:00:00Z")
    >>> uuid = deterministic_uuid("emissions_calculation_001")
"""

from .determinism import (
    DeterministicClock,
    DeterminismValidator,
    deterministic_uuid,
    calculate_provenance_hash,
    create_emissions_uuid,
    create_audit_hash,
    validate_deterministic_result
)

__all__ = [
    "DeterministicClock",
    "DeterminismValidator",
    "deterministic_uuid",
    "calculate_provenance_hash",
    "create_emissions_uuid",
    "create_audit_hash",
    "validate_deterministic_result"
]
