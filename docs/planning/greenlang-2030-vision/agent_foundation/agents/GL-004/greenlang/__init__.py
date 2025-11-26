# -*- coding: utf-8 -*-
"""GreenLang Core Package for GL-004 BURNMASTER.

This package provides deterministic utilities for zero-hallucination
calculations in the GL-004 Burner Optimization Agent.

Key Components:
- DeterministicClock: Reproducible timestamps for testing
- deterministic_uuid: Reproducible UUID generation
- calculate_provenance_hash: SHA-256 hashing for audit trails

Author: GreenLang AI Agent Factory
License: Proprietary
Version: 1.0.0
"""

from .determinism import (
    DeterministicClock,
    deterministic_uuid,
    calculate_provenance_hash,
)

__all__ = [
    'DeterministicClock',
    'deterministic_uuid',
    'calculate_provenance_hash',
]

__version__ = '1.0.0'
