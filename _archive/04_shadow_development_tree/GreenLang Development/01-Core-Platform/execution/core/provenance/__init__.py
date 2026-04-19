# -*- coding: utf-8 -*-
"""
GreenLang Core Provenance - Calculation Provenance Tracking

Standardized provenance tracking for all GreenLang calculators based on
CSRD, CBAM, and GL-001 through GL-010 best practices.

This module provides zero-hallucination calculation tracking with complete
audit trails, step-by-step recording, and SHA-256 hash verification.
"""

from .calculation_provenance import (
    CalculationStep,
    CalculationProvenance,
    ProvenanceMetadata,
    OperationType,
    stable_hash,
)
from .storage import (
    ProvenanceStorage,
    SQLiteProvenanceStorage,
)

__all__ = [
    # Core Models
    'CalculationStep',
    'CalculationProvenance',
    'ProvenanceMetadata',
    'OperationType',

    # Utilities
    'stable_hash',

    # Storage
    'ProvenanceStorage',
    'SQLiteProvenanceStorage',
]

__version__ = "1.0.0"
