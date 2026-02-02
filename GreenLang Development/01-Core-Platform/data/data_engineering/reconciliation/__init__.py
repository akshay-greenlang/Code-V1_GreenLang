"""
Reconciliation Module
====================

Cross-source validation, conflict detection, and factor selection.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from greenlang.data_engineering.reconciliation.factor_reconciliation import (
    FactorReconciler,
    ReconciliationResult,
    ConflictResolutionStrategy,
    SourcePriority,
    FactorConflict,
)

__all__ = [
    "FactorReconciler",
    "ReconciliationResult",
    "ConflictResolutionStrategy",
    "SourcePriority",
    "FactorConflict",
]
