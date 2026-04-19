# -*- coding: utf-8 -*-
"""
Climate Ledger - Calculation Provenance
=========================================

Re-exports calculation-level provenance tracking from
``greenlang.execution.core.provenance`` for the v3 Climate Ledger
product surface.

These classes enable step-by-step recording of deterministic
calculations with SHA-256 integrity hashes, satisfying CSRD, CBAM,
ISO 14064, and GHG Protocol audit requirements.

Exported symbols:

- ``CalculationStep`` -- a single discrete calculation operation
- ``CalculationProvenance`` -- complete audit trail for a calculation
- ``ProvenanceMetadata`` -- agent/calculation metadata
- ``OperationType`` -- standard operation type enum
- ``stable_hash`` -- deterministic SHA-256 hashing for any data
- ``ProvenanceStorage`` -- protocol for persistent storage backends
- ``SQLiteProvenanceStorage`` -- lightweight file-based storage

Example::

    >>> from greenlang.climate_ledger.calculation import (
    ...     CalculationProvenance, OperationType,
    ... )
    >>> prov = CalculationProvenance.create(
    ...     agent_name="EmissionsCalc",
    ...     agent_version="1.0.0",
    ...     calculation_type="scope1_emissions",
    ...     input_data={"fuel_kg": 1000, "fuel_type": "natural_gas"},
    ... )
    >>> prov.add_step(
    ...     operation=OperationType.LOOKUP,
    ...     description="Lookup emission factor",
    ...     inputs={"fuel_type": "natural_gas"},
    ...     output=0.18414,
    ...     data_source="EPA eGRID 2023",
    ... )
    >>> prov.finalize(output_data={"total_kg_co2e": 184.14})

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

# Re-exports from the canonical calculation provenance leaf modules.
# We import from the subpackage __init__ directly (not from
# greenlang.execution.core) to avoid the heavy orchestrator import chain
# in greenlang.execution.core.__init__.
from greenlang.execution.core.provenance.calculation_provenance import (
    CalculationProvenance,
    CalculationStep,
    OperationType,
    ProvenanceMetadata,
    stable_hash,
)
from greenlang.execution.core.provenance.storage import (
    ProvenanceStorage,
    SQLiteProvenanceStorage,
)

__all__ = [
    "CalculationStep",
    "CalculationProvenance",
    "ProvenanceMetadata",
    "OperationType",
    "stable_hash",
    "ProvenanceStorage",
    "SQLiteProvenanceStorage",
]
