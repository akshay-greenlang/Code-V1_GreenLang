# -*- coding: utf-8 -*-
"""
GreenLang Climate Ledger - v3 System of Record
================================================

The Climate Ledger is GreenLang's L2 "System of Record" layer, providing a
unified API for immutable audit trails, chain-hashed provenance, and
calculation reproducibility.

This module is a **product facade** that re-exports and lightly wraps the
existing provenance infrastructure scattered across:

- ``greenlang.data_commons.provenance`` -- SHA-256 chain-hashing tracker
- ``greenlang.utilities.provenance`` -- Full provenance framework
- ``greenlang.execution.core.provenance`` -- Calculation step tracking
- ``greenlang.execution.infrastructure.provenance`` -- Infrastructure lineage
- ``greenlang.utilities.provenance.ledger`` -- JSONL run ledger writer

All numeric operations use deterministic SHA-256 hashing with zero-hallucination
guarantees. No LLM calls are made in any hashing or verification path.

Example::

    >>> from greenlang.climate_ledger import ClimateLedger, hash_data, MerkleTree
    >>> ledger = ClimateLedger(agent_name="scope1-calc")
    >>> entry_hash = ledger.record_entry("emission", "e-001", "calculate", "abc123")
    >>> valid, chain = ledger.verify("e-001")
    >>> assert valid is True

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

# -- Core ledger facade ---------------------------------------------------
from greenlang.climate_ledger.ledger import ClimateLedger

# -- Hashing utilities ----------------------------------------------------
from greenlang.climate_ledger.hashing import (
    content_address,
    hash_data,
    hash_file,
    MerkleTree,
)

# -- Signing utilities -----------------------------------------------------
from greenlang.climate_ledger.signing import (
    sign_artifact,
    verify_artifact,
)

# -- Calculation provenance ------------------------------------------------
from greenlang.climate_ledger.calculation import (
    CalculationProvenance,
    CalculationStep,
    OperationType,
    ProvenanceMetadata,
    ProvenanceStorage,
    SQLiteProvenanceStorage,
    stable_hash,
)

__version__ = "0.1.0"

__all__ = [
    # Core ledger
    "ClimateLedger",
    # Hashing
    "content_address",
    "hash_data",
    "hash_file",
    "MerkleTree",
    # Signing
    "sign_artifact",
    "verify_artifact",
    # Calculation provenance
    "CalculationProvenance",
    "CalculationStep",
    "OperationType",
    "ProvenanceMetadata",
    "ProvenanceStorage",
    "SQLiteProvenanceStorage",
    "stable_hash",
]
