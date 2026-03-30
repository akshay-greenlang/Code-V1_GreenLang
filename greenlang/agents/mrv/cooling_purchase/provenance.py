# -*- coding: utf-8 -*-
"""
Provenance Tracking for Cooling Purchase Agent - AGENT-MRV-012

Provides SHA-256 based audit trail tracking for all Cooling Purchase Agent
operations. Implements a chain of SHA-256 hashes for each calculation stage,
ensuring complete audit trail and deterministic reproducibility across cooling
purchase calculations, electric chiller COP/IPLV lookups, absorption chiller
heat-source factor retrieval, free cooling economizer calculations, thermal
energy storage (TES) temporal shifting, district cooling distribution loss
adjustments, refrigerant leakage (informational), per-gas GHG breakdowns,
GWP conversions, uncertainty quantification, compliance checks, and
provenance sealing.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256 via hashlib
    - Chain hashing links operations in sequence (append-only)
    - Each entry records previous_hash for tamper detection
    - JSON canonical form with sort_keys=True for reproducibility
    - Decimal values converted to string for hashing consistency
    - Complete provenance for every Cooling Purchase operation

Provenance Stages (19):
    1.  INPUT_VALIDATION             - Validate cooling request parameters
    2.  TECHNOLOGY_LOOKUP            - Look up COP/IPLV from cooling database
    3.  EFFICIENCY_CONVERSION        - Convert between COP/EER/kW_per_ton/SEER
    4.  PART_LOAD_CALCULATION        - Calculate IPLV/NPLV from part-load COPs
    5.  ENERGY_INPUT_CALCULATION     - Calculate electrical/thermal input from COP
    6.  AUXILIARY_ENERGY             - Calculate auxiliary/parasitic energy
    7.  GRID_FACTOR_APPLICATION      - Apply grid emission factor to electricity
    8.  HEAT_SOURCE_FACTOR_APPLICATION - Apply heat source EF (absorption)
    9.  GAS_DECOMPOSITION            - Decompose CO2e into CO2, CH4, N2O
    10. GWP_APPLICATION              - Apply GWP factors
    11. FREE_COOLING_CALCULATION     - Calculate free cooling pump energy
    12. TES_TEMPORAL_SHIFTING        - Calculate TES charge/discharge emissions
    13. DISTRICT_LOSS_ADJUSTMENT     - Adjust for distribution network losses
    14. REFRIGERANT_LEAKAGE          - Calculate refrigerant leakage (informational)
    15. UNCERTAINTY_QUANTIFICATION   - Run uncertainty analysis
    16. COMPLIANCE_CHECK             - Check regulatory compliance
    17. AGGREGATION                  - Aggregate results
    18. BATCH_ASSEMBLY               - Assemble batch results
    19. RESULT_FINALIZATION          - Finalize and hash result

Chain Architecture:
    The provenance chain is an ordered, append-only list of frozen
    ProvenanceEntry dataclass instances. Each entry stores the SHA-256
    hash of the previous entry (empty string for the first entry).
    This chain structure allows:

    1. **Tamper Detection**: Any modification to an intermediate entry
       invalidates all subsequent hashes in the chain.
    2. **Deterministic Replay**: Given identical inputs and timestamps,
       the chain produces identical hashes.
    3. **Audit Trail**: External auditors can verify the complete
       calculation lineage from input through output.
    4. **Batch Merging**: Two independent chains can be merged for
       batch processing while preserving individual chain integrity.

Hashing Protocol:
    Each hash is computed as:
        payload = f"{previous_hash}|{stage}|{canonical_json}"
        hash_value = SHA-256(payload.encode("utf-8")).hexdigest()

    Where:
        - previous_hash: The hash_value of the preceding ProvenanceEntry
          (empty string "" for the first entry in the chain).
        - stage: A string label identifying the calculation stage.
        - canonical_json: json.dumps(data, sort_keys=True, default=str)
          ensuring deterministic serialization.

Supported Regulatory Frameworks:
    - GHG Protocol Scope 2 Guidance (2015) - Cooling Purchase
    - GHG Protocol Corporate Standard (Ch. 7)
    - ISO 14064-1:2018 (Category 2)
    - CSRD / ESRS E1 (Energy-related)
    - ASHRAE 90.1 (Chiller Efficiency)
    - EPA Mandatory Reporting Rule (40 CFR 98)
    - EU ETS (Monitoring and Reporting Regulation)
    - CDP Climate Change Questionnaire

Supported Cooling Technologies:
    - Electric Chillers (centrifugal, screw, scroll, reciprocating)
    - Absorption Chillers (single-effect, double-effect, triple-effect)
    - Free Cooling / Economizer Systems (air-side, water-side)
    - Thermal Energy Storage (ice, chilled water, eutectic)
    - District Cooling Networks (chilled water distribution)

Example:
    >>> from greenlang.agents.mrv.cooling_purchase.provenance import (
    ...     CoolingPurchaseProvenance, get_provenance,
    ... )
    >>> prov = get_provenance()
    >>> chain_id = prov.create_chain("calc-001")
    >>> h1 = prov.add_stage(chain_id, "INPUT_VALIDATION", {
    ...     "facility_id": "FAC-001",
    ...     "cooling_technology": "electric_chiller",
    ... })
    >>> h2 = prov.add_stage(chain_id, "TECHNOLOGY_LOOKUP", {
    ...     "chiller_type": "centrifugal",
    ...     "cop": 6.1,
    ...     "iplv": 9.7,
    ... })
    >>> h3 = prov.add_stage(chain_id, "ENERGY_INPUT_CALCULATION", {
    ...     "cooling_kwh": 500000.0,
    ...     "cop": 6.1,
    ...     "electrical_input_kwh": 81967.21,
    ... })
    >>> final = prov.seal_chain(chain_id)
    >>> assert len(final) == 64
    >>> assert prov.verify_chain(chain_id) is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-012 Cooling Purchase (GL-MRV-X-023)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """Convert a value to its string representation for hashing.

    Handles Decimal, float, int, bool, None, and arbitrary objects
    by delegating to ``str()``. This ensures consistent serialization
    across all numeric types used in emission calculations.

    Args:
        value: Any Python value to convert.

    Returns:
        String representation suitable for inclusion in hash payloads.
    """
    if value is None:
        return "null"
    return str(value)

def _canonical_json(data: Dict[str, Any]) -> str:
    """Serialize a dictionary to canonical JSON form.

    Uses ``sort_keys=True`` and ``default=str`` to produce deterministic
    output regardless of insertion order or non-standard types (Decimal,
    datetime, UUID, etc.).

    Args:
        data: Dictionary to serialize. All values must be JSON-serializable
            or convertible via ``str()``.

    Returns:
        Canonical JSON string with sorted keys.
    """
    return json.dumps(data, sort_keys=True, default=str)

# ---------------------------------------------------------------------------
# Valid provenance stages (19 stages)
# ---------------------------------------------------------------------------

VALID_STAGES = frozenset({
    # Stage 1: Input validation
    "INPUT_VALIDATION",
    # Stage 2: Technology lookup
    "TECHNOLOGY_LOOKUP",
    # Stage 3: Efficiency conversion
    "EFFICIENCY_CONVERSION",
    # Stage 4: Part-load calculation
    "PART_LOAD_CALCULATION",
    # Stage 5: Energy input calculation
    "ENERGY_INPUT_CALCULATION",
    # Stage 6: Auxiliary energy
    "AUXILIARY_ENERGY",
    # Stage 7: Grid factor application
    "GRID_FACTOR_APPLICATION",
    # Stage 8: Heat source factor application
    "HEAT_SOURCE_FACTOR_APPLICATION",
    # Stage 9: Gas decomposition
    "GAS_DECOMPOSITION",
    # Stage 10: GWP application
    "GWP_APPLICATION",
    # Stage 11: Free cooling calculation
    "FREE_COOLING_CALCULATION",
    # Stage 12: TES temporal shifting
    "TES_TEMPORAL_SHIFTING",
    # Stage 13: District loss adjustment
    "DISTRICT_LOSS_ADJUSTMENT",
    # Stage 14: Refrigerant leakage
    "REFRIGERANT_LEAKAGE",
    # Stage 15: Uncertainty quantification
    "UNCERTAINTY_QUANTIFICATION",
    # Stage 16: Compliance check
    "COMPLIANCE_CHECK",
    # Stage 17: Aggregation
    "AGGREGATION",
    # Stage 18: Batch assembly
    "BATCH_ASSEMBLY",
    # Stage 19: Result finalization
    "RESULT_FINALIZATION",
    # Seal stage (appended by seal_chain)
    "PROVENANCE_SEALED",
})

#: Canonical ordered list of 19 stages for pipeline reference.
STAGE_ORDER: List[str] = [
    "INPUT_VALIDATION",
    "TECHNOLOGY_LOOKUP",
    "EFFICIENCY_CONVERSION",
    "PART_LOAD_CALCULATION",
    "ENERGY_INPUT_CALCULATION",
    "AUXILIARY_ENERGY",
    "GRID_FACTOR_APPLICATION",
    "HEAT_SOURCE_FACTOR_APPLICATION",
    "GAS_DECOMPOSITION",
    "GWP_APPLICATION",
    "FREE_COOLING_CALCULATION",
    "TES_TEMPORAL_SHIFTING",
    "DISTRICT_LOSS_ADJUSTMENT",
    "REFRIGERANT_LEAKAGE",
    "UNCERTAINTY_QUANTIFICATION",
    "COMPLIANCE_CHECK",
    "AGGREGATION",
    "BATCH_ASSEMBLY",
    "RESULT_FINALIZATION",
]

#: Number of defined provenance stages.
STAGE_COUNT: int = len(STAGE_ORDER)

# ---------------------------------------------------------------------------
# ProvenanceEntry dataclass (frozen for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceEntry:
    """A single immutable, tamper-evident provenance record for a Cooling
    Purchase emission calculation stage.

    Each entry in the provenance chain captures the stage name, its SHA-256
    hash (incorporating the previous entry's hash for chain integrity), the
    ISO-formatted UTC timestamp, the link to the previous hash, and a
    metadata dictionary containing the raw data used to compute this hash.

    The ``frozen=True`` decorator ensures that once created, an entry
    cannot be modified in-place, enforcing the append-only property of
    the provenance chain.

    Attributes:
        stage: Identifies the calculation stage that produced this entry.
            Must be one of the 19 VALID_STAGES or a custom stage.
        hash_value: SHA-256 hex digest of this entry's payload, computed as
            ``SHA256(f"{previous_hash}|{stage}|{canonical_json(metadata)}")``.
        timestamp: ISO 8601 formatted UTC timestamp string recording when
            this entry was created.
        previous_hash: The ``hash_value`` of the immediately preceding
            entry in the chain. For the first entry, this is an empty
            string ``""``.
        metadata: Dictionary of additional contextual fields that were
            used to compute this entry's hash.
        chain_id: Identifier of the provenance chain this entry belongs to.

    Example:
        >>> entry = ProvenanceEntry(
        ...     stage="ENERGY_INPUT_CALCULATION",
        ...     hash_value="a1b2c3...",
        ...     timestamp="2026-02-22T10:30:00+00:00",
        ...     previous_hash="d4e5f6...",
        ...     metadata={"cooling_kwh": 500000.0, "cop": 6.1},
        ...     chain_id="chain-001",
        ... )
        >>> entry.stage
        'ENERGY_INPUT_CALCULATION'
    """

    stage: str
    hash_value: str
    timestamp: str
    previous_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chain_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a plain dictionary.

        Returns a shallow copy of all fields as a dictionary suitable for
        JSON serialization, database persistence, or API responses.

        Returns:
            Dictionary with keys ``stage``, ``hash_value``, ``timestamp``,
            ``previous_hash``, ``metadata``, and ``chain_id``.
        """
        return {
            "stage": self.stage,
            "hash_value": self.hash_value,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "metadata": dict(self.metadata) if self.metadata else {},
            "chain_id": self.chain_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ProvenanceEntry:
        """Deserialize a dictionary into a ProvenanceEntry.

        Performs minimal validation to ensure required fields are present.
        The metadata field defaults to an empty dictionary if not provided.

        Args:
            data: Dictionary containing at least ``stage``, ``hash_value``,
                ``timestamp``, and ``previous_hash`` keys.

        Returns:
            A new frozen ProvenanceEntry instance.

        Raises:
            KeyError: If a required field is missing from the dictionary.
            TypeError: If the data argument is not a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict, got {type(data).__name__}"
            )
        return ProvenanceEntry(
            stage=data["stage"],
            hash_value=data["hash_value"],
            timestamp=data["timestamp"],
            previous_hash=data["previous_hash"],
            metadata=data.get("metadata", {}),
            chain_id=data.get("chain_id", ""),
        )

# ---------------------------------------------------------------------------
# CoolingPurchaseProvenance (Thread-safe Singleton)
# ---------------------------------------------------------------------------

class CoolingPurchaseProvenance:
    """SHA-256 provenance chain tracker for Cooling Purchase emission
    calculations.

    Implements a multi-chain provenance system where each calculation request
    gets its own independent chain identified by a ``chain_id``. Each chain
    is an ordered, append-only list of frozen ProvenanceEntry instances. Every
    entry's hash incorporates the previous entry's hash, creating a
    tamper-evident linked list of cryptographic digests.

    This class provides methods for the complete 19-stage provenance pipeline:

    1.  **INPUT_VALIDATION**: Validate cooling request parameters
    2.  **TECHNOLOGY_LOOKUP**: Look up COP/IPLV from cooling database
    3.  **EFFICIENCY_CONVERSION**: Convert between COP/EER/kW_per_ton/SEER
    4.  **PART_LOAD_CALCULATION**: Calculate IPLV/NPLV from part-load COPs
    5.  **ENERGY_INPUT_CALCULATION**: Calculate electrical/thermal input
    6.  **AUXILIARY_ENERGY**: Calculate auxiliary/parasitic energy
    7.  **GRID_FACTOR_APPLICATION**: Apply grid emission factor to electricity
    8.  **HEAT_SOURCE_FACTOR_APPLICATION**: Apply heat source EF (absorption)
    9.  **GAS_DECOMPOSITION**: Decompose CO2e into CO2, CH4, N2O
    10. **GWP_APPLICATION**: Apply GWP factors
    11. **FREE_COOLING_CALCULATION**: Calculate free cooling pump energy
    12. **TES_TEMPORAL_SHIFTING**: Calculate TES charge/discharge emissions
    13. **DISTRICT_LOSS_ADJUSTMENT**: Adjust for distribution network losses
    14. **REFRIGERANT_LEAKAGE**: Calculate refrigerant leakage (informational)
    15. **UNCERTAINTY_QUANTIFICATION**: Run uncertainty analysis
    16. **COMPLIANCE_CHECK**: Check regulatory compliance
    17. **AGGREGATION**: Aggregate results
    18. **BATCH_ASSEMBLY**: Assemble batch results
    19. **RESULT_FINALIZATION**: Finalize and hash result

    Thread Safety:
        This class implements the singleton pattern with a reentrant lock
        (``threading.RLock``). All public methods that modify chain state are
        protected by the lock. This allows safe concurrent access from
        multiple threads within the same calculation pipeline.

    Singleton Access:
        Use the module-level ``get_provenance()`` function to obtain the
        singleton instance. Use ``reset()`` classmethod to destroy the
        singleton for test isolation.

    Attributes:
        _chains: Dictionary mapping chain_id to list of ProvenanceEntry.
        _sealed: Set of chain_ids that have been sealed.
        _lock: Reentrant lock for thread-safe chain modifications.
        _max_entries_per_chain: Maximum entries per chain before warning.

    Example:
        >>> prov = CoolingPurchaseProvenance()
        >>> cid = prov.create_chain("calc-001")
        >>> h1 = prov.add_stage(cid, "INPUT_VALIDATION", {"f": "FAC-001"})
        >>> h2 = prov.add_stage(cid, "TECHNOLOGY_LOOKUP", {"cop": 6.1})
        >>> seal = prov.seal_chain(cid)
        >>> assert prov.verify_chain(cid) is True
        >>> assert len(prov.get_chain(cid)) >= 2
    """

    # ------------------------------------------------------------------
    # Singleton machinery
    # ------------------------------------------------------------------

    _instance: Optional[CoolingPurchaseProvenance] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> CoolingPurchaseProvenance:
        """Create or return the singleton instance.

        Uses double-checked locking with the class-level RLock to ensure
        thread-safe singleton creation. Only a single instance of
        CoolingPurchaseProvenance will ever exist in the process.

        Returns:
            The singleton CoolingPurchaseProvenance instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
                    logger.info(
                        "CoolingPurchaseProvenance singleton created"
                    )
        return cls._instance

    # ------------------------------------------------------------------
    # Class constants
    # ------------------------------------------------------------------

    #: Default maximum number of entries per chain before warnings.
    DEFAULT_MAX_ENTRIES_PER_CHAIN: int = 10000

    #: Maximum number of concurrent chains.
    DEFAULT_MAX_CHAINS: int = 50000

    #: Prefix used for all Cooling Purchase provenance identifiers.
    PREFIX: str = "gl_cp"

    #: Agent identifier for this provenance tracker.
    AGENT_ID: str = "AGENT-MRV-012"

    #: Machine-readable agent code for cross-referencing.
    AGENT_CODE: str = "GL-MRV-X-023"

    #: Human-readable agent name.
    AGENT_NAME: str = "Cooling Purchase"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_entries_per_chain: int = DEFAULT_MAX_ENTRIES_PER_CHAIN,
        max_chains: int = DEFAULT_MAX_CHAINS,
    ) -> None:
        """Initialize the provenance tracker (only on first creation).

        Creates the internal data structures for tracking multiple
        independent provenance chains. Subsequent calls to ``__init__``
        on the singleton are no-ops due to the ``_initialized`` guard.

        Args:
            max_entries_per_chain: Maximum entries per chain before a
                warning is logged. Defaults to 10000.
            max_chains: Maximum concurrent chains. When exceeded, the
                oldest non-sealed chains are evicted. Defaults to 50000.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> assert prov.AGENT_ID == "AGENT-MRV-012"
        """
        if getattr(self, "_initialized", False):
            return

        self._chains: Dict[str, List[ProvenanceEntry]] = {}
        self._sealed: set = set()
        self._instance_lock: threading.RLock = threading.RLock()
        self._max_entries_per_chain: int = max(1, max_entries_per_chain)
        self._max_chains: int = max(1, max_chains)
        self._chain_creation_times: Dict[str, str] = {}
        self._initialized = True

        logger.info(
            "%s provenance tracker initialized "
            "(max_entries_per_chain=%d, max_chains=%d)",
            self.AGENT_ID,
            self._max_entries_per_chain,
            self._max_chains,
        )

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton instance for test isolation.

        After calling this method, the next instantiation will create
        a fresh CoolingPurchaseProvenance with empty chains. Primarily
        intended for use in test teardown to prevent state leakage
        between test cases.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> prov.create_chain("test-1")
            'test-1'
            >>> CoolingPurchaseProvenance.reset()
            >>> prov2 = CoolingPurchaseProvenance()
            >>> # prov2 is a fresh instance with no chains
        """
        with cls._lock:
            cls._instance = None
        logger.info(
            "CoolingPurchaseProvenance singleton reset to None"
        )

    # ------------------------------------------------------------------
    # Core hashing engine
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute a standalone SHA-256 hash of serialized data.

        Converts the input to canonical JSON form and computes its
        SHA-256 digest. This method does not modify any chain state
        and can be used for pre-computing or verifying hashes externally.

        Args:
            data: Any JSON-serializable data (dict, list, str, number,
                etc.). Dictionaries are serialized with ``sort_keys=True``
                and ``default=str`` for deterministic output.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Example:
            >>> h = CoolingPurchaseProvenance.compute_hash({"key": "val"})
            >>> assert len(h) == 64
            >>> h2 = CoolingPurchaseProvenance.compute_hash({"key": "val"})
            >>> assert h == h2  # deterministic
        """
        if isinstance(data, dict):
            canonical = _canonical_json(data)
        elif isinstance(data, str):
            canonical = data
        else:
            canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _compute_chain_hash(
        self,
        chain_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Compute a SHA-256 hash for a stage and append to the chain.

        This is the core hashing method used by ``add_stage``. It
        implements the GreenLang provenance hashing protocol:

        1. Retrieve the hash_value of the last entry in the chain (or
           empty string if the chain is empty).
        2. Serialize ``data`` to canonical JSON (``sort_keys=True``,
           ``default=str``).
        3. Construct the payload as ``f"{previous}|{stage}|{canonical}"``.
        4. Compute ``SHA-256(payload.encode("utf-8")).hexdigest()``.
        5. Create a frozen ProvenanceEntry and append to the chain.
        6. Return the hex digest.

        Args:
            chain_id: Identifier of the chain to append to.
            stage: String label identifying the calculation stage.
            data: Dictionary of values to include in the hash.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``chain_id`` does not exist, ``stage`` is empty,
                or the chain has already been sealed.
            TypeError: If ``data`` is not a dictionary.
        """
        if not stage:
            raise ValueError("stage must not be empty")
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}"
            )
        if chain_id not in self._chains:
            raise ValueError(
                f"Chain '{chain_id}' does not exist. "
                f"Call create_chain() first."
            )
        if chain_id in self._sealed:
            raise ValueError(
                f"Chain '{chain_id}' has already been sealed. "
                f"No further stages can be added."
            )

        chain = self._chains[chain_id]
        previous = chain[-1].hash_value if chain else ""
        canonical = _canonical_json(data)
        payload = f"{previous}|{stage}|{canonical}"
        hash_value = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

        entry = ProvenanceEntry(
            stage=stage,
            hash_value=hash_value,
            timestamp=utcnow().isoformat(),
            previous_hash=previous,
            metadata=data,
            chain_id=chain_id,
        )
        chain.append(entry)

        # Warn if chain is getting large
        if len(chain) > self._max_entries_per_chain:
            logger.warning(
                "Chain '%s' has %d entries, exceeding max_entries_per_chain=%d",
                chain_id,
                len(chain),
                self._max_entries_per_chain,
            )

        logger.debug(
            "Provenance entry: chain=%s stage=%s hash_prefix=%s len=%d",
            chain_id,
            stage,
            hash_value[:16],
            len(chain),
        )
        return hash_value

    def _compute_hash_no_append(
        self,
        chain_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Compute a SHA-256 hash without appending to the chain.

        Useful for pre-computing hashes for validation or comparison
        without side effects on the provenance chain.

        Args:
            chain_id: Identifier of the chain to reference for previous hash.
            stage: String label identifying the calculation stage.
            data: Dictionary of values to include in the hash.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If chain_id does not exist.
        """
        if chain_id not in self._chains:
            raise ValueError(
                f"Chain '{chain_id}' does not exist."
            )
        chain = self._chains[chain_id]
        previous = chain[-1].hash_value if chain else ""
        canonical = _canonical_json(data)
        payload = f"{previous}|{stage}|{canonical}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Chain lifecycle methods
    # ------------------------------------------------------------------

    def create_chain(self, calc_id: str) -> str:
        """Start a new provenance chain for a calculation.

        Creates an empty chain identified by ``calc_id``. The chain will
        be populated as calculation stages call ``add_stage``. If a chain
        with the given ID already exists, a ValueError is raised to
        prevent accidental overwriting of audit data.

        If the maximum number of concurrent chains is reached, the oldest
        non-sealed chain is evicted with a warning.

        Args:
            calc_id: Unique identifier for this calculation run. This
                becomes the chain_id used in all subsequent operations.
                Must not be empty.

        Returns:
            The chain_id (same as ``calc_id``) for use in subsequent calls.

        Raises:
            ValueError: If ``calc_id`` is empty or a chain with the same
                ID already exists.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("calc-001")
            >>> assert cid == "calc-001"
        """
        if not calc_id:
            raise ValueError("calc_id must not be empty")

        with self._instance_lock:
            if calc_id in self._chains:
                raise ValueError(
                    f"Chain '{calc_id}' already exists. Use a unique ID."
                )

            # Evict oldest non-sealed chain if at capacity
            self._evict_chains_if_needed()

            self._chains[calc_id] = []
            self._chain_creation_times[calc_id] = utcnow().isoformat()

        logger.info(
            "Created provenance chain: chain_id=%s agent=%s",
            calc_id,
            self.AGENT_ID,
        )
        return calc_id

    def start_chain(self, calculation_id: str) -> str:
        """Start a new provenance chain and return the initial hash.

        Convenience method that creates a chain and adds an initial
        INPUT_VALIDATION stage with the calculation ID. Returns the
        hash of the initial entry.

        Args:
            calculation_id: Unique identifier for this calculation run.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the
            initial entry.

        Raises:
            ValueError: If ``calculation_id`` is empty or already exists.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> h = prov.start_chain("calc-start-001")
            >>> assert len(h) == 64
        """
        if not calculation_id:
            raise ValueError("calculation_id must not be empty")

        self.create_chain(calculation_id)
        return self.add_stage(
            calculation_id,
            "INPUT_VALIDATION",
            {
                "calculation_id": calculation_id,
                "agent_id": self.AGENT_ID,
                "agent_code": self.AGENT_CODE,
                "initialized_at": utcnow().isoformat(),
            },
        )

    def add_stage(
        self,
        chain_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Add a stage entry to an existing provenance chain.

        Appends a new ProvenanceEntry to the chain identified by
        ``chain_id``. The entry's SHA-256 hash incorporates the
        previous entry's hash, the stage name, and the canonical JSON
        of the data dictionary.

        The stage name is validated against VALID_STAGES. Unknown stage
        names are permitted but trigger a warning log to support future
        extensibility.

        Args:
            chain_id: Identifier of the chain to append to. Must have
                been previously created via ``create_chain``.
            stage: One of the 19 VALID_STAGES (e.g.,
                ``"INPUT_VALIDATION"``, ``"ENERGY_INPUT_CALCULATION"``).
                Unknown stages are allowed but logged as warnings.
            data: Dictionary of contextual data for this stage. All
                values must be JSON-serializable or convertible via
                ``str()``. Common keys depend on the stage:

                - INPUT_VALIDATION: ``facility_id``,
                  ``cooling_technology``, ``reporting_year``
                - TECHNOLOGY_LOOKUP: ``chiller_type``, ``cop``,
                  ``iplv``, ``condenser_type``
                - EFFICIENCY_CONVERSION: ``from_metric``,
                  ``to_metric``, ``value``, ``converted_value``
                - PART_LOAD_CALCULATION: ``cop_100``, ``cop_75``,
                  ``cop_50``, ``cop_25``, ``iplv``
                - ENERGY_INPUT_CALCULATION: ``cooling_kwh``, ``cop``,
                  ``electrical_input_kwh``
                - AUXILIARY_ENERGY: ``auxiliary_kwh``,
                  ``auxiliary_pct``, ``condenser_type``
                - GRID_FACTOR_APPLICATION: ``grid_ef``,
                  ``electricity_kwh``, ``emissions_kgco2e``
                - HEAT_SOURCE_FACTOR_APPLICATION: ``heat_source``,
                  ``heat_source_ef``, ``thermal_input_kwh``
                - GAS_DECOMPOSITION: ``co2_kg``, ``ch4_kg``,
                  ``n2o_kg``, ``total_co2e_kg``
                - GWP_APPLICATION: ``gwp_source``, ``gwp_ch4``,
                  ``gwp_n2o``
                - FREE_COOLING_CALCULATION: ``pump_energy_kwh``,
                  ``grid_ef``, ``savings_kwh``
                - TES_TEMPORAL_SHIFTING: ``charge_kwh``,
                  ``discharge_kwh``, ``grid_ef_charge``,
                  ``grid_ef_peak``
                - DISTRICT_LOSS_ADJUSTMENT: ``distribution_loss_pct``,
                  ``pump_energy_kwh``, ``adjusted_emissions``
                - REFRIGERANT_LEAKAGE: ``refrigerant``,
                  ``charge_kg``, ``leak_rate``, ``gwp``
                - UNCERTAINTY_QUANTIFICATION: ``mean``,
                  ``std_dev``, ``ci_lower``, ``ci_upper``
                - COMPLIANCE_CHECK: ``framework``, ``compliant``,
                  ``findings``
                - AGGREGATION: ``group_key``, ``count``,
                  ``total_emissions``
                - BATCH_ASSEMBLY: ``batch_id``, ``total``,
                  ``completed``
                - RESULT_FINALIZATION: ``total_co2e_kg``,
                  ``validation_status``

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the new
            entry.

        Raises:
            ValueError: If ``chain_id`` does not exist, ``stage`` is
                empty, or the chain has been sealed.
            TypeError: If ``data`` is not a dictionary.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("calc-002")
            >>> h = prov.add_stage(cid, "INPUT_VALIDATION", {
            ...     "facility_id": "FAC-002",
            ...     "cooling_technology": "absorption_chiller",
            ... })
            >>> assert len(h) == 64
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        if stage not in VALID_STAGES:
            logger.warning(
                "Unknown provenance stage '%s' for chain '%s'. "
                "Expected one of %d valid stages.",
                stage,
                chain_id,
                STAGE_COUNT,
            )

        with self._instance_lock:
            return self._compute_chain_hash(chain_id, stage, data)

    def seal_chain(self, chain_id: str) -> str:
        """Seal a provenance chain with a final SHA-256 hash.

        Appends a PROVENANCE_SEALED entry to the chain and marks the
        chain as sealed. No further stages can be added after sealing.
        The seal entry includes metadata about the chain's total length,
        stage summary, and creation timestamp.

        The returned hash is the cumulative chain hash: because each
        entry's hash incorporates all previous entries, this single
        64-character string cryptographically represents the entire
        calculation lineage.

        Args:
            chain_id: Identifier of the chain to seal.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the
            final sealed entry.

        Raises:
            ValueError: If ``chain_id`` does not exist or has already
                been sealed.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("calc-003")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "FAC-003"})
            '...'
            >>> seal = prov.seal_chain(cid)
            >>> assert len(seal) == 64
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            if chain_id in self._sealed:
                raise ValueError(
                    f"Chain '{chain_id}' has already been sealed."
                )

            chain = self._chains[chain_id]
            stage_summary = self._get_stage_summary(chain)

            seal_data = {
                "chain_id": chain_id,
                "chain_length": len(chain),
                "stage_summary": stage_summary,
                "created_at": self._chain_creation_times.get(
                    chain_id, ""
                ),
                "sealed_at": utcnow().isoformat(),
                "agent_id": self.AGENT_ID,
                "agent_code": self.AGENT_CODE,
                "agent_name": self.AGENT_NAME,
            }

            seal_hash = self._compute_chain_hash(
                chain_id, "PROVENANCE_SEALED", seal_data
            )
            self._sealed.add(chain_id)

        logger.info(
            "Sealed provenance chain: chain_id=%s entries=%d "
            "hash_prefix=%s",
            chain_id,
            len(chain),
            seal_hash[:16],
        )
        return seal_hash

    def finalize_chain(
        self,
        chain_id: str,
        final_result: Dict[str, Any],
    ) -> str:
        """Finalize a chain with a result and seal it.

        Convenience method that adds a RESULT_FINALIZATION stage with
        the provided final_result data and then seals the chain.

        Args:
            chain_id: Identifier of the chain to finalize.
            final_result: Dictionary of final calculation results to
                record before sealing.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the
            sealed entry.

        Raises:
            ValueError: If ``chain_id`` does not exist or is sealed.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("finalize-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
            '...'
            >>> seal = prov.finalize_chain(cid, {
            ...     "total_co2e_kg": 12500.0,
            ...     "validation_status": "PASS",
            ... })
            >>> assert len(seal) == 64
        """
        self.add_stage(chain_id, "RESULT_FINALIZATION", final_result)
        return self.seal_chain(chain_id)

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self, chain_id: str) -> bool:
        """Verify the integrity of a provenance chain.

        Walks the chain in order and recomputes each entry's hash from
        the previous entry's hash, the stage, and the metadata. Returns
        True if every entry's hash matches the expected recomputed value,
        and the chain linkage (previous_hash pointers) is intact.

        For the first entry, previous_hash must be an empty string.
        For each subsequent entry, previous_hash must match the
        preceding entry's hash_value.

        This method is designed for audit verification and should be
        called after deserialization or at the end of a calculation
        pipeline to confirm chain integrity.

        Args:
            chain_id: Identifier of the chain to verify.

        Returns:
            True if the chain is intact, False if any entry fails
            verification or the chain links are broken.

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("verify-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
            '...'
            >>> assert prov.verify_chain(cid) is True
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            chain = list(self._chains[chain_id])

        if not chain:
            logger.debug(
                "verify_chain: chain '%s' is empty - trivially valid",
                chain_id,
            )
            return True

        for i, entry in enumerate(chain):
            # Verify chain linkage
            if i == 0:
                if entry.previous_hash != "":
                    logger.warning(
                        "verify_chain: chain '%s' entry[0] previous_hash "
                        "is not empty string (got '%s')",
                        chain_id,
                        entry.previous_hash[:16],
                    )
                    return False
            else:
                if entry.previous_hash != chain[i - 1].hash_value:
                    logger.warning(
                        "verify_chain: chain '%s' chain break between "
                        "entry[%d] and entry[%d]",
                        chain_id,
                        i - 1,
                        i,
                    )
                    return False

            # Recompute and verify hash
            canonical = _canonical_json(entry.metadata)
            payload = f"{entry.previous_hash}|{entry.stage}|{canonical}"
            expected_hash = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()

            if entry.hash_value != expected_hash:
                logger.warning(
                    "verify_chain: chain '%s' entry[%d] hash mismatch "
                    "(stage=%s expected=%s got=%s)",
                    chain_id,
                    i,
                    entry.stage,
                    expected_hash[:16],
                    entry.hash_value[:16],
                )
                return False

        logger.debug(
            "verify_chain: chain '%s' %d entries verified successfully",
            chain_id,
            len(chain),
        )
        return True

    def verify_chain_detailed(
        self, chain_id: str,
    ) -> Tuple[bool, Optional[str], int]:
        """Verify chain integrity with detailed error information.

        Similar to ``verify_chain`` but returns a tuple with the
        validation result, an optional error message, and the index
        of the first failed entry (or -1 if all entries pass).

        Args:
            chain_id: Identifier of the chain to verify.

        Returns:
            Tuple of ``(is_valid, error_message, failed_index)``.
            When the chain is intact, returns ``(True, None, -1)``.

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("detail-001")
            >>> valid, msg, idx = prov.verify_chain_detailed(cid)
            >>> assert valid is True
            >>> assert msg is None
            >>> assert idx == -1
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            chain = list(self._chains[chain_id])

        if not chain:
            return True, None, -1

        for i, entry in enumerate(chain):
            # Verify required fields
            if not entry.stage:
                return False, f"entry[{i}] has empty stage", i
            if not entry.hash_value:
                return False, f"entry[{i}] has empty hash_value", i
            if not entry.timestamp:
                return False, f"entry[{i}] has empty timestamp", i

            # Verify chain linkage
            if i == 0:
                if entry.previous_hash != "":
                    return (
                        False,
                        f"entry[0] previous_hash is not empty "
                        f"(got '{entry.previous_hash[:16]}')",
                        i,
                    )
            else:
                if entry.previous_hash != chain[i - 1].hash_value:
                    return (
                        False,
                        f"chain break between entry[{i - 1}] and "
                        f"entry[{i}]",
                        i,
                    )

            # Recompute and verify hash
            canonical = _canonical_json(entry.metadata)
            payload = f"{entry.previous_hash}|{entry.stage}|{canonical}"
            expected = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()

            if entry.hash_value != expected:
                return (
                    False,
                    f"entry[{i}] hash mismatch (stage={entry.stage})",
                    i,
                )

        return True, None, -1

    # ------------------------------------------------------------------
    # Chain query and inspection
    # ------------------------------------------------------------------

    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all stages in a provenance chain as dictionaries.

        Returns a list of serialized ProvenanceEntry dictionaries for
        the specified chain, in insertion order. This is the primary
        method for inspecting the complete audit trail of a calculation.

        Args:
            chain_id: Identifier of the chain to retrieve.

        Returns:
            List of entry dictionaries with keys ``stage``,
            ``hash_value``, ``timestamp``, ``previous_hash``,
            ``metadata``, and ``chain_id``. Returns an empty list
            if the chain exists but has no entries.

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("get-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "FAC-001"})
            '...'
            >>> chain = prov.get_chain(cid)
            >>> assert len(chain) == 1
            >>> assert chain[0]["stage"] == "INPUT_VALIDATION"
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            return [entry.to_dict() for entry in self._chains[chain_id]]

    def get_chain_entries(self, chain_id: str) -> List[ProvenanceEntry]:
        """Get all stages in a provenance chain as ProvenanceEntry objects.

        Returns a shallow copy of the internal chain list as frozen
        ProvenanceEntry instances. Use this when you need typed access
        to entry attributes rather than dictionary access.

        Args:
            chain_id: Identifier of the chain to retrieve.

        Returns:
            List of ProvenanceEntry objects in insertion order.

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("entries-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"valid": True})
            '...'
            >>> entries = prov.get_chain_entries(cid)
            >>> assert entries[0].stage == "INPUT_VALIDATION"
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            return list(self._chains[chain_id])

    def get_chain_hash(self, chain_id: str) -> str:
        """Get the cumulative hash of a provenance chain.

        Returns the hash_value of the most recent entry in the chain.
        Since each entry's hash transitively incorporates all preceding
        entries via the chain linking protocol, this single hash
        uniquely identifies the complete state of the provenance chain.

        If the chain is empty, returns an empty string.

        Args:
            chain_id: Identifier of the chain.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters), or empty
            string if the chain has no entries.

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("hash-001")
            >>> assert prov.get_chain_hash(cid) == ""
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
            '...'
            >>> assert len(prov.get_chain_hash(cid)) == 64
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            chain = self._chains[chain_id]
            if not chain:
                return ""
            return chain[-1].hash_value

    def get_chain_length(self, chain_id: str) -> int:
        """Return the number of entries in a provenance chain.

        Args:
            chain_id: Identifier of the chain.

        Returns:
            Integer count of entries in the chain.

        Raises:
            ValueError: If ``chain_id`` does not exist.
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            return len(self._chains[chain_id])

    def get_entries_by_stage(
        self, chain_id: str, stage: str,
    ) -> List[ProvenanceEntry]:
        """Return all entries in a chain matching a specific stage.

        Filters the chain to return only entries with the given stage
        label. Useful for extracting all COP lookups or compliance
        checks from a multi-step calculation.

        Args:
            chain_id: Identifier of the chain to search.
            stage: Stage label to filter by (e.g.,
                ``"TECHNOLOGY_LOOKUP"``, ``"COMPLIANCE_CHECK"``).

        Returns:
            List of matching ProvenanceEntry objects in insertion order.

        Raises:
            ValueError: If ``chain_id`` does not exist.
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            return [
                e for e in self._chains[chain_id]
                if e.stage == stage
            ]

    def get_entry_by_index(
        self, chain_id: str, index: int,
    ) -> Optional[ProvenanceEntry]:
        """Return a specific entry by its position in the chain.

        Args:
            chain_id: Identifier of the chain.
            index: Zero-based index into the chain. Negative indices
                are supported (e.g., -1 for the last entry).

        Returns:
            The ProvenanceEntry at the specified index, or None if the
            index is out of range.

        Raises:
            ValueError: If ``chain_id`` does not exist.
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            try:
                return self._chains[chain_id][index]
            except IndexError:
                return None

    def get_latest_entry(
        self, chain_id: str,
    ) -> Optional[ProvenanceEntry]:
        """Return the most recent entry in the chain.

        Args:
            chain_id: Identifier of the chain.

        Returns:
            The last ProvenanceEntry, or None if the chain is empty.

        Raises:
            ValueError: If ``chain_id`` does not exist.
        """
        return self.get_entry_by_index(chain_id, -1)

    def get_stage_summary(self, chain_id: str) -> Dict[str, int]:
        """Return a summary of entry counts grouped by stage.

        Args:
            chain_id: Identifier of the chain.

        Returns:
            Dictionary mapping stage names to their occurrence count.

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("summary-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
            '...'
            >>> prov.add_stage(cid, "TECHNOLOGY_LOOKUP", {"cop": 6.1})
            '...'
            >>> summary = prov.get_stage_summary(cid)
            >>> assert summary == {
            ...     "INPUT_VALIDATION": 1,
            ...     "TECHNOLOGY_LOOKUP": 1,
            ... }
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            return self._get_stage_summary(self._chains[chain_id])

    def is_sealed(self, chain_id: str) -> bool:
        """Check whether a chain has been sealed.

        Args:
            chain_id: Identifier of the chain.

        Returns:
            True if the chain has been sealed via ``seal_chain``.

        Raises:
            ValueError: If ``chain_id`` does not exist.
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )
            return chain_id in self._sealed

    def list_chains(self) -> List[Dict[str, Any]]:
        """List all tracked provenance chains with summary information.

        Returns:
            List of dictionaries, each containing:
            - ``chain_id``: Chain identifier
            - ``length``: Number of entries
            - ``sealed``: Whether the chain is sealed
            - ``created_at``: ISO timestamp of chain creation
            - ``chain_hash``: Current cumulative hash (or "" if empty)
        """
        with self._instance_lock:
            result = []
            for cid, chain in self._chains.items():
                result.append({
                    "chain_id": cid,
                    "length": len(chain),
                    "sealed": cid in self._sealed,
                    "created_at": self._chain_creation_times.get(cid, ""),
                    "chain_hash": (
                        chain[-1].hash_value if chain else ""
                    ),
                })
            return result

    def chain_count(self) -> int:
        """Return the number of tracked chains.

        Returns:
            Integer count of all chains (sealed and unsealed).
        """
        with self._instance_lock:
            return len(self._chains)

    # ------------------------------------------------------------------
    # Export / Serialization
    # ------------------------------------------------------------------

    def export_chain(self, chain_id: str) -> Dict[str, Any]:
        """Export a complete provenance chain for external audit.

        Produces a self-contained dictionary with all chain metadata,
        entries, stage summary, and verification status. This is the
        primary method for generating audit artifacts.

        Args:
            chain_id: Identifier of the chain to export.

        Returns:
            Dictionary with keys:
            - ``agent_id``: Agent identifier (AGENT-MRV-012)
            - ``agent_code``: Agent code (GL-MRV-X-023)
            - ``agent_name``: Human-readable agent name
            - ``prefix``: Provenance prefix (gl_cp)
            - ``chain_id``: Chain identifier
            - ``chain_length``: Number of entries
            - ``chain_hash``: Cumulative chain hash
            - ``sealed``: Whether the chain is sealed
            - ``created_at``: ISO timestamp of chain creation
            - ``exported_at``: ISO timestamp of export
            - ``entries``: List of entry dictionaries
            - ``stage_summary``: Dict of stage name to count
            - ``stages_defined``: Total defined stages (19)
            - ``verification``: Chain verification result

        Raises:
            ValueError: If ``chain_id`` does not exist.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> cid = prov.create_chain("export-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
            '...'
            >>> export = prov.export_chain(cid)
            >>> assert export["agent_id"] == "AGENT-MRV-012"
            >>> assert export["chain_length"] == 1
        """
        if not chain_id:
            raise ValueError("chain_id must not be empty")

        with self._instance_lock:
            if chain_id not in self._chains:
                raise ValueError(
                    f"Chain '{chain_id}' does not exist."
                )

            chain = self._chains[chain_id]
            entries = [entry.to_dict() for entry in chain]
            chain_hash = chain[-1].hash_value if chain else ""
            stage_summary = self._get_stage_summary(chain)
            sealed = chain_id in self._sealed
            created_at = self._chain_creation_times.get(chain_id, "")

        # Verify chain (outside lock to avoid long hold)
        is_valid = self.verify_chain(chain_id)

        export_data = {
            "agent_id": self.AGENT_ID,
            "agent_code": self.AGENT_CODE,
            "agent_name": self.AGENT_NAME,
            "prefix": self.PREFIX,
            "chain_id": chain_id,
            "chain_length": len(entries),
            "chain_hash": chain_hash,
            "sealed": sealed,
            "created_at": created_at,
            "exported_at": utcnow().isoformat(),
            "entries": entries,
            "stage_summary": stage_summary,
            "stages_defined": STAGE_COUNT,
            "verification": {
                "is_valid": is_valid,
                "verified_at": utcnow().isoformat(),
            },
        }

        logger.info(
            "Exported provenance chain: chain_id=%s entries=%d "
            "sealed=%s valid=%s",
            chain_id,
            len(entries),
            sealed,
            is_valid,
        )
        return export_data

    def export_chain_json(
        self, chain_id: str, indent: int = 2,
    ) -> str:
        """Export a provenance chain as a formatted JSON string.

        Convenience method that calls ``export_chain()`` and serializes
        the result to a formatted JSON string.

        Args:
            chain_id: Identifier of the chain to export.
            indent: JSON indentation level (default 2).

        Returns:
            Formatted JSON string.

        Raises:
            ValueError: If ``chain_id`` does not exist.
        """
        return json.dumps(
            self.export_chain(chain_id), indent=indent, default=str
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire provenance tracker state to a dictionary.

        Returns a complete representation of all chains, their entries,
        and metadata. Intended for state persistence and recovery.

        Returns:
            Dictionary with keys:
            - ``agent_id``: Agent identifier string.
            - ``agent_code``: Agent code string.
            - ``agent_name``: Human-readable agent name.
            - ``prefix``: Provenance prefix string.
            - ``chain_count``: Number of chains.
            - ``sealed_count``: Number of sealed chains.
            - ``chains``: Dict mapping chain_id to list of entry dicts.
            - ``sealed``: List of sealed chain IDs.
            - ``creation_times``: Dict mapping chain_id to creation time.
            - ``max_entries_per_chain``: Max entries setting.
            - ``max_chains``: Max chains setting.
            - ``created_at``: ISO timestamp of serialization.
        """
        with self._instance_lock:
            chains_data = {}
            for cid, chain in self._chains.items():
                chains_data[cid] = [e.to_dict() for e in chain]

            return {
                "agent_id": self.AGENT_ID,
                "agent_code": self.AGENT_CODE,
                "agent_name": self.AGENT_NAME,
                "prefix": self.PREFIX,
                "chain_count": len(self._chains),
                "sealed_count": len(self._sealed),
                "chains": chains_data,
                "sealed": list(self._sealed),
                "creation_times": dict(self._chain_creation_times),
                "max_entries_per_chain": self._max_entries_per_chain,
                "max_chains": self._max_chains,
                "created_at": utcnow().isoformat(),
            }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any],
    ) -> CoolingPurchaseProvenance:
        """Deserialize a dictionary into a CoolingPurchaseProvenance.

        Reconstructs the provenance tracker state from a dictionary
        produced by ``to_dict()``. The singleton is reset and replaced
        with the deserialized instance. Chain integrity is verified
        after deserialization; if verification fails, a warning is
        logged but the instance is still returned.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            The reconstructed CoolingPurchaseProvenance instance.

        Raises:
            TypeError: If ``data`` is not a dictionary.
            KeyError: If required keys are missing.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict, got {type(data).__name__}"
            )

        # Reset singleton to allow fresh creation
        cls.reset()
        instance = cls(
            max_entries_per_chain=data.get(
                "max_entries_per_chain",
                cls.DEFAULT_MAX_ENTRIES_PER_CHAIN,
            ),
            max_chains=data.get("max_chains", cls.DEFAULT_MAX_CHAINS),
        )

        chains_data = data.get("chains", {})
        for cid, entries_list in chains_data.items():
            instance._chains[cid] = [
                ProvenanceEntry.from_dict(e) for e in entries_list
            ]

        instance._sealed = set(data.get("sealed", []))
        instance._chain_creation_times = dict(
            data.get("creation_times", {})
        )

        # Verify all chains
        for cid in instance._chains:
            if not instance.verify_chain(cid):
                logger.warning(
                    "from_dict: chain '%s' failed integrity verification",
                    cid,
                )

        logger.info(
            "Deserialized %s provenance tracker with %d chains",
            cls.AGENT_ID,
            len(instance._chains),
        )
        return instance

    # ------------------------------------------------------------------
    # Domain-specific stage helpers
    # ------------------------------------------------------------------

    def hash_input_validation(
        self,
        chain_id: str,
        facility_id: str,
        cooling_technology: str,
        reporting_year: int,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
        cooling_kwh: Optional[float] = None,
    ) -> str:
        """Hash cooling request parameter validation (Stage 1).

        Records the initial validation of cooling purchase calculation
        request parameters including facility, technology type, and
        reporting period.

        Args:
            chain_id: Provenance chain identifier.
            facility_id: Facility identifier.
            cooling_technology: Type of cooling technology
                (``"electric_chiller"``, ``"absorption_chiller"``,
                ``"free_cooling"``, ``"tes"``, ``"district_cooling"``).
            reporting_year: Reporting year.
            tenant_id: Optional tenant identifier.
            request_id: Optional request identifier.
            cooling_kwh: Optional total cooling demand in kWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if not cooling_technology:
            raise ValueError("cooling_technology must not be empty")
        if reporting_year <= 0:
            raise ValueError(
                f"reporting_year must be positive, got {reporting_year}"
            )

        data: Dict[str, Any] = {
            "facility_id": facility_id,
            "cooling_technology": cooling_technology,
            "reporting_year": reporting_year,
        }
        if tenant_id is not None:
            data["tenant_id"] = tenant_id
        if request_id is not None:
            data["request_id"] = request_id
        if cooling_kwh is not None:
            data["cooling_kwh"] = _safe_str(cooling_kwh)

        logger.info(
            "Hashing input validation: facility=%s tech=%s year=%d",
            facility_id,
            cooling_technology,
            reporting_year,
        )
        return self.add_stage(chain_id, "INPUT_VALIDATION", data)

    def hash_technology_lookup(
        self,
        chain_id: str,
        chiller_type: str,
        cop: float,
        iplv: Optional[float] = None,
        condenser_type: Optional[str] = None,
        refrigerant: Optional[str] = None,
        capacity_kw: Optional[float] = None,
        manufacturer: Optional[str] = None,
        data_source: Optional[str] = None,
    ) -> str:
        """Hash COP/IPLV technology lookup (Stage 2).

        Records the chiller technology parameters retrieved from the
        cooling equipment database, including COP, IPLV, condenser
        type, and refrigerant information.

        Args:
            chain_id: Provenance chain identifier.
            chiller_type: Chiller type (``"centrifugal"``,
                ``"screw"``, ``"scroll"``, ``"reciprocating"``,
                ``"absorption_single"``, ``"absorption_double"``,
                ``"absorption_triple"``).
            cop: Coefficient of Performance at full load.
                Must be positive.
            iplv: Optional Integrated Part Load Value.
            condenser_type: Optional condenser type
                (``"air_cooled"``, ``"water_cooled"``).
            refrigerant: Optional refrigerant type
                (e.g., ``"R-134a"``, ``"R-410A"``, ``"R-1234ze"``).
            capacity_kw: Optional cooling capacity in kW.
            manufacturer: Optional equipment manufacturer.
            data_source: Optional source of technology data
                (e.g., ``"ASHRAE_90.1"``, ``"manufacturer_spec"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not chiller_type:
            raise ValueError("chiller_type must not be empty")
        if cop <= 0:
            raise ValueError(
                f"cop must be positive, got {cop}"
            )

        data: Dict[str, Any] = {
            "chiller_type": chiller_type,
            "cop": _safe_str(cop),
        }
        if iplv is not None:
            data["iplv"] = _safe_str(iplv)
        if condenser_type is not None:
            data["condenser_type"] = condenser_type
        if refrigerant is not None:
            data["refrigerant"] = refrigerant
        if capacity_kw is not None:
            data["capacity_kw"] = _safe_str(capacity_kw)
        if manufacturer is not None:
            data["manufacturer"] = manufacturer
        if data_source is not None:
            data["data_source"] = data_source

        logger.info(
            "Hashing technology lookup: type=%s cop=%s iplv=%s "
            "condenser=%s",
            chiller_type,
            _safe_str(cop),
            _safe_str(iplv) if iplv is not None else "N/A",
            condenser_type or "N/A",
        )
        return self.add_stage(chain_id, "TECHNOLOGY_LOOKUP", data)

    def hash_efficiency_conversion(
        self,
        chain_id: str,
        from_metric: str,
        to_metric: str,
        value: float,
        converted_value: float,
        conversion_formula: Optional[str] = None,
    ) -> str:
        """Hash efficiency metric conversion (Stage 3).

        Records the conversion between different cooling efficiency
        metrics. Supported conversions include:

        - COP to EER: EER = COP * 3.412
        - EER to COP: COP = EER / 3.412
        - COP to kW/ton: kW/ton = 12.0 / (COP * 3.412)
        - kW/ton to COP: COP = 12.0 / (kW_per_ton * 3.412)
        - SEER to COP: COP = SEER / 3.412
        - IPLV to NPLV (with correction factors)

        Args:
            chain_id: Provenance chain identifier.
            from_metric: Source efficiency metric (``"COP"``,
                ``"EER"``, ``"kW_per_ton"``, ``"SEER"``,
                ``"IPLV"``, ``"NPLV"``).
            to_metric: Target efficiency metric.
            value: Original value in source metric.
            converted_value: Converted value in target metric.
            conversion_formula: Optional formula description.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not from_metric:
            raise ValueError("from_metric must not be empty")
        if not to_metric:
            raise ValueError("to_metric must not be empty")

        data: Dict[str, Any] = {
            "from_metric": from_metric,
            "to_metric": to_metric,
            "value": _safe_str(value),
            "converted_value": _safe_str(converted_value),
        }
        if conversion_formula is not None:
            data["conversion_formula"] = conversion_formula

        logger.info(
            "Hashing efficiency conversion: %s %s -> %s %s",
            _safe_str(value),
            from_metric,
            _safe_str(converted_value),
            to_metric,
        )
        return self.add_stage(
            chain_id, "EFFICIENCY_CONVERSION", data
        )

    def hash_part_load_calculation(
        self,
        chain_id: str,
        cop_100: float,
        cop_75: float,
        cop_50: float,
        cop_25: float,
        iplv: float,
        nplv: Optional[float] = None,
        weighting_100: float = 0.01,
        weighting_75: float = 0.42,
        weighting_50: float = 0.45,
        weighting_25: float = 0.12,
        standard: Optional[str] = None,
    ) -> str:
        """Hash part-load IPLV/NPLV calculation (Stage 4).

        Records the Integrated Part Load Value calculation using
        AHRI Standard 550/590 weightings:

            IPLV = 0.01 * COP_100 + 0.42 * COP_75
                 + 0.45 * COP_50 + 0.12 * COP_25

        Or Non-standard Part Load Value (NPLV) with custom weightings
        reflecting actual building load profiles.

        Args:
            chain_id: Provenance chain identifier.
            cop_100: COP at 100% load. Must be positive.
            cop_75: COP at 75% load. Must be positive.
            cop_50: COP at 50% load. Must be positive.
            cop_25: COP at 25% load. Must be positive.
            iplv: Calculated IPLV value.
            nplv: Optional NPLV if non-standard weightings used.
            weighting_100: Weighting for 100% load (default 0.01).
            weighting_75: Weighting for 75% load (default 0.42).
            weighting_50: Weighting for 50% load (default 0.45).
            weighting_25: Weighting for 25% load (default 0.12).
            standard: Optional standard reference
                (e.g., ``"AHRI_550_590"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if cop_100 <= 0:
            raise ValueError(
                f"cop_100 must be positive, got {cop_100}"
            )
        if cop_75 <= 0:
            raise ValueError(
                f"cop_75 must be positive, got {cop_75}"
            )
        if cop_50 <= 0:
            raise ValueError(
                f"cop_50 must be positive, got {cop_50}"
            )
        if cop_25 <= 0:
            raise ValueError(
                f"cop_25 must be positive, got {cop_25}"
            )

        data: Dict[str, Any] = {
            "cop_100": _safe_str(cop_100),
            "cop_75": _safe_str(cop_75),
            "cop_50": _safe_str(cop_50),
            "cop_25": _safe_str(cop_25),
            "iplv": _safe_str(iplv),
            "weighting_100": _safe_str(weighting_100),
            "weighting_75": _safe_str(weighting_75),
            "weighting_50": _safe_str(weighting_50),
            "weighting_25": _safe_str(weighting_25),
        }
        if nplv is not None:
            data["nplv"] = _safe_str(nplv)
        if standard is not None:
            data["standard"] = standard

        logger.info(
            "Hashing part-load calc: COP[100/75/50/25]="
            "[%s/%s/%s/%s] IPLV=%s",
            _safe_str(cop_100),
            _safe_str(cop_75),
            _safe_str(cop_50),
            _safe_str(cop_25),
            _safe_str(iplv),
        )
        return self.add_stage(
            chain_id, "PART_LOAD_CALCULATION", data
        )

    def hash_energy_input_calculation(
        self,
        chain_id: str,
        cooling_kwh: float,
        cop: float,
        electrical_input_kwh: float,
        thermal_input_kwh: Optional[float] = None,
        calculation_method: Optional[str] = None,
    ) -> str:
        """Hash energy input calculation from COP (Stage 5).

        Records the calculation of electrical or thermal energy input
        required to produce the requested cooling output. The core
        formula for electric chillers:

            electrical_input_kwh = cooling_kwh / COP

        For absorption chillers:

            thermal_input_kwh = cooling_kwh / COP
            electrical_input_kwh = parasitic_ratio * thermal_input_kwh

        Args:
            chain_id: Provenance chain identifier.
            cooling_kwh: Cooling output in kWh. Must be non-negative.
            cop: Coefficient of Performance used. Must be positive.
            electrical_input_kwh: Calculated electrical energy input
                in kWh.
            thermal_input_kwh: Optional thermal energy input for
                absorption chillers in kWh.
            calculation_method: Optional method identifier
                (``"cop_division"``, ``"iplv_weighted"``,
                ``"nplv_weighted"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if cooling_kwh < 0:
            raise ValueError(
                f"cooling_kwh must be non-negative, got {cooling_kwh}"
            )
        if cop <= 0:
            raise ValueError(
                f"cop must be positive, got {cop}"
            )

        data: Dict[str, Any] = {
            "cooling_kwh": _safe_str(cooling_kwh),
            "cop": _safe_str(cop),
            "electrical_input_kwh": _safe_str(electrical_input_kwh),
        }
        if thermal_input_kwh is not None:
            data["thermal_input_kwh"] = _safe_str(thermal_input_kwh)
        if calculation_method is not None:
            data["calculation_method"] = calculation_method

        logger.info(
            "Hashing energy input calc: %s kWh cooling / COP %s "
            "= %s kWh electrical",
            _safe_str(cooling_kwh),
            _safe_str(cop),
            _safe_str(electrical_input_kwh),
        )
        return self.add_stage(
            chain_id, "ENERGY_INPUT_CALCULATION", data
        )

    def hash_auxiliary_energy(
        self,
        chain_id: str,
        auxiliary_kwh: float,
        auxiliary_pct: float,
        condenser_type: Optional[str] = None,
        pump_kwh: Optional[float] = None,
        fan_kwh: Optional[float] = None,
        controls_kwh: Optional[float] = None,
    ) -> str:
        """Hash auxiliary/parasitic energy calculation (Stage 6).

        Records the auxiliary energy consumption for cooling system
        components including condenser fans, pumps, and controls.

        Typical auxiliary percentages:
        - Air-cooled condenser: 5-15% of compressor input
        - Water-cooled condenser (cooling tower): 3-8%
        - Chilled water pumps: 2-5%
        - Controls and instrumentation: 0.5-1%

        Args:
            chain_id: Provenance chain identifier.
            auxiliary_kwh: Total auxiliary energy in kWh.
                Must be non-negative.
            auxiliary_pct: Auxiliary energy as percentage of
                compressor input (0-100).
            condenser_type: Optional condenser type
                (``"air_cooled"``, ``"water_cooled"``).
            pump_kwh: Optional pump energy component in kWh.
            fan_kwh: Optional fan energy component in kWh.
            controls_kwh: Optional controls energy component in kWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if auxiliary_kwh < 0:
            raise ValueError(
                f"auxiliary_kwh must be non-negative, "
                f"got {auxiliary_kwh}"
            )
        if auxiliary_pct < 0 or auxiliary_pct > 100:
            raise ValueError(
                f"auxiliary_pct must be in [0, 100], "
                f"got {auxiliary_pct}"
            )

        data: Dict[str, Any] = {
            "auxiliary_kwh": _safe_str(auxiliary_kwh),
            "auxiliary_pct": _safe_str(auxiliary_pct),
        }
        if condenser_type is not None:
            data["condenser_type"] = condenser_type
        if pump_kwh is not None:
            data["pump_kwh"] = _safe_str(pump_kwh)
        if fan_kwh is not None:
            data["fan_kwh"] = _safe_str(fan_kwh)
        if controls_kwh is not None:
            data["controls_kwh"] = _safe_str(controls_kwh)

        logger.info(
            "Hashing auxiliary energy: %s kWh (%s%%) condenser=%s",
            _safe_str(auxiliary_kwh),
            _safe_str(auxiliary_pct),
            condenser_type or "N/A",
        )
        return self.add_stage(chain_id, "AUXILIARY_ENERGY", data)

    def hash_grid_factor_application(
        self,
        chain_id: str,
        grid_ef: float,
        electricity_kwh: float,
        emissions_kgco2e: float,
        grid_region: Optional[str] = None,
        ef_source: Optional[str] = None,
        ef_year: Optional[int] = None,
        method: Optional[str] = None,
    ) -> str:
        """Hash grid emission factor application (Stage 7).

        Records the application of a grid electricity emission factor
        to the calculated electrical energy input. This is the core
        Scope 2 emissions calculation step.

        The formula:
            emissions_kgco2e = electricity_kwh * grid_ef

        Where grid_ef is in kg CO2e / kWh.

        Args:
            chain_id: Provenance chain identifier.
            grid_ef: Grid emission factor (kg CO2e/kWh).
                Must be non-negative.
            electricity_kwh: Electricity consumed in kWh.
                Must be non-negative.
            emissions_kgco2e: Calculated emissions in kg CO2e.
            grid_region: Optional grid region identifier.
            ef_source: Optional emission factor source (e.g.,
                ``"IEA_2024"``, ``"eGRID_2023"``, ``"DEFRA_2024"``,
                ``"residual_mix"``).
            ef_year: Optional year of the emission factor dataset.
            method: Optional accounting method (``"location_based"``,
                ``"market_based"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if grid_ef < 0:
            raise ValueError(
                f"grid_ef must be non-negative, got {grid_ef}"
            )
        if electricity_kwh < 0:
            raise ValueError(
                f"electricity_kwh must be non-negative, "
                f"got {electricity_kwh}"
            )

        data: Dict[str, Any] = {
            "grid_ef": _safe_str(grid_ef),
            "electricity_kwh": _safe_str(electricity_kwh),
            "emissions_kgco2e": _safe_str(emissions_kgco2e),
        }
        if grid_region is not None:
            data["grid_region"] = grid_region
        if ef_source is not None:
            data["ef_source"] = ef_source
        if ef_year is not None:
            data["ef_year"] = ef_year
        if method is not None:
            data["method"] = method

        logger.info(
            "Hashing grid factor: %s kWh * %s kgCO2e/kWh = %s kgCO2e "
            "region=%s",
            _safe_str(electricity_kwh),
            _safe_str(grid_ef),
            _safe_str(emissions_kgco2e),
            grid_region or "N/A",
        )
        return self.add_stage(
            chain_id, "GRID_FACTOR_APPLICATION", data
        )

    def hash_heat_source_factor_application(
        self,
        chain_id: str,
        heat_source: str,
        heat_source_ef: float,
        thermal_input_kwh: float,
        emissions_kgco2e: float,
        absorption_type: Optional[str] = None,
        cop: Optional[float] = None,
        parasitic_ratio: Optional[float] = None,
        grid_ef: Optional[float] = None,
        parasitic_emissions_kgco2e: Optional[float] = None,
    ) -> str:
        """Hash heat source emission factor application (Stage 8).

        Records the application of a heat source emission factor for
        absorption chiller calculations. Absorption chillers use
        thermal energy (steam, hot water, natural gas) instead of
        electricity as the primary energy input.

        The formula:
            thermal_emissions = thermal_input_kwh * heat_source_ef
            parasitic_emissions = parasitic_kwh * grid_ef
            total_emissions = thermal_emissions + parasitic_emissions

        Args:
            chain_id: Provenance chain identifier.
            heat_source: Heat source type (``"natural_gas"``,
                ``"steam"``, ``"hot_water"``, ``"waste_heat"``,
                ``"solar_thermal"``).
            heat_source_ef: Heat source emission factor (kg CO2e/kWh).
                Must be non-negative.
            thermal_input_kwh: Thermal energy input in kWh.
                Must be non-negative.
            emissions_kgco2e: Calculated total emissions in kg CO2e.
            absorption_type: Optional absorption type
                (``"single_effect"``, ``"double_effect"``,
                ``"triple_effect"``).
            cop: Optional COP of the absorption chiller.
            parasitic_ratio: Optional parasitic electricity ratio
                (0.0-1.0, typically 0.01-0.05 for absorption).
            grid_ef: Optional grid emission factor for parasitic
                electricity (kg CO2e/kWh).
            parasitic_emissions_kgco2e: Optional parasitic electricity
                emissions in kg CO2e.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not heat_source:
            raise ValueError("heat_source must not be empty")
        if heat_source_ef < 0:
            raise ValueError(
                f"heat_source_ef must be non-negative, "
                f"got {heat_source_ef}"
            )
        if thermal_input_kwh < 0:
            raise ValueError(
                f"thermal_input_kwh must be non-negative, "
                f"got {thermal_input_kwh}"
            )

        data: Dict[str, Any] = {
            "heat_source": heat_source,
            "heat_source_ef": _safe_str(heat_source_ef),
            "thermal_input_kwh": _safe_str(thermal_input_kwh),
            "emissions_kgco2e": _safe_str(emissions_kgco2e),
        }
        if absorption_type is not None:
            data["absorption_type"] = absorption_type
        if cop is not None:
            data["cop"] = _safe_str(cop)
        if parasitic_ratio is not None:
            data["parasitic_ratio"] = _safe_str(parasitic_ratio)
        if grid_ef is not None:
            data["grid_ef"] = _safe_str(grid_ef)
        if parasitic_emissions_kgco2e is not None:
            data["parasitic_emissions_kgco2e"] = _safe_str(
                parasitic_emissions_kgco2e
            )

        logger.info(
            "Hashing heat source factor: %s %s kWh * %s kgCO2e/kWh "
            "= %s kgCO2e",
            heat_source,
            _safe_str(thermal_input_kwh),
            _safe_str(heat_source_ef),
            _safe_str(emissions_kgco2e),
        )
        return self.add_stage(
            chain_id, "HEAT_SOURCE_FACTOR_APPLICATION", data
        )

    def hash_gas_decomposition(
        self,
        chain_id: str,
        co2_kg: float,
        ch4_kg: float,
        n2o_kg: float,
        total_co2e_kg: float,
        decomposition_source: Optional[str] = None,
    ) -> str:
        """Hash CO2e decomposition into constituent gases (Stage 9).

        Records the decomposition of total CO2-equivalent emissions
        into individual greenhouse gas components (CO2, CH4, N2O).
        The decomposition uses grid-specific or fuel-specific gas
        composition ratios.

        Args:
            chain_id: Provenance chain identifier.
            co2_kg: CO2 component in kg.
            ch4_kg: CH4 component in kg.
            n2o_kg: N2O component in kg.
            total_co2e_kg: Total CO2-equivalent in kg (for
                verification: should equal co2 + ch4*gwp + n2o*gwp).
            decomposition_source: Optional source of gas ratios
                (e.g., ``"IEA_grid_mix"``, ``"fuel_specific"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data: Dict[str, Any] = {
            "co2_kg": _safe_str(co2_kg),
            "ch4_kg": _safe_str(ch4_kg),
            "n2o_kg": _safe_str(n2o_kg),
            "total_co2e_kg": _safe_str(total_co2e_kg),
        }
        if decomposition_source is not None:
            data["decomposition_source"] = decomposition_source

        logger.info(
            "Hashing gas decomposition: CO2=%s CH4=%s N2O=%s "
            "total=%s kgCO2e",
            _safe_str(co2_kg),
            _safe_str(ch4_kg),
            _safe_str(n2o_kg),
            _safe_str(total_co2e_kg),
        )
        return self.add_stage(chain_id, "GAS_DECOMPOSITION", data)

    def hash_gwp_application(
        self,
        chain_id: str,
        gwp_source: str,
        gwp_ch4: float,
        gwp_n2o: float,
        ch4_kg: float,
        n2o_kg: float,
        ch4_co2e_kg: float,
        n2o_co2e_kg: float,
        total_co2e_kg: float,
    ) -> str:
        """Hash GWP factor application (Stage 10).

        Records the application of Global Warming Potential factors
        to convert CH4 and N2O mass emissions into CO2-equivalent.

        Standard GWP values (100-year):
        - IPCC AR5: CH4 = 28, N2O = 265
        - IPCC AR6: CH4 = 27.9, N2O = 273
        - IPCC AR4: CH4 = 25, N2O = 298

        The formula:
            ch4_co2e = ch4_kg * gwp_ch4
            n2o_co2e = n2o_kg * gwp_n2o
            total_co2e = co2_kg + ch4_co2e + n2o_co2e

        Args:
            chain_id: Provenance chain identifier.
            gwp_source: GWP source (``"IPCC_AR5"``, ``"IPCC_AR6"``,
                ``"IPCC_AR4"``).
            gwp_ch4: GWP value for CH4.
            gwp_n2o: GWP value for N2O.
            ch4_kg: CH4 mass emissions in kg.
            n2o_kg: N2O mass emissions in kg.
            ch4_co2e_kg: CH4 emissions in CO2-equivalent kg.
            n2o_co2e_kg: N2O emissions in CO2-equivalent kg.
            total_co2e_kg: Total CO2-equivalent emissions in kg.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not gwp_source:
            raise ValueError("gwp_source must not be empty")

        data: Dict[str, Any] = {
            "gwp_source": gwp_source,
            "gwp_ch4": _safe_str(gwp_ch4),
            "gwp_n2o": _safe_str(gwp_n2o),
            "ch4_kg": _safe_str(ch4_kg),
            "n2o_kg": _safe_str(n2o_kg),
            "ch4_co2e_kg": _safe_str(ch4_co2e_kg),
            "n2o_co2e_kg": _safe_str(n2o_co2e_kg),
            "total_co2e_kg": _safe_str(total_co2e_kg),
        }

        logger.info(
            "Hashing GWP application: source=%s CH4_GWP=%s "
            "N2O_GWP=%s total=%s kgCO2e",
            gwp_source,
            _safe_str(gwp_ch4),
            _safe_str(gwp_n2o),
            _safe_str(total_co2e_kg),
        )
        return self.add_stage(chain_id, "GWP_APPLICATION", data)

    def hash_free_cooling_calculation(
        self,
        chain_id: str,
        source: str,
        cop: float,
        grid_ef: float,
        cooling_kwh: float,
        pump_energy_kwh: float,
        fan_energy_kwh: Optional[float] = None,
        emissions_kgco2e: float = 0.0,
        savings_kwh: Optional[float] = None,
        hours_available: Optional[float] = None,
    ) -> str:
        """Hash free cooling / economizer pump energy calculation (Stage 11).

        Records the emissions calculation for free cooling modes where
        ambient conditions allow cooling without mechanical
        refrigeration. Only pump and fan energy is consumed.

        Free cooling types:
        - Air-side economizer: uses outdoor air directly
        - Water-side economizer: uses cooling tower water directly
        - Hybrid: transitions between free and mechanical cooling

        The formula:
            total_energy = pump_energy_kwh + fan_energy_kwh
            emissions = total_energy * grid_ef

        Args:
            chain_id: Provenance chain identifier.
            source: Free cooling source (``"air_side_economizer"``,
                ``"water_side_economizer"``, ``"hybrid"``).
            cop: Effective COP of free cooling (typically 15-50).
                Must be positive.
            grid_ef: Grid emission factor (kg CO2e/kWh).
                Must be non-negative.
            cooling_kwh: Cooling output in kWh. Must be non-negative.
            pump_energy_kwh: Pump energy consumption in kWh.
                Must be non-negative.
            fan_energy_kwh: Optional fan energy consumption in kWh.
            emissions_kgco2e: Calculated emissions in kg CO2e.
            savings_kwh: Optional energy savings compared to
                mechanical cooling in kWh.
            hours_available: Optional annual hours of free cooling
                availability.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not source:
            raise ValueError("source must not be empty")
        if cop <= 0:
            raise ValueError(
                f"cop must be positive, got {cop}"
            )
        if grid_ef < 0:
            raise ValueError(
                f"grid_ef must be non-negative, got {grid_ef}"
            )
        if cooling_kwh < 0:
            raise ValueError(
                f"cooling_kwh must be non-negative, got {cooling_kwh}"
            )
        if pump_energy_kwh < 0:
            raise ValueError(
                f"pump_energy_kwh must be non-negative, "
                f"got {pump_energy_kwh}"
            )

        data: Dict[str, Any] = {
            "source": source,
            "cop": _safe_str(cop),
            "grid_ef": _safe_str(grid_ef),
            "cooling_kwh": _safe_str(cooling_kwh),
            "pump_energy_kwh": _safe_str(pump_energy_kwh),
            "emissions_kgco2e": _safe_str(emissions_kgco2e),
        }
        if fan_energy_kwh is not None:
            data["fan_energy_kwh"] = _safe_str(fan_energy_kwh)
        if savings_kwh is not None:
            data["savings_kwh"] = _safe_str(savings_kwh)
        if hours_available is not None:
            data["hours_available"] = _safe_str(hours_available)

        logger.info(
            "Hashing free cooling: source=%s cop=%s pump=%s kWh "
            "= %s kgCO2e",
            source,
            _safe_str(cop),
            _safe_str(pump_energy_kwh),
            _safe_str(emissions_kgco2e),
        )
        return self.add_stage(
            chain_id, "FREE_COOLING_CALCULATION", data
        )

    def hash_tes_temporal_shifting(
        self,
        chain_id: str,
        tes_type: str,
        cop_charge: float,
        round_trip_eff: float,
        grid_ef_charge: float,
        grid_ef_peak: float,
        capacity_kwh: float,
        charge_kwh: float,
        discharge_kwh: float,
        charge_emissions_kgco2e: float,
        avoided_peak_emissions_kgco2e: Optional[float] = None,
        net_emissions_kgco2e: Optional[float] = None,
    ) -> str:
        """Hash TES temporal emission shifting calculation (Stage 12).

        Records the Thermal Energy Storage charge/discharge emissions
        calculation. TES systems charge during off-peak hours (often
        with lower grid emission factors) and discharge during peak
        hours.

        The formula:
            charge_energy = charge_kwh / cop_charge
            charge_emissions = charge_energy * grid_ef_charge
            peak_avoided = discharge_kwh / cop_mechanical * grid_ef_peak
            net_benefit = peak_avoided - charge_emissions

        TES types:
        - Ice storage (latent heat at 0 deg C)
        - Chilled water storage (sensible heat, 4-6 deg C)
        - Eutectic salt storage (phase change materials)

        Args:
            chain_id: Provenance chain identifier.
            tes_type: TES technology type (``"ice"``,
                ``"chilled_water"``, ``"eutectic"``).
            cop_charge: COP during charging period. Must be positive.
            round_trip_eff: Round-trip efficiency (0.0-1.0).
            grid_ef_charge: Grid EF during charging period
                (kg CO2e/kWh). Must be non-negative.
            grid_ef_peak: Grid EF during peak discharge period
                (kg CO2e/kWh). Must be non-negative.
            capacity_kwh: TES capacity in kWh-thermal.
                Must be non-negative.
            charge_kwh: Energy used for charging in kWh.
                Must be non-negative.
            discharge_kwh: Cooling energy discharged in kWh.
                Must be non-negative.
            charge_emissions_kgco2e: Emissions from charging in
                kg CO2e.
            avoided_peak_emissions_kgco2e: Optional emissions avoided
                by not running mechanical cooling during peak.
            net_emissions_kgco2e: Optional net emissions after
                accounting for temporal shifting benefit.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not tes_type:
            raise ValueError("tes_type must not be empty")
        if cop_charge <= 0:
            raise ValueError(
                f"cop_charge must be positive, got {cop_charge}"
            )
        if round_trip_eff <= 0 or round_trip_eff > 1.0:
            raise ValueError(
                f"round_trip_eff must be in (0.0, 1.0], "
                f"got {round_trip_eff}"
            )
        if grid_ef_charge < 0:
            raise ValueError(
                f"grid_ef_charge must be non-negative, "
                f"got {grid_ef_charge}"
            )
        if grid_ef_peak < 0:
            raise ValueError(
                f"grid_ef_peak must be non-negative, "
                f"got {grid_ef_peak}"
            )

        data: Dict[str, Any] = {
            "tes_type": tes_type,
            "cop_charge": _safe_str(cop_charge),
            "round_trip_eff": _safe_str(round_trip_eff),
            "grid_ef_charge": _safe_str(grid_ef_charge),
            "grid_ef_peak": _safe_str(grid_ef_peak),
            "capacity_kwh": _safe_str(capacity_kwh),
            "charge_kwh": _safe_str(charge_kwh),
            "discharge_kwh": _safe_str(discharge_kwh),
            "charge_emissions_kgco2e": _safe_str(
                charge_emissions_kgco2e
            ),
        }
        if avoided_peak_emissions_kgco2e is not None:
            data["avoided_peak_emissions_kgco2e"] = _safe_str(
                avoided_peak_emissions_kgco2e
            )
        if net_emissions_kgco2e is not None:
            data["net_emissions_kgco2e"] = _safe_str(
                net_emissions_kgco2e
            )

        logger.info(
            "Hashing TES temporal shifting: type=%s charge=%s kWh "
            "discharge=%s kWh charge_emissions=%s kgCO2e",
            tes_type,
            _safe_str(charge_kwh),
            _safe_str(discharge_kwh),
            _safe_str(charge_emissions_kgco2e),
        )
        return self.add_stage(
            chain_id, "TES_TEMPORAL_SHIFTING", data
        )

    def hash_district_loss_adjustment(
        self,
        chain_id: str,
        region: str,
        distribution_loss_pct: float,
        pump_energy_kwh: float,
        grid_ef: float,
        cooling_kwh: float,
        adjusted_cooling_kwh: float,
        pump_emissions_kgco2e: float,
        total_adjusted_emissions_kgco2e: float,
        network_id: Optional[str] = None,
        pipe_length_km: Optional[float] = None,
    ) -> str:
        """Hash district cooling distribution loss adjustment (Stage 13).

        Records the adjustment for distribution network losses in
        district cooling systems. Losses include thermal gains in
        distribution pipes and pump energy for circulation.

        The formula:
            adjusted_cooling = cooling_kwh / (1 - loss_pct/100)
            pump_emissions = pump_energy_kwh * grid_ef
            total_adjusted = base_emissions * (1 + loss_pct/100)
                           + pump_emissions

        Typical distribution losses:
        - Modern insulated: 2-5%
        - Older networks: 5-15%
        - Long-distance: 10-20%

        Args:
            chain_id: Provenance chain identifier.
            region: Region or network identifier.
            distribution_loss_pct: Distribution loss percentage (0-50).
            pump_energy_kwh: Distribution pump energy in kWh.
                Must be non-negative.
            grid_ef: Grid emission factor for pump energy
                (kg CO2e/kWh). Must be non-negative.
            cooling_kwh: Original cooling demand in kWh.
                Must be non-negative.
            adjusted_cooling_kwh: Cooling demand adjusted for losses
                in kWh.
            pump_emissions_kgco2e: Pump energy emissions in kg CO2e.
            total_adjusted_emissions_kgco2e: Total adjusted emissions
                in kg CO2e.
            network_id: Optional district cooling network identifier.
            pipe_length_km: Optional distribution pipe length in km.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not region:
            raise ValueError("region must not be empty")
        if distribution_loss_pct < 0 or distribution_loss_pct > 50:
            raise ValueError(
                f"distribution_loss_pct must be in [0, 50], "
                f"got {distribution_loss_pct}"
            )
        if pump_energy_kwh < 0:
            raise ValueError(
                f"pump_energy_kwh must be non-negative, "
                f"got {pump_energy_kwh}"
            )
        if grid_ef < 0:
            raise ValueError(
                f"grid_ef must be non-negative, got {grid_ef}"
            )

        data: Dict[str, Any] = {
            "region": region,
            "distribution_loss_pct": _safe_str(distribution_loss_pct),
            "pump_energy_kwh": _safe_str(pump_energy_kwh),
            "grid_ef": _safe_str(grid_ef),
            "cooling_kwh": _safe_str(cooling_kwh),
            "adjusted_cooling_kwh": _safe_str(adjusted_cooling_kwh),
            "pump_emissions_kgco2e": _safe_str(pump_emissions_kgco2e),
            "total_adjusted_emissions_kgco2e": _safe_str(
                total_adjusted_emissions_kgco2e
            ),
        }
        if network_id is not None:
            data["network_id"] = network_id
        if pipe_length_km is not None:
            data["pipe_length_km"] = _safe_str(pipe_length_km)

        logger.info(
            "Hashing district loss adjustment: region=%s loss=%s%% "
            "pump=%s kWh adjusted=%s kgCO2e",
            region,
            _safe_str(distribution_loss_pct),
            _safe_str(pump_energy_kwh),
            _safe_str(total_adjusted_emissions_kgco2e),
        )
        return self.add_stage(
            chain_id, "DISTRICT_LOSS_ADJUSTMENT", data
        )

    def hash_refrigerant_leakage(
        self,
        chain_id: str,
        refrigerant: str,
        charge_kg: float,
        leak_rate: float,
        gwp: float,
        annual_leakage_kg: Optional[float] = None,
        leakage_emissions_kgco2e: Optional[float] = None,
        is_informational: bool = True,
    ) -> str:
        """Hash refrigerant leakage calculation (Stage 14).

        Records the informational refrigerant leakage emissions for
        the cooling system. Per GHG Protocol, direct refrigerant
        emissions are Scope 1 (not Scope 2), so this stage is
        informational only for the Cooling Purchase agent.

        The formula:
            annual_leakage_kg = charge_kg * leak_rate
            leakage_emissions = annual_leakage_kg * gwp

        Typical leak rates:
        - Hermetic systems: 0.5-2%/year
        - Semi-hermetic: 2-5%/year
        - Open drive: 5-15%/year
        - District cooling: 1-3%/year

        Args:
            chain_id: Provenance chain identifier.
            refrigerant: Refrigerant type (``"R-134a"``,
                ``"R-410A"``, ``"R-1234ze"``, ``"R-717"``,
                ``"R-744"``, ``"R-290"``).
            charge_kg: System refrigerant charge in kg.
                Must be non-negative.
            leak_rate: Annual leak rate (0.0-1.0, e.g., 0.02 = 2%).
                Must be in [0.0, 1.0].
            gwp: Global Warming Potential of the refrigerant.
                Must be non-negative.
            annual_leakage_kg: Optional calculated annual leakage
                in kg.
            leakage_emissions_kgco2e: Optional calculated leakage
                emissions in kg CO2e.
            is_informational: Whether this is informational only
                (Scope 1, not included in Scope 2 total).
                Defaults to True.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not refrigerant:
            raise ValueError("refrigerant must not be empty")
        if charge_kg < 0:
            raise ValueError(
                f"charge_kg must be non-negative, got {charge_kg}"
            )
        if leak_rate < 0 or leak_rate > 1.0:
            raise ValueError(
                f"leak_rate must be in [0.0, 1.0], got {leak_rate}"
            )
        if gwp < 0:
            raise ValueError(
                f"gwp must be non-negative, got {gwp}"
            )

        data: Dict[str, Any] = {
            "refrigerant": refrigerant,
            "charge_kg": _safe_str(charge_kg),
            "leak_rate": _safe_str(leak_rate),
            "gwp": _safe_str(gwp),
            "is_informational": is_informational,
        }
        if annual_leakage_kg is not None:
            data["annual_leakage_kg"] = _safe_str(annual_leakage_kg)
        if leakage_emissions_kgco2e is not None:
            data["leakage_emissions_kgco2e"] = _safe_str(
                leakage_emissions_kgco2e
            )

        logger.info(
            "Hashing refrigerant leakage: %s charge=%s kg "
            "rate=%s gwp=%s (informational=%s)",
            refrigerant,
            _safe_str(charge_kg),
            _safe_str(leak_rate),
            _safe_str(gwp),
            is_informational,
        )
        return self.add_stage(
            chain_id, "REFRIGERANT_LEAKAGE", data
        )

    def hash_uncertainty_quantification(
        self,
        chain_id: str,
        calc_id: str,
        mean: float,
        std_dev: float,
        ci_lower: float,
        ci_upper: float,
        method: str,
        iterations: Optional[int] = None,
        confidence_level: Optional[float] = None,
        data_quality_score: Optional[float] = None,
        cop_uncertainty_pct: Optional[float] = None,
        ef_uncertainty_pct: Optional[float] = None,
    ) -> str:
        """Hash uncertainty analysis result (Stage 15).

        Records the uncertainty quantification results for the cooling
        emission calculation. Cooling calculations have uncertainty
        from COP variability, grid emission factor uncertainty, and

from greenlang.schemas import utcnow
        measurement errors.

        Supported methods:
        - ``"monte_carlo"``: Monte Carlo simulation
        - ``"analytical"``: Error propagation formula
        - ``"ipcc_tier1"``: IPCC default uncertainty ranges
        - ``"taylor_series"``: Taylor series approximation

        Typical uncertainty ranges for cooling:
        - COP uncertainty: +/- 5-15%
        - Grid EF uncertainty: +/- 5-20%
        - Activity data: +/- 2-10%
        - Combined: +/- 10-30%

        Args:
            chain_id: Provenance chain identifier.
            calc_id: Calculation identifier for cross-reference.
            mean: Mean emission estimate (kg CO2e).
            std_dev: Standard deviation (kg CO2e).
            ci_lower: Lower confidence interval bound (kg CO2e).
            ci_upper: Upper confidence interval bound (kg CO2e).
            method: Uncertainty quantification method.
            iterations: Optional number of Monte Carlo iterations.
            confidence_level: Optional confidence level (e.g., 0.95).
            data_quality_score: Optional data quality score (0.0-1.0).
            cop_uncertainty_pct: Optional COP uncertainty percentage.
            ef_uncertainty_pct: Optional EF uncertainty percentage.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not method:
            raise ValueError("method must not be empty")
        if ci_lower > ci_upper:
            raise ValueError(
                f"ci_lower ({ci_lower}) must not exceed "
                f"ci_upper ({ci_upper})"
            )

        data: Dict[str, Any] = {
            "calc_id": calc_id,
            "mean": _safe_str(mean),
            "std_dev": _safe_str(std_dev),
            "ci_lower": _safe_str(ci_lower),
            "ci_upper": _safe_str(ci_upper),
            "method": method,
        }
        if iterations is not None:
            data["iterations"] = iterations
        if confidence_level is not None:
            data["confidence_level"] = _safe_str(confidence_level)
        if data_quality_score is not None:
            data["data_quality_score"] = _safe_str(data_quality_score)
        if cop_uncertainty_pct is not None:
            data["cop_uncertainty_pct"] = _safe_str(
                cop_uncertainty_pct
            )
        if ef_uncertainty_pct is not None:
            data["ef_uncertainty_pct"] = _safe_str(
                ef_uncertainty_pct
            )

        logger.info(
            "Hashing uncertainty: calc=%s mean=%s std=%s "
            "CI=[%s, %s] method=%s",
            calc_id,
            _safe_str(mean),
            _safe_str(std_dev),
            _safe_str(ci_lower),
            _safe_str(ci_upper),
            method,
        )
        return self.add_stage(
            chain_id, "UNCERTAINTY_QUANTIFICATION", data
        )

    def hash_compliance_check(
        self,
        chain_id: str,
        calc_id: str,
        framework: str,
        status: str,
        findings_count: int,
        findings: Optional[List[str]] = None,
        score: Optional[float] = None,
        checked_rules: Optional[int] = None,
        passed_rules: Optional[int] = None,
    ) -> str:
        """Hash regulatory compliance check result (Stage 16).

        Records the result of verifying the cooling emission calculation
        against a regulatory framework's requirements.

        Supported frameworks:
        - ``"GHG_PROTOCOL"``: GHG Protocol Corporate Standard
        - ``"GHG_PROTOCOL_SCOPE2"``: GHG Protocol Scope 2 Guidance
        - ``"ISO_14064"``: ISO 14064-1:2018
        - ``"CSRD_ESRS_E1"``: CSRD / ESRS E1
        - ``"ASHRAE_90_1"``: ASHRAE Standard 90.1
        - ``"EPA_MRR"``: EPA Mandatory Reporting Rule
        - ``"EU_ETS"``: EU Emissions Trading System
        - ``"CDP"``: CDP Climate Change Questionnaire

        Args:
            chain_id: Provenance chain identifier.
            calc_id: Calculation identifier for cross-reference.
            framework: Regulatory framework identifier.
            status: Compliance status (``"COMPLIANT"``,
                ``"NON_COMPLIANT"``, ``"PARTIAL"``, ``"NOT_APPLICABLE"``).
            findings_count: Number of compliance findings.
            findings: Optional list of finding descriptions.
            score: Optional compliance score (0.0-1.0).
            checked_rules: Optional total rules checked.
            passed_rules: Optional rules that passed.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not framework:
            raise ValueError("framework must not be empty")
        if not status:
            raise ValueError("status must not be empty")

        data: Dict[str, Any] = {
            "calc_id": calc_id,
            "framework": framework,
            "status": status,
            "findings_count": findings_count,
        }
        if findings is not None:
            data["findings"] = findings
        if score is not None:
            data["score"] = _safe_str(score)
        if checked_rules is not None:
            data["checked_rules"] = checked_rules
        if passed_rules is not None:
            data["passed_rules"] = passed_rules

        logger.info(
            "Hashing compliance check: calc=%s framework=%s "
            "status=%s findings=%d",
            calc_id,
            framework,
            status,
            findings_count,
        )
        return self.add_stage(chain_id, "COMPLIANCE_CHECK", data)

    def hash_aggregation(
        self,
        chain_id: str,
        group_key: str,
        count: int,
        total_emissions_kgco2e: float,
        total_cooling_kwh: Optional[float] = None,
        total_electricity_kwh: Optional[float] = None,
        avg_cop: Optional[float] = None,
        weighted_ef: Optional[float] = None,
    ) -> str:
        """Hash results aggregation (Stage 17).

        Records the aggregation of multiple calculation results into
        summary groups (by technology, facility, time period, etc.).

        Args:
            chain_id: Provenance chain identifier.
            group_key: Aggregation group key (e.g.,
                ``"facility:FAC-001"``, ``"technology:centrifugal"``,
                ``"month:2026-01"``).
            count: Number of records aggregated.
            total_emissions_kgco2e: Total aggregated emissions in
                kg CO2e.
            total_cooling_kwh: Optional total cooling output in kWh.
            total_electricity_kwh: Optional total electricity input
                in kWh.
            avg_cop: Optional average COP across aggregated records.
            weighted_ef: Optional weighted average emission factor.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not group_key:
            raise ValueError("group_key must not be empty")
        if count < 0:
            raise ValueError(
                f"count must be non-negative, got {count}"
            )

        data: Dict[str, Any] = {
            "group_key": group_key,
            "count": count,
            "total_emissions_kgco2e": _safe_str(
                total_emissions_kgco2e
            ),
        }
        if total_cooling_kwh is not None:
            data["total_cooling_kwh"] = _safe_str(total_cooling_kwh)
        if total_electricity_kwh is not None:
            data["total_electricity_kwh"] = _safe_str(
                total_electricity_kwh
            )
        if avg_cop is not None:
            data["avg_cop"] = _safe_str(avg_cop)
        if weighted_ef is not None:
            data["weighted_ef"] = _safe_str(weighted_ef)

        logger.info(
            "Hashing aggregation: group=%s count=%d total=%s kgCO2e",
            group_key,
            count,
            _safe_str(total_emissions_kgco2e),
        )
        return self.add_stage(chain_id, "AGGREGATION", data)

    def hash_batch_assembly(
        self,
        chain_id: str,
        batch_id: str,
        total: int,
        completed: int,
        failed: int,
        total_emissions_kgco2e: float,
        total_cooling_kwh: Optional[float] = None,
        processing_time_ms: Optional[float] = None,
    ) -> str:
        """Hash batch result assembly (Stage 18).

        Records the assembly of batch calculation results, tracking
        completion status and aggregate emissions.

        Args:
            chain_id: Provenance chain identifier.
            batch_id: Batch identifier.
            total: Total number of calculations in the batch.
            completed: Number of successfully completed calculations.
            failed: Number of failed calculations.
            total_emissions_kgco2e: Total batch emissions in kg CO2e.
            total_cooling_kwh: Optional total cooling demand in kWh.
            processing_time_ms: Optional total processing time in ms.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not batch_id:
            raise ValueError("batch_id must not be empty")
        if total < 0:
            raise ValueError(
                f"total must be non-negative, got {total}"
            )
        if completed < 0:
            raise ValueError(
                f"completed must be non-negative, got {completed}"
            )
        if failed < 0:
            raise ValueError(
                f"failed must be non-negative, got {failed}"
            )

        data: Dict[str, Any] = {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "failed": failed,
            "total_emissions_kgco2e": _safe_str(
                total_emissions_kgco2e
            ),
        }
        if total_cooling_kwh is not None:
            data["total_cooling_kwh"] = _safe_str(total_cooling_kwh)
        if processing_time_ms is not None:
            data["processing_time_ms"] = _safe_str(processing_time_ms)

        logger.info(
            "Hashing batch assembly: batch=%s total=%d completed=%d "
            "failed=%d emissions=%s kgCO2e",
            batch_id,
            total,
            completed,
            failed,
            _safe_str(total_emissions_kgco2e),
        )
        return self.add_stage(chain_id, "BATCH_ASSEMBLY", data)

    def hash_result_finalization(
        self,
        chain_id: str,
        total_co2e_kg: float,
        total_co2e_tonnes: Optional[float] = None,
        validation_status: str = "PASS",
        cooling_technology: Optional[str] = None,
        total_cooling_kwh: Optional[float] = None,
        total_electricity_kwh: Optional[float] = None,
        effective_cop: Optional[float] = None,
        grid_ef_used: Optional[float] = None,
        accounting_method: Optional[str] = None,
        biogenic_co2_kg: Optional[float] = None,
        data_quality_score: Optional[float] = None,
    ) -> str:
        """Hash final result (Stage 19).

        Records the final assembled result of the Cooling Purchase
        emission calculation, including totals, validation status,
        and summary statistics.

        Args:
            chain_id: Provenance chain identifier.
            total_co2e_kg: Total CO2e emissions in kg.
            total_co2e_tonnes: Optional total in metric tonnes.
            validation_status: Validation result (``"PASS"`` or
                ``"FAIL"``).
            cooling_technology: Optional cooling technology type.
            total_cooling_kwh: Optional total cooling output in kWh.
            total_electricity_kwh: Optional total electricity input
                in kWh.
            effective_cop: Optional effective COP (cooling/electricity).
            grid_ef_used: Optional grid emission factor used.
            accounting_method: Optional accounting method
                (``"location_based"``, ``"market_based"``).
            biogenic_co2_kg: Optional biogenic CO2 component in kg.
            data_quality_score: Optional data quality score (0.0-1.0).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data: Dict[str, Any] = {
            "total_co2e_kg": _safe_str(total_co2e_kg),
            "validation_status": validation_status,
        }
        if total_co2e_tonnes is not None:
            data["total_co2e_tonnes"] = _safe_str(total_co2e_tonnes)
        if cooling_technology is not None:
            data["cooling_technology"] = cooling_technology
        if total_cooling_kwh is not None:
            data["total_cooling_kwh"] = _safe_str(total_cooling_kwh)
        if total_electricity_kwh is not None:
            data["total_electricity_kwh"] = _safe_str(
                total_electricity_kwh
            )
        if effective_cop is not None:
            data["effective_cop"] = _safe_str(effective_cop)
        if grid_ef_used is not None:
            data["grid_ef_used"] = _safe_str(grid_ef_used)
        if accounting_method is not None:
            data["accounting_method"] = accounting_method
        if biogenic_co2_kg is not None:
            data["biogenic_co2_kg"] = _safe_str(biogenic_co2_kg)
        if data_quality_score is not None:
            data["data_quality_score"] = _safe_str(data_quality_score)

        logger.info(
            "Hashing result finalization: total=%s kgCO2e status=%s "
            "tech=%s",
            _safe_str(total_co2e_kg),
            validation_status,
            cooling_technology or "N/A",
        )
        return self.add_stage(
            chain_id, "RESULT_FINALIZATION", data
        )

    # ------------------------------------------------------------------
    # Convenience hash helpers (standalone data hashing)
    # ------------------------------------------------------------------

    def hash_electric_chiller_input(
        self,
        technology: str,
        cop: float,
        iplv: float,
        grid_ef: float,
        cooling_kwh: float,
        auxiliary_pct: float,
        condenser_type: str,
    ) -> str:
        """Compute standalone hash for electric chiller input parameters.

        This is a convenience method that produces a standalone SHA-256
        hash without appending to any chain. Useful for deduplication
        or pre-flight checks.

        Args:
            technology: Chiller technology type.
            cop: Coefficient of Performance.
            iplv: Integrated Part Load Value.
            grid_ef: Grid emission factor (kg CO2e/kWh).
            cooling_kwh: Cooling demand in kWh.
            auxiliary_pct: Auxiliary energy percentage.
            condenser_type: Condenser type.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "technology": technology,
            "cop": _safe_str(cop),
            "iplv": _safe_str(iplv),
            "grid_ef": _safe_str(grid_ef),
            "cooling_kwh": _safe_str(cooling_kwh),
            "auxiliary_pct": _safe_str(auxiliary_pct),
            "condenser_type": condenser_type,
        }
        return self.compute_hash(data)

    def hash_absorption_input(
        self,
        absorption_type: str,
        heat_source: str,
        cop: float,
        parasitic_ratio: float,
        grid_ef: float,
        heat_source_ef: float,
    ) -> str:
        """Compute standalone hash for absorption chiller input parameters.

        Args:
            absorption_type: Absorption type (single/double/triple).
            heat_source: Heat source type.
            cop: Coefficient of Performance.
            parasitic_ratio: Parasitic electricity ratio.
            grid_ef: Grid emission factor (kg CO2e/kWh).
            heat_source_ef: Heat source emission factor (kg CO2e/kWh).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "absorption_type": absorption_type,
            "heat_source": heat_source,
            "cop": _safe_str(cop),
            "parasitic_ratio": _safe_str(parasitic_ratio),
            "grid_ef": _safe_str(grid_ef),
            "heat_source_ef": _safe_str(heat_source_ef),
        }
        return self.compute_hash(data)

    def hash_free_cooling_input(
        self,
        source: str,
        cop: float,
        grid_ef: float,
        cooling_kwh: float,
    ) -> str:
        """Compute standalone hash for free cooling input parameters.

        Args:
            source: Free cooling source type.
            cop: Effective COP of free cooling.
            grid_ef: Grid emission factor (kg CO2e/kWh).
            cooling_kwh: Cooling demand in kWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "source": source,
            "cop": _safe_str(cop),
            "grid_ef": _safe_str(grid_ef),
            "cooling_kwh": _safe_str(cooling_kwh),
        }
        return self.compute_hash(data)

    def hash_tes_input(
        self,
        tes_type: str,
        cop_charge: float,
        round_trip_eff: float,
        grid_ef_charge: float,
        grid_ef_peak: float,
        capacity_kwh: float,
    ) -> str:
        """Compute standalone hash for TES input parameters.

        Args:
            tes_type: TES technology type.
            cop_charge: COP during charging.
            round_trip_eff: Round-trip efficiency.
            grid_ef_charge: Grid EF during charging (kg CO2e/kWh).
            grid_ef_peak: Grid EF during peak (kg CO2e/kWh).
            capacity_kwh: TES capacity in kWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "tes_type": tes_type,
            "cop_charge": _safe_str(cop_charge),
            "round_trip_eff": _safe_str(round_trip_eff),
            "grid_ef_charge": _safe_str(grid_ef_charge),
            "grid_ef_peak": _safe_str(grid_ef_peak),
            "capacity_kwh": _safe_str(capacity_kwh),
        }
        return self.compute_hash(data)

    def hash_district_cooling_input(
        self,
        region: str,
        distribution_loss: float,
        pump_energy: float,
        grid_ef: float,
        cooling_kwh: float,
    ) -> str:
        """Compute standalone hash for district cooling input parameters.

        Args:
            region: Region or network identifier.
            distribution_loss: Distribution loss percentage.
            pump_energy: Pump energy in kWh.
            grid_ef: Grid emission factor (kg CO2e/kWh).
            cooling_kwh: Cooling demand in kWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "region": region,
            "distribution_loss": _safe_str(distribution_loss),
            "pump_energy": _safe_str(pump_energy),
            "grid_ef": _safe_str(grid_ef),
            "cooling_kwh": _safe_str(cooling_kwh),
        }
        return self.compute_hash(data)

    def hash_calculation_result(
        self,
        calc_id: str,
        emissions_kgco2e: float,
        cop_used: float,
        energy_input: float,
    ) -> str:
        """Compute standalone hash for a calculation result.

        Args:
            calc_id: Calculation identifier.
            emissions_kgco2e: Total emissions in kg CO2e.
            cop_used: COP value used in calculation.
            energy_input: Energy input in kWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "calc_id": calc_id,
            "emissions_kgco2e": _safe_str(emissions_kgco2e),
            "cop_used": _safe_str(cop_used),
            "energy_input": _safe_str(energy_input),
        }
        return self.compute_hash(data)

    def hash_uncertainty_result(
        self,
        calc_id: str,
        mean: float,
        std_dev: float,
        ci_lower: float,
        ci_upper: float,
        iterations: int,
    ) -> str:
        """Compute standalone hash for an uncertainty result.

        Args:
            calc_id: Calculation identifier.
            mean: Mean emission estimate.
            std_dev: Standard deviation.
            ci_lower: Lower confidence interval bound.
            ci_upper: Upper confidence interval bound.
            iterations: Number of Monte Carlo iterations.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "calc_id": calc_id,
            "mean": _safe_str(mean),
            "std_dev": _safe_str(std_dev),
            "ci_lower": _safe_str(ci_lower),
            "ci_upper": _safe_str(ci_upper),
            "iterations": iterations,
        }
        return self.compute_hash(data)

    def hash_compliance_result(
        self,
        calc_id: str,
        framework: str,
        status: str,
        findings_count: int,
    ) -> str:
        """Compute standalone hash for a compliance result.

        Args:
            calc_id: Calculation identifier.
            framework: Regulatory framework identifier.
            status: Compliance status.
            findings_count: Number of findings.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "calc_id": calc_id,
            "framework": framework,
            "status": status,
            "findings_count": findings_count,
        }
        return self.compute_hash(data)

    def hash_batch_result(
        self,
        batch_id: str,
        total: int,
        completed: int,
        total_emissions: float,
    ) -> str:
        """Compute standalone hash for a batch result.

        Args:
            batch_id: Batch identifier.
            total: Total calculations in batch.
            completed: Completed calculations.
            total_emissions: Total emissions in kg CO2e.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "total_emissions": _safe_str(total_emissions),
        }
        return self.compute_hash(data)

    def hash_aggregation_result(
        self,
        group_key: str,
        count: int,
        total_emissions: float,
    ) -> str:
        """Compute standalone hash for an aggregation result.

        Args:
            group_key: Aggregation group key.
            count: Number of records in group.
            total_emissions: Total emissions in kg CO2e.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data = {
            "group_key": group_key,
            "count": count,
            "total_emissions": _safe_str(total_emissions),
        }
        return self.compute_hash(data)

    # ------------------------------------------------------------------
    # Batch and merge operations
    # ------------------------------------------------------------------

    def merge_chains(
        self,
        target_chain_id: str,
        source_chain_id: str,
    ) -> str:
        """Merge a source chain reference into a target chain.

        Appends a merge record to the target chain that captures the
        chain hash of the source chain. This is used in batch processing
        scenarios where multiple independent calculations are combined
        into a single reporting chain.

        The merge does not copy individual entries. It records the source
        chain's cumulative hash as a reference, creating a compact proof
        that the source chain's state was incorporated at this point.

        Args:
            target_chain_id: Chain to merge into.
            source_chain_id: Chain whose hash will be recorded.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the
            merge entry.

        Raises:
            ValueError: If either chain does not exist.
        """
        if not target_chain_id:
            raise ValueError("target_chain_id must not be empty")
        if not source_chain_id:
            raise ValueError("source_chain_id must not be empty")

        with self._instance_lock:
            if target_chain_id not in self._chains:
                raise ValueError(
                    f"Target chain '{target_chain_id}' does not exist."
                )
            if source_chain_id not in self._chains:
                raise ValueError(
                    f"Source chain '{source_chain_id}' does not exist."
                )

            source_chain = self._chains[source_chain_id]
            source_hash = (
                source_chain[-1].hash_value if source_chain else ""
            )
            source_length = len(source_chain)
            source_summary = self._get_stage_summary(source_chain)

        merge_data = {
            "source_chain_id": source_chain_id,
            "source_chain_hash": source_hash,
            "source_chain_length": source_length,
            "source_stage_summary": source_summary,
            "merge_timestamp": utcnow().isoformat(),
        }

        logger.info(
            "Merging chain '%s' into '%s' (source_hash=%s len=%d)",
            source_chain_id,
            target_chain_id,
            source_hash[:16] if source_hash else "empty",
            source_length,
        )

        # Use a generic stage for merge (not in VALID_STAGES, but
        # add_stage permits unknown stages with a warning)
        return self.add_stage(
            target_chain_id, "CHAIN_MERGE", merge_data
        )

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def delete_chain(self, chain_id: str) -> bool:
        """Delete a provenance chain.

        Removes the chain and all its entries from memory. This is an
        irreversible operation. Sealed chains can also be deleted.

        Args:
            chain_id: Identifier of the chain to delete.

        Returns:
            True if the chain was deleted, False if it did not exist.
        """
        with self._instance_lock:
            if chain_id not in self._chains:
                return False

            del self._chains[chain_id]
            self._sealed.discard(chain_id)
            self._chain_creation_times.pop(chain_id, None)

        logger.info(
            "Deleted provenance chain: chain_id=%s", chain_id
        )
        return True

    def clear_all(self) -> None:
        """Clear all provenance chains and reset to empty state.

        Removes all chains, sealed markers, and creation timestamps.
        After calling this method, the provenance tracker behaves as
        if newly constructed. Intended for testing or starting a new
        calculation batch.

        Example:
            >>> prov = CoolingPurchaseProvenance()
            >>> prov.create_chain("test-1")
            'test-1'
            >>> prov.clear_all()
            >>> assert prov.chain_count() == 0
        """
        with self._instance_lock:
            self._chains.clear()
            self._sealed.clear()
            self._chain_creation_times.clear()

        logger.info(
            "%s provenance tracker cleared (all chains removed)",
            self.AGENT_ID,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_stage_summary(
        self, chain: List[ProvenanceEntry],
    ) -> Dict[str, int]:
        """Compute stage occurrence counts for a chain.

        Args:
            chain: List of ProvenanceEntry instances.

        Returns:
            Dictionary mapping stage names to occurrence counts.
        """
        summary: Dict[str, int] = {}
        for entry in chain:
            summary[entry.stage] = summary.get(entry.stage, 0) + 1
        return summary

    def _evict_chains_if_needed(self) -> None:
        """Evict oldest non-sealed chains if at capacity.

        Must be called while holding ``self._instance_lock``. Removes
        non-sealed chains in creation order to maintain the max_chains
        constraint.
        """
        if len(self._chains) < self._max_chains:
            return

        # Find non-sealed chains sorted by creation time
        eviction_candidates = [
            cid for cid in self._chains
            if cid not in self._sealed
        ]

        # Sort by creation time (oldest first)
        eviction_candidates.sort(
            key=lambda cid: self._chain_creation_times.get(cid, "")
        )

        # Evict enough to make room
        evict_count = len(self._chains) - self._max_chains + 1
        evicted = 0
        for cid in eviction_candidates:
            if evicted >= evict_count:
                break
            del self._chains[cid]
            self._chain_creation_times.pop(cid, None)
            evicted += 1
            logger.warning(
                "Evicted oldest non-sealed chain '%s' "
                "(max_chains=%d reached)",
                cid,
                self._max_chains,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_entries(self) -> int:
        """Return the total number of entries across all chains.

        Returns:
            Integer count of all entries in all chains.
        """
        with self._instance_lock:
            return sum(
                len(chain) for chain in self._chains.values()
            )

    @property
    def sealed_count(self) -> int:
        """Return the number of sealed chains.

        Returns:
            Integer count of sealed chains.
        """
        with self._instance_lock:
            return len(self._sealed)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing chain count and total entries.
        """
        return (
            f"CoolingPurchaseProvenance("
            f"chains={self.chain_count()}, "
            f"entries={self.total_entries}, "
            f"sealed={self.sealed_count})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            Descriptive string with agent ID and chain statistics.
        """
        return (
            f"{self.AGENT_ID} Provenance Tracker: "
            f"{self.chain_count()} chains, "
            f"{self.total_entries} total entries, "
            f"{self.sealed_count} sealed"
        )

    def __len__(self) -> int:
        """Return the number of tracked chains.

        Returns:
            Integer count of all chains.
        """
        return self.chain_count()

    def __bool__(self) -> bool:
        """Return True if there is at least one chain.

        Returns:
            Boolean indicating whether any chains exist.
        """
        with self._instance_lock:
            return len(self._chains) > 0

    def __contains__(self, chain_id: str) -> bool:
        """Check if a chain_id exists.

        Args:
            chain_id: Chain identifier to look for.

        Returns:
            True if the chain exists.
        """
        with self._instance_lock:
            return chain_id in self._chains

# ---------------------------------------------------------------------------
# Module-level factory and singleton access
# ---------------------------------------------------------------------------

def create_provenance(
    max_entries_per_chain: int = (
        CoolingPurchaseProvenance.DEFAULT_MAX_ENTRIES_PER_CHAIN
    ),
    max_chains: int = CoolingPurchaseProvenance.DEFAULT_MAX_CHAINS,
) -> CoolingPurchaseProvenance:
    """Create (or return) the CoolingPurchaseProvenance singleton.

    Factory function that provides a clean entry point for obtaining
    the provenance tracker. Since CoolingPurchaseProvenance is a
    singleton, this always returns the same instance.

    Args:
        max_entries_per_chain: Maximum entries per chain before warnings.
            Only effective on first creation.
        max_chains: Maximum concurrent chains. Only effective on first
            creation.

    Returns:
        The singleton CoolingPurchaseProvenance instance.

    Example:
        >>> prov = create_provenance()
        >>> assert isinstance(prov, CoolingPurchaseProvenance)
    """
    return CoolingPurchaseProvenance(
        max_entries_per_chain=max_entries_per_chain,
        max_chains=max_chains,
    )

def get_provenance() -> CoolingPurchaseProvenance:
    """Return the process-wide singleton CoolingPurchaseProvenance.

    This is the recommended entry point for obtaining the provenance
    tracker in production code. The singleton is created lazily on
    first call and reused for all subsequent calls.

    Returns:
        The singleton CoolingPurchaseProvenance instance.

    Example:
        >>> tracker_a = get_provenance()
        >>> tracker_b = get_provenance()
        >>> assert tracker_a is tracker_b
    """
    return CoolingPurchaseProvenance()

def reset_provenance() -> None:
    """Destroy the current singleton and reset to None.

    The next call to ``get_provenance()`` will create a fresh instance.
    Intended for use in test teardown to prevent state leakage between
    test cases.

    Example:
        >>> reset_provenance()
        >>> tracker = get_provenance()  # fresh instance
    """
    CoolingPurchaseProvenance.reset()

# Aliases for backward compatibility with sibling MRV agents
reset_provenance_tracker = reset_provenance
get_provenance_tracker = get_provenance

# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def compute_standalone_hash(data: Dict[str, Any]) -> str:
    """Compute a standalone SHA-256 hash for arbitrary data.

    This function is not part of a provenance chain. It produces
    a deterministic hash from the canonical JSON representation of
    the input dictionary. Useful for pre-computing hashes for
    comparison or deduplication.

    Args:
        data: Dictionary to hash. All values must be JSON-serializable
            or convertible via ``str()``.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> h = compute_standalone_hash({"key": "value"})
        >>> assert len(h) == 64
    """
    canonical = _canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_hash(
    data: Dict[str, Any], expected_hash: str,
) -> bool:
    """Verify that a data dictionary produces the expected hash.

    Args:
        data: Dictionary to hash.
        expected_hash: Expected hex-encoded SHA-256 hash.

    Returns:
        True if the computed hash matches the expected hash.

    Example:
        >>> h = compute_standalone_hash({"key": "value"})
        >>> assert verify_hash({"key": "value"}, h) is True
    """
    computed = compute_standalone_hash(data)
    return computed == expected_hash

def compute_chain_entry_hash(
    previous_hash: str,
    stage: str,
    data: Dict[str, Any],
) -> str:
    """Compute a provenance chain entry hash without creating an entry.

    Implements the same hashing protocol as
    ``CoolingPurchaseProvenance._compute_chain_hash`` but without
    side effects. Useful for external verification of chain entries.

    Args:
        previous_hash: Previous entry's hash (empty string for first).
        stage: Stage identifier string.
        data: Dictionary of metadata.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> h = compute_chain_entry_hash(
        ...     "", "INPUT_VALIDATION", {"f": "FAC-001"}
        ... )
        >>> assert len(h) == 64
    """
    canonical = _canonical_json(data)
    payload = f"{previous_hash}|{stage}|{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def verify_chain_entries(
    entries: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Verify a list of serialized chain entries for integrity.

    Takes a list of entry dictionaries (as produced by
    ``ProvenanceEntry.to_dict()``) and verifies that the chain
    linkage and hash values are consistent.

    Args:
        entries: List of entry dictionaries with keys ``stage``,
            ``hash_value``, ``previous_hash``, ``metadata``.

    Returns:
        Tuple of ``(is_valid, error_message)``. When the chain is
        intact, returns ``(True, None)``.

    Example:
        >>> prov = get_provenance()
        >>> cid = prov.create_chain("verify-ext-001")
        >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
        '...'
        >>> entries = prov.get_chain(cid)
        >>> valid, msg = verify_chain_entries(entries)
        >>> assert valid is True
    """
    if not entries:
        return True, None

    for i, entry in enumerate(entries):
        stage = entry.get("stage", "")
        hash_value = entry.get("hash_value", "")
        previous_hash = entry.get("previous_hash", "")
        metadata = entry.get("metadata", {})

        if not stage:
            return False, f"entry[{i}] has empty stage"
        if not hash_value:
            return False, f"entry[{i}] has empty hash_value"

        # Verify chain linkage
        if i == 0:
            if previous_hash != "":
                return (
                    False,
                    f"entry[0] previous_hash is not empty "
                    f"(got '{previous_hash[:16]}')",
                )
        else:
            prev_entry_hash = entries[i - 1].get("hash_value", "")
            if previous_hash != prev_entry_hash:
                return (
                    False,
                    f"chain break between entry[{i - 1}] and "
                    f"entry[{i}]",
                )

        # Recompute and verify hash
        canonical = _canonical_json(metadata)
        payload = f"{previous_hash}|{stage}|{canonical}"
        expected = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

        if hash_value != expected:
            return (
                False,
                f"entry[{i}] hash mismatch (stage={stage})",
            )

    return True, None

def verify_sealed_export(
    export_data: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Verify a complete exported provenance chain.

    Takes the output of ``CoolingPurchaseProvenance.export_chain()``
    and performs comprehensive verification including:

    1. Agent identity validation
    2. Chain entry integrity verification
    3. Seal status confirmation
    4. Entry count consistency

    Args:
        export_data: Dictionary produced by ``export_chain()``.

    Returns:
        Tuple of ``(is_valid, error_message)``.

    Example:
        >>> prov = get_provenance()
        >>> cid = prov.create_chain("seal-verify-001")
        >>> prov.add_stage(cid, "INPUT_VALIDATION", {"f": "1"})
        '...'
        >>> prov.seal_chain(cid)
        '...'
        >>> export = prov.export_chain(cid)
        >>> valid, msg = verify_sealed_export(export)
        >>> assert valid is True
    """
    if not isinstance(export_data, dict):
        return False, "export_data must be a dictionary"

    # Verify agent identity
    agent_id = export_data.get("agent_id", "")
    if agent_id != "AGENT-MRV-012":
        return (
            False,
            f"Expected agent_id 'AGENT-MRV-012', got '{agent_id}'",
        )

    # Verify agent code
    agent_code = export_data.get("agent_code", "")
    if agent_code != "GL-MRV-X-023":
        return (
            False,
            f"Expected agent_code 'GL-MRV-X-023', "
            f"got '{agent_code}'",
        )

    # Verify entry count consistency
    entries = export_data.get("entries", [])
    chain_length = export_data.get("chain_length", 0)
    if len(entries) != chain_length:
        return (
            False,
            f"chain_length ({chain_length}) does not match "
            f"entries count ({len(entries)})",
        )

    # Verify chain entry integrity
    valid, msg = verify_chain_entries(entries)
    if not valid:
        return False, f"Chain entry verification failed: {msg}"

    # Verify chain hash matches last entry
    if entries:
        last_hash = entries[-1].get("hash_value", "")
        chain_hash = export_data.get("chain_hash", "")
        if last_hash != chain_hash:
            return (
                False,
                f"chain_hash mismatch (last_entry={last_hash[:16]}, "
                f"chain_hash={chain_hash[:16]})",
            )

    # Verify seal status
    sealed = export_data.get("sealed", False)
    if sealed and entries:
        last_stage = entries[-1].get("stage", "")
        if last_stage != "PROVENANCE_SEALED":
            return (
                False,
                f"Chain marked sealed but last stage is "
                f"'{last_stage}', expected 'PROVENANCE_SEALED'",
            )

    return True, None
