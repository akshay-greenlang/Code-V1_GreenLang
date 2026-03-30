# -*- coding: utf-8 -*-
"""
Provenance Tracking for Steam/Heat Purchase Agent - AGENT-MRV-011

Provides SHA-256 based audit trail tracking for all Steam/Heat Purchase Agent
operations. Implements a chain of SHA-256 hashes for each calculation stage,
ensuring complete audit trail and deterministic reproducibility across steam
purchase calculations, district heating/cooling factor lookups, CHP allocation,
boiler efficiency computations, fuel-based emission factor retrieval, biogenic
CO2 separation, per-gas GHG breakdowns, GWP conversions, unit conversions,
uncertainty quantification, compliance checks, and provenance sealing.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256 via hashlib
    - Chain hashing links operations in sequence (append-only)
    - Each entry records previous_hash for tamper detection
    - JSON canonical form with sort_keys=True for reproducibility
    - Decimal values converted to string for hashing consistency
    - Complete provenance for every Steam/Heat Purchase operation

Provenance Stages (19):
    1.  REQUEST_RECEIVED        - Initial calculation request received
    2.  INPUT_VALIDATED          - Input parameters validated
    3.  FACILITY_RESOLVED        - Facility information retrieved
    4.  SUPPLIER_RESOLVED        - Steam/heat supplier info retrieved
    5.  FUEL_EF_RETRIEVED        - Fuel emission factors loaded
    6.  DH_EF_RETRIEVED          - District heating factors loaded
    7.  COOLING_PARAMS_RETRIEVED - Cooling system parameters loaded
    8.  CHP_PARAMS_RETRIEVED     - CHP parameters loaded
    9.  UNIT_CONVERTED           - Energy units converted to GJ
    10. STEAM_CALCULATED         - Steam emissions calculated
    11. HEATING_CALCULATED       - District heating emissions calculated
    12. COOLING_CALCULATED       - District cooling emissions calculated
    13. CHP_ALLOCATED            - CHP emissions allocated
    14. BIOGENIC_SEPARATED       - Biogenic CO2 separated
    15. GAS_BREAKDOWN_COMPUTED   - Per-gas breakdown with GWP applied
    16. UNCERTAINTY_QUANTIFIED   - Uncertainty analysis completed
    17. COMPLIANCE_CHECKED       - Regulatory compliance verified
    18. RESULT_ASSEMBLED         - Final result assembled
    19. PROVENANCE_SEALED        - SHA-256 chain sealed

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
    - GHG Protocol Scope 2 Guidance (2015) - Steam/Heat
    - GHG Protocol Corporate Standard (Ch. 7)
    - ISO 14064-1:2018 (Category 2)
    - CSRD / ESRS E1 (Energy-related)
    - UK SECR / Streamlined Energy and Carbon Reporting
    - EPA Mandatory Reporting Rule (40 CFR 98)
    - EU ETS (Monitoring and Reporting Regulation)
    - CDP Climate Change Questionnaire

Supported Energy Carriers:
    - Purchased Steam (saturated, superheated)
    - District Heating (hot water, medium-temperature)
    - District Cooling (chilled water, absorption)
    - Combined Heat and Power (CHP) Outputs
    - Waste-Heat Recovery Streams

Example:
    >>> from greenlang.agents.mrv.steam_heat_purchase.provenance import (
    ...     SteamHeatPurchaseProvenance, get_provenance,
    ... )
    >>> prov = get_provenance()
    >>> chain_id = prov.create_chain("calc-001")
    >>> h1 = prov.add_stage(chain_id, "REQUEST_RECEIVED", {
    ...     "facility_id": "FAC-001",
    ...     "energy_type": "steam",
    ... })
    >>> h2 = prov.add_stage(chain_id, "INPUT_VALIDATED", {
    ...     "valid": True,
    ...     "field_count": 12,
    ... })
    >>> h3 = prov.add_stage(chain_id, "STEAM_CALCULATED", {
    ...     "steam_gj": 5000.0,
    ...     "emission_factor": 66.5,
    ...     "emissions_kg": 332500.0,
    ... })
    >>> final = prov.seal_chain(chain_id)
    >>> assert len(final) == 64
    >>> assert prov.verify_chain(chain_id) is True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-011 Steam/Heat Purchase (GL-MRV-X-022)
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

from greenlang.schemas import utcnow

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
    # Stage 1: Request intake
    "REQUEST_RECEIVED",
    # Stage 2: Input validation
    "INPUT_VALIDATED",
    # Stage 3: Facility resolution
    "FACILITY_RESOLVED",
    # Stage 4: Supplier resolution
    "SUPPLIER_RESOLVED",
    # Stage 5: Fuel emission factors
    "FUEL_EF_RETRIEVED",
    # Stage 6: District heating factors
    "DH_EF_RETRIEVED",
    # Stage 7: Cooling parameters
    "COOLING_PARAMS_RETRIEVED",
    # Stage 8: CHP parameters
    "CHP_PARAMS_RETRIEVED",
    # Stage 9: Unit conversion
    "UNIT_CONVERTED",
    # Stage 10: Steam calculation
    "STEAM_CALCULATED",
    # Stage 11: Heating calculation
    "HEATING_CALCULATED",
    # Stage 12: Cooling calculation
    "COOLING_CALCULATED",
    # Stage 13: CHP allocation
    "CHP_ALLOCATED",
    # Stage 14: Biogenic separation
    "BIOGENIC_SEPARATED",
    # Stage 15: Gas breakdown
    "GAS_BREAKDOWN_COMPUTED",
    # Stage 16: Uncertainty
    "UNCERTAINTY_QUANTIFIED",
    # Stage 17: Compliance
    "COMPLIANCE_CHECKED",
    # Stage 18: Result assembly
    "RESULT_ASSEMBLED",
    # Stage 19: Seal
    "PROVENANCE_SEALED",
})

#: Canonical ordered list of 19 stages for pipeline reference.
STAGE_ORDER: List[str] = [
    "REQUEST_RECEIVED",
    "INPUT_VALIDATED",
    "FACILITY_RESOLVED",
    "SUPPLIER_RESOLVED",
    "FUEL_EF_RETRIEVED",
    "DH_EF_RETRIEVED",
    "COOLING_PARAMS_RETRIEVED",
    "CHP_PARAMS_RETRIEVED",
    "UNIT_CONVERTED",
    "STEAM_CALCULATED",
    "HEATING_CALCULATED",
    "COOLING_CALCULATED",
    "CHP_ALLOCATED",
    "BIOGENIC_SEPARATED",
    "GAS_BREAKDOWN_COMPUTED",
    "UNCERTAINTY_QUANTIFIED",
    "COMPLIANCE_CHECKED",
    "RESULT_ASSEMBLED",
    "PROVENANCE_SEALED",
]

#: Number of defined provenance stages.
STAGE_COUNT: int = len(STAGE_ORDER)

# ---------------------------------------------------------------------------
# ProvenanceEntry dataclass (frozen for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProvenanceEntry:
    """A single immutable, tamper-evident provenance record for a Steam/Heat
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
        ...     stage="STEAM_CALCULATED",
        ...     hash_value="a1b2c3...",
        ...     timestamp="2026-02-15T10:30:00+00:00",
        ...     previous_hash="d4e5f6...",
        ...     metadata={"steam_gj": 5000.0, "emissions_kg": 332500.0},
        ...     chain_id="chain-001",
        ... )
        >>> entry.stage
        'STEAM_CALCULATED'
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
# SteamHeatPurchaseProvenance (Thread-safe Singleton)
# ---------------------------------------------------------------------------

class SteamHeatPurchaseProvenance:
    """SHA-256 provenance chain tracker for Steam/Heat Purchase emission
    calculations.

    Implements a multi-chain provenance system where each calculation request
    gets its own independent chain identified by a ``chain_id``. Each chain
    is an ordered, append-only list of frozen ProvenanceEntry instances. Every
    entry's hash incorporates the previous entry's hash, creating a
    tamper-evident linked list of cryptographic digests.

    This class provides methods for the complete 19-stage provenance pipeline:

    1.  **REQUEST_RECEIVED**: Initial calculation request intake
    2.  **INPUT_VALIDATED**: Input parameter validation
    3.  **FACILITY_RESOLVED**: Facility information retrieval
    4.  **SUPPLIER_RESOLVED**: Steam/heat supplier resolution
    5.  **FUEL_EF_RETRIEVED**: Fuel emission factor loading
    6.  **DH_EF_RETRIEVED**: District heating factor loading
    7.  **COOLING_PARAMS_RETRIEVED**: Cooling system parameter loading
    8.  **CHP_PARAMS_RETRIEVED**: CHP parameter loading
    9.  **UNIT_CONVERTED**: Energy unit conversion to GJ
    10. **STEAM_CALCULATED**: Steam emissions calculation
    11. **HEATING_CALCULATED**: District heating emissions calculation
    12. **COOLING_CALCULATED**: District cooling emissions calculation
    13. **CHP_ALLOCATED**: CHP emissions allocation
    14. **BIOGENIC_SEPARATED**: Biogenic CO2 separation
    15. **GAS_BREAKDOWN_COMPUTED**: Per-gas breakdown with GWP
    16. **UNCERTAINTY_QUANTIFIED**: Uncertainty analysis
    17. **COMPLIANCE_CHECKED**: Regulatory compliance verification
    18. **RESULT_ASSEMBLED**: Final result assembly
    19. **PROVENANCE_SEALED**: SHA-256 chain sealing

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
        >>> prov = SteamHeatPurchaseProvenance()
        >>> cid = prov.create_chain("calc-001")
        >>> h1 = prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "FAC-001"})
        >>> h2 = prov.add_stage(cid, "STEAM_CALCULATED", {"kg": 1000.0})
        >>> seal = prov.seal_chain(cid)
        >>> assert prov.verify_chain(cid) is True
        >>> assert len(prov.get_chain(cid)) >= 2
    """

    # ------------------------------------------------------------------
    # Singleton machinery
    # ------------------------------------------------------------------

    _instance: Optional[SteamHeatPurchaseProvenance] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> SteamHeatPurchaseProvenance:
        """Create or return the singleton instance.

        Uses double-checked locking with the class-level RLock to ensure
        thread-safe singleton creation. Only a single instance of
        SteamHeatPurchaseProvenance will ever exist in the process.

        Returns:
            The singleton SteamHeatPurchaseProvenance instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
                    logger.info(
                        "SteamHeatPurchaseProvenance singleton created"
                    )
        return cls._instance

    # ------------------------------------------------------------------
    # Class constants
    # ------------------------------------------------------------------

    #: Default maximum number of entries per chain before warnings.
    DEFAULT_MAX_ENTRIES_PER_CHAIN: int = 10000

    #: Maximum number of concurrent chains.
    DEFAULT_MAX_CHAINS: int = 50000

    #: Prefix used for all Steam/Heat Purchase provenance identifiers.
    PREFIX: str = "gl_shp"

    #: Agent identifier for this provenance tracker.
    AGENT_ID: str = "AGENT-MRV-011"

    #: Human-readable agent name.
    AGENT_NAME: str = "Steam/Heat Purchase"

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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> assert prov.AGENT_ID == "AGENT-MRV-011"
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
        a fresh SteamHeatPurchaseProvenance with empty chains. Primarily
        intended for use in test teardown to prevent state leakage
        between test cases.

        Example:
            >>> prov = SteamHeatPurchaseProvenance()
            >>> prov.create_chain("test-1")
            'test-1'
            >>> SteamHeatPurchaseProvenance.reset()
            >>> prov2 = SteamHeatPurchaseProvenance()
            >>> # prov2 is a fresh instance with no chains
        """
        with cls._lock:
            cls._instance = None
        logger.info(
            "SteamHeatPurchaseProvenance singleton reset to None"
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
            >>> h = SteamHeatPurchaseProvenance.compute_hash({"key": "val"})
            >>> assert len(h) == 64
            >>> h2 = SteamHeatPurchaseProvenance.compute_hash({"key": "val"})
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
            >>> prov = SteamHeatPurchaseProvenance()
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
                ``"REQUEST_RECEIVED"``, ``"STEAM_CALCULATED"``).
                Unknown stages are allowed but logged as warnings.
            data: Dictionary of contextual data for this stage. All
                values must be JSON-serializable or convertible via
                ``str()``. Common keys depend on the stage:

                - REQUEST_RECEIVED: ``facility_id``, ``energy_type``,
                  ``reporting_year``, ``tenant_id``
                - INPUT_VALIDATED: ``valid``, ``field_count``, ``errors``
                - FACILITY_RESOLVED: ``facility_name``, ``country``,
                  ``grid_region``
                - SUPPLIER_RESOLVED: ``supplier_id``, ``supplier_name``,
                  ``energy_carrier``
                - FUEL_EF_RETRIEVED: ``fuel_type``, ``emission_factor``,
                  ``ef_source``, ``ef_year``
                - DH_EF_RETRIEVED: ``region``, ``dh_factor``,
                  ``factor_source``
                - COOLING_PARAMS_RETRIEVED: ``cop``, ``cooling_type``,
                  ``refrigerant``
                - CHP_PARAMS_RETRIEVED: ``chp_type``, ``heat_efficiency``,
                  ``power_efficiency``, ``allocation_method``
                - UNIT_CONVERTED: ``from_unit``, ``to_unit``, ``value``,
                  ``converted_value``
                - STEAM_CALCULATED: ``steam_gj``, ``emission_factor``,
                  ``emissions_kg``, ``boiler_efficiency``
                - HEATING_CALCULATED: ``heating_gj``, ``dh_factor``,
                  ``emissions_kg``
                - COOLING_CALCULATED: ``cooling_gj``, ``cop``,
                  ``emissions_kg``
                - CHP_ALLOCATED: ``total_emissions_kg``,
                  ``heat_allocation_pct``, ``allocated_emissions_kg``
                - BIOGENIC_SEPARATED: ``total_co2_kg``,
                  ``fossil_co2_kg``, ``biogenic_co2_kg``
                - GAS_BREAKDOWN_COMPUTED: ``co2_kg``, ``ch4_kg``,
                  ``n2o_kg``, ``co2e_kg``, ``gwp_source``
                - UNCERTAINTY_QUANTIFIED: ``mean``, ``std_dev``,
                  ``ci_lower``, ``ci_upper``, ``method``
                - COMPLIANCE_CHECKED: ``framework``, ``compliant``,
                  ``findings``
                - RESULT_ASSEMBLED: ``total_co2e_kg``,
                  ``validation_status``

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the new
            entry.

        Raises:
            ValueError: If ``chain_id`` does not exist, ``stage`` is
                empty, or the chain has been sealed.
            TypeError: If ``data`` is not a dictionary.

        Example:
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("calc-002")
            >>> h = prov.add_stage(cid, "REQUEST_RECEIVED", {
            ...     "facility_id": "FAC-002",
            ...     "energy_type": "district_heating",
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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("calc-003")
            >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "FAC-003"})
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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("verify-001")
            >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "1"})
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
            >>> prov = SteamHeatPurchaseProvenance()
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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("get-001")
            >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "FAC-001"})
            '...'
            >>> chain = prov.get_chain(cid)
            >>> assert len(chain) == 1
            >>> assert chain[0]["stage"] == "REQUEST_RECEIVED"
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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("entries-001")
            >>> prov.add_stage(cid, "INPUT_VALIDATED", {"valid": True})
            '...'
            >>> entries = prov.get_chain_entries(cid)
            >>> assert entries[0].stage == "INPUT_VALIDATED"
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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("hash-001")
            >>> assert prov.get_chain_hash(cid) == ""
            >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "1"})
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
        label. Useful for extracting all emission factor lookups or
        compliance checks from a multi-step calculation.

        Args:
            chain_id: Identifier of the chain to search.
            stage: Stage label to filter by (e.g.,
                ``"STEAM_CALCULATED"``, ``"COMPLIANCE_CHECKED"``).

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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("summary-001")
            >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "1"})
            '...'
            >>> prov.add_stage(cid, "STEAM_CALCULATED", {"e": 1000})
            '...'
            >>> summary = prov.get_stage_summary(cid)
            >>> assert summary == {
            ...     "REQUEST_RECEIVED": 1,
            ...     "STEAM_CALCULATED": 1,
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
            - ``agent_id``: Agent identifier (AGENT-MRV-011)
            - ``agent_name``: Human-readable agent name
            - ``prefix``: Provenance prefix (gl_shp)
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
            >>> prov = SteamHeatPurchaseProvenance()
            >>> cid = prov.create_chain("export-001")
            >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "1"})
            '...'
            >>> export = prov.export_chain(cid)
            >>> assert export["agent_id"] == "AGENT-MRV-011"
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
    ) -> SteamHeatPurchaseProvenance:
        """Deserialize a dictionary into a SteamHeatPurchaseProvenance.

        Reconstructs the provenance tracker state from a dictionary
        produced by ``to_dict()``. The singleton is reset and replaced
        with the deserialized instance. Chain integrity is verified
        after deserialization; if verification fails, a warning is
        logged but the instance is still returned.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            The reconstructed SteamHeatPurchaseProvenance instance.

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

    def hash_request_received(
        self,
        chain_id: str,
        facility_id: str,
        energy_type: str,
        reporting_year: int,
        tenant_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Hash the initial calculation request (Stage 1).

        Records the initial request intake for a Steam/Heat Purchase
        emission calculation.

        Args:
            chain_id: Provenance chain identifier.
            facility_id: Facility identifier.
            energy_type: Type of purchased energy (``"steam"``,
                ``"district_heating"``, ``"district_cooling"``,
                ``"chp"``).
            reporting_year: Reporting year.
            tenant_id: Optional tenant identifier.
            request_id: Optional request identifier.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if not energy_type:
            raise ValueError("energy_type must not be empty")
        if reporting_year <= 0:
            raise ValueError(
                f"reporting_year must be positive, got {reporting_year}"
            )

        data: Dict[str, Any] = {
            "facility_id": facility_id,
            "energy_type": energy_type,
            "reporting_year": reporting_year,
        }
        if tenant_id is not None:
            data["tenant_id"] = tenant_id
        if request_id is not None:
            data["request_id"] = request_id

        logger.info(
            "Hashing request received: facility=%s type=%s year=%d",
            facility_id,
            energy_type,
            reporting_year,
        )
        return self.add_stage(chain_id, "REQUEST_RECEIVED", data)

    def hash_input_validated(
        self,
        chain_id: str,
        valid: bool,
        field_count: int,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> str:
        """Hash input validation result (Stage 2).

        Records whether input parameters passed validation and any
        errors or warnings encountered.

        Args:
            chain_id: Provenance chain identifier.
            valid: Whether all inputs passed validation.
            field_count: Number of input fields validated.
            errors: Optional list of validation error messages.
            warnings: Optional list of validation warning messages.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data: Dict[str, Any] = {
            "valid": valid,
            "field_count": field_count,
        }
        if errors is not None:
            data["errors"] = errors
        if warnings is not None:
            data["warnings"] = warnings

        logger.info(
            "Hashing input validation: valid=%s fields=%d errors=%d",
            valid,
            field_count,
            len(errors) if errors else 0,
        )
        return self.add_stage(chain_id, "INPUT_VALIDATED", data)

    def hash_facility_resolved(
        self,
        chain_id: str,
        facility_id: str,
        facility_name: str,
        country: str,
        grid_region: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> str:
        """Hash facility resolution result (Stage 3).

        Records the facility information retrieved from the registry.

        Args:
            chain_id: Provenance chain identifier.
            facility_id: Facility identifier.
            facility_name: Human-readable facility name.
            country: ISO 3166-1 alpha-2 country code.
            grid_region: Optional grid region identifier.
            sector: Optional industry sector.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not facility_id:
            raise ValueError("facility_id must not be empty")
        if not country:
            raise ValueError("country must not be empty")

        data: Dict[str, Any] = {
            "facility_id": facility_id,
            "facility_name": facility_name,
            "country": country,
        }
        if grid_region is not None:
            data["grid_region"] = grid_region
        if sector is not None:
            data["sector"] = sector

        logger.info(
            "Hashing facility resolved: id=%s name=%s country=%s",
            facility_id,
            facility_name,
            country,
        )
        return self.add_stage(chain_id, "FACILITY_RESOLVED", data)

    def hash_supplier_resolved(
        self,
        chain_id: str,
        supplier_id: str,
        supplier_name: str,
        energy_carrier: str,
        country: Optional[str] = None,
        fuel_type: Optional[str] = None,
    ) -> str:
        """Hash supplier resolution result (Stage 4).

        Records the steam/heat supplier information retrieved from the
        supplier registry.

        Args:
            chain_id: Provenance chain identifier.
            supplier_id: Supplier identifier.
            supplier_name: Human-readable supplier name.
            energy_carrier: Type of energy carrier (``"steam"``,
                ``"hot_water"``, ``"chilled_water"``).
            country: Optional ISO country code.
            fuel_type: Optional primary fuel used by supplier.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not supplier_id:
            raise ValueError("supplier_id must not be empty")
        if not energy_carrier:
            raise ValueError("energy_carrier must not be empty")

        data: Dict[str, Any] = {
            "supplier_id": supplier_id,
            "supplier_name": supplier_name,
            "energy_carrier": energy_carrier,
        }
        if country is not None:
            data["country"] = country
        if fuel_type is not None:
            data["fuel_type"] = fuel_type

        logger.info(
            "Hashing supplier resolved: id=%s name=%s carrier=%s",
            supplier_id,
            supplier_name,
            energy_carrier,
        )
        return self.add_stage(chain_id, "SUPPLIER_RESOLVED", data)

    def hash_fuel_ef_retrieved(
        self,
        chain_id: str,
        fuel_type: str,
        emission_factor: float,
        ef_source: str,
        ef_year: int,
        ef_unit: Optional[str] = None,
        co2_factor: Optional[float] = None,
        ch4_factor: Optional[float] = None,
        n2o_factor: Optional[float] = None,
    ) -> str:
        """Hash fuel emission factor retrieval (Stage 5).

        Records the fuel-based emission factors loaded from a reference
        database for the supplier's fuel type.

        Args:
            chain_id: Provenance chain identifier.
            fuel_type: Fuel type identifier (e.g., ``"natural_gas"``,
                ``"coal"``, ``"fuel_oil"``).
            emission_factor: Combined emission factor (kg CO2e/GJ).
                Must be non-negative.
            ef_source: Emission factor source (e.g., ``"IPCC_2006"``,
                ``"EPA_2024"``, ``"DEFRA_2024"``).
            ef_year: Year of the emission factor dataset.
            ef_unit: Optional unit of the emission factor.
            co2_factor: Optional CO2-specific factor (kg CO2/GJ).
            ch4_factor: Optional CH4-specific factor (kg CH4/GJ).
            n2o_factor: Optional N2O-specific factor (kg N2O/GJ).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not fuel_type:
            raise ValueError("fuel_type must not be empty")
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, "
                f"got {emission_factor}"
            )
        if not ef_source:
            raise ValueError("ef_source must not be empty")

        data: Dict[str, Any] = {
            "fuel_type": fuel_type,
            "emission_factor": _safe_str(emission_factor),
            "ef_source": ef_source,
            "ef_year": ef_year,
        }
        if ef_unit is not None:
            data["ef_unit"] = ef_unit
        if co2_factor is not None:
            data["co2_factor"] = _safe_str(co2_factor)
        if ch4_factor is not None:
            data["ch4_factor"] = _safe_str(ch4_factor)
        if n2o_factor is not None:
            data["n2o_factor"] = _safe_str(n2o_factor)

        logger.info(
            "Hashing fuel EF retrieved: fuel=%s ef=%s source=%s year=%d",
            fuel_type,
            _safe_str(emission_factor),
            ef_source,
            ef_year,
        )
        return self.add_stage(chain_id, "FUEL_EF_RETRIEVED", data)

    def hash_dh_ef_retrieved(
        self,
        chain_id: str,
        region: str,
        dh_factor: float,
        factor_source: str,
        factor_year: Optional[int] = None,
        factor_unit: Optional[str] = None,
        country: Optional[str] = None,
    ) -> str:
        """Hash district heating factor retrieval (Stage 6).

        Records the district heating emission factors loaded from a
        regional or national reference database.

        Args:
            chain_id: Provenance chain identifier.
            region: Region or grid zone identifier.
            dh_factor: District heating emission factor (kg CO2e/GJ).
                Must be non-negative.
            factor_source: Factor source reference.
            factor_year: Optional year of the factor dataset.
            factor_unit: Optional unit of the factor.
            country: Optional ISO country code.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not region:
            raise ValueError("region must not be empty")
        if dh_factor < 0:
            raise ValueError(
                f"dh_factor must be non-negative, got {dh_factor}"
            )
        if not factor_source:
            raise ValueError("factor_source must not be empty")

        data: Dict[str, Any] = {
            "region": region,
            "dh_factor": _safe_str(dh_factor),
            "factor_source": factor_source,
        }
        if factor_year is not None:
            data["factor_year"] = factor_year
        if factor_unit is not None:
            data["factor_unit"] = factor_unit
        if country is not None:
            data["country"] = country

        logger.info(
            "Hashing DH EF retrieved: region=%s factor=%s source=%s",
            region,
            _safe_str(dh_factor),
            factor_source,
        )
        return self.add_stage(chain_id, "DH_EF_RETRIEVED", data)

    def hash_cooling_params_retrieved(
        self,
        chain_id: str,
        cop: float,
        cooling_type: str,
        electricity_ef: Optional[float] = None,
        refrigerant: Optional[str] = None,
        capacity_kw: Optional[float] = None,
    ) -> str:
        """Hash cooling system parameter retrieval (Stage 7).

        Records the cooling system parameters loaded for district
        cooling or absorption chiller calculations.

        Args:
            chain_id: Provenance chain identifier.
            cop: Coefficient of Performance. Must be positive.
            cooling_type: Cooling system type (``"electric_chiller"``,
                ``"absorption_chiller"``, ``"district_cooling"``).
            electricity_ef: Optional electricity emission factor
                for electric chillers (kg CO2e/kWh).
            refrigerant: Optional refrigerant type (e.g., ``"R-134a"``).
            capacity_kw: Optional cooling capacity in kW.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if cop <= 0:
            raise ValueError(
                f"cop must be positive, got {cop}"
            )
        if not cooling_type:
            raise ValueError("cooling_type must not be empty")

        data: Dict[str, Any] = {
            "cop": _safe_str(cop),
            "cooling_type": cooling_type,
        }
        if electricity_ef is not None:
            data["electricity_ef"] = _safe_str(electricity_ef)
        if refrigerant is not None:
            data["refrigerant"] = refrigerant
        if capacity_kw is not None:
            data["capacity_kw"] = _safe_str(capacity_kw)

        logger.info(
            "Hashing cooling params: cop=%s type=%s refrigerant=%s",
            _safe_str(cop),
            cooling_type,
            refrigerant or "N/A",
        )
        return self.add_stage(
            chain_id, "COOLING_PARAMS_RETRIEVED", data
        )

    def hash_chp_params_retrieved(
        self,
        chain_id: str,
        chp_type: str,
        heat_efficiency: float,
        power_efficiency: float,
        allocation_method: str,
        total_fuel_input_gj: Optional[float] = None,
        heat_output_gj: Optional[float] = None,
        power_output_mwh: Optional[float] = None,
    ) -> str:
        """Hash CHP parameter retrieval (Stage 8).

        Records the Combined Heat and Power parameters loaded for
        CHP allocation calculations.

        Args:
            chain_id: Provenance chain identifier.
            chp_type: CHP plant type (``"gas_turbine"``,
                ``"steam_turbine"``, ``"combined_cycle"``,
                ``"reciprocating_engine"``).
            heat_efficiency: Thermal efficiency (0.0-1.0).
            power_efficiency: Electrical efficiency (0.0-1.0).
            allocation_method: Allocation method (``"energy"``,
                ``"exergy"``, ``"efficiency"``, ``"residual_heat"``).
            total_fuel_input_gj: Optional total fuel input in GJ.
            heat_output_gj: Optional heat output in GJ.
            power_output_mwh: Optional power output in MWh.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not chp_type:
            raise ValueError("chp_type must not be empty")
        if heat_efficiency < 0 or heat_efficiency > 1.0:
            raise ValueError(
                f"heat_efficiency must be in [0.0, 1.0], "
                f"got {heat_efficiency}"
            )
        if power_efficiency < 0 or power_efficiency > 1.0:
            raise ValueError(
                f"power_efficiency must be in [0.0, 1.0], "
                f"got {power_efficiency}"
            )
        if not allocation_method:
            raise ValueError("allocation_method must not be empty")

        data: Dict[str, Any] = {
            "chp_type": chp_type,
            "heat_efficiency": _safe_str(heat_efficiency),
            "power_efficiency": _safe_str(power_efficiency),
            "allocation_method": allocation_method,
        }
        if total_fuel_input_gj is not None:
            data["total_fuel_input_gj"] = _safe_str(total_fuel_input_gj)
        if heat_output_gj is not None:
            data["heat_output_gj"] = _safe_str(heat_output_gj)
        if power_output_mwh is not None:
            data["power_output_mwh"] = _safe_str(power_output_mwh)

        logger.info(
            "Hashing CHP params: type=%s heat_eff=%s power_eff=%s "
            "method=%s",
            chp_type,
            _safe_str(heat_efficiency),
            _safe_str(power_efficiency),
            allocation_method,
        )
        return self.add_stage(
            chain_id, "CHP_PARAMS_RETRIEVED", data
        )

    def hash_unit_converted(
        self,
        chain_id: str,
        from_unit: str,
        to_unit: str,
        value: float,
        converted_value: float,
        conversion_factor: Optional[float] = None,
    ) -> str:
        """Hash energy unit conversion (Stage 9).

        Records the conversion of energy quantities to the standard
        GJ unit used in all emission calculations.

        Supported conversions include:
        - MWh to GJ (factor: 3.6)
        - MMBtu to GJ (factor: 1.055056)
        - kWh to GJ (factor: 0.0036)
        - Therms to GJ (factor: 0.105506)
        - Steam lb to GJ (varies by pressure/temperature)

        Args:
            chain_id: Provenance chain identifier.
            from_unit: Source energy unit.
            to_unit: Target energy unit (typically ``"GJ"``).
            value: Original value in source units.
            converted_value: Converted value in target units.
            conversion_factor: Optional explicit conversion factor used.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not from_unit:
            raise ValueError("from_unit must not be empty")
        if not to_unit:
            raise ValueError("to_unit must not be empty")

        data: Dict[str, Any] = {
            "from_unit": from_unit,
            "to_unit": to_unit,
            "value": _safe_str(value),
            "converted_value": _safe_str(converted_value),
        }
        if conversion_factor is not None:
            data["conversion_factor"] = _safe_str(conversion_factor)

        logger.info(
            "Hashing unit conversion: %s %s -> %s %s",
            _safe_str(value),
            from_unit,
            _safe_str(converted_value),
            to_unit,
        )
        return self.add_stage(chain_id, "UNIT_CONVERTED", data)

    def hash_steam_calculated(
        self,
        chain_id: str,
        steam_gj: float,
        emission_factor: float,
        emissions_kg: float,
        boiler_efficiency: Optional[float] = None,
        steam_pressure_bar: Optional[float] = None,
        steam_temperature_c: Optional[float] = None,
        calculation_method: Optional[str] = None,
    ) -> str:
        """Hash steam emissions calculation result (Stage 10).

        Records the emission calculation for purchased steam. The
        fundamental formula is:

            emissions_kg = (steam_gj / boiler_efficiency) * emission_factor

        Where boiler_efficiency defaults to 0.80 (80%) if not specified.

        Args:
            chain_id: Provenance chain identifier.
            steam_gj: Steam energy consumed in GJ. Must be non-negative.
            emission_factor: Emission factor (kg CO2e/GJ fuel input).
                Must be non-negative.
            emissions_kg: Calculated emissions in kg CO2e.
            boiler_efficiency: Optional boiler efficiency (0.0-1.0).
            steam_pressure_bar: Optional steam pressure in bar.
            steam_temperature_c: Optional steam temperature in Celsius.
            calculation_method: Optional method (``"fuel_based"``,
                ``"supplier_specific"``, ``"default_factor"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if steam_gj < 0:
            raise ValueError(
                f"steam_gj must be non-negative, got {steam_gj}"
            )
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, "
                f"got {emission_factor}"
            )

        data: Dict[str, Any] = {
            "steam_gj": _safe_str(steam_gj),
            "emission_factor": _safe_str(emission_factor),
            "emissions_kg": _safe_str(emissions_kg),
        }
        if boiler_efficiency is not None:
            data["boiler_efficiency"] = _safe_str(boiler_efficiency)
        if steam_pressure_bar is not None:
            data["steam_pressure_bar"] = _safe_str(steam_pressure_bar)
        if steam_temperature_c is not None:
            data["steam_temperature_c"] = _safe_str(steam_temperature_c)
        if calculation_method is not None:
            data["calculation_method"] = calculation_method

        logger.info(
            "Hashing steam calc: %s GJ * %s kgCO2e/GJ = %s kgCO2e",
            _safe_str(steam_gj),
            _safe_str(emission_factor),
            _safe_str(emissions_kg),
        )
        return self.add_stage(chain_id, "STEAM_CALCULATED", data)

    def hash_heating_calculated(
        self,
        chain_id: str,
        heating_gj: float,
        dh_factor: float,
        emissions_kg: float,
        network_id: Optional[str] = None,
        network_efficiency: Optional[float] = None,
        distribution_loss_pct: Optional[float] = None,
    ) -> str:
        """Hash district heating emissions calculation result (Stage 11).

        Records the emission calculation for purchased district heating.
        The formula is:

            emissions_kg = heating_gj * dh_factor

        Where dh_factor is the regional district heating emission factor
        in kg CO2e per GJ of delivered heat.

        Args:
            chain_id: Provenance chain identifier.
            heating_gj: District heating consumed in GJ.
                Must be non-negative.
            dh_factor: District heating emission factor (kg CO2e/GJ).
                Must be non-negative.
            emissions_kg: Calculated emissions in kg CO2e.
            network_id: Optional district heating network identifier.
            network_efficiency: Optional network efficiency (0.0-1.0).
            distribution_loss_pct: Optional distribution loss percentage.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if heating_gj < 0:
            raise ValueError(
                f"heating_gj must be non-negative, got {heating_gj}"
            )
        if dh_factor < 0:
            raise ValueError(
                f"dh_factor must be non-negative, got {dh_factor}"
            )

        data: Dict[str, Any] = {
            "heating_gj": _safe_str(heating_gj),
            "dh_factor": _safe_str(dh_factor),
            "emissions_kg": _safe_str(emissions_kg),
        }
        if network_id is not None:
            data["network_id"] = network_id
        if network_efficiency is not None:
            data["network_efficiency"] = _safe_str(network_efficiency)
        if distribution_loss_pct is not None:
            data["distribution_loss_pct"] = _safe_str(
                distribution_loss_pct
            )

        logger.info(
            "Hashing heating calc: %s GJ * %s kgCO2e/GJ = %s kgCO2e",
            _safe_str(heating_gj),
            _safe_str(dh_factor),
            _safe_str(emissions_kg),
        )
        return self.add_stage(chain_id, "HEATING_CALCULATED", data)

    def hash_cooling_calculated(
        self,
        chain_id: str,
        cooling_gj: float,
        cop: float,
        emissions_kg: float,
        electricity_consumed_kwh: Optional[float] = None,
        electricity_ef: Optional[float] = None,
        cooling_type: Optional[str] = None,
    ) -> str:
        """Hash district cooling emissions calculation result (Stage 12).

        Records the emission calculation for purchased district cooling.
        For electric chillers, the formula is:

            electricity_kwh = cooling_gj * 1000 / (3.6 * cop)
            emissions_kg = electricity_kwh * electricity_ef

        For district cooling with a regional factor:

            emissions_kg = cooling_gj * cooling_ef

        Args:
            chain_id: Provenance chain identifier.
            cooling_gj: Cooling energy consumed in GJ.
                Must be non-negative.
            cop: Coefficient of Performance. Must be positive.
            emissions_kg: Calculated emissions in kg CO2e.
            electricity_consumed_kwh: Optional electricity consumed.
            electricity_ef: Optional electricity emission factor.
            cooling_type: Optional cooling system type.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if cooling_gj < 0:
            raise ValueError(
                f"cooling_gj must be non-negative, got {cooling_gj}"
            )
        if cop <= 0:
            raise ValueError(
                f"cop must be positive, got {cop}"
            )

        data: Dict[str, Any] = {
            "cooling_gj": _safe_str(cooling_gj),
            "cop": _safe_str(cop),
            "emissions_kg": _safe_str(emissions_kg),
        }
        if electricity_consumed_kwh is not None:
            data["electricity_consumed_kwh"] = _safe_str(
                electricity_consumed_kwh
            )
        if electricity_ef is not None:
            data["electricity_ef"] = _safe_str(electricity_ef)
        if cooling_type is not None:
            data["cooling_type"] = cooling_type

        logger.info(
            "Hashing cooling calc: %s GJ COP=%s = %s kgCO2e",
            _safe_str(cooling_gj),
            _safe_str(cop),
            _safe_str(emissions_kg),
        )
        return self.add_stage(chain_id, "COOLING_CALCULATED", data)

    def hash_chp_allocated(
        self,
        chain_id: str,
        total_emissions_kg: float,
        heat_allocation_pct: float,
        allocated_emissions_kg: float,
        allocation_method: str,
        power_allocation_pct: Optional[float] = None,
        total_fuel_input_gj: Optional[float] = None,
        heat_output_gj: Optional[float] = None,
        power_output_mwh: Optional[float] = None,
    ) -> str:
        """Hash CHP emissions allocation result (Stage 13).

        Records the allocation of CHP plant emissions between heat and
        power outputs. The allocation determines what fraction of the
        CHP plant's total emissions are attributable to the purchased
        steam/heat.

        Supported allocation methods:
        - ``"energy"``: Proportional to energy output (GJ heat / GJ total)
        - ``"exergy"``: Proportional to exergy output
        - ``"efficiency"``: Based on reference efficiency comparison
        - ``"residual_heat"``: All emissions to power, residual to heat

        Args:
            chain_id: Provenance chain identifier.
            total_emissions_kg: Total CHP emissions in kg CO2e.
            heat_allocation_pct: Percentage allocated to heat (0-100).
            allocated_emissions_kg: Emissions allocated to heat in kg CO2e.
            allocation_method: Allocation method used.
            power_allocation_pct: Optional power allocation percentage.
            total_fuel_input_gj: Optional total fuel input.
            heat_output_gj: Optional heat output.
            power_output_mwh: Optional power output.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if total_emissions_kg < 0:
            raise ValueError(
                f"total_emissions_kg must be non-negative, "
                f"got {total_emissions_kg}"
            )
        if heat_allocation_pct < 0 or heat_allocation_pct > 100:
            raise ValueError(
                f"heat_allocation_pct must be in [0, 100], "
                f"got {heat_allocation_pct}"
            )
        if not allocation_method:
            raise ValueError("allocation_method must not be empty")

        data: Dict[str, Any] = {
            "total_emissions_kg": _safe_str(total_emissions_kg),
            "heat_allocation_pct": _safe_str(heat_allocation_pct),
            "allocated_emissions_kg": _safe_str(allocated_emissions_kg),
            "allocation_method": allocation_method,
        }
        if power_allocation_pct is not None:
            data["power_allocation_pct"] = _safe_str(
                power_allocation_pct
            )
        if total_fuel_input_gj is not None:
            data["total_fuel_input_gj"] = _safe_str(total_fuel_input_gj)
        if heat_output_gj is not None:
            data["heat_output_gj"] = _safe_str(heat_output_gj)
        if power_output_mwh is not None:
            data["power_output_mwh"] = _safe_str(power_output_mwh)

        logger.info(
            "Hashing CHP allocation: total=%s heat_pct=%s "
            "allocated=%s method=%s",
            _safe_str(total_emissions_kg),
            _safe_str(heat_allocation_pct),
            _safe_str(allocated_emissions_kg),
            allocation_method,
        )
        return self.add_stage(chain_id, "CHP_ALLOCATED", data)

    def hash_biogenic_separated(
        self,
        chain_id: str,
        total_co2_kg: float,
        fossil_co2_kg: float,
        biogenic_co2_kg: float,
        biogenic_fraction: Optional[float] = None,
        biomass_fuel_type: Optional[str] = None,
    ) -> str:
        """Hash biogenic CO2 separation result (Stage 14).

        Records the separation of total CO2 emissions into fossil and
        biogenic components. Biogenic CO2 is reported separately per
        GHG Protocol and CSRD requirements.

        Args:
            chain_id: Provenance chain identifier.
            total_co2_kg: Total CO2 before separation (kg).
            fossil_co2_kg: Fossil CO2 component (kg).
            biogenic_co2_kg: Biogenic CO2 component (kg).
            biogenic_fraction: Optional biogenic fraction (0.0-1.0).
            biomass_fuel_type: Optional biomass fuel identifier.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data: Dict[str, Any] = {
            "total_co2_kg": _safe_str(total_co2_kg),
            "fossil_co2_kg": _safe_str(fossil_co2_kg),
            "biogenic_co2_kg": _safe_str(biogenic_co2_kg),
        }
        if biogenic_fraction is not None:
            data["biogenic_fraction"] = _safe_str(biogenic_fraction)
        if biomass_fuel_type is not None:
            data["biomass_fuel_type"] = biomass_fuel_type

        logger.info(
            "Hashing biogenic separation: total=%s fossil=%s "
            "biogenic=%s",
            _safe_str(total_co2_kg),
            _safe_str(fossil_co2_kg),
            _safe_str(biogenic_co2_kg),
        )
        return self.add_stage(chain_id, "BIOGENIC_SEPARATED", data)

    def hash_gas_breakdown_computed(
        self,
        chain_id: str,
        co2_kg: float,
        ch4_kg: float,
        n2o_kg: float,
        co2e_kg: float,
        gwp_source: str,
        gwp_ch4: Optional[float] = None,
        gwp_n2o: Optional[float] = None,
    ) -> str:
        """Hash per-gas breakdown computation (Stage 15).

        Records the decomposition of total emissions into individual
        greenhouse gases (CO2, CH4, N2O) and their conversion to CO2e
        using GWP values.

        Args:
            chain_id: Provenance chain identifier.
            co2_kg: CO2 emissions in kg.
            ch4_kg: CH4 emissions in kg.
            n2o_kg: N2O emissions in kg.
            co2e_kg: Total CO2-equivalent emissions in kg.
            gwp_source: GWP value source (``"IPCC_AR5"``,
                ``"IPCC_AR6"``, ``"IPCC_AR4"``).
            gwp_ch4: Optional GWP value used for CH4.
            gwp_n2o: Optional GWP value used for N2O.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not gwp_source:
            raise ValueError("gwp_source must not be empty")

        data: Dict[str, Any] = {
            "co2_kg": _safe_str(co2_kg),
            "ch4_kg": _safe_str(ch4_kg),
            "n2o_kg": _safe_str(n2o_kg),
            "co2e_kg": _safe_str(co2e_kg),
            "gwp_source": gwp_source,
        }
        if gwp_ch4 is not None:
            data["gwp_ch4"] = _safe_str(gwp_ch4)
        if gwp_n2o is not None:
            data["gwp_n2o"] = _safe_str(gwp_n2o)

        logger.info(
            "Hashing gas breakdown: CO2=%s CH4=%s N2O=%s CO2e=%s "
            "source=%s",
            _safe_str(co2_kg),
            _safe_str(ch4_kg),
            _safe_str(n2o_kg),
            _safe_str(co2e_kg),
            gwp_source,
        )
        return self.add_stage(
            chain_id, "GAS_BREAKDOWN_COMPUTED", data
        )

    def hash_uncertainty_quantified(
        self,
        chain_id: str,
        mean: float,
        std_dev: float,
        ci_lower: float,
        ci_upper: float,
        method: str,
        confidence_level: Optional[float] = None,
        n_simulations: Optional[int] = None,
        data_quality_score: Optional[float] = None,
    ) -> str:
        """Hash uncertainty quantification result (Stage 16).

        Records the uncertainty analysis results for the emission
        calculation, including confidence intervals and data quality.

        Supported methods:
        - ``"monte_carlo"``: Monte Carlo simulation
        - ``"analytical"``: Error propagation formula
        - ``"ipcc_tier1"``: IPCC default uncertainty ranges
        - ``"expert_judgment"``: Expert-derived ranges

        Args:
            chain_id: Provenance chain identifier.
            mean: Mean emission estimate (kg CO2e).
            std_dev: Standard deviation (kg CO2e).
            ci_lower: Lower confidence interval bound (kg CO2e).
            ci_upper: Upper confidence interval bound (kg CO2e).
            method: Uncertainty quantification method.
            confidence_level: Optional confidence level (e.g., 0.95).
            n_simulations: Optional number of Monte Carlo simulations.
            data_quality_score: Optional data quality score (0.0-1.0).

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
            "mean": _safe_str(mean),
            "std_dev": _safe_str(std_dev),
            "ci_lower": _safe_str(ci_lower),
            "ci_upper": _safe_str(ci_upper),
            "method": method,
        }
        if confidence_level is not None:
            data["confidence_level"] = _safe_str(confidence_level)
        if n_simulations is not None:
            data["n_simulations"] = n_simulations
        if data_quality_score is not None:
            data["data_quality_score"] = _safe_str(data_quality_score)

        logger.info(
            "Hashing uncertainty: mean=%s std=%s CI=[%s, %s] "
            "method=%s",
            _safe_str(mean),
            _safe_str(std_dev),
            _safe_str(ci_lower),
            _safe_str(ci_upper),
            method,
        )
        return self.add_stage(
            chain_id, "UNCERTAINTY_QUANTIFIED", data
        )

    def hash_compliance_checked(
        self,
        chain_id: str,
        framework: str,
        compliant: bool,
        findings: Optional[List[str]] = None,
        score: Optional[float] = None,
        checked_rules: Optional[int] = None,
        passed_rules: Optional[int] = None,
    ) -> str:
        """Hash regulatory compliance check result (Stage 17).

        Records the result of verifying the emission calculation against
        a regulatory framework's requirements.

        Supported frameworks:
        - ``"GHG_PROTOCOL"``: GHG Protocol Corporate Standard
        - ``"GHG_PROTOCOL_SCOPE2"``: GHG Protocol Scope 2 Guidance
        - ``"ISO_14064"``: ISO 14064-1:2018
        - ``"CSRD_ESRS_E1"``: CSRD / ESRS E1
        - ``"UK_SECR"``: UK Streamlined Energy and Carbon Reporting
        - ``"EPA_MRR"``: EPA Mandatory Reporting Rule
        - ``"EU_ETS"``: EU Emissions Trading System
        - ``"CDP"``: CDP Climate Change Questionnaire

        Args:
            chain_id: Provenance chain identifier.
            framework: Regulatory framework identifier.
            compliant: Whether the calculation is compliant.
            findings: Optional list of compliance findings.
            score: Optional compliance score (0.0-1.0).
            checked_rules: Optional total rules checked.
            passed_rules: Optional rules that passed.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not framework:
            raise ValueError("framework must not be empty")

        data: Dict[str, Any] = {
            "framework": framework,
            "compliant": compliant,
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
            "Hashing compliance check: framework=%s compliant=%s "
            "score=%s",
            framework,
            compliant,
            _safe_str(score) if score is not None else "N/A",
        )
        return self.add_stage(chain_id, "COMPLIANCE_CHECKED", data)

    def hash_result_assembled(
        self,
        chain_id: str,
        total_co2e_kg: float,
        total_co2e_tonnes: Optional[float] = None,
        validation_status: str = "PASS",
        energy_type: Optional[str] = None,
        steam_co2e_kg: Optional[float] = None,
        heating_co2e_kg: Optional[float] = None,
        cooling_co2e_kg: Optional[float] = None,
        biogenic_co2_kg: Optional[float] = None,
        data_quality_score: Optional[float] = None,
    ) -> str:
        """Hash final result assembly (Stage 18).

        Records the final assembled result of the Steam/Heat Purchase
        emission calculation, including totals across all energy
        carriers and validation status.

        Args:
            chain_id: Provenance chain identifier.
            total_co2e_kg: Total CO2e emissions in kg.
            total_co2e_tonnes: Optional total in metric tonnes.
            validation_status: Validation result (``"PASS"`` or
                ``"FAIL"``).
            energy_type: Optional primary energy type.
            steam_co2e_kg: Optional steam contribution in kg CO2e.
            heating_co2e_kg: Optional heating contribution in kg CO2e.
            cooling_co2e_kg: Optional cooling contribution in kg CO2e.
            biogenic_co2_kg: Optional biogenic CO2 in kg.
            data_quality_score: Optional data quality score.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        data: Dict[str, Any] = {
            "total_co2e_kg": _safe_str(total_co2e_kg),
            "validation_status": validation_status,
        }
        if total_co2e_tonnes is not None:
            data["total_co2e_tonnes"] = _safe_str(total_co2e_tonnes)
        if energy_type is not None:
            data["energy_type"] = energy_type
        if steam_co2e_kg is not None:
            data["steam_co2e_kg"] = _safe_str(steam_co2e_kg)
        if heating_co2e_kg is not None:
            data["heating_co2e_kg"] = _safe_str(heating_co2e_kg)
        if cooling_co2e_kg is not None:
            data["cooling_co2e_kg"] = _safe_str(cooling_co2e_kg)
        if biogenic_co2_kg is not None:
            data["biogenic_co2_kg"] = _safe_str(biogenic_co2_kg)
        if data_quality_score is not None:
            data["data_quality_score"] = _safe_str(data_quality_score)

        logger.info(
            "Hashing result assembled: total=%s kgCO2e status=%s",
            _safe_str(total_co2e_kg),
            validation_status,
        )
        return self.add_stage(chain_id, "RESULT_ASSEMBLED", data)

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
            >>> prov = SteamHeatPurchaseProvenance()
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
            f"SteamHeatPurchaseProvenance("
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
        SteamHeatPurchaseProvenance.DEFAULT_MAX_ENTRIES_PER_CHAIN
    ),
    max_chains: int = SteamHeatPurchaseProvenance.DEFAULT_MAX_CHAINS,
) -> SteamHeatPurchaseProvenance:
    """Create (or return) the SteamHeatPurchaseProvenance singleton.

    Factory function that provides a clean entry point for obtaining
    the provenance tracker. Since SteamHeatPurchaseProvenance is a
    singleton, this always returns the same instance.

    Args:
        max_entries_per_chain: Maximum entries per chain before warnings.
            Only effective on first creation.
        max_chains: Maximum concurrent chains. Only effective on first
            creation.

    Returns:
        The singleton SteamHeatPurchaseProvenance instance.

    Example:
        >>> prov = create_provenance()
        >>> assert isinstance(prov, SteamHeatPurchaseProvenance)
    """
    return SteamHeatPurchaseProvenance(
        max_entries_per_chain=max_entries_per_chain,
        max_chains=max_chains,
    )

def get_provenance() -> SteamHeatPurchaseProvenance:
    """Return the process-wide singleton SteamHeatPurchaseProvenance.

    This is the recommended entry point for obtaining the provenance
    tracker in production code. The singleton is created lazily on
    first call and reused for all subsequent calls.

    Returns:
        The singleton SteamHeatPurchaseProvenance instance.

    Example:
        >>> tracker_a = get_provenance()
        >>> tracker_b = get_provenance()
        >>> assert tracker_a is tracker_b
    """
    return SteamHeatPurchaseProvenance()

def reset_provenance() -> None:
    """Destroy the current singleton and reset to None.

    The next call to ``get_provenance()`` will create a fresh instance.
    Intended for use in test teardown to prevent state leakage between
    test cases.

    Example:
        >>> reset_provenance()
        >>> tracker = get_provenance()  # fresh instance
    """
    SteamHeatPurchaseProvenance.reset()

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
    ``SteamHeatPurchaseProvenance._compute_chain_hash`` but without
    side effects. Useful for external verification of chain entries.

    Args:
        previous_hash: Previous entry's hash (empty string for first).
        stage: Stage identifier string.
        data: Dictionary of metadata.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> h = compute_chain_entry_hash(
        ...     "", "REQUEST_RECEIVED", {"f": "FAC-001"}
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
        >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "1"})
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

    Takes the output of ``SteamHeatPurchaseProvenance.export_chain()``
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
        >>> prov.add_stage(cid, "REQUEST_RECEIVED", {"f": "1"})
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
    if agent_id != "AGENT-MRV-011":
        return (
            False,
            f"Expected agent_id 'AGENT-MRV-011', got '{agent_id}'",
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
