# -*- coding: utf-8 -*-
"""
Provenance Tracking for Scope 2 Location-Based Emissions Agent - AGENT-MRV-009

Provides SHA-256 based audit trail tracking for all Scope 2 location-based
emissions agent operations. Implements a chain of SHA-256 hashes for each
calculation stage, ensuring complete audit trail and deterministic
reproducibility across grid emission factor lookups, T&D loss calculations,
electricity/steam/heat/cooling emissions, per-gas GHG breakdowns, GWP
conversions, compliance checks, and uncertainty quantification.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256 via hashlib
    - Chain hashing links operations in sequence (append-only)
    - Each entry records previous_hash for tamper detection
    - JSON canonical form with sort_keys=True for reproducibility
    - Decimal values converted to string for hashing consistency
    - Complete provenance for every Scope 2 location-based operation

Provenance Stages:
    - input: Raw input data hashing at intake
    - grid_factor_lookup: Grid emission factor retrieval from database
    - td_loss: Transmission and distribution loss parameter hashing
    - electricity_calc: Electricity consumption emission calculation
    - steam_heat_cooling_calc: Steam, heat, or cooling emission calculation
    - gas_breakdown: Per-gas (CO2, CH4, N2O) breakdown with GWP source
    - gwp_conversion: Individual GHG to CO2e conversion step
    - compliance_check: Regulatory framework compliance check result
    - uncertainty: Monte Carlo or analytical uncertainty result
    - output: Final output data hashing at completion
    - merge: Chain merge operation for batch processing

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
    - GHG Protocol Scope 2 Guidance (2015)
    - ISO 14064-1:2018
    - CSRD / ESRS E1
    - UK SECR / Streamlined Energy and Carbon Reporting
    - EPA 40 CFR Part 98 (eGRID)
    - EU ETS (Monitoring and Reporting Regulation)
    - CDP Climate Change Questionnaire
    - RE100 Reporting Criteria

Example:
    >>> from greenlang.scope2_location.provenance import (
    ...     Scope2LocationProvenance, create_provenance,
    ... )
    >>> prov = create_provenance()
    >>> h1 = prov.hash_input({"facility_id": "FAC-001", "year": 2025})
    >>> h2 = prov.hash_grid_factor(
    ...     region_id="US-WECC",
    ...     source="eGRID2023",
    ...     year=2023,
    ...     co2_ef=0.3127,
    ...     ch4_ef=0.0000112,
    ...     n2o_ef=0.0000042,
    ... )
    >>> h3 = prov.hash_electricity_calculation(
    ...     consumption_mwh=5000.0,
    ...     ef_co2e=0.3127,
    ...     td_loss_pct=5.3,
    ...     total_co2e=1646.84,
    ... )
    >>> chain = prov.get_chain()
    >>> assert len(chain) == 3
    >>> assert prov.verify_chain() is True
    >>> chain_hash = prov.get_chain_hash()
    >>> assert len(chain_hash) == 64

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed.

    Returns:
        UTC datetime with microsecond component set to zero for
        reproducible ISO timestamp strings.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


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
# Valid provenance stages
# ---------------------------------------------------------------------------

VALID_STAGES = frozenset({
    # Lifecycle stages
    "input",
    "output",
    # Grid emission factor operations
    "grid_factor_lookup",
    "grid_factor_update",
    "grid_factor_validate",
    # T&D loss operations
    "td_loss",
    "td_loss_validate",
    "td_loss_country_lookup",
    # Electricity calculation stages
    "electricity_calc",
    "electricity_calc_validate",
    "electricity_calc_batch",
    # Steam, heat, and cooling calculation stages
    "steam_heat_cooling_calc",
    "steam_calc",
    "heat_calc",
    "cooling_calc",
    # Per-gas breakdown stages
    "gas_breakdown",
    "gas_breakdown_validate",
    # GWP conversion stages
    "gwp_conversion",
    "gwp_conversion_co2",
    "gwp_conversion_ch4",
    "gwp_conversion_n2o",
    # Compliance stages
    "compliance_check",
    "compliance_check_ghg_protocol",
    "compliance_check_iso14064",
    "compliance_check_csrd",
    "compliance_check_uk_secr",
    "compliance_check_epa_egrid",
    "compliance_check_eu_ets",
    "compliance_check_cdp",
    "compliance_check_re100",
    # Uncertainty stages
    "uncertainty",
    "uncertainty_monte_carlo",
    "uncertainty_analytical",
    "uncertainty_propagation",
    # Aggregation stages
    "aggregation",
    "aggregation_facility",
    "aggregation_organization",
    "aggregation_temporal",
    # Batch and merge stages
    "batch",
    "merge",
    "merge_chains",
    # Audit stages
    "audit_export",
    "audit_verify",
})


# ---------------------------------------------------------------------------
# ProvenanceEntry dataclass (frozen for immutability)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceEntry:
    """A single immutable, tamper-evident provenance record for a Scope 2
    location-based emission calculation stage.

    Each entry in the provenance chain captures the stage name, its SHA-256
    hash (incorporating the previous entry's hash for chain integrity), the
    ISO-formatted UTC timestamp, the link to the previous hash, and a
    metadata dictionary containing the raw data used to compute this hash.

    The ``frozen=True`` decorator ensures that once created, an entry
    cannot be modified in-place, enforcing the append-only property of
    the provenance chain.

    Attributes:
        stage: Identifies the calculation stage that produced this entry.
            Examples: ``"input"``, ``"grid_factor_lookup"``, ``"td_loss"``,
            ``"electricity_calc"``, ``"steam_heat_cooling_calc"``,
            ``"gas_breakdown"``, ``"gwp_conversion"``,
            ``"compliance_check"``, ``"uncertainty"``, ``"output"``.
        hash_value: SHA-256 hex digest of this entry's payload, computed as
            ``SHA256(f"{previous_hash}|{stage}|{canonical_json(metadata)}")``.
            This is the primary identifier used for chain linking and
            tamper detection.
        timestamp: ISO 8601 formatted UTC timestamp string recording when
            this entry was created. Microseconds are zeroed for
            reproducibility in test scenarios.
        previous_hash: The ``hash_value`` of the immediately preceding
            entry in the chain. For the first entry, this is an empty
            string ``""``. This linkage is what makes the chain
            tamper-evident: modifying any entry invalidates all subsequent
            hashes.
        metadata: Dictionary of additional contextual fields that were
            used to compute this entry's hash. This preserves the raw
            data for audit inspection without requiring recalculation.
            Values may include floats, strings, integers, and nested
            dictionaries.

    Example:
        >>> entry = ProvenanceEntry(
        ...     stage="electricity_calc",
        ...     hash_value="a1b2c3...",
        ...     timestamp="2025-06-15T10:30:00+00:00",
        ...     previous_hash="d4e5f6...",
        ...     metadata={"consumption_mwh": 5000.0, "total_co2e": 1646.84},
        ... )
        >>> entry.stage
        'electricity_calc'
        >>> entry.hash_value[:6]
        'a1b2c3'
    """

    stage: str
    hash_value: str
    timestamp: str
    previous_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entry to a plain dictionary.

        Returns a shallow copy of all fields as a dictionary suitable for
        JSON serialization, database persistence, or API responses.

        Returns:
            Dictionary with keys ``stage``, ``hash_value``, ``timestamp``,
            ``previous_hash``, and ``metadata``.
        """
        return {
            "stage": self.stage,
            "hash_value": self.hash_value,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "metadata": dict(self.metadata) if self.metadata else {},
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
        )


# ---------------------------------------------------------------------------
# Scope2LocationProvenance
# ---------------------------------------------------------------------------


class Scope2LocationProvenance:
    """SHA-256 provenance chain for Scope 2 location-based emission calculations.

    Implements a chain of SHA-256 hashes for each calculation stage, ensuring
    complete audit trail and deterministic reproducibility. The chain is
    append-only during calculation: each new entry's hash incorporates the
    previous entry's hash, creating a tamper-evident linked list of
    cryptographic digests.

    This class provides domain-specific hashing methods for every stage in
    the Scope 2 location-based emissions calculation pipeline:

    1. **Input Hashing** (``hash_input``): Captures the raw input data
       (facility ID, reporting period, consumption data) at the start of
       the calculation.

    2. **Grid Emission Factor Hashing** (``hash_grid_factor``): Records
       the grid emission factor lookup including region, source database,
       reference year, and per-gas factors (CO2, CH4, N2O).

    3. **T&D Loss Hashing** (``hash_td_loss``): Records transmission and
       distribution loss parameters including country code, loss percentage,
       and the method used to determine losses.

    4. **Electricity Calculation Hashing** (``hash_electricity_calculation``):
       Records the core electricity emission calculation including
       consumption (MWh), emission factor (tCO2e/MWh), T&D loss adjustment,
       and the resulting total CO2e.

    5. **Steam/Heat/Cooling Hashing** (``hash_steam_heat_cooling``): Records
       calculations for purchased steam, district heating, and cooling
       energy types.

    6. **Gas Breakdown Hashing** (``hash_gas_breakdown``): Records the
       per-gas decomposition (CO2, CH4, N2O) with the GWP source reference
       and total CO2e.

    7. **GWP Conversion Hashing** (``hash_gwp_conversion``): Records
       individual greenhouse gas to CO2e conversion steps.

    8. **Compliance Check Hashing** (``hash_compliance_check``): Records
       the result of a regulatory framework compliance check including
       the framework name, pass/fail status, and number of findings.

    9. **Uncertainty Hashing** (``hash_uncertainty``): Records uncertainty
       quantification results including method, mean, standard deviation,
       and confidence interval bounds.

    10. **Output Hashing** (``hash_output``): Captures the final output
        result at the end of the calculation.

    Thread Safety:
        All public methods that modify the chain are protected by a
        reentrant lock (``threading.RLock``). This allows safe concurrent
        access from multiple threads within the same calculation pipeline.

    Attributes:
        _chain: Ordered list of frozen ProvenanceEntry instances.
        _lock: Reentrant lock for thread-safe chain modifications.
        _max_entries: Maximum chain length before eviction of oldest entries.

    Example:
        >>> prov = Scope2LocationProvenance()
        >>> h1 = prov.hash_input({"facility_id": "FAC-001"})
        >>> h2 = prov.hash_grid_factor("US-WECC", "eGRID", 2023, 0.31, 0.00001, 0.000004)
        >>> h3 = prov.hash_output({"total_co2e": 1646.84})
        >>> assert prov.verify_chain() is True
        >>> assert len(prov.get_chain()) == 3
    """

    # ------------------------------------------------------------------
    # Class constants
    # ------------------------------------------------------------------

    #: Default maximum number of entries before eviction triggers.
    DEFAULT_MAX_ENTRIES: int = 50000

    #: Prefix used for all Scope 2 location-based provenance identifiers.
    PREFIX: str = "gl_s2l"

    #: Agent identifier for this provenance tracker.
    AGENT_ID: str = "AGENT-MRV-009"

    #: Human-readable agent name.
    AGENT_NAME: str = "Scope 2 Location-Based Emissions"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Initialize an empty provenance chain.

        Creates a new Scope2LocationProvenance instance with an empty
        chain of ProvenanceEntry records. The chain will be populated
        as calculation stages call the domain-specific hash methods.

        Args:
            max_entries: Maximum number of provenance entries to retain
                in memory. When exceeded, the oldest entries are evicted
                to prevent unbounded memory growth. Defaults to 50000.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> assert len(prov.get_chain()) == 0
            >>> assert prov.get_chain_hash() == ""
        """
        self._chain: List[ProvenanceEntry] = []
        self._lock: threading.RLock = threading.RLock()
        self._max_entries: int = max(1, max_entries)
        logger.info(
            "%s provenance tracker initialized (max_entries=%d)",
            self.AGENT_ID,
            self._max_entries,
        )

    # ------------------------------------------------------------------
    # Core hashing engine
    # ------------------------------------------------------------------

    def _compute_hash(self, stage: str, data: Dict[str, Any]) -> str:
        """Compute a SHA-256 hash for a calculation stage and append to chain.

        This is the core hashing method used by all domain-specific hash
        methods. It implements the GreenLang provenance hashing protocol:

        1. Retrieve the hash_value of the last entry in the chain (or
           empty string if the chain is empty).
        2. Serialize ``data`` to canonical JSON (``sort_keys=True``,
           ``default=str``).
        3. Construct the payload as ``f"{previous}|{stage}|{canonical}"``.
        4. Compute ``SHA-256(payload.encode("utf-8")).hexdigest()``.
        5. Create a frozen ProvenanceEntry and append to the chain.
        6. Return the hex digest.

        The pipe-delimited payload format ensures that the stage label
        is unambiguously separated from the data, preventing hash
        collisions between stages that share similar data.

        Args:
            stage: String label identifying the calculation stage. Should
                be one of the values in ``VALID_STAGES``, though custom
                stages are permitted for extensibility.
            data: Dictionary of values to include in the hash. All values
                must be JSON-serializable or convertible via ``str()``.
                Typically includes the numeric inputs and outputs of the
                calculation step.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            TypeError: If ``data`` is not a dictionary.
            ValueError: If ``stage`` is empty.
        """
        if not stage:
            raise ValueError("stage must not be empty")
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dict, got {type(data).__name__}"
            )

        with self._lock:
            previous = self._chain[-1].hash_value if self._chain else ""
            canonical = _canonical_json(data)
            payload = f"{previous}|{stage}|{canonical}"
            hash_value = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()

            entry = ProvenanceEntry(
                stage=stage,
                hash_value=hash_value,
                timestamp=_utcnow().isoformat(),
                previous_hash=previous,
                metadata=data,
            )
            self._chain.append(entry)

            # Evict oldest entries if max_entries exceeded
            self._evict_if_needed()

        logger.debug(
            "Provenance entry: stage=%s hash_prefix=%s chain_len=%d",
            stage,
            hash_value[:16],
            len(self._chain),
        )
        return hash_value

    def _compute_hash_no_append(
        self, stage: str, data: Dict[str, Any],
    ) -> str:
        """Compute a SHA-256 hash without appending to the chain.

        Useful for pre-computing hashes for validation or comparison
        without side effects on the provenance chain.

        Args:
            stage: String label identifying the calculation stage.
            data: Dictionary of values to include in the hash.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        with self._lock:
            previous = self._chain[-1].hash_value if self._chain else ""
        canonical = _canonical_json(data)
        payload = f"{previous}|{stage}|{canonical}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Input / Output
    # ------------------------------------------------------------------

    def hash_input(self, data: Dict[str, Any]) -> str:
        """Hash input data and add to the provenance chain.

        Records the raw input data at the start of a Scope 2 location-based
        emission calculation. This is typically the first entry in the chain
        and captures facility identifiers, reporting period, consumption
        data, and any configuration parameters.

        The input data dictionary is hashed as-is using canonical JSON
        serialization. Callers should include all data that influences
        the calculation result to ensure complete reproducibility.

        Args:
            data: Dictionary of input data. Common keys include:
                - ``facility_id``: Facility identifier string.
                - ``reporting_year``: Integer reporting year.
                - ``reporting_period``: Period string (e.g., "2025-Q1").
                - ``electricity_mwh``: Electricity consumption in MWh.
                - ``steam_gj``: Purchased steam in GJ.
                - ``heat_gj``: District heating in GJ.
                - ``cooling_gj``: Cooling energy in GJ.
                - ``region_id``: Grid region identifier.
                - ``country_code``: ISO 3166-1 alpha-2 country code.
                - ``tenant_id``: Multi-tenant identifier.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            TypeError: If ``data`` is not a dictionary.
            ValueError: If ``data`` is empty.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> h = prov.hash_input({
            ...     "facility_id": "FAC-001",
            ...     "reporting_year": 2025,
            ...     "electricity_mwh": 5000.0,
            ...     "region_id": "US-WECC",
            ... })
            >>> assert len(h) == 64
        """
        if not data:
            raise ValueError("Input data must not be empty")

        logger.info(
            "Hashing input data with %d keys for %s",
            len(data),
            self.AGENT_ID,
        )
        return self._compute_hash("input", data)

    def hash_output(self, result: Dict[str, Any]) -> str:
        """Hash the final output result and add to the provenance chain.

        Records the final calculation result at the end of the Scope 2
        location-based emission calculation pipeline. This is typically
        the last entry in the chain and captures the aggregated totals,
        per-gas breakdowns, and any quality metrics.

        Args:
            result: Dictionary of output data. Common keys include:
                - ``total_co2e_kg``: Total CO2e emissions in kilograms.
                - ``total_co2e_tonnes``: Total CO2e in metric tonnes.
                - ``electricity_co2e``: Electricity-related CO2e.
                - ``steam_co2e``: Steam-related CO2e.
                - ``heat_co2e``: Heat-related CO2e.
                - ``cooling_co2e``: Cooling-related CO2e.
                - ``co2_kg``: Total CO2 in kilograms.
                - ``ch4_kg``: Total CH4 in kilograms.
                - ``n2o_kg``: Total N2O in kilograms.
                - ``data_quality_score``: Quality score (0.0-1.0).
                - ``validation_status``: PASS or FAIL.
                - ``calculation_method``: Method used (e.g., "location").

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            TypeError: If ``result`` is not a dictionary.
            ValueError: If ``result`` is empty.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_output({
            ...     "total_co2e_tonnes": 1646.84,
            ...     "validation_status": "PASS",
            ... })
            >>> assert len(h) == 64
        """
        if not result:
            raise ValueError("Output result must not be empty")

        logger.info(
            "Hashing output result with %d keys for %s",
            len(result),
            self.AGENT_ID,
        )
        return self._compute_hash("output", result)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Grid Emission Factor
    # ------------------------------------------------------------------

    def hash_grid_factor(
        self,
        region_id: str,
        source: str,
        year: int,
        co2_ef: float,
        ch4_ef: float,
        n2o_ef: float,
    ) -> str:
        """Hash a grid emission factor lookup and add to the provenance chain.

        Records the retrieval of grid-average emission factors from a
        reference database (e.g., eGRID, IEA, DEFRA, national grid
        operators). These factors are used in the location-based method
        to convert electricity consumption (MWh) to greenhouse gas
        emissions (tCO2e).

        The grid emission factor is a critical input to the calculation.
        Hashing it ensures that any change to the factor database is
        captured in the provenance chain, enabling auditors to verify
        which exact factors were used.

        Args:
            region_id: Grid region identifier (e.g., ``"US-WECC"``,
                ``"US-ERCOT"``, ``"GB"``, ``"DE"``, ``"JP-TEPCO"``).
                This identifies the geographic scope of the emission
                factor.
            source: Reference database or publication source (e.g.,
                ``"eGRID2023"``, ``"IEA2023"``, ``"DEFRA2024"``,
                ``"NationalGrid2023"``).
            year: Reference year for the emission factor data (e.g.,
                2023, 2024). This is the year the factor data represents,
                not necessarily the publication year.
            co2_ef: CO2 emission factor in tonnes CO2 per MWh
                (tCO2/MWh) or kg CO2 per kWh depending on the source.
                Must be non-negative.
            ch4_ef: CH4 emission factor in tonnes CH4 per MWh
                (tCH4/MWh) or equivalent units. Must be non-negative.
            n2o_ef: N2O emission factor in tonnes N2O per MWh
                (tN2O/MWh) or equivalent units. Must be non-negative.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``region_id`` or ``source`` is empty,
                ``year`` is not positive, or any emission factor
                is negative.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_grid_factor(
            ...     region_id="US-WECC",
            ...     source="eGRID2023",
            ...     year=2023,
            ...     co2_ef=0.3127,
            ...     ch4_ef=0.0000112,
            ...     n2o_ef=0.0000042,
            ... )
            >>> assert len(h) == 64
        """
        if not region_id:
            raise ValueError("region_id must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        if year <= 0:
            raise ValueError(f"year must be positive, got {year}")
        if co2_ef < 0:
            raise ValueError(f"co2_ef must be non-negative, got {co2_ef}")
        if ch4_ef < 0:
            raise ValueError(f"ch4_ef must be non-negative, got {ch4_ef}")
        if n2o_ef < 0:
            raise ValueError(f"n2o_ef must be non-negative, got {n2o_ef}")

        data = {
            "region_id": region_id,
            "source": source,
            "year": year,
            "co2_ef": _safe_str(co2_ef),
            "ch4_ef": _safe_str(ch4_ef),
            "n2o_ef": _safe_str(n2o_ef),
        }

        logger.info(
            "Hashing grid factor: region=%s source=%s year=%d",
            region_id,
            source,
            year,
        )
        return self._compute_hash("grid_factor_lookup", data)

    def hash_grid_factor_extended(
        self,
        region_id: str,
        source: str,
        year: int,
        co2_ef: float,
        ch4_ef: float,
        n2o_ef: float,
        total_co2e_ef: float,
        generation_mix: Optional[Dict[str, float]] = None,
        data_quality: Optional[str] = None,
        vintage: Optional[str] = None,
    ) -> str:
        """Hash an extended grid emission factor lookup with generation mix.

        Similar to ``hash_grid_factor`` but includes additional metadata
        about the electricity generation mix, data quality assessment,
        and factor vintage for enhanced audit trails.

        Args:
            region_id: Grid region identifier.
            source: Reference database or publication source.
            year: Reference year for the emission factor data.
            co2_ef: CO2 emission factor (tCO2/MWh).
            ch4_ef: CH4 emission factor (tCH4/MWh).
            n2o_ef: N2O emission factor (tN2O/MWh).
            total_co2e_ef: Combined CO2e emission factor (tCO2e/MWh).
            generation_mix: Optional dictionary mapping fuel sources to
                percentages (e.g., ``{"coal": 0.25, "gas": 0.40}``).
            data_quality: Optional data quality indicator (e.g.,
                ``"measured"``, ``"estimated"``, ``"default"``).
            vintage: Optional vintage label (e.g., ``"2023-final"``,
                ``"2024-preliminary"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If required string fields are empty or numeric
                values are negative.
        """
        if not region_id:
            raise ValueError("region_id must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        if year <= 0:
            raise ValueError(f"year must be positive, got {year}")

        data: Dict[str, Any] = {
            "region_id": region_id,
            "source": source,
            "year": year,
            "co2_ef": _safe_str(co2_ef),
            "ch4_ef": _safe_str(ch4_ef),
            "n2o_ef": _safe_str(n2o_ef),
            "total_co2e_ef": _safe_str(total_co2e_ef),
        }
        if generation_mix is not None:
            data["generation_mix"] = {
                k: _safe_str(v) for k, v in generation_mix.items()
            }
        if data_quality is not None:
            data["data_quality"] = data_quality
        if vintage is not None:
            data["vintage"] = vintage

        logger.info(
            "Hashing extended grid factor: region=%s source=%s year=%d "
            "co2e_ef=%s",
            region_id,
            source,
            year,
            _safe_str(total_co2e_ef),
        )
        return self._compute_hash("grid_factor_lookup", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: T&D Loss
    # ------------------------------------------------------------------

    def hash_td_loss(
        self,
        country_code: str,
        td_loss_pct: float,
        method: str,
    ) -> str:
        """Hash T&D loss parameters and add to the provenance chain.

        Records the transmission and distribution (T&D) loss percentage
        used to adjust gross electricity consumption to account for
        energy lost between generation and point of delivery. The
        GHG Protocol Scope 2 Guidance recommends applying T&D losses
        when using the location-based method.

        Common T&D loss sources include:
        - IEA World Energy Balances
        - National grid operator statistics
        - Regional transmission organization data
        - Country-specific default values from IPCC

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g.,
                ``"US"``, ``"GB"``, ``"DE"``, ``"JP"``). Used to
                identify the country whose T&D loss data applies.
            td_loss_pct: Transmission and distribution loss as a
                percentage (e.g., 5.3 means 5.3% loss). Must be
                in the range [0.0, 100.0].
            method: Method used to determine the T&D loss value.
                Common values: ``"iea_default"``, ``"national_grid"``,
                ``"measured"``, ``"estimated"``, ``"country_default"``.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``country_code`` or ``method`` is empty,
                or ``td_loss_pct`` is outside [0.0, 100.0].

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_td_loss(
            ...     country_code="US",
            ...     td_loss_pct=5.3,
            ...     method="iea_default",
            ... )
            >>> assert len(h) == 64
        """
        if not country_code:
            raise ValueError("country_code must not be empty")
        if not method:
            raise ValueError("method must not be empty")
        if td_loss_pct < 0.0 or td_loss_pct > 100.0:
            raise ValueError(
                f"td_loss_pct must be in [0.0, 100.0], got {td_loss_pct}"
            )

        data = {
            "country_code": country_code,
            "td_loss_pct": _safe_str(td_loss_pct),
            "method": method,
        }

        logger.info(
            "Hashing T&D loss: country=%s td_loss=%.2f%% method=%s",
            country_code,
            td_loss_pct,
            method,
        )
        return self._compute_hash("td_loss", data)

    def hash_td_loss_extended(
        self,
        country_code: str,
        td_loss_pct: float,
        method: str,
        source: str,
        reference_year: int,
        voltage_level: Optional[str] = None,
        transmission_loss_pct: Optional[float] = None,
        distribution_loss_pct: Optional[float] = None,
    ) -> str:
        """Hash extended T&D loss parameters with source attribution.

        Similar to ``hash_td_loss`` but includes additional fields for
        the loss data source, reference year, voltage level, and separate
        transmission vs. distribution loss breakdowns.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            td_loss_pct: Total T&D loss percentage [0.0, 100.0].
            method: Method used to determine the T&D loss value.
            source: Data source reference (e.g., ``"IEA WEB 2023"``).
            reference_year: Year the T&D loss data represents.
            voltage_level: Optional voltage level (e.g., ``"high"``,
                ``"medium"``, ``"low"``).
            transmission_loss_pct: Optional transmission-only loss
                percentage.
            distribution_loss_pct: Optional distribution-only loss
                percentage.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If required fields are empty or numeric values
                are out of range.
        """
        if not country_code:
            raise ValueError("country_code must not be empty")
        if not method:
            raise ValueError("method must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        if td_loss_pct < 0.0 or td_loss_pct > 100.0:
            raise ValueError(
                f"td_loss_pct must be in [0.0, 100.0], got {td_loss_pct}"
            )

        data: Dict[str, Any] = {
            "country_code": country_code,
            "td_loss_pct": _safe_str(td_loss_pct),
            "method": method,
            "source": source,
            "reference_year": reference_year,
        }
        if voltage_level is not None:
            data["voltage_level"] = voltage_level
        if transmission_loss_pct is not None:
            data["transmission_loss_pct"] = _safe_str(transmission_loss_pct)
        if distribution_loss_pct is not None:
            data["distribution_loss_pct"] = _safe_str(distribution_loss_pct)

        logger.info(
            "Hashing extended T&D loss: country=%s td_loss=%.2f%% "
            "source=%s year=%d",
            country_code,
            td_loss_pct,
            source,
            reference_year,
        )
        return self._compute_hash("td_loss", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Electricity Calculation
    # ------------------------------------------------------------------

    def hash_electricity_calculation(
        self,
        consumption_mwh: float,
        ef_co2e: float,
        td_loss_pct: float,
        total_co2e: float,
    ) -> str:
        """Hash an electricity emission calculation and add to the chain.

        Records the core Scope 2 location-based electricity emission
        calculation. The formula implemented is:

            total_co2e = consumption_mwh * ef_co2e * (1 + td_loss_pct / 100)

        This method hashes the inputs and the calculated result to ensure
        that the arithmetic is captured in the provenance trail. Auditors
        can verify the total_co2e value by recomputing from the hashed
        inputs.

        Args:
            consumption_mwh: Electricity consumption in megawatt-hours
                (MWh). Must be non-negative.
            ef_co2e: Combined CO2e emission factor in tonnes CO2e per
                MWh (tCO2e/MWh). This is typically the sum of per-gas
                emission factors weighted by their GWP values. Must be
                non-negative.
            td_loss_pct: Transmission and distribution loss percentage
                applied to adjust consumption. Must be in [0.0, 100.0].
            total_co2e: The calculated total CO2e emissions resulting
                from the electricity consumption. This is the output
                of the deterministic formula.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``consumption_mwh`` or ``ef_co2e`` is negative,
                or ``td_loss_pct`` is outside [0.0, 100.0].

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_electricity_calculation(
            ...     consumption_mwh=5000.0,
            ...     ef_co2e=0.3127,
            ...     td_loss_pct=5.3,
            ...     total_co2e=1646.84,
            ... )
            >>> assert len(h) == 64
        """
        if consumption_mwh < 0:
            raise ValueError(
                f"consumption_mwh must be non-negative, got {consumption_mwh}"
            )
        if ef_co2e < 0:
            raise ValueError(
                f"ef_co2e must be non-negative, got {ef_co2e}"
            )
        if td_loss_pct < 0.0 or td_loss_pct > 100.0:
            raise ValueError(
                f"td_loss_pct must be in [0.0, 100.0], got {td_loss_pct}"
            )

        data = {
            "consumption_mwh": _safe_str(consumption_mwh),
            "ef_co2e": _safe_str(ef_co2e),
            "td_loss_pct": _safe_str(td_loss_pct),
            "total_co2e": _safe_str(total_co2e),
        }

        logger.info(
            "Hashing electricity calc: %.2f MWh * %.6f tCO2e/MWh "
            "(T&D=%.2f%%) = %.4f tCO2e",
            consumption_mwh,
            ef_co2e,
            td_loss_pct,
            total_co2e,
        )
        return self._compute_hash("electricity_calc", data)

    def hash_electricity_calculation_extended(
        self,
        consumption_mwh: float,
        ef_co2e: float,
        td_loss_pct: float,
        total_co2e: float,
        region_id: str,
        facility_id: str,
        reporting_period: str,
        consumption_source: Optional[str] = None,
        meter_id: Optional[str] = None,
        load_factor: Optional[float] = None,
    ) -> str:
        """Hash an extended electricity emission calculation with context.

        Similar to ``hash_electricity_calculation`` but includes
        additional metadata for enhanced traceability including facility,
        region, reporting period, and metering details.

        Args:
            consumption_mwh: Electricity consumption in MWh.
            ef_co2e: Combined CO2e emission factor (tCO2e/MWh).
            td_loss_pct: T&D loss percentage [0.0, 100.0].
            total_co2e: Calculated total CO2e emissions.
            region_id: Grid region identifier.
            facility_id: Facility identifier.
            reporting_period: Reporting period string.
            consumption_source: Optional source of consumption data
                (e.g., ``"utility_bill"``, ``"smart_meter"``).
            meter_id: Optional meter identifier.
            load_factor: Optional facility load factor.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if consumption_mwh < 0:
            raise ValueError(
                f"consumption_mwh must be non-negative, got {consumption_mwh}"
            )
        if ef_co2e < 0:
            raise ValueError(
                f"ef_co2e must be non-negative, got {ef_co2e}"
            )

        data: Dict[str, Any] = {
            "consumption_mwh": _safe_str(consumption_mwh),
            "ef_co2e": _safe_str(ef_co2e),
            "td_loss_pct": _safe_str(td_loss_pct),
            "total_co2e": _safe_str(total_co2e),
            "region_id": region_id,
            "facility_id": facility_id,
            "reporting_period": reporting_period,
        }
        if consumption_source is not None:
            data["consumption_source"] = consumption_source
        if meter_id is not None:
            data["meter_id"] = meter_id
        if load_factor is not None:
            data["load_factor"] = _safe_str(load_factor)

        logger.info(
            "Hashing extended electricity calc: facility=%s region=%s "
            "%.2f MWh = %.4f tCO2e",
            facility_id,
            region_id,
            consumption_mwh,
            total_co2e,
        )
        return self._compute_hash("electricity_calc", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Steam / Heat / Cooling
    # ------------------------------------------------------------------

    def hash_steam_heat_cooling(
        self,
        energy_type: str,
        consumption_gj: float,
        ef: float,
        total_co2e: float,
    ) -> str:
        """Hash a steam, heat, or cooling emission calculation.

        Records the emission calculation for purchased steam, district
        heating, or cooling. These energy types are included in Scope 2
        under the GHG Protocol and use energy-based emission factors
        (tCO2e per GJ or per MWh thermal).

        Supported energy types:
        - ``"steam"``: Purchased steam
        - ``"heat"``: District heating / hot water
        - ``"cooling"``: District cooling / chilled water

        The formula is:
            total_co2e = consumption_gj * ef

        Args:
            energy_type: Type of energy purchased. Must be one of
                ``"steam"``, ``"heat"``, or ``"cooling"``.
            consumption_gj: Energy consumption in gigajoules (GJ).
                Must be non-negative.
            ef: Emission factor in tonnes CO2e per GJ (tCO2e/GJ).
                Must be non-negative.
            total_co2e: The calculated total CO2e emissions.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``energy_type`` is not recognized, or if
                ``consumption_gj`` or ``ef`` is negative.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_steam_heat_cooling(
            ...     energy_type="steam",
            ...     consumption_gj=1200.0,
            ...     ef=0.0667,
            ...     total_co2e=80.04,
            ... )
            >>> assert len(h) == 64
        """
        valid_types = {"steam", "heat", "cooling"}
        if energy_type not in valid_types:
            raise ValueError(
                f"energy_type must be one of {valid_types}, "
                f"got '{energy_type}'"
            )
        if consumption_gj < 0:
            raise ValueError(
                f"consumption_gj must be non-negative, got {consumption_gj}"
            )
        if ef < 0:
            raise ValueError(
                f"ef must be non-negative, got {ef}"
            )

        data = {
            "energy_type": energy_type,
            "consumption_gj": _safe_str(consumption_gj),
            "ef": _safe_str(ef),
            "total_co2e": _safe_str(total_co2e),
        }

        logger.info(
            "Hashing %s calc: %.2f GJ * %.6f tCO2e/GJ = %.4f tCO2e",
            energy_type,
            consumption_gj,
            ef,
            total_co2e,
        )
        return self._compute_hash("steam_heat_cooling_calc", data)

    def hash_steam_heat_cooling_extended(
        self,
        energy_type: str,
        consumption_gj: float,
        ef: float,
        total_co2e: float,
        supplier_id: Optional[str] = None,
        supplier_ef_source: Optional[str] = None,
        efficiency: Optional[float] = None,
        fuel_mix: Optional[Dict[str, float]] = None,
        facility_id: Optional[str] = None,
        reporting_period: Optional[str] = None,
    ) -> str:
        """Hash an extended steam/heat/cooling calculation with supplier data.

        Similar to ``hash_steam_heat_cooling`` but includes additional
        metadata for supplier-specific emission factors, thermal
        efficiency, and fuel mix information.

        Args:
            energy_type: Type of energy (``"steam"``, ``"heat"``,
                ``"cooling"``).
            consumption_gj: Energy consumption in GJ.
            ef: Emission factor (tCO2e/GJ).
            total_co2e: Calculated total CO2e.
            supplier_id: Optional supplier identifier.
            supplier_ef_source: Optional source of the supplier-specific
                emission factor.
            efficiency: Optional thermal efficiency (0.0-1.0).
            fuel_mix: Optional fuel mix dictionary.
            facility_id: Optional facility identifier.
            reporting_period: Optional reporting period string.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        valid_types = {"steam", "heat", "cooling"}
        if energy_type not in valid_types:
            raise ValueError(
                f"energy_type must be one of {valid_types}, "
                f"got '{energy_type}'"
            )
        if consumption_gj < 0:
            raise ValueError(
                f"consumption_gj must be non-negative, got {consumption_gj}"
            )

        data: Dict[str, Any] = {
            "energy_type": energy_type,
            "consumption_gj": _safe_str(consumption_gj),
            "ef": _safe_str(ef),
            "total_co2e": _safe_str(total_co2e),
        }
        if supplier_id is not None:
            data["supplier_id"] = supplier_id
        if supplier_ef_source is not None:
            data["supplier_ef_source"] = supplier_ef_source
        if efficiency is not None:
            data["efficiency"] = _safe_str(efficiency)
        if fuel_mix is not None:
            data["fuel_mix"] = {
                k: _safe_str(v) for k, v in fuel_mix.items()
            }
        if facility_id is not None:
            data["facility_id"] = facility_id
        if reporting_period is not None:
            data["reporting_period"] = reporting_period

        logger.info(
            "Hashing extended %s calc: %.2f GJ = %.4f tCO2e "
            "(supplier=%s)",
            energy_type,
            consumption_gj,
            total_co2e,
            supplier_id or "N/A",
        )
        return self._compute_hash("steam_heat_cooling_calc", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Gas Breakdown
    # ------------------------------------------------------------------

    def hash_gas_breakdown(
        self,
        co2_kg: float,
        ch4_kg: float,
        n2o_kg: float,
        gwp_source: str,
        total_co2e: float,
    ) -> str:
        """Hash a per-gas greenhouse gas breakdown.

        Records the decomposition of total emissions into individual
        greenhouse gas contributions (CO2, CH4, N2O) along with the
        GWP source used for the CO2e conversion. This breakdown is
        required by the GHG Protocol and most regulatory frameworks.

        The per-gas breakdown is computed deterministically:
            total_co2e = co2_kg + (ch4_kg * gwp_ch4) + (n2o_kg * gwp_n2o)

        where gwp_ch4 and gwp_n2o are the Global Warming Potential
        values from the specified GWP source (e.g., IPCC AR5, AR6).

        Args:
            co2_kg: CO2 emissions in kilograms. Must be non-negative.
            ch4_kg: CH4 emissions in kilograms. Must be non-negative.
            n2o_kg: N2O emissions in kilograms. Must be non-negative.
            gwp_source: Source of GWP values used for the CO2e
                conversion. Common values: ``"IPCC_AR5"``,
                ``"IPCC_AR6"``, ``"IPCC_AR4"``, ``"IPCC_SAR"``.
            total_co2e: Total CO2-equivalent emissions in kilograms,
                computed using the GWP values from the specified source.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If any gas quantity is negative, ``gwp_source``
                is empty, or ``total_co2e`` is negative.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_gas_breakdown(
            ...     co2_kg=1563500.0,
            ...     ch4_kg=56.0,
            ...     n2o_kg=21.0,
            ...     gwp_source="IPCC_AR5",
            ...     total_co2e=1571653.0,
            ... )
            >>> assert len(h) == 64
        """
        if co2_kg < 0:
            raise ValueError(f"co2_kg must be non-negative, got {co2_kg}")
        if ch4_kg < 0:
            raise ValueError(f"ch4_kg must be non-negative, got {ch4_kg}")
        if n2o_kg < 0:
            raise ValueError(f"n2o_kg must be non-negative, got {n2o_kg}")
        if not gwp_source:
            raise ValueError("gwp_source must not be empty")
        if total_co2e < 0:
            raise ValueError(
                f"total_co2e must be non-negative, got {total_co2e}"
            )

        data = {
            "co2_kg": _safe_str(co2_kg),
            "ch4_kg": _safe_str(ch4_kg),
            "n2o_kg": _safe_str(n2o_kg),
            "gwp_source": gwp_source,
            "total_co2e": _safe_str(total_co2e),
        }

        logger.info(
            "Hashing gas breakdown: CO2=%.2f CH4=%.4f N2O=%.4f "
            "GWP=%s total=%.2f kg CO2e",
            co2_kg,
            ch4_kg,
            n2o_kg,
            gwp_source,
            total_co2e,
        )
        return self._compute_hash("gas_breakdown", data)

    def hash_gas_breakdown_extended(
        self,
        co2_kg: float,
        ch4_kg: float,
        n2o_kg: float,
        gwp_source: str,
        total_co2e: float,
        gwp_ch4: float,
        gwp_n2o: float,
        biogenic_co2_kg: Optional[float] = None,
        sf6_kg: Optional[float] = None,
        hfc_kg: Optional[float] = None,
        pfc_kg: Optional[float] = None,
    ) -> str:
        """Hash an extended per-gas breakdown including GWP factors.

        Similar to ``hash_gas_breakdown`` but includes the actual GWP
        factor values used and optional additional greenhouse gases.

        Args:
            co2_kg: CO2 emissions in kilograms.
            ch4_kg: CH4 emissions in kilograms.
            n2o_kg: N2O emissions in kilograms.
            gwp_source: Source of GWP values.
            total_co2e: Total CO2e emissions in kilograms.
            gwp_ch4: GWP factor used for CH4 (e.g., 28 for AR5).
            gwp_n2o: GWP factor used for N2O (e.g., 265 for AR5).
            biogenic_co2_kg: Optional biogenic CO2 (reported separately
                per GHG Protocol guidance).
            sf6_kg: Optional SF6 emissions in kilograms.
            hfc_kg: Optional HFC emissions in kilograms.
            pfc_kg: Optional PFC emissions in kilograms.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if co2_kg < 0:
            raise ValueError(f"co2_kg must be non-negative, got {co2_kg}")
        if ch4_kg < 0:
            raise ValueError(f"ch4_kg must be non-negative, got {ch4_kg}")
        if n2o_kg < 0:
            raise ValueError(f"n2o_kg must be non-negative, got {n2o_kg}")
        if not gwp_source:
            raise ValueError("gwp_source must not be empty")

        data: Dict[str, Any] = {
            "co2_kg": _safe_str(co2_kg),
            "ch4_kg": _safe_str(ch4_kg),
            "n2o_kg": _safe_str(n2o_kg),
            "gwp_source": gwp_source,
            "total_co2e": _safe_str(total_co2e),
            "gwp_ch4": _safe_str(gwp_ch4),
            "gwp_n2o": _safe_str(gwp_n2o),
        }
        if biogenic_co2_kg is not None:
            data["biogenic_co2_kg"] = _safe_str(biogenic_co2_kg)
        if sf6_kg is not None:
            data["sf6_kg"] = _safe_str(sf6_kg)
        if hfc_kg is not None:
            data["hfc_kg"] = _safe_str(hfc_kg)
        if pfc_kg is not None:
            data["pfc_kg"] = _safe_str(pfc_kg)

        logger.info(
            "Hashing extended gas breakdown: CO2=%.2f CH4=%.4f N2O=%.4f "
            "GWP=%s (CH4=%s, N2O=%s)",
            co2_kg,
            ch4_kg,
            n2o_kg,
            gwp_source,
            _safe_str(gwp_ch4),
            _safe_str(gwp_n2o),
        )
        return self._compute_hash("gas_breakdown", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: GWP Conversion
    # ------------------------------------------------------------------

    def hash_gwp_conversion(
        self,
        gas: str,
        quantity_kg: float,
        gwp_factor: float,
        co2e_kg: float,
    ) -> str:
        """Hash an individual GWP conversion step.

        Records the conversion of a single greenhouse gas quantity to
        its CO2-equivalent using the specified Global Warming Potential
        factor. This is a granular provenance record for each gas in
        the breakdown.

        The conversion formula is:
            co2e_kg = quantity_kg * gwp_factor

        For CO2, the gwp_factor is always 1.0. For CH4 and N2O, the
        factor depends on the GWP source (AR4, AR5, AR6).

        Args:
            gas: Greenhouse gas identifier. Common values:
                ``"CO2"``, ``"CH4"``, ``"N2O"``, ``"SF6"``,
                ``"HFC-134a"``, etc.
            quantity_kg: Mass of the greenhouse gas in kilograms.
                Must be non-negative.
            gwp_factor: Global Warming Potential factor used for
                conversion to CO2e. Must be positive.
            co2e_kg: The resulting CO2-equivalent mass in kilograms.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``gas`` is empty, ``quantity_kg`` is negative,
                or ``gwp_factor`` is not positive.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_gwp_conversion(
            ...     gas="CH4",
            ...     quantity_kg=56.0,
            ...     gwp_factor=28.0,
            ...     co2e_kg=1568.0,
            ... )
            >>> assert len(h) == 64
        """
        if not gas:
            raise ValueError("gas must not be empty")
        if quantity_kg < 0:
            raise ValueError(
                f"quantity_kg must be non-negative, got {quantity_kg}"
            )
        if gwp_factor <= 0:
            raise ValueError(
                f"gwp_factor must be positive, got {gwp_factor}"
            )

        data = {
            "gas": gas,
            "quantity_kg": _safe_str(quantity_kg),
            "gwp_factor": _safe_str(gwp_factor),
            "co2e_kg": _safe_str(co2e_kg),
        }

        logger.info(
            "Hashing GWP conversion: %s %.4f kg * GWP %.1f = %.4f kg CO2e",
            gas,
            quantity_kg,
            gwp_factor,
            co2e_kg,
        )
        return self._compute_hash("gwp_conversion", data)

    def hash_gwp_conversion_extended(
        self,
        gas: str,
        quantity_kg: float,
        gwp_factor: float,
        co2e_kg: float,
        gwp_source: str,
        gwp_time_horizon: int = 100,
        includes_feedback: bool = False,
    ) -> str:
        """Hash an extended GWP conversion step with source attribution.

        Similar to ``hash_gwp_conversion`` but includes additional
        fields for GWP source, time horizon, and whether climate-carbon
        feedbacks are included in the GWP factor.

        Args:
            gas: Greenhouse gas identifier.
            quantity_kg: Mass of the greenhouse gas in kilograms.
            gwp_factor: GWP factor for conversion.
            co2e_kg: Resulting CO2e mass in kilograms.
            gwp_source: Source of GWP value (e.g., ``"IPCC_AR5"``).
            gwp_time_horizon: Time horizon in years (default 100).
            includes_feedback: Whether the GWP includes climate-carbon
                feedback effects.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not gas:
            raise ValueError("gas must not be empty")
        if quantity_kg < 0:
            raise ValueError(
                f"quantity_kg must be non-negative, got {quantity_kg}"
            )
        if gwp_factor <= 0:
            raise ValueError(
                f"gwp_factor must be positive, got {gwp_factor}"
            )

        data = {
            "gas": gas,
            "quantity_kg": _safe_str(quantity_kg),
            "gwp_factor": _safe_str(gwp_factor),
            "co2e_kg": _safe_str(co2e_kg),
            "gwp_source": gwp_source,
            "gwp_time_horizon": gwp_time_horizon,
            "includes_feedback": includes_feedback,
        }

        logger.info(
            "Hashing extended GWP conversion: %s %.4f kg "
            "GWP=%s (%.1f, %dyr, feedback=%s)",
            gas,
            quantity_kg,
            gwp_source,
            gwp_factor,
            gwp_time_horizon,
            includes_feedback,
        )
        return self._compute_hash("gwp_conversion", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Compliance Check
    # ------------------------------------------------------------------

    def hash_compliance_check(
        self,
        framework: str,
        status: str,
        findings_count: int,
    ) -> str:
        """Hash a regulatory compliance check result.

        Records the outcome of a compliance validation against a
        regulatory framework. The compliance check verifies that the
        Scope 2 location-based calculation meets all requirements
        specified by the framework (data completeness, methodology,
        reporting format, etc.).

        Supported frameworks:
        - ``"GHG_PROTOCOL_SCOPE2"``: GHG Protocol Scope 2 Guidance
        - ``"ISO_14064"``: ISO 14064-1:2018
        - ``"CSRD_ESRS_E1"``: CSRD / European Sustainability Reporting
        - ``"UK_SECR"``: UK Streamlined Energy and Carbon Reporting
        - ``"EPA_EGRID"``: EPA eGRID-based reporting
        - ``"EU_ETS"``: EU Emissions Trading System
        - ``"CDP"``: CDP Climate Change Questionnaire
        - ``"RE100"``: RE100 Reporting Criteria

        Args:
            framework: Regulatory framework identifier. Must not be
                empty.
            status: Compliance status. Must be one of ``"PASS"``,
                ``"FAIL"``, ``"PARTIAL"``, ``"NOT_APPLICABLE"``, or
                ``"PENDING"``.
            findings_count: Number of compliance findings (issues)
                detected. Must be non-negative. Zero findings
                typically corresponds to ``"PASS"`` status.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``framework`` is empty, ``status`` is not
                recognized, or ``findings_count`` is negative.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_compliance_check(
            ...     framework="GHG_PROTOCOL_SCOPE2",
            ...     status="PASS",
            ...     findings_count=0,
            ... )
            >>> assert len(h) == 64
        """
        if not framework:
            raise ValueError("framework must not be empty")
        valid_statuses = {"PASS", "FAIL", "PARTIAL", "NOT_APPLICABLE", "PENDING"}
        if status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got '{status}'"
            )
        if findings_count < 0:
            raise ValueError(
                f"findings_count must be non-negative, got {findings_count}"
            )

        data = {
            "framework": framework,
            "status": status,
            "findings_count": findings_count,
        }

        logger.info(
            "Hashing compliance check: framework=%s status=%s findings=%d",
            framework,
            status,
            findings_count,
        )
        return self._compute_hash("compliance_check", data)

    def hash_compliance_check_extended(
        self,
        framework: str,
        status: str,
        findings_count: int,
        findings: Optional[List[Dict[str, Any]]] = None,
        checked_rules: Optional[int] = None,
        passed_rules: Optional[int] = None,
        failed_rules: Optional[int] = None,
        warnings_count: Optional[int] = None,
        check_timestamp: Optional[str] = None,
        framework_version: Optional[str] = None,
    ) -> str:
        """Hash an extended compliance check result with detailed findings.

        Similar to ``hash_compliance_check`` but includes additional
        fields for individual findings, rule counts, warnings, and
        framework version information.

        Args:
            framework: Regulatory framework identifier.
            status: Compliance status (PASS/FAIL/PARTIAL/NOT_APPLICABLE/PENDING).
            findings_count: Number of compliance findings.
            findings: Optional list of finding dictionaries, each
                containing ``rule_id``, ``severity``, ``message``.
            checked_rules: Optional total number of rules checked.
            passed_rules: Optional number of rules that passed.
            failed_rules: Optional number of rules that failed.
            warnings_count: Optional number of warning-level findings.
            check_timestamp: Optional ISO timestamp of the check.
            framework_version: Optional framework version string.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not framework:
            raise ValueError("framework must not be empty")
        valid_statuses = {"PASS", "FAIL", "PARTIAL", "NOT_APPLICABLE", "PENDING"}
        if status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got '{status}'"
            )
        if findings_count < 0:
            raise ValueError(
                f"findings_count must be non-negative, got {findings_count}"
            )

        data: Dict[str, Any] = {
            "framework": framework,
            "status": status,
            "findings_count": findings_count,
        }
        if findings is not None:
            data["findings"] = findings
        if checked_rules is not None:
            data["checked_rules"] = checked_rules
        if passed_rules is not None:
            data["passed_rules"] = passed_rules
        if failed_rules is not None:
            data["failed_rules"] = failed_rules
        if warnings_count is not None:
            data["warnings_count"] = warnings_count
        if check_timestamp is not None:
            data["check_timestamp"] = check_timestamp
        if framework_version is not None:
            data["framework_version"] = framework_version

        logger.info(
            "Hashing extended compliance check: framework=%s(%s) "
            "status=%s findings=%d rules=%s/%s",
            framework,
            framework_version or "N/A",
            status,
            findings_count,
            _safe_str(passed_rules),
            _safe_str(checked_rules),
        )
        return self._compute_hash("compliance_check", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Uncertainty
    # ------------------------------------------------------------------

    def hash_uncertainty(
        self,
        method: str,
        mean: float,
        std_dev: float,
        ci_lower: float,
        ci_upper: float,
    ) -> str:
        """Hash an uncertainty quantification result.

        Records the outcome of an uncertainty analysis for the Scope 2
        location-based emission calculation. Uncertainty may be computed
        using Monte Carlo simulation, analytical error propagation, or
        IPCC Approach 1/2 methods.

        The confidence interval (ci_lower, ci_upper) represents the
        range within which the true emission value is expected to fall
        with a specified confidence level (typically 95%).

        Args:
            method: Uncertainty quantification method used. Common
                values: ``"monte_carlo"``, ``"analytical"``,
                ``"ipcc_approach_1"``, ``"ipcc_approach_2"``,
                ``"error_propagation"``.
            mean: Mean (expected value) of the emission estimate.
            std_dev: Standard deviation of the emission estimate.
                Must be non-negative.
            ci_lower: Lower bound of the confidence interval.
            ci_upper: Upper bound of the confidence interval.
                Must be greater than or equal to ``ci_lower``.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``method`` is empty, ``std_dev`` is negative,
                or ``ci_upper`` is less than ``ci_lower``.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_uncertainty(
            ...     method="monte_carlo",
            ...     mean=1646.84,
            ...     std_dev=82.34,
            ...     ci_lower=1485.46,
            ...     ci_upper=1808.22,
            ... )
            >>> assert len(h) == 64
        """
        if not method:
            raise ValueError("method must not be empty")
        if std_dev < 0:
            raise ValueError(
                f"std_dev must be non-negative, got {std_dev}"
            )
        if ci_upper < ci_lower:
            raise ValueError(
                f"ci_upper ({ci_upper}) must be >= ci_lower ({ci_lower})"
            )

        data = {
            "method": method,
            "mean": _safe_str(mean),
            "std_dev": _safe_str(std_dev),
            "ci_lower": _safe_str(ci_lower),
            "ci_upper": _safe_str(ci_upper),
        }

        logger.info(
            "Hashing uncertainty: method=%s mean=%.4f std=%.4f "
            "CI=[%.4f, %.4f]",
            method,
            mean,
            std_dev,
            ci_lower,
            ci_upper,
        )
        return self._compute_hash("uncertainty", data)

    def hash_uncertainty_extended(
        self,
        method: str,
        mean: float,
        std_dev: float,
        ci_lower: float,
        ci_upper: float,
        confidence_level: float = 0.95,
        n_simulations: Optional[int] = None,
        cv_percent: Optional[float] = None,
        percentile_5: Optional[float] = None,
        percentile_25: Optional[float] = None,
        percentile_50: Optional[float] = None,
        percentile_75: Optional[float] = None,
        percentile_95: Optional[float] = None,
        variance_contributions: Optional[Dict[str, float]] = None,
        data_quality_score: Optional[float] = None,
    ) -> str:
        """Hash an extended uncertainty result with full statistical detail.

        Similar to ``hash_uncertainty`` but includes additional
        statistical measures such as percentiles, number of Monte Carlo
        simulations, coefficient of variation, variance contributions
        by parameter, and data quality scoring.

        Args:
            method: Uncertainty method (e.g., ``"monte_carlo"``).
            mean: Mean of the emission estimate.
            std_dev: Standard deviation.
            ci_lower: Lower CI bound.
            ci_upper: Upper CI bound.
            confidence_level: Confidence level for the CI (default 0.95).
            n_simulations: Optional number of Monte Carlo simulations.
            cv_percent: Optional coefficient of variation (%).
            percentile_5: Optional 5th percentile.
            percentile_25: Optional 25th percentile.
            percentile_50: Optional 50th percentile (median).
            percentile_75: Optional 75th percentile.
            percentile_95: Optional 95th percentile.
            variance_contributions: Optional dictionary mapping parameter
                names to their contribution to total variance.
            data_quality_score: Optional data quality score (0.0-1.0).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not method:
            raise ValueError("method must not be empty")
        if std_dev < 0:
            raise ValueError(
                f"std_dev must be non-negative, got {std_dev}"
            )

        data: Dict[str, Any] = {
            "method": method,
            "mean": _safe_str(mean),
            "std_dev": _safe_str(std_dev),
            "ci_lower": _safe_str(ci_lower),
            "ci_upper": _safe_str(ci_upper),
            "confidence_level": _safe_str(confidence_level),
        }
        if n_simulations is not None:
            data["n_simulations"] = n_simulations
        if cv_percent is not None:
            data["cv_percent"] = _safe_str(cv_percent)
        if percentile_5 is not None:
            data["percentile_5"] = _safe_str(percentile_5)
        if percentile_25 is not None:
            data["percentile_25"] = _safe_str(percentile_25)
        if percentile_50 is not None:
            data["percentile_50"] = _safe_str(percentile_50)
        if percentile_75 is not None:
            data["percentile_75"] = _safe_str(percentile_75)
        if percentile_95 is not None:
            data["percentile_95"] = _safe_str(percentile_95)
        if variance_contributions is not None:
            data["variance_contributions"] = {
                k: _safe_str(v) for k, v in variance_contributions.items()
            }
        if data_quality_score is not None:
            data["data_quality_score"] = _safe_str(data_quality_score)

        logger.info(
            "Hashing extended uncertainty: method=%s mean=%.4f "
            "CI=[%.4f, %.4f] n_sim=%s",
            method,
            mean,
            ci_lower,
            ci_upper,
            _safe_str(n_simulations),
        )
        return self._compute_hash("uncertainty", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Aggregation
    # ------------------------------------------------------------------

    def hash_aggregation(
        self,
        aggregation_type: str,
        group_key: str,
        total_co2e: float,
        record_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Hash an aggregation step and add to the provenance chain.

        Records the aggregation of multiple emission calculation results
        into a summary total. Aggregation may be by facility, organization,
        time period, region, or energy type.

        Args:
            aggregation_type: Type of aggregation performed. Common
                values: ``"facility"``, ``"organization"``,
                ``"temporal"``, ``"region"``, ``"energy_type"``.
            group_key: The grouping key value (e.g., facility ID,
                organization name, year-month).
            total_co2e: Aggregated total CO2e emissions.
            record_count: Number of records aggregated.
            metadata: Optional additional context dictionary.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``aggregation_type`` or ``group_key`` is empty,
                or ``record_count`` is negative.
        """
        if not aggregation_type:
            raise ValueError("aggregation_type must not be empty")
        if not group_key:
            raise ValueError("group_key must not be empty")
        if record_count < 0:
            raise ValueError(
                f"record_count must be non-negative, got {record_count}"
            )

        data: Dict[str, Any] = {
            "aggregation_type": aggregation_type,
            "group_key": group_key,
            "total_co2e": _safe_str(total_co2e),
            "record_count": record_count,
        }
        if metadata:
            data["extra_metadata"] = metadata

        logger.info(
            "Hashing aggregation: type=%s key=%s total=%.4f count=%d",
            aggregation_type,
            group_key,
            total_co2e,
            record_count,
        )
        return self._compute_hash("aggregation", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Batch
    # ------------------------------------------------------------------

    def hash_batch(
        self,
        batch_id: str,
        item_count: int,
        total_co2e: float,
        status: str,
        processing_time_ms: Optional[float] = None,
    ) -> str:
        """Hash a batch calculation summary.

        Records the outcome of a batch processing operation that
        calculates emissions for multiple facilities, meters, or
        periods in a single run.

        Args:
            batch_id: Unique identifier for the batch job.
            item_count: Number of items processed in the batch.
            total_co2e: Aggregated total CO2e from all batch items.
            status: Batch processing status (``"COMPLETE"``,
                ``"PARTIAL"``, ``"FAILED"``).
            processing_time_ms: Optional total processing time in
                milliseconds.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``batch_id`` is empty, ``item_count`` is
                negative, or ``status`` is not recognized.
        """
        if not batch_id:
            raise ValueError("batch_id must not be empty")
        if item_count < 0:
            raise ValueError(
                f"item_count must be non-negative, got {item_count}"
            )
        valid_statuses = {"COMPLETE", "PARTIAL", "FAILED"}
        if status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got '{status}'"
            )

        data: Dict[str, Any] = {
            "batch_id": batch_id,
            "item_count": item_count,
            "total_co2e": _safe_str(total_co2e),
            "status": status,
        }
        if processing_time_ms is not None:
            data["processing_time_ms"] = _safe_str(processing_time_ms)

        logger.info(
            "Hashing batch: id=%s items=%d total=%.4f status=%s",
            batch_id,
            item_count,
            total_co2e,
            status,
        )
        return self._compute_hash("batch", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Custom stage
    # ------------------------------------------------------------------

    def hash_custom(
        self,
        stage: str,
        data: Dict[str, Any],
    ) -> str:
        """Hash a custom calculation stage and add to the chain.

        Provides an extensibility point for hashing stages not covered
        by the domain-specific methods. This allows downstream agents
        or pipeline steps to contribute to the provenance chain without
        modifying this class.

        Args:
            stage: Custom stage identifier string.
            data: Dictionary of data to hash.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``stage`` is empty.
            TypeError: If ``data`` is not a dictionary.
        """
        if not stage:
            raise ValueError("stage must not be empty")

        logger.info(
            "Hashing custom stage: %s with %d keys",
            stage,
            len(data),
        )
        return self._compute_hash(stage, data)

    # ------------------------------------------------------------------
    # Chain query and inspection
    # ------------------------------------------------------------------

    def get_chain_hash(self) -> str:
        """Compute the cumulative chain hash representing the entire trail.

        Returns the hash_value of the most recent entry in the chain.
        Since each entry's hash transitively incorporates all preceding
        entries via the chain linking protocol, this single hash
        uniquely identifies the complete state of the provenance chain.

        If the chain is empty, returns an empty string.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters), or empty
            string if the chain has no entries.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> assert prov.get_chain_hash() == ""
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> assert len(prov.get_chain_hash()) == 64
        """
        with self._lock:
            if not self._chain:
                return ""
            return self._chain[-1].hash_value

    def get_chain(self) -> List[ProvenanceEntry]:
        """Return the full provenance chain as a list of entries.

        Returns a shallow copy of the internal chain list to prevent
        external modification of the chain state. The entries themselves
        are frozen dataclasses and cannot be modified.

        Returns:
            List of ProvenanceEntry objects in insertion order (oldest
            first). Returns an empty list if no entries have been
            recorded.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> assert prov.get_chain() == []
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> chain = prov.get_chain()
            >>> assert len(chain) == 1
            >>> assert chain[0].stage == "input"
        """
        with self._lock:
            return list(self._chain)

    def get_entries_by_stage(self, stage: str) -> List[ProvenanceEntry]:
        """Return all entries matching a specific stage.

        Filters the chain to return only entries with the given stage
        label. Useful for extracting all grid factor lookups or all
        compliance checks from a multi-step calculation.

        Args:
            stage: Stage label to filter by (e.g., ``"grid_factor_lookup"``,
                ``"electricity_calc"``, ``"compliance_check"``).

        Returns:
            List of matching ProvenanceEntry objects in insertion order.
        """
        with self._lock:
            return [e for e in self._chain if e.stage == stage]

    def get_entry_by_index(self, index: int) -> Optional[ProvenanceEntry]:
        """Return a specific entry by its position in the chain.

        Args:
            index: Zero-based index into the chain. Negative indices
                are supported (e.g., -1 for the last entry).

        Returns:
            The ProvenanceEntry at the specified index, or None if the
            index is out of range.
        """
        with self._lock:
            try:
                return self._chain[index]
            except IndexError:
                return None

    def get_latest_entry(self) -> Optional[ProvenanceEntry]:
        """Return the most recent entry in the chain.

        Returns:
            The last ProvenanceEntry, or None if the chain is empty.
        """
        with self._lock:
            if not self._chain:
                return None
            return self._chain[-1]

    def get_stage_summary(self) -> Dict[str, int]:
        """Return a summary of entry counts grouped by stage.

        Returns:
            Dictionary mapping stage names to their occurrence count
            in the chain.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"f": "1"})
            '...'
            >>> prov.hash_output({"r": "1"})
            '...'
            >>> prov.get_stage_summary()
            {'input': 1, 'output': 1}
        """
        with self._lock:
            summary: Dict[str, int] = {}
            for entry in self._chain:
                summary[entry.stage] = summary.get(entry.stage, 0) + 1
            return summary

    # ------------------------------------------------------------------
    # Chain verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> bool:
        """Verify the integrity of the provenance chain.

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

        Returns:
            True if the chain is intact, False if any entry fails
            verification or the chain links are broken.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> assert prov.verify_chain() is True  # empty chain
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> assert prov.verify_chain() is True
        """
        with self._lock:
            chain = list(self._chain)

        if not chain:
            logger.debug("verify_chain: empty chain - trivially valid")
            return True

        for i, entry in enumerate(chain):
            # Verify chain linkage
            if i == 0:
                if entry.previous_hash != "":
                    logger.warning(
                        "verify_chain: entry[0] previous_hash is not "
                        "empty string (got '%s')",
                        entry.previous_hash[:16],
                    )
                    return False
            else:
                if entry.previous_hash != chain[i - 1].hash_value:
                    logger.warning(
                        "verify_chain: chain break between entry[%d] "
                        "and entry[%d]",
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
                    "verify_chain: entry[%d] hash mismatch "
                    "(expected=%s, got=%s)",
                    i,
                    expected_hash[:16],
                    entry.hash_value[:16],
                )
                return False

        logger.debug(
            "verify_chain: %d entries verified successfully", len(chain)
        )
        return True

    def verify_chain_detailed(self) -> Tuple[bool, Optional[str], int]:
        """Verify chain integrity with detailed error information.

        Similar to ``verify_chain`` but returns a tuple with the
        validation result, an optional error message, and the index
        of the first failed entry (or -1 if all entries pass).

        Returns:
            Tuple of ``(is_valid, error_message, failed_index)``.
            When the chain is intact, returns ``(True, None, -1)``.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> valid, msg, idx = prov.verify_chain_detailed()
            >>> assert valid is True
            >>> assert msg is None
            >>> assert idx == -1
        """
        with self._lock:
            chain = list(self._chain)

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
    # Serialization / Deserialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the provenance chain to a dictionary.

        Returns a complete representation of the provenance state
        including the chain entries, metadata about the agent, and
        a summary of chain statistics.

        Returns:
            Dictionary with keys:
            - ``agent_id``: Agent identifier string.
            - ``agent_name``: Human-readable agent name.
            - ``prefix``: Provenance prefix string.
            - ``chain_length``: Number of entries in the chain.
            - ``chain_hash``: Current cumulative chain hash.
            - ``max_entries``: Maximum chain length setting.
            - ``entries``: List of entry dictionaries.
            - ``stage_summary``: Dict of stage name to count.
            - ``created_at``: ISO timestamp of serialization.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> d = prov.to_dict()
            >>> assert d["chain_length"] == 1
            >>> assert d["agent_id"] == "AGENT-MRV-009"
        """
        with self._lock:
            entries = [entry.to_dict() for entry in self._chain]
            chain_hash = self._chain[-1].hash_value if self._chain else ""
            stage_summary = self.get_stage_summary()

        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "prefix": self.PREFIX,
            "chain_length": len(entries),
            "chain_hash": chain_hash,
            "max_entries": self._max_entries,
            "entries": entries,
            "stage_summary": stage_summary,
            "created_at": _utcnow().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scope2LocationProvenance:
        """Deserialize a dictionary into a Scope2LocationProvenance instance.

        Reconstructs the provenance chain from a dictionary produced by
        ``to_dict()``. The chain integrity is verified after
        deserialization; if verification fails, a warning is logged but
        the instance is still returned (to allow inspection of corrupt
        chains).

        Args:
            data: Dictionary containing at least ``entries`` and
                optionally ``max_entries``. The ``entries`` value must
                be a list of dictionaries compatible with
                ``ProvenanceEntry.from_dict()``.

        Returns:
            A new Scope2LocationProvenance instance with the
            deserialized chain.

        Raises:
            KeyError: If ``entries`` key is missing from the dictionary.
            TypeError: If ``data`` is not a dictionary.
            ValueError: If any entry dictionary is malformed.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> serialized = prov.to_dict()
            >>> restored = Scope2LocationProvenance.from_dict(serialized)
            >>> assert len(restored.get_chain()) == 1
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict, got {type(data).__name__}"
            )

        max_entries = data.get("max_entries", cls.DEFAULT_MAX_ENTRIES)
        instance = cls(max_entries=max_entries)

        entries_data = data.get("entries", [])
        if not isinstance(entries_data, list):
            raise TypeError(
                f"Expected list for 'entries', got "
                f"{type(entries_data).__name__}"
            )

        for entry_dict in entries_data:
            entry = ProvenanceEntry.from_dict(entry_dict)
            instance._chain.append(entry)

        # Verify chain integrity after deserialization
        if instance._chain and not instance.verify_chain():
            logger.warning(
                "from_dict: deserialized chain failed integrity "
                "verification (length=%d)",
                len(instance._chain),
            )

        logger.info(
            "Deserialized %s provenance chain with %d entries",
            cls.AGENT_ID,
            len(instance._chain),
        )
        return instance

    def to_json(self, indent: int = 2) -> str:
        """Serialize the provenance chain to a JSON string.

        Convenience method that calls ``to_dict()`` and serializes
        the result to a formatted JSON string.

        Args:
            indent: JSON indentation level (default 2).

        Returns:
            Formatted JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> Scope2LocationProvenance:
        """Deserialize a JSON string into a Scope2LocationProvenance.

        Args:
            json_str: JSON string produced by ``to_json()``.

        Returns:
            A new Scope2LocationProvenance instance.

        Raises:
            json.JSONDecodeError: If the string is not valid JSON.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Chain management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the provenance chain and reset to empty state.

        Removes all entries from the chain. After calling this method,
        the provenance tracker behaves as if newly constructed.
        Primarily intended for testing or starting a new calculation
        run.

        Example:
            >>> prov = Scope2LocationProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> assert len(prov.get_chain()) == 1
            >>> prov.reset()
            >>> assert len(prov.get_chain()) == 0
        """
        with self._lock:
            self._chain.clear()
        logger.info(
            "%s provenance chain reset to empty state", self.AGENT_ID
        )

    # Alias for backward compatibility with sibling MRV agents
    clear = reset
    clear_trail = reset

    def merge_chains(
        self, other: Scope2LocationProvenance,
    ) -> str:
        """Merge another provenance chain into this one.

        Appends a merge record that captures the chain hash of the
        other provenance instance. This is used in batch processing
        scenarios where multiple independent calculations are combined
        into a single reporting chain.

        The merge does not copy individual entries from the other chain.
        Instead, it records the other chain's cumulative hash as a
        reference, creating a compact proof that the other chain's
        state was incorporated at this point.

        Args:
            other: Another Scope2LocationProvenance instance whose
                chain hash will be recorded in this chain.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters) of the
            merge entry.

        Raises:
            TypeError: If ``other`` is not a Scope2LocationProvenance.

        Example:
            >>> prov_a = Scope2LocationProvenance()
            >>> prov_a.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> prov_b = Scope2LocationProvenance()
            >>> prov_b.hash_input({"facility_id": "FAC-002"})
            '...'
            >>> merge_hash = prov_a.merge_chains(prov_b)
            >>> assert len(merge_hash) == 64
        """
        if not isinstance(other, Scope2LocationProvenance):
            raise TypeError(
                f"other must be a Scope2LocationProvenance, "
                f"got {type(other).__name__}"
            )

        other_chain_hash = other.get_chain_hash()
        other_chain_length = len(other.get_chain())
        other_stage_summary = other.get_stage_summary()

        data = {
            "merged_chain_hash": other_chain_hash,
            "merged_chain_length": other_chain_length,
            "merged_stage_summary": other_stage_summary,
            "merge_timestamp": _utcnow().isoformat(),
        }

        logger.info(
            "Merging chain: other_hash_prefix=%s other_length=%d",
            other_chain_hash[:16] if other_chain_hash else "empty",
            other_chain_length,
        )
        return self._compute_hash("merge_chains", data)

    def merge_chains_full(
        self, other: Scope2LocationProvenance,
    ) -> str:
        """Merge another chain by copying all its entries into this chain.

        Unlike ``merge_chains`` which only records the other chain's
        hash, this method copies each entry from the other chain into
        this chain as new entries with the stage ``"merge"`` prefix.
        This provides a complete audit trail when full entry-level
        detail is required.

        Args:
            other: Another Scope2LocationProvenance instance.

        Returns:
            Hex-encoded SHA-256 hash string of the final merge entry.

        Raises:
            TypeError: If ``other`` is not a Scope2LocationProvenance.
        """
        if not isinstance(other, Scope2LocationProvenance):
            raise TypeError(
                f"other must be a Scope2LocationProvenance, "
                f"got {type(other).__name__}"
            )

        other_chain = other.get_chain()
        last_hash = ""

        for entry in other_chain:
            merge_data = {
                "original_stage": entry.stage,
                "original_hash": entry.hash_value,
                "original_timestamp": entry.timestamp,
                "original_metadata": entry.metadata,
            }
            last_hash = self._compute_hash("merge", merge_data)

        if not last_hash:
            # Other chain was empty, record an empty merge
            last_hash = self._compute_hash("merge", {
                "merged_chain_hash": "",
                "merged_chain_length": 0,
            })

        logger.info(
            "Full merge completed: %d entries from other chain",
            len(other_chain),
        )
        return last_hash

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if chain exceeds max_entries.

        Must be called while holding self._lock. Removes entries from
        the beginning of the chain to maintain the size constraint.
        """
        overflow = len(self._chain) - self._max_entries
        if overflow <= 0:
            return

        self._chain = self._chain[overflow:]
        logger.debug(
            "Evicted %d oldest provenance entries (max_entries=%d)",
            overflow,
            self._max_entries,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chain_length(self) -> int:
        """Return the number of entries in the provenance chain.

        Returns:
            Integer count of chain entries.
        """
        with self._lock:
            return len(self._chain)

    @property
    def max_entries(self) -> int:
        """Return the maximum chain length before eviction.

        Returns:
            Maximum number of entries.
        """
        return self._max_entries

    @property
    def is_empty(self) -> bool:
        """Return True if the chain has no entries.

        Returns:
            Boolean indicating whether the chain is empty.
        """
        with self._lock:
            return len(self._chain) == 0

    @property
    def first_hash(self) -> str:
        """Return the hash of the first entry in the chain.

        Returns:
            Hex-encoded SHA-256 hash or empty string if chain is empty.
        """
        with self._lock:
            if not self._chain:
                return ""
            return self._chain[0].hash_value

    @property
    def last_hash(self) -> str:
        """Return the hash of the last entry in the chain.

        Alias for ``get_chain_hash()``.

        Returns:
            Hex-encoded SHA-256 hash or empty string if chain is empty.
        """
        return self.get_chain_hash()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries in the provenance chain.

        Returns:
            Integer count of chain entries.
        """
        return self.chain_length

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing chain length and chain hash prefix.
        """
        chain_hash = self.get_chain_hash()
        hash_preview = chain_hash[:12] if chain_hash else "empty"
        return (
            f"Scope2LocationProvenance("
            f"entries={self.chain_length}, "
            f"chain_hash={hash_preview})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            Descriptive string with agent ID and chain statistics.
        """
        return (
            f"{self.AGENT_ID} Provenance Chain: "
            f"{self.chain_length} entries, "
            f"stages={self.get_stage_summary()}"
        )

    def __bool__(self) -> bool:
        """Return True if the chain has at least one entry.

        Returns:
            Boolean indicating whether the chain is non-empty.
        """
        return not self.is_empty

    def __eq__(self, other: object) -> bool:
        """Check equality based on chain hash.

        Two Scope2LocationProvenance instances are considered equal
        if they have the same chain hash (which implies identical
        chains given the collision resistance of SHA-256).

        Args:
            other: Another object to compare.

        Returns:
            True if both instances have the same chain hash.
        """
        if not isinstance(other, Scope2LocationProvenance):
            return NotImplemented
        return self.get_chain_hash() == other.get_chain_hash()

    def __contains__(self, stage: str) -> bool:
        """Check if a stage exists in the chain.

        Args:
            stage: Stage name to look for.

        Returns:
            True if at least one entry with the given stage exists.
        """
        with self._lock:
            return any(e.stage == stage for e in self._chain)

    def __iter__(self):
        """Iterate over entries in the provenance chain.

        Yields:
            ProvenanceEntry instances in insertion order.
        """
        with self._lock:
            chain_copy = list(self._chain)
        return iter(chain_copy)

    def __getitem__(self, index: int) -> ProvenanceEntry:
        """Get an entry by index.

        Args:
            index: Zero-based index (negative indices supported).

        Returns:
            ProvenanceEntry at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        with self._lock:
            return self._chain[index]


# ---------------------------------------------------------------------------
# Module-level factory function
# ---------------------------------------------------------------------------


def create_provenance(
    max_entries: int = Scope2LocationProvenance.DEFAULT_MAX_ENTRIES,
) -> Scope2LocationProvenance:
    """Create a new Scope2LocationProvenance instance.

    Factory function that provides a clean entry point for creating
    provenance trackers. This is the recommended way to instantiate
    a provenance chain for Scope 2 location-based emission calculations.

    Args:
        max_entries: Maximum number of provenance entries to retain.
            Defaults to ``Scope2LocationProvenance.DEFAULT_MAX_ENTRIES``
            (50000).

    Returns:
        A new empty Scope2LocationProvenance instance.

    Example:
        >>> prov = create_provenance()
        >>> assert prov.is_empty
        >>> prov.hash_input({"facility_id": "FAC-001"})
        '...'
        >>> assert not prov.is_empty
    """
    return Scope2LocationProvenance(max_entries=max_entries)


# ---------------------------------------------------------------------------
# Thread-safe singleton helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_instance: Optional[Scope2LocationProvenance] = None


def get_provenance_tracker() -> Scope2LocationProvenance:
    """Return the process-wide singleton Scope2LocationProvenance.

    Creates the instance on first call (lazy initialization). Subsequent
    calls return the same object. The function is thread-safe via
    double-checked locking.

    Returns:
        The singleton Scope2LocationProvenance instance.

    Example:
        >>> tracker_a = get_provenance_tracker()
        >>> tracker_b = get_provenance_tracker()
        >>> assert tracker_a is tracker_b
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = Scope2LocationProvenance()
                logger.info(
                    "Scope 2 location-based singleton provenance "
                    "tracker created"
                )
    return _singleton_instance


def set_provenance_tracker(
    tracker: Scope2LocationProvenance,
) -> None:
    """Replace the process-wide singleton with a custom tracker.

    Useful in tests that need isolated provenance tracker instances
    or when injecting a pre-configured tracker.

    Args:
        tracker: The Scope2LocationProvenance instance to install.

    Raises:
        TypeError: If ``tracker`` is not a Scope2LocationProvenance.
    """
    if not isinstance(tracker, Scope2LocationProvenance):
        raise TypeError(
            f"tracker must be a Scope2LocationProvenance, "
            f"got {type(tracker).__name__}"
        )
    global _singleton_instance
    with _singleton_lock:
        _singleton_instance = tracker
    logger.info(
        "Scope 2 location-based provenance tracker singleton replaced"
    )


def reset_provenance_tracker() -> None:
    """Destroy the current singleton and reset to None.

    The next call to ``get_provenance_tracker()`` will create a fresh
    instance. Intended for use in test teardown to prevent state
    leakage between test cases.

    Example:
        >>> reset_provenance_tracker()
        >>> tracker = get_provenance_tracker()  # fresh instance
    """
    global _singleton_instance
    with _singleton_lock:
        _singleton_instance = None
    logger.info(
        "Scope 2 location-based provenance tracker singleton reset"
    )


# ---------------------------------------------------------------------------
# Utility functions
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
    ``Scope2LocationProvenance._compute_hash`` but without side effects.
    Useful for external verification of chain entries.

    Args:
        previous_hash: Previous entry's hash (empty string for first).
        stage: Stage identifier string.
        data: Dictionary of metadata.

    Returns:
        Hex-encoded SHA-256 hash string (64 characters).

    Example:
        >>> h = compute_chain_entry_hash("", "input", {"f": "1"})
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
        Tuple of ``(is_valid, error_message)``.

    Example:
        >>> prov = create_provenance()
        >>> prov.hash_input({"f": "1"})
        '...'
        >>> entries = [e.to_dict() for e in prov.get_chain()]
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
                    f"chain break between entry[{i - 1}] "
                    f"and entry[{i}]",
                )

        # Recompute and verify hash
        expected = compute_chain_entry_hash(
            previous_hash, stage, metadata
        )
        if hash_value != expected:
            return (
                False,
                f"entry[{i}] hash mismatch (stage={stage})",
            )

    return True, None


def diff_chains(
    chain_a: Scope2LocationProvenance,
    chain_b: Scope2LocationProvenance,
) -> Dict[str, Any]:
    """Compare two provenance chains and return their differences.

    Useful for debugging when two calculations that should be identical
    produce different results. The diff includes which entries differ,
    the first divergence point, and summary statistics.

    Args:
        chain_a: First provenance chain.
        chain_b: Second provenance chain.

    Returns:
        Dictionary containing:
        - ``identical``: Boolean indicating if chains are identical.
        - ``length_a``: Length of chain A.
        - ``length_b``: Length of chain B.
        - ``chain_hash_a``: Chain hash of A.
        - ``chain_hash_b``: Chain hash of B.
        - ``first_divergence_index``: Index where chains first differ
          (-1 if identical up to the shorter chain's length).
        - ``divergent_stages``: List of (index, stage_a, stage_b) tuples
          for entries where stages differ.
    """
    entries_a = chain_a.get_chain()
    entries_b = chain_b.get_chain()

    first_divergence = -1
    divergent_stages: List[Tuple[int, str, str]] = []

    min_len = min(len(entries_a), len(entries_b))
    for i in range(min_len):
        if entries_a[i].hash_value != entries_b[i].hash_value:
            if first_divergence == -1:
                first_divergence = i
            divergent_stages.append(
                (i, entries_a[i].stage, entries_b[i].stage)
            )

    if len(entries_a) != len(entries_b) and first_divergence == -1:
        first_divergence = min_len

    return {
        "identical": (
            chain_a.get_chain_hash() == chain_b.get_chain_hash()
        ),
        "length_a": len(entries_a),
        "length_b": len(entries_b),
        "chain_hash_a": chain_a.get_chain_hash(),
        "chain_hash_b": chain_b.get_chain_hash(),
        "first_divergence_index": first_divergence,
        "divergent_stages": divergent_stages,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclass
    "ProvenanceEntry",
    # Constants
    "VALID_STAGES",
    # Main class
    "Scope2LocationProvenance",
    # Factory function
    "create_provenance",
    # Singleton helpers
    "get_provenance_tracker",
    "set_provenance_tracker",
    "reset_provenance_tracker",
    # Utility functions
    "compute_standalone_hash",
    "verify_hash",
    "compute_chain_entry_hash",
    "verify_chain_entries",
    "diff_chains",
]
