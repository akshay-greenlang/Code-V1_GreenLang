# -*- coding: utf-8 -*-
"""
Provenance Tracking for Scope 2 Market-Based Emissions Agent - AGENT-MRV-010

Provides SHA-256 based audit trail tracking for all Scope 2 market-based
emissions agent operations. Implements a chain of SHA-256 hashes for each
calculation stage, ensuring complete audit trail and deterministic
reproducibility across contractual instrument lookups, instrument allocation,
quality assessment, covered/uncovered emission calculations, residual mix
factors, supplier-specific factors, per-gas GHG breakdowns, GWP conversions,
certificate retirements, dual reporting comparisons, compliance checks,
and uncertainty quantification.

Zero-Hallucination Guarantees:
    - All hashes are deterministic SHA-256 via hashlib
    - Chain hashing links operations in sequence (append-only)
    - Each entry records previous_hash for tamper detection
    - JSON canonical form with sort_keys=True for reproducibility
    - Decimal values converted to string for hashing consistency
    - Complete provenance for every Scope 2 market-based operation

Provenance Stages:
    - input: Raw input data hashing at intake
    - instrument_lookup: Contractual instrument retrieval from database
    - instrument_allocation: Allocation of instrument MWh to consumption
    - quality_assessment: Quality criteria scoring for instruments
    - covered_calculation: Emission calc for instrument-covered consumption
    - uncovered_calculation: Emission calc for residual/uncovered consumption
    - residual_mix_lookup: Residual mix factor retrieval
    - supplier_factor: Supplier-specific emission factor hashing
    - gas_breakdown: Per-gas (CO2, CH4, N2O) breakdown with GWP source
    - gwp_conversion: Individual GHG to CO2e conversion step
    - certificate_retirement: Certificate/instrument retirement record
    - dual_reporting: Location vs market comparison record
    - compliance_check: Regulatory framework compliance check result
    - uncertainty: Monte Carlo or analytical uncertainty result
    - output: Final output data hashing at completion
    - aggregation: Aggregation step for batch calculations
    - batch: Batch processing summary
    - merge: Chain merge operation for batch processing
    - merge_chains: Chain merge reference operation

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
    - GHG Protocol Scope 2 Guidance (2015) - Market-Based Method
    - ISO 14064-1:2018
    - CSRD / ESRS E1
    - UK SECR / Streamlined Energy and Carbon Reporting
    - EPA Green Power Partnership
    - EU ETS (Monitoring and Reporting Regulation)
    - CDP Climate Change Questionnaire
    - RE100 Reporting Criteria

Supported Contractual Instruments:
    - Energy Attribute Certificates (EACs): GO, REGO, I-REC, TIGR
    - Renewable Energy Certificates (RECs): US Green-e
    - Power Purchase Agreements (PPAs): Physical, Virtual/Financial
    - Supplier-Specific Emission Factors
    - Residual Mix Factors (e.g., AIB European Residual Mix)
    - Green Tariff / Utility Programs
    - Direct Line / On-site Generation

Example:
    >>> from greenlang.agents.mrv.scope2_market.provenance import (
    ...     Scope2MarketProvenance, create_provenance,
    ... )
    >>> prov = create_provenance()
    >>> h1 = prov.hash_input({"facility_id": "FAC-001", "year": 2025})
    >>> h2 = prov.hash_instrument_lookup(
    ...     instrument_id="REC-2025-001",
    ...     instrument_type="REC",
    ...     quantity_mwh=3000.0,
    ...     emission_factor=0.0,
    ... )
    >>> h3 = prov.hash_instrument_allocation(
    ...     purchase_id="PUR-2025-001",
    ...     instrument_id="REC-2025-001",
    ...     mwh_allocated=3000.0,
    ... )
    >>> h4 = prov.hash_covered_calculation(
    ...     instrument_id="REC-2025-001",
    ...     mwh=3000.0,
    ...     emission_factor=0.0,
    ...     emissions_kg=0.0,
    ... )
    >>> chain = prov.get_chain()
    >>> assert len(chain) == 4
    >>> assert prov.verify_chain() is True
    >>> chain_hash = prov.get_chain_hash()
    >>> assert len(chain_hash) == 64

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
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
    # Instrument operations
    "instrument_lookup",
    "instrument_lookup_validate",
    "instrument_lookup_batch",
    "instrument_allocation",
    "instrument_allocation_validate",
    "instrument_allocation_partial",
    # Quality assessment stages
    "quality_assessment",
    "quality_assessment_criteria",
    "quality_assessment_temporal",
    "quality_assessment_geographic",
    "quality_assessment_technology",
    # Covered calculation stages
    "covered_calculation",
    "covered_calculation_validate",
    "covered_calculation_batch",
    "covered_calculation_ppa",
    "covered_calculation_rec",
    "covered_calculation_go",
    "covered_calculation_green_tariff",
    # Uncovered / residual calculation stages
    "uncovered_calculation",
    "uncovered_calculation_validate",
    "uncovered_calculation_batch",
    # Residual mix stages
    "residual_mix_lookup",
    "residual_mix_lookup_validate",
    "residual_mix_country",
    "residual_mix_regional",
    # Supplier factor stages
    "supplier_factor",
    "supplier_factor_validate",
    "supplier_factor_update",
    # Per-gas breakdown stages
    "gas_breakdown",
    "gas_breakdown_validate",
    # GWP conversion stages
    "gwp_conversion",
    "gwp_conversion_co2",
    "gwp_conversion_ch4",
    "gwp_conversion_n2o",
    # Certificate retirement stages
    "certificate_retirement",
    "certificate_retirement_validate",
    "certificate_retirement_batch",
    # Dual reporting stages
    "dual_reporting",
    "dual_reporting_validate",
    "dual_reporting_reconciliation",
    # Compliance stages
    "compliance_check",
    "compliance_check_ghg_protocol",
    "compliance_check_iso14064",
    "compliance_check_csrd",
    "compliance_check_uk_secr",
    "compliance_check_epa_gpp",
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
    "aggregation_instrument_type",
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
    market-based emission calculation stage.

    Each entry in the provenance chain captures the stage name, its SHA-256
    hash (incorporating the previous entry's hash for chain integrity), the
    ISO-formatted UTC timestamp, the link to the previous hash, and a
    metadata dictionary containing the raw data used to compute this hash.

    The ``frozen=True`` decorator ensures that once created, an entry
    cannot be modified in-place, enforcing the append-only property of
    the provenance chain.

    Attributes:
        stage: Identifies the calculation stage that produced this entry.
            Examples: ``"input"``, ``"instrument_lookup"``,
            ``"instrument_allocation"``, ``"quality_assessment"``,
            ``"covered_calculation"``, ``"uncovered_calculation"``,
            ``"residual_mix_lookup"``, ``"supplier_factor"``,
            ``"gas_breakdown"``, ``"gwp_conversion"``,
            ``"certificate_retirement"``, ``"dual_reporting"``,
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
        ...     stage="covered_calculation",
        ...     hash_value="a1b2c3...",
        ...     timestamp="2025-06-15T10:30:00+00:00",
        ...     previous_hash="d4e5f6...",
        ...     metadata={"mwh": 3000.0, "emissions_kg": 0.0},
        ... )
        >>> entry.stage
        'covered_calculation'
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
# Scope2MarketProvenance
# ---------------------------------------------------------------------------

class Scope2MarketProvenance:
    """SHA-256 provenance chain for Scope 2 market-based emission calculations.

    Implements a chain of SHA-256 hashes for each calculation stage, ensuring
    complete audit trail and deterministic reproducibility. The chain is
    append-only during calculation: each new entry's hash incorporates the
    previous entry's hash, creating a tamper-evident linked list of
    cryptographic digests.

    This class provides domain-specific hashing methods for every stage in
    the Scope 2 market-based emissions calculation pipeline:

    1. **Input Hashing** (``hash_input``): Captures the raw input data
       (facility ID, reporting period, consumption data, instrument
       portfolio) at the start of the calculation.

    2. **Instrument Lookup Hashing** (``hash_instrument_lookup``): Records
       the contractual instrument retrieval including instrument ID, type,
       quantity, and associated emission factor.

    3. **Instrument Allocation Hashing** (``hash_instrument_allocation``):
       Records the allocation of instrument MWh to facility consumption,
       tracking purchase-to-consumption linkage.

    4. **Quality Assessment Hashing** (``hash_quality_assessment``):
       Records quality criteria scoring for contractual instruments
       per GHG Protocol Scope 2 Quality Criteria.

    5. **Covered Calculation Hashing** (``hash_covered_calculation``):
       Records the emission calculation for consumption covered by
       contractual instruments (RECs, GOs, PPAs, etc.).

    6. **Uncovered Calculation Hashing** (``hash_uncovered_calculation``):
       Records the emission calculation for consumption not covered by
       any contractual instrument, using residual mix factors.

    7. **Residual Mix Lookup Hashing** (``hash_residual_mix_lookup``):
       Records the retrieval of residual mix emission factors for
       uncovered consumption.

    8. **Supplier Factor Hashing** (``hash_supplier_factor``):
       Records supplier-specific emission factors used for market-based
       calculations.

    9. **Gas Breakdown Hashing** (``hash_gas_breakdown``): Records the
       per-gas decomposition (CO2, CH4, N2O) with the GWP source
       reference and total CO2e.

    10. **GWP Conversion Hashing** (``hash_gwp_conversion``): Records
        individual greenhouse gas to CO2e conversion steps.

    11. **Certificate Retirement Hashing** (``hash_certificate_retirement``):
        Records the retirement or cancellation of energy attribute
        certificates for market-based claims.

    12. **Dual Reporting Hashing** (``hash_dual_reporting``): Records the
        comparison between location-based and market-based results as
        required by GHG Protocol Scope 2 Guidance.

    13. **Compliance Check Hashing** (``hash_compliance_check``): Records
        the result of a regulatory framework compliance check.

    14. **Uncertainty Hashing** (``hash_uncertainty``): Records uncertainty
        quantification results.

    15. **Output Hashing** (``hash_output``): Captures the final output
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
        >>> prov = Scope2MarketProvenance()
        >>> h1 = prov.hash_input({"facility_id": "FAC-001"})
        >>> h2 = prov.hash_instrument_lookup("REC-001", "REC", 3000.0, 0.0)
        >>> h3 = prov.hash_output({"total_co2e": 500.0})
        >>> assert prov.verify_chain() is True
        >>> assert len(prov.get_chain()) == 3
    """

    # ------------------------------------------------------------------
    # Class constants
    # ------------------------------------------------------------------

    #: Default maximum number of entries before eviction triggers.
    DEFAULT_MAX_ENTRIES: int = 50000

    #: Prefix used for all Scope 2 market-based provenance identifiers.
    PREFIX: str = "gl_s2m"

    #: Agent identifier for this provenance tracker.
    AGENT_ID: str = "AGENT-MRV-010"

    #: Human-readable agent name.
    AGENT_NAME: str = "Scope 2 Market-Based Emissions"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        """Initialize an empty provenance chain.

        Creates a new Scope2MarketProvenance instance with an empty
        chain of ProvenanceEntry records. The chain will be populated
        as calculation stages call the domain-specific hash methods.

        Args:
            max_entries: Maximum number of provenance entries to retain
                in memory. When exceeded, the oldest entries are evicted
                to prevent unbounded memory growth. Defaults to 50000.

        Example:
            >>> prov = Scope2MarketProvenance()
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

        Args:
            stage: String label identifying the calculation stage.
            data: Dictionary of values to include in the hash.

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
                timestamp=utcnow().isoformat(),
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

        Records the raw input data at the start of a Scope 2 market-based
        emission calculation. This is typically the first entry in the chain
        and captures facility identifiers, reporting period, consumption
        data, instrument portfolio, and configuration parameters.

        Args:
            data: Dictionary of input data. Common keys include:
                - ``facility_id``: Facility identifier string.
                - ``reporting_year``: Integer reporting year.
                - ``reporting_period``: Period string (e.g., "2025-Q1").
                - ``electricity_mwh``: Total electricity consumption in MWh.
                - ``instruments``: List of contractual instrument IDs.
                - ``region_id``: Grid region identifier.
                - ``country_code``: ISO 3166-1 alpha-2 country code.
                - ``tenant_id``: Multi-tenant identifier.
                - ``calculation_method``: Must be "market".

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            TypeError: If ``data`` is not a dictionary.
            ValueError: If ``data`` is empty.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> h = prov.hash_input({
            ...     "facility_id": "FAC-001",
            ...     "reporting_year": 2025,
            ...     "electricity_mwh": 10000.0,
            ...     "calculation_method": "market",
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
        market-based emission calculation pipeline. This is typically
        the last entry in the chain and captures the aggregated totals
        for covered and uncovered consumption, per-gas breakdowns, dual
        reporting comparison, and quality metrics.

        Args:
            result: Dictionary of output data. Common keys include:
                - ``total_co2e_kg``: Total CO2e emissions in kilograms.
                - ``total_co2e_tonnes``: Total CO2e in metric tonnes.
                - ``covered_co2e_kg``: CO2e from instrument-covered MWh.
                - ``uncovered_co2e_kg``: CO2e from residual/uncovered MWh.
                - ``covered_mwh``: Total MWh covered by instruments.
                - ``uncovered_mwh``: Total MWh not covered.
                - ``instrument_count``: Number of instruments applied.
                - ``location_based_co2e_kg``: Comparison location-based.
                - ``data_quality_score``: Quality score (0.0-1.0).
                - ``validation_status``: PASS or FAIL.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            TypeError: If ``result`` is not a dictionary.
            ValueError: If ``result`` is empty.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_output({
            ...     "total_co2e_tonnes": 500.0,
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
    # Domain-specific hashing methods: Instrument Lookup
    # ------------------------------------------------------------------

    def hash_instrument_lookup(
        self,
        instrument_id: str,
        instrument_type: str,
        quantity_mwh: float,
        emission_factor: float,
    ) -> str:
        """Hash a contractual instrument lookup and add to the chain.

        Records the retrieval of a contractual instrument from the
        instrument registry. Contractual instruments are the foundation
        of the market-based method and include RECs, GOs, I-RECs, PPAs,
        supplier-specific factors, and green tariffs.

        Supported instrument types:
        - ``"REC"``: Renewable Energy Certificate (US Green-e)
        - ``"GO"``: Guarantee of Origin (EU)
        - ``"I-REC"``: International REC
        - ``"TIGR"``: Tradable Instrument for Global Renewables
        - ``"PPA_PHYSICAL"``: Physical Power Purchase Agreement
        - ``"PPA_VIRTUAL"``: Virtual/Financial PPA (VPPA)
        - ``"SUPPLIER_SPECIFIC"``: Supplier-specific emission factor
        - ``"GREEN_TARIFF"``: Green tariff / utility program
        - ``"DIRECT_LINE"``: Direct line / on-site generation
        - ``"RESIDUAL_MIX"``: Residual mix factor (fallback)

        Args:
            instrument_id: Unique identifier for the instrument (e.g.,
                ``"REC-2025-001"``, ``"GO-DE-2025-00042"``).
            instrument_type: Type classification of the instrument.
                Must not be empty.
            quantity_mwh: Quantity of electricity represented by the
                instrument in megawatt-hours. Must be non-negative.
            emission_factor: Emission factor associated with this
                instrument in kg CO2e per MWh. For zero-carbon
                instruments (RECs, GOs), this is typically 0.0.
                Must be non-negative.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``instrument_id`` or ``instrument_type``
                is empty, or numeric values are negative.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_instrument_lookup(
            ...     instrument_id="REC-2025-001",
            ...     instrument_type="REC",
            ...     quantity_mwh=3000.0,
            ...     emission_factor=0.0,
            ... )
            >>> assert len(h) == 64
        """
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if not instrument_type:
            raise ValueError("instrument_type must not be empty")
        if quantity_mwh < 0:
            raise ValueError(
                f"quantity_mwh must be non-negative, got {quantity_mwh}"
            )
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, got {emission_factor}"
            )

        data = {
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "quantity_mwh": _safe_str(quantity_mwh),
            "emission_factor": _safe_str(emission_factor),
        }

        logger.info(
            "Hashing instrument lookup: id=%s type=%s qty=%.2f MWh ef=%s",
            instrument_id,
            instrument_type,
            quantity_mwh,
            _safe_str(emission_factor),
        )
        return self._compute_hash("instrument_lookup", data)

    def hash_instrument_lookup_extended(
        self,
        instrument_id: str,
        instrument_type: str,
        quantity_mwh: float,
        emission_factor: float,
        issuing_body: Optional[str] = None,
        generation_source: Optional[str] = None,
        generation_country: Optional[str] = None,
        generation_period: Optional[str] = None,
        vintage_year: Optional[int] = None,
        tracking_system: Optional[str] = None,
        certificate_status: Optional[str] = None,
    ) -> str:
        """Hash an extended instrument lookup with full metadata.

        Similar to ``hash_instrument_lookup`` but includes additional
        metadata about the instrument's provenance, generation source,
        and tracking system registration.

        Args:
            instrument_id: Unique identifier for the instrument.
            instrument_type: Type classification of the instrument.
            quantity_mwh: Quantity in MWh.
            emission_factor: Emission factor (kg CO2e/MWh).
            issuing_body: Optional issuing body (e.g., ``"Green-e"``,
                ``"AIB"``, ``"I-REC Standard"``).
            generation_source: Optional energy source (e.g., ``"wind"``,
                ``"solar"``, ``"hydro"``).
            generation_country: Optional ISO country code of generation.
            generation_period: Optional generation period string.
            vintage_year: Optional vintage year of the certificate.
            tracking_system: Optional tracking system (e.g., ``"M-RETS"``,
                ``"WREGIS"``, ``"PJM-GATS"``).
            certificate_status: Optional status (e.g., ``"active"``,
                ``"retired"``, ``"expired"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if not instrument_type:
            raise ValueError("instrument_type must not be empty")
        if quantity_mwh < 0:
            raise ValueError(
                f"quantity_mwh must be non-negative, got {quantity_mwh}"
            )
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, got {emission_factor}"
            )

        data: Dict[str, Any] = {
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "quantity_mwh": _safe_str(quantity_mwh),
            "emission_factor": _safe_str(emission_factor),
        }
        if issuing_body is not None:
            data["issuing_body"] = issuing_body
        if generation_source is not None:
            data["generation_source"] = generation_source
        if generation_country is not None:
            data["generation_country"] = generation_country
        if generation_period is not None:
            data["generation_period"] = generation_period
        if vintage_year is not None:
            data["vintage_year"] = vintage_year
        if tracking_system is not None:
            data["tracking_system"] = tracking_system
        if certificate_status is not None:
            data["certificate_status"] = certificate_status

        logger.info(
            "Hashing extended instrument lookup: id=%s type=%s "
            "qty=%.2f MWh source=%s country=%s",
            instrument_id,
            instrument_type,
            quantity_mwh,
            generation_source or "N/A",
            generation_country or "N/A",
        )
        return self._compute_hash("instrument_lookup", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Instrument Allocation
    # ------------------------------------------------------------------

    def hash_instrument_allocation(
        self,
        purchase_id: str,
        instrument_id: str,
        mwh_allocated: float,
    ) -> str:
        """Hash an instrument allocation step and add to the chain.

        Records the allocation of a contractual instrument's MWh to a
        facility's electricity consumption. A single instrument may be
        partially allocated across multiple facilities, or a facility
        may have multiple instruments allocated to it.

        The allocation ensures that:
        - No double-counting occurs across facilities
        - Total allocated MWh does not exceed instrument quantity
        - Allocation is within the same market boundary

        Args:
            purchase_id: Unique identifier for the purchase/procurement
                record (e.g., ``"PUR-2025-001"``).
            instrument_id: Identifier of the instrument being allocated
                (must match a previously looked-up instrument).
            mwh_allocated: Number of MWh from this instrument allocated
                to the facility. Must be non-negative.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``purchase_id`` or ``instrument_id`` is empty,
                or ``mwh_allocated`` is negative.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_instrument_allocation(
            ...     purchase_id="PUR-2025-001",
            ...     instrument_id="REC-2025-001",
            ...     mwh_allocated=3000.0,
            ... )
            >>> assert len(h) == 64
        """
        if not purchase_id:
            raise ValueError("purchase_id must not be empty")
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if mwh_allocated < 0:
            raise ValueError(
                f"mwh_allocated must be non-negative, got {mwh_allocated}"
            )

        data = {
            "purchase_id": purchase_id,
            "instrument_id": instrument_id,
            "mwh_allocated": _safe_str(mwh_allocated),
        }

        logger.info(
            "Hashing instrument allocation: purchase=%s instrument=%s "
            "allocated=%.2f MWh",
            purchase_id,
            instrument_id,
            mwh_allocated,
        )
        return self._compute_hash("instrument_allocation", data)

    def hash_instrument_allocation_extended(
        self,
        purchase_id: str,
        instrument_id: str,
        mwh_allocated: float,
        facility_id: Optional[str] = None,
        allocation_pct: Optional[float] = None,
        remaining_mwh: Optional[float] = None,
        allocation_method: Optional[str] = None,
        market_boundary: Optional[str] = None,
        reporting_period: Optional[str] = None,
    ) -> str:
        """Hash an extended instrument allocation with full context.

        Similar to ``hash_instrument_allocation`` but includes facility
        context, allocation percentage, remaining balance, and market
        boundary information.

        Args:
            purchase_id: Purchase record identifier.
            instrument_id: Instrument identifier.
            mwh_allocated: MWh allocated from this instrument.
            facility_id: Optional target facility identifier.
            allocation_pct: Optional percentage of instrument allocated
                (0.0-100.0).
            remaining_mwh: Optional remaining unallocated MWh on the
                instrument after this allocation.
            allocation_method: Optional method (e.g., ``"pro_rata"``,
                ``"first_come"``, ``"manual"``).
            market_boundary: Optional market boundary identifier (e.g.,
                ``"US"``, ``"EU"``, ``"ERCOT"``).
            reporting_period: Optional reporting period string.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not purchase_id:
            raise ValueError("purchase_id must not be empty")
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if mwh_allocated < 0:
            raise ValueError(
                f"mwh_allocated must be non-negative, got {mwh_allocated}"
            )

        data: Dict[str, Any] = {
            "purchase_id": purchase_id,
            "instrument_id": instrument_id,
            "mwh_allocated": _safe_str(mwh_allocated),
        }
        if facility_id is not None:
            data["facility_id"] = facility_id
        if allocation_pct is not None:
            data["allocation_pct"] = _safe_str(allocation_pct)
        if remaining_mwh is not None:
            data["remaining_mwh"] = _safe_str(remaining_mwh)
        if allocation_method is not None:
            data["allocation_method"] = allocation_method
        if market_boundary is not None:
            data["market_boundary"] = market_boundary
        if reporting_period is not None:
            data["reporting_period"] = reporting_period

        logger.info(
            "Hashing extended allocation: purchase=%s instrument=%s "
            "allocated=%.2f MWh facility=%s method=%s",
            purchase_id,
            instrument_id,
            mwh_allocated,
            facility_id or "N/A",
            allocation_method or "N/A",
        )
        return self._compute_hash("instrument_allocation", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Quality Assessment
    # ------------------------------------------------------------------

    def hash_quality_assessment(
        self,
        instrument_id: str,
        criteria_scores: Dict[str, Any],
        overall_score: float,
    ) -> str:
        """Hash a quality assessment for a contractual instrument.

        Records the quality criteria evaluation for a contractual
        instrument per the GHG Protocol Scope 2 Quality Criteria.
        The quality criteria assess the reliability and appropriateness
        of the instrument for market-based accounting.

        GHG Protocol Scope 2 Quality Criteria include:
        - Contractual instrument conveys emission factor attributes
        - Unique claim (no double-counting)
        - Sourced from the same market boundary
        - Applied within a defined time period (vintage)
        - Instrument is retired/cancelled on behalf of the user
        - Generated from the specified energy source
        - Verified by an independent third party

        Args:
            instrument_id: Identifier of the instrument being assessed.
            criteria_scores: Dictionary mapping quality criteria names
                to their scores. Keys may include ``"conveys_attributes"``,
                ``"unique_claim"``, ``"market_boundary"``,
                ``"time_period"``, ``"retirement"``, ``"energy_source"``,
                ``"verification"``. Values are typically floats in
                [0.0, 1.0] or booleans.
            overall_score: Aggregate quality score (0.0-1.0) computed
                from the individual criteria scores. Must be in [0.0, 1.0].

from greenlang.schemas import utcnow

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``instrument_id`` is empty, ``criteria_scores``
                is empty, or ``overall_score`` is outside [0.0, 1.0].

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_quality_assessment(
            ...     instrument_id="REC-2025-001",
            ...     criteria_scores={
            ...         "conveys_attributes": True,
            ...         "unique_claim": True,
            ...         "market_boundary": True,
            ...     },
            ...     overall_score=0.95,
            ... )
            >>> assert len(h) == 64
        """
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if not criteria_scores:
            raise ValueError("criteria_scores must not be empty")
        if overall_score < 0.0 or overall_score > 1.0:
            raise ValueError(
                f"overall_score must be in [0.0, 1.0], got {overall_score}"
            )

        data = {
            "instrument_id": instrument_id,
            "criteria_scores": {
                k: _safe_str(v) for k, v in criteria_scores.items()
            },
            "overall_score": _safe_str(overall_score),
        }

        logger.info(
            "Hashing quality assessment: instrument=%s "
            "score=%.4f criteria_count=%d",
            instrument_id,
            overall_score,
            len(criteria_scores),
        )
        return self._compute_hash("quality_assessment", data)

    def hash_quality_assessment_extended(
        self,
        instrument_id: str,
        criteria_scores: Dict[str, Any],
        overall_score: float,
        instrument_type: Optional[str] = None,
        assessor: Optional[str] = None,
        assessment_date: Optional[str] = None,
        notes: Optional[str] = None,
        meets_minimum: Optional[bool] = None,
        ghg_protocol_compliant: Optional[bool] = None,
    ) -> str:
        """Hash an extended quality assessment with assessor metadata.

        Similar to ``hash_quality_assessment`` but includes additional
        fields for the assessor identity, assessment date, notes, and
        compliance determination.

        Args:
            instrument_id: Instrument identifier.
            criteria_scores: Quality criteria scores dictionary.
            overall_score: Aggregate quality score (0.0-1.0).
            instrument_type: Optional instrument type for context.
            assessor: Optional assessor identity or system name.
            assessment_date: Optional ISO date of assessment.
            notes: Optional free-text notes.
            meets_minimum: Optional flag indicating whether the
                instrument meets minimum quality requirements.
            ghg_protocol_compliant: Optional flag indicating GHG
                Protocol Scope 2 Quality Criteria compliance.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if not criteria_scores:
            raise ValueError("criteria_scores must not be empty")
        if overall_score < 0.0 or overall_score > 1.0:
            raise ValueError(
                f"overall_score must be in [0.0, 1.0], got {overall_score}"
            )

        data: Dict[str, Any] = {
            "instrument_id": instrument_id,
            "criteria_scores": {
                k: _safe_str(v) for k, v in criteria_scores.items()
            },
            "overall_score": _safe_str(overall_score),
        }
        if instrument_type is not None:
            data["instrument_type"] = instrument_type
        if assessor is not None:
            data["assessor"] = assessor
        if assessment_date is not None:
            data["assessment_date"] = assessment_date
        if notes is not None:
            data["notes"] = notes
        if meets_minimum is not None:
            data["meets_minimum"] = meets_minimum
        if ghg_protocol_compliant is not None:
            data["ghg_protocol_compliant"] = ghg_protocol_compliant

        logger.info(
            "Hashing extended quality assessment: instrument=%s "
            "score=%.4f compliant=%s",
            instrument_id,
            overall_score,
            _safe_str(ghg_protocol_compliant),
        )
        return self._compute_hash("quality_assessment", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Covered Calculation
    # ------------------------------------------------------------------

    def hash_covered_calculation(
        self,
        instrument_id: str,
        mwh: float,
        emission_factor: float,
        emissions_kg: float,
    ) -> str:
        """Hash an emission calculation for instrument-covered consumption.

        Records the emission calculation for electricity consumption
        that is covered by a contractual instrument. The formula is:

            emissions_kg = mwh * emission_factor

        For zero-carbon instruments (RECs, GOs for renewable energy),
        the emission_factor is 0.0 and emissions_kg is 0.0. For
        supplier-specific factors or PPAs from mixed sources, the
        emission factor reflects the generation mix.

        Args:
            instrument_id: Identifier of the covering instrument.
            mwh: Electricity consumption covered by this instrument
                in megawatt-hours. Must be non-negative.
            emission_factor: Emission factor for this instrument
                in kg CO2e per MWh. Must be non-negative.
            emissions_kg: Calculated emissions in kg CO2e. This is
                the output of the deterministic formula.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``instrument_id`` is empty, or numeric
                values are negative.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_covered_calculation(
            ...     instrument_id="REC-2025-001",
            ...     mwh=3000.0,
            ...     emission_factor=0.0,
            ...     emissions_kg=0.0,
            ... )
            >>> assert len(h) == 64
        """
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if mwh < 0:
            raise ValueError(
                f"mwh must be non-negative, got {mwh}"
            )
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, got {emission_factor}"
            )

        data = {
            "instrument_id": instrument_id,
            "mwh": _safe_str(mwh),
            "emission_factor": _safe_str(emission_factor),
            "emissions_kg": _safe_str(emissions_kg),
        }

        logger.info(
            "Hashing covered calc: instrument=%s %.2f MWh * %s "
            "kgCO2e/MWh = %s kgCO2e",
            instrument_id,
            mwh,
            _safe_str(emission_factor),
            _safe_str(emissions_kg),
        )
        return self._compute_hash("covered_calculation", data)

    def hash_covered_calculation_extended(
        self,
        instrument_id: str,
        mwh: float,
        emission_factor: float,
        emissions_kg: float,
        instrument_type: Optional[str] = None,
        facility_id: Optional[str] = None,
        reporting_period: Optional[str] = None,
        ef_source: Optional[str] = None,
        generation_technology: Optional[str] = None,
        ppa_contract_id: Optional[str] = None,
    ) -> str:
        """Hash an extended covered calculation with full context.

        Similar to ``hash_covered_calculation`` but includes additional
        metadata for instrument type, facility, PPA contract reference,
        and generation technology.

        Args:
            instrument_id: Covering instrument identifier.
            mwh: Electricity consumption in MWh.
            emission_factor: Emission factor (kg CO2e/MWh).
            emissions_kg: Calculated emissions (kg CO2e).
            instrument_type: Optional instrument type.
            facility_id: Optional facility identifier.
            reporting_period: Optional reporting period.
            ef_source: Optional emission factor source reference.
            generation_technology: Optional technology (e.g., ``"wind"``,
                ``"solar_pv"``, ``"hydro_run_of_river"``).
            ppa_contract_id: Optional PPA contract identifier.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not instrument_id:
            raise ValueError("instrument_id must not be empty")
        if mwh < 0:
            raise ValueError(
                f"mwh must be non-negative, got {mwh}"
            )

        data: Dict[str, Any] = {
            "instrument_id": instrument_id,
            "mwh": _safe_str(mwh),
            "emission_factor": _safe_str(emission_factor),
            "emissions_kg": _safe_str(emissions_kg),
        }
        if instrument_type is not None:
            data["instrument_type"] = instrument_type
        if facility_id is not None:
            data["facility_id"] = facility_id
        if reporting_period is not None:
            data["reporting_period"] = reporting_period
        if ef_source is not None:
            data["ef_source"] = ef_source
        if generation_technology is not None:
            data["generation_technology"] = generation_technology
        if ppa_contract_id is not None:
            data["ppa_contract_id"] = ppa_contract_id

        logger.info(
            "Hashing extended covered calc: instrument=%s type=%s "
            "%.2f MWh = %s kgCO2e tech=%s",
            instrument_id,
            instrument_type or "N/A",
            mwh,
            _safe_str(emissions_kg),
            generation_technology or "N/A",
        )
        return self._compute_hash("covered_calculation", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Uncovered Calculation
    # ------------------------------------------------------------------

    def hash_uncovered_calculation(
        self,
        mwh: float,
        region: str,
        residual_ef: float,
        emissions_kg: float,
    ) -> str:
        """Hash an emission calculation for uncovered consumption.

        Records the emission calculation for electricity consumption
        not covered by any contractual instrument. Uncovered consumption
        is calculated using the residual mix emission factor for the
        applicable market/region.

        The formula is:
            emissions_kg = mwh * residual_ef

        The residual mix represents the generation mix remaining after
        all contractual instruments (RECs, GOs, etc.) have been
        subtracted from the total generation mix. It is published by
        organizations such as AIB (European Residual Mix), Green-e
        (US residual mix), and national grid operators.

        Args:
            mwh: Uncovered electricity consumption in megawatt-hours.
                Must be non-negative.
            region: Market region or grid zone identifier (e.g.,
                ``"US-ERCOT"``, ``"DE"``, ``"GB"``).
            residual_ef: Residual mix emission factor in kg CO2e per
                MWh. Must be non-negative.
            emissions_kg: Calculated emissions in kg CO2e.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``region`` is empty, or ``mwh`` or
                ``residual_ef`` is negative.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_uncovered_calculation(
            ...     mwh=2000.0,
            ...     region="US-WECC",
            ...     residual_ef=350.0,
            ...     emissions_kg=700000.0,
            ... )
            >>> assert len(h) == 64
        """
        if mwh < 0:
            raise ValueError(
                f"mwh must be non-negative, got {mwh}"
            )
        if not region:
            raise ValueError("region must not be empty")
        if residual_ef < 0:
            raise ValueError(
                f"residual_ef must be non-negative, got {residual_ef}"
            )

        data = {
            "mwh": _safe_str(mwh),
            "region": region,
            "residual_ef": _safe_str(residual_ef),
            "emissions_kg": _safe_str(emissions_kg),
        }

        logger.info(
            "Hashing uncovered calc: %.2f MWh * %s kgCO2e/MWh "
            "(region=%s) = %s kgCO2e",
            mwh,
            _safe_str(residual_ef),
            region,
            _safe_str(emissions_kg),
        )
        return self._compute_hash("uncovered_calculation", data)

    def hash_uncovered_calculation_extended(
        self,
        mwh: float,
        region: str,
        residual_ef: float,
        emissions_kg: float,
        facility_id: Optional[str] = None,
        reporting_period: Optional[str] = None,
        residual_mix_source: Optional[str] = None,
        residual_mix_year: Optional[int] = None,
        total_consumption_mwh: Optional[float] = None,
        covered_mwh: Optional[float] = None,
    ) -> str:
        """Hash an extended uncovered calculation with full context.

        Similar to ``hash_uncovered_calculation`` but includes additional
        metadata for the residual mix source, facility context, and
        covered/uncovered breakdown.

        Args:
            mwh: Uncovered MWh.
            region: Market region identifier.
            residual_ef: Residual mix emission factor (kg CO2e/MWh).
            emissions_kg: Calculated emissions (kg CO2e).
            facility_id: Optional facility identifier.
            reporting_period: Optional reporting period.
            residual_mix_source: Optional source (e.g., ``"AIB_2024"``,
                ``"Green-e_2024"``).
            residual_mix_year: Optional year of the residual mix data.
            total_consumption_mwh: Optional total facility consumption.
            covered_mwh: Optional covered consumption for context.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if mwh < 0:
            raise ValueError(
                f"mwh must be non-negative, got {mwh}"
            )
        if not region:
            raise ValueError("region must not be empty")

        data: Dict[str, Any] = {
            "mwh": _safe_str(mwh),
            "region": region,
            "residual_ef": _safe_str(residual_ef),
            "emissions_kg": _safe_str(emissions_kg),
        }
        if facility_id is not None:
            data["facility_id"] = facility_id
        if reporting_period is not None:
            data["reporting_period"] = reporting_period
        if residual_mix_source is not None:
            data["residual_mix_source"] = residual_mix_source
        if residual_mix_year is not None:
            data["residual_mix_year"] = residual_mix_year
        if total_consumption_mwh is not None:
            data["total_consumption_mwh"] = _safe_str(total_consumption_mwh)
        if covered_mwh is not None:
            data["covered_mwh"] = _safe_str(covered_mwh)

        logger.info(
            "Hashing extended uncovered calc: %.2f MWh region=%s "
            "source=%s = %s kgCO2e",
            mwh,
            region,
            residual_mix_source or "N/A",
            _safe_str(emissions_kg),
        )
        return self._compute_hash("uncovered_calculation", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Residual Mix Lookup
    # ------------------------------------------------------------------

    def hash_residual_mix_lookup(
        self,
        region: str,
        factor: float,
        source: str,
    ) -> str:
        """Hash a residual mix emission factor lookup.

        Records the retrieval of a residual mix emission factor from
        a reference database. The residual mix represents the grid
        generation mix after subtracting all tracked contractual
        instruments. It is the default emission factor applied to
        uncovered consumption under the market-based method.

        Key residual mix sources:
        - AIB European Residual Mix (annual publication)
        - Green-e US Residual Mix (annual publication)
        - National grid operator residual mix publications
        - Regulatory default residual mix factors

        Args:
            region: Market region or country identifier (e.g.,
                ``"DE"``, ``"US-ERCOT"``, ``"GB"``).
            factor: Residual mix emission factor in kg CO2e per MWh.
                Must be non-negative.
            source: Source database or publication reference (e.g.,
                ``"AIB_RM_2024"``, ``"Green-e_RM_2024"``).

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``region`` or ``source`` is empty, or
                ``factor`` is negative.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_residual_mix_lookup(
            ...     region="DE",
            ...     factor=420.5,
            ...     source="AIB_RM_2024",
            ... )
            >>> assert len(h) == 64
        """
        if not region:
            raise ValueError("region must not be empty")
        if factor < 0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if not source:
            raise ValueError("source must not be empty")

        data = {
            "region": region,
            "factor": _safe_str(factor),
            "source": source,
        }

        logger.info(
            "Hashing residual mix lookup: region=%s factor=%s "
            "kgCO2e/MWh source=%s",
            region,
            _safe_str(factor),
            source,
        )
        return self._compute_hash("residual_mix_lookup", data)

    def hash_residual_mix_lookup_extended(
        self,
        region: str,
        factor: float,
        source: str,
        reference_year: Optional[int] = None,
        co2_factor: Optional[float] = None,
        ch4_factor: Optional[float] = None,
        n2o_factor: Optional[float] = None,
        generation_mix: Optional[Dict[str, float]] = None,
        data_quality: Optional[str] = None,
        publication_date: Optional[str] = None,
    ) -> str:
        """Hash an extended residual mix lookup with per-gas breakdown.

        Similar to ``hash_residual_mix_lookup`` but includes per-gas
        emission factors, generation mix, and data quality information.

        Args:
            region: Market region identifier.
            factor: Combined residual mix factor (kg CO2e/MWh).
            source: Source reference.
            reference_year: Optional year the factor represents.
            co2_factor: Optional CO2 factor (kg CO2/MWh).
            ch4_factor: Optional CH4 factor (kg CH4/MWh).
            n2o_factor: Optional N2O factor (kg N2O/MWh).
            generation_mix: Optional dict of fuel source to percentage.
            data_quality: Optional data quality indicator.
            publication_date: Optional date the factor was published.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not region:
            raise ValueError("region must not be empty")
        if factor < 0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if not source:
            raise ValueError("source must not be empty")

        data: Dict[str, Any] = {
            "region": region,
            "factor": _safe_str(factor),
            "source": source,
        }
        if reference_year is not None:
            data["reference_year"] = reference_year
        if co2_factor is not None:
            data["co2_factor"] = _safe_str(co2_factor)
        if ch4_factor is not None:
            data["ch4_factor"] = _safe_str(ch4_factor)
        if n2o_factor is not None:
            data["n2o_factor"] = _safe_str(n2o_factor)
        if generation_mix is not None:
            data["generation_mix"] = {
                k: _safe_str(v) for k, v in generation_mix.items()
            }
        if data_quality is not None:
            data["data_quality"] = data_quality
        if publication_date is not None:
            data["publication_date"] = publication_date

        logger.info(
            "Hashing extended residual mix: region=%s factor=%s "
            "source=%s year=%s",
            region,
            _safe_str(factor),
            source,
            _safe_str(reference_year),
        )
        return self._compute_hash("residual_mix_lookup", data)

    # ------------------------------------------------------------------
    # Domain-specific hashing methods: Supplier Factor
    # ------------------------------------------------------------------

    def hash_supplier_factor(
        self,
        supplier_id: str,
        emission_factor: float,
        year: int,
    ) -> str:
        """Hash a supplier-specific emission factor.

        Records the emission factor provided by an electricity supplier
        for market-based calculations. Supplier-specific factors are
        one of the contractual instruments in the GHG Protocol Scope 2
        hierarchy and represent the actual generation mix of the
        supplier.

        Supplier-specific factors rank higher than residual mix in the
        GHG Protocol hierarchy when the supplier provides a factor
        that meets the Scope 2 Quality Criteria.

        Args:
            supplier_id: Identifier for the electricity supplier (e.g.,
                ``"SUP-ENEL-IT"``, ``"SUP-EDF-FR"``).
            emission_factor: Supplier's emission factor in kg CO2e per
                MWh. Must be non-negative.
            year: Year the emission factor applies to. Must be positive.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Raises:
            ValueError: If ``supplier_id`` is empty, ``emission_factor``
                is negative, or ``year`` is not positive.

        Example:
            >>> prov = Scope2MarketProvenance()
            >>> prov.hash_input({"facility_id": "FAC-001"})
            '...'
            >>> h = prov.hash_supplier_factor(
            ...     supplier_id="SUP-ENEL-IT",
            ...     emission_factor=280.0,
            ...     year=2024,
            ... )
            >>> assert len(h) == 64
        """
        if not supplier_id:
            raise ValueError("supplier_id must not be empty")
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, "
                f"got {emission_factor}"
            )
        if year <= 0:
            raise ValueError(f"year must be positive, got {year}")

        data = {
            "supplier_id": supplier_id,
            "emission_factor": _safe_str(emission_factor),
            "year": year,
        }

        logger.info(
            "Hashing supplier factor: supplier=%s ef=%s "
            "kgCO2e/MWh year=%d",
            supplier_id,
            _safe_str(emission_factor),
            year,
        )
        return self._compute_hash("supplier_factor", data)

    def hash_supplier_factor_extended(
        self,
        supplier_id: str,
        emission_factor: float,
        year: int,
        supplier_name: Optional[str] = None,
        country: Optional[str] = None,
        verification_status: Optional[str] = None,
        verification_body: Optional[str] = None,
        co2_ef: Optional[float] = None,
        ch4_ef: Optional[float] = None,
        n2o_ef: Optional[float] = None,
        generation_mix: Optional[Dict[str, float]] = None,
        meets_quality_criteria: Optional[bool] = None,
    ) -> str:
        """Hash an extended supplier factor with verification details.

        Similar to ``hash_supplier_factor`` but includes verification
        status, per-gas breakdown, generation mix, and quality criteria
        compliance.

        Args:
            supplier_id: Supplier identifier.
            emission_factor: Combined emission factor (kg CO2e/MWh).
            year: Applicable year.
            supplier_name: Optional human-readable supplier name.
            country: Optional ISO country code.
            verification_status: Optional status (e.g., ``"verified"``,
                ``"self_reported"``, ``"estimated"``).
            verification_body: Optional name of verification body.
            co2_ef: Optional CO2 factor (kg CO2/MWh).
            ch4_ef: Optional CH4 factor (kg CH4/MWh).
            n2o_ef: Optional N2O factor (kg N2O/MWh).
            generation_mix: Optional generation mix dictionary.
            meets_quality_criteria: Optional GHG Protocol quality flag.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        if not supplier_id:
            raise ValueError("supplier_id must not be empty")
        if emission_factor < 0:
            raise ValueError(
                f"emission_factor must be non-negative, "
                f"got {emission_factor}"
            )
        if year <= 0:
            raise ValueError(f"year must be positive, got {year}")

        data: Dict[str, Any] = {
            "supplier_id": supplier_id,
            "emission_factor": _safe_str(emission_factor),
            "year": year,
        }
        if supplier_name is not None:
            data["supplier_name"] = supplier_name
        if country is not None:
            data["country"] = country
        if verification_status is not None:
            data["verification_status"] = verification_status
        if verification_body is not None:
            data["verification_body"] = verification_body
        if co2_ef is not None:
            data["co2_ef"] = _safe_str(co2_ef)
        if ch4_ef is not None:
            data["ch4_ef"] = _safe_str(ch4_ef)
        if n2o_ef is not None:
            data["n2o_ef"] = _safe_str(n2o_ef)
        if generation_mix is not None:
            data["generation_mix"] = {
                k: _safe_str(v) for k, v in generation_mix.items()
            }
        if meets_quality_criteria is not None:
            data["meets_quality_criteria"] = meets_quality_criteria

        logger.info(
            "Hashing extended supplier factor: supplier=%s(%s) "
            "ef=%s year=%d verified=%s",
            supplier_id,
            supplier_name or "N/A",
            _safe_str(emission_factor),
            year,
            verification_status or "N/A",
        )
        return self._compute_hash("supplier_factor", data)
