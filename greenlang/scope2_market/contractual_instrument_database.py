# -*- coding: utf-8 -*-
"""
Engine 1: Contractual Instrument Database Engine for AGENT-MRV-010.

Stores and retrieves contractual instrument metadata, residual mix factors,
energy source emission factors, supplier-specific factors, and quality
criteria for Scope 2 market-based emission calculations per GHG Protocol
Scope 2 Guidance (2015).

Built-in Data:
- 10 contractual instrument types (RECs, GOs, I-RECs, PPAs, etc.)
- 60+ residual mix factors by country/region (kgCO2e/kWh)
- 11 energy source emission factors (kgCO2e/kWh)
- 30+ supplier default emission factors by country (kgCO2e/kWh)
- 7 quality criteria with weights for instrument validation
- 8 tracking system registries (ERCOT, WREGIS, M-RETS, GATS, etc.)

All values as Decimal with ROUND_HALF_UP for zero-hallucination guarantees.
Thread-safe singleton via RLock. SHA-256 provenance on every operation.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-010 Scope 2 Market-Based Emissions (GL-MRV-SCOPE2-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _canonical_json(data: Dict[str, Any]) -> str:
    """Serialize dictionary to canonical JSON for hashing.

    Uses sort_keys=True and default=str for deterministic output
    regardless of insertion order or non-standard types.

    Args:
        data: Dictionary to serialize.

    Returns:
        Canonical JSON string with sorted keys.
    """
    return json.dumps(data, sort_keys=True, default=str)


def _sha256(payload: str) -> str:
    """Compute SHA-256 hex digest of a string payload.

    Args:
        payload: String to hash.

    Returns:
        64-character lowercase hex digest.
    """
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Contractual Instrument Types (10 types)
# ---------------------------------------------------------------------------
# Source: GHG Protocol Scope 2 Guidance (2015), Section 7.
# Each type includes metadata about applicability, tracking systems,
# whether it is renewable-only, typical EF, and vintage validity.
# ---------------------------------------------------------------------------

INSTRUMENT_TYPES: Dict[str, Dict[str, Any]] = {
    "REC": {
        "name": "Renewable Energy Certificate",
        "description": (
            "A market-based instrument representing the property rights to "
            "the environmental, social, and other non-power attributes of "
            "renewable electricity generation. One REC equals 1 MWh of "
            "renewable generation. Widely used in the US under Green-e "
            "certification."
        ),
        "region_applicability": ["US", "CA"],
        "tracking_systems": ["ERCOT", "WREGIS", "M-RETS", "GATS", "NEPOOL-GIS", "PJM-GATS"],
        "renewable_only": True,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 5,
    },
    "GO": {
        "name": "Guarantee of Origin",
        "description": (
            "An electronic certificate issued under EU Directive 2018/2001 "
            "(Renewable Energy Directive II) confirming the source and "
            "attributes of electricity generation in European markets. "
            "One GO equals 1 MWh. Tracked via the AIB Hub."
        ),
        "region_applicability": ["EU", "EEA", "CH", "GB"],
        "tracking_systems": ["AIB", "OFGEM-REGO"],
        "renewable_only": False,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 2,
    },
    "I-REC": {
        "name": "International Renewable Energy Certificate",
        "description": (
            "An international tracking instrument standardised by the "
            "I-REC Standard Foundation for countries without their own "
            "national certificate systems. Widely used in Latin America, "
            "Asia-Pacific, Middle East, and Africa."
        ),
        "region_applicability": ["GLOBAL"],
        "tracking_systems": ["I-REC"],
        "renewable_only": True,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 3,
    },
    "PPA_PHYSICAL": {
        "name": "Physical Power Purchase Agreement",
        "description": (
            "A long-term bilateral contract for the physical delivery of "
            "electricity from a specific generation asset to the purchaser's "
            "load. Includes both the energy and the environmental attributes. "
            "Commonly used for on-site or near-site renewable generation."
        ),
        "region_applicability": ["GLOBAL"],
        "tracking_systems": ["ERCOT", "WREGIS", "M-RETS", "AIB", "I-REC"],
        "renewable_only": False,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 1,
    },
    "PPA_VIRTUAL": {
        "name": "Virtual Power Purchase Agreement",
        "description": (
            "A financial contract-for-differences (CFD) between a buyer "
            "and a renewable energy generator where no physical electricity "
            "delivery occurs. The buyer receives the environmental attributes "
            "(RECs/GOs) separately from the financial settlement."
        ),
        "region_applicability": ["GLOBAL"],
        "tracking_systems": ["ERCOT", "WREGIS", "M-RETS", "AIB", "I-REC"],
        "renewable_only": False,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 1,
    },
    "GREEN_TARIFF": {
        "name": "Green Tariff / Utility Green Pricing",
        "description": (
            "A utility-offered retail electricity product where the utility "
            "procures renewable energy or RECs on behalf of the customer. "
            "The customer pays a premium price per kWh and receives the "
            "environmental attributes associated with the renewable "
            "generation."
        ),
        "region_applicability": ["US", "EU", "GB", "AU"],
        "tracking_systems": ["ERCOT", "WREGIS", "M-RETS", "AIB"],
        "renewable_only": True,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 1,
    },
    "SUPPLIER_SPECIFIC": {
        "name": "Supplier-Specific Emission Factor",
        "description": (
            "An emission factor provided directly by the electricity "
            "supplier reflecting their generation or procurement mix. "
            "The supplier discloses their portfolio emission intensity, "
            "which the customer uses for market-based accounting. "
            "Must be supported by contractual evidence."
        ),
        "region_applicability": ["GLOBAL"],
        "tracking_systems": [],
        "renewable_only": False,
        "typical_ef_kgco2e_kwh": Decimal("0.350"),
        "vintage_max_years": 2,
    },
    "RESIDUAL_MIX": {
        "name": "Residual Mix Emission Factor",
        "description": (
            "A grid-average emission factor adjusted to remove the "
            "attributes claimed through contractual instruments (RECs, GOs). "
            "Used as the default market-based factor when no contractual "
            "instrument is held. Published by AIB in Europe and Green-e "
            "in the US."
        ),
        "region_applicability": ["GLOBAL"],
        "tracking_systems": [],
        "renewable_only": False,
        "typical_ef_kgco2e_kwh": Decimal("0.450"),
        "vintage_max_years": 2,
    },
    "REGO": {
        "name": "Renewable Energy Guarantee of Origin (UK)",
        "description": (
            "UK-specific certificate tracked by Ofgem confirming that "
            "electricity was generated from a renewable source. One REGO "
            "equals 1 MWh. Used for UK market-based Scope 2 reporting "
            "post-Brexit."
        ),
        "region_applicability": ["GB"],
        "tracking_systems": ["OFGEM-REGO"],
        "renewable_only": True,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 2,
    },
    "TIGR": {
        "name": "Tradable Instrument for Global Renewables",
        "description": (
            "A certificate standard developed by APX (now part of Regis) "
            "for international markets, designed to provide a bridge "
            "where I-REC or other systems are not available. One TIGR "
            "represents 1 MWh of renewable generation."
        ),
        "region_applicability": ["GLOBAL"],
        "tracking_systems": ["APX-TIGR"],
        "renewable_only": True,
        "typical_ef_kgco2e_kwh": Decimal("0.000"),
        "vintage_max_years": 3,
    },
}


# ---------------------------------------------------------------------------
# Residual Mix Factors by Country/Region (kgCO2e/kWh)
# ---------------------------------------------------------------------------
# Source: AIB European Residual Mixes (2024), Green-e Residual Mix (US),
#         National registry data for non-EU/US markets.
#
# Residual mix = grid average minus attributes claimed via instruments.
# Used as default market-based EF when organisation holds no instruments.
# ---------------------------------------------------------------------------

RESIDUAL_MIX_FACTORS: Dict[str, Decimal] = {
    # EU Member States (AIB 2024 residual mixes, kgCO2e/kWh)
    "AT": Decimal("0.227"),
    "BE": Decimal("0.210"),
    "BG": Decimal("0.432"),
    "HR": Decimal("0.218"),
    "CY": Decimal("0.634"),
    "CZ": Decimal("0.483"),
    "DK": Decimal("0.351"),
    "EE": Decimal("0.723"),
    "FI": Decimal("0.131"),
    "FR": Decimal("0.059"),
    "DE": Decimal("0.427"),
    "GR": Decimal("0.423"),
    "HU": Decimal("0.262"),
    "IE": Decimal("0.380"),
    "IT": Decimal("0.305"),
    "LV": Decimal("0.141"),
    "LT": Decimal("0.100"),
    "LU": Decimal("0.357"),
    "MT": Decimal("0.441"),
    "NL": Decimal("0.437"),
    "PL": Decimal("0.719"),
    "PT": Decimal("0.215"),
    "RO": Decimal("0.307"),
    "SK": Decimal("0.145"),
    "SI": Decimal("0.277"),
    "ES": Decimal("0.176"),
    "SE": Decimal("0.029"),
    # EEA and European non-EU
    "NO": Decimal("0.395"),
    "CH": Decimal("0.021"),
    "GB": Decimal("0.299"),
    "IS": Decimal("0.000"),
    "RS": Decimal("0.690"),
    "BA": Decimal("0.755"),
    "MK": Decimal("0.610"),
    "AL": Decimal("0.015"),
    "ME": Decimal("0.390"),
    # US regions (Green-e residual mix, approximated per eGRID subregion)
    "US": Decimal("0.425"),
    "US-CAMX": Decimal("0.295"),
    "US-ERCT": Decimal("0.420"),
    "US-FRCC": Decimal("0.430"),
    "US-MROW": Decimal("0.500"),
    "US-NEWE": Decimal("0.280"),
    "US-NWPP": Decimal("0.310"),
    "US-NYUP": Decimal("0.155"),
    "US-RFCE": Decimal("0.350"),
    "US-RFCW": Decimal("0.510"),
    "US-RMPA": Decimal("0.585"),
    "US-SPNO": Decimal("0.480"),
    "US-SPSO": Decimal("0.460"),
    "US-SRMV": Decimal("0.385"),
    "US-SRMW": Decimal("0.680"),
    "US-SRSO": Decimal("0.430"),
    "US-SRTV": Decimal("0.415"),
    "US-SRVC": Decimal("0.330"),
    # Americas
    "CA": Decimal("0.140"),
    "MX": Decimal("0.460"),
    "BR": Decimal("0.090"),
    "AR": Decimal("0.340"),
    "CL": Decimal("0.380"),
    "CO": Decimal("0.170"),
    # Asia-Pacific
    "CN": Decimal("0.581"),
    "JP": Decimal("0.497"),
    "IN": Decimal("0.725"),
    "KR": Decimal("0.455"),
    "AU": Decimal("0.680"),
    "NZ": Decimal("0.100"),
    "TW": Decimal("0.530"),
    "SG": Decimal("0.425"),
    "TH": Decimal("0.465"),
    "MY": Decimal("0.575"),
    "ID": Decimal("0.730"),
    "PH": Decimal("0.565"),
    "VN": Decimal("0.510"),
    # Middle East
    "AE": Decimal("0.415"),
    "SA": Decimal("0.600"),
    "IL": Decimal("0.510"),
    # Africa
    "ZA": Decimal("0.950"),
    "EG": Decimal("0.450"),
    "NG": Decimal("0.430"),
    "KE": Decimal("0.110"),
    # Other
    "TR": Decimal("0.420"),
    "RU": Decimal("0.360"),
    # World fallback
    "WORLD": Decimal("0.436"),
}


# ---------------------------------------------------------------------------
# Energy Source Emission Factors (kgCO2e/kWh)
# ---------------------------------------------------------------------------
# Source: IPCC AR6, GHG Protocol, lifecycle emission factors for
#         electricity generation by source. Zero for renewables per
#         GHG Protocol Scope 2 market-based convention (combustion only).
# ---------------------------------------------------------------------------

ENERGY_SOURCE_EF: Dict[str, Decimal] = {
    "solar": Decimal("0.000"),
    "wind": Decimal("0.000"),
    "hydro": Decimal("0.000"),
    "nuclear": Decimal("0.000"),
    "biomass": Decimal("0.000"),
    "geothermal": Decimal("0.000"),
    "natural_gas_ccgt": Decimal("0.340"),
    "natural_gas_ocgt": Decimal("0.490"),
    "coal": Decimal("0.910"),
    "oil": Decimal("0.650"),
    "mixed": Decimal("0.436"),
}

#: Set of energy sources with zero combustion emissions.
_ZERO_EMISSION_SOURCES: frozenset = frozenset({
    "solar", "wind", "hydro", "nuclear", "biomass", "geothermal",
})


# ---------------------------------------------------------------------------
# Supplier Default Emission Factors by Country (kgCO2e/kWh)
# ---------------------------------------------------------------------------
# Source: National registry data, IEA electricity mix estimates.
# Used when supplier does not disclose their specific emission factor.
# Falls back to residual mix if country not available.
# ---------------------------------------------------------------------------

SUPPLIER_DEFAULT_EF: Dict[str, Decimal] = {
    "US": Decimal("0.390"),
    "CA": Decimal("0.130"),
    "MX": Decimal("0.440"),
    "BR": Decimal("0.080"),
    "AR": Decimal("0.325"),
    "CL": Decimal("0.370"),
    "CO": Decimal("0.160"),
    "GB": Decimal("0.265"),
    "DE": Decimal("0.400"),
    "FR": Decimal("0.060"),
    "IT": Decimal("0.275"),
    "ES": Decimal("0.165"),
    "NL": Decimal("0.395"),
    "BE": Decimal("0.195"),
    "PL": Decimal("0.670"),
    "SE": Decimal("0.012"),
    "NO": Decimal("0.010"),
    "DK": Decimal("0.140"),
    "FI": Decimal("0.085"),
    "AT": Decimal("0.095"),
    "CH": Decimal("0.018"),
    "IE": Decimal("0.320"),
    "PT": Decimal("0.195"),
    "CN": Decimal("0.570"),
    "JP": Decimal("0.475"),
    "IN": Decimal("0.720"),
    "KR": Decimal("0.430"),
    "AU": Decimal("0.670"),
    "NZ": Decimal("0.095"),
    "SG": Decimal("0.415"),
    "ZA": Decimal("0.940"),
    "AE": Decimal("0.405"),
    "SA": Decimal("0.590"),
    "TR": Decimal("0.400"),
    "RU": Decimal("0.350"),
}


# ---------------------------------------------------------------------------
# Instrument Quality Criteria (7 criteria)
# ---------------------------------------------------------------------------
# Source: GHG Protocol Scope 2 Guidance, Section 7.2 Quality Criteria.
# Each criterion has a description, weight (0-1), and scoring guidance.
# ---------------------------------------------------------------------------

QUALITY_CRITERIA: Dict[str, Dict[str, Any]] = {
    "conveyance": {
        "name": "Conveyance of Attributes",
        "description": (
            "The contractual instrument must explicitly convey the "
            "environmental attributes (emission rate, fuel type, generation "
            "technology) from the generator to the reporting entity."
        ),
        "weight": Decimal("0.20"),
        "scoring": "binary",
    },
    "unique_claim": {
        "name": "Unique Claim",
        "description": (
            "The instrument must represent a unique claim on generation "
            "attributes. The same MWh of generation must not be claimed "
            "by multiple parties. Tracked via registry serial numbers."
        ),
        "weight": Decimal("0.20"),
        "scoring": "binary",
    },
    "geographic_match": {
        "name": "Geographic / Market Boundary Match",
        "description": (
            "The generation facility associated with the instrument must "
            "be located within the same market boundary (grid interconnect "
            "or country) as the consumer's load."
        ),
        "weight": Decimal("0.15"),
        "scoring": "tiered",
    },
    "temporal_match": {
        "name": "Temporal / Vintage Match",
        "description": (
            "The instrument vintage (generation date) must fall within "
            "or close to the reporting period. Shorter vintage gaps yield "
            "higher quality scores."
        ),
        "weight": Decimal("0.15"),
        "scoring": "tiered",
    },
    "tracking_verified": {
        "name": "Verified Tracking System",
        "description": (
            "The instrument must be issued, tracked, and retired in a "
            "recognised tracking system or registry (e.g. WREGIS, AIB, "
            "I-REC, M-RETS) to prevent double-counting."
        ),
        "weight": Decimal("0.15"),
        "scoring": "binary",
    },
    "retirement_evidence": {
        "name": "Retirement / Cancellation Evidence",
        "description": (
            "The instrument must be retired or cancelled in the registry "
            "on behalf of the reporting entity, with a retirement "
            "certificate or confirmation as evidence."
        ),
        "weight": Decimal("0.10"),
        "scoring": "binary",
    },
    "additionality": {
        "name": "Additionality",
        "description": (
            "Whether the instrument supports new or additional renewable "
            "capacity beyond business-as-usual. Not mandatory under "
            "GHG Protocol Scope 2, but recommended for enhanced claims "
            "and valued by RE100."
        ),
        "weight": Decimal("0.05"),
        "scoring": "binary",
    },
}


# ---------------------------------------------------------------------------
# Tracking Systems (8 registries)
# ---------------------------------------------------------------------------
# Source: Registry operator websites and GHG Protocol Scope 2 Appendix.
# ---------------------------------------------------------------------------

TRACKING_SYSTEMS: Dict[str, Dict[str, Any]] = {
    "ERCOT": {
        "name": "Electric Reliability Council of Texas",
        "region": "US-TX",
        "url": "https://www.texasrenewables.com",
        "instrument_types": ["REC"],
        "description": "Texas renewable energy credit tracking system.",
    },
    "WREGIS": {
        "name": "Western Renewable Energy Generation Information System",
        "region": "US-West",
        "url": "https://www.wecc.org/WREGIS",
        "instrument_types": ["REC"],
        "description": (
            "Certificate tracking for the western US interconnection "
            "covering 14 western states and 2 Canadian provinces."
        ),
    },
    "M-RETS": {
        "name": "Midwest Renewable Energy Tracking System",
        "region": "US-Midwest",
        "url": "https://www.mrets.org",
        "instrument_types": ["REC"],
        "description": "REC tracking for midwestern and central US states.",
    },
    "GATS": {
        "name": "Generation Attribute Tracking System (PJM-EIS)",
        "region": "US-PJM",
        "url": "https://gats.pjm-eis.com",
        "instrument_types": ["REC"],
        "description": "Attribute tracking for PJM Interconnection territory.",
    },
    "NEPOOL-GIS": {
        "name": "New England Power Pool Generation Information System",
        "region": "US-NE",
        "url": "https://www.nepoolgis.com",
        "instrument_types": ["REC"],
        "description": "Certificate tracking for the New England states.",
    },
    "AIB": {
        "name": "Association of Issuing Bodies",
        "region": "EU/EEA",
        "url": "https://www.aib-net.org",
        "instrument_types": ["GO"],
        "description": (
            "European hub for Guarantee of Origin issuance, transfer, and "
            "cancellation across EU/EEA member states."
        ),
    },
    "OFGEM-REGO": {
        "name": "Ofgem Renewables and CHP Register",
        "region": "GB",
        "url": "https://www.ofgem.gov.uk/environmental-programmes",
        "instrument_types": ["REGO"],
        "description": (
            "UK Ofgem registry for Renewable Energy Guarantees of Origin "
            "(REGO) certificates."
        ),
    },
    "I-REC": {
        "name": "International REC Standard Foundation",
        "region": "GLOBAL",
        "url": "https://www.irecstandard.org",
        "instrument_types": ["I-REC"],
        "description": (
            "International tracking standard for renewable energy "
            "certificates in countries without national systems."
        ),
    },
}


# ===========================================================================
# ContractualInstrumentDatabaseEngine
# ===========================================================================


class ContractualInstrumentDatabaseEngine:
    """Engine 1: Contractual instrument database for Scope 2 market-based
    emission calculations.

    Manages a comprehensive database of contractual instrument types,
    residual mix emission factors, energy source emission factors,
    supplier-specific factors, and quality criteria for GHG Protocol
    Scope 2 market-based accounting.

    Implements the thread-safe singleton pattern using RLock to ensure
    exactly one instance per process. All arithmetic uses Decimal with
    ROUND_HALF_UP for zero-hallucination deterministic calculations.
    Every lookup and mutation produces a SHA-256 provenance hash for
    complete audit trails.

    Thread Safety:
        Uses ``threading.RLock`` for singleton creation and all mutable
        state access. Immutable built-in data (INSTRUMENT_TYPES,
        RESIDUAL_MIX_FACTORS, ENERGY_SOURCE_EF) is inherently thread-safe.

    Attributes:
        ENGINE_ID: Constant identifier for this engine.
        ENGINE_VERSION: Semantic version string.

    Example:
        >>> engine = ContractualInstrumentDatabaseEngine()
        >>> ef = engine.get_residual_mix_factor("DE")
        >>> assert ef == Decimal("0.427")
        >>> info = engine.get_instrument_info("REC")
        >>> assert info["renewable_only"] is True
    """

    ENGINE_ID: str = "contractual_instrument_database"
    ENGINE_VERSION: str = "1.0.0"

    _instance: Optional[ContractualInstrumentDatabaseEngine] = None
    _lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Singleton construction
    # ------------------------------------------------------------------

    def __new__(
        cls,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> ContractualInstrumentDatabaseEngine:
        """Return the singleton ContractualInstrumentDatabaseEngine instance.

        Uses double-checked locking with an RLock to ensure exactly one
        instance is created even under concurrent first-access.

        Args:
            config: Optional configuration object (ignored after first init).
            metrics: Optional metrics recorder (ignored after first init).
            provenance: Optional provenance tracker (ignored after first init).

        Returns:
            The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(
        self,
        config: Any = None,
        metrics: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize the contractual instrument database engine.

        Idempotent: after the first call, subsequent invocations are
        silently skipped to prevent duplicate initialisation.

        Args:
            config: Optional configuration object for engine tuning.
            metrics: Optional Prometheus metrics recorder. Must expose
                ``record_residual_mix_lookup(source)`` and
                ``record_instrument_registered(type)`` methods.
            provenance: Optional provenance tracker for chain hashing.
        """
        if self._initialized:
            return

        self._config = config
        self._metrics = metrics
        self._provenance = provenance

        # Mutable state protected by _state_lock
        self._state_lock = threading.RLock()
        self._custom_residual_mix: Dict[str, Dict[str, Any]] = {}
        self._supplier_factors: Dict[str, Dict[str, Any]] = {}

        # Counters
        self._lookup_count: int = 0
        self._mutation_count: int = 0
        self._validation_count: int = 0
        self._provenance_hashes: List[str] = []

        self._initialized = True
        logger.info(
            "ContractualInstrumentDatabaseEngine v%s initialized "
            "(instrument_types=%d, residual_mix_regions=%d, "
            "energy_sources=%d, supplier_defaults=%d)",
            self.ENGINE_VERSION,
            len(INSTRUMENT_TYPES),
            len(RESIDUAL_MIX_FACTORS),
            len(ENERGY_SOURCE_EF),
            len(SUPPLIER_DEFAULT_EF),
        )

    # ------------------------------------------------------------------
    # Provenance helper
    # ------------------------------------------------------------------

    def _record_provenance(
        self,
        operation: str,
        data: Dict[str, Any],
    ) -> str:
        """Compute and record a SHA-256 provenance hash for an operation.

        Args:
            operation: Name of the operation (e.g. 'get_residual_mix_factor').
            data: Dictionary of operation inputs and outputs to hash.

        Returns:
            64-character SHA-256 hex digest.
        """
        payload = {
            "engine": self.ENGINE_ID,
            "operation": operation,
            "timestamp": _utcnow().isoformat(),
            "data": data,
        }
        hash_value = _sha256(_canonical_json(payload))
        with self._state_lock:
            self._provenance_hashes.append(hash_value)
        if self._provenance:
            try:
                self._provenance.record(operation, hash_value, data)
            except Exception:
                pass
        return hash_value

    # ------------------------------------------------------------------
    # Metrics helper
    # ------------------------------------------------------------------

    def _record_metric_residual_mix(self, source: str) -> None:
        """Record a residual mix factor lookup metric.

        Args:
            source: The source label (e.g. 'builtin', 'custom').
        """
        if self._metrics:
            try:
                self._metrics.record_residual_mix_lookup(source)
            except Exception:
                pass

    def _record_metric_instrument(self, instrument_type: str) -> None:
        """Record an instrument registration metric.

        Args:
            instrument_type: The instrument type label.
        """
        if self._metrics:
            try:
                self._metrics.record_instrument_registered(instrument_type)
            except Exception:
                pass

    # ==================================================================
    # PUBLIC METHODS: Instrument Type Lookups
    # ==================================================================

    def get_instrument_info(
        self,
        instrument_type: str,
    ) -> Dict[str, Any]:
        """Retrieve metadata for a contractual instrument type.

        Args:
            instrument_type: Instrument type code (e.g. 'REC', 'GO',
                'I-REC', 'PPA_PHYSICAL', 'PPA_VIRTUAL', 'GREEN_TARIFF',
                'SUPPLIER_SPECIFIC', 'RESIDUAL_MIX', 'REGO', 'TIGR').

        Returns:
            Dictionary with name, description, region_applicability,
            tracking_systems, renewable_only, typical_ef_kgco2e_kwh,
            vintage_max_years, and provenance_hash.

        Raises:
            ValueError: If instrument_type is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = instrument_type.upper().replace("-", "-")
        if key not in INSTRUMENT_TYPES:
            raise ValueError(
                f"Unknown instrument type: {instrument_type!r}. "
                f"Valid types: {sorted(INSTRUMENT_TYPES.keys())}"
            )

        info = dict(INSTRUMENT_TYPES[key])
        info["instrument_type"] = key
        provenance_hash = self._record_provenance(
            "get_instrument_info",
            {"instrument_type": key},
        )
        info["provenance_hash"] = provenance_hash

        logger.debug(
            "Instrument info lookup: type=%s, name=%s",
            key,
            info["name"],
        )
        return info

    def list_instruments_by_region(
        self,
        region: str,
    ) -> List[Dict[str, Any]]:
        """List all instrument types applicable to a region.

        Args:
            region: Region code (ISO 3166-1 alpha-2 or 'GLOBAL', 'EU',
                'EEA').

        Returns:
            List of instrument info dictionaries matching the region.
        """
        with self._state_lock:
            self._lookup_count += 1

        region_upper = region.upper()
        results: List[Dict[str, Any]] = []

        for key, info in INSTRUMENT_TYPES.items():
            applicability: List[str] = info["region_applicability"]
            if region_upper in applicability or "GLOBAL" in applicability:
                entry = dict(info)
                entry["instrument_type"] = key
                results.append(entry)

        provenance_hash = self._record_provenance(
            "list_instruments_by_region",
            {"region": region_upper, "count": len(results)},
        )

        for r in results:
            r["provenance_hash"] = provenance_hash

        logger.debug(
            "Instruments by region: region=%s, count=%d",
            region_upper,
            len(results),
        )
        return results

    def get_vintage_validity(
        self,
        instrument_type: str,
    ) -> int:
        """Get the maximum vintage validity period for an instrument type.

        Args:
            instrument_type: Instrument type code.

        Returns:
            Maximum number of years a vintage is considered valid.

        Raises:
            ValueError: If instrument_type is not recognised.
        """
        key = instrument_type.upper()
        if key not in INSTRUMENT_TYPES:
            raise ValueError(
                f"Unknown instrument type: {instrument_type!r}. "
                f"Valid types: {sorted(INSTRUMENT_TYPES.keys())}"
            )
        return INSTRUMENT_TYPES[key]["vintage_max_years"]

    def validate_vintage(
        self,
        instrument_type: str,
        vintage_year: int,
        reporting_year: int,
    ) -> bool:
        """Validate that an instrument vintage is within the allowable period.

        Args:
            instrument_type: Instrument type code.
            vintage_year: Year the energy was generated.
            reporting_year: Year being reported.

        Returns:
            True if the vintage is valid for the reporting year.

        Raises:
            ValueError: If instrument_type is not recognised or years
                are invalid.
        """
        if vintage_year > reporting_year:
            raise ValueError(
                f"vintage_year ({vintage_year}) cannot be after "
                f"reporting_year ({reporting_year})"
            )
        if vintage_year < 1990 or reporting_year < 1990:
            raise ValueError("Years must be >= 1990")

        max_years = self.get_vintage_validity(instrument_type)
        age = reporting_year - vintage_year
        is_valid = age <= max_years

        self._record_provenance(
            "validate_vintage",
            {
                "instrument_type": instrument_type.upper(),
                "vintage_year": vintage_year,
                "reporting_year": reporting_year,
                "max_years": max_years,
                "age": age,
                "is_valid": is_valid,
            },
        )

        logger.debug(
            "Vintage validation: type=%s, vintage=%d, reporting=%d, "
            "age=%d, max=%d, valid=%s",
            instrument_type,
            vintage_year,
            reporting_year,
            age,
            max_years,
            is_valid,
        )
        return is_valid

    # ==================================================================
    # PUBLIC METHODS: Residual Mix Factors
    # ==================================================================

    def get_residual_mix_factor(
        self,
        region: str,
    ) -> Decimal:
        """Get the residual mix emission factor for a region.

        Checks custom factors first, then built-in data, then falls back
        to the WORLD average.

        Args:
            region: Region code (ISO 3166-1 alpha-2, US subregion like
                'US-CAMX', or 'WORLD').

        Returns:
            Residual mix emission factor in kgCO2e/kWh as Decimal.
        """
        with self._state_lock:
            self._lookup_count += 1

        region_upper = region.upper()

        # Check custom factors first
        with self._state_lock:
            if region_upper in self._custom_residual_mix:
                factor = self._custom_residual_mix[region_upper]["factor"]
                self._record_metric_residual_mix("custom")
                self._record_provenance(
                    "get_residual_mix_factor",
                    {"region": region_upper, "factor": str(factor), "source": "custom"},
                )
                return factor

        # Built-in data
        if region_upper in RESIDUAL_MIX_FACTORS:
            factor = RESIDUAL_MIX_FACTORS[region_upper]
            self._record_metric_residual_mix("builtin")
            self._record_provenance(
                "get_residual_mix_factor",
                {"region": region_upper, "factor": str(factor), "source": "builtin"},
            )
            return factor

        # World fallback
        logger.warning(
            "No residual mix factor for region %s, using WORLD average",
            region_upper,
        )
        factor = RESIDUAL_MIX_FACTORS["WORLD"]
        self._record_metric_residual_mix("world_fallback")
        self._record_provenance(
            "get_residual_mix_factor",
            {"region": region_upper, "factor": str(factor), "source": "world_fallback"},
        )
        return factor

    def get_residual_mix_with_metadata(
        self,
        region: str,
    ) -> Dict[str, Any]:
        """Get residual mix factor with full metadata for audit trails.

        Args:
            region: Region code.

        Returns:
            Dictionary with region, factor, source, year, provenance_hash,
            and data_quality_tier.
        """
        with self._state_lock:
            self._lookup_count += 1

        region_upper = region.upper()
        source = "builtin"
        year = 2024
        tier = "tier_1"

        # Check custom factors
        with self._state_lock:
            if region_upper in self._custom_residual_mix:
                custom = self._custom_residual_mix[region_upper]
                factor = custom["factor"]
                source = custom.get("source", "custom")
                year = custom.get("year", 2024)
                tier = "tier_2"
            elif region_upper in RESIDUAL_MIX_FACTORS:
                factor = RESIDUAL_MIX_FACTORS[region_upper]
            else:
                factor = RESIDUAL_MIX_FACTORS["WORLD"]
                source = "world_fallback"
                tier = "tier_3"

        provenance_hash = self._record_provenance(
            "get_residual_mix_with_metadata",
            {
                "region": region_upper,
                "factor": str(factor),
                "source": source,
                "year": year,
            },
        )

        self._record_metric_residual_mix(source)

        return {
            "region": region_upper,
            "factor_kgco2e_kwh": factor,
            "source": source,
            "year": year,
            "data_quality_tier": tier,
            "provenance_hash": provenance_hash,
        }

    def list_residual_mix_factors(self) -> List[Dict[str, Any]]:
        """List all available residual mix factors (built-in and custom).

        Returns:
            List of dictionaries, each with region and
            factor_kgco2e_kwh, sorted by region.
        """
        with self._state_lock:
            self._lookup_count += 1

        results: List[Dict[str, Any]] = []

        # Built-in factors
        for region, factor in sorted(RESIDUAL_MIX_FACTORS.items()):
            results.append({
                "region": region,
                "factor_kgco2e_kwh": factor,
                "source": "builtin",
            })

        # Custom overrides (append or update)
        with self._state_lock:
            for region, data in sorted(self._custom_residual_mix.items()):
                # Check if it overrides a built-in
                existing = next(
                    (r for r in results if r["region"] == region), None,
                )
                if existing:
                    existing["factor_kgco2e_kwh"] = data["factor"]
                    existing["source"] = data.get("source", "custom")
                else:
                    results.append({
                        "region": region,
                        "factor_kgco2e_kwh": data["factor"],
                        "source": data.get("source", "custom"),
                    })

        provenance_hash = self._record_provenance(
            "list_residual_mix_factors",
            {"count": len(results)},
        )
        for r in results:
            r["provenance_hash"] = provenance_hash

        return results

    # ==================================================================
    # PUBLIC METHODS: Custom Residual Mix Management
    # ==================================================================

    def set_custom_residual_mix(
        self,
        region: str,
        factor: Decimal,
        source: str = "custom",
        year: int = 2024,
    ) -> str:
        """Set a custom residual mix factor for a region.

        Overrides the built-in factor for subsequent lookups.

        Args:
            region: Region code (ISO alpha-2 or custom identifier).
            factor: Emission factor in kgCO2e/kWh. Must be >= 0.
            source: Data source description.
            year: Reference year for the factor.

        Returns:
            Provenance hash of the mutation.

        Raises:
            ValueError: If factor is negative.
        """
        if factor < Decimal("0"):
            raise ValueError(
                f"Emission factor must be >= 0, got {factor}"
            )

        region_upper = region.upper()
        entry = {
            "factor": factor.quantize(Decimal("0.001"), ROUND_HALF_UP),
            "source": source,
            "year": year,
            "created_at": _utcnow().isoformat(),
        }

        with self._state_lock:
            self._custom_residual_mix[region_upper] = entry
            self._mutation_count += 1

        provenance_hash = self._record_provenance(
            "set_custom_residual_mix",
            {
                "region": region_upper,
                "factor": str(entry["factor"]),
                "source": source,
                "year": year,
            },
        )

        self._record_metric_instrument("custom_residual_mix")

        logger.info(
            "Set custom residual mix: region=%s, factor=%s kgCO2e/kWh, "
            "source=%s, year=%d",
            region_upper,
            entry["factor"],
            source,
            year,
        )
        return provenance_hash

    def remove_custom_residual_mix(
        self,
        region: str,
    ) -> bool:
        """Remove a custom residual mix factor, reverting to built-in.

        Args:
            region: Region code to remove.

        Returns:
            True if the custom factor was removed, False if it did not exist.
        """
        region_upper = region.upper()

        with self._state_lock:
            if region_upper in self._custom_residual_mix:
                del self._custom_residual_mix[region_upper]
                self._mutation_count += 1
                self._record_provenance(
                    "remove_custom_residual_mix",
                    {"region": region_upper, "removed": True},
                )
                logger.info(
                    "Removed custom residual mix for region=%s",
                    region_upper,
                )
                return True

        self._record_provenance(
            "remove_custom_residual_mix",
            {"region": region_upper, "removed": False},
        )
        return False

    def list_custom_residual_mix(self) -> List[Dict[str, Any]]:
        """List all custom residual mix factors currently registered.

        Returns:
            List of dictionaries with region, factor, source, year.
        """
        with self._state_lock:
            results = []
            for region, data in sorted(self._custom_residual_mix.items()):
                results.append({
                    "region": region,
                    "factor_kgco2e_kwh": data["factor"],
                    "source": data.get("source", "custom"),
                    "year": data.get("year", 2024),
                    "created_at": data.get("created_at", ""),
                })
        return results

    # ==================================================================
    # PUBLIC METHODS: Energy Source Emission Factors
    # ==================================================================

    def get_energy_source_ef(
        self,
        source: str,
    ) -> Decimal:
        """Get the emission factor for a specific energy source.

        Args:
            source: Energy source identifier (solar, wind, hydro,
                nuclear, biomass, geothermal, natural_gas_ccgt,
                natural_gas_ocgt, coal, oil, mixed).

        Returns:
            Emission factor in kgCO2e/kWh as Decimal.

        Raises:
            ValueError: If source is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = source.lower().strip()
        if key not in ENERGY_SOURCE_EF:
            raise ValueError(
                f"Unknown energy source: {source!r}. "
                f"Valid sources: {sorted(ENERGY_SOURCE_EF.keys())}"
            )

        factor = ENERGY_SOURCE_EF[key]
        self._record_provenance(
            "get_energy_source_ef",
            {"source": key, "factor": str(factor)},
        )
        return factor

    def is_renewable_source(
        self,
        source: str,
    ) -> bool:
        """Check whether an energy source is classified as renewable/zero.

        Args:
            source: Energy source identifier.

        Returns:
            True if the source has zero combustion emissions.
        """
        return source.lower().strip() in _ZERO_EMISSION_SOURCES

    def get_zero_emission_sources(self) -> List[str]:
        """Get a sorted list of all zero-emission energy sources.

        Returns:
            Sorted list of energy source identifiers with zero EF.
        """
        return sorted(_ZERO_EMISSION_SOURCES)

    # ==================================================================
    # PUBLIC METHODS: Supplier Emission Factors
    # ==================================================================

    def get_supplier_ef(
        self,
        country_or_supplier: str,
    ) -> Decimal:
        """Get the supplier default emission factor.

        Checks custom supplier factors first (by supplier_id), then
        built-in country defaults, then falls back to WORLD residual mix.

        Args:
            country_or_supplier: ISO country code or custom supplier ID.

        Returns:
            Emission factor in kgCO2e/kWh as Decimal.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = country_or_supplier.upper().strip()

        # Check custom supplier factors first
        with self._state_lock:
            if key in self._supplier_factors:
                factor = self._supplier_factors[key]["ef_kgco2e_kwh"]
                self._record_provenance(
                    "get_supplier_ef",
                    {"key": key, "factor": str(factor), "source": "custom_supplier"},
                )
                return factor

        # Built-in country defaults
        if key in SUPPLIER_DEFAULT_EF:
            factor = SUPPLIER_DEFAULT_EF[key]
            self._record_provenance(
                "get_supplier_ef",
                {"key": key, "factor": str(factor), "source": "country_default"},
            )
            return factor

        # Check residual mix as fallback
        if key in RESIDUAL_MIX_FACTORS:
            factor = RESIDUAL_MIX_FACTORS[key]
            self._record_provenance(
                "get_supplier_ef",
                {"key": key, "factor": str(factor), "source": "residual_mix_fallback"},
            )
            logger.warning(
                "No supplier EF for %s, using residual mix factor",
                key,
            )
            return factor

        # Ultimate fallback: world average
        factor = RESIDUAL_MIX_FACTORS["WORLD"]
        self._record_provenance(
            "get_supplier_ef",
            {"key": key, "factor": str(factor), "source": "world_fallback"},
        )
        logger.warning(
            "No supplier EF for %s, using WORLD residual mix average",
            key,
        )
        return factor

    def set_supplier_factor(
        self,
        supplier_id: str,
        ef: Decimal,
        country: str = "",
        fuel_mix: Optional[Dict[str, Decimal]] = None,
        year: int = 2024,
    ) -> str:
        """Register a supplier-specific emission factor.

        Args:
            supplier_id: Unique supplier identifier.
            ef: Supplier emission factor in kgCO2e/kWh. Must be >= 0.
            country: ISO country code where the supplier operates.
            fuel_mix: Optional dictionary of fuel source percentages
                (e.g. {"solar": Decimal("0.30"), "wind": Decimal("0.20"),
                "natural_gas_ccgt": Decimal("0.50")}).
            year: Reference year for the factor.

        Returns:
            Provenance hash of the registration.

        Raises:
            ValueError: If ef is negative or fuel_mix percentages
                do not sum to approximately 1.0.
        """
        if ef < Decimal("0"):
            raise ValueError(
                f"Supplier emission factor must be >= 0, got {ef}"
            )

        if fuel_mix is not None:
            total_mix = sum(fuel_mix.values())
            if abs(total_mix - Decimal("1.0")) > Decimal("0.01"):
                raise ValueError(
                    f"Fuel mix percentages must sum to ~1.0, got {total_mix}"
                )

        sid = supplier_id.upper().strip()
        entry: Dict[str, Any] = {
            "supplier_id": sid,
            "ef_kgco2e_kwh": ef.quantize(Decimal("0.001"), ROUND_HALF_UP),
            "country": country.upper().strip(),
            "fuel_mix": {k: str(v) for k, v in fuel_mix.items()} if fuel_mix else {},
            "year": year,
            "created_at": _utcnow().isoformat(),
        }

        with self._state_lock:
            self._supplier_factors[sid] = entry
            self._mutation_count += 1

        provenance_hash = self._record_provenance(
            "set_supplier_factor",
            {
                "supplier_id": sid,
                "ef": str(entry["ef_kgco2e_kwh"]),
                "country": entry["country"],
                "year": year,
            },
        )

        self._record_metric_instrument("supplier_specific")

        logger.info(
            "Registered supplier factor: supplier=%s, ef=%s kgCO2e/kWh, "
            "country=%s, year=%d",
            sid,
            entry["ef_kgco2e_kwh"],
            entry["country"],
            year,
        )
        return provenance_hash

    def get_supplier_factor(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Get the full registration record for a supplier-specific factor.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Dictionary with supplier_id, ef_kgco2e_kwh, country,
            fuel_mix, year, created_at, and provenance_hash.

        Raises:
            ValueError: If supplier_id is not registered.
        """
        with self._state_lock:
            self._lookup_count += 1

        sid = supplier_id.upper().strip()

        with self._state_lock:
            if sid not in self._supplier_factors:
                raise ValueError(
                    f"No supplier factor registered for: {supplier_id!r}. "
                    f"Registered suppliers: "
                    f"{sorted(self._supplier_factors.keys())}"
                )
            result = dict(self._supplier_factors[sid])

        provenance_hash = self._record_provenance(
            "get_supplier_factor",
            {"supplier_id": sid, "ef": str(result["ef_kgco2e_kwh"])},
        )
        result["provenance_hash"] = provenance_hash
        return result

    def list_supplier_factors(self) -> List[Dict[str, Any]]:
        """List all registered supplier-specific emission factors.

        Returns:
            List of supplier factor dictionaries, sorted by supplier_id.
        """
        with self._state_lock:
            results = []
            for sid, data in sorted(self._supplier_factors.items()):
                entry = dict(data)
                results.append(entry)
        return results

    # ==================================================================
    # PUBLIC METHODS: Quality Criteria
    # ==================================================================

    def get_quality_criteria(self) -> List[Dict[str, Any]]:
        """Get all 7 instrument quality criteria with descriptions and weights.

        Returns:
            List of quality criteria dictionaries, each with criterion_id,
            name, description, weight, and scoring type.
        """
        results: List[Dict[str, Any]] = []
        for criterion_id, data in QUALITY_CRITERIA.items():
            entry = dict(data)
            entry["criterion_id"] = criterion_id
            results.append(entry)

        provenance_hash = self._record_provenance(
            "get_quality_criteria",
            {"count": len(results)},
        )
        for r in results:
            r["provenance_hash"] = provenance_hash

        return results

    def validate_instrument_quality(
        self,
        instrument: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a contractual instrument against 7 quality criteria.

        Evaluates the instrument record against each GHG Protocol Scope 2
        quality criterion and produces a weighted overall quality score.

        Args:
            instrument: Dictionary containing instrument metadata with
                keys: instrument_type, tracking_system, region,
                vintage_year, reporting_year, retirement_id,
                is_additional (optional).

        Returns:
            Dictionary with overall_score (Decimal 0-1), status ('PASS'
            or 'FAIL'), criteria_results (list of per-criterion results),
            and provenance_hash.
        """
        with self._state_lock:
            self._validation_count += 1

        instrument_type = instrument.get("instrument_type", "").upper()
        tracking_system = instrument.get("tracking_system", "")
        region = instrument.get("region", "").upper()
        vintage_year = instrument.get("vintage_year", 0)
        reporting_year = instrument.get("reporting_year", _utcnow().year)
        retirement_id = instrument.get("retirement_id", "")
        is_additional = instrument.get("is_additional", False)

        criteria_results: List[Dict[str, Any]] = []
        total_score = Decimal("0")
        total_weight = Decimal("0")

        # Criterion 1: Conveyance of Attributes
        conveyance_pass = instrument_type in INSTRUMENT_TYPES
        conveyance_score = Decimal("1.0") if conveyance_pass else Decimal("0.0")
        weight = QUALITY_CRITERIA["conveyance"]["weight"]
        total_score += conveyance_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "conveyance",
            "name": QUALITY_CRITERIA["conveyance"]["name"],
            "score": conveyance_score,
            "weight": weight,
            "passed": conveyance_pass,
            "detail": (
                f"Instrument type {instrument_type!r} is a recognised "
                "contractual instrument type"
                if conveyance_pass
                else f"Instrument type {instrument_type!r} is not recognised"
            ),
        })

        # Criterion 2: Unique Claim
        unique_pass = bool(retirement_id and len(retirement_id) >= 4)
        unique_score = Decimal("1.0") if unique_pass else Decimal("0.0")
        weight = QUALITY_CRITERIA["unique_claim"]["weight"]
        total_score += unique_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "unique_claim",
            "name": QUALITY_CRITERIA["unique_claim"]["name"],
            "score": unique_score,
            "weight": weight,
            "passed": unique_pass,
            "detail": (
                f"Retirement ID {retirement_id!r} provides unique claim"
                if unique_pass
                else "No valid retirement ID provided for unique claim"
            ),
        })

        # Criterion 3: Geographic / Market Boundary Match
        geo_score = self._evaluate_geographic_match(
            instrument_type, region,
        )
        weight = QUALITY_CRITERIA["geographic_match"]["weight"]
        total_score += geo_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "geographic_match",
            "name": QUALITY_CRITERIA["geographic_match"]["name"],
            "score": geo_score,
            "weight": weight,
            "passed": geo_score >= Decimal("0.5"),
            "detail": (
                f"Geographic match score: {geo_score} for region {region}"
            ),
        })

        # Criterion 4: Temporal / Vintage Match
        temporal_score = self._evaluate_temporal_match(
            instrument_type, vintage_year, reporting_year,
        )
        weight = QUALITY_CRITERIA["temporal_match"]["weight"]
        total_score += temporal_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "temporal_match",
            "name": QUALITY_CRITERIA["temporal_match"]["name"],
            "score": temporal_score,
            "weight": weight,
            "passed": temporal_score >= Decimal("0.5"),
            "detail": (
                f"Vintage {vintage_year} for reporting year "
                f"{reporting_year}: score {temporal_score}"
            ),
        })

        # Criterion 5: Verified Tracking System
        tracking_pass = self._evaluate_tracking_system(
            tracking_system, instrument_type,
        )
        tracking_score = Decimal("1.0") if tracking_pass else Decimal("0.0")
        weight = QUALITY_CRITERIA["tracking_verified"]["weight"]
        total_score += tracking_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "tracking_verified",
            "name": QUALITY_CRITERIA["tracking_verified"]["name"],
            "score": tracking_score,
            "weight": weight,
            "passed": tracking_pass,
            "detail": (
                f"Tracking system {tracking_system!r} is verified"
                if tracking_pass
                else f"Tracking system {tracking_system!r} is not recognised"
            ),
        })

        # Criterion 6: Retirement / Cancellation Evidence
        retirement_pass = bool(retirement_id and len(retirement_id) >= 4)
        retirement_score = Decimal("1.0") if retirement_pass else Decimal("0.0")
        weight = QUALITY_CRITERIA["retirement_evidence"]["weight"]
        total_score += retirement_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "retirement_evidence",
            "name": QUALITY_CRITERIA["retirement_evidence"]["name"],
            "score": retirement_score,
            "weight": weight,
            "passed": retirement_pass,
            "detail": (
                "Retirement evidence provided"
                if retirement_pass
                else "No retirement evidence provided"
            ),
        })

        # Criterion 7: Additionality
        additionality_score = Decimal("1.0") if is_additional else Decimal("0.0")
        weight = QUALITY_CRITERIA["additionality"]["weight"]
        total_score += additionality_score * weight
        total_weight += weight
        criteria_results.append({
            "criterion_id": "additionality",
            "name": QUALITY_CRITERIA["additionality"]["name"],
            "score": additionality_score,
            "weight": weight,
            "passed": bool(is_additional),
            "detail": (
                "Instrument supports additionality"
                if is_additional
                else "Additionality not demonstrated (optional)"
            ),
        })

        # Calculate overall weighted score
        if total_weight > Decimal("0"):
            overall_score = (total_score / total_weight).quantize(
                Decimal("0.001"), ROUND_HALF_UP,
            )
        else:
            overall_score = Decimal("0.000")

        # Normalise: total_score already accumulated as score*weight
        overall_score = total_score.quantize(Decimal("0.001"), ROUND_HALF_UP)

        # PASS if score >= 0.60 (60% threshold)
        status = "PASS" if overall_score >= Decimal("0.600") else "FAIL"

        provenance_hash = self._record_provenance(
            "validate_instrument_quality",
            {
                "instrument_type": instrument_type,
                "overall_score": str(overall_score),
                "status": status,
                "criteria_count": len(criteria_results),
            },
        )

        logger.info(
            "Instrument quality validation: type=%s, score=%s, status=%s",
            instrument_type,
            overall_score,
            status,
        )

        return {
            "overall_score": overall_score,
            "status": status,
            "criteria_results": criteria_results,
            "instrument_type": instrument_type,
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # Private quality evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_geographic_match(
        self,
        instrument_type: str,
        region: str,
    ) -> Decimal:
        """Evaluate geographic match score for an instrument.

        Args:
            instrument_type: Instrument type code.
            region: Region code of the consumer's load.

        Returns:
            Score between 0.0 and 1.0.
        """
        if instrument_type not in INSTRUMENT_TYPES:
            return Decimal("0.0")

        info = INSTRUMENT_TYPES[instrument_type]
        applicability: List[str] = info["region_applicability"]

        # Exact match or GLOBAL applicability
        if region in applicability or "GLOBAL" in applicability:
            return Decimal("1.0")

        # Partial match: region starts with a known applicable region
        # e.g. region='US-CAMX' matches applicability=['US']
        for app_region in applicability:
            if region.startswith(app_region):
                return Decimal("0.8")

        # EU/EEA umbrella match
        eu_countries = {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE",
        }
        if "EU" in applicability and region in eu_countries:
            return Decimal("0.9")
        if "EEA" in applicability and (region in eu_countries or region in {"NO", "IS", "LI"}):
            return Decimal("0.9")

        return Decimal("0.0")

    def _evaluate_temporal_match(
        self,
        instrument_type: str,
        vintage_year: int,
        reporting_year: int,
    ) -> Decimal:
        """Evaluate temporal match score for an instrument vintage.

        Args:
            instrument_type: Instrument type code.
            vintage_year: Year the energy was generated.
            reporting_year: Year being reported.

        Returns:
            Score between 0.0 and 1.0.
        """
        if vintage_year <= 0 or reporting_year <= 0:
            return Decimal("0.0")

        age = reporting_year - vintage_year
        if age < 0:
            return Decimal("0.0")

        # Same year: perfect match
        if age == 0:
            return Decimal("1.0")

        # Within 1 year: excellent
        if age == 1:
            return Decimal("0.8")

        # Get max vintage for the type
        max_years = INSTRUMENT_TYPES.get(
            instrument_type, {},
        ).get("vintage_max_years", 2)

        # Within allowable period
        if age <= max_years:
            # Linear decay from 0.7 to 0.3 within the period
            decay = Decimal(str(0.7 - (0.4 * (age - 1) / max(max_years - 1, 1))))
            return max(decay.quantize(Decimal("0.01"), ROUND_HALF_UP), Decimal("0.0"))

        # Beyond max vintage: invalid
        return Decimal("0.0")

    def _evaluate_tracking_system(
        self,
        tracking_system: str,
        instrument_type: str,
    ) -> bool:
        """Evaluate whether the tracking system is verified for the instrument.

        Args:
            tracking_system: Tracking system identifier.
            instrument_type: Instrument type code.

        Returns:
            True if the tracking system is recognised and applicable.
        """
        if not tracking_system:
            return False

        ts_upper = tracking_system.upper().strip()

        # Check if the tracking system exists in our registry
        if ts_upper not in TRACKING_SYSTEMS:
            return False

        # Check if the instrument type has this tracking system
        if instrument_type in INSTRUMENT_TYPES:
            allowed_systems = INSTRUMENT_TYPES[instrument_type]["tracking_systems"]
            if ts_upper in allowed_systems:
                return True

        # Also verify the tracking system supports this instrument type
        ts_info = TRACKING_SYSTEMS[ts_upper]
        if instrument_type in ts_info.get("instrument_types", []):
            return True

        return False

    # ==================================================================
    # PUBLIC METHODS: Tracking Systems
    # ==================================================================

    def get_tracking_system_info(
        self,
        system: str,
    ) -> Dict[str, Any]:
        """Get metadata for a certificate tracking system/registry.

        Args:
            system: Tracking system identifier (e.g. 'WREGIS', 'AIB',
                'I-REC', 'OFGEM-REGO').

        Returns:
            Dictionary with name, region, url, instrument_types,
            description, and provenance_hash.

        Raises:
            ValueError: If the tracking system is not recognised.
        """
        with self._state_lock:
            self._lookup_count += 1

        key = system.upper().strip()
        if key not in TRACKING_SYSTEMS:
            raise ValueError(
                f"Unknown tracking system: {system!r}. "
                f"Valid systems: {sorted(TRACKING_SYSTEMS.keys())}"
            )

        info = dict(TRACKING_SYSTEMS[key])
        info["system_id"] = key

        provenance_hash = self._record_provenance(
            "get_tracking_system_info",
            {"system": key},
        )
        info["provenance_hash"] = provenance_hash
        return info

    # ==================================================================
    # PUBLIC METHODS: Emission Factor Resolution
    # ==================================================================

    def resolve_emission_factor(
        self,
        instrument: Dict[str, Any],
    ) -> Decimal:
        """Resolve the emission factor for a contractual instrument.

        Uses the following priority hierarchy:
        1. Custom EF: instrument["custom_ef"] if provided
        2. Source EF: energy source emission factor if instrument
           specifies an energy_source
        3. Supplier EF: supplier-specific factor if instrument
           specifies a supplier_id
        4. Residual Mix: region-based residual mix factor as fallback

        Args:
            instrument: Dictionary with instrument metadata. May include:
                custom_ef (Decimal), energy_source (str),
                supplier_id (str), region (str).

        Returns:
            Resolved emission factor in kgCO2e/kWh as Decimal.
        """
        with self._state_lock:
            self._lookup_count += 1

        # Priority 1: Custom EF
        custom_ef = instrument.get("custom_ef")
        if custom_ef is not None:
            ef = Decimal(str(custom_ef)).quantize(
                Decimal("0.001"), ROUND_HALF_UP,
            )
            self._record_provenance(
                "resolve_emission_factor",
                {"source": "custom_ef", "factor": str(ef)},
            )
            logger.debug("Resolved EF via custom_ef: %s", ef)
            return ef

        # Priority 2: Energy source EF
        energy_source = instrument.get("energy_source")
        if energy_source:
            key = energy_source.lower().strip()
            if key in ENERGY_SOURCE_EF:
                ef = ENERGY_SOURCE_EF[key]
                self._record_provenance(
                    "resolve_emission_factor",
                    {"source": "energy_source", "energy_source": key, "factor": str(ef)},
                )
                logger.debug(
                    "Resolved EF via energy source %s: %s",
                    key,
                    ef,
                )
                return ef

        # Priority 3: Supplier-specific EF
        supplier_id = instrument.get("supplier_id")
        if supplier_id:
            sid = supplier_id.upper().strip()
            with self._state_lock:
                if sid in self._supplier_factors:
                    ef = self._supplier_factors[sid]["ef_kgco2e_kwh"]
                    self._record_provenance(
                        "resolve_emission_factor",
                        {"source": "supplier", "supplier_id": sid, "factor": str(ef)},
                    )
                    logger.debug(
                        "Resolved EF via supplier %s: %s",
                        sid,
                        ef,
                    )
                    return ef

            # Fall back to country-level supplier default
            country = instrument.get("region", "").upper().strip()
            if country and country in SUPPLIER_DEFAULT_EF:
                ef = SUPPLIER_DEFAULT_EF[country]
                self._record_provenance(
                    "resolve_emission_factor",
                    {"source": "supplier_country_default", "country": country, "factor": str(ef)},
                )
                logger.debug(
                    "Resolved EF via supplier country default %s: %s",
                    country,
                    ef,
                )
                return ef

        # Priority 4: Residual mix
        region = instrument.get("region", "WORLD").upper().strip()
        ef = self.get_residual_mix_factor(region)
        self._record_provenance(
            "resolve_emission_factor",
            {"source": "residual_mix", "region": region, "factor": str(ef)},
        )
        logger.debug(
            "Resolved EF via residual mix for region %s: %s",
            region,
            ef,
        )
        return ef

    # ==================================================================
    # PUBLIC METHODS: Search and Query
    # ==================================================================

    def search_factors(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search residual mix and supplier factors by text query.

        Matches the query against region codes and supplier IDs
        (case-insensitive substring match).

        Args:
            query: Search string.

        Returns:
            List of matching factor entries with type, key, and factor.
        """
        with self._state_lock:
            self._lookup_count += 1

        q = query.upper().strip()
        results: List[Dict[str, Any]] = []

        # Search residual mix factors
        for region, factor in RESIDUAL_MIX_FACTORS.items():
            if q in region:
                results.append({
                    "type": "residual_mix",
                    "key": region,
                    "factor_kgco2e_kwh": factor,
                    "source": "builtin",
                })

        # Search custom residual mix
        with self._state_lock:
            for region, data in self._custom_residual_mix.items():
                if q in region:
                    results.append({
                        "type": "custom_residual_mix",
                        "key": region,
                        "factor_kgco2e_kwh": data["factor"],
                        "source": data.get("source", "custom"),
                    })

        # Search supplier defaults
        for country, factor in SUPPLIER_DEFAULT_EF.items():
            if q in country:
                results.append({
                    "type": "supplier_default",
                    "key": country,
                    "factor_kgco2e_kwh": factor,
                    "source": "builtin",
                })

        # Search custom suppliers
        with self._state_lock:
            for sid, data in self._supplier_factors.items():
                if q in sid or q in data.get("country", ""):
                    results.append({
                        "type": "custom_supplier",
                        "key": sid,
                        "factor_kgco2e_kwh": data["ef_kgco2e_kwh"],
                        "source": "custom",
                        "country": data.get("country", ""),
                    })

        # Search energy sources
        for source, factor in ENERGY_SOURCE_EF.items():
            if q.lower() in source:
                results.append({
                    "type": "energy_source",
                    "key": source,
                    "factor_kgco2e_kwh": factor,
                    "source": "builtin",
                })

        provenance_hash = self._record_provenance(
            "search_factors",
            {"query": q, "result_count": len(results)},
        )
        for r in results:
            r["provenance_hash"] = provenance_hash

        logger.debug(
            "Factor search: query=%s, results=%d",
            q,
            len(results),
        )
        return results

    # ==================================================================
    # PUBLIC METHODS: Statistics and Engine Info
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics including counts and operational metrics.

        Returns:
            Dictionary with instrument_types, residual_mix_regions,
            custom_residual_mix, energy_sources, supplier_defaults,
            custom_suppliers, quality_criteria, tracking_systems,
            total_lookups, total_mutations, total_validations,
            provenance_hashes_count.
        """
        with self._state_lock:
            stats = {
                "instrument_types": len(INSTRUMENT_TYPES),
                "residual_mix_regions": len(RESIDUAL_MIX_FACTORS),
                "custom_residual_mix": len(self._custom_residual_mix),
                "energy_sources": len(ENERGY_SOURCE_EF),
                "zero_emission_sources": len(_ZERO_EMISSION_SOURCES),
                "supplier_defaults": len(SUPPLIER_DEFAULT_EF),
                "custom_suppliers": len(self._supplier_factors),
                "quality_criteria": len(QUALITY_CRITERIA),
                "tracking_systems": len(TRACKING_SYSTEMS),
                "total_lookups": self._lookup_count,
                "total_mutations": self._mutation_count,
                "total_validations": self._validation_count,
                "provenance_hashes_count": len(self._provenance_hashes),
            }

        provenance_hash = self._record_provenance(
            "get_statistics",
            stats,
        )
        stats["provenance_hash"] = provenance_hash
        return stats

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine identification and version metadata.

        Returns:
            Dictionary with engine_id, version, description, data_sources,
            capabilities, and provenance_hash.
        """
        info = {
            "engine_id": self.ENGINE_ID,
            "version": self.ENGINE_VERSION,
            "description": (
                "Contractual Instrument Database Engine for Scope 2 "
                "market-based emission calculations per GHG Protocol "
                "Scope 2 Guidance (2015)."
            ),
            "data_sources": [
                "GHG Protocol Scope 2 Guidance (2015)",
                "AIB European Residual Mixes (2024)",
                "Green-e US Residual Mix",
                "IEA CO2 Emissions from Fuel Combustion",
                "IPCC AR6 Lifecycle Emission Factors",
            ],
            "capabilities": [
                "instrument_type_lookup",
                "residual_mix_factor_lookup",
                "energy_source_ef_lookup",
                "supplier_ef_lookup",
                "custom_residual_mix_management",
                "custom_supplier_management",
                "instrument_quality_validation",
                "tracking_system_lookup",
                "emission_factor_resolution",
                "factor_search",
                "vintage_validation",
            ],
            "built_in_data": {
                "instrument_types": len(INSTRUMENT_TYPES),
                "residual_mix_regions": len(RESIDUAL_MIX_FACTORS),
                "energy_sources": len(ENERGY_SOURCE_EF),
                "supplier_defaults": len(SUPPLIER_DEFAULT_EF),
                "quality_criteria": len(QUALITY_CRITERIA),
                "tracking_systems": len(TRACKING_SYSTEMS),
            },
        }

        provenance_hash = self._record_provenance(
            "get_engine_info",
            {"engine_id": self.ENGINE_ID, "version": self.ENGINE_VERSION},
        )
        info["provenance_hash"] = provenance_hash
        return info

    # ==================================================================
    # PUBLIC METHODS: Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset all mutable state to initial values.

        Clears custom residual mix factors, custom supplier factors,
        operational counters, and provenance hashes. Built-in data
        is not affected.

        Primarily used for testing and development.
        """
        with self._state_lock:
            self._custom_residual_mix.clear()
            self._supplier_factors.clear()
            self._lookup_count = 0
            self._mutation_count = 0
            self._validation_count = 0
            self._provenance_hashes.clear()

        logger.info(
            "ContractualInstrumentDatabaseEngine reset: "
            "all mutable state cleared"
        )

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance for testing.

        After calling this method, the next instantiation will create
        a fresh instance. This is intended for test isolation only.
        """
        with cls._lock:
            cls._instance = None
        logger.info("ContractualInstrumentDatabaseEngine singleton reset")


# ===========================================================================
# Module-level convenience functions
# ===========================================================================

_module_engine: Optional[ContractualInstrumentDatabaseEngine] = None
_module_lock = threading.Lock()


def get_engine(
    config: Any = None,
    metrics: Any = None,
    provenance: Any = None,
) -> ContractualInstrumentDatabaseEngine:
    """Get or create the module-level engine singleton.

    Args:
        config: Optional configuration object.
        metrics: Optional metrics recorder.
        provenance: Optional provenance tracker.

    Returns:
        The ContractualInstrumentDatabaseEngine singleton instance.
    """
    global _module_engine
    if _module_engine is None:
        with _module_lock:
            if _module_engine is None:
                _module_engine = ContractualInstrumentDatabaseEngine(
                    config=config,
                    metrics=metrics,
                    provenance=provenance,
                )
    return _module_engine


def get_instrument_info(instrument_type: str) -> Dict[str, Any]:
    """Module-level convenience: get instrument type metadata.

    Args:
        instrument_type: Instrument type code.

    Returns:
        Instrument metadata dictionary.
    """
    return get_engine().get_instrument_info(instrument_type)


def get_residual_mix_factor(region: str) -> Decimal:
    """Module-level convenience: get residual mix factor.

    Args:
        region: Region code.

    Returns:
        Emission factor in kgCO2e/kWh.
    """
    return get_engine().get_residual_mix_factor(region)


def get_energy_source_ef(source: str) -> Decimal:
    """Module-level convenience: get energy source emission factor.

    Args:
        source: Energy source identifier.

    Returns:
        Emission factor in kgCO2e/kWh.
    """
    return get_engine().get_energy_source_ef(source)


def get_supplier_ef(country_or_supplier: str) -> Decimal:
    """Module-level convenience: get supplier default emission factor.

    Args:
        country_or_supplier: ISO country code or supplier ID.

    Returns:
        Emission factor in kgCO2e/kWh.
    """
    return get_engine().get_supplier_ef(country_or_supplier)


def resolve_emission_factor(instrument: Dict[str, Any]) -> Decimal:
    """Module-level convenience: resolve emission factor for an instrument.

    Args:
        instrument: Instrument metadata dictionary.

    Returns:
        Resolved emission factor in kgCO2e/kWh.
    """
    return get_engine().resolve_emission_factor(instrument)


def validate_instrument_quality(instrument: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level convenience: validate instrument quality.

    Args:
        instrument: Instrument metadata dictionary.

    Returns:
        Quality validation result dictionary.
    """
    return get_engine().validate_instrument_quality(instrument)


def search_factors(query: str) -> List[Dict[str, Any]]:
    """Module-level convenience: search all factor databases.

    Args:
        query: Search string.

    Returns:
        List of matching factor entries.
    """
    return get_engine().search_factors(query)


def get_statistics() -> Dict[str, Any]:
    """Module-level convenience: get database statistics.

    Returns:
        Statistics dictionary.
    """
    return get_engine().get_statistics()
