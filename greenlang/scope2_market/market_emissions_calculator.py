# -*- coding: utf-8 -*-
"""
MarketEmissionsCalculatorEngine - Engine 3: Scope 2 Market-Based Emissions Agent (AGENT-MRV-010)

Core Scope 2 market-based emission calculations using contractual instruments
(RECs, GOs, PPAs, supplier-specific EFs) and residual mix factors per the
GHG Protocol Scope 2 Guidance market-based method.

Formulas:
    Covered emissions   = instrument_mwh x instrument_ef (kgCO2e/kWh) x 1000
    Uncovered emissions = uncovered_mwh x residual_mix_ef (kgCO2e/kWh) x 1000
    Total market-based  = covered_emissions + uncovered_emissions

Per-gas breakdown:
    CO2 component = total x co2_fraction
    CH4 component = total x ch4_fraction / GWP_CH4 * GWP_CH4 (identity, for tracking)
    N2O component = total x n2o_fraction / GWP_N2O * GWP_N2O (identity, for tracking)

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places internal)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic: same input -> same output (bit-perfect)

Example:
    >>> from greenlang.scope2_market.market_emissions_calculator import MarketEmissionsCalculatorEngine
    >>> from decimal import Decimal
    >>> engine = MarketEmissionsCalculatorEngine()
    >>> result = engine.calculate_covered_emissions(
    ...     instrument_type="REC",
    ...     mwh=Decimal("500"),
    ...     ef_kgco2e_kwh=Decimal("0.000"),
    ... )
    >>> assert result["total_co2e_kg"] == Decimal("0")

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
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------
_PRECISION_INTERNAL = Decimal("0.00000001")  # 8 decimal places internal
_PRECISION_OUTPUT = Decimal("0.001")          # 3 decimal places output

_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")

# ---------------------------------------------------------------------------
# Unit Conversion Constants (all Decimal for zero-hallucination)
# ---------------------------------------------------------------------------

#: 1 kWh = 0.001 MWh
_KWH_TO_MWH = Decimal("0.001")

#: 1 GJ = 0.277778 MWh (1 / 3.6)
_GJ_TO_MWH = Decimal("0.277778")

#: 1 MMBTU = 0.293071 MWh
_MMBTU_TO_MWH = Decimal("0.293071")

#: 1 TJ = 277.778 MWh
_TJ_TO_MWH = Decimal("277.778")

#: kg to tonnes
_KG_TO_TONNES = Decimal("0.001")

#: tonnes to kg
_TONNES_TO_KG = Decimal("1000")

#: kgCO2e/kWh to kgCO2e/MWh
_KGCO2E_KWH_TO_MWH = Decimal("1000")


# ---------------------------------------------------------------------------
# GWP Values by IPCC Assessment Report
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}


# ---------------------------------------------------------------------------
# Default Gas Fractions for Residual Mix
# ---------------------------------------------------------------------------
# When no per-gas emission factors are available for the residual mix,
# the total CO2e is apportioned using these default fractions.
# Source: Typical fossil-heavy residual mix profile.

DEFAULT_GAS_FRACTIONS: Dict[str, Decimal] = {
    "co2": Decimal("0.95"),
    "ch4": Decimal("0.03"),
    "n2o": Decimal("0.02"),
}


# ---------------------------------------------------------------------------
# Renewable Energy Emission Factors (zero EF instruments)
# ---------------------------------------------------------------------------
# Contractual instruments backed by these generation sources carry a zero
# emission factor per GHG Protocol Scope 2 Guidance.

RENEWABLE_EF: Dict[str, Decimal] = {
    "solar": Decimal("0.000"),
    "wind": Decimal("0.000"),
    "hydro": Decimal("0.000"),
    "nuclear": Decimal("0.000"),
    "geothermal": Decimal("0.000"),
}


# ---------------------------------------------------------------------------
# Biogenic Sources
# ---------------------------------------------------------------------------
# Generation sources considered biogenic; emissions reported as memo items
# and excluded from Scope 2 totals per GHG Protocol.

BIOGENIC_SOURCES: Set[str] = {
    "biomass",
    "biogas",
    "landfill_gas",
    "wood_waste",
    "agricultural_waste",
    "municipal_solid_waste_biogenic",
    "sewage_gas",
}


# ---------------------------------------------------------------------------
# Valid Instrument Types
# ---------------------------------------------------------------------------

_VALID_INSTRUMENT_TYPES: Set[str] = {
    "REC",            # Renewable Energy Certificate (US)
    "GO",             # Guarantee of Origin (EU)
    "I-REC",          # International REC
    "REGO",           # Renewable Energy Guarantee of Origin (UK)
    "LGC",            # Large-scale Generation Certificate (AU)
    "GEC",            # Green Energy Certificate
    "TIGR",           # Tradable Instrument for Global Renewables
    "PPA",            # Power Purchase Agreement
    "GREEN_TARIFF",   # Utility green tariff / green pricing
    "VPPA",           # Virtual Power Purchase Agreement
    "DIRECT_LINE",    # Direct line / behind-the-meter
    "SUPPLIER_SPECIFIC",  # Supplier-specific emission factor
    "RESIDUAL_MIX",       # Residual mix (uncovered)
    "OTHER",              # Other contractual instrument
}


# ---------------------------------------------------------------------------
# Default Residual Mix Emission Factors by Region (kgCO2e/kWh)
# ---------------------------------------------------------------------------
# Sources: AIB European Residual Mixes 2023, EPA eGRID, IEA.

_DEFAULT_RESIDUAL_MIX_EF: Dict[str, Dict[str, Any]] = {
    "US": {
        "ef_kgco2e_kwh": Decimal("0.425"),
        "source": "EPA_eGRID_2022_RESIDUAL",
    },
    "US_NEWE": {
        "ef_kgco2e_kwh": Decimal("0.310"),
        "source": "EPA_eGRID_2022_NEWE_RESIDUAL",
    },
    "US_CAMX": {
        "ef_kgco2e_kwh": Decimal("0.280"),
        "source": "EPA_eGRID_2022_CAMX_RESIDUAL",
    },
    "US_ERCT": {
        "ef_kgco2e_kwh": Decimal("0.410"),
        "source": "EPA_eGRID_2022_ERCT_RESIDUAL",
    },
    "US_RFCW": {
        "ef_kgco2e_kwh": Decimal("0.510"),
        "source": "EPA_eGRID_2022_RFCW_RESIDUAL",
    },
    "EU": {
        "ef_kgco2e_kwh": Decimal("0.420"),
        "source": "AIB_EUROPEAN_RESIDUAL_MIX_2023",
    },
    "DE": {
        "ef_kgco2e_kwh": Decimal("0.560"),
        "source": "AIB_2023_DE",
    },
    "FR": {
        "ef_kgco2e_kwh": Decimal("0.055"),
        "source": "AIB_2023_FR",
    },
    "GB": {
        "ef_kgco2e_kwh": Decimal("0.320"),
        "source": "DEFRA_2023_RESIDUAL",
    },
    "NL": {
        "ef_kgco2e_kwh": Decimal("0.480"),
        "source": "AIB_2023_NL",
    },
    "IT": {
        "ef_kgco2e_kwh": Decimal("0.390"),
        "source": "AIB_2023_IT",
    },
    "ES": {
        "ef_kgco2e_kwh": Decimal("0.270"),
        "source": "AIB_2023_ES",
    },
    "SE": {
        "ef_kgco2e_kwh": Decimal("0.035"),
        "source": "AIB_2023_SE",
    },
    "NO": {
        "ef_kgco2e_kwh": Decimal("0.420"),
        "source": "AIB_2023_NO_RESIDUAL",
    },
    "JP": {
        "ef_kgco2e_kwh": Decimal("0.470"),
        "source": "JAPAN_RESIDUAL_2023",
    },
    "AU": {
        "ef_kgco2e_kwh": Decimal("0.680"),
        "source": "AUS_RESIDUAL_2023",
    },
    "CN": {
        "ef_kgco2e_kwh": Decimal("0.580"),
        "source": "CHINA_GRID_RESIDUAL_2023",
    },
    "IN": {
        "ef_kgco2e_kwh": Decimal("0.720"),
        "source": "INDIA_CEA_RESIDUAL_2023",
    },
    "BR": {
        "ef_kgco2e_kwh": Decimal("0.090"),
        "source": "BRAZIL_SIN_RESIDUAL_2023",
    },
    "CA": {
        "ef_kgco2e_kwh": Decimal("0.140"),
        "source": "CANADA_NIR_RESIDUAL_2023",
    },
    "KR": {
        "ef_kgco2e_kwh": Decimal("0.450"),
        "source": "KOREA_RESIDUAL_2023",
    },
    "ZA": {
        "ef_kgco2e_kwh": Decimal("0.950"),
        "source": "SA_ESKOM_RESIDUAL_2023",
    },
    "GLOBAL": {
        "ef_kgco2e_kwh": Decimal("0.450"),
        "source": "IEA_GLOBAL_AVG_RESIDUAL_2023",
    },
}


# ---------------------------------------------------------------------------
# Prometheus metrics helpers (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

if _PROMETHEUS_AVAILABLE:
    _s2m_calculations_total = Counter(
        "gl_s2m_calculations_total",
        "Total Scope 2 market-based emission calculations",
        labelnames=["instrument_type", "method", "status"],
    )
    _s2m_emissions_kg_co2e_total = Counter(
        "gl_s2m_emissions_kg_co2e_total",
        "Cumulative Scope 2 market-based emissions in kg CO2e",
        labelnames=["instrument_type", "gas"],
    )
    _s2m_batch_jobs_total = Counter(
        "gl_s2m_batch_jobs_total",
        "Total Scope 2 market-based batch jobs",
        labelnames=["status"],
    )
    _s2m_calculation_duration = Histogram(
        "gl_s2m_calculation_duration_seconds",
        "Duration of Scope 2 market-based calculations in seconds",
        labelnames=["operation"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _s2m_active_calculations = Gauge(
        "gl_s2m_active_calculations",
        "Number of currently active Scope 2 market-based calculations",
    )
else:
    _s2m_calculations_total = None          # type: ignore[assignment]
    _s2m_emissions_kg_co2e_total = None     # type: ignore[assignment]
    _s2m_batch_jobs_total = None            # type: ignore[assignment]
    _s2m_calculation_duration = None        # type: ignore[assignment]
    _s2m_active_calculations = None         # type: ignore[assignment]


def _record_calculation(instrument_type: str, method: str, status: str) -> None:
    """Record a Scope 2 market-based calculation metric."""
    if _PROMETHEUS_AVAILABLE and _s2m_calculations_total is not None:
        _s2m_calculations_total.labels(
            instrument_type=instrument_type, method=method, status=status,
        ).inc()


def _record_emissions(instrument_type: str, gas: str, kg: float) -> None:
    """Record cumulative emissions metric."""
    if _PROMETHEUS_AVAILABLE and _s2m_emissions_kg_co2e_total is not None:
        _s2m_emissions_kg_co2e_total.labels(
            instrument_type=instrument_type, gas=gas,
        ).inc(kg)


def _record_batch_metric(status: str) -> None:
    """Record batch job metric."""
    if _PROMETHEUS_AVAILABLE and _s2m_batch_jobs_total is not None:
        _s2m_batch_jobs_total.labels(status=status).inc()


def _observe_duration(operation: str, seconds: float) -> None:
    """Record calculation duration metric."""
    if _PROMETHEUS_AVAILABLE and _s2m_calculation_duration is not None:
        _s2m_calculation_duration.labels(operation=operation).observe(seconds)


# ---------------------------------------------------------------------------
# Provenance helper (lightweight inline tracker)
# ---------------------------------------------------------------------------

class _ProvenanceTracker:
    """Lightweight chain-hashing provenance tracker for market-based emissions.

    Each recorded entry chains its SHA-256 hash to the previous entry,
    producing a tamper-evident audit log.
    """

    def __init__(
        self,
        genesis: str = "GL-MRV-010-SCOPE2-MARKET-EMISSIONS-GENESIS",
    ) -> None:
        """Initialize the provenance tracker.

        Args:
            genesis: Seed string for the genesis hash.
        """
        self._genesis: str = hashlib.sha256(genesis.encode("utf-8")).hexdigest()
        self._last_hash: str = self._genesis
        self._entries: List[Dict[str, Any]] = []
        self._lock: threading.RLock = threading.RLock()

    def record(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a provenance entry and return it.

        Args:
            entity_type: Category of the recorded entity.
            action: Action name (e.g. method name).
            entity_id: Unique identifier for this entry.
            data: Optional data payload for hashing.

        Returns:
            Dictionary with entry metadata and chain hash.
        """
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        data_hash = hashlib.sha256(
            json.dumps(data or {}, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        with self._lock:
            parent = self._last_hash
            payload = f"{parent}|{data_hash}|{action}|{ts}"
            chain_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            entry = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "hash_value": chain_hash,
                "parent_hash": parent,
                "timestamp": ts,
                "data_hash": data_hash,
            }
            self._entries.append(entry)
            self._last_hash = chain_hash
        return entry

    def verify_chain(self) -> bool:
        """Verify integrity of the entire provenance chain.

        Returns:
            True if chain is valid, False if tampered.
        """
        with self._lock:
            if not self._entries:
                return True
            expected_parent = self._genesis
            for entry in self._entries:
                if entry["parent_hash"] != expected_parent:
                    return False
                expected_parent = entry["hash_value"]
            return True

    def get_entries(self) -> List[Dict[str, Any]]:
        """Return a copy of all provenance entries.

        Returns:
            List of provenance entry dictionaries.
        """
        with self._lock:
            return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Return the number of provenance entries recorded."""
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# Utility: UTC now helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Utility: Safe Decimal conversion
# ---------------------------------------------------------------------------

def _to_decimal(value: Any) -> Decimal:
    """Convert a value to Decimal safely.

    Args:
        value: int, float, str, or Decimal.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal: {exc}") from exc


def _q(value: Decimal) -> Decimal:
    """Quantize a Decimal to 8 internal decimal places.

    Args:
        value: Decimal to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION_INTERNAL, rounding=ROUND_HALF_UP)


def _q_out(value: Decimal) -> Decimal:
    """Quantize a Decimal to 3 output decimal places.

    Args:
        value: Decimal to quantize.

    Returns:
        Quantized Decimal.
    """
    return value.quantize(_PRECISION_OUTPUT, rounding=ROUND_HALF_UP)


# ===========================================================================
# MarketEmissionsCalculatorEngine
# ===========================================================================


class MarketEmissionsCalculatorEngine:
    """Core Scope 2 market-based emission calculation engine.

    Implements the GHG Protocol Scope 2 Guidance market-based method using
    contractual instruments (RECs, GOs, PPAs, supplier-specific EFs) and
    residual mix factors. Supports covered/uncovered splits, per-gas
    breakdowns, facility-level and batch calculations, and multiple
    aggregation dimensions.

    Zero-Hallucination Guarantees:
        - All arithmetic uses ``Decimal`` with ``ROUND_HALF_UP``
        - 8 decimal places for internal precision, 3 for output
        - No LLM calls in the calculation path
        - SHA-256 provenance hash on every result
        - Complete calculation trace for audit
        - Thread-safe via ``threading.RLock``

    Attributes:
        _config: Configuration dictionary.
        _provenance: Chain-hashing provenance tracker.
        _lock: Reentrant thread lock for shared mutable state.

    Example:
        >>> engine = MarketEmissionsCalculatorEngine()
        >>> r = engine.calculate_covered_emissions("REC", Decimal("500"), Decimal("0.000"))
        >>> assert r["total_co2e_kg"] == Decimal("0.000")
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        residual_mix_db: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        provenance: Optional[Any] = None,
    ) -> None:
        """Initialize MarketEmissionsCalculatorEngine.

        Args:
            residual_mix_db: Optional external residual mix factor database.
                Falls back to built-in defaults if not provided.
            config: Optional configuration dictionary. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                    Default True.
                - ``default_gwp_source`` (str): Default GWP source.
                    Default ``AR5``.
                - ``default_region`` (str): Default region for residual mix
                    lookup. Default ``GLOBAL``.
            provenance: Optional external provenance tracker. If None, an
                internal ``_ProvenanceTracker`` is created.
        """
        self._residual_mix_db = residual_mix_db
        self._config: Dict[str, Any] = config or {}
        self._lock = threading.RLock()

        # Provenance
        self._enable_provenance: bool = self._config.get("enable_provenance", True)
        if provenance is not None:
            self._provenance = provenance
        elif self._enable_provenance:
            self._provenance = _ProvenanceTracker()
        else:
            self._provenance = None

        # Defaults
        self._default_gwp: str = self._config.get("default_gwp_source", "AR5")
        self._default_region: str = self._config.get("default_region", "GLOBAL")

        # Statistics counters
        self._stats_lock = threading.RLock()
        self._total_calculations: int = 0
        self._total_batches: int = 0
        self._total_covered_mwh = Decimal("0")
        self._total_uncovered_mwh = Decimal("0")
        self._total_co2e_kg_processed = Decimal("0")
        self._total_errors: int = 0

        logger.info(
            "MarketEmissionsCalculatorEngine initialized "
            "(provenance=%s, default_gwp=%s, default_region=%s)",
            self._enable_provenance,
            self._default_gwp,
            self._default_region,
        )

    # ==================================================================
    # 1. calculate_covered_emissions
    # ==================================================================

    def calculate_covered_emissions(
        self,
        instrument_type: str,
        mwh: Decimal,
        ef_kgco2e_kwh: Decimal,
    ) -> Dict[str, Any]:
        """Calculate emissions for electricity covered by a contractual instrument.

        Formula:
            ef_kgco2e_mwh = ef_kgco2e_kwh x 1000
            total_co2e_kg = mwh x ef_kgco2e_mwh
            total_co2e_tonnes = total_co2e_kg / 1000

        For renewable instruments (RECs, GOs backed by solar/wind/etc.),
        the ef_kgco2e_kwh is typically 0.000, yielding zero emissions.

        Args:
            instrument_type: Type of contractual instrument (e.g. REC, GO,
                PPA, SUPPLIER_SPECIFIC). Must be a recognized type.
            mwh: Electricity consumption covered by this instrument in MWh.
                Must be >= 0.
            ef_kgco2e_kwh: Emission factor in kg CO2e per kWh from the
                instrument. Must be >= 0.

        Returns:
            Dictionary with keys:
                - instrument_type (str)
                - consumption_mwh (Decimal)
                - ef_kgco2e_kwh (Decimal)
                - ef_kgco2e_mwh (Decimal)
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - coverage_type (str): "COVERED"
                - calculation_trace (list[str])
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If mwh or ef is negative, or instrument_type
                is not recognized.
        """
        start = time.monotonic()
        mwh = _to_decimal(mwh)
        ef_kgco2e_kwh = _to_decimal(ef_kgco2e_kwh)
        instrument_type = instrument_type.strip().upper()

        # Validate
        errors = self._validate_covered_inputs(instrument_type, mwh, ef_kgco2e_kwh)
        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        trace: List[str] = []
        trace.append(
            f"[1] Covered input: instrument={instrument_type}, "
            f"mwh={mwh}, ef={ef_kgco2e_kwh} kgCO2e/kWh"
        )

        # Convert EF from kgCO2e/kWh to kgCO2e/MWh
        ef_kgco2e_mwh = _q(ef_kgco2e_kwh * _KGCO2E_KWH_TO_MWH)
        trace.append(f"[2] EF conversion: {ef_kgco2e_kwh} kgCO2e/kWh x 1000 = {ef_kgco2e_mwh} kgCO2e/MWh")

        # Calculate emissions
        total_co2e_kg = _q(mwh * ef_kgco2e_mwh)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        trace.append(
            f"[3] Emissions: {mwh} MWh x {ef_kgco2e_mwh} kgCO2e/MWh = "
            f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
        )

        # Provenance
        prov_data = {
            "method": "calculate_covered_emissions",
            "instrument_type": instrument_type,
            "mwh": str(mwh),
            "ef_kgco2e_kwh": str(ef_kgco2e_kwh),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[4] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start
        self._update_stats(total_co2e_kg, covered_mwh=mwh)
        _record_calculation(instrument_type, "calculate_covered_emissions", "completed")
        _record_emissions(instrument_type, "CO2e", float(total_co2e_kg))
        _observe_duration("calculate_covered_emissions", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "covered_calculation", "calculate_covered_emissions",
                f"cov_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "instrument_type": instrument_type,
            "consumption_mwh": mwh,
            "ef_kgco2e_kwh": ef_kgco2e_kwh,
            "ef_kgco2e_mwh": ef_kgco2e_mwh,
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "coverage_type": "COVERED",
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 2. calculate_uncovered_emissions
    # ==================================================================

    def calculate_uncovered_emissions(
        self,
        mwh: Decimal,
        region: str,
        residual_mix_ef: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions for electricity not covered by any instrument.

        Uncovered consumption uses the residual mix emission factor for the
        specified region. If no explicit EF is provided, the built-in
        default for the region is used.

        Formula:
            ef_kgco2e_mwh = residual_mix_ef (kgCO2e/kWh) x 1000
            total_co2e_kg = mwh x ef_kgco2e_mwh
            total_co2e_tonnes = total_co2e_kg / 1000

        Args:
            mwh: Uncovered electricity consumption in MWh. Must be >= 0.
            region: Region identifier for residual mix lookup (e.g. US, EU,
                DE, GB). Case-insensitive.
            residual_mix_ef: Optional explicit residual mix emission factor
                in kgCO2e/kWh. If None, the built-in default for the
                region is used.

        Returns:
            Dictionary with keys:
                - consumption_mwh (Decimal)
                - region (str)
                - residual_mix_ef_kgco2e_kwh (Decimal)
                - ef_kgco2e_mwh (Decimal)
                - ef_source (str)
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - coverage_type (str): "UNCOVERED"
                - calculation_trace (list[str])
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If mwh is negative or region has no residual
                mix factor and none is provided.
        """
        start = time.monotonic()
        mwh = _to_decimal(mwh)
        region = region.strip().upper()

        if mwh < _ZERO:
            raise ValueError("mwh must be >= 0")

        trace: List[str] = []

        # Resolve residual mix EF
        if residual_mix_ef is not None:
            ef_kwh = _to_decimal(residual_mix_ef)
            ef_source = "USER_PROVIDED"
            trace.append(f"[1] User-provided residual mix EF: {ef_kwh} kgCO2e/kWh")
        else:
            ef_kwh, ef_source = self._resolve_residual_mix_ef(region, trace)

        if ef_kwh < _ZERO:
            raise ValueError("residual_mix_ef must be >= 0")

        trace.append(
            f"[2] Uncovered input: mwh={mwh}, region={region}, "
            f"ef={ef_kwh} kgCO2e/kWh (source={ef_source})"
        )

        # Convert to kgCO2e/MWh and calculate
        ef_kgco2e_mwh = _q(ef_kwh * _KGCO2E_KWH_TO_MWH)
        total_co2e_kg = _q(mwh * ef_kgco2e_mwh)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        trace.append(
            f"[3] Emissions: {mwh} MWh x {ef_kgco2e_mwh} kgCO2e/MWh = "
            f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
        )

        prov_data = {
            "method": "calculate_uncovered_emissions",
            "mwh": str(mwh),
            "region": region,
            "residual_mix_ef": str(ef_kwh),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[4] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start
        self._update_stats(total_co2e_kg, uncovered_mwh=mwh)
        _record_calculation("RESIDUAL_MIX", "calculate_uncovered_emissions", "completed")
        _record_emissions("RESIDUAL_MIX", "CO2e", float(total_co2e_kg))
        _observe_duration("calculate_uncovered_emissions", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "uncovered_calculation", "calculate_uncovered_emissions",
                f"uncov_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "consumption_mwh": mwh,
            "region": region,
            "residual_mix_ef_kgco2e_kwh": ef_kwh,
            "ef_kgco2e_mwh": ef_kgco2e_mwh,
            "ef_source": ef_source,
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "coverage_type": "UNCOVERED",
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 3. calculate_market_based
    # ==================================================================

    def calculate_market_based(
        self,
        total_mwh: Decimal,
        covered_results: List[Dict[str, Any]],
        uncovered_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate total market-based Scope 2 emissions.

        Combines emissions from all covered instruments and the uncovered
        residual mix portion into a single market-based total.

        Formula:
            total = sum(covered_emissions) + uncovered_emissions

        Args:
            total_mwh: Total electricity consumption in MWh for
                cross-check. Must be >= 0.
            covered_results: List of results from
                ``calculate_covered_emissions``. Each must contain
                ``total_co2e_kg``, ``consumption_mwh``.
            uncovered_result: Result from ``calculate_uncovered_emissions``.
                Must contain ``total_co2e_kg``, ``consumption_mwh``.

        Returns:
            Dictionary with keys:
                - total_mwh (Decimal)
                - covered_mwh (Decimal)
                - uncovered_mwh (Decimal)
                - coverage_pct (Decimal): Percentage covered by instruments
                - covered_co2e_kg (Decimal)
                - uncovered_co2e_kg (Decimal)
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - instrument_count (int)
                - mwh_balance_check (str): PASS or WARN
                - covered_details (list)
                - uncovered_details (dict)
                - calculation_trace (list[str])
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If total_mwh is negative.
        """
        start = time.monotonic()
        total_mwh = _to_decimal(total_mwh)
        if total_mwh < _ZERO:
            raise ValueError("total_mwh must be >= 0")

        trace: List[str] = []
        trace.append(f"[1] Market-based calc: total_mwh={total_mwh}")

        # Sum covered emissions
        covered_co2e_kg = _ZERO
        covered_mwh = _ZERO
        for i, cr in enumerate(covered_results):
            cr_co2e = _to_decimal(cr.get("total_co2e_kg", 0))
            cr_mwh = _to_decimal(cr.get("consumption_mwh", 0))
            covered_co2e_kg += cr_co2e
            covered_mwh += cr_mwh
            trace.append(
                f"[2.{i+1}] Covered instrument: "
                f"{cr.get('instrument_type', 'UNKNOWN')} "
                f"{cr_mwh} MWh -> {cr_co2e} kgCO2e"
            )

        covered_co2e_kg = _q(covered_co2e_kg)
        covered_mwh = _q(covered_mwh)

        # Uncovered emissions
        uncov_co2e = _to_decimal(uncovered_result.get("total_co2e_kg", 0))
        uncov_mwh = _to_decimal(uncovered_result.get("consumption_mwh", 0))
        uncov_co2e = _q(uncov_co2e)
        uncov_mwh = _q(uncov_mwh)
        trace.append(f"[3] Uncovered: {uncov_mwh} MWh -> {uncov_co2e} kgCO2e")

        # Totals
        total_co2e_kg = _q(covered_co2e_kg + uncov_co2e)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        trace.append(
            f"[4] Total: {covered_co2e_kg} + {uncov_co2e} = "
            f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
        )

        # Coverage percentage
        if total_mwh > _ZERO:
            coverage_pct = _q((covered_mwh / total_mwh) * Decimal("100"))
        else:
            coverage_pct = _ZERO
        trace.append(f"[5] Coverage: {covered_mwh}/{total_mwh} = {coverage_pct}%")

        # MWh balance check
        balance_mwh = _q(covered_mwh + uncov_mwh)
        balance_diff = abs(balance_mwh - total_mwh)
        balance_threshold = Decimal("0.01")
        if balance_diff <= balance_threshold:
            balance_check = "PASS"
        else:
            balance_check = "WARN"
            logger.warning(
                "MWh balance mismatch: covered(%s) + uncovered(%s) = %s vs total(%s), diff=%s",
                covered_mwh, uncov_mwh, balance_mwh, total_mwh, balance_diff,
            )
        trace.append(
            f"[6] Balance check: {covered_mwh}+{uncov_mwh}={balance_mwh} vs "
            f"{total_mwh} -> {balance_check}"
        )

        prov_data = {
            "method": "calculate_market_based",
            "total_mwh": str(total_mwh),
            "covered_co2e_kg": str(covered_co2e_kg),
            "uncovered_co2e_kg": str(uncov_co2e),
            "total_co2e_kg": str(total_co2e_kg),
            "instrument_count": len(covered_results),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[7] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start
        _observe_duration("calculate_market_based", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "market_based_calculation", "calculate_market_based",
                f"mkt_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "total_mwh": total_mwh,
            "covered_mwh": _q_out(covered_mwh),
            "uncovered_mwh": _q_out(uncov_mwh),
            "coverage_pct": _q_out(coverage_pct),
            "covered_co2e_kg": _q_out(covered_co2e_kg),
            "uncovered_co2e_kg": _q_out(uncov_co2e),
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "instrument_count": len(covered_results),
            "mwh_balance_check": balance_check,
            "covered_details": covered_results,
            "uncovered_details": uncovered_result,
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 4. calculate_with_gas_breakdown
    # ==================================================================

    def calculate_with_gas_breakdown(
        self,
        total_mwh: Decimal,
        ef: Decimal,
        gwp_source: str = "AR5",
        gas_fractions: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """Calculate market-based emissions with per-gas CO2/CH4/N2O breakdown.

        When only a total CO2e emission factor is available (typical for
        market-based instruments), the total is apportioned to individual
        gases using gas fractions and converted using GWP values.

        Per-gas calculation:
            gas_co2e_kg = total_co2e_kg x gas_fraction
            gas_mass_kg = gas_co2e_kg / GWP_gas  (mass of the actual gas)

        Args:
            total_mwh: Total electricity consumption in MWh.
            ef: Total emission factor in kgCO2e/kWh. Must be >= 0.
            gwp_source: IPCC GWP source (AR4, AR5, AR6, AR6_20YR).
                Default AR5.
            gas_fractions: Optional per-gas fractions. If None, uses
                DEFAULT_GAS_FRACTIONS. Keys: co2, ch4, n2o. Values must
                sum to 1.0.

        Returns:
            Dictionary with keys:
                - consumption_mwh (Decimal)
                - ef_kgco2e_kwh (Decimal)
                - ef_kgco2e_mwh (Decimal)
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - gas_breakdown (list[dict]): Per-gas results
                - gwp_source (str)
                - gas_fractions_used (dict)
                - calculation_trace (list[str])
                - provenance_hash (str)
                - processing_time_ms (float)

        Raises:
            ValueError: If inputs are invalid or GWP source unknown.
        """
        start = time.monotonic()
        total_mwh = _to_decimal(total_mwh)
        ef = _to_decimal(ef)

        if total_mwh < _ZERO:
            raise ValueError("total_mwh must be >= 0")
        if ef < _ZERO:
            raise ValueError("ef must be >= 0")
        if gwp_source not in GWP_VALUES:
            raise ValueError(
                f"Unknown gwp_source: {gwp_source}. "
                f"Must be one of {list(GWP_VALUES.keys())}"
            )

        fractions = gas_fractions if gas_fractions is not None else dict(DEFAULT_GAS_FRACTIONS)
        self._validate_gas_fractions(fractions)

        gwp = GWP_VALUES[gwp_source]
        trace: List[str] = []
        trace.append(
            f"[1] Gas breakdown input: mwh={total_mwh}, "
            f"ef={ef} kgCO2e/kWh, gwp={gwp_source}"
        )

        # Total CO2e
        ef_mwh = _q(ef * _KGCO2E_KWH_TO_MWH)
        total_co2e_kg = _q(total_mwh * ef_mwh)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)
        trace.append(
            f"[2] Total: {total_mwh} x {ef_mwh} = "
            f"{total_co2e_kg} kgCO2e = {total_co2e_tonnes} tCO2e"
        )

        # Per-gas breakdown
        gas_breakdown: List[Dict[str, Any]] = []
        gas_map = [
            ("CO2", "co2", gwp["CO2"]),
            ("CH4", "ch4", gwp["CH4"]),
            ("N2O", "n2o", gwp["N2O"]),
        ]

        for gas_name, frac_key, gwp_factor in gas_map:
            fraction = _to_decimal(fractions.get(frac_key, _ZERO))
            gas_co2e_kg = _q(total_co2e_kg * fraction)
            gas_mass_kg = _q(gas_co2e_kg / gwp_factor) if gwp_factor > _ZERO else _ZERO
            gas_breakdown.append({
                "gas": gas_name,
                "fraction": fraction,
                "co2e_kg": _q_out(gas_co2e_kg),
                "mass_kg": _q_out(gas_mass_kg),
                "gwp_factor": gwp_factor,
            })
            trace.append(
                f"[3] {gas_name}: fraction={fraction}, "
                f"co2e={gas_co2e_kg} kg, mass={gas_mass_kg} kg "
                f"(GWP={gwp_factor})"
            )

        prov_data = {
            "method": "calculate_with_gas_breakdown",
            "total_mwh": str(total_mwh),
            "ef_kgco2e_kwh": str(ef),
            "gwp_source": gwp_source,
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[4] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start
        _observe_duration("calculate_with_gas_breakdown", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "gas_breakdown_calculation", "calculate_with_gas_breakdown",
                f"gas_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "consumption_mwh": total_mwh,
            "ef_kgco2e_kwh": ef,
            "ef_kgco2e_mwh": ef_mwh,
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "gas_breakdown": gas_breakdown,
            "gwp_source": gwp_source,
            "gas_fractions_used": fractions,
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 5. calculate_for_facility
    # ==================================================================

    def calculate_for_facility(
        self,
        facility_id: str,
        purchases_with_allocations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate market-based emissions for a single facility.

        Accepts a list of electricity purchases, each with instrument
        allocation details, and computes covered/uncovered emissions.

        Each purchase dict must contain:
            - mwh (Decimal): Consumption in MWh
            - instrument_type (str): e.g. REC, GO, PPA, RESIDUAL_MIX
            - ef_kgco2e_kwh (Decimal): Emission factor in kgCO2e/kWh
        Optionally:
            - region (str): For residual mix lookups
            - description (str): Human-readable description

        Args:
            facility_id: Unique facility identifier.
            purchases_with_allocations: List of purchase dictionaries.

        Returns:
            Dictionary with facility-level totals including covered and
            uncovered splits, provenance, and trace.

        Raises:
            ValueError: If facility_id is empty.
        """
        start = time.monotonic()
        calc_id = f"s2m_fac_{uuid.uuid4().hex[:12]}"
        trace: List[str] = []

        if not facility_id or not facility_id.strip():
            raise ValueError("facility_id must not be empty")

        trace.append(
            f"[1] Facility calc: id={facility_id}, "
            f"purchases={len(purchases_with_allocations)}"
        )

        try:
            covered_results: List[Dict[str, Any]] = []
            uncovered_co2e_kg = _ZERO
            uncovered_mwh_total = _ZERO
            total_mwh = _ZERO

            for i, purchase in enumerate(purchases_with_allocations):
                p_mwh = _to_decimal(purchase.get("mwh", 0))
                p_type = purchase.get("instrument_type", "RESIDUAL_MIX").strip().upper()
                p_ef = _to_decimal(purchase.get("ef_kgco2e_kwh", 0))
                p_region = purchase.get("region", self._default_region)
                total_mwh += p_mwh

                if p_type == "RESIDUAL_MIX":
                    uncov = self.calculate_uncovered_emissions(
                        mwh=p_mwh, region=p_region, residual_mix_ef=p_ef if p_ef > _ZERO else None,
                    )
                    uncovered_co2e_kg += _to_decimal(uncov["total_co2e_kg"])
                    uncovered_mwh_total += p_mwh
                    trace.append(
                        f"[2.{i+1}] Uncovered: {p_mwh} MWh, region={p_region} -> "
                        f"{uncov['total_co2e_kg']} kgCO2e"
                    )
                else:
                    cov = self.calculate_covered_emissions(
                        instrument_type=p_type, mwh=p_mwh, ef_kgco2e_kwh=p_ef,
                    )
                    covered_results.append(cov)
                    trace.append(
                        f"[2.{i+1}] Covered: {p_type} {p_mwh} MWh, "
                        f"ef={p_ef} -> {cov['total_co2e_kg']} kgCO2e"
                    )

            # Aggregate
            covered_co2e_kg = _ZERO
            covered_mwh_total = _ZERO
            for cr in covered_results:
                covered_co2e_kg += _to_decimal(cr["total_co2e_kg"])
                covered_mwh_total += _to_decimal(cr["consumption_mwh"])

            covered_co2e_kg = _q(covered_co2e_kg)
            uncovered_co2e_kg = _q(uncovered_co2e_kg)
            total_co2e_kg = _q(covered_co2e_kg + uncovered_co2e_kg)
            total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)

            if total_mwh > _ZERO:
                coverage_pct = _q((covered_mwh_total / total_mwh) * Decimal("100"))
            else:
                coverage_pct = _ZERO

            trace.append(
                f"[3] Facility total: covered={covered_co2e_kg} + "
                f"uncovered={uncovered_co2e_kg} = {total_co2e_kg} kgCO2e"
            )

            prov_data = {
                "method": "calculate_for_facility",
                "calc_id": calc_id,
                "facility_id": facility_id,
                "total_mwh": str(total_mwh),
                "total_co2e_kg": str(total_co2e_kg),
                "instrument_count": len(covered_results),
            }
            provenance_hash = self._compute_provenance_hash(prov_data)

            elapsed = time.monotonic() - start
            _observe_duration("calculate_for_facility", elapsed)

            if self._provenance is not None:
                self._provenance.record(
                    "facility_calculation", "calculate_for_facility",
                    calc_id, prov_data,
                )

            return {
                "calculation_id": calc_id,
                "facility_id": facility_id,
                "status": "SUCCESS",
                "total_mwh": _q_out(total_mwh),
                "covered_mwh": _q_out(covered_mwh_total),
                "uncovered_mwh": _q_out(uncovered_mwh_total),
                "coverage_pct": _q_out(coverage_pct),
                "covered_co2e_kg": _q_out(covered_co2e_kg),
                "uncovered_co2e_kg": _q_out(uncovered_co2e_kg),
                "total_co2e_kg": _q_out(total_co2e_kg),
                "total_co2e_tonnes": _q_out(total_co2e_tonnes),
                "instrument_count": len(covered_results),
                "covered_results": covered_results,
                "provenance_hash": provenance_hash,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed * 1000, 3),
                "timestamp": _utcnow().isoformat(),
                "error_message": None,
            }

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error(
                "Facility calculation failed for %s: %s",
                facility_id, exc, exc_info=True,
            )
            self._update_stats_error()
            return {
                "calculation_id": calc_id,
                "facility_id": facility_id,
                "status": "FAILED",
                "total_mwh": _ZERO,
                "covered_mwh": _ZERO,
                "uncovered_mwh": _ZERO,
                "coverage_pct": _ZERO,
                "covered_co2e_kg": _ZERO,
                "uncovered_co2e_kg": _ZERO,
                "total_co2e_kg": _ZERO,
                "total_co2e_tonnes": _ZERO,
                "instrument_count": 0,
                "covered_results": [],
                "provenance_hash": "",
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed * 1000, 3),
                "timestamp": _utcnow().isoformat(),
                "error_message": str(exc),
            }

    # ==================================================================
    # 6. calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a batch of market-based emission calculations.

        Each request dict must contain at minimum:
            - mwh (Decimal or numeric)
            - instrument_type (str)
            - ef_kgco2e_kwh (Decimal or numeric)
        Optionally: region (str) for residual mix.

        Args:
            requests: List of request dictionaries.

        Returns:
            Dictionary with:
                - batch_id (str)
                - results (list of individual result dicts)
                - total_co2e_kg, total_co2e_tonnes (Decimal)
                - success_count, failure_count (int)
                - processing_time_ms (float)
                - provenance_hash (str)
        """
        start = time.monotonic()
        batch_id = f"batch_s2m_{uuid.uuid4().hex[:12]}"
        results: List[Dict[str, Any]] = []
        total_co2e_kg = _ZERO
        success_count = 0
        failure_count = 0

        for i, req in enumerate(requests):
            try:
                p_type = req.get("instrument_type", "RESIDUAL_MIX").strip().upper()
                p_mwh = _to_decimal(req.get("mwh", 0))
                p_ef = _to_decimal(req.get("ef_kgco2e_kwh", 0))
                p_region = req.get("region", self._default_region)

                if p_type == "RESIDUAL_MIX":
                    result = self.calculate_uncovered_emissions(
                        mwh=p_mwh, region=p_region,
                        residual_mix_ef=p_ef if p_ef > _ZERO else None,
                    )
                else:
                    result = self.calculate_covered_emissions(
                        instrument_type=p_type, mwh=p_mwh, ef_kgco2e_kwh=p_ef,
                    )

                result["request_index"] = i
                result["status"] = "SUCCESS"
                results.append(result)
                total_co2e_kg += _to_decimal(result["total_co2e_kg"])
                success_count += 1

            except Exception as exc:
                results.append({
                    "request_index": i,
                    "status": "FAILED",
                    "error_message": str(exc),
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                })
                failure_count += 1

        total_co2e_kg = _q(total_co2e_kg)
        total_co2e_tonnes = _q(total_co2e_kg * _KG_TO_TONNES)

        prov_data = {
            "method": "calculate_batch",
            "batch_id": batch_id,
            "request_count": len(requests),
            "success_count": success_count,
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        status = "success" if failure_count == 0 else (
            "partial" if success_count > 0 else "failure"
        )
        _record_batch_metric(status)
        _observe_duration("calculate_batch", elapsed)

        with self._stats_lock:
            self._total_batches += 1

        if self._provenance is not None:
            self._provenance.record("batch", "calculate_batch", batch_id, prov_data)

        return {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": _q_out(total_co2e_kg),
            "total_co2e_tonnes": _q_out(total_co2e_tonnes),
            "success_count": success_count,
            "failure_count": failure_count,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 7. calculate_renewable_instrument
    # ==================================================================

    def calculate_renewable_instrument(
        self,
        instrument_type: str,
        mwh: Decimal,
    ) -> Dict[str, Any]:
        """Calculate emissions for a renewable energy instrument (zero EF).

        Renewable instruments (RECs, GOs backed by solar, wind, hydro,
        nuclear, geothermal) carry a zero emission factor, resulting in
        zero Scope 2 market-based emissions for the covered MWh.

        Args:
            instrument_type: Type of renewable instrument (e.g. REC, GO).
                Must be a recognized instrument type.
            mwh: Electricity consumption covered by the instrument in MWh.
                Must be >= 0.

        Returns:
            Dictionary with zero emissions and renewable metadata.

        Raises:
            ValueError: If mwh is negative or instrument_type invalid.
        """
        start = time.monotonic()
        mwh = _to_decimal(mwh)
        instrument_type = instrument_type.strip().upper()

        if mwh < _ZERO:
            raise ValueError("mwh must be >= 0")

        trace: List[str] = []
        trace.append(
            f"[1] Renewable instrument: type={instrument_type}, mwh={mwh}"
        )
        trace.append("[2] Renewable EF = 0.000 kgCO2e/kWh (zero emissions)")

        result = self.calculate_covered_emissions(
            instrument_type=instrument_type,
            mwh=mwh,
            ef_kgco2e_kwh=Decimal("0.000"),
        )

        result["is_renewable"] = True
        result["renewable_generation_source"] = "UNSPECIFIED"
        result["calculation_trace"] = trace + result.get("calculation_trace", [])

        elapsed = time.monotonic() - start
        _observe_duration("calculate_renewable_instrument", elapsed)

        return result

    # ==================================================================
    # 8. calculate_supplier_specific
    # ==================================================================

    def calculate_supplier_specific(
        self,
        supplier_ef: Decimal,
        mwh: Decimal,
    ) -> Dict[str, Any]:
        """Calculate emissions using a supplier-specific emission factor.

        The supplier-specific method is the highest-quality market-based
        approach per GHG Protocol Scope 2 Guidance hierarchy.

        Args:
            supplier_ef: Supplier-specific emission factor in kgCO2e/kWh.
                Must be >= 0.
            mwh: Electricity consumption from this supplier in MWh.
                Must be >= 0.

        Returns:
            Dictionary with supplier-specific calculation results.

        Raises:
            ValueError: If supplier_ef or mwh is negative.
        """
        start = time.monotonic()
        supplier_ef = _to_decimal(supplier_ef)
        mwh = _to_decimal(mwh)

        if supplier_ef < _ZERO:
            raise ValueError("supplier_ef must be >= 0")
        if mwh < _ZERO:
            raise ValueError("mwh must be >= 0")

        result = self.calculate_covered_emissions(
            instrument_type="SUPPLIER_SPECIFIC",
            mwh=mwh,
            ef_kgco2e_kwh=supplier_ef,
        )

        result["market_method_rank"] = 1
        result["method_description"] = (
            "Supplier-specific emission factor - highest quality "
            "in GHG Protocol Scope 2 market-based hierarchy"
        )

        elapsed = time.monotonic() - start
        _observe_duration("calculate_supplier_specific", elapsed)

        return result

    # ==================================================================
    # 9. calculate_ppa_emissions
    # ==================================================================

    def calculate_ppa_emissions(
        self,
        ppa_source: str,
        mwh: Decimal,
        custom_ef: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate emissions for a Power Purchase Agreement (PPA).

        For PPAs backed by renewable sources (solar, wind, etc.), the
        emission factor defaults to zero. For other PPA sources, a
        custom emission factor must be provided.

        Args:
            ppa_source: Generation source of the PPA (e.g. solar, wind,
                natural_gas). Case-insensitive.
            mwh: Electricity consumed under the PPA in MWh.
                Must be >= 0.
            custom_ef: Optional emission factor in kgCO2e/kWh. If None,
                looks up from RENEWABLE_EF for renewable sources.

        Returns:
            Dictionary with PPA-specific calculation results.

        Raises:
            ValueError: If mwh is negative, or if the source is not
                renewable and no custom_ef is provided.
        """
        start = time.monotonic()
        mwh = _to_decimal(mwh)
        source_lower = ppa_source.strip().lower()

        if mwh < _ZERO:
            raise ValueError("mwh must be >= 0")

        trace: List[str] = []
        trace.append(f"[1] PPA: source={ppa_source}, mwh={mwh}")

        # Determine emission factor
        if custom_ef is not None:
            ef = _to_decimal(custom_ef)
            ef_source = "CUSTOM_PPA_EF"
            trace.append(f"[2] Custom PPA EF: {ef} kgCO2e/kWh")
        elif source_lower in RENEWABLE_EF:
            ef = RENEWABLE_EF[source_lower]
            ef_source = f"RENEWABLE_EF_{source_lower.upper()}"
            trace.append(f"[2] Renewable PPA EF: {ef} kgCO2e/kWh (source={source_lower})")
        elif source_lower in BIOGENIC_SOURCES:
            ef = Decimal("0.000")
            ef_source = f"BIOGENIC_{source_lower.upper()}"
            trace.append(
                f"[2] Biogenic PPA source: {source_lower}. "
                f"EF=0.000 (biogenic, reported as memo item)"
            )
        else:
            raise ValueError(
                f"PPA source '{ppa_source}' is not renewable/biogenic and "
                f"no custom_ef provided. Provide a custom_ef for non-renewable PPAs."
            )

        # Determine instrument type
        instrument_type = "VPPA" if source_lower in RENEWABLE_EF else "PPA"

        result = self.calculate_covered_emissions(
            instrument_type=instrument_type,
            mwh=mwh,
            ef_kgco2e_kwh=ef,
        )

        result["ppa_source"] = ppa_source
        result["ppa_ef_source"] = ef_source
        result["is_renewable_ppa"] = source_lower in RENEWABLE_EF
        result["is_biogenic_ppa"] = source_lower in BIOGENIC_SOURCES
        result["calculation_trace"] = trace + result.get("calculation_trace", [])

        elapsed = time.monotonic() - start
        _observe_duration("calculate_ppa_emissions", elapsed)

        return result

    # ==================================================================
    # 10. aggregate_by_instrument
    # ==================================================================

    def aggregate_by_instrument(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate calculation results by instrument type.

        Groups results by their ``instrument_type`` field and sums
        emissions and consumption for each type.

        Args:
            results: List of calculation result dictionaries. Each must
                contain ``instrument_type``, ``total_co2e_kg``,
                ``consumption_mwh``.

        Returns:
            Dictionary with per-instrument aggregations and grand total.
        """
        start = time.monotonic()
        by_instrument: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            inst = r.get("instrument_type", r.get("coverage_type", "UNKNOWN"))
            if inst not in by_instrument:
                by_instrument[inst] = {
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "consumption_mwh": _ZERO,
                    "count": _ZERO,
                }
            by_instrument[inst]["total_co2e_kg"] += _to_decimal(r.get("total_co2e_kg", 0))
            by_instrument[inst]["total_co2e_tonnes"] += _to_decimal(r.get("total_co2e_tonnes", 0))
            by_instrument[inst]["consumption_mwh"] += _to_decimal(r.get("consumption_mwh", 0))
            by_instrument[inst]["count"] += _ONE

        grand_co2e_kg = _ZERO
        grand_co2e_tonnes = _ZERO
        grand_mwh = _ZERO
        instrument_summaries: Dict[str, Dict[str, Any]] = {}

        for inst, agg in sorted(by_instrument.items()):
            co2e_kg = _q(agg["total_co2e_kg"])
            co2e_t = _q(agg["total_co2e_tonnes"])
            cons = _q(agg["consumption_mwh"])
            instrument_summaries[inst] = {
                "total_co2e_kg": _q_out(co2e_kg),
                "total_co2e_tonnes": _q_out(co2e_t),
                "consumption_mwh": _q_out(cons),
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg
            grand_co2e_tonnes += co2e_t
            grand_mwh += cons

        prov_data = {
            "method": "aggregate_by_instrument",
            "instrument_count": len(by_instrument),
            "grand_co2e_kg": str(grand_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "by_instrument": instrument_summaries,
            "grand_total": {
                "total_co2e_kg": _q_out(grand_co2e_kg),
                "total_co2e_tonnes": _q_out(grand_co2e_tonnes),
                "consumption_mwh": _q_out(grand_mwh),
            },
            "instrument_type_count": len(by_instrument),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 11. aggregate_by_facility
    # ==================================================================

    def aggregate_by_facility(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate calculation results by facility_id.

        Groups results by their ``facility_id`` and sums emissions.

        Args:
            results: List of calculation result dictionaries. Each must
                contain ``facility_id``, ``total_co2e_kg``,
                ``total_co2e_tonnes``.

        Returns:
            Dictionary mapping facility_id to aggregated totals, plus
            a ``grand_total`` entry.
        """
        start = time.monotonic()
        by_facility: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            fid = r.get("facility_id", "unknown")
            if fid not in by_facility:
                by_facility[fid] = {
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "total_mwh": _ZERO,
                    "covered_mwh": _ZERO,
                    "uncovered_mwh": _ZERO,
                    "count": _ZERO,
                }
            by_facility[fid]["total_co2e_kg"] += _to_decimal(r.get("total_co2e_kg", 0))
            by_facility[fid]["total_co2e_tonnes"] += _to_decimal(r.get("total_co2e_tonnes", 0))
            by_facility[fid]["total_mwh"] += _to_decimal(r.get("total_mwh", r.get("consumption_mwh", 0)))
            by_facility[fid]["covered_mwh"] += _to_decimal(r.get("covered_mwh", 0))
            by_facility[fid]["uncovered_mwh"] += _to_decimal(r.get("uncovered_mwh", 0))
            by_facility[fid]["count"] += _ONE

        grand_co2e_kg = _ZERO
        grand_co2e_tonnes = _ZERO
        grand_mwh = _ZERO
        facility_summaries: Dict[str, Dict[str, Any]] = {}

        for fid, agg in sorted(by_facility.items()):
            co2e_kg = _q(agg["total_co2e_kg"])
            co2e_t = _q(agg["total_co2e_tonnes"])
            t_mwh = _q(agg["total_mwh"])
            facility_summaries[fid] = {
                "total_co2e_kg": _q_out(co2e_kg),
                "total_co2e_tonnes": _q_out(co2e_t),
                "total_mwh": _q_out(t_mwh),
                "covered_mwh": _q_out(_q(agg["covered_mwh"])),
                "uncovered_mwh": _q_out(_q(agg["uncovered_mwh"])),
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg
            grand_co2e_tonnes += co2e_t
            grand_mwh += t_mwh

        prov_data = {
            "method": "aggregate_by_facility",
            "facility_count": len(by_facility),
            "grand_co2e_kg": str(grand_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "by_facility": facility_summaries,
            "grand_total": {
                "total_co2e_kg": _q_out(grand_co2e_kg),
                "total_co2e_tonnes": _q_out(grand_co2e_tonnes),
                "total_mwh": _q_out(grand_mwh),
            },
            "facility_count": len(by_facility),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 12. aggregate_by_period
    # ==================================================================

    def aggregate_by_period(
        self,
        results: List[Dict[str, Any]],
        period_type: str = "annual",
    ) -> Dict[str, Any]:
        """Aggregate calculation results by time period.

        Groups results by the specified period (monthly, quarterly, annual)
        based on the ``month``, ``timestamp``, or ``date`` field in each
        result.

        Args:
            results: List of calculation result dictionaries.
            period_type: Aggregation period. One of ``monthly``,
                ``quarterly``, ``annual``. Default ``annual``.

        Returns:
            Dictionary with period-level aggregations and grand total.

        Raises:
            ValueError: If period_type is not valid.
        """
        start = time.monotonic()
        valid_periods = ("monthly", "quarterly", "annual")
        if period_type not in valid_periods:
            raise ValueError(
                f"period_type must be one of {valid_periods}, got '{period_type}'"
            )

        by_period: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            key = self._get_period_key(r, period_type)
            if key not in by_period:
                by_period[key] = {
                    "total_co2e_kg": _ZERO,
                    "total_co2e_tonnes": _ZERO,
                    "consumption_mwh": _ZERO,
                    "count": _ZERO,
                }
            by_period[key]["total_co2e_kg"] += _to_decimal(r.get("total_co2e_kg", 0))
            by_period[key]["total_co2e_tonnes"] += _to_decimal(r.get("total_co2e_tonnes", 0))
            by_period[key]["consumption_mwh"] += _to_decimal(
                r.get("total_mwh", r.get("consumption_mwh", 0))
            )
            by_period[key]["count"] += _ONE

        grand_co2e_kg = _ZERO
        period_summaries: Dict[str, Dict[str, Any]] = {}

        for key, agg in sorted(by_period.items()):
            co2e_kg = _q(agg["total_co2e_kg"])
            co2e_t = _q(agg["total_co2e_tonnes"])
            cons = _q(agg["consumption_mwh"])
            period_summaries[key] = {
                "total_co2e_kg": _q_out(co2e_kg),
                "total_co2e_tonnes": _q_out(co2e_t),
                "consumption_mwh": _q_out(cons),
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg

        prov_data = {
            "method": "aggregate_by_period",
            "period_type": period_type,
            "grand_co2e_kg": str(grand_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "period_type": period_type,
            "by_period": period_summaries,
            "grand_total_co2e_kg": _q_out(grand_co2e_kg),
            "period_count": len(period_summaries),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 13. aggregate_by_coverage
    # ==================================================================

    def aggregate_by_coverage(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate calculation results by coverage type (covered vs uncovered).

        Splits results into two buckets based on the ``coverage_type``
        field: COVERED (instrument-backed) and UNCOVERED (residual mix).

        Args:
            results: List of calculation result dictionaries. Each should
                contain ``coverage_type`` (or ``instrument_type``),
                ``total_co2e_kg``, ``consumption_mwh``.

        Returns:
            Dictionary with covered/uncovered totals, percentages, and
            grand total.
        """
        start = time.monotonic()

        covered_co2e_kg = _ZERO
        covered_mwh = _ZERO
        covered_count = 0
        uncovered_co2e_kg = _ZERO
        uncovered_mwh = _ZERO
        uncovered_count = 0

        for r in results:
            cov_type = r.get("coverage_type", "COVERED")
            co2e = _to_decimal(r.get("total_co2e_kg", 0))
            mwh_val = _to_decimal(r.get("consumption_mwh", 0))

            if cov_type == "UNCOVERED":
                uncovered_co2e_kg += co2e
                uncovered_mwh += mwh_val
                uncovered_count += 1
            else:
                covered_co2e_kg += co2e
                covered_mwh += mwh_val
                covered_count += 1

        covered_co2e_kg = _q(covered_co2e_kg)
        uncovered_co2e_kg = _q(uncovered_co2e_kg)
        total_co2e_kg = _q(covered_co2e_kg + uncovered_co2e_kg)
        total_mwh = _q(covered_mwh + uncovered_mwh)

        if total_co2e_kg > _ZERO:
            covered_pct_emissions = _q(
                (covered_co2e_kg / total_co2e_kg) * Decimal("100")
            )
            uncovered_pct_emissions = _q(
                (uncovered_co2e_kg / total_co2e_kg) * Decimal("100")
            )
        else:
            covered_pct_emissions = _ZERO
            uncovered_pct_emissions = _ZERO

        if total_mwh > _ZERO:
            covered_pct_mwh = _q((covered_mwh / total_mwh) * Decimal("100"))
            uncovered_pct_mwh = _q((uncovered_mwh / total_mwh) * Decimal("100"))
        else:
            covered_pct_mwh = _ZERO
            uncovered_pct_mwh = _ZERO

        prov_data = {
            "method": "aggregate_by_coverage",
            "covered_co2e_kg": str(covered_co2e_kg),
            "uncovered_co2e_kg": str(uncovered_co2e_kg),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "covered": {
                "total_co2e_kg": _q_out(covered_co2e_kg),
                "consumption_mwh": _q_out(covered_mwh),
                "calculation_count": covered_count,
                "pct_of_emissions": _q_out(covered_pct_emissions),
                "pct_of_consumption": _q_out(covered_pct_mwh),
            },
            "uncovered": {
                "total_co2e_kg": _q_out(uncovered_co2e_kg),
                "consumption_mwh": _q_out(uncovered_mwh),
                "calculation_count": uncovered_count,
                "pct_of_emissions": _q_out(uncovered_pct_emissions),
                "pct_of_consumption": _q_out(uncovered_pct_mwh),
            },
            "grand_total": {
                "total_co2e_kg": _q_out(total_co2e_kg),
                "total_co2e_tonnes": _q_out(_q(total_co2e_kg * _KG_TO_TONNES)),
                "total_mwh": _q_out(total_mwh),
            },
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 14-17. Unit Conversions
    # ==================================================================

    def kwh_to_mwh(self, kwh: Decimal) -> Decimal:
        """Convert kilowatt-hours to megawatt-hours.

        Args:
            kwh: Energy in kWh. Must be >= 0.

        Returns:
            Energy in MWh (Decimal).

        Raises:
            ValueError: If kwh is negative.
        """
        kwh = _to_decimal(kwh)
        if kwh < _ZERO:
            raise ValueError("kwh must be >= 0")
        return _q(kwh * _KWH_TO_MWH)

    def gj_to_mwh(self, gj: Decimal) -> Decimal:
        """Convert gigajoules to megawatt-hours.

        Conversion: 1 GJ = 0.277778 MWh (1 MWh = 3.6 GJ).

        Args:
            gj: Energy in GJ. Must be >= 0.

        Returns:
            Energy in MWh (Decimal).

        Raises:
            ValueError: If gj is negative.
        """
        gj = _to_decimal(gj)
        if gj < _ZERO:
            raise ValueError("gj must be >= 0")
        return _q(gj * _GJ_TO_MWH)

    def mmbtu_to_mwh(self, mmbtu: Decimal) -> Decimal:
        """Convert million British thermal units to megawatt-hours.

        Conversion: 1 MMBTU = 0.293071 MWh.

        Args:
            mmbtu: Energy in MMBTU. Must be >= 0.

        Returns:
            Energy in MWh (Decimal).

        Raises:
            ValueError: If mmbtu is negative.
        """
        mmbtu = _to_decimal(mmbtu)
        if mmbtu < _ZERO:
            raise ValueError("mmbtu must be >= 0")
        return _q(mmbtu * _MMBTU_TO_MWH)

    def normalize_consumption(self, quantity: Decimal, unit: str) -> Decimal:
        """Normalize an energy quantity from any supported unit to MWh.

        Supported units (case-insensitive):
            kWh, MWh, GJ, MMBTU, TJ

        Args:
            quantity: Energy quantity. Must be >= 0.
            unit: Unit string (case-insensitive).

        Returns:
            Energy in MWh (Decimal).

        Raises:
            ValueError: If quantity is negative or unit is unsupported.
        """
        quantity = _to_decimal(quantity)
        if quantity < _ZERO:
            raise ValueError("quantity must be >= 0")

        unit_upper = unit.strip().upper()
        conversions: Dict[str, Decimal] = {
            "KWH": _KWH_TO_MWH,
            "MWH": _ONE,
            "GJ": _GJ_TO_MWH,
            "MMBTU": _MMBTU_TO_MWH,
            "TJ": _TJ_TO_MWH,
        }

        factor = conversions.get(unit_upper)
        if factor is None:
            raise ValueError(
                f"Unsupported energy unit '{unit}'. "
                f"Supported: {list(conversions.keys())}"
            )

        return _q(quantity * factor)

    # ==================================================================
    # 18. validate_consumption
    # ==================================================================

    def validate_consumption(self, mwh: Decimal) -> List[str]:
        """Validate electricity consumption data for reasonableness.

        Checks:
            - Must be a valid Decimal
            - Must be >= 0
            - Warns if > 1,000,000 MWh (unusually large)
            - Warns if == 0 (no emissions to calculate)

        Args:
            mwh: Consumption value to validate.

        Returns:
            List of validation error/warning strings. Empty if valid.
        """
        errors: List[str] = []
        try:
            val = _to_decimal(mwh)
        except (ValueError, TypeError):
            errors.append(
                f"Cannot convert consumption to Decimal: {mwh!r}"
            )
            return errors

        if val < _ZERO:
            errors.append(f"consumption_mwh must be >= 0, got {val}")
        if val == _ZERO:
            errors.append(
                "WARNING: consumption_mwh is 0; no emissions to calculate"
            )
        if val > Decimal("1000000"):
            errors.append(
                f"WARNING: consumption_mwh={val} exceeds 1,000,000 MWh; "
                "verify this is correct"
            )

        return errors

    # ==================================================================
    # 19. validate_emission_factor
    # ==================================================================

    def validate_emission_factor(self, ef: Decimal) -> List[str]:
        """Validate an emission factor for reasonableness.

        Checks:
            - Must be a valid Decimal
            - Must be >= 0
            - Warns if > 2.0 kgCO2e/kWh (exceeds coal-fired max)
            - Acceptable if == 0.0 (renewable instruments)

        Args:
            ef: Emission factor value to validate (kgCO2e/kWh).

        Returns:
            List of validation error/warning strings. Empty if valid.
        """
        errors: List[str] = []
        try:
            val = _to_decimal(ef)
        except (ValueError, TypeError):
            errors.append(
                f"Cannot convert emission factor to Decimal: {ef!r}"
            )
            return errors

        if val < _ZERO:
            errors.append(f"Emission factor must be >= 0, got {val}")
        if val > Decimal("2.0"):
            errors.append(
                f"WARNING: EF={val} kgCO2e/kWh exceeds 2.0; "
                "verify this is correct (typical range: 0-1.2)"
            )

        return errors

    # ==================================================================
    # 20. get_statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary with:
                - total_calculations (int)
                - total_batches (int)
                - total_covered_mwh (Decimal)
                - total_uncovered_mwh (Decimal)
                - total_co2e_kg_processed (Decimal)
                - total_errors (int)
                - provenance_entry_count (int)
                - supported_instrument_types (list)
                - supported_residual_mix_regions (list)
                - supported_gwp_sources (list)
                - supported_renewable_sources (list)
                - default_gwp_source (str)
                - default_region (str)
                - provenance_enabled (bool)
        """
        with self._stats_lock:
            return {
                "total_calculations": self._total_calculations,
                "total_batches": self._total_batches,
                "total_covered_mwh": self._total_covered_mwh,
                "total_uncovered_mwh": self._total_uncovered_mwh,
                "total_co2e_kg_processed": self._total_co2e_kg_processed,
                "total_errors": self._total_errors,
                "provenance_entry_count": (
                    self._provenance.entry_count if self._provenance else 0
                ),
                "supported_instrument_types": sorted(_VALID_INSTRUMENT_TYPES),
                "supported_residual_mix_regions": sorted(
                    _DEFAULT_RESIDUAL_MIX_EF.keys()
                ),
                "supported_gwp_sources": sorted(GWP_VALUES.keys()),
                "supported_renewable_sources": sorted(RENEWABLE_EF.keys()),
                "supported_biogenic_sources": sorted(BIOGENIC_SOURCES),
                "default_gwp_source": self._default_gwp,
                "default_region": self._default_region,
                "provenance_enabled": self._enable_provenance,
            }

    # ==================================================================
    # 21. reset
    # ==================================================================

    def reset(self) -> None:
        """Reset all engine statistics and provenance to initial state.

        Clears all counters and creates a fresh provenance tracker.
        This method is thread-safe.
        """
        with self._stats_lock:
            self._total_calculations = 0
            self._total_batches = 0
            self._total_covered_mwh = _ZERO
            self._total_uncovered_mwh = _ZERO
            self._total_co2e_kg_processed = _ZERO
            self._total_errors = 0

        if self._enable_provenance:
            self._provenance = _ProvenanceTracker()
        else:
            self._provenance = None

        logger.info("MarketEmissionsCalculatorEngine reset to initial state")

    # ==================================================================
    # Internal: Covered Input Validation
    # ==================================================================

    def _validate_covered_inputs(
        self,
        instrument_type: str,
        mwh: Decimal,
        ef_kgco2e_kwh: Decimal,
    ) -> List[str]:
        """Validate inputs for covered emissions calculation.

        Args:
            instrument_type: Instrument type string (already uppercased).
            mwh: Consumption in MWh.
            ef_kgco2e_kwh: Emission factor in kgCO2e/kWh.

        Returns:
            List of error strings. Empty if all valid.
        """
        errors: List[str] = []

        if instrument_type not in _VALID_INSTRUMENT_TYPES:
            errors.append(
                f"Unrecognized instrument_type: '{instrument_type}'. "
                f"Must be one of: {sorted(_VALID_INSTRUMENT_TYPES)}"
            )

        if mwh < _ZERO:
            errors.append("mwh must be >= 0")

        if ef_kgco2e_kwh < _ZERO:
            errors.append("ef_kgco2e_kwh must be >= 0")

        if ef_kgco2e_kwh > Decimal("5.0"):
            errors.append(
                f"ef_kgco2e_kwh={ef_kgco2e_kwh} exceeds 5.0 kgCO2e/kWh; "
                "this is physically unrealistic"
            )

        return errors

    # ==================================================================
    # Internal: Gas Fractions Validation
    # ==================================================================

    def _validate_gas_fractions(
        self,
        fractions: Dict[str, Decimal],
    ) -> None:
        """Validate that gas fractions are well-formed and sum to ~1.0.

        Args:
            fractions: Dictionary with keys co2, ch4, n2o.

        Raises:
            ValueError: If fractions are missing, negative, or do not
                sum to approximately 1.0.
        """
        required_keys = {"co2", "ch4", "n2o"}
        missing = required_keys - set(fractions.keys())
        if missing:
            raise ValueError(
                f"Missing gas fraction keys: {missing}. "
                f"Required: {required_keys}"
            )

        total = _ZERO
        for key in required_keys:
            val = _to_decimal(fractions[key])
            if val < _ZERO:
                raise ValueError(f"Gas fraction '{key}' must be >= 0, got {val}")
            total += val

        tolerance = Decimal("0.001")
        if abs(total - _ONE) > tolerance:
            raise ValueError(
                f"Gas fractions must sum to 1.0 (tolerance {tolerance}), "
                f"got {total}"
            )

    # ==================================================================
    # Internal: Residual Mix EF Resolution
    # ==================================================================

    def _resolve_residual_mix_ef(
        self,
        region: str,
        trace: List[str],
    ) -> Tuple[Decimal, str]:
        """Resolve the residual mix emission factor for a region.

        Resolution order:
        1. External residual_mix_db (if provided)
        2. Built-in regional defaults
        3. GLOBAL fallback

        Args:
            region: Region identifier (uppercase).
            trace: Calculation trace list.

        Returns:
            Tuple of (ef_kgco2e_kwh, source_string).

        Raises:
            ValueError: If no residual mix EF can be resolved.
        """
        # 1. Try external database
        if self._residual_mix_db is not None:
            try:
                ext_ef = self._lookup_external_residual_mix(region)
                if ext_ef is not None:
                    ef_val = _to_decimal(ext_ef.get("ef_kgco2e_kwh", 0))
                    source = ext_ef.get("source", "EXTERNAL_DB")
                    trace.append(
                        f"[RM] Resolved from external DB: "
                        f"region={region}, ef={ef_val}, source={source}"
                    )
                    return ef_val, source
            except Exception as exc:
                logger.warning(
                    "External residual mix DB lookup failed for %s: %s; "
                    "falling back to defaults",
                    region, exc,
                )
                trace.append(
                    f"[RM] External DB lookup failed: {exc}"
                )

        # 2. Built-in regional defaults
        if region in _DEFAULT_RESIDUAL_MIX_EF:
            entry = _DEFAULT_RESIDUAL_MIX_EF[region]
            ef_val = _to_decimal(entry["ef_kgco2e_kwh"])
            source = entry["source"]
            trace.append(
                f"[RM] Built-in default: region={region}, "
                f"ef={ef_val}, source={source}"
            )
            return ef_val, source

        # 3. Global fallback
        if "GLOBAL" in _DEFAULT_RESIDUAL_MIX_EF:
            entry = _DEFAULT_RESIDUAL_MIX_EF["GLOBAL"]
            ef_val = _to_decimal(entry["ef_kgco2e_kwh"])
            source = entry["source"]
            trace.append(
                f"[RM] Fallback to GLOBAL: ef={ef_val}, source={source}"
            )
            return ef_val, source

        raise ValueError(
            f"No residual mix emission factor found for region '{region}' "
            "and no GLOBAL fallback available"
        )

    def _lookup_external_residual_mix(
        self,
        region: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up residual mix EF from external database.

        Args:
            region: Region identifier.

        Returns:
            Dictionary with ef_kgco2e_kwh and source, or None.
        """
        if hasattr(self._residual_mix_db, "get_residual_mix_factor"):
            return self._residual_mix_db.get_residual_mix_factor(region=region)
        if hasattr(self._residual_mix_db, "get_factor"):
            return self._residual_mix_db.get_factor(region=region)
        return None

    # ==================================================================
    # Internal: Period Key Extraction
    # ==================================================================

    def _get_period_key(self, result: Dict[str, Any], period: str) -> str:
        """Extract the period key from a result dictionary.

        Uses ``month``, ``timestamp``, or ``date`` fields.

        Args:
            result: Calculation result dictionary.
            period: ``monthly``, ``quarterly``, or ``annual``.

        Returns:
            String key for grouping (e.g. ``2026-01``, ``2026-Q1``,
            ``2026``).
        """
        month = result.get("month")
        ts = result.get("timestamp")
        date_str = result.get("date")

        if period == "annual":
            if isinstance(ts, str) and len(ts) >= 4:
                return ts[:4]
            if date_str and len(str(date_str)) >= 4:
                return str(date_str)[:4]
            return "unknown"

        if period == "monthly":
            if month is not None:
                year = result.get("year", "unknown")
                return f"{year}-{int(month):02d}"
            if isinstance(ts, str) and len(ts) >= 7:
                return ts[:7]
            return "unknown"

        if period == "quarterly":
            if month is not None:
                year = result.get("year", "unknown")
                q = (int(month) - 1) // 3 + 1
                return f"{year}-Q{q}"
            if isinstance(ts, str) and len(ts) >= 7:
                m = int(ts[5:7])
                q = (m - 1) // 3 + 1
                return f"{ts[:4]}-Q{q}"
            return "unknown"

        return "unknown"

    # ==================================================================
    # Internal: Provenance Hash
    # ==================================================================

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of calculation data.

        Serializes the data dictionary to sorted JSON and computes the
        SHA-256 hex digest. Deterministic for identical inputs.

        Args:
            data: Dictionary of calculation data.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # ==================================================================
    # Internal: Statistics
    # ==================================================================

    def _update_stats(
        self,
        co2e_kg: Decimal,
        covered_mwh: Decimal = _ZERO,
        uncovered_mwh: Decimal = _ZERO,
    ) -> None:
        """Update running statistics counters (thread-safe).

        Args:
            co2e_kg: Emissions processed in this calculation (kg CO2e).
            covered_mwh: Covered MWh processed in this calculation.
            uncovered_mwh: Uncovered MWh processed in this calculation.
        """
        with self._stats_lock:
            self._total_calculations += 1
            self._total_co2e_kg_processed += co2e_kg
            self._total_covered_mwh += covered_mwh
            self._total_uncovered_mwh += uncovered_mwh

    def _update_stats_error(self) -> None:
        """Increment error counter (thread-safe)."""
        with self._stats_lock:
            self._total_errors += 1
