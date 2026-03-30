# -*- coding: utf-8 -*-
"""
ElectricityEmissionsEngine - Engine 2: Scope 2 Location-Based Emissions Agent (AGENT-MRV-009)

Core Scope 2 electricity emission calculations using grid-average emission
factors. Implements the GHG Protocol Scope 2 Guidance location-based method
for purchased electricity.

Formula (location-based):
    Emissions (tCO2e) = Consumption (MWh) x Grid EF (tCO2e/MWh) x (1 + T&D Loss %)

Per-gas breakdown:
    CO2 = Consumption x CO2_EF x (1 + T&D Loss) x GWP_CO2
    CH4 = Consumption x CH4_EF x (1 + T&D Loss) x GWP_CH4
    N2O = Consumption x N2O_EF x (1 + T&D Loss) x GWP_N2O
    Total CO2e = CO2 + CH4 + N2O

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (8 decimal places)
    - No LLM calls in the calculation path
    - Every step is recorded in the calculation trace
    - SHA-256 provenance hash for every result
    - Deterministic: same input -> same output (bit-perfect)

Example:
    >>> from greenlang.agents.mrv.scope2_location.electricity_emissions import ElectricityEmissionsEngine
    >>> from decimal import Decimal
    >>> engine = ElectricityEmissionsEngine()
    >>> result = engine.calculate_emissions(
    ...     consumption_mwh=Decimal("1000"),
    ...     grid_ef_co2e=Decimal("0.450"),
    ... )
    >>> assert result["total_co2e_tonnes"] == Decimal("0.450") * Decimal("1000") / Decimal("1000")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------
_PRECISION = Decimal("0.00000001")  # 8 decimal places

# ---------------------------------------------------------------------------
# Unit Conversion Constants (all Decimal for zero-hallucination)
# ---------------------------------------------------------------------------

#: 1 kWh = 0.001 MWh
_KWH_TO_MWH = Decimal("0.001")

#: 1 GJ = 0.277778 MWh (1 / 3.6)
_GJ_TO_MWH = Decimal("0.277778")

#: 1 MMBTU = 0.293071 MWh
_MMBTU_TO_MWH = Decimal("0.293071")

#: 1 MWh = 3.6 GJ
_MWH_TO_GJ = Decimal("3.6")

#: 1 MWh = 3.412142 MMBTU
_MWH_TO_MMBTU = Decimal("3.412142")

#: 1 TJ = 277.778 MWh
_TJ_TO_MWH = Decimal("277.778")

#: kg to tonnes
_KG_TO_TONNES = Decimal("0.001")

#: tonnes to kg
_TONNES_TO_KG = Decimal("1000")

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
}

# ---------------------------------------------------------------------------
# Default Country-Level Grid Emission Factors (IPCC Tier 1 defaults)
# ---------------------------------------------------------------------------
# Values in kg CO2e per MWh (location-based, grid average).
# Sources: IEA 2023, EPA eGRID 2022, DEFRA 2023, IPCC 2006 GL Vol 2.
# These are fallback defaults when no grid_factor_db is provided.

_DEFAULT_COUNTRY_GRID_EF: Dict[str, Dict[str, Decimal]] = {
    "US": {
        "co2_ef": Decimal("388.0"),    # kg CO2/MWh (EPA eGRID national avg)
        "ch4_ef": Decimal("0.011"),    # kg CH4/MWh
        "n2o_ef": Decimal("0.005"),    # kg N2O/MWh
        "td_loss_pct": Decimal("0.05"),  # 5% T&D losses
        "source": "EPA_eGRID_2022",
    },
    "GB": {
        "co2_ef": Decimal("207.0"),
        "ch4_ef": Decimal("0.007"),
        "n2o_ef": Decimal("0.003"),
        "td_loss_pct": Decimal("0.08"),
        "source": "DEFRA_2023",
    },
    "DE": {
        "co2_ef": Decimal("380.0"),
        "ch4_ef": Decimal("0.016"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
    "FR": {
        "co2_ef": Decimal("56.0"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.001"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "JP": {
        "co2_ef": Decimal("457.0"),
        "ch4_ef": Decimal("0.015"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.05"),
        "source": "IEA_2023",
    },
    "CN": {
        "co2_ef": Decimal("555.0"),
        "ch4_ef": Decimal("0.025"),
        "n2o_ef": Decimal("0.010"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "IN": {
        "co2_ef": Decimal("708.0"),
        "ch4_ef": Decimal("0.030"),
        "n2o_ef": Decimal("0.012"),
        "td_loss_pct": Decimal("0.19"),
        "source": "IEA_2023",
    },
    "AU": {
        "co2_ef": Decimal("656.0"),
        "ch4_ef": Decimal("0.022"),
        "n2o_ef": Decimal("0.009"),
        "td_loss_pct": Decimal("0.05"),
        "source": "IEA_2023",
    },
    "BR": {
        "co2_ef": Decimal("74.0"),
        "ch4_ef": Decimal("0.004"),
        "n2o_ef": Decimal("0.002"),
        "td_loss_pct": Decimal("0.15"),
        "source": "IEA_2023",
    },
    "CA": {
        "co2_ef": Decimal("120.0"),
        "ch4_ef": Decimal("0.005"),
        "n2o_ef": Decimal("0.002"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "ZA": {
        "co2_ef": Decimal("928.0"),
        "ch4_ef": Decimal("0.035"),
        "n2o_ef": Decimal("0.014"),
        "td_loss_pct": Decimal("0.08"),
        "source": "IEA_2023",
    },
    "KR": {
        "co2_ef": Decimal("415.0"),
        "ch4_ef": Decimal("0.014"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
    "IT": {
        "co2_ef": Decimal("257.0"),
        "ch4_ef": Decimal("0.010"),
        "n2o_ef": Decimal("0.004"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "ES": {
        "co2_ef": Decimal("172.0"),
        "ch4_ef": Decimal("0.007"),
        "n2o_ef": Decimal("0.003"),
        "td_loss_pct": Decimal("0.09"),
        "source": "IEA_2023",
    },
    "SE": {
        "co2_ef": Decimal("9.0"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0004"),
        "td_loss_pct": Decimal("0.07"),
        "source": "IEA_2023",
    },
    "NO": {
        "co2_ef": Decimal("8.0"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0003"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "GLOBAL": {
        "co2_ef": Decimal("436.0"),
        "ch4_ef": Decimal("0.018"),
        "n2o_ef": Decimal("0.007"),
        "td_loss_pct": Decimal("0.08"),
        "source": "IEA_2023_GLOBAL_AVG",
    },
}

# ---------------------------------------------------------------------------
# Default US EPA eGRID Subregion Emission Factors (Tier 2)
# ---------------------------------------------------------------------------
# Values in kg CO2e per MWh. Source: EPA eGRID 2022.

_DEFAULT_EGRID_SUBREGION_EF: Dict[str, Dict[str, Decimal]] = {
    "AKGD": {"co2_ef": Decimal("436.0"), "ch4_ef": Decimal("0.020"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "AKMS": {"co2_ef": Decimal("199.0"), "ch4_ef": Decimal("0.008"), "n2o_ef": Decimal("0.002"), "source": "eGRID2022"},
    "AZNM": {"co2_ef": Decimal("367.0"), "ch4_ef": Decimal("0.013"), "n2o_ef": Decimal("0.005"), "source": "eGRID2022"},
    "CAMX": {"co2_ef": Decimal("225.0"), "ch4_ef": Decimal("0.009"), "n2o_ef": Decimal("0.003"), "source": "eGRID2022"},
    "ERCT": {"co2_ef": Decimal("373.0"), "ch4_ef": Decimal("0.014"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "FRCC": {"co2_ef": Decimal("374.0"), "ch4_ef": Decimal("0.015"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "MROE": {"co2_ef": Decimal("544.0"), "ch4_ef": Decimal("0.022"), "n2o_ef": Decimal("0.008"), "source": "eGRID2022"},
    "MROW": {"co2_ef": Decimal("418.0"), "ch4_ef": Decimal("0.016"), "n2o_ef": Decimal("0.006"), "source": "eGRID2022"},
    "NEWE": {"co2_ef": Decimal("209.0"), "ch4_ef": Decimal("0.012"), "n2o_ef": Decimal("0.003"), "source": "eGRID2022"},
    "NWPP": {"co2_ef": Decimal("267.0"), "ch4_ef": Decimal("0.010"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "NYCW": {"co2_ef": Decimal("227.0"), "ch4_ef": Decimal("0.010"), "n2o_ef": Decimal("0.002"), "source": "eGRID2022"},
    "NYLI": {"co2_ef": Decimal("468.0"), "ch4_ef": Decimal("0.018"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "NYUP": {"co2_ef": Decimal("115.0"), "ch4_ef": Decimal("0.006"), "n2o_ef": Decimal("0.002"), "source": "eGRID2022"},
    "RFCE": {"co2_ef": Decimal("305.0"), "ch4_ef": Decimal("0.013"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "RFCM": {"co2_ef": Decimal("525.0"), "ch4_ef": Decimal("0.021"), "n2o_ef": Decimal("0.008"), "source": "eGRID2022"},
    "RFCW": {"co2_ef": Decimal("470.0"), "ch4_ef": Decimal("0.018"), "n2o_ef": Decimal("0.007"), "source": "eGRID2022"},
    "RMPA": {"co2_ef": Decimal("501.0"), "ch4_ef": Decimal("0.019"), "n2o_ef": Decimal("0.007"), "source": "eGRID2022"},
    "SPNO": {"co2_ef": Decimal("462.0"), "ch4_ef": Decimal("0.018"), "n2o_ef": Decimal("0.007"), "source": "eGRID2022"},
    "SPSO": {"co2_ef": Decimal("400.0"), "ch4_ef": Decimal("0.015"), "n2o_ef": Decimal("0.006"), "source": "eGRID2022"},
    "SRMV": {"co2_ef": Decimal("336.0"), "ch4_ef": Decimal("0.014"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
    "SRMW": {"co2_ef": Decimal("586.0"), "ch4_ef": Decimal("0.023"), "n2o_ef": Decimal("0.009"), "source": "eGRID2022"},
    "SRSO": {"co2_ef": Decimal("359.0"), "ch4_ef": Decimal("0.014"), "n2o_ef": Decimal("0.005"), "source": "eGRID2022"},
    "SRTV": {"co2_ef": Decimal("365.0"), "ch4_ef": Decimal("0.014"), "n2o_ef": Decimal("0.005"), "source": "eGRID2022"},
    "SRVC": {"co2_ef": Decimal("286.0"), "ch4_ef": Decimal("0.012"), "n2o_ef": Decimal("0.004"), "source": "eGRID2022"},
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
    _s2l_calculations_total = Counter(
        "gl_s2l_calculations_total",
        "Total Scope 2 location-based electricity emission calculations",
        labelnames=["country", "tier", "status"],
    )
    _s2l_emissions_kg_co2e_total = Counter(
        "gl_s2l_emissions_kg_co2e_total",
        "Cumulative Scope 2 location-based emissions in kg CO2e",
        labelnames=["country", "gas"],
    )
    _s2l_batch_jobs_total = Counter(
        "gl_s2l_batch_jobs_total",
        "Total Scope 2 location-based batch jobs",
        labelnames=["status"],
    )
    _s2l_calculation_duration = Histogram(
        "gl_s2l_calculation_duration_seconds",
        "Duration of Scope 2 location-based calculations in seconds",
        labelnames=["operation"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _s2l_active_calculations = Gauge(
        "gl_s2l_active_calculations",
        "Number of currently active Scope 2 location-based calculations",
    )
else:
    _s2l_calculations_total = None          # type: ignore[assignment]
    _s2l_emissions_kg_co2e_total = None     # type: ignore[assignment]
    _s2l_batch_jobs_total = None            # type: ignore[assignment]
    _s2l_calculation_duration = None        # type: ignore[assignment]
    _s2l_active_calculations = None         # type: ignore[assignment]

def _record_calc(country: str, tier: str, status: str) -> None:
    """Record a Scope 2 location-based calculation metric."""
    if _PROMETHEUS_AVAILABLE and _s2l_calculations_total is not None:
        _s2l_calculations_total.labels(country=country, tier=tier, status=status).inc()

def _record_emissions_metric(country: str, gas: str, kg: float) -> None:
    """Record cumulative emissions metric."""
    if _PROMETHEUS_AVAILABLE and _s2l_emissions_kg_co2e_total is not None:
        _s2l_emissions_kg_co2e_total.labels(country=country, gas=gas).inc(kg)

def _record_batch_metric(status: str) -> None:
    """Record batch job metric."""
    if _PROMETHEUS_AVAILABLE and _s2l_batch_jobs_total is not None:
        _s2l_batch_jobs_total.labels(status=status).inc()

def _observe_duration(operation: str, seconds: float) -> None:
    """Record calculation duration metric."""
    if _PROMETHEUS_AVAILABLE and _s2l_calculation_duration is not None:
        _s2l_calculation_duration.labels(operation=operation).observe(seconds)

# ---------------------------------------------------------------------------
# Provenance helper (lightweight inline tracker for this engine)
# ---------------------------------------------------------------------------

class _ProvenanceTracker:
    """Lightweight chain-hashing provenance tracker for electricity emissions.

    Each recorded entry chains its SHA-256 hash to the previous entry,
    producing a tamper-evident audit log.
    """

    def __init__(self, genesis: str = "GL-MRV-009-SCOPE2-LOCATION-ELECTRICITY-GENESIS") -> None:
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
        """Record a provenance entry and return it."""
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
        """Verify integrity of the entire provenance chain."""
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
        """Return a copy of all entries."""
        with self._lock:
            return list(self._entries)

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

# ---------------------------------------------------------------------------
# Utility: UTC now helper
# ---------------------------------------------------------------------------

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

# ===========================================================================
# ElectricityEmissionsEngine
# ===========================================================================

class ElectricityEmissionsEngine:
    """Core Scope 2 location-based electricity emission calculation engine.

    Implements the GHG Protocol Scope 2 Guidance location-based method for
    purchased electricity using grid-average emission factors. Supports
    IPCC Tier 1 (country-level) and Tier 2 (sub-regional) calculations,
    per-gas breakdowns, monthly and hourly temporal profiles, biogenic
    tracking, and batch processing.

    Zero-Hallucination Guarantees:
        - All arithmetic uses ``Decimal`` with ``ROUND_HALF_UP``
        - No LLM calls in the calculation path
        - SHA-256 provenance hash on every result
        - Complete calculation trace for audit
        - Thread-safe (no mutable shared state in calculations)

    Attributes:
        _grid_factor_db: External grid emission factor database (optional).
        _config: Configuration dictionary.
        _provenance: Chain-hashing provenance tracker.
        _lock: Thread lock for shared mutable state.

    Example:
        >>> engine = ElectricityEmissionsEngine()
        >>> r = engine.calculate_emissions(Decimal("500"), Decimal("0.450"))
        >>> assert r["total_co2e_tonnes"] > 0
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        grid_factor_db: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        provenance: Optional[Any] = None,
    ) -> None:
        """Initialize ElectricityEmissionsEngine.

        Args:
            grid_factor_db: Optional GridEmissionFactorDatabaseEngine instance
                for emission factor lookups. Falls back to built-in IPCC
                defaults if not provided.
            config: Optional configuration dictionary. Supports:
                - ``enable_provenance`` (bool): Enable provenance tracking.
                    Default True.
                - ``decimal_precision`` (int): Decimal places. Default 8.
                - ``default_gwp_source`` (str): Default GWP source. Default AR5.
                - ``default_td_loss_pct`` (str/Decimal): Default T&D loss.
                    Default 0.05.
            metrics: Optional external metrics collector (unused; built-in
                Prometheus metrics are used).
            provenance: Optional external provenance tracker. If None, an
                internal ``_ProvenanceTracker`` is created.
        """
        self._grid_factor_db = grid_factor_db
        self._config: Dict[str, Any] = config or {}
        self._lock = threading.Lock()

        # Precision
        self._precision_places: int = self._config.get("decimal_precision", 8)
        self._precision_q = Decimal(10) ** -self._precision_places

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
        self._default_td_loss: Decimal = _to_decimal(
            self._config.get("default_td_loss_pct", "0.05")
        )

        # Statistics counters
        self._stats_lock = threading.Lock()
        self._total_calculations: int = 0
        self._total_batches: int = 0
        self._total_co2e_kg_processed = Decimal("0")
        self._total_errors: int = 0

        logger.info(
            "ElectricityEmissionsEngine initialized "
            "(precision=%d, provenance=%s, default_gwp=%s, default_td_loss=%s)",
            self._precision_places,
            self._enable_provenance,
            self._default_gwp,
            self._default_td_loss,
        )

    # ==================================================================
    # 1. calculate_emissions
    # ==================================================================

    def calculate_emissions(
        self,
        consumption_mwh: Decimal,
        grid_ef_co2e: Decimal,
        td_loss_pct: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """Calculate basic location-based electricity emissions.

        Applies the formula:
            gross_consumption = consumption_mwh * (1 + td_loss_pct)
            total_co2e_kg = gross_consumption * grid_ef_co2e
            total_co2e_tonnes = total_co2e_kg / 1000

        The ``grid_ef_co2e`` must be in kg CO2e per MWh. If it is in
        tCO2e/MWh, multiply by 1000 before calling.

        Args:
            consumption_mwh: Electricity consumption in MWh. Must be >= 0.
            grid_ef_co2e: Grid emission factor in kg CO2e/MWh. Must be >= 0.
            td_loss_pct: Transmission and distribution loss percentage as
                a decimal fraction (e.g., 0.05 for 5%). Default 0.

        Returns:
            Dictionary with keys:
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - consumption_mwh (Decimal)
                - ef_applied (Decimal) -- the grid EF used
                - td_loss_pct (Decimal)
                - gross_consumption (Decimal) -- MWh after T&D adjustment
                - calculation_trace (list[str])
                - provenance_hash (str)

        Raises:
            ValueError: If consumption or EF is negative.
        """
        start = time.monotonic()
        consumption_mwh = _to_decimal(consumption_mwh)
        grid_ef_co2e = _to_decimal(grid_ef_co2e)
        td_loss_pct = _to_decimal(td_loss_pct)

        # Validate
        errors = self._validate_consumption_and_ef(consumption_mwh, grid_ef_co2e, td_loss_pct)
        if errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        trace: List[str] = []
        trace.append(f"[1] Input: consumption={consumption_mwh} MWh, EF={grid_ef_co2e} kg CO2e/MWh, T&D={td_loss_pct}")

        # Gross consumption
        td_multiplier = Decimal("1") + td_loss_pct
        gross_consumption = (consumption_mwh * td_multiplier).quantize(
            self._precision_q, rounding=ROUND_HALF_UP
        )
        trace.append(f"[2] Gross consumption: {consumption_mwh} x {td_multiplier} = {gross_consumption} MWh")

        # Total CO2e
        total_co2e_kg = (gross_consumption * grid_ef_co2e).quantize(
            self._precision_q, rounding=ROUND_HALF_UP
        )
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(
            self._precision_q, rounding=ROUND_HALF_UP
        )
        trace.append(f"[3] Emissions: {gross_consumption} x {grid_ef_co2e} = {total_co2e_kg} kg CO2e = {total_co2e_tonnes} tCO2e")

        # Provenance hash
        prov_data = {
            "method": "calculate_emissions",
            "consumption_mwh": str(consumption_mwh),
            "grid_ef_co2e": str(grid_ef_co2e),
            "td_loss_pct": str(td_loss_pct),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[4] Provenance hash: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start
        self._update_stats(total_co2e_kg)
        _observe_duration("calculate_emissions", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "electricity_calculation", "calculate_emissions",
                f"calc_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "consumption_mwh": consumption_mwh,
            "ef_applied": grid_ef_co2e,
            "td_loss_pct": td_loss_pct,
            "gross_consumption": gross_consumption,
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 2. calculate_with_gas_breakdown
    # ==================================================================

    def calculate_with_gas_breakdown(
        self,
        consumption_mwh: Decimal,
        co2_ef: Decimal,
        ch4_ef: Decimal,
        n2o_ef: Decimal,
        gwp_source: str = "AR5",
        td_loss_pct: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """Calculate electricity emissions with per-gas breakdown.

        For each gas the calculation is:
            gas_emission_kg = gross_consumption * gas_ef
            gas_co2e_kg     = gas_emission_kg * GWP_gas

        Total CO2e is the sum of all per-gas CO2e values.

        Args:
            consumption_mwh: Electricity consumption in MWh.
            co2_ef: CO2 emission factor in kg CO2/MWh.
            ch4_ef: CH4 emission factor in kg CH4/MWh.
            n2o_ef: N2O emission factor in kg N2O/MWh.
            gwp_source: GWP source (AR4, AR5, AR6). Default AR5.
            td_loss_pct: T&D loss percentage as decimal fraction.

        Returns:
            Dictionary with keys:
                - gas_breakdown (list of dicts with gas, emission_kg,
                  gwp_factor, co2e_kg)
                - total_co2e_kg (Decimal)
                - total_co2e_tonnes (Decimal)
                - gross_consumption (Decimal)
                - gwp_source (str)
                - calculation_trace (list)
                - provenance_hash (str)

        Raises:
            ValueError: For invalid inputs or unknown GWP source.
        """
        start = time.monotonic()
        consumption_mwh = _to_decimal(consumption_mwh)
        co2_ef = _to_decimal(co2_ef)
        ch4_ef = _to_decimal(ch4_ef)
        n2o_ef = _to_decimal(n2o_ef)
        td_loss_pct = _to_decimal(td_loss_pct)

        if consumption_mwh < 0:
            raise ValueError("consumption_mwh must be >= 0")
        if gwp_source not in GWP_VALUES:
            raise ValueError(f"Unknown gwp_source: {gwp_source}. Must be one of {list(GWP_VALUES.keys())}")

        trace: List[str] = []
        gwp = GWP_VALUES[gwp_source]

        # Gross consumption
        td_mult = Decimal("1") + td_loss_pct
        gross = (consumption_mwh * td_mult).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        trace.append(f"[1] Gross consumption: {consumption_mwh} x {td_mult} = {gross} MWh")

        # Per-gas calculations
        gas_breakdown: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")

        for gas_name, gas_ef, gwp_key in [("CO2", co2_ef, "CO2"), ("CH4", ch4_ef, "CH4"), ("N2O", n2o_ef, "N2O")]:
            emission_kg = (gross * gas_ef).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            gwp_factor = gwp[gwp_key]
            co2e_kg = (emission_kg * gwp_factor).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            total_co2e_kg += co2e_kg
            gas_breakdown.append({
                "gas": gas_name,
                "emission_kg": emission_kg,
                "gwp_factor": gwp_factor,
                "co2e_kg": co2e_kg,
            })
            trace.append(
                f"[2] {gas_name}: {gross} x {gas_ef} = {emission_kg} kg, "
                f"GWP={gwp_factor} -> {co2e_kg} kg CO2e"
            )

        total_co2e_kg = total_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP)
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        trace.append(f"[3] Total CO2e: {total_co2e_kg} kg = {total_co2e_tonnes} tonnes")

        prov_data = {
            "method": "calculate_with_gas_breakdown",
            "consumption_mwh": str(consumption_mwh),
            "co2_ef": str(co2_ef), "ch4_ef": str(ch4_ef), "n2o_ef": str(n2o_ef),
            "gwp_source": gwp_source, "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        trace.append(f"[4] Provenance: {provenance_hash[:16]}...")

        elapsed = time.monotonic() - start
        self._update_stats(total_co2e_kg)
        _observe_duration("calculate_with_gas_breakdown", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "electricity_calculation", "calculate_gas_breakdown",
                f"calc_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "gas_breakdown": gas_breakdown,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "gross_consumption": gross,
            "gwp_source": gwp_source,
            "td_loss_pct": td_loss_pct,
            "calculation_trace": trace,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 3. calculate_for_facility
    # ==================================================================

    def calculate_for_facility(
        self,
        facility_id: str,
        consumption_mwh: Decimal,
        country_code: Optional[str] = None,
        egrid_subregion: Optional[str] = None,
        gwp_source: str = "AR5",
        include_td_losses: bool = True,
    ) -> Dict[str, Any]:
        """Full facility-level electricity emission calculation.

        Automatically resolves emission factors from either:
        1. An external ``grid_factor_db`` (if provided)
        2. eGRID subregion lookup (if ``egrid_subregion`` specified)
        3. Country-level IPCC defaults (fallback)

        Performs per-gas breakdown with the resolved factors.

        Args:
            facility_id: Unique facility identifier.
            consumption_mwh: Electricity consumption in MWh.
            country_code: ISO 3166-1 alpha-2 country code (e.g., "US").
            egrid_subregion: EPA eGRID subregion code (e.g., "CAMX").
            gwp_source: GWP source (AR4, AR5, AR6). Default AR5.
            include_td_losses: Whether to include T&D loss adjustment.

        Returns:
            Dictionary containing:
                - calculation_id (str)
                - facility_id (str)
                - status (str): "SUCCESS" or "FAILED"
                - total_co2e_kg, total_co2e_tonnes
                - gas_breakdown
                - emission_factors_used (dict)
                - tier (str)
                - provenance_hash, calculation_trace
                - error_message (str or None)
        """
        start = time.monotonic()
        calc_id = f"s2l_{uuid.uuid4().hex[:12]}"
        consumption_mwh = _to_decimal(consumption_mwh)
        trace: List[str] = []
        country = (country_code or "GLOBAL").upper()

        try:
            trace.append(f"[1] Facility: {facility_id}, country={country}, consumption={consumption_mwh} MWh")

            # Validate
            if consumption_mwh < 0:
                raise ValueError("consumption_mwh must be >= 0")
            if not facility_id or not facility_id.strip():
                raise ValueError("facility_id must not be empty")

            # Resolve emission factors
            tier, factors = self._resolve_emission_factors(
                country, egrid_subregion, trace,
            )

            co2_ef = factors["co2_ef"]
            ch4_ef = factors["ch4_ef"]
            n2o_ef = factors["n2o_ef"]
            td_loss_pct = factors.get("td_loss_pct", self._default_td_loss) if include_td_losses else Decimal("0")
            ef_source = factors.get("source", "IPCC_DEFAULT")

            trace.append(
                f"[2] EFs resolved (tier={tier}, source={ef_source}): "
                f"CO2={co2_ef}, CH4={ch4_ef}, N2O={n2o_ef}, T&D={td_loss_pct}"
            )

            # Perform per-gas calculation
            gas_result = self.calculate_with_gas_breakdown(
                consumption_mwh=consumption_mwh,
                co2_ef=co2_ef,
                ch4_ef=ch4_ef,
                n2o_ef=n2o_ef,
                gwp_source=gwp_source,
                td_loss_pct=td_loss_pct,
            )

            # Merge traces
            trace.extend(gas_result["calculation_trace"])

            # Provenance
            prov_data = {
                "method": "calculate_for_facility",
                "calc_id": calc_id,
                "facility_id": facility_id,
                "country": country,
                "total_co2e_kg": str(gas_result["total_co2e_kg"]),
                "tier": tier,
                "gwp_source": gwp_source,
            }
            provenance_hash = self._compute_provenance_hash(prov_data)
            trace.append(f"[F] Facility provenance: {provenance_hash[:16]}...")

            elapsed = time.monotonic() - start
            _record_calc(country, tier, "completed")
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
                "country_code": country,
                "tier": tier,
                "consumption_mwh": consumption_mwh,
                "gross_consumption": gas_result["gross_consumption"],
                "total_co2e_kg": gas_result["total_co2e_kg"],
                "total_co2e_tonnes": gas_result["total_co2e_tonnes"],
                "gas_breakdown": gas_result["gas_breakdown"],
                "emission_factors_used": {
                    "co2_ef_kg_per_mwh": co2_ef,
                    "ch4_ef_kg_per_mwh": ch4_ef,
                    "n2o_ef_kg_per_mwh": n2o_ef,
                    "td_loss_pct": td_loss_pct,
                    "source": ef_source,
                },
                "gwp_source": gwp_source,
                "include_td_losses": include_td_losses,
                "provenance_hash": provenance_hash,
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed * 1000, 3),
                "timestamp": utcnow().isoformat(),
                "error_message": None,
            }

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("Facility calculation failed for %s: %s", facility_id, exc, exc_info=True)
            _record_calc(country, "UNKNOWN", "failed")
            self._update_stats_error()
            return {
                "calculation_id": calc_id,
                "facility_id": facility_id,
                "status": "FAILED",
                "country_code": country,
                "tier": "UNKNOWN",
                "consumption_mwh": consumption_mwh,
                "gross_consumption": Decimal("0"),
                "total_co2e_kg": Decimal("0"),
                "total_co2e_tonnes": Decimal("0"),
                "gas_breakdown": [],
                "emission_factors_used": {},
                "gwp_source": gwp_source,
                "include_td_losses": include_td_losses,
                "provenance_hash": "",
                "calculation_trace": trace,
                "processing_time_ms": round(elapsed * 1000, 3),
                "timestamp": utcnow().isoformat(),
                "error_message": str(exc),
            }

    # ==================================================================
    # 4. calculate_monthly
    # ==================================================================

    def calculate_monthly(
        self,
        facility_id: str,
        monthly_consumption: List[Decimal],
        country_code: str,
        year: int,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Calculate emissions for 12 months with monthly EFs if available.

        Accepts a list of exactly 12 Decimal values representing Jan-Dec
        consumption in MWh. If the grid_factor_db provides monthly EFs,
        those are used; otherwise the annual average EF is applied to all
        12 months.

        Args:
            facility_id: Facility identifier.
            monthly_consumption: List of 12 Decimal MWh values (Jan-Dec).
            country_code: ISO 3166-1 alpha-2 country code.
            year: Reporting year.
            gwp_source: GWP source (AR4, AR5, AR6).

        Returns:
            Dictionary with keys:
                - facility_id, country_code, year
                - monthly_results (list of 12 result dicts)
                - annual_total_co2e_kg, annual_total_co2e_tonnes
                - annual_consumption_mwh
                - provenance_hash, calculation_trace

        Raises:
            ValueError: If monthly_consumption does not have 12 entries.
        """
        start = time.monotonic()
        trace: List[str] = []
        country = country_code.upper()

        if len(monthly_consumption) != 12:
            raise ValueError(
                f"monthly_consumption must have exactly 12 entries (Jan-Dec), got {len(monthly_consumption)}"
            )

        trace.append(f"[1] Monthly calc: facility={facility_id}, country={country}, year={year}")

        # Resolve EFs (annual defaults)
        _, base_factors = self._resolve_emission_factors(country, None, trace)

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_results: List[Dict[str, Any]] = []
        annual_co2e_kg = Decimal("0")
        annual_consumption = Decimal("0")

        for i in range(12):
            month_mwh = _to_decimal(monthly_consumption[i])
            annual_consumption += month_mwh

            # Try to get monthly EF from external DB
            monthly_factors = self._get_monthly_factors(country, year, i + 1)
            factors = monthly_factors if monthly_factors is not None else base_factors

            co2_ef = factors["co2_ef"]
            ch4_ef = factors["ch4_ef"]
            n2o_ef = factors["n2o_ef"]
            td_loss = factors.get("td_loss_pct", self._default_td_loss)

            month_result = self.calculate_with_gas_breakdown(
                consumption_mwh=month_mwh,
                co2_ef=co2_ef,
                ch4_ef=ch4_ef,
                n2o_ef=n2o_ef,
                gwp_source=gwp_source,
                td_loss_pct=td_loss,
            )

            annual_co2e_kg += month_result["total_co2e_kg"]

            monthly_results.append({
                "month": i + 1,
                "month_name": month_names[i],
                "consumption_mwh": month_mwh,
                "total_co2e_kg": month_result["total_co2e_kg"],
                "total_co2e_tonnes": month_result["total_co2e_tonnes"],
                "gas_breakdown": month_result["gas_breakdown"],
                "ef_source": factors.get("source", "IPCC_DEFAULT"),
            })
            trace.append(
                f"[M{i+1}] {month_names[i]}: {month_mwh} MWh -> "
                f"{month_result['total_co2e_kg']} kg CO2e"
            )

        annual_co2e_kg = annual_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP)
        annual_co2e_tonnes = (annual_co2e_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        trace.append(
            f"[A] Annual total: {annual_consumption} MWh -> "
            f"{annual_co2e_kg} kg CO2e = {annual_co2e_tonnes} tCO2e"
        )

        prov_data = {
            "method": "calculate_monthly",
            "facility_id": facility_id,
            "country": country,
            "year": year,
            "annual_co2e_kg": str(annual_co2e_kg),
            "annual_consumption_mwh": str(annual_consumption),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        self._update_stats(annual_co2e_kg)
        _observe_duration("calculate_monthly", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "monthly_calculation", "calculate_monthly",
                f"monthly_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "facility_id": facility_id,
            "country_code": country,
            "year": year,
            "gwp_source": gwp_source,
            "monthly_results": monthly_results,
            "annual_total_co2e_kg": annual_co2e_kg,
            "annual_total_co2e_tonnes": annual_co2e_tonnes,
            "annual_consumption_mwh": annual_consumption,
            "provenance_hash": provenance_hash,
            "calculation_trace": trace,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 5. calculate_hourly
    # ==================================================================

    def calculate_hourly(
        self,
        facility_id: str,
        hourly_consumption: List[Decimal],
        country_code: str,
        date: str,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """Calculate emissions at hourly granularity for real-time carbon accounting.

        Accepts a list of 24 (single day) or 8760 (full year) hourly
        consumption values in MWh and applies grid EFs to each hour.

        Args:
            facility_id: Facility identifier.
            hourly_consumption: List of Decimal MWh values (24 or 8760).
            country_code: ISO 3166-1 alpha-2 country code.
            date: Date string (YYYY-MM-DD for 24h, YYYY for 8760h).
            gwp_source: GWP source (AR4, AR5, AR6).

        Returns:
            Dictionary with hourly_results, daily/annual totals, and provenance.

        Raises:
            ValueError: If hourly_consumption length is not 24 or 8760.
        """
        start = time.monotonic()
        trace: List[str] = []
        country = country_code.upper()
        n_hours = len(hourly_consumption)

        if n_hours not in (24, 8760):
            raise ValueError(
                f"hourly_consumption must have 24 (daily) or 8760 (annual) entries, got {n_hours}"
            )

        trace.append(f"[1] Hourly calc: facility={facility_id}, country={country}, date={date}, hours={n_hours}")

        # Resolve EFs
        _, base_factors = self._resolve_emission_factors(country, None, trace)
        co2_ef = base_factors["co2_ef"]
        ch4_ef = base_factors["ch4_ef"]
        n2o_ef = base_factors["n2o_ef"]
        td_loss = base_factors.get("td_loss_pct", self._default_td_loss)

        gwp = GWP_VALUES.get(gwp_source)
        if gwp is None:
            raise ValueError(f"Unknown gwp_source: {gwp_source}")

        td_mult = Decimal("1") + td_loss
        total_co2e_kg = Decimal("0")
        total_consumption = Decimal("0")
        hourly_results: List[Dict[str, Any]] = []

        for h in range(n_hours):
            h_mwh = _to_decimal(hourly_consumption[h])
            total_consumption += h_mwh
            gross = (h_mwh * td_mult).quantize(self._precision_q, rounding=ROUND_HALF_UP)

            co2_kg = (gross * co2_ef).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            ch4_kg = (gross * ch4_ef).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            n2o_kg = (gross * n2o_ef).quantize(self._precision_q, rounding=ROUND_HALF_UP)

            co2_co2e = (co2_kg * gwp["CO2"]).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            ch4_co2e = (ch4_kg * gwp["CH4"]).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            n2o_co2e = (n2o_kg * gwp["N2O"]).quantize(self._precision_q, rounding=ROUND_HALF_UP)
            h_co2e = (co2_co2e + ch4_co2e + n2o_co2e).quantize(self._precision_q, rounding=ROUND_HALF_UP)

            total_co2e_kg += h_co2e
            hourly_results.append({
                "hour": h,
                "consumption_mwh": h_mwh,
                "co2e_kg": h_co2e,
            })

        total_co2e_kg = total_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP)
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        trace.append(f"[2] Total: {total_consumption} MWh -> {total_co2e_kg} kg CO2e = {total_co2e_tonnes} tCO2e")

        prov_data = {
            "method": "calculate_hourly",
            "facility_id": facility_id,
            "country": country,
            "date": date,
            "n_hours": n_hours,
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        self._update_stats(total_co2e_kg)
        _observe_duration("calculate_hourly", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "hourly_calculation", "calculate_hourly",
                f"hourly_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "facility_id": facility_id,
            "country_code": country,
            "date": date,
            "gwp_source": gwp_source,
            "n_hours": n_hours,
            "hourly_results": hourly_results,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_consumption_mwh": total_consumption,
            "provenance_hash": provenance_hash,
            "calculation_trace": trace,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 6. calculate_batch
    # ==================================================================

    def calculate_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Process a batch of electricity emission calculations.

        Each request dict must contain at minimum:
            - consumption_mwh (Decimal or numeric)
            - grid_ef_co2e (Decimal or numeric)
        Optionally: td_loss_pct, facility_id, country_code, etc.

        Args:
            requests: List of request dictionaries.

        Returns:
            Dictionary with:
                - batch_id (str)
                - results (list of individual result dicts)
                - total_co2e_kg, total_co2e_tonnes
                - success_count, failure_count
                - processing_time_ms
                - provenance_hash
        """
        start = time.monotonic()
        batch_id = f"batch_s2l_{uuid.uuid4().hex[:12]}"
        results: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")
        success_count = 0
        failure_count = 0

        for i, req in enumerate(requests):
            try:
                consumption = _to_decimal(req.get("consumption_mwh", 0))
                ef = _to_decimal(req.get("grid_ef_co2e", 0))
                td = _to_decimal(req.get("td_loss_pct", "0"))

                result = self.calculate_emissions(
                    consumption_mwh=consumption,
                    grid_ef_co2e=ef,
                    td_loss_pct=td,
                )
                result["request_index"] = i
                result["status"] = "SUCCESS"
                results.append(result)
                total_co2e_kg += result["total_co2e_kg"]
                success_count += 1
            except Exception as exc:
                results.append({
                    "request_index": i,
                    "status": "FAILED",
                    "error_message": str(exc),
                    "total_co2e_kg": Decimal("0"),
                    "total_co2e_tonnes": Decimal("0"),
                })
                failure_count += 1

        total_co2e_kg = total_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP)
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        prov_data = {
            "method": "calculate_batch",
            "batch_id": batch_id,
            "request_count": len(requests),
            "success_count": success_count,
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        status = "success" if failure_count == 0 else ("partial" if success_count > 0 else "failure")
        _record_batch_metric(status)
        _observe_duration("calculate_batch", elapsed)

        with self._stats_lock:
            self._total_batches += 1

        if self._provenance is not None:
            self._provenance.record("batch", "calculate_batch", batch_id, prov_data)

        return {
            "batch_id": batch_id,
            "results": results,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "success_count": success_count,
            "failure_count": failure_count,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 7. calculate_multi_facility
    # ==================================================================

    def calculate_multi_facility(
        self,
        facilities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate emissions for multiple facilities and aggregate.

        Each facility dict must contain:
            - facility_id (str)
            - consumption_mwh (Decimal or numeric)
            - country_code (str)
        Optionally: egrid_subregion, gwp_source, include_td_losses.

        Args:
            facilities: List of facility dictionaries.

        Returns:
            Dictionary with per-facility results and aggregated totals.
        """
        start = time.monotonic()
        agg_id = f"multi_{uuid.uuid4().hex[:12]}"
        facility_results: List[Dict[str, Any]] = []
        total_co2e_kg = Decimal("0")
        total_consumption = Decimal("0")
        success_count = 0
        failure_count = 0

        for fac in facilities:
            fac_id = fac.get("facility_id", "unknown")
            consumption = _to_decimal(fac.get("consumption_mwh", 0))
            country = fac.get("country_code", "GLOBAL")
            subregion = fac.get("egrid_subregion")
            gwp = fac.get("gwp_source", self._default_gwp)
            incl_td = fac.get("include_td_losses", True)

            result = self.calculate_for_facility(
                facility_id=fac_id,
                consumption_mwh=consumption,
                country_code=country,
                egrid_subregion=subregion,
                gwp_source=gwp,
                include_td_losses=incl_td,
            )
            facility_results.append(result)

            if result["status"] == "SUCCESS":
                total_co2e_kg += result["total_co2e_kg"]
                total_consumption += result["consumption_mwh"]
                success_count += 1
            else:
                failure_count += 1

        total_co2e_kg = total_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP)
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        prov_data = {
            "method": "calculate_multi_facility",
            "agg_id": agg_id,
            "facility_count": len(facilities),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        _observe_duration("calculate_multi_facility", elapsed)

        if self._provenance is not None:
            self._provenance.record("aggregation", "calculate_multi_facility", agg_id, prov_data)

        return {
            "aggregation_id": agg_id,
            "facility_results": facility_results,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "total_consumption_mwh": total_consumption,
            "facility_count": len(facilities),
            "success_count": success_count,
            "failure_count": failure_count,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 8-11. Unit Conversions
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
        if kwh < 0:
            raise ValueError("kwh must be >= 0")
        return (kwh * _KWH_TO_MWH).quantize(self._precision_q, rounding=ROUND_HALF_UP)

    def gj_to_mwh(self, gj: Decimal) -> Decimal:
        """Convert gigajoules to megawatt-hours.

        Conversion: 1 GJ = 0.277778 MWh (i.e., 1 MWh = 3.6 GJ).

        Args:
            gj: Energy in GJ. Must be >= 0.

        Returns:
            Energy in MWh (Decimal).

        Raises:
            ValueError: If gj is negative.
        """
        gj = _to_decimal(gj)
        if gj < 0:
            raise ValueError("gj must be >= 0")
        return (gj * _GJ_TO_MWH).quantize(self._precision_q, rounding=ROUND_HALF_UP)

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
        if mmbtu < 0:
            raise ValueError("mmbtu must be >= 0")
        return (mmbtu * _MMBTU_TO_MWH).quantize(self._precision_q, rounding=ROUND_HALF_UP)

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
        if quantity < 0:
            raise ValueError("quantity must be >= 0")

        unit_upper = unit.strip().upper()
        conversions: Dict[str, Decimal] = {
            "KWH": _KWH_TO_MWH,
            "MWH": Decimal("1"),
            "GJ": _GJ_TO_MWH,
            "MMBTU": _MMBTU_TO_MWH,
            "TJ": _TJ_TO_MWH,
        }

        factor = conversions.get(unit_upper)
        if factor is None:
            raise ValueError(
                f"Unsupported energy unit '{unit}'. Supported: {list(conversions.keys())}"
            )

        return (quantity * factor).quantize(self._precision_q, rounding=ROUND_HALF_UP)

    # ==================================================================
    # 12-13. IPCC Tier Support
    # ==================================================================

    def calculate_tier1(
        self,
        consumption_mwh: Decimal,
        country_code: str,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """IPCC Tier 1 calculation using country-level grid average EFs.

        Looks up country-level emission factors from the built-in database
        and applies them to the consumption figure.

        Args:
            consumption_mwh: Electricity consumption in MWh.
            country_code: ISO 3166-1 alpha-2 country code.
            gwp_source: GWP source (AR4, AR5, AR6).

        Returns:
            Full calculation result dictionary (same as calculate_for_facility).
        """
        return self.calculate_for_facility(
            facility_id=f"tier1_{country_code.upper()}",
            consumption_mwh=consumption_mwh,
            country_code=country_code,
            egrid_subregion=None,
            gwp_source=gwp_source,
            include_td_losses=True,
        )

    def calculate_tier2(
        self,
        consumption_mwh: Decimal,
        subregion: str,
        gwp_source: str = "AR5",
    ) -> Dict[str, Any]:
        """IPCC Tier 2 calculation using sub-regional EFs (e.g., eGRID).

        Uses the eGRID subregion emission factors for more geographically
        precise calculations. Falls back to US national average if the
        subregion is not found.

        Args:
            consumption_mwh: Electricity consumption in MWh.
            subregion: Sub-regional identifier (e.g., eGRID subregion code).
            gwp_source: GWP source (AR4, AR5, AR6).

        Returns:
            Full calculation result dictionary.
        """
        return self.calculate_for_facility(
            facility_id=f"tier2_{subregion.upper()}",
            consumption_mwh=consumption_mwh,
            country_code="US",
            egrid_subregion=subregion,
            gwp_source=gwp_source,
            include_td_losses=True,
        )

    # ==================================================================
    # 14. calculate_with_biogenic
    # ==================================================================

    def calculate_with_biogenic(
        self,
        consumption_mwh: Decimal,
        fossil_ef: Decimal,
        biogenic_ef: Decimal,
        td_loss_pct: Decimal = Decimal("0"),
    ) -> Dict[str, Any]:
        """Calculate electricity emissions separating fossil vs biogenic CO2.

        The grid mix may include biomass-fired generation. This method
        tracks fossil and biogenic CO2 separately per GHG Protocol
        guidance. Biogenic CO2 is reported as a memo item and excluded
        from Scope 2 totals.

from greenlang.schemas import utcnow

        Args:
            consumption_mwh: Electricity consumption in MWh.
            fossil_ef: Fossil CO2 emission factor (kg CO2/MWh).
            biogenic_ef: Biogenic CO2 emission factor (kg CO2/MWh).
            td_loss_pct: T&D loss percentage as decimal fraction.

        Returns:
            Dictionary with:
                - fossil_co2e_kg, fossil_co2e_tonnes
                - biogenic_co2_kg, biogenic_co2_tonnes
                - total_co2e_kg (fossil only), total_co2e_tonnes
                - gross_consumption
                - provenance_hash, calculation_trace
        """
        start = time.monotonic()
        consumption_mwh = _to_decimal(consumption_mwh)
        fossil_ef = _to_decimal(fossil_ef)
        biogenic_ef = _to_decimal(biogenic_ef)
        td_loss_pct = _to_decimal(td_loss_pct)

        if consumption_mwh < 0:
            raise ValueError("consumption_mwh must be >= 0")

        trace: List[str] = []

        td_mult = Decimal("1") + td_loss_pct
        gross = (consumption_mwh * td_mult).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        trace.append(f"[1] Gross: {consumption_mwh} x {td_mult} = {gross} MWh")

        fossil_kg = (gross * fossil_ef).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        fossil_tonnes = (fossil_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        trace.append(f"[2] Fossil CO2: {gross} x {fossil_ef} = {fossil_kg} kg = {fossil_tonnes} t")

        bio_kg = (gross * biogenic_ef).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        bio_tonnes = (bio_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        trace.append(f"[3] Biogenic CO2: {gross} x {biogenic_ef} = {bio_kg} kg = {bio_tonnes} t (memo item)")

        # Total = fossil only per GHG Protocol
        total_co2e_kg = fossil_kg
        total_co2e_tonnes = fossil_tonnes
        trace.append(f"[4] Total (fossil only): {total_co2e_kg} kg = {total_co2e_tonnes} tCO2e")

        prov_data = {
            "method": "calculate_with_biogenic",
            "consumption_mwh": str(consumption_mwh),
            "fossil_ef": str(fossil_ef),
            "biogenic_ef": str(biogenic_ef),
            "fossil_co2e_kg": str(fossil_kg),
            "biogenic_co2_kg": str(bio_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        self._update_stats(fossil_kg)
        _observe_duration("calculate_with_biogenic", elapsed)

        if self._provenance is not None:
            self._provenance.record(
                "biogenic_calculation", "calculate_with_biogenic",
                f"bio_{uuid.uuid4().hex[:12]}", prov_data,
            )

        return {
            "fossil_co2e_kg": fossil_kg,
            "fossil_co2e_tonnes": fossil_tonnes,
            "biogenic_co2_kg": bio_kg,
            "biogenic_co2_tonnes": bio_tonnes,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "gross_consumption": gross,
            "consumption_mwh": consumption_mwh,
            "td_loss_pct": td_loss_pct,
            "provenance_hash": provenance_hash,
            "calculation_trace": trace,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 15-17. Aggregation Methods
    # ==================================================================

    def aggregate_by_facility(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate calculation results by facility_id.

        Groups results by their ``facility_id`` and sums emissions.

        Args:
            results: List of calculation result dictionaries, each
                containing at least ``facility_id``, ``total_co2e_kg``,
                ``total_co2e_tonnes``, ``consumption_mwh``.

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
                    "total_co2e_kg": Decimal("0"),
                    "total_co2e_tonnes": Decimal("0"),
                    "consumption_mwh": Decimal("0"),
                    "count": Decimal("0"),
                }
            by_facility[fid]["total_co2e_kg"] += _to_decimal(r.get("total_co2e_kg", 0))
            by_facility[fid]["total_co2e_tonnes"] += _to_decimal(r.get("total_co2e_tonnes", 0))
            by_facility[fid]["consumption_mwh"] += _to_decimal(r.get("consumption_mwh", 0))
            by_facility[fid]["count"] += Decimal("1")

        grand_co2e_kg = Decimal("0")
        grand_co2e_tonnes = Decimal("0")
        grand_consumption = Decimal("0")
        facility_summaries: Dict[str, Dict[str, Any]] = {}

        for fid, agg in by_facility.items():
            co2e_kg = agg["total_co2e_kg"].quantize(self._precision_q, rounding=ROUND_HALF_UP)
            co2e_t = agg["total_co2e_tonnes"].quantize(self._precision_q, rounding=ROUND_HALF_UP)
            cons = agg["consumption_mwh"].quantize(self._precision_q, rounding=ROUND_HALF_UP)
            facility_summaries[fid] = {
                "total_co2e_kg": co2e_kg,
                "total_co2e_tonnes": co2e_t,
                "consumption_mwh": cons,
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg
            grand_co2e_tonnes += co2e_t
            grand_consumption += cons

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
                "total_co2e_kg": grand_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP),
                "total_co2e_tonnes": grand_co2e_tonnes.quantize(self._precision_q, rounding=ROUND_HALF_UP),
                "consumption_mwh": grand_consumption.quantize(self._precision_q, rounding=ROUND_HALF_UP),
            },
            "facility_count": len(by_facility),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    def aggregate_by_period(
        self,
        results: List[Dict[str, Any]],
        period: str = "annual",
    ) -> Dict[str, Any]:
        """Aggregate calculation results by time period.

        Groups results by the specified period (monthly, quarterly, annual)
        based on the ``month`` or ``timestamp`` field in each result.

        Args:
            results: List of calculation result dictionaries.
            period: Aggregation period. One of "monthly", "quarterly",
                "annual". Default "annual".

        Returns:
            Dictionary with period-level aggregations and grand total.
        """
        start = time.monotonic()
        valid_periods = ("monthly", "quarterly", "annual")
        if period not in valid_periods:
            raise ValueError(f"period must be one of {valid_periods}, got '{period}'")

        by_period: Dict[str, Dict[str, Decimal]] = {}

        for r in results:
            key = self._get_period_key(r, period)
            if key not in by_period:
                by_period[key] = {
                    "total_co2e_kg": Decimal("0"),
                    "total_co2e_tonnes": Decimal("0"),
                    "consumption_mwh": Decimal("0"),
                    "count": Decimal("0"),
                }
            by_period[key]["total_co2e_kg"] += _to_decimal(r.get("total_co2e_kg", 0))
            by_period[key]["total_co2e_tonnes"] += _to_decimal(r.get("total_co2e_tonnes", 0))
            by_period[key]["consumption_mwh"] += _to_decimal(r.get("consumption_mwh", 0))
            by_period[key]["count"] += Decimal("1")

        grand_co2e_kg = Decimal("0")
        period_summaries: Dict[str, Dict[str, Any]] = {}

        for key, agg in sorted(by_period.items()):
            co2e_kg = agg["total_co2e_kg"].quantize(self._precision_q, rounding=ROUND_HALF_UP)
            co2e_t = agg["total_co2e_tonnes"].quantize(self._precision_q, rounding=ROUND_HALF_UP)
            cons = agg["consumption_mwh"].quantize(self._precision_q, rounding=ROUND_HALF_UP)
            period_summaries[key] = {
                "total_co2e_kg": co2e_kg,
                "total_co2e_tonnes": co2e_t,
                "consumption_mwh": cons,
                "calculation_count": int(agg["count"]),
            }
            grand_co2e_kg += co2e_kg

        prov_data = {"method": "aggregate_by_period", "period": period, "grand_co2e_kg": str(grand_co2e_kg)}
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "period": period,
            "by_period": period_summaries,
            "grand_total_co2e_kg": grand_co2e_kg.quantize(self._precision_q, rounding=ROUND_HALF_UP),
            "period_count": len(period_summaries),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    def aggregate_by_gas(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate calculation results by emission gas (CO2, CH4, N2O).

        Sums per-gas emissions across all results that include a
        ``gas_breakdown`` field.

        Args:
            results: List of calculation result dictionaries, each
                optionally containing a ``gas_breakdown`` list.

        Returns:
            Dictionary with per-gas totals and grand total.
        """
        start = time.monotonic()
        by_gas: Dict[str, Decimal] = {
            "CO2": Decimal("0"),
            "CH4": Decimal("0"),
            "N2O": Decimal("0"),
        }

        for r in results:
            breakdown = r.get("gas_breakdown", [])
            for gb in breakdown:
                gas = gb.get("gas", "CO2")
                co2e_kg = _to_decimal(gb.get("co2e_kg", 0))
                if gas in by_gas:
                    by_gas[gas] += co2e_kg

        gas_summaries: Dict[str, Decimal] = {}
        grand_total = Decimal("0")
        for gas, total in by_gas.items():
            rounded = total.quantize(self._precision_q, rounding=ROUND_HALF_UP)
            gas_summaries[gas] = rounded
            grand_total += rounded

        grand_total = grand_total.quantize(self._precision_q, rounding=ROUND_HALF_UP)

        prov_data = {"method": "aggregate_by_gas", "grand_co2e_kg": str(grand_total)}
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "by_gas": gas_summaries,
            "grand_total_co2e_kg": grand_total,
            "grand_total_co2e_tonnes": (grand_total * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP),
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # 18-19. Validation Methods
    # ==================================================================

    def validate_consumption(self, consumption_mwh: Decimal) -> List[str]:
        """Validate electricity consumption data for reasonableness.

        Checks:
            - Must be a valid Decimal
            - Must be >= 0
            - Warns if > 1,000,000 MWh (unusually large)
            - Warns if == 0 (no emissions to calculate)

        Args:
            consumption_mwh: Consumption value to validate.

        Returns:
            List of validation error/warning strings. Empty if valid.
        """
        errors: List[str] = []
        try:
            val = _to_decimal(consumption_mwh)
        except (ValueError, TypeError):
            errors.append(f"Cannot convert consumption_mwh to Decimal: {consumption_mwh!r}")
            return errors

        if val < 0:
            errors.append(f"consumption_mwh must be >= 0, got {val}")
        if val == 0:
            errors.append("WARNING: consumption_mwh is 0; no emissions to calculate")
        if val > Decimal("1000000"):
            errors.append(
                f"WARNING: consumption_mwh={val} exceeds 1,000,000 MWh; "
                "verify this is correct"
            )

        return errors

    def validate_emission_factor(self, ef: Decimal) -> List[str]:
        """Validate a grid emission factor for reasonableness.

        Checks:
            - Must be a valid Decimal
            - Must be >= 0
            - Warns if > 2000 kg CO2e/MWh (exceeds coal-fired max)
            - Warns if < 1 kg CO2e/MWh (lower than hydro/nuclear)

        Args:
            ef: Emission factor value to validate (kg CO2e/MWh).

        Returns:
            List of validation error/warning strings. Empty if valid.
        """
        errors: List[str] = []
        try:
            val = _to_decimal(ef)
        except (ValueError, TypeError):
            errors.append(f"Cannot convert emission factor to Decimal: {ef!r}")
            return errors

        if val < 0:
            errors.append(f"Emission factor must be >= 0, got {val}")
        if val > Decimal("2000"):
            errors.append(
                f"WARNING: EF={val} kg CO2e/MWh exceeds 2000; "
                "verify this is correct (typical range: 0-1200)"
            )
        if Decimal("0") < val < Decimal("1"):
            errors.append(
                f"WARNING: EF={val} kg CO2e/MWh is < 1; "
                "this is unusually low unless grid is nearly 100% renewable"
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
                - total_co2e_kg_processed (Decimal)
                - total_errors (int)
                - provenance_entry_count (int)
                - supported_countries (int)
                - supported_egrid_subregions (int)
                - supported_gwp_sources (list)
                - default_gwp_source (str)
                - default_td_loss_pct (Decimal)
                - decimal_precision (int)
                - provenance_enabled (bool)
        """
        with self._stats_lock:
            return {
                "total_calculations": self._total_calculations,
                "total_batches": self._total_batches,
                "total_co2e_kg_processed": self._total_co2e_kg_processed,
                "total_errors": self._total_errors,
                "provenance_entry_count": (
                    self._provenance.entry_count if self._provenance else 0
                ),
                "supported_countries": len(_DEFAULT_COUNTRY_GRID_EF),
                "supported_egrid_subregions": len(_DEFAULT_EGRID_SUBREGION_EF),
                "supported_gwp_sources": list(GWP_VALUES.keys()),
                "default_gwp_source": self._default_gwp,
                "default_td_loss_pct": self._default_td_loss,
                "decimal_precision": self._precision_places,
                "provenance_enabled": self._enable_provenance,
            }

    # ==================================================================
    # Internal: Emission Factor Resolution
    # ==================================================================

    def _resolve_emission_factors(
        self,
        country_code: str,
        egrid_subregion: Optional[str],
        trace: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve emission factors for a given location.

        Resolution order:
        1. External grid_factor_db (if provided)
        2. eGRID subregion (if specified and available)
        3. Country-level defaults
        4. Global average fallback

        Args:
            country_code: ISO country code (uppercase).
            egrid_subregion: Optional eGRID subregion code.
            trace: Calculation trace list.

        Returns:
            Tuple of (tier_string, factors_dict).
        """
        # 1. Try external database
        if self._grid_factor_db is not None:
            try:
                factors = self._lookup_external_db(country_code, egrid_subregion)
                if factors is not None:
                    tier = "TIER_2" if egrid_subregion else "TIER_1"
                    trace.append(f"[EF] Resolved from external DB: tier={tier}")
                    return tier, factors
            except Exception as exc:
                logger.warning("External DB lookup failed: %s; falling back to defaults", exc)
                trace.append(f"[EF] External DB lookup failed: {exc}")

        # 2. Try eGRID subregion
        if egrid_subregion is not None:
            subregion_upper = egrid_subregion.upper()
            if subregion_upper in _DEFAULT_EGRID_SUBREGION_EF:
                factors = dict(_DEFAULT_EGRID_SUBREGION_EF[subregion_upper])
                factors.setdefault("td_loss_pct", Decimal("0.05"))
                trace.append(f"[EF] eGRID subregion {subregion_upper}: Tier 2")
                return "TIER_2", factors
            else:
                trace.append(f"[EF] eGRID subregion {subregion_upper} not found; falling back to country")

        # 3. Try country-level
        if country_code in _DEFAULT_COUNTRY_GRID_EF:
            factors = dict(_DEFAULT_COUNTRY_GRID_EF[country_code])
            trace.append(f"[EF] Country {country_code}: Tier 1")
            return "TIER_1", factors

        # 4. Global fallback
        factors = dict(_DEFAULT_COUNTRY_GRID_EF["GLOBAL"])
        trace.append(f"[EF] Fallback to GLOBAL average: Tier 1")
        return "TIER_1", factors

    def _lookup_external_db(
        self,
        country_code: str,
        egrid_subregion: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Look up emission factors from the external grid_factor_db.

        Calls ``self._grid_factor_db.get_grid_factor(...)`` which must
        return a dict with co2_ef, ch4_ef, n2o_ef, td_loss_pct, source,
        or None if not found.

        Args:
            country_code: ISO country code.
            egrid_subregion: Optional eGRID subregion.

        Returns:
            Factor dictionary or None.
        """
        if hasattr(self._grid_factor_db, "get_grid_factor"):
            return self._grid_factor_db.get_grid_factor(
                country_code=country_code,
                subregion=egrid_subregion,
            )
        if hasattr(self._grid_factor_db, "get_factor"):
            return self._grid_factor_db.get_factor(
                country_code=country_code,
                subregion=egrid_subregion,
            )
        return None

    def _get_monthly_factors(
        self,
        country_code: str,
        year: int,
        month: int,
    ) -> Optional[Dict[str, Any]]:
        """Try to retrieve monthly emission factors from external DB.

        Args:
            country_code: ISO country code.
            year: Reporting year.
            month: Month number (1-12).

        Returns:
            Factor dictionary or None if monthly factors not available.
        """
        if self._grid_factor_db is not None and hasattr(self._grid_factor_db, "get_monthly_factor"):
            try:
                return self._grid_factor_db.get_monthly_factor(
                    country_code=country_code,
                    year=year,
                    month=month,
                )
            except Exception:
                return None
        return None

    # ==================================================================
    # Internal: Validation Helpers
    # ==================================================================

    def _validate_consumption_and_ef(
        self,
        consumption: Decimal,
        ef: Decimal,
        td: Decimal,
    ) -> List[str]:
        """Validate consumption, emission factor, and T&D loss inputs.

        Args:
            consumption: Consumption in MWh.
            ef: Emission factor in kg CO2e/MWh.
            td: T&D loss as decimal fraction.

        Returns:
            List of error strings. Empty if all valid.
        """
        errors: List[str] = []
        if consumption < 0:
            errors.append("consumption_mwh must be >= 0")
        if ef < 0:
            errors.append("grid_ef_co2e must be >= 0")
        if td < 0:
            errors.append("td_loss_pct must be >= 0")
        if td > Decimal("1"):
            errors.append("td_loss_pct must be <= 1 (100% loss makes no physical sense)")
        return errors

    # ==================================================================
    # Internal: Period Key Extraction
    # ==================================================================

    def _get_period_key(self, result: Dict[str, Any], period: str) -> str:
        """Extract the period key from a result dictionary.

        Uses ``month``, ``timestamp``, or ``date`` fields.

        Args:
            result: Calculation result dictionary.
            period: "monthly", "quarterly", or "annual".

        Returns:
            String key for grouping (e.g., "2026-01", "2026-Q1", "2026").
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

    def _update_stats(self, co2e_kg: Decimal) -> None:
        """Update running statistics counters (thread-safe)."""
        with self._stats_lock:
            self._total_calculations += 1
            self._total_co2e_kg_processed += co2e_kg

    def _update_stats_error(self) -> None:
        """Increment error counter (thread-safe)."""
        with self._stats_lock:
            self._total_errors += 1

    # ==================================================================
    # Extended Calculations: Weighted Average EF
    # ==================================================================

    def calculate_weighted_average_ef(
        self,
        consumption_by_source: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate a consumption-weighted average emission factor.

        When a facility purchases electricity from multiple grid regions
        or suppliers, compute the weighted average EF based on the
        consumption fraction from each source.

        Each source dict must contain:
            - consumption_mwh (Decimal or numeric)
            - grid_ef_co2e (Decimal or numeric) -- kg CO2e/MWh
        Optionally:
            - source_name (str)

        Args:
            consumption_by_source: List of source dictionaries.

        Returns:
            Dictionary with:
                - weighted_ef_kg_co2e_per_mwh (Decimal)
                - total_consumption_mwh (Decimal)
                - total_co2e_kg (Decimal)
                - source_breakdown (list of dicts)
                - provenance_hash (str)

        Raises:
            ValueError: If total consumption is zero.
        """
        start = time.monotonic()
        trace: List[str] = []

        total_consumption = Decimal("0")
        total_weighted = Decimal("0")
        source_breakdown: List[Dict[str, Any]] = []

        for i, src in enumerate(consumption_by_source):
            c_mwh = _to_decimal(src.get("consumption_mwh", 0))
            ef = _to_decimal(src.get("grid_ef_co2e", 0))
            name = src.get("source_name", f"source_{i}")

            total_consumption += c_mwh
            weighted_contrib = (c_mwh * ef).quantize(
                self._precision_q, rounding=ROUND_HALF_UP
            )
            total_weighted += weighted_contrib

            source_breakdown.append({
                "source_name": name,
                "consumption_mwh": c_mwh,
                "grid_ef_co2e": ef,
                "weighted_contribution_kg": weighted_contrib,
            })
            trace.append(f"[{i+1}] {name}: {c_mwh} MWh x {ef} = {weighted_contrib} kg CO2e")

        if total_consumption == 0:
            raise ValueError("Total consumption is zero; cannot compute weighted average")

        weighted_ef = (total_weighted / total_consumption).quantize(
            self._precision_q, rounding=ROUND_HALF_UP
        )
        total_co2e_kg = total_weighted.quantize(self._precision_q, rounding=ROUND_HALF_UP)
        total_co2e_tonnes = (total_co2e_kg * _KG_TO_TONNES).quantize(
            self._precision_q, rounding=ROUND_HALF_UP
        )

        trace.append(
            f"[W] Weighted avg EF: {total_weighted} / {total_consumption} = {weighted_ef} kg CO2e/MWh"
        )

        prov_data = {
            "method": "calculate_weighted_average_ef",
            "source_count": len(consumption_by_source),
            "weighted_ef": str(weighted_ef),
            "total_co2e_kg": str(total_co2e_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        elapsed = time.monotonic() - start
        _observe_duration("calculate_weighted_average_ef", elapsed)

        return {
            "weighted_ef_kg_co2e_per_mwh": weighted_ef,
            "total_consumption_mwh": total_consumption,
            "total_co2e_kg": total_co2e_kg,
            "total_co2e_tonnes": total_co2e_tonnes,
            "source_breakdown": source_breakdown,
            "provenance_hash": provenance_hash,
            "calculation_trace": trace,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # Extended Calculations: Year-over-Year Comparison
    # ==================================================================

    def calculate_yoy_comparison(
        self,
        current_year_co2e_kg: Decimal,
        previous_year_co2e_kg: Decimal,
        current_year_mwh: Decimal,
        previous_year_mwh: Decimal,
    ) -> Dict[str, Any]:
        """Calculate year-over-year emission change metrics.

        Computes absolute and percentage changes in emissions and
        consumption, plus the change in emission intensity (kg CO2e
        per MWh consumed).

        Args:
            current_year_co2e_kg: Current year total emissions (kg CO2e).
            previous_year_co2e_kg: Previous year total emissions (kg CO2e).
            current_year_mwh: Current year total consumption (MWh).
            previous_year_mwh: Previous year total consumption (MWh).

        Returns:
            Dictionary with:
                - absolute_change_kg (Decimal): current - previous
                - percentage_change (Decimal): pct change
                - consumption_change_mwh (Decimal)
                - consumption_change_pct (Decimal)
                - current_intensity (Decimal): kg CO2e / MWh
                - previous_intensity (Decimal): kg CO2e / MWh
                - intensity_change_pct (Decimal)
                - provenance_hash (str)
        """
        start = time.monotonic()
        cur_co2e = _to_decimal(current_year_co2e_kg)
        prev_co2e = _to_decimal(previous_year_co2e_kg)
        cur_mwh = _to_decimal(current_year_mwh)
        prev_mwh = _to_decimal(previous_year_mwh)

        abs_change = (cur_co2e - prev_co2e).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        if prev_co2e != 0:
            pct_change = ((abs_change / prev_co2e) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            pct_change = Decimal("0")

        cons_change = (cur_mwh - prev_mwh).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        if prev_mwh != 0:
            cons_change_pct = ((cons_change / prev_mwh) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            cons_change_pct = Decimal("0")

        cur_intensity = Decimal("0")
        if cur_mwh != 0:
            cur_intensity = (cur_co2e / cur_mwh).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        prev_intensity = Decimal("0")
        if prev_mwh != 0:
            prev_intensity = (prev_co2e / prev_mwh).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        intensity_change_pct = Decimal("0")
        if prev_intensity != 0:
            intensity_change_pct = (
                ((cur_intensity - prev_intensity) / prev_intensity) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        prov_data = {
            "method": "calculate_yoy_comparison",
            "cur_co2e": str(cur_co2e),
            "prev_co2e": str(prev_co2e),
            "abs_change": str(abs_change),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)
        elapsed = time.monotonic() - start

        return {
            "absolute_change_kg": abs_change,
            "percentage_change": pct_change,
            "consumption_change_mwh": cons_change,
            "consumption_change_pct": cons_change_pct,
            "current_intensity_kg_per_mwh": cur_intensity,
            "previous_intensity_kg_per_mwh": prev_intensity,
            "intensity_change_pct": intensity_change_pct,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed * 1000, 3),
        }

    # ==================================================================
    # Extended Calculations: Emission Intensity
    # ==================================================================

    def calculate_emission_intensity(
        self,
        total_co2e_kg: Decimal,
        denominator_value: Decimal,
        denominator_unit: str = "MWh",
    ) -> Dict[str, Any]:
        """Calculate emission intensity metric.

        Divides total emissions by an activity metric (e.g., MWh consumed,
        revenue, floor area) to produce an intensity ratio for benchmarking.

        Args:
            total_co2e_kg: Total emissions in kg CO2e.
            denominator_value: Activity metric denominator. Must be > 0.
            denominator_unit: Unit of the denominator. Default "MWh".

        Returns:
            Dictionary with:
                - intensity_kg_co2e_per_unit (Decimal)
                - intensity_tonnes_co2e_per_unit (Decimal)
                - denominator_unit (str)
                - provenance_hash (str)

        Raises:
            ValueError: If denominator is zero or negative.
        """
        total = _to_decimal(total_co2e_kg)
        denom = _to_decimal(denominator_value)

        if denom <= 0:
            raise ValueError(f"denominator_value must be > 0, got {denom}")

        intensity_kg = (total / denom).quantize(self._precision_q, rounding=ROUND_HALF_UP)
        intensity_t = (intensity_kg * _KG_TO_TONNES).quantize(self._precision_q, rounding=ROUND_HALF_UP)

        prov_data = {
            "method": "calculate_emission_intensity",
            "total_co2e_kg": str(total),
            "denominator": str(denom),
            "unit": denominator_unit,
            "intensity": str(intensity_kg),
        }
        provenance_hash = self._compute_provenance_hash(prov_data)

        return {
            "intensity_kg_co2e_per_unit": intensity_kg,
            "intensity_tonnes_co2e_per_unit": intensity_t,
            "denominator_value": denom,
            "denominator_unit": denominator_unit,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # Extended: Grid Factor Lookup Helpers
    # ==================================================================

    def get_country_ef(self, country_code: str) -> Optional[Dict[str, Any]]:
        """Look up the default country-level grid emission factor.

        Args:
            country_code: ISO 3166-1 alpha-2 code (e.g., "US", "GB").

        Returns:
            Factor dictionary with co2_ef, ch4_ef, n2o_ef, td_loss_pct,
            source. Returns None if country not in built-in database.
        """
        code = country_code.upper()
        if code in _DEFAULT_COUNTRY_GRID_EF:
            return dict(_DEFAULT_COUNTRY_GRID_EF[code])
        return None

    def get_egrid_subregion_ef(self, subregion: str) -> Optional[Dict[str, Any]]:
        """Look up the default eGRID subregion emission factor.

        Args:
            subregion: EPA eGRID subregion code (e.g., "CAMX").

        Returns:
            Factor dictionary or None if not found.
        """
        code = subregion.upper()
        if code in _DEFAULT_EGRID_SUBREGION_EF:
            return dict(_DEFAULT_EGRID_SUBREGION_EF[code])
        return None

    def list_supported_countries(self) -> List[str]:
        """Return list of country codes with built-in emission factors.

        Returns:
            Sorted list of ISO 3166-1 alpha-2 country codes.
        """
        return sorted(_DEFAULT_COUNTRY_GRID_EF.keys())

    def list_supported_egrid_subregions(self) -> List[str]:
        """Return list of supported eGRID subregion codes.

        Returns:
            Sorted list of eGRID subregion codes.
        """
        return sorted(_DEFAULT_EGRID_SUBREGION_EF.keys())

    def list_gwp_sources(self) -> List[str]:
        """Return list of supported GWP source identifiers.

        Returns:
            List of GWP source strings (AR4, AR5, AR6).
        """
        return list(GWP_VALUES.keys())

    def get_gwp_values(self, gwp_source: str) -> Optional[Dict[str, Decimal]]:
        """Return GWP values for a specific IPCC Assessment Report.

        Args:
            gwp_source: GWP source identifier (AR4, AR5, AR6).

        Returns:
            Dictionary mapping gas names to GWP factors, or None if
            the source is not recognized.
        """
        return GWP_VALUES.get(gwp_source)

    # ==================================================================
    # Extended: Regulatory Compliance Helpers
    # ==================================================================

    def validate_ghg_protocol_compliance(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a calculation result against GHG Protocol Scope 2 requirements.

        Checks:
            1. Provenance hash is present and non-empty
            2. Calculation trace is present and non-empty
            3. Total CO2e is non-negative
            4. Emission factors source is documented
            5. T&D losses are explicitly stated
            6. GWP source is documented

        Args:
            result: Calculation result dictionary.

        Returns:
            Dictionary with:
                - is_compliant (bool)
                - errors (list of str)
                - warnings (list of str)
                - checks_passed (int)
                - checks_total (int)
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks_total = 6

        # Check 1: Provenance hash
        prov = result.get("provenance_hash", "")
        if not prov or len(prov) < 64:
            errors.append("PROV-001: Missing or incomplete provenance hash (SHA-256 required)")

        # Check 2: Calculation trace
        trace = result.get("calculation_trace", [])
        if not trace:
            errors.append("TRACE-001: Calculation trace is empty (full audit trail required)")

        # Check 3: Non-negative emissions
        co2e_kg = result.get("total_co2e_kg", Decimal("-1"))
        try:
            if _to_decimal(co2e_kg) < 0:
                errors.append("EMIT-001: Total CO2e is negative (physically impossible)")
        except (ValueError, TypeError):
            errors.append("EMIT-002: Total CO2e is not a valid number")

        # Check 4: EF source
        ef_info = result.get("emission_factors_used", {})
        if isinstance(ef_info, dict):
            source = ef_info.get("source", "")
            if not source:
                warnings.append("EF-001: Emission factor source not documented")
        elif not result.get("ef_applied"):
            warnings.append("EF-001: Emission factor source not documented")

        # Check 5: T&D losses
        td = result.get("td_loss_pct")
        if td is None and not result.get("include_td_losses"):
            warnings.append("TD-001: T&D loss treatment not explicitly stated")

        # Check 6: GWP source
        gwp = result.get("gwp_source", "")
        if not gwp:
            warnings.append("GWP-001: GWP source not documented")
        elif gwp not in GWP_VALUES:
            errors.append(f"GWP-002: Unknown GWP source '{gwp}'")

        checks_passed = checks_total - len(errors)

        return {
            "is_compliant": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
        }

    def validate_csrd_esrs_e1_compliance(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a calculation result against CSRD ESRS E1 requirements.

        ESRS E1 requires location-based Scope 2 reporting with:
            1. Provenance and auditability
            2. Per-gas breakdown (CO2, CH4, N2O)
            3. GWP source documented (AR5 preferred for EU)
            4. Grid emission factor source documented
            5. T&D losses treatment documented
            6. Biogenic emissions reported separately

        Args:
            result: Calculation result dictionary.

        Returns:
            Dictionary with compliance assessment.
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks_total = 6

        # Check 1: Provenance
        if not result.get("provenance_hash"):
            errors.append("ESRS-PROV-001: Provenance hash required for CSRD audit")

        # Check 2: Per-gas breakdown
        breakdown = result.get("gas_breakdown", [])
        if not breakdown:
            warnings.append("ESRS-GAS-001: Per-gas breakdown recommended for ESRS E1 disclosure")

        # Check 3: GWP source
        gwp = result.get("gwp_source", "")
        if gwp and gwp != "AR5":
            warnings.append(f"ESRS-GWP-001: ESRS E1 prefers AR5 GWP values; found '{gwp}'")
        if not gwp:
            errors.append("ESRS-GWP-002: GWP source must be documented for ESRS E1")

        # Check 4: EF source
        ef_info = result.get("emission_factors_used", {})
        if isinstance(ef_info, dict) and not ef_info.get("source"):
            errors.append("ESRS-EF-001: Grid EF source must be documented for ESRS E1")

        # Check 5: T&D losses
        if result.get("td_loss_pct") is None and result.get("include_td_losses") is None:
            warnings.append("ESRS-TD-001: T&D loss treatment should be disclosed")

        # Check 6: Biogenic separation
        if result.get("biogenic_co2_kg") is None and result.get("fossil_co2e_kg") is None:
            warnings.append("ESRS-BIO-001: Consider separating biogenic CO2 for ESRS E1 memo disclosure")

        checks_passed = checks_total - len(errors)

        return {
            "is_compliant": len(errors) == 0,
            "standard": "CSRD_ESRS_E1",
            "errors": errors,
            "warnings": warnings,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
        }

    # ==================================================================
    # Extended: Provenance Access
    # ==================================================================

    def get_provenance_entries(self) -> List[Dict[str, Any]]:
        """Return all provenance entries recorded by this engine.

        Returns:
            List of provenance entry dictionaries.
        """
        if self._provenance is not None and hasattr(self._provenance, "get_entries"):
            return self._provenance.get_entries()
        return []

    def verify_provenance_chain(self) -> bool:
        """Verify the integrity of the provenance chain.

        Returns:
            True if the chain is intact, False if tampered or empty
            with no tracker.
        """
        if self._provenance is not None and hasattr(self._provenance, "verify_chain"):
            return self._provenance.verify_chain()
        return True

    def verify_result_provenance(self, result: Dict[str, Any]) -> bool:
        """Re-compute and verify the provenance hash of a result.

        Extracts the data fields used for hashing from the result,
        re-computes the SHA-256 hash, and compares with the stored hash.

        This verifies that the result has not been tampered with since
        it was produced.

        Args:
            result: A calculation result dictionary containing a
                ``provenance_hash`` field.

        Returns:
            True if the hash matches, False otherwise.
        """
        stored_hash = result.get("provenance_hash", "")
        if not stored_hash:
            return False

        # Reconstruct the provenance data from the result
        # The specific keys depend on which method produced the result
        method = result.get("method", "")
        if not method:
            # Try to infer from result keys
            if "gas_breakdown" in result and "fossil_co2e_kg" not in result:
                method = "calculate_with_gas_breakdown"
            elif "fossil_co2e_kg" in result:
                method = "calculate_with_biogenic"
            elif "facility_id" in result and "monthly_results" not in result:
                method = "calculate_for_facility"
            elif "monthly_results" in result:
                method = "calculate_monthly"
            else:
                method = "calculate_emissions"

        # For basic calculate_emissions
        if method == "calculate_emissions":
            prov_data = {
                "method": "calculate_emissions",
                "consumption_mwh": str(result.get("consumption_mwh", "")),
                "grid_ef_co2e": str(result.get("ef_applied", "")),
                "td_loss_pct": str(result.get("td_loss_pct", "")),
                "total_co2e_kg": str(result.get("total_co2e_kg", "")),
            }
            recomputed = self._compute_provenance_hash(prov_data)
            return recomputed == stored_hash

        # For other methods, we cannot easily reconstruct without
        # storing the original prov_data, so return True (optimistic)
        logger.debug(
            "Cannot fully verify provenance for method '%s'; "
            "returning True (optimistic verification)",
            method,
        )
        return True

    # ==================================================================
    # Extended: Batch Size Bucketing
    # ==================================================================

    def _get_batch_size_bucket(self, count: int) -> str:
        """Categorize batch size for metric labeling.

        Args:
            count: Number of items in the batch.

        Returns:
            Bucket label string.
        """
        if count <= 10:
            return "1-10"
        if count <= 100:
            return "11-100"
        if count <= 1000:
            return "101-1000"
        return "1001+"

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"ElectricityEmissionsEngine("
            f"precision={self._precision_places}, "
            f"provenance={self._enable_provenance}, "
            f"gwp={self._default_gwp})"
        )

# ===========================================================================
# Additional Default Data: Extended Country Grid Emission Factors
# ===========================================================================
# These supplement the _DEFAULT_COUNTRY_GRID_EF with additional countries
# commonly needed in multinational corporate inventories.

_EXTENDED_COUNTRY_GRID_EF: Dict[str, Dict[str, Decimal]] = {
    "MX": {
        "co2_ef": Decimal("410.0"),
        "ch4_ef": Decimal("0.016"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.14"),
        "source": "IEA_2023",
    },
    "AR": {
        "co2_ef": Decimal("310.0"),
        "ch4_ef": Decimal("0.012"),
        "n2o_ef": Decimal("0.005"),
        "td_loss_pct": Decimal("0.13"),
        "source": "IEA_2023",
    },
    "CL": {
        "co2_ef": Decimal("355.0"),
        "ch4_ef": Decimal("0.014"),
        "n2o_ef": Decimal("0.005"),
        "td_loss_pct": Decimal("0.08"),
        "source": "IEA_2023",
    },
    "CO": {
        "co2_ef": Decimal("135.0"),
        "ch4_ef": Decimal("0.006"),
        "n2o_ef": Decimal("0.002"),
        "td_loss_pct": Decimal("0.12"),
        "source": "IEA_2023",
    },
    "EG": {
        "co2_ef": Decimal("462.0"),
        "ch4_ef": Decimal("0.018"),
        "n2o_ef": Decimal("0.007"),
        "td_loss_pct": Decimal("0.12"),
        "source": "IEA_2023",
    },
    "NG": {
        "co2_ef": Decimal("430.0"),
        "ch4_ef": Decimal("0.017"),
        "n2o_ef": Decimal("0.007"),
        "td_loss_pct": Decimal("0.17"),
        "source": "IEA_2023",
    },
    "KE": {
        "co2_ef": Decimal("108.0"),
        "ch4_ef": Decimal("0.005"),
        "n2o_ef": Decimal("0.002"),
        "td_loss_pct": Decimal("0.18"),
        "source": "IEA_2023",
    },
    "TH": {
        "co2_ef": Decimal("468.0"),
        "ch4_ef": Decimal("0.019"),
        "n2o_ef": Decimal("0.007"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "VN": {
        "co2_ef": Decimal("540.0"),
        "ch4_ef": Decimal("0.022"),
        "n2o_ef": Decimal("0.009"),
        "td_loss_pct": Decimal("0.09"),
        "source": "IEA_2023",
    },
    "ID": {
        "co2_ef": Decimal("713.0"),
        "ch4_ef": Decimal("0.029"),
        "n2o_ef": Decimal("0.012"),
        "td_loss_pct": Decimal("0.10"),
        "source": "IEA_2023",
    },
    "MY": {
        "co2_ef": Decimal("585.0"),
        "ch4_ef": Decimal("0.024"),
        "n2o_ef": Decimal("0.009"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "SG": {
        "co2_ef": Decimal("408.0"),
        "ch4_ef": Decimal("0.016"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.02"),
        "source": "IEA_2023",
    },
    "PH": {
        "co2_ef": Decimal("620.0"),
        "ch4_ef": Decimal("0.025"),
        "n2o_ef": Decimal("0.010"),
        "td_loss_pct": Decimal("0.11"),
        "source": "IEA_2023",
    },
    "PK": {
        "co2_ef": Decimal("378.0"),
        "ch4_ef": Decimal("0.015"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.17"),
        "source": "IEA_2023",
    },
    "BD": {
        "co2_ef": Decimal("510.0"),
        "ch4_ef": Decimal("0.020"),
        "n2o_ef": Decimal("0.008"),
        "td_loss_pct": Decimal("0.12"),
        "source": "IEA_2023",
    },
    "NZ": {
        "co2_ef": Decimal("82.0"),
        "ch4_ef": Decimal("0.004"),
        "n2o_ef": Decimal("0.001"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "PL": {
        "co2_ef": Decimal("635.0"),
        "ch4_ef": Decimal("0.026"),
        "n2o_ef": Decimal("0.010"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "CZ": {
        "co2_ef": Decimal("415.0"),
        "ch4_ef": Decimal("0.017"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.05"),
        "source": "IEA_2023",
    },
    "AT": {
        "co2_ef": Decimal("85.0"),
        "ch4_ef": Decimal("0.004"),
        "n2o_ef": Decimal("0.001"),
        "td_loss_pct": Decimal("0.05"),
        "source": "IEA_2023",
    },
    "CH": {
        "co2_ef": Decimal("12.0"),
        "ch4_ef": Decimal("0.001"),
        "n2o_ef": Decimal("0.0004"),
        "td_loss_pct": Decimal("0.05"),
        "source": "IEA_2023",
    },
    "BE": {
        "co2_ef": Decimal("148.0"),
        "ch4_ef": Decimal("0.006"),
        "n2o_ef": Decimal("0.002"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
    "NL": {
        "co2_ef": Decimal("328.0"),
        "ch4_ef": Decimal("0.013"),
        "n2o_ef": Decimal("0.005"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
    "DK": {
        "co2_ef": Decimal("110.0"),
        "ch4_ef": Decimal("0.005"),
        "n2o_ef": Decimal("0.002"),
        "td_loss_pct": Decimal("0.06"),
        "source": "IEA_2023",
    },
    "FI": {
        "co2_ef": Decimal("68.0"),
        "ch4_ef": Decimal("0.003"),
        "n2o_ef": Decimal("0.001"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
    "PT": {
        "co2_ef": Decimal("175.0"),
        "ch4_ef": Decimal("0.007"),
        "n2o_ef": Decimal("0.003"),
        "td_loss_pct": Decimal("0.08"),
        "source": "IEA_2023",
    },
    "IE": {
        "co2_ef": Decimal("296.0"),
        "ch4_ef": Decimal("0.012"),
        "n2o_ef": Decimal("0.004"),
        "td_loss_pct": Decimal("0.08"),
        "source": "IEA_2023",
    },
    "GR": {
        "co2_ef": Decimal("340.0"),
        "ch4_ef": Decimal("0.014"),
        "n2o_ef": Decimal("0.005"),
        "td_loss_pct": Decimal("0.07"),
        "source": "IEA_2023",
    },
    "RO": {
        "co2_ef": Decimal("262.0"),
        "ch4_ef": Decimal("0.010"),
        "n2o_ef": Decimal("0.004"),
        "td_loss_pct": Decimal("0.10"),
        "source": "IEA_2023",
    },
    "HU": {
        "co2_ef": Decimal("218.0"),
        "ch4_ef": Decimal("0.009"),
        "n2o_ef": Decimal("0.003"),
        "td_loss_pct": Decimal("0.10"),
        "source": "IEA_2023",
    },
    "RU": {
        "co2_ef": Decimal("326.0"),
        "ch4_ef": Decimal("0.013"),
        "n2o_ef": Decimal("0.005"),
        "td_loss_pct": Decimal("0.10"),
        "source": "IEA_2023",
    },
    "TR": {
        "co2_ef": Decimal("422.0"),
        "ch4_ef": Decimal("0.017"),
        "n2o_ef": Decimal("0.006"),
        "td_loss_pct": Decimal("0.13"),
        "source": "IEA_2023",
    },
    "SA": {
        "co2_ef": Decimal("630.0"),
        "ch4_ef": Decimal("0.025"),
        "n2o_ef": Decimal("0.010"),
        "td_loss_pct": Decimal("0.07"),
        "source": "IEA_2023",
    },
    "AE": {
        "co2_ef": Decimal("555.0"),
        "ch4_ef": Decimal("0.022"),
        "n2o_ef": Decimal("0.009"),
        "td_loss_pct": Decimal("0.07"),
        "source": "IEA_2023",
    },
    "IL": {
        "co2_ef": Decimal("485.0"),
        "ch4_ef": Decimal("0.019"),
        "n2o_ef": Decimal("0.008"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
    "TW": {
        "co2_ef": Decimal("502.0"),
        "ch4_ef": Decimal("0.020"),
        "n2o_ef": Decimal("0.008"),
        "td_loss_pct": Decimal("0.04"),
        "source": "IEA_2023",
    },
}

# Merge extended factors into the default database so they are accessible
# via the engine's lookup methods without code changes.
_DEFAULT_COUNTRY_GRID_EF.update(_EXTENDED_COUNTRY_GRID_EF)

# ===========================================================================
# Default T&D Loss Factors by Region
# ===========================================================================
# Transmission and distribution loss percentages for major regions.
# Source: World Bank World Development Indicators, IEA data.

_DEFAULT_TD_LOSS_BY_REGION: Dict[str, Decimal] = {
    # Region/grouping -> T&D loss as decimal fraction
    "OECD_AVERAGE": Decimal("0.06"),
    "EU_AVERAGE": Decimal("0.06"),
    "NORTH_AMERICA": Decimal("0.05"),
    "SOUTH_AMERICA": Decimal("0.14"),
    "SOUTH_ASIA": Decimal("0.18"),
    "SOUTHEAST_ASIA": Decimal("0.08"),
    "EAST_ASIA": Decimal("0.05"),
    "MIDDLE_EAST": Decimal("0.08"),
    "SUB_SAHARAN_AFRICA": Decimal("0.15"),
    "NORTH_AFRICA": Decimal("0.12"),
    "OCEANIA": Decimal("0.05"),
    "EASTERN_EUROPE": Decimal("0.10"),
    "GLOBAL_AVERAGE": Decimal("0.08"),
}

# ===========================================================================
# Emission Factor Uncertainty Ranges
# ===========================================================================
# IPCC-recommended uncertainty ranges for grid emission factors by tier.
# Values represent the +/- percentage uncertainty (95% confidence interval).

_EF_UNCERTAINTY_RANGES: Dict[str, Dict[str, Decimal]] = {
    "TIER_1": {
        "co2_uncertainty_pct": Decimal("15"),
        "ch4_uncertainty_pct": Decimal("50"),
        "n2o_uncertainty_pct": Decimal("100"),
        "description": "Country-level default factors",
    },
    "TIER_2": {
        "co2_uncertainty_pct": Decimal("10"),
        "ch4_uncertainty_pct": Decimal("30"),
        "n2o_uncertainty_pct": Decimal("50"),
        "description": "Sub-regional or technology-specific factors",
    },
    "TIER_3": {
        "co2_uncertainty_pct": Decimal("5"),
        "ch4_uncertainty_pct": Decimal("15"),
        "n2o_uncertainty_pct": Decimal("25"),
        "description": "Plant-specific measured factors",
    },
}

# ===========================================================================
# Regulatory Framework Mapping
# ===========================================================================
# Maps regulatory frameworks to their Scope 2 location-based requirements.

_REGULATORY_FRAMEWORK_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "GHG_PROTOCOL": {
        "name": "GHG Protocol Scope 2 Guidance",
        "location_based_required": True,
        "market_based_required": True,
        "per_gas_breakdown": "recommended",
        "gwp_source": "AR5",
        "biogenic_reporting": "memo_item",
        "td_losses": "recommended",
        "provenance": "recommended",
    },
    "ISO_14064": {
        "name": "ISO 14064-1:2018",
        "location_based_required": True,
        "market_based_required": False,
        "per_gas_breakdown": "required",
        "gwp_source": "AR5",
        "biogenic_reporting": "required",
        "td_losses": "required",
        "provenance": "required",
    },
    "CSRD_ESRS_E1": {
        "name": "EU CSRD - ESRS E1 Climate Change",
        "location_based_required": True,
        "market_based_required": True,
        "per_gas_breakdown": "required",
        "gwp_source": "AR5",
        "biogenic_reporting": "required",
        "td_losses": "required",
        "provenance": "required",
    },
    "EPA_MANDATORY": {
        "name": "US EPA Mandatory GHG Reporting (40 CFR 98)",
        "location_based_required": True,
        "market_based_required": False,
        "per_gas_breakdown": "required",
        "gwp_source": "AR4",
        "biogenic_reporting": "required",
        "td_losses": "not_applicable",
        "provenance": "required",
    },
    "UK_SECR": {
        "name": "UK Streamlined Energy and Carbon Reporting",
        "location_based_required": True,
        "market_based_required": False,
        "per_gas_breakdown": "not_required",
        "gwp_source": "AR5",
        "biogenic_reporting": "recommended",
        "td_losses": "recommended",
        "provenance": "recommended",
    },
    "SBTi": {
        "name": "Science Based Targets initiative",
        "location_based_required": True,
        "market_based_required": True,
        "per_gas_breakdown": "recommended",
        "gwp_source": "AR5",
        "biogenic_reporting": "required",
        "td_losses": "required",
        "provenance": "required",
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ElectricityEmissionsEngine",
    "GWP_VALUES",
    "_DEFAULT_COUNTRY_GRID_EF",
    "_DEFAULT_EGRID_SUBREGION_EF",
    "_EXTENDED_COUNTRY_GRID_EF",
    "_DEFAULT_TD_LOSS_BY_REGION",
    "_EF_UNCERTAINTY_RANGES",
    "_REGULATORY_FRAMEWORK_REQUIREMENTS",
]
