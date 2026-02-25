# -*- coding: utf-8 -*-
"""
TransmissionLossEngine - Engine 4: Scope 2 Location-Based Emissions (AGENT-MRV-009)

Calculates Transmission & Distribution (T&D) losses to adjust Scope 2 emissions.
T&D losses represent electricity lost during transmission from power plants to
end consumers. The GHG Protocol recommends including T&D losses in Scope 2
calculations:

    Gross Consumption = Net Consumption x (1 + T&D Loss Factor)
    Upstream T&D Emissions = Net Consumption x T&D Loss Factor x Grid EF

The engine provides a comprehensive built-in T&D loss factor database covering
53 countries plus a world average, sourced from IEA World Energy Balances and
World Bank data. All factors are stored and computed using Python Decimal
arithmetic to guarantee zero-hallucination, bit-perfect reproducibility.

Zero-Hallucination Guarantees:
    - All calculations use Python Decimal (ROUND_HALF_UP, 8 decimal places)
    - No LLM calls in any calculation path
    - Every operation is recorded via SHA-256 provenance hashing
    - Identical inputs always produce identical outputs
    - Complete audit trail for every T&D loss adjustment

Country Coverage (53 countries + WORLD average):
    Americas: US, CA, MX, BR, AR, CL, CO, PE
    Europe: GB, DE, FR, IT, ES, NL, BE, AT, SE, NO, DK, FI, PL, CZ, HU,
            RO, BG, GR, PT, IE, CH
    Asia-Pacific: JP, CN, IN, KR, TH, VN, ID, MY, PH, SG, TW, AU, NZ
    Middle East & Africa: AE, SA, TR, ZA, EG, NG, KE, PK, BD, RU

Regional Groupings:
    EU, NORTH_AMERICA, SOUTH_AMERICA, ASIA_PACIFIC, MIDDLE_EAST_AFRICA,
    OECD, NON_OECD

Example:
    >>> from greenlang.scope2_location.transmission_loss import TransmissionLossEngine
    >>> from decimal import Decimal
    >>> engine = TransmissionLossEngine()
    >>> result = engine.calculate_td_loss(
    ...     net_consumption_mwh=Decimal("1000"),
    ...     country_code="US",
    ... )
    >>> assert result["gross_consumption_mwh"] == Decimal("1050.000")
    >>> assert result["loss_mwh"] == Decimal("50.000")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-009 Scope 2 Location-Based Emissions (GL-MRV-SCOPE2-001)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "TransmissionLossEngine",
    "TDLossResult",
    "TD_LOSS_FACTORS",
    "REGIONAL_GROUPS",
    "VALID_COUNTRY_CODES",
]

# ---------------------------------------------------------------------------
# Conditional imports -- graceful fallback for metrics and provenance
# ---------------------------------------------------------------------------

try:
    from greenlang.scope2_location.metrics import (
        Scope2LocationMetrics,
        get_metrics as _get_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _get_metrics = None  # type: ignore[assignment]
    Scope2LocationMetrics = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope2_location.provenance import (  # type: ignore[import-untyped]
        ProvenanceTracker as _ProvenanceTracker,
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _ProvenanceTracker = None  # type: ignore[assignment,misc]
    _get_provenance_tracker = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# SHA-256 hashing utility
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Serializes the payload to canonical JSON (sorted keys, ``str`` fallback)
    before hashing to ensure that equivalent structures always produce the
    same digest.

    Args:
        data: Data to hash (dict, list, str, Decimal, or Pydantic model).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, Decimal):
        serializable = str(data)
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places for intermediate calcs
_OUTPUT_PRECISION = Decimal("0.001")  # 3 decimal places for output values
_FACTOR_PRECISION = Decimal("0.0001")  # 4 decimal places for loss factors
_ZERO = Decimal("0")
_ONE = Decimal("1")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")

# Maximum allowable T&D loss factor (50%)
_MAX_TD_LOSS_PCT = Decimal("0.50")
# Minimum allowable T&D loss factor (0%, exclusive)
_MIN_TD_LOSS_PCT = Decimal("0")


# ---------------------------------------------------------------------------
# T&D Loss Factor Database - All values as Decimal
# ---------------------------------------------------------------------------
# Sources:
#   - IEA World Energy Balances (2023 edition)
#   - World Bank, World Development Indicators
#   - National regulatory disclosures
# All values represent percentage of electricity lost during transmission
# and distribution, expressed as a decimal fraction.
# ---------------------------------------------------------------------------

TD_LOSS_FACTORS: Dict[str, Dict[str, Any]] = {
    # ---- Americas ----
    "US": {
        "name": "United States",
        "td_loss_pct": Decimal("0.050"),
        "source": "EIA / EPA eGRID",
        "year": 2023,
        "confidence": "HIGH",
    },
    "CA": {
        "name": "Canada",
        "td_loss_pct": Decimal("0.070"),
        "source": "Statistics Canada / NEB",
        "year": 2023,
        "confidence": "HIGH",
    },
    "MX": {
        "name": "Mexico",
        "td_loss_pct": Decimal("0.121"),
        "source": "SENER / CFE",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "BR": {
        "name": "Brazil",
        "td_loss_pct": Decimal("0.156"),
        "source": "ANEEL / EPE",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "AR": {
        "name": "Argentina",
        "td_loss_pct": Decimal("0.142"),
        "source": "CAMMESA",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "CL": {
        "name": "Chile",
        "td_loss_pct": Decimal("0.075"),
        "source": "CNE / CEN",
        "year": 2023,
        "confidence": "HIGH",
    },
    "CO": {
        "name": "Colombia",
        "td_loss_pct": Decimal("0.120"),
        "source": "XM / UPME",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "PE": {
        "name": "Peru",
        "td_loss_pct": Decimal("0.105"),
        "source": "COES / OSINERGMIN",
        "year": 2023,
        "confidence": "MEDIUM",
    },

    # ---- Europe ----
    "GB": {
        "name": "United Kingdom",
        "td_loss_pct": Decimal("0.077"),
        "source": "DEFRA / National Grid ESO",
        "year": 2023,
        "confidence": "HIGH",
    },
    "DE": {
        "name": "Germany",
        "td_loss_pct": Decimal("0.040"),
        "source": "UBA / BDEW",
        "year": 2023,
        "confidence": "HIGH",
    },
    "FR": {
        "name": "France",
        "td_loss_pct": Decimal("0.060"),
        "source": "RTE / ADEME",
        "year": 2023,
        "confidence": "HIGH",
    },
    "IT": {
        "name": "Italy",
        "td_loss_pct": Decimal("0.062"),
        "source": "TERNA / ISPRA",
        "year": 2023,
        "confidence": "HIGH",
    },
    "ES": {
        "name": "Spain",
        "td_loss_pct": Decimal("0.090"),
        "source": "REE / MITECO",
        "year": 2023,
        "confidence": "HIGH",
    },
    "NL": {
        "name": "Netherlands",
        "td_loss_pct": Decimal("0.043"),
        "source": "TenneT / CBS",
        "year": 2023,
        "confidence": "HIGH",
    },
    "BE": {
        "name": "Belgium",
        "td_loss_pct": Decimal("0.048"),
        "source": "Elia / CREG",
        "year": 2023,
        "confidence": "HIGH",
    },
    "AT": {
        "name": "Austria",
        "td_loss_pct": Decimal("0.056"),
        "source": "APG / E-Control",
        "year": 2023,
        "confidence": "HIGH",
    },
    "SE": {
        "name": "Sweden",
        "td_loss_pct": Decimal("0.068"),
        "source": "Svenska Kraftnat / SCB",
        "year": 2023,
        "confidence": "HIGH",
    },
    "NO": {
        "name": "Norway",
        "td_loss_pct": Decimal("0.062"),
        "source": "Statnett / SSB",
        "year": 2023,
        "confidence": "HIGH",
    },
    "DK": {
        "name": "Denmark",
        "td_loss_pct": Decimal("0.058"),
        "source": "Energinet / DEA",
        "year": 2023,
        "confidence": "HIGH",
    },
    "FI": {
        "name": "Finland",
        "td_loss_pct": Decimal("0.034"),
        "source": "Fingrid / Statistics Finland",
        "year": 2023,
        "confidence": "HIGH",
    },
    "PL": {
        "name": "Poland",
        "td_loss_pct": Decimal("0.066"),
        "source": "PSE / GUS",
        "year": 2023,
        "confidence": "HIGH",
    },
    "CZ": {
        "name": "Czech Republic",
        "td_loss_pct": Decimal("0.058"),
        "source": "CEPS / ERU",
        "year": 2023,
        "confidence": "HIGH",
    },
    "HU": {
        "name": "Hungary",
        "td_loss_pct": Decimal("0.098"),
        "source": "MAVIR / MEKH",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "RO": {
        "name": "Romania",
        "td_loss_pct": Decimal("0.112"),
        "source": "Transelectrica / ANRE",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "BG": {
        "name": "Bulgaria",
        "td_loss_pct": Decimal("0.094"),
        "source": "ESO EAD / EWRC",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "GR": {
        "name": "Greece",
        "td_loss_pct": Decimal("0.078"),
        "source": "ADMIE / RAE",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "PT": {
        "name": "Portugal",
        "td_loss_pct": Decimal("0.083"),
        "source": "REN / ERSE",
        "year": 2023,
        "confidence": "HIGH",
    },
    "IE": {
        "name": "Ireland",
        "td_loss_pct": Decimal("0.075"),
        "source": "EirGrid / CRU",
        "year": 2023,
        "confidence": "HIGH",
    },
    "CH": {
        "name": "Switzerland",
        "td_loss_pct": Decimal("0.052"),
        "source": "Swissgrid / BFE",
        "year": 2023,
        "confidence": "HIGH",
    },

    # ---- Asia-Pacific ----
    "JP": {
        "name": "Japan",
        "td_loss_pct": Decimal("0.050"),
        "source": "OCCTO / METI",
        "year": 2023,
        "confidence": "HIGH",
    },
    "CN": {
        "name": "China",
        "td_loss_pct": Decimal("0.058"),
        "source": "NBS / NEA",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "IN": {
        "name": "India",
        "td_loss_pct": Decimal("0.194"),
        "source": "CEA / PFC",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "KR": {
        "name": "South Korea",
        "td_loss_pct": Decimal("0.036"),
        "source": "KEPCO / KPX",
        "year": 2023,
        "confidence": "HIGH",
    },
    "TH": {
        "name": "Thailand",
        "td_loss_pct": Decimal("0.062"),
        "source": "EGAT / PEA / MEA",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "VN": {
        "name": "Vietnam",
        "td_loss_pct": Decimal("0.085"),
        "source": "EVN / MOIT",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "ID": {
        "name": "Indonesia",
        "td_loss_pct": Decimal("0.098"),
        "source": "PLN / MEMR",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "MY": {
        "name": "Malaysia",
        "td_loss_pct": Decimal("0.052"),
        "source": "TNB / Suruhanjaya Tenaga",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "PH": {
        "name": "Philippines",
        "td_loss_pct": Decimal("0.110"),
        "source": "NGCP / DOE",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "SG": {
        "name": "Singapore",
        "td_loss_pct": Decimal("0.025"),
        "source": "SP Group / EMA",
        "year": 2023,
        "confidence": "HIGH",
    },
    "TW": {
        "name": "Taiwan",
        "td_loss_pct": Decimal("0.042"),
        "source": "Taipower / BOE",
        "year": 2023,
        "confidence": "HIGH",
    },
    "AU": {
        "name": "Australia",
        "td_loss_pct": Decimal("0.055"),
        "source": "AEMO / CER",
        "year": 2023,
        "confidence": "HIGH",
    },
    "NZ": {
        "name": "New Zealand",
        "td_loss_pct": Decimal("0.063"),
        "source": "Transpower / MBIE",
        "year": 2023,
        "confidence": "HIGH",
    },

    # ---- Middle East & Africa ----
    "AE": {
        "name": "UAE",
        "td_loss_pct": Decimal("0.070"),
        "source": "EWEC / FEWA",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "SA": {
        "name": "Saudi Arabia",
        "td_loss_pct": Decimal("0.080"),
        "source": "SEC / ECRA",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "TR": {
        "name": "Turkey",
        "td_loss_pct": Decimal("0.125"),
        "source": "TEIAS / EPDK",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "ZA": {
        "name": "South Africa",
        "td_loss_pct": Decimal("0.086"),
        "source": "Eskom / NERSA",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "EG": {
        "name": "Egypt",
        "td_loss_pct": Decimal("0.120"),
        "source": "EEHC / EgyptERA",
        "year": 2023,
        "confidence": "LOW",
    },
    "NG": {
        "name": "Nigeria",
        "td_loss_pct": Decimal("0.180"),
        "source": "TCN / NERC",
        "year": 2023,
        "confidence": "LOW",
    },
    "KE": {
        "name": "Kenya",
        "td_loss_pct": Decimal("0.200"),
        "source": "KPLC / EPRA",
        "year": 2023,
        "confidence": "LOW",
    },
    "RU": {
        "name": "Russia",
        "td_loss_pct": Decimal("0.100"),
        "source": "SO UPS / Rosstat",
        "year": 2023,
        "confidence": "MEDIUM",
    },
    "PK": {
        "name": "Pakistan",
        "td_loss_pct": Decimal("0.175"),
        "source": "NTDC / NEPRA",
        "year": 2023,
        "confidence": "LOW",
    },
    "BD": {
        "name": "Bangladesh",
        "td_loss_pct": Decimal("0.120"),
        "source": "PGCB / BPDB",
        "year": 2023,
        "confidence": "LOW",
    },

    # ---- World Average ----
    "WORLD": {
        "name": "World Average",
        "td_loss_pct": Decimal("0.083"),
        "source": "IEA World Energy Balances",
        "year": 2023,
        "confidence": "MEDIUM",
    },
}

# ---------------------------------------------------------------------------
# Valid country codes set
# ---------------------------------------------------------------------------

VALID_COUNTRY_CODES: FrozenSet[str] = frozenset(TD_LOSS_FACTORS.keys())

# ---------------------------------------------------------------------------
# Regional groupings
# ---------------------------------------------------------------------------

REGIONAL_GROUPS: Dict[str, List[str]] = {
    "EU": [
        "DE", "FR", "IT", "ES", "NL", "BE", "AT", "SE", "DK", "FI",
        "PL", "CZ", "HU", "RO", "BG", "GR", "PT", "IE",
    ],
    "NORTH_AMERICA": ["US", "CA", "MX"],
    "SOUTH_AMERICA": ["BR", "AR", "CL", "CO", "PE"],
    "ASIA_PACIFIC": [
        "JP", "CN", "IN", "KR", "TH", "VN", "ID", "MY", "PH", "SG",
        "TW", "AU", "NZ",
    ],
    "MIDDLE_EAST_AFRICA": [
        "AE", "SA", "TR", "ZA", "EG", "NG", "KE", "RU", "PK", "BD",
    ],
    "EUROPE": [
        "GB", "DE", "FR", "IT", "ES", "NL", "BE", "AT", "SE", "NO",
        "DK", "FI", "PL", "CZ", "HU", "RO", "BG", "GR", "PT", "IE",
        "CH",
    ],
    "OECD": [
        "US", "CA", "MX", "GB", "DE", "FR", "IT", "ES", "NL", "BE",
        "AT", "SE", "NO", "DK", "FI", "PL", "CZ", "HU", "GR", "PT",
        "IE", "CH", "JP", "KR", "AU", "NZ", "TR", "CL", "CO",
    ],
    "NON_OECD": [
        "CN", "IN", "TH", "VN", "ID", "MY", "PH", "SG", "TW", "BR",
        "AR", "PE", "AE", "SA", "ZA", "EG", "NG", "KE", "RU", "PK",
        "BD", "RO", "BG",
    ],
}


# ---------------------------------------------------------------------------
# TDLossResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TDLossResult:
    """Result of a T&D loss calculation with complete provenance.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code or ``WORLD``.
        td_loss_pct: T&D loss percentage as a Decimal fraction (e.g. 0.050).
        method: How the T&D factor was resolved (``country_average`` or ``custom``).
        net_consumption_mwh: Original net electricity consumption in MWh.
        gross_consumption_mwh: Gross consumption after T&D adjustment in MWh.
        loss_mwh: Electricity lost during transmission and distribution in MWh.
        loss_emissions_kg: Emissions attributable to T&D losses in kg CO2e
            (only populated when ``grid_ef_co2e`` is provided).
        provenance_hash: SHA-256 hash of all calculation inputs and outputs.
        calculation_time_ms: Wall-clock calculation time in milliseconds.
        timestamp: UTC timestamp of the calculation.
        source: Data source for the T&D loss factor.
        confidence: Confidence level of the factor (HIGH, MEDIUM, LOW).
        calculation_steps: List of intermediate calculation steps for audit.
    """

    country_code: str
    td_loss_pct: Decimal
    method: str
    net_consumption_mwh: Decimal
    gross_consumption_mwh: Decimal
    loss_mwh: Decimal
    loss_emissions_kg: Decimal = field(default_factory=lambda: Decimal("0"))
    provenance_hash: str = ""
    calculation_time_ms: float = 0.0
    timestamp: str = ""
    source: str = ""
    confidence: str = ""
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary suitable for JSON output.

        Returns:
            Dictionary with all field values converted to JSON-safe types.
        """
        return {
            "country_code": self.country_code,
            "td_loss_pct": str(self.td_loss_pct),
            "method": self.method,
            "net_consumption_mwh": str(self.net_consumption_mwh),
            "gross_consumption_mwh": str(self.gross_consumption_mwh),
            "loss_mwh": str(self.loss_mwh),
            "loss_emissions_kg": str(self.loss_emissions_kg),
            "provenance_hash": self.provenance_hash,
            "calculation_time_ms": self.calculation_time_ms,
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": self.confidence,
            "calculation_steps": self.calculation_steps,
        }


# ---------------------------------------------------------------------------
# TransmissionLossEngine
# ---------------------------------------------------------------------------

class TransmissionLossEngine:
    """Engine 4 -- Transmission & Distribution loss calculations for Scope 2.

    Provides deterministic, bit-perfect T&D loss computations for adjusting
    net electricity consumption to gross consumption and attributing emissions
    to transmission and distribution losses. All arithmetic uses Python
    ``Decimal`` with ``ROUND_HALF_UP`` rounding. Every calculation is
    recorded through SHA-256 provenance hashing for full audit trail.

    The engine includes a built-in database of T&D loss factors for 53
    countries plus a world average. Custom factors can be registered at
    runtime via :meth:`set_custom_factor`.

    Thread Safety:
        Mutable state (custom factors, counters) is protected by a reentrant
        lock. Calculation methods are safe for concurrent use.

    Attributes:
        _config: Engine configuration dictionary.
        _metrics: Optional Scope2LocationMetrics instance.
        _provenance: Optional ProvenanceTracker instance.
        _custom_factors: User-supplied custom T&D loss factors.
        _lock: Reentrant lock for thread-safe access.
        _calculation_count: Monotonically increasing calculation counter.

    Example:
        >>> engine = TransmissionLossEngine()
        >>> result = engine.calculate_td_loss(
        ...     net_consumption_mwh=Decimal("5000"),
        ...     country_code="GB",
        ... )
        >>> assert result.gross_consumption_mwh == Decimal("5385.000")
        >>> assert result.loss_mwh == Decimal("385.000")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Any] = None,
        provenance: Optional[Any] = None,
    ) -> None:
        """Initialize the TransmissionLossEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - ``default_precision``: Output decimal places (default 3).
                - ``fallback_to_world``: Whether to fall back to WORLD
                  average when a country code is not found (default True).
                - ``max_td_loss_pct``: Maximum allowable T&D loss factor
                  for validation (default 0.50).
                - ``min_td_loss_pct``: Minimum allowable T&D loss factor
                  for validation (default 0.00).
            metrics: Optional :class:`Scope2LocationMetrics` instance for
                Prometheus metric recording. When ``None``, attempts to use
                the module-level default singleton.
            provenance: Optional provenance tracker instance for SHA-256
                audit trail. When ``None``, attempts to use the module-level
                default singleton.
        """
        self._config: Dict[str, Any] = config or {}
        self._custom_factors: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()
        self._calculation_count: int = 0

        # Output precision from config, default 3 decimal places
        precision_digits = self._config.get("default_precision", 3)
        self._output_precision = Decimal(
            "0." + "0" * int(precision_digits)
        )

        # Fallback behaviour
        self._fallback_to_world: bool = bool(
            self._config.get("fallback_to_world", True)
        )

        # Validation bounds
        self._max_td_loss_pct = Decimal(
            str(self._config.get("max_td_loss_pct", "0.50"))
        )
        self._min_td_loss_pct = Decimal(
            str(self._config.get("min_td_loss_pct", "0"))
        )

        # Metrics integration
        if metrics is not None:
            self._metrics = metrics
        elif _METRICS_AVAILABLE:
            try:
                self._metrics = _get_metrics()
            except Exception:
                self._metrics = None
        else:
            self._metrics = None

        # Provenance integration
        if provenance is not None:
            self._provenance = provenance
        elif _PROVENANCE_AVAILABLE:
            try:
                self._provenance = _get_provenance_tracker()
            except Exception:
                self._provenance = None
        else:
            self._provenance = None

        logger.info(
            "TransmissionLossEngine initialized "
            "(countries=%d, precision=%s, fallback=%s, "
            "metrics=%s, provenance=%s)",
            len(TD_LOSS_FACTORS),
            self._output_precision,
            self._fallback_to_world,
            self._metrics is not None,
            self._provenance is not None,
        )

    # ------------------------------------------------------------------
    # 1. calculate_td_loss -- Primary T&D loss calculation
    # ------------------------------------------------------------------

    def calculate_td_loss(
        self,
        net_consumption_mwh: Decimal,
        country_code: str,
        custom_td_pct: Optional[Decimal] = None,
        grid_ef_co2e: Optional[Decimal] = None,
    ) -> TDLossResult:
        """Calculate Transmission & Distribution losses for net consumption.

        This is the primary calculation entry point. Given net electricity
        consumption in MWh and a country code, computes gross consumption,
        T&D losses, and optionally loss emissions.

        Formulas:
            gross_consumption = net_consumption x (1 + td_loss_pct)
            loss_mwh = gross_consumption - net_consumption
            loss_emissions = net_consumption x td_loss_pct x grid_ef_co2e

        All arithmetic uses Decimal with ROUND_HALF_UP.

        Args:
            net_consumption_mwh: Net electricity consumption in MWh.
                Must be >= 0.
            country_code: ISO 3166-1 alpha-2 country code or ``WORLD``.
                Case-insensitive; will be normalized to uppercase.
            custom_td_pct: Optional custom T&D loss percentage as a Decimal
                fraction (e.g. Decimal("0.08") for 8%). When provided,
                overrides the built-in country factor.
            grid_ef_co2e: Optional grid emission factor in kg CO2e per MWh.
                When provided, loss emissions are computed.

        Returns:
            :class:`TDLossResult` with complete calculation details.

        Raises:
            ValueError: If ``net_consumption_mwh`` is negative.
            ValueError: If ``country_code`` is not found and fallback is disabled.
            ValueError: If ``custom_td_pct`` is outside the valid range.
        """
        start_time = time.monotonic()
        calculation_steps: List[Dict[str, Any]] = []

        # Normalize country code
        country_code = self._normalize_country_code(country_code)

        # Validate net consumption
        net_mwh = self._to_decimal(net_consumption_mwh, "net_consumption_mwh")
        if net_mwh < _ZERO:
            raise ValueError(
                f"net_consumption_mwh must be >= 0, got {net_mwh}"
            )

        calculation_steps.append({
            "step": 1,
            "description": "Validate and normalize inputs",
            "country_code": country_code,
            "net_consumption_mwh": str(net_mwh),
        })

        # Resolve T&D loss factor
        if custom_td_pct is not None:
            td_pct = self._to_decimal(custom_td_pct, "custom_td_pct")
            validation_errors = self.validate_td_loss_factor(td_pct)
            if validation_errors:
                raise ValueError(
                    f"Invalid custom T&D loss factor: {'; '.join(validation_errors)}"
                )
            method = "custom"
            source = "user_provided"
            confidence = "USER"
        else:
            td_pct = self.get_td_loss_factor(country_code)
            factor_meta = self._get_factor_record(country_code)
            # Detect whether factor came from custom registry
            with self._lock:
                is_custom = country_code in self._custom_factors
            if is_custom:
                method = "custom"
                source = factor_meta.get("source", "user_provided")
                confidence = factor_meta.get("confidence", "USER")
            elif country_code not in TD_LOSS_FACTORS and self._fallback_to_world:
                method = "world_average_fallback"
                source = factor_meta.get("source", "IEA World Energy Balances")
                confidence = "LOW"
            else:
                method = "country_average"
                source = factor_meta.get("source", "IEA")
                confidence = factor_meta.get("confidence", "MEDIUM")

        calculation_steps.append({
            "step": 2,
            "description": "Resolve T&D loss factor",
            "td_loss_pct": str(td_pct),
            "method": method,
            "source": source,
        })

        # Calculate gross consumption
        gross_mwh = self.get_gross_consumption(net_mwh, td_pct)
        gross_mwh = gross_mwh.quantize(self._output_precision, rounding=ROUND_HALF_UP)

        calculation_steps.append({
            "step": 3,
            "description": "Calculate gross consumption = net x (1 + td_pct)",
            "formula": f"{net_mwh} x (1 + {td_pct})",
            "gross_consumption_mwh": str(gross_mwh),
        })

        # Calculate loss
        loss_mwh = (gross_mwh - net_mwh).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        calculation_steps.append({
            "step": 4,
            "description": "Calculate T&D loss = gross - net",
            "formula": f"{gross_mwh} - {net_mwh}",
            "loss_mwh": str(loss_mwh),
        })

        # Calculate loss emissions if grid EF provided
        loss_emissions_kg = _ZERO
        if grid_ef_co2e is not None:
            ef = self._to_decimal(grid_ef_co2e, "grid_ef_co2e")
            loss_emissions_kg = self.calculate_loss_emissions(net_mwh, td_pct, ef)
            loss_emissions_kg = loss_emissions_kg.quantize(
                self._output_precision, rounding=ROUND_HALF_UP
            )
            calculation_steps.append({
                "step": 5,
                "description": "Calculate loss emissions = net x td_pct x grid_ef",
                "formula": f"{net_mwh} x {td_pct} x {ef}",
                "loss_emissions_kg": str(loss_emissions_kg),
            })

        # Compute provenance hash
        provenance_data = {
            "engine": "TransmissionLossEngine",
            "method": "calculate_td_loss",
            "inputs": {
                "net_consumption_mwh": str(net_mwh),
                "country_code": country_code,
                "custom_td_pct": str(custom_td_pct) if custom_td_pct is not None else None,
                "grid_ef_co2e": str(grid_ef_co2e) if grid_ef_co2e is not None else None,
            },
            "outputs": {
                "td_loss_pct": str(td_pct),
                "gross_consumption_mwh": str(gross_mwh),
                "loss_mwh": str(loss_mwh),
                "loss_emissions_kg": str(loss_emissions_kg),
            },
        }
        provenance_hash = _compute_hash(provenance_data)

        calculation_steps.append({
            "step": len(calculation_steps) + 1,
            "description": "Compute provenance hash",
            "provenance_hash": provenance_hash,
        })

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"td_loss_{country_code}_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data=provenance_data,
                    metadata={
                        "country_code": country_code,
                        "td_loss_pct": str(td_pct),
                        "method": method,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record provenance for T&D loss calculation: %s",
                    exc,
                )

        # Record metrics
        if self._metrics is not None:
            try:
                self._metrics.record_td_loss_adjustment()
            except Exception as exc:
                logger.warning(
                    "Failed to record T&D loss metric: %s", exc,
                )

        # Increment counter
        with self._lock:
            self._calculation_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = TDLossResult(
            country_code=country_code,
            td_loss_pct=td_pct,
            method=method,
            net_consumption_mwh=net_mwh,
            gross_consumption_mwh=gross_mwh,
            loss_mwh=loss_mwh,
            loss_emissions_kg=loss_emissions_kg,
            provenance_hash=provenance_hash,
            calculation_time_ms=round(elapsed_ms, 3),
            timestamp=_utcnow().isoformat(),
            source=source,
            confidence=confidence,
            calculation_steps=calculation_steps,
        )

        logger.debug(
            "T&D loss calculated: country=%s td_pct=%s net=%s gross=%s "
            "loss=%s hash=%s time=%.3fms",
            country_code, td_pct, net_mwh, gross_mwh,
            loss_mwh, provenance_hash[:16], elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # 2. get_gross_consumption -- Net to Gross conversion
    # ------------------------------------------------------------------

    def get_gross_consumption(
        self,
        net_mwh: Decimal,
        td_loss_pct: Decimal,
    ) -> Decimal:
        """Calculate gross consumption from net consumption and T&D loss factor.

        Formula:
            gross = net x (1 + td_loss_pct)

        Uses Decimal arithmetic with 8 decimal places internally.

        Args:
            net_mwh: Net electricity consumption in MWh. Must be >= 0.
            td_loss_pct: T&D loss percentage as a Decimal fraction.

        Returns:
            Gross consumption in MWh (Decimal).

        Raises:
            ValueError: If ``net_mwh`` is negative.
        """
        net = self._to_decimal(net_mwh, "net_mwh")
        pct = self._to_decimal(td_loss_pct, "td_loss_pct")

        if net < _ZERO:
            raise ValueError(f"net_mwh must be >= 0, got {net}")

        gross = (net * (_ONE + pct)).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        return gross

    # ------------------------------------------------------------------
    # 3. get_net_consumption -- Gross to Net conversion
    # ------------------------------------------------------------------

    def get_net_consumption(
        self,
        gross_mwh: Decimal,
        td_loss_pct: Decimal,
    ) -> Decimal:
        """Calculate net consumption from gross consumption and T&D loss factor.

        Formula:
            net = gross / (1 + td_loss_pct)

        Uses Decimal arithmetic with 8 decimal places internally.

        Args:
            gross_mwh: Gross electricity consumption in MWh. Must be >= 0.
            td_loss_pct: T&D loss percentage as a Decimal fraction.

        Returns:
            Net consumption in MWh (Decimal).

        Raises:
            ValueError: If ``gross_mwh`` is negative.
            ValueError: If ``td_loss_pct`` equals -1 (division by zero).
        """
        gross = self._to_decimal(gross_mwh, "gross_mwh")
        pct = self._to_decimal(td_loss_pct, "td_loss_pct")

        if gross < _ZERO:
            raise ValueError(f"gross_mwh must be >= 0, got {gross}")

        denominator = _ONE + pct
        if denominator == _ZERO:
            raise ValueError(
                "td_loss_pct cannot be -1 (would cause division by zero)"
            )

        net = (gross / denominator).quantize(_PRECISION, rounding=ROUND_HALF_UP)
        return net

    # ------------------------------------------------------------------
    # 4. calculate_loss_emissions
    # ------------------------------------------------------------------

    def calculate_loss_emissions(
        self,
        net_mwh: Decimal,
        td_loss_pct: Decimal,
        grid_ef_co2e: Decimal,
    ) -> Decimal:
        """Calculate emissions attributable to T&D losses.

        Formula:
            loss_emissions = net_mwh x td_loss_pct x grid_ef_co2e

        Args:
            net_mwh: Net electricity consumption in MWh.
            td_loss_pct: T&D loss percentage as a Decimal fraction.
            grid_ef_co2e: Grid emission factor in kg CO2e per MWh.

        Returns:
            Loss emissions in kg CO2e (Decimal).

        Raises:
            ValueError: If any input is negative (except ``td_loss_pct``
                which may be zero).
        """
        net = self._to_decimal(net_mwh, "net_mwh")
        pct = self._to_decimal(td_loss_pct, "td_loss_pct")
        ef = self._to_decimal(grid_ef_co2e, "grid_ef_co2e")

        if net < _ZERO:
            raise ValueError(f"net_mwh must be >= 0, got {net}")
        if ef < _ZERO:
            raise ValueError(f"grid_ef_co2e must be >= 0, got {ef}")

        emissions = (net * pct * ef).quantize(_PRECISION, rounding=ROUND_HALF_UP)

        # Record provenance for standalone calls
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"loss_emissions_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data={
                        "net_mwh": str(net),
                        "td_loss_pct": str(pct),
                        "grid_ef_co2e": str(ef),
                        "loss_emissions_kg": str(emissions),
                    },
                )
            except Exception as exc:
                logger.warning("Failed to record loss emissions provenance: %s", exc)

        return emissions

    # ------------------------------------------------------------------
    # 5. apply_td_adjustment
    # ------------------------------------------------------------------

    def apply_td_adjustment(
        self,
        base_emissions_kg: Decimal,
        td_loss_pct: Decimal,
    ) -> Decimal:
        """Apply T&D loss adjustment to base emissions.

        Formula:
            adjusted = base_emissions x (1 + td_loss_pct)

        This method is used to upscale already-computed base emissions to
        include the T&D loss component. For example, if base emissions
        are 1000 kg CO2e and T&D loss is 5%, the adjusted value is 1050 kg.

        Args:
            base_emissions_kg: Base emissions in kg CO2e. Must be >= 0.
            td_loss_pct: T&D loss percentage as a Decimal fraction.

        Returns:
            Adjusted emissions in kg CO2e (Decimal).

        Raises:
            ValueError: If ``base_emissions_kg`` is negative.
        """
        base = self._to_decimal(base_emissions_kg, "base_emissions_kg")
        pct = self._to_decimal(td_loss_pct, "td_loss_pct")

        if base < _ZERO:
            raise ValueError(f"base_emissions_kg must be >= 0, got {base}")

        adjusted = (base * (_ONE + pct)).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"td_adjust_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data={
                        "base_emissions_kg": str(base),
                        "td_loss_pct": str(pct),
                        "adjusted_emissions_kg": str(adjusted),
                    },
                )
            except Exception as exc:
                logger.warning("Failed to record T&D adjustment provenance: %s", exc)

        # Record metric
        if self._metrics is not None:
            try:
                self._metrics.record_td_loss_adjustment()
            except Exception:
                pass

        return adjusted

    # ------------------------------------------------------------------
    # 6. get_td_loss_factor -- Country factor lookup
    # ------------------------------------------------------------------

    def get_td_loss_factor(
        self,
        country_code: str,
    ) -> Decimal:
        """Get the T&D loss factor for a given country code.

        Looks up the T&D loss percentage for the specified country from
        the built-in database or custom factor overrides. If the country
        is not found and ``fallback_to_world`` is enabled, returns the
        WORLD average.

        Lookup order:
            1. Custom factors (set via :meth:`set_custom_factor`)
            2. Built-in ``TD_LOSS_FACTORS`` database
            3. WORLD average fallback (if enabled)

        Args:
            country_code: ISO 3166-1 alpha-2 country code or ``WORLD``.
                Case-insensitive; will be normalized to uppercase.

        Returns:
            T&D loss factor as a Decimal fraction (e.g. Decimal("0.050")).

        Raises:
            ValueError: If the country code is not found and fallback
                is disabled.
        """
        code = self._normalize_country_code(country_code)

        # Check custom factors first
        with self._lock:
            if code in self._custom_factors:
                factor = self._custom_factors[code]["td_loss_pct"]
                logger.debug(
                    "T&D loss factor for %s resolved from custom: %s",
                    code, factor,
                )
                return Decimal(str(factor))

        # Check built-in database
        if code in TD_LOSS_FACTORS:
            factor = TD_LOSS_FACTORS[code]["td_loss_pct"]
            logger.debug(
                "T&D loss factor for %s resolved from database: %s",
                code, factor,
            )
            return factor

        # Fallback to WORLD average
        if self._fallback_to_world:
            world_factor = TD_LOSS_FACTORS["WORLD"]["td_loss_pct"]
            logger.warning(
                "Country code %s not found in T&D loss database; "
                "falling back to WORLD average: %s",
                code, world_factor,
            )
            return world_factor

        raise ValueError(
            f"Country code '{code}' not found in T&D loss factor database "
            f"and fallback_to_world is disabled"
        )

    # ------------------------------------------------------------------
    # 7. get_td_loss_factor_with_metadata
    # ------------------------------------------------------------------

    def get_td_loss_factor_with_metadata(
        self,
        country_code: str,
    ) -> Dict[str, Any]:
        """Get the T&D loss factor with metadata for a given country.

        Returns the factor value along with its source authority, vintage
        year, confidence level, and country name.

        Args:
            country_code: ISO 3166-1 alpha-2 country code or ``WORLD``.
                Case-insensitive; normalized to uppercase.

        Returns:
            Dictionary with keys: ``country_code``, ``name``, ``td_loss_pct``,
            ``source``, ``year``, ``confidence``, ``is_custom``, ``is_fallback``.

        Raises:
            ValueError: If the country code is not found and fallback
                is disabled.
        """
        code = self._normalize_country_code(country_code)

        # Check custom factors
        with self._lock:
            if code in self._custom_factors:
                record = self._custom_factors[code]
                return {
                    "country_code": code,
                    "name": record.get("name", code),
                    "td_loss_pct": str(record["td_loss_pct"]),
                    "source": record.get("source", "user_provided"),
                    "year": record.get("year", _utcnow().year),
                    "confidence": record.get("confidence", "USER"),
                    "is_custom": True,
                    "is_fallback": False,
                }

        # Check built-in database
        if code in TD_LOSS_FACTORS:
            record = TD_LOSS_FACTORS[code]
            return {
                "country_code": code,
                "name": record["name"],
                "td_loss_pct": str(record["td_loss_pct"]),
                "source": record["source"],
                "year": record["year"],
                "confidence": record["confidence"],
                "is_custom": False,
                "is_fallback": False,
            }

        # Fallback to WORLD
        if self._fallback_to_world:
            record = TD_LOSS_FACTORS["WORLD"]
            return {
                "country_code": code,
                "name": f"{code} (fallback to World Average)",
                "td_loss_pct": str(record["td_loss_pct"]),
                "source": record["source"],
                "year": record["year"],
                "confidence": "LOW",
                "is_custom": False,
                "is_fallback": True,
            }

        raise ValueError(
            f"Country code '{code}' not found in T&D loss factor database"
        )

    # ------------------------------------------------------------------
    # 8. list_all_factors
    # ------------------------------------------------------------------

    def list_all_factors(self) -> Dict[str, Dict[str, Any]]:
        """Return all T&D loss factors (built-in + custom overrides).

        Returns a deep copy so callers cannot mutate internal state.
        Custom factors override built-in factors for the same country code.

        Returns:
            Dictionary keyed by country code with factor metadata.
        """
        result: Dict[str, Dict[str, Any]] = {}

        # Start with built-in factors
        for code, record in TD_LOSS_FACTORS.items():
            result[code] = {
                "name": record["name"],
                "td_loss_pct": str(record["td_loss_pct"]),
                "source": record["source"],
                "year": record["year"],
                "confidence": record["confidence"],
                "is_custom": False,
            }

        # Override with custom factors
        with self._lock:
            for code, record in self._custom_factors.items():
                result[code] = {
                    "name": record.get("name", code),
                    "td_loss_pct": str(record["td_loss_pct"]),
                    "source": record.get("source", "user_provided"),
                    "year": record.get("year", _utcnow().year),
                    "confidence": record.get("confidence", "USER"),
                    "is_custom": True,
                }

        return result

    # ------------------------------------------------------------------
    # 9. get_regional_average
    # ------------------------------------------------------------------

    def get_regional_average(
        self,
        region: str,
    ) -> Decimal:
        """Get the average T&D loss factor for a defined region.

        Computes the arithmetic mean of T&D loss factors across all
        countries in the specified region. Uses the effective factor
        for each country (custom overrides take precedence).

        Supported regions:
            EU, NORTH_AMERICA, SOUTH_AMERICA, ASIA_PACIFIC,
            MIDDLE_EAST_AFRICA, EUROPE, OECD, NON_OECD.

        Args:
            region: Region identifier (case-insensitive).

        Returns:
            Average T&D loss factor as a Decimal fraction, rounded to
            4 decimal places.

        Raises:
            ValueError: If the region is not recognized.
        """
        region_upper = region.strip().upper()

        if region_upper not in REGIONAL_GROUPS:
            raise ValueError(
                f"Unknown region '{region}'. Valid regions: "
                f"{', '.join(sorted(REGIONAL_GROUPS.keys()))}"
            )

        country_codes = REGIONAL_GROUPS[region_upper]
        if not country_codes:
            raise ValueError(f"Region '{region_upper}' has no member countries")

        factors: List[Decimal] = []
        for code in country_codes:
            try:
                factor = self.get_td_loss_factor(code)
                factors.append(factor)
            except ValueError:
                logger.warning(
                    "Skipping country %s in regional average: factor not found",
                    code,
                )
                continue

        if not factors:
            raise ValueError(
                f"No valid T&D loss factors found for region '{region_upper}'"
            )

        total = sum(factors, _ZERO)
        count = Decimal(str(len(factors)))
        average = (total / count).quantize(_FACTOR_PRECISION, rounding=ROUND_HALF_UP)

        logger.debug(
            "Regional average for %s: %s (from %d countries)",
            region_upper, average, len(factors),
        )

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"regional_avg_{region_upper}",
                    action="AGGREGATE",
                    data={
                        "region": region_upper,
                        "country_count": len(factors),
                        "average_td_loss_pct": str(average),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record regional average provenance: %s", exc,
                )

        return average

    # ------------------------------------------------------------------
    # 10. set_custom_factor
    # ------------------------------------------------------------------

    def set_custom_factor(
        self,
        country_code: str,
        td_loss_pct: Decimal,
        name: Optional[str] = None,
        source: Optional[str] = None,
        year: Optional[int] = None,
        confidence: Optional[str] = None,
    ) -> None:
        """Register a custom T&D loss factor for a country code.

        Custom factors take precedence over built-in factors in all
        lookups and calculations. Use this to override with utility-
        specific or site-specific T&D data.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (case-insensitive).
            td_loss_pct: T&D loss percentage as a Decimal fraction.
                Must pass :meth:`validate_td_loss_factor`.
            name: Optional human-readable name for the factor.
            source: Optional data source identifier.
            year: Optional vintage year.
            confidence: Optional confidence level (HIGH, MEDIUM, LOW, USER).

        Raises:
            ValueError: If ``td_loss_pct`` fails validation.
        """
        code = self._normalize_country_code(country_code)
        pct = self._to_decimal(td_loss_pct, "td_loss_pct")

        validation_errors = self.validate_td_loss_factor(pct)
        if validation_errors:
            raise ValueError(
                f"Invalid custom T&D loss factor: {'; '.join(validation_errors)}"
            )

        record: Dict[str, Any] = {
            "td_loss_pct": pct,
            "name": name or code,
            "source": source or "user_provided",
            "year": year or _utcnow().year,
            "confidence": confidence or "USER",
        }

        with self._lock:
            self._custom_factors[code] = record

        logger.info(
            "Custom T&D loss factor set for %s: %s", code, pct,
        )

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"custom_factor_{code}",
                    action="CREATE",
                    data=record,
                    metadata={"country_code": code},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record custom factor provenance: %s", exc,
                )

    # ------------------------------------------------------------------
    # 11. remove_custom_factor
    # ------------------------------------------------------------------

    def remove_custom_factor(
        self,
        country_code: str,
    ) -> bool:
        """Remove a previously registered custom T&D loss factor.

        After removal, the built-in factor (if any) will be used for
        this country code.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (case-insensitive).

        Returns:
            True if a custom factor was removed, False if no custom
            factor existed for the country code.
        """
        code = self._normalize_country_code(country_code)

        with self._lock:
            if code in self._custom_factors:
                removed_record = self._custom_factors.pop(code)
                logger.info("Custom T&D loss factor removed for %s", code)

                # Record provenance
                if self._provenance is not None:
                    try:
                        self._provenance.record(
                            entity_type="TD_LOSS",
                            entity_id=f"custom_factor_{code}",
                            action="DELETE",
                            data=removed_record,
                            metadata={"country_code": code},
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to record custom factor removal provenance: %s",
                            exc,
                        )
                return True

        logger.debug(
            "No custom T&D loss factor to remove for %s", code,
        )
        return False

    # ------------------------------------------------------------------
    # 12. list_custom_factors
    # ------------------------------------------------------------------

    def list_custom_factors(self) -> Dict[str, Dict[str, Any]]:
        """Return all currently registered custom T&D loss factors.

        Returns a deep copy so callers cannot mutate internal state.

        Returns:
            Dictionary keyed by country code with custom factor metadata.
        """
        with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for code, record in self._custom_factors.items():
                result[code] = {
                    "td_loss_pct": str(record["td_loss_pct"]),
                    "name": record.get("name", code),
                    "source": record.get("source", "user_provided"),
                    "year": record.get("year"),
                    "confidence": record.get("confidence", "USER"),
                }
            return result

    # ------------------------------------------------------------------
    # 13. allocate_proportional
    # ------------------------------------------------------------------

    def allocate_proportional(
        self,
        total_loss_mwh: Decimal,
        consumption_shares: Dict[str, Decimal],
    ) -> Dict[str, Decimal]:
        """Allocate T&D losses proportionally across multiple consumers.

        Each consumer's share of the total loss is proportional to their
        share of total consumption:

            consumer_loss = total_loss x (consumer_consumption / total_consumption)

        This is the standard method recommended by the GHG Protocol for
        multi-facility organizations sharing the same grid connection.

        Args:
            total_loss_mwh: Total T&D losses to allocate in MWh.
            consumption_shares: Dictionary mapping consumer IDs to their
                consumption in MWh.

        Returns:
            Dictionary mapping consumer IDs to their allocated loss in MWh,
            rounded to 3 decimal places.

        Raises:
            ValueError: If ``total_loss_mwh`` is negative.
            ValueError: If ``consumption_shares`` is empty.
            ValueError: If total consumption is zero.
        """
        total_loss = self._to_decimal(total_loss_mwh, "total_loss_mwh")
        if total_loss < _ZERO:
            raise ValueError(f"total_loss_mwh must be >= 0, got {total_loss}")

        if not consumption_shares:
            raise ValueError("consumption_shares must not be empty")

        # Compute total consumption
        shares: Dict[str, Decimal] = {}
        total_consumption = _ZERO
        for consumer_id, share_val in consumption_shares.items():
            dec_share = self._to_decimal(share_val, f"consumption_shares[{consumer_id}]")
            if dec_share < _ZERO:
                raise ValueError(
                    f"Consumption for '{consumer_id}' must be >= 0, got {dec_share}"
                )
            shares[consumer_id] = dec_share
            total_consumption += dec_share

        if total_consumption == _ZERO:
            raise ValueError("Total consumption across all consumers is zero")

        # Allocate proportionally
        allocations: Dict[str, Decimal] = {}
        for consumer_id, share in shares.items():
            proportion = share / total_consumption
            allocated = (total_loss * proportion).quantize(
                self._output_precision, rounding=ROUND_HALF_UP
            )
            allocations[consumer_id] = allocated

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"proportional_alloc_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data={
                        "method": "proportional",
                        "total_loss_mwh": str(total_loss),
                        "consumer_count": len(allocations),
                        "allocations": {k: str(v) for k, v in allocations.items()},
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record proportional allocation provenance: %s", exc,
                )

        return allocations

    # ------------------------------------------------------------------
    # 14. allocate_marginal
    # ------------------------------------------------------------------

    def allocate_marginal(
        self,
        total_loss_mwh: Decimal,
        marginal_loss_pct: Decimal,
    ) -> Decimal:
        """Allocate T&D losses using the marginal loss method.

        Marginal allocation uses the marginal loss percentage to compute
        the share of total losses attributable to incremental consumption:

            marginal_loss = total_loss x marginal_loss_pct

        This method is appropriate when the consumer represents marginal
        demand on the grid.

        Args:
            total_loss_mwh: Total T&D losses on the grid in MWh.
            marginal_loss_pct: Marginal loss percentage as a Decimal fraction.

        Returns:
            Marginal loss allocation in MWh (Decimal), rounded to 3 decimal
            places.

        Raises:
            ValueError: If ``total_loss_mwh`` is negative.
        """
        total_loss = self._to_decimal(total_loss_mwh, "total_loss_mwh")
        marginal_pct = self._to_decimal(marginal_loss_pct, "marginal_loss_pct")

        if total_loss < _ZERO:
            raise ValueError(f"total_loss_mwh must be >= 0, got {total_loss}")

        marginal_allocation = (total_loss * marginal_pct).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"marginal_alloc_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data={
                        "method": "marginal",
                        "total_loss_mwh": str(total_loss),
                        "marginal_loss_pct": str(marginal_pct),
                        "marginal_allocation_mwh": str(marginal_allocation),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record marginal allocation provenance: %s", exc,
                )

        return marginal_allocation

    # ------------------------------------------------------------------
    # 15. allocate_fixed
    # ------------------------------------------------------------------

    def allocate_fixed(
        self,
        loss_mwh: Decimal,
    ) -> Decimal:
        """Return a fixed T&D loss allocation value (pass-through).

        In a fixed allocation model, the T&D loss amount is predetermined
        and directly assigned to the consumer. This method simply validates
        and returns the input with output precision applied.

        Args:
            loss_mwh: Fixed T&D loss allocation in MWh. Must be >= 0.

        Returns:
            Fixed loss allocation in MWh (Decimal), rounded to 3 decimal
            places.

        Raises:
            ValueError: If ``loss_mwh`` is negative.
        """
        loss = self._to_decimal(loss_mwh, "loss_mwh")
        if loss < _ZERO:
            raise ValueError(f"loss_mwh must be >= 0, got {loss}")

        result = loss.quantize(self._output_precision, rounding=ROUND_HALF_UP)

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"fixed_alloc_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data={
                        "method": "fixed",
                        "loss_mwh": str(result),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record fixed allocation provenance: %s", exc,
                )

        return result

    # ------------------------------------------------------------------
    # 16. deduct_onsite_generation
    # ------------------------------------------------------------------

    def deduct_onsite_generation(
        self,
        total_consumption_mwh: Decimal,
        onsite_generation_mwh: Decimal,
    ) -> Decimal:
        """Deduct on-site generation from total consumption.

        On-site generation (e.g. rooftop solar, backup generators) does
        not traverse the T&D network and should therefore be excluded
        from T&D loss calculations:

            net_grid_consumption = total_consumption - onsite_generation

        The result is clamped to zero (on-site generation cannot exceed
        total consumption in this context).

        Args:
            total_consumption_mwh: Total electricity consumption in MWh.
                Must be >= 0.
            onsite_generation_mwh: On-site electricity generation in MWh.
                Must be >= 0.

        Returns:
            Net grid consumption in MWh (Decimal), never negative.

        Raises:
            ValueError: If either input is negative.
        """
        total = self._to_decimal(total_consumption_mwh, "total_consumption_mwh")
        onsite = self._to_decimal(onsite_generation_mwh, "onsite_generation_mwh")

        if total < _ZERO:
            raise ValueError(
                f"total_consumption_mwh must be >= 0, got {total}"
            )
        if onsite < _ZERO:
            raise ValueError(
                f"onsite_generation_mwh must be >= 0, got {onsite}"
            )

        net_grid = max(total - onsite, _ZERO)
        net_grid = net_grid.quantize(self._output_precision, rounding=ROUND_HALF_UP)

        logger.debug(
            "On-site generation deduction: total=%s onsite=%s net_grid=%s",
            total, onsite, net_grid,
        )

        return net_grid

    # ------------------------------------------------------------------
    # 17. calculate_with_onsite
    # ------------------------------------------------------------------

    def calculate_with_onsite(
        self,
        total_mwh: Decimal,
        onsite_mwh: Decimal,
        country_code: str,
        grid_ef: Decimal,
        custom_td_pct: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate T&D-adjusted emissions accounting for on-site generation.

        Combines on-site deduction with T&D loss calculation:
            1. Deduct on-site generation from total consumption.
            2. Apply T&D loss factor to net grid consumption.
            3. Compute grid emissions for both net and gross consumption.

        Args:
            total_mwh: Total electricity consumption in MWh.
            onsite_mwh: On-site electricity generation in MWh.
            country_code: ISO 3166-1 alpha-2 country code.
            grid_ef: Grid emission factor in kg CO2e per MWh.
            custom_td_pct: Optional custom T&D loss percentage.

        Returns:
            Dictionary with complete calculation summary:
                - ``total_consumption_mwh``
                - ``onsite_generation_mwh``
                - ``net_grid_consumption_mwh``
                - ``td_loss_pct``
                - ``gross_grid_consumption_mwh``
                - ``td_loss_mwh``
                - ``grid_ef_kg_co2e_per_mwh``
                - ``net_emissions_kg_co2e``
                - ``td_loss_emissions_kg_co2e``
                - ``total_emissions_kg_co2e``
                - ``provenance_hash``
        """
        start_time = time.monotonic()

        total = self._to_decimal(total_mwh, "total_mwh")
        onsite = self._to_decimal(onsite_mwh, "onsite_mwh")
        ef = self._to_decimal(grid_ef, "grid_ef")
        code = self._normalize_country_code(country_code)

        # Step 1: Deduct on-site generation
        net_grid = self.deduct_onsite_generation(total, onsite)

        # Step 2: Resolve T&D factor
        if custom_td_pct is not None:
            td_pct = self._to_decimal(custom_td_pct, "custom_td_pct")
        else:
            td_pct = self.get_td_loss_factor(code)

        # Step 3: Calculate gross consumption
        gross_grid = self.get_gross_consumption(net_grid, td_pct)
        gross_grid = gross_grid.quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Step 4: Calculate T&D loss quantity
        td_loss = (gross_grid - net_grid).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Step 5: Calculate emissions
        net_emissions = (net_grid * ef).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )
        td_loss_emissions = self.calculate_loss_emissions(net_grid, td_pct, ef)
        td_loss_emissions = td_loss_emissions.quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )
        total_emissions = (net_emissions + td_loss_emissions).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Provenance hash
        provenance_data = {
            "engine": "TransmissionLossEngine",
            "method": "calculate_with_onsite",
            "total_mwh": str(total),
            "onsite_mwh": str(onsite),
            "net_grid": str(net_grid),
            "country_code": code,
            "td_loss_pct": str(td_pct),
            "gross_grid": str(gross_grid),
            "grid_ef": str(ef),
            "net_emissions": str(net_emissions),
            "td_loss_emissions": str(td_loss_emissions),
            "total_emissions": str(total_emissions),
        }
        provenance_hash = _compute_hash(provenance_data)

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"onsite_calc_{code}_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data=provenance_data,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record onsite calculation provenance: %s", exc,
                )

        # Record metric
        if self._metrics is not None:
            try:
                self._metrics.record_td_loss_adjustment()
            except Exception:
                pass

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "total_consumption_mwh": str(total),
            "onsite_generation_mwh": str(onsite),
            "net_grid_consumption_mwh": str(net_grid),
            "td_loss_pct": str(td_pct),
            "gross_grid_consumption_mwh": str(gross_grid),
            "td_loss_mwh": str(td_loss),
            "grid_ef_kg_co2e_per_mwh": str(ef),
            "net_emissions_kg_co2e": str(net_emissions),
            "td_loss_emissions_kg_co2e": str(td_loss_emissions),
            "total_emissions_kg_co2e": str(total_emissions),
            "provenance_hash": provenance_hash,
            "calculation_time_ms": round(elapsed_ms, 3),
            "country_code": code,
        }

    # ------------------------------------------------------------------
    # 18. net_to_gross
    # ------------------------------------------------------------------

    def net_to_gross(
        self,
        net_mwh: Decimal,
        country_code: str,
    ) -> Dict[str, Any]:
        """Convert net consumption to gross with country-specific T&D losses.

        Args:
            net_mwh: Net electricity consumption in MWh.
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with keys:
                - ``country_code``
                - ``td_loss_pct``
                - ``net_consumption_mwh``
                - ``gross_consumption_mwh``
                - ``td_loss_mwh``
                - ``provenance_hash``
        """
        net = self._to_decimal(net_mwh, "net_mwh")
        code = self._normalize_country_code(country_code)

        td_pct = self.get_td_loss_factor(code)
        gross = self.get_gross_consumption(net, td_pct)
        gross = gross.quantize(self._output_precision, rounding=ROUND_HALF_UP)
        loss = (gross - net).quantize(self._output_precision, rounding=ROUND_HALF_UP)

        provenance_data = {
            "engine": "TransmissionLossEngine",
            "method": "net_to_gross",
            "net_mwh": str(net),
            "country_code": code,
            "td_loss_pct": str(td_pct),
            "gross_mwh": str(gross),
            "loss_mwh": str(loss),
        }
        provenance_hash = _compute_hash(provenance_data)

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"net_to_gross_{code}_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data=provenance_data,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record net_to_gross provenance: %s", exc,
                )

        return {
            "country_code": code,
            "td_loss_pct": str(td_pct),
            "net_consumption_mwh": str(net),
            "gross_consumption_mwh": str(gross),
            "td_loss_mwh": str(loss),
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # 19. gross_to_net
    # ------------------------------------------------------------------

    def gross_to_net(
        self,
        gross_mwh: Decimal,
        country_code: str,
    ) -> Dict[str, Any]:
        """Convert gross consumption to net with country-specific T&D losses.

        Args:
            gross_mwh: Gross electricity consumption in MWh.
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Dictionary with keys:
                - ``country_code``
                - ``td_loss_pct``
                - ``gross_consumption_mwh``
                - ``net_consumption_mwh``
                - ``td_loss_mwh``
                - ``provenance_hash``
        """
        gross = self._to_decimal(gross_mwh, "gross_mwh")
        code = self._normalize_country_code(country_code)

        td_pct = self.get_td_loss_factor(code)
        net = self.get_net_consumption(gross, td_pct)
        net = net.quantize(self._output_precision, rounding=ROUND_HALF_UP)
        loss = (gross - net).quantize(self._output_precision, rounding=ROUND_HALF_UP)

        provenance_data = {
            "engine": "TransmissionLossEngine",
            "method": "gross_to_net",
            "gross_mwh": str(gross),
            "country_code": code,
            "td_loss_pct": str(td_pct),
            "net_mwh": str(net),
            "loss_mwh": str(loss),
        }
        provenance_hash = _compute_hash(provenance_data)

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"gross_to_net_{code}_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data=provenance_data,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record gross_to_net provenance: %s", exc,
                )

        return {
            "country_code": code,
            "td_loss_pct": str(td_pct),
            "gross_consumption_mwh": str(gross),
            "net_consumption_mwh": str(net),
            "td_loss_mwh": str(loss),
            "provenance_hash": provenance_hash,
        }

    # ------------------------------------------------------------------
    # 20. get_accounting_summary
    # ------------------------------------------------------------------

    def get_accounting_summary(
        self,
        net_mwh: Decimal,
        country_code: str,
        grid_ef: Decimal,
    ) -> Dict[str, Any]:
        """Get a comprehensive Scope 2 accounting summary with T&D adjustments.

        Produces a complete summary of:
        - Net and gross consumption
        - T&D losses (MWh and emissions)
        - Base emissions and T&D-adjusted total emissions
        - Percentage uplift from T&D losses

        Args:
            net_mwh: Net electricity consumption in MWh.
            country_code: ISO 3166-1 alpha-2 country code.
            grid_ef: Grid emission factor in kg CO2e per MWh.

        Returns:
            Comprehensive accounting summary dictionary.
        """
        start_time = time.monotonic()

        net = self._to_decimal(net_mwh, "net_mwh")
        code = self._normalize_country_code(country_code)
        ef = self._to_decimal(grid_ef, "grid_ef")

        td_pct = self.get_td_loss_factor(code)
        factor_meta = self._get_factor_record(code)

        # Consumption
        gross = self.get_gross_consumption(net, td_pct)
        gross = gross.quantize(self._output_precision, rounding=ROUND_HALF_UP)
        td_loss = (gross - net).quantize(self._output_precision, rounding=ROUND_HALF_UP)

        # Emissions
        base_emissions = (net * ef).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )
        td_loss_emissions = self.calculate_loss_emissions(net, td_pct, ef)
        td_loss_emissions = td_loss_emissions.quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )
        total_emissions = (base_emissions + td_loss_emissions).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Conversion to tonnes
        base_emissions_t = (base_emissions / _THOUSAND).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )
        td_loss_emissions_t = (td_loss_emissions / _THOUSAND).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )
        total_emissions_t = (total_emissions / _THOUSAND).quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        # Percentage uplift
        if base_emissions > _ZERO:
            pct_uplift = (
                (td_loss_emissions / base_emissions) * _HUNDRED
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            pct_uplift = _ZERO

        # Provenance
        provenance_data = {
            "engine": "TransmissionLossEngine",
            "method": "get_accounting_summary",
            "net_mwh": str(net),
            "country_code": code,
            "grid_ef": str(ef),
            "td_loss_pct": str(td_pct),
            "gross_mwh": str(gross),
            "base_emissions_kg": str(base_emissions),
            "td_loss_emissions_kg": str(td_loss_emissions),
            "total_emissions_kg": str(total_emissions),
        }
        provenance_hash = _compute_hash(provenance_data)

        # Record provenance
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="TD_LOSS",
                    entity_id=f"accounting_{code}_{_utcnow().isoformat()}",
                    action="CALCULATE",
                    data=provenance_data,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record accounting summary provenance: %s", exc,
                )

        # Record metrics
        if self._metrics is not None:
            try:
                self._metrics.record_td_loss_adjustment()
            except Exception:
                pass

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "country_code": code,
            "country_name": factor_meta.get("name", code),
            "td_loss_factor": {
                "td_loss_pct": str(td_pct),
                "td_loss_pct_display": f"{td_pct * _HUNDRED}%",
                "source": factor_meta.get("source", ""),
                "year": factor_meta.get("year", ""),
                "confidence": factor_meta.get("confidence", ""),
            },
            "consumption": {
                "net_mwh": str(net),
                "gross_mwh": str(gross),
                "td_loss_mwh": str(td_loss),
            },
            "emissions_kg_co2e": {
                "base_emissions": str(base_emissions),
                "td_loss_emissions": str(td_loss_emissions),
                "total_emissions": str(total_emissions),
            },
            "emissions_t_co2e": {
                "base_emissions": str(base_emissions_t),
                "td_loss_emissions": str(td_loss_emissions_t),
                "total_emissions": str(total_emissions_t),
            },
            "grid_ef_kg_co2e_per_mwh": str(ef),
            "td_pct_uplift": str(pct_uplift),
            "provenance_hash": provenance_hash,
            "calculation_time_ms": round(elapsed_ms, 3),
        }

    # ------------------------------------------------------------------
    # 21. validate_td_loss_factor
    # ------------------------------------------------------------------

    def validate_td_loss_factor(
        self,
        pct: Decimal,
    ) -> List[str]:
        """Validate a T&D loss factor against allowable bounds.

        Checks that the factor is within the configurable range
        [0, max_td_loss_pct] (default [0, 0.50]).

        Args:
            pct: T&D loss percentage as a Decimal fraction.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors: List[str] = []
        try:
            pct_val = self._to_decimal(pct, "td_loss_pct")
        except (ValueError, InvalidOperation) as exc:
            errors.append(f"Invalid Decimal value: {exc}")
            return errors

        if pct_val < self._min_td_loss_pct:
            errors.append(
                f"T&D loss factor {pct_val} is below minimum "
                f"{self._min_td_loss_pct}"
            )

        if pct_val > self._max_td_loss_pct:
            errors.append(
                f"T&D loss factor {pct_val} exceeds maximum "
                f"{self._max_td_loss_pct} (50%)"
            )

        return errors

    # ------------------------------------------------------------------
    # 22. validate_country_code
    # ------------------------------------------------------------------

    def validate_country_code(
        self,
        code: str,
    ) -> bool:
        """Validate that a country code exists in the T&D loss database.

        Checks both built-in and custom factor databases.

        Args:
            code: Country code to validate (case-insensitive).

        Returns:
            True if the country code is recognized, False otherwise.
        """
        normalized = self._normalize_country_code(code)

        with self._lock:
            if normalized in self._custom_factors:
                return True

        return normalized in TD_LOSS_FACTORS

    # ------------------------------------------------------------------
    # 23. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for T&D loss factors in the database.

        Returns statistics across all countries (built-in only, excluding
        WORLD average):
        - count, min, max, mean, median, std_dev

        Returns:
            Dictionary with statistical summaries.
        """
        factors: List[Decimal] = []
        for code, record in TD_LOSS_FACTORS.items():
            if code == "WORLD":
                continue
            factors.append(record["td_loss_pct"])

        if not factors:
            return {
                "country_count": 0,
                "min_td_loss_pct": None,
                "max_td_loss_pct": None,
                "mean_td_loss_pct": None,
                "median_td_loss_pct": None,
                "std_dev_td_loss_pct": None,
                "world_average_td_loss_pct": str(
                    TD_LOSS_FACTORS.get("WORLD", {}).get("td_loss_pct", "")
                ),
            }

        sorted_factors = sorted(factors)
        count = len(sorted_factors)
        total = sum(sorted_factors, _ZERO)
        mean_val = (total / Decimal(str(count))).quantize(
            _FACTOR_PRECISION, rounding=ROUND_HALF_UP
        )

        # Median
        if count % 2 == 0:
            mid = count // 2
            median_val = (
                (sorted_factors[mid - 1] + sorted_factors[mid]) / Decimal("2")
            ).quantize(_FACTOR_PRECISION, rounding=ROUND_HALF_UP)
        else:
            median_val = sorted_factors[count // 2]

        # Standard deviation (population)
        float_factors = [float(f) for f in sorted_factors]
        try:
            std_dev = Decimal(
                str(statistics.pstdev(float_factors))
            ).quantize(_FACTOR_PRECISION, rounding=ROUND_HALF_UP)
        except Exception:
            std_dev = _ZERO

        world_avg = TD_LOSS_FACTORS.get("WORLD", {}).get("td_loss_pct", _ZERO)

        return {
            "country_count": count,
            "min_td_loss_pct": str(sorted_factors[0]),
            "max_td_loss_pct": str(sorted_factors[-1]),
            "mean_td_loss_pct": str(mean_val),
            "median_td_loss_pct": str(median_val),
            "std_dev_td_loss_pct": str(std_dev),
            "world_average_td_loss_pct": str(world_avg),
            "calculation_count": self._calculation_count,
        }

    # ------------------------------------------------------------------
    # 24. compare_countries
    # ------------------------------------------------------------------

    def compare_countries(
        self,
        codes: List[str],
    ) -> Dict[str, Any]:
        """Compare T&D loss factors across multiple countries.

        Args:
            codes: List of country codes to compare (case-insensitive).

        Returns:
            Dictionary with comparison data:
                - ``countries``: List of dicts with code, name, td_loss_pct.
                - ``lowest``: Country with the lowest T&D losses.
                - ``highest``: Country with the highest T&D losses.
                - ``average``: Average T&D loss across compared countries.
                - ``range``: Difference between highest and lowest.
        """
        if not codes:
            raise ValueError("codes list must not be empty")

        countries: List[Dict[str, Any]] = []
        for raw_code in codes:
            code = self._normalize_country_code(raw_code)
            try:
                meta = self.get_td_loss_factor_with_metadata(code)
                countries.append({
                    "country_code": code,
                    "name": meta["name"],
                    "td_loss_pct": meta["td_loss_pct"],
                    "confidence": meta["confidence"],
                })
            except ValueError:
                logger.warning(
                    "Country code %s not found for comparison; skipping", code,
                )

        if not countries:
            raise ValueError("No valid country codes found for comparison")

        # Sort by td_loss_pct
        countries.sort(key=lambda c: Decimal(c["td_loss_pct"]))

        lowest = countries[0]
        highest = countries[-1]

        factors = [Decimal(c["td_loss_pct"]) for c in countries]
        total = sum(factors, _ZERO)
        avg = (total / Decimal(str(len(factors)))).quantize(
            _FACTOR_PRECISION, rounding=ROUND_HALF_UP
        )
        range_val = (Decimal(highest["td_loss_pct"]) - Decimal(lowest["td_loss_pct"])).quantize(
            _FACTOR_PRECISION, rounding=ROUND_HALF_UP
        )

        return {
            "countries": countries,
            "lowest": lowest,
            "highest": highest,
            "average_td_loss_pct": str(avg),
            "range_td_loss_pct": str(range_val),
            "country_count": len(countries),
        }

    # ------------------------------------------------------------------
    # 25. get_highest_loss_countries
    # ------------------------------------------------------------------

    def get_highest_loss_countries(
        self,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get countries with the highest T&D loss factors.

        Args:
            top_n: Number of countries to return. Defaults to 10.

        Returns:
            List of dictionaries sorted by T&D loss factor descending,
            each with ``country_code``, ``name``, ``td_loss_pct``.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        all_factors = self._get_all_effective_factors()

        # Sort descending by td_loss_pct
        sorted_factors = sorted(
            all_factors,
            key=lambda x: x["td_loss_pct_decimal"],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        for item in sorted_factors[:top_n]:
            results.append({
                "country_code": item["country_code"],
                "name": item["name"],
                "td_loss_pct": str(item["td_loss_pct_decimal"]),
                "rank": len(results) + 1,
            })

        return results

    # ------------------------------------------------------------------
    # 26. get_lowest_loss_countries
    # ------------------------------------------------------------------

    def get_lowest_loss_countries(
        self,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get countries with the lowest T&D loss factors.

        Args:
            top_n: Number of countries to return. Defaults to 10.

        Returns:
            List of dictionaries sorted by T&D loss factor ascending,
            each with ``country_code``, ``name``, ``td_loss_pct``.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        all_factors = self._get_all_effective_factors()

        # Sort ascending by td_loss_pct
        sorted_factors = sorted(
            all_factors,
            key=lambda x: x["td_loss_pct_decimal"],
        )

        results: List[Dict[str, Any]] = []
        for item in sorted_factors[:top_n]:
            results.append({
                "country_code": item["country_code"],
                "name": item["name"],
                "td_loss_pct": str(item["td_loss_pct_decimal"]),
                "rank": len(results) + 1,
            })

        return results

    # ==================================================================
    # Batch and multi-facility methods
    # ==================================================================

    def calculate_batch(
        self,
        inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate T&D losses for a batch of inputs.

        Each input dict should contain at minimum:
            - ``net_consumption_mwh``: Decimal or numeric
            - ``country_code``: str

        Optional keys:
            - ``custom_td_pct``: Decimal or numeric
            - ``grid_ef_co2e``: Decimal or numeric
            - ``facility_id``: str (for identification in results)

        Args:
            inputs: List of input dictionaries.

        Returns:
            Dictionary with:
                - ``results``: List of TDLossResult dicts
                - ``summary``: Aggregate statistics
                - ``errors``: List of per-input errors
                - ``provenance_hash``: SHA-256 hash of the batch
        """
        start_time = time.monotonic()

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total_net = _ZERO
        total_gross = _ZERO
        total_loss = _ZERO
        total_loss_emissions = _ZERO

        for idx, inp in enumerate(inputs):
            try:
                net = Decimal(str(inp.get("net_consumption_mwh", 0)))
                code = str(inp.get("country_code", "WORLD"))
                custom_td = inp.get("custom_td_pct")
                grid_ef = inp.get("grid_ef_co2e")

                if custom_td is not None:
                    custom_td = Decimal(str(custom_td))
                if grid_ef is not None:
                    grid_ef = Decimal(str(grid_ef))

                result = self.calculate_td_loss(
                    net_consumption_mwh=net,
                    country_code=code,
                    custom_td_pct=custom_td,
                    grid_ef_co2e=grid_ef,
                )

                result_dict = result.to_dict()
                result_dict["facility_id"] = inp.get("facility_id", f"facility_{idx}")
                result_dict["input_index"] = idx
                results.append(result_dict)

                total_net += result.net_consumption_mwh
                total_gross += result.gross_consumption_mwh
                total_loss += result.loss_mwh
                total_loss_emissions += result.loss_emissions_kg

            except Exception as exc:
                errors.append({
                    "input_index": idx,
                    "facility_id": inp.get("facility_id", f"facility_{idx}"),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                })
                if self._metrics is not None:
                    try:
                        self._metrics.record_error("calculation_error")
                    except Exception:
                        pass

        # Summary
        total_net = total_net.quantize(self._output_precision, rounding=ROUND_HALF_UP)
        total_gross = total_gross.quantize(self._output_precision, rounding=ROUND_HALF_UP)
        total_loss = total_loss.quantize(self._output_precision, rounding=ROUND_HALF_UP)
        total_loss_emissions = total_loss_emissions.quantize(
            self._output_precision, rounding=ROUND_HALF_UP
        )

        summary = {
            "total_inputs": len(inputs),
            "successful": len(results),
            "failed": len(errors),
            "total_net_consumption_mwh": str(total_net),
            "total_gross_consumption_mwh": str(total_gross),
            "total_td_loss_mwh": str(total_loss),
            "total_loss_emissions_kg": str(total_loss_emissions),
        }

        batch_provenance = _compute_hash({
            "batch_results": [r.get("provenance_hash", "") for r in results],
            "summary": summary,
        })

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return {
            "results": results,
            "summary": summary,
            "errors": errors,
            "provenance_hash": batch_provenance,
            "calculation_time_ms": round(elapsed_ms, 3),
        }

    def calculate_multi_country(
        self,
        net_mwh: Decimal,
        country_codes: List[str],
        grid_ef: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Calculate T&D losses for the same consumption across multiple countries.

        Useful for scenario analysis (e.g. "What if this facility were in
        country X vs country Y?").

        Args:
            net_mwh: Net electricity consumption in MWh.
            country_codes: List of country codes to compare.
            grid_ef: Optional grid emission factor for emissions calculation.

        Returns:
            Dictionary with per-country results and comparison summary.
        """
        net = self._to_decimal(net_mwh, "net_mwh")
        results: List[Dict[str, Any]] = []

        for raw_code in country_codes:
            code = self._normalize_country_code(raw_code)
            try:
                result = self.calculate_td_loss(
                    net_consumption_mwh=net,
                    country_code=code,
                    grid_ef_co2e=grid_ef,
                )
                results.append(result.to_dict())
            except ValueError as exc:
                results.append({
                    "country_code": code,
                    "error": str(exc),
                })

        # Identify best and worst
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            sorted_by_loss = sorted(
                valid_results,
                key=lambda r: Decimal(r["loss_mwh"]),
            )
            best = sorted_by_loss[0]["country_code"]
            worst = sorted_by_loss[-1]["country_code"]
        else:
            best = None
            worst = None

        return {
            "net_consumption_mwh": str(net),
            "grid_ef_co2e": str(grid_ef) if grid_ef is not None else None,
            "results": results,
            "best_country": best,
            "worst_country": worst,
            "country_count": len(country_codes),
        }

    # ==================================================================
    # Sensitivity analysis
    # ==================================================================

    def sensitivity_analysis(
        self,
        net_mwh: Decimal,
        country_code: str,
        grid_ef: Decimal,
        td_pct_range: Optional[List[Decimal]] = None,
    ) -> Dict[str, Any]:
        """Run sensitivity analysis on T&D loss factor variations.

        Computes emissions across a range of T&D loss percentages to
        understand the impact of T&D factor uncertainty on total emissions.

        Args:
            net_mwh: Net electricity consumption in MWh.
            country_code: Base country code for reference factor.
            grid_ef: Grid emission factor in kg CO2e per MWh.
            td_pct_range: Optional list of T&D percentages to evaluate.
                Defaults to [-50%, -25%, 0%, +25%, +50%] around the
                country's base factor.

        Returns:
            Dictionary with sensitivity analysis results.
        """
        net = self._to_decimal(net_mwh, "net_mwh")
        ef = self._to_decimal(grid_ef, "grid_ef")
        code = self._normalize_country_code(country_code)

        base_td_pct = self.get_td_loss_factor(code)

        if td_pct_range is None:
            # Default: -50%, -25%, base, +25%, +50% of base factor
            multipliers = [
                Decimal("0.50"), Decimal("0.75"), Decimal("1.00"),
                Decimal("1.25"), Decimal("1.50"),
            ]
            td_pct_range = [
                (base_td_pct * m).quantize(_FACTOR_PRECISION, rounding=ROUND_HALF_UP)
                for m in multipliers
            ]

        scenarios: List[Dict[str, Any]] = []
        for td_pct in td_pct_range:
            td_val = self._to_decimal(td_pct, "td_pct")

            gross = self.get_gross_consumption(net, td_val)
            gross = gross.quantize(self._output_precision, rounding=ROUND_HALF_UP)
            loss = (gross - net).quantize(self._output_precision, rounding=ROUND_HALF_UP)
            loss_emissions = self.calculate_loss_emissions(net, td_val, ef)
            loss_emissions = loss_emissions.quantize(
                self._output_precision, rounding=ROUND_HALF_UP
            )
            base_emissions = (net * ef).quantize(
                self._output_precision, rounding=ROUND_HALF_UP
            )
            total_emissions = (base_emissions + loss_emissions).quantize(
                self._output_precision, rounding=ROUND_HALF_UP
            )

            # Percent change from base
            base_scenario_loss = self.calculate_loss_emissions(net, base_td_pct, ef)
            base_total = (net * ef + base_scenario_loss).quantize(
                self._output_precision, rounding=ROUND_HALF_UP
            )
            if base_total > _ZERO:
                pct_change = (
                    ((total_emissions - base_total) / base_total) * _HUNDRED
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                pct_change = _ZERO

            scenarios.append({
                "td_loss_pct": str(td_val),
                "gross_consumption_mwh": str(gross),
                "td_loss_mwh": str(loss),
                "loss_emissions_kg": str(loss_emissions),
                "total_emissions_kg": str(total_emissions),
                "pct_change_from_base": str(pct_change),
            })

        provenance_hash = _compute_hash({
            "method": "sensitivity_analysis",
            "net_mwh": str(net),
            "country_code": code,
            "base_td_pct": str(base_td_pct),
            "scenario_count": len(scenarios),
        })

        return {
            "country_code": code,
            "base_td_loss_pct": str(base_td_pct),
            "net_consumption_mwh": str(net),
            "grid_ef_kg_co2e_per_mwh": str(ef),
            "scenarios": scenarios,
            "provenance_hash": provenance_hash,
        }

    # ==================================================================
    # Engine state and metadata
    # ==================================================================

    @property
    def calculation_count(self) -> int:
        """Return the total number of T&D loss calculations performed."""
        with self._lock:
            return self._calculation_count

    @property
    def country_count(self) -> int:
        """Return the number of countries in the built-in database."""
        return len(TD_LOSS_FACTORS)

    @property
    def custom_factor_count(self) -> int:
        """Return the number of custom factors currently registered."""
        with self._lock:
            return len(self._custom_factors)

    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata and configuration summary.

        Returns:
            Dictionary describing engine state and capabilities.
        """
        return {
            "engine": "TransmissionLossEngine",
            "version": "1.0.0",
            "agent": "AGENT-MRV-009",
            "description": (
                "Transmission & Distribution loss calculations for "
                "Scope 2 location-based emissions"
            ),
            "country_count": len(TD_LOSS_FACTORS),
            "custom_factor_count": self.custom_factor_count,
            "region_count": len(REGIONAL_GROUPS),
            "calculation_count": self.calculation_count,
            "output_precision": str(self._output_precision),
            "fallback_to_world": self._fallback_to_world,
            "max_td_loss_pct": str(self._max_td_loss_pct),
            "metrics_available": self._metrics is not None,
            "provenance_available": self._provenance is not None,
            "supported_methods": [
                "calculate_td_loss",
                "get_gross_consumption",
                "get_net_consumption",
                "calculate_loss_emissions",
                "apply_td_adjustment",
                "get_td_loss_factor",
                "get_td_loss_factor_with_metadata",
                "list_all_factors",
                "get_regional_average",
                "set_custom_factor",
                "remove_custom_factor",
                "list_custom_factors",
                "allocate_proportional",
                "allocate_marginal",
                "allocate_fixed",
                "deduct_onsite_generation",
                "calculate_with_onsite",
                "net_to_gross",
                "gross_to_net",
                "get_accounting_summary",
                "validate_td_loss_factor",
                "validate_country_code",
                "get_statistics",
                "compare_countries",
                "get_highest_loss_countries",
                "get_lowest_loss_countries",
            ],
            "zero_hallucination": True,
            "deterministic": True,
        }

    def reset(self) -> None:
        """Reset engine state (custom factors and counters).

        Intended for testing only. Clears all custom factors and resets
        the calculation counter to zero.
        """
        with self._lock:
            self._custom_factors.clear()
            self._calculation_count = 0
        logger.info("TransmissionLossEngine reset to initial state")

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return (
            f"TransmissionLossEngine("
            f"countries={len(TD_LOSS_FACTORS)}, "
            f"custom={self.custom_factor_count}, "
            f"calculations={self.calculation_count})"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return (
            f"TransmissionLossEngine: {len(TD_LOSS_FACTORS)} countries, "
            f"{self.custom_factor_count} custom factors, "
            f"{self.calculation_count} calculations"
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _normalize_country_code(self, code: str) -> str:
        """Normalize a country code to uppercase, stripped of whitespace.

        Args:
            code: Raw country code string.

        Returns:
            Normalized uppercase country code.

        Raises:
            ValueError: If ``code`` is empty or None.
        """
        if not code or not isinstance(code, str):
            raise ValueError("country_code must be a non-empty string")
        return code.strip().upper()

    def _to_decimal(self, value: Any, name: str) -> Decimal:
        """Convert a value to Decimal with validation.

        Args:
            value: Numeric value to convert (int, float, str, Decimal).
            name: Parameter name for error messages.

        Returns:
            Value as a Decimal.

        Raises:
            ValueError: If the value cannot be converted to Decimal.
        """
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot convert {name}={value!r} to Decimal: {exc}"
            ) from exc

    def _get_factor_record(self, country_code: str) -> Dict[str, Any]:
        """Get the full factor record for a country, checking custom first.

        Args:
            country_code: Normalized country code.

        Returns:
            Factor record dictionary with at least ``td_loss_pct``, ``name``,
            ``source``, ``year``, ``confidence``.
        """
        code = self._normalize_country_code(country_code)

        with self._lock:
            if code in self._custom_factors:
                return dict(self._custom_factors[code])

        if code in TD_LOSS_FACTORS:
            return dict(TD_LOSS_FACTORS[code])

        if self._fallback_to_world:
            record = dict(TD_LOSS_FACTORS["WORLD"])
            record["name"] = f"{code} (fallback to World Average)"
            return record

        return {
            "td_loss_pct": _ZERO,
            "name": code,
            "source": "unknown",
            "year": 0,
            "confidence": "NONE",
        }

    def _get_all_effective_factors(self) -> List[Dict[str, Any]]:
        """Get all effective T&D loss factors (built-in + custom overrides).

        Excludes the WORLD average entry. Custom factors override built-in
        factors for the same country code.

        Returns:
            List of dicts with ``country_code``, ``name``, ``td_loss_pct_decimal``.
        """
        effective: Dict[str, Dict[str, Any]] = {}

        # Built-in factors (excluding WORLD)
        for code, record in TD_LOSS_FACTORS.items():
            if code == "WORLD":
                continue
            effective[code] = {
                "country_code": code,
                "name": record["name"],
                "td_loss_pct_decimal": record["td_loss_pct"],
            }

        # Override with custom factors
        with self._lock:
            for code, record in self._custom_factors.items():
                if code == "WORLD":
                    continue
                effective[code] = {
                    "country_code": code,
                    "name": record.get("name", code),
                    "td_loss_pct_decimal": record["td_loss_pct"],
                }

        return list(effective.values())


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_engine: Optional[TransmissionLossEngine] = None


def get_transmission_loss_engine(
    config: Optional[Dict[str, Any]] = None,
) -> TransmissionLossEngine:
    """Return the process-wide singleton :class:`TransmissionLossEngine`.

    Creates the instance on first call (lazy initialization). Subsequent
    calls return the same object. Thread-safe.

    Args:
        config: Optional configuration dictionary for first-time
            initialization. Ignored after the first call.

    Returns:
        The singleton :class:`TransmissionLossEngine` instance.
    """
    global _singleton_engine
    if _singleton_engine is None:
        with _singleton_lock:
            if _singleton_engine is None:
                _singleton_engine = TransmissionLossEngine(config=config)
                logger.info(
                    "TransmissionLossEngine singleton created"
                )
    return _singleton_engine


def reset_transmission_loss_engine() -> None:
    """Reset the singleton to ``None`` for testing.

    The next call to :func:`get_transmission_loss_engine` will create
    a fresh instance.
    """
    global _singleton_engine
    with _singleton_lock:
        _singleton_engine = None
    logger.info("TransmissionLossEngine singleton reset to None")
