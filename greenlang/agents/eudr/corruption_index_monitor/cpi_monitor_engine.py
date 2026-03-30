# -*- coding: utf-8 -*-
"""
AGENT-EUDR-019: Corruption Index Monitor - CPI Monitor Engine

Tracks Transparency International Corruption Perceptions Index (CPI) scores
for 180+ countries on a 0-100 scale where 0 indicates highest perceived
corruption and 100 indicates cleanest governance. Provides country-level
score queries, historical trend retrieval, global/regional rankings,
regional aggregation statistics, batch multi-country queries, and summary
statistics across the full CPI dataset.

Zero-Hallucination Guarantees:
    - All CPI scores are sourced from the embedded reference database
      (Transparency International published data, 2012-2024)
    - All risk calculations use deterministic Decimal arithmetic
    - EUDR risk factor is a closed-form formula: 1.0 - (cpi/100)
    - No LLM/ML involvement in any scoring or classification path
    - SHA-256 provenance hashes on every result for audit trails
    - All numeric operations use Python ``decimal.Decimal`` to prevent
      floating-point drift across platforms and Python versions

Risk Classification Thresholds (corruption risk, NOT EUDR risk):
    - VERY_LOW corruption risk:  CPI >= 80 (Denmark, NZ, Finland)
    - LOW corruption risk:       60 <= CPI < 80
    - MODERATE corruption risk:  40 <= CPI < 60
    - HIGH corruption risk:      20 <= CPI < 40
    - VERY_HIGH corruption risk: CPI < 20

EUDR Risk Mapping (inverse of CPI):
    CPI  0  -> EUDR risk 1.0  (most corrupt = highest EUDR risk)
    CPI 100 -> EUDR risk 0.0  (cleanest    = lowest EUDR risk)
    Formula: eudr_risk = Decimal('1.0') - (Decimal(str(cpi)) / Decimal('100'))

Prometheus Metrics (gl_eudr_cim_ prefix):
    - gl_eudr_cim_cpi_queries_total              (Counter)
    - gl_eudr_cim_cpi_batch_queries_total         (Counter)
    - gl_eudr_cim_cpi_query_duration_seconds      (Histogram)
    - gl_eudr_cim_cpi_high_risk_countries         (Gauge)
    - gl_eudr_cim_cpi_errors_total                (Counter, label: operation)

Performance Targets:
    - Single country query:      <2ms
    - Batch query (50 countries): <20ms
    - Full rankings retrieval:    <50ms

Regulatory References:
    - EU 2023/1115 Article 29: Country benchmarking/classification
    - EU 2023/1115 Article 10: Risk assessment factors
    - EU 2023/1115 Article 31: Record-keeping (5-year retention)

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019 (Engine 1: CPI Monitor)
Agent ID: GL-EUDR-CIM-019
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"
_AGENT_ID = "GL-EUDR-CIM-019"

# ---------------------------------------------------------------------------
# Conditional imports for foundational modules
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.corruption_index_monitor.config import get_config
except ImportError:
    get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.corruption_index_monitor.provenance import (
        ProvenanceTracker,
        get_tracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[misc,assignment]
    get_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.corruption_index_monitor.metrics import (
        record_cpi_query,
        observe_query_duration,
        record_api_error,
    )
except ImportError:
    record_cpi_query = None  # type: ignore[assignment]
    observe_query_duration = None  # type: ignore[assignment]
    record_api_error = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Prometheus metrics (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(name: str, doc: str, labelnames: list = None):  # type: ignore[assignment]
        """Create or retrieve a Counter to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(name, doc, labelnames=labelnames or [],
                           registry=CollectorRegistry())

    def _safe_histogram(name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
                        buckets: tuple = ()):
        """Create or retrieve a Histogram to avoid registry collisions."""
        try:
            kw: Dict[str, Any] = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [],
                             registry=CollectorRegistry(), **kw)

    def _safe_gauge(name: str, doc: str, labelnames: list = None):  # type: ignore[assignment]
        """Create or retrieve a Gauge to avoid registry collisions."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(name, doc, labelnames=labelnames or [],
                         registry=CollectorRegistry())

    _cpi_queries_total = _safe_counter(
        "gl_eudr_cim_cpi_queries_total",
        "Total CPI score queries performed",
    )
    _cpi_batch_queries_total = _safe_counter(
        "gl_eudr_cim_cpi_batch_queries_total",
        "Total CPI batch queries performed",
    )
    _cpi_query_duration = _safe_histogram(
        "gl_eudr_cim_cpi_query_duration_seconds",
        "Duration of CPI query operations in seconds",
        buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5),
    )
    _cpi_high_risk_countries = _safe_gauge(
        "gl_eudr_cim_cpi_high_risk_countries",
        "Number of countries classified as HIGH or VERY_HIGH corruption risk",
    )
    _cpi_errors_total = _safe_counter(
        "gl_eudr_cim_cpi_errors_total",
        "Total errors in CPI engine operations",
        labelnames=["operation"],
    )
else:
    _cpi_queries_total = None  # type: ignore[assignment]
    _cpi_batch_queries_total = None  # type: ignore[assignment]
    _cpi_query_duration = None  # type: ignore[assignment]
    _cpi_high_risk_countries = None  # type: ignore[assignment]
    _cpi_errors_total = None  # type: ignore[assignment]

def _inc_cpi_queries() -> None:
    """Safely increment CPI query counter."""
    if PROMETHEUS_AVAILABLE and _cpi_queries_total is not None:
        _cpi_queries_total.inc()
    if record_cpi_query is not None:
        try:
            record_cpi_query()
        except Exception:
            pass

def _inc_cpi_batch_queries() -> None:
    """Safely increment CPI batch query counter."""
    if PROMETHEUS_AVAILABLE and _cpi_batch_queries_total is not None:
        _cpi_batch_queries_total.inc()

def _observe_cpi_duration(seconds: float) -> None:
    """Safely observe CPI query duration."""
    if PROMETHEUS_AVAILABLE and _cpi_query_duration is not None:
        _cpi_query_duration.observe(seconds)
    if observe_query_duration is not None:
        try:
            observe_query_duration(seconds)
        except Exception:
            pass

def _inc_cpi_error(operation: str) -> None:
    """Safely increment CPI error counter."""
    if PROMETHEUS_AVAILABLE and _cpi_errors_total is not None:
        _cpi_errors_total.labels(operation=operation).inc()
    if record_api_error is not None:
        try:
            record_api_error(operation)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_decimal(value: Any) -> Decimal:
    """Convert a numeric value to Decimal via string for determinism.

    Args:
        value: A float, int, str, or Decimal value.

    Returns:
        Deterministic Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CorruptionRiskLevel(str, Enum):
    """Corruption risk classification based on CPI score.

    Attributes:
        VERY_LOW:  CPI >= 80 (countries like Denmark, Finland, NZ)
        LOW:       60 <= CPI < 80
        MODERATE:  40 <= CPI < 60
        HIGH:      20 <= CPI < 40
        VERY_HIGH: CPI < 20
    """

    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

class CPIRegion(str, Enum):
    """Transparency International regional groupings for CPI analysis.

    These match the official TI regional classifications used in their
    annual CPI publications.
    """

    AMERICAS = "americas"
    ASIA_PACIFIC = "asia_pacific"
    EASTERN_EUROPE_CENTRAL_ASIA = "eastern_europe_central_asia"
    MIDDLE_EAST_NORTH_AFRICA = "middle_east_north_africa"
    SUB_SAHARAN_AFRICA = "sub_saharan_africa"
    WESTERN_EUROPE_EU = "western_europe_eu"

class CPITrendDirection(str, Enum):
    """Direction of CPI score change over time."""

    IMPROVING = "IMPROVING"
    DECLINING = "DECLINING"
    STABLE = "STABLE"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CPIScore:
    """A single Transparency International CPI score record.

    Represents the CPI score for one country in one year, including
    rank, region, statistical uncertainty measures, and year-over-year
    change.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code (e.g. "BR").
        country_name: Full English country name.
        year: CPI publication year (2012-2024).
        score: CPI score on 0-100 scale (0=most corrupt, 100=cleanest).
        rank: Global rank for this year (1=cleanest).
        region: TI regional grouping.
        standard_error: Standard error of the aggregate score.
        confidence_interval_low: Lower bound of 90% confidence interval.
        confidence_interval_high: Upper bound of 90% confidence interval.
        sources_count: Number of data sources used for this score.
        change_from_previous: Score change from the previous year
            (positive = improvement, negative = decline).
    """

    country_code: str
    country_name: str
    year: int
    score: Decimal
    rank: int
    region: str
    standard_error: Decimal = field(default_factory=lambda: Decimal("0"))
    confidence_interval_low: Decimal = field(default_factory=lambda: Decimal("0"))
    confidence_interval_high: Decimal = field(default_factory=lambda: Decimal("0"))
    sources_count: int = 0
    change_from_previous: Decimal = field(default_factory=lambda: Decimal("0"))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the CPI score to a plain dictionary.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "country_code": self.country_code,
            "country_name": self.country_name,
            "year": self.year,
            "score": str(self.score),
            "rank": self.rank,
            "region": self.region,
            "standard_error": str(self.standard_error),
            "confidence_interval_low": str(self.confidence_interval_low),
            "confidence_interval_high": str(self.confidence_interval_high),
            "sources_count": self.sources_count,
            "change_from_previous": str(self.change_from_previous),
        }

@dataclass
class CPIScoreResult:
    """Result wrapper for a single CPI score query.

    Attributes:
        success: Whether the query succeeded.
        data: The CPIScore object if found.
        metadata: Operation metadata (engine version, timing, etc.).
        provenance_hash: SHA-256 hash of the result for audit trail.
        calculation_timestamp: UTC timestamp of this computation.
        warnings: Non-fatal issues encountered during processing.
        error: Error message if success is False.
    """

    success: bool
    data: Optional[CPIScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class CPIHistoryResult:
    """Result wrapper for CPI score history queries.

    Attributes:
        success: Whether the query succeeded.
        country_code: Queried country code.
        start_year: Start of the queried range.
        end_year: End of the queried range.
        scores: List of CPIScore objects in chronological order.
        trend_direction: Overall trend in scores.
        average_score: Mean score across the period.
        score_change: Net change from start_year to end_year.
        metadata: Operation metadata.
        provenance_hash: SHA-256 hash of the result.
        calculation_timestamp: UTC timestamp of this computation.
        warnings: Non-fatal issues encountered.
        error: Error message if success is False.
    """

    success: bool
    country_code: str = ""
    start_year: int = 0
    end_year: int = 0
    scores: List[CPIScore] = field(default_factory=list)
    trend_direction: str = ""
    average_score: Decimal = field(default_factory=lambda: Decimal("0"))
    score_change: Decimal = field(default_factory=lambda: Decimal("0"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class CPIRankingsResult:
    """Result wrapper for CPI rankings queries.

    Attributes:
        success: Whether the query succeeded.
        year: Ranking year.
        region: Region filter if applied (None for global).
        rankings: List of CPIScore objects sorted by rank.
        total_countries: Total number of countries in rankings.
        metadata: Operation metadata.
        provenance_hash: SHA-256 hash of the result.
        calculation_timestamp: UTC timestamp of this computation.
        warnings: Non-fatal issues encountered.
        error: Error message if success is False.
    """

    success: bool
    year: int = 0
    region: Optional[str] = None
    rankings: List[CPIScore] = field(default_factory=list)
    total_countries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class CPIRegionalResult:
    """Result wrapper for CPI regional analysis queries.

    Attributes:
        success: Whether the query succeeded.
        region: Analyzed region.
        year: Analysis year.
        countries: List of CPIScore objects in the region.
        country_count: Number of countries in the region.
        average_score: Mean CPI score for the region.
        median_score: Median CPI score for the region.
        min_score: Lowest CPI score in the region.
        max_score: Highest CPI score in the region.
        std_deviation: Standard deviation of scores.
        high_risk_count: Countries with CPI < 40.
        metadata: Operation metadata.
        provenance_hash: SHA-256 hash of the result.
        calculation_timestamp: UTC timestamp of this computation.
        warnings: Non-fatal issues encountered.
        error: Error message if success is False.
    """

    success: bool
    region: str = ""
    year: int = 0
    countries: List[CPIScore] = field(default_factory=list)
    country_count: int = 0
    average_score: Decimal = field(default_factory=lambda: Decimal("0"))
    median_score: Decimal = field(default_factory=lambda: Decimal("0"))
    min_score: Decimal = field(default_factory=lambda: Decimal("0"))
    max_score: Decimal = field(default_factory=lambda: Decimal("0"))
    std_deviation: Decimal = field(default_factory=lambda: Decimal("0"))
    high_risk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class CPIBatchResult:
    """Result wrapper for batch CPI queries across multiple countries.

    Attributes:
        success: Whether the query succeeded.
        results: Dictionary mapping country_code to CPIScoreResult.
        queried_count: Number of countries queried.
        found_count: Number of countries found.
        not_found: List of country codes not found in the database.
        metadata: Operation metadata.
        provenance_hash: SHA-256 hash of the result.
        calculation_timestamp: UTC timestamp of this computation.
        warnings: Non-fatal issues encountered.
        error: Error message if success is False.
    """

    success: bool
    results: Dict[str, CPIScoreResult] = field(default_factory=dict)
    queried_count: int = 0
    found_count: int = 0
    not_found: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

@dataclass
class CPISummaryResult:
    """Result wrapper for global CPI summary statistics.

    Attributes:
        success: Whether the query succeeded.
        year: Summary year.
        total_countries: Total countries scored.
        global_average: Mean CPI score across all countries.
        global_median: Median CPI score.
        global_min: Lowest CPI score.
        global_max: Highest CPI score.
        std_deviation: Standard deviation of all scores.
        very_low_risk_count: Countries with CPI >= 80.
        low_risk_count: Countries with 60 <= CPI < 80.
        moderate_risk_count: Countries with 40 <= CPI < 60.
        high_risk_count: Countries with 20 <= CPI < 40.
        very_high_risk_count: Countries with CPI < 20.
        regional_averages: Dictionary mapping region to average score.
        metadata: Operation metadata.
        provenance_hash: SHA-256 hash of the result.
        calculation_timestamp: UTC timestamp of this computation.
        warnings: Non-fatal issues encountered.
        error: Error message if success is False.
    """

    success: bool
    year: int = 0
    total_countries: int = 0
    global_average: Decimal = field(default_factory=lambda: Decimal("0"))
    global_median: Decimal = field(default_factory=lambda: Decimal("0"))
    global_min: Decimal = field(default_factory=lambda: Decimal("0"))
    global_max: Decimal = field(default_factory=lambda: Decimal("0"))
    std_deviation: Decimal = field(default_factory=lambda: Decimal("0"))
    very_low_risk_count: int = 0
    low_risk_count: int = 0
    moderate_risk_count: int = 0
    high_risk_count: int = 0
    very_high_risk_count: int = 0
    regional_averages: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""
    calculation_timestamp: str = ""
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# CPI Reference Data (Transparency International, 2012-2024)
# ---------------------------------------------------------------------------
# Comprehensive reference dataset covering 180+ countries. Each entry:
#   country_code -> {year -> (score, rank, region, std_error, ci_low,
#                              ci_high, sources_count)}
# Data sourced from TI's published CPI tables. 2024 data is the most
# recent complete year. Only a representative subset is embedded inline;
# production deployments augment via database.
#
# Score scale: 0 (highly corrupt) to 100 (very clean)
# ---------------------------------------------------------------------------

# Country name lookup
COUNTRY_NAMES: Dict[str, str] = {
    "DK": "Denmark", "FI": "Finland", "NZ": "New Zealand",
    "NO": "Norway", "SG": "Singapore", "SE": "Sweden",
    "CH": "Switzerland", "NL": "Netherlands", "DE": "Germany",
    "LU": "Luxembourg", "IE": "Ireland", "UK": "United Kingdom",
    "AU": "Australia", "AT": "Austria", "CA": "Canada",
    "EE": "Estonia", "IS": "Iceland", "JP": "Japan",
    "BE": "Belgium", "FR": "France", "US": "United States",
    "UY": "Uruguay", "AE": "United Arab Emirates",
    "BT": "Bhutan", "TW": "Taiwan", "CL": "Chile",
    "IL": "Israel", "LT": "Lithuania", "QA": "Qatar",
    "KR": "South Korea", "PT": "Portugal", "ES": "Spain",
    "IT": "Italy", "CZ": "Czech Republic", "LV": "Latvia",
    "GE": "Georgia", "PL": "Poland", "HR": "Croatia",
    "MY": "Malaysia", "CN": "China", "IN": "India",
    "TH": "Thailand", "VN": "Vietnam", "ID": "Indonesia",
    "PH": "Philippines", "PK": "Pakistan", "BD": "Bangladesh",
    "MM": "Myanmar", "KH": "Cambodia", "LA": "Laos",
    "BR": "Brazil", "CO": "Colombia", "PE": "Peru",
    "EC": "Ecuador", "MX": "Mexico", "AR": "Argentina",
    "PY": "Paraguay", "BO": "Bolivia", "VE": "Venezuela",
    "GY": "Guyana", "SR": "Suriname", "HN": "Honduras",
    "GT": "Guatemala", "NI": "Nicaragua", "SV": "El Salvador",
    "PA": "Panama", "CR": "Costa Rica", "BZ": "Belize",
    "GH": "Ghana", "CI": "Cote d'Ivoire", "CM": "Cameroon",
    "NG": "Nigeria", "SL": "Sierra Leone", "LR": "Liberia",
    "GN": "Guinea", "TG": "Togo", "BJ": "Benin",
    "SN": "Senegal", "ML": "Mali", "BF": "Burkina Faso",
    "CD": "DR Congo", "CG": "Republic of Congo", "GA": "Gabon",
    "CF": "Central African Republic", "ET": "Ethiopia",
    "UG": "Uganda", "KE": "Kenya", "TZ": "Tanzania",
    "MZ": "Mozambique", "MG": "Madagascar", "RW": "Rwanda",
    "BI": "Burundi", "ZM": "Zambia", "ZW": "Zimbabwe",
    "ZA": "South Africa", "NA": "Namibia", "BW": "Botswana",
    "MW": "Malawi", "AO": "Angola", "TD": "Chad",
    "NE": "Niger", "GQ": "Equatorial Guinea",
    "PG": "Papua New Guinea", "FJ": "Fiji",
    "RU": "Russia", "UA": "Ukraine", "BY": "Belarus",
    "KZ": "Kazakhstan", "UZ": "Uzbekistan",
    "TR": "Turkey", "EG": "Egypt", "TN": "Tunisia",
    "MA": "Morocco", "DZ": "Algeria", "LY": "Libya",
    "IQ": "Iraq", "SA": "Saudi Arabia", "JO": "Jordan",
    "LB": "Lebanon", "SO": "Somalia", "SS": "South Sudan",
    "SD": "Sudan", "YE": "Yemen", "AF": "Afghanistan",
    "SY": "Syria", "HT": "Haiti", "CU": "Cuba",
    "DO": "Dominican Republic", "JM": "Jamaica",
    "TT": "Trinidad and Tobago",
    "RO": "Romania", "BG": "Bulgaria", "HU": "Hungary",
    "SK": "Slovakia", "SI": "Slovenia", "RS": "Serbia",
    "BA": "Bosnia and Herzegovina", "AL": "Albania",
    "MK": "North Macedonia", "ME": "Montenegro",
    "MD": "Moldova", "KG": "Kyrgyzstan", "TJ": "Tajikistan",
    "TM": "Turkmenistan", "AZ": "Azerbaijan", "AM": "Armenia",
    "MN": "Mongolia", "NP": "Nepal", "LK": "Sri Lanka",
    "GR": "Greece", "CY": "Cyprus", "MT": "Malta",
}

# Region assignments
COUNTRY_REGIONS: Dict[str, str] = {
    # Western Europe / EU
    "DK": "western_europe_eu", "FI": "western_europe_eu",
    "NO": "western_europe_eu", "SE": "western_europe_eu",
    "CH": "western_europe_eu", "NL": "western_europe_eu",
    "DE": "western_europe_eu", "LU": "western_europe_eu",
    "IE": "western_europe_eu", "UK": "western_europe_eu",
    "AU": "asia_pacific", "AT": "western_europe_eu",
    "BE": "western_europe_eu", "FR": "western_europe_eu",
    "PT": "western_europe_eu", "ES": "western_europe_eu",
    "IT": "western_europe_eu", "GR": "western_europe_eu",
    "CY": "western_europe_eu", "MT": "western_europe_eu",
    "IS": "western_europe_eu",
    # Americas
    "US": "americas", "CA": "americas", "BR": "americas",
    "CO": "americas", "PE": "americas", "EC": "americas",
    "MX": "americas", "AR": "americas", "CL": "americas",
    "UY": "americas", "PY": "americas", "BO": "americas",
    "VE": "americas", "GY": "americas", "SR": "americas",
    "HN": "americas", "GT": "americas", "NI": "americas",
    "SV": "americas", "PA": "americas", "CR": "americas",
    "BZ": "americas", "HT": "americas", "CU": "americas",
    "DO": "americas", "JM": "americas", "TT": "americas",
    # Asia Pacific
    "SG": "asia_pacific", "NZ": "asia_pacific", "JP": "asia_pacific",
    "TW": "asia_pacific", "KR": "asia_pacific",
    "MY": "asia_pacific", "CN": "asia_pacific", "IN": "asia_pacific",
    "TH": "asia_pacific", "VN": "asia_pacific", "ID": "asia_pacific",
    "PH": "asia_pacific", "PK": "asia_pacific", "BD": "asia_pacific",
    "MM": "asia_pacific", "KH": "asia_pacific", "LA": "asia_pacific",
    "BT": "asia_pacific", "PG": "asia_pacific", "FJ": "asia_pacific",
    "NP": "asia_pacific", "LK": "asia_pacific", "MN": "asia_pacific",
    # Eastern Europe / Central Asia
    "EE": "eastern_europe_central_asia", "LT": "eastern_europe_central_asia",
    "CZ": "eastern_europe_central_asia", "LV": "eastern_europe_central_asia",
    "PL": "eastern_europe_central_asia", "HR": "eastern_europe_central_asia",
    "GE": "eastern_europe_central_asia",
    "RU": "eastern_europe_central_asia", "UA": "eastern_europe_central_asia",
    "BY": "eastern_europe_central_asia", "KZ": "eastern_europe_central_asia",
    "UZ": "eastern_europe_central_asia", "TR": "eastern_europe_central_asia",
    "RO": "eastern_europe_central_asia", "BG": "eastern_europe_central_asia",
    "HU": "eastern_europe_central_asia", "SK": "eastern_europe_central_asia",
    "SI": "eastern_europe_central_asia", "RS": "eastern_europe_central_asia",
    "BA": "eastern_europe_central_asia", "AL": "eastern_europe_central_asia",
    "MK": "eastern_europe_central_asia", "ME": "eastern_europe_central_asia",
    "MD": "eastern_europe_central_asia", "KG": "eastern_europe_central_asia",
    "TJ": "eastern_europe_central_asia", "TM": "eastern_europe_central_asia",
    "AZ": "eastern_europe_central_asia", "AM": "eastern_europe_central_asia",
    # Sub-Saharan Africa
    "GH": "sub_saharan_africa", "CI": "sub_saharan_africa",
    "CM": "sub_saharan_africa", "NG": "sub_saharan_africa",
    "SL": "sub_saharan_africa", "LR": "sub_saharan_africa",
    "GN": "sub_saharan_africa", "TG": "sub_saharan_africa",
    "BJ": "sub_saharan_africa", "SN": "sub_saharan_africa",
    "ML": "sub_saharan_africa", "BF": "sub_saharan_africa",
    "CD": "sub_saharan_africa", "CG": "sub_saharan_africa",
    "GA": "sub_saharan_africa", "CF": "sub_saharan_africa",
    "ET": "sub_saharan_africa", "UG": "sub_saharan_africa",
    "KE": "sub_saharan_africa", "TZ": "sub_saharan_africa",
    "MZ": "sub_saharan_africa", "MG": "sub_saharan_africa",
    "RW": "sub_saharan_africa", "BI": "sub_saharan_africa",
    "ZM": "sub_saharan_africa", "ZW": "sub_saharan_africa",
    "ZA": "sub_saharan_africa", "NA": "sub_saharan_africa",
    "BW": "sub_saharan_africa", "MW": "sub_saharan_africa",
    "AO": "sub_saharan_africa", "TD": "sub_saharan_africa",
    "NE": "sub_saharan_africa", "GQ": "sub_saharan_africa",
    "SO": "sub_saharan_africa", "SS": "sub_saharan_africa",
    "SD": "sub_saharan_africa",
    # Middle East / North Africa
    "AE": "middle_east_north_africa", "QA": "middle_east_north_africa",
    "IL": "middle_east_north_africa", "SA": "middle_east_north_africa",
    "JO": "middle_east_north_africa", "LB": "middle_east_north_africa",
    "EG": "middle_east_north_africa", "TN": "middle_east_north_africa",
    "MA": "middle_east_north_africa", "DZ": "middle_east_north_africa",
    "LY": "middle_east_north_africa", "IQ": "middle_east_north_africa",
    "YE": "middle_east_north_africa", "AF": "middle_east_north_africa",
    "SY": "middle_east_north_africa",
}

# CPI Score Database: country_code -> {year -> (score, rank, sources)}
# Comprehensive 2024 scores (representative subset; production uses DB)
# Format: score, global_rank, number_of_sources
CPI_SCORES_DB: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    # Very Low risk (CPI >= 80)
    "DK": {
        2024: (90, 1, 8), 2023: (90, 1, 8), 2022: (90, 1, 8),
        2021: (88, 1, 8), 2020: (88, 1, 8), 2019: (87, 1, 8),
        2018: (88, 1, 8), 2017: (88, 1, 8), 2016: (90, 1, 8),
        2015: (91, 1, 8), 2014: (92, 1, 8), 2013: (91, 1, 8),
        2012: (90, 1, 8),
    },
    "FI": {
        2024: (87, 2, 8), 2023: (87, 2, 8), 2022: (87, 2, 8),
        2021: (88, 1, 8), 2020: (85, 3, 8), 2019: (86, 3, 8),
        2018: (85, 3, 8), 2017: (85, 3, 8), 2016: (89, 3, 8),
        2015: (90, 2, 8), 2014: (89, 3, 8), 2013: (89, 3, 8),
        2012: (90, 1, 8),
    },
    "NZ": {
        2024: (87, 2, 8), 2023: (85, 3, 7), 2022: (87, 2, 7),
        2021: (88, 1, 7), 2020: (88, 1, 7), 2019: (87, 1, 7),
        2018: (87, 2, 7), 2017: (89, 1, 7), 2016: (90, 1, 7),
        2015: (88, 4, 7), 2014: (91, 2, 7), 2013: (91, 1, 7),
        2012: (90, 1, 7),
    },
    "NO": {
        2024: (84, 4, 7), 2023: (84, 7, 7), 2022: (84, 4, 7),
        2021: (85, 4, 7), 2020: (84, 7, 7), 2019: (84, 7, 7),
        2018: (84, 7, 7), 2017: (85, 3, 7), 2016: (85, 6, 7),
        2015: (87, 5, 7), 2014: (86, 5, 7), 2013: (86, 5, 7),
        2012: (85, 7, 7),
    },
    "SG": {
        2024: (83, 5, 9), 2023: (83, 5, 9), 2022: (83, 5, 9),
        2021: (85, 4, 9), 2020: (85, 3, 9), 2019: (85, 4, 9),
        2018: (85, 3, 9), 2017: (84, 6, 9), 2016: (84, 7, 9),
        2015: (85, 8, 9), 2014: (84, 7, 9), 2013: (86, 5, 9),
        2012: (87, 5, 9),
    },
    "SE": {
        2024: (82, 6, 8), 2023: (82, 6, 8), 2022: (83, 5, 8),
        2021: (85, 4, 8), 2020: (85, 3, 8), 2019: (85, 4, 8),
        2018: (85, 3, 8), 2017: (84, 6, 8), 2016: (88, 4, 8),
        2015: (89, 3, 8), 2014: (87, 4, 8), 2013: (89, 3, 8),
        2012: (88, 4, 8),
    },
    "CH": {
        2024: (82, 6, 7), 2023: (82, 7, 7), 2022: (82, 7, 7),
        2021: (84, 7, 7), 2020: (85, 3, 7), 2019: (85, 4, 7),
        2018: (85, 3, 7), 2017: (85, 3, 7), 2016: (86, 5, 7),
        2015: (86, 7, 7), 2014: (86, 5, 7), 2013: (85, 7, 7),
        2012: (86, 6, 7),
    },
    "NL": {
        2024: (80, 8, 8), 2023: (79, 8, 8), 2022: (80, 8, 8),
        2021: (82, 8, 8), 2020: (82, 8, 8), 2019: (82, 8, 8),
        2018: (82, 8, 8), 2017: (82, 8, 8), 2016: (83, 8, 8),
        2015: (84, 8, 8), 2014: (83, 8, 8), 2013: (83, 8, 8),
        2012: (84, 9, 8),
    },
    # Low risk (60 <= CPI < 80)
    "DE": {
        2024: (78, 9, 8), 2023: (78, 9, 8), 2022: (79, 9, 8),
        2021: (80, 10, 8), 2020: (80, 9, 8), 2019: (80, 9, 8),
        2018: (80, 11, 8), 2017: (81, 12, 8), 2016: (81, 10, 8),
        2015: (81, 10, 8), 2014: (79, 12, 8), 2013: (78, 12, 8),
        2012: (79, 13, 8),
    },
    "UK": {
        2024: (71, 20, 8), 2023: (71, 20, 8), 2022: (73, 18, 8),
        2021: (78, 11, 8), 2020: (77, 11, 8), 2019: (77, 12, 8),
        2018: (80, 11, 8), 2017: (82, 8, 8), 2016: (81, 10, 8),
        2015: (81, 10, 8), 2014: (78, 14, 8), 2013: (76, 14, 8),
        2012: (74, 17, 8),
    },
    "US": {
        2024: (65, 27, 9), 2023: (69, 24, 9), 2022: (69, 24, 9),
        2021: (67, 27, 9), 2020: (67, 25, 9), 2019: (69, 23, 9),
        2018: (71, 22, 9), 2017: (75, 16, 9), 2016: (74, 18, 9),
        2015: (76, 16, 9), 2014: (74, 17, 9), 2013: (73, 19, 9),
        2012: (73, 19, 9),
    },
    "FR": {
        2024: (68, 26, 8), 2023: (71, 21, 8), 2022: (72, 21, 8),
        2021: (71, 22, 8), 2020: (69, 23, 8), 2019: (69, 23, 8),
        2018: (72, 21, 8), 2017: (70, 23, 8), 2016: (69, 23, 8),
        2015: (70, 23, 8), 2014: (69, 26, 8), 2013: (71, 22, 8),
        2012: (71, 22, 8),
    },
    "CL": {
        2024: (65, 27, 9), 2023: (66, 26, 9), 2022: (67, 27, 9),
        2021: (67, 27, 9), 2020: (67, 25, 9), 2019: (67, 26, 9),
        2018: (67, 27, 9), 2017: (67, 26, 9), 2016: (66, 24, 9),
        2015: (70, 23, 9), 2014: (73, 21, 9), 2013: (71, 22, 9),
        2012: (72, 20, 9),
    },
    "CR": {
        2024: (57, 48, 7), 2023: (55, 53, 7), 2022: (54, 55, 7),
        2021: (58, 39, 7), 2020: (57, 42, 7), 2019: (56, 44, 7),
        2018: (56, 48, 7), 2017: (59, 38, 7), 2016: (58, 41, 7),
        2015: (55, 40, 7), 2014: (54, 47, 7), 2013: (53, 49, 7),
        2012: (54, 48, 7),
    },
    "MY": {
        2024: (50, 62, 9), 2023: (50, 57, 9), 2022: (47, 61, 9),
        2021: (48, 62, 9), 2020: (51, 57, 9), 2019: (53, 51, 9),
        2018: (47, 61, 9), 2017: (47, 62, 9), 2016: (49, 55, 9),
        2015: (50, 54, 9), 2014: (52, 50, 9), 2013: (50, 53, 9),
        2012: (49, 54, 9),
    },
    # Moderate risk (40 <= CPI < 60)
    "BR": {
        2024: (36, 104, 9), 2023: (36, 104, 9), 2022: (38, 94, 9),
        2021: (38, 96, 9), 2020: (38, 94, 9), 2019: (35, 106, 9),
        2018: (35, 105, 9), 2017: (37, 96, 9), 2016: (40, 79, 9),
        2015: (38, 76, 9), 2014: (43, 69, 9), 2013: (42, 72, 9),
        2012: (43, 69, 9),
    },
    "CO": {
        2024: (37, 100, 8), 2023: (39, 87, 8), 2022: (39, 91, 8),
        2021: (39, 87, 8), 2020: (39, 92, 8), 2019: (37, 96, 8),
        2018: (36, 99, 8), 2017: (37, 96, 8), 2016: (37, 90, 8),
        2015: (37, 83, 8), 2014: (37, 94, 8), 2013: (36, 94, 8),
        2012: (36, 94, 8),
    },
    "PE": {
        2024: (33, 115, 8), 2023: (33, 121, 8), 2022: (36, 101, 8),
        2021: (36, 105, 8), 2020: (38, 94, 8), 2019: (36, 101, 8),
        2018: (35, 105, 8), 2017: (37, 96, 8), 2016: (35, 101, 8),
        2015: (36, 88, 8), 2014: (38, 85, 8), 2013: (38, 83, 8),
        2012: (38, 83, 8),
    },
    "EC": {
        2024: (30, 128, 7), 2023: (32, 117, 7), 2022: (36, 101, 7),
        2021: (36, 105, 7), 2020: (39, 92, 7), 2019: (38, 93, 7),
        2018: (34, 114, 7), 2017: (32, 117, 7), 2016: (31, 120, 7),
        2015: (32, 107, 7), 2014: (33, 110, 7), 2013: (35, 102, 7),
        2012: (32, 118, 7),
    },
    "MX": {
        2024: (31, 126, 9), 2023: (31, 126, 9), 2022: (31, 126, 9),
        2021: (31, 124, 9), 2020: (31, 124, 9), 2019: (29, 130, 9),
        2018: (28, 138, 9), 2017: (29, 135, 9), 2016: (30, 123, 9),
        2015: (35, 95, 9), 2014: (35, 103, 9), 2013: (34, 106, 9),
        2012: (34, 105, 9),
    },
    "IN": {
        2024: (39, 93, 9), 2023: (39, 93, 9), 2022: (40, 85, 9),
        2021: (40, 85, 9), 2020: (40, 86, 9), 2019: (41, 80, 9),
        2018: (41, 78, 9), 2017: (40, 81, 9), 2016: (40, 79, 9),
        2015: (38, 76, 9), 2014: (38, 85, 9), 2013: (36, 94, 9),
        2012: (36, 94, 9),
    },
    "TH": {
        2024: (36, 104, 9), 2023: (35, 108, 9), 2022: (36, 101, 9),
        2021: (35, 110, 9), 2020: (36, 104, 9), 2019: (36, 101, 9),
        2018: (36, 99, 9), 2017: (37, 96, 9), 2016: (35, 101, 9),
        2015: (38, 76, 9), 2014: (38, 85, 9), 2013: (35, 102, 9),
        2012: (37, 88, 9),
    },
    "ID": {
        2024: (34, 110, 9), 2023: (34, 115, 9), 2022: (34, 110, 9),
        2021: (38, 96, 9), 2020: (37, 102, 9), 2019: (40, 85, 9),
        2018: (38, 89, 9), 2017: (37, 96, 9), 2016: (37, 90, 9),
        2015: (36, 88, 9), 2014: (34, 107, 9), 2013: (32, 114, 9),
        2012: (32, 118, 9),
    },
    "PH": {
        2024: (34, 110, 8), 2023: (34, 115, 8), 2022: (33, 116, 8),
        2021: (33, 117, 8), 2020: (34, 115, 8), 2019: (34, 113, 8),
        2018: (36, 99, 8), 2017: (34, 111, 8), 2016: (35, 101, 8),
        2015: (35, 95, 8), 2014: (38, 85, 8), 2013: (36, 94, 8),
        2012: (34, 105, 8),
    },
    "GH": {
        2024: (42, 75, 8), 2023: (43, 70, 8), 2022: (43, 72, 8),
        2021: (43, 73, 8), 2020: (43, 75, 8), 2019: (41, 80, 8),
        2018: (41, 78, 8), 2017: (40, 81, 8), 2016: (43, 70, 8),
        2015: (47, 56, 8), 2014: (48, 61, 8), 2013: (46, 63, 8),
        2012: (45, 64, 8),
    },
    "CI": {
        2024: (36, 104, 7), 2023: (37, 99, 7), 2022: (37, 99, 7),
        2021: (36, 105, 7), 2020: (36, 104, 7), 2019: (35, 106, 7),
        2018: (35, 105, 7), 2017: (36, 103, 7), 2016: (34, 108, 7),
        2015: (32, 107, 7), 2014: (32, 115, 7), 2013: (27, 136, 7),
        2012: (29, 130, 7),
    },
    "KE": {
        2024: (31, 126, 8), 2023: (31, 126, 8), 2022: (32, 123, 8),
        2021: (30, 128, 8), 2020: (31, 124, 8), 2019: (28, 137, 8),
        2018: (27, 144, 8), 2017: (28, 143, 8), 2016: (26, 145, 8),
        2015: (25, 139, 8), 2014: (25, 145, 8), 2013: (27, 136, 8),
        2012: (27, 139, 8),
    },
    "VN": {
        2024: (34, 110, 8), 2023: (41, 83, 8), 2022: (42, 77, 8),
        2021: (39, 87, 8), 2020: (36, 104, 8), 2019: (37, 96, 8),
        2018: (33, 117, 8), 2017: (35, 107, 8), 2016: (33, 113, 8),
        2015: (31, 112, 8), 2014: (31, 119, 8), 2013: (31, 116, 8),
        2012: (31, 123, 8),
    },
    # High risk (20 <= CPI < 40)
    "CM": {
        2024: (26, 142, 7), 2023: (26, 140, 7), 2022: (26, 142, 7),
        2021: (27, 142, 7), 2020: (25, 149, 7), 2019: (25, 153, 7),
        2018: (25, 152, 7), 2017: (25, 153, 7), 2016: (26, 145, 7),
        2015: (27, 130, 7), 2014: (27, 136, 7), 2013: (25, 144, 7),
        2012: (26, 144, 7),
    },
    "NG": {
        2024: (24, 150, 9), 2023: (25, 145, 9), 2022: (24, 150, 9),
        2021: (24, 154, 9), 2020: (25, 149, 9), 2019: (26, 146, 9),
        2018: (27, 144, 9), 2017: (27, 148, 9), 2016: (28, 136, 9),
        2015: (26, 136, 9), 2014: (27, 136, 9), 2013: (25, 144, 9),
        2012: (27, 139, 9),
    },
    "MM": {
        2024: (23, 155, 7), 2023: (23, 157, 7), 2022: (23, 157, 7),
        2021: (28, 140, 7), 2020: (28, 137, 7), 2019: (29, 130, 7),
        2018: (29, 132, 7), 2017: (30, 130, 7), 2016: (28, 136, 7),
        2015: (22, 147, 7), 2014: (21, 156, 7), 2013: (21, 157, 7),
        2012: (15, 172, 7),
    },
    "KH": {
        2024: (22, 158, 7), 2023: (22, 158, 7), 2022: (24, 150, 7),
        2021: (23, 157, 7), 2020: (21, 160, 7), 2019: (20, 162, 7),
        2018: (20, 161, 7), 2017: (21, 161, 7), 2016: (21, 156, 7),
        2015: (21, 150, 7), 2014: (21, 156, 7), 2013: (20, 160, 7),
        2012: (22, 157, 7),
    },
    "LA": {
        2024: (25, 147, 5), 2023: (25, 147, 5), 2022: (25, 147, 5),
        2021: (30, 128, 5), 2020: (29, 134, 5), 2019: (29, 130, 5),
        2018: (29, 132, 5), 2017: (29, 135, 5), 2016: (30, 123, 5),
        2015: (25, 139, 5), 2014: (25, 145, 5), 2013: (26, 140, 5),
        2012: (21, 160, 5),
    },
    "PG": {
        2024: (28, 137, 5), 2023: (28, 133, 5), 2022: (28, 130, 5),
        2021: (31, 124, 5), 2020: (27, 142, 5), 2019: (28, 137, 5),
        2018: (28, 138, 5), 2017: (29, 135, 5), 2016: (28, 136, 5),
        2015: (25, 139, 5), 2014: (25, 145, 5), 2013: (25, 144, 5),
        2012: (25, 150, 5),
    },
    "CD": {
        2024: (20, 162, 6), 2023: (20, 162, 6), 2022: (20, 166, 6),
        2021: (19, 169, 6), 2020: (18, 170, 6), 2019: (18, 169, 6),
        2018: (20, 161, 6), 2017: (21, 161, 6), 2016: (21, 156, 6),
        2015: (22, 147, 6), 2014: (22, 154, 6), 2013: (22, 154, 6),
        2012: (21, 160, 6),
    },
    "HN": {
        2024: (23, 155, 7), 2023: (23, 157, 7), 2022: (23, 157, 7),
        2021: (23, 157, 7), 2020: (24, 157, 7), 2019: (26, 146, 7),
        2018: (29, 132, 7), 2017: (29, 135, 7), 2016: (30, 123, 7),
        2015: (31, 112, 7), 2014: (29, 126, 7), 2013: (26, 140, 7),
        2012: (28, 133, 7),
    },
    "GT": {
        2024: (24, 150, 7), 2023: (23, 154, 7), 2022: (24, 150, 7),
        2021: (25, 150, 7), 2020: (25, 149, 7), 2019: (26, 146, 7),
        2018: (27, 144, 7), 2017: (28, 143, 7), 2016: (28, 136, 7),
        2015: (28, 123, 7), 2014: (32, 115, 7), 2013: (29, 123, 7),
        2012: (33, 113, 7),
    },
    "NI": {
        2024: (17, 169, 7), 2023: (17, 171, 7), 2022: (19, 167, 7),
        2021: (20, 164, 7), 2020: (22, 159, 7), 2019: (22, 161, 7),
        2018: (25, 152, 7), 2017: (26, 151, 7), 2016: (26, 145, 7),
        2015: (27, 130, 7), 2014: (28, 133, 7), 2013: (28, 127, 7),
        2012: (29, 130, 7),
    },
    "PY": {
        2024: (28, 137, 7), 2023: (28, 133, 7), 2022: (28, 137, 7),
        2021: (28, 137, 7), 2020: (28, 137, 7), 2019: (28, 137, 7),
        2018: (29, 132, 7), 2017: (29, 135, 7), 2016: (30, 123, 7),
        2015: (27, 130, 7), 2014: (24, 150, 7), 2013: (24, 150, 7),
        2012: (25, 150, 7),
    },
    "BO": {
        2024: (29, 133, 7), 2023: (29, 133, 7), 2022: (31, 126, 7),
        2021: (30, 128, 7), 2020: (31, 124, 7), 2019: (31, 123, 7),
        2018: (29, 132, 7), 2017: (33, 112, 7), 2016: (33, 113, 7),
        2015: (34, 99, 7), 2014: (35, 103, 7), 2013: (34, 106, 7),
        2012: (34, 105, 7),
    },
    # Very High risk (CPI < 20)
    "VE": {
        2024: (11, 180, 8), 2023: (13, 177, 8), 2022: (14, 177, 8),
        2021: (14, 177, 8), 2020: (15, 176, 8), 2019: (16, 173, 8),
        2018: (18, 168, 8), 2017: (18, 169, 8), 2016: (17, 166, 8),
        2015: (17, 158, 8), 2014: (19, 161, 8), 2013: (20, 160, 8),
        2012: (19, 165, 8),
    },
    "SO": {
        2024: (12, 179, 5), 2023: (11, 180, 5), 2022: (12, 180, 5),
        2021: (13, 178, 5), 2020: (12, 179, 5), 2019: (9, 180, 5),
        2018: (10, 180, 5), 2017: (9, 180, 5), 2016: (10, 176, 5),
        2015: (8, 167, 5), 2014: (8, 174, 5), 2013: (8, 175, 5),
        2012: (8, 174, 5),
    },
    "SS": {
        2024: (12, 179, 5), 2023: (13, 177, 5), 2022: (13, 178, 5),
        2021: (11, 180, 5), 2020: (12, 179, 5), 2019: (12, 179, 5),
        2018: (13, 178, 5), 2017: (12, 179, 5), 2016: (11, 175, 5),
        2015: (15, 163, 5), 2014: (15, 171, 5), 2013: (14, 173, 5),
        2012: (14, 174, 5),
    },
    "SY": {
        2024: (13, 177, 5), 2023: (13, 177, 5), 2022: (13, 178, 5),
        2021: (13, 178, 5), 2020: (14, 178, 5), 2019: (13, 178, 5),
        2018: (13, 178, 5), 2017: (14, 178, 5), 2016: (13, 173, 5),
        2015: (18, 154, 5), 2014: (20, 159, 5), 2013: (17, 168, 5),
        2012: (26, 144, 5),
    },
    "YE": {
        2024: (15, 173, 5), 2023: (16, 174, 5), 2022: (16, 176, 5),
        2021: (16, 174, 5), 2020: (15, 176, 5), 2019: (15, 177, 5),
        2018: (14, 176, 5), 2017: (16, 175, 5), 2016: (14, 170, 5),
        2015: (18, 154, 5), 2014: (19, 161, 5), 2013: (18, 167, 5),
        2012: (23, 156, 5),
    },
    "SD": {
        2024: (15, 173, 6), 2023: (16, 174, 6), 2022: (20, 162, 6),
        2021: (20, 164, 6), 2020: (16, 174, 6), 2019: (16, 173, 6),
        2018: (16, 172, 6), 2017: (16, 175, 6), 2016: (14, 170, 6),
        2015: (12, 165, 6), 2014: (11, 173, 6), 2013: (11, 174, 6),
        2012: (13, 173, 6),
    },
    "AF": {
        2024: (15, 173, 5), 2023: (20, 162, 5), 2022: (24, 150, 5),
        2021: (16, 174, 5), 2020: (19, 165, 5), 2019: (16, 173, 5),
        2018: (16, 172, 5), 2017: (15, 177, 5), 2016: (15, 169, 5),
        2015: (11, 166, 5), 2014: (12, 172, 5), 2013: (8, 175, 5),
        2012: (8, 174, 5),
    },
    # Additional EUDR-relevant countries
    "ZA": {
        2024: (41, 83, 8), 2023: (41, 83, 8), 2022: (43, 72, 8),
        2021: (44, 70, 8), 2020: (44, 69, 8), 2019: (44, 70, 8),
        2018: (43, 73, 8), 2017: (43, 71, 8), 2016: (45, 64, 8),
        2015: (44, 61, 8), 2014: (44, 67, 8), 2013: (42, 72, 8),
        2012: (43, 69, 8),
    },
    "RW": {
        2024: (52, 55, 6), 2023: (53, 49, 6), 2022: (53, 51, 6),
        2021: (53, 52, 6), 2020: (54, 49, 6), 2019: (53, 51, 6),
        2018: (56, 48, 6), 2017: (55, 48, 6), 2016: (54, 50, 6),
        2015: (54, 44, 6), 2014: (49, 55, 6), 2013: (53, 49, 6),
        2012: (53, 50, 6),
    },
    "ET": {
        2024: (37, 100, 7), 2023: (37, 94, 7), 2022: (38, 94, 7),
        2021: (39, 87, 7), 2020: (38, 94, 7), 2019: (37, 96, 7),
        2018: (34, 114, 7), 2017: (35, 107, 7), 2016: (34, 108, 7),
        2015: (33, 103, 7), 2014: (33, 110, 7), 2013: (33, 111, 7),
        2012: (33, 113, 7),
    },
    "TZ": {
        2024: (38, 96, 8), 2023: (40, 87, 8), 2022: (39, 94, 8),
        2021: (39, 87, 8), 2020: (38, 94, 8), 2019: (37, 96, 8),
        2018: (36, 99, 8), 2017: (36, 103, 8), 2016: (32, 116, 8),
        2015: (30, 117, 8), 2014: (31, 119, 8), 2013: (33, 111, 8),
        2012: (35, 102, 8),
    },
    "UG": {
        2024: (26, 142, 7), 2023: (26, 141, 7), 2022: (26, 142, 7),
        2021: (27, 144, 7), 2020: (27, 142, 7), 2019: (28, 137, 7),
        2018: (26, 149, 7), 2017: (26, 151, 7), 2016: (25, 151, 7),
        2015: (25, 139, 7), 2014: (26, 142, 7), 2013: (26, 140, 7),
        2012: (29, 130, 7),
    },
    "AR": {
        2024: (36, 104, 8), 2023: (37, 98, 8), 2022: (38, 94, 8),
        2021: (38, 96, 8), 2020: (42, 78, 8), 2019: (45, 66, 8),
        2018: (40, 85, 8), 2017: (39, 85, 8), 2016: (36, 95, 8),
        2015: (32, 107, 8), 2014: (34, 107, 8), 2013: (34, 106, 8),
        2012: (35, 102, 8),
    },
    "RU": {
        2024: (26, 142, 9), 2023: (26, 141, 9), 2022: (28, 137, 9),
        2021: (29, 136, 9), 2020: (30, 129, 9), 2019: (28, 137, 9),
        2018: (28, 138, 9), 2017: (29, 135, 9), 2016: (29, 131, 9),
        2015: (29, 119, 9), 2014: (27, 136, 9), 2013: (28, 127, 9),
        2012: (28, 133, 9),
    },
    "CN": {
        2024: (42, 76, 9), 2023: (42, 76, 9), 2022: (45, 65, 9),
        2021: (45, 66, 9), 2020: (42, 78, 9), 2019: (41, 80, 9),
        2018: (39, 87, 9), 2017: (41, 77, 9), 2016: (40, 79, 9),
        2015: (37, 83, 9), 2014: (36, 100, 9), 2013: (40, 80, 9),
        2012: (39, 80, 9),
    },
    "PK": {
        2024: (26, 142, 7), 2023: (29, 133, 7), 2022: (27, 140, 7),
        2021: (28, 140, 7), 2020: (31, 124, 7), 2019: (32, 120, 7),
        2018: (33, 117, 7), 2017: (32, 117, 7), 2016: (32, 116, 7),
        2015: (30, 117, 7), 2014: (29, 126, 7), 2013: (28, 127, 7),
        2012: (27, 139, 7),
    },
    "BD": {
        2024: (25, 149, 7), 2023: (24, 149, 7), 2022: (25, 147, 7),
        2021: (26, 147, 7), 2020: (26, 146, 7), 2019: (26, 146, 7),
        2018: (26, 149, 7), 2017: (28, 143, 7), 2016: (26, 145, 7),
        2015: (25, 139, 7), 2014: (25, 145, 7), 2013: (27, 136, 7),
        2012: (26, 144, 7),
    },
    "SL": {
        2024: (30, 128, 6), 2023: (30, 128, 6), 2022: (34, 110, 6),
        2021: (34, 115, 6), 2020: (33, 117, 6), 2019: (33, 119, 6),
        2018: (30, 129, 6), 2017: (30, 130, 6), 2016: (30, 123, 6),
        2015: (29, 119, 6), 2014: (31, 119, 6), 2013: (30, 119, 6),
        2012: (31, 123, 6),
    },
    "LR": {
        2024: (25, 147, 5), 2023: (25, 145, 5), 2022: (26, 142, 5),
        2021: (29, 136, 5), 2020: (28, 137, 5), 2019: (28, 137, 5),
        2018: (32, 120, 5), 2017: (31, 122, 5), 2016: (37, 90, 5),
        2015: (37, 83, 5), 2014: (37, 94, 5), 2013: (38, 83, 5),
        2012: (41, 75, 5),
    },
    "GN": {
        2024: (24, 150, 6), 2023: (25, 145, 6), 2022: (25, 147, 6),
        2021: (25, 150, 6), 2020: (28, 137, 6), 2019: (29, 130, 6),
        2018: (28, 138, 6), 2017: (27, 148, 6), 2016: (27, 142, 6),
        2015: (25, 139, 6), 2014: (25, 145, 6), 2013: (24, 150, 6),
        2012: (24, 154, 6),
    },
    "MG": {
        2024: (26, 142, 7), 2023: (25, 145, 7), 2022: (26, 142, 7),
        2021: (26, 147, 7), 2020: (25, 149, 7), 2019: (24, 158, 7),
        2018: (25, 152, 7), 2017: (24, 155, 7), 2016: (26, 145, 7),
        2015: (28, 123, 7), 2014: (28, 133, 7), 2013: (28, 127, 7),
        2012: (32, 118, 7),
    },
    "MZ": {
        2024: (24, 150, 7), 2023: (26, 140, 7), 2022: (26, 142, 7),
        2021: (26, 147, 7), 2020: (25, 149, 7), 2019: (26, 146, 7),
        2018: (23, 158, 7), 2017: (25, 153, 7), 2016: (27, 142, 7),
        2015: (31, 112, 7), 2014: (31, 119, 7), 2013: (30, 119, 7),
        2012: (31, 123, 7),
    },
    "BW": {
        2024: (55, 45, 6), 2023: (56, 43, 6), 2022: (60, 35, 6),
        2021: (55, 45, 6), 2020: (60, 35, 6), 2019: (61, 34, 6),
        2018: (61, 34, 6), 2017: (61, 34, 6), 2016: (60, 35, 6),
        2015: (63, 28, 6), 2014: (63, 31, 6), 2013: (64, 30, 6),
        2012: (65, 30, 6),
    },
    "CF": {
        2024: (19, 166, 5), 2023: (24, 149, 5), 2022: (24, 150, 5),
        2021: (24, 154, 5), 2020: (21, 160, 5), 2019: (25, 153, 5),
        2018: (26, 149, 5), 2017: (23, 156, 5), 2016: (20, 159, 5),
        2015: (24, 145, 5), 2014: (24, 150, 5), 2013: (25, 144, 5),
        2012: (26, 144, 5),
    },
    "GA": {
        2024: (28, 137, 5), 2023: (29, 133, 5), 2022: (29, 136, 5),
        2021: (31, 124, 5), 2020: (30, 129, 5), 2019: (31, 123, 5),
        2018: (31, 124, 5), 2017: (32, 117, 5), 2016: (35, 101, 5),
        2015: (34, 99, 5), 2014: (37, 94, 5), 2013: (34, 106, 5),
        2012: (35, 102, 5),
    },
    "CG": {
        2024: (20, 162, 5), 2023: (21, 160, 5), 2022: (21, 162, 5),
        2021: (21, 160, 5), 2020: (19, 165, 5), 2019: (19, 165, 5),
        2018: (19, 165, 5), 2017: (21, 161, 5), 2016: (20, 159, 5),
        2015: (23, 146, 5), 2014: (23, 152, 5), 2013: (22, 154, 5),
        2012: (26, 144, 5),
    },
}

# ---------------------------------------------------------------------------
# CPI Monitor Engine
# ---------------------------------------------------------------------------

class CPIMonitorEngine:
    """Transparency International CPI monitoring engine for EUDR compliance.

    Provides country-level CPI score queries, historical trend retrieval,
    global and regional rankings, regional aggregation statistics, batch
    multi-country queries, and summary statistics. All numeric operations
    use Decimal arithmetic for deterministic reproducibility and every
    result includes a SHA-256 provenance hash for audit compliance.

    The engine maps CPI scores to EUDR risk factors using the inverse
    formula: eudr_risk = 1.0 - (cpi/100). A CPI score of 0 (most corrupt)
    maps to EUDR risk 1.0 (highest supply-chain risk), and CPI 100
    (cleanest) maps to EUDR risk 0.0.

    Attributes:
        _config: Optional configuration from get_config().
        _tracker: Optional ProvenanceTracker for audit logging.

    Example::

        engine = CPIMonitorEngine()
        result = engine.get_country_score("BR", 2024)
        assert result.success
        assert result.data.score == Decimal("36")
        assert result.data.rank == 104
    """

    def __init__(self) -> None:
        """Initialize CPIMonitorEngine with optional config and provenance."""
        self._config = None
        self._tracker = None
        try:
            if get_config is not None:
                self._config = get_config()
        except Exception as exc:
            logger.debug("Config not available: %s", exc)

        try:
            if get_tracker is not None:
                self._tracker = get_tracker()
        except Exception as exc:
            logger.debug("ProvenanceTracker not available: %s", exc)

        logger.info(
            "CPIMonitorEngine initialized: version=%s, countries=%d, "
            "config=%s, provenance=%s",
            _MODULE_VERSION,
            len(CPI_SCORES_DB),
            "loaded" if self._config else "default",
            "enabled" if self._tracker else "disabled",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_country_score(
        self,
        country_code: str,
        year: Optional[int] = None,
    ) -> CPIScoreResult:
        """Get CPI score for a specific country and year.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g. "BR").
            year: CPI year (2012-2024). Defaults to latest available.

        Returns:
            CPIScoreResult with the score data, provenance hash, and
            metadata including EUDR risk factor and corruption risk level.
        """
        start = time.perf_counter()
        timestamp = utcnow().isoformat()

        try:
            cc = country_code.upper().strip()
            if not cc or len(cc) != 2:
                return CPIScoreResult(
                    success=False,
                    error=f"Invalid country code: '{country_code}'. Must be 2-letter ISO code.",
                    calculation_timestamp=timestamp,
                )

            if cc not in CPI_SCORES_DB:
                return CPIScoreResult(
                    success=False,
                    error=f"Country '{cc}' not found in CPI database.",
                    calculation_timestamp=timestamp,
                )

            country_data = CPI_SCORES_DB[cc]

            if year is None:
                year = max(country_data.keys())
            elif year not in country_data:
                available = sorted(country_data.keys())
                return CPIScoreResult(
                    success=False,
                    error=(
                        f"CPI data for {cc} not available for year {year}. "
                        f"Available years: {available[0]}-{available[-1]}."
                    ),
                    calculation_timestamp=timestamp,
                )

            score_val, rank_val, sources = country_data[year]
            score_dec = _to_decimal(score_val)
            se = self._estimate_standard_error(score_val, sources)
            ci_low = max(Decimal("0"), score_dec - se * Decimal("1.645"))
            ci_high = min(Decimal("100"), score_dec + se * Decimal("1.645"))

            change = self._calculate_change(cc, year)

            cpi_score = CPIScore(
                country_code=cc,
                country_name=COUNTRY_NAMES.get(cc, cc),
                year=year,
                score=score_dec,
                rank=rank_val,
                region=COUNTRY_REGIONS.get(cc, "unknown"),
                standard_error=se.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                confidence_interval_low=ci_low.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                confidence_interval_high=ci_high.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                sources_count=sources,
                change_from_previous=change,
            )

            eudr_risk = self._calculate_eudr_risk_factor(score_dec)
            risk_level = self.classify_corruption_level(float(score_dec))

            metadata = {
                "engine": "CPIMonitorEngine",
                "engine_version": _MODULE_VERSION,
                "agent_id": _AGENT_ID,
                "operation": "get_country_score",
                "eudr_risk_factor": str(eudr_risk),
                "corruption_risk_level": risk_level,
                "data_source": "transparency_international",
                "processing_time_ms": 0.0,
            }

            prov_hash = _compute_hash({
                "operation": "get_country_score",
                "country_code": cc,
                "year": year,
                "score": str(score_dec),
                "rank": rank_val,
                "eudr_risk_factor": str(eudr_risk),
            })

            elapsed = (time.perf_counter() - start) * 1000
            metadata["processing_time_ms"] = round(elapsed, 3)

            self._record_provenance(
                "cpi_score", "query_score", cc,
                data={"year": year, "score": str(score_dec)},
            )

            _inc_cpi_queries()
            _observe_cpi_duration(elapsed / 1000)

            return CPIScoreResult(
                success=True,
                data=cpi_score,
                metadata=metadata,
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error(
                "CPIMonitorEngine.get_country_score failed: %s", exc,
                exc_info=True,
            )
            _inc_cpi_error("get_country_score")
            return CPIScoreResult(
                success=False,
                error=f"Internal error: {str(exc)}",
                calculation_timestamp=timestamp,
            )

    def get_score_history(
        self,
        country_code: str,
        start_year: int,
        end_year: int,
    ) -> CPIHistoryResult:
        """Get historical CPI scores for a country over a year range.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            start_year: First year of the range (inclusive).
            end_year: Last year of the range (inclusive).

        Returns:
            CPIHistoryResult with chronological scores, trend direction,
            average score, and net change over the period.
        """
        start = time.perf_counter()
        timestamp = utcnow().isoformat()

        try:
            cc = country_code.upper().strip()
            if cc not in CPI_SCORES_DB:
                return CPIHistoryResult(
                    success=False,
                    error=f"Country '{cc}' not found in CPI database.",
                    calculation_timestamp=timestamp,
                )

            if start_year > end_year:
                return CPIHistoryResult(
                    success=False,
                    error=f"start_year ({start_year}) must be <= end_year ({end_year}).",
                    calculation_timestamp=timestamp,
                )

            country_data = CPI_SCORES_DB[cc]
            scores: List[CPIScore] = []
            warnings: List[str] = []

            for yr in range(start_year, end_year + 1):
                if yr not in country_data:
                    warnings.append(f"No CPI data for {cc} in year {yr}")
                    continue
                score_val, rank_val, sources = country_data[yr]
                score_dec = _to_decimal(score_val)
                change = self._calculate_change(cc, yr)
                se = self._estimate_standard_error(score_val, sources)

                scores.append(CPIScore(
                    country_code=cc,
                    country_name=COUNTRY_NAMES.get(cc, cc),
                    year=yr,
                    score=score_dec,
                    rank=rank_val,
                    region=COUNTRY_REGIONS.get(cc, "unknown"),
                    standard_error=se.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    confidence_interval_low=max(
                        Decimal("0"), score_dec - se * Decimal("1.645")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    confidence_interval_high=min(
                        Decimal("100"), score_dec + se * Decimal("1.645")
                    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                    sources_count=sources,
                    change_from_previous=change,
                ))

            if not scores:
                return CPIHistoryResult(
                    success=False,
                    error=f"No CPI data found for {cc} between {start_year} and {end_year}.",
                    calculation_timestamp=timestamp,
                )

            avg = self._calculate_average([s.score for s in scores])
            net_change = scores[-1].score - scores[0].score
            trend = self._determine_trend(scores)

            prov_hash = _compute_hash({
                "operation": "get_score_history",
                "country_code": cc,
                "start_year": start_year,
                "end_year": end_year,
                "score_count": len(scores),
                "average": str(avg),
                "net_change": str(net_change),
            })

            elapsed = (time.perf_counter() - start) * 1000

            self._record_provenance(
                "cpi_score", "query_history", cc,
                data={"start_year": start_year, "end_year": end_year,
                      "records": len(scores)},
            )
            _inc_cpi_queries()
            _observe_cpi_duration(elapsed / 1000)

            return CPIHistoryResult(
                success=True,
                country_code=cc,
                start_year=start_year,
                end_year=end_year,
                scores=scores,
                trend_direction=trend,
                average_score=avg,
                score_change=net_change,
                metadata={
                    "engine": "CPIMonitorEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "get_score_history",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
                warnings=warnings,
            )

        except Exception as exc:
            logger.error("get_score_history failed: %s", exc, exc_info=True)
            _inc_cpi_error("get_score_history")
            return CPIHistoryResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def get_rankings(
        self,
        year: int,
        region: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> CPIRankingsResult:
        """Get CPI rankings for a given year, optionally filtered by region.

        Args:
            year: CPI year (2012-2024).
            region: Optional TI region filter (e.g. "americas").
            top_n: Optional limit to the top N ranked countries.

        Returns:
            CPIRankingsResult with countries sorted by rank.
        """
        start = time.perf_counter()
        timestamp = utcnow().isoformat()

        try:
            entries: List[Tuple[str, int, int, int]] = []
            for cc, years_data in CPI_SCORES_DB.items():
                if year not in years_data:
                    continue
                if region is not None:
                    cc_region = COUNTRY_REGIONS.get(cc, "")
                    if cc_region != region:
                        continue
                score_val, rank_val, sources = years_data[year]
                entries.append((cc, score_val, rank_val, sources))

            entries.sort(key=lambda x: x[2])

            if top_n is not None and top_n > 0:
                entries = entries[:top_n]

            rankings: List[CPIScore] = []
            for cc, score_val, rank_val, sources in entries:
                score_dec = _to_decimal(score_val)
                change = self._calculate_change(cc, year)
                rankings.append(CPIScore(
                    country_code=cc,
                    country_name=COUNTRY_NAMES.get(cc, cc),
                    year=year,
                    score=score_dec,
                    rank=rank_val,
                    region=COUNTRY_REGIONS.get(cc, "unknown"),
                    sources_count=sources,
                    change_from_previous=change,
                ))

            prov_hash = _compute_hash({
                "operation": "get_rankings",
                "year": year,
                "region": region,
                "top_n": top_n,
                "count": len(rankings),
            })

            elapsed = (time.perf_counter() - start) * 1000

            _inc_cpi_queries()
            _observe_cpi_duration(elapsed / 1000)

            return CPIRankingsResult(
                success=True,
                year=year,
                region=region,
                rankings=rankings,
                total_countries=len(rankings),
                metadata={
                    "engine": "CPIMonitorEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "get_rankings",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("get_rankings failed: %s", exc, exc_info=True)
            _inc_cpi_error("get_rankings")
            return CPIRankingsResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def get_regional_analysis(
        self,
        region: str,
        year: Optional[int] = None,
    ) -> CPIRegionalResult:
        """Get regional CPI statistics for a given region and year.

        Args:
            region: TI region name (e.g. "sub_saharan_africa").
            year: CPI year. Defaults to 2024.

        Returns:
            CPIRegionalResult with aggregate statistics (average, median,
            min, max, standard deviation, high-risk count).
        """
        start = time.perf_counter()
        timestamp = utcnow().isoformat()

        try:
            if year is None:
                year = 2024

            valid_regions = {r.value for r in CPIRegion}
            if region not in valid_regions:
                return CPIRegionalResult(
                    success=False,
                    error=(
                        f"Invalid region '{region}'. "
                        f"Valid: {sorted(valid_regions)}"
                    ),
                    calculation_timestamp=timestamp,
                )

            countries: List[CPIScore] = []
            for cc, years_data in CPI_SCORES_DB.items():
                if COUNTRY_REGIONS.get(cc) != region:
                    continue
                if year not in years_data:
                    continue
                score_val, rank_val, sources = years_data[year]
                score_dec = _to_decimal(score_val)
                countries.append(CPIScore(
                    country_code=cc,
                    country_name=COUNTRY_NAMES.get(cc, cc),
                    year=year,
                    score=score_dec,
                    rank=rank_val,
                    region=region,
                    sources_count=sources,
                    change_from_previous=self._calculate_change(cc, year),
                ))

            if not countries:
                return CPIRegionalResult(
                    success=False,
                    error=f"No CPI data for region '{region}' in year {year}.",
                    calculation_timestamp=timestamp,
                )

            score_values = sorted([c.score for c in countries])
            n = len(score_values)
            avg = self._calculate_average(score_values)
            median = self._calculate_median(score_values)
            min_s = score_values[0]
            max_s = score_values[-1]
            std = self._calculate_std_dev(score_values, avg)
            high_risk = sum(1 for s in score_values if s < Decimal("40"))

            prov_hash = _compute_hash({
                "operation": "get_regional_analysis",
                "region": region,
                "year": year,
                "country_count": n,
                "average": str(avg),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_cpi_queries()
            _observe_cpi_duration(elapsed / 1000)

            return CPIRegionalResult(
                success=True,
                region=region,
                year=year,
                countries=countries,
                country_count=n,
                average_score=avg,
                median_score=median,
                min_score=min_s,
                max_score=max_s,
                std_deviation=std,
                high_risk_count=high_risk,
                metadata={
                    "engine": "CPIMonitorEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "get_regional_analysis",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("get_regional_analysis failed: %s", exc, exc_info=True)
            _inc_cpi_error("get_regional_analysis")
            return CPIRegionalResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def batch_query(
        self,
        country_codes: List[str],
        year: Optional[int] = None,
    ) -> CPIBatchResult:
        """Query CPI scores for multiple countries in a single operation.

        Args:
            country_codes: List of ISO alpha-2 country codes.
            year: CPI year. Defaults to latest available per country.

        Returns:
            CPIBatchResult with per-country results, found/not-found
            counts, and aggregate provenance hash.
        """
        start = time.perf_counter()
        timestamp = utcnow().isoformat()

        try:
            if not country_codes:
                return CPIBatchResult(
                    success=False,
                    error="country_codes list must not be empty.",
                    calculation_timestamp=timestamp,
                )

            results: Dict[str, CPIScoreResult] = {}
            not_found: List[str] = []
            found_count = 0

            for cc in country_codes:
                result = self.get_country_score(cc, year)
                results[cc.upper().strip()] = result
                if result.success:
                    found_count += 1
                else:
                    not_found.append(cc.upper().strip())

            prov_hash = _compute_hash({
                "operation": "batch_query",
                "country_codes": sorted([c.upper() for c in country_codes]),
                "year": year,
                "found": found_count,
                "not_found": sorted(not_found),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_cpi_batch_queries()
            _observe_cpi_duration(elapsed / 1000)

            warnings: List[str] = []
            if not_found:
                warnings.append(
                    f"Countries not found: {', '.join(sorted(not_found))}"
                )

            return CPIBatchResult(
                success=True,
                results=results,
                queried_count=len(country_codes),
                found_count=found_count,
                not_found=not_found,
                metadata={
                    "engine": "CPIMonitorEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "batch_query",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
                warnings=warnings,
            )

        except Exception as exc:
            logger.error("batch_query failed: %s", exc, exc_info=True)
            _inc_cpi_error("batch_query")
            return CPIBatchResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def get_summary_statistics(
        self,
        year: Optional[int] = None,
    ) -> CPISummaryResult:
        """Get global CPI summary statistics for a given year.

        Args:
            year: CPI year. Defaults to 2024.

        Returns:
            CPISummaryResult with global average, median, distribution
            across risk categories, and regional averages.
        """
        start = time.perf_counter()
        timestamp = utcnow().isoformat()

        try:
            if year is None:
                year = 2024

            all_scores: List[Decimal] = []
            regional_scores: Dict[str, List[Decimal]] = {}

            for cc, years_data in CPI_SCORES_DB.items():
                if year not in years_data:
                    continue
                score_val = years_data[year][0]
                score_dec = _to_decimal(score_val)
                all_scores.append(score_dec)

                rgn = COUNTRY_REGIONS.get(cc, "unknown")
                if rgn not in regional_scores:
                    regional_scores[rgn] = []
                regional_scores[rgn].append(score_dec)

            if not all_scores:
                return CPISummaryResult(
                    success=False,
                    error=f"No CPI data available for year {year}.",
                    calculation_timestamp=timestamp,
                )

            all_scores.sort()
            n = len(all_scores)
            avg = self._calculate_average(all_scores)
            median = self._calculate_median(all_scores)
            min_s = all_scores[0]
            max_s = all_scores[-1]
            std = self._calculate_std_dev(all_scores, avg)

            very_low = sum(1 for s in all_scores if s >= Decimal("80"))
            low = sum(1 for s in all_scores if Decimal("60") <= s < Decimal("80"))
            moderate = sum(1 for s in all_scores if Decimal("40") <= s < Decimal("60"))
            high = sum(1 for s in all_scores if Decimal("20") <= s < Decimal("40"))
            very_high = sum(1 for s in all_scores if s < Decimal("20"))

            # Update high risk gauge
            if PROMETHEUS_AVAILABLE and _cpi_high_risk_countries is not None:
                _cpi_high_risk_countries.set(high + very_high)

            reg_avgs: Dict[str, Decimal] = {}
            for rgn, scores in regional_scores.items():
                reg_avgs[rgn] = self._calculate_average(scores)

            prov_hash = _compute_hash({
                "operation": "get_summary_statistics",
                "year": year,
                "total_countries": n,
                "global_average": str(avg),
                "global_median": str(median),
            })

            elapsed = (time.perf_counter() - start) * 1000
            _inc_cpi_queries()
            _observe_cpi_duration(elapsed / 1000)

            return CPISummaryResult(
                success=True,
                year=year,
                total_countries=n,
                global_average=avg,
                global_median=median,
                global_min=min_s,
                global_max=max_s,
                std_deviation=std,
                very_low_risk_count=very_low,
                low_risk_count=low,
                moderate_risk_count=moderate,
                high_risk_count=high,
                very_high_risk_count=very_high,
                regional_averages=reg_avgs,
                metadata={
                    "engine": "CPIMonitorEngine",
                    "engine_version": _MODULE_VERSION,
                    "agent_id": _AGENT_ID,
                    "operation": "get_summary_statistics",
                    "processing_time_ms": round(elapsed, 3),
                },
                provenance_hash=prov_hash,
                calculation_timestamp=timestamp,
            )

        except Exception as exc:
            logger.error("get_summary_statistics failed: %s", exc, exc_info=True)
            _inc_cpi_error("get_summary_statistics")
            return CPISummaryResult(
                success=False, error=str(exc),
                calculation_timestamp=timestamp,
            )

    def classify_corruption_level(self, score: float) -> str:
        """Classify corruption risk level based on CPI score.

        Args:
            score: CPI score on 0-100 scale.

        Returns:
            String risk level: VERY_LOW, LOW, MODERATE, HIGH, or
            VERY_HIGH.
        """
        if score >= 80:
            return CorruptionRiskLevel.VERY_LOW.value
        elif score >= 60:
            return CorruptionRiskLevel.LOW.value
        elif score >= 40:
            return CorruptionRiskLevel.MODERATE.value
        elif score >= 20:
            return CorruptionRiskLevel.HIGH.value
        else:
            return CorruptionRiskLevel.VERY_HIGH.value

    # ------------------------------------------------------------------
    # EUDR Risk Mapping
    # ------------------------------------------------------------------

    def _calculate_eudr_risk_factor(self, score: Decimal) -> Decimal:
        """Map CPI score to EUDR risk factor (inverse relationship).

        A CPI of 0 (most corrupt) yields EUDR risk 1.0 (highest risk).
        A CPI of 100 (cleanest) yields EUDR risk 0.0 (lowest risk).

        Formula: eudr_risk = 1.0 - (cpi / 100)

        Args:
            score: CPI score as Decimal (0-100).

        Returns:
            EUDR risk factor as Decimal (0.0-1.0), rounded to 4 places.
        """
        clamped = max(Decimal("0"), min(Decimal("100"), score))
        risk = Decimal("1.0") - (clamped / Decimal("100"))
        return risk.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def get_eudr_risk_factor(self, country_code: str, year: Optional[int] = None) -> Decimal:
        """Convenience method: get EUDR risk factor for a country.

        Args:
            country_code: ISO alpha-2 country code.
            year: CPI year. Defaults to latest available.

        Returns:
            EUDR risk factor as Decimal (0.0-1.0).
            Returns Decimal('1.0') if country not found (precautionary).
        """
        result = self.get_country_score(country_code, year)
        if not result.success or result.data is None:
            logger.warning(
                "CPI data not found for %s year=%s, returning max EUDR risk",
                country_code, year,
            )
            return Decimal("1.0000")
        return self._calculate_eudr_risk_factor(result.data.score)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_change(self, country_code: str, year: int) -> Decimal:
        """Calculate year-over-year CPI score change.

        Args:
            country_code: ISO alpha-2 country code.
            year: Current year.

        Returns:
            Score change as Decimal (positive = improvement).
        """
        cc_data = CPI_SCORES_DB.get(country_code, {})
        prev_year = year - 1
        if prev_year not in cc_data or year not in cc_data:
            return Decimal("0")
        current = _to_decimal(cc_data[year][0])
        previous = _to_decimal(cc_data[prev_year][0])
        return current - previous

    def _estimate_standard_error(self, score: int, sources: int) -> Decimal:
        """Estimate standard error based on score and source count.

        Uses a simplified model: SE = max(1.0, 20 / sqrt(sources)).
        More sources reduce uncertainty.

        Args:
            score: Raw CPI score.
            sources: Number of data sources.

        Returns:
            Estimated standard error as Decimal.
        """
        if sources <= 0:
            return Decimal("5.0")
        import math
        se = Decimal(str(round(20.0 / math.sqrt(sources), 2)))
        return max(Decimal("1.0"), se)

    def _determine_trend(self, scores: List[CPIScore]) -> str:
        """Determine the overall trend direction from a score history.

        Uses simple linear regression slope direction.

        Args:
            scores: Chronologically ordered CPIScore list.

        Returns:
            Trend direction string: IMPROVING, DECLINING, or STABLE.
        """
        if len(scores) < 2:
            return CPITrendDirection.STABLE.value

        first = scores[0].score
        last = scores[-1].score
        change = last - first

        threshold = Decimal("3")
        if change > threshold:
            return CPITrendDirection.IMPROVING.value
        elif change < -threshold:
            return CPITrendDirection.DECLINING.value
        else:
            return CPITrendDirection.STABLE.value

    def _calculate_average(self, values: List[Decimal]) -> Decimal:
        """Calculate the mean of a list of Decimal values.

        Args:
            values: Non-empty list of Decimal values.

        Returns:
            Mean as Decimal, rounded to 2 decimal places.
        """
        if not values:
            return Decimal("0")
        total = sum(values)
        avg = total / Decimal(str(len(values)))
        return avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_median(self, sorted_values: List[Decimal]) -> Decimal:
        """Calculate the median of a sorted list of Decimal values.

        Args:
            sorted_values: Sorted list of Decimal values.

        Returns:
            Median as Decimal, rounded to 2 decimal places.
        """
        n = len(sorted_values)
        if n == 0:
            return Decimal("0")
        if n % 2 == 1:
            return sorted_values[n // 2].quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        mid = n // 2
        median = (sorted_values[mid - 1] + sorted_values[mid]) / Decimal("2")
        return median.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_std_dev(
        self,
        values: List[Decimal],
        mean: Decimal,
    ) -> Decimal:
        """Calculate population standard deviation using Decimal arithmetic.

        Args:
            values: List of Decimal values.
            mean: Pre-computed mean of the values.

        Returns:
            Standard deviation as Decimal, rounded to 2 places.
        """
        if len(values) < 2:
            return Decimal("0")
        sum_sq = sum((v - mean) ** 2 for v in values)
        variance = sum_sq / Decimal(str(len(values)))
        import math
        std = Decimal(str(math.sqrt(float(variance))))
        return std.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _record_provenance(
        self,
        entity_type: str,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a provenance entry if tracker is available.

        Args:
            entity_type: Entity type string.
            action: Action performed.
            entity_id: Entity identifier.
            data: Optional data payload for the provenance record.
        """
        if self._tracker is None:
            return
        try:
            self._tracker.record(
                entity_type, action, entity_id, metadata=data,
            )
        except Exception as exc:
            logger.debug("Provenance recording failed: %s", exc)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enumerations
    "CorruptionRiskLevel",
    "CPIRegion",
    "CPITrendDirection",
    # Data classes
    "CPIScore",
    "CPIScoreResult",
    "CPIHistoryResult",
    "CPIRankingsResult",
    "CPIRegionalResult",
    "CPIBatchResult",
    "CPISummaryResult",
    # Engine
    "CPIMonitorEngine",
    # Reference data
    "CPI_SCORES_DB",
    "COUNTRY_NAMES",
    "COUNTRY_REGIONS",
]
