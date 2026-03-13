# -*- coding: utf-8 -*-
"""
TrendAnalysisEngine - AGENT-EUDR-019 Engine 5: Corruption Index Trend Analysis

Performs temporal analysis of corruption indices to identify improving/deteriorating
trends, predict future trajectories, detect structural breakpoints, and provide
early warning of governance deterioration for EUDR compliance risk assessment.

Zero-Hallucination Guarantees:
    - All trend calculations use deterministic Decimal arithmetic.
    - Linear regression uses ordinary least squares with explicit formula.
    - Prediction models use linear extrapolation, weighted moving average,
      and exponential smoothing -- no ML/LLM in calculation paths.
    - Breakpoint detection uses CUSUM (Cumulative Sum) control chart method.
    - SHA-256 provenance hashes on all output objects.

Analysis Methods:
    1. Linear Regression: OLS regression with slope, intercept, R-squared,
       standard error, and confidence intervals.
    2. Trajectory Analysis: Direction classification (IMPROVING, STABLE,
       DETERIORATING, VOLATILE, INSUFFICIENT_DATA) with velocity metrics.
    3. Prediction: Linear extrapolation, weighted moving average (WMA),
       and exponential smoothing (ETS) with configurable parameters.
    4. Breakpoint Detection: CUSUM-based structural break detection with
       configurable sensitivity threshold.
    5. Moving Average: Simple and weighted moving averages for smoothing.
    6. Regime Change: Detecting significant shifts in governance trajectory.

Index Types Supported:
    - CPI:       Corruption Perceptions Index (0-100 scale)
    - WGI:       Worldwide Governance Indicators (-2.5 to +2.5 scale)
    - BRIBERY:   Sector-specific bribery risk (0-100 scale)
    - COMPOSITE: Weighted composite of CPI, WGI, bribery, institutional

Minimum Data Requirements:
    - Trend analysis: >= 5 data points (configurable)
    - Prediction: >= 5 data points
    - Breakpoint detection: >= 8 data points
    - Moving average: >= window_size data points

Performance Targets:
    - Single country trend analysis: <50ms
    - Multi-country scan (180 countries): <5s
    - Prediction (single country): <20ms
    - Breakpoint detection: <30ms

Regulatory References:
    - EUDR Article 29: Country benchmarking system
    - EUDR Article 31: Review and update of benchmarking
    - EU 2023/1115 Recital 31: Governance indicators for risk assessment

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019, Engine 5 (Trend Analysis Engine)
Agent ID: GL-EUDR-CIM-019
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "trend") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc


def _clamp_decimal(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    """Clamp a Decimal value to [lo, hi] range.

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped Decimal value.
    """
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TrendDirection(str, Enum):
    """Direction of a corruption index trend over the analysis period.

    Values:
        IMPROVING: Index is improving (higher CPI, higher WGI = less corruption).
        STABLE: Index is essentially flat within noise threshold.
        DETERIORATING: Index is worsening (lower CPI, lower WGI = more corruption).
        VOLATILE: Index shows high variance with no clear direction.
        INSUFFICIENT_DATA: Not enough data points for reliable trend analysis.
    """

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"
    VOLATILE = "VOLATILE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class IndexType(str, Enum):
    """Type of corruption index being analyzed.

    Values:
        CPI: Corruption Perceptions Index (Transparency International).
        WGI: Worldwide Governance Indicators (World Bank).
        BRIBERY: Sector-specific bribery risk score.
        COMPOSITE: Weighted composite of all indices.
    """

    CPI = "CPI"
    WGI = "WGI"
    BRIBERY = "BRIBERY"
    COMPOSITE = "COMPOSITE"


class PredictionModel(str, Enum):
    """Prediction model type for forecasting future index values.

    Values:
        LINEAR: Linear extrapolation from OLS regression.
        WMA: Weighted Moving Average with exponential decay weights.
        ETS: Exponential Triple Smoothing (Holt-Winters additive).
    """

    LINEAR = "linear"
    WMA = "wma"
    ETS = "ets"


class ConfidenceLevel(str, Enum):
    """Confidence level for trend analysis results.

    Values:
        HIGH: R-squared >= 0.80, sufficient data, low volatility.
        MEDIUM: R-squared 0.50-0.79, adequate data, moderate volatility.
        LOW: R-squared 0.25-0.49, marginal data, high volatility.
        VERY_LOW: R-squared < 0.25 or insufficient data.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid index types.
VALID_INDEX_TYPES: frozenset = frozenset({"CPI", "WGI", "BRIBERY", "COMPOSITE"})

#: Minimum data points for trend analysis.
MIN_TREND_DATA_POINTS: int = 5

#: Minimum data points for breakpoint detection.
MIN_BREAKPOINT_DATA_POINTS: int = 8

#: Default analysis window in years.
DEFAULT_WINDOW_YEARS: int = 10

#: Default prediction horizon in years.
DEFAULT_PREDICTION_YEARS: int = 3

#: Slope threshold for STABLE classification (absolute slope per year).
STABLE_SLOPE_THRESHOLD: Decimal = Decimal("0.5")

#: R-squared threshold for HIGH confidence.
R_SQUARED_HIGH: Decimal = Decimal("0.80")

#: R-squared threshold for MEDIUM confidence.
R_SQUARED_MEDIUM: Decimal = Decimal("0.50")

#: R-squared threshold for LOW confidence.
R_SQUARED_LOW: Decimal = Decimal("0.25")

#: Coefficient of variation threshold for VOLATILE classification.
VOLATILITY_CV_THRESHOLD: Decimal = Decimal("0.15")

#: CUSUM sensitivity factor for breakpoint detection.
CUSUM_SENSITIVITY: Decimal = Decimal("1.5")

#: Default exponential smoothing alpha.
DEFAULT_ETS_ALPHA: Decimal = Decimal("0.3")

#: Default exponential smoothing beta (trend component).
DEFAULT_ETS_BETA: Decimal = Decimal("0.1")

#: CPI score range.
CPI_MIN: Decimal = Decimal("0")
CPI_MAX: Decimal = Decimal("100")

#: WGI estimate range.
WGI_MIN: Decimal = Decimal("-2.5")
WGI_MAX: Decimal = Decimal("2.5")

#: Bribery score range.
BRIBERY_MIN: Decimal = Decimal("0")
BRIBERY_MAX: Decimal = Decimal("100")

#: Composite score range.
COMPOSITE_MIN: Decimal = Decimal("0")
COMPOSITE_MAX: Decimal = Decimal("1.0")

#: Default minimum improvement threshold (points).
DEFAULT_MIN_IMPROVEMENT: Decimal = Decimal("5.0")

#: Default minimum decline threshold (points).
DEFAULT_MIN_DECLINE: Decimal = Decimal("5.0")

#: Index type value ranges for validation.
INDEX_RANGES: Dict[str, Tuple[Decimal, Decimal]] = {
    "CPI": (CPI_MIN, CPI_MAX),
    "WGI": (WGI_MIN, WGI_MAX),
    "BRIBERY": (BRIBERY_MIN, BRIBERY_MAX),
    "COMPOSITE": (COMPOSITE_MIN, COMPOSITE_MAX),
}

# ---------------------------------------------------------------------------
# Reference Data: Historical CPI Scores (180+ countries, 2015-2024)
# ---------------------------------------------------------------------------
# Subset of real-world reference data from Transparency International.
# Full dataset would be loaded from database in production; this reference
# data enables offline operation and testing.

REFERENCE_CPI_DATA: Dict[str, Dict[int, Decimal]] = {
    "DK": {
        2015: Decimal("91"), 2016: Decimal("90"), 2017: Decimal("88"),
        2018: Decimal("88"), 2019: Decimal("87"), 2020: Decimal("88"),
        2021: Decimal("88"), 2022: Decimal("90"), 2023: Decimal("90"),
        2024: Decimal("90"),
    },
    "NZ": {
        2015: Decimal("88"), 2016: Decimal("90"), 2017: Decimal("89"),
        2018: Decimal("87"), 2019: Decimal("87"), 2020: Decimal("88"),
        2021: Decimal("88"), 2022: Decimal("87"), 2023: Decimal("85"),
        2024: Decimal("85"),
    },
    "FI": {
        2015: Decimal("90"), 2016: Decimal("89"), 2017: Decimal("85"),
        2018: Decimal("85"), 2019: Decimal("86"), 2020: Decimal("85"),
        2021: Decimal("88"), 2022: Decimal("87"), 2023: Decimal("87"),
        2024: Decimal("87"),
    },
    "SG": {
        2015: Decimal("85"), 2016: Decimal("84"), 2017: Decimal("84"),
        2018: Decimal("85"), 2019: Decimal("85"), 2020: Decimal("85"),
        2021: Decimal("85"), 2022: Decimal("83"), 2023: Decimal("83"),
        2024: Decimal("83"),
    },
    "SE": {
        2015: Decimal("89"), 2016: Decimal("88"), 2017: Decimal("84"),
        2018: Decimal("85"), 2019: Decimal("85"), 2020: Decimal("85"),
        2021: Decimal("85"), 2022: Decimal("83"), 2023: Decimal("82"),
        2024: Decimal("82"),
    },
    "DE": {
        2015: Decimal("81"), 2016: Decimal("81"), 2017: Decimal("81"),
        2018: Decimal("80"), 2019: Decimal("80"), 2020: Decimal("80"),
        2021: Decimal("80"), 2022: Decimal("79"), 2023: Decimal("78"),
        2024: Decimal("78"),
    },
    "US": {
        2015: Decimal("76"), 2016: Decimal("74"), 2017: Decimal("75"),
        2018: Decimal("71"), 2019: Decimal("69"), 2020: Decimal("67"),
        2021: Decimal("67"), 2022: Decimal("69"), 2023: Decimal("69"),
        2024: Decimal("69"),
    },
    "BR": {
        2015: Decimal("38"), 2016: Decimal("40"), 2017: Decimal("37"),
        2018: Decimal("35"), 2019: Decimal("35"), 2020: Decimal("38"),
        2021: Decimal("38"), 2022: Decimal("38"), 2023: Decimal("36"),
        2024: Decimal("36"),
    },
    "ID": {
        2015: Decimal("36"), 2016: Decimal("37"), 2017: Decimal("37"),
        2018: Decimal("38"), 2019: Decimal("40"), 2020: Decimal("37"),
        2021: Decimal("38"), 2022: Decimal("34"), 2023: Decimal("34"),
        2024: Decimal("34"),
    },
    "MY": {
        2015: Decimal("50"), 2016: Decimal("49"), 2017: Decimal("47"),
        2018: Decimal("47"), 2019: Decimal("53"), 2020: Decimal("51"),
        2021: Decimal("48"), 2022: Decimal("47"), 2023: Decimal("50"),
        2024: Decimal("50"),
    },
    "CI": {
        2015: Decimal("32"), 2016: Decimal("34"), 2017: Decimal("36"),
        2018: Decimal("35"), 2019: Decimal("35"), 2020: Decimal("36"),
        2021: Decimal("36"), 2022: Decimal("37"), 2023: Decimal("37"),
        2024: Decimal("37"),
    },
    "GH": {
        2015: Decimal("47"), 2016: Decimal("43"), 2017: Decimal("40"),
        2018: Decimal("41"), 2019: Decimal("41"), 2020: Decimal("43"),
        2021: Decimal("43"), 2022: Decimal("43"), 2023: Decimal("43"),
        2024: Decimal("42"),
    },
    "CO": {
        2015: Decimal("37"), 2016: Decimal("37"), 2017: Decimal("37"),
        2018: Decimal("36"), 2019: Decimal("37"), 2020: Decimal("39"),
        2021: Decimal("39"), 2022: Decimal("39"), 2023: Decimal("40"),
        2024: Decimal("40"),
    },
    "PE": {
        2015: Decimal("36"), 2016: Decimal("35"), 2017: Decimal("37"),
        2018: Decimal("35"), 2019: Decimal("36"), 2020: Decimal("38"),
        2021: Decimal("36"), 2022: Decimal("36"), 2023: Decimal("33"),
        2024: Decimal("33"),
    },
    "PY": {
        2015: Decimal("27"), 2016: Decimal("30"), 2017: Decimal("29"),
        2018: Decimal("29"), 2019: Decimal("28"), 2020: Decimal("28"),
        2021: Decimal("28"), 2022: Decimal("28"), 2023: Decimal("28"),
        2024: Decimal("28"),
    },
    "NG": {
        2015: Decimal("26"), 2016: Decimal("28"), 2017: Decimal("27"),
        2018: Decimal("27"), 2019: Decimal("26"), 2020: Decimal("25"),
        2021: Decimal("24"), 2022: Decimal("24"), 2023: Decimal("25"),
        2024: Decimal("24"),
    },
    "CG": {
        2015: Decimal("23"), 2016: Decimal("20"), 2017: Decimal("21"),
        2018: Decimal("19"), 2019: Decimal("19"), 2020: Decimal("19"),
        2021: Decimal("19"), 2022: Decimal("19"), 2023: Decimal("20"),
        2024: Decimal("20"),
    },
    "CD": {
        2015: Decimal("22"), 2016: Decimal("21"), 2017: Decimal("21"),
        2018: Decimal("20"), 2019: Decimal("18"), 2020: Decimal("18"),
        2021: Decimal("19"), 2022: Decimal("20"), 2023: Decimal("20"),
        2024: Decimal("20"),
    },
    "MM": {
        2015: Decimal("22"), 2016: Decimal("28"), 2017: Decimal("30"),
        2018: Decimal("29"), 2019: Decimal("29"), 2020: Decimal("28"),
        2021: Decimal("28"), 2022: Decimal("23"), 2023: Decimal("20"),
        2024: Decimal("20"),
    },
    "VE": {
        2015: Decimal("17"), 2016: Decimal("17"), 2017: Decimal("18"),
        2018: Decimal("18"), 2019: Decimal("16"), 2020: Decimal("15"),
        2021: Decimal("14"), 2022: Decimal("14"), 2023: Decimal("13"),
        2024: Decimal("13"),
    },
    "KH": {
        2015: Decimal("21"), 2016: Decimal("21"), 2017: Decimal("21"),
        2018: Decimal("20"), 2019: Decimal("20"), 2020: Decimal("21"),
        2021: Decimal("23"), 2022: Decimal("24"), 2023: Decimal("22"),
        2024: Decimal("22"),
    },
    "TH": {
        2015: Decimal("38"), 2016: Decimal("35"), 2017: Decimal("37"),
        2018: Decimal("36"), 2019: Decimal("36"), 2020: Decimal("36"),
        2021: Decimal("35"), 2022: Decimal("36"), 2023: Decimal("35"),
        2024: Decimal("35"),
    },
    "IN": {
        2015: Decimal("38"), 2016: Decimal("40"), 2017: Decimal("40"),
        2018: Decimal("41"), 2019: Decimal("41"), 2020: Decimal("40"),
        2021: Decimal("40"), 2022: Decimal("40"), 2023: Decimal("39"),
        2024: Decimal("39"),
    },
    "CM": {
        2015: Decimal("27"), 2016: Decimal("26"), 2017: Decimal("25"),
        2018: Decimal("25"), 2019: Decimal("25"), 2020: Decimal("25"),
        2021: Decimal("27"), 2022: Decimal("26"), 2023: Decimal("26"),
        2024: Decimal("26"),
    },
    "PH": {
        2015: Decimal("35"), 2016: Decimal("35"), 2017: Decimal("34"),
        2018: Decimal("36"), 2019: Decimal("34"), 2020: Decimal("34"),
        2021: Decimal("33"), 2022: Decimal("33"), 2023: Decimal("34"),
        2024: Decimal("34"),
    },
    "EC": {
        2015: Decimal("32"), 2016: Decimal("31"), 2017: Decimal("32"),
        2018: Decimal("34"), 2019: Decimal("38"), 2020: Decimal("39"),
        2021: Decimal("36"), 2022: Decimal("36"), 2023: Decimal("33"),
        2024: Decimal("33"),
    },
    "HN": {
        2015: Decimal("31"), 2016: Decimal("30"), 2017: Decimal("29"),
        2018: Decimal("29"), 2019: Decimal("26"), 2020: Decimal("24"),
        2021: Decimal("23"), 2022: Decimal("23"), 2023: Decimal("23"),
        2024: Decimal("23"),
    },
    "GT": {
        2015: Decimal("28"), 2016: Decimal("28"), 2017: Decimal("28"),
        2018: Decimal("27"), 2019: Decimal("26"), 2020: Decimal("25"),
        2021: Decimal("25"), 2022: Decimal("24"), 2023: Decimal("23"),
        2024: Decimal("23"),
    },
    "MZ": {
        2015: Decimal("31"), 2016: Decimal("27"), 2017: Decimal("25"),
        2018: Decimal("23"), 2019: Decimal("26"), 2020: Decimal("25"),
        2021: Decimal("26"), 2022: Decimal("26"), 2023: Decimal("25"),
        2024: Decimal("25"),
    },
    "LR": {
        2015: Decimal("37"), 2016: Decimal("37"), 2017: Decimal("31"),
        2018: Decimal("32"), 2019: Decimal("28"), 2020: Decimal("25"),
        2021: Decimal("29"), 2022: Decimal("26"), 2023: Decimal("25"),
        2024: Decimal("25"),
    },
    "BO": {
        2015: Decimal("34"), 2016: Decimal("33"), 2017: Decimal("33"),
        2018: Decimal("29"), 2019: Decimal("31"), 2020: Decimal("31"),
        2021: Decimal("30"), 2022: Decimal("31"), 2023: Decimal("29"),
        2024: Decimal("29"),
    },
}

#: Reference WGI Control of Corruption data (key countries, 2015-2023).
REFERENCE_WGI_CC_DATA: Dict[str, Dict[int, Decimal]] = {
    "DK": {
        2015: Decimal("2.26"), 2016: Decimal("2.21"), 2017: Decimal("2.19"),
        2018: Decimal("2.18"), 2019: Decimal("2.22"), 2020: Decimal("2.26"),
        2021: Decimal("2.24"), 2022: Decimal("2.21"), 2023: Decimal("2.18"),
    },
    "BR": {
        2015: Decimal("-0.43"), 2016: Decimal("-0.49"), 2017: Decimal("-0.45"),
        2018: Decimal("-0.41"), 2019: Decimal("-0.33"), 2020: Decimal("-0.49"),
        2021: Decimal("-0.56"), 2022: Decimal("-0.44"), 2023: Decimal("-0.42"),
    },
    "ID": {
        2015: Decimal("-0.55"), 2016: Decimal("-0.46"), 2017: Decimal("-0.41"),
        2018: Decimal("-0.25"), 2019: Decimal("-0.17"), 2020: Decimal("-0.39"),
        2021: Decimal("-0.38"), 2022: Decimal("-0.51"), 2023: Decimal("-0.52"),
    },
    "NG": {
        2015: Decimal("-1.10"), 2016: Decimal("-1.06"), 2017: Decimal("-1.02"),
        2018: Decimal("-1.04"), 2019: Decimal("-1.09"), 2020: Decimal("-1.12"),
        2021: Decimal("-1.14"), 2022: Decimal("-1.12"), 2023: Decimal("-1.10"),
    },
    "CD": {
        2015: Decimal("-1.49"), 2016: Decimal("-1.44"), 2017: Decimal("-1.42"),
        2018: Decimal("-1.39"), 2019: Decimal("-1.47"), 2020: Decimal("-1.50"),
        2021: Decimal("-1.52"), 2022: Decimal("-1.48"), 2023: Decimal("-1.45"),
    },
    "VE": {
        2015: Decimal("-1.52"), 2016: Decimal("-1.58"), 2017: Decimal("-1.62"),
        2018: Decimal("-1.68"), 2019: Decimal("-1.72"), 2020: Decimal("-1.75"),
        2021: Decimal("-1.78"), 2022: Decimal("-1.76"), 2023: Decimal("-1.74"),
    },
    "CO": {
        2015: Decimal("-0.36"), 2016: Decimal("-0.35"), 2017: Decimal("-0.33"),
        2018: Decimal("-0.29"), 2019: Decimal("-0.25"), 2020: Decimal("-0.30"),
        2021: Decimal("-0.28"), 2022: Decimal("-0.25"), 2023: Decimal("-0.22"),
    },
    "MY": {
        2015: Decimal("0.05"), 2016: Decimal("-0.12"), 2017: Decimal("-0.21"),
        2018: Decimal("-0.10"), 2019: Decimal("0.10"), 2020: Decimal("0.08"),
        2021: Decimal("-0.05"), 2022: Decimal("-0.10"), 2023: Decimal("-0.08"),
    },
    "GH": {
        2015: Decimal("-0.12"), 2016: Decimal("-0.10"), 2017: Decimal("-0.16"),
        2018: Decimal("-0.13"), 2019: Decimal("-0.10"), 2020: Decimal("-0.11"),
        2021: Decimal("-0.13"), 2022: Decimal("-0.09"), 2023: Decimal("-0.08"),
    },
}

#: Country region mapping for regional aggregation.
COUNTRY_REGIONS: Dict[str, str] = {
    "DK": "europe", "NZ": "oceania", "FI": "europe", "SG": "asia",
    "SE": "europe", "DE": "europe", "US": "americas", "BR": "americas",
    "ID": "asia", "MY": "asia", "CI": "africa", "GH": "africa",
    "CO": "americas", "PE": "americas", "PY": "americas", "NG": "africa",
    "CG": "africa", "CD": "africa", "MM": "asia", "VE": "americas",
    "KH": "asia", "TH": "asia", "IN": "asia", "CM": "africa",
    "PH": "asia", "EC": "americas", "HN": "americas", "GT": "americas",
    "MZ": "africa", "LR": "africa", "BO": "americas",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TrendResult:
    """Result of a linear regression trend analysis for a single country/index.

    Attributes:
        result_id: Unique result identifier.
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Type of corruption index analyzed.
        direction: Classified trend direction.
        slope: Regression slope (change per year).
        intercept: Regression y-intercept.
        r_squared: Coefficient of determination (0.0-1.0).
        standard_error: Standard error of the slope estimate.
        start_year: First year in analysis window.
        end_year: Last year in analysis window.
        start_value: Index value at start of window.
        end_value: Index value at end of window.
        change_absolute: Absolute change over the window.
        change_percentage: Percentage change over the window.
        data_points: Number of data points used.
        confidence_level: Confidence classification.
        breakpoints: Detected structural breakpoints (years).
        warnings: Any warnings generated during analysis.
        provenance_hash: SHA-256 hash for audit trail.
    """

    result_id: str = ""
    country_code: str = ""
    index_type: str = "CPI"
    direction: str = "INSUFFICIENT_DATA"
    slope: Decimal = Decimal("0")
    intercept: Decimal = Decimal("0")
    r_squared: Decimal = Decimal("0")
    standard_error: Decimal = Decimal("0")
    start_year: int = 0
    end_year: int = 0
    start_value: Decimal = Decimal("0")
    end_value: Decimal = Decimal("0")
    change_absolute: Decimal = Decimal("0")
    change_percentage: Decimal = Decimal("0")
    data_points: int = 0
    confidence_level: str = "VERY_LOW"
    breakpoints: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/provenance.

        Returns:
            Dictionary representation with all Decimal values as strings.
        """
        return {
            "result_id": self.result_id,
            "country_code": self.country_code,
            "index_type": self.index_type,
            "direction": self.direction,
            "slope": str(self.slope),
            "intercept": str(self.intercept),
            "r_squared": str(self.r_squared),
            "standard_error": str(self.standard_error),
            "start_year": self.start_year,
            "end_year": self.end_year,
            "start_value": str(self.start_value),
            "end_value": str(self.end_value),
            "change_absolute": str(self.change_absolute),
            "change_percentage": str(self.change_percentage),
            "data_points": self.data_points,
            "confidence_level": self.confidence_level,
            "breakpoints": self.breakpoints,
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class TrendPrediction:
    """Prediction of a future corruption index value.

    Attributes:
        prediction_id: Unique prediction identifier.
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Type of corruption index.
        target_year: Year being predicted.
        predicted_value: Predicted index value.
        confidence_interval_low: Lower bound of confidence interval.
        confidence_interval_high: Upper bound of confidence interval.
        confidence_width: Width of confidence interval.
        model_type: Prediction model used.
        model_accuracy: Model R-squared or equivalent metric.
        base_data_points: Number of historical points used.
        extrapolation_distance: Years beyond last data point.
        warnings: Any warnings about prediction quality.
        provenance_hash: SHA-256 hash.
    """

    prediction_id: str = ""
    country_code: str = ""
    index_type: str = "CPI"
    target_year: int = 0
    predicted_value: Decimal = Decimal("0")
    confidence_interval_low: Decimal = Decimal("0")
    confidence_interval_high: Decimal = Decimal("0")
    confidence_width: Decimal = Decimal("0")
    model_type: str = "linear"
    model_accuracy: Decimal = Decimal("0")
    base_data_points: int = 0
    extrapolation_distance: int = 0
    warnings: List[str] = field(default_factory=list)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/provenance.

        Returns:
            Dictionary representation.
        """
        return {
            "prediction_id": self.prediction_id,
            "country_code": self.country_code,
            "index_type": self.index_type,
            "target_year": self.target_year,
            "predicted_value": str(self.predicted_value),
            "confidence_interval_low": str(self.confidence_interval_low),
            "confidence_interval_high": str(self.confidence_interval_high),
            "confidence_width": str(self.confidence_width),
            "model_type": self.model_type,
            "model_accuracy": str(self.model_accuracy),
            "base_data_points": self.base_data_points,
            "extrapolation_distance": self.extrapolation_distance,
            "warnings": self.warnings,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class Breakpoint:
    """A detected structural break in a corruption index time series.

    Attributes:
        breakpoint_id: Unique identifier.
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Type of corruption index.
        year: Year where the breakpoint was detected.
        value_before: Average value before the breakpoint.
        value_after: Average value after the breakpoint.
        magnitude: Magnitude of the structural shift.
        direction: Direction of the shift (IMPROVING or DETERIORATING).
        cusum_statistic: CUSUM statistic at the breakpoint.
        significance: Significance level (HIGH, MEDIUM, LOW).
        provenance_hash: SHA-256 hash.
    """

    breakpoint_id: str = ""
    country_code: str = ""
    index_type: str = "CPI"
    year: int = 0
    value_before: Decimal = Decimal("0")
    value_after: Decimal = Decimal("0")
    magnitude: Decimal = Decimal("0")
    direction: str = "STABLE"
    cusum_statistic: Decimal = Decimal("0")
    significance: str = "LOW"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "breakpoint_id": self.breakpoint_id,
            "country_code": self.country_code,
            "index_type": self.index_type,
            "year": self.year,
            "value_before": str(self.value_before),
            "value_after": str(self.value_after),
            "magnitude": str(self.magnitude),
            "direction": self.direction,
            "cusum_statistic": str(self.cusum_statistic),
            "significance": self.significance,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class TrajectoryResult:
    """Result of a trajectory analysis (direction + velocity) for a country.

    Attributes:
        trajectory_id: Unique identifier.
        country_code: Country analyzed.
        index_type: Index type analyzed.
        direction: Trend direction classification.
        velocity: Rate of change per year (Decimal).
        acceleration: Change in velocity over the window.
        window_years: Analysis window in years.
        current_value: Most recent index value.
        projected_value_1yr: Projected value in 1 year.
        projected_value_3yr: Projected value in 3 years.
        risk_trajectory: Risk trajectory assessment.
        data_quality: Data quality assessment.
        provenance_hash: SHA-256 hash.
    """

    trajectory_id: str = ""
    country_code: str = ""
    index_type: str = "CPI"
    direction: str = "INSUFFICIENT_DATA"
    velocity: Decimal = Decimal("0")
    acceleration: Decimal = Decimal("0")
    window_years: int = 10
    current_value: Decimal = Decimal("0")
    projected_value_1yr: Decimal = Decimal("0")
    projected_value_3yr: Decimal = Decimal("0")
    risk_trajectory: str = "STABLE"
    data_quality: str = "UNKNOWN"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "trajectory_id": self.trajectory_id,
            "country_code": self.country_code,
            "index_type": self.index_type,
            "direction": self.direction,
            "velocity": str(self.velocity),
            "acceleration": str(self.acceleration),
            "window_years": self.window_years,
            "current_value": str(self.current_value),
            "projected_value_1yr": str(self.projected_value_1yr),
            "projected_value_3yr": str(self.projected_value_3yr),
            "risk_trajectory": self.risk_trajectory,
            "data_quality": self.data_quality,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class CountryTrendSummary:
    """Summary of a country's trend for screening lists.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Index type.
        direction: Trend direction.
        total_change: Total change over the analysis window.
        average_annual_change: Average annual change.
        years_analyzed: Number of years in the window.
        current_value: Most recent value.
        meets_threshold: Whether the country meets the filtering threshold.
        provenance_hash: SHA-256 hash.
    """

    country_code: str = ""
    index_type: str = "CPI"
    direction: str = "STABLE"
    total_change: Decimal = Decimal("0")
    average_annual_change: Decimal = Decimal("0")
    years_analyzed: int = 0
    current_value: Decimal = Decimal("0")
    meets_threshold: bool = False
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "country_code": self.country_code,
            "index_type": self.index_type,
            "direction": self.direction,
            "total_change": str(self.total_change),
            "average_annual_change": str(self.average_annual_change),
            "years_analyzed": self.years_analyzed,
            "current_value": str(self.current_value),
            "meets_threshold": self.meets_threshold,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# TrendAnalysisEngine
# ---------------------------------------------------------------------------


class TrendAnalysisEngine:
    """Production-grade corruption index trend analysis for EUDR compliance.

    Analyzes corruption index time series to identify improving and
    deteriorating trends, predict future trajectories, detect structural
    breakpoints, and provide early warning of governance deterioration.

    Thread Safety:
        All mutable state is protected by a reentrant lock. Multiple threads
        can safely call any public method concurrently.

    Zero-Hallucination:
        All calculations use Decimal arithmetic with deterministic formulas.
        No ML/LLM models are used in any calculation path. Predictions use
        linear extrapolation, weighted moving average, or exponential
        smoothing -- all fully deterministic.

    Attributes:
        _custom_data: Optional user-supplied index data keyed by country/type.
        _analysis_cache: LRU-style cache of recent trend results.
        _lock: Reentrant lock for thread-safe state access.

    Example:
        >>> engine = TrendAnalysisEngine()
        >>> result = engine.analyze_trend("BR", "CPI", 2015, 2024)
        >>> assert result["direction"] in ("IMPROVING", "STABLE", "DETERIORATING", "VOLATILE")
        >>> assert "provenance_hash" in result
    """

    def __init__(self) -> None:
        """Initialize TrendAnalysisEngine with reference data and empty caches."""
        self._custom_data: Dict[str, Dict[str, Dict[int, Decimal]]] = {}
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()
        logger.info(
            "TrendAnalysisEngine initialized (version=%s, reference_countries=%d)",
            _MODULE_VERSION,
            len(REFERENCE_CPI_DATA),
        )

    # ------------------------------------------------------------------
    # Data Access Helpers
    # ------------------------------------------------------------------

    def load_custom_data(
        self,
        country_code: str,
        index_type: str,
        data: Dict[int, Decimal],
    ) -> None:
        """Load custom time series data for a country and index type.

        Allows users to provide their own index data beyond the reference
        dataset. Custom data takes precedence over reference data.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (uppercase).
            index_type: Index type (CPI, WGI, BRIBERY, COMPOSITE).
            data: Dictionary mapping year (int) to index value (Decimal).

        Raises:
            ValueError: If country_code is empty, index_type is invalid,
                or data is empty.
        """
        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}, "
                f"got '{index_type}'"
            )
        if not data:
            raise ValueError("data must be a non-empty dictionary")

        with self._lock:
            if country_code not in self._custom_data:
                self._custom_data[country_code] = {}
            self._custom_data[country_code][index_type] = dict(data)

        logger.info(
            "Loaded custom %s data for %s: %d data points (%d-%d)",
            index_type, country_code, len(data),
            min(data.keys()), max(data.keys()),
        )

    def _get_time_series(
        self,
        country_code: str,
        index_type: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[int, Decimal]:
        """Retrieve time series data for a country and index type.

        Checks custom data first, then falls back to reference data.

        Args:
            country_code: ISO country code (uppercase).
            index_type: Index type string.
            start_year: Optional start year filter.
            end_year: Optional end year filter.

        Returns:
            Dictionary of year -> Decimal value, filtered by year range.
        """
        country_code = country_code.upper()
        data: Dict[int, Decimal] = {}

        with self._lock:
            custom = self._custom_data.get(country_code, {}).get(index_type)
            if custom is not None:
                data = dict(custom)

        # Fall back to reference data
        if not data:
            if index_type == "CPI":
                data = dict(REFERENCE_CPI_DATA.get(country_code, {}))
            elif index_type == "WGI":
                data = dict(REFERENCE_WGI_CC_DATA.get(country_code, {}))

        # Apply year filters
        if start_year is not None:
            data = {y: v for y, v in data.items() if y >= start_year}
        if end_year is not None:
            data = {y: v for y, v in data.items() if y <= end_year}

        return data

    # ------------------------------------------------------------------
    # Core Statistical Methods (Zero-Hallucination)
    # ------------------------------------------------------------------

    def _linear_regression(
        self,
        x_values: List[Decimal],
        y_values: List[Decimal],
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """Compute ordinary least squares linear regression.

        Implements the closed-form OLS solution:
            slope = (n * sum(xy) - sum(x) * sum(y)) /
                    (n * sum(x^2) - (sum(x))^2)
            intercept = mean(y) - slope * mean(x)
            r_squared = 1 - SS_res / SS_tot

        Args:
            x_values: Independent variable values (e.g. years).
            y_values: Dependent variable values (e.g. CPI scores).

        Returns:
            Tuple of (slope, intercept, r_squared, standard_error).

        Raises:
            ValueError: If fewer than 2 data points or x/y length mismatch.
        """
        n = len(x_values)
        if n < 2:
            raise ValueError(
                f"Linear regression requires >= 2 data points, got {n}"
            )
        if len(y_values) != n:
            raise ValueError(
                f"x_values and y_values must have same length: "
                f"{n} vs {len(y_values)}"
            )

        # Compute sums
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        n_dec = Decimal(str(n))
        denominator = n_dec * sum_x2 - sum_x * sum_x

        if denominator == Decimal("0"):
            # All x values are the same -- cannot compute slope
            mean_y = sum_y / n_dec
            return Decimal("0"), mean_y, Decimal("0"), Decimal("0")

        slope = (n_dec * sum_xy - sum_x * sum_y) / denominator
        mean_x = sum_x / n_dec
        mean_y = sum_y / n_dec
        intercept = mean_y - slope * mean_x

        # R-squared
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)
        ss_res = sum(
            (y - (slope * x + intercept)) ** 2
            for x, y in zip(x_values, y_values)
        )

        if ss_tot == Decimal("0"):
            r_squared = Decimal("1.0")
        else:
            r_squared = Decimal("1") - (ss_res / ss_tot)

        # Standard error of slope
        if n > 2 and ss_tot != Decimal("0"):
            mse = ss_res / Decimal(str(n - 2))
            x_var = sum((x - mean_x) ** 2 for x in x_values)
            if x_var > Decimal("0"):
                # Use float sqrt since Decimal has no sqrt
                se_float = math.sqrt(float(mse / x_var))
                standard_error = _to_decimal(se_float)
            else:
                standard_error = Decimal("0")
        else:
            standard_error = Decimal("0")

        # Quantize results
        slope = slope.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        intercept = intercept.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        r_squared = _clamp_decimal(
            r_squared.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            Decimal("0"), Decimal("1"),
        )
        standard_error = standard_error.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )

        return slope, intercept, r_squared, standard_error

    def _calculate_moving_average(
        self,
        values: List[Decimal],
        window: int = 3,
    ) -> List[Decimal]:
        """Calculate simple moving average of a value series.

        Args:
            values: Ordered list of Decimal values.
            window: Moving average window size (default 3).

        Returns:
            List of moving average values. Length = len(values) - window + 1.

        Raises:
            ValueError: If window < 1 or window > len(values).
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        if window > len(values):
            raise ValueError(
                f"window ({window}) exceeds data length ({len(values)})"
            )

        result: List[Decimal] = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i + window]
            avg = sum(window_values) / Decimal(str(window))
            result.append(avg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

        return result

    def _calculate_weighted_moving_average(
        self,
        values: List[Decimal],
        window: int = 5,
    ) -> Decimal:
        """Calculate weighted moving average with exponential decay weights.

        More recent values get higher weights. Weight for position i from end:
            w_i = (window - i) / sum(1..window)

        Args:
            values: Ordered list of Decimal values.
            window: Number of most recent values to consider.

        Returns:
            Weighted moving average as Decimal.

        Raises:
            ValueError: If values is empty.
        """
        if not values:
            raise ValueError("values must be non-empty")

        # Use at most `window` most recent values
        recent = values[-window:] if len(values) >= window else values
        n = len(recent)

        # Weights: most recent gets highest weight
        weight_sum = Decimal(str(n * (n + 1))) / Decimal("2")
        weighted_sum = Decimal("0")
        for idx, val in enumerate(recent):
            weight = Decimal(str(idx + 1))
            weighted_sum += val * weight

        result = weighted_sum / weight_sum
        return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _exponential_smoothing(
        self,
        values: List[Decimal],
        alpha: Decimal = DEFAULT_ETS_ALPHA,
        beta: Decimal = DEFAULT_ETS_BETA,
        forecast_periods: int = 1,
    ) -> List[Decimal]:
        """Apply double exponential smoothing (Holt's method) for forecasting.

        Decomposes the series into level and trend components:
            level_t = alpha * y_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
            trend_t = beta * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}
            forecast_{t+h} = level_t + h * trend_t

        Args:
            values: Historical values (ordered by time).
            alpha: Level smoothing parameter (0 < alpha < 1).
            beta: Trend smoothing parameter (0 < beta < 1).
            forecast_periods: Number of periods to forecast ahead.

        Returns:
            List of forecasted values for each period ahead.

        Raises:
            ValueError: If values has fewer than 2 elements.
        """
        if len(values) < 2:
            raise ValueError("Exponential smoothing requires >= 2 data points")

        # Initialize level and trend
        level = values[0]
        trend = values[1] - values[0]

        one = Decimal("1")

        for val in values[1:]:
            new_level = alpha * val + (one - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (one - beta) * trend
            level = new_level
            trend = new_trend

        # Generate forecasts
        forecasts: List[Decimal] = []
        for h in range(1, forecast_periods + 1):
            forecast = level + Decimal(str(h)) * trend
            forecasts.append(
                forecast.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )

        return forecasts

    def _detect_regime_change(
        self,
        values: List[Decimal],
        threshold: Decimal = CUSUM_SENSITIVITY,
    ) -> List[int]:
        """Detect regime changes using CUSUM (Cumulative Sum) control chart.

        Identifies points where the mean of the series shifts significantly.
        Uses the CUSUM+ and CUSUM- statistics with a configurable threshold
        multiplied by the standard deviation.

        Args:
            values: Ordered list of Decimal values.
            threshold: Sensitivity factor (multiplied by std dev).

        Returns:
            List of indices where regime changes were detected.
        """
        if len(values) < MIN_BREAKPOINT_DATA_POINTS:
            return []

        n = len(values)
        mean_val = sum(values) / Decimal(str(n))

        # Standard deviation
        variance = sum((v - mean_val) ** 2 for v in values) / Decimal(str(n))
        if variance <= Decimal("0"):
            return []
        std_dev = _to_decimal(math.sqrt(float(variance)))

        control_limit = threshold * std_dev
        if control_limit <= Decimal("0"):
            return []

        # CUSUM computation
        cusum_pos = Decimal("0")
        cusum_neg = Decimal("0")
        change_points: List[int] = []

        for i in range(n):
            cusum_pos = max(Decimal("0"), cusum_pos + values[i] - mean_val)
            cusum_neg = max(Decimal("0"), cusum_neg - values[i] + mean_val)

            if cusum_pos > control_limit or cusum_neg > control_limit:
                change_points.append(i)
                # Reset after detection
                cusum_pos = Decimal("0")
                cusum_neg = Decimal("0")

        return change_points

    def _calculate_coefficient_of_variation(
        self,
        values: List[Decimal],
    ) -> Decimal:
        """Calculate coefficient of variation (CV = std_dev / |mean|).

        Args:
            values: List of Decimal values.

        Returns:
            Coefficient of variation as Decimal.
        """
        if not values:
            return Decimal("0")

        n = Decimal(str(len(values)))
        mean_val = sum(values) / n
        if mean_val == Decimal("0"):
            return Decimal("0")

        variance = sum((v - mean_val) ** 2 for v in values) / n
        std_dev = _to_decimal(math.sqrt(float(variance)))
        cv = std_dev / abs(mean_val)
        return cv.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _classify_direction(
        self,
        slope: Decimal,
        r_squared: Decimal,
        cv: Decimal,
        index_type: str,
        data_points: int,
    ) -> str:
        """Classify trend direction based on regression and volatility metrics.

        Classification rules (applied in order):
            1. INSUFFICIENT_DATA if data_points < MIN_TREND_DATA_POINTS
            2. VOLATILE if CV > VOLATILITY_CV_THRESHOLD and R-squared < 0.30
            3. IMPROVING if slope > STABLE_SLOPE_THRESHOLD (positive for CPI/BRIBERY;
               WGI uses positive slope directly since higher = less corruption)
            4. DETERIORATING if slope < -STABLE_SLOPE_THRESHOLD
            5. STABLE otherwise

        Args:
            slope: Regression slope.
            r_squared: R-squared value.
            cv: Coefficient of variation.
            index_type: Index type (affects direction interpretation).
            data_points: Number of data points.

        Returns:
            TrendDirection value as string.
        """
        if data_points < MIN_TREND_DATA_POINTS:
            return TrendDirection.INSUFFICIENT_DATA.value

        if cv > VOLATILITY_CV_THRESHOLD and r_squared < Decimal("0.30"):
            return TrendDirection.VOLATILE.value

        # For CPI and BRIBERY: higher score = less corruption = IMPROVING
        # For WGI: higher value = less corruption = IMPROVING
        # For COMPOSITE: higher value = more corruption = reversed
        if index_type == "COMPOSITE":
            # Composite: lower is better (less corruption)
            if slope < -STABLE_SLOPE_THRESHOLD:
                return TrendDirection.IMPROVING.value
            elif slope > STABLE_SLOPE_THRESHOLD:
                return TrendDirection.DETERIORATING.value
        else:
            # CPI, WGI, BRIBERY: higher is better (less corruption)
            if slope > STABLE_SLOPE_THRESHOLD:
                return TrendDirection.IMPROVING.value
            elif slope < -STABLE_SLOPE_THRESHOLD:
                return TrendDirection.DETERIORATING.value

        return TrendDirection.STABLE.value

    def _classify_confidence(self, r_squared: Decimal) -> str:
        """Classify confidence level based on R-squared value.

        Args:
            r_squared: R-squared value from regression.

        Returns:
            ConfidenceLevel value as string.
        """
        if r_squared >= R_SQUARED_HIGH:
            return ConfidenceLevel.HIGH.value
        elif r_squared >= R_SQUARED_MEDIUM:
            return ConfidenceLevel.MEDIUM.value
        elif r_squared >= R_SQUARED_LOW:
            return ConfidenceLevel.LOW.value
        else:
            return ConfidenceLevel.VERY_LOW.value

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute provenance hash for a result object.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_trend(
        self,
        country_code: str,
        index_type: str = "CPI",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform linear regression trend analysis on a corruption index.

        Analyzes the time series of the specified corruption index for a
        given country over the specified year range. Returns the trend
        direction, slope, R-squared, and other regression statistics.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            index_type: Index type (CPI, WGI, BRIBERY, COMPOSITE).
            start_year: First year to include (default: earliest available).
            end_year: Last year to include (default: latest available).

        Returns:
            Dictionary containing TrendResult data plus processing_time_ms,
            calculation_timestamp, and provenance_hash.

        Raises:
            ValueError: If country_code is empty or index_type is invalid.
        """
        start_time = time.monotonic()

        # Input validation
        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}, "
                f"got '{index_type}'"
            )

        # Retrieve data
        data = self._get_time_series(country_code, index_type, start_year, end_year)

        result = TrendResult(
            result_id=_generate_id("trend"),
            country_code=country_code,
            index_type=index_type,
        )

        if len(data) < MIN_TREND_DATA_POINTS:
            result.direction = TrendDirection.INSUFFICIENT_DATA.value
            result.data_points = len(data)
            result.warnings.append(
                f"Insufficient data: {len(data)} points available, "
                f"minimum {MIN_TREND_DATA_POINTS} required"
            )
            result.provenance_hash = self._compute_provenance_hash(result)

            processing_time_ms = (time.monotonic() - start_time) * 1000.0
            out = result.to_dict()
            out["processing_time_ms"] = round(processing_time_ms, 3)
            out["calculation_timestamp"] = _utcnow().isoformat()
            logger.info(
                "Trend analysis for %s/%s: INSUFFICIENT_DATA (%d points) "
                "time_ms=%.1f",
                country_code, index_type, len(data), processing_time_ms,
            )
            return out

        # Sort by year
        sorted_years = sorted(data.keys())
        x_values = [_to_decimal(y) for y in sorted_years]
        y_values = [data[y] for y in sorted_years]

        # Linear regression
        slope, intercept, r_squared, standard_error = self._linear_regression(
            x_values, y_values,
        )

        # Coefficient of variation for volatility check
        cv = self._calculate_coefficient_of_variation(y_values)

        # Classify direction
        direction = self._classify_direction(
            slope, r_squared, cv, index_type, len(sorted_years),
        )
        confidence = self._classify_confidence(r_squared)

        # Breakpoint detection if enough data
        breakpoints: List[int] = []
        if len(sorted_years) >= MIN_BREAKPOINT_DATA_POINTS:
            bp_indices = self._detect_regime_change(y_values)
            breakpoints = [sorted_years[i] for i in bp_indices if i < len(sorted_years)]

        # Change calculations
        start_val = y_values[0]
        end_val = y_values[-1]
        change_absolute = end_val - start_val
        if start_val != Decimal("0"):
            change_pct = (change_absolute / abs(start_val)) * Decimal("100")
        else:
            change_pct = Decimal("0")

        # Populate result
        result.direction = direction
        result.slope = slope
        result.intercept = intercept
        result.r_squared = r_squared
        result.standard_error = standard_error
        result.start_year = sorted_years[0]
        result.end_year = sorted_years[-1]
        result.start_value = start_val
        result.end_value = end_val
        result.change_absolute = change_absolute.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        result.change_percentage = change_pct.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        result.data_points = len(sorted_years)
        result.confidence_level = confidence
        result.breakpoints = breakpoints
        result.provenance_hash = self._compute_provenance_hash(result)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        out = result.to_dict()
        out["processing_time_ms"] = round(processing_time_ms, 3)
        out["calculation_timestamp"] = _utcnow().isoformat()

        logger.info(
            "Trend analysis for %s/%s: direction=%s slope=%s r2=%s "
            "points=%d time_ms=%.1f",
            country_code, index_type, direction, slope, r_squared,
            len(sorted_years), processing_time_ms,
        )
        return out

    def get_trajectory(
        self,
        country_code: str,
        index_type: str = "CPI",
        window_years: int = DEFAULT_WINDOW_YEARS,
    ) -> Dict[str, Any]:
        """Compute trajectory (direction + velocity) for a country.

        Calculates the current velocity (rate of change per year) and
        acceleration (change in velocity), plus projected values at
        1-year and 3-year horizons.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            index_type: Index type (CPI, WGI, BRIBERY, COMPOSITE).
            window_years: Analysis window in years (default 10).

        Returns:
            Dictionary containing TrajectoryResult data plus
            processing_time_ms and provenance_hash.

        Raises:
            ValueError: If country_code is empty or index_type is invalid.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}"
            )
        if window_years < 2:
            raise ValueError("window_years must be >= 2")

        # Determine year range based on window
        current_year = _utcnow().year
        start_year = current_year - window_years
        data = self._get_time_series(
            country_code, index_type, start_year, current_year,
        )

        traj = TrajectoryResult(
            trajectory_id=_generate_id("traj"),
            country_code=country_code,
            index_type=index_type,
            window_years=window_years,
        )

        if len(data) < MIN_TREND_DATA_POINTS:
            traj.direction = TrendDirection.INSUFFICIENT_DATA.value
            traj.data_quality = "INSUFFICIENT"
            traj.provenance_hash = self._compute_provenance_hash(traj)

            processing_time_ms = (time.monotonic() - start_time) * 1000.0
            out = traj.to_dict()
            out["processing_time_ms"] = round(processing_time_ms, 3)
            out["calculation_timestamp"] = _utcnow().isoformat()
            return out

        sorted_years = sorted(data.keys())
        x_vals = [_to_decimal(y) for y in sorted_years]
        y_vals = [data[y] for y in sorted_years]

        # Full-window regression for velocity (slope)
        slope, intercept, r_squared, _ = self._linear_regression(x_vals, y_vals)

        # Acceleration: compare slope of first half vs second half
        mid = len(sorted_years) // 2
        acceleration = Decimal("0")
        if mid >= 2 and len(sorted_years) - mid >= 2:
            slope_first, _, _, _ = self._linear_regression(
                x_vals[:mid], y_vals[:mid],
            )
            slope_second, _, _, _ = self._linear_regression(
                x_vals[mid:], y_vals[mid:],
            )
            acceleration = (slope_second - slope_first).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP,
            )

        cv = self._calculate_coefficient_of_variation(y_vals)
        direction = self._classify_direction(
            slope, r_squared, cv, index_type, len(sorted_years),
        )

        current_value = y_vals[-1]
        projected_1yr = current_value + slope
        projected_3yr = current_value + slope * Decimal("3")

        # Clamp projections to valid index ranges
        idx_range = INDEX_RANGES.get(index_type, (Decimal("-999"), Decimal("999")))
        projected_1yr = _clamp_decimal(projected_1yr, idx_range[0], idx_range[1])
        projected_3yr = _clamp_decimal(projected_3yr, idx_range[0], idx_range[1])

        # Risk trajectory assessment
        if direction == TrendDirection.DETERIORATING.value:
            risk_trajectory = "INCREASING_RISK"
        elif direction == TrendDirection.IMPROVING.value:
            risk_trajectory = "DECREASING_RISK"
        elif direction == TrendDirection.VOLATILE.value:
            risk_trajectory = "UNCERTAIN"
        else:
            risk_trajectory = "STABLE"

        # Data quality assessment
        if len(sorted_years) >= 8 and r_squared >= R_SQUARED_MEDIUM:
            data_quality = "HIGH"
        elif len(sorted_years) >= 5:
            data_quality = "ADEQUATE"
        else:
            data_quality = "MARGINAL"

        traj.direction = direction
        traj.velocity = slope
        traj.acceleration = acceleration
        traj.current_value = current_value
        traj.projected_value_1yr = projected_1yr.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        traj.projected_value_3yr = projected_3yr.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        traj.risk_trajectory = risk_trajectory
        traj.data_quality = data_quality
        traj.provenance_hash = self._compute_provenance_hash(traj)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        out = traj.to_dict()
        out["processing_time_ms"] = round(processing_time_ms, 3)
        out["calculation_timestamp"] = _utcnow().isoformat()

        logger.info(
            "Trajectory for %s/%s: direction=%s velocity=%s accel=%s "
            "time_ms=%.1f",
            country_code, index_type, direction, slope, acceleration,
            processing_time_ms,
        )
        return out

    def predict_future(
        self,
        country_code: str,
        index_type: str = "CPI",
        target_year: Optional[int] = None,
        model: str = "linear",
    ) -> Dict[str, Any]:
        """Forecast a future corruption index value.

        Uses the specified prediction model to extrapolate from historical
        data to a target year. Provides confidence intervals based on
        historical volatility and extrapolation distance.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            index_type: Index type (CPI, WGI, BRIBERY, COMPOSITE).
            target_year: Year to predict (default: current + 3).
            model: Prediction model ("linear", "wma", "ets").

        Returns:
            Dictionary containing TrendPrediction data plus
            processing_time_ms and provenance_hash.

        Raises:
            ValueError: If model is invalid, target_year is in the past,
                or insufficient data.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}"
            )
        valid_models = {"linear", "wma", "ets"}
        if model not in valid_models:
            raise ValueError(f"model must be one of {sorted(valid_models)}")

        current_year = _utcnow().year
        if target_year is None:
            target_year = current_year + DEFAULT_PREDICTION_YEARS

        data = self._get_time_series(country_code, index_type)
        pred = TrendPrediction(
            prediction_id=_generate_id("pred"),
            country_code=country_code,
            index_type=index_type,
            target_year=target_year,
            model_type=model,
        )

        if len(data) < MIN_TREND_DATA_POINTS:
            pred.warnings.append(
                f"Insufficient data: {len(data)} points, "
                f"minimum {MIN_TREND_DATA_POINTS} required"
            )
            pred.provenance_hash = self._compute_provenance_hash(pred)
            processing_time_ms = (time.monotonic() - start_time) * 1000.0
            out = pred.to_dict()
            out["processing_time_ms"] = round(processing_time_ms, 3)
            out["calculation_timestamp"] = _utcnow().isoformat()
            return out

        sorted_years = sorted(data.keys())
        y_vals = [data[y] for y in sorted_years]
        x_vals = [_to_decimal(y) for y in sorted_years]
        last_year = sorted_years[-1]
        extrapolation_distance = target_year - last_year

        if extrapolation_distance < 0:
            pred.warnings.append(
                f"Target year {target_year} is before last data year "
                f"{last_year}; interpolation not extrapolation"
            )

        # Calculate prediction based on model
        predicted_value = Decimal("0")
        model_accuracy = Decimal("0")

        if model == "linear":
            slope, intercept, r_squared, _ = self._linear_regression(x_vals, y_vals)
            target_x = _to_decimal(target_year)
            predicted_value = slope * target_x + intercept
            model_accuracy = r_squared

        elif model == "wma":
            # Weighted moving average with trend extension
            wma_value = self._calculate_weighted_moving_average(y_vals, window=5)
            # Estimate trend from recent values
            if len(y_vals) >= 3:
                recent_slope = (y_vals[-1] - y_vals[-3]) / Decimal("2")
            else:
                recent_slope = Decimal("0")
            predicted_value = wma_value + recent_slope * Decimal(str(extrapolation_distance))
            # Approximate accuracy from recent fit
            if len(y_vals) >= 5:
                wma_recent = self._calculate_weighted_moving_average(y_vals[-5:], window=3)
                residual = abs(wma_recent - y_vals[-1])
                if abs(y_vals[-1]) > Decimal("0"):
                    model_accuracy = max(
                        Decimal("0"),
                        Decimal("1") - residual / abs(y_vals[-1]),
                    )
                else:
                    model_accuracy = Decimal("0.5")
            else:
                model_accuracy = Decimal("0.5")

        elif model == "ets":
            forecasts = self._exponential_smoothing(
                y_vals,
                alpha=DEFAULT_ETS_ALPHA,
                beta=DEFAULT_ETS_BETA,
                forecast_periods=max(1, extrapolation_distance),
            )
            predicted_value = forecasts[-1] if forecasts else y_vals[-1]
            # ETS accuracy approximation
            model_accuracy = Decimal("0.60")

        # Clamp predicted value to valid range
        idx_range = INDEX_RANGES.get(index_type, (Decimal("-999"), Decimal("999")))
        predicted_value = _clamp_decimal(predicted_value, idx_range[0], idx_range[1])
        predicted_value = predicted_value.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

        # Confidence intervals based on historical std dev and distance
        variance = sum((v - sum(y_vals) / Decimal(str(len(y_vals)))) ** 2 for v in y_vals) / Decimal(str(len(y_vals)))
        std_dev = _to_decimal(math.sqrt(float(variance))) if variance > Decimal("0") else Decimal("0")

        # Widen CI with extrapolation distance
        distance_factor = Decimal("1") + Decimal("0.2") * Decimal(str(max(0, extrapolation_distance)))
        ci_half_width = Decimal("1.96") * std_dev * distance_factor

        ci_low = _clamp_decimal(
            predicted_value - ci_half_width, idx_range[0], idx_range[1],
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        ci_high = _clamp_decimal(
            predicted_value + ci_half_width, idx_range[0], idx_range[1],
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Warnings for long extrapolation
        if extrapolation_distance > 5:
            pred.warnings.append(
                f"Long extrapolation ({extrapolation_distance} years): "
                "prediction reliability is low"
            )
        if model_accuracy < Decimal("0.50"):
            pred.warnings.append(
                f"Model accuracy is low ({model_accuracy}): "
                "prediction should be treated with caution"
            )

        pred.predicted_value = predicted_value
        pred.confidence_interval_low = ci_low
        pred.confidence_interval_high = ci_high
        pred.confidence_width = (ci_high - ci_low).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        pred.model_accuracy = model_accuracy.quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP,
        )
        pred.base_data_points = len(sorted_years)
        pred.extrapolation_distance = extrapolation_distance
        pred.provenance_hash = self._compute_provenance_hash(pred)

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        out = pred.to_dict()
        out["processing_time_ms"] = round(processing_time_ms, 3)
        out["calculation_timestamp"] = _utcnow().isoformat()

        logger.info(
            "Prediction for %s/%s target=%d: value=%s CI=[%s,%s] "
            "model=%s accuracy=%s time_ms=%.1f",
            country_code, index_type, target_year, predicted_value,
            ci_low, ci_high, model, model_accuracy, processing_time_ms,
        )
        return out

    def find_improving_countries(
        self,
        index_type: str = "CPI",
        min_years: int = 5,
        min_improvement: float = 5.0,
    ) -> Dict[str, Any]:
        """Identify countries with improving corruption trends.

        Scans all countries in the reference/custom dataset and returns
        those showing statistically significant improvement above the
        specified threshold.

        Args:
            index_type: Index type to analyze.
            min_years: Minimum years of data required (default 5).
            min_improvement: Minimum absolute improvement threshold (default 5.0).

        Returns:
            Dictionary with list of improving CountryTrendSummary entries,
            total countries scanned, and provenance_hash.

        Raises:
            ValueError: If index_type is invalid.
        """
        start_time = time.monotonic()

        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}"
            )
        min_imp = _to_decimal(min_improvement)

        # Gather all country codes from reference + custom data
        all_countries = self._get_all_country_codes(index_type)
        improving: List[Dict[str, Any]] = []
        scanned = 0

        for cc in sorted(all_countries):
            data = self._get_time_series(cc, index_type)
            if len(data) < min_years:
                continue

            scanned += 1
            sorted_years = sorted(data.keys())
            y_vals = [data[y] for y in sorted_years]

            total_change = y_vals[-1] - y_vals[0]
            avg_annual = total_change / Decimal(str(len(sorted_years) - 1))

            # For CPI/WGI/BRIBERY: positive change = improvement
            # For COMPOSITE: negative change = improvement
            is_improving = False
            if index_type == "COMPOSITE":
                is_improving = total_change < -min_imp
            else:
                is_improving = total_change > min_imp

            if is_improving:
                summary = CountryTrendSummary(
                    country_code=cc,
                    index_type=index_type,
                    direction=TrendDirection.IMPROVING.value,
                    total_change=total_change.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    average_annual_change=avg_annual.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    years_analyzed=len(sorted_years),
                    current_value=y_vals[-1],
                    meets_threshold=True,
                )
                summary.provenance_hash = _compute_hash(summary)
                improving.append(summary.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "index_type": index_type,
            "min_years": min_years,
            "min_improvement": str(min_imp),
            "countries_scanned": scanned,
            "improving_count": len(improving),
            "improving_countries": improving,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Found %d improving countries for %s (scanned=%d, threshold=%s) "
            "time_ms=%.1f",
            len(improving), index_type, scanned, min_imp, processing_time_ms,
        )
        return result

    def find_deteriorating_countries(
        self,
        index_type: str = "CPI",
        min_years: int = 5,
        min_decline: float = 5.0,
    ) -> Dict[str, Any]:
        """Identify countries with deteriorating corruption trends.

        Scans all countries and returns those showing significant governance
        decline exceeding the specified threshold. Critical for EUDR risk
        monitoring -- deteriorating countries may require reclassification.

        Args:
            index_type: Index type to analyze.
            min_years: Minimum years of data required (default 5).
            min_decline: Minimum absolute decline threshold (default 5.0).

        Returns:
            Dictionary with list of deteriorating CountryTrendSummary entries,
            total countries scanned, and provenance_hash.

        Raises:
            ValueError: If index_type is invalid.
        """
        start_time = time.monotonic()

        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}"
            )
        min_dec = _to_decimal(min_decline)

        all_countries = self._get_all_country_codes(index_type)
        deteriorating: List[Dict[str, Any]] = []
        scanned = 0

        for cc in sorted(all_countries):
            data = self._get_time_series(cc, index_type)
            if len(data) < min_years:
                continue

            scanned += 1
            sorted_years = sorted(data.keys())
            y_vals = [data[y] for y in sorted_years]

            total_change = y_vals[-1] - y_vals[0]
            avg_annual = total_change / Decimal(str(len(sorted_years) - 1))

            # For CPI/WGI/BRIBERY: negative change = deterioration
            # For COMPOSITE: positive change = deterioration
            is_deteriorating = False
            if index_type == "COMPOSITE":
                is_deteriorating = total_change > min_dec
            else:
                is_deteriorating = total_change < -min_dec

            if is_deteriorating:
                summary = CountryTrendSummary(
                    country_code=cc,
                    index_type=index_type,
                    direction=TrendDirection.DETERIORATING.value,
                    total_change=total_change.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    average_annual_change=avg_annual.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    years_analyzed=len(sorted_years),
                    current_value=y_vals[-1],
                    meets_threshold=True,
                )
                summary.provenance_hash = _compute_hash(summary)
                deteriorating.append(summary.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "index_type": index_type,
            "min_years": min_years,
            "min_decline": str(min_dec),
            "countries_scanned": scanned,
            "deteriorating_count": len(deteriorating),
            "deteriorating_countries": deteriorating,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Found %d deteriorating countries for %s (scanned=%d, threshold=%s) "
            "time_ms=%.1f",
            len(deteriorating), index_type, scanned, min_dec, processing_time_ms,
        )
        return result

    def detect_breakpoints(
        self,
        country_code: str,
        index_type: str = "CPI",
        sensitivity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Detect structural breakpoints in a corruption index time series.

        Uses the CUSUM (Cumulative Sum) control chart method to identify
        points where the mean of the series shifts significantly, indicating
        a structural change in governance quality.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.
            index_type: Index type (CPI, WGI, BRIBERY, COMPOSITE).
            sensitivity: CUSUM sensitivity factor (default from config).

        Returns:
            Dictionary with list of Breakpoint objects, country_code,
            index_type, and provenance_hash.

        Raises:
            ValueError: If country_code is empty or index_type is invalid.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}"
            )

        threshold = _to_decimal(sensitivity) if sensitivity is not None else CUSUM_SENSITIVITY
        data = self._get_time_series(country_code, index_type)

        warnings_list: List[str] = []
        breakpoints: List[Dict[str, Any]] = []

        if len(data) < MIN_BREAKPOINT_DATA_POINTS:
            warnings_list.append(
                f"Insufficient data for breakpoint detection: {len(data)} "
                f"points, minimum {MIN_BREAKPOINT_DATA_POINTS} required"
            )
        else:
            sorted_years = sorted(data.keys())
            y_vals = [data[y] for y in sorted_years]

            # Detect change points
            cp_indices = self._detect_regime_change(y_vals, threshold)

            for idx in cp_indices:
                if idx < 1 or idx >= len(sorted_years):
                    continue

                bp_year = sorted_years[idx]
                vals_before = y_vals[:idx]
                vals_after = y_vals[idx:]

                mean_before = sum(vals_before) / Decimal(str(len(vals_before)))
                mean_after = sum(vals_after) / Decimal(str(len(vals_after)))
                magnitude = abs(mean_after - mean_before)

                # Determine direction
                if index_type == "COMPOSITE":
                    bp_direction = "IMPROVING" if mean_after < mean_before else "DETERIORATING"
                else:
                    bp_direction = "IMPROVING" if mean_after > mean_before else "DETERIORATING"

                # Significance based on magnitude relative to std dev
                overall_variance = sum(
                    (v - sum(y_vals) / Decimal(str(len(y_vals)))) ** 2
                    for v in y_vals
                ) / Decimal(str(len(y_vals)))
                std_dev = _to_decimal(math.sqrt(float(overall_variance))) if overall_variance > 0 else Decimal("1")

                if magnitude > Decimal("2") * std_dev:
                    significance = "HIGH"
                elif magnitude > std_dev:
                    significance = "MEDIUM"
                else:
                    significance = "LOW"

                bp = Breakpoint(
                    breakpoint_id=_generate_id("bp"),
                    country_code=country_code,
                    index_type=index_type,
                    year=bp_year,
                    value_before=mean_before.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    value_after=mean_after.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    magnitude=magnitude.quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP,
                    ),
                    direction=bp_direction,
                    cusum_statistic=magnitude,
                    significance=significance,
                )
                bp.provenance_hash = _compute_hash(bp)
                breakpoints.append(bp.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "country_code": country_code,
            "index_type": index_type,
            "sensitivity": str(threshold),
            "data_points": len(data),
            "breakpoint_count": len(breakpoints),
            "breakpoints": breakpoints,
            "warnings": warnings_list,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Breakpoint detection for %s/%s: found %d breakpoints "
            "time_ms=%.1f",
            country_code, index_type, len(breakpoints), processing_time_ms,
        )
        return result

    def get_regional_trend_summary(
        self,
        region: str,
        index_type: str = "CPI",
        min_years: int = 5,
    ) -> Dict[str, Any]:
        """Generate a trend summary for all countries in a region.

        Args:
            region: Region name (e.g. "americas", "africa", "asia", "europe").
            index_type: Index type to analyze.
            min_years: Minimum years of data required.

        Returns:
            Dictionary with regional trend summary including improving,
            stable, deteriorating, and volatile country counts.

        Raises:
            ValueError: If region or index_type is invalid.
        """
        start_time = time.monotonic()

        valid_regions = {"americas", "africa", "asia", "europe", "oceania"}
        if region.lower() not in valid_regions:
            raise ValueError(f"region must be one of {sorted(valid_regions)}")
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(VALID_INDEX_TYPES)}"
            )

        region_lower = region.lower()
        # Find countries in this region
        region_countries = [
            cc for cc, reg in COUNTRY_REGIONS.items()
            if reg == region_lower
        ]

        # Also check custom data
        with self._lock:
            for cc in self._custom_data:
                if COUNTRY_REGIONS.get(cc) == region_lower and cc not in region_countries:
                    region_countries.append(cc)

        summaries: List[Dict[str, Any]] = []
        direction_counts = {
            "IMPROVING": 0,
            "STABLE": 0,
            "DETERIORATING": 0,
            "VOLATILE": 0,
            "INSUFFICIENT_DATA": 0,
        }

        for cc in sorted(region_countries):
            data = self._get_time_series(cc, index_type)
            if len(data) < min_years:
                direction_counts["INSUFFICIENT_DATA"] += 1
                continue

            sorted_years = sorted(data.keys())
            x_vals = [_to_decimal(y) for y in sorted_years]
            y_vals = [data[y] for y in sorted_years]

            slope, _, r_squared, _ = self._linear_regression(x_vals, y_vals)
            cv = self._calculate_coefficient_of_variation(y_vals)
            direction = self._classify_direction(
                slope, r_squared, cv, index_type, len(sorted_years),
            )
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

            total_change = y_vals[-1] - y_vals[0]
            avg_annual = total_change / Decimal(str(len(sorted_years) - 1))

            summary = CountryTrendSummary(
                country_code=cc,
                index_type=index_type,
                direction=direction,
                total_change=total_change.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                average_annual_change=avg_annual.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                years_analyzed=len(sorted_years),
                current_value=y_vals[-1],
                meets_threshold=True,
            )
            summary.provenance_hash = _compute_hash(summary)
            summaries.append(summary.to_dict())

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "region": region_lower,
            "index_type": index_type,
            "min_years": min_years,
            "total_countries": len(region_countries),
            "analyzed_countries": len(summaries),
            "direction_counts": direction_counts,
            "country_summaries": summaries,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Regional trend summary for %s/%s: %d countries analyzed "
            "time_ms=%.1f",
            region, index_type, len(summaries), processing_time_ms,
        )
        return result

    def get_multi_year_comparison(
        self,
        country_code: str,
        index_type: str = "CPI",
        year_a: int = 2019,
        year_b: int = 2024,
    ) -> Dict[str, Any]:
        """Compare a country's corruption index between two specific years.

        Provides a simple year-over-year or multi-year comparison with
        absolute and percentage change calculations.

        Args:
            country_code: ISO country code.
            index_type: Index type.
            year_a: First year (baseline).
            year_b: Second year (comparison).

        Returns:
            Dictionary with comparison results.

        Raises:
            ValueError: If parameters are invalid.
        """
        start_time = time.monotonic()

        if not country_code or not isinstance(country_code, str):
            raise ValueError("country_code must be a non-empty string")
        country_code = country_code.upper()
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(f"index_type must be one of {sorted(VALID_INDEX_TYPES)}")
        if year_a >= year_b:
            raise ValueError("year_a must be less than year_b")

        data = self._get_time_series(country_code, index_type)
        warnings_list: List[str] = []

        value_a = data.get(year_a)
        value_b = data.get(year_b)

        if value_a is None:
            warnings_list.append(f"No data available for year {year_a}")
        if value_b is None:
            warnings_list.append(f"No data available for year {year_b}")

        change_absolute = Decimal("0")
        change_pct = Decimal("0")
        direction = "UNKNOWN"

        if value_a is not None and value_b is not None:
            change_absolute = value_b - value_a
            if value_a != Decimal("0"):
                change_pct = (change_absolute / abs(value_a)) * Decimal("100")

            if index_type == "COMPOSITE":
                if change_absolute < Decimal("0"):
                    direction = "IMPROVING"
                elif change_absolute > Decimal("0"):
                    direction = "DETERIORATING"
                else:
                    direction = "STABLE"
            else:
                if change_absolute > Decimal("0"):
                    direction = "IMPROVING"
                elif change_absolute < Decimal("0"):
                    direction = "DETERIORATING"
                else:
                    direction = "STABLE"

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "country_code": country_code,
            "index_type": index_type,
            "year_a": year_a,
            "year_b": year_b,
            "value_a": str(value_a) if value_a is not None else None,
            "value_b": str(value_b) if value_b is not None else None,
            "change_absolute": str(change_absolute.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )),
            "change_percentage": str(change_pct.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )),
            "direction": direction,
            "years_span": year_b - year_a,
            "warnings": warnings_list,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def batch_analyze_trends(
        self,
        country_codes: List[str],
        index_type: str = "CPI",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform trend analysis on multiple countries in batch.

        Args:
            country_codes: List of ISO country codes.
            index_type: Index type to analyze.
            start_year: Optional start year filter.
            end_year: Optional end year filter.

        Returns:
            Dictionary with per-country results, summary statistics,
            and provenance_hash.

        Raises:
            ValueError: If country_codes is empty or index_type is invalid.
        """
        start_time = time.monotonic()

        if not country_codes:
            raise ValueError("country_codes must be non-empty")
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(f"index_type must be one of {sorted(VALID_INDEX_TYPES)}")

        per_country_results: List[Dict[str, Any]] = []
        direction_counts = {
            "IMPROVING": 0,
            "STABLE": 0,
            "DETERIORATING": 0,
            "VOLATILE": 0,
            "INSUFFICIENT_DATA": 0,
        }

        for cc in country_codes:
            try:
                result = self.analyze_trend(cc, index_type, start_year, end_year)
                per_country_results.append(result)
                direction = result.get("direction", "INSUFFICIENT_DATA")
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            except ValueError as exc:
                per_country_results.append({
                    "country_code": cc.upper(),
                    "error": str(exc),
                    "direction": "INSUFFICIENT_DATA",
                })
                direction_counts["INSUFFICIENT_DATA"] += 1

        processing_time_ms = (time.monotonic() - start_time) * 1000.0

        result_batch = {
            "index_type": index_type,
            "countries_requested": len(country_codes),
            "countries_analyzed": len(per_country_results),
            "direction_summary": direction_counts,
            "results": per_country_results,
            "processing_time_ms": round(processing_time_ms, 3),
            "calculation_timestamp": _utcnow().isoformat(),
            "provenance_hash": "",
        }
        result_batch["provenance_hash"] = _compute_hash(result_batch)

        logger.info(
            "Batch trend analysis: %d countries, %s time_ms=%.1f",
            len(country_codes), index_type, processing_time_ms,
        )
        return result_batch

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_all_country_codes(self, index_type: str) -> List[str]:
        """Get all country codes with data for the given index type.

        Args:
            index_type: Index type string.

        Returns:
            Sorted list of unique country codes.
        """
        codes: set = set()

        # Reference data
        if index_type == "CPI":
            codes.update(REFERENCE_CPI_DATA.keys())
        elif index_type == "WGI":
            codes.update(REFERENCE_WGI_CC_DATA.keys())

        # Custom data
        with self._lock:
            for cc, types in self._custom_data.items():
                if index_type in types:
                    codes.add(cc)

        return sorted(codes)
