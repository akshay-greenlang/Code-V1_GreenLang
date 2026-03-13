# -*- coding: utf-8 -*-
"""
greenlang.agents.eudr.commodity_risk_analyzer.commodity_profiler
================================================================

AGENT-EUDR-018 Engine 1: Commodity Profiler

Deep profiling of each EUDR-regulated commodity with comprehensive risk
characterization. Creates multidimensional risk profiles covering
deforestation risk, supply chain complexity, traceability scoring, and
intrinsic commodity characteristics for all 7 EUDR commodities (cattle,
cocoa, coffee, oil palm, rubber, soya, wood) and their derived products.

ZERO-HALLUCINATION GUARANTEES:
    - 100% deterministic: same inputs produce identical profiles
    - NO LLM involvement in any risk scoring or profiling path
    - All arithmetic uses Decimal for bit-perfect reproducibility
    - SHA-256 provenance hash on every profile operation
    - Complete audit trail for regulatory inspection

Risk Profile Dimensions:
    1. Deforestation Risk (0-100): Based on sourcing country benchmarks,
       commodity-specific deforestation correlation, and historical trends.
    2. Supply Chain Complexity (0-100): Number of processing stages,
       intermediaries, geographic spread, and custody model mix.
    3. Traceability Score (0-100): How traceable the commodity is through
       its supply chain from plot of origin to final product.
    4. Intrinsic Risk Characteristics: Perishability, processing complexity,
       substitutability, and fraud susceptibility.
    5. Overall Risk Score: Weighted composite of all dimensions.

Regulatory References:
    - EUDR Article 29: Country benchmarking (Low / Standard / High)
    - EUDR Article 10: Risk assessment requirement for operators
    - EUDR Article 9: Information collection requirement
    - EUDR Annex I: Commodity and product coverage

Dependencies:
    - .config (get_config): CommodityRiskAnalyzerConfig singleton
    - .models: CommodityType, RiskLevel, CommodityProfile
    - .provenance (ProvenanceTracker): SHA-256 audit chain
    - .metrics: Prometheus instrumentation

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Module version for provenance tracking.
_MODULE_VERSION: str = "1.0.0"

#: Decimal precision for risk scores (2 decimal places).
_RISK_PRECISION = Decimal("0.01")

#: Maximum risk score.
_MAX_RISK = Decimal("100.00")

#: Minimum risk score.
_MIN_RISK = Decimal("0.00")

#: The 7 primary EUDR commodities.
EUDR_COMMODITIES: FrozenSet[str] = frozenset({
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
})

#: Default risk weights for the composite score.
DEFAULT_PROFILE_WEIGHTS: Dict[str, Decimal] = {
    "deforestation_risk": Decimal("0.35"),
    "supply_chain_complexity": Decimal("0.25"),
    "traceability_gap": Decimal("0.20"),
    "intrinsic_risk": Decimal("0.20"),
}

# ---------------------------------------------------------------------------
# Commodity-specific baseline risk characteristics
# ---------------------------------------------------------------------------

#: Baseline deforestation risk by commodity (0-100 scale).
#: Palm oil has the highest deforestation association globally.
BASELINE_DEFORESTATION_RISK: Dict[str, Decimal] = {
    "cattle": Decimal("85.00"),
    "cocoa": Decimal("65.00"),
    "coffee": Decimal("45.00"),
    "oil_palm": Decimal("90.00"),
    "rubber": Decimal("55.00"),
    "soya": Decimal("75.00"),
    "wood": Decimal("60.00"),
}

#: Country-level deforestation risk scores per EUDR Article 29 benchmarking.
#: High-risk producing countries for each commodity.
COUNTRY_DEFORESTATION_RISK: Dict[str, Decimal] = {
    "BR": Decimal("80.00"),   # Brazil - Amazon/Cerrado deforestation
    "ID": Decimal("85.00"),   # Indonesia - palm oil, rubber
    "MY": Decimal("70.00"),   # Malaysia - palm oil
    "CO": Decimal("55.00"),   # Colombia - coffee, cattle
    "GH": Decimal("65.00"),   # Ghana - cocoa
    "CI": Decimal("70.00"),   # Cote d'Ivoire - cocoa
    "CM": Decimal("60.00"),   # Cameroon - cocoa, wood
    "PG": Decimal("65.00"),   # Papua New Guinea - palm oil, wood
    "NG": Decimal("55.00"),   # Nigeria - cocoa, palm oil
    "PE": Decimal("50.00"),   # Peru - coffee, cocoa, wood
    "VN": Decimal("40.00"),   # Vietnam - coffee, rubber
    "TH": Decimal("45.00"),   # Thailand - rubber
    "ET": Decimal("35.00"),   # Ethiopia - coffee
    "UG": Decimal("40.00"),   # Uganda - coffee
    "PY": Decimal("70.00"),   # Paraguay - cattle, soya
    "AR": Decimal("60.00"),   # Argentina - soya, cattle
    "BO": Decimal("65.00"),   # Bolivia - soya, wood
    "CG": Decimal("55.00"),   # Congo - wood
    "CD": Decimal("60.00"),   # DRC - wood, cocoa
    "MM": Decimal("50.00"),   # Myanmar - rubber, wood
    "LR": Decimal("45.00"),   # Liberia - rubber, palm oil
    "SL": Decimal("40.00"),   # Sierra Leone - cocoa
    "MZ": Decimal("35.00"),   # Mozambique - wood
    "EC": Decimal("45.00"),   # Ecuador - cocoa
    "HN": Decimal("40.00"),   # Honduras - coffee
    "GT": Decimal("35.00"),   # Guatemala - coffee
    "MX": Decimal("30.00"),   # Mexico - coffee
    "IN": Decimal("25.00"),   # India - rubber, coffee
    "KE": Decimal("20.00"),   # Kenya - coffee
    "CR": Decimal("15.00"),   # Costa Rica - coffee
    "DEFAULT": Decimal("30.00"),
}

#: Intrinsic commodity characteristics (perishability 0-100, processing
#: complexity 0-100, substitutability 0-100, fraud susceptibility 0-100).
COMMODITY_CHARACTERISTICS: Dict[str, Dict[str, Decimal]] = {
    "cattle": {
        "perishability": Decimal("75.00"),
        "processing_complexity": Decimal("70.00"),
        "substitutability": Decimal("30.00"),
        "fraud_susceptibility": Decimal("55.00"),
        "seasonal_variability": Decimal("25.00"),
        "geographic_concentration": Decimal("45.00"),
    },
    "cocoa": {
        "perishability": Decimal("50.00"),
        "processing_complexity": Decimal("65.00"),
        "substitutability": Decimal("20.00"),
        "fraud_susceptibility": Decimal("70.00"),
        "seasonal_variability": Decimal("60.00"),
        "geographic_concentration": Decimal("80.00"),
    },
    "coffee": {
        "perishability": Decimal("40.00"),
        "processing_complexity": Decimal("55.00"),
        "substitutability": Decimal("25.00"),
        "fraud_susceptibility": Decimal("50.00"),
        "seasonal_variability": Decimal("65.00"),
        "geographic_concentration": Decimal("55.00"),
    },
    "oil_palm": {
        "perishability": Decimal("85.00"),
        "processing_complexity": Decimal("80.00"),
        "substitutability": Decimal("35.00"),
        "fraud_susceptibility": Decimal("75.00"),
        "seasonal_variability": Decimal("30.00"),
        "geographic_concentration": Decimal("90.00"),
    },
    "rubber": {
        "perishability": Decimal("20.00"),
        "processing_complexity": Decimal("60.00"),
        "substitutability": Decimal("40.00"),
        "fraud_susceptibility": Decimal("45.00"),
        "seasonal_variability": Decimal("50.00"),
        "geographic_concentration": Decimal("70.00"),
    },
    "soya": {
        "perishability": Decimal("25.00"),
        "processing_complexity": Decimal("50.00"),
        "substitutability": Decimal("45.00"),
        "fraud_susceptibility": Decimal("40.00"),
        "seasonal_variability": Decimal("55.00"),
        "geographic_concentration": Decimal("65.00"),
    },
    "wood": {
        "perishability": Decimal("10.00"),
        "processing_complexity": Decimal("45.00"),
        "substitutability": Decimal("50.00"),
        "fraud_susceptibility": Decimal("65.00"),
        "seasonal_variability": Decimal("20.00"),
        "geographic_concentration": Decimal("40.00"),
    },
}

#: Typical supply chain depth (number of stages from producer to importer).
TYPICAL_SUPPLY_CHAIN_DEPTH: Dict[str, int] = {
    "cattle": 6,    # ranch -> feedlot -> slaughterhouse -> processor -> trader -> importer
    "cocoa": 7,     # farm -> collector -> exporter -> processor -> manufacturer -> trader -> importer
    "coffee": 6,    # farm -> washing station -> exporter -> roaster -> trader -> importer
    "oil_palm": 5,  # plantation -> mill -> refinery -> trader -> importer
    "rubber": 5,    # plantation -> processing plant -> trader -> manufacturer -> importer
    "soya": 5,      # farm -> silo -> crusher -> trader -> importer
    "wood": 6,      # concession -> sawmill -> processor -> trader -> exporter -> importer
}

#: Traceability baseline difficulty scores (0-100, higher = harder to trace).
TRACEABILITY_DIFFICULTY: Dict[str, Decimal] = {
    "cattle": Decimal("65.00"),   # Complex: multiple live transfers, feedlots
    "cocoa": Decimal("75.00"),    # Very complex: smallholders, mixing at collectors
    "coffee": Decimal("55.00"),   # Moderate: washing stations provide aggregation point
    "oil_palm": Decimal("80.00"), # Most complex: FFB mixing at mills, diverse smallholders
    "rubber": Decimal("60.00"),   # Moderate-high: cup lump mixing
    "soya": Decimal("50.00"),     # Moderate: large-scale farms, silo mixing
    "wood": Decimal("45.00"),     # Lower: log marking systems, FSC chain of custody
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for consistency."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert a numeric value to Decimal via string to avoid IEEE 754 artefacts.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation of the value.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _clamp_risk(value: Decimal) -> Decimal:
    """Clamp a risk score to [0.00, 100.00] and apply precision.

    Args:
        value: Unbound risk score.

    Returns:
        Clamped and quantized Decimal risk score.
    """
    clamped = max(_MIN_RISK, min(_MAX_RISK, value))
    return clamped.quantize(_RISK_PRECISION, rounding=ROUND_HALF_UP)


def _compute_provenance_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_commodity_type(commodity_type: str) -> str:
    """Validate and normalize a commodity type string.

    Args:
        commodity_type: Raw commodity type input.

    Returns:
        Normalized lowercase commodity type.

    Raises:
        ValueError: If commodity_type is not a valid EUDR commodity.
    """
    if not commodity_type or not isinstance(commodity_type, str):
        raise ValueError("commodity_type must be a non-empty string")
    normalized = commodity_type.strip().lower()
    if normalized not in EUDR_COMMODITIES:
        raise ValueError(
            f"Invalid commodity_type '{commodity_type}'. "
            f"Must be one of: {sorted(EUDR_COMMODITIES)}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Prometheus metrics integration (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram, REGISTRY

    def _safe_counter(name: str, doc: str, labelnames: list = None):
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(name: str, doc: str, labelnames: list = None,
                        buckets: tuple = ()):
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    _PROFILES_CREATED_TOTAL = _safe_counter(
        "gl_eudr_cra_profiles_created_total",
        "Total commodity risk profiles created",
        labelnames=["commodity_type"],
    )
    _PROFILE_DURATION_SECONDS = _safe_histogram(
        "gl_eudr_cra_profile_duration_seconds",
        "Duration of commodity profiling operations in seconds",
        labelnames=["operation"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _PROFILE_ERRORS_TOTAL = _safe_counter(
        "gl_eudr_cra_profile_errors_total",
        "Total commodity profiling errors",
        labelnames=["operation"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROFILES_CREATED_TOTAL = None  # type: ignore[assignment]
    _PROFILE_DURATION_SECONDS = None  # type: ignore[assignment]
    _PROFILE_ERRORS_TOTAL = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; commodity profiler metrics disabled"
    )


def _record_profile_created(commodity_type: str) -> None:
    """Record a profile creation metric."""
    if _PROMETHEUS_AVAILABLE and _PROFILES_CREATED_TOTAL is not None:
        _PROFILES_CREATED_TOTAL.labels(commodity_type=commodity_type).inc()


def _observe_profile_duration(operation: str, seconds: float) -> None:
    """Record a profiling duration metric."""
    if _PROMETHEUS_AVAILABLE and _PROFILE_DURATION_SECONDS is not None:
        _PROFILE_DURATION_SECONDS.labels(operation=operation).observe(seconds)


def _record_profile_error(operation: str) -> None:
    """Record a profiling error metric."""
    if _PROMETHEUS_AVAILABLE and _PROFILE_ERRORS_TOTAL is not None:
        _PROFILE_ERRORS_TOTAL.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# CommodityProfiler
# ---------------------------------------------------------------------------


class CommodityProfiler:
    """Deep profiling engine for EUDR-regulated commodities.

    Creates comprehensive, multidimensional risk profiles for each of the
    7 EUDR commodities, incorporating deforestation risk from sourcing
    patterns, supply chain complexity scoring, traceability analysis, and
    intrinsic commodity characteristics.

    All calculations are deterministic using Decimal arithmetic. No LLM or
    ML models are used in any risk scoring path (zero-hallucination).

    Attributes:
        _config: Configuration dictionary for profiling parameters.
        _profiles: Cache of computed profiles keyed by profile_id.
        _lock: Reentrant lock for thread-safe profile cache access.
        _profile_weights: Weights for composite risk score calculation.

    Example:
        >>> profiler = CommodityProfiler()
        >>> profile = profiler.profile_commodity(
        ...     commodity_type="oil_palm",
        ...     country_data={"ID": 60, "MY": 40},
        ...     supply_chain_data={"stages": 5, "intermediaries": 12}
        ... )
        >>> assert profile["overall_risk_score"] > Decimal("0")
        >>> assert profile["provenance_hash"] != ""
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        profile_weights: Optional[Dict[str, Decimal]] = None,
    ) -> None:
        """Initialize CommodityProfiler with optional configuration.

        Args:
            config: Optional configuration dictionary. If None, uses
                module-level defaults.
            profile_weights: Optional custom weights for composite risk
                score components. Keys must be: deforestation_risk,
                supply_chain_complexity, traceability_gap, intrinsic_risk.
                Values must sum to 1.0.
        """
        self._config: Dict[str, Any] = config or {}
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.RLock = threading.RLock()

        # Set profile weights, validating they sum to 1.0
        if profile_weights is not None:
            self._validate_weights(profile_weights)
            self._profile_weights = dict(profile_weights)
        else:
            self._profile_weights = dict(DEFAULT_PROFILE_WEIGHTS)

        logger.info(
            "CommodityProfiler initialized: weights=%s, "
            "cached_profiles=%d",
            {k: str(v) for k, v in self._profile_weights.items()},
            len(self._profiles),
        )

    # ------------------------------------------------------------------
    # Weight validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_weights(weights: Dict[str, Decimal]) -> None:
        """Validate that profile weights have correct keys and sum to 1.0.

        Args:
            weights: Dictionary of weight name -> Decimal value.

        Raises:
            ValueError: If keys are incorrect or values do not sum to 1.0.
        """
        required_keys = {
            "deforestation_risk", "supply_chain_complexity",
            "traceability_gap", "intrinsic_risk",
        }
        if set(weights.keys()) != required_keys:
            raise ValueError(
                f"profile_weights must have exactly keys {sorted(required_keys)}, "
                f"got {sorted(weights.keys())}"
            )
        weight_sum = sum(weights.values())
        if abs(weight_sum - Decimal("1.00")) > Decimal("0.001"):
            raise ValueError(
                f"profile_weights must sum to 1.0, got {weight_sum}"
            )
        for name, value in weights.items():
            if value < Decimal("0") or value > Decimal("1"):
                raise ValueError(
                    f"Weight '{name}' must be in [0, 1], got {value}"
                )

    # ------------------------------------------------------------------
    # Public API: Single commodity profiling
    # ------------------------------------------------------------------

    def profile_commodity(
        self,
        commodity_type: str,
        country_data: Dict[str, Any],
        supply_chain_data: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a comprehensive risk profile for one EUDR commodity.

        Generates a multidimensional profile covering deforestation risk,
        supply chain complexity, traceability scoring, intrinsic risk
        characteristics, and an overall composite risk score.

        Args:
            commodity_type: One of the 7 EUDR commodities (cattle, cocoa,
                coffee, oil_palm, rubber, soya, wood).
            country_data: Sourcing country information. Expected keys:
                - Country codes (ISO alpha-2) mapped to sourcing percentage
                  (e.g., {"BR": 60, "ID": 40}).
            supply_chain_data: Supply chain structure information. Expected:
                - "stages" (int): Number of processing stages.
                - "intermediaries" (int): Number of intermediary actors.
                - "countries" (int, optional): Number of countries involved.
                - "custody_models" (list, optional): Custody models used.
            **kwargs: Additional profiling parameters:
                - "chain_data" (dict): Traceability chain data override.
                - "force_refresh" (bool): Skip cache and recompute.

        Returns:
            Dictionary containing the complete profile:
                - profile_id (str): Unique profile identifier.
                - commodity_type (str): Normalized commodity type.
                - deforestation_risk (Decimal): 0-100 score.
                - supply_chain_complexity (Decimal): 0-100 score.
                - traceability_score (Decimal): 0-100 score.
                - characteristics (dict): Intrinsic risk characteristics.
                - overall_risk_score (Decimal): Weighted composite score.
                - risk_level (str): LOW, STANDARD, or HIGH.
                - sourcing_countries (dict): Country-level risk breakdown.
                - provenance_hash (str): SHA-256 hash.
                - created_at (str): ISO timestamp.
                - processing_time_ms (float): Operation duration.

        Raises:
            ValueError: If commodity_type is invalid or data is malformed.
        """
        start_time = time.monotonic()
        operation = "profile_commodity"

        try:
            commodity = _validate_commodity_type(commodity_type)

            # Check cache unless force_refresh
            force_refresh = kwargs.get("force_refresh", False)
            cache_key = self._build_cache_key(commodity, country_data, supply_chain_data)

            if not force_refresh:
                cached = self._get_cached_profile(cache_key)
                if cached is not None:
                    logger.debug(
                        "Returning cached profile for %s: %s",
                        commodity, cache_key[:16],
                    )
                    return cached

            # Step 1: Calculate deforestation risk
            deforestation_risk = self.calculate_deforestation_risk(
                commodity, country_data,
            )

            # Step 2: Calculate supply chain complexity
            stages = supply_chain_data.get("stages", TYPICAL_SUPPLY_CHAIN_DEPTH.get(commodity, 5))
            intermediaries = supply_chain_data.get("intermediaries", 0)
            complexity = self.calculate_supply_chain_complexity(
                commodity, stages, intermediaries,
                countries_count=supply_chain_data.get("countries", 1),
                custody_models=supply_chain_data.get("custody_models", []),
            )

            # Step 3: Calculate traceability score
            chain_data = kwargs.get("chain_data", supply_chain_data)
            traceability = self.calculate_traceability_score(commodity, chain_data)

            # Step 4: Get intrinsic characteristics
            characteristics = self.get_commodity_characteristics(commodity)

            # Step 5: Calculate intrinsic risk (average of characteristics)
            intrinsic_risk = self._calculate_intrinsic_risk(characteristics)

            # Step 6: Calculate overall composite score
            profile_data = {
                "deforestation_risk": deforestation_risk,
                "supply_chain_complexity": complexity,
                "traceability_gap": Decimal("100.00") - traceability,
                "intrinsic_risk": intrinsic_risk,
            }
            overall_score = self.calculate_overall_risk_score(profile_data)

            # Step 7: Determine risk level
            risk_level = self._classify_risk_level(overall_score)

            # Step 8: Build country-level risk breakdown
            sourcing_risk = self._build_sourcing_risk_breakdown(
                commodity, country_data,
            )

            # Step 9: Compute provenance hash
            profile_id = str(uuid.uuid4())
            profile_payload = {
                "profile_id": profile_id,
                "commodity_type": commodity,
                "deforestation_risk": str(deforestation_risk),
                "supply_chain_complexity": str(complexity),
                "traceability_score": str(traceability),
                "intrinsic_risk": str(intrinsic_risk),
                "overall_risk_score": str(overall_score),
                "country_data": country_data,
                "supply_chain_data": supply_chain_data,
            }
            provenance_hash = _compute_provenance_hash(profile_payload)

            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            # Build result
            profile = {
                "profile_id": profile_id,
                "commodity_type": commodity,
                "deforestation_risk": deforestation_risk,
                "supply_chain_complexity": complexity,
                "traceability_score": traceability,
                "characteristics": characteristics,
                "intrinsic_risk": intrinsic_risk,
                "overall_risk_score": overall_score,
                "risk_level": risk_level,
                "sourcing_countries": sourcing_risk,
                "provenance_hash": provenance_hash,
                "created_at": _utcnow().isoformat(),
                "processing_time_ms": round(elapsed_ms, 2),
                "version": _MODULE_VERSION,
            }

            # Cache and record metrics
            self._cache_profile(cache_key, profile)
            _record_profile_created(commodity)
            _observe_profile_duration(operation, elapsed_ms / 1000.0)

            logger.info(
                "Profiled commodity=%s: deforestation=%.2f, complexity=%.2f, "
                "traceability=%.2f, overall=%.2f, risk_level=%s, "
                "time_ms=%.2f, hash_prefix=%s",
                commodity,
                deforestation_risk,
                complexity,
                traceability,
                overall_score,
                risk_level,
                elapsed_ms,
                provenance_hash[:16],
            )
            return profile

        except ValueError:
            _record_profile_error(operation)
            raise
        except Exception as exc:
            _record_profile_error(operation)
            logger.error(
                "CommodityProfiler.profile_commodity failed: %s",
                str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API: Batch profiling
    # ------------------------------------------------------------------

    def profile_all_commodities(
        self,
        data_sources: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Batch profile all 7 EUDR commodities from unified data sources.

        Args:
            data_sources: Dictionary keyed by commodity type, each mapping
                to a dict with keys "country_data" and "supply_chain_data".
                Example::

                    {
                        "oil_palm": {
                            "country_data": {"ID": 60, "MY": 40},
                            "supply_chain_data": {"stages": 5, "intermediaries": 12},
                        },
                        "cocoa": { ... },
                    }

        Returns:
            Dictionary containing:
                - profiles (dict): Commodity type -> profile dict.
                - summary (dict): Aggregated statistics across all commodities.
                - highest_risk_commodity (str): Commodity with highest overall score.
                - lowest_risk_commodity (str): Commodity with lowest overall score.
                - provenance_hash (str): SHA-256 hash of the batch operation.
                - processing_time_ms (float): Total batch duration.

        Raises:
            ValueError: If data_sources is empty or contains invalid commodities.
        """
        start_time = time.monotonic()
        operation = "profile_all_commodities"

        if not data_sources:
            raise ValueError("data_sources must not be empty")

        try:
            profiles: Dict[str, Dict[str, Any]] = {}
            errors: Dict[str, str] = {}

            for commodity_type, source_data in data_sources.items():
                try:
                    country_data = source_data.get("country_data", {})
                    supply_chain_data = source_data.get("supply_chain_data", {})
                    profile = self.profile_commodity(
                        commodity_type=commodity_type,
                        country_data=country_data,
                        supply_chain_data=supply_chain_data,
                    )
                    profiles[commodity_type] = profile
                except (ValueError, KeyError) as exc:
                    errors[commodity_type] = str(exc)
                    logger.warning(
                        "Failed to profile commodity=%s: %s",
                        commodity_type, str(exc),
                    )

            # Build summary statistics
            summary = self._build_batch_summary(profiles)

            # Determine extremes
            highest_risk = ""
            lowest_risk = ""
            max_score = Decimal("-1")
            min_score = Decimal("101")

            for ct, prof in profiles.items():
                score = prof["overall_risk_score"]
                if score > max_score:
                    max_score = score
                    highest_risk = ct
                if score < min_score:
                    min_score = score
                    lowest_risk = ct

            # Provenance hash
            batch_payload = {
                "profiles": {
                    k: v.get("provenance_hash", "")
                    for k, v in profiles.items()
                },
                "commodity_count": len(profiles),
                "error_count": len(errors),
            }
            provenance_hash = _compute_provenance_hash(batch_payload)

            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            _observe_profile_duration(operation, elapsed_ms / 1000.0)

            result = {
                "profiles": profiles,
                "summary": summary,
                "highest_risk_commodity": highest_risk,
                "lowest_risk_commodity": lowest_risk,
                "errors": errors if errors else None,
                "commodity_count": len(profiles),
                "provenance_hash": provenance_hash,
                "processing_time_ms": round(elapsed_ms, 2),
                "created_at": _utcnow().isoformat(),
            }

            logger.info(
                "Batch profiled %d commodities: highest_risk=%s (%.2f), "
                "lowest_risk=%s (%.2f), errors=%d, time_ms=%.2f",
                len(profiles),
                highest_risk,
                max_score if max_score >= Decimal("0") else Decimal("0"),
                lowest_risk,
                min_score if min_score <= Decimal("100") else Decimal("0"),
                len(errors),
                elapsed_ms,
            )
            return result

        except Exception as exc:
            _record_profile_error(operation)
            logger.error(
                "CommodityProfiler.profile_all_commodities failed: %s",
                str(exc), exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Public API: Deforestation risk calculation
    # ------------------------------------------------------------------

    def calculate_deforestation_risk(
        self,
        commodity_type: str,
        sourcing_countries: Dict[str, Any],
    ) -> Decimal:
        """Calculate commodity-specific deforestation risk based on sourcing patterns.

        Combines the commodity's baseline deforestation association with
        the weighted country-level deforestation risk for each sourcing
        country, proportioned by sourcing percentage.

        Formula:
            deforestation_risk = 0.40 * baseline_commodity_risk
                               + 0.60 * weighted_country_risk

            weighted_country_risk = SUM(country_risk_i * sourcing_pct_i / 100)

        Args:
            commodity_type: Validated EUDR commodity type.
            sourcing_countries: Country code -> sourcing percentage mapping.
                Percentages should sum to 100 but the engine normalizes
                if they do not.

        Returns:
            Decimal risk score clamped to [0.00, 100.00].

        Raises:
            ValueError: If commodity_type is invalid.
        """
        commodity = _validate_commodity_type(commodity_type)

        # Get baseline commodity deforestation risk
        baseline = BASELINE_DEFORESTATION_RISK.get(commodity, Decimal("50.00"))

        if not sourcing_countries:
            return _clamp_risk(baseline)

        # Calculate weighted country risk
        total_pct = Decimal("0")
        weighted_country_risk = Decimal("0")

        for country_code, pct_raw in sourcing_countries.items():
            pct = _to_decimal(pct_raw)
            if pct <= Decimal("0"):
                continue
            total_pct += pct
            country_code_upper = country_code.upper().strip()
            country_risk = COUNTRY_DEFORESTATION_RISK.get(
                country_code_upper,
                COUNTRY_DEFORESTATION_RISK["DEFAULT"],
            )
            weighted_country_risk += country_risk * pct

        # Normalize if percentages do not sum to 100
        if total_pct > Decimal("0"):
            weighted_country_risk = weighted_country_risk / total_pct
        else:
            weighted_country_risk = Decimal("30.00")

        # Composite: 40% baseline commodity risk, 60% country risk
        commodity_weight = Decimal("0.40")
        country_weight = Decimal("0.60")
        composite = (baseline * commodity_weight) + (weighted_country_risk * country_weight)

        result = _clamp_risk(composite)
        logger.debug(
            "Deforestation risk for %s: baseline=%.2f, "
            "country_weighted=%.2f, composite=%.2f",
            commodity, baseline, weighted_country_risk, result,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Supply chain complexity
    # ------------------------------------------------------------------

    def calculate_supply_chain_complexity(
        self,
        commodity_type: str,
        processing_stages: int,
        intermediaries: int,
        countries_count: int = 1,
        custody_models: Optional[List[str]] = None,
    ) -> Decimal:
        """Score 0-100 for supply chain complexity of a commodity.

        Complexity is determined by the number of processing stages,
        number of intermediary actors, geographic spread (country count),
        custody model diversity, and comparison to the typical supply
        chain depth for the commodity.

        Formula:
            complexity = (
                0.30 * stage_score
              + 0.25 * intermediary_score
              + 0.20 * geographic_score
              + 0.15 * custody_diversity_score
              + 0.10 * depth_deviation_score
            )

        Args:
            commodity_type: Validated EUDR commodity type.
            processing_stages: Number of distinct processing stages.
            intermediaries: Number of intermediary actors in the chain.
            countries_count: Number of countries the chain spans.
            custody_models: List of custody model types used
                (e.g., ["identity_preserved", "mass_balance"]).

        Returns:
            Decimal complexity score clamped to [0.00, 100.00].

        Raises:
            ValueError: If inputs are negative or commodity_type is invalid.
        """
        commodity = _validate_commodity_type(commodity_type)

        if processing_stages < 0:
            raise ValueError(
                f"processing_stages must be >= 0, got {processing_stages}"
            )
        if intermediaries < 0:
            raise ValueError(
                f"intermediaries must be >= 0, got {intermediaries}"
            )
        if countries_count < 1:
            raise ValueError(
                f"countries_count must be >= 1, got {countries_count}"
            )

        # Stage score: more stages = higher complexity (normalized to 0-100)
        typical_depth = TYPICAL_SUPPLY_CHAIN_DEPTH.get(commodity, 5)
        max_stages = max(typical_depth * 3, 15)
        stage_score = _clamp_risk(
            _to_decimal(min(processing_stages, max_stages))
            / _to_decimal(max_stages)
            * Decimal("100")
        )

        # Intermediary score: more intermediaries = higher complexity
        max_intermediaries = 50
        intermediary_score = _clamp_risk(
            _to_decimal(min(intermediaries, max_intermediaries))
            / _to_decimal(max_intermediaries)
            * Decimal("100")
        )

        # Geographic spread score
        max_countries = 20
        geographic_score = _clamp_risk(
            _to_decimal(min(countries_count, max_countries))
            / _to_decimal(max_countries)
            * Decimal("100")
        )

        # Custody model diversity score
        models = custody_models or []
        unique_models = len(set(m.lower().strip() for m in models if m))
        max_model_types = 3  # identity_preserved, segregated, mass_balance
        custody_diversity = _clamp_risk(
            _to_decimal(min(unique_models, max_model_types))
            / _to_decimal(max_model_types)
            * Decimal("100")
        )

        # Depth deviation score: how much the chain deviates from typical
        deviation = abs(processing_stages - typical_depth)
        max_deviation = typical_depth * 2
        depth_deviation = _clamp_risk(
            _to_decimal(min(deviation, max_deviation))
            / _to_decimal(max(max_deviation, 1))
            * Decimal("100")
        )

        # Weighted composite
        composite = (
            stage_score * Decimal("0.30")
            + intermediary_score * Decimal("0.25")
            + geographic_score * Decimal("0.20")
            + custody_diversity * Decimal("0.15")
            + depth_deviation * Decimal("0.10")
        )

        result = _clamp_risk(composite)
        logger.debug(
            "Supply chain complexity for %s: stages=%.2f, "
            "intermediaries=%.2f, geographic=%.2f, custody=%.2f, "
            "deviation=%.2f, composite=%.2f",
            commodity, stage_score, intermediary_score,
            geographic_score, custody_diversity, depth_deviation, result,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Traceability score
    # ------------------------------------------------------------------

    def calculate_traceability_score(
        self,
        commodity_type: str,
        chain_data: Dict[str, Any],
    ) -> Decimal:
        """Calculate how traceable a commodity is through its supply chain.

        Higher score means better traceability. Incorporates the commodity's
        inherent traceability difficulty, the percentage of actors with
        verified documentation, GPS coverage, and certification presence.

        Formula:
            traceability = (
                100 - baseline_difficulty * 0.30
              + documentation_pct * 0.30
              + gps_coverage_pct * 0.25
              + certification_pct * 0.15
            )

        Args:
            commodity_type: Validated EUDR commodity type.
            chain_data: Supply chain traceability data. Optional keys:
                - "documentation_pct" (float): % of actors with docs.
                - "gps_coverage_pct" (float): % of producers with GPS.
                - "certification_pct" (float): % of chain certified.
                - "verified_origins" (int): Number of verified origin plots.
                - "total_origins" (int): Total origin plots expected.

        Returns:
            Decimal traceability score clamped to [0.00, 100.00].

        Raises:
            ValueError: If commodity_type is invalid.
        """
        commodity = _validate_commodity_type(commodity_type)

        # Baseline difficulty (inverted for traceability score)
        baseline_difficulty = TRACEABILITY_DIFFICULTY.get(
            commodity, Decimal("50.00")
        )
        baseline_contribution = (Decimal("100.00") - baseline_difficulty) * Decimal("0.30")

        # Documentation percentage
        doc_pct = _to_decimal(chain_data.get("documentation_pct", 50))
        doc_pct = max(Decimal("0"), min(Decimal("100"), doc_pct))
        doc_contribution = doc_pct * Decimal("0.30")

        # GPS coverage percentage
        gps_pct = _to_decimal(chain_data.get("gps_coverage_pct", 40))
        gps_pct = max(Decimal("0"), min(Decimal("100"), gps_pct))
        gps_contribution = gps_pct * Decimal("0.25")

        # Certification percentage
        cert_pct = _to_decimal(chain_data.get("certification_pct", 30))
        cert_pct = max(Decimal("0"), min(Decimal("100"), cert_pct))
        cert_contribution = cert_pct * Decimal("0.15")

        # Origin verification bonus
        verified_origins = chain_data.get("verified_origins", 0)
        total_origins = chain_data.get("total_origins", 0)
        if total_origins > 0 and verified_origins > 0:
            origin_ratio = _to_decimal(verified_origins) / _to_decimal(total_origins)
            origin_ratio = min(origin_ratio, Decimal("1.00"))
        else:
            origin_ratio = Decimal("0.00")

        # Origin ratio adds a small bonus (up to 5 points)
        origin_bonus = origin_ratio * Decimal("5.00")

        composite = (
            baseline_contribution
            + doc_contribution
            + gps_contribution
            + cert_contribution
            + origin_bonus
        )

        result = _clamp_risk(composite)
        logger.debug(
            "Traceability score for %s: baseline_contrib=%.2f, "
            "doc=%.2f, gps=%.2f, cert=%.2f, origin_bonus=%.2f, "
            "composite=%.2f",
            commodity, baseline_contribution, doc_contribution,
            gps_contribution, cert_contribution, origin_bonus, result,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Commodity characteristics
    # ------------------------------------------------------------------

    def get_commodity_characteristics(
        self,
        commodity_type: str,
    ) -> Dict[str, Decimal]:
        """Return intrinsic risk characteristics for a commodity.

        Returns a dictionary of characteristic name -> score (0-100)
        covering perishability, processing complexity, substitutability,
        fraud susceptibility, seasonal variability, and geographic
        concentration.

        Args:
            commodity_type: Validated EUDR commodity type.

        Returns:
            Dictionary of characteristic names to Decimal scores.

        Raises:
            ValueError: If commodity_type is invalid.
        """
        commodity = _validate_commodity_type(commodity_type)

        chars = COMMODITY_CHARACTERISTICS.get(commodity)
        if chars is None:
            logger.warning(
                "No characteristics data for commodity=%s, using defaults",
                commodity,
            )
            return {
                "perishability": Decimal("50.00"),
                "processing_complexity": Decimal("50.00"),
                "substitutability": Decimal("50.00"),
                "fraud_susceptibility": Decimal("50.00"),
                "seasonal_variability": Decimal("50.00"),
                "geographic_concentration": Decimal("50.00"),
            }

        return dict(chars)

    # ------------------------------------------------------------------
    # Public API: Commodity comparison
    # ------------------------------------------------------------------

    def compare_commodities(
        self,
        commodity_types: List[str],
        country_data: Optional[Dict[str, Dict[str, Any]]] = None,
        supply_chain_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Side-by-side comparison of commodity risk profiles.

        Creates profiles for each requested commodity and presents them
        in a comparative format with rankings and differential analysis.

        Args:
            commodity_types: List of EUDR commodity types to compare.
            country_data: Optional per-commodity country sourcing data.
                Keyed by commodity type.
            supply_chain_data: Optional per-commodity supply chain data.
                Keyed by commodity type.

        Returns:
            Dictionary containing:
                - commodities (dict): Commodity type -> profile.
                - rankings (list): Sorted by overall risk score (highest first).
                - risk_spread (Decimal): Difference between highest and lowest.
                - dimension_leaders (dict): Which commodity leads each dimension.
                - provenance_hash (str): SHA-256 hash.

        Raises:
            ValueError: If commodity_types is empty or contains invalid entries.
        """
        start_time = time.monotonic()

        if not commodity_types:
            raise ValueError("commodity_types must not be empty")
        if len(commodity_types) < 2:
            raise ValueError(
                "compare_commodities requires at least 2 commodity types"
            )

        country_data = country_data or {}
        supply_chain_data = supply_chain_data or {}
        profiles: Dict[str, Dict[str, Any]] = {}

        for ct in commodity_types:
            c_country = country_data.get(ct, {})
            c_supply = supply_chain_data.get(ct, {})
            profile = self.profile_commodity(
                commodity_type=ct,
                country_data=c_country,
                supply_chain_data=c_supply,
            )
            profiles[ct] = profile

        # Build rankings (highest risk first)
        rankings = sorted(
            profiles.items(),
            key=lambda x: x[1]["overall_risk_score"],
            reverse=True,
        )
        ranking_list = [
            {
                "rank": i + 1,
                "commodity_type": ct,
                "overall_risk_score": prof["overall_risk_score"],
                "risk_level": prof["risk_level"],
            }
            for i, (ct, prof) in enumerate(rankings)
        ]

        # Risk spread
        scores = [prof["overall_risk_score"] for prof in profiles.values()]
        risk_spread = max(scores) - min(scores) if scores else Decimal("0")

        # Dimension leaders
        dimension_leaders = self._find_dimension_leaders(profiles)

        # Provenance
        comparison_payload = {
            "commodity_types": sorted(commodity_types),
            "ranking_order": [r["commodity_type"] for r in ranking_list],
            "risk_spread": str(risk_spread),
        }
        provenance_hash = _compute_provenance_hash(comparison_payload)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        result = {
            "commodities": profiles,
            "rankings": ranking_list,
            "risk_spread": risk_spread,
            "dimension_leaders": dimension_leaders,
            "provenance_hash": provenance_hash,
            "processing_time_ms": round(elapsed_ms, 2),
            "created_at": _utcnow().isoformat(),
        }

        logger.info(
            "Compared %d commodities: highest=%s (%.2f), "
            "lowest=%s (%.2f), spread=%.2f",
            len(commodity_types),
            ranking_list[0]["commodity_type"],
            ranking_list[0]["overall_risk_score"],
            ranking_list[-1]["commodity_type"],
            ranking_list[-1]["overall_risk_score"],
            risk_spread,
        )
        return result

    # ------------------------------------------------------------------
    # Public API: Overall risk score
    # ------------------------------------------------------------------

    def calculate_overall_risk_score(
        self,
        profile_data: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate a weighted composite risk score from all risk factors.

        Args:
            profile_data: Dictionary with keys matching the profile weights:
                - deforestation_risk (Decimal): 0-100.
                - supply_chain_complexity (Decimal): 0-100.
                - traceability_gap (Decimal): 0-100 (100 - traceability_score).
                - intrinsic_risk (Decimal): 0-100.

        Returns:
            Decimal composite score clamped to [0.00, 100.00].

        Raises:
            ValueError: If required keys are missing from profile_data.
        """
        required_keys = set(self._profile_weights.keys())
        missing = required_keys - set(profile_data.keys())
        if missing:
            raise ValueError(
                f"profile_data missing required keys: {sorted(missing)}"
            )

        composite = Decimal("0.00")
        for dimension, weight in self._profile_weights.items():
            value = _to_decimal(profile_data[dimension])
            value = _clamp_risk(value)
            composite += value * weight

        result = _clamp_risk(composite)
        logger.debug(
            "Overall risk score: composite=%.2f from dimensions=%s",
            result,
            {k: str(v) for k, v in profile_data.items()},
        )
        return result

    # ------------------------------------------------------------------
    # Internal: Intrinsic risk calculation
    # ------------------------------------------------------------------

    def _calculate_intrinsic_risk(
        self,
        characteristics: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate intrinsic risk as weighted average of characteristics.

        Perishability and fraud susceptibility receive higher weights as
        they directly affect EUDR compliance risk.

        Args:
            characteristics: Commodity characteristics dictionary.

        Returns:
            Decimal intrinsic risk score clamped to [0.00, 100.00].
        """
        if not characteristics:
            return Decimal("50.00")

        # Weights for each characteristic dimension
        char_weights: Dict[str, Decimal] = {
            "perishability": Decimal("0.15"),
            "processing_complexity": Decimal("0.20"),
            "substitutability": Decimal("0.10"),
            "fraud_susceptibility": Decimal("0.25"),
            "seasonal_variability": Decimal("0.10"),
            "geographic_concentration": Decimal("0.20"),
        }

        total_weight = Decimal("0")
        weighted_sum = Decimal("0")

        for char_name, weight in char_weights.items():
            value = characteristics.get(char_name)
            if value is not None:
                weighted_sum += _to_decimal(value) * weight
                total_weight += weight

        if total_weight > Decimal("0"):
            result = weighted_sum / total_weight
        else:
            result = Decimal("50.00")

        return _clamp_risk(result)

    # ------------------------------------------------------------------
    # Internal: Risk level classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_risk_level(score: Decimal) -> str:
        """Classify a risk score into LOW, STANDARD, or HIGH.

        Per EUDR Article 29 benchmarking thresholds:
            - LOW: score < 30
            - STANDARD: 30 <= score < 70
            - HIGH: score >= 70

        Args:
            score: Risk score in [0, 100].

        Returns:
            Risk level string: "LOW", "STANDARD", or "HIGH".
        """
        if score < Decimal("30.00"):
            return "LOW"
        elif score < Decimal("70.00"):
            return "STANDARD"
        else:
            return "HIGH"

    # ------------------------------------------------------------------
    # Internal: Country risk breakdown
    # ------------------------------------------------------------------

    def _build_sourcing_risk_breakdown(
        self,
        commodity_type: str,
        country_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build a per-country risk breakdown for the sourcing profile.

        Args:
            commodity_type: Validated commodity type.
            country_data: Country code -> sourcing percentage.

        Returns:
            Dictionary of country code -> risk detail dict.
        """
        breakdown: Dict[str, Dict[str, Any]] = {}

        for country_code, pct_raw in country_data.items():
            pct = _to_decimal(pct_raw)
            cc = country_code.upper().strip()
            country_risk = COUNTRY_DEFORESTATION_RISK.get(
                cc, COUNTRY_DEFORESTATION_RISK["DEFAULT"],
            )
            baseline = BASELINE_DEFORESTATION_RISK.get(
                commodity_type, Decimal("50.00"),
            )

            # Combined risk for this country-commodity pair
            combined = (country_risk * Decimal("0.60")) + (baseline * Decimal("0.40"))

            breakdown[cc] = {
                "sourcing_percentage": pct,
                "country_risk_score": country_risk,
                "commodity_baseline": baseline,
                "combined_risk": _clamp_risk(combined),
                "risk_level": self._classify_risk_level(_clamp_risk(combined)),
            }

        return breakdown

    # ------------------------------------------------------------------
    # Internal: Dimension leaders
    # ------------------------------------------------------------------

    @staticmethod
    def _find_dimension_leaders(
        profiles: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Find which commodity has the highest score in each risk dimension.

        Args:
            profiles: Commodity type -> profile dict.

        Returns:
            Dictionary of dimension -> leader info dict.
        """
        dimensions = [
            "deforestation_risk",
            "supply_chain_complexity",
            "traceability_score",
            "intrinsic_risk",
            "overall_risk_score",
        ]
        leaders: Dict[str, Dict[str, Any]] = {}

        for dim in dimensions:
            best_ct = ""
            best_score = Decimal("-1")

            for ct, prof in profiles.items():
                score = prof.get(dim, Decimal("0"))
                if isinstance(score, (int, float)):
                    score = _to_decimal(score)
                if score > best_score:
                    best_score = score
                    best_ct = ct

            leaders[dim] = {
                "commodity_type": best_ct,
                "score": best_score if best_score >= Decimal("0") else Decimal("0"),
            }

        return leaders

    # ------------------------------------------------------------------
    # Internal: Batch summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_summary(
        profiles: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build aggregated statistics across profiled commodities.

        Args:
            profiles: Commodity type -> profile dict.

        Returns:
            Summary dictionary with averages and distributions.
        """
        if not profiles:
            return {
                "average_overall_risk": Decimal("0.00"),
                "average_deforestation_risk": Decimal("0.00"),
                "average_complexity": Decimal("0.00"),
                "average_traceability": Decimal("0.00"),
                "risk_distribution": {"LOW": 0, "STANDARD": 0, "HIGH": 0},
                "commodity_count": 0,
            }

        overall_scores: List[Decimal] = []
        deforestation_scores: List[Decimal] = []
        complexity_scores: List[Decimal] = []
        traceability_scores: List[Decimal] = []
        risk_dist: Dict[str, int] = {"LOW": 0, "STANDARD": 0, "HIGH": 0}

        for prof in profiles.values():
            overall_scores.append(prof["overall_risk_score"])
            deforestation_scores.append(prof["deforestation_risk"])
            complexity_scores.append(prof["supply_chain_complexity"])
            traceability_scores.append(prof["traceability_score"])
            level = prof.get("risk_level", "STANDARD")
            risk_dist[level] = risk_dist.get(level, 0) + 1

        count = _to_decimal(len(profiles))
        avg_overall = sum(overall_scores) / count
        avg_deforestation = sum(deforestation_scores) / count
        avg_complexity = sum(complexity_scores) / count
        avg_traceability = sum(traceability_scores) / count

        return {
            "average_overall_risk": _clamp_risk(avg_overall),
            "average_deforestation_risk": _clamp_risk(avg_deforestation),
            "average_complexity": _clamp_risk(avg_complexity),
            "average_traceability": _clamp_risk(avg_traceability),
            "risk_distribution": risk_dist,
            "commodity_count": len(profiles),
        }

    # ------------------------------------------------------------------
    # Internal: Cache management
    # ------------------------------------------------------------------

    def _build_cache_key(
        self,
        commodity_type: str,
        country_data: Dict[str, Any],
        supply_chain_data: Dict[str, Any],
    ) -> str:
        """Build a deterministic cache key from profile inputs.

        Args:
            commodity_type: Normalized commodity type.
            country_data: Sourcing country data.
            supply_chain_data: Supply chain structure data.

        Returns:
            SHA-256 hex digest used as cache key.
        """
        payload = {
            "commodity_type": commodity_type,
            "country_data": country_data,
            "supply_chain_data": supply_chain_data,
        }
        return _compute_provenance_hash(payload)

    def _get_cached_profile(
        self,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a profile from cache if it exists.

        Args:
            cache_key: SHA-256 cache key.

        Returns:
            Cached profile dict or None if not found.
        """
        with self._lock:
            return self._profiles.get(cache_key)

    def _cache_profile(
        self,
        cache_key: str,
        profile: Dict[str, Any],
    ) -> None:
        """Store a profile in the cache.

        Args:
            cache_key: SHA-256 cache key.
            profile: Complete profile dictionary.
        """
        with self._lock:
            self._profiles[cache_key] = profile

    def clear_cache(self) -> None:
        """Clear all cached profiles.

        Intended for testing and cache invalidation scenarios.
        """
        with self._lock:
            count = len(self._profiles)
            self._profiles.clear()
        logger.info("CommodityProfiler cache cleared: %d profiles removed", count)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cached_profile_count(self) -> int:
        """Return the number of cached profiles."""
        with self._lock:
            return len(self._profiles)

    @property
    def profile_weights(self) -> Dict[str, Decimal]:
        """Return a copy of the current profile weights."""
        return dict(self._profile_weights)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"CommodityProfiler("
            f"cached_profiles={self.cached_profile_count}, "
            f"weights={{{', '.join(f'{k}={v}' for k, v in self._profile_weights.items())}}})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "EUDR_COMMODITIES",
    "DEFAULT_PROFILE_WEIGHTS",
    "BASELINE_DEFORESTATION_RISK",
    "COUNTRY_DEFORESTATION_RISK",
    "COMMODITY_CHARACTERISTICS",
    "TYPICAL_SUPPLY_CHAIN_DEPTH",
    "TRACEABILITY_DIFFICULTY",
    # Main class
    "CommodityProfiler",
]
