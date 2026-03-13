# -*- coding: utf-8 -*-
"""
Risk Assessment Engine Configuration - AGENT-EUDR-028

Centralized configuration for the Risk Assessment Engine covering:
- Database and cache connection settings (PostgreSQL, Redis)
- Risk dimension weights: 8 configurable weights for composite scoring
  (country, commodity, supplier, deforestation, corruption,
   supply chain complexity, mixing risk, circumvention risk)
- Risk classification thresholds: NEGLIGIBLE/LOW/STANDARD/HIGH/CRITICAL
  with hysteresis buffer to prevent boundary oscillation
- Country benchmarking multipliers: LOW/STANDARD/HIGH per Article 29(2)
- Article 13 simplified due diligence eligibility settings
- Upstream agent URLs: EUDR-016 to EUDR-020 (risk assessment agents)
- Report generation settings: format, decomposition, trend data
- Trend analysis: rolling window, minimum data points
- Manual override: maximum delta, justification requirement
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_rae_

All settings overridable via environment variables with ``GL_EUDR_RAE_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-028 Risk Assessment Engine (GL-EUDR-RAE-028)
Regulation: EU 2023/1115 (EUDR) Articles 10, 13, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENV_PREFIX = "GL_EUDR_RAE_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_RAE_ prefix."""
    return os.environ.get(f"{_ENV_PREFIX}{key}", default)


def _env_int(key: str, default: int) -> int:
    val = _env(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    val = _env(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    val = _env(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_decimal(key: str, default: str) -> Decimal:
    val = _env(key)
    return Decimal(val) if val is not None else Decimal(default)


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class RiskAssessmentEngineConfig:
    """Centralized configuration for AGENT-EUDR-028.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_RAE_ environment variables.
    """

    # -- Database ---------------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL",
            "postgresql+asyncpg://gl:gl@localhost:5432/greenlang",
        )
    )
    pool_size: int = field(default_factory=lambda: _env_int("POOL_SIZE", 10))
    pool_timeout: int = field(default_factory=lambda: _env_int("POOL_TIMEOUT", 30))

    # -- Redis ------------------------------------------------------------------
    redis_url: str = field(
        default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0")
    )
    redis_ttl_seconds: int = field(
        default_factory=lambda: _env_int("REDIS_TTL", 3600)
    )
    redis_key_prefix: str = "eudr_rae:"

    # -- Logging ----------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # -- Risk Dimension Weights -------------------------------------------------
    # Eight configurable weights controlling each dimension's contribution
    # to the composite risk score.  Weights MUST sum to 1.00.
    country_weight: Decimal = field(
        default_factory=lambda: _env_decimal("COUNTRY_WEIGHT", "0.20")
    )
    commodity_weight: Decimal = field(
        default_factory=lambda: _env_decimal("COMMODITY_WEIGHT", "0.15")
    )
    supplier_weight: Decimal = field(
        default_factory=lambda: _env_decimal("SUPPLIER_WEIGHT", "0.20")
    )
    deforestation_weight: Decimal = field(
        default_factory=lambda: _env_decimal("DEFORESTATION_WEIGHT", "0.20")
    )
    corruption_weight: Decimal = field(
        default_factory=lambda: _env_decimal("CORRUPTION_WEIGHT", "0.10")
    )
    supply_chain_complexity_weight: Decimal = field(
        default_factory=lambda: _env_decimal("SUPPLY_CHAIN_COMPLEXITY_WEIGHT", "0.05")
    )
    mixing_risk_weight: Decimal = field(
        default_factory=lambda: _env_decimal("MIXING_RISK_WEIGHT", "0.05")
    )
    circumvention_risk_weight: Decimal = field(
        default_factory=lambda: _env_decimal("CIRCUMVENTION_RISK_WEIGHT", "0.05")
    )

    # -- Risk Classification Thresholds -----------------------------------------
    # Score ranges: [0, negligible) -> NEGLIGIBLE, [negligible, low) -> LOW, etc.
    negligible_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("NEGLIGIBLE_THRESHOLD", "15")
    )
    low_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("LOW_THRESHOLD", "30")
    )
    standard_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("STANDARD_THRESHOLD", "60")
    )
    high_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("HIGH_THRESHOLD", "80")
    )
    critical_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("CRITICAL_THRESHOLD", "100")
    )
    hysteresis_buffer: Decimal = field(
        default_factory=lambda: _env_decimal("HYSTERESIS_BUFFER", "3")
    )

    # -- Country Benchmarking ---------------------------------------------------
    # Multipliers applied to composite score based on EC country benchmarking
    # per Article 29(2).
    benchmark_low_multiplier: Decimal = field(
        default_factory=lambda: _env_decimal("BENCHMARK_LOW_MULTIPLIER", "0.70")
    )
    benchmark_standard_multiplier: Decimal = field(
        default_factory=lambda: _env_decimal("BENCHMARK_STANDARD_MULTIPLIER", "1.00")
    )
    benchmark_high_multiplier: Decimal = field(
        default_factory=lambda: _env_decimal("BENCHMARK_HIGH_MULTIPLIER", "1.50")
    )

    # -- Article 13 Simplified Due Diligence ------------------------------------
    simplified_dd_enabled: bool = field(
        default_factory=lambda: _env_bool("SIMPLIFIED_DD_ENABLED", True)
    )
    simplified_dd_max_score: Decimal = field(
        default_factory=lambda: _env_decimal("SIMPLIFIED_DD_MAX_SCORE", "30")
    )
    simplified_dd_require_all_low: bool = field(
        default_factory=lambda: _env_bool("SIMPLIFIED_DD_REQUIRE_ALL_LOW", True)
    )

    # -- Upstream Agent URLs (EUDR-016 to EUDR-020) -----------------------------
    country_risk_url: str = field(
        default_factory=lambda: _env(
            "COUNTRY_RISK_URL",
            "http://eudr-country-risk:8016/api/v1",
        )
    )
    supplier_risk_url: str = field(
        default_factory=lambda: _env(
            "SUPPLIER_RISK_URL",
            "http://eudr-supplier-risk:8017/api/v1",
        )
    )
    commodity_risk_url: str = field(
        default_factory=lambda: _env(
            "COMMODITY_RISK_URL",
            "http://eudr-commodity-risk:8018/api/v1",
        )
    )
    corruption_index_url: str = field(
        default_factory=lambda: _env(
            "CORRUPTION_INDEX_URL",
            "http://eudr-corruption-index:8019/api/v1",
        )
    )
    deforestation_alert_url: str = field(
        default_factory=lambda: _env(
            "DEFORESTATION_ALERT_URL",
            "http://eudr-deforestation-alert:8020/api/v1",
        )
    )

    # -- Report Generation ------------------------------------------------------
    report_format: str = field(
        default_factory=lambda: _env("REPORT_FORMAT", "json")
    )
    include_factor_decomposition: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_FACTOR_DECOMPOSITION", True)
    )
    include_trend_data: bool = field(
        default_factory=lambda: _env_bool("INCLUDE_TREND_DATA", True)
    )

    # -- Trend Analysis ---------------------------------------------------------
    trend_window_days: int = field(
        default_factory=lambda: _env_int("TREND_WINDOW_DAYS", 365)
    )
    min_data_points: int = field(
        default_factory=lambda: _env_int("MIN_DATA_POINTS", 3)
    )

    # -- Override ---------------------------------------------------------------
    max_override_delta: Decimal = field(
        default_factory=lambda: _env_decimal("MAX_OVERRIDE_DELTA", "30")
    )
    require_justification: bool = field(
        default_factory=lambda: _env_bool("REQUIRE_JUSTIFICATION", True)
    )

    # -- Rate Limiting ----------------------------------------------------------
    rate_limit_anonymous: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ANONYMOUS", 10)
    )
    rate_limit_basic: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_BASIC", 50)
    )
    rate_limit_standard: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_STANDARD", 200)
    )
    rate_limit_premium: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PREMIUM", 1000)
    )
    rate_limit_admin: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ADMIN", 5000)
    )

    # -- Circuit Breaker --------------------------------------------------------
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # -- Batch Processing -------------------------------------------------------
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 50)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

    # -- Provenance -------------------------------------------------------------
    provenance_enabled: bool = field(
        default_factory=lambda: _env_bool("PROVENANCE_ENABLED", True)
    )
    provenance_algorithm: str = "sha256"
    provenance_genesis_hash: str = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )

    # -- Metrics ----------------------------------------------------------------
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_rae_"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate dimension weights sum to 1.00
        total_weight = (
            self.country_weight
            + self.commodity_weight
            + self.supplier_weight
            + self.deforestation_weight
            + self.corruption_weight
            + self.supply_chain_complexity_weight
            + self.mixing_risk_weight
            + self.circumvention_risk_weight
        )
        if total_weight != Decimal("1.00"):
            logger.warning(
                "Risk dimension weights sum to %s (expected 1.00). "
                "Scores will be normalized at runtime.",
                total_weight,
            )

        # Validate threshold ordering
        if not (
            self.negligible_threshold
            < self.low_threshold
            < self.standard_threshold
            < self.high_threshold
            <= self.critical_threshold
        ):
            logger.warning(
                "Risk classification thresholds are not in ascending order: "
                "negligible=%s, low=%s, standard=%s, high=%s, critical=%s",
                self.negligible_threshold,
                self.low_threshold,
                self.standard_threshold,
                self.high_threshold,
                self.critical_threshold,
            )

        logger.info(
            "RiskAssessmentEngineConfig initialized: "
            "8 dimension weights (sum=%s), "
            "5 thresholds (NEGLIGIBLE<%s, LOW<%s, STANDARD<%s, HIGH<%s, CRITICAL<=%s), "
            "hysteresis_buffer=%s, "
            "simplified_dd=%s, benchmark_multipliers=[%s, %s, %s]",
            total_weight,
            self.negligible_threshold,
            self.low_threshold,
            self.standard_threshold,
            self.high_threshold,
            self.critical_threshold,
            self.hysteresis_buffer,
            self.simplified_dd_enabled,
            self.benchmark_low_multiplier,
            self.benchmark_standard_multiplier,
            self.benchmark_high_multiplier,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[RiskAssessmentEngineConfig] = None
_config_lock = threading.Lock()


def get_config() -> RiskAssessmentEngineConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        RiskAssessmentEngineConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = RiskAssessmentEngineConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
