# -*- coding: utf-8 -*-
"""
Information Gathering Agent Configuration - AGENT-EUDR-027

Centralized configuration for the Information Gathering Agent covering:
- Database and cache connection settings (PostgreSQL, Redis)
- External database connectors: EU TRACES, CITES, FLEGT/VPA, UN COMTRADE,
  FAO STAT, Global Forest Watch, World Bank WGI, Transparency Intl CPI,
  EU Sanctions, national customs/land registries -- each with base_url,
  rate_limit_rps, timeout, retry, cache_ttl
- Certification bodies: FSC, RSPO, PEFC, Rainforest Alliance, UTZ, EU Organic
- Public data mining: harvest intervals, freshness thresholds, incremental mode
- Supplier aggregation: fuzzy match threshold, dedup, entity resolution algo
- Completeness validation: Article 9 element weights, classification thresholds
- Data normalization: coordinate system, currency, ECB exchange rate URL
- Package assembly: S3 bucket, retention, max size
- Provenance: SHA-256 hash chain settings
- Metrics: Prometheus prefix gl_eudr_iga_
- Rate limiting: 5 tiers (anonymous/basic/standard/premium/admin)
- Circuit breaker: failure threshold, reset timeout, half-open calls
- Batch processing: max concurrent, timeout, batch size

All settings overridable via environment variables with ``GL_EUDR_IGA_`` prefix.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 Information Gathering Agent (GL-EUDR-IGA-027)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENV_PREFIX = "GL_EUDR_IGA_"


def _env(key: str, default: Any = None) -> Any:
    """Read environment variable with GL_EUDR_IGA_ prefix."""
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
# External Database Source Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalSourceConfig:
    """Configuration for a single external database source."""

    name: str
    base_url: str
    rate_limit_rps: int = 10
    timeout_seconds: int = 30
    retry_max: int = 3
    cache_ttl_hours: int = 24
    enabled: bool = True


# ---------------------------------------------------------------------------
# Certification Body Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CertificationBodyConfig:
    """Configuration for a single certification body connector."""

    name: str
    api_url: str
    verification_cache_ttl_hours: int = 24
    batch_size_max: int = 100
    enabled: bool = True


# ---------------------------------------------------------------------------
# Main Config
# ---------------------------------------------------------------------------


@dataclass
class InformationGatheringConfig:
    """Centralized configuration for AGENT-EUDR-027.

    Thread-safe singleton accessed via ``get_config()``.
    All values overridable via GL_EUDR_IGA_ environment variables.
    """

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL",
            "postgresql+asyncpg://gl:gl@localhost:5432/greenlang",
        )
    )
    pool_size: int = field(default_factory=lambda: _env_int("POOL_SIZE", 10))
    pool_timeout: int = field(default_factory=lambda: _env_int("POOL_TIMEOUT", 30))

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = field(
        default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0")
    )
    redis_ttl_seconds: int = field(
        default_factory=lambda: _env_int("REDIS_TTL", 86400)
    )
    redis_key_prefix: str = "eudr_iga:"

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ── External Database Sources ─────────────────────────────────────────
    external_sources: Dict[str, ExternalSourceConfig] = field(default_factory=dict)

    # ── Certification Bodies ──────────────────────────────────────────────
    certification_bodies: Dict[str, CertificationBodyConfig] = field(
        default_factory=dict
    )

    # ── Public Data Mining ────────────────────────────────────────────────
    harvest_interval_hours: int = field(
        default_factory=lambda: _env_int("HARVEST_INTERVAL_HOURS", 24)
    )
    freshness_threshold_hours: int = field(
        default_factory=lambda: _env_int("FRESHNESS_THRESHOLD_HOURS", 48)
    )
    incremental_updates_enabled: bool = field(
        default_factory=lambda: _env_bool("INCREMENTAL_UPDATES", True)
    )

    # ── Supplier Aggregation ──────────────────────────────────────────────
    fuzzy_match_threshold: float = field(
        default_factory=lambda: _env_float("FUZZY_MATCH_THRESHOLD", 0.85)
    )
    dedup_enabled: bool = field(
        default_factory=lambda: _env_bool("DEDUP_ENABLED", True)
    )
    max_supplier_batch_size: int = field(
        default_factory=lambda: _env_int("MAX_SUPPLIER_BATCH", 10000)
    )
    entity_resolution_algorithm: str = field(
        default_factory=lambda: _env("ENTITY_RESOLUTION_ALGO", "jaro_winkler")
    )

    # ── Completeness Validation ───────────────────────────────────────────
    element_weights: Dict[str, Decimal] = field(default_factory=dict)
    insufficient_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("INSUFFICIENT_THRESHOLD", "60")
    )
    partial_threshold: Decimal = field(
        default_factory=lambda: _env_decimal("PARTIAL_THRESHOLD", "90")
    )
    simplified_dd_enabled: bool = field(
        default_factory=lambda: _env_bool("SIMPLIFIED_DD_ENABLED", True)
    )

    # ── Data Normalization ────────────────────────────────────────────────
    default_coordinate_system: str = "WGS84"
    default_currency: str = "EUR"
    ecb_exchange_rate_url: str = field(
        default_factory=lambda: _env(
            "ECB_EXCHANGE_RATE_URL",
            "https://data.ecb.europa.eu/data-detail/EXR",
        )
    )

    # ── Package Assembly ──────────────────────────────────────────────────
    s3_bucket: str = field(
        default_factory=lambda: _env("S3_BUCKET", "greenlang-eudr-packages")
    )
    s3_prefix: str = field(
        default_factory=lambda: _env("S3_PREFIX", "information-packages/")
    )
    retention_days: int = field(
        default_factory=lambda: _env_int("RETENTION_DAYS", 1825)
    )
    max_package_size_mb: int = field(
        default_factory=lambda: _env_int("MAX_PACKAGE_SIZE_MB", 500)
    )

    # ── Provenance ────────────────────────────────────────────────────────
    provenance_enabled: bool = field(
        default_factory=lambda: _env_bool("PROVENANCE_ENABLED", True)
    )
    provenance_algorithm: str = "sha256"
    provenance_genesis_hash: str = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics_enabled: bool = field(
        default_factory=lambda: _env_bool("METRICS_ENABLED", True)
    )
    metrics_prefix: str = "gl_eudr_iga_"

    # ── Rate Limiting ─────────────────────────────────────────────────────
    rate_limit_anonymous: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ANONYMOUS", 10)
    )
    rate_limit_basic: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_BASIC", 30)
    )
    rate_limit_standard: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_STANDARD", 60)
    )
    rate_limit_premium: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PREMIUM", 120)
    )
    rate_limit_admin: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_ADMIN", 300)
    )

    # ── Circuit Breaker ───────────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("CB_FAILURE_THRESHOLD", 5)
    )
    circuit_breaker_reset_timeout: int = field(
        default_factory=lambda: _env_int("CB_RESET_TIMEOUT", 60)
    )
    circuit_breaker_half_open_max: int = field(
        default_factory=lambda: _env_int("CB_HALF_OPEN_MAX", 3)
    )

    # ── Batch Processing ──────────────────────────────────────────────────
    max_concurrent: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT", 50)
    )
    batch_timeout_seconds: int = field(
        default_factory=lambda: _env_int("BATCH_TIMEOUT", 300)
    )

    def __post_init__(self) -> None:
        """Initialize default external sources and certification bodies."""
        if not self.external_sources:
            self.external_sources = {
                "eu_traces": ExternalSourceConfig(
                    name="EU TRACES",
                    base_url=_env("EU_TRACES_URL", "https://webgate.ec.europa.eu/tracesnt/api"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=12,
                ),
                "cites": ExternalSourceConfig(
                    name="CITES Trade Database",
                    base_url=_env("CITES_URL", "https://trade.cites.org/api"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=720,
                ),
                "flegt_vpa": ExternalSourceConfig(
                    name="FLEGT/VPA System",
                    base_url=_env("FLEGT_URL", "https://ec.europa.eu/environment/forests/flegt/api"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=168,
                ),
                "un_comtrade": ExternalSourceConfig(
                    name="UN COMTRADE",
                    base_url=_env("COMTRADE_URL", "https://comtradeapi.un.org/data/v1"),
                    rate_limit_rps=3, timeout_seconds=60, cache_ttl_hours=720,
                ),
                "fao_stat": ExternalSourceConfig(
                    name="FAO STAT",
                    base_url=_env("FAO_URL", "https://fenixservices.fao.org/faostat/api/v1"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=720,
                ),
                "global_forest_watch": ExternalSourceConfig(
                    name="Global Forest Watch",
                    base_url=_env("GFW_URL", "https://data-api.globalforestwatch.org"),
                    rate_limit_rps=10, timeout_seconds=30, cache_ttl_hours=168,
                ),
                "world_bank_wgi": ExternalSourceConfig(
                    name="World Bank WGI",
                    base_url=_env("WGI_URL", "https://api.worldbank.org/v2"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=8760,
                ),
                "transparency_cpi": ExternalSourceConfig(
                    name="Transparency International CPI",
                    base_url=_env("CPI_URL", "https://www.transparency.org/api"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=8760,
                ),
                "eu_sanctions": ExternalSourceConfig(
                    name="EU Sanctions Lists",
                    base_url=_env("SANCTIONS_URL", "https://webgate.ec.europa.eu/fsd/fsf/api"),
                    rate_limit_rps=5, timeout_seconds=30, cache_ttl_hours=24,
                ),
                "national_customs": ExternalSourceConfig(
                    name="National Customs Registries",
                    base_url=_env("CUSTOMS_URL", ""),
                    rate_limit_rps=3, timeout_seconds=60, cache_ttl_hours=24,
                    enabled=False,
                ),
                "national_land_registry": ExternalSourceConfig(
                    name="National Land Registries",
                    base_url=_env("LAND_REGISTRY_URL", ""),
                    rate_limit_rps=2, timeout_seconds=60, cache_ttl_hours=720,
                    enabled=False,
                ),
            }

        if not self.certification_bodies:
            self.certification_bodies = {
                "fsc": CertificationBodyConfig(
                    name="FSC",
                    api_url=_env("FSC_API_URL", "https://info.fsc.org/api"),
                    verification_cache_ttl_hours=24, batch_size_max=100,
                ),
                "rspo": CertificationBodyConfig(
                    name="RSPO",
                    api_url=_env("RSPO_API_URL", "https://rspo.org/palmtrace/api"),
                    verification_cache_ttl_hours=24, batch_size_max=100,
                ),
                "pefc": CertificationBodyConfig(
                    name="PEFC",
                    api_url=_env("PEFC_API_URL", "https://www.pefc.org/find-certified/api"),
                    verification_cache_ttl_hours=24, batch_size_max=100,
                ),
                "rainforest_alliance": CertificationBodyConfig(
                    name="Rainforest Alliance",
                    api_url=_env("RA_API_URL", "https://www.rainforest-alliance.org/api"),
                    verification_cache_ttl_hours=24, batch_size_max=50,
                ),
                "utz": CertificationBodyConfig(
                    name="UTZ/SAN",
                    api_url=_env("UTZ_API_URL", "https://www.utz.org/api"),
                    verification_cache_ttl_hours=24, batch_size_max=100,
                ),
                "eu_organic": CertificationBodyConfig(
                    name="EU Organic",
                    api_url=_env("EU_ORGANIC_URL", "https://ec.europa.eu/agriculture/ofis/api"),
                    verification_cache_ttl_hours=48, batch_size_max=100,
                ),
            }

        if not self.element_weights:
            self.element_weights = {
                "product_description": Decimal("0.10"),
                "quantity": Decimal("0.10"),
                "country_of_production": Decimal("0.10"),
                "geolocation": Decimal("0.10"),
                "production_date_range": Decimal("0.10"),
                "supplier_identification": Decimal("0.10"),
                "buyer_identification": Decimal("0.10"),
                "deforestation_free_evidence": Decimal("0.10"),
                "legal_production_evidence": Decimal("0.10"),
                "supply_chain_information": Decimal("0.10"),
            }

        logger.info(
            "InformationGatheringConfig initialized: "
            "%d external sources, %d certification bodies, "
            "%d Article 9 elements",
            len(self.external_sources),
            len(self.certification_bodies),
            len(self.element_weights),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config_instance: Optional[InformationGatheringConfig] = None
_config_lock = threading.Lock()


def get_config() -> InformationGatheringConfig:
    """Return the thread-safe singleton configuration instance.

    Returns:
        InformationGatheringConfig singleton.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = InformationGatheringConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the singleton (for testing only)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
