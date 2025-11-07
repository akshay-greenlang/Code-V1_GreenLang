"""
Factor Broker Configuration Management
GL-VCCI Scope 3 Platform

Configuration management for Factor Broker service including environment variables,
source priorities, and service settings.

Version: 1.0.0
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class GWPStandard(str, Enum):
    """GWP (Global Warming Potential) standards supported."""
    AR5 = "AR5"  # IPCC Fifth Assessment Report (2014)
    AR6 = "AR6"  # IPCC Sixth Assessment Report (2021)


class SourceType(str, Enum):
    """Types of emission factor data sources."""
    ECOINVENT = "ecoinvent"
    DESNZ_UK = "desnz_uk"
    EPA_US = "epa_us"
    PROXY = "proxy"


@dataclass
class SourceConfig:
    """
    Configuration for a single data source.

    Attributes:
        name: Source name
        enabled: Whether source is enabled
        priority: Priority in cascade (1=highest)
        api_endpoint: API endpoint URL
        api_key: API key for authentication (if required)
        rate_limit: Rate limit in requests per minute
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_backoff_base: Base delay for exponential backoff (seconds)
    """
    name: SourceType
    enabled: bool = True
    priority: int = 999
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_base: float = 1.0


@dataclass
class CacheConfig:
    """
    Configuration for Redis caching.

    Attributes:
        enabled: Whether caching is enabled
        redis_host: Redis server host
        redis_port: Redis server port
        redis_db: Redis database number
        redis_password: Redis password (if required)
        ttl_seconds: Time-to-live for cache entries (24h for ecoinvent compliance)
        max_size_mb: Maximum cache size in megabytes
        key_prefix: Prefix for all cache keys
    """
    enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    ttl_seconds: int = 86400  # 24 hours (ecoinvent license compliance)
    max_size_mb: int = 500
    key_prefix: str = "factor_broker"


@dataclass
class PerformanceConfig:
    """
    Performance targets and thresholds.

    Attributes:
        target_p50_latency_ms: Target 50th percentile latency
        target_p95_latency_ms: Target 95th percentile latency
        target_p99_latency_ms: Target 99th percentile latency
        target_cache_hit_rate: Target cache hit rate (0-1)
        target_requests_per_second: Target throughput
        max_requests_per_second: Maximum throughput
    """
    target_p50_latency_ms: float = 10.0
    target_p95_latency_ms: float = 50.0
    target_p99_latency_ms: float = 100.0
    target_cache_hit_rate: float = 0.85
    target_requests_per_second: int = 1000
    max_requests_per_second: int = 5000


@dataclass
class ProxyConfig:
    """
    Configuration for proxy factor calculation.

    Attributes:
        enabled: Whether proxy calculation is enabled
        method: Proxy calculation method (category_average, industry_estimate)
        quality_score: Quality score assigned to proxy factors (0-100)
        uncertainty: Uncertainty percentage for proxy factors (0-1)
        flag_in_response: Whether to flag proxy factors in response
        min_category_samples: Minimum samples needed for category average
    """
    enabled: bool = True
    method: str = "category_average"
    quality_score: int = 50
    uncertainty: float = 0.50  # 50%
    flag_in_response: bool = True
    min_category_samples: int = 3


@dataclass
class FactorBrokerConfig:
    """
    Main configuration class for Factor Broker service.

    This class consolidates all configuration settings and provides
    methods to load from environment variables.
    """
    # Data sources
    sources: Dict[SourceType, SourceConfig] = field(default_factory=dict)

    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Performance configuration
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Proxy configuration
    proxy: ProxyConfig = field(default_factory=ProxyConfig)

    # Default GWP standard
    default_gwp_standard: GWPStandard = GWPStandard.AR6

    # Source cascade order
    cascade_order: List[SourceType] = field(default_factory=list)

    # License compliance
    license_compliance_mode: bool = True

    def __post_init__(self):
        """Initialize default sources and cascade order if not provided."""
        if not self.sources:
            self._init_default_sources()

        if not self.cascade_order:
            self._init_cascade_order()

    def _init_default_sources(self):
        """Initialize default data source configurations."""
        self.sources = {
            SourceType.ECOINVENT: SourceConfig(
                name=SourceType.ECOINVENT,
                enabled=True,
                priority=1,
                api_endpoint=os.getenv("ECOINVENT_API_ENDPOINT"),
                api_key=os.getenv("ECOINVENT_API_KEY"),
                rate_limit=1000,
                timeout_seconds=30,
                max_retries=3,
                retry_backoff_base=1.0
            ),
            SourceType.DESNZ_UK: SourceConfig(
                name=SourceType.DESNZ_UK,
                enabled=True,
                priority=2,
                api_endpoint=os.getenv(
                    "DESNZ_API_ENDPOINT",
                    "https://api.gov.uk/desnz/emission-factors"
                ),
                api_key=None,  # No API key required
                rate_limit=1000,
                timeout_seconds=30,
                max_retries=3,
                retry_backoff_base=1.0
            ),
            SourceType.EPA_US: SourceConfig(
                name=SourceType.EPA_US,
                enabled=True,
                priority=3,
                api_endpoint=os.getenv(
                    "EPA_API_ENDPOINT",
                    "https://api.epa.gov/easey/emission-factors"
                ),
                api_key=None,  # No API key required
                rate_limit=1000,
                timeout_seconds=30,
                max_retries=3,
                retry_backoff_base=1.0
            ),
            SourceType.PROXY: SourceConfig(
                name=SourceType.PROXY,
                enabled=True,
                priority=4,
                api_endpoint=None,  # Local calculation
                api_key=None,
                rate_limit=0,  # No limit for local calculation
                timeout_seconds=5,
                max_retries=0
            )
        }

    def _init_cascade_order(self):
        """Initialize cascade order based on source priorities."""
        # Sort sources by priority
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: x[1].priority
        )

        self.cascade_order = [
            source_type for source_type, config in sorted_sources
            if config.enabled
        ]

    @classmethod
    def from_env(cls) -> "FactorBrokerConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            ECOINVENT_API_ENDPOINT: ecoinvent API endpoint
            ECOINVENT_API_KEY: ecoinvent API key
            DESNZ_API_ENDPOINT: DESNZ API endpoint
            EPA_API_ENDPOINT: EPA API endpoint
            REDIS_HOST: Redis host
            REDIS_PORT: Redis port
            REDIS_DB: Redis database number
            REDIS_PASSWORD: Redis password
            CACHE_TTL_SECONDS: Cache TTL in seconds
            DEFAULT_GWP_STANDARD: Default GWP standard (AR5 or AR6)
            LICENSE_COMPLIANCE_MODE: Enable license compliance checks

        Returns:
            FactorBrokerConfig instance
        """
        # Cache configuration
        cache_config = CacheConfig(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
            max_size_mb=int(os.getenv("CACHE_MAX_SIZE_MB", "500")),
            key_prefix=os.getenv("CACHE_KEY_PREFIX", "factor_broker")
        )

        # Performance configuration
        performance_config = PerformanceConfig(
            target_p50_latency_ms=float(os.getenv("TARGET_P50_LATENCY_MS", "10.0")),
            target_p95_latency_ms=float(os.getenv("TARGET_P95_LATENCY_MS", "50.0")),
            target_p99_latency_ms=float(os.getenv("TARGET_P99_LATENCY_MS", "100.0")),
            target_cache_hit_rate=float(os.getenv("TARGET_CACHE_HIT_RATE", "0.85")),
            target_requests_per_second=int(os.getenv("TARGET_RPS", "1000")),
            max_requests_per_second=int(os.getenv("MAX_RPS", "5000"))
        )

        # Proxy configuration
        proxy_config = ProxyConfig(
            enabled=os.getenv("PROXY_ENABLED", "true").lower() == "true",
            method=os.getenv("PROXY_METHOD", "category_average"),
            quality_score=int(os.getenv("PROXY_QUALITY_SCORE", "50")),
            uncertainty=float(os.getenv("PROXY_UNCERTAINTY", "0.50")),
            flag_in_response=os.getenv("PROXY_FLAG_IN_RESPONSE", "true").lower() == "true",
            min_category_samples=int(os.getenv("PROXY_MIN_CATEGORY_SAMPLES", "3"))
        )

        # Default GWP standard
        default_gwp_standard = GWPStandard(
            os.getenv("DEFAULT_GWP_STANDARD", "AR6")
        )

        # License compliance mode
        license_compliance_mode = (
            os.getenv("LICENSE_COMPLIANCE_MODE", "true").lower() == "true"
        )

        return cls(
            cache=cache_config,
            performance=performance_config,
            proxy=proxy_config,
            default_gwp_standard=default_gwp_standard,
            license_compliance_mode=license_compliance_mode
        )

    def get_source_config(self, source_type: SourceType) -> Optional[SourceConfig]:
        """
        Get configuration for a specific source.

        Args:
            source_type: Type of data source

        Returns:
            SourceConfig if found, None otherwise
        """
        return self.sources.get(source_type)

    def is_source_enabled(self, source_type: SourceType) -> bool:
        """
        Check if a source is enabled.

        Args:
            source_type: Type of data source

        Returns:
            True if source is enabled, False otherwise
        """
        config = self.get_source_config(source_type)
        return config.enabled if config else False

    def get_cascade_order(self) -> List[SourceType]:
        """
        Get cascade order for source fallback.

        Returns:
            List of source types in priority order
        """
        return self.cascade_order

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check ecoinvent configuration
        ecoinvent_config = self.get_source_config(SourceType.ECOINVENT)
        if ecoinvent_config and ecoinvent_config.enabled:
            if not ecoinvent_config.api_endpoint:
                errors.append("ECOINVENT_API_ENDPOINT is required when ecoinvent is enabled")
            if not ecoinvent_config.api_key:
                errors.append("ECOINVENT_API_KEY is required when ecoinvent is enabled")

        # Check cache TTL for license compliance
        if self.license_compliance_mode and self.cache.ttl_seconds > 86400:
            errors.append(
                f"Cache TTL ({self.cache.ttl_seconds}s) exceeds ecoinvent "
                f"license limit (86400s / 24 hours)"
            )

        # Check at least one source is enabled
        enabled_sources = [s for s in self.sources.values() if s.enabled]
        if not enabled_sources:
            errors.append("At least one data source must be enabled")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "sources": {
                source_type.value: {
                    "enabled": config.enabled,
                    "priority": config.priority,
                    "rate_limit": config.rate_limit,
                    "timeout_seconds": config.timeout_seconds
                }
                for source_type, config in self.sources.items()
            },
            "cache": {
                "enabled": self.cache.enabled,
                "redis_host": self.cache.redis_host,
                "redis_port": self.cache.redis_port,
                "ttl_seconds": self.cache.ttl_seconds,
                "max_size_mb": self.cache.max_size_mb
            },
            "performance": {
                "target_p95_latency_ms": self.performance.target_p95_latency_ms,
                "target_cache_hit_rate": self.performance.target_cache_hit_rate,
                "target_requests_per_second": self.performance.target_requests_per_second
            },
            "proxy": {
                "enabled": self.proxy.enabled,
                "method": self.proxy.method,
                "quality_score": self.proxy.quality_score
            },
            "default_gwp_standard": self.default_gwp_standard.value,
            "cascade_order": [s.value for s in self.cascade_order],
            "license_compliance_mode": self.license_compliance_mode
        }


# Global configuration instance
_config: Optional[FactorBrokerConfig] = None


def get_config() -> FactorBrokerConfig:
    """
    Get global configuration instance.

    Loads configuration from environment on first call,
    then returns cached instance.

    Returns:
        FactorBrokerConfig instance
    """
    global _config

    if _config is None:
        _config = FactorBrokerConfig.from_env()

    return _config


def reset_config():
    """
    Reset global configuration instance.

    Useful for testing or when configuration needs to be reloaded.
    """
    global _config
    _config = None
