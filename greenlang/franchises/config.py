# -*- coding: utf-8 -*-
"""
Franchises Configuration - AGENT-MRV-027

Thread-safe singleton configuration for GL-MRV-S3-014.
All environment variables prefixed with GL_FRN_.

This module provides comprehensive configuration management for the Franchises
agent (GHG Protocol Scope 3 Category 14), supporting:
- Franchise-specific energy-based calculations (Tier 1)
- Average-data EUI-benchmark calculations (Tier 2)
- Spend-based EEIO calculations (Tier 3)
- Hybrid waterfall method across heterogeneous franchise networks
- 10 franchise types (QSR, hotel, convenience store, retail, etc.)
- 7 emission sources (stationary/mobile combustion, refrigerants, electricity, etc.)
- 7 regulatory frameworks (GHG Protocol, ISO 14064, CSRD, CDP, SBTi, SB 253, GRI)
- Double-counting prevention (DC-FRN-001 through DC-FRN-008)
- SHA-256 provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-014'
    >>> config.franchise_specific.enable_cooking_energy
    True
    >>> config.compliance.default_frameworks
    ['ghg_protocol']

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_FRN_ prefix.  See individual config sections for specific variables.

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: GENERAL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GeneralConfig:
    """
    General configuration for Franchises agent.

    Attributes:
        enabled: Master switch for the agent (GL_FRN_ENABLED)
        debug: Enable debug mode with verbose logging (GL_FRN_DEBUG)
        log_level: Logging level (GL_FRN_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_FRN_AGENT_ID)
        agent_component: Agent component identifier (GL_FRN_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_FRN_VERSION)
        api_prefix: API route prefix (GL_FRN_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_FRN_MAX_BATCH_SIZE)
        default_gwp: Default GWP assessment report version (GL_FRN_DEFAULT_GWP)
        default_ef_source: Default emission factor source (GL_FRN_DEFAULT_EF_SOURCE)

    Example:
        >>> general = GeneralConfig()
        >>> general.agent_id
        'GL-MRV-S3-014'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-014"
    agent_component: str = "AGENT-MRV-027"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/franchises"
    max_batch_size: int = 1000
    default_gwp: str = "AR6"
    default_ef_source: str = "DEFRA_2024"

    def validate(self) -> None:
        """Validate general configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"Invalid log_level '{self.log_level}'. "
                f"Must be one of {valid_log_levels}"
            )

        if not self.agent_id:
            raise ValueError("agent_id cannot be empty")

        if not self.agent_component:
            raise ValueError("agent_component cannot be empty")

        if not self.version:
            raise ValueError("version cannot be empty")

        version_parts = self.version.split(".")
        if len(version_parts) != 3:
            raise ValueError(
                f"Invalid version '{self.version}'. Must follow SemVer (e.g., '1.0.0')"
            )

        if not self.api_prefix:
            raise ValueError("api_prefix cannot be empty")

        if not self.api_prefix.startswith("/"):
            raise ValueError("api_prefix must start with '/'")

        if self.max_batch_size < 1 or self.max_batch_size > 100000:
            raise ValueError("max_batch_size must be between 1 and 100000")

        valid_gwp_versions = {"AR5", "AR6"}
        if self.default_gwp not in valid_gwp_versions:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. "
                f"Must be one of {valid_gwp_versions}"
            )

        valid_ef_sources = {
            "DEFRA_2024", "EPA_2024", "IEA_2024", "EGRID_2024",
            "IPCC_AR6", "CUSTOM",
        }
        if self.default_ef_source not in valid_ef_sources:
            raise ValueError(
                f"Invalid default_ef_source '{self.default_ef_source}'. "
                f"Must be one of {valid_ef_sources}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "debug": self.debug,
            "log_level": self.log_level,
            "agent_id": self.agent_id,
            "agent_component": self.agent_component,
            "version": self.version,
            "api_prefix": self.api_prefix,
            "max_batch_size": self.max_batch_size,
            "default_gwp": self.default_gwp,
            "default_ef_source": self.default_ef_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_FRN_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_FRN_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_FRN_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_FRN_AGENT_ID", "GL-MRV-S3-014"),
            agent_component=os.getenv("GL_FRN_AGENT_COMPONENT", "AGENT-MRV-027"),
            version=os.getenv("GL_FRN_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_FRN_API_PREFIX", "/api/v1/franchises"),
            max_batch_size=int(os.getenv("GL_FRN_MAX_BATCH_SIZE", "1000")),
            default_gwp=os.getenv("GL_FRN_DEFAULT_GWP", "AR6"),
            default_ef_source=os.getenv("GL_FRN_DEFAULT_EF_SOURCE", "DEFRA_2024"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Franchises agent.

    Attributes:
        host: PostgreSQL host (GL_FRN_DB_HOST)
        port: PostgreSQL port (GL_FRN_DB_PORT)
        database: Database name (GL_FRN_DB_DATABASE)
        username: Database username (GL_FRN_DB_USERNAME)
        password: Database password (GL_FRN_DB_PASSWORD)
        schema: Database schema name (GL_FRN_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_FRN_DB_TABLE_PREFIX)
        pool_min: Minimum connection pool size (GL_FRN_DB_POOL_MIN)
        pool_max: Maximum connection pool size (GL_FRN_DB_POOL_MAX)
        ssl_mode: SSL connection mode (GL_FRN_DB_SSL_MODE)
        connection_timeout: Connection timeout in seconds (GL_FRN_DB_CONNECTION_TIMEOUT)

    Example:
        >>> db = DatabaseConfig()
        >>> db.table_prefix
        'gl_frn_'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "franchises_service"
    table_prefix: str = "gl_frn_"
    pool_min: int = 2
    pool_max: int = 10
    ssl_mode: str = "prefer"
    connection_timeout: int = 30

    def validate(self) -> None:
        """Validate database configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.host:
            raise ValueError("host cannot be empty")

        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")

        if not self.database:
            raise ValueError("database cannot be empty")

        if not self.schema:
            raise ValueError("schema cannot be empty")

        if not self.table_prefix:
            raise ValueError("table_prefix cannot be empty")

        if not self.table_prefix.endswith("_"):
            raise ValueError("table_prefix must end with '_'")

        if self.pool_min < 1:
            raise ValueError("pool_min must be >= 1")

        if self.pool_max < 1:
            raise ValueError("pool_max must be >= 1")

        if self.pool_min > self.pool_max:
            raise ValueError("pool_min must be <= pool_max")

        valid_ssl_modes = {
            "disable", "allow", "prefer", "require", "verify-ca", "verify-full",
        }
        if self.ssl_mode not in valid_ssl_modes:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. "
                f"Must be one of {valid_ssl_modes}"
            )

        if self.connection_timeout < 1 or self.connection_timeout > 300:
            raise ValueError(
                "connection_timeout must be between 1 and 300 seconds"
            )

    def get_connection_url(self) -> str:
        """Build PostgreSQL connection URL."""
        auth = f"{self.username}:{self.password}" if self.password else self.username
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
            "schema": self.schema,
            "table_prefix": self.table_prefix,
            "pool_min": self.pool_min,
            "pool_max": self.pool_max,
            "ssl_mode": self.ssl_mode,
            "connection_timeout": self.connection_timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_FRN_DB_HOST", "localhost"),
            port=int(os.getenv("GL_FRN_DB_PORT", "5432")),
            database=os.getenv("GL_FRN_DB_DATABASE", "greenlang"),
            username=os.getenv("GL_FRN_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_FRN_DB_PASSWORD", ""),
            schema=os.getenv("GL_FRN_DB_SCHEMA", "franchises_service"),
            table_prefix=os.getenv("GL_FRN_DB_TABLE_PREFIX", "gl_frn_"),
            pool_min=int(os.getenv("GL_FRN_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_FRN_DB_POOL_MAX", "10")),
            ssl_mode=os.getenv("GL_FRN_DB_SSL_MODE", "prefer"),
            connection_timeout=int(
                os.getenv("GL_FRN_DB_CONNECTION_TIMEOUT", "30")
            ),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Franchises agent.

    Attributes:
        host: Redis host (GL_FRN_REDIS_HOST)
        port: Redis port (GL_FRN_REDIS_PORT)
        db: Redis database index (GL_FRN_REDIS_DB)
        password: Redis password (GL_FRN_REDIS_PASSWORD)
        ssl: Enable SSL connection (GL_FRN_REDIS_SSL)
        prefix: Key prefix for namespacing (GL_FRN_REDIS_PREFIX)
        max_connections: Max connections in pool (GL_FRN_REDIS_MAX_CONNECTIONS)
        socket_timeout: Socket timeout in seconds (GL_FRN_REDIS_SOCKET_TIMEOUT)

    Example:
        >>> redis = RedisConfig()
        >>> redis.prefix
        'gl_frn:'
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    prefix: str = "gl_frn:"
    max_connections: int = 20
    socket_timeout: int = 5

    def validate(self) -> None:
        """Validate Redis configuration values."""
        if not self.host:
            raise ValueError("host cannot be empty")

        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")

        if self.db < 0 or self.db > 15:
            raise ValueError("db must be between 0 and 15")

        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        if not self.prefix.endswith(":"):
            raise ValueError("prefix must end with ':'")

        if self.max_connections < 1 or self.max_connections > 1000:
            raise ValueError("max_connections must be between 1 and 1000")

        if self.socket_timeout < 1 or self.socket_timeout > 60:
            raise ValueError(
                "socket_timeout must be between 1 and 60 seconds"
            )

    def get_connection_url(self) -> str:
        """Build Redis connection URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "ssl": self.ssl,
            "prefix": self.prefix,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_FRN_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_FRN_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_FRN_REDIS_DB", "0")),
            password=os.getenv("GL_FRN_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_FRN_REDIS_SSL", "false").lower() == "true",
            prefix=os.getenv("GL_FRN_REDIS_PREFIX", "gl_frn:"),
            max_connections=int(
                os.getenv("GL_FRN_REDIS_MAX_CONNECTIONS", "20")
            ),
            socket_timeout=int(
                os.getenv("GL_FRN_REDIS_SOCKET_TIMEOUT", "5")
            ),
        )


# =============================================================================
# SECTION 4: FRANCHISE-SPECIFIC CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class FranchiseSpecificConfig:
    """
    Configuration for franchise-specific (Tier 1) calculations.

    Controls the primary-data-based calculation method where individual
    franchise units provide metered energy and refrigerant data.

    Attributes:
        enable_cooking_energy: Track restaurant cooking energy (GL_FRN_FS_ENABLE_COOKING_ENERGY)
        enable_refrigerant_tracking: Track refrigerant leakage (GL_FRN_FS_ENABLE_REFRIGERANT)
        enable_delivery_fleet: Track delivery fleet emissions (GL_FRN_FS_ENABLE_DELIVERY_FLEET)
        enable_wtt: Include well-to-tank emissions (GL_FRN_FS_ENABLE_WTT)
        default_leakage_rate: Default refrigerant leakage rate (GL_FRN_FS_DEFAULT_LEAKAGE_RATE)
        pro_rata_partial_year: Enable pro-rata for partial-year units (GL_FRN_FS_PRO_RATA)
        min_operating_months: Minimum operating months to include (GL_FRN_FS_MIN_OPERATING_MONTHS)

    Example:
        >>> fs = FranchiseSpecificConfig()
        >>> fs.enable_cooking_energy
        True
    """

    enable_cooking_energy: bool = True
    enable_refrigerant_tracking: bool = True
    enable_delivery_fleet: bool = True
    enable_wtt: bool = True
    default_leakage_rate: Decimal = Decimal("0.10")
    pro_rata_partial_year: bool = True
    min_operating_months: int = 1

    def validate(self) -> None:
        """Validate franchise-specific configuration."""
        if self.default_leakage_rate < Decimal("0") or self.default_leakage_rate > Decimal("1"):
            raise ValueError("default_leakage_rate must be between 0 and 1")

        if self.min_operating_months < 0 or self.min_operating_months > 12:
            raise ValueError("min_operating_months must be between 0 and 12")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_cooking_energy": self.enable_cooking_energy,
            "enable_refrigerant_tracking": self.enable_refrigerant_tracking,
            "enable_delivery_fleet": self.enable_delivery_fleet,
            "enable_wtt": self.enable_wtt,
            "default_leakage_rate": str(self.default_leakage_rate),
            "pro_rata_partial_year": self.pro_rata_partial_year,
            "min_operating_months": self.min_operating_months,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FranchiseSpecificConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_leakage_rate" in data_copy:
            data_copy["default_leakage_rate"] = Decimal(
                data_copy["default_leakage_rate"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "FranchiseSpecificConfig":
        """Load from environment variables."""
        return cls(
            enable_cooking_energy=os.getenv(
                "GL_FRN_FS_ENABLE_COOKING_ENERGY", "true"
            ).lower() == "true",
            enable_refrigerant_tracking=os.getenv(
                "GL_FRN_FS_ENABLE_REFRIGERANT", "true"
            ).lower() == "true",
            enable_delivery_fleet=os.getenv(
                "GL_FRN_FS_ENABLE_DELIVERY_FLEET", "true"
            ).lower() == "true",
            enable_wtt=os.getenv(
                "GL_FRN_FS_ENABLE_WTT", "true"
            ).lower() == "true",
            default_leakage_rate=Decimal(
                os.getenv("GL_FRN_FS_DEFAULT_LEAKAGE_RATE", "0.10")
            ),
            pro_rata_partial_year=os.getenv(
                "GL_FRN_FS_PRO_RATA", "true"
            ).lower() == "true",
            min_operating_months=int(
                os.getenv("GL_FRN_FS_MIN_OPERATING_MONTHS", "1")
            ),
        )


# =============================================================================
# SECTION 5: AVERAGE-DATA CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AverageDataConfig:
    """
    Configuration for average-data (Tier 2) calculations.

    Controls the EUI-benchmark-based estimation method.

    Attributes:
        default_climate_zone: Default climate zone (GL_FRN_AD_DEFAULT_CLIMATE_ZONE)
        enable_hotel_class_adjustment: Hotel class energy multipliers (GL_FRN_AD_HOTEL_ADJUST)
        enable_climate_adjustment: Climate zone EUI adjustment (GL_FRN_AD_CLIMATE_ADJUST)
        default_grid_ef_country: Default country for grid EF (GL_FRN_AD_DEFAULT_GRID_COUNTRY)
        enable_currency_conversion: Enable USD conversion (GL_FRN_AD_CURRENCY_CONVERT)
        default_currency: Default input currency (GL_FRN_AD_DEFAULT_CURRENCY)

    Example:
        >>> ad = AverageDataConfig()
        >>> ad.default_climate_zone
        'temperate'
    """

    default_climate_zone: str = "temperate"
    enable_hotel_class_adjustment: bool = True
    enable_climate_adjustment: bool = True
    default_grid_ef_country: str = "US"
    enable_currency_conversion: bool = True
    default_currency: str = "USD"

    def validate(self) -> None:
        """Validate average-data configuration."""
        valid_zones = {"tropical", "arid", "temperate", "continental", "polar"}
        if self.default_climate_zone not in valid_zones:
            raise ValueError(
                f"Invalid default_climate_zone '{self.default_climate_zone}'. "
                f"Must be one of {valid_zones}"
            )

        if len(self.default_grid_ef_country) < 2:
            raise ValueError("default_grid_ef_country must be at least 2 characters")

        valid_currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}
        if self.default_currency not in valid_currencies:
            raise ValueError(
                f"Invalid default_currency '{self.default_currency}'. "
                f"Must be one of {valid_currencies}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_climate_zone": self.default_climate_zone,
            "enable_hotel_class_adjustment": self.enable_hotel_class_adjustment,
            "enable_climate_adjustment": self.enable_climate_adjustment,
            "default_grid_ef_country": self.default_grid_ef_country,
            "enable_currency_conversion": self.enable_currency_conversion,
            "default_currency": self.default_currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AverageDataConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AverageDataConfig":
        """Load from environment variables."""
        return cls(
            default_climate_zone=os.getenv(
                "GL_FRN_AD_DEFAULT_CLIMATE_ZONE", "temperate"
            ),
            enable_hotel_class_adjustment=os.getenv(
                "GL_FRN_AD_HOTEL_ADJUST", "true"
            ).lower() == "true",
            enable_climate_adjustment=os.getenv(
                "GL_FRN_AD_CLIMATE_ADJUST", "true"
            ).lower() == "true",
            default_grid_ef_country=os.getenv(
                "GL_FRN_AD_DEFAULT_GRID_COUNTRY", "US"
            ),
            enable_currency_conversion=os.getenv(
                "GL_FRN_AD_CURRENCY_CONVERT", "true"
            ).lower() == "true",
            default_currency=os.getenv("GL_FRN_AD_DEFAULT_CURRENCY", "USD"),
        )


# =============================================================================
# SECTION 6: SPEND-BASED CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendBasedConfig:
    """
    Configuration for spend-based (Tier 3) calculations.

    Controls the EEIO-factor-based estimation method.

    Attributes:
        default_eeio_source: Default EEIO source (GL_FRN_SB_DEFAULT_EEIO_SOURCE)
        enable_cpi_deflation: Enable CPI year deflation (GL_FRN_SB_ENABLE_CPI)
        enable_margin_removal: Remove profit margins (GL_FRN_SB_ENABLE_MARGIN)
        default_margin_rate: Default profit margin to remove (GL_FRN_SB_DEFAULT_MARGIN)
        base_year: EEIO base year (GL_FRN_SB_BASE_YEAR)

    Example:
        >>> sb = SpendBasedConfig()
        >>> sb.default_eeio_source
        'USEEIO_v2'
    """

    default_eeio_source: str = "USEEIO_v2"
    enable_cpi_deflation: bool = True
    enable_margin_removal: bool = True
    default_margin_rate: Decimal = Decimal("0.15")
    base_year: int = 2022

    def validate(self) -> None:
        """Validate spend-based configuration."""
        valid_sources = {"USEEIO_v2", "EXIOBASE_3", "CUSTOM"}
        if self.default_eeio_source not in valid_sources:
            raise ValueError(
                f"Invalid default_eeio_source '{self.default_eeio_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.default_margin_rate < Decimal("0") or self.default_margin_rate > Decimal("1"):
            raise ValueError("default_margin_rate must be between 0 and 1")

        if self.base_year < 2015 or self.base_year > 2035:
            raise ValueError("base_year must be between 2015 and 2035")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_eeio_source": self.default_eeio_source,
            "enable_cpi_deflation": self.enable_cpi_deflation,
            "enable_margin_removal": self.enable_margin_removal,
            "default_margin_rate": str(self.default_margin_rate),
            "base_year": self.base_year,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendBasedConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_margin_rate" in data_copy:
            data_copy["default_margin_rate"] = Decimal(
                data_copy["default_margin_rate"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendBasedConfig":
        """Load from environment variables."""
        return cls(
            default_eeio_source=os.getenv(
                "GL_FRN_SB_DEFAULT_EEIO_SOURCE", "USEEIO_v2"
            ),
            enable_cpi_deflation=os.getenv(
                "GL_FRN_SB_ENABLE_CPI", "true"
            ).lower() == "true",
            enable_margin_removal=os.getenv(
                "GL_FRN_SB_ENABLE_MARGIN", "true"
            ).lower() == "true",
            default_margin_rate=Decimal(
                os.getenv("GL_FRN_SB_DEFAULT_MARGIN", "0.15")
            ),
            base_year=int(os.getenv("GL_FRN_SB_BASE_YEAR", "2022")),
        )


# =============================================================================
# SECTION 7: HYBRID CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class HybridConfig:
    """
    Configuration for hybrid (waterfall) calculations.

    Controls blending of franchise-specific, average-data, and spend-based
    methods across a heterogeneous franchise network.

    Attributes:
        waterfall_order: Method priority order (GL_FRN_HY_WATERFALL_ORDER)
        data_coverage_target: Target primary data coverage (GL_FRN_HY_COVERAGE_TARGET)
        enable_weighted_dqi: Enable weighted DQI (GL_FRN_HY_WEIGHTED_DQI)
        enable_uncertainty_aggregation: Aggregate uncertainty (GL_FRN_HY_UNCERTAINTY_AGG)

    Example:
        >>> hy = HybridConfig()
        >>> hy.data_coverage_target
        Decimal('0.80')
    """

    waterfall_order: str = "franchise_specific,average_data,spend_based"
    data_coverage_target: Decimal = Decimal("0.80")
    enable_weighted_dqi: bool = True
    enable_uncertainty_aggregation: bool = True

    def validate(self) -> None:
        """Validate hybrid configuration."""
        if self.data_coverage_target < Decimal("0") or self.data_coverage_target > Decimal("1"):
            raise ValueError("data_coverage_target must be between 0 and 1")

        valid_methods = {"franchise_specific", "average_data", "spend_based"}
        for method in self.waterfall_order.split(","):
            if method.strip() not in valid_methods:
                raise ValueError(
                    f"Invalid method '{method.strip()}' in waterfall_order. "
                    f"Must be one of {valid_methods}"
                )

    def get_waterfall_list(self) -> List[str]:
        """Return waterfall order as a list."""
        return [m.strip() for m in self.waterfall_order.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "waterfall_order": self.waterfall_order,
            "data_coverage_target": str(self.data_coverage_target),
            "enable_weighted_dqi": self.enable_weighted_dqi,
            "enable_uncertainty_aggregation": self.enable_uncertainty_aggregation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "data_coverage_target" in data_copy:
            data_copy["data_coverage_target"] = Decimal(
                data_copy["data_coverage_target"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "HybridConfig":
        """Load from environment variables."""
        return cls(
            waterfall_order=os.getenv(
                "GL_FRN_HY_WATERFALL_ORDER",
                "franchise_specific,average_data,spend_based",
            ),
            data_coverage_target=Decimal(
                os.getenv("GL_FRN_HY_COVERAGE_TARGET", "0.80")
            ),
            enable_weighted_dqi=os.getenv(
                "GL_FRN_HY_WEIGHTED_DQI", "true"
            ).lower() == "true",
            enable_uncertainty_aggregation=os.getenv(
                "GL_FRN_HY_UNCERTAINTY_AGG", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 8: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for regulatory framework checks.

    Attributes:
        default_frameworks: Default frameworks to validate (GL_FRN_COMP_FRAMEWORKS)
        require_dc_check: Require DC rule checks (GL_FRN_COMP_REQUIRE_DC)
        strict_boundary_enforcement: Strict company-owned exclusion (GL_FRN_COMP_STRICT_BOUNDARY)
        min_data_coverage_pct: Minimum data coverage percentage (GL_FRN_COMP_MIN_COVERAGE)

    Example:
        >>> comp = ComplianceConfig()
        >>> comp.default_frameworks
        'ghg_protocol'
    """

    default_frameworks: str = "ghg_protocol"
    require_dc_check: bool = True
    strict_boundary_enforcement: bool = True
    min_data_coverage_pct: Decimal = Decimal("50.0")

    def validate(self) -> None:
        """Validate compliance configuration."""
        valid_frameworks = {
            "ghg_protocol", "iso_14064", "csrd_esrs", "cdp",
            "sbti", "sb_253", "gri",
        }
        for fw in self.default_frameworks.split(","):
            if fw.strip() and fw.strip() not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{fw.strip()}'. "
                    f"Must be one of {valid_frameworks}"
                )

        if self.min_data_coverage_pct < Decimal("0") or self.min_data_coverage_pct > Decimal("100"):
            raise ValueError("min_data_coverage_pct must be between 0 and 100")

    def get_frameworks_list(self) -> List[str]:
        """Return default frameworks as list."""
        return [fw.strip() for fw in self.default_frameworks.split(",") if fw.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_frameworks": self.default_frameworks,
            "require_dc_check": self.require_dc_check,
            "strict_boundary_enforcement": self.strict_boundary_enforcement,
            "min_data_coverage_pct": str(self.min_data_coverage_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "min_data_coverage_pct" in data_copy:
            data_copy["min_data_coverage_pct"] = Decimal(
                data_copy["min_data_coverage_pct"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            default_frameworks=os.getenv(
                "GL_FRN_COMP_FRAMEWORKS", "ghg_protocol"
            ),
            require_dc_check=os.getenv(
                "GL_FRN_COMP_REQUIRE_DC", "true"
            ).lower() == "true",
            strict_boundary_enforcement=os.getenv(
                "GL_FRN_COMP_STRICT_BOUNDARY", "true"
            ).lower() == "true",
            min_data_coverage_pct=Decimal(
                os.getenv("GL_FRN_COMP_MIN_COVERAGE", "50.0")
            ),
        )


# =============================================================================
# SECTION 9: EF SOURCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EFSourceConfig:
    """
    Emission factor source configuration.

    Attributes:
        primary_source: Primary EF source (GL_FRN_EF_PRIMARY)
        fallback_source: Fallback EF source (GL_FRN_EF_FALLBACK)
        enable_custom_factors: Allow custom EFs (GL_FRN_EF_ENABLE_CUSTOM)
        grid_ef_source: Grid EF source (GL_FRN_EF_GRID_SOURCE)
        fuel_ef_source: Fuel EF source (GL_FRN_EF_FUEL_SOURCE)

    Example:
        >>> ef = EFSourceConfig()
        >>> ef.primary_source
        'DEFRA_2024'
    """

    primary_source: str = "DEFRA_2024"
    fallback_source: str = "EPA_2024"
    enable_custom_factors: bool = False
    grid_ef_source: str = "EGRID_2024"
    fuel_ef_source: str = "DEFRA_2024"

    def validate(self) -> None:
        """Validate EF source configuration."""
        valid_sources = {
            "DEFRA_2024", "EPA_2024", "IEA_2024", "EGRID_2024",
            "IPCC_AR6", "CUSTOM",
        }
        if self.primary_source not in valid_sources:
            raise ValueError(
                f"Invalid primary_source '{self.primary_source}'. "
                f"Must be one of {valid_sources}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_source": self.primary_source,
            "fallback_source": self.fallback_source,
            "enable_custom_factors": self.enable_custom_factors,
            "grid_ef_source": self.grid_ef_source,
            "fuel_ef_source": self.fuel_ef_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EFSourceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "EFSourceConfig":
        """Load from environment variables."""
        return cls(
            primary_source=os.getenv("GL_FRN_EF_PRIMARY", "DEFRA_2024"),
            fallback_source=os.getenv("GL_FRN_EF_FALLBACK", "EPA_2024"),
            enable_custom_factors=os.getenv(
                "GL_FRN_EF_ENABLE_CUSTOM", "false"
            ).lower() == "true",
            grid_ef_source=os.getenv("GL_FRN_EF_GRID_SOURCE", "EGRID_2024"),
            fuel_ef_source=os.getenv("GL_FRN_EF_FUEL_SOURCE", "DEFRA_2024"),
        )


# =============================================================================
# SECTION 10: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty quantification configuration.

    Attributes:
        default_method: Default uncertainty method (GL_FRN_UNC_METHOD)
        confidence_level: Default confidence level (GL_FRN_UNC_CONFIDENCE)
        monte_carlo_iterations: Number of MC iterations (GL_FRN_UNC_MC_ITERATIONS)

    Example:
        >>> unc = UncertaintyConfig()
        >>> unc.default_method
        'ipcc_tier2'
    """

    default_method: str = "ipcc_tier2"
    confidence_level: Decimal = Decimal("0.95")
    monte_carlo_iterations: int = 10000

    def validate(self) -> None:
        """Validate uncertainty configuration."""
        valid_methods = {"monte_carlo", "analytical", "ipcc_tier2"}
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.confidence_level < Decimal("0.50") or self.confidence_level > Decimal("0.99"):
            raise ValueError("confidence_level must be between 0.50 and 0.99")

        if self.monte_carlo_iterations < 100 or self.monte_carlo_iterations > 1000000:
            raise ValueError("monte_carlo_iterations must be between 100 and 1000000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "confidence_level": str(self.confidence_level),
            "monte_carlo_iterations": self.monte_carlo_iterations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "confidence_level" in data_copy:
            data_copy["confidence_level"] = Decimal(
                data_copy["confidence_level"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        return cls(
            default_method=os.getenv("GL_FRN_UNC_METHOD", "ipcc_tier2"),
            confidence_level=Decimal(
                os.getenv("GL_FRN_UNC_CONFIDENCE", "0.95")
            ),
            monte_carlo_iterations=int(
                os.getenv("GL_FRN_UNC_MC_ITERATIONS", "10000")
            ),
        )


# =============================================================================
# SECTION 11: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache layer configuration.

    Attributes:
        enabled: Enable caching layer (GL_FRN_CACHE_ENABLED)
        ef_ttl_seconds: EF cache TTL (GL_FRN_CACHE_EF_TTL)
        benchmark_ttl_seconds: Benchmark cache TTL (GL_FRN_CACHE_BENCHMARK_TTL)
        max_entries: Maximum cache entries (GL_FRN_CACHE_MAX_ENTRIES)

    Example:
        >>> cache = CacheConfig()
        >>> cache.ef_ttl_seconds
        3600
    """

    enabled: bool = True
    ef_ttl_seconds: int = 3600
    benchmark_ttl_seconds: int = 86400
    max_entries: int = 10000

    def validate(self) -> None:
        """Validate cache configuration."""
        if self.ef_ttl_seconds < 60 or self.ef_ttl_seconds > 604800:
            raise ValueError("ef_ttl_seconds must be between 60 and 604800")

        if self.benchmark_ttl_seconds < 60 or self.benchmark_ttl_seconds > 604800:
            raise ValueError("benchmark_ttl_seconds must be between 60 and 604800")

        if self.max_entries < 100 or self.max_entries > 1000000:
            raise ValueError("max_entries must be between 100 and 1000000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ef_ttl_seconds": self.ef_ttl_seconds,
            "benchmark_ttl_seconds": self.benchmark_ttl_seconds,
            "max_entries": self.max_entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv(
                "GL_FRN_CACHE_ENABLED", "true"
            ).lower() == "true",
            ef_ttl_seconds=int(os.getenv("GL_FRN_CACHE_EF_TTL", "3600")),
            benchmark_ttl_seconds=int(
                os.getenv("GL_FRN_CACHE_BENCHMARK_TTL", "86400")
            ),
            max_entries=int(os.getenv("GL_FRN_CACHE_MAX_ENTRIES", "10000")),
        )


# =============================================================================
# SECTION 12: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    API endpoint configuration.

    Attributes:
        host: API bind host (GL_FRN_API_HOST)
        port: API bind port (GL_FRN_API_PORT)
        workers: Number of workers (GL_FRN_API_WORKERS)
        cors_origins: CORS allowed origins (GL_FRN_API_CORS_ORIGINS)
        rate_limit_rpm: Rate limit in requests per minute (GL_FRN_API_RATE_LIMIT)

    Example:
        >>> api = APIConfig()
        >>> api.port
        8027
    """

    host: str = "0.0.0.0"
    port: int = 8027
    workers: int = 4
    cors_origins: str = "*"
    rate_limit_rpm: int = 100

    def validate(self) -> None:
        """Validate API configuration."""
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")

        if self.workers < 1 or self.workers > 32:
            raise ValueError("workers must be between 1 and 32")

        if self.rate_limit_rpm < 1 or self.rate_limit_rpm > 10000:
            raise ValueError("rate_limit_rpm must be between 1 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "cors_origins": self.cors_origins,
            "rate_limit_rpm": self.rate_limit_rpm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_FRN_API_HOST", "0.0.0.0"),
            port=int(os.getenv("GL_FRN_API_PORT", "8027")),
            workers=int(os.getenv("GL_FRN_API_WORKERS", "4")),
            cors_origins=os.getenv("GL_FRN_API_CORS_ORIGINS", "*"),
            rate_limit_rpm=int(os.getenv("GL_FRN_API_RATE_LIMIT", "100")),
        )


# =============================================================================
# SECTION 13: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance tracking configuration.

    Attributes:
        enabled: Enable provenance tracking (GL_FRN_PROV_ENABLED)
        hash_algorithm: Hash algorithm (GL_FRN_PROV_ALGORITHM)
        enable_merkle: Enable Merkle-style batch hashing (GL_FRN_PROV_MERKLE)
        max_chain_entries: Maximum entries per chain (GL_FRN_PROV_MAX_ENTRIES)

    Example:
        >>> prov = ProvenanceConfig()
        >>> prov.hash_algorithm
        'sha256'
    """

    enabled: bool = True
    hash_algorithm: str = "sha256"
    enable_merkle: bool = True
    max_chain_entries: int = 100

    def validate(self) -> None:
        """Validate provenance configuration."""
        valid_algorithms = {"sha256", "sha512"}
        if self.hash_algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid hash_algorithm '{self.hash_algorithm}'. "
                f"Must be one of {valid_algorithms}"
            )

        if self.max_chain_entries < 10 or self.max_chain_entries > 10000:
            raise ValueError("max_chain_entries must be between 10 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "hash_algorithm": self.hash_algorithm,
            "enable_merkle": self.enable_merkle,
            "max_chain_entries": self.max_chain_entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv(
                "GL_FRN_PROV_ENABLED", "true"
            ).lower() == "true",
            hash_algorithm=os.getenv("GL_FRN_PROV_ALGORITHM", "sha256"),
            enable_merkle=os.getenv(
                "GL_FRN_PROV_MERKLE", "true"
            ).lower() == "true",
            max_chain_entries=int(
                os.getenv("GL_FRN_PROV_MAX_ENTRIES", "100")
            ),
        )


# =============================================================================
# SECTION 14: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Prometheus metrics configuration.

    Attributes:
        enabled: Enable Prometheus metrics (GL_FRN_METRICS_ENABLED)
        prefix: Metrics prefix (GL_FRN_METRICS_PREFIX)
        enable_histograms: Enable histogram metrics (GL_FRN_METRICS_HISTOGRAMS)

    Example:
        >>> met = MetricsConfig()
        >>> met.prefix
        'gl_frn_'
    """

    enabled: bool = True
    prefix: str = "gl_frn_"
    enable_histograms: bool = True

    def validate(self) -> None:
        """Validate metrics configuration."""
        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        if not self.prefix.endswith("_"):
            raise ValueError("prefix must end with '_'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "enable_histograms": self.enable_histograms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv(
                "GL_FRN_METRICS_ENABLED", "true"
            ).lower() == "true",
            prefix=os.getenv("GL_FRN_METRICS_PREFIX", "gl_frn_"),
            enable_histograms=os.getenv(
                "GL_FRN_METRICS_HISTOGRAMS", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 15: HOTEL-SPECIFIC CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class HotelConfig:
    """
    Hotel franchise-specific configuration.

    Attributes:
        default_class_type: Default hotel class (GL_FRN_HOTEL_CLASS)
        laundry_energy_adjustment: Energy adjustment for on-site laundry (GL_FRN_HOTEL_LAUNDRY)
        pool_energy_adjustment: Energy adjustment for on-site pool (GL_FRN_HOTEL_POOL)
        restaurant_energy_adjustment: Energy adjustment for on-site restaurant (GL_FRN_HOTEL_RESTAURANT)
        spa_energy_adjustment: Energy adjustment for on-site spa (GL_FRN_HOTEL_SPA)

    Example:
        >>> hotel = HotelConfig()
        >>> hotel.laundry_energy_adjustment
        Decimal('0.12')
    """

    default_class_type: str = "midscale"
    laundry_energy_adjustment: Decimal = Decimal("0.12")
    pool_energy_adjustment: Decimal = Decimal("0.08")
    restaurant_energy_adjustment: Decimal = Decimal("0.15")
    spa_energy_adjustment: Decimal = Decimal("0.10")

    def validate(self) -> None:
        """Validate hotel configuration."""
        valid_classes = {"economy", "midscale", "upscale", "luxury"}
        if self.default_class_type not in valid_classes:
            raise ValueError(
                f"Invalid default_class_type '{self.default_class_type}'. "
                f"Must be one of {valid_classes}"
            )

        for attr_name in [
            "laundry_energy_adjustment",
            "pool_energy_adjustment",
            "restaurant_energy_adjustment",
            "spa_energy_adjustment",
        ]:
            val = getattr(self, attr_name)
            if val < Decimal("0") or val > Decimal("1"):
                raise ValueError(f"{attr_name} must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_class_type": self.default_class_type,
            "laundry_energy_adjustment": str(self.laundry_energy_adjustment),
            "pool_energy_adjustment": str(self.pool_energy_adjustment),
            "restaurant_energy_adjustment": str(self.restaurant_energy_adjustment),
            "spa_energy_adjustment": str(self.spa_energy_adjustment),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HotelConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "laundry_energy_adjustment", "pool_energy_adjustment",
            "restaurant_energy_adjustment", "spa_energy_adjustment",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "HotelConfig":
        """Load from environment variables."""
        return cls(
            default_class_type=os.getenv("GL_FRN_HOTEL_CLASS", "midscale"),
            laundry_energy_adjustment=Decimal(
                os.getenv("GL_FRN_HOTEL_LAUNDRY", "0.12")
            ),
            pool_energy_adjustment=Decimal(
                os.getenv("GL_FRN_HOTEL_POOL", "0.08")
            ),
            restaurant_energy_adjustment=Decimal(
                os.getenv("GL_FRN_HOTEL_RESTAURANT", "0.15")
            ),
            spa_energy_adjustment=Decimal(
                os.getenv("GL_FRN_HOTEL_SPA", "0.10")
            ),
        )


# =============================================================================
# SECTION 16: QSR-SPECIFIC CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class QSRConfig:
    """
    QSR (quick-service restaurant) franchise-specific configuration.

    Attributes:
        default_cooking_fuel_split: Default cooking fuel % as gas (GL_FRN_QSR_GAS_SPLIT)
        fryer_energy_kwh_per_year: Annual fryer energy (GL_FRN_QSR_FRYER_ENERGY)
        enable_24h_adjustment: Enable 24-hour operation adjustment (GL_FRN_QSR_24H)

    Example:
        >>> qsr = QSRConfig()
        >>> qsr.default_cooking_fuel_split
        Decimal('0.55')
    """

    default_cooking_fuel_split: Decimal = Decimal("0.55")
    fryer_energy_kwh_per_year: Decimal = Decimal("15000")
    enable_24h_adjustment: bool = False

    def validate(self) -> None:
        """Validate QSR configuration."""
        if self.default_cooking_fuel_split < Decimal("0") or self.default_cooking_fuel_split > Decimal("1"):
            raise ValueError("default_cooking_fuel_split must be between 0 and 1")

        if self.fryer_energy_kwh_per_year < Decimal("0"):
            raise ValueError("fryer_energy_kwh_per_year must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_cooking_fuel_split": str(self.default_cooking_fuel_split),
            "fryer_energy_kwh_per_year": str(self.fryer_energy_kwh_per_year),
            "enable_24h_adjustment": self.enable_24h_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QSRConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_cooking_fuel_split", "fryer_energy_kwh_per_year"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "QSRConfig":
        """Load from environment variables."""
        return cls(
            default_cooking_fuel_split=Decimal(
                os.getenv("GL_FRN_QSR_GAS_SPLIT", "0.55")
            ),
            fryer_energy_kwh_per_year=Decimal(
                os.getenv("GL_FRN_QSR_FRYER_ENERGY", "15000")
            ),
            enable_24h_adjustment=os.getenv(
                "GL_FRN_QSR_24H", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 17: CONVENIENCE STORE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ConvenienceStoreConfig:
    """
    Convenience store franchise-specific configuration.

    Attributes:
        default_24h_operation: Default 24-hour operation flag (GL_FRN_CS_24H)
        fuel_station_adjustment: Energy adjustment for co-located fuel station (GL_FRN_CS_FUEL_STATION)
        refrigeration_heavy: Enable high-refrigeration profile (GL_FRN_CS_REFRIG_HEAVY)

    Example:
        >>> cs = ConvenienceStoreConfig()
        >>> cs.fuel_station_adjustment
        Decimal('0.20')
    """

    default_24h_operation: bool = True
    fuel_station_adjustment: Decimal = Decimal("0.20")
    refrigeration_heavy: bool = True

    def validate(self) -> None:
        """Validate convenience store configuration."""
        if self.fuel_station_adjustment < Decimal("0") or self.fuel_station_adjustment > Decimal("1"):
            raise ValueError("fuel_station_adjustment must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_24h_operation": self.default_24h_operation,
            "fuel_station_adjustment": str(self.fuel_station_adjustment),
            "refrigeration_heavy": self.refrigeration_heavy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvenienceStoreConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "fuel_station_adjustment" in data_copy:
            data_copy["fuel_station_adjustment"] = Decimal(
                data_copy["fuel_station_adjustment"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ConvenienceStoreConfig":
        """Load from environment variables."""
        return cls(
            default_24h_operation=os.getenv(
                "GL_FRN_CS_24H", "true"
            ).lower() == "true",
            fuel_station_adjustment=Decimal(
                os.getenv("GL_FRN_CS_FUEL_STATION", "0.20")
            ),
            refrigeration_heavy=os.getenv(
                "GL_FRN_CS_REFRIG_HEAVY", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 18: RETAIL STORE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RetailConfig:
    """
    Retail store franchise-specific configuration.

    Attributes:
        default_floor_area_m2: Default retail floor area (GL_FRN_RETAIL_AREA)
        enable_lighting_adjustment: Enable high-lighting profile (GL_FRN_RETAIL_LIGHTING)

    Example:
        >>> retail = RetailConfig()
        >>> retail.default_floor_area_m2
        Decimal('400')
    """

    default_floor_area_m2: Decimal = Decimal("400")
    enable_lighting_adjustment: bool = True

    def validate(self) -> None:
        """Validate retail configuration."""
        if self.default_floor_area_m2 < Decimal("10") or self.default_floor_area_m2 > Decimal("100000"):
            raise ValueError("default_floor_area_m2 must be between 10 and 100000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_floor_area_m2": str(self.default_floor_area_m2),
            "enable_lighting_adjustment": self.enable_lighting_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetailConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_floor_area_m2" in data_copy:
            data_copy["default_floor_area_m2"] = Decimal(
                data_copy["default_floor_area_m2"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RetailConfig":
        """Load from environment variables."""
        return cls(
            default_floor_area_m2=Decimal(
                os.getenv("GL_FRN_RETAIL_AREA", "400")
            ),
            enable_lighting_adjustment=os.getenv(
                "GL_FRN_RETAIL_LIGHTING", "true"
            ).lower() == "true",
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


class FranchisesConfig:
    """
    Master configuration class for Franchises agent (AGENT-MRV-027).

    Aggregates all 18 configuration sections and provides a unified
    interface for accessing configuration values.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        redis: Redis cache configuration
        franchise_specific: Franchise-specific (Tier 1) configuration
        average_data: Average-data (Tier 2) configuration
        spend_based: Spend-based (Tier 3) configuration
        hybrid: Hybrid waterfall configuration
        compliance: Compliance framework configuration
        ef_source: Emission factor source configuration
        uncertainty: Uncertainty quantification configuration
        cache: Cache layer configuration
        api: API endpoint configuration
        provenance: Provenance tracking configuration
        metrics: Prometheus metrics configuration
        hotel: Hotel franchise configuration
        qsr: QSR franchise configuration
        convenience_store: Convenience store configuration
        retail: Retail store configuration

    Example:
        >>> config = FranchisesConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-014'
    """

    def __init__(
        self,
        general: Optional[GeneralConfig] = None,
        database: Optional[DatabaseConfig] = None,
        redis: Optional[RedisConfig] = None,
        franchise_specific: Optional[FranchiseSpecificConfig] = None,
        average_data: Optional[AverageDataConfig] = None,
        spend_based: Optional[SpendBasedConfig] = None,
        hybrid: Optional[HybridConfig] = None,
        compliance: Optional[ComplianceConfig] = None,
        ef_source: Optional[EFSourceConfig] = None,
        uncertainty: Optional[UncertaintyConfig] = None,
        cache: Optional[CacheConfig] = None,
        api: Optional[APIConfig] = None,
        provenance: Optional[ProvenanceConfig] = None,
        metrics: Optional[MetricsConfig] = None,
        hotel: Optional[HotelConfig] = None,
        qsr: Optional[QSRConfig] = None,
        convenience_store: Optional[ConvenienceStoreConfig] = None,
        retail: Optional[RetailConfig] = None,
    ):
        """Initialize with all configuration sections."""
        self.general = general or GeneralConfig()
        self.database = database or DatabaseConfig()
        self.redis = redis or RedisConfig()
        self.franchise_specific = franchise_specific or FranchiseSpecificConfig()
        self.average_data = average_data or AverageDataConfig()
        self.spend_based = spend_based or SpendBasedConfig()
        self.hybrid = hybrid or HybridConfig()
        self.compliance = compliance or ComplianceConfig()
        self.ef_source = ef_source or EFSourceConfig()
        self.uncertainty = uncertainty or UncertaintyConfig()
        self.cache = cache or CacheConfig()
        self.api = api or APIConfig()
        self.provenance = provenance or ProvenanceConfig()
        self.metrics = metrics or MetricsConfig()
        self.hotel = hotel or HotelConfig()
        self.qsr = qsr or QSRConfig()
        self.convenience_store = convenience_store or ConvenienceStoreConfig()
        self.retail = retail or RetailConfig()

    def validate_all(self) -> List[str]:
        """
        Validate all configuration sections.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        sections = [
            ("general", self.general),
            ("database", self.database),
            ("redis", self.redis),
            ("franchise_specific", self.franchise_specific),
            ("average_data", self.average_data),
            ("spend_based", self.spend_based),
            ("hybrid", self.hybrid),
            ("compliance", self.compliance),
            ("ef_source", self.ef_source),
            ("uncertainty", self.uncertainty),
            ("cache", self.cache),
            ("api", self.api),
            ("provenance", self.provenance),
            ("metrics", self.metrics),
            ("hotel", self.hotel),
            ("qsr", self.qsr),
            ("convenience_store", self.convenience_store),
            ("retail", self.retail),
        ]
        for name, section in sections:
            try:
                section.validate()
            except ValueError as e:
                errors.append(f"[{name}] {e}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert all sections to dictionary."""
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "redis": self.redis.to_dict(),
            "franchise_specific": self.franchise_specific.to_dict(),
            "average_data": self.average_data.to_dict(),
            "spend_based": self.spend_based.to_dict(),
            "hybrid": self.hybrid.to_dict(),
            "compliance": self.compliance.to_dict(),
            "ef_source": self.ef_source.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "cache": self.cache.to_dict(),
            "api": self.api.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": self.metrics.to_dict(),
            "hotel": self.hotel.to_dict(),
            "qsr": self.qsr.to_dict(),
            "convenience_store": self.convenience_store.to_dict(),
            "retail": self.retail.to_dict(),
        }

    @classmethod
    def from_env(cls) -> "FranchisesConfig":
        """Load all sections from environment variables."""
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            franchise_specific=FranchiseSpecificConfig.from_env(),
            average_data=AverageDataConfig.from_env(),
            spend_based=SpendBasedConfig.from_env(),
            hybrid=HybridConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            ef_source=EFSourceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            cache=CacheConfig.from_env(),
            api=APIConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            hotel=HotelConfig.from_env(),
            qsr=QSRConfig.from_env(),
            convenience_store=ConvenienceStoreConfig.from_env(),
            retail=RetailConfig.from_env(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FranchisesConfig":
        """Load all sections from dictionary."""
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})) if "general" in data else None,
            database=DatabaseConfig.from_dict(data.get("database", {})) if "database" in data else None,
            redis=RedisConfig.from_dict(data.get("redis", {})) if "redis" in data else None,
            franchise_specific=FranchiseSpecificConfig.from_dict(data.get("franchise_specific", {})) if "franchise_specific" in data else None,
            average_data=AverageDataConfig.from_dict(data.get("average_data", {})) if "average_data" in data else None,
            spend_based=SpendBasedConfig.from_dict(data.get("spend_based", {})) if "spend_based" in data else None,
            hybrid=HybridConfig.from_dict(data.get("hybrid", {})) if "hybrid" in data else None,
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})) if "compliance" in data else None,
            ef_source=EFSourceConfig.from_dict(data.get("ef_source", {})) if "ef_source" in data else None,
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})) if "uncertainty" in data else None,
            cache=CacheConfig.from_dict(data.get("cache", {})) if "cache" in data else None,
            api=APIConfig.from_dict(data.get("api", {})) if "api" in data else None,
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})) if "provenance" in data else None,
            metrics=MetricsConfig.from_dict(data.get("metrics", {})) if "metrics" in data else None,
            hotel=HotelConfig.from_dict(data.get("hotel", {})) if "hotel" in data else None,
            qsr=QSRConfig.from_dict(data.get("qsr", {})) if "qsr" in data else None,
            convenience_store=ConvenienceStoreConfig.from_dict(data.get("convenience_store", {})) if "convenience_store" in data else None,
            retail=RetailConfig.from_dict(data.get("retail", {})) if "retail" in data else None,
        )


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_config_instance: Optional[FranchisesConfig] = None
_config_lock = threading.RLock()


def get_config() -> FranchisesConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization from environment variables.
    Subsequent calls return the cached instance.

    Returns:
        FranchisesConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-014'
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info("Initializing FranchisesConfig from environment")
                config = FranchisesConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning(
                            "Configuration validation warning: %s", error
                        )
                _config_instance = config
                logger.info(
                    "FranchisesConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: FranchisesConfig) -> None:
    """
    Set the singleton configuration instance manually.

    Args:
        config: FranchisesConfig instance to set

    Raises:
        TypeError: If config is not a FranchisesConfig instance
    """
    global _config_instance

    if not isinstance(config, FranchisesConfig):
        raise TypeError(
            f"config must be a FranchisesConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("FranchisesConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Forces the next call to get_config() to reload from environment.
    Primarily used in testing scenarios.

    Example:
        >>> reset_config()
        >>> config = get_config()  # Reloads from environment
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("FranchisesConfig singleton reset")


# Backward-compatible alias used by test suites
_reset_config = reset_config


def validate_config(config: FranchisesConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate_all()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Config sections
    "GeneralConfig",
    "DatabaseConfig",
    "RedisConfig",
    "FranchiseSpecificConfig",
    "AverageDataConfig",
    "SpendBasedConfig",
    "HybridConfig",
    "ComplianceConfig",
    "EFSourceConfig",
    "UncertaintyConfig",
    "CacheConfig",
    "APIConfig",
    "ProvenanceConfig",
    "MetricsConfig",
    "HotelConfig",
    "QSRConfig",
    "ConvenienceStoreConfig",
    "RetailConfig",
    # Master config
    "FranchisesConfig",
    # Singleton access
    "get_config",
    "set_config",
    "reset_config",
    "_reset_config",
    "validate_config",
]
