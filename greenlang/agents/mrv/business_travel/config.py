# -*- coding: utf-8 -*-
"""
Business Travel Configuration - AGENT-MRV-019

Thread-safe singleton configuration for GL-MRV-S3-006.
All environment variables prefixed with GL_BT_.

This module provides comprehensive configuration management for the Business
Travel agent (GHG Protocol Scope 3 Category 6), supporting:
- Air travel emissions (domestic/short-haul/long-haul, radiative forcing)
- Rail travel emissions (national/international/metro/light rail)
- Road travel emissions (car/taxi/bus/motorcycle, fuel-based and distance-based)
- Hotel accommodation emissions (country-specific, class-adjusted)
- Spend-based fallback calculations (EEIO, CPI deflation, margin removal)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi, GRI, DEFRA)
- Radiative forcing (RF) disclosure and RFI multiplier handling
- Multi-source emission factors (DEFRA, ICAO, EPA, IEA, EEIO)
- Double-counting prevention and boundary enforcement
- Provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-006'
    >>> config.air_travel.default_uplift_factor
    Decimal('0.08')
    >>> config.compliance.require_rf_disclosure
    True

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_BT_ prefix. See individual config sections for specific variables.
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
    General configuration for Business Travel agent.

    Attributes:
        enabled: Master switch for the agent (GL_BT_ENABLED)
        debug: Enable debug mode with verbose logging (GL_BT_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_BT_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_BT_AGENT_ID)
        agent_component: Agent component identifier (GL_BT_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_BT_VERSION)
        api_prefix: API route prefix (GL_BT_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_BT_MAX_BATCH_SIZE)
        default_gwp: Default GWP assessment report version (GL_BT_DEFAULT_GWP)
        default_ef_source: Default emission factor source (GL_BT_DEFAULT_EF_SOURCE)
        default_rf_option: Default radiative forcing option (GL_BT_DEFAULT_RF_OPTION)
        default_uplift_factor: Default distance uplift factor for air travel (GL_BT_DEFAULT_UPLIFT_FACTOR)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-006",
        ...     agent_component="AGENT-MRV-019",
        ...     version="1.0.0",
        ...     api_prefix="/api/v1/business-travel",
        ...     max_batch_size=1000,
        ...     default_gwp="AR5",
        ...     default_ef_source="DEFRA",
        ...     default_rf_option="WITH_RF",
        ...     default_uplift_factor=Decimal("0.08")
        ... )
        >>> general.agent_id
        'GL-MRV-S3-006'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-006"
    agent_component: str = "AGENT-MRV-019"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/business-travel"
    max_batch_size: int = 1000
    default_gwp: str = "AR5"
    default_ef_source: str = "DEFRA"
    default_rf_option: str = "WITH_RF"
    default_uplift_factor: Decimal = Decimal("0.08")

    def validate(self) -> None:
        """
        Validate general configuration values.

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

        # Validate SemVer format (basic check)
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

        valid_gwp_versions = {"AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp_versions:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. "
                f"Must be one of {valid_gwp_versions}"
            )

        valid_ef_sources = {"DEFRA", "ICAO", "EPA", "IEA", "EEIO", "CUSTOM"}
        if self.default_ef_source not in valid_ef_sources:
            raise ValueError(
                f"Invalid default_ef_source '{self.default_ef_source}'. "
                f"Must be one of {valid_ef_sources}"
            )

        valid_rf_options = {"WITH_RF", "WITHOUT_RF", "SEPARATE"}
        if self.default_rf_option not in valid_rf_options:
            raise ValueError(
                f"Invalid default_rf_option '{self.default_rf_option}'. "
                f"Must be one of {valid_rf_options}"
            )

        if self.default_uplift_factor < Decimal("0") or self.default_uplift_factor > Decimal("1"):
            raise ValueError("default_uplift_factor must be between 0 and 1")

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
            "default_rf_option": self.default_rf_option,
            "default_uplift_factor": str(self.default_uplift_factor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_uplift_factor" in data_copy:
            data_copy["default_uplift_factor"] = Decimal(data_copy["default_uplift_factor"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_BT_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_BT_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_BT_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_BT_AGENT_ID", "GL-MRV-S3-006"),
            agent_component=os.getenv("GL_BT_AGENT_COMPONENT", "AGENT-MRV-019"),
            version=os.getenv("GL_BT_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_BT_API_PREFIX", "/api/v1/business-travel"),
            max_batch_size=int(os.getenv("GL_BT_MAX_BATCH_SIZE", "1000")),
            default_gwp=os.getenv("GL_BT_DEFAULT_GWP", "AR5"),
            default_ef_source=os.getenv("GL_BT_DEFAULT_EF_SOURCE", "DEFRA"),
            default_rf_option=os.getenv("GL_BT_DEFAULT_RF_OPTION", "WITH_RF"),
            default_uplift_factor=Decimal(
                os.getenv("GL_BT_DEFAULT_UPLIFT_FACTOR", "0.08")
            ),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Business Travel agent.

    Attributes:
        host: PostgreSQL host (GL_BT_DB_HOST)
        port: PostgreSQL port (GL_BT_DB_PORT)
        database: Database name (GL_BT_DB_DATABASE)
        username: Database username (GL_BT_DB_USERNAME)
        password: Database password (GL_BT_DB_PASSWORD)
        schema: Database schema name (GL_BT_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_BT_DB_TABLE_PREFIX)
        pool_min: Minimum connection pool size (GL_BT_DB_POOL_MIN)
        pool_max: Maximum connection pool size (GL_BT_DB_POOL_MAX)
        ssl_mode: SSL connection mode (GL_BT_DB_SSL_MODE)
        connection_timeout: Connection timeout in seconds (GL_BT_DB_CONNECTION_TIMEOUT)

    Example:
        >>> db = DatabaseConfig(
        ...     host="localhost",
        ...     port=5432,
        ...     database="greenlang",
        ...     username="greenlang",
        ...     password="secret",
        ...     schema="business_travel_service",
        ...     table_prefix="gl_bt_",
        ...     pool_min=2,
        ...     pool_max=10,
        ...     ssl_mode="prefer",
        ...     connection_timeout=30
        ... )
        >>> db.table_prefix
        'gl_bt_'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "business_travel_service"
    table_prefix: str = "gl_bt_"
    pool_min: int = 2
    pool_max: int = 10
    ssl_mode: str = "prefer"
    connection_timeout: int = 30

    def validate(self) -> None:
        """
        Validate database configuration values.

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

        valid_ssl_modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl_modes:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. "
                f"Must be one of {valid_ssl_modes}"
            )

        if self.connection_timeout < 1 or self.connection_timeout > 300:
            raise ValueError("connection_timeout must be between 1 and 300 seconds")

    def get_connection_url(self) -> str:
        """
        Build PostgreSQL connection URL.

        Returns:
            PostgreSQL connection URL string
        """
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
            host=os.getenv("GL_BT_DB_HOST", "localhost"),
            port=int(os.getenv("GL_BT_DB_PORT", "5432")),
            database=os.getenv("GL_BT_DB_DATABASE", "greenlang"),
            username=os.getenv("GL_BT_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_BT_DB_PASSWORD", ""),
            schema=os.getenv("GL_BT_DB_SCHEMA", "business_travel_service"),
            table_prefix=os.getenv("GL_BT_DB_TABLE_PREFIX", "gl_bt_"),
            pool_min=int(os.getenv("GL_BT_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_BT_DB_POOL_MAX", "10")),
            ssl_mode=os.getenv("GL_BT_DB_SSL_MODE", "prefer"),
            connection_timeout=int(os.getenv("GL_BT_DB_CONNECTION_TIMEOUT", "30")),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Business Travel agent.

    Attributes:
        host: Redis host (GL_BT_REDIS_HOST)
        port: Redis port (GL_BT_REDIS_PORT)
        db: Redis database index (GL_BT_REDIS_DB)
        password: Redis password (GL_BT_REDIS_PASSWORD)
        ssl: Enable SSL connection (GL_BT_REDIS_SSL)
        prefix: Key prefix for namespacing (GL_BT_REDIS_PREFIX)
        max_connections: Max connections in pool (GL_BT_REDIS_MAX_CONNECTIONS)
        socket_timeout: Socket timeout in seconds (GL_BT_REDIS_SOCKET_TIMEOUT)

    Example:
        >>> redis = RedisConfig(
        ...     host="localhost",
        ...     port=6379,
        ...     db=0,
        ...     password="",
        ...     ssl=False,
        ...     prefix="gl_bt:",
        ...     max_connections=20,
        ...     socket_timeout=5
        ... )
        >>> redis.prefix
        'gl_bt:'
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    prefix: str = "gl_bt:"
    max_connections: int = 20
    socket_timeout: int = 5

    def validate(self) -> None:
        """
        Validate Redis configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
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
            raise ValueError("socket_timeout must be between 1 and 60 seconds")

    def get_connection_url(self) -> str:
        """
        Build Redis connection URL.

        Returns:
            Redis connection URL string
        """
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
            host=os.getenv("GL_BT_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_BT_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_BT_REDIS_DB", "0")),
            password=os.getenv("GL_BT_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_BT_REDIS_SSL", "false").lower() == "true",
            prefix=os.getenv("GL_BT_REDIS_PREFIX", "gl_bt:"),
            max_connections=int(os.getenv("GL_BT_REDIS_MAX_CONNECTIONS", "20")),
            socket_timeout=int(os.getenv("GL_BT_REDIS_SOCKET_TIMEOUT", "5")),
        )


# =============================================================================
# SECTION 4: AIR TRAVEL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AirTravelConfig:
    """
    Air travel configuration for Business Travel agent.

    This section configures air travel emissions calculations including distance
    uplift factors (accounting for non-great-circle routing), radiative forcing
    (RF) options for high-altitude impacts, and haul distance classification
    thresholds per DEFRA/ICAO methodologies.

    Attributes:
        default_uplift_factor: Default distance uplift factor (GL_BT_AIR_DEFAULT_UPLIFT_FACTOR)
        default_rf_option: Default radiative forcing option (GL_BT_AIR_DEFAULT_RF_OPTION)
        default_rfi_multiplier: Default radiative forcing index multiplier (GL_BT_AIR_DEFAULT_RFI_MULTIPLIER)
        domestic_threshold_km: Max distance for domestic classification (GL_BT_AIR_DOMESTIC_THRESHOLD_KM)
        short_haul_threshold_km: Max distance for short-haul classification (GL_BT_AIR_SHORT_HAUL_THRESHOLD_KM)
        enable_wtt: Enable well-to-tank emissions (GL_BT_AIR_ENABLE_WTT)
        earth_radius_km: Earth radius for Haversine distance calculation (GL_BT_AIR_EARTH_RADIUS_KM)
        default_cabin_class: Default cabin class (GL_BT_AIR_DEFAULT_CABIN_CLASS)
        enable_seating_class_factors: Enable cabin class emission multipliers (GL_BT_AIR_ENABLE_SEATING_CLASS_FACTORS)
        economy_factor: Economy class emission multiplier (GL_BT_AIR_ECONOMY_FACTOR)
        premium_economy_factor: Premium economy class emission multiplier (GL_BT_AIR_PREMIUM_ECONOMY_FACTOR)
        business_factor: Business class emission multiplier (GL_BT_AIR_BUSINESS_FACTOR)
        first_class_factor: First class emission multiplier (GL_BT_AIR_FIRST_CLASS_FACTOR)

    Example:
        >>> air = AirTravelConfig(
        ...     default_uplift_factor=Decimal("0.08"),
        ...     default_rf_option="WITH_RF",
        ...     default_rfi_multiplier=Decimal("1.891"),
        ...     domestic_threshold_km=500,
        ...     short_haul_threshold_km=3700,
        ...     enable_wtt=True,
        ...     earth_radius_km=Decimal("6371.0"),
        ...     default_cabin_class="ECONOMY",
        ...     enable_seating_class_factors=True,
        ...     economy_factor=Decimal("1.0"),
        ...     premium_economy_factor=Decimal("1.6"),
        ...     business_factor=Decimal("2.9"),
        ...     first_class_factor=Decimal("4.0")
        ... )
        >>> air.default_rfi_multiplier
        Decimal('1.891')
    """

    default_uplift_factor: Decimal = Decimal("0.08")
    default_rf_option: str = "WITH_RF"
    default_rfi_multiplier: Decimal = Decimal("1.891")
    domestic_threshold_km: int = 500
    short_haul_threshold_km: int = 3700
    enable_wtt: bool = True
    earth_radius_km: Decimal = Decimal("6371.0")
    default_cabin_class: str = "ECONOMY"
    enable_seating_class_factors: bool = True
    economy_factor: Decimal = Decimal("1.0")
    premium_economy_factor: Decimal = Decimal("1.6")
    business_factor: Decimal = Decimal("2.9")
    first_class_factor: Decimal = Decimal("4.0")

    def validate(self) -> None:
        """
        Validate air travel configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_uplift_factor < Decimal("0") or self.default_uplift_factor > Decimal("1"):
            raise ValueError("default_uplift_factor must be between 0 and 1")

        valid_rf_options = {"WITH_RF", "WITHOUT_RF", "SEPARATE"}
        if self.default_rf_option not in valid_rf_options:
            raise ValueError(
                f"Invalid default_rf_option '{self.default_rf_option}'. "
                f"Must be one of {valid_rf_options}"
            )

        if self.default_rfi_multiplier < Decimal("1") or self.default_rfi_multiplier > Decimal("5"):
            raise ValueError("default_rfi_multiplier must be between 1 and 5")

        if self.domestic_threshold_km < 100 or self.domestic_threshold_km > 2000:
            raise ValueError("domestic_threshold_km must be between 100 and 2000")

        if self.short_haul_threshold_km < 1000 or self.short_haul_threshold_km > 10000:
            raise ValueError("short_haul_threshold_km must be between 1000 and 10000")

        if self.domestic_threshold_km >= self.short_haul_threshold_km:
            raise ValueError(
                "domestic_threshold_km must be less than short_haul_threshold_km"
            )

        if self.earth_radius_km < Decimal("6300") or self.earth_radius_km > Decimal("6400"):
            raise ValueError("earth_radius_km must be between 6300 and 6400")

        valid_cabin_classes = {"ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST", "AVERAGE"}
        if self.default_cabin_class not in valid_cabin_classes:
            raise ValueError(
                f"Invalid default_cabin_class '{self.default_cabin_class}'. "
                f"Must be one of {valid_cabin_classes}"
            )

        if self.economy_factor < Decimal("0.1") or self.economy_factor > Decimal("10"):
            raise ValueError("economy_factor must be between 0.1 and 10")

        if self.premium_economy_factor < Decimal("0.1") or self.premium_economy_factor > Decimal("10"):
            raise ValueError("premium_economy_factor must be between 0.1 and 10")

        if self.business_factor < Decimal("0.1") or self.business_factor > Decimal("10"):
            raise ValueError("business_factor must be between 0.1 and 10")

        if self.first_class_factor < Decimal("0.1") or self.first_class_factor > Decimal("10"):
            raise ValueError("first_class_factor must be between 0.1 and 10")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_uplift_factor": str(self.default_uplift_factor),
            "default_rf_option": self.default_rf_option,
            "default_rfi_multiplier": str(self.default_rfi_multiplier),
            "domestic_threshold_km": self.domestic_threshold_km,
            "short_haul_threshold_km": self.short_haul_threshold_km,
            "enable_wtt": self.enable_wtt,
            "earth_radius_km": str(self.earth_radius_km),
            "default_cabin_class": self.default_cabin_class,
            "enable_seating_class_factors": self.enable_seating_class_factors,
            "economy_factor": str(self.economy_factor),
            "premium_economy_factor": str(self.premium_economy_factor),
            "business_factor": str(self.business_factor),
            "first_class_factor": str(self.first_class_factor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AirTravelConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "default_uplift_factor", "default_rfi_multiplier", "earth_radius_km",
            "economy_factor", "premium_economy_factor", "business_factor",
            "first_class_factor",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "AirTravelConfig":
        """Load from environment variables."""
        return cls(
            default_uplift_factor=Decimal(
                os.getenv("GL_BT_AIR_DEFAULT_UPLIFT_FACTOR", "0.08")
            ),
            default_rf_option=os.getenv("GL_BT_AIR_DEFAULT_RF_OPTION", "WITH_RF"),
            default_rfi_multiplier=Decimal(
                os.getenv("GL_BT_AIR_DEFAULT_RFI_MULTIPLIER", "1.891")
            ),
            domestic_threshold_km=int(
                os.getenv("GL_BT_AIR_DOMESTIC_THRESHOLD_KM", "500")
            ),
            short_haul_threshold_km=int(
                os.getenv("GL_BT_AIR_SHORT_HAUL_THRESHOLD_KM", "3700")
            ),
            enable_wtt=os.getenv("GL_BT_AIR_ENABLE_WTT", "true").lower() == "true",
            earth_radius_km=Decimal(
                os.getenv("GL_BT_AIR_EARTH_RADIUS_KM", "6371.0")
            ),
            default_cabin_class=os.getenv("GL_BT_AIR_DEFAULT_CABIN_CLASS", "ECONOMY"),
            enable_seating_class_factors=os.getenv(
                "GL_BT_AIR_ENABLE_SEATING_CLASS_FACTORS", "true"
            ).lower() == "true",
            economy_factor=Decimal(
                os.getenv("GL_BT_AIR_ECONOMY_FACTOR", "1.0")
            ),
            premium_economy_factor=Decimal(
                os.getenv("GL_BT_AIR_PREMIUM_ECONOMY_FACTOR", "1.6")
            ),
            business_factor=Decimal(
                os.getenv("GL_BT_AIR_BUSINESS_FACTOR", "2.9")
            ),
            first_class_factor=Decimal(
                os.getenv("GL_BT_AIR_FIRST_CLASS_FACTOR", "4.0")
            ),
        )


# =============================================================================
# SECTION 5: RAIL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RailConfig:
    """
    Rail travel configuration for Business Travel agent.

    This section configures rail travel emissions calculations. Rail is
    typically one of the lowest-emission transport modes and supports
    national, international, metro, light rail, and high-speed rail types.

    Attributes:
        default_rail_type: Default rail type (GL_BT_RAIL_DEFAULT_RAIL_TYPE)
        enable_wtt: Enable well-to-tank emissions (GL_BT_RAIL_ENABLE_WTT)
        default_country_code: Default country for EF lookup (GL_BT_RAIL_DEFAULT_COUNTRY_CODE)
        enable_electricity_grid_factor: Use country grid EF for electric rail (GL_BT_RAIL_ENABLE_ELECTRICITY_GRID_FACTOR)
        occupancy_adjustment: Enable passenger occupancy adjustment (GL_BT_RAIL_OCCUPANCY_ADJUSTMENT)
        default_occupancy_rate: Default occupancy rate (GL_BT_RAIL_DEFAULT_OCCUPANCY_RATE)

    Example:
        >>> rail = RailConfig(
        ...     default_rail_type="NATIONAL",
        ...     enable_wtt=True,
        ...     default_country_code="GB",
        ...     enable_electricity_grid_factor=True,
        ...     occupancy_adjustment=False,
        ...     default_occupancy_rate=Decimal("0.60")
        ... )
        >>> rail.default_rail_type
        'NATIONAL'
    """

    default_rail_type: str = "NATIONAL"
    enable_wtt: bool = True
    default_country_code: str = "GB"
    enable_electricity_grid_factor: bool = True
    occupancy_adjustment: bool = False
    default_occupancy_rate: Decimal = Decimal("0.60")

    def validate(self) -> None:
        """
        Validate rail configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_rail_types = {
            "NATIONAL",
            "INTERNATIONAL",
            "METRO",
            "LIGHT_RAIL",
            "HIGH_SPEED",
            "COMMUTER",
            "AVERAGE",
        }
        if self.default_rail_type not in valid_rail_types:
            raise ValueError(
                f"Invalid default_rail_type '{self.default_rail_type}'. "
                f"Must be one of {valid_rail_types}"
            )

        if not self.default_country_code:
            raise ValueError("default_country_code cannot be empty")

        if len(self.default_country_code) < 2 or len(self.default_country_code) > 3:
            raise ValueError(
                "default_country_code must be a 2 or 3 character ISO code"
            )

        if self.default_occupancy_rate < Decimal("0") or self.default_occupancy_rate > Decimal("1"):
            raise ValueError("default_occupancy_rate must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_rail_type": self.default_rail_type,
            "enable_wtt": self.enable_wtt,
            "default_country_code": self.default_country_code,
            "enable_electricity_grid_factor": self.enable_electricity_grid_factor,
            "occupancy_adjustment": self.occupancy_adjustment,
            "default_occupancy_rate": str(self.default_occupancy_rate),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RailConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_occupancy_rate" in data_copy:
            data_copy["default_occupancy_rate"] = Decimal(data_copy["default_occupancy_rate"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RailConfig":
        """Load from environment variables."""
        return cls(
            default_rail_type=os.getenv("GL_BT_RAIL_DEFAULT_RAIL_TYPE", "NATIONAL"),
            enable_wtt=os.getenv("GL_BT_RAIL_ENABLE_WTT", "true").lower() == "true",
            default_country_code=os.getenv("GL_BT_RAIL_DEFAULT_COUNTRY_CODE", "GB"),
            enable_electricity_grid_factor=os.getenv(
                "GL_BT_RAIL_ENABLE_ELECTRICITY_GRID_FACTOR", "true"
            ).lower() == "true",
            occupancy_adjustment=os.getenv(
                "GL_BT_RAIL_OCCUPANCY_ADJUSTMENT", "false"
            ).lower() == "true",
            default_occupancy_rate=Decimal(
                os.getenv("GL_BT_RAIL_DEFAULT_OCCUPANCY_RATE", "0.60")
            ),
        )


# =============================================================================
# SECTION 6: ROAD CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RoadConfig:
    """
    Road travel configuration for Business Travel agent.

    This section configures road travel emissions calculations for cars,
    taxis, buses, and motorcycles. Supports both distance-based and
    fuel-based calculation methods with unit conversion utilities.

    Attributes:
        default_vehicle_type: Default vehicle type (GL_BT_ROAD_DEFAULT_VEHICLE_TYPE)
        default_fuel_type: Default fuel type (GL_BT_ROAD_DEFAULT_FUEL_TYPE)
        enable_wtt: Enable well-to-tank emissions (GL_BT_ROAD_ENABLE_WTT)
        miles_to_km: Miles to kilometers conversion factor (GL_BT_ROAD_MILES_TO_KM)
        gallons_to_litres: US gallons to litres conversion factor (GL_BT_ROAD_GALLONS_TO_LITRES)
        default_vehicle_size: Default vehicle size category (GL_BT_ROAD_DEFAULT_VEHICLE_SIZE)
        enable_congestion_factor: Enable urban congestion emission adjustment (GL_BT_ROAD_ENABLE_CONGESTION_FACTOR)
        default_congestion_factor: Default congestion emission multiplier (GL_BT_ROAD_DEFAULT_CONGESTION_FACTOR)
        enable_ev_emissions: Include EV grid electricity emissions (GL_BT_ROAD_ENABLE_EV_EMISSIONS)

    Example:
        >>> road = RoadConfig(
        ...     default_vehicle_type="CAR_AVERAGE",
        ...     default_fuel_type="PETROL",
        ...     enable_wtt=True,
        ...     miles_to_km=Decimal("1.60934"),
        ...     gallons_to_litres=Decimal("3.78541"),
        ...     default_vehicle_size="MEDIUM",
        ...     enable_congestion_factor=False,
        ...     default_congestion_factor=Decimal("1.15"),
        ...     enable_ev_emissions=True
        ... )
        >>> road.miles_to_km
        Decimal('1.60934')
    """

    default_vehicle_type: str = "CAR_AVERAGE"
    default_fuel_type: str = "PETROL"
    enable_wtt: bool = True
    miles_to_km: Decimal = Decimal("1.60934")
    gallons_to_litres: Decimal = Decimal("3.78541")
    default_vehicle_size: str = "MEDIUM"
    enable_congestion_factor: bool = False
    default_congestion_factor: Decimal = Decimal("1.15")
    enable_ev_emissions: bool = True

    def validate(self) -> None:
        """
        Validate road configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_vehicle_types = {
            "CAR_AVERAGE",
            "CAR_SMALL",
            "CAR_MEDIUM",
            "CAR_LARGE",
            "CAR_EXECUTIVE",
            "TAXI",
            "BUS_LOCAL",
            "BUS_COACH",
            "MOTORCYCLE",
            "VAN",
        }
        if self.default_vehicle_type not in valid_vehicle_types:
            raise ValueError(
                f"Invalid default_vehicle_type '{self.default_vehicle_type}'. "
                f"Must be one of {valid_vehicle_types}"
            )

        valid_fuel_types = {
            "PETROL",
            "DIESEL",
            "HYBRID",
            "PLUGIN_HYBRID",
            "ELECTRIC",
            "CNG",
            "LPG",
            "UNKNOWN",
        }
        if self.default_fuel_type not in valid_fuel_types:
            raise ValueError(
                f"Invalid default_fuel_type '{self.default_fuel_type}'. "
                f"Must be one of {valid_fuel_types}"
            )

        if self.miles_to_km < Decimal("1.5") or self.miles_to_km > Decimal("1.7"):
            raise ValueError("miles_to_km must be between 1.5 and 1.7")

        if self.gallons_to_litres < Decimal("3.5") or self.gallons_to_litres > Decimal("4.6"):
            raise ValueError("gallons_to_litres must be between 3.5 and 4.6")

        valid_vehicle_sizes = {"SMALL", "MEDIUM", "LARGE", "AVERAGE"}
        if self.default_vehicle_size not in valid_vehicle_sizes:
            raise ValueError(
                f"Invalid default_vehicle_size '{self.default_vehicle_size}'. "
                f"Must be one of {valid_vehicle_sizes}"
            )

        if self.default_congestion_factor < Decimal("1.0") or self.default_congestion_factor > Decimal("2.0"):
            raise ValueError("default_congestion_factor must be between 1.0 and 2.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_vehicle_type": self.default_vehicle_type,
            "default_fuel_type": self.default_fuel_type,
            "enable_wtt": self.enable_wtt,
            "miles_to_km": str(self.miles_to_km),
            "gallons_to_litres": str(self.gallons_to_litres),
            "default_vehicle_size": self.default_vehicle_size,
            "enable_congestion_factor": self.enable_congestion_factor,
            "default_congestion_factor": str(self.default_congestion_factor),
            "enable_ev_emissions": self.enable_ev_emissions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["miles_to_km", "gallons_to_litres", "default_congestion_factor"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RoadConfig":
        """Load from environment variables."""
        return cls(
            default_vehicle_type=os.getenv(
                "GL_BT_ROAD_DEFAULT_VEHICLE_TYPE", "CAR_AVERAGE"
            ),
            default_fuel_type=os.getenv("GL_BT_ROAD_DEFAULT_FUEL_TYPE", "PETROL"),
            enable_wtt=os.getenv("GL_BT_ROAD_ENABLE_WTT", "true").lower() == "true",
            miles_to_km=Decimal(
                os.getenv("GL_BT_ROAD_MILES_TO_KM", "1.60934")
            ),
            gallons_to_litres=Decimal(
                os.getenv("GL_BT_ROAD_GALLONS_TO_LITRES", "3.78541")
            ),
            default_vehicle_size=os.getenv(
                "GL_BT_ROAD_DEFAULT_VEHICLE_SIZE", "MEDIUM"
            ),
            enable_congestion_factor=os.getenv(
                "GL_BT_ROAD_ENABLE_CONGESTION_FACTOR", "false"
            ).lower() == "true",
            default_congestion_factor=Decimal(
                os.getenv("GL_BT_ROAD_DEFAULT_CONGESTION_FACTOR", "1.15")
            ),
            enable_ev_emissions=os.getenv(
                "GL_BT_ROAD_ENABLE_EV_EMISSIONS", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 7: HOTEL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class HotelConfig:
    """
    Hotel accommodation configuration for Business Travel agent.

    This section configures hotel stay emissions calculations. Emissions
    are based on per-night factors that vary by country and hotel class
    (star rating), with optional extended stay discounts.

    Attributes:
        default_country_code: Default country code for hotel EF lookup (GL_BT_HOTEL_DEFAULT_COUNTRY_CODE)
        default_hotel_class: Default hotel class/star rating (GL_BT_HOTEL_DEFAULT_HOTEL_CLASS)
        enable_class_adjustment: Enable hotel class emission adjustment (GL_BT_HOTEL_ENABLE_CLASS_ADJUSTMENT)
        extended_stay_threshold_nights: Night threshold for extended stay discount (GL_BT_HOTEL_EXTENDED_STAY_THRESHOLD_NIGHTS)
        extended_stay_discount: Discount factor for extended stays (GL_BT_HOTEL_EXTENDED_STAY_DISCOUNT)
        include_laundry: Include laundry emissions estimate (GL_BT_HOTEL_INCLUDE_LAUNDRY)
        include_meals: Include hotel meals emissions estimate (GL_BT_HOTEL_INCLUDE_MEALS)
        default_meal_factor: Default meal emission factor per night (GL_BT_HOTEL_DEFAULT_MEAL_FACTOR)
        luxury_multiplier: Luxury hotel emission multiplier (GL_BT_HOTEL_LUXURY_MULTIPLIER)
        budget_multiplier: Budget hotel emission multiplier (GL_BT_HOTEL_BUDGET_MULTIPLIER)
        standard_multiplier: Standard hotel emission multiplier (GL_BT_HOTEL_STANDARD_MULTIPLIER)
        premium_multiplier: Premium hotel emission multiplier (GL_BT_HOTEL_PREMIUM_MULTIPLIER)

    Example:
        >>> hotel = HotelConfig(
        ...     default_country_code="GLOBAL",
        ...     default_hotel_class="STANDARD",
        ...     enable_class_adjustment=True,
        ...     extended_stay_threshold_nights=14,
        ...     extended_stay_discount=Decimal("0.85"),
        ...     include_laundry=False,
        ...     include_meals=False,
        ...     default_meal_factor=Decimal("5.0"),
        ...     luxury_multiplier=Decimal("1.58"),
        ...     budget_multiplier=Decimal("0.78"),
        ...     standard_multiplier=Decimal("1.0"),
        ...     premium_multiplier=Decimal("1.32")
        ... )
        >>> hotel.extended_stay_discount
        Decimal('0.85')
    """

    default_country_code: str = "GLOBAL"
    default_hotel_class: str = "STANDARD"
    enable_class_adjustment: bool = True
    extended_stay_threshold_nights: int = 14
    extended_stay_discount: Decimal = Decimal("0.85")
    include_laundry: bool = False
    include_meals: bool = False
    default_meal_factor: Decimal = Decimal("5.0")
    luxury_multiplier: Decimal = Decimal("1.58")
    budget_multiplier: Decimal = Decimal("0.78")
    standard_multiplier: Decimal = Decimal("1.0")
    premium_multiplier: Decimal = Decimal("1.32")

    def validate(self) -> None:
        """
        Validate hotel configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.default_country_code:
            raise ValueError("default_country_code cannot be empty")

        valid_hotel_classes = {"BUDGET", "STANDARD", "PREMIUM", "LUXURY", "AVERAGE"}
        if self.default_hotel_class not in valid_hotel_classes:
            raise ValueError(
                f"Invalid default_hotel_class '{self.default_hotel_class}'. "
                f"Must be one of {valid_hotel_classes}"
            )

        if self.extended_stay_threshold_nights < 1 or self.extended_stay_threshold_nights > 365:
            raise ValueError(
                "extended_stay_threshold_nights must be between 1 and 365"
            )

        if self.extended_stay_discount < Decimal("0.1") or self.extended_stay_discount > Decimal("1.0"):
            raise ValueError("extended_stay_discount must be between 0.1 and 1.0")

        if self.default_meal_factor < Decimal("0"):
            raise ValueError("default_meal_factor must be >= 0")

        if self.luxury_multiplier < Decimal("0.1") or self.luxury_multiplier > Decimal("10"):
            raise ValueError("luxury_multiplier must be between 0.1 and 10")

        if self.budget_multiplier < Decimal("0.1") or self.budget_multiplier > Decimal("10"):
            raise ValueError("budget_multiplier must be between 0.1 and 10")

        if self.standard_multiplier < Decimal("0.1") or self.standard_multiplier > Decimal("10"):
            raise ValueError("standard_multiplier must be between 0.1 and 10")

        if self.premium_multiplier < Decimal("0.1") or self.premium_multiplier > Decimal("10"):
            raise ValueError("premium_multiplier must be between 0.1 and 10")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_country_code": self.default_country_code,
            "default_hotel_class": self.default_hotel_class,
            "enable_class_adjustment": self.enable_class_adjustment,
            "extended_stay_threshold_nights": self.extended_stay_threshold_nights,
            "extended_stay_discount": str(self.extended_stay_discount),
            "include_laundry": self.include_laundry,
            "include_meals": self.include_meals,
            "default_meal_factor": str(self.default_meal_factor),
            "luxury_multiplier": str(self.luxury_multiplier),
            "budget_multiplier": str(self.budget_multiplier),
            "standard_multiplier": str(self.standard_multiplier),
            "premium_multiplier": str(self.premium_multiplier),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HotelConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "extended_stay_discount", "default_meal_factor",
            "luxury_multiplier", "budget_multiplier",
            "standard_multiplier", "premium_multiplier",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "HotelConfig":
        """Load from environment variables."""
        return cls(
            default_country_code=os.getenv(
                "GL_BT_HOTEL_DEFAULT_COUNTRY_CODE", "GLOBAL"
            ),
            default_hotel_class=os.getenv(
                "GL_BT_HOTEL_DEFAULT_HOTEL_CLASS", "STANDARD"
            ),
            enable_class_adjustment=os.getenv(
                "GL_BT_HOTEL_ENABLE_CLASS_ADJUSTMENT", "true"
            ).lower() == "true",
            extended_stay_threshold_nights=int(
                os.getenv("GL_BT_HOTEL_EXTENDED_STAY_THRESHOLD_NIGHTS", "14")
            ),
            extended_stay_discount=Decimal(
                os.getenv("GL_BT_HOTEL_EXTENDED_STAY_DISCOUNT", "0.85")
            ),
            include_laundry=os.getenv(
                "GL_BT_HOTEL_INCLUDE_LAUNDRY", "false"
            ).lower() == "true",
            include_meals=os.getenv(
                "GL_BT_HOTEL_INCLUDE_MEALS", "false"
            ).lower() == "true",
            default_meal_factor=Decimal(
                os.getenv("GL_BT_HOTEL_DEFAULT_MEAL_FACTOR", "5.0")
            ),
            luxury_multiplier=Decimal(
                os.getenv("GL_BT_HOTEL_LUXURY_MULTIPLIER", "1.58")
            ),
            budget_multiplier=Decimal(
                os.getenv("GL_BT_HOTEL_BUDGET_MULTIPLIER", "0.78")
            ),
            standard_multiplier=Decimal(
                os.getenv("GL_BT_HOTEL_STANDARD_MULTIPLIER", "1.0")
            ),
            premium_multiplier=Decimal(
                os.getenv("GL_BT_HOTEL_PREMIUM_MULTIPLIER", "1.32")
            ),
        )


# =============================================================================
# SECTION 8: SPEND CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendConfig:
    """
    Spend-based calculation configuration for Business Travel agent.

    This section configures spend-based emission calculations used as a
    fallback when distance/activity data is unavailable. Includes CPI
    deflation to normalize spending to a base year and optional margin
    removal to isolate the service cost from profit margins.

    Attributes:
        default_currency: Default currency for spend data (GL_BT_SPEND_DEFAULT_CURRENCY)
        base_year: Base year for CPI deflation (GL_BT_SPEND_BASE_YEAR)
        enable_cpi_deflation: Enable CPI deflation adjustment (GL_BT_SPEND_ENABLE_CPI_DEFLATION)
        enable_margin_removal: Enable profit margin removal (GL_BT_SPEND_ENABLE_MARGIN_REMOVAL)
        default_margin_rate: Default profit margin rate (GL_BT_SPEND_DEFAULT_MARGIN_RATE)
        supported_currencies: Comma-separated supported currencies (GL_BT_SPEND_SUPPORTED_CURRENCIES)
        default_eeio_sector: Default EEIO sector for air travel spend (GL_BT_SPEND_DEFAULT_EEIO_SECTOR)
        enable_ppp_adjustment: Enable purchasing power parity adjustment (GL_BT_SPEND_ENABLE_PPP_ADJUSTMENT)

    Example:
        >>> spend = SpendConfig(
        ...     default_currency="USD",
        ...     base_year=2021,
        ...     enable_cpi_deflation=True,
        ...     enable_margin_removal=False,
        ...     default_margin_rate=Decimal("0.15"),
        ...     supported_currencies="USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED",
        ...     default_eeio_sector="AIR_TRANSPORTATION",
        ...     enable_ppp_adjustment=False
        ... )
        >>> spend.default_margin_rate
        Decimal('0.15')
    """

    default_currency: str = "USD"
    base_year: int = 2021
    enable_cpi_deflation: bool = True
    enable_margin_removal: bool = False
    default_margin_rate: Decimal = Decimal("0.15")
    supported_currencies: str = (
        "USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED"
    )
    default_eeio_sector: str = "AIR_TRANSPORTATION"
    enable_ppp_adjustment: bool = False

    def validate(self) -> None:
        """
        Validate spend configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.default_currency:
            raise ValueError("default_currency cannot be empty")

        if len(self.default_currency) != 3:
            raise ValueError("default_currency must be a 3-character ISO 4217 code")

        # Validate default currency is in supported list
        supported = self.get_supported_currencies()
        if self.default_currency not in supported:
            raise ValueError(
                f"default_currency '{self.default_currency}' must be in "
                f"supported_currencies list"
            )

        if self.base_year < 2000 or self.base_year > 2030:
            raise ValueError("base_year must be between 2000 and 2030")

        if self.default_margin_rate < Decimal("0") or self.default_margin_rate > Decimal("1"):
            raise ValueError("default_margin_rate must be between 0 and 1")

        if not self.supported_currencies:
            raise ValueError("supported_currencies cannot be empty")

        valid_eeio_sectors = {
            "AIR_TRANSPORTATION",
            "RAIL_TRANSPORTATION",
            "GROUND_TRANSPORTATION",
            "ACCOMMODATION",
            "TRAVEL_SERVICES",
        }
        if self.default_eeio_sector not in valid_eeio_sectors:
            raise ValueError(
                f"Invalid default_eeio_sector '{self.default_eeio_sector}'. "
                f"Must be one of {valid_eeio_sectors}"
            )

    def get_supported_currencies(self) -> List[str]:
        """Parse supported currencies string into list."""
        return [c.strip() for c in self.supported_currencies.split(",") if c.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_currency": self.default_currency,
            "base_year": self.base_year,
            "enable_cpi_deflation": self.enable_cpi_deflation,
            "enable_margin_removal": self.enable_margin_removal,
            "default_margin_rate": str(self.default_margin_rate),
            "supported_currencies": self.supported_currencies,
            "default_eeio_sector": self.default_eeio_sector,
            "enable_ppp_adjustment": self.enable_ppp_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_margin_rate" in data_copy:
            data_copy["default_margin_rate"] = Decimal(data_copy["default_margin_rate"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendConfig":
        """Load from environment variables."""
        return cls(
            default_currency=os.getenv("GL_BT_SPEND_DEFAULT_CURRENCY", "USD"),
            base_year=int(os.getenv("GL_BT_SPEND_BASE_YEAR", "2021")),
            enable_cpi_deflation=os.getenv(
                "GL_BT_SPEND_ENABLE_CPI_DEFLATION", "true"
            ).lower() == "true",
            enable_margin_removal=os.getenv(
                "GL_BT_SPEND_ENABLE_MARGIN_REMOVAL", "false"
            ).lower() == "true",
            default_margin_rate=Decimal(
                os.getenv("GL_BT_SPEND_DEFAULT_MARGIN_RATE", "0.15")
            ),
            supported_currencies=os.getenv(
                "GL_BT_SPEND_SUPPORTED_CURRENCIES",
                "USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED",
            ),
            default_eeio_sector=os.getenv(
                "GL_BT_SPEND_DEFAULT_EEIO_SECTOR", "AIR_TRANSPORTATION"
            ),
            enable_ppp_adjustment=os.getenv(
                "GL_BT_SPEND_ENABLE_PPP_ADJUSTMENT", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 9: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Business Travel agent.

    This section configures regulatory framework compliance checks for
    Scope 3 Category 6 business travel emissions reporting. Supports
    7 frameworks and enforces radiative forcing disclosure requirements.

    Attributes:
        compliance_frameworks: Enabled frameworks (GL_BT_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict compliance mode (GL_BT_COMPLIANCE_STRICT_MODE)
        materiality_threshold: Materiality threshold percentage (GL_BT_COMPLIANCE_MATERIALITY_THRESHOLD)
        require_rf_disclosure: Require radiative forcing disclosure (GL_BT_COMPLIANCE_REQUIRE_RF_DISCLOSURE)
        double_counting_check: Check for double counting (GL_BT_COMPLIANCE_DOUBLE_COUNTING_CHECK)
        boundary_enforcement: Enforce Scope 3 boundary (GL_BT_COMPLIANCE_BOUNDARY_ENFORCEMENT)
        require_data_quality: Require data quality scoring (GL_BT_COMPLIANCE_REQUIRE_DATA_QUALITY)
        min_data_quality_score: Minimum acceptable DQI score (GL_BT_COMPLIANCE_MIN_DATA_QUALITY_SCORE)

    Example:
        >>> compliance = ComplianceConfig(
        ...     compliance_frameworks="GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,DEFRA_BEIS",
        ...     strict_mode=False,
        ...     materiality_threshold=Decimal("0.01"),
        ...     require_rf_disclosure=True,
        ...     double_counting_check=True,
        ...     boundary_enforcement=True,
        ...     require_data_quality=True,
        ...     min_data_quality_score=Decimal("2.0")
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E1', 'CDP', 'SBTI', 'GRI', 'DEFRA_BEIS']
    """

    compliance_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,DEFRA_BEIS"
    )
    strict_mode: bool = False
    materiality_threshold: Decimal = Decimal("0.01")
    require_rf_disclosure: bool = True
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    require_data_quality: bool = True
    min_data_quality_score: Decimal = Decimal("2.0")

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL_SCOPE3",
            "ISO_14064",
            "CSRD_ESRS_E1",
            "CDP",
            "SBTI",
            "GRI",
            "DEFRA_BEIS",
        }

        frameworks = self.get_frameworks()
        if not frameworks:
            raise ValueError("At least one compliance framework must be enabled")

        for framework in frameworks:
            if framework not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{framework}'. Must be one of {valid_frameworks}"
                )

        if self.materiality_threshold < Decimal("0") or self.materiality_threshold > Decimal("1"):
            raise ValueError("materiality_threshold must be between 0 and 1")

        if self.min_data_quality_score < Decimal("1") or self.min_data_quality_score > Decimal("5"):
            raise ValueError("min_data_quality_score must be between 1 and 5")

    def get_frameworks(self) -> List[str]:
        """Parse compliance frameworks string into list."""
        return [f.strip() for f in self.compliance_frameworks.split(",") if f.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compliance_frameworks": self.compliance_frameworks,
            "strict_mode": self.strict_mode,
            "materiality_threshold": str(self.materiality_threshold),
            "require_rf_disclosure": self.require_rf_disclosure,
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "require_data_quality": self.require_data_quality,
            "min_data_quality_score": str(self.min_data_quality_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["materiality_threshold", "min_data_quality_score"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            compliance_frameworks=os.getenv(
                "GL_BT_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,DEFRA_BEIS",
            ),
            strict_mode=os.getenv(
                "GL_BT_COMPLIANCE_STRICT_MODE", "false"
            ).lower() == "true",
            materiality_threshold=Decimal(
                os.getenv("GL_BT_COMPLIANCE_MATERIALITY_THRESHOLD", "0.01")
            ),
            require_rf_disclosure=os.getenv(
                "GL_BT_COMPLIANCE_REQUIRE_RF_DISCLOSURE", "true"
            ).lower() == "true",
            double_counting_check=os.getenv(
                "GL_BT_COMPLIANCE_DOUBLE_COUNTING_CHECK", "true"
            ).lower() == "true",
            boundary_enforcement=os.getenv(
                "GL_BT_COMPLIANCE_BOUNDARY_ENFORCEMENT", "true"
            ).lower() == "true",
            require_data_quality=os.getenv(
                "GL_BT_COMPLIANCE_REQUIRE_DATA_QUALITY", "true"
            ).lower() == "true",
            min_data_quality_score=Decimal(
                os.getenv("GL_BT_COMPLIANCE_MIN_DATA_QUALITY_SCORE", "2.0")
            ),
        )


# =============================================================================
# SECTION 10: EF SOURCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EFSourceConfig:
    """
    Emission Factor Source configuration for Business Travel agent.

    This section configures the emission factor source hierarchy. The agent
    supports multiple EF databases (DEFRA, ICAO, EPA, IEA, EEIO) with
    a configurable fallback chain for maximum coverage.

    Attributes:
        primary_source: Primary EF source (GL_BT_EF_PRIMARY_SOURCE)
        fallback_chain: Comma-separated fallback EF sources (GL_BT_EF_FALLBACK_CHAIN)
        allow_custom: Allow custom emission factors (GL_BT_EF_ALLOW_CUSTOM)
        cache_ttl_seconds: EF cache TTL in seconds (GL_BT_EF_CACHE_TTL_SECONDS)
        custom_ef_path: Path to custom EF file (GL_BT_EF_CUSTOM_PATH)
        validate_ef_ranges: Validate EF values against expected ranges (GL_BT_EF_VALIDATE_RANGES)
        ef_year: Default emission factor year (GL_BT_EF_YEAR)

    Example:
        >>> ef_source = EFSourceConfig(
        ...     primary_source="DEFRA",
        ...     fallback_chain="DEFRA,ICAO,EPA,IEA,EEIO",
        ...     allow_custom=True,
        ...     cache_ttl_seconds=3600,
        ...     custom_ef_path=None,
        ...     validate_ef_ranges=True,
        ...     ef_year=2024
        ... )
        >>> ef_source.get_fallback_chain()
        ['DEFRA', 'ICAO', 'EPA', 'IEA', 'EEIO']
    """

    primary_source: str = "DEFRA"
    fallback_chain: str = "DEFRA,ICAO,EPA,IEA,EEIO"
    allow_custom: bool = True
    cache_ttl_seconds: int = 3600
    custom_ef_path: Optional[str] = None
    validate_ef_ranges: bool = True
    ef_year: int = 2024

    def validate(self) -> None:
        """
        Validate EF source configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_sources = {"DEFRA", "ICAO", "EPA", "IEA", "EEIO", "ECOINVENT", "CUSTOM"}

        if self.primary_source not in valid_sources:
            raise ValueError(
                f"Invalid primary_source '{self.primary_source}'. "
                f"Must be one of {valid_sources}"
            )

        chain = self.get_fallback_chain()
        if not chain:
            raise ValueError("fallback_chain must contain at least one source")

        for source in chain:
            if source not in valid_sources:
                raise ValueError(
                    f"Invalid source '{source}' in fallback_chain. "
                    f"Must be one of {valid_sources}"
                )

        if self.cache_ttl_seconds < 0 or self.cache_ttl_seconds > 86400:
            raise ValueError("cache_ttl_seconds must be between 0 and 86400")

        if self.primary_source == "CUSTOM" and not self.custom_ef_path:
            raise ValueError("custom_ef_path must be set when primary_source is CUSTOM")

        if self.ef_year < 2000 or self.ef_year > 2030:
            raise ValueError("ef_year must be between 2000 and 2030")

    def get_fallback_chain(self) -> List[str]:
        """Parse fallback chain string into list."""
        return [s.strip() for s in self.fallback_chain.split(",") if s.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_source": self.primary_source,
            "fallback_chain": self.fallback_chain,
            "allow_custom": self.allow_custom,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "custom_ef_path": self.custom_ef_path,
            "validate_ef_ranges": self.validate_ef_ranges,
            "ef_year": self.ef_year,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EFSourceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "EFSourceConfig":
        """Load from environment variables."""
        return cls(
            primary_source=os.getenv("GL_BT_EF_PRIMARY_SOURCE", "DEFRA"),
            fallback_chain=os.getenv(
                "GL_BT_EF_FALLBACK_CHAIN", "DEFRA,ICAO,EPA,IEA,EEIO"
            ),
            allow_custom=os.getenv("GL_BT_EF_ALLOW_CUSTOM", "true").lower() == "true",
            cache_ttl_seconds=int(
                os.getenv("GL_BT_EF_CACHE_TTL_SECONDS", "3600")
            ),
            custom_ef_path=os.getenv("GL_BT_EF_CUSTOM_PATH"),
            validate_ef_ranges=os.getenv(
                "GL_BT_EF_VALIDATE_RANGES", "true"
            ).lower() == "true",
            ef_year=int(os.getenv("GL_BT_EF_YEAR", "2024")),
        )


# =============================================================================
# SECTION 11: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty configuration for Business Travel agent.

    This section configures uncertainty quantification for business travel
    emissions calculations. Supports IPCC default ranges, Monte Carlo
    simulation, and bootstrap methods.

    Attributes:
        default_method: Default uncertainty method (GL_BT_UNCERTAINTY_DEFAULT_METHOD)
        default_iterations: Monte Carlo iterations (GL_BT_UNCERTAINTY_DEFAULT_ITERATIONS)
        default_confidence_level: Confidence level for intervals (GL_BT_UNCERTAINTY_DEFAULT_CONFIDENCE_LEVEL)
        include_parameter_uncertainty: Include parameter uncertainty (GL_BT_UNCERTAINTY_INCLUDE_PARAMETER)
        include_model_uncertainty: Include model uncertainty (GL_BT_UNCERTAINTY_INCLUDE_MODEL)
        include_activity_uncertainty: Include activity data uncertainty (GL_BT_UNCERTAINTY_INCLUDE_ACTIVITY)
        seed: Random seed for reproducibility (GL_BT_UNCERTAINTY_SEED)

    Example:
        >>> uncertainty = UncertaintyConfig(
        ...     default_method="MONTE_CARLO",
        ...     default_iterations=10000,
        ...     default_confidence_level=Decimal("0.95"),
        ...     include_parameter_uncertainty=True,
        ...     include_model_uncertainty=False,
        ...     include_activity_uncertainty=True,
        ...     seed=42
        ... )
        >>> uncertainty.default_method
        'MONTE_CARLO'
    """

    default_method: str = "MONTE_CARLO"
    default_iterations: int = 10000
    default_confidence_level: Decimal = Decimal("0.95")
    include_parameter_uncertainty: bool = True
    include_model_uncertainty: bool = False
    include_activity_uncertainty: bool = True
    seed: Optional[int] = None

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {
            "MONTE_CARLO",
            "BOOTSTRAP",
            "IPCC_DEFAULT",
            "BAYESIAN",
            "ANALYTICAL",
            "NONE",
        }
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.default_iterations < 100 or self.default_iterations > 1000000:
            raise ValueError("default_iterations must be between 100 and 1000000")

        if self.default_confidence_level < Decimal("0.5") or self.default_confidence_level > Decimal("0.999"):
            raise ValueError("default_confidence_level must be between 0.5 and 0.999")

        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 when specified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "default_iterations": self.default_iterations,
            "default_confidence_level": str(self.default_confidence_level),
            "include_parameter_uncertainty": self.include_parameter_uncertainty,
            "include_model_uncertainty": self.include_model_uncertainty,
            "include_activity_uncertainty": self.include_activity_uncertainty,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_confidence_level" in data_copy:
            data_copy["default_confidence_level"] = Decimal(
                data_copy["default_confidence_level"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        seed_str = os.getenv("GL_BT_UNCERTAINTY_SEED")
        seed_val = int(seed_str) if seed_str else None

        return cls(
            default_method=os.getenv(
                "GL_BT_UNCERTAINTY_DEFAULT_METHOD", "MONTE_CARLO"
            ),
            default_iterations=int(
                os.getenv("GL_BT_UNCERTAINTY_DEFAULT_ITERATIONS", "10000")
            ),
            default_confidence_level=Decimal(
                os.getenv("GL_BT_UNCERTAINTY_DEFAULT_CONFIDENCE_LEVEL", "0.95")
            ),
            include_parameter_uncertainty=os.getenv(
                "GL_BT_UNCERTAINTY_INCLUDE_PARAMETER", "true"
            ).lower() == "true",
            include_model_uncertainty=os.getenv(
                "GL_BT_UNCERTAINTY_INCLUDE_MODEL", "false"
            ).lower() == "true",
            include_activity_uncertainty=os.getenv(
                "GL_BT_UNCERTAINTY_INCLUDE_ACTIVITY", "true"
            ).lower() == "true",
            seed=seed_val,
        )


# =============================================================================
# SECTION 12: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Business Travel agent.

    This section configures in-memory and Redis caching for emission factor
    lookups, calculation results, and intermediate data.

    Attributes:
        enabled: Enable caching (GL_BT_CACHE_ENABLED)
        ttl_seconds: Cache TTL in seconds (GL_BT_CACHE_TTL_SECONDS)
        max_entries: Max cache entries (GL_BT_CACHE_MAX_ENTRIES)
        key_prefix: Cache key prefix (GL_BT_CACHE_KEY_PREFIX)
        cache_ef_lookups: Cache emission factor lookups (GL_BT_CACHE_EF_LOOKUPS)
        cache_calculations: Cache calculation results (GL_BT_CACHE_CALCULATIONS)
        cache_distance_lookups: Cache distance calculations (GL_BT_CACHE_DISTANCE_LOOKUPS)
        eviction_policy: Cache eviction policy (GL_BT_CACHE_EVICTION_POLICY)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl_seconds=3600,
        ...     max_entries=10000,
        ...     key_prefix="gl_bt:",
        ...     cache_ef_lookups=True,
        ...     cache_calculations=True,
        ...     cache_distance_lookups=True,
        ...     eviction_policy="LRU"
        ... )
        >>> cache.ttl_seconds
        3600
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 10000
    key_prefix: str = "gl_bt:"
    cache_ef_lookups: bool = True
    cache_calculations: bool = True
    cache_distance_lookups: bool = True
    eviction_policy: str = "LRU"

    def validate(self) -> None:
        """
        Validate cache configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.ttl_seconds < 0 or self.ttl_seconds > 86400:
            raise ValueError("ttl_seconds must be between 0 and 86400 (24 hours)")

        if self.max_entries < 1 or self.max_entries > 1000000:
            raise ValueError("max_entries must be between 1 and 1000000")

        if not self.key_prefix:
            raise ValueError("key_prefix cannot be empty")

        if not self.key_prefix.endswith(":"):
            raise ValueError("key_prefix must end with ':'")

        valid_policies = {"LRU", "LFU", "FIFO", "TTL"}
        if self.eviction_policy not in valid_policies:
            raise ValueError(
                f"Invalid eviction_policy '{self.eviction_policy}'. "
                f"Must be one of {valid_policies}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "key_prefix": self.key_prefix,
            "cache_ef_lookups": self.cache_ef_lookups,
            "cache_calculations": self.cache_calculations,
            "cache_distance_lookups": self.cache_distance_lookups,
            "eviction_policy": self.eviction_policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_BT_CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("GL_BT_CACHE_TTL_SECONDS", "3600")),
            max_entries=int(os.getenv("GL_BT_CACHE_MAX_ENTRIES", "10000")),
            key_prefix=os.getenv("GL_BT_CACHE_KEY_PREFIX", "gl_bt:"),
            cache_ef_lookups=os.getenv(
                "GL_BT_CACHE_EF_LOOKUPS", "true"
            ).lower() == "true",
            cache_calculations=os.getenv(
                "GL_BT_CACHE_CALCULATIONS", "true"
            ).lower() == "true",
            cache_distance_lookups=os.getenv(
                "GL_BT_CACHE_DISTANCE_LOOKUPS", "true"
            ).lower() == "true",
            eviction_policy=os.getenv("GL_BT_CACHE_EVICTION_POLICY", "LRU"),
        )


# =============================================================================
# SECTION 13: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    API configuration for Business Travel agent.

    This section configures the REST API endpoints, rate limiting,
    batch processing, and concurrency controls.

    Attributes:
        rate_limit: Requests per minute per tenant (GL_BT_API_RATE_LIMIT)
        timeout_seconds: Request timeout in seconds (GL_BT_API_TIMEOUT_SECONDS)
        enable_batch: Enable batch processing endpoint (GL_BT_API_ENABLE_BATCH)
        max_concurrent: Max concurrent requests (GL_BT_API_MAX_CONCURRENT)
        max_batch_size: Max records per batch request (GL_BT_API_MAX_BATCH_SIZE)
        enable_streaming: Enable streaming responses (GL_BT_API_ENABLE_STREAMING)
        cors_origins: Comma-separated CORS allowed origins (GL_BT_API_CORS_ORIGINS)
        worker_threads: Worker thread count (GL_BT_API_WORKER_THREADS)

    Example:
        >>> api = APIConfig(
        ...     rate_limit=100,
        ...     timeout_seconds=30,
        ...     enable_batch=True,
        ...     max_concurrent=10,
        ...     max_batch_size=500,
        ...     enable_streaming=False,
        ...     cors_origins="*",
        ...     worker_threads=4
        ... )
        >>> api.rate_limit
        100
    """

    rate_limit: int = 100
    timeout_seconds: int = 30
    enable_batch: bool = True
    max_concurrent: int = 10
    max_batch_size: int = 500
    enable_streaming: bool = False
    cors_origins: str = "*"
    worker_threads: int = 4

    def validate(self) -> None:
        """
        Validate API configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.rate_limit < 1 or self.rate_limit > 10000:
            raise ValueError("rate_limit must be between 1 and 10000")

        if self.timeout_seconds < 1 or self.timeout_seconds > 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600 seconds")

        if self.max_concurrent < 1 or self.max_concurrent > 1000:
            raise ValueError("max_concurrent must be between 1 and 1000")

        if self.max_batch_size < 1 or self.max_batch_size > 10000:
            raise ValueError("max_batch_size must be between 1 and 10000")

        if not self.cors_origins:
            raise ValueError("cors_origins cannot be empty")

        if self.worker_threads < 1 or self.worker_threads > 64:
            raise ValueError("worker_threads must be between 1 and 64")

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins string into list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rate_limit": self.rate_limit,
            "timeout_seconds": self.timeout_seconds,
            "enable_batch": self.enable_batch,
            "max_concurrent": self.max_concurrent,
            "max_batch_size": self.max_batch_size,
            "enable_streaming": self.enable_streaming,
            "cors_origins": self.cors_origins,
            "worker_threads": self.worker_threads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load from environment variables."""
        return cls(
            rate_limit=int(os.getenv("GL_BT_API_RATE_LIMIT", "100")),
            timeout_seconds=int(os.getenv("GL_BT_API_TIMEOUT_SECONDS", "30")),
            enable_batch=os.getenv(
                "GL_BT_API_ENABLE_BATCH", "true"
            ).lower() == "true",
            max_concurrent=int(os.getenv("GL_BT_API_MAX_CONCURRENT", "10")),
            max_batch_size=int(os.getenv("GL_BT_API_MAX_BATCH_SIZE", "500")),
            enable_streaming=os.getenv(
                "GL_BT_API_ENABLE_STREAMING", "false"
            ).lower() == "true",
            cors_origins=os.getenv("GL_BT_API_CORS_ORIGINS", "*"),
            worker_threads=int(os.getenv("GL_BT_API_WORKER_THREADS", "4")),
        )


# =============================================================================
# SECTION 14: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Business Travel agent.

    This section configures data provenance tracking with SHA-256 hashing
    for complete audit trails. Supports chain hashing and intermediate
    result storage for reproducibility.

    Attributes:
        enabled: Enable provenance tracking (GL_BT_PROVENANCE_ENABLED)
        hash_algorithm: Hash algorithm for provenance (GL_BT_PROVENANCE_HASH_ALGORITHM)
        store_intermediate: Store intermediate hashes (GL_BT_PROVENANCE_STORE_INTERMEDIATE)
        chain_hashes: Chain hashes across pipeline steps (GL_BT_PROVENANCE_CHAIN_HASHES)
        include_config_hash: Include config hash in provenance (GL_BT_PROVENANCE_INCLUDE_CONFIG_HASH)
        include_ef_hash: Include EF data hash in provenance (GL_BT_PROVENANCE_INCLUDE_EF_HASH)

    Example:
        >>> provenance = ProvenanceConfig(
        ...     enabled=True,
        ...     hash_algorithm="sha256",
        ...     store_intermediate=True,
        ...     chain_hashes=True,
        ...     include_config_hash=True,
        ...     include_ef_hash=True
        ... )
        >>> provenance.hash_algorithm
        'sha256'
    """

    enabled: bool = True
    hash_algorithm: str = "sha256"
    store_intermediate: bool = True
    chain_hashes: bool = True
    include_config_hash: bool = True
    include_ef_hash: bool = True

    def validate(self) -> None:
        """
        Validate provenance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_algorithms = {"sha256", "sha512", "blake2b"}
        if self.hash_algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid hash_algorithm '{self.hash_algorithm}'. "
                f"Must be one of {valid_algorithms}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "hash_algorithm": self.hash_algorithm,
            "store_intermediate": self.store_intermediate,
            "chain_hashes": self.chain_hashes,
            "include_config_hash": self.include_config_hash,
            "include_ef_hash": self.include_ef_hash,
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
                "GL_BT_PROVENANCE_ENABLED", "true"
            ).lower() == "true",
            hash_algorithm=os.getenv("GL_BT_PROVENANCE_HASH_ALGORITHM", "sha256"),
            store_intermediate=os.getenv(
                "GL_BT_PROVENANCE_STORE_INTERMEDIATE", "true"
            ).lower() == "true",
            chain_hashes=os.getenv(
                "GL_BT_PROVENANCE_CHAIN_HASHES", "true"
            ).lower() == "true",
            include_config_hash=os.getenv(
                "GL_BT_PROVENANCE_INCLUDE_CONFIG_HASH", "true"
            ).lower() == "true",
            include_ef_hash=os.getenv(
                "GL_BT_PROVENANCE_INCLUDE_EF_HASH", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 15: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Business Travel agent.

    This section configures Prometheus-compatible metrics collection
    including histogram buckets for latency tracking and tenant-level
    metric isolation.

    Attributes:
        enabled: Enable metrics collection (GL_BT_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_BT_METRICS_PREFIX)
        include_tenant: Include tenant label in metrics (GL_BT_METRICS_INCLUDE_TENANT)
        histogram_buckets: Comma-separated histogram bucket boundaries (GL_BT_METRICS_HISTOGRAM_BUCKETS)
        enable_latency_tracking: Track per-engine latency (GL_BT_METRICS_ENABLE_LATENCY_TRACKING)
        enable_error_counting: Track error counts by type (GL_BT_METRICS_ENABLE_ERROR_COUNTING)
        enable_throughput_tracking: Track records per second (GL_BT_METRICS_ENABLE_THROUGHPUT_TRACKING)

    Example:
        >>> metrics = MetricsConfig(
        ...     enabled=True,
        ...     prefix="gl_bt",
        ...     include_tenant=True,
        ...     histogram_buckets="0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
        ...     enable_latency_tracking=True,
        ...     enable_error_counting=True,
        ...     enable_throughput_tracking=True
        ... )
        >>> metrics.get_histogram_buckets()
        (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    """

    enabled: bool = True
    prefix: str = "gl_bt"
    include_tenant: bool = True
    histogram_buckets: str = "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
    enable_latency_tracking: bool = True
    enable_error_counting: bool = True
    enable_throughput_tracking: bool = True

    def validate(self) -> None:
        """
        Validate metrics configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        # Validate buckets format
        try:
            buckets = self.get_histogram_buckets()
            if not buckets:
                raise ValueError("At least one histogram bucket must be defined")
            for bucket in buckets:
                if bucket <= 0:
                    raise ValueError("All histogram buckets must be positive")
            # Validate buckets are sorted ascending
            for i in range(1, len(buckets)):
                if buckets[i] <= buckets[i - 1]:
                    raise ValueError("histogram_buckets must be in ascending order")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid histogram_buckets format: {e}")

    def get_histogram_buckets(self) -> Tuple[float, ...]:
        """Parse histogram buckets string into tuple of floats."""
        return tuple(float(x.strip()) for x in self.histogram_buckets.split(","))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "include_tenant": self.include_tenant,
            "histogram_buckets": self.histogram_buckets,
            "enable_latency_tracking": self.enable_latency_tracking,
            "enable_error_counting": self.enable_error_counting,
            "enable_throughput_tracking": self.enable_throughput_tracking,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_BT_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_BT_METRICS_PREFIX", "gl_bt"),
            include_tenant=os.getenv(
                "GL_BT_METRICS_INCLUDE_TENANT", "true"
            ).lower() == "true",
            histogram_buckets=os.getenv(
                "GL_BT_METRICS_HISTOGRAM_BUCKETS",
                "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
            enable_latency_tracking=os.getenv(
                "GL_BT_METRICS_ENABLE_LATENCY_TRACKING", "true"
            ).lower() == "true",
            enable_error_counting=os.getenv(
                "GL_BT_METRICS_ENABLE_ERROR_COUNTING", "true"
            ).lower() == "true",
            enable_throughput_tracking=os.getenv(
                "GL_BT_METRICS_ENABLE_THROUGHPUT_TRACKING", "true"
            ).lower() == "true",
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass(frozen=True)
class BusinessTravelConfig:
    """
    Master configuration class for Business Travel agent (AGENT-MRV-019).

    This frozen dataclass aggregates all 15 configuration sections and provides
    a unified interface for accessing configuration values. It implements the
    singleton pattern with thread-safe access.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        redis: Redis cache configuration
        air_travel: Air travel emissions configuration
        rail: Rail travel emissions configuration
        road: Road travel emissions configuration
        hotel: Hotel accommodation emissions configuration
        spend: Spend-based calculation configuration
        compliance: Regulatory compliance configuration
        ef_source: Emission factor source configuration
        uncertainty: Uncertainty quantification configuration
        cache: Cache layer configuration
        api: REST API configuration
        provenance: Data provenance configuration
        metrics: Prometheus metrics configuration

    Example:
        >>> config = BusinessTravelConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-006'
        >>> config.air_travel.default_rfi_multiplier
        Decimal('1.891')
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    air_travel: AirTravelConfig = field(default_factory=AirTravelConfig)
    rail: RailConfig = field(default_factory=RailConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    hotel: HotelConfig = field(default_factory=HotelConfig)
    spend: SpendConfig = field(default_factory=SpendConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    ef_source: EFSourceConfig = field(default_factory=EFSourceConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def validate_all(self) -> List[str]:
        """
        Validate all configuration sections and return list of errors.

        Unlike individual validate() methods which raise on first error,
        this method collects all validation errors across all sections.

        Returns:
            List of validation error messages (empty if all valid)

        Example:
            >>> config = BusinessTravelConfig.from_env()
            >>> errors = config.validate_all()
            >>> if errors:
            ...     for error in errors:
            ...         print(f"  - {error}")
        """
        errors: List[str] = []

        sections = [
            ("general", self.general),
            ("database", self.database),
            ("redis", self.redis),
            ("air_travel", self.air_travel),
            ("rail", self.rail),
            ("road", self.road),
            ("hotel", self.hotel),
            ("spend", self.spend),
            ("compliance", self.compliance),
            ("ef_source", self.ef_source),
            ("uncertainty", self.uncertainty),
            ("cache", self.cache),
            ("api", self.api),
            ("provenance", self.provenance),
            ("metrics", self.metrics),
        ]

        for section_name, section in sections:
            try:
                section.validate()
            except ValueError as e:
                errors.append(f"{section_name}: {str(e)}")

        # Cross-section validation
        errors.extend(self._cross_validate())

        return errors

    def _cross_validate(self) -> List[str]:
        """
        Perform cross-section validation checks.

        Returns:
            List of cross-validation error messages
        """
        errors: List[str] = []

        # Ensure general API prefix is consistent
        if self.general.api_prefix and not self.general.api_prefix.startswith("/api/"):
            errors.append(
                "cross-validation: general.api_prefix should start with '/api/'"
            )

        # Ensure cache key_prefix matches redis prefix
        if self.cache.enabled and self.cache.key_prefix != self.redis.prefix:
            logger.warning(
                "cache.key_prefix '%s' differs from redis.prefix '%s' - "
                "this may cause key namespace confusion",
                self.cache.key_prefix,
                self.redis.prefix,
            )

        # Ensure air travel thresholds are consistent
        if self.air_travel.domestic_threshold_km >= self.air_travel.short_haul_threshold_km:
            errors.append(
                "cross-validation: air_travel.domestic_threshold_km must be less "
                "than air_travel.short_haul_threshold_km"
            )

        # Ensure pool_min <= pool_max for database
        if self.database.pool_min > self.database.pool_max:
            errors.append(
                "cross-validation: database.pool_min must be <= database.pool_max"
            )

        # Ensure batch size consistency
        if self.api.max_batch_size > self.general.max_batch_size:
            errors.append(
                "cross-validation: api.max_batch_size should not exceed "
                "general.max_batch_size"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire configuration to dictionary.

        Returns:
            Dictionary representation of all configuration sections
        """
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "redis": self.redis.to_dict(),
            "air_travel": self.air_travel.to_dict(),
            "rail": self.rail.to_dict(),
            "road": self.road.to_dict(),
            "hotel": self.hotel.to_dict(),
            "spend": self.spend.to_dict(),
            "compliance": self.compliance.to_dict(),
            "ef_source": self.ef_source.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "cache": self.cache.to_dict(),
            "api": self.api.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusinessTravelConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            BusinessTravelConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            redis=RedisConfig.from_dict(data.get("redis", {})),
            air_travel=AirTravelConfig.from_dict(data.get("air_travel", {})),
            rail=RailConfig.from_dict(data.get("rail", {})),
            road=RoadConfig.from_dict(data.get("road", {})),
            hotel=HotelConfig.from_dict(data.get("hotel", {})),
            spend=SpendConfig.from_dict(data.get("spend", {})),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            ef_source=EFSourceConfig.from_dict(data.get("ef_source", {})),
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            api=APIConfig.from_dict(data.get("api", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
        )

    @classmethod
    def from_env(cls) -> "BusinessTravelConfig":
        """
        Load configuration from environment variables.

        Returns:
            BusinessTravelConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            air_travel=AirTravelConfig.from_env(),
            rail=RailConfig.from_env(),
            road=RoadConfig.from_env(),
            hotel=HotelConfig.from_env(),
            spend=SpendConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            ef_source=EFSourceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            cache=CacheConfig.from_env(),
            api=APIConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            metrics=MetricsConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[BusinessTravelConfig] = None
_config_lock = threading.RLock()


def get_config() -> BusinessTravelConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton. The first call will load configuration from
    environment variables. Subsequent calls return the cached instance.

    Returns:
        BusinessTravelConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-006'

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The configuration is initialized only once using
        double-checked locking.
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            # Double-checked locking pattern
            if _config_instance is None:
                logger.info("Initializing BusinessTravelConfig from environment")
                config = BusinessTravelConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning("Configuration validation warning: %s", error)
                _config_instance = config
                logger.info(
                    "BusinessTravelConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: BusinessTravelConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.

    Args:
        config: BusinessTravelConfig instance to set as singleton

    Raises:
        TypeError: If config is not a BusinessTravelConfig instance

    Example:
        >>> custom_config = BusinessTravelConfig.from_dict({...})
        >>> set_config(custom_config)

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently.
    """
    global _config_instance

    if not isinstance(config, BusinessTravelConfig):
        raise TypeError(
            f"config must be a BusinessTravelConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("BusinessTravelConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    This function clears the cached configuration singleton, forcing the next
    call to get_config() to reload from environment variables. Primarily
    useful for testing scenarios.

    Example:
        >>> reset_config()
        >>> config = get_config()  # Reloads from environment

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("BusinessTravelConfig singleton reset")


def validate_config(config: BusinessTravelConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    This function validates all configuration sections and returns a list of
    validation errors. Unlike individual validate() methods which raise on
    first error, this function collects all errors across all sections.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> config = get_config()
        >>> errors = validate_config(config)
        >>> if errors:
        ...     print(f"Configuration errors: {errors}")
    """
    return config.validate_all()


def print_config(config: BusinessTravelConfig) -> None:
    """
    Print configuration in human-readable format.

    This function prints all configuration sections in a formatted,
    human-readable manner. Sensitive fields (passwords, connection URLs)
    are redacted for security. Useful for debugging and verification.

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
        ===== Business Travel Configuration (AGENT-MRV-019) =====
        [GENERAL]
        enabled: True
        debug: False
        ...
    """
    # Fields that should be redacted in output
    redacted_fields = {"password", "database_url", "redis_url", "secret", "token", "key"}

    def _should_redact(field_name: str) -> bool:
        """Check if a field should be redacted."""
        field_lower = field_name.lower()
        return any(r in field_lower for r in redacted_fields)

    def _print_section(name: str, data: Dict[str, Any]) -> None:
        """Print a single configuration section."""
        print(f"\n[{name}]")
        for key, value in data.items():
            if _should_redact(key) and value:
                print(f"  {key}: [REDACTED]")
            else:
                print(f"  {key}: {value}")

    print("=" * 64)
    print("  Business Travel Configuration (AGENT-MRV-019)")
    print("  Agent ID: " + config.general.agent_id)
    print("  Version:  " + config.general.version)
    print("=" * 64)

    _print_section("GENERAL", config.general.to_dict())
    _print_section("DATABASE", config.database.to_dict())
    _print_section("REDIS", config.redis.to_dict())
    _print_section("AIR_TRAVEL", config.air_travel.to_dict())
    _print_section("RAIL", config.rail.to_dict())
    _print_section("ROAD", config.road.to_dict())
    _print_section("HOTEL", config.hotel.to_dict())
    _print_section("SPEND", config.spend.to_dict())
    _print_section("COMPLIANCE", config.compliance.to_dict())
    _print_section("EF_SOURCE", config.ef_source.to_dict())
    _print_section("UNCERTAINTY", config.uncertainty.to_dict())
    _print_section("CACHE", config.cache.to_dict())
    _print_section("API", config.api.to_dict())
    _print_section("PROVENANCE", config.provenance.to_dict())
    _print_section("METRICS", config.metrics.to_dict())

    # Print validation summary
    errors = config.validate_all()
    print("\n[VALIDATION]")
    if errors:
        print(f"  status: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"  - {error}")
    else:
        print("  status: PASSED (all 15 sections valid)")

    print("\n" + "=" * 64)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration dataclasses
    "GeneralConfig",
    "DatabaseConfig",
    "RedisConfig",
    "AirTravelConfig",
    "RailConfig",
    "RoadConfig",
    "HotelConfig",
    "SpendConfig",
    "ComplianceConfig",
    "EFSourceConfig",
    "UncertaintyConfig",
    "CacheConfig",
    "APIConfig",
    "ProvenanceConfig",
    "MetricsConfig",
    # Master configuration
    "BusinessTravelConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
]
