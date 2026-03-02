# -*- coding: utf-8 -*-
"""
Employee Commuting Configuration - AGENT-MRV-020

Thread-safe singleton configuration for GL-MRV-S3-007.
All environment variables prefixed with GL_EC_.

This module provides comprehensive configuration management for the Employee
Commuting agent (GHG Protocol Scope 3 Category 7), supporting:
- Multi-mode commute calculation (car/bus/rail/subway/bicycle/walk/motorcycle/ferry)
- Single occupancy vehicle (SOV) vs carpool occupancy modeling
- Telework/remote work energy consumption modeling (laptop, heating, lighting)
- Survey-based extrapolation with statistical confidence intervals
- Working days calculation with regional holidays, PTO, and sick leave
- Spend-based EEIO fallback calculation method
- Well-to-tank (WTT) upstream fuel cycle emissions
- Emission factor hierarchy (employee-reported, DEFRA, EPA, IEA, Census, EEIO)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi, GRI, EPA)
- Monte Carlo uncertainty quantification
- Provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-007'
    >>> config.commute_mode.max_distance_km
    500.0
    >>> config.telework.default_daily_kwh
    Decimal('4.0')
    >>> config.survey.confidence_level
    Decimal('0.95')

Thread Safety:
    All configuration operations are protected by threading.Lock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_EC_ prefix. See individual config sections for specific variables.
"""

import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# SECTION 1: GENERAL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GeneralConfig:
    """
    General configuration for Employee Commuting agent.

    Controls agent identity, logging, retry behavior, and timeout settings
    for the GL-MRV-S3-007 (Scope 3 Category 7) agent.

    Attributes:
        enabled: Master switch for the agent (GL_EC_ENABLED)
        debug: Enable debug mode with verbose logging (GL_EC_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_EC_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_EC_AGENT_ID)
        version: Agent version following SemVer (GL_EC_VERSION)
        table_prefix: Prefix for all database tables (GL_EC_TABLE_PREFIX)
        max_retries: Maximum retry attempts for transient failures (GL_EC_MAX_RETRIES)
        timeout: Default operation timeout in seconds (GL_EC_TIMEOUT)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-007",
        ...     version="1.0.0",
        ...     table_prefix="gl_ec_",
        ...     max_retries=3,
        ...     timeout=300
        ... )
        >>> general.agent_id
        'GL-MRV-S3-007'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-007"
    version: str = "1.0.0"
    table_prefix: str = "gl_ec_"
    max_retries: int = 3
    timeout: int = 300

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

        if not self.version:
            raise ValueError("version cannot be empty")

        # Validate SemVer format (basic check)
        version_parts = self.version.split(".")
        if len(version_parts) != 3:
            raise ValueError(
                f"Invalid version '{self.version}'. Must follow SemVer (e.g., '1.0.0')"
            )

        for part in version_parts:
            if not part.isdigit():
                raise ValueError(
                    f"Invalid version '{self.version}'. Each segment must be numeric"
                )

        if not self.table_prefix:
            raise ValueError("table_prefix cannot be empty")

        if not self.table_prefix.endswith("_"):
            raise ValueError("table_prefix must end with '_'")

        if self.max_retries < 0 or self.max_retries > 20:
            raise ValueError("max_retries must be between 0 and 20")

        if self.timeout < 1 or self.timeout > 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "debug": self.debug,
            "log_level": self.log_level,
            "agent_id": self.agent_id,
            "version": self.version,
            "table_prefix": self.table_prefix,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_EC_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_EC_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_EC_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_EC_AGENT_ID", "GL-MRV-S3-007"),
            version=os.getenv("GL_EC_VERSION", "1.0.0"),
            table_prefix=os.getenv("GL_EC_TABLE_PREFIX", "gl_ec_"),
            max_retries=int(os.getenv("GL_EC_MAX_RETRIES", "3")),
            timeout=int(os.getenv("GL_EC_TIMEOUT", "300")),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Employee Commuting agent.

    Manages PostgreSQL connection parameters including connection pooling,
    SSL mode, and query timeouts for the employee commuting service schema.

    Attributes:
        host: Database hostname (GL_EC_DB_HOST)
        port: Database port (GL_EC_DB_PORT)
        name: Database name (GL_EC_DB_NAME)
        user: Database user (GL_EC_DB_USER)
        password: Database password (GL_EC_DB_PASSWORD)
        pool_min: Minimum pool connections (GL_EC_DB_POOL_MIN)
        pool_max: Maximum pool connections (GL_EC_DB_POOL_MAX)
        ssl: Enable SSL connection (GL_EC_DB_SSL)
        timeout: Query timeout in seconds (GL_EC_DB_TIMEOUT)
        schema: Database schema name (GL_EC_DB_SCHEMA)

    Example:
        >>> db = DatabaseConfig(
        ...     host="localhost",
        ...     port=5432,
        ...     name="greenlang",
        ...     user="greenlang",
        ...     password="secret",
        ...     pool_min=2,
        ...     pool_max=10,
        ...     ssl=True,
        ...     timeout=30,
        ...     schema="employee_commuting_service"
        ... )
        >>> db.pool_max
        10
    """

    host: str = "localhost"
    port: int = 5432
    name: str = "greenlang"
    user: str = "greenlang"
    password: str = ""
    pool_min: int = 2
    pool_max: int = 10
    ssl: bool = False
    timeout: int = 30
    schema: str = "employee_commuting_service"

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

        if not self.name:
            raise ValueError("name cannot be empty")

        if not self.user:
            raise ValueError("user cannot be empty")

        if self.pool_min < 1:
            raise ValueError("pool_min must be >= 1")

        if self.pool_max < 1:
            raise ValueError("pool_max must be >= 1")

        if self.pool_min > self.pool_max:
            raise ValueError("pool_min must be <= pool_max")

        if self.timeout < 1:
            raise ValueError("timeout must be >= 1")

        if not self.schema:
            raise ValueError("schema cannot be empty")

    def get_connection_url(self) -> str:
        """
        Build PostgreSQL connection URL.

        Returns:
            Connection URL string with or without SSL parameter
        """
        ssl_param = "?sslmode=require" if self.ssl else ""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.name}{ssl_param}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "user": self.user,
            "password": "***REDACTED***",
            "pool_min": self.pool_min,
            "pool_max": self.pool_max,
            "ssl": self.ssl,
            "timeout": self.timeout,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        # Restore redacted password if present
        if data_copy.get("password") == "***REDACTED***":
            data_copy["password"] = ""
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_EC_DB_HOST", "localhost"),
            port=int(os.getenv("GL_EC_DB_PORT", "5432")),
            name=os.getenv("GL_EC_DB_NAME", "greenlang"),
            user=os.getenv("GL_EC_DB_USER", "greenlang"),
            password=os.getenv("GL_EC_DB_PASSWORD", ""),
            pool_min=int(os.getenv("GL_EC_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_EC_DB_POOL_MAX", "10")),
            ssl=os.getenv("GL_EC_DB_SSL", "false").lower() == "true",
            timeout=int(os.getenv("GL_EC_DB_TIMEOUT", "30")),
            schema=os.getenv("GL_EC_DB_SCHEMA", "employee_commuting_service"),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Employee Commuting agent.

    Controls caching layer parameters for emission factor lookups,
    survey results, and intermediate calculation caching.

    Attributes:
        host: Redis hostname (GL_EC_REDIS_HOST)
        port: Redis port (GL_EC_REDIS_PORT)
        db: Redis database number (GL_EC_REDIS_DB)
        password: Redis password (GL_EC_REDIS_PASSWORD)
        ssl: Enable SSL connection (GL_EC_REDIS_SSL)
        ttl_seconds: Default TTL in seconds (GL_EC_REDIS_TTL)
        max_connections: Max connection pool size (GL_EC_REDIS_MAX_CONNECTIONS)
        prefix: Key namespace prefix (GL_EC_REDIS_PREFIX)

    Example:
        >>> redis = RedisConfig(
        ...     host="localhost",
        ...     port=6379,
        ...     db=0,
        ...     password="",
        ...     ssl=False,
        ...     ttl_seconds=3600,
        ...     max_connections=20,
        ...     prefix="gl_ec:"
        ... )
        >>> redis.ttl_seconds
        3600
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    ttl_seconds: int = 3600
    max_connections: int = 20
    prefix: str = "gl_ec:"

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

        if self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")

        if self.max_connections < 1:
            raise ValueError("max_connections must be >= 1")

        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        if not self.prefix.endswith(":"):
            raise ValueError("prefix must end with ':'")

    def get_connection_url(self) -> str:
        """
        Build Redis connection URL.

        Returns:
            Redis connection URL string
        """
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": "***REDACTED***" if self.password else "",
            "ssl": self.ssl,
            "ttl_seconds": self.ttl_seconds,
            "max_connections": self.max_connections,
            "prefix": self.prefix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if data_copy.get("password") == "***REDACTED***":
            data_copy["password"] = ""
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_EC_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_EC_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_EC_REDIS_DB", "0")),
            password=os.getenv("GL_EC_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_EC_REDIS_SSL", "false").lower() == "true",
            ttl_seconds=int(os.getenv("GL_EC_REDIS_TTL", "3600")),
            max_connections=int(os.getenv("GL_EC_REDIS_MAX_CONNECTIONS", "20")),
            prefix=os.getenv("GL_EC_REDIS_PREFIX", "gl_ec:"),
        )


# =============================================================================
# SECTION 4: COMMUTE MODE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CommuteModeConfig:
    """
    Commute mode configuration for Employee Commuting agent.

    Governs vehicle type defaults, fuel type defaults, distance limits,
    occupancy settings, and well-to-tank inclusion for commute mode
    calculations.

    Supported commute modes:
    - CAR_SOV (single occupancy vehicle)
    - CAR_CARPOOL (carpooling)
    - BUS_LOCAL / BUS_COACH
    - RAIL_COMMUTER / RAIL_INTERCITY / RAIL_HIGH_SPEED
    - SUBWAY / LIGHT_RAIL / TRAM
    - MOTORCYCLE / SCOOTER
    - BICYCLE / E_BICYCLE
    - WALK
    - FERRY
    - TAXI / RIDE_SHARE
    - VAN_POOL

    Attributes:
        default_vehicle_type: Default vehicle type for car commutes (GL_EC_DEFAULT_VEHICLE_TYPE)
        default_fuel_type: Default fuel type for car commutes (GL_EC_DEFAULT_FUEL_TYPE)
        max_distance_km: Maximum one-way commute distance in km (GL_EC_MAX_DISTANCE_KM)
        max_occupancy: Maximum vehicle occupancy (GL_EC_MAX_OCCUPANCY)
        include_wtt: Include well-to-tank upstream emissions (GL_EC_INCLUDE_WTT)
        default_occupancy_sov: Default occupancy for single occupancy vehicle (GL_EC_DEFAULT_OCCUPANCY_SOV)
        default_occupancy_carpool: Default occupancy for carpool (GL_EC_DEFAULT_OCCUPANCY_CARPOOL)
        default_occupancy_vanpool: Default occupancy for vanpool (GL_EC_DEFAULT_OCCUPANCY_VANPOOL)
        default_occupancy_bus: Default occupancy for bus (GL_EC_DEFAULT_OCCUPANCY_BUS)
        default_gwp_source: Default GWP assessment report source (GL_EC_DEFAULT_GWP_SOURCE)
        round_trip_factor: Multiply one-way distance by this factor (GL_EC_ROUND_TRIP_FACTOR)
        e_bicycle_ef_fraction: E-bicycle EF as fraction of car EF (GL_EC_E_BICYCLE_EF_FRACTION)

    Example:
        >>> commute = CommuteModeConfig(
        ...     default_vehicle_type="AVERAGE_CAR",
        ...     default_fuel_type="GASOLINE",
        ...     max_distance_km=500.0,
        ...     max_occupancy=15,
        ...     include_wtt=True,
        ...     default_occupancy_sov=Decimal("1.0"),
        ...     default_occupancy_carpool=Decimal("2.5")
        ... )
        >>> commute.max_distance_km
        500.0
    """

    default_vehicle_type: str = "AVERAGE_CAR"
    default_fuel_type: str = "GASOLINE"
    max_distance_km: float = 500.0
    max_occupancy: int = 15
    include_wtt: bool = True
    default_occupancy_sov: Decimal = Decimal("1.0")
    default_occupancy_carpool: Decimal = Decimal("2.5")
    default_occupancy_vanpool: Decimal = Decimal("7.0")
    default_occupancy_bus: Decimal = Decimal("30.0")
    default_gwp_source: str = "AR5"
    round_trip_factor: Decimal = Decimal("2.0")
    e_bicycle_ef_fraction: Decimal = Decimal("0.05")

    def validate(self) -> None:
        """
        Validate commute mode configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_vehicle_types = {
            "AVERAGE_CAR",
            "SMALL_CAR",
            "MEDIUM_CAR",
            "LARGE_CAR",
            "SUV",
            "PICKUP_TRUCK",
            "HYBRID",
            "PLUG_IN_HYBRID",
            "BATTERY_ELECTRIC",
            "MOTORCYCLE",
            "SCOOTER",
        }
        if self.default_vehicle_type not in valid_vehicle_types:
            raise ValueError(
                f"Invalid default_vehicle_type '{self.default_vehicle_type}'. "
                f"Must be one of {valid_vehicle_types}"
            )

        valid_fuel_types = {
            "GASOLINE",
            "DIESEL",
            "CNG",
            "LPG",
            "E85",
            "BIODIESEL",
            "ELECTRICITY",
            "HYDROGEN",
            "HYBRID_GASOLINE",
            "HYBRID_DIESEL",
        }
        if self.default_fuel_type not in valid_fuel_types:
            raise ValueError(
                f"Invalid default_fuel_type '{self.default_fuel_type}'. "
                f"Must be one of {valid_fuel_types}"
            )

        if self.max_distance_km <= 0 or self.max_distance_km > 2000:
            raise ValueError("max_distance_km must be between 0 (exclusive) and 2000")

        if self.max_occupancy < 1 or self.max_occupancy > 60:
            raise ValueError("max_occupancy must be between 1 and 60")

        if self.default_occupancy_sov < Decimal("1.0") or self.default_occupancy_sov > Decimal("2.0"):
            raise ValueError("default_occupancy_sov must be between 1.0 and 2.0")

        if self.default_occupancy_carpool < Decimal("2.0") or self.default_occupancy_carpool > Decimal("8.0"):
            raise ValueError("default_occupancy_carpool must be between 2.0 and 8.0")

        if self.default_occupancy_vanpool < Decimal("3.0") or self.default_occupancy_vanpool > Decimal("15.0"):
            raise ValueError("default_occupancy_vanpool must be between 3.0 and 15.0")

        if self.default_occupancy_bus < Decimal("5.0") or self.default_occupancy_bus > Decimal("80.0"):
            raise ValueError("default_occupancy_bus must be between 5.0 and 80.0")

        valid_gwp_sources = {"AR4", "AR5", "AR6"}
        if self.default_gwp_source not in valid_gwp_sources:
            raise ValueError(
                f"Invalid default_gwp_source '{self.default_gwp_source}'. "
                f"Must be one of {valid_gwp_sources}"
            )

        if self.round_trip_factor < Decimal("1.0") or self.round_trip_factor > Decimal("3.0"):
            raise ValueError("round_trip_factor must be between 1.0 and 3.0")

        if self.e_bicycle_ef_fraction < Decimal("0.0") or self.e_bicycle_ef_fraction > Decimal("1.0"):
            raise ValueError("e_bicycle_ef_fraction must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_vehicle_type": self.default_vehicle_type,
            "default_fuel_type": self.default_fuel_type,
            "max_distance_km": self.max_distance_km,
            "max_occupancy": self.max_occupancy,
            "include_wtt": self.include_wtt,
            "default_occupancy_sov": str(self.default_occupancy_sov),
            "default_occupancy_carpool": str(self.default_occupancy_carpool),
            "default_occupancy_vanpool": str(self.default_occupancy_vanpool),
            "default_occupancy_bus": str(self.default_occupancy_bus),
            "default_gwp_source": self.default_gwp_source,
            "round_trip_factor": str(self.round_trip_factor),
            "e_bicycle_ef_fraction": str(self.e_bicycle_ef_fraction),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommuteModeConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "default_occupancy_sov",
            "default_occupancy_carpool",
            "default_occupancy_vanpool",
            "default_occupancy_bus",
            "round_trip_factor",
            "e_bicycle_ef_fraction",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "CommuteModeConfig":
        """Load from environment variables."""
        return cls(
            default_vehicle_type=os.getenv(
                "GL_EC_DEFAULT_VEHICLE_TYPE", "AVERAGE_CAR"
            ),
            default_fuel_type=os.getenv("GL_EC_DEFAULT_FUEL_TYPE", "GASOLINE"),
            max_distance_km=float(os.getenv("GL_EC_MAX_DISTANCE_KM", "500.0")),
            max_occupancy=int(os.getenv("GL_EC_MAX_OCCUPANCY", "15")),
            include_wtt=os.getenv("GL_EC_INCLUDE_WTT", "true").lower() == "true",
            default_occupancy_sov=Decimal(
                os.getenv("GL_EC_DEFAULT_OCCUPANCY_SOV", "1.0")
            ),
            default_occupancy_carpool=Decimal(
                os.getenv("GL_EC_DEFAULT_OCCUPANCY_CARPOOL", "2.5")
            ),
            default_occupancy_vanpool=Decimal(
                os.getenv("GL_EC_DEFAULT_OCCUPANCY_VANPOOL", "7.0")
            ),
            default_occupancy_bus=Decimal(
                os.getenv("GL_EC_DEFAULT_OCCUPANCY_BUS", "30.0")
            ),
            default_gwp_source=os.getenv("GL_EC_DEFAULT_GWP_SOURCE", "AR5"),
            round_trip_factor=Decimal(
                os.getenv("GL_EC_ROUND_TRIP_FACTOR", "2.0")
            ),
            e_bicycle_ef_fraction=Decimal(
                os.getenv("GL_EC_E_BICYCLE_EF_FRACTION", "0.05")
            ),
        )


# =============================================================================
# SECTION 5: TELEWORK CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TeleworkConfig:
    """
    Telework/remote work configuration for Employee Commuting agent.

    Models the energy consumption and associated emissions from employees
    working from home. Supports device-level breakdowns (laptop, heating,
    cooling, lighting) and seasonal adjustment patterns for grid electricity
    emission factors.

    Seasonal adjustment methods:
    - FULL_SEASONAL: Applies monthly grid EF + heating/cooling profiles
    - SUMMER_WINTER: Simplified two-season model
    - ANNUAL_AVERAGE: No seasonal adjustment (constant annual EF)

    Grid EF sources:
    - IEA: International Energy Agency country-level factors
    - EPA_EGRID: US EPA eGRID subregional factors
    - DEFRA: UK DEFRA country-level factors
    - CUSTOM: User-provided custom emission factors

    Attributes:
        enabled: Enable telework emissions calculation (GL_EC_TELEWORK_ENABLED)
        default_daily_kwh: Default total daily energy consumption in kWh (GL_EC_TELEWORK_DAILY_KWH)
        laptop_kwh: Default daily laptop energy in kWh (GL_EC_TELEWORK_LAPTOP_KWH)
        monitor_kwh: Default daily external monitor energy in kWh (GL_EC_TELEWORK_MONITOR_KWH)
        heating_kwh: Default daily heating energy in kWh (GL_EC_TELEWORK_HEATING_KWH)
        cooling_kwh: Default daily cooling energy in kWh (GL_EC_TELEWORK_COOLING_KWH)
        lighting_kwh: Default daily lighting energy in kWh (GL_EC_TELEWORK_LIGHTING_KWH)
        internet_kwh: Default daily internet router energy in kWh (GL_EC_TELEWORK_INTERNET_KWH)
        seasonal_adjustment: Seasonal adjustment method (GL_EC_TELEWORK_SEASONAL_ADJUSTMENT)
        default_grid_ef_source: Default grid emission factor source (GL_EC_TELEWORK_GRID_EF_SOURCE)
        include_cooling: Include cooling energy in telework calculation (GL_EC_TELEWORK_INCLUDE_COOLING)
        summer_heating_fraction: Fraction of heating active in summer (GL_EC_TELEWORK_SUMMER_HEATING_FRACTION)
        winter_cooling_fraction: Fraction of cooling active in winter (GL_EC_TELEWORK_WINTER_COOLING_FRACTION)

    Example:
        >>> telework = TeleworkConfig(
        ...     enabled=True,
        ...     default_daily_kwh=Decimal("4.0"),
        ...     laptop_kwh=Decimal("0.3"),
        ...     heating_kwh=Decimal("3.5"),
        ...     lighting_kwh=Decimal("0.2"),
        ...     seasonal_adjustment="FULL_SEASONAL",
        ...     default_grid_ef_source="IEA"
        ... )
        >>> telework.default_daily_kwh
        Decimal('4.0')
    """

    enabled: bool = True
    default_daily_kwh: Decimal = Decimal("4.0")
    laptop_kwh: Decimal = Decimal("0.3")
    monitor_kwh: Decimal = Decimal("0.1")
    heating_kwh: Decimal = Decimal("3.5")
    cooling_kwh: Decimal = Decimal("1.5")
    lighting_kwh: Decimal = Decimal("0.2")
    internet_kwh: Decimal = Decimal("0.1")
    seasonal_adjustment: str = "FULL_SEASONAL"
    default_grid_ef_source: str = "IEA"
    include_cooling: bool = True
    summer_heating_fraction: Decimal = Decimal("0.0")
    winter_cooling_fraction: Decimal = Decimal("0.0")

    def validate(self) -> None:
        """
        Validate telework configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_daily_kwh < Decimal("0"):
            raise ValueError("default_daily_kwh must be >= 0")

        if self.default_daily_kwh > Decimal("50"):
            raise ValueError("default_daily_kwh must be <= 50")

        if self.laptop_kwh < Decimal("0"):
            raise ValueError("laptop_kwh must be >= 0")

        if self.laptop_kwh > Decimal("5"):
            raise ValueError("laptop_kwh must be <= 5")

        if self.monitor_kwh < Decimal("0"):
            raise ValueError("monitor_kwh must be >= 0")

        if self.monitor_kwh > Decimal("5"):
            raise ValueError("monitor_kwh must be <= 5")

        if self.heating_kwh < Decimal("0"):
            raise ValueError("heating_kwh must be >= 0")

        if self.heating_kwh > Decimal("30"):
            raise ValueError("heating_kwh must be <= 30")

        if self.cooling_kwh < Decimal("0"):
            raise ValueError("cooling_kwh must be >= 0")

        if self.cooling_kwh > Decimal("20"):
            raise ValueError("cooling_kwh must be <= 20")

        if self.lighting_kwh < Decimal("0"):
            raise ValueError("lighting_kwh must be >= 0")

        if self.lighting_kwh > Decimal("5"):
            raise ValueError("lighting_kwh must be <= 5")

        if self.internet_kwh < Decimal("0"):
            raise ValueError("internet_kwh must be >= 0")

        if self.internet_kwh > Decimal("2"):
            raise ValueError("internet_kwh must be <= 2")

        valid_seasonal = {"FULL_SEASONAL", "SUMMER_WINTER", "ANNUAL_AVERAGE"}
        if self.seasonal_adjustment not in valid_seasonal:
            raise ValueError(
                f"Invalid seasonal_adjustment '{self.seasonal_adjustment}'. "
                f"Must be one of {valid_seasonal}"
            )

        valid_grid_sources = {"IEA", "EPA_EGRID", "DEFRA", "CUSTOM"}
        if self.default_grid_ef_source not in valid_grid_sources:
            raise ValueError(
                f"Invalid default_grid_ef_source '{self.default_grid_ef_source}'. "
                f"Must be one of {valid_grid_sources}"
            )

        if (
            self.summer_heating_fraction < Decimal("0")
            or self.summer_heating_fraction > Decimal("1")
        ):
            raise ValueError("summer_heating_fraction must be between 0 and 1")

        if (
            self.winter_cooling_fraction < Decimal("0")
            or self.winter_cooling_fraction > Decimal("1")
        ):
            raise ValueError("winter_cooling_fraction must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "default_daily_kwh": str(self.default_daily_kwh),
            "laptop_kwh": str(self.laptop_kwh),
            "monitor_kwh": str(self.monitor_kwh),
            "heating_kwh": str(self.heating_kwh),
            "cooling_kwh": str(self.cooling_kwh),
            "lighting_kwh": str(self.lighting_kwh),
            "internet_kwh": str(self.internet_kwh),
            "seasonal_adjustment": self.seasonal_adjustment,
            "default_grid_ef_source": self.default_grid_ef_source,
            "include_cooling": self.include_cooling,
            "summer_heating_fraction": str(self.summer_heating_fraction),
            "winter_cooling_fraction": str(self.winter_cooling_fraction),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeleworkConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "default_daily_kwh",
            "laptop_kwh",
            "monitor_kwh",
            "heating_kwh",
            "cooling_kwh",
            "lighting_kwh",
            "internet_kwh",
            "summer_heating_fraction",
            "winter_cooling_fraction",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "TeleworkConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_EC_TELEWORK_ENABLED", "true").lower() == "true",
            default_daily_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_DAILY_KWH", "4.0")
            ),
            laptop_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_LAPTOP_KWH", "0.3")
            ),
            monitor_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_MONITOR_KWH", "0.1")
            ),
            heating_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_HEATING_KWH", "3.5")
            ),
            cooling_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_COOLING_KWH", "1.5")
            ),
            lighting_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_LIGHTING_KWH", "0.2")
            ),
            internet_kwh=Decimal(
                os.getenv("GL_EC_TELEWORK_INTERNET_KWH", "0.1")
            ),
            seasonal_adjustment=os.getenv(
                "GL_EC_TELEWORK_SEASONAL_ADJUSTMENT", "FULL_SEASONAL"
            ),
            default_grid_ef_source=os.getenv(
                "GL_EC_TELEWORK_GRID_EF_SOURCE", "IEA"
            ),
            include_cooling=os.getenv(
                "GL_EC_TELEWORK_INCLUDE_COOLING", "true"
            ).lower()
            == "true",
            summer_heating_fraction=Decimal(
                os.getenv("GL_EC_TELEWORK_SUMMER_HEATING_FRACTION", "0.0")
            ),
            winter_cooling_fraction=Decimal(
                os.getenv("GL_EC_TELEWORK_WINTER_COOLING_FRACTION", "0.0")
            ),
        )


# =============================================================================
# SECTION 6: SURVEY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SurveyConfig:
    """
    Survey configuration for Employee Commuting agent.

    Controls survey-based data collection and statistical extrapolation
    parameters. Supports various survey methods and enforces minimum
    sample sizes and response rates to ensure statistically valid results.

    Survey methods:
    - RANDOM_SAMPLE: Simple random sampling from employee population
    - STRATIFIED: Stratified sampling by department/location/role
    - CENSUS: Full population survey (all employees)
    - CLUSTER: Cluster sampling by office location/building

    Attributes:
        min_response_rate: Minimum acceptable response rate (GL_EC_SURVEY_MIN_RESPONSE_RATE)
        min_sample_size: Minimum sample size for valid extrapolation (GL_EC_SURVEY_MIN_SAMPLE_SIZE)
        confidence_level: Statistical confidence level (GL_EC_SURVEY_CONFIDENCE_LEVEL)
        z_score: Z-score corresponding to confidence level (GL_EC_SURVEY_Z_SCORE)
        max_extrapolation_factor: Maximum extrapolation multiplier (GL_EC_SURVEY_MAX_EXTRAPOLATION_FACTOR)
        default_survey_method: Default survey methodology (GL_EC_SURVEY_DEFAULT_METHOD)
        margin_of_error: Target margin of error (GL_EC_SURVEY_MARGIN_OF_ERROR)
        response_weighting: Enable response weighting by stratum (GL_EC_SURVEY_RESPONSE_WEIGHTING)
        outlier_removal: Enable statistical outlier removal from responses (GL_EC_SURVEY_OUTLIER_REMOVAL)
        outlier_z_threshold: Z-score threshold for outlier identification (GL_EC_SURVEY_OUTLIER_Z_THRESHOLD)
        allow_partial_responses: Allow incomplete survey responses (GL_EC_SURVEY_ALLOW_PARTIAL)
        min_completeness_rate: Minimum completeness for partial responses (GL_EC_SURVEY_MIN_COMPLETENESS)

    Example:
        >>> survey = SurveyConfig(
        ...     min_response_rate=Decimal("0.1"),
        ...     min_sample_size=30,
        ...     confidence_level=Decimal("0.95"),
        ...     z_score=Decimal("1.96"),
        ...     max_extrapolation_factor=Decimal("20.0"),
        ...     default_survey_method="RANDOM_SAMPLE"
        ... )
        >>> survey.confidence_level
        Decimal('0.95')
    """

    min_response_rate: Decimal = Decimal("0.1")
    min_sample_size: int = 30
    confidence_level: Decimal = Decimal("0.95")
    z_score: Decimal = Decimal("1.96")
    max_extrapolation_factor: Decimal = Decimal("20.0")
    default_survey_method: str = "RANDOM_SAMPLE"
    margin_of_error: Decimal = Decimal("0.05")
    response_weighting: bool = True
    outlier_removal: bool = True
    outlier_z_threshold: Decimal = Decimal("3.0")
    allow_partial_responses: bool = True
    min_completeness_rate: Decimal = Decimal("0.7")

    def validate(self) -> None:
        """
        Validate survey configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.min_response_rate < Decimal("0.01") or self.min_response_rate > Decimal("1.0"):
            raise ValueError("min_response_rate must be between 0.01 and 1.0")

        if self.min_sample_size < 1 or self.min_sample_size > 100000:
            raise ValueError("min_sample_size must be between 1 and 100000")

        if self.confidence_level < Decimal("0.5") or self.confidence_level > Decimal("0.999"):
            raise ValueError("confidence_level must be between 0.5 and 0.999")

        if self.z_score < Decimal("0.5") or self.z_score > Decimal("5.0"):
            raise ValueError("z_score must be between 0.5 and 5.0")

        if self.max_extrapolation_factor < Decimal("1.0") or self.max_extrapolation_factor > Decimal("100.0"):
            raise ValueError("max_extrapolation_factor must be between 1.0 and 100.0")

        valid_methods = {"RANDOM_SAMPLE", "STRATIFIED", "CENSUS", "CLUSTER"}
        if self.default_survey_method not in valid_methods:
            raise ValueError(
                f"Invalid default_survey_method '{self.default_survey_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.margin_of_error < Decimal("0.001") or self.margin_of_error > Decimal("0.5"):
            raise ValueError("margin_of_error must be between 0.001 and 0.5")

        if self.outlier_z_threshold < Decimal("1.0") or self.outlier_z_threshold > Decimal("10.0"):
            raise ValueError("outlier_z_threshold must be between 1.0 and 10.0")

        if self.min_completeness_rate < Decimal("0.1") or self.min_completeness_rate > Decimal("1.0"):
            raise ValueError("min_completeness_rate must be between 0.1 and 1.0")

        # Cross-field validation: z_score should correspond to confidence_level
        expected_z_scores = {
            Decimal("0.90"): Decimal("1.645"),
            Decimal("0.95"): Decimal("1.96"),
            Decimal("0.99"): Decimal("2.576"),
        }
        if self.confidence_level in expected_z_scores:
            expected_z = expected_z_scores[self.confidence_level]
            if abs(self.z_score - expected_z) > Decimal("0.1"):
                raise ValueError(
                    f"z_score {self.z_score} does not correspond to "
                    f"confidence_level {self.confidence_level} "
                    f"(expected approximately {expected_z})"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_response_rate": str(self.min_response_rate),
            "min_sample_size": self.min_sample_size,
            "confidence_level": str(self.confidence_level),
            "z_score": str(self.z_score),
            "max_extrapolation_factor": str(self.max_extrapolation_factor),
            "default_survey_method": self.default_survey_method,
            "margin_of_error": str(self.margin_of_error),
            "response_weighting": self.response_weighting,
            "outlier_removal": self.outlier_removal,
            "outlier_z_threshold": str(self.outlier_z_threshold),
            "allow_partial_responses": self.allow_partial_responses,
            "min_completeness_rate": str(self.min_completeness_rate),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurveyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "min_response_rate",
            "confidence_level",
            "z_score",
            "max_extrapolation_factor",
            "margin_of_error",
            "outlier_z_threshold",
            "min_completeness_rate",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SurveyConfig":
        """Load from environment variables."""
        return cls(
            min_response_rate=Decimal(
                os.getenv("GL_EC_SURVEY_MIN_RESPONSE_RATE", "0.1")
            ),
            min_sample_size=int(
                os.getenv("GL_EC_SURVEY_MIN_SAMPLE_SIZE", "30")
            ),
            confidence_level=Decimal(
                os.getenv("GL_EC_SURVEY_CONFIDENCE_LEVEL", "0.95")
            ),
            z_score=Decimal(
                os.getenv("GL_EC_SURVEY_Z_SCORE", "1.96")
            ),
            max_extrapolation_factor=Decimal(
                os.getenv("GL_EC_SURVEY_MAX_EXTRAPOLATION_FACTOR", "20.0")
            ),
            default_survey_method=os.getenv(
                "GL_EC_SURVEY_DEFAULT_METHOD", "RANDOM_SAMPLE"
            ),
            margin_of_error=Decimal(
                os.getenv("GL_EC_SURVEY_MARGIN_OF_ERROR", "0.05")
            ),
            response_weighting=os.getenv(
                "GL_EC_SURVEY_RESPONSE_WEIGHTING", "true"
            ).lower()
            == "true",
            outlier_removal=os.getenv(
                "GL_EC_SURVEY_OUTLIER_REMOVAL", "true"
            ).lower()
            == "true",
            outlier_z_threshold=Decimal(
                os.getenv("GL_EC_SURVEY_OUTLIER_Z_THRESHOLD", "3.0")
            ),
            allow_partial_responses=os.getenv(
                "GL_EC_SURVEY_ALLOW_PARTIAL", "true"
            ).lower()
            == "true",
            min_completeness_rate=Decimal(
                os.getenv("GL_EC_SURVEY_MIN_COMPLETENESS", "0.7")
            ),
        )


# =============================================================================
# SECTION 7: WORKING DAYS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class WorkingDaysConfig:
    """
    Working days configuration for Employee Commuting agent.

    Controls how annual working days are calculated for each employee or
    employee group. Supports regional calendar differences, holiday
    schedules, PTO (paid time off), and sick leave deductions.

    Supported regions:
    - GLOBAL: Generic 230-day default
    - US: United States (typically 250 - holidays - PTO)
    - UK: United Kingdom (typically 252 - bank holidays - leave)
    - EU: European Union average
    - APAC: Asia-Pacific average
    - LATAM: Latin America average
    - CUSTOM: User-provided working days

    Attributes:
        default_region: Default calendar region (GL_EC_WORKING_DAYS_REGION)
        default_working_days: Default annual working days (GL_EC_WORKING_DAYS_DEFAULT)
        include_holidays: Deduct public holidays (GL_EC_WORKING_DAYS_INCLUDE_HOLIDAYS)
        include_pto: Deduct PTO days (GL_EC_WORKING_DAYS_INCLUDE_PTO)
        include_sick: Deduct sick days (GL_EC_WORKING_DAYS_INCLUDE_SICK)
        default_holidays: Default number of public holidays (GL_EC_WORKING_DAYS_HOLIDAYS)
        default_pto_days: Default number of PTO days (GL_EC_WORKING_DAYS_PTO)
        default_sick_days: Default number of sick days (GL_EC_WORKING_DAYS_SICK)
        min_working_days: Minimum valid working days (GL_EC_WORKING_DAYS_MIN)
        max_working_days: Maximum valid working days (GL_EC_WORKING_DAYS_MAX)
        part_time_adjustment: Enable part-time working days adjustment (GL_EC_WORKING_DAYS_PART_TIME)
        default_part_time_fraction: Default part-time fraction (GL_EC_WORKING_DAYS_PT_FRACTION)

    Example:
        >>> working_days = WorkingDaysConfig(
        ...     default_region="GLOBAL",
        ...     default_working_days=230,
        ...     include_holidays=True,
        ...     include_pto=True,
        ...     include_sick=True
        ... )
        >>> working_days.default_working_days
        230
    """

    default_region: str = "GLOBAL"
    default_working_days: int = 230
    include_holidays: bool = True
    include_pto: bool = True
    include_sick: bool = True
    default_holidays: int = 10
    default_pto_days: int = 15
    default_sick_days: int = 5
    min_working_days: int = 50
    max_working_days: int = 365
    part_time_adjustment: bool = True
    default_part_time_fraction: Decimal = Decimal("0.5")

    def validate(self) -> None:
        """
        Validate working days configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_regions = {"GLOBAL", "US", "UK", "EU", "APAC", "LATAM", "CUSTOM"}
        if self.default_region not in valid_regions:
            raise ValueError(
                f"Invalid default_region '{self.default_region}'. "
                f"Must be one of {valid_regions}"
            )

        if self.default_working_days < 1 or self.default_working_days > 366:
            raise ValueError("default_working_days must be between 1 and 366")

        if self.default_holidays < 0 or self.default_holidays > 30:
            raise ValueError("default_holidays must be between 0 and 30")

        if self.default_pto_days < 0 or self.default_pto_days > 60:
            raise ValueError("default_pto_days must be between 0 and 60")

        if self.default_sick_days < 0 or self.default_sick_days > 30:
            raise ValueError("default_sick_days must be between 0 and 30")

        if self.min_working_days < 1:
            raise ValueError("min_working_days must be >= 1")

        if self.max_working_days > 366:
            raise ValueError("max_working_days must be <= 366")

        if self.min_working_days > self.max_working_days:
            raise ValueError("min_working_days must be <= max_working_days")

        if (
            self.default_part_time_fraction < Decimal("0.1")
            or self.default_part_time_fraction > Decimal("1.0")
        ):
            raise ValueError("default_part_time_fraction must be between 0.1 and 1.0")

        # Cross-field: ensure default_working_days is within min/max range
        if self.default_working_days < self.min_working_days:
            raise ValueError(
                f"default_working_days ({self.default_working_days}) "
                f"must be >= min_working_days ({self.min_working_days})"
            )

        if self.default_working_days > self.max_working_days:
            raise ValueError(
                f"default_working_days ({self.default_working_days}) "
                f"must be <= max_working_days ({self.max_working_days})"
            )

    def get_effective_working_days(self) -> int:
        """
        Calculate effective working days after deductions.

        Returns:
            Net working days after holiday, PTO, and sick deductions
        """
        days = self.default_working_days
        if self.include_holidays:
            days -= self.default_holidays
        if self.include_pto:
            days -= self.default_pto_days
        if self.include_sick:
            days -= self.default_sick_days
        return max(days, self.min_working_days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_region": self.default_region,
            "default_working_days": self.default_working_days,
            "include_holidays": self.include_holidays,
            "include_pto": self.include_pto,
            "include_sick": self.include_sick,
            "default_holidays": self.default_holidays,
            "default_pto_days": self.default_pto_days,
            "default_sick_days": self.default_sick_days,
            "min_working_days": self.min_working_days,
            "max_working_days": self.max_working_days,
            "part_time_adjustment": self.part_time_adjustment,
            "default_part_time_fraction": str(self.default_part_time_fraction),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingDaysConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_part_time_fraction" in data_copy:
            data_copy["default_part_time_fraction"] = Decimal(
                str(data_copy["default_part_time_fraction"])
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "WorkingDaysConfig":
        """Load from environment variables."""
        return cls(
            default_region=os.getenv("GL_EC_WORKING_DAYS_REGION", "GLOBAL"),
            default_working_days=int(
                os.getenv("GL_EC_WORKING_DAYS_DEFAULT", "230")
            ),
            include_holidays=os.getenv(
                "GL_EC_WORKING_DAYS_INCLUDE_HOLIDAYS", "true"
            ).lower()
            == "true",
            include_pto=os.getenv(
                "GL_EC_WORKING_DAYS_INCLUDE_PTO", "true"
            ).lower()
            == "true",
            include_sick=os.getenv(
                "GL_EC_WORKING_DAYS_INCLUDE_SICK", "true"
            ).lower()
            == "true",
            default_holidays=int(
                os.getenv("GL_EC_WORKING_DAYS_HOLIDAYS", "10")
            ),
            default_pto_days=int(
                os.getenv("GL_EC_WORKING_DAYS_PTO", "15")
            ),
            default_sick_days=int(
                os.getenv("GL_EC_WORKING_DAYS_SICK", "5")
            ),
            min_working_days=int(
                os.getenv("GL_EC_WORKING_DAYS_MIN", "50")
            ),
            max_working_days=int(
                os.getenv("GL_EC_WORKING_DAYS_MAX", "365")
            ),
            part_time_adjustment=os.getenv(
                "GL_EC_WORKING_DAYS_PART_TIME", "true"
            ).lower()
            == "true",
            default_part_time_fraction=Decimal(
                os.getenv("GL_EC_WORKING_DAYS_PT_FRACTION", "0.5")
            ),
        )


# =============================================================================
# SECTION 8: SPEND CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendConfig:
    """
    Spend-based calculation configuration for Employee Commuting agent.

    Controls the spend-based EEIO fallback calculation method when
    employee-level commute data is not available. Supports currency
    conversion, CPI deflation, and margin removal.

    EEIO sources:
    - USEEIO: US Environmentally Extended Input-Output model
    - EXIOBASE: Multi-regional EEIO model
    - DEFRA: UK DEFRA spend-based emission factors

    Attributes:
        default_currency: Default currency for spend data (GL_EC_SPEND_CURRENCY)
        base_year: Base year for CPI deflation (GL_EC_SPEND_BASE_YEAR)
        eeio_source: Default EEIO model source (GL_EC_SPEND_EEIO_SOURCE)
        margin_removal_rate: Margin removal rate for spend-to-output conversion (GL_EC_SPEND_MARGIN_RATE)
        enable_cpi_deflation: Enable CPI deflation for multi-year data (GL_EC_SPEND_CPI_DEFLATION)
        enable_currency_conversion: Enable multi-currency support (GL_EC_SPEND_CURRENCY_CONVERSION)
        transport_sector_code: Default NAICS/ISIC sector code for commuting (GL_EC_SPEND_SECTOR_CODE)
        purchaser_price_adjustment: Adjust from purchaser to producer prices (GL_EC_SPEND_PPA)

    Example:
        >>> spend = SpendConfig(
        ...     default_currency="USD",
        ...     base_year=2021,
        ...     eeio_source="USEEIO",
        ...     margin_removal_rate=Decimal("0.15")
        ... )
        >>> spend.eeio_source
        'USEEIO'
    """

    default_currency: str = "USD"
    base_year: int = 2021
    eeio_source: str = "USEEIO"
    margin_removal_rate: Decimal = Decimal("0.15")
    enable_cpi_deflation: bool = True
    enable_currency_conversion: bool = True
    transport_sector_code: str = "485000"
    purchaser_price_adjustment: bool = True

    def validate(self) -> None:
        """
        Validate spend configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_currencies = {
            "USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD",
            "CHF", "SEK", "NOK", "DKK", "INR", "BRL", "KRW",
            "MXN", "ZAR", "SGD", "HKD", "NZD", "TWD",
        }
        if self.default_currency not in valid_currencies:
            raise ValueError(
                f"Invalid default_currency '{self.default_currency}'. "
                f"Must be one of {valid_currencies}"
            )

        if self.base_year < 2000 or self.base_year > 2030:
            raise ValueError("base_year must be between 2000 and 2030")

        valid_eeio_sources = {"USEEIO", "EXIOBASE", "DEFRA"}
        if self.eeio_source not in valid_eeio_sources:
            raise ValueError(
                f"Invalid eeio_source '{self.eeio_source}'. "
                f"Must be one of {valid_eeio_sources}"
            )

        if self.margin_removal_rate < Decimal("0.0") or self.margin_removal_rate > Decimal("0.5"):
            raise ValueError("margin_removal_rate must be between 0.0 and 0.5")

        if not self.transport_sector_code:
            raise ValueError("transport_sector_code cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_currency": self.default_currency,
            "base_year": self.base_year,
            "eeio_source": self.eeio_source,
            "margin_removal_rate": str(self.margin_removal_rate),
            "enable_cpi_deflation": self.enable_cpi_deflation,
            "enable_currency_conversion": self.enable_currency_conversion,
            "transport_sector_code": self.transport_sector_code,
            "purchaser_price_adjustment": self.purchaser_price_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "margin_removal_rate" in data_copy:
            data_copy["margin_removal_rate"] = Decimal(
                str(data_copy["margin_removal_rate"])
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendConfig":
        """Load from environment variables."""
        return cls(
            default_currency=os.getenv("GL_EC_SPEND_CURRENCY", "USD"),
            base_year=int(os.getenv("GL_EC_SPEND_BASE_YEAR", "2021")),
            eeio_source=os.getenv("GL_EC_SPEND_EEIO_SOURCE", "USEEIO"),
            margin_removal_rate=Decimal(
                os.getenv("GL_EC_SPEND_MARGIN_RATE", "0.15")
            ),
            enable_cpi_deflation=os.getenv(
                "GL_EC_SPEND_CPI_DEFLATION", "true"
            ).lower()
            == "true",
            enable_currency_conversion=os.getenv(
                "GL_EC_SPEND_CURRENCY_CONVERSION", "true"
            ).lower()
            == "true",
            transport_sector_code=os.getenv(
                "GL_EC_SPEND_SECTOR_CODE", "485000"
            ),
            purchaser_price_adjustment=os.getenv(
                "GL_EC_SPEND_PPA", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 9: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Employee Commuting agent.

    Controls regulatory framework alignment, strict mode enforcement,
    and disclosure requirements for Scope 3 Category 7 reporting.

    Supported frameworks:
    - GHG_PROTOCOL_SCOPE3: GHG Protocol Corporate Value Chain (Scope 3) Standard
    - ISO_14064: ISO 14064-1:2018 Organizational GHG quantification
    - CSRD_ESRS_E1: EU Corporate Sustainability Reporting Directive (Climate)
    - CDP: Carbon Disclosure Project Climate Change questionnaire
    - SBTI: Science Based Targets initiative
    - GRI_305: Global Reporting Initiative 305 (Emissions)
    - EPA_CCL: US EPA Center for Corporate Leadership

    Attributes:
        compliance_frameworks: Comma-separated enabled frameworks (GL_EC_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict validation for all frameworks (GL_EC_COMPLIANCE_STRICT_MODE)
        telework_disclosure_required: Require telework emissions disclosure (GL_EC_COMPLIANCE_TELEWORK_DISCLOSURE)
        mode_share_required: Require commute mode share breakdown (GL_EC_COMPLIANCE_MODE_SHARE)
        double_counting_check: Enable double-counting prevention (GL_EC_COMPLIANCE_DOUBLE_COUNTING)
        boundary_enforcement: Enforce Scope 3 Cat 7 boundary rules (GL_EC_COMPLIANCE_BOUNDARY)
        data_quality_required: Require DQI scoring for all inputs (GL_EC_COMPLIANCE_DQI_REQUIRED)
        minimum_dqi_score: Minimum acceptable DQI score (GL_EC_COMPLIANCE_MIN_DQI)

    Example:
        >>> compliance = ComplianceConfig(
        ...     compliance_frameworks="GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1",
        ...     strict_mode=False,
        ...     telework_disclosure_required=True,
        ...     mode_share_required=True
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E1']
    """

    compliance_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI_305,EPA_CCL"
    )
    strict_mode: bool = False
    telework_disclosure_required: bool = True
    mode_share_required: bool = True
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    data_quality_required: bool = True
    minimum_dqi_score: Decimal = Decimal("2.0")

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
            "GRI_305",
            "EPA_CCL",
        }

        frameworks = self.get_frameworks()
        if not frameworks:
            raise ValueError("At least one compliance framework must be enabled")

        for framework in frameworks:
            if framework not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{framework}'. Must be one of {valid_frameworks}"
                )

        if self.minimum_dqi_score < Decimal("1.0") or self.minimum_dqi_score > Decimal("5.0"):
            raise ValueError("minimum_dqi_score must be between 1.0 and 5.0")

    def get_frameworks(self) -> List[str]:
        """Parse compliance frameworks string into list."""
        return [f.strip() for f in self.compliance_frameworks.split(",") if f.strip()]

    def has_framework(self, framework: str) -> bool:
        """
        Check if a specific framework is enabled.

        Args:
            framework: Framework identifier to check

        Returns:
            True if the framework is in the enabled list
        """
        return framework in self.get_frameworks()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compliance_frameworks": self.compliance_frameworks,
            "strict_mode": self.strict_mode,
            "telework_disclosure_required": self.telework_disclosure_required,
            "mode_share_required": self.mode_share_required,
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "data_quality_required": self.data_quality_required,
            "minimum_dqi_score": str(self.minimum_dqi_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "minimum_dqi_score" in data_copy:
            data_copy["minimum_dqi_score"] = Decimal(
                str(data_copy["minimum_dqi_score"])
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            compliance_frameworks=os.getenv(
                "GL_EC_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI_305,EPA_CCL",
            ),
            strict_mode=os.getenv(
                "GL_EC_COMPLIANCE_STRICT_MODE", "false"
            ).lower()
            == "true",
            telework_disclosure_required=os.getenv(
                "GL_EC_COMPLIANCE_TELEWORK_DISCLOSURE", "true"
            ).lower()
            == "true",
            mode_share_required=os.getenv(
                "GL_EC_COMPLIANCE_MODE_SHARE", "true"
            ).lower()
            == "true",
            double_counting_check=os.getenv(
                "GL_EC_COMPLIANCE_DOUBLE_COUNTING", "true"
            ).lower()
            == "true",
            boundary_enforcement=os.getenv(
                "GL_EC_COMPLIANCE_BOUNDARY", "true"
            ).lower()
            == "true",
            data_quality_required=os.getenv(
                "GL_EC_COMPLIANCE_DQI_REQUIRED", "true"
            ).lower()
            == "true",
            minimum_dqi_score=Decimal(
                os.getenv("GL_EC_COMPLIANCE_MIN_DQI", "2.0")
            ),
        )


# =============================================================================
# SECTION 10: EMISSION FACTOR SOURCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EFSourceConfig:
    """
    Emission Factor Source configuration for Employee Commuting agent.

    Defines the emission factor lookup hierarchy and fallback chain.
    The agent walks the hierarchy from highest-priority to lowest-priority
    source when resolving emission factors for a given commute mode.

    EF hierarchy (default order):
    1. EMPLOYEE - Employee-specific data (highest quality)
    2. DEFRA - UK DEFRA BEIS GHG conversion factors
    3. EPA - US EPA emission factors
    4. IEA - International Energy Agency factors
    5. CENSUS - National census commuting data averages
    6. EEIO - Spend-based EEIO fallback (lowest quality)

    Attributes:
        hierarchy: Comma-separated EF source priority list (GL_EC_EF_HIERARCHY)
        default_source: Default EF source when hierarchy is not applicable (GL_EC_EF_DEFAULT_SOURCE)
        allow_custom: Allow custom/user-provided emission factors (GL_EC_EF_ALLOW_CUSTOM)
        custom_ef_path: Path to custom EF file (GL_EC_EF_CUSTOM_PATH)
        cache_ef_lookups: Cache emission factor lookups (GL_EC_EF_CACHE_LOOKUPS)
        ef_year: Target year for emission factors (GL_EC_EF_YEAR)
        wtt_source: Well-to-tank emission factor source (GL_EC_EF_WTT_SOURCE)
        fallback_enabled: Enable automatic fallback to lower-priority sources (GL_EC_EF_FALLBACK_ENABLED)

    Example:
        >>> ef = EFSourceConfig(
        ...     hierarchy="EMPLOYEE,DEFRA,EPA,IEA,CENSUS,EEIO",
        ...     default_source="DEFRA",
        ...     allow_custom=True
        ... )
        >>> ef.get_hierarchy()
        ['EMPLOYEE', 'DEFRA', 'EPA', 'IEA', 'CENSUS', 'EEIO']
    """

    hierarchy: str = "EMPLOYEE,DEFRA,EPA,IEA,CENSUS,EEIO"
    default_source: str = "DEFRA"
    allow_custom: bool = True
    custom_ef_path: Optional[str] = None
    cache_ef_lookups: bool = True
    ef_year: int = 2024
    wtt_source: str = "DEFRA"
    fallback_enabled: bool = True

    def validate(self) -> None:
        """
        Validate EF source configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_sources = {
            "EMPLOYEE",
            "DEFRA",
            "EPA",
            "IEA",
            "CENSUS",
            "EEIO",
            "CUSTOM",
        }

        hierarchy_list = self.get_hierarchy()
        if not hierarchy_list:
            raise ValueError("At least one EF source must be in the hierarchy")

        for source in hierarchy_list:
            if source not in valid_sources:
                raise ValueError(
                    f"Invalid EF source '{source}' in hierarchy. "
                    f"Must be one of {valid_sources}"
                )

        if self.default_source not in valid_sources:
            raise ValueError(
                f"Invalid default_source '{self.default_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.default_source == "CUSTOM" and not self.allow_custom:
            raise ValueError(
                "default_source is CUSTOM but allow_custom is False"
            )

        if self.default_source == "CUSTOM" and not self.custom_ef_path:
            raise ValueError(
                "custom_ef_path must be set when default_source is CUSTOM"
            )

        if self.ef_year < 2000 or self.ef_year > 2030:
            raise ValueError("ef_year must be between 2000 and 2030")

        valid_wtt_sources = {"DEFRA", "EPA", "IEA", "CUSTOM"}
        if self.wtt_source not in valid_wtt_sources:
            raise ValueError(
                f"Invalid wtt_source '{self.wtt_source}'. "
                f"Must be one of {valid_wtt_sources}"
            )

    def get_hierarchy(self) -> List[str]:
        """Parse hierarchy string into ordered list."""
        return [s.strip() for s in self.hierarchy.split(",") if s.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hierarchy": self.hierarchy,
            "default_source": self.default_source,
            "allow_custom": self.allow_custom,
            "custom_ef_path": self.custom_ef_path,
            "cache_ef_lookups": self.cache_ef_lookups,
            "ef_year": self.ef_year,
            "wtt_source": self.wtt_source,
            "fallback_enabled": self.fallback_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EFSourceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "EFSourceConfig":
        """Load from environment variables."""
        return cls(
            hierarchy=os.getenv(
                "GL_EC_EF_HIERARCHY", "EMPLOYEE,DEFRA,EPA,IEA,CENSUS,EEIO"
            ),
            default_source=os.getenv("GL_EC_EF_DEFAULT_SOURCE", "DEFRA"),
            allow_custom=os.getenv("GL_EC_EF_ALLOW_CUSTOM", "true").lower()
            == "true",
            custom_ef_path=os.getenv("GL_EC_EF_CUSTOM_PATH"),
            cache_ef_lookups=os.getenv(
                "GL_EC_EF_CACHE_LOOKUPS", "true"
            ).lower()
            == "true",
            ef_year=int(os.getenv("GL_EC_EF_YEAR", "2024")),
            wtt_source=os.getenv("GL_EC_EF_WTT_SOURCE", "DEFRA"),
            fallback_enabled=os.getenv(
                "GL_EC_EF_FALLBACK_ENABLED", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 11: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty configuration for Employee Commuting agent.

    Controls Monte Carlo simulation parameters and uncertainty
    quantification for commuting emission estimates.

    Uncertainty methods:
    - MONTE_CARLO: Monte Carlo simulation with configurable iterations
    - IPCC_DEFAULT: IPCC default uncertainty ranges by source quality
    - BOOTSTRAP: Non-parametric bootstrap resampling
    - ANALYTICAL: Analytical error propagation (GUM method)
    - NONE: No uncertainty quantification

    Attributes:
        method: Uncertainty quantification method (GL_EC_UNCERTAINTY_METHOD)
        iterations: Monte Carlo simulation iterations (GL_EC_UNCERTAINTY_ITERATIONS)
        confidence_level: Confidence level for intervals (GL_EC_UNCERTAINTY_CONFIDENCE)
        seed: Random seed for reproducibility (GL_EC_UNCERTAINTY_SEED)
        include_ef_uncertainty: Include EF uncertainty in propagation (GL_EC_UNCERTAINTY_INCLUDE_EF)
        include_activity_uncertainty: Include activity data uncertainty (GL_EC_UNCERTAINTY_INCLUDE_ACTIVITY)
        include_survey_uncertainty: Include survey sampling uncertainty (GL_EC_UNCERTAINTY_INCLUDE_SURVEY)
        distribution_type: Default probability distribution (GL_EC_UNCERTAINTY_DISTRIBUTION)

    Example:
        >>> uncertainty = UncertaintyConfig(
        ...     method="MONTE_CARLO",
        ...     iterations=10000,
        ...     confidence_level=Decimal("0.95"),
        ...     seed=42
        ... )
        >>> uncertainty.method
        'MONTE_CARLO'
    """

    method: str = "MONTE_CARLO"
    iterations: int = 10000
    confidence_level: Decimal = Decimal("0.95")
    seed: int = 42
    include_ef_uncertainty: bool = True
    include_activity_uncertainty: bool = True
    include_survey_uncertainty: bool = True
    distribution_type: str = "LOGNORMAL"

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {
            "MONTE_CARLO",
            "IPCC_DEFAULT",
            "BOOTSTRAP",
            "ANALYTICAL",
            "NONE",
        }
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.iterations < 100 or self.iterations > 1000000:
            raise ValueError("iterations must be between 100 and 1000000")

        if self.confidence_level < Decimal("0.5") or self.confidence_level > Decimal("0.999"):
            raise ValueError("confidence_level must be between 0.5 and 0.999")

        if self.seed < 0:
            raise ValueError("seed must be >= 0")

        valid_distributions = {"NORMAL", "LOGNORMAL", "UNIFORM", "TRIANGULAR", "BETA"}
        if self.distribution_type not in valid_distributions:
            raise ValueError(
                f"Invalid distribution_type '{self.distribution_type}'. "
                f"Must be one of {valid_distributions}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "iterations": self.iterations,
            "confidence_level": str(self.confidence_level),
            "seed": self.seed,
            "include_ef_uncertainty": self.include_ef_uncertainty,
            "include_activity_uncertainty": self.include_activity_uncertainty,
            "include_survey_uncertainty": self.include_survey_uncertainty,
            "distribution_type": self.distribution_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "confidence_level" in data_copy:
            data_copy["confidence_level"] = Decimal(
                str(data_copy["confidence_level"])
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        return cls(
            method=os.getenv("GL_EC_UNCERTAINTY_METHOD", "MONTE_CARLO"),
            iterations=int(
                os.getenv("GL_EC_UNCERTAINTY_ITERATIONS", "10000")
            ),
            confidence_level=Decimal(
                os.getenv("GL_EC_UNCERTAINTY_CONFIDENCE", "0.95")
            ),
            seed=int(os.getenv("GL_EC_UNCERTAINTY_SEED", "42")),
            include_ef_uncertainty=os.getenv(
                "GL_EC_UNCERTAINTY_INCLUDE_EF", "true"
            ).lower()
            == "true",
            include_activity_uncertainty=os.getenv(
                "GL_EC_UNCERTAINTY_INCLUDE_ACTIVITY", "true"
            ).lower()
            == "true",
            include_survey_uncertainty=os.getenv(
                "GL_EC_UNCERTAINTY_INCLUDE_SURVEY", "true"
            ).lower()
            == "true",
            distribution_type=os.getenv(
                "GL_EC_UNCERTAINTY_DISTRIBUTION", "LOGNORMAL"
            ),
        )


# =============================================================================
# SECTION 12: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Employee Commuting agent.

    Controls in-memory and Redis-based caching for emission factor lookups,
    calculation results, and survey data aggregations.

    Attributes:
        enabled: Enable caching layer (GL_EC_CACHE_ENABLED)
        ttl: Cache TTL in seconds (GL_EC_CACHE_TTL)
        max_size: Maximum number of cache entries (GL_EC_CACHE_MAX_SIZE)
        warm_on_startup: Pre-warm cache on agent startup (GL_EC_CACHE_WARM_ON_STARTUP)
        cache_ef_lookups: Cache emission factor lookups (GL_EC_CACHE_EF_LOOKUPS)
        cache_calculations: Cache intermediate calculation results (GL_EC_CACHE_CALCULATIONS)
        cache_survey_data: Cache survey aggregation results (GL_EC_CACHE_SURVEY_DATA)
        eviction_policy: Cache eviction policy (GL_EC_CACHE_EVICTION_POLICY)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl=3600,
        ...     max_size=10000,
        ...     warm_on_startup=True
        ... )
        >>> cache.ttl
        3600
    """

    enabled: bool = True
    ttl: int = 3600
    max_size: int = 10000
    warm_on_startup: bool = True
    cache_ef_lookups: bool = True
    cache_calculations: bool = True
    cache_survey_data: bool = True
    eviction_policy: str = "LRU"

    def validate(self) -> None:
        """
        Validate cache configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.ttl < 1 or self.ttl > 86400:
            raise ValueError("ttl must be between 1 and 86400 (24 hours)")

        if self.max_size < 1 or self.max_size > 1000000:
            raise ValueError("max_size must be between 1 and 1000000")

        valid_eviction_policies = {"LRU", "LFU", "FIFO", "TTL"}
        if self.eviction_policy not in valid_eviction_policies:
            raise ValueError(
                f"Invalid eviction_policy '{self.eviction_policy}'. "
                f"Must be one of {valid_eviction_policies}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ttl": self.ttl,
            "max_size": self.max_size,
            "warm_on_startup": self.warm_on_startup,
            "cache_ef_lookups": self.cache_ef_lookups,
            "cache_calculations": self.cache_calculations,
            "cache_survey_data": self.cache_survey_data,
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
            enabled=os.getenv("GL_EC_CACHE_ENABLED", "true").lower() == "true",
            ttl=int(os.getenv("GL_EC_CACHE_TTL", "3600")),
            max_size=int(os.getenv("GL_EC_CACHE_MAX_SIZE", "10000")),
            warm_on_startup=os.getenv(
                "GL_EC_CACHE_WARM_ON_STARTUP", "true"
            ).lower()
            == "true",
            cache_ef_lookups=os.getenv(
                "GL_EC_CACHE_EF_LOOKUPS", "true"
            ).lower()
            == "true",
            cache_calculations=os.getenv(
                "GL_EC_CACHE_CALCULATIONS", "true"
            ).lower()
            == "true",
            cache_survey_data=os.getenv(
                "GL_EC_CACHE_SURVEY_DATA", "true"
            ).lower()
            == "true",
            eviction_policy=os.getenv("GL_EC_CACHE_EVICTION_POLICY", "LRU"),
        )


# =============================================================================
# SECTION 13: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    API configuration for Employee Commuting agent.

    Controls the REST API endpoint settings, pagination, rate limiting,
    and request/response parameters.

    Attributes:
        prefix: API route prefix (GL_EC_API_PREFIX)
        page_size: Default page size for list endpoints (GL_EC_API_PAGE_SIZE)
        max_page_size: Maximum allowed page size (GL_EC_API_MAX_PAGE_SIZE)
        rate_limit: Requests per minute rate limit (GL_EC_API_RATE_LIMIT)
        timeout: API request timeout in seconds (GL_EC_API_TIMEOUT)
        enable_bulk: Enable bulk calculation endpoint (GL_EC_API_ENABLE_BULK)
        max_bulk_size: Maximum records in a single bulk request (GL_EC_API_MAX_BULK_SIZE)
        cors_enabled: Enable CORS headers (GL_EC_API_CORS_ENABLED)
        cors_origins: Comma-separated allowed CORS origins (GL_EC_API_CORS_ORIGINS)

    Example:
        >>> api = APIConfig(
        ...     prefix="/api/v1/employee-commuting",
        ...     page_size=50,
        ...     max_page_size=500,
        ...     rate_limit=100
        ... )
        >>> api.prefix
        '/api/v1/employee-commuting'
    """

    prefix: str = "/api/v1/employee-commuting"
    page_size: int = 50
    max_page_size: int = 500
    rate_limit: int = 100
    timeout: int = 300
    enable_bulk: bool = True
    max_bulk_size: int = 1000
    cors_enabled: bool = True
    cors_origins: str = "*"

    def validate(self) -> None:
        """
        Validate API configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        if not self.prefix.startswith("/"):
            raise ValueError("prefix must start with '/'")

        if self.page_size < 1 or self.page_size > 1000:
            raise ValueError("page_size must be between 1 and 1000")

        if self.max_page_size < 1 or self.max_page_size > 10000:
            raise ValueError("max_page_size must be between 1 and 10000")

        if self.page_size > self.max_page_size:
            raise ValueError("page_size must be <= max_page_size")

        if self.rate_limit < 1 or self.rate_limit > 10000:
            raise ValueError("rate_limit must be between 1 and 10000")

        if self.timeout < 1 or self.timeout > 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")

        if self.max_bulk_size < 1 or self.max_bulk_size > 50000:
            raise ValueError("max_bulk_size must be between 1 and 50000")

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins string into list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prefix": self.prefix,
            "page_size": self.page_size,
            "max_page_size": self.max_page_size,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
            "enable_bulk": self.enable_bulk,
            "max_bulk_size": self.max_bulk_size,
            "cors_enabled": self.cors_enabled,
            "cors_origins": self.cors_origins,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load from environment variables."""
        return cls(
            prefix=os.getenv(
                "GL_EC_API_PREFIX", "/api/v1/employee-commuting"
            ),
            page_size=int(os.getenv("GL_EC_API_PAGE_SIZE", "50")),
            max_page_size=int(os.getenv("GL_EC_API_MAX_PAGE_SIZE", "500")),
            rate_limit=int(os.getenv("GL_EC_API_RATE_LIMIT", "100")),
            timeout=int(os.getenv("GL_EC_API_TIMEOUT", "300")),
            enable_bulk=os.getenv("GL_EC_API_ENABLE_BULK", "true").lower()
            == "true",
            max_bulk_size=int(os.getenv("GL_EC_API_MAX_BULK_SIZE", "1000")),
            cors_enabled=os.getenv("GL_EC_API_CORS_ENABLED", "true").lower()
            == "true",
            cors_origins=os.getenv("GL_EC_API_CORS_ORIGINS", "*"),
        )


# =============================================================================
# SECTION 14: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Employee Commuting agent.

    Controls SHA-256 (or other algorithm) hashing for complete audit trails,
    chain validation for sequential calculation steps, and intermediate
    result storage for full reproducibility.

    Attributes:
        enabled: Enable provenance tracking (GL_EC_PROVENANCE_ENABLED)
        hash_algorithm: Hash algorithm for provenance (GL_EC_PROVENANCE_ALGORITHM)
        chain_validation: Enable chain hash validation (GL_EC_PROVENANCE_CHAIN_VALIDATION)
        store_intermediates: Store intermediate calculation hashes (GL_EC_PROVENANCE_STORE_INTERMEDIATES)
        include_config_hash: Include configuration hash in provenance (GL_EC_PROVENANCE_INCLUDE_CONFIG)
        include_ef_hash: Include emission factor hash in provenance (GL_EC_PROVENANCE_INCLUDE_EF)
        include_survey_hash: Include survey data hash in provenance (GL_EC_PROVENANCE_INCLUDE_SURVEY)
        retention_days: Days to retain provenance records (GL_EC_PROVENANCE_RETENTION_DAYS)

    Example:
        >>> provenance = ProvenanceConfig(
        ...     enabled=True,
        ...     hash_algorithm="sha256",
        ...     chain_validation=True,
        ...     store_intermediates=True
        ... )
        >>> provenance.hash_algorithm
        'sha256'
    """

    enabled: bool = True
    hash_algorithm: str = "sha256"
    chain_validation: bool = True
    store_intermediates: bool = True
    include_config_hash: bool = True
    include_ef_hash: bool = True
    include_survey_hash: bool = True
    retention_days: int = 365

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

        if self.retention_days < 1 or self.retention_days > 3650:
            raise ValueError("retention_days must be between 1 and 3650 (10 years)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "hash_algorithm": self.hash_algorithm,
            "chain_validation": self.chain_validation,
            "store_intermediates": self.store_intermediates,
            "include_config_hash": self.include_config_hash,
            "include_ef_hash": self.include_ef_hash,
            "include_survey_hash": self.include_survey_hash,
            "retention_days": self.retention_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_EC_PROVENANCE_ENABLED", "true").lower()
            == "true",
            hash_algorithm=os.getenv("GL_EC_PROVENANCE_ALGORITHM", "sha256"),
            chain_validation=os.getenv(
                "GL_EC_PROVENANCE_CHAIN_VALIDATION", "true"
            ).lower()
            == "true",
            store_intermediates=os.getenv(
                "GL_EC_PROVENANCE_STORE_INTERMEDIATES", "true"
            ).lower()
            == "true",
            include_config_hash=os.getenv(
                "GL_EC_PROVENANCE_INCLUDE_CONFIG", "true"
            ).lower()
            == "true",
            include_ef_hash=os.getenv(
                "GL_EC_PROVENANCE_INCLUDE_EF", "true"
            ).lower()
            == "true",
            include_survey_hash=os.getenv(
                "GL_EC_PROVENANCE_INCLUDE_SURVEY", "true"
            ).lower()
            == "true",
            retention_days=int(
                os.getenv("GL_EC_PROVENANCE_RETENTION_DAYS", "365")
            ),
        )


# =============================================================================
# SECTION 15: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Employee Commuting agent.

    Controls Prometheus-compatible metrics collection including histograms,
    counters, and gauges for observability and performance monitoring.

    Attributes:
        enabled: Enable metrics collection (GL_EC_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_EC_METRICS_PREFIX)
        collect_histograms: Collect histogram metrics (GL_EC_METRICS_COLLECT_HISTOGRAMS)
        histogram_buckets: Histogram bucket boundaries (GL_EC_METRICS_HISTOGRAM_BUCKETS)
        collect_per_mode: Collect per-commute-mode metrics (GL_EC_METRICS_COLLECT_PER_MODE)
        collect_per_framework: Collect per-framework compliance metrics (GL_EC_METRICS_COLLECT_PER_FRAMEWORK)
        collection_interval: Metrics collection interval in seconds (GL_EC_METRICS_INTERVAL)
        include_survey_metrics: Include survey response rate metrics (GL_EC_METRICS_INCLUDE_SURVEY)

    Example:
        >>> metrics = MetricsConfig(
        ...     enabled=True,
        ...     prefix="gl_ec_",
        ...     collect_histograms=True,
        ...     histogram_buckets="0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_buckets()
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    """

    enabled: bool = True
    prefix: str = "gl_ec_"
    collect_histograms: bool = True
    histogram_buckets: str = "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
    collect_per_mode: bool = True
    collect_per_framework: bool = True
    collection_interval: int = 60
    include_survey_metrics: bool = True

    def validate(self) -> None:
        """
        Validate metrics configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        if not self.prefix.endswith("_"):
            raise ValueError("prefix must end with '_'")

        # Validate buckets format
        try:
            buckets = self.get_buckets()
            if not buckets:
                raise ValueError("At least one bucket must be defined")
            for bucket in buckets:
                if bucket <= 0:
                    raise ValueError("All buckets must be positive")
            # Verify buckets are sorted ascending
            for i in range(1, len(buckets)):
                if buckets[i] <= buckets[i - 1]:
                    raise ValueError("Histogram buckets must be in strictly ascending order")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid histogram_buckets format: {e}")

        if self.collection_interval < 1 or self.collection_interval > 3600:
            raise ValueError("collection_interval must be between 1 and 3600 seconds")

    def get_buckets(self) -> List[float]:
        """Parse histogram buckets string into list of floats."""
        return [float(x.strip()) for x in self.histogram_buckets.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "collect_histograms": self.collect_histograms,
            "histogram_buckets": self.histogram_buckets,
            "collect_per_mode": self.collect_per_mode,
            "collect_per_framework": self.collect_per_framework,
            "collection_interval": self.collection_interval,
            "include_survey_metrics": self.include_survey_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_EC_METRICS_ENABLED", "true").lower()
            == "true",
            prefix=os.getenv("GL_EC_METRICS_PREFIX", "gl_ec_"),
            collect_histograms=os.getenv(
                "GL_EC_METRICS_COLLECT_HISTOGRAMS", "true"
            ).lower()
            == "true",
            histogram_buckets=os.getenv(
                "GL_EC_METRICS_HISTOGRAM_BUCKETS",
                "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
            collect_per_mode=os.getenv(
                "GL_EC_METRICS_COLLECT_PER_MODE", "true"
            ).lower()
            == "true",
            collect_per_framework=os.getenv(
                "GL_EC_METRICS_COLLECT_PER_FRAMEWORK", "true"
            ).lower()
            == "true",
            collection_interval=int(
                os.getenv("GL_EC_METRICS_INTERVAL", "60")
            ),
            include_survey_metrics=os.getenv(
                "GL_EC_METRICS_INCLUDE_SURVEY", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass(frozen=True)
class EmployeeCommutingConfig:
    """
    Master configuration class for Employee Commuting agent (AGENT-MRV-020).

    This frozen dataclass aggregates all 15 configuration sections and provides
    a unified interface for accessing configuration values. It implements the
    singleton pattern with thread-safe access via get_config()/set_config().

    Attributes:
        general: General agent configuration (identity, logging, retries)
        database: PostgreSQL database configuration (pooling, SSL, schema)
        redis: Redis cache configuration (TTL, connections, prefix)
        commute_mode: Commute mode configuration (vehicle, fuel, occupancy)
        telework: Telework energy configuration (kWh breakdowns, seasonal)
        survey: Survey statistics configuration (sample size, confidence)
        working_days: Working days configuration (regional, holidays, PTO)
        spend: Spend-based EEIO configuration (currency, margins)
        compliance: Regulatory framework configuration (7 frameworks)
        ef_source: Emission factor hierarchy configuration (6 sources)
        uncertainty: Monte Carlo uncertainty configuration (iterations, seed)
        cache: Cache layer configuration (TTL, warm-up, eviction)
        api: REST API configuration (pagination, rate limiting)
        provenance: SHA-256 provenance configuration (chain, intermediates)
        metrics: Prometheus metrics configuration (histograms, per-mode)

    Example:
        >>> config = EmployeeCommutingConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-007'
        >>> config.commute_mode.max_distance_km
        500.0
        >>> config.validate_all()
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    commute_mode: CommuteModeConfig = field(default_factory=CommuteModeConfig)
    telework: TeleworkConfig = field(default_factory=TeleworkConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)
    working_days: WorkingDaysConfig = field(default_factory=WorkingDaysConfig)
    spend: SpendConfig = field(default_factory=SpendConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    ef_source: EFSourceConfig = field(default_factory=EFSourceConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def validate_all(self) -> None:
        """
        Validate all configuration sections comprehensively.

        Calls validate() on each of the 15 configuration sections and
        performs cross-section validation to ensure consistency.

        Raises:
            ValueError: If any configuration section or cross-section
                        constraint is invalid
        """
        # Validate each section individually
        self.general.validate()
        self.database.validate()
        self.redis.validate()
        self.commute_mode.validate()
        self.telework.validate()
        self.survey.validate()
        self.working_days.validate()
        self.spend.validate()
        self.compliance.validate()
        self.ef_source.validate()
        self.uncertainty.validate()
        self.cache.validate()
        self.api.validate()
        self.provenance.validate()
        self.metrics.validate()

        # Cross-section validation: table_prefix consistency
        if self.general.table_prefix != "gl_ec_":
            # Warn-level: non-standard prefix
            pass

        # Cross-section validation: metrics prefix should match table prefix
        if self.metrics.prefix != self.general.table_prefix:
            raise ValueError(
                f"metrics.prefix '{self.metrics.prefix}' must match "
                f"general.table_prefix '{self.general.table_prefix}'"
            )

        # Cross-section validation: survey confidence should match uncertainty
        if self.survey.confidence_level != self.uncertainty.confidence_level:
            # This is intentionally allowed but worth noting
            pass

        # Cross-section validation: cache enabled requires Redis or in-memory
        # No hard constraint here, but validate cache TTL vs Redis TTL
        if self.cache.enabled and self.cache.ttl > self.redis.ttl_seconds:
            raise ValueError(
                f"cache.ttl ({self.cache.ttl}) should not exceed "
                f"redis.ttl_seconds ({self.redis.ttl_seconds})"
            )

        # Cross-section validation: EF source year should be reasonable
        if self.ef_source.ef_year < self.spend.base_year:
            raise ValueError(
                f"ef_source.ef_year ({self.ef_source.ef_year}) should not be "
                f"earlier than spend.base_year ({self.spend.base_year})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire configuration to dictionary.

        Returns:
            Dictionary representation of all 15 configuration sections
        """
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "redis": self.redis.to_dict(),
            "commute_mode": self.commute_mode.to_dict(),
            "telework": self.telework.to_dict(),
            "survey": self.survey.to_dict(),
            "working_days": self.working_days.to_dict(),
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
    def from_dict(cls, data: Dict[str, Any]) -> "EmployeeCommutingConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all 15 configuration sections

        Returns:
            EmployeeCommutingConfig instance

        Raises:
            KeyError: If a required configuration section is missing
        """
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            redis=RedisConfig.from_dict(data.get("redis", {})),
            commute_mode=CommuteModeConfig.from_dict(data.get("commute_mode", {})),
            telework=TeleworkConfig.from_dict(data.get("telework", {})),
            survey=SurveyConfig.from_dict(data.get("survey", {})),
            working_days=WorkingDaysConfig.from_dict(data.get("working_days", {})),
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
    def from_env(cls) -> "EmployeeCommutingConfig":
        """
        Load configuration from environment variables.

        All environment variables use the GL_EC_ prefix. Each section
        has its own from_env() that reads section-specific variables.

        Returns:
            EmployeeCommutingConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            commute_mode=CommuteModeConfig.from_env(),
            telework=TeleworkConfig.from_env(),
            survey=SurveyConfig.from_env(),
            working_days=WorkingDaysConfig.from_env(),
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


_config: Optional[EmployeeCommutingConfig] = None
_config_lock = threading.Lock()


def get_config() -> EmployeeCommutingConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton using double-checked locking. The first call
    will load configuration from environment variables and validate all
    sections. Subsequent calls return the cached instance.

    Returns:
        EmployeeCommutingConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-007'

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The configuration is initialized only once via
        double-checked locking with threading.Lock().
    """
    global _config

    # First check without lock (fast path)
    if _config is None:
        with _config_lock:
            # Double-checked locking pattern
            if _config is None:
                _config = EmployeeCommutingConfig.from_env()
                _config.validate_all()

    return _config


def set_config(config: EmployeeCommutingConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.
    The provided configuration is validated before being set.

    Args:
        config: EmployeeCommutingConfig instance to set as singleton

    Raises:
        ValueError: If the provided configuration fails validation

    Example:
        >>> custom_config = EmployeeCommutingConfig.from_dict({...})
        >>> set_config(custom_config)

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The lock ensures atomic replacement of the singleton.
    """
    global _config

    with _config_lock:
        config.validate_all()
        _config = config


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    This function clears the cached configuration singleton, forcing the
    next call to get_config() to reload from environment variables.
    Primarily useful for testing scenarios where environment variables
    may change between test cases.

    Example:
        >>> reset_config()
        >>> config = get_config()  # Reloads from environment

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The lock ensures atomic clearing of the singleton.
    """
    global _config

    with _config_lock:
        _config = None


def _load_from_env() -> EmployeeCommutingConfig:
    """
    Load configuration from environment variables (internal helper).

    This is an internal helper function that loads all configuration sections
    from environment variables. Use get_config() instead for normal usage.

    Returns:
        EmployeeCommutingConfig instance loaded from environment

    Note:
        This function is for internal use. Use get_config() for normal access.
    """
    return EmployeeCommutingConfig.from_env()


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================


def validate_config(config: EmployeeCommutingConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    This function validates all configuration sections and returns a list of
    validation errors. Unlike validate_all() which raises on first error,
    this function collects all errors for comprehensive reporting.

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
    errors: List[str] = []

    # Validate each section independently
    sections: List[Tuple[str, Any]] = [
        ("general", config.general),
        ("database", config.database),
        ("redis", config.redis),
        ("commute_mode", config.commute_mode),
        ("telework", config.telework),
        ("survey", config.survey),
        ("working_days", config.working_days),
        ("spend", config.spend),
        ("compliance", config.compliance),
        ("ef_source", config.ef_source),
        ("uncertainty", config.uncertainty),
        ("cache", config.cache),
        ("api", config.api),
        ("provenance", config.provenance),
        ("metrics", config.metrics),
    ]

    for section_name, section in sections:
        try:
            section.validate()
        except ValueError as e:
            errors.append(f"{section_name}: {str(e)}")

    # Cross-section validations
    try:
        if config.metrics.prefix != config.general.table_prefix:
            errors.append(
                f"cross-section: metrics.prefix '{config.metrics.prefix}' "
                f"must match general.table_prefix '{config.general.table_prefix}'"
            )
    except Exception as e:
        errors.append(f"cross-section: {str(e)}")

    try:
        if config.cache.enabled and config.cache.ttl > config.redis.ttl_seconds:
            errors.append(
                f"cross-section: cache.ttl ({config.cache.ttl}) should not exceed "
                f"redis.ttl_seconds ({config.redis.ttl_seconds})"
            )
    except Exception as e:
        errors.append(f"cross-section: {str(e)}")

    try:
        if config.ef_source.ef_year < config.spend.base_year:
            errors.append(
                f"cross-section: ef_source.ef_year ({config.ef_source.ef_year}) "
                f"should not be earlier than spend.base_year ({config.spend.base_year})"
            )
    except Exception as e:
        errors.append(f"cross-section: {str(e)}")

    return errors


def print_config(config: EmployeeCommutingConfig) -> None:
    """
    Print configuration in human-readable format.

    This function prints all 15 configuration sections in a formatted,
    human-readable manner. Sensitive fields (passwords) are redacted.
    Useful for debugging and verification during agent startup.

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
        ===== Employee Commuting Configuration (GL-MRV-S3-007) =====
        [GENERAL]
        enabled: True
        ...
    """
    print("===== Employee Commuting Configuration (GL-MRV-S3-007) =====")

    print("\n[GENERAL]")
    for key, value in config.general.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[DATABASE]")
    for key, value in config.database.to_dict().items():
        if key == "password":
            print(f"  {key}: [REDACTED]")
        else:
            print(f"  {key}: {value}")

    print("\n[REDIS]")
    for key, value in config.redis.to_dict().items():
        if key == "password":
            print(f"  {key}: [REDACTED]")
        else:
            print(f"  {key}: {value}")

    print("\n[COMMUTE_MODE]")
    for key, value in config.commute_mode.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[TELEWORK]")
    for key, value in config.telework.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[SURVEY]")
    for key, value in config.survey.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[WORKING_DAYS]")
    for key, value in config.working_days.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[SPEND]")
    for key, value in config.spend.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[COMPLIANCE]")
    for key, value in config.compliance.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[EF_SOURCE]")
    for key, value in config.ef_source.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[UNCERTAINTY]")
    for key, value in config.uncertainty.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[CACHE]")
    for key, value in config.cache.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[API]")
    for key, value in config.api.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[PROVENANCE]")
    for key, value in config.provenance.to_dict().items():
        print(f"  {key}: {value}")

    print("\n[METRICS]")
    for key, value in config.metrics.to_dict().items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 64)


def get_config_summary(config: EmployeeCommutingConfig) -> Dict[str, Any]:
    """
    Get a summary of key configuration values.

    Returns a dictionary of the most important configuration values
    for quick inspection and logging during agent startup.

    Args:
        config: Configuration instance to summarize

    Returns:
        Dictionary with key configuration highlights

    Example:
        >>> config = get_config()
        >>> summary = get_config_summary(config)
        >>> summary['agent_id']
        'GL-MRV-S3-007'
    """
    return {
        "agent_id": config.general.agent_id,
        "version": config.general.version,
        "enabled": config.general.enabled,
        "table_prefix": config.general.table_prefix,
        "log_level": config.general.log_level,
        "db_host": config.database.host,
        "db_schema": config.database.schema,
        "redis_host": config.redis.host,
        "redis_prefix": config.redis.prefix,
        "default_vehicle_type": config.commute_mode.default_vehicle_type,
        "default_fuel_type": config.commute_mode.default_fuel_type,
        "include_wtt": config.commute_mode.include_wtt,
        "telework_enabled": config.telework.enabled,
        "telework_daily_kwh": str(config.telework.default_daily_kwh),
        "survey_method": config.survey.default_survey_method,
        "survey_confidence": str(config.survey.confidence_level),
        "working_days_region": config.working_days.default_region,
        "working_days_default": config.working_days.default_working_days,
        "eeio_source": config.spend.eeio_source,
        "compliance_frameworks": config.compliance.get_frameworks(),
        "ef_hierarchy": config.ef_source.get_hierarchy(),
        "uncertainty_method": config.uncertainty.method,
        "uncertainty_iterations": config.uncertainty.iterations,
        "cache_enabled": config.cache.enabled,
        "api_prefix": config.api.prefix,
        "provenance_enabled": config.provenance.enabled,
        "provenance_algorithm": config.provenance.hash_algorithm,
        "metrics_enabled": config.metrics.enabled,
        "metrics_prefix": config.metrics.prefix,
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration section classes (15 sections)
    "GeneralConfig",
    "DatabaseConfig",
    "RedisConfig",
    "CommuteModeConfig",
    "TeleworkConfig",
    "SurveyConfig",
    "WorkingDaysConfig",
    "SpendConfig",
    "ComplianceConfig",
    "EFSourceConfig",
    "UncertaintyConfig",
    "CacheConfig",
    "APIConfig",
    "ProvenanceConfig",
    "MetricsConfig",
    # Master configuration class
    "EmployeeCommutingConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
    "get_config_summary",
]
