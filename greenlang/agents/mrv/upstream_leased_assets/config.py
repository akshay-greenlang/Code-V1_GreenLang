# -*- coding: utf-8 -*-
"""
Upstream Leased Assets Configuration - AGENT-MRV-021

Thread-safe singleton configuration for GL-MRV-S3-008.
All environment variables prefixed with GL_ULA_.

This module provides comprehensive configuration management for the Upstream
Leased Assets agent (GHG Protocol Scope 3 Category 8), supporting:
- Building emissions (office, warehouse, retail, data center, manufacturing)
- Vehicle fleet emissions (company cars, trucks, delivery vehicles)
- Equipment emissions (generators, compressors, forklifts, HVAC)
- IT asset emissions (servers, storage, networking, data center PUE)
- Lease classification (IFRS 16 / ASC 842 operating vs finance)
- Floor-area, headcount, FTE, and custom allocation methods
- Partial-year proration for mid-year lease commencement/termination
- Spend-based EEIO fallback calculation method
- Well-to-tank (WTT) upstream fuel cycle emissions
- Emission factor hierarchy (DEFRA, EPA eGRID, IEA, Energy Star, EEIO)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi,
  SB 253, GRI)
- Monte Carlo uncertainty quantification
- Provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-008'
    >>> config.building.default_building_type
    'OFFICE'
    >>> config.allocation.default_method
    'FLOOR_AREA'
    >>> config.uncertainty.confidence_level
    Decimal('0.95')

Thread Safety:
    All configuration operations are protected by threading.Lock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_ULA_ prefix. See individual config sections for specific variables.
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
    General configuration for Upstream Leased Assets agent.

    Controls agent identity, logging, retry behavior, and timeout settings
    for the GL-MRV-S3-008 (Scope 3 Category 8) agent.

    Attributes:
        enabled: Master switch for the agent (GL_ULA_ENABLED)
        debug: Enable debug mode with verbose logging (GL_ULA_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_ULA_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_ULA_AGENT_ID)
        component: Component identifier for AGENT-MRV-021 (GL_ULA_COMPONENT)
        version: Agent version following SemVer (GL_ULA_VERSION)
        table_prefix: Prefix for all database tables (GL_ULA_TABLE_PREFIX)
        max_retries: Maximum retry attempts for transient failures (GL_ULA_MAX_RETRIES)
        timeout: Default operation timeout in seconds (GL_ULA_TIMEOUT)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-008",
        ...     component="AGENT-MRV-021",
        ...     version="1.0.0",
        ...     table_prefix="gl_ula_",
        ...     max_retries=3,
        ...     timeout=300
        ... )
        >>> general.agent_id
        'GL-MRV-S3-008'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-008"
    component: str = "AGENT-MRV-021"
    version: str = "1.0.0"
    table_prefix: str = "gl_ula_"
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

        if not self.component:
            raise ValueError("component cannot be empty")

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
            "component": self.component,
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
            enabled=os.getenv("GL_ULA_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_ULA_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_ULA_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_ULA_AGENT_ID", "GL-MRV-S3-008"),
            component=os.getenv("GL_ULA_COMPONENT", "AGENT-MRV-021"),
            version=os.getenv("GL_ULA_VERSION", "1.0.0"),
            table_prefix=os.getenv("GL_ULA_TABLE_PREFIX", "gl_ula_"),
            max_retries=int(os.getenv("GL_ULA_MAX_RETRIES", "3")),
            timeout=int(os.getenv("GL_ULA_TIMEOUT", "300")),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Upstream Leased Assets agent.

    Manages PostgreSQL connection parameters including connection pooling,
    SSL mode, and query timeouts for the upstream leased assets service schema.

    Attributes:
        host: Database hostname (GL_ULA_DB_HOST)
        port: Database port (GL_ULA_DB_PORT)
        name: Database name (GL_ULA_DB_NAME)
        user: Database user (GL_ULA_DB_USER)
        password: Database password (GL_ULA_DB_PASSWORD)
        pool_min: Minimum pool connections (GL_ULA_DB_POOL_MIN)
        pool_max: Maximum pool connections (GL_ULA_DB_POOL_MAX)
        ssl: Enable SSL connection (GL_ULA_DB_SSL)
        timeout: Query timeout in seconds (GL_ULA_DB_TIMEOUT)
        schema: Database schema name (GL_ULA_DB_SCHEMA)

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
        ...     schema="upstream_leased_assets_service"
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
    schema: str = "upstream_leased_assets_service"

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
            host=os.getenv("GL_ULA_DB_HOST", "localhost"),
            port=int(os.getenv("GL_ULA_DB_PORT", "5432")),
            name=os.getenv("GL_ULA_DB_NAME", "greenlang"),
            user=os.getenv("GL_ULA_DB_USER", "greenlang"),
            password=os.getenv("GL_ULA_DB_PASSWORD", ""),
            pool_min=int(os.getenv("GL_ULA_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_ULA_DB_POOL_MAX", "10")),
            ssl=os.getenv("GL_ULA_DB_SSL", "false").lower() == "true",
            timeout=int(os.getenv("GL_ULA_DB_TIMEOUT", "30")),
            schema=os.getenv("GL_ULA_DB_SCHEMA", "upstream_leased_assets_service"),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Upstream Leased Assets agent.

    Controls caching layer parameters for emission factor lookups,
    building energy benchmarks, and intermediate calculation caching.

    Attributes:
        host: Redis hostname (GL_ULA_REDIS_HOST)
        port: Redis port (GL_ULA_REDIS_PORT)
        db: Redis database number (GL_ULA_REDIS_DB)
        password: Redis password (GL_ULA_REDIS_PASSWORD)
        ssl: Enable SSL connection (GL_ULA_REDIS_SSL)
        ttl_seconds: Default TTL in seconds (GL_ULA_REDIS_TTL)
        max_connections: Max connection pool size (GL_ULA_REDIS_MAX_CONNECTIONS)
        prefix: Key namespace prefix (GL_ULA_REDIS_PREFIX)

    Example:
        >>> redis = RedisConfig(
        ...     host="localhost",
        ...     port=6379,
        ...     db=0,
        ...     password="",
        ...     ssl=False,
        ...     ttl_seconds=3600,
        ...     max_connections=20,
        ...     prefix="gl_ula:"
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
    prefix: str = "gl_ula:"

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
            host=os.getenv("GL_ULA_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_ULA_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_ULA_REDIS_DB", "0")),
            password=os.getenv("GL_ULA_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_ULA_REDIS_SSL", "false").lower() == "true",
            ttl_seconds=int(os.getenv("GL_ULA_REDIS_TTL", "3600")),
            max_connections=int(os.getenv("GL_ULA_REDIS_MAX_CONNECTIONS", "20")),
            prefix=os.getenv("GL_ULA_REDIS_PREFIX", "gl_ula:"),
        )


# =============================================================================
# SECTION 4: BUILDING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class BuildingConfig:
    """
    Building emissions configuration for Upstream Leased Assets agent.

    Governs building-level energy consumption modelling including building
    types, climate zones, energy use intensity (EUI) benchmarks, and
    refrigerant leakage estimation for leased commercial buildings.

    Supported building types:
    - OFFICE: Commercial office space (cubicle, open plan, executive)
    - WAREHOUSE: Storage and distribution facilities
    - RETAIL: Retail stores and shopping centres
    - DATA_CENTER: Server rooms and co-location facilities
    - MANUFACTURING: Light manufacturing and assembly
    - MIXED_USE: Multi-purpose buildings
    - LABORATORY: Research and laboratory facilities
    - HEALTHCARE: Clinics and medical offices (not hospitals)

    Supported climate zones:
    - TROPICAL: Hot-humid year-round (ASHRAE 0A-1A)
    - ARID: Hot-dry / warm-dry (ASHRAE 2B-3B)
    - TEMPERATE: Moderate heating and cooling (ASHRAE 3A-4A)
    - CONTINENTAL: Cold winters, warm summers (ASHRAE 5A-6A)
    - POLAR: Very cold / subarctic (ASHRAE 7-8)

    Attributes:
        default_building_type: Default building type (GL_ULA_DEFAULT_BUILDING_TYPE)
        default_climate_zone: Default climate zone (GL_ULA_DEFAULT_CLIMATE_ZONE)
        allocation_method: Default building allocation method (GL_ULA_BUILDING_ALLOCATION)
        include_refrigerants: Include HVAC refrigerant leakage (GL_ULA_BUILDING_INCLUDE_REFRIGERANTS)
        eui_source: Energy use intensity benchmark source (GL_ULA_BUILDING_EUI_SOURCE)
        max_floor_area_sqm: Maximum floor area in square metres (GL_ULA_BUILDING_MAX_FLOOR_AREA)
        default_eui_kwh_per_sqm: Default annual EUI in kWh/sqm (GL_ULA_BUILDING_DEFAULT_EUI)
        include_common_areas: Include common area allocation (GL_ULA_BUILDING_INCLUDE_COMMON)
        refrigerant_leak_rate: Annual refrigerant leak rate fraction (GL_ULA_BUILDING_REFRIGERANT_LEAK_RATE)
        default_refrigerant_type: Default refrigerant type (GL_ULA_BUILDING_DEFAULT_REFRIGERANT)
        include_wtt: Include WTT upstream energy emissions (GL_ULA_BUILDING_INCLUDE_WTT)
        building_age_adjustment: Apply age-based efficiency degradation (GL_ULA_BUILDING_AGE_ADJUSTMENT)

    Example:
        >>> building = BuildingConfig(
        ...     default_building_type="OFFICE",
        ...     default_climate_zone="TEMPERATE",
        ...     allocation_method="FLOOR_AREA",
        ...     include_refrigerants=False,
        ...     eui_source="ENERGY_STAR"
        ... )
        >>> building.default_building_type
        'OFFICE'
    """

    default_building_type: str = "OFFICE"
    default_climate_zone: str = "TEMPERATE"
    allocation_method: str = "FLOOR_AREA"
    include_refrigerants: bool = False
    eui_source: str = "ENERGY_STAR"
    max_floor_area_sqm: float = 500000.0
    default_eui_kwh_per_sqm: Decimal = Decimal("200.0")
    include_common_areas: bool = True
    refrigerant_leak_rate: Decimal = Decimal("0.05")
    default_refrigerant_type: str = "R410A"
    include_wtt: bool = True
    building_age_adjustment: bool = True

    def validate(self) -> None:
        """
        Validate building configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_building_types = {
            "OFFICE",
            "WAREHOUSE",
            "RETAIL",
            "DATA_CENTER",
            "MANUFACTURING",
            "MIXED_USE",
            "LABORATORY",
            "HEALTHCARE",
        }
        if self.default_building_type not in valid_building_types:
            raise ValueError(
                f"Invalid default_building_type '{self.default_building_type}'. "
                f"Must be one of {valid_building_types}"
            )

        valid_climate_zones = {
            "TROPICAL",
            "ARID",
            "TEMPERATE",
            "CONTINENTAL",
            "POLAR",
        }
        if self.default_climate_zone not in valid_climate_zones:
            raise ValueError(
                f"Invalid default_climate_zone '{self.default_climate_zone}'. "
                f"Must be one of {valid_climate_zones}"
            )

        valid_allocation_methods = {
            "FLOOR_AREA",
            "HEADCOUNT",
            "FTE",
            "REVENUE",
            "CUSTOM",
        }
        if self.allocation_method not in valid_allocation_methods:
            raise ValueError(
                f"Invalid allocation_method '{self.allocation_method}'. "
                f"Must be one of {valid_allocation_methods}"
            )

        valid_eui_sources = {
            "ENERGY_STAR",
            "CIBSE",
            "ASHRAE",
            "CUSTOM",
        }
        if self.eui_source not in valid_eui_sources:
            raise ValueError(
                f"Invalid eui_source '{self.eui_source}'. "
                f"Must be one of {valid_eui_sources}"
            )

        if self.max_floor_area_sqm <= 0 or self.max_floor_area_sqm > 10000000:
            raise ValueError("max_floor_area_sqm must be between 0 (exclusive) and 10000000")

        if self.default_eui_kwh_per_sqm < Decimal("0"):
            raise ValueError("default_eui_kwh_per_sqm must be >= 0")

        if self.default_eui_kwh_per_sqm > Decimal("2000"):
            raise ValueError("default_eui_kwh_per_sqm must be <= 2000")

        if self.refrigerant_leak_rate < Decimal("0.0") or self.refrigerant_leak_rate > Decimal("0.5"):
            raise ValueError("refrigerant_leak_rate must be between 0.0 and 0.5")

        valid_refrigerants = {
            "R410A", "R134A", "R407C", "R404A", "R32",
            "R290", "R744", "R717", "R1234YF", "R1234ZE",
        }
        if self.default_refrigerant_type not in valid_refrigerants:
            raise ValueError(
                f"Invalid default_refrigerant_type '{self.default_refrigerant_type}'. "
                f"Must be one of {valid_refrigerants}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_building_type": self.default_building_type,
            "default_climate_zone": self.default_climate_zone,
            "allocation_method": self.allocation_method,
            "include_refrigerants": self.include_refrigerants,
            "eui_source": self.eui_source,
            "max_floor_area_sqm": self.max_floor_area_sqm,
            "default_eui_kwh_per_sqm": str(self.default_eui_kwh_per_sqm),
            "include_common_areas": self.include_common_areas,
            "refrigerant_leak_rate": str(self.refrigerant_leak_rate),
            "default_refrigerant_type": self.default_refrigerant_type,
            "include_wtt": self.include_wtt,
            "building_age_adjustment": self.building_age_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildingConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "default_eui_kwh_per_sqm",
            "refrigerant_leak_rate",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "BuildingConfig":
        """Load from environment variables."""
        return cls(
            default_building_type=os.getenv(
                "GL_ULA_DEFAULT_BUILDING_TYPE", "OFFICE"
            ),
            default_climate_zone=os.getenv(
                "GL_ULA_DEFAULT_CLIMATE_ZONE", "TEMPERATE"
            ),
            allocation_method=os.getenv("GL_ULA_BUILDING_ALLOCATION", "FLOOR_AREA"),
            include_refrigerants=os.getenv(
                "GL_ULA_BUILDING_INCLUDE_REFRIGERANTS", "false"
            ).lower()
            == "true",
            eui_source=os.getenv("GL_ULA_BUILDING_EUI_SOURCE", "ENERGY_STAR"),
            max_floor_area_sqm=float(
                os.getenv("GL_ULA_BUILDING_MAX_FLOOR_AREA", "500000.0")
            ),
            default_eui_kwh_per_sqm=Decimal(
                os.getenv("GL_ULA_BUILDING_DEFAULT_EUI", "200.0")
            ),
            include_common_areas=os.getenv(
                "GL_ULA_BUILDING_INCLUDE_COMMON", "true"
            ).lower()
            == "true",
            refrigerant_leak_rate=Decimal(
                os.getenv("GL_ULA_BUILDING_REFRIGERANT_LEAK_RATE", "0.05")
            ),
            default_refrigerant_type=os.getenv(
                "GL_ULA_BUILDING_DEFAULT_REFRIGERANT", "R410A"
            ),
            include_wtt=os.getenv(
                "GL_ULA_BUILDING_INCLUDE_WTT", "true"
            ).lower()
            == "true",
            building_age_adjustment=os.getenv(
                "GL_ULA_BUILDING_AGE_ADJUSTMENT", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 5: VEHICLE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class VehicleConfig:
    """
    Vehicle fleet emissions configuration for Upstream Leased Assets agent.

    Governs vehicle-level emission calculations for leased fleet assets
    including cars, vans, trucks, and specialty vehicles. Supports fuel-type
    differentiation, well-to-tank upstream emissions, and age-based
    efficiency degradation modelling.

    Supported vehicle types:
    - PASSENGER_CAR: Sedan, hatchback, estate
    - SUV: Sport utility vehicle, crossover
    - LIGHT_VAN: Light commercial vehicle (< 3.5t)
    - HEAVY_VAN: Heavy commercial vehicle (3.5-7.5t)
    - RIGID_TRUCK: Rigid-body truck (7.5-26t)
    - ARTICULATED_TRUCK: Articulated / semi-trailer (> 26t)
    - MOTORCYCLE: Powered two-wheeler
    - ELECTRIC_VEHICLE: Battery electric (BEV)

    Attributes:
        default_fuel_type: Default fuel type for vehicles (GL_ULA_VEHICLE_FUEL_TYPE)
        default_annual_km: Default annual distance in km (GL_ULA_VEHICLE_ANNUAL_KM)
        include_wtt: Include well-to-tank upstream emissions (GL_ULA_VEHICLE_INCLUDE_WTT)
        age_degradation: Apply age-based efficiency degradation (GL_ULA_VEHICLE_AGE_DEGRADATION)
        max_vehicle_age: Maximum vehicle age in years (GL_ULA_VEHICLE_MAX_AGE)
        degradation_rate_per_year: Annual efficiency degradation fraction (GL_ULA_VEHICLE_DEGRADATION_RATE)
        default_vehicle_type: Default vehicle category (GL_ULA_VEHICLE_DEFAULT_TYPE)
        max_annual_km: Maximum annual distance limit in km (GL_ULA_VEHICLE_MAX_ANNUAL_KM)
        include_maintenance: Include maintenance-related emissions (GL_ULA_VEHICLE_INCLUDE_MAINTENANCE)

    Example:
        >>> vehicle = VehicleConfig(
        ...     default_fuel_type="DIESEL",
        ...     default_annual_km=15000,
        ...     include_wtt=True,
        ...     age_degradation=True
        ... )
        >>> vehicle.default_fuel_type
        'DIESEL'
    """

    default_fuel_type: str = "DIESEL"
    default_annual_km: int = 15000
    include_wtt: bool = True
    age_degradation: bool = True
    max_vehicle_age: int = 25
    degradation_rate_per_year: Decimal = Decimal("0.01")
    default_vehicle_type: str = "PASSENGER_CAR"
    max_annual_km: int = 500000
    include_maintenance: bool = False

    def validate(self) -> None:
        """
        Validate vehicle configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_fuel_types = {
            "DIESEL",
            "PETROL",
            "GASOLINE",
            "CNG",
            "LPG",
            "BIODIESEL",
            "ELECTRICITY",
            "HYDROGEN",
            "HYBRID_PETROL",
            "HYBRID_DIESEL",
            "PLUG_IN_HYBRID",
        }
        if self.default_fuel_type not in valid_fuel_types:
            raise ValueError(
                f"Invalid default_fuel_type '{self.default_fuel_type}'. "
                f"Must be one of {valid_fuel_types}"
            )

        valid_vehicle_types = {
            "PASSENGER_CAR",
            "SUV",
            "LIGHT_VAN",
            "HEAVY_VAN",
            "RIGID_TRUCK",
            "ARTICULATED_TRUCK",
            "MOTORCYCLE",
            "ELECTRIC_VEHICLE",
        }
        if self.default_vehicle_type not in valid_vehicle_types:
            raise ValueError(
                f"Invalid default_vehicle_type '{self.default_vehicle_type}'. "
                f"Must be one of {valid_vehicle_types}"
            )

        if self.default_annual_km < 0 or self.default_annual_km > 500000:
            raise ValueError("default_annual_km must be between 0 and 500000")

        if self.max_vehicle_age < 1 or self.max_vehicle_age > 50:
            raise ValueError("max_vehicle_age must be between 1 and 50")

        if self.degradation_rate_per_year < Decimal("0.0") or self.degradation_rate_per_year > Decimal("0.1"):
            raise ValueError("degradation_rate_per_year must be between 0.0 and 0.1")

        if self.max_annual_km < 1000 or self.max_annual_km > 1000000:
            raise ValueError("max_annual_km must be between 1000 and 1000000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_fuel_type": self.default_fuel_type,
            "default_annual_km": self.default_annual_km,
            "include_wtt": self.include_wtt,
            "age_degradation": self.age_degradation,
            "max_vehicle_age": self.max_vehicle_age,
            "degradation_rate_per_year": str(self.degradation_rate_per_year),
            "default_vehicle_type": self.default_vehicle_type,
            "max_annual_km": self.max_annual_km,
            "include_maintenance": self.include_maintenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VehicleConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "degradation_rate_per_year" in data_copy:
            data_copy["degradation_rate_per_year"] = Decimal(
                str(data_copy["degradation_rate_per_year"])
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "VehicleConfig":
        """Load from environment variables."""
        return cls(
            default_fuel_type=os.getenv("GL_ULA_VEHICLE_FUEL_TYPE", "DIESEL"),
            default_annual_km=int(os.getenv("GL_ULA_VEHICLE_ANNUAL_KM", "15000")),
            include_wtt=os.getenv("GL_ULA_VEHICLE_INCLUDE_WTT", "true").lower() == "true",
            age_degradation=os.getenv("GL_ULA_VEHICLE_AGE_DEGRADATION", "true").lower() == "true",
            max_vehicle_age=int(os.getenv("GL_ULA_VEHICLE_MAX_AGE", "25")),
            degradation_rate_per_year=Decimal(
                os.getenv("GL_ULA_VEHICLE_DEGRADATION_RATE", "0.01")
            ),
            default_vehicle_type=os.getenv("GL_ULA_VEHICLE_DEFAULT_TYPE", "PASSENGER_CAR"),
            max_annual_km=int(os.getenv("GL_ULA_VEHICLE_MAX_ANNUAL_KM", "500000")),
            include_maintenance=os.getenv(
                "GL_ULA_VEHICLE_INCLUDE_MAINTENANCE", "false"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 6: EQUIPMENT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EquipmentConfig:
    """
    Equipment emissions configuration for Upstream Leased Assets agent.

    Governs emission calculations for leased stationary and mobile equipment
    including generators, compressors, forklifts, and HVAC systems.

    Supported equipment types:
    - GENERATOR: Diesel/gas backup and standby generators
    - COMPRESSOR: Air compressors, gas compressors
    - FORKLIFT: Electric and diesel forklifts
    - HVAC: Heating/ventilation/air conditioning systems
    - PUMP: Water and fluid pumps
    - CHILLER: Industrial and commercial chillers
    - BOILER: Steam and hot water boilers
    - CRANE: Mobile and tower cranes
    - CUSTOM: User-defined equipment type

    Attributes:
        default_load_factor: Default equipment load factor 0-1 (GL_ULA_EQUIPMENT_LOAD_FACTOR)
        default_operating_hours: Default annual operating hours (GL_ULA_EQUIPMENT_OPERATING_HOURS)
        fuel_consumption_method: Fuel consumption estimation method (GL_ULA_EQUIPMENT_FUEL_METHOD)
        max_operating_hours: Maximum annual operating hours (GL_ULA_EQUIPMENT_MAX_HOURS)
        include_wtt: Include WTT upstream fuel emissions (GL_ULA_EQUIPMENT_INCLUDE_WTT)
        default_fuel_type: Default equipment fuel type (GL_ULA_EQUIPMENT_FUEL_TYPE)
        include_refrigerants: Include refrigerant leakage for HVAC/chillers (GL_ULA_EQUIPMENT_INCLUDE_REFRIGERANTS)
        efficiency_degradation: Apply age-based efficiency loss (GL_ULA_EQUIPMENT_EFFICIENCY_DEGRADATION)
        degradation_rate_per_year: Annual efficiency loss fraction (GL_ULA_EQUIPMENT_DEGRADATION_RATE)

    Example:
        >>> equipment = EquipmentConfig(
        ...     default_load_factor=Decimal("0.6"),
        ...     default_operating_hours=4000,
        ...     fuel_consumption_method="BENCHMARK"
        ... )
        >>> equipment.default_load_factor
        Decimal('0.6')
    """

    default_load_factor: Decimal = Decimal("0.6")
    default_operating_hours: int = 4000
    fuel_consumption_method: str = "BENCHMARK"
    max_operating_hours: int = 8760
    include_wtt: bool = True
    default_fuel_type: str = "DIESEL"
    include_refrigerants: bool = False
    efficiency_degradation: bool = True
    degradation_rate_per_year: Decimal = Decimal("0.02")

    def validate(self) -> None:
        """
        Validate equipment configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_load_factor < Decimal("0.0") or self.default_load_factor > Decimal("1.0"):
            raise ValueError("default_load_factor must be between 0.0 and 1.0")

        if self.default_operating_hours < 0 or self.default_operating_hours > 8760:
            raise ValueError("default_operating_hours must be between 0 and 8760")

        valid_methods = {"BENCHMARK", "NAMEPLATE", "METERED", "MANUFACTURER", "CUSTOM"}
        if self.fuel_consumption_method not in valid_methods:
            raise ValueError(
                f"Invalid fuel_consumption_method '{self.fuel_consumption_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.max_operating_hours < 1 or self.max_operating_hours > 8760:
            raise ValueError("max_operating_hours must be between 1 and 8760")

        valid_fuel_types = {
            "DIESEL", "PETROL", "GASOLINE", "NATURAL_GAS", "LPG",
            "ELECTRICITY", "HYDROGEN", "PROPANE", "BIODIESEL",
        }
        if self.default_fuel_type not in valid_fuel_types:
            raise ValueError(
                f"Invalid default_fuel_type '{self.default_fuel_type}'. "
                f"Must be one of {valid_fuel_types}"
            )

        if self.degradation_rate_per_year < Decimal("0.0") or self.degradation_rate_per_year > Decimal("0.2"):
            raise ValueError("degradation_rate_per_year must be between 0.0 and 0.2")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_load_factor": str(self.default_load_factor),
            "default_operating_hours": self.default_operating_hours,
            "fuel_consumption_method": self.fuel_consumption_method,
            "max_operating_hours": self.max_operating_hours,
            "include_wtt": self.include_wtt,
            "default_fuel_type": self.default_fuel_type,
            "include_refrigerants": self.include_refrigerants,
            "efficiency_degradation": self.efficiency_degradation,
            "degradation_rate_per_year": str(self.degradation_rate_per_year),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquipmentConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_load_factor", "degradation_rate_per_year"]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "EquipmentConfig":
        """Load from environment variables."""
        return cls(
            default_load_factor=Decimal(
                os.getenv("GL_ULA_EQUIPMENT_LOAD_FACTOR", "0.6")
            ),
            default_operating_hours=int(
                os.getenv("GL_ULA_EQUIPMENT_OPERATING_HOURS", "4000")
            ),
            fuel_consumption_method=os.getenv(
                "GL_ULA_EQUIPMENT_FUEL_METHOD", "BENCHMARK"
            ),
            max_operating_hours=int(
                os.getenv("GL_ULA_EQUIPMENT_MAX_HOURS", "8760")
            ),
            include_wtt=os.getenv(
                "GL_ULA_EQUIPMENT_INCLUDE_WTT", "true"
            ).lower()
            == "true",
            default_fuel_type=os.getenv("GL_ULA_EQUIPMENT_FUEL_TYPE", "DIESEL"),
            include_refrigerants=os.getenv(
                "GL_ULA_EQUIPMENT_INCLUDE_REFRIGERANTS", "false"
            ).lower()
            == "true",
            efficiency_degradation=os.getenv(
                "GL_ULA_EQUIPMENT_EFFICIENCY_DEGRADATION", "true"
            ).lower()
            == "true",
            degradation_rate_per_year=Decimal(
                os.getenv("GL_ULA_EQUIPMENT_DEGRADATION_RATE", "0.02")
            ),
        )


# =============================================================================
# SECTION 7: IT ASSET CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ITConfig:
    """
    IT asset emissions configuration for Upstream Leased Assets agent.

    Governs emission calculations for leased IT equipment including servers,
    storage arrays, networking gear, and co-located data center space.
    Uses PUE (Power Usage Effectiveness) to model data center overhead.

    Attributes:
        default_pue: Default Power Usage Effectiveness ratio (GL_ULA_IT_DEFAULT_PUE)
        default_utilization: Default server utilization fraction 0-1 (GL_ULA_IT_DEFAULT_UTILIZATION)
        include_embodied: Include embodied emissions for IT hardware (GL_ULA_IT_INCLUDE_EMBODIED)
        include_cooling: Include cooling energy for servers (GL_ULA_IT_INCLUDE_COOLING)
        default_server_watts: Default server power draw in watts (GL_ULA_IT_DEFAULT_SERVER_WATTS)
        default_storage_watts: Default storage array power in watts (GL_ULA_IT_DEFAULT_STORAGE_WATTS)
        default_network_watts: Default network device power in watts (GL_ULA_IT_DEFAULT_NETWORK_WATTS)
        max_pue: Maximum valid PUE value (GL_ULA_IT_MAX_PUE)
        include_wtt: Include WTT upstream electricity emissions (GL_ULA_IT_INCLUDE_WTT)
        embodied_amortization_years: Years over which to amortize embodied emissions (GL_ULA_IT_EMBODIED_YEARS)

    Example:
        >>> it = ITConfig(
        ...     default_pue=Decimal("1.58"),
        ...     default_utilization=Decimal("0.3"),
        ...     include_embodied=False
        ... )
        >>> it.default_pue
        Decimal('1.58')
    """

    default_pue: Decimal = Decimal("1.58")
    default_utilization: Decimal = Decimal("0.3")
    include_embodied: bool = False
    include_cooling: bool = True
    default_server_watts: int = 500
    default_storage_watts: int = 200
    default_network_watts: int = 50
    max_pue: Decimal = Decimal("3.0")
    include_wtt: bool = True
    embodied_amortization_years: int = 4

    def validate(self) -> None:
        """
        Validate IT asset configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_pue < Decimal("1.0") or self.default_pue > Decimal("3.0"):
            raise ValueError("default_pue must be between 1.0 and 3.0")

        if self.default_utilization < Decimal("0.0") or self.default_utilization > Decimal("1.0"):
            raise ValueError("default_utilization must be between 0.0 and 1.0")

        if self.default_server_watts < 0 or self.default_server_watts > 10000:
            raise ValueError("default_server_watts must be between 0 and 10000")

        if self.default_storage_watts < 0 or self.default_storage_watts > 10000:
            raise ValueError("default_storage_watts must be between 0 and 10000")

        if self.default_network_watts < 0 or self.default_network_watts > 5000:
            raise ValueError("default_network_watts must be between 0 and 5000")

        if self.max_pue < Decimal("1.0") or self.max_pue > Decimal("5.0"):
            raise ValueError("max_pue must be between 1.0 and 5.0")

        if self.embodied_amortization_years < 1 or self.embodied_amortization_years > 10:
            raise ValueError("embodied_amortization_years must be between 1 and 10")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_pue": str(self.default_pue),
            "default_utilization": str(self.default_utilization),
            "include_embodied": self.include_embodied,
            "include_cooling": self.include_cooling,
            "default_server_watts": self.default_server_watts,
            "default_storage_watts": self.default_storage_watts,
            "default_network_watts": self.default_network_watts,
            "max_pue": str(self.max_pue),
            "include_wtt": self.include_wtt,
            "embodied_amortization_years": self.embodied_amortization_years,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ITConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_pue", "default_utilization", "max_pue"]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ITConfig":
        """Load from environment variables."""
        return cls(
            default_pue=Decimal(os.getenv("GL_ULA_IT_DEFAULT_PUE", "1.58")),
            default_utilization=Decimal(
                os.getenv("GL_ULA_IT_DEFAULT_UTILIZATION", "0.3")
            ),
            include_embodied=os.getenv(
                "GL_ULA_IT_INCLUDE_EMBODIED", "false"
            ).lower()
            == "true",
            include_cooling=os.getenv(
                "GL_ULA_IT_INCLUDE_COOLING", "true"
            ).lower()
            == "true",
            default_server_watts=int(
                os.getenv("GL_ULA_IT_DEFAULT_SERVER_WATTS", "500")
            ),
            default_storage_watts=int(
                os.getenv("GL_ULA_IT_DEFAULT_STORAGE_WATTS", "200")
            ),
            default_network_watts=int(
                os.getenv("GL_ULA_IT_DEFAULT_NETWORK_WATTS", "50")
            ),
            max_pue=Decimal(os.getenv("GL_ULA_IT_MAX_PUE", "3.0")),
            include_wtt=os.getenv(
                "GL_ULA_IT_INCLUDE_WTT", "true"
            ).lower()
            == "true",
            embodied_amortization_years=int(
                os.getenv("GL_ULA_IT_EMBODIED_YEARS", "4")
            ),
        )


# =============================================================================
# SECTION 8: ALLOCATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AllocationConfig:
    """
    Allocation configuration for Upstream Leased Assets agent.

    Controls how emissions from shared leased assets are allocated to the
    reporting company based on their share of use. Supports partial-year
    proration for leases that begin or end mid-year.

    Allocation methods:
    - FLOOR_AREA: Allocate by lessee's share of total floor area (sq m / sq ft)
    - HEADCOUNT: Allocate by number of employees occupying the space
    - FTE: Allocate by full-time equivalent employees
    - REVENUE: Allocate by revenue proportion
    - CUSTOM: User-defined allocation basis

    Attributes:
        default_method: Default allocation method (GL_ULA_ALLOCATION_METHOD)
        allow_custom: Allow custom allocation factors (GL_ULA_ALLOCATION_ALLOW_CUSTOM)
        partial_year_proration: Enable partial-year proration (GL_ULA_ALLOCATION_PARTIAL_YEAR)
        min_allocation_factor: Minimum valid allocation factor (GL_ULA_ALLOCATION_MIN_FACTOR)
        max_allocation_factor: Maximum valid allocation factor (GL_ULA_ALLOCATION_MAX_FACTOR)
        common_area_inclusion: Include common area allocation (GL_ULA_ALLOCATION_COMMON_AREAS)
        common_area_method: Method for common area distribution (GL_ULA_ALLOCATION_COMMON_METHOD)

    Example:
        >>> allocation = AllocationConfig(
        ...     default_method="FLOOR_AREA",
        ...     allow_custom=True,
        ...     partial_year_proration=True
        ... )
        >>> allocation.default_method
        'FLOOR_AREA'
    """

    default_method: str = "FLOOR_AREA"
    allow_custom: bool = True
    partial_year_proration: bool = True
    min_allocation_factor: Decimal = Decimal("0.0")
    max_allocation_factor: Decimal = Decimal("1.0")
    common_area_inclusion: bool = True
    common_area_method: str = "PROPORTIONAL"

    def validate(self) -> None:
        """
        Validate allocation configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"FLOOR_AREA", "HEADCOUNT", "FTE", "REVENUE", "CUSTOM"}
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.min_allocation_factor < Decimal("0.0"):
            raise ValueError("min_allocation_factor must be >= 0.0")

        if self.max_allocation_factor > Decimal("1.0"):
            raise ValueError("max_allocation_factor must be <= 1.0")

        if self.min_allocation_factor > self.max_allocation_factor:
            raise ValueError("min_allocation_factor must be <= max_allocation_factor")

        valid_common_methods = {"PROPORTIONAL", "EQUAL", "NONE"}
        if self.common_area_method not in valid_common_methods:
            raise ValueError(
                f"Invalid common_area_method '{self.common_area_method}'. "
                f"Must be one of {valid_common_methods}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "allow_custom": self.allow_custom,
            "partial_year_proration": self.partial_year_proration,
            "min_allocation_factor": str(self.min_allocation_factor),
            "max_allocation_factor": str(self.max_allocation_factor),
            "common_area_inclusion": self.common_area_inclusion,
            "common_area_method": self.common_area_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["min_allocation_factor", "max_allocation_factor"]:
            if key in data_copy:
                data_copy[key] = Decimal(str(data_copy[key]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "AllocationConfig":
        """Load from environment variables."""
        return cls(
            default_method=os.getenv("GL_ULA_ALLOCATION_METHOD", "FLOOR_AREA"),
            allow_custom=os.getenv(
                "GL_ULA_ALLOCATION_ALLOW_CUSTOM", "true"
            ).lower()
            == "true",
            partial_year_proration=os.getenv(
                "GL_ULA_ALLOCATION_PARTIAL_YEAR", "true"
            ).lower()
            == "true",
            min_allocation_factor=Decimal(
                os.getenv("GL_ULA_ALLOCATION_MIN_FACTOR", "0.0")
            ),
            max_allocation_factor=Decimal(
                os.getenv("GL_ULA_ALLOCATION_MAX_FACTOR", "1.0")
            ),
            common_area_inclusion=os.getenv(
                "GL_ULA_ALLOCATION_COMMON_AREAS", "true"
            ).lower()
            == "true",
            common_area_method=os.getenv(
                "GL_ULA_ALLOCATION_COMMON_METHOD", "PROPORTIONAL"
            ),
        )


# =============================================================================
# SECTION 9: LEASE CLASSIFICATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class LeaseClassificationConfig:
    """
    Lease classification configuration for Upstream Leased Assets agent.

    Controls lease classification under IFRS 16 / ASC 842 accounting
    standards. Finance leases (where the lessee has substantially all
    risks and rewards of ownership) should be reported in Scope 1/2
    rather than Scope 3 Category 8.

    Accounting standards:
    - IFRS_16: International Financial Reporting Standard 16 (Leases)
    - ASC_842: US GAAP ASC 842 (Leases)
    - BOTH: Apply both standards and report any classification differences

    Attributes:
        accounting_standard: Lease accounting standard (GL_ULA_LEASE_ACCOUNTING_STANDARD)
        auto_classify: Automatically classify operating vs finance leases (GL_ULA_LEASE_AUTO_CLASSIFY)
        finance_lease_to_scope12: Redirect finance lease emissions to Scope 1/2 (GL_ULA_LEASE_FINANCE_TO_SCOPE12)
        max_lease_term_months: Maximum lease term in months (GL_ULA_LEASE_MAX_TERM_MONTHS)
        short_term_exempt: Exempt short-term leases (< 12 months) (GL_ULA_LEASE_SHORT_TERM_EXEMPT)
        low_value_exempt: Exempt low-value asset leases (GL_ULA_LEASE_LOW_VALUE_EXEMPT)
        low_value_threshold: Threshold for low-value exemption in USD (GL_ULA_LEASE_LOW_VALUE_THRESHOLD)

    Example:
        >>> lease = LeaseClassificationConfig(
        ...     accounting_standard="IFRS_16",
        ...     auto_classify=True,
        ...     finance_lease_to_scope12=True
        ... )
        >>> lease.accounting_standard
        'IFRS_16'
    """

    accounting_standard: str = "IFRS_16"
    auto_classify: bool = True
    finance_lease_to_scope12: bool = True
    max_lease_term_months: int = 600
    short_term_exempt: bool = True
    low_value_exempt: bool = True
    low_value_threshold: Decimal = Decimal("5000.00")

    def validate(self) -> None:
        """
        Validate lease classification configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_standards = {"IFRS_16", "ASC_842", "BOTH"}
        if self.accounting_standard not in valid_standards:
            raise ValueError(
                f"Invalid accounting_standard '{self.accounting_standard}'. "
                f"Must be one of {valid_standards}"
            )

        if self.max_lease_term_months < 1 or self.max_lease_term_months > 1200:
            raise ValueError("max_lease_term_months must be between 1 and 1200 (100 years)")

        if self.low_value_threshold < Decimal("0.0") or self.low_value_threshold > Decimal("100000.0"):
            raise ValueError("low_value_threshold must be between 0 and 100000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accounting_standard": self.accounting_standard,
            "auto_classify": self.auto_classify,
            "finance_lease_to_scope12": self.finance_lease_to_scope12,
            "max_lease_term_months": self.max_lease_term_months,
            "short_term_exempt": self.short_term_exempt,
            "low_value_exempt": self.low_value_exempt,
            "low_value_threshold": str(self.low_value_threshold),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeaseClassificationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "low_value_threshold" in data_copy:
            data_copy["low_value_threshold"] = Decimal(str(data_copy["low_value_threshold"]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "LeaseClassificationConfig":
        """Load from environment variables."""
        return cls(
            accounting_standard=os.getenv("GL_ULA_LEASE_ACCOUNTING_STANDARD", "IFRS_16"),
            auto_classify=os.getenv("GL_ULA_LEASE_AUTO_CLASSIFY", "true").lower() == "true",
            finance_lease_to_scope12=os.getenv(
                "GL_ULA_LEASE_FINANCE_TO_SCOPE12", "true"
            ).lower()
            == "true",
            max_lease_term_months=int(os.getenv("GL_ULA_LEASE_MAX_TERM_MONTHS", "600")),
            short_term_exempt=os.getenv("GL_ULA_LEASE_SHORT_TERM_EXEMPT", "true").lower() == "true",
            low_value_exempt=os.getenv("GL_ULA_LEASE_LOW_VALUE_EXEMPT", "true").lower() == "true",
            low_value_threshold=Decimal(
                os.getenv("GL_ULA_LEASE_LOW_VALUE_THRESHOLD", "5000.00")
            ),
        )


# =============================================================================
# SECTION 10: EMISSION FACTOR SOURCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EFSourceConfig:
    """
    Emission Factor Source configuration for Upstream Leased Assets agent.

    Defines the emission factor lookup hierarchy and fallback chain for
    building energy, vehicle fuel, equipment fuel, and IT electricity
    emission factors.

    EF hierarchy (default order):
    1. SUPPLIER - Supplier/lessor-provided specific EFs (highest quality)
    2. DEFRA - UK DEFRA BEIS GHG conversion factors
    3. EPA - US EPA emission factors
    4. IEA - International Energy Agency country-level factors
    5. ENERGY_STAR - Energy Star benchmarks for buildings
    6. EEIO - Spend-based EEIO fallback (lowest quality)

    Attributes:
        primary_source: Primary EF source (GL_ULA_EF_PRIMARY_SOURCE)
        grid_source: Grid electricity EF source (GL_ULA_EF_GRID_SOURCE)
        egrid_enabled: Enable EPA eGRID subregional factors (GL_ULA_EF_EGRID_ENABLED)
        wtt_source: WTT emission factor source (GL_ULA_EF_WTT_SOURCE)
        allow_custom: Allow custom EFs (GL_ULA_EF_ALLOW_CUSTOM)
        custom_ef_path: Path to custom EF file (GL_ULA_EF_CUSTOM_PATH)
        cache_ef_lookups: Cache EF lookups (GL_ULA_EF_CACHE_LOOKUPS)
        ef_year: Target emission factor year (GL_ULA_EF_YEAR)
        fallback_enabled: Enable automatic fallback (GL_ULA_EF_FALLBACK_ENABLED)
        hierarchy: Comma-separated EF source priority list (GL_ULA_EF_HIERARCHY)

    Example:
        >>> ef = EFSourceConfig(
        ...     primary_source="DEFRA_2024",
        ...     grid_source="IEA_2024",
        ...     egrid_enabled=True,
        ...     wtt_source="DEFRA_2024"
        ... )
        >>> ef.primary_source
        'DEFRA_2024'
    """

    primary_source: str = "DEFRA_2024"
    grid_source: str = "IEA_2024"
    egrid_enabled: bool = True
    wtt_source: str = "DEFRA_2024"
    allow_custom: bool = True
    custom_ef_path: Optional[str] = None
    cache_ef_lookups: bool = True
    ef_year: int = 2024
    fallback_enabled: bool = True
    hierarchy: str = "SUPPLIER,DEFRA,EPA,IEA,ENERGY_STAR,EEIO"

    def validate(self) -> None:
        """
        Validate EF source configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_primary_sources = {
            "DEFRA_2024", "DEFRA_2023", "EPA_2024", "EPA_2023",
            "IEA_2024", "IEA_2023", "CUSTOM",
        }
        if self.primary_source not in valid_primary_sources:
            raise ValueError(
                f"Invalid primary_source '{self.primary_source}'. "
                f"Must be one of {valid_primary_sources}"
            )

        valid_grid_sources = {
            "IEA_2024", "IEA_2023", "EPA_EGRID_2024", "EPA_EGRID_2023",
            "DEFRA_2024", "DEFRA_2023", "CUSTOM",
        }
        if self.grid_source not in valid_grid_sources:
            raise ValueError(
                f"Invalid grid_source '{self.grid_source}'. "
                f"Must be one of {valid_grid_sources}"
            )

        valid_wtt_sources = {"DEFRA_2024", "DEFRA_2023", "EPA_2024", "IEA_2024", "CUSTOM"}
        if self.wtt_source not in valid_wtt_sources:
            raise ValueError(
                f"Invalid wtt_source '{self.wtt_source}'. "
                f"Must be one of {valid_wtt_sources}"
            )

        if self.primary_source == "CUSTOM" and not self.allow_custom:
            raise ValueError("primary_source is CUSTOM but allow_custom is False")

        if self.primary_source == "CUSTOM" and not self.custom_ef_path:
            raise ValueError("custom_ef_path must be set when primary_source is CUSTOM")

        if self.ef_year < 2000 or self.ef_year > 2030:
            raise ValueError("ef_year must be between 2000 and 2030")

        hierarchy_list = self.get_hierarchy()
        if not hierarchy_list:
            raise ValueError("At least one EF source must be in the hierarchy")

        valid_hierarchy_sources = {
            "SUPPLIER", "DEFRA", "EPA", "IEA", "ENERGY_STAR", "EEIO", "CUSTOM",
        }
        for source in hierarchy_list:
            if source not in valid_hierarchy_sources:
                raise ValueError(
                    f"Invalid EF source '{source}' in hierarchy. "
                    f"Must be one of {valid_hierarchy_sources}"
                )

    def get_hierarchy(self) -> List[str]:
        """Parse hierarchy string into ordered list."""
        return [s.strip() for s in self.hierarchy.split(",") if s.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_source": self.primary_source,
            "grid_source": self.grid_source,
            "egrid_enabled": self.egrid_enabled,
            "wtt_source": self.wtt_source,
            "allow_custom": self.allow_custom,
            "custom_ef_path": self.custom_ef_path,
            "cache_ef_lookups": self.cache_ef_lookups,
            "ef_year": self.ef_year,
            "fallback_enabled": self.fallback_enabled,
            "hierarchy": self.hierarchy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EFSourceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "EFSourceConfig":
        """Load from environment variables."""
        return cls(
            primary_source=os.getenv("GL_ULA_EF_PRIMARY_SOURCE", "DEFRA_2024"),
            grid_source=os.getenv("GL_ULA_EF_GRID_SOURCE", "IEA_2024"),
            egrid_enabled=os.getenv("GL_ULA_EF_EGRID_ENABLED", "true").lower() == "true",
            wtt_source=os.getenv("GL_ULA_EF_WTT_SOURCE", "DEFRA_2024"),
            allow_custom=os.getenv("GL_ULA_EF_ALLOW_CUSTOM", "true").lower() == "true",
            custom_ef_path=os.getenv("GL_ULA_EF_CUSTOM_PATH"),
            cache_ef_lookups=os.getenv("GL_ULA_EF_CACHE_LOOKUPS", "true").lower() == "true",
            ef_year=int(os.getenv("GL_ULA_EF_YEAR", "2024")),
            fallback_enabled=os.getenv("GL_ULA_EF_FALLBACK_ENABLED", "true").lower() == "true",
            hierarchy=os.getenv("GL_ULA_EF_HIERARCHY", "SUPPLIER,DEFRA,EPA,IEA,ENERGY_STAR,EEIO"),
        )


# =============================================================================
# SECTION 11: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Upstream Leased Assets agent.

    Controls regulatory framework alignment, strict mode enforcement,
    and disclosure requirements for Scope 3 Category 8 reporting.

    Supported frameworks:
    - GHG_PROTOCOL: GHG Protocol Corporate Value Chain (Scope 3) Standard
    - ISO_14064: ISO 14064-1:2018 Organizational GHG quantification
    - CSRD_ESRS: EU Corporate Sustainability Reporting Directive (Climate)
    - CDP: Carbon Disclosure Project Climate Change questionnaire
    - SBTI: Science Based Targets initiative
    - SB_253: California SB 253 Climate Corporate Data Accountability Act
    - GRI: Global Reporting Initiative 305 (Emissions)

    Attributes:
        frameworks: Comma-separated enabled frameworks (GL_ULA_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict validation for all frameworks (GL_ULA_COMPLIANCE_STRICT_MODE)
        dqi_threshold: Minimum DQI score threshold (GL_ULA_COMPLIANCE_DQI_THRESHOLD)
        double_counting_check: Enable double-counting prevention (GL_ULA_COMPLIANCE_DOUBLE_COUNTING)
        boundary_enforcement: Enforce Scope 3 Cat 8 boundary rules (GL_ULA_COMPLIANCE_BOUNDARY)
        data_quality_required: Require DQI scoring for all inputs (GL_ULA_COMPLIANCE_DQI_REQUIRED)
        lease_classification_required: Require explicit lease classification (GL_ULA_COMPLIANCE_LEASE_CLASS)
        asset_type_disclosure: Require asset type breakdown (GL_ULA_COMPLIANCE_ASSET_DISCLOSURE)

    Example:
        >>> compliance = ComplianceConfig(
        ...     frameworks="GHG_PROTOCOL,ISO_14064,CSRD_ESRS,CDP,SBTI,SB_253,GRI",
        ...     strict_mode=False,
        ...     dqi_threshold=Decimal("2.5")
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL', 'ISO_14064', 'CSRD_ESRS', 'CDP', 'SBTI', 'SB_253', 'GRI']
    """

    frameworks: str = "GHG_PROTOCOL,ISO_14064,CSRD_ESRS,CDP,SBTI,SB_253,GRI"
    strict_mode: bool = False
    dqi_threshold: Decimal = Decimal("2.5")
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    data_quality_required: bool = True
    lease_classification_required: bool = True
    asset_type_disclosure: bool = True

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL",
            "ISO_14064",
            "CSRD_ESRS",
            "CDP",
            "SBTI",
            "SB_253",
            "GRI",
        }

        framework_list = self.get_frameworks()
        if not framework_list:
            raise ValueError("At least one compliance framework must be enabled")

        for framework in framework_list:
            if framework not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{framework}'. Must be one of {valid_frameworks}"
                )

        if self.dqi_threshold < Decimal("1.0") or self.dqi_threshold > Decimal("5.0"):
            raise ValueError("dqi_threshold must be between 1.0 and 5.0")

    def get_frameworks(self) -> List[str]:
        """Parse compliance frameworks string into list."""
        return [f.strip() for f in self.frameworks.split(",") if f.strip()]

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
            "frameworks": self.frameworks,
            "strict_mode": self.strict_mode,
            "dqi_threshold": str(self.dqi_threshold),
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "data_quality_required": self.data_quality_required,
            "lease_classification_required": self.lease_classification_required,
            "asset_type_disclosure": self.asset_type_disclosure,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "dqi_threshold" in data_copy:
            data_copy["dqi_threshold"] = Decimal(str(data_copy["dqi_threshold"]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            frameworks=os.getenv(
                "GL_ULA_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL,ISO_14064,CSRD_ESRS,CDP,SBTI,SB_253,GRI",
            ),
            strict_mode=os.getenv("GL_ULA_COMPLIANCE_STRICT_MODE", "false").lower() == "true",
            dqi_threshold=Decimal(os.getenv("GL_ULA_COMPLIANCE_DQI_THRESHOLD", "2.5")),
            double_counting_check=os.getenv(
                "GL_ULA_COMPLIANCE_DOUBLE_COUNTING", "true"
            ).lower()
            == "true",
            boundary_enforcement=os.getenv(
                "GL_ULA_COMPLIANCE_BOUNDARY", "true"
            ).lower()
            == "true",
            data_quality_required=os.getenv(
                "GL_ULA_COMPLIANCE_DQI_REQUIRED", "true"
            ).lower()
            == "true",
            lease_classification_required=os.getenv(
                "GL_ULA_COMPLIANCE_LEASE_CLASS", "true"
            ).lower()
            == "true",
            asset_type_disclosure=os.getenv(
                "GL_ULA_COMPLIANCE_ASSET_DISCLOSURE", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 12: SPEND CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendConfig:
    """
    Spend-based calculation configuration for Upstream Leased Assets agent.

    Controls the spend-based EEIO fallback calculation method when
    asset-level energy data is not available. Supports currency
    conversion, CPI deflation, and margin removal.

    Attributes:
        default_currency: Default currency for spend data (GL_ULA_SPEND_CURRENCY)
        eeio_source: Default EEIO model source (GL_ULA_SPEND_EEIO_SOURCE)
        cpi_base_year: Base year for CPI deflation (GL_ULA_SPEND_CPI_BASE_YEAR)
        margin_removal_rate: Margin removal rate (GL_ULA_SPEND_MARGIN_RATE)
        enable_cpi_deflation: Enable CPI deflation (GL_ULA_SPEND_CPI_DEFLATION)
        enable_currency_conversion: Enable multi-currency support (GL_ULA_SPEND_CURRENCY_CONVERSION)
        real_estate_sector_code: Default NAICS sector for real estate leases (GL_ULA_SPEND_RE_SECTOR)
        purchaser_price_adjustment: Adjust from purchaser to producer prices (GL_ULA_SPEND_PPA)

    Example:
        >>> spend = SpendConfig(
        ...     default_currency="USD",
        ...     eeio_source="EPA_USEEIO_V2",
        ...     cpi_base_year=2021
        ... )
        >>> spend.eeio_source
        'EPA_USEEIO_V2'
    """

    default_currency: str = "USD"
    eeio_source: str = "EPA_USEEIO_V2"
    cpi_base_year: int = 2021
    margin_removal_rate: Decimal = Decimal("0.15")
    enable_cpi_deflation: bool = True
    enable_currency_conversion: bool = True
    real_estate_sector_code: str = "531000"
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

        valid_eeio_sources = {"EPA_USEEIO_V2", "EXIOBASE", "DEFRA"}
        if self.eeio_source not in valid_eeio_sources:
            raise ValueError(
                f"Invalid eeio_source '{self.eeio_source}'. "
                f"Must be one of {valid_eeio_sources}"
            )

        if self.cpi_base_year < 2000 or self.cpi_base_year > 2030:
            raise ValueError("cpi_base_year must be between 2000 and 2030")

        if self.margin_removal_rate < Decimal("0.0") or self.margin_removal_rate > Decimal("0.5"):
            raise ValueError("margin_removal_rate must be between 0.0 and 0.5")

        if not self.real_estate_sector_code:
            raise ValueError("real_estate_sector_code cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_currency": self.default_currency,
            "eeio_source": self.eeio_source,
            "cpi_base_year": self.cpi_base_year,
            "margin_removal_rate": str(self.margin_removal_rate),
            "enable_cpi_deflation": self.enable_cpi_deflation,
            "enable_currency_conversion": self.enable_currency_conversion,
            "real_estate_sector_code": self.real_estate_sector_code,
            "purchaser_price_adjustment": self.purchaser_price_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "margin_removal_rate" in data_copy:
            data_copy["margin_removal_rate"] = Decimal(str(data_copy["margin_removal_rate"]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendConfig":
        """Load from environment variables."""
        return cls(
            default_currency=os.getenv("GL_ULA_SPEND_CURRENCY", "USD"),
            eeio_source=os.getenv("GL_ULA_SPEND_EEIO_SOURCE", "EPA_USEEIO_V2"),
            cpi_base_year=int(os.getenv("GL_ULA_SPEND_CPI_BASE_YEAR", "2021")),
            margin_removal_rate=Decimal(
                os.getenv("GL_ULA_SPEND_MARGIN_RATE", "0.15")
            ),
            enable_cpi_deflation=os.getenv(
                "GL_ULA_SPEND_CPI_DEFLATION", "true"
            ).lower()
            == "true",
            enable_currency_conversion=os.getenv(
                "GL_ULA_SPEND_CURRENCY_CONVERSION", "true"
            ).lower()
            == "true",
            real_estate_sector_code=os.getenv("GL_ULA_SPEND_RE_SECTOR", "531000"),
            purchaser_price_adjustment=os.getenv(
                "GL_ULA_SPEND_PPA", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 13: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty configuration for Upstream Leased Assets agent.

    Controls Monte Carlo simulation parameters and uncertainty
    quantification for leased asset emission estimates.

    Attributes:
        method: Uncertainty quantification method (GL_ULA_UNCERTAINTY_METHOD)
        iterations: Monte Carlo simulation iterations (GL_ULA_UNCERTAINTY_ITERATIONS)
        confidence: Confidence level for intervals (GL_ULA_UNCERTAINTY_CONFIDENCE)
        seed: Random seed for reproducibility (GL_ULA_UNCERTAINTY_SEED)
        include_ef_uncertainty: Include EF uncertainty (GL_ULA_UNCERTAINTY_INCLUDE_EF)
        include_activity_uncertainty: Include activity data uncertainty (GL_ULA_UNCERTAINTY_INCLUDE_ACTIVITY)
        include_allocation_uncertainty: Include allocation factor uncertainty (GL_ULA_UNCERTAINTY_INCLUDE_ALLOCATION)
        distribution_type: Default probability distribution (GL_ULA_UNCERTAINTY_DISTRIBUTION)

    Example:
        >>> uncertainty = UncertaintyConfig(
        ...     method="MONTE_CARLO",
        ...     iterations=10000,
        ...     confidence=Decimal("0.95"),
        ...     seed=42
        ... )
        >>> uncertainty.method
        'MONTE_CARLO'
    """

    method: str = "MONTE_CARLO"
    iterations: int = 10000
    confidence: Decimal = Decimal("0.95")
    seed: int = 42
    include_ef_uncertainty: bool = True
    include_activity_uncertainty: bool = True
    include_allocation_uncertainty: bool = True
    distribution_type: str = "LOGNORMAL"

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"MONTE_CARLO", "IPCC_DEFAULT", "BOOTSTRAP", "ANALYTICAL", "NONE"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.iterations < 100 or self.iterations > 1000000:
            raise ValueError("iterations must be between 100 and 1000000")

        if self.confidence < Decimal("0.5") or self.confidence > Decimal("0.999"):
            raise ValueError("confidence must be between 0.5 and 0.999")

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
            "confidence": str(self.confidence),
            "seed": self.seed,
            "include_ef_uncertainty": self.include_ef_uncertainty,
            "include_activity_uncertainty": self.include_activity_uncertainty,
            "include_allocation_uncertainty": self.include_allocation_uncertainty,
            "distribution_type": self.distribution_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "confidence" in data_copy:
            data_copy["confidence"] = Decimal(str(data_copy["confidence"]))
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        return cls(
            method=os.getenv("GL_ULA_UNCERTAINTY_METHOD", "MONTE_CARLO"),
            iterations=int(os.getenv("GL_ULA_UNCERTAINTY_ITERATIONS", "10000")),
            confidence=Decimal(os.getenv("GL_ULA_UNCERTAINTY_CONFIDENCE", "0.95")),
            seed=int(os.getenv("GL_ULA_UNCERTAINTY_SEED", "42")),
            include_ef_uncertainty=os.getenv(
                "GL_ULA_UNCERTAINTY_INCLUDE_EF", "true"
            ).lower()
            == "true",
            include_activity_uncertainty=os.getenv(
                "GL_ULA_UNCERTAINTY_INCLUDE_ACTIVITY", "true"
            ).lower()
            == "true",
            include_allocation_uncertainty=os.getenv(
                "GL_ULA_UNCERTAINTY_INCLUDE_ALLOCATION", "true"
            ).lower()
            == "true",
            distribution_type=os.getenv("GL_ULA_UNCERTAINTY_DISTRIBUTION", "LOGNORMAL"),
        )


# =============================================================================
# SECTION 14: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Upstream Leased Assets agent.

    Controls SHA-256 hashing for complete audit trails, chain validation
    for sequential calculation steps, and intermediate result storage.

    Attributes:
        algorithm: Hash algorithm for provenance (GL_ULA_PROVENANCE_ALGORITHM)
        chain_enabled: Enable chain hash validation (GL_ULA_PROVENANCE_CHAIN_ENABLED)
        retention_days: Days to retain provenance records (GL_ULA_PROVENANCE_RETENTION_DAYS)
        store_intermediates: Store intermediate calculation hashes (GL_ULA_PROVENANCE_STORE_INTERMEDIATES)
        include_config_hash: Include configuration hash in provenance (GL_ULA_PROVENANCE_INCLUDE_CONFIG)
        include_ef_hash: Include emission factor hash in provenance (GL_ULA_PROVENANCE_INCLUDE_EF)
        include_allocation_hash: Include allocation hash in provenance (GL_ULA_PROVENANCE_INCLUDE_ALLOCATION)
        enabled: Enable provenance tracking (GL_ULA_PROVENANCE_ENABLED)

    Example:
        >>> provenance = ProvenanceConfig(
        ...     algorithm="sha256",
        ...     chain_enabled=True,
        ...     retention_days=2555
        ... )
        >>> provenance.algorithm
        'sha256'
    """

    algorithm: str = "sha256"
    chain_enabled: bool = True
    retention_days: int = 2555
    store_intermediates: bool = True
    include_config_hash: bool = True
    include_ef_hash: bool = True
    include_allocation_hash: bool = True
    enabled: bool = True

    def validate(self) -> None:
        """
        Validate provenance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_algorithms = {"sha256", "sha512", "blake2b"}
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid algorithm '{self.algorithm}'. "
                f"Must be one of {valid_algorithms}"
            )

        if self.retention_days < 1 or self.retention_days > 3650:
            raise ValueError("retention_days must be between 1 and 3650 (10 years)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "chain_enabled": self.chain_enabled,
            "retention_days": self.retention_days,
            "store_intermediates": self.store_intermediates,
            "include_config_hash": self.include_config_hash,
            "include_ef_hash": self.include_ef_hash,
            "include_allocation_hash": self.include_allocation_hash,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            algorithm=os.getenv("GL_ULA_PROVENANCE_ALGORITHM", "sha256"),
            chain_enabled=os.getenv("GL_ULA_PROVENANCE_CHAIN_ENABLED", "true").lower() == "true",
            retention_days=int(os.getenv("GL_ULA_PROVENANCE_RETENTION_DAYS", "2555")),
            store_intermediates=os.getenv(
                "GL_ULA_PROVENANCE_STORE_INTERMEDIATES", "true"
            ).lower()
            == "true",
            include_config_hash=os.getenv(
                "GL_ULA_PROVENANCE_INCLUDE_CONFIG", "true"
            ).lower()
            == "true",
            include_ef_hash=os.getenv(
                "GL_ULA_PROVENANCE_INCLUDE_EF", "true"
            ).lower()
            == "true",
            include_allocation_hash=os.getenv(
                "GL_ULA_PROVENANCE_INCLUDE_ALLOCATION", "true"
            ).lower()
            == "true",
            enabled=os.getenv("GL_ULA_PROVENANCE_ENABLED", "true").lower() == "true",
        )


# =============================================================================
# SECTION 15: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Upstream Leased Assets agent.

    Controls Prometheus-compatible metrics collection including histograms,
    counters, and gauges for observability and performance monitoring.

    Attributes:
        enabled: Enable metrics collection (GL_ULA_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_ULA_METRICS_PREFIX)
        collect_histograms: Collect histogram metrics (GL_ULA_METRICS_COLLECT_HISTOGRAMS)
        histogram_buckets: Histogram bucket boundaries (GL_ULA_METRICS_HISTOGRAM_BUCKETS)
        collect_per_asset_type: Collect per-asset-type metrics (GL_ULA_METRICS_COLLECT_PER_ASSET_TYPE)
        collect_per_framework: Collect per-framework compliance metrics (GL_ULA_METRICS_COLLECT_PER_FRAMEWORK)
        collection_interval: Metrics collection interval in seconds (GL_ULA_METRICS_INTERVAL)

    Example:
        >>> metrics = MetricsConfig(
        ...     enabled=True,
        ...     prefix="gl_ula_",
        ...     collect_histograms=True,
        ...     histogram_buckets="0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_buckets()
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    """

    enabled: bool = True
    prefix: str = "gl_ula_"
    collect_histograms: bool = True
    histogram_buckets: str = "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
    collect_per_asset_type: bool = True
    collect_per_framework: bool = True
    collection_interval: int = 60

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
            "collect_per_asset_type": self.collect_per_asset_type,
            "collect_per_framework": self.collect_per_framework,
            "collection_interval": self.collection_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_ULA_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_ULA_METRICS_PREFIX", "gl_ula_"),
            collect_histograms=os.getenv(
                "GL_ULA_METRICS_COLLECT_HISTOGRAMS", "true"
            ).lower()
            == "true",
            histogram_buckets=os.getenv(
                "GL_ULA_METRICS_HISTOGRAM_BUCKETS",
                "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
            collect_per_asset_type=os.getenv(
                "GL_ULA_METRICS_COLLECT_PER_ASSET_TYPE", "true"
            ).lower()
            == "true",
            collect_per_framework=os.getenv(
                "GL_ULA_METRICS_COLLECT_PER_FRAMEWORK", "true"
            ).lower()
            == "true",
            collection_interval=int(os.getenv("GL_ULA_METRICS_INTERVAL", "60")),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass(frozen=True)
class UpstreamLeasedAssetsConfig:
    """
    Master configuration class for Upstream Leased Assets agent (AGENT-MRV-021).

    This frozen dataclass aggregates all 15 configuration sections and provides
    a unified interface for accessing configuration values. It implements the
    singleton pattern with thread-safe access via get_config()/set_config().

    Attributes:
        general: General agent configuration (identity, logging, retries)
        database: PostgreSQL database configuration (pooling, SSL, schema)
        redis: Redis cache configuration (TTL, connections, prefix)
        building: Building emissions configuration (EUI, climate, allocation)
        vehicle: Vehicle fleet emissions configuration (fuel, distance, WTT)
        equipment: Equipment emissions configuration (load factor, operating hours)
        it_asset: IT asset emissions configuration (PUE, utilization, embodied)
        allocation: Allocation method configuration (floor area, headcount)
        lease_classification: Lease classification configuration (IFRS 16, ASC 842)
        ef_source: Emission factor hierarchy configuration (DEFRA, EPA, IEA)
        compliance: Regulatory framework configuration (7 frameworks)
        spend: Spend-based EEIO configuration (currency, margins)
        uncertainty: Monte Carlo uncertainty configuration (iterations, seed)
        provenance: SHA-256 provenance configuration (chain, intermediates)
        metrics: Prometheus metrics configuration (histograms, per-asset-type)

    Example:
        >>> config = UpstreamLeasedAssetsConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-008'
        >>> config.building.default_building_type
        'OFFICE'
        >>> config.validate_all()
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    equipment: EquipmentConfig = field(default_factory=EquipmentConfig)
    it_asset: ITConfig = field(default_factory=ITConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    lease_classification: LeaseClassificationConfig = field(
        default_factory=LeaseClassificationConfig
    )
    ef_source: EFSourceConfig = field(default_factory=EFSourceConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    spend: SpendConfig = field(default_factory=SpendConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
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
        self.building.validate()
        self.vehicle.validate()
        self.equipment.validate()
        self.it_asset.validate()
        self.allocation.validate()
        self.lease_classification.validate()
        self.ef_source.validate()
        self.compliance.validate()
        self.spend.validate()
        self.uncertainty.validate()
        self.provenance.validate()
        self.metrics.validate()

        # Cross-section validation: table_prefix consistency
        if self.general.table_prefix != "gl_ula_":
            # Warn-level: non-standard prefix
            pass

        # Cross-section validation: metrics prefix should match table prefix
        if self.metrics.prefix != self.general.table_prefix:
            raise ValueError(
                f"metrics.prefix '{self.metrics.prefix}' must match "
                f"general.table_prefix '{self.general.table_prefix}'"
            )

        # Cross-section validation: EF source year should be reasonable
        if self.ef_source.ef_year < self.spend.cpi_base_year:
            raise ValueError(
                f"ef_source.ef_year ({self.ef_source.ef_year}) should not be "
                f"earlier than spend.cpi_base_year ({self.spend.cpi_base_year})"
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
            "building": self.building.to_dict(),
            "vehicle": self.vehicle.to_dict(),
            "equipment": self.equipment.to_dict(),
            "it_asset": self.it_asset.to_dict(),
            "allocation": self.allocation.to_dict(),
            "lease_classification": self.lease_classification.to_dict(),
            "ef_source": self.ef_source.to_dict(),
            "compliance": self.compliance.to_dict(),
            "spend": self.spend.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpstreamLeasedAssetsConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all 15 configuration sections

        Returns:
            UpstreamLeasedAssetsConfig instance

        Raises:
            KeyError: If a required configuration section is missing
        """
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            redis=RedisConfig.from_dict(data.get("redis", {})),
            building=BuildingConfig.from_dict(data.get("building", {})),
            vehicle=VehicleConfig.from_dict(data.get("vehicle", {})),
            equipment=EquipmentConfig.from_dict(data.get("equipment", {})),
            it_asset=ITConfig.from_dict(data.get("it_asset", {})),
            allocation=AllocationConfig.from_dict(data.get("allocation", {})),
            lease_classification=LeaseClassificationConfig.from_dict(
                data.get("lease_classification", {})
            ),
            ef_source=EFSourceConfig.from_dict(data.get("ef_source", {})),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            spend=SpendConfig.from_dict(data.get("spend", {})),
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
        )

    @classmethod
    def from_env(cls) -> "UpstreamLeasedAssetsConfig":
        """
        Load configuration from environment variables.

        All environment variables use the GL_ULA_ prefix. Each section
        has its own from_env() that reads section-specific variables.

        Returns:
            UpstreamLeasedAssetsConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            building=BuildingConfig.from_env(),
            vehicle=VehicleConfig.from_env(),
            equipment=EquipmentConfig.from_env(),
            it_asset=ITConfig.from_env(),
            allocation=AllocationConfig.from_env(),
            lease_classification=LeaseClassificationConfig.from_env(),
            ef_source=EFSourceConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            spend=SpendConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            metrics=MetricsConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config: Optional[UpstreamLeasedAssetsConfig] = None
_config_lock = threading.Lock()


def get_config() -> UpstreamLeasedAssetsConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton using double-checked locking. The first call
    will load configuration from environment variables and validate all
    sections. Subsequent calls return the cached instance.

    Returns:
        UpstreamLeasedAssetsConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-008'

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
                _config = UpstreamLeasedAssetsConfig.from_env()
                _config.validate_all()

    return _config


def set_config(config: UpstreamLeasedAssetsConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.
    The provided configuration is validated before being set.

    Args:
        config: UpstreamLeasedAssetsConfig instance to set as singleton

    Raises:
        ValueError: If the provided configuration fails validation

    Example:
        >>> custom_config = UpstreamLeasedAssetsConfig.from_dict({...})
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


def _load_from_env() -> UpstreamLeasedAssetsConfig:
    """
    Load configuration from environment variables (internal helper).

    This is an internal helper function that loads all configuration sections
    from environment variables. Use get_config() instead for normal usage.

    Returns:
        UpstreamLeasedAssetsConfig instance loaded from environment

    Note:
        This function is for internal use. Use get_config() for normal access.
    """
    return UpstreamLeasedAssetsConfig.from_env()


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================


def validate_config(config: UpstreamLeasedAssetsConfig) -> List[str]:
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
        ("building", config.building),
        ("vehicle", config.vehicle),
        ("equipment", config.equipment),
        ("it_asset", config.it_asset),
        ("allocation", config.allocation),
        ("lease_classification", config.lease_classification),
        ("ef_source", config.ef_source),
        ("compliance", config.compliance),
        ("spend", config.spend),
        ("uncertainty", config.uncertainty),
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
        if config.ef_source.ef_year < config.spend.cpi_base_year:
            errors.append(
                f"cross-section: ef_source.ef_year ({config.ef_source.ef_year}) "
                f"should not be earlier than spend.cpi_base_year ({config.spend.cpi_base_year})"
            )
    except Exception as e:
        errors.append(f"cross-section: {str(e)}")

    return errors


def print_config(config: UpstreamLeasedAssetsConfig) -> None:
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
        ===== Upstream Leased Assets Configuration (GL-MRV-S3-008) =====
        [GENERAL]
        enabled: True
        ...
    """
    print("===== Upstream Leased Assets Configuration (GL-MRV-S3-008) =====")

    section_names = [
        ("GENERAL", config.general),
        ("DATABASE", config.database),
        ("REDIS", config.redis),
        ("BUILDING", config.building),
        ("VEHICLE", config.vehicle),
        ("EQUIPMENT", config.equipment),
        ("IT_ASSET", config.it_asset),
        ("ALLOCATION", config.allocation),
        ("LEASE_CLASSIFICATION", config.lease_classification),
        ("EF_SOURCE", config.ef_source),
        ("COMPLIANCE", config.compliance),
        ("SPEND", config.spend),
        ("UNCERTAINTY", config.uncertainty),
        ("PROVENANCE", config.provenance),
        ("METRICS", config.metrics),
    ]

    for name, section in section_names:
        print(f"\n[{name}]")
        for key, value in section.to_dict().items():
            if key == "password":
                print(f"  {key}: [REDACTED]")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 64)


def get_config_summary(config: UpstreamLeasedAssetsConfig) -> Dict[str, Any]:
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
        'GL-MRV-S3-008'
    """
    return {
        "agent_id": config.general.agent_id,
        "component": config.general.component,
        "version": config.general.version,
        "enabled": config.general.enabled,
        "table_prefix": config.general.table_prefix,
        "log_level": config.general.log_level,
        "database_host": config.database.host,
        "database_schema": config.database.schema,
        "redis_prefix": config.redis.prefix,
        "default_building_type": config.building.default_building_type,
        "default_climate_zone": config.building.default_climate_zone,
        "allocation_method": config.allocation.default_method,
        "lease_standard": config.lease_classification.accounting_standard,
        "ef_primary_source": config.ef_source.primary_source,
        "ef_year": config.ef_source.ef_year,
        "compliance_frameworks": config.compliance.get_frameworks(),
        "uncertainty_method": config.uncertainty.method,
        "uncertainty_confidence": str(config.uncertainty.confidence),
        "provenance_algorithm": config.provenance.algorithm,
        "metrics_prefix": config.metrics.prefix,
    }
