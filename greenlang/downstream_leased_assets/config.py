# -*- coding: utf-8 -*-
"""
Downstream Leased Assets Configuration - AGENT-MRV-026

Thread-safe singleton configuration for GL-MRV-S3-013.
All environment variables prefixed with GL_DLA_.

This module provides comprehensive configuration management for the Downstream
Leased Assets agent (GHG Protocol Scope 3 Category 13), supporting:
- Building asset emissions (8 building types, climate zone adjustments)
- Vehicle asset emissions (8 vehicle types, WTT inclusion)
- Equipment asset emissions (6 equipment types, load factor adjustments)
- IT asset emissions (7 IT asset types, PUE adjustments)
- Asset-specific calculation method (direct metering, sub-metering)
- Average-data calculation method (benchmarks, regional defaults)
- Spend-based fallback calculations (EEIO, CPI deflation)
- Hybrid method with waterfall and gap-filling
- Allocation methods (floor area, headcount, FTE, revenue, custom)
- Vacancy tracking and common area allocation
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi, GRI, DEFRA)
- Double-counting prevention and boundary enforcement
- Provenance tracking and audit trails
- Uncertainty quantification (Monte Carlo, bootstrap)
- 5-dimension Data Quality Indicator scoring

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-013'
    >>> config.building.default_climate_zone
    'TEMPERATE'
    >>> config.compliance.strict_mode
    True

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_DLA_ prefix. See individual config sections for specific variables.
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
    General configuration for Downstream Leased Assets agent.

    Attributes:
        enabled: Master switch for the agent (GL_DLA_ENABLED)
        debug: Enable debug mode with verbose logging (GL_DLA_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_DLA_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_DLA_AGENT_ID)
        agent_component: Agent component identifier (GL_DLA_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_DLA_VERSION)
        api_prefix: API route prefix (GL_DLA_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_DLA_MAX_BATCH_SIZE)
        default_gwp: Default GWP assessment report version (GL_DLA_DEFAULT_GWP)
        default_ef_source: Default emission factor source (GL_DLA_DEFAULT_EF_SOURCE)
        table_prefix: Prefix for all database tables (GL_DLA_TABLE_PREFIX)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-013",
        ...     agent_component="AGENT-MRV-026",
        ...     version="1.0.0",
        ...     api_prefix="/api/v1/downstream-leased-assets",
        ...     max_batch_size=1000,
        ...     default_gwp="AR5",
        ...     default_ef_source="DEFRA",
        ...     table_prefix="gl_dla_"
        ... )
        >>> general.agent_id
        'GL-MRV-S3-013'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-013"
    agent_component: str = "AGENT-MRV-026"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/downstream-leased-assets"
    max_batch_size: int = 1000
    default_gwp: str = "AR5"
    default_ef_source: str = "DEFRA"
    table_prefix: str = "gl_dla_"

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

        valid_ef_sources = {"DEFRA", "EPA", "IEA", "EEIO", "CIBSE", "ASHRAE", "CUSTOM"}
        if self.default_ef_source not in valid_ef_sources:
            raise ValueError(
                f"Invalid default_ef_source '{self.default_ef_source}'. "
                f"Must be one of {valid_ef_sources}"
            )

        if not self.table_prefix:
            raise ValueError("table_prefix cannot be empty")

        if not self.table_prefix.endswith("_"):
            raise ValueError("table_prefix must end with '_'")

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
            "table_prefix": self.table_prefix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_DLA_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_DLA_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_DLA_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_DLA_AGENT_ID", "GL-MRV-S3-013"),
            agent_component=os.getenv("GL_DLA_AGENT_COMPONENT", "AGENT-MRV-026"),
            version=os.getenv("GL_DLA_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_DLA_API_PREFIX", "/api/v1/downstream-leased-assets"),
            max_batch_size=int(os.getenv("GL_DLA_MAX_BATCH_SIZE", "1000")),
            default_gwp=os.getenv("GL_DLA_DEFAULT_GWP", "AR5"),
            default_ef_source=os.getenv("GL_DLA_DEFAULT_EF_SOURCE", "DEFRA"),
            table_prefix=os.getenv("GL_DLA_TABLE_PREFIX", "gl_dla_"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Downstream Leased Assets agent.

    Attributes:
        host: PostgreSQL host (GL_DLA_DB_HOST)
        port: PostgreSQL port (GL_DLA_DB_PORT)
        database: Database name (GL_DLA_DB_DATABASE)
        username: Database username (GL_DLA_DB_USERNAME)
        password: Database password (GL_DLA_DB_PASSWORD)
        schema: Database schema name (GL_DLA_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_DLA_DB_TABLE_PREFIX)
        pool_min: Minimum connection pool size (GL_DLA_DB_POOL_MIN)
        pool_max: Maximum connection pool size (GL_DLA_DB_POOL_MAX)
        pool_size: Default pool size (GL_DLA_DB_POOL_SIZE)
        ssl_mode: SSL connection mode (GL_DLA_DB_SSL_MODE)
        statement_timeout: Statement timeout in milliseconds (GL_DLA_DB_STATEMENT_TIMEOUT)
        connection_timeout: Connection timeout in seconds (GL_DLA_DB_CONNECTION_TIMEOUT)

    Example:
        >>> db = DatabaseConfig(
        ...     host="localhost",
        ...     port=5432,
        ...     database="greenlang",
        ...     schema="downstream_leased_assets_service",
        ...     pool_size=10,
        ...     ssl_mode="prefer",
        ...     statement_timeout=30000
        ... )
        >>> db.table_prefix
        'gl_dla_'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "downstream_leased_assets_service"
    table_prefix: str = "gl_dla_"
    pool_min: int = 2
    pool_max: int = 10
    pool_size: int = 10
    ssl_mode: str = "prefer"
    statement_timeout: int = 30000
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

        if self.pool_size < 1 or self.pool_size > 100:
            raise ValueError("pool_size must be between 1 and 100")

        valid_ssl_modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl_modes:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. "
                f"Must be one of {valid_ssl_modes}"
            )

        if self.statement_timeout < 1000 or self.statement_timeout > 300000:
            raise ValueError("statement_timeout must be between 1000 and 300000 ms")

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
            "pool_size": self.pool_size,
            "ssl_mode": self.ssl_mode,
            "statement_timeout": self.statement_timeout,
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
            host=os.getenv("GL_DLA_DB_HOST", "localhost"),
            port=int(os.getenv("GL_DLA_DB_PORT", "5432")),
            database=os.getenv("GL_DLA_DB_DATABASE", "greenlang"),
            username=os.getenv("GL_DLA_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_DLA_DB_PASSWORD", ""),
            schema=os.getenv("GL_DLA_DB_SCHEMA", "downstream_leased_assets_service"),
            table_prefix=os.getenv("GL_DLA_DB_TABLE_PREFIX", "gl_dla_"),
            pool_min=int(os.getenv("GL_DLA_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_DLA_DB_POOL_MAX", "10")),
            pool_size=int(os.getenv("GL_DLA_DB_POOL_SIZE", "10")),
            ssl_mode=os.getenv("GL_DLA_DB_SSL_MODE", "prefer"),
            statement_timeout=int(os.getenv("GL_DLA_DB_STATEMENT_TIMEOUT", "30000")),
            connection_timeout=int(os.getenv("GL_DLA_DB_CONNECTION_TIMEOUT", "30")),
        )


# =============================================================================
# SECTION 3: BUILDING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class BuildingConfig:
    """
    Building asset configuration for Downstream Leased Assets agent.

    Configures emissions calculations for 8 building types leased to downstream
    tenants. Supports climate zone adjustments, vacancy tracking, and common
    area allocation per GHG Protocol Scope 3 Category 13 guidance.

    Building Types (8):
        OFFICE, RETAIL, WAREHOUSE, INDUSTRIAL, RESIDENTIAL,
        DATA_CENTER, MIXED_USE, OTHER

    Attributes:
        enable_office: Enable office building type (GL_DLA_BLDG_ENABLE_OFFICE)
        enable_retail: Enable retail building type (GL_DLA_BLDG_ENABLE_RETAIL)
        enable_warehouse: Enable warehouse type (GL_DLA_BLDG_ENABLE_WAREHOUSE)
        enable_industrial: Enable industrial type (GL_DLA_BLDG_ENABLE_INDUSTRIAL)
        enable_residential: Enable residential type (GL_DLA_BLDG_ENABLE_RESIDENTIAL)
        enable_data_center: Enable data center type (GL_DLA_BLDG_ENABLE_DATA_CENTER)
        enable_mixed_use: Enable mixed-use type (GL_DLA_BLDG_ENABLE_MIXED_USE)
        enable_other: Enable other building type (GL_DLA_BLDG_ENABLE_OTHER)
        enable_all_8_types: Convenience flag to enable all 8 types (GL_DLA_BLDG_ENABLE_ALL)
        default_climate_zone: Default ASHRAE climate zone (GL_DLA_BLDG_DEFAULT_CLIMATE_ZONE)
        vacancy_tracking: Enable vacancy rate tracking (GL_DLA_BLDG_VACANCY_TRACKING)
        common_area_allocation: Include common area in allocation (GL_DLA_BLDG_COMMON_AREA_ALLOCATION)
        default_eui_unit: Default energy use intensity unit (GL_DLA_BLDG_DEFAULT_EUI_UNIT)
        default_area_unit: Default floor area unit (GL_DLA_BLDG_DEFAULT_AREA_UNIT)

    Example:
        >>> bldg = BuildingConfig(
        ...     enable_all_8_types=True,
        ...     default_climate_zone="TEMPERATE",
        ...     vacancy_tracking=True,
        ...     common_area_allocation=True
        ... )
        >>> bldg.default_climate_zone
        'TEMPERATE'
    """

    enable_office: bool = True
    enable_retail: bool = True
    enable_warehouse: bool = True
    enable_industrial: bool = True
    enable_residential: bool = True
    enable_data_center: bool = True
    enable_mixed_use: bool = True
    enable_other: bool = True
    enable_all_8_types: bool = True
    default_climate_zone: str = "TEMPERATE"
    vacancy_tracking: bool = True
    common_area_allocation: bool = True
    default_eui_unit: str = "kWh/m2/year"
    default_area_unit: str = "m2"

    def validate(self) -> None:
        """
        Validate building configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_climate_zones = {
            "TROPICAL", "ARID", "TEMPERATE", "CONTINENTAL", "POLAR",
            "1A", "2A", "2B", "3A", "3B", "3C", "4A", "4B", "4C",
            "5A", "5B", "5C", "6A", "6B", "7", "8",
        }
        if self.default_climate_zone not in valid_climate_zones:
            raise ValueError(
                f"Invalid default_climate_zone '{self.default_climate_zone}'. "
                f"Must be one of {valid_climate_zones}"
            )

        valid_eui_units = {"kWh/m2/year", "kBtu/ft2/year", "MJ/m2/year", "GJ/m2/year"}
        if self.default_eui_unit not in valid_eui_units:
            raise ValueError(
                f"Invalid default_eui_unit '{self.default_eui_unit}'. "
                f"Must be one of {valid_eui_units}"
            )

        valid_area_units = {"m2", "ft2"}
        if self.default_area_unit not in valid_area_units:
            raise ValueError(
                f"Invalid default_area_unit '{self.default_area_unit}'. "
                f"Must be one of {valid_area_units}"
            )

    def get_enabled_types(self) -> List[str]:
        """Return list of enabled building types."""
        types: List[str] = []
        if self.enable_all_8_types:
            return [
                "OFFICE", "RETAIL", "WAREHOUSE", "INDUSTRIAL",
                "RESIDENTIAL", "DATA_CENTER", "MIXED_USE", "OTHER",
            ]
        if self.enable_office:
            types.append("OFFICE")
        if self.enable_retail:
            types.append("RETAIL")
        if self.enable_warehouse:
            types.append("WAREHOUSE")
        if self.enable_industrial:
            types.append("INDUSTRIAL")
        if self.enable_residential:
            types.append("RESIDENTIAL")
        if self.enable_data_center:
            types.append("DATA_CENTER")
        if self.enable_mixed_use:
            types.append("MIXED_USE")
        if self.enable_other:
            types.append("OTHER")
        return types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_office": self.enable_office,
            "enable_retail": self.enable_retail,
            "enable_warehouse": self.enable_warehouse,
            "enable_industrial": self.enable_industrial,
            "enable_residential": self.enable_residential,
            "enable_data_center": self.enable_data_center,
            "enable_mixed_use": self.enable_mixed_use,
            "enable_other": self.enable_other,
            "enable_all_8_types": self.enable_all_8_types,
            "default_climate_zone": self.default_climate_zone,
            "vacancy_tracking": self.vacancy_tracking,
            "common_area_allocation": self.common_area_allocation,
            "default_eui_unit": self.default_eui_unit,
            "default_area_unit": self.default_area_unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildingConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "BuildingConfig":
        """Load from environment variables."""
        return cls(
            enable_office=os.getenv("GL_DLA_BLDG_ENABLE_OFFICE", "true").lower() == "true",
            enable_retail=os.getenv("GL_DLA_BLDG_ENABLE_RETAIL", "true").lower() == "true",
            enable_warehouse=os.getenv("GL_DLA_BLDG_ENABLE_WAREHOUSE", "true").lower() == "true",
            enable_industrial=os.getenv("GL_DLA_BLDG_ENABLE_INDUSTRIAL", "true").lower() == "true",
            enable_residential=os.getenv("GL_DLA_BLDG_ENABLE_RESIDENTIAL", "true").lower() == "true",
            enable_data_center=os.getenv("GL_DLA_BLDG_ENABLE_DATA_CENTER", "true").lower() == "true",
            enable_mixed_use=os.getenv("GL_DLA_BLDG_ENABLE_MIXED_USE", "true").lower() == "true",
            enable_other=os.getenv("GL_DLA_BLDG_ENABLE_OTHER", "true").lower() == "true",
            enable_all_8_types=os.getenv("GL_DLA_BLDG_ENABLE_ALL", "true").lower() == "true",
            default_climate_zone=os.getenv("GL_DLA_BLDG_DEFAULT_CLIMATE_ZONE", "TEMPERATE"),
            vacancy_tracking=os.getenv("GL_DLA_BLDG_VACANCY_TRACKING", "true").lower() == "true",
            common_area_allocation=os.getenv(
                "GL_DLA_BLDG_COMMON_AREA_ALLOCATION", "true"
            ).lower() == "true",
            default_eui_unit=os.getenv("GL_DLA_BLDG_DEFAULT_EUI_UNIT", "kWh/m2/year"),
            default_area_unit=os.getenv("GL_DLA_BLDG_DEFAULT_AREA_UNIT", "m2"),
        )


# =============================================================================
# SECTION 4: VEHICLE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class VehicleConfig:
    """
    Vehicle asset configuration for Downstream Leased Assets agent.

    Configures emissions calculations for 8 vehicle types leased to downstream
    users. Supports fuel-based and distance-based methods with well-to-tank
    (WTT) emissions inclusion.

    Vehicle Types (8):
        PASSENGER_CAR, LIGHT_TRUCK, HEAVY_TRUCK, BUS, VAN,
        MOTORCYCLE, SPECIAL_PURPOSE, OTHER

    Attributes:
        enable_passenger_car: Enable passenger car type (GL_DLA_VEH_ENABLE_PASSENGER_CAR)
        enable_light_truck: Enable light truck type (GL_DLA_VEH_ENABLE_LIGHT_TRUCK)
        enable_heavy_truck: Enable heavy truck type (GL_DLA_VEH_ENABLE_HEAVY_TRUCK)
        enable_bus: Enable bus type (GL_DLA_VEH_ENABLE_BUS)
        enable_van: Enable van type (GL_DLA_VEH_ENABLE_VAN)
        enable_motorcycle: Enable motorcycle type (GL_DLA_VEH_ENABLE_MOTORCYCLE)
        enable_special_purpose: Enable special purpose type (GL_DLA_VEH_ENABLE_SPECIAL_PURPOSE)
        enable_other: Enable other vehicle type (GL_DLA_VEH_ENABLE_OTHER)
        enable_all_8_types: Convenience flag to enable all 8 types (GL_DLA_VEH_ENABLE_ALL)
        default_fuel_type: Default fuel type (GL_DLA_VEH_DEFAULT_FUEL_TYPE)
        include_wtt: Include well-to-tank emissions (GL_DLA_VEH_INCLUDE_WTT)
        default_distance_unit: Default distance unit (GL_DLA_VEH_DEFAULT_DISTANCE_UNIT)
        default_fuel_unit: Default fuel unit (GL_DLA_VEH_DEFAULT_FUEL_UNIT)

    Example:
        >>> veh = VehicleConfig(
        ...     enable_all_8_types=True,
        ...     default_fuel_type="DIESEL",
        ...     include_wtt=True
        ... )
        >>> veh.default_fuel_type
        'DIESEL'
    """

    enable_passenger_car: bool = True
    enable_light_truck: bool = True
    enable_heavy_truck: bool = True
    enable_bus: bool = True
    enable_van: bool = True
    enable_motorcycle: bool = True
    enable_special_purpose: bool = True
    enable_other: bool = True
    enable_all_8_types: bool = True
    default_fuel_type: str = "DIESEL"
    include_wtt: bool = True
    default_distance_unit: str = "km"
    default_fuel_unit: str = "litres"

    def validate(self) -> None:
        """
        Validate vehicle configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_fuel_types = {
            "DIESEL", "PETROL", "GASOLINE", "LPG", "CNG", "LNG",
            "ELECTRIC", "HYBRID", "HYDROGEN", "BIOFUEL", "OTHER",
        }
        if self.default_fuel_type not in valid_fuel_types:
            raise ValueError(
                f"Invalid default_fuel_type '{self.default_fuel_type}'. "
                f"Must be one of {valid_fuel_types}"
            )

        valid_distance_units = {"km", "miles"}
        if self.default_distance_unit not in valid_distance_units:
            raise ValueError(
                f"Invalid default_distance_unit '{self.default_distance_unit}'. "
                f"Must be one of {valid_distance_units}"
            )

        valid_fuel_units = {"litres", "gallons", "kg", "m3", "kWh"}
        if self.default_fuel_unit not in valid_fuel_units:
            raise ValueError(
                f"Invalid default_fuel_unit '{self.default_fuel_unit}'. "
                f"Must be one of {valid_fuel_units}"
            )

    def get_enabled_types(self) -> List[str]:
        """Return list of enabled vehicle types."""
        if self.enable_all_8_types:
            return [
                "PASSENGER_CAR", "LIGHT_TRUCK", "HEAVY_TRUCK", "BUS",
                "VAN", "MOTORCYCLE", "SPECIAL_PURPOSE", "OTHER",
            ]
        types: List[str] = []
        if self.enable_passenger_car:
            types.append("PASSENGER_CAR")
        if self.enable_light_truck:
            types.append("LIGHT_TRUCK")
        if self.enable_heavy_truck:
            types.append("HEAVY_TRUCK")
        if self.enable_bus:
            types.append("BUS")
        if self.enable_van:
            types.append("VAN")
        if self.enable_motorcycle:
            types.append("MOTORCYCLE")
        if self.enable_special_purpose:
            types.append("SPECIAL_PURPOSE")
        if self.enable_other:
            types.append("OTHER")
        return types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_passenger_car": self.enable_passenger_car,
            "enable_light_truck": self.enable_light_truck,
            "enable_heavy_truck": self.enable_heavy_truck,
            "enable_bus": self.enable_bus,
            "enable_van": self.enable_van,
            "enable_motorcycle": self.enable_motorcycle,
            "enable_special_purpose": self.enable_special_purpose,
            "enable_other": self.enable_other,
            "enable_all_8_types": self.enable_all_8_types,
            "default_fuel_type": self.default_fuel_type,
            "include_wtt": self.include_wtt,
            "default_distance_unit": self.default_distance_unit,
            "default_fuel_unit": self.default_fuel_unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VehicleConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "VehicleConfig":
        """Load from environment variables."""
        return cls(
            enable_passenger_car=os.getenv(
                "GL_DLA_VEH_ENABLE_PASSENGER_CAR", "true"
            ).lower() == "true",
            enable_light_truck=os.getenv(
                "GL_DLA_VEH_ENABLE_LIGHT_TRUCK", "true"
            ).lower() == "true",
            enable_heavy_truck=os.getenv(
                "GL_DLA_VEH_ENABLE_HEAVY_TRUCK", "true"
            ).lower() == "true",
            enable_bus=os.getenv("GL_DLA_VEH_ENABLE_BUS", "true").lower() == "true",
            enable_van=os.getenv("GL_DLA_VEH_ENABLE_VAN", "true").lower() == "true",
            enable_motorcycle=os.getenv(
                "GL_DLA_VEH_ENABLE_MOTORCYCLE", "true"
            ).lower() == "true",
            enable_special_purpose=os.getenv(
                "GL_DLA_VEH_ENABLE_SPECIAL_PURPOSE", "true"
            ).lower() == "true",
            enable_other=os.getenv("GL_DLA_VEH_ENABLE_OTHER", "true").lower() == "true",
            enable_all_8_types=os.getenv(
                "GL_DLA_VEH_ENABLE_ALL", "true"
            ).lower() == "true",
            default_fuel_type=os.getenv("GL_DLA_VEH_DEFAULT_FUEL_TYPE", "DIESEL"),
            include_wtt=os.getenv("GL_DLA_VEH_INCLUDE_WTT", "true").lower() == "true",
            default_distance_unit=os.getenv("GL_DLA_VEH_DEFAULT_DISTANCE_UNIT", "km"),
            default_fuel_unit=os.getenv("GL_DLA_VEH_DEFAULT_FUEL_UNIT", "litres"),
        )


# =============================================================================
# SECTION 5: EQUIPMENT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EquipmentConfig:
    """
    Equipment asset configuration for Downstream Leased Assets agent.

    Configures emissions calculations for 6 equipment types leased to downstream
    users. Supports load factor adjustments and operating hour tracking.

    Equipment Types (6):
        GENERATOR, COMPRESSOR, HVAC, REFRIGERATION, MACHINERY, OTHER

    Attributes:
        enable_generator: Enable generator type (GL_DLA_EQUIP_ENABLE_GENERATOR)
        enable_compressor: Enable compressor type (GL_DLA_EQUIP_ENABLE_COMPRESSOR)
        enable_hvac: Enable HVAC type (GL_DLA_EQUIP_ENABLE_HVAC)
        enable_refrigeration: Enable refrigeration type (GL_DLA_EQUIP_ENABLE_REFRIGERATION)
        enable_machinery: Enable machinery type (GL_DLA_EQUIP_ENABLE_MACHINERY)
        enable_other: Enable other equipment type (GL_DLA_EQUIP_ENABLE_OTHER)
        enable_all_6_types: Convenience flag to enable all 6 types (GL_DLA_EQUIP_ENABLE_ALL)
        default_load_factor: Default equipment load factor 0-1 (GL_DLA_EQUIP_DEFAULT_LOAD_FACTOR)
        track_operating_hours: Track operating hours (GL_DLA_EQUIP_TRACK_OPERATING_HOURS)
        default_operating_hours: Default annual operating hours (GL_DLA_EQUIP_DEFAULT_OPERATING_HOURS)

    Example:
        >>> equip = EquipmentConfig(
        ...     enable_all_6_types=True,
        ...     default_load_factor=Decimal("0.6")
        ... )
        >>> equip.default_load_factor
        Decimal('0.6')
    """

    enable_generator: bool = True
    enable_compressor: bool = True
    enable_hvac: bool = True
    enable_refrigeration: bool = True
    enable_machinery: bool = True
    enable_other: bool = True
    enable_all_6_types: bool = True
    default_load_factor: Decimal = Decimal("0.6")
    track_operating_hours: bool = True
    default_operating_hours: int = 2000

    def validate(self) -> None:
        """
        Validate equipment configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_load_factor < Decimal("0") or self.default_load_factor > Decimal("1"):
            raise ValueError("default_load_factor must be between 0 and 1")

        if self.default_operating_hours < 1 or self.default_operating_hours > 8784:
            raise ValueError(
                "default_operating_hours must be between 1 and 8784 (hours per year)"
            )

    def get_enabled_types(self) -> List[str]:
        """Return list of enabled equipment types."""
        if self.enable_all_6_types:
            return [
                "GENERATOR", "COMPRESSOR", "HVAC",
                "REFRIGERATION", "MACHINERY", "OTHER",
            ]
        types: List[str] = []
        if self.enable_generator:
            types.append("GENERATOR")
        if self.enable_compressor:
            types.append("COMPRESSOR")
        if self.enable_hvac:
            types.append("HVAC")
        if self.enable_refrigeration:
            types.append("REFRIGERATION")
        if self.enable_machinery:
            types.append("MACHINERY")
        if self.enable_other:
            types.append("OTHER")
        return types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_generator": self.enable_generator,
            "enable_compressor": self.enable_compressor,
            "enable_hvac": self.enable_hvac,
            "enable_refrigeration": self.enable_refrigeration,
            "enable_machinery": self.enable_machinery,
            "enable_other": self.enable_other,
            "enable_all_6_types": self.enable_all_6_types,
            "default_load_factor": str(self.default_load_factor),
            "track_operating_hours": self.track_operating_hours,
            "default_operating_hours": self.default_operating_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquipmentConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_load_factor" in data_copy:
            data_copy["default_load_factor"] = Decimal(data_copy["default_load_factor"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "EquipmentConfig":
        """Load from environment variables."""
        return cls(
            enable_generator=os.getenv(
                "GL_DLA_EQUIP_ENABLE_GENERATOR", "true"
            ).lower() == "true",
            enable_compressor=os.getenv(
                "GL_DLA_EQUIP_ENABLE_COMPRESSOR", "true"
            ).lower() == "true",
            enable_hvac=os.getenv("GL_DLA_EQUIP_ENABLE_HVAC", "true").lower() == "true",
            enable_refrigeration=os.getenv(
                "GL_DLA_EQUIP_ENABLE_REFRIGERATION", "true"
            ).lower() == "true",
            enable_machinery=os.getenv(
                "GL_DLA_EQUIP_ENABLE_MACHINERY", "true"
            ).lower() == "true",
            enable_other=os.getenv(
                "GL_DLA_EQUIP_ENABLE_OTHER", "true"
            ).lower() == "true",
            enable_all_6_types=os.getenv(
                "GL_DLA_EQUIP_ENABLE_ALL", "true"
            ).lower() == "true",
            default_load_factor=Decimal(
                os.getenv("GL_DLA_EQUIP_DEFAULT_LOAD_FACTOR", "0.6")
            ),
            track_operating_hours=os.getenv(
                "GL_DLA_EQUIP_TRACK_OPERATING_HOURS", "true"
            ).lower() == "true",
            default_operating_hours=int(
                os.getenv("GL_DLA_EQUIP_DEFAULT_OPERATING_HOURS", "2000")
            ),
        )


# =============================================================================
# SECTION 6: IT ASSET CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ITAssetConfig:
    """
    IT asset configuration for Downstream Leased Assets agent.

    Configures emissions calculations for 7 IT asset types leased to downstream
    users. Supports PUE (Power Usage Effectiveness) adjustments for data center
    equipment and energy consumption tracking.

    IT Asset Types (7):
        SERVER, STORAGE, NETWORK_EQUIPMENT, DESKTOP, LAPTOP,
        PRINTER, OTHER

    Attributes:
        enable_server: Enable server type (GL_DLA_IT_ENABLE_SERVER)
        enable_storage: Enable storage type (GL_DLA_IT_ENABLE_STORAGE)
        enable_network: Enable network equipment type (GL_DLA_IT_ENABLE_NETWORK)
        enable_desktop: Enable desktop type (GL_DLA_IT_ENABLE_DESKTOP)
        enable_laptop: Enable laptop type (GL_DLA_IT_ENABLE_LAPTOP)
        enable_printer: Enable printer type (GL_DLA_IT_ENABLE_PRINTER)
        enable_other: Enable other IT asset type (GL_DLA_IT_ENABLE_OTHER)
        enable_all_7_types: Convenience flag to enable all 7 types (GL_DLA_IT_ENABLE_ALL)
        default_pue: Default Power Usage Effectiveness (GL_DLA_IT_DEFAULT_PUE)
        include_cooling: Include cooling energy in PUE (GL_DLA_IT_INCLUDE_COOLING)
        default_utilization: Default server utilization rate (GL_DLA_IT_DEFAULT_UTILIZATION)
        default_power_unit: Default power consumption unit (GL_DLA_IT_DEFAULT_POWER_UNIT)

    Example:
        >>> it = ITAssetConfig(
        ...     enable_all_7_types=True,
        ...     default_pue=Decimal("1.6")
        ... )
        >>> it.default_pue
        Decimal('1.6')
    """

    enable_server: bool = True
    enable_storage: bool = True
    enable_network: bool = True
    enable_desktop: bool = True
    enable_laptop: bool = True
    enable_printer: bool = True
    enable_other: bool = True
    enable_all_7_types: bool = True
    default_pue: Decimal = Decimal("1.6")
    include_cooling: bool = True
    default_utilization: Decimal = Decimal("0.5")
    default_power_unit: str = "kWh"

    def validate(self) -> None:
        """
        Validate IT asset configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_pue < Decimal("1.0") or self.default_pue > Decimal("3.0"):
            raise ValueError("default_pue must be between 1.0 and 3.0")

        if self.default_utilization < Decimal("0") or self.default_utilization > Decimal("1"):
            raise ValueError("default_utilization must be between 0 and 1")

        valid_power_units = {"kWh", "MWh", "GJ", "BTU"}
        if self.default_power_unit not in valid_power_units:
            raise ValueError(
                f"Invalid default_power_unit '{self.default_power_unit}'. "
                f"Must be one of {valid_power_units}"
            )

    def get_enabled_types(self) -> List[str]:
        """Return list of enabled IT asset types."""
        if self.enable_all_7_types:
            return [
                "SERVER", "STORAGE", "NETWORK_EQUIPMENT",
                "DESKTOP", "LAPTOP", "PRINTER", "OTHER",
            ]
        types: List[str] = []
        if self.enable_server:
            types.append("SERVER")
        if self.enable_storage:
            types.append("STORAGE")
        if self.enable_network:
            types.append("NETWORK_EQUIPMENT")
        if self.enable_desktop:
            types.append("DESKTOP")
        if self.enable_laptop:
            types.append("LAPTOP")
        if self.enable_printer:
            types.append("PRINTER")
        if self.enable_other:
            types.append("OTHER")
        return types

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_server": self.enable_server,
            "enable_storage": self.enable_storage,
            "enable_network": self.enable_network,
            "enable_desktop": self.enable_desktop,
            "enable_laptop": self.enable_laptop,
            "enable_printer": self.enable_printer,
            "enable_other": self.enable_other,
            "enable_all_7_types": self.enable_all_7_types,
            "default_pue": str(self.default_pue),
            "include_cooling": self.include_cooling,
            "default_utilization": str(self.default_utilization),
            "default_power_unit": self.default_power_unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ITAssetConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_pue", "default_utilization"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ITAssetConfig":
        """Load from environment variables."""
        return cls(
            enable_server=os.getenv("GL_DLA_IT_ENABLE_SERVER", "true").lower() == "true",
            enable_storage=os.getenv("GL_DLA_IT_ENABLE_STORAGE", "true").lower() == "true",
            enable_network=os.getenv("GL_DLA_IT_ENABLE_NETWORK", "true").lower() == "true",
            enable_desktop=os.getenv("GL_DLA_IT_ENABLE_DESKTOP", "true").lower() == "true",
            enable_laptop=os.getenv("GL_DLA_IT_ENABLE_LAPTOP", "true").lower() == "true",
            enable_printer=os.getenv("GL_DLA_IT_ENABLE_PRINTER", "true").lower() == "true",
            enable_other=os.getenv("GL_DLA_IT_ENABLE_OTHER", "true").lower() == "true",
            enable_all_7_types=os.getenv("GL_DLA_IT_ENABLE_ALL", "true").lower() == "true",
            default_pue=Decimal(os.getenv("GL_DLA_IT_DEFAULT_PUE", "1.6")),
            include_cooling=os.getenv(
                "GL_DLA_IT_INCLUDE_COOLING", "true"
            ).lower() == "true",
            default_utilization=Decimal(
                os.getenv("GL_DLA_IT_DEFAULT_UTILIZATION", "0.5")
            ),
            default_power_unit=os.getenv("GL_DLA_IT_DEFAULT_POWER_UNIT", "kWh"),
        )


# =============================================================================
# SECTION 7: AVERAGE DATA CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AverageDataConfig:
    """
    Average-data calculation configuration for Downstream Leased Assets agent.

    Configures average-data method that uses industry benchmarks and regional
    defaults when asset-specific data is unavailable.

    Attributes:
        default_region: Default geographic region (GL_DLA_AVG_DEFAULT_REGION)
        fallback_to_benchmark: Fall back to benchmarks when data missing (GL_DLA_AVG_FALLBACK_TO_BENCHMARK)
        benchmark_source: Default benchmark data source (GL_DLA_AVG_BENCHMARK_SOURCE)
        benchmark_year: Benchmark data year (GL_DLA_AVG_BENCHMARK_YEAR)
        enable_climate_adjustment: Adjust benchmarks for climate zone (GL_DLA_AVG_ENABLE_CLIMATE_ADJUSTMENT)
        enable_vintage_adjustment: Adjust for building vintage (GL_DLA_AVG_ENABLE_VINTAGE_ADJUSTMENT)
        default_building_vintage: Default building vintage year (GL_DLA_AVG_DEFAULT_BUILDING_VINTAGE)

    Example:
        >>> avg = AverageDataConfig(
        ...     default_region="GLOBAL",
        ...     fallback_to_benchmark=True
        ... )
        >>> avg.default_region
        'GLOBAL'
    """

    default_region: str = "GLOBAL"
    fallback_to_benchmark: bool = True
    benchmark_source: str = "CIBSE"
    benchmark_year: int = 2024
    enable_climate_adjustment: bool = True
    enable_vintage_adjustment: bool = True
    default_building_vintage: int = 2000

    def validate(self) -> None:
        """
        Validate average-data configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_regions = {
            "GLOBAL", "NORTH_AMERICA", "EUROPE", "ASIA_PACIFIC",
            "LATIN_AMERICA", "MIDDLE_EAST", "AFRICA", "OCEANIA",
        }
        if self.default_region not in valid_regions:
            raise ValueError(
                f"Invalid default_region '{self.default_region}'. "
                f"Must be one of {valid_regions}"
            )

        valid_sources = {"CIBSE", "ASHRAE", "ENERGY_STAR", "CRREM", "CUSTOM"}
        if self.benchmark_source not in valid_sources:
            raise ValueError(
                f"Invalid benchmark_source '{self.benchmark_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.benchmark_year < 2000 or self.benchmark_year > 2030:
            raise ValueError("benchmark_year must be between 2000 and 2030")

        if self.default_building_vintage < 1900 or self.default_building_vintage > 2030:
            raise ValueError("default_building_vintage must be between 1900 and 2030")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_region": self.default_region,
            "fallback_to_benchmark": self.fallback_to_benchmark,
            "benchmark_source": self.benchmark_source,
            "benchmark_year": self.benchmark_year,
            "enable_climate_adjustment": self.enable_climate_adjustment,
            "enable_vintage_adjustment": self.enable_vintage_adjustment,
            "default_building_vintage": self.default_building_vintage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AverageDataConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AverageDataConfig":
        """Load from environment variables."""
        return cls(
            default_region=os.getenv("GL_DLA_AVG_DEFAULT_REGION", "GLOBAL"),
            fallback_to_benchmark=os.getenv(
                "GL_DLA_AVG_FALLBACK_TO_BENCHMARK", "true"
            ).lower() == "true",
            benchmark_source=os.getenv("GL_DLA_AVG_BENCHMARK_SOURCE", "CIBSE"),
            benchmark_year=int(os.getenv("GL_DLA_AVG_BENCHMARK_YEAR", "2024")),
            enable_climate_adjustment=os.getenv(
                "GL_DLA_AVG_ENABLE_CLIMATE_ADJUSTMENT", "true"
            ).lower() == "true",
            enable_vintage_adjustment=os.getenv(
                "GL_DLA_AVG_ENABLE_VINTAGE_ADJUSTMENT", "true"
            ).lower() == "true",
            default_building_vintage=int(
                os.getenv("GL_DLA_AVG_DEFAULT_BUILDING_VINTAGE", "2000")
            ),
        )


# =============================================================================
# SECTION 8: SPEND-BASED CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendBasedConfig:
    """
    Spend-based calculation configuration for Downstream Leased Assets agent.

    Configures spend-based fallback calculations using EEIO factors when
    asset-specific or average data is unavailable.

    Attributes:
        enable_eeio: Enable EEIO-based calculation (GL_DLA_SPEND_ENABLE_EEIO)
        default_currency: Default currency code (GL_DLA_SPEND_DEFAULT_CURRENCY)
        cpi_deflation: Enable CPI deflation adjustment (GL_DLA_SPEND_CPI_DEFLATION)
        base_year: Base year for CPI deflation (GL_DLA_SPEND_BASE_YEAR)
        enable_margin_removal: Enable margin removal from spend (GL_DLA_SPEND_ENABLE_MARGIN_REMOVAL)
        default_margin_rate: Default profit margin rate (GL_DLA_SPEND_DEFAULT_MARGIN_RATE)
        supported_currencies: Comma-separated supported currencies (GL_DLA_SPEND_SUPPORTED_CURRENCIES)
        enable_ppp_adjustment: Enable purchasing power parity adjustment (GL_DLA_SPEND_ENABLE_PPP_ADJUSTMENT)

    Example:
        >>> spend = SpendBasedConfig(
        ...     enable_eeio=True,
        ...     default_currency="USD",
        ...     cpi_deflation=True
        ... )
        >>> spend.default_currency
        'USD'
    """

    enable_eeio: bool = True
    default_currency: str = "USD"
    cpi_deflation: bool = True
    base_year: int = 2021
    enable_margin_removal: bool = False
    default_margin_rate: Decimal = Decimal("0.15")
    supported_currencies: str = (
        "USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED"
    )
    enable_ppp_adjustment: bool = False

    def validate(self) -> None:
        """
        Validate spend-based configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.default_currency:
            raise ValueError("default_currency cannot be empty")

        if len(self.default_currency) != 3:
            raise ValueError("default_currency must be a 3-character ISO 4217 code")

        if self.base_year < 2000 or self.base_year > 2030:
            raise ValueError("base_year must be between 2000 and 2030")

        if self.default_margin_rate < Decimal("0") or self.default_margin_rate > Decimal("1"):
            raise ValueError("default_margin_rate must be between 0 and 1")

        currencies = self.get_supported_currencies()
        if not currencies:
            raise ValueError("At least one supported currency must be defined")

    def get_supported_currencies(self) -> List[str]:
        """Parse supported currencies string into list."""
        return [c.strip() for c in self.supported_currencies.split(",") if c.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_eeio": self.enable_eeio,
            "default_currency": self.default_currency,
            "cpi_deflation": self.cpi_deflation,
            "base_year": self.base_year,
            "enable_margin_removal": self.enable_margin_removal,
            "default_margin_rate": str(self.default_margin_rate),
            "supported_currencies": self.supported_currencies,
            "enable_ppp_adjustment": self.enable_ppp_adjustment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendBasedConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_margin_rate" in data_copy:
            data_copy["default_margin_rate"] = Decimal(data_copy["default_margin_rate"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendBasedConfig":
        """Load from environment variables."""
        return cls(
            enable_eeio=os.getenv("GL_DLA_SPEND_ENABLE_EEIO", "true").lower() == "true",
            default_currency=os.getenv("GL_DLA_SPEND_DEFAULT_CURRENCY", "USD"),
            cpi_deflation=os.getenv(
                "GL_DLA_SPEND_CPI_DEFLATION", "true"
            ).lower() == "true",
            base_year=int(os.getenv("GL_DLA_SPEND_BASE_YEAR", "2021")),
            enable_margin_removal=os.getenv(
                "GL_DLA_SPEND_ENABLE_MARGIN_REMOVAL", "false"
            ).lower() == "true",
            default_margin_rate=Decimal(
                os.getenv("GL_DLA_SPEND_DEFAULT_MARGIN_RATE", "0.15")
            ),
            supported_currencies=os.getenv(
                "GL_DLA_SPEND_SUPPORTED_CURRENCIES",
                "USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED",
            ),
            enable_ppp_adjustment=os.getenv(
                "GL_DLA_SPEND_ENABLE_PPP_ADJUSTMENT", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 9: HYBRID CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class HybridConfig:
    """
    Hybrid calculation configuration for Downstream Leased Assets agent.

    Configures the hybrid method that combines multiple calculation approaches
    in a waterfall pattern with gap-filling for missing data.

    Attributes:
        method_waterfall: Comma-separated method priority order (GL_DLA_HYBRID_METHOD_WATERFALL)
        gap_filling: Enable gap-filling with lower-priority methods (GL_DLA_HYBRID_GAP_FILLING)
        min_asset_specific_pct: Min percentage of asset-specific data to qualify (GL_DLA_HYBRID_MIN_ASSET_SPECIFIC_PCT)
        blend_methods: Allow blending results from multiple methods (GL_DLA_HYBRID_BLEND_METHODS)
        track_method_coverage: Track per-method data coverage (GL_DLA_HYBRID_TRACK_METHOD_COVERAGE)

    Example:
        >>> hybrid = HybridConfig(
        ...     method_waterfall="ASSET_SPECIFIC,AVERAGE_DATA,SPEND_BASED",
        ...     gap_filling=True
        ... )
        >>> hybrid.get_waterfall_order()
        ['ASSET_SPECIFIC', 'AVERAGE_DATA', 'SPEND_BASED']
    """

    method_waterfall: str = "ASSET_SPECIFIC,AVERAGE_DATA,SPEND_BASED"
    gap_filling: bool = True
    min_asset_specific_pct: Decimal = Decimal("0.0")
    blend_methods: bool = False
    track_method_coverage: bool = True

    def validate(self) -> None:
        """
        Validate hybrid configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"ASSET_SPECIFIC", "AVERAGE_DATA", "SPEND_BASED"}
        waterfall = self.get_waterfall_order()

        if not waterfall:
            raise ValueError("method_waterfall must contain at least one method")

        for method in waterfall:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method '{method}' in method_waterfall. "
                    f"Must be one of {valid_methods}"
                )

        if self.min_asset_specific_pct < Decimal("0") or self.min_asset_specific_pct > Decimal("1"):
            raise ValueError("min_asset_specific_pct must be between 0 and 1")

    def get_waterfall_order(self) -> List[str]:
        """Parse method waterfall string into ordered list."""
        return [m.strip() for m in self.method_waterfall.split(",") if m.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_waterfall": self.method_waterfall,
            "gap_filling": self.gap_filling,
            "min_asset_specific_pct": str(self.min_asset_specific_pct),
            "blend_methods": self.blend_methods,
            "track_method_coverage": self.track_method_coverage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "min_asset_specific_pct" in data_copy:
            data_copy["min_asset_specific_pct"] = Decimal(
                data_copy["min_asset_specific_pct"]
            )
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "HybridConfig":
        """Load from environment variables."""
        return cls(
            method_waterfall=os.getenv(
                "GL_DLA_HYBRID_METHOD_WATERFALL",
                "ASSET_SPECIFIC,AVERAGE_DATA,SPEND_BASED",
            ),
            gap_filling=os.getenv(
                "GL_DLA_HYBRID_GAP_FILLING", "true"
            ).lower() == "true",
            min_asset_specific_pct=Decimal(
                os.getenv("GL_DLA_HYBRID_MIN_ASSET_SPECIFIC_PCT", "0.0")
            ),
            blend_methods=os.getenv(
                "GL_DLA_HYBRID_BLEND_METHODS", "false"
            ).lower() == "true",
            track_method_coverage=os.getenv(
                "GL_DLA_HYBRID_TRACK_METHOD_COVERAGE", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 10: ALLOCATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AllocationConfig:
    """
    Allocation configuration for Downstream Leased Assets agent.

    Configures how emissions are allocated between lessor and lessee for
    multi-tenant or partially-leased assets. Supports multiple allocation
    methods per GHG Protocol Scope 3 Category 13 guidance.

    Attributes:
        default_method: Default allocation method (GL_DLA_ALLOC_DEFAULT_METHOD)
        common_area_included: Include common area in allocation (GL_DLA_ALLOC_COMMON_AREA_INCLUDED)
        vacancy_adjustment: Adjust for vacancy rates (GL_DLA_ALLOC_VACANCY_ADJUSTMENT)
        allow_custom_method: Allow custom allocation methods (GL_DLA_ALLOC_ALLOW_CUSTOM_METHOD)
        proportional_common_area: Proportionally allocate common area (GL_DLA_ALLOC_PROPORTIONAL_COMMON_AREA)
        validate_allocation_sum: Validate allocations sum to 100 percent (GL_DLA_ALLOC_VALIDATE_SUM)

    Example:
        >>> alloc = AllocationConfig(
        ...     default_method="FLOOR_AREA",
        ...     common_area_included=True,
        ...     vacancy_adjustment=True
        ... )
        >>> alloc.default_method
        'FLOOR_AREA'
    """

    default_method: str = "FLOOR_AREA"
    common_area_included: bool = True
    vacancy_adjustment: bool = True
    allow_custom_method: bool = True
    proportional_common_area: bool = True
    validate_allocation_sum: bool = True

    def validate(self) -> None:
        """
        Validate allocation configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {
            "FLOOR_AREA", "HEADCOUNT", "FTE", "REVENUE",
            "ENERGY_CONSUMPTION", "EQUAL", "CUSTOM",
        }
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "common_area_included": self.common_area_included,
            "vacancy_adjustment": self.vacancy_adjustment,
            "allow_custom_method": self.allow_custom_method,
            "proportional_common_area": self.proportional_common_area,
            "validate_allocation_sum": self.validate_allocation_sum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AllocationConfig":
        """Load from environment variables."""
        return cls(
            default_method=os.getenv("GL_DLA_ALLOC_DEFAULT_METHOD", "FLOOR_AREA"),
            common_area_included=os.getenv(
                "GL_DLA_ALLOC_COMMON_AREA_INCLUDED", "true"
            ).lower() == "true",
            vacancy_adjustment=os.getenv(
                "GL_DLA_ALLOC_VACANCY_ADJUSTMENT", "true"
            ).lower() == "true",
            allow_custom_method=os.getenv(
                "GL_DLA_ALLOC_ALLOW_CUSTOM_METHOD", "true"
            ).lower() == "true",
            proportional_common_area=os.getenv(
                "GL_DLA_ALLOC_PROPORTIONAL_COMMON_AREA", "true"
            ).lower() == "true",
            validate_allocation_sum=os.getenv(
                "GL_DLA_ALLOC_VALIDATE_SUM", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 11: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Downstream Leased Assets agent.

    Configures regulatory framework compliance checks for Scope 3 Category 13
    downstream leased assets emissions reporting. Supports 7 frameworks and
    enforces double-counting prevention.

    Attributes:
        compliance_frameworks: Enabled frameworks (GL_DLA_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict compliance mode (GL_DLA_COMPLIANCE_STRICT_MODE)
        materiality_threshold: Materiality threshold percentage (GL_DLA_COMPLIANCE_MATERIALITY_THRESHOLD)
        double_counting_check: Check for double counting (GL_DLA_COMPLIANCE_DOUBLE_COUNTING_CHECK)
        boundary_enforcement: Enforce Scope 3 boundary (GL_DLA_COMPLIANCE_BOUNDARY_ENFORCEMENT)
        require_data_quality: Require data quality scoring (GL_DLA_COMPLIANCE_REQUIRE_DATA_QUALITY)
        min_data_quality_score: Min acceptable DQI score (GL_DLA_COMPLIANCE_MIN_DATA_QUALITY_SCORE)
        require_allocation_disclosure: Require allocation method disclosure (GL_DLA_COMPLIANCE_REQUIRE_ALLOCATION_DISCLOSURE)

    Example:
        >>> compliance = ComplianceConfig(
        ...     strict_mode=True,
        ...     double_counting_check=True
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E1', 'CDP', 'SBTI', 'GRI', 'DEFRA_BEIS']
    """

    compliance_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,DEFRA_BEIS"
    )
    strict_mode: bool = True
    materiality_threshold: Decimal = Decimal("0.01")
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    require_data_quality: bool = True
    min_data_quality_score: Decimal = Decimal("2.0")
    require_allocation_disclosure: bool = True

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
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "require_data_quality": self.require_data_quality,
            "min_data_quality_score": str(self.min_data_quality_score),
            "require_allocation_disclosure": self.require_allocation_disclosure,
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
                "GL_DLA_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,DEFRA_BEIS",
            ),
            strict_mode=os.getenv(
                "GL_DLA_COMPLIANCE_STRICT_MODE", "true"
            ).lower() == "true",
            materiality_threshold=Decimal(
                os.getenv("GL_DLA_COMPLIANCE_MATERIALITY_THRESHOLD", "0.01")
            ),
            double_counting_check=os.getenv(
                "GL_DLA_COMPLIANCE_DOUBLE_COUNTING_CHECK", "true"
            ).lower() == "true",
            boundary_enforcement=os.getenv(
                "GL_DLA_COMPLIANCE_BOUNDARY_ENFORCEMENT", "true"
            ).lower() == "true",
            require_data_quality=os.getenv(
                "GL_DLA_COMPLIANCE_REQUIRE_DATA_QUALITY", "true"
            ).lower() == "true",
            min_data_quality_score=Decimal(
                os.getenv("GL_DLA_COMPLIANCE_MIN_DATA_QUALITY_SCORE", "2.0")
            ),
            require_allocation_disclosure=os.getenv(
                "GL_DLA_COMPLIANCE_REQUIRE_ALLOCATION_DISCLOSURE", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 12: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Downstream Leased Assets agent.

    Configures data provenance tracking with SHA-256 hashing for complete
    audit trails. Supports chain hashing and Merkle tree aggregation.

    Attributes:
        enabled: Enable provenance tracking (GL_DLA_PROVENANCE_ENABLED)
        hash_algorithm: Hash algorithm for provenance (GL_DLA_PROVENANCE_HASH_ALGORITHM)
        chain_enabled: Enable chain hashing across stages (GL_DLA_PROVENANCE_CHAIN_ENABLED)
        merkle_enabled: Enable Merkle tree batch aggregation (GL_DLA_PROVENANCE_MERKLE_ENABLED)
        store_intermediate: Store intermediate hashes (GL_DLA_PROVENANCE_STORE_INTERMEDIATE)
        include_config_hash: Include config hash in provenance (GL_DLA_PROVENANCE_INCLUDE_CONFIG_HASH)

    Example:
        >>> prov = ProvenanceConfig(
        ...     hash_algorithm="sha256",
        ...     chain_enabled=True,
        ...     merkle_enabled=True
        ... )
        >>> prov.hash_algorithm
        'sha256'
    """

    enabled: bool = True
    hash_algorithm: str = "sha256"
    chain_enabled: bool = True
    merkle_enabled: bool = True
    store_intermediate: bool = True
    include_config_hash: bool = True

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
            "chain_enabled": self.chain_enabled,
            "merkle_enabled": self.merkle_enabled,
            "store_intermediate": self.store_intermediate,
            "include_config_hash": self.include_config_hash,
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
                "GL_DLA_PROVENANCE_ENABLED", "true"
            ).lower() == "true",
            hash_algorithm=os.getenv("GL_DLA_PROVENANCE_HASH_ALGORITHM", "sha256"),
            chain_enabled=os.getenv(
                "GL_DLA_PROVENANCE_CHAIN_ENABLED", "true"
            ).lower() == "true",
            merkle_enabled=os.getenv(
                "GL_DLA_PROVENANCE_MERKLE_ENABLED", "true"
            ).lower() == "true",
            store_intermediate=os.getenv(
                "GL_DLA_PROVENANCE_STORE_INTERMEDIATE", "true"
            ).lower() == "true",
            include_config_hash=os.getenv(
                "GL_DLA_PROVENANCE_INCLUDE_CONFIG_HASH", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 13: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty configuration for Downstream Leased Assets agent.

    Configures uncertainty quantification for emissions calculations.
    Supports Monte Carlo simulation, bootstrap, and IPCC default ranges.

    Attributes:
        method: Default uncertainty method (GL_DLA_UNCERTAINTY_METHOD)
        iterations: Monte Carlo iterations (GL_DLA_UNCERTAINTY_ITERATIONS)
        confidence: Confidence level for intervals (GL_DLA_UNCERTAINTY_CONFIDENCE)
        include_parameter: Include parameter uncertainty (GL_DLA_UNCERTAINTY_INCLUDE_PARAMETER)
        include_model: Include model uncertainty (GL_DLA_UNCERTAINTY_INCLUDE_MODEL)
        include_activity: Include activity data uncertainty (GL_DLA_UNCERTAINTY_INCLUDE_ACTIVITY)
        seed: Random seed for reproducibility (GL_DLA_UNCERTAINTY_SEED)

    Example:
        >>> unc = UncertaintyConfig(
        ...     method="MONTE_CARLO",
        ...     iterations=10000,
        ...     confidence=Decimal("0.95")
        ... )
        >>> unc.method
        'MONTE_CARLO'
    """

    method: str = "MONTE_CARLO"
    iterations: int = 10000
    confidence: Decimal = Decimal("0.95")
    include_parameter: bool = True
    include_model: bool = False
    include_activity: bool = True
    seed: Optional[int] = None

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {
            "MONTE_CARLO", "BOOTSTRAP", "IPCC_DEFAULT",
            "BAYESIAN", "ANALYTICAL", "NONE",
        }
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.iterations < 100 or self.iterations > 1000000:
            raise ValueError("iterations must be between 100 and 1000000")

        if self.confidence < Decimal("0.5") or self.confidence > Decimal("0.999"):
            raise ValueError("confidence must be between 0.5 and 0.999")

        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 when specified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "iterations": self.iterations,
            "confidence": str(self.confidence),
            "include_parameter": self.include_parameter,
            "include_model": self.include_model,
            "include_activity": self.include_activity,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "confidence" in data_copy:
            data_copy["confidence"] = Decimal(data_copy["confidence"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        seed_str = os.getenv("GL_DLA_UNCERTAINTY_SEED")
        seed_val = int(seed_str) if seed_str else None

        return cls(
            method=os.getenv("GL_DLA_UNCERTAINTY_METHOD", "MONTE_CARLO"),
            iterations=int(os.getenv("GL_DLA_UNCERTAINTY_ITERATIONS", "10000")),
            confidence=Decimal(os.getenv("GL_DLA_UNCERTAINTY_CONFIDENCE", "0.95")),
            include_parameter=os.getenv(
                "GL_DLA_UNCERTAINTY_INCLUDE_PARAMETER", "true"
            ).lower() == "true",
            include_model=os.getenv(
                "GL_DLA_UNCERTAINTY_INCLUDE_MODEL", "false"
            ).lower() == "true",
            include_activity=os.getenv(
                "GL_DLA_UNCERTAINTY_INCLUDE_ACTIVITY", "true"
            ).lower() == "true",
            seed=seed_val,
        )


# =============================================================================
# SECTION 14: DQI CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DQIConfig:
    """
    Data Quality Indicator configuration for Downstream Leased Assets agent.

    Configures the 5-dimension DQI scoring system per GHG Protocol guidance.
    Each dimension is scored 1-5 (1=best, 5=worst) with configurable weights
    that must sum to 1.0.

    Dimensions:
        1. Technological representativeness
        2. Temporal representativeness
        3. Geographical representativeness
        4. Completeness
        5. Reliability

    Attributes:
        min_score: Minimum acceptable DQI score (GL_DLA_DQI_MIN_SCORE)
        weight_technological: Weight for technological dimension (GL_DLA_DQI_WEIGHT_TECH)
        weight_temporal: Weight for temporal dimension (GL_DLA_DQI_WEIGHT_TEMPORAL)
        weight_geographical: Weight for geographical dimension (GL_DLA_DQI_WEIGHT_GEO)
        weight_completeness: Weight for completeness dimension (GL_DLA_DQI_WEIGHT_COMPLETENESS)
        weight_reliability: Weight for reliability dimension (GL_DLA_DQI_WEIGHT_RELIABILITY)
        enable_per_asset_scoring: Score each asset individually (GL_DLA_DQI_ENABLE_PER_ASSET)
        aggregate_method: Aggregation method for portfolio DQI (GL_DLA_DQI_AGGREGATE_METHOD)

    Example:
        >>> dqi = DQIConfig(
        ...     min_score=Decimal("2.0"),
        ...     weight_technological=Decimal("0.2"),
        ...     weight_temporal=Decimal("0.2"),
        ...     weight_geographical=Decimal("0.2"),
        ...     weight_completeness=Decimal("0.2"),
        ...     weight_reliability=Decimal("0.2")
        ... )
        >>> dqi.get_weights_sum()
        Decimal('1.0')
    """

    min_score: Decimal = Decimal("2.0")
    weight_technological: Decimal = Decimal("0.2")
    weight_temporal: Decimal = Decimal("0.2")
    weight_geographical: Decimal = Decimal("0.2")
    weight_completeness: Decimal = Decimal("0.2")
    weight_reliability: Decimal = Decimal("0.2")
    enable_per_asset_scoring: bool = True
    aggregate_method: str = "WEIGHTED_AVERAGE"

    def validate(self) -> None:
        """
        Validate DQI configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.min_score < Decimal("1") or self.min_score > Decimal("5"):
            raise ValueError("min_score must be between 1 and 5")

        weights = [
            self.weight_technological,
            self.weight_temporal,
            self.weight_geographical,
            self.weight_completeness,
            self.weight_reliability,
        ]

        for w in weights:
            if w < Decimal("0") or w > Decimal("1"):
                raise ValueError("All DQI weights must be between 0 and 1")

        weight_sum = sum(weights)
        if abs(weight_sum - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(
                f"DQI weights must sum to 1.0, got {weight_sum}"
            )

        valid_aggregations = {"WEIGHTED_AVERAGE", "EMISSIONS_WEIGHTED", "MINIMUM", "MAXIMUM"}
        if self.aggregate_method not in valid_aggregations:
            raise ValueError(
                f"Invalid aggregate_method '{self.aggregate_method}'. "
                f"Must be one of {valid_aggregations}"
            )

    def get_weights_sum(self) -> Decimal:
        """Calculate the sum of all dimension weights."""
        return (
            self.weight_technological
            + self.weight_temporal
            + self.weight_geographical
            + self.weight_completeness
            + self.weight_reliability
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_score": str(self.min_score),
            "weight_technological": str(self.weight_technological),
            "weight_temporal": str(self.weight_temporal),
            "weight_geographical": str(self.weight_geographical),
            "weight_completeness": str(self.weight_completeness),
            "weight_reliability": str(self.weight_reliability),
            "enable_per_asset_scoring": self.enable_per_asset_scoring,
            "aggregate_method": self.aggregate_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DQIConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "min_score", "weight_technological", "weight_temporal",
            "weight_geographical", "weight_completeness", "weight_reliability",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "DQIConfig":
        """Load from environment variables."""
        return cls(
            min_score=Decimal(os.getenv("GL_DLA_DQI_MIN_SCORE", "2.0")),
            weight_technological=Decimal(
                os.getenv("GL_DLA_DQI_WEIGHT_TECH", "0.2")
            ),
            weight_temporal=Decimal(
                os.getenv("GL_DLA_DQI_WEIGHT_TEMPORAL", "0.2")
            ),
            weight_geographical=Decimal(
                os.getenv("GL_DLA_DQI_WEIGHT_GEO", "0.2")
            ),
            weight_completeness=Decimal(
                os.getenv("GL_DLA_DQI_WEIGHT_COMPLETENESS", "0.2")
            ),
            weight_reliability=Decimal(
                os.getenv("GL_DLA_DQI_WEIGHT_RELIABILITY", "0.2")
            ),
            enable_per_asset_scoring=os.getenv(
                "GL_DLA_DQI_ENABLE_PER_ASSET", "true"
            ).lower() == "true",
            aggregate_method=os.getenv("GL_DLA_DQI_AGGREGATE_METHOD", "WEIGHTED_AVERAGE"),
        )


# =============================================================================
# SECTION 15: PIPELINE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline configuration for Downstream Leased Assets agent.

    Configures the calculation pipeline including batch processing,
    timeouts, retries, and parallel execution settings.

    Attributes:
        batch_size: Records per batch (GL_DLA_PIPELINE_BATCH_SIZE)
        timeout: Pipeline timeout in seconds (GL_DLA_PIPELINE_TIMEOUT)
        max_retries: Maximum retry attempts (GL_DLA_PIPELINE_MAX_RETRIES)
        parallel_execution: Enable parallel execution (GL_DLA_PIPELINE_PARALLEL_EXECUTION)
        worker_count: Number of parallel workers (GL_DLA_PIPELINE_WORKER_COUNT)
        retry_delay: Delay between retries in seconds (GL_DLA_PIPELINE_RETRY_DELAY)
        enable_checkpointing: Enable pipeline checkpointing (GL_DLA_PIPELINE_ENABLE_CHECKPOINTING)

    Example:
        >>> pipeline = PipelineConfig(
        ...     batch_size=500,
        ...     timeout=300,
        ...     max_retries=3,
        ...     parallel_execution=True
        ... )
        >>> pipeline.batch_size
        500
    """

    batch_size: int = 500
    timeout: int = 300
    max_retries: int = 3
    parallel_execution: bool = True
    worker_count: int = 4
    retry_delay: int = 5
    enable_checkpointing: bool = True

    def validate(self) -> None:
        """
        Validate pipeline configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.batch_size < 1 or self.batch_size > 10000:
            raise ValueError("batch_size must be between 1 and 10000")

        if self.timeout < 10 or self.timeout > 3600:
            raise ValueError("timeout must be between 10 and 3600 seconds")

        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")

        if self.worker_count < 1 or self.worker_count > 64:
            raise ValueError("worker_count must be between 1 and 64")

        if self.retry_delay < 0 or self.retry_delay > 300:
            raise ValueError("retry_delay must be between 0 and 300 seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "parallel_execution": self.parallel_execution,
            "worker_count": self.worker_count,
            "retry_delay": self.retry_delay,
            "enable_checkpointing": self.enable_checkpointing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load from environment variables."""
        return cls(
            batch_size=int(os.getenv("GL_DLA_PIPELINE_BATCH_SIZE", "500")),
            timeout=int(os.getenv("GL_DLA_PIPELINE_TIMEOUT", "300")),
            max_retries=int(os.getenv("GL_DLA_PIPELINE_MAX_RETRIES", "3")),
            parallel_execution=os.getenv(
                "GL_DLA_PIPELINE_PARALLEL_EXECUTION", "true"
            ).lower() == "true",
            worker_count=int(os.getenv("GL_DLA_PIPELINE_WORKER_COUNT", "4")),
            retry_delay=int(os.getenv("GL_DLA_PIPELINE_RETRY_DELAY", "5")),
            enable_checkpointing=os.getenv(
                "GL_DLA_PIPELINE_ENABLE_CHECKPOINTING", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 16: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Downstream Leased Assets agent.

    Configures in-memory and Redis caching for emission factor lookups,
    calculation results, and benchmark data.

    Attributes:
        enabled: Enable caching (GL_DLA_CACHE_ENABLED)
        ttl: Cache TTL in seconds (GL_DLA_CACHE_TTL)
        max_size: Maximum cache entries (GL_DLA_CACHE_MAX_SIZE)
        key_prefix: Cache key prefix (GL_DLA_CACHE_KEY_PREFIX)
        cache_ef_lookups: Cache emission factor lookups (GL_DLA_CACHE_EF_LOOKUPS)
        cache_calculations: Cache calculation results (GL_DLA_CACHE_CALCULATIONS)
        cache_benchmarks: Cache benchmark data (GL_DLA_CACHE_BENCHMARKS)
        eviction_policy: Cache eviction policy (GL_DLA_CACHE_EVICTION_POLICY)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl=3600,
        ...     max_size=10000,
        ...     key_prefix="gl_dla:"
        ... )
        >>> cache.ttl
        3600
    """

    enabled: bool = True
    ttl: int = 3600
    max_size: int = 10000
    key_prefix: str = "gl_dla:"
    cache_ef_lookups: bool = True
    cache_calculations: bool = True
    cache_benchmarks: bool = True
    eviction_policy: str = "LRU"

    def validate(self) -> None:
        """
        Validate cache configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.ttl < 0 or self.ttl > 86400:
            raise ValueError("ttl must be between 0 and 86400 (24 hours)")

        if self.max_size < 1 or self.max_size > 1000000:
            raise ValueError("max_size must be between 1 and 1000000")

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
            "ttl": self.ttl,
            "max_size": self.max_size,
            "key_prefix": self.key_prefix,
            "cache_ef_lookups": self.cache_ef_lookups,
            "cache_calculations": self.cache_calculations,
            "cache_benchmarks": self.cache_benchmarks,
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
            enabled=os.getenv("GL_DLA_CACHE_ENABLED", "true").lower() == "true",
            ttl=int(os.getenv("GL_DLA_CACHE_TTL", "3600")),
            max_size=int(os.getenv("GL_DLA_CACHE_MAX_SIZE", "10000")),
            key_prefix=os.getenv("GL_DLA_CACHE_KEY_PREFIX", "gl_dla:"),
            cache_ef_lookups=os.getenv(
                "GL_DLA_CACHE_EF_LOOKUPS", "true"
            ).lower() == "true",
            cache_calculations=os.getenv(
                "GL_DLA_CACHE_CALCULATIONS", "true"
            ).lower() == "true",
            cache_benchmarks=os.getenv(
                "GL_DLA_CACHE_BENCHMARKS", "true"
            ).lower() == "true",
            eviction_policy=os.getenv("GL_DLA_CACHE_EVICTION_POLICY", "LRU"),
        )


# =============================================================================
# SECTION 17: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Downstream Leased Assets agent.

    Configures Prometheus-compatible metrics collection including histogram
    buckets for latency tracking.

    Attributes:
        enabled: Enable metrics collection (GL_DLA_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_DLA_METRICS_PREFIX)
        include_tenant: Include tenant label in metrics (GL_DLA_METRICS_INCLUDE_TENANT)
        histogram_buckets: Comma-separated histogram bucket boundaries (GL_DLA_METRICS_HISTOGRAM_BUCKETS)
        enable_latency_tracking: Track per-engine latency (GL_DLA_METRICS_ENABLE_LATENCY_TRACKING)
        enable_error_counting: Track error counts by type (GL_DLA_METRICS_ENABLE_ERROR_COUNTING)
        enable_throughput_tracking: Track records per second (GL_DLA_METRICS_ENABLE_THROUGHPUT_TRACKING)

    Example:
        >>> metrics = MetricsConfig(
        ...     prefix="gl_dla_",
        ...     histogram_buckets="0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_histogram_buckets()
        (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    """

    enabled: bool = True
    prefix: str = "gl_dla_"
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

        try:
            buckets = self.get_histogram_buckets()
            if not buckets:
                raise ValueError("At least one histogram bucket must be defined")
            for bucket in buckets:
                if bucket <= 0:
                    raise ValueError("All histogram buckets must be positive")
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
            enabled=os.getenv("GL_DLA_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_DLA_METRICS_PREFIX", "gl_dla_"),
            include_tenant=os.getenv(
                "GL_DLA_METRICS_INCLUDE_TENANT", "true"
            ).lower() == "true",
            histogram_buckets=os.getenv(
                "GL_DLA_METRICS_HISTOGRAM_BUCKETS",
                "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
            enable_latency_tracking=os.getenv(
                "GL_DLA_METRICS_ENABLE_LATENCY_TRACKING", "true"
            ).lower() == "true",
            enable_error_counting=os.getenv(
                "GL_DLA_METRICS_ENABLE_ERROR_COUNTING", "true"
            ).lower() == "true",
            enable_throughput_tracking=os.getenv(
                "GL_DLA_METRICS_ENABLE_THROUGHPUT_TRACKING", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 18: SECURITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SecurityConfig:
    """
    Security configuration for Downstream Leased Assets agent.

    Configures tenant isolation, audit logging, and rate limiting.

    Attributes:
        tenant_isolation: Enable tenant isolation (GL_DLA_SECURITY_TENANT_ISOLATION)
        audit_enabled: Enable audit logging (GL_DLA_SECURITY_AUDIT_ENABLED)
        rate_limiting: Enable rate limiting (GL_DLA_SECURITY_RATE_LIMITING)
        rate_limit_rpm: Rate limit requests per minute (GL_DLA_SECURITY_RATE_LIMIT_RPM)
        max_concurrent_requests: Max concurrent requests (GL_DLA_SECURITY_MAX_CONCURRENT)
        input_sanitization: Enable input sanitization (GL_DLA_SECURITY_INPUT_SANITIZATION)

    Example:
        >>> sec = SecurityConfig(
        ...     tenant_isolation=True,
        ...     audit_enabled=True,
        ...     rate_limiting=True
        ... )
        >>> sec.tenant_isolation
        True
    """

    tenant_isolation: bool = True
    audit_enabled: bool = True
    rate_limiting: bool = True
    rate_limit_rpm: int = 100
    max_concurrent_requests: int = 10
    input_sanitization: bool = True

    def validate(self) -> None:
        """
        Validate security configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.rate_limit_rpm < 1 or self.rate_limit_rpm > 10000:
            raise ValueError("rate_limit_rpm must be between 1 and 10000")

        if self.max_concurrent_requests < 1 or self.max_concurrent_requests > 1000:
            raise ValueError("max_concurrent_requests must be between 1 and 1000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_isolation": self.tenant_isolation,
            "audit_enabled": self.audit_enabled,
            "rate_limiting": self.rate_limiting,
            "rate_limit_rpm": self.rate_limit_rpm,
            "max_concurrent_requests": self.max_concurrent_requests,
            "input_sanitization": self.input_sanitization,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load from environment variables."""
        return cls(
            tenant_isolation=os.getenv(
                "GL_DLA_SECURITY_TENANT_ISOLATION", "true"
            ).lower() == "true",
            audit_enabled=os.getenv(
                "GL_DLA_SECURITY_AUDIT_ENABLED", "true"
            ).lower() == "true",
            rate_limiting=os.getenv(
                "GL_DLA_SECURITY_RATE_LIMITING", "true"
            ).lower() == "true",
            rate_limit_rpm=int(os.getenv("GL_DLA_SECURITY_RATE_LIMIT_RPM", "100")),
            max_concurrent_requests=int(
                os.getenv("GL_DLA_SECURITY_MAX_CONCURRENT", "10")
            ),
            input_sanitization=os.getenv(
                "GL_DLA_SECURITY_INPUT_SANITIZATION", "true"
            ).lower() == "true",
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass(frozen=True)
class DownstreamLeasedAssetsConfig:
    """
    Master configuration class for Downstream Leased Assets agent (AGENT-MRV-026).

    This frozen dataclass aggregates all 18 configuration sections and provides
    a unified interface for accessing configuration values. It implements the
    singleton pattern with thread-safe access.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        building: Building asset configuration
        vehicle: Vehicle asset configuration
        equipment: Equipment asset configuration
        it_asset: IT asset configuration
        average_data: Average-data calculation configuration
        spend_based: Spend-based calculation configuration
        hybrid: Hybrid calculation configuration
        allocation: Allocation method configuration
        compliance: Regulatory compliance configuration
        provenance: Data provenance configuration
        uncertainty: Uncertainty quantification configuration
        dqi: Data Quality Indicator configuration
        pipeline: Pipeline processing configuration
        cache: Cache layer configuration
        metrics: Prometheus metrics configuration
        security: Security configuration

    Example:
        >>> config = DownstreamLeasedAssetsConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-013'
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    equipment: EquipmentConfig = field(default_factory=EquipmentConfig)
    it_asset: ITAssetConfig = field(default_factory=ITAssetConfig)
    average_data: AverageDataConfig = field(default_factory=AverageDataConfig)
    spend_based: SpendBasedConfig = field(default_factory=SpendBasedConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    dqi: DQIConfig = field(default_factory=DQIConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def validate_all(self) -> List[str]:
        """
        Validate all configuration sections and return list of errors.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors: List[str] = []

        sections = [
            ("general", self.general),
            ("database", self.database),
            ("building", self.building),
            ("vehicle", self.vehicle),
            ("equipment", self.equipment),
            ("it_asset", self.it_asset),
            ("average_data", self.average_data),
            ("spend_based", self.spend_based),
            ("hybrid", self.hybrid),
            ("allocation", self.allocation),
            ("compliance", self.compliance),
            ("provenance", self.provenance),
            ("uncertainty", self.uncertainty),
            ("dqi", self.dqi),
            ("pipeline", self.pipeline),
            ("cache", self.cache),
            ("metrics", self.metrics),
            ("security", self.security),
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

        # Ensure database pool_min <= pool_max
        if self.database.pool_min > self.database.pool_max:
            errors.append(
                "cross-validation: database.pool_min must be <= database.pool_max"
            )

        # Ensure pool_size is within pool_min and pool_max range
        if self.database.pool_size > self.database.pool_max:
            errors.append(
                "cross-validation: database.pool_size should not exceed database.pool_max"
            )

        # Ensure pipeline batch_size does not exceed general max_batch_size
        if self.pipeline.batch_size > self.general.max_batch_size:
            errors.append(
                "cross-validation: pipeline.batch_size should not exceed "
                "general.max_batch_size"
            )

        # Ensure DQI min_score matches compliance min_data_quality_score
        if self.dqi.min_score != self.compliance.min_data_quality_score:
            logger.warning(
                "dqi.min_score '%s' differs from compliance.min_data_quality_score '%s'",
                self.dqi.min_score,
                self.compliance.min_data_quality_score,
            )

        # Ensure vacancy tracking consistency
        if self.allocation.vacancy_adjustment and not self.building.vacancy_tracking:
            errors.append(
                "cross-validation: allocation.vacancy_adjustment requires "
                "building.vacancy_tracking to be enabled"
            )

        # Ensure common area allocation consistency
        if self.allocation.common_area_included and not self.building.common_area_allocation:
            errors.append(
                "cross-validation: allocation.common_area_included requires "
                "building.common_area_allocation to be enabled"
            )

        # Ensure cache key_prefix uses expected naming
        if self.cache.enabled and not self.cache.key_prefix.startswith("gl_dla"):
            logger.warning(
                "cache.key_prefix '%s' does not start with 'gl_dla' - "
                "this may cause key namespace confusion",
                self.cache.key_prefix,
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
            "building": self.building.to_dict(),
            "vehicle": self.vehicle.to_dict(),
            "equipment": self.equipment.to_dict(),
            "it_asset": self.it_asset.to_dict(),
            "average_data": self.average_data.to_dict(),
            "spend_based": self.spend_based.to_dict(),
            "hybrid": self.hybrid.to_dict(),
            "allocation": self.allocation.to_dict(),
            "compliance": self.compliance.to_dict(),
            "provenance": self.provenance.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "dqi": self.dqi.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "cache": self.cache.to_dict(),
            "metrics": self.metrics.to_dict(),
            "security": self.security.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownstreamLeasedAssetsConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            DownstreamLeasedAssetsConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            building=BuildingConfig.from_dict(data.get("building", {})),
            vehicle=VehicleConfig.from_dict(data.get("vehicle", {})),
            equipment=EquipmentConfig.from_dict(data.get("equipment", {})),
            it_asset=ITAssetConfig.from_dict(data.get("it_asset", {})),
            average_data=AverageDataConfig.from_dict(data.get("average_data", {})),
            spend_based=SpendBasedConfig.from_dict(data.get("spend_based", {})),
            hybrid=HybridConfig.from_dict(data.get("hybrid", {})),
            allocation=AllocationConfig.from_dict(data.get("allocation", {})),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})),
            dqi=DQIConfig.from_dict(data.get("dqi", {})),
            pipeline=PipelineConfig.from_dict(data.get("pipeline", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
            security=SecurityConfig.from_dict(data.get("security", {})),
        )

    @classmethod
    def from_env(cls) -> "DownstreamLeasedAssetsConfig":
        """
        Load configuration from environment variables.

        Returns:
            DownstreamLeasedAssetsConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            building=BuildingConfig.from_env(),
            vehicle=VehicleConfig.from_env(),
            equipment=EquipmentConfig.from_env(),
            it_asset=ITAssetConfig.from_env(),
            average_data=AverageDataConfig.from_env(),
            spend_based=SpendBasedConfig.from_env(),
            hybrid=HybridConfig.from_env(),
            allocation=AllocationConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            dqi=DQIConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            cache=CacheConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            security=SecurityConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[DownstreamLeasedAssetsConfig] = None
_config_lock = threading.RLock()


def get_config() -> DownstreamLeasedAssetsConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization of the configuration singleton.
    The first call loads configuration from environment variables.
    Subsequent calls return the cached instance.

    Returns:
        DownstreamLeasedAssetsConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-013'
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info("Initializing DownstreamLeasedAssetsConfig from environment")
                config = DownstreamLeasedAssetsConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning("Configuration validation warning: %s", error)
                _config_instance = config
                logger.info(
                    "DownstreamLeasedAssetsConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: DownstreamLeasedAssetsConfig) -> None:
    """
    Set the singleton configuration instance.

    Allows manual configuration, primarily useful for testing.

    Args:
        config: DownstreamLeasedAssetsConfig instance to set as singleton

    Raises:
        TypeError: If config is not a DownstreamLeasedAssetsConfig instance
    """
    global _config_instance

    if not isinstance(config, DownstreamLeasedAssetsConfig):
        raise TypeError(
            f"config must be a DownstreamLeasedAssetsConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("DownstreamLeasedAssetsConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Clears the cached singleton, forcing the next call to get_config()
    to reload from environment variables.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("DownstreamLeasedAssetsConfig singleton reset")


def validate_config(config: DownstreamLeasedAssetsConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate_all()
