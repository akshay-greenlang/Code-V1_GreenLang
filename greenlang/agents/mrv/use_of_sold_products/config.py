# -*- coding: utf-8 -*-
"""
Use of Sold Products Configuration - AGENT-MRV-024

Thread-safe singleton configuration for GL-MRV-S3-011.
All environment variables prefixed with GL_USP_.

This module provides comprehensive configuration management for the Use of
Sold Products agent (GHG Protocol Scope 3 Category 11), supporting:
- Direct use-phase emissions (fuel-consuming, refrigerant-containing, chemical products)
- Indirect use-phase emissions (electricity-consuming, heating, steam products)
- Fuels and feedstocks sold for combustion or processing
- Product lifetime modeling (survival curves, degradation)
- Grid emission factor management (IEA, eGRID, regional)
- Fuel emission factor management (DEFRA, EPA, IPCC)
- Refrigerant GWP management (AR4, AR5, AR6)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi, GRI, SEC)
- Uncertainty quantification (Monte Carlo, IPCC default, analytical)
- Data quality indicator (DQI) scoring
- Provenance tracking and audit trails
- Double-counting prevention (vs Scope 1, Cat 1, Cat 12)

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-011'
    >>> config.direct_emissions.enable_fuel
    True
    >>> config.compliance.strict_mode
    False

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_USP_ prefix. See individual config sections for specific variables.
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
    General configuration for Use of Sold Products agent.

    Attributes:
        enabled: Master switch for the agent (GL_USP_ENABLED)
        debug: Enable debug mode with verbose logging (GL_USP_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_USP_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_USP_AGENT_ID)
        agent_component: Agent component identifier (GL_USP_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_USP_VERSION)
        api_prefix: API route prefix (GL_USP_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_USP_MAX_BATCH_SIZE)
        table_prefix: Database table prefix (GL_USP_TABLE_PREFIX)
        default_gwp: Default GWP assessment report version (GL_USP_DEFAULT_GWP)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-011",
        ...     agent_component="AGENT-MRV-024",
        ...     version="1.0.0",
        ...     api_prefix="/api/v1/use-of-sold-products",
        ...     max_batch_size=1000,
        ...     table_prefix="gl_usp_",
        ...     default_gwp="AR6"
        ... )
        >>> general.agent_id
        'GL-MRV-S3-011'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-011"
    agent_component: str = "AGENT-MRV-024"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/use-of-sold-products"
    max_batch_size: int = 1000
    table_prefix: str = "gl_usp_"
    default_gwp: str = "AR6"

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

        if not self.table_prefix:
            raise ValueError("table_prefix cannot be empty")

        if not self.table_prefix.endswith("_"):
            raise ValueError("table_prefix must end with '_'")

        valid_gwp_versions = {"AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp_versions:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. "
                f"Must be one of {valid_gwp_versions}"
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
            "table_prefix": self.table_prefix,
            "default_gwp": self.default_gwp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_USP_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_USP_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_USP_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_USP_AGENT_ID", "GL-MRV-S3-011"),
            agent_component=os.getenv("GL_USP_AGENT_COMPONENT", "AGENT-MRV-024"),
            version=os.getenv("GL_USP_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_USP_API_PREFIX", "/api/v1/use-of-sold-products"),
            max_batch_size=int(os.getenv("GL_USP_MAX_BATCH_SIZE", "1000")),
            table_prefix=os.getenv("GL_USP_TABLE_PREFIX", "gl_usp_"),
            default_gwp=os.getenv("GL_USP_DEFAULT_GWP", "AR6"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Use of Sold Products agent.

    Attributes:
        host: PostgreSQL host (GL_USP_DB_HOST)
        port: PostgreSQL port (GL_USP_DB_PORT)
        database: Database name (GL_USP_DB_DATABASE)
        username: Database username (GL_USP_DB_USERNAME)
        password: Database password (GL_USP_DB_PASSWORD)
        schema: Database schema name (GL_USP_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_USP_DB_TABLE_PREFIX)
        pool_min: Minimum connection pool size (GL_USP_DB_POOL_MIN)
        pool_max: Maximum connection pool size (GL_USP_DB_POOL_MAX)
        ssl_mode: SSL connection mode (GL_USP_DB_SSL_MODE)
        connection_timeout: Connection timeout in seconds (GL_USP_DB_CONNECTION_TIMEOUT)

    Example:
        >>> db = DatabaseConfig(
        ...     host="localhost",
        ...     port=5432,
        ...     database="greenlang",
        ...     username="greenlang",
        ...     password="secret",
        ...     schema="use_of_sold_products_service",
        ...     table_prefix="gl_usp_",
        ...     pool_min=2,
        ...     pool_max=10,
        ...     ssl_mode="prefer",
        ...     connection_timeout=30
        ... )
        >>> db.table_prefix
        'gl_usp_'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "use_of_sold_products_service"
    table_prefix: str = "gl_usp_"
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
            host=os.getenv("GL_USP_DB_HOST", "localhost"),
            port=int(os.getenv("GL_USP_DB_PORT", "5432")),
            database=os.getenv("GL_USP_DB_DATABASE", "greenlang"),
            username=os.getenv("GL_USP_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_USP_DB_PASSWORD", ""),
            schema=os.getenv("GL_USP_DB_SCHEMA", "use_of_sold_products_service"),
            table_prefix=os.getenv("GL_USP_DB_TABLE_PREFIX", "gl_usp_"),
            pool_min=int(os.getenv("GL_USP_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_USP_DB_POOL_MAX", "10")),
            ssl_mode=os.getenv("GL_USP_DB_SSL_MODE", "prefer"),
            connection_timeout=int(os.getenv("GL_USP_DB_CONNECTION_TIMEOUT", "30")),
        )


# =============================================================================
# SECTION 3: DIRECT EMISSIONS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DirectEmissionsConfig:
    """
    Direct use-phase emissions configuration for Use of Sold Products agent.

    Direct emissions arise from products that directly emit GHGs during use,
    including fuel-consuming products (vehicles, generators), refrigerant-
    containing products (HVAC, refrigerators), and chemical products
    (solvents, aerosols, fire suppressants).

    Attributes:
        enable_fuel: Enable fuel-consuming product calculations (GL_USP_DIRECT_ENABLE_FUEL)
        enable_refrigerant: Enable refrigerant-containing product calculations (GL_USP_DIRECT_ENABLE_REFRIGERANT)
        enable_chemical: Enable chemical product calculations (GL_USP_DIRECT_ENABLE_CHEMICAL)
        default_gwp_standard: Default GWP assessment report (GL_USP_DIRECT_DEFAULT_GWP_STANDARD)
        default_oxidation_factor: Default fuel oxidation factor (GL_USP_DIRECT_DEFAULT_OXIDATION_FACTOR)
        include_biogenic_co2: Include biogenic CO2 in calculations (GL_USP_DIRECT_INCLUDE_BIOGENIC_CO2)
        enable_abatement: Enable abatement/control technology adjustments (GL_USP_DIRECT_ENABLE_ABATEMENT)
        default_leak_rate: Default annual refrigerant leak rate (GL_USP_DIRECT_DEFAULT_LEAK_RATE)
        default_eol_recovery: Default end-of-life refrigerant recovery rate (GL_USP_DIRECT_DEFAULT_EOL_RECOVERY)

    Example:
        >>> direct = DirectEmissionsConfig(
        ...     enable_fuel=True,
        ...     enable_refrigerant=True,
        ...     enable_chemical=True,
        ...     default_gwp_standard="AR6",
        ...     default_oxidation_factor=Decimal("0.99"),
        ...     include_biogenic_co2=False,
        ...     enable_abatement=True,
        ...     default_leak_rate=Decimal("0.02"),
        ...     default_eol_recovery=Decimal("0.70")
        ... )
        >>> direct.default_gwp_standard
        'AR6'
    """

    enable_fuel: bool = True
    enable_refrigerant: bool = True
    enable_chemical: bool = True
    default_gwp_standard: str = "AR6"
    default_oxidation_factor: Decimal = Decimal("0.99")
    include_biogenic_co2: bool = False
    enable_abatement: bool = True
    default_leak_rate: Decimal = Decimal("0.02")
    default_eol_recovery: Decimal = Decimal("0.70")

    def validate(self) -> None:
        """
        Validate direct emissions configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_gwp_standards = {"AR4", "AR5", "AR6"}
        if self.default_gwp_standard not in valid_gwp_standards:
            raise ValueError(
                f"Invalid default_gwp_standard '{self.default_gwp_standard}'. "
                f"Must be one of {valid_gwp_standards}"
            )

        if self.default_oxidation_factor < Decimal("0") or self.default_oxidation_factor > Decimal("1"):
            raise ValueError("default_oxidation_factor must be between 0 and 1")

        if self.default_leak_rate < Decimal("0") or self.default_leak_rate > Decimal("1"):
            raise ValueError("default_leak_rate must be between 0 and 1")

        if self.default_eol_recovery < Decimal("0") or self.default_eol_recovery > Decimal("1"):
            raise ValueError("default_eol_recovery must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_fuel": self.enable_fuel,
            "enable_refrigerant": self.enable_refrigerant,
            "enable_chemical": self.enable_chemical,
            "default_gwp_standard": self.default_gwp_standard,
            "default_oxidation_factor": str(self.default_oxidation_factor),
            "include_biogenic_co2": self.include_biogenic_co2,
            "enable_abatement": self.enable_abatement,
            "default_leak_rate": str(self.default_leak_rate),
            "default_eol_recovery": str(self.default_eol_recovery),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirectEmissionsConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_oxidation_factor", "default_leak_rate", "default_eol_recovery"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "DirectEmissionsConfig":
        """Load from environment variables."""
        return cls(
            enable_fuel=os.getenv(
                "GL_USP_DIRECT_ENABLE_FUEL", "true"
            ).lower() == "true",
            enable_refrigerant=os.getenv(
                "GL_USP_DIRECT_ENABLE_REFRIGERANT", "true"
            ).lower() == "true",
            enable_chemical=os.getenv(
                "GL_USP_DIRECT_ENABLE_CHEMICAL", "true"
            ).lower() == "true",
            default_gwp_standard=os.getenv(
                "GL_USP_DIRECT_DEFAULT_GWP_STANDARD", "AR6"
            ),
            default_oxidation_factor=Decimal(
                os.getenv("GL_USP_DIRECT_DEFAULT_OXIDATION_FACTOR", "0.99")
            ),
            include_biogenic_co2=os.getenv(
                "GL_USP_DIRECT_INCLUDE_BIOGENIC_CO2", "false"
            ).lower() == "true",
            enable_abatement=os.getenv(
                "GL_USP_DIRECT_ENABLE_ABATEMENT", "true"
            ).lower() == "true",
            default_leak_rate=Decimal(
                os.getenv("GL_USP_DIRECT_DEFAULT_LEAK_RATE", "0.02")
            ),
            default_eol_recovery=Decimal(
                os.getenv("GL_USP_DIRECT_DEFAULT_EOL_RECOVERY", "0.70")
            ),
        )


# =============================================================================
# SECTION 4: INDIRECT EMISSIONS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class IndirectEmissionsConfig:
    """
    Indirect use-phase emissions configuration for Use of Sold Products agent.

    Indirect emissions arise from products that consume electricity, heating,
    or steam during their use phase. Emissions are attributed based on the
    energy consumed multiplied by grid or fuel emission factors.

    Attributes:
        enable_electricity: Enable electricity-consuming product calculations (GL_USP_INDIRECT_ENABLE_ELECTRICITY)
        enable_heating: Enable heating-consuming product calculations (GL_USP_INDIRECT_ENABLE_HEATING)
        enable_steam: Enable steam-consuming product calculations (GL_USP_INDIRECT_ENABLE_STEAM)
        default_grid_region: Default grid region for electricity EF (GL_USP_INDIRECT_DEFAULT_GRID_REGION)
        default_electricity_ef_source: Default electricity EF source (GL_USP_INDIRECT_DEFAULT_ELEC_EF_SOURCE)
        include_td_losses: Include transmission and distribution losses (GL_USP_INDIRECT_INCLUDE_TD_LOSSES)
        default_td_loss_factor: Default T&D loss factor (GL_USP_INDIRECT_DEFAULT_TD_LOSS_FACTOR)
        default_heating_fuel: Default heating fuel type (GL_USP_INDIRECT_DEFAULT_HEATING_FUEL)
        default_steam_efficiency: Default steam system efficiency (GL_USP_INDIRECT_DEFAULT_STEAM_EFFICIENCY)

    Example:
        >>> indirect = IndirectEmissionsConfig(
        ...     enable_electricity=True,
        ...     enable_heating=True,
        ...     enable_steam=True,
        ...     default_grid_region="US_AVERAGE",
        ...     default_electricity_ef_source="IEA",
        ...     include_td_losses=True,
        ...     default_td_loss_factor=Decimal("0.05"),
        ...     default_heating_fuel="NATURAL_GAS",
        ...     default_steam_efficiency=Decimal("0.80")
        ... )
        >>> indirect.default_grid_region
        'US_AVERAGE'
    """

    enable_electricity: bool = True
    enable_heating: bool = True
    enable_steam: bool = True
    default_grid_region: str = "US_AVERAGE"
    default_electricity_ef_source: str = "IEA"
    include_td_losses: bool = True
    default_td_loss_factor: Decimal = Decimal("0.05")
    default_heating_fuel: str = "NATURAL_GAS"
    default_steam_efficiency: Decimal = Decimal("0.80")

    def validate(self) -> None:
        """
        Validate indirect emissions configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.default_grid_region:
            raise ValueError("default_grid_region cannot be empty")

        valid_ef_sources = {"IEA", "EGRID", "EEA", "DEFRA", "EPA", "CUSTOM"}
        if self.default_electricity_ef_source not in valid_ef_sources:
            raise ValueError(
                f"Invalid default_electricity_ef_source '{self.default_electricity_ef_source}'. "
                f"Must be one of {valid_ef_sources}"
            )

        if self.default_td_loss_factor < Decimal("0") or self.default_td_loss_factor > Decimal("0.5"):
            raise ValueError("default_td_loss_factor must be between 0 and 0.5")

        valid_heating_fuels = {
            "NATURAL_GAS", "OIL", "COAL", "BIOMASS", "ELECTRIC",
            "DISTRICT_HEATING", "LPG", "UNKNOWN",
        }
        if self.default_heating_fuel not in valid_heating_fuels:
            raise ValueError(
                f"Invalid default_heating_fuel '{self.default_heating_fuel}'. "
                f"Must be one of {valid_heating_fuels}"
            )

        if self.default_steam_efficiency < Decimal("0.1") or self.default_steam_efficiency > Decimal("1.0"):
            raise ValueError("default_steam_efficiency must be between 0.1 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_electricity": self.enable_electricity,
            "enable_heating": self.enable_heating,
            "enable_steam": self.enable_steam,
            "default_grid_region": self.default_grid_region,
            "default_electricity_ef_source": self.default_electricity_ef_source,
            "include_td_losses": self.include_td_losses,
            "default_td_loss_factor": str(self.default_td_loss_factor),
            "default_heating_fuel": self.default_heating_fuel,
            "default_steam_efficiency": str(self.default_steam_efficiency),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndirectEmissionsConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_td_loss_factor", "default_steam_efficiency"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "IndirectEmissionsConfig":
        """Load from environment variables."""
        return cls(
            enable_electricity=os.getenv(
                "GL_USP_INDIRECT_ENABLE_ELECTRICITY", "true"
            ).lower() == "true",
            enable_heating=os.getenv(
                "GL_USP_INDIRECT_ENABLE_HEATING", "true"
            ).lower() == "true",
            enable_steam=os.getenv(
                "GL_USP_INDIRECT_ENABLE_STEAM", "true"
            ).lower() == "true",
            default_grid_region=os.getenv(
                "GL_USP_INDIRECT_DEFAULT_GRID_REGION", "US_AVERAGE"
            ),
            default_electricity_ef_source=os.getenv(
                "GL_USP_INDIRECT_DEFAULT_ELEC_EF_SOURCE", "IEA"
            ),
            include_td_losses=os.getenv(
                "GL_USP_INDIRECT_INCLUDE_TD_LOSSES", "true"
            ).lower() == "true",
            default_td_loss_factor=Decimal(
                os.getenv("GL_USP_INDIRECT_DEFAULT_TD_LOSS_FACTOR", "0.05")
            ),
            default_heating_fuel=os.getenv(
                "GL_USP_INDIRECT_DEFAULT_HEATING_FUEL", "NATURAL_GAS"
            ),
            default_steam_efficiency=Decimal(
                os.getenv("GL_USP_INDIRECT_DEFAULT_STEAM_EFFICIENCY", "0.80")
            ),
        )


# =============================================================================
# SECTION 5: FUELS AND FEEDSTOCKS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class FuelsAndFeedstocksConfig:
    """
    Fuels and feedstocks sold configuration for Use of Sold Products agent.

    Covers products sold as fuels (gasoline, diesel, natural gas, coal) or
    feedstocks that are combusted or processed by end users. This is the
    dominant emission source for oil and gas companies.

    Attributes:
        enable_fuel_sales: Enable fuel sales calculations (GL_USP_FUELS_ENABLE_FUEL_SALES)
        enable_feedstocks: Enable feedstock calculations (GL_USP_FUELS_ENABLE_FEEDSTOCKS)
        default_oxidation_factor: Default fuel oxidation factor (GL_USP_FUELS_DEFAULT_OXIDATION_FACTOR)
        include_biogenic: Include biogenic CO2 in fuel calculations (GL_USP_FUELS_INCLUDE_BIOGENIC)
        upstream_included: Include upstream (WTT) in fuel calculations (GL_USP_FUELS_UPSTREAM_INCLUDED)
        default_fuel_ef_source: Default fuel emission factor source (GL_USP_FUELS_DEFAULT_EF_SOURCE)
        default_density_unit: Default fuel density unit (GL_USP_FUELS_DEFAULT_DENSITY_UNIT)
        enable_blend_tracking: Enable biofuel blend ratio tracking (GL_USP_FUELS_ENABLE_BLEND_TRACKING)
        default_blend_ratio: Default biofuel blend ratio (GL_USP_FUELS_DEFAULT_BLEND_RATIO)

    Example:
        >>> fuels = FuelsAndFeedstocksConfig(
        ...     enable_fuel_sales=True,
        ...     enable_feedstocks=True,
        ...     default_oxidation_factor=Decimal("0.99"),
        ...     include_biogenic=False,
        ...     upstream_included=False,
        ...     default_fuel_ef_source="IPCC",
        ...     default_density_unit="kg_per_litre",
        ...     enable_blend_tracking=True,
        ...     default_blend_ratio=Decimal("0.0")
        ... )
        >>> fuels.default_oxidation_factor
        Decimal('0.99')
    """

    enable_fuel_sales: bool = True
    enable_feedstocks: bool = True
    default_oxidation_factor: Decimal = Decimal("0.99")
    include_biogenic: bool = False
    upstream_included: bool = False
    default_fuel_ef_source: str = "IPCC"
    default_density_unit: str = "kg_per_litre"
    enable_blend_tracking: bool = True
    default_blend_ratio: Decimal = Decimal("0.0")

    def validate(self) -> None:
        """
        Validate fuels and feedstocks configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_oxidation_factor < Decimal("0") or self.default_oxidation_factor > Decimal("1"):
            raise ValueError("default_oxidation_factor must be between 0 and 1")

        valid_ef_sources = {"IPCC", "DEFRA", "EPA", "IEA", "CUSTOM"}
        if self.default_fuel_ef_source not in valid_ef_sources:
            raise ValueError(
                f"Invalid default_fuel_ef_source '{self.default_fuel_ef_source}'. "
                f"Must be one of {valid_ef_sources}"
            )

        valid_density_units = {"kg_per_litre", "kg_per_gallon", "kg_per_m3"}
        if self.default_density_unit not in valid_density_units:
            raise ValueError(
                f"Invalid default_density_unit '{self.default_density_unit}'. "
                f"Must be one of {valid_density_units}"
            )

        if self.default_blend_ratio < Decimal("0") or self.default_blend_ratio > Decimal("1"):
            raise ValueError("default_blend_ratio must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_fuel_sales": self.enable_fuel_sales,
            "enable_feedstocks": self.enable_feedstocks,
            "default_oxidation_factor": str(self.default_oxidation_factor),
            "include_biogenic": self.include_biogenic,
            "upstream_included": self.upstream_included,
            "default_fuel_ef_source": self.default_fuel_ef_source,
            "default_density_unit": self.default_density_unit,
            "enable_blend_tracking": self.enable_blend_tracking,
            "default_blend_ratio": str(self.default_blend_ratio),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FuelsAndFeedstocksConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_oxidation_factor", "default_blend_ratio"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "FuelsAndFeedstocksConfig":
        """Load from environment variables."""
        return cls(
            enable_fuel_sales=os.getenv(
                "GL_USP_FUELS_ENABLE_FUEL_SALES", "true"
            ).lower() == "true",
            enable_feedstocks=os.getenv(
                "GL_USP_FUELS_ENABLE_FEEDSTOCKS", "true"
            ).lower() == "true",
            default_oxidation_factor=Decimal(
                os.getenv("GL_USP_FUELS_DEFAULT_OXIDATION_FACTOR", "0.99")
            ),
            include_biogenic=os.getenv(
                "GL_USP_FUELS_INCLUDE_BIOGENIC", "false"
            ).lower() == "true",
            upstream_included=os.getenv(
                "GL_USP_FUELS_UPSTREAM_INCLUDED", "false"
            ).lower() == "true",
            default_fuel_ef_source=os.getenv(
                "GL_USP_FUELS_DEFAULT_EF_SOURCE", "IPCC"
            ),
            default_density_unit=os.getenv(
                "GL_USP_FUELS_DEFAULT_DENSITY_UNIT", "kg_per_litre"
            ),
            enable_blend_tracking=os.getenv(
                "GL_USP_FUELS_ENABLE_BLEND_TRACKING", "true"
            ).lower() == "true",
            default_blend_ratio=Decimal(
                os.getenv("GL_USP_FUELS_DEFAULT_BLEND_RATIO", "0.0")
            ),
        )


# =============================================================================
# SECTION 6: LIFETIME MODELING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class LifetimeModelingConfig:
    """
    Product lifetime modeling configuration for Use of Sold Products agent.

    GHG Protocol requires companies to estimate the total lifetime emissions
    of products sold. This section configures the default lifetime assumption,
    survival curve modeling, and efficiency degradation over product life.

    Attributes:
        default_adjustment: Default lifetime adjustment method (GL_USP_LIFETIME_DEFAULT_ADJUSTMENT)
        enable_degradation: Enable efficiency degradation over lifetime (GL_USP_LIFETIME_ENABLE_DEGRADATION)
        enable_survival_curves: Enable Weibull/exponential survival curves (GL_USP_LIFETIME_ENABLE_SURVIVAL_CURVES)
        default_lifetime_years: Default product lifetime in years (GL_USP_LIFETIME_DEFAULT_YEARS)
        max_lifetime_years: Maximum allowable product lifetime (GL_USP_LIFETIME_MAX_YEARS)
        default_survival_model: Default survival curve model (GL_USP_LIFETIME_DEFAULT_SURVIVAL_MODEL)
        default_weibull_shape: Default Weibull shape parameter (GL_USP_LIFETIME_DEFAULT_WEIBULL_SHAPE)
        default_weibull_scale: Default Weibull scale parameter (GL_USP_LIFETIME_DEFAULT_WEIBULL_SCALE)
        default_degradation_rate: Default annual degradation rate (GL_USP_LIFETIME_DEFAULT_DEGRADATION_RATE)
        enable_usage_profiles: Enable usage profile adjustments (GL_USP_LIFETIME_ENABLE_USAGE_PROFILES)

    Example:
        >>> lifetime = LifetimeModelingConfig(
        ...     default_adjustment="STANDARD",
        ...     enable_degradation=True,
        ...     enable_survival_curves=True,
        ...     default_lifetime_years=10,
        ...     max_lifetime_years=50,
        ...     default_survival_model="WEIBULL",
        ...     default_weibull_shape=Decimal("2.0"),
        ...     default_weibull_scale=Decimal("12.0"),
        ...     default_degradation_rate=Decimal("0.01"),
        ...     enable_usage_profiles=True
        ... )
        >>> lifetime.default_adjustment
        'STANDARD'
    """

    default_adjustment: str = "STANDARD"
    enable_degradation: bool = True
    enable_survival_curves: bool = True
    default_lifetime_years: int = 10
    max_lifetime_years: int = 50
    default_survival_model: str = "WEIBULL"
    default_weibull_shape: Decimal = Decimal("2.0")
    default_weibull_scale: Decimal = Decimal("12.0")
    default_degradation_rate: Decimal = Decimal("0.01")
    enable_usage_profiles: bool = True

    def validate(self) -> None:
        """
        Validate lifetime modeling configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_adjustments = {"STANDARD", "CONSERVATIVE", "AGGRESSIVE", "CUSTOM"}
        if self.default_adjustment not in valid_adjustments:
            raise ValueError(
                f"Invalid default_adjustment '{self.default_adjustment}'. "
                f"Must be one of {valid_adjustments}"
            )

        if self.default_lifetime_years < 1 or self.default_lifetime_years > 100:
            raise ValueError("default_lifetime_years must be between 1 and 100")

        if self.max_lifetime_years < 1 or self.max_lifetime_years > 200:
            raise ValueError("max_lifetime_years must be between 1 and 200")

        if self.default_lifetime_years > self.max_lifetime_years:
            raise ValueError(
                "default_lifetime_years must be <= max_lifetime_years"
            )

        valid_survival_models = {"WEIBULL", "EXPONENTIAL", "LINEAR", "STEP", "CUSTOM"}
        if self.default_survival_model not in valid_survival_models:
            raise ValueError(
                f"Invalid default_survival_model '{self.default_survival_model}'. "
                f"Must be one of {valid_survival_models}"
            )

        if self.default_weibull_shape < Decimal("0.1") or self.default_weibull_shape > Decimal("20"):
            raise ValueError("default_weibull_shape must be between 0.1 and 20")

        if self.default_weibull_scale < Decimal("0.1") or self.default_weibull_scale > Decimal("200"):
            raise ValueError("default_weibull_scale must be between 0.1 and 200")

        if self.default_degradation_rate < Decimal("0") or self.default_degradation_rate > Decimal("0.5"):
            raise ValueError("default_degradation_rate must be between 0 and 0.5")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_adjustment": self.default_adjustment,
            "enable_degradation": self.enable_degradation,
            "enable_survival_curves": self.enable_survival_curves,
            "default_lifetime_years": self.default_lifetime_years,
            "max_lifetime_years": self.max_lifetime_years,
            "default_survival_model": self.default_survival_model,
            "default_weibull_shape": str(self.default_weibull_shape),
            "default_weibull_scale": str(self.default_weibull_scale),
            "default_degradation_rate": str(self.default_degradation_rate),
            "enable_usage_profiles": self.enable_usage_profiles,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifetimeModelingConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_weibull_shape", "default_weibull_scale", "default_degradation_rate"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "LifetimeModelingConfig":
        """Load from environment variables."""
        return cls(
            default_adjustment=os.getenv(
                "GL_USP_LIFETIME_DEFAULT_ADJUSTMENT", "STANDARD"
            ),
            enable_degradation=os.getenv(
                "GL_USP_LIFETIME_ENABLE_DEGRADATION", "true"
            ).lower() == "true",
            enable_survival_curves=os.getenv(
                "GL_USP_LIFETIME_ENABLE_SURVIVAL_CURVES", "true"
            ).lower() == "true",
            default_lifetime_years=int(
                os.getenv("GL_USP_LIFETIME_DEFAULT_YEARS", "10")
            ),
            max_lifetime_years=int(
                os.getenv("GL_USP_LIFETIME_MAX_YEARS", "50")
            ),
            default_survival_model=os.getenv(
                "GL_USP_LIFETIME_DEFAULT_SURVIVAL_MODEL", "WEIBULL"
            ),
            default_weibull_shape=Decimal(
                os.getenv("GL_USP_LIFETIME_DEFAULT_WEIBULL_SHAPE", "2.0")
            ),
            default_weibull_scale=Decimal(
                os.getenv("GL_USP_LIFETIME_DEFAULT_WEIBULL_SCALE", "12.0")
            ),
            default_degradation_rate=Decimal(
                os.getenv("GL_USP_LIFETIME_DEFAULT_DEGRADATION_RATE", "0.01")
            ),
            enable_usage_profiles=os.getenv(
                "GL_USP_LIFETIME_ENABLE_USAGE_PROFILES", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 7: GRID FACTORS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GridFactorsConfig:
    """
    Grid emission factor configuration for Use of Sold Products agent.

    Manages electricity grid emission factors used for indirect use-phase
    emissions of electricity-consuming products. Supports IEA, eGRID, and
    EU EEA country-level factors with configurable fallback.

    Attributes:
        default_region: Default grid region (GL_USP_GRID_DEFAULT_REGION)
        update_frequency: EF update frequency (GL_USP_GRID_UPDATE_FREQUENCY)
        fallback_to_global: Fall back to global average when region not found (GL_USP_GRID_FALLBACK_TO_GLOBAL)
        primary_source: Primary grid EF source (GL_USP_GRID_PRIMARY_SOURCE)
        fallback_source: Fallback grid EF source (GL_USP_GRID_FALLBACK_SOURCE)
        ef_year: Default emission factor year (GL_USP_GRID_EF_YEAR)
        enable_marginal_factors: Enable marginal grid EFs (GL_USP_GRID_ENABLE_MARGINAL)
        cache_ttl_seconds: Grid EF cache TTL in seconds (GL_USP_GRID_CACHE_TTL)

    Example:
        >>> grid = GridFactorsConfig(
        ...     default_region="US_AVERAGE",
        ...     update_frequency="ANNUAL",
        ...     fallback_to_global=True,
        ...     primary_source="IEA",
        ...     fallback_source="DEFRA",
        ...     ef_year=2024,
        ...     enable_marginal_factors=False,
        ...     cache_ttl_seconds=3600
        ... )
        >>> grid.default_region
        'US_AVERAGE'
    """

    default_region: str = "US_AVERAGE"
    update_frequency: str = "ANNUAL"
    fallback_to_global: bool = True
    primary_source: str = "IEA"
    fallback_source: str = "DEFRA"
    ef_year: int = 2024
    enable_marginal_factors: bool = False
    cache_ttl_seconds: int = 3600

    def validate(self) -> None:
        """
        Validate grid factors configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.default_region:
            raise ValueError("default_region cannot be empty")

        valid_frequencies = {"ANNUAL", "QUARTERLY", "MONTHLY", "REAL_TIME"}
        if self.update_frequency not in valid_frequencies:
            raise ValueError(
                f"Invalid update_frequency '{self.update_frequency}'. "
                f"Must be one of {valid_frequencies}"
            )

        valid_sources = {"IEA", "EGRID", "EEA", "DEFRA", "EPA", "CUSTOM"}
        if self.primary_source not in valid_sources:
            raise ValueError(
                f"Invalid primary_source '{self.primary_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.fallback_source not in valid_sources:
            raise ValueError(
                f"Invalid fallback_source '{self.fallback_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.ef_year < 2000 or self.ef_year > 2030:
            raise ValueError("ef_year must be between 2000 and 2030")

        if self.cache_ttl_seconds < 0 or self.cache_ttl_seconds > 86400:
            raise ValueError("cache_ttl_seconds must be between 0 and 86400")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_region": self.default_region,
            "update_frequency": self.update_frequency,
            "fallback_to_global": self.fallback_to_global,
            "primary_source": self.primary_source,
            "fallback_source": self.fallback_source,
            "ef_year": self.ef_year,
            "enable_marginal_factors": self.enable_marginal_factors,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridFactorsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "GridFactorsConfig":
        """Load from environment variables."""
        return cls(
            default_region=os.getenv("GL_USP_GRID_DEFAULT_REGION", "US_AVERAGE"),
            update_frequency=os.getenv("GL_USP_GRID_UPDATE_FREQUENCY", "ANNUAL"),
            fallback_to_global=os.getenv(
                "GL_USP_GRID_FALLBACK_TO_GLOBAL", "true"
            ).lower() == "true",
            primary_source=os.getenv("GL_USP_GRID_PRIMARY_SOURCE", "IEA"),
            fallback_source=os.getenv("GL_USP_GRID_FALLBACK_SOURCE", "DEFRA"),
            ef_year=int(os.getenv("GL_USP_GRID_EF_YEAR", "2024")),
            enable_marginal_factors=os.getenv(
                "GL_USP_GRID_ENABLE_MARGINAL", "false"
            ).lower() == "true",
            cache_ttl_seconds=int(os.getenv("GL_USP_GRID_CACHE_TTL", "3600")),
        )


# =============================================================================
# SECTION 8: FUEL FACTORS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class FuelFactorsConfig:
    """
    Fuel emission factor configuration for Use of Sold Products agent.

    Manages combustion emission factors for fuels sold as products.
    Factors vary by fuel type, carbon content, and calorific value.

    Attributes:
        include_biogenic: Include biogenic CO2 in fuel EFs (GL_USP_FUEL_FACTORS_INCLUDE_BIOGENIC)
        upstream_included: Include upstream (WTT) in fuel EFs (GL_USP_FUEL_FACTORS_UPSTREAM_INCLUDED)
        primary_source: Primary fuel EF source (GL_USP_FUEL_FACTORS_PRIMARY_SOURCE)
        fallback_source: Fallback fuel EF source (GL_USP_FUEL_FACTORS_FALLBACK_SOURCE)
        ef_year: Default emission factor year (GL_USP_FUEL_FACTORS_EF_YEAR)
        default_calorific_basis: Default calorific value basis (GL_USP_FUEL_FACTORS_CALORIFIC_BASIS)
        cache_ttl_seconds: Fuel EF cache TTL in seconds (GL_USP_FUEL_FACTORS_CACHE_TTL)
        validate_ranges: Validate EF values against expected ranges (GL_USP_FUEL_FACTORS_VALIDATE_RANGES)

    Example:
        >>> fuel_factors = FuelFactorsConfig(
        ...     include_biogenic=False,
        ...     upstream_included=False,
        ...     primary_source="IPCC",
        ...     fallback_source="DEFRA",
        ...     ef_year=2024,
        ...     default_calorific_basis="NCV",
        ...     cache_ttl_seconds=3600,
        ...     validate_ranges=True
        ... )
        >>> fuel_factors.default_calorific_basis
        'NCV'
    """

    include_biogenic: bool = False
    upstream_included: bool = False
    primary_source: str = "IPCC"
    fallback_source: str = "DEFRA"
    ef_year: int = 2024
    default_calorific_basis: str = "NCV"
    cache_ttl_seconds: int = 3600
    validate_ranges: bool = True

    def validate(self) -> None:
        """
        Validate fuel factors configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_sources = {"IPCC", "DEFRA", "EPA", "IEA", "CUSTOM"}
        if self.primary_source not in valid_sources:
            raise ValueError(
                f"Invalid primary_source '{self.primary_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.fallback_source not in valid_sources:
            raise ValueError(
                f"Invalid fallback_source '{self.fallback_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.ef_year < 2000 or self.ef_year > 2030:
            raise ValueError("ef_year must be between 2000 and 2030")

        valid_calorific = {"NCV", "GCV", "HHV", "LHV"}
        if self.default_calorific_basis not in valid_calorific:
            raise ValueError(
                f"Invalid default_calorific_basis '{self.default_calorific_basis}'. "
                f"Must be one of {valid_calorific}"
            )

        if self.cache_ttl_seconds < 0 or self.cache_ttl_seconds > 86400:
            raise ValueError("cache_ttl_seconds must be between 0 and 86400")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "include_biogenic": self.include_biogenic,
            "upstream_included": self.upstream_included,
            "primary_source": self.primary_source,
            "fallback_source": self.fallback_source,
            "ef_year": self.ef_year,
            "default_calorific_basis": self.default_calorific_basis,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "validate_ranges": self.validate_ranges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FuelFactorsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "FuelFactorsConfig":
        """Load from environment variables."""
        return cls(
            include_biogenic=os.getenv(
                "GL_USP_FUEL_FACTORS_INCLUDE_BIOGENIC", "false"
            ).lower() == "true",
            upstream_included=os.getenv(
                "GL_USP_FUEL_FACTORS_UPSTREAM_INCLUDED", "false"
            ).lower() == "true",
            primary_source=os.getenv("GL_USP_FUEL_FACTORS_PRIMARY_SOURCE", "IPCC"),
            fallback_source=os.getenv("GL_USP_FUEL_FACTORS_FALLBACK_SOURCE", "DEFRA"),
            ef_year=int(os.getenv("GL_USP_FUEL_FACTORS_EF_YEAR", "2024")),
            default_calorific_basis=os.getenv(
                "GL_USP_FUEL_FACTORS_CALORIFIC_BASIS", "NCV"
            ),
            cache_ttl_seconds=int(os.getenv("GL_USP_FUEL_FACTORS_CACHE_TTL", "3600")),
            validate_ranges=os.getenv(
                "GL_USP_FUEL_FACTORS_VALIDATE_RANGES", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 9: REFRIGERANTS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RefrigerantsConfig:
    """
    Refrigerant configuration for Use of Sold Products agent.

    Manages GWP values for refrigerant-containing products sold.
    Covers HFCs, HCFCs, PFCs, and refrigerant blends with
    support for AR4/AR5/AR6 GWP assessments and end-of-life leakage.

    Attributes:
        default_gwp_standard: Default GWP assessment report (GL_USP_REFRIG_DEFAULT_GWP_STANDARD)
        include_eol_leakage: Include end-of-life leakage (GL_USP_REFRIG_INCLUDE_EOL_LEAKAGE)
        default_annual_leak_rate: Default annual leak rate (GL_USP_REFRIG_DEFAULT_ANNUAL_LEAK_RATE)
        default_eol_leak_rate: Default end-of-life leak rate (GL_USP_REFRIG_DEFAULT_EOL_LEAK_RATE)
        enable_blend_decomposition: Decompose blends into constituent GWPs (GL_USP_REFRIG_ENABLE_BLEND_DECOMPOSITION)
        default_charge_mass_kg: Default refrigerant charge mass (GL_USP_REFRIG_DEFAULT_CHARGE_MASS_KG)
        enable_kigali_phasedown: Apply Kigali amendment phase-down (GL_USP_REFRIG_ENABLE_KIGALI)
        cache_ttl_seconds: Refrigerant GWP cache TTL (GL_USP_REFRIG_CACHE_TTL)

    Example:
        >>> refrig = RefrigerantsConfig(
        ...     default_gwp_standard="AR6",
        ...     include_eol_leakage=True,
        ...     default_annual_leak_rate=Decimal("0.02"),
        ...     default_eol_leak_rate=Decimal("0.15"),
        ...     enable_blend_decomposition=True,
        ...     default_charge_mass_kg=Decimal("1.5"),
        ...     enable_kigali_phasedown=True,
        ...     cache_ttl_seconds=3600
        ... )
        >>> refrig.default_gwp_standard
        'AR6'
    """

    default_gwp_standard: str = "AR6"
    include_eol_leakage: bool = True
    default_annual_leak_rate: Decimal = Decimal("0.02")
    default_eol_leak_rate: Decimal = Decimal("0.15")
    enable_blend_decomposition: bool = True
    default_charge_mass_kg: Decimal = Decimal("1.5")
    enable_kigali_phasedown: bool = True
    cache_ttl_seconds: int = 3600

    def validate(self) -> None:
        """
        Validate refrigerants configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_gwp_standards = {"AR4", "AR5", "AR6"}
        if self.default_gwp_standard not in valid_gwp_standards:
            raise ValueError(
                f"Invalid default_gwp_standard '{self.default_gwp_standard}'. "
                f"Must be one of {valid_gwp_standards}"
            )

        if self.default_annual_leak_rate < Decimal("0") or self.default_annual_leak_rate > Decimal("1"):
            raise ValueError("default_annual_leak_rate must be between 0 and 1")

        if self.default_eol_leak_rate < Decimal("0") or self.default_eol_leak_rate > Decimal("1"):
            raise ValueError("default_eol_leak_rate must be between 0 and 1")

        if self.default_charge_mass_kg < Decimal("0"):
            raise ValueError("default_charge_mass_kg must be >= 0")

        if self.cache_ttl_seconds < 0 or self.cache_ttl_seconds > 86400:
            raise ValueError("cache_ttl_seconds must be between 0 and 86400")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_gwp_standard": self.default_gwp_standard,
            "include_eol_leakage": self.include_eol_leakage,
            "default_annual_leak_rate": str(self.default_annual_leak_rate),
            "default_eol_leak_rate": str(self.default_eol_leak_rate),
            "enable_blend_decomposition": self.enable_blend_decomposition,
            "default_charge_mass_kg": str(self.default_charge_mass_kg),
            "enable_kigali_phasedown": self.enable_kigali_phasedown,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefrigerantsConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_annual_leak_rate", "default_eol_leak_rate", "default_charge_mass_kg"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RefrigerantsConfig":
        """Load from environment variables."""
        return cls(
            default_gwp_standard=os.getenv(
                "GL_USP_REFRIG_DEFAULT_GWP_STANDARD", "AR6"
            ),
            include_eol_leakage=os.getenv(
                "GL_USP_REFRIG_INCLUDE_EOL_LEAKAGE", "true"
            ).lower() == "true",
            default_annual_leak_rate=Decimal(
                os.getenv("GL_USP_REFRIG_DEFAULT_ANNUAL_LEAK_RATE", "0.02")
            ),
            default_eol_leak_rate=Decimal(
                os.getenv("GL_USP_REFRIG_DEFAULT_EOL_LEAK_RATE", "0.15")
            ),
            enable_blend_decomposition=os.getenv(
                "GL_USP_REFRIG_ENABLE_BLEND_DECOMPOSITION", "true"
            ).lower() == "true",
            default_charge_mass_kg=Decimal(
                os.getenv("GL_USP_REFRIG_DEFAULT_CHARGE_MASS_KG", "1.5")
            ),
            enable_kigali_phasedown=os.getenv(
                "GL_USP_REFRIG_ENABLE_KIGALI", "true"
            ).lower() == "true",
            cache_ttl_seconds=int(os.getenv("GL_USP_REFRIG_CACHE_TTL", "3600")),
        )


# =============================================================================
# SECTION 10: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Use of Sold Products agent.

    Configures regulatory framework compliance checks for Scope 3
    Category 11 use-of-sold-products emissions reporting.

    Attributes:
        enabled_frameworks: Enabled frameworks comma-separated (GL_USP_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict compliance mode (GL_USP_COMPLIANCE_STRICT_MODE)
        materiality_threshold: Materiality threshold percentage (GL_USP_COMPLIANCE_MATERIALITY_THRESHOLD)
        double_counting_check: Check for double counting (GL_USP_COMPLIANCE_DOUBLE_COUNTING_CHECK)
        boundary_enforcement: Enforce Scope 3 boundary (GL_USP_COMPLIANCE_BOUNDARY_ENFORCEMENT)
        require_data_quality: Require data quality scoring (GL_USP_COMPLIANCE_REQUIRE_DATA_QUALITY)
        min_data_quality_score: Minimum acceptable DQI score (GL_USP_COMPLIANCE_MIN_DATA_QUALITY_SCORE)
        require_lifetime_disclosure: Require product lifetime disclosure (GL_USP_COMPLIANCE_REQUIRE_LIFETIME_DISCLOSURE)

    Example:
        >>> compliance = ComplianceConfig(
        ...     enabled_frameworks="GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,SEC_CLIMATE",
        ...     strict_mode=False,
        ...     materiality_threshold=Decimal("0.01"),
        ...     double_counting_check=True,
        ...     boundary_enforcement=True,
        ...     require_data_quality=True,
        ...     min_data_quality_score=Decimal("2.0"),
        ...     require_lifetime_disclosure=True
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E1', 'CDP', 'SBTI', 'GRI', 'SEC_CLIMATE']
    """

    enabled_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,SEC_CLIMATE"
    )
    strict_mode: bool = False
    materiality_threshold: Decimal = Decimal("0.01")
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    require_data_quality: bool = True
    min_data_quality_score: Decimal = Decimal("2.0")
    require_lifetime_disclosure: bool = True

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
            "SEC_CLIMATE",
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
        """Parse enabled frameworks string into list."""
        return [f.strip() for f in self.enabled_frameworks.split(",") if f.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_frameworks": self.enabled_frameworks,
            "strict_mode": self.strict_mode,
            "materiality_threshold": str(self.materiality_threshold),
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "require_data_quality": self.require_data_quality,
            "min_data_quality_score": str(self.min_data_quality_score),
            "require_lifetime_disclosure": self.require_lifetime_disclosure,
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
            enabled_frameworks=os.getenv(
                "GL_USP_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,SEC_CLIMATE",
            ),
            strict_mode=os.getenv(
                "GL_USP_COMPLIANCE_STRICT_MODE", "false"
            ).lower() == "true",
            materiality_threshold=Decimal(
                os.getenv("GL_USP_COMPLIANCE_MATERIALITY_THRESHOLD", "0.01")
            ),
            double_counting_check=os.getenv(
                "GL_USP_COMPLIANCE_DOUBLE_COUNTING_CHECK", "true"
            ).lower() == "true",
            boundary_enforcement=os.getenv(
                "GL_USP_COMPLIANCE_BOUNDARY_ENFORCEMENT", "true"
            ).lower() == "true",
            require_data_quality=os.getenv(
                "GL_USP_COMPLIANCE_REQUIRE_DATA_QUALITY", "true"
            ).lower() == "true",
            min_data_quality_score=Decimal(
                os.getenv("GL_USP_COMPLIANCE_MIN_DATA_QUALITY_SCORE", "2.0")
            ),
            require_lifetime_disclosure=os.getenv(
                "GL_USP_COMPLIANCE_REQUIRE_LIFETIME_DISCLOSURE", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 11: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Use of Sold Products agent.

    Configures data provenance tracking with SHA-256 hashing for
    complete audit trails and reproducibility.

    Attributes:
        hash_algorithm: Hash algorithm for provenance (GL_USP_PROVENANCE_HASH_ALGORITHM)
        chain_enabled: Enable chain hashing across pipeline (GL_USP_PROVENANCE_CHAIN_ENABLED)
        merkle_enabled: Enable Merkle root for batch aggregation (GL_USP_PROVENANCE_MERKLE_ENABLED)
        store_intermediate: Store intermediate hashes (GL_USP_PROVENANCE_STORE_INTERMEDIATE)
        include_config_hash: Include config hash in provenance (GL_USP_PROVENANCE_INCLUDE_CONFIG_HASH)
        include_ef_hash: Include EF data hash in provenance (GL_USP_PROVENANCE_INCLUDE_EF_HASH)

    Example:
        >>> provenance = ProvenanceConfig(
        ...     hash_algorithm="sha256",
        ...     chain_enabled=True,
        ...     merkle_enabled=True,
        ...     store_intermediate=True,
        ...     include_config_hash=True,
        ...     include_ef_hash=True
        ... )
        >>> provenance.hash_algorithm
        'sha256'
    """

    hash_algorithm: str = "sha256"
    chain_enabled: bool = True
    merkle_enabled: bool = True
    store_intermediate: bool = True
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
            "hash_algorithm": self.hash_algorithm,
            "chain_enabled": self.chain_enabled,
            "merkle_enabled": self.merkle_enabled,
            "store_intermediate": self.store_intermediate,
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
            hash_algorithm=os.getenv("GL_USP_PROVENANCE_HASH_ALGORITHM", "sha256"),
            chain_enabled=os.getenv(
                "GL_USP_PROVENANCE_CHAIN_ENABLED", "true"
            ).lower() == "true",
            merkle_enabled=os.getenv(
                "GL_USP_PROVENANCE_MERKLE_ENABLED", "true"
            ).lower() == "true",
            store_intermediate=os.getenv(
                "GL_USP_PROVENANCE_STORE_INTERMEDIATE", "true"
            ).lower() == "true",
            include_config_hash=os.getenv(
                "GL_USP_PROVENANCE_INCLUDE_CONFIG_HASH", "true"
            ).lower() == "true",
            include_ef_hash=os.getenv(
                "GL_USP_PROVENANCE_INCLUDE_EF_HASH", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 12: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty configuration for Use of Sold Products agent.

    Configures uncertainty quantification for use-phase emissions
    calculations. Supports IPCC default ranges, Monte Carlo simulation,
    and analytical propagation methods.

    Attributes:
        method: Default uncertainty method (GL_USP_UNCERTAINTY_METHOD)
        iterations: Monte Carlo iterations (GL_USP_UNCERTAINTY_ITERATIONS)
        confidence_level: Confidence level for intervals (GL_USP_UNCERTAINTY_CONFIDENCE_LEVEL)
        include_parameter: Include parameter uncertainty (GL_USP_UNCERTAINTY_INCLUDE_PARAMETER)
        include_model: Include model uncertainty (GL_USP_UNCERTAINTY_INCLUDE_MODEL)
        include_activity: Include activity data uncertainty (GL_USP_UNCERTAINTY_INCLUDE_ACTIVITY)
        seed: Random seed for reproducibility (GL_USP_UNCERTAINTY_SEED)

    Example:
        >>> uncertainty = UncertaintyConfig(
        ...     method="MONTE_CARLO",
        ...     iterations=10000,
        ...     confidence_level=Decimal("0.95"),
        ...     include_parameter=True,
        ...     include_model=False,
        ...     include_activity=True,
        ...     seed=42
        ... )
        >>> uncertainty.method
        'MONTE_CARLO'
    """

    method: str = "MONTE_CARLO"
    iterations: int = 10000
    confidence_level: Decimal = Decimal("0.95")
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
            "MONTE_CARLO",
            "BOOTSTRAP",
            "IPCC_DEFAULT",
            "BAYESIAN",
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

        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 when specified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "iterations": self.iterations,
            "confidence_level": str(self.confidence_level),
            "include_parameter": self.include_parameter,
            "include_model": self.include_model,
            "include_activity": self.include_activity,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "confidence_level" in data_copy:
            data_copy["confidence_level"] = Decimal(data_copy["confidence_level"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        seed_str = os.getenv("GL_USP_UNCERTAINTY_SEED")
        seed_val = int(seed_str) if seed_str else None

        return cls(
            method=os.getenv("GL_USP_UNCERTAINTY_METHOD", "MONTE_CARLO"),
            iterations=int(os.getenv("GL_USP_UNCERTAINTY_ITERATIONS", "10000")),
            confidence_level=Decimal(
                os.getenv("GL_USP_UNCERTAINTY_CONFIDENCE_LEVEL", "0.95")
            ),
            include_parameter=os.getenv(
                "GL_USP_UNCERTAINTY_INCLUDE_PARAMETER", "true"
            ).lower() == "true",
            include_model=os.getenv(
                "GL_USP_UNCERTAINTY_INCLUDE_MODEL", "false"
            ).lower() == "true",
            include_activity=os.getenv(
                "GL_USP_UNCERTAINTY_INCLUDE_ACTIVITY", "true"
            ).lower() == "true",
            seed=seed_val,
        )


# =============================================================================
# SECTION 13: DQI CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DQIConfig:
    """
    Data Quality Indicator configuration for Use of Sold Products agent.

    Configures the 5-dimensional DQI scoring system per GHG Protocol
    guidance. Each dimension is scored 1-5 (1=best, 5=worst) and combined
    using configurable weights.

    Attributes:
        min_score: Minimum acceptable overall DQI score (GL_USP_DQI_MIN_SCORE)
        weight_technological: Weight for technological representativeness (GL_USP_DQI_WEIGHT_TECH)
        weight_temporal: Weight for temporal representativeness (GL_USP_DQI_WEIGHT_TEMPORAL)
        weight_geographical: Weight for geographical representativeness (GL_USP_DQI_WEIGHT_GEO)
        weight_completeness: Weight for data completeness (GL_USP_DQI_WEIGHT_COMPLETENESS)
        weight_reliability: Weight for data reliability (GL_USP_DQI_WEIGHT_RELIABILITY)
        enable_auto_scoring: Enable automatic DQI scoring (GL_USP_DQI_ENABLE_AUTO)
        fail_on_low_quality: Fail calculation if DQI below threshold (GL_USP_DQI_FAIL_ON_LOW)

    Example:
        >>> dqi = DQIConfig(
        ...     min_score=Decimal("2.0"),
        ...     weight_technological=Decimal("0.20"),
        ...     weight_temporal=Decimal("0.20"),
        ...     weight_geographical=Decimal("0.20"),
        ...     weight_completeness=Decimal("0.20"),
        ...     weight_reliability=Decimal("0.20"),
        ...     enable_auto_scoring=True,
        ...     fail_on_low_quality=False
        ... )
        >>> dqi.min_score
        Decimal('2.0')
    """

    min_score: Decimal = Decimal("2.0")
    weight_technological: Decimal = Decimal("0.20")
    weight_temporal: Decimal = Decimal("0.20")
    weight_geographical: Decimal = Decimal("0.20")
    weight_completeness: Decimal = Decimal("0.20")
    weight_reliability: Decimal = Decimal("0.20")
    enable_auto_scoring: bool = True
    fail_on_low_quality: bool = False

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
                raise ValueError("All DQI dimension weights must be between 0 and 1")

        total = sum(weights)
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(
                f"DQI dimension weights must sum to 1.0, got {total}"
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
            "enable_auto_scoring": self.enable_auto_scoring,
            "fail_on_low_quality": self.fail_on_low_quality,
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
            min_score=Decimal(os.getenv("GL_USP_DQI_MIN_SCORE", "2.0")),
            weight_technological=Decimal(
                os.getenv("GL_USP_DQI_WEIGHT_TECH", "0.20")
            ),
            weight_temporal=Decimal(
                os.getenv("GL_USP_DQI_WEIGHT_TEMPORAL", "0.20")
            ),
            weight_geographical=Decimal(
                os.getenv("GL_USP_DQI_WEIGHT_GEO", "0.20")
            ),
            weight_completeness=Decimal(
                os.getenv("GL_USP_DQI_WEIGHT_COMPLETENESS", "0.20")
            ),
            weight_reliability=Decimal(
                os.getenv("GL_USP_DQI_WEIGHT_RELIABILITY", "0.20")
            ),
            enable_auto_scoring=os.getenv(
                "GL_USP_DQI_ENABLE_AUTO", "true"
            ).lower() == "true",
            fail_on_low_quality=os.getenv(
                "GL_USP_DQI_FAIL_ON_LOW", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 14: PIPELINE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline configuration for Use of Sold Products agent.

    Configures batch processing, timeouts, retry logic, and
    concurrency controls for the calculation pipeline.

    Attributes:
        batch_size: Default batch processing size (GL_USP_PIPELINE_BATCH_SIZE)
        timeout_seconds: Pipeline timeout in seconds (GL_USP_PIPELINE_TIMEOUT)
        max_retries: Maximum retry attempts (GL_USP_PIPELINE_MAX_RETRIES)
        retry_delay_seconds: Delay between retries in seconds (GL_USP_PIPELINE_RETRY_DELAY)
        max_concurrent: Maximum concurrent pipeline executions (GL_USP_PIPELINE_MAX_CONCURRENT)
        enable_parallel: Enable parallel stage execution (GL_USP_PIPELINE_ENABLE_PARALLEL)
        worker_threads: Worker thread count (GL_USP_PIPELINE_WORKER_THREADS)
        enable_checkpointing: Enable pipeline checkpointing (GL_USP_PIPELINE_ENABLE_CHECKPOINTING)

    Example:
        >>> pipeline = PipelineConfig(
        ...     batch_size=500,
        ...     timeout_seconds=300,
        ...     max_retries=3,
        ...     retry_delay_seconds=5,
        ...     max_concurrent=10,
        ...     enable_parallel=True,
        ...     worker_threads=4,
        ...     enable_checkpointing=True
        ... )
        >>> pipeline.batch_size
        500
    """

    batch_size: int = 500
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5
    max_concurrent: int = 10
    enable_parallel: bool = True
    worker_threads: int = 4
    enable_checkpointing: bool = True

    def validate(self) -> None:
        """
        Validate pipeline configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.batch_size < 1 or self.batch_size > 100000:
            raise ValueError("batch_size must be between 1 and 100000")

        if self.timeout_seconds < 1 or self.timeout_seconds > 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")

        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")

        if self.retry_delay_seconds < 0 or self.retry_delay_seconds > 300:
            raise ValueError("retry_delay_seconds must be between 0 and 300")

        if self.max_concurrent < 1 or self.max_concurrent > 1000:
            raise ValueError("max_concurrent must be between 1 and 1000")

        if self.worker_threads < 1 or self.worker_threads > 64:
            raise ValueError("worker_threads must be between 1 and 64")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "max_concurrent": self.max_concurrent,
            "enable_parallel": self.enable_parallel,
            "worker_threads": self.worker_threads,
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
            batch_size=int(os.getenv("GL_USP_PIPELINE_BATCH_SIZE", "500")),
            timeout_seconds=int(os.getenv("GL_USP_PIPELINE_TIMEOUT", "300")),
            max_retries=int(os.getenv("GL_USP_PIPELINE_MAX_RETRIES", "3")),
            retry_delay_seconds=int(os.getenv("GL_USP_PIPELINE_RETRY_DELAY", "5")),
            max_concurrent=int(os.getenv("GL_USP_PIPELINE_MAX_CONCURRENT", "10")),
            enable_parallel=os.getenv(
                "GL_USP_PIPELINE_ENABLE_PARALLEL", "true"
            ).lower() == "true",
            worker_threads=int(os.getenv("GL_USP_PIPELINE_WORKER_THREADS", "4")),
            enable_checkpointing=os.getenv(
                "GL_USP_PIPELINE_ENABLE_CHECKPOINTING", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 15: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Use of Sold Products agent.

    Configures in-memory and Redis caching for emission factor lookups,
    calculation results, and intermediate data.

    Attributes:
        enabled: Enable caching (GL_USP_CACHE_ENABLED)
        ttl_seconds: Cache TTL in seconds (GL_USP_CACHE_TTL)
        max_size: Maximum cache entries (GL_USP_CACHE_MAX_SIZE)
        key_prefix: Cache key prefix (GL_USP_CACHE_KEY_PREFIX)
        cache_ef_lookups: Cache emission factor lookups (GL_USP_CACHE_EF_LOOKUPS)
        cache_calculations: Cache calculation results (GL_USP_CACHE_CALCULATIONS)
        eviction_policy: Cache eviction policy (GL_USP_CACHE_EVICTION_POLICY)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl_seconds=3600,
        ...     max_size=10000,
        ...     key_prefix="gl_usp:",
        ...     cache_ef_lookups=True,
        ...     cache_calculations=True,
        ...     eviction_policy="LRU"
        ... )
        >>> cache.ttl_seconds
        3600
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    key_prefix: str = "gl_usp:"
    cache_ef_lookups: bool = True
    cache_calculations: bool = True
    eviction_policy: str = "LRU"

    def validate(self) -> None:
        """
        Validate cache configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.ttl_seconds < 0 or self.ttl_seconds > 86400:
            raise ValueError("ttl_seconds must be between 0 and 86400 (24 hours)")

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
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "key_prefix": self.key_prefix,
            "cache_ef_lookups": self.cache_ef_lookups,
            "cache_calculations": self.cache_calculations,
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
            enabled=os.getenv("GL_USP_CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("GL_USP_CACHE_TTL", "3600")),
            max_size=int(os.getenv("GL_USP_CACHE_MAX_SIZE", "10000")),
            key_prefix=os.getenv("GL_USP_CACHE_KEY_PREFIX", "gl_usp:"),
            cache_ef_lookups=os.getenv(
                "GL_USP_CACHE_EF_LOOKUPS", "true"
            ).lower() == "true",
            cache_calculations=os.getenv(
                "GL_USP_CACHE_CALCULATIONS", "true"
            ).lower() == "true",
            eviction_policy=os.getenv("GL_USP_CACHE_EVICTION_POLICY", "LRU"),
        )


# =============================================================================
# SECTION 16: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Use of Sold Products agent.

    Configures Prometheus-compatible metrics collection including
    histogram buckets for latency tracking and tenant-level isolation.

    Attributes:
        enabled: Enable metrics collection (GL_USP_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_USP_METRICS_PREFIX)
        include_tenant: Include tenant label in metrics (GL_USP_METRICS_INCLUDE_TENANT)
        histogram_buckets: Comma-separated histogram bucket boundaries (GL_USP_METRICS_HISTOGRAM_BUCKETS)
        enable_latency_tracking: Track per-engine latency (GL_USP_METRICS_ENABLE_LATENCY)
        enable_error_counting: Track error counts by type (GL_USP_METRICS_ENABLE_ERRORS)
        enable_throughput_tracking: Track records per second (GL_USP_METRICS_ENABLE_THROUGHPUT)

    Example:
        >>> metrics_cfg = MetricsConfig(
        ...     enabled=True,
        ...     prefix="gl_usp_",
        ...     include_tenant=True,
        ...     histogram_buckets="0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
        ...     enable_latency_tracking=True,
        ...     enable_error_counting=True,
        ...     enable_throughput_tracking=True
        ... )
        >>> metrics_cfg.get_histogram_buckets()
        (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    """

    enabled: bool = True
    prefix: str = "gl_usp_"
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
            enabled=os.getenv("GL_USP_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_USP_METRICS_PREFIX", "gl_usp_"),
            include_tenant=os.getenv(
                "GL_USP_METRICS_INCLUDE_TENANT", "true"
            ).lower() == "true",
            histogram_buckets=os.getenv(
                "GL_USP_METRICS_HISTOGRAM_BUCKETS",
                "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
            enable_latency_tracking=os.getenv(
                "GL_USP_METRICS_ENABLE_LATENCY", "true"
            ).lower() == "true",
            enable_error_counting=os.getenv(
                "GL_USP_METRICS_ENABLE_ERRORS", "true"
            ).lower() == "true",
            enable_throughput_tracking=os.getenv(
                "GL_USP_METRICS_ENABLE_THROUGHPUT", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 17: SECURITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SecurityConfig:
    """
    Security configuration for Use of Sold Products agent.

    Configures tenant isolation, audit logging, and access control
    for multi-tenant deployments.

    Attributes:
        tenant_isolation: Enable tenant data isolation (GL_USP_SECURITY_TENANT_ISOLATION)
        audit_enabled: Enable audit logging (GL_USP_SECURITY_AUDIT_ENABLED)
        rate_limit: Requests per minute per tenant (GL_USP_SECURITY_RATE_LIMIT)
        max_request_size_mb: Maximum request body size in MB (GL_USP_SECURITY_MAX_REQUEST_SIZE_MB)
        enable_input_sanitization: Enable input sanitization (GL_USP_SECURITY_ENABLE_INPUT_SANITIZATION)
        require_authentication: Require authentication for API (GL_USP_SECURITY_REQUIRE_AUTH)

    Example:
        >>> security = SecurityConfig(
        ...     tenant_isolation=True,
        ...     audit_enabled=True,
        ...     rate_limit=100,
        ...     max_request_size_mb=10,
        ...     enable_input_sanitization=True,
        ...     require_authentication=True
        ... )
        >>> security.tenant_isolation
        True
    """

    tenant_isolation: bool = True
    audit_enabled: bool = True
    rate_limit: int = 100
    max_request_size_mb: int = 10
    enable_input_sanitization: bool = True
    require_authentication: bool = True

    def validate(self) -> None:
        """
        Validate security configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.rate_limit < 1 or self.rate_limit > 10000:
            raise ValueError("rate_limit must be between 1 and 10000")

        if self.max_request_size_mb < 1 or self.max_request_size_mb > 100:
            raise ValueError("max_request_size_mb must be between 1 and 100")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_isolation": self.tenant_isolation,
            "audit_enabled": self.audit_enabled,
            "rate_limit": self.rate_limit,
            "max_request_size_mb": self.max_request_size_mb,
            "enable_input_sanitization": self.enable_input_sanitization,
            "require_authentication": self.require_authentication,
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
                "GL_USP_SECURITY_TENANT_ISOLATION", "true"
            ).lower() == "true",
            audit_enabled=os.getenv(
                "GL_USP_SECURITY_AUDIT_ENABLED", "true"
            ).lower() == "true",
            rate_limit=int(os.getenv("GL_USP_SECURITY_RATE_LIMIT", "100")),
            max_request_size_mb=int(
                os.getenv("GL_USP_SECURITY_MAX_REQUEST_SIZE_MB", "10")
            ),
            enable_input_sanitization=os.getenv(
                "GL_USP_SECURITY_ENABLE_INPUT_SANITIZATION", "true"
            ).lower() == "true",
            require_authentication=os.getenv(
                "GL_USP_SECURITY_REQUIRE_AUTH", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 18: DEGRADATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DegradationConfig:
    """
    Efficiency degradation configuration for Use of Sold Products agent.

    Configures how product efficiency degrades over its lifetime,
    affecting energy consumption and emissions. Supports linear,
    exponential, and step-function degradation models.

    Attributes:
        curves_enabled: Enable degradation curve modeling (GL_USP_DEGRADATION_CURVES_ENABLED)
        max_degradation_pct: Maximum total degradation percentage (GL_USP_DEGRADATION_MAX_PCT)
        default_model: Default degradation model type (GL_USP_DEGRADATION_DEFAULT_MODEL)
        default_annual_rate: Default annual degradation rate (GL_USP_DEGRADATION_DEFAULT_ANNUAL_RATE)
        enable_maintenance_offset: Enable maintenance offset factor (GL_USP_DEGRADATION_ENABLE_MAINTENANCE)
        default_maintenance_factor: Default maintenance factor (GL_USP_DEGRADATION_DEFAULT_MAINTENANCE_FACTOR)
        enable_climate_adjustment: Enable climate-zone degradation adjustment (GL_USP_DEGRADATION_ENABLE_CLIMATE)
        min_efficiency_floor: Minimum efficiency floor (GL_USP_DEGRADATION_MIN_EFFICIENCY_FLOOR)

    Example:
        >>> degradation = DegradationConfig(
        ...     curves_enabled=True,
        ...     max_degradation_pct=Decimal("50"),
        ...     default_model="LINEAR",
        ...     default_annual_rate=Decimal("0.01"),
        ...     enable_maintenance_offset=True,
        ...     default_maintenance_factor=Decimal("0.95"),
        ...     enable_climate_adjustment=False,
        ...     min_efficiency_floor=Decimal("0.50")
        ... )
        >>> degradation.max_degradation_pct
        Decimal('50')
    """

    curves_enabled: bool = True
    max_degradation_pct: Decimal = Decimal("50")
    default_model: str = "LINEAR"
    default_annual_rate: Decimal = Decimal("0.01")
    enable_maintenance_offset: bool = True
    default_maintenance_factor: Decimal = Decimal("0.95")
    enable_climate_adjustment: bool = False
    min_efficiency_floor: Decimal = Decimal("0.50")

    def validate(self) -> None:
        """
        Validate degradation configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.max_degradation_pct < Decimal("0") or self.max_degradation_pct > Decimal("100"):
            raise ValueError("max_degradation_pct must be between 0 and 100")

        valid_models = {"LINEAR", "EXPONENTIAL", "STEP", "LOGARITHMIC", "CUSTOM"}
        if self.default_model not in valid_models:
            raise ValueError(
                f"Invalid default_model '{self.default_model}'. "
                f"Must be one of {valid_models}"
            )

        if self.default_annual_rate < Decimal("0") or self.default_annual_rate > Decimal("0.5"):
            raise ValueError("default_annual_rate must be between 0 and 0.5")

        if self.default_maintenance_factor < Decimal("0.1") or self.default_maintenance_factor > Decimal("1.0"):
            raise ValueError("default_maintenance_factor must be between 0.1 and 1.0")

        if self.min_efficiency_floor < Decimal("0") or self.min_efficiency_floor > Decimal("1.0"):
            raise ValueError("min_efficiency_floor must be between 0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "curves_enabled": self.curves_enabled,
            "max_degradation_pct": str(self.max_degradation_pct),
            "default_model": self.default_model,
            "default_annual_rate": str(self.default_annual_rate),
            "enable_maintenance_offset": self.enable_maintenance_offset,
            "default_maintenance_factor": str(self.default_maintenance_factor),
            "enable_climate_adjustment": self.enable_climate_adjustment,
            "min_efficiency_floor": str(self.min_efficiency_floor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DegradationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "max_degradation_pct", "default_annual_rate",
            "default_maintenance_factor", "min_efficiency_floor",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "DegradationConfig":
        """Load from environment variables."""
        return cls(
            curves_enabled=os.getenv(
                "GL_USP_DEGRADATION_CURVES_ENABLED", "true"
            ).lower() == "true",
            max_degradation_pct=Decimal(
                os.getenv("GL_USP_DEGRADATION_MAX_PCT", "50")
            ),
            default_model=os.getenv("GL_USP_DEGRADATION_DEFAULT_MODEL", "LINEAR"),
            default_annual_rate=Decimal(
                os.getenv("GL_USP_DEGRADATION_DEFAULT_ANNUAL_RATE", "0.01")
            ),
            enable_maintenance_offset=os.getenv(
                "GL_USP_DEGRADATION_ENABLE_MAINTENANCE", "true"
            ).lower() == "true",
            default_maintenance_factor=Decimal(
                os.getenv("GL_USP_DEGRADATION_DEFAULT_MAINTENANCE_FACTOR", "0.95")
            ),
            enable_climate_adjustment=os.getenv(
                "GL_USP_DEGRADATION_ENABLE_CLIMATE", "false"
            ).lower() == "true",
            min_efficiency_floor=Decimal(
                os.getenv("GL_USP_DEGRADATION_MIN_EFFICIENCY_FLOOR", "0.50")
            ),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass(frozen=True)
class UseOfSoldProductsConfig:
    """
    Master configuration class for Use of Sold Products agent (AGENT-MRV-024).

    This frozen dataclass aggregates all 18 configuration sections and provides
    a unified interface for accessing configuration values. It implements the
    singleton pattern with thread-safe access.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        direct_emissions: Direct use-phase emissions configuration
        indirect_emissions: Indirect use-phase emissions configuration
        fuels_and_feedstocks: Fuels and feedstocks sold configuration
        lifetime_modeling: Product lifetime modeling configuration
        grid_factors: Grid emission factor configuration
        fuel_factors: Fuel emission factor configuration
        refrigerants: Refrigerant GWP configuration
        compliance: Regulatory compliance configuration
        provenance: Data provenance configuration
        uncertainty: Uncertainty quantification configuration
        dqi: Data quality indicator configuration
        pipeline: Pipeline processing configuration
        cache: Cache layer configuration
        metrics: Prometheus metrics configuration
        security: Security and access control configuration
        degradation: Efficiency degradation configuration

    Example:
        >>> config = UseOfSoldProductsConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-011'
        >>> config.direct_emissions.enable_fuel
        True
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    direct_emissions: DirectEmissionsConfig = field(default_factory=DirectEmissionsConfig)
    indirect_emissions: IndirectEmissionsConfig = field(default_factory=IndirectEmissionsConfig)
    fuels_and_feedstocks: FuelsAndFeedstocksConfig = field(default_factory=FuelsAndFeedstocksConfig)
    lifetime_modeling: LifetimeModelingConfig = field(default_factory=LifetimeModelingConfig)
    grid_factors: GridFactorsConfig = field(default_factory=GridFactorsConfig)
    fuel_factors: FuelFactorsConfig = field(default_factory=FuelFactorsConfig)
    refrigerants: RefrigerantsConfig = field(default_factory=RefrigerantsConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    dqi: DQIConfig = field(default_factory=DQIConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    def validate_all(self) -> List[str]:
        """
        Validate all configuration sections and return list of errors.

        Unlike individual validate() methods which raise on first error,
        this method collects all validation errors across all sections.

        Returns:
            List of validation error messages (empty if all valid)

        Example:
            >>> config = UseOfSoldProductsConfig.from_env()
            >>> errors = config.validate_all()
            >>> if errors:
            ...     for error in errors:
            ...         print(f"  - {error}")
        """
        errors: List[str] = []

        sections = [
            ("general", self.general),
            ("database", self.database),
            ("direct_emissions", self.direct_emissions),
            ("indirect_emissions", self.indirect_emissions),
            ("fuels_and_feedstocks", self.fuels_and_feedstocks),
            ("lifetime_modeling", self.lifetime_modeling),
            ("grid_factors", self.grid_factors),
            ("fuel_factors", self.fuel_factors),
            ("refrigerants", self.refrigerants),
            ("compliance", self.compliance),
            ("provenance", self.provenance),
            ("uncertainty", self.uncertainty),
            ("dqi", self.dqi),
            ("pipeline", self.pipeline),
            ("cache", self.cache),
            ("metrics", self.metrics),
            ("security", self.security),
            ("degradation", self.degradation),
        ]

        for section_name, section in sections:
            try:
                section.validate()
            except ValueError as e:
                errors.append(f"{section_name}: {str(e)}")

        errors.extend(self._cross_validate())

        return errors

    def _cross_validate(self) -> List[str]:
        """
        Perform cross-section validation checks.

        Returns:
            List of cross-validation error messages
        """
        errors: List[str] = []

        if self.general.api_prefix and not self.general.api_prefix.startswith("/api/"):
            errors.append(
                "cross-validation: general.api_prefix should start with '/api/'"
            )

        if self.database.pool_min > self.database.pool_max:
            errors.append(
                "cross-validation: database.pool_min must be <= database.pool_max"
            )

        if self.pipeline.batch_size > self.general.max_batch_size:
            errors.append(
                "cross-validation: pipeline.batch_size should not exceed "
                "general.max_batch_size"
            )

        if self.lifetime_modeling.default_lifetime_years > self.lifetime_modeling.max_lifetime_years:
            errors.append(
                "cross-validation: lifetime_modeling.default_lifetime_years must be <= "
                "lifetime_modeling.max_lifetime_years"
            )

        if (
            self.direct_emissions.default_gwp_standard
            != self.refrigerants.default_gwp_standard
        ):
            logger.warning(
                "direct_emissions.default_gwp_standard '%s' differs from "
                "refrigerants.default_gwp_standard '%s' - ensure consistency",
                self.direct_emissions.default_gwp_standard,
                self.refrigerants.default_gwp_standard,
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
            "direct_emissions": self.direct_emissions.to_dict(),
            "indirect_emissions": self.indirect_emissions.to_dict(),
            "fuels_and_feedstocks": self.fuels_and_feedstocks.to_dict(),
            "lifetime_modeling": self.lifetime_modeling.to_dict(),
            "grid_factors": self.grid_factors.to_dict(),
            "fuel_factors": self.fuel_factors.to_dict(),
            "refrigerants": self.refrigerants.to_dict(),
            "compliance": self.compliance.to_dict(),
            "provenance": self.provenance.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "dqi": self.dqi.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "cache": self.cache.to_dict(),
            "metrics": self.metrics.to_dict(),
            "security": self.security.to_dict(),
            "degradation": self.degradation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UseOfSoldProductsConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            UseOfSoldProductsConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            direct_emissions=DirectEmissionsConfig.from_dict(data.get("direct_emissions", {})),
            indirect_emissions=IndirectEmissionsConfig.from_dict(data.get("indirect_emissions", {})),
            fuels_and_feedstocks=FuelsAndFeedstocksConfig.from_dict(data.get("fuels_and_feedstocks", {})),
            lifetime_modeling=LifetimeModelingConfig.from_dict(data.get("lifetime_modeling", {})),
            grid_factors=GridFactorsConfig.from_dict(data.get("grid_factors", {})),
            fuel_factors=FuelFactorsConfig.from_dict(data.get("fuel_factors", {})),
            refrigerants=RefrigerantsConfig.from_dict(data.get("refrigerants", {})),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})),
            dqi=DQIConfig.from_dict(data.get("dqi", {})),
            pipeline=PipelineConfig.from_dict(data.get("pipeline", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
            security=SecurityConfig.from_dict(data.get("security", {})),
            degradation=DegradationConfig.from_dict(data.get("degradation", {})),
        )

    @classmethod
    def from_env(cls) -> "UseOfSoldProductsConfig":
        """
        Load configuration from environment variables.

        Returns:
            UseOfSoldProductsConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            direct_emissions=DirectEmissionsConfig.from_env(),
            indirect_emissions=IndirectEmissionsConfig.from_env(),
            fuels_and_feedstocks=FuelsAndFeedstocksConfig.from_env(),
            lifetime_modeling=LifetimeModelingConfig.from_env(),
            grid_factors=GridFactorsConfig.from_env(),
            fuel_factors=FuelFactorsConfig.from_env(),
            refrigerants=RefrigerantsConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            dqi=DQIConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            cache=CacheConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            security=SecurityConfig.from_env(),
            degradation=DegradationConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[UseOfSoldProductsConfig] = None
_config_lock = threading.RLock()


def get_config() -> UseOfSoldProductsConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton. The first call will load configuration from
    environment variables. Subsequent calls return the cached instance.

    Returns:
        UseOfSoldProductsConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-011'

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The configuration is initialized only once using
        double-checked locking.
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info("Initializing UseOfSoldProductsConfig from environment")
                config = UseOfSoldProductsConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning("Configuration validation warning: %s", error)
                _config_instance = config
                logger.info(
                    "UseOfSoldProductsConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: UseOfSoldProductsConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.

    Args:
        config: UseOfSoldProductsConfig instance to set as singleton

    Raises:
        TypeError: If config is not a UseOfSoldProductsConfig instance

    Example:
        >>> custom_config = UseOfSoldProductsConfig.from_dict({...})
        >>> set_config(custom_config)

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently.
    """
    global _config_instance

    if not isinstance(config, UseOfSoldProductsConfig):
        raise TypeError(
            f"config must be a UseOfSoldProductsConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("UseOfSoldProductsConfig manually set")


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
        logger.info("UseOfSoldProductsConfig singleton reset")


def validate_config(config: UseOfSoldProductsConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

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


def print_config(config: UseOfSoldProductsConfig) -> None:
    """
    Print configuration in human-readable format.

    Sensitive fields (passwords, connection URLs) are redacted for security.
    Useful for debugging and verification.

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
    """
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
    print("  Use of Sold Products Configuration (AGENT-MRV-024)")
    print("  Agent ID: " + config.general.agent_id)
    print("  Version:  " + config.general.version)
    print("=" * 64)

    config_dict = config.to_dict()
    for section_name, section_data in config_dict.items():
        _print_section(section_name.upper(), section_data)

    print("\n" + "=" * 64)
    errors = config.validate_all()
    if errors:
        print(f"  VALIDATION ERRORS: {len(errors)}")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  VALIDATION: ALL SECTIONS PASSED")
    print("=" * 64)
