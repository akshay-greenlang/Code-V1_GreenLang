# -*- coding: utf-8 -*-
"""
Downstream Transportation & Distribution Configuration - AGENT-MRV-022

Thread-safe singleton configuration for GL-MRV-S3-009.
All environment variables prefixed with GL_DTO_.

This module provides comprehensive configuration management for the Downstream
Transportation & Distribution agent (GHG Protocol Scope 3 Category 9),
supporting:
- Outbound transportation emissions (road/rail/maritime/air/pipeline/intermodal)
- Distribution centre and warehouse operations (ambient/cold/retail)
- Last-mile delivery emissions (van/cargo bike/drone/locker)
- Spend-based EEIO calculations with CPI deflation and margin removal
- Average-data channel-level defaults
- Cold-chain reefer uplift and temperature regime tracking
- Return logistics and reverse distribution
- Allocation by mass, volume, revenue, or unit count
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi, GRI, GLEC)
- Provenance tracking with SHA-256 chain hashing
- Uncertainty quantification (Monte Carlo, IPCC default, bootstrap)

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-009'
    >>> config.transport.default_mode
    'ROAD'
    >>> config.compliance.enabled_frameworks
    'GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,GLEC'

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_DTO_ prefix. See individual config sections for specific variables.

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-009
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
    General configuration for Downstream Transportation agent.

    Attributes:
        enabled: Master switch for the agent (GL_DTO_ENABLED)
        debug: Enable debug mode with verbose logging (GL_DTO_DEBUG)
        log_level: Logging level (GL_DTO_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_DTO_AGENT_ID)
        agent_component: Agent component identifier (GL_DTO_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_DTO_VERSION)
        api_prefix: API route prefix (GL_DTO_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_DTO_MAX_BATCH_SIZE)
        default_gwp: Default GWP assessment report version (GL_DTO_DEFAULT_GWP)
        default_ef_source: Default emission factor source (GL_DTO_DEFAULT_EF_SOURCE)

    Example:
        >>> general = GeneralConfig()
        >>> general.agent_id
        'GL-MRV-S3-009'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-009"
    agent_component: str = "AGENT-MRV-022"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/downstream-transportation"
    max_batch_size: int = 1000
    default_gwp: str = "AR5"
    default_ef_source: str = "DEFRA"

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

        valid_gwp_versions = {"AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp_versions:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. "
                f"Must be one of {valid_gwp_versions}"
            )

        valid_ef_sources = {"DEFRA", "EPA", "GLEC", "ECOINVENT", "EEIO", "CUSTOM"}
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
            enabled=os.getenv("GL_DTO_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_DTO_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_DTO_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_DTO_AGENT_ID", "GL-MRV-S3-009"),
            agent_component=os.getenv("GL_DTO_AGENT_COMPONENT", "AGENT-MRV-022"),
            version=os.getenv("GL_DTO_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_DTO_API_PREFIX", "/api/v1/downstream-transportation"),
            max_batch_size=int(os.getenv("GL_DTO_MAX_BATCH_SIZE", "1000")),
            default_gwp=os.getenv("GL_DTO_DEFAULT_GWP", "AR5"),
            default_ef_source=os.getenv("GL_DTO_DEFAULT_EF_SOURCE", "DEFRA"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Downstream Transportation agent.

    Attributes:
        host: PostgreSQL host (GL_DTO_DB_HOST)
        port: PostgreSQL port (GL_DTO_DB_PORT)
        database: Database name (GL_DTO_DB_DATABASE)
        username: Database username (GL_DTO_DB_USERNAME)
        password: Database password (GL_DTO_DB_PASSWORD)
        schema: Database schema name (GL_DTO_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_DTO_DB_TABLE_PREFIX)
        pool_size: Connection pool size (GL_DTO_DB_POOL_SIZE)
        max_overflow: Max overflow connections (GL_DTO_DB_MAX_OVERFLOW)
        timeout: Connection timeout in seconds (GL_DTO_DB_TIMEOUT)
        echo: Enable SQL echo for debugging (GL_DTO_DB_ECHO)
        ssl_mode: SSL connection mode (GL_DTO_DB_SSL_MODE)

    Example:
        >>> db = DatabaseConfig()
        >>> db.table_prefix
        'gl_dto_'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "downstream_transportation_service"
    table_prefix: str = "gl_dto_"
    pool_size: int = 5
    max_overflow: int = 10
    timeout: int = 30
    echo: bool = False
    ssl_mode: str = "prefer"

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

        if self.pool_size < 1 or self.pool_size > 100:
            raise ValueError("pool_size must be between 1 and 100")

        if self.max_overflow < 0 or self.max_overflow > 200:
            raise ValueError("max_overflow must be between 0 and 200")

        if self.timeout < 1 or self.timeout > 300:
            raise ValueError("timeout must be between 1 and 300 seconds")

        valid_ssl_modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl_modes:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. "
                f"Must be one of {valid_ssl_modes}"
            )

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
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "timeout": self.timeout,
            "echo": self.echo,
            "ssl_mode": self.ssl_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_DTO_DB_HOST", "localhost"),
            port=int(os.getenv("GL_DTO_DB_PORT", "5432")),
            database=os.getenv("GL_DTO_DB_DATABASE", "greenlang"),
            username=os.getenv("GL_DTO_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_DTO_DB_PASSWORD", ""),
            schema=os.getenv("GL_DTO_DB_SCHEMA", "downstream_transportation_service"),
            table_prefix=os.getenv("GL_DTO_DB_TABLE_PREFIX", "gl_dto_"),
            pool_size=int(os.getenv("GL_DTO_DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("GL_DTO_DB_MAX_OVERFLOW", "10")),
            timeout=int(os.getenv("GL_DTO_DB_TIMEOUT", "30")),
            echo=os.getenv("GL_DTO_DB_ECHO", "false").lower() == "true",
            ssl_mode=os.getenv("GL_DTO_DB_SSL_MODE", "prefer"),
        )


# =============================================================================
# SECTION 3: TRANSPORT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TransportConfig:
    """
    Transport mode configuration for Downstream Transportation agent.

    Configures default transport parameters for outbound shipment emissions
    including mode selection, load factor, well-to-tank, and cold chain.

    Attributes:
        default_mode: Default transport mode (GL_DTO_TRANSPORT_DEFAULT_MODE)
        default_load_factor: Default vehicle utilisation rate (GL_DTO_TRANSPORT_DEFAULT_LOAD_FACTOR)
        wtt_enabled: Enable well-to-tank emissions (GL_DTO_TRANSPORT_WTT_ENABLED)
        cold_chain_enabled: Enable cold chain tracking (GL_DTO_TRANSPORT_COLD_CHAIN_ENABLED)
        enable_multileg: Enable multi-leg transport chains (GL_DTO_TRANSPORT_ENABLE_MULTILEG)
        max_legs: Maximum number of legs per shipment (GL_DTO_TRANSPORT_MAX_LEGS)
        include_hub_emissions: Include transshipment hub emissions (GL_DTO_TRANSPORT_INCLUDE_HUB_EMISSIONS)
        default_ef_standard: Default EF standard (GL_DTO_TRANSPORT_DEFAULT_EF_STANDARD)

    Example:
        >>> transport = TransportConfig()
        >>> transport.default_mode
        'ROAD'
        >>> transport.wtt_enabled
        True
    """

    default_mode: str = "ROAD"
    default_load_factor: Decimal = Decimal("0.65")
    wtt_enabled: bool = True
    cold_chain_enabled: bool = True
    enable_multileg: bool = True
    max_legs: int = 10
    include_hub_emissions: bool = True
    default_ef_standard: str = "GLEC"

    def validate(self) -> None:
        """
        Validate transport configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_modes = {
            "ROAD", "RAIL", "MARITIME", "AIR", "PIPELINE", "INTERMODAL",
        }
        if self.default_mode not in valid_modes:
            raise ValueError(
                f"Invalid default_mode '{self.default_mode}'. "
                f"Must be one of {valid_modes}"
            )

        if self.default_load_factor < Decimal("0.01") or self.default_load_factor > Decimal("1.0"):
            raise ValueError("default_load_factor must be between 0.01 and 1.0")

        if self.max_legs < 1 or self.max_legs > 50:
            raise ValueError("max_legs must be between 1 and 50")

        valid_ef_standards = {"DEFRA", "EPA", "GLEC", "ECOINVENT", "CUSTOM"}
        if self.default_ef_standard not in valid_ef_standards:
            raise ValueError(
                f"Invalid default_ef_standard '{self.default_ef_standard}'. "
                f"Must be one of {valid_ef_standards}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_mode": self.default_mode,
            "default_load_factor": str(self.default_load_factor),
            "wtt_enabled": self.wtt_enabled,
            "cold_chain_enabled": self.cold_chain_enabled,
            "enable_multileg": self.enable_multileg,
            "max_legs": self.max_legs,
            "include_hub_emissions": self.include_hub_emissions,
            "default_ef_standard": self.default_ef_standard,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransportConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_load_factor" in data_copy:
            data_copy["default_load_factor"] = Decimal(data_copy["default_load_factor"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "TransportConfig":
        """Load from environment variables."""
        return cls(
            default_mode=os.getenv("GL_DTO_TRANSPORT_DEFAULT_MODE", "ROAD"),
            default_load_factor=Decimal(
                os.getenv("GL_DTO_TRANSPORT_DEFAULT_LOAD_FACTOR", "0.65")
            ),
            wtt_enabled=os.getenv(
                "GL_DTO_TRANSPORT_WTT_ENABLED", "true"
            ).lower() == "true",
            cold_chain_enabled=os.getenv(
                "GL_DTO_TRANSPORT_COLD_CHAIN_ENABLED", "true"
            ).lower() == "true",
            enable_multileg=os.getenv(
                "GL_DTO_TRANSPORT_ENABLE_MULTILEG", "true"
            ).lower() == "true",
            max_legs=int(os.getenv("GL_DTO_TRANSPORT_MAX_LEGS", "10")),
            include_hub_emissions=os.getenv(
                "GL_DTO_TRANSPORT_INCLUDE_HUB_EMISSIONS", "true"
            ).lower() == "true",
            default_ef_standard=os.getenv(
                "GL_DTO_TRANSPORT_DEFAULT_EF_STANDARD", "GLEC"
            ),
        )


# =============================================================================
# SECTION 4: DISTANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DistanceConfig:
    """
    Distance-based calculation configuration for Downstream Transportation agent.

    Configures tonne-km based calculations with distance limits and default
    vehicle types per transport mode.

    Attributes:
        max_distance_km: Maximum allowable distance in km (GL_DTO_DISTANCE_MAX_DISTANCE_KM)
        min_weight_tonnes: Minimum weight in tonnes (GL_DTO_DISTANCE_MIN_WEIGHT_TONNES)
        default_vehicle_type: Default road vehicle type (GL_DTO_DISTANCE_DEFAULT_VEHICLE_TYPE)
        default_vessel_type: Default maritime vessel type (GL_DTO_DISTANCE_DEFAULT_VESSEL_TYPE)
        default_aircraft_type: Default aircraft type (GL_DTO_DISTANCE_DEFAULT_AIRCRAFT_TYPE)
        distance_uplift_factor: Distance uplift for routing (GL_DTO_DISTANCE_UPLIFT_FACTOR)
        enable_return_trip: Include empty return leg (GL_DTO_DISTANCE_ENABLE_RETURN_TRIP)
        empty_return_factor: Load factor for empty return trip (GL_DTO_DISTANCE_EMPTY_RETURN_FACTOR)

    Example:
        >>> distance = DistanceConfig()
        >>> distance.max_distance_km
        50000
    """

    max_distance_km: int = 50000
    min_weight_tonnes: Decimal = Decimal("0.001")
    default_vehicle_type: str = "ARTICULATED_TRUCK"
    default_vessel_type: str = "CONTAINER_SHIP"
    default_aircraft_type: str = "FREIGHTER_MEDIUM"
    distance_uplift_factor: Decimal = Decimal("1.10")
    enable_return_trip: bool = False
    empty_return_factor: Decimal = Decimal("0.0")

    def validate(self) -> None:
        """
        Validate distance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.max_distance_km < 1 or self.max_distance_km > 100000:
            raise ValueError("max_distance_km must be between 1 and 100000")

        if self.min_weight_tonnes < Decimal("0") or self.min_weight_tonnes > Decimal("100"):
            raise ValueError("min_weight_tonnes must be between 0 and 100")

        valid_vehicle_types = {
            "ARTICULATED_TRUCK", "RIGID_TRUCK", "LCV", "HGV_AVERAGE",
            "VAN", "CONTAINER_TRUCK",
        }
        if self.default_vehicle_type not in valid_vehicle_types:
            raise ValueError(
                f"Invalid default_vehicle_type '{self.default_vehicle_type}'. "
                f"Must be one of {valid_vehicle_types}"
            )

        valid_vessel_types = {
            "CONTAINER_SHIP", "BULK_CARRIER", "TANKER", "GENERAL_CARGO",
            "RORO", "AVERAGE",
        }
        if self.default_vessel_type not in valid_vessel_types:
            raise ValueError(
                f"Invalid default_vessel_type '{self.default_vessel_type}'. "
                f"Must be one of {valid_vessel_types}"
            )

        valid_aircraft_types = {
            "FREIGHTER_SMALL", "FREIGHTER_MEDIUM", "FREIGHTER_LARGE",
            "BELLY_FREIGHT", "AVERAGE",
        }
        if self.default_aircraft_type not in valid_aircraft_types:
            raise ValueError(
                f"Invalid default_aircraft_type '{self.default_aircraft_type}'. "
                f"Must be one of {valid_aircraft_types}"
            )

        if self.distance_uplift_factor < Decimal("1.0") or self.distance_uplift_factor > Decimal("2.0"):
            raise ValueError("distance_uplift_factor must be between 1.0 and 2.0")

        if self.empty_return_factor < Decimal("0.0") or self.empty_return_factor > Decimal("1.0"):
            raise ValueError("empty_return_factor must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_distance_km": self.max_distance_km,
            "min_weight_tonnes": str(self.min_weight_tonnes),
            "default_vehicle_type": self.default_vehicle_type,
            "default_vessel_type": self.default_vessel_type,
            "default_aircraft_type": self.default_aircraft_type,
            "distance_uplift_factor": str(self.distance_uplift_factor),
            "enable_return_trip": self.enable_return_trip,
            "empty_return_factor": str(self.empty_return_factor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistanceConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["min_weight_tonnes", "distance_uplift_factor", "empty_return_factor"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "DistanceConfig":
        """Load from environment variables."""
        return cls(
            max_distance_km=int(os.getenv("GL_DTO_DISTANCE_MAX_DISTANCE_KM", "50000")),
            min_weight_tonnes=Decimal(
                os.getenv("GL_DTO_DISTANCE_MIN_WEIGHT_TONNES", "0.001")
            ),
            default_vehicle_type=os.getenv(
                "GL_DTO_DISTANCE_DEFAULT_VEHICLE_TYPE", "ARTICULATED_TRUCK"
            ),
            default_vessel_type=os.getenv(
                "GL_DTO_DISTANCE_DEFAULT_VESSEL_TYPE", "CONTAINER_SHIP"
            ),
            default_aircraft_type=os.getenv(
                "GL_DTO_DISTANCE_DEFAULT_AIRCRAFT_TYPE", "FREIGHTER_MEDIUM"
            ),
            distance_uplift_factor=Decimal(
                os.getenv("GL_DTO_DISTANCE_UPLIFT_FACTOR", "1.10")
            ),
            enable_return_trip=os.getenv(
                "GL_DTO_DISTANCE_ENABLE_RETURN_TRIP", "false"
            ).lower() == "true",
            empty_return_factor=Decimal(
                os.getenv("GL_DTO_DISTANCE_EMPTY_RETURN_FACTOR", "0.0")
            ),
        )


# =============================================================================
# SECTION 5: SPEND CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendConfig:
    """
    Spend-based calculation configuration for Downstream Transportation agent.

    Configures EEIO spend-based emission calculations with CPI deflation,
    margin removal, and currency conversion.

    Attributes:
        base_currency: Base currency for EEIO factors (GL_DTO_SPEND_BASE_CURRENCY)
        base_year: Base year for CPI deflation (GL_DTO_SPEND_BASE_YEAR)
        margin_removal_enabled: Enable profit margin removal (GL_DTO_SPEND_MARGIN_REMOVAL_ENABLED)
        margin_pct: Default profit margin percentage (GL_DTO_SPEND_MARGIN_PCT)
        enable_cpi_deflation: Enable CPI deflation (GL_DTO_SPEND_ENABLE_CPI_DEFLATION)
        supported_currencies: Comma-separated supported currencies (GL_DTO_SPEND_SUPPORTED_CURRENCIES)
        enable_ppp_adjustment: Enable purchasing power parity (GL_DTO_SPEND_ENABLE_PPP_ADJUSTMENT)
        default_naics_code: Default NAICS sector code (GL_DTO_SPEND_DEFAULT_NAICS_CODE)

    Example:
        >>> spend = SpendConfig()
        >>> spend.base_currency
        'USD'
    """

    base_currency: str = "USD"
    base_year: int = 2021
    margin_removal_enabled: bool = True
    margin_pct: Decimal = Decimal("0.15")
    enable_cpi_deflation: bool = True
    supported_currencies: str = (
        "USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED"
    )
    enable_ppp_adjustment: bool = False
    default_naics_code: str = "484000"

    def validate(self) -> None:
        """
        Validate spend configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.base_currency:
            raise ValueError("base_currency cannot be empty")

        if len(self.base_currency) != 3:
            raise ValueError("base_currency must be a 3-character ISO 4217 code")

        supported = self.get_supported_currencies()
        if self.base_currency not in supported:
            raise ValueError(
                f"base_currency '{self.base_currency}' must be in "
                f"supported_currencies list"
            )

        if self.base_year < 2000 or self.base_year > 2030:
            raise ValueError("base_year must be between 2000 and 2030")

        if self.margin_pct < Decimal("0") or self.margin_pct > Decimal("1"):
            raise ValueError("margin_pct must be between 0 and 1")

        if not self.supported_currencies:
            raise ValueError("supported_currencies cannot be empty")

        if not self.default_naics_code:
            raise ValueError("default_naics_code cannot be empty")

    def get_supported_currencies(self) -> List[str]:
        """Parse supported currencies string into list."""
        return [c.strip() for c in self.supported_currencies.split(",") if c.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_currency": self.base_currency,
            "base_year": self.base_year,
            "margin_removal_enabled": self.margin_removal_enabled,
            "margin_pct": str(self.margin_pct),
            "enable_cpi_deflation": self.enable_cpi_deflation,
            "supported_currencies": self.supported_currencies,
            "enable_ppp_adjustment": self.enable_ppp_adjustment,
            "default_naics_code": self.default_naics_code,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "margin_pct" in data_copy:
            data_copy["margin_pct"] = Decimal(data_copy["margin_pct"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendConfig":
        """Load from environment variables."""
        return cls(
            base_currency=os.getenv("GL_DTO_SPEND_BASE_CURRENCY", "USD"),
            base_year=int(os.getenv("GL_DTO_SPEND_BASE_YEAR", "2021")),
            margin_removal_enabled=os.getenv(
                "GL_DTO_SPEND_MARGIN_REMOVAL_ENABLED", "true"
            ).lower() == "true",
            margin_pct=Decimal(os.getenv("GL_DTO_SPEND_MARGIN_PCT", "0.15")),
            enable_cpi_deflation=os.getenv(
                "GL_DTO_SPEND_ENABLE_CPI_DEFLATION", "true"
            ).lower() == "true",
            supported_currencies=os.getenv(
                "GL_DTO_SPEND_SUPPORTED_CURRENCIES",
                "USD,EUR,GBP,JPY,CNY,AUD,CAD,CHF,INR,BRL,KRW,SGD,HKD,NZD,SEK,NOK,DKK,MXN,ZAR,AED",
            ),
            enable_ppp_adjustment=os.getenv(
                "GL_DTO_SPEND_ENABLE_PPP_ADJUSTMENT", "false"
            ).lower() == "true",
            default_naics_code=os.getenv("GL_DTO_SPEND_DEFAULT_NAICS_CODE", "484000"),
        )


# =============================================================================
# SECTION 6: WAREHOUSE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class WarehouseConfig:
    """
    Warehouse and distribution centre configuration.

    Configures emission calculations for distribution centres, cold storage
    facilities, and third-party retail storage (sub-activities 9b and 9c).

    Attributes:
        default_type: Default warehouse type (GL_DTO_WAREHOUSE_DEFAULT_TYPE)
        default_allocation: Default allocation method (GL_DTO_WAREHOUSE_DEFAULT_ALLOCATION)
        energy_source: Default energy source (GL_DTO_WAREHOUSE_ENERGY_SOURCE)
        grid_factor_country: Default country for grid EF (GL_DTO_WAREHOUSE_GRID_FACTOR_COUNTRY)
        include_refrigeration: Include refrigeration emissions (GL_DTO_WAREHOUSE_INCLUDE_REFRIGERATION)
        default_area_sqm: Default warehouse area in sqm (GL_DTO_WAREHOUSE_DEFAULT_AREA_SQM)
        default_throughput_tonnes: Default annual throughput (GL_DTO_WAREHOUSE_DEFAULT_THROUGHPUT_TONNES)
        include_material_handling: Include forklift/handling emissions (GL_DTO_WAREHOUSE_INCLUDE_MATERIAL_HANDLING)

    Example:
        >>> wh = WarehouseConfig()
        >>> wh.default_type
        'AMBIENT'
    """

    default_type: str = "AMBIENT"
    default_allocation: str = "MASS"
    energy_source: str = "GRID_ELECTRICITY"
    grid_factor_country: str = "US"
    include_refrigeration: bool = True
    default_area_sqm: Decimal = Decimal("10000")
    default_throughput_tonnes: Decimal = Decimal("50000")
    include_material_handling: bool = True

    def validate(self) -> None:
        """
        Validate warehouse configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_types = {"AMBIENT", "COLD_STORAGE", "FROZEN", "RETAIL", "CROSS_DOCK"}
        if self.default_type not in valid_types:
            raise ValueError(
                f"Invalid default_type '{self.default_type}'. "
                f"Must be one of {valid_types}"
            )

        valid_allocations = {"MASS", "VOLUME", "REVENUE", "UNIT_COUNT", "FLOOR_AREA"}
        if self.default_allocation not in valid_allocations:
            raise ValueError(
                f"Invalid default_allocation '{self.default_allocation}'. "
                f"Must be one of {valid_allocations}"
            )

        valid_sources = {"GRID_ELECTRICITY", "NATURAL_GAS", "MIXED", "RENEWABLE", "DIESEL"}
        if self.energy_source not in valid_sources:
            raise ValueError(
                f"Invalid energy_source '{self.energy_source}'. "
                f"Must be one of {valid_sources}"
            )

        if not self.grid_factor_country:
            raise ValueError("grid_factor_country cannot be empty")

        if len(self.grid_factor_country) < 2 or len(self.grid_factor_country) > 3:
            raise ValueError("grid_factor_country must be a 2 or 3 character ISO code")

        if self.default_area_sqm < Decimal("0"):
            raise ValueError("default_area_sqm must be >= 0")

        if self.default_throughput_tonnes < Decimal("0"):
            raise ValueError("default_throughput_tonnes must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_type": self.default_type,
            "default_allocation": self.default_allocation,
            "energy_source": self.energy_source,
            "grid_factor_country": self.grid_factor_country,
            "include_refrigeration": self.include_refrigeration,
            "default_area_sqm": str(self.default_area_sqm),
            "default_throughput_tonnes": str(self.default_throughput_tonnes),
            "include_material_handling": self.include_material_handling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarehouseConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_area_sqm", "default_throughput_tonnes"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "WarehouseConfig":
        """Load from environment variables."""
        return cls(
            default_type=os.getenv("GL_DTO_WAREHOUSE_DEFAULT_TYPE", "AMBIENT"),
            default_allocation=os.getenv("GL_DTO_WAREHOUSE_DEFAULT_ALLOCATION", "MASS"),
            energy_source=os.getenv("GL_DTO_WAREHOUSE_ENERGY_SOURCE", "GRID_ELECTRICITY"),
            grid_factor_country=os.getenv("GL_DTO_WAREHOUSE_GRID_FACTOR_COUNTRY", "US"),
            include_refrigeration=os.getenv(
                "GL_DTO_WAREHOUSE_INCLUDE_REFRIGERATION", "true"
            ).lower() == "true",
            default_area_sqm=Decimal(
                os.getenv("GL_DTO_WAREHOUSE_DEFAULT_AREA_SQM", "10000")
            ),
            default_throughput_tonnes=Decimal(
                os.getenv("GL_DTO_WAREHOUSE_DEFAULT_THROUGHPUT_TONNES", "50000")
            ),
            include_material_handling=os.getenv(
                "GL_DTO_WAREHOUSE_INCLUDE_MATERIAL_HANDLING", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 7: LAST MILE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class LastMileConfig:
    """
    Last-mile delivery configuration for Downstream Transportation agent.

    Configures final-mile delivery emissions calculations including
    delivery type, area classification, and failed delivery redelivery.

    Attributes:
        default_type: Default last-mile vehicle type (GL_DTO_LASTMILE_DEFAULT_TYPE)
        default_area: Default delivery area classification (GL_DTO_LASTMILE_DEFAULT_AREA)
        failed_delivery_rate: Failed delivery rate (GL_DTO_LASTMILE_FAILED_DELIVERY_RATE)
        redelivery_factor: Emission uplift for redeliveries (GL_DTO_LASTMILE_REDELIVERY_FACTOR)
        enable_parcel_lockers: Include parcel locker option (GL_DTO_LASTMILE_ENABLE_PARCEL_LOCKERS)
        locker_collection_km: Average consumer collection distance km (GL_DTO_LASTMILE_LOCKER_COLLECTION_KM)
        default_parcel_weight_kg: Default parcel weight (GL_DTO_LASTMILE_DEFAULT_PARCEL_WEIGHT_KG)
        max_delivery_distance_km: Maximum delivery distance (GL_DTO_LASTMILE_MAX_DELIVERY_DISTANCE_KM)

    Example:
        >>> lm = LastMileConfig()
        >>> lm.default_type
        'VAN_DIESEL'
    """

    default_type: str = "VAN_DIESEL"
    default_area: str = "URBAN"
    failed_delivery_rate: Decimal = Decimal("0.05")
    redelivery_factor: Decimal = Decimal("1.0")
    enable_parcel_lockers: bool = True
    locker_collection_km: Decimal = Decimal("2.0")
    default_parcel_weight_kg: Decimal = Decimal("5.0")
    max_delivery_distance_km: int = 200

    def validate(self) -> None:
        """
        Validate last-mile configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_types = {
            "VAN_DIESEL", "VAN_ELECTRIC", "CARGO_BIKE", "DRONE",
            "MOTORCYCLE", "CAR_PETROL", "CAR_ELECTRIC", "AVERAGE",
        }
        if self.default_type not in valid_types:
            raise ValueError(
                f"Invalid default_type '{self.default_type}'. "
                f"Must be one of {valid_types}"
            )

        valid_areas = {"URBAN", "SUBURBAN", "RURAL", "REMOTE"}
        if self.default_area not in valid_areas:
            raise ValueError(
                f"Invalid default_area '{self.default_area}'. "
                f"Must be one of {valid_areas}"
            )

        if self.failed_delivery_rate < Decimal("0") or self.failed_delivery_rate > Decimal("1"):
            raise ValueError("failed_delivery_rate must be between 0 and 1")

        if self.redelivery_factor < Decimal("0") or self.redelivery_factor > Decimal("5"):
            raise ValueError("redelivery_factor must be between 0 and 5")

        if self.locker_collection_km < Decimal("0") or self.locker_collection_km > Decimal("50"):
            raise ValueError("locker_collection_km must be between 0 and 50")

        if self.default_parcel_weight_kg < Decimal("0.01") or self.default_parcel_weight_kg > Decimal("1000"):
            raise ValueError("default_parcel_weight_kg must be between 0.01 and 1000")

        if self.max_delivery_distance_km < 1 or self.max_delivery_distance_km > 1000:
            raise ValueError("max_delivery_distance_km must be between 1 and 1000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_type": self.default_type,
            "default_area": self.default_area,
            "failed_delivery_rate": str(self.failed_delivery_rate),
            "redelivery_factor": str(self.redelivery_factor),
            "enable_parcel_lockers": self.enable_parcel_lockers,
            "locker_collection_km": str(self.locker_collection_km),
            "default_parcel_weight_kg": str(self.default_parcel_weight_kg),
            "max_delivery_distance_km": self.max_delivery_distance_km,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LastMileConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "failed_delivery_rate", "redelivery_factor",
            "locker_collection_km", "default_parcel_weight_kg",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "LastMileConfig":
        """Load from environment variables."""
        return cls(
            default_type=os.getenv("GL_DTO_LASTMILE_DEFAULT_TYPE", "VAN_DIESEL"),
            default_area=os.getenv("GL_DTO_LASTMILE_DEFAULT_AREA", "URBAN"),
            failed_delivery_rate=Decimal(
                os.getenv("GL_DTO_LASTMILE_FAILED_DELIVERY_RATE", "0.05")
            ),
            redelivery_factor=Decimal(
                os.getenv("GL_DTO_LASTMILE_REDELIVERY_FACTOR", "1.0")
            ),
            enable_parcel_lockers=os.getenv(
                "GL_DTO_LASTMILE_ENABLE_PARCEL_LOCKERS", "true"
            ).lower() == "true",
            locker_collection_km=Decimal(
                os.getenv("GL_DTO_LASTMILE_LOCKER_COLLECTION_KM", "2.0")
            ),
            default_parcel_weight_kg=Decimal(
                os.getenv("GL_DTO_LASTMILE_DEFAULT_PARCEL_WEIGHT_KG", "5.0")
            ),
            max_delivery_distance_km=int(
                os.getenv("GL_DTO_LASTMILE_MAX_DELIVERY_DISTANCE_KM", "200")
            ),
        )


# =============================================================================
# SECTION 8: AVERAGE DATA CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AverageDataConfig:
    """
    Average-data calculation configuration for Downstream Transportation agent.

    Configures industry-average emission calculations by distribution channel
    when shipment-level data is unavailable.

    Attributes:
        default_channel: Default distribution channel (GL_DTO_AVGDATA_DEFAULT_CHANNEL)
        use_channel_defaults: Use channel-level defaults (GL_DTO_AVGDATA_USE_CHANNEL_DEFAULTS)
        data_vintage_years: Max data age in years (GL_DTO_AVGDATA_DATA_VINTAGE_YEARS)
        default_region: Default region for average data (GL_DTO_AVGDATA_DEFAULT_REGION)
        include_warehousing: Include warehouse in avg data (GL_DTO_AVGDATA_INCLUDE_WAREHOUSING)
        include_last_mile: Include last mile in avg data (GL_DTO_AVGDATA_INCLUDE_LAST_MILE)

    Example:
        >>> avg = AverageDataConfig()
        >>> avg.default_channel
        'RETAIL'
    """

    default_channel: str = "RETAIL"
    use_channel_defaults: bool = True
    data_vintage_years: int = 3
    default_region: str = "GLOBAL"
    include_warehousing: bool = True
    include_last_mile: bool = True

    def validate(self) -> None:
        """
        Validate average data configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_channels = {
            "RETAIL", "WHOLESALE", "ECOMMERCE", "DIRECT_TO_CONSUMER",
            "DISTRIBUTOR", "FRANCHISE", "MIXED",
        }
        if self.default_channel not in valid_channels:
            raise ValueError(
                f"Invalid default_channel '{self.default_channel}'. "
                f"Must be one of {valid_channels}"
            )

        if self.data_vintage_years < 1 or self.data_vintage_years > 10:
            raise ValueError("data_vintage_years must be between 1 and 10")

        if not self.default_region:
            raise ValueError("default_region cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_channel": self.default_channel,
            "use_channel_defaults": self.use_channel_defaults,
            "data_vintage_years": self.data_vintage_years,
            "default_region": self.default_region,
            "include_warehousing": self.include_warehousing,
            "include_last_mile": self.include_last_mile,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AverageDataConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AverageDataConfig":
        """Load from environment variables."""
        return cls(
            default_channel=os.getenv("GL_DTO_AVGDATA_DEFAULT_CHANNEL", "RETAIL"),
            use_channel_defaults=os.getenv(
                "GL_DTO_AVGDATA_USE_CHANNEL_DEFAULTS", "true"
            ).lower() == "true",
            data_vintage_years=int(
                os.getenv("GL_DTO_AVGDATA_DATA_VINTAGE_YEARS", "3")
            ),
            default_region=os.getenv("GL_DTO_AVGDATA_DEFAULT_REGION", "GLOBAL"),
            include_warehousing=os.getenv(
                "GL_DTO_AVGDATA_INCLUDE_WAREHOUSING", "true"
            ).lower() == "true",
            include_last_mile=os.getenv(
                "GL_DTO_AVGDATA_INCLUDE_LAST_MILE", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 9: COLD CHAIN CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ColdChainConfig:
    """
    Cold chain configuration for Downstream Transportation agent.

    Configures temperature-controlled transport parameters including
    reefer uplift and refrigerant leakage tracking.

    Attributes:
        default_regime: Default temperature regime (GL_DTO_COLDCHAIN_DEFAULT_REGIME)
        uplift_enabled: Enable cold chain uplift factor (GL_DTO_COLDCHAIN_UPLIFT_ENABLED)
        reefer_fuel_surcharge: Reefer fuel surcharge multiplier (GL_DTO_COLDCHAIN_REEFER_FUEL_SURCHARGE)
        chilled_uplift: Chilled regime emission uplift (GL_DTO_COLDCHAIN_CHILLED_UPLIFT)
        frozen_uplift: Frozen regime emission uplift (GL_DTO_COLDCHAIN_FROZEN_UPLIFT)
        deep_frozen_uplift: Deep-frozen regime emission uplift (GL_DTO_COLDCHAIN_DEEP_FROZEN_UPLIFT)
        include_refrigerant_leakage: Track refrigerant leakage (GL_DTO_COLDCHAIN_INCLUDE_REFRIGERANT_LEAKAGE)
        default_refrigerant: Default refrigerant type (GL_DTO_COLDCHAIN_DEFAULT_REFRIGERANT)

    Example:
        >>> cc = ColdChainConfig()
        >>> cc.default_regime
        'AMBIENT'
    """

    default_regime: str = "AMBIENT"
    uplift_enabled: bool = True
    reefer_fuel_surcharge: Decimal = Decimal("1.20")
    chilled_uplift: Decimal = Decimal("1.15")
    frozen_uplift: Decimal = Decimal("1.30")
    deep_frozen_uplift: Decimal = Decimal("1.50")
    include_refrigerant_leakage: bool = True
    default_refrigerant: str = "R404A"

    def validate(self) -> None:
        """
        Validate cold chain configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_regimes = {"AMBIENT", "CHILLED", "FROZEN", "DEEP_FROZEN", "PHARMA"}
        if self.default_regime not in valid_regimes:
            raise ValueError(
                f"Invalid default_regime '{self.default_regime}'. "
                f"Must be one of {valid_regimes}"
            )

        if self.reefer_fuel_surcharge < Decimal("1.0") or self.reefer_fuel_surcharge > Decimal("3.0"):
            raise ValueError("reefer_fuel_surcharge must be between 1.0 and 3.0")

        if self.chilled_uplift < Decimal("1.0") or self.chilled_uplift > Decimal("3.0"):
            raise ValueError("chilled_uplift must be between 1.0 and 3.0")

        if self.frozen_uplift < Decimal("1.0") or self.frozen_uplift > Decimal("3.0"):
            raise ValueError("frozen_uplift must be between 1.0 and 3.0")

        if self.deep_frozen_uplift < Decimal("1.0") or self.deep_frozen_uplift > Decimal("5.0"):
            raise ValueError("deep_frozen_uplift must be between 1.0 and 5.0")

        valid_refrigerants = {"R404A", "R134A", "R410A", "R744", "R290", "R600A", "NONE"}
        if self.default_refrigerant not in valid_refrigerants:
            raise ValueError(
                f"Invalid default_refrigerant '{self.default_refrigerant}'. "
                f"Must be one of {valid_refrigerants}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_regime": self.default_regime,
            "uplift_enabled": self.uplift_enabled,
            "reefer_fuel_surcharge": str(self.reefer_fuel_surcharge),
            "chilled_uplift": str(self.chilled_uplift),
            "frozen_uplift": str(self.frozen_uplift),
            "deep_frozen_uplift": str(self.deep_frozen_uplift),
            "include_refrigerant_leakage": self.include_refrigerant_leakage,
            "default_refrigerant": self.default_refrigerant,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColdChainConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "reefer_fuel_surcharge", "chilled_uplift",
            "frozen_uplift", "deep_frozen_uplift",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ColdChainConfig":
        """Load from environment variables."""
        return cls(
            default_regime=os.getenv("GL_DTO_COLDCHAIN_DEFAULT_REGIME", "AMBIENT"),
            uplift_enabled=os.getenv(
                "GL_DTO_COLDCHAIN_UPLIFT_ENABLED", "true"
            ).lower() == "true",
            reefer_fuel_surcharge=Decimal(
                os.getenv("GL_DTO_COLDCHAIN_REEFER_FUEL_SURCHARGE", "1.20")
            ),
            chilled_uplift=Decimal(
                os.getenv("GL_DTO_COLDCHAIN_CHILLED_UPLIFT", "1.15")
            ),
            frozen_uplift=Decimal(
                os.getenv("GL_DTO_COLDCHAIN_FROZEN_UPLIFT", "1.30")
            ),
            deep_frozen_uplift=Decimal(
                os.getenv("GL_DTO_COLDCHAIN_DEEP_FROZEN_UPLIFT", "1.50")
            ),
            include_refrigerant_leakage=os.getenv(
                "GL_DTO_COLDCHAIN_INCLUDE_REFRIGERANT_LEAKAGE", "true"
            ).lower() == "true",
            default_refrigerant=os.getenv(
                "GL_DTO_COLDCHAIN_DEFAULT_REFRIGERANT", "R404A"
            ),
        )


# =============================================================================
# SECTION 10: RETURN LOGISTICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ReturnLogisticsConfig:
    """
    Return logistics configuration for Downstream Transportation agent.

    Configures reverse logistics emission tracking for product returns.

    Attributes:
        include_returns: Include return trip emissions (GL_DTO_RETURNS_INCLUDE_RETURNS)
        default_return_rate: Default product return rate (GL_DTO_RETURNS_DEFAULT_RETURN_RATE)
        consolidation_factor: Consolidation efficiency factor (GL_DTO_RETURNS_CONSOLIDATION_FACTOR)
        return_mode: Default return transport mode (GL_DTO_RETURNS_RETURN_MODE)
        include_repackaging: Include repackaging emissions (GL_DTO_RETURNS_INCLUDE_REPACKAGING)
        repackaging_factor_kg: Emission factor per return repackaging (GL_DTO_RETURNS_REPACKAGING_FACTOR_KG)

    Example:
        >>> ret = ReturnLogisticsConfig()
        >>> ret.default_return_rate
        Decimal('0.10')
    """

    include_returns: bool = True
    default_return_rate: Decimal = Decimal("0.10")
    consolidation_factor: Decimal = Decimal("0.80")
    return_mode: str = "ROAD"
    include_repackaging: bool = False
    repackaging_factor_kg: Decimal = Decimal("0.05")

    def validate(self) -> None:
        """
        Validate return logistics configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_return_rate < Decimal("0") or self.default_return_rate > Decimal("1"):
            raise ValueError("default_return_rate must be between 0 and 1")

        if self.consolidation_factor < Decimal("0.1") or self.consolidation_factor > Decimal("1"):
            raise ValueError("consolidation_factor must be between 0.1 and 1")

        valid_modes = {"ROAD", "RAIL", "MARITIME", "AIR", "INTERMODAL"}
        if self.return_mode not in valid_modes:
            raise ValueError(
                f"Invalid return_mode '{self.return_mode}'. "
                f"Must be one of {valid_modes}"
            )

        if self.repackaging_factor_kg < Decimal("0"):
            raise ValueError("repackaging_factor_kg must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "include_returns": self.include_returns,
            "default_return_rate": str(self.default_return_rate),
            "consolidation_factor": str(self.consolidation_factor),
            "return_mode": self.return_mode,
            "include_repackaging": self.include_repackaging,
            "repackaging_factor_kg": str(self.repackaging_factor_kg),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReturnLogisticsConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_return_rate", "consolidation_factor", "repackaging_factor_kg"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ReturnLogisticsConfig":
        """Load from environment variables."""
        return cls(
            include_returns=os.getenv(
                "GL_DTO_RETURNS_INCLUDE_RETURNS", "true"
            ).lower() == "true",
            default_return_rate=Decimal(
                os.getenv("GL_DTO_RETURNS_DEFAULT_RETURN_RATE", "0.10")
            ),
            consolidation_factor=Decimal(
                os.getenv("GL_DTO_RETURNS_CONSOLIDATION_FACTOR", "0.80")
            ),
            return_mode=os.getenv("GL_DTO_RETURNS_RETURN_MODE", "ROAD"),
            include_repackaging=os.getenv(
                "GL_DTO_RETURNS_INCLUDE_REPACKAGING", "false"
            ).lower() == "true",
            repackaging_factor_kg=Decimal(
                os.getenv("GL_DTO_RETURNS_REPACKAGING_FACTOR_KG", "0.05")
            ),
        )


# =============================================================================
# SECTION 11: ALLOCATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AllocationConfig:
    """
    Allocation configuration for Downstream Transportation agent.

    Configures how shared transport/warehouse emissions are allocated
    to individual products or shipments.

    Attributes:
        default_method: Default allocation method (GL_DTO_ALLOCATION_DEFAULT_METHOD)
        product_level_enabled: Enable product-level allocation (GL_DTO_ALLOCATION_PRODUCT_LEVEL_ENABLED)
        allocation_threshold: Minimum allocation share (GL_DTO_ALLOCATION_THRESHOLD)
        mass_unit: Mass unit for mass-based allocation (GL_DTO_ALLOCATION_MASS_UNIT)
        volume_unit: Volume unit for volume-based allocation (GL_DTO_ALLOCATION_VOLUME_UNIT)
        revenue_currency: Currency for revenue-based allocation (GL_DTO_ALLOCATION_REVENUE_CURRENCY)

    Example:
        >>> alloc = AllocationConfig()
        >>> alloc.default_method
        'MASS'
    """

    default_method: str = "MASS"
    product_level_enabled: bool = True
    allocation_threshold: Decimal = Decimal("0.001")
    mass_unit: str = "KG"
    volume_unit: str = "CBM"
    revenue_currency: str = "USD"

    def validate(self) -> None:
        """
        Validate allocation configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"MASS", "VOLUME", "REVENUE", "UNIT_COUNT", "TONNE_KM"}
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.allocation_threshold < Decimal("0") or self.allocation_threshold > Decimal("1"):
            raise ValueError("allocation_threshold must be between 0 and 1")

        valid_mass_units = {"KG", "TONNE", "LB"}
        if self.mass_unit not in valid_mass_units:
            raise ValueError(
                f"Invalid mass_unit '{self.mass_unit}'. "
                f"Must be one of {valid_mass_units}"
            )

        valid_volume_units = {"CBM", "LITRE", "CUFT"}
        if self.volume_unit not in valid_volume_units:
            raise ValueError(
                f"Invalid volume_unit '{self.volume_unit}'. "
                f"Must be one of {valid_volume_units}"
            )

        if len(self.revenue_currency) != 3:
            raise ValueError("revenue_currency must be a 3-character ISO 4217 code")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "product_level_enabled": self.product_level_enabled,
            "allocation_threshold": str(self.allocation_threshold),
            "mass_unit": self.mass_unit,
            "volume_unit": self.volume_unit,
            "revenue_currency": self.revenue_currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "allocation_threshold" in data_copy:
            data_copy["allocation_threshold"] = Decimal(data_copy["allocation_threshold"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "AllocationConfig":
        """Load from environment variables."""
        return cls(
            default_method=os.getenv("GL_DTO_ALLOCATION_DEFAULT_METHOD", "MASS"),
            product_level_enabled=os.getenv(
                "GL_DTO_ALLOCATION_PRODUCT_LEVEL_ENABLED", "true"
            ).lower() == "true",
            allocation_threshold=Decimal(
                os.getenv("GL_DTO_ALLOCATION_THRESHOLD", "0.001")
            ),
            mass_unit=os.getenv("GL_DTO_ALLOCATION_MASS_UNIT", "KG"),
            volume_unit=os.getenv("GL_DTO_ALLOCATION_VOLUME_UNIT", "CBM"),
            revenue_currency=os.getenv("GL_DTO_ALLOCATION_REVENUE_CURRENCY", "USD"),
        )


# =============================================================================
# SECTION 12: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Downstream Transportation agent.

    Configures regulatory framework compliance checks for Scope 3
    Category 9 downstream transportation emissions reporting.

    Attributes:
        enabled_frameworks: Comma-separated enabled frameworks (GL_DTO_COMPLIANCE_ENABLED_FRAMEWORKS)
        strict_mode: Enforce strict compliance (GL_DTO_COMPLIANCE_STRICT_MODE)
        minimum_score: Minimum compliance score (GL_DTO_COMPLIANCE_MINIMUM_SCORE)
        double_counting_check: Check for double counting vs Cat 4 (GL_DTO_COMPLIANCE_DOUBLE_COUNTING_CHECK)
        boundary_enforcement: Enforce Scope 3 Cat 9 boundary (GL_DTO_COMPLIANCE_BOUNDARY_ENFORCEMENT)
        require_incoterm: Require Incoterm for Cat 4/9 split (GL_DTO_COMPLIANCE_REQUIRE_INCOTERM)
        require_data_quality: Require DQI scoring (GL_DTO_COMPLIANCE_REQUIRE_DATA_QUALITY)
        min_data_quality_score: Minimum DQI score (GL_DTO_COMPLIANCE_MIN_DATA_QUALITY_SCORE)

    Example:
        >>> compliance = ComplianceConfig()
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E1', 'CDP', 'SBTI', 'GRI', 'GLEC']
    """

    enabled_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,GLEC"
    )
    strict_mode: bool = False
    minimum_score: Decimal = Decimal("0.60")
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    require_incoterm: bool = True
    require_data_quality: bool = True
    min_data_quality_score: Decimal = Decimal("2.0")

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
            "CDP", "SBTI", "GRI", "GLEC",
        }

        frameworks = self.get_frameworks()
        if not frameworks:
            raise ValueError("At least one compliance framework must be enabled")

        for framework in frameworks:
            if framework not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{framework}'. Must be one of {valid_frameworks}"
                )

        if self.minimum_score < Decimal("0") or self.minimum_score > Decimal("1"):
            raise ValueError("minimum_score must be between 0 and 1")

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
            "minimum_score": str(self.minimum_score),
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "require_incoterm": self.require_incoterm,
            "require_data_quality": self.require_data_quality,
            "min_data_quality_score": str(self.min_data_quality_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["minimum_score", "min_data_quality_score"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            enabled_frameworks=os.getenv(
                "GL_DTO_COMPLIANCE_ENABLED_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,GRI,GLEC",
            ),
            strict_mode=os.getenv(
                "GL_DTO_COMPLIANCE_STRICT_MODE", "false"
            ).lower() == "true",
            minimum_score=Decimal(
                os.getenv("GL_DTO_COMPLIANCE_MINIMUM_SCORE", "0.60")
            ),
            double_counting_check=os.getenv(
                "GL_DTO_COMPLIANCE_DOUBLE_COUNTING_CHECK", "true"
            ).lower() == "true",
            boundary_enforcement=os.getenv(
                "GL_DTO_COMPLIANCE_BOUNDARY_ENFORCEMENT", "true"
            ).lower() == "true",
            require_incoterm=os.getenv(
                "GL_DTO_COMPLIANCE_REQUIRE_INCOTERM", "true"
            ).lower() == "true",
            require_data_quality=os.getenv(
                "GL_DTO_COMPLIANCE_REQUIRE_DATA_QUALITY", "true"
            ).lower() == "true",
            min_data_quality_score=Decimal(
                os.getenv("GL_DTO_COMPLIANCE_MIN_DATA_QUALITY_SCORE", "2.0")
            ),
        )


# =============================================================================
# SECTION 13: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance tracking configuration for Downstream Transportation agent.

    Attributes:
        hash_algorithm: Hash algorithm (GL_DTO_PROVENANCE_HASH_ALGORITHM)
        chain_enabled: Enable chain hashing (GL_DTO_PROVENANCE_CHAIN_ENABLED)
        merkle_tree_enabled: Enable Merkle tree for batches (GL_DTO_PROVENANCE_MERKLE_TREE_ENABLED)
        store_intermediate: Store intermediate hashes (GL_DTO_PROVENANCE_STORE_INTERMEDIATE)
        include_config_hash: Include config hash (GL_DTO_PROVENANCE_INCLUDE_CONFIG_HASH)
        include_ef_hash: Include EF data hash (GL_DTO_PROVENANCE_INCLUDE_EF_HASH)

    Example:
        >>> prov = ProvenanceConfig()
        >>> prov.hash_algorithm
        'sha256'
    """

    hash_algorithm: str = "sha256"
    chain_enabled: bool = True
    merkle_tree_enabled: bool = True
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
            "merkle_tree_enabled": self.merkle_tree_enabled,
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
            hash_algorithm=os.getenv("GL_DTO_PROVENANCE_HASH_ALGORITHM", "sha256"),
            chain_enabled=os.getenv(
                "GL_DTO_PROVENANCE_CHAIN_ENABLED", "true"
            ).lower() == "true",
            merkle_tree_enabled=os.getenv(
                "GL_DTO_PROVENANCE_MERKLE_TREE_ENABLED", "true"
            ).lower() == "true",
            store_intermediate=os.getenv(
                "GL_DTO_PROVENANCE_STORE_INTERMEDIATE", "true"
            ).lower() == "true",
            include_config_hash=os.getenv(
                "GL_DTO_PROVENANCE_INCLUDE_CONFIG_HASH", "true"
            ).lower() == "true",
            include_ef_hash=os.getenv(
                "GL_DTO_PROVENANCE_INCLUDE_EF_HASH", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 14: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty quantification configuration for Downstream Transportation agent.

    Attributes:
        method: Default uncertainty method (GL_DTO_UNCERTAINTY_METHOD)
        monte_carlo_iterations: Monte Carlo iterations (GL_DTO_UNCERTAINTY_MONTE_CARLO_ITERATIONS)
        confidence_level: Confidence level (GL_DTO_UNCERTAINTY_CONFIDENCE_LEVEL)
        include_parameter_uncertainty: Include parameter uncertainty (GL_DTO_UNCERTAINTY_INCLUDE_PARAMETER)
        include_model_uncertainty: Include model uncertainty (GL_DTO_UNCERTAINTY_INCLUDE_MODEL)
        seed: Random seed for reproducibility (GL_DTO_UNCERTAINTY_SEED)

    Example:
        >>> unc = UncertaintyConfig()
        >>> unc.method
        'MONTE_CARLO'
    """

    method: str = "MONTE_CARLO"
    monte_carlo_iterations: int = 10000
    confidence_level: Decimal = Decimal("0.95")
    include_parameter_uncertainty: bool = True
    include_model_uncertainty: bool = False
    seed: Optional[int] = None

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"MONTE_CARLO", "BOOTSTRAP", "IPCC_DEFAULT", "ANALYTICAL", "NONE"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.monte_carlo_iterations < 100 or self.monte_carlo_iterations > 1000000:
            raise ValueError("monte_carlo_iterations must be between 100 and 1000000")

        if self.confidence_level < Decimal("0.5") or self.confidence_level > Decimal("0.999"):
            raise ValueError("confidence_level must be between 0.5 and 0.999")

        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be >= 0 when specified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "confidence_level": str(self.confidence_level),
            "include_parameter_uncertainty": self.include_parameter_uncertainty,
            "include_model_uncertainty": self.include_model_uncertainty,
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
        seed_str = os.getenv("GL_DTO_UNCERTAINTY_SEED")
        seed_val = int(seed_str) if seed_str else None
        return cls(
            method=os.getenv("GL_DTO_UNCERTAINTY_METHOD", "MONTE_CARLO"),
            monte_carlo_iterations=int(
                os.getenv("GL_DTO_UNCERTAINTY_MONTE_CARLO_ITERATIONS", "10000")
            ),
            confidence_level=Decimal(
                os.getenv("GL_DTO_UNCERTAINTY_CONFIDENCE_LEVEL", "0.95")
            ),
            include_parameter_uncertainty=os.getenv(
                "GL_DTO_UNCERTAINTY_INCLUDE_PARAMETER", "true"
            ).lower() == "true",
            include_model_uncertainty=os.getenv(
                "GL_DTO_UNCERTAINTY_INCLUDE_MODEL", "false"
            ).lower() == "true",
            seed=seed_val,
        )


# =============================================================================
# SECTION 15: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Downstream Transportation agent.

    Attributes:
        enabled: Enable caching (GL_DTO_CACHE_ENABLED)
        ttl_seconds: Cache TTL in seconds (GL_DTO_CACHE_TTL_SECONDS)
        max_entries: Max cache entries (GL_DTO_CACHE_MAX_ENTRIES)
        key_prefix: Cache key prefix (GL_DTO_CACHE_KEY_PREFIX)
        cache_ef_lookups: Cache emission factor lookups (GL_DTO_CACHE_EF_LOOKUPS)
        cache_calculations: Cache calculation results (GL_DTO_CACHE_CALCULATIONS)
        eviction_policy: Cache eviction policy (GL_DTO_CACHE_EVICTION_POLICY)

    Example:
        >>> cache = CacheConfig()
        >>> cache.ttl_seconds
        3600
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 10000
    key_prefix: str = "gl_dto:"
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
            raise ValueError("ttl_seconds must be between 0 and 86400")

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
            enabled=os.getenv("GL_DTO_CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("GL_DTO_CACHE_TTL_SECONDS", "3600")),
            max_entries=int(os.getenv("GL_DTO_CACHE_MAX_ENTRIES", "10000")),
            key_prefix=os.getenv("GL_DTO_CACHE_KEY_PREFIX", "gl_dto:"),
            cache_ef_lookups=os.getenv(
                "GL_DTO_CACHE_EF_LOOKUPS", "true"
            ).lower() == "true",
            cache_calculations=os.getenv(
                "GL_DTO_CACHE_CALCULATIONS", "true"
            ).lower() == "true",
            eviction_policy=os.getenv("GL_DTO_CACHE_EVICTION_POLICY", "LRU"),
        )


# =============================================================================
# SECTION 16: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Downstream Transportation agent.

    Attributes:
        host: Redis host (GL_DTO_REDIS_HOST)
        port: Redis port (GL_DTO_REDIS_PORT)
        db: Redis database index (GL_DTO_REDIS_DB)
        password: Redis password (GL_DTO_REDIS_PASSWORD)
        ssl: Enable SSL (GL_DTO_REDIS_SSL)
        prefix: Key prefix (GL_DTO_REDIS_PREFIX)
        max_connections: Max pool connections (GL_DTO_REDIS_MAX_CONNECTIONS)
        socket_timeout: Socket timeout seconds (GL_DTO_REDIS_SOCKET_TIMEOUT)

    Example:
        >>> redis = RedisConfig()
        >>> redis.prefix
        'gl_dto:'
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    prefix: str = "gl_dto:"
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
            raise ValueError("socket_timeout must be between 1 and 60")

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
            host=os.getenv("GL_DTO_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_DTO_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_DTO_REDIS_DB", "0")),
            password=os.getenv("GL_DTO_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_DTO_REDIS_SSL", "false").lower() == "true",
            prefix=os.getenv("GL_DTO_REDIS_PREFIX", "gl_dto:"),
            max_connections=int(os.getenv("GL_DTO_REDIS_MAX_CONNECTIONS", "20")),
            socket_timeout=int(os.getenv("GL_DTO_REDIS_SOCKET_TIMEOUT", "5")),
        )


# =============================================================================
# SECTION 17: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    API configuration for Downstream Transportation agent.

    Attributes:
        rate_limit: Requests per minute per tenant (GL_DTO_API_RATE_LIMIT)
        timeout_seconds: Request timeout seconds (GL_DTO_API_TIMEOUT_SECONDS)
        enable_batch: Enable batch endpoint (GL_DTO_API_ENABLE_BATCH)
        max_concurrent: Max concurrent requests (GL_DTO_API_MAX_CONCURRENT)
        max_batch_size: Max records per batch (GL_DTO_API_MAX_BATCH_SIZE)
        cors_origins: Comma-separated CORS origins (GL_DTO_API_CORS_ORIGINS)
        worker_threads: Worker thread count (GL_DTO_API_WORKER_THREADS)

    Example:
        >>> api = APIConfig()
        >>> api.rate_limit
        100
    """

    rate_limit: int = 100
    timeout_seconds: int = 30
    enable_batch: bool = True
    max_concurrent: int = 10
    max_batch_size: int = 500
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
            raise ValueError("timeout_seconds must be between 1 and 3600")

        if self.max_concurrent < 1 or self.max_concurrent > 1000:
            raise ValueError("max_concurrent must be between 1 and 1000")

        if self.max_batch_size < 1 or self.max_batch_size > 10000:
            raise ValueError("max_batch_size must be between 1 and 10000")

        if not self.cors_origins:
            raise ValueError("cors_origins cannot be empty")

        if self.worker_threads < 1 or self.worker_threads > 64:
            raise ValueError("worker_threads must be between 1 and 64")

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins into list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rate_limit": self.rate_limit,
            "timeout_seconds": self.timeout_seconds,
            "enable_batch": self.enable_batch,
            "max_concurrent": self.max_concurrent,
            "max_batch_size": self.max_batch_size,
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
            rate_limit=int(os.getenv("GL_DTO_API_RATE_LIMIT", "100")),
            timeout_seconds=int(os.getenv("GL_DTO_API_TIMEOUT_SECONDS", "30")),
            enable_batch=os.getenv(
                "GL_DTO_API_ENABLE_BATCH", "true"
            ).lower() == "true",
            max_concurrent=int(os.getenv("GL_DTO_API_MAX_CONCURRENT", "10")),
            max_batch_size=int(os.getenv("GL_DTO_API_MAX_BATCH_SIZE", "500")),
            cors_origins=os.getenv("GL_DTO_API_CORS_ORIGINS", "*"),
            worker_threads=int(os.getenv("GL_DTO_API_WORKER_THREADS", "4")),
        )


# =============================================================================
# SECTION 18: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Downstream Transportation agent.

    Attributes:
        enabled: Enable metrics collection (GL_DTO_METRICS_ENABLED)
        prefix: Metrics prefix (GL_DTO_METRICS_PREFIX)
        include_tenant: Include tenant label (GL_DTO_METRICS_INCLUDE_TENANT)
        histogram_buckets: Comma-separated histogram buckets (GL_DTO_METRICS_HISTOGRAM_BUCKETS)
        enable_latency_tracking: Track per-engine latency (GL_DTO_METRICS_ENABLE_LATENCY_TRACKING)
        enable_error_counting: Track error counts (GL_DTO_METRICS_ENABLE_ERROR_COUNTING)

    Example:
        >>> metrics = MetricsConfig()
        >>> metrics.prefix
        'gl_dto'
    """

    enabled: bool = True
    prefix: str = "gl_dto"
    include_tenant: bool = True
    histogram_buckets: str = "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
    enable_latency_tracking: bool = True
    enable_error_counting: bool = True

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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_DTO_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_DTO_METRICS_PREFIX", "gl_dto"),
            include_tenant=os.getenv(
                "GL_DTO_METRICS_INCLUDE_TENANT", "true"
            ).lower() == "true",
            histogram_buckets=os.getenv(
                "GL_DTO_METRICS_HISTOGRAM_BUCKETS",
                "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
            enable_latency_tracking=os.getenv(
                "GL_DTO_METRICS_ENABLE_LATENCY_TRACKING", "true"
            ).lower() == "true",
            enable_error_counting=os.getenv(
                "GL_DTO_METRICS_ENABLE_ERROR_COUNTING", "true"
            ).lower() == "true",
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass(frozen=True)
class DownstreamTransportConfig:
    """
    Master configuration class for Downstream Transportation agent (AGENT-MRV-022).

    Aggregates all 18 configuration sections and provides a unified interface
    for accessing configuration values. Implements thread-safe singleton.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        transport: Transport mode configuration
        distance: Distance-based calculation configuration
        spend: Spend-based calculation configuration
        warehouse: Warehouse and distribution configuration
        last_mile: Last-mile delivery configuration
        average_data: Average-data calculation configuration
        cold_chain: Cold chain configuration
        return_logistics: Return logistics configuration
        allocation: Allocation configuration
        compliance: Regulatory compliance configuration
        provenance: Data provenance configuration
        uncertainty: Uncertainty quantification configuration
        cache: Cache configuration
        redis: Redis configuration
        api: REST API configuration
        metrics: Metrics configuration

    Example:
        >>> config = DownstreamTransportConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-009'
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    spend: SpendConfig = field(default_factory=SpendConfig)
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    last_mile: LastMileConfig = field(default_factory=LastMileConfig)
    average_data: AverageDataConfig = field(default_factory=AverageDataConfig)
    cold_chain: ColdChainConfig = field(default_factory=ColdChainConfig)
    return_logistics: ReturnLogisticsConfig = field(default_factory=ReturnLogisticsConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

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
            ("transport", self.transport),
            ("distance", self.distance),
            ("spend", self.spend),
            ("warehouse", self.warehouse),
            ("last_mile", self.last_mile),
            ("average_data", self.average_data),
            ("cold_chain", self.cold_chain),
            ("return_logistics", self.return_logistics),
            ("allocation", self.allocation),
            ("compliance", self.compliance),
            ("provenance", self.provenance),
            ("uncertainty", self.uncertainty),
            ("cache", self.cache),
            ("redis", self.redis),
            ("api", self.api),
            ("metrics", self.metrics),
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

        if self.cache.enabled and self.cache.key_prefix != self.redis.prefix:
            logger.warning(
                "cache.key_prefix '%s' differs from redis.prefix '%s'",
                self.cache.key_prefix,
                self.redis.prefix,
            )

        if self.api.max_batch_size > self.general.max_batch_size:
            errors.append(
                "cross-validation: api.max_batch_size should not exceed "
                "general.max_batch_size"
            )

        if self.database.pool_size > self.database.max_overflow + self.database.pool_size:
            # This is always false but included for pattern consistency
            pass

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "transport": self.transport.to_dict(),
            "distance": self.distance.to_dict(),
            "spend": self.spend.to_dict(),
            "warehouse": self.warehouse.to_dict(),
            "last_mile": self.last_mile.to_dict(),
            "average_data": self.average_data.to_dict(),
            "cold_chain": self.cold_chain.to_dict(),
            "return_logistics": self.return_logistics.to_dict(),
            "allocation": self.allocation.to_dict(),
            "compliance": self.compliance.to_dict(),
            "provenance": self.provenance.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "cache": self.cache.to_dict(),
            "redis": self.redis.to_dict(),
            "api": self.api.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownstreamTransportConfig":
        """Create configuration from dictionary."""
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            transport=TransportConfig.from_dict(data.get("transport", {})),
            distance=DistanceConfig.from_dict(data.get("distance", {})),
            spend=SpendConfig.from_dict(data.get("spend", {})),
            warehouse=WarehouseConfig.from_dict(data.get("warehouse", {})),
            last_mile=LastMileConfig.from_dict(data.get("last_mile", {})),
            average_data=AverageDataConfig.from_dict(data.get("average_data", {})),
            cold_chain=ColdChainConfig.from_dict(data.get("cold_chain", {})),
            return_logistics=ReturnLogisticsConfig.from_dict(data.get("return_logistics", {})),
            allocation=AllocationConfig.from_dict(data.get("allocation", {})),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})),
            cache=CacheConfig.from_dict(data.get("cache", {})),
            redis=RedisConfig.from_dict(data.get("redis", {})),
            api=APIConfig.from_dict(data.get("api", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
        )

    @classmethod
    def from_env(cls) -> "DownstreamTransportConfig":
        """Load configuration from environment variables."""
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            transport=TransportConfig.from_env(),
            distance=DistanceConfig.from_env(),
            spend=SpendConfig.from_env(),
            warehouse=WarehouseConfig.from_env(),
            last_mile=LastMileConfig.from_env(),
            average_data=AverageDataConfig.from_env(),
            cold_chain=ColdChainConfig.from_env(),
            return_logistics=ReturnLogisticsConfig.from_env(),
            allocation=AllocationConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            cache=CacheConfig.from_env(),
            redis=RedisConfig.from_env(),
            api=APIConfig.from_env(),
            metrics=MetricsConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================

_config_instance: Optional[DownstreamTransportConfig] = None
_config_lock = threading.RLock()


def get_config() -> DownstreamTransportConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization with double-checked locking.

    Returns:
        DownstreamTransportConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-009'
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info("Initializing DownstreamTransportConfig from environment")
                config = DownstreamTransportConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning("Configuration validation warning: %s", error)
                _config_instance = config
                logger.info(
                    "DownstreamTransportConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: DownstreamTransportConfig) -> None:
    """
    Set the singleton configuration instance.

    Args:
        config: DownstreamTransportConfig instance to set

    Raises:
        TypeError: If config is not a DownstreamTransportConfig instance
    """
    global _config_instance

    if not isinstance(config, DownstreamTransportConfig):
        raise TypeError(
            f"config must be a DownstreamTransportConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("DownstreamTransportConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Forces next get_config() call to reload from environment.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("DownstreamTransportConfig singleton reset")


def validate_config(config: DownstreamTransportConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate_all()


def print_config(config: DownstreamTransportConfig) -> None:
    """
    Print configuration in human-readable format.

    Sensitive fields (passwords) are redacted for security.

    Args:
        config: Configuration instance to print
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
    print("  Downstream Transportation Configuration (AGENT-MRV-022)")
    print("  Agent ID: " + config.general.agent_id)
    print("  Version:  " + config.general.version)
    print("=" * 64)

    _print_section("GENERAL", config.general.to_dict())
    _print_section("DATABASE", config.database.to_dict())
    _print_section("TRANSPORT", config.transport.to_dict())
    _print_section("DISTANCE", config.distance.to_dict())
    _print_section("SPEND", config.spend.to_dict())
    _print_section("WAREHOUSE", config.warehouse.to_dict())
    _print_section("LAST_MILE", config.last_mile.to_dict())
    _print_section("AVERAGE_DATA", config.average_data.to_dict())
    _print_section("COLD_CHAIN", config.cold_chain.to_dict())
    _print_section("RETURN_LOGISTICS", config.return_logistics.to_dict())
    _print_section("ALLOCATION", config.allocation.to_dict())
    _print_section("COMPLIANCE", config.compliance.to_dict())
    _print_section("PROVENANCE", config.provenance.to_dict())
    _print_section("UNCERTAINTY", config.uncertainty.to_dict())
    _print_section("CACHE", config.cache.to_dict())
    _print_section("REDIS", config.redis.to_dict())
    _print_section("API", config.api.to_dict())
    _print_section("METRICS", config.metrics.to_dict())

    # Validation summary
    errors = config.validate_all()
    if errors:
        print(f"\nValidation Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nValidation: ALL SECTIONS PASS")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Config sections
    "GeneralConfig",
    "DatabaseConfig",
    "TransportConfig",
    "DistanceConfig",
    "SpendConfig",
    "WarehouseConfig",
    "LastMileConfig",
    "AverageDataConfig",
    "ColdChainConfig",
    "ReturnLogisticsConfig",
    "AllocationConfig",
    "ComplianceConfig",
    "ProvenanceConfig",
    "UncertaintyConfig",
    "CacheConfig",
    "RedisConfig",
    "APIConfig",
    "MetricsConfig",
    # Master config
    "DownstreamTransportConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    "validate_config",
    "print_config",
]
