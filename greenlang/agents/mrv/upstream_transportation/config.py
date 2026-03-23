# -*- coding: utf-8 -*-
"""
Upstream Transportation & Distribution Configuration - AGENT-MRV-017

Thread-safe singleton configuration for GL-MRV-S3-004.
All environment variables prefixed with GL_UTO_.

This module provides comprehensive configuration management for the Upstream
Transportation & Distribution agent, supporting:
- Multi-modal transportation (road, rail, maritime, air, pipeline)
- Distance-based and spend-based calculation methods
- WTW/TTW/WTT emission scopes
- Load factor and empty running corrections
- Reefer/warehousing/hub emissions
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14083, CSRD, CDP, SBTi, GRI, GLEC)
- Incoterms boundary enforcement
- Provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.calculation.default_calculation_method
    'DISTANCE_BASED'
    >>> config.transport.road_default_vehicle
    'ARTICULATED_40_44T'
    >>> config.compliance.incoterms_enforcement
    True

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_UTO_ prefix. See individual config sections for specific variables.
"""

import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


# =============================================================================
# SECTION 1: GENERAL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GeneralConfig:
    """
    General configuration for Upstream Transportation & Distribution agent.

    Attributes:
        enabled: Master switch for the agent (GL_UTO_ENABLED)
        debug: Enable debug mode with verbose logging (GL_UTO_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_UTO_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_UTO_AGENT_ID)
        version: Agent version following SemVer (GL_UTO_VERSION)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-004",
        ...     version="1.0.0"
        ... )
        >>> general.agent_id
        'GL-MRV-S3-004'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-004"
    version: str = "1.0.0"

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "debug": self.debug,
            "log_level": self.log_level,
            "agent_id": self.agent_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_UTO_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_UTO_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_UTO_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_UTO_AGENT_ID", "GL-MRV-S3-004"),
            version=os.getenv("GL_UTO_VERSION", "1.0.0"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Upstream Transportation & Distribution agent.

    Attributes:
        database_url: PostgreSQL connection URL (GL_UTO_DATABASE_URL)
        pool_size: Connection pool size (GL_UTO_DATABASE_POOL_SIZE)
        max_overflow: Max overflow connections (GL_UTO_DATABASE_MAX_OVERFLOW)
        pool_timeout: Pool timeout in seconds (GL_UTO_DATABASE_POOL_TIMEOUT)
        schema: Database schema name (GL_UTO_DATABASE_SCHEMA)
        table_prefix: Prefix for all tables (GL_UTO_TABLE_PREFIX)

    Example:
        >>> db = DatabaseConfig(
        ...     database_url="postgresql://user:pass@localhost:5432/greenlang",
        ...     pool_size=10,
        ...     max_overflow=20,
        ...     pool_timeout=30,
        ...     schema="upstream_transportation_service",
        ...     table_prefix="gl_uto_"
        ... )
        >>> db.table_prefix
        'gl_uto_'
    """

    database_url: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    schema: str = "upstream_transportation_service"
    table_prefix: str = "gl_uto_"

    def validate(self) -> None:
        """
        Validate database configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.database_url:
            raise ValueError("database_url cannot be empty")

        if not self.database_url.startswith("postgresql://"):
            raise ValueError("database_url must start with 'postgresql://'")

        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")

        if self.max_overflow < 0:
            raise ValueError("max_overflow must be >= 0")

        if self.pool_timeout < 1:
            raise ValueError("pool_timeout must be >= 1")

        if not self.schema:
            raise ValueError("schema cannot be empty")

        if not self.table_prefix:
            raise ValueError("table_prefix cannot be empty")

        if not self.table_prefix.endswith("_"):
            raise ValueError("table_prefix must end with '_'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "schema": self.schema,
            "table_prefix": self.table_prefix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load from environment variables."""
        return cls(
            database_url=os.getenv(
                "GL_UTO_DATABASE_URL",
                "postgresql://greenlang:greenlang@localhost:5432/greenlang",
            ),
            pool_size=int(os.getenv("GL_UTO_DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("GL_UTO_DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("GL_UTO_DATABASE_POOL_TIMEOUT", "30")),
            schema=os.getenv(
                "GL_UTO_DATABASE_SCHEMA", "upstream_transportation_service"
            ),
            table_prefix=os.getenv("GL_UTO_TABLE_PREFIX", "gl_uto_"),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Upstream Transportation & Distribution agent.

    Attributes:
        redis_url: Redis connection URL (GL_UTO_REDIS_URL)
        ttl: Default cache TTL in seconds (GL_UTO_REDIS_TTL)
        prefix: Key prefix for namespacing (GL_UTO_REDIS_PREFIX)
        max_connections: Max connections in pool (GL_UTO_REDIS_MAX_CONNECTIONS)

    Example:
        >>> redis = RedisConfig(
        ...     redis_url="redis://localhost:6379/0",
        ...     ttl=3600,
        ...     prefix="gl_uto:",
        ...     max_connections=20
        ... )
        >>> redis.prefix
        'gl_uto:'
    """

    redis_url: str = ""
    ttl: int = 3600
    prefix: str = "gl_uto:"
    max_connections: int = 20

    def validate(self) -> None:
        """
        Validate Redis configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.redis_url:
            raise ValueError("redis_url cannot be empty")

        if not self.redis_url.startswith("redis://"):
            raise ValueError("redis_url must start with 'redis://'")

        if self.ttl < 1:
            raise ValueError("ttl must be >= 1")

        if not self.prefix:
            raise ValueError("prefix cannot be empty")

        if not self.prefix.endswith(":"):
            raise ValueError("prefix must end with ':'")

        if self.max_connections < 1:
            raise ValueError("max_connections must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "redis_url": self.redis_url,
            "ttl": self.ttl,
            "prefix": self.prefix,
            "max_connections": self.max_connections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load from environment variables."""
        return cls(
            redis_url=os.getenv("GL_UTO_REDIS_URL", "redis://localhost:6379/0"),
            ttl=int(os.getenv("GL_UTO_REDIS_TTL", "3600")),
            prefix=os.getenv("GL_UTO_REDIS_PREFIX", "gl_uto:"),
            max_connections=int(os.getenv("GL_UTO_REDIS_MAX_CONNECTIONS", "20")),
        )


# =============================================================================
# SECTION 4: CALCULATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CalculationConfig:
    """
    Calculation configuration for Upstream Transportation & Distribution agent.

    Attributes:
        default_calculation_method: Default calc method (GL_UTO_DEFAULT_CALCULATION_METHOD)
        default_ef_scope: Default emission factor scope (GL_UTO_DEFAULT_EF_SCOPE)
        default_gwp_source: Default GWP source (GL_UTO_DEFAULT_GWP_SOURCE)
        default_allocation_method: Default allocation method (GL_UTO_DEFAULT_ALLOCATION_METHOD)
        decimal_precision: Decimal precision for calculations (GL_UTO_DECIMAL_PRECISION)
        monte_carlo_iterations: Monte Carlo iterations (GL_UTO_MONTE_CARLO_ITERATIONS)
        confidence_intervals: Confidence intervals (GL_UTO_CONFIDENCE_INTERVALS)
        gcd_correction_factor: Great circle distance correction (GL_UTO_GCD_CORRECTION_FACTOR)
        default_laden_state: Default laden state (GL_UTO_DEFAULT_LADEN_STATE)
        include_reefer: Include reefer emissions (GL_UTO_INCLUDE_REEFER)
        include_warehousing: Include warehousing emissions (GL_UTO_INCLUDE_WAREHOUSING)
        include_hub_emissions: Include hub emissions (GL_UTO_INCLUDE_HUB_EMISSIONS)

    Example:
        >>> calc = CalculationConfig(
        ...     default_calculation_method="DISTANCE_BASED",
        ...     default_ef_scope="WTW",
        ...     default_gwp_source="AR5",
        ...     default_allocation_method="MASS",
        ...     decimal_precision=8,
        ...     monte_carlo_iterations=5000,
        ...     confidence_intervals="90,95,99",
        ...     gcd_correction_factor=Decimal("1.09"),
        ...     default_laden_state="AVERAGE",
        ...     include_reefer=True,
        ...     include_warehousing=True,
        ...     include_hub_emissions=True
        ... )
        >>> calc.default_calculation_method
        'DISTANCE_BASED'
    """

    default_calculation_method: str = "DISTANCE_BASED"
    default_ef_scope: str = "WTW"
    default_gwp_source: str = "AR5"
    default_allocation_method: str = "MASS"
    decimal_precision: int = 8
    monte_carlo_iterations: int = 5000
    confidence_intervals: str = "90,95,99"
    gcd_correction_factor: Decimal = Decimal("1.09")
    default_laden_state: str = "AVERAGE"
    include_reefer: bool = True
    include_warehousing: bool = True
    include_hub_emissions: bool = True

    def validate(self) -> None:
        """
        Validate calculation configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"DISTANCE_BASED", "SPEND_BASED", "FUEL_BASED", "WEIGHT_DISTANCE"}
        if self.default_calculation_method not in valid_methods:
            raise ValueError(
                f"Invalid default_calculation_method '{self.default_calculation_method}'. "
                f"Must be one of {valid_methods}"
            )

        valid_scopes = {"WTW", "TTW", "WTT"}
        if self.default_ef_scope not in valid_scopes:
            raise ValueError(
                f"Invalid default_ef_scope '{self.default_ef_scope}'. "
                f"Must be one of {valid_scopes}"
            )

        valid_gwp_sources = {"AR4", "AR5", "AR6"}
        if self.default_gwp_source not in valid_gwp_sources:
            raise ValueError(
                f"Invalid default_gwp_source '{self.default_gwp_source}'. "
                f"Must be one of {valid_gwp_sources}"
            )

        valid_allocation = {"MASS", "VOLUME", "ECONOMIC", "DISTANCE"}
        if self.default_allocation_method not in valid_allocation:
            raise ValueError(
                f"Invalid default_allocation_method '{self.default_allocation_method}'. "
                f"Must be one of {valid_allocation}"
            )

        if self.decimal_precision < 1 or self.decimal_precision > 18:
            raise ValueError("decimal_precision must be between 1 and 18")

        if self.monte_carlo_iterations < 100 or self.monte_carlo_iterations > 100000:
            raise ValueError("monte_carlo_iterations must be between 100 and 100000")

        # Validate confidence intervals format
        try:
            intervals = [int(x.strip()) for x in self.confidence_intervals.split(",")]
            for interval in intervals:
                if interval < 1 or interval > 99:
                    raise ValueError("Each confidence interval must be between 1 and 99")
        except Exception as e:
            raise ValueError(f"Invalid confidence_intervals format: {e}")

        if self.gcd_correction_factor < Decimal("1.0") or self.gcd_correction_factor > Decimal("2.0"):
            raise ValueError("gcd_correction_factor must be between 1.0 and 2.0")

        valid_laden_states = {"LADEN", "EMPTY", "AVERAGE"}
        if self.default_laden_state not in valid_laden_states:
            raise ValueError(
                f"Invalid default_laden_state '{self.default_laden_state}'. "
                f"Must be one of {valid_laden_states}"
            )

    def get_confidence_intervals(self) -> List[int]:
        """Parse confidence intervals string into list of integers."""
        return [int(x.strip()) for x in self.confidence_intervals.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_calculation_method": self.default_calculation_method,
            "default_ef_scope": self.default_ef_scope,
            "default_gwp_source": self.default_gwp_source,
            "default_allocation_method": self.default_allocation_method,
            "decimal_precision": self.decimal_precision,
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "confidence_intervals": self.confidence_intervals,
            "gcd_correction_factor": str(self.gcd_correction_factor),
            "default_laden_state": self.default_laden_state,
            "include_reefer": self.include_reefer,
            "include_warehousing": self.include_warehousing,
            "include_hub_emissions": self.include_hub_emissions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "gcd_correction_factor" in data_copy:
            data_copy["gcd_correction_factor"] = Decimal(data_copy["gcd_correction_factor"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "CalculationConfig":
        """Load from environment variables."""
        return cls(
            default_calculation_method=os.getenv(
                "GL_UTO_DEFAULT_CALCULATION_METHOD", "DISTANCE_BASED"
            ),
            default_ef_scope=os.getenv("GL_UTO_DEFAULT_EF_SCOPE", "WTW"),
            default_gwp_source=os.getenv("GL_UTO_DEFAULT_GWP_SOURCE", "AR5"),
            default_allocation_method=os.getenv(
                "GL_UTO_DEFAULT_ALLOCATION_METHOD", "MASS"
            ),
            decimal_precision=int(os.getenv("GL_UTO_DECIMAL_PRECISION", "8")),
            monte_carlo_iterations=int(
                os.getenv("GL_UTO_MONTE_CARLO_ITERATIONS", "5000")
            ),
            confidence_intervals=os.getenv("GL_UTO_CONFIDENCE_INTERVALS", "90,95,99"),
            gcd_correction_factor=Decimal(
                os.getenv("GL_UTO_GCD_CORRECTION_FACTOR", "1.09")
            ),
            default_laden_state=os.getenv("GL_UTO_DEFAULT_LADEN_STATE", "AVERAGE"),
            include_reefer=os.getenv("GL_UTO_INCLUDE_REEFER", "true").lower() == "true",
            include_warehousing=os.getenv("GL_UTO_INCLUDE_WAREHOUSING", "true").lower()
            == "true",
            include_hub_emissions=os.getenv("GL_UTO_INCLUDE_HUB_EMISSIONS", "true").lower()
            == "true",
        )


# =============================================================================
# SECTION 5: TRANSPORT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TransportConfig:
    """
    Transport configuration for Upstream Transportation & Distribution agent.

    Attributes:
        road_default_vehicle: Default road vehicle type (GL_UTO_ROAD_DEFAULT_VEHICLE)
        rail_default_type: Default rail type (GL_UTO_RAIL_DEFAULT_TYPE)
        maritime_default_vessel: Default maritime vessel (GL_UTO_MARITIME_DEFAULT_VESSEL)
        air_default_aircraft: Default aircraft type (GL_UTO_AIR_DEFAULT_AIRCRAFT)
        pipeline_default_type: Default pipeline type (GL_UTO_PIPELINE_DEFAULT_TYPE)
        default_temperature_control: Default temp control (GL_UTO_DEFAULT_TEMPERATURE_CONTROL)
        load_factor_enabled: Enable load factor corrections (GL_UTO_LOAD_FACTOR_ENABLED)
        empty_running_enabled: Enable empty running corrections (GL_UTO_EMPTY_RUNNING_ENABLED)

    Example:
        >>> transport = TransportConfig(
        ...     road_default_vehicle="ARTICULATED_40_44T",
        ...     rail_default_type="AVERAGE",
        ...     maritime_default_vessel="CONTAINER_PANAMAX",
        ...     air_default_aircraft="WIDEBODY_FREIGHTER",
        ...     pipeline_default_type="REFINED_PRODUCTS",
        ...     default_temperature_control="AMBIENT",
        ...     load_factor_enabled=True,
        ...     empty_running_enabled=True
        ... )
        >>> transport.road_default_vehicle
        'ARTICULATED_40_44T'
    """

    road_default_vehicle: str = "ARTICULATED_40_44T"
    rail_default_type: str = "AVERAGE"
    maritime_default_vessel: str = "CONTAINER_PANAMAX"
    air_default_aircraft: str = "WIDEBODY_FREIGHTER"
    pipeline_default_type: str = "REFINED_PRODUCTS"
    default_temperature_control: str = "AMBIENT"
    load_factor_enabled: bool = True
    empty_running_enabled: bool = True

    def validate(self) -> None:
        """
        Validate transport configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_road_vehicles = {
            "RIGID_LT_3_5T",
            "RIGID_3_5_7_5T",
            "RIGID_7_5_17T",
            "RIGID_17T_PLUS",
            "ARTICULATED_3_5_33T",
            "ARTICULATED_33_40T",
            "ARTICULATED_40_44T",
            "ARTICULATED_44T_PLUS",
        }
        if self.road_default_vehicle not in valid_road_vehicles:
            raise ValueError(
                f"Invalid road_default_vehicle '{self.road_default_vehicle}'. "
                f"Must be one of {valid_road_vehicles}"
            )

        valid_rail_types = {"FREIGHT", "PASSENGER", "AVERAGE"}
        if self.rail_default_type not in valid_rail_types:
            raise ValueError(
                f"Invalid rail_default_type '{self.rail_default_type}'. "
                f"Must be one of {valid_rail_types}"
            )

        valid_vessels = {
            "BULK_CARRIER_SMALL",
            "BULK_CARRIER_HANDYSIZE",
            "BULK_CARRIER_PANAMAX",
            "BULK_CARRIER_CAPESIZE",
            "CONTAINER_FEEDER",
            "CONTAINER_PANAMAX",
            "CONTAINER_POST_PANAMAX",
            "CONTAINER_ULCV",
            "TANKER_AFRAMAX",
            "TANKER_SUEZMAX",
            "TANKER_VLCC",
            "RO_RO",
            "GENERAL_CARGO",
        }
        if self.maritime_default_vessel not in valid_vessels:
            raise ValueError(
                f"Invalid maritime_default_vessel '{self.maritime_default_vessel}'. "
                f"Must be one of {valid_vessels}"
            )

        valid_aircraft = {
            "WIDEBODY_FREIGHTER",
            "NARROWBODY_FREIGHTER",
            "WIDEBODY_PASSENGER_BELLY",
            "NARROWBODY_PASSENGER_BELLY",
            "REGIONAL_FREIGHTER",
        }
        if self.air_default_aircraft not in valid_aircraft:
            raise ValueError(
                f"Invalid air_default_aircraft '{self.air_default_aircraft}'. "
                f"Must be one of {valid_aircraft}"
            )

        valid_pipeline_types = {"CRUDE_OIL", "REFINED_PRODUCTS", "NATURAL_GAS", "CO2"}
        if self.pipeline_default_type not in valid_pipeline_types:
            raise ValueError(
                f"Invalid pipeline_default_type '{self.pipeline_default_type}'. "
                f"Must be one of {valid_pipeline_types}"
            )

        valid_temp_controls = {"AMBIENT", "REFRIGERATED", "FROZEN", "CONTROLLED"}
        if self.default_temperature_control not in valid_temp_controls:
            raise ValueError(
                f"Invalid default_temperature_control '{self.default_temperature_control}'. "
                f"Must be one of {valid_temp_controls}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "road_default_vehicle": self.road_default_vehicle,
            "rail_default_type": self.rail_default_type,
            "maritime_default_vessel": self.maritime_default_vessel,
            "air_default_aircraft": self.air_default_aircraft,
            "pipeline_default_type": self.pipeline_default_type,
            "default_temperature_control": self.default_temperature_control,
            "load_factor_enabled": self.load_factor_enabled,
            "empty_running_enabled": self.empty_running_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransportConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "TransportConfig":
        """Load from environment variables."""
        return cls(
            road_default_vehicle=os.getenv(
                "GL_UTO_ROAD_DEFAULT_VEHICLE", "ARTICULATED_40_44T"
            ),
            rail_default_type=os.getenv("GL_UTO_RAIL_DEFAULT_TYPE", "AVERAGE"),
            maritime_default_vessel=os.getenv(
                "GL_UTO_MARITIME_DEFAULT_VESSEL", "CONTAINER_PANAMAX"
            ),
            air_default_aircraft=os.getenv(
                "GL_UTO_AIR_DEFAULT_AIRCRAFT", "WIDEBODY_FREIGHTER"
            ),
            pipeline_default_type=os.getenv(
                "GL_UTO_PIPELINE_DEFAULT_TYPE", "REFINED_PRODUCTS"
            ),
            default_temperature_control=os.getenv(
                "GL_UTO_DEFAULT_TEMPERATURE_CONTROL", "AMBIENT"
            ),
            load_factor_enabled=os.getenv("GL_UTO_LOAD_FACTOR_ENABLED", "true").lower()
            == "true",
            empty_running_enabled=os.getenv(
                "GL_UTO_EMPTY_RUNNING_ENABLED", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 6: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Upstream Transportation & Distribution agent.

    Attributes:
        compliance_frameworks: Enabled frameworks (GL_UTO_COMPLIANCE_FRAMEWORKS)
        strict_boundary: Enforce strict boundary (GL_UTO_STRICT_BOUNDARY)
        double_counting_check: Check for double counting (GL_UTO_DOUBLE_COUNTING_CHECK)
        incoterms_enforcement: Enforce incoterms (GL_UTO_INCOTERMS_ENFORCEMENT)

    Example:
        >>> compliance = ComplianceConfig(
        ...     compliance_frameworks="GHG_PROTOCOL_SCOPE3,ISO_14083,CSRD_ESRS_E1",
        ...     strict_boundary=True,
        ...     double_counting_check=True,
        ...     incoterms_enforcement=True
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14083', 'CSRD_ESRS_E1']
    """

    compliance_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14083,CSRD_ESRS_E1,CDP,SBTI,GRI_305,GLEC_FRAMEWORK"
    )
    strict_boundary: bool = True
    double_counting_check: bool = True
    incoterms_enforcement: bool = True

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL_SCOPE3",
            "ISO_14083",
            "CSRD_ESRS_E1",
            "CDP",
            "SBTI",
            "GRI_305",
            "GLEC_FRAMEWORK",
        }

        frameworks = self.get_frameworks()
        if not frameworks:
            raise ValueError("At least one compliance framework must be enabled")

        for framework in frameworks:
            if framework not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{framework}'. Must be one of {valid_frameworks}"
                )

    def get_frameworks(self) -> List[str]:
        """Parse compliance frameworks string into list."""
        return [f.strip() for f in self.compliance_frameworks.split(",") if f.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compliance_frameworks": self.compliance_frameworks,
            "strict_boundary": self.strict_boundary,
            "double_counting_check": self.double_counting_check,
            "incoterms_enforcement": self.incoterms_enforcement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            compliance_frameworks=os.getenv(
                "GL_UTO_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14083,CSRD_ESRS_E1,CDP,SBTI,GRI_305,GLEC_FRAMEWORK",
            ),
            strict_boundary=os.getenv("GL_UTO_STRICT_BOUNDARY", "true").lower()
            == "true",
            double_counting_check=os.getenv(
                "GL_UTO_DOUBLE_COUNTING_CHECK", "true"
            ).lower()
            == "true",
            incoterms_enforcement=os.getenv(
                "GL_UTO_INCOTERMS_ENFORCEMENT", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 7: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    API configuration for Upstream Transportation & Distribution agent.

    Attributes:
        api_prefix: API route prefix (GL_UTO_API_PREFIX)
        max_batch_size: Max batch size (GL_UTO_API_MAX_BATCH_SIZE)
        rate_limit: Requests per minute (GL_UTO_API_RATE_LIMIT)
        timeout: Request timeout in seconds (GL_UTO_API_TIMEOUT)
        worker_threads: Worker thread count (GL_UTO_WORKER_THREADS)

    Example:
        >>> api = APIConfig(
        ...     api_prefix="/api/v1/upstream-transportation",
        ...     max_batch_size=500,
        ...     rate_limit=100,
        ...     timeout=300,
        ...     worker_threads=4
        ... )
        >>> api.api_prefix
        '/api/v1/upstream-transportation'
    """

    api_prefix: str = "/api/v1/upstream-transportation"
    max_batch_size: int = 500
    rate_limit: int = 100
    timeout: int = 300
    worker_threads: int = 4

    def validate(self) -> None:
        """
        Validate API configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.api_prefix:
            raise ValueError("api_prefix cannot be empty")

        if not self.api_prefix.startswith("/"):
            raise ValueError("api_prefix must start with '/'")

        if self.max_batch_size < 1 or self.max_batch_size > 10000:
            raise ValueError("max_batch_size must be between 1 and 10000")

        if self.rate_limit < 1 or self.rate_limit > 10000:
            raise ValueError("rate_limit must be between 1 and 10000")

        if self.timeout < 1 or self.timeout > 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")

        if self.worker_threads < 1 or self.worker_threads > 64:
            raise ValueError("worker_threads must be between 1 and 64")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_prefix": self.api_prefix,
            "max_batch_size": self.max_batch_size,
            "rate_limit": self.rate_limit,
            "timeout": self.timeout,
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
            api_prefix=os.getenv(
                "GL_UTO_API_PREFIX", "/api/v1/upstream-transportation"
            ),
            max_batch_size=int(os.getenv("GL_UTO_API_MAX_BATCH_SIZE", "500")),
            rate_limit=int(os.getenv("GL_UTO_API_RATE_LIMIT", "100")),
            timeout=int(os.getenv("GL_UTO_API_TIMEOUT", "300")),
            worker_threads=int(os.getenv("GL_UTO_WORKER_THREADS", "4")),
        )


# =============================================================================
# SECTION 8: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Upstream Transportation & Distribution agent.

    Attributes:
        provenance_enabled: Enable provenance tracking (GL_UTO_PROVENANCE_ENABLED)
        provenance_algorithm: Hash algorithm (GL_UTO_PROVENANCE_ALGORITHM)
        provenance_chain: Chain hashes (GL_UTO_PROVENANCE_CHAIN)

    Example:
        >>> provenance = ProvenanceConfig(
        ...     provenance_enabled=True,
        ...     provenance_algorithm="sha256",
        ...     provenance_chain=True
        ... )
        >>> provenance.provenance_algorithm
        'sha256'
    """

    provenance_enabled: bool = True
    provenance_algorithm: str = "sha256"
    provenance_chain: bool = True

    def validate(self) -> None:
        """
        Validate provenance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_algorithms = {"sha256", "sha512", "blake2b"}
        if self.provenance_algorithm not in valid_algorithms:
            raise ValueError(
                f"Invalid provenance_algorithm '{self.provenance_algorithm}'. "
                f"Must be one of {valid_algorithms}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provenance_enabled": self.provenance_enabled,
            "provenance_algorithm": self.provenance_algorithm,
            "provenance_chain": self.provenance_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            provenance_enabled=os.getenv("GL_UTO_PROVENANCE_ENABLED", "true").lower()
            == "true",
            provenance_algorithm=os.getenv("GL_UTO_PROVENANCE_ALGORITHM", "sha256"),
            provenance_chain=os.getenv("GL_UTO_PROVENANCE_CHAIN", "true").lower()
            == "true",
        )


# =============================================================================
# SECTION 9: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Upstream Transportation & Distribution agent.

    Attributes:
        metrics_enabled: Enable metrics collection (GL_UTO_METRICS_ENABLED)
        metrics_prefix: Metrics name prefix (GL_UTO_METRICS_PREFIX)
        metrics_buckets: Histogram buckets (GL_UTO_METRICS_BUCKETS)

    Example:
        >>> metrics = MetricsConfig(
        ...     metrics_enabled=True,
        ...     metrics_prefix="gl_uto_",
        ...     metrics_buckets="0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_buckets()
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    """

    metrics_enabled: bool = True
    metrics_prefix: str = "gl_uto_"
    metrics_buckets: str = "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"

    def validate(self) -> None:
        """
        Validate metrics configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.metrics_prefix:
            raise ValueError("metrics_prefix cannot be empty")

        if not self.metrics_prefix.endswith("_"):
            raise ValueError("metrics_prefix must end with '_'")

        # Validate buckets format
        try:
            buckets = self.get_buckets()
            if not buckets:
                raise ValueError("At least one bucket must be defined")
            for bucket in buckets:
                if bucket <= 0:
                    raise ValueError("All buckets must be positive")
        except Exception as e:
            raise ValueError(f"Invalid metrics_buckets format: {e}")

    def get_buckets(self) -> List[float]:
        """Parse metrics buckets string into list of floats."""
        return [float(x.strip()) for x in self.metrics_buckets.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics_enabled": self.metrics_enabled,
            "metrics_prefix": self.metrics_prefix,
            "metrics_buckets": self.metrics_buckets,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            metrics_enabled=os.getenv("GL_UTO_METRICS_ENABLED", "true").lower()
            == "true",
            metrics_prefix=os.getenv("GL_UTO_METRICS_PREFIX", "gl_uto_"),
            metrics_buckets=os.getenv(
                "GL_UTO_METRICS_BUCKETS", "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
            ),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


class UpstreamTransportationConfig:
    """
    Master configuration class for Upstream Transportation & Distribution agent.

    This class aggregates all configuration sections and provides a unified
    interface for accessing configuration values. It implements the singleton
    pattern with thread-safe access.

    Attributes:
        general: General configuration
        database: Database configuration
        redis: Redis configuration
        calculation: Calculation configuration
        transport: Transport configuration
        compliance: Compliance configuration
        api: API configuration
        provenance: Provenance configuration
        metrics: Metrics configuration

    Example:
        >>> config = UpstreamTransportationConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-004'
        >>> config.calculation.default_calculation_method
        'DISTANCE_BASED'
        >>> config.validate_all()
    """

    def __init__(
        self,
        general: GeneralConfig,
        database: DatabaseConfig,
        redis: RedisConfig,
        calculation: CalculationConfig,
        transport: TransportConfig,
        compliance: ComplianceConfig,
        api: APIConfig,
        provenance: ProvenanceConfig,
        metrics: MetricsConfig,
    ):
        """
        Initialize master configuration.

        Args:
            general: General configuration
            database: Database configuration
            redis: Redis configuration
            calculation: Calculation configuration
            transport: Transport configuration
            compliance: Compliance configuration
            api: API configuration
            provenance: Provenance configuration
            metrics: Metrics configuration
        """
        self.general = general
        self.database = database
        self.redis = redis
        self.calculation = calculation
        self.transport = transport
        self.compliance = compliance
        self.api = api
        self.provenance = provenance
        self.metrics = metrics

    def validate_all(self) -> None:
        """
        Validate all configuration sections.

        Raises:
            ValueError: If any configuration section is invalid
        """
        self.general.validate()
        self.database.validate()
        self.redis.validate()
        self.calculation.validate()
        self.transport.validate()
        self.compliance.validate()
        self.api.validate()
        self.provenance.validate()
        self.metrics.validate()

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
            "calculation": self.calculation.to_dict(),
            "transport": self.transport.to_dict(),
            "compliance": self.compliance.to_dict(),
            "api": self.api.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpstreamTransportationConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            UpstreamTransportationConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data["general"]),
            database=DatabaseConfig.from_dict(data["database"]),
            redis=RedisConfig.from_dict(data["redis"]),
            calculation=CalculationConfig.from_dict(data["calculation"]),
            transport=TransportConfig.from_dict(data["transport"]),
            compliance=ComplianceConfig.from_dict(data["compliance"]),
            api=APIConfig.from_dict(data["api"]),
            provenance=ProvenanceConfig.from_dict(data["provenance"]),
            metrics=MetricsConfig.from_dict(data["metrics"]),
        )

    @classmethod
    def from_env(cls) -> "UpstreamTransportationConfig":
        """
        Load configuration from environment variables.

        Returns:
            UpstreamTransportationConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            calculation=CalculationConfig.from_env(),
            transport=TransportConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            api=APIConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            metrics=MetricsConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[UpstreamTransportationConfig] = None
_config_lock = threading.RLock()


def get_config() -> UpstreamTransportationConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton. The first call will load configuration from
    environment variables. Subsequent calls return the cached instance.

    Returns:
        UpstreamTransportationConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-004'

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The configuration is initialized only once.
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            # Double-checked locking pattern
            if _config_instance is None:
                _config_instance = UpstreamTransportationConfig.from_env()
                _config_instance.validate_all()

    return _config_instance


def set_config(config: UpstreamTransportationConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.

    Args:
        config: UpstreamTransportationConfig instance to set as singleton

    Example:
        >>> custom_config = UpstreamTransportationConfig.from_dict({...})
        >>> set_config(custom_config)

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently.
    """
    global _config_instance

    with _config_lock:
        config.validate_all()
        _config_instance = config


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


def _load_from_env() -> UpstreamTransportationConfig:
    """
    Load configuration from environment variables (internal helper).

    This is an internal helper function that loads all configuration sections
    from environment variables. Use get_config() instead for normal usage.

    Returns:
        UpstreamTransportationConfig instance loaded from environment

    Note:
        This function is for internal use. Use get_config() for normal access.
    """
    return UpstreamTransportationConfig.from_env()


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================


def validate_config(config: UpstreamTransportationConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    This function validates all configuration sections and returns a list of
    validation errors. Unlike validate_all() which raises on first error,
    this function collects all errors.

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
    errors = []

    # Validate each section
    sections = [
        ("general", config.general),
        ("database", config.database),
        ("redis", config.redis),
        ("calculation", config.calculation),
        ("transport", config.transport),
        ("compliance", config.compliance),
        ("api", config.api),
        ("provenance", config.provenance),
        ("metrics", config.metrics),
    ]

    for section_name, section in sections:
        try:
            section.validate()
        except ValueError as e:
            errors.append(f"{section_name}: {str(e)}")

    return errors


def print_config(config: UpstreamTransportationConfig) -> None:
    """
    Print configuration in human-readable format.

    This function prints all configuration sections in a formatted,
    human-readable manner. Useful for debugging and verification.

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
        ===== Upstream Transportation & Distribution Configuration =====
        [GENERAL]
        enabled: True
        debug: False
        ...
    """
    print("===== Upstream Transportation & Distribution Configuration =====")
    print("\n[GENERAL]")
    for key, value in config.general.to_dict().items():
        print(f"{key}: {value}")

    print("\n[DATABASE]")
    for key, value in config.database.to_dict().items():
        if key == "database_url":
            # Mask password in URL
            print(f"{key}: [REDACTED]")
        else:
            print(f"{key}: {value}")

    print("\n[REDIS]")
    for key, value in config.redis.to_dict().items():
        if key == "redis_url":
            # Mask password in URL
            print(f"{key}: [REDACTED]")
        else:
            print(f"{key}: {value}")

    print("\n[CALCULATION]")
    for key, value in config.calculation.to_dict().items():
        print(f"{key}: {value}")

    print("\n[TRANSPORT]")
    for key, value in config.transport.to_dict().items():
        print(f"{key}: {value}")

    print("\n[COMPLIANCE]")
    for key, value in config.compliance.to_dict().items():
        print(f"{key}: {value}")

    print("\n[API]")
    for key, value in config.api.to_dict().items():
        print(f"{key}: {value}")

    print("\n[PROVENANCE]")
    for key, value in config.provenance.to_dict().items():
        print(f"{key}: {value}")

    print("\n[METRICS]")
    for key, value in config.metrics.to_dict().items():
        print(f"{key}: {value}")

    print("\n" + "=" * 64)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration classes
    "GeneralConfig",
    "DatabaseConfig",
    "RedisConfig",
    "CalculationConfig",
    "TransportConfig",
    "ComplianceConfig",
    "APIConfig",
    "ProvenanceConfig",
    "MetricsConfig",
    "UpstreamTransportationConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
]
