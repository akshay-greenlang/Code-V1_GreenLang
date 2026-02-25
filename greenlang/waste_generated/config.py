# -*- coding: utf-8 -*-
"""
Waste Generated in Operations Configuration - AGENT-MRV-018

Thread-safe singleton configuration for GL-MRV-S3-005.
All environment variables prefixed with GL_WG_.

This module provides comprehensive configuration management for the Waste
Generated in Operations agent, supporting:
- Landfill emissions (CH4/N2O with DOCF/F/oxidation modeling)
- Incineration emissions (biogenic/fossil CO2, N2O, CH4)
- Recycling/recovery (cut-off/allocation/substitution approaches)
- Composting emissions (aerobic/anaerobic, CH4/N2O)
- Wastewater treatment emissions (COD/BOD-based CH4, N2O)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, EPA, DEFRA, SBTi, CDP)
- Multi-year landfill gas generation projections (FOD model)
- Double-counting prevention and boundary enforcement
- Provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.calculation.default_gwp_source
    'AR5'
    >>> config.landfill.default_docf
    Decimal('0.5')
    >>> config.compliance.double_counting_check
    True

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_WG_ prefix. See individual config sections for specific variables.
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
    General configuration for Waste Generated in Operations agent.

    Attributes:
        enabled: Master switch for the agent (GL_WG_ENABLED)
        debug: Enable debug mode with verbose logging (GL_WG_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_WG_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_WG_AGENT_ID)
        version: Agent version following SemVer (GL_WG_VERSION)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-005",
        ...     version="1.0.0"
        ... )
        >>> general.agent_id
        'GL-MRV-S3-005'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-005"
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
            enabled=os.getenv("GL_WG_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_WG_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_WG_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_WG_AGENT_ID", "GL-MRV-S3-005"),
            version=os.getenv("GL_WG_VERSION", "1.0.0"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Waste Generated in Operations agent.

    Attributes:
        database_url: PostgreSQL connection URL (GL_WG_DATABASE_URL)
        pool_size: Connection pool size (GL_WG_DATABASE_POOL_SIZE)
        max_overflow: Max overflow connections (GL_WG_DATABASE_MAX_OVERFLOW)
        pool_timeout: Pool timeout in seconds (GL_WG_DATABASE_POOL_TIMEOUT)
        schema: Database schema name (GL_WG_DATABASE_SCHEMA)
        table_prefix: Prefix for all tables (GL_WG_TABLE_PREFIX)

    Example:
        >>> db = DatabaseConfig(
        ...     database_url="postgresql://user:pass@localhost:5432/greenlang",
        ...     pool_size=10,
        ...     max_overflow=20,
        ...     pool_timeout=30,
        ...     schema="waste_generated_service",
        ...     table_prefix="gl_wg_"
        ... )
        >>> db.table_prefix
        'gl_wg_'
    """

    database_url: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    schema: str = "waste_generated_service"
    table_prefix: str = "gl_wg_"

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
                "GL_WG_DATABASE_URL",
                "postgresql://greenlang:greenlang@localhost:5432/greenlang",
            ),
            pool_size=int(os.getenv("GL_WG_DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("GL_WG_DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("GL_WG_DATABASE_POOL_TIMEOUT", "30")),
            schema=os.getenv("GL_WG_DATABASE_SCHEMA", "waste_generated_service"),
            table_prefix=os.getenv("GL_WG_TABLE_PREFIX", "gl_wg_"),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Waste Generated in Operations agent.

    Attributes:
        redis_url: Redis connection URL (GL_WG_REDIS_URL)
        ttl: Default cache TTL in seconds (GL_WG_REDIS_TTL)
        prefix: Key prefix for namespacing (GL_WG_REDIS_PREFIX)
        max_connections: Max connections in pool (GL_WG_REDIS_MAX_CONNECTIONS)

    Example:
        >>> redis = RedisConfig(
        ...     redis_url="redis://localhost:6379/0",
        ...     ttl=3600,
        ...     prefix="gl_wg:",
        ...     max_connections=20
        ... )
        >>> redis.prefix
        'gl_wg:'
    """

    redis_url: str = ""
    ttl: int = 3600
    prefix: str = "gl_wg:"
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
            redis_url=os.getenv("GL_WG_REDIS_URL", "redis://localhost:6379/0"),
            ttl=int(os.getenv("GL_WG_REDIS_TTL", "3600")),
            prefix=os.getenv("GL_WG_REDIS_PREFIX", "gl_wg:"),
            max_connections=int(os.getenv("GL_WG_REDIS_MAX_CONNECTIONS", "20")),
        )


# =============================================================================
# SECTION 4: LANDFILL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class LandfillConfig:
    """
    Landfill configuration for Waste Generated in Operations agent.

    Attributes:
        default_docf: Default degradable organic carbon fraction (GL_WG_LANDFILL_DEFAULT_DOCF)
        default_f: Default fraction of CH4 in landfill gas (GL_WG_LANDFILL_DEFAULT_F)
        default_ox_with_cover: Default oxidation factor with cover (GL_WG_LANDFILL_DEFAULT_OX_WITH_COVER)
        default_ox_without: Default oxidation factor without cover (GL_WG_LANDFILL_DEFAULT_OX_WITHOUT)
        default_gas_capture: Default gas capture efficiency (GL_WG_LANDFILL_DEFAULT_GAS_CAPTURE)
        multi_year_projection_years: Years for FOD model projection (GL_WG_LANDFILL_MULTI_YEAR_PROJECTION_YEARS)
        default_gwp_version: Default GWP version (GL_WG_LANDFILL_DEFAULT_GWP_VERSION)
        default_decay_rate: Default methane generation decay rate k (GL_WG_LANDFILL_DEFAULT_DECAY_RATE)
        default_mcf: Default methane correction factor (GL_WG_LANDFILL_DEFAULT_MCF)

    Example:
        >>> landfill = LandfillConfig(
        ...     default_docf=Decimal("0.5"),
        ...     default_f=Decimal("0.5"),
        ...     default_ox_with_cover=Decimal("0.1"),
        ...     default_ox_without=Decimal("0.0"),
        ...     default_gas_capture=Decimal("0.0"),
        ...     multi_year_projection_years=100,
        ...     default_gwp_version="AR5",
        ...     default_decay_rate=Decimal("0.05"),
        ...     default_mcf=Decimal("1.0")
        ... )
        >>> landfill.default_docf
        Decimal('0.5')
    """

    default_docf: Decimal = Decimal("0.5")
    default_f: Decimal = Decimal("0.5")
    default_ox_with_cover: Decimal = Decimal("0.1")
    default_ox_without: Decimal = Decimal("0.0")
    default_gas_capture: Decimal = Decimal("0.0")
    multi_year_projection_years: int = 100
    default_gwp_version: str = "AR5"
    default_decay_rate: Decimal = Decimal("0.05")
    default_mcf: Decimal = Decimal("1.0")

    def validate(self) -> None:
        """
        Validate landfill configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_docf < Decimal("0") or self.default_docf > Decimal("1"):
            raise ValueError("default_docf must be between 0 and 1")

        if self.default_f < Decimal("0") or self.default_f > Decimal("1"):
            raise ValueError("default_f must be between 0 and 1")

        if self.default_ox_with_cover < Decimal("0") or self.default_ox_with_cover > Decimal("1"):
            raise ValueError("default_ox_with_cover must be between 0 and 1")

        if self.default_ox_without < Decimal("0") or self.default_ox_without > Decimal("1"):
            raise ValueError("default_ox_without must be between 0 and 1")

        if self.default_gas_capture < Decimal("0") or self.default_gas_capture > Decimal("1"):
            raise ValueError("default_gas_capture must be between 0 and 1")

        if self.multi_year_projection_years < 1 or self.multi_year_projection_years > 200:
            raise ValueError("multi_year_projection_years must be between 1 and 200")

        valid_gwp_versions = {"AR4", "AR5", "AR6"}
        if self.default_gwp_version not in valid_gwp_versions:
            raise ValueError(
                f"Invalid default_gwp_version '{self.default_gwp_version}'. "
                f"Must be one of {valid_gwp_versions}"
            )

        if self.default_decay_rate <= Decimal("0") or self.default_decay_rate > Decimal("1"):
            raise ValueError("default_decay_rate must be between 0 and 1 (exclusive of 0)")

        if self.default_mcf < Decimal("0") or self.default_mcf > Decimal("1"):
            raise ValueError("default_mcf must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_docf": str(self.default_docf),
            "default_f": str(self.default_f),
            "default_ox_with_cover": str(self.default_ox_with_cover),
            "default_ox_without": str(self.default_ox_without),
            "default_gas_capture": str(self.default_gas_capture),
            "multi_year_projection_years": self.multi_year_projection_years,
            "default_gwp_version": self.default_gwp_version,
            "default_decay_rate": str(self.default_decay_rate),
            "default_mcf": str(self.default_mcf),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LandfillConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_docf", "default_f", "default_ox_with_cover",
                    "default_ox_without", "default_gas_capture", "default_decay_rate",
                    "default_mcf"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "LandfillConfig":
        """Load from environment variables."""
        return cls(
            default_docf=Decimal(os.getenv("GL_WG_LANDFILL_DEFAULT_DOCF", "0.5")),
            default_f=Decimal(os.getenv("GL_WG_LANDFILL_DEFAULT_F", "0.5")),
            default_ox_with_cover=Decimal(
                os.getenv("GL_WG_LANDFILL_DEFAULT_OX_WITH_COVER", "0.1")
            ),
            default_ox_without=Decimal(
                os.getenv("GL_WG_LANDFILL_DEFAULT_OX_WITHOUT", "0.0")
            ),
            default_gas_capture=Decimal(
                os.getenv("GL_WG_LANDFILL_DEFAULT_GAS_CAPTURE", "0.0")
            ),
            multi_year_projection_years=int(
                os.getenv("GL_WG_LANDFILL_MULTI_YEAR_PROJECTION_YEARS", "100")
            ),
            default_gwp_version=os.getenv("GL_WG_LANDFILL_DEFAULT_GWP_VERSION", "AR5"),
            default_decay_rate=Decimal(
                os.getenv("GL_WG_LANDFILL_DEFAULT_DECAY_RATE", "0.05")
            ),
            default_mcf=Decimal(os.getenv("GL_WG_LANDFILL_DEFAULT_MCF", "1.0")),
        )


# =============================================================================
# SECTION 5: INCINERATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class IncinerationConfig:
    """
    Incineration configuration for Waste Generated in Operations agent.

    Attributes:
        default_oxidation_factor: Default oxidation factor (GL_WG_INCINERATION_DEFAULT_OXIDATION_FACTOR)
        include_biogenic_co2: Include biogenic CO2 (GL_WG_INCINERATION_INCLUDE_BIOGENIC_CO2)
        default_incinerator_type: Default incinerator type (GL_WG_INCINERATION_DEFAULT_TYPE)
        default_energy_recovery: Default energy recovery efficiency (GL_WG_INCINERATION_DEFAULT_ENERGY_RECOVERY)
        default_fossil_carbon_fraction: Default fossil carbon fraction (GL_WG_INCINERATION_DEFAULT_FOSSIL_FRACTION)

    Example:
        >>> incineration = IncinerationConfig(
        ...     default_oxidation_factor=Decimal("1.0"),
        ...     include_biogenic_co2=False,
        ...     default_incinerator_type="CONTINUOUS_STOKER",
        ...     default_energy_recovery=Decimal("0.0"),
        ...     default_fossil_carbon_fraction=Decimal("1.0")
        ... )
        >>> incineration.default_incinerator_type
        'CONTINUOUS_STOKER'
    """

    default_oxidation_factor: Decimal = Decimal("1.0")
    include_biogenic_co2: bool = False
    default_incinerator_type: str = "CONTINUOUS_STOKER"
    default_energy_recovery: Decimal = Decimal("0.0")
    default_fossil_carbon_fraction: Decimal = Decimal("1.0")

    def validate(self) -> None:
        """
        Validate incineration configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_oxidation_factor < Decimal("0") or self.default_oxidation_factor > Decimal("1"):
            raise ValueError("default_oxidation_factor must be between 0 and 1")

        valid_types = {
            "CONTINUOUS_STOKER",
            "BATCH_STOKER",
            "ROTARY_KILN",
            "FLUIDIZED_BED",
            "GASIFICATION",
            "PYROLYSIS",
        }
        if self.default_incinerator_type not in valid_types:
            raise ValueError(
                f"Invalid default_incinerator_type '{self.default_incinerator_type}'. "
                f"Must be one of {valid_types}"
            )

        if self.default_energy_recovery < Decimal("0") or self.default_energy_recovery > Decimal("1"):
            raise ValueError("default_energy_recovery must be between 0 and 1")

        if self.default_fossil_carbon_fraction < Decimal("0") or self.default_fossil_carbon_fraction > Decimal("1"):
            raise ValueError("default_fossil_carbon_fraction must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_oxidation_factor": str(self.default_oxidation_factor),
            "include_biogenic_co2": self.include_biogenic_co2,
            "default_incinerator_type": self.default_incinerator_type,
            "default_energy_recovery": str(self.default_energy_recovery),
            "default_fossil_carbon_fraction": str(self.default_fossil_carbon_fraction),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncinerationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_oxidation_factor", "default_energy_recovery",
                    "default_fossil_carbon_fraction"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "IncinerationConfig":
        """Load from environment variables."""
        return cls(
            default_oxidation_factor=Decimal(
                os.getenv("GL_WG_INCINERATION_DEFAULT_OXIDATION_FACTOR", "1.0")
            ),
            include_biogenic_co2=os.getenv(
                "GL_WG_INCINERATION_INCLUDE_BIOGENIC_CO2", "false"
            ).lower()
            == "true",
            default_incinerator_type=os.getenv(
                "GL_WG_INCINERATION_DEFAULT_TYPE", "CONTINUOUS_STOKER"
            ),
            default_energy_recovery=Decimal(
                os.getenv("GL_WG_INCINERATION_DEFAULT_ENERGY_RECOVERY", "0.0")
            ),
            default_fossil_carbon_fraction=Decimal(
                os.getenv("GL_WG_INCINERATION_DEFAULT_FOSSIL_FRACTION", "1.0")
            ),
        )


# =============================================================================
# SECTION 6: RECYCLING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RecyclingConfig:
    """
    Recycling configuration for Waste Generated in Operations agent.

    Attributes:
        default_approach: Default recycling approach (GL_WG_RECYCLING_DEFAULT_APPROACH)
        include_avoided_emissions: Include avoided emissions (GL_WG_RECYCLING_INCLUDE_AVOIDED)
        default_quality_factor: Default quality factor (GL_WG_RECYCLING_DEFAULT_QUALITY_FACTOR)
        default_mrf_ef: Default MRF emission factor (GL_WG_RECYCLING_DEFAULT_MRF_EF)
        allocation_factor: Allocation factor for recycled content (GL_WG_RECYCLING_ALLOCATION_FACTOR)

    Example:
        >>> recycling = RecyclingConfig(
        ...     default_approach="CUT_OFF",
        ...     include_avoided_emissions=False,
        ...     default_quality_factor=Decimal("1.0"),
        ...     default_mrf_ef=Decimal("0.0"),
        ...     allocation_factor=Decimal("0.5")
        ... )
        >>> recycling.default_approach
        'CUT_OFF'
    """

    default_approach: str = "CUT_OFF"
    include_avoided_emissions: bool = False
    default_quality_factor: Decimal = Decimal("1.0")
    default_mrf_ef: Decimal = Decimal("0.0")
    allocation_factor: Decimal = Decimal("0.5")

    def validate(self) -> None:
        """
        Validate recycling configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_approaches = {"CUT_OFF", "ALLOCATION", "SUBSTITUTION", "CONSEQUENTIAL"}
        if self.default_approach not in valid_approaches:
            raise ValueError(
                f"Invalid default_approach '{self.default_approach}'. "
                f"Must be one of {valid_approaches}"
            )

        if self.default_quality_factor < Decimal("0") or self.default_quality_factor > Decimal("1"):
            raise ValueError("default_quality_factor must be between 0 and 1")

        if self.default_mrf_ef < Decimal("0"):
            raise ValueError("default_mrf_ef must be >= 0")

        if self.allocation_factor < Decimal("0") or self.allocation_factor > Decimal("1"):
            raise ValueError("allocation_factor must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_approach": self.default_approach,
            "include_avoided_emissions": self.include_avoided_emissions,
            "default_quality_factor": str(self.default_quality_factor),
            "default_mrf_ef": str(self.default_mrf_ef),
            "allocation_factor": str(self.allocation_factor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecyclingConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_quality_factor", "default_mrf_ef", "allocation_factor"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RecyclingConfig":
        """Load from environment variables."""
        return cls(
            default_approach=os.getenv("GL_WG_RECYCLING_DEFAULT_APPROACH", "CUT_OFF"),
            include_avoided_emissions=os.getenv(
                "GL_WG_RECYCLING_INCLUDE_AVOIDED", "false"
            ).lower()
            == "true",
            default_quality_factor=Decimal(
                os.getenv("GL_WG_RECYCLING_DEFAULT_QUALITY_FACTOR", "1.0")
            ),
            default_mrf_ef=Decimal(
                os.getenv("GL_WG_RECYCLING_DEFAULT_MRF_EF", "0.0")
            ),
            allocation_factor=Decimal(
                os.getenv("GL_WG_RECYCLING_ALLOCATION_FACTOR", "0.5")
            ),
        )


# =============================================================================
# SECTION 7: COMPOSTING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CompostingConfig:
    """
    Composting configuration for Waste Generated in Operations agent.

    Attributes:
        default_basis: Default basis (GL_WG_COMPOSTING_DEFAULT_BASIS)
        ch4_ef_wet: CH4 emission factor wet weight basis (GL_WG_COMPOSTING_CH4_EF_WET)
        n2o_ef_wet: N2O emission factor wet weight basis (GL_WG_COMPOSTING_N2O_EF_WET)
        ch4_ef_dry: CH4 emission factor dry weight basis (GL_WG_COMPOSTING_CH4_EF_DRY)
        n2o_ef_dry: N2O emission factor dry weight basis (GL_WG_COMPOSTING_N2O_EF_DRY)
        default_moisture_content: Default moisture content fraction (GL_WG_COMPOSTING_DEFAULT_MOISTURE)

    Example:
        >>> composting = CompostingConfig(
        ...     default_basis="WET_WEIGHT",
        ...     ch4_ef_wet=Decimal("4.0"),
        ...     n2o_ef_wet=Decimal("0.3"),
        ...     ch4_ef_dry=Decimal("10.0"),
        ...     n2o_ef_dry=Decimal("0.6"),
        ...     default_moisture_content=Decimal("0.5")
        ... )
        >>> composting.default_basis
        'WET_WEIGHT'
    """

    default_basis: str = "WET_WEIGHT"
    ch4_ef_wet: Decimal = Decimal("4.0")
    n2o_ef_wet: Decimal = Decimal("0.3")
    ch4_ef_dry: Decimal = Decimal("10.0")
    n2o_ef_dry: Decimal = Decimal("0.6")
    default_moisture_content: Decimal = Decimal("0.5")

    def validate(self) -> None:
        """
        Validate composting configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_bases = {"WET_WEIGHT", "DRY_WEIGHT"}
        if self.default_basis not in valid_bases:
            raise ValueError(
                f"Invalid default_basis '{self.default_basis}'. "
                f"Must be one of {valid_bases}"
            )

        if self.ch4_ef_wet < Decimal("0"):
            raise ValueError("ch4_ef_wet must be >= 0")

        if self.n2o_ef_wet < Decimal("0"):
            raise ValueError("n2o_ef_wet must be >= 0")

        if self.ch4_ef_dry < Decimal("0"):
            raise ValueError("ch4_ef_dry must be >= 0")

        if self.n2o_ef_dry < Decimal("0"):
            raise ValueError("n2o_ef_dry must be >= 0")

        if self.default_moisture_content < Decimal("0") or self.default_moisture_content > Decimal("1"):
            raise ValueError("default_moisture_content must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_basis": self.default_basis,
            "ch4_ef_wet": str(self.ch4_ef_wet),
            "n2o_ef_wet": str(self.n2o_ef_wet),
            "ch4_ef_dry": str(self.ch4_ef_dry),
            "n2o_ef_dry": str(self.n2o_ef_dry),
            "default_moisture_content": str(self.default_moisture_content),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompostingConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["ch4_ef_wet", "n2o_ef_wet", "ch4_ef_dry", "n2o_ef_dry",
                    "default_moisture_content"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "CompostingConfig":
        """Load from environment variables."""
        return cls(
            default_basis=os.getenv("GL_WG_COMPOSTING_DEFAULT_BASIS", "WET_WEIGHT"),
            ch4_ef_wet=Decimal(os.getenv("GL_WG_COMPOSTING_CH4_EF_WET", "4.0")),
            n2o_ef_wet=Decimal(os.getenv("GL_WG_COMPOSTING_N2O_EF_WET", "0.3")),
            ch4_ef_dry=Decimal(os.getenv("GL_WG_COMPOSTING_CH4_EF_DRY", "10.0")),
            n2o_ef_dry=Decimal(os.getenv("GL_WG_COMPOSTING_N2O_EF_DRY", "0.6")),
            default_moisture_content=Decimal(
                os.getenv("GL_WG_COMPOSTING_DEFAULT_MOISTURE", "0.5")
            ),
        )


# =============================================================================
# SECTION 8: WASTEWATER CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class WastewaterConfig:
    """
    Wastewater configuration for Waste Generated in Operations agent.

    Attributes:
        default_bo_cod: Default B0 from COD (GL_WG_WASTEWATER_DEFAULT_BO_COD)
        default_bo_bod: Default B0 from BOD (GL_WG_WASTEWATER_DEFAULT_BO_BOD)
        default_n2o_ef: Default N2O emission factor (GL_WG_WASTEWATER_DEFAULT_N2O_EF)
        default_treatment_type: Default treatment type (GL_WG_WASTEWATER_DEFAULT_TREATMENT_TYPE)
        default_mcf: Default methane correction factor (GL_WG_WASTEWATER_DEFAULT_MCF)

    Example:
        >>> wastewater = WastewaterConfig(
        ...     default_bo_cod=Decimal("0.25"),
        ...     default_bo_bod=Decimal("0.60"),
        ...     default_n2o_ef=Decimal("0.005"),
        ...     default_treatment_type="AEROBIC",
        ...     default_mcf=Decimal("0.0")
        ... )
        >>> wastewater.default_treatment_type
        'AEROBIC'
    """

    default_bo_cod: Decimal = Decimal("0.25")
    default_bo_bod: Decimal = Decimal("0.60")
    default_n2o_ef: Decimal = Decimal("0.005")
    default_treatment_type: str = "AEROBIC"
    default_mcf: Decimal = Decimal("0.0")

    def validate(self) -> None:
        """
        Validate wastewater configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_bo_cod < Decimal("0") or self.default_bo_cod > Decimal("1"):
            raise ValueError("default_bo_cod must be between 0 and 1")

        if self.default_bo_bod < Decimal("0") or self.default_bo_bod > Decimal("1"):
            raise ValueError("default_bo_bod must be between 0 and 1")

        if self.default_n2o_ef < Decimal("0"):
            raise ValueError("default_n2o_ef must be >= 0")

        valid_treatment_types = {
            "AEROBIC",
            "ANAEROBIC",
            "LAGOON",
            "SEPTIC_TANK",
            "CENTRALIZED_AEROBIC",
            "CENTRALIZED_ANAEROBIC",
        }
        if self.default_treatment_type not in valid_treatment_types:
            raise ValueError(
                f"Invalid default_treatment_type '{self.default_treatment_type}'. "
                f"Must be one of {valid_treatment_types}"
            )

        if self.default_mcf < Decimal("0") or self.default_mcf > Decimal("1"):
            raise ValueError("default_mcf must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_bo_cod": str(self.default_bo_cod),
            "default_bo_bod": str(self.default_bo_bod),
            "default_n2o_ef": str(self.default_n2o_ef),
            "default_treatment_type": self.default_treatment_type,
            "default_mcf": str(self.default_mcf),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WastewaterConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_bo_cod", "default_bo_bod", "default_n2o_ef", "default_mcf"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "WastewaterConfig":
        """Load from environment variables."""
        return cls(
            default_bo_cod=Decimal(
                os.getenv("GL_WG_WASTEWATER_DEFAULT_BO_COD", "0.25")
            ),
            default_bo_bod=Decimal(
                os.getenv("GL_WG_WASTEWATER_DEFAULT_BO_BOD", "0.60")
            ),
            default_n2o_ef=Decimal(
                os.getenv("GL_WG_WASTEWATER_DEFAULT_N2O_EF", "0.005")
            ),
            default_treatment_type=os.getenv(
                "GL_WG_WASTEWATER_DEFAULT_TREATMENT_TYPE", "AEROBIC"
            ),
            default_mcf=Decimal(os.getenv("GL_WG_WASTEWATER_DEFAULT_MCF", "0.0")),
        )


# =============================================================================
# SECTION 9: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Waste Generated in Operations agent.

    Attributes:
        compliance_frameworks: Enabled frameworks (GL_WG_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict mode (GL_WG_COMPLIANCE_STRICT_MODE)
        double_counting_check: Check for double counting (GL_WG_COMPLIANCE_DOUBLE_COUNTING_CHECK)
        boundary_enforcement: Enforce Scope 3 boundary (GL_WG_COMPLIANCE_BOUNDARY_ENFORCEMENT)

    Example:
        >>> compliance = ComplianceConfig(
        ...     compliance_frameworks="GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E5",
        ...     strict_mode=True,
        ...     double_counting_check=True,
        ...     boundary_enforcement=True
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E5']
    """

    compliance_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E5,EPA_WARM,DEFRA_BEIS,SBTI,CDP"
    )
    strict_mode: bool = True
    double_counting_check: bool = True
    boundary_enforcement: bool = True

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL_SCOPE3",
            "ISO_14064",
            "CSRD_ESRS_E5",
            "EPA_WARM",
            "DEFRA_BEIS",
            "SBTI",
            "CDP",
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
            "strict_mode": self.strict_mode,
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
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
                "GL_WG_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E5,EPA_WARM,DEFRA_BEIS,SBTI,CDP",
            ),
            strict_mode=os.getenv("GL_WG_COMPLIANCE_STRICT_MODE", "true").lower()
            == "true",
            double_counting_check=os.getenv(
                "GL_WG_COMPLIANCE_DOUBLE_COUNTING_CHECK", "true"
            ).lower()
            == "true",
            boundary_enforcement=os.getenv(
                "GL_WG_COMPLIANCE_BOUNDARY_ENFORCEMENT", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 10: EF SOURCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EFSourceConfig:
    """
    Emission Factor Source configuration for Waste Generated in Operations agent.

    Attributes:
        default_source: Default EF source (GL_WG_EF_DEFAULT_SOURCE)
        fallback_source: Fallback EF source (GL_WG_EF_FALLBACK_SOURCE)
        custom_ef_path: Custom EF file path (GL_WG_EF_CUSTOM_PATH)
        cache_ef_lookups: Cache EF lookups (GL_WG_EF_CACHE_LOOKUPS)

    Example:
        >>> ef_source = EFSourceConfig(
        ...     default_source="EPA_WARM",
        ...     fallback_source="DEFRA_BEIS",
        ...     custom_ef_path=None,
        ...     cache_ef_lookups=True
        ... )
        >>> ef_source.default_source
        'EPA_WARM'
    """

    default_source: str = "EPA_WARM"
    fallback_source: str = "DEFRA_BEIS"
    custom_ef_path: Optional[str] = None
    cache_ef_lookups: bool = True

    def validate(self) -> None:
        """
        Validate EF source configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_sources = {
            "EPA_WARM",
            "DEFRA_BEIS",
            "IPCC_2006",
            "ECOINVENT",
            "CUSTOM",
        }

        if self.default_source not in valid_sources:
            raise ValueError(
                f"Invalid default_source '{self.default_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.fallback_source not in valid_sources:
            raise ValueError(
                f"Invalid fallback_source '{self.fallback_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.default_source == "CUSTOM" and not self.custom_ef_path:
            raise ValueError("custom_ef_path must be set when default_source is CUSTOM")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_source": self.default_source,
            "fallback_source": self.fallback_source,
            "custom_ef_path": self.custom_ef_path,
            "cache_ef_lookups": self.cache_ef_lookups,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EFSourceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "EFSourceConfig":
        """Load from environment variables."""
        return cls(
            default_source=os.getenv("GL_WG_EF_DEFAULT_SOURCE", "EPA_WARM"),
            fallback_source=os.getenv("GL_WG_EF_FALLBACK_SOURCE", "DEFRA_BEIS"),
            custom_ef_path=os.getenv("GL_WG_EF_CUSTOM_PATH"),
            cache_ef_lookups=os.getenv("GL_WG_EF_CACHE_LOOKUPS", "true").lower()
            == "true",
        )


# =============================================================================
# SECTION 11: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty configuration for Waste Generated in Operations agent.

    Attributes:
        default_method: Default uncertainty method (GL_WG_UNCERTAINTY_DEFAULT_METHOD)
        default_confidence: Default confidence level (GL_WG_UNCERTAINTY_DEFAULT_CONFIDENCE)
        monte_carlo_iterations: Monte Carlo iterations (GL_WG_UNCERTAINTY_MC_ITERATIONS)
        include_parameter_uncertainty: Include parameter uncertainty (GL_WG_UNCERTAINTY_INCLUDE_PARAMETER)
        include_model_uncertainty: Include model uncertainty (GL_WG_UNCERTAINTY_INCLUDE_MODEL)

    Example:
        >>> uncertainty = UncertaintyConfig(
        ...     default_method="IPCC_DEFAULT",
        ...     default_confidence=Decimal("0.95"),
        ...     monte_carlo_iterations=10000,
        ...     include_parameter_uncertainty=True,
        ...     include_model_uncertainty=False
        ... )
        >>> uncertainty.default_method
        'IPCC_DEFAULT'
    """

    default_method: str = "IPCC_DEFAULT"
    default_confidence: Decimal = Decimal("0.95")
    monte_carlo_iterations: int = 10000
    include_parameter_uncertainty: bool = True
    include_model_uncertainty: bool = False

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {
            "IPCC_DEFAULT",
            "MONTE_CARLO",
            "BOOTSTRAP",
            "BAYESIAN",
            "NONE",
        }
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

        if self.default_confidence < Decimal("0") or self.default_confidence > Decimal("1"):
            raise ValueError("default_confidence must be between 0 and 1")

        if self.monte_carlo_iterations < 100 or self.monte_carlo_iterations > 1000000:
            raise ValueError("monte_carlo_iterations must be between 100 and 1000000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "default_confidence": str(self.default_confidence),
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "include_parameter_uncertainty": self.include_parameter_uncertainty,
            "include_model_uncertainty": self.include_model_uncertainty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_confidence" in data_copy:
            data_copy["default_confidence"] = Decimal(data_copy["default_confidence"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        return cls(
            default_method=os.getenv(
                "GL_WG_UNCERTAINTY_DEFAULT_METHOD", "IPCC_DEFAULT"
            ),
            default_confidence=Decimal(
                os.getenv("GL_WG_UNCERTAINTY_DEFAULT_CONFIDENCE", "0.95")
            ),
            monte_carlo_iterations=int(
                os.getenv("GL_WG_UNCERTAINTY_MC_ITERATIONS", "10000")
            ),
            include_parameter_uncertainty=os.getenv(
                "GL_WG_UNCERTAINTY_INCLUDE_PARAMETER", "true"
            ).lower()
            == "true",
            include_model_uncertainty=os.getenv(
                "GL_WG_UNCERTAINTY_INCLUDE_MODEL", "false"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 12: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Waste Generated in Operations agent.

    Attributes:
        enabled: Enable caching (GL_WG_CACHE_ENABLED)
        ttl_seconds: Cache TTL in seconds (GL_WG_CACHE_TTL_SECONDS)
        max_size: Max cache size entries (GL_WG_CACHE_MAX_SIZE)
        cache_ef_lookups: Cache EF lookups (GL_WG_CACHE_EF_LOOKUPS)
        cache_calculations: Cache calculations (GL_WG_CACHE_CALCULATIONS)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl_seconds=3600,
        ...     max_size=10000,
        ...     cache_ef_lookups=True,
        ...     cache_calculations=True
        ... )
        >>> cache.ttl_seconds
        3600
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    cache_ef_lookups: bool = True
    cache_calculations: bool = True

    def validate(self) -> None:
        """
        Validate cache configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.ttl_seconds < 1 or self.ttl_seconds > 86400:
            raise ValueError("ttl_seconds must be between 1 and 86400 (24 hours)")

        if self.max_size < 1 or self.max_size > 1000000:
            raise ValueError("max_size must be between 1 and 1000000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "cache_ef_lookups": self.cache_ef_lookups,
            "cache_calculations": self.cache_calculations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_WG_CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("GL_WG_CACHE_TTL_SECONDS", "3600")),
            max_size=int(os.getenv("GL_WG_CACHE_MAX_SIZE", "10000")),
            cache_ef_lookups=os.getenv("GL_WG_CACHE_EF_LOOKUPS", "true").lower()
            == "true",
            cache_calculations=os.getenv("GL_WG_CACHE_CALCULATIONS", "true").lower()
            == "true",
        )


# =============================================================================
# SECTION 13: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    API configuration for Waste Generated in Operations agent.

    Attributes:
        api_prefix: API route prefix (GL_WG_API_PREFIX)
        max_batch_size: Max batch size (GL_WG_API_MAX_BATCH_SIZE)
        rate_limit: Requests per minute (GL_WG_API_RATE_LIMIT)
        timeout: Request timeout in seconds (GL_WG_API_TIMEOUT)
        worker_threads: Worker thread count (GL_WG_WORKER_THREADS)

    Example:
        >>> api = APIConfig(
        ...     api_prefix="/api/v1/waste-generated",
        ...     max_batch_size=500,
        ...     rate_limit=100,
        ...     timeout=300,
        ...     worker_threads=4
        ... )
        >>> api.api_prefix
        '/api/v1/waste-generated'
    """

    api_prefix: str = "/api/v1/waste-generated"
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
            api_prefix=os.getenv("GL_WG_API_PREFIX", "/api/v1/waste-generated"),
            max_batch_size=int(os.getenv("GL_WG_API_MAX_BATCH_SIZE", "500")),
            rate_limit=int(os.getenv("GL_WG_API_RATE_LIMIT", "100")),
            timeout=int(os.getenv("GL_WG_API_TIMEOUT", "300")),
            worker_threads=int(os.getenv("GL_WG_WORKER_THREADS", "4")),
        )


# =============================================================================
# SECTION 14: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance configuration for Waste Generated in Operations agent.

    Attributes:
        provenance_enabled: Enable provenance tracking (GL_WG_PROVENANCE_ENABLED)
        provenance_algorithm: Hash algorithm (GL_WG_PROVENANCE_ALGORITHM)
        provenance_chain: Chain hashes (GL_WG_PROVENANCE_CHAIN)

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
            provenance_enabled=os.getenv("GL_WG_PROVENANCE_ENABLED", "true").lower()
            == "true",
            provenance_algorithm=os.getenv("GL_WG_PROVENANCE_ALGORITHM", "sha256"),
            provenance_chain=os.getenv("GL_WG_PROVENANCE_CHAIN", "true").lower()
            == "true",
        )


# =============================================================================
# SECTION 15: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Waste Generated in Operations agent.

    Attributes:
        metrics_enabled: Enable metrics collection (GL_WG_METRICS_ENABLED)
        metrics_prefix: Metrics name prefix (GL_WG_METRICS_PREFIX)
        metrics_buckets: Histogram buckets (GL_WG_METRICS_BUCKETS)

    Example:
        >>> metrics = MetricsConfig(
        ...     metrics_enabled=True,
        ...     metrics_prefix="gl_wg_",
        ...     metrics_buckets="0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_buckets()
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    """

    metrics_enabled: bool = True
    metrics_prefix: str = "gl_wg_"
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
            metrics_enabled=os.getenv("GL_WG_METRICS_ENABLED", "true").lower()
            == "true",
            metrics_prefix=os.getenv("GL_WG_METRICS_PREFIX", "gl_wg_"),
            metrics_buckets=os.getenv(
                "GL_WG_METRICS_BUCKETS", "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
            ),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


class WasteGeneratedConfig:
    """
    Master configuration class for Waste Generated in Operations agent.

    This class aggregates all configuration sections and provides a unified
    interface for accessing configuration values. It implements the singleton
    pattern with thread-safe access.

    Attributes:
        general: General configuration
        database: Database configuration
        redis: Redis configuration
        landfill: Landfill configuration
        incineration: Incineration configuration
        recycling: Recycling configuration
        composting: Composting configuration
        wastewater: Wastewater configuration
        compliance: Compliance configuration
        ef_source: EF source configuration
        uncertainty: Uncertainty configuration
        cache: Cache configuration
        api: API configuration
        provenance: Provenance configuration
        metrics: Metrics configuration

    Example:
        >>> config = WasteGeneratedConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-005'
        >>> config.landfill.default_docf
        Decimal('0.5')
        >>> config.validate_all()
    """

    def __init__(
        self,
        general: GeneralConfig,
        database: DatabaseConfig,
        redis: RedisConfig,
        landfill: LandfillConfig,
        incineration: IncinerationConfig,
        recycling: RecyclingConfig,
        composting: CompostingConfig,
        wastewater: WastewaterConfig,
        compliance: ComplianceConfig,
        ef_source: EFSourceConfig,
        uncertainty: UncertaintyConfig,
        cache: CacheConfig,
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
            landfill: Landfill configuration
            incineration: Incineration configuration
            recycling: Recycling configuration
            composting: Composting configuration
            wastewater: Wastewater configuration
            compliance: Compliance configuration
            ef_source: EF source configuration
            uncertainty: Uncertainty configuration
            cache: Cache configuration
            api: API configuration
            provenance: Provenance configuration
            metrics: Metrics configuration
        """
        self.general = general
        self.database = database
        self.redis = redis
        self.landfill = landfill
        self.incineration = incineration
        self.recycling = recycling
        self.composting = composting
        self.wastewater = wastewater
        self.compliance = compliance
        self.ef_source = ef_source
        self.uncertainty = uncertainty
        self.cache = cache
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
        self.landfill.validate()
        self.incineration.validate()
        self.recycling.validate()
        self.composting.validate()
        self.wastewater.validate()
        self.compliance.validate()
        self.ef_source.validate()
        self.uncertainty.validate()
        self.cache.validate()
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
            "landfill": self.landfill.to_dict(),
            "incineration": self.incineration.to_dict(),
            "recycling": self.recycling.to_dict(),
            "composting": self.composting.to_dict(),
            "wastewater": self.wastewater.to_dict(),
            "compliance": self.compliance.to_dict(),
            "ef_source": self.ef_source.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "cache": self.cache.to_dict(),
            "api": self.api.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WasteGeneratedConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            WasteGeneratedConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data["general"]),
            database=DatabaseConfig.from_dict(data["database"]),
            redis=RedisConfig.from_dict(data["redis"]),
            landfill=LandfillConfig.from_dict(data["landfill"]),
            incineration=IncinerationConfig.from_dict(data["incineration"]),
            recycling=RecyclingConfig.from_dict(data["recycling"]),
            composting=CompostingConfig.from_dict(data["composting"]),
            wastewater=WastewaterConfig.from_dict(data["wastewater"]),
            compliance=ComplianceConfig.from_dict(data["compliance"]),
            ef_source=EFSourceConfig.from_dict(data["ef_source"]),
            uncertainty=UncertaintyConfig.from_dict(data["uncertainty"]),
            cache=CacheConfig.from_dict(data["cache"]),
            api=APIConfig.from_dict(data["api"]),
            provenance=ProvenanceConfig.from_dict(data["provenance"]),
            metrics=MetricsConfig.from_dict(data["metrics"]),
        )

    @classmethod
    def from_env(cls) -> "WasteGeneratedConfig":
        """
        Load configuration from environment variables.

        Returns:
            WasteGeneratedConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            landfill=LandfillConfig.from_env(),
            incineration=IncinerationConfig.from_env(),
            recycling=RecyclingConfig.from_env(),
            composting=CompostingConfig.from_env(),
            wastewater=WastewaterConfig.from_env(),
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


_config_instance: Optional[WasteGeneratedConfig] = None
_config_lock = threading.RLock()


def get_config() -> WasteGeneratedConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton. The first call will load configuration from
    environment variables. Subsequent calls return the cached instance.

    Returns:
        WasteGeneratedConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-005'

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The configuration is initialized only once.
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            # Double-checked locking pattern
            if _config_instance is None:
                _config_instance = WasteGeneratedConfig.from_env()
                _config_instance.validate_all()

    return _config_instance


def set_config(config: WasteGeneratedConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.

    Args:
        config: WasteGeneratedConfig instance to set as singleton

    Example:
        >>> custom_config = WasteGeneratedConfig.from_dict({...})
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


def _load_from_env() -> WasteGeneratedConfig:
    """
    Load configuration from environment variables (internal helper).

    This is an internal helper function that loads all configuration sections
    from environment variables. Use get_config() instead for normal usage.

    Returns:
        WasteGeneratedConfig instance loaded from environment

    Note:
        This function is for internal use. Use get_config() for normal access.
    """
    return WasteGeneratedConfig.from_env()


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================


def validate_config(config: WasteGeneratedConfig) -> List[str]:
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
        ("landfill", config.landfill),
        ("incineration", config.incineration),
        ("recycling", config.recycling),
        ("composting", config.composting),
        ("wastewater", config.wastewater),
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

    return errors


def print_config(config: WasteGeneratedConfig) -> None:
    """
    Print configuration in human-readable format.

    This function prints all configuration sections in a formatted,
    human-readable manner. Useful for debugging and verification.

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
        ===== Waste Generated in Operations Configuration =====
        [GENERAL]
        enabled: True
        debug: False
        ...
    """
    print("===== Waste Generated in Operations Configuration =====")
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

    print("\n[LANDFILL]")
    for key, value in config.landfill.to_dict().items():
        print(f"{key}: {value}")

    print("\n[INCINERATION]")
    for key, value in config.incineration.to_dict().items():
        print(f"{key}: {value}")

    print("\n[RECYCLING]")
    for key, value in config.recycling.to_dict().items():
        print(f"{key}: {value}")

    print("\n[COMPOSTING]")
    for key, value in config.composting.to_dict().items():
        print(f"{key}: {value}")

    print("\n[WASTEWATER]")
    for key, value in config.wastewater.to_dict().items():
        print(f"{key}: {value}")

    print("\n[COMPLIANCE]")
    for key, value in config.compliance.to_dict().items():
        print(f"{key}: {value}")

    print("\n[EF_SOURCE]")
    for key, value in config.ef_source.to_dict().items():
        print(f"{key}: {value}")

    print("\n[UNCERTAINTY]")
    for key, value in config.uncertainty.to_dict().items():
        print(f"{key}: {value}")

    print("\n[CACHE]")
    for key, value in config.cache.to_dict().items():
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
    "LandfillConfig",
    "IncinerationConfig",
    "RecyclingConfig",
    "CompostingConfig",
    "WastewaterConfig",
    "ComplianceConfig",
    "EFSourceConfig",
    "UncertaintyConfig",
    "CacheConfig",
    "APIConfig",
    "ProvenanceConfig",
    "MetricsConfig",
    "WasteGeneratedConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
]
