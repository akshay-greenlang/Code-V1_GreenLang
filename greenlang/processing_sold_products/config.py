# -*- coding: utf-8 -*-
"""
Processing of Sold Products Configuration - AGENT-MRV-023

Thread-safe singleton configuration for GL-MRV-S3-010.
All environment variables prefixed with GL_PSP_.

This module provides comprehensive configuration management for the Processing
of Sold Products agent (GHG Protocol Scope 3 Category 10), supporting:
- Site-specific processing emissions (direct measurement, energy, fuel)
- Average-data processing emissions (industry-average EFs per product type)
- Spend-based processing emissions (EEIO with CPI deflation, margin removal)
- Hybrid aggregation (method waterfall with gap-filling)
- Multi-step processing chains (sequential transformation stages)
- Allocation methods (mass, revenue, units, equal, proportional)
- Grid electricity factors (eGRID, IEA, EU EEA, residual mix)
- Fuel combustion factors (upstream WTT, direct combustion)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD, CDP, SBTi, GRI, DEFRA)
- Double-counting prevention (vs Scope 1/2 and Cat 1/Cat 2/Cat 11/Cat 12)
- Data Quality Indicator (5-dimension scoring)
- Uncertainty quantification (Monte Carlo, IPCC default, bootstrap)
- Provenance tracking and Merkle-tree audit trails

GHG Protocol Scope 3 Category 10 covers emissions from processing of sold
intermediate products by third parties (e.g., manufacturers) after sale by the
reporting company and before end use. This applies to companies that sell
intermediate products requiring further processing, transformation, or
inclusion in another product before use by the end consumer.

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-010'
    >>> config.processing.default_processing_type
    'INDUSTRIAL_MANUFACTURING'
    >>> config.compliance.double_counting_check
    True

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_PSP_ prefix. See individual config sections for specific variables.
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
    General configuration for Processing of Sold Products agent.

    Attributes:
        enabled: Master switch for the agent (GL_PSP_ENABLED)
        debug: Enable debug mode with verbose logging (GL_PSP_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_PSP_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_PSP_AGENT_ID)
        agent_component: Agent component identifier (GL_PSP_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_PSP_VERSION)
        table_prefix: Database table prefix (GL_PSP_TABLE_PREFIX)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-010",
        ...     agent_component="AGENT-MRV-023",
        ...     version="1.0.0",
        ...     table_prefix="gl_psp_"
        ... )
        >>> general.agent_id
        'GL-MRV-S3-010'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-010"
    agent_component: str = "AGENT-MRV-023"
    version: str = "1.0.0"
    table_prefix: str = "gl_psp_"

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
            enabled=os.getenv("GL_PSP_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_PSP_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_PSP_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_PSP_AGENT_ID", "GL-MRV-S3-010"),
            agent_component=os.getenv("GL_PSP_AGENT_COMPONENT", "AGENT-MRV-023"),
            version=os.getenv("GL_PSP_VERSION", "1.0.0"),
            table_prefix=os.getenv("GL_PSP_TABLE_PREFIX", "gl_psp_"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Processing of Sold Products agent.

    Attributes:
        database_url: PostgreSQL connection URL (GL_PSP_DATABASE_URL)
        pool_size: Connection pool size (GL_PSP_DATABASE_POOL_SIZE)
        max_overflow: Max overflow connections (GL_PSP_DATABASE_MAX_OVERFLOW)
        pool_timeout: Pool timeout in seconds (GL_PSP_DATABASE_POOL_TIMEOUT)
        pool_recycle: Connection recycle interval in seconds (GL_PSP_DATABASE_POOL_RECYCLE)
        schema: Database schema name (GL_PSP_DATABASE_SCHEMA)
        statement_timeout: Statement timeout in milliseconds (GL_PSP_DATABASE_STATEMENT_TIMEOUT)
        retry_attempts: Number of retry attempts for transient failures (GL_PSP_DATABASE_RETRY_ATTEMPTS)
        retry_delay: Delay between retries in seconds (GL_PSP_DATABASE_RETRY_DELAY)

    Example:
        >>> db = DatabaseConfig(
        ...     database_url="postgresql://user:pass@localhost:5432/greenlang",
        ...     pool_size=10,
        ...     max_overflow=20,
        ...     pool_timeout=30,
        ...     pool_recycle=3600,
        ...     schema="processing_sold_products_service",
        ...     statement_timeout=60000,
        ...     retry_attempts=3,
        ...     retry_delay=1
        ... )
        >>> db.schema
        'processing_sold_products_service'
    """

    database_url: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    schema: str = "processing_sold_products_service"
    statement_timeout: int = 60000
    retry_attempts: int = 3
    retry_delay: int = 1

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

        if self.pool_recycle < 0:
            raise ValueError("pool_recycle must be >= 0")

        if not self.schema:
            raise ValueError("schema cannot be empty")

        if self.statement_timeout < 1000:
            raise ValueError("statement_timeout must be >= 1000 milliseconds")

        if self.retry_attempts < 0 or self.retry_attempts > 10:
            raise ValueError("retry_attempts must be between 0 and 10")

        if self.retry_delay < 0 or self.retry_delay > 60:
            raise ValueError("retry_delay must be between 0 and 60 seconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "schema": self.schema,
            "statement_timeout": self.statement_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
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
                "GL_PSP_DATABASE_URL",
                "postgresql://greenlang:greenlang@localhost:5432/greenlang",
            ),
            pool_size=int(os.getenv("GL_PSP_DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("GL_PSP_DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("GL_PSP_DATABASE_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("GL_PSP_DATABASE_POOL_RECYCLE", "3600")),
            schema=os.getenv(
                "GL_PSP_DATABASE_SCHEMA", "processing_sold_products_service"
            ),
            statement_timeout=int(
                os.getenv("GL_PSP_DATABASE_STATEMENT_TIMEOUT", "60000")
            ),
            retry_attempts=int(os.getenv("GL_PSP_DATABASE_RETRY_ATTEMPTS", "3")),
            retry_delay=int(os.getenv("GL_PSP_DATABASE_RETRY_DELAY", "1")),
        )


# =============================================================================
# SECTION 3: PROCESSING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProcessingConfig:
    """
    Processing configuration for Processing of Sold Products agent.

    Controls the core processing behaviour for Category 10 calculations
    including the default processing type, energy intensity source, and
    whether multi-step chain processing is enabled.

    Attributes:
        default_processing_type: Default processing type (GL_PSP_PROCESSING_DEFAULT_TYPE)
        energy_intensity_source: Default energy intensity data source (GL_PSP_PROCESSING_ENERGY_SOURCE)
        chain_enabled: Enable multi-step processing chain tracking (GL_PSP_PROCESSING_CHAIN_ENABLED)
        default_gwp: Default GWP assessment report version (GL_PSP_PROCESSING_DEFAULT_GWP)
        include_upstream_energy: Include upstream energy emissions (GL_PSP_PROCESSING_INCLUDE_UPSTREAM)
        max_chain_steps: Maximum processing chain steps (GL_PSP_PROCESSING_MAX_CHAIN_STEPS)

    Example:
        >>> processing = ProcessingConfig(
        ...     default_processing_type="INDUSTRIAL_MANUFACTURING",
        ...     energy_intensity_source="IEA",
        ...     chain_enabled=True,
        ...     default_gwp="AR5",
        ...     include_upstream_energy=True,
        ...     max_chain_steps=10
        ... )
        >>> processing.default_processing_type
        'INDUSTRIAL_MANUFACTURING'
    """

    default_processing_type: str = "INDUSTRIAL_MANUFACTURING"
    energy_intensity_source: str = "IEA"
    chain_enabled: bool = True
    default_gwp: str = "AR5"
    include_upstream_energy: bool = True
    max_chain_steps: int = 10

    def validate(self) -> None:
        """
        Validate processing configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_types = {
            "INDUSTRIAL_MANUFACTURING",
            "CHEMICAL_PROCESSING",
            "METAL_SMELTING",
            "FOOD_PROCESSING",
            "TEXTILE_PROCESSING",
            "PAPER_PULP",
            "ASSEMBLY",
            "REFINING",
            "MIXING_BLENDING",
            "PACKAGING",
            "CUSTOM",
        }
        if self.default_processing_type not in valid_types:
            raise ValueError(
                f"Invalid default_processing_type '{self.default_processing_type}'. "
                f"Must be one of {valid_types}"
            )

        valid_energy_sources = {"IEA", "EGRID", "EU_EEA", "DEFRA", "CUSTOM"}
        if self.energy_intensity_source not in valid_energy_sources:
            raise ValueError(
                f"Invalid energy_intensity_source '{self.energy_intensity_source}'. "
                f"Must be one of {valid_energy_sources}"
            )

        valid_gwp_versions = {"AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp_versions:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. "
                f"Must be one of {valid_gwp_versions}"
            )

        if self.max_chain_steps < 1 or self.max_chain_steps > 50:
            raise ValueError("max_chain_steps must be between 1 and 50")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_processing_type": self.default_processing_type,
            "energy_intensity_source": self.energy_intensity_source,
            "chain_enabled": self.chain_enabled,
            "default_gwp": self.default_gwp,
            "include_upstream_energy": self.include_upstream_energy,
            "max_chain_steps": self.max_chain_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        """Load from environment variables."""
        return cls(
            default_processing_type=os.getenv(
                "GL_PSP_PROCESSING_DEFAULT_TYPE", "INDUSTRIAL_MANUFACTURING"
            ),
            energy_intensity_source=os.getenv(
                "GL_PSP_PROCESSING_ENERGY_SOURCE", "IEA"
            ),
            chain_enabled=os.getenv(
                "GL_PSP_PROCESSING_CHAIN_ENABLED", "true"
            ).lower()
            == "true",
            default_gwp=os.getenv("GL_PSP_PROCESSING_DEFAULT_GWP", "AR5"),
            include_upstream_energy=os.getenv(
                "GL_PSP_PROCESSING_INCLUDE_UPSTREAM", "true"
            ).lower()
            == "true",
            max_chain_steps=int(
                os.getenv("GL_PSP_PROCESSING_MAX_CHAIN_STEPS", "10")
            ),
        )


# =============================================================================
# SECTION 4: SITE-SPECIFIC CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SiteSpecificConfig:
    """
    Site-specific calculation configuration.

    Controls parameters for site-specific (primary data) calculations where
    the downstream processor provides direct emissions data.

    Attributes:
        enable_direct: Enable direct emissions data (GL_PSP_SITE_ENABLE_DIRECT)
        enable_energy: Enable energy-based calculations (GL_PSP_SITE_ENABLE_ENERGY)
        enable_fuel: Enable fuel-based calculations (GL_PSP_SITE_ENABLE_FUEL)
        data_validation_strict: Strict validation for site data (GL_PSP_SITE_STRICT_VALIDATION)
        require_third_party_verification: Require third-party verification (GL_PSP_SITE_REQUIRE_VERIFICATION)
        min_completeness_pct: Minimum data completeness percentage (GL_PSP_SITE_MIN_COMPLETENESS)

    Example:
        >>> site = SiteSpecificConfig(
        ...     enable_direct=True,
        ...     enable_energy=True,
        ...     enable_fuel=True,
        ...     data_validation_strict=True,
        ...     require_third_party_verification=False,
        ...     min_completeness_pct=Decimal("80.0")
        ... )
        >>> site.enable_direct
        True
    """

    enable_direct: bool = True
    enable_energy: bool = True
    enable_fuel: bool = True
    data_validation_strict: bool = True
    require_third_party_verification: bool = False
    min_completeness_pct: Decimal = Decimal("80.0")

    def validate(self) -> None:
        """
        Validate site-specific configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.min_completeness_pct < Decimal("0") or self.min_completeness_pct > Decimal("100"):
            raise ValueError("min_completeness_pct must be between 0 and 100")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_direct": self.enable_direct,
            "enable_energy": self.enable_energy,
            "enable_fuel": self.enable_fuel,
            "data_validation_strict": self.data_validation_strict,
            "require_third_party_verification": self.require_third_party_verification,
            "min_completeness_pct": str(self.min_completeness_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteSpecificConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "min_completeness_pct" in data_copy:
            data_copy["min_completeness_pct"] = Decimal(data_copy["min_completeness_pct"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SiteSpecificConfig":
        """Load from environment variables."""
        return cls(
            enable_direct=os.getenv("GL_PSP_SITE_ENABLE_DIRECT", "true").lower() == "true",
            enable_energy=os.getenv("GL_PSP_SITE_ENABLE_ENERGY", "true").lower() == "true",
            enable_fuel=os.getenv("GL_PSP_SITE_ENABLE_FUEL", "true").lower() == "true",
            data_validation_strict=os.getenv(
                "GL_PSP_SITE_STRICT_VALIDATION", "true"
            ).lower()
            == "true",
            require_third_party_verification=os.getenv(
                "GL_PSP_SITE_REQUIRE_VERIFICATION", "false"
            ).lower()
            == "true",
            min_completeness_pct=Decimal(
                os.getenv("GL_PSP_SITE_MIN_COMPLETENESS", "80.0")
            ),
        )


# =============================================================================
# SECTION 5: AVERAGE-DATA CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AverageDataConfig:
    """
    Average-data calculation configuration.

    Controls parameters for average-data calculations using industry-average
    emission factors from DEFRA, EPA, or ecoinvent databases.

    Attributes:
        default_ef_source: Default emission factor source (GL_PSP_AVG_DEFAULT_EF_SOURCE)
        fallback_ef_source: Fallback EF source (GL_PSP_AVG_FALLBACK_EF_SOURCE)
        fallback_to_global: Fall back to global averages (GL_PSP_AVG_FALLBACK_GLOBAL)
        custom_ef_path: Path to custom EF file (GL_PSP_AVG_CUSTOM_EF_PATH)
        cache_ef_lookups: Cache emission factor lookups (GL_PSP_AVG_CACHE_LOOKUPS)
        prefer_region_specific: Prefer region-specific EFs (GL_PSP_AVG_PREFER_REGIONAL)

    Example:
        >>> avg = AverageDataConfig(
        ...     default_ef_source="DEFRA",
        ...     fallback_ef_source="EPA",
        ...     fallback_to_global=True,
        ...     custom_ef_path=None,
        ...     cache_ef_lookups=True,
        ...     prefer_region_specific=True
        ... )
        >>> avg.default_ef_source
        'DEFRA'
    """

    default_ef_source: str = "DEFRA"
    fallback_ef_source: str = "EPA"
    fallback_to_global: bool = True
    custom_ef_path: Optional[str] = None
    cache_ef_lookups: bool = True
    prefer_region_specific: bool = True

    def validate(self) -> None:
        """
        Validate average-data configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_sources = {"DEFRA", "EPA", "ECOINVENT", "IPCC", "CUSTOM"}
        if self.default_ef_source not in valid_sources:
            raise ValueError(
                f"Invalid default_ef_source '{self.default_ef_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.fallback_ef_source not in valid_sources:
            raise ValueError(
                f"Invalid fallback_ef_source '{self.fallback_ef_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.default_ef_source == "CUSTOM" and not self.custom_ef_path:
            raise ValueError("custom_ef_path must be set when default_ef_source is CUSTOM")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_ef_source": self.default_ef_source,
            "fallback_ef_source": self.fallback_ef_source,
            "fallback_to_global": self.fallback_to_global,
            "custom_ef_path": self.custom_ef_path,
            "cache_ef_lookups": self.cache_ef_lookups,
            "prefer_region_specific": self.prefer_region_specific,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AverageDataConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AverageDataConfig":
        """Load from environment variables."""
        return cls(
            default_ef_source=os.getenv("GL_PSP_AVG_DEFAULT_EF_SOURCE", "DEFRA"),
            fallback_ef_source=os.getenv("GL_PSP_AVG_FALLBACK_EF_SOURCE", "EPA"),
            fallback_to_global=os.getenv(
                "GL_PSP_AVG_FALLBACK_GLOBAL", "true"
            ).lower()
            == "true",
            custom_ef_path=os.getenv("GL_PSP_AVG_CUSTOM_EF_PATH"),
            cache_ef_lookups=os.getenv(
                "GL_PSP_AVG_CACHE_LOOKUPS", "true"
            ).lower()
            == "true",
            prefer_region_specific=os.getenv(
                "GL_PSP_AVG_PREFER_REGIONAL", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 6: SPEND-BASED CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SpendBasedConfig:
    """
    Spend-based calculation configuration.

    Controls parameters for spend-based calculations using EEIO models
    (USEEIO, EXIOBASE, DEFRA), including CPI deflation and margin removal.

    Attributes:
        eeio_source: EEIO model source (GL_PSP_SPEND_EEIO_SOURCE)
        margin_adjustment: Enable margin removal from spend (GL_PSP_SPEND_MARGIN_ADJUST)
        default_margin_pct: Default margin percentage to remove (GL_PSP_SPEND_DEFAULT_MARGIN)
        cpi_base_year: CPI deflation base year (GL_PSP_SPEND_CPI_BASE_YEAR)
        cpi_enabled: Enable CPI deflation (GL_PSP_SPEND_CPI_ENABLED)
        default_currency: Default currency for spend data (GL_PSP_SPEND_DEFAULT_CURRENCY)
        exchange_rate_source: Exchange rate data source (GL_PSP_SPEND_FX_SOURCE)

    Example:
        >>> spend = SpendBasedConfig(
        ...     eeio_source="USEEIO",
        ...     margin_adjustment=True,
        ...     default_margin_pct=Decimal("20.0"),
        ...     cpi_base_year=2021,
        ...     cpi_enabled=True,
        ...     default_currency="USD",
        ...     exchange_rate_source="ECB"
        ... )
        >>> spend.eeio_source
        'USEEIO'
    """

    eeio_source: str = "USEEIO"
    margin_adjustment: bool = True
    default_margin_pct: Decimal = Decimal("20.0")
    cpi_base_year: int = 2021
    cpi_enabled: bool = True
    default_currency: str = "USD"
    exchange_rate_source: str = "ECB"

    def validate(self) -> None:
        """
        Validate spend-based configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_eeio_sources = {"USEEIO", "EXIOBASE", "DEFRA", "CUSTOM"}
        if self.eeio_source not in valid_eeio_sources:
            raise ValueError(
                f"Invalid eeio_source '{self.eeio_source}'. "
                f"Must be one of {valid_eeio_sources}"
            )

        if self.default_margin_pct < Decimal("0") or self.default_margin_pct > Decimal("100"):
            raise ValueError("default_margin_pct must be between 0 and 100")

        if self.cpi_base_year < 2000 or self.cpi_base_year > 2030:
            raise ValueError("cpi_base_year must be between 2000 and 2030")

        valid_currencies = {
            "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "AUD", "CAD",
            "SEK", "NOK", "DKK", "INR", "BRL", "KRW", "SGD", "HKD",
            "NZD", "ZAR", "MXN", "TWD",
        }
        if self.default_currency not in valid_currencies:
            raise ValueError(
                f"Invalid default_currency '{self.default_currency}'. "
                f"Must be one of {valid_currencies}"
            )

        valid_fx_sources = {"ECB", "FED", "OANDA", "CUSTOM"}
        if self.exchange_rate_source not in valid_fx_sources:
            raise ValueError(
                f"Invalid exchange_rate_source '{self.exchange_rate_source}'. "
                f"Must be one of {valid_fx_sources}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "eeio_source": self.eeio_source,
            "margin_adjustment": self.margin_adjustment,
            "default_margin_pct": str(self.default_margin_pct),
            "cpi_base_year": self.cpi_base_year,
            "cpi_enabled": self.cpi_enabled,
            "default_currency": self.default_currency,
            "exchange_rate_source": self.exchange_rate_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendBasedConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_margin_pct" in data_copy:
            data_copy["default_margin_pct"] = Decimal(data_copy["default_margin_pct"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "SpendBasedConfig":
        """Load from environment variables."""
        return cls(
            eeio_source=os.getenv("GL_PSP_SPEND_EEIO_SOURCE", "USEEIO"),
            margin_adjustment=os.getenv(
                "GL_PSP_SPEND_MARGIN_ADJUST", "true"
            ).lower()
            == "true",
            default_margin_pct=Decimal(
                os.getenv("GL_PSP_SPEND_DEFAULT_MARGIN", "20.0")
            ),
            cpi_base_year=int(os.getenv("GL_PSP_SPEND_CPI_BASE_YEAR", "2021")),
            cpi_enabled=os.getenv("GL_PSP_SPEND_CPI_ENABLED", "true").lower() == "true",
            default_currency=os.getenv("GL_PSP_SPEND_DEFAULT_CURRENCY", "USD"),
            exchange_rate_source=os.getenv("GL_PSP_SPEND_FX_SOURCE", "ECB"),
        )


# =============================================================================
# SECTION 7: HYBRID CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class HybridConfig:
    """
    Hybrid calculation configuration.

    Controls method waterfall order and gap-filling for hybrid aggregation
    that combines site-specific, average-data, and spend-based methods.

    Attributes:
        method_waterfall: Ordered list of methods to try (GL_PSP_HYBRID_WATERFALL)
        gap_fill_enabled: Enable gap-filling with lower-quality methods (GL_PSP_HYBRID_GAP_FILL)
        min_site_specific_pct: Minimum percentage of site-specific data (GL_PSP_HYBRID_MIN_SITE_PCT)
        blend_weights_enabled: Enable weighted blending of methods (GL_PSP_HYBRID_BLEND_WEIGHTS)
        auto_select_method: Auto-select best method per product (GL_PSP_HYBRID_AUTO_SELECT)

    Example:
        >>> hybrid = HybridConfig(
        ...     method_waterfall="SITE_SPECIFIC,AVERAGE_DATA,SPEND_BASED",
        ...     gap_fill_enabled=True,
        ...     min_site_specific_pct=Decimal("50.0"),
        ...     blend_weights_enabled=False,
        ...     auto_select_method=True
        ... )
        >>> hybrid.get_waterfall()
        ['SITE_SPECIFIC', 'AVERAGE_DATA', 'SPEND_BASED']
    """

    method_waterfall: str = "SITE_SPECIFIC,AVERAGE_DATA,SPEND_BASED"
    gap_fill_enabled: bool = True
    min_site_specific_pct: Decimal = Decimal("50.0")
    blend_weights_enabled: bool = False
    auto_select_method: bool = True

    def validate(self) -> None:
        """
        Validate hybrid configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"SITE_SPECIFIC", "AVERAGE_DATA", "SPEND_BASED"}
        waterfall = self.get_waterfall()
        if not waterfall:
            raise ValueError("method_waterfall cannot be empty")

        for method in waterfall:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method in waterfall '{method}'. "
                    f"Must be one of {valid_methods}"
                )

        if self.min_site_specific_pct < Decimal("0") or self.min_site_specific_pct > Decimal("100"):
            raise ValueError("min_site_specific_pct must be between 0 and 100")

    def get_waterfall(self) -> List[str]:
        """Parse method waterfall string into list."""
        return [m.strip() for m in self.method_waterfall.split(",") if m.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_waterfall": self.method_waterfall,
            "gap_fill_enabled": self.gap_fill_enabled,
            "min_site_specific_pct": str(self.min_site_specific_pct),
            "blend_weights_enabled": self.blend_weights_enabled,
            "auto_select_method": self.auto_select_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "min_site_specific_pct" in data_copy:
            data_copy["min_site_specific_pct"] = Decimal(data_copy["min_site_specific_pct"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "HybridConfig":
        """Load from environment variables."""
        return cls(
            method_waterfall=os.getenv(
                "GL_PSP_HYBRID_WATERFALL", "SITE_SPECIFIC,AVERAGE_DATA,SPEND_BASED"
            ),
            gap_fill_enabled=os.getenv(
                "GL_PSP_HYBRID_GAP_FILL", "true"
            ).lower()
            == "true",
            min_site_specific_pct=Decimal(
                os.getenv("GL_PSP_HYBRID_MIN_SITE_PCT", "50.0")
            ),
            blend_weights_enabled=os.getenv(
                "GL_PSP_HYBRID_BLEND_WEIGHTS", "false"
            ).lower()
            == "true",
            auto_select_method=os.getenv(
                "GL_PSP_HYBRID_AUTO_SELECT", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 8: ALLOCATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AllocationConfig:
    """
    Allocation configuration for Processing of Sold Products agent.

    Controls how processing emissions are allocated across sold products
    when a processor handles multiple inputs.

    Attributes:
        default_method: Default allocation method (GL_PSP_ALLOC_DEFAULT_METHOD)
        proportional_enabled: Enable proportional allocation (GL_PSP_ALLOC_PROPORTIONAL)
        allow_equal_split: Allow equal allocation across products (GL_PSP_ALLOC_EQUAL_SPLIT)
        require_allocation_justification: Require justification (GL_PSP_ALLOC_REQUIRE_JUSTIF)
        default_mass_unit: Default mass unit for mass-based allocation (GL_PSP_ALLOC_MASS_UNIT)

    Example:
        >>> alloc = AllocationConfig(
        ...     default_method="MASS",
        ...     proportional_enabled=True,
        ...     allow_equal_split=True,
        ...     require_allocation_justification=False,
        ...     default_mass_unit="TONNES"
        ... )
        >>> alloc.default_method
        'MASS'
    """

    default_method: str = "MASS"
    proportional_enabled: bool = True
    allow_equal_split: bool = True
    require_allocation_justification: bool = False
    default_mass_unit: str = "TONNES"

    def validate(self) -> None:
        """
        Validate allocation configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"MASS", "REVENUE", "UNITS", "EQUAL", "ENERGY", "CUSTOM"}
        if self.default_method not in valid_methods:
            raise ValueError(
                f"Invalid default_method '{self.default_method}'. "
                f"Must be one of {valid_methods}"
            )

        valid_mass_units = {"TONNES", "KG", "LBS"}
        if self.default_mass_unit not in valid_mass_units:
            raise ValueError(
                f"Invalid default_mass_unit '{self.default_mass_unit}'. "
                f"Must be one of {valid_mass_units}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_method": self.default_method,
            "proportional_enabled": self.proportional_enabled,
            "allow_equal_split": self.allow_equal_split,
            "require_allocation_justification": self.require_allocation_justification,
            "default_mass_unit": self.default_mass_unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AllocationConfig":
        """Load from environment variables."""
        return cls(
            default_method=os.getenv("GL_PSP_ALLOC_DEFAULT_METHOD", "MASS"),
            proportional_enabled=os.getenv(
                "GL_PSP_ALLOC_PROPORTIONAL", "true"
            ).lower()
            == "true",
            allow_equal_split=os.getenv(
                "GL_PSP_ALLOC_EQUAL_SPLIT", "true"
            ).lower()
            == "true",
            require_allocation_justification=os.getenv(
                "GL_PSP_ALLOC_REQUIRE_JUSTIF", "false"
            ).lower()
            == "true",
            default_mass_unit=os.getenv("GL_PSP_ALLOC_MASS_UNIT", "TONNES"),
        )


# =============================================================================
# SECTION 9: GRID FACTORS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GridFactorsConfig:
    """
    Grid electricity emission factor configuration.

    Controls how grid emission factors are resolved for electricity consumed
    during processing of sold products at downstream facilities.

    Attributes:
        default_region: Default grid region (GL_PSP_GRID_DEFAULT_REGION)
        update_frequency: EF update frequency in hours (GL_PSP_GRID_UPDATE_FREQ)
        fallback_to_global: Fall back to global grid average (GL_PSP_GRID_FALLBACK_GLOBAL)
        default_source: Default grid EF source (GL_PSP_GRID_DEFAULT_SOURCE)
        include_td_losses: Include transmission and distribution losses (GL_PSP_GRID_INCLUDE_TD)
        default_td_loss_pct: Default T&D loss percentage (GL_PSP_GRID_DEFAULT_TD_LOSS)

    Example:
        >>> grid = GridFactorsConfig(
        ...     default_region="US_NATIONAL",
        ...     update_frequency=24,
        ...     fallback_to_global=True,
        ...     default_source="EGRID",
        ...     include_td_losses=True,
        ...     default_td_loss_pct=Decimal("5.0")
        ... )
        >>> grid.default_source
        'EGRID'
    """

    default_region: str = "US_NATIONAL"
    update_frequency: int = 24
    fallback_to_global: bool = True
    default_source: str = "EGRID"
    include_td_losses: bool = True
    default_td_loss_pct: Decimal = Decimal("5.0")

    def validate(self) -> None:
        """
        Validate grid factors configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if not self.default_region:
            raise ValueError("default_region cannot be empty")

        if self.update_frequency < 1 or self.update_frequency > 8760:
            raise ValueError("update_frequency must be between 1 and 8760 hours")

        valid_sources = {"EGRID", "IEA", "EU_EEA", "DEFRA", "CUSTOM"}
        if self.default_source not in valid_sources:
            raise ValueError(
                f"Invalid default_source '{self.default_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.default_td_loss_pct < Decimal("0") or self.default_td_loss_pct > Decimal("50"):
            raise ValueError("default_td_loss_pct must be between 0 and 50")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_region": self.default_region,
            "update_frequency": self.update_frequency,
            "fallback_to_global": self.fallback_to_global,
            "default_source": self.default_source,
            "include_td_losses": self.include_td_losses,
            "default_td_loss_pct": str(self.default_td_loss_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridFactorsConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_td_loss_pct" in data_copy:
            data_copy["default_td_loss_pct"] = Decimal(data_copy["default_td_loss_pct"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "GridFactorsConfig":
        """Load from environment variables."""
        return cls(
            default_region=os.getenv("GL_PSP_GRID_DEFAULT_REGION", "US_NATIONAL"),
            update_frequency=int(os.getenv("GL_PSP_GRID_UPDATE_FREQ", "24")),
            fallback_to_global=os.getenv(
                "GL_PSP_GRID_FALLBACK_GLOBAL", "true"
            ).lower()
            == "true",
            default_source=os.getenv("GL_PSP_GRID_DEFAULT_SOURCE", "EGRID"),
            include_td_losses=os.getenv(
                "GL_PSP_GRID_INCLUDE_TD", "true"
            ).lower()
            == "true",
            default_td_loss_pct=Decimal(
                os.getenv("GL_PSP_GRID_DEFAULT_TD_LOSS", "5.0")
            ),
        )


# =============================================================================
# SECTION 10: FUEL FACTORS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class FuelFactorsConfig:
    """
    Fuel emission factor configuration for processing fuel consumption.

    Attributes:
        default_fuel: Default fuel type (GL_PSP_FUEL_DEFAULT)
        include_upstream: Include well-to-tank upstream emissions (GL_PSP_FUEL_INCLUDE_UPSTREAM)
        default_source: Default fuel EF source (GL_PSP_FUEL_DEFAULT_SOURCE)
        biofuel_blending_enabled: Enable biofuel blending adjustments (GL_PSP_FUEL_BIOFUEL_BLENDING)
        default_biofuel_pct: Default biofuel blend percentage (GL_PSP_FUEL_DEFAULT_BIOFUEL_PCT)

    Example:
        >>> fuel = FuelFactorsConfig(
        ...     default_fuel="NATURAL_GAS",
        ...     include_upstream=True,
        ...     default_source="DEFRA",
        ...     biofuel_blending_enabled=False,
        ...     default_biofuel_pct=Decimal("0.0")
        ... )
        >>> fuel.default_fuel
        'NATURAL_GAS'
    """

    default_fuel: str = "NATURAL_GAS"
    include_upstream: bool = True
    default_source: str = "DEFRA"
    biofuel_blending_enabled: bool = False
    default_biofuel_pct: Decimal = Decimal("0.0")

    def validate(self) -> None:
        """
        Validate fuel factors configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_fuels = {
            "NATURAL_GAS", "COAL", "DIESEL", "FUEL_OIL", "LPG",
            "PROPANE", "BIOMASS", "BIOGAS", "HYDROGEN", "CUSTOM",
        }
        if self.default_fuel not in valid_fuels:
            raise ValueError(
                f"Invalid default_fuel '{self.default_fuel}'. "
                f"Must be one of {valid_fuels}"
            )

        valid_sources = {"DEFRA", "EPA", "IPCC", "ECOINVENT", "CUSTOM"}
        if self.default_source not in valid_sources:
            raise ValueError(
                f"Invalid default_source '{self.default_source}'. "
                f"Must be one of {valid_sources}"
            )

        if self.default_biofuel_pct < Decimal("0") or self.default_biofuel_pct > Decimal("100"):
            raise ValueError("default_biofuel_pct must be between 0 and 100")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_fuel": self.default_fuel,
            "include_upstream": self.include_upstream,
            "default_source": self.default_source,
            "biofuel_blending_enabled": self.biofuel_blending_enabled,
            "default_biofuel_pct": str(self.default_biofuel_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FuelFactorsConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "default_biofuel_pct" in data_copy:
            data_copy["default_biofuel_pct"] = Decimal(data_copy["default_biofuel_pct"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "FuelFactorsConfig":
        """Load from environment variables."""
        return cls(
            default_fuel=os.getenv("GL_PSP_FUEL_DEFAULT", "NATURAL_GAS"),
            include_upstream=os.getenv(
                "GL_PSP_FUEL_INCLUDE_UPSTREAM", "true"
            ).lower()
            == "true",
            default_source=os.getenv("GL_PSP_FUEL_DEFAULT_SOURCE", "DEFRA"),
            biofuel_blending_enabled=os.getenv(
                "GL_PSP_FUEL_BIOFUEL_BLENDING", "false"
            ).lower()
            == "true",
            default_biofuel_pct=Decimal(
                os.getenv("GL_PSP_FUEL_DEFAULT_BIOFUEL_PCT", "0.0")
            ),
        )


# =============================================================================
# SECTION 11: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for Processing of Sold Products agent.

    Attributes:
        enabled_frameworks: Comma-separated enabled frameworks (GL_PSP_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict compliance mode (GL_PSP_COMPLIANCE_STRICT)
        auto_check: Automatically run compliance checks (GL_PSP_COMPLIANCE_AUTO_CHECK)
        double_counting_check: Enable double-counting prevention (GL_PSP_COMPLIANCE_DC_CHECK)
        boundary_enforcement: Enforce Scope 3 Cat 10 boundary (GL_PSP_COMPLIANCE_BOUNDARY)
        dc_check_categories: Categories to check for double-counting (GL_PSP_COMPLIANCE_DC_CATS)

    Example:
        >>> compliance = ComplianceConfig(
        ...     enabled_frameworks="GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD,CDP,SBTI,GRI,DEFRA",
        ...     strict_mode=True,
        ...     auto_check=True,
        ...     double_counting_check=True,
        ...     boundary_enforcement=True,
        ...     dc_check_categories="SCOPE1,SCOPE2,CAT1,CAT2,CAT11,CAT12"
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD', 'CDP', 'SBTI', 'GRI', 'DEFRA']
    """

    enabled_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD,CDP,SBTI,GRI,DEFRA"
    )
    strict_mode: bool = True
    auto_check: bool = True
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    dc_check_categories: str = "SCOPE1,SCOPE2,CAT1,CAT2,CAT11,CAT12"

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL_SCOPE3",
            "ISO_14064",
            "CSRD",
            "CDP",
            "SBTI",
            "GRI",
            "DEFRA",
        }
        frameworks = self.get_frameworks()
        if not frameworks:
            raise ValueError("At least one compliance framework must be enabled")

        for framework in frameworks:
            if framework not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{framework}'. Must be one of {valid_frameworks}"
                )

        valid_dc_categories = {
            "SCOPE1", "SCOPE2", "CAT1", "CAT2", "CAT3",
            "CAT4", "CAT5", "CAT11", "CAT12",
        }
        dc_cats = self.get_dc_categories()
        for cat in dc_cats:
            if cat not in valid_dc_categories:
                raise ValueError(
                    f"Invalid dc_check_category '{cat}'. Must be one of {valid_dc_categories}"
                )

    def get_frameworks(self) -> List[str]:
        """Parse enabled frameworks string into list."""
        return [f.strip() for f in self.enabled_frameworks.split(",") if f.strip()]

    def get_dc_categories(self) -> List[str]:
        """Parse double-counting categories string into list."""
        return [c.strip() for c in self.dc_check_categories.split(",") if c.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_frameworks": self.enabled_frameworks,
            "strict_mode": self.strict_mode,
            "auto_check": self.auto_check,
            "double_counting_check": self.double_counting_check,
            "boundary_enforcement": self.boundary_enforcement,
            "dc_check_categories": self.dc_check_categories,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        return cls(
            enabled_frameworks=os.getenv(
                "GL_PSP_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD,CDP,SBTI,GRI,DEFRA",
            ),
            strict_mode=os.getenv("GL_PSP_COMPLIANCE_STRICT", "true").lower() == "true",
            auto_check=os.getenv("GL_PSP_COMPLIANCE_AUTO_CHECK", "true").lower() == "true",
            double_counting_check=os.getenv(
                "GL_PSP_COMPLIANCE_DC_CHECK", "true"
            ).lower()
            == "true",
            boundary_enforcement=os.getenv(
                "GL_PSP_COMPLIANCE_BOUNDARY", "true"
            ).lower()
            == "true",
            dc_check_categories=os.getenv(
                "GL_PSP_COMPLIANCE_DC_CATS", "SCOPE1,SCOPE2,CAT1,CAT2,CAT11,CAT12"
            ),
        )


# =============================================================================
# SECTION 12: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance tracking configuration.

    Attributes:
        hash_algorithm: Hash algorithm for provenance (GL_PSP_PROV_ALGORITHM)
        chain_enabled: Enable chain hashing (GL_PSP_PROV_CHAIN_ENABLED)
        merkle_enabled: Enable Merkle tree root computation (GL_PSP_PROV_MERKLE_ENABLED)
        store_intermediate: Store intermediate hashes (GL_PSP_PROV_STORE_INTERMEDIATE)

    Example:
        >>> prov = ProvenanceConfig(
        ...     hash_algorithm="sha256",
        ...     chain_enabled=True,
        ...     merkle_enabled=True,
        ...     store_intermediate=True
        ... )
        >>> prov.hash_algorithm
        'sha256'
    """

    hash_algorithm: str = "sha256"
    chain_enabled: bool = True
    merkle_enabled: bool = True
    store_intermediate: bool = True

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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            hash_algorithm=os.getenv("GL_PSP_PROV_ALGORITHM", "sha256"),
            chain_enabled=os.getenv("GL_PSP_PROV_CHAIN_ENABLED", "true").lower() == "true",
            merkle_enabled=os.getenv("GL_PSP_PROV_MERKLE_ENABLED", "true").lower() == "true",
            store_intermediate=os.getenv(
                "GL_PSP_PROV_STORE_INTERMEDIATE", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 13: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty quantification configuration.

    Attributes:
        method: Default uncertainty method (GL_PSP_UNCERT_METHOD)
        iterations: Monte Carlo iterations (GL_PSP_UNCERT_ITERATIONS)
        confidence_level: Default confidence level (GL_PSP_UNCERT_CONFIDENCE)
        include_parameter: Include parameter uncertainty (GL_PSP_UNCERT_INCLUDE_PARAM)
        include_model: Include model uncertainty (GL_PSP_UNCERT_INCLUDE_MODEL)
        seed: Random seed for reproducibility (GL_PSP_UNCERT_SEED)

    Example:
        >>> uncert = UncertaintyConfig(
        ...     method="IPCC_DEFAULT",
        ...     iterations=10000,
        ...     confidence_level=Decimal("0.95"),
        ...     include_parameter=True,
        ...     include_model=False,
        ...     seed=42
        ... )
        >>> uncert.method
        'IPCC_DEFAULT'
    """

    method: str = "IPCC_DEFAULT"
    iterations: int = 10000
    confidence_level: Decimal = Decimal("0.95")
    include_parameter: bool = True
    include_model: bool = False
    seed: Optional[int] = None

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"IPCC_DEFAULT", "MONTE_CARLO", "BOOTSTRAP", "BAYESIAN", "NONE"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. Must be one of {valid_methods}"
            )

        if self.iterations < 100 or self.iterations > 1000000:
            raise ValueError("iterations must be between 100 and 1000000")

        if self.confidence_level < Decimal("0") or self.confidence_level > Decimal("1"):
            raise ValueError("confidence_level must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "iterations": self.iterations,
            "confidence_level": str(self.confidence_level),
            "include_parameter": self.include_parameter,
            "include_model": self.include_model,
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
        seed_str = os.getenv("GL_PSP_UNCERT_SEED")
        seed_val = int(seed_str) if seed_str else None
        return cls(
            method=os.getenv("GL_PSP_UNCERT_METHOD", "IPCC_DEFAULT"),
            iterations=int(os.getenv("GL_PSP_UNCERT_ITERATIONS", "10000")),
            confidence_level=Decimal(
                os.getenv("GL_PSP_UNCERT_CONFIDENCE", "0.95")
            ),
            include_parameter=os.getenv(
                "GL_PSP_UNCERT_INCLUDE_PARAM", "true"
            ).lower()
            == "true",
            include_model=os.getenv(
                "GL_PSP_UNCERT_INCLUDE_MODEL", "false"
            ).lower()
            == "true",
            seed=seed_val,
        )


# =============================================================================
# SECTION 14: DQI CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DQIConfig:
    """
    Data Quality Indicator configuration.

    Controls the 5-dimension DQI scoring system aligned with GHG Protocol
    guidance for Scope 3 data quality assessment.

    Attributes:
        min_acceptable_score: Minimum acceptable composite DQI score (GL_PSP_DQI_MIN_SCORE)
        weight_technological: Weight for technological representativeness (GL_PSP_DQI_W_TECH)
        weight_temporal: Weight for temporal representativeness (GL_PSP_DQI_W_TEMPORAL)
        weight_geographical: Weight for geographical representativeness (GL_PSP_DQI_W_GEO)
        weight_completeness: Weight for completeness (GL_PSP_DQI_W_COMPLETENESS)
        weight_reliability: Weight for reliability (GL_PSP_DQI_W_RELIABILITY)

    Example:
        >>> dqi = DQIConfig(
        ...     min_acceptable_score=Decimal("2.0"),
        ...     weight_technological=Decimal("0.25"),
        ...     weight_temporal=Decimal("0.20"),
        ...     weight_geographical=Decimal("0.20"),
        ...     weight_completeness=Decimal("0.20"),
        ...     weight_reliability=Decimal("0.15")
        ... )
        >>> dqi.min_acceptable_score
        Decimal('2.0')
    """

    min_acceptable_score: Decimal = Decimal("2.0")
    weight_technological: Decimal = Decimal("0.25")
    weight_temporal: Decimal = Decimal("0.20")
    weight_geographical: Decimal = Decimal("0.20")
    weight_completeness: Decimal = Decimal("0.20")
    weight_reliability: Decimal = Decimal("0.15")

    def validate(self) -> None:
        """
        Validate DQI configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.min_acceptable_score < Decimal("1.0") or self.min_acceptable_score > Decimal("5.0"):
            raise ValueError("min_acceptable_score must be between 1.0 and 5.0")

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

        total = sum(weights)
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"DQI weights must sum to 1.0 (current sum: {total})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_acceptable_score": str(self.min_acceptable_score),
            "weight_technological": str(self.weight_technological),
            "weight_temporal": str(self.weight_temporal),
            "weight_geographical": str(self.weight_geographical),
            "weight_completeness": str(self.weight_completeness),
            "weight_reliability": str(self.weight_reliability),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DQIConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in data_copy:
            data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "DQIConfig":
        """Load from environment variables."""
        return cls(
            min_acceptable_score=Decimal(
                os.getenv("GL_PSP_DQI_MIN_SCORE", "2.0")
            ),
            weight_technological=Decimal(
                os.getenv("GL_PSP_DQI_W_TECH", "0.25")
            ),
            weight_temporal=Decimal(
                os.getenv("GL_PSP_DQI_W_TEMPORAL", "0.20")
            ),
            weight_geographical=Decimal(
                os.getenv("GL_PSP_DQI_W_GEO", "0.20")
            ),
            weight_completeness=Decimal(
                os.getenv("GL_PSP_DQI_W_COMPLETENESS", "0.20")
            ),
            weight_reliability=Decimal(
                os.getenv("GL_PSP_DQI_W_RELIABILITY", "0.15")
            ),
        )


# =============================================================================
# SECTION 15: PIPELINE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline execution configuration.

    Attributes:
        batch_size: Default batch size for batch processing (GL_PSP_PIPELINE_BATCH_SIZE)
        timeout_seconds: Pipeline timeout in seconds (GL_PSP_PIPELINE_TIMEOUT)
        max_retries: Maximum pipeline retries (GL_PSP_PIPELINE_MAX_RETRIES)
        parallel_products: Max parallel product calculations (GL_PSP_PIPELINE_PARALLEL)
        worker_threads: Worker thread count (GL_PSP_PIPELINE_WORKERS)
        rate_limit: Requests per minute (GL_PSP_PIPELINE_RATE_LIMIT)

    Example:
        >>> pipeline = PipelineConfig(
        ...     batch_size=500,
        ...     timeout_seconds=300,
        ...     max_retries=3,
        ...     parallel_products=10,
        ...     worker_threads=4,
        ...     rate_limit=100
        ... )
        >>> pipeline.batch_size
        500
    """

    batch_size: int = 500
    timeout_seconds: int = 300
    max_retries: int = 3
    parallel_products: int = 10
    worker_threads: int = 4
    rate_limit: int = 100

    def validate(self) -> None:
        """
        Validate pipeline configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.batch_size < 1 or self.batch_size > 10000:
            raise ValueError("batch_size must be between 1 and 10000")

        if self.timeout_seconds < 1 or self.timeout_seconds > 3600:
            raise ValueError("timeout_seconds must be between 1 and 3600")

        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")

        if self.parallel_products < 1 or self.parallel_products > 100:
            raise ValueError("parallel_products must be between 1 and 100")

        if self.worker_threads < 1 or self.worker_threads > 64:
            raise ValueError("worker_threads must be between 1 and 64")

        if self.rate_limit < 1 or self.rate_limit > 10000:
            raise ValueError("rate_limit must be between 1 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "parallel_products": self.parallel_products,
            "worker_threads": self.worker_threads,
            "rate_limit": self.rate_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load from environment variables."""
        return cls(
            batch_size=int(os.getenv("GL_PSP_PIPELINE_BATCH_SIZE", "500")),
            timeout_seconds=int(os.getenv("GL_PSP_PIPELINE_TIMEOUT", "300")),
            max_retries=int(os.getenv("GL_PSP_PIPELINE_MAX_RETRIES", "3")),
            parallel_products=int(os.getenv("GL_PSP_PIPELINE_PARALLEL", "10")),
            worker_threads=int(os.getenv("GL_PSP_PIPELINE_WORKERS", "4")),
            rate_limit=int(os.getenv("GL_PSP_PIPELINE_RATE_LIMIT", "100")),
        )


# =============================================================================
# SECTION 16: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for Processing of Sold Products agent.

    Attributes:
        enabled: Enable caching (GL_PSP_CACHE_ENABLED)
        ttl_seconds: Cache TTL in seconds (GL_PSP_CACHE_TTL)
        max_size: Max cache entries (GL_PSP_CACHE_MAX_SIZE)
        cache_ef_lookups: Cache emission factor lookups (GL_PSP_CACHE_EF_LOOKUPS)
        cache_calculations: Cache calculation results (GL_PSP_CACHE_CALCULATIONS)
        cache_grid_factors: Cache grid factors (GL_PSP_CACHE_GRID_FACTORS)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl_seconds=3600,
        ...     max_size=10000,
        ...     cache_ef_lookups=True,
        ...     cache_calculations=True,
        ...     cache_grid_factors=True
        ... )
        >>> cache.ttl_seconds
        3600
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    cache_ef_lookups: bool = True
    cache_calculations: bool = True
    cache_grid_factors: bool = True

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
            "cache_grid_factors": self.cache_grid_factors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_PSP_CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("GL_PSP_CACHE_TTL", "3600")),
            max_size=int(os.getenv("GL_PSP_CACHE_MAX_SIZE", "10000")),
            cache_ef_lookups=os.getenv(
                "GL_PSP_CACHE_EF_LOOKUPS", "true"
            ).lower()
            == "true",
            cache_calculations=os.getenv(
                "GL_PSP_CACHE_CALCULATIONS", "true"
            ).lower()
            == "true",
            cache_grid_factors=os.getenv(
                "GL_PSP_CACHE_GRID_FACTORS", "true"
            ).lower()
            == "true",
        )


# =============================================================================
# SECTION 17: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Metrics configuration for Processing of Sold Products agent.

    Attributes:
        enabled: Enable Prometheus metrics collection (GL_PSP_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_PSP_METRICS_PREFIX)
        histogram_buckets: Histogram bucket boundaries (GL_PSP_METRICS_BUCKETS)

    Example:
        >>> metrics = MetricsConfig(
        ...     enabled=True,
        ...     prefix="gl_psp_",
        ...     histogram_buckets="0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_buckets()
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    """

    enabled: bool = True
    prefix: str = "gl_psp_"
    histogram_buckets: str = "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"

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

        try:
            buckets = self.get_buckets()
            if not buckets:
                raise ValueError("At least one bucket must be defined")
            for bucket in buckets:
                if bucket <= 0:
                    raise ValueError("All buckets must be positive")
        except Exception as e:
            raise ValueError(f"Invalid histogram_buckets format: {e}")

    def get_buckets(self) -> List[float]:
        """Parse histogram buckets string into list of floats."""
        return [float(x.strip()) for x in self.histogram_buckets.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "histogram_buckets": self.histogram_buckets,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_PSP_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_PSP_METRICS_PREFIX", "gl_psp_"),
            histogram_buckets=os.getenv(
                "GL_PSP_METRICS_BUCKETS",
                "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
            ),
        )


# =============================================================================
# SECTION 18: SECURITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SecurityConfig:
    """
    Security configuration for Processing of Sold Products agent.

    Attributes:
        tenant_isolation: Enable tenant data isolation (GL_PSP_SEC_TENANT_ISOLATION)
        audit_enabled: Enable audit logging (GL_PSP_SEC_AUDIT_ENABLED)
        encrypt_at_rest: Encrypt sensitive data at rest (GL_PSP_SEC_ENCRYPT_REST)
        max_request_size_mb: Maximum request size in MB (GL_PSP_SEC_MAX_REQUEST_MB)
        api_prefix: API route prefix (GL_PSP_SEC_API_PREFIX)

    Example:
        >>> security = SecurityConfig(
        ...     tenant_isolation=True,
        ...     audit_enabled=True,
        ...     encrypt_at_rest=True,
        ...     max_request_size_mb=50,
        ...     api_prefix="/api/v1/processing-sold-products"
        ... )
        >>> security.tenant_isolation
        True
    """

    tenant_isolation: bool = True
    audit_enabled: bool = True
    encrypt_at_rest: bool = True
    max_request_size_mb: int = 50
    api_prefix: str = "/api/v1/processing-sold-products"

    def validate(self) -> None:
        """
        Validate security configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.max_request_size_mb < 1 or self.max_request_size_mb > 500:
            raise ValueError("max_request_size_mb must be between 1 and 500")

        if not self.api_prefix:
            raise ValueError("api_prefix cannot be empty")

        if not self.api_prefix.startswith("/"):
            raise ValueError("api_prefix must start with '/'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_isolation": self.tenant_isolation,
            "audit_enabled": self.audit_enabled,
            "encrypt_at_rest": self.encrypt_at_rest,
            "max_request_size_mb": self.max_request_size_mb,
            "api_prefix": self.api_prefix,
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
                "GL_PSP_SEC_TENANT_ISOLATION", "true"
            ).lower()
            == "true",
            audit_enabled=os.getenv(
                "GL_PSP_SEC_AUDIT_ENABLED", "true"
            ).lower()
            == "true",
            encrypt_at_rest=os.getenv(
                "GL_PSP_SEC_ENCRYPT_REST", "true"
            ).lower()
            == "true",
            max_request_size_mb=int(
                os.getenv("GL_PSP_SEC_MAX_REQUEST_MB", "50")
            ),
            api_prefix=os.getenv(
                "GL_PSP_SEC_API_PREFIX", "/api/v1/processing-sold-products"
            ),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


class ProcessingSoldProductsConfig:
    """
    Master configuration class for Processing of Sold Products agent.

    This class aggregates all 18 configuration sections and provides a unified
    interface for accessing configuration values. It implements the singleton
    pattern with thread-safe access.

    Attributes:
        general: General configuration (agent_id, version, table_prefix, debug, log_level)
        database: Database configuration (connection pool, timeouts, retry)
        processing: Processing configuration (type, energy source, chain, GWP)
        site_specific: Site-specific calculation configuration
        average_data: Average-data calculation configuration
        spend_based: Spend-based calculation configuration (EEIO, CPI, margin)
        hybrid: Hybrid method waterfall configuration
        allocation: Allocation method configuration (mass/revenue/units/equal)
        grid_factors: Grid electricity factor configuration
        fuel_factors: Fuel emission factor configuration
        compliance: Compliance framework configuration
        provenance: Provenance tracking configuration
        uncertainty: Uncertainty quantification configuration
        dqi: Data Quality Indicator configuration
        pipeline: Pipeline execution configuration
        cache: Cache configuration
        metrics: Prometheus metrics configuration
        security: Security and tenant isolation configuration

    Example:
        >>> config = ProcessingSoldProductsConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-010'
        >>> config.processing.default_processing_type
        'INDUSTRIAL_MANUFACTURING'
        >>> config.validate_all()
    """

    def __init__(
        self,
        general: GeneralConfig,
        database: DatabaseConfig,
        processing: ProcessingConfig,
        site_specific: SiteSpecificConfig,
        average_data: AverageDataConfig,
        spend_based: SpendBasedConfig,
        hybrid: HybridConfig,
        allocation: AllocationConfig,
        grid_factors: GridFactorsConfig,
        fuel_factors: FuelFactorsConfig,
        compliance: ComplianceConfig,
        provenance: ProvenanceConfig,
        uncertainty: UncertaintyConfig,
        dqi: DQIConfig,
        pipeline: PipelineConfig,
        cache: CacheConfig,
        metrics: MetricsConfig,
        security: SecurityConfig,
    ):
        """
        Initialize master configuration.

        Args:
            general: General configuration
            database: Database configuration
            processing: Processing configuration
            site_specific: Site-specific configuration
            average_data: Average-data configuration
            spend_based: Spend-based configuration
            hybrid: Hybrid configuration
            allocation: Allocation configuration
            grid_factors: Grid factors configuration
            fuel_factors: Fuel factors configuration
            compliance: Compliance configuration
            provenance: Provenance configuration
            uncertainty: Uncertainty configuration
            dqi: DQI configuration
            pipeline: Pipeline configuration
            cache: Cache configuration
            metrics: Metrics configuration
            security: Security configuration
        """
        self.general = general
        self.database = database
        self.processing = processing
        self.site_specific = site_specific
        self.average_data = average_data
        self.spend_based = spend_based
        self.hybrid = hybrid
        self.allocation = allocation
        self.grid_factors = grid_factors
        self.fuel_factors = fuel_factors
        self.compliance = compliance
        self.provenance = provenance
        self.uncertainty = uncertainty
        self.dqi = dqi
        self.pipeline = pipeline
        self.cache = cache
        self.metrics = metrics
        self.security = security

    def validate_all(self) -> None:
        """
        Validate all configuration sections.

        Raises:
            ValueError: If any configuration section is invalid
        """
        self.general.validate()
        self.database.validate()
        self.processing.validate()
        self.site_specific.validate()
        self.average_data.validate()
        self.spend_based.validate()
        self.hybrid.validate()
        self.allocation.validate()
        self.grid_factors.validate()
        self.fuel_factors.validate()
        self.compliance.validate()
        self.provenance.validate()
        self.uncertainty.validate()
        self.dqi.validate()
        self.pipeline.validate()
        self.cache.validate()
        self.metrics.validate()
        self.security.validate()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire configuration to dictionary.

        Returns:
            Dictionary representation of all configuration sections
        """
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "processing": self.processing.to_dict(),
            "site_specific": self.site_specific.to_dict(),
            "average_data": self.average_data.to_dict(),
            "spend_based": self.spend_based.to_dict(),
            "hybrid": self.hybrid.to_dict(),
            "allocation": self.allocation.to_dict(),
            "grid_factors": self.grid_factors.to_dict(),
            "fuel_factors": self.fuel_factors.to_dict(),
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
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingSoldProductsConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            ProcessingSoldProductsConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data["general"]),
            database=DatabaseConfig.from_dict(data["database"]),
            processing=ProcessingConfig.from_dict(data["processing"]),
            site_specific=SiteSpecificConfig.from_dict(data["site_specific"]),
            average_data=AverageDataConfig.from_dict(data["average_data"]),
            spend_based=SpendBasedConfig.from_dict(data["spend_based"]),
            hybrid=HybridConfig.from_dict(data["hybrid"]),
            allocation=AllocationConfig.from_dict(data["allocation"]),
            grid_factors=GridFactorsConfig.from_dict(data["grid_factors"]),
            fuel_factors=FuelFactorsConfig.from_dict(data["fuel_factors"]),
            compliance=ComplianceConfig.from_dict(data["compliance"]),
            provenance=ProvenanceConfig.from_dict(data["provenance"]),
            uncertainty=UncertaintyConfig.from_dict(data["uncertainty"]),
            dqi=DQIConfig.from_dict(data["dqi"]),
            pipeline=PipelineConfig.from_dict(data["pipeline"]),
            cache=CacheConfig.from_dict(data["cache"]),
            metrics=MetricsConfig.from_dict(data["metrics"]),
            security=SecurityConfig.from_dict(data["security"]),
        )

    @classmethod
    def from_env(cls) -> "ProcessingSoldProductsConfig":
        """
        Load configuration from environment variables.

        Returns:
            ProcessingSoldProductsConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            processing=ProcessingConfig.from_env(),
            site_specific=SiteSpecificConfig.from_env(),
            average_data=AverageDataConfig.from_env(),
            spend_based=SpendBasedConfig.from_env(),
            hybrid=HybridConfig.from_env(),
            allocation=AllocationConfig.from_env(),
            grid_factors=GridFactorsConfig.from_env(),
            fuel_factors=FuelFactorsConfig.from_env(),
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


_config_instance: Optional[ProcessingSoldProductsConfig] = None
_config_lock = threading.RLock()


def get_config() -> ProcessingSoldProductsConfig:
    """
    Get the singleton configuration instance.

    This function implements thread-safe lazy initialization of the
    configuration singleton. The first call will load configuration from
    environment variables. Subsequent calls return the cached instance.

    Returns:
        ProcessingSoldProductsConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-010'

    Thread Safety:
        This function is thread-safe and can be called from multiple threads
        concurrently. The configuration is initialized only once.
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            # Double-checked locking pattern
            if _config_instance is None:
                _config_instance = ProcessingSoldProductsConfig.from_env()
                _config_instance.validate_all()

    return _config_instance


def set_config(config: ProcessingSoldProductsConfig) -> None:
    """
    Set the singleton configuration instance.

    This function allows manual configuration of the singleton instance,
    primarily useful for testing or non-standard initialization scenarios.

    Args:
        config: ProcessingSoldProductsConfig instance to set as singleton

    Example:
        >>> custom_config = ProcessingSoldProductsConfig.from_dict({...})
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


# =============================================================================
# CONFIGURATION VALIDATION UTILITIES
# =============================================================================


def validate_config(config: ProcessingSoldProductsConfig) -> List[str]:
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
    errors: List[str] = []

    sections = [
        ("general", config.general),
        ("database", config.database),
        ("processing", config.processing),
        ("site_specific", config.site_specific),
        ("average_data", config.average_data),
        ("spend_based", config.spend_based),
        ("hybrid", config.hybrid),
        ("allocation", config.allocation),
        ("grid_factors", config.grid_factors),
        ("fuel_factors", config.fuel_factors),
        ("compliance", config.compliance),
        ("provenance", config.provenance),
        ("uncertainty", config.uncertainty),
        ("dqi", config.dqi),
        ("pipeline", config.pipeline),
        ("cache", config.cache),
        ("metrics", config.metrics),
        ("security", config.security),
    ]

    for section_name, section in sections:
        try:
            section.validate()
        except ValueError as e:
            errors.append(f"{section_name}: {str(e)}")

    return errors


def print_config(config: ProcessingSoldProductsConfig) -> None:
    """
    Print configuration in human-readable format.

    This function prints all configuration sections in a formatted,
    human-readable manner. Useful for debugging and verification.

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
        ===== Processing of Sold Products Configuration =====
        [GENERAL]
        enabled: True
        debug: False
        ...
    """
    section_labels = [
        ("GENERAL", config.general),
        ("DATABASE", config.database),
        ("PROCESSING", config.processing),
        ("SITE_SPECIFIC", config.site_specific),
        ("AVERAGE_DATA", config.average_data),
        ("SPEND_BASED", config.spend_based),
        ("HYBRID", config.hybrid),
        ("ALLOCATION", config.allocation),
        ("GRID_FACTORS", config.grid_factors),
        ("FUEL_FACTORS", config.fuel_factors),
        ("COMPLIANCE", config.compliance),
        ("PROVENANCE", config.provenance),
        ("UNCERTAINTY", config.uncertainty),
        ("DQI", config.dqi),
        ("PIPELINE", config.pipeline),
        ("CACHE", config.cache),
        ("METRICS", config.metrics),
        ("SECURITY", config.security),
    ]

    print("===== Processing of Sold Products Configuration =====")

    for label, section in section_labels:
        print(f"\n[{label}]")
        for key, value in section.to_dict().items():
            if key in ("database_url",):
                print(f"{key}: [REDACTED]")
            else:
                print(f"{key}: {value}")

    print("\n" + "=" * 64)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration section classes
    "GeneralConfig",
    "DatabaseConfig",
    "ProcessingConfig",
    "SiteSpecificConfig",
    "AverageDataConfig",
    "SpendBasedConfig",
    "HybridConfig",
    "AllocationConfig",
    "GridFactorsConfig",
    "FuelFactorsConfig",
    "ComplianceConfig",
    "ProvenanceConfig",
    "UncertaintyConfig",
    "DQIConfig",
    "PipelineConfig",
    "CacheConfig",
    "MetricsConfig",
    "SecurityConfig",
    # Master configuration class
    "ProcessingSoldProductsConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
]
