# -*- coding: utf-8 -*-
"""
Investments Configuration - AGENT-MRV-028

Thread-safe singleton configuration for GL-MRV-S3-015.
All environment variables prefixed with GL_INV_.

This module provides comprehensive configuration management for the
Investments agent (GHG Protocol Scope 3 Category 15), supporting:
- Equity investments (listed/unlisted, EVIC-based attribution)
- Debt investments (corporate bonds, loans, securitized debt)
- Project finance (construction/operational phase, lifetime allocation)
- Commercial real estate (ENERGY STAR, GRESB benchmarks, EUI-based)
- Mortgages (LTV-based attribution, property-level estimation)
- Motor vehicle loans (distance-based, fleet composition)
- Sovereign bonds (GDP-based attribution, UNFCCC/EDGAR sources)
- PCAF data quality scoring (1-5 scale across all asset classes)
- 9 regulatory frameworks (GHG Protocol, PCAF, ISO 14064, CSRD, CDP,
  SBTi, GRI, SEC Climate, EU Taxonomy)
- Uncertainty quantification (Monte Carlo, confidence intervals)
- Portfolio analytics (WACI, financed emissions intensity, alignment)

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-015'
    >>> config.equity.default_scope_inclusion
    'SCOPE_1_2'
    >>> config.pcaf.default_data_quality_score
    4

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_INV_ prefix. See individual config sections for specific variables.
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
    General configuration for Investments agent.

    Attributes:
        enabled: Master switch for the agent (GL_INV_ENABLED)
        debug: Enable debug mode with verbose logging (GL_INV_DEBUG)
        log_level: Logging level (GL_INV_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_INV_AGENT_ID)
        agent_component: Agent component identifier (GL_INV_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_INV_VERSION)
        api_prefix: API route prefix (GL_INV_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_INV_MAX_BATCH_SIZE)
        default_gwp: Default GWP assessment report version (GL_INV_DEFAULT_GWP)
        default_reporting_year: Default reporting year (GL_INV_DEFAULT_REPORTING_YEAR)
        default_currency: Default currency ISO 4217 (GL_INV_DEFAULT_CURRENCY)

    Example:
        >>> general = GeneralConfig()
        >>> general.agent_id
        'GL-MRV-S3-015'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-015"
    agent_component: str = "AGENT-MRV-028"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/investments"
    max_batch_size: int = 5000
    default_gwp: str = "AR5"
    default_reporting_year: int = 2025
    default_currency: str = "USD"

    def validate(self) -> None:
        """Validate general configuration values."""
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
        valid_gwp = {"AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. Must be one of {valid_gwp}"
            )
        if self.default_reporting_year < 2000 or self.default_reporting_year > 2100:
            raise ValueError("default_reporting_year must be between 2000 and 2100")
        if len(self.default_currency) != 3:
            raise ValueError("default_currency must be a 3-letter ISO 4217 code")

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
            "default_reporting_year": self.default_reporting_year,
            "default_currency": self.default_currency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_INV_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_INV_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_INV_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_INV_AGENT_ID", "GL-MRV-S3-015"),
            agent_component=os.getenv("GL_INV_AGENT_COMPONENT", "AGENT-MRV-028"),
            version=os.getenv("GL_INV_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_INV_API_PREFIX", "/api/v1/investments"),
            max_batch_size=int(os.getenv("GL_INV_MAX_BATCH_SIZE", "5000")),
            default_gwp=os.getenv("GL_INV_DEFAULT_GWP", "AR5"),
            default_reporting_year=int(os.getenv("GL_INV_DEFAULT_REPORTING_YEAR", "2025")),
            default_currency=os.getenv("GL_INV_DEFAULT_CURRENCY", "USD"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Investments agent.

    Attributes:
        host: PostgreSQL host (GL_INV_DB_HOST)
        port: PostgreSQL port (GL_INV_DB_PORT)
        database: Database name (GL_INV_DB_DATABASE)
        username: Database username (GL_INV_DB_USERNAME)
        password: Database password (GL_INV_DB_PASSWORD)
        schema: Database schema name (GL_INV_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_INV_DB_TABLE_PREFIX)
        pool_min: Minimum connection pool size (GL_INV_DB_POOL_MIN)
        pool_max: Maximum connection pool size (GL_INV_DB_POOL_MAX)
        ssl_mode: SSL connection mode (GL_INV_DB_SSL_MODE)
        connection_timeout: Connection timeout in seconds (GL_INV_DB_CONNECTION_TIMEOUT)

    Example:
        >>> db = DatabaseConfig()
        >>> db.table_prefix
        'gl_inv_'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "investments_service"
    table_prefix: str = "gl_inv_"
    pool_min: int = 2
    pool_max: int = 10
    ssl_mode: str = "prefer"
    connection_timeout: int = 30

    def validate(self) -> None:
        """Validate database configuration values."""
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
        valid_ssl = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. Must be one of {valid_ssl}"
            )
        if self.connection_timeout < 1 or self.connection_timeout > 300:
            raise ValueError("connection_timeout must be between 1 and 300 seconds")

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
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_INV_DB_HOST", "localhost"),
            port=int(os.getenv("GL_INV_DB_PORT", "5432")),
            database=os.getenv("GL_INV_DB_DATABASE", "greenlang"),
            username=os.getenv("GL_INV_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_INV_DB_PASSWORD", ""),
            schema=os.getenv("GL_INV_DB_SCHEMA", "investments_service"),
            table_prefix=os.getenv("GL_INV_DB_TABLE_PREFIX", "gl_inv_"),
            pool_min=int(os.getenv("GL_INV_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_INV_DB_POOL_MAX", "10")),
            ssl_mode=os.getenv("GL_INV_DB_SSL_MODE", "prefer"),
            connection_timeout=int(os.getenv("GL_INV_DB_CONNECTION_TIMEOUT", "30")),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Investments agent.

    Attributes:
        host: Redis host (GL_INV_REDIS_HOST)
        port: Redis port (GL_INV_REDIS_PORT)
        db: Redis database index (GL_INV_REDIS_DB)
        password: Redis password (GL_INV_REDIS_PASSWORD)
        ssl: Enable SSL connection (GL_INV_REDIS_SSL)
        prefix: Key prefix for namespacing (GL_INV_REDIS_PREFIX)
        max_connections: Max connections in pool (GL_INV_REDIS_MAX_CONNECTIONS)
        socket_timeout: Socket timeout in seconds (GL_INV_REDIS_SOCKET_TIMEOUT)
        ef_ttl: Emission factor cache TTL in seconds (GL_INV_REDIS_EF_TTL)
        result_ttl: Calculation result cache TTL (GL_INV_REDIS_RESULT_TTL)

    Example:
        >>> redis = RedisConfig()
        >>> redis.prefix
        'gl_inv:'
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    prefix: str = "gl_inv:"
    max_connections: int = 20
    socket_timeout: int = 5
    ef_ttl: int = 86400
    result_ttl: int = 3600

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
        if self.max_connections < 1 or self.max_connections > 1000:
            raise ValueError("max_connections must be between 1 and 1000")
        if self.socket_timeout < 1 or self.socket_timeout > 60:
            raise ValueError("socket_timeout must be between 1 and 60 seconds")
        if self.ef_ttl < 0:
            raise ValueError("ef_ttl must be >= 0")
        if self.result_ttl < 0:
            raise ValueError("result_ttl must be >= 0")

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
            "ef_ttl": self.ef_ttl,
            "result_ttl": self.result_ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_INV_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_INV_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_INV_REDIS_DB", "0")),
            password=os.getenv("GL_INV_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_INV_REDIS_SSL", "false").lower() == "true",
            prefix=os.getenv("GL_INV_REDIS_PREFIX", "gl_inv:"),
            max_connections=int(os.getenv("GL_INV_REDIS_MAX_CONNECTIONS", "20")),
            socket_timeout=int(os.getenv("GL_INV_REDIS_SOCKET_TIMEOUT", "5")),
            ef_ttl=int(os.getenv("GL_INV_REDIS_EF_TTL", "86400")),
            result_ttl=int(os.getenv("GL_INV_REDIS_RESULT_TTL", "3600")),
        )


# =============================================================================
# SECTION 4: EQUITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EquityConfig:
    """
    Equity investment calculation configuration.

    PCAF asset class: Listed equity and unlisted equity.
    Attribution factor = Outstanding amount / EVIC (or market cap).

    Attributes:
        default_scope_inclusion: Scope coverage (GL_INV_EQUITY_SCOPE_INCLUSION)
        evic_source_priority: Ordered EVIC data sources (GL_INV_EQUITY_EVIC_SOURCE_PRIORITY)
        market_cap_sources: Market cap data sources (GL_INV_EQUITY_MARKET_CAP_SOURCES)
        use_revenue_fallback: Fall back to revenue if EVIC unavailable (GL_INV_EQUITY_USE_REVENUE_FALLBACK)
        listed_pcaf_score_default: Default PCAF score for listed equity (GL_INV_EQUITY_LISTED_PCAF_DEFAULT)
        unlisted_pcaf_score_default: Default PCAF score for unlisted equity (GL_INV_EQUITY_UNLISTED_PCAF_DEFAULT)
        attribution_cap: Maximum attribution factor allowed (GL_INV_EQUITY_ATTRIBUTION_CAP)
        include_scope_3: Include Scope 3 in financed emissions (GL_INV_EQUITY_INCLUDE_SCOPE_3)

    Example:
        >>> equity = EquityConfig()
        >>> equity.default_scope_inclusion
        'SCOPE_1_2'
    """

    default_scope_inclusion: str = "SCOPE_1_2"
    evic_source_priority: Tuple[str, ...] = ("bloomberg", "refinitiv", "msci", "cdp", "manual")
    market_cap_sources: Tuple[str, ...] = ("bloomberg", "refinitiv", "yahoo_finance")
    use_revenue_fallback: bool = True
    listed_pcaf_score_default: int = 3
    unlisted_pcaf_score_default: int = 4
    attribution_cap: Decimal = Decimal("1.0")
    include_scope_3: bool = False

    def validate(self) -> None:
        """Validate equity configuration values."""
        valid_scopes = {"SCOPE_1", "SCOPE_1_2", "SCOPE_1_2_3"}
        if self.default_scope_inclusion not in valid_scopes:
            raise ValueError(
                f"Invalid default_scope_inclusion '{self.default_scope_inclusion}'. "
                f"Must be one of {valid_scopes}"
            )
        if not self.evic_source_priority:
            raise ValueError("evic_source_priority cannot be empty")
        if self.listed_pcaf_score_default < 1 or self.listed_pcaf_score_default > 5:
            raise ValueError("listed_pcaf_score_default must be between 1 and 5")
        if self.unlisted_pcaf_score_default < 1 or self.unlisted_pcaf_score_default > 5:
            raise ValueError("unlisted_pcaf_score_default must be between 1 and 5")
        if self.attribution_cap <= Decimal("0") or self.attribution_cap > Decimal("1.0"):
            raise ValueError("attribution_cap must be > 0 and <= 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_scope_inclusion": self.default_scope_inclusion,
            "evic_source_priority": list(self.evic_source_priority),
            "market_cap_sources": list(self.market_cap_sources),
            "use_revenue_fallback": self.use_revenue_fallback,
            "listed_pcaf_score_default": self.listed_pcaf_score_default,
            "unlisted_pcaf_score_default": self.unlisted_pcaf_score_default,
            "attribution_cap": str(self.attribution_cap),
            "include_scope_3": self.include_scope_3,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquityConfig":
        """Create from dictionary."""
        d = data.copy()
        if "evic_source_priority" in d and isinstance(d["evic_source_priority"], list):
            d["evic_source_priority"] = tuple(d["evic_source_priority"])
        if "market_cap_sources" in d and isinstance(d["market_cap_sources"], list):
            d["market_cap_sources"] = tuple(d["market_cap_sources"])
        if "attribution_cap" in d:
            d["attribution_cap"] = Decimal(str(d["attribution_cap"]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "EquityConfig":
        """Load from environment variables."""
        evic_raw = os.getenv(
            "GL_INV_EQUITY_EVIC_SOURCE_PRIORITY",
            "bloomberg,refinitiv,msci,cdp,manual",
        )
        mktcap_raw = os.getenv(
            "GL_INV_EQUITY_MARKET_CAP_SOURCES",
            "bloomberg,refinitiv,yahoo_finance",
        )
        return cls(
            default_scope_inclusion=os.getenv("GL_INV_EQUITY_SCOPE_INCLUSION", "SCOPE_1_2"),
            evic_source_priority=tuple(s.strip() for s in evic_raw.split(",")),
            market_cap_sources=tuple(s.strip() for s in mktcap_raw.split(",")),
            use_revenue_fallback=os.getenv(
                "GL_INV_EQUITY_USE_REVENUE_FALLBACK", "true"
            ).lower() == "true",
            listed_pcaf_score_default=int(
                os.getenv("GL_INV_EQUITY_LISTED_PCAF_DEFAULT", "3")
            ),
            unlisted_pcaf_score_default=int(
                os.getenv("GL_INV_EQUITY_UNLISTED_PCAF_DEFAULT", "4")
            ),
            attribution_cap=Decimal(os.getenv("GL_INV_EQUITY_ATTRIBUTION_CAP", "1.0")),
            include_scope_3=os.getenv(
                "GL_INV_EQUITY_INCLUDE_SCOPE_3", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 5: DEBT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DebtConfig:
    """
    Debt investment calculation configuration.

    PCAF asset class: Corporate bonds and loans.
    Attribution factor = Outstanding amount / (Total equity + Total debt).

    Attributes:
        default_attribution_denominator: Denominator type (GL_INV_DEBT_ATTRIBUTION_DENOM)
        include_revolving_credit: Include revolving facilities (GL_INV_DEBT_INCLUDE_REVOLVING)
        maturity_handling: How to handle maturity (GL_INV_DEBT_MATURITY_HANDLING)
        default_recovery_rate: Default recovery rate for defaulted debt (GL_INV_DEBT_RECOVERY_RATE)
        bond_pcaf_score_default: Default PCAF score for bonds (GL_INV_DEBT_BOND_PCAF_DEFAULT)
        loan_pcaf_score_default: Default PCAF score for loans (GL_INV_DEBT_LOAN_PCAF_DEFAULT)
        securitized_pcaf_score_default: Default PCAF for securitized (GL_INV_DEBT_SECURITIZED_PCAF_DEFAULT)
        use_committed_amount: Use committed vs drawn amount (GL_INV_DEBT_USE_COMMITTED)

    Example:
        >>> debt = DebtConfig()
        >>> debt.default_attribution_denominator
        'EVIC'
    """

    default_attribution_denominator: str = "EVIC"
    include_revolving_credit: bool = True
    maturity_handling: str = "OUTSTANDING_BALANCE"
    default_recovery_rate: Decimal = Decimal("0.40")
    bond_pcaf_score_default: int = 3
    loan_pcaf_score_default: int = 3
    securitized_pcaf_score_default: int = 4
    use_committed_amount: bool = False

    def validate(self) -> None:
        """Validate debt configuration values."""
        valid_denoms = {"EVIC", "TOTAL_ASSETS", "EQUITY_PLUS_DEBT", "REVENUE"}
        if self.default_attribution_denominator not in valid_denoms:
            raise ValueError(
                f"Invalid default_attribution_denominator "
                f"'{self.default_attribution_denominator}'. Must be one of {valid_denoms}"
            )
        valid_maturity = {"OUTSTANDING_BALANCE", "COMMITTED_AMOUNT", "FACE_VALUE"}
        if self.maturity_handling not in valid_maturity:
            raise ValueError(
                f"Invalid maturity_handling '{self.maturity_handling}'. "
                f"Must be one of {valid_maturity}"
            )
        if self.default_recovery_rate < Decimal("0") or self.default_recovery_rate > Decimal("1"):
            raise ValueError("default_recovery_rate must be between 0 and 1")
        for name, val in [
            ("bond_pcaf_score_default", self.bond_pcaf_score_default),
            ("loan_pcaf_score_default", self.loan_pcaf_score_default),
            ("securitized_pcaf_score_default", self.securitized_pcaf_score_default),
        ]:
            if val < 1 or val > 5:
                raise ValueError(f"{name} must be between 1 and 5")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_attribution_denominator": self.default_attribution_denominator,
            "include_revolving_credit": self.include_revolving_credit,
            "maturity_handling": self.maturity_handling,
            "default_recovery_rate": str(self.default_recovery_rate),
            "bond_pcaf_score_default": self.bond_pcaf_score_default,
            "loan_pcaf_score_default": self.loan_pcaf_score_default,
            "securitized_pcaf_score_default": self.securitized_pcaf_score_default,
            "use_committed_amount": self.use_committed_amount,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebtConfig":
        """Create from dictionary."""
        d = data.copy()
        if "default_recovery_rate" in d:
            d["default_recovery_rate"] = Decimal(str(d["default_recovery_rate"]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "DebtConfig":
        """Load from environment variables."""
        return cls(
            default_attribution_denominator=os.getenv("GL_INV_DEBT_ATTRIBUTION_DENOM", "EVIC"),
            include_revolving_credit=os.getenv(
                "GL_INV_DEBT_INCLUDE_REVOLVING", "true"
            ).lower() == "true",
            maturity_handling=os.getenv("GL_INV_DEBT_MATURITY_HANDLING", "OUTSTANDING_BALANCE"),
            default_recovery_rate=Decimal(
                os.getenv("GL_INV_DEBT_RECOVERY_RATE", "0.40")
            ),
            bond_pcaf_score_default=int(os.getenv("GL_INV_DEBT_BOND_PCAF_DEFAULT", "3")),
            loan_pcaf_score_default=int(os.getenv("GL_INV_DEBT_LOAN_PCAF_DEFAULT", "3")),
            securitized_pcaf_score_default=int(
                os.getenv("GL_INV_DEBT_SECURITIZED_PCAF_DEFAULT", "4")
            ),
            use_committed_amount=os.getenv(
                "GL_INV_DEBT_USE_COMMITTED", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 6: PROJECT FINANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProjectFinanceConfig:
    """
    Project finance calculation configuration.

    Covers infrastructure, energy, and industrial project investments.
    Attribution based on proportional share of total project financing.

    Attributes:
        default_project_lifetime_years: Default project lifetime (GL_INV_PF_LIFETIME_YEARS)
        construction_phase_multiplier: Multiplier for construction emissions (GL_INV_PF_CONSTRUCTION_MULT)
        operational_phase_method: Method for operational phase (GL_INV_PF_OPERATIONAL_METHOD)
        include_decommissioning: Include decommissioning emissions (GL_INV_PF_INCLUDE_DECOMMISSION)
        pcaf_score_default: Default PCAF score (GL_INV_PF_PCAF_DEFAULT)
        allocation_method: How to allocate to investor (GL_INV_PF_ALLOCATION_METHOD)
        annualize_lifetime_emissions: Annualize over lifetime (GL_INV_PF_ANNUALIZE)
        default_capacity_factor: Default capacity factor for energy projects (GL_INV_PF_CAPACITY_FACTOR)

    Example:
        >>> pf = ProjectFinanceConfig()
        >>> pf.default_project_lifetime_years
        25
    """

    default_project_lifetime_years: int = 25
    construction_phase_multiplier: Decimal = Decimal("1.0")
    operational_phase_method: str = "ANNUAL_ACTUAL"
    include_decommissioning: bool = False
    pcaf_score_default: int = 4
    allocation_method: str = "PROPORTIONAL_SHARE"
    annualize_lifetime_emissions: bool = True
    default_capacity_factor: Decimal = Decimal("0.30")

    def validate(self) -> None:
        """Validate project finance configuration values."""
        if self.default_project_lifetime_years < 1 or self.default_project_lifetime_years > 100:
            raise ValueError("default_project_lifetime_years must be between 1 and 100")
        if self.construction_phase_multiplier <= Decimal("0"):
            raise ValueError("construction_phase_multiplier must be > 0")
        valid_methods = {"ANNUAL_ACTUAL", "LIFETIME_AMORTIZED", "CAPACITY_BASED"}
        if self.operational_phase_method not in valid_methods:
            raise ValueError(
                f"Invalid operational_phase_method '{self.operational_phase_method}'. "
                f"Must be one of {valid_methods}"
            )
        if self.pcaf_score_default < 1 or self.pcaf_score_default > 5:
            raise ValueError("pcaf_score_default must be between 1 and 5")
        valid_alloc = {"PROPORTIONAL_SHARE", "EQUITY_SHARE", "COMMITMENT_SHARE"}
        if self.allocation_method not in valid_alloc:
            raise ValueError(
                f"Invalid allocation_method '{self.allocation_method}'. "
                f"Must be one of {valid_alloc}"
            )
        if self.default_capacity_factor <= Decimal("0") or self.default_capacity_factor > Decimal("1"):
            raise ValueError("default_capacity_factor must be > 0 and <= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_project_lifetime_years": self.default_project_lifetime_years,
            "construction_phase_multiplier": str(self.construction_phase_multiplier),
            "operational_phase_method": self.operational_phase_method,
            "include_decommissioning": self.include_decommissioning,
            "pcaf_score_default": self.pcaf_score_default,
            "allocation_method": self.allocation_method,
            "annualize_lifetime_emissions": self.annualize_lifetime_emissions,
            "default_capacity_factor": str(self.default_capacity_factor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectFinanceConfig":
        """Create from dictionary."""
        d = data.copy()
        for k in ("construction_phase_multiplier", "default_capacity_factor"):
            if k in d:
                d[k] = Decimal(str(d[k]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ProjectFinanceConfig":
        """Load from environment variables."""
        return cls(
            default_project_lifetime_years=int(
                os.getenv("GL_INV_PF_LIFETIME_YEARS", "25")
            ),
            construction_phase_multiplier=Decimal(
                os.getenv("GL_INV_PF_CONSTRUCTION_MULT", "1.0")
            ),
            operational_phase_method=os.getenv("GL_INV_PF_OPERATIONAL_METHOD", "ANNUAL_ACTUAL"),
            include_decommissioning=os.getenv(
                "GL_INV_PF_INCLUDE_DECOMMISSION", "false"
            ).lower() == "true",
            pcaf_score_default=int(os.getenv("GL_INV_PF_PCAF_DEFAULT", "4")),
            allocation_method=os.getenv("GL_INV_PF_ALLOCATION_METHOD", "PROPORTIONAL_SHARE"),
            annualize_lifetime_emissions=os.getenv(
                "GL_INV_PF_ANNUALIZE", "true"
            ).lower() == "true",
            default_capacity_factor=Decimal(
                os.getenv("GL_INV_PF_CAPACITY_FACTOR", "0.30")
            ),
        )


# =============================================================================
# SECTION 7: COMMERCIAL REAL ESTATE (CRE) CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CREConfig:
    """
    Commercial real estate investment configuration.

    PCAF asset class: Commercial real estate.
    Attribution based on property value or floor area.

    Attributes:
        benchmark_sources: Building benchmark data sources (GL_INV_CRE_BENCHMARK_SOURCES)
        default_eui_kwh_per_m2: Default energy use intensity (GL_INV_CRE_DEFAULT_EUI)
        default_building_type: Default building type (GL_INV_CRE_DEFAULT_BUILDING_TYPE)
        energy_star_threshold: ENERGY STAR eligibility score (GL_INV_CRE_ENERGY_STAR_THRESHOLD)
        gresb_integration: Enable GRESB score integration (GL_INV_CRE_GRESB_INTEGRATION)
        pcaf_score_default: Default PCAF score (GL_INV_CRE_PCAF_DEFAULT)
        attribution_method: Attribution method (GL_INV_CRE_ATTRIBUTION_METHOD)
        include_scope_3_tenant: Include tenant Scope 3 (GL_INV_CRE_INCLUDE_SCOPE_3_TENANT)

    Example:
        >>> cre = CREConfig()
        >>> cre.default_eui_kwh_per_m2
        Decimal('200.0')
    """

    benchmark_sources: Tuple[str, ...] = ("energy_star", "gresb", "crrem", "leed")
    default_eui_kwh_per_m2: Decimal = Decimal("200.0")
    default_building_type: str = "OFFICE"
    energy_star_threshold: int = 75
    gresb_integration: bool = True
    pcaf_score_default: int = 4
    attribution_method: str = "PROPERTY_VALUE"
    include_scope_3_tenant: bool = False

    def validate(self) -> None:
        """Validate CRE configuration values."""
        if not self.benchmark_sources:
            raise ValueError("benchmark_sources cannot be empty")
        if self.default_eui_kwh_per_m2 <= Decimal("0"):
            raise ValueError("default_eui_kwh_per_m2 must be > 0")
        valid_types = {
            "OFFICE", "RETAIL", "INDUSTRIAL", "WAREHOUSE", "HOTEL",
            "HOSPITAL", "RESIDENTIAL_MULTI", "MIXED_USE", "DATA_CENTER",
        }
        if self.default_building_type not in valid_types:
            raise ValueError(
                f"Invalid default_building_type '{self.default_building_type}'. "
                f"Must be one of {valid_types}"
            )
        if self.energy_star_threshold < 1 or self.energy_star_threshold > 100:
            raise ValueError("energy_star_threshold must be between 1 and 100")
        if self.pcaf_score_default < 1 or self.pcaf_score_default > 5:
            raise ValueError("pcaf_score_default must be between 1 and 5")
        valid_attr = {"PROPERTY_VALUE", "FLOOR_AREA", "LOAN_VALUE"}
        if self.attribution_method not in valid_attr:
            raise ValueError(
                f"Invalid attribution_method '{self.attribution_method}'. "
                f"Must be one of {valid_attr}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_sources": list(self.benchmark_sources),
            "default_eui_kwh_per_m2": str(self.default_eui_kwh_per_m2),
            "default_building_type": self.default_building_type,
            "energy_star_threshold": self.energy_star_threshold,
            "gresb_integration": self.gresb_integration,
            "pcaf_score_default": self.pcaf_score_default,
            "attribution_method": self.attribution_method,
            "include_scope_3_tenant": self.include_scope_3_tenant,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CREConfig":
        """Create from dictionary."""
        d = data.copy()
        if "benchmark_sources" in d and isinstance(d["benchmark_sources"], list):
            d["benchmark_sources"] = tuple(d["benchmark_sources"])
        if "default_eui_kwh_per_m2" in d:
            d["default_eui_kwh_per_m2"] = Decimal(str(d["default_eui_kwh_per_m2"]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "CREConfig":
        """Load from environment variables."""
        bench_raw = os.getenv("GL_INV_CRE_BENCHMARK_SOURCES", "energy_star,gresb,crrem,leed")
        return cls(
            benchmark_sources=tuple(s.strip() for s in bench_raw.split(",")),
            default_eui_kwh_per_m2=Decimal(
                os.getenv("GL_INV_CRE_DEFAULT_EUI", "200.0")
            ),
            default_building_type=os.getenv("GL_INV_CRE_DEFAULT_BUILDING_TYPE", "OFFICE"),
            energy_star_threshold=int(
                os.getenv("GL_INV_CRE_ENERGY_STAR_THRESHOLD", "75")
            ),
            gresb_integration=os.getenv(
                "GL_INV_CRE_GRESB_INTEGRATION", "true"
            ).lower() == "true",
            pcaf_score_default=int(os.getenv("GL_INV_CRE_PCAF_DEFAULT", "4")),
            attribution_method=os.getenv("GL_INV_CRE_ATTRIBUTION_METHOD", "PROPERTY_VALUE"),
            include_scope_3_tenant=os.getenv(
                "GL_INV_CRE_INCLUDE_SCOPE_3_TENANT", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 8: MORTGAGE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MortgageConfig:
    """
    Mortgage investment configuration.

    PCAF asset class: Mortgages (residential and commercial).
    Attribution based on LTV ratio and property-level emissions.

    Attributes:
        default_ltv_ratio: Default loan-to-value ratio (GL_INV_MORT_DEFAULT_LTV)
        ltv_cap: Maximum LTV for attribution (GL_INV_MORT_LTV_CAP)
        default_property_eui: Default property EUI kWh/m2 (GL_INV_MORT_DEFAULT_EUI)
        default_property_area_m2: Default floor area (GL_INV_MORT_DEFAULT_AREA)
        include_embodied_carbon: Include embodied carbon (GL_INV_MORT_INCLUDE_EMBODIED)
        pcaf_score_default: Default PCAF score (GL_INV_MORT_PCAF_DEFAULT)
        epc_rating_source: EPC rating data source (GL_INV_MORT_EPC_SOURCE)
        default_grid_factor_kgco2e_kwh: Default grid EF (GL_INV_MORT_GRID_FACTOR)

    Example:
        >>> mort = MortgageConfig()
        >>> mort.default_ltv_ratio
        Decimal('0.80')
    """

    default_ltv_ratio: Decimal = Decimal("0.80")
    ltv_cap: Decimal = Decimal("1.0")
    default_property_eui: Decimal = Decimal("150.0")
    default_property_area_m2: Decimal = Decimal("120.0")
    include_embodied_carbon: bool = False
    pcaf_score_default: int = 4
    epc_rating_source: str = "NATIONAL_REGISTRY"
    default_grid_factor_kgco2e_kwh: Decimal = Decimal("0.40")

    def validate(self) -> None:
        """Validate mortgage configuration values."""
        if self.default_ltv_ratio <= Decimal("0") or self.default_ltv_ratio > Decimal("2.0"):
            raise ValueError("default_ltv_ratio must be > 0 and <= 2.0")
        if self.ltv_cap <= Decimal("0") or self.ltv_cap > Decimal("2.0"):
            raise ValueError("ltv_cap must be > 0 and <= 2.0")
        if self.default_property_eui <= Decimal("0"):
            raise ValueError("default_property_eui must be > 0")
        if self.default_property_area_m2 <= Decimal("0"):
            raise ValueError("default_property_area_m2 must be > 0")
        if self.pcaf_score_default < 1 or self.pcaf_score_default > 5:
            raise ValueError("pcaf_score_default must be between 1 and 5")
        valid_sources = {"NATIONAL_REGISTRY", "CRREM", "ESTIMATED", "MANUAL"}
        if self.epc_rating_source not in valid_sources:
            raise ValueError(
                f"Invalid epc_rating_source '{self.epc_rating_source}'. "
                f"Must be one of {valid_sources}"
            )
        if self.default_grid_factor_kgco2e_kwh <= Decimal("0"):
            raise ValueError("default_grid_factor_kgco2e_kwh must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_ltv_ratio": str(self.default_ltv_ratio),
            "ltv_cap": str(self.ltv_cap),
            "default_property_eui": str(self.default_property_eui),
            "default_property_area_m2": str(self.default_property_area_m2),
            "include_embodied_carbon": self.include_embodied_carbon,
            "pcaf_score_default": self.pcaf_score_default,
            "epc_rating_source": self.epc_rating_source,
            "default_grid_factor_kgco2e_kwh": str(self.default_grid_factor_kgco2e_kwh),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MortgageConfig":
        """Create from dictionary."""
        d = data.copy()
        for k in (
            "default_ltv_ratio", "ltv_cap", "default_property_eui",
            "default_property_area_m2", "default_grid_factor_kgco2e_kwh",
        ):
            if k in d:
                d[k] = Decimal(str(d[k]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "MortgageConfig":
        """Load from environment variables."""
        return cls(
            default_ltv_ratio=Decimal(os.getenv("GL_INV_MORT_DEFAULT_LTV", "0.80")),
            ltv_cap=Decimal(os.getenv("GL_INV_MORT_LTV_CAP", "1.0")),
            default_property_eui=Decimal(os.getenv("GL_INV_MORT_DEFAULT_EUI", "150.0")),
            default_property_area_m2=Decimal(os.getenv("GL_INV_MORT_DEFAULT_AREA", "120.0")),
            include_embodied_carbon=os.getenv(
                "GL_INV_MORT_INCLUDE_EMBODIED", "false"
            ).lower() == "true",
            pcaf_score_default=int(os.getenv("GL_INV_MORT_PCAF_DEFAULT", "4")),
            epc_rating_source=os.getenv("GL_INV_MORT_EPC_SOURCE", "NATIONAL_REGISTRY"),
            default_grid_factor_kgco2e_kwh=Decimal(
                os.getenv("GL_INV_MORT_GRID_FACTOR", "0.40")
            ),
        )


# =============================================================================
# SECTION 9: MOTOR VEHICLE LOAN CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MotorVehicleConfig:
    """
    Motor vehicle loan configuration.

    PCAF asset class: Motor vehicle loans.
    Attribution based on loan outstanding vs vehicle value.

    Attributes:
        default_annual_distance_km: Default annual distance (GL_INV_MV_ANNUAL_DISTANCE)
        default_vehicle_lifetime_years: Default vehicle lifetime (GL_INV_MV_LIFETIME_YEARS)
        default_fuel_type: Default fuel type (GL_INV_MV_DEFAULT_FUEL)
        default_vehicle_class: Default vehicle class (GL_INV_MV_DEFAULT_CLASS)
        pcaf_score_default: Default PCAF score (GL_INV_MV_PCAF_DEFAULT)
        include_upstream_fuel: Include WTT fuel emissions (GL_INV_MV_INCLUDE_UPSTREAM)
        ev_grid_factor_source: Grid EF source for EVs (GL_INV_MV_EV_GRID_SOURCE)
        default_ef_gco2e_per_km: Default tailpipe EF g/km (GL_INV_MV_DEFAULT_EF)

    Example:
        >>> mv = MotorVehicleConfig()
        >>> mv.default_annual_distance_km
        15000
    """

    default_annual_distance_km: int = 15000
    default_vehicle_lifetime_years: int = 12
    default_fuel_type: str = "GASOLINE"
    default_vehicle_class: str = "MEDIUM_CAR"
    pcaf_score_default: int = 3
    include_upstream_fuel: bool = False
    ev_grid_factor_source: str = "IEA"
    default_ef_gco2e_per_km: Decimal = Decimal("170.0")

    def validate(self) -> None:
        """Validate motor vehicle configuration values."""
        if self.default_annual_distance_km < 100 or self.default_annual_distance_km > 200000:
            raise ValueError("default_annual_distance_km must be between 100 and 200000")
        if self.default_vehicle_lifetime_years < 1 or self.default_vehicle_lifetime_years > 50:
            raise ValueError("default_vehicle_lifetime_years must be between 1 and 50")
        valid_fuels = {
            "GASOLINE", "DIESEL", "HYBRID", "PLUGIN_HYBRID", "ELECTRIC",
            "CNG", "LPG", "HYDROGEN",
        }
        if self.default_fuel_type not in valid_fuels:
            raise ValueError(
                f"Invalid default_fuel_type '{self.default_fuel_type}'. "
                f"Must be one of {valid_fuels}"
            )
        valid_classes = {
            "SMALL_CAR", "MEDIUM_CAR", "LARGE_CAR", "SUV", "LIGHT_TRUCK",
            "HEAVY_TRUCK", "MOTORCYCLE", "VAN",
        }
        if self.default_vehicle_class not in valid_classes:
            raise ValueError(
                f"Invalid default_vehicle_class '{self.default_vehicle_class}'. "
                f"Must be one of {valid_classes}"
            )
        if self.pcaf_score_default < 1 or self.pcaf_score_default > 5:
            raise ValueError("pcaf_score_default must be between 1 and 5")
        valid_ev_sources = {"IEA", "EGRID", "NATIONAL_GRID", "CUSTOM"}
        if self.ev_grid_factor_source not in valid_ev_sources:
            raise ValueError(
                f"Invalid ev_grid_factor_source '{self.ev_grid_factor_source}'. "
                f"Must be one of {valid_ev_sources}"
            )
        if self.default_ef_gco2e_per_km < Decimal("0"):
            raise ValueError("default_ef_gco2e_per_km must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_annual_distance_km": self.default_annual_distance_km,
            "default_vehicle_lifetime_years": self.default_vehicle_lifetime_years,
            "default_fuel_type": self.default_fuel_type,
            "default_vehicle_class": self.default_vehicle_class,
            "pcaf_score_default": self.pcaf_score_default,
            "include_upstream_fuel": self.include_upstream_fuel,
            "ev_grid_factor_source": self.ev_grid_factor_source,
            "default_ef_gco2e_per_km": str(self.default_ef_gco2e_per_km),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotorVehicleConfig":
        """Create from dictionary."""
        d = data.copy()
        if "default_ef_gco2e_per_km" in d:
            d["default_ef_gco2e_per_km"] = Decimal(str(d["default_ef_gco2e_per_km"]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "MotorVehicleConfig":
        """Load from environment variables."""
        return cls(
            default_annual_distance_km=int(
                os.getenv("GL_INV_MV_ANNUAL_DISTANCE", "15000")
            ),
            default_vehicle_lifetime_years=int(
                os.getenv("GL_INV_MV_LIFETIME_YEARS", "12")
            ),
            default_fuel_type=os.getenv("GL_INV_MV_DEFAULT_FUEL", "GASOLINE"),
            default_vehicle_class=os.getenv("GL_INV_MV_DEFAULT_CLASS", "MEDIUM_CAR"),
            pcaf_score_default=int(os.getenv("GL_INV_MV_PCAF_DEFAULT", "3")),
            include_upstream_fuel=os.getenv(
                "GL_INV_MV_INCLUDE_UPSTREAM", "false"
            ).lower() == "true",
            ev_grid_factor_source=os.getenv("GL_INV_MV_EV_GRID_SOURCE", "IEA"),
            default_ef_gco2e_per_km=Decimal(
                os.getenv("GL_INV_MV_DEFAULT_EF", "170.0")
            ),
        )


# =============================================================================
# SECTION 10: SOVEREIGN BOND CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SovereignConfig:
    """
    Sovereign bond investment configuration.

    PCAF asset class: Sovereign bonds.
    Attribution factor = Investment amount / PPP-adjusted GDP.

    Attributes:
        gdp_source: GDP data source (GL_INV_SOV_GDP_SOURCE)
        emission_source: National emissions data source (GL_INV_SOV_EMISSION_SOURCE)
        use_ppp_adjusted_gdp: Use PPP-adjusted GDP (GL_INV_SOV_USE_PPP)
        scope_coverage: Emission scope to include (GL_INV_SOV_SCOPE_COVERAGE)
        include_lulucf: Include LULUCF emissions (GL_INV_SOV_INCLUDE_LULUCF)
        pcaf_score_default: Default PCAF score (GL_INV_SOV_PCAF_DEFAULT)
        default_attribution_method: Attribution method (GL_INV_SOV_ATTRIBUTION_METHOD)
        gdp_lag_years: Years of GDP data lag (GL_INV_SOV_GDP_LAG_YEARS)

    Example:
        >>> sov = SovereignConfig()
        >>> sov.gdp_source
        'IMF'
    """

    gdp_source: str = "IMF"
    emission_source: str = "UNFCCC"
    use_ppp_adjusted_gdp: bool = True
    scope_coverage: str = "PRODUCTION_BASED"
    include_lulucf: bool = False
    pcaf_score_default: int = 2
    default_attribution_method: str = "GDP_BASED"
    gdp_lag_years: int = 2

    def validate(self) -> None:
        """Validate sovereign bond configuration values."""
        valid_gdp = {"IMF", "WORLD_BANK", "OECD", "NATIONAL"}
        if self.gdp_source not in valid_gdp:
            raise ValueError(
                f"Invalid gdp_source '{self.gdp_source}'. Must be one of {valid_gdp}"
            )
        valid_emission = {"UNFCCC", "EDGAR", "PRIMAP", "CAIT", "NATIONAL"}
        if self.emission_source not in valid_emission:
            raise ValueError(
                f"Invalid emission_source '{self.emission_source}'. "
                f"Must be one of {valid_emission}"
            )
        valid_scope = {"PRODUCTION_BASED", "CONSUMPTION_BASED"}
        if self.scope_coverage not in valid_scope:
            raise ValueError(
                f"Invalid scope_coverage '{self.scope_coverage}'. "
                f"Must be one of {valid_scope}"
            )
        if self.pcaf_score_default < 1 or self.pcaf_score_default > 5:
            raise ValueError("pcaf_score_default must be between 1 and 5")
        valid_attr = {"GDP_BASED", "REVENUE_BASED", "DEBT_SHARE"}
        if self.default_attribution_method not in valid_attr:
            raise ValueError(
                f"Invalid default_attribution_method '{self.default_attribution_method}'. "
                f"Must be one of {valid_attr}"
            )
        if self.gdp_lag_years < 0 or self.gdp_lag_years > 5:
            raise ValueError("gdp_lag_years must be between 0 and 5")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gdp_source": self.gdp_source,
            "emission_source": self.emission_source,
            "use_ppp_adjusted_gdp": self.use_ppp_adjusted_gdp,
            "scope_coverage": self.scope_coverage,
            "include_lulucf": self.include_lulucf,
            "pcaf_score_default": self.pcaf_score_default,
            "default_attribution_method": self.default_attribution_method,
            "gdp_lag_years": self.gdp_lag_years,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SovereignConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "SovereignConfig":
        """Load from environment variables."""
        return cls(
            gdp_source=os.getenv("GL_INV_SOV_GDP_SOURCE", "IMF"),
            emission_source=os.getenv("GL_INV_SOV_EMISSION_SOURCE", "UNFCCC"),
            use_ppp_adjusted_gdp=os.getenv(
                "GL_INV_SOV_USE_PPP", "true"
            ).lower() == "true",
            scope_coverage=os.getenv("GL_INV_SOV_SCOPE_COVERAGE", "PRODUCTION_BASED"),
            include_lulucf=os.getenv(
                "GL_INV_SOV_INCLUDE_LULUCF", "false"
            ).lower() == "true",
            pcaf_score_default=int(os.getenv("GL_INV_SOV_PCAF_DEFAULT", "2")),
            default_attribution_method=os.getenv(
                "GL_INV_SOV_ATTRIBUTION_METHOD", "GDP_BASED"
            ),
            gdp_lag_years=int(os.getenv("GL_INV_SOV_GDP_LAG_YEARS", "2")),
        )


# =============================================================================
# SECTION 11: PCAF DATA QUALITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PCAFConfig:
    """
    PCAF data quality scoring configuration.

    Implements PCAF Global GHG Accounting and Reporting Standard
    data quality scoring framework (1 = highest, 5 = lowest).

    Attributes:
        default_data_quality_score: Default fallback score (GL_INV_PCAF_DEFAULT_DQ)
        quality_improvement_target: Annual improvement target (GL_INV_PCAF_IMPROVEMENT_TARGET)
        weight_emission_data: Weight for emission data quality (GL_INV_PCAF_WEIGHT_EMISSION)
        weight_financial_data: Weight for financial data quality (GL_INV_PCAF_WEIGHT_FINANCIAL)
        weight_allocation: Weight for allocation quality (GL_INV_PCAF_WEIGHT_ALLOCATION)
        minimum_score_threshold: Min acceptable score (GL_INV_PCAF_MIN_THRESHOLD)
        enable_score_override: Allow manual override (GL_INV_PCAF_ENABLE_OVERRIDE)
        score_version: PCAF standard version (GL_INV_PCAF_SCORE_VERSION)

    Example:
        >>> pcaf = PCAFConfig()
        >>> pcaf.default_data_quality_score
        4
    """

    default_data_quality_score: int = 4
    quality_improvement_target: Decimal = Decimal("0.5")
    weight_emission_data: Decimal = Decimal("0.40")
    weight_financial_data: Decimal = Decimal("0.30")
    weight_allocation: Decimal = Decimal("0.30")
    minimum_score_threshold: int = 5
    enable_score_override: bool = False
    score_version: str = "PCAF_2022"

    def validate(self) -> None:
        """Validate PCAF configuration values."""
        if self.default_data_quality_score < 1 or self.default_data_quality_score > 5:
            raise ValueError("default_data_quality_score must be between 1 and 5")
        if self.quality_improvement_target < Decimal("0") or self.quality_improvement_target > Decimal("5"):
            raise ValueError("quality_improvement_target must be between 0 and 5")
        weights_sum = self.weight_emission_data + self.weight_financial_data + self.weight_allocation
        if abs(weights_sum - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"PCAF weights must sum to 1.0, got {weights_sum}"
            )
        if self.minimum_score_threshold < 1 or self.minimum_score_threshold > 5:
            raise ValueError("minimum_score_threshold must be between 1 and 5")
        valid_versions = {"PCAF_2020", "PCAF_2022"}
        if self.score_version not in valid_versions:
            raise ValueError(
                f"Invalid score_version '{self.score_version}'. "
                f"Must be one of {valid_versions}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_data_quality_score": self.default_data_quality_score,
            "quality_improvement_target": str(self.quality_improvement_target),
            "weight_emission_data": str(self.weight_emission_data),
            "weight_financial_data": str(self.weight_financial_data),
            "weight_allocation": str(self.weight_allocation),
            "minimum_score_threshold": self.minimum_score_threshold,
            "enable_score_override": self.enable_score_override,
            "score_version": self.score_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PCAFConfig":
        """Create from dictionary."""
        d = data.copy()
        for k in (
            "quality_improvement_target", "weight_emission_data",
            "weight_financial_data", "weight_allocation",
        ):
            if k in d:
                d[k] = Decimal(str(d[k]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "PCAFConfig":
        """Load from environment variables."""
        return cls(
            default_data_quality_score=int(os.getenv("GL_INV_PCAF_DEFAULT_DQ", "4")),
            quality_improvement_target=Decimal(
                os.getenv("GL_INV_PCAF_IMPROVEMENT_TARGET", "0.5")
            ),
            weight_emission_data=Decimal(
                os.getenv("GL_INV_PCAF_WEIGHT_EMISSION", "0.40")
            ),
            weight_financial_data=Decimal(
                os.getenv("GL_INV_PCAF_WEIGHT_FINANCIAL", "0.30")
            ),
            weight_allocation=Decimal(
                os.getenv("GL_INV_PCAF_WEIGHT_ALLOCATION", "0.30")
            ),
            minimum_score_threshold=int(os.getenv("GL_INV_PCAF_MIN_THRESHOLD", "5")),
            enable_score_override=os.getenv(
                "GL_INV_PCAF_ENABLE_OVERRIDE", "false"
            ).lower() == "true",
            score_version=os.getenv("GL_INV_PCAF_SCORE_VERSION", "PCAF_2022"),
        )


# =============================================================================
# SECTION 12: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Regulatory compliance configuration.

    Supports 9 frameworks relevant to Scope 3 Category 15 investments.

    Attributes:
        enabled_frameworks: Frameworks to validate (GL_INV_COMPLIANCE_FRAMEWORKS)
        strictness_level: Validation strictness (GL_INV_COMPLIANCE_STRICTNESS)
        require_pcaf_disclosure: Require PCAF data quality disclosure (GL_INV_COMPLIANCE_REQUIRE_PCAF)
        require_scope_3_disclosure: Require investee Scope 3 (GL_INV_COMPLIANCE_REQUIRE_SCOPE3)
        fail_on_warning: Treat warnings as failures (GL_INV_COMPLIANCE_FAIL_ON_WARNING)
        max_acceptable_pcaf_score: Max PCAF score for compliance (GL_INV_COMPLIANCE_MAX_PCAF)
        enable_eu_taxonomy: Enable EU Taxonomy alignment (GL_INV_COMPLIANCE_EU_TAXONOMY)
        double_counting_check: Enable double-counting detection (GL_INV_COMPLIANCE_DOUBLE_COUNT)

    Example:
        >>> comp = ComplianceConfig()
        >>> 'GHG_PROTOCOL' in comp.enabled_frameworks
        True
    """

    enabled_frameworks: Tuple[str, ...] = (
        "GHG_PROTOCOL", "PCAF", "ISO_14064", "CSRD", "CDP",
        "SBTI", "GRI", "SEC_CLIMATE", "EU_TAXONOMY",
    )
    strictness_level: str = "STANDARD"
    require_pcaf_disclosure: bool = True
    require_scope_3_disclosure: bool = False
    fail_on_warning: bool = False
    max_acceptable_pcaf_score: int = 4
    enable_eu_taxonomy: bool = True
    double_counting_check: bool = True

    def validate(self) -> None:
        """Validate compliance configuration values."""
        valid_frameworks = {
            "GHG_PROTOCOL", "PCAF", "ISO_14064", "CSRD", "CDP",
            "SBTI", "GRI", "SEC_CLIMATE", "EU_TAXONOMY",
        }
        for fw in self.enabled_frameworks:
            if fw not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{fw}'. Must be one of {valid_frameworks}"
                )
        valid_strictness = {"LENIENT", "STANDARD", "STRICT"}
        if self.strictness_level not in valid_strictness:
            raise ValueError(
                f"Invalid strictness_level '{self.strictness_level}'. "
                f"Must be one of {valid_strictness}"
            )
        if self.max_acceptable_pcaf_score < 1 or self.max_acceptable_pcaf_score > 5:
            raise ValueError("max_acceptable_pcaf_score must be between 1 and 5")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled_frameworks": list(self.enabled_frameworks),
            "strictness_level": self.strictness_level,
            "require_pcaf_disclosure": self.require_pcaf_disclosure,
            "require_scope_3_disclosure": self.require_scope_3_disclosure,
            "fail_on_warning": self.fail_on_warning,
            "max_acceptable_pcaf_score": self.max_acceptable_pcaf_score,
            "enable_eu_taxonomy": self.enable_eu_taxonomy,
            "double_counting_check": self.double_counting_check,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        d = data.copy()
        if "enabled_frameworks" in d and isinstance(d["enabled_frameworks"], list):
            d["enabled_frameworks"] = tuple(d["enabled_frameworks"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        fw_raw = os.getenv(
            "GL_INV_COMPLIANCE_FRAMEWORKS",
            "GHG_PROTOCOL,PCAF,ISO_14064,CSRD,CDP,SBTI,GRI,SEC_CLIMATE,EU_TAXONOMY",
        )
        return cls(
            enabled_frameworks=tuple(s.strip() for s in fw_raw.split(",")),
            strictness_level=os.getenv("GL_INV_COMPLIANCE_STRICTNESS", "STANDARD"),
            require_pcaf_disclosure=os.getenv(
                "GL_INV_COMPLIANCE_REQUIRE_PCAF", "true"
            ).lower() == "true",
            require_scope_3_disclosure=os.getenv(
                "GL_INV_COMPLIANCE_REQUIRE_SCOPE3", "false"
            ).lower() == "true",
            fail_on_warning=os.getenv(
                "GL_INV_COMPLIANCE_FAIL_ON_WARNING", "false"
            ).lower() == "true",
            max_acceptable_pcaf_score=int(
                os.getenv("GL_INV_COMPLIANCE_MAX_PCAF", "4")
            ),
            enable_eu_taxonomy=os.getenv(
                "GL_INV_COMPLIANCE_EU_TAXONOMY", "true"
            ).lower() == "true",
            double_counting_check=os.getenv(
                "GL_INV_COMPLIANCE_DOUBLE_COUNT", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 13: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Data provenance tracking configuration.

    Attributes:
        hash_algorithm: Hash algorithm for provenance (GL_INV_PROV_HASH_ALGO)
        chain_validation_enabled: Validate chain on seal (GL_INV_PROV_CHAIN_VALIDATION)
        store_intermediate_hashes: Store per-stage hashes (GL_INV_PROV_STORE_INTERMEDIATE)
        max_chain_length: Maximum entries per chain (GL_INV_PROV_MAX_CHAIN_LENGTH)

    Example:
        >>> prov = ProvenanceConfig()
        >>> prov.hash_algorithm
        'sha256'
    """

    hash_algorithm: str = "sha256"
    chain_validation_enabled: bool = True
    store_intermediate_hashes: bool = True
    max_chain_length: int = 100

    def validate(self) -> None:
        """Validate provenance configuration values."""
        valid_algos = {"sha256", "sha384", "sha512"}
        if self.hash_algorithm not in valid_algos:
            raise ValueError(
                f"Invalid hash_algorithm '{self.hash_algorithm}'. "
                f"Must be one of {valid_algos}"
            )
        if self.max_chain_length < 1 or self.max_chain_length > 10000:
            raise ValueError("max_chain_length must be between 1 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hash_algorithm": self.hash_algorithm,
            "chain_validation_enabled": self.chain_validation_enabled,
            "store_intermediate_hashes": self.store_intermediate_hashes,
            "max_chain_length": self.max_chain_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            hash_algorithm=os.getenv("GL_INV_PROV_HASH_ALGO", "sha256"),
            chain_validation_enabled=os.getenv(
                "GL_INV_PROV_CHAIN_VALIDATION", "true"
            ).lower() == "true",
            store_intermediate_hashes=os.getenv(
                "GL_INV_PROV_STORE_INTERMEDIATE", "true"
            ).lower() == "true",
            max_chain_length=int(os.getenv("GL_INV_PROV_MAX_CHAIN_LENGTH", "100")),
        )


# =============================================================================
# SECTION 14: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty quantification configuration.

    Attributes:
        monte_carlo_iterations: Number of MC iterations (GL_INV_UNC_MC_ITERATIONS)
        confidence_level: Confidence interval level (GL_INV_UNC_CONFIDENCE)
        enable_sensitivity_analysis: Enable sensitivity analysis (GL_INV_UNC_SENSITIVITY)
        parameter_uncertainty_sources: Sources of parameter uncertainty (GL_INV_UNC_SOURCES)
        seed: Random seed for reproducibility (GL_INV_UNC_SEED)

    Example:
        >>> unc = UncertaintyConfig()
        >>> unc.monte_carlo_iterations
        10000
    """

    monte_carlo_iterations: int = 10000
    confidence_level: Decimal = Decimal("0.95")
    enable_sensitivity_analysis: bool = True
    parameter_uncertainty_sources: Tuple[str, ...] = (
        "emission_factor", "attribution_factor", "financial_data", "activity_data",
    )
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate uncertainty configuration values."""
        if self.monte_carlo_iterations < 100 or self.monte_carlo_iterations > 1000000:
            raise ValueError("monte_carlo_iterations must be between 100 and 1000000")
        if self.confidence_level <= Decimal("0") or self.confidence_level >= Decimal("1"):
            raise ValueError("confidence_level must be between 0 and 1 (exclusive)")
        if not self.parameter_uncertainty_sources:
            raise ValueError("parameter_uncertainty_sources cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "monte_carlo_iterations": self.monte_carlo_iterations,
            "confidence_level": str(self.confidence_level),
            "enable_sensitivity_analysis": self.enable_sensitivity_analysis,
            "parameter_uncertainty_sources": list(self.parameter_uncertainty_sources),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyConfig":
        """Create from dictionary."""
        d = data.copy()
        if "confidence_level" in d:
            d["confidence_level"] = Decimal(str(d["confidence_level"]))
        if "parameter_uncertainty_sources" in d and isinstance(
            d["parameter_uncertainty_sources"], list
        ):
            d["parameter_uncertainty_sources"] = tuple(d["parameter_uncertainty_sources"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "UncertaintyConfig":
        """Load from environment variables."""
        src_raw = os.getenv(
            "GL_INV_UNC_SOURCES",
            "emission_factor,attribution_factor,financial_data,activity_data",
        )
        seed_raw = os.getenv("GL_INV_UNC_SEED", "")
        return cls(
            monte_carlo_iterations=int(os.getenv("GL_INV_UNC_MC_ITERATIONS", "10000")),
            confidence_level=Decimal(os.getenv("GL_INV_UNC_CONFIDENCE", "0.95")),
            enable_sensitivity_analysis=os.getenv(
                "GL_INV_UNC_SENSITIVITY", "true"
            ).lower() == "true",
            parameter_uncertainty_sources=tuple(s.strip() for s in src_raw.split(",")),
            seed=int(seed_raw) if seed_raw else None,
        )


# =============================================================================
# SECTION 15: DATA QUALITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DataQualityConfig:
    """
    Data quality assessment configuration.

    Implements PCAF data quality scoring with weighted dimensions.

    Attributes:
        pcaf_score_weights: Per-asset-class weight config (GL_INV_DQ_PCAF_WEIGHTS)
        dqi_emission_weight: DQI weight for emission accuracy (GL_INV_DQ_EMISSION_WEIGHT)
        dqi_temporal_weight: DQI weight for temporal relevance (GL_INV_DQ_TEMPORAL_WEIGHT)
        dqi_geographic_weight: DQI weight for geographic scope (GL_INV_DQ_GEOGRAPHIC_WEIGHT)
        dqi_completeness_weight: DQI weight for data completeness (GL_INV_DQ_COMPLETENESS_WEIGHT)
        dqi_reliability_weight: DQI weight for data reliability (GL_INV_DQ_RELIABILITY_WEIGHT)
        enable_auto_scoring: Enable automatic quality scoring (GL_INV_DQ_AUTO_SCORING)

    Example:
        >>> dq = DataQualityConfig()
        >>> dq.dqi_emission_weight
        Decimal('0.30')
    """

    dqi_emission_weight: Decimal = Decimal("0.30")
    dqi_temporal_weight: Decimal = Decimal("0.20")
    dqi_geographic_weight: Decimal = Decimal("0.15")
    dqi_completeness_weight: Decimal = Decimal("0.20")
    dqi_reliability_weight: Decimal = Decimal("0.15")
    enable_auto_scoring: bool = True
    minimum_coverage_pct: Decimal = Decimal("0.80")

    def validate(self) -> None:
        """Validate data quality configuration values."""
        weights_sum = (
            self.dqi_emission_weight + self.dqi_temporal_weight
            + self.dqi_geographic_weight + self.dqi_completeness_weight
            + self.dqi_reliability_weight
        )
        if abs(weights_sum - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(f"DQI weights must sum to 1.0, got {weights_sum}")
        if self.minimum_coverage_pct < Decimal("0") or self.minimum_coverage_pct > Decimal("1"):
            raise ValueError("minimum_coverage_pct must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dqi_emission_weight": str(self.dqi_emission_weight),
            "dqi_temporal_weight": str(self.dqi_temporal_weight),
            "dqi_geographic_weight": str(self.dqi_geographic_weight),
            "dqi_completeness_weight": str(self.dqi_completeness_weight),
            "dqi_reliability_weight": str(self.dqi_reliability_weight),
            "enable_auto_scoring": self.enable_auto_scoring,
            "minimum_coverage_pct": str(self.minimum_coverage_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataQualityConfig":
        """Create from dictionary."""
        d = data.copy()
        for k in (
            "dqi_emission_weight", "dqi_temporal_weight", "dqi_geographic_weight",
            "dqi_completeness_weight", "dqi_reliability_weight", "minimum_coverage_pct",
        ):
            if k in d:
                d[k] = Decimal(str(d[k]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "DataQualityConfig":
        """Load from environment variables."""
        return cls(
            dqi_emission_weight=Decimal(os.getenv("GL_INV_DQ_EMISSION_WEIGHT", "0.30")),
            dqi_temporal_weight=Decimal(os.getenv("GL_INV_DQ_TEMPORAL_WEIGHT", "0.20")),
            dqi_geographic_weight=Decimal(os.getenv("GL_INV_DQ_GEOGRAPHIC_WEIGHT", "0.15")),
            dqi_completeness_weight=Decimal(os.getenv("GL_INV_DQ_COMPLETENESS_WEIGHT", "0.20")),
            dqi_reliability_weight=Decimal(os.getenv("GL_INV_DQ_RELIABILITY_WEIGHT", "0.15")),
            enable_auto_scoring=os.getenv(
                "GL_INV_DQ_AUTO_SCORING", "true"
            ).lower() == "true",
            minimum_coverage_pct=Decimal(
                os.getenv("GL_INV_DQ_MINIMUM_COVERAGE", "0.80")
            ),
        )


# =============================================================================
# SECTION 16: PIPELINE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline orchestration configuration.

    Attributes:
        stage_timeout_seconds: Per-stage timeout (GL_INV_PIPE_STAGE_TIMEOUT)
        batch_size: Default batch processing size (GL_INV_PIPE_BATCH_SIZE)
        max_concurrency: Max parallel calculations (GL_INV_PIPE_MAX_CONCURRENCY)
        retry_max_attempts: Max retry attempts (GL_INV_PIPE_RETRY_MAX)
        retry_backoff_factor: Exponential backoff factor (GL_INV_PIPE_RETRY_BACKOFF)
        enable_checkpointing: Enable pipeline checkpoints (GL_INV_PIPE_CHECKPOINTING)
        checkpoint_interval: Checkpoint every N records (GL_INV_PIPE_CHECKPOINT_INTERVAL)

    Example:
        >>> pipe = PipelineConfig()
        >>> pipe.batch_size
        1000
    """

    stage_timeout_seconds: int = 300
    batch_size: int = 1000
    max_concurrency: int = 4
    retry_max_attempts: int = 3
    retry_backoff_factor: Decimal = Decimal("2.0")
    enable_checkpointing: bool = True
    checkpoint_interval: int = 500

    def validate(self) -> None:
        """Validate pipeline configuration values."""
        if self.stage_timeout_seconds < 1 or self.stage_timeout_seconds > 3600:
            raise ValueError("stage_timeout_seconds must be between 1 and 3600")
        if self.batch_size < 1 or self.batch_size > 100000:
            raise ValueError("batch_size must be between 1 and 100000")
        if self.max_concurrency < 1 or self.max_concurrency > 64:
            raise ValueError("max_concurrency must be between 1 and 64")
        if self.retry_max_attempts < 0 or self.retry_max_attempts > 10:
            raise ValueError("retry_max_attempts must be between 0 and 10")
        if self.retry_backoff_factor <= Decimal("0"):
            raise ValueError("retry_backoff_factor must be > 0")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_timeout_seconds": self.stage_timeout_seconds,
            "batch_size": self.batch_size,
            "max_concurrency": self.max_concurrency,
            "retry_max_attempts": self.retry_max_attempts,
            "retry_backoff_factor": str(self.retry_backoff_factor),
            "enable_checkpointing": self.enable_checkpointing,
            "checkpoint_interval": self.checkpoint_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        d = data.copy()
        if "retry_backoff_factor" in d:
            d["retry_backoff_factor"] = Decimal(str(d["retry_backoff_factor"]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load from environment variables."""
        return cls(
            stage_timeout_seconds=int(os.getenv("GL_INV_PIPE_STAGE_TIMEOUT", "300")),
            batch_size=int(os.getenv("GL_INV_PIPE_BATCH_SIZE", "1000")),
            max_concurrency=int(os.getenv("GL_INV_PIPE_MAX_CONCURRENCY", "4")),
            retry_max_attempts=int(os.getenv("GL_INV_PIPE_RETRY_MAX", "3")),
            retry_backoff_factor=Decimal(os.getenv("GL_INV_PIPE_RETRY_BACKOFF", "2.0")),
            enable_checkpointing=os.getenv(
                "GL_INV_PIPE_CHECKPOINTING", "true"
            ).lower() == "true",
            checkpoint_interval=int(os.getenv("GL_INV_PIPE_CHECKPOINT_INTERVAL", "500")),
        )


# =============================================================================
# SECTION 17: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Prometheus metrics configuration.

    Attributes:
        namespace: Prometheus metric namespace (GL_INV_METRICS_NAMESPACE)
        enabled: Enable metrics collection (GL_INV_METRICS_ENABLED)
        collection_interval_seconds: Collection interval (GL_INV_METRICS_INTERVAL)
        histogram_buckets: Duration histogram buckets (GL_INV_METRICS_BUCKETS)

    Example:
        >>> met = MetricsConfig()
        >>> met.namespace
        'gl_inv'
    """

    namespace: str = "gl_inv"
    enabled: bool = True
    collection_interval_seconds: int = 15
    histogram_buckets: Tuple[float, ...] = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    )

    def validate(self) -> None:
        """Validate metrics configuration values."""
        if not self.namespace:
            raise ValueError("namespace cannot be empty")
        if self.collection_interval_seconds < 1 or self.collection_interval_seconds > 300:
            raise ValueError("collection_interval_seconds must be between 1 and 300")
        if not self.histogram_buckets:
            raise ValueError("histogram_buckets cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "namespace": self.namespace,
            "enabled": self.enabled,
            "collection_interval_seconds": self.collection_interval_seconds,
            "histogram_buckets": list(self.histogram_buckets),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        d = data.copy()
        if "histogram_buckets" in d and isinstance(d["histogram_buckets"], list):
            d["histogram_buckets"] = tuple(d["histogram_buckets"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        buckets_raw = os.getenv(
            "GL_INV_METRICS_BUCKETS",
            "0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0",
        )
        return cls(
            namespace=os.getenv("GL_INV_METRICS_NAMESPACE", "gl_inv"),
            enabled=os.getenv("GL_INV_METRICS_ENABLED", "true").lower() == "true",
            collection_interval_seconds=int(
                os.getenv("GL_INV_METRICS_INTERVAL", "15")
            ),
            histogram_buckets=tuple(float(b.strip()) for b in buckets_raw.split(",")),
        )


# =============================================================================
# SECTION 18: API CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class APIConfig:
    """
    REST API configuration.

    Attributes:
        rate_limit_per_minute: Max requests per minute (GL_INV_API_RATE_LIMIT)
        max_page_size: Maximum pagination page size (GL_INV_API_MAX_PAGE_SIZE)
        default_page_size: Default pagination page size (GL_INV_API_DEFAULT_PAGE_SIZE)
        max_batch_size: Max positions per batch request (GL_INV_API_MAX_BATCH_SIZE)
        enable_streaming: Enable streaming responses (GL_INV_API_STREAMING)
        request_timeout_seconds: Request timeout (GL_INV_API_REQUEST_TIMEOUT)

    Example:
        >>> api = APIConfig()
        >>> api.max_batch_size
        50000
    """

    rate_limit_per_minute: int = 120
    max_page_size: int = 1000
    default_page_size: int = 100
    max_batch_size: int = 50000
    enable_streaming: bool = False
    request_timeout_seconds: int = 120

    def validate(self) -> None:
        """Validate API configuration values."""
        if self.rate_limit_per_minute < 1 or self.rate_limit_per_minute > 10000:
            raise ValueError("rate_limit_per_minute must be between 1 and 10000")
        if self.max_page_size < 1 or self.max_page_size > 10000:
            raise ValueError("max_page_size must be between 1 and 10000")
        if self.default_page_size < 1 or self.default_page_size > self.max_page_size:
            raise ValueError(
                f"default_page_size must be between 1 and {self.max_page_size}"
            )
        if self.max_batch_size < 1 or self.max_batch_size > 500000:
            raise ValueError("max_batch_size must be between 1 and 500000")
        if self.request_timeout_seconds < 1 or self.request_timeout_seconds > 600:
            raise ValueError("request_timeout_seconds must be between 1 and 600")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "max_page_size": self.max_page_size,
            "default_page_size": self.default_page_size,
            "max_batch_size": self.max_batch_size,
            "enable_streaming": self.enable_streaming,
            "request_timeout_seconds": self.request_timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load from environment variables."""
        return cls(
            rate_limit_per_minute=int(os.getenv("GL_INV_API_RATE_LIMIT", "120")),
            max_page_size=int(os.getenv("GL_INV_API_MAX_PAGE_SIZE", "1000")),
            default_page_size=int(os.getenv("GL_INV_API_DEFAULT_PAGE_SIZE", "100")),
            max_batch_size=int(os.getenv("GL_INV_API_MAX_BATCH_SIZE", "50000")),
            enable_streaming=os.getenv(
                "GL_INV_API_STREAMING", "false"
            ).lower() == "true",
            request_timeout_seconds=int(
                os.getenv("GL_INV_API_REQUEST_TIMEOUT", "120")
            ),
        )


# =============================================================================
# SECTION 19: PORTFOLIO CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PortfolioConfig:
    """
    Portfolio-level analytics configuration.

    Attributes:
        max_positions: Maximum positions per portfolio (GL_INV_PORT_MAX_POSITIONS)
        waci_calculation_enabled: Enable WACI metric (GL_INV_PORT_WACI_ENABLED)
        alignment_target_degrees: Temperature alignment target (GL_INV_PORT_ALIGNMENT_TARGET)
        sector_classification: Sector taxonomy (GL_INV_PORT_SECTOR_CLASSIFICATION)
        benchmark_index: Default benchmark (GL_INV_PORT_BENCHMARK)
        enable_paris_alignment: Enable Paris alignment check (GL_INV_PORT_PARIS_ALIGNMENT)
        carbon_budget_methodology: Carbon budget method (GL_INV_PORT_CARBON_BUDGET_METHOD)

    Example:
        >>> port = PortfolioConfig()
        >>> port.max_positions
        100000
    """

    max_positions: int = 100000
    waci_calculation_enabled: bool = True
    alignment_target_degrees: Decimal = Decimal("1.5")
    sector_classification: str = "GICS"
    benchmark_index: str = "MSCI_WORLD"
    enable_paris_alignment: bool = True
    carbon_budget_methodology: str = "SDA"

    def validate(self) -> None:
        """Validate portfolio configuration values."""
        if self.max_positions < 1 or self.max_positions > 1000000:
            raise ValueError("max_positions must be between 1 and 1000000")
        if self.alignment_target_degrees < Decimal("1.0") or self.alignment_target_degrees > Decimal("4.0"):
            raise ValueError("alignment_target_degrees must be between 1.0 and 4.0")
        valid_sectors = {"GICS", "ICB", "NACE", "SIC", "NAICS"}
        if self.sector_classification not in valid_sectors:
            raise ValueError(
                f"Invalid sector_classification '{self.sector_classification}'. "
                f"Must be one of {valid_sectors}"
            )
        valid_benchmarks = {
            "MSCI_WORLD", "MSCI_ACWI", "SP500", "STOXX600",
            "FTSE100", "CUSTOM", "NONE",
        }
        if self.benchmark_index not in valid_benchmarks:
            raise ValueError(
                f"Invalid benchmark_index '{self.benchmark_index}'. "
                f"Must be one of {valid_benchmarks}"
            )
        valid_carbon = {"SDA", "GEVA", "ABSOLUTE_CONTRACTION", "RPS"}
        if self.carbon_budget_methodology not in valid_carbon:
            raise ValueError(
                f"Invalid carbon_budget_methodology '{self.carbon_budget_methodology}'. "
                f"Must be one of {valid_carbon}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_positions": self.max_positions,
            "waci_calculation_enabled": self.waci_calculation_enabled,
            "alignment_target_degrees": str(self.alignment_target_degrees),
            "sector_classification": self.sector_classification,
            "benchmark_index": self.benchmark_index,
            "enable_paris_alignment": self.enable_paris_alignment,
            "carbon_budget_methodology": self.carbon_budget_methodology,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioConfig":
        """Create from dictionary."""
        d = data.copy()
        if "alignment_target_degrees" in d:
            d["alignment_target_degrees"] = Decimal(str(d["alignment_target_degrees"]))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "PortfolioConfig":
        """Load from environment variables."""
        return cls(
            max_positions=int(os.getenv("GL_INV_PORT_MAX_POSITIONS", "100000")),
            waci_calculation_enabled=os.getenv(
                "GL_INV_PORT_WACI_ENABLED", "true"
            ).lower() == "true",
            alignment_target_degrees=Decimal(
                os.getenv("GL_INV_PORT_ALIGNMENT_TARGET", "1.5")
            ),
            sector_classification=os.getenv("GL_INV_PORT_SECTOR_CLASSIFICATION", "GICS"),
            benchmark_index=os.getenv("GL_INV_PORT_BENCHMARK", "MSCI_WORLD"),
            enable_paris_alignment=os.getenv(
                "GL_INV_PORT_PARIS_ALIGNMENT", "true"
            ).lower() == "true",
            carbon_budget_methodology=os.getenv(
                "GL_INV_PORT_CARBON_BUDGET_METHOD", "SDA"
            ),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass
class InvestmentsConfig:
    """
    Master configuration for Investments agent (AGENT-MRV-028).

    Composes all 18 configuration sections into a single unified
    configuration object with cross-validation support.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        redis: Redis cache configuration
        equity: Equity investment configuration
        debt: Debt investment configuration
        project_finance: Project finance configuration
        cre: Commercial real estate configuration
        mortgage: Mortgage configuration
        motor_vehicle: Motor vehicle loan configuration
        sovereign: Sovereign bond configuration
        pcaf: PCAF data quality configuration
        compliance: Regulatory compliance configuration
        provenance: Data provenance configuration
        uncertainty: Uncertainty quantification configuration
        data_quality: Data quality assessment configuration
        pipeline: Pipeline orchestration configuration
        metrics: Prometheus metrics configuration
        api: REST API configuration
        portfolio: Portfolio analytics configuration

    Example:
        >>> config = InvestmentsConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-015'
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    equity: EquityConfig = field(default_factory=EquityConfig)
    debt: DebtConfig = field(default_factory=DebtConfig)
    project_finance: ProjectFinanceConfig = field(default_factory=ProjectFinanceConfig)
    cre: CREConfig = field(default_factory=CREConfig)
    mortgage: MortgageConfig = field(default_factory=MortgageConfig)
    motor_vehicle: MotorVehicleConfig = field(default_factory=MotorVehicleConfig)
    sovereign: SovereignConfig = field(default_factory=SovereignConfig)
    pcaf: PCAFConfig = field(default_factory=PCAFConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    api: APIConfig = field(default_factory=APIConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)

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
            ("redis", self.redis),
            ("equity", self.equity),
            ("debt", self.debt),
            ("project_finance", self.project_finance),
            ("cre", self.cre),
            ("mortgage", self.mortgage),
            ("motor_vehicle", self.motor_vehicle),
            ("sovereign", self.sovereign),
            ("pcaf", self.pcaf),
            ("compliance", self.compliance),
            ("provenance", self.provenance),
            ("uncertainty", self.uncertainty),
            ("data_quality", self.data_quality),
            ("pipeline", self.pipeline),
            ("metrics", self.metrics),
            ("api", self.api),
            ("portfolio", self.portfolio),
        ]

        for section_name, section in sections:
            try:
                section.validate()
            except ValueError as e:
                errors.append(f"{section_name}: {str(e)}")

        errors.extend(self._cross_validate())
        return errors

    def _cross_validate(self) -> List[str]:
        """Perform cross-section validation checks."""
        errors: List[str] = []

        if self.general.api_prefix and not self.general.api_prefix.startswith("/api/"):
            errors.append(
                "cross-validation: general.api_prefix should start with '/api/'"
            )

        if self.database.pool_min > self.database.pool_max:
            errors.append(
                "cross-validation: database.pool_min must be <= database.pool_max"
            )

        if self.api.max_batch_size > self.general.max_batch_size * 100:
            errors.append(
                "cross-validation: api.max_batch_size is excessively large "
                "relative to general.max_batch_size"
            )

        if (
            self.compliance.max_acceptable_pcaf_score
            < self.pcaf.default_data_quality_score
        ):
            logger.warning(
                "Default PCAF score (%d) exceeds compliance max (%d) - "
                "positions may fail compliance by default",
                self.pcaf.default_data_quality_score,
                self.compliance.max_acceptable_pcaf_score,
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "redis": self.redis.to_dict(),
            "equity": self.equity.to_dict(),
            "debt": self.debt.to_dict(),
            "project_finance": self.project_finance.to_dict(),
            "cre": self.cre.to_dict(),
            "mortgage": self.mortgage.to_dict(),
            "motor_vehicle": self.motor_vehicle.to_dict(),
            "sovereign": self.sovereign.to_dict(),
            "pcaf": self.pcaf.to_dict(),
            "compliance": self.compliance.to_dict(),
            "provenance": self.provenance.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "data_quality": self.data_quality.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "metrics": self.metrics.to_dict(),
            "api": self.api.to_dict(),
            "portfolio": self.portfolio.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestmentsConfig":
        """Create configuration from dictionary."""
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            redis=RedisConfig.from_dict(data.get("redis", {})),
            equity=EquityConfig.from_dict(data.get("equity", {})),
            debt=DebtConfig.from_dict(data.get("debt", {})),
            project_finance=ProjectFinanceConfig.from_dict(data.get("project_finance", {})),
            cre=CREConfig.from_dict(data.get("cre", {})),
            mortgage=MortgageConfig.from_dict(data.get("mortgage", {})),
            motor_vehicle=MotorVehicleConfig.from_dict(data.get("motor_vehicle", {})),
            sovereign=SovereignConfig.from_dict(data.get("sovereign", {})),
            pcaf=PCAFConfig.from_dict(data.get("pcaf", {})),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
            uncertainty=UncertaintyConfig.from_dict(data.get("uncertainty", {})),
            data_quality=DataQualityConfig.from_dict(data.get("data_quality", {})),
            pipeline=PipelineConfig.from_dict(data.get("pipeline", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
            api=APIConfig.from_dict(data.get("api", {})),
            portfolio=PortfolioConfig.from_dict(data.get("portfolio", {})),
        )

    @classmethod
    def from_env(cls) -> "InvestmentsConfig":
        """Load configuration from environment variables."""
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            equity=EquityConfig.from_env(),
            debt=DebtConfig.from_env(),
            project_finance=ProjectFinanceConfig.from_env(),
            cre=CREConfig.from_env(),
            mortgage=MortgageConfig.from_env(),
            motor_vehicle=MotorVehicleConfig.from_env(),
            sovereign=SovereignConfig.from_env(),
            pcaf=PCAFConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            data_quality=DataQualityConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            api=APIConfig.from_env(),
            portfolio=PortfolioConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[InvestmentsConfig] = None
_config_lock = threading.RLock()


def get_config() -> InvestmentsConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization. First call loads from environment
    variables. Subsequent calls return the cached instance.

    Returns:
        InvestmentsConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-015'
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info("Initializing InvestmentsConfig from environment")
                config = InvestmentsConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning("Configuration validation warning: %s", error)
                _config_instance = config
                logger.info(
                    "InvestmentsConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: InvestmentsConfig) -> None:
    """
    Set the singleton configuration instance.

    Args:
        config: InvestmentsConfig instance to set as singleton

    Raises:
        TypeError: If config is not an InvestmentsConfig instance
    """
    global _config_instance

    if not isinstance(config, InvestmentsConfig):
        raise TypeError(
            f"config must be an InvestmentsConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("InvestmentsConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Clears the cached configuration singleton, forcing the next call to
    get_config() to reload from environment variables.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("InvestmentsConfig singleton reset")


def validate_config(config: InvestmentsConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate_all()


def print_config(config: InvestmentsConfig) -> None:
    """
    Print configuration in human-readable format.

    Sensitive fields (passwords, connection URLs) are redacted.

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
    print("  Investments Configuration (AGENT-MRV-028)")
    print("  Agent ID: " + config.general.agent_id)
    print("  Version:  " + config.general.version)
    print("=" * 64)

    _print_section("GENERAL", config.general.to_dict())
    _print_section("DATABASE", config.database.to_dict())
    _print_section("REDIS", config.redis.to_dict())
    _print_section("EQUITY", config.equity.to_dict())
    _print_section("DEBT", config.debt.to_dict())
    _print_section("PROJECT_FINANCE", config.project_finance.to_dict())
    _print_section("CRE", config.cre.to_dict())
    _print_section("MORTGAGE", config.mortgage.to_dict())
    _print_section("MOTOR_VEHICLE", config.motor_vehicle.to_dict())
    _print_section("SOVEREIGN", config.sovereign.to_dict())
    _print_section("PCAF", config.pcaf.to_dict())
    _print_section("COMPLIANCE", config.compliance.to_dict())
    _print_section("PROVENANCE", config.provenance.to_dict())
    _print_section("UNCERTAINTY", config.uncertainty.to_dict())
    _print_section("DATA_QUALITY", config.data_quality.to_dict())
    _print_section("PIPELINE", config.pipeline.to_dict())
    _print_section("METRICS", config.metrics.to_dict())
    _print_section("API", config.api.to_dict())
    _print_section("PORTFOLIO", config.portfolio.to_dict())

    errors = config.validate_all()
    print("\n[VALIDATION]")
    if errors:
        print(f"  status: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"  - {error}")
    else:
        print("  status: PASSED (all 19 sections valid)")

    print("\n" + "=" * 64)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration dataclasses
    "GeneralConfig",
    "DatabaseConfig",
    "RedisConfig",
    "EquityConfig",
    "DebtConfig",
    "ProjectFinanceConfig",
    "CREConfig",
    "MortgageConfig",
    "MotorVehicleConfig",
    "SovereignConfig",
    "PCAFConfig",
    "ComplianceConfig",
    "ProvenanceConfig",
    "UncertaintyConfig",
    "DataQualityConfig",
    "PipelineConfig",
    "MetricsConfig",
    "APIConfig",
    "PortfolioConfig",
    # Master configuration
    "InvestmentsConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
]
