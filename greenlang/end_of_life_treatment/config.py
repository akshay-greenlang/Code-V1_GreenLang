# -*- coding: utf-8 -*-
"""
End-of-Life Treatment of Sold Products Configuration - AGENT-MRV-025

Thread-safe singleton configuration for GL-MRV-S3-012.
All environment variables prefixed with GL_EOL_.

This module provides comprehensive configuration management for the End-of-Life
Treatment of Sold Products agent (GHG Protocol Scope 3 Category 12), supporting:
- Landfill emissions (CH4/N2O with DOC/DOCf/MCF/FOD modelling)
- Incineration emissions (biogenic/fossil CO2, N2O, CH4, energy recovery credits)
- Recycling emissions (cut-off/allocation/substitution, MRF processing, avoided)
- Composting emissions (aerobic/anaerobic, CH4/N2O)
- Anaerobic digestion emissions (biogas capture, digestate management)
- Open burning emissions (uncontrolled combustion)
- Average-data and producer-specific calculation methods
- Hybrid method with waterfall gap-filling across data tiers
- Circular economy metrics (recycling rate, diversion, circularity index)
- 7 regulatory frameworks (GHG Protocol Scope 3, ISO 14064, CSRD ESRS E1+E5,
  CDP, SBTi, EU Waste Framework Directive, EPA)
- Double-counting prevention (vs Cat 1/Cat 5/Scope 1)
- Provenance tracking and audit trails

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-S3-012'
    >>> config.landfill.default_docf
    Decimal('0.5')
    >>> config.compliance.double_counting_check
    True

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_EOL_ prefix. See individual config sections for specific variables.
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
    General configuration for End-of-Life Treatment of Sold Products agent.

    Attributes:
        enabled: Master switch for the agent (GL_EOL_ENABLED)
        debug: Enable debug mode with verbose logging (GL_EOL_DEBUG)
        log_level: Logging level - DEBUG/INFO/WARNING/ERROR/CRITICAL (GL_EOL_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_EOL_AGENT_ID)
        agent_component: Agent component identifier (GL_EOL_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_EOL_VERSION)
        table_prefix: Database table prefix (GL_EOL_TABLE_PREFIX)
        api_prefix: API route prefix (GL_EOL_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_EOL_MAX_BATCH_SIZE)
        default_gwp: Default GWP assessment report version (GL_EOL_DEFAULT_GWP)

    Example:
        >>> general = GeneralConfig(
        ...     enabled=True,
        ...     debug=False,
        ...     log_level="INFO",
        ...     agent_id="GL-MRV-S3-012",
        ...     agent_component="AGENT-MRV-025",
        ...     version="1.0.0",
        ...     table_prefix="gl_eol_",
        ...     api_prefix="/api/v1/end-of-life-treatment",
        ...     max_batch_size=500,
        ...     default_gwp="AR5"
        ... )
        >>> general.agent_id
        'GL-MRV-S3-012'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-S3-012"
    agent_component: str = "AGENT-MRV-025"
    version: str = "1.0.0"
    table_prefix: str = "gl_eol_"
    api_prefix: str = "/api/v1/end-of-life-treatment"
    max_batch_size: int = 500
    default_gwp: str = "AR5"

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

        if not self.api_prefix:
            raise ValueError("api_prefix cannot be empty")

        if not self.api_prefix.startswith("/"):
            raise ValueError("api_prefix must start with '/'")

        if self.max_batch_size < 1 or self.max_batch_size > 10000:
            raise ValueError("max_batch_size must be between 1 and 10000")

        valid_gwp = {"AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. Must be one of {valid_gwp}"
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
            "table_prefix": self.table_prefix,
            "api_prefix": self.api_prefix,
            "max_batch_size": self.max_batch_size,
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
            enabled=os.getenv("GL_EOL_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_EOL_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_EOL_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_EOL_AGENT_ID", "GL-MRV-S3-012"),
            agent_component=os.getenv("GL_EOL_AGENT_COMPONENT", "AGENT-MRV-025"),
            version=os.getenv("GL_EOL_VERSION", "1.0.0"),
            table_prefix=os.getenv("GL_EOL_TABLE_PREFIX", "gl_eol_"),
            api_prefix=os.getenv("GL_EOL_API_PREFIX", "/api/v1/end-of-life-treatment"),
            max_batch_size=int(os.getenv("GL_EOL_MAX_BATCH_SIZE", "500")),
            default_gwp=os.getenv("GL_EOL_DEFAULT_GWP", "AR5"),
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for End-of-Life Treatment of Sold Products agent.

    Attributes:
        database_url: PostgreSQL connection URL (GL_EOL_DATABASE_URL)
        pool_size: Connection pool size (GL_EOL_DATABASE_POOL_SIZE)
        max_overflow: Max overflow connections (GL_EOL_DATABASE_MAX_OVERFLOW)
        pool_timeout: Pool timeout in seconds (GL_EOL_DATABASE_POOL_TIMEOUT)
        schema: Database schema name (GL_EOL_DATABASE_SCHEMA)
        ssl_mode: PostgreSQL SSL mode (GL_EOL_DATABASE_SSL_MODE)
        statement_timeout: SQL statement timeout in ms (GL_EOL_DATABASE_STATEMENT_TIMEOUT)
        connect_timeout: Connection timeout in seconds (GL_EOL_DATABASE_CONNECT_TIMEOUT)

    Example:
        >>> db = DatabaseConfig(
        ...     database_url="postgresql://greenlang:greenlang@localhost:5432/greenlang",
        ...     pool_size=10,
        ...     max_overflow=20,
        ...     pool_timeout=30,
        ...     schema="end_of_life_treatment_service",
        ...     ssl_mode="prefer",
        ...     statement_timeout=30000,
        ...     connect_timeout=10
        ... )
        >>> db.schema
        'end_of_life_treatment_service'
    """

    database_url: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    schema: str = "end_of_life_treatment_service"
    ssl_mode: str = "prefer"
    statement_timeout: int = 30000
    connect_timeout: int = 10

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

        valid_ssl_modes = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl_modes:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. Must be one of {valid_ssl_modes}"
            )

        if self.statement_timeout < 1000:
            raise ValueError("statement_timeout must be >= 1000 ms")

        if self.connect_timeout < 1:
            raise ValueError("connect_timeout must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "schema": self.schema,
            "ssl_mode": self.ssl_mode,
            "statement_timeout": self.statement_timeout,
            "connect_timeout": self.connect_timeout,
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
                "GL_EOL_DATABASE_URL",
                "postgresql://greenlang:greenlang@localhost:5432/greenlang",
            ),
            pool_size=int(os.getenv("GL_EOL_DATABASE_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("GL_EOL_DATABASE_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("GL_EOL_DATABASE_POOL_TIMEOUT", "30")),
            schema=os.getenv("GL_EOL_DATABASE_SCHEMA", "end_of_life_treatment_service"),
            ssl_mode=os.getenv("GL_EOL_DATABASE_SSL_MODE", "prefer"),
            statement_timeout=int(os.getenv("GL_EOL_DATABASE_STATEMENT_TIMEOUT", "30000")),
            connect_timeout=int(os.getenv("GL_EOL_DATABASE_CONNECT_TIMEOUT", "10")),
        )


# =============================================================================
# SECTION 3: WASTE TYPE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class WasteTypeConfig:
    """
    Waste type and treatment pathway configuration for end-of-life treatment.

    Controls which treatment pathways are enabled and provides defaults for
    waste classification when product-specific end-of-life data is unavailable.

    Attributes:
        enable_landfill: Enable landfill treatment pathway (GL_EOL_ENABLE_LANDFILL)
        enable_incineration: Enable incineration pathway (GL_EOL_ENABLE_INCINERATION)
        enable_recycling: Enable recycling pathway (GL_EOL_ENABLE_RECYCLING)
        enable_composting: Enable composting pathway (GL_EOL_ENABLE_COMPOSTING)
        enable_ad: Enable anaerobic digestion pathway (GL_EOL_ENABLE_AD)
        enable_open_burning: Enable open burning pathway (GL_EOL_ENABLE_OPEN_BURNING)
        default_climate_zone: Default IPCC climate zone (GL_EOL_DEFAULT_CLIMATE_ZONE)
        default_landfill_type: Default landfill type (GL_EOL_DEFAULT_LANDFILL_TYPE)

    Example:
        >>> waste_type = WasteTypeConfig(
        ...     enable_landfill=True,
        ...     enable_incineration=True,
        ...     enable_recycling=True,
        ...     enable_composting=True,
        ...     enable_ad=True,
        ...     enable_open_burning=False,
        ...     default_climate_zone="TEMPERATE_WET",
        ...     default_landfill_type="MANAGED_ANAEROBIC"
        ... )
        >>> waste_type.default_climate_zone
        'TEMPERATE_WET'
    """

    enable_landfill: bool = True
    enable_incineration: bool = True
    enable_recycling: bool = True
    enable_composting: bool = True
    enable_ad: bool = True
    enable_open_burning: bool = False
    default_climate_zone: str = "TEMPERATE_WET"
    default_landfill_type: str = "MANAGED_ANAEROBIC"

    def validate(self) -> None:
        """
        Validate waste type configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_climate_zones = {
            "BOREAL_ARCTIC_DRY", "BOREAL_ARCTIC_WET",
            "TEMPERATE_DRY", "TEMPERATE_WET",
            "TROPICAL_DRY", "TROPICAL_WET",
            "TROPICAL_MONTANE", "DEFAULT",
        }
        if self.default_climate_zone not in valid_climate_zones:
            raise ValueError(
                f"Invalid default_climate_zone '{self.default_climate_zone}'. "
                f"Must be one of {valid_climate_zones}"
            )

        valid_landfill_types = {
            "MANAGED_AEROBIC", "MANAGED_ANAEROBIC",
            "UNMANAGED_SHALLOW", "UNMANAGED_DEEP",
            "SEMI_AEROBIC",
        }
        if self.default_landfill_type not in valid_landfill_types:
            raise ValueError(
                f"Invalid default_landfill_type '{self.default_landfill_type}'. "
                f"Must be one of {valid_landfill_types}"
            )

        # At least one treatment must be enabled
        any_enabled = (
            self.enable_landfill or self.enable_incineration
            or self.enable_recycling or self.enable_composting
            or self.enable_ad or self.enable_open_burning
        )
        if not any_enabled:
            raise ValueError("At least one treatment pathway must be enabled")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_landfill": self.enable_landfill,
            "enable_incineration": self.enable_incineration,
            "enable_recycling": self.enable_recycling,
            "enable_composting": self.enable_composting,
            "enable_ad": self.enable_ad,
            "enable_open_burning": self.enable_open_burning,
            "default_climate_zone": self.default_climate_zone,
            "default_landfill_type": self.default_landfill_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WasteTypeConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "WasteTypeConfig":
        """Load from environment variables."""
        return cls(
            enable_landfill=os.getenv("GL_EOL_ENABLE_LANDFILL", "true").lower() == "true",
            enable_incineration=os.getenv("GL_EOL_ENABLE_INCINERATION", "true").lower() == "true",
            enable_recycling=os.getenv("GL_EOL_ENABLE_RECYCLING", "true").lower() == "true",
            enable_composting=os.getenv("GL_EOL_ENABLE_COMPOSTING", "true").lower() == "true",
            enable_ad=os.getenv("GL_EOL_ENABLE_AD", "true").lower() == "true",
            enable_open_burning=os.getenv("GL_EOL_ENABLE_OPEN_BURNING", "false").lower() == "true",
            default_climate_zone=os.getenv("GL_EOL_DEFAULT_CLIMATE_ZONE", "TEMPERATE_WET"),
            default_landfill_type=os.getenv("GL_EOL_DEFAULT_LANDFILL_TYPE", "MANAGED_ANAEROBIC"),
        )


# =============================================================================
# SECTION 4: AVERAGE DATA CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AverageDataConfig:
    """
    Average-data calculation method configuration.

    Controls the average-data fallback method used when product-specific or
    producer-specific end-of-life treatment data is unavailable. Uses regional
    waste composition and treatment split statistics.

    Attributes:
        default_region: Default region for waste statistics (GL_EOL_AVG_DEFAULT_REGION)
        fallback_to_mixed: Fall back to mixed waste EFs (GL_EOL_AVG_FALLBACK_TO_MIXED)
        weight_estimation: Enable weight estimation from product type (GL_EOL_AVG_WEIGHT_ESTIMATION)
        default_treatment_split_source: Source for treatment split (GL_EOL_AVG_TREATMENT_SPLIT_SOURCE)
        include_transport_to_treatment: Include transport emissions (GL_EOL_AVG_INCLUDE_TRANSPORT)

    Example:
        >>> avg = AverageDataConfig(
        ...     default_region="GLOBAL",
        ...     fallback_to_mixed=True,
        ...     weight_estimation=True
        ... )
        >>> avg.default_region
        'GLOBAL'
    """

    default_region: str = "GLOBAL"
    fallback_to_mixed: bool = True
    weight_estimation: bool = True
    default_treatment_split_source: str = "EUROSTAT"
    include_transport_to_treatment: bool = False

    def validate(self) -> None:
        """
        Validate average data configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_regions = {
            "GLOBAL", "EU27", "US", "UK", "CN", "IN", "JP",
            "BR", "AU", "CA", "KR", "OECD", "NON_OECD",
        }
        if self.default_region not in valid_regions:
            raise ValueError(
                f"Invalid default_region '{self.default_region}'. "
                f"Must be one of {valid_regions}"
            )

        valid_sources = {"EUROSTAT", "EPA", "DEFRA", "OECD", "WORLD_BANK", "CUSTOM"}
        if self.default_treatment_split_source not in valid_sources:
            raise ValueError(
                f"Invalid default_treatment_split_source "
                f"'{self.default_treatment_split_source}'. "
                f"Must be one of {valid_sources}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_region": self.default_region,
            "fallback_to_mixed": self.fallback_to_mixed,
            "weight_estimation": self.weight_estimation,
            "default_treatment_split_source": self.default_treatment_split_source,
            "include_transport_to_treatment": self.include_transport_to_treatment,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AverageDataConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "AverageDataConfig":
        """Load from environment variables."""
        return cls(
            default_region=os.getenv("GL_EOL_AVG_DEFAULT_REGION", "GLOBAL"),
            fallback_to_mixed=os.getenv(
                "GL_EOL_AVG_FALLBACK_TO_MIXED", "true"
            ).lower() == "true",
            weight_estimation=os.getenv(
                "GL_EOL_AVG_WEIGHT_ESTIMATION", "true"
            ).lower() == "true",
            default_treatment_split_source=os.getenv(
                "GL_EOL_AVG_TREATMENT_SPLIT_SOURCE", "EUROSTAT"
            ),
            include_transport_to_treatment=os.getenv(
                "GL_EOL_AVG_INCLUDE_TRANSPORT", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 5: PRODUCER-SPECIFIC CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProducerSpecificConfig:
    """
    Producer-specific calculation method configuration.

    Controls how producer-provided end-of-life data (EPD, take-back data,
    product stewardship reports) is processed and validated.

    Attributes:
        require_verification: Require third-party verification (GL_EOL_PS_REQUIRE_VERIFICATION)
        epd_validation: Enable EPD validation (GL_EOL_PS_EPD_VALIDATION)
        epr_tracking: Enable Extended Producer Responsibility tracking (GL_EOL_PS_EPR_TRACKING)
        accept_unverified: Accept unverified producer data with flag (GL_EOL_PS_ACCEPT_UNVERIFIED)
        min_confidence: Minimum confidence for producer data (GL_EOL_PS_MIN_CONFIDENCE)

    Example:
        >>> ps = ProducerSpecificConfig(
        ...     require_verification=True,
        ...     epd_validation=True,
        ...     epr_tracking=True
        ... )
        >>> ps.require_verification
        True
    """

    require_verification: bool = True
    epd_validation: bool = True
    epr_tracking: bool = True
    accept_unverified: bool = True
    min_confidence: Decimal = Decimal("0.7")

    def validate(self) -> None:
        """
        Validate producer-specific configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.min_confidence < Decimal("0") or self.min_confidence > Decimal("1"):
            raise ValueError("min_confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "require_verification": self.require_verification,
            "epd_validation": self.epd_validation,
            "epr_tracking": self.epr_tracking,
            "accept_unverified": self.accept_unverified,
            "min_confidence": str(self.min_confidence),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProducerSpecificConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "min_confidence" in data_copy:
            data_copy["min_confidence"] = Decimal(data_copy["min_confidence"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "ProducerSpecificConfig":
        """Load from environment variables."""
        return cls(
            require_verification=os.getenv(
                "GL_EOL_PS_REQUIRE_VERIFICATION", "true"
            ).lower() == "true",
            epd_validation=os.getenv(
                "GL_EOL_PS_EPD_VALIDATION", "true"
            ).lower() == "true",
            epr_tracking=os.getenv(
                "GL_EOL_PS_EPR_TRACKING", "true"
            ).lower() == "true",
            accept_unverified=os.getenv(
                "GL_EOL_PS_ACCEPT_UNVERIFIED", "true"
            ).lower() == "true",
            min_confidence=Decimal(os.getenv("GL_EOL_PS_MIN_CONFIDENCE", "0.7")),
        )


# =============================================================================
# SECTION 6: HYBRID METHOD CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class HybridConfig:
    """
    Hybrid calculation method configuration.

    Controls the hybrid method that combines multiple data sources in a
    waterfall pattern, filling gaps with progressively lower-tier data.

    Attributes:
        method_waterfall: Ordered list of methods to try (GL_EOL_HYBRID_METHOD_WATERFALL)
        gap_filling: Enable gap-filling from lower-tier data (GL_EOL_HYBRID_GAP_FILLING)
        avoided_emissions_separate: Report avoided emissions separately (GL_EOL_HYBRID_AVOIDED_SEPARATE)
        min_coverage: Minimum data coverage required (GL_EOL_HYBRID_MIN_COVERAGE)

    Example:
        >>> hybrid = HybridConfig(
        ...     method_waterfall="PRODUCER_SPECIFIC,WASTE_TYPE_SPECIFIC,AVERAGE_DATA",
        ...     gap_filling=True,
        ...     avoided_emissions_separate=True
        ... )
        >>> hybrid.get_waterfall()
        ['PRODUCER_SPECIFIC', 'WASTE_TYPE_SPECIFIC', 'AVERAGE_DATA']
    """

    method_waterfall: str = "PRODUCER_SPECIFIC,WASTE_TYPE_SPECIFIC,AVERAGE_DATA"
    gap_filling: bool = True
    avoided_emissions_separate: bool = True
    min_coverage: Decimal = Decimal("0.8")

    def validate(self) -> None:
        """
        Validate hybrid method configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {
            "PRODUCER_SPECIFIC", "WASTE_TYPE_SPECIFIC",
            "TREATMENT_SPECIFIC", "AVERAGE_DATA",
        }

        waterfall = self.get_waterfall()
        if not waterfall:
            raise ValueError("method_waterfall must contain at least one method")

        for method in waterfall:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method in waterfall '{method}'. "
                    f"Must be one of {valid_methods}"
                )

        if self.min_coverage < Decimal("0") or self.min_coverage > Decimal("1"):
            raise ValueError("min_coverage must be between 0 and 1")

    def get_waterfall(self) -> List[str]:
        """Parse method waterfall string into list."""
        return [m.strip() for m in self.method_waterfall.split(",") if m.strip()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_waterfall": self.method_waterfall,
            "gap_filling": self.gap_filling,
            "avoided_emissions_separate": self.avoided_emissions_separate,
            "min_coverage": str(self.min_coverage),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        if "min_coverage" in data_copy:
            data_copy["min_coverage"] = Decimal(data_copy["min_coverage"])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "HybridConfig":
        """Load from environment variables."""
        return cls(
            method_waterfall=os.getenv(
                "GL_EOL_HYBRID_METHOD_WATERFALL",
                "PRODUCER_SPECIFIC,WASTE_TYPE_SPECIFIC,AVERAGE_DATA",
            ),
            gap_filling=os.getenv(
                "GL_EOL_HYBRID_GAP_FILLING", "true"
            ).lower() == "true",
            avoided_emissions_separate=os.getenv(
                "GL_EOL_HYBRID_AVOIDED_SEPARATE", "true"
            ).lower() == "true",
            min_coverage=Decimal(os.getenv("GL_EOL_HYBRID_MIN_COVERAGE", "0.8")),
        )


# =============================================================================
# SECTION 7: LANDFILL CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class LandfillConfig:
    """
    Landfill configuration for end-of-life treatment emissions.

    Controls IPCC First Order Decay (FOD) model parameters for landfill
    methane generation modelling. FOD model: CH4 = DDOCm x F x (16/12)
    where DDOCm = W x DOC x DOCf x MCF.

    Attributes:
        fod_model_years: Years for FOD model projection (GL_EOL_LANDFILL_FOD_YEARS)
        default_doc: Default degradable organic carbon (GL_EOL_LANDFILL_DEFAULT_DOC)
        default_docf: Default DOC fraction dissimilated (GL_EOL_LANDFILL_DEFAULT_DOCF)
        default_mcf: Default methane correction factor (GL_EOL_LANDFILL_DEFAULT_MCF)
        default_f: Default fraction of CH4 in landfill gas (GL_EOL_LANDFILL_DEFAULT_F)
        default_oxidation: Default oxidation factor (GL_EOL_LANDFILL_DEFAULT_OXIDATION)
        default_decay_rate: Default decay rate k (GL_EOL_LANDFILL_DEFAULT_DECAY_RATE)
        default_gas_capture: Default gas capture efficiency (GL_EOL_LANDFILL_DEFAULT_GAS_CAPTURE)
        methane_gwp: GWP of methane (GL_EOL_LANDFILL_METHANE_GWP)

    Example:
        >>> landfill = LandfillConfig(
        ...     fod_model_years=100,
        ...     default_doc=Decimal("0.15"),
        ...     default_docf=Decimal("0.5"),
        ...     default_mcf=Decimal("1.0"),
        ...     methane_gwp=28
        ... )
        >>> landfill.default_docf
        Decimal('0.5')
    """

    fod_model_years: int = 100
    default_doc: Decimal = Decimal("0.15")
    default_docf: Decimal = Decimal("0.5")
    default_mcf: Decimal = Decimal("1.0")
    default_f: Decimal = Decimal("0.5")
    default_oxidation: Decimal = Decimal("0.1")
    default_decay_rate: Decimal = Decimal("0.05")
    default_gas_capture: Decimal = Decimal("0.0")
    methane_gwp: int = 28

    def validate(self) -> None:
        """
        Validate landfill configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.fod_model_years < 1 or self.fod_model_years > 200:
            raise ValueError("fod_model_years must be between 1 and 200")

        if self.default_doc < Decimal("0") or self.default_doc > Decimal("1"):
            raise ValueError("default_doc must be between 0 and 1")

        if self.default_docf < Decimal("0") or self.default_docf > Decimal("1"):
            raise ValueError("default_docf must be between 0 and 1")

        if self.default_mcf < Decimal("0") or self.default_mcf > Decimal("1"):
            raise ValueError("default_mcf must be between 0 and 1")

        if self.default_f < Decimal("0") or self.default_f > Decimal("1"):
            raise ValueError("default_f must be between 0 and 1")

        if self.default_oxidation < Decimal("0") or self.default_oxidation > Decimal("1"):
            raise ValueError("default_oxidation must be between 0 and 1")

        if self.default_decay_rate <= Decimal("0") or self.default_decay_rate > Decimal("1"):
            raise ValueError("default_decay_rate must be between 0 (exclusive) and 1")

        if self.default_gas_capture < Decimal("0") or self.default_gas_capture > Decimal("1"):
            raise ValueError("default_gas_capture must be between 0 and 1")

        if self.methane_gwp < 1 or self.methane_gwp > 100:
            raise ValueError("methane_gwp must be between 1 and 100")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fod_model_years": self.fod_model_years,
            "default_doc": str(self.default_doc),
            "default_docf": str(self.default_docf),
            "default_mcf": str(self.default_mcf),
            "default_f": str(self.default_f),
            "default_oxidation": str(self.default_oxidation),
            "default_decay_rate": str(self.default_decay_rate),
            "default_gas_capture": str(self.default_gas_capture),
            "methane_gwp": self.methane_gwp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LandfillConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in [
            "default_doc", "default_docf", "default_mcf", "default_f",
            "default_oxidation", "default_decay_rate", "default_gas_capture",
        ]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "LandfillConfig":
        """Load from environment variables."""
        return cls(
            fod_model_years=int(os.getenv("GL_EOL_LANDFILL_FOD_YEARS", "100")),
            default_doc=Decimal(os.getenv("GL_EOL_LANDFILL_DEFAULT_DOC", "0.15")),
            default_docf=Decimal(os.getenv("GL_EOL_LANDFILL_DEFAULT_DOCF", "0.5")),
            default_mcf=Decimal(os.getenv("GL_EOL_LANDFILL_DEFAULT_MCF", "1.0")),
            default_f=Decimal(os.getenv("GL_EOL_LANDFILL_DEFAULT_F", "0.5")),
            default_oxidation=Decimal(
                os.getenv("GL_EOL_LANDFILL_DEFAULT_OXIDATION", "0.1")
            ),
            default_decay_rate=Decimal(
                os.getenv("GL_EOL_LANDFILL_DEFAULT_DECAY_RATE", "0.05")
            ),
            default_gas_capture=Decimal(
                os.getenv("GL_EOL_LANDFILL_DEFAULT_GAS_CAPTURE", "0.0")
            ),
            methane_gwp=int(os.getenv("GL_EOL_LANDFILL_METHANE_GWP", "28")),
        )


# =============================================================================
# SECTION 8: INCINERATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class IncinerationConfig:
    """
    Incineration configuration for end-of-life treatment emissions.

    Controls parameters for waste incineration emissions modelling including
    biogenic/fossil CO2 separation and energy recovery credits.

    Attributes:
        include_biogenic_co2: Include biogenic CO2 in total (GL_EOL_INCIN_INCLUDE_BIOGENIC)
        energy_recovery_credit: Enable energy recovery credit (GL_EOL_INCIN_ENERGY_CREDIT)
        default_oxidation: Default oxidation factor (GL_EOL_INCIN_DEFAULT_OXIDATION)
        default_fossil_fraction: Default fossil carbon fraction (GL_EOL_INCIN_DEFAULT_FOSSIL)
        default_energy_recovery: Default energy recovery efficiency (GL_EOL_INCIN_DEFAULT_RECOVERY)
        default_incinerator_type: Default incinerator type (GL_EOL_INCIN_DEFAULT_TYPE)

    Example:
        >>> incin = IncinerationConfig(
        ...     include_biogenic_co2=False,
        ...     energy_recovery_credit=True,
        ...     default_oxidation=Decimal("1.0")
        ... )
        >>> incin.energy_recovery_credit
        True
    """

    include_biogenic_co2: bool = False
    energy_recovery_credit: bool = True
    default_oxidation: Decimal = Decimal("1.0")
    default_fossil_fraction: Decimal = Decimal("1.0")
    default_energy_recovery: Decimal = Decimal("0.0")
    default_incinerator_type: str = "MASS_BURN"

    def validate(self) -> None:
        """
        Validate incineration configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.default_oxidation < Decimal("0") or self.default_oxidation > Decimal("1"):
            raise ValueError("default_oxidation must be between 0 and 1")

        if self.default_fossil_fraction < Decimal("0") or self.default_fossil_fraction > Decimal("1"):
            raise ValueError("default_fossil_fraction must be between 0 and 1")

        if self.default_energy_recovery < Decimal("0") or self.default_energy_recovery > Decimal("1"):
            raise ValueError("default_energy_recovery must be between 0 and 1")

        valid_types = {
            "MASS_BURN", "FLUIDIZED_BED", "ROTARY_KILN",
            "RDF", "GASIFICATION", "PYROLYSIS",
        }
        if self.default_incinerator_type not in valid_types:
            raise ValueError(
                f"Invalid default_incinerator_type '{self.default_incinerator_type}'. "
                f"Must be one of {valid_types}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "include_biogenic_co2": self.include_biogenic_co2,
            "energy_recovery_credit": self.energy_recovery_credit,
            "default_oxidation": str(self.default_oxidation),
            "default_fossil_fraction": str(self.default_fossil_fraction),
            "default_energy_recovery": str(self.default_energy_recovery),
            "default_incinerator_type": self.default_incinerator_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncinerationConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_oxidation", "default_fossil_fraction", "default_energy_recovery"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "IncinerationConfig":
        """Load from environment variables."""
        return cls(
            include_biogenic_co2=os.getenv(
                "GL_EOL_INCIN_INCLUDE_BIOGENIC", "false"
            ).lower() == "true",
            energy_recovery_credit=os.getenv(
                "GL_EOL_INCIN_ENERGY_CREDIT", "true"
            ).lower() == "true",
            default_oxidation=Decimal(
                os.getenv("GL_EOL_INCIN_DEFAULT_OXIDATION", "1.0")
            ),
            default_fossil_fraction=Decimal(
                os.getenv("GL_EOL_INCIN_DEFAULT_FOSSIL", "1.0")
            ),
            default_energy_recovery=Decimal(
                os.getenv("GL_EOL_INCIN_DEFAULT_RECOVERY", "0.0")
            ),
            default_incinerator_type=os.getenv(
                "GL_EOL_INCIN_DEFAULT_TYPE", "MASS_BURN"
            ),
        )


# =============================================================================
# SECTION 9: RECYCLING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RecyclingConfig:
    """
    Recycling configuration for end-of-life treatment emissions.

    Controls recycling approach (cut-off vs allocation vs substitution),
    transport to MRF, and avoided emissions methodology.

    Attributes:
        approach: Recycling approach (GL_EOL_RECYCLING_APPROACH)
        include_transport: Include transport to MRF (GL_EOL_RECYCLING_INCLUDE_TRANSPORT)
        include_mrf: Include MRF processing emissions (GL_EOL_RECYCLING_INCLUDE_MRF)
        avoided_emissions_method: Method for avoided emissions (GL_EOL_RECYCLING_AVOIDED_METHOD)
        default_quality_factor: Default quality factor for downcycling (GL_EOL_RECYCLING_QUALITY_FACTOR)
        allocation_factor: Allocation factor for shared benefit (GL_EOL_RECYCLING_ALLOCATION_FACTOR)

    Example:
        >>> recycling = RecyclingConfig(
        ...     approach="CUT_OFF",
        ...     include_transport=True,
        ...     include_mrf=True,
        ...     avoided_emissions_method="SUBSTITUTION"
        ... )
        >>> recycling.approach
        'CUT_OFF'
    """

    approach: str = "CUT_OFF"
    include_transport: bool = True
    include_mrf: bool = True
    avoided_emissions_method: str = "SUBSTITUTION"
    default_quality_factor: Decimal = Decimal("1.0")
    allocation_factor: Decimal = Decimal("0.5")

    def validate(self) -> None:
        """
        Validate recycling configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_approaches = {"CUT_OFF", "ALLOCATION", "SUBSTITUTION", "CONSEQUENTIAL"}
        if self.approach not in valid_approaches:
            raise ValueError(
                f"Invalid approach '{self.approach}'. Must be one of {valid_approaches}"
            )

        valid_avoided_methods = {"SUBSTITUTION", "SYSTEM_EXPANSION", "ALLOCATION", "NONE"}
        if self.avoided_emissions_method not in valid_avoided_methods:
            raise ValueError(
                f"Invalid avoided_emissions_method '{self.avoided_emissions_method}'. "
                f"Must be one of {valid_avoided_methods}"
            )

        if self.default_quality_factor < Decimal("0") or self.default_quality_factor > Decimal("1"):
            raise ValueError("default_quality_factor must be between 0 and 1")

        if self.allocation_factor < Decimal("0") or self.allocation_factor > Decimal("1"):
            raise ValueError("allocation_factor must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "approach": self.approach,
            "include_transport": self.include_transport,
            "include_mrf": self.include_mrf,
            "avoided_emissions_method": self.avoided_emissions_method,
            "default_quality_factor": str(self.default_quality_factor),
            "allocation_factor": str(self.allocation_factor),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecyclingConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["default_quality_factor", "allocation_factor"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "RecyclingConfig":
        """Load from environment variables."""
        return cls(
            approach=os.getenv("GL_EOL_RECYCLING_APPROACH", "CUT_OFF"),
            include_transport=os.getenv(
                "GL_EOL_RECYCLING_INCLUDE_TRANSPORT", "true"
            ).lower() == "true",
            include_mrf=os.getenv(
                "GL_EOL_RECYCLING_INCLUDE_MRF", "true"
            ).lower() == "true",
            avoided_emissions_method=os.getenv(
                "GL_EOL_RECYCLING_AVOIDED_METHOD", "SUBSTITUTION"
            ),
            default_quality_factor=Decimal(
                os.getenv("GL_EOL_RECYCLING_QUALITY_FACTOR", "1.0")
            ),
            allocation_factor=Decimal(
                os.getenv("GL_EOL_RECYCLING_ALLOCATION_FACTOR", "0.5")
            ),
        )


# =============================================================================
# SECTION 10: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Compliance configuration for end-of-life treatment regulatory checking.

    Enables validation against 7 regulatory frameworks and double-counting
    prevention against Cat 1 (purchased goods), Cat 5 (waste generated),
    and Scope 1 (direct emissions from owned waste treatment).

    Attributes:
        compliance_frameworks: Enabled frameworks (GL_EOL_COMPLIANCE_FRAMEWORKS)
        strict_mode: Enforce strict compliance (GL_EOL_COMPLIANCE_STRICT_MODE)
        double_counting_check: Check double counting (GL_EOL_COMPLIANCE_DC_CHECK)
        boundary_enforcement: Enforce Scope 3 boundary (GL_EOL_COMPLIANCE_BOUNDARY)
        esrs_e5_circular_economy: Enable ESRS E5 circular economy (GL_EOL_COMPLIANCE_ESRS_E5)

    Example:
        >>> compliance = ComplianceConfig(
        ...     strict_mode=True,
        ...     double_counting_check=True,
        ...     esrs_e5_circular_economy=True
        ... )
        >>> compliance.get_frameworks()
        ['GHG_PROTOCOL_SCOPE3', 'ISO_14064', 'CSRD_ESRS_E1', 'CDP', 'SBTI', 'EU_WASTE_FRAMEWORK', 'EPA']
    """

    compliance_frameworks: str = (
        "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,EU_WASTE_FRAMEWORK,EPA"
    )
    strict_mode: bool = True
    double_counting_check: bool = True
    boundary_enforcement: bool = True
    esrs_e5_circular_economy: bool = True

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL_SCOPE3", "ISO_14064", "CSRD_ESRS_E1",
            "CDP", "SBTI", "EU_WASTE_FRAMEWORK", "EPA",
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
            "esrs_e5_circular_economy": self.esrs_e5_circular_economy,
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
                "GL_EOL_COMPLIANCE_FRAMEWORKS",
                "GHG_PROTOCOL_SCOPE3,ISO_14064,CSRD_ESRS_E1,CDP,SBTI,EU_WASTE_FRAMEWORK,EPA",
            ),
            strict_mode=os.getenv("GL_EOL_COMPLIANCE_STRICT_MODE", "true").lower() == "true",
            double_counting_check=os.getenv(
                "GL_EOL_COMPLIANCE_DC_CHECK", "true"
            ).lower() == "true",
            boundary_enforcement=os.getenv(
                "GL_EOL_COMPLIANCE_BOUNDARY", "true"
            ).lower() == "true",
            esrs_e5_circular_economy=os.getenv(
                "GL_EOL_COMPLIANCE_ESRS_E5", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 11: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Provenance tracking configuration for audit trails.

    Controls SHA-256 chain hashing, Merkle tree aggregation, and
    provenance chain management for zero-hallucination guarantees.

    Attributes:
        hash_algorithm: Hash algorithm for provenance (GL_EOL_PROVENANCE_ALGORITHM)
        chain_enabled: Enable chain hashing (GL_EOL_PROVENANCE_CHAIN)
        merkle_enabled: Enable Merkle tree aggregation (GL_EOL_PROVENANCE_MERKLE)
        store_raw_data: Store raw input/output data alongside hashes (GL_EOL_PROVENANCE_STORE_RAW)

    Example:
        >>> provenance = ProvenanceConfig(
        ...     hash_algorithm="sha256",
        ...     chain_enabled=True,
        ...     merkle_enabled=True
        ... )
        >>> provenance.hash_algorithm
        'sha256'
    """

    hash_algorithm: str = "sha256"
    chain_enabled: bool = True
    merkle_enabled: bool = True
    store_raw_data: bool = False

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
            "store_raw_data": self.store_raw_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            hash_algorithm=os.getenv("GL_EOL_PROVENANCE_ALGORITHM", "sha256"),
            chain_enabled=os.getenv("GL_EOL_PROVENANCE_CHAIN", "true").lower() == "true",
            merkle_enabled=os.getenv("GL_EOL_PROVENANCE_MERKLE", "true").lower() == "true",
            store_raw_data=os.getenv("GL_EOL_PROVENANCE_STORE_RAW", "false").lower() == "true",
        )


# =============================================================================
# SECTION 12: UNCERTAINTY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class UncertaintyConfig:
    """
    Uncertainty quantification configuration.

    Controls Monte Carlo simulation, confidence intervals, and parametric
    uncertainty for emissions estimates.

    Attributes:
        method: Uncertainty method (GL_EOL_UNCERTAINTY_METHOD)
        iterations: Monte Carlo iterations (GL_EOL_UNCERTAINTY_ITERATIONS)
        confidence: Confidence level (GL_EOL_UNCERTAINTY_CONFIDENCE)
        include_parameter: Include parameter uncertainty (GL_EOL_UNCERTAINTY_PARAMETER)
        include_model: Include model uncertainty (GL_EOL_UNCERTAINTY_MODEL)

    Example:
        >>> uncertainty = UncertaintyConfig(
        ...     method="MONTE_CARLO",
        ...     iterations=10000,
        ...     confidence=Decimal("0.95")
        ... )
        >>> uncertainty.method
        'MONTE_CARLO'
    """

    method: str = "MONTE_CARLO"
    iterations: int = 10000
    confidence: Decimal = Decimal("0.95")
    include_parameter: bool = True
    include_model: bool = False

    def validate(self) -> None:
        """
        Validate uncertainty configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_methods = {"MONTE_CARLO", "BOOTSTRAP", "BAYESIAN", "IPCC_DEFAULT", "NONE"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}'. Must be one of {valid_methods}"
            )

        if self.iterations < 100 or self.iterations > 1000000:
            raise ValueError("iterations must be between 100 and 1000000")

        if self.confidence < Decimal("0") or self.confidence > Decimal("1"):
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "iterations": self.iterations,
            "confidence": str(self.confidence),
            "include_parameter": self.include_parameter,
            "include_model": self.include_model,
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
        return cls(
            method=os.getenv("GL_EOL_UNCERTAINTY_METHOD", "MONTE_CARLO"),
            iterations=int(os.getenv("GL_EOL_UNCERTAINTY_ITERATIONS", "10000")),
            confidence=Decimal(os.getenv("GL_EOL_UNCERTAINTY_CONFIDENCE", "0.95")),
            include_parameter=os.getenv(
                "GL_EOL_UNCERTAINTY_PARAMETER", "true"
            ).lower() == "true",
            include_model=os.getenv(
                "GL_EOL_UNCERTAINTY_MODEL", "false"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 13: DQI CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DQIConfig:
    """
    Data Quality Indicator (DQI) configuration.

    Configures the 5-dimension data quality scoring system aligned with
    GHG Protocol Scope 3 data quality guidance.

    Attributes:
        min_score: Minimum acceptable DQI composite score (GL_EOL_DQI_MIN_SCORE)
        weight_reliability: Weight for reliability dimension (GL_EOL_DQI_W_RELIABILITY)
        weight_completeness: Weight for completeness dimension (GL_EOL_DQI_W_COMPLETENESS)
        weight_temporal: Weight for temporal correlation (GL_EOL_DQI_W_TEMPORAL)
        weight_geographical: Weight for geographical correlation (GL_EOL_DQI_W_GEOGRAPHICAL)
        weight_technological: Weight for technological correlation (GL_EOL_DQI_W_TECHNOLOGICAL)

    Example:
        >>> dqi = DQIConfig(
        ...     min_score=Decimal("2.0"),
        ...     weight_reliability=Decimal("0.25"),
        ...     weight_completeness=Decimal("0.25"),
        ...     weight_temporal=Decimal("0.20"),
        ...     weight_geographical=Decimal("0.15"),
        ...     weight_technological=Decimal("0.15")
        ... )
        >>> dqi.min_score
        Decimal('2.0')
    """

    min_score: Decimal = Decimal("2.0")
    weight_reliability: Decimal = Decimal("0.25")
    weight_completeness: Decimal = Decimal("0.25")
    weight_temporal: Decimal = Decimal("0.20")
    weight_geographical: Decimal = Decimal("0.15")
    weight_technological: Decimal = Decimal("0.15")

    def validate(self) -> None:
        """
        Validate DQI configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.min_score < Decimal("0") or self.min_score > Decimal("5"):
            raise ValueError("min_score must be between 0 and 5")

        weights = [
            self.weight_reliability,
            self.weight_completeness,
            self.weight_temporal,
            self.weight_geographical,
            self.weight_technological,
        ]

        for w in weights:
            if w < Decimal("0") or w > Decimal("1"):
                raise ValueError("All DQI weights must be between 0 and 1")

        total = sum(weights)
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"DQI weights must sum to 1.0 (currently {total})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_score": str(self.min_score),
            "weight_reliability": str(self.weight_reliability),
            "weight_completeness": str(self.weight_completeness),
            "weight_temporal": str(self.weight_temporal),
            "weight_geographical": str(self.weight_geographical),
            "weight_technological": str(self.weight_technological),
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
            min_score=Decimal(os.getenv("GL_EOL_DQI_MIN_SCORE", "2.0")),
            weight_reliability=Decimal(os.getenv("GL_EOL_DQI_W_RELIABILITY", "0.25")),
            weight_completeness=Decimal(os.getenv("GL_EOL_DQI_W_COMPLETENESS", "0.25")),
            weight_temporal=Decimal(os.getenv("GL_EOL_DQI_W_TEMPORAL", "0.20")),
            weight_geographical=Decimal(os.getenv("GL_EOL_DQI_W_GEOGRAPHICAL", "0.15")),
            weight_technological=Decimal(os.getenv("GL_EOL_DQI_W_TECHNOLOGICAL", "0.15")),
        )


# =============================================================================
# SECTION 14: PIPELINE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline orchestration configuration.

    Controls batch processing, timeouts, retries, and parallelism
    for the end-of-life treatment calculation pipeline.

    Attributes:
        batch_size: Default batch size (GL_EOL_PIPELINE_BATCH_SIZE)
        timeout: Pipeline timeout in seconds (GL_EOL_PIPELINE_TIMEOUT)
        max_retries: Maximum retry attempts (GL_EOL_PIPELINE_MAX_RETRIES)
        parallel_execution: Enable parallel execution (GL_EOL_PIPELINE_PARALLEL)
        worker_threads: Number of worker threads (GL_EOL_PIPELINE_WORKERS)

    Example:
        >>> pipeline = PipelineConfig(
        ...     batch_size=500,
        ...     timeout=300,
        ...     max_retries=3,
        ...     parallel_execution=True,
        ...     worker_threads=4
        ... )
        >>> pipeline.batch_size
        500
    """

    batch_size: int = 500
    timeout: int = 300
    max_retries: int = 3
    parallel_execution: bool = True
    worker_threads: int = 4

    def validate(self) -> None:
        """
        Validate pipeline configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.batch_size < 1 or self.batch_size > 10000:
            raise ValueError("batch_size must be between 1 and 10000")

        if self.timeout < 1 or self.timeout > 3600:
            raise ValueError("timeout must be between 1 and 3600 seconds")

        if self.max_retries < 0 or self.max_retries > 10:
            raise ValueError("max_retries must be between 0 and 10")

        if self.worker_threads < 1 or self.worker_threads > 64:
            raise ValueError("worker_threads must be between 1 and 64")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "parallel_execution": self.parallel_execution,
            "worker_threads": self.worker_threads,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load from environment variables."""
        return cls(
            batch_size=int(os.getenv("GL_EOL_PIPELINE_BATCH_SIZE", "500")),
            timeout=int(os.getenv("GL_EOL_PIPELINE_TIMEOUT", "300")),
            max_retries=int(os.getenv("GL_EOL_PIPELINE_MAX_RETRIES", "3")),
            parallel_execution=os.getenv(
                "GL_EOL_PIPELINE_PARALLEL", "true"
            ).lower() == "true",
            worker_threads=int(os.getenv("GL_EOL_PIPELINE_WORKERS", "4")),
        )


# =============================================================================
# SECTION 15: CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheConfig:
    """
    Cache configuration for end-of-life treatment agent.

    Controls Redis/in-memory caching for emission factor lookups,
    calculation results, and treatment split data.

    Attributes:
        enabled: Enable caching (GL_EOL_CACHE_ENABLED)
        ttl: Default cache TTL in seconds (GL_EOL_CACHE_TTL)
        max_size: Maximum cache entries (GL_EOL_CACHE_MAX_SIZE)
        key_prefix: Cache key prefix (GL_EOL_CACHE_KEY_PREFIX)
        cache_ef_lookups: Cache emission factor lookups (GL_EOL_CACHE_EF)
        cache_calculations: Cache calculation results (GL_EOL_CACHE_CALCULATIONS)

    Example:
        >>> cache = CacheConfig(
        ...     enabled=True,
        ...     ttl=3600,
        ...     max_size=10000,
        ...     key_prefix="gl_eol:"
        ... )
        >>> cache.key_prefix
        'gl_eol:'
    """

    enabled: bool = True
    ttl: int = 3600
    max_size: int = 10000
    key_prefix: str = "gl_eol:"
    cache_ef_lookups: bool = True
    cache_calculations: bool = True

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

        if not self.key_prefix:
            raise ValueError("key_prefix cannot be empty")

        if not self.key_prefix.endswith(":"):
            raise ValueError("key_prefix must end with ':'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ttl": self.ttl,
            "max_size": self.max_size,
            "key_prefix": self.key_prefix,
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
            enabled=os.getenv("GL_EOL_CACHE_ENABLED", "true").lower() == "true",
            ttl=int(os.getenv("GL_EOL_CACHE_TTL", "3600")),
            max_size=int(os.getenv("GL_EOL_CACHE_MAX_SIZE", "10000")),
            key_prefix=os.getenv("GL_EOL_CACHE_KEY_PREFIX", "gl_eol:"),
            cache_ef_lookups=os.getenv("GL_EOL_CACHE_EF", "true").lower() == "true",
            cache_calculations=os.getenv(
                "GL_EOL_CACHE_CALCULATIONS", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 16: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Prometheus metrics configuration.

    Controls metrics collection prefix, histogram buckets, and
    metrics enablement.

    Attributes:
        enabled: Enable metrics collection (GL_EOL_METRICS_ENABLED)
        prefix: Metrics name prefix (GL_EOL_METRICS_PREFIX)
        histogram_buckets: Histogram bucket boundaries (GL_EOL_METRICS_BUCKETS)

    Example:
        >>> metrics = MetricsConfig(
        ...     enabled=True,
        ...     prefix="gl_eol_",
        ...     histogram_buckets="0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
        ... )
        >>> metrics.get_buckets()
        [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    """

    enabled: bool = True
    prefix: str = "gl_eol_"
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
            enabled=os.getenv("GL_EOL_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_EOL_METRICS_PREFIX", "gl_eol_"),
            histogram_buckets=os.getenv(
                "GL_EOL_METRICS_BUCKETS", "0.01,0.05,0.1,0.25,0.5,1.0,2.5,5.0,10.0"
            ),
        )


# =============================================================================
# SECTION 17: SECURITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class SecurityConfig:
    """
    Security configuration for tenant isolation and audit.

    Controls multi-tenant isolation, audit logging, and rate limiting
    for the end-of-life treatment agent.

    Attributes:
        tenant_isolation: Enable tenant isolation (GL_EOL_SECURITY_TENANT_ISOLATION)
        audit_enabled: Enable audit logging (GL_EOL_SECURITY_AUDIT)
        rate_limiting: Enable rate limiting (GL_EOL_SECURITY_RATE_LIMITING)
        max_requests_per_minute: Max requests per minute per tenant (GL_EOL_SECURITY_RPM)

    Example:
        >>> security = SecurityConfig(
        ...     tenant_isolation=True,
        ...     audit_enabled=True,
        ...     rate_limiting=True,
        ...     max_requests_per_minute=100
        ... )
        >>> security.tenant_isolation
        True
    """

    tenant_isolation: bool = True
    audit_enabled: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 100

    def validate(self) -> None:
        """
        Validate security configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.max_requests_per_minute < 1 or self.max_requests_per_minute > 10000:
            raise ValueError("max_requests_per_minute must be between 1 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_isolation": self.tenant_isolation,
            "audit_enabled": self.audit_enabled,
            "rate_limiting": self.rate_limiting,
            "max_requests_per_minute": self.max_requests_per_minute,
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
                "GL_EOL_SECURITY_TENANT_ISOLATION", "true"
            ).lower() == "true",
            audit_enabled=os.getenv(
                "GL_EOL_SECURITY_AUDIT", "true"
            ).lower() == "true",
            rate_limiting=os.getenv(
                "GL_EOL_SECURITY_RATE_LIMITING", "true"
            ).lower() == "true",
            max_requests_per_minute=int(os.getenv("GL_EOL_SECURITY_RPM", "100")),
        )


# =============================================================================
# SECTION 18: CIRCULARITY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CircularityConfig:
    """
    Circular economy metrics configuration.

    Controls tracking of recycling rate, waste diversion, and circularity
    index for ESRS E5 circular economy disclosure and product stewardship.

    Attributes:
        track_recycling_rate: Track recycling rate (GL_EOL_CIRC_RECYCLING_RATE)
        track_diversion: Track waste diversion rate (GL_EOL_CIRC_DIVERSION)
        track_circularity_index: Track Material Circularity Index (GL_EOL_CIRC_INDEX)
        waste_hierarchy_scoring: Enable waste hierarchy scoring (GL_EOL_CIRC_HIERARCHY)
        target_recycling_rate: Target recycling rate (GL_EOL_CIRC_TARGET_RECYCLING)
        target_diversion_rate: Target waste diversion rate (GL_EOL_CIRC_TARGET_DIVERSION)

    Example:
        >>> circ = CircularityConfig(
        ...     track_recycling_rate=True,
        ...     track_diversion=True,
        ...     track_circularity_index=True,
        ...     waste_hierarchy_scoring=True
        ... )
        >>> circ.track_circularity_index
        True
    """

    track_recycling_rate: bool = True
    track_diversion: bool = True
    track_circularity_index: bool = True
    waste_hierarchy_scoring: bool = True
    target_recycling_rate: Decimal = Decimal("0.65")
    target_diversion_rate: Decimal = Decimal("0.90")

    def validate(self) -> None:
        """
        Validate circularity configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.target_recycling_rate < Decimal("0") or self.target_recycling_rate > Decimal("1"):
            raise ValueError("target_recycling_rate must be between 0 and 1")

        if self.target_diversion_rate < Decimal("0") or self.target_diversion_rate > Decimal("1"):
            raise ValueError("target_diversion_rate must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "track_recycling_rate": self.track_recycling_rate,
            "track_diversion": self.track_diversion,
            "track_circularity_index": self.track_circularity_index,
            "waste_hierarchy_scoring": self.waste_hierarchy_scoring,
            "target_recycling_rate": str(self.target_recycling_rate),
            "target_diversion_rate": str(self.target_diversion_rate),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircularityConfig":
        """Create from dictionary."""
        data_copy = data.copy()
        for key in ["target_recycling_rate", "target_diversion_rate"]:
            if key in data_copy:
                data_copy[key] = Decimal(data_copy[key])
        return cls(**data_copy)

    @classmethod
    def from_env(cls) -> "CircularityConfig":
        """Load from environment variables."""
        return cls(
            track_recycling_rate=os.getenv(
                "GL_EOL_CIRC_RECYCLING_RATE", "true"
            ).lower() == "true",
            track_diversion=os.getenv(
                "GL_EOL_CIRC_DIVERSION", "true"
            ).lower() == "true",
            track_circularity_index=os.getenv(
                "GL_EOL_CIRC_INDEX", "true"
            ).lower() == "true",
            waste_hierarchy_scoring=os.getenv(
                "GL_EOL_CIRC_HIERARCHY", "true"
            ).lower() == "true",
            target_recycling_rate=Decimal(
                os.getenv("GL_EOL_CIRC_TARGET_RECYCLING", "0.65")
            ),
            target_diversion_rate=Decimal(
                os.getenv("GL_EOL_CIRC_TARGET_DIVERSION", "0.90")
            ),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


class EndOfLifeTreatmentConfig:
    """
    Master configuration class for End-of-Life Treatment of Sold Products agent.

    Aggregates all 18 configuration sections and provides a unified interface
    for accessing configuration values. Implements thread-safe singleton pattern.

    Attributes:
        general: General agent configuration
        database: Database connection configuration
        waste_type: Waste type and treatment pathway configuration
        average_data: Average-data calculation method configuration
        producer_specific: Producer-specific calculation method configuration
        hybrid: Hybrid calculation method configuration
        landfill: Landfill emissions configuration
        incineration: Incineration emissions configuration
        recycling: Recycling emissions configuration
        compliance: Compliance and regulatory configuration
        provenance: Provenance tracking configuration
        uncertainty: Uncertainty quantification configuration
        dqi: Data Quality Indicator configuration
        pipeline: Pipeline orchestration configuration
        cache: Cache configuration
        metrics: Metrics configuration
        security: Security configuration
        circularity: Circular economy metrics configuration

    Example:
        >>> config = EndOfLifeTreatmentConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-S3-012'
        >>> config.landfill.default_docf
        Decimal('0.5')
        >>> config.validate_all()
    """

    def __init__(
        self,
        general: GeneralConfig,
        database: DatabaseConfig,
        waste_type: WasteTypeConfig,
        average_data: AverageDataConfig,
        producer_specific: ProducerSpecificConfig,
        hybrid: HybridConfig,
        landfill: LandfillConfig,
        incineration: IncinerationConfig,
        recycling: RecyclingConfig,
        compliance: ComplianceConfig,
        provenance: ProvenanceConfig,
        uncertainty: UncertaintyConfig,
        dqi: DQIConfig,
        pipeline: PipelineConfig,
        cache: CacheConfig,
        metrics: MetricsConfig,
        security: SecurityConfig,
        circularity: CircularityConfig,
    ):
        """
        Initialize master configuration.

        Args:
            general: General agent configuration
            database: Database connection configuration
            waste_type: Waste type and treatment pathway configuration
            average_data: Average-data calculation method configuration
            producer_specific: Producer-specific calculation method configuration
            hybrid: Hybrid calculation method configuration
            landfill: Landfill emissions configuration
            incineration: Incineration emissions configuration
            recycling: Recycling emissions configuration
            compliance: Compliance and regulatory configuration
            provenance: Provenance tracking configuration
            uncertainty: Uncertainty quantification configuration
            dqi: Data Quality Indicator configuration
            pipeline: Pipeline orchestration configuration
            cache: Cache configuration
            metrics: Metrics configuration
            security: Security configuration
            circularity: Circular economy metrics configuration
        """
        self.general = general
        self.database = database
        self.waste_type = waste_type
        self.average_data = average_data
        self.producer_specific = producer_specific
        self.hybrid = hybrid
        self.landfill = landfill
        self.incineration = incineration
        self.recycling = recycling
        self.compliance = compliance
        self.provenance = provenance
        self.uncertainty = uncertainty
        self.dqi = dqi
        self.pipeline = pipeline
        self.cache = cache
        self.metrics = metrics
        self.security = security
        self.circularity = circularity

    def validate_all(self) -> None:
        """
        Validate all configuration sections and run cross-validation.

        Raises:
            ValueError: If any configuration section is invalid
        """
        self.general.validate()
        self.database.validate()
        self.waste_type.validate()
        self.average_data.validate()
        self.producer_specific.validate()
        self.hybrid.validate()
        self.landfill.validate()
        self.incineration.validate()
        self.recycling.validate()
        self.compliance.validate()
        self.provenance.validate()
        self.uncertainty.validate()
        self.dqi.validate()
        self.pipeline.validate()
        self.cache.validate()
        self.metrics.validate()
        self.security.validate()
        self.circularity.validate()

        # Cross-validation checks
        self._cross_validate()

    def _cross_validate(self) -> None:
        """
        Run cross-validation checks between configuration sections.

        Raises:
            ValueError: If cross-validation fails
        """
        # Pipeline batch_size should not exceed general max_batch_size
        if self.pipeline.batch_size > self.general.max_batch_size:
            raise ValueError(
                f"pipeline.batch_size ({self.pipeline.batch_size}) "
                f"exceeds general.max_batch_size ({self.general.max_batch_size})"
            )

        # If recycling is disabled, recycling approach is irrelevant but not invalid
        # If landfill is disabled but landfill config has non-default values, warn
        if not self.waste_type.enable_landfill:
            logger.info(
                "Landfill treatment is disabled; landfill config parameters "
                "will not be used in calculations"
            )

        # Hybrid waterfall should only reference enabled methods
        # Producer-specific requires at least one validation method
        if self.producer_specific.require_verification and not self.producer_specific.epd_validation:
            logger.warning(
                "Producer-specific requires verification but EPD validation "
                "is disabled; only manual verification will be accepted"
            )

        # ESRS E5 circular economy requires circularity tracking
        if self.compliance.esrs_e5_circular_economy:
            if not (self.circularity.track_recycling_rate or self.circularity.track_diversion):
                raise ValueError(
                    "ESRS E5 circular economy is enabled but circularity "
                    "tracking (recycling_rate or diversion) is disabled"
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire configuration to dictionary.

        Returns:
            Dictionary representation of all configuration sections
        """
        return {
            "general": self.general.to_dict(),
            "database": self.database.to_dict(),
            "waste_type": self.waste_type.to_dict(),
            "average_data": self.average_data.to_dict(),
            "producer_specific": self.producer_specific.to_dict(),
            "hybrid": self.hybrid.to_dict(),
            "landfill": self.landfill.to_dict(),
            "incineration": self.incineration.to_dict(),
            "recycling": self.recycling.to_dict(),
            "compliance": self.compliance.to_dict(),
            "provenance": self.provenance.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "dqi": self.dqi.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "cache": self.cache.to_dict(),
            "metrics": self.metrics.to_dict(),
            "security": self.security.to_dict(),
            "circularity": self.circularity.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndOfLifeTreatmentConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Dictionary containing all configuration sections

        Returns:
            EndOfLifeTreatmentConfig instance
        """
        return cls(
            general=GeneralConfig.from_dict(data["general"]),
            database=DatabaseConfig.from_dict(data["database"]),
            waste_type=WasteTypeConfig.from_dict(data["waste_type"]),
            average_data=AverageDataConfig.from_dict(data["average_data"]),
            producer_specific=ProducerSpecificConfig.from_dict(data["producer_specific"]),
            hybrid=HybridConfig.from_dict(data["hybrid"]),
            landfill=LandfillConfig.from_dict(data["landfill"]),
            incineration=IncinerationConfig.from_dict(data["incineration"]),
            recycling=RecyclingConfig.from_dict(data["recycling"]),
            compliance=ComplianceConfig.from_dict(data["compliance"]),
            provenance=ProvenanceConfig.from_dict(data["provenance"]),
            uncertainty=UncertaintyConfig.from_dict(data["uncertainty"]),
            dqi=DQIConfig.from_dict(data["dqi"]),
            pipeline=PipelineConfig.from_dict(data["pipeline"]),
            cache=CacheConfig.from_dict(data["cache"]),
            metrics=MetricsConfig.from_dict(data["metrics"]),
            security=SecurityConfig.from_dict(data["security"]),
            circularity=CircularityConfig.from_dict(data["circularity"]),
        )

    @classmethod
    def from_env(cls) -> "EndOfLifeTreatmentConfig":
        """
        Load configuration from environment variables.

        Returns:
            EndOfLifeTreatmentConfig instance loaded from environment
        """
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            waste_type=WasteTypeConfig.from_env(),
            average_data=AverageDataConfig.from_env(),
            producer_specific=ProducerSpecificConfig.from_env(),
            hybrid=HybridConfig.from_env(),
            landfill=LandfillConfig.from_env(),
            incineration=IncinerationConfig.from_env(),
            recycling=RecyclingConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
            uncertainty=UncertaintyConfig.from_env(),
            dqi=DQIConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            cache=CacheConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            security=SecurityConfig.from_env(),
            circularity=CircularityConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[EndOfLifeTreatmentConfig] = None
_config_lock = threading.RLock()


def get_config() -> EndOfLifeTreatmentConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization. The first call loads from environment
    variables and validates. Subsequent calls return the cached instance.

    Returns:
        EndOfLifeTreatmentConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-S3-012'

    Thread Safety:
        Uses double-checked locking pattern with RLock for thread safety.
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            # Double-checked locking pattern
            if _config_instance is None:
                _config_instance = EndOfLifeTreatmentConfig.from_env()
                _config_instance.validate_all()

    return _config_instance


def set_config(config: EndOfLifeTreatmentConfig) -> None:
    """
    Set the singleton configuration instance.

    Validates the configuration before setting. Primarily for testing
    or custom initialization scenarios.

    Args:
        config: EndOfLifeTreatmentConfig instance to set as singleton

    Example:
        >>> custom_config = EndOfLifeTreatmentConfig.from_dict({...})
        >>> set_config(custom_config)

    Thread Safety:
        Thread-safe with RLock protection.
    """
    global _config_instance

    with _config_lock:
        config.validate_all()
        _config_instance = config


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Clears the cached singleton, forcing the next get_config() call
    to reload from environment variables. Primarily for testing.

    Example:
        >>> reset_config()
        >>> config = get_config()  # Reloads from environment

    Thread Safety:
        Thread-safe with RLock protection.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None


def validate_config(config: EndOfLifeTreatmentConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Unlike validate_all() which raises on first error, this function
    collects all errors across all sections.

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
        ("waste_type", config.waste_type),
        ("average_data", config.average_data),
        ("producer_specific", config.producer_specific),
        ("hybrid", config.hybrid),
        ("landfill", config.landfill),
        ("incineration", config.incineration),
        ("recycling", config.recycling),
        ("compliance", config.compliance),
        ("provenance", config.provenance),
        ("uncertainty", config.uncertainty),
        ("dqi", config.dqi),
        ("pipeline", config.pipeline),
        ("cache", config.cache),
        ("metrics", config.metrics),
        ("security", config.security),
        ("circularity", config.circularity),
    ]

    for section_name, section in sections:
        try:
            section.validate()
        except ValueError as e:
            errors.append(f"{section_name}: {str(e)}")

    # Cross-validation
    try:
        config._cross_validate()
    except ValueError as e:
        errors.append(f"cross_validation: {str(e)}")

    return errors


def print_config(config: EndOfLifeTreatmentConfig) -> None:
    """
    Print configuration in human-readable format.

    Redacts sensitive fields (database URLs, credentials).

    Args:
        config: Configuration instance to print

    Example:
        >>> config = get_config()
        >>> print_config(config)
    """
    print("===== End-of-Life Treatment of Sold Products Configuration =====")

    section_names = [
        ("GENERAL", config.general),
        ("DATABASE", config.database),
        ("WASTE_TYPE", config.waste_type),
        ("AVERAGE_DATA", config.average_data),
        ("PRODUCER_SPECIFIC", config.producer_specific),
        ("HYBRID", config.hybrid),
        ("LANDFILL", config.landfill),
        ("INCINERATION", config.incineration),
        ("RECYCLING", config.recycling),
        ("COMPLIANCE", config.compliance),
        ("PROVENANCE", config.provenance),
        ("UNCERTAINTY", config.uncertainty),
        ("DQI", config.dqi),
        ("PIPELINE", config.pipeline),
        ("CACHE", config.cache),
        ("METRICS", config.metrics),
        ("SECURITY", config.security),
        ("CIRCULARITY", config.circularity),
    ]

    sensitive_keys = {"database_url", "redis_url"}

    for name, section in section_names:
        print(f"\n[{name}]")
        for key, value in section.to_dict().items():
            if key in sensitive_keys:
                print(f"  {key}: [REDACTED]")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 64)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration classes
    "GeneralConfig",
    "DatabaseConfig",
    "WasteTypeConfig",
    "AverageDataConfig",
    "ProducerSpecificConfig",
    "HybridConfig",
    "LandfillConfig",
    "IncinerationConfig",
    "RecyclingConfig",
    "ComplianceConfig",
    "ProvenanceConfig",
    "UncertaintyConfig",
    "DQIConfig",
    "PipelineConfig",
    "CacheConfig",
    "MetricsConfig",
    "SecurityConfig",
    "CircularityConfig",
    "EndOfLifeTreatmentConfig",
    # Singleton functions
    "get_config",
    "set_config",
    "reset_config",
    # Utility functions
    "validate_config",
    "print_config",
]
