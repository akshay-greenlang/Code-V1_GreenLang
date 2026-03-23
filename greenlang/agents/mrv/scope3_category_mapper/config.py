# -*- coding: utf-8 -*-
"""
Scope 3 Category Mapper Configuration - AGENT-MRV-029

Thread-safe singleton configuration for GL-MRV-X-040.
All environment variables prefixed with GL_SCM_.

This module provides comprehensive configuration management for the
Scope 3 Category Mapper agent, supporting:
- Classification engine settings (confidence thresholds per method)
- Boundary determination (consolidation, Incoterms, lease rules)
- Completeness screening (materiality, industry benchmarks)
- Double-counting prevention (10 DC-SCM rules)
- Compliance assessment (8 frameworks)
- Database and metrics configuration
- Provenance tracking settings

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-X-040'
    >>> config.classification.naics_confidence
    0.95

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_SCM_ prefix. See individual config sections for specific variables.
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
    General configuration for Scope 3 Category Mapper agent.

    Attributes:
        enabled: Master switch for the agent (GL_SCM_ENABLED)
        debug: Enable debug mode with verbose logging (GL_SCM_DEBUG)
        log_level: Logging level (GL_SCM_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_SCM_AGENT_ID)
        agent_component: Agent component identifier (GL_SCM_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_SCM_VERSION)
        api_prefix: API route prefix (GL_SCM_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_SCM_MAX_BATCH_SIZE)
        default_currency: Default currency ISO 4217 (GL_SCM_DEFAULT_CURRENCY)
        default_gwp: Default GWP assessment report version (GL_SCM_DEFAULT_GWP)

    Example:
        >>> general = GeneralConfig()
        >>> general.agent_id
        'GL-MRV-X-040'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-X-040"
    agent_component: str = "AGENT-MRV-029"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/scope3-category-mapper"
    max_batch_size: int = 50000
    default_currency: str = "USD"
    default_gwp: str = "AR5"

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
        if self.max_batch_size < 1 or self.max_batch_size > 500000:
            raise ValueError("max_batch_size must be between 1 and 500000")
        valid_gwp = {"SAR", "AR4", "AR5", "AR6"}
        if self.default_gwp not in valid_gwp:
            raise ValueError(
                f"Invalid default_gwp '{self.default_gwp}'. Must be one of {valid_gwp}"
            )
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
            "default_currency": self.default_currency,
            "default_gwp": self.default_gwp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_SCM_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_SCM_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_SCM_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_SCM_AGENT_ID", "GL-MRV-X-040"),
            agent_component=os.getenv("GL_SCM_AGENT_COMPONENT", "AGENT-MRV-029"),
            version=os.getenv("GL_SCM_VERSION", "1.0.0"),
            api_prefix=os.getenv("GL_SCM_API_PREFIX", "/api/v1/scope3-category-mapper"),
            max_batch_size=int(os.getenv("GL_SCM_MAX_BATCH_SIZE", "50000")),
            default_currency=os.getenv("GL_SCM_DEFAULT_CURRENCY", "USD"),
            default_gwp=os.getenv("GL_SCM_DEFAULT_GWP", "AR5"),
        )


# =============================================================================
# SECTION 2: CLASSIFICATION CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ClassificationConfig:
    """
    Classification engine configuration.

    Controls confidence thresholds per classification method, default
    category fallback, and multi-category split settings.

    Attributes:
        min_confidence_threshold: Minimum confidence to accept a mapping (GL_SCM_MIN_CONFIDENCE)
        naics_confidence: Confidence score for NAICS code matches (GL_SCM_NAICS_CONFIDENCE)
        isic_confidence: Confidence score for ISIC code matches (GL_SCM_ISIC_CONFIDENCE)
        gl_account_confidence: Confidence for GL account matches (GL_SCM_GL_ACCOUNT_CONFIDENCE)
        keyword_confidence: Confidence for keyword matches (GL_SCM_KEYWORD_CONFIDENCE)
        default_category: Fallback category when classification is uncertain (GL_SCM_DEFAULT_CATEGORY)
        enable_multi_category_split: Allow splitting records across categories (GL_SCM_ENABLE_SPLIT)
        max_split_categories: Maximum categories per split (GL_SCM_MAX_SPLIT_CATEGORIES)

    Example:
        >>> cls_config = ClassificationConfig()
        >>> cls_config.naics_confidence
        0.95
    """

    min_confidence_threshold: float = 0.3
    naics_confidence: float = 0.95
    isic_confidence: float = 0.90
    gl_account_confidence: float = 0.85
    keyword_confidence: float = 0.40
    default_category: str = "1_purchased_goods_services"
    enable_multi_category_split: bool = True
    max_split_categories: int = 3

    def validate(self) -> None:
        """Validate classification configuration values."""
        if self.min_confidence_threshold < 0.0 or self.min_confidence_threshold > 1.0:
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        for name, val in [
            ("naics_confidence", self.naics_confidence),
            ("isic_confidence", self.isic_confidence),
            ("gl_account_confidence", self.gl_account_confidence),
            ("keyword_confidence", self.keyword_confidence),
        ]:
            if val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not self.default_category:
            raise ValueError("default_category cannot be empty")
        if self.max_split_categories < 1 or self.max_split_categories > 15:
            raise ValueError("max_split_categories must be between 1 and 15")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "naics_confidence": self.naics_confidence,
            "isic_confidence": self.isic_confidence,
            "gl_account_confidence": self.gl_account_confidence,
            "keyword_confidence": self.keyword_confidence,
            "default_category": self.default_category,
            "enable_multi_category_split": self.enable_multi_category_split,
            "max_split_categories": self.max_split_categories,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ClassificationConfig":
        """Load from environment variables."""
        return cls(
            min_confidence_threshold=float(
                os.getenv("GL_SCM_MIN_CONFIDENCE", "0.3")
            ),
            naics_confidence=float(os.getenv("GL_SCM_NAICS_CONFIDENCE", "0.95")),
            isic_confidence=float(os.getenv("GL_SCM_ISIC_CONFIDENCE", "0.90")),
            gl_account_confidence=float(
                os.getenv("GL_SCM_GL_ACCOUNT_CONFIDENCE", "0.85")
            ),
            keyword_confidence=float(os.getenv("GL_SCM_KEYWORD_CONFIDENCE", "0.40")),
            default_category=os.getenv(
                "GL_SCM_DEFAULT_CATEGORY", "1_purchased_goods_services"
            ),
            enable_multi_category_split=os.getenv(
                "GL_SCM_ENABLE_SPLIT", "true"
            ).lower() == "true",
            max_split_categories=int(
                os.getenv("GL_SCM_MAX_SPLIT_CATEGORIES", "3")
            ),
        )


# =============================================================================
# SECTION 3: BOUNDARY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class BoundaryConfig:
    """
    Organizational and operational boundary configuration.

    Controls how the mapper determines whether data falls within the
    reporting company's Scope 3 boundary, including consolidation
    approach, CAPEX thresholds, Incoterms, and lease rules.

    Attributes:
        default_consolidation: Default consolidation approach (GL_SCM_CONSOLIDATION)
        capex_threshold: CAPEX threshold for Cat 1 vs Cat 2 boundary (GL_SCM_CAPEX_THRESHOLD)
        default_incoterm: Default Incoterm when not specified (GL_SCM_DEFAULT_INCOTERM)
        lease_scope_boundary_months: Short-term lease threshold in months (GL_SCM_LEASE_BOUNDARY_MONTHS)

    Example:
        >>> boundary = BoundaryConfig()
        >>> boundary.default_consolidation
        'operational_control'
    """

    default_consolidation: str = "operational_control"
    capex_threshold: float = 5000.0
    default_incoterm: str = "FCA"
    lease_scope_boundary_months: int = 12

    def validate(self) -> None:
        """Validate boundary configuration values."""
        valid_consolidation = {
            "operational_control", "financial_control", "equity_share",
        }
        if self.default_consolidation not in valid_consolidation:
            raise ValueError(
                f"Invalid default_consolidation '{self.default_consolidation}'. "
                f"Must be one of {valid_consolidation}"
            )
        if self.capex_threshold < 0:
            raise ValueError("capex_threshold must be >= 0")
        valid_incoterms = {
            "EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP",
            "FAS", "FOB", "CFR", "CIF",
        }
        if self.default_incoterm not in valid_incoterms:
            raise ValueError(
                f"Invalid default_incoterm '{self.default_incoterm}'. "
                f"Must be one of {valid_incoterms}"
            )
        if self.lease_scope_boundary_months < 1 or self.lease_scope_boundary_months > 120:
            raise ValueError("lease_scope_boundary_months must be between 1 and 120")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_consolidation": self.default_consolidation,
            "capex_threshold": self.capex_threshold,
            "default_incoterm": self.default_incoterm,
            "lease_scope_boundary_months": self.lease_scope_boundary_months,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundaryConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "BoundaryConfig":
        """Load from environment variables."""
        return cls(
            default_consolidation=os.getenv(
                "GL_SCM_CONSOLIDATION", "operational_control"
            ),
            capex_threshold=float(os.getenv("GL_SCM_CAPEX_THRESHOLD", "5000.0")),
            default_incoterm=os.getenv("GL_SCM_DEFAULT_INCOTERM", "FCA"),
            lease_scope_boundary_months=int(
                os.getenv("GL_SCM_LEASE_BOUNDARY_MONTHS", "12")
            ),
        )


# =============================================================================
# SECTION 4: COMPLETENESS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CompletenessConfig:
    """
    Category completeness screening configuration.

    Controls materiality thresholds, minimum reported categories, and
    industry benchmark settings for completeness assessment.

    Attributes:
        materiality_threshold_pct: Materiality threshold as % of total Scope 3 (GL_SCM_MATERIALITY_PCT)
        min_categories_reported: Minimum categories expected (GL_SCM_MIN_CATEGORIES)
        enable_industry_benchmarks: Use industry-specific benchmarks (GL_SCM_ENABLE_BENCHMARKS)

    Example:
        >>> comp = CompletenessConfig()
        >>> comp.materiality_threshold_pct
        1.0
    """

    materiality_threshold_pct: float = 1.0
    min_categories_reported: int = 5
    enable_industry_benchmarks: bool = True

    def validate(self) -> None:
        """Validate completeness configuration values."""
        if self.materiality_threshold_pct < 0.0 or self.materiality_threshold_pct > 100.0:
            raise ValueError("materiality_threshold_pct must be between 0.0 and 100.0")
        if self.min_categories_reported < 0 or self.min_categories_reported > 15:
            raise ValueError("min_categories_reported must be between 0 and 15")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "materiality_threshold_pct": self.materiality_threshold_pct,
            "min_categories_reported": self.min_categories_reported,
            "enable_industry_benchmarks": self.enable_industry_benchmarks,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletenessConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "CompletenessConfig":
        """Load from environment variables."""
        return cls(
            materiality_threshold_pct=float(
                os.getenv("GL_SCM_MATERIALITY_PCT", "1.0")
            ),
            min_categories_reported=int(
                os.getenv("GL_SCM_MIN_CATEGORIES", "5")
            ),
            enable_industry_benchmarks=os.getenv(
                "GL_SCM_ENABLE_BENCHMARKS", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 5: DOUBLE-COUNTING CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DoubleCountingConfig:
    """
    Double-counting prevention configuration.

    Controls which DC-SCM rules are enabled and their enforcement mode.

    Attributes:
        enable_dc_checks: Master switch for DC checks (GL_SCM_DC_ENABLED)
        strict_mode: If True, block records with DC violations (GL_SCM_DC_STRICT)
        rules_enabled: List of enabled DC-SCM rule IDs (GL_SCM_DC_RULES)

    Example:
        >>> dc = DoubleCountingConfig()
        >>> dc.enable_dc_checks
        True
        >>> len(dc.rules_enabled)
        10
    """

    enable_dc_checks: bool = True
    strict_mode: bool = False
    rules_enabled: Tuple[str, ...] = (
        "DC-SCM-001", "DC-SCM-002", "DC-SCM-003", "DC-SCM-004", "DC-SCM-005",
        "DC-SCM-006", "DC-SCM-007", "DC-SCM-008", "DC-SCM-009", "DC-SCM-010",
    )

    def validate(self) -> None:
        """Validate double-counting configuration values."""
        valid_rules = {
            f"DC-SCM-{i:03d}" for i in range(1, 11)
        }
        for rule in self.rules_enabled:
            if rule not in valid_rules:
                raise ValueError(
                    f"Invalid DC rule '{rule}'. "
                    f"Must be one of {sorted(valid_rules)}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_dc_checks": self.enable_dc_checks,
            "strict_mode": self.strict_mode,
            "rules_enabled": list(self.rules_enabled),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoubleCountingConfig":
        """Create from dictionary."""
        d = data.copy()
        if "rules_enabled" in d and isinstance(d["rules_enabled"], list):
            d["rules_enabled"] = tuple(d["rules_enabled"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "DoubleCountingConfig":
        """Load from environment variables."""
        rules_raw = os.getenv(
            "GL_SCM_DC_RULES",
            "DC-SCM-001,DC-SCM-002,DC-SCM-003,DC-SCM-004,DC-SCM-005,"
            "DC-SCM-006,DC-SCM-007,DC-SCM-008,DC-SCM-009,DC-SCM-010",
        )
        return cls(
            enable_dc_checks=os.getenv(
                "GL_SCM_DC_ENABLED", "true"
            ).lower() == "true",
            strict_mode=os.getenv(
                "GL_SCM_DC_STRICT", "false"
            ).lower() == "true",
            rules_enabled=tuple(s.strip() for s in rules_raw.split(",")),
        )


# =============================================================================
# SECTION 6: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Regulatory compliance assessment configuration.

    Controls which frameworks are assessed and the minimum passing score.

    Attributes:
        frameworks_enabled: Frameworks to assess (GL_SCM_COMPLIANCE_FRAMEWORKS)
        minimum_compliance_score: Minimum score to pass (GL_SCM_COMPLIANCE_MIN_SCORE)

    Example:
        >>> comp = ComplianceConfig()
        >>> 'GHG_PROTOCOL' in comp.frameworks_enabled
        True
    """

    frameworks_enabled: Tuple[str, ...] = (
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS", "CDP",
        "SBTI", "SB_253", "SEC_CLIMATE", "EU_TAXONOMY",
    )
    minimum_compliance_score: float = 70.0

    def validate(self) -> None:
        """Validate compliance configuration values."""
        valid_frameworks = {
            "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS", "CDP",
            "SBTI", "SB_253", "SEC_CLIMATE", "EU_TAXONOMY",
        }
        for fw in self.frameworks_enabled:
            if fw not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{fw}'. Must be one of {valid_frameworks}"
                )
        if self.minimum_compliance_score < 0.0 or self.minimum_compliance_score > 100.0:
            raise ValueError("minimum_compliance_score must be between 0.0 and 100.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frameworks_enabled": list(self.frameworks_enabled),
            "minimum_compliance_score": self.minimum_compliance_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        d = data.copy()
        if "frameworks_enabled" in d and isinstance(d["frameworks_enabled"], list):
            d["frameworks_enabled"] = tuple(d["frameworks_enabled"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        fw_raw = os.getenv(
            "GL_SCM_COMPLIANCE_FRAMEWORKS",
            "GHG_PROTOCOL,ISO_14064,CSRD_ESRS,CDP,SBTI,SB_253,SEC_CLIMATE,EU_TAXONOMY",
        )
        return cls(
            frameworks_enabled=tuple(s.strip() for s in fw_raw.split(",")),
            minimum_compliance_score=float(
                os.getenv("GL_SCM_COMPLIANCE_MIN_SCORE", "70.0")
            ),
        )


# =============================================================================
# SECTION 7: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Scope 3 Category Mapper agent.

    Attributes:
        host: PostgreSQL host (GL_SCM_DB_HOST)
        port: PostgreSQL port (GL_SCM_DB_PORT)
        name: Database name (GL_SCM_DB_NAME)
        username: Database username (GL_SCM_DB_USERNAME)
        password: Database password (GL_SCM_DB_PASSWORD)
        schema: Database schema name (GL_SCM_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_SCM_DB_TABLE_PREFIX)
        pool_size: Connection pool size (GL_SCM_DB_POOL_SIZE)
        ssl_mode: SSL connection mode (GL_SCM_DB_SSL_MODE)

    Example:
        >>> db = DatabaseConfig()
        >>> db.table_prefix
        'gl_scm_'
    """

    host: str = "localhost"
    port: int = 5432
    name: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "scope3_mapper_service"
    table_prefix: str = "gl_scm_"
    pool_size: int = 10
    ssl_mode: str = "prefer"

    def validate(self) -> None:
        """Validate database configuration values."""
        if not self.host:
            raise ValueError("host cannot be empty")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if not self.name:
            raise ValueError("name cannot be empty")
        if not self.schema:
            raise ValueError("schema cannot be empty")
        if not self.table_prefix:
            raise ValueError("table_prefix cannot be empty")
        if not self.table_prefix.endswith("_"):
            raise ValueError("table_prefix must end with '_'")
        if self.pool_size < 1 or self.pool_size > 100:
            raise ValueError("pool_size must be between 1 and 100")
        valid_ssl = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
        if self.ssl_mode not in valid_ssl:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. Must be one of {valid_ssl}"
            )

    def get_connection_url(self) -> str:
        """Build PostgreSQL connection URL."""
        auth = f"{self.username}:{self.password}" if self.password else self.username
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "name": self.name,
            "username": self.username,
            "password": self.password,
            "schema": self.schema,
            "table_prefix": self.table_prefix,
            "pool_size": self.pool_size,
            "ssl_mode": self.ssl_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_SCM_DB_HOST", "localhost"),
            port=int(os.getenv("GL_SCM_DB_PORT", "5432")),
            name=os.getenv("GL_SCM_DB_NAME", "greenlang"),
            username=os.getenv("GL_SCM_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_SCM_DB_PASSWORD", ""),
            schema=os.getenv("GL_SCM_DB_SCHEMA", "scope3_mapper_service"),
            table_prefix=os.getenv("GL_SCM_DB_TABLE_PREFIX", "gl_scm_"),
            pool_size=int(os.getenv("GL_SCM_DB_POOL_SIZE", "10")),
            ssl_mode=os.getenv("GL_SCM_DB_SSL_MODE", "prefer"),
        )


# =============================================================================
# SECTION 8: METRICS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class MetricsConfig:
    """
    Prometheus metrics configuration.

    Attributes:
        enabled: Enable metrics collection (GL_SCM_METRICS_ENABLED)
        prefix: Prometheus metric prefix (GL_SCM_METRICS_PREFIX)
        port: Metrics endpoint port (GL_SCM_METRICS_PORT)

    Example:
        >>> met = MetricsConfig()
        >>> met.prefix
        'gl_scm_'
    """

    enabled: bool = True
    prefix: str = "gl_scm_"
    port: int = 9090

    def validate(self) -> None:
        """Validate metrics configuration values."""
        if not self.prefix:
            raise ValueError("prefix cannot be empty")
        if not self.prefix.endswith("_"):
            raise ValueError("prefix must end with '_'")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "port": self.port,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_SCM_METRICS_ENABLED", "true").lower() == "true",
            prefix=os.getenv("GL_SCM_METRICS_PREFIX", "gl_scm_"),
            port=int(os.getenv("GL_SCM_METRICS_PORT", "9090")),
        )


# =============================================================================
# SECTION 9: PROVENANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ProvenanceConfig:
    """
    Data provenance tracking configuration.

    Attributes:
        enabled: Enable provenance tracking (GL_SCM_PROV_ENABLED)
        hash_algorithm: Hash algorithm for provenance (GL_SCM_PROV_HASH_ALGO)
        chain_length: Maximum entries per provenance chain (GL_SCM_PROV_CHAIN_LENGTH)

    Example:
        >>> prov = ProvenanceConfig()
        >>> prov.hash_algorithm
        'sha256'
    """

    enabled: bool = True
    hash_algorithm: str = "sha256"
    chain_length: int = 10

    def validate(self) -> None:
        """Validate provenance configuration values."""
        valid_algos = {"sha256", "sha384", "sha512"}
        if self.hash_algorithm not in valid_algos:
            raise ValueError(
                f"Invalid hash_algorithm '{self.hash_algorithm}'. "
                f"Must be one of {valid_algos}"
            )
        if self.chain_length < 1 or self.chain_length > 10000:
            raise ValueError("chain_length must be between 1 and 10000")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "hash_algorithm": self.hash_algorithm,
            "chain_length": self.chain_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ProvenanceConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv(
                "GL_SCM_PROV_ENABLED", "true"
            ).lower() == "true",
            hash_algorithm=os.getenv("GL_SCM_PROV_HASH_ALGO", "sha256"),
            chain_length=int(os.getenv("GL_SCM_PROV_CHAIN_LENGTH", "10")),
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass
class Scope3CategoryMapperConfig:
    """
    Master configuration for Scope 3 Category Mapper agent (AGENT-MRV-029).

    Composes all 9 configuration sections into a single unified
    configuration object with cross-validation support.

    Attributes:
        general: General agent configuration
        classification: Classification engine configuration
        boundary: Boundary determination configuration
        completeness: Completeness screening configuration
        double_counting: Double-counting prevention configuration
        compliance: Compliance assessment configuration
        database: PostgreSQL database configuration
        metrics: Prometheus metrics configuration
        provenance: Data provenance configuration

    Example:
        >>> config = Scope3CategoryMapperConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-X-040'
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    completeness: CompletenessConfig = field(default_factory=CompletenessConfig)
    double_counting: DoubleCountingConfig = field(default_factory=DoubleCountingConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)

    def validate_all(self) -> List[str]:
        """
        Validate all configuration sections and return list of errors.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors: List[str] = []

        sections = [
            ("general", self.general),
            ("classification", self.classification),
            ("boundary", self.boundary),
            ("completeness", self.completeness),
            ("double_counting", self.double_counting),
            ("compliance", self.compliance),
            ("database", self.database),
            ("metrics", self.metrics),
            ("provenance", self.provenance),
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

        # API prefix should start with /api/
        if self.general.api_prefix and not self.general.api_prefix.startswith("/api/"):
            errors.append(
                "cross-validation: general.api_prefix should start with '/api/'"
            )

        # Classification min confidence should be <= keyword confidence
        if self.classification.min_confidence_threshold > self.classification.keyword_confidence:
            logger.warning(
                "min_confidence_threshold (%.2f) exceeds keyword_confidence (%.2f) -- "
                "keyword matches will always be below threshold",
                self.classification.min_confidence_threshold,
                self.classification.keyword_confidence,
            )

        # Metrics prefix should match table prefix
        if self.metrics.prefix != self.database.table_prefix:
            logger.info(
                "metrics.prefix (%s) differs from database.table_prefix (%s)",
                self.metrics.prefix,
                self.database.table_prefix,
            )

        return errors

    def validate(self) -> None:
        """Validate and raise on first error. Convenience wrapper."""
        errors = self.validate_all()
        if errors:
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s): "
                + "; ".join(errors)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "general": self.general.to_dict(),
            "classification": self.classification.to_dict(),
            "boundary": self.boundary.to_dict(),
            "completeness": self.completeness.to_dict(),
            "double_counting": self.double_counting.to_dict(),
            "compliance": self.compliance.to_dict(),
            "database": self.database.to_dict(),
            "metrics": self.metrics.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scope3CategoryMapperConfig":
        """Create configuration from dictionary."""
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            classification=ClassificationConfig.from_dict(
                data.get("classification", {})
            ),
            boundary=BoundaryConfig.from_dict(data.get("boundary", {})),
            completeness=CompletenessConfig.from_dict(
                data.get("completeness", {})
            ),
            double_counting=DoubleCountingConfig.from_dict(
                data.get("double_counting", {})
            ),
            compliance=ComplianceConfig.from_dict(data.get("compliance", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            metrics=MetricsConfig.from_dict(data.get("metrics", {})),
            provenance=ProvenanceConfig.from_dict(data.get("provenance", {})),
        )

    @classmethod
    def from_env(cls) -> "Scope3CategoryMapperConfig":
        """Load configuration from environment variables."""
        return cls(
            general=GeneralConfig.from_env(),
            classification=ClassificationConfig.from_env(),
            boundary=BoundaryConfig.from_env(),
            completeness=CompletenessConfig.from_env(),
            double_counting=DoubleCountingConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
            database=DatabaseConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            provenance=ProvenanceConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[Scope3CategoryMapperConfig] = None
_config_lock = threading.RLock()


def get_config() -> Scope3CategoryMapperConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization. First call loads from environment
    variables. Subsequent calls return the cached instance.

    Returns:
        Scope3CategoryMapperConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-X-040'
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info("Initializing Scope3CategoryMapperConfig from environment")
                config = Scope3CategoryMapperConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning("Configuration validation warning: %s", error)
                _config_instance = config
                logger.info(
                    "Scope3CategoryMapperConfig initialized: agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: Scope3CategoryMapperConfig) -> None:
    """
    Set the singleton configuration instance.

    Args:
        config: Scope3CategoryMapperConfig instance to set as singleton

    Raises:
        TypeError: If config is not a Scope3CategoryMapperConfig instance
    """
    global _config_instance

    if not isinstance(config, Scope3CategoryMapperConfig):
        raise TypeError(
            f"config must be a Scope3CategoryMapperConfig instance, got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("Scope3CategoryMapperConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Clears the cached configuration singleton, forcing the next call to
    get_config() to reload from environment variables.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("Scope3CategoryMapperConfig singleton reset")


def validate_config(config: Scope3CategoryMapperConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate_all()


def print_config(config: Scope3CategoryMapperConfig) -> None:
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
    print("  Scope 3 Category Mapper Configuration (AGENT-MRV-029)")
    print("  Agent ID: " + config.general.agent_id)
    print("  Version:  " + config.general.version)
    print("=" * 64)

    _print_section("GENERAL", config.general.to_dict())
    _print_section("CLASSIFICATION", config.classification.to_dict())
    _print_section("BOUNDARY", config.boundary.to_dict())
    _print_section("COMPLETENESS", config.completeness.to_dict())
    _print_section("DOUBLE_COUNTING", config.double_counting.to_dict())
    _print_section("COMPLIANCE", config.compliance.to_dict())
    _print_section("DATABASE", config.database.to_dict())
    _print_section("METRICS", config.metrics.to_dict())
    _print_section("PROVENANCE", config.provenance.to_dict())

    errors = config.validate_all()
    print("\n[VALIDATION]")
    if errors:
        print(f"  status: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"  - {error}")
    else:
        print("  status: PASSED (0 errors)")
