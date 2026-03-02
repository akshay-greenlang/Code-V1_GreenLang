# -*- coding: utf-8 -*-
"""
Audit Trail & Lineage Configuration - AGENT-MRV-030

Thread-safe singleton configuration for GL-MRV-X-042.
All environment variables prefixed with GL_ATL_.

This module provides comprehensive configuration management for the
Audit Trail & Lineage agent, supporting:
- Immutable audit event chains with SHA-256 hash linking
- Data lineage graph tracking (source-to-report provenance)
- Change detection and materiality impact analysis
- Evidence packaging for third-party assurance (limited/reasonable)
- Digital signature verification (Ed25519, RSA, ECDSA)
- Compliance coverage assessment across 9 regulatory frameworks
  (GHG Protocol, ISO 14064, CSRD/ESRS, CDP, SBTi, SB 253,
   SEC Climate, EU Taxonomy, ISAE 3410)
- Blockchain-anchored genesis hashing for tamper-proof chains
- Database and Redis caching configuration
- Thread-safe singleton with double-checked locking

Example:
    >>> config = get_config()
    >>> config.general.agent_id
    'GL-MRV-X-042'
    >>> config.audit.enable_chain_verification
    True
    >>> config.compliance.supported_frameworks
    ['GHG_PROTOCOL', 'ISO_14064', 'CSRD_ESRS', 'CDP', 'SBTI', 'SB_253', 'SEC_CLIMATE', 'EU_TAXONOMY', 'ISAE_3410']

Thread Safety:
    All configuration operations are protected by threading.RLock() to ensure
    thread-safe singleton access in multi-threaded environments.

Environment Variables:
    All configuration values can be set via environment variables with the
    GL_ATL_ prefix. See individual config sections for specific variables.

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-X-042
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
    General configuration for Audit Trail & Lineage agent.

    Attributes:
        enabled: Master switch for the agent (GL_ATL_ENABLED)
        debug: Enable debug mode with verbose logging (GL_ATL_DEBUG)
        log_level: Logging level (GL_ATL_LOG_LEVEL)
        agent_id: Unique agent identifier (GL_ATL_AGENT_ID)
        agent_component: Agent component identifier (GL_ATL_AGENT_COMPONENT)
        version: Agent version following SemVer (GL_ATL_VERSION)
        api_prefix: API route prefix (GL_ATL_API_PREFIX)
        max_batch_size: Maximum records per batch (GL_ATL_MAX_BATCH_SIZE)
        chain_hash_algorithm: Hash algorithm for audit chain linking (GL_ATL_CHAIN_HASH_ALGORITHM)
        genesis_hash: Seed value for the first link in every audit chain (GL_ATL_GENESIS_HASH)
        max_chain_length: Maximum events allowed in a single audit chain (GL_ATL_MAX_CHAIN_LENGTH)
        enable_signatures: Enable digital signature verification on audit events (GL_ATL_ENABLE_SIGNATURES)

    Example:
        >>> general = GeneralConfig()
        >>> general.agent_id
        'GL-MRV-X-042'
        >>> general.chain_hash_algorithm
        'sha256'
    """

    enabled: bool = True
    debug: bool = False
    log_level: str = "INFO"
    agent_id: str = "GL-MRV-X-042"
    agent_component: str = "AGENT-MRV-030"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1/audit-trail-lineage"
    max_batch_size: int = 10000
    chain_hash_algorithm: str = "sha256"
    genesis_hash: str = "greenlang-atl-genesis-v1"
    max_chain_length: int = 10000000
    enable_signatures: bool = True

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
        if self.max_batch_size < 1 or self.max_batch_size > 500000:
            raise ValueError("max_batch_size must be between 1 and 500000")
        valid_hash_algorithms = {"sha256", "sha384", "sha512"}
        if self.chain_hash_algorithm not in valid_hash_algorithms:
            raise ValueError(
                f"Invalid chain_hash_algorithm '{self.chain_hash_algorithm}'. "
                f"Must be one of {valid_hash_algorithms}"
            )
        if not self.genesis_hash:
            raise ValueError("genesis_hash cannot be empty")
        if len(self.genesis_hash) < 5:
            raise ValueError("genesis_hash must be at least 5 characters")
        if self.max_chain_length < 1 or self.max_chain_length > 100000000:
            raise ValueError("max_chain_length must be between 1 and 100000000")

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
            "chain_hash_algorithm": self.chain_hash_algorithm,
            "genesis_hash": self.genesis_hash,
            "max_chain_length": self.max_chain_length,
            "enable_signatures": self.enable_signatures,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneralConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "GeneralConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GL_ATL_ENABLED", "true").lower() == "true",
            debug=os.getenv("GL_ATL_DEBUG", "false").lower() == "true",
            log_level=os.getenv("GL_ATL_LOG_LEVEL", "INFO"),
            agent_id=os.getenv("GL_ATL_AGENT_ID", "GL-MRV-X-042"),
            agent_component=os.getenv("GL_ATL_AGENT_COMPONENT", "AGENT-MRV-030"),
            version=os.getenv("GL_ATL_VERSION", "1.0.0"),
            api_prefix=os.getenv(
                "GL_ATL_API_PREFIX", "/api/v1/audit-trail-lineage"
            ),
            max_batch_size=int(os.getenv("GL_ATL_MAX_BATCH_SIZE", "10000")),
            chain_hash_algorithm=os.getenv(
                "GL_ATL_CHAIN_HASH_ALGORITHM", "sha256"
            ),
            genesis_hash=os.getenv(
                "GL_ATL_GENESIS_HASH", "greenlang-atl-genesis-v1"
            ),
            max_chain_length=int(
                os.getenv("GL_ATL_MAX_CHAIN_LENGTH", "10000000")
            ),
            enable_signatures=os.getenv(
                "GL_ATL_ENABLE_SIGNATURES", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 2: DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration for Audit Trail & Lineage agent.

    Attributes:
        host: PostgreSQL host (GL_ATL_DB_HOST)
        port: PostgreSQL port (GL_ATL_DB_PORT)
        database: Database name (GL_ATL_DB_NAME)
        username: Database username (GL_ATL_DB_USERNAME)
        password: Database password (GL_ATL_DB_PASSWORD)
        schema: Database schema name (GL_ATL_DB_SCHEMA)
        table_prefix: Prefix for all tables (GL_ATL_DB_TABLE_PREFIX)
        pool_min: Minimum connections in pool (GL_ATL_DB_POOL_MIN)
        pool_max: Maximum connections in pool (GL_ATL_DB_POOL_MAX)
        statement_timeout_ms: Statement timeout in milliseconds (GL_ATL_DB_STATEMENT_TIMEOUT_MS)
        ssl_mode: SSL connection mode (GL_ATL_DB_SSL_MODE)

    Example:
        >>> db = DatabaseConfig()
        >>> db.table_prefix
        'gl_atl_'
        >>> db.schema
        'audit_trail_lineage_service'
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "greenlang"
    username: str = "greenlang"
    password: str = ""
    schema: str = "audit_trail_lineage_service"
    table_prefix: str = "gl_atl_"
    pool_min: int = 2
    pool_max: int = 10
    statement_timeout_ms: int = 30000
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
        if self.pool_min < 1 or self.pool_min > 50:
            raise ValueError("pool_min must be between 1 and 50")
        if self.pool_max < 1 or self.pool_max > 100:
            raise ValueError("pool_max must be between 1 and 100")
        if self.pool_min > self.pool_max:
            raise ValueError("pool_min cannot exceed pool_max")
        if self.statement_timeout_ms < 1000 or self.statement_timeout_ms > 300000:
            raise ValueError(
                "statement_timeout_ms must be between 1000 and 300000"
            )
        valid_ssl = {
            "disable", "allow", "prefer", "require",
            "verify-ca", "verify-full",
        }
        if self.ssl_mode not in valid_ssl:
            raise ValueError(
                f"Invalid ssl_mode '{self.ssl_mode}'. Must be one of {valid_ssl}"
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
            "pool_min": self.pool_min,
            "pool_max": self.pool_max,
            "statement_timeout_ms": self.statement_timeout_ms,
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
            host=os.getenv("GL_ATL_DB_HOST", "localhost"),
            port=int(os.getenv("GL_ATL_DB_PORT", "5432")),
            database=os.getenv("GL_ATL_DB_NAME", "greenlang"),
            username=os.getenv("GL_ATL_DB_USERNAME", "greenlang"),
            password=os.getenv("GL_ATL_DB_PASSWORD", ""),
            schema=os.getenv(
                "GL_ATL_DB_SCHEMA", "audit_trail_lineage_service"
            ),
            table_prefix=os.getenv("GL_ATL_DB_TABLE_PREFIX", "gl_atl_"),
            pool_min=int(os.getenv("GL_ATL_DB_POOL_MIN", "2")),
            pool_max=int(os.getenv("GL_ATL_DB_POOL_MAX", "10")),
            statement_timeout_ms=int(
                os.getenv("GL_ATL_DB_STATEMENT_TIMEOUT_MS", "30000")
            ),
            ssl_mode=os.getenv("GL_ATL_DB_SSL_MODE", "prefer"),
        )


# =============================================================================
# SECTION 3: REDIS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for Audit Trail & Lineage agent.

    Used for caching lineage graph lookups, chain-head pointers,
    and recently verified chain hashes.

    Attributes:
        host: Redis host (GL_ATL_REDIS_HOST)
        port: Redis port (GL_ATL_REDIS_PORT)
        db: Redis database index (GL_ATL_REDIS_DB)
        password: Redis password (GL_ATL_REDIS_PASSWORD)
        ssl: Enable SSL connection (GL_ATL_REDIS_SSL)
        prefix: Key prefix for namespacing (GL_ATL_REDIS_PREFIX)
        ttl_seconds: Default TTL in seconds for cached entries (GL_ATL_REDIS_TTL_SECONDS)
        max_connections: Max connections in pool (GL_ATL_REDIS_MAX_CONNECTIONS)
        socket_timeout: Socket timeout in seconds (GL_ATL_REDIS_SOCKET_TIMEOUT)

    Example:
        >>> redis = RedisConfig()
        >>> redis.prefix
        'gl_atl:'
        >>> redis.ttl_seconds
        3600
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ssl: bool = False
    prefix: str = "gl_atl:"
    ttl_seconds: int = 3600
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
        if self.ttl_seconds < 1 or self.ttl_seconds > 604800:
            raise ValueError("ttl_seconds must be between 1 and 604800 (7 days)")
        if self.max_connections < 1 or self.max_connections > 1000:
            raise ValueError("max_connections must be between 1 and 1000")
        if self.socket_timeout < 1 or self.socket_timeout > 60:
            raise ValueError("socket_timeout must be between 1 and 60 seconds")

    def get_connection_url(self) -> str:
        """
        Build Redis connection URL.

        Returns:
            Redis connection URL string
        """
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
            "ttl_seconds": self.ttl_seconds,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load from environment variables."""
        return cls(
            host=os.getenv("GL_ATL_REDIS_HOST", "localhost"),
            port=int(os.getenv("GL_ATL_REDIS_PORT", "6379")),
            db=int(os.getenv("GL_ATL_REDIS_DB", "0")),
            password=os.getenv("GL_ATL_REDIS_PASSWORD", ""),
            ssl=os.getenv("GL_ATL_REDIS_SSL", "false").lower() == "true",
            prefix=os.getenv("GL_ATL_REDIS_PREFIX", "gl_atl:"),
            ttl_seconds=int(os.getenv("GL_ATL_REDIS_TTL_SECONDS", "3600")),
            max_connections=int(
                os.getenv("GL_ATL_REDIS_MAX_CONNECTIONS", "20")
            ),
            socket_timeout=int(
                os.getenv("GL_ATL_REDIS_SOCKET_TIMEOUT", "5")
            ),
        )


# =============================================================================
# SECTION 4: AUDIT CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class AuditConfig:
    """
    Audit event chain and lineage graph configuration.

    Controls the immutable audit event chain behaviour, lineage graph
    traversal limits, change detection sensitivity, and materiality
    thresholds for impact analysis.

    Attributes:
        max_event_payload_bytes: Maximum payload size per audit event (GL_ATL_MAX_EVENT_PAYLOAD_BYTES)
        enable_chain_verification: Enable periodic chain integrity verification (GL_ATL_ENABLE_CHAIN_VERIFICATION)
        chain_verification_interval_events: Verify chain every N events (GL_ATL_CHAIN_VERIFICATION_INTERVAL)
        max_lineage_depth: Maximum depth for lineage graph traversal (GL_ATL_MAX_LINEAGE_DEPTH)
        max_lineage_nodes: Maximum nodes in the lineage graph (GL_ATL_MAX_LINEAGE_NODES)
        max_lineage_edges: Maximum edges in the lineage graph (GL_ATL_MAX_LINEAGE_EDGES)
        enable_change_detection: Enable change detection on data updates (GL_ATL_ENABLE_CHANGE_DETECTION)
        materiality_threshold_pct: Minimum % change to flag as material (GL_ATL_MATERIALITY_THRESHOLD_PCT)

    Example:
        >>> audit = AuditConfig()
        >>> audit.enable_chain_verification
        True
        >>> audit.materiality_threshold_pct
        Decimal('5.0')
    """

    max_event_payload_bytes: int = 65536
    enable_chain_verification: bool = True
    chain_verification_interval_events: int = 1000
    max_lineage_depth: int = 50
    max_lineage_nodes: int = 1000000
    max_lineage_edges: int = 5000000
    enable_change_detection: bool = True
    materiality_threshold_pct: Decimal = Decimal("5.0")

    def validate(self) -> None:
        """
        Validate audit configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        if self.max_event_payload_bytes < 1024 or self.max_event_payload_bytes > 16777216:
            raise ValueError(
                "max_event_payload_bytes must be between 1024 (1 KB) "
                "and 16777216 (16 MB)"
            )
        if self.chain_verification_interval_events < 1 or \
                self.chain_verification_interval_events > 1000000:
            raise ValueError(
                "chain_verification_interval_events must be between 1 and 1000000"
            )
        if self.max_lineage_depth < 1 or self.max_lineage_depth > 500:
            raise ValueError("max_lineage_depth must be between 1 and 500")
        if self.max_lineage_nodes < 1 or self.max_lineage_nodes > 100000000:
            raise ValueError(
                "max_lineage_nodes must be between 1 and 100000000"
            )
        if self.max_lineage_edges < 1 or self.max_lineage_edges > 500000000:
            raise ValueError(
                "max_lineage_edges must be between 1 and 500000000"
            )
        if self.materiality_threshold_pct < Decimal("0.0") or \
                self.materiality_threshold_pct > Decimal("100.0"):
            raise ValueError(
                "materiality_threshold_pct must be between 0.0 and 100.0"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_event_payload_bytes": self.max_event_payload_bytes,
            "enable_chain_verification": self.enable_chain_verification,
            "chain_verification_interval_events": self.chain_verification_interval_events,
            "max_lineage_depth": self.max_lineage_depth,
            "max_lineage_nodes": self.max_lineage_nodes,
            "max_lineage_edges": self.max_lineage_edges,
            "enable_change_detection": self.enable_change_detection,
            "materiality_threshold_pct": str(self.materiality_threshold_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditConfig":
        """Create from dictionary."""
        d = data.copy()
        if "materiality_threshold_pct" in d and not isinstance(
            d["materiality_threshold_pct"], Decimal
        ):
            d["materiality_threshold_pct"] = Decimal(
                str(d["materiality_threshold_pct"])
            )
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "AuditConfig":
        """Load from environment variables."""
        return cls(
            max_event_payload_bytes=int(
                os.getenv("GL_ATL_MAX_EVENT_PAYLOAD_BYTES", "65536")
            ),
            enable_chain_verification=os.getenv(
                "GL_ATL_ENABLE_CHAIN_VERIFICATION", "true"
            ).lower() == "true",
            chain_verification_interval_events=int(
                os.getenv("GL_ATL_CHAIN_VERIFICATION_INTERVAL", "1000")
            ),
            max_lineage_depth=int(
                os.getenv("GL_ATL_MAX_LINEAGE_DEPTH", "50")
            ),
            max_lineage_nodes=int(
                os.getenv("GL_ATL_MAX_LINEAGE_NODES", "1000000")
            ),
            max_lineage_edges=int(
                os.getenv("GL_ATL_MAX_LINEAGE_EDGES", "5000000")
            ),
            enable_change_detection=os.getenv(
                "GL_ATL_ENABLE_CHANGE_DETECTION", "true"
            ).lower() == "true",
            materiality_threshold_pct=Decimal(
                os.getenv("GL_ATL_MATERIALITY_THRESHOLD_PCT", "5.0")
            ),
        )


# =============================================================================
# SECTION 5: EVIDENCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EvidenceConfig:
    """
    Evidence packaging and assurance configuration.

    Controls how audit evidence is bundled for third-party assurance
    engagements, including digital signatures, compression, and
    retention policies.

    Attributes:
        default_assurance_level: Default assurance level for evidence packages (GL_ATL_DEFAULT_ASSURANCE_LEVEL)
        max_package_size_mb: Maximum evidence package size in megabytes (GL_ATL_MAX_PACKAGE_SIZE_MB)
        enable_digital_signatures: Enable digital signatures on evidence packages (GL_ATL_ENABLE_DIGITAL_SIGNATURES)
        default_signature_algorithm: Default digital signature algorithm (GL_ATL_DEFAULT_SIGNATURE_ALGORITHM)
        package_retention_years: Years to retain evidence packages (GL_ATL_PACKAGE_RETENTION_YEARS)
        enable_compression: Enable compression for evidence packages (GL_ATL_ENABLE_COMPRESSION)

    Example:
        >>> evidence = EvidenceConfig()
        >>> evidence.default_assurance_level
        'limited'
        >>> evidence.default_signature_algorithm
        'ed25519'
    """

    default_assurance_level: str = "limited"
    max_package_size_mb: int = 100
    enable_digital_signatures: bool = True
    default_signature_algorithm: str = "ed25519"
    package_retention_years: int = 10
    enable_compression: bool = True

    def validate(self) -> None:
        """
        Validate evidence configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_assurance_levels = {"limited", "reasonable", "none"}
        if self.default_assurance_level not in valid_assurance_levels:
            raise ValueError(
                f"Invalid default_assurance_level '{self.default_assurance_level}'. "
                f"Must be one of {valid_assurance_levels}"
            )
        if self.max_package_size_mb < 1 or self.max_package_size_mb > 10000:
            raise ValueError(
                "max_package_size_mb must be between 1 and 10000"
            )
        valid_signature_algorithms = {"ed25519", "rsa", "ecdsa"}
        if self.default_signature_algorithm not in valid_signature_algorithms:
            raise ValueError(
                f"Invalid default_signature_algorithm "
                f"'{self.default_signature_algorithm}'. "
                f"Must be one of {valid_signature_algorithms}"
            )
        if self.package_retention_years < 1 or self.package_retention_years > 100:
            raise ValueError(
                "package_retention_years must be between 1 and 100"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_assurance_level": self.default_assurance_level,
            "max_package_size_mb": self.max_package_size_mb,
            "enable_digital_signatures": self.enable_digital_signatures,
            "default_signature_algorithm": self.default_signature_algorithm,
            "package_retention_years": self.package_retention_years,
            "enable_compression": self.enable_compression,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "EvidenceConfig":
        """Load from environment variables."""
        return cls(
            default_assurance_level=os.getenv(
                "GL_ATL_DEFAULT_ASSURANCE_LEVEL", "limited"
            ),
            max_package_size_mb=int(
                os.getenv("GL_ATL_MAX_PACKAGE_SIZE_MB", "100")
            ),
            enable_digital_signatures=os.getenv(
                "GL_ATL_ENABLE_DIGITAL_SIGNATURES", "true"
            ).lower() == "true",
            default_signature_algorithm=os.getenv(
                "GL_ATL_DEFAULT_SIGNATURE_ALGORITHM", "ed25519"
            ),
            package_retention_years=int(
                os.getenv("GL_ATL_PACKAGE_RETENTION_YEARS", "10")
            ),
            enable_compression=os.getenv(
                "GL_ATL_ENABLE_COMPRESSION", "true"
            ).lower() == "true",
        )


# =============================================================================
# SECTION 6: COMPLIANCE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ComplianceConfig:
    """
    Regulatory compliance coverage assessment configuration.

    Controls which frameworks are assessed for audit trail and lineage
    completeness, coverage thresholds for pass/warn/fail, and gap
    analysis features.

    Attributes:
        supported_frameworks: List of supported regulatory frameworks (GL_ATL_SUPPORTED_FRAMEWORKS)
        coverage_warn_threshold: Coverage percentage that triggers a warning (GL_ATL_COVERAGE_WARN_THRESHOLD)
        coverage_fail_threshold: Coverage percentage below which compliance fails (GL_ATL_COVERAGE_FAIL_THRESHOLD)
        enable_gap_analysis: Enable gap analysis for missing requirements (GL_ATL_ENABLE_GAP_ANALYSIS)
        enable_requirement_mapping: Enable mapping of audit events to framework requirements (GL_ATL_ENABLE_REQUIREMENT_MAPPING)

    Example:
        >>> compliance = ComplianceConfig()
        >>> len(compliance.supported_frameworks)
        9
        >>> compliance.coverage_warn_threshold
        Decimal('0.80')
    """

    supported_frameworks: Tuple[str, ...] = (
        "GHG_PROTOCOL",
        "ISO_14064",
        "CSRD_ESRS",
        "CDP",
        "SBTI",
        "SB_253",
        "SEC_CLIMATE",
        "EU_TAXONOMY",
        "ISAE_3410",
    )
    coverage_warn_threshold: Decimal = Decimal("0.80")
    coverage_fail_threshold: Decimal = Decimal("0.50")
    enable_gap_analysis: bool = True
    enable_requirement_mapping: bool = True

    def validate(self) -> None:
        """
        Validate compliance configuration values.

        Raises:
            ValueError: If configuration values are invalid
        """
        valid_frameworks = {
            "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS", "CDP",
            "SBTI", "SB_253", "SEC_CLIMATE", "EU_TAXONOMY", "ISAE_3410",
        }
        for fw in self.supported_frameworks:
            if fw not in valid_frameworks:
                raise ValueError(
                    f"Invalid framework '{fw}'. "
                    f"Must be one of {sorted(valid_frameworks)}"
                )
        if len(self.supported_frameworks) != len(set(self.supported_frameworks)):
            raise ValueError("supported_frameworks contains duplicate entries")
        if self.coverage_warn_threshold < Decimal("0.0") or \
                self.coverage_warn_threshold > Decimal("1.0"):
            raise ValueError(
                "coverage_warn_threshold must be between 0.0 and 1.0"
            )
        if self.coverage_fail_threshold < Decimal("0.0") or \
                self.coverage_fail_threshold > Decimal("1.0"):
            raise ValueError(
                "coverage_fail_threshold must be between 0.0 and 1.0"
            )
        if self.coverage_fail_threshold > self.coverage_warn_threshold:
            raise ValueError(
                "coverage_fail_threshold cannot exceed coverage_warn_threshold"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "supported_frameworks": list(self.supported_frameworks),
            "coverage_warn_threshold": str(self.coverage_warn_threshold),
            "coverage_fail_threshold": str(self.coverage_fail_threshold),
            "enable_gap_analysis": self.enable_gap_analysis,
            "enable_requirement_mapping": self.enable_requirement_mapping,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplianceConfig":
        """Create from dictionary."""
        d = data.copy()
        if "supported_frameworks" in d and isinstance(
            d["supported_frameworks"], list
        ):
            d["supported_frameworks"] = tuple(d["supported_frameworks"])
        if "coverage_warn_threshold" in d and not isinstance(
            d["coverage_warn_threshold"], Decimal
        ):
            d["coverage_warn_threshold"] = Decimal(
                str(d["coverage_warn_threshold"])
            )
        if "coverage_fail_threshold" in d and not isinstance(
            d["coverage_fail_threshold"], Decimal
        ):
            d["coverage_fail_threshold"] = Decimal(
                str(d["coverage_fail_threshold"])
            )
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ComplianceConfig":
        """Load from environment variables."""
        fw_raw = os.getenv(
            "GL_ATL_SUPPORTED_FRAMEWORKS",
            "GHG_PROTOCOL,ISO_14064,CSRD_ESRS,CDP,SBTI,"
            "SB_253,SEC_CLIMATE,EU_TAXONOMY,ISAE_3410",
        )
        return cls(
            supported_frameworks=tuple(
                s.strip() for s in fw_raw.split(",")
            ),
            coverage_warn_threshold=Decimal(
                os.getenv("GL_ATL_COVERAGE_WARN_THRESHOLD", "0.80")
            ),
            coverage_fail_threshold=Decimal(
                os.getenv("GL_ATL_COVERAGE_FAIL_THRESHOLD", "0.50")
            ),
            enable_gap_analysis=os.getenv(
                "GL_ATL_ENABLE_GAP_ANALYSIS", "true"
            ).lower() == "true",
            enable_requirement_mapping=os.getenv(
                "GL_ATL_ENABLE_REQUIREMENT_MAPPING", "true"
            ).lower() == "true",
        )


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================


@dataclass
class AuditTrailLineageConfig:
    """
    Master configuration for Audit Trail & Lineage agent (AGENT-MRV-030).

    Composes all 6 configuration sections into a single unified
    configuration object with cross-validation support.

    Attributes:
        general: General agent configuration
        database: PostgreSQL database configuration
        redis: Redis caching configuration
        audit: Audit event chain and lineage graph configuration
        evidence: Evidence packaging and assurance configuration
        compliance: Regulatory compliance coverage configuration

    Example:
        >>> config = AuditTrailLineageConfig.from_env()
        >>> config.general.agent_id
        'GL-MRV-X-042'
        >>> errors = config.validate_all()
        >>> len(errors)
        0
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

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
            ("audit", self.audit),
            ("evidence", self.evidence),
            ("compliance", self.compliance),
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

        # API prefix should start with /api/
        if self.general.api_prefix and not self.general.api_prefix.startswith(
            "/api/"
        ):
            errors.append(
                "cross-validation: general.api_prefix should start with '/api/'"
            )

        # Evidence digital signatures require general signatures to be enabled
        if self.evidence.enable_digital_signatures and \
                not self.general.enable_signatures:
            logger.warning(
                "evidence.enable_digital_signatures is True but "
                "general.enable_signatures is False -- "
                "digital signatures will not function correctly"
            )

        # Chain verification interval should be reasonable relative to max chain length
        if self.audit.enable_chain_verification and \
                self.audit.chain_verification_interval_events > \
                self.general.max_chain_length:
            logger.warning(
                "chain_verification_interval_events (%d) exceeds "
                "max_chain_length (%d) -- "
                "chain verification will never trigger",
                self.audit.chain_verification_interval_events,
                self.general.max_chain_length,
            )

        # Redis prefix should match the agent naming convention
        expected_prefix = "gl_atl:"
        if self.redis.prefix != expected_prefix:
            logger.info(
                "redis.prefix (%s) differs from expected (%s)",
                self.redis.prefix,
                expected_prefix,
            )

        # Database table prefix should match the agent naming convention
        expected_table_prefix = "gl_atl_"
        if self.database.table_prefix != expected_table_prefix:
            logger.info(
                "database.table_prefix (%s) differs from expected (%s)",
                self.database.table_prefix,
                expected_table_prefix,
            )

        # Coverage fail threshold should be strictly less than warn threshold
        if self.compliance.coverage_fail_threshold >= \
                self.compliance.coverage_warn_threshold:
            errors.append(
                "cross-validation: compliance.coverage_fail_threshold must be "
                "less than compliance.coverage_warn_threshold"
            )

        return errors

    def validate(self) -> None:
        """
        Validate and raise on first error. Convenience wrapper.

        Raises:
            ValueError: If any configuration section fails validation
        """
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
            "database": self.database.to_dict(),
            "redis": self.redis.to_dict(),
            "audit": self.audit.to_dict(),
            "evidence": self.evidence.to_dict(),
            "compliance": self.compliance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditTrailLineageConfig":
        """Create configuration from dictionary."""
        return cls(
            general=GeneralConfig.from_dict(data.get("general", {})),
            database=DatabaseConfig.from_dict(data.get("database", {})),
            redis=RedisConfig.from_dict(data.get("redis", {})),
            audit=AuditConfig.from_dict(data.get("audit", {})),
            evidence=EvidenceConfig.from_dict(data.get("evidence", {})),
            compliance=ComplianceConfig.from_dict(
                data.get("compliance", {})
            ),
        )

    @classmethod
    def from_env(cls) -> "AuditTrailLineageConfig":
        """Load configuration from environment variables."""
        return cls(
            general=GeneralConfig.from_env(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            audit=AuditConfig.from_env(),
            evidence=EvidenceConfig.from_env(),
            compliance=ComplianceConfig.from_env(),
        )


# =============================================================================
# THREAD-SAFE SINGLETON PATTERN
# =============================================================================


_config_instance: Optional[AuditTrailLineageConfig] = None
_config_lock = threading.RLock()


def get_config() -> AuditTrailLineageConfig:
    """
    Get the singleton configuration instance.

    Thread-safe lazy initialization using double-checked locking.
    First call loads from environment variables and validates.
    Subsequent calls return the cached instance.

    Returns:
        AuditTrailLineageConfig singleton instance

    Example:
        >>> config = get_config()
        >>> config.general.agent_id
        'GL-MRV-X-042'
    """
    global _config_instance

    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                logger.info(
                    "Initializing AuditTrailLineageConfig from environment"
                )
                config = AuditTrailLineageConfig.from_env()
                errors = config.validate_all()
                if errors:
                    for error in errors:
                        logger.warning(
                            "Configuration validation warning: %s", error
                        )
                _config_instance = config
                logger.info(
                    "AuditTrailLineageConfig initialized: "
                    "agent_id=%s, version=%s",
                    config.general.agent_id,
                    config.general.version,
                )

    return _config_instance


def set_config(config: AuditTrailLineageConfig) -> None:
    """
    Set the singleton configuration instance.

    Validates the provided configuration before accepting it as the
    new singleton. Warnings are logged but do not prevent setting.

    Args:
        config: AuditTrailLineageConfig instance to set as singleton

    Raises:
        TypeError: If config is not an AuditTrailLineageConfig instance
    """
    global _config_instance

    if not isinstance(config, AuditTrailLineageConfig):
        raise TypeError(
            f"config must be an AuditTrailLineageConfig instance, "
            f"got {type(config)}"
        )

    with _config_lock:
        errors = config.validate_all()
        if errors:
            for error in errors:
                logger.warning("Configuration validation warning: %s", error)
        _config_instance = config
        logger.info("AuditTrailLineageConfig manually set")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.

    Clears the cached configuration singleton, forcing the next call to
    get_config() to reload from environment variables.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.info("AuditTrailLineageConfig singleton reset")


def validate_config(config: AuditTrailLineageConfig) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration instance to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    return config.validate_all()


def print_config(config: AuditTrailLineageConfig) -> None:
    """
    Print configuration in human-readable format.

    Sensitive fields (passwords, connection URLs) are redacted.

    Args:
        config: Configuration instance to print
    """
    redacted_fields = {
        "password", "database_url", "redis_url", "secret", "token", "key",
    }

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
    print("  Audit Trail & Lineage Configuration (AGENT-MRV-030)")
    print("  Agent ID: " + config.general.agent_id)
    print("  Version:  " + config.general.version)
    print("=" * 64)

    _print_section("GENERAL", config.general.to_dict())
    _print_section("DATABASE", config.database.to_dict())
    _print_section("REDIS", config.redis.to_dict())
    _print_section("AUDIT", config.audit.to_dict())
    _print_section("EVIDENCE", config.evidence.to_dict())
    _print_section("COMPLIANCE", config.compliance.to_dict())

    errors = config.validate_all()
    print("\n[VALIDATION]")
    if errors:
        print(f"  status: FAILED ({len(errors)} errors)")
        for error in errors:
            print(f"  - {error}")
    else:
        print("  status: PASSED (0 errors)")
