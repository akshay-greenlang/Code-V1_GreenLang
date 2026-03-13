# -*- coding: utf-8 -*-
"""
Due Diligence Orchestrator Configuration - AGENT-EUDR-026

Centralized configuration for the Due Diligence Orchestrator Agent covering:

- Database and cache connection settings (PostgreSQL, Redis) with configurable
  pool sizes, timeouts, and key prefixes using ``gl_eudr_ddo_`` namespace
- Workflow orchestration: maximum concurrent agents per workflow (default 10,
  max 25), global concurrency limit (100), workflow timeout (3600s), and
  checkpoint interval settings
- Quality gate thresholds: QG-1 information gathering completeness (90%),
  QG-2 risk assessment coverage (95%), QG-3 mitigation residual risk (15)
  with simplified due diligence relaxed thresholds
- Retry and circuit breaker: exponential backoff base delay (1s), max delay
  (300s), max retries (5), circuit breaker failure threshold (5), reset
  timeout (60s), success threshold (2)
- Risk assessment weights: country (0.15), supplier (0.12), commodity (0.10),
  corruption (0.08), deforestation (0.15), indigenous (0.10), protected (0.10),
  legal (0.10), audit (0.05), mitigation (0.05) -- sum = 1.00
- Risk mitigation thresholds: negligible risk (20), standard threshold (50),
  residual risk target (15)
- Package generation: output formats (PDF, JSON, HTML), supported languages
  (EN, FR, DE, ES, PT), DDS schema version
- Agent endpoint configuration for all 25 upstream EUDR agents

All settings can be overridden via environment variables with the
``GL_EUDR_DDO_`` prefix (e.g. ``GL_EUDR_DDO_DATABASE_URL``,
``GL_EUDR_DDO_MAX_CONCURRENT_AGENTS``).

Environment Variable Reference (GL_EUDR_DDO_ prefix):
    GL_EUDR_DDO_DATABASE_URL                    - PostgreSQL connection URL
    GL_EUDR_DDO_REDIS_URL                       - Redis connection URL
    GL_EUDR_DDO_LOG_LEVEL                       - Logging level
    GL_EUDR_DDO_POOL_SIZE                       - Database pool size
    GL_EUDR_DDO_POOL_TIMEOUT_S                  - Pool timeout seconds
    GL_EUDR_DDO_POOL_RECYCLE_S                  - Pool recycle seconds
    GL_EUDR_DDO_REDIS_TTL_S                     - Redis cache TTL seconds
    GL_EUDR_DDO_REDIS_KEY_PREFIX                - Redis key prefix
    GL_EUDR_DDO_MAX_CONCURRENT_AGENTS           - Max concurrent agents per workflow
    GL_EUDR_DDO_GLOBAL_CONCURRENCY_LIMIT        - Global concurrency limit
    GL_EUDR_DDO_WORKFLOW_TIMEOUT_S              - Workflow timeout seconds
    GL_EUDR_DDO_CHECKPOINT_INTERVAL_S           - Checkpoint interval seconds
    GL_EUDR_DDO_QG1_COMPLETENESS_THRESHOLD      - QG-1 info gathering threshold
    GL_EUDR_DDO_QG1_SIMPLIFIED_THRESHOLD        - QG-1 simplified threshold
    GL_EUDR_DDO_QG2_COVERAGE_THRESHOLD          - QG-2 risk coverage threshold
    GL_EUDR_DDO_QG2_SIMPLIFIED_THRESHOLD        - QG-2 simplified threshold
    GL_EUDR_DDO_QG3_RESIDUAL_RISK_THRESHOLD     - QG-3 residual risk threshold
    GL_EUDR_DDO_QG3_SIMPLIFIED_THRESHOLD        - QG-3 simplified threshold
    GL_EUDR_DDO_RETRY_BASE_DELAY_S              - Retry base delay seconds
    GL_EUDR_DDO_RETRY_MAX_DELAY_S               - Retry max delay seconds
    GL_EUDR_DDO_RETRY_MAX_ATTEMPTS              - Retry max attempts
    GL_EUDR_DDO_RETRY_JITTER_MAX_S              - Retry jitter max seconds
    GL_EUDR_DDO_CB_FAILURE_THRESHOLD            - Circuit breaker failure threshold
    GL_EUDR_DDO_CB_RESET_TIMEOUT_S              - Circuit breaker reset timeout
    GL_EUDR_DDO_CB_SUCCESS_THRESHOLD            - Circuit breaker success threshold
    GL_EUDR_DDO_NEGLIGIBLE_RISK_THRESHOLD       - Negligible risk score threshold
    GL_EUDR_DDO_STANDARD_RISK_THRESHOLD         - Standard risk score threshold
    GL_EUDR_DDO_RESIDUAL_RISK_TARGET            - Residual risk target after mitigation
    GL_EUDR_DDO_W_COUNTRY                       - Weight: country risk
    GL_EUDR_DDO_W_SUPPLIER                      - Weight: supplier risk
    GL_EUDR_DDO_W_COMMODITY                     - Weight: commodity risk
    GL_EUDR_DDO_W_CORRUPTION                    - Weight: corruption risk
    GL_EUDR_DDO_W_DEFORESTATION                 - Weight: deforestation risk
    GL_EUDR_DDO_W_INDIGENOUS                    - Weight: indigenous rights risk
    GL_EUDR_DDO_W_PROTECTED                     - Weight: protected area risk
    GL_EUDR_DDO_W_LEGAL                         - Weight: legal compliance risk
    GL_EUDR_DDO_W_AUDIT                         - Weight: audit risk
    GL_EUDR_DDO_W_MITIGATION                    - Weight: mitigation readiness
    GL_EUDR_DDO_AGENT_BASE_URL                  - Base URL for EUDR agents
    GL_EUDR_DDO_AGENT_TIMEOUT_S                 - Agent call timeout seconds
    GL_EUDR_DDO_RETENTION_YEARS                 - Data retention years
    GL_EUDR_DDO_ENABLE_PROVENANCE               - Enable provenance tracking
    GL_EUDR_DDO_GENESIS_HASH                    - Genesis hash anchor
    GL_EUDR_DDO_CHAIN_ALGORITHM                 - Hash algorithm
    GL_EUDR_DDO_ENABLE_METRICS                  - Enable Prometheus metrics
    GL_EUDR_DDO_METRICS_PREFIX                  - Prometheus metrics prefix
    GL_EUDR_DDO_S3_BUCKET                       - S3 bucket for packages
    GL_EUDR_DDO_S3_PREFIX                       - S3 key prefix for packages
    GL_EUDR_DDO_DDS_SCHEMA_VERSION              - DDS schema version

Example:
    >>> from greenlang.agents.eudr.due_diligence_orchestrator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.max_concurrent_agents, cfg.qg1_completeness_threshold)
    10 0.90

    >>> from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    ...     set_config, reset_config, DueDiligenceOrchestratorConfig,
    ... )
    >>> set_config(DueDiligenceOrchestratorConfig(max_concurrent_agents=5))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_DDO_"

# ---------------------------------------------------------------------------
# Valid constants
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_CHAIN_ALGORITHMS = frozenset({"sha256", "sha384", "sha512"})
_VALID_OUTPUT_FORMATS = frozenset({"pdf", "json", "html", "excel", "csv", "zip"})
_VALID_REPORT_LANGUAGES = frozenset({"en", "fr", "de", "es", "pt"})

# ---------------------------------------------------------------------------
# Default risk assessment weights (sum = 1.00)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    "country": Decimal("0.15"),
    "supplier": Decimal("0.12"),
    "commodity": Decimal("0.10"),
    "corruption": Decimal("0.08"),
    "deforestation": Decimal("0.15"),
    "indigenous": Decimal("0.10"),
    "protected": Decimal("0.10"),
    "legal": Decimal("0.10"),
    "audit": Decimal("0.05"),
    "mitigation": Decimal("0.05"),
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class DueDiligenceOrchestratorConfig:
    """Configuration for the Due Diligence Orchestrator Agent (AGENT-EUDR-026).

    This dataclass encapsulates all configuration settings for workflow
    orchestration, quality gate enforcement, retry/circuit breaker policies,
    risk assessment weight parameters, package generation, and upstream
    agent connectivity. All settings have sensible defaults aligned with
    EUDR Articles 4, 8, 9, 10, 11, 12, and 13 requirements and can be
    overridden via environment variables with the GL_EUDR_DDO_ prefix.

    Attributes:
        database_url: PostgreSQL connection URL.
        pool_size: Connection pool size.
        pool_timeout_s: Connection pool timeout seconds.
        pool_recycle_s: Connection pool recycle seconds.
        redis_url: Redis connection URL.
        redis_ttl_s: Redis cache TTL seconds.
        redis_key_prefix: Redis key prefix for DDO namespace.
        log_level: Logging level.
        max_concurrent_agents: Max concurrent agents per workflow.
        global_concurrency_limit: Global agent concurrency across workflows.
        workflow_timeout_s: Maximum workflow execution time seconds.
        checkpoint_interval_s: Checkpoint write interval seconds.
        max_workflow_age_days: Max days a workflow can remain active.
        qg1_completeness_threshold: QG-1 info gathering completeness (0-1).
        qg1_simplified_threshold: QG-1 simplified due diligence threshold.
        qg2_coverage_threshold: QG-2 risk assessment coverage (0-1).
        qg2_simplified_threshold: QG-2 simplified due diligence threshold.
        qg3_residual_risk_threshold: QG-3 max residual risk (0-100).
        qg3_simplified_threshold: QG-3 simplified due diligence threshold.
        retry_base_delay_s: Exponential backoff base delay seconds.
        retry_max_delay_s: Exponential backoff max delay seconds.
        retry_max_attempts: Maximum retry attempts per agent.
        retry_jitter_max_s: Maximum jitter seconds added to backoff.
        cb_failure_threshold: Circuit breaker failure count threshold.
        cb_reset_timeout_s: Circuit breaker reset timeout seconds.
        cb_success_threshold: Circuit breaker half-open success threshold.
        negligible_risk_threshold: Risk score below which mitigation is skipped.
        standard_risk_threshold: Risk score above which enhanced mitigation applies.
        residual_risk_target: Target residual risk after mitigation (0-100).
        w_country: Weight for country risk in composite score.
        w_supplier: Weight for supplier risk.
        w_commodity: Weight for commodity risk.
        w_corruption: Weight for corruption risk.
        w_deforestation: Weight for deforestation risk.
        w_indigenous: Weight for indigenous rights risk.
        w_protected: Weight for protected area risk.
        w_legal: Weight for legal compliance risk.
        w_audit: Weight for audit risk.
        w_mitigation: Weight for mitigation readiness.
        agent_base_url: Base URL for EUDR agent HTTP endpoints.
        agent_timeout_s: Default timeout for agent HTTP calls.
        agent_endpoints: Per-agent endpoint path overrides.
        output_formats: Report output formats.
        default_language: Default report language.
        supported_languages: Supported report languages.
        s3_bucket: S3 bucket for due diligence packages.
        s3_prefix: S3 key prefix for packages.
        dds_schema_version: EU DDS schema version.
        retention_years: Data retention years per EUDR Article 31.
        enable_provenance: Enable provenance tracking.
        genesis_hash: Genesis hash anchor for provenance chain.
        chain_algorithm: Hash algorithm for provenance chain.
        enable_metrics: Enable Prometheus metrics collection.
        metrics_prefix: Prometheus metrics prefix.
        batch_max_size: Maximum workflows in a batch.
        batch_concurrency: Concurrent workflows in batch processing.
        rate_limit_anonymous: Rate limit anonymous tier.
        rate_limit_basic: Rate limit basic tier.
        rate_limit_standard: Rate limit standard tier.
        rate_limit_premium: Rate limit premium tier.
        rate_limit_admin: Rate limit admin tier.
    """

    # -----------------------------------------------------------------------
    # Database settings
    # -----------------------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    pool_size: int = 20
    pool_timeout_s: int = 30
    pool_recycle_s: int = 3600

    # -----------------------------------------------------------------------
    # Redis settings
    # -----------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_s: int = 3600
    redis_key_prefix: str = "gl:eudr:ddo:"

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = "INFO"

    # -----------------------------------------------------------------------
    # Workflow orchestration settings
    # -----------------------------------------------------------------------
    max_concurrent_agents: int = 10
    global_concurrency_limit: int = 100
    workflow_timeout_s: int = 3600
    checkpoint_interval_s: int = 5
    max_workflow_age_days: int = 30

    # -----------------------------------------------------------------------
    # Quality gate thresholds -- standard due diligence
    # -----------------------------------------------------------------------
    qg1_completeness_threshold: Decimal = Decimal("0.90")
    qg1_simplified_threshold: Decimal = Decimal("0.80")
    qg2_coverage_threshold: Decimal = Decimal("0.95")
    qg2_simplified_threshold: Decimal = Decimal("0.85")
    qg3_residual_risk_threshold: Decimal = Decimal("15")
    qg3_simplified_threshold: Decimal = Decimal("25")

    # -----------------------------------------------------------------------
    # Retry / exponential backoff settings
    # -----------------------------------------------------------------------
    retry_base_delay_s: Decimal = Decimal("1.0")
    retry_max_delay_s: Decimal = Decimal("300.0")
    retry_max_attempts: int = 5
    retry_jitter_max_s: Decimal = Decimal("2.0")

    # -----------------------------------------------------------------------
    # Circuit breaker settings
    # -----------------------------------------------------------------------
    cb_failure_threshold: int = 5
    cb_reset_timeout_s: int = 60
    cb_success_threshold: int = 2

    # -----------------------------------------------------------------------
    # Risk mitigation thresholds
    # -----------------------------------------------------------------------
    negligible_risk_threshold: Decimal = Decimal("20")
    standard_risk_threshold: Decimal = Decimal("50")
    residual_risk_target: Decimal = Decimal("15")

    # -----------------------------------------------------------------------
    # Risk assessment weights (must sum to 1.00)
    # -----------------------------------------------------------------------
    w_country: Decimal = _DEFAULT_WEIGHTS["country"]
    w_supplier: Decimal = _DEFAULT_WEIGHTS["supplier"]
    w_commodity: Decimal = _DEFAULT_WEIGHTS["commodity"]
    w_corruption: Decimal = _DEFAULT_WEIGHTS["corruption"]
    w_deforestation: Decimal = _DEFAULT_WEIGHTS["deforestation"]
    w_indigenous: Decimal = _DEFAULT_WEIGHTS["indigenous"]
    w_protected: Decimal = _DEFAULT_WEIGHTS["protected"]
    w_legal: Decimal = _DEFAULT_WEIGHTS["legal"]
    w_audit: Decimal = _DEFAULT_WEIGHTS["audit"]
    w_mitigation: Decimal = _DEFAULT_WEIGHTS["mitigation"]

    # -----------------------------------------------------------------------
    # Agent connectivity
    # -----------------------------------------------------------------------
    agent_base_url: str = "http://localhost:8000/api/v1/eudr"
    agent_timeout_s: int = 120
    agent_endpoints: Dict[str, str] = field(default_factory=lambda: {
        "EUDR-001": "/supply-chain-mapping/execute",
        "EUDR-002": "/geolocation-verification/execute",
        "EUDR-003": "/satellite-monitoring/execute",
        "EUDR-004": "/forest-cover-analysis/execute",
        "EUDR-005": "/land-use-change/execute",
        "EUDR-006": "/plot-boundary/execute",
        "EUDR-007": "/gps-validation/execute",
        "EUDR-008": "/multi-tier-supplier/execute",
        "EUDR-009": "/chain-of-custody/execute",
        "EUDR-010": "/segregation-verifier/execute",
        "EUDR-011": "/mass-balance/execute",
        "EUDR-012": "/document-authentication/execute",
        "EUDR-013": "/blockchain-integration/execute",
        "EUDR-014": "/qr-code-generator/execute",
        "EUDR-015": "/mobile-data-collector/execute",
        "EUDR-016": "/country-risk/execute",
        "EUDR-017": "/supplier-risk/execute",
        "EUDR-018": "/commodity-risk/execute",
        "EUDR-019": "/corruption-index/execute",
        "EUDR-020": "/deforestation-alert/execute",
        "EUDR-021": "/indigenous-rights/execute",
        "EUDR-022": "/protected-area/execute",
        "EUDR-023": "/legal-compliance/execute",
        "EUDR-024": "/third-party-audit/execute",
        "EUDR-025": "/risk-mitigation/execute",
    })

    # -----------------------------------------------------------------------
    # Package generation / reporting
    # -----------------------------------------------------------------------
    output_formats: List[str] = field(
        default_factory=lambda: ["pdf", "json", "html", "zip"]
    )
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt"]
    )
    s3_bucket: str = "greenlang-eudr-packages"
    s3_prefix: str = "ddo/packages/"
    dds_schema_version: str = "1.0.0"

    # -----------------------------------------------------------------------
    # Data retention
    # -----------------------------------------------------------------------
    retention_years: int = 5

    # -----------------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-DDO-026-DUE-DILIGENCE-ORCHESTRATOR-GENESIS"
    chain_algorithm: str = "sha256"

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    enable_metrics: bool = True
    metrics_prefix: str = "gl_eudr_ddo_"

    # -----------------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 10

    # -----------------------------------------------------------------------
    # Rate limiting (requests per minute)
    # -----------------------------------------------------------------------
    rate_limit_anonymous: int = 10
    rate_limit_basic: int = 60
    rate_limit_standard: int = 300
    rate_limit_premium: int = 1000
    rate_limit_admin: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_log_level()
        self._validate_chain_algorithm()
        self._validate_concurrency()
        self._validate_quality_gates()
        self._validate_retry_settings()
        self._validate_circuit_breaker()
        self._validate_risk_weights()
        self._validate_risk_thresholds()
        self._validate_positive_integers()
        self._validate_output_formats()
        self._validate_languages()

        logger.info(
            f"DueDiligenceOrchestratorConfig initialized: "
            f"max_concurrent_agents={self.max_concurrent_agents}, "
            f"qg1_threshold={self.qg1_completeness_threshold}, "
            f"retry_max={self.retry_max_attempts}"
        )

    # -----------------------------------------------------------------------
    # Validation methods
    # -----------------------------------------------------------------------

    def _validate_log_level(self) -> None:
        """Validate log_level is a recognized Python logging level."""
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: {self.log_level}. "
                f"Must be one of {sorted(_VALID_LOG_LEVELS)}"
            )

    def _validate_chain_algorithm(self) -> None:
        """Validate chain_algorithm is a supported hash algorithm."""
        if self.chain_algorithm not in _VALID_CHAIN_ALGORITHMS:
            raise ValueError(
                f"Invalid chain_algorithm: {self.chain_algorithm}. "
                f"Must be one of {sorted(_VALID_CHAIN_ALGORITHMS)}"
            )

    def _validate_concurrency(self) -> None:
        """Validate workflow concurrency parameters."""
        if self.max_concurrent_agents < 1:
            raise ValueError(
                f"max_concurrent_agents must be >= 1, "
                f"got {self.max_concurrent_agents}"
            )
        if self.max_concurrent_agents > 25:
            raise ValueError(
                f"max_concurrent_agents must be <= 25 (total EUDR agents), "
                f"got {self.max_concurrent_agents}"
            )
        if self.global_concurrency_limit < self.max_concurrent_agents:
            raise ValueError(
                f"global_concurrency_limit ({self.global_concurrency_limit}) "
                f"must be >= max_concurrent_agents ({self.max_concurrent_agents})"
            )
        if self.workflow_timeout_s < 60:
            raise ValueError(
                f"workflow_timeout_s must be >= 60, got {self.workflow_timeout_s}"
            )

    def _validate_quality_gates(self) -> None:
        """Validate quality gate threshold parameters."""
        for name, val in [
            ("qg1_completeness_threshold", self.qg1_completeness_threshold),
            ("qg1_simplified_threshold", self.qg1_simplified_threshold),
            ("qg2_coverage_threshold", self.qg2_coverage_threshold),
            ("qg2_simplified_threshold", self.qg2_simplified_threshold),
        ]:
            if not Decimal("0") < val <= Decimal("1"):
                raise ValueError(
                    f"{name} must be between 0 (exclusive) and 1 (inclusive), "
                    f"got {val}"
                )
        for name, val in [
            ("qg3_residual_risk_threshold", self.qg3_residual_risk_threshold),
            ("qg3_simplified_threshold", self.qg3_simplified_threshold),
        ]:
            if not Decimal("0") <= val <= Decimal("100"):
                raise ValueError(
                    f"{name} must be between 0 and 100, got {val}"
                )
        # Simplified thresholds should be more relaxed
        if self.qg1_simplified_threshold > self.qg1_completeness_threshold:
            raise ValueError(
                f"qg1_simplified_threshold ({self.qg1_simplified_threshold}) "
                f"should be <= qg1_completeness_threshold "
                f"({self.qg1_completeness_threshold})"
            )
        if self.qg2_simplified_threshold > self.qg2_coverage_threshold:
            raise ValueError(
                f"qg2_simplified_threshold ({self.qg2_simplified_threshold}) "
                f"should be <= qg2_coverage_threshold "
                f"({self.qg2_coverage_threshold})"
            )

    def _validate_retry_settings(self) -> None:
        """Validate retry and exponential backoff parameters."""
        if self.retry_base_delay_s <= Decimal("0"):
            raise ValueError(
                f"retry_base_delay_s must be > 0, got {self.retry_base_delay_s}"
            )
        if self.retry_max_delay_s < self.retry_base_delay_s:
            raise ValueError(
                f"retry_max_delay_s ({self.retry_max_delay_s}) must be >= "
                f"retry_base_delay_s ({self.retry_base_delay_s})"
            )
        if self.retry_max_attempts < 1:
            raise ValueError(
                f"retry_max_attempts must be >= 1, got {self.retry_max_attempts}"
            )
        if self.retry_jitter_max_s < Decimal("0"):
            raise ValueError(
                f"retry_jitter_max_s must be >= 0, got {self.retry_jitter_max_s}"
            )

    def _validate_circuit_breaker(self) -> None:
        """Validate circuit breaker parameters."""
        if self.cb_failure_threshold < 1:
            raise ValueError(
                f"cb_failure_threshold must be >= 1, "
                f"got {self.cb_failure_threshold}"
            )
        if self.cb_reset_timeout_s < 1:
            raise ValueError(
                f"cb_reset_timeout_s must be >= 1, "
                f"got {self.cb_reset_timeout_s}"
            )
        if self.cb_success_threshold < 1:
            raise ValueError(
                f"cb_success_threshold must be >= 1, "
                f"got {self.cb_success_threshold}"
            )

    def _validate_risk_weights(self) -> None:
        """Validate risk assessment weights sum to 1.00."""
        weights = [
            ("w_country", self.w_country),
            ("w_supplier", self.w_supplier),
            ("w_commodity", self.w_commodity),
            ("w_corruption", self.w_corruption),
            ("w_deforestation", self.w_deforestation),
            ("w_indigenous", self.w_indigenous),
            ("w_protected", self.w_protected),
            ("w_legal", self.w_legal),
            ("w_audit", self.w_audit),
            ("w_mitigation", self.w_mitigation),
        ]
        for name, w in weights:
            if w < Decimal("0") or w > Decimal("1"):
                raise ValueError(
                    f"{name} must be between 0 and 1, got {w}"
                )
        total = sum(w for _, w in weights)
        if abs(total - Decimal("1")) > Decimal("0.001"):
            raise ValueError(
                f"Risk assessment weights must sum to 1.00, got {total}"
            )

    def _validate_risk_thresholds(self) -> None:
        """Validate risk mitigation thresholds are ordered correctly."""
        if not (Decimal("0") <= self.residual_risk_target
                < self.negligible_risk_threshold
                <= self.standard_risk_threshold
                <= Decimal("100")):
            raise ValueError(
                f"Risk thresholds must satisfy: "
                f"0 <= residual_target < negligible <= standard <= 100. "
                f"Got residual={self.residual_risk_target}, "
                f"negligible={self.negligible_risk_threshold}, "
                f"standard={self.standard_risk_threshold}"
            )

    def _validate_positive_integers(self) -> None:
        """Validate fields that must be positive integers."""
        checks = [
            ("pool_size", self.pool_size),
            ("retention_years", self.retention_years),
            ("batch_max_size", self.batch_max_size),
            ("batch_concurrency", self.batch_concurrency),
            ("agent_timeout_s", self.agent_timeout_s),
        ]
        for name, val in checks:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

    def _validate_output_formats(self) -> None:
        """Validate output formats against allowed set."""
        for fmt in self.output_formats:
            if fmt not in _VALID_OUTPUT_FORMATS:
                raise ValueError(
                    f"Invalid output format: {fmt}. "
                    f"Must be one of {sorted(_VALID_OUTPUT_FORMATS)}"
                )

    def _validate_languages(self) -> None:
        """Validate language settings against allowed set."""
        if self.default_language not in _VALID_REPORT_LANGUAGES:
            raise ValueError(
                f"Invalid default_language: {self.default_language}. "
                f"Must be one of {sorted(_VALID_REPORT_LANGUAGES)}"
            )
        for lang in self.supported_languages:
            if lang not in _VALID_REPORT_LANGUAGES:
                raise ValueError(
                    f"Invalid supported language: {lang}. "
                    f"Must be one of {sorted(_VALID_REPORT_LANGUAGES)}"
                )

    def get_weight_dict(self) -> Dict[str, Decimal]:
        """Return risk assessment weights as a dictionary.

        Returns:
            Dictionary mapping risk dimension names to their weights.

        Example:
            >>> cfg = DueDiligenceOrchestratorConfig()
            >>> weights = cfg.get_weight_dict()
            >>> assert weights["country"] == Decimal("0.15")
        """
        return {
            "country": self.w_country,
            "supplier": self.w_supplier,
            "commodity": self.w_commodity,
            "corruption": self.w_corruption,
            "deforestation": self.w_deforestation,
            "indigenous": self.w_indigenous,
            "protected": self.w_protected,
            "legal": self.w_legal,
            "audit": self.w_audit,
            "mitigation": self.w_mitigation,
        }

    def get_agent_url(self, agent_id: str) -> str:
        """Build full URL for a specific EUDR agent.

        Args:
            agent_id: Agent identifier (e.g. "EUDR-001").

        Returns:
            Full HTTP URL for the agent endpoint.

        Raises:
            ValueError: If agent_id is not recognized.

        Example:
            >>> cfg = DueDiligenceOrchestratorConfig()
            >>> url = cfg.get_agent_url("EUDR-001")
            >>> assert "/supply-chain-mapping/execute" in url
        """
        path = self.agent_endpoints.get(agent_id)
        if path is None:
            raise ValueError(
                f"Unknown agent_id: {agent_id}. "
                f"Known agents: {sorted(self.agent_endpoints.keys())}"
            )
        return f"{self.agent_base_url.rstrip('/')}{path}"

    @classmethod
    def from_env(cls) -> "DueDiligenceOrchestratorConfig":
        """Create configuration from environment variables.

        Reads all GL_EUDR_DDO_* environment variables and overrides
        default values. Non-existent variables use defaults.

        Returns:
            DueDiligenceOrchestratorConfig instance with env overrides applied.

        Example:
            >>> import os
            >>> os.environ["GL_EUDR_DDO_MAX_CONCURRENT_AGENTS"] = "5"
            >>> cfg = DueDiligenceOrchestratorConfig.from_env()
            >>> assert cfg.max_concurrent_agents == 5
        """
        kwargs: Dict[str, Any] = {}

        # Database settings
        if val := os.getenv(f"{_ENV_PREFIX}DATABASE_URL"):
            kwargs["database_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}POOL_SIZE"):
            kwargs["pool_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_TIMEOUT_S"):
            kwargs["pool_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}POOL_RECYCLE_S"):
            kwargs["pool_recycle_s"] = int(val)

        # Redis settings
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_URL"):
            kwargs["redis_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_TTL_S"):
            kwargs["redis_ttl_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}REDIS_KEY_PREFIX"):
            kwargs["redis_key_prefix"] = val

        # Logging
        if val := os.getenv(f"{_ENV_PREFIX}LOG_LEVEL"):
            kwargs["log_level"] = val.upper()

        # Workflow orchestration
        if val := os.getenv(f"{_ENV_PREFIX}MAX_CONCURRENT_AGENTS"):
            kwargs["max_concurrent_agents"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}GLOBAL_CONCURRENCY_LIMIT"):
            kwargs["global_concurrency_limit"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}WORKFLOW_TIMEOUT_S"):
            kwargs["workflow_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CHECKPOINT_INTERVAL_S"):
            kwargs["checkpoint_interval_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}MAX_WORKFLOW_AGE_DAYS"):
            kwargs["max_workflow_age_days"] = int(val)

        # Quality gate thresholds
        if val := os.getenv(f"{_ENV_PREFIX}QG1_COMPLETENESS_THRESHOLD"):
            kwargs["qg1_completeness_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}QG1_SIMPLIFIED_THRESHOLD"):
            kwargs["qg1_simplified_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}QG2_COVERAGE_THRESHOLD"):
            kwargs["qg2_coverage_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}QG2_SIMPLIFIED_THRESHOLD"):
            kwargs["qg2_simplified_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}QG3_RESIDUAL_RISK_THRESHOLD"):
            kwargs["qg3_residual_risk_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}QG3_SIMPLIFIED_THRESHOLD"):
            kwargs["qg3_simplified_threshold"] = Decimal(val)

        # Retry settings
        if val := os.getenv(f"{_ENV_PREFIX}RETRY_BASE_DELAY_S"):
            kwargs["retry_base_delay_s"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RETRY_MAX_DELAY_S"):
            kwargs["retry_max_delay_s"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RETRY_MAX_ATTEMPTS"):
            kwargs["retry_max_attempts"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RETRY_JITTER_MAX_S"):
            kwargs["retry_jitter_max_s"] = Decimal(val)

        # Circuit breaker settings
        if val := os.getenv(f"{_ENV_PREFIX}CB_FAILURE_THRESHOLD"):
            kwargs["cb_failure_threshold"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CB_RESET_TIMEOUT_S"):
            kwargs["cb_reset_timeout_s"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}CB_SUCCESS_THRESHOLD"):
            kwargs["cb_success_threshold"] = int(val)

        # Risk thresholds
        if val := os.getenv(f"{_ENV_PREFIX}NEGLIGIBLE_RISK_THRESHOLD"):
            kwargs["negligible_risk_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}STANDARD_RISK_THRESHOLD"):
            kwargs["standard_risk_threshold"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}RESIDUAL_RISK_TARGET"):
            kwargs["residual_risk_target"] = Decimal(val)

        # Risk weights
        if val := os.getenv(f"{_ENV_PREFIX}W_COUNTRY"):
            kwargs["w_country"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_SUPPLIER"):
            kwargs["w_supplier"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_COMMODITY"):
            kwargs["w_commodity"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_CORRUPTION"):
            kwargs["w_corruption"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_DEFORESTATION"):
            kwargs["w_deforestation"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_INDIGENOUS"):
            kwargs["w_indigenous"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_PROTECTED"):
            kwargs["w_protected"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_LEGAL"):
            kwargs["w_legal"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_AUDIT"):
            kwargs["w_audit"] = Decimal(val)
        if val := os.getenv(f"{_ENV_PREFIX}W_MITIGATION"):
            kwargs["w_mitigation"] = Decimal(val)

        # Agent connectivity
        if val := os.getenv(f"{_ENV_PREFIX}AGENT_BASE_URL"):
            kwargs["agent_base_url"] = val
        if val := os.getenv(f"{_ENV_PREFIX}AGENT_TIMEOUT_S"):
            kwargs["agent_timeout_s"] = int(val)

        # Package generation
        if val := os.getenv(f"{_ENV_PREFIX}OUTPUT_FORMATS"):
            kwargs["output_formats"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}DEFAULT_LANGUAGE"):
            kwargs["default_language"] = val
        if val := os.getenv(f"{_ENV_PREFIX}SUPPORTED_LANGUAGES"):
            kwargs["supported_languages"] = [x.strip() for x in val.split(",")]
        if val := os.getenv(f"{_ENV_PREFIX}S3_BUCKET"):
            kwargs["s3_bucket"] = val
        if val := os.getenv(f"{_ENV_PREFIX}S3_PREFIX"):
            kwargs["s3_prefix"] = val
        if val := os.getenv(f"{_ENV_PREFIX}DDS_SCHEMA_VERSION"):
            kwargs["dds_schema_version"] = val

        # Data retention
        if val := os.getenv(f"{_ENV_PREFIX}RETENTION_YEARS"):
            kwargs["retention_years"] = int(val)

        # Provenance
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_PROVENANCE"):
            kwargs["enable_provenance"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}GENESIS_HASH"):
            kwargs["genesis_hash"] = val
        if val := os.getenv(f"{_ENV_PREFIX}CHAIN_ALGORITHM"):
            kwargs["chain_algorithm"] = val

        # Metrics
        if val := os.getenv(f"{_ENV_PREFIX}ENABLE_METRICS"):
            kwargs["enable_metrics"] = val.lower() in ("true", "1", "yes")
        if val := os.getenv(f"{_ENV_PREFIX}METRICS_PREFIX"):
            kwargs["metrics_prefix"] = val

        # Batch processing
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_MAX_SIZE"):
            kwargs["batch_max_size"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}BATCH_CONCURRENCY"):
            kwargs["batch_concurrency"] = int(val)

        # Rate limiting
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_ANONYMOUS"):
            kwargs["rate_limit_anonymous"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_BASIC"):
            kwargs["rate_limit_basic"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_STANDARD"):
            kwargs["rate_limit_standard"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_PREMIUM"):
            kwargs["rate_limit_premium"] = int(val)
        if val := os.getenv(f"{_ENV_PREFIX}RATE_LIMIT_ADMIN"):
            kwargs["rate_limit_admin"] = int(val)

        return cls(**kwargs)

    def to_dict(self, redact_secrets: bool = True) -> Dict[str, Any]:
        """Export configuration as a dictionary.

        Args:
            redact_secrets: If True, redact sensitive fields like database_url.

        Returns:
            Dictionary representation of configuration.

        Example:
            >>> cfg = DueDiligenceOrchestratorConfig()
            >>> d = cfg.to_dict(redact_secrets=True)
            >>> assert "REDACTED" in d["database_url"]
        """
        data: Dict[str, Any] = {
            "database_url": self.database_url,
            "pool_size": self.pool_size,
            "pool_timeout_s": self.pool_timeout_s,
            "pool_recycle_s": self.pool_recycle_s,
            "redis_url": self.redis_url,
            "redis_ttl_s": self.redis_ttl_s,
            "redis_key_prefix": self.redis_key_prefix,
            "log_level": self.log_level,
            "max_concurrent_agents": self.max_concurrent_agents,
            "global_concurrency_limit": self.global_concurrency_limit,
            "workflow_timeout_s": self.workflow_timeout_s,
            "checkpoint_interval_s": self.checkpoint_interval_s,
            "max_workflow_age_days": self.max_workflow_age_days,
            "qg1_completeness_threshold": str(self.qg1_completeness_threshold),
            "qg1_simplified_threshold": str(self.qg1_simplified_threshold),
            "qg2_coverage_threshold": str(self.qg2_coverage_threshold),
            "qg2_simplified_threshold": str(self.qg2_simplified_threshold),
            "qg3_residual_risk_threshold": str(self.qg3_residual_risk_threshold),
            "qg3_simplified_threshold": str(self.qg3_simplified_threshold),
            "retry_base_delay_s": str(self.retry_base_delay_s),
            "retry_max_delay_s": str(self.retry_max_delay_s),
            "retry_max_attempts": self.retry_max_attempts,
            "retry_jitter_max_s": str(self.retry_jitter_max_s),
            "cb_failure_threshold": self.cb_failure_threshold,
            "cb_reset_timeout_s": self.cb_reset_timeout_s,
            "cb_success_threshold": self.cb_success_threshold,
            "negligible_risk_threshold": str(self.negligible_risk_threshold),
            "standard_risk_threshold": str(self.standard_risk_threshold),
            "residual_risk_target": str(self.residual_risk_target),
            "w_country": str(self.w_country),
            "w_supplier": str(self.w_supplier),
            "w_commodity": str(self.w_commodity),
            "w_corruption": str(self.w_corruption),
            "w_deforestation": str(self.w_deforestation),
            "w_indigenous": str(self.w_indigenous),
            "w_protected": str(self.w_protected),
            "w_legal": str(self.w_legal),
            "w_audit": str(self.w_audit),
            "w_mitigation": str(self.w_mitigation),
            "agent_base_url": self.agent_base_url,
            "agent_timeout_s": self.agent_timeout_s,
            "output_formats": self.output_formats,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "dds_schema_version": self.dds_schema_version,
            "retention_years": self.retention_years,
            "enable_provenance": self.enable_provenance,
            "chain_algorithm": self.chain_algorithm,
            "enable_metrics": self.enable_metrics,
            "metrics_prefix": self.metrics_prefix,
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
        }

        if redact_secrets:
            if "://" in data["database_url"]:
                data["database_url"] = "REDACTED"
            if "://" in data["redis_url"]:
                data["redis_url"] = "REDACTED"

        return data


# ---------------------------------------------------------------------------
# Thread-safe singleton pattern (double-checked locking)
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_global_config: Optional[DueDiligenceOrchestratorConfig] = None


def get_config() -> DueDiligenceOrchestratorConfig:
    """Get the global DueDiligenceOrchestratorConfig singleton instance.

    Thread-safe lazy initialization from environment variables on first call.
    Subsequent calls return the same instance. Uses double-checked locking
    to minimize contention after initialization.

    Returns:
        DueDiligenceOrchestratorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> assert cfg.max_concurrent_agents == 10
        >>> cfg2 = get_config()
        >>> assert cfg is cfg2
    """
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = DueDiligenceOrchestratorConfig.from_env()
    return _global_config


def set_config(config: DueDiligenceOrchestratorConfig) -> None:
    """Set the global DueDiligenceOrchestratorConfig singleton instance.

    Used for testing and programmatic configuration override.

    Args:
        config: DueDiligenceOrchestratorConfig instance to set as global.

    Example:
        >>> test_cfg = DueDiligenceOrchestratorConfig(max_concurrent_agents=5)
        >>> set_config(test_cfg)
    """
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global DueDiligenceOrchestratorConfig singleton to None.

    Used for testing teardown to ensure clean state between tests.

    Example:
        >>> reset_config()
        >>> # Next get_config() call will re-initialize from environment
    """
    global _global_config
    with _config_lock:
        _global_config = None
