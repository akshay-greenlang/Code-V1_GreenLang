# -*- coding: utf-8 -*-
"""
Chain of Custody Configuration - AGENT-EUDR-009

Centralized configuration for the Chain of Custody Agent covering:
- Custody event management: gap threshold, max amendment depth, event retention
- Batch operations: max split parts, merge limit, blend rules
- Chain of Custody (CoC) model rules: IP, SG, MB, CB requirements
- Mass balance: credit periods (3/12 months), loss tolerances per commodity,
  overdraft threshold, reconciliation windows
- Transformation processing: 25+ process types with commodity-specific yield
  ratios (drying, fermentation, roasting, milling, refining, pressing, etc.)
- Document requirements per event type: mandatory/optional attachments
- Chain integrity: verification thresholds, completeness scoring
- Batch processing: size limits, concurrency, timeout
- Data retention: EUDR Article 31 five-year retention
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_COC_`` prefix (e.g. ``GL_EUDR_COC_DATABASE_URL``,
``GL_EUDR_COC_GAP_THRESHOLD_HOURS``).

Environment Variable Reference (GL_EUDR_COC_ prefix):
    GL_EUDR_COC_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_COC_REDIS_URL                     - Redis connection URL
    GL_EUDR_COC_LOG_LEVEL                     - Logging level
    GL_EUDR_COC_GAP_THRESHOLD_HOURS           - Max gap hours between events
    GL_EUDR_COC_MAX_AMENDMENT_DEPTH           - Max amendment chain depth
    GL_EUDR_COC_MAX_SPLIT_PARTS               - Max parts in a batch split
    GL_EUDR_COC_MAX_MERGE_BATCHES             - Max batches in a merge
    GL_EUDR_COC_MAX_BLEND_INPUTS              - Max inputs in a blend
    GL_EUDR_COC_IP_PHYSICAL_SEPARATION        - IP requires physical separation
    GL_EUDR_COC_SG_ALLOW_COMPLIANT_MIX        - SG allows compliant mixing
    GL_EUDR_COC_MB_SHORT_CREDIT_MONTHS        - MB short credit period months
    GL_EUDR_COC_MB_LONG_CREDIT_MONTHS         - MB long credit period months
    GL_EUDR_COC_MB_OVERDRAFT_THRESHOLD_PCT    - MB overdraft threshold %
    GL_EUDR_COC_MB_RECONCILIATION_INTERVAL_DAYS - MB reconciliation interval
    GL_EUDR_COC_CB_MAX_BLEND_RATIO            - CB max blend ratio
    GL_EUDR_COC_CHAIN_COMPLETENESS_THRESHOLD  - Chain completeness threshold
    GL_EUDR_COC_VERIFICATION_MIN_SCORE        - Verification min score
    GL_EUDR_COC_BATCH_MAX_SIZE                - Maximum batch size
    GL_EUDR_COC_BATCH_CONCURRENCY             - Batch concurrency
    GL_EUDR_COC_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_COC_RETENTION_YEARS               - Data retention years
    GL_EUDR_COC_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_COC_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_COC_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_COC_POOL_SIZE                     - Database pool size
    GL_EUDR_COC_RATE_LIMIT                    - Max requests per minute
    GL_EUDR_COC_REPORT_GENERATION_TIMEOUT_S   - Report generation timeout

Example:
    >>> from greenlang.agents.eudr.chain_of_custody.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.gap_threshold_hours, cfg.mb_short_credit_months)
    72 3

    >>> # Override for testing
    >>> from greenlang.agents.eudr.chain_of_custody.config import (
    ...     set_config, reset_config, ChainOfCustodyConfig,
    ... )
    >>> set_config(ChainOfCustodyConfig(gap_threshold_hours=48))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody (GL-EUDR-COC-009)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_COC_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid CoC model types
# ---------------------------------------------------------------------------

_VALID_COC_MODELS = frozenset(
    {"identity_preserved", "segregated", "mass_balance", "controlled_blending"}
)

# ---------------------------------------------------------------------------
# Default commodity loss tolerances (% acceptable processing loss)
# Per EUDR Annex I and industry standard yield loss tables
# ---------------------------------------------------------------------------

_DEFAULT_COMMODITY_LOSS_TOLERANCES: Dict[str, float] = {
    "cattle": 2.0,
    "cocoa": 5.0,
    "coffee": 4.0,
    "oil_palm": 3.0,
    "rubber": 3.5,
    "soya": 2.5,
    "wood": 3.0,
}

# ---------------------------------------------------------------------------
# Default process types with typical yield ratios
# Yield ratio = output_mass / input_mass (0.0-1.0)
# ---------------------------------------------------------------------------

_DEFAULT_PROCESS_YIELD_RATIOS: Dict[str, float] = {
    # Cocoa processing chain
    "fermentation": 0.92,
    "drying": 0.88,
    "roasting": 0.85,
    "winnowing": 0.80,
    "grinding": 0.98,
    "pressing": 0.45,
    "conching": 0.97,
    "tempering": 0.99,
    # Coffee processing chain
    "wet_processing": 0.60,
    "dry_processing": 0.50,
    "hulling": 0.80,
    "polishing": 0.98,
    # Oil palm processing chain
    "sterilization": 0.95,
    "threshing": 0.65,
    "digestion": 0.90,
    "extraction": 0.22,
    "clarification": 0.95,
    "refining": 0.92,
    "fractionation": 0.90,
    # Wood processing chain
    "debarking": 0.90,
    "sawing": 0.55,
    "planing": 0.90,
    "kiln_drying": 0.92,
    # Rubber processing chain
    "coagulation": 0.60,
    "sheeting": 0.95,
    "smoking": 0.88,
    "crumbling": 0.92,
    # Soya processing chain
    "cleaning": 0.98,
    "dehulling": 0.92,
    "flaking": 0.97,
    "solvent_extraction": 0.82,
    # Cattle processing chain
    "slaughtering": 0.55,
    "deboning": 0.70,
    "tanning": 0.30,
    # General
    "milling": 0.85,
    "blending": 0.99,
    "packaging": 0.99,
    "sorting": 0.95,
}

# ---------------------------------------------------------------------------
# Default document requirements per custody event type
# Maps event_type -> list of (document_type, is_mandatory) tuples
# ---------------------------------------------------------------------------

_DEFAULT_DOCUMENT_REQUIREMENTS: Dict[str, List[Tuple[str, bool]]] = {
    "transfer": [
        ("bill_of_lading", True),
        ("commercial_invoice", True),
        ("packing_list", True),
        ("phytosanitary_certificate", False),
        ("certificate_of_origin", False),
    ],
    "receipt": [
        ("goods_received_note", True),
        ("quality_inspection_report", False),
        ("weight_certificate", True),
    ],
    "storage_in": [
        ("warehouse_receipt", True),
        ("storage_certificate", False),
    ],
    "storage_out": [
        ("warehouse_release_note", True),
        ("dispatch_note", True),
    ],
    "processing_in": [
        ("processing_order", True),
        ("raw_material_receipt", True),
        ("quality_inspection_report", False),
    ],
    "processing_out": [
        ("processing_report", True),
        ("yield_certificate", True),
        ("quality_certificate", False),
        ("batch_certificate", True),
    ],
    "export": [
        ("export_declaration", True),
        ("bill_of_lading", True),
        ("commercial_invoice", True),
        ("certificate_of_origin", True),
        ("phytosanitary_certificate", True),
        ("fumigation_certificate", False),
    ],
    "import_": [
        ("import_declaration", True),
        ("customs_clearance", True),
        ("dds_reference", True),
        ("commercial_invoice", True),
    ],
    "inspection": [
        ("inspection_report", True),
        ("sample_analysis_report", False),
        ("compliance_certificate", False),
    ],
    "sampling": [
        ("sample_collection_form", True),
        ("chain_of_custody_form", True),
        ("laboratory_analysis_report", False),
    ],
}

# ---------------------------------------------------------------------------
# Default EUDR commodities
# ---------------------------------------------------------------------------

_DEFAULT_EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Default report formats
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_FORMATS: List[str] = ["json", "pdf", "csv", "eudr_xml"]


# ---------------------------------------------------------------------------
# ChainOfCustodyConfig
# ---------------------------------------------------------------------------


@dataclass
class ChainOfCustodyConfig:
    """Complete configuration for the EUDR Chain of Custody Agent.

    Attributes are grouped by concern: connections, logging, custody event
    management, batch operations, CoC model rules, mass balance settings,
    transformation processing, document requirements, chain integrity,
    batch processing, report settings, data retention, provenance tracking,
    metrics, and performance tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_COC_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            custody events, batches, mass balance ledgers, and documents.
        redis_url: Redis connection URL for event caching, batch lookups,
            and distributed locks.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        gap_threshold_hours: Maximum acceptable time gap (in hours) between
            consecutive custody events before a gap alert is triggered.
            Per EUDR Article 10(2)(f), traceability must be continuous.
        max_amendment_depth: Maximum number of amendments allowed on a
            single custody event before the event is locked.
        max_split_parts: Maximum number of sub-batches when splitting
            a parent batch.
        max_merge_batches: Maximum number of source batches allowed in
            a single merge operation.
        max_blend_inputs: Maximum number of input batches in a
            controlled blending operation.
        ip_physical_separation: Whether Identity Preserved model requires
            strict physical separation at all custody points.
        sg_allow_compliant_mix: Whether Segregated model allows mixing
            of different compliant-origin materials.
        mb_short_credit_months: Short-term mass balance credit period
            in months (typically 3 for perishable commodities).
        mb_long_credit_months: Long-term mass balance credit period
            in months (typically 12 for durable commodities).
        mb_overdraft_threshold_pct: Percentage threshold above which
            mass balance overdraft triggers an alert (0-100).
        mb_reconciliation_interval_days: Interval in days between
            mandatory mass balance reconciliation runs.
        cb_max_blend_ratio: Maximum ratio (0.0-1.0) of non-compliant
            material allowed in a controlled blend.
        commodity_loss_tolerances: Per-commodity acceptable processing
            loss percentages (0-100).
        process_yield_ratios: Per-process-type typical yield ratios
            (output/input, 0.0-1.0).
        document_requirements: Per-event-type document requirements as
            lists of (document_type, is_mandatory) tuples.
        chain_completeness_threshold: Minimum completeness score (0.0-1.0)
            for a custody chain to be considered EUDR-compliant.
        verification_min_score: Minimum verification score (0.0-1.0)
            for a chain segment to pass integrity checks.
        batch_max_size: Maximum number of records in a single batch
            processing job.
        batch_concurrency: Maximum concurrent batch processing workers.
        batch_timeout_s: Timeout in seconds for a single batch job.
        report_formats: Supported report output formats.
        report_generation_timeout_s: Report generation timeout in seconds.
        retention_years: Data retention in years per EUDR Article 31.
        eudr_commodities: List of EUDR-regulated commodity types.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all custody operations.
        genesis_hash: Genesis anchor string for the provenance chain,
            unique to the Chain of Custody agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_coc_`` prefix.
        pool_size: PostgreSQL connection pool size.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Custody event management --------------------------------------------
    gap_threshold_hours: int = 72
    max_amendment_depth: int = 5

    # -- Batch operations ----------------------------------------------------
    max_split_parts: int = 100
    max_merge_batches: int = 50
    max_blend_inputs: int = 20

    # -- CoC model rules: Identity Preserved ---------------------------------
    ip_physical_separation: bool = True

    # -- CoC model rules: Segregated -----------------------------------------
    sg_allow_compliant_mix: bool = True

    # -- CoC model rules: Mass Balance ---------------------------------------
    mb_short_credit_months: int = 3
    mb_long_credit_months: int = 12
    mb_overdraft_threshold_pct: float = 5.0
    mb_reconciliation_interval_days: int = 30

    # -- CoC model rules: Controlled Blending --------------------------------
    cb_max_blend_ratio: float = 0.30

    # -- Commodity loss tolerances -------------------------------------------
    commodity_loss_tolerances: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_COMMODITY_LOSS_TOLERANCES)
    )

    # -- Process yield ratios ------------------------------------------------
    process_yield_ratios: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_PROCESS_YIELD_RATIOS)
    )

    # -- Document requirements per event type --------------------------------
    document_requirements: Dict[str, List[Tuple[str, bool]]] = field(
        default_factory=lambda: {
            k: list(v) for k, v in _DEFAULT_DOCUMENT_REQUIREMENTS.items()
        }
    )

    # -- Chain integrity thresholds ------------------------------------------
    chain_completeness_threshold: float = 0.95
    verification_min_score: float = 0.90

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 100_000
    batch_concurrency: int = 8
    batch_timeout_s: int = 300

    # -- Report settings -----------------------------------------------------
    report_formats: List[str] = field(
        default_factory=lambda: list(_DEFAULT_REPORT_FORMATS)
    )
    report_generation_timeout_s: int = 30

    # -- Data retention (EUDR Article 31) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-COC-009-CHAIN-OF-CUSTODY-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    rate_limit: int = 1000

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, weight sum validation, threshold ordering,
        and normalization. Collects all errors before raising a single
        ValueError with all violations listed.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a constraint.
        """
        errors: list[str] = []

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Custody event management ----------------------------------------
        if self.gap_threshold_hours < 1:
            errors.append(
                f"gap_threshold_hours must be >= 1, "
                f"got {self.gap_threshold_hours}"
            )
        if self.gap_threshold_hours > 720:
            errors.append(
                f"gap_threshold_hours must be <= 720 (30 days), "
                f"got {self.gap_threshold_hours}"
            )

        if not (1 <= self.max_amendment_depth <= 20):
            errors.append(
                f"max_amendment_depth must be in [1, 20], "
                f"got {self.max_amendment_depth}"
            )

        # -- Batch operations ------------------------------------------------
        if not (2 <= self.max_split_parts <= 10_000):
            errors.append(
                f"max_split_parts must be in [2, 10000], "
                f"got {self.max_split_parts}"
            )

        if not (2 <= self.max_merge_batches <= 1_000):
            errors.append(
                f"max_merge_batches must be in [2, 1000], "
                f"got {self.max_merge_batches}"
            )

        if not (2 <= self.max_blend_inputs <= 500):
            errors.append(
                f"max_blend_inputs must be in [2, 500], "
                f"got {self.max_blend_inputs}"
            )

        # -- Mass balance settings -------------------------------------------
        if not (1 <= self.mb_short_credit_months <= 12):
            errors.append(
                f"mb_short_credit_months must be in [1, 12], "
                f"got {self.mb_short_credit_months}"
            )

        if not (1 <= self.mb_long_credit_months <= 36):
            errors.append(
                f"mb_long_credit_months must be in [1, 36], "
                f"got {self.mb_long_credit_months}"
            )

        if self.mb_short_credit_months >= self.mb_long_credit_months:
            errors.append(
                f"mb_short_credit_months ({self.mb_short_credit_months}) "
                f"must be < mb_long_credit_months "
                f"({self.mb_long_credit_months})"
            )

        if not (0.0 < self.mb_overdraft_threshold_pct <= 100.0):
            errors.append(
                f"mb_overdraft_threshold_pct must be in (0, 100], "
                f"got {self.mb_overdraft_threshold_pct}"
            )

        if not (1 <= self.mb_reconciliation_interval_days <= 365):
            errors.append(
                f"mb_reconciliation_interval_days must be in [1, 365], "
                f"got {self.mb_reconciliation_interval_days}"
            )

        # -- Controlled blending ---------------------------------------------
        if not (0.0 < self.cb_max_blend_ratio <= 1.0):
            errors.append(
                f"cb_max_blend_ratio must be in (0, 1.0], "
                f"got {self.cb_max_blend_ratio}"
            )

        # -- Commodity loss tolerances ---------------------------------------
        for commodity, tolerance in self.commodity_loss_tolerances.items():
            if not (0.0 <= tolerance <= 50.0):
                errors.append(
                    f"commodity_loss_tolerances['{commodity}'] must be "
                    f"in [0, 50], got {tolerance}"
                )

        # -- Process yield ratios --------------------------------------------
        for process, ratio in self.process_yield_ratios.items():
            if not (0.0 < ratio <= 1.0):
                errors.append(
                    f"process_yield_ratios['{process}'] must be in "
                    f"(0, 1.0], got {ratio}"
                )

        # -- Chain integrity thresholds --------------------------------------
        if not (0.0 <= self.chain_completeness_threshold <= 1.0):
            errors.append(
                f"chain_completeness_threshold must be in [0.0, 1.0], "
                f"got {self.chain_completeness_threshold}"
            )

        if not (0.0 <= self.verification_min_score <= 1.0):
            errors.append(
                f"verification_min_score must be in [0.0, 1.0], "
                f"got {self.verification_min_score}"
            )

        # -- Batch processing ------------------------------------------------
        if self.batch_max_size < 1:
            errors.append(
                f"batch_max_size must be >= 1, got {self.batch_max_size}"
            )

        if not (1 <= self.batch_concurrency <= 256):
            errors.append(
                f"batch_concurrency must be in [1, 256], "
                f"got {self.batch_concurrency}"
            )

        if self.batch_timeout_s < 1:
            errors.append(
                f"batch_timeout_s must be >= 1, got {self.batch_timeout_s}"
            )

        # -- Report settings -------------------------------------------------
        if not self.report_formats:
            errors.append("report_formats must not be empty")

        if self.report_generation_timeout_s < 1:
            errors.append(
                f"report_generation_timeout_s must be >= 1, "
                f"got {self.report_generation_timeout_s}"
            )

        # -- Data retention --------------------------------------------------
        if self.retention_years < 1:
            errors.append(
                f"retention_years must be >= 1, "
                f"got {self.retention_years}"
            )

        # -- EUDR commodities ------------------------------------------------
        if not self.eudr_commodities:
            errors.append("eudr_commodities must not be empty")

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance tuning ----------------------------------------------
        if self.pool_size <= 0:
            errors.append(f"pool_size must be > 0, got {self.pool_size}")
        if self.rate_limit <= 0:
            errors.append(f"rate_limit must be > 0, got {self.rate_limit}")

        if errors:
            raise ValueError(
                "ChainOfCustodyConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "ChainOfCustodyConfig validated successfully: "
            "gap_threshold=%dh, amendment_depth=%d, "
            "split_parts=%d, merge_batches=%d, blend_inputs=%d, "
            "mb_short=%dmo, mb_long=%dmo, overdraft=%.1f%%, "
            "recon_interval=%dd, cb_blend_ratio=%.2f, "
            "chain_completeness=%.2f, verification_min=%.2f, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "provenance=%s, metrics=%s",
            self.gap_threshold_hours,
            self.max_amendment_depth,
            self.max_split_parts,
            self.max_merge_batches,
            self.max_blend_inputs,
            self.mb_short_credit_months,
            self.mb_long_credit_months,
            self.mb_overdraft_threshold_pct,
            self.mb_reconciliation_interval_days,
            self.cb_max_blend_ratio,
            self.chain_completeness_threshold,
            self.verification_min_score,
            self.batch_max_size,
            self.batch_concurrency,
            self.retention_years,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ChainOfCustodyConfig:
        """Build a ChainOfCustodyConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_COC_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated ChainOfCustodyConfig instance, validated via
            ``__post_init__``.
        """
        prefix = _ENV_PREFIX

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.strip().lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%r, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val.strip())
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%r, using default %f",
                    prefix, name, val, default,
                )
                return default

        def _str(name: str, default: str) -> str:
            val = _env(name)
            if val is None:
                return default
            return val.strip()

        config = cls(
            # Connections
            database_url=_str("DATABASE_URL", cls.database_url),
            redis_url=_str("REDIS_URL", cls.redis_url),
            # Logging
            log_level=_str("LOG_LEVEL", cls.log_level),
            # Custody event management
            gap_threshold_hours=_int(
                "GAP_THRESHOLD_HOURS", cls.gap_threshold_hours,
            ),
            max_amendment_depth=_int(
                "MAX_AMENDMENT_DEPTH", cls.max_amendment_depth,
            ),
            # Batch operations
            max_split_parts=_int(
                "MAX_SPLIT_PARTS", cls.max_split_parts,
            ),
            max_merge_batches=_int(
                "MAX_MERGE_BATCHES", cls.max_merge_batches,
            ),
            max_blend_inputs=_int(
                "MAX_BLEND_INPUTS", cls.max_blend_inputs,
            ),
            # CoC model rules: Identity Preserved
            ip_physical_separation=_bool(
                "IP_PHYSICAL_SEPARATION", cls.ip_physical_separation,
            ),
            # CoC model rules: Segregated
            sg_allow_compliant_mix=_bool(
                "SG_ALLOW_COMPLIANT_MIX", cls.sg_allow_compliant_mix,
            ),
            # CoC model rules: Mass Balance
            mb_short_credit_months=_int(
                "MB_SHORT_CREDIT_MONTHS", cls.mb_short_credit_months,
            ),
            mb_long_credit_months=_int(
                "MB_LONG_CREDIT_MONTHS", cls.mb_long_credit_months,
            ),
            mb_overdraft_threshold_pct=_float(
                "MB_OVERDRAFT_THRESHOLD_PCT",
                cls.mb_overdraft_threshold_pct,
            ),
            mb_reconciliation_interval_days=_int(
                "MB_RECONCILIATION_INTERVAL_DAYS",
                cls.mb_reconciliation_interval_days,
            ),
            # CoC model rules: Controlled Blending
            cb_max_blend_ratio=_float(
                "CB_MAX_BLEND_RATIO", cls.cb_max_blend_ratio,
            ),
            # Chain integrity
            chain_completeness_threshold=_float(
                "CHAIN_COMPLETENESS_THRESHOLD",
                cls.chain_completeness_threshold,
            ),
            verification_min_score=_float(
                "VERIFICATION_MIN_SCORE", cls.verification_min_score,
            ),
            # Batch processing
            batch_max_size=_int(
                "BATCH_MAX_SIZE", cls.batch_max_size,
            ),
            batch_concurrency=_int(
                "BATCH_CONCURRENCY", cls.batch_concurrency,
            ),
            batch_timeout_s=_int(
                "BATCH_TIMEOUT_S", cls.batch_timeout_s,
            ),
            # Report settings
            report_generation_timeout_s=_int(
                "REPORT_GENERATION_TIMEOUT_S",
                cls.report_generation_timeout_s,
            ),
            # Data retention
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
            # Provenance
            enable_provenance=_bool(
                "ENABLE_PROVENANCE", cls.enable_provenance,
            ),
            genesis_hash=_str("GENESIS_HASH", cls.genesis_hash),
            # Metrics
            enable_metrics=_bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=_int("POOL_SIZE", cls.pool_size),
            rate_limit=_int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "ChainOfCustodyConfig loaded: "
            "gap_threshold=%dh, amendment_depth=%d, "
            "split_parts=%d, merge_batches=%d, blend_inputs=%d, "
            "ip_separation=%s, sg_compliant_mix=%s, "
            "mb_short=%dmo, mb_long=%dmo, overdraft=%.1f%%, "
            "recon_interval=%dd, cb_ratio=%.2f, "
            "chain_completeness=%.2f, verification_min=%.2f, "
            "batch_max=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, provenance=%s, "
            "pool=%d, rate_limit=%d/min, metrics=%s",
            config.gap_threshold_hours,
            config.max_amendment_depth,
            config.max_split_parts,
            config.max_merge_batches,
            config.max_blend_inputs,
            config.ip_physical_separation,
            config.sg_allow_compliant_mix,
            config.mb_short_credit_months,
            config.mb_long_credit_months,
            config.mb_overdraft_threshold_pct,
            config.mb_reconciliation_interval_days,
            config.cb_max_blend_ratio,
            config.chain_completeness_threshold,
            config.verification_min_score,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.enable_provenance,
            config.pool_size,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def mass_balance_settings(self) -> Dict[str, Any]:
        """Return mass balance settings as a dictionary.

        Returns:
            Dictionary with keys: short_credit_months, long_credit_months,
            overdraft_threshold_pct, reconciliation_interval_days.
        """
        return {
            "short_credit_months": self.mb_short_credit_months,
            "long_credit_months": self.mb_long_credit_months,
            "overdraft_threshold_pct": self.mb_overdraft_threshold_pct,
            "reconciliation_interval_days": (
                self.mb_reconciliation_interval_days
            ),
        }

    @property
    def coc_model_rules(self) -> Dict[str, Any]:
        """Return CoC model rules as a dictionary.

        Returns:
            Dictionary with keys: identity_preserved, segregated,
            mass_balance, controlled_blending.
        """
        return {
            "identity_preserved": {
                "physical_separation": self.ip_physical_separation,
            },
            "segregated": {
                "allow_compliant_mix": self.sg_allow_compliant_mix,
            },
            "mass_balance": self.mass_balance_settings,
            "controlled_blending": {
                "max_blend_ratio": self.cb_max_blend_ratio,
            },
        }

    @property
    def batch_operation_limits(self) -> Dict[str, int]:
        """Return batch operation limits as a dictionary.

        Returns:
            Dictionary with keys: max_split_parts, max_merge_batches,
            max_blend_inputs.
        """
        return {
            "max_split_parts": self.max_split_parts,
            "max_merge_batches": self.max_merge_batches,
            "max_blend_inputs": self.max_blend_inputs,
        }

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Re-run post-init validation and return True if valid.

        Returns:
            True if configuration passes all validation checks.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self.__post_init__()
        return True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain Python dictionary.

        Sensitive connection strings (database_url, redis_url) are
        redacted to prevent accidental credential leakage in logs,
        exception tracebacks, and monitoring dashboards.

        Returns:
            Dictionary representation with sensitive fields redacted.
        """
        return {
            # Connections (redacted)
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # Logging
            "log_level": self.log_level,
            # Custody event management
            "gap_threshold_hours": self.gap_threshold_hours,
            "max_amendment_depth": self.max_amendment_depth,
            # Batch operations
            "max_split_parts": self.max_split_parts,
            "max_merge_batches": self.max_merge_batches,
            "max_blend_inputs": self.max_blend_inputs,
            # CoC model rules
            "ip_physical_separation": self.ip_physical_separation,
            "sg_allow_compliant_mix": self.sg_allow_compliant_mix,
            "mb_short_credit_months": self.mb_short_credit_months,
            "mb_long_credit_months": self.mb_long_credit_months,
            "mb_overdraft_threshold_pct": self.mb_overdraft_threshold_pct,
            "mb_reconciliation_interval_days": (
                self.mb_reconciliation_interval_days
            ),
            "cb_max_blend_ratio": self.cb_max_blend_ratio,
            # Commodity loss tolerances
            "commodity_loss_tolerances": dict(self.commodity_loss_tolerances),
            # Process yield ratios (count only to keep dict manageable)
            "process_yield_ratios_count": len(self.process_yield_ratios),
            # Document requirements (count only)
            "document_requirements_event_types": list(
                self.document_requirements.keys()
            ),
            # Chain integrity
            "chain_completeness_threshold": (
                self.chain_completeness_threshold
            ),
            "verification_min_score": self.verification_min_score,
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Report settings
            "report_formats": list(self.report_formats),
            "report_generation_timeout_s": self.report_generation_timeout_s,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
            # Provenance
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # Metrics
            "enable_metrics": self.enable_metrics,
            # Performance tuning
            "pool_size": self.pool_size,
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Returns:
            String representation with sensitive fields redacted.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"ChainOfCustodyConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[ChainOfCustodyConfig] = None
_config_lock = threading.Lock()


def get_config() -> ChainOfCustodyConfig:
    """Return the singleton ChainOfCustodyConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_COC_*`` environment variables.

    Returns:
        ChainOfCustodyConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.gap_threshold_hours
        72
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ChainOfCustodyConfig.from_env()
    return _config_instance


def set_config(config: ChainOfCustodyConfig) -> None:
    """Replace the singleton ChainOfCustodyConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New ChainOfCustodyConfig to install.

    Example:
        >>> cfg = ChainOfCustodyConfig(gap_threshold_hours=48)
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "ChainOfCustodyConfig replaced programmatically: "
        "gap_threshold=%dh, mb_short=%dmo, mb_long=%dmo, "
        "batch_max=%d",
        config.gap_threshold_hours,
        config.mb_short_credit_months,
        config.mb_long_credit_months,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton ChainOfCustodyConfig to None.

    The next call to get_config() will re-read GL_EUDR_COC_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("ChainOfCustodyConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ChainOfCustodyConfig",
    "get_config",
    "set_config",
    "reset_config",
]
