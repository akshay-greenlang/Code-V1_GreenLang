# -*- coding: utf-8 -*-
"""
Mass Balance Calculator Configuration - AGENT-EUDR-011

Centralized configuration for the Mass Balance Calculator Agent covering:
- Double-entry ledger management: balance tracking, entry recording, voiding
- Overdraft detection and enforcement: zero tolerance, percentage, absolute
  modes with configurable thresholds and resolution deadlines
- Credit period lifecycle: RSPO (90d), FSC (365d), ISCC (365d), EUDR default
  (365d) with grace periods and carry-forward rules
- Conversion factor validation: warn and reject deviation thresholds per
  ISO 22095:2020 mass balance requirements
- Loss and waste tracking: processing, transport, storage, quality rejection,
  spillage, contamination losses with tolerance validation
- Reconciliation and sign-off: variance classification (acceptable, warning,
  violation), anomaly detection, trend analysis
- Multi-facility consolidation: regional, country, commodity, custom grouping
- Batch processing: size limits, concurrency, timeout
- Data retention: EUDR Article 14 five-year retention
- Database and cache connection settings
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_EUDR_MBC_`` prefix (e.g. ``GL_EUDR_MBC_DATABASE_URL``,
``GL_EUDR_MBC_OVERDRAFT_MODE``).

Environment Variable Reference (GL_EUDR_MBC_ prefix):
    GL_EUDR_MBC_DATABASE_URL                  - PostgreSQL connection URL
    GL_EUDR_MBC_REDIS_URL                     - Redis connection URL
    GL_EUDR_MBC_LOG_LEVEL                     - Logging level
    GL_EUDR_MBC_OVERDRAFT_MODE                - Overdraft enforcement mode
    GL_EUDR_MBC_OVERDRAFT_TOLERANCE_PERCENT   - Overdraft tolerance as %
    GL_EUDR_MBC_OVERDRAFT_TOLERANCE_KG        - Overdraft tolerance in kg
    GL_EUDR_MBC_DEFAULT_CREDIT_PERIOD_DAYS    - Default credit period in days
    GL_EUDR_MBC_RSPO_CREDIT_PERIOD_DAYS       - RSPO credit period in days
    GL_EUDR_MBC_FSC_CREDIT_PERIOD_DAYS        - FSC credit period in days
    GL_EUDR_MBC_ISCC_CREDIT_PERIOD_DAYS       - ISCC credit period in days
    GL_EUDR_MBC_GRACE_PERIOD_DAYS             - Grace period in days
    GL_EUDR_MBC_MAX_CARRY_FORWARD_PERCENT     - Max carry-forward %
    GL_EUDR_MBC_CONVERSION_FACTOR_WARN_DEVIATION  - Warn deviation threshold
    GL_EUDR_MBC_CONVERSION_FACTOR_REJECT_DEVIATION - Reject deviation threshold
    GL_EUDR_MBC_VARIANCE_ACCEPTABLE_PERCENT   - Acceptable variance %
    GL_EUDR_MBC_VARIANCE_WARNING_PERCENT      - Warning variance %
    GL_EUDR_MBC_LOSS_VALIDATION_ENABLED       - Enable loss validation
    GL_EUDR_MBC_BY_PRODUCT_CREDIT_ENABLED     - Enable by-product credits
    GL_EUDR_MBC_OVERDRAFT_RESOLUTION_HOURS    - Overdraft resolution deadline
    GL_EUDR_MBC_BATCH_MAX_SIZE                - Maximum batch size
    GL_EUDR_MBC_BATCH_CONCURRENCY             - Batch concurrency
    GL_EUDR_MBC_BATCH_TIMEOUT_S               - Batch timeout seconds
    GL_EUDR_MBC_RETENTION_YEARS               - Data retention years
    GL_EUDR_MBC_REPORT_DEFAULT_FORMAT         - Default report format
    GL_EUDR_MBC_REPORT_RETENTION_DAYS         - Report retention in days
    GL_EUDR_MBC_ENABLE_PROVENANCE             - Enable provenance tracking
    GL_EUDR_MBC_GENESIS_HASH                  - Genesis hash anchor
    GL_EUDR_MBC_ENABLE_METRICS                - Enable Prometheus metrics
    GL_EUDR_MBC_POOL_SIZE                     - Database pool size
    GL_EUDR_MBC_RATE_LIMIT                    - Max requests per minute
    GL_EUDR_MBC_RECONCILIATION_AUTO_ROLLOVER  - Auto-rollover on reconciliation
    GL_EUDR_MBC_ANOMALY_DETECTION_ENABLED     - Enable anomaly detection
    GL_EUDR_MBC_TREND_WINDOW_PERIODS          - Trend analysis window
    GL_EUDR_MBC_MIN_ENTRIES_FOR_TREND         - Min entries for trend analysis

Example:
    >>> from greenlang.agents.eudr.mass_balance_calculator.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.overdraft_mode, cfg.default_credit_period_days)
    zero_tolerance 365

    >>> # Override for testing
    >>> from greenlang.agents.eudr.mass_balance_calculator.config import (
    ...     set_config, reset_config, MassBalanceCalculatorConfig,
    ... )
    >>> set_config(MassBalanceCalculatorConfig(overdraft_mode="percentage"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator (GL-EUDR-MBC-011)
Status: Production Ready
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_EUDR_MBC_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid overdraft enforcement modes
# ---------------------------------------------------------------------------

_VALID_OVERDRAFT_MODES = frozenset(
    {"zero_tolerance", "percentage", "absolute"}
)

# ---------------------------------------------------------------------------
# Valid report formats
# ---------------------------------------------------------------------------

_VALID_REPORT_FORMATS = frozenset(
    {"json", "csv", "pdf", "eudr_xml"}
)

# ---------------------------------------------------------------------------
# Default credit period days per certification standard
# ---------------------------------------------------------------------------

_DEFAULT_CREDIT_PERIODS: Dict[str, int] = {
    "eudr_default": 365,
    "rspo": 90,
    "fsc": 365,
    "iscc": 365,
    "utz_ra": 365,
    "fairtrade": 365,
}

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
# Default maximum loss tolerances by loss type (%)
# Used for loss validation when LOSS_VALIDATION_ENABLED is True
# ---------------------------------------------------------------------------

_DEFAULT_LOSS_TYPE_TOLERANCES: Dict[str, float] = {
    "processing_loss": 15.0,
    "transport_loss": 3.0,
    "storage_loss": 5.0,
    "quality_rejection": 10.0,
    "spillage": 2.0,
    "contamination_loss": 5.0,
}

# ---------------------------------------------------------------------------
# Default reference conversion factors (yield ratios)
# commodity -> process -> yield_ratio (output_mass / input_mass)
# ---------------------------------------------------------------------------

_DEFAULT_REFERENCE_CONVERSION_FACTORS: Dict[str, Dict[str, float]] = {
    "cocoa": {
        "fermentation": 0.92,
        "drying": 0.88,
        "roasting": 0.85,
        "winnowing": 0.80,
        "grinding": 0.98,
        "pressing": 0.45,
        "conching": 0.97,
        "tempering": 0.99,
    },
    "coffee": {
        "wet_processing": 0.60,
        "dry_processing": 0.50,
        "hulling": 0.80,
        "polishing": 0.98,
        "roasting": 0.82,
    },
    "oil_palm": {
        "sterilization": 0.95,
        "threshing": 0.65,
        "digestion": 0.90,
        "extraction": 0.22,
        "clarification": 0.95,
        "refining": 0.92,
        "fractionation": 0.90,
    },
    "wood": {
        "debarking": 0.90,
        "sawing": 0.55,
        "planing": 0.90,
        "kiln_drying": 0.92,
        "milling": 0.85,
    },
    "rubber": {
        "coagulation": 0.60,
        "sheeting": 0.95,
        "smoking": 0.88,
        "crumbling": 0.92,
    },
    "soya": {
        "cleaning": 0.98,
        "dehulling": 0.92,
        "flaking": 0.97,
        "solvent_extraction": 0.82,
        "refining": 0.92,
    },
    "cattle": {
        "slaughtering": 0.55,
        "deboning": 0.70,
        "tanning": 0.30,
    },
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

_DEFAULT_REPORT_FORMATS: List[str] = ["json", "csv", "pdf", "eudr_xml"]


# ---------------------------------------------------------------------------
# MassBalanceCalculatorConfig
# ---------------------------------------------------------------------------


@dataclass
class MassBalanceCalculatorConfig:
    """Complete configuration for the EUDR Mass Balance Calculator Agent.

    Attributes are grouped by concern: connections, logging, overdraft
    enforcement, credit period lifecycle, conversion factor validation,
    variance thresholds, loss/waste tracking, batch processing, reporting,
    data retention, reconciliation, anomaly detection, provenance tracking,
    metrics, and performance tuning.

    All attributes can be overridden via environment variables using
    the ``GL_EUDR_MBC_`` prefix.

    Attributes:
        database_url: PostgreSQL connection URL for persistent storage of
            ledgers, entries, credit periods, and reconciliation records.
        redis_url: Redis connection URL for balance caching, distributed
            locks, and overdraft alert queues.
        log_level: Logging verbosity level. Accepts DEBUG, INFO,
            WARNING, ERROR, or CRITICAL.
        overdraft_mode: Overdraft enforcement mode. ``zero_tolerance``
            rejects any negative balance. ``percentage`` allows overdraft
            up to overdraft_tolerance_percent of total inputs.
            ``absolute`` allows overdraft up to overdraft_tolerance_kg.
        overdraft_tolerance_percent: Maximum overdraft as a percentage
            of total period inputs (used when overdraft_mode=percentage).
        overdraft_tolerance_kg: Maximum overdraft in kilograms (used when
            overdraft_mode=absolute).
        default_credit_period_days: Default credit period length in days
            for standards without a specific period configured.
        rspo_credit_period_days: RSPO mass balance credit period in days
            (typically 90 days per RSPO SCC 2020).
        fsc_credit_period_days: FSC mass balance credit period in days
            (typically 365 days per FSC-STD-40-004).
        iscc_credit_period_days: ISCC mass balance credit period in days
            (typically 365 days per ISCC 203).
        grace_period_days: Grace period after credit period end before
            carry-forward balance expires.
        max_carry_forward_percent: Maximum percentage of period-end balance
            that can be carried forward to the next period (0-100).
        conversion_factor_warn_deviation: Deviation threshold (0.0-1.0)
            from reference conversion factor that triggers a warning.
        conversion_factor_reject_deviation: Deviation threshold (0.0-1.0)
            from reference conversion factor that triggers rejection.
        variance_acceptable_percent: Maximum variance percentage considered
            acceptable during reconciliation.
        variance_warning_percent: Variance percentage threshold above which
            a warning is issued during reconciliation.
        loss_validation_enabled: Whether to validate reported losses against
            expected loss tolerances per commodity and loss type.
        by_product_credit_enabled: Whether by-products from processing
            generate credits in the mass balance ledger.
        overdraft_resolution_hours: Number of hours within which an
            overdraft event must be resolved before escalation.
        batch_max_size: Maximum number of records in a single batch
            processing job.
        batch_concurrency: Maximum concurrent batch processing workers.
        batch_timeout_s: Timeout in seconds for a single batch job.
        retention_years: Data retention in years per EUDR Article 14.
        report_default_format: Default output format for generated reports.
        report_retention_days: Number of days to retain generated reports.
        eudr_commodities: List of EUDR-regulated commodity types.
        commodity_loss_tolerances: Per-commodity acceptable processing
            loss percentages (0-100).
        loss_type_tolerances: Per-loss-type maximum acceptable loss
            percentages (0-100).
        reference_conversion_factors: Reference yield ratios organized
            by commodity and process type.
        credit_period_days_by_standard: Credit period days by standard.
        enable_provenance: Enable SHA-256 provenance chain tracking
            for all mass balance operations.
        genesis_hash: Genesis anchor string for the provenance chain,
            unique to the Mass Balance Calculator agent.
        enable_metrics: Enable Prometheus metrics export under the
            ``gl_eudr_mbc_`` prefix.
        pool_size: PostgreSQL connection pool size.
        rate_limit: Maximum inbound API requests per minute.
        reconciliation_auto_rollover: Whether to automatically create a
            new credit period when reconciliation closes the current one.
        anomaly_detection_enabled: Whether to enable statistical anomaly
            detection during reconciliation (Z-score, trend deviation).
        trend_window_periods: Number of historical periods to include
            in trend analysis calculations.
        min_entries_for_trend: Minimum number of ledger entries required
            before trend analysis is performed.
    """

    # -- Connections ---------------------------------------------------------
    database_url: str = "postgresql://localhost:5432/greenlang"
    redis_url: str = "redis://localhost:6379/0"

    # -- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # -- Overdraft enforcement -----------------------------------------------
    overdraft_mode: str = "zero_tolerance"
    overdraft_tolerance_percent: float = 0.0
    overdraft_tolerance_kg: float = 0.0
    overdraft_resolution_hours: int = 48

    # -- Credit period lifecycle ---------------------------------------------
    default_credit_period_days: int = 365
    rspo_credit_period_days: int = 90
    fsc_credit_period_days: int = 365
    iscc_credit_period_days: int = 365
    grace_period_days: int = 5
    max_carry_forward_percent: float = 100.0

    # -- Conversion factor validation ----------------------------------------
    conversion_factor_warn_deviation: float = 0.05
    conversion_factor_reject_deviation: float = 0.15

    # -- Variance thresholds -------------------------------------------------
    variance_acceptable_percent: float = 1.0
    variance_warning_percent: float = 3.0

    # -- Loss/waste tracking -------------------------------------------------
    loss_validation_enabled: bool = True
    by_product_credit_enabled: bool = True

    # -- Batch processing ----------------------------------------------------
    batch_max_size: int = 500
    batch_concurrency: int = 4
    batch_timeout_s: int = 120

    # -- Reporting -----------------------------------------------------------
    report_default_format: str = "json"
    report_retention_days: int = 1825

    # -- Data retention (EUDR Article 14) ------------------------------------
    retention_years: int = 5

    # -- EUDR commodities ----------------------------------------------------
    eudr_commodities: List[str] = field(
        default_factory=lambda: list(_DEFAULT_EUDR_COMMODITIES)
    )

    # -- Commodity loss tolerances -------------------------------------------
    commodity_loss_tolerances: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_COMMODITY_LOSS_TOLERANCES)
    )

    # -- Loss type tolerances ------------------------------------------------
    loss_type_tolerances: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_LOSS_TYPE_TOLERANCES)
    )

    # -- Reference conversion factors ----------------------------------------
    reference_conversion_factors: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            k: dict(v)
            for k, v in _DEFAULT_REFERENCE_CONVERSION_FACTORS.items()
        }
    )

    # -- Credit period days by standard --------------------------------------
    credit_period_days_by_standard: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_CREDIT_PERIODS)
    )

    # -- Reconciliation ------------------------------------------------------
    reconciliation_auto_rollover: bool = True

    # -- Anomaly detection ---------------------------------------------------
    anomaly_detection_enabled: bool = True
    trend_window_periods: int = 6
    min_entries_for_trend: int = 10

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "GL-EUDR-MBC-011-MASS-BALANCE-CALCULATOR-GENESIS"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 10
    rate_limit: int = 300

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialization.

        Performs range checks on all numeric fields, enumeration checks
        on string fields, threshold ordering validation, and
        normalization. Collects all errors before raising a single
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

        # -- Overdraft enforcement -------------------------------------------
        normalised_mode = self.overdraft_mode.lower().strip()
        if normalised_mode not in _VALID_OVERDRAFT_MODES:
            errors.append(
                f"overdraft_mode must be one of "
                f"{sorted(_VALID_OVERDRAFT_MODES)}, "
                f"got '{self.overdraft_mode}'"
            )
        else:
            self.overdraft_mode = normalised_mode

        if self.overdraft_tolerance_percent < 0.0:
            errors.append(
                f"overdraft_tolerance_percent must be >= 0.0, "
                f"got {self.overdraft_tolerance_percent}"
            )
        if self.overdraft_tolerance_percent > 100.0:
            errors.append(
                f"overdraft_tolerance_percent must be <= 100.0, "
                f"got {self.overdraft_tolerance_percent}"
            )

        if self.overdraft_tolerance_kg < 0.0:
            errors.append(
                f"overdraft_tolerance_kg must be >= 0.0, "
                f"got {self.overdraft_tolerance_kg}"
            )

        if self.overdraft_resolution_hours < 1:
            errors.append(
                f"overdraft_resolution_hours must be >= 1, "
                f"got {self.overdraft_resolution_hours}"
            )
        if self.overdraft_resolution_hours > 720:
            errors.append(
                f"overdraft_resolution_hours must be <= 720 (30 days), "
                f"got {self.overdraft_resolution_hours}"
            )

        # -- Credit period lifecycle -----------------------------------------
        if self.default_credit_period_days < 1:
            errors.append(
                f"default_credit_period_days must be >= 1, "
                f"got {self.default_credit_period_days}"
            )
        if self.default_credit_period_days > 730:
            errors.append(
                f"default_credit_period_days must be <= 730 (2 years), "
                f"got {self.default_credit_period_days}"
            )

        if not (1 <= self.rspo_credit_period_days <= 365):
            errors.append(
                f"rspo_credit_period_days must be in [1, 365], "
                f"got {self.rspo_credit_period_days}"
            )

        if not (1 <= self.fsc_credit_period_days <= 730):
            errors.append(
                f"fsc_credit_period_days must be in [1, 730], "
                f"got {self.fsc_credit_period_days}"
            )

        if not (1 <= self.iscc_credit_period_days <= 730):
            errors.append(
                f"iscc_credit_period_days must be in [1, 730], "
                f"got {self.iscc_credit_period_days}"
            )

        if self.grace_period_days < 0:
            errors.append(
                f"grace_period_days must be >= 0, "
                f"got {self.grace_period_days}"
            )
        if self.grace_period_days > 90:
            errors.append(
                f"grace_period_days must be <= 90, "
                f"got {self.grace_period_days}"
            )

        if not (0.0 <= self.max_carry_forward_percent <= 100.0):
            errors.append(
                f"max_carry_forward_percent must be in [0.0, 100.0], "
                f"got {self.max_carry_forward_percent}"
            )

        # -- Conversion factor validation ------------------------------------
        if not (0.0 < self.conversion_factor_warn_deviation <= 1.0):
            errors.append(
                f"conversion_factor_warn_deviation must be in (0.0, 1.0], "
                f"got {self.conversion_factor_warn_deviation}"
            )

        if not (0.0 < self.conversion_factor_reject_deviation <= 1.0):
            errors.append(
                f"conversion_factor_reject_deviation must be in (0.0, 1.0], "
                f"got {self.conversion_factor_reject_deviation}"
            )

        if (
            self.conversion_factor_warn_deviation
            >= self.conversion_factor_reject_deviation
        ):
            errors.append(
                f"conversion_factor_warn_deviation "
                f"({self.conversion_factor_warn_deviation}) must be < "
                f"conversion_factor_reject_deviation "
                f"({self.conversion_factor_reject_deviation})"
            )

        # -- Variance thresholds ---------------------------------------------
        if self.variance_acceptable_percent < 0.0:
            errors.append(
                f"variance_acceptable_percent must be >= 0.0, "
                f"got {self.variance_acceptable_percent}"
            )

        if self.variance_warning_percent < 0.0:
            errors.append(
                f"variance_warning_percent must be >= 0.0, "
                f"got {self.variance_warning_percent}"
            )

        if (
            self.variance_acceptable_percent
            >= self.variance_warning_percent
        ):
            errors.append(
                f"variance_acceptable_percent "
                f"({self.variance_acceptable_percent}) must be < "
                f"variance_warning_percent "
                f"({self.variance_warning_percent})"
            )

        # -- Commodity loss tolerances ---------------------------------------
        for commodity, tolerance in self.commodity_loss_tolerances.items():
            if not (0.0 <= tolerance <= 50.0):
                errors.append(
                    f"commodity_loss_tolerances['{commodity}'] must be "
                    f"in [0, 50], got {tolerance}"
                )

        # -- Loss type tolerances --------------------------------------------
        for loss_type, tolerance in self.loss_type_tolerances.items():
            if not (0.0 <= tolerance <= 100.0):
                errors.append(
                    f"loss_type_tolerances['{loss_type}'] must be "
                    f"in [0, 100], got {tolerance}"
                )

        # -- Reference conversion factors ------------------------------------
        for commodity, factors in self.reference_conversion_factors.items():
            for process, ratio in factors.items():
                if not (0.0 < ratio <= 1.0):
                    errors.append(
                        f"reference_conversion_factors['{commodity}']"
                        f"['{process}'] must be in (0, 1.0], got {ratio}"
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
        normalised_format = self.report_default_format.lower().strip()
        if normalised_format not in _VALID_REPORT_FORMATS:
            errors.append(
                f"report_default_format must be one of "
                f"{sorted(_VALID_REPORT_FORMATS)}, "
                f"got '{self.report_default_format}'"
            )
        else:
            self.report_default_format = normalised_format

        if self.report_retention_days < 1:
            errors.append(
                f"report_retention_days must be >= 1, "
                f"got {self.report_retention_days}"
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

        # -- Reconciliation / anomaly detection ------------------------------
        if self.trend_window_periods < 1:
            errors.append(
                f"trend_window_periods must be >= 1, "
                f"got {self.trend_window_periods}"
            )

        if self.min_entries_for_trend < 1:
            errors.append(
                f"min_entries_for_trend must be >= 1, "
                f"got {self.min_entries_for_trend}"
            )

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
                "MassBalanceCalculatorConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "MassBalanceCalculatorConfig validated successfully: "
            "overdraft_mode=%s, tolerance_pct=%.1f%%, tolerance_kg=%.1f, "
            "default_period=%dd, rspo=%dd, fsc=%dd, iscc=%dd, "
            "grace=%dd, carry_fwd=%.0f%%, "
            "cf_warn=%.2f, cf_reject=%.2f, "
            "var_accept=%.1f%%, var_warn=%.1f%%, "
            "loss_validation=%s, by_product_credit=%s, "
            "batch_max=%d, concurrency=%d, retention=%dy, "
            "auto_rollover=%s, anomaly=%s, "
            "provenance=%s, metrics=%s",
            self.overdraft_mode,
            self.overdraft_tolerance_percent,
            self.overdraft_tolerance_kg,
            self.default_credit_period_days,
            self.rspo_credit_period_days,
            self.fsc_credit_period_days,
            self.iscc_credit_period_days,
            self.grace_period_days,
            self.max_carry_forward_percent,
            self.conversion_factor_warn_deviation,
            self.conversion_factor_reject_deviation,
            self.variance_acceptable_percent,
            self.variance_warning_percent,
            self.loss_validation_enabled,
            self.by_product_credit_enabled,
            self.batch_max_size,
            self.batch_concurrency,
            self.retention_years,
            self.reconciliation_auto_rollover,
            self.anomaly_detection_enabled,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> MassBalanceCalculatorConfig:
        """Build a MassBalanceCalculatorConfig from environment variables.

        Every field can be overridden via ``GL_EUDR_MBC_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.
        Unknown or malformed values fall back to the class-level default
        and emit a WARNING log so the issue is visible in deployment logs.

        Returns:
            Populated MassBalanceCalculatorConfig instance, validated via
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
            # Overdraft enforcement
            overdraft_mode=_str("OVERDRAFT_MODE", cls.overdraft_mode),
            overdraft_tolerance_percent=_float(
                "OVERDRAFT_TOLERANCE_PERCENT",
                cls.overdraft_tolerance_percent,
            ),
            overdraft_tolerance_kg=_float(
                "OVERDRAFT_TOLERANCE_KG",
                cls.overdraft_tolerance_kg,
            ),
            overdraft_resolution_hours=_int(
                "OVERDRAFT_RESOLUTION_HOURS",
                cls.overdraft_resolution_hours,
            ),
            # Credit period lifecycle
            default_credit_period_days=_int(
                "DEFAULT_CREDIT_PERIOD_DAYS",
                cls.default_credit_period_days,
            ),
            rspo_credit_period_days=_int(
                "RSPO_CREDIT_PERIOD_DAYS",
                cls.rspo_credit_period_days,
            ),
            fsc_credit_period_days=_int(
                "FSC_CREDIT_PERIOD_DAYS",
                cls.fsc_credit_period_days,
            ),
            iscc_credit_period_days=_int(
                "ISCC_CREDIT_PERIOD_DAYS",
                cls.iscc_credit_period_days,
            ),
            grace_period_days=_int(
                "GRACE_PERIOD_DAYS",
                cls.grace_period_days,
            ),
            max_carry_forward_percent=_float(
                "MAX_CARRY_FORWARD_PERCENT",
                cls.max_carry_forward_percent,
            ),
            # Conversion factor validation
            conversion_factor_warn_deviation=_float(
                "CONVERSION_FACTOR_WARN_DEVIATION",
                cls.conversion_factor_warn_deviation,
            ),
            conversion_factor_reject_deviation=_float(
                "CONVERSION_FACTOR_REJECT_DEVIATION",
                cls.conversion_factor_reject_deviation,
            ),
            # Variance thresholds
            variance_acceptable_percent=_float(
                "VARIANCE_ACCEPTABLE_PERCENT",
                cls.variance_acceptable_percent,
            ),
            variance_warning_percent=_float(
                "VARIANCE_WARNING_PERCENT",
                cls.variance_warning_percent,
            ),
            # Loss/waste tracking
            loss_validation_enabled=_bool(
                "LOSS_VALIDATION_ENABLED",
                cls.loss_validation_enabled,
            ),
            by_product_credit_enabled=_bool(
                "BY_PRODUCT_CREDIT_ENABLED",
                cls.by_product_credit_enabled,
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
            # Reporting
            report_default_format=_str(
                "REPORT_DEFAULT_FORMAT",
                cls.report_default_format,
            ),
            report_retention_days=_int(
                "REPORT_RETENTION_DAYS",
                cls.report_retention_days,
            ),
            # Data retention
            retention_years=_int(
                "RETENTION_YEARS", cls.retention_years,
            ),
            # Reconciliation
            reconciliation_auto_rollover=_bool(
                "RECONCILIATION_AUTO_ROLLOVER",
                cls.reconciliation_auto_rollover,
            ),
            # Anomaly detection
            anomaly_detection_enabled=_bool(
                "ANOMALY_DETECTION_ENABLED",
                cls.anomaly_detection_enabled,
            ),
            trend_window_periods=_int(
                "TREND_WINDOW_PERIODS",
                cls.trend_window_periods,
            ),
            min_entries_for_trend=_int(
                "MIN_ENTRIES_FOR_TREND",
                cls.min_entries_for_trend,
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
            "MassBalanceCalculatorConfig loaded: "
            "overdraft_mode=%s, tolerance_pct=%.1f%%, "
            "tolerance_kg=%.1f, resolution=%dh, "
            "default_period=%dd, rspo=%dd, fsc=%dd, iscc=%dd, "
            "grace=%dd, carry_fwd=%.0f%%, "
            "cf_warn=%.2f, cf_reject=%.2f, "
            "var_accept=%.1f%%, var_warn=%.1f%%, "
            "loss_validation=%s, by_product=%s, "
            "batch_max=%d, concurrency=%d, timeout=%ds, "
            "retention=%dy, report_format=%s, "
            "auto_rollover=%s, anomaly=%s, "
            "trend_window=%d, min_trend=%d, "
            "provenance=%s, pool=%d, rate_limit=%d/min, "
            "metrics=%s",
            config.overdraft_mode,
            config.overdraft_tolerance_percent,
            config.overdraft_tolerance_kg,
            config.overdraft_resolution_hours,
            config.default_credit_period_days,
            config.rspo_credit_period_days,
            config.fsc_credit_period_days,
            config.iscc_credit_period_days,
            config.grace_period_days,
            config.max_carry_forward_percent,
            config.conversion_factor_warn_deviation,
            config.conversion_factor_reject_deviation,
            config.variance_acceptable_percent,
            config.variance_warning_percent,
            config.loss_validation_enabled,
            config.by_product_credit_enabled,
            config.batch_max_size,
            config.batch_concurrency,
            config.batch_timeout_s,
            config.retention_years,
            config.report_default_format,
            config.reconciliation_auto_rollover,
            config.anomaly_detection_enabled,
            config.trend_window_periods,
            config.min_entries_for_trend,
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
    def overdraft_settings(self) -> Dict[str, Any]:
        """Return overdraft enforcement settings as a dictionary.

        Returns:
            Dictionary with keys: mode, tolerance_percent,
            tolerance_kg, resolution_hours.
        """
        return {
            "mode": self.overdraft_mode,
            "tolerance_percent": self.overdraft_tolerance_percent,
            "tolerance_kg": self.overdraft_tolerance_kg,
            "resolution_hours": self.overdraft_resolution_hours,
        }

    @property
    def credit_period_settings(self) -> Dict[str, Any]:
        """Return credit period settings as a dictionary.

        Returns:
            Dictionary with keys: default_days, rspo_days, fsc_days,
            iscc_days, grace_period_days, max_carry_forward_percent.
        """
        return {
            "default_days": self.default_credit_period_days,
            "rspo_days": self.rspo_credit_period_days,
            "fsc_days": self.fsc_credit_period_days,
            "iscc_days": self.iscc_credit_period_days,
            "grace_period_days": self.grace_period_days,
            "max_carry_forward_percent": self.max_carry_forward_percent,
        }

    @property
    def conversion_factor_settings(self) -> Dict[str, float]:
        """Return conversion factor validation thresholds.

        Returns:
            Dictionary with keys: warn_deviation, reject_deviation.
        """
        return {
            "warn_deviation": self.conversion_factor_warn_deviation,
            "reject_deviation": self.conversion_factor_reject_deviation,
        }

    @property
    def variance_settings(self) -> Dict[str, float]:
        """Return variance classification thresholds.

        Returns:
            Dictionary with keys: acceptable_percent, warning_percent.
        """
        return {
            "acceptable_percent": self.variance_acceptable_percent,
            "warning_percent": self.variance_warning_percent,
        }

    @property
    def reconciliation_settings(self) -> Dict[str, Any]:
        """Return reconciliation and anomaly detection settings.

        Returns:
            Dictionary with keys: auto_rollover, anomaly_detection,
            trend_window_periods, min_entries_for_trend.
        """
        return {
            "auto_rollover": self.reconciliation_auto_rollover,
            "anomaly_detection": self.anomaly_detection_enabled,
            "trend_window_periods": self.trend_window_periods,
            "min_entries_for_trend": self.min_entries_for_trend,
        }

    def get_credit_period_days(self, standard: str) -> int:
        """Return the credit period days for a given certification standard.

        Args:
            standard: Certification standard identifier (rspo, fsc,
                iscc, utz_ra, fairtrade, eudr_default).

        Returns:
            Number of days for the credit period.
        """
        standard_lower = standard.lower().strip()
        mapping = {
            "rspo": self.rspo_credit_period_days,
            "fsc": self.fsc_credit_period_days,
            "iscc": self.iscc_credit_period_days,
        }
        return mapping.get(standard_lower, self.default_credit_period_days)

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
            # Overdraft enforcement
            "overdraft_mode": self.overdraft_mode,
            "overdraft_tolerance_percent": self.overdraft_tolerance_percent,
            "overdraft_tolerance_kg": self.overdraft_tolerance_kg,
            "overdraft_resolution_hours": self.overdraft_resolution_hours,
            # Credit period lifecycle
            "default_credit_period_days": self.default_credit_period_days,
            "rspo_credit_period_days": self.rspo_credit_period_days,
            "fsc_credit_period_days": self.fsc_credit_period_days,
            "iscc_credit_period_days": self.iscc_credit_period_days,
            "grace_period_days": self.grace_period_days,
            "max_carry_forward_percent": self.max_carry_forward_percent,
            # Conversion factor validation
            "conversion_factor_warn_deviation": (
                self.conversion_factor_warn_deviation
            ),
            "conversion_factor_reject_deviation": (
                self.conversion_factor_reject_deviation
            ),
            # Variance thresholds
            "variance_acceptable_percent": self.variance_acceptable_percent,
            "variance_warning_percent": self.variance_warning_percent,
            # Loss/waste tracking
            "loss_validation_enabled": self.loss_validation_enabled,
            "by_product_credit_enabled": self.by_product_credit_enabled,
            # Commodity loss tolerances
            "commodity_loss_tolerances": dict(self.commodity_loss_tolerances),
            # Loss type tolerances
            "loss_type_tolerances": dict(self.loss_type_tolerances),
            # Reference conversion factors (count only)
            "reference_conversion_factors_commodities": list(
                self.reference_conversion_factors.keys()
            ),
            # Batch processing
            "batch_max_size": self.batch_max_size,
            "batch_concurrency": self.batch_concurrency,
            "batch_timeout_s": self.batch_timeout_s,
            # Reporting
            "report_default_format": self.report_default_format,
            "report_retention_days": self.report_retention_days,
            # Data retention
            "retention_years": self.retention_years,
            # EUDR commodities
            "eudr_commodities": list(self.eudr_commodities),
            # Reconciliation
            "reconciliation_auto_rollover": (
                self.reconciliation_auto_rollover
            ),
            # Anomaly detection
            "anomaly_detection_enabled": self.anomaly_detection_enabled,
            "trend_window_periods": self.trend_window_periods,
            "min_entries_for_trend": self.min_entries_for_trend,
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
        return f"MassBalanceCalculatorConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[MassBalanceCalculatorConfig] = None
_config_lock = threading.Lock()


def get_config() -> MassBalanceCalculatorConfig:
    """Return the singleton MassBalanceCalculatorConfig, creating from env if needed.

    Uses double-checked locking for thread safety with minimal
    contention on the hot path. The instance is created on first call
    by reading all ``GL_EUDR_MBC_*`` environment variables.

    Returns:
        MassBalanceCalculatorConfig singleton instance.

    Example:
        >>> cfg = get_config()
        >>> cfg.overdraft_mode
        'zero_tolerance'
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = MassBalanceCalculatorConfig.from_env()
    return _config_instance


def set_config(config: MassBalanceCalculatorConfig) -> None:
    """Replace the singleton MassBalanceCalculatorConfig.

    Primarily intended for testing and dependency injection.

    Args:
        config: New MassBalanceCalculatorConfig to install.

    Example:
        >>> cfg = MassBalanceCalculatorConfig(overdraft_mode="percentage")
        >>> set_config(cfg)
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info(
        "MassBalanceCalculatorConfig replaced programmatically: "
        "overdraft_mode=%s, default_period=%dd, "
        "batch_max=%d",
        config.overdraft_mode,
        config.default_credit_period_days,
        config.batch_max_size,
    )


def reset_config() -> None:
    """Reset the singleton MassBalanceCalculatorConfig to None.

    The next call to get_config() will re-read GL_EUDR_MBC_* env vars
    and construct a fresh instance. Intended for test teardown.

    Example:
        >>> reset_config()
        >>> cfg = get_config()  # re-reads env vars
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
    logger.debug("MassBalanceCalculatorConfig singleton reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "MassBalanceCalculatorConfig",
    "get_config",
    "set_config",
    "reset_config",
]
