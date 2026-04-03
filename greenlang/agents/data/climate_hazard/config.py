# -*- coding: utf-8 -*-
"""
Climate Hazard Connector Service Configuration - AGENT-DATA-020

Centralized configuration for the Climate Hazard Connector Agent SDK covering:
- Database, cache, and connection defaults
- Climate scenario and time horizon defaults (SSP scenarios, RCP pathways)
- Hazard source capacity limits (max sources, assets, risk indices)
- Risk index weight distribution (probability, intensity, frequency, duration)
- Vulnerability weight distribution (exposure, sensitivity, adaptive capacity)
- Risk classification thresholds (extreme, high, medium, low)
- Default report format (json, csv, pdf)
- Pipeline processing limits (max pipeline runs, reports)
- Provenance tracking (genesis hash, SHA-256 chain anchoring)
- Connection pool sizing, rate limiting, and cache TTL
- Prometheus metrics export toggle

All settings can be overridden via environment variables with the
``GL_CLIMATE_HAZARD_`` prefix (e.g. ``GL_CLIMATE_HAZARD_DEFAULT_SCENARIO``,
``GL_CLIMATE_HAZARD_MAX_ASSETS``).

Environment Variable Reference (GL_CLIMATE_HAZARD_ prefix):
    GL_CLIMATE_HAZARD_DATABASE_URL              - PostgreSQL connection URL
    GL_CLIMATE_HAZARD_REDIS_URL                 - Redis connection URL
    GL_CLIMATE_HAZARD_LOG_LEVEL                 - Logging level (DEBUG/INFO/WARNING/ERROR)
    GL_CLIMATE_HAZARD_DEFAULT_SCENARIO          - Default climate scenario (SSP1-2.6,
                                                  SSP2-4.5, SSP3-7.0, SSP5-8.5,
                                                  RCP2.6, RCP4.5, RCP6.0, RCP8.5)
    GL_CLIMATE_HAZARD_DEFAULT_TIME_HORIZON      - Default time horizon (SHORT_TERM,
                                                  MID_TERM, LONG_TERM)
    GL_CLIMATE_HAZARD_DEFAULT_REPORT_FORMAT     - Default report format (json, csv, pdf)
    GL_CLIMATE_HAZARD_MAX_HAZARD_SOURCES        - Maximum hazard data sources
    GL_CLIMATE_HAZARD_MAX_ASSETS                - Maximum assets per analysis run
    GL_CLIMATE_HAZARD_MAX_RISK_INDICES          - Maximum risk index records in storage
    GL_CLIMATE_HAZARD_RISK_WEIGHT_PROBABILITY   - Risk index weight for probability (0-1)
    GL_CLIMATE_HAZARD_RISK_WEIGHT_INTENSITY     - Risk index weight for intensity (0-1)
    GL_CLIMATE_HAZARD_RISK_WEIGHT_FREQUENCY     - Risk index weight for frequency (0-1)
    GL_CLIMATE_HAZARD_RISK_WEIGHT_DURATION      - Risk index weight for duration (0-1)
    GL_CLIMATE_HAZARD_VULN_WEIGHT_EXPOSURE      - Vulnerability weight for exposure (0-1)
    GL_CLIMATE_HAZARD_VULN_WEIGHT_SENSITIVITY   - Vulnerability weight for sensitivity (0-1)
    GL_CLIMATE_HAZARD_VULN_WEIGHT_ADAPTIVE      - Vulnerability weight for adaptive
                                                  capacity (0-1)
    GL_CLIMATE_HAZARD_THRESHOLD_EXTREME         - Score threshold for extreme risk (0-100)
    GL_CLIMATE_HAZARD_THRESHOLD_HIGH            - Score threshold for high risk (0-100)
    GL_CLIMATE_HAZARD_THRESHOLD_MEDIUM          - Score threshold for medium risk (0-100)
    GL_CLIMATE_HAZARD_THRESHOLD_LOW             - Score threshold for low risk (0-100)
    GL_CLIMATE_HAZARD_MAX_PIPELINE_RUNS         - Maximum concurrent pipeline runs
    GL_CLIMATE_HAZARD_MAX_REPORTS               - Maximum stored reports
    GL_CLIMATE_HAZARD_ENABLE_PROVENANCE         - Enable SHA-256 provenance chain tracking
    GL_CLIMATE_HAZARD_GENESIS_HASH              - Genesis anchor string for provenance chain
    GL_CLIMATE_HAZARD_ENABLE_METRICS            - Enable Prometheus metrics export
    GL_CLIMATE_HAZARD_POOL_SIZE                 - Database connection pool size
    GL_CLIMATE_HAZARD_CACHE_TTL                 - Cache time-to-live in seconds
    GL_CLIMATE_HAZARD_RATE_LIMIT                - Max API requests per minute

Example:
    >>> from greenlang.agents.data.climate_hazard.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.default_scenario, cfg.max_assets)
    SSP2-4.5 10000

    >>> # Override for testing
    >>> from greenlang.agents.data.climate_hazard.config import set_config, reset_config
    >>> from greenlang.agents.data.climate_hazard.config import ClimateHazardConfig
    >>> set_config(ClimateHazardConfig(max_assets=500, default_scenario="SSP5-8.5"))
    >>> reset_config()  # teardown

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from greenlang.data_commons.config_base import (
    BaseDataConfig,
    EnvReader,
    create_config_singleton,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------

_ENV_PREFIX = "GL_CLIMATE_HAZARD_"

# ---------------------------------------------------------------------------
# Valid log levels
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)

# ---------------------------------------------------------------------------
# Valid climate scenarios (IPCC SSP + legacy RCP)
# ---------------------------------------------------------------------------

_VALID_SCENARIOS = frozenset(
    {
        "SSP1-1.9",
        "SSP1-2.6",
        "SSP2-4.5",
        "SSP3-7.0",
        "SSP5-8.5",
        "RCP2.6",
        "RCP4.5",
        "RCP6.0",
        "RCP8.5",
    }
)

# ---------------------------------------------------------------------------
# Valid time horizons
# ---------------------------------------------------------------------------

_VALID_TIME_HORIZONS = frozenset(
    {"SHORT_TERM", "MID_TERM", "LONG_TERM"}
)

# ---------------------------------------------------------------------------
# Valid report formats
# ---------------------------------------------------------------------------

_VALID_REPORT_FORMATS = frozenset(
    {"json", "csv", "pdf"}
)


# ---------------------------------------------------------------------------
# ClimateHazardConfig
# ---------------------------------------------------------------------------


@dataclass
class ClimateHazardConfig(BaseDataConfig):
    """Configuration for the GreenLang Climate Hazard Connector Agent SDK.

    Inherits shared connection, pool, batch, and logging fields from
    ``BaseDataConfig``.  Only climate-hazard-specific fields are declared here.

    All attributes can be overridden via environment variables using the
    ``GL_CLIMATE_HAZARD_`` prefix (e.g. ``GL_CLIMATE_HAZARD_MAX_ASSETS=20000``).

    Attributes:
        default_scenario: Default IPCC climate scenario for hazard projections.
        default_time_horizon: Default temporal projection window.
        default_report_format: Default output format for generated reports.
        max_hazard_sources: Maximum number of external hazard data sources.
        max_assets: Maximum assets per single pipeline run.
        max_risk_indices: Maximum risk index records in persistent storage.
        risk_weight_probability: Weight for hazard probability (0.0-1.0).
        risk_weight_intensity: Weight for hazard intensity (0.0-1.0).
        risk_weight_frequency: Weight for hazard frequency (0.0-1.0).
        risk_weight_duration: Weight for hazard duration (0.0-1.0).
        vuln_weight_exposure: Weight for exposure (0.0-1.0).
        vuln_weight_sensitivity: Weight for sensitivity (0.0-1.0).
        vuln_weight_adaptive: Weight for adaptive capacity (0.0-1.0).
        threshold_extreme: Score threshold for EXTREME risk (0-100).
        threshold_high: Score threshold for HIGH risk (0-100).
        threshold_medium: Score threshold for MEDIUM risk (0-100).
        threshold_low: Score threshold for LOW risk (0-100).
        max_pipeline_runs: Maximum concurrent or queued pipeline runs.
        max_reports: Maximum generated reports retained.
        enable_provenance: Whether SHA-256 provenance tracking is enabled.
        genesis_hash: Anchor string for every provenance chain.
        enable_metrics: Whether Prometheus metrics are exported.
        pool_size: PostgreSQL connection pool size.
        cache_ttl: TTL (seconds) for cached hazard lookups.
        rate_limit: Maximum inbound API requests per minute.
    """

    # -- Climate scenario defaults -------------------------------------------
    default_scenario: str = "SSP2-4.5"
    default_time_horizon: str = "MID_TERM"

    # -- Report format -------------------------------------------------------
    default_report_format: str = "json"

    # -- Hazard source capacity ----------------------------------------------
    max_hazard_sources: int = 50

    # -- Asset capacity ------------------------------------------------------
    max_assets: int = 10_000

    # -- Risk index capacity -------------------------------------------------
    max_risk_indices: int = 5_000

    # -- Risk index weights --------------------------------------------------
    # Must sum to 1.0 (validated in __post_init__)
    risk_weight_probability: float = 0.30
    risk_weight_intensity: float = 0.30
    risk_weight_frequency: float = 0.25
    risk_weight_duration: float = 0.15

    # -- Vulnerability weights -----------------------------------------------
    # Must sum to 1.0 (validated in __post_init__)
    vuln_weight_exposure: float = 0.40
    vuln_weight_sensitivity: float = 0.35
    vuln_weight_adaptive: float = 0.25

    # -- Risk classification thresholds (0-100 scale) ------------------------
    # Scores: >= extreme -> EXTREME, >= high -> HIGH, >= medium -> MEDIUM,
    #          >= low -> LOW, < low -> NEGLIGIBLE
    # Must satisfy: extreme > high > medium > low > 0
    threshold_extreme: float = 80.0
    threshold_high: float = 60.0
    threshold_medium: float = 40.0
    threshold_low: float = 20.0

    # -- Pipeline processing limits ------------------------------------------
    max_pipeline_runs: int = 500
    max_reports: int = 1_000

    # -- Provenance tracking -------------------------------------------------
    enable_provenance: bool = True
    genesis_hash: str = "greenlang-climate-hazard-genesis"

    # -- Metrics export ------------------------------------------------------
    enable_metrics: bool = True

    # -- Performance tuning --------------------------------------------------
    pool_size: int = 5
    cache_ttl: int = 300
    rate_limit: int = 200

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration constraints after initialisation.

        Raises:
            ValueError: If any configuration value is outside its valid
                range or violates a relational constraint.
        """
        errors: list[str] = []

        # -- Connections -----------------------------------------------------
        if not self.database_url:
            logger.warning(
                "ClimateHazardConfig: database_url is empty; "
                "the service will fail to connect until "
                "GL_CLIMATE_HAZARD_DATABASE_URL is set."
            )
        if not self.redis_url:
            logger.warning(
                "ClimateHazardConfig: redis_url is empty; "
                "caching and distributed locks are disabled."
            )

        # -- Logging ---------------------------------------------------------
        normalised_log = self.log_level.upper()
        if normalised_log not in _VALID_LOG_LEVELS:
            errors.append(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
                f"got '{self.log_level}'"
            )
        else:
            self.log_level = normalised_log

        # -- Climate scenario ------------------------------------------------
        if self.default_scenario not in _VALID_SCENARIOS:
            errors.append(
                f"default_scenario must be one of "
                f"{sorted(_VALID_SCENARIOS)}, "
                f"got '{self.default_scenario}'"
            )

        # -- Time horizon ----------------------------------------------------
        normalised_horizon = self.default_time_horizon.upper()
        if normalised_horizon not in _VALID_TIME_HORIZONS:
            errors.append(
                f"default_time_horizon must be one of "
                f"{sorted(_VALID_TIME_HORIZONS)}, "
                f"got '{self.default_time_horizon}'"
            )
        else:
            self.default_time_horizon = normalised_horizon

        # -- Report format ---------------------------------------------------
        normalised_format = self.default_report_format.lower()
        if normalised_format not in _VALID_REPORT_FORMATS:
            errors.append(
                f"default_report_format must be one of "
                f"{sorted(_VALID_REPORT_FORMATS)}, "
                f"got '{self.default_report_format}'"
            )
        else:
            self.default_report_format = normalised_format

        # -- Hazard source capacity ------------------------------------------
        if self.max_hazard_sources <= 0:
            errors.append(
                f"max_hazard_sources must be > 0, "
                f"got {self.max_hazard_sources}"
            )
        if self.max_hazard_sources > 1000:
            errors.append(
                f"max_hazard_sources must be <= 1000, "
                f"got {self.max_hazard_sources}"
            )

        # -- Asset capacity --------------------------------------------------
        if self.max_assets <= 0:
            errors.append(
                f"max_assets must be > 0, got {self.max_assets}"
            )
        if self.max_assets > 1_000_000:
            errors.append(
                f"max_assets must be <= 1000000, "
                f"got {self.max_assets}"
            )

        # -- Risk index capacity ---------------------------------------------
        if self.max_risk_indices <= 0:
            errors.append(
                f"max_risk_indices must be > 0, "
                f"got {self.max_risk_indices}"
            )

        # -- Risk index weights ----------------------------------------------
        for wname, wval in [
            ("risk_weight_probability", self.risk_weight_probability),
            ("risk_weight_intensity", self.risk_weight_intensity),
            ("risk_weight_frequency", self.risk_weight_frequency),
            ("risk_weight_duration", self.risk_weight_duration),
        ]:
            if not (0.0 <= wval <= 1.0):
                errors.append(
                    f"{wname} must be in [0.0, 1.0], got {wval}"
                )

        risk_weight_sum = (
            self.risk_weight_probability
            + self.risk_weight_intensity
            + self.risk_weight_frequency
            + self.risk_weight_duration
        )
        if abs(risk_weight_sum - 1.0) > 1e-6:
            errors.append(
                f"risk index weights must sum to 1.0, "
                f"got {risk_weight_sum:.6f} "
                f"(probability={self.risk_weight_probability}, "
                f"intensity={self.risk_weight_intensity}, "
                f"frequency={self.risk_weight_frequency}, "
                f"duration={self.risk_weight_duration})"
            )

        # -- Vulnerability weights -------------------------------------------
        for wname, wval in [
            ("vuln_weight_exposure", self.vuln_weight_exposure),
            ("vuln_weight_sensitivity", self.vuln_weight_sensitivity),
            ("vuln_weight_adaptive", self.vuln_weight_adaptive),
        ]:
            if not (0.0 <= wval <= 1.0):
                errors.append(
                    f"{wname} must be in [0.0, 1.0], got {wval}"
                )

        vuln_weight_sum = (
            self.vuln_weight_exposure
            + self.vuln_weight_sensitivity
            + self.vuln_weight_adaptive
        )
        if abs(vuln_weight_sum - 1.0) > 1e-6:
            errors.append(
                f"vulnerability weights must sum to 1.0, "
                f"got {vuln_weight_sum:.6f} "
                f"(exposure={self.vuln_weight_exposure}, "
                f"sensitivity={self.vuln_weight_sensitivity}, "
                f"adaptive={self.vuln_weight_adaptive})"
            )

        # -- Risk classification thresholds ----------------------------------
        for tname, tval in [
            ("threshold_extreme", self.threshold_extreme),
            ("threshold_high", self.threshold_high),
            ("threshold_medium", self.threshold_medium),
            ("threshold_low", self.threshold_low),
        ]:
            if not (0.0 <= tval <= 100.0):
                errors.append(
                    f"{tname} must be in [0.0, 100.0], got {tval}"
                )

        # Strict ordering: extreme > high > medium > low > 0
        if (
            0.0 <= self.threshold_extreme <= 100.0
            and 0.0 <= self.threshold_high <= 100.0
            and self.threshold_extreme <= self.threshold_high
        ):
            errors.append(
                f"threshold_extreme ({self.threshold_extreme}) must be "
                f"strictly greater than threshold_high "
                f"({self.threshold_high})"
            )
        if (
            0.0 <= self.threshold_high <= 100.0
            and 0.0 <= self.threshold_medium <= 100.0
            and self.threshold_high <= self.threshold_medium
        ):
            errors.append(
                f"threshold_high ({self.threshold_high}) must be "
                f"strictly greater than threshold_medium "
                f"({self.threshold_medium})"
            )
        if (
            0.0 <= self.threshold_medium <= 100.0
            and 0.0 <= self.threshold_low <= 100.0
            and self.threshold_medium <= self.threshold_low
        ):
            errors.append(
                f"threshold_medium ({self.threshold_medium}) must be "
                f"strictly greater than threshold_low "
                f"({self.threshold_low})"
            )
        if (
            0.0 <= self.threshold_low <= 100.0
            and self.threshold_low <= 0.0
        ):
            errors.append(
                f"threshold_low ({self.threshold_low}) must be "
                f"strictly greater than 0"
            )

        # -- Pipeline processing limits --------------------------------------
        if self.max_pipeline_runs <= 0:
            errors.append(
                f"max_pipeline_runs must be > 0, "
                f"got {self.max_pipeline_runs}"
            )
        if self.max_pipeline_runs > 10_000:
            errors.append(
                f"max_pipeline_runs must be <= 10000, "
                f"got {self.max_pipeline_runs}"
            )
        if self.max_reports <= 0:
            errors.append(
                f"max_reports must be > 0, got {self.max_reports}"
            )
        if self.max_reports > 100_000:
            errors.append(
                f"max_reports must be <= 100000, "
                f"got {self.max_reports}"
            )

        # -- Provenance ------------------------------------------------------
        if not self.genesis_hash:
            errors.append("genesis_hash must not be empty")

        # -- Performance -----------------------------------------------------
        if self.pool_size <= 0:
            errors.append(
                f"pool_size must be > 0, got {self.pool_size}"
            )
        if self.cache_ttl <= 0:
            errors.append(
                f"cache_ttl must be > 0, got {self.cache_ttl}"
            )
        if self.rate_limit <= 0:
            errors.append(
                f"rate_limit must be > 0, got {self.rate_limit}"
            )

        if errors:
            raise ValueError(
                "ClimateHazardConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.debug(
            "ClimateHazardConfig validated successfully: "
            "scenario=%s, time_horizon=%s, report_format=%s, "
            "max_hazard_sources=%d, max_assets=%d, "
            "max_risk_indices=%d, "
            "risk_weights=(prob=%.2f, int=%.2f, freq=%.2f, dur=%.2f), "
            "vuln_weights=(exp=%.2f, sens=%.2f, adapt=%.2f), "
            "thresholds=(extreme=%.1f, high=%.1f, med=%.1f, low=%.1f), "
            "max_pipeline_runs=%d, max_reports=%d, "
            "provenance=%s, metrics=%s",
            self.default_scenario,
            self.default_time_horizon,
            self.default_report_format,
            self.max_hazard_sources,
            self.max_assets,
            self.max_risk_indices,
            self.risk_weight_probability,
            self.risk_weight_intensity,
            self.risk_weight_frequency,
            self.risk_weight_duration,
            self.vuln_weight_exposure,
            self.vuln_weight_sensitivity,
            self.vuln_weight_adaptive,
            self.threshold_extreme,
            self.threshold_high,
            self.threshold_medium,
            self.threshold_low,
            self.max_pipeline_runs,
            self.max_reports,
            self.enable_provenance,
            self.enable_metrics,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> ClimateHazardConfig:
        """Build a ClimateHazardConfig from environment variables.

        Every field can be overridden via ``GL_CLIMATE_HAZARD_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).

        Returns:
            Populated ClimateHazardConfig instance, validated via
            ``__post_init__``.

        Example:
            >>> import os
            >>> os.environ["GL_CLIMATE_HAZARD_MAX_ASSETS"] = "20000"
            >>> cfg = ClimateHazardConfig.from_env()
            >>> cfg.max_assets
            20000
        """
        env = EnvReader(_ENV_PREFIX)
        base_kwargs = cls._base_kwargs_from_env(env)

        config = cls(
            **base_kwargs,
            # Climate scenario defaults
            default_scenario=env.str(
                "DEFAULT_SCENARIO",
                cls.default_scenario,
            ),
            default_time_horizon=env.str(
                "DEFAULT_TIME_HORIZON",
                cls.default_time_horizon,
            ),
            # Report format
            default_report_format=env.str(
                "DEFAULT_REPORT_FORMAT",
                cls.default_report_format,
            ),
            # Hazard source capacity
            max_hazard_sources=env.int(
                "MAX_HAZARD_SOURCES",
                cls.max_hazard_sources,
            ),
            # Asset capacity
            max_assets=env.int("MAX_ASSETS", cls.max_assets),
            # Risk index capacity
            max_risk_indices=env.int(
                "MAX_RISK_INDICES",
                cls.max_risk_indices,
            ),
            # Risk index weights
            risk_weight_probability=env.float(
                "RISK_WEIGHT_PROBABILITY",
                cls.risk_weight_probability,
            ),
            risk_weight_intensity=env.float(
                "RISK_WEIGHT_INTENSITY",
                cls.risk_weight_intensity,
            ),
            risk_weight_frequency=env.float(
                "RISK_WEIGHT_FREQUENCY",
                cls.risk_weight_frequency,
            ),
            risk_weight_duration=env.float(
                "RISK_WEIGHT_DURATION",
                cls.risk_weight_duration,
            ),
            # Vulnerability weights
            vuln_weight_exposure=env.float(
                "VULN_WEIGHT_EXPOSURE",
                cls.vuln_weight_exposure,
            ),
            vuln_weight_sensitivity=env.float(
                "VULN_WEIGHT_SENSITIVITY",
                cls.vuln_weight_sensitivity,
            ),
            vuln_weight_adaptive=env.float(
                "VULN_WEIGHT_ADAPTIVE",
                cls.vuln_weight_adaptive,
            ),
            # Risk classification thresholds
            threshold_extreme=env.float(
                "THRESHOLD_EXTREME",
                cls.threshold_extreme,
            ),
            threshold_high=env.float(
                "THRESHOLD_HIGH",
                cls.threshold_high,
            ),
            threshold_medium=env.float(
                "THRESHOLD_MEDIUM",
                cls.threshold_medium,
            ),
            threshold_low=env.float(
                "THRESHOLD_LOW",
                cls.threshold_low,
            ),
            # Pipeline processing limits
            max_pipeline_runs=env.int(
                "MAX_PIPELINE_RUNS",
                cls.max_pipeline_runs,
            ),
            max_reports=env.int("MAX_REPORTS", cls.max_reports),
            # Provenance tracking
            enable_provenance=env.bool(
                "ENABLE_PROVENANCE",
                cls.enable_provenance,
            ),
            genesis_hash=env.str("GENESIS_HASH", cls.genesis_hash),
            # Metrics export
            enable_metrics=env.bool("ENABLE_METRICS", cls.enable_metrics),
            # Performance tuning
            pool_size=env.int("POOL_SIZE", cls.pool_size),
            cache_ttl=env.int("CACHE_TTL", cls.cache_ttl),
            rate_limit=env.int("RATE_LIMIT", cls.rate_limit),
        )

        logger.info(
            "ClimateHazardConfig loaded: "
            "scenario=%s, time_horizon=%s, report_format=%s, "
            "max_hazard_sources=%d, max_assets=%d, "
            "max_risk_indices=%d, "
            "risk_weights=(prob=%.2f, int=%.2f, freq=%.2f, dur=%.2f), "
            "vuln_weights=(exp=%.2f, sens=%.2f, adapt=%.2f), "
            "thresholds=(extreme=%.1f, high=%.1f, med=%.1f, low=%.1f), "
            "max_pipeline_runs=%d, max_reports=%d, "
            "provenance=%s, "
            "pool=%d, cache_ttl=%ds, rate_limit=%d/min, "
            "metrics=%s",
            config.default_scenario,
            config.default_time_horizon,
            config.default_report_format,
            config.max_hazard_sources,
            config.max_assets,
            config.max_risk_indices,
            config.risk_weight_probability,
            config.risk_weight_intensity,
            config.risk_weight_frequency,
            config.risk_weight_duration,
            config.vuln_weight_exposure,
            config.vuln_weight_sensitivity,
            config.vuln_weight_adaptive,
            config.threshold_extreme,
            config.threshold_high,
            config.threshold_medium,
            config.threshold_low,
            config.max_pipeline_runs,
            config.max_reports,
            config.enable_provenance,
            config.pool_size,
            config.cache_ttl,
            config.rate_limit,
            config.enable_metrics,
        )
        return config

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a plain Python dictionary.

        Sensitive connection strings (``database_url``, ``redis_url``) are
        redacted to prevent accidental credential leakage in logs.

        Returns:
            Dictionary representation of the configuration with sensitive
            fields redacted.

        Example:
            >>> cfg = ClimateHazardConfig()
            >>> d = cfg.to_dict()
            >>> d["max_assets"]
            10000
            >>> d["database_url"]  # redacted
            '***'
        """
        return {
            # -- Connections (redacted) --------------------------------------
            "database_url": "***" if self.database_url else "",
            "redis_url": "***" if self.redis_url else "",
            # -- Logging -----------------------------------------------------
            "log_level": self.log_level,
            # -- Climate scenario defaults -----------------------------------
            "default_scenario": self.default_scenario,
            "default_time_horizon": self.default_time_horizon,
            # -- Report format -----------------------------------------------
            "default_report_format": self.default_report_format,
            # -- Hazard source capacity --------------------------------------
            "max_hazard_sources": self.max_hazard_sources,
            # -- Asset capacity ----------------------------------------------
            "max_assets": self.max_assets,
            # -- Risk index capacity -----------------------------------------
            "max_risk_indices": self.max_risk_indices,
            # -- Risk index weights ------------------------------------------
            "risk_weight_probability": self.risk_weight_probability,
            "risk_weight_intensity": self.risk_weight_intensity,
            "risk_weight_frequency": self.risk_weight_frequency,
            "risk_weight_duration": self.risk_weight_duration,
            # -- Vulnerability weights ---------------------------------------
            "vuln_weight_exposure": self.vuln_weight_exposure,
            "vuln_weight_sensitivity": self.vuln_weight_sensitivity,
            "vuln_weight_adaptive": self.vuln_weight_adaptive,
            # -- Risk classification thresholds ------------------------------
            "threshold_extreme": self.threshold_extreme,
            "threshold_high": self.threshold_high,
            "threshold_medium": self.threshold_medium,
            "threshold_low": self.threshold_low,
            # -- Pipeline processing limits ----------------------------------
            "max_pipeline_runs": self.max_pipeline_runs,
            "max_reports": self.max_reports,
            # -- Provenance tracking -----------------------------------------
            "enable_provenance": self.enable_provenance,
            "genesis_hash": self.genesis_hash,
            # -- Metrics export ----------------------------------------------
            "enable_metrics": self.enable_metrics,
            # -- Performance tuning ------------------------------------------
            "pool_size": self.pool_size,
            "cache_ttl": self.cache_ttl,
            "rate_limit": self.rate_limit,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly, credential-safe representation.

        Sensitive fields (database_url, redis_url) are replaced with
        ``'***'`` so that repr output is safe to include in log messages
        and exception tracebacks.

        Returns:
            String representation of the configuration.
        """
        d = self.to_dict()
        pairs = ", ".join(f"{k}={v!r}" for k, v in d.items())
        return f"ClimateHazardConfig({pairs})"


# ---------------------------------------------------------------------------
# Thread-safe singleton accessor
# ---------------------------------------------------------------------------

get_config, set_config, reset_config = create_config_singleton(
    ClimateHazardConfig, _ENV_PREFIX,
)

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ClimateHazardConfig",
    "get_config",
    "set_config",
    "reset_config",
]
