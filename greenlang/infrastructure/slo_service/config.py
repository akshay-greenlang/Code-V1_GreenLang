# -*- coding: utf-8 -*-
"""
SLO Service Configuration - OBS-005: SLO/SLI Definitions & Error Budget Management

Centralized configuration for the SLO/SLI service covering:
- Prometheus connection for SLI metric queries
- Grafana connection for dashboard generation
- Redis connection for budget caching
- Database connection for persistence (TimescaleDB)
- Burn rate window definitions (fast, medium, slow)
- Error budget thresholds (warning, critical, exhausted)
- Budget exhaustion policy actions
- Compliance reporting settings

All settings can be overridden via environment variables with the
``GL_SLO_`` prefix (e.g. ``GL_SLO_PROMETHEUS_URL``).

Example:
    >>> from greenlang.infrastructure.slo_service.config import get_config
    >>> cfg = get_config()
    >>> print(cfg.prometheus_url, cfg.budget_threshold_warning)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
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
# Default burn rate window definitions (Google SRE Book)
# ---------------------------------------------------------------------------

_DEFAULT_BURN_RATE_WINDOWS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "long_window": "1h",
        "short_window": "5m",
        "threshold": 14.4,
        "long_window_minutes": 60,
        "short_window_minutes": 5,
    },
    "medium": {
        "long_window": "6h",
        "short_window": "30m",
        "threshold": 6.0,
        "long_window_minutes": 360,
        "short_window_minutes": 30,
    },
    "slow": {
        "long_window": "3d",
        "short_window": "6h",
        "threshold": 1.0,
        "long_window_minutes": 4320,
        "short_window_minutes": 360,
    },
}


# ---------------------------------------------------------------------------
# SLOServiceConfig
# ---------------------------------------------------------------------------


@dataclass
class SLOServiceConfig:
    """Complete configuration for the GreenLang SLO/SLI Service.

    Attributes are grouped by concern: general, Prometheus, Grafana, Redis,
    database, burn rate windows, error budget thresholds, budget policies,
    compliance reporting, evaluation intervals, and output paths.

    All attributes can be overridden via environment variables using the
    ``GL_SLO_`` prefix.
    """

    # -- General -------------------------------------------------------------
    service_name: str = "greenlang-slo"
    environment: str = "production"
    enabled: bool = True

    # -- Prometheus ----------------------------------------------------------
    prometheus_url: str = "http://prometheus:9090"
    prometheus_timeout_seconds: int = 30
    prometheus_verify_ssl: bool = True

    # -- Grafana -------------------------------------------------------------
    grafana_url: str = "http://grafana:3000"
    grafana_api_key: str = ""
    grafana_org_id: int = 1
    grafana_folder_uid: str = "slo-dashboards"

    # -- Redis ---------------------------------------------------------------
    redis_url: str = "redis://redis:6379/3"
    redis_key_prefix: str = "gl:slo:"
    redis_cache_ttl_seconds: int = 60

    # -- Database (TimescaleDB) ----------------------------------------------
    database_url: str = "postgresql://greenlang:greenlang@localhost:5432/greenlang"
    database_pool_size: int = 5

    # -- SLO definitions YAML -----------------------------------------------
    slo_definitions_path: str = "slo_definitions.yaml"
    slo_reload_interval_seconds: int = 300

    # -- Burn rate windows ---------------------------------------------------
    burn_rate_windows: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: dict(_DEFAULT_BURN_RATE_WINDOWS),
    )

    # -- Error budget thresholds (percent consumed) --------------------------
    budget_threshold_warning: float = 20.0
    budget_threshold_critical: float = 50.0
    budget_threshold_exhausted: float = 100.0

    # -- Budget exhaustion action --------------------------------------------
    # Options: "freeze_deployments", "alert_only", "none"
    budget_exhausted_action: str = "alert_only"

    # -- Evaluation ----------------------------------------------------------
    evaluation_interval_seconds: int = 60
    evaluation_batch_size: int = 50

    # -- Compliance reporting ------------------------------------------------
    compliance_enabled: bool = True
    compliance_weekly_day: str = "monday"
    compliance_retention_days: int = 365

    # -- Recording rules output ----------------------------------------------
    recording_rules_output_path: str = "/etc/prometheus/rules/slo_recording_rules.yaml"
    alert_rules_output_path: str = "/etc/prometheus/rules/slo_alert_rules.yaml"

    # -- Dashboard output ----------------------------------------------------
    dashboards_output_dir: str = "/var/lib/grafana/dashboards/slo"

    # -- Alerting bridge (OBS-004 integration) -------------------------------
    alerting_bridge_enabled: bool = True

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> SLOServiceConfig:
        """Build an SLOServiceConfig from environment variables.

        Every field can be overridden via ``GL_SLO_<FIELD_UPPER>``.
        Boolean values accept ``true/1/yes`` (case-insensitive).
        Integer values are parsed via ``int()``.
        Float values are parsed via ``float()``.

        Returns:
            Populated SLOServiceConfig instance.
        """
        prefix = "GL_SLO_"

        def _env(name: str, default: Any = None) -> Optional[str]:
            return os.environ.get(f"{prefix}{name}", default)

        def _bool(name: str, default: bool) -> bool:
            val = _env(name)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def _int(name: str, default: int) -> int:
            val = _env(name)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                logger.warning(
                    "Invalid integer for %s%s=%s, using default %d",
                    prefix, name, val, default,
                )
                return default

        def _float(name: str, default: float) -> float:
            val = _env(name)
            if val is None:
                return default
            try:
                return float(val)
            except ValueError:
                logger.warning(
                    "Invalid float for %s%s=%s, using default %.1f",
                    prefix, name, val, default,
                )
                return default

        config = cls(
            service_name=_env("SERVICE_NAME", cls.service_name) or cls.service_name,
            environment=_env("ENVIRONMENT", cls.environment) or cls.environment,
            enabled=_bool("ENABLED", cls.enabled),
            # Prometheus
            prometheus_url=_env("PROMETHEUS_URL", cls.prometheus_url) or cls.prometheus_url,
            prometheus_timeout_seconds=_int("PROMETHEUS_TIMEOUT_SECONDS", cls.prometheus_timeout_seconds),
            prometheus_verify_ssl=_bool("PROMETHEUS_VERIFY_SSL", cls.prometheus_verify_ssl),
            # Grafana
            grafana_url=_env("GRAFANA_URL", cls.grafana_url) or cls.grafana_url,
            grafana_api_key=_env("GRAFANA_API_KEY", "") or "",
            grafana_org_id=_int("GRAFANA_ORG_ID", cls.grafana_org_id),
            grafana_folder_uid=_env("GRAFANA_FOLDER_UID", cls.grafana_folder_uid) or cls.grafana_folder_uid,
            # Redis
            redis_url=_env("REDIS_URL", cls.redis_url) or cls.redis_url,
            redis_key_prefix=_env("REDIS_KEY_PREFIX", cls.redis_key_prefix) or cls.redis_key_prefix,
            redis_cache_ttl_seconds=_int("REDIS_CACHE_TTL_SECONDS", cls.redis_cache_ttl_seconds),
            # Database
            database_url=_env("DATABASE_URL", cls.database_url) or cls.database_url,
            database_pool_size=_int("DATABASE_POOL_SIZE", cls.database_pool_size),
            # SLO definitions
            slo_definitions_path=_env("SLO_DEFINITIONS_PATH", cls.slo_definitions_path) or cls.slo_definitions_path,
            slo_reload_interval_seconds=_int("SLO_RELOAD_INTERVAL_SECONDS", cls.slo_reload_interval_seconds),
            # Error budget thresholds
            budget_threshold_warning=_float("BUDGET_THRESHOLD_WARNING", cls.budget_threshold_warning),
            budget_threshold_critical=_float("BUDGET_THRESHOLD_CRITICAL", cls.budget_threshold_critical),
            budget_threshold_exhausted=_float("BUDGET_THRESHOLD_EXHAUSTED", cls.budget_threshold_exhausted),
            # Budget exhaustion action
            budget_exhausted_action=_env("BUDGET_EXHAUSTED_ACTION", cls.budget_exhausted_action) or cls.budget_exhausted_action,
            # Evaluation
            evaluation_interval_seconds=_int("EVALUATION_INTERVAL_SECONDS", cls.evaluation_interval_seconds),
            evaluation_batch_size=_int("EVALUATION_BATCH_SIZE", cls.evaluation_batch_size),
            # Compliance
            compliance_enabled=_bool("COMPLIANCE_ENABLED", cls.compliance_enabled),
            compliance_weekly_day=_env("COMPLIANCE_WEEKLY_DAY", cls.compliance_weekly_day) or cls.compliance_weekly_day,
            compliance_retention_days=_int("COMPLIANCE_RETENTION_DAYS", cls.compliance_retention_days),
            # Recording rules
            recording_rules_output_path=_env("RECORDING_RULES_OUTPUT_PATH", cls.recording_rules_output_path) or cls.recording_rules_output_path,
            alert_rules_output_path=_env("ALERT_RULES_OUTPUT_PATH", cls.alert_rules_output_path) or cls.alert_rules_output_path,
            # Dashboards
            dashboards_output_dir=_env("DASHBOARDS_OUTPUT_DIR", cls.dashboards_output_dir) or cls.dashboards_output_dir,
            # Alerting bridge
            alerting_bridge_enabled=_bool("ALERTING_BRIDGE_ENABLED", cls.alerting_bridge_enabled),
        )

        logger.info(
            "SLOServiceConfig loaded: env=%s, prometheus=%s, "
            "evaluation_interval=%ds, budget_exhausted_action=%s",
            config.environment,
            config.prometheus_url,
            config.evaluation_interval_seconds,
            config.budget_exhausted_action,
        )
        return config


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_config_instance: Optional[SLOServiceConfig] = None
_config_lock = threading.Lock()


def get_config() -> SLOServiceConfig:
    """Return the singleton SLOServiceConfig, creating from env if needed.

    Returns:
        SLOServiceConfig singleton instance.
    """
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = SLOServiceConfig.from_env()
    return _config_instance


def set_config(config: SLOServiceConfig) -> None:
    """Replace the singleton SLOServiceConfig (useful for testing).

    Args:
        config: New configuration to install.
    """
    global _config_instance
    with _config_lock:
        _config_instance = config
    logger.info("SLOServiceConfig replaced programmatically")


def reset_config() -> None:
    """Reset the singleton (primarily for test teardown)."""
    global _config_instance
    with _config_lock:
        _config_instance = None
