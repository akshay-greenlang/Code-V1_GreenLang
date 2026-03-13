# -*- coding: utf-8 -*-
"""
Unit tests for GrievanceMechanismManagerConfig - AGENT-EUDR-032

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
    get_config,
    reset_config,
    _env,
    _env_int,
    _env_float,
    _env_bool,
    _env_decimal,
    _ENV_PREFIX,
)


class TestConfigDefaults:
    def test_db_host_default(self, sample_config):
        assert sample_config.db_host == "localhost"

    def test_db_port_default(self, sample_config):
        assert sample_config.db_port == 5432

    def test_db_name_default(self, sample_config):
        assert sample_config.db_name == "greenlang"

    def test_db_user_default(self, sample_config):
        assert sample_config.db_user == "gl"

    def test_db_pool_min_default(self, sample_config):
        assert sample_config.db_pool_min == 2

    def test_db_pool_max_default(self, sample_config):
        assert sample_config.db_pool_max == 10

    def test_redis_host_default(self, sample_config):
        assert sample_config.redis_host == "localhost"

    def test_redis_port_default(self, sample_config):
        assert sample_config.redis_port == 6379

    def test_cache_ttl_default(self, sample_config):
        assert sample_config.cache_ttl == 3600

    def test_analytics_window_default(self, sample_config):
        assert sample_config.analytics_default_window_days == 90

    def test_analytics_min_grievances_default(self, sample_config):
        assert sample_config.analytics_min_grievances_for_pattern == 3

    def test_root_cause_max_depth_default(self, sample_config):
        assert sample_config.root_cause_max_depth == 5

    def test_root_cause_min_confidence_default(self, sample_config):
        assert sample_config.root_cause_min_confidence == Decimal("50")

    def test_root_cause_default_method(self, sample_config):
        assert sample_config.root_cause_default_method == "five_whys"

    def test_mediation_max_sessions_default(self, sample_config):
        assert sample_config.mediation_max_sessions == 20

    def test_mediation_session_minutes_default(self, sample_config):
        assert sample_config.mediation_default_session_minutes == 120

    def test_remediation_verify_required(self, sample_config):
        assert sample_config.remediation_verification_required is True

    def test_remediation_min_satisfaction(self, sample_config):
        assert sample_config.remediation_min_satisfaction == Decimal("3.0")

    def test_risk_window_days_default(self, sample_config):
        assert sample_config.risk_scoring_window_days == 180

    def test_risk_weights_sum_to_one(self, sample_config):
        total = (
            sample_config.risk_weight_frequency + sample_config.risk_weight_severity
            + sample_config.risk_weight_resolution + sample_config.risk_weight_escalation
            + sample_config.risk_weight_unresolved
        )
        assert abs(total - Decimal("1.0")) < Decimal("0.01")

    def test_risk_level_thresholds(self, sample_config):
        assert sample_config.risk_level_negligible_max == Decimal("15")
        assert sample_config.risk_level_low_max == Decimal("30")
        assert sample_config.risk_level_moderate_max == Decimal("60")
        assert sample_config.risk_level_high_max == Decimal("80")

    def test_collective_min_stakeholders(self, sample_config):
        assert sample_config.collective_min_stakeholders == 3

    def test_retention_years_default(self, sample_config):
        assert sample_config.retention_years == 5

    def test_metrics_prefix(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_gmm_"

    def test_provenance_enabled(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_rate_limits(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10
        assert sample_config.rate_limit_basic == 30
        assert sample_config.rate_limit_standard == 100
        assert sample_config.rate_limit_premium == 500
        assert sample_config.rate_limit_admin == 2000

    def test_circuit_breaker_defaults(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5
        assert sample_config.circuit_breaker_reset_timeout == 60


class TestConfigEnvOverrides:
    def test_env_prefix(self):
        assert _ENV_PREFIX == "GL_EUDR_GMM_"

    def test_env_int_override(self):
        with patch.dict(os.environ, {"GL_EUDR_GMM_DB_PORT": "5433"}):
            cfg = GrievanceMechanismManagerConfig()
            assert cfg.db_port == 5433

    def test_env_bool_override(self):
        with patch.dict(os.environ, {"GL_EUDR_GMM_PROVENANCE_ENABLED": "false"}):
            cfg = GrievanceMechanismManagerConfig()
            assert cfg.provenance_enabled is False

    def test_env_decimal_override(self):
        with patch.dict(os.environ, {"GL_EUDR_GMM_RC_MIN_CONFIDENCE": "75"}):
            cfg = GrievanceMechanismManagerConfig()
            assert cfg.root_cause_min_confidence == Decimal("75")

    def test_env_helper_returns_default(self):
        assert _env("NONEXISTENT", "default") == "default"

    def test_env_int_returns_default(self):
        assert _env_int("NONEXISTENT", 42) == 42

    def test_env_float_returns_default(self):
        assert _env_float("NONEXISTENT", 3.14) == 3.14

    def test_env_bool_returns_default(self):
        assert _env_bool("NONEXISTENT", True) is True

    def test_env_decimal_returns_default(self):
        assert _env_decimal("NONEXISTENT", "1.23") == Decimal("1.23")


class TestConfigSingleton:
    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, GrievanceMechanismManagerConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_singleton(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2


class TestConfigMethods:
    def test_get_risk_level_negligible(self, sample_config):
        assert sample_config.get_risk_level(Decimal("10")) == "negligible"

    def test_get_risk_level_low(self, sample_config):
        assert sample_config.get_risk_level(Decimal("25")) == "low"

    def test_get_risk_level_moderate(self, sample_config):
        assert sample_config.get_risk_level(Decimal("50")) == "moderate"

    def test_get_risk_level_high(self, sample_config):
        assert sample_config.get_risk_level(Decimal("75")) == "high"

    def test_get_risk_level_critical(self, sample_config):
        assert sample_config.get_risk_level(Decimal("90")) == "critical"

    def test_get_mediation_stage_sla(self, sample_config):
        assert sample_config.get_mediation_stage_sla("preparation") == 7
        assert sample_config.get_mediation_stage_sla("dialogue") == 14
        assert sample_config.get_mediation_stage_sla("negotiation") == 21
        assert sample_config.get_mediation_stage_sla("settlement") == 14
        assert sample_config.get_mediation_stage_sla("implementation") == 30
        assert sample_config.get_mediation_stage_sla("unknown") == 14

    def test_get_risk_weights(self, sample_config):
        weights = sample_config.get_risk_weights()
        assert "frequency" in weights
        assert "severity" in weights
        assert len(weights) == 5
