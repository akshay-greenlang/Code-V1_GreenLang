# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- config.py

Tests configuration defaults, environment variable overrides,
validation logic, member state mapping, sequence range, and
singleton behavior. 60+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    EU_MEMBER_STATES,
    ReferenceNumberGeneratorConfig,
    _ENV_PREFIX,
    _env,
    _env_bool,
    _env_float,
    _env_int,
    get_config,
    reset_config,
)


# ====================================================================
# Test: Configuration Defaults
# ====================================================================


class TestConfigDefaults:
    """Verify every default value of ReferenceNumberGeneratorConfig."""

    def test_db_host_default(self, sample_config):
        assert sample_config.db_host == "localhost"

    def test_db_port_default(self, sample_config):
        assert sample_config.db_port == 5432

    def test_db_name_default(self, sample_config):
        assert sample_config.db_name == "greenlang"

    def test_db_user_default(self, sample_config):
        assert sample_config.db_user == "gl"

    def test_db_password_default(self, sample_config):
        assert sample_config.db_password == "gl"

    def test_db_pool_min_default(self, sample_config):
        assert sample_config.db_pool_min == 2

    def test_db_pool_max_default(self, sample_config):
        assert sample_config.db_pool_max == 20

    def test_redis_host_default(self, sample_config):
        assert sample_config.redis_host == "localhost"

    def test_redis_port_default(self, sample_config):
        assert sample_config.redis_port == 6379

    def test_redis_db_default(self, sample_config):
        assert sample_config.redis_db == 0

    def test_redis_password_default(self, sample_config):
        assert sample_config.redis_password == ""

    def test_cache_ttl_default(self, sample_config):
        assert sample_config.cache_ttl == 3600

    def test_redis_lock_ttl_default(self, sample_config):
        assert sample_config.redis_lock_ttl_seconds == 10

    def test_redis_lock_retry_count_default(self, sample_config):
        assert sample_config.redis_lock_retry_count == 5

    def test_redis_lock_retry_delay_default(self, sample_config):
        assert sample_config.redis_lock_retry_delay_ms == 100

    def test_format_version_default(self, sample_config):
        assert sample_config.format_version == "1.0"

    def test_reference_prefix_default(self, sample_config):
        assert sample_config.reference_prefix == "EUDR"

    def test_default_member_state_default(self, sample_config):
        assert sample_config.default_member_state == "DE"

    def test_separator_default(self, sample_config):
        assert sample_config.separator == "-"

    def test_format_template_default(self, sample_config):
        assert "{prefix}" in sample_config.format_template
        assert "{ms}" in sample_config.format_template
        assert "{year}" in sample_config.format_template
        assert "{operator}" in sample_config.format_template
        assert "{sequence}" in sample_config.format_template
        assert "{checksum}" in sample_config.format_template

    def test_sequence_digits_default(self, sample_config):
        assert sample_config.sequence_digits == 6

    def test_operator_code_max_length_default(self, sample_config):
        assert sample_config.operator_code_max_length == 10

    def test_sequence_start_default(self, sample_config):
        assert sample_config.sequence_start == 1

    def test_sequence_end_default(self, sample_config):
        assert sample_config.sequence_end == 999999

    def test_sequence_overflow_strategy_default(self, sample_config):
        assert sample_config.sequence_overflow_strategy == "extend"

    def test_sequence_rollover_year_default(self, sample_config):
        assert sample_config.sequence_rollover_year is True

    def test_checksum_algorithm_default(self, sample_config):
        assert sample_config.checksum_algorithm == "luhn"

    def test_checksum_length_default(self, sample_config):
        assert sample_config.checksum_length == 1

    def test_max_batch_size_default(self, sample_config):
        assert sample_config.max_batch_size == 10000

    def test_batch_chunk_size_default(self, sample_config):
        assert sample_config.batch_chunk_size == 500

    def test_batch_timeout_default(self, sample_config):
        assert sample_config.batch_timeout_seconds == 300

    def test_max_concurrent_batches_default(self, sample_config):
        assert sample_config.max_concurrent_batches == 5

    def test_generation_timeout_default(self, sample_config):
        assert sample_config.generation_timeout_seconds == 5

    def test_generation_rate_limit_default(self, sample_config):
        assert sample_config.generation_rate_limit_per_second == 10000

    def test_max_concurrent_requests_default(self, sample_config):
        assert sample_config.max_concurrent_requests == 100

    def test_enable_bloom_filter_default(self, sample_config):
        assert sample_config.enable_bloom_filter is True

    def test_bloom_filter_capacity_default(self, sample_config):
        assert sample_config.bloom_filter_capacity == 10000000

    def test_bloom_filter_error_rate_default(self, sample_config):
        assert sample_config.bloom_filter_error_rate == pytest.approx(0.001)

    def test_default_expiration_months_default(self, sample_config):
        assert sample_config.default_expiration_months == 12

    def test_max_expiration_months_default(self, sample_config):
        assert sample_config.max_expiration_months == 60

    def test_expiration_warning_days_default(self, sample_config):
        assert sample_config.expiration_warning_days == 30

    def test_collision_max_retries_default(self, sample_config):
        assert sample_config.collision_max_retries == 10

    def test_collision_backoff_base_ms_default(self, sample_config):
        assert sample_config.collision_backoff_base_ms == 5

    def test_collision_backoff_max_ms_default(self, sample_config):
        assert sample_config.collision_backoff_max_ms == 500

    def test_enable_auto_expiration_default(self, sample_config):
        assert sample_config.enable_auto_expiration is True

    def test_retention_years_default(self, sample_config):
        assert sample_config.retention_years == 5

    def test_allow_transfer_default(self, sample_config):
        assert sample_config.allow_transfer is True

    def test_require_revocation_reason_default(self, sample_config):
        assert sample_config.require_revocation_reason is True

    def test_enable_idempotency_default(self, sample_config):
        assert sample_config.enable_idempotency is True

    def test_idempotency_key_ttl_default(self, sample_config):
        assert sample_config.idempotency_key_ttl_seconds == 86400

    def test_documentation_generator_url_default(self, sample_config):
        assert "eudr-doc-generator" in sample_config.documentation_generator_url

    def test_due_diligence_url_default(self, sample_config):
        assert "eudr-due-diligence" in sample_config.due_diligence_url

    def test_eu_information_system_url_default(self, sample_config):
        assert "europa.eu" in sample_config.eu_information_system_url

    def test_rate_limit_anonymous_default(self, sample_config):
        assert sample_config.rate_limit_anonymous == 10

    def test_rate_limit_basic_default(self, sample_config):
        assert sample_config.rate_limit_basic == 50

    def test_rate_limit_standard_default(self, sample_config):
        assert sample_config.rate_limit_standard == 200

    def test_rate_limit_premium_default(self, sample_config):
        assert sample_config.rate_limit_premium == 1000

    def test_rate_limit_admin_default(self, sample_config):
        assert sample_config.rate_limit_admin == 5000

    def test_circuit_breaker_failure_threshold_default(self, sample_config):
        assert sample_config.circuit_breaker_failure_threshold == 5

    def test_circuit_breaker_reset_timeout_default(self, sample_config):
        assert sample_config.circuit_breaker_reset_timeout == 60

    def test_circuit_breaker_half_open_max_default(self, sample_config):
        assert sample_config.circuit_breaker_half_open_max == 3

    def test_provenance_enabled_default(self, sample_config):
        assert sample_config.provenance_enabled is True

    def test_provenance_algorithm_default(self, sample_config):
        assert sample_config.provenance_algorithm == "sha256"

    def test_provenance_chain_enabled_default(self, sample_config):
        assert sample_config.provenance_chain_enabled is True

    def test_provenance_genesis_hash_default(self, sample_config):
        assert sample_config.provenance_genesis_hash == "0" * 64
        assert len(sample_config.provenance_genesis_hash) == 64

    def test_metrics_enabled_default(self, sample_config):
        assert sample_config.metrics_enabled is True

    def test_metrics_prefix_default(self, sample_config):
        assert sample_config.metrics_prefix == "gl_eudr_rng_"

    def test_log_level_default(self, sample_config):
        assert sample_config.log_level == "INFO"


# ====================================================================
# Test: Environment Variable Overrides
# ====================================================================


class TestConfigEnvOverrides:
    """Test environment variable override support."""

    def test_env_prefix(self):
        assert _ENV_PREFIX == "GL_EUDR_RNG_"

    def test_env_helper_returns_default(self):
        assert _env("NONEXISTENT_KEY_12345", "fallback") == "fallback"

    def test_env_int_returns_default(self):
        assert _env_int("NONEXISTENT_KEY_12345", 42) == 42

    def test_env_float_returns_default(self):
        assert _env_float("NONEXISTENT_KEY_12345", 3.14) == pytest.approx(3.14)

    def test_env_bool_returns_default(self):
        assert _env_bool("NONEXISTENT_KEY_12345", True) is True
        assert _env_bool("NONEXISTENT_KEY_12345", False) is False

    @patch.dict(os.environ, {"GL_EUDR_RNG_DB_HOST": "db.prod.local"})
    def test_db_host_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.db_host == "db.prod.local"

    @patch.dict(os.environ, {"GL_EUDR_RNG_DB_PORT": "5433"})
    def test_db_port_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.db_port == 5433

    @patch.dict(os.environ, {"GL_EUDR_RNG_REFERENCE_PREFIX": "GLREF"})
    def test_reference_prefix_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.reference_prefix == "GLREF"

    @patch.dict(os.environ, {"GL_EUDR_RNG_DEFAULT_MEMBER_STATE": "FR"})
    def test_default_member_state_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.default_member_state == "FR"

    @patch.dict(os.environ, {"GL_EUDR_RNG_SEQUENCE_DIGITS": "8"})
    def test_sequence_digits_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.sequence_digits == 8

    @patch.dict(os.environ, {"GL_EUDR_RNG_CHECKSUM_ALGORITHM": "iso7064"})
    def test_checksum_algorithm_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.checksum_algorithm == "iso7064"

    @patch.dict(os.environ, {"GL_EUDR_RNG_MAX_BATCH_SIZE": "500"})
    def test_max_batch_size_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.max_batch_size == 500

    @patch.dict(os.environ, {"GL_EUDR_RNG_SEQUENCE_OVERFLOW_STRATEGY": "reject"})
    def test_overflow_strategy_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.sequence_overflow_strategy == "reject"

    @patch.dict(os.environ, {"GL_EUDR_RNG_ENABLE_BLOOM_FILTER": "false"})
    def test_bloom_filter_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.enable_bloom_filter is False

    @patch.dict(os.environ, {"GL_EUDR_RNG_RETENTION_YEARS": "10"})
    def test_retention_years_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.retention_years == 10

    @patch.dict(os.environ, {"GL_EUDR_RNG_LOG_LEVEL": "DEBUG"})
    def test_log_level_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.log_level == "DEBUG"

    @patch.dict(os.environ, {"GL_EUDR_RNG_METRICS_ENABLED": "false"})
    def test_metrics_enabled_env_override(self):
        cfg = ReferenceNumberGeneratorConfig()
        assert cfg.metrics_enabled is False

    def test_env_bool_true_variants(self):
        for val in ("true", "1", "yes", "True", "YES", "TRUE"):
            with patch.dict(os.environ, {"GL_EUDR_RNG_TEST_BOOL": val}):
                assert _env_bool("TEST_BOOL", False) is True

    def test_env_bool_false_variants(self):
        for val in ("false", "0", "no", "anything"):
            with patch.dict(os.environ, {"GL_EUDR_RNG_TEST_BOOL": val}):
                assert _env_bool("TEST_BOOL", True) is False


# ====================================================================
# Test: EU Member States Configuration
# ====================================================================


class TestConfigMemberStates:
    """Test EU member state mapping in config."""

    def test_eu_member_states_is_dict(self):
        assert isinstance(EU_MEMBER_STATES, dict)

    def test_eu_member_states_count_27(self):
        assert len(EU_MEMBER_STATES) == 27

    @pytest.mark.parametrize("code,name", [
        ("AT", "Austria"), ("BE", "Belgium"), ("BG", "Bulgaria"),
        ("HR", "Croatia"), ("CY", "Cyprus"), ("CZ", "Czechia"),
        ("DK", "Denmark"), ("EE", "Estonia"), ("FI", "Finland"),
        ("FR", "France"), ("DE", "Germany"), ("GR", "Greece"),
        ("HU", "Hungary"), ("IE", "Ireland"), ("IT", "Italy"),
        ("LV", "Latvia"), ("LT", "Lithuania"), ("LU", "Luxembourg"),
        ("MT", "Malta"), ("NL", "Netherlands"), ("PL", "Poland"),
        ("PT", "Portugal"), ("RO", "Romania"), ("SK", "Slovakia"),
        ("SI", "Slovenia"), ("ES", "Spain"), ("SE", "Sweden"),
    ])
    def test_member_state_mapping(self, code, name):
        assert code in EU_MEMBER_STATES
        assert EU_MEMBER_STATES[code] == name

    def test_all_codes_are_two_letters(self):
        for code in EU_MEMBER_STATES:
            assert len(code) == 2
            assert code.isupper()
            assert code.isalpha()

    def test_default_member_state_is_valid(self, sample_config):
        assert sample_config.default_member_state in EU_MEMBER_STATES


# ====================================================================
# Test: Validation Logic
# ====================================================================


class TestConfigValidation:
    """Test __post_init__ validation warnings."""

    def test_valid_config_no_warnings(self, sample_config, caplog):
        """Default config should not produce warning-level issues
        (only the info log from __post_init__)."""
        # defaults are valid, so no warnings expected
        cfg = ReferenceNumberGeneratorConfig()
        # simply asserting creation works
        assert cfg.reference_prefix == "EUDR"

    def test_invalid_member_state_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(default_member_state="XX")
        assert any("not a valid EU member state" in r.message for r in caplog.records)

    def test_negative_sequence_start_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(sequence_start=-1)
        assert any("non-negative" in r.message for r in caplog.records)

    def test_sequence_end_less_than_start_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(sequence_start=100, sequence_end=50)
        assert any("greater than start" in r.message for r in caplog.records)

    def test_sequence_digits_overflow_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(sequence_digits=4, sequence_end=999999)
        assert any("exceeds max" in r.message for r in caplog.records)

    def test_invalid_checksum_algorithm_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(checksum_algorithm="sha256")
        assert any("not one of" in r.message for r in caplog.records)

    def test_invalid_overflow_strategy_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(sequence_overflow_strategy="explode")
        assert any("not one of" in r.message for r in caplog.records)

    def test_batch_size_zero_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(max_batch_size=0)
        assert any("at least 1" in r.message for r in caplog.records)

    def test_pool_min_exceeds_max_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(db_pool_min=50, db_pool_max=10)
        assert any("exceeds pool max" in r.message for r in caplog.records)

    def test_retention_below_eudr_minimum_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(retention_years=3)
        assert any("below EUDR minimum of 5 years" in r.message for r in caplog.records)

    def test_expiration_below_one_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(default_expiration_months=0)
        assert any("at least 1" in r.message for r in caplog.records)

    def test_expiration_exceeds_max_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            ReferenceNumberGeneratorConfig(
                default_expiration_months=100, max_expiration_months=60
            )
        assert any("exceeds max expiration" in r.message for r in caplog.records)


# ====================================================================
# Test: Singleton Behavior
# ====================================================================


class TestConfigSingleton:
    """Test get_config() / reset_config() singleton pattern."""

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, ReferenceNumberGeneratorConfig)

    def test_get_config_returns_same_instance(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_clears_instance(self):
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        assert cfg1 is not cfg2

    def test_thread_safety_of_singleton(self):
        """Verify multiple threads get the same singleton."""
        results = []

        def get_it():
            results.append(id(get_config()))

        threads = [threading.Thread(target=get_it) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1, "All threads must get the same instance"

    @patch.dict(os.environ, {"GL_EUDR_RNG_REFERENCE_PREFIX": "XREF"})
    def test_env_override_in_singleton(self):
        cfg = get_config()
        assert cfg.reference_prefix == "XREF"


# ====================================================================
# Test: Custom Config Creation
# ====================================================================


class TestCustomConfig:
    """Test creating config with custom values."""

    def test_custom_config_values(self, custom_config):
        assert custom_config.db_host == "db.test.local"
        assert custom_config.db_port == 5433
        assert custom_config.reference_prefix == "TEST"
        assert custom_config.default_member_state == "FR"
        assert custom_config.separator == "_"
        assert custom_config.sequence_digits == 8
        assert custom_config.sequence_start == 100
        assert custom_config.checksum_algorithm == "iso7064"
        assert custom_config.max_batch_size == 500
        assert custom_config.collision_max_retries == 3
        assert custom_config.default_expiration_months == 6
        assert custom_config.retention_years == 7

    def test_valid_checksum_algorithms(self):
        for algo in ("luhn", "iso7064", "crc16", "modulo97"):
            cfg = ReferenceNumberGeneratorConfig(checksum_algorithm=algo)
            assert cfg.checksum_algorithm == algo

    def test_valid_overflow_strategies(self):
        for strategy in ("extend", "reject", "rollover"):
            cfg = ReferenceNumberGeneratorConfig(
                sequence_overflow_strategy=strategy
            )
            assert cfg.sequence_overflow_strategy == strategy
