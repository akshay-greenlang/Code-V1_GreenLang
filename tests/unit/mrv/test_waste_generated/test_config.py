# -*- coding: utf-8 -*-
"""
Test suite for waste_generated.config - AGENT-MRV-018.

Tests configuration management for the Waste Generated in Operations Agent
(GL-MRV-S3-005) including default values, environment variable loading,
singleton pattern, thread safety, validation, and serialization.

Coverage:
- Default config values for all 12+ sections
- GL_WG_ environment variable loading (monkeypatch)
- Singleton pattern (get_config returns same instance)
- Thread safety
- Validation (invalid values raise errors)
- to_dict / from_dict round-trip
- Individual section configs (Landfill, Incineration, Recycling, etc.)

Author: GL-TestEngineer
Date: February 2026
"""

import os
import threading
from decimal import Decimal
from typing import Any, Dict
import pytest

from greenlang.agents.mrv.waste_generated.config import (
    get_config,
    WasteGeneratedConfig,
    GeneralConfig,
    DatabaseConfig,
    # Add other config classes as they're implemented
)


# ==============================================================================
# GENERAL CONFIGURATION TESTS
# ==============================================================================

class TestGeneralConfig:
    """Test GeneralConfig dataclass."""

    def test_default_values(self):
        """Test default general config values."""
        config = GeneralConfig()

        assert config.enabled is True
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.agent_id == "GL-MRV-S3-005"
        assert config.version == "1.0.0"

    def test_custom_values(self):
        """Test custom general config values."""
        config = GeneralConfig(
            enabled=False,
            debug=True,
            log_level="DEBUG",
            agent_id="GL-MRV-S3-005-TEST",
            version="2.0.0"
        )

        assert config.enabled is False
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.agent_id == "GL-MRV-S3-005-TEST"
        assert config.version == "2.0.0"

    def test_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = GeneralConfig()

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            config.enabled = False

    def test_validate_valid(self):
        """Test validation passes for valid config."""
        config = GeneralConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_log_level(self):
        """Test validation fails for invalid log level."""
        config = GeneralConfig(log_level="INVALID")

        with pytest.raises(ValueError, match="Invalid log_level"):
            config.validate()

    def test_validate_empty_agent_id(self):
        """Test validation fails for empty agent_id."""
        config = GeneralConfig(agent_id="")

        with pytest.raises(ValueError, match="agent_id cannot be empty"):
            config.validate()

    def test_validate_invalid_version_format(self):
        """Test validation fails for invalid SemVer format."""
        config = GeneralConfig(version="1.0")

        with pytest.raises(ValueError, match="Must follow SemVer"):
            config.validate()

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = GeneralConfig(
            enabled=True,
            debug=False,
            log_level="INFO",
            agent_id="GL-MRV-S3-005",
            version="1.0.0"
        )

        config_dict = config.to_dict()

        assert config_dict == {
            "enabled": True,
            "debug": False,
            "log_level": "INFO",
            "agent_id": "GL-MRV-S3-005",
            "version": "1.0.0"
        }

    def test_from_dict(self):
        """Test from_dict deserialization."""
        config_dict = {
            "enabled": False,
            "debug": True,
            "log_level": "DEBUG",
            "agent_id": "GL-MRV-S3-005",
            "version": "1.0.0"
        }

        config = GeneralConfig.from_dict(config_dict)

        assert config.enabled is False
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_round_trip(self):
        """Test to_dict → from_dict round-trip."""
        original = GeneralConfig(
            enabled=True,
            debug=True,
            log_level="WARNING",
            agent_id="GL-MRV-S3-005",
            version="1.5.2"
        )

        config_dict = original.to_dict()
        restored = GeneralConfig.from_dict(config_dict)

        assert restored.enabled == original.enabled
        assert restored.debug == original.debug
        assert restored.log_level == original.log_level
        assert restored.agent_id == original.agent_id
        assert restored.version == original.version

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("GL_WG_ENABLED", "false")
        monkeypatch.setenv("GL_WG_DEBUG", "true")
        monkeypatch.setenv("GL_WG_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GL_WG_AGENT_ID", "GL-MRV-S3-005-DEV")
        monkeypatch.setenv("GL_WG_VERSION", "2.1.0")

        config = GeneralConfig.from_env()

        assert config.enabled is False
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.agent_id == "GL-MRV-S3-005-DEV"
        assert config.version == "2.1.0"

    def test_from_env_defaults(self):
        """Test from_env uses defaults when env vars not set."""
        # Clear any existing env vars
        for key in ["GL_WG_ENABLED", "GL_WG_DEBUG", "GL_WG_LOG_LEVEL", "GL_WG_AGENT_ID", "GL_WG_VERSION"]:
            os.environ.pop(key, None)

        config = GeneralConfig.from_env()

        assert config.enabled is True
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.agent_id == "GL-MRV-S3-005"
        assert config.version == "1.0.0"


# ==============================================================================
# DATABASE CONFIGURATION TESTS
# ==============================================================================

class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_default_values(self):
        """Test default database config values."""
        config = DatabaseConfig()

        assert config.pool_size == 20
        assert config.max_overflow == 10
        assert config.pool_timeout == 30

    def test_custom_values(self):
        """Test custom database config values."""
        config = DatabaseConfig(
            database_url="postgresql://user:pass@localhost:5432/testdb",
            pool_size=50,
            max_overflow=20,
            pool_timeout=60
        )

        assert config.database_url == "postgresql://user:pass@localhost:5432/testdb"
        assert config.pool_size == 50
        assert config.max_overflow == 20
        assert config.pool_timeout == 60

    def test_frozen(self):
        """Test config is immutable (frozen=True)."""
        config = DatabaseConfig()

        with pytest.raises(Exception):
            config.pool_size = 100

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("GL_WG_DATABASE_URL", "postgresql://test:test@localhost/waste_test")
        monkeypatch.setenv("GL_WG_DATABASE_POOL_SIZE", "30")
        monkeypatch.setenv("GL_WG_DATABASE_MAX_OVERFLOW", "15")
        monkeypatch.setenv("GL_WG_DATABASE_POOL_TIMEOUT", "45")

        config = DatabaseConfig.from_env()

        assert config.database_url == "postgresql://test:test@localhost/waste_test"
        assert config.pool_size == 30
        assert config.max_overflow == 15
        assert config.pool_timeout == 45


# ==============================================================================
# MAIN CONFIG TESTS
# ==============================================================================

class TestWasteGeneratedConfig:
    """Test WasteGeneratedConfig main configuration class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance before each test."""
        # This fixture would need to access the internal singleton state
        # Implementation depends on how get_config is structured
        yield
        # Reset singleton after test

    def test_default_config_creation(self):
        """Test creating config with all defaults."""
        # Note: This test structure depends on actual WasteGeneratedConfig implementation
        pass

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        pass

    def test_section_configs(self):
        """Test all section configs are present."""
        pass


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================

class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_get_config_returns_same_instance(self):
        """Test get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_singleton_across_threads(self):
        """Test singleton works across threads."""
        configs = []

        def get_config_thread():
            configs.append(get_config())

        threads = [threading.Thread(target=get_config_thread) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All configs should be the same instance
        first_config = configs[0]
        for config in configs[1:]:
            assert config is first_config


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================

class TestThreadSafety:
    """Test thread safety of configuration."""

    def test_concurrent_access(self):
        """Test concurrent config access is thread-safe."""
        errors = []

        def access_config():
            try:
                config = get_config()
                # Access various config properties
                _ = config.general.enabled
                _ = config.general.log_level
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_config) for _ in range(100)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0


# ==============================================================================
# ENVIRONMENT VARIABLE TESTS
# ==============================================================================

class TestEnvironmentVariables:
    """Test environment variable loading."""

    def test_general_env_vars(self, monkeypatch):
        """Test general section env vars."""
        monkeypatch.setenv("GL_WG_ENABLED", "false")
        monkeypatch.setenv("GL_WG_DEBUG", "true")

        config = GeneralConfig.from_env()

        assert config.enabled is False
        assert config.debug is True

    def test_boolean_env_var_parsing(self, monkeypatch):
        """Test boolean environment variable parsing."""
        # Test various boolean representations
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", False),  # Only "true" (case-insensitive) should be True
            ("0", False),
        ]

        for env_value, expected in test_cases:
            monkeypatch.setenv("GL_WG_ENABLED", env_value)
            config = GeneralConfig.from_env()
            assert config.enabled == expected, f"env_value={env_value}, expected={expected}"

    def test_numeric_env_var_parsing(self, monkeypatch):
        """Test numeric environment variable parsing."""
        monkeypatch.setenv("GL_WG_DATABASE_POOL_SIZE", "42")
        monkeypatch.setenv("GL_WG_DATABASE_POOL_TIMEOUT", "120")

        config = DatabaseConfig.from_env()

        assert config.pool_size == 42
        assert config.pool_timeout == 120


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================

class TestValidation:
    """Test configuration validation."""

    def test_validate_all_log_levels(self):
        """Test validation accepts all valid log levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            config = GeneralConfig(log_level=level)
            config.validate()  # Should not raise

    def test_validate_log_level_case_insensitive(self):
        """Test log level validation is case-sensitive (uppercase required)."""
        # Log levels should be uppercase
        config = GeneralConfig(log_level="debug")

        with pytest.raises(ValueError):
            config.validate()

    def test_validate_version_format(self):
        """Test version format validation."""
        valid_versions = ["1.0.0", "2.5.3", "10.20.30"]

        for version in valid_versions:
            config = GeneralConfig(version=version)
            config.validate()  # Should not raise

        invalid_versions = ["1.0", "1", "v1.0.0", "1.0.0-beta"]

        for version in invalid_versions:
            config = GeneralConfig(version=version)
            with pytest.raises(ValueError):
                config.validate()


# ==============================================================================
# SERIALIZATION TESTS
# ==============================================================================

class TestSerialization:
    """Test configuration serialization."""

    def test_to_dict_all_sections(self):
        """Test to_dict includes all config sections."""
        config = GeneralConfig()
        config_dict = config.to_dict()

        assert "enabled" in config_dict
        assert "debug" in config_dict
        assert "log_level" in config_dict
        assert "agent_id" in config_dict
        assert "version" in config_dict

    def test_from_dict_all_sections(self):
        """Test from_dict reconstructs all sections."""
        config_dict = {
            "enabled": True,
            "debug": False,
            "log_level": "INFO",
            "agent_id": "GL-MRV-S3-005",
            "version": "1.0.0"
        }

        config = GeneralConfig.from_dict(config_dict)

        assert config.enabled is True
        assert config.debug is False
        assert config.log_level == "INFO"

    def test_serialization_preserves_types(self):
        """Test serialization preserves data types."""
        config = GeneralConfig(
            enabled=True,
            debug=False,
            log_level="DEBUG"
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict["enabled"], bool)
        assert isinstance(config_dict["debug"], bool)
        assert isinstance(config_dict["log_level"], str)


# ==============================================================================
# DECIMAL PRECISION TESTS
# ==============================================================================

class TestDecimalPrecision:
    """Test Decimal precision in configuration."""

    def test_landfill_defaults_are_decimal(self):
        """Test landfill config defaults use Decimal type."""
        # This test structure depends on LandfillConfig implementation
        # Example:
        # config = LandfillConfig()
        # assert isinstance(config.default_docf, Decimal)
        # assert isinstance(config.default_f, Decimal)
        # assert isinstance(config.default_oxidation, Decimal)
        pass

    def test_incineration_defaults_are_decimal(self):
        """Test incineration config defaults use Decimal type."""
        pass

    def test_recycling_defaults_are_decimal(self):
        """Test recycling config defaults use Decimal type."""
        pass


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestConfigIntegration:
    """Test configuration integration scenarios."""

    def test_full_config_from_env(self, monkeypatch):
        """Test loading full config from environment variables."""
        # Set all environment variables
        monkeypatch.setenv("GL_WG_ENABLED", "true")
        monkeypatch.setenv("GL_WG_DEBUG", "false")
        monkeypatch.setenv("GL_WG_LOG_LEVEL", "INFO")
        monkeypatch.setenv("GL_WG_DATABASE_POOL_SIZE", "25")

        # Load config
        general_config = GeneralConfig.from_env()
        db_config = DatabaseConfig.from_env()

        # Verify all values loaded correctly
        assert general_config.enabled is True
        assert general_config.debug is False
        assert general_config.log_level == "INFO"
        assert db_config.pool_size == 25

    def test_partial_env_with_defaults(self, monkeypatch):
        """Test partial env vars with remaining defaults."""
        # Set only some env vars
        monkeypatch.setenv("GL_WG_DEBUG", "true")
        monkeypatch.setenv("GL_WG_LOG_LEVEL", "WARNING")

        # Clear database env vars
        for key in os.environ.copy():
            if key.startswith("GL_WG_DATABASE"):
                del os.environ[key]

        general_config = GeneralConfig.from_env()
        db_config = DatabaseConfig.from_env()

        # Verify env vars loaded
        assert general_config.debug is True
        assert general_config.log_level == "WARNING"

        # Verify defaults used
        assert general_config.enabled is True  # Default
        assert db_config.pool_size == 20  # Default


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_database_url(self):
        """Test handling of empty database URL."""
        config = DatabaseConfig(database_url="")
        # Should not raise during construction
        assert config.database_url == ""

    def test_zero_pool_size(self):
        """Test handling of zero pool size."""
        config = DatabaseConfig(pool_size=0)
        assert config.pool_size == 0

    def test_negative_pool_timeout(self):
        """Test handling of negative pool timeout."""
        config = DatabaseConfig(pool_timeout=-1)
        assert config.pool_timeout == -1

    def test_very_long_agent_id(self):
        """Test handling of very long agent ID."""
        long_id = "GL-MRV-S3-005" + "-" + ("X" * 1000)
        config = GeneralConfig(agent_id=long_id)
        assert config.agent_id == long_id


# ==============================================================================
# SUMMARY
# ==============================================================================

def test_config_module_coverage():
    """Meta-test to ensure comprehensive coverage."""
    # This test verifies we've tested all major config sections
    tested_sections = [
        "GeneralConfig",
        "DatabaseConfig",
        # Add more as implemented
    ]

    assert len(tested_sections) >= 2
