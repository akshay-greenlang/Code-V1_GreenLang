# -*- coding: utf-8 -*-
"""
Unit tests for configuration management functionality.

Tests cover:
- Configuration loading from files
- Environment variable handling
- Default values
- Validation
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from typing import Dict, Any

from greenlang.agents.base import AgentConfig
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.boiler_agent import BoilerAgent
from greenlang.utils.net import NetworkPolicy


class TestAgentConfig:
    """Test AgentConfig model and validation."""

    def test_agent_config_creation(self):
        """Test basic agent config creation."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent for testing",
            version="1.0.0"
        )

        assert config.name == "test_agent"
        assert config.description == "Test agent for testing"
        assert config.version == "1.0.0"
        assert config.enabled is True
        assert config.parameters == {}

    def test_agent_config_defaults(self):
        """Test agent config default values."""
        config = AgentConfig(
            name="test_agent",
            description="Test description"
        )

        assert config.version == "0.0.1"
        assert config.enabled is True
        assert config.parameters == {}

    def test_agent_config_with_parameters(self):
        """Test agent config with custom parameters."""
        params = {
            "threshold": 0.5,
            "mode": "production",
            "features": ["feature1", "feature2"]
        }

        config = AgentConfig(
            name="test_agent",
            description="Test description",
            parameters=params
        )

        assert config.parameters == params
        assert config.parameters["threshold"] == 0.5
        assert config.parameters["mode"] == "production"
        assert len(config.parameters["features"]) == 2


class TestFuelConfigLoading:
    """Test fuel agent configuration loading."""

    def test_load_fuel_config_from_file(self):
        """Test loading fuel config from existing file."""
        # Create a temporary config file
        config_data = {
            "fuel_properties": {
                "electricity": {
                    "energy_content": {"value": 3412, "unit": "Btu/kWh"}
                },
                "natural_gas": {
                    "energy_content": {"value": 100000, "unit": "Btu/therm"}
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Mock the config path to point to our temp file
            with patch.object(Path, 'exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
                    config = FuelAgent.load_fuel_config()

                    assert "fuel_properties" in config
                    assert "electricity" in config["fuel_properties"]
                    assert config["fuel_properties"]["electricity"]["energy_content"]["value"] == 3412
        finally:
            os.unlink(temp_path)

    def test_load_fuel_config_fallback(self):
        """Test fuel config fallback when file doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            config = FuelAgent.load_fuel_config()

            assert "fuel_properties" in config
            assert "electricity" in config["fuel_properties"]
            assert "natural_gas" in config["fuel_properties"]
            # Note: diesel might not be in fallback config, check actual implementation

    def test_fuel_config_caching(self):
        """Test that fuel config is cached properly."""
        # Clear the cache first
        FuelAgent.load_fuel_config.cache_clear()

        with patch.object(Path, 'exists', return_value=False):
            # First call
            config1 = FuelAgent.load_fuel_config()
            # Second call (should use cache)
            config2 = FuelAgent.load_fuel_config()

            # Should be the same object due to caching
            assert config1 is config2

            # Check cache info
            cache_info = FuelAgent.load_fuel_config.cache_info()
            assert cache_info.hits == 1
            assert cache_info.misses == 1


class TestBoilerConfigLoading:
    """Test boiler agent configuration loading."""

    def test_load_boiler_efficiency_config(self):
        """Test loading boiler efficiency configuration."""
        config_data = {
            "boiler_efficiencies": {
                "natural_gas": {
                    "condensing": 0.95,
                    "non_condensing": 0.85
                },
                "oil": {
                    "conventional": 0.80,
                    "high_efficiency": 0.87
                }
            }
        }

        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
                config = BoilerAgent.load_efficiency_config()

                assert "boiler_efficiencies" in config
                assert "natural_gas" in config["boiler_efficiencies"]
                assert config["boiler_efficiencies"]["natural_gas"]["condensing"] == 0.95


class TestNetworkPolicyConfig:
    """Test network policy configuration loading."""

    def _create_network_policy(self, **kwargs):
        """Helper to create NetworkPolicy with mocked home directory."""
        with patch.object(Path, 'home', return_value=Path("/fake/home")):
            with patch.object(Path, 'exists', return_value=False):
                return NetworkPolicy()

    def test_network_policy_empty_defaults(self):
        """Test that default policy has empty allowed domains (secure by default)."""
        # Create policy without any configuration
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'home', return_value=Path("/fake/home")):
                    policy = NetworkPolicy()

                    # Should have empty allowed domains by default (deny all)
                    assert len(policy.allowed_domains) == 0
                    assert len(policy.blocked_domains) == 0

    def test_network_policy_env_domains(self):
        """Test loading domains from environment variables."""
        test_domains = "example.com,test.org"

        with patch.dict(os.environ, {'GL_ALLOWED_DOMAINS': test_domains}):
            policy = self._create_network_policy()

            assert "example.com" in policy.allowed_domains
            assert "test.org" in policy.allowed_domains

    def test_network_policy_blocked_domains_env(self):
        """Test loading blocked domains from environment."""
        blocked_domains = "malicious.com,spam.org"

        with patch.dict(os.environ, {'GL_BLOCKED_DOMAINS': blocked_domains}):
            policy = self._create_network_policy()

            assert "malicious.com" in policy.blocked_domains
            assert "spam.org" in policy.blocked_domains

    def test_network_policy_config_file_loading(self):
        """Test loading allowed domains from config file."""
        config_content = "example.com\ntest.org\n# This is a comment\nvalid.domain"

        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=config_content)):
                policy = NetworkPolicy()

                assert "example.com" in policy.allowed_domains
                assert "test.org" in policy.allowed_domains
                assert "valid.domain" in policy.allowed_domains
                # Comments should be ignored
                assert "# This is a comment" not in policy.allowed_domains

    def test_network_policy_blocklist_file_loading(self):
        """Test loading blocked domains from config file."""
        config_content = "blocked1.com\nblocked2.org\n# Comment\nblocked3.net"

        with patch.object(Path, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=config_content)):
                # Mock the blocklist file specifically
                with patch.object(NetworkPolicy, '_load_blocked_domains') as mock_load:
                    mock_load.return_value = ["blocked1.com", "blocked2.org", "blocked3.net"]
                    policy = NetworkPolicy()

                    assert "blocked1.com" in policy.blocked_domains
                    assert "blocked2.org" in policy.blocked_domains
                    assert "blocked3.net" in policy.blocked_domains


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_agent_config_validation_required_fields(self):
        """Test that required fields are validated."""
        import pydantic_core

        with pytest.raises(pydantic_core._pydantic_core.ValidationError):  # Pydantic validation error
            AgentConfig()  # Missing required name and description

    def test_agent_config_validation_types(self):
        """Test that field types are validated."""
        import pydantic_core

        with pytest.raises(pydantic_core._pydantic_core.ValidationError):  # Pydantic validation error
            AgentConfig(
                name=123,  # Should be string
                description="Valid description"
            )

        # Test that invalid types for version are handled correctly
        config = AgentConfig(
            name="Valid name",
            description="Valid description",
            enabled="yes"  # Pydantic will coerce this to True
        )
        assert config.enabled is True  # Pydantic coerces "yes" to True

    def test_agent_config_validation_version_format(self):
        """Test version format validation."""
        # Valid versions
        valid_configs = [
            AgentConfig(name="test", description="test", version="1.0.0"),
            AgentConfig(name="test", description="test", version="0.1.0"),
            AgentConfig(name="test", description="test", version="2.5.1-alpha"),
        ]

        for config in valid_configs:
            assert config.version is not None

    def test_network_policy_url_validation(self):
        """Test URL validation in network policy."""
        policy = NetworkPolicy()

        # Valid URLs from allowed domains
        valid_urls = [
            "https://greenlang.io/api/test",
            "http://hub.greenlang.io/packs",
            "https://github.com/user/repo"
        ]

        for url in valid_urls:
            assert policy.check_url(url, "test") is True

        # Invalid URLs (not in allowlist)
        invalid_urls = [
            "https://malicious.example.com/bad",
            "http://unknown.domain.com/test"
        ]

        for url in invalid_urls:
            assert policy.check_url(url, "test") is False


class TestEnvironmentVariableConfig:
    """Test configuration loading from environment variables."""

    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        test_env = {
            'GL_ALLOWED_DOMAINS': 'env.example.com,env.test.org',
            'GL_BLOCKED_DOMAINS': 'env.blocked.com'
        }

        with patch.dict(os.environ, test_env):
            policy = NetworkPolicy()

            assert "env.example.com" in policy.allowed_domains
            assert "env.test.org" in policy.allowed_domains
            assert "env.blocked.com" in policy.blocked_domains

    def test_empty_environment_variables(self):
        """Test handling of empty environment variables."""
        with patch.dict(os.environ, {'GL_ALLOWED_DOMAINS': ''}):
            with patch.object(Path, 'exists', return_value=False):
                policy = NetworkPolicy()

                # Should have empty domains when no config provided
                assert len(policy.allowed_domains) == 0

    def test_environment_variable_configuration(self):
        """Test that environment variables configure allowed domains."""
        with patch.dict(os.environ, {'GL_ALLOWED_DOMAINS': 'custom.domain.com,another.domain.org'}):
            with patch.object(Path, 'exists', return_value=False):
                policy = NetworkPolicy()

                # Should have only the configured domains
                assert "custom.domain.com" in policy.allowed_domains
                assert "another.domain.org" in policy.allowed_domains
                assert len(policy.allowed_domains) == 2


class TestConfigDefaults:
    """Test default configuration values."""

    def test_agent_config_defaults(self):
        """Test that agent configuration has proper defaults."""
        config = AgentConfig(
            name="test_agent",
            description="Test description"
        )

        assert config.version == "0.0.1"
        assert config.enabled is True
        assert isinstance(config.parameters, dict)
        assert len(config.parameters) == 0

    def test_network_policy_secure_defaults(self):
        """Test that network policy has secure defaults (empty allowlist)."""
        # Create policy without any environment variables
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, 'exists', return_value=False):
                with patch.object(Path, 'home', return_value=Path("/fake/home")):
                    policy = NetworkPolicy()

                    # Should have empty allowed domains by default (secure)
                    assert len(policy.allowed_domains) == 0
                    # Should have empty blocked list by default
                    assert len(policy.blocked_domains) == 0