"""
Oracle Connector Configuration Tests
GL-VCCI Scope 3 Platform

Tests for configuration management, environment variables,
validation, and endpoint configuration.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Test Count: 12
"""

import pytest
import os
from pydantic import ValidationError

from connectors.oracle.config import (
    OracleConnectorConfig,
    OAuth2Config,
    RESTEndpoint,
    OracleModule,
    OracleEnvironment,
    RateLimitConfig,
    RetryConfig,
    TimeoutConfig,
    get_config,
    reset_config
)


class TestOAuth2Config:
    """Tests for OAuth2Config."""

    def test_oauth_config_creation(self):
        """Test creating OAuth config with valid parameters."""
        config = OAuth2Config(
            client_id="test_client",
            client_secret="test_secret",
            token_url="https://test.oracle.com/oauth/token"
        )

        assert config.client_id == "test_client"
        assert config.client_secret == "test_secret"
        assert config.token_url == "https://test.oracle.com/oauth/token"
        assert config.grant_type == "client_credentials"
        assert config.scope == "urn:opc:resource:consumer::all"

    def test_oauth_config_custom_ttl(self):
        """Test OAuth config with custom token TTL."""
        config = OAuth2Config(
            client_id="test",
            client_secret="secret",
            token_url="https://test.com/oauth",
            token_cache_ttl=1800  # 30 minutes
        )

        assert config.token_cache_ttl == 1800


class TestRESTEndpoint:
    """Tests for RESTEndpoint configuration."""

    def test_endpoint_creation(self):
        """Test creating REST endpoint configuration."""
        endpoint = RESTEndpoint(
            name="purchase_orders",
            resource_path="/fscmRestApi/resources/11.13.18.05/purchaseOrders",
            module=OracleModule.PROCUREMENT,
            batch_size=500
        )

        assert endpoint.name == "purchase_orders"
        assert endpoint.module == OracleModule.PROCUREMENT
        assert endpoint.batch_size == 500
        assert endpoint.enabled is True

    def test_endpoint_default_values(self):
        """Test endpoint default values."""
        endpoint = RESTEndpoint(
            name="test",
            resource_path="/test",
            module=OracleModule.SCM
        )

        assert endpoint.enabled is True
        assert endpoint.batch_size == 1000
        assert endpoint.api_version == "11.13.18.05"

    def test_endpoint_validation_batch_size(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError):
            RESTEndpoint(
                name="test",
                resource_path="/test",
                module=OracleModule.PROCUREMENT,
                batch_size=0  # Invalid: must be >= 1
            )

        with pytest.raises(ValidationError):
            RESTEndpoint(
                name="test",
                resource_path="/test",
                module=OracleModule.PROCUREMENT,
                batch_size=20000  # Invalid: must be <= 10000
            )


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_retry_config_defaults(self):
        """Test retry config default values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 8.0
        assert config.backoff_multiplier == 2.0
        assert 429 in config.retry_on_status_codes
        assert 503 in config.retry_on_status_codes

    def test_retry_config_validation(self):
        """Test retry config validation."""
        # max_delay must be greater than base_delay
        with pytest.raises(ValidationError):
            RetryConfig(
                base_delay=10.0,
                max_delay=5.0  # Invalid: less than base_delay
            )


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_rate_limit_config_defaults(self):
        """Test rate limit config defaults."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 10
        assert config.enabled is True
        assert config.burst_size == 5


class TestOracleConnectorConfig:
    """Tests for main OracleConnectorConfig."""

    def test_config_from_env(self, mock_env_vars):
        """Test loading configuration from environment variables."""
        config = OracleConnectorConfig.from_env()

        assert config.environment == OracleEnvironment.SANDBOX
        assert config.base_url == "https://test.oraclecloud.com"
        assert config.oauth.client_id == "test_client_id"
        assert config.default_batch_size == 1000

    def test_config_missing_required_env_vars(self, monkeypatch):
        """Test error when required env vars are missing."""
        # Missing ORACLE_BASE_URL
        monkeypatch.setenv("ORACLE_CLIENT_ID", "test")
        monkeypatch.setenv("ORACLE_CLIENT_SECRET", "secret")
        monkeypatch.setenv("ORACLE_TOKEN_URL", "https://test.com/oauth")

        with pytest.raises(ValueError, match="ORACLE_BASE_URL"):
            OracleConnectorConfig.from_env()

    def test_config_default_endpoints(self, oracle_config):
        """Test default endpoint initialization."""
        assert "purchase_orders" in oracle_config.endpoints
        assert "purchase_requisitions" in oracle_config.endpoints
        assert "suppliers" in oracle_config.endpoints
        assert "shipments" in oracle_config.endpoints
        assert "transportation_orders" in oracle_config.endpoints
        assert "fixed_assets" in oracle_config.endpoints

        # Verify endpoint modules
        assert oracle_config.endpoints["purchase_orders"].module == OracleModule.PROCUREMENT
        assert oracle_config.endpoints["shipments"].module == OracleModule.SCM
        assert oracle_config.endpoints["fixed_assets"].module == OracleModule.FINANCIALS

    def test_get_endpoint_config(self, oracle_config):
        """Test getting endpoint configuration."""
        endpoint = oracle_config.get_endpoint_config("purchase_orders")

        assert endpoint is not None
        assert endpoint.name == "purchase_orders"
        assert endpoint.module == OracleModule.PROCUREMENT

    def test_get_endpoint_config_unknown(self, oracle_config):
        """Test getting unknown endpoint returns None."""
        endpoint = oracle_config.get_endpoint_config("unknown_endpoint")
        assert endpoint is None

    def test_is_endpoint_enabled(self, oracle_config):
        """Test checking if endpoint is enabled."""
        assert oracle_config.is_endpoint_enabled("purchase_orders") is True
        assert oracle_config.is_endpoint_enabled("unknown_endpoint") is False

    def test_get_full_endpoint_url(self, oracle_config):
        """Test getting full endpoint URL."""
        url = oracle_config.get_full_endpoint_url("purchase_orders")

        assert url is not None
        assert url.startswith("https://test.oraclecloud.com")
        assert "/purchaseOrders" in url

    def test_config_validation(self, oracle_config):
        """Test configuration validation."""
        errors = oracle_config.validate()
        assert len(errors) == 0

    def test_config_validation_errors(self, oauth_config):
        """Test configuration validation with errors."""
        # Create config with invalid base URL
        config = OracleConnectorConfig(
            environment=OracleEnvironment.SANDBOX,
            base_url="invalid_url",  # No http:// or https://
            oauth=oauth_config
        )

        errors = config.validate()
        assert len(errors) > 0
        assert any("base_url" in error for error in errors)

    def test_config_to_dict(self, oracle_config):
        """Test converting config to dictionary."""
        config_dict = oracle_config.to_dict()

        assert "environment" in config_dict
        assert "base_url" in config_dict
        assert "oauth" in config_dict
        assert "endpoints" in config_dict
        # Secret should be excluded
        assert "client_secret" not in config_dict.get("oauth", {})

    def test_global_config_singleton(self, mock_env_vars):
        """Test global config singleton pattern."""
        reset_config()

        config1 = get_config()
        config2 = get_config()

        # Should return same instance
        assert config1 is config2
