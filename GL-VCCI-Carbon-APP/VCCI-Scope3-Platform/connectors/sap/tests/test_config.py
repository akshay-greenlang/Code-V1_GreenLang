"""
Configuration Tests for SAP Connector
GL-VCCI Scope 3 Platform

Tests for SAP connector configuration management including:
- Configuration loading from environment variables
- Default values and validation
- Endpoint configuration
- Retry and rate limit settings
- Environment-specific configurations

Test Count: 15 tests
Coverage Target: 95%+

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import os
import pytest
from pydantic import ValidationError

from connectors.sap.config import (
    SAPConnectorConfig,
    OAuth2Config,
    ODataEndpoint,
    RetryConfig,
    RateLimitConfig,
    TimeoutConfig,
    SAPEnvironment,
    SAPModule,
    get_config,
    reset_config,
)


class TestOAuth2Config:
    """Tests for OAuth2Config model."""

    def test_should_create_oauth_config_with_required_fields(self):
        """Test OAuth config creation with required fields."""
        config = OAuth2Config(
            client_id="test-id",
            client_secret="test-secret",
            token_url="https://auth.sap.com/oauth/token"
        )

        assert config.client_id == "test-id"
        assert config.client_secret == "test-secret"
        assert config.token_url == "https://auth.sap.com/oauth/token"
        assert config.scope == "API_BUSINESS_PARTNER"  # Default
        assert config.grant_type == "client_credentials"  # Default
        assert config.token_cache_ttl == 3300  # Default

    def test_should_validate_token_cache_ttl_range(self):
        """Test token cache TTL validation."""
        # Valid range (60-86400)
        config = OAuth2Config(
            client_id="test-id",
            client_secret="test-secret",
            token_url="https://auth.sap.com/oauth/token",
            token_cache_ttl=3600
        )
        assert config.token_cache_ttl == 3600

        # Below minimum
        with pytest.raises(ValidationError):
            OAuth2Config(
                client_id="test-id",
                client_secret="test-secret",
                token_url="https://auth.sap.com/oauth/token",
                token_cache_ttl=30
            )

        # Above maximum
        with pytest.raises(ValidationError):
            OAuth2Config(
                client_id="test-id",
                client_secret="test-secret",
                token_url="https://auth.sap.com/oauth/token",
                token_cache_ttl=100000
            )


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_should_create_retry_config_with_defaults(self):
        """Test retry config with default values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 8.0
        assert config.backoff_multiplier == 2.0
        assert 429 in config.retry_on_status_codes
        assert 500 in config.retry_on_status_codes

    def test_should_validate_max_delay_greater_than_base_delay(self):
        """Test max_delay must be greater than base_delay."""
        # Valid: max_delay > base_delay
        config = RetryConfig(base_delay=1.0, max_delay=10.0)
        assert config.max_delay == 10.0

        # Invalid: max_delay < base_delay
        with pytest.raises(ValidationError) as exc_info:
            RetryConfig(base_delay=10.0, max_delay=5.0)

        assert "max_delay must be greater than base_delay" in str(exc_info.value)


class TestRateLimitConfig:
    """Tests for RateLimitConfig model."""

    def test_should_create_rate_limit_config_with_defaults(self):
        """Test rate limit config with defaults."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 10
        assert config.enabled is True
        assert config.burst_size == 5

    def test_should_validate_requests_per_minute_range(self):
        """Test requests per minute validation."""
        # Valid
        config = RateLimitConfig(requests_per_minute=100)
        assert config.requests_per_minute == 100

        # Below minimum
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_minute=0)

        # Above maximum
        with pytest.raises(ValidationError):
            RateLimitConfig(requests_per_minute=2000)


class TestODataEndpoint:
    """Tests for ODataEndpoint model."""

    def test_should_create_endpoint_with_required_fields(self):
        """Test endpoint creation with required fields."""
        endpoint = ODataEndpoint(
            name="purchase_orders",
            service_path="/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV",
            entity_set="C_PurchaseOrderTP",
            module=SAPModule.MM
        )

        assert endpoint.name == "purchase_orders"
        assert endpoint.service_path == "/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV"
        assert endpoint.entity_set == "C_PurchaseOrderTP"
        assert endpoint.module == SAPModule.MM
        assert endpoint.enabled is True
        assert endpoint.batch_size == 1000

    def test_should_validate_batch_size_range(self):
        """Test batch size validation."""
        # Valid
        endpoint = ODataEndpoint(
            name="test",
            service_path="/test",
            entity_set="Test",
            module=SAPModule.MM,
            batch_size=500
        )
        assert endpoint.batch_size == 500

        # Below minimum
        with pytest.raises(ValidationError):
            ODataEndpoint(
                name="test",
                service_path="/test",
                entity_set="Test",
                module=SAPModule.MM,
                batch_size=0
            )


class TestSAPConnectorConfig:
    """Tests for SAPConnectorConfig main configuration."""

    def test_should_create_config_from_environment(self, mock_env_vars):
        """Test configuration loading from environment variables."""
        config = SAPConnectorConfig.from_env()

        assert config.environment == SAPEnvironment.SANDBOX
        assert config.base_url == "https://sandbox.api.sap.com"
        assert config.oauth.client_id == "test-client-id"
        assert config.oauth.client_secret == "test-client-secret"
        assert config.rate_limit.requests_per_minute == 10
        assert config.default_batch_size == 1000
        assert config.debug_mode is True

    def test_should_raise_error_when_base_url_missing(self, monkeypatch):
        """Test error when SAP_BASE_URL is missing."""
        monkeypatch.delenv("SAP_BASE_URL", raising=False)
        monkeypatch.setenv("SAP_CLIENT_ID", "test-id")
        monkeypatch.setenv("SAP_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv("SAP_TOKEN_URL", "https://auth.sap.com/oauth/token")

        with pytest.raises(ValueError) as exc_info:
            SAPConnectorConfig.from_env()

        assert "SAP_BASE_URL" in str(exc_info.value)

    def test_should_raise_error_when_oauth_credentials_missing(self, monkeypatch):
        """Test error when OAuth credentials are missing."""
        monkeypatch.setenv("SAP_BASE_URL", "https://sandbox.api.sap.com")
        monkeypatch.delenv("SAP_CLIENT_ID", raising=False)
        monkeypatch.delenv("SAP_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("SAP_TOKEN_URL", raising=False)

        with pytest.raises(ValueError) as exc_info:
            SAPConnectorConfig.from_env()

        assert "SAP_CLIENT_ID" in str(exc_info.value)
        assert "SAP_CLIENT_SECRET" in str(exc_info.value)
        assert "SAP_TOKEN_URL" in str(exc_info.value)

    def test_should_initialize_default_endpoints(self, sap_config):
        """Test default endpoints are initialized."""
        assert len(sap_config.endpoints) == 7

        # Check MM endpoints
        assert "purchase_orders" in sap_config.endpoints
        assert "goods_receipts" in sap_config.endpoints
        assert "vendor_master" in sap_config.endpoints
        assert "material_master" in sap_config.endpoints

        # Check SD endpoints
        assert "outbound_deliveries" in sap_config.endpoints
        assert "transportation_orders" in sap_config.endpoints

        # Check FI endpoints
        assert "fixed_assets" in sap_config.endpoints

    def test_should_get_endpoint_config(self, sap_config):
        """Test getting endpoint configuration."""
        po_endpoint = sap_config.get_endpoint_config("purchase_orders")

        assert po_endpoint is not None
        assert po_endpoint.name == "purchase_orders"
        assert po_endpoint.module == SAPModule.MM
        assert po_endpoint.enabled is True

        # Non-existent endpoint
        assert sap_config.get_endpoint_config("invalid_endpoint") is None

    def test_should_check_if_endpoint_is_enabled(self, sap_config):
        """Test checking if endpoint is enabled."""
        assert sap_config.is_endpoint_enabled("purchase_orders") is True
        assert sap_config.is_endpoint_enabled("invalid_endpoint") is False

    def test_should_get_full_endpoint_url(self, sap_config):
        """Test building full endpoint URL."""
        url = sap_config.get_full_endpoint_url("purchase_orders")

        assert url == "https://sandbox.api.sap.com/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV"

        # Non-existent endpoint
        assert sap_config.get_full_endpoint_url("invalid_endpoint") is None

    def test_should_validate_configuration(self, sap_config):
        """Test configuration validation."""
        errors = sap_config.validate()

        assert len(errors) == 0

    def test_should_return_validation_errors_for_invalid_config(self):
        """Test validation errors for invalid configuration."""
        config = SAPConnectorConfig(
            base_url="invalid-url",  # Invalid URL
            oauth=OAuth2Config(
                client_id="",  # Empty client ID
                client_secret="secret",
                token_url="https://auth.sap.com/oauth/token"
            )
        )

        errors = config.validate()

        assert len(errors) > 0
        assert any("base_url must start with http" in err for err in errors)
        assert any("client_id is required" in err for err in errors)

    def test_should_convert_config_to_dict_without_secrets(self, sap_config):
        """Test converting config to dict (excluding secrets)."""
        config_dict = sap_config.to_dict()

        assert "base_url" in config_dict
        assert "oauth" in config_dict
        assert "client_id" in config_dict["oauth"]
        assert "client_secret" not in config_dict["oauth"]  # Secret excluded
        assert "endpoints" in config_dict

    def test_should_use_environment_override(self, mock_env_vars):
        """Test environment override in from_env."""
        config = SAPConnectorConfig.from_env(environment="production")

        assert config.environment == SAPEnvironment.PRODUCTION


class TestConfigGlobalInstance:
    """Tests for global configuration instance."""

    def test_should_get_global_config_instance(self, mock_env_vars):
        """Test getting global config instance."""
        reset_config()  # Ensure clean state

        config1 = get_config()
        config2 = get_config()

        # Should return same instance
        assert config1 is config2

    def test_should_reset_global_config(self, mock_env_vars):
        """Test resetting global config."""
        config1 = get_config()
        reset_config()
        config2 = get_config()

        # Should be different instances after reset
        assert config1 is not config2
