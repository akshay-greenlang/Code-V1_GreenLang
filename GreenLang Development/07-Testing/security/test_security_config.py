# -*- coding: utf-8 -*-
"""
Tests for SecurityConfig
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from greenlang.security.config import (
    APIKeyConfig,
    AuditConfig,
    AuthenticationConfig,
    CORSConfig,
    EncryptionConfig,
    RateLimitConfig,
    SecurityConfig,
    SecurityHeaders,
    SecurityLevel,
    configure_security,
    get_security_config,
)


class TestSecurityHeaders:
    """Test SecurityHeaders configuration."""

    def test_default_headers(self):
        """Test default security headers."""
        headers = SecurityHeaders()

        assert headers.x_frame_options == "DENY"
        assert headers.x_content_type_options == "nosniff"
        assert "max-age" in headers.strict_transport_security

    def test_to_dict(self):
        """Test conversion to dictionary."""
        headers = SecurityHeaders()
        header_dict = headers.to_dict()

        assert "X-Frame-Options" in header_dict
        assert "Content-Security-Policy" in header_dict
        assert header_dict["X-Frame-Options"] == "DENY"


class TestRateLimitConfig:
    """Test RateLimitConfig."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()

        assert config.enabled is True
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000

    def test_validation(self):
        """Test rate limit validation."""
        # Valid
        config = RateLimitConfig(requests_per_minute=100)
        assert config.requests_per_minute == 100

        # Invalid - too low
        with pytest.raises(PydanticValidationError):
            RateLimitConfig(requests_per_minute=0)

        # Invalid - too high
        with pytest.raises(PydanticValidationError):
            RateLimitConfig(requests_per_minute=20000)


class TestCORSConfig:
    """Test CORSConfig."""

    def test_default_config(self):
        """Test default CORS configuration."""
        config = CORSConfig()

        assert config.enabled is True
        assert "GET" in config.allowed_methods
        assert "Authorization" in config.allowed_headers

    def test_wildcard_validation(self):
        """Test wildcard origin validation."""
        # Valid - wildcard only
        config = CORSConfig(allowed_origins=["*"])
        assert config.allowed_origins == ["*"]

        # Invalid - wildcard with others
        with pytest.raises(PydanticValidationError, match="wildcard"):
            CORSConfig(allowed_origins=["*", "https://example.com"])


class TestAPIKeyConfig:
    """Test APIKeyConfig."""

    def test_default_config(self):
        """Test default API key configuration."""
        config = APIKeyConfig()

        assert config.enabled is True
        assert config.header_name == "X-API-Key"
        assert config.key_prefix == "gl_"
        assert config.key_length >= 16

    def test_key_length_validation(self):
        """Test key length validation."""
        # Valid
        config = APIKeyConfig(key_length=32)
        assert config.key_length == 32

        # Invalid - too short
        with pytest.raises(PydanticValidationError):
            APIKeyConfig(key_length=8)

        # Invalid - too long
        with pytest.raises(PydanticValidationError):
            APIKeyConfig(key_length=200)


class TestAuthenticationConfig:
    """Test AuthenticationConfig."""

    def test_default_config(self):
        """Test default authentication configuration."""
        config = AuthenticationConfig()

        assert config.password_min_length == 12
        assert config.password_require_uppercase is True
        assert config.password_require_lowercase is True
        assert config.password_require_numbers is True
        assert config.password_require_special is True
        assert config.max_login_attempts == 5

    def test_strong_password_requirements(self):
        """Test strong password requirements."""
        config = AuthenticationConfig()

        assert config.password_min_length >= 12
        assert config.password_require_uppercase
        assert config.password_require_numbers


class TestEncryptionConfig:
    """Test EncryptionConfig."""

    def test_default_config(self):
        """Test default encryption configuration."""
        config = EncryptionConfig()

        assert config.encrypt_data_at_rest is True
        assert config.encryption_algorithm == "AES-256-GCM"
        assert config.tls_version == "1.3"

    def test_strong_ciphers(self):
        """Test strong cipher configuration."""
        config = EncryptionConfig()

        assert "AES" in str(config.tls_ciphers)
        assert len(config.tls_ciphers) > 0


class TestAuditConfig:
    """Test AuditConfig."""

    def test_default_config(self):
        """Test default audit configuration."""
        config = AuditConfig()

        assert config.enabled is True
        assert config.log_authentication is True
        assert config.log_authorization is True
        assert config.log_data_access is True
        assert config.retention_days == 365

    def test_comprehensive_logging(self):
        """Test comprehensive audit logging."""
        config = AuditConfig()

        assert config.log_authentication
        assert config.log_authorization
        assert config.log_config_changes
        assert config.log_agent_execution


class TestSecurityConfig:
    """Test SecurityConfig."""

    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()

        assert config.level == SecurityLevel.HIGH
        assert config.enable_security_headers is True
        assert config.enable_audit_logging is True
        assert config.enable_xss_protection is True

    def test_create_for_environment_development(self):
        """Test development environment config."""
        config = SecurityConfig.create_for_environment("development")

        assert config.level == SecurityLevel.LOW
        assert config.rate_limiting.enabled is False
        assert config.audit.enabled is False

    def test_create_for_environment_production(self):
        """Test production environment config."""
        config = SecurityConfig.create_for_environment("production")

        assert config.level == SecurityLevel.HIGH
        assert config.rate_limiting.enabled is True
        assert config.audit.enabled is True

    def test_is_production_ready(self):
        """Test production readiness check."""
        # Production config should be ready
        prod_config = SecurityConfig.create_for_environment("production")
        assert prod_config.is_production_ready()

        # Development config should not be ready
        dev_config = SecurityConfig.create_for_environment("development")
        assert not dev_config.is_production_ready()

    def test_all_security_features_enabled(self):
        """Test all security features enabled by default."""
        config = SecurityConfig()

        assert config.enable_security_headers
        assert config.enable_input_validation
        assert config.enable_audit_logging
        assert config.enable_xss_protection
        assert config.enable_csrf_protection
        assert config.enable_sql_injection_protection
        assert config.enable_path_traversal_protection

    def test_blocked_file_extensions(self):
        """Test blocked file extensions configuration."""
        config = SecurityConfig()

        dangerous_extensions = [".exe", ".dll", ".bat", ".sh"]
        for ext in dangerous_extensions:
            assert ext in config.blocked_file_extensions


class TestGlobalSecurityConfig:
    """Test global security configuration."""

    def test_get_security_config(self):
        """Test getting global security config."""
        config = get_security_config()
        assert config is not None
        assert isinstance(config, SecurityConfig)

    def test_configure_security(self):
        """Test configuring global security."""
        custom_config = SecurityConfig(level=SecurityLevel.MAXIMUM)
        configured = configure_security(custom_config)

        assert configured.level == SecurityLevel.MAXIMUM

        # Verify it's now the global config
        global_config = get_security_config()
        assert global_config.level == SecurityLevel.MAXIMUM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
