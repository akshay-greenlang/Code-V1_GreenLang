# -*- coding: utf-8 -*-
"""
JWT Validation Security Tests - CRITICAL SECURITY FIX

This test suite validates the JWT authentication implementation at
tenancy/tenant_context.py to prevent authentication bypass vulnerabilities.

Test Coverage:
- Token signature verification
- Expiration validation
- Issuer and audience validation
- Missing claims detection
- Malformed token handling
- Token generation and validation
- Error handling and security boundaries

CWE-287: Improper Authentication Prevention
"""

import pytest
import jwt
import time
from datetime import datetime, timedelta
from uuid import uuid4

import sys
import os
from greenlang.determinism import deterministic_uuid, DeterministicClock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tenancy.tenant_context import JWTValidator, AuthenticationError


class TestJWTValidatorInitialization:
    """Test JWTValidator initialization and configuration."""

    def test_validator_initialization_success(self):
        """Test successful validator initialization."""
        validator = JWTValidator(
            secret_key="test-secret-key-min-32-chars",
            algorithm="HS256",
            issuer="greenlang.ai",
            audience="greenlang-api"
        )
        assert validator.secret_key == "test-secret-key-min-32-chars"
        assert validator.algorithm == "HS256"
        assert validator.issuer == "greenlang.ai"
        assert validator.audience == "greenlang-api"

    def test_validator_empty_secret_key(self):
        """Test that empty secret key raises ValueError."""
        with pytest.raises(ValueError, match="secret_key cannot be empty"):
            JWTValidator(secret_key="")

    def test_validator_unsupported_algorithm(self):
        """Test that unsupported algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            JWTValidator(secret_key="test-key", algorithm="INVALID")

    def test_validator_supported_algorithms(self):
        """Test all supported algorithms can be initialized."""
        algorithms = [
            "HS256", "HS384", "HS512",  # HMAC
            "RS256", "RS384", "RS512",  # RSA
            "ES256", "ES384", "ES512"   # ECDSA
        ]
        for algo in algorithms:
            validator = JWTValidator(secret_key="test-key", algorithm=algo)
            assert validator.algorithm == algo


class TestTokenGeneration:
    """Test JWT token generation."""

    def test_generate_valid_access_token(self):
        """Test generation of valid access token."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            token_type="access",
            expires_in=3600
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_refresh_token(self):
        """Test generation of refresh token."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            token_type="refresh",
            expires_in=604800  # 7 days
        )

        assert token is not None
        payload = validator.validate_token(token)
        assert payload["type"] == "refresh"

    def test_generate_token_with_issuer_and_audience(self):
        """Test token generation includes issuer and audience."""
        validator = JWTValidator(
            secret_key="test-secret-key",
            issuer="greenlang.ai",
            audience="greenlang-api"
        )
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        token = validator.generate_token(tenant_id=tenant_id, user_id=user_id)
        payload = validator.validate_token(token)

        assert payload["iss"] == "greenlang.ai"
        assert payload["aud"] == "greenlang-api"

    def test_generate_token_with_additional_claims(self):
        """Test token generation with additional custom claims."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            additional_claims={
                "role": "admin",
                "permissions": ["read", "write"]
            }
        )

        payload = validator.validate_token(token)
        assert payload["role"] == "admin"
        assert payload["permissions"] == ["read", "write"]

    def test_generate_token_empty_tenant_id(self):
        """Test that empty tenant_id raises ValueError."""
        validator = JWTValidator(secret_key="test-secret-key")
        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            validator.generate_token(tenant_id="", user_id="user-123")

    def test_generate_token_empty_user_id(self):
        """Test that empty user_id raises ValueError."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        with pytest.raises(ValueError, match="user_id cannot be empty"):
            validator.generate_token(tenant_id=tenant_id, user_id="")

    def test_generate_token_invalid_type(self):
        """Test that invalid token type raises ValueError."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        with pytest.raises(ValueError, match="Invalid token_type"):
            validator.generate_token(
                tenant_id=tenant_id,
                user_id="user-123",
                token_type="invalid"
            )


class TestTokenValidation:
    """Test JWT token validation - CRITICAL SECURITY TESTS."""

    def test_validate_valid_token(self):
        """Test validation of valid token succeeds."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        token = validator.generate_token(tenant_id=tenant_id, user_id=user_id)
        payload = validator.validate_token(token)

        assert payload["tenant_id"] == tenant_id
        assert payload["sub"] == user_id
        assert payload["type"] == "access"

    def test_validate_expired_token(self):
        """Test that expired token raises AuthenticationError."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        # Generate token that expires immediately
        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            expires_in=-1  # Already expired
        )

        with pytest.raises(AuthenticationError, match="expired"):
            validator.validate_token(token)

    def test_validate_invalid_signature(self):
        """Test that invalid signature raises AuthenticationError."""
        validator1 = JWTValidator(secret_key="secret-1")
        validator2 = JWTValidator(secret_key="secret-2")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        # Generate token with secret-1
        token = validator1.generate_token(tenant_id=tenant_id, user_id=user_id)

        # Try to validate with secret-2 (different key = invalid signature)
        with pytest.raises(AuthenticationError, match="signature"):
            validator2.validate_token(token)

    def test_validate_token_missing_tenant_id(self):
        """Test that token without tenant_id raises AuthenticationError."""
        validator = JWTValidator(secret_key="test-secret-key")

        # Manually create token without tenant_id
        now = DeterministicClock.utcnow()
        payload = {
            "sub": "user-123",
            "type": "access",
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")

        with pytest.raises(AuthenticationError, match="tenant_id"):
            validator.validate_token(token)

    def test_validate_token_missing_user_id(self):
        """Test that token without user_id (sub) raises AuthenticationError."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Manually create token without sub (user_id)
        now = DeterministicClock.utcnow()
        payload = {
            "tenant_id": tenant_id,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")

        with pytest.raises(AuthenticationError, match="user_id"):
            validator.validate_token(token)

    def test_validate_token_invalid_type(self):
        """Test that token with invalid type raises AuthenticationError."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Manually create token with invalid type
        now = DeterministicClock.utcnow()
        payload = {
            "tenant_id": tenant_id,
            "sub": "user-123",
            "type": "invalid",
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")

        with pytest.raises(AuthenticationError, match="Invalid token type"):
            validator.validate_token(token)

    def test_validate_token_empty_string(self):
        """Test that empty token string raises AuthenticationError."""
        validator = JWTValidator(secret_key="test-secret-key")
        with pytest.raises(AuthenticationError, match="cannot be empty"):
            validator.validate_token("")

    def test_validate_malformed_token(self):
        """Test that malformed token raises AuthenticationError."""
        validator = JWTValidator(secret_key="test-secret-key")
        with pytest.raises(AuthenticationError, match="Malformed token"):
            validator.validate_token("not.a.valid.jwt.token")

    def test_validate_token_wrong_issuer(self):
        """Test that token with wrong issuer raises AuthenticationError."""
        validator = JWTValidator(
            secret_key="test-secret-key",
            issuer="expected-issuer"
        )
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Generate token with different issuer
        now = DeterministicClock.utcnow()
        payload = {
            "tenant_id": tenant_id,
            "sub": "user-123",
            "type": "access",
            "iss": "wrong-issuer",
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")

        with pytest.raises(AuthenticationError, match="issuer"):
            validator.validate_token(token)

    def test_validate_token_wrong_audience(self):
        """Test that token with wrong audience raises AuthenticationError."""
        validator = JWTValidator(
            secret_key="test-secret-key",
            audience="expected-audience"
        )
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Generate token with different audience
        now = DeterministicClock.utcnow()
        payload = {
            "tenant_id": tenant_id,
            "sub": "user-123",
            "type": "access",
            "aud": "wrong-audience",
            "iat": now,
            "exp": now + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")

        with pytest.raises(AuthenticationError, match="audience"):
            validator.validate_token(token)


class TestTokenLeeway:
    """Test clock skew tolerance (leeway)."""

    def test_leeway_allows_near_expired_token(self):
        """Test that leeway allows tokens expiring within tolerance."""
        validator = JWTValidator(secret_key="test-secret-key", leeway=60)
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        # Create token that expires in 30 seconds
        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            expires_in=30
        )

        # Should be valid with 60-second leeway
        payload = validator.validate_token(token)
        assert payload["tenant_id"] == tenant_id


class TestSecurityBoundaries:
    """Test security boundaries and edge cases."""

    def test_cannot_override_protected_claims(self):
        """Test that protected claims cannot be overridden in additional_claims."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-123"

        # Try to override protected claim
        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            additional_claims={
                "tenant_id": "hacker-tenant",  # Should be ignored
                "sub": "hacker-user",          # Should be ignored
                "custom": "allowed"             # Should be included
            }
        )

        payload = validator.validate_token(token)
        # Protected claims should not be overridden
        assert payload["tenant_id"] == tenant_id
        assert payload["sub"] == user_id
        # Custom claim should be included
        assert payload["custom"] == "allowed"

    def test_token_signature_always_verified(self):
        """Test that signature verification cannot be disabled."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Create token with correct secret
        token = validator.generate_token(tenant_id=tenant_id, user_id="user-123")

        # Create validator with different secret
        wrong_validator = JWTValidator(secret_key="wrong-secret-key")

        # Validation MUST fail due to signature mismatch
        with pytest.raises(AuthenticationError, match="signature"):
            wrong_validator.validate_token(token)

    def test_expiration_always_checked(self):
        """Test that expiration is always checked."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))

        # Create expired token
        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id="user-123",
            expires_in=-3600  # Expired 1 hour ago
        )

        # Validation MUST fail due to expiration
        with pytest.raises(AuthenticationError, match="expired"):
            validator.validate_token(token)


class TestTokenRoundTrip:
    """Test complete token generation and validation cycle."""

    def test_access_token_roundtrip(self):
        """Test complete access token lifecycle."""
        validator = JWTValidator(
            secret_key="test-secret-key-min-32-chars",
            algorithm="HS256",
            issuer="greenlang.ai",
            audience="greenlang-api"
        )

        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-456"

        # Generate token
        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            token_type="access",
            expires_in=3600,
            additional_claims={"role": "admin"}
        )

        # Validate token
        payload = validator.validate_token(token)

        # Verify all claims
        assert payload["tenant_id"] == tenant_id
        assert payload["sub"] == user_id
        assert payload["type"] == "access"
        assert payload["iss"] == "greenlang.ai"
        assert payload["aud"] == "greenlang-api"
        assert payload["role"] == "admin"
        assert "iat" in payload
        assert "exp" in payload
        assert "nbf" in payload

    def test_refresh_token_roundtrip(self):
        """Test complete refresh token lifecycle."""
        validator = JWTValidator(secret_key="test-secret-key")
        tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        user_id = "user-789"

        # Generate refresh token (7 days)
        token = validator.generate_token(
            tenant_id=tenant_id,
            user_id=user_id,
            token_type="refresh",
            expires_in=604800
        )

        # Validate token
        payload = validator.validate_token(token)

        # Verify claims
        assert payload["tenant_id"] == tenant_id
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
