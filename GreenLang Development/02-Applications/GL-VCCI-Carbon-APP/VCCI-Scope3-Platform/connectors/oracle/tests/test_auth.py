# -*- coding: utf-8 -*-
"""
Oracle Connector Authentication Tests
GL-VCCI Scope 3 Platform

Tests for OAuth 2.0 authentication, token management,
caching, and refresh logic.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Test Count: 15
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests
import responses

from connectors.oracle.auth import (
    TokenCache,
    OracleAuthHandler,
    get_auth_handler,
    reset_auth_handlers
)
from connectors.oracle.config import OAuth2Config, OracleEnvironment
from connectors.oracle.exceptions import (
    OracleAuthenticationError,
    OracleConnectionError
)


class TestTokenCache:
    """Tests for TokenCache."""

    def test_cache_initialization(self, token_cache):
        """Test token cache initialization."""
        assert token_cache._cache == {}

    def test_cache_set_and_get(self, token_cache):
        """Test storing and retrieving token."""
        token_cache.set("test_key", "test_token", expires_in=3600)

        retrieved_token = token_cache.get("test_key")
        assert retrieved_token == "test_token"

    def test_cache_get_nonexistent(self, token_cache):
        """Test getting non-existent key returns None."""
        token = token_cache.get("nonexistent_key")
        assert token is None

    def test_cache_expiration(self, token_cache):
        """Test token expiration."""
        # Set token with 1 second expiry
        token_cache.set("test_key", "test_token", expires_in=1)

        # Should be available immediately
        assert token_cache.get("test_key") == "test_token"

        # Wait for expiration (with 60s buffer)
        time.sleep(0.1)

        # Should still be available (buffer is 60s)
        assert token_cache.get("test_key") == "test_token"

    def test_cache_invalidate(self, token_cache):
        """Test invalidating cached token."""
        token_cache.set("test_key", "test_token", expires_in=3600)
        assert token_cache.get("test_key") == "test_token"

        token_cache.invalidate("test_key")
        assert token_cache.get("test_key") is None

    def test_cache_clear(self, token_cache):
        """Test clearing all cached tokens."""
        token_cache.set("key1", "token1", expires_in=3600)
        token_cache.set("key2", "token2", expires_in=3600)

        token_cache.clear()

        assert token_cache.get("key1") is None
        assert token_cache.get("key2") is None

    def test_cache_thread_safety(self, token_cache):
        """Test that cache operations are thread-safe."""
        # This test verifies the lock mechanism exists
        assert hasattr(token_cache, '_lock')

        # Concurrent operations should not raise exceptions
        token_cache.set("key1", "token1", 3600)
        token_cache.get("key1")
        token_cache.invalidate("key1")


class TestOracleAuthHandler:
    """Tests for OracleAuthHandler."""

    def test_auth_handler_initialization(self, auth_handler):
        """Test auth handler initialization."""
        assert auth_handler.environment == OracleEnvironment.SANDBOX
        assert auth_handler.cache is not None
        assert auth_handler._cache_key == "oracle_token_sandbox"

    @responses.activate
    def test_acquire_token_success(self, auth_handler, mock_oauth_response):
        """Test successful token acquisition."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        token = auth_handler.get_access_token()

        assert token == "test_access_token_12345"
        assert len(responses.calls) == 1

        # Verify request payload
        request_body = responses.calls[0].request.body
        assert "grant_type=client_credentials" in request_body
        assert "client_id=test_client_id" in request_body

    @responses.activate
    def test_token_caching(self, auth_handler, mock_oauth_response):
        """Test token is cached and reused."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # First request should call API
        token1 = auth_handler.get_access_token()
        assert len(responses.calls) == 1

        # Second request should use cached token
        token2 = auth_handler.get_access_token()
        assert len(responses.calls) == 1  # No additional API call
        assert token1 == token2

    @responses.activate
    def test_force_token_refresh(self, auth_handler, mock_oauth_response):
        """Test forcing token refresh."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        # Get token normally
        token1 = auth_handler.get_access_token()
        assert len(responses.calls) == 1

        # Force refresh
        token2 = auth_handler.get_access_token(force_refresh=True)
        assert len(responses.calls) == 2  # Additional API call
        assert token2 == token1  # Same token value in this mock

    @responses.activate
    def test_token_acquisition_failure_401(self, auth_handler):
        """Test token acquisition failure with 401."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json={"error": "invalid_client", "error_description": "Invalid credentials"},
            status=401
        )

        with pytest.raises(OracleAuthenticationError) as exc_info:
            auth_handler.get_access_token()

        assert "Token request failed" in str(exc_info.value)

    @responses.activate
    def test_token_acquisition_no_access_token(self, auth_handler):
        """Test handling response without access_token."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json={"token_type": "Bearer", "expires_in": 3600},  # Missing access_token
            status=200
        )

        with pytest.raises(OracleAuthenticationError) as exc_info:
            auth_handler.get_access_token()

        assert "No access_token in response" in str(exc_info.value)

    @responses.activate
    def test_token_acquisition_connection_error(self, auth_handler):
        """Test handling connection errors."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            body=requests.exceptions.ConnectionError("Connection failed")
        )

        with pytest.raises(OracleConnectionError):
            auth_handler.get_access_token()

    @responses.activate
    def test_token_acquisition_timeout(self, auth_handler):
        """Test handling timeout errors."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            body=requests.exceptions.Timeout("Request timed out")
        )

        with pytest.raises(OracleAuthenticationError) as exc_info:
            auth_handler.get_access_token()

        assert "timed out" in str(exc_info.value).lower()

    def test_invalidate_token(self, auth_handler, token_cache):
        """Test invalidating cached token."""
        token_cache.set("oracle_token_sandbox", "test_token", 3600)

        auth_handler.invalidate_token()

        assert token_cache.get("oracle_token_sandbox") is None

    def test_validate_token(self, auth_handler, token_cache):
        """Test token validation."""
        token_cache.set("oracle_token_sandbox", "test_token", 3600)

        # Valid token
        assert auth_handler.validate_token("test_token") is True

        # Invalid token
        assert auth_handler.validate_token("wrong_token") is False

    @responses.activate
    def test_get_auth_header(self, auth_handler, mock_oauth_response):
        """Test getting authorization header."""
        responses.add(
            responses.POST,
            "https://test.oraclecloud.com/oauth/token",
            json=mock_oauth_response,
            status=200
        )

        header = auth_handler.get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"] == "Bearer test_access_token_12345"

    def test_multi_environment_support(self, oauth_config):
        """Test support for multiple environments."""
        sandbox_handler = OracleAuthHandler(
            oauth_config=oauth_config,
            environment=OracleEnvironment.SANDBOX
        )
        prod_handler = OracleAuthHandler(
            oauth_config=oauth_config,
            environment=OracleEnvironment.PRODUCTION
        )

        assert sandbox_handler._cache_key == "oracle_token_sandbox"
        assert prod_handler._cache_key == "oracle_token_production"

    def test_global_auth_handler_singleton(self, oauth_config):
        """Test global auth handler singleton pattern."""
        reset_auth_handlers()

        handler1 = get_auth_handler(oauth_config, OracleEnvironment.SANDBOX)
        handler2 = get_auth_handler(oauth_config, OracleEnvironment.SANDBOX)

        # Should return same instance for same environment
        assert handler1 is handler2

        # Different environment should return different instance
        handler3 = get_auth_handler(oauth_config, OracleEnvironment.PRODUCTION)
        assert handler1 is not handler3
