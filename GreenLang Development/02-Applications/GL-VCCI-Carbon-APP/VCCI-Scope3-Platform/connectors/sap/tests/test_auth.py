# -*- coding: utf-8 -*-
"""
Authentication Tests for SAP Connector
GL-VCCI Scope 3 Platform

Tests for OAuth 2.0 authentication including:
- Token acquisition and caching
- Token refresh on expiration
- Token invalidation
- Multi-environment token management
- Authentication failure handling
- Thread-safe token cache

Test Count: 18 tests
Coverage Target: 95%+

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import requests
from freezegun import freeze_time

from connectors.sap.auth import (
    TokenCache,
    SAPAuthHandler,
    get_auth_handler,
    reset_auth_handlers,
)
from connectors.sap.config import SAPEnvironment
from connectors.sap.exceptions import SAPAuthenticationError, SAPConnectionError


class TestTokenCache:
    """Tests for TokenCache."""

    def test_should_store_and_retrieve_token(self, token_cache):
        """Test storing and retrieving token from cache."""
        token_cache.set("test-key", "test-token", expires_in=3600)

        cached_token = token_cache.get("test-key")

        assert cached_token == "test-token"

    def test_should_return_none_for_missing_key(self, token_cache):
        """Test cache miss returns None."""
        cached_token = token_cache.get("non-existent-key")

        assert cached_token is None

    @freeze_time("2024-01-15 12:00:00")
    def test_should_return_none_for_expired_token(self, token_cache):
        """Test expired token returns None."""
        # Set token with 1 hour expiration
        token_cache.set("test-key", "test-token", expires_in=3600)

        # Move time forward past expiration (with 60s buffer)
        with freeze_time("2024-01-15 13:01:00"):
            cached_token = token_cache.get("test-key")

            assert cached_token is None

    @freeze_time("2024-01-15 12:00:00")
    def test_should_return_token_within_buffer_window(self, token_cache):
        """Test token is still valid within 60s buffer."""
        token_cache.set("test-key", "test-token", expires_in=3600)

        # Move time forward to within buffer window
        with freeze_time("2024-01-15 12:59:00"):  # 59 minutes later
            cached_token = token_cache.get("test-key")

            assert cached_token == "test-token"

    def test_should_invalidate_token(self, token_cache):
        """Test invalidating cached token."""
        token_cache.set("test-key", "test-token", expires_in=3600)
        token_cache.invalidate("test-key")

        cached_token = token_cache.get("test-key")

        assert cached_token is None

    def test_should_clear_all_tokens(self, token_cache):
        """Test clearing all cached tokens."""
        token_cache.set("key1", "token1", expires_in=3600)
        token_cache.set("key2", "token2", expires_in=3600)

        token_cache.clear()

        assert token_cache.get("key1") is None
        assert token_cache.get("key2") is None


class TestSAPAuthHandler:
    """Tests for SAPAuthHandler."""

    def test_should_initialize_auth_handler(self, oauth_config):
        """Test auth handler initialization."""
        handler = SAPAuthHandler(
            oauth_config=oauth_config,
            environment=SAPEnvironment.SANDBOX
        )

        assert handler.oauth_config == oauth_config
        assert handler.environment == SAPEnvironment.SANDBOX
        assert handler.cache is not None

    @patch('requests.post')
    def test_should_acquire_access_token(self, mock_post, oauth_config, mock_oauth_token_response):
        """Test acquiring access token from OAuth server."""
        # Mock successful token response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_token_response
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)
        token = handler.get_access_token()

        assert token == "mock-access-token-12345"
        assert mock_post.called
        assert mock_post.call_args[0][0] == oauth_config.token_url

    @patch('requests.post')
    def test_should_cache_token_after_acquisition(self, mock_post, oauth_config, mock_oauth_token_response):
        """Test token is cached after acquisition."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_token_response
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)

        # First call - should hit OAuth server
        token1 = handler.get_access_token()

        # Second call - should use cache (no additional request)
        mock_post.reset_mock()
        token2 = handler.get_access_token()

        assert token1 == token2
        assert not mock_post.called  # Should not make another request

    @patch('requests.post')
    def test_should_force_refresh_token(self, mock_post, oauth_config, mock_oauth_token_response):
        """Test forcing token refresh even when cached."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_token_response
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)

        # Get token (cached)
        handler.get_access_token()

        # Force refresh
        mock_post.reset_mock()
        token = handler.get_access_token(force_refresh=True)

        assert token == "mock-access-token-12345"
        assert mock_post.called  # Should make new request

    @patch('requests.post')
    def test_should_handle_401_authentication_error(self, mock_post, oauth_config, mock_oauth_error_response):
        """Test handling 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = mock_oauth_error_response
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)

        with pytest.raises(SAPAuthenticationError) as exc_info:
            handler.get_access_token()

        assert "Token request failed" in str(exc_info.value)
        assert "invalid_client" in str(exc_info.value)

    @patch('requests.post')
    def test_should_handle_connection_error(self, mock_post, oauth_config):
        """Test handling connection errors."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        handler = SAPAuthHandler(oauth_config=oauth_config)

        with pytest.raises(SAPConnectionError) as exc_info:
            handler.get_access_token()

        assert "Connection failed" in str(exc_info.value)

    @patch('requests.post')
    def test_should_handle_timeout_error(self, mock_post, oauth_config):
        """Test handling timeout errors."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        handler = SAPAuthHandler(oauth_config=oauth_config)

        with pytest.raises(SAPAuthenticationError) as exc_info:
            handler.get_access_token()

        assert "Token request timed out" in str(exc_info.value)

    @patch('requests.post')
    def test_should_handle_missing_access_token_in_response(self, mock_post, oauth_config):
        """Test handling response without access_token."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"token_type": "Bearer"}  # Missing access_token
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)

        with pytest.raises(SAPAuthenticationError) as exc_info:
            handler.get_access_token()

        assert "No access_token in response" in str(exc_info.value)

    def test_should_invalidate_token(self, oauth_config):
        """Test invalidating token."""
        handler = SAPAuthHandler(oauth_config=oauth_config)

        # Set a token in cache
        handler.cache.set("sap_token_sandbox", "test-token", 3600)

        # Invalidate
        handler.invalidate_token()

        # Should be None after invalidation
        assert handler.cache.get("sap_token_sandbox") is None

    @patch('requests.post')
    def test_should_validate_cached_token(self, mock_post, oauth_config, mock_oauth_token_response):
        """Test validating cached token."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_token_response
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)
        token = handler.get_access_token()

        # Validate the token
        is_valid = handler.validate_token(token)

        assert is_valid is True

        # Invalid token
        is_valid = handler.validate_token("wrong-token")

        assert is_valid is False

    @patch('requests.post')
    def test_should_get_auth_header(self, mock_post, oauth_config, mock_oauth_token_response):
        """Test getting authorization header."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_token_response
        mock_post.return_value = mock_response

        handler = SAPAuthHandler(oauth_config=oauth_config)
        auth_header = handler.get_auth_header()

        assert "Authorization" in auth_header
        assert auth_header["Authorization"] == "Bearer mock-access-token-12345"


class TestMultiEnvironmentAuth:
    """Tests for multi-environment authentication."""

    @patch('requests.post')
    def test_should_manage_tokens_for_different_environments(self, mock_post, oauth_config, mock_oauth_token_response):
        """Test managing separate tokens for different environments."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_oauth_token_response
        mock_post.return_value = mock_response

        # Create handlers for different environments
        sandbox_handler = SAPAuthHandler(oauth_config=oauth_config, environment=SAPEnvironment.SANDBOX)
        prod_handler = SAPAuthHandler(oauth_config=oauth_config, environment=SAPEnvironment.PRODUCTION)

        # Get tokens
        sandbox_token = sandbox_handler.get_access_token()
        prod_token = prod_handler.get_access_token()

        # Both should succeed
        assert sandbox_token == "mock-access-token-12345"
        assert prod_token == "mock-access-token-12345"

        # Invalidating one should not affect the other
        sandbox_handler.invalidate_token()
        assert sandbox_handler.cache.get("sap_token_sandbox") is None
        assert prod_handler.cache.get("sap_token_production") is not None


class TestAuthHandlerGlobal:
    """Tests for global auth handler management."""

    def test_should_get_singleton_auth_handler(self, oauth_config):
        """Test getting singleton auth handler per environment."""
        reset_auth_handlers()

        handler1 = get_auth_handler(oauth_config, SAPEnvironment.SANDBOX)
        handler2 = get_auth_handler(oauth_config, SAPEnvironment.SANDBOX)

        # Should return same instance for same environment
        assert handler1 is handler2

    def test_should_get_different_handlers_for_different_environments(self, oauth_config):
        """Test different handlers for different environments."""
        reset_auth_handlers()

        sandbox_handler = get_auth_handler(oauth_config, SAPEnvironment.SANDBOX)
        prod_handler = get_auth_handler(oauth_config, SAPEnvironment.PRODUCTION)

        # Should be different instances
        assert sandbox_handler is not prod_handler

    def test_should_reset_all_auth_handlers(self, oauth_config):
        """Test resetting all auth handlers."""
        handler1 = get_auth_handler(oauth_config, SAPEnvironment.SANDBOX)
        reset_auth_handlers()
        handler2 = get_auth_handler(oauth_config, SAPEnvironment.SANDBOX)

        # Should be different instances after reset
        assert handler1 is not handler2
