# -*- coding: utf-8 -*-
"""
SAP S/4HANA OAuth 2.0 Authentication Handler
GL-VCCI Scope 3 Platform

OAuth 2.0 client credentials flow with token caching and refresh logic.

Version: 1.0.0
Phase: 4 (Weeks 19-22)
Date: 2025-11-06
"""

import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
from threading import Lock

from .config import OAuth2Config, SAPEnvironment
from .exceptions import SAPAuthenticationError, SAPConnectionError
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class TokenCache:
    """
    Thread-safe in-memory token cache.

    In production, this would be replaced with Redis integration
    for distributed caching across multiple instances.
    """

    def __init__(self):
        """Initialize token cache."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        """
        Get token from cache if valid.

        Args:
            key: Cache key (typically environment name)

        Returns:
            Access token if valid, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                return None

            cached_data = self._cache[key]
            expires_at = cached_data.get("expires_at")

            # Check if token is still valid (with 60s buffer)
            if expires_at and DeterministicClock.now() < expires_at - timedelta(seconds=60):
                logger.debug(f"Token cache hit for key: {key}")
                return cached_data.get("access_token")

            # Token expired, remove from cache
            logger.debug(f"Token cache miss (expired) for key: {key}")
            del self._cache[key]
            return None

    def set(self, key: str, access_token: str, expires_in: int):
        """
        Store token in cache.

        Args:
            key: Cache key (typically environment name)
            access_token: OAuth access token
            expires_in: Token expiration time in seconds
        """
        with self._lock:
            expires_at = DeterministicClock.now() + timedelta(seconds=expires_in)
            self._cache[key] = {
                "access_token": access_token,
                "expires_at": expires_at,
                "cached_at": DeterministicClock.now()
            }
            logger.debug(f"Token cached for key: {key}, expires at: {expires_at}")

    def invalidate(self, key: str):
        """
        Invalidate cached token.

        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Token cache invalidated for key: {key}")

    def clear(self):
        """Clear all cached tokens."""
        with self._lock:
            self._cache.clear()
            logger.debug("Token cache cleared")


class SAPAuthHandler:
    """
    OAuth 2.0 authentication handler for SAP S/4HANA.

    Handles token acquisition, caching, and refresh using OAuth 2.0
    client credentials flow.
    """

    def __init__(
        self,
        oauth_config: OAuth2Config,
        environment: SAPEnvironment = SAPEnvironment.SANDBOX,
        cache: Optional[TokenCache] = None
    ):
        """
        Initialize authentication handler.

        Args:
            oauth_config: OAuth 2.0 configuration
            environment: SAP environment
            cache: Token cache instance (creates new if not provided)
        """
        self.oauth_config = oauth_config
        self.environment = environment
        self.cache = cache or TokenCache()
        self._cache_key = f"sap_token_{environment.value}"

        logger.info(f"Initialized SAP auth handler for environment: {environment.value}")

    def get_access_token(self, force_refresh: bool = False) -> str:
        """
        Get valid access token, refreshing if necessary.

        Args:
            force_refresh: Force token refresh even if cached token is valid

        Returns:
            Valid OAuth access token

        Raises:
            SAPAuthenticationError: If token acquisition fails
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_token = self.cache.get(self._cache_key)
            if cached_token:
                return cached_token

        # Acquire new token
        logger.info(f"Acquiring new access token for environment: {self.environment.value}")
        return self._acquire_token()

    def _acquire_token(self) -> str:
        """
        Acquire new access token from OAuth server.

        Returns:
            Access token

        Raises:
            SAPAuthenticationError: If token acquisition fails
        """
        try:
            # Prepare token request
            token_data = {
                "grant_type": self.oauth_config.grant_type,
                "client_id": self.oauth_config.client_id,
                "client_secret": self.oauth_config.client_secret,
                "scope": self.oauth_config.scope
            }

            # Make token request
            start_time = time.time()
            response = requests.post(
                self.oauth_config.token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            elapsed_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Token request completed in {elapsed_ms:.2f}ms "
                f"(status: {response.status_code})"
            )

            # Handle error responses
            if response.status_code != 200:
                error_msg = self._parse_error_response(response)
                raise SAPAuthenticationError(
                    reason=f"Token request failed: {error_msg}",
                    original_exception=None
                )

            # Parse token response
            token_response = response.json()
            access_token = token_response.get("access_token")
            expires_in = token_response.get("expires_in", 3600)

            if not access_token:
                raise SAPAuthenticationError(
                    reason="No access_token in response"
                )

            # Cache the token
            self.cache.set(self._cache_key, access_token, expires_in)

            logger.info(
                f"Successfully acquired access token "
                f"(expires in {expires_in}s)"
            )

            return access_token

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during token acquisition: {e}")
            raise SAPConnectionError(
                endpoint=self.oauth_config.token_url,
                reason="Connection failed",
                original_exception=e
            )

        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout during token acquisition: {e}")
            raise SAPAuthenticationError(
                reason="Token request timed out",
                original_exception=e
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during token acquisition: {e}")
            raise SAPAuthenticationError(
                reason=f"Token request failed: {str(e)}",
                original_exception=e
            )

        except Exception as e:
            logger.error(f"Unexpected error during token acquisition: {e}")
            raise SAPAuthenticationError(
                reason=f"Unexpected error: {str(e)}",
                original_exception=e
            )

    def _parse_error_response(self, response: requests.Response) -> str:
        """
        Parse error response from OAuth server.

        Args:
            response: HTTP response from token endpoint

        Returns:
            Human-readable error message
        """
        try:
            error_data = response.json()
            error = error_data.get("error", "unknown_error")
            error_description = error_data.get("error_description", "No description provided")
            return f"{error}: {error_description}"
        except Exception:
            return f"HTTP {response.status_code}: {response.text[:200]}"

    def invalidate_token(self):
        """Invalidate cached token, forcing refresh on next request."""
        self.cache.invalidate(self._cache_key)
        logger.info(f"Invalidated token for environment: {self.environment.value}")

    def validate_token(self, token: str) -> bool:
        """
        Validate token by checking if it's cached and not expired.

        Note: This is a simple cache check. For proper validation,
        you would need to call SAP's token introspection endpoint.

        Args:
            token: Access token to validate

        Returns:
            True if token appears valid, False otherwise
        """
        cached_token = self.cache.get(self._cache_key)
        return cached_token == token if cached_token else False

    def get_auth_header(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get authorization header with valid token.

        Args:
            force_refresh: Force token refresh

        Returns:
            Dictionary with Authorization header

        Raises:
            SAPAuthenticationError: If token acquisition fails
        """
        token = self.get_access_token(force_refresh=force_refresh)
        return {"Authorization": f"Bearer {token}"}


# Global auth handler instance per environment
_auth_handlers: Dict[str, SAPAuthHandler] = {}
_auth_lock = Lock()


def get_auth_handler(
    oauth_config: OAuth2Config,
    environment: SAPEnvironment = SAPEnvironment.SANDBOX
) -> SAPAuthHandler:
    """
    Get or create auth handler for environment.

    This provides a singleton pattern per environment to share
    token cache across the application.

    Args:
        oauth_config: OAuth configuration
        environment: SAP environment

    Returns:
        SAPAuthHandler instance
    """
    global _auth_handlers

    cache_key = f"{environment.value}"

    with _auth_lock:
        if cache_key not in _auth_handlers:
            _auth_handlers[cache_key] = SAPAuthHandler(
                oauth_config=oauth_config,
                environment=environment
            )

        return _auth_handlers[cache_key]


def reset_auth_handlers():
    """
    Reset all auth handlers.

    Useful for testing or when configuration changes.
    """
    global _auth_handlers

    with _auth_lock:
        _auth_handlers.clear()
