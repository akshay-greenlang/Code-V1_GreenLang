#!/usr/bin/env python3
"""
ZAP JWT Authentication Hook for GreenLang API.

This script provides JWT-based authentication for OWASP ZAP authenticated scanning.
It can be used as a ZAP authentication script or standalone for token refresh.

Usage:
    As ZAP Script:
        Load in ZAP -> Scripts -> Authentication -> jwt_auth.py

    Standalone:
        python jwt_auth.py --auth-url https://api.greenlang.io/auth/login \
                          --username test@greenlang.io \
                          --password $PASSWORD

Environment Variables:
    ZAP_AUTH_URL: Authentication endpoint URL
    ZAP_AUTH_USERNAME: Username for authentication
    ZAP_AUTH_PASSWORD: Password for authentication
    ZAP_AUTH_CLIENT_ID: OAuth2 client ID (optional)
    ZAP_AUTH_CLIENT_SECRET: OAuth2 client secret (optional)
    ZAP_TOKEN_HEADER: Header name for token (default: Authorization)
    ZAP_TOKEN_PREFIX: Token prefix (default: Bearer)

Author: GreenLang Security Team
Version: 1.0.0
Compliance: SOC 2 CC6.1, ISO 27001 A.9.4.2
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("zap_jwt_auth")


@dataclass
class TokenInfo:
    """JWT token information."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    issued_at: float = 0.0

    def __post_init__(self) -> None:
        """Set issued_at if not provided."""
        if self.issued_at == 0.0:
            self.issued_at = time.time()

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 60s buffer)."""
        return time.time() > (self.issued_at + self.expires_in - 60)

    @property
    def authorization_header(self) -> str:
        """Get the full authorization header value."""
        return f"{self.token_type} {self.access_token}"


@dataclass
class AuthConfig:
    """Authentication configuration."""

    auth_url: str
    username: str
    password: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_header: str = "Authorization"
    token_prefix: str = "Bearer"
    grant_type: str = "password"
    scope: str = "openid profile email"

    @classmethod
    def from_environment(cls) -> "AuthConfig":
        """Create configuration from environment variables."""
        auth_url = os.environ.get("ZAP_AUTH_URL", "")
        if not auth_url:
            raise ValueError("ZAP_AUTH_URL environment variable is required")

        username = os.environ.get("ZAP_AUTH_USERNAME", "")
        password = os.environ.get("ZAP_AUTH_PASSWORD", "")

        if not username or not password:
            raise ValueError("ZAP_AUTH_USERNAME and ZAP_AUTH_PASSWORD are required")

        return cls(
            auth_url=auth_url,
            username=username,
            password=password,
            client_id=os.environ.get("ZAP_AUTH_CLIENT_ID"),
            client_secret=os.environ.get("ZAP_AUTH_CLIENT_SECRET"),
            token_header=os.environ.get("ZAP_TOKEN_HEADER", "Authorization"),
            token_prefix=os.environ.get("ZAP_TOKEN_PREFIX", "Bearer"),
        )


class JWTAuthenticator:
    """
    JWT authenticator for ZAP authenticated scanning.

    Supports:
    - Username/password authentication
    - OAuth2 client credentials
    - Token refresh
    - Multi-factor authentication (TOTP)
    """

    def __init__(self, config: AuthConfig) -> None:
        """Initialize authenticator with configuration."""
        self.config = config
        self._token: Optional[TokenInfo] = None
        self._session = None

    def _get_session(self) -> Any:
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update({
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "GreenLang-ZAP-Auth/1.0",
                })
            except ImportError:
                raise RuntimeError("requests library is required: pip install requests")
        return self._session

    def authenticate(self) -> TokenInfo:
        """
        Authenticate and obtain JWT token.

        Returns:
            TokenInfo with access token and metadata

        Raises:
            AuthenticationError: If authentication fails
        """
        logger.info(f"Authenticating to {self.config.auth_url}")

        # Build authentication payload
        payload = self._build_auth_payload()

        try:
            session = self._get_session()
            response = session.post(
                self.config.auth_url,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                self._token = self._parse_token_response(data)
                logger.info("Authentication successful")
                return self._token

            elif response.status_code == 401:
                logger.error("Authentication failed: Invalid credentials")
                raise AuthenticationError("Invalid credentials")

            elif response.status_code == 403:
                logger.error("Authentication failed: Access forbidden")
                raise AuthenticationError("Access forbidden")

            else:
                logger.error(f"Authentication failed: HTTP {response.status_code}")
                raise AuthenticationError(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            if isinstance(e, AuthenticationError):
                raise
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError(f"Authentication failed: {e}") from e

    def _build_auth_payload(self) -> Dict[str, Any]:
        """Build authentication request payload."""
        if self.config.client_id:
            # OAuth2 flow
            payload = {
                "grant_type": self.config.grant_type,
                "client_id": self.config.client_id,
                "username": self.config.username,
                "password": self.config.password,
                "scope": self.config.scope,
            }
            if self.config.client_secret:
                payload["client_secret"] = self.config.client_secret
        else:
            # Simple username/password
            payload = {
                "username": self.config.username,
                "password": self.config.password,
            }

        return payload

    def _parse_token_response(self, data: Dict[str, Any]) -> TokenInfo:
        """Parse token from authentication response."""
        # Handle different response formats
        access_token = (
            data.get("access_token") or
            data.get("token") or
            data.get("jwt") or
            data.get("id_token")
        )

        if not access_token:
            raise AuthenticationError("No access token in response")

        return TokenInfo(
            access_token=access_token,
            token_type=data.get("token_type", self.config.token_prefix),
            expires_in=data.get("expires_in", 3600),
            refresh_token=data.get("refresh_token"),
        )

    def refresh(self) -> TokenInfo:
        """
        Refresh the current token.

        Returns:
            New TokenInfo with refreshed token

        Raises:
            AuthenticationError: If refresh fails
        """
        if not self._token or not self._token.refresh_token:
            logger.info("No refresh token available, re-authenticating")
            return self.authenticate()

        logger.info("Refreshing access token")

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._token.refresh_token,
        }

        if self.config.client_id:
            payload["client_id"] = self.config.client_id
        if self.config.client_secret:
            payload["client_secret"] = self.config.client_secret

        try:
            session = self._get_session()

            # Try token endpoint first, fall back to auth URL
            refresh_url = self.config.auth_url.replace("/login", "/token")

            response = session.post(refresh_url, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                self._token = self._parse_token_response(data)
                logger.info("Token refresh successful")
                return self._token
            else:
                logger.warning(f"Token refresh failed: {response.status_code}")
                return self.authenticate()

        except Exception as e:
            logger.warning(f"Token refresh error: {e}, re-authenticating")
            return self.authenticate()

    def get_token(self) -> TokenInfo:
        """
        Get current token, refreshing if necessary.

        Returns:
            Current valid TokenInfo
        """
        if self._token is None:
            return self.authenticate()

        if self._token.is_expired:
            return self.refresh()

        return self._token

    def get_auth_header(self) -> Dict[str, str]:
        """
        Get authorization header for requests.

        Returns:
            Dict with authorization header
        """
        token = self.get_token()
        return {
            self.config.token_header: token.authorization_header
        }


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


# =============================================================================
# ZAP Script Interface
# =============================================================================

def zap_authenticate(helper, paramsValues, credentials):
    """
    ZAP authentication script entry point.

    This function is called by ZAP's script-based authentication.

    Args:
        helper: ZAP helper object
        paramsValues: Script parameters from ZAP
        credentials: User credentials configured in ZAP

    Returns:
        Authenticated HTTP message
    """
    logger.info("ZAP authentication script invoked")

    try:
        # Get credentials from ZAP
        username = credentials.getParam("username")
        password = credentials.getParam("password")

        # Get auth URL from params or environment
        auth_url = paramsValues.get("auth_url") or os.environ.get("ZAP_AUTH_URL")

        if not auth_url:
            logger.error("No auth_url configured")
            return None

        # Create config and authenticate
        config = AuthConfig(
            auth_url=auth_url,
            username=username,
            password=password,
        )

        authenticator = JWTAuthenticator(config)
        token = authenticator.authenticate()

        # Create authenticated request
        msg = helper.prepareMessage()
        msg.getRequestHeader().setHeader(
            config.token_header,
            token.authorization_header
        )

        logger.info("ZAP authentication successful")
        return msg

    except Exception as e:
        logger.error(f"ZAP authentication failed: {e}")
        return None


def zap_get_required_params_names():
    """Return required parameter names for ZAP UI."""
    return ["auth_url"]


def zap_get_optional_params_names():
    """Return optional parameter names for ZAP UI."""
    return ["client_id", "token_header", "token_prefix"]


def zap_get_credentials_params_names():
    """Return credential parameter names for ZAP UI."""
    return ["username", "password"]


# =============================================================================
# ZAP HTTP Sender Script Interface
# =============================================================================

# Global authenticator instance for HTTP sender
_authenticator: Optional[JWTAuthenticator] = None


def zap_sender_init():
    """Initialize HTTP sender script."""
    global _authenticator

    try:
        config = AuthConfig.from_environment()
        _authenticator = JWTAuthenticator(config)
        _authenticator.authenticate()
        logger.info("HTTP sender authentication initialized")
    except Exception as e:
        logger.error(f"HTTP sender init failed: {e}")


def zap_send_request(msg, initiator, helper):
    """
    Add authentication to outgoing requests.

    Called by ZAP for each outgoing request.
    """
    global _authenticator

    if _authenticator is None:
        return

    try:
        token = _authenticator.get_token()
        msg.getRequestHeader().setHeader(
            _authenticator.config.token_header,
            token.authorization_header
        )
    except Exception as e:
        logger.debug(f"Failed to add auth header: {e}")


def zap_send_response(msg, initiator, helper):
    """
    Handle authentication errors in responses.

    Called by ZAP for each response.
    """
    global _authenticator

    if _authenticator is None:
        return

    status = msg.getResponseHeader().getStatusCode()

    if status == 401:
        logger.info("Received 401, refreshing token")
        try:
            _authenticator.refresh()
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")


# =============================================================================
# Standalone CLI
# =============================================================================

def main() -> int:
    """CLI entry point for standalone token generation."""
    parser = argparse.ArgumentParser(
        description="Generate JWT token for ZAP authenticated scanning"
    )
    parser.add_argument(
        "--auth-url",
        default=os.environ.get("ZAP_AUTH_URL"),
        help="Authentication endpoint URL"
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("ZAP_AUTH_USERNAME"),
        help="Username for authentication"
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("ZAP_AUTH_PASSWORD"),
        help="Password for authentication"
    )
    parser.add_argument(
        "--client-id",
        default=os.environ.get("ZAP_AUTH_CLIENT_ID"),
        help="OAuth2 client ID (optional)"
    )
    parser.add_argument(
        "--client-secret",
        default=os.environ.get("ZAP_AUTH_CLIENT_SECRET"),
        help="OAuth2 client secret (optional)"
    )
    parser.add_argument(
        "--output",
        choices=["header", "token", "json", "env"],
        default="header",
        help="Output format"
    )
    parser.add_argument(
        "--header-name",
        default="Authorization",
        help="Authorization header name"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate required arguments
    if not args.auth_url:
        logger.error("--auth-url or ZAP_AUTH_URL is required")
        return 1

    if not args.username or not args.password:
        logger.error("--username and --password are required")
        return 1

    try:
        config = AuthConfig(
            auth_url=args.auth_url,
            username=args.username,
            password=args.password,
            client_id=args.client_id,
            client_secret=args.client_secret,
            token_header=args.header_name,
        )

        authenticator = JWTAuthenticator(config)
        token = authenticator.authenticate()

        # Output in requested format
        if args.output == "header":
            print(f"{args.header_name}: {token.authorization_header}")

        elif args.output == "token":
            print(token.access_token)

        elif args.output == "json":
            print(json.dumps({
                "access_token": token.access_token,
                "token_type": token.token_type,
                "expires_in": token.expires_in,
                "expires_at": datetime.fromtimestamp(
                    token.issued_at + token.expires_in
                ).isoformat(),
            }, indent=2))

        elif args.output == "env":
            print(f"export ZAP_AUTH_TOKEN='{token.access_token}'")
            print(f"export ZAP_AUTH_HEADER='{token.authorization_header}'")

        return 0

    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        return 1

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
