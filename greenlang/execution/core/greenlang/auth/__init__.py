# -*- coding: utf-8 -*-
"""
GreenLang Core Authentication Module

Enterprise-grade authentication infrastructure including:
- JWT token management with RS256 signing
- API key management with SHA-256 hashing
- FastAPI authentication middleware
"""

from .jwt_handler import (
    JWTHandler,
    JWTConfig,
    JWTClaims,
    JWTError,
    TokenExpiredError,
    InvalidTokenError,
    InvalidSignatureError,
)

from .api_key_manager import (
    APIKeyManager,
    APIKeyConfig,
    APIKeyRecord,
    APIKeyError,
    InvalidAPIKeyError,
    ExpiredAPIKeyError,
)

from .middleware import (
    AuthenticationMiddleware,
    JWTAuthBackend,
    APIKeyAuthBackend,
    AuthContext,
    get_current_user,
    require_auth,
    require_roles,
    require_permissions,
)

__all__ = [
    # JWT
    "JWTHandler",
    "JWTConfig",
    "JWTClaims",
    "JWTError",
    "TokenExpiredError",
    "InvalidTokenError",
    "InvalidSignatureError",
    # API Key
    "APIKeyManager",
    "APIKeyConfig",
    "APIKeyRecord",
    "APIKeyError",
    "InvalidAPIKeyError",
    "ExpiredAPIKeyError",
    # Middleware
    "AuthenticationMiddleware",
    "JWTAuthBackend",
    "APIKeyAuthBackend",
    "AuthContext",
    "get_current_user",
    "require_auth",
    "require_roles",
    "require_permissions",
]
