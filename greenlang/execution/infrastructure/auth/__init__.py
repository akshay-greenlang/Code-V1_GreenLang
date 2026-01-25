"""
GreenLang Authentication and Authorization Infrastructure
==========================================================

This module provides OAuth2/OIDC authentication and RBAC authorization
for the GreenLang Process Heat Agents platform.

Components:
    - OAuth2Provider: OAuth2/OIDC token validation and introspection
    - RBACManager: Role-based access control for agent permissions
    - SecurityMiddleware: FastAPI security dependencies

OWASP Compliance:
    - A02:2021 Cryptographic Failures: Secure token validation
    - A07:2021 Authentication Failures: Keycloak integration
    - A01:2021 Broken Access Control: RBAC enforcement

Example:
    >>> from greenlang.infrastructure.auth import OAuth2Provider, RBACManager
    >>> oauth2 = OAuth2Provider(config)
    >>> rbac = RBACManager(oauth2)
    >>>
    >>> # Validate token and check permissions
    >>> user = await oauth2.validate_token(token)
    >>> if rbac.can_execute_agent(user, "gl-010"):
    ...     result = await agent.execute()

Author: Security Team
Created: 2025-12-06
TASK-152: Implement OAuth2/OIDC (Keycloak) for Process Heat Agents
"""

from greenlang.infrastructure.auth.oauth2_provider import (
    OAuth2Provider,
    OAuth2ProviderConfig,
    TokenInfo,
    UserInfo,
    OAuth2Error,
    TokenValidationError,
    TokenExpiredError,
    InsufficientScopeError,
)
from greenlang.infrastructure.auth.rbac_manager import (
    RBACManager,
    RBACConfig,
    Permission,
    Role,
    AuthorizationDecision,
    AuthorizationError,
)

__all__ = [
    # OAuth2 Provider
    "OAuth2Provider",
    "OAuth2ProviderConfig",
    "TokenInfo",
    "UserInfo",
    "OAuth2Error",
    "TokenValidationError",
    "TokenExpiredError",
    "InsufficientScopeError",
    # RBAC Manager
    "RBACManager",
    "RBACConfig",
    "Permission",
    "Role",
    "AuthorizationDecision",
    "AuthorizationError",
]

__version__ = "1.0.0"
