"""
GreenLang GraphQL Middleware Package

This package contains middleware for the GraphQL API.

Middleware:
- Authentication: JWT-based auth and authorization
- Logging: Request/response logging and audit trails

Usage:
    from app.graphql.middleware import (
        AuthMiddleware,
        LoggingMiddleware,
        get_context_with_auth,
    )
"""

from app.graphql.middleware.auth import (
    AuthMiddleware,
    AuthContext,
    Permission,
    Role,
    get_context_with_auth,
    require_permission,
    require_role,
    verify_jwt_token,
    create_jwt_token,
)

from app.graphql.middleware.logging import (
    LoggingMiddleware,
    RequestLogger,
    AuditLogger,
    GraphQLMetrics,
)

__all__ = [
    # Auth middleware
    "AuthMiddleware",
    "AuthContext",
    "Permission",
    "Role",
    "get_context_with_auth",
    "require_permission",
    "require_role",
    "verify_jwt_token",
    "create_jwt_token",
    # Logging middleware
    "LoggingMiddleware",
    "RequestLogger",
    "AuditLogger",
    "GraphQLMetrics",
]
