"""
GL-006 HEATRECLAIM - API Module

REST and GraphQL API endpoints for heat recovery optimization.

Provides:
- REST endpoints for synchronous operations
- GraphQL schema for flexible queries
- Streaming endpoints for real-time updates
- Authentication and rate limiting
"""

from .rest_api import create_app, router
from .graphql_schema import schema, HENQuery, HENMutation
from .middleware import (
    RateLimitMiddleware,
    AuthenticationMiddleware,
    LoggingMiddleware,
)

__all__ = [
    "create_app",
    "router",
    "schema",
    "HENQuery",
    "HENMutation",
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "LoggingMiddleware",
]
