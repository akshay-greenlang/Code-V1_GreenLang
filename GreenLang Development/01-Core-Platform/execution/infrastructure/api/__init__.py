"""
GreenLang Infrastructure - API Framework

This module provides API framework components for GreenLang services,
including REST, GraphQL, gRPC, webhooks, SSE, and persistent storage.

All components follow:
- OpenAPI 3.0 specifications
- Rate limiting (token bucket, sliding window, fixed window, leaky bucket)
- Authentication/Authorization
- Versioning
- Monitoring
- Pluggable storage backends (InMemory, SQLite, PostgreSQL)
"""

from greenlang.infrastructure.api.openapi_generator import (
    OpenAPIGenerator,
    OpenAPIGeneratorConfig,
    OpenAPIVersion,
    SecuritySchemeType,
    ParameterLocation,
    create_greenlang_openapi_generator,
    generate_openapi_from_app,
)
from greenlang.infrastructure.api.rest_router import RESTRouter, APIVersion
from greenlang.infrastructure.api.graphql_schema import GraphQLSchemaBuilder
from greenlang.infrastructure.api.grpc_services import GRPCServiceRegistry
from greenlang.infrastructure.api.webhooks import WebhookManager
from greenlang.infrastructure.api.sse_stream import SSEManager

# Domain-specific REST routes
try:
    from greenlang.infrastructure.api.routes import (
        emissions_router,
        agents_router,
        jobs_router,
        compliance_router,
    )
    _ROUTES_AVAILABLE = True
except ImportError:
    emissions_router = None
    agents_router = None
    jobs_router = None
    compliance_router = None
    _ROUTES_AVAILABLE = False
from greenlang.infrastructure.api.rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    RateLimitStrategy,
    RateLimitScope,
    RateLimitInfo,
    TokenBucket,
    SlidingWindowCounter,
    FixedWindowCounter,
    LeakyBucket,
    RateLimitMiddleware,
    rate_limit,
)

# Storage layer - always available
from greenlang.infrastructure.api.storage import (
    StorageFactory,
    # Webhook stores
    InMemoryWebhookStore,
    SQLiteWebhookStore,
    PostgresWebhookStore,
    # Agent state stores
    InMemoryAgentStateStore,
    SQLiteAgentStateStore,
    PostgresAgentStateStore,
    # Convenience functions
    get_default_webhook_store,
    get_default_agent_store,
    initialize_stores,
    close_stores,
)

# Redis rate limiter - imported separately to handle optional dependency
try:
    from greenlang.infrastructure.api.rate_limiter_redis import (
        RedisRateLimiter,
        RedisRateLimiterConfig,
        ManagedRedisRateLimiter,
        create_rate_limiter,
    )
    _REDIS_EXPORTS = [
        "RedisRateLimiter",
        "RedisRateLimiterConfig",
        "ManagedRedisRateLimiter",
        "create_rate_limiter",
    ]
except ImportError:
    _REDIS_EXPORTS = []

__all__ = [
    # OpenAPI Generator
    "OpenAPIGenerator",
    "OpenAPIGeneratorConfig",
    "OpenAPIVersion",
    "SecuritySchemeType",
    "ParameterLocation",
    "create_greenlang_openapi_generator",
    "generate_openapi_from_app",
    # REST Router
    "RESTRouter",
    "APIVersion",
    "GraphQLSchemaBuilder",
    "GRPCServiceRegistry",
    "WebhookManager",
    "SSEManager",
    # Rate limiting
    "RateLimiter",
    "RateLimiterConfig",
    "RateLimitStrategy",
    "RateLimitScope",
    "RateLimitInfo",
    "TokenBucket",
    "SlidingWindowCounter",
    "FixedWindowCounter",
    "LeakyBucket",
    "RateLimitMiddleware",
    "rate_limit",
    # Storage layer
    "StorageFactory",
    "InMemoryWebhookStore",
    "SQLiteWebhookStore",
    "PostgresWebhookStore",
    "InMemoryAgentStateStore",
    "SQLiteAgentStateStore",
    "PostgresAgentStateStore",
    "get_default_webhook_store",
    "get_default_agent_store",
    "initialize_stores",
    "close_stores",
] + _REDIS_EXPORTS + (
    [
        "emissions_router",
        "agents_router",
        "jobs_router",
        "compliance_router",
    ] if _ROUTES_AVAILABLE else []
)
