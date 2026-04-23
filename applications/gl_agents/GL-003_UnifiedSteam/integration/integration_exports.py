"""
GL-003 UNIFIEDSTEAM - Integration Module Exports

This module provides convenient imports for the GraphQL connector:

Usage:
    from integration.integration_exports import (
        GraphQLConnector,
        GraphQLConfig,
        GraphQLResult,
        create_graphql_connector,
        create_oauth2_graphql_connector,
    )
"""

from .graphql_connector import (
    GraphQLConnector,
    GraphQLConfig,
    GraphQLResult,
    OAuth2Config,
    AuthType,
    RetryConfig as GraphQLRetryConfig,
    RetryStrategy as GraphQLRetryStrategy,
    CircuitBreakerConfig as GraphQLCircuitBreakerConfig,
    CircuitBreaker as GraphQLCircuitBreaker,
    CircuitBreakerState as GraphQLCircuitBreakerState,
    CacheConfig,
    CacheEntry,
    QueryCache,
    ConnectorMetrics as GraphQLConnectorMetrics,
    create_graphql_connector,
    create_oauth2_graphql_connector,
)

__all__ = [
    # GraphQL Connector
    "GraphQLConnector",
    "GraphQLConfig",
    "GraphQLResult",
    "OAuth2Config",
    "AuthType",
    "GraphQLRetryConfig",
    "GraphQLRetryStrategy",
    "GraphQLCircuitBreakerConfig",
    "GraphQLCircuitBreaker",
    "GraphQLCircuitBreakerState",
    "CacheConfig",
    "CacheEntry",
    "QueryCache",
    "GraphQLConnectorMetrics",
    "create_graphql_connector",
    "create_oauth2_graphql_connector",
]
