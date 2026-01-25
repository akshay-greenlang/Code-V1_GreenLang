"""
GreenLang Process Heat GraphQL API

Complete GraphQL API layer for all 143 Process Heat agents.
Provides modern integration capabilities including:
- Queries for agents, calculations, and registry statistics
- Mutations for agent execution and configuration
- Subscriptions for real-time events and progress
- JWT/API key authentication
- Multi-tenant isolation
- Full audit trails for compliance

Directory Structure:
    graphql/
    |-- __init__.py          # This file - main exports
    |-- schema.py            # Main GraphQL schema
    |-- types/               # GraphQL type definitions
    |   |-- __init__.py
    |   |-- agent.py         # Agent types
    |   |-- calculation.py   # Calculation result types
    |   +-- events.py        # Event types for subscriptions
    |-- resolvers/           # Query/mutation resolvers
    |   |-- __init__.py
    |   |-- agents.py        # Agent resolvers
    |   +-- subscriptions.py # Subscription resolvers
    +-- middleware/          # Middleware components
        |-- __init__.py
        |-- auth.py          # Authentication middleware
        +-- logging.py       # Logging middleware

Usage:
    from fastapi import FastAPI
    from app.graphql import create_graphql_app

    app = FastAPI()
    graphql_app = create_graphql_app()
    app.include_router(graphql_app, prefix="/graphql")

GraphQL Endpoints:
    POST /graphql         - GraphQL queries and mutations
    GET  /graphql         - GraphiQL IDE
    WS   /graphql         - WebSocket subscriptions

Example Queries:
    # Get a specific agent
    query {
        agent(id: "GL-022") {
            id
            name
            category
            type
            status
            healthScore
        }
    }

    # List agents by category
    query {
        agents(category: "Steam Systems", first: 10) {
            edges {
                node { id, name, status }
            }
            pageInfo { hasNextPage, totalCount }
        }
    }

    # Execute an agent
    mutation {
        runAgent(id: "GL-022", input: {steamPressure: 150}) {
            success
            calculation { id, status, result { value, unit } }
        }
    }

    # Subscribe to events
    subscription {
        agentEvents(agentId: "GL-022") {
            eventType
            timestamp
            message
        }
    }
"""

# Schema exports
from app.graphql.schema import (
    schema,
    create_graphql_app,
    export_schema_sdl,
    get_schema,
    graphql_health_check,
    Query,
    Mutation,
    Subscription,
    GreenLangContext,
    get_context,
)

# Type exports
from app.graphql.types import (
    # Agent types
    ProcessHeatAgentType,
    AgentInfoType,
    AgentStatusEnum,
    AgentCategoryEnum,
    AgentTypeEnum,
    HealthStatusType,
    AgentMetricsType,
    AgentConfigType,
    AgentConnection,
    AgentEdge,
    # Calculation types
    CalculationResultType,
    CalculationStatusEnum,
    CalculationInputType,
    CalculationOutputType,
    EmissionFactorType,
    ProvenanceType,
    QualityScoreType,
    # Event types
    AgentEventType,
    EventTypeEnum,
    ProgressType,
    ExecutionEventType,
    SystemEventType,
    CalculationProgressType,
)

# Resolver exports
from app.graphql.resolvers import (
    AgentResolver,
    SubscriptionResolver,
    get_agent,
    get_agents,
    run_agent,
    configure_agent,
)

# Middleware exports
from app.graphql.middleware import (
    AuthMiddleware,
    AuthContext,
    Permission,
    Role,
    get_context_with_auth,
    require_permission,
    require_role,
    LoggingMiddleware,
    GraphQLMetrics,
)

__all__ = [
    # Schema
    "schema",
    "create_graphql_app",
    "export_schema_sdl",
    "get_schema",
    "graphql_health_check",
    "Query",
    "Mutation",
    "Subscription",
    "GreenLangContext",
    "get_context",
    # Agent types
    "ProcessHeatAgentType",
    "AgentInfoType",
    "AgentStatusEnum",
    "AgentCategoryEnum",
    "AgentTypeEnum",
    "HealthStatusType",
    "AgentMetricsType",
    "AgentConfigType",
    "AgentConnection",
    "AgentEdge",
    # Calculation types
    "CalculationResultType",
    "CalculationStatusEnum",
    "CalculationInputType",
    "CalculationOutputType",
    "EmissionFactorType",
    "ProvenanceType",
    "QualityScoreType",
    # Event types
    "AgentEventType",
    "EventTypeEnum",
    "ProgressType",
    "ExecutionEventType",
    "SystemEventType",
    "CalculationProgressType",
    # Resolvers
    "AgentResolver",
    "SubscriptionResolver",
    "get_agent",
    "get_agents",
    "run_agent",
    "configure_agent",
    # Middleware
    "AuthMiddleware",
    "AuthContext",
    "Permission",
    "Role",
    "get_context_with_auth",
    "require_permission",
    "require_role",
    "LoggingMiddleware",
    "GraphQLMetrics",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
