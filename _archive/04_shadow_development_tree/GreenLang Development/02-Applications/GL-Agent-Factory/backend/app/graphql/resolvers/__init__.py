"""
GreenLang GraphQL Resolvers Package

This package contains all GraphQL resolvers for the Process Heat agents API.

Resolvers:
- Agent resolvers: Queries and mutations for agents
- Subscription resolvers: Real-time event streaming

Usage:
    from app.graphql.resolvers import (
        AgentResolver,
        SubscriptionResolver,
    )
"""

from app.graphql.resolvers.agents import (
    AgentResolver,
    get_agent,
    get_agents,
    get_agent_health,
    run_agent,
    configure_agent,
    get_registry_stats,
    search_agents,
)

from app.graphql.resolvers.subscriptions import (
    SubscriptionResolver,
    agent_events_generator,
    calculation_progress_generator,
    system_events_generator,
)

__all__ = [
    # Agent resolvers
    "AgentResolver",
    "get_agent",
    "get_agents",
    "get_agent_health",
    "run_agent",
    "configure_agent",
    "get_registry_stats",
    "search_agents",
    # Subscription resolvers
    "SubscriptionResolver",
    "agent_events_generator",
    "calculation_progress_generator",
    "system_events_generator",
]
