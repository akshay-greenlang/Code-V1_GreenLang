"""
Database package for GreenLang Agent Registry.

Provides SQLAlchemy models and async database client.
"""

from greenlang_registry.db.client import (
    DatabaseClient,
    get_database_client,
    init_database,
    close_database,
)
from greenlang_registry.db.models import (
    Base,
    Agent,
    AgentVersion,
    EvaluationResult,
    StateTransition,
    UsageMetric,
    AuditLog,
    GovernancePolicy,
    LifecycleState,
)

__all__ = [
    # Client
    "DatabaseClient",
    "get_database_client",
    "init_database",
    "close_database",
    # Models
    "Base",
    "Agent",
    "AgentVersion",
    "EvaluationResult",
    "StateTransition",
    "UsageMetric",
    "AuditLog",
    "GovernancePolicy",
    "LifecycleState",
]
