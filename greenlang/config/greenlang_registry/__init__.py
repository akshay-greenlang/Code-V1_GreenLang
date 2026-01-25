"""
GreenLang Agent Registry

The centralized metadata repository that treats agents as first-class,
versioned, governed assets within the GreenLang Agent Factory.

Key Features:
- Immutable agent versioning with semantic versioning (SemVer)
- Multi-tenant governance and access control
- Lifecycle state management (draft -> experimental -> certified -> deprecated)
- Rich metadata and capability indexing
- Integration with evaluation results and quality metrics
- API-first design for programmatic access

Example:
    >>> from greenlang_registry import create_app
    >>> app = create_app()
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"

from greenlang_registry.db.client import (
    DatabaseClient,
    get_database_client,
    init_database,
    close_database,
)
from greenlang_registry.db.models import (
    Agent,
    AgentVersion,
    EvaluationResult,
    StateTransition,
    UsageMetric,
    AuditLog,
    GovernancePolicy,
    LifecycleState,
)
from greenlang_registry.clients import (
    OCIManifest,
    OCIDescriptor,
    OCIAuth,
    OCIClient,
)

__all__ = [
    # Version
    "__version__",
    # Database
    "DatabaseClient",
    "get_database_client",
    "init_database",
    "close_database",
    # Models
    "Agent",
    "AgentVersion",
    "EvaluationResult",
    "StateTransition",
    "UsageMetric",
    "AuditLog",
    "GovernancePolicy",
    "LifecycleState",
    # OCI Client
    "OCIManifest",
    "OCIDescriptor",
    "OCIAuth",
    "OCIClient",
]
