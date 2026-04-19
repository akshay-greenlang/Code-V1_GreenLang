# -*- coding: utf-8 -*-
"""
Database module for GreenLang

Provides SQLAlchemy models, database utilities, and performance optimization.

Phase 4: Authentication models
Phase 5: Connection pooling, query optimization, and performance indexes

Author: GreenLang Infrastructure Team
Version: 5.0.0
"""

from greenlang.db.base import Base, get_engine, get_session, init_db
from greenlang.db.models_auth import (
    User,
    Role,
    Permission,
    UserRole,
    Session,
    APIKey,
    AuditLog,
    SAMLProvider,
    OAuthProvider,
    LDAPConfig,
)

# Phase 5: Database optimization
from greenlang.db.connection import (
    DatabaseConnectionPool,
    ConnectionState,
    ConnectionMetrics,
    get_connection_pool,
    initialize_connection_pool,
)

from greenlang.db.query_optimizer import (
    QueryOptimizer,
    QueryMetrics,
    SlowQuery,
    QueryPlan,
    QueryCache,
    QueryTracker,
    get_query_optimizer,
    initialize_query_optimizer,
)

__all__ = [
    # Base
    "Base",
    "get_engine",
    "get_session",
    "init_db",

    # Auth Models
    "User",
    "Role",
    "Permission",
    "UserRole",
    "Session",
    "APIKey",
    "AuditLog",
    "SAMLProvider",
    "OAuthProvider",
    "LDAPConfig",

    # Connection Pooling (Phase 5)
    "DatabaseConnectionPool",
    "ConnectionState",
    "ConnectionMetrics",
    "get_connection_pool",
    "initialize_connection_pool",

    # Query Optimization (Phase 5)
    "QueryOptimizer",
    "QueryMetrics",
    "SlowQuery",
    "QueryPlan",
    "QueryCache",
    "QueryTracker",
    "get_query_optimizer",
    "initialize_query_optimizer",
]

__version__ = "5.0.0"
