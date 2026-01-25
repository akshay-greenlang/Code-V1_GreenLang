"""
DEPRECATED: Database Module

This module has been consolidated into greenlang.db.
This file now provides backward-compatible re-exports with deprecation warnings.

Please update your imports:
    OLD: from greenlang.database import DatabaseConnection
    NEW: from greenlang.db import get_session, get_engine
    OR:  from greenlang.db import Base

The consolidated greenlang.db module includes:
- Phase 4: Authentication models (User, Role, Permission, etc.)
- Phase 5: Connection pooling and query optimization
- Advanced database features and performance monitoring

This re-export will be removed in version 2.0.0.

Migration Guide:
    - DatabaseConnection -> use get_session() or get_engine() from greenlang.db
    - EmissionFactorModel -> import specific models from greenlang.db
    - ConnectionPool -> use DatabaseConnectionPool from greenlang.db
"""

import warnings

# Backward-compatible re-exports
from greenlang.db import (
    Base as BaseModel,
    get_engine,
    get_session,
    init_db,
    DatabaseConnectionPool as ConnectionPool,
)

# Issue deprecation warning on import
warnings.warn(
    "greenlang.database is deprecated. "
    "Import from greenlang.db instead. "
    "This compatibility layer will be removed in version 2.0.0.",
    DeprecationWarning,
    stacklevel=2
)


# Expose legacy names for compatibility
class DatabaseConnection:
    """Deprecated: Use get_session() or get_engine() from greenlang.db instead."""
    def __init__(self):
        warnings.warn(
            "DatabaseConnection is deprecated. Use get_session() or get_engine() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.session = get_session()
        self.engine = get_engine()


class ConnectionConfig:
    """Deprecated: Configuration is now handled through greenlang.config module."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ConnectionConfig is deprecated. Use greenlang.config instead.",
            DeprecationWarning,
            stacklevel=2
        )


class DatabaseError(Exception):
    """Database operation error - kept for backward compatibility."""
    pass


# Placeholder models - recommend migrating to greenlang.db models
class EmissionFactorModel:
    """Deprecated: Define your models using Base from greenlang.db"""
    pass


class ActivityDataModel:
    """Deprecated: Define your models using Base from greenlang.db"""
    pass


class SupplierModel:
    """Deprecated: Define your models using Base from greenlang.db"""
    pass


class AuditLogModel:
    """Deprecated: Use AuditLog from greenlang.db.models_auth"""
    pass


__all__ = [
    # Connection classes (deprecated)
    'DatabaseConnection',
    'ConnectionConfig',
    'ConnectionPool',
    'DatabaseError',
    # Model classes (deprecated)
    'BaseModel',
    'EmissionFactorModel',
    'ActivityDataModel',
    'SupplierModel',
    'AuditLogModel',
]

__version__ = '1.0.0-deprecated'
