"""
GreenLang Database Module
=========================

Database connection and model management for GreenLang.

This module provides database connectivity, ORM models, and query utilities
for GreenLang applications.

Example:
    >>> from greenlang.database import DatabaseConnection, EmissionFactorModel
    >>> db = DatabaseConnection()
    >>> factors = db.query(EmissionFactorModel).all()
"""

from greenlang.database.connection import (
    DatabaseConnection,
    ConnectionConfig,
    ConnectionPool,
    DatabaseError,
)
from greenlang.database.models import (
    BaseModel,
    EmissionFactorModel,
    ActivityDataModel,
    SupplierModel,
    AuditLogModel,
)

__all__ = [
    # Connection classes
    'DatabaseConnection',
    'ConnectionConfig',
    'ConnectionPool',
    'DatabaseError',
    # Model classes
    'BaseModel',
    'EmissionFactorModel',
    'ActivityDataModel',
    'SupplierModel',
    'AuditLogModel',
]

__version__ = '1.0.0'