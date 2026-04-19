"""
GreenLang Authentication Backend Modules

This package contains backend implementations for the GreenLang authentication system,
including PostgreSQL storage backend and future support for other databases.

Author: GreenLang Framework Team
Date: November 2025
"""

from .postgresql import (
    PostgreSQLBackend,
    DatabaseSession,
    get_db_session,
    init_database
)

__all__ = [
    'PostgreSQLBackend',
    'DatabaseSession',
    'get_db_session',
    'init_database'
]