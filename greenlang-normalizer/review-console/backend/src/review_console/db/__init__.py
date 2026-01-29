"""
Database layer for Review Console Backend.

This module provides SQLAlchemy models and session management for the
review queue persistence layer.

Components:
    - models: SQLAlchemy ORM models for review queue items and resolutions
    - session: Async database session management with connection pooling

Example:
    >>> from review_console.db.session import get_db
    >>> from review_console.db.models import ReviewQueueItem
    >>> async with get_db() as db:
    ...     items = await db.execute(select(ReviewQueueItem))
"""

from review_console.db.models import (
    Base,
    ReviewQueueItem,
    Resolution,
    AuditLogEntry,
    VocabularySuggestion,
    ReviewStatus,
    AuditAction,
    SuggestionStatus,
)
from review_console.db.session import (
    get_db,
    get_async_session,
    async_engine,
    AsyncSessionLocal,
)

__all__ = [
    # Models
    "Base",
    "ReviewQueueItem",
    "Resolution",
    "AuditLogEntry",
    "VocabularySuggestion",
    # Enums
    "ReviewStatus",
    "AuditAction",
    "SuggestionStatus",
    # Session
    "get_db",
    "get_async_session",
    "async_engine",
    "AsyncSessionLocal",
]
