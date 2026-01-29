"""
Service layer for Review Console Backend.

This module provides business logic services for the review queue,
including queue management, resolution processing, and statistics.

Components:
    - queue: ReviewQueue service for managing review items
    - resolution: Resolution service for processing reviewer decisions

Example:
    >>> from review_console.services.queue import ReviewQueueService
    >>> from review_console.services.resolution import ResolutionService
    >>> queue_service = ReviewQueueService(db_session)
    >>> items = await queue_service.get_pending_items()
"""

from review_console.services.queue import ReviewQueueService
from review_console.services.resolution import ResolutionService

__all__ = [
    "ReviewQueueService",
    "ResolutionService",
]
