"""
API layer for Review Console Backend.

This module provides FastAPI routes and Pydantic models for the
review queue REST API.

Components:
    - routes: FastAPI route handlers for all API endpoints
    - models: Pydantic request/response models with validation

Example:
    >>> from review_console.api.routes import router
    >>> app.include_router(router, prefix="/api")
"""

from review_console.api.routes import router
from review_console.api.models import (
    # Request models
    ResolveItemRequest,
    RejectItemRequest,
    VocabularySuggestionRequest,
    QueueFilterParams,
    # Response models
    ReviewQueueItemResponse,
    ReviewQueueItemDetailResponse,
    ResolutionResponse,
    QueueStatsResponse,
    VocabularySuggestionResponse,
    PaginatedResponse,
    HealthResponse,
    # Shared models
    CandidateInfo,
    ContextInfo,
)

__all__ = [
    # Routes
    "router",
    # Request models
    "ResolveItemRequest",
    "RejectItemRequest",
    "VocabularySuggestionRequest",
    "QueueFilterParams",
    # Response models
    "ReviewQueueItemResponse",
    "ReviewQueueItemDetailResponse",
    "ResolutionResponse",
    "QueueStatsResponse",
    "VocabularySuggestionResponse",
    "PaginatedResponse",
    "HealthResponse",
    # Shared models
    "CandidateInfo",
    "ContextInfo",
]
