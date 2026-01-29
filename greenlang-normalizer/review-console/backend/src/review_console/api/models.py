"""
Pydantic models for Review Console API.

This module defines all request and response models used by the Review Console
API endpoints. Models include comprehensive validation and documentation.

Model Categories:
    - Request models: Input validation for API endpoints
    - Response models: Structured API responses
    - Shared models: Reusable model components

Example:
    >>> from review_console.api.models import ResolveItemRequest
    >>> request = ResolveItemRequest(
    ...     selected_canonical_id="GL-FUEL-NATGAS",
    ...     reviewer_notes="Exact match confirmed"
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums
# ============================================================================


class ReviewStatusEnum(str, Enum):
    """Review status enum for API models."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    AUTO_RESOLVED = "auto_resolved"


class EntityTypeEnum(str, Enum):
    """Entity type enum for API models."""
    FUEL = "fuel"
    MATERIAL = "material"
    PROCESS = "process"
    ACTIVITY = "activity"
    EMISSION_FACTOR = "emission_factor"
    LOCATION = "location"


class SuggestionStatusEnum(str, Enum):
    """Vocabulary suggestion status enum for API models."""
    PENDING = "pending"
    PR_CREATED = "pr_created"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================================
# Shared Models
# ============================================================================


class CandidateInfo(BaseModel):
    """
    Information about a candidate match.

    Represents a potential canonical entity match with its confidence score
    and metadata.

    Attributes:
        id: Canonical entity ID
        name: Canonical entity name
        score: Confidence score (0.0-1.0)
        source: Source vocabulary/database
        match_method: How the match was found (exact, fuzzy, llm)
        metadata: Additional metadata about the candidate
    """
    id: str = Field(..., description="Canonical entity ID", min_length=1, max_length=256)
    name: str = Field(..., description="Canonical entity name", min_length=1, max_length=500)
    score: float = Field(..., description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    source: str = Field(..., description="Source vocabulary/database", min_length=1, max_length=128)
    match_method: Optional[str] = Field(None, description="How the match was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "GL-FUEL-NATGAS",
                "name": "Natural gas",
                "score": 0.92,
                "source": "fuels_vocab",
                "match_method": "fuzzy",
                "metadata": {"region": "global"}
            }
        }


class ContextInfo(BaseModel):
    """
    Context information for entity resolution.

    Provides additional context that was used during resolution,
    such as industry sector, region, or temporal scope.

    Attributes:
        industry_sector: Industry sector (e.g., "energy", "manufacturing")
        region: Geographic region (e.g., "US", "EU", "GLOBAL")
        temporal_scope: Time period (e.g., "2024", "2020-2025")
        source_field: Original field name in source data
        additional_hints: Additional hints from source data
    """
    industry_sector: Optional[str] = Field(None, description="Industry sector", max_length=64)
    region: Optional[str] = Field(None, description="Geographic region", max_length=64)
    temporal_scope: Optional[str] = Field(None, description="Time period", max_length=32)
    source_field: Optional[str] = Field(None, description="Original field name")
    additional_hints: Dict[str, Any] = Field(default_factory=dict, description="Additional hints")


# ============================================================================
# Request Models
# ============================================================================


class ResolveItemRequest(BaseModel):
    """
    Request to resolve a review queue item.

    Submits a resolution decision for an item, selecting a canonical entity
    and optionally providing reviewer notes.

    Attributes:
        selected_canonical_id: ID of the selected canonical entity
        selected_canonical_name: Name of the selected canonical entity (optional)
        reviewer_notes: Notes explaining the decision (optional)
        confidence_override: Optional confidence override (0.0-1.0)
    """
    selected_canonical_id: str = Field(
        ...,
        description="ID of the selected canonical entity",
        min_length=1,
        max_length=256,
    )
    selected_canonical_name: Optional[str] = Field(
        None,
        description="Name of the selected canonical entity",
        max_length=500,
    )
    reviewer_notes: Optional[str] = Field(
        None,
        description="Notes explaining the decision",
        max_length=2000,
    )
    confidence_override: Optional[float] = Field(
        None,
        description="Optional confidence override (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "selected_canonical_id": "GL-FUEL-NATGAS",
                "selected_canonical_name": "Natural gas",
                "reviewer_notes": "Exact match confirmed via industry reference",
                "confidence_override": 0.98
            }
        }


class RejectItemRequest(BaseModel):
    """
    Request to reject or escalate a review queue item.

    Marks an item as rejected (no valid match found) or escalates it
    to a senior reviewer.

    Attributes:
        reason: Reason for rejection/escalation
        escalate_to: User ID to escalate to (optional, triggers escalation)
        suggest_vocabulary: Whether to suggest a new vocabulary entry
    """
    reason: str = Field(
        ...,
        description="Reason for rejection or escalation",
        min_length=10,
        max_length=2000,
    )
    escalate_to: Optional[str] = Field(
        None,
        description="User ID to escalate to (triggers escalation)",
        max_length=256,
    )
    suggest_vocabulary: bool = Field(
        False,
        description="Whether to suggest a new vocabulary entry"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reason": "No matching entity found in current vocabulary. Input appears to be a regional variant of diesel fuel.",
                "escalate_to": "senior-reviewer@greenlang.io",
                "suggest_vocabulary": True
            }
        }


class VocabularySuggestionRequest(BaseModel):
    """
    Request to suggest a new vocabulary entry.

    Allows reviewers to suggest new canonical entities when no suitable
    match exists in the current vocabulary.

    Attributes:
        entity_type: Type of entity (fuel, material, process)
        canonical_name: Suggested canonical name
        aliases: List of suggested aliases
        source: Source/justification for the suggestion
        properties: Suggested properties for the entity
    """
    entity_type: EntityTypeEnum = Field(
        ...,
        description="Type of entity (fuel, material, process)"
    )
    canonical_name: str = Field(
        ...,
        description="Suggested canonical name",
        min_length=1,
        max_length=500,
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="List of suggested aliases",
        max_length=50,  # Max 50 aliases
    )
    source: str = Field(
        ...,
        description="Source/justification for the suggestion",
        min_length=10,
        max_length=2000,
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested properties for the entity"
    )

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, v: List[str]) -> List[str]:
        """Validate and normalize aliases."""
        # Remove duplicates and empty strings, normalize whitespace
        seen = set()
        result = []
        for alias in v:
            normalized = alias.strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                result.append(normalized)
        return result

    class Config:
        json_schema_extra = {
            "example": {
                "entity_type": "fuel",
                "canonical_name": "Sustainable Aviation Fuel (SAF)",
                "aliases": ["SAF", "Bio-jet fuel", "Renewable jet fuel"],
                "source": "ICAO CORSIA eligible fuels list, confirmed by customer (ABC Aviation)",
                "properties": {
                    "carbon_intensity": 0.3,
                    "certification": "RSB"
                }
            }
        }


class QueueFilterParams(BaseModel):
    """
    Filter parameters for queue listing.

    Allows filtering the review queue by various criteria.

    Attributes:
        entity_type: Filter by entity type
        org_id: Filter by organization ID
        status: Filter by review status
        date_from: Filter by created date (from)
        date_to: Filter by created date (to)
        min_confidence: Filter by minimum confidence
        max_confidence: Filter by maximum confidence
        assigned_to: Filter by assigned reviewer
        search: Search in input_text
    """
    entity_type: Optional[EntityTypeEnum] = Field(None, description="Filter by entity type")
    org_id: Optional[str] = Field(None, description="Filter by organization ID", max_length=100)
    status: Optional[ReviewStatusEnum] = Field(None, description="Filter by review status")
    date_from: Optional[datetime] = Field(None, description="Filter by created date (from)")
    date_to: Optional[datetime] = Field(None, description="Filter by created date (to)")
    min_confidence: Optional[float] = Field(None, description="Filter by min confidence", ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, description="Filter by max confidence", ge=0.0, le=1.0)
    assigned_to: Optional[str] = Field(None, description="Filter by assigned reviewer", max_length=256)
    search: Optional[str] = Field(None, description="Search in input_text", max_length=200)


# ============================================================================
# Response Models
# ============================================================================


class ReviewQueueItemResponse(BaseModel):
    """
    Summary response for a review queue item.

    Used in list responses to provide essential information about
    review queue items.

    Attributes:
        id: Unique identifier
        input_text: Original input text
        entity_type: Type of entity
        org_id: Organization ID
        top_candidate_id: ID of highest-scoring candidate
        top_candidate_name: Name of highest-scoring candidate
        confidence: Confidence score of top candidate
        match_method: Method used to find candidates
        status: Current review status
        priority: Priority level
        assigned_to: Assigned reviewer ID
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: str = Field(..., description="Unique identifier")
    input_text: str = Field(..., description="Original input text")
    entity_type: str = Field(..., description="Type of entity")
    org_id: str = Field(..., description="Organization ID")
    top_candidate_id: Optional[str] = Field(None, description="ID of highest-scoring candidate")
    top_candidate_name: Optional[str] = Field(None, description="Name of highest-scoring candidate")
    confidence: float = Field(..., description="Confidence score of top candidate")
    match_method: Optional[str] = Field(None, description="Method used to find candidates")
    status: ReviewStatusEnum = Field(..., description="Current review status")
    priority: int = Field(..., description="Priority level")
    assigned_to: Optional[str] = Field(None, description="Assigned reviewer ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "input_text": "Nat Gas",
                "entity_type": "fuel",
                "org_id": "org-123",
                "top_candidate_id": "GL-FUEL-NATGAS",
                "top_candidate_name": "Natural gas",
                "confidence": 0.72,
                "match_method": "fuzzy",
                "status": "pending",
                "priority": 0,
                "assigned_to": None,
                "created_at": "2026-01-30T10:30:00Z",
                "updated_at": "2026-01-30T10:30:00Z"
            }
        }


class ReviewQueueItemDetailResponse(ReviewQueueItemResponse):
    """
    Detailed response for a single review queue item.

    Extends the summary response with full candidate list, context,
    and resolution information.

    Additional Attributes:
        source_record_id: ID of the source record in normalizer
        pipeline_id: Pipeline that generated this item
        candidates: Full list of candidate matches
        context: Resolution context information
        vocabulary_version: Vocabulary version used
        resolved_at: Resolution timestamp
        resolved_by: User who resolved the item
        resolution: Resolution details if resolved
    """
    source_record_id: Optional[str] = Field(None, description="Source record ID")
    pipeline_id: Optional[str] = Field(None, description="Pipeline ID")
    candidates: List[CandidateInfo] = Field(default_factory=list, description="Candidate matches")
    context: ContextInfo = Field(default_factory=ContextInfo, description="Resolution context")
    vocabulary_version: Optional[str] = Field(None, description="Vocabulary version used")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolved_by: Optional[str] = Field(None, description="User who resolved")
    resolution: Optional["ResolutionResponse"] = Field(None, description="Resolution details")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "input_text": "Nat Gas",
                "entity_type": "fuel",
                "org_id": "org-123",
                "source_record_id": "src-456",
                "pipeline_id": "pipeline-789",
                "top_candidate_id": "GL-FUEL-NATGAS",
                "top_candidate_name": "Natural gas",
                "confidence": 0.72,
                "match_method": "fuzzy",
                "status": "pending",
                "priority": 0,
                "assigned_to": None,
                "candidates": [
                    {
                        "id": "GL-FUEL-NATGAS",
                        "name": "Natural gas",
                        "score": 0.72,
                        "source": "fuels_vocab",
                        "match_method": "fuzzy"
                    },
                    {
                        "id": "GL-FUEL-LNG",
                        "name": "Liquefied natural gas",
                        "score": 0.58,
                        "source": "fuels_vocab",
                        "match_method": "fuzzy"
                    }
                ],
                "context": {
                    "industry_sector": "energy",
                    "region": "US"
                },
                "vocabulary_version": "2026.01.0",
                "created_at": "2026-01-30T10:30:00Z",
                "updated_at": "2026-01-30T10:30:00Z",
                "resolved_at": None,
                "resolved_by": None,
                "resolution": None
            }
        }


class ResolutionResponse(BaseModel):
    """
    Response for a resolution decision.

    Provides details about how a review queue item was resolved.

    Attributes:
        id: Resolution ID
        item_id: Review queue item ID
        canonical_id: Selected canonical entity ID
        canonical_name: Selected canonical entity name
        reviewer_id: User ID of reviewer
        reviewer_email: Email of reviewer
        notes: Reviewer notes
        confidence_override: Confidence override if set
        created_at: Resolution timestamp
    """
    id: str = Field(..., description="Resolution ID")
    item_id: str = Field(..., description="Review queue item ID")
    canonical_id: str = Field(..., description="Selected canonical entity ID")
    canonical_name: str = Field(..., description="Selected canonical entity name")
    reviewer_id: str = Field(..., description="User ID of reviewer")
    reviewer_email: Optional[str] = Field(None, description="Email of reviewer")
    notes: Optional[str] = Field(None, description="Reviewer notes")
    confidence_override: Optional[float] = Field(None, description="Confidence override")
    created_at: datetime = Field(..., description="Resolution timestamp")

    class Config:
        from_attributes = True


class QueueStatsResponse(BaseModel):
    """
    Dashboard statistics for the review queue.

    Provides aggregate statistics for monitoring queue health
    and reviewer productivity.

    Attributes:
        pending_count: Number of items pending review
        in_progress_count: Number of items in progress
        resolved_today: Number of items resolved today
        rejected_today: Number of items rejected today
        escalated_count: Number of items escalated
        avg_resolution_time_seconds: Average resolution time in seconds
        avg_confidence: Average confidence of pending items
        oldest_pending_age_hours: Age of oldest pending item in hours
        items_by_entity_type: Breakdown by entity type
        items_by_org: Breakdown by organization (top 10)
    """
    pending_count: int = Field(..., description="Items pending review")
    in_progress_count: int = Field(..., description="Items in progress")
    resolved_today: int = Field(..., description="Items resolved today")
    rejected_today: int = Field(..., description="Items rejected today")
    escalated_count: int = Field(..., description="Items escalated")
    avg_resolution_time_seconds: Optional[float] = Field(
        None, description="Average resolution time in seconds"
    )
    avg_confidence: Optional[float] = Field(
        None, description="Average confidence of pending items"
    )
    oldest_pending_age_hours: Optional[float] = Field(
        None, description="Age of oldest pending item in hours"
    )
    items_by_entity_type: Dict[str, int] = Field(
        default_factory=dict, description="Breakdown by entity type"
    )
    items_by_org: Dict[str, int] = Field(
        default_factory=dict, description="Breakdown by organization"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pending_count": 42,
                "in_progress_count": 5,
                "resolved_today": 28,
                "rejected_today": 3,
                "escalated_count": 2,
                "avg_resolution_time_seconds": 145.5,
                "avg_confidence": 0.68,
                "oldest_pending_age_hours": 12.5,
                "items_by_entity_type": {
                    "fuel": 25,
                    "material": 12,
                    "process": 5
                },
                "items_by_org": {
                    "org-123": 20,
                    "org-456": 15,
                    "org-789": 7
                }
            }
        }


class VocabularySuggestionResponse(BaseModel):
    """
    Response for a vocabulary suggestion.

    Provides status and details about a vocabulary suggestion,
    including PR information if created.

    Attributes:
        id: Suggestion ID
        entity_type: Type of entity
        canonical_name: Suggested canonical name
        aliases: Suggested aliases
        source: Source/justification
        status: Current suggestion status
        pr_url: GitHub PR URL if created
        pr_number: GitHub PR number if created
        suggested_by: User who suggested
        created_at: Creation timestamp
    """
    id: str = Field(..., description="Suggestion ID")
    entity_type: str = Field(..., description="Type of entity")
    canonical_name: str = Field(..., description="Suggested canonical name")
    aliases: List[str] = Field(default_factory=list, description="Suggested aliases")
    source: str = Field(..., description="Source/justification")
    status: SuggestionStatusEnum = Field(..., description="Current status")
    pr_url: Optional[str] = Field(None, description="GitHub PR URL")
    pr_number: Optional[int] = Field(None, description="GitHub PR number")
    suggested_by: str = Field(..., description="User who suggested")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True


# Generic paginated response
T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Paginated response wrapper.

    Wraps list responses with pagination metadata.

    Attributes:
        items: List of items
        total: Total number of items
        page: Current page number (1-indexed)
        page_size: Items per page
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_prev: Whether there is a previous page
    """
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")


class HealthResponse(BaseModel):
    """
    Health check response.

    Provides service health status and dependency checks.

    Attributes:
        status: Overall health status
        timestamp: Check timestamp
        version: Service version
        checks: Individual dependency checks
    """
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Dependency checks")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-01-30T10:30:00Z",
                "version": "0.1.0",
                "checks": {
                    "database": True,
                    "redis": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response.

    Provides structured error information for API error responses.

    Attributes:
        error: Error code
        message: Human-readable error message
        details: Additional error details
        request_id: Request ID for correlation
    """
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for correlation")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ITEM_NOT_FOUND",
                "message": "Review queue item not found",
                "details": {"item_id": "550e8400-e29b-41d4-a716-446655440000"},
                "request_id": "req-abc123"
            }
        }
