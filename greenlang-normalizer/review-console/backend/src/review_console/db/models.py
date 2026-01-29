"""
SQLAlchemy ORM models for Review Console Backend.

This module defines the database models for the human review queue,
including review queue items, resolutions, audit logs, and vocabulary
suggestions.

Models:
    - ReviewQueueItem: Items pending human review from entity resolution
    - Resolution: Resolution decisions made by reviewers
    - AuditLogEntry: Immutable audit trail for all actions
    - VocabularySuggestion: Suggested vocabulary entries awaiting governance

Example:
    >>> from review_console.db.models import ReviewQueueItem, ReviewStatus
    >>> item = ReviewQueueItem(
    ...     input_text="Nat Gas",
    ...     entity_type="fuel",
    ...     org_id="org-123",
    ...     confidence=0.72,
    ...     status=ReviewStatus.PENDING
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import (
    Column,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Enum as SQLEnum,
    JSON,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class ReviewStatus(str, Enum):
    """
    Status of a review queue item.

    Attributes:
        PENDING: Item awaiting review
        IN_PROGRESS: Item currently being reviewed
        RESOLVED: Item has been resolved with a canonical ID
        REJECTED: Item was rejected (no valid match)
        ESCALATED: Item escalated to senior reviewer
        AUTO_RESOLVED: Item automatically resolved by high-confidence match
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    AUTO_RESOLVED = "auto_resolved"


class AuditAction(str, Enum):
    """
    Type of action recorded in audit log.

    Attributes:
        ITEM_CREATED: Review queue item created
        ITEM_VIEWED: Item viewed by reviewer
        ITEM_ASSIGNED: Item assigned to reviewer
        ITEM_RESOLVED: Item resolved with canonical ID
        ITEM_REJECTED: Item rejected
        ITEM_ESCALATED: Item escalated
        VOCAB_SUGGESTED: Vocabulary entry suggested
        VOCAB_PR_CREATED: GitHub PR created for vocabulary
    """
    ITEM_CREATED = "item_created"
    ITEM_VIEWED = "item_viewed"
    ITEM_ASSIGNED = "item_assigned"
    ITEM_RESOLVED = "item_resolved"
    ITEM_REJECTED = "item_rejected"
    ITEM_ESCALATED = "item_escalated"
    VOCAB_SUGGESTED = "vocab_suggested"
    VOCAB_PR_CREATED = "vocab_pr_created"


class SuggestionStatus(str, Enum):
    """
    Status of a vocabulary suggestion.

    Attributes:
        PENDING: Suggestion awaiting governance review
        PR_CREATED: GitHub PR created
        APPROVED: Suggestion approved and merged
        REJECTED: Suggestion rejected by governance
    """
    PENDING = "pending"
    PR_CREATED = "pr_created"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewQueueItem(Base):
    """
    A review queue item awaiting human review.

    Represents an entity resolution result that requires human review,
    typically because the confidence score was below the threshold or
    multiple candidates had similar scores.

    Attributes:
        id: Unique identifier (UUID)
        input_text: Original input text that was resolved
        entity_type: Type of entity (fuel, material, process)
        org_id: Organization ID for tenant isolation
        source_record_id: ID of the source record in the normalizer
        pipeline_id: ID of the pipeline that generated this item
        candidates: JSON list of candidate matches with scores
        top_candidate_id: ID of the highest-scoring candidate
        top_candidate_name: Name of the highest-scoring candidate
        confidence: Confidence score of the top candidate (0.0-1.0)
        match_method: Method used to find candidates (exact, fuzzy, llm)
        context: Additional context for the resolution (JSON)
        status: Current status of the review item
        priority: Priority level (higher = more urgent)
        assigned_to: User ID of assigned reviewer (nullable)
        created_at: Timestamp when item was created
        updated_at: Timestamp of last update
        resolved_at: Timestamp when item was resolved (nullable)
        resolved_by: User ID who resolved the item (nullable)
        vocabulary_version: Version of vocabulary used for resolution

    Relationships:
        resolution: The Resolution decision for this item (one-to-one)
        audit_entries: Audit log entries for this item (one-to-many)
    """

    __tablename__ = "review_queue_items"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique identifier"
    )
    input_text: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        comment="Original input text for resolution"
    )
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of entity (fuel, material, process)"
    )
    org_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Organization ID for tenant isolation"
    )
    source_record_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="ID of the source record in normalizer"
    )
    pipeline_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Pipeline that generated this item"
    )

    # Candidate information
    candidates: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of candidate matches with scores"
    )
    top_candidate_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="ID of highest-scoring candidate"
    )
    top_candidate_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Name of highest-scoring candidate"
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
        comment="Confidence score of top candidate (0.0-1.0)"
    )
    match_method: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Method used to find candidates"
    )

    # Context and metadata
    context: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context for resolution"
    )
    vocabulary_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Vocabulary version used for resolution"
    )

    # Status and assignment
    status: Mapped[ReviewStatus] = mapped_column(
        SQLEnum(ReviewStatus),
        nullable=False,
        default=ReviewStatus.PENDING,
        index=True,
        comment="Current review status"
    )
    priority: Mapped[int] = mapped_column(
        default=0,
        comment="Priority level (higher = more urgent)"
    )
    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        index=True,
        comment="User ID of assigned reviewer"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp"
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Resolution timestamp"
    )
    resolved_by: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="User ID who resolved the item"
    )

    # Relationships
    resolution: Mapped[Optional["Resolution"]] = relationship(
        "Resolution",
        back_populates="queue_item",
        uselist=False,
        cascade="all, delete-orphan"
    )
    audit_entries: Mapped[List["AuditLogEntry"]] = relationship(
        "AuditLogEntry",
        back_populates="queue_item",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_review_queue_org_status", "org_id", "status"),
        Index("ix_review_queue_entity_status", "entity_type", "status"),
        Index("ix_review_queue_created_at", "created_at"),
        Index("ix_review_queue_confidence", "confidence"),
    )

    def __repr__(self) -> str:
        return (
            f"<ReviewQueueItem(id={self.id}, input_text='{self.input_text[:30]}...', "
            f"status={self.status.value}, confidence={self.confidence})>"
        )


class Resolution(Base):
    """
    Resolution decision for a review queue item.

    Records the canonical entity selected by the reviewer, along with
    any notes explaining the decision.

    Attributes:
        id: Unique identifier (UUID)
        item_id: ID of the review queue item
        canonical_id: Selected canonical entity ID
        canonical_name: Name of the selected canonical entity
        reviewer_id: User ID of the reviewer
        reviewer_email: Email of the reviewer
        notes: Reviewer notes explaining the decision
        confidence_override: Optional confidence override by reviewer
        created_at: Timestamp when resolution was created

    Relationships:
        queue_item: The ReviewQueueItem this resolves (many-to-one)
    """

    __tablename__ = "resolutions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique identifier"
    )
    item_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("review_queue_items.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="ID of the review queue item"
    )
    canonical_id: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        index=True,
        comment="Selected canonical entity ID"
    )
    canonical_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Name of selected canonical entity"
    )
    reviewer_id: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        index=True,
        comment="User ID of the reviewer"
    )
    reviewer_email: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Email of the reviewer"
    )
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Reviewer notes explaining the decision"
    )
    confidence_override: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Optional confidence override by reviewer"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Resolution timestamp"
    )

    # Relationships
    queue_item: Mapped["ReviewQueueItem"] = relationship(
        "ReviewQueueItem",
        back_populates="resolution"
    )

    __table_args__ = (
        Index("ix_resolutions_reviewer", "reviewer_id"),
        Index("ix_resolutions_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Resolution(id={self.id}, canonical_id='{self.canonical_id}', "
            f"reviewer_id='{self.reviewer_id}')>"
        )


class AuditLogEntry(Base):
    """
    Immutable audit log entry for tracking all actions.

    Records all actions taken on review queue items for compliance
    and debugging purposes. Audit entries are append-only.

    Attributes:
        id: Unique identifier (UUID)
        item_id: ID of the review queue item (nullable for global actions)
        action: Type of action performed
        actor_id: User ID who performed the action
        actor_email: Email of the actor
        timestamp: When the action occurred
        details: JSON details about the action
        ip_address: IP address of the actor
        user_agent: User agent string

    Relationships:
        queue_item: The ReviewQueueItem this entry relates to (many-to-one)
    """

    __tablename__ = "audit_log_entries"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique identifier"
    )
    item_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("review_queue_items.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="ID of related review queue item"
    )
    action: Mapped[AuditAction] = mapped_column(
        SQLEnum(AuditAction),
        nullable=False,
        index=True,
        comment="Type of action performed"
    )
    actor_id: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        index=True,
        comment="User ID who performed the action"
    )
    actor_email: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Email of the actor"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="When the action occurred"
    )
    details: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Details about the action"
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        String(45),
        nullable=True,
        comment="IP address of the actor"
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="User agent string"
    )

    # Relationships
    queue_item: Mapped[Optional["ReviewQueueItem"]] = relationship(
        "ReviewQueueItem",
        back_populates="audit_entries"
    )

    __table_args__ = (
        Index("ix_audit_log_timestamp", "timestamp"),
        Index("ix_audit_log_actor_action", "actor_id", "action"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLogEntry(id={self.id}, action={self.action.value}, "
            f"actor_id='{self.actor_id}', timestamp={self.timestamp})>"
        )


class VocabularySuggestion(Base):
    """
    Suggested vocabulary entry awaiting governance review.

    When reviewers cannot find a suitable canonical match, they can
    suggest new vocabulary entries. These suggestions go through
    a governance process before being added to the vocabulary.

    Attributes:
        id: Unique identifier (UUID)
        entity_type: Type of entity (fuel, material, process)
        canonical_name: Suggested canonical name
        aliases: List of suggested aliases
        source: Source/justification for the suggestion
        properties: Suggested properties for the entity
        suggested_by: User ID who suggested the entry
        suggested_by_email: Email of the suggester
        status: Current status of the suggestion
        pr_url: GitHub PR URL if created
        pr_number: GitHub PR number if created
        governance_notes: Notes from governance review
        created_at: Timestamp when suggestion was created
        updated_at: Timestamp of last update
        org_id: Organization ID for tenant isolation
    """

    __tablename__ = "vocabulary_suggestions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        comment="Unique identifier"
    )
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of entity (fuel, material, process)"
    )
    canonical_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Suggested canonical name"
    )
    aliases: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of suggested aliases"
    )
    source: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Source/justification for the suggestion"
    )
    properties: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Suggested properties for the entity"
    )
    suggested_by: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        index=True,
        comment="User ID who suggested the entry"
    )
    suggested_by_email: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Email of the suggester"
    )
    status: Mapped[SuggestionStatus] = mapped_column(
        SQLEnum(SuggestionStatus),
        nullable=False,
        default=SuggestionStatus.PENDING,
        index=True,
        comment="Current suggestion status"
    )
    pr_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="GitHub PR URL if created"
    )
    pr_number: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="GitHub PR number if created"
    )
    governance_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Notes from governance review"
    )
    org_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Organization ID for tenant isolation"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Creation timestamp"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp"
    )

    __table_args__ = (
        Index("ix_vocab_suggestions_status", "status"),
        Index("ix_vocab_suggestions_org", "org_id", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<VocabularySuggestion(id={self.id}, canonical_name='{self.canonical_name}', "
            f"entity_type='{self.entity_type}', status={self.status.value})>"
        )
