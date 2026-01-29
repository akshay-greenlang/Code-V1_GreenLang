"""
Resolution Service for processing reviewer decisions.

This module provides the ResolutionService class for handling resolution
and rejection of review queue items, including audit trail creation
and vocabulary suggestion management.

Features:
    - Resolve items with selected canonical entity
    - Reject items with reason tracking
    - Escalate items to senior reviewers
    - Create vocabulary suggestions
    - Generate GitHub PRs for vocabulary changes

Example:
    >>> from review_console.services.resolution import ResolutionService
    >>> service = ResolutionService(db_session)
    >>> resolution = await service.resolve_item(
    ...     item_id="550e8400-...",
    ...     canonical_id="GL-FUEL-NATGAS",
    ...     canonical_name="Natural gas",
    ...     reviewer_id="reviewer@greenlang.io"
    ... )
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
import structlog

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from review_console.config import get_settings
from review_console.db.models import (
    ReviewQueueItem,
    Resolution,
    AuditLogEntry,
    VocabularySuggestion,
    ReviewStatus,
    AuditAction,
    SuggestionStatus,
)

logger = structlog.get_logger()
settings = get_settings()


class ResolutionError(Exception):
    """Exception raised for resolution errors."""

    def __init__(self, message: str, code: str = "RESOLUTION_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class ResolutionService:
    """
    Service for processing resolution decisions.

    Handles the resolution, rejection, and escalation of review queue
    items, with full audit trail support.

    Attributes:
        db: Async database session

    Example:
        >>> service = ResolutionService(db_session)
        >>> item = await service.resolve_item(
        ...     item_id="550e8400-...",
        ...     canonical_id="GL-FUEL-NATGAS",
        ...     canonical_name="Natural gas",
        ...     reviewer_id="reviewer@greenlang.io"
        ... )
    """

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize the service with a database session.

        Args:
            db: Async SQLAlchemy session
        """
        self.db = db

    async def resolve_item(
        self,
        item_id: str,
        canonical_id: str,
        canonical_name: str,
        reviewer_id: str,
        reviewer_email: Optional[str] = None,
        notes: Optional[str] = None,
        confidence_override: Optional[float] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Resolution:
        """
        Resolve a review queue item with a selected canonical entity.

        Creates a resolution record and updates the item status to RESOLVED.
        Also creates an audit trail entry.

        Args:
            item_id: UUID of the review queue item
            canonical_id: ID of the selected canonical entity
            canonical_name: Name of the selected canonical entity
            reviewer_id: User ID of the reviewer
            reviewer_email: Email of the reviewer
            notes: Reviewer notes explaining the decision
            confidence_override: Optional confidence override
            ip_address: IP address for audit
            user_agent: User agent for audit

        Returns:
            Created Resolution record

        Raises:
            ResolutionError: If item not found or already resolved

        Example:
            >>> resolution = await service.resolve_item(
            ...     item_id="550e8400-...",
            ...     canonical_id="GL-FUEL-NATGAS",
            ...     canonical_name="Natural gas",
            ...     reviewer_id="reviewer@greenlang.io",
            ...     notes="Exact match confirmed"
            ... )
        """
        # Get the item
        query = select(ReviewQueueItem).where(ReviewQueueItem.id == item_id)
        result = await self.db.execute(query)
        item = result.scalar_one_or_none()

        if not item:
            raise ResolutionError(
                f"Review queue item {item_id} not found",
                code="ITEM_NOT_FOUND"
            )

        if item.status == ReviewStatus.RESOLVED:
            raise ResolutionError(
                f"Review queue item {item_id} is already resolved",
                code="ITEM_ALREADY_RESOLVED"
            )

        # Create resolution
        resolution = Resolution(
            item_id=item_id,
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            reviewer_id=reviewer_id,
            reviewer_email=reviewer_email,
            notes=notes,
            confidence_override=confidence_override,
        )

        self.db.add(resolution)

        # Update item status
        item.status = ReviewStatus.RESOLVED
        item.resolved_at = datetime.now(timezone.utc)
        item.resolved_by = reviewer_id

        # Create audit entry
        audit_entry = AuditLogEntry(
            item_id=item_id,
            action=AuditAction.ITEM_RESOLVED,
            actor_id=reviewer_id,
            actor_email=reviewer_email,
            details={
                "canonical_id": canonical_id,
                "canonical_name": canonical_name,
                "notes": notes,
                "confidence_override": confidence_override,
                "original_confidence": item.confidence,
                "original_top_candidate": item.top_candidate_id,
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.db.add(audit_entry)

        await self.db.commit()
        await self.db.refresh(resolution)

        logger.info(
            "Item resolved",
            item_id=item_id,
            canonical_id=canonical_id,
            reviewer_id=reviewer_id,
        )

        return resolution

    async def reject_item(
        self,
        item_id: str,
        reason: str,
        reviewer_id: str,
        reviewer_email: Optional[str] = None,
        escalate_to: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ReviewQueueItem:
        """
        Reject a review queue item.

        Marks the item as REJECTED or ESCALATED if escalate_to is provided.
        Creates an audit trail entry.

        Args:
            item_id: UUID of the review queue item
            reason: Reason for rejection/escalation
            reviewer_id: User ID of the reviewer
            reviewer_email: Email of the reviewer
            escalate_to: User ID to escalate to (triggers escalation)
            ip_address: IP address for audit
            user_agent: User agent for audit

        Returns:
            Updated ReviewQueueItem

        Raises:
            ResolutionError: If item not found or already resolved

        Example:
            >>> item = await service.reject_item(
            ...     item_id="550e8400-...",
            ...     reason="No matching entity in vocabulary",
            ...     reviewer_id="reviewer@greenlang.io"
            ... )
        """
        # Get the item
        query = select(ReviewQueueItem).where(ReviewQueueItem.id == item_id)
        result = await self.db.execute(query)
        item = result.scalar_one_or_none()

        if not item:
            raise ResolutionError(
                f"Review queue item {item_id} not found",
                code="ITEM_NOT_FOUND"
            )

        if item.status in (ReviewStatus.RESOLVED, ReviewStatus.REJECTED):
            raise ResolutionError(
                f"Review queue item {item_id} is already {item.status.value}",
                code="ITEM_ALREADY_CLOSED"
            )

        # Determine action and new status
        if escalate_to:
            new_status = ReviewStatus.ESCALATED
            action = AuditAction.ITEM_ESCALATED
            item.assigned_to = escalate_to
        else:
            new_status = ReviewStatus.REJECTED
            action = AuditAction.ITEM_REJECTED
            item.resolved_at = datetime.now(timezone.utc)
            item.resolved_by = reviewer_id

        item.status = new_status

        # Create audit entry
        audit_entry = AuditLogEntry(
            item_id=item_id,
            action=action,
            actor_id=reviewer_id,
            actor_email=reviewer_email,
            details={
                "reason": reason,
                "escalate_to": escalate_to,
                "previous_status": item.status.value,
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.db.add(audit_entry)

        await self.db.commit()
        await self.db.refresh(item)

        logger.info(
            "Item rejected/escalated",
            item_id=item_id,
            action=action.value,
            reviewer_id=reviewer_id,
            escalate_to=escalate_to,
        )

        return item

    async def create_vocabulary_suggestion(
        self,
        entity_type: str,
        canonical_name: str,
        aliases: List[str],
        source: str,
        suggested_by: str,
        org_id: str,
        suggested_by_email: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> VocabularySuggestion:
        """
        Create a vocabulary suggestion.

        Creates a new vocabulary suggestion for governance review.
        Optionally creates a GitHub PR if configured.

        Args:
            entity_type: Type of entity (fuel, material, process)
            canonical_name: Suggested canonical name
            aliases: List of suggested aliases
            source: Source/justification for the suggestion
            suggested_by: User ID who suggested the entry
            org_id: Organization ID
            suggested_by_email: Email of the suggester
            properties: Suggested properties for the entity
            ip_address: IP address for audit
            user_agent: User agent for audit

        Returns:
            Created VocabularySuggestion

        Example:
            >>> suggestion = await service.create_vocabulary_suggestion(
            ...     entity_type="fuel",
            ...     canonical_name="Sustainable Aviation Fuel",
            ...     aliases=["SAF", "Bio-jet fuel"],
            ...     source="ICAO CORSIA eligible fuels list",
            ...     suggested_by="reviewer@greenlang.io",
            ...     org_id="org-123"
            ... )
        """
        # Create suggestion
        suggestion = VocabularySuggestion(
            entity_type=entity_type,
            canonical_name=canonical_name,
            aliases=aliases,
            source=source,
            suggested_by=suggested_by,
            suggested_by_email=suggested_by_email,
            properties=properties or {},
            org_id=org_id,
            status=SuggestionStatus.PENDING,
        )

        self.db.add(suggestion)

        # Create audit entry
        audit_entry = AuditLogEntry(
            item_id=None,  # Not tied to a specific item
            action=AuditAction.VOCAB_SUGGESTED,
            actor_id=suggested_by,
            actor_email=suggested_by_email,
            details={
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "source": source,
            },
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.db.add(audit_entry)

        await self.db.commit()
        await self.db.refresh(suggestion)

        logger.info(
            "Vocabulary suggestion created",
            suggestion_id=suggestion.id,
            entity_type=entity_type,
            canonical_name=canonical_name,
            suggested_by=suggested_by,
        )

        # Try to create GitHub PR if configured
        if settings.github_token:
            try:
                await self._create_github_pr(suggestion)
            except Exception as e:
                logger.warning(
                    "Failed to create GitHub PR for vocabulary suggestion",
                    suggestion_id=suggestion.id,
                    error=str(e),
                )

        return suggestion

    async def _create_github_pr(self, suggestion: VocabularySuggestion) -> None:
        """
        Create a GitHub PR for a vocabulary suggestion.

        Internal method that creates a pull request in the vocabulary
        repository for governance review.

        Args:
            suggestion: VocabularySuggestion to create PR for

        Raises:
            Exception: If PR creation fails
        """
        if not settings.github_token:
            return

        # Generate branch name
        branch_name = f"vocab-suggestion/{suggestion.entity_type}/{suggestion.id[:8]}"

        # Generate PR title
        pr_title = f"[Vocabulary] Add {suggestion.entity_type}: {suggestion.canonical_name}"

        # Generate PR body
        pr_body = f"""## Vocabulary Suggestion

**Entity Type:** {suggestion.entity_type}
**Canonical Name:** {suggestion.canonical_name}
**Aliases:** {', '.join(suggestion.aliases)}

### Source/Justification
{suggestion.source}

### Properties
```json
{suggestion.properties}
```

### Metadata
- **Suggested By:** {suggestion.suggested_by_email or suggestion.suggested_by}
- **Organization:** {suggestion.org_id}
- **Created At:** {suggestion.created_at.isoformat()}

---
*This PR was automatically created by the Review Console.*
"""

        # Note: In a real implementation, this would:
        # 1. Create a branch
        # 2. Add/modify vocabulary YAML files
        # 3. Create a pull request
        # For now, we log the intent

        logger.info(
            "Would create GitHub PR",
            branch_name=branch_name,
            pr_title=pr_title,
            repository=settings.github_repo,
        )

        # Update suggestion with placeholder PR info
        # In production, this would be the actual PR URL and number
        # suggestion.pr_url = f"https://github.com/{settings.github_repo}/pull/XXX"
        # suggestion.pr_number = XXX
        # suggestion.status = SuggestionStatus.PR_CREATED

    async def get_suggestion_by_id(
        self,
        suggestion_id: str,
    ) -> Optional[VocabularySuggestion]:
        """
        Get a vocabulary suggestion by ID.

        Args:
            suggestion_id: UUID of the suggestion

        Returns:
            VocabularySuggestion if found, None otherwise

        Example:
            >>> suggestion = await service.get_suggestion_by_id("550e8400-...")
        """
        query = select(VocabularySuggestion).where(
            VocabularySuggestion.id == suggestion_id
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_resolution_by_item_id(
        self,
        item_id: str,
    ) -> Optional[Resolution]:
        """
        Get a resolution by item ID.

        Args:
            item_id: UUID of the review queue item

        Returns:
            Resolution if found, None otherwise

        Example:
            >>> resolution = await service.get_resolution_by_item_id("550e8400-...")
        """
        query = select(Resolution).where(Resolution.item_id == item_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
