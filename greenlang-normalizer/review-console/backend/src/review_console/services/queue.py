"""
Review Queue Service for managing review items.

This module provides the ReviewQueueService class for querying and managing
review queue items, including filtering, pagination, and statistics.

Features:
    - Query review items with flexible filtering
    - Pagination support with configurable page sizes
    - Dashboard statistics calculation
    - Item assignment and status management

Example:
    >>> from review_console.services.queue import ReviewQueueService
    >>> service = ReviewQueueService(db_session)
    >>> items, total = await service.get_items(
    ...     entity_type="fuel",
    ...     status=ReviewStatus.PENDING,
    ...     page=1,
    ...     page_size=20
    ... )
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from review_console.db.models import (
    ReviewQueueItem,
    Resolution,
    AuditLogEntry,
    ReviewStatus,
    AuditAction,
)
from review_console.api.models import QueueFilterParams


class ReviewQueueService:
    """
    Service for managing review queue items.

    Provides methods for querying, filtering, and managing review queue
    items with support for pagination and statistics.

    Attributes:
        db: Async database session

    Example:
        >>> service = ReviewQueueService(db_session)
        >>> stats = await service.get_stats(org_id="org-123")
    """

    def __init__(self, db: AsyncSession) -> None:
        """
        Initialize the service with a database session.

        Args:
            db: Async SQLAlchemy session
        """
        self.db = db

    async def get_items(
        self,
        filters: Optional[QueueFilterParams] = None,
        page: int = 1,
        page_size: int = 20,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> Tuple[List[ReviewQueueItem], int]:
        """
        Get review queue items with filtering and pagination.

        Retrieves items matching the specified filters with pagination
        support. Returns both the items and total count for pagination.

        Args:
            filters: Filter parameters (entity_type, org_id, status, etc.)
            page: Page number (1-indexed)
            page_size: Number of items per page
            order_by: Field to order by
            order_desc: Whether to order descending

        Returns:
            Tuple of (items list, total count)

        Example:
            >>> items, total = await service.get_items(
            ...     filters=QueueFilterParams(entity_type="fuel"),
            ...     page=1,
            ...     page_size=20
            ... )
        """
        # Build base query
        query = select(ReviewQueueItem)
        count_query = select(func.count(ReviewQueueItem.id))

        # Apply filters
        conditions = []

        if filters:
            if filters.entity_type:
                conditions.append(ReviewQueueItem.entity_type == filters.entity_type.value)

            if filters.org_id:
                conditions.append(ReviewQueueItem.org_id == filters.org_id)

            if filters.status:
                conditions.append(ReviewQueueItem.status == ReviewStatus(filters.status.value))

            if filters.date_from:
                conditions.append(ReviewQueueItem.created_at >= filters.date_from)

            if filters.date_to:
                conditions.append(ReviewQueueItem.created_at <= filters.date_to)

            if filters.min_confidence is not None:
                conditions.append(ReviewQueueItem.confidence >= filters.min_confidence)

            if filters.max_confidence is not None:
                conditions.append(ReviewQueueItem.confidence <= filters.max_confidence)

            if filters.assigned_to:
                conditions.append(ReviewQueueItem.assigned_to == filters.assigned_to)

            if filters.search:
                search_pattern = f"%{filters.search}%"
                conditions.append(ReviewQueueItem.input_text.ilike(search_pattern))

        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))

        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply ordering
        order_column = getattr(ReviewQueueItem, order_by, ReviewQueueItem.created_at)
        if order_desc:
            query = query.order_by(order_column.desc())
        else:
            query = query.order_by(order_column.asc())

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # Execute query
        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return items, total

    async def get_item_by_id(
        self,
        item_id: str,
        include_resolution: bool = True,
    ) -> Optional[ReviewQueueItem]:
        """
        Get a single review queue item by ID.

        Retrieves a review queue item with optional resolution details.

        Args:
            item_id: UUID of the item
            include_resolution: Whether to include resolution details

        Returns:
            ReviewQueueItem if found, None otherwise

        Example:
            >>> item = await service.get_item_by_id("550e8400-...")
            >>> if item:
            ...     print(item.input_text)
        """
        query = select(ReviewQueueItem).where(ReviewQueueItem.id == item_id)

        if include_resolution:
            query = query.options(selectinload(ReviewQueueItem.resolution))

        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def assign_item(
        self,
        item_id: str,
        user_id: str,
    ) -> Optional[ReviewQueueItem]:
        """
        Assign a review queue item to a reviewer.

        Updates the item's assigned_to field and sets status to IN_PROGRESS.

        Args:
            item_id: UUID of the item
            user_id: User ID of the reviewer

        Returns:
            Updated ReviewQueueItem if found, None otherwise

        Example:
            >>> item = await service.assign_item(
            ...     item_id="550e8400-...",
            ...     user_id="reviewer@greenlang.io"
            ... )
        """
        item = await self.get_item_by_id(item_id, include_resolution=False)

        if not item:
            return None

        item.assigned_to = user_id
        if item.status == ReviewStatus.PENDING:
            item.status = ReviewStatus.IN_PROGRESS

        await self.db.commit()
        await self.db.refresh(item)

        return item

    async def update_status(
        self,
        item_id: str,
        status: ReviewStatus,
        resolved_by: Optional[str] = None,
    ) -> Optional[ReviewQueueItem]:
        """
        Update the status of a review queue item.

        Changes the item's status and optionally records the resolver.

        Args:
            item_id: UUID of the item
            status: New status
            resolved_by: User ID who resolved (for RESOLVED status)

        Returns:
            Updated ReviewQueueItem if found, None otherwise

        Example:
            >>> item = await service.update_status(
            ...     item_id="550e8400-...",
            ...     status=ReviewStatus.RESOLVED,
            ...     resolved_by="reviewer@greenlang.io"
            ... )
        """
        item = await self.get_item_by_id(item_id, include_resolution=False)

        if not item:
            return None

        item.status = status

        if status in (ReviewStatus.RESOLVED, ReviewStatus.REJECTED):
            item.resolved_at = datetime.now(timezone.utc)
            if resolved_by:
                item.resolved_by = resolved_by

        await self.db.commit()
        await self.db.refresh(item)

        return item

    async def get_stats(
        self,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get dashboard statistics for the review queue.

        Calculates aggregate statistics including pending counts,
        resolution rates, and breakdowns by entity type.

        Args:
            org_id: Optional organization ID to filter stats

        Returns:
            Dictionary with queue statistics

        Example:
            >>> stats = await service.get_stats(org_id="org-123")
            >>> print(f"Pending: {stats['pending_count']}")
        """
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Base condition for org filtering
        org_condition = ReviewQueueItem.org_id == org_id if org_id else True

        # Pending count
        pending_query = select(func.count(ReviewQueueItem.id)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.PENDING,
                org_condition
            )
        )
        pending_result = await self.db.execute(pending_query)
        pending_count = pending_result.scalar() or 0

        # In progress count
        in_progress_query = select(func.count(ReviewQueueItem.id)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.IN_PROGRESS,
                org_condition
            )
        )
        in_progress_result = await self.db.execute(in_progress_query)
        in_progress_count = in_progress_result.scalar() or 0

        # Escalated count
        escalated_query = select(func.count(ReviewQueueItem.id)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.ESCALATED,
                org_condition
            )
        )
        escalated_result = await self.db.execute(escalated_query)
        escalated_count = escalated_result.scalar() or 0

        # Resolved today
        resolved_today_query = select(func.count(ReviewQueueItem.id)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.RESOLVED,
                ReviewQueueItem.resolved_at >= today_start,
                org_condition
            )
        )
        resolved_today_result = await self.db.execute(resolved_today_query)
        resolved_today = resolved_today_result.scalar() or 0

        # Rejected today
        rejected_today_query = select(func.count(ReviewQueueItem.id)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.REJECTED,
                ReviewQueueItem.resolved_at >= today_start,
                org_condition
            )
        )
        rejected_today_result = await self.db.execute(rejected_today_query)
        rejected_today = rejected_today_result.scalar() or 0

        # Average confidence of pending items
        avg_confidence_query = select(func.avg(ReviewQueueItem.confidence)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.PENDING,
                org_condition
            )
        )
        avg_confidence_result = await self.db.execute(avg_confidence_query)
        avg_confidence = avg_confidence_result.scalar()

        # Oldest pending item age
        oldest_pending_query = select(func.min(ReviewQueueItem.created_at)).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.PENDING,
                org_condition
            )
        )
        oldest_pending_result = await self.db.execute(oldest_pending_query)
        oldest_pending_created = oldest_pending_result.scalar()

        oldest_pending_age_hours = None
        if oldest_pending_created:
            # Handle timezone-aware comparison
            if oldest_pending_created.tzinfo is None:
                oldest_pending_created = oldest_pending_created.replace(tzinfo=timezone.utc)
            age_delta = now - oldest_pending_created
            oldest_pending_age_hours = age_delta.total_seconds() / 3600

        # Average resolution time (for items resolved today)
        # This requires joining with resolved_at and created_at
        avg_resolution_time_query = select(
            func.avg(
                func.extract('epoch', ReviewQueueItem.resolved_at) -
                func.extract('epoch', ReviewQueueItem.created_at)
            )
        ).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.RESOLVED,
                ReviewQueueItem.resolved_at >= today_start,
                org_condition
            )
        )
        avg_time_result = await self.db.execute(avg_resolution_time_query)
        avg_resolution_time = avg_time_result.scalar()

        # Items by entity type
        entity_type_query = select(
            ReviewQueueItem.entity_type,
            func.count(ReviewQueueItem.id)
        ).where(
            and_(
                ReviewQueueItem.status == ReviewStatus.PENDING,
                org_condition
            )
        ).group_by(ReviewQueueItem.entity_type)
        entity_type_result = await self.db.execute(entity_type_query)
        items_by_entity_type = dict(entity_type_result.all())

        # Items by org (top 10, only if not filtering by org)
        items_by_org: Dict[str, int] = {}
        if not org_id:
            org_query = select(
                ReviewQueueItem.org_id,
                func.count(ReviewQueueItem.id)
            ).where(
                ReviewQueueItem.status == ReviewStatus.PENDING
            ).group_by(ReviewQueueItem.org_id).order_by(
                func.count(ReviewQueueItem.id).desc()
            ).limit(10)
            org_result = await self.db.execute(org_query)
            items_by_org = dict(org_result.all())

        return {
            "pending_count": pending_count,
            "in_progress_count": in_progress_count,
            "escalated_count": escalated_count,
            "resolved_today": resolved_today,
            "rejected_today": rejected_today,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
            "oldest_pending_age_hours": round(oldest_pending_age_hours, 2) if oldest_pending_age_hours else None,
            "avg_resolution_time_seconds": round(avg_resolution_time, 2) if avg_resolution_time else None,
            "items_by_entity_type": items_by_entity_type,
            "items_by_org": items_by_org,
        }

    async def create_audit_entry(
        self,
        item_id: Optional[str],
        action: AuditAction,
        actor_id: str,
        actor_email: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Create an audit log entry.

        Records an action in the audit log for compliance and debugging.

        Args:
            item_id: UUID of related review queue item (optional)
            action: Type of action performed
            actor_id: User ID who performed the action
            actor_email: Email of the actor
            details: Additional details about the action
            ip_address: IP address of the actor
            user_agent: User agent string

        Returns:
            Created AuditLogEntry

        Example:
            >>> entry = await service.create_audit_entry(
            ...     item_id="550e8400-...",
            ...     action=AuditAction.ITEM_VIEWED,
            ...     actor_id="reviewer@greenlang.io"
            ... )
        """
        entry = AuditLogEntry(
            item_id=item_id,
            action=action,
            actor_id=actor_id,
            actor_email=actor_email,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)

        return entry
