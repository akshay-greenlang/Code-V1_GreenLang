"""
Review Queue Management

CRUD operations for human review queue with JSON/SQLite persistence.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from ..models import ReviewQueueItem, ReviewStatus, ReviewAction
from ..config import get_config
from ..exceptions import QueueItemNotFoundError, QueuePersistenceError

logger = logging.getLogger(__name__)


class ReviewQueue:
    """
    Human review queue management.

    Features:
    - JSON-based persistence
    - Priority-based sorting
    - CRUD operations
    - Status tracking
    - Auto-cleanup of old items
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize review queue.

        Args:
            storage_path: Path to queue storage directory
        """
        self.config = get_config().review_queue
        self.storage_path = storage_path or self.config.storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.storage_path / "queue.json"

        # Load existing queue
        self._load_queue()

        logger.info(f"Initialized ReviewQueue at {self.storage_path}")

    def _load_queue(self) -> None:
        """Load queue from storage."""
        if self.queue_file.exists():
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert to ReviewQueueItem objects
                self.items: Dict[str, ReviewQueueItem] = {}
                for key, value in data.items():
                    try:
                        self.items[key] = ReviewQueueItem(**value)
                    except Exception as e:
                        logger.warning(f"Skipping invalid queue item {key}: {e}")

                logger.info(f"Loaded {len(self.items)} items from queue")

            except Exception as e:
                logger.error(f"Failed to load queue: {e}, starting fresh")
                self.items = {}
        else:
            self.items = {}
            logger.info("No existing queue found, starting fresh")

    def _save_queue(self) -> None:
        """Save queue to storage."""
        try:
            # Convert items to dicts
            data = {
                key: item.model_dump(mode='json')
                for key, item in self.items.items()
            }

            # Save to file
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved {len(self.items)} items to queue")

        except Exception as e:
            raise QueuePersistenceError(
                f"Failed to save queue: {str(e)}",
                details={"queue_file": str(self.queue_file)}
            ) from e

    def add(self, item: ReviewQueueItem) -> str:
        """
        Add item to review queue.

        Args:
            item: Queue item to add

        Returns:
            Queue item ID
        """
        # Check queue size limit
        if len(self.items) >= self.config.max_queue_size:
            logger.warning(f"Queue size limit reached ({self.config.max_queue_size})")
            # Remove oldest completed items
            self._cleanup_old_items()

        self.items[item.queue_item_id] = item
        self._save_queue()

        logger.info(
            f"Added item to queue: {item.queue_item_id} "
            f"(priority: {item.priority}, entity: {item.original_name})"
        )

        return item.queue_item_id

    def get(self, queue_item_id: str) -> ReviewQueueItem:
        """
        Get queue item by ID.

        Args:
            queue_item_id: Queue item ID

        Returns:
            Queue item

        Raises:
            QueueItemNotFoundError: If item not found
        """
        if queue_item_id not in self.items:
            raise QueueItemNotFoundError(
                f"Queue item not found: {queue_item_id}",
                details={"queue_item_id": queue_item_id}
            )

        return self.items[queue_item_id]

    def update(self, item: ReviewQueueItem) -> None:
        """
        Update queue item.

        Args:
            item: Updated queue item
        """
        if item.queue_item_id not in self.items:
            raise QueueItemNotFoundError(f"Queue item not found: {item.queue_item_id}")

        self.items[item.queue_item_id] = item
        self._save_queue()

        logger.info(f"Updated queue item: {item.queue_item_id}")

    def list_pending(self, limit: Optional[int] = None) -> List[ReviewQueueItem]:
        """
        List pending items sorted by priority.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of pending queue items
        """
        pending = [
            item for item in self.items.values()
            if item.status == ReviewStatus.PENDING
        ]

        # Sort by priority (high -> medium -> low) then by created_at
        priority_order = {"high": 0, "medium": 1, "low": 2}
        pending.sort(key=lambda x: (priority_order.get(x.priority, 3), x.created_at))

        if limit:
            pending = pending[:limit]

        logger.info(f"Listed {len(pending)} pending items")
        return pending

    def list_by_status(
        self,
        status: ReviewStatus,
        limit: Optional[int] = None
    ) -> List[ReviewQueueItem]:
        """List items by status."""
        items = [item for item in self.items.values() if item.status == status]
        items.sort(key=lambda x: x.created_at, reverse=True)

        if limit:
            items = items[:limit]

        return items

    def update_status(
        self,
        queue_item_id: str,
        status: ReviewStatus,
        **kwargs
    ) -> ReviewQueueItem:
        """
        Update item status.

        Args:
            queue_item_id: Queue item ID
            status: New status
            **kwargs: Additional fields to update

        Returns:
            Updated queue item
        """
        item = self.get(queue_item_id)
        item.status = status

        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)

        self._save_queue()

        logger.info(f"Updated status: {queue_item_id} -> {status}")
        return item

    def assign(self, queue_item_id: str, reviewer: str) -> ReviewQueueItem:
        """Assign item to reviewer."""
        item = self.get(queue_item_id)
        item.assigned_to = reviewer
        item.status = ReviewStatus.IN_REVIEW
        self._save_queue()

        logger.info(f"Assigned {queue_item_id} to {reviewer}")
        return item

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        stats = {
            "total_items": len(self.items),
            "by_status": {},
            "by_priority": {},
            "oldest_pending": None,
            "avg_resolution_time_hours": None,
        }

        # Count by status
        for status in ReviewStatus:
            count = sum(1 for item in self.items.values() if item.status == status)
            stats["by_status"][status.value] = count

        # Count by priority
        for priority in ["high", "medium", "low"]:
            count = sum(
                1 for item in self.items.values()
                if item.priority == priority and item.status == ReviewStatus.PENDING
            )
            stats["by_priority"][priority] = count

        # Oldest pending
        pending = [
            item for item in self.items.values()
            if item.status == ReviewStatus.PENDING
        ]
        if pending:
            oldest = min(pending, key=lambda x: x.created_at)
            age = datetime.utcnow() - oldest.created_at
            stats["oldest_pending"] = {
                "queue_item_id": oldest.queue_item_id,
                "age_hours": age.total_seconds() / 3600,
            }

        # Average resolution time
        resolved = [
            item for item in self.items.values()
            if item.reviewed_at is not None
        ]
        if resolved:
            total_time = sum(
                (item.reviewed_at - item.created_at).total_seconds()
                for item in resolved
            )
            avg_seconds = total_time / len(resolved)
            stats["avg_resolution_time_hours"] = avg_seconds / 3600

        return stats

    def _cleanup_old_items(self, days: Optional[int] = None) -> int:
        """
        Remove old completed items.

        Args:
            days: Age threshold in days (default from config)

        Returns:
            Number of items removed
        """
        days = days or self.config.auto_cleanup_days
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Find old completed items
        to_remove = [
            item_id for item_id, item in self.items.items()
            if item.status in (ReviewStatus.APPROVED, ReviewStatus.REJECTED)
            and item.reviewed_at is not None
            and item.reviewed_at < cutoff
        ]

        # Remove items
        for item_id in to_remove:
            del self.items[item_id]

        if to_remove:
            self._save_queue()
            logger.info(f"Cleaned up {len(to_remove)} old items")

        return len(to_remove)

    def clear(self) -> None:
        """Clear entire queue (use with caution)."""
        self.items.clear()
        self._save_queue()
        logger.warning("Queue cleared")


__all__ = ["ReviewQueue"]
