# -*- coding: utf-8 -*-
"""
Review Actions

Approve, reject, merge, split operations for review queue items.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models import ReviewQueueItem, ReviewAction, ReviewStatus
from ..exceptions import InvalidActionError, ReviewAlreadyCompletedError
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ReviewActions:
    """
    Review action handlers for queue items.

    Available Actions:
    - Approve: Accept suggested match
    - Reject: Reject all suggestions
    - Merge: Merge multiple candidates
    - Split: Split into multiple entities
    - Request Info: Request additional information
    """

    def approve(
        self,
        item: ReviewQueueItem,
        canonical_id: str,
        canonical_name: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """
        Approve match and set canonical ID.

        Args:
            item: Queue item
            canonical_id: Approved canonical entity ID
            canonical_name: Approved canonical name
            reviewer: Reviewer identifier
            notes: Optional review notes

        Returns:
            Updated queue item

        Raises:
            ReviewAlreadyCompletedError: If already reviewed
        """
        self._check_not_completed(item)

        item.status = ReviewStatus.APPROVED
        item.resolved_canonical_id = canonical_id
        item.resolved_canonical_name = canonical_name
        item.action_taken = ReviewAction.APPROVE
        item.reviewed_by = reviewer
        item.reviewed_at = DeterministicClock.utcnow()
        item.reviewer_notes = notes

        item.action_details = {
            "canonical_id": canonical_id,
            "canonical_name": canonical_name,
        }

        logger.info(
            f"Approved: {item.queue_item_id} -> {canonical_id} by {reviewer}"
        )

        return item

    def reject(
        self,
        item: ReviewQueueItem,
        reason: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """
        Reject all match suggestions.

        Args:
            item: Queue item
            reason: Rejection reason
            reviewer: Reviewer identifier
            notes: Optional review notes

        Returns:
            Updated queue item

        Raises:
            ReviewAlreadyCompletedError: If already reviewed
        """
        self._check_not_completed(item)

        item.status = ReviewStatus.REJECTED
        item.action_taken = ReviewAction.REJECT
        item.reviewed_by = reviewer
        item.reviewed_at = DeterministicClock.utcnow()
        item.reviewer_notes = notes or reason

        item.action_details = {
            "reason": reason,
        }

        logger.info(
            f"Rejected: {item.queue_item_id} by {reviewer} (reason: {reason})"
        )

        return item

    def merge(
        self,
        item: ReviewQueueItem,
        canonical_ids: List[str],
        merged_canonical_id: str,
        merged_canonical_name: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """
        Merge multiple candidate entities.

        Args:
            item: Queue item
            canonical_ids: List of canonical IDs to merge
            merged_canonical_id: New merged canonical ID
            merged_canonical_name: New merged canonical name
            reviewer: Reviewer identifier
            notes: Optional review notes

        Returns:
            Updated queue item
        """
        self._check_not_completed(item)

        item.status = ReviewStatus.MERGED
        item.resolved_canonical_id = merged_canonical_id
        item.resolved_canonical_name = merged_canonical_name
        item.action_taken = ReviewAction.MERGE
        item.reviewed_by = reviewer
        item.reviewed_at = DeterministicClock.utcnow()
        item.reviewer_notes = notes

        item.action_details = {
            "source_canonical_ids": canonical_ids,
            "merged_canonical_id": merged_canonical_id,
            "merged_canonical_name": merged_canonical_name,
        }

        logger.info(
            f"Merged: {item.queue_item_id} -> {merged_canonical_id} "
            f"(from {len(canonical_ids)} entities) by {reviewer}"
        )

        return item

    def split(
        self,
        item: ReviewQueueItem,
        split_entities: List[Dict[str, str]],
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """
        Split item into multiple entities.

        Args:
            item: Queue item
            split_entities: List of split entities with canonical_id and canonical_name
            reviewer: Reviewer identifier
            notes: Optional review notes

        Returns:
            Updated queue item
        """
        self._check_not_completed(item)

        item.status = ReviewStatus.SPLIT
        item.action_taken = ReviewAction.SPLIT
        item.reviewed_by = reviewer
        item.reviewed_at = DeterministicClock.utcnow()
        item.reviewer_notes = notes

        item.action_details = {
            "split_count": len(split_entities),
            "split_entities": split_entities,
        }

        logger.info(
            f"Split: {item.queue_item_id} into {len(split_entities)} entities by {reviewer}"
        )

        return item

    def request_info(
        self,
        item: ReviewQueueItem,
        info_requested: str,
        reviewer: str,
        notes: Optional[str] = None,
    ) -> ReviewQueueItem:
        """
        Request additional information.

        Args:
            item: Queue item
            info_requested: Description of information needed
            reviewer: Reviewer identifier
            notes: Optional review notes

        Returns:
            Updated queue item
        """
        # Don't check completed - can request info on pending items

        item.action_taken = ReviewAction.REQUEST_INFO
        item.reviewer_notes = notes or info_requested
        item.additional_context = info_requested

        item.action_details = {
            "info_requested": info_requested,
            "requested_at": DeterministicClock.utcnow().isoformat(),
            "requested_by": reviewer,
        }

        logger.info(
            f"Info requested: {item.queue_item_id} by {reviewer}"
        )

        return item

    def _check_not_completed(self, item: ReviewQueueItem) -> None:
        """
        Check that item is not already completed.

        Args:
            item: Queue item

        Raises:
            ReviewAlreadyCompletedError: If already reviewed
        """
        if item.status in (ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.MERGED):
            raise ReviewAlreadyCompletedError(
                f"Queue item {item.queue_item_id} already completed with status: {item.status}",
                details={
                    "queue_item_id": item.queue_item_id,
                    "status": item.status.value,
                    "reviewed_by": item.reviewed_by,
                    "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
                }
            )


__all__ = ["ReviewActions"]
