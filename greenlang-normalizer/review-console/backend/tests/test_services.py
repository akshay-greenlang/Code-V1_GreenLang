"""
Tests for Review Console services.

This module contains unit tests for the ReviewQueueService and
ResolutionService classes.
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from review_console.db.models import (
    ReviewQueueItem,
    Resolution,
    AuditLogEntry,
    VocabularySuggestion,
    ReviewStatus,
    AuditAction,
    SuggestionStatus,
)
from review_console.services.queue import ReviewQueueService
from review_console.services.resolution import ResolutionService, ResolutionError
from review_console.api.models import QueueFilterParams, EntityTypeEnum, ReviewStatusEnum


# ============================================================================
# ReviewQueueService Tests
# ============================================================================


class TestReviewQueueService:
    """Tests for ReviewQueueService."""

    @pytest.mark.asyncio
    async def test_get_items_empty(self, db_session: AsyncSession):
        """Test getting items from empty queue."""
        service = ReviewQueueService(db_session)
        items, total = await service.get_items()

        assert total == 0
        assert items == []

    @pytest.mark.asyncio
    async def test_get_items_with_data(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test getting items with data."""
        service = ReviewQueueService(db_session)
        items, total = await service.get_items()

        assert total == 5
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_get_items_pagination(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test pagination."""
        service = ReviewQueueService(db_session)
        items, total = await service.get_items(page=1, page_size=2)

        assert total == 5
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_get_items_filter_entity_type(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test filtering by entity type."""
        service = ReviewQueueService(db_session)
        filters = QueueFilterParams(entity_type=EntityTypeEnum.FUEL)
        items, total = await service.get_items(filters=filters)

        assert total == 3
        for item in items:
            assert item.entity_type == "fuel"

    @pytest.mark.asyncio
    async def test_get_items_filter_org_id(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test filtering by organization ID."""
        service = ReviewQueueService(db_session)
        filters = QueueFilterParams(org_id="test-org-001")
        items, total = await service.get_items(filters=filters)

        assert total == 3
        for item in items:
            assert item.org_id == "test-org-001"

    @pytest.mark.asyncio
    async def test_get_items_filter_confidence(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test filtering by confidence range."""
        service = ReviewQueueService(db_session)
        filters = QueueFilterParams(min_confidence=0.80)
        items, total = await service.get_items(filters=filters)

        for item in items:
            assert item.confidence >= 0.80

    @pytest.mark.asyncio
    async def test_get_items_search(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test search functionality."""
        service = ReviewQueueService(db_session)
        filters = QueueFilterParams(search="Nat")
        items, total = await service.get_items(filters=filters)

        assert total == 1
        assert "Nat" in items[0].input_text

    @pytest.mark.asyncio
    async def test_get_item_by_id(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test getting item by ID."""
        service = ReviewQueueService(db_session)
        item = await service.get_item_by_id(sample_queue_item.id)

        assert item is not None
        assert item.id == sample_queue_item.id
        assert item.input_text == "Nat Gas"

    @pytest.mark.asyncio
    async def test_get_item_by_id_not_found(self, db_session: AsyncSession):
        """Test getting non-existent item."""
        service = ReviewQueueService(db_session)
        item = await service.get_item_by_id("00000000-0000-0000-0000-000000000000")

        assert item is None

    @pytest.mark.asyncio
    async def test_assign_item(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test assigning item to reviewer."""
        service = ReviewQueueService(db_session)
        item = await service.assign_item(
            sample_queue_item.id,
            "reviewer@test.com",
        )

        assert item is not None
        assert item.assigned_to == "reviewer@test.com"
        assert item.status == ReviewStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_status(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test updating item status."""
        service = ReviewQueueService(db_session)
        item = await service.update_status(
            sample_queue_item.id,
            ReviewStatus.RESOLVED,
            resolved_by="reviewer@test.com",
        )

        assert item is not None
        assert item.status == ReviewStatus.RESOLVED
        assert item.resolved_by == "reviewer@test.com"
        assert item.resolved_at is not None

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test getting queue statistics."""
        service = ReviewQueueService(db_session)
        stats = await service.get_stats()

        assert stats["pending_count"] == 5
        assert stats["in_progress_count"] == 0
        assert stats["items_by_entity_type"]["fuel"] == 3
        assert stats["items_by_entity_type"]["material"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_by_org(
        self,
        db_session: AsyncSession,
        sample_queue_items: list[ReviewQueueItem],
    ):
        """Test getting stats filtered by organization."""
        service = ReviewQueueService(db_session)
        stats = await service.get_stats(org_id="test-org-001")

        assert stats["pending_count"] == 3

    @pytest.mark.asyncio
    async def test_create_audit_entry(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test creating audit log entry."""
        service = ReviewQueueService(db_session)
        entry = await service.create_audit_entry(
            item_id=sample_queue_item.id,
            action=AuditAction.ITEM_VIEWED,
            actor_id="viewer@test.com",
            details={"test": True},
        )

        assert entry is not None
        assert entry.item_id == sample_queue_item.id
        assert entry.action == AuditAction.ITEM_VIEWED
        assert entry.actor_id == "viewer@test.com"


# ============================================================================
# ResolutionService Tests
# ============================================================================


class TestResolutionService:
    """Tests for ResolutionService."""

    @pytest.mark.asyncio
    async def test_resolve_item(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test resolving a queue item."""
        service = ResolutionService(db_session)
        resolution = await service.resolve_item(
            item_id=sample_queue_item.id,
            canonical_id="GL-FUEL-NATGAS",
            canonical_name="Natural gas",
            reviewer_id="reviewer@test.com",
            notes="Exact match confirmed",
        )

        assert resolution is not None
        assert resolution.canonical_id == "GL-FUEL-NATGAS"
        assert resolution.canonical_name == "Natural gas"
        assert resolution.reviewer_id == "reviewer@test.com"
        assert resolution.notes == "Exact match confirmed"

        # Verify item status updated
        queue_service = ReviewQueueService(db_session)
        item = await queue_service.get_item_by_id(sample_queue_item.id)
        assert item.status == ReviewStatus.RESOLVED
        assert item.resolved_by == "reviewer@test.com"

    @pytest.mark.asyncio
    async def test_resolve_item_not_found(self, db_session: AsyncSession):
        """Test resolving non-existent item."""
        service = ResolutionService(db_session)

        with pytest.raises(ResolutionError) as exc_info:
            await service.resolve_item(
                item_id="00000000-0000-0000-0000-000000000000",
                canonical_id="GL-FUEL-NATGAS",
                canonical_name="Natural gas",
                reviewer_id="reviewer@test.com",
            )

        assert exc_info.value.code == "ITEM_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_resolve_item_already_resolved(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test resolving already resolved item."""
        service = ResolutionService(db_session)

        # First resolution
        await service.resolve_item(
            item_id=sample_queue_item.id,
            canonical_id="GL-FUEL-NATGAS",
            canonical_name="Natural gas",
            reviewer_id="reviewer@test.com",
        )

        # Second resolution should fail
        with pytest.raises(ResolutionError) as exc_info:
            await service.resolve_item(
                item_id=sample_queue_item.id,
                canonical_id="GL-FUEL-LNG",
                canonical_name="LNG",
                reviewer_id="reviewer2@test.com",
            )

        assert exc_info.value.code == "ITEM_ALREADY_RESOLVED"

    @pytest.mark.asyncio
    async def test_reject_item(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test rejecting a queue item."""
        service = ResolutionService(db_session)
        item = await service.reject_item(
            item_id=sample_queue_item.id,
            reason="No matching entity found",
            reviewer_id="reviewer@test.com",
        )

        assert item.status == ReviewStatus.REJECTED
        assert item.resolved_at is not None
        assert item.resolved_by == "reviewer@test.com"

    @pytest.mark.asyncio
    async def test_escalate_item(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test escalating a queue item."""
        service = ResolutionService(db_session)
        item = await service.reject_item(
            item_id=sample_queue_item.id,
            reason="Needs expert review",
            reviewer_id="reviewer@test.com",
            escalate_to="senior@test.com",
        )

        assert item.status == ReviewStatus.ESCALATED
        assert item.assigned_to == "senior@test.com"
        assert item.resolved_at is None  # Not resolved, just escalated

    @pytest.mark.asyncio
    async def test_create_vocabulary_suggestion(
        self,
        db_session: AsyncSession,
    ):
        """Test creating vocabulary suggestion."""
        service = ResolutionService(db_session)
        suggestion = await service.create_vocabulary_suggestion(
            entity_type="fuel",
            canonical_name="Sustainable Aviation Fuel",
            aliases=["SAF", "Bio-jet fuel"],
            source="ICAO CORSIA eligible fuels list",
            suggested_by="reviewer@test.com",
            org_id="test-org-001",
        )

        assert suggestion is not None
        assert suggestion.canonical_name == "Sustainable Aviation Fuel"
        assert suggestion.entity_type == "fuel"
        assert "SAF" in suggestion.aliases
        assert suggestion.status == SuggestionStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_suggestion_by_id(
        self,
        db_session: AsyncSession,
    ):
        """Test getting vocabulary suggestion by ID."""
        service = ResolutionService(db_session)

        # Create suggestion
        suggestion = await service.create_vocabulary_suggestion(
            entity_type="fuel",
            canonical_name="Test Fuel",
            aliases=["TF"],
            source="Test source",
            suggested_by="reviewer@test.com",
            org_id="test-org-001",
        )

        # Retrieve it
        retrieved = await service.get_suggestion_by_id(suggestion.id)

        assert retrieved is not None
        assert retrieved.id == suggestion.id
        assert retrieved.canonical_name == "Test Fuel"

    @pytest.mark.asyncio
    async def test_get_resolution_by_item_id(
        self,
        db_session: AsyncSession,
        sample_queue_item: ReviewQueueItem,
    ):
        """Test getting resolution by item ID."""
        service = ResolutionService(db_session)

        # Create resolution
        await service.resolve_item(
            item_id=sample_queue_item.id,
            canonical_id="GL-FUEL-NATGAS",
            canonical_name="Natural gas",
            reviewer_id="reviewer@test.com",
        )

        # Retrieve it
        resolution = await service.get_resolution_by_item_id(sample_queue_item.id)

        assert resolution is not None
        assert resolution.item_id == sample_queue_item.id
        assert resolution.canonical_id == "GL-FUEL-NATGAS"
