"""
Tests for Review Console Pydantic models.

This module contains unit tests for request/response model validation.
"""

import pytest
from datetime import datetime, timezone

from pydantic import ValidationError

from review_console.api.models import (
    # Request models
    ResolveItemRequest,
    RejectItemRequest,
    VocabularySuggestionRequest,
    QueueFilterParams,
    # Response models
    ReviewQueueItemResponse,
    CandidateInfo,
    ContextInfo,
    QueueStatsResponse,
    # Enums
    EntityTypeEnum,
    ReviewStatusEnum,
)


class TestResolveItemRequest:
    """Tests for ResolveItemRequest model."""

    def test_valid_request(self):
        """Test valid resolve request."""
        request = ResolveItemRequest(
            selected_canonical_id="GL-FUEL-NATGAS",
            selected_canonical_name="Natural gas",
            reviewer_notes="Exact match confirmed",
        )

        assert request.selected_canonical_id == "GL-FUEL-NATGAS"
        assert request.selected_canonical_name == "Natural gas"
        assert request.reviewer_notes == "Exact match confirmed"

    def test_minimal_request(self):
        """Test minimal resolve request."""
        request = ResolveItemRequest(
            selected_canonical_id="GL-FUEL-NATGAS",
        )

        assert request.selected_canonical_id == "GL-FUEL-NATGAS"
        assert request.selected_canonical_name is None
        assert request.reviewer_notes is None

    def test_empty_canonical_id(self):
        """Test that empty canonical ID is rejected."""
        with pytest.raises(ValidationError):
            ResolveItemRequest(
                selected_canonical_id="",
            )

    def test_confidence_override_bounds(self):
        """Test confidence override validation."""
        # Valid values
        request = ResolveItemRequest(
            selected_canonical_id="GL-FUEL-NATGAS",
            confidence_override=0.95,
        )
        assert request.confidence_override == 0.95

        # Invalid - too high
        with pytest.raises(ValidationError):
            ResolveItemRequest(
                selected_canonical_id="GL-FUEL-NATGAS",
                confidence_override=1.5,
            )

        # Invalid - negative
        with pytest.raises(ValidationError):
            ResolveItemRequest(
                selected_canonical_id="GL-FUEL-NATGAS",
                confidence_override=-0.1,
            )


class TestRejectItemRequest:
    """Tests for RejectItemRequest model."""

    def test_valid_request(self):
        """Test valid reject request."""
        request = RejectItemRequest(
            reason="No matching entity found in vocabulary",
        )

        assert "No matching" in request.reason
        assert request.escalate_to is None

    def test_escalate_request(self):
        """Test reject with escalation."""
        request = RejectItemRequest(
            reason="Needs expert review for this complex case",
            escalate_to="senior@test.com",
        )

        assert request.escalate_to == "senior@test.com"

    def test_reason_too_short(self):
        """Test that too short reason is rejected."""
        with pytest.raises(ValidationError):
            RejectItemRequest(
                reason="short",  # Less than 10 chars
            )


class TestVocabularySuggestionRequest:
    """Tests for VocabularySuggestionRequest model."""

    def test_valid_request(self):
        """Test valid vocabulary suggestion."""
        request = VocabularySuggestionRequest(
            entity_type=EntityTypeEnum.FUEL,
            canonical_name="Sustainable Aviation Fuel",
            aliases=["SAF", "Bio-jet fuel"],
            source="ICAO CORSIA eligible fuels list",
        )

        assert request.entity_type == EntityTypeEnum.FUEL
        assert request.canonical_name == "Sustainable Aviation Fuel"
        assert len(request.aliases) == 2

    def test_alias_normalization(self):
        """Test that aliases are normalized."""
        request = VocabularySuggestionRequest(
            entity_type=EntityTypeEnum.FUEL,
            canonical_name="Test Fuel",
            aliases=["  SAF  ", "saf", "SAF", "Bio-jet"],  # Duplicates and whitespace
            source="Test source for validation",
        )

        # Duplicates should be removed, whitespace stripped
        assert len(request.aliases) == 2
        assert "SAF" in request.aliases
        assert "Bio-jet" in request.aliases

    def test_source_too_short(self):
        """Test that too short source is rejected."""
        with pytest.raises(ValidationError):
            VocabularySuggestionRequest(
                entity_type=EntityTypeEnum.FUEL,
                canonical_name="Test Fuel",
                aliases=[],
                source="short",  # Less than 10 chars
            )


class TestCandidateInfo:
    """Tests for CandidateInfo model."""

    def test_valid_candidate(self):
        """Test valid candidate info."""
        candidate = CandidateInfo(
            id="GL-FUEL-NATGAS",
            name="Natural gas",
            score=0.92,
            source="fuels_vocab",
        )

        assert candidate.id == "GL-FUEL-NATGAS"
        assert candidate.score == 0.92

    def test_score_bounds(self):
        """Test score validation bounds."""
        # Valid bounds
        CandidateInfo(id="X", name="X", score=0.0, source="X")
        CandidateInfo(id="X", name="X", score=1.0, source="X")

        # Invalid - too high
        with pytest.raises(ValidationError):
            CandidateInfo(id="X", name="X", score=1.1, source="X")

        # Invalid - negative
        with pytest.raises(ValidationError):
            CandidateInfo(id="X", name="X", score=-0.1, source="X")


class TestQueueFilterParams:
    """Tests for QueueFilterParams model."""

    def test_empty_filters(self):
        """Test empty filter params."""
        filters = QueueFilterParams()

        assert filters.entity_type is None
        assert filters.org_id is None
        assert filters.status is None

    def test_all_filters(self):
        """Test all filter params."""
        filters = QueueFilterParams(
            entity_type=EntityTypeEnum.FUEL,
            org_id="org-123",
            status=ReviewStatusEnum.PENDING,
            min_confidence=0.5,
            max_confidence=0.9,
            search="natural",
        )

        assert filters.entity_type == EntityTypeEnum.FUEL
        assert filters.org_id == "org-123"
        assert filters.min_confidence == 0.5

    def test_confidence_bounds(self):
        """Test confidence filter bounds."""
        # Valid
        QueueFilterParams(min_confidence=0.0, max_confidence=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            QueueFilterParams(min_confidence=1.5)

        with pytest.raises(ValidationError):
            QueueFilterParams(max_confidence=-0.1)


class TestQueueStatsResponse:
    """Tests for QueueStatsResponse model."""

    def test_valid_stats(self):
        """Test valid stats response."""
        stats = QueueStatsResponse(
            pending_count=42,
            in_progress_count=5,
            resolved_today=28,
            rejected_today=3,
            escalated_count=2,
            avg_resolution_time_seconds=145.5,
            avg_confidence=0.68,
            oldest_pending_age_hours=12.5,
            items_by_entity_type={"fuel": 25, "material": 12},
            items_by_org={"org-123": 20, "org-456": 15},
        )

        assert stats.pending_count == 42
        assert stats.items_by_entity_type["fuel"] == 25

    def test_minimal_stats(self):
        """Test minimal stats response."""
        stats = QueueStatsResponse(
            pending_count=0,
            in_progress_count=0,
            resolved_today=0,
            rejected_today=0,
            escalated_count=0,
        )

        assert stats.pending_count == 0
        assert stats.avg_resolution_time_seconds is None


class TestReviewQueueItemResponse:
    """Tests for ReviewQueueItemResponse model."""

    def test_valid_response(self):
        """Test valid item response."""
        now = datetime.now(timezone.utc)
        response = ReviewQueueItemResponse(
            id="550e8400-e29b-41d4-a716-446655440000",
            input_text="Nat Gas",
            entity_type="fuel",
            org_id="org-123",
            confidence=0.72,
            status=ReviewStatusEnum.PENDING,
            priority=0,
            created_at=now,
            updated_at=now,
        )

        assert response.id == "550e8400-e29b-41d4-a716-446655440000"
        assert response.input_text == "Nat Gas"
        assert response.status == ReviewStatusEnum.PENDING
