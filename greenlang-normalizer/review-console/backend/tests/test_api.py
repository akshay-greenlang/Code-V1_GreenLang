"""
Tests for Review Console API endpoints.

This module contains integration tests for all API endpoints,
including queue management, resolution, and statistics.
"""

import pytest
from httpx import AsyncClient

from review_console.db.models import ReviewQueueItem, ReviewStatus


# ============================================================================
# Health Check Tests
# ============================================================================


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Test health check endpoint returns 200."""
    response = await async_client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert "checks" in data


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Test root endpoint returns service info."""
    response = await async_client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Review Console API"
    assert "version" in data


@pytest.mark.asyncio
async def test_ready_endpoint(async_client: AsyncClient):
    """Test readiness probe endpoint."""
    response = await async_client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True


@pytest.mark.asyncio
async def test_live_endpoint(async_client: AsyncClient):
    """Test liveness probe endpoint."""
    response = await async_client.get("/live")

    assert response.status_code == 200
    data = response.json()
    assert data["live"] is True


# ============================================================================
# Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_queue_requires_auth(async_client: AsyncClient):
    """Test that queue endpoint requires authentication."""
    response = await async_client.get("/api/queue")

    assert response.status_code == 403  # No token provided


@pytest.mark.asyncio
async def test_queue_with_invalid_token(async_client: AsyncClient):
    """Test that invalid token is rejected."""
    response = await async_client.get(
        "/api/queue",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401


# ============================================================================
# Queue Listing Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_queue_empty(async_client: AsyncClient, auth_headers: dict):
    """Test getting empty queue."""
    response = await async_client.get("/api/queue", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_get_queue_with_items(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_items: list[ReviewQueueItem],
):
    """Test getting queue with items."""
    response = await async_client.get("/api/queue", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["items"]) == 5
    assert data["page"] == 1
    assert data["total_pages"] == 1


@pytest.mark.asyncio
async def test_get_queue_pagination(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_items: list[ReviewQueueItem],
):
    """Test queue pagination."""
    response = await async_client.get(
        "/api/queue",
        params={"page": 1, "page_size": 2},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["items"]) == 2
    assert data["page"] == 1
    assert data["page_size"] == 2
    assert data["total_pages"] == 3
    assert data["has_next"] is True
    assert data["has_prev"] is False


@pytest.mark.asyncio
async def test_get_queue_filter_by_entity_type(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_items: list[ReviewQueueItem],
):
    """Test filtering queue by entity type."""
    response = await async_client.get(
        "/api/queue",
        params={"entity_type": "fuel"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    # Items 0, 2, 4 are fuel
    assert data["total"] == 3
    for item in data["items"]:
        assert item["entity_type"] == "fuel"


@pytest.mark.asyncio
async def test_get_queue_filter_by_org_id(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_items: list[ReviewQueueItem],
):
    """Test filtering queue by organization ID."""
    response = await async_client.get(
        "/api/queue",
        params={"org_id": "test-org-001"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    # Items 0, 1, 2 are org-001
    assert data["total"] == 3
    for item in data["items"]:
        assert item["org_id"] == "test-org-001"


@pytest.mark.asyncio
async def test_get_queue_filter_by_confidence(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_items: list[ReviewQueueItem],
):
    """Test filtering queue by confidence range."""
    response = await async_client.get(
        "/api/queue",
        params={"min_confidence": 0.80, "max_confidence": 0.95},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert 0.80 <= item["confidence"] <= 0.95


# ============================================================================
# Queue Item Detail Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_queue_item(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_item: ReviewQueueItem,
):
    """Test getting single queue item details."""
    response = await async_client.get(
        f"/api/queue/{sample_queue_item.id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(sample_queue_item.id)
    assert data["input_text"] == "Nat Gas"
    assert data["entity_type"] == "fuel"
    assert data["confidence"] == 0.72
    assert len(data["candidates"]) == 2
    assert data["candidates"][0]["id"] == "GL-FUEL-NATGAS"
    assert data["context"]["industry_sector"] == "energy"


@pytest.mark.asyncio
async def test_get_queue_item_not_found(
    async_client: AsyncClient,
    auth_headers: dict,
):
    """Test getting non-existent queue item."""
    response = await async_client.get(
        "/api/queue/00000000-0000-0000-0000-000000000000",
        headers=auth_headers,
    )

    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["error"] == "ITEM_NOT_FOUND"


# ============================================================================
# Resolution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_resolve_item(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_item: ReviewQueueItem,
):
    """Test resolving a queue item."""
    response = await async_client.post(
        f"/api/queue/{sample_queue_item.id}/resolve",
        headers=auth_headers,
        json={
            "selected_canonical_id": "GL-FUEL-NATGAS",
            "selected_canonical_name": "Natural gas",
            "reviewer_notes": "Exact match confirmed",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["canonical_id"] == "GL-FUEL-NATGAS"
    assert data["canonical_name"] == "Natural gas"
    assert data["notes"] == "Exact match confirmed"
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_resolve_item_not_found(
    async_client: AsyncClient,
    auth_headers: dict,
):
    """Test resolving non-existent item."""
    response = await async_client.post(
        "/api/queue/00000000-0000-0000-0000-000000000000/resolve",
        headers=auth_headers,
        json={
            "selected_canonical_id": "GL-FUEL-NATGAS",
        },
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_resolve_item_already_resolved(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_item: ReviewQueueItem,
):
    """Test resolving an already resolved item."""
    # First resolution
    await async_client.post(
        f"/api/queue/{sample_queue_item.id}/resolve",
        headers=auth_headers,
        json={"selected_canonical_id": "GL-FUEL-NATGAS"},
    )

    # Second resolution should fail
    response = await async_client.post(
        f"/api/queue/{sample_queue_item.id}/resolve",
        headers=auth_headers,
        json={"selected_canonical_id": "GL-FUEL-LNG"},
    )

    assert response.status_code == 409
    data = response.json()
    assert data["detail"]["error"] == "ITEM_ALREADY_RESOLVED"


# ============================================================================
# Rejection Tests
# ============================================================================


@pytest.mark.asyncio
async def test_reject_item(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_item: ReviewQueueItem,
):
    """Test rejecting a queue item."""
    response = await async_client.post(
        f"/api/queue/{sample_queue_item.id}/reject",
        headers=auth_headers,
        json={
            "reason": "No matching entity found in vocabulary",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"


@pytest.mark.asyncio
async def test_escalate_item(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_item: ReviewQueueItem,
):
    """Test escalating a queue item."""
    response = await async_client.post(
        f"/api/queue/{sample_queue_item.id}/reject",
        headers=auth_headers,
        json={
            "reason": "Needs expert review",
            "escalate_to": "senior-reviewer@test.com",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "escalated"
    assert data["assigned_to"] == "senior-reviewer@test.com"


# ============================================================================
# Statistics Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_stats_empty(
    async_client: AsyncClient,
    auth_headers: dict,
):
    """Test getting stats with empty queue."""
    response = await async_client.get("/api/stats", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["pending_count"] == 0
    assert data["resolved_today"] == 0


@pytest.mark.asyncio
async def test_get_stats_with_items(
    async_client: AsyncClient,
    auth_headers: dict,
    sample_queue_items: list[ReviewQueueItem],
):
    """Test getting stats with items."""
    response = await async_client.get("/api/stats", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["pending_count"] == 5
    assert "items_by_entity_type" in data
    assert "fuel" in data["items_by_entity_type"]
    assert "material" in data["items_by_entity_type"]


# ============================================================================
# Vocabulary Suggestion Tests
# ============================================================================


@pytest.mark.asyncio
async def test_suggest_vocabulary(
    async_client: AsyncClient,
    auth_headers: dict,
):
    """Test suggesting a new vocabulary entry."""
    response = await async_client.post(
        "/api/vocabulary/suggest",
        params={"org_id": "test-org-001"},
        headers=auth_headers,
        json={
            "entity_type": "fuel",
            "canonical_name": "Sustainable Aviation Fuel",
            "aliases": ["SAF", "Bio-jet fuel"],
            "source": "ICAO CORSIA eligible fuels list",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["canonical_name"] == "Sustainable Aviation Fuel"
    assert data["entity_type"] == "fuel"
    assert "SAF" in data["aliases"]
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_suggest_vocabulary_validation(
    async_client: AsyncClient,
    auth_headers: dict,
):
    """Test vocabulary suggestion validation."""
    response = await async_client.post(
        "/api/vocabulary/suggest",
        params={"org_id": "test-org-001"},
        headers=auth_headers,
        json={
            "entity_type": "fuel",
            "canonical_name": "",  # Empty name should fail
            "aliases": [],
            "source": "Test",  # Too short
        },
    )

    assert response.status_code == 422
