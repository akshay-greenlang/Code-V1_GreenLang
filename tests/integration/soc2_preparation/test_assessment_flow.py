# -*- coding: utf-8 -*-
"""
Integration tests for SOC 2 Assessment Flow - SEC-009 Phase 11

End-to-end tests for the assessment workflow:
1. Create new assessment
2. Run assessment across all criteria
3. Calculate readiness scores
4. Identify gaps
5. Generate recommendations

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def integrated_app() -> FastAPI:
    """Create fully integrated FastAPI app for testing."""
    from greenlang.infrastructure.soc2_preparation.api import soc2_router

    app = FastAPI(title="SOC 2 Integration Test")
    app.include_router(soc2_router)
    return app


@pytest.fixture
def integrated_client(integrated_app: FastAPI) -> TestClient:
    """Create test client for integrated app."""
    return TestClient(integrated_app)


# ============================================================================
# Assessment Flow Tests
# ============================================================================


class TestAssessmentFlowIntegration:
    """Integration tests for assessment workflow."""

    def test_full_assessment_workflow(self, integrated_client: TestClient) -> None:
        """Test complete assessment workflow end-to-end."""
        # Step 1: Get current assessment status
        response = integrated_client.get("/api/v1/soc2/assessment")
        assert response.status_code == 200
        initial_status = response.json()
        assert "overall_score" in initial_status

        # Step 2: Run new assessment
        response = integrated_client.post(
            "/api/v1/soc2/assessment/run",
            json={"categories": ["security"]},
        )
        assert response.status_code == 202
        run_data = response.json()
        assert run_data["status"] == "running"
        assessment_id = run_data["assessment_id"]

        # Step 3: Get readiness score
        response = integrated_client.get("/api/v1/soc2/assessment/score")
        assert response.status_code == 200
        score_data = response.json()
        assert "overall_score" in score_data
        assert "category_scores" in score_data
        assert "security" in score_data["category_scores"]

        # Step 4: Get identified gaps
        response = integrated_client.get("/api/v1/soc2/assessment/gaps")
        assert response.status_code == 200
        gaps_data = response.json()
        assert "total_gaps" in gaps_data
        assert "gaps" in gaps_data

        # Step 5: Get all criteria
        response = integrated_client.get("/api/v1/soc2/assessment/criteria")
        assert response.status_code == 200
        criteria_data = response.json()
        assert isinstance(criteria_data, list)

    def test_assessment_run_with_all_categories(
        self, integrated_client: TestClient
    ) -> None:
        """Test assessment run covering all TSC categories."""
        categories = [
            "security",
            "availability",
            "confidentiality",
            "processing_integrity",
            "privacy",
        ]

        response = integrated_client.post(
            "/api/v1/soc2/assessment/run",
            json={"categories": categories},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["criteria_count"] > 0

    def test_assessment_criterion_update_workflow(
        self, integrated_client: TestClient
    ) -> None:
        """Test updating criterion status through workflow."""
        criterion_id = "CC6.1"

        # Get initial status
        response = integrated_client.get("/api/v1/soc2/assessment/criteria")
        assert response.status_code == 200

        # Update to in_progress
        response = integrated_client.put(
            f"/api/v1/soc2/assessment/criteria/{criterion_id}",
            json={"status": "in_progress", "notes": "Started implementation"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "in_progress"

        # Update to implemented
        response = integrated_client.put(
            f"/api/v1/soc2/assessment/criteria/{criterion_id}",
            json={"status": "implemented"},
        )
        assert response.status_code == 200

        # Update to verified
        response = integrated_client.put(
            f"/api/v1/soc2/assessment/criteria/{criterion_id}",
            json={"status": "verified"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "verified"

    def test_gap_to_finding_workflow(self, integrated_client: TestClient) -> None:
        """Test workflow from gap identification to finding creation."""
        # Get gaps
        response = integrated_client.get("/api/v1/soc2/assessment/gaps?severity=high")
        assert response.status_code == 200
        gaps_data = response.json()

        if gaps_data["total_gaps"] > 0:
            gap = gaps_data["gaps"][0]

            # Create finding from gap
            response = integrated_client.post(
                "/api/v1/soc2/findings",
                json={
                    "title": gap["title"],
                    "description": gap["description"],
                    "criterion_id": gap["criterion_id"],
                    "severity": gap["severity"],
                    "source": "self_assessment",
                },
            )
            assert response.status_code == 201


# ============================================================================
# Score Calculation Tests
# ============================================================================


class TestScoreCalculationIntegration:
    """Integration tests for score calculation workflow."""

    def test_score_reflects_criteria_status(
        self, integrated_client: TestClient
    ) -> None:
        """Test that score calculation reflects criteria status."""
        # Get score
        response = integrated_client.get("/api/v1/soc2/assessment/score")
        assert response.status_code == 200
        score_data = response.json()

        # Score should be between 0 and 100
        assert 0 <= score_data["overall_score"] <= 100

        # All category scores should be valid
        for category, cat_score in score_data["category_scores"].items():
            assert 0 <= cat_score <= 100

    def test_score_includes_all_criteria(self, integrated_client: TestClient) -> None:
        """Test that score includes all relevant criteria."""
        response = integrated_client.get("/api/v1/soc2/assessment/score")
        assert response.status_code == 200
        score_data = response.json()

        # Should have criteria scores
        assert len(score_data["criteria_scores"]) > 0

    def test_recommendations_based_on_score(
        self, integrated_client: TestClient
    ) -> None:
        """Test that recommendations are generated based on score."""
        response = integrated_client.get(
            "/api/v1/soc2/assessment/score?include_recommendations=true"
        )
        assert response.status_code == 200
        score_data = response.json()

        # If score is below 100, should have recommendations
        if score_data["overall_score"] < 100:
            assert len(score_data["recommendations"]) > 0


# ============================================================================
# Gap Analysis Integration Tests
# ============================================================================


class TestGapAnalysisIntegration:
    """Integration tests for gap analysis workflow."""

    def test_gaps_linked_to_criteria(self, integrated_client: TestClient) -> None:
        """Test that gaps are properly linked to criteria."""
        response = integrated_client.get("/api/v1/soc2/assessment/gaps")
        assert response.status_code == 200
        gaps_data = response.json()

        for gap in gaps_data["gaps"]:
            # Each gap should have a valid criterion ID
            assert gap["criterion_id"]
            assert gap["criterion_id"].startswith("CC") or gap["criterion_id"].startswith("A")

    def test_gap_filtering_workflow(self, integrated_client: TestClient) -> None:
        """Test gap filtering by various criteria."""
        # Filter by severity
        response = integrated_client.get("/api/v1/soc2/assessment/gaps?severity=high")
        assert response.status_code == 200

        # Filter by status
        response = integrated_client.get("/api/v1/soc2/assessment/gaps?status=open")
        assert response.status_code == 200

        # Combined filters
        response = integrated_client.get(
            "/api/v1/soc2/assessment/gaps?severity=high&status=open"
        )
        assert response.status_code == 200
