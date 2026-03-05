# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Recommendation Engine.

Tests recommendation generation, prioritization, sector best practices,
and implementation guides with 16+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    TCFDPillar,
    SectorType,
    SECTOR_TRANSITION_PROFILES,
)
from services.models import (
    Recommendation,
    _new_id,
)


# ===========================================================================
# Recommendation Generation
# ===========================================================================

class TestRecommendationGeneration:
    """Test recommendation generation."""

    def test_recommendation_creation(self, sample_recommendation):
        assert sample_recommendation.title == "Conduct Multi-Scenario Analysis"
        assert sample_recommendation.category == "strategy"

    def test_recommendation_has_id(self, sample_recommendation):
        assert len(sample_recommendation.id) == 36

    def test_recommendation_description(self, sample_recommendation):
        assert len(sample_recommendation.description) > 0

    @pytest.mark.parametrize("category", [
        "governance", "strategy", "risk_management", "metrics",
    ])
    def test_recommendation_categories(self, category):
        rec = Recommendation(
            org_id=_new_id(),
            category=category,
            title=f"Test {category}",
        )
        assert rec.category == category

    def test_recommendation_timestamps(self, sample_recommendation):
        assert sample_recommendation.created_at is not None


# ===========================================================================
# Prioritization
# ===========================================================================

class TestPrioritization:
    """Test recommendation prioritization."""

    def test_priority_score(self, sample_recommendation):
        assert sample_recommendation.priority == 1

    @pytest.mark.parametrize("priority", range(1, 6))
    def test_priority_range(self, priority):
        rec = Recommendation(
            org_id=_new_id(),
            priority=priority,
            title=f"Priority {priority}",
        )
        assert rec.priority == priority

    def test_priority_sorting(self):
        recs = [
            Recommendation(org_id=_new_id(), priority=3, title="Medium"),
            Recommendation(org_id=_new_id(), priority=1, title="Highest"),
            Recommendation(org_id=_new_id(), priority=5, title="Lowest"),
        ]
        sorted_recs = sorted(recs, key=lambda r: r.priority)
        assert sorted_recs[0].title == "Highest"
        assert sorted_recs[-1].title == "Lowest"

    @pytest.mark.parametrize("impact", ["low", "medium", "high"])
    def test_estimated_impact(self, impact):
        rec = Recommendation(
            org_id=_new_id(),
            title="Test",
            estimated_impact=impact,
        )
        assert rec.estimated_impact == impact

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_estimated_effort(self, effort):
        rec = Recommendation(
            org_id=_new_id(),
            title="Test",
            estimated_effort=effort,
        )
        assert rec.estimated_effort == effort


# ===========================================================================
# Sector Best Practices
# ===========================================================================

class TestSectorBestPractices:
    """Test sector-specific best practice recommendations."""

    @pytest.mark.parametrize("sector", list(SectorType))
    def test_sector_has_decarbonization_pathway(self, sector):
        profile = SECTOR_TRANSITION_PROFILES[sector]
        assert "decarbonization_pathway" in profile
        assert len(profile["decarbonization_pathway"]) > 0

    def test_energy_sector_best_practices(self):
        profile = SECTOR_TRANSITION_PROFILES[SectorType.ENERGY]
        pathway = profile["decarbonization_pathway"].lower()
        assert "renewables" in pathway or "ccs" in pathway


# ===========================================================================
# Implementation Guides
# ===========================================================================

class TestImplementationGuides:
    """Test implementation guidance."""

    def test_guidance_present(self, sample_recommendation):
        assert len(sample_recommendation.implementation_guidance) > 0

    def test_guidance_steps(self, sample_recommendation):
        guidance = sample_recommendation.implementation_guidance
        assert "1." in guidance  # Has numbered steps

    def test_empty_guidance(self):
        rec = Recommendation(
            org_id=_new_id(),
            title="Minimal Recommendation",
        )
        assert rec.implementation_guidance == ""
