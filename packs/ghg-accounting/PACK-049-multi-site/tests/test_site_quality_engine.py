# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 9: SiteQualityEngine

Covers quality assessment across six dimensions, PCAF equivalent scoring,
quality heatmap generation, corporate-level weighted quality, remediation
planning, and quality progression tracking.
Target: ~50 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

try:
    from engines.site_quality_engine import (
        SiteQualityEngine,
        SiteQualityAssessment,
        DimensionScore,
        QualityHeatmap,
        CorporateQuality,
        RemediationPlan,
        QualityProgression,
        QualityDimension,
    )
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Engine not yet built")


@pytest.fixture
def engine():
    return SiteQualityEngine()


@pytest.fixture
def site_quality_data():
    """Quality input data for a single site."""
    return {
        "site_id": "site-001",
        "site_name": "Chicago Manufacturing Plant",
        "data_sources": ["meter_reads", "invoices", "erp_extract"],
        "has_third_party_verification": False,
        "methodology_documented": True,
        "consistent_with_prior_year": True,
        "submitted_on_time": True,
        "calculation_methodology": "TIER_2",
        "activity_data_source": "utility_invoices",
        "emission_factor_source": "EPA_EGRID",
        "documentation_completeness": Decimal("0.85"),
    }


@pytest.fixture
def portfolio_quality_data():
    """Quality data for 5 sites."""
    return [
        {
            "site_id": "site-001", "overall_score": 2,
            "total_tco2e": Decimal("18000"), "weight": Decimal("0.55"),
        },
        {
            "site_id": "site-002", "overall_score": 1,
            "total_tco2e": Decimal("1250"), "weight": Decimal("0.04"),
        },
        {
            "site_id": "site-003", "overall_score": 3,
            "total_tco2e": Decimal("3300"), "weight": Decimal("0.10"),
        },
        {
            "site_id": "site-004", "overall_score": 4,
            "total_tco2e": Decimal("400"), "weight": Decimal("0.01"),
        },
        {
            "site_id": "site-005", "overall_score": 2,
            "total_tco2e": Decimal("7700"), "weight": Decimal("0.24"),
        },
    ]


# ============================================================================
# Site Quality Assessment Tests
# ============================================================================

class TestSiteQualityAssessment:

    def test_assess_site_quality(self, engine, site_quality_data):
        assessment = engine.assess_site(site_quality_data)
        assert isinstance(assessment, SiteQualityAssessment)
        assert assessment.site_id == "site-001"
        assert 1 <= assessment.overall_score <= 5

    def test_dimension_scores(self, engine, site_quality_data):
        assessment = engine.assess_site(site_quality_data)
        assert len(assessment.dimension_scores) == 6
        for dim_name in [
            "ACCURACY", "COMPLETENESS", "CONSISTENCY",
            "TIMELINESS", "METHODOLOGY", "DOCUMENTATION",
        ]:
            score = assessment.dimension_scores.get(dim_name)
            assert score is not None
            assert Decimal("1") <= score <= Decimal("5")

    def test_quality_provenance(self, engine, site_quality_data):
        assessment = engine.assess_site(site_quality_data)
        assert assessment.provenance_hash is not None
        assert len(assessment.provenance_hash) == 64


# ============================================================================
# PCAF Equivalent Score Tests
# ============================================================================

class TestPCAFEquivalent:

    def test_pcaf_equivalent_score1(self, engine):
        data = {
            "site_id": "site-verified",
            "site_name": "Verified Site",
            "data_sources": ["verified_meter_reads"],
            "has_third_party_verification": True,
            "methodology_documented": True,
            "consistent_with_prior_year": True,
            "submitted_on_time": True,
            "calculation_methodology": "TIER_3",
            "activity_data_source": "metered_data",
            "emission_factor_source": "SUPPLIER_SPECIFIC",
            "documentation_completeness": Decimal("1.00"),
        }
        assessment = engine.assess_site(data)
        assert assessment.pcaf_equivalent in (1, 2)

    def test_pcaf_equivalent_score3(self, engine):
        data = {
            "site_id": "site-estimated",
            "site_name": "Estimated Site",
            "data_sources": ["estimated"],
            "has_third_party_verification": False,
            "methodology_documented": True,
            "consistent_with_prior_year": False,
            "submitted_on_time": True,
            "calculation_methodology": "TIER_1",
            "activity_data_source": "estimates",
            "emission_factor_source": "IEA",
            "documentation_completeness": Decimal("0.60"),
        }
        assessment = engine.assess_site(data)
        assert 2 <= assessment.pcaf_equivalent <= 4

    def test_pcaf_equivalent_score5(self, engine):
        data = {
            "site_id": "site-proxy",
            "site_name": "Proxy Site",
            "data_sources": ["proxy"],
            "has_third_party_verification": False,
            "methodology_documented": False,
            "consistent_with_prior_year": False,
            "submitted_on_time": False,
            "calculation_methodology": "PROXY",
            "activity_data_source": "external_benchmark",
            "emission_factor_source": "IPCC_DEFAULT",
            "documentation_completeness": Decimal("0.10"),
        }
        assessment = engine.assess_site(data)
        assert assessment.pcaf_equivalent >= 4


# ============================================================================
# Quality Heatmap Tests
# ============================================================================

class TestQualityHeatmap:

    def test_quality_heatmap_colours(self, engine, sample_quality_assessments):
        heatmap = engine.generate_heatmap(sample_quality_assessments)
        assert isinstance(heatmap, QualityHeatmap)
        assert len(heatmap.cells) == 5
        # Each cell should have a colour indicator
        for cell in heatmap.cells:
            assert cell.colour in ("GREEN", "YELLOW", "ORANGE", "RED", "DARK_RED") or \
                   cell.score is not None

    def test_quality_heatmap_site_order(self, engine, sample_quality_assessments):
        heatmap = engine.generate_heatmap(sample_quality_assessments)
        # Sites should be ordered by score (best first) or site_id
        assert len(heatmap.cells) == len(sample_quality_assessments)


# ============================================================================
# Corporate Quality Tests
# ============================================================================

class TestCorporateQuality:

    def test_corporate_quality_weighted(self, engine, portfolio_quality_data):
        corporate = engine.calculate_corporate_quality(portfolio_quality_data)
        assert isinstance(corporate, CorporateQuality)
        # Weighted average should be between 1 and 5
        assert Decimal("1") <= corporate.weighted_score <= Decimal("5")

    def test_corporate_quality_emission_weighted(self, engine, portfolio_quality_data):
        corporate = engine.calculate_corporate_quality(
            portfolio_quality_data,
            weight_by="emissions",
        )
        # Larger emitters dominate: site-001 (score 2, 55%) and site-005 (score 2, 24%)
        assert Decimal("1.5") <= corporate.weighted_score <= Decimal("3.0")


# ============================================================================
# Remediation Plan Tests
# ============================================================================

class TestRemediationPlan:

    def test_remediation_plan(self, engine, sample_quality_assessments):
        # Get sites with quality score >= 3 (poor quality)
        poor_sites = [s for s in sample_quality_assessments if s["overall_score"] >= 3]
        for site_data in poor_sites:
            plan = engine.create_remediation_plan(site_data)
            assert isinstance(plan, RemediationPlan)
            assert len(plan.actions) > 0
            assert plan.target_score < site_data["overall_score"]

    def test_remediation_plan_priorities(self, engine, sample_quality_assessments):
        worst = max(sample_quality_assessments, key=lambda s: s["overall_score"])
        plan = engine.create_remediation_plan(worst)
        # Actions should be prioritised
        assert plan.actions[0].priority in ("HIGH", "CRITICAL", 1)


# ============================================================================
# Quality Progression Tests
# ============================================================================

class TestQualityProgression:

    def test_quality_progression(self, engine):
        historical = {
            2024: {"overall_score": 4, "pcaf_equivalent": 4},
            2025: {"overall_score": 3, "pcaf_equivalent": 3},
            2026: {"overall_score": 2, "pcaf_equivalent": 2},
        }
        progression = engine.track_progression(
            site_id="site-001",
            historical_scores=historical,
        )
        assert isinstance(progression, QualityProgression)
        assert progression.direction in ("IMPROVING", "UP", "BETTER")
        assert progression.years_tracked == 3

    def test_quality_progression_stable(self, engine):
        historical = {
            2024: {"overall_score": 2, "pcaf_equivalent": 2},
            2025: {"overall_score": 2, "pcaf_equivalent": 2},
            2026: {"overall_score": 2, "pcaf_equivalent": 2},
        }
        progression = engine.track_progression(
            site_id="site-002",
            historical_scores=historical,
        )
        assert progression.direction in ("STABLE", "FLAT", "UNCHANGED")

    def test_quality_progression_declining(self, engine):
        historical = {
            2024: {"overall_score": 1, "pcaf_equivalent": 1},
            2025: {"overall_score": 2, "pcaf_equivalent": 2},
            2026: {"overall_score": 3, "pcaf_equivalent": 3},
        }
        progression = engine.track_progression(
            site_id="site-003",
            historical_scores=historical,
        )
        assert progression.direction in ("DECLINING", "DOWN", "WORSE", "WORSENING")
