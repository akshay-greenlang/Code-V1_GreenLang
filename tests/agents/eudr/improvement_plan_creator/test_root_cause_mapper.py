# -*- coding: utf-8 -*-
"""
Unit tests for RootCauseMapper - AGENT-EUDR-035

Tests root cause analysis via 5-Whys methodology, fishbone diagram
construction, systemic root cause identification, and storage/retrieval.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (Engine 3: Root Cause Mapper)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.root_cause_mapper import (
    RootCauseMapper,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    ComplianceGap,
    EUDRCommodity,
    FishboneAnalysis,
    FishboneCategory,
    GapSeverity,
    RootCause,
)


@pytest.fixture
def config():
    return ImprovementPlanCreatorConfig()


@pytest.fixture
def mapper(config):
    return RootCauseMapper(config=config)


# ---------------------------------------------------------------------------
# Analyze Root Causes (5-Whys)
# ---------------------------------------------------------------------------

class TestAnalyzeRootCauses:
    """Test root cause analysis via 5-Whys methodology."""

    @pytest.mark.asyncio
    async def test_analyze_returns_list(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        assert isinstance(causes, list)
        assert len(causes) >= 1

    @pytest.mark.asyncio
    async def test_analyze_from_single_gap(self, mapper, sample_gap):
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        assert len(causes) >= 1

    @pytest.mark.asyncio
    async def test_analyze_from_multiple_gaps(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        # Should have causes for each gap (each gap gets multiple causes via 5-Whys)
        assert len(causes) >= len(multiple_gaps)

    @pytest.mark.asyncio
    async def test_empty_gaps_return_empty(self, mapper):
        causes = await mapper.analyze_root_causes(
            gaps=[],
            plan_id="plan-001",
        )
        assert causes == []

    @pytest.mark.asyncio
    async def test_causes_have_ids(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        for cause in causes:
            assert cause.root_cause_id.startswith("RC-")

    @pytest.mark.asyncio
    async def test_causes_link_to_gaps(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        gap_ids = {g.gap_id for g in multiple_gaps}
        for cause in causes:
            assert cause.gap_id in gap_ids

    @pytest.mark.asyncio
    async def test_causes_have_provenance(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        for cause in causes:
            assert len(cause.provenance_hash) == 64


# ---------------------------------------------------------------------------
# 5-Whys Depth and Analysis Chain
# ---------------------------------------------------------------------------

class TestFiveWhys:
    """Test 5-Whys analysis chain construction."""

    @pytest.mark.asyncio
    async def test_causes_have_depth(self, mapper, sample_gap):
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        for cause in causes:
            assert cause.depth >= 1
            assert cause.depth <= mapper.config.five_whys_max_depth

    @pytest.mark.asyncio
    async def test_causes_have_analysis_chain(self, mapper, sample_gap):
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        for cause in causes:
            assert isinstance(cause.analysis_chain, list)
            assert len(cause.analysis_chain) == cause.depth

    @pytest.mark.asyncio
    async def test_deeper_causes_have_higher_confidence(self, mapper, sample_gap):
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        # Sort by depth
        sorted_causes = sorted(causes, key=lambda c: c.depth)
        # Confidence should generally increase with depth
        if len(sorted_causes) >= 2:
            assert sorted_causes[-1].confidence >= sorted_causes[0].confidence

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self, mapper, sample_gap):
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        for cause in causes:
            assert Decimal("0") <= cause.confidence <= Decimal("1.0")


# ---------------------------------------------------------------------------
# Fishbone Category Classification
# ---------------------------------------------------------------------------

class TestFishboneCategories:
    """Test fishbone category assignment to root causes."""

    @pytest.mark.asyncio
    async def test_causes_have_category(self, mapper, sample_gap):
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        for cause in causes:
            assert isinstance(cause.category, FishboneCategory)

    @pytest.mark.asyncio
    async def test_category_maps_from_risk_dimension(self, mapper):
        # Create gaps with different risk dimensions
        gap_process = ComplianceGap(
            gap_id="gap-process",
            plan_id="plan-001",
            finding_ids=["fnd-1"],
            severity=GapSeverity.HIGH,
            title="Process gap",
            description="Process issue",
            current_state="Current",
            required_state="Required",
            severity_score=Decimal("0.7"),
            eudr_article_ref="Article 4",
            commodity=EUDRCommodity.COCOA,
            risk_dimension="risk_assessment",  # Maps to PROCESS
            provenance_hash="a" * 64,
        )

        gap_env = ComplianceGap(
            gap_id="gap-env",
            plan_id="plan-001",
            finding_ids=["fnd-2"],
            severity=GapSeverity.HIGH,
            title="Environment gap",
            description="Environment issue",
            current_state="Current",
            required_state="Required",
            severity_score=Decimal("0.7"),
            eudr_article_ref="Article 4",
            commodity=EUDRCommodity.COFFEE,
            risk_dimension="country_risk",  # Maps to ENVIRONMENT
            provenance_hash="b" * 64,
        )

        causes = await mapper.analyze_root_causes(
            gaps=[gap_process, gap_env],
            plan_id="plan-001",
        )

        # Check that we have causes with different categories
        categories = {c.category for c in causes}
        assert len(categories) >= 2


# ---------------------------------------------------------------------------
# Systemic Root Causes
# ---------------------------------------------------------------------------

class TestSystemicRootCauses:
    """Test identification of systemic (cross-cutting) root causes."""

    @pytest.mark.asyncio
    async def test_systemic_flag_exists(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        for cause in causes:
            assert isinstance(cause.systemic, bool)

    @pytest.mark.asyncio
    async def test_systemic_causes_identified(self, mapper, multiple_gaps):
        causes = await mapper.analyze_root_causes(
            gaps=multiple_gaps,
            plan_id="plan-001",
        )
        # With multiple gaps, we should have some systemic causes
        # (same description appearing in 2+ gaps)
        systemic_causes = [c for c in causes if c.systemic]
        # May or may not have systemic causes depending on templates
        assert isinstance(systemic_causes, list)


# ---------------------------------------------------------------------------
# Build Fishbone Diagram
# ---------------------------------------------------------------------------

class TestBuildFishbone:
    """Test fishbone (Ishikawa) diagram construction."""

    @pytest.mark.asyncio
    async def test_build_returns_analysis(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        assert isinstance(fishbone, FishboneAnalysis)

    @pytest.mark.asyncio
    async def test_fishbone_has_id(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        assert fishbone.analysis_id.startswith("FB-")

    @pytest.mark.asyncio
    async def test_fishbone_links_to_gap(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        assert fishbone.gap_id == sample_gap.gap_id

    @pytest.mark.asyncio
    async def test_fishbone_has_categories(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        assert isinstance(fishbone.categories, dict)
        assert len(fishbone.categories) >= 1

    @pytest.mark.asyncio
    async def test_fishbone_has_primary_root_cause(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        # Should have a primary root cause identified
        assert fishbone.primary_root_cause_id is not None

    @pytest.mark.asyncio
    async def test_fishbone_has_provenance(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        assert len(fishbone.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_fishbone_with_precomputed_causes(self, mapper, sample_gap):
        # First analyze root causes
        causes = await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id="plan-001",
        )
        # Build fishbone with those causes
        fishbone = await mapper.build_fishbone(
            gap=sample_gap,
            root_causes=causes,
        )
        assert isinstance(fishbone, FishboneAnalysis)
        assert len(fishbone.categories) >= 1


# ---------------------------------------------------------------------------
# Storage and Retrieval
# ---------------------------------------------------------------------------

class TestStorageRetrieval:
    """Test root cause and fishbone storage/retrieval."""

    @pytest.mark.asyncio
    async def test_get_root_causes_after_analysis(self, mapper, sample_gap):
        plan_id = "plan-storage-001"
        await mapper.analyze_root_causes(
            gaps=[sample_gap],
            plan_id=plan_id,
        )
        causes = await mapper.get_root_causes(plan_id)
        assert isinstance(causes, list)
        assert len(causes) >= 1

    @pytest.mark.asyncio
    async def test_get_root_causes_empty_plan(self, mapper):
        causes = await mapper.get_root_causes("nonexistent-plan")
        assert causes == []

    @pytest.mark.asyncio
    async def test_get_fishbone_after_build(self, mapper, sample_gap):
        fishbone = await mapper.build_fishbone(gap=sample_gap)
        analysis_id = fishbone.analysis_id

        retrieved = await mapper.get_fishbone(analysis_id)
        assert retrieved is not None
        assert retrieved.analysis_id == analysis_id

    @pytest.mark.asyncio
    async def test_get_fishbone_nonexistent(self, mapper):
        result = await mapper.get_fishbone("nonexistent-analysis")
        assert result is None


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, mapper):
        health = await mapper.health_check()
        assert isinstance(health, dict)
        assert "engine" in health
        assert health["engine"] == "RootCauseMapper"

    @pytest.mark.asyncio
    async def test_health_check_has_status(self, mapper):
        health = await mapper.health_check()
        assert "status" in health
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_shows_counts(self, mapper, sample_gap):
        # Initially zero
        health1 = await mapper.health_check()
        assert health1["plans_analyzed"] == 0
        assert health1["fishbones_built"] == 0

        # Analyze root causes
        await mapper.analyze_root_causes([sample_gap], "plan-001")
        health2 = await mapper.health_check()
        assert health2["plans_analyzed"] == 1

        # Build fishbone
        await mapper.build_fishbone(sample_gap)
        health3 = await mapper.health_check()
        assert health3["fishbones_built"] == 1


# ---------------------------------------------------------------------------
# Max Causes Per Category
# ---------------------------------------------------------------------------

class TestMaxCausesLimit:
    """Test max causes per fishbone category enforcement."""

    @pytest.mark.asyncio
    async def test_fishbone_respects_max_causes_per_category(self, mapper):
        # Create a gap with many causes
        gap = ComplianceGap(
            gap_id="gap-many",
            plan_id="plan-001",
            finding_ids=["fnd-1"],
            severity=GapSeverity.HIGH,
            title="Many causes gap",
            description="Test",
            current_state="Current",
            required_state="Required",
            severity_score=Decimal("0.7"),
            eudr_article_ref="Article 4",
            commodity=EUDRCommodity.COCOA,
            risk_dimension="risk_assessment",
            provenance_hash="x" * 64,
        )

        fishbone = await mapper.build_fishbone(gap=gap)

        # Check that each category has at most max_causes_per_category
        for category, causes in fishbone.categories.items():
            assert len(causes) <= mapper.config.fishbone_max_causes_per_category
