# -*- coding: utf-8 -*-
"""
Unit tests for GapAnalyzer - AGENT-EUDR-035

Tests gap analysis from aggregated findings, severity classification,
gap storage, and health checks.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (Engine 2: Gap Analyzer)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.gap_analyzer import (
    GapAnalyzer,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    ComplianceGap,
    EUDRCommodity,
    Finding,
    FindingSource,
    GapSeverity,
)


@pytest.fixture
def config():
    return ImprovementPlanCreatorConfig()


@pytest.fixture
def analyzer(config):
    return GapAnalyzer(config=config)


# ---------------------------------------------------------------------------
# Analyze Gaps
# ---------------------------------------------------------------------------

class TestAnalyzeGaps:
    """Test gap analysis from aggregated findings."""

    @pytest.mark.asyncio
    async def test_analyze_returns_gap_list(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert isinstance(gaps, list)
        assert len(gaps) > 0

    @pytest.mark.asyncio
    async def test_analyze_from_single_finding(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 1

    @pytest.mark.asyncio
    async def test_analyze_from_multiple_findings(self, analyzer, multiple_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=multiple_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 4

    @pytest.mark.asyncio
    async def test_analyze_from_empty_findings(self, analyzer):
        from greenlang.agents.eudr.improvement_plan_creator.models import AggregatedFindings
        empty_agg = AggregatedFindings(
            aggregation_id="agg-empty",
            operator_id="operator-001",
            findings=[],
            total_findings=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            source_agents=[],
            duplicates_removed=0,
            provenance_hash="e" * 64,
        )
        gaps = await analyzer.analyze_gaps(
            aggregation=empty_agg,
            plan_id="plan-001",
        )
        assert gaps == []

    @pytest.mark.asyncio
    async def test_gaps_have_ids(self, analyzer, multiple_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=multiple_aggregated_findings,
            plan_id="plan-001",
        )
        for gap in gaps:
            assert gap.gap_id.startswith("GAP-")

    @pytest.mark.asyncio
    async def test_gaps_link_to_findings(self, analyzer, multiple_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=multiple_aggregated_findings,
            plan_id="plan-001",
        )
        for gap in gaps:
            assert len(gap.finding_ids) >= 1

    @pytest.mark.asyncio
    async def test_gaps_have_provenance(self, analyzer, multiple_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=multiple_aggregated_findings,
            plan_id="plan-001",
        )
        for gap in gaps:
            assert len(gap.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_gaps_have_plan_id(self, analyzer, multiple_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=multiple_aggregated_findings,
            plan_id="plan-test-001",
        )
        for gap in gaps:
            assert gap.plan_id == "plan-test-001"


# ---------------------------------------------------------------------------
# Gap Severity Classification
# ---------------------------------------------------------------------------

class TestGapSeverity:
    """Test gap severity assignment based on finding risk scores."""

    @pytest.mark.asyncio
    async def test_critical_severity_from_high_risk(self, analyzer):
        from greenlang.agents.eudr.improvement_plan_creator.models import AggregatedFindings, Finding
        finding = Finding(
            finding_id="fnd-critical",
            operator_id="operator-001",
            source=FindingSource.LEGAL_COMPLIANCE,
            severity=GapSeverity.CRITICAL,
            title="Critical compliance issue",
            description="High risk score finding",
            commodity=EUDRCommodity.COCOA,
            eudr_article_ref="Article 4",
            detected_at=datetime.now(tz=timezone.utc),
            risk_score=Decimal("95.00"),  # Will normalize to 0.95
            provenance_hash="a" * 64,
        )
        agg = AggregatedFindings(
            aggregation_id="agg-crit",
            operator_id="operator-001",
            findings=[finding],
            total_findings=1,
            critical_count=1,
            high_count=0,
            medium_count=0,
            low_count=0,
            source_agents=["legal_compliance"],
            duplicates_removed=0,
            provenance_hash="z" * 64,
        )
        gaps = await analyzer.analyze_gaps(aggregation=agg, plan_id="plan-001")
        assert len(gaps) == 1
        assert gaps[0].severity == GapSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_high_severity_from_medium_high_risk(self, analyzer):
        from greenlang.agents.eudr.improvement_plan_creator.models import AggregatedFindings, Finding
        finding = Finding(
            finding_id="fnd-high",
            operator_id="operator-001",
            source=FindingSource.SUPPLIER_RISK,
            severity=GapSeverity.HIGH,
            title="High risk finding",
            description="Medium-high risk score",
            commodity=EUDRCommodity.COFFEE,
            eudr_article_ref="Article 10",
            detected_at=datetime.now(tz=timezone.utc),
            risk_score=Decimal("65.00"),  # Will normalize to 0.65
            provenance_hash="b" * 64,
        )
        agg = AggregatedFindings(
            aggregation_id="agg-high",
            operator_id="operator-001",
            findings=[finding],
            total_findings=1,
            critical_count=0,
            high_count=1,
            medium_count=0,
            low_count=0,
            source_agents=["supplier_risk"],
            duplicates_removed=0,
            provenance_hash="z" * 64,
        )
        gaps = await analyzer.analyze_gaps(aggregation=agg, plan_id="plan-001")
        assert len(gaps) == 1
        assert gaps[0].severity == GapSeverity.HIGH

    @pytest.mark.asyncio
    async def test_medium_severity_from_medium_risk(self, analyzer):
        from greenlang.agents.eudr.improvement_plan_creator.models import AggregatedFindings, Finding
        finding = Finding(
            finding_id="fnd-medium",
            operator_id="operator-001",
            source=FindingSource.COUNTRY_RISK,
            severity=GapSeverity.MEDIUM,
            title="Medium risk finding",
            description="Medium risk score",
            commodity=EUDRCommodity.SOYA,
            eudr_article_ref="Article 10",
            detected_at=datetime.now(tz=timezone.utc),
            risk_score=Decimal("45.00"),  # Will normalize to 0.45
            provenance_hash="c" * 64,
        )
        agg = AggregatedFindings(
            aggregation_id="agg-medium",
            operator_id="operator-001",
            findings=[finding],
            total_findings=1,
            critical_count=0,
            high_count=0,
            medium_count=1,
            low_count=0,
            source_agents=["country_risk"],
            duplicates_removed=0,
            provenance_hash="z" * 64,
        )
        gaps = await analyzer.analyze_gaps(aggregation=agg, plan_id="plan-001")
        assert len(gaps) == 1
        assert gaps[0].severity == GapSeverity.MEDIUM


# ---------------------------------------------------------------------------
# Gap Storage and Retrieval
# ---------------------------------------------------------------------------

class TestGapStorage:
    """Test gap storage and retrieval."""

    @pytest.mark.asyncio
    async def test_get_gaps_after_analysis(self, analyzer, sample_aggregated_findings):
        plan_id = "plan-storage-001"
        await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id=plan_id,
        )
        gaps = await analyzer.get_gaps(plan_id)
        assert isinstance(gaps, list)
        assert len(gaps) == 1

    @pytest.mark.asyncio
    async def test_get_gaps_empty_plan(self, analyzer):
        gaps = await analyzer.get_gaps("nonexistent-plan")
        assert gaps == []

    @pytest.mark.asyncio
    async def test_get_gaps_multiple_plans(self, analyzer, sample_aggregated_findings):
        # Analyze for multiple plans
        await analyzer.analyze_gaps(sample_aggregated_findings, "plan-A")
        await analyzer.analyze_gaps(sample_aggregated_findings, "plan-B")

        gaps_a = await analyzer.get_gaps("plan-A")
        gaps_b = await analyzer.get_gaps("plan-B")

        assert len(gaps_a) == 1
        assert len(gaps_b) == 1
        assert gaps_a[0].plan_id == "plan-A"
        assert gaps_b[0].plan_id == "plan-B"


# ---------------------------------------------------------------------------
# EUDR Article Mapping
# ---------------------------------------------------------------------------

class TestEUDRArticleMapping:
    """Test EUDR article reference assignment to gaps."""

    @pytest.mark.asyncio
    async def test_article_mapping_from_finding(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 1
        # Finding has Article 9(1)(d), should be preserved
        assert "Article 9" in gaps[0].eudr_article_ref

    @pytest.mark.asyncio
    async def test_article_mapping_from_source(self, analyzer):
        from greenlang.agents.eudr.improvement_plan_creator.models import AggregatedFindings, Finding
        finding = Finding(
            finding_id="fnd-source-map",
            operator_id="operator-001",
            source=FindingSource.RISK_ASSESSMENT,  # Maps to Article 10
            severity=GapSeverity.HIGH,
            title="Risk assessment finding",
            description="Test",
            commodity=EUDRCommodity.COCOA,
            # eudr_article_ref omitted, should default to source mapping
            detected_at=datetime.now(tz=timezone.utc),
            risk_score=Decimal("70.00"),
            provenance_hash="d" * 64,
        )
        agg = AggregatedFindings(
            aggregation_id="agg-map",
            operator_id="operator-001",
            findings=[finding],
            total_findings=1,
            critical_count=0,
            high_count=1,
            medium_count=0,
            low_count=0,
            source_agents=["risk_assessment"],
            duplicates_removed=0,
            provenance_hash="z" * 64,
        )
        gaps = await analyzer.analyze_gaps(aggregation=agg, plan_id="plan-001")
        assert len(gaps) == 1
        # Should map to Article 10 based on source (formatted as "Art. 10 - ...")
        assert "Art. 10" in gaps[0].eudr_article_ref


# ---------------------------------------------------------------------------
# Gap Details
# ---------------------------------------------------------------------------

class TestGapDetails:
    """Test gap detail fields."""

    @pytest.mark.asyncio
    async def test_gap_title_from_finding(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 1
        assert gaps[0].title.startswith("Gap:")

    @pytest.mark.asyncio
    async def test_gap_description_includes_source(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 1
        assert "risk_assessment" in gaps[0].description

    @pytest.mark.asyncio
    async def test_gap_has_current_and_required_state(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 1
        assert gaps[0].current_state is not None
        assert gaps[0].required_state is not None
        assert len(gaps[0].current_state) > 0
        assert len(gaps[0].required_state) > 0

    @pytest.mark.asyncio
    async def test_gap_severity_score_normalized(self, analyzer, sample_aggregated_findings):
        gaps = await analyzer.analyze_gaps(
            aggregation=sample_aggregated_findings,
            plan_id="plan-001",
        )
        assert len(gaps) == 1
        # Severity score should be normalized to 0-1
        assert Decimal("0") <= gaps[0].severity_score <= Decimal("1")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, analyzer):
        health = await analyzer.health_check()
        assert isinstance(health, dict)
        assert "engine" in health
        assert health["engine"] == "GapAnalyzer"

    @pytest.mark.asyncio
    async def test_health_check_has_status(self, analyzer):
        health = await analyzer.health_check()
        assert "status" in health
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_shows_plans_analyzed(self, analyzer, sample_aggregated_findings):
        # Initially zero
        health1 = await analyzer.health_check()
        assert health1["plans_analyzed"] == 0

        # Analyze one plan
        await analyzer.analyze_gaps(sample_aggregated_findings, "plan-001")
        health2 = await analyzer.health_check()
        assert health2["plans_analyzed"] == 1

        # Analyze another plan
        await analyzer.analyze_gaps(sample_aggregated_findings, "plan-002")
        health3 = await analyzer.health_check()
        assert health3["plans_analyzed"] == 2


# ---------------------------------------------------------------------------
# Max Gaps Limit
# ---------------------------------------------------------------------------

class TestMaxGapsLimit:
    """Test max gaps per analysis enforcement."""

    @pytest.mark.asyncio
    async def test_max_gaps_enforced(self, analyzer):
        from greenlang.agents.eudr.improvement_plan_creator.models import AggregatedFindings, Finding
        # Create many findings
        findings = []
        for i in range(150):  # More than max_gaps_per_analysis (100)
            findings.append(Finding(
                finding_id=f"fnd-{i}",
                operator_id="operator-001",
                source=FindingSource.MANUAL,
                severity=GapSeverity.LOW,
                title=f"Finding {i}",
                description=f"Test finding {i}",
                commodity=EUDRCommodity.COCOA,
                eudr_article_ref="Article 4",
                detected_at=datetime.now(tz=timezone.utc),
                risk_score=Decimal("20.00"),
                provenance_hash="x" * 64,
            ))

        agg = AggregatedFindings(
            aggregation_id="agg-many",
            operator_id="operator-001",
            findings=findings,
            total_findings=len(findings),
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=len(findings),
            source_agents=["manual"],
            duplicates_removed=0,
            provenance_hash="z" * 64,
        )

        gaps = await analyzer.analyze_gaps(aggregation=agg, plan_id="plan-001")
        # Should be limited to max_gaps_per_analysis
        assert len(gaps) <= analyzer.config.max_gaps_per_analysis
