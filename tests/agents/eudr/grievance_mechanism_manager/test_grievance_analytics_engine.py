# -*- coding: utf-8 -*-
"""
Unit tests for GrievanceAnalyticsEngine - AGENT-EUDR-032

Tests pattern detection, trend assessment, recommendation generation,
root cause extraction, record retrieval, listing, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.grievance_analytics_engine import (
    GrievanceAnalyticsEngine,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    GrievanceAnalyticsRecord,
    PatternType,
    TrendDirection,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def engine(config):
    return GrievanceAnalyticsEngine(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_engine_created(self, engine):
        assert engine is not None

    def test_engine_uses_config(self, config):
        engine = GrievanceAnalyticsEngine(config=config)
        assert engine.config is config

    def test_engine_default_config(self):
        engine = GrievanceAnalyticsEngine()
        assert engine.config is not None

    def test_records_empty_on_init(self, engine):
        assert len(engine._records) == 0


# ---------------------------------------------------------------------------
# Pattern Detection
# ---------------------------------------------------------------------------


class TestPatternDetection:
    def test_empty_grievances_returns_isolated(self, engine):
        result = engine._detect_pattern([], {}, {})
        assert result == PatternType.ISOLATED

    def test_single_grievance_returns_isolated(self, engine):
        g = [{"severity": "low", "category": "process"}]
        result = engine._detect_pattern(g, {"process": 1}, {"low": 1})
        assert result == PatternType.ISOLATED

    def test_two_grievances_returns_clustered(self, engine):
        g = [{"severity": "low"}, {"severity": "low"}]
        result = engine._detect_pattern(g, {"process": 2}, {"low": 2})
        assert result == PatternType.CLUSTERED

    def test_escalating_when_high_critical_majority(self, engine):
        g = [{"severity": "high"}] * 6 + [{"severity": "low"}] * 4
        sev = {"high": 6, "low": 4}
        cat = {"environmental": 10}
        result = engine._detect_pattern(g, cat, sev)
        assert result == PatternType.ESCALATING

    def test_systemic_when_single_category_dominates(self, engine):
        g = [{"severity": "low"}] * 4
        cat = {"environmental": 4}
        sev = {"low": 4}
        result = engine._detect_pattern(g, cat, sev)
        assert result == PatternType.SYSTEMIC

    def test_recurring_when_enough_grievances(self, engine):
        g = [{"severity": "low"}] * 3
        cat = {"environmental": 1, "labor": 1, "process": 1}
        sev = {"low": 3}
        result = engine._detect_pattern(g, cat, sev)
        assert result == PatternType.RECURRING

    def test_escalating_takes_priority_over_systemic(self, engine):
        """If > 50% are high/critical, escalating wins even if one category dominates."""
        g = [{"severity": "critical"}] * 4
        cat = {"environmental": 4}
        sev = {"critical": 4}
        result = engine._detect_pattern(g, cat, sev)
        assert result == PatternType.ESCALATING


# ---------------------------------------------------------------------------
# Trend Assessment
# ---------------------------------------------------------------------------


class TestTrendAssessment:
    def test_stable_with_single_grievance(self, engine):
        result = engine._assess_trend([{"severity": "high"}])
        assert result == TrendDirection.STABLE

    def test_stable_with_empty_list(self, engine):
        result = engine._assess_trend([])
        assert result == TrendDirection.STABLE

    def test_worsening_when_second_half_more_critical(self, engine):
        g = (
            [{"severity": "low"}] * 5
            + [{"severity": "critical"}] * 5
        )
        result = engine._assess_trend(g)
        assert result == TrendDirection.WORSENING

    def test_improving_when_first_half_more_critical(self, engine):
        g = (
            [{"severity": "critical"}] * 5
            + [{"severity": "low"}] * 5
        )
        result = engine._assess_trend(g)
        assert result == TrendDirection.IMPROVING

    def test_stable_when_equal_halves(self, engine):
        g = (
            [{"severity": "high"}] * 3
            + [{"severity": "high"}] * 3
        )
        result = engine._assess_trend(g)
        assert result == TrendDirection.STABLE


# ---------------------------------------------------------------------------
# Describe Pattern
# ---------------------------------------------------------------------------


class TestDescribePattern:
    def test_recurring_description(self, engine):
        desc = engine._describe_pattern(PatternType.RECURRING, 5)
        assert "Recurring" in desc
        assert "5" in desc

    def test_clustered_description(self, engine):
        desc = engine._describe_pattern(PatternType.CLUSTERED, 3)
        assert "Clustered" in desc

    def test_systemic_description(self, engine):
        desc = engine._describe_pattern(PatternType.SYSTEMIC, 10)
        assert "Systemic" in desc

    def test_isolated_description(self, engine):
        desc = engine._describe_pattern(PatternType.ISOLATED, 1)
        assert "Isolated" in desc

    def test_escalating_description(self, engine):
        desc = engine._describe_pattern(PatternType.ESCALATING, 8)
        assert "Escalating" in desc


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_systemic_recommendation(self, engine):
        recs = engine._generate_recommendations(
            PatternType.SYSTEMIC, {"environmental": 10}, {"high": 5},
        )
        actions = [r["action"] for r in recs]
        assert any("systemic review" in a for a in actions)
        assert any("senior management" in a for a in actions)

    def test_escalating_recommendation(self, engine):
        recs = engine._generate_recommendations(
            PatternType.ESCALATING, {"labor": 5}, {"medium": 5},
        )
        actions = [r["action"] for r in recs]
        assert any("escalation protocol" in a for a in actions)

    def test_recurring_recommendation(self, engine):
        recs = engine._generate_recommendations(
            PatternType.RECURRING, {"process": 3}, {"low": 3},
        )
        actions = [r["action"] for r in recs]
        assert any("root cause" in a for a in actions)

    def test_critical_severity_recommendation(self, engine):
        recs = engine._generate_recommendations(
            PatternType.ISOLATED, {"process": 1}, {"critical": 1},
        )
        actions = [r["action"] for r in recs]
        assert any("senior management" in a for a in actions)

    def test_default_recommendation_when_none_match(self, engine):
        recs = engine._generate_recommendations(
            PatternType.ISOLATED, {"process": 1}, {"low": 1},
        )
        assert len(recs) >= 1
        assert recs[0]["priority"] == "medium"


# ---------------------------------------------------------------------------
# Root Cause Extraction
# ---------------------------------------------------------------------------


class TestRootCauseExtraction:
    def test_empty_grievances(self, engine):
        causes = engine._extract_root_causes([])
        assert causes == []

    def test_no_investigation_notes(self, engine):
        causes = engine._extract_root_causes([{"id": "g-001"}])
        assert causes == []

    def test_extracts_from_notes(self, engine):
        g = [
            {"investigation_notes": {"root_cause": "pollution"}},
            {"investigation_notes": {"root_cause": "pollution"}},
            {"investigation_notes": {"root_cause": "deforestation"}},
        ]
        causes = engine._extract_root_causes(g)
        assert len(causes) == 2
        assert causes[0]["cause"] == "pollution"
        assert causes[0]["frequency"] == 2

    def test_limits_to_top_five(self, engine):
        g = [
            {"investigation_notes": {"root_cause": f"cause_{i}"}}
            for i in range(10)
        ]
        causes = engine._extract_root_causes(g)
        assert len(causes) <= 5


# ---------------------------------------------------------------------------
# Analyze Patterns (async integration)
# ---------------------------------------------------------------------------


class TestAnalyzePatterns:
    @pytest.mark.asyncio
    async def test_analyze_returns_record(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert isinstance(record, GrievanceAnalyticsRecord)
        assert record.operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_analytics_id_is_uuid(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.analytics_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_grievance_ids_populated(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.grievance_ids) == len(sample_grievances)

    @pytest.mark.asyncio
    async def test_pattern_type_assigned(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert record.pattern_type in PatternType

    @pytest.mark.asyncio
    async def test_severity_distribution_populated(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.severity_distribution) > 0

    @pytest.mark.asyncio
    async def test_category_distribution_populated(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.category_distribution) > 0

    @pytest.mark.asyncio
    async def test_trend_direction_assigned(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert record.trend_direction in TrendDirection

    @pytest.mark.asyncio
    async def test_trend_confidence_set(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert record.trend_confidence == Decimal("75")

    @pytest.mark.asyncio
    async def test_recommendations_generated(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.recommendations) > 0

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored_internally(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert record.analytics_id in engine._records

    @pytest.mark.asyncio
    async def test_stakeholder_count(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert record.affected_stakeholder_count == 5

    @pytest.mark.asyncio
    async def test_custom_period(self, engine, sample_grievances):
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=30)
        record = await engine.analyze_patterns(
            "OP-001", sample_grievances,
            period_start=start, period_end=now,
        )
        assert record.analysis_period_start == start

    @pytest.mark.asyncio
    async def test_empty_grievances_list(self, engine):
        record = await engine.analyze_patterns("OP-001", [])
        assert record.pattern_type == PatternType.ISOLATED
        assert record.affected_stakeholder_count == 0

    @pytest.mark.asyncio
    async def test_pattern_description_not_empty(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        assert len(record.pattern_description) > 0


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_analytics_returns_record(self, engine, sample_grievances):
        record = await engine.analyze_patterns("OP-001", sample_grievances)
        retrieved = await engine.get_analytics(record.analytics_id)
        assert retrieved is not None
        assert retrieved.analytics_id == record.analytics_id

    @pytest.mark.asyncio
    async def test_get_analytics_returns_none_for_unknown(self, engine):
        result = await engine.get_analytics("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_analytics_all(self, engine, sample_grievances):
        await engine.analyze_patterns("OP-001", sample_grievances)
        await engine.analyze_patterns("OP-002", sample_grievances[:2])
        results = await engine.list_analytics()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_analytics_filter_operator(self, engine, sample_grievances):
        await engine.analyze_patterns("OP-001", sample_grievances)
        await engine.analyze_patterns("OP-002", sample_grievances[:2])
        results = await engine.list_analytics(operator_id="OP-001")
        assert len(results) == 1
        assert results[0].operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_list_analytics_filter_pattern(self, engine):
        await engine.analyze_patterns("OP-001", [])
        results = await engine.list_analytics(pattern_type="isolated")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_analytics_empty(self, engine):
        results = await engine.list_analytics()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, engine):
        health = await engine.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "GrievanceAnalyticsEngine"

    @pytest.mark.asyncio
    async def test_health_check_record_count(self, engine, sample_grievances):
        await engine.analyze_patterns("OP-001", sample_grievances)
        health = await engine.health_check()
        assert health["record_count"] == 1
