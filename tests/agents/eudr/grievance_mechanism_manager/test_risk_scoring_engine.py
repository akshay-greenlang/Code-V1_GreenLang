# -*- coding: utf-8 -*-
"""
Unit tests for RiskScoringEngine - AGENT-EUDR-032

Tests multi-factor weighted risk scoring, factor computation,
risk level classification, trend detection, confidence calculation,
record retrieval, listing, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.risk_scoring_engine import (
    RiskScoringEngine,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    RiskLevel,
    RiskScope,
    RiskScoreRecord,
    ScoreFactor,
    TrendDirection,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def engine(config):
    return RiskScoringEngine(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_engine_created(self, engine):
        assert engine is not None

    def test_default_config(self):
        e = RiskScoringEngine()
        assert e.config is not None

    def test_empty_scores(self, engine):
        assert len(engine._scores) == 0


# ---------------------------------------------------------------------------
# Compute Risk Score
# ---------------------------------------------------------------------------


class TestComputeRiskScore:
    @pytest.mark.asyncio
    async def test_returns_record(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert isinstance(record, RiskScoreRecord)

    @pytest.mark.asyncio
    async def test_operator_scope(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert record.scope == RiskScope.OPERATOR

    @pytest.mark.asyncio
    async def test_supplier_scope(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "supplier", "SUP-001", sample_grievances,
        )
        assert record.scope == RiskScope.SUPPLIER

    @pytest.mark.asyncio
    async def test_commodity_scope(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "commodity", "palm_oil", sample_grievances,
        )
        assert record.scope == RiskScope.COMMODITY

    @pytest.mark.asyncio
    async def test_region_scope(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "region", "APAC", sample_grievances,
        )
        assert record.scope == RiskScope.REGION

    @pytest.mark.asyncio
    async def test_invalid_scope_defaults(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "invalid", "X", sample_grievances,
        )
        assert record.scope == RiskScope.OPERATOR

    @pytest.mark.asyncio
    async def test_score_between_0_and_100(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert Decimal("0") <= record.risk_score <= Decimal("100")

    @pytest.mark.asyncio
    async def test_risk_level_assigned(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert record.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_five_score_factors(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert len(record.score_factors) == 5
        names = {f.factor_name for f in record.score_factors}
        assert names == {"frequency", "severity", "resolution", "escalation", "unresolved"}

    @pytest.mark.asyncio
    async def test_frequency_factor(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        freq = next(f for f in record.score_factors if f.factor_name == "frequency")
        assert freq.raw_value == Decimal("5")  # 5 grievances

    @pytest.mark.asyncio
    async def test_grievance_frequency_set(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert record.grievance_frequency == 5

    @pytest.mark.asyncio
    async def test_unresolved_count(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        # Only "g-001" and "g-004" are resolved
        assert record.unresolved_count == 3

    @pytest.mark.asyncio
    async def test_escalation_rate(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        # g-005 is appealed = 1 out of 5 = 20%
        assert record.escalation_rate == Decimal("20.00")

    @pytest.mark.asyncio
    async def test_prediction_confidence(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        # confidence = min(95, 5 * 5 + 20) = 45
        assert record.prediction_confidence == Decimal("45.00")

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        assert record.risk_score_id in engine._scores

    @pytest.mark.asyncio
    async def test_empty_grievances(self, engine):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", [],
        )
        assert record.risk_score == Decimal("0")
        assert record.grievance_frequency == 0


# ---------------------------------------------------------------------------
# Risk Level Classification
# ---------------------------------------------------------------------------


class TestRiskLevelClassification:
    @pytest.mark.asyncio
    async def test_low_risk_all_resolved(self, engine):
        grievances = [
            {"severity": "low", "status": "resolved"},
            {"severity": "low", "status": "resolved"},
        ]
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", grievances,
        )
        assert record.risk_level in (RiskLevel.NEGLIGIBLE, RiskLevel.LOW, RiskLevel.MODERATE)

    @pytest.mark.asyncio
    async def test_high_risk_all_critical_unresolved(self, engine):
        grievances = [
            {"severity": "critical", "status": "submitted"},
        ] * 8
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", grievances,
        )
        assert record.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


# ---------------------------------------------------------------------------
# Trend Detection
# ---------------------------------------------------------------------------


class TestTrendDetection:
    @pytest.mark.asyncio
    async def test_worsening_trend(self, engine):
        """Many unresolved -> worsening."""
        grievances = [
            {"severity": "high", "status": "submitted"},
        ] * 5
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", grievances,
        )
        assert record.resolution_time_trend == TrendDirection.WORSENING

    @pytest.mark.asyncio
    async def test_improving_trend(self, engine):
        """All resolved -> improving."""
        grievances = [
            {"severity": "medium", "status": "resolved"},
        ] * 10
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", grievances,
        )
        assert record.resolution_time_trend == TrendDirection.IMPROVING

    @pytest.mark.asyncio
    async def test_stable_trend(self, engine):
        """Empty -> stable."""
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", [],
        )
        assert record.resolution_time_trend == TrendDirection.STABLE


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_risk_score(self, engine, sample_grievances):
        record = await engine.compute_risk_score(
            "OP-001", "operator", "OP-001", sample_grievances,
        )
        retrieved = await engine.get_risk_score(record.risk_score_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_risk_score_not_found(self, engine):
        result = await engine.get_risk_score("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, engine, sample_grievances):
        await engine.compute_risk_score("OP-001", "operator", "OP-001", sample_grievances)
        await engine.compute_risk_score("OP-002", "supplier", "SUP-001", sample_grievances[:2])
        results = await engine.list_risk_scores()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_filter_operator(self, engine, sample_grievances):
        await engine.compute_risk_score("OP-001", "operator", "OP-001", sample_grievances)
        await engine.compute_risk_score("OP-002", "operator", "OP-002", sample_grievances[:2])
        results = await engine.list_risk_scores(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_scope(self, engine, sample_grievances):
        await engine.compute_risk_score("OP-001", "operator", "OP-001", sample_grievances)
        await engine.compute_risk_score("OP-001", "supplier", "SUP-001", sample_grievances[:2])
        results = await engine.list_risk_scores(scope="supplier")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, engine):
        results = await engine.list_risk_scores()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        health = await engine.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "RiskScoringEngine"

    @pytest.mark.asyncio
    async def test_health_check_counts(self, engine):
        grievances = [{"severity": "critical", "status": "submitted"}] * 5
        await engine.compute_risk_score("OP-001", "operator", "OP-001", grievances)
        health = await engine.health_check()
        assert health["score_count"] == 1
        assert health["high_risk_count"] >= 0
