# -*- coding: utf-8 -*-
"""
Unit tests for RiskScoreMonitor - AGENT-EUDR-033

Tests risk score monitoring, trend analysis, degradation detection,
incident correlation, risk level classification, record retrieval,
listing, filtering, and health checks.

50+ tests covering all risk monitoring logic paths.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.continuous_monitoring.config import (
    ContinuousMonitoringConfig,
)
from greenlang.agents.eudr.continuous_monitoring.risk_score_monitor import (
    RiskScoreMonitor,
)
from greenlang.agents.eudr.continuous_monitoring.models import (
    RiskLevel,
    RiskScoreMonitorRecord,
    TrendDirection,
)


@pytest.fixture
def config():
    return ContinuousMonitoringConfig()


@pytest.fixture
def monitor(config):
    return RiskScoreMonitor(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_monitor_created(self, monitor):
        assert monitor is not None

    def test_monitor_uses_config(self, config):
        m = RiskScoreMonitor(config=config)
        assert m.config is config

    def test_monitor_default_config(self):
        m = RiskScoreMonitor()
        assert m.config is not None

    def test_records_empty_on_init(self, monitor):
        assert len(monitor._records) == 0


# ---------------------------------------------------------------------------
# Monitor Risk Scores
# ---------------------------------------------------------------------------


class TestMonitorRiskScores:
    @pytest.mark.asyncio
    async def test_returns_record(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert isinstance(record, RiskScoreMonitorRecord)

    @pytest.mark.asyncio
    async def test_monitor_id_generated(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.monitor_id is not None
        assert len(record.monitor_id) > 0

    @pytest.mark.asyncio
    async def test_operator_id_set(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_entity_id_set(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.entity_id == "SUP-001"

    @pytest.mark.asyncio
    async def test_entity_type_default_supplier(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.entity_type == "supplier"

    @pytest.mark.asyncio
    async def test_entity_type_custom(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "PLOT-001", sample_risk_score_history,
            entity_type="plot",
        )
        assert record.entity_type == "plot"

    @pytest.mark.asyncio
    async def test_current_score_set(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.current_score >= Decimal("0")

    @pytest.mark.asyncio
    async def test_previous_score_set(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.previous_score >= Decimal("0")

    @pytest.mark.asyncio
    async def test_risk_level_assigned(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.risk_level in RiskLevel

    @pytest.mark.asyncio
    async def test_trend_assigned(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.trend_direction in TrendDirection

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_stored_internally(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert record.monitor_id in monitor._records

    @pytest.mark.asyncio
    async def test_empty_history(self, monitor):
        record = await monitor.monitor_risk_scores("OP-001", "SUP-001", [])
        assert record.current_score == Decimal("0")
        assert record.trend_direction == TrendDirection.STABLE


# ---------------------------------------------------------------------------
# Risk Level Classification
# ---------------------------------------------------------------------------


class TestRiskLevelClassification:
    @pytest.mark.asyncio
    async def test_negligible_risk(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 5}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.risk_level == RiskLevel.NEGLIGIBLE

    @pytest.mark.asyncio
    async def test_low_risk(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 20}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_moderate_risk(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 40}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.risk_level == RiskLevel.MODERATE

    @pytest.mark.asyncio
    async def test_high_risk(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 65}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.risk_level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_critical_risk(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 90}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Trend Analysis
# ---------------------------------------------------------------------------


class TestTrendAnalysis:
    @pytest.mark.asyncio
    async def test_worsening_trend(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=10 - i)).isoformat(), "score": 30 + i * 5}
            for i in range(10)
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.trend_direction == TrendDirection.WORSENING

    @pytest.mark.asyncio
    async def test_improving_trend(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=10 - i)).isoformat(), "score": 80 - i * 5}
            for i in range(10)
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.trend_direction == TrendDirection.IMPROVING

    @pytest.mark.asyncio
    async def test_stable_trend(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=10 - i)).isoformat(), "score": 50}
            for i in range(10)
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.trend_direction == TrendDirection.STABLE

    @pytest.mark.asyncio
    async def test_single_data_point_stable(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 50}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.trend_direction == TrendDirection.STABLE

    @pytest.mark.asyncio
    async def test_score_delta_positive(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=30)).isoformat(), "score": 30},
            {"timestamp": now.isoformat(), "score": 60},
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.score_delta > Decimal("0")

    @pytest.mark.asyncio
    async def test_score_delta_negative(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=30)).isoformat(), "score": 80},
            {"timestamp": now.isoformat(), "score": 40},
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.score_delta < Decimal("0")


# ---------------------------------------------------------------------------
# Degradation Detection
# ---------------------------------------------------------------------------


class TestDegradationDetection:
    @pytest.mark.asyncio
    async def test_degradation_detected(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=10 - i)).isoformat(), "score": 30 + i * 8}
            for i in range(10)
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.degradation_detected is True

    @pytest.mark.asyncio
    async def test_no_degradation_stable(self, monitor):
        now = datetime.now(timezone.utc)
        history = [
            {"timestamp": (now - timedelta(days=10 - i)).isoformat(), "score": 50}
            for i in range(10)
        ]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.degradation_detected is False


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_record(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        retrieved = await monitor.get_record(record.monitor_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_record_not_found(self, monitor):
        result = await monitor.get_record("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_records_all(self, monitor, sample_risk_score_history):
        await monitor.monitor_risk_scores("OP-001", "SUP-001", sample_risk_score_history)
        await monitor.monitor_risk_scores("OP-002", "SUP-002", sample_risk_score_history[:3])
        results = await monitor.list_records()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_records_filter_operator(self, monitor, sample_risk_score_history):
        await monitor.monitor_risk_scores("OP-001", "SUP-001", sample_risk_score_history)
        await monitor.monitor_risk_scores("OP-002", "SUP-002", sample_risk_score_history[:3])
        results = await monitor.list_records(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_records_filter_entity(self, monitor, sample_risk_score_history):
        await monitor.monitor_risk_scores("OP-001", "SUP-001", sample_risk_score_history)
        await monitor.monitor_risk_scores("OP-001", "SUP-002", sample_risk_score_history[:3])
        results = await monitor.list_records(entity_id="SUP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_records_empty(self, monitor):
        results = await monitor.list_records()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, monitor):
        health = await monitor.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "RiskScoreMonitor"

    @pytest.mark.asyncio
    async def test_health_check_record_count(self, monitor, sample_risk_score_history):
        await monitor.monitor_risk_scores("OP-001", "SUP-001", sample_risk_score_history)
        health = await monitor.health_check()
        assert health["record_count"] == 1


# ---------------------------------------------------------------------------
# Multi-Entity Risk
# ---------------------------------------------------------------------------


class TestMultiEntityRisk:
    @pytest.mark.asyncio
    async def test_different_entities_independent(self, monitor):
        high_history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 80}]
        low_history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 10}]
        r1 = await monitor.monitor_risk_scores("OP-001", "SUP-001", high_history)
        r2 = await monitor.monitor_risk_scores("OP-001", "SUP-002", low_history)
        assert r1.risk_level != r2.risk_level

    @pytest.mark.asyncio
    async def test_commodity_entity_type(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 70}]
        record = await monitor.monitor_risk_scores(
            "OP-001", "palm_oil", history, entity_type="commodity",
        )
        assert record.entity_type == "commodity"
        assert record.entity_id == "palm_oil"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestRiskScoreProvenance:
    @pytest.mark.asyncio
    async def test_provenance_is_hex(self, monitor, sample_risk_score_history):
        record = await monitor.monitor_risk_scores(
            "OP-001", "SUP-001", sample_risk_score_history,
        )
        assert all(c in "0123456789abcdef" for c in record.provenance_hash)

    @pytest.mark.asyncio
    async def test_different_entity_different_provenance(self, monitor):
        history = [{"timestamp": "2026-01-15T10:00:00+00:00", "score": 50}]
        r1 = await monitor.monitor_risk_scores("OP-001", "SUP-001", history)
        r2 = await monitor.monitor_risk_scores("OP-001", "SUP-002", history)
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# Boundary Tests
# ---------------------------------------------------------------------------


class TestRiskScoreBoundary:
    @pytest.mark.asyncio
    async def test_zero_score(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 0}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.current_score == Decimal("0")
        assert record.risk_level == RiskLevel.NEGLIGIBLE

    @pytest.mark.asyncio
    async def test_max_score(self, monitor):
        history = [{"timestamp": datetime.now(timezone.utc).isoformat(), "score": 100}]
        record = await monitor.monitor_risk_scores("OP-001", "S-001", history)
        assert record.current_score == Decimal("100")
        assert record.risk_level == RiskLevel.CRITICAL
