# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 RiskClassificationEngine.

Tests 5-tier threshold classification, hysteresis buffer behaviour,
Article 10 criteria-based escalation, and simplified due diligence
eligibility checks.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10CriteriaResult,
    Article10CriterionEvaluation,
    Article10Criterion,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    DimensionScore,
    RiskDimension,
    RiskLevel,
    SimplifiedDDEligibility,
    SourceAgent,
)


def _make_engine_config():
    """Build a mock config with dict-style risk_thresholds and simplified_dd."""
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    cfg.risk_thresholds = {
        "negligible": 15,
        "low": 30,
        "standard": 60,
        "high": 80,
        "critical": 100,
    }
    cfg.hysteresis_buffer = Decimal("3")
    cfg.simplified_dd = {
        "max_score": 30,
        "require_all_low": True,
    }
    return cfg


def _make_engine():
    """Instantiate RiskClassificationEngine with mocked dependencies."""
    from greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine import (
        RiskClassificationEngine,
    )
    cfg = _make_engine_config()
    with patch(
        "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
    ):
        return RiskClassificationEngine(config=cfg)


def _make_composite(score: Decimal, level: RiskLevel = RiskLevel.STANDARD) -> CompositeRiskScore:
    return CompositeRiskScore(
        overall_score=score,
        risk_level=level,
        dimension_scores=[],
        provenance_hash="a" * 64,
    )


def _make_article10(
    concern_count: int = 0,
    pass_count: int = 10,
    fail_count: int = 0,
    total_evaluated: int = 10,
) -> Article10CriteriaResult:
    """Build an Article10CriteriaResult mock with summary counts."""
    result = MagicMock(spec=Article10CriteriaResult)
    result.concern_count = concern_count
    result.pass_count = pass_count
    result.fail_count = fail_count
    result.total_evaluated = total_evaluated
    result.evaluations = []
    return result


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyRisk:
    """Test basic threshold-based classification without hysteresis."""

    def test_classify_negligible(self):
        """Score 10 -> NEGLIGIBLE (threshold <= 15)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("10"))
        assert level == RiskLevel.NEGLIGIBLE

    def test_classify_low(self):
        """Score 25 -> LOW (15 < score <= 30)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("25"))
        assert level == RiskLevel.LOW

    def test_classify_standard(self):
        """Score 45 -> STANDARD (30 < score <= 60)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("45"))
        assert level == RiskLevel.STANDARD

    def test_classify_high(self):
        """Score 70 -> HIGH (60 < score <= 80)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("70"))
        assert level == RiskLevel.HIGH

    def test_classify_critical(self):
        """Score 90 -> CRITICAL (score > 80)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("90"))
        assert level == RiskLevel.CRITICAL

    def test_classify_at_threshold_boundary_30(self):
        """Score exactly 30 -> LOW (score <= 30)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("30"))
        assert level == RiskLevel.LOW

    def test_classify_at_threshold_boundary_31(self):
        """Score 31 -> STANDARD (score > 30)."""
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(Decimal("31"))
        assert level == RiskLevel.STANDARD


# ---------------------------------------------------------------------------
# Hysteresis tests
# ---------------------------------------------------------------------------


class TestHysteresis:
    """Test hysteresis buffer prevents oscillation."""

    def test_hysteresis_prevents_downgrade(self):
        """Score drops from 62 to 59 (within buffer) -> stays HIGH.

        HIGH boundary is (61, 80). Hysteresis buffer = 3.
        Downgrade boundary = 61 - 3 = 58.
        Score 59 > 58, so hysteresis keeps HIGH.
        """
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(
                Decimal("59"), previous_level=RiskLevel.HIGH
            )
        assert level == RiskLevel.HIGH

    def test_hysteresis_allows_large_change(self):
        """Score drops from 80 to 55 (well below buffer) -> goes to STANDARD.

        The hysteresis boundary for downgrade from HIGH is 61-3=58.
        Score 55 < 58, so downgrade is allowed.
        """
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            level = engine.classify_risk(
                Decimal("55"), previous_level=RiskLevel.HIGH
            )
        assert level == RiskLevel.STANDARD


# ---------------------------------------------------------------------------
# Article 10 escalation tests
# ---------------------------------------------------------------------------


class TestArticle10Escalation:
    """Test Article 10 criteria-based escalation/de-escalation."""

    def test_classify_with_article10_escalation(self):
        """LOW + 3 CONCERN criteria -> escalate to STANDARD."""
        engine = _make_engine()
        composite = _make_composite(Decimal("25"), RiskLevel.LOW)
        article10 = _make_article10(concern_count=3, pass_count=7)
        level = engine.classify_with_article10(composite, article10)
        assert level == RiskLevel.STANDARD

    def test_classify_with_article10_no_escalation(self):
        """LOW + 1 CONCERN -> stays LOW (threshold is 3)."""
        engine = _make_engine()
        composite = _make_composite(Decimal("25"), RiskLevel.LOW)
        article10 = _make_article10(concern_count=1, pass_count=9)
        level = engine.classify_with_article10(composite, article10)
        assert level == RiskLevel.LOW

    def test_classify_with_article10_negligible_escalation(self):
        """NEGLIGIBLE + any concern -> escalate to LOW."""
        engine = _make_engine()
        composite = _make_composite(Decimal("10"), RiskLevel.NEGLIGIBLE)
        article10 = _make_article10(concern_count=1, pass_count=9)
        level = engine.classify_with_article10(composite, article10)
        assert level == RiskLevel.LOW


# ---------------------------------------------------------------------------
# Simplified DD eligibility
# ---------------------------------------------------------------------------


class TestSimplifiedDD:
    """Test simplified due diligence eligibility checks."""

    def test_simplified_dd_eligible(self):
        """All LOW countries + score < 30 + low deforestation -> eligible."""
        engine = _make_engine()
        composite = CompositeRiskScore(
            overall_score=Decimal("20"),
            risk_level=RiskLevel.LOW,
            dimension_scores=[
                DimensionScore(
                    dimension=RiskDimension.DEFORESTATION,
                    weighted_score=Decimal("2.00"),
                    raw_score=Decimal("10"),
                    weight=Decimal("0.20"),
                    confidence=Decimal("0.90"),
                    source_agent=SourceAgent.EUDR_020_DEFORESTATION,
                ),
            ],
            provenance_hash="a" * 64,
        )
        benchmarks = [
            CountryBenchmark(
                country_code="DE",
                benchmark_level=CountryBenchmarkLevel.LOW,
            ),
        ]
        # The engine uses .level, mock the attribute
        for b in benchmarks:
            b.level = b.benchmark_level
        result = engine.check_simplified_dd_eligibility(composite, benchmarks)
        assert result.eligible is True

    def test_simplified_dd_not_eligible_high_country(self):
        """HIGH country present -> not eligible."""
        engine = _make_engine()
        composite = CompositeRiskScore(
            overall_score=Decimal("20"),
            risk_level=RiskLevel.LOW,
            dimension_scores=[
                DimensionScore(
                    dimension=RiskDimension.DEFORESTATION,
                    weighted_score=Decimal("2.00"),
                    raw_score=Decimal("10"),
                    weight=Decimal("0.20"),
                    confidence=Decimal("0.90"),
                    source_agent=SourceAgent.EUDR_020_DEFORESTATION,
                ),
            ],
            provenance_hash="a" * 64,
        )
        benchmarks = [
            CountryBenchmark(
                country_code="BR",
                benchmark_level=CountryBenchmarkLevel.HIGH,
            ),
        ]
        for b in benchmarks:
            b.level = b.benchmark_level
        result = engine.check_simplified_dd_eligibility(composite, benchmarks)
        assert result.eligible is False

    def test_simplified_dd_not_eligible_high_score(self):
        """Composite score > 30 -> not eligible."""
        engine = _make_engine()
        composite = CompositeRiskScore(
            overall_score=Decimal("45"),
            risk_level=RiskLevel.STANDARD,
            dimension_scores=[],
            provenance_hash="a" * 64,
        )
        benchmarks = [
            CountryBenchmark(
                country_code="DE",
                benchmark_level=CountryBenchmarkLevel.LOW,
            ),
        ]
        for b in benchmarks:
            b.level = b.benchmark_level
        result = engine.check_simplified_dd_eligibility(composite, benchmarks)
        assert result.eligible is False


# ---------------------------------------------------------------------------
# Classification stats
# ---------------------------------------------------------------------------


class TestClassificationStats:
    """Test classification engine statistics."""

    def test_classification_stats(self):
        engine = _make_engine()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_classification_engine.record_risk_classification"
        ):
            engine.classify_risk(Decimal("50"))

        stats = engine.get_classification_stats()
        assert stats["total_classifications"] >= 1
        assert "hysteresis_applied" in stats
        assert "thresholds" in stats
        assert "hysteresis_buffer" in stats
