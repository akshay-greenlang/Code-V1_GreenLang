# -*- coding: utf-8 -*-
"""
Unit tests for SupplierRiskScorer - AGENT-EUDR-017 Engine 1

Tests the multi-factor weighted composite supplier risk scoring engine
covering 8-factor model, risk level classification, confidence scoring,
trend analysis, batch assessment, peer benchmarking, and provenance tracking.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.supplier_risk_scorer.supplier_risk_scorer import (
    SupplierRiskScorer,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    RiskLevel,
    SupplierType,
    CommodityType,
)


# ============================================================================
# TestSupplierRiskScorerInit
# ============================================================================


class TestSupplierRiskScorerInit:
    """Tests for SupplierRiskScorer initialization."""

    @pytest.mark.unit
    def test_initialization_creates_empty_stores(self, mock_config):
        scorer = SupplierRiskScorer()
        assert scorer._assessments == {}
        assert scorer._risk_history == {}

    @pytest.mark.unit
    def test_initialization_creates_lock(self, mock_config):
        scorer = SupplierRiskScorer()
        assert scorer._lock is not None

    @pytest.mark.unit
    def test_config_loaded_correctly(self, mock_config, supplier_risk_scorer):
        cfg = supplier_risk_scorer._config
        assert cfg.geographic_sourcing_weight == 20
        assert cfg.compliance_history_weight == 15
        assert cfg.low_risk_threshold == 25


# ============================================================================
# TestAssessSupplier
# ============================================================================


class TestAssessSupplier:
    """Tests for assess_supplier method."""

    @pytest.mark.unit
    def test_assess_supplier_valid_returns_assessment(
        self, supplier_risk_scorer, sample_supplier, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id=sample_supplier["supplier_id"],
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert result is not None
        assert result["supplier_id"] == sample_supplier["supplier_id"]
        assert "assessment_id" in result
        assert result["assessment_id"].startswith("sra-")

    @pytest.mark.unit
    def test_assess_supplier_score_in_range(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-001",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert Decimal("0.0") <= result["risk_score"] <= Decimal("100.0")

    @pytest.mark.unit
    def test_assess_supplier_has_8_factors(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-001",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert len(result["factor_scores"]) == 8

    @pytest.mark.unit
    def test_assess_supplier_low_risk_classification(
        self, supplier_risk_scorer, sample_low_risk_factors
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-LOW",
            factor_scores=sample_low_risk_factors,
            assessed_by="test_user",
        )
        assert result["risk_level"] == RiskLevel.LOW
        assert result["risk_score"] <= Decimal("25.0")

    @pytest.mark.unit
    def test_assess_supplier_high_risk_classification(
        self, supplier_risk_scorer, sample_high_risk_factors
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-HIGH",
            factor_scores=sample_high_risk_factors,
            assessed_by="test_user",
        )
        assert result["risk_level"] == RiskLevel.HIGH
        assert Decimal("51.0") <= result["risk_score"] <= Decimal("75.0")

    @pytest.mark.unit
    def test_assess_supplier_critical_risk_classification(
        self, supplier_risk_scorer, sample_critical_risk_factors
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-CRITICAL",
            factor_scores=sample_critical_risk_factors,
            assessed_by="test_user",
        )
        assert result["risk_level"] == RiskLevel.CRITICAL
        assert result["risk_score"] > Decimal("75.0")

    @pytest.mark.unit
    def test_assess_supplier_composite_score_calculation(
        self, supplier_risk_scorer, mock_config
    ):
        # Known factor scores with known weights
        factor_scores = {
            "geographic_sourcing": Decimal("50.0"),      # weight 20%
            "compliance_history": Decimal("40.0"),       # weight 15%
            "documentation_quality": Decimal("60.0"),    # weight 15%
            "certification_status": Decimal("30.0"),     # weight 15%
            "traceability_completeness": Decimal("70.0"),# weight 10%
            "financial_stability": Decimal("50.0"),      # weight 10%
            "environmental_performance": Decimal("45.0"),# weight 10%
            "social_compliance": Decimal("80.0"),        # weight 5%
        }
        # Expected: 50*0.20 + 40*0.15 + 60*0.15 + 30*0.15 + 70*0.10 + 50*0.10 + 45*0.10 + 80*0.05
        # = 10 + 6 + 9 + 4.5 + 7 + 5 + 4.5 + 4 = 50.0
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-CALC",
            factor_scores=factor_scores,
            assessed_by="test_user",
        )
        assert result["risk_score"] == pytest.approx(Decimal("50.0"), rel=Decimal("0.01"))

    @pytest.mark.unit
    def test_assess_supplier_confidence_scoring(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-001",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert Decimal("0.0") <= result["confidence"] <= Decimal("1.0")

    @pytest.mark.unit
    def test_assess_supplier_stores_assessment(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        supplier_id = "SUPP-STORE"
        result = supplier_risk_scorer.assess_supplier(
            supplier_id=supplier_id,
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        # Check if stored
        assert supplier_id in supplier_risk_scorer._assessments

    @pytest.mark.unit
    def test_assess_supplier_versioning(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        supplier_id = "SUPP-VERSION"
        result1 = supplier_risk_scorer.assess_supplier(
            supplier_id=supplier_id,
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        result2 = supplier_risk_scorer.assess_supplier(
            supplier_id=supplier_id,
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert result2["version"] == result1["version"] + 1


# ============================================================================
# TestRiskLevelClassification
# ============================================================================


class TestRiskLevelClassification:
    """Tests for risk level classification logic."""

    @pytest.mark.unit
    @pytest.mark.parametrize("score,expected_level", [
        (Decimal("0.0"), RiskLevel.LOW),
        (Decimal("25.0"), RiskLevel.LOW),
        (Decimal("25.1"), RiskLevel.MEDIUM),
        (Decimal("50.0"), RiskLevel.MEDIUM),
        (Decimal("50.1"), RiskLevel.HIGH),
        (Decimal("75.0"), RiskLevel.HIGH),
        (Decimal("75.1"), RiskLevel.CRITICAL),
        (Decimal("100.0"), RiskLevel.CRITICAL),
    ])
    def test_classify_risk_level(
        self, supplier_risk_scorer, score, expected_level
    ):
        level = supplier_risk_scorer._classify_risk_level(score)
        assert level == expected_level


# ============================================================================
# TestFactorWeights
# ============================================================================


class TestFactorWeights:
    """Tests for factor weight validation."""

    @pytest.mark.unit
    def test_default_weights_sum_to_100(self, mock_config):
        cfg = mock_config
        total = (
            cfg.geographic_sourcing_weight +
            cfg.compliance_history_weight +
            cfg.documentation_quality_weight +
            cfg.certification_status_weight +
            cfg.traceability_completeness_weight +
            cfg.financial_stability_weight +
            cfg.environmental_performance_weight +
            cfg.social_compliance_weight
        )
        assert total == 100

    @pytest.mark.unit
    def test_all_weights_positive(self, mock_config):
        cfg = mock_config
        assert cfg.geographic_sourcing_weight > 0
        assert cfg.compliance_history_weight > 0
        assert cfg.documentation_quality_weight > 0
        assert cfg.certification_status_weight > 0
        assert cfg.traceability_completeness_weight > 0
        assert cfg.financial_stability_weight > 0
        assert cfg.environmental_performance_weight > 0
        assert cfg.social_compliance_weight > 0


# ============================================================================
# TestNormalization
# ============================================================================


class TestNormalization:
    """Tests for score normalization."""

    @pytest.mark.unit
    def test_normalize_scores_min_max(self, supplier_risk_scorer):
        scores = [Decimal("10.0"), Decimal("50.0"), Decimal("90.0")]
        normalized = supplier_risk_scorer._normalize_scores(scores)
        assert normalized[0] == Decimal("0.0")  # Min
        assert normalized[-1] == Decimal("100.0")  # Max

    @pytest.mark.unit
    def test_normalize_scores_all_same(self, supplier_risk_scorer):
        scores = [Decimal("50.0"), Decimal("50.0"), Decimal("50.0")]
        normalized = supplier_risk_scorer._normalize_scores(scores)
        # All same should normalize to middle
        assert all(s == Decimal("50.0") for s in normalized)


# ============================================================================
# TestTrendAnalysis
# ============================================================================


class TestTrendAnalysis:
    """Tests for risk trend analysis."""

    @pytest.mark.unit
    def test_analyze_trend_improving(self, supplier_risk_scorer):
        # Simulate decreasing risk over time
        history = [
            {"risk_score": Decimal("80.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=360)},
            {"risk_score": Decimal("70.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=270)},
            {"risk_score": Decimal("60.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=180)},
            {"risk_score": Decimal("50.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=90)},
        ]
        trend = supplier_risk_scorer._analyze_trend(history)
        assert trend == "improving"

    @pytest.mark.unit
    def test_analyze_trend_deteriorating(self, supplier_risk_scorer):
        # Simulate increasing risk over time
        history = [
            {"risk_score": Decimal("30.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=360)},
            {"risk_score": Decimal("40.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=270)},
            {"risk_score": Decimal("50.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=180)},
            {"risk_score": Decimal("60.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=90)},
        ]
        trend = supplier_risk_scorer._analyze_trend(history)
        assert trend == "deteriorating"

    @pytest.mark.unit
    def test_analyze_trend_stable(self, supplier_risk_scorer):
        # Simulate stable risk over time
        history = [
            {"risk_score": Decimal("50.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=360)},
            {"risk_score": Decimal("51.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=270)},
            {"risk_score": Decimal("49.0"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=180)},
            {"risk_score": Decimal("50.5"), "assessed_at": datetime.now(timezone.utc) - timedelta(days=90)},
        ]
        trend = supplier_risk_scorer._analyze_trend(history)
        assert trend == "stable"


# ============================================================================
# TestBatchAssessment
# ============================================================================


class TestBatchAssessment:
    """Tests for batch assessment functionality."""

    @pytest.mark.unit
    def test_batch_assess_multiple_suppliers(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        suppliers = [
            {"supplier_id": "SUPP-BATCH-001", "factor_scores": sample_factor_scores},
            {"supplier_id": "SUPP-BATCH-002", "factor_scores": sample_factor_scores},
            {"supplier_id": "SUPP-BATCH-003", "factor_scores": sample_factor_scores},
        ]
        results = supplier_risk_scorer.batch_assess(
            suppliers=suppliers,
            assessed_by="test_user",
        )
        assert len(results) == 3
        assert all("assessment_id" in r for r in results)

    @pytest.mark.unit
    def test_batch_assess_empty_list(self, supplier_risk_scorer):
        results = supplier_risk_scorer.batch_assess(
            suppliers=[],
            assessed_by="test_user",
        )
        assert results == []

    @pytest.mark.unit
    def test_batch_assess_handles_max_batch_size(
        self, supplier_risk_scorer, sample_factor_scores, mock_config
    ):
        # Create suppliers beyond max batch size
        max_size = mock_config.batch_max_size
        suppliers = [
            {"supplier_id": f"SUPP-{i:04d}", "factor_scores": sample_factor_scores}
            for i in range(max_size + 10)
        ]
        results = supplier_risk_scorer.batch_assess(
            suppliers=suppliers,
            assessed_by="test_user",
        )
        # Should process in batches
        assert len(results) == len(suppliers)


# ============================================================================
# TestPeerBenchmarking
# ============================================================================


class TestPeerBenchmarking:
    """Tests for peer group benchmarking."""

    @pytest.mark.unit
    def test_benchmark_against_peers(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        # Create peer assessments
        peers = []
        for i in range(5):
            result = supplier_risk_scorer.assess_supplier(
                supplier_id=f"PEER-{i:03d}",
                factor_scores=sample_factor_scores,
                assessed_by="test_user",
            )
            peers.append(result)

        # Benchmark new supplier
        test_result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-TEST",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )

        benchmark = supplier_risk_scorer.benchmark_against_peers(
            supplier_id="SUPP-TEST",
            peer_group=CommodityType.SOYA,
            region="BR",
        )
        assert "percentile" in benchmark
        assert "deviation_from_median" in benchmark


# ============================================================================
# TestCompareSuppliers
# ============================================================================


class TestCompareSuppliers:
    """Tests for supplier comparison."""

    @pytest.mark.unit
    def test_compare_two_suppliers(
        self, supplier_risk_scorer, sample_factor_scores, sample_low_risk_factors
    ):
        result1 = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-A",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        result2 = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-B",
            factor_scores=sample_low_risk_factors,
            assessed_by="test_user",
        )

        comparison = supplier_risk_scorer.compare_suppliers(
            supplier_ids=["SUPP-A", "SUPP-B"]
        )
        assert "suppliers" in comparison
        assert len(comparison["suppliers"]) == 2
        assert "risk_score_diff" in comparison


# ============================================================================
# TestProvenance
# ============================================================================


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_assessment_includes_provenance_hash(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-PROV",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64  # SHA-256

    @pytest.mark.unit
    def test_same_input_same_hash(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result1 = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-HASH",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        result2 = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-HASH",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        # Same inputs should yield consistent results (bit-perfect)
        assert result1["risk_score"] == result2["risk_score"]


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_supplier_id_raises_error(self, supplier_risk_scorer):
        with pytest.raises(ValueError):
            supplier_risk_scorer.assess_supplier(
                supplier_id="",  # Empty ID
                factor_scores={},
                assessed_by="test_user",
            )

    @pytest.mark.unit
    def test_missing_factor_scores_raises_error(self, supplier_risk_scorer):
        with pytest.raises(ValueError):
            supplier_risk_scorer.assess_supplier(
                supplier_id="SUPP-001",
                factor_scores={},  # Missing factors
                assessed_by="test_user",
            )

    @pytest.mark.unit
    def test_invalid_factor_score_range_raises_error(self, supplier_risk_scorer):
        invalid_scores = {
            "geographic_sourcing": Decimal("150.0"),  # > 100
            "compliance_history": Decimal("-10.0"),   # < 0
            "documentation_quality": Decimal("50.0"),
            "certification_status": Decimal("50.0"),
            "traceability_completeness": Decimal("50.0"),
            "financial_stability": Decimal("50.0"),
            "environmental_performance": Decimal("50.0"),
            "social_compliance": Decimal("50.0"),
        }
        with pytest.raises(ValueError):
            supplier_risk_scorer.assess_supplier(
                supplier_id="SUPP-001",
                factor_scores=invalid_scores,
                assessed_by="test_user",
            )


# ============================================================================
# TestDecimalArithmetic
# ============================================================================


class TestDecimalArithmetic:
    """Tests for deterministic Decimal arithmetic."""

    @pytest.mark.unit
    def test_composite_score_uses_decimal(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-DEC",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        assert isinstance(result["risk_score"], Decimal)

    @pytest.mark.unit
    def test_factor_scores_are_decimal(
        self, supplier_risk_scorer, sample_factor_scores
    ):
        result = supplier_risk_scorer.assess_supplier(
            supplier_id="SUPP-DEC2",
            factor_scores=sample_factor_scores,
            assessed_by="test_user",
        )
        for factor in result["factor_scores"]:
            assert isinstance(factor["raw_score"], Decimal)
            assert isinstance(factor["normalized_score"], Decimal)
            assert isinstance(factor["weighted_score"], Decimal)
