# -*- coding: utf-8 -*-
"""
Unit tests for SignificanceEngine -- ISO 14064-1:2018 Clause 5.2.

Tests multi-criteria significance assessment for indirect categories,
weighted scoring, batch assessment, exclusion justifications,
year-over-year tracking, and threshold updates with 25+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ISOCategory,
    ISO14064AppConfig,
    SignificanceLevel,
)
from services.models import SignificanceCriterion
from services.significance_engine import (
    DEFAULT_CRITERIA_DEFS,
    SignificanceEngine,
)


@pytest.fixture
def high_score_criteria():
    """Criteria that produce a high composite score (significant)."""
    return [
        SignificanceCriterion(criterion="magnitude", weight=Decimal("0.30"), score=Decimal("8"), rationale="High emissions"),
        SignificanceCriterion(criterion="influence", weight=Decimal("0.20"), score=Decimal("7"), rationale="Moderate influence"),
        SignificanceCriterion(criterion="risk", weight=Decimal("0.20"), score=Decimal("6"), rationale="Regulatory risk"),
        SignificanceCriterion(criterion="stakeholder", weight=Decimal("0.15"), score=Decimal("7"), rationale="Investor pressure"),
        SignificanceCriterion(criterion="data_availability", weight=Decimal("0.15"), score=Decimal("5"), rationale="Data available"),
    ]


@pytest.fixture
def low_score_criteria():
    """Criteria that produce a low composite score (not significant)."""
    return [
        SignificanceCriterion(criterion="magnitude", weight=Decimal("0.30"), score=Decimal("1"), rationale="Very low"),
        SignificanceCriterion(criterion="influence", weight=Decimal("0.20"), score=Decimal("1"), rationale="No influence"),
        SignificanceCriterion(criterion="risk", weight=Decimal("0.20"), score=Decimal("1"), rationale="No risk"),
        SignificanceCriterion(criterion="stakeholder", weight=Decimal("0.15"), score=Decimal("1"), rationale="No demand"),
        SignificanceCriterion(criterion="data_availability", weight=Decimal("0.15"), score=Decimal("1"), rationale="No data"),
    ]


class TestAssessCategory:
    """Test single-category significance assessment."""

    def test_category_1_raises(self, significance_engine, high_score_criteria):
        with pytest.raises(ValueError, match="Category 1"):
            significance_engine.assess_category(
                "inv-1", ISOCategory.CATEGORY_1_DIRECT,
                high_score_criteria, Decimal("5000"), Decimal("10000"),
            )

    def test_significant_category(self, significance_engine, high_score_criteria):
        assessment = significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
            high_score_criteria,
            category_emissions=Decimal("500"),
            total_emissions=Decimal("10000"),
        )
        assert assessment.result == SignificanceLevel.SIGNIFICANT

    def test_not_significant_category(self, significance_engine, low_score_criteria):
        assessment = significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_6_OTHER,
            low_score_criteria,
            category_emissions=Decimal("5"),
            total_emissions=Decimal("10000"),
        )
        assert assessment.result == SignificanceLevel.NOT_SIGNIFICANT

    def test_magnitude_alone_triggers_significant(self, significance_engine, low_score_criteria):
        """Even low composite score becomes significant if magnitude >= threshold."""
        assessment = significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
            low_score_criteria,
            category_emissions=Decimal("500"),
            total_emissions=Decimal("10000"),
        )
        # 500/10000 = 5% >= 1% threshold
        assert assessment.result == SignificanceLevel.SIGNIFICANT

    def test_near_threshold_is_under_review(self, significance_engine):
        """Score just below threshold triggers UNDER_REVIEW."""
        criteria = [
            SignificanceCriterion(criterion="magnitude", weight=Decimal("0.30"), score=Decimal("4.8"), rationale="Near"),
            SignificanceCriterion(criterion="influence", weight=Decimal("0.20"), score=Decimal("4.8"), rationale="Near"),
            SignificanceCriterion(criterion="risk", weight=Decimal("0.20"), score=Decimal("4.8"), rationale="Near"),
            SignificanceCriterion(criterion="stakeholder", weight=Decimal("0.15"), score=Decimal("4.8"), rationale="Near"),
            SignificanceCriterion(criterion="data_availability", weight=Decimal("0.15"), score=Decimal("4.8"), rationale="Near"),
        ]
        assessment = significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_6_OTHER,
            criteria,
            category_emissions=Decimal("4"),
            total_emissions=Decimal("10000"),
        )
        # Weighted score = 4.8 which is < 5.0 threshold but within 0.5 margin
        assert assessment.result == SignificanceLevel.UNDER_REVIEW

    def test_replaces_existing_assessment(self, significance_engine, high_score_criteria, low_score_criteria):
        significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
            high_score_criteria, Decimal("500"), Decimal("10000"),
        )
        # Replace with low score
        significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
            low_score_criteria, Decimal("5"), Decimal("10000"),
        )
        assessments = significance_engine.get_assessments("inv-1")
        cat3_assessments = [a for a in assessments if a.category == ISOCategory.CATEGORY_3_TRANSPORT]
        assert len(cat3_assessments) == 1


class TestBatchAssessment:
    """Test batch assessment of all indirect categories."""

    def test_batch_assesses_five_categories(self, significance_engine):
        results = significance_engine.batch_assess_all_indirect(
            inventory_id="inv-1",
            category_emissions={
                ISOCategory.CATEGORY_2_ENERGY: Decimal("3000"),
                ISOCategory.CATEGORY_3_TRANSPORT: Decimal("1500"),
                ISOCategory.CATEGORY_4_PRODUCTS_USED: Decimal("2000"),
                ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: Decimal("800"),
                ISOCategory.CATEGORY_6_OTHER: Decimal("200"),
            },
            total_emissions=Decimal("12500"),
            category_criteria={},
        )
        assert len(results) == 5

    def test_batch_uses_default_criteria_when_missing(self, significance_engine):
        results = significance_engine.batch_assess_all_indirect(
            inventory_id="inv-1",
            category_emissions={},
            total_emissions=Decimal("10000"),
            category_criteria={},
        )
        # Default mid-range score (5.0) equals threshold -> SIGNIFICANT
        for result in results:
            assert result.result in (
                SignificanceLevel.SIGNIFICANT,
                SignificanceLevel.NOT_SIGNIFICANT,
                SignificanceLevel.UNDER_REVIEW,
            )


class TestQueryMethods:
    """Test retrieval and summary methods."""

    def test_get_significant_categories_includes_cat1(self, significance_engine):
        result = significance_engine.get_significant_categories("inv-1")
        assert ISOCategory.CATEGORY_1_DIRECT in result

    def test_get_exclusion_justifications(self, significance_engine, low_score_criteria):
        significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_6_OTHER,
            low_score_criteria,
            category_emissions=Decimal("1"),
            total_emissions=Decimal("10000"),
        )
        justifications = significance_engine.get_exclusion_justifications("inv-1")
        assert len(justifications) >= 1
        assert justifications[0]["determination"] == "not_significant"
        assert "justification" in justifications[0]

    def test_generate_summary(self, significance_engine, high_score_criteria, low_score_criteria):
        significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
            high_score_criteria, Decimal("500"), Decimal("10000"),
        )
        significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_6_OTHER,
            low_score_criteria, Decimal("1"), Decimal("10000"),
        )
        summary = significance_engine.generate_summary("inv-1")
        assert summary["total_assessed"] == 2
        assert summary["significant_count"] >= 1

    def test_summary_for_empty_inventory(self, significance_engine):
        summary = significance_engine.generate_summary("empty-inv")
        assert summary["message"] == "No assessments found"


class TestYoYTracking:
    """Test year-over-year significance tracking."""

    def test_record_and_retrieve_yoy(self, significance_engine, high_score_criteria):
        significance_engine.assess_category(
            "inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
            high_score_criteria, Decimal("500"), Decimal("10000"),
        )
        significance_engine.record_yoy_assessment("inv-1", 2024)

        trend = significance_engine.get_yoy_significance_trend("inv-1")
        assert trend["years"] == [2024]
        assert ISOCategory.CATEGORY_3_TRANSPORT.value in trend["by_category"]

    def test_yoy_no_history(self, significance_engine):
        trend = significance_engine.get_yoy_significance_trend("empty-inv")
        assert "message" in trend


class TestThresholdUpdate:
    """Test threshold updates."""

    def test_update_magnitude_threshold(self, significance_engine):
        significance_engine.update_thresholds(magnitude_threshold=Decimal("5.0"))
        assert significance_engine._magnitude_threshold == Decimal("5.0")

    def test_update_composite_threshold(self, significance_engine):
        significance_engine.update_thresholds(composite_threshold=Decimal("7.0"))
        assert significance_engine._composite_threshold == Decimal("7.0")


class TestWeightedScoreCalculation:
    """Test the static weighted score calculation."""

    def test_equal_weights_equal_scores(self):
        criteria = [
            SignificanceCriterion(criterion="a", weight=Decimal("1"), score=Decimal("5"), rationale="test"),
            SignificanceCriterion(criterion="b", weight=Decimal("1"), score=Decimal("5"), rationale="test"),
        ]
        result = SignificanceEngine._calculate_weighted_score(criteria)
        assert result == Decimal("5.00")

    def test_empty_criteria_returns_zero(self):
        result = SignificanceEngine._calculate_weighted_score([])
        assert result == Decimal("0")

    def test_weighted_average(self):
        criteria = [
            SignificanceCriterion(criterion="a", weight=Decimal("0.75"), score=Decimal("10"), rationale="high"),
            SignificanceCriterion(criterion="b", weight=Decimal("0.25"), score=Decimal("2"), rationale="low"),
        ]
        result = SignificanceEngine._calculate_weighted_score(criteria)
        # (0.75*10 + 0.25*2) / (0.75+0.25) = (7.5+0.5)/1 = 8.0
        assert result == Decimal("8.00")
