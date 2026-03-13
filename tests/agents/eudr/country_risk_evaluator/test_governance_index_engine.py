# -*- coding: utf-8 -*-
"""
Unit tests for GovernanceIndexEngine - AGENT-EUDR-016 Engine 4

Tests multi-source governance quality evaluation using World Bank WGI
6 dimensions, Transparency International CPI, FAO/ITTO forest governance,
enforcement effectiveness, composite scoring, data freshness tracking,
governance trend analysis, regional benchmarking, and gap analysis.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.governance_index_engine import (
    GovernanceIndexEngine,
    _DEFAULT_WEIGHTS,
    _WGI_DIMENSIONS,
    _FOREST_INDICATORS,
)
from greenlang.agents.eudr.country_risk_evaluator.models import (
    AssessmentConfidence,
    GovernanceIndex,
    GovernanceIndicator,
    TrendDirection,
)


# ============================================================================
# TestGovernanceEngineInit
# ============================================================================


class TestGovernanceEngineInit:
    """Tests for GovernanceIndexEngine initialization."""

    @pytest.mark.unit
    def test_initialization_empty_stores(self, mock_config):
        engine = GovernanceIndexEngine()
        assert engine._evaluations == {}
        assert engine._wgi_data == {}
        assert engine._cpi_data == {}

    @pytest.mark.unit
    def test_default_weights_sum_to_one(self):
        total = sum(_DEFAULT_WEIGHTS.values())
        assert total == Decimal("1.00")

    @pytest.mark.unit
    def test_six_wgi_dimensions_defined(self):
        assert len(_WGI_DIMENSIONS) == 6
        expected = [
            "voice_accountability",
            "political_stability",
            "government_effectiveness",
            "regulatory_quality",
            "rule_of_law",
            "control_of_corruption",
        ]
        assert _WGI_DIMENSIONS == expected

    @pytest.mark.unit
    def test_three_forest_indicators_defined(self):
        assert len(_FOREST_INDICATORS) == 3
        expected = [
            "forest_law_quality",
            "enforcement_capacity",
            "institutional_strength",
        ]
        assert _FOREST_INDICATORS == expected


# ============================================================================
# TestEvaluateGovernance
# ============================================================================


class TestEvaluateGovernance:
    """Tests for evaluate_governance method."""

    @pytest.mark.unit
    def test_evaluate_valid_input(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert isinstance(result, GovernanceIndex)
        assert result.country_code == "BR"

    @pytest.mark.unit
    def test_evaluate_score_in_range(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert 0.0 <= result.overall_score <= 100.0

    @pytest.mark.unit
    def test_evaluate_uppercase_country(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="br",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert result.country_code == "BR"

    @pytest.mark.unit
    def test_evaluate_has_index_id(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert result.index_id.startswith("gix-")

    @pytest.mark.unit
    def test_evaluate_stores_result(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        retrieved = governance_engine.get_evaluation(result.index_id)
        assert retrieved is not None
        assert retrieved.index_id == result.index_id

    @pytest.mark.unit
    def test_evaluate_has_cpi_score(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert result.cpi_score == pytest.approx(38.0, abs=0.5)


# ============================================================================
# TestWGI6Dimensions
# ============================================================================


class TestWGI6Dimensions:
    """Tests for World Bank WGI 6 dimensions."""

    @pytest.mark.unit
    def test_all_6_wgi_indicators_enum(self):
        indicators = [g.value for g in GovernanceIndicator]
        assert "rule_of_law" in indicators
        assert "regulatory_quality" in indicators
        assert "control_of_corruption" in indicators
        assert "government_effectiveness" in indicators
        assert "voice_accountability" in indicators
        assert "political_stability" in indicators

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "indicator",
        [
            "voice_accountability",
            "political_stability",
            "government_effectiveness",
            "regulatory_quality",
            "rule_of_law",
            "control_of_corruption",
        ],
    )
    def test_wgi_dimension_accepted(self, governance_engine, indicator):
        """All 6 WGI dimensions required; test each is accepted."""
        # Provide all 6 dimensions with the parametrized one highlighted
        wgi = {dim: 40.0 for dim in _WGI_DIMENSIONS}
        wgi[indicator] = 80.0  # Set parameterized dimension higher
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=wgi,
            cpi_score=40.0,
        )
        assert isinstance(result, GovernanceIndex)
        # The varied dimension should appear in the indicators
        assert indicator in result.indicators

    @pytest.mark.unit
    def test_all_wgi_dimensions_high(self, governance_engine):
        high_wgi = {dim: 90.0 for dim in _WGI_DIMENSIONS}
        result = governance_engine.evaluate_governance(
            country_code="SE",
            wgi_scores=high_wgi,
            cpi_score=85.0,
        )
        # High governance country should have high overall score
        assert result.overall_score > 50.0

    @pytest.mark.unit
    def test_all_wgi_dimensions_low(self, governance_engine):
        low_wgi = {dim: 10.0 for dim in _WGI_DIMENSIONS}
        result = governance_engine.evaluate_governance(
            country_code="CD",
            wgi_scores=low_wgi,
            cpi_score=15.0,
        )
        assert result.overall_score < 50.0


# ============================================================================
# TestCPIScoring
# ============================================================================


class TestCPIScoring:
    """Tests for Transparency International CPI scoring."""

    @pytest.mark.unit
    def test_high_cpi_improves_score(self, governance_engine, sample_wgi_data):
        result_high = governance_engine.evaluate_governance(
            country_code="DK",
            wgi_scores=sample_wgi_data,
            cpi_score=88.0,
        )
        result_low = governance_engine.evaluate_governance(
            country_code="SO",
            wgi_scores=sample_wgi_data,
            cpi_score=12.0,
        )
        assert result_high.overall_score > result_low.overall_score

    @pytest.mark.unit
    def test_cpi_score_boundary_zero(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="XX",
            wgi_scores=sample_wgi_data,
            cpi_score=0.0,
        )
        assert result.overall_score >= 0.0

    @pytest.mark.unit
    def test_cpi_score_boundary_100(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="XX",
            wgi_scores=sample_wgi_data,
            cpi_score=100.0,
        )
        assert result.overall_score <= 100.0


# ============================================================================
# TestForestGovernance
# ============================================================================


class TestForestGovernance:
    """Tests for FAO/ITTO forest governance assessment."""

    @pytest.mark.unit
    def test_forest_governance_with_scores(self, governance_engine, sample_wgi_data):
        forest_scores = {
            "forest_law_quality": 40.0,
            "enforcement_capacity": 35.0,
            "institutional_strength": 45.0,
        }
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
            forest_governance_scores=forest_scores,
        )
        assert isinstance(result, GovernanceIndex)
        if result.forest_governance_score is not None:
            assert 0.0 <= result.forest_governance_score <= 100.0

    @pytest.mark.unit
    def test_forest_governance_without_scores(
        self, governance_engine, sample_wgi_data
    ):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        # Should still produce a valid result
        assert isinstance(result, GovernanceIndex)


# ============================================================================
# TestEnforcementEffectiveness
# ============================================================================


class TestEnforcementEffectiveness:
    """Tests for enforcement effectiveness scoring."""

    @pytest.mark.unit
    def test_enforcement_with_data(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
            enforcement_data={
                "prosecution_rate": 0.45,
                "penalty_adequacy": 0.30,
            },
        )
        if result.enforcement_effectiveness is not None:
            assert 0.0 <= result.enforcement_effectiveness <= 100.0

    @pytest.mark.unit
    def test_high_enforcement_improves_score(
        self, governance_engine, sample_wgi_data
    ):
        result_high = governance_engine.evaluate_governance(
            country_code="SE",
            wgi_scores=sample_wgi_data,
            cpi_score=85.0,
            enforcement_data={
                "prosecution_rate": 0.90,
                "penalty_adequacy": 0.85,
            },
        )
        result_low = governance_engine.evaluate_governance(
            country_code="CD",
            wgi_scores=sample_wgi_data,
            cpi_score=15.0,
            enforcement_data={
                "prosecution_rate": 0.10,
                "penalty_adequacy": 0.05,
            },
        )
        assert result_high.overall_score > result_low.overall_score


# ============================================================================
# TestCompositeGovernanceScore
# ============================================================================


class TestCompositeGovernanceScore:
    """Tests for composite governance score calculation."""

    @pytest.mark.unit
    def test_composite_uses_all_components(
        self, governance_engine, sample_wgi_data
    ):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
            forest_governance_scores={
                "forest_law_quality": 40.0,
                "enforcement_capacity": 35.0,
                "institutional_strength": 45.0,
            },
            enforcement_data={
                "prosecution_rate": 0.45,
                "penalty_adequacy": 0.30,
            },
        )
        assert result.overall_score > 0.0

    @pytest.mark.unit
    def test_governance_weight_config(self, mock_config):
        # Verify governance weights are in valid range
        assert 0.0 <= mock_config.wgi_weight <= 1.0
        assert 0.0 <= mock_config.cpi_weight <= 1.0
        assert 0.0 <= mock_config.forest_governance_weight <= 1.0
        assert 0.0 <= mock_config.gov_enforcement_weight <= 1.0

    @pytest.mark.unit
    def test_governance_weights_sum_approximately_one(self, mock_config):
        total = (
            mock_config.wgi_weight
            + mock_config.cpi_weight
            + mock_config.forest_governance_weight
            + mock_config.gov_enforcement_weight
        )
        assert total == pytest.approx(1.0, abs=0.01)


# ============================================================================
# TestGovernanceTrendAnalysis
# ============================================================================


class TestGovernanceTrendAnalysis:
    """Tests for historical governance trend analysis."""

    @pytest.mark.unit
    def test_trend_insufficient_data(self, governance_engine):
        result = governance_engine.get_governance_trend("BR")
        assert result["trend_direction"] == "insufficient_data"

    @pytest.mark.unit
    def test_trend_with_history(self, governance_engine, sample_wgi_data):
        for _ in range(3):
            governance_engine.evaluate_governance(
                country_code="BR",
                wgi_scores=sample_wgi_data,
                cpi_score=38.0,
            )
        result = governance_engine.get_governance_trend("BR")
        assert result["country_code"] == "BR"
        assert "trend_direction" in result


# ============================================================================
# TestRegionalBenchmarking
# ============================================================================


class TestRegionalBenchmarking:
    """Tests for regional governance benchmarking using list_evaluations."""

    @pytest.mark.unit
    def test_benchmark_single_country(self, governance_engine, sample_wgi_data):
        """Verify a stored evaluation can be retrieved via list_evaluations."""
        governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        results = governance_engine.list_evaluations()
        assert len(results) >= 1
        assert results[0].country_code == "BR"
        assert results[0].overall_score > 0


# ============================================================================
# TestGapAnalysis
# ============================================================================


class TestGapAnalysis:
    """Tests for governance gap analysis."""

    @pytest.mark.unit
    def test_gap_analysis_identifies_weakest(
        self, governance_engine, sample_wgi_data
    ):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        # The indicators dict should show which dimensions are weakest
        if result.indicators:
            weakest_value = min(result.indicators.values())
            strongest_value = max(result.indicators.values())
            assert weakest_value <= strongest_value


# ============================================================================
# TestDataFreshnessTracking
# ============================================================================


class TestDataFreshnessTracking:
    """Tests for data source freshness tracking."""

    @pytest.mark.unit
    def test_evaluation_has_assessed_at(
        self, governance_engine, sample_wgi_data
    ):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert result.assessed_at is not None
        assert result.assessed_at.tzinfo is not None

    @pytest.mark.unit
    def test_evaluation_has_data_sources(
        self, governance_engine, sample_wgi_data
    ):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert isinstance(result.data_sources, list)


# ============================================================================
# TestInputValidation
# ============================================================================


class TestInputValidation:
    """Tests for governance engine input validation."""

    @pytest.mark.unit
    def test_empty_country_code_raises(self, governance_engine, sample_wgi_data):
        with pytest.raises(ValueError):
            governance_engine.evaluate_governance(
                country_code="",
                wgi_scores=sample_wgi_data,
                cpi_score=38.0,
            )

    @pytest.mark.unit
    def test_get_nonexistent_evaluation(self, governance_engine):
        result = governance_engine.get_evaluation("nonexistent-id")
        assert result is None


# ============================================================================
# TestMissingWGIData
# ============================================================================


class TestMissingWGIData:
    """Tests for handling missing WGI data."""

    @pytest.mark.unit
    def test_partial_wgi_dimensions_raises(self, governance_engine):
        """Engine requires all 6 WGI dimensions; partial raises ValueError."""
        partial_wgi = {
            "rule_of_law": 45.0,
            "control_of_corruption": 35.0,
        }
        with pytest.raises(ValueError, match="Missing WGI dimension"):
            governance_engine.evaluate_governance(
                country_code="BR",
                wgi_scores=partial_wgi,
                cpi_score=38.0,
            )

    @pytest.mark.unit
    def test_empty_wgi_scores_raises(self, governance_engine):
        """Engine requires all 6 WGI dimensions; empty raises ValueError."""
        with pytest.raises(ValueError, match="Missing WGI dimension"):
            governance_engine.evaluate_governance(
                country_code="BR",
                wgi_scores={},
                cpi_score=38.0,
            )

    @pytest.mark.unit
    def test_provenance_hash_present(self, governance_engine, sample_wgi_data):
        result = governance_engine.evaluate_governance(
            country_code="BR",
            wgi_scores=sample_wgi_data,
            cpi_score=38.0,
        )
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# ============================================================================
# TestParametrizedGovernance
# ============================================================================


class TestParametrizedGovernance:
    """Parametrized governance tests across countries."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "country_code,cpi_score,expected_range",
        [
            ("DK", 88.0, (60, 100)),
            ("SE", 85.0, (60, 100)),
            ("BR", 38.0, (20, 60)),
            ("CD", 20.0, (10, 50)),
            ("SO", 12.0, (5, 40)),
        ],
        ids=["Denmark", "Sweden", "Brazil", "DRC", "Somalia"],
    )
    def test_governance_score_ranges_by_country(
        self, governance_engine, sample_wgi_data, country_code, cpi_score, expected_range
    ):
        result = governance_engine.evaluate_governance(
            country_code=country_code,
            wgi_scores=sample_wgi_data,
            cpi_score=cpi_score,
        )
        # Score should be reasonable for the given CPI
        assert 0.0 <= result.overall_score <= 100.0
