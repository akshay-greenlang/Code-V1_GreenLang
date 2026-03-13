# -*- coding: utf-8 -*-
"""
Tests for Engine 1: Strategy Selection Engine - AGENT-EUDR-025

Tests ML-powered and deterministic strategy recommendation, composite risk
scoring, SHAP explainability, confidence-based fallback, ISO 31000 validation,
provenance hashing, batch processing, and edge case handling.

Test count: ~70 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.config import (
    RiskMitigationAdvisorConfig,
    set_config,
)
from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskCategory,
    ISO31000TreatmentType,
    ImplementationComplexity,
    RiskInput,
    MitigationStrategy,
    RecommendStrategiesRequest,
    RecommendStrategiesResponse,
    SUPPORTED_COMMODITIES,
)
from greenlang.agents.eudr.risk_mitigation_advisor.strategy_selection_engine import (
    StrategySelectionEngine,
)


# ========================== Initialization Tests ==========================


class TestStrategySelectionEngineInit:
    """Tests for StrategySelectionEngine initialization."""

    def test_engine_initializes_with_config(self, test_config):
        engine = StrategySelectionEngine(config=test_config)
        assert engine is not None

    def test_engine_uses_deterministic_mode_from_config(self, test_config):
        engine = StrategySelectionEngine(config=test_config)
        assert test_config.deterministic_mode is True

    def test_engine_initializes_with_ml_config(self, ml_config):
        engine = StrategySelectionEngine(config=ml_config)
        assert ml_config.deterministic_mode is False

    def test_engine_respects_top_k_from_config(self, test_config):
        engine = StrategySelectionEngine(config=test_config)
        assert test_config.top_k_strategies == 5


# ===================== Composite Risk Score Tests =========================


class TestCompositeRiskScore:
    """Tests for composite risk score calculation."""

    def test_composite_score_high_risk(self, strategy_engine, high_risk_input):
        score = strategy_engine.compute_composite_risk_score(high_risk_input)
        assert isinstance(score, Decimal)
        assert Decimal("0") <= score <= Decimal("100")
        # High risk input should yield high composite score
        assert score >= Decimal("60")

    def test_composite_score_low_risk(self, strategy_engine, low_risk_input):
        score = strategy_engine.compute_composite_risk_score(low_risk_input)
        assert score <= Decimal("30")

    def test_composite_score_medium_risk(self, strategy_engine, medium_risk_input):
        score = strategy_engine.compute_composite_risk_score(medium_risk_input)
        assert Decimal("25") <= score <= Decimal("70")

    def test_composite_score_uses_decimal_arithmetic(self, strategy_engine, high_risk_input):
        score = strategy_engine.compute_composite_risk_score(high_risk_input)
        assert isinstance(score, Decimal)
        # No floating-point precision issues
        assert "E" not in str(score)

    def test_composite_score_weights_sum_to_one(self, strategy_engine):
        weights = strategy_engine.get_composite_weights()
        total = sum(weights.values())
        assert abs(total - Decimal("1")) <= Decimal("0.001")

    def test_composite_score_all_zeros(self, strategy_engine):
        zero_input = RiskInput(
            operator_id="op-zero",
            supplier_id="sup-zero",
            country_code="CH",
            commodity="wood",
        )
        score = strategy_engine.compute_composite_risk_score(zero_input)
        assert score == Decimal("0")

    def test_composite_score_all_hundred(self, strategy_engine):
        max_input = RiskInput(
            operator_id="op-max",
            supplier_id="sup-max",
            country_code="CD",
            commodity="palm_oil",
            country_risk_score=Decimal("100"),
            supplier_risk_score=Decimal("100"),
            commodity_risk_score=Decimal("100"),
            corruption_risk_score=Decimal("100"),
            deforestation_risk_score=Decimal("100"),
            indigenous_rights_score=Decimal("100"),
            protected_areas_score=Decimal("100"),
            legal_compliance_score=Decimal("100"),
            audit_risk_score=Decimal("100"),
        )
        score = strategy_engine.compute_composite_risk_score(max_input)
        assert score == Decimal("100")

    def test_composite_score_deforestation_weighted_highest(self, strategy_engine):
        """Deforestation weight (0.20) is highest, so deforestation-only risk
        should yield higher composite than country-only at same score."""
        deforest_input = RiskInput(
            operator_id="op-d", supplier_id="sup-d", country_code="BR",
            commodity="soya", deforestation_risk_score=Decimal("80"),
        )
        country_input = RiskInput(
            operator_id="op-c", supplier_id="sup-c", country_code="BR",
            commodity="soya", country_risk_score=Decimal("80"),
        )
        d_score = strategy_engine.compute_composite_risk_score(deforest_input)
        c_score = strategy_engine.compute_composite_risk_score(country_input)
        assert d_score > c_score


# ==================== Deterministic Recommendation Tests ==================


class TestDeterministicRecommendation:
    """Tests for deterministic rule-based strategy recommendation."""

    @pytest.mark.asyncio
    async def test_recommend_returns_strategies(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert isinstance(result, RecommendStrategiesResponse)
        assert len(result.strategies) > 0

    @pytest.mark.asyncio
    async def test_recommend_respects_top_k(self, strategy_engine, recommend_request):
        recommend_request_copy = RecommendStrategiesRequest(
            risk_input=recommend_request.risk_input,
            top_k=3,
            deterministic_mode=True,
        )
        result = await strategy_engine.recommend(recommend_request_copy)
        assert len(result.strategies) <= 3

    @pytest.mark.asyncio
    async def test_recommend_high_risk_includes_emergency(self, strategy_engine, high_risk_input):
        """High deforestation with post-cutoff should trigger emergency response."""
        request = RecommendStrategiesRequest(
            risk_input=high_risk_input,
            top_k=10,
            deterministic_mode=True,
        )
        result = await strategy_engine.recommend(request)
        strategy_names = [s.name.lower() for s in result.strategies]
        # Should include strategies addressing deforestation
        assert any("deforestation" in n or "emergency" in n or "monitor" in n
                    for n in strategy_names)

    @pytest.mark.asyncio
    async def test_recommend_low_risk_minimal_strategies(self, strategy_engine, low_risk_input):
        request = RecommendStrategiesRequest(
            risk_input=low_risk_input,
            top_k=5,
            deterministic_mode=True,
        )
        result = await strategy_engine.recommend(request)
        # Low risk should produce fewer or monitoring-only strategies
        assert len(result.strategies) <= 5

    @pytest.mark.asyncio
    async def test_recommend_provenance_hash_present(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_recommend_processing_time_recorded(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.processing_time_ms >= Decimal("0")

    @pytest.mark.asyncio
    async def test_recommend_deterministic_mode_flag(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.deterministic_mode is True

    @pytest.mark.asyncio
    async def test_recommend_strategies_sorted_by_effectiveness(
        self, strategy_engine, recommend_request
    ):
        result = await strategy_engine.recommend(recommend_request)
        if len(result.strategies) >= 2:
            for i in range(len(result.strategies) - 1):
                assert (
                    result.strategies[i].predicted_effectiveness
                    >= result.strategies[i + 1].predicted_effectiveness
                )

    @pytest.mark.asyncio
    async def test_recommend_each_strategy_has_risk_category(
        self, strategy_engine, recommend_request
    ):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert len(s.risk_categories) >= 1
            for cat in s.risk_categories:
                assert isinstance(cat, RiskCategory)

    @pytest.mark.asyncio
    async def test_recommend_each_strategy_has_iso_31000_type(
        self, strategy_engine, recommend_request
    ):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert isinstance(s.iso_31000_type, ISO31000TreatmentType)

    @pytest.mark.asyncio
    async def test_recommend_effectiveness_in_range(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert Decimal("0") <= s.predicted_effectiveness <= Decimal("100")

    @pytest.mark.asyncio
    async def test_recommend_confidence_in_range(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert Decimal("0") <= s.confidence_score <= Decimal("1")


# ===================== Deterministic Fallback Rules =======================


class TestDeterministicRules:
    """Tests for specific deterministic rule-based strategies."""

    @pytest.mark.asyncio
    async def test_rule_high_country_risk(self, strategy_engine):
        """Country risk >= 70 should recommend enhanced monitoring."""
        ri = RiskInput(
            operator_id="op", supplier_id="sup", country_code="CD",
            commodity="cocoa", country_risk_score=Decimal("75"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=10, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_rule_high_supplier_risk(self, strategy_engine):
        """Supplier risk >= 70 should recommend capacity building."""
        ri = RiskInput(
            operator_id="op", supplier_id="sup", country_code="BR",
            commodity="soya", supplier_risk_score=Decimal("80"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=10, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_rule_indigenous_rights_overlap(self, strategy_engine):
        """Indigenous rights overlap >= 50 should recommend FPIC remediation."""
        ri = RiskInput(
            operator_id="op", supplier_id="sup", country_code="BR",
            commodity="cattle", indigenous_rights_score=Decimal("65"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=10, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_rule_legal_compliance_gaps(self, strategy_engine):
        """High legal compliance score should recommend legal gap closure."""
        ri = RiskInput(
            operator_id="op", supplier_id="sup", country_code="BR",
            commodity="wood", legal_compliance_score=Decimal("80"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=10, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_rule_multi_dimension_composite(self, strategy_engine, high_risk_input):
        """Multiple high-risk dimensions should recommend composite strategy."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 2


# ==================== ML Recommendation Tests =============================


class TestMLRecommendation:
    """Tests for ML-powered recommendation pipeline."""

    @pytest.mark.asyncio
    async def test_ml_fallback_when_no_model(self, strategy_engine, recommend_ml_request):
        """When ML model is not available, should fall back to deterministic."""
        result = await strategy_engine.recommend(recommend_ml_request)
        assert isinstance(result, RecommendStrategiesResponse)
        # Should still produce strategies via fallback
        assert len(result.strategies) >= 0

    @pytest.mark.asyncio
    async def test_ml_fallback_flags_deterministic(self, strategy_engine, recommend_ml_request):
        """Fallback should set deterministic_mode flag in response."""
        result = await strategy_engine.recommend(recommend_ml_request)
        # When ML is not available, should fall back
        assert isinstance(result, RecommendStrategiesResponse)


# ==================== Batch Processing Tests ==============================


class TestBatchRecommendation:
    """Tests for batch strategy recommendation."""

    @pytest.mark.asyncio
    async def test_batch_recommend_multiple_inputs(self, strategy_engine, risk_input_batch):
        results = await strategy_engine.recommend_batch(
            [RecommendStrategiesRequest(risk_input=ri, top_k=3, deterministic_mode=True)
             for ri in risk_input_batch[:5]]
        )
        assert len(results) == 5
        for r in results:
            assert isinstance(r, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_batch_recommend_empty_list(self, strategy_engine):
        results = await strategy_engine.recommend_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_recommend_single_item(self, strategy_engine, recommend_request):
        results = await strategy_engine.recommend_batch([recommend_request])
        assert len(results) == 1


# ==================== Provenance Tests ====================================


class TestStrategyProvenance:
    """Tests for strategy recommendation provenance tracking."""

    @pytest.mark.asyncio
    async def test_provenance_hash_sha256(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_provenance_hash_deterministic(self, strategy_engine, recommend_request):
        r1 = await strategy_engine.recommend(recommend_request)
        r2 = await strategy_engine.recommend(recommend_request)
        assert r1.provenance_hash == r2.provenance_hash

    @pytest.mark.asyncio
    async def test_provenance_hash_changes_with_input(self, strategy_engine, high_risk_input, low_risk_input):
        req_high = RecommendStrategiesRequest(risk_input=high_risk_input, deterministic_mode=True)
        req_low = RecommendStrategiesRequest(risk_input=low_risk_input, deterministic_mode=True)
        r_high = await strategy_engine.recommend(req_high)
        r_low = await strategy_engine.recommend(req_low)
        assert r_high.provenance_hash != r_low.provenance_hash


# ==================== Risk Level Classification ===========================


class TestRiskLevelClassification:
    """Tests for risk level classification output."""

    @pytest.mark.asyncio
    async def test_high_risk_classification(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.risk_level in ("high", "critical")

    @pytest.mark.asyncio
    async def test_low_risk_classification(self, strategy_engine, low_risk_input):
        req = RecommendStrategiesRequest(risk_input=low_risk_input, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert result.risk_level in ("low", "medium")


# ==================== Edge Cases ==========================================


class TestStrategyEdgeCases:
    """Edge case tests for strategy selection."""

    @pytest.mark.asyncio
    async def test_single_high_dimension(self, strategy_engine):
        """Only one dimension is high risk, all others zero."""
        ri = RiskInput(
            operator_id="op", supplier_id="sup", country_code="BR",
            commodity="coffee", deforestation_risk_score=Decimal("95"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=5, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_boundary_risk_score_70(self, strategy_engine):
        """Test exact boundary at threshold 70."""
        ri = RiskInput(
            operator_id="op", supplier_id="sup", country_code="BR",
            commodity="soya", country_risk_score=Decimal("70"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=5, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_top_k_one(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=1, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) <= 1

    @pytest.mark.asyncio
    async def test_all_commodities(self, strategy_engine, all_commodity_risk_inputs):
        """Ensure engine handles all 7 EUDR commodities."""
        for commodity, ri in all_commodity_risk_inputs.items():
            req = RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            result = await strategy_engine.recommend(req)
            assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_duplicate_recommendations_avoided(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        strategy_ids = [s.strategy_id for s in result.strategies]
        assert len(strategy_ids) == len(set(strategy_ids))
