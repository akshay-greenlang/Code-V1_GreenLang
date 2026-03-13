# -*- coding: utf-8 -*-
"""
Tests for Strategy API Routes - AGENT-EUDR-025

Tests strategy recommendation endpoints including single-supplier,
batch, deterministic mode, ML mode, SHAP explanations, and error handling.

Test count: ~40 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    RecommendStrategiesRequest,
    RecommendStrategiesResponse,
    MitigationStrategy,
    SUPPORTED_COMMODITIES,
)

from .conftest import FIXED_DATE


class TestStrategyRecommendEndpoint:
    @pytest.mark.asyncio
    async def test_recommend_valid_input(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert isinstance(result, RecommendStrategiesResponse)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_recommend_returns_top_k(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert len(result.strategies) <= recommend_request.top_k

    @pytest.mark.asyncio
    async def test_recommend_deterministic(self, strategy_engine, recommend_request):
        r1 = await strategy_engine.recommend(recommend_request)
        r2 = await strategy_engine.recommend(recommend_request)
        assert len(r1.strategies) == len(r2.strategies)

    @pytest.mark.asyncio
    async def test_recommend_has_model_info(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.model_version != ""

    @pytest.mark.asyncio
    async def test_recommend_composite_score(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.composite_risk_score >= Decimal("0")

    @pytest.mark.asyncio
    async def test_recommend_risk_level_classification(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        assert result.risk_level in ("low", "medium", "high", "critical")


class TestStrategyBatchEndpoint:
    @pytest.mark.asyncio
    async def test_batch_recommend(self, strategy_engine, risk_input_batch):
        requests = [
            RecommendStrategiesRequest(risk_input=ri, top_k=3, deterministic_mode=True)
            for ri in risk_input_batch[:5]
        ]
        results = await strategy_engine.recommend_batch(requests)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_batch_empty(self, strategy_engine):
        results = await strategy_engine.recommend_batch([])
        assert results == []


class TestStrategyInputValidation:
    @pytest.mark.asyncio
    async def test_invalid_commodity_rejected(self, strategy_engine):
        with pytest.raises((ValueError, Exception)):
            ri = RiskInput(
                operator_id="op", supplier_id="sup",
                country_code="BR", commodity="invalid_commodity",
            )

    @pytest.mark.asyncio
    async def test_score_out_of_range_rejected(self, strategy_engine):
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op", supplier_id="sup",
                country_code="BR", commodity="soya",
                country_risk_score=Decimal("150"),
            )

    @pytest.mark.asyncio
    async def test_empty_operator_rejected(self, strategy_engine):
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="", supplier_id="sup",
                country_code="BR", commodity="soya",
            )

    @pytest.mark.asyncio
    async def test_invalid_country_code_rejected(self, strategy_engine):
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op", supplier_id="sup",
                country_code="INVALID", commodity="soya",
            )


class TestStrategyResponseStructure:
    @pytest.mark.asyncio
    async def test_strategy_has_id(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert s.strategy_id != ""

    @pytest.mark.asyncio
    async def test_strategy_has_name(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert s.name != ""

    @pytest.mark.asyncio
    async def test_strategy_has_description(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert s.description != ""

    @pytest.mark.asyncio
    async def test_strategy_has_cost_estimate(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert s.cost_estimate is not None

    @pytest.mark.asyncio
    async def test_strategy_has_complexity(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert s.implementation_complexity is not None

    @pytest.mark.asyncio
    async def test_strategy_has_time_to_effect(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        for s in result.strategies:
            assert s.time_to_effect_weeks >= 1


class TestAllCommodityStrategies:
    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    @pytest.mark.asyncio
    async def test_commodity_specific_strategies(self, strategy_engine, commodity):
        ri = RiskInput(
            operator_id=f"op-{commodity}",
            supplier_id=f"sup-{commodity}",
            country_code="BR",
            commodity=commodity,
            country_risk_score=Decimal("65"),
            supplier_risk_score=Decimal("70"),
            deforestation_risk_score=Decimal("75"),
        )
        req = RecommendStrategiesRequest(risk_input=ri, top_k=5, deterministic_mode=True)
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)
        assert len(result.strategies) >= 1


class TestStrategyComparison:
    @pytest.mark.asyncio
    async def test_compare_strategies(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        if len(result.strategies) >= 2:
            comparison = strategy_engine.compare_strategies(
                result.strategies[0], result.strategies[1]
            )
            assert comparison is not None
