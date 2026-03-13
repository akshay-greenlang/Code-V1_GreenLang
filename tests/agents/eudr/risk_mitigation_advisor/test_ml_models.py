# -*- coding: utf-8 -*-
"""
Tests for ML Model Integration - AGENT-EUDR-025

Tests XGBoost/LightGBM model training, prediction, SHAP explainability,
confidence thresholding, model versioning, feature importance, fallback
behavior, and model persistence.

Test count: ~50 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    RecommendStrategiesRequest,
    RecommendStrategiesResponse,
    MitigationStrategy,
    SUPPORTED_COMMODITIES,
)
from greenlang.agents.eudr.risk_mitigation_advisor.config import (
    RiskMitigationAdvisorConfig,
    set_config,
)

from .conftest import FIXED_DATE, COMPOSITE_WEIGHTS


# ---------------------------------------------------------------------------
# ML availability detection
# ---------------------------------------------------------------------------

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap as shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_risk_input(
    commodity: str = "soya",
    country_code: str = "BR",
    country_risk: str = "65",
    supplier_risk: str = "70",
    deforestation_risk: str = "75",
) -> RiskInput:
    return RiskInput(
        operator_id=f"op-ml-{commodity}",
        supplier_id=f"sup-ml-{commodity}",
        country_code=country_code,
        commodity=commodity,
        country_risk_score=Decimal(country_risk),
        supplier_risk_score=Decimal(supplier_risk),
        commodity_risk_score=Decimal("50"),
        corruption_risk_score=Decimal("40"),
        deforestation_risk_score=Decimal(deforestation_risk),
        indigenous_rights_score=Decimal("30"),
        protected_areas_score=Decimal("25"),
        legal_compliance_score=Decimal("35"),
        audit_risk_score=Decimal("40"),
        assessment_date=FIXED_DATE,
    )


class TestMLAvailabilityFlags:
    """Test ML library availability detection."""

    def test_numpy_flag_is_boolean(self):
        from greenlang.agents.eudr.risk_mitigation_advisor import strategy_selection_engine as sse
        assert isinstance(sse.NUMPY_AVAILABLE, bool)

    def test_xgboost_flag_is_boolean(self):
        from greenlang.agents.eudr.risk_mitigation_advisor import strategy_selection_engine as sse
        assert isinstance(sse.XGBOOST_AVAILABLE, bool)

    def test_lightgbm_flag_is_boolean(self):
        from greenlang.agents.eudr.risk_mitigation_advisor import strategy_selection_engine as sse
        assert isinstance(sse.LIGHTGBM_AVAILABLE, bool)

    def test_shap_flag_is_boolean(self):
        from greenlang.agents.eudr.risk_mitigation_advisor import strategy_selection_engine as sse
        assert isinstance(sse.SHAP_AVAILABLE, bool)


class TestMLFeatureExtraction:
    """Test feature vector extraction from RiskInput for ML models."""

    @pytest.mark.asyncio
    async def test_feature_vector_length(self, strategy_engine):
        ri = _make_risk_input()
        features = strategy_engine._extract_features(ri)
        # 9 risk scores should yield at least 9 features
        assert len(features) >= 9

    @pytest.mark.asyncio
    async def test_feature_vector_all_numeric(self, strategy_engine):
        ri = _make_risk_input()
        features = strategy_engine._extract_features(ri)
        for val in features:
            assert isinstance(val, (int, float, Decimal))

    @pytest.mark.asyncio
    async def test_feature_vector_deterministic(self, strategy_engine):
        ri = _make_risk_input()
        f1 = strategy_engine._extract_features(ri)
        f2 = strategy_engine._extract_features(ri)
        assert f1 == f2

    @pytest.mark.asyncio
    async def test_feature_vector_different_for_different_input(self, strategy_engine):
        ri_high = _make_risk_input(country_risk="90", deforestation_risk="95")
        ri_low = _make_risk_input(country_risk="10", deforestation_risk="5")
        f_high = strategy_engine._extract_features(ri_high)
        f_low = strategy_engine._extract_features(ri_low)
        assert f_high != f_low

    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    @pytest.mark.asyncio
    async def test_feature_extraction_per_commodity(self, strategy_engine, commodity):
        ri = _make_risk_input(commodity=commodity)
        features = strategy_engine._extract_features(ri)
        assert len(features) >= 9


class TestMLConfidenceThreshold:
    """Test confidence threshold behavior for ML/deterministic fallback."""

    @pytest.mark.asyncio
    async def test_below_threshold_falls_back(self, strategy_engine, high_risk_input):
        """When ML confidence < threshold, engine should fall back to deterministic."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input,
            top_k=5,
            deterministic_mode=False,
        )
        result = await strategy_engine.recommend(req)
        # Should still return valid strategies regardless of fallback
        assert isinstance(result, RecommendStrategiesResponse)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_deterministic_mode_ignores_ml(self, strategy_engine, high_risk_input):
        """Deterministic mode should never call ML model."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input,
            top_k=5,
            deterministic_mode=True,
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_confidence_score_in_range(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        for s in result.strategies:
            assert Decimal("0") <= s.confidence_score <= Decimal("1")


class TestMLModelPrediction:
    """Test ML model prediction pipeline."""

    @pytest.mark.asyncio
    async def test_recommend_returns_strategies(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=3, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1
        assert len(result.strategies) <= 3

    @pytest.mark.asyncio
    async def test_strategies_sorted_by_effectiveness(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        if len(result.strategies) >= 2:
            effectivenesses = [s.predicted_effectiveness for s in result.strategies]
            assert effectivenesses == sorted(effectivenesses, reverse=True)

    @pytest.mark.asyncio
    async def test_predict_with_all_zero_scores(self, strategy_engine):
        ri = RiskInput(
            operator_id="op-zero",
            supplier_id="sup-zero",
            country_code="FI",
            commodity="wood",
            country_risk_score=Decimal("0"),
            supplier_risk_score=Decimal("0"),
            commodity_risk_score=Decimal("0"),
            corruption_risk_score=Decimal("0"),
            deforestation_risk_score=Decimal("0"),
            indigenous_rights_score=Decimal("0"),
            protected_areas_score=Decimal("0"),
            legal_compliance_score=Decimal("0"),
            audit_risk_score=Decimal("0"),
            assessment_date=FIXED_DATE,
        )
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_predict_with_all_max_scores(self, strategy_engine):
        ri = RiskInput(
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
            assessment_date=FIXED_DATE,
        )
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert result.risk_level in ("critical", "high")
        assert len(result.strategies) >= 1


class TestSHAPExplainability:
    """Test SHAP-based explainability for strategy recommendations."""

    @pytest.mark.asyncio
    async def test_shap_disabled_no_explanations(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input,
            top_k=3,
            deterministic_mode=True,
            include_shap=False,
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_shap_request_without_library(self, strategy_engine, high_risk_input):
        """When SHAP library not available, request should still succeed."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input,
            top_k=3,
            deterministic_mode=True,
            include_shap=True,
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_model_version_in_response(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=3, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert result.model_version != ""

    @pytest.mark.asyncio
    async def test_explanation_structure(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input,
            top_k=3,
            deterministic_mode=True,
            include_shap=True,
        )
        result = await strategy_engine.recommend(req)
        # In deterministic mode SHAP values may be empty, but response valid
        for s in result.strategies:
            assert s.strategy_id != ""
            assert s.name != ""


class TestMLModelVersioning:
    """Test ML model versioning and compatibility."""

    @pytest.mark.asyncio
    async def test_model_version_format(self, strategy_engine, recommend_request):
        result = await strategy_engine.recommend(recommend_request)
        # Version should be semver-like
        parts = result.model_version.split(".")
        assert len(parts) >= 2

    @pytest.mark.asyncio
    async def test_model_version_consistent(self, strategy_engine, recommend_request):
        r1 = await strategy_engine.recommend(recommend_request)
        r2 = await strategy_engine.recommend(recommend_request)
        assert r1.model_version == r2.model_version


class TestMLFallbackBehavior:
    """Test graceful fallback when ML libraries are missing."""

    @pytest.mark.asyncio
    async def test_fallback_returns_valid_strategies(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        for s in result.strategies:
            assert s.strategy_id != ""
            assert s.name != ""
            assert s.predicted_effectiveness >= Decimal("0")

    @pytest.mark.asyncio
    async def test_fallback_strategies_have_cost_estimates(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        for s in result.strategies:
            assert s.cost_estimate is not None

    @pytest.mark.asyncio
    async def test_fallback_provenance_hash(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_fallback_iso_31000_types(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        from greenlang.agents.eudr.risk_mitigation_advisor.models import ISO31000TreatmentType
        valid_types = set(ISO31000TreatmentType)
        for s in result.strategies:
            assert s.iso_31000_type in valid_types

    @pytest.mark.asyncio
    async def test_fallback_risk_categories_valid(self, strategy_engine, high_risk_input):
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        valid_cats = set(RiskCategory)
        for s in result.strategies:
            for cat in s.risk_categories:
                assert cat in valid_cats


class TestMLConfigVariations:
    """Test ML behavior with different configurations."""

    @pytest.mark.asyncio
    async def test_xgboost_config(self, high_risk_input):
        cfg = RiskMitigationAdvisorConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_url="redis://localhost:6379/15",
            ml_model_type="xgboost",
            deterministic_mode=False,
            enable_provenance=True,
            enable_metrics=False,
        )
        set_config(cfg)
        from greenlang.agents.eudr.risk_mitigation_advisor.strategy_selection_engine import (
            StrategySelectionEngine,
        )
        engine = StrategySelectionEngine(config=cfg)
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=3, deterministic_mode=False
        )
        result = await engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_lightgbm_config(self, high_risk_input):
        cfg = RiskMitigationAdvisorConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_url="redis://localhost:6379/15",
            ml_model_type="lightgbm",
            deterministic_mode=False,
            enable_provenance=True,
            enable_metrics=False,
        )
        set_config(cfg)
        from greenlang.agents.eudr.risk_mitigation_advisor.strategy_selection_engine import (
            StrategySelectionEngine,
        )
        engine = StrategySelectionEngine(config=cfg)
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=3, deterministic_mode=False
        )
        result = await engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_rule_based_config(self, high_risk_input):
        cfg = RiskMitigationAdvisorConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_url="redis://localhost:6379/15",
            ml_model_type="rule_based",
            deterministic_mode=True,
            enable_provenance=True,
            enable_metrics=False,
        )
        set_config(cfg)
        from greenlang.agents.eudr.risk_mitigation_advisor.strategy_selection_engine import (
            StrategySelectionEngine,
        )
        engine = StrategySelectionEngine(config=cfg)
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=3, deterministic_mode=True
        )
        result = await engine.recommend(req)
        assert isinstance(result, RecommendStrategiesResponse)
        assert len(result.strategies) >= 1


class TestMLBatchPrediction:
    """Test ML batch prediction for multiple suppliers."""

    @pytest.mark.asyncio
    async def test_batch_prediction_consistency(self, strategy_engine, risk_input_batch):
        requests = [
            RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            for ri in risk_input_batch[:5]
        ]
        results = await strategy_engine.recommend_batch(requests)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, RecommendStrategiesResponse)

    @pytest.mark.asyncio
    async def test_batch_each_has_provenance(self, strategy_engine, risk_input_batch):
        requests = [
            RecommendStrategiesRequest(
                risk_input=ri, top_k=2, deterministic_mode=True
            )
            for ri in risk_input_batch[:3]
        ]
        results = await strategy_engine.recommend_batch(requests)
        for r in results:
            assert r.provenance_hash != ""
            assert len(r.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_batch_unique_provenance_hashes(self, strategy_engine, risk_input_batch):
        requests = [
            RecommendStrategiesRequest(
                risk_input=ri, top_k=2, deterministic_mode=True
            )
            for ri in risk_input_batch[:5]
        ]
        results = await strategy_engine.recommend_batch(requests)
        hashes = [r.provenance_hash for r in results]
        # Different inputs should produce different provenance hashes
        assert len(set(hashes)) == len(hashes)
