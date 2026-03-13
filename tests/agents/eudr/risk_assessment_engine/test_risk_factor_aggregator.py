# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-028 RiskFactorAggregator.

Tests upstream agent data collection (simulated stubs), score normalization,
synthetic dimension derivation, and aggregation statistics.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    RiskDimension,
    RiskFactorInput,
    SourceAgent,
)


def _make_aggregator():
    """Instantiate RiskFactorAggregator with mocked config and metrics."""
    from greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator import (
        RiskFactorAggregator,
    )
    cfg = MagicMock(spec=RiskAssessmentEngineConfig)
    cfg.country_risk_url = "http://localhost:8016"
    cfg.supplier_risk_url = "http://localhost:8017"
    cfg.commodity_risk_url = "http://localhost:8018"
    cfg.corruption_index_url = "http://localhost:8019"
    cfg.deforestation_alert_url = "http://localhost:8020"
    with patch(
        "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.record_factor_aggregation"
    ), patch(
        "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.observe_aggregation_duration"
    ):
        return RiskFactorAggregator(config=cfg)


class TestAggregateFactors:
    """Test aggregate_factors end-to-end."""

    def test_aggregate_factors_all_dimensions(self):
        """All 8 dimensions should be present after aggregation."""
        agg = _make_aggregator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.record_factor_aggregation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.observe_aggregation_duration"
        ):
            factors = agg.aggregate_factors(
                operator_id="OP-001",
                commodity="cocoa",
                country_codes=["BR", "GH"],
                supplier_ids=["SUP-001", "SUP-002"],
            )

        assert len(factors) > 0
        dims_present = {f.dimension for f in factors}
        # All 8 dimensions should be represented
        assert RiskDimension.COUNTRY in dims_present
        assert RiskDimension.SUPPLIER in dims_present
        assert RiskDimension.COMMODITY in dims_present
        assert RiskDimension.CORRUPTION in dims_present
        assert RiskDimension.DEFORESTATION in dims_present
        assert RiskDimension.SUPPLY_CHAIN_COMPLEXITY in dims_present
        assert RiskDimension.MIXING_RISK in dims_present
        assert RiskDimension.CIRCUMVENTION_RISK in dims_present

    def test_empty_inputs(self):
        """No countries and no suppliers -> only commodity + derived dimensions."""
        agg = _make_aggregator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.record_factor_aggregation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.observe_aggregation_duration"
        ):
            factors = agg.aggregate_factors(
                operator_id="OP-002",
                commodity="coffee",
                country_codes=[],
                supplier_ids=[],
            )

        # Should still get commodity + 3 derived dimensions
        dims = {f.dimension for f in factors}
        assert RiskDimension.COMMODITY in dims
        assert RiskDimension.SUPPLY_CHAIN_COMPLEXITY in dims
        assert RiskDimension.MIXING_RISK in dims
        assert RiskDimension.CIRCUMVENTION_RISK in dims


class TestFetchMethods:
    """Test individual upstream agent fetch stubs."""

    def test_fetch_country_risk(self):
        agg = _make_aggregator()
        factors = agg._fetch_country_risk(["BR", "DE"])
        assert len(factors) == 2
        for f in factors:
            assert f.dimension == RiskDimension.COUNTRY
            assert Decimal("0") <= f.raw_score <= Decimal("100")
            assert f.source_agent == SourceAgent.EUDR_016_COUNTRY

    def test_fetch_supplier_risk(self):
        agg = _make_aggregator()
        factors = agg._fetch_supplier_risk(["SUP-001", "SUP-002"])
        assert len(factors) == 2
        for f in factors:
            assert f.dimension == RiskDimension.SUPPLIER
            assert f.source_agent == SourceAgent.EUDR_017_SUPPLIER

    def test_fetch_commodity_risk(self):
        agg = _make_aggregator()
        factors = agg._fetch_commodity_risk("cocoa")
        assert len(factors) == 1
        assert factors[0].dimension == RiskDimension.COMMODITY
        assert factors[0].source_agent == SourceAgent.EUDR_018_COMMODITY

    def test_fetch_corruption_risk(self):
        agg = _make_aggregator()
        factors = agg._fetch_corruption_risk(["BR"])
        assert len(factors) == 1
        assert factors[0].dimension == RiskDimension.CORRUPTION
        assert factors[0].source_agent == SourceAgent.EUDR_019_CORRUPTION

    def test_fetch_deforestation_risk(self):
        agg = _make_aggregator()
        factors = agg._fetch_deforestation_risk(["BR", "ID"])
        assert len(factors) == 2
        for f in factors:
            assert f.dimension == RiskDimension.DEFORESTATION
            assert f.source_agent == SourceAgent.EUDR_020_DEFORESTATION


class TestNormalization:
    """Test score normalization."""

    def test_normalize_score_scales_correctly(self):
        agg = _make_aggregator()
        # Identity: score already in 0-100 range
        result = agg._normalize_score(Decimal("50"), (Decimal("0"), Decimal("100")))
        assert result == Decimal("50.00")

        # Scale 0-50 -> 0-100: 25 should become 50
        result = agg._normalize_score(Decimal("25"), (Decimal("0"), Decimal("50")))
        assert result == Decimal("50.00")

        # Edge: min score -> 0
        result = agg._normalize_score(Decimal("0"), (Decimal("0"), Decimal("100")))
        assert result == Decimal("0.00")

        # Edge: max score -> 100
        result = agg._normalize_score(Decimal("100"), (Decimal("0"), Decimal("100")))
        assert result == Decimal("100.00")


class TestAggregationStats:
    """Test aggregation statistics."""

    def test_aggregation_stats(self):
        agg = _make_aggregator()
        with patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.record_factor_aggregation"
        ), patch(
            "greenlang.agents.eudr.risk_assessment_engine.risk_factor_aggregator.observe_aggregation_duration"
        ):
            agg.aggregate_factors(
                operator_id="OP-001",
                commodity="cocoa",
                country_codes=["BR"],
                supplier_ids=["SUP-001"],
            )

        stats = agg.get_aggregation_stats()
        assert stats["total_aggregations"] >= 1
        assert stats["total_factors_collected"] > 0
        assert "average_factors_per_aggregation" in stats
