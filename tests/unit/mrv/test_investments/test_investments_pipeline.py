# -*- coding: utf-8 -*-
"""
Test suite for investments.investments_pipeline - AGENT-MRV-028.

Tests the InvestmentsPipelineEngine (Engine 7) for the Investments Agent
(GL-MRV-S3-015) covering the full 10-stage pipeline, individual stage
execution, portfolio pipeline with mixed asset classes, batch pipeline,
and error recovery.

Coverage:
- Full 10-stage pipeline execution
- Individual stage tests (validate, classify, normalize, resolve_efs,
  calculate_equity, calculate_debt, calculate_real_assets,
  calculate_sovereign, compliance, seal)
- Portfolio pipeline with mixed asset classes
- Batch pipeline processing
- Error recovery and partial failure handling
- Parametrized tests for pipeline stages

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from greenlang.agents.mrv.investments.investments_pipeline import (
    InvestmentsPipelineEngine,
)
from greenlang.agents.mrv.investments.models import (
    AssetClass,
    ProvenanceStage,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    InvestmentsPipelineEngine.reset_instance()
    yield
    InvestmentsPipelineEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh InvestmentsPipelineEngine with mocked dependencies."""
    with patch(
        "greenlang.agents.mrv.investments.investments_pipeline.get_config"
    ) as mock_config:
        cfg = MagicMock()
        cfg.general.enabled = True
        cfg.general.max_batch_size = 1000
        cfg.general.default_gwp = "AR5"
        cfg.compliance.get_frameworks.return_value = [
            "GHG_PROTOCOL_SCOPE3", "PCAF",
        ]
        mock_config.return_value = cfg
        eng = InvestmentsPipelineEngine()
        yield eng


def _make_equity_input():
    """Build a standard equity pipeline input."""
    return {
        "asset_class": "listed_equity",
        "investee_name": "Apple Inc.",
        "isin": "US0378331005",
        "outstanding_amount": Decimal("100000000"),
        "evic": Decimal("3000000000000"),
        "investee_scope1": Decimal("22400"),
        "investee_scope2": Decimal("9100"),
        "sector": "information_technology",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 1,
    }


def _make_bond_input():
    """Build a standard corporate bond pipeline input."""
    return {
        "asset_class": "corporate_bond",
        "investee_name": "Tesla Inc.",
        "outstanding_amount": Decimal("75000000"),
        "evic": Decimal("500000000000"),
        "investee_scope1": Decimal("30000"),
        "investee_scope2": Decimal("12000"),
        "sector": "consumer_discretionary",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 1,
    }


def _make_cre_input():
    """Build a standard CRE pipeline input."""
    return {
        "asset_class": "commercial_real_estate",
        "property_name": "Office Tower",
        "outstanding_amount": Decimal("25000000"),
        "property_value": Decimal("50000000"),
        "floor_area_m2": Decimal("10000"),
        "property_type": "office",
        "epc_rating": "B",
        "climate_zone": "temperate",
        "country": "US",
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 2,
    }


def _make_sovereign_input():
    """Build a standard sovereign bond pipeline input."""
    return {
        "asset_class": "sovereign_bond",
        "country": "US",
        "outstanding_amount": Decimal("500000000"),
        "gdp_ppp": Decimal("25460000000000"),
        "country_emissions": Decimal("5222000000"),
        "include_lulucf": False,
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 4,
    }


def _make_portfolio_input():
    """Build a mixed portfolio input."""
    return {
        "portfolio_name": "Test Portfolio",
        "reporting_year": 2024,
        "currency": "USD",
        "investments": [
            _make_equity_input(),
            _make_bond_input(),
            _make_cre_input(),
            _make_sovereign_input(),
        ],
    }


# ==============================================================================
# FULL PIPELINE TESTS
# ==============================================================================


class TestFullPipeline:
    """Test full 10-stage pipeline execution."""

    def test_pipeline_single_equity(self, engine):
        """Test full pipeline with single equity investment."""
        result = engine.process(_make_equity_input())
        assert result is not None
        assert result["financed_emissions"] > Decimal("0")
        assert "provenance_hash" in result

    def test_pipeline_single_bond(self, engine):
        """Test full pipeline with single corporate bond."""
        result = engine.process(_make_bond_input())
        assert result is not None
        assert result["financed_emissions"] > Decimal("0")

    def test_pipeline_single_cre(self, engine):
        """Test full pipeline with single CRE investment."""
        result = engine.process(_make_cre_input())
        assert result is not None
        assert result["financed_emissions"] > Decimal("0")

    def test_pipeline_single_sovereign(self, engine):
        """Test full pipeline with single sovereign bond."""
        result = engine.process(_make_sovereign_input())
        assert result is not None
        assert result["financed_emissions"] > Decimal("0")

    def test_pipeline_includes_provenance(self, engine):
        """Test pipeline result includes provenance chain."""
        result = engine.process(_make_equity_input())
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_pipeline_includes_pcaf_score(self, engine):
        """Test pipeline result includes PCAF quality score."""
        result = engine.process(_make_equity_input())
        assert "pcaf_quality_score" in result

    def test_pipeline_includes_compliance(self, engine):
        """Test pipeline result includes compliance check results."""
        result = engine.process(_make_equity_input())
        assert "compliance_results" in result or "compliance" in result

    def test_pipeline_deterministic(self, engine):
        """Test pipeline produces deterministic results."""
        data = _make_equity_input()
        r1 = engine.process(data)
        r2 = engine.process(data)
        assert r1["financed_emissions"] == r2["financed_emissions"]
        assert r1["provenance_hash"] == r2["provenance_hash"]


# ==============================================================================
# INDIVIDUAL STAGE TESTS
# ==============================================================================


class TestIndividualStages:
    """Test individual pipeline stages."""

    def test_validate_stage(self, engine):
        """Test validate stage accepts valid input."""
        data = _make_equity_input()
        result = engine._run_stage("validate", data)
        assert result is not None

    def test_validate_stage_rejects_invalid(self, engine):
        """Test validate stage rejects invalid input."""
        with pytest.raises((ValueError, KeyError)):
            engine._run_stage("validate", {})

    def test_classify_stage(self, engine):
        """Test classify stage identifies asset class."""
        data = _make_equity_input()
        validated = engine._run_stage("validate", data)
        result = engine._run_stage("classify", validated)
        assert result.get("asset_class") == "listed_equity"

    def test_normalize_stage(self, engine):
        """Test normalize stage converts currencies."""
        data = _make_equity_input()
        validated = engine._run_stage("validate", data)
        result = engine._run_stage("normalize", validated)
        assert result is not None

    def test_resolve_efs_stage(self, engine):
        """Test resolve_efs stage retrieves emission factors."""
        data = _make_equity_input()
        result = engine._run_stage("resolve_efs", data)
        assert result is not None

    def test_calculate_equity_stage(self, engine):
        """Test calculate_equity stage computes financed emissions."""
        data = _make_equity_input()
        result = engine._run_stage("calculate_equity", data)
        assert result.get("financed_emissions") is not None

    def test_calculate_debt_stage(self, engine):
        """Test calculate_debt stage computes financed emissions."""
        data = _make_bond_input()
        result = engine._run_stage("calculate_debt", data)
        assert result.get("financed_emissions") is not None

    def test_calculate_real_assets_stage(self, engine):
        """Test calculate_real_assets stage."""
        data = _make_cre_input()
        result = engine._run_stage("calculate_real_assets", data)
        assert result.get("financed_emissions") is not None

    def test_calculate_sovereign_stage(self, engine):
        """Test calculate_sovereign stage."""
        data = _make_sovereign_input()
        result = engine._run_stage("calculate_sovereign", data)
        assert result.get("financed_emissions") is not None

    def test_seal_stage(self, engine):
        """Test seal stage produces final provenance hash."""
        data = _make_equity_input()
        data["financed_emissions"] = Decimal("1050")
        result = engine._run_stage("seal", data)
        assert "provenance_hash" in result


# ==============================================================================
# PORTFOLIO PIPELINE TESTS
# ==============================================================================


class TestPortfolioPipeline:
    """Test portfolio pipeline with mixed asset classes."""

    def test_portfolio_pipeline(self, engine):
        """Test portfolio pipeline with multiple asset classes."""
        data = _make_portfolio_input()
        result = engine.process_portfolio(data)
        assert result is not None
        assert result["total_financed_emissions"] > Decimal("0")

    def test_portfolio_asset_class_breakdown(self, engine):
        """Test portfolio result includes asset class breakdown."""
        data = _make_portfolio_input()
        result = engine.process_portfolio(data)
        assert "asset_class_breakdown" in result
        assert len(result["asset_class_breakdown"]) > 0

    def test_portfolio_weighted_pcaf_score(self, engine):
        """Test portfolio result includes weighted PCAF score."""
        data = _make_portfolio_input()
        result = engine.process_portfolio(data)
        assert "weighted_pcaf_score" in result
        assert Decimal("1") <= result["weighted_pcaf_score"] <= Decimal("5")

    def test_portfolio_provenance_hash(self, engine):
        """Test portfolio result includes provenance hash."""
        data = _make_portfolio_input()
        result = engine.process_portfolio(data)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_portfolio_individual_results(self, engine):
        """Test portfolio includes individual investment results."""
        data = _make_portfolio_input()
        result = engine.process_portfolio(data)
        assert "individual_results" in result or "results" in result

    def test_portfolio_deterministic(self, engine):
        """Test portfolio pipeline is deterministic."""
        data = _make_portfolio_input()
        r1 = engine.process_portfolio(data)
        r2 = engine.process_portfolio(data)
        assert r1["total_financed_emissions"] == r2["total_financed_emissions"]


# ==============================================================================
# BATCH PIPELINE TESTS
# ==============================================================================


class TestBatchPipeline:
    """Test batch pipeline processing."""

    def test_batch_pipeline(self, engine):
        """Test batch pipeline with multiple investments."""
        items = [
            _make_equity_input(),
            _make_bond_input(),
            _make_cre_input(),
        ]
        results = engine.process_batch(items)
        assert len(results) == 3

    def test_batch_all_successful(self, engine):
        """Test batch where all items succeed."""
        items = [_make_equity_input() for _ in range(5)]
        results = engine.process_batch(items)
        assert len(results) == 5
        for r in results:
            assert r["financed_emissions"] > Decimal("0")

    def test_batch_empty_list(self, engine):
        """Test batch with empty list."""
        results = engine.process_batch([])
        assert len(results) == 0

    def test_batch_single_item(self, engine):
        """Test batch with single item."""
        results = engine.process_batch([_make_equity_input()])
        assert len(results) == 1


# ==============================================================================
# ERROR RECOVERY TESTS
# ==============================================================================


class TestErrorRecovery:
    """Test error recovery in pipeline."""

    def test_invalid_asset_class_raises(self, engine):
        """Test invalid asset class raises error."""
        data = _make_equity_input()
        data["asset_class"] = "invalid_class"
        with pytest.raises((ValueError, KeyError)):
            engine.process(data)

    def test_missing_required_field_raises(self, engine):
        """Test missing required field raises error."""
        data = _make_equity_input()
        del data["outstanding_amount"]
        with pytest.raises((ValueError, KeyError)):
            engine.process(data)

    def test_zero_denominator_raises(self, engine):
        """Test zero denominator raises error."""
        data = _make_equity_input(evic=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.process(data)

    def test_batch_partial_failure(self, engine):
        """Test batch handles partial failures gracefully."""
        items = [
            _make_equity_input(),
            {"asset_class": "invalid"},  # Will fail
            _make_bond_input(),
        ]
        try:
            results = engine.process_batch(items)
            # Either skips invalid or returns error marker
            assert len(results) >= 2
        except (ValueError, KeyError):
            pass  # Also acceptable to raise on invalid input


# ==============================================================================
# PARAMETRIZED STAGE TESTS
# ==============================================================================


class TestParametrizedStages:
    """Parametrized tests across pipeline stages."""

    @pytest.mark.parametrize("stage", [
        "validate", "classify", "normalize", "resolve_efs",
        "calculate_equity", "calculate_debt", "calculate_real_assets",
        "calculate_sovereign", "compliance", "seal",
    ])
    def test_stage_names_valid(self, engine, stage):
        """Test all 10 pipeline stage names are recognized."""
        assert stage in [s.value for s in ProvenanceStage]

    @pytest.mark.parametrize("asset_class,factory", [
        ("listed_equity", _make_equity_input),
        ("corporate_bond", _make_bond_input),
        ("commercial_real_estate", _make_cre_input),
        ("sovereign_bond", _make_sovereign_input),
    ])
    def test_pipeline_all_asset_classes(self, engine, asset_class, factory):
        """Test pipeline processes all asset classes."""
        data = factory()
        result = engine.process(data)
        assert result["financed_emissions"] > Decimal("0")
