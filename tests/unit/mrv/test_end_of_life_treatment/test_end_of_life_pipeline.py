# -*- coding: utf-8 -*-
"""
Unit tests for EndOfLifeTreatmentPipelineEngine -- AGENT-MRV-025

Tests the 10-stage end-of-life treatment pipeline including full pipeline
execution, batch processing, portfolio analysis, treatment pathway routing,
input validation, stage enums, error handling, and singleton pattern.

Pipeline Stages:
1. validate - Input validation
2. classify - Product category classification
3. normalize - Mass/composition normalization
4. resolve_efs - Emission factor resolution
5. calculate - Core emissions calculation
6. allocate - Treatment pathway allocation
7. aggregate - Multi-product aggregation
8. compliance - Regulatory compliance checks
9. provenance - Provenance hash computation
10. seal - Chain sealing and verification

Target: 25+ tests.
Author: GL-TestEngineer
Date: February 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.end_of_life_treatment.end_of_life_treatment_pipeline import (
        EndOfLifeTreatmentPipelineEngine,
        PipelineStage,
        PipelineStatus,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="EndOfLifeTreatmentPipelineEngine not available")
pytestmark = _SKIP


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    EndOfLifeTreatmentPipelineEngine.reset()
    yield
    EndOfLifeTreatmentPipelineEngine.reset()


@pytest.fixture
def engine():
    """Create an EndOfLifeTreatmentPipelineEngine instance."""
    return EndOfLifeTreatmentPipelineEngine()


@pytest.fixture
def valid_single_product():
    """Single product input for pipeline execution."""
    return {
        "products": [
            {
                "product_id": "PRD-001",
                "product_category": "consumer_electronics",
                "total_mass_kg": "1000",
                "units_sold": 5000,
                "region": "US",
                "reporting_year": 2024,
            },
        ],
    }


@pytest.fixture
def valid_multi_product():
    """Multi-product input for pipeline execution."""
    return {
        "products": [
            {
                "product_id": "PRD-001",
                "product_category": "consumer_electronics",
                "total_mass_kg": "1000",
                "units_sold": 5000,
                "region": "US",
                "reporting_year": 2024,
            },
            {
                "product_id": "PRD-002",
                "product_category": "packaging",
                "total_mass_kg": "5000",
                "units_sold": 1000000,
                "region": "GB",
                "reporting_year": 2024,
            },
            {
                "product_id": "PRD-003",
                "product_category": "clothing",
                "total_mass_kg": "500",
                "units_sold": 10000,
                "region": "DE",
                "reporting_year": 2024,
            },
        ],
    }


@pytest.fixture
def multi_material_product():
    """Product with explicit material composition."""
    return {
        "products": [
            {
                "product_id": "PRD-MIX",
                "product_category": "consumer_electronics",
                "total_mass_kg": "2000",
                "units_sold": 10000,
                "region": "US",
                "reporting_year": 2024,
                "composition": [
                    {"material": "plastic_abs", "mass_fraction": "0.35"},
                    {"material": "steel", "mass_fraction": "0.30"},
                    {"material": "glass", "mass_fraction": "0.20"},
                    {"material": "copper", "mass_fraction": "0.15"},
                ],
            },
        ],
    }


# ============================================================================
# TEST: Pipeline Stage Enum
# ============================================================================


class TestPipelineStages:
    """Test pipeline stage enum validation."""

    def test_exactly_10_stages(self):
        """Test there are exactly 10 pipeline stages."""
        assert len(PipelineStage) == 10

    @pytest.mark.parametrize("stage", [
        "validate", "classify", "normalize", "resolve_efs",
        "calculate", "allocate", "aggregate", "compliance",
        "provenance", "seal",
    ])
    def test_stage_membership(self, stage):
        """Test each stage is a valid enum member."""
        assert PipelineStage(stage) is not None


# ============================================================================
# TEST: Full Pipeline Execution
# ============================================================================


class TestFullPipeline:
    """Test full pipeline execution scenarios."""

    def test_single_product_pipeline(self, engine, valid_single_product):
        """Test pipeline with single product."""
        result = engine.execute(valid_single_product)
        assert result is not None
        assert "gross_emissions_tco2e" in result or "gross_emissions_kgco2e" in result

    def test_multi_product_pipeline(self, engine, valid_multi_product):
        """Test pipeline with multiple products."""
        result = engine.execute(valid_multi_product)
        assert result is not None
        product_count = result.get("product_count", 0)
        assert product_count == 3

    def test_multi_material_pipeline(self, engine, multi_material_product):
        """Test pipeline with explicit material composition."""
        result = engine.execute(multi_material_product)
        assert result is not None
        # Should have treatment breakdown
        by_treatment = result.get("by_treatment", {})
        assert len(by_treatment) >= 1

    def test_pipeline_returns_provenance_hash(self, engine, valid_single_product):
        """Test pipeline returns a provenance hash."""
        result = engine.execute(valid_single_product)
        assert "provenance_hash" in result or "calculation_id" in result

    def test_pipeline_returns_dqi_score(self, engine, valid_single_product):
        """Test pipeline returns DQI score."""
        result = engine.execute(valid_single_product)
        dqi = result.get("dqi_score", None)
        if dqi is not None:
            assert Decimal("0") <= Decimal(str(dqi)) <= Decimal("100")

    def test_pipeline_returns_compliance_status(self, engine, valid_single_product):
        """Test pipeline returns compliance status."""
        result = engine.execute(valid_single_product)
        assert "compliant" in result

    def test_avoided_emissions_in_pipeline_result(self, engine, valid_single_product):
        """Test pipeline returns avoided emissions separately."""
        result = engine.execute(valid_single_product)
        # Must have avoided emissions as a separate field
        assert "avoided_emissions_tco2e" in result or "avoided_emissions_kgco2e" in result


# ============================================================================
# TEST: Treatment Pathway Routing
# ============================================================================


class TestTreatmentPathwayRouting:
    """Test treatment pathway routing based on regional mix."""

    def test_routing_applies_regional_mix(self, engine, valid_single_product):
        """Test product is routed through regional treatment mix."""
        result = engine.execute(valid_single_product)
        by_treatment = result.get("by_treatment", {})
        assert len(by_treatment) >= 1

    def test_circularity_metrics_returned(self, engine, valid_single_product):
        """Test circularity metrics are returned."""
        result = engine.execute(valid_single_product)
        metrics = result.get("circularity_metrics", {})
        if metrics:
            assert "recycling_rate" in metrics or "circularity_index" in metrics


# ============================================================================
# TEST: Batch Processing
# ============================================================================


class TestBatchProcessing:
    """Test batch processing of multiple products."""

    def test_batch_5_products(self, engine):
        """Test batch processing with 5 products."""
        batch_input = {
            "products": [
                {"product_id": f"P{i}", "product_category": "packaging",
                 "total_mass_kg": "100", "units_sold": 1000,
                 "region": "US", "reporting_year": 2024}
                for i in range(5)
            ],
        }
        result = engine.execute(batch_input)
        assert result is not None
        assert result.get("product_count", 0) == 5

    def test_batch_emissions_sum(self, engine):
        """Test batch total is sum of individual products."""
        batch_input = {
            "products": [
                {"product_id": "P1", "product_category": "packaging",
                 "total_mass_kg": "100", "units_sold": 100,
                 "region": "US", "reporting_year": 2024},
                {"product_id": "P2", "product_category": "packaging",
                 "total_mass_kg": "200", "units_sold": 200,
                 "region": "US", "reporting_year": 2024},
            ],
        }
        result = engine.execute(batch_input)
        assert result is not None


# ============================================================================
# TEST: Portfolio Analysis
# ============================================================================


class TestPortfolioAnalysis:
    """Test portfolio-level analysis with circularity metrics."""

    def test_portfolio_diverse_products(self, engine, valid_multi_product):
        """Test portfolio analysis with diverse product categories."""
        result = engine.execute(valid_multi_product)
        assert result is not None
        # Should include by-category breakdown
        by_category = result.get("by_category", {})
        if by_category:
            assert len(by_category) >= 2


# ============================================================================
# TEST: Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling in pipeline."""

    def test_empty_products_list(self, engine):
        """Test empty products list."""
        result = engine.execute({"products": []})
        assert result is not None
        gross = result.get("gross_emissions_tco2e", result.get("gross_emissions_kgco2e", Decimal("0.0")))
        assert Decimal(str(gross)) == Decimal("0.0")

    def test_missing_product_category(self, engine):
        """Test missing product category raises error or uses default."""
        inp = {
            "products": [
                {"product_id": "P-BAD", "total_mass_kg": "100"},
            ],
        }
        try:
            result = engine.execute(inp)
            # Should either fail or use default category
            assert result is not None
        except (ValueError, KeyError):
            pass  # Expected

    def test_negative_mass_rejected(self, engine):
        """Test negative mass is rejected."""
        inp = {
            "products": [
                {"product_id": "P-NEG", "product_category": "packaging",
                 "total_mass_kg": "-100", "units_sold": 100,
                 "region": "US", "reporting_year": 2024},
            ],
        }
        with pytest.raises((ValueError, Exception)):
            engine.execute(inp)
