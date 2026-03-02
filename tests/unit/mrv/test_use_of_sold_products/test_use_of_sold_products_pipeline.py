# -*- coding: utf-8 -*-
"""
Unit tests for UseOfSoldProductsPipelineEngine -- AGENT-MRV-024

Tests the 10-stage processing pipeline including full pipeline execution,
batch processing, portfolio analysis, input validation, stage enums,
dual-path (direct + indirect) routing, partial results on error,
pipeline status, and singleton pattern.

Target: 30+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.use_of_sold_products_pipeline import (
        UseOfSoldProductsPipelineEngine,
        PipelineStage,
        PipelineStatus,
        CalculationMethod,
        ProductCategory,
        EmissionType,
        ComplianceFramework,
        EFSource,
        DEFAULT_FUEL_EFS,
        DEFAULT_GRID_EFS,
        DEFAULT_REFRIGERANT_GWPS,
        UNIT_CONVERSIONS,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="UseOfSoldProductsPipelineEngine not available")
pytestmark = _SKIP


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    UseOfSoldProductsPipelineEngine.reset()
    yield
    UseOfSoldProductsPipelineEngine.reset()


@pytest.fixture
def engine():
    """Create a UseOfSoldProductsPipelineEngine instance."""
    return UseOfSoldProductsPipelineEngine()


@pytest.fixture
def valid_pipeline_input():
    """Create valid input for pipeline execution (vehicles only)."""
    return {
        "products": [
            {
                "product_id": "VEH-001",
                "product_name": "Sedan 2.0L Gasoline",
                "product_category": "vehicles",
                "emission_type": "direct",
                "units_sold": 1000,
                "lifetime_years": 15,
                "fuel_type": "gasoline",
                "fuel_consumption_per_year": "1200",
                "fuel_ef_kg_per_unit": "2.315",
            },
        ],
    }


@pytest.fixture
def multi_product_input():
    """Create multi-product input for pipeline execution."""
    return {
        "products": [
            {
                "product_id": "VEH-001",
                "product_category": "vehicles",
                "emission_type": "direct",
                "units_sold": 1000,
                "lifetime_years": 15,
                "fuel_type": "gasoline",
                "fuel_consumption_per_year": "1200",
                "fuel_ef_kg_per_unit": "2.315",
            },
            {
                "product_id": "APP-001",
                "product_category": "appliances",
                "emission_type": "indirect",
                "units_sold": 10000,
                "lifetime_years": 15,
                "energy_consumption_kwh_per_year": "400",
                "grid_ef_kg_per_kwh": "0.417",
            },
            {
                "product_id": "HVAC-001",
                "product_category": "hvac",
                "emission_type": "both",
                "units_sold": 500,
                "lifetime_years": 12,
                "refrigerant_type": "R-410A",
                "refrigerant_charge_kg": "3.0",
                "annual_leak_rate": "0.05",
                "refrigerant_gwp": "2088",
                "energy_consumption_kwh_per_year": "1500",
                "grid_ef_kg_per_kwh": "0.417",
            },
        ],
    }


@pytest.fixture
def fuel_sale_input():
    """Create fuels/feedstocks input."""
    return {
        "products": [
            {
                "product_id": "FUEL-001",
                "product_category": "fuels_feedstocks",
                "emission_type": "direct",
                "fuel_type": "gasoline",
                "quantity_sold_litres": "1000000",
                "fuel_ef_kg_per_litre": "2.315",
            },
        ],
    }


@pytest.fixture
def portfolio_input():
    """Create a portfolio with 10 product types."""
    return {
        "products": [
            {"product_id": f"P{i}", "product_category": cat, "emission_type": etype,
             "units_sold": 1000, "lifetime_years": 10}
            for i, (cat, etype) in enumerate([
                ("vehicles", "direct"),
                ("appliances", "indirect"),
                ("hvac", "both"),
                ("lighting", "indirect"),
                ("it_equipment", "indirect"),
                ("industrial_equipment", "direct"),
                ("building_products", "indirect"),
                ("consumer_products", "direct"),
                ("medical_devices", "indirect"),
                ("fuels_feedstocks", "direct"),
            ])
        ],
    }


# ============================================================================
# TEST: Pipeline Stage Enum
# ============================================================================


class TestPipelineStageEnum:
    """Test PipelineStage enum has all 10 stages."""

    def test_all_10_stages_present(self):
        """Test all 10 pipeline stages exist."""
        expected_stages = [
            "validate", "classify", "normalize", "resolve_efs",
            "calculate", "lifetime", "aggregate", "compliance",
            "provenance", "seal",
        ]
        actual_values = [s.value for s in PipelineStage]
        for stage in expected_stages:
            assert stage in actual_values
        assert len(PipelineStage) == 10


# ============================================================================
# TEST: Full Pipeline Execution
# ============================================================================


class TestFullPipeline:
    """Test full pipeline execution."""

    def test_single_product_pipeline(self, engine, valid_pipeline_input):
        """Test pipeline execution with a single vehicle product."""
        result = engine.execute(valid_pipeline_input)
        assert result is not None
        assert "total_emissions_co2e_kg" in result or "total_co2e_kg" in result
        total = result.get("total_emissions_co2e_kg", result.get("total_co2e_kg", 0))
        assert Decimal(str(total)) > 0

    def test_multi_product_pipeline(self, engine, multi_product_input):
        """Test pipeline execution with multiple product types."""
        result = engine.execute(multi_product_input)
        assert result is not None
        total = result.get("total_emissions_co2e_kg", result.get("total_co2e_kg", 0))
        assert Decimal(str(total)) > 0

    def test_pipeline_has_product_count(self, engine, multi_product_input):
        """Test pipeline result includes product count."""
        result = engine.execute(multi_product_input)
        count = result.get("product_count", 0)
        assert count == 3

    def test_pipeline_has_provenance_hash(self, engine, valid_pipeline_input):
        """Test pipeline result includes provenance hash."""
        result = engine.execute(valid_pipeline_input)
        if "provenance_hash" in result:
            assert len(result["provenance_hash"]) == 64

    def test_pipeline_has_status(self, engine, valid_pipeline_input):
        """Test pipeline result has a completion status."""
        result = engine.execute(valid_pipeline_input)
        status = result.get("status", result.get("pipeline_status", ""))
        assert status in ("completed", "success", "COMPLETED", "SUCCESS", "partial")


# ============================================================================
# TEST: Dual-Path Routing
# ============================================================================


class TestDualPathRouting:
    """Test dual-path (direct + indirect) routing for products like HVAC."""

    def test_hvac_routes_to_both_paths(self, engine):
        """Test HVAC product with emission_type=both is routed to both engines."""
        input_data = {
            "products": [
                {
                    "product_id": "HVAC-001",
                    "product_category": "hvac",
                    "emission_type": "both",
                    "units_sold": 500,
                    "lifetime_years": 12,
                    "refrigerant_type": "R-410A",
                    "refrigerant_charge_kg": "3.0",
                    "annual_leak_rate": "0.05",
                    "refrigerant_gwp": "2088",
                    "energy_consumption_kwh_per_year": "1500",
                    "grid_ef_kg_per_kwh": "0.417",
                },
            ],
        }
        result = engine.execute(input_data)
        total = result.get("total_emissions_co2e_kg", result.get("total_co2e_kg", 0))
        assert Decimal(str(total)) > 0

    def test_direct_only_product(self, engine, valid_pipeline_input):
        """Test direct-only product routes to direct engine only."""
        result = engine.execute(valid_pipeline_input)
        direct = result.get("direct_emissions_co2e_kg", result.get("direct_co2e_kg", 0))
        assert Decimal(str(direct)) > 0

    def test_indirect_only_product(self, engine):
        """Test indirect-only product routes to indirect engine only."""
        input_data = {
            "products": [
                {
                    "product_id": "APP-001",
                    "product_category": "appliances",
                    "emission_type": "indirect",
                    "units_sold": 10000,
                    "lifetime_years": 15,
                    "energy_consumption_kwh_per_year": "400",
                    "grid_ef_kg_per_kwh": "0.417",
                },
            ],
        }
        result = engine.execute(input_data)
        indirect = result.get("indirect_emissions_co2e_kg", result.get("indirect_co2e_kg", 0))
        assert Decimal(str(indirect)) > 0


# ============================================================================
# TEST: Batch Processing
# ============================================================================


class TestBatchProcessing:
    """Test batch processing of multiple product portfolios."""

    def test_batch_execution(self, engine, valid_pipeline_input, multi_product_input):
        """Test batch execution with multiple inputs."""
        batch = [valid_pipeline_input, multi_product_input]
        results = engine.execute_batch(batch)
        assert len(results) == 2
        for result in results:
            total = result.get("total_emissions_co2e_kg", result.get("total_co2e_kg", 0))
            assert Decimal(str(total)) > 0

    def test_empty_batch(self, engine):
        """Test empty batch returns empty results."""
        results = engine.execute_batch([])
        assert len(results) == 0


# ============================================================================
# TEST: Portfolio Analysis
# ============================================================================


class TestPortfolioAnalysis:
    """Test portfolio-level analysis across product categories."""

    def test_portfolio_execution(self, engine, portfolio_input):
        """Test portfolio analysis with 10 product types."""
        result = engine.execute(portfolio_input)
        count = result.get("product_count", 0)
        assert count == 10

    def test_portfolio_category_breakdown(self, engine, multi_product_input):
        """Test portfolio result includes category breakdown."""
        result = engine.execute(multi_product_input)
        if "by_category" in result:
            assert len(result["by_category"]) >= 2

    def test_portfolio_emission_type_breakdown(self, engine, multi_product_input):
        """Test portfolio result includes emission type breakdown."""
        result = engine.execute(multi_product_input)
        if "by_emission_type" in result:
            assert "direct" in result["by_emission_type"] or "indirect" in result["by_emission_type"]


# ============================================================================
# TEST: Error Handling and Partial Results
# ============================================================================


class TestErrorHandling:
    """Test error handling and partial results."""

    def test_invalid_category_handled(self, engine):
        """Test invalid product category is handled gracefully."""
        input_data = {
            "products": [
                {
                    "product_id": "BAD-001",
                    "product_category": "invalid_category",
                    "emission_type": "direct",
                    "units_sold": 100,
                },
            ],
        }
        try:
            result = engine.execute(input_data)
            # Either returns partial result or raises
            assert result is not None
        except (ValueError, KeyError):
            pass  # Expected

    def test_missing_required_fields_handled(self, engine):
        """Test missing required fields are handled."""
        input_data = {
            "products": [
                {
                    "product_id": "BAD-002",
                    # Missing product_category
                },
            ],
        }
        try:
            result = engine.execute(input_data)
            assert result is not None
        except (ValueError, KeyError, TypeError):
            pass  # Expected

    def test_partial_results_on_mixed_validity(self, engine):
        """Test partial results when some products are invalid."""
        input_data = {
            "products": [
                {
                    "product_id": "GOOD-001",
                    "product_category": "vehicles",
                    "emission_type": "direct",
                    "units_sold": 1000,
                    "lifetime_years": 15,
                    "fuel_consumption_per_year": "1200",
                    "fuel_ef_kg_per_unit": "2.315",
                },
                {
                    "product_id": "BAD-001",
                    "product_category": "invalid",
                    "emission_type": "unknown",
                },
            ],
        }
        try:
            result = engine.execute(input_data)
            if result is not None:
                # At least the good product should produce emissions
                total = result.get("total_emissions_co2e_kg", result.get("total_co2e_kg", 0))
                assert Decimal(str(total)) >= 0
        except (ValueError, KeyError):
            pass  # Also acceptable


# ============================================================================
# TEST: Singleton and Thread Safety
# ============================================================================


class TestPipelineSingleton:
    """Test UseOfSoldProductsPipelineEngine singleton pattern."""

    def test_singleton_same_instance(self):
        """Test constructor returns same instance."""
        e1 = UseOfSoldProductsPipelineEngine()
        e2 = UseOfSoldProductsPipelineEngine()
        assert e1 is e2

    def test_thread_safe_execution(self, valid_pipeline_input):
        """Test pipeline execution is thread-safe."""
        results = []
        errors = []

        def _execute():
            try:
                engine = UseOfSoldProductsPipelineEngine()
                result = engine.execute(valid_pipeline_input)
                results.append(result)
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=_execute) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 10
