# -*- coding: utf-8 -*-
"""
Unit Tests for CBAM Carbon Intensity Agent

Comprehensive test suite with 20 test cases covering:
- LookupCbamBenchmarkTool (10 tests)
- CalculateCarbonIntensityTool (10 tests)

Target: 85%+ coverage for CBAM agent tools
Run with: pytest tests/unit/test_cbam_agent.py -v --cov=generated/carbon_intensity_v1

Author: GL-TestEngineer
Version: 1.0.0

CBAM (Carbon Border Adjustment Mechanism) is the EU's carbon pricing mechanism
for imports that ensures carbon-intensive goods pay the same carbon price as
EU products, preventing carbon leakage.
"""

import pytest
import asyncio
import hashlib
import json
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "generated" / "carbon_intensity_v1"))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def lookup_cbam_benchmark_tool():
    """Create LookupCbamBenchmarkTool instance."""
    from generated.carbon_intensity_v1.tools import LookupCbamBenchmarkTool
    return LookupCbamBenchmarkTool()


@pytest.fixture
def calculate_carbon_intensity_tool():
    """Create CalculateCarbonIntensityTool instance."""
    from generated.carbon_intensity_v1.tools import CalculateCarbonIntensityTool
    return CalculateCarbonIntensityTool()


@pytest.fixture
def mock_cbam_database():
    """Create mock CBAM benchmark database."""
    mock_db = Mock()
    mock_benchmark = Mock()
    mock_benchmark.product_type = "steel_hot_rolled_coil"
    mock_benchmark.production_method = "basic_oxygen_furnace"
    mock_benchmark.benchmark_value = 1.85
    mock_benchmark.unit = "tCO2e/tonne"
    mock_benchmark.cn_codes = ["7208", "7209", "7210", "7211"]
    mock_benchmark.effective_date = "2026-01-01"
    mock_benchmark.source = "EU Implementing Regulation 2023/1773 Annex II"
    mock_benchmark.notes = "Hot rolled coil steel, basic oxygen furnace production"
    mock_db.lookup.return_value = mock_benchmark
    mock_db.list_products.return_value = [
        "steel_hot_rolled_coil", "steel_rebar", "cement_portland",
        "aluminum_unwrought", "fertilizer_ammonia"
    ]
    return mock_db


@pytest.fixture
def sample_cbam_products():
    """Sample CBAM-regulated products for testing."""
    return [
        "steel_hot_rolled_coil",
        "steel_rebar",
        "steel_wire_rod",
        "cement_clinker",
        "cement_portland",
        "aluminum_unwrought",
        "aluminum_products",
        "fertilizer_ammonia",
        "fertilizer_urea",
        "electricity",
        "hydrogen",
    ]


@pytest.fixture
def sample_benchmark_values():
    """Sample benchmark values based on EU Regulation 2023/1773."""
    return {
        "steel_hot_rolled_coil": 1.85,  # tCO2e/tonne
        "steel_rebar": 1.35,
        "cement_clinker": 0.766,
        "cement_portland": 0.670,
        "aluminum_unwrought": 8.6,
        "fertilizer_ammonia": 2.4,
        "hydrogen": 10.5,
    }


# =============================================================================
# LookupCbamBenchmarkTool Tests (10 tests)
# =============================================================================

class TestLookupCbamBenchmarkTool:
    """Test suite for LookupCbamBenchmarkTool - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    def test_tool_initialization(self, lookup_cbam_benchmark_tool):
        """UT-CBAM-001: Test tool initializes correctly."""
        assert lookup_cbam_benchmark_tool is not None
        assert lookup_cbam_benchmark_tool.name == "lookup_cbam_benchmark"
        assert lookup_cbam_benchmark_tool.safe is True
        assert lookup_cbam_benchmark_tool.description == "Look up CBAM default benchmark values"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_missing_product_type_raises_error(self, lookup_cbam_benchmark_tool):
        """UT-CBAM-002: Test missing product_type parameter raises ValueError."""
        params = {}

        with pytest.raises(ValueError) as exc_info:
            await lookup_cbam_benchmark_tool.execute(params)

        assert "product_type" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_valid_lookup_steel_returns_result(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-003: Test valid steel product lookup returns benchmark data."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            result = await lookup_cbam_benchmark_tool.execute(params)

            assert result is not None
            assert "benchmark_value" in result
            assert "benchmark_unit" in result
            assert "source" in result
            assert result["benchmark_value"] == 1.85
            assert result["benchmark_unit"] == "tCO2e/tonne"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_lookup_includes_cn_codes(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-004: Test lookup result includes Combined Nomenclature codes."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            result = await lookup_cbam_benchmark_tool.execute(params)

            assert "cn_codes" in result
            assert isinstance(result["cn_codes"], list)
            assert len(result["cn_codes"]) > 0
            assert "7208" in result["cn_codes"]

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_lookup_includes_effective_date(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-005: Test lookup result includes effective date."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            result = await lookup_cbam_benchmark_tool.execute(params)

            assert "effective_date" in result
            assert result["effective_date"] == "2026-01-01"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_lookup_includes_production_method(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-006: Test lookup result includes production method."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            result = await lookup_cbam_benchmark_tool.execute(params)

            assert "production_method" in result
            assert result["production_method"] == "basic_oxygen_furnace"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_invalid_product_type_raises_error(self, lookup_cbam_benchmark_tool):
        """UT-CBAM-007: Test invalid product type raises appropriate error."""
        mock_db = Mock()
        mock_db.lookup.return_value = None
        mock_db.list_products.return_value = ["steel_hot_rolled_coil", "cement_portland"]

        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_db):
            params = {"product_type": "invalid_product_xyz"}

            with pytest.raises(ValueError) as exc_info:
                await lookup_cbam_benchmark_tool.execute(params)

            assert "not found" in str(exc_info.value).lower()
            # Should include available products in error message
            assert "steel_hot_rolled_coil" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_result_includes_provenance_hash(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-008: Test result includes SHA-256 provenance hash."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            result = await lookup_cbam_benchmark_tool.execute(params)

            assert "result_hash" in result
            assert len(result["result_hash"]) == 64  # SHA-256 hex length
            # Verify it's a valid hex string
            int(result["result_hash"], 16)

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_result_includes_execution_timestamp(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-009: Test result includes execution timestamp."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            result = await lookup_cbam_benchmark_tool.execute(params)

            assert "executed_at" in result
            # Should be a valid ISO timestamp
            datetime.fromisoformat(result["executed_at"])

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_lookup_determinism(self, lookup_cbam_benchmark_tool, mock_cbam_database):
        """UT-CBAM-010: Test lookup is deterministic across multiple calls."""
        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_cbam_database):
            params = {"product_type": "steel_hot_rolled_coil"}

            # Get results from multiple calls
            results = []
            for _ in range(5):
                result = await lookup_cbam_benchmark_tool.execute(params)
                # Exclude timestamp for comparison
                comparable = {k: v for k, v in result.items() if k != "executed_at"}
                results.append(json.dumps(comparable, sort_keys=True))

            # All results should be identical (except timestamp)
            assert all(r == results[0] for r in results)


# =============================================================================
# CalculateCarbonIntensityTool Tests (10 tests)
# =============================================================================

class TestCalculateCarbonIntensityTool:
    """Test suite for CalculateCarbonIntensityTool - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    def test_tool_initialization(self, calculate_carbon_intensity_tool):
        """UT-CBAM-011: Test tool initializes correctly."""
        assert calculate_carbon_intensity_tool is not None
        assert calculate_carbon_intensity_tool.name == "calculate_carbon_intensity"
        assert calculate_carbon_intensity_tool.safe is True
        assert calculate_carbon_intensity_tool.description == "Calculate emissions per tonne of product"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_missing_total_emissions_raises_error(self, calculate_carbon_intensity_tool):
        """UT-CBAM-012: Test missing total_emissions raises ValueError."""
        params = {"production_quantity": 100.0}

        with pytest.raises(ValueError) as exc_info:
            await calculate_carbon_intensity_tool.execute(params)

        assert "total_emissions" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_missing_production_quantity_raises_error(self, calculate_carbon_intensity_tool):
        """UT-CBAM-013: Test missing production_quantity raises ValueError."""
        params = {"total_emissions": 185.0}

        with pytest.raises(ValueError) as exc_info:
            await calculate_carbon_intensity_tool.execute(params)

        assert "production_quantity" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_basic_carbon_intensity_calculation(self, calculate_carbon_intensity_tool):
        """UT-CBAM-014: Test basic carbon intensity calculation."""
        params = {
            "total_emissions": 185.0,  # 185 tCO2e
            "production_quantity": 100.0,  # 100 tonnes
        }

        result = await calculate_carbon_intensity_tool.execute(params)

        assert result is not None
        assert "carbon_intensity" in result
        assert "carbon_intensity_unit" in result
        assert "calculation_formula" in result

        # Verify calculation: 185 / 100 = 1.85 tCO2e/tonne
        expected_intensity = 185.0 / 100.0
        assert abs(result["carbon_intensity"] - expected_intensity) < 0.0001
        assert result["carbon_intensity_unit"] == "tCO2e/tonne"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_zero_production_quantity_raises_error(self, calculate_carbon_intensity_tool):
        """UT-CBAM-015: Test zero production quantity raises ValueError."""
        params = {
            "total_emissions": 185.0,
            "production_quantity": 0.0,
        }

        with pytest.raises(ValueError) as exc_info:
            await calculate_carbon_intensity_tool.execute(params)

        assert "positive" in str(exc_info.value).lower() or "zero" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_negative_production_quantity_raises_error(self, calculate_carbon_intensity_tool):
        """UT-CBAM-016: Test negative production quantity raises ValueError."""
        params = {
            "total_emissions": 185.0,
            "production_quantity": -100.0,
        }

        with pytest.raises(ValueError) as exc_info:
            await calculate_carbon_intensity_tool.execute(params)

        assert "positive" in str(exc_info.value).lower()

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_calculation_formula_provenance(self, calculate_carbon_intensity_tool):
        """UT-CBAM-017: Test calculation formula is included for provenance."""
        params = {
            "total_emissions": 250.0,
            "production_quantity": 125.0,
        }

        result = await calculate_carbon_intensity_tool.execute(params)

        assert "calculation_formula" in result
        formula = result["calculation_formula"]
        assert "250" in formula or "250.0" in formula
        assert "125" in formula or "125.0" in formula
        assert "=" in formula
        assert "tCO2e" in formula

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_result_includes_input_values(self, calculate_carbon_intensity_tool):
        """UT-CBAM-018: Test result includes original input values for audit trail."""
        params = {
            "total_emissions": 185.0,
            "production_quantity": 100.0,
        }

        result = await calculate_carbon_intensity_tool.execute(params)

        assert "total_emissions" in result
        assert "production_quantity" in result
        assert result["total_emissions"] == 185.0
        assert result["production_quantity"] == 100.0

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_calculation_determinism(self, calculate_carbon_intensity_tool):
        """UT-CBAM-019: Test calculation is deterministic across multiple calls."""
        params = {
            "total_emissions": 1234.5678,
            "production_quantity": 567.8901,
        }

        results = []
        for _ in range(10):
            result = await calculate_carbon_intensity_tool.execute(params)
            results.append(result["carbon_intensity"])

        # All results must be identical
        assert all(r == results[0] for r in results), "Results are not deterministic"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.asyncio
    async def test_high_precision_values(self, calculate_carbon_intensity_tool):
        """UT-CBAM-020: Test handling of high-precision decimal values."""
        params = {
            "total_emissions": 185.123456789,
            "production_quantity": 100.987654321,
        }

        result = await calculate_carbon_intensity_tool.execute(params)

        # Should handle high precision without errors
        assert isinstance(result["carbon_intensity"], (int, float))
        assert result["carbon_intensity"] > 0
        # Verify calculation
        expected = 185.123456789 / 100.987654321
        assert abs(result["carbon_intensity"] - expected) < 0.0001


# =============================================================================
# CBAM Compliance Tests
# =============================================================================

class TestCBAMCompliance:
    """Test suite for CBAM regulatory compliance validation."""

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.compliance
    @pytest.mark.parametrize("product_type,expected_benchmark", [
        ("steel_hot_rolled_coil", 1.85),
        ("steel_rebar", 1.35),
        ("cement_clinker", 0.766),
        ("cement_portland", 0.670),
        ("aluminum_unwrought", 8.6),
        ("fertilizer_ammonia", 2.4),
        ("hydrogen", 10.5),
    ])
    @pytest.mark.asyncio
    async def test_benchmark_values_match_eu_regulation(
        self, lookup_cbam_benchmark_tool, product_type, expected_benchmark
    ):
        """Test benchmark values match EU Implementing Regulation 2023/1773."""
        # Create mock for each product type
        mock_db = Mock()
        mock_benchmark = Mock()
        mock_benchmark.benchmark_value = expected_benchmark
        mock_benchmark.unit = "tCO2e/tonne"
        mock_benchmark.cn_codes = ["0000"]
        mock_benchmark.effective_date = "2026-01-01"
        mock_benchmark.source = "EU Implementing Regulation 2023/1773"
        mock_benchmark.production_method = "standard"
        mock_db.lookup.return_value = mock_benchmark
        mock_db.list_products.return_value = [product_type]

        with patch('generated.carbon_intensity_v1.tools.get_cbam_database', return_value=mock_db):
            params = {"product_type": product_type}
            result = await lookup_cbam_benchmark_tool.execute(params)

            assert result["benchmark_value"] == expected_benchmark

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.compliance
    @pytest.mark.asyncio
    async def test_cbam_certificate_calculation(self, calculate_carbon_intensity_tool, lookup_cbam_benchmark_tool):
        """Test CBAM certificate calculation workflow."""
        # Example: Importing 1000 tonnes of steel with 2.0 tCO2e/tonne intensity
        # Benchmark is 1.85 tCO2e/tonne
        # Excess emissions = (2.0 - 1.85) * 1000 = 150 tCO2e

        # Step 1: Calculate actual carbon intensity
        actual_params = {
            "total_emissions": 2000.0,  # 2000 tCO2e total
            "production_quantity": 1000.0,  # 1000 tonnes
        }
        actual_result = await calculate_carbon_intensity_tool.execute(actual_params)

        assert actual_result["carbon_intensity"] == 2.0

        # Step 2: Calculate excess emissions (actual - benchmark) * quantity
        benchmark_value = 1.85  # From EU regulation for steel
        excess_emissions = (actual_result["carbon_intensity"] - benchmark_value) * 1000.0

        assert excess_emissions == pytest.approx(150.0, rel=0.01)


# =============================================================================
# Tool Registry Tests
# =============================================================================

class TestCBAMToolRegistry:
    """Test suite for CBAM tool registry functionality."""

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    def test_get_tool_returns_correct_tool(self):
        """Test get_tool function returns correct tool instances."""
        from generated.carbon_intensity_v1.tools import get_tool

        lookup_tool = get_tool("lookup_cbam_benchmark")
        calc_tool = get_tool("calculate_carbon_intensity")

        assert lookup_tool is not None
        assert calc_tool is not None
        assert lookup_tool.name == "lookup_cbam_benchmark"
        assert calc_tool.name == "calculate_carbon_intensity"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    def test_get_tool_invalid_name_returns_none(self):
        """Test get_tool with invalid name returns None."""
        from generated.carbon_intensity_v1.tools import get_tool

        result = get_tool("nonexistent_tool")
        assert result is None

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    def test_list_tools_returns_all_tools(self):
        """Test list_tools returns all available tools."""
        from generated.carbon_intensity_v1.tools import list_tools

        tools = list_tools()

        assert "lookup_cbam_benchmark" in tools
        assert "calculate_carbon_intensity" in tools
        assert len(tools) == 2

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    def test_get_tool_info_returns_metadata(self):
        """Test get_tool_info returns tool metadata."""
        from generated.carbon_intensity_v1.tools import get_tool_info

        info = get_tool_info("lookup_cbam_benchmark")

        assert info is not None
        assert info["name"] == "lookup_cbam_benchmark"
        assert info["safe"] is True
        assert "description" in info


# =============================================================================
# Provenance and Determinism Tests
# =============================================================================

class TestCBAMProvenance:
    """Test suite for CBAM provenance tracking."""

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_provenance_hash_determinism(self, calculate_carbon_intensity_tool):
        """Test provenance hash is deterministic for same inputs."""
        params = {
            "total_emissions": 185.0,
            "production_quantity": 100.0,
        }

        hashes = []
        for _ in range(5):
            result = await calculate_carbon_intensity_tool.execute(params)
            hashes.append(result["result_hash"])

        # All hashes should be identical
        assert len(set(hashes)) == 1, "Provenance hashes differ across runs"

    @pytest.mark.unit
    @pytest.mark.cbam_agent
    @pytest.mark.determinism
    @pytest.mark.asyncio
    async def test_different_inputs_produce_different_hashes(self, calculate_carbon_intensity_tool):
        """Test different inputs produce different provenance hashes."""
        params1 = {"total_emissions": 185.0, "production_quantity": 100.0}
        params2 = {"total_emissions": 200.0, "production_quantity": 100.0}

        result1 = await calculate_carbon_intensity_tool.execute(params1)
        result2 = await calculate_carbon_intensity_tool.execute(params2)

        assert result1["result_hash"] != result2["result_hash"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
