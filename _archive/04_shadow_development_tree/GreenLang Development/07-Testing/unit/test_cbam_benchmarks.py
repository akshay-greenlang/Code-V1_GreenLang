# -*- coding: utf-8 -*-
"""
Unit Tests for CBAM Benchmarks Database

Comprehensive test suite with 10 test cases covering:
- CBAMBenchmarkDatabase initialization and loading
- Product type lookups
- CN code lookups
- EU Regulation compliance

Target: 85%+ coverage for CBAM benchmarks database
Run with: pytest tests/unit/test_cbam_benchmarks.py -v --cov=core/greenlang/data/cbam_benchmarks

Author: GL-TestEngineer
Version: 1.0.0

CBAM benchmarks are EU default values from Implementing Regulation 2023/1773.
"""

import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def cbam_database():
    """Create CBAMBenchmarkDatabase instance."""
    from core.greenlang.data.cbam_benchmarks import CBAMBenchmarkDatabase
    return CBAMBenchmarkDatabase()


@pytest.fixture
def expected_benchmarks():
    """Expected CBAM benchmark values from EU Regulation 2023/1773."""
    return {
        "steel_hot_rolled_coil": 1.85,
        "steel_rebar": 1.35,
        "steel_wire_rod": 1.75,
        "cement_clinker": 0.766,
        "cement_portland": 0.670,
        "aluminum_unwrought": 8.6,
        "aluminum_products": 1.5,
        "fertilizer_ammonia": 2.4,
        "fertilizer_urea": 1.6,
        "fertilizer_nitric_acid": 0.5,
        "electricity": 0.429,
        "hydrogen": 10.5,
    }


# =============================================================================
# CBAMBenchmarkDatabase Tests (10 tests)
# =============================================================================

class TestCBAMBenchmarkDatabase:
    """Test suite for CBAMBenchmarkDatabase - 10 test cases."""

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_database_initialization(self, cbam_database):
        """UT-CBAM-DB-001: Test database initializes correctly."""
        assert cbam_database is not None
        assert cbam_database.benchmarks is not None
        assert isinstance(cbam_database.benchmarks, dict)
        assert len(cbam_database.benchmarks) > 0

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_cbam_database_singleton(self):
        """UT-CBAM-DB-002: Test get_cbam_database returns singleton instance."""
        from core.greenlang.data.cbam_benchmarks import get_cbam_database

        db1 = get_cbam_database()
        db2 = get_cbam_database()

        # Should return same instance
        assert db1 is db2

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_steel_benchmark(self, cbam_database):
        """UT-CBAM-DB-003: Test lookup returns correct steel benchmark."""
        result = cbam_database.lookup("steel_hot_rolled_coil")

        assert result is not None
        assert result.benchmark_value == 1.85
        assert result.unit == "tCO2e/tonne"
        assert result.production_method == "basic_oxygen_furnace"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_cement_benchmark(self, cbam_database):
        """UT-CBAM-DB-004: Test lookup returns correct cement benchmark."""
        result = cbam_database.lookup("cement_portland")

        assert result is not None
        assert result.benchmark_value == 0.670
        assert result.unit == "tCO2e/tonne"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_aluminum_benchmark(self, cbam_database):
        """UT-CBAM-DB-005: Test lookup returns correct aluminum benchmark."""
        result = cbam_database.lookup("aluminum_unwrought")

        assert result is not None
        assert result.benchmark_value == 8.6
        assert result.production_method == "electrolysis"

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_lookup_returns_none_for_invalid(self, cbam_database):
        """UT-CBAM-DB-006: Test lookup returns None for invalid product type."""
        result = cbam_database.lookup("invalid_product_xyz")

        assert result is None

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_list_products_returns_all_products(self, cbam_database):
        """UT-CBAM-DB-007: Test list_products returns all CBAM products."""
        products = cbam_database.list_products()

        assert isinstance(products, list)
        assert len(products) >= 10  # At least 10 CBAM products

        # Check key products are present
        assert "steel_hot_rolled_coil" in products
        assert "cement_portland" in products
        assert "aluminum_unwrought" in products

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_by_cn_code_steel(self, cbam_database):
        """UT-CBAM-DB-008: Test lookup by CN code for steel products."""
        result = cbam_database.get_by_cn_code("7208")

        assert result is not None
        assert "steel" in result.product_type.lower()

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_get_by_cn_code_cement(self, cbam_database):
        """UT-CBAM-DB-009: Test lookup by CN code for cement products."""
        result = cbam_database.get_by_cn_code("2523")

        assert result is not None
        assert "cement" in result.product_type.lower()

    @pytest.mark.unit
    @pytest.mark.core_library
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
    def test_benchmark_values_match_eu_regulation(
        self,
        cbam_database,
        product_type,
        expected_benchmark
    ):
        """UT-CBAM-DB-010: Test benchmark values match EU Implementing Regulation 2023/1773."""
        result = cbam_database.lookup(product_type)

        assert result is not None, f"Product {product_type} not found"
        assert result.benchmark_value == expected_benchmark, \
            f"Benchmark for {product_type} should be {expected_benchmark}, got {result.benchmark_value}"


# =============================================================================
# CBAM Data Quality Tests
# =============================================================================

class TestCBAMDataQuality:
    """Test suite for CBAM data quality and provenance."""

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_benchmark_has_source_citation(self, cbam_database):
        """Test benchmark includes source citation from EU Regulation."""
        result = cbam_database.lookup("steel_hot_rolled_coil")

        assert result is not None
        assert result.source is not None
        assert "EU" in result.source or "Regulation" in result.source

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_benchmark_has_effective_date(self, cbam_database):
        """Test benchmark includes effective date."""
        result = cbam_database.lookup("steel_hot_rolled_coil")

        assert result is not None
        assert result.effective_date is not None
        assert "2026" in result.effective_date  # CBAM full implementation in 2026

    @pytest.mark.unit
    @pytest.mark.core_library
    def test_benchmark_has_cn_codes(self, cbam_database):
        """Test benchmark includes Combined Nomenclature codes."""
        result = cbam_database.lookup("steel_hot_rolled_coil")

        assert result is not None
        assert result.cn_codes is not None
        assert len(result.cn_codes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
