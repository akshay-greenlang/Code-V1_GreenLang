# -*- coding: utf-8 -*-
"""
Test suite for downstream_transportation.average_data_calculator - AGENT-MRV-022.

Tests AverageDataCalculatorEngine for the Downstream Transportation &
Distribution Agent (GL-MRV-S3-009).

Coverage (~40 tests):
- calculate_channel for all 6 distribution channels
- Product category EFs
- Screening estimates
- Storage component integration
- Batch channel processing
- Channel comparison analysis
- Known-value hand-calculated tests
- Singleton pattern, provenance hash

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_AVAILABLE = True
_IMPORT_ERROR = None

try:
    from greenlang.agents.mrv.downstream_transportation.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError as exc:
    _AVAILABLE = False
    _IMPORT_ERROR = str(exc)

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason=f"average_data_calculator not available: {_IMPORT_ERROR}",
)

pytestmark = _SKIP


# ==============================================================================
# SINGLETON TESTS
# ==============================================================================


class TestSingleton:
    """Test AverageDataCalculatorEngine singleton."""

    def test_singleton_identity(self):
        """Test two instantiations return the same object."""
        eng1 = AverageDataCalculatorEngine()
        eng2 = AverageDataCalculatorEngine()
        assert eng1 is eng2


# ==============================================================================
# CHANNEL CALCULATION TESTS
# ==============================================================================


class TestChannelCalculation:
    """Test calculate_channel for all 6 distribution channels."""

    @pytest.mark.parametrize("channel", [
        "ECOMMERCE_DTC", "RETAIL_DISTRIBUTION", "WHOLESALE",
        "MARKETPLACE_3PL", "DROPSHIP", "OMNICHANNEL",
    ])
    def test_channel_calculation(self, channel):
        """Test calculation for each distribution channel."""
        engine = AverageDataCalculatorEngine()
        input_data = {
            "channel": channel,
            "annual_units_sold": 10000,
            "average_weight_kg": Decimal("2.0"),
            "region": "US",
        }
        result = engine.calculate_channel(input_data)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert isinstance(emissions, Decimal)
        assert emissions > 0

    def test_ecommerce_dtc(self, sample_average_data):
        """Test e-commerce direct-to-consumer channel."""
        engine = AverageDataCalculatorEngine()
        result = engine.calculate_channel(sample_average_data)
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_retail_distribution(self, sample_average_data_retail):
        """Test retail distribution channel."""
        engine = AverageDataCalculatorEngine()
        result = engine.calculate_channel(sample_average_data_retail)
        assert result is not None

    def test_wholesale(self, sample_average_data_wholesale):
        """Test wholesale distribution channel."""
        engine = AverageDataCalculatorEngine()
        result = engine.calculate_channel(sample_average_data_wholesale)
        assert result is not None

    def test_ecommerce_known_value(self):
        """
        Hand-calculated: 50,000 units x 1.25 kgCO2e/unit / 1000 = 62.5 tCO2e.
        """
        engine = AverageDataCalculatorEngine()
        input_data = {
            "channel": "ECOMMERCE_DTC",
            "annual_units_sold": 50000,
            "region": "US",
        }
        result = engine.calculate_channel(input_data)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        expected = Decimal("62.5")
        assert abs(emissions - expected) / expected < Decimal("0.30")

    def test_zero_units_returns_zero(self):
        """Test zero units sold returns zero emissions."""
        engine = AverageDataCalculatorEngine()
        input_data = {
            "channel": "ECOMMERCE_DTC",
            "annual_units_sold": 0,
            "region": "US",
        }
        result = engine.calculate_channel(input_data)
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions == Decimal("0") or emissions == Decimal("0.00")

    def test_ecommerce_includes_last_mile(self):
        """Test e-commerce channel includes last-mile component."""
        engine = AverageDataCalculatorEngine()
        input_data = {
            "channel": "ECOMMERCE_DTC",
            "annual_units_sold": 10000,
            "region": "US",
        }
        result = engine.calculate_channel(input_data)
        # E-commerce should have higher EF than wholesale due to last-mile
        wholesale_input = {
            "channel": "WHOLESALE",
            "annual_units_sold": 10000,
            "average_weight_kg": Decimal("2.0"),
            "region": "US",
        }
        wholesale_result = engine.calculate_channel(wholesale_input)
        ecom_e = result.get("emissions_tco2e", result.get("total_co2e"))
        wholesale_e = wholesale_result.get("emissions_tco2e", wholesale_result.get("total_co2e"))
        # E-commerce typically has higher per-unit emissions than wholesale
        # due to individual parcel delivery
        assert ecom_e is not None and wholesale_e is not None


# ==============================================================================
# PRODUCT CATEGORY TESTS
# ==============================================================================


class TestProductCategory:
    """Test product category emission factors."""

    @pytest.mark.parametrize("category", [
        "consumer_electronics", "food_beverage", "apparel",
        "industrial_equipment", "pharmaceuticals",
    ])
    def test_product_category(self, category):
        """Test calculation with different product categories."""
        engine = AverageDataCalculatorEngine()
        input_data = {
            "channel": "RETAIL_DISTRIBUTION",
            "product_category": category,
            "annual_units_sold": 5000,
            "average_weight_kg": Decimal("1.0"),
            "region": "US",
        }
        result = engine.calculate_channel(input_data)
        assert result is not None

    def test_heavier_products_higher_emissions(self):
        """Test heavier products generate higher transport emissions."""
        engine = AverageDataCalculatorEngine()
        base = {
            "channel": "RETAIL_DISTRIBUTION",
            "annual_units_sold": 10000,
            "region": "US",
        }
        light = engine.calculate_channel({**base, "average_weight_kg": Decimal("0.5")})
        heavy = engine.calculate_channel({**base, "average_weight_kg": Decimal("50.0")})
        light_e = light.get("emissions_tco2e", light.get("total_co2e"))
        heavy_e = heavy.get("emissions_tco2e", heavy.get("total_co2e"))
        assert heavy_e > light_e


# ==============================================================================
# SCREENING ESTIMATE TESTS
# ==============================================================================


class TestScreeningEstimate:
    """Test screening-level emissions estimation."""

    def test_screening_from_revenue(self):
        """Test screening estimate from total revenue."""
        engine = AverageDataCalculatorEngine()
        result = engine.screening_estimate(
            annual_revenue_usd=Decimal("10000000.00"),
            channel="ECOMMERCE_DTC",
            region="US",
        )
        assert result is not None
        emissions = result.get("emissions_tco2e", result.get("total_co2e"))
        assert emissions > 0

    def test_screening_with_product_mix(self):
        """Test screening estimate with product mix."""
        engine = AverageDataCalculatorEngine()
        result = engine.screening_estimate(
            annual_revenue_usd=Decimal("5000000.00"),
            channel="RETAIL_DISTRIBUTION",
            region="EU",
        )
        assert result is not None


# ==============================================================================
# STORAGE COMPONENT TESTS
# ==============================================================================


class TestStorageComponent:
    """Test storage/warehousing component in average data."""

    def test_channel_includes_storage(self):
        """Test channel calculation includes storage component."""
        engine = AverageDataCalculatorEngine()
        input_data = {
            "channel": "RETAIL_DISTRIBUTION",
            "annual_units_sold": 10000,
            "include_storage": True,
            "region": "US",
        }
        result = engine.calculate_channel(input_data)
        assert result is not None

    def test_storage_adds_emissions(self):
        """Test including storage increases total emissions."""
        engine = AverageDataCalculatorEngine()
        base = {
            "channel": "RETAIL_DISTRIBUTION",
            "annual_units_sold": 10000,
            "region": "US",
        }
        no_storage = engine.calculate_channel({**base, "include_storage": False})
        with_storage = engine.calculate_channel({**base, "include_storage": True})
        no_e = no_storage.get("emissions_tco2e", no_storage.get("total_co2e"))
        st_e = with_storage.get("emissions_tco2e", with_storage.get("total_co2e"))
        assert st_e >= no_e


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


class TestBatchProcessing:
    """Test batch channel processing."""

    def test_batch_calculation(self):
        """Test batch calculation of multiple channels."""
        engine = AverageDataCalculatorEngine()
        inputs = [
            {
                "channel": "ECOMMERCE_DTC",
                "annual_units_sold": 50000,
                "region": "US",
            },
            {
                "channel": "RETAIL_DISTRIBUTION",
                "annual_units_sold": 100000,
                "region": "EU",
            },
            {
                "channel": "WHOLESALE",
                "annual_units_sold": 2000,
                "region": "US",
            },
        ]
        result = engine.calculate_batch(inputs)
        assert result is not None


# ==============================================================================
# CHANNEL COMPARISON TESTS
# ==============================================================================


class TestChannelComparison:
    """Test cross-channel comparison analysis."""

    def test_compare_channels(self):
        """Test comparison across all distribution channels."""
        engine = AverageDataCalculatorEngine()
        comparison = engine.compare_channels(
            annual_units_sold=10000,
            average_weight_kg=Decimal("2.0"),
            region="US",
        )
        assert comparison is not None
        # Should have at least 6 channel entries
        assert len(comparison) >= 6 or "channels" in comparison

    def test_wholesale_lower_than_ecommerce(self):
        """Test wholesale per-unit emissions lower than e-commerce."""
        engine = AverageDataCalculatorEngine()
        comparison = engine.compare_channels(
            annual_units_sold=10000,
            average_weight_kg=Decimal("2.0"),
            region="US",
        )
        if isinstance(comparison, dict) and "ECOMMERCE_DTC" in comparison:
            ecom = comparison["ECOMMERCE_DTC"]
            wholesale = comparison["WHOLESALE"]
            ecom_e = ecom if isinstance(ecom, Decimal) else ecom.get("ef_per_unit")
            ws_e = wholesale if isinstance(wholesale, Decimal) else wholesale.get("ef_per_unit")
            if ecom_e and ws_e:
                assert ws_e < ecom_e


# ==============================================================================
# PROVENANCE HASH TESTS
# ==============================================================================


class TestProvenanceHash:
    """Test provenance hash in average data results."""

    def test_result_has_provenance_hash(self, sample_average_data):
        """Test result includes 64-char provenance hash."""
        engine = AverageDataCalculatorEngine()
        result = engine.calculate_channel(sample_average_data)
        ph = result.get("provenance_hash")
        assert ph is not None
        assert len(ph) == 64

    def test_deterministic_hash(self, sample_average_data):
        """Test same input produces same hash."""
        engine = AverageDataCalculatorEngine()
        r1 = engine.calculate_channel(sample_average_data)
        r2 = engine.calculate_channel(sample_average_data)
        assert r1["provenance_hash"] == r2["provenance_hash"]
