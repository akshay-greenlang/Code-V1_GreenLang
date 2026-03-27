"""
Test suite for AGENT-MRV-014 metrics.

Tests the PurchasedGoodsServicesMetrics singleton and Prometheus metrics collection.
"""

import pytest
from decimal import Decimal

from greenlang.agents.mrv.purchased_goods_services.metrics import PurchasedGoodsServicesMetrics
from greenlang.agents.mrv.purchased_goods_services.models import CalculationMethod, EEIODatabase


class TestPurchasedGoodsServicesMetrics:
    """Test PurchasedGoodsServicesMetrics singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        PurchasedGoodsServicesMetrics.reset()

    def test_singleton_pattern(self):
        """Test metrics follows singleton pattern."""
        metrics1 = PurchasedGoodsServicesMetrics()
        metrics2 = PurchasedGoodsServicesMetrics()
        assert metrics1 is metrics2

    def test_metric_prefix(self):
        """Test all metrics have gl_pgs_ prefix."""
        metrics = PurchasedGoodsServicesMetrics()

        # Check counter names
        assert metrics.items_processed._name == "gl_pgs_items_processed_total"
        assert metrics.calculations_performed._name == "gl_pgs_calculations_performed_total"
        assert metrics.supplier_records_retrieved._name == "gl_pgs_supplier_records_retrieved_total"
        assert metrics.eeio_lookups._name == "gl_pgs_eeio_lookups_total"
        assert metrics.validation_errors._name == "gl_pgs_validation_errors_total"

        # Check histogram names
        assert metrics.calculation_duration._name == "gl_pgs_calculation_duration_seconds"
        assert metrics.emission_intensity._name == "gl_pgs_emission_intensity_kg_per_unit"
        assert metrics.data_quality_score._name == "gl_pgs_data_quality_score"
        assert metrics.supplier_coverage._name == "gl_pgs_supplier_coverage_percentage"
        assert metrics.batch_size._name == "gl_pgs_batch_size"

    def test_record_items_processed(self):
        """Test recording items processed counter."""
        metrics = PurchasedGoodsServicesMetrics()

        # Record some items
        metrics.record_items_processed(
            method=CalculationMethod.SPEND_BASED,
            category="metals_steel",
            count=10
        )
        metrics.record_items_processed(
            method=CalculationMethod.AVERAGE_DATA,
            category="cement_lime",
            count=5
        )

        # Verify counter incremented (checking internal state)
        assert metrics.items_processed._metrics != {}

    def test_record_calculations_performed(self):
        """Test recording calculations performed counter."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_calculations_performed(
            method=CalculationMethod.SUPPLIER_SPECIFIC,
            success=True
        )
        metrics.record_calculations_performed(
            method=CalculationMethod.HYBRID,
            success=False
        )

        assert metrics.calculations_performed._metrics != {}

    def test_record_supplier_records_retrieved(self):
        """Test recording supplier records retrieved counter."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_supplier_records_retrieved(
            data_source="EPD",
            count=25
        )
        metrics.record_supplier_records_retrieved(
            data_source="CDP",
            count=15
        )

        assert metrics.supplier_records_retrieved._metrics != {}

    def test_record_eeio_lookups(self):
        """Test recording EEIO lookups counter."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_eeio_lookups(
            database=EEIODatabase.EXIOBASE,
            sector="Iron and steel",
            hit=True
        )
        metrics.record_eeio_lookups(
            database=EEIODatabase.USEEIO,
            sector="Unknown sector",
            hit=False
        )

        assert metrics.eeio_lookups._metrics != {}

    def test_record_validation_errors(self):
        """Test recording validation errors counter."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_validation_errors(
            error_type="invalid_emission_factor",
            severity="high"
        )
        metrics.record_validation_errors(
            error_type="missing_supplier_data",
            severity="medium"
        )

        assert metrics.validation_errors._metrics != {}

    def test_record_calculation_duration(self):
        """Test recording calculation duration histogram."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_calculation_duration(
            method=CalculationMethod.SPEND_BASED,
            duration_seconds=1.25
        )
        metrics.record_calculation_duration(
            method=CalculationMethod.AVERAGE_DATA,
            duration_seconds=0.85
        )

        assert metrics.calculation_duration._metrics != {}

    def test_record_emission_intensity(self):
        """Test recording emission intensity histogram."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_emission_intensity(
            category="metals_aluminum",
            intensity=Decimal("8.5")
        )
        metrics.record_emission_intensity(
            category="plastics",
            intensity=Decimal("3.2")
        )

        assert metrics.emission_intensity._metrics != {}

    def test_record_data_quality_score(self):
        """Test recording data quality score histogram."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_data_quality_score(
            method=CalculationMethod.SUPPLIER_SPECIFIC,
            score=Decimal("4.5")
        )
        metrics.record_data_quality_score(
            method=CalculationMethod.SPEND_BASED,
            score=Decimal("2.8")
        )

        assert metrics.data_quality_score._metrics != {}

    def test_record_supplier_coverage(self):
        """Test recording supplier coverage histogram."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_supplier_coverage(
            category="electronics",
            coverage_percentage=Decimal("75.0")
        )
        metrics.record_supplier_coverage(
            category="business_services",
            coverage_percentage=Decimal("25.0")
        )

        assert metrics.supplier_coverage._metrics != {}

    def test_record_batch_size(self):
        """Test recording batch size histogram."""
        metrics = PurchasedGoodsServicesMetrics()

        metrics.record_batch_size(batch_size=1000)
        metrics.record_batch_size(batch_size=2500)

        assert metrics.batch_size._metrics != {}

    def test_reset_clears_singleton(self):
        """Test reset() clears the singleton instance."""
        metrics1 = PurchasedGoodsServicesMetrics()
        metrics1_id = id(metrics1)

        # Record some data
        metrics1.record_items_processed(
            method=CalculationMethod.SPEND_BASED,
            category="metals_steel",
            count=10
        )

        PurchasedGoodsServicesMetrics.reset()

        metrics2 = PurchasedGoodsServicesMetrics()
        metrics2_id = id(metrics2)

        # After reset, should get a new instance
        assert metrics1_id != metrics2_id
