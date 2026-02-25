"""
Test suite for AGENT-MRV-014 provenance tracking.

Tests the PurchasedGoodsProvenanceTracker singleton and SHA-256 hash chain integrity.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from greenlang.purchased_goods_services.provenance import PurchasedGoodsProvenanceTracker
from greenlang.purchased_goods_services.models import (
    CalculationMethod,
    EEIODatabase,
    PhysicalEFSource,
    SupplierDataSource,
)


class TestPurchasedGoodsProvenanceTracker:
    """Test PurchasedGoodsProvenanceTracker singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        PurchasedGoodsProvenanceTracker.reset()

    def test_singleton_pattern(self):
        """Test provenance tracker follows singleton pattern."""
        tracker1 = PurchasedGoodsProvenanceTracker()
        tracker2 = PurchasedGoodsProvenanceTracker()
        assert tracker1 is tracker2

    def test_record_procurement_intake(self):
        """Test recording procurement intake stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 produces 64-character hex string

    def test_record_method_selection(self):
        """Test recording method selection stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_method_selection(
            item_id="ITEM-001",
            selected_method=CalculationMethod.SPEND_BASED,
            reasons=["No supplier-specific data available", "Spend data is reliable"],
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_eeio_lookup(self):
        """Test recording EEIO lookup stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_eeio_lookup(
            item_id="ITEM-001",
            database=EEIODatabase.EXIOBASE,
            sector="Iron and steel",
            emission_factor=Decimal("0.833"),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_physical_ef_lookup(self):
        """Test recording physical emission factor lookup stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_physical_ef_lookup(
            item_id="ITEM-002",
            ef_source=PhysicalEFSource.IPCC_2006,
            material="Cement",
            emission_factor=Decimal("0.85"),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_supplier_data_retrieval(self):
        """Test recording supplier data retrieval stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_supplier_data_retrieval(
            item_id="ITEM-003",
            supplier_id="SUP-CONCRETE-789",
            data_source=SupplierDataSource.EPD,
            emission_factor=Decimal("0.245"),
            epd_number="EPD-CONC-2024-001",
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_currency_conversion(self):
        """Test recording currency conversion stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_currency_conversion(
            item_id="ITEM-001",
            from_currency="EUR",
            to_currency="USD",
            exchange_rate=Decimal("1.12"),
            converted_amount=Decimal("42000.00"),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_emission_calculation(self):
        """Test recording emission calculation stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_emission_calculation(
            item_id="ITEM-001",
            method=CalculationMethod.SPEND_BASED,
            total_emissions=Decimal("12500.00"),
            calculation_formula="spend * emission_factor",
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_allocation_applied(self):
        """Test recording allocation application stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_allocation_applied(
            item_id="ITEM-004",
            allocation_method="economic",
            allocation_factor=Decimal("0.65"),
            allocated_emissions=Decimal("8500.00"),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_uncertainty_analysis(self):
        """Test recording uncertainty analysis stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_uncertainty_analysis(
            item_id="ITEM-001",
            distribution="lognormal",
            confidence_interval_95=(Decimal("7500.00"), Decimal("12500.00")),
            monte_carlo_iterations=10000,
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_dqi_assessment(self):
        """Test recording DQI assessment stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_dqi_assessment(
            item_id="ITEM-001",
            technological=4,
            temporal=5,
            geographical=3,
            completeness=4,
            reliability=5,
            overall_score=Decimal("4.2"),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_validation_checks(self):
        """Test recording validation checks stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_validation_checks(
            item_id="ITEM-001",
            checks_performed=["emission_factor_range", "data_completeness", "consistency"],
            all_passed=True,
            warnings=["Low supplier coverage"],
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_aggregation(self):
        """Test recording aggregation stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_aggregation(
            batch_id="BATCH-2024-Q1",
            item_count=150,
            total_emissions=Decimal("325000.00"),
            aggregation_method="sum",
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_compliance_mapping(self):
        """Test recording compliance mapping stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_compliance_mapping(
            item_id="ITEM-001",
            framework="GHG_PROTOCOL",
            scope="Scope 3",
            category="Category 1",
            disclosure_requirements=["Total emissions", "Data quality rating"],
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_report_generation(self):
        """Test recording report generation stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_report_generation(
            report_id="RPT-2024-Q1",
            report_type="compliance_disclosure",
            framework="GHG_PROTOCOL",
            timestamp=datetime(2024, 4, 1, 10, 0, 0),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_database_persistence(self):
        """Test recording database persistence stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_database_persistence(
            item_id="ITEM-001",
            table_names=["gl_pgs_procurement_items", "gl_pgs_spend_based_results"],
            row_count=2,
            transaction_id="TXN-123456",
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_export(self):
        """Test recording export stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_export(
            export_id="EXP-2024-001",
            export_format="json",
            record_count=150,
            file_path="/exports/pgs-2024-q1.json",
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_record_audit_finalization(self):
        """Test recording audit finalization stage."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash_value = tracker.record_audit_finalization(
            audit_id="AUDIT-2024-Q1",
            total_items=150,
            total_emissions=Decimal("325000.00"),
            completion_timestamp=datetime(2024, 4, 1, 15, 30, 0),
        )

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_compute_chain_hash(self):
        """Test computing chain hash from multiple stages."""
        tracker = PurchasedGoodsProvenanceTracker()

        # Record multiple stages
        hash1 = tracker.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        hash2 = tracker.record_method_selection(
            item_id="ITEM-001",
            selected_method=CalculationMethod.SPEND_BASED,
            reasons=["No supplier-specific data"],
        )

        hash3 = tracker.record_eeio_lookup(
            item_id="ITEM-001",
            database=EEIODatabase.EXIOBASE,
            sector="Iron and steel",
            emission_factor=Decimal("0.833"),
        )

        # Compute chain hash
        chain_hash = tracker.compute_chain_hash([hash1, hash2, hash3])

        assert isinstance(chain_hash, str)
        assert len(chain_hash) == 64

    def test_verify_chain_integrity(self):
        """Test verifying chain integrity."""
        tracker = PurchasedGoodsProvenanceTracker()

        # Create a hash chain
        hash1 = tracker.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        hash2 = tracker.record_emission_calculation(
            item_id="ITEM-001",
            method=CalculationMethod.SPEND_BASED,
            total_emissions=Decimal("12500.00"),
            calculation_formula="spend * emission_factor",
        )

        expected_chain_hash = tracker.compute_chain_hash([hash1, hash2])

        # Verify integrity
        is_valid = tracker.verify_chain_integrity([hash1, hash2], expected_chain_hash)
        assert is_valid is True

        # Test with tampered chain
        tampered_hash = "a" * 64
        is_valid_tampered = tracker.verify_chain_integrity(
            [hash1, tampered_hash], expected_chain_hash
        )
        assert is_valid_tampered is False

    def test_batch_provenance(self):
        """Test provenance tracking for batch processing."""
        tracker = PurchasedGoodsProvenanceTracker()

        # Record batch stages
        hash1 = tracker.record_procurement_intake(
            item_id="BATCH-ITEM-001",
            item_name="Item 1",
            quantity=Decimal("1000.00"),
            spend=Decimal("5000.00"),
        )

        hash2 = tracker.record_procurement_intake(
            item_id="BATCH-ITEM-002",
            item_name="Item 2",
            quantity=Decimal("2000.00"),
            spend=Decimal("10000.00"),
        )

        hash3 = tracker.record_aggregation(
            batch_id="BATCH-2024-Q1",
            item_count=2,
            total_emissions=Decimal("15000.00"),
            aggregation_method="sum",
        )

        # All hashes should be unique
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3

        # All hashes should be valid SHA-256
        assert len(hash1) == 64
        assert len(hash2) == 64
        assert len(hash3) == 64

    def test_reset_clears_singleton(self):
        """Test reset() clears the singleton instance."""
        tracker1 = PurchasedGoodsProvenanceTracker()
        tracker1_id = id(tracker1)

        # Record some data
        tracker1.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        PurchasedGoodsProvenanceTracker.reset()

        tracker2 = PurchasedGoodsProvenanceTracker()
        tracker2_id = id(tracker2)

        # After reset, should get a new instance
        assert tracker1_id != tracker2_id

    def test_hash_determinism(self):
        """Test that same input produces same hash."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash1 = tracker.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        # Reset and record same data
        PurchasedGoodsProvenanceTracker.reset()
        tracker = PurchasedGoodsProvenanceTracker()

        hash2 = tracker.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        # Same input should produce same hash
        assert hash1 == hash2

    def test_hash_sensitivity(self):
        """Test that different input produces different hash."""
        tracker = PurchasedGoodsProvenanceTracker()

        hash1 = tracker.record_procurement_intake(
            item_id="ITEM-001",
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        hash2 = tracker.record_procurement_intake(
            item_id="ITEM-002",  # Different ID
            item_name="Steel beams",
            quantity=Decimal("15000.00"),
            spend=Decimal("37500.00"),
        )

        # Different input should produce different hash
        assert hash1 != hash2
