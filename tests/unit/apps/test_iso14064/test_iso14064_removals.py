# -*- coding: utf-8 -*-
"""
Unit tests for RemovalsTracker -- ISO 14064-1:2018 Clause 6.2.

Tests removal source CRUD, permanence assessment, discount factors,
net emissions calculation, biogenic CO2 balance, verification status
lifecycle, and summary with 25+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    PermanenceLevel,
    RemovalType,
    VerificationStage,
)
from services.removals_tracker import (
    PERMANENCE_DISCOUNT_FACTORS,
    PERMANENCE_GUIDANCE,
    RemovalsTracker,
)


class TestAddRemovalSource:
    """Test adding removal sources."""

    def test_add_forestry_removal(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="forestry",
            source_name="Amazon Reforestation",
            gross_removals_tco2e=Decimal("5000"),
        )
        assert source.removal_type == RemovalType.FORESTRY
        assert source.permanence_level == PermanenceLevel.LONG_TERM
        assert source.permanence_discount_factor == Decimal("0.90")
        assert source.credited_removals_tco2e == Decimal("4500.0000")

    def test_add_ccs_removal_permanent(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="ccs",
            source_name="Geological CCS",
            gross_removals_tco2e=Decimal("10000"),
        )
        assert source.permanence_level == PermanenceLevel.PERMANENT
        assert source.credited_removals_tco2e == Decimal("10000.0000")

    def test_add_soil_carbon_medium_term(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="soil_carbon",
            source_name="Regenerative Ag",
            gross_removals_tco2e=Decimal("1000"),
        )
        assert source.permanence_level == PermanenceLevel.MEDIUM_TERM
        assert source.credited_removals_tco2e == Decimal("700.0000")

    def test_override_permanence_level(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="forestry",
            source_name="Managed Forest",
            gross_removals_tco2e=Decimal("1000"),
            permanence_level="permanent",
        )
        assert source.permanence_level == PermanenceLevel.PERMANENT
        assert source.credited_removals_tco2e == Decimal("1000.0000")

    def test_negative_gross_raises(self, removals_tracker):
        with pytest.raises(ValueError, match="negative"):
            removals_tracker.add_removal_source(
                inventory_id="inv-1",
                removal_type="forestry",
                source_name="X",
                gross_removals_tco2e=Decimal("-100"),
            )


class TestRemovalSourceCRUD:
    """Test get, update, delete operations."""

    def test_get_removal_source(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="forestry",
            source_name="Test",
            gross_removals_tco2e=Decimal("1000"),
        )
        retrieved = removals_tracker.get_removal_source(source.id)
        assert retrieved is not None
        assert retrieved.id == source.id

    def test_update_gross_removals_recalculates(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="forestry",
            source_name="Forest",
            gross_removals_tco2e=Decimal("1000"),
        )
        updated = removals_tracker.update_removal_source(
            source.id, gross_removals_tco2e=Decimal("2000"),
        )
        assert updated.gross_removals_tco2e == Decimal("2000")
        # Long-term discount 0.90 -> 2000*0.90 = 1800
        assert updated.credited_removals_tco2e == Decimal("1800.0000")

    def test_update_permanence_level_recalculates(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="forestry",
            source_name="Forest",
            gross_removals_tco2e=Decimal("1000"),
        )
        updated = removals_tracker.update_removal_source(
            source.id, permanence_level="short_term",
        )
        assert updated.permanence_level == PermanenceLevel.SHORT_TERM
        assert updated.credited_removals_tco2e == Decimal("400.0000")

    def test_delete_removal_source(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="ccs",
            source_name="CCS",
            gross_removals_tco2e=Decimal("500"),
        )
        result = removals_tracker.delete_removal_source(source.id)
        assert result is True
        assert removals_tracker.get_removal_source(source.id) is None

    def test_delete_nonexistent_raises(self, removals_tracker):
        with pytest.raises(ValueError, match="not found"):
            removals_tracker.delete_removal_source("bad-id")


class TestRemovalQueries:
    """Test query methods."""

    def test_get_removals_by_inventory(self, removals_tracker):
        removals_tracker.add_removal_source(
            "inv-1", "forestry", "F1", Decimal("1000"),
        )
        removals_tracker.add_removal_source(
            "inv-1", "ccs", "C1", Decimal("5000"),
        )
        removals_tracker.add_removal_source(
            "inv-2", "forestry", "F2", Decimal("2000"),
        )
        results = removals_tracker.get_removals_by_inventory("inv-1")
        assert len(results) == 2

    def test_get_removals_by_type(self, removals_tracker):
        removals_tracker.add_removal_source(
            "inv-1", "forestry", "F1", Decimal("1000"),
        )
        removals_tracker.add_removal_source(
            "inv-1", "ccs", "C1", Decimal("5000"),
        )
        results = removals_tracker.get_removals_by_type("inv-1", "forestry")
        assert len(results) == 1
        assert results[0].removal_type == RemovalType.FORESTRY

    def test_get_removals_by_facility(self, removals_tracker):
        removals_tracker.add_removal_source(
            "inv-1", "forestry", "F1", Decimal("1000"),
            facility_id="fac-A",
        )
        results = removals_tracker.get_removals_by_facility("inv-1", "fac-A")
        assert len(results) == 1


class TestVerificationStatus:
    """Test verification status lifecycle."""

    def test_initial_status_is_draft(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            "inv-1", "forestry", "Forest", Decimal("1000"),
        )
        assert source.verification_status == VerificationStage.DRAFT

    def test_valid_transition_draft_to_internal_review(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            "inv-1", "forestry", "Forest", Decimal("1000"),
        )
        updated = removals_tracker.update_verification_status(
            source.id, "internal_review",
        )
        assert updated.verification_status == VerificationStage.INTERNAL_REVIEW

    def test_invalid_transition_raises(self, removals_tracker):
        source = removals_tracker.add_removal_source(
            "inv-1", "forestry", "Forest", Decimal("1000"),
        )
        with pytest.raises(ValueError, match="Invalid status transition"):
            removals_tracker.update_verification_status(source.id, "verified")


class TestPermanenceAssessment:
    """Test permanence assessment with storage duration and monitoring."""

    def test_default_permanence_for_ccs(self, removals_tracker):
        result = removals_tracker.assess_permanence("ccs")
        assert result["base_permanence"] == "permanent"
        assert result["assessed_permanence"] == "permanent"

    def test_storage_duration_overrides(self, removals_tracker):
        result = removals_tracker.assess_permanence(
            "forestry", storage_duration_years=50,
        )
        assert result["assessed_permanence"] == "medium_term"

    def test_monitoring_bonus(self, removals_tracker):
        result = removals_tracker.assess_permanence(
            "forestry", has_monitoring=True,
        )
        base_factor = Decimal(result["base_discount_factor"])
        bonus = Decimal(result["confidence_bonus"])
        assert bonus == Decimal("0.05")
        effective = Decimal(result["effective_discount_factor"])
        assert effective == min(base_factor + bonus, Decimal("1.00"))

    def test_monitoring_and_buffer_pool_bonus(self, removals_tracker):
        result = removals_tracker.assess_permanence(
            "soil_carbon", has_monitoring=True, has_buffer_pool=True,
        )
        assert Decimal(result["confidence_bonus"]) == Decimal("0.10")

    def test_very_long_storage_is_permanent(self, removals_tracker):
        result = removals_tracker.assess_permanence(
            "other", storage_duration_years=5000,
        )
        assert result["assessed_permanence"] == "permanent"


class TestNetEmissions:
    """Test net emissions calculation."""

    def test_net_emissions_basic(self, removals_tracker):
        removals_tracker.add_removal_source(
            "inv-1", "ccs", "CCS1", Decimal("2000"),
        )
        result = removals_tracker.calculate_net_emissions(
            "inv-1", Decimal("10000"),
        )
        assert result["gross_emissions_tco2e"] == Decimal("10000")
        assert result["total_credited_removals"] == Decimal("2000.0000")
        assert result["net_emissions_tco2e"] == Decimal("8000.0000")
        assert result["is_net_negative"] is False

    def test_net_negative(self, removals_tracker):
        removals_tracker.add_removal_source(
            "inv-1", "ccs", "CCS1", Decimal("15000"),
        )
        result = removals_tracker.calculate_net_emissions(
            "inv-1", Decimal("10000"),
        )
        assert result["is_net_negative"] is True

    def test_net_emissions_empty_inventory(self, removals_tracker):
        result = removals_tracker.calculate_net_emissions(
            "empty-inv", Decimal("5000"),
        )
        assert result["net_emissions_tco2e"] == Decimal("5000")
        assert result["removal_source_count"] == 0


class TestBiogenicBalance:
    """Test biogenic CO2 tracking."""

    def test_biogenic_balance(self, removals_tracker):
        removals_tracker.add_removal_source(
            inventory_id="inv-1",
            removal_type="forestry",
            source_name="Forest",
            gross_removals_tco2e=Decimal("5000"),
            biogenic_co2_removals=Decimal("4000"),
            biogenic_co2_emissions=Decimal("500"),
        )
        result = removals_tracker.calculate_biogenic_balance("inv-1")
        assert result["total_biogenic_removals_tco2"] == Decimal("4000")
        assert result["total_biogenic_emissions_tco2"] == Decimal("500")
        assert result["net_biogenic_tco2"] == Decimal("3500")
        assert result["is_net_sequestration"] is True


class TestPermanenceDiscountFactors:
    """Test that discount factor constants are correct."""

    def test_permanent_no_discount(self):
        assert PERMANENCE_DISCOUNT_FACTORS[PermanenceLevel.PERMANENT] == Decimal("1.00")

    def test_long_term_10pct_discount(self):
        assert PERMANENCE_DISCOUNT_FACTORS[PermanenceLevel.LONG_TERM] == Decimal("0.90")

    def test_medium_term_30pct_discount(self):
        assert PERMANENCE_DISCOUNT_FACTORS[PermanenceLevel.MEDIUM_TERM] == Decimal("0.70")

    def test_short_term_60pct_discount(self):
        assert PERMANENCE_DISCOUNT_FACTORS[PermanenceLevel.SHORT_TERM] == Decimal("0.40")

    def test_reversible_90pct_discount(self):
        assert PERMANENCE_DISCOUNT_FACTORS[PermanenceLevel.REVERSIBLE] == Decimal("0.10")
