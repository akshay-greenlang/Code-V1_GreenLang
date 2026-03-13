# -*- coding: utf-8 -*-
"""
Tests for SupplierProfileManager - AGENT-EUDR-008 Engine 2: Profile Management

Comprehensive test suite covering:
- CRUD operations with full validation (F2.1-F2.8)
- Profile completeness scoring with all weight categories (F2.9)
- Profile versioning and merge (F2.10)
- Search with various criteria
- Edge cases: empty fields, invalid country codes, missing data

Test count: 65+ tests
Coverage target: >= 85% of SupplierProfileManager module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    COCOA_IMPORTER_EU,
    COCOA_TRADER_GH,
    COCOA_PROCESSOR_GH,
    COCOA_FARMER_1_GH,
    COCOA_FARMER_2_GH,
    COFFEE_EXPORTER_CO,
    PALM_REFINERY_ID,
    PALM_SMALLHOLDER_ID,
    TIMBER_SAWMILL_CD,
    SUP_ID_COCOA_TRADER_GH,
    SUP_ID_COCOA_FARMER_1_GH,
    SUP_ID_PALM_SMALLHOLDER_ID,
    EUDR_COMMODITIES,
    PROFILE_COMPLETENESS_WEIGHTS,
    SHA256_HEX_LENGTH,
    make_supplier,
    assert_valid_completeness_score,
    compute_sha256,
)


# ===========================================================================
# 1. Create Operations
# ===========================================================================


class TestSupplierProfileCreate:
    """Test creation of new supplier profiles."""

    def test_create_full_profile(self, supplier_profile_manager):
        """Create a supplier with all fields populated."""
        supplier = make_supplier(
            legal_name="Full Profile Corp",
            registration_id="GH-BRN-FULL-001",
            tax_id="GH_TAX_FULL",
            country_iso="GH",
            tier=1,
            role="trader",
            commodity="cocoa",
            gps_lat=5.6037,
            gps_lon=-0.1870,
            primary_contact="Alice",
            compliance_contact="Bob",
            certifications=["UTZ-001"],
            dds_references=["DDS-001"],
        )
        result = supplier_profile_manager.create(supplier)
        assert result is not None
        assert result["legal_name"] == "Full Profile Corp"
        assert result.get("supplier_id") is not None

    def test_create_minimal_profile(self, supplier_profile_manager):
        """Create a supplier with only required fields."""
        supplier = make_supplier(
            legal_name="Minimal Supplier",
            country_iso="GH",
            commodity="cocoa",
        )
        result = supplier_profile_manager.create(supplier)
        assert result is not None
        assert result["legal_name"] == "Minimal Supplier"

    def test_create_assigns_supplier_id(self, supplier_profile_manager):
        """Creating a profile without supplier_id auto-assigns one."""
        supplier = make_supplier(supplier_id=None, legal_name="Auto-ID")
        result = supplier_profile_manager.create(supplier)
        assert result["supplier_id"] is not None
        assert len(result["supplier_id"]) > 0

    def test_create_duplicate_supplier_id_raises(self, supplier_profile_manager):
        """Creating with an existing supplier_id raises an error."""
        supplier = make_supplier(supplier_id="SUP-DUP-001", legal_name="First")
        supplier_profile_manager.create(supplier)
        duplicate = make_supplier(supplier_id="SUP-DUP-001", legal_name="Second")
        with pytest.raises((ValueError, KeyError)):
            supplier_profile_manager.create(duplicate)

    def test_create_missing_legal_name_raises(self, supplier_profile_manager):
        """Profile without legal_name should raise ValueError."""
        supplier = make_supplier(legal_name="")
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_create_with_each_commodity(self, supplier_profile_manager, commodity):
        """Profile creation works for each of the 7 EUDR commodities."""
        supplier = make_supplier(legal_name=f"{commodity} Supplier", commodity=commodity)
        result = supplier_profile_manager.create(supplier)
        assert commodity in result["commodities"]

    def test_create_sets_timestamp(self, supplier_profile_manager):
        """Profile creation should set created_at timestamp."""
        supplier = make_supplier(legal_name="Timestamp Test")
        result = supplier_profile_manager.create(supplier)
        assert "created_at" in result or "created_at" in result.get("metadata", {})

    def test_create_provenance_hash(self, supplier_profile_manager):
        """Profile creation generates provenance hash."""
        supplier = make_supplier(legal_name="Provenance Test")
        result = supplier_profile_manager.create(supplier)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. Read Operations
# ===========================================================================


class TestSupplierProfileRead:
    """Test reading/retrieving supplier profiles."""

    def test_get_existing_profile(self, supplier_profile_manager):
        """Retrieve an existing supplier profile by ID."""
        supplier = make_supplier(supplier_id="SUP-READ-001", legal_name="Readable Corp")
        supplier_profile_manager.create(supplier)
        result = supplier_profile_manager.get("SUP-READ-001")
        assert result is not None
        assert result["legal_name"] == "Readable Corp"

    def test_get_nonexistent_profile_returns_none(self, supplier_profile_manager):
        """Requesting a non-existent profile returns None or raises."""
        result = supplier_profile_manager.get("SUP-NONEXISTENT")
        assert result is None

    def test_get_returns_all_fields(self, supplier_profile_manager):
        """Retrieved profile includes all stored fields."""
        supplier = make_supplier(
            supplier_id="SUP-ALLFIELDS",
            legal_name="AllFields Corp",
            registration_id="REG-001",
            tax_id="TAX-001",
            gps_lat=5.6037,
            gps_lon=-0.1870,
        )
        supplier_profile_manager.create(supplier)
        result = supplier_profile_manager.get("SUP-ALLFIELDS")
        assert result["registration_id"] == "REG-001"
        assert result["tax_id"] == "TAX-001"
        assert result["gps_lat"] == pytest.approx(5.6037, abs=0.001)

    def test_list_all_profiles(self, supplier_profile_manager):
        """List all stored supplier profiles."""
        for i in range(3):
            supplier_profile_manager.create(
                make_supplier(supplier_id=f"SUP-LIST-{i}", legal_name=f"List Supplier {i}")
            )
        result = supplier_profile_manager.list_all()
        assert len(result) >= 3


# ===========================================================================
# 3. Update Operations
# ===========================================================================


class TestSupplierProfileUpdate:
    """Test updating existing supplier profiles."""

    def test_update_legal_name(self, supplier_profile_manager):
        """Update the legal name of a supplier."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-UPD-001", legal_name="Old Name")
        )
        result = supplier_profile_manager.update("SUP-UPD-001", {"legal_name": "New Name"})
        assert result["legal_name"] == "New Name"

    def test_update_gps_coordinates(self, supplier_profile_manager):
        """Update GPS coordinates."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-UPD-GPS", gps_lat=0.0, gps_lon=0.0)
        )
        result = supplier_profile_manager.update(
            "SUP-UPD-GPS", {"gps_lat": 5.6037, "gps_lon": -0.1870}
        )
        assert result["gps_lat"] == pytest.approx(5.6037, abs=0.001)

    def test_update_nonexistent_raises(self, supplier_profile_manager):
        """Updating a non-existent profile raises an error."""
        with pytest.raises((ValueError, KeyError)):
            supplier_profile_manager.update("SUP-GHOST", {"legal_name": "Ghost"})

    def test_update_preserves_unmodified_fields(self, supplier_profile_manager):
        """Updating one field does not alter other fields."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-PRESERVE", legal_name="Preserve",
                          registration_id="REG-PRESERVE")
        )
        supplier_profile_manager.update("SUP-PRESERVE", {"legal_name": "Updated Name"})
        result = supplier_profile_manager.get("SUP-PRESERVE")
        assert result["registration_id"] == "REG-PRESERVE"

    def test_update_adds_certifications(self, supplier_profile_manager):
        """Add a certification to an existing profile."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-CERT-UPD", certifications=[])
        )
        result = supplier_profile_manager.update(
            "SUP-CERT-UPD", {"certifications": ["FSC-001"]}
        )
        assert "FSC-001" in result["certifications"]

    def test_update_increments_version(self, supplier_profile_manager):
        """Each update increments the profile version."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-VER", legal_name="Version Test")
        )
        result1 = supplier_profile_manager.update("SUP-VER", {"legal_name": "V2"})
        result2 = supplier_profile_manager.update("SUP-VER", {"legal_name": "V3"})
        assert result2.get("version", 0) > result1.get("version", 0)


# ===========================================================================
# 4. Delete / Deactivate Operations
# ===========================================================================


class TestSupplierProfileDeactivate:
    """Test supplier deactivation (soft delete)."""

    def test_deactivate_supplier(self, supplier_profile_manager):
        """Deactivate a supplier sets status to inactive."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-DEACT", status="active")
        )
        result = supplier_profile_manager.deactivate("SUP-DEACT")
        assert result["status"] in ("inactive", "deactivated", "terminated")

    def test_deactivate_nonexistent_raises(self, supplier_profile_manager):
        """Deactivating a non-existent supplier raises error."""
        with pytest.raises((ValueError, KeyError)):
            supplier_profile_manager.deactivate("SUP-NOPE")

    def test_deactivate_preserves_data(self, supplier_profile_manager):
        """Deactivated supplier data is still retrievable."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SOFT-DEL", legal_name="Soft Delete Corp")
        )
        supplier_profile_manager.deactivate("SUP-SOFT-DEL")
        result = supplier_profile_manager.get("SUP-SOFT-DEL")
        assert result is not None
        assert result["legal_name"] == "Soft Delete Corp"


# ===========================================================================
# 5. Profile Completeness Scoring
# ===========================================================================


class TestProfileCompletenessScoring:
    """Test completeness scoring (F2.9) with all weight categories."""

    def test_full_profile_score_100(self, supplier_profile_manager):
        """Fully populated profile scores close to 100."""
        supplier = make_supplier(
            legal_name="Complete Corp",
            registration_id="REG-001",
            country_iso="GH",
            gps_lat=5.6037,
            gps_lon=-0.1870,
            commodity="cocoa",
            certifications=["UTZ-001"],
            dds_references=["DDS-001"],
            primary_contact="Alice",
            compliance_contact="Bob",
            annual_volume_mt=1000.0,
        )
        score = supplier_profile_manager.calculate_completeness(supplier)
        assert_valid_completeness_score(score)
        assert score >= 90.0

    def test_empty_profile_score_low(self, supplier_profile_manager):
        """Profile with only legal_name scores low."""
        supplier = make_supplier(
            legal_name="Empty Corp",
            registration_id=None,
            tax_id=None,
            gps_lat=None,
            gps_lon=None,
            certifications=[],
            dds_references=[],
            primary_contact=None,
            compliance_contact=None,
        )
        score = supplier_profile_manager.calculate_completeness(supplier)
        assert_valid_completeness_score(score)
        assert score < 50.0

    @pytest.mark.parametrize("weight_category,weight", [
        ("legal_identity", 0.25),
        ("location", 0.20),
        ("commodity", 0.15),
        ("certification", 0.15),
        ("compliance", 0.15),
        ("contact", 0.10),
    ])
    def test_completeness_weights_match_prd(self, supplier_profile_manager,
                                             weight_category, weight):
        """Verify completeness weight categories match PRD Appendix D."""
        weights = supplier_profile_manager.get_completeness_weights()
        assert weight_category in weights
        assert weights[weight_category] == pytest.approx(weight, abs=0.01)

    def test_completeness_weights_sum_to_one(self, supplier_profile_manager):
        """All completeness weights must sum to 1.0."""
        weights = supplier_profile_manager.get_completeness_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_completeness_with_only_legal_identity(self, supplier_profile_manager):
        """Profile with only legal identity populated scores ~25%."""
        supplier = make_supplier(
            legal_name="Legal Only",
            registration_id="REG-001",
            country_iso="GH",
            gps_lat=None,
            gps_lon=None,
            certifications=[],
            dds_references=[],
            primary_contact=None,
            compliance_contact=None,
        )
        score = supplier_profile_manager.calculate_completeness(supplier)
        assert_valid_completeness_score(score)
        assert 15.0 <= score <= 35.0

    def test_completeness_with_location_only(self, supplier_profile_manager):
        """Profile with legal name + location scores ~45%."""
        supplier = make_supplier(
            legal_name="Location Corp",
            registration_id="REG-002",
            country_iso="GH",
            gps_lat=5.6037,
            gps_lon=-0.1870,
            certifications=[],
            dds_references=[],
            primary_contact=None,
            compliance_contact=None,
        )
        score = supplier_profile_manager.calculate_completeness(supplier)
        assert_valid_completeness_score(score)
        assert 30.0 <= score <= 55.0

    def test_completeness_missing_fields_identified(self, supplier_profile_manager):
        """Completeness result identifies which fields are missing."""
        supplier = make_supplier(
            legal_name="Missing Fields",
            registration_id=None,
            gps_lat=None,
            gps_lon=None,
        )
        details = supplier_profile_manager.get_completeness_details(supplier)
        assert "missing_fields" in details
        assert len(details["missing_fields"]) > 0

    def test_completeness_score_is_deterministic(self, supplier_profile_manager):
        """Same input always produces same completeness score."""
        supplier = make_supplier(legal_name="Deterministic", registration_id="DET-001")
        score1 = supplier_profile_manager.calculate_completeness(supplier)
        score2 = supplier_profile_manager.calculate_completeness(supplier)
        assert score1 == score2

    @pytest.mark.parametrize("tier", [0, 1, 2, 3, 4, 5])
    def test_completeness_across_tiers(self, supplier_profile_manager, tier):
        """Completeness scoring works for all tier levels."""
        supplier = make_supplier(legal_name=f"Tier {tier} Supplier", tier=tier)
        score = supplier_profile_manager.calculate_completeness(supplier)
        assert_valid_completeness_score(score)


# ===========================================================================
# 6. Profile Versioning and Merge
# ===========================================================================


class TestProfileVersioning:
    """Test profile versioning and change history (F2.10)."""

    def test_initial_version_is_one(self, supplier_profile_manager):
        """Newly created profile has version 1."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-V1", legal_name="Version One")
        )
        result = supplier_profile_manager.get("SUP-V1")
        assert result.get("version", 1) == 1

    def test_update_increments_version_number(self, supplier_profile_manager):
        """Each update increments version by 1."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-V-INC", legal_name="V Inc Test")
        )
        supplier_profile_manager.update("SUP-V-INC", {"legal_name": "V2"})
        result = supplier_profile_manager.get("SUP-V-INC")
        assert result.get("version", 0) >= 2

    def test_version_history_tracks_changes(self, supplier_profile_manager):
        """Version history contains all previous versions."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-VHIST", legal_name="Original")
        )
        supplier_profile_manager.update("SUP-VHIST", {"legal_name": "Updated 1"})
        supplier_profile_manager.update("SUP-VHIST", {"legal_name": "Updated 2"})
        history = supplier_profile_manager.get_version_history("SUP-VHIST")
        assert len(history) >= 3

    def test_merge_profiles(self, supplier_profile_manager):
        """Merge two partial profiles into a complete one."""
        profile_a = make_supplier(
            supplier_id="SUP-MERGE-A",
            legal_name="Merge Corp",
            registration_id="REG-MERGE",
            tax_id=None,
            gps_lat=5.6037,
            gps_lon=None,
        )
        profile_b = make_supplier(
            supplier_id="SUP-MERGE-B",
            legal_name="Merge Corp",
            registration_id="REG-MERGE",
            tax_id="TAX-MERGE",
            gps_lat=None,
            gps_lon=-0.1870,
        )
        result = supplier_profile_manager.merge_profiles(profile_a, profile_b)
        assert result["tax_id"] == "TAX-MERGE"
        assert result["gps_lat"] == pytest.approx(5.6037, abs=0.001)
        assert result["gps_lon"] == pytest.approx(-0.1870, abs=0.001)

    def test_merge_prefers_newer_data(self, supplier_profile_manager):
        """When both profiles have a field, prefer the newer value."""
        profile_old = make_supplier(
            supplier_id="SUP-MERGE-OLD",
            legal_name="Old Name",
            primary_contact="Old Contact",
        )
        profile_new = make_supplier(
            supplier_id="SUP-MERGE-NEW",
            legal_name="New Name",
            primary_contact="New Contact",
        )
        result = supplier_profile_manager.merge_profiles(
            profile_old, profile_new, prefer="newer"
        )
        assert result["primary_contact"] == "New Contact"


# ===========================================================================
# 7. Search
# ===========================================================================


class TestSupplierProfileSearch:
    """Test search across supplier profiles with various criteria."""

    def test_search_by_country(self, supplier_profile_manager):
        """Search suppliers by country code."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-GH", country_iso="GH", legal_name="Ghana Sup")
        )
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-CO", country_iso="CO", legal_name="Colombia Sup")
        )
        results = supplier_profile_manager.search(country_iso="GH")
        assert all(r["country_iso"] == "GH" for r in results)

    def test_search_by_commodity(self, supplier_profile_manager):
        """Search suppliers by commodity type."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-COC", commodity="cocoa", legal_name="Cocoa Sup")
        )
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-COF", commodity="coffee", legal_name="Coffee Sup")
        )
        results = supplier_profile_manager.search(commodity="cocoa")
        assert all("cocoa" in r["commodities"] for r in results)

    def test_search_by_name(self, supplier_profile_manager):
        """Search suppliers by partial name match."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-NM", legal_name="UniqueSearchableName Corp")
        )
        results = supplier_profile_manager.search(legal_name="UniqueSearchableName")
        assert len(results) >= 1
        assert results[0]["legal_name"] == "UniqueSearchableName Corp"

    def test_search_by_tier(self, supplier_profile_manager):
        """Search suppliers by tier level."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-T3", tier=3, legal_name="Tier 3 Sup")
        )
        results = supplier_profile_manager.search(tier=3)
        assert all(r["tier"] == 3 for r in results)

    def test_search_by_status(self, supplier_profile_manager):
        """Search suppliers by status."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-SRC-ACT", status="active", legal_name="Active")
        )
        results = supplier_profile_manager.search(status="active")
        assert all(r["status"] == "active" for r in results)

    def test_search_no_results(self, supplier_profile_manager):
        """Search with no matches returns empty list."""
        results = supplier_profile_manager.search(country_iso="ZZ")
        assert len(results) == 0

    def test_search_combined_criteria(self, supplier_profile_manager):
        """Search with multiple criteria narrows results."""
        supplier_profile_manager.create(
            make_supplier(supplier_id="SUP-COMBO", country_iso="GH",
                          commodity="cocoa", tier=2, legal_name="Combo")
        )
        results = supplier_profile_manager.search(country_iso="GH", commodity="cocoa", tier=2)
        assert len(results) >= 1


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestSupplierProfileEdgeCases:
    """Test edge cases in profile management."""

    def test_invalid_country_code_raises(self, supplier_profile_manager):
        """Invalid ISO country code raises ValueError."""
        supplier = make_supplier(legal_name="Invalid Country", country_iso="ZZZ")
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_empty_string_legal_name_raises(self, supplier_profile_manager):
        """Empty string legal name is rejected."""
        supplier = make_supplier(legal_name="")
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_whitespace_only_legal_name_raises(self, supplier_profile_manager):
        """Whitespace-only legal name is rejected."""
        supplier = make_supplier(legal_name="   ")
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_negative_tier_raises(self, supplier_profile_manager):
        """Negative tier value is rejected."""
        supplier = make_supplier(legal_name="Neg Tier", tier=-1)
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_gps_lat_out_of_range_raises(self, supplier_profile_manager):
        """GPS latitude > 90 or < -90 is rejected."""
        supplier = make_supplier(legal_name="Bad Lat", gps_lat=91.0, gps_lon=0.0)
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_gps_lon_out_of_range_raises(self, supplier_profile_manager):
        """GPS longitude > 180 or < -180 is rejected."""
        supplier = make_supplier(legal_name="Bad Lon", gps_lat=0.0, gps_lon=181.0)
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_null_gps_allowed(self, supplier_profile_manager):
        """GPS coordinates can be None (not yet known)."""
        supplier = make_supplier(legal_name="No GPS", gps_lat=None, gps_lon=None)
        result = supplier_profile_manager.create(supplier)
        assert result["gps_lat"] is None
        assert result["gps_lon"] is None

    def test_negative_volume_raises(self, supplier_profile_manager):
        """Negative annual volume is rejected."""
        supplier = make_supplier(legal_name="Neg Volume", annual_volume_mt=-100.0)
        with pytest.raises(ValueError):
            supplier_profile_manager.create(supplier)

    def test_very_long_legal_name(self, supplier_profile_manager):
        """Very long legal name (500+ chars) is handled."""
        long_name = "A" * 500
        supplier = make_supplier(legal_name=long_name)
        result = supplier_profile_manager.create(supplier)
        assert len(result["legal_name"]) > 0

    def test_unicode_legal_name(self, supplier_profile_manager):
        """Unicode characters in legal name are preserved."""
        supplier = make_supplier(legal_name="Caf\u00e9 Producci\u00f3n S.A.")
        result = supplier_profile_manager.create(supplier)
        assert result["legal_name"] == "Caf\u00e9 Producci\u00f3n S.A."

    @pytest.mark.parametrize("country_iso", [
        "GH", "CI", "CO", "BR", "ID", "MY", "TH", "CD", "DE", "NL", "FR",
    ])
    def test_valid_country_codes_accepted(self, supplier_profile_manager, country_iso):
        """All valid ISO 3166-1 alpha-2 codes are accepted."""
        supplier = make_supplier(
            legal_name=f"Country {country_iso}", country_iso=country_iso
        )
        result = supplier_profile_manager.create(supplier)
        assert result["country_iso"] == country_iso

    def test_multiple_commodities(self, supplier_profile_manager):
        """Supplier handling multiple commodities is supported."""
        supplier = make_supplier(legal_name="Multi Commodity")
        supplier["commodities"] = ["cocoa", "coffee", "palm_oil"]
        result = supplier_profile_manager.create(supplier)
        assert len(result["commodities"]) == 3
