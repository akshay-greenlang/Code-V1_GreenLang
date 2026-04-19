# -*- coding: utf-8 -*-
"""
Tests for PACK-049 Engine 1: SiteRegistryEngine

Covers site registration, update, decommission, classification, grouping,
portfolio summary, filtering, and completeness validation.
Target: ~55 tests.
"""

import pytest
from decimal import Decimal
from datetime import date, datetime, timezone
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.site_registry_engine import (
    SiteRegistryEngine,
    SiteRecord,
    SiteGroup,
    SiteRegistryResult,
    SiteClassification,
    SiteCompletenessResult,
    FacilityCharacteristics,
    FacilityType,
    LifecycleStatus,
    GroupType,
    CompletenessLevel,
    SIZE_CLASS_THRESHOLDS,
    HIGH_INTENSITY_FACILITY_TYPES,
    MEDIUM_INTENSITY_FACILITY_TYPES,
)


@pytest.fixture
def engine():
    return SiteRegistryEngine()


@pytest.fixture
def registered_site(engine, sample_site_record):
    return engine.register_site(sample_site_record)


@pytest.fixture
def five_sites(engine, sample_site_records):
    sites = []
    for data in sample_site_records:
        sites.append(engine.register_site(data))
    return sites


# ============================================================================
# Registration Tests
# ============================================================================

class TestSiteRegistration:

    def test_register_site_basic(self, engine, sample_site_record):
        site = engine.register_site(sample_site_record)
        assert isinstance(site, SiteRecord)
        assert site.site_code == "US-CHI-MFG-01"
        assert site.site_name == "Chicago Manufacturing Plant"
        assert site.is_active is True

    def test_register_site_with_characteristics(self, engine, sample_site_record):
        site = engine.register_site(sample_site_record)
        assert site.characteristics is not None
        assert site.characteristics.floor_area_m2 == Decimal("10000")
        assert site.characteristics.headcount == 500
        assert site.characteristics.operating_hours_per_year == 6000

    @pytest.mark.parametrize("facility_type", [
        "MANUFACTURING", "OFFICE", "WAREHOUSE", "DATA_CENTER", "RETAIL",
        "LABORATORY", "HOSPITAL", "DISTRIBUTION_CENTER", "REFINERY",
        "POWER_PLANT", "MINE", "AGRICULTURAL", "MIXED_USE", "OTHER",
    ])
    def test_register_site_all_facility_types(self, engine, facility_type):
        data = {
            "site_code": f"XX-{facility_type[:3]}-01",
            "site_name": f"Test {facility_type} Site",
            "facility_type": facility_type,
            "legal_entity_id": "LE-001",
            "country": "US",
            "characteristics": {
                "floor_area_m2": Decimal("5000"),
                "headcount": 100,
                "operating_hours_per_year": 4000,
            },
        }
        site = engine.register_site(data)
        assert site.facility_type == facility_type

    def test_register_site_generates_uuid(self, engine, sample_site_record):
        site = engine.register_site(sample_site_record)
        assert site.site_id is not None
        assert len(site.site_id) == 36  # UUID4 format

    def test_register_site_duplicate_code_raises(self, engine, sample_site_record):
        engine.register_site(sample_site_record)
        with pytest.raises(ValueError, match="already exists"):
            engine.register_site(sample_site_record)

    def test_register_site_timestamps(self, engine, sample_site_record):
        site = engine.register_site(sample_site_record)
        assert site.created_at is not None
        assert site.updated_at is not None
        assert site.created_at.tzinfo is not None


# ============================================================================
# Update Tests
# ============================================================================

class TestSiteUpdate:

    def test_update_site_name(self, engine, registered_site):
        updated = engine.update_site(registered_site.site_id, {"site_name": "New Name"})
        assert updated.site_name == "New Name"

    def test_update_site_lifecycle(self, engine, registered_site):
        updated = engine.update_site(
            registered_site.site_id,
            {"lifecycle_status": "MOTHBALLED"},
        )
        assert updated.lifecycle_status == "MOTHBALLED"

    def test_update_site_not_found_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.update_site("nonexistent-id", {"site_name": "X"})

    def test_update_site_immutable_field_ignored(self, engine, registered_site):
        original_id = registered_site.site_id
        updated = engine.update_site(original_id, {"site_id": "new-id"})
        assert updated.site_id == original_id

    def test_update_site_characteristics_partial(self, engine, registered_site):
        updated = engine.update_site(
            registered_site.site_id,
            {"characteristics": {"headcount": 600}},
        )
        assert updated.characteristics.headcount == 600
        assert updated.characteristics.floor_area_m2 == Decimal("10000")


# ============================================================================
# Decommission Tests
# ============================================================================

class TestSiteDecommission:

    def test_decommission_site(self, engine, registered_site):
        result = engine.decommission_site(
            registered_site.site_id,
            decommission_date=date(2026, 12, 31),
            reason="Facility closed",
        )
        assert result.lifecycle_status == LifecycleStatus.DECOMMISSIONED.value
        assert result.is_active is False

    def test_decommission_sets_inactive(self, engine, registered_site):
        result = engine.decommission_site(
            registered_site.site_id,
            decommission_date=date(2026, 6, 30),
            reason="Sold to third party",
        )
        assert result.is_active is False
        assert result.decommissioning_date == date(2026, 6, 30)

    def test_decommission_already_decommissioned_raises(self, engine, registered_site):
        engine.decommission_site(
            registered_site.site_id,
            decommission_date=date(2026, 12, 31),
            reason="Closed",
        )
        with pytest.raises(ValueError, match="already decommissioned"):
            engine.decommission_site(
                registered_site.site_id,
                decommission_date=date(2026, 12, 31),
                reason="Closed again",
            )

    def test_decommission_not_found_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.decommission_site("bad-id", date(2026, 1, 1), "reason")


# ============================================================================
# Classification Tests
# ============================================================================

class TestSiteClassification:

    def test_classify_site_by_type(self, engine, registered_site):
        result = engine.classify_site(registered_site.site_id)
        assert isinstance(result, SiteClassification)
        assert result.primary_category == "MANUFACTURING"
        assert result.emission_intensity_class == "HIGH"

    def test_classify_site_by_geography(self, engine, sample_site_records):
        site = engine.register_site(sample_site_records[1])  # London Office
        result = engine.classify_site(site.site_id)
        assert result.primary_category == "OFFICE"
        assert result.emission_intensity_class == "LOW"

    def test_classify_site_size_class_large(self, engine, registered_site):
        result = engine.classify_site(registered_site.site_id)
        # floor_area=10000 falls in LARGE (10000, 100000)
        assert result.size_class == "LARGE"

    def test_classify_site_size_class_small(self, engine):
        data = {
            "site_code": "SMALL-01",
            "site_name": "Small Site",
            "facility_type": "RETAIL",
            "legal_entity_id": "LE-001",
            "country": "US",
            "characteristics": {
                "floor_area_m2": Decimal("500"),
                "headcount": 10,
                "operating_hours_per_year": 2500,
            },
        }
        site = engine.register_site(data)
        result = engine.classify_site(site.site_id)
        assert result.size_class == "SMALL"

    def test_classify_site_provenance_hash(self, engine, registered_site):
        result = engine.classify_site(registered_site.site_id)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_classify_site_not_found_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.classify_site("nonexistent-id")


# ============================================================================
# Group Tests
# ============================================================================

class TestSiteGroups:

    def test_create_site_group(self, engine, five_sites):
        site_ids = [s.site_id for s in five_sites[:3]]
        group = engine.create_site_group(
            group_name="North America",
            group_type="REGION",
            site_ids=site_ids,
        )
        assert isinstance(group, SiteGroup)
        assert group.group_name == "North America"
        assert len(group.member_site_ids) == 3

    def test_create_site_group_custom(self, engine, five_sites):
        site_ids = [five_sites[0].site_id]
        group = engine.create_site_group(
            group_name="Priority Sites",
            group_type="CUSTOM",
            site_ids=site_ids,
            description="High priority sites for audit",
        )
        assert group.group_type == "CUSTOM"
        assert group.description == "High priority sites for audit"

    def test_create_group_invalid_site_raises(self, engine, five_sites):
        with pytest.raises(ValueError, match="not found"):
            engine.create_site_group(
                group_name="Bad Group",
                group_type="REGION",
                site_ids=["nonexistent-id"],
            )

    def test_create_duplicate_group_raises(self, engine, five_sites):
        site_ids = [five_sites[0].site_id]
        engine.create_site_group("Group A", "REGION", site_ids)
        with pytest.raises(ValueError, match="already exists"):
            engine.create_site_group("Group A", "REGION", site_ids)


# ============================================================================
# Portfolio Summary Tests
# ============================================================================

class TestPortfolioSummary:

    def test_get_portfolio_summary_basic(self, engine, five_sites):
        summary = engine.get_portfolio_summary()
        assert isinstance(summary, SiteRegistryResult)
        assert summary.total_active_sites == 5

    def test_portfolio_summary_total_floor_area(self, engine, five_sites):
        summary = engine.get_portfolio_summary()
        # 25000 + 3000 + 8000 + 1500 + 5000 = 42500
        assert summary.total_floor_area == Decimal("42500.00")

    def test_portfolio_summary_countries(self, engine, five_sites):
        summary = engine.get_portfolio_summary()
        # US: site-001, site-004; GB: site-002; DE: site-003, site-005
        assert summary.countries_covered == 3

    def test_facility_type_distribution(self, engine, five_sites):
        summary = engine.get_portfolio_summary()
        dist = summary.facility_type_distribution
        assert dist.get("MANUFACTURING", 0) == 1
        assert dist.get("OFFICE", 0) == 1
        assert dist.get("WAREHOUSE", 0) == 1
        assert dist.get("RETAIL", 0) == 1
        assert dist.get("DATA_CENTER", 0) == 1

    def test_geographic_distribution(self, engine, five_sites):
        summary = engine.get_portfolio_summary()
        geo = summary.geographic_distribution
        assert geo.get("US", 0) == 2
        assert geo.get("GB", 0) == 1
        assert geo.get("DE", 0) == 2

    def test_empty_portfolio(self, engine):
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 0
        assert summary.total_floor_area == Decimal("0")

    def test_single_site_portfolio(self, engine, registered_site):
        summary = engine.get_portfolio_summary()
        assert summary.total_active_sites == 1

    def test_provenance_hash_valid(self, engine, five_sites):
        s1 = engine.get_portfolio_summary()
        assert s1.provenance_hash is not None
        assert len(s1.provenance_hash) == 64

    def test_provenance_hash_changes_on_update(self, engine, five_sites):
        s1 = engine.get_portfolio_summary()
        engine.update_site(five_sites[0].site_id, {"site_name": "Changed Name"})
        s2 = engine.get_portfolio_summary()
        assert s1.provenance_hash != s2.provenance_hash


# ============================================================================
# Filter Tests
# ============================================================================

class TestSiteFilter:

    def test_get_sites_by_filter_country(self, engine, five_sites):
        us_sites = engine.get_sites_by_filter(filters={"country": "US"})
        assert len(us_sites) == 2

    def test_get_sites_by_filter_type(self, engine, five_sites):
        mfg_sites = engine.get_sites_by_filter(
            filters={"facility_type": "MANUFACTURING"},
        )
        assert len(mfg_sites) == 1

    def test_get_sites_by_filter_active(self, engine, five_sites):
        engine.decommission_site(
            five_sites[0].site_id, date(2026, 12, 31), "Closed",
        )
        active = engine.get_sites_by_filter(filters={"is_active": True})
        assert len(active) == 4

    def test_get_sites_by_filter_no_filter(self, engine, five_sites):
        all_sites = engine.get_sites_by_filter()
        assert len(all_sites) == 5


# ============================================================================
# Completeness Validation Tests
# ============================================================================

class TestSiteCompleteness:

    def test_validate_completeness_complete(self, engine, registered_site):
        result = engine.validate_site_completeness(registered_site)
        assert isinstance(result, SiteCompletenessResult)
        assert result.completeness_pct >= Decimal("80")

    def test_validate_completeness_missing_characteristics(self, engine):
        data = {
            "site_code": "BARE-01",
            "site_name": "Bare Site",
            "facility_type": "OTHER",
            "legal_entity_id": "LE-001",
            "country": "US",
        }
        site = engine.register_site(data)
        result = engine.validate_site_completeness(site)
        assert "characteristics" in result.missing_optional
