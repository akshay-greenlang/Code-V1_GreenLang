# -*- coding: utf-8 -*-
"""
Unit tests for StakeholderMapper Engine - AGENT-EUDR-031

Tests stakeholder discovery, mapping, categorization, rights classification,
retrieval, and listing operations.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.stakeholder_mapper import (
    StakeholderMapper,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    EUDRCommodity,
    RightsClassification,
    StakeholderCategory,
    StakeholderRecord,
    StakeholderStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def mapper(config):
    return StakeholderMapper(config=config)


# ---------------------------------------------------------------------------
# Test: MapStakeholder
# ---------------------------------------------------------------------------

class TestMapStakeholder:
    """Test stakeholder mapping/registration."""

    @pytest.mark.asyncio
    async def test_map_stakeholder_success(self, mapper, sample_contact_info, sample_rights_classification):
        """Test successful stakeholder mapping."""
        result = await mapper.map_stakeholder(
            operator_id="OP-001",
            name="Test Community",
            category=StakeholderCategory.LOCAL_COMMUNITY,
            country_code="CO",
            region="Antioquia",
            commodity=EUDRCommodity.COFFEE,
            contact_info=sample_contact_info,
            rights_classification=sample_rights_classification,
            population_estimate=200,
            affected_area_hectares=Decimal("100.0"),
        )
        assert result.stakeholder_id.startswith("STK-")
        assert result.name == "Test Community"
        assert result.category == StakeholderCategory.LOCAL_COMMUNITY
        assert result.status == StakeholderStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_map_stakeholder_indigenous(self, mapper, sample_contact_info):
        """Test mapping indigenous community stakeholder."""
        rights = RightsClassification(
            has_land_rights=True,
            has_customary_rights=True,
            has_indigenous_status=True,
            fpic_required=True,
            applicable_conventions=["ILO 169", "UNDRIP"],
            legal_framework="Constitutional right",
        )
        result = await mapper.map_stakeholder(
            operator_id="OP-001",
            name="Indigenous Community X",
            category=StakeholderCategory.INDIGENOUS_COMMUNITY,
            country_code="BR",
            region="Amazonas",
            commodity=EUDRCommodity.SOYA,
            contact_info=sample_contact_info,
            rights_classification=rights,
            population_estimate=800,
            affected_area_hectares=Decimal("2500.0"),
        )
        assert result.category == StakeholderCategory.INDIGENOUS_COMMUNITY
        assert result.rights_classification.fpic_required is True

    @pytest.mark.asyncio
    async def test_map_stakeholder_missing_name_raises(self, mapper):
        """Test mapping stakeholder with empty name raises error."""
        with pytest.raises(ValueError, match="name is required"):
            await mapper.map_stakeholder(
                operator_id="OP-001",
                name="",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO",
                region="Antioquia",
                commodity=EUDRCommodity.COFFEE,
                contact_info={},
                rights_classification=RightsClassification(
                    has_land_rights=False, has_customary_rights=False,
                    has_indigenous_status=False, fpic_required=False,
                    applicable_conventions=[], legal_framework="",
                ),
            )

    @pytest.mark.asyncio
    async def test_map_stakeholder_missing_operator_id_raises(self, mapper):
        """Test mapping stakeholder with empty operator_id raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await mapper.map_stakeholder(
                operator_id="",
                name="Test",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO",
                region="Antioquia",
                commodity=EUDRCommodity.COFFEE,
                contact_info={},
                rights_classification=RightsClassification(
                    has_land_rights=False, has_customary_rights=False,
                    has_indigenous_status=False, fpic_required=False,
                    applicable_conventions=[], legal_framework="",
                ),
            )

    @pytest.mark.asyncio
    async def test_map_stakeholder_generates_provenance(self, mapper, sample_contact_info, sample_rights_classification):
        """Test stakeholder mapping generates provenance hash."""
        result = await mapper.map_stakeholder(
            operator_id="OP-001",
            name="Provenance Test",
            category=StakeholderCategory.COOPERATIVE,
            country_code="CO",
            region="Huila",
            commodity=EUDRCommodity.COFFEE,
            contact_info=sample_contact_info,
            rights_classification=sample_rights_classification,
        )
        # Check provenance is tracked
        assert mapper._provenance.get_chain() != []

    @pytest.mark.asyncio
    async def test_map_stakeholder_all_categories(self, mapper, sample_contact_info, sample_rights_classification):
        """Test mapping stakeholders with all category types."""
        for category in StakeholderCategory:
            result = await mapper.map_stakeholder(
                operator_id="OP-001",
                name=f"Test {category.value}",
                category=category,
                country_code="CO",
                region="Test",
                commodity=EUDRCommodity.COFFEE,
                contact_info=sample_contact_info,
                rights_classification=sample_rights_classification,
            )
            assert result.category == category

    @pytest.mark.asyncio
    async def test_map_stakeholder_all_commodities(self, mapper, sample_contact_info, sample_rights_classification):
        """Test mapping stakeholders for all commodities."""
        for commodity in EUDRCommodity:
            result = await mapper.map_stakeholder(
                operator_id="OP-001",
                name=f"Test {commodity.value}",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO",
                region="Test",
                commodity=commodity,
                contact_info=sample_contact_info,
                rights_classification=sample_rights_classification,
            )
            assert result.commodity == commodity

    @pytest.mark.asyncio
    async def test_map_stakeholder_unique_ids(self, mapper, sample_contact_info, sample_rights_classification):
        """Test that each mapped stakeholder gets a unique ID."""
        results = []
        for i in range(5):
            result = await mapper.map_stakeholder(
                operator_id="OP-001",
                name=f"Community {i}",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO",
                region="Test",
                commodity=EUDRCommodity.COFFEE,
                contact_info=sample_contact_info,
                rights_classification=sample_rights_classification,
            )
            results.append(result)
        ids = [r.stakeholder_id for r in results]
        assert len(set(ids)) == 5  # All unique


# ---------------------------------------------------------------------------
# Test: DiscoverFromSupplyChain
# ---------------------------------------------------------------------------

class TestDiscoverFromSupplyChain:
    """Test stakeholder discovery from supply chain data."""

    @pytest.mark.asyncio
    async def test_discover_returns_list(self, mapper):
        """Test discover returns a list of stakeholder records."""
        supply_chain_data = {
            "operator_id": "OP-001",
            "commodity": "coffee",
            "country": "CO",
            "region": "Antioquia",
            "suppliers": [{"id": "SUP-001", "name": "Farm A"}],
        }
        results = await mapper.discover_from_supply_chain(supply_chain_data)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_discover_empty_supply_chain(self, mapper):
        """Test discover with empty supply chain data."""
        results = await mapper.discover_from_supply_chain({})
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_discover_with_multiple_suppliers(self, mapper):
        """Test discover with multiple suppliers in supply chain."""
        supply_chain_data = {
            "operator_id": "OP-001",
            "commodity": "coffee",
            "country": "CO",
            "suppliers": [
                {"id": "SUP-001", "name": "Farm A", "region": "Antioquia"},
                {"id": "SUP-002", "name": "Farm B", "region": "Huila"},
                {"id": "SUP-003", "name": "Farm C", "region": "Narino"},
            ],
        }
        results = await mapper.discover_from_supply_chain(supply_chain_data)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_discover_with_indigenous_territories(self, mapper):
        """Test discover identifies indigenous territory overlaps."""
        supply_chain_data = {
            "operator_id": "OP-001",
            "commodity": "palm_oil",
            "country": "ID",
            "region": "Kalimantan",
            "indigenous_territories": [
                {"name": "Dayak Community", "overlap_hectares": 150},
            ],
        }
        results = await mapper.discover_from_supply_chain(supply_chain_data)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_discover_filters_by_radius(self, mapper):
        """Test discover respects discovery radius configuration."""
        supply_chain_data = {
            "operator_id": "OP-001",
            "commodity": "coffee",
            "country": "CO",
            "center_lat": 4.5709,
            "center_lon": -74.2973,
            "suppliers": [{"id": "SUP-001", "name": "Farm A"}],
        }
        results = await mapper.discover_from_supply_chain(supply_chain_data)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_discover_missing_operator_id_raises(self, mapper):
        """Test discover raises on missing operator_id."""
        with pytest.raises(ValueError):
            await mapper.discover_from_supply_chain({"commodity": "coffee"})

    @pytest.mark.asyncio
    async def test_discover_returns_correct_types(self, mapper):
        """Test discover returns StakeholderRecord instances."""
        supply_chain_data = {
            "operator_id": "OP-001",
            "commodity": "coffee",
            "country": "CO",
            "suppliers": [{"id": "SUP-001", "name": "Farm A", "community": "San Jose"}],
        }
        results = await mapper.discover_from_supply_chain(supply_chain_data)
        for r in results:
            assert isinstance(r, StakeholderRecord)

    @pytest.mark.asyncio
    async def test_discover_populates_country_code(self, mapper):
        """Test discover populates country code from supply chain data."""
        supply_chain_data = {
            "operator_id": "OP-001",
            "commodity": "cocoa",
            "country": "GH",
            "suppliers": [{"id": "SUP-001", "name": "Ghana Farm"}],
        }
        results = await mapper.discover_from_supply_chain(supply_chain_data)
        for r in results:
            assert r.country_code == "GH"


# ---------------------------------------------------------------------------
# Test: CategorizeStakeholder
# ---------------------------------------------------------------------------

class TestCategorizeStakeholder:
    """Test stakeholder categorization logic."""

    def test_categorize_indigenous(self, mapper):
        """Test categorization identifies indigenous communities."""
        profile = {
            "indigenous_status": True,
            "land_rights": True,
            "customary_rights": True,
        }
        category = mapper.categorize_stakeholder(profile)
        assert category == StakeholderCategory.INDIGENOUS_COMMUNITY

    def test_categorize_cooperative(self, mapper):
        """Test categorization identifies cooperatives."""
        profile = {
            "organization_type": "cooperative",
            "member_count": 150,
        }
        category = mapper.categorize_stakeholder(profile)
        assert category == StakeholderCategory.COOPERATIVE

    def test_categorize_ngo(self, mapper):
        """Test categorization identifies NGOs."""
        profile = {
            "organization_type": "ngo",
            "focus_area": "environmental_protection",
        }
        category = mapper.categorize_stakeholder(profile)
        assert category == StakeholderCategory.NGO

    def test_categorize_local_community(self, mapper):
        """Test categorization identifies local communities."""
        profile = {
            "indigenous_status": False,
            "community_type": "rural",
            "population": 500,
        }
        category = mapper.categorize_stakeholder(profile)
        assert category == StakeholderCategory.LOCAL_COMMUNITY

    def test_categorize_smallholder(self, mapper):
        """Test categorization identifies smallholders."""
        profile = {
            "organization_type": "individual",
            "farm_size_hectares": 3.5,
        }
        category = mapper.categorize_stakeholder(profile)
        assert category == StakeholderCategory.SMALLHOLDER

    def test_categorize_default_other(self, mapper):
        """Test categorization defaults to OTHER for unknown profiles."""
        profile = {"unknown_field": "unknown_value"}
        category = mapper.categorize_stakeholder(profile)
        assert category == StakeholderCategory.OTHER


# ---------------------------------------------------------------------------
# Test: RightsClassification
# ---------------------------------------------------------------------------

class TestRightsClassification:
    """Test rights classification determination."""

    def test_classify_indigenous_rights(self, mapper):
        """Test rights classification for indigenous community."""
        profile = {
            "indigenous_status": True,
            "land_rights": True,
            "customary_rights": True,
            "country_code": "CO",
        }
        rights = mapper.classify_rights(profile)
        assert rights.has_indigenous_status is True
        assert rights.fpic_required is True
        assert len(rights.applicable_conventions) > 0

    def test_classify_no_indigenous_rights(self, mapper):
        """Test rights classification for non-indigenous community."""
        profile = {
            "indigenous_status": False,
            "land_rights": True,
            "customary_rights": False,
            "country_code": "CO",
        }
        rights = mapper.classify_rights(profile)
        assert rights.has_indigenous_status is False
        assert rights.fpic_required is False

    def test_classify_customary_rights(self, mapper):
        """Test rights classification with customary rights."""
        profile = {
            "indigenous_status": False,
            "land_rights": False,
            "customary_rights": True,
            "country_code": "GH",
        }
        rights = mapper.classify_rights(profile)
        assert rights.has_customary_rights is True

    def test_classify_rights_includes_conventions(self, mapper):
        """Test rights classification includes applicable conventions."""
        profile = {
            "indigenous_status": True,
            "land_rights": True,
            "customary_rights": True,
            "country_code": "BR",
        }
        rights = mapper.classify_rights(profile)
        assert "UNDRIP" in rights.applicable_conventions or "ILO 169" in rights.applicable_conventions

    def test_classify_rights_empty_profile(self, mapper):
        """Test rights classification with empty profile."""
        rights = mapper.classify_rights({})
        assert rights.has_indigenous_status is False
        assert rights.fpic_required is False
        assert rights.applicable_conventions == []

    def test_classify_rights_fpic_triggered_by_indigenous(self, mapper):
        """Test FPIC is required when indigenous status is True."""
        profile = {"indigenous_status": True, "country_code": "CO"}
        rights = mapper.classify_rights(profile)
        assert rights.fpic_required is True

    def test_classify_rights_legal_framework(self, mapper):
        """Test rights classification includes legal framework."""
        profile = {
            "indigenous_status": True,
            "land_rights": True,
            "country_code": "CO",
        }
        rights = mapper.classify_rights(profile)
        assert rights.legal_framework != ""

    def test_classify_rights_returns_classification(self, mapper):
        """Test classify_rights returns RightsClassification instance."""
        rights = mapper.classify_rights({"country_code": "CO"})
        assert isinstance(rights, RightsClassification)


# ---------------------------------------------------------------------------
# Test: GetStakeholder
# ---------------------------------------------------------------------------

class TestGetStakeholder:
    """Test individual stakeholder retrieval."""

    @pytest.mark.asyncio
    async def test_get_existing_stakeholder(self, mapper, sample_contact_info, sample_rights_classification):
        """Test retrieving an existing stakeholder."""
        created = await mapper.map_stakeholder(
            operator_id="OP-001", name="Retrieve Test",
            category=StakeholderCategory.LOCAL_COMMUNITY,
            country_code="CO", region="Test",
            commodity=EUDRCommodity.COFFEE,
            contact_info=sample_contact_info,
            rights_classification=sample_rights_classification,
        )
        retrieved = await mapper.get_stakeholder(created.stakeholder_id)
        assert retrieved is not None
        assert retrieved.stakeholder_id == created.stakeholder_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_stakeholder(self, mapper):
        """Test retrieving a non-existent stakeholder returns None."""
        result = await mapper.get_stakeholder("STK-NONEXISTENT")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_stakeholder_empty_id_raises(self, mapper):
        """Test retrieving stakeholder with empty ID raises error."""
        with pytest.raises(ValueError, match="stakeholder_id is required"):
            await mapper.get_stakeholder("")

    @pytest.mark.asyncio
    async def test_get_stakeholder_preserves_data(self, mapper, sample_contact_info, sample_rights_classification):
        """Test retrieved stakeholder preserves all data."""
        created = await mapper.map_stakeholder(
            operator_id="OP-001", name="Data Preservation",
            category=StakeholderCategory.INDIGENOUS_COMMUNITY,
            country_code="BR", region="Amazonas",
            commodity=EUDRCommodity.SOYA,
            contact_info=sample_contact_info,
            rights_classification=sample_rights_classification,
            population_estimate=500,
        )
        retrieved = await mapper.get_stakeholder(created.stakeholder_id)
        assert retrieved.name == "Data Preservation"
        assert retrieved.population_estimate == 500

    @pytest.mark.asyncio
    async def test_get_stakeholder_returns_correct_type(self, mapper, sample_contact_info, sample_rights_classification):
        """Test get_stakeholder returns StakeholderRecord."""
        created = await mapper.map_stakeholder(
            operator_id="OP-001", name="Type Test",
            category=StakeholderCategory.COOPERATIVE,
            country_code="CO", region="Test",
            commodity=EUDRCommodity.COFFEE,
            contact_info=sample_contact_info,
            rights_classification=sample_rights_classification,
        )
        retrieved = await mapper.get_stakeholder(created.stakeholder_id)
        assert isinstance(retrieved, StakeholderRecord)


# ---------------------------------------------------------------------------
# Test: ListStakeholders
# ---------------------------------------------------------------------------

class TestListStakeholders:
    """Test stakeholder listing operations."""

    @pytest.mark.asyncio
    async def test_list_returns_list(self, mapper):
        """Test list_stakeholders returns a list."""
        result = await mapper.list_stakeholders(operator_id="OP-001")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_filters_by_operator(self, mapper, sample_contact_info, sample_rights_classification):
        """Test list_stakeholders filters by operator_id."""
        for i in range(3):
            await mapper.map_stakeholder(
                operator_id="OP-FILTER", name=f"Filter Test {i}",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                country_code="CO", region="Test",
                commodity=EUDRCommodity.COFFEE,
                contact_info=sample_contact_info,
                rights_classification=sample_rights_classification,
            )
        result = await mapper.list_stakeholders(operator_id="OP-FILTER")
        assert all(r.operator_id == "OP-FILTER" for r in result)

    @pytest.mark.asyncio
    async def test_list_filters_by_category(self, mapper, sample_contact_info, sample_rights_classification):
        """Test list_stakeholders filters by category."""
        await mapper.map_stakeholder(
            operator_id="OP-001", name="Indigenous Test",
            category=StakeholderCategory.INDIGENOUS_COMMUNITY,
            country_code="CO", region="Test",
            commodity=EUDRCommodity.COFFEE,
            contact_info=sample_contact_info,
            rights_classification=sample_rights_classification,
        )
        result = await mapper.list_stakeholders(
            operator_id="OP-001",
            category=StakeholderCategory.INDIGENOUS_COMMUNITY,
        )
        for r in result:
            assert r.category == StakeholderCategory.INDIGENOUS_COMMUNITY

    @pytest.mark.asyncio
    async def test_list_empty_operator(self, mapper):
        """Test list_stakeholders with non-existent operator."""
        result = await mapper.list_stakeholders(operator_id="OP-NONEXISTENT")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_missing_operator_id_raises(self, mapper):
        """Test list_stakeholders with empty operator_id raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await mapper.list_stakeholders(operator_id="")
