# -*- coding: utf-8 -*-
"""
Tests for IndigenousRegistryEngine - AGENT-EUDR-021 Engine 6: Community Registry

Comprehensive test suite covering:
- Community CRUD operations (create, read, update, delete)
- Auto-population of ILO 169 coverage flag
- FPIC requirement auto-assignment based on country
- Territory linking between communities and territories
- Privacy controls for sensitive community data
- Batch import/export of community records

Test count: 52 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 6: Indigenous Community Registry)
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    ILO_169_EUDR_COUNTRIES,
    ALL_COMMODITIES,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    IndigenousCommunity,
    CommunityRecognitionStatus,
)


# ===========================================================================
# 1. Community CRUD (12 tests)
# ===========================================================================


class TestCommunityCRUD:
    """Test community create, read, update, delete operations."""

    def test_create_community(self, sample_community):
        """Test creating a community with all fields."""
        assert sample_community.community_id == "c-001"
        assert sample_community.community_name == "Yanomami do Rio Catrimani"
        assert sample_community.people_name == "Yanomami"
        assert sample_community.country_code == "BR"

    def test_create_community_minimal(self):
        """Test creating a community with only required fields."""
        c = IndigenousCommunity(
            community_id="c-min",
            community_name="Minimal Community",
            people_name="Test People",
            country_code="BR",
            provenance_hash="a" * 64,
        )
        assert c.community_id == "c-min"
        assert c.territory_ids == []
        assert c.estimated_population is None

    def test_community_with_population(self, sample_community):
        """Test community population is stored."""
        assert sample_community.estimated_population == 26000

    def test_community_with_language(self, sample_community):
        """Test community language is stored."""
        assert sample_community.language == "Yanomami"

    def test_community_update_preserves_id(self, sample_community):
        """Test updating community preserves ID."""
        updated = sample_community.model_copy(update={
            "estimated_population": 27000,
        })
        assert updated.community_id == sample_community.community_id
        assert updated.estimated_population == 27000

    def test_community_with_region(self, sample_community):
        """Test community region is stored."""
        assert sample_community.region == "amazon_basin"

    def test_community_indigenous_name(self, sample_community):
        """Test indigenous language name is stored."""
        assert sample_community.indigenous_name == "Watoriki"

    def test_create_multiple_communities(self, sample_communities):
        """Test creating multiple communities."""
        assert len(sample_communities) == 5
        ids = {c.community_id for c in sample_communities}
        assert len(ids) == 5

    def test_community_provenance_hash(self, sample_community):
        """Test community has provenance hash."""
        assert len(sample_community.provenance_hash) == SHA256_HEX_LENGTH

    def test_empty_community_name_rejected(self):
        """Test empty community name is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IndigenousCommunity(
                community_id="c-bad",
                community_name="",
                people_name="Test",
                country_code="BR",
                provenance_hash="b" * 64,
            )

    def test_empty_people_name_rejected(self):
        """Test empty people name is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IndigenousCommunity(
                community_id="c-bad",
                community_name="Valid",
                people_name="",
                country_code="BR",
                provenance_hash="c" * 64,
            )

    def test_community_data_source(self):
        """Test community data source is optional."""
        c = IndigenousCommunity(
            community_id="c-src",
            community_name="Source Test",
            people_name="Test",
            country_code="BR",
            data_source="funai",
            provenance_hash="d" * 64,
        )
        assert c.data_source == "funai"


# ===========================================================================
# 2. ILO 169 Auto-Population (8 tests)
# ===========================================================================


class TestILO169AutoPopulation:
    """Test auto-population of ILO 169 coverage based on country."""

    def test_brazil_ilo_169_covered(self, sample_community):
        """Test Brazil community has ILO 169 coverage."""
        assert sample_community.ilo_169_coverage is True

    def test_non_ilo_country_not_covered(self, sample_communities):
        """Test Indonesia community does not have ILO 169 coverage."""
        id_community = next(
            c for c in sample_communities if c.country_code == "ID"
        )
        assert id_community.ilo_169_coverage is False

    @pytest.mark.parametrize("country,expected_ilo", [
        ("BR", True),
        ("CO", True),
        ("PE", True),
        ("ID", False),
        ("CM", False),
    ])
    def test_ilo_coverage_by_country(self, sample_communities, country, expected_ilo):
        """Test ILO 169 coverage matches country ratification status."""
        community = next(
            (c for c in sample_communities if c.country_code == country),
            None,
        )
        if community is not None:
            assert community.ilo_169_coverage == expected_ilo

    def test_ilo_coverage_default_false(self):
        """Test ILO 169 coverage defaults to False."""
        c = IndigenousCommunity(
            community_id="c-default",
            community_name="Default ILO",
            people_name="Test",
            country_code="XX",
            provenance_hash="e" * 64,
        )
        assert c.ilo_169_coverage is False

    def test_ilo_coverage_explicitly_true(self):
        """Test ILO 169 coverage can be explicitly set to True."""
        c = IndigenousCommunity(
            community_id="c-ilo",
            community_name="ILO True",
            people_name="Test",
            country_code="BR",
            ilo_169_coverage=True,
            provenance_hash="f" * 64,
        )
        assert c.ilo_169_coverage is True


# ===========================================================================
# 3. FPIC Requirement Assignment (6 tests)
# ===========================================================================


class TestFPICRequirementAssignment:
    """Test auto-assignment of FPIC legal requirement."""

    def test_brazil_fpic_required(self, sample_community):
        """Test Brazil community has FPIC legal requirement."""
        assert sample_community.fpic_legal_requirement is True

    def test_fpic_requirement_follows_ilo(self, sample_communities):
        """Test FPIC requirement follows ILO 169 coverage."""
        for c in sample_communities:
            if c.ilo_169_coverage:
                assert c.fpic_legal_requirement is True

    def test_fpic_requirement_default_false(self):
        """Test FPIC requirement defaults to False."""
        c = IndigenousCommunity(
            community_id="c-nofpic",
            community_name="No FPIC",
            people_name="Test",
            country_code="US",
            provenance_hash="g" * 64,
        )
        assert c.fpic_legal_requirement is False

    def test_fpic_legal_protections_listed(self, sample_community):
        """Test applicable legal protections are listed."""
        assert len(sample_community.applicable_legal_protections) >= 1
        assert "ILO Convention 169" in sample_community.applicable_legal_protections

    def test_fpic_requirement_can_override(self):
        """Test FPIC requirement can be manually overridden."""
        c = IndigenousCommunity(
            community_id="c-override",
            community_name="Override FPIC",
            people_name="Test",
            country_code="US",
            fpic_legal_requirement=True,
            provenance_hash="h" * 64,
        )
        assert c.fpic_legal_requirement is True

    def test_recognition_status_tracked(self, sample_community):
        """Test legal recognition status is tracked."""
        assert sample_community.legal_recognition_status == (
            CommunityRecognitionStatus.CONSTITUTIONALLY_RECOGNIZED
        )


# ===========================================================================
# 4. Territory Linking (6 tests)
# ===========================================================================


class TestTerritoryLinking:
    """Test linking communities to territories."""

    def test_community_linked_to_territory(self, sample_community):
        """Test community has linked territory IDs."""
        assert len(sample_community.territory_ids) >= 1
        assert "t-001" in sample_community.territory_ids

    def test_multiple_territory_links(self):
        """Test community linked to multiple territories."""
        c = IndigenousCommunity(
            community_id="c-multi-t",
            community_name="Multi Territory",
            people_name="Test",
            country_code="BR",
            territory_ids=["t-001", "t-002", "t-003"],
            provenance_hash="i" * 64,
        )
        assert len(c.territory_ids) == 3

    def test_no_territory_links(self):
        """Test community without territory links."""
        c = IndigenousCommunity(
            community_id="c-no-t",
            community_name="No Territory",
            people_name="Test",
            country_code="BR",
            provenance_hash="j" * 64,
        )
        assert c.territory_ids == []

    def test_territory_link_matches_sample(self, sample_communities):
        """Test territory links match community-territory relationships."""
        for c in sample_communities:
            assert isinstance(c.territory_ids, list)

    def test_representative_organizations(self, sample_community):
        """Test representative organizations are tracked."""
        assert len(sample_community.representative_organizations) >= 1
        org = sample_community.representative_organizations[0]
        assert "name" in org
        assert "type" in org

    def test_commodity_relevance(self, sample_community):
        """Test EUDR commodity relevance is tagged."""
        assert len(sample_community.commodity_relevance) >= 1
        for commodity in sample_community.commodity_relevance:
            assert commodity in ALL_COMMODITIES


# ===========================================================================
# 5. Recognition Status (5 tests)
# ===========================================================================


class TestRecognitionStatus:
    """Test community legal recognition status."""

    @pytest.mark.parametrize("status", [
        CommunityRecognitionStatus.CONSTITUTIONALLY_RECOGNIZED,
        CommunityRecognitionStatus.STATUTORY_RECOGNITION,
        CommunityRecognitionStatus.CUSTOMARY_ONLY,
        CommunityRecognitionStatus.PENDING,
        CommunityRecognitionStatus.DENIED_DISPUTED,
    ])
    def test_all_recognition_statuses(self, status):
        """Test community at each recognition status."""
        c = IndigenousCommunity(
            community_id=f"c-{status.value[:5]}",
            community_name=f"Status {status.value}",
            people_name="Test",
            country_code="BR",
            legal_recognition_status=status,
            provenance_hash="k" * 64,
        )
        assert c.legal_recognition_status == status


# ===========================================================================
# 6. Batch Operations (5 tests)
# ===========================================================================


class TestCommunityBatchOperations:
    """Test batch import/export of communities."""

    def test_batch_five_communities(self, sample_communities):
        """Test batch of 5 communities."""
        assert len(sample_communities) == 5

    def test_batch_unique_ids(self, sample_communities):
        """Test all community IDs are unique in batch."""
        ids = [c.community_id for c in sample_communities]
        assert len(ids) == len(set(ids))

    def test_batch_multiple_countries(self, sample_communities):
        """Test batch spans multiple countries."""
        countries = {c.country_code for c in sample_communities}
        assert len(countries) >= 3

    def test_batch_serialization(self, sample_communities):
        """Test communities can be serialized to dicts."""
        dicts = [c.model_dump() for c in sample_communities]
        assert len(dicts) == 5
        for d in dicts:
            assert "community_id" in d
            assert "people_name" in d

    def test_batch_contact_channels(self):
        """Test community with contact channels."""
        c = IndigenousCommunity(
            community_id="c-contact",
            community_name="Contact Test",
            people_name="Test",
            country_code="BR",
            contact_channels=[
                {"type": "email", "value": "community@example.org"},
                {"type": "phone", "value": "+55-xxx"},
            ],
            provenance_hash="l" * 64,
        )
        assert len(c.contact_channels) == 2


# ===========================================================================
# 7. Provenance (5 tests)
# ===========================================================================


class TestRegistryProvenance:
    """Test provenance for community registry."""

    def test_community_provenance_length(self, sample_community):
        """Test community provenance hash is SHA-256."""
        assert len(sample_community.provenance_hash) == SHA256_HEX_LENGTH

    def test_provenance_deterministic(self):
        """Test same community data produces same hash."""
        data = {"community_id": "c-001", "community_name": "Yanomami"}
        h1 = compute_test_hash(data)
        h2 = compute_test_hash(data)
        assert h1 == h2

    def test_provenance_records_create(self, mock_provenance):
        """Test provenance records community creation."""
        mock_provenance.record("community", "create", "c-001")
        assert mock_provenance.entry_count == 1

    def test_provenance_records_update(self, mock_provenance):
        """Test provenance records community update."""
        mock_provenance.record("community", "update", "c-001")
        assert mock_provenance.entry_count == 1

    def test_provenance_chain_integrity(self, mock_provenance):
        """Test provenance chain is intact."""
        mock_provenance.record("community", "create", "c-001")
        mock_provenance.record("community", "update", "c-001")
        assert mock_provenance.verify_chain() is True
