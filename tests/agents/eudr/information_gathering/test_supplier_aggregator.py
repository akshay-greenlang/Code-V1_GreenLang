# -*- coding: utf-8 -*-
"""
Unit tests for SupplierInformationAggregator - AGENT-EUDR-027

Tests supplier data aggregation from multiple sources, entity resolution
via Jaro-Winkler similarity, duplicate detection, discrepancy detection,
completeness scoring, batch aggregation, profile caching, and statistics.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 4: Supplier Information Aggregator)
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.information_gathering.config import InformationGatheringConfig
from greenlang.agents.eudr.information_gathering.models import (
    EUDRCommodity,
    SupplierProfile,
)
from greenlang.agents.eudr.information_gathering.supplier_information_aggregator import (
    SupplierInformationAggregator,
    _jaro_winkler_similarity,
)


@pytest.fixture
def aggregator(config) -> SupplierInformationAggregator:
    return SupplierInformationAggregator(config)


@pytest.fixture
def sample_sources():
    return {
        "government_registry": {
            "name": "Green Coffee Exporters S.A.",
            "country_code": "CO",
            "registration_number": "REG-CO-12345",
            "postal_address": "Calle 100 #15-20, Bogota",
            "commodities": ["coffee"],
        },
        "supplier_self_declared": {
            "name": "Green Coffee Exporters",
            "country_code": "CO",
            "email": "contact@greencoffee.co",
            "alternative_names": ["GCE Ltd"],
            "commodities": ["coffee"],
            "plot_ids": ["PLOT-CO-001", "PLOT-CO-002"],
            "tier_depth": 1,
        },
    }


class TestSupplierAggregatorInit:
    """Test engine initialization."""

    def test_engine_initialization(self, config):
        agg = SupplierInformationAggregator(config)
        stats = agg.get_aggregation_stats()
        assert stats["profiles_cached"] == 0
        assert stats["dedup_enabled"] is True
        assert stats["fuzzy_threshold"] == 0.85
        assert stats["algorithm"] == "jaro_winkler"


class TestAggregateSupplier:
    """Test supplier data aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_supplier(self, aggregator, sample_sources):
        profile = await aggregator.aggregate_supplier("SUP-001", sample_sources)
        assert profile.supplier_id == "SUP-001"
        # Government registry name should win (higher priority)
        assert profile.name == "Green Coffee Exporters S.A."
        assert profile.country_code == "CO"
        assert profile.registration_number == "REG-CO-12345"
        assert profile.email == "contact@greencoffee.co"
        assert EUDRCommodity.COFFEE in profile.commodities
        assert "PLOT-CO-001" in profile.plot_ids
        assert profile.tier_depth == 1
        assert len(profile.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_aggregate_supplier_completeness(self, aggregator, sample_sources):
        profile = await aggregator.aggregate_supplier("SUP-002", sample_sources)
        assert profile.completeness_score > Decimal("0")
        # With name, country_code, postal_address, registration, commodities,
        # plot_ids, email, alternative_names, tier_depth -> high completeness
        assert profile.completeness_score >= Decimal("70")

    @pytest.mark.asyncio
    async def test_aggregate_supplier_confidence(self, aggregator, sample_sources):
        profile = await aggregator.aggregate_supplier("SUP-003", sample_sources)
        assert profile.confidence_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_aggregate_supplier_stores_profile(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-004", sample_sources)
        stored = aggregator.get_profile("SUP-004")
        assert stored is not None
        assert stored.supplier_id == "SUP-004"


class TestBatchAggregate:
    """Test batch supplier aggregation."""

    @pytest.mark.asyncio
    async def test_batch_aggregate(self, aggregator, sample_sources):
        supplier_sources = {
            "SUP-A": sample_sources,
            "SUP-B": {
                "supplier_self_declared": {
                    "name": "Timber Holdings Inc",
                    "country_code": "ID",
                    "commodities": ["wood"],
                },
            },
        }
        profiles = await aggregator.batch_aggregate(supplier_sources)
        assert len(profiles) == 2

    @pytest.mark.asyncio
    async def test_batch_aggregate_empty(self, aggregator):
        profiles = await aggregator.batch_aggregate({})
        assert profiles == []


class TestResolveEntity:
    """Test entity resolution via fuzzy matching."""

    @pytest.mark.asyncio
    async def test_resolve_entity_matching(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-010", sample_sources)
        # Exact-ish name match
        match = aggregator.resolve_entity("Green Coffee Exporters S.A.")
        assert match is not None
        assert match.supplier_id == "SUP-010"

    @pytest.mark.asyncio
    async def test_resolve_entity_by_alt_name(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-011", sample_sources)
        # Match via alternative name
        match = aggregator.resolve_entity("GCE Ltd")
        assert match is not None
        assert match.supplier_id == "SUP-011"

    @pytest.mark.asyncio
    async def test_resolve_entity_by_reg_number(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-012", sample_sources)
        match = aggregator.resolve_entity(
            "Any Name",
            reg_number="REG-CO-12345",
        )
        assert match is not None
        assert match.supplier_id == "SUP-012"

    @pytest.mark.asyncio
    async def test_resolve_entity_no_match(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-013", sample_sources)
        match = aggregator.resolve_entity("Completely Different Company XYZ")
        assert match is None

    @pytest.mark.asyncio
    async def test_resolve_entity_country_filter(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-014", sample_sources)
        # Wrong country filter should exclude match
        match = aggregator.resolve_entity(
            "Green Coffee Exporters S.A.",
            country="DE",
        )
        assert match is None


class TestDetectDuplicates:
    """Test duplicate detection."""

    def test_detect_duplicates(self, aggregator):
        profiles = [
            SupplierProfile(
                supplier_id="SUP-A",
                name="Green Coffee Exporters",
                country_code="CO",
            ),
            SupplierProfile(
                supplier_id="SUP-B",
                name="Green Coffee Exporters Ltd",
                country_code="CO",
            ),
            SupplierProfile(
                supplier_id="SUP-C",
                name="Totally Different Company",
                country_code="DE",
            ),
        ]
        duplicates = aggregator.detect_duplicates(profiles)
        # SUP-A and SUP-B should be detected as potential duplicates
        assert len(duplicates) >= 1
        dup_ids = [(d[0], d[1]) for d in duplicates]
        assert ("SUP-A", "SUP-B") in dup_ids

    def test_detect_duplicates_empty(self, aggregator):
        duplicates = aggregator.detect_duplicates([])
        assert duplicates == []

    def test_detect_duplicates_single(self, aggregator):
        profiles = [
            SupplierProfile(supplier_id="SUP-X", name="Only Supplier"),
        ]
        duplicates = aggregator.detect_duplicates(profiles)
        assert duplicates == []


class TestDetectDiscrepancies:
    """Test discrepancy detection between data sources."""

    def test_detect_discrepancies(self, aggregator):
        profile_data = {
            "government_registry": {
                "name": "Green Coffee SA",
                "country_code": "CO",
            },
            "supplier_self_declared": {
                "name": "Green Coffee Ltd",
                "country_code": "BR",
            },
        }
        discrepancies = aggregator.detect_discrepancies(profile_data)
        assert len(discrepancies) >= 1
        # Country code discrepancy should be detected
        country_disc = [d for d in discrepancies if d.field_name == "country_code"]
        assert len(country_disc) == 1
        assert country_disc[0].severity == "high"

    def test_detect_discrepancies_no_conflicts(self, aggregator):
        profile_data = {
            "government_registry": {
                "name": "Green Coffee SA",
                "country_code": "CO",
            },
            "supplier_self_declared": {
                "name": "Green Coffee SA",
                "country_code": "CO",
            },
        }
        discrepancies = aggregator.detect_discrepancies(profile_data)
        assert len(discrepancies) == 0

    def test_detect_discrepancies_registration_number(self, aggregator):
        profile_data = {
            "government_registry": {
                "registration_number": "REG-001",
            },
            "customs_record": {
                "registration_number": "REG-002",
            },
        }
        discrepancies = aggregator.detect_discrepancies(profile_data)
        reg_disc = [d for d in discrepancies if d.field_name == "registration_number"]
        assert len(reg_disc) == 1
        assert reg_disc[0].severity == "critical"


class TestJaroWinklerSimilarity:
    """Test Jaro-Winkler similarity implementation."""

    def test_identical_strings(self):
        assert _jaro_winkler_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        assert _jaro_winkler_similarity("abc", "xyz") < 0.5

    def test_similar_strings(self):
        score = _jaro_winkler_similarity("Green Coffee", "Green Coffe")
        assert score > 0.9

    def test_case_insensitive(self):
        score = _jaro_winkler_similarity("Green Coffee", "GREEN COFFEE")
        assert score == 1.0

    def test_empty_strings(self):
        assert _jaro_winkler_similarity("", "") == 1.0

    def test_one_empty_string(self):
        assert _jaro_winkler_similarity("hello", "") == 0.0


class TestCompletenessScoring:
    """Test completeness score calculation."""

    def test_completeness_score_full_profile(self, aggregator):
        profile = SupplierProfile(
            supplier_id="SUP-FULL",
            name="Complete Supplier",
            alternative_names=["CS"],
            postal_address="123 Main St",
            country_code="DE",
            email="test@test.com",
            registration_number="REG-123",
            commodities=[EUDRCommodity.COFFEE],
            certifications=[],  # No certs but certifications field empty
            plot_ids=["PLOT-001"],
            tier_depth=1,
        )
        score = aggregator.get_completeness_score(profile)
        # Missing: certifications (0.10 weight)
        # Present: name(0.15) + country(0.15) + address(0.10) + reg(0.15) +
        #          commodities(0.10) + plot_ids(0.10) + email(0.05) +
        #          alt_names(0.05) + tier_depth(0.05) = 0.90
        assert score == Decimal("90.00")

    def test_completeness_score_empty_profile(self, aggregator):
        profile = SupplierProfile(supplier_id="SUP-EMPTY", name="")
        score = aggregator.get_completeness_score(profile)
        assert score == Decimal("0.00")


class TestProfileManagement:
    """Test profile get, get_all, and clear operations."""

    @pytest.mark.asyncio
    async def test_get_profile(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-GET", sample_sources)
        profile = aggregator.get_profile("SUP-GET")
        assert profile is not None
        assert profile.supplier_id == "SUP-GET"

    def test_get_profile_not_found(self, aggregator):
        assert aggregator.get_profile("NONEXISTENT") is None

    @pytest.mark.asyncio
    async def test_get_all_profiles(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-ALL-1", sample_sources)
        await aggregator.aggregate_supplier("SUP-ALL-2", sample_sources)
        profiles = aggregator.get_all_profiles()
        assert len(profiles) == 2

    @pytest.mark.asyncio
    async def test_clear_profiles(self, aggregator, sample_sources):
        await aggregator.aggregate_supplier("SUP-CLR", sample_sources)
        aggregator.clear_profiles()
        assert aggregator.get_profile("SUP-CLR") is None
        assert aggregator.get_all_profiles() == []
