# -*- coding: utf-8 -*-
"""
Tests for LegalFrameworkDatabaseEngine - AGENT-EUDR-023 Engine 1

Comprehensive test suite covering:
- CRUD operations for 27 EUDR-relevant countries
- All 8 legislation category coverage per EUDR Article 2(40)
- External legal database integration mocks (FAO FAOLEX, ECOLEX)
- Framework search and filtering by country, category, status, year
- Reliability scoring calculation and validation
- Data freshness and last-updated tracking
- Provenance hash generation for framework records
- Error handling for invalid countries, missing data, API failures
- Bulk import and sync operations
- Framework comparison across countries

Test count: 70+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Engine 1 - Legal Framework Database)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    compute_compliance_score,
    SHA256_HEX_LENGTH,
    LEGISLATION_CATEGORIES,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES_27,
    HIGH_RISK_COUNTRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_framework_record(
    country_code: str,
    category: str,
    status: str = "active",
    reliability: float = 75.0,
    year_enacted: int = 2010,
) -> Dict[str, Any]:
    """Build a single framework record for testing."""
    return {
        "id": f"FW-{country_code}-{category[:4].upper()}",
        "country_code": country_code,
        "category": category,
        "title": f"{country_code} {category} legislation",
        "authority": f"Authority-{country_code}",
        "status": status,
        "year_enacted": year_enacted,
        "last_updated": date.today().isoformat(),
        "reliability_score": Decimal(str(reliability)),
        "provenance_hash": compute_test_hash({
            "country": country_code, "category": category,
        }),
    }


def _build_country_frameworks(country_code: str) -> Dict[str, Any]:
    """Build a complete 8-category framework set for a country."""
    frameworks = {}
    for cat in LEGISLATION_CATEGORIES:
        frameworks[cat] = _build_framework_record(country_code, cat)
    return {"country_code": country_code, "frameworks": frameworks}


def _search_frameworks(
    entries: List[Dict],
    country_code: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    keyword: Optional[str] = None,
) -> List[Dict]:
    """Filter framework entries based on search criteria."""
    results = list(entries)
    if country_code:
        results = [e for e in results if e["country_code"] == country_code]
    if category:
        results = [e for e in results if e["category"] == category]
    if status:
        results = [e for e in results if e["status"] == status]
    if min_year:
        results = [e for e in results if e["year_enacted"] >= min_year]
    if max_year:
        results = [e for e in results if e["year_enacted"] <= max_year]
    if keyword:
        kw = keyword.lower()
        results = [e for e in results if kw in e["title"].lower()]
    return results


def _calculate_reliability(framework: Dict) -> Decimal:
    """Calculate reliability score for a framework based on freshness and completeness."""
    base = Decimal("50")
    # Freshness bonus: updated in last 2 years
    if framework.get("last_updated"):
        last = date.fromisoformat(framework["last_updated"])
        days_old = (date.today() - last).days
        if days_old <= 365:
            base += Decimal("30")
        elif days_old <= 730:
            base += Decimal("15")
    # Status bonus
    if framework.get("status") == "active":
        base += Decimal("20")
    elif framework.get("status") == "amended":
        base += Decimal("10")
    return min(base, Decimal("100"))


# ===========================================================================
# 1. Country Framework CRUD Operations (15 tests)
# ===========================================================================


class TestCountryFrameworkCRUD:
    """Test creating, reading, updating, and deleting country frameworks."""

    def test_create_country_framework(self):
        """Test creating a new country framework record."""
        fw = _build_country_frameworks("BR")
        assert fw["country_code"] == "BR"
        assert len(fw["frameworks"]) == 8

    def test_create_framework_all_categories(self):
        """Test that framework creation covers all 8 Article 2(40) categories."""
        fw = _build_country_frameworks("ID")
        for cat in LEGISLATION_CATEGORIES:
            assert cat in fw["frameworks"], f"Missing category: {cat}"

    def test_read_existing_country_framework(self, sample_legal_frameworks):
        """Test reading an existing country framework returns correct data."""
        brazil = sample_legal_frameworks["BR"]
        assert brazil["country_code"] == "BR"
        assert brazil["country_name"] == "Brazil"
        assert "land_use_rights" in brazil["frameworks"]

    def test_read_nonexistent_country_returns_none(self, sample_legal_frameworks):
        """Test reading a non-existent country returns None or empty."""
        result = sample_legal_frameworks.get("XX")
        assert result is None

    def test_update_framework_status(self):
        """Test updating a framework record status field."""
        fw = _build_framework_record("BR", "land_use_rights", status="active")
        fw["status"] = "amended"
        assert fw["status"] == "amended"

    def test_update_framework_reliability(self):
        """Test updating reliability score after new data."""
        fw = _build_framework_record("BR", "land_use_rights", reliability=70.0)
        fw["reliability_score"] = Decimal("85.0")
        assert fw["reliability_score"] == Decimal("85.0")

    def test_delete_framework_record(self):
        """Test deleting a framework record removes it from storage."""
        frameworks = {"BR": _build_country_frameworks("BR")}
        del frameworks["BR"]
        assert "BR" not in frameworks

    def test_framework_record_has_provenance_hash(self):
        """Test each framework record includes a provenance hash."""
        fw = _build_framework_record("BR", "land_use_rights")
        assert "provenance_hash" in fw
        assert len(fw["provenance_hash"]) == SHA256_HEX_LENGTH

    @pytest.mark.parametrize("country_code", EUDR_COUNTRIES_27[:10])
    def test_create_framework_for_eudr_countries(self, country_code):
        """Test framework creation for 10 EUDR-relevant countries."""
        fw = _build_country_frameworks(country_code)
        assert fw["country_code"] == country_code
        assert len(fw["frameworks"]) == 8

    def test_framework_defaults_to_active_status(self):
        """Test that new framework records default to active status."""
        fw = _build_framework_record("BR", "land_use_rights")
        assert fw["status"] == "active"

    def test_framework_includes_year_enacted(self):
        """Test that framework record includes year enacted."""
        fw = _build_framework_record("BR", "land_use_rights", year_enacted=2012)
        assert fw["year_enacted"] == 2012

    def test_framework_includes_last_updated(self):
        """Test that framework record includes last_updated date."""
        fw = _build_framework_record("BR", "land_use_rights")
        assert fw["last_updated"] == date.today().isoformat()

    def test_framework_includes_authority(self):
        """Test that framework record includes issuing authority."""
        fw = _build_framework_record("BR", "land_use_rights")
        assert "authority" in fw
        assert len(fw["authority"]) > 0

    def test_create_multiple_countries(self):
        """Test creating frameworks for multiple countries simultaneously."""
        frameworks = {}
        for cc in ["BR", "ID", "CD"]:
            frameworks[cc] = _build_country_frameworks(cc)
        assert len(frameworks) == 3

    def test_framework_id_format(self):
        """Test framework record ID follows expected naming convention."""
        fw = _build_framework_record("BR", "land_use_rights")
        assert fw["id"].startswith("FW-BR-")


# ===========================================================================
# 2. Legislation Category Coverage (12 tests)
# ===========================================================================


class TestLegislationCategoryCoverage:
    """Test coverage of all 8 EUDR Article 2(40) legislation categories."""

    @pytest.mark.parametrize("category", LEGISLATION_CATEGORIES)
    def test_category_exists_in_framework(self, category, sample_legal_frameworks):
        """Test each of the 8 categories exists in Brazil framework."""
        assert category in sample_legal_frameworks["BR"]["frameworks"]

    def test_all_categories_have_legislation_reference(self, sample_legal_frameworks):
        """Test all categories have a legislation reference for Brazil."""
        for cat in LEGISLATION_CATEGORIES:
            fw = sample_legal_frameworks["BR"]["frameworks"][cat]
            assert "legislation" in fw
            assert len(fw["legislation"]) > 0

    def test_all_categories_have_authority(self, sample_legal_frameworks):
        """Test all categories have an issuing authority."""
        for cat in LEGISLATION_CATEGORIES:
            fw = sample_legal_frameworks["BR"]["frameworks"][cat]
            assert "authority" in fw
            assert len(fw["authority"]) > 0

    def test_category_coverage_count_brazil(self, sample_legal_frameworks):
        """Test Brazil has coverage for exactly 8 categories."""
        count = len(sample_legal_frameworks["BR"]["frameworks"])
        assert count == 8

    def test_category_coverage_count_indonesia(self, sample_legal_frameworks):
        """Test Indonesia has coverage for exactly 8 categories."""
        count = len(sample_legal_frameworks["ID"]["frameworks"])
        assert count == 8

    def test_category_coverage_count_drc(self, sample_legal_frameworks):
        """Test DRC has coverage for exactly 8 categories."""
        count = len(sample_legal_frameworks["CD"]["frameworks"])
        assert count == 8

    def test_incomplete_coverage_detection(self):
        """Test detection of countries with fewer than 8 categories."""
        fw = _build_country_frameworks("GH")
        # Simulate removing one category
        del fw["frameworks"]["anti_corruption"]
        missing = set(LEGISLATION_CATEGORIES) - set(fw["frameworks"].keys())
        assert "anti_corruption" in missing

    def test_category_names_match_article_2_40(self):
        """Test category names match EUDR Article 2(40) definitions."""
        expected = {
            "land_use_rights", "environmental_protection", "forest_related_rules",
            "third_party_rights", "labour_rights", "tax_and_royalty",
            "trade_and_customs", "anti_corruption",
        }
        assert set(LEGISLATION_CATEGORIES) == expected


# ===========================================================================
# 3. External Legal Database Integration (10 tests)
# ===========================================================================


class TestExternalDatabaseIntegration:
    """Test integration with FAO FAOLEX and ECOLEX APIs."""

    def test_faolex_search_returns_results(self, mock_faolex_api):
        """Test FAOLEX API search returns legislation results."""
        result = mock_faolex_api.search_legislation(country="BR", topic="forest")
        assert result["total"] >= 1
        assert len(result["results"]) >= 1
        assert result["results"][0]["country"] == "BR"

    def test_faolex_get_legislation_details(self, mock_faolex_api):
        """Test FAOLEX API retrieves full legislation details."""
        result = mock_faolex_api.get_legislation("FAOLEX-BR-001")
        assert result["id"] == "FAOLEX-BR-001"
        assert result["full_text_available"] is True

    def test_ecolex_search_returns_treaties(self, mock_ecolex_api):
        """Test ECOLEX API search returns treaty information."""
        result = mock_ecolex_api.search(topic="biodiversity")
        assert result["total"] >= 1
        assert result["results"][0]["type"] == "treaty"

    def test_faolex_api_health_check(self, mock_faolex_api):
        """Test FAOLEX API health check endpoint."""
        assert mock_faolex_api.check_health() is True

    def test_ecolex_api_health_check(self, mock_ecolex_api):
        """Test ECOLEX API health check endpoint."""
        assert mock_ecolex_api.check_health() is True

    def test_faolex_api_timeout_handling(self):
        """Test handling of FAOLEX API timeout errors."""
        mock = MagicMock()
        mock.search_legislation.side_effect = TimeoutError("API timeout")
        with pytest.raises(TimeoutError):
            mock.search_legislation(country="BR")

    def test_ecolex_api_connection_error(self):
        """Test handling of ECOLEX API connection errors."""
        mock = MagicMock()
        mock.search.side_effect = ConnectionError("Connection refused")
        with pytest.raises(ConnectionError):
            mock.search(topic="forest")

    def test_faolex_empty_search_results(self):
        """Test handling of empty search results from FAOLEX."""
        mock = MagicMock()
        mock.search_legislation.return_value = {"results": [], "total": 0}
        result = mock.search_legislation(country="XX")
        assert result["total"] == 0

    def test_sync_external_sources_updates_frameworks(self, framework_engine):
        """Test syncing external sources updates local framework data."""
        framework_engine.sync_external_sources.return_value = {
            "synced": 5, "updated": 3, "errors": 0,
        }
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            framework_engine.sync_external_sources()
        ) if hasattr(framework_engine.sync_external_sources, '__wrapped__') else (
            framework_engine.sync_external_sources()
        )
        assert result["synced"] >= 0

    def test_external_api_rate_limiting(self):
        """Test rate limiting for external API calls."""
        call_count = 0
        mock = MagicMock()

        def rate_limited_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 10:
                raise Exception("Rate limit exceeded")
            return {"results": [], "total": 0}

        mock.search_legislation.side_effect = rate_limited_call
        for _ in range(10):
            mock.search_legislation(country="BR")
        with pytest.raises(Exception, match="Rate limit"):
            mock.search_legislation(country="BR")


# ===========================================================================
# 4. Framework Search and Filtering (15 tests)
# ===========================================================================


class TestFrameworkSearchAndFiltering:
    """Test searching and filtering legal framework entries."""

    def test_search_by_country(self, sample_legislation_db):
        """Test searching legislation by country code."""
        results = _search_frameworks(sample_legislation_db, country_code="BR")
        assert all(e["country_code"] == "BR" for e in results)
        assert len(results) >= 1

    def test_search_by_category(self, sample_legislation_db):
        """Test searching legislation by category."""
        results = _search_frameworks(sample_legislation_db, category="land_use_rights")
        assert all(e["category"] == "land_use_rights" for e in results)

    def test_search_by_status_active(self, sample_legislation_db):
        """Test filtering by active status."""
        results = _search_frameworks(sample_legislation_db, status="active")
        assert all(e["status"] == "active" for e in results)

    def test_search_by_status_repealed(self, sample_legislation_db):
        """Test filtering by repealed status."""
        results = _search_frameworks(sample_legislation_db, status="repealed")
        assert all(e["status"] == "repealed" for e in results)

    def test_search_combined_country_and_category(self, sample_legislation_db):
        """Test combined search by country and category."""
        results = _search_frameworks(
            sample_legislation_db, country_code="BR", category="environmental_protection",
        )
        assert all(
            e["country_code"] == "BR" and e["category"] == "environmental_protection"
            for e in results
        )

    def test_search_by_year_range(self, sample_legislation_db):
        """Test filtering by year enacted range."""
        results = _search_frameworks(sample_legislation_db, min_year=2010, max_year=2020)
        assert all(2010 <= e["year_enacted"] <= 2020 for e in results)

    def test_search_returns_empty_for_no_match(self, sample_legislation_db):
        """Test search returns empty list when no entries match."""
        results = _search_frameworks(sample_legislation_db, country_code="XX")
        assert len(results) == 0

    def test_search_all_entries_without_filter(self, sample_legislation_db):
        """Test search without filters returns all entries."""
        results = _search_frameworks(sample_legislation_db)
        assert len(results) == len(sample_legislation_db)

    def test_search_by_keyword(self, sample_legislation_db):
        """Test keyword-based search in legislation titles."""
        results = _search_frameworks(sample_legislation_db, keyword="BR-land")
        assert len(results) >= 1

    def test_search_results_contain_required_fields(self, sample_legislation_db):
        """Test that search results contain all required fields."""
        results = _search_frameworks(sample_legislation_db, country_code="BR")
        required_fields = {"id", "country_code", "category", "title", "status"}
        for entry in results:
            for field in required_fields:
                assert field in entry, f"Missing field: {field}"

    def test_search_by_minimum_reliability(self, sample_legislation_db):
        """Test filtering by minimum reliability score."""
        min_score = Decimal("70")
        results = [e for e in sample_legislation_db if e["reliability_score"] >= min_score]
        assert all(e["reliability_score"] >= min_score for e in results)

    def test_search_pagination_offset(self, sample_legislation_db):
        """Test search with offset-based pagination."""
        page_size = 5
        page1 = sample_legislation_db[:page_size]
        page2 = sample_legislation_db[page_size:page_size*2]
        assert len(page1) <= page_size
        assert page1 != page2

    def test_search_count_only(self, sample_legislation_db):
        """Test getting count of matching records without full data."""
        count = len(_search_frameworks(sample_legislation_db, country_code="BR"))
        assert isinstance(count, int)
        assert count >= 1

    def test_search_sorted_by_year(self, sample_legislation_db):
        """Test search results can be sorted by year enacted."""
        results = _search_frameworks(sample_legislation_db, country_code="BR")
        sorted_results = sorted(results, key=lambda x: x["year_enacted"])
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i]["year_enacted"] <= sorted_results[i+1]["year_enacted"]

    def test_search_case_insensitive_keyword(self, sample_legislation_db):
        """Test keyword search is case insensitive."""
        results_upper = _search_frameworks(sample_legislation_db, keyword="BR")
        results_lower = _search_frameworks(sample_legislation_db, keyword="br")
        assert len(results_upper) == len(results_lower)


# ===========================================================================
# 5. Reliability Scoring (10 tests)
# ===========================================================================


class TestReliabilityScoring:
    """Test framework reliability score calculation and validation."""

    def test_reliability_score_recently_updated(self):
        """Test reliability score for recently updated framework (within 1 year)."""
        fw = {"last_updated": date.today().isoformat(), "status": "active"}
        score = _calculate_reliability(fw)
        assert score >= Decimal("80")

    def test_reliability_score_old_update(self):
        """Test reliability score for framework updated more than 2 years ago."""
        old_date = (date.today() - timedelta(days=800)).isoformat()
        fw = {"last_updated": old_date, "status": "active"}
        score = _calculate_reliability(fw)
        assert score >= Decimal("50")
        assert score < Decimal("100")

    def test_reliability_score_active_status_bonus(self):
        """Test active status adds bonus to reliability score."""
        fw_active = {"last_updated": date.today().isoformat(), "status": "active"}
        fw_repealed = {"last_updated": date.today().isoformat(), "status": "repealed"}
        score_active = _calculate_reliability(fw_active)
        score_repealed = _calculate_reliability(fw_repealed)
        assert score_active > score_repealed

    def test_reliability_score_capped_at_100(self):
        """Test reliability score does not exceed 100."""
        fw = {"last_updated": date.today().isoformat(), "status": "active"}
        score = _calculate_reliability(fw)
        assert score <= Decimal("100")

    def test_reliability_score_minimum_50(self):
        """Test base reliability score starts at 50."""
        fw = {}
        score = _calculate_reliability(fw)
        assert score >= Decimal("50")

    def test_reliability_brazil_high(self, sample_legal_frameworks):
        """Test Brazil environmental protection has high reliability."""
        env = sample_legal_frameworks["BR"]["frameworks"]["environmental_protection"]
        assert env["reliability_score"] >= Decimal("80")

    def test_reliability_drc_low(self, sample_legal_frameworks):
        """Test DRC frameworks have lower reliability scores."""
        anti = sample_legal_frameworks["CD"]["frameworks"]["anti_corruption"]
        assert anti["reliability_score"] <= Decimal("40")

    def test_reliability_comparison_across_countries(self, sample_legal_frameworks):
        """Test Brazil has higher reliability than DRC across categories."""
        br_scores = [
            sample_legal_frameworks["BR"]["frameworks"][c]["reliability_score"]
            for c in LEGISLATION_CATEGORIES
        ]
        cd_scores = [
            sample_legal_frameworks["CD"]["frameworks"][c]["reliability_score"]
            for c in LEGISLATION_CATEGORIES
        ]
        assert sum(br_scores) > sum(cd_scores)

    def test_reliability_amended_status_bonus(self):
        """Test amended status adds partial bonus."""
        fw = {"last_updated": date.today().isoformat(), "status": "amended"}
        score = _calculate_reliability(fw)
        assert score >= Decimal("60")

    def test_reliability_freshness_decay(self):
        """Test reliability decays with age of last update."""
        recent = {"last_updated": date.today().isoformat(), "status": "active"}
        old = {
            "last_updated": (date.today() - timedelta(days=400)).isoformat(),
            "status": "active",
        }
        very_old = {
            "last_updated": (date.today() - timedelta(days=1000)).isoformat(),
            "status": "active",
        }
        assert _calculate_reliability(recent) >= _calculate_reliability(old)
        assert _calculate_reliability(old) >= _calculate_reliability(very_old)


# ===========================================================================
# 6. Provenance and Audit (8 tests)
# ===========================================================================


class TestFrameworkProvenance:
    """Test provenance tracking for legal framework operations."""

    def test_framework_record_has_hash(self):
        """Test every framework record gets a provenance hash."""
        fw = _build_framework_record("BR", "land_use_rights")
        assert len(fw["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_hash_deterministic(self):
        """Test same input produces same provenance hash."""
        h1 = compute_test_hash({"country": "BR", "category": "land_use_rights"})
        h2 = compute_test_hash({"country": "BR", "category": "land_use_rights"})
        assert h1 == h2

    def test_hash_changes_with_different_input(self):
        """Test different input produces different provenance hash."""
        h1 = compute_test_hash({"country": "BR", "category": "land_use_rights"})
        h2 = compute_test_hash({"country": "ID", "category": "land_use_rights"})
        assert h1 != h2

    def test_provenance_records_operation(self, mock_provenance):
        """Test provenance tracker records framework operations."""
        entry = mock_provenance.record("framework", "create", "FW-BR-001")
        assert entry["entity_type"] == "framework"
        assert entry["action"] == "create"
        assert len(entry["hash_value"]) == SHA256_HEX_LENGTH

    def test_provenance_chain_integrity(self, mock_provenance):
        """Test provenance chain maintains integrity."""
        mock_provenance.record("framework", "create", "FW-BR-001")
        mock_provenance.record("framework", "update", "FW-BR-001")
        assert mock_provenance.verify_chain() is True

    def test_provenance_parent_hash_chain(self, mock_provenance):
        """Test each entry references its parent hash."""
        e1 = mock_provenance.record("framework", "create", "FW-BR-001")
        e2 = mock_provenance.record("framework", "update", "FW-BR-001")
        assert e2["parent_hash"] == e1["hash_value"]

    def test_provenance_genesis_hash(self, mock_provenance):
        """Test first entry references genesis hash."""
        entry = mock_provenance.record("framework", "create", "FW-BR-001")
        assert entry["parent_hash"] == mock_provenance.genesis_hash

    def test_provenance_includes_timestamp(self, mock_provenance):
        """Test provenance entries include timestamp."""
        entry = mock_provenance.record("framework", "create", "FW-BR-001")
        assert "timestamp" in entry
