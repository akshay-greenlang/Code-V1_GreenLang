# -*- coding: utf-8 -*-
"""
Unit tests for RegulatoryComplianceEngine (AGENT-EUDR-018 Engine 6).

Tests EUDR article mapping, compliance checking, penalty risk assessment,
documentation requirements, import ban risk, regulatory updates,
compliance scoring, and gap analysis for all 7 EUDR commodities.

Coverage target: 85%+
"""

from decimal import Decimal
import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.engines.regulatory_compliance_engine import (
    RegulatoryComplianceEngine,
    EUDR_COMMODITIES,
    EUDR_ARTICLES,
    COMMODITY_DOCUMENTATION,
    MEMBER_STATE_PENALTIES,
    COUNTRY_RISK_BENCHMARKS,
    REGULATORY_UPDATES,
)

SEVEN_COMMODITIES = sorted(EUDR_COMMODITIES)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for RegulatoryComplianceEngine initialization."""

    @pytest.mark.unit
    def test_init_empty_state(self):
        """Engine initializes with empty compliance records."""
        engine = RegulatoryComplianceEngine()
        assert engine._compliance_records == {}

    @pytest.mark.unit
    def test_init_creates_lock(self):
        """Engine creates a reentrant lock."""
        engine = RegulatoryComplianceEngine()
        assert engine._lock is not None


# ---------------------------------------------------------------------------
# TestGetRequirements
# ---------------------------------------------------------------------------

class TestGetRequirements:
    """Tests for get_requirements method."""

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_requirements_non_empty(self, regulatory_compliance_engine, commodity):
        """Every commodity returns a non-empty requirements list."""
        reqs = regulatory_compliance_engine.get_requirements(commodity)
        assert len(reqs) > 0

    @pytest.mark.unit
    def test_requirements_include_articles_and_docs(self, regulatory_compliance_engine):
        """Requirements combine article-level and documentation-level entries."""
        reqs = regulatory_compliance_engine.get_requirements("wood")
        # Should have article entries (10 articles) plus wood-specific docs
        article_count = len(EUDR_ARTICLES)
        doc_count = len(COMMODITY_DOCUMENTATION["wood"])
        assert len(reqs) == article_count + doc_count

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, regulatory_compliance_engine):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="not a valid EUDR commodity"):
            regulatory_compliance_engine.get_requirements("banana")

    @pytest.mark.unit
    def test_invalid_market_raises(self, regulatory_compliance_engine):
        """Invalid market raises ValueError."""
        with pytest.raises(ValueError, match="not a valid market"):
            regulatory_compliance_engine.get_requirements("soya", market="JP")


# ---------------------------------------------------------------------------
# TestCheckCompliance
# ---------------------------------------------------------------------------

class TestCheckCompliance:
    """Tests for check_compliance method."""

    @pytest.mark.unit
    def test_full_compliance(self, regulatory_compliance_engine):
        """Providing all documentation yields high compliance score."""
        docs = COMMODITY_DOCUMENTATION["soya"]
        documentation = {doc["doc_type"]: True for doc in docs}
        supplier_data = {"supplier_id": "S-001", "origin_country": "BR"}

        result = regulatory_compliance_engine.check_compliance("soya", supplier_data, documentation)
        assert Decimal(result["compliance_score"]) == Decimal("100.00")
        assert result["overall_status"] == "COMPLIANT"
        assert result["missing_mandatory_count"] == 0

    @pytest.mark.unit
    def test_no_documentation_non_compliant(self, regulatory_compliance_engine):
        """Providing zero documentation yields NON_COMPLIANT status."""
        supplier_data = {"supplier_id": "S-002", "origin_country": "ID"}
        result = regulatory_compliance_engine.check_compliance("oil_palm", supplier_data, {})
        assert result["overall_status"] == "NON_COMPLIANT"
        assert result["missing_mandatory_count"] > 0

    @pytest.mark.unit
    def test_country_risk_included(self, regulatory_compliance_engine):
        """Origin country risk is included in compliance result."""
        supplier_data = {"supplier_id": "S-003", "origin_country": "BR"}
        result = regulatory_compliance_engine.check_compliance("cattle", supplier_data, {})
        assert result["country_risk"] == "HIGH"

    @pytest.mark.unit
    def test_missing_supplier_id_raises(self, regulatory_compliance_engine):
        """Missing supplier_id in supplier_data raises ValueError."""
        with pytest.raises(ValueError, match="supplier_id"):
            regulatory_compliance_engine.check_compliance("soya", {}, {})


# ---------------------------------------------------------------------------
# TestPenaltyRisk
# ---------------------------------------------------------------------------

class TestPenaltyRisk:
    """Tests for assess_penalty_risk method."""

    @pytest.mark.unit
    def test_critical_gaps_critical_risk(self, regulatory_compliance_engine):
        """Critical compliance gaps produce CRITICAL penalty risk."""
        gaps = [
            {"severity": "CRITICAL", "requirement": "deforestation_declaration"},
            {"severity": "HIGH", "requirement": "farm_gps_coordinates"},
        ]
        result = regulatory_compliance_engine.assess_penalty_risk("soya", gaps, "DE")
        assert result["penalty_risk_level"] == "CRITICAL"
        assert result["criminal_liability"] is True

    @pytest.mark.unit
    def test_no_gaps_low_risk(self, regulatory_compliance_engine):
        """No gaps produce LOW penalty risk with zero fine range."""
        result = regulatory_compliance_engine.assess_penalty_risk("cocoa", [], "NL")
        assert result["penalty_risk_level"] == "LOW"
        assert result["estimated_fine_range"]["min_eur"] == str(Decimal("0"))

    @pytest.mark.unit
    def test_default_member_state(self, regulatory_compliance_engine):
        """DEFAULT member state uses default penalty data."""
        gaps = [{"severity": "MEDIUM"}]
        result = regulatory_compliance_engine.assess_penalty_risk("rubber", gaps)
        assert result["member_state"] == "DEFAULT"
        assert result["member_state_name"] == "Default EU Member State"

    @pytest.mark.unit
    def test_contributing_factors_listed(self, regulatory_compliance_engine):
        """Contributing factors include criminal liability when applicable."""
        gaps = [{"severity": "CRITICAL"}]
        result = regulatory_compliance_engine.assess_penalty_risk("wood", gaps, "FR")
        factors = result["contributing_factors"]
        assert any("Criminal" in f for f in factors)


# ---------------------------------------------------------------------------
# TestDocumentationRequirements
# ---------------------------------------------------------------------------

class TestDocumentationRequirements:
    """Tests for get_documentation_requirements method."""

    @pytest.mark.unit
    def test_wood_has_species_id(self, regulatory_compliance_engine):
        """Wood documentation includes species_identification."""
        docs = regulatory_compliance_engine.get_documentation_requirements("wood")
        doc_types = [d["doc_type"] for d in docs]
        assert "species_identification" in doc_types

    @pytest.mark.unit
    def test_oil_palm_has_ndpe(self, regulatory_compliance_engine):
        """Oil palm documentation includes ndpe_compliance."""
        docs = regulatory_compliance_engine.get_documentation_requirements("oil_palm")
        doc_types = [d["doc_type"] for d in docs]
        assert "ndpe_compliance" in doc_types

    @pytest.mark.unit
    def test_soya_has_gmo_declaration(self, regulatory_compliance_engine):
        """Soya documentation includes gmo_status_declaration."""
        docs = regulatory_compliance_engine.get_documentation_requirements("soya")
        doc_types = [d["doc_type"] for d in docs]
        assert "gmo_status_declaration" in doc_types

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_all_commodities_have_deforestation_declaration(
        self, regulatory_compliance_engine, commodity,
    ):
        """Every commodity requires deforestation_free_declaration."""
        docs = regulatory_compliance_engine.get_documentation_requirements(commodity)
        doc_types = [d["doc_type"] for d in docs]
        assert "deforestation_free_declaration" in doc_types


# ---------------------------------------------------------------------------
# TestArticleMapping
# ---------------------------------------------------------------------------

class TestArticleMapping:
    """Tests for map_articles_to_commodity method."""

    @pytest.mark.unit
    def test_mapping_contains_all_articles(self, regulatory_compliance_engine):
        """Mapping includes all 10 EUDR articles."""
        result = regulatory_compliance_engine.map_articles_to_commodity("cattle")
        assert result["total_articles"] == len(EUDR_ARTICLES)

    @pytest.mark.unit
    def test_article_3_present(self, regulatory_compliance_engine):
        """Article 3 (Prohibition) is in the mapping."""
        result = regulatory_compliance_engine.map_articles_to_commodity("coffee")
        assert "article_3" in result["article_mapping"]
        assert result["article_mapping"]["article_3"]["title"] == "Prohibition"

    @pytest.mark.unit
    def test_article_9_present(self, regulatory_compliance_engine):
        """Article 9 (Due diligence statements) is mapped."""
        result = regulatory_compliance_engine.map_articles_to_commodity("soya")
        assert "article_9" in result["article_mapping"]

    @pytest.mark.unit
    def test_article_13_present(self, regulatory_compliance_engine):
        """Article 13 (Record keeping) is mapped."""
        result = regulatory_compliance_engine.map_articles_to_commodity("rubber")
        assert "article_13" in result["article_mapping"]

    @pytest.mark.unit
    def test_article_8_has_commodity_docs(self, regulatory_compliance_engine):
        """Article 8 has commodity-specific documentation requirements."""
        result = regulatory_compliance_engine.map_articles_to_commodity("wood")
        art8 = result["article_mapping"]["article_8"]
        assert art8["requirement_count"] > 0


# ---------------------------------------------------------------------------
# TestImportBanRisk
# ---------------------------------------------------------------------------

class TestImportBanRisk:
    """Tests for check_import_ban_risk method."""

    @pytest.mark.unit
    def test_high_risk_country_commodity(self, regulatory_compliance_engine):
        """Brazil soya is high risk for import ban."""
        result = regulatory_compliance_engine.check_import_ban_risk("soya", "BR")
        assert result["country_benchmark"] == "HIGH"
        assert result["ban_risk_level"] in ("HIGH", "MEDIUM")
        assert result["enhanced_due_diligence_required"] is True

    @pytest.mark.unit
    def test_low_risk_country(self, regulatory_compliance_engine):
        """US wood is low risk for import ban."""
        result = regulatory_compliance_engine.check_import_ban_risk("wood", "US")
        assert result["country_benchmark"] == "LOW"
        assert result["ban_risk_level"] == "LOW"

    @pytest.mark.unit
    def test_empty_country_raises(self, regulatory_compliance_engine):
        """Empty origin_country raises ValueError."""
        with pytest.raises(ValueError, match="origin_country"):
            regulatory_compliance_engine.check_import_ban_risk("cocoa", "")


# ---------------------------------------------------------------------------
# TestRegulatoryUpdates
# ---------------------------------------------------------------------------

class TestRegulatoryUpdates:
    """Tests for get_regulatory_updates method."""

    @pytest.mark.unit
    def test_all_updates(self, regulatory_compliance_engine):
        """Unfiltered query returns all updates."""
        updates = regulatory_compliance_engine.get_regulatory_updates()
        assert len(updates) == len(REGULATORY_UPDATES)

    @pytest.mark.unit
    def test_filter_by_commodity(self, regulatory_compliance_engine):
        """Filter by oil_palm returns only oil_palm-relevant updates."""
        updates = regulatory_compliance_engine.get_regulatory_updates(commodity_type="oil_palm")
        for u in updates:
            assert "oil_palm" in u["affected_commodities"]

    @pytest.mark.unit
    def test_filter_by_date(self, regulatory_compliance_engine):
        """Filter by since_date returns only newer updates."""
        updates = regulatory_compliance_engine.get_regulatory_updates(since_date="2026-01-01")
        for u in updates:
            assert u["date"] >= "2026-01-01"

    @pytest.mark.unit
    def test_updates_sorted_descending(self, regulatory_compliance_engine):
        """Updates are sorted by date descending."""
        updates = regulatory_compliance_engine.get_regulatory_updates()
        dates = [u["date"] for u in updates]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# TestComplianceScore
# ---------------------------------------------------------------------------

class TestComplianceScore:
    """Tests for calculate_compliance_score method."""

    @pytest.mark.unit
    def test_full_evidence_100(self, regulatory_compliance_engine):
        """Full evidence package yields score of 100."""
        docs = COMMODITY_DOCUMENTATION["coffee"]
        evidence = {doc["doc_type"]: True for doc in docs}
        score = regulatory_compliance_engine.calculate_compliance_score("coffee", evidence)
        assert score == Decimal("100.00")

    @pytest.mark.unit
    def test_empty_evidence_zero(self, regulatory_compliance_engine):
        """Empty evidence yields score of 0."""
        score = regulatory_compliance_engine.calculate_compliance_score("cattle", {})
        assert score == Decimal("0")

    @pytest.mark.unit
    def test_partial_evidence_mid(self, regulatory_compliance_engine):
        """Partial evidence yields score between 0 and 100."""
        docs = COMMODITY_DOCUMENTATION["rubber"]
        # Provide only first 3 docs
        evidence = {docs[i]["doc_type"]: True for i in range(3)}
        score = regulatory_compliance_engine.calculate_compliance_score("rubber", evidence)
        assert Decimal("0") < score < Decimal("100")


# ---------------------------------------------------------------------------
# TestGapAnalysis
# ---------------------------------------------------------------------------

class TestGapAnalysis:
    """Tests for generate_compliance_gap_analysis method."""

    @pytest.mark.unit
    def test_full_evidence_no_gaps(self, regulatory_compliance_engine):
        """Full evidence results in zero gaps."""
        docs = COMMODITY_DOCUMENTATION["cocoa"]
        state = {doc["doc_type"]: True for doc in docs}
        result = regulatory_compliance_engine.generate_compliance_gap_analysis("cocoa", state)
        assert result["gap_count"] == 0

    @pytest.mark.unit
    def test_no_evidence_all_gaps(self, regulatory_compliance_engine):
        """No evidence means all docs are gaps."""
        doc_count = len(COMMODITY_DOCUMENTATION["cattle"])
        result = regulatory_compliance_engine.generate_compliance_gap_analysis("cattle", {})
        assert result["gap_count"] == doc_count

    @pytest.mark.unit
    def test_gaps_sorted_by_severity(self, regulatory_compliance_engine):
        """Gaps are sorted with CRITICAL first."""
        result = regulatory_compliance_engine.generate_compliance_gap_analysis("oil_palm", {})
        if result["gap_count"] > 1:
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            severities = [g["severity"] for g in result["gaps"]]
            indices = [severity_order.get(s, 99) for s in severities]
            assert indices == sorted(indices)

    @pytest.mark.unit
    def test_gap_has_priority_rank(self, regulatory_compliance_engine):
        """Each gap has a priority_rank starting at 1."""
        result = regulatory_compliance_engine.generate_compliance_gap_analysis("wood", {})
        if result["gap_count"] > 0:
            assert result["gaps"][0]["priority_rank"] == 1

    @pytest.mark.unit
    def test_estimated_effort_positive(self, regulatory_compliance_engine):
        """Total estimated effort is positive when gaps exist."""
        result = regulatory_compliance_engine.generate_compliance_gap_analysis("soya", {})
        assert result["estimated_total_effort_days"] > 0


# ---------------------------------------------------------------------------
# TestAllCommodities
# ---------------------------------------------------------------------------

class TestAllCommodities:
    """Parametrized tests across all 7 EUDR commodities."""

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_documentation_requirements_non_empty(
        self, regulatory_compliance_engine, commodity,
    ):
        """Every commodity has at least 5 documentation requirements."""
        docs = regulatory_compliance_engine.get_documentation_requirements(commodity)
        assert len(docs) >= 5

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity", SEVEN_COMMODITIES)
    def test_gap_analysis_runs(self, regulatory_compliance_engine, commodity):
        """Gap analysis runs without error for every commodity."""
        result = regulatory_compliance_engine.generate_compliance_gap_analysis(commodity, {})
        assert "gap_count" in result
        assert "compliance_score" in result


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------

class TestProvenance:
    """Tests for provenance hash integrity."""

    @pytest.mark.unit
    def test_compliance_check_provenance(self, regulatory_compliance_engine):
        """Compliance check result has a 64-char SHA-256 hash."""
        supplier_data = {"supplier_id": "S-PROV", "origin_country": "CI"}
        result = regulatory_compliance_engine.check_compliance("cocoa", supplier_data, {})
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_article_mapping_provenance(self, regulatory_compliance_engine):
        """Article mapping result has provenance hash."""
        result = regulatory_compliance_engine.map_articles_to_commodity("rubber")
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_gap_analysis_provenance(self, regulatory_compliance_engine):
        """Gap analysis result has provenance hash."""
        result = regulatory_compliance_engine.generate_compliance_gap_analysis("coffee", {})
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling edge cases."""

    @pytest.mark.unit
    def test_invalid_commodity_in_score(self, regulatory_compliance_engine):
        """Invalid commodity in calculate_compliance_score raises ValueError."""
        with pytest.raises(ValueError):
            regulatory_compliance_engine.calculate_compliance_score("wheat", {})

    @pytest.mark.unit
    def test_invalid_commodity_in_gap_analysis(self, regulatory_compliance_engine):
        """Invalid commodity in gap analysis raises ValueError."""
        with pytest.raises(ValueError):
            regulatory_compliance_engine.generate_compliance_gap_analysis("wheat", {})

    @pytest.mark.unit
    def test_invalid_commodity_in_import_ban(self, regulatory_compliance_engine):
        """Invalid commodity in import ban check raises ValueError."""
        with pytest.raises(ValueError):
            regulatory_compliance_engine.check_import_ban_risk("wheat", "BR")

    @pytest.mark.unit
    def test_invalid_commodity_in_updates(self, regulatory_compliance_engine):
        """Invalid commodity filter in updates raises ValueError."""
        with pytest.raises(ValueError):
            regulatory_compliance_engine.get_regulatory_updates(commodity_type="wheat")
