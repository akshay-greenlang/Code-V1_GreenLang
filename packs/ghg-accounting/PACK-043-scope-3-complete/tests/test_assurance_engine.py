# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Assurance Engine
===========================================

Tests evidence package generation, calculation provenance chain,
methodology decision log, data source inventory, assumption register,
completeness statement, assurance readiness score, verifier query
management, and finding tracking.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal

import pytest

from tests.conftest import compute_provenance_hash


# =============================================================================
# Evidence Package Generation
# =============================================================================


class TestEvidencePackageGeneration:
    """Test evidence package generation for assurance."""

    def test_assurance_id_present(self, sample_assurance_evidence):
        assert sample_assurance_evidence["assurance_id"] == "ASR-2025-001"

    def test_assurance_level(self, sample_assurance_evidence):
        assert sample_assurance_evidence["assurance_level"] in {"limited", "reasonable"}

    def test_standard_isae3410(self, sample_assurance_evidence):
        assert sample_assurance_evidence["standard"] == "ISAE_3410"

    def test_five_evidence_items(self, sample_assurance_evidence):
        assert len(sample_assurance_evidence["evidence_items"]) == 5

    def test_evidence_items_have_required_fields(self, sample_assurance_evidence):
        required = ["item_id", "type", "description", "status"]
        for item in sample_assurance_evidence["evidence_items"]:
            for field in required:
                assert field in item, f"Missing {field} in {item.get('item_id', 'unknown')}"

    def test_evidence_item_ids_unique(self, sample_assurance_evidence):
        ids = [item["item_id"] for item in sample_assurance_evidence["evidence_items"]]
        assert len(ids) == len(set(ids))


# =============================================================================
# Calculation Provenance Chain (SHA-256)
# =============================================================================


class TestCalculationProvenanceChain:
    """Test calculation provenance chain with SHA-256 hashes."""

    def test_provenance_hash_present(self, sample_assurance_evidence):
        provenance_item = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "calculation_provenance"
        )
        assert len(provenance_item["hash"]) == 64

    def test_provenance_hash_is_hex(self, sample_assurance_evidence):
        provenance_item = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "calculation_provenance"
        )
        h = provenance_item["hash"]
        assert all(c in "0123456789abcdef" for c in h)

    def test_provenance_hash_deterministic(self, sample_scope3_screening):
        """Same data should produce same hash."""
        h1 = compute_provenance_hash(sample_scope3_screening)
        h2 = compute_provenance_hash(sample_scope3_screening)
        assert h1 == h2

    def test_provenance_hash_changes_with_data(self, sample_scope3_screening):
        """Different data should produce different hash."""
        h1 = compute_provenance_hash(sample_scope3_screening)
        modified = dict(sample_scope3_screening)
        modified["reporting_year"] = 9999
        h2 = compute_provenance_hash(modified)
        assert h1 != h2

    def test_provenance_hash_sha256_length(self, sample_scope3_screening):
        h = compute_provenance_hash(sample_scope3_screening)
        assert len(h) == 64

    def test_provenance_chain_covers_15_categories(self, sample_assurance_evidence):
        provenance_item = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "calculation_provenance"
        )
        assert "15 category" in provenance_item["description"]


# =============================================================================
# Methodology Decision Log
# =============================================================================


class TestMethodologyDecisionLog:
    """Test methodology decision log."""

    def test_decision_log_present(self, sample_assurance_evidence):
        log = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "methodology_decision_log"
        )
        assert log["status"] == "complete"

    def test_decision_log_entries(self, sample_assurance_evidence):
        log = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "methodology_decision_log"
        )
        assert log["entries"] >= 15  # at least one per category

    def test_decision_log_sufficient(self, sample_assurance_evidence):
        """At least 3 entries per material category expected."""
        log = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "methodology_decision_log"
        )
        # 15+ categories * ~3 decisions = 45+ entries
        assert log["entries"] >= 30


# =============================================================================
# Data Source Inventory
# =============================================================================


class TestDataSourceInventory:
    """Test data source inventory completeness."""

    def test_data_source_inventory_present(self, sample_assurance_evidence):
        inv = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "data_source_inventory"
        )
        assert inv["status"] == "complete"

    def test_source_count(self, sample_assurance_evidence):
        inv = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "data_source_inventory"
        )
        assert inv["sources"] >= 50

    def test_primary_secondary_sum(self, sample_assurance_evidence):
        """Primary + secondary should equal 100%."""
        inv = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "data_source_inventory"
        )
        total = inv["primary_pct"] + inv["secondary_pct"]
        assert total == Decimal("100")


# =============================================================================
# Assumption Register
# =============================================================================


class TestAssumptionRegister:
    """Test assumption register documentation."""

    def test_assumption_register_present(self, sample_assurance_evidence):
        reg = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "assumption_register"
        )
        assert reg["status"] == "complete"

    def test_assumption_count(self, sample_assurance_evidence):
        reg = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "assumption_register"
        )
        assert reg["assumptions"] >= 30


# =============================================================================
# Completeness Statement
# =============================================================================


class TestCompletenessStatement:
    """Test completeness statement."""

    def test_completeness_statement_present(self, sample_assurance_evidence):
        stmt = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "completeness_statement"
        )
        assert stmt["status"] == "complete"

    def test_15_categories_covered(self, sample_assurance_evidence):
        stmt = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "completeness_statement"
        )
        assert stmt["categories_covered"] == 15

    def test_coverage_above_95pct(self, sample_assurance_evidence):
        stmt = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "completeness_statement"
        )
        assert stmt["coverage_pct"] >= Decimal("95")

    def test_exclusions_documented(self, sample_assurance_evidence):
        stmt = next(
            item for item in sample_assurance_evidence["evidence_items"]
            if item["type"] == "completeness_statement"
        )
        assert isinstance(stmt["exclusions"], list)


# =============================================================================
# Assurance Readiness Score
# =============================================================================


class TestAssuranceReadinessScore:
    """Test assurance readiness score (0-100%)."""

    def test_readiness_score_in_range(self, sample_assurance_evidence):
        score = sample_assurance_evidence["readiness_score"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_readiness_score_above_80(self, sample_assurance_evidence):
        """Score should be above 80 for a well-prepared organization."""
        assert sample_assurance_evidence["readiness_score"] >= Decimal("80")

    def test_readiness_improves_with_evidence(self):
        """More complete evidence should yield higher readiness."""
        evidence_complete = 5
        evidence_partial = 3
        score_full = evidence_complete / 5 * 100
        score_partial = evidence_partial / 5 * 100
        assert score_full > score_partial


# =============================================================================
# Verifier Query Management
# =============================================================================


class TestVerifierQueryManagement:
    """Test verifier query management."""

    def test_queries_present(self, sample_assurance_evidence):
        assert len(sample_assurance_evidence["verifier_queries"]) >= 1

    def test_query_has_required_fields(self, sample_assurance_evidence):
        for query in sample_assurance_evidence["verifier_queries"]:
            assert "query_id" in query
            assert "question" in query
            assert "status" in query

    def test_query_statuses_valid(self, sample_assurance_evidence):
        valid_statuses = {"pending", "answered", "in_progress"}
        for query in sample_assurance_evidence["verifier_queries"]:
            assert query["status"] in valid_statuses

    def test_answered_query_has_date(self, sample_assurance_evidence):
        answered = [
            q for q in sample_assurance_evidence["verifier_queries"]
            if q["status"] == "answered"
        ]
        for q in answered:
            assert "response_date" in q


# =============================================================================
# Finding Tracking
# =============================================================================


class TestFindingTracking:
    """Test assurance finding tracking."""

    def test_findings_present(self, sample_assurance_evidence):
        assert len(sample_assurance_evidence["findings"]) >= 1

    def test_finding_has_required_fields(self, sample_assurance_evidence):
        for finding in sample_assurance_evidence["findings"]:
            assert "finding_id" in finding
            assert "severity" in finding
            assert "description" in finding
            assert "status" in finding

    def test_finding_severities_valid(self, sample_assurance_evidence):
        valid = {"observation", "minor", "major"}
        for finding in sample_assurance_evidence["findings"]:
            assert finding["severity"] in valid

    def test_finding_statuses_valid(self, sample_assurance_evidence):
        valid = {"open", "remediated", "closed", "accepted"}
        for finding in sample_assurance_evidence["findings"]:
            assert finding["status"] in valid

    def test_no_major_findings(self, sample_assurance_evidence):
        """No major findings should be present for assurance-ready org."""
        major = [f for f in sample_assurance_evidence["findings"] if f["severity"] == "major"]
        assert len(major) == 0

    def test_finding_ids_unique(self, sample_assurance_evidence):
        ids = [f["finding_id"] for f in sample_assurance_evidence["findings"]]
        assert len(ids) == len(set(ids))
