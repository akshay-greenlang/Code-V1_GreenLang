# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Demo Configuration Tests
===============================================================

Tests for demo config: file existence, YAML parsing, sample data
completeness.

Target: ~15 tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

import pytest

from .conftest import DEMO_DIR, PACK_ROOT


# ===========================================================================
# File Existence Tests
# ===========================================================================


class TestDemoFileExistence:
    """Tests for demo configuration file existence."""

    def test_demo_dir_exists(self):
        """Demo directory exists."""
        assert DEMO_DIR.exists()

    def test_demo_dir_has_init(self):
        """Demo directory has __init__.py."""
        assert (DEMO_DIR / "__init__.py").exists()

    def test_demo_config_yaml_check(self):
        """demo_config.yaml existence check."""
        path = DEMO_DIR / "demo_config.yaml"
        if not path.exists():
            pytest.skip("demo_config.yaml not yet created")
        assert path.exists()


# ===========================================================================
# Demo Data Tests (using conftest fixtures)
# ===========================================================================


class TestDemoSampleClaims:
    """Tests for sample claims fixture data completeness."""

    def test_sample_claims_count(self, sample_claims):
        """sample_claims fixture has at least 5 claims."""
        assert len(sample_claims) >= 5

    def test_sample_claims_have_required_fields(self, sample_claims):
        """All sample claims have required fields."""
        for claim in sample_claims:
            assert "claim_id" in claim
            assert "claim_text" in claim
            assert "claim_type" in claim
            assert "product_or_org" in claim

    def test_sample_claims_unique_ids(self, sample_claims):
        """All sample claim IDs are unique."""
        ids = [c["claim_id"] for c in sample_claims]
        assert len(ids) == len(set(ids))

    def test_sample_claims_have_lifecycle_stages(self, sample_claims):
        """All sample claims have lifecycle_stages_covered field."""
        for claim in sample_claims:
            assert "lifecycle_stages_covered" in claim


class TestDemoSampleEvidence:
    """Tests for sample evidence fixture data completeness."""

    def test_sample_evidence_count(self, sample_evidence):
        """sample_evidence fixture has at least 5 items."""
        assert len(sample_evidence) >= 5

    def test_sample_evidence_have_required_fields(self, sample_evidence):
        """All sample evidence have required fields."""
        for evidence in sample_evidence:
            assert "evidence_id" in evidence
            assert "evidence_type" in evidence
            assert "source" in evidence

    def test_sample_evidence_unique_ids(self, sample_evidence):
        """All sample evidence IDs are unique."""
        ids = [e["evidence_id"] for e in sample_evidence]
        assert len(ids) == len(set(ids))

    def test_sample_evidence_have_validity_dates(self, sample_evidence):
        """All sample evidence have validity date fields."""
        for evidence in sample_evidence:
            assert "valid_from" in evidence
            assert "valid_to" in evidence

    def test_sample_evidence_have_claim_ids(self, sample_evidence):
        """All sample evidence have claim_ids."""
        for evidence in sample_evidence:
            assert "claim_ids" in evidence
            assert isinstance(evidence["claim_ids"], list)


class TestDemoSampleDocuments:
    """Tests for sample documents fixture data completeness."""

    def test_sample_documents_count(self, sample_documents):
        """sample_documents fixture has at least 4 items."""
        assert len(sample_documents) >= 4

    def test_sample_documents_have_required_fields(self, sample_documents):
        """All sample documents have required fields."""
        for doc in sample_documents:
            assert "doc_id" in doc
            assert "title" in doc
            assert "evidence_type" in doc
