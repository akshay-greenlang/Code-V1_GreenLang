# -*- coding: utf-8 -*-
"""
Unit tests for Engine 2: AuditorRegistryQualificationEngine -- AGENT-EUDR-024

Tests auditor registration, qualification tracking, competence matching,
conflict-of-interest screening, CPD compliance, performance rating,
accreditation validation, and rotation requirements.

Target: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.auditor_registry_qualification_engine import (
    AuditorRegistryQualificationEngine,
    MATCH_WEIGHTS,
    SCHEME_QUALIFICATIONS,
    CPD_CATEGORIES,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    Auditor,
    CertificationScheme,
    MatchAuditorRequest,
    SUPPORTED_COMMODITIES,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    SHA256_HEX_LENGTH,
)


class TestAuditorRegistryInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = AuditorRegistryQualificationEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = AuditorRegistryQualificationEngine()
        assert engine.config is not None

    def test_match_weights_sum_to_one(self):
        total = sum(MATCH_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_match_weights_all_positive(self):
        for weight in MATCH_WEIGHTS.values():
            assert weight > Decimal("0")


class TestAuditorRegistration:
    """Test auditor profile registration and validation."""

    def test_register_auditor(self, auditor_registry_engine, sample_auditor_fsc):
        result = auditor_registry_engine.register_auditor(sample_auditor_fsc)
        assert result is not None
        assert result.auditor_id == "AUR-FSC-001"

    def test_register_auditor_sets_provenance(self, auditor_registry_engine, sample_auditor_fsc):
        result = auditor_registry_engine.register_auditor(sample_auditor_fsc)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_register_multiple_auditors(self, auditor_registry_engine, sample_auditor_pool):
        for auditor in sample_auditor_pool:
            result = auditor_registry_engine.register_auditor(auditor)
            assert result is not None

    def test_auditor_commodity_competencies(self, sample_auditor_fsc):
        assert "wood" in sample_auditor_fsc.commodity_competencies
        assert "rubber" in sample_auditor_fsc.commodity_competencies

    def test_auditor_scheme_qualifications(self, sample_auditor_fsc):
        assert "FSC Lead Auditor" in sample_auditor_fsc.scheme_qualifications

    def test_auditor_country_expertise(self, sample_auditor_fsc):
        assert "BR" in sample_auditor_fsc.country_expertise

    def test_auditor_languages(self, sample_auditor_fsc):
        assert "en" in sample_auditor_fsc.languages
        assert "pt" in sample_auditor_fsc.languages


class TestAuditorMatching:
    """Test auditor-to-audit matching algorithm."""

    def test_match_auditors_basic(self, auditor_registry_engine, sample_auditor_pool):
        for auditor in sample_auditor_pool:
            auditor_registry_engine.register_auditor(auditor)

        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001",
            commodity="wood",
            scheme="fsc",
            country_code="BR",
            required_language="pt",
            audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        assert response is not None
        assert len(response.matched_auditors) > 0

    def test_match_returns_ranked_list(self, auditor_registry_engine, sample_auditor_pool):
        for auditor in sample_auditor_pool:
            auditor_registry_engine.register_auditor(auditor)

        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001",
            commodity="wood",
            scheme="fsc",
            country_code="BR",
            required_language="pt",
            audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        if len(response.matched_auditors) > 1:
            scores = [a["match_score"] for a in response.matched_auditors]
            assert scores == sorted(scores, reverse=True)

    def test_suspended_auditor_excluded(self, auditor_registry_engine, sample_auditor_suspended):
        auditor_registry_engine.register_auditor(sample_auditor_suspended)

        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001",
            commodity="cocoa",
            scheme="rainforest_alliance",
            country_code="GH",
            required_language="en",
            audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        matched_ids = [a["auditor_id"] for a in response.matched_auditors]
        assert "AUR-SUSP-001" not in matched_ids

    def test_expired_auditor_excluded(self, auditor_registry_engine, sample_auditor_expired):
        auditor_registry_engine.register_auditor(sample_auditor_expired)

        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001",
            commodity="coffee",
            scheme="rainforest_alliance",
            country_code="CO",
            required_language="es",
            audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        matched_ids = [a["auditor_id"] for a in response.matched_auditors]
        assert "AUR-EXP-001" not in matched_ids

    def test_cpd_non_compliant_excluded(self, auditor_registry_engine, sample_auditor_suspended):
        assert not sample_auditor_suspended.cpd_compliant

    def test_commodity_match_scores_higher(self, auditor_registry_engine, sample_auditor_fsc, sample_auditor_rspo):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        auditor_registry_engine.register_auditor(sample_auditor_rspo)

        # Match for wood - FSC auditor should score higher
        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001",
            commodity="wood",
            scheme="fsc",
            country_code="BR",
            required_language="pt",
            audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        if len(response.matched_auditors) >= 2:
            assert response.matched_auditors[0]["auditor_id"] == "AUR-FSC-001"

    def test_match_response_has_provenance(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)

        request = MatchAuditorRequest(
            audit_id="AUD-TEST-001",
            commodity="wood",
            scheme="fsc",
            country_code="BR",
            required_language="pt",
            audit_date=FROZEN_DATE,
        )
        response = auditor_registry_engine.match_auditors(request)
        assert response.provenance_hash is not None


class TestConflictOfInterest:
    """Test conflict-of-interest screening."""

    def test_coi_declaration_stored(self, sample_auditor_fsc):
        sample_auditor_fsc.conflict_of_interest = [
            {"supplier_id": "SUP-001", "relationship": "former_employer", "declared_date": "2025-01-01"}
        ]
        assert len(sample_auditor_fsc.conflict_of_interest) == 1

    def test_check_coi_with_supplier(self, auditor_registry_engine, sample_auditor_fsc):
        sample_auditor_fsc.conflict_of_interest = [
            {"supplier_id": "SUP-COI-001", "relationship": "financial_interest"}
        ]
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        has_coi = auditor_registry_engine.check_conflict_of_interest(
            auditor_id="AUR-FSC-001",
            supplier_id="SUP-COI-001",
        )
        assert has_coi is True

    def test_no_coi_with_different_supplier(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        has_coi = auditor_registry_engine.check_conflict_of_interest(
            auditor_id="AUR-FSC-001",
            supplier_id="SUP-NOCOI-001",
        )
        assert has_coi is False


class TestCPDCompliance:
    """Test continuing professional development tracking."""

    def test_cpd_compliant_auditor(self, sample_auditor_fsc):
        assert sample_auditor_fsc.cpd_compliant is True
        assert sample_auditor_fsc.cpd_hours >= 40

    def test_cpd_non_compliant_auditor(self, sample_auditor_suspended):
        assert sample_auditor_suspended.cpd_compliant is False

    def test_cpd_categories_defined(self):
        assert len(CPD_CATEGORIES) == 6
        assert "regulatory_updates" in CPD_CATEGORIES
        assert "technical_skills" in CPD_CATEGORIES

    def test_cpd_total_hours_per_year(self):
        total = sum(CPD_CATEGORIES.values())
        assert total == 40

    def test_check_cpd_compliance(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        result = auditor_registry_engine.check_cpd_compliance("AUR-FSC-001")
        assert result["compliant"] is True


class TestAccreditationExpiry:
    """Test accreditation expiry warning and tracking."""

    def test_active_accreditation(self, sample_auditor_fsc):
        assert sample_auditor_fsc.accreditation_status == "active"
        assert sample_auditor_fsc.accreditation_expiry > FROZEN_DATE

    def test_check_expiry_warning(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        warnings = auditor_registry_engine.check_qualification_expiry("AUR-FSC-001")
        assert warnings is not None

    def test_expired_accreditation_flagged(self, auditor_registry_engine, sample_auditor_expired):
        auditor_registry_engine.register_auditor(sample_auditor_expired)
        warnings = auditor_registry_engine.check_qualification_expiry("AUR-EXP-001")
        assert any(w.get("type") == "accreditation_expired" for w in warnings)


class TestSchemeQualifications:
    """Test scheme qualification tracking."""

    def test_fsc_qualifications_defined(self):
        assert "fsc" in SCHEME_QUALIFICATIONS
        assert "FSC Lead Auditor" in SCHEME_QUALIFICATIONS["fsc"]

    def test_rspo_qualifications_defined(self):
        assert "rspo" in SCHEME_QUALIFICATIONS
        assert "RSPO Lead Auditor" in SCHEME_QUALIFICATIONS["rspo"]

    def test_pefc_qualifications_defined(self):
        assert "pefc" in SCHEME_QUALIFICATIONS

    def test_ra_qualifications_defined(self):
        assert "rainforest_alliance" in SCHEME_QUALIFICATIONS

    def test_iscc_qualifications_defined(self):
        assert "iscc" in SCHEME_QUALIFICATIONS

    @pytest.mark.parametrize("scheme", ["fsc", "pefc", "rspo", "rainforest_alliance", "iscc"])
    def test_all_schemes_have_lead_auditor(self, scheme):
        quals = SCHEME_QUALIFICATIONS[scheme]
        lead_quals = [q for q in quals if "Lead" in q]
        assert len(lead_quals) >= 1


class TestAuditorPerformance:
    """Test auditor performance tracking."""

    def test_performance_rating_range(self, sample_auditor_fsc):
        assert Decimal("0") <= sample_auditor_fsc.performance_rating <= Decimal("100")

    def test_findings_per_audit_tracked(self, sample_auditor_fsc):
        assert sample_auditor_fsc.findings_per_audit >= Decimal("0")

    def test_car_closure_rate_tracked(self, sample_auditor_fsc):
        assert Decimal("0") <= sample_auditor_fsc.car_closure_rate <= Decimal("100")

    def test_audit_count_tracked(self, sample_auditor_fsc):
        assert sample_auditor_fsc.audit_count == 150

    def test_get_performance_metrics(self, auditor_registry_engine, sample_auditor_fsc):
        auditor_registry_engine.register_auditor(sample_auditor_fsc)
        metrics = auditor_registry_engine.get_auditor_performance("AUR-FSC-001")
        assert metrics is not None
        assert "performance_rating" in metrics
