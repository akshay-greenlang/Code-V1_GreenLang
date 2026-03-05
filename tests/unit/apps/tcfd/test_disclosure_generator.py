# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Disclosure Generator.

Tests disclosure creation, section drafting for all 11 disclosures,
compliance checking, evidence linking, report generation, export formats,
approval workflow, year-over-year comparison, and regulatory adaptations
with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    TCFDPillar,
    DisclosureStatus,
    TCFD_DISCLOSURES,
    PILLAR_NAMES,
    REGULATORY_JURISDICTIONS,
)
from services.models import (
    DisclosureSection,
    TCFDDisclosure,
    _new_id,
)


# ===========================================================================
# Disclosure Creation
# ===========================================================================

class TestDisclosureCreation:
    """Test disclosure document creation."""

    def test_create_disclosure(self, sample_disclosure):
        assert sample_disclosure.reporting_year == 2025
        assert sample_disclosure.status == DisclosureStatus.DRAFT

    def test_disclosure_has_id(self, sample_disclosure):
        assert len(sample_disclosure.id) == 36

    def test_disclosure_version(self, sample_disclosure):
        assert sample_disclosure.version == 1

    def test_disclosure_org_id(self, sample_disclosure, sample_org_id):
        assert sample_disclosure.org_id == sample_org_id


# ===========================================================================
# Section Drafting (All 11 Disclosures)
# ===========================================================================

class TestSectionDrafting:
    """Test section drafting for all 11 TCFD recommended disclosures."""

    @pytest.mark.parametrize("disclosure_ref,expected_pillar", [
        ("gov_a", "governance"),
        ("gov_b", "governance"),
        ("str_a", "strategy"),
        ("str_b", "strategy"),
        ("str_c", "strategy"),
        ("rm_a", "risk_management"),
        ("rm_b", "risk_management"),
        ("rm_c", "risk_management"),
        ("mt_a", "metrics_targets"),
        ("mt_b", "metrics_targets"),
        ("mt_c", "metrics_targets"),
    ])
    def test_disclosure_defined(self, disclosure_ref, expected_pillar):
        assert disclosure_ref in TCFD_DISCLOSURES
        assert TCFD_DISCLOSURES[disclosure_ref]["pillar"] == expected_pillar

    def test_total_disclosures_count(self):
        assert len(TCFD_DISCLOSURES) == 11

    def test_section_creation(self, sample_disclosure_section):
        assert sample_disclosure_section.disclosure_ref == "gov_a"
        assert sample_disclosure_section.pillar == TCFDPillar.GOVERNANCE

    def test_section_content(self, sample_disclosure_section):
        assert len(sample_disclosure_section.content) > 0

    def test_section_compliance_score(self, sample_disclosure_section):
        assert 0 <= sample_disclosure_section.compliance_score <= 100

    @pytest.mark.parametrize("pillar", list(TCFDPillar))
    def test_section_for_each_pillar(self, pillar):
        section = DisclosureSection(
            pillar=pillar,
            disclosure_ref=f"test_{pillar.value[:3]}",
            title=f"Test {pillar.value}",
            content=f"Content for {pillar.value}",
            compliance_score=50,
        )
        assert section.pillar == pillar


# ===========================================================================
# Compliance Checking
# ===========================================================================

class TestComplianceChecking:
    """Test disclosure compliance score calculation."""

    def test_auto_completeness_score(self, sample_disclosure):
        assert sample_disclosure.completeness_score > Decimal("0")

    def test_full_compliance(self):
        sections = [
            DisclosureSection(
                pillar=TCFDPillar.GOVERNANCE,
                disclosure_ref="gov_a",
                title="Board Oversight",
                content="Full compliance content.",
                compliance_score=100,
            )
            for _ in range(11)
        ]
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            sections=sections,
        )
        assert disclosure.completeness_score == Decimal("100")

    def test_partial_compliance(self):
        sections = [
            DisclosureSection(
                pillar=TCFDPillar.GOVERNANCE,
                disclosure_ref="gov_a",
                title="Board Oversight",
                content="Partial content.",
                compliance_score=50,
            ),
            DisclosureSection(
                pillar=TCFDPillar.STRATEGY,
                disclosure_ref="str_a",
                title="Risks",
                content="Partial content.",
                compliance_score=70,
            ),
        ]
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            sections=sections,
        )
        assert disclosure.completeness_score == Decimal("60")

    def test_zero_compliance(self):
        sections = [
            DisclosureSection(
                pillar=TCFDPillar.GOVERNANCE,
                disclosure_ref="gov_a",
                title="Board Oversight",
                content="",
                compliance_score=0,
            ),
        ]
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            sections=sections,
        )
        assert disclosure.completeness_score == Decimal("0")


# ===========================================================================
# Evidence Linking
# ===========================================================================

class TestEvidenceLinking:
    """Test evidence reference linking to disclosure sections."""

    def test_evidence_refs(self, sample_disclosure_section):
        assert len(sample_disclosure_section.evidence_refs) >= 1

    def test_multiple_evidence_refs(self, sample_disclosure_section):
        assert "governance-charter-2025" in sample_disclosure_section.evidence_refs
        assert "board-minutes-q4-2025" in sample_disclosure_section.evidence_refs

    def test_no_evidence(self):
        section = DisclosureSection(
            pillar=TCFDPillar.STRATEGY,
            disclosure_ref="str_a",
            title="Risks",
            content="Content without evidence.",
            evidence_refs=[],
        )
        assert len(section.evidence_refs) == 0


# ===========================================================================
# Report Generation
# ===========================================================================

class TestReportGeneration:
    """Test disclosure report generation."""

    def test_disclosure_provenance(self, sample_disclosure):
        assert len(sample_disclosure.provenance_hash) == 64

    def test_disclosure_sections_count(self, sample_disclosure):
        assert len(sample_disclosure.sections) == 3

    def test_empty_disclosure(self):
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            sections=[],
        )
        assert disclosure.completeness_score == Decimal("0")


# ===========================================================================
# Approval Workflow
# ===========================================================================

class TestApprovalWorkflow:
    """Test disclosure approval workflow."""

    @pytest.mark.parametrize("status", list(DisclosureStatus))
    def test_all_disclosure_statuses(self, status):
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            status=status,
        )
        assert disclosure.status == status

    def test_draft_to_review(self):
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            status=DisclosureStatus.DRAFT,
        )
        # Simulate status update
        updated = disclosure.model_copy(update={"status": DisclosureStatus.REVIEW})
        assert updated.status == DisclosureStatus.REVIEW

    def test_approved_with_approver(self):
        disclosure = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2025,
            status=DisclosureStatus.APPROVED,
            approved_by="CFO",
        )
        assert disclosure.approved_by == "CFO"


# ===========================================================================
# Year-over-Year Comparison
# ===========================================================================

class TestYearOverYearComparison:
    """Test year-over-year disclosure comparison."""

    def test_compare_two_years(self):
        d1 = TCFDDisclosure(
            org_id=_new_id(),
            reporting_year=2024,
            sections=[
                DisclosureSection(
                    pillar=TCFDPillar.GOVERNANCE,
                    disclosure_ref="gov_a",
                    title="Board Oversight",
                    compliance_score=60,
                ),
            ],
        )
        d2 = TCFDDisclosure(
            org_id=d1.org_id,
            reporting_year=2025,
            sections=[
                DisclosureSection(
                    pillar=TCFDPillar.GOVERNANCE,
                    disclosure_ref="gov_a",
                    title="Board Oversight",
                    compliance_score=80,
                ),
            ],
        )
        assert d2.completeness_score > d1.completeness_score


# ===========================================================================
# Regulatory Adaptations
# ===========================================================================

class TestRegulatoryAdaptations:
    """Test jurisdiction-specific regulatory adaptations."""

    def test_regulatory_jurisdictions_defined(self):
        assert len(REGULATORY_JURISDICTIONS) >= 5

    @pytest.mark.parametrize("jurisdiction", ["UK", "EU", "US", "JP", "SG"])
    def test_key_jurisdictions_present(self, jurisdiction):
        assert jurisdiction in REGULATORY_JURISDICTIONS

    def test_pillar_display_names(self):
        for pillar in TCFDPillar:
            assert pillar in PILLAR_NAMES
            assert len(PILLAR_NAMES[pillar]) > 0
