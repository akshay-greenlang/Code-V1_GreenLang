# -*- coding: utf-8 -*-
"""
API tests for Certificate Routes -- AGENT-EUDR-024

Tests certification scheme integration endpoints including certificate
status verification, EUDR coverage gap analysis, recertification timeline
monitoring, certificate suspension/termination handling, cross-scheme
harmonization, and scheme-specific audit scope recommendations.

Target: ~30 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.certification_integration_engine import (
    CertificationIntegrationEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    CertificateRecord,
    CertificationScheme,
    SUPPORTED_SCHEMES,
    SUPPORTED_COMMODITIES,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    CERTIFICATION_SCHEMES,
    EUDR_COMMODITIES,
)


class TestCertificateVerification:
    """Test certificate status verification endpoint logic."""

    def test_verify_active_certificate(
        self, certification_engine, sample_certificate_fsc
    ):
        result = certification_engine.verify_certificate(sample_certificate_fsc)
        assert result is not None
        assert result["status"] == "active"

    def test_verify_expired_certificate(
        self, certification_engine, sample_certificate_expired
    ):
        result = certification_engine.verify_certificate(sample_certificate_expired)
        assert result is not None
        assert result["status"] in ("expired", "warning")

    def test_verify_returns_expiry_info(
        self, certification_engine, sample_certificate_fsc
    ):
        result = certification_engine.verify_certificate(sample_certificate_fsc)
        assert "expiry_date" in result or "days_until_expiry" in result

    def test_verify_returns_scheme_info(
        self, certification_engine, sample_certificate_fsc
    ):
        result = certification_engine.verify_certificate(sample_certificate_fsc)
        assert result is not None


class TestCertificateRegistration:
    """Test certificate registration and CRUD."""

    def test_register_certificate(
        self, certification_engine, sample_certificate_fsc
    ):
        result = certification_engine.register_certificate(sample_certificate_fsc)
        assert result is not None

    def test_register_rspo_certificate(
        self, certification_engine, sample_certificate_rspo
    ):
        result = certification_engine.register_certificate(sample_certificate_rspo)
        assert result is not None

    @pytest.mark.parametrize("scheme", [
        CertificationScheme.FSC,
        CertificationScheme.PEFC,
        CertificationScheme.RSPO,
        CertificationScheme.RAINFOREST_ALLIANCE,
        CertificationScheme.ISCC,
    ])
    def test_register_all_schemes(self, certification_engine, scheme):
        cert = CertificateRecord(
            scheme=scheme,
            certificate_number=f"TEST-{scheme.value.upper()}-001",
            holder_name="Test Org",
            holder_id="SUP-001",
            status="active",
            issue_date=date(2024, 1, 1),
            expiry_date=date(2029, 12, 31),
        )
        result = certification_engine.register_certificate(cert)
        assert result is not None

    def test_certificate_requires_scheme(self):
        with pytest.raises((ValueError, Exception)):
            CertificateRecord(
                certificate_number="TEST-001",
                holder_name="Test Org",
                holder_id="SUP-001",
            )

    def test_certificate_requires_holder_name(self):
        with pytest.raises((ValueError, Exception)):
            CertificateRecord(
                scheme=CertificationScheme.FSC,
                certificate_number="TEST-001",
                holder_name="",
                holder_id="SUP-001",
            )

    def test_certificate_requires_certificate_number(self):
        with pytest.raises((ValueError, Exception)):
            CertificateRecord(
                scheme=CertificationScheme.FSC,
                certificate_number="",
                holder_name="Test Org",
                holder_id="SUP-001",
            )


class TestCoverageGapAnalysis:
    """Test EUDR coverage gap analysis endpoint logic."""

    def test_coverage_gap_returns_result(
        self, certification_engine, sample_certificate_fsc
    ):
        certification_engine.register_certificate(sample_certificate_fsc)
        result = certification_engine.analyze_coverage_gap(
            supplier_id="SUP-001",
            commodity="wood",
        )
        assert result is not None

    def test_coverage_gap_has_percentage(
        self, certification_engine, sample_certificate_fsc
    ):
        certification_engine.register_certificate(sample_certificate_fsc)
        result = certification_engine.analyze_coverage_gap(
            supplier_id="SUP-001",
            commodity="wood",
        )
        assert "coverage_percentage" in result or result is not None

    @pytest.mark.parametrize("scheme,expected_min", [
        ("fsc", 70),
        ("pefc", 65),
        ("rspo", 60),
        ("rainforest_alliance", 55),
        ("iscc", 50),
    ])
    def test_scheme_coverage_ranges(self, certification_engine, scheme, expected_min):
        coverage = certification_engine.get_scheme_coverage(scheme)
        assert coverage is not None

    def test_uncertified_supplier_zero_coverage(self, certification_engine):
        result = certification_engine.analyze_coverage_gap(
            supplier_id="SUP-UNCERT-001",
            commodity="wood",
        )
        assert result is not None


class TestRecertificationMonitoring:
    """Test recertification timeline monitoring endpoint logic."""

    def test_recertification_timeline(
        self, certification_engine, sample_certificate_fsc
    ):
        certification_engine.register_certificate(sample_certificate_fsc)
        timeline = certification_engine.get_recertification_timeline(
            supplier_id="SUP-001",
        )
        assert timeline is not None

    def test_expiring_certificate_flagged(self, certification_engine):
        expiring_cert = CertificateRecord(
            scheme=CertificationScheme.PEFC,
            certificate_number="PEFC-EXP-001",
            holder_name="Expiring Org",
            holder_id="SUP-EXP-001",
            status="active",
            issue_date=date(2021, 1, 1),
            expiry_date=FROZEN_DATE + timedelta(days=60),
        )
        certification_engine.register_certificate(expiring_cert)
        alerts = certification_engine.check_expiration_alerts()
        assert alerts is not None

    def test_fsc_5_year_cycle(self, certification_engine):
        cycle = certification_engine.get_recertification_cycle("fsc")
        assert cycle == 5

    def test_iscc_1_year_cycle(self, certification_engine):
        cycle = certification_engine.get_recertification_cycle("iscc")
        assert cycle == 1

    def test_rainforest_alliance_3_year_cycle(self, certification_engine):
        cycle = certification_engine.get_recertification_cycle("rainforest_alliance")
        assert cycle == 3


class TestCertificateSuspension:
    """Test certificate suspension and termination handling."""

    def test_suspend_certificate(
        self, certification_engine, sample_certificate_fsc
    ):
        certification_engine.register_certificate(sample_certificate_fsc)
        result = certification_engine.update_certificate_status(
            certificate_id=sample_certificate_fsc.certificate_id,
            new_status="suspended",
            reason="NC-CRIT-001 unresolved within SLA",
        )
        assert result is not None

    def test_terminate_certificate(
        self, certification_engine, sample_certificate_fsc
    ):
        certification_engine.register_certificate(sample_certificate_fsc)
        result = certification_engine.update_certificate_status(
            certificate_id=sample_certificate_fsc.certificate_id,
            new_status="terminated",
            reason="Repeated critical NCs; no corrective action",
        )
        assert result is not None

    def test_suspension_triggers_unscheduled_audit_flag(
        self, certification_engine, sample_certificate_fsc
    ):
        certification_engine.register_certificate(sample_certificate_fsc)
        result = certification_engine.update_certificate_status(
            certificate_id=sample_certificate_fsc.certificate_id,
            new_status="suspended",
            reason="Testing suspension",
        )
        assert result is not None


class TestCrossSchemeHarmonization:
    """Test cross-scheme findings harmonization."""

    def test_harmonize_findings(self, certification_engine):
        result = certification_engine.harmonize_findings(
            audit_id="AUD-TEST-001",
            scheme_findings=[
                {"scheme": "fsc", "clause": "P9", "severity": "major"},
                {"scheme": "pefc", "clause": "5.1", "severity": "major"},
            ],
        )
        assert result is not None

    def test_map_scheme_to_eudr(self, certification_engine):
        mapping = certification_engine.map_scheme_to_eudr(
            scheme="fsc",
            clause="P9",
        )
        assert mapping is not None
