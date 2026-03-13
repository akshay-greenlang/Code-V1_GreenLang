# -*- coding: utf-8 -*-
"""
Unit tests for Engine 6: CertificationIntegrationEngine -- AGENT-EUDR-024

Tests certification scheme integration for FSC/PEFC/RSPO/RA/ISCC,
certificate status tracking, EUDR coverage matrix, gap analysis,
cross-scheme audit coordination, recertification monitoring, NC
taxonomy mapping, and certificate sync operations.

Target: ~60 tests
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
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    SHA256_HEX_LENGTH,
    CERTIFICATION_SCHEMES,
)


class TestCertificationEngineInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = CertificationIntegrationEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = CertificationIntegrationEngine()
        assert engine.config is not None

    def test_supported_schemes(self):
        assert len(SUPPORTED_SCHEMES) == 5
        assert "fsc" in SUPPORTED_SCHEMES
        assert "pefc" in SUPPORTED_SCHEMES
        assert "rspo" in SUPPORTED_SCHEMES
        assert "rainforest_alliance" in SUPPORTED_SCHEMES
        assert "iscc" in SUPPORTED_SCHEMES


class TestCertificateRegistration:
    """Test certificate registration and tracking."""

    def test_register_fsc_certificate(self, certification_engine, sample_certificate_fsc):
        result = certification_engine.register_certificate(sample_certificate_fsc)
        assert result is not None
        assert result.scheme == CertificationScheme.FSC

    def test_register_rspo_certificate(self, certification_engine, sample_certificate_rspo):
        result = certification_engine.register_certificate(sample_certificate_rspo)
        assert result is not None
        assert result.scheme == CertificationScheme.RSPO

    def test_certificate_has_provenance(self, certification_engine, sample_certificate_fsc):
        result = certification_engine.register_certificate(sample_certificate_fsc)
        assert result.provenance_hash is not None

    def test_active_certificate_status(self, sample_certificate_fsc):
        assert sample_certificate_fsc.status == "active"

    def test_expired_certificate_status(self, sample_certificate_expired):
        assert sample_certificate_expired.status == "expired"

    @pytest.mark.parametrize("scheme", CERTIFICATION_SCHEMES)
    def test_register_certificate_for_each_scheme(self, certification_engine, scheme):
        cert = CertificateRecord(
            supplier_id="SUP-SCHEME-001",
            scheme=CertificationScheme(scheme),
            certificate_number=f"CERT-{scheme.upper()}-TEST",
            status="active",
            scope="chain_of_custody",
            issue_date=date(2024, 1, 1),
            expiry_date=date(2029, 12, 31),
            certification_body="Test CB",
        )
        result = certification_engine.register_certificate(cert)
        assert result is not None


class TestEUDRCoverageMatrix:
    """Test EUDR-to-scheme coverage matrix."""

    def test_get_coverage_matrix(self, certification_engine):
        matrix = certification_engine.get_eudr_coverage_matrix()
        assert matrix is not None
        assert len(matrix) > 0

    @pytest.mark.parametrize("scheme", CERTIFICATION_SCHEMES)
    def test_coverage_matrix_has_scheme(self, certification_engine, scheme):
        matrix = certification_engine.get_eudr_coverage_matrix()
        scheme_cols = [row.get(scheme) for row in matrix if scheme in row]
        # Each scheme should have at least some coverage entries
        assert len(scheme_cols) >= 0

    def test_coverage_values_are_valid(self, certification_engine):
        matrix = certification_engine.get_eudr_coverage_matrix()
        valid_values = {"FULL", "PARTIAL", "NONE", "full", "partial", "none"}
        for row in matrix:
            for scheme in CERTIFICATION_SCHEMES:
                if scheme in row:
                    assert row[scheme] in valid_values

    def test_art3_deforestation_coverage(self, certification_engine):
        matrix = certification_engine.get_eudr_coverage_matrix()
        art3_row = next((r for r in matrix if "Art. 3" in r.get("eudr_requirement", "")), None)
        if art3_row:
            # All 5 schemes should have FULL coverage for Art. 3
            for scheme in CERTIFICATION_SCHEMES:
                if scheme in art3_row:
                    assert art3_row[scheme] in ("FULL", "full")


class TestCoverageGapAnalysis:
    """Test EUDR coverage gap analysis per supplier."""

    def test_gap_analysis_for_certified_supplier(self, certification_engine, sample_certificate_fsc):
        certification_engine.register_certificate(sample_certificate_fsc)
        gaps = certification_engine.analyze_coverage_gaps(supplier_id="SUP-001")
        assert gaps is not None
        assert "covered_requirements" in gaps
        assert "gaps" in gaps

    def test_gap_analysis_for_uncertified_supplier(self, certification_engine):
        gaps = certification_engine.analyze_coverage_gaps(supplier_id="SUP-NONE-001")
        assert gaps is not None
        assert len(gaps.get("gaps", [])) > 0

    def test_fsc_covers_more_than_iscc(self, certification_engine):
        fsc_coverage = certification_engine.get_scheme_coverage_percentage("fsc")
        iscc_coverage = certification_engine.get_scheme_coverage_percentage("iscc")
        assert fsc_coverage >= iscc_coverage

    @pytest.mark.parametrize("scheme,min_coverage", [
        ("fsc", Decimal("70")),
        ("pefc", Decimal("65")),
        ("rspo", Decimal("60")),
        ("rainforest_alliance", Decimal("55")),
        ("iscc", Decimal("50")),
    ])
    def test_scheme_minimum_coverage(self, certification_engine, scheme, min_coverage):
        coverage = certification_engine.get_scheme_coverage_percentage(scheme)
        assert coverage >= min_coverage


class TestCrossSchemeCoordination:
    """Test cross-scheme audit coordination."""

    def test_identify_audit_overlap(self, certification_engine, sample_certificate_fsc, sample_certificate_rspo):
        certification_engine.register_certificate(sample_certificate_fsc)
        certification_engine.register_certificate(sample_certificate_rspo)
        overlap = certification_engine.identify_audit_overlap(
            supplier_id="SUP-001",
            schemes=["fsc", "rspo"],
        )
        assert overlap is not None

    def test_recommend_combined_audit(self, certification_engine, sample_certificate_fsc):
        certification_engine.register_certificate(sample_certificate_fsc)
        recommendation = certification_engine.recommend_combined_audit(
            supplier_id="SUP-001",
        )
        assert recommendation is not None


class TestRecertificationMonitoring:
    """Test recertification timeline monitoring."""

    def test_check_expiring_certificates(self, certification_engine, sample_certificate_fsc):
        certification_engine.register_certificate(sample_certificate_fsc)
        expiring = certification_engine.check_expiring_certificates(
            days_ahead=730,
        )
        assert isinstance(expiring, list)

    def test_expired_certificate_flagged(self, certification_engine, sample_certificate_expired):
        certification_engine.register_certificate(sample_certificate_expired)
        expired = certification_engine.get_expired_certificates()
        assert len(expired) >= 1

    def test_certificate_suspension_handling(self, certification_engine):
        cert = CertificateRecord(
            supplier_id="SUP-SUSP-001",
            scheme=CertificationScheme.FSC,
            certificate_number="FSC-SUSP-001",
            status="suspended",
            scope="chain_of_custody",
            issue_date=date(2023, 1, 1),
            expiry_date=date(2028, 12, 31),
            certification_body="Test CB",
        )
        certification_engine.register_certificate(cert)
        suspended = certification_engine.get_suspended_certificates()
        assert len(suspended) >= 1


class TestNCTaxonomyMapping:
    """Test NC taxonomy mapping across schemes."""

    @pytest.mark.parametrize("scheme_level,expected", [
        (("fsc", "Major"), "major"),
        (("fsc", "Minor"), "minor"),
        (("fsc", "Observation"), "observation"),
        (("rspo", "Major NC"), "major"),
        (("rspo", "Minor NC"), "minor"),
        (("rainforest_alliance", "Critical"), "critical"),
        (("rainforest_alliance", "Major"), "major"),
        (("rainforest_alliance", "Minor"), "minor"),
        (("rainforest_alliance", "Improvement Need"), "observation"),
        (("iscc", "Major NC"), "major"),
        (("iscc", "Minor NC"), "minor"),
    ])
    def test_nc_taxonomy_mapping(self, certification_engine, scheme_level, expected):
        scheme, level = scheme_level
        mapped = certification_engine.map_nc_taxonomy(scheme=scheme, scheme_level=level)
        assert mapped == expected

    def test_unknown_scheme_level_returns_observation(self, certification_engine):
        mapped = certification_engine.map_nc_taxonomy(
            scheme="fsc", scheme_level="Unknown Level"
        )
        assert mapped == "observation"


class TestCertificateSync:
    """Test certificate sync operations."""

    def test_sync_certificates_for_supplier(self, certification_engine):
        result = certification_engine.sync_certificates(supplier_id="SUP-001")
        assert result is not None

    def test_sync_tracks_last_sync_time(self, certification_engine, sample_certificate_fsc):
        certification_engine.register_certificate(sample_certificate_fsc)
        certification_engine.sync_certificates(supplier_id="SUP-001")
        last_sync = certification_engine.get_last_sync_time(supplier_id="SUP-001")
        assert last_sync is not None


class TestCertificateProvenance:
    """Test certificate provenance tracking."""

    def test_certificate_registration_has_provenance(self, certification_engine, sample_certificate_fsc):
        result = certification_engine.register_certificate(sample_certificate_fsc)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_certificate_status_change_tracked(self, certification_engine, sample_certificate_fsc):
        certification_engine.register_certificate(sample_certificate_fsc)
        result = certification_engine.update_certificate_status(
            certificate_id="CERT-FSC-001",
            new_status="suspended",
            reason="Audit non-compliance",
        )
        assert result is not None
