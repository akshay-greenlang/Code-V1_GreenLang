# -*- coding: utf-8 -*-
"""
Unit tests for CertificationValidator - AGENT-EUDR-017 Engine 4

Tests comprehensive supplier certification validation covering 8 certification
schemes, expiry monitoring, scope verification, chain-of-custody validation,
volume alignment, and fraud detection.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pytest

from greenlang.agents.eudr.supplier_risk_scorer.certification_validator import (
    CertificationValidator,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    CertificationScheme,
    CertificationStatus,
    CommodityType,
)


class TestCertificationValidatorInit:
    """Tests for CertificationValidator initialization."""

    @pytest.mark.unit
    def test_initialization(self, mock_config):
        validator = CertificationValidator()
        assert validator._certifications == {}


class TestValidateCertification:
    """Tests for validate_certification method."""

    @pytest.mark.unit
    def test_validate_certification_valid(
        self, certification_validator, sample_certification
    ):
        result = certification_validator.validate_certification(
            supplier_id=sample_certification["supplier_id"],
            scheme=sample_certification["scheme"],
            certificate_number=sample_certification["certificate_number"],
            expiry_date=sample_certification["expiry_date"],
        )
        assert result["status"] == CertificationStatus.VALID

    @pytest.mark.unit
    @pytest.mark.parametrize("scheme", [
        CertificationScheme.FSC, CertificationScheme.PEFC,
        CertificationScheme.RSPO, CertificationScheme.RAINFOREST_ALLIANCE,
        CertificationScheme.UTZ, CertificationScheme.ORGANIC,
        CertificationScheme.FAIR_TRADE, CertificationScheme.ISCC
    ])
    def test_validate_all_schemes(self, certification_validator, scheme):
        """Test validation for all 8 supported certification schemes."""
        result = certification_validator.validate_certification(
            supplier_id="SUPP-001",
            scheme=scheme,
            certificate_number=f"{scheme.value}-TEST-001",
            expiry_date=datetime.now(timezone.utc) + timedelta(days=365),
        )
        assert result is not None
        assert result["scheme"] == scheme


class TestCheckExpiryAlerts:
    """Tests for expiry alert checking."""

    @pytest.mark.unit
    def test_check_expiry_warns_within_buffer(
        self, certification_validator, mock_config
    ):
        expiry_date = datetime.now(timezone.utc) + timedelta(days=60)
        result = certification_validator.check_expiry_alert(
            expiry_date=expiry_date,
            buffer_days=mock_config.cert_expiry_buffer_days,
        )
        assert result["alert"] is True
        assert result["days_until_expiry"] < mock_config.cert_expiry_buffer_days


class TestVerifyScope:
    """Tests for scope verification."""

    @pytest.mark.unit
    def test_verify_scope_matching_commodity(self, certification_validator):
        result = certification_validator.verify_scope(
            scheme=CertificationScheme.FSC,
            certificate_scope=["wood"],
            supplier_commodity=CommodityType.WOOD,
        )
        assert result["scope_valid"] is True

    @pytest.mark.unit
    def test_verify_scope_mismatched_commodity(self, certification_validator):
        result = certification_validator.verify_scope(
            scheme=CertificationScheme.RSPO,
            certificate_scope=["oil_palm"],
            supplier_commodity=CommodityType.SOYA,
        )
        assert result["scope_valid"] is False


class TestVerifyChainOfCustody:
    """Tests for chain-of-custody verification."""

    @pytest.mark.unit
    def test_verify_coc_fsc(self, certification_validator):
        result = certification_validator.verify_chain_of_custody(
            scheme=CertificationScheme.FSC,
            coc_type="FSC-COC",
            certificate_number="FSC-C123456",
        )
        assert "coc_valid" in result


class TestAggregateScore:
    """Tests for aggregate certification scoring."""

    @pytest.mark.unit
    def test_aggregate_score_multiple_certifications(
        self, certification_validator
    ):
        certifications = [
            {
                "scheme": CertificationScheme.FSC,
                "status": CertificationStatus.VALID,
                "expiry_date": datetime.now(timezone.utc) + timedelta(days=365),
            },
            {
                "scheme": CertificationScheme.PEFC,
                "status": CertificationStatus.VALID,
                "expiry_date": datetime.now(timezone.utc) + timedelta(days=180),
            },
        ]
        score = certification_validator.aggregate_certification_score(certifications)
        assert Decimal("0.0") <= score <= Decimal("100.0")


class TestVolumeAlignment:
    """Tests for volume alignment checking."""

    @pytest.mark.unit
    def test_volume_alignment_check(self, certification_validator):
        result = certification_validator.check_volume_alignment(
            certified_volume=8000.0,
            total_volume=10000.0,
            claimed_percentage=80.0,
        )
        assert result["aligned"] is True

    @pytest.mark.unit
    def test_volume_alignment_mismatch(self, certification_validator):
        result = certification_validator.check_volume_alignment(
            certified_volume=5000.0,
            total_volume=10000.0,
            claimed_percentage=80.0,  # Claims 80% but only 50% certified
        )
        assert result["aligned"] is False


class TestFraudDetection:
    """Tests for fraud detection."""

    @pytest.mark.unit
    def test_detect_fraud_indicators(self, certification_validator):
        indicators = certification_validator.detect_fraud_indicators(
            certificate_number="FAKE-CERT-001",
            certification_body="Unknown Body",
            issue_date=datetime.now(timezone.utc) + timedelta(days=1),  # Future date
        )
        assert len(indicators) > 0


class TestSchemeEquivalence:
    """Tests for scheme equivalence mapping."""

    @pytest.mark.unit
    def test_check_scheme_equivalence(self, certification_validator):
        result = certification_validator.check_scheme_equivalence(
            scheme1=CertificationScheme.FSC,
            scheme2=CertificationScheme.PEFC,
        )
        assert "equivalent" in result


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_validation_includes_provenance_hash(
        self, certification_validator, sample_certification
    ):
        result = certification_validator.validate_certification(
            supplier_id=sample_certification["supplier_id"],
            scheme=sample_certification["scheme"],
            certificate_number=sample_certification["certificate_number"],
            expiry_date=sample_certification["expiry_date"],
        )
        assert "provenance_hash" in result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_certificate_number_raises_error(self, certification_validator):
        with pytest.raises(ValueError):
            certification_validator.validate_certification(
                supplier_id="SUPP-001",
                scheme=CertificationScheme.FSC,
                certificate_number="",  # Empty
                expiry_date=datetime.now(timezone.utc) + timedelta(days=365),
            )
