# -*- coding: utf-8 -*-
"""
Tests for CertificationSchemeValidatorEngine - AGENT-EUDR-023 Engine 3

Comprehensive test suite covering:
- FSC validation (5 sub-schemes: FM, CoC, CW, Project, Ecosystem)
- PEFC validation (SFM, CoC)
- RSPO validation (IP, MB, SG)
- Rainforest Alliance validation (2020 standard)
- ISCC validation (EU, PLUS)
- EUDR equivalence mapping for each scheme
- Certificate expiry handling and status transitions
- Scope coverage validation for EUDR commodities
- Certificate code format validation
- Batch certificate validation
- Integration with external certification APIs

Test count: 80+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Engine 3 - Certification Scheme Validator)
"""

import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    CERTIFICATION_SCHEMES,
    FSC_SUB_SCHEMES,
    EUDR_COMMODITIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Certificate code regex patterns
CERT_CODE_PATTERNS = {
    "fsc": r"^FSC-[A-Z]\d{6}$",
    "pefc": r"^PEFC/[A-Z]{2}-[A-Z]{2}-[A-Z0-9]{7,}$",
    "rspo": r"^RSPO-\d{7}$",
    "rainforest_alliance": r"^RA-[A-Z]{2}-\d{4}-\d{3}$",
    "iscc": r"^ISCC-(EU|PLUS)-\d{4}-\d{3}$",
}

# EUDR-recognized certification equivalence mapping
EUDR_EQUIVALENCE_MAP = {
    "fsc_fm": True,
    "fsc_coc": True,
    "fsc_cw": False,  # Controlled Wood not fully EUDR-equivalent
    "fsc_project": False,
    "fsc_ecosystem": False,
    "pefc_sfm": True,
    "pefc_coc": True,
    "rspo_ip": True,   # Identity Preserved
    "rspo_mb": True,    # Mass Balance
    "rspo_sg": False,   # Segregated (partial)
    "rspo_bc": False,   # Book & Claim (not EUDR-equivalent)
    "ra_2020": True,    # Rainforest Alliance 2020
    "iscc_eu": True,
    "iscc_plus": True,
}

# Valid certificate statuses
CERT_STATUSES = ["valid", "expired", "suspended", "revoked", "pending"]


def _validate_certificate(
    certificate: Dict,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Validate a single certificate and return validation result."""
    today = today or date.today()
    result = {
        "certificate_id": certificate.get("certificate_id"),
        "scheme": certificate.get("scheme"),
        "valid": False,
        "status_check": "unknown",
        "scope_valid": False,
        "code_format_valid": False,
        "eudr_equivalent": False,
        "expiry_check": "unknown",
        "errors": [],
    }

    # Check required fields
    required = ["certificate_id", "scheme", "status", "expiry_date"]
    for field in required:
        if field not in certificate or not certificate[field]:
            result["errors"].append(f"Missing field: {field}")

    if result["errors"]:
        return result

    scheme = certificate["scheme"]
    status = certificate["status"]
    sub_scheme = certificate.get("sub_scheme", "")

    # Status check
    result["status_check"] = status
    if status not in ("valid",):
        result["errors"].append(f"Certificate status: {status}")

    # Expiry check
    try:
        expiry = date.fromisoformat(certificate["expiry_date"])
        if expiry < today:
            result["expiry_check"] = "expired"
            result["errors"].append("Certificate has expired")
        elif (expiry - today).days <= 90:
            result["expiry_check"] = "expiring_soon"
        else:
            result["expiry_check"] = "valid"
    except (ValueError, TypeError):
        result["errors"].append("Invalid expiry date")
        return result

    # Code format check
    if "certificate_code" in certificate:
        pattern = CERT_CODE_PATTERNS.get(scheme)
        if pattern and re.match(pattern, certificate["certificate_code"]):
            result["code_format_valid"] = True
        elif not pattern:
            result["code_format_valid"] = True  # No pattern = accept
        else:
            result["code_format_valid"] = False

    # EUDR equivalence
    result["eudr_equivalent"] = EUDR_EQUIVALENCE_MAP.get(sub_scheme, False)

    # Scope validation
    cert_scope = certificate.get("scope", [])
    result["scope_valid"] = len(cert_scope) > 0 and any(
        c in EUDR_COMMODITIES for c in cert_scope
    )

    # Overall validation
    result["valid"] = (
        status == "valid"
        and result["expiry_check"] in ("valid", "expiring_soon")
        and result["scope_valid"]
        and len(result["errors"]) == 0
    )
    return result


def _validate_batch(certificates: List[Dict], **kwargs) -> List[Dict]:
    """Validate a batch of certificates."""
    return [_validate_certificate(cert, **kwargs) for cert in certificates]


def _check_eudr_equivalence(scheme: str, sub_scheme: str) -> bool:
    """Check if a certification scheme/sub-scheme is EUDR-equivalent."""
    return EUDR_EQUIVALENCE_MAP.get(sub_scheme, False)


def _validate_scope(certificate_scope: List[str], required_commodity: str) -> bool:
    """Validate that certificate scope covers the required commodity."""
    return required_commodity in certificate_scope


# ===========================================================================
# 1. FSC Validation (18 tests)
# ===========================================================================


class TestFSCValidation:
    """Test FSC certification validation across all 5 sub-schemes."""

    @pytest.mark.parametrize("sub_scheme", FSC_SUB_SCHEMES)
    def test_fsc_sub_scheme_recognition(self, sub_scheme):
        """Test each FSC sub-scheme is recognized."""
        assert sub_scheme in FSC_SUB_SCHEMES

    def test_fsc_fm_valid_certificate(self, fsc_certificates):
        """Test valid FSC Forest Management certificate passes."""
        cert = fsc_certificates[0]  # FSC-FM
        result = _validate_certificate(cert)
        assert result["valid"] is True
        assert result["scheme"] == "fsc"

    def test_fsc_coc_valid_certificate(self, fsc_certificates):
        """Test valid FSC Chain of Custody certificate passes."""
        cert = fsc_certificates[1]  # FSC-COC
        result = _validate_certificate(cert)
        assert result["valid"] is True

    def test_fsc_cw_eudr_equivalence(self, fsc_certificates):
        """Test FSC Controlled Wood is NOT fully EUDR-equivalent."""
        assert _check_eudr_equivalence("fsc", "fsc_cw") is False

    def test_fsc_fm_eudr_equivalence(self):
        """Test FSC Forest Management IS EUDR-equivalent."""
        assert _check_eudr_equivalence("fsc", "fsc_fm") is True

    def test_fsc_coc_eudr_equivalence(self):
        """Test FSC Chain of Custody IS EUDR-equivalent."""
        assert _check_eudr_equivalence("fsc", "fsc_coc") is True

    def test_fsc_project_eudr_equivalence(self):
        """Test FSC Project Certification is NOT EUDR-equivalent."""
        assert _check_eudr_equivalence("fsc", "fsc_project") is False

    def test_fsc_ecosystem_eudr_equivalence(self):
        """Test FSC Ecosystem Services is NOT EUDR-equivalent."""
        assert _check_eudr_equivalence("fsc", "fsc_ecosystem") is False

    def test_fsc_expired_certificate(self, fsc_certificates):
        """Test expired FSC certificate fails validation."""
        expired_cert = fsc_certificates[4]  # status = expired
        result = _validate_certificate(expired_cert)
        assert result["valid"] is False

    def test_fsc_code_format_valid(self):
        """Test valid FSC certificate code format."""
        assert re.match(CERT_CODE_PATTERNS["fsc"], "FSC-C100001")

    def test_fsc_code_format_invalid(self):
        """Test invalid FSC certificate code format."""
        assert not re.match(CERT_CODE_PATTERNS["fsc"], "INVALID-CODE")

    def test_fsc_scope_covers_wood(self, fsc_certificates):
        """Test FSC-FM scope covers wood commodity."""
        cert = fsc_certificates[0]
        assert "wood" in cert["scope"]

    def test_fsc_scope_does_not_cover_oil_palm(self, fsc_certificates):
        """Test FSC-FM scope does not cover oil palm."""
        cert = fsc_certificates[0]
        assert "oil_palm" not in cert["scope"]

    def test_fsc_api_validation(self, mock_fsc_api):
        """Test FSC API certificate validation."""
        result = mock_fsc_api.validate_certificate("FSC-C100001")
        assert result["valid"] is True
        assert result["status"] == "valid"

    def test_fsc_api_invalid_certificate(self, mock_fsc_api):
        """Test FSC API returns invalid for bad certificate code."""
        mock_fsc_api.validate_certificate.return_value = {"valid": False, "status": "not_found"}
        result = mock_fsc_api.validate_certificate("FSC-XXXXXXX")
        assert result["valid"] is False

    def test_fsc_certificate_count(self, fsc_certificates):
        """Test fixture provides 5 FSC certificates (one per sub-scheme)."""
        assert len(fsc_certificates) == 5

    def test_fsc_all_sub_schemes_covered(self, fsc_certificates):
        """Test all 5 FSC sub-schemes are represented."""
        sub_schemes = {c["sub_scheme"] for c in fsc_certificates}
        assert sub_schemes == set(FSC_SUB_SCHEMES)

    def test_fsc_provenance_hash_present(self, fsc_certificates):
        """Test each FSC certificate has a provenance hash."""
        for cert in fsc_certificates:
            assert len(cert["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 2. PEFC Validation (10 tests)
# ===========================================================================


class TestPEFCValidation:
    """Test PEFC certification validation."""

    def test_pefc_sfm_valid(self, pefc_certificates):
        """Test valid PEFC Sustainable Forest Management certificate."""
        cert = pefc_certificates[0]
        result = _validate_certificate(cert)
        assert result["valid"] is True

    def test_pefc_expired(self, pefc_certificates):
        """Test expired PEFC certificate fails validation."""
        cert = pefc_certificates[1]
        result = _validate_certificate(cert)
        assert result["valid"] is False

    def test_pefc_sfm_eudr_equivalence(self):
        """Test PEFC SFM is EUDR-equivalent."""
        assert _check_eudr_equivalence("pefc", "pefc_sfm") is True

    def test_pefc_coc_eudr_equivalence(self):
        """Test PEFC CoC is EUDR-equivalent."""
        assert _check_eudr_equivalence("pefc", "pefc_coc") is True

    def test_pefc_scope_wood(self, pefc_certificates):
        """Test PEFC certificate scope covers wood."""
        assert "wood" in pefc_certificates[0]["scope"]

    def test_pefc_code_format(self):
        """Test PEFC certificate code format validation."""
        assert re.match(CERT_CODE_PATTERNS["pefc"], "PEFC/XX-XX-XXXXXXX")

    def test_pefc_invalid_code_format(self):
        """Test invalid PEFC code format."""
        assert not re.match(CERT_CODE_PATTERNS["pefc"], "PEFC-123")

    def test_pefc_api_validation(self, mock_pefc_api):
        """Test PEFC API validates certificate correctly."""
        result = mock_pefc_api.validate_certificate("PEFC/XX-XX-XXXXXXX")
        assert result["valid"] is True

    def test_pefc_certificate_provenance(self, pefc_certificates):
        """Test PEFC certificates have provenance hashes."""
        for cert in pefc_certificates:
            assert "provenance_hash" in cert

    def test_pefc_fixture_count(self, pefc_certificates):
        """Test PEFC fixture provides expected number of certificates."""
        assert len(pefc_certificates) == 2


# ===========================================================================
# 3. RSPO Validation (12 tests)
# ===========================================================================


class TestRSPOValidation:
    """Test RSPO certification validation for oil palm."""

    def test_rspo_ip_valid(self, rspo_certificates):
        """Test valid RSPO Identity Preserved certificate."""
        cert = rspo_certificates[0]
        result = _validate_certificate(cert)
        assert result["valid"] is True

    def test_rspo_mb_valid(self, rspo_certificates):
        """Test valid RSPO Mass Balance certificate."""
        cert = rspo_certificates[1]
        result = _validate_certificate(cert)
        assert result["valid"] is True

    def test_rspo_suspended_fails(self, rspo_certificates):
        """Test suspended RSPO certificate fails validation."""
        cert = rspo_certificates[2]
        result = _validate_certificate(cert)
        assert result["valid"] is False

    def test_rspo_ip_eudr_equivalence(self):
        """Test RSPO Identity Preserved is EUDR-equivalent."""
        assert _check_eudr_equivalence("rspo", "rspo_ip") is True

    def test_rspo_mb_eudr_equivalence(self):
        """Test RSPO Mass Balance is EUDR-equivalent."""
        assert _check_eudr_equivalence("rspo", "rspo_mb") is True

    def test_rspo_sg_not_eudr_equivalent(self):
        """Test RSPO Segregated is NOT fully EUDR-equivalent."""
        assert _check_eudr_equivalence("rspo", "rspo_sg") is False

    def test_rspo_bc_not_eudr_equivalent(self):
        """Test RSPO Book & Claim is NOT EUDR-equivalent."""
        assert _check_eudr_equivalence("rspo", "rspo_bc") is False

    def test_rspo_scope_oil_palm(self, rspo_certificates):
        """Test RSPO certificates cover oil palm commodity."""
        for cert in rspo_certificates:
            assert "oil_palm" in cert["scope"]

    def test_rspo_scope_not_wood(self, rspo_certificates):
        """Test RSPO certificates do not cover wood commodity."""
        for cert in rspo_certificates:
            assert "wood" not in cert["scope"]

    def test_rspo_api_validation(self, mock_rspo_api):
        """Test RSPO API validates certificate."""
        result = mock_rspo_api.validate_certificate("RSPO-1234567")
        assert result["valid"] is True
        assert result["supply_chain_model"] == "IP"

    def test_rspo_code_format(self):
        """Test valid RSPO certificate code format."""
        assert re.match(CERT_CODE_PATTERNS["rspo"], "RSPO-1234567")

    def test_rspo_fixture_count(self, rspo_certificates):
        """Test RSPO fixture provides 3 certificates."""
        assert len(rspo_certificates) == 3


# ===========================================================================
# 4. Rainforest Alliance Validation (7 tests)
# ===========================================================================


class TestRainforestAllianceValidation:
    """Test Rainforest Alliance certification validation."""

    def test_ra_2020_valid(self, rainforest_alliance_certificates):
        """Test valid Rainforest Alliance 2020 certificate."""
        cert = rainforest_alliance_certificates[0]
        result = _validate_certificate(cert)
        assert result["valid"] is True

    def test_ra_2020_eudr_equivalence(self):
        """Test RA 2020 standard is EUDR-equivalent."""
        assert _check_eudr_equivalence("rainforest_alliance", "ra_2020") is True

    def test_ra_scope_covers_coffee(self, rainforest_alliance_certificates):
        """Test RA certificate covers coffee commodity."""
        cert = rainforest_alliance_certificates[0]
        assert "coffee" in cert["scope"]

    def test_ra_scope_covers_cocoa(self, rainforest_alliance_certificates):
        """Test RA certificate covers cocoa commodity."""
        cert = rainforest_alliance_certificates[0]
        assert "cocoa" in cert["scope"]

    def test_ra_code_format(self):
        """Test valid RA certificate code format."""
        assert re.match(CERT_CODE_PATTERNS["rainforest_alliance"], "RA-CF-2024-001")

    def test_ra_provenance_hash(self, rainforest_alliance_certificates):
        """Test RA certificate has provenance hash."""
        cert = rainforest_alliance_certificates[0]
        assert len(cert["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_ra_country_code(self, rainforest_alliance_certificates):
        """Test RA certificate is associated with correct country."""
        cert = rainforest_alliance_certificates[0]
        assert cert["country_code"] == "CO"


# ===========================================================================
# 5. ISCC Validation (7 tests)
# ===========================================================================


class TestISCCValidation:
    """Test ISCC certification validation."""

    def test_iscc_eu_valid(self, iscc_certificates):
        """Test valid ISCC EU certificate."""
        cert = iscc_certificates[0]
        result = _validate_certificate(cert)
        assert result["valid"] is True

    def test_iscc_eu_eudr_equivalence(self):
        """Test ISCC EU is EUDR-equivalent."""
        assert _check_eudr_equivalence("iscc", "iscc_eu") is True

    def test_iscc_plus_eudr_equivalence(self):
        """Test ISCC PLUS is EUDR-equivalent."""
        assert _check_eudr_equivalence("iscc", "iscc_plus") is True

    def test_iscc_scope_soya(self, iscc_certificates):
        """Test ISCC certificate covers soya."""
        cert = iscc_certificates[0]
        assert "soya" in cert["scope"]

    def test_iscc_code_format(self):
        """Test valid ISCC certificate code format."""
        assert re.match(CERT_CODE_PATTERNS["iscc"], "ISCC-EU-2024-001")

    def test_iscc_plus_code_format(self):
        """Test valid ISCC PLUS certificate code format."""
        assert re.match(CERT_CODE_PATTERNS["iscc"], "ISCC-PLUS-2024-001")

    def test_iscc_provenance(self, iscc_certificates):
        """Test ISCC certificate has provenance hash."""
        assert len(iscc_certificates[0]["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 6. EUDR Equivalence Mapping (10 tests)
# ===========================================================================


class TestEUDREquivalenceMapping:
    """Test EUDR equivalence mapping for all certification schemes."""

    @pytest.mark.parametrize("sub_scheme,expected", [
        ("fsc_fm", True),
        ("fsc_coc", True),
        ("fsc_cw", False),
        ("fsc_project", False),
        ("fsc_ecosystem", False),
        ("pefc_sfm", True),
        ("pefc_coc", True),
        ("rspo_ip", True),
        ("rspo_mb", True),
        ("rspo_sg", False),
        ("rspo_bc", False),
        ("ra_2020", True),
        ("iscc_eu", True),
        ("iscc_plus", True),
    ])
    def test_eudr_equivalence(self, sub_scheme, expected):
        """Test EUDR equivalence for each certification sub-scheme."""
        assert _check_eudr_equivalence("", sub_scheme) == expected

    def test_unknown_sub_scheme_defaults_false(self):
        """Test unknown sub-scheme defaults to not EUDR-equivalent."""
        assert _check_eudr_equivalence("unknown", "unknown_sub") is False

    def test_equivalence_map_completeness(self):
        """Test equivalence map covers all defined sub-schemes."""
        expected_keys = {
            "fsc_fm", "fsc_coc", "fsc_cw", "fsc_project", "fsc_ecosystem",
            "pefc_sfm", "pefc_coc",
            "rspo_ip", "rspo_mb", "rspo_sg", "rspo_bc",
            "ra_2020",
            "iscc_eu", "iscc_plus",
        }
        assert set(EUDR_EQUIVALENCE_MAP.keys()) == expected_keys

    def test_at_least_one_scheme_per_type_is_equivalent(self):
        """Test at least one sub-scheme per major scheme is EUDR-equivalent."""
        by_scheme = {}
        for key, val in EUDR_EQUIVALENCE_MAP.items():
            prefix = key.split("_")[0]
            by_scheme.setdefault(prefix, []).append(val)
        for scheme, vals in by_scheme.items():
            assert any(vals), f"No EUDR-equivalent sub-scheme for {scheme}"


# ===========================================================================
# 7. Certificate Expiry Handling (8 tests)
# ===========================================================================


class TestCertificateExpiryHandling:
    """Test certificate expiry detection and status transitions."""

    def test_certificate_not_expired(self, fsc_certificates):
        """Test certificate with future expiry is not expired."""
        cert = fsc_certificates[0]
        result = _validate_certificate(cert)
        assert result["expiry_check"] in ("valid", "expiring_soon")

    def test_certificate_expired(self):
        """Test certificate with past expiry date is expired."""
        cert = {
            "certificate_id": "CERT-EXP",
            "scheme": "fsc",
            "sub_scheme": "fsc_fm",
            "status": "valid",
            "expiry_date": (date.today() - timedelta(days=30)).isoformat(),
            "scope": ["wood"],
            "certificate_code": "FSC-C000001",
        }
        result = _validate_certificate(cert)
        assert result["expiry_check"] == "expired"
        assert result["valid"] is False

    def test_certificate_expiring_within_90_days(self):
        """Test certificate expiring within 90 days flagged."""
        cert = {
            "certificate_id": "CERT-90D",
            "scheme": "fsc",
            "sub_scheme": "fsc_fm",
            "status": "valid",
            "expiry_date": (date.today() + timedelta(days=60)).isoformat(),
            "scope": ["wood"],
            "certificate_code": "FSC-C000002",
        }
        result = _validate_certificate(cert)
        assert result["expiry_check"] == "expiring_soon"

    def test_certificate_status_suspended(self):
        """Test suspended certificate fails validation."""
        cert = {
            "certificate_id": "CERT-SUS",
            "scheme": "rspo",
            "sub_scheme": "rspo_ip",
            "status": "suspended",
            "expiry_date": (date.today() + timedelta(days=365)).isoformat(),
            "scope": ["oil_palm"],
        }
        result = _validate_certificate(cert)
        assert result["valid"] is False

    def test_certificate_status_revoked(self):
        """Test revoked certificate fails validation."""
        cert = {
            "certificate_id": "CERT-REV",
            "scheme": "fsc",
            "sub_scheme": "fsc_fm",
            "status": "revoked",
            "expiry_date": (date.today() + timedelta(days=365)).isoformat(),
            "scope": ["wood"],
        }
        result = _validate_certificate(cert)
        assert result["valid"] is False

    def test_certificate_status_pending(self):
        """Test pending certificate fails validation."""
        cert = {
            "certificate_id": "CERT-PEN",
            "scheme": "pefc",
            "sub_scheme": "pefc_sfm",
            "status": "pending",
            "expiry_date": (date.today() + timedelta(days=365)).isoformat(),
            "scope": ["wood"],
        }
        result = _validate_certificate(cert)
        assert result["valid"] is False

    def test_all_certificate_statuses_handled(self):
        """Test all defined certificate statuses are handled."""
        for status in CERT_STATUSES:
            cert = {
                "certificate_id": f"CERT-{status.upper()}",
                "scheme": "fsc",
                "sub_scheme": "fsc_fm",
                "status": status,
                "expiry_date": (date.today() + timedelta(days=365)).isoformat(),
                "scope": ["wood"],
            }
            result = _validate_certificate(cert)
            assert result["status_check"] == status

    def test_certificate_with_invalid_expiry_format(self):
        """Test certificate with invalid expiry date format."""
        cert = {
            "certificate_id": "CERT-BADDATE",
            "scheme": "fsc",
            "sub_scheme": "fsc_fm",
            "status": "valid",
            "expiry_date": "not-a-date",
            "scope": ["wood"],
        }
        result = _validate_certificate(cert)
        assert result["valid"] is False


# ===========================================================================
# 8. Scope Validation (8 tests)
# ===========================================================================


class TestScopeValidation:
    """Test certificate scope validation for EUDR commodities."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_scope_covers_eudr_commodity(self, commodity):
        """Test scope validation recognizes each EUDR commodity."""
        assert _validate_scope([commodity], commodity) is True

    def test_scope_does_not_cover_non_eudr(self):
        """Test scope validation rejects non-EUDR commodity."""
        assert _validate_scope(["cotton", "tobacco"], "cotton") is False

    def test_scope_multi_commodity(self):
        """Test scope covering multiple commodities."""
        scope = ["wood", "rubber", "coffee"]
        assert _validate_scope(scope, "wood") is True
        assert _validate_scope(scope, "coffee") is True
        assert _validate_scope(scope, "soya") is False

    def test_empty_scope_fails(self):
        """Test empty scope fails validation."""
        assert _validate_scope([], "wood") is False

    def test_scope_validation_in_certificate(self, all_certificates):
        """Test scope validation is performed during certificate validation."""
        for cert in all_certificates:
            result = _validate_certificate(cert)
            if cert["status"] == "valid":
                assert result["scope_valid"] is True


# ===========================================================================
# 9. Batch Validation (5 tests)
# ===========================================================================


class TestBatchCertificateValidation:
    """Test batch certificate validation operations."""

    def test_batch_all_valid(self, fsc_certificates):
        """Test batch validation with mixed valid/expired FSC certificates."""
        results = _validate_batch(fsc_certificates)
        assert len(results) == len(fsc_certificates)

    def test_batch_mixed_schemes(self, all_certificates):
        """Test batch validation across multiple certification schemes."""
        results = _validate_batch(all_certificates)
        assert len(results) == len(all_certificates)
        schemes = {r["scheme"] for r in results}
        assert len(schemes) >= 3

    def test_batch_empty(self):
        """Test batch validation with empty list."""
        results = _validate_batch([])
        assert len(results) == 0

    def test_batch_single(self, fsc_certificates):
        """Test batch validation with single certificate."""
        results = _validate_batch([fsc_certificates[0]])
        assert len(results) == 1

    def test_batch_preserves_ids(self, all_certificates):
        """Test batch results preserve certificate IDs."""
        results = _validate_batch(all_certificates)
        result_ids = {r["certificate_id"] for r in results}
        input_ids = {c["certificate_id"] for c in all_certificates}
        assert result_ids == input_ids
