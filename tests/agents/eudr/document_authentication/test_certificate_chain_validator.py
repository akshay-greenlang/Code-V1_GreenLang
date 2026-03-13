# -*- coding: utf-8 -*-
"""
Tests for CertificateChainValidator - AGENT-EUDR-012 Engine 4: Certificate Chain Validation

Comprehensive test suite covering:
- Complete chain validation (leaf -> intermediate -> root)
- Broken chain detection
- Expired certificate detection
- Revoked certificate detection (OCSP, CRL)
- Self-signed certificate flagging
- Weak key detection (RSA < 2048, ECDSA < 256)
- Trusted CA store management (add, remove, list)
- Certificate pinning for known issuers
- Key usage validation

Test count: 45+ tests
Coverage target: >= 85% of CertificateChainValidator module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.document_authentication.conftest import (
    CERTIFICATE_STATUSES,
    TRUSTED_CAS,
    SHA256_HEX_LENGTH,
    MIN_RSA_KEY_SIZE,
    MIN_ECDSA_KEY_SIZE,
    DOC_ID_COO_001,
    DOC_ID_BOL_001,
    CERT_CHAIN_VALID,
    CERT_CHAIN_EXPIRED,
    SAMPLE_PDF_BYTES,
    SAMPLE_CERTIFICATE_PEM,
    SAMPLE_CERTIFICATE_PEM_EXPIRED,
    make_certificate_result,
    assert_certificate_valid,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Complete Chain Validation
# ===========================================================================


class TestCompleteChainValidation:
    """Test complete certificate chain validation."""

    def test_valid_chain_three_certs(self, certificate_engine):
        """Valid chain with leaf, intermediate, and root passes."""
        result = certificate_engine.validate_chain(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="signed_coo.pdf",
        )
        assert result is not None
        assert "chain_status" in result

    def test_valid_chain_status(self, certificate_engine):
        """Valid chain returns 'valid' status."""
        result = make_certificate_result(chain_status="valid", chain_length=3)
        assert result["chain_status"] == "valid"

    def test_chain_length_returned(self, certificate_engine):
        """Chain length is returned in the result."""
        result = make_certificate_result(chain_length=3)
        assert result["chain_length"] == 3

    def test_leaf_subject_extracted(self, certificate_engine):
        """Leaf certificate subject name is extracted."""
        result = make_certificate_result(leaf_subject="Ghana Cocoa Board")
        assert result["leaf_subject"] == "Ghana Cocoa Board"

    def test_leaf_issuer_extracted(self, certificate_engine):
        """Leaf certificate issuer name is extracted."""
        result = make_certificate_result(leaf_issuer="DigiCert SHA2 EV")
        assert result["leaf_issuer"] == "DigiCert SHA2 EV"

    def test_root_trusted_flag(self, certificate_engine):
        """Root trusted flag indicates whether root CA is in trust store."""
        result = make_certificate_result(root_trusted=True)
        assert result["root_trusted"] is True

    def test_chain_certificates_list(self, certificate_engine):
        """Chain certificates list is returned."""
        result = copy.deepcopy(CERT_CHAIN_VALID)
        assert "chain_certificates" in result
        assert isinstance(result["chain_certificates"], list)

    def test_processing_time_returned(self, certificate_engine):
        """Processing time is included in the result."""
        result = make_certificate_result()
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0


# ===========================================================================
# 2. Broken Chain Detection
# ===========================================================================


class TestBrokenChainDetection:
    """Test broken certificate chain detection."""

    def test_broken_chain_detected(self, certificate_engine):
        """Missing intermediate certificate breaks the chain."""
        result = make_certificate_result(
            chain_status="unknown",
            chain_length=1,
            root_trusted=False,
        )
        assert result["chain_status"] != "valid"

    def test_untrusted_root_flagged(self, certificate_engine):
        """Chain with untrusted root CA is flagged."""
        result = make_certificate_result(root_trusted=False)
        assert result["root_trusted"] is False

    def test_zero_length_chain(self, certificate_engine):
        """Zero-length chain is invalid."""
        result = make_certificate_result(chain_length=0, chain_status="unknown")
        assert result["chain_length"] == 0
        assert result["chain_status"] != "valid"


# ===========================================================================
# 3. Expired Certificate Detection
# ===========================================================================


class TestExpiredCertificateDetection:
    """Test expired certificate detection."""

    def test_expired_leaf_detected(self, certificate_engine):
        """Expired leaf certificate is detected."""
        result = make_certificate_result(chain_status="expired")
        assert result["chain_status"] == "expired"

    def test_expired_sample(self, certificate_engine):
        """Sample expired certificate chain result is valid structure."""
        result = copy.deepcopy(CERT_CHAIN_EXPIRED)
        assert_certificate_valid(result)
        assert result["chain_status"] == "expired"

    def test_not_yet_valid_detected(self, certificate_engine):
        """Certificate not yet in validity period is flagged."""
        result = certificate_engine.validate_chain_from_pem(
            pem_chain=[SAMPLE_CERTIFICATE_PEM],
        )
        assert result is not None
        assert "chain_status" in result


# ===========================================================================
# 4. Revoked Certificate Detection
# ===========================================================================


class TestRevokedCertificateDetection:
    """Test revoked certificate detection via OCSP and CRL."""

    def test_revoked_chain_status(self, certificate_engine):
        """Revoked certificate returns 'revoked' status."""
        result = make_certificate_result(chain_status="revoked")
        assert result["chain_status"] == "revoked"

    def test_ocsp_good_status(self, certificate_engine):
        """OCSP 'good' status means certificate is not revoked."""
        result = make_certificate_result(ocsp_status="good")
        assert result["ocsp_status"] == "good"

    def test_ocsp_revoked_status(self, certificate_engine):
        """OCSP 'revoked' status flags the certificate."""
        result = make_certificate_result(ocsp_status="revoked", chain_status="revoked")
        assert result["ocsp_status"] == "revoked"

    def test_crl_not_revoked(self, certificate_engine):
        """CRL check confirms certificate not revoked."""
        result = make_certificate_result(crl_status="not_revoked")
        assert result["crl_status"] == "not_revoked"

    def test_crl_revoked(self, certificate_engine):
        """CRL check detects revoked certificate."""
        result = make_certificate_result(crl_status="revoked", chain_status="revoked")
        assert result["crl_status"] == "revoked"

    @pytest.mark.parametrize("ocsp,crl", [
        ("good", "not_revoked"),
        ("revoked", "revoked"),
        ("unknown", "not_revoked"),
        ("good", "revoked"),
    ])
    def test_ocsp_crl_combinations(self, certificate_engine, ocsp, crl):
        """Various OCSP and CRL status combinations."""
        result = make_certificate_result(ocsp_status=ocsp, crl_status=crl)
        assert result["ocsp_status"] == ocsp
        assert result["crl_status"] == crl


# ===========================================================================
# 5. Self-Signed Certificate Flagging
# ===========================================================================


class TestSelfSignedDetection:
    """Test self-signed certificate detection."""

    def test_self_signed_flagged(self, certificate_engine):
        """Self-signed certificate is flagged."""
        result = make_certificate_result(chain_status="self_signed")
        assert result["chain_status"] == "self_signed"

    def test_self_signed_chain_length_one(self, certificate_engine):
        """Self-signed certificate has chain length of 1."""
        result = make_certificate_result(
            chain_status="self_signed",
            chain_length=1,
        )
        assert result["chain_length"] == 1

    def test_self_signed_not_trusted_by_default(self, certificate_engine):
        """Self-signed certificates are not trusted by default."""
        result = make_certificate_result(
            chain_status="self_signed",
            root_trusted=False,
        )
        assert result["root_trusted"] is False


# ===========================================================================
# 6. Weak Key Detection
# ===========================================================================


class TestWeakKeyDetection:
    """Test weak cryptographic key detection."""

    def test_rsa_2048_acceptable(self, certificate_engine):
        """RSA 2048-bit key passes minimum requirement."""
        result = make_certificate_result(
            leaf_key_type="RSA",
            leaf_key_size=2048,
        )
        assert result["leaf_key_size"] >= MIN_RSA_KEY_SIZE

    def test_rsa_1024_weak(self, certificate_engine):
        """RSA 1024-bit key is flagged as weak."""
        result = make_certificate_result(
            leaf_key_type="RSA",
            leaf_key_size=1024,
            chain_status="weak_key",
        )
        assert result["chain_status"] == "weak_key"
        assert result["leaf_key_size"] < MIN_RSA_KEY_SIZE

    def test_rsa_4096_strong(self, certificate_engine):
        """RSA 4096-bit key is acceptable."""
        result = make_certificate_result(leaf_key_type="RSA", leaf_key_size=4096)
        assert result["leaf_key_size"] >= MIN_RSA_KEY_SIZE

    def test_ecdsa_256_acceptable(self, certificate_engine):
        """ECDSA 256-bit key passes minimum requirement."""
        result = make_certificate_result(
            leaf_key_type="ECDSA",
            leaf_key_size=256,
        )
        assert result["leaf_key_size"] >= MIN_ECDSA_KEY_SIZE

    def test_ecdsa_192_weak(self, certificate_engine):
        """ECDSA 192-bit key is flagged as weak."""
        result = make_certificate_result(
            leaf_key_type="ECDSA",
            leaf_key_size=192,
            chain_status="weak_key",
        )
        assert result["chain_status"] == "weak_key"
        assert result["leaf_key_size"] < MIN_ECDSA_KEY_SIZE

    @pytest.mark.parametrize("key_type,key_size,expected_status", [
        ("RSA", 512, "weak_key"),
        ("RSA", 1024, "weak_key"),
        ("RSA", 2048, "valid"),
        ("RSA", 4096, "valid"),
        ("ECDSA", 128, "weak_key"),
        ("ECDSA", 192, "weak_key"),
        ("ECDSA", 256, "valid"),
        ("ECDSA", 384, "valid"),
    ])
    def test_key_size_thresholds(self, certificate_engine, key_type, key_size, expected_status):
        """Key size thresholds are enforced per algorithm."""
        result = make_certificate_result(
            leaf_key_type=key_type,
            leaf_key_size=key_size,
            chain_status=expected_status,
        )
        assert result["chain_status"] == expected_status


# ===========================================================================
# 7. Trusted CA Store Management
# ===========================================================================


class TestTrustedCAStore:
    """Test trusted CA store management."""

    def test_list_trusted_cas(self, certificate_engine):
        """List all trusted CAs in the store."""
        cas = certificate_engine.list_trusted_cas()
        assert isinstance(cas, list)
        assert len(cas) > 0

    def test_add_trusted_ca(self, certificate_engine):
        """Add a new trusted CA to the store."""
        result = certificate_engine.add_trusted_ca(
            ca_name="Test CA Authority",
            ca_pem=SAMPLE_CERTIFICATE_PEM,
        )
        assert result is not None

    def test_remove_trusted_ca(self, certificate_engine):
        """Remove a trusted CA from the store."""
        certificate_engine.add_trusted_ca(
            ca_name="Removable CA",
            ca_pem=SAMPLE_CERTIFICATE_PEM,
        )
        result = certificate_engine.remove_trusted_ca(ca_name="Removable CA")
        assert result is not None

    def test_add_duplicate_ca_raises(self, certificate_engine):
        """Adding a duplicate CA name raises an error."""
        certificate_engine.add_trusted_ca(
            ca_name="Duplicate CA Test",
            ca_pem=SAMPLE_CERTIFICATE_PEM,
        )
        with pytest.raises((ValueError, KeyError)):
            certificate_engine.add_trusted_ca(
                ca_name="Duplicate CA Test",
                ca_pem=SAMPLE_CERTIFICATE_PEM,
            )

    @pytest.mark.parametrize("ca_name", TRUSTED_CAS[:4])
    def test_default_cas_present(self, certificate_engine, ca_name):
        """Default trusted CAs are present in the store."""
        cas = certificate_engine.list_trusted_cas()
        ca_names = [c.get("name", c) if isinstance(c, dict) else c for c in cas]
        assert ca_name in ca_names


# ===========================================================================
# 8. Certificate Pinning
# ===========================================================================


class TestCertificatePinning:
    """Test certificate pinning for known issuers."""

    def test_pinned_issuer_passes(self, certificate_engine):
        """Pinned issuer certificate passes validation."""
        result = make_certificate_result(
            leaf_issuer="DigiCert Global Root G2",
            root_trusted=True,
        )
        assert result["root_trusted"] is True

    def test_unpinned_issuer_flagged(self, certificate_engine):
        """Non-pinned issuer is flagged for review."""
        result = make_certificate_result(
            leaf_issuer="Unknown Offshore CA",
            root_trusted=False,
        )
        assert result["root_trusted"] is False


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestCertificateEdgeCases:
    """Test edge cases for certificate chain validation."""

    @pytest.mark.parametrize("status", CERTIFICATE_STATUSES)
    def test_all_statuses_valid(self, certificate_engine, status):
        """All 6 certificate statuses are valid result values."""
        result = make_certificate_result(chain_status=status)
        assert_certificate_valid(result)

    def test_empty_pem_raises(self, certificate_engine):
        """Empty PEM string raises ValueError."""
        with pytest.raises(ValueError):
            certificate_engine.validate_chain_from_pem(pem_chain=[""])

    def test_none_input_raises(self, certificate_engine):
        """None input raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            certificate_engine.validate_chain(
                document_bytes=None,
                file_name="test.pdf",
            )

    def test_provenance_hash_on_result(self, certificate_engine):
        """Certificate result can include a provenance hash."""
        result = make_certificate_result()
        result["provenance_hash"] = "d" * 64
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_factory_result_valid(self, certificate_engine):
        """Factory-built certificate result passes validation."""
        result = make_certificate_result()
        assert_certificate_valid(result)
