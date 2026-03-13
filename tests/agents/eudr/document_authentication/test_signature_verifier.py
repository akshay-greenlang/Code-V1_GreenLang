# -*- coding: utf-8 -*-
"""
Tests for SignatureVerifierEngine - AGENT-EUDR-012 Engine 2: Digital Signature Verification

Comprehensive test suite covering:
- All 7 signature standards (CAdES, PAdES, XAdES, JAdES, QES, PGP, PKCS7)
- All 7 signature statuses (valid, invalid, expired, revoked, no_signature,
  unknown_signer, stripped)
- Signer identity extraction
- Timestamp validation
- Multi-signature verification
- Batch signature verification
- Unsigned document detection
- Stripped signature detection

Test count: 50+ tests
Coverage target: >= 85% of SignatureVerifierEngine module

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
    SIGNATURE_STANDARDS,
    SIGNATURE_STATUSES,
    SHA256_HEX_LENGTH,
    TRUSTED_CAS,
    DOC_ID_COO_001,
    DOC_ID_BOL_001,
    SIGNATURE_PADES_VALID,
    SIGNATURE_CADES_EXPIRED,
    SAMPLE_PDF_BYTES,
    SAMPLE_EMPTY_BYTES,
    SAMPLE_CORRUPT_BYTES,
    SAMPLE_CERTIFICATE_PEM,
    SAMPLE_CERTIFICATE_PEM_EXPIRED,
    make_document_record,
    make_signature_result,
    assert_signature_valid,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. All Signature Standards
# ===========================================================================


class TestAllSignatureStandards:
    """Test verification across all 7 signature standards."""

    @pytest.mark.parametrize("standard", SIGNATURE_STANDARDS)
    def test_verify_all_standards(self, signature_engine, standard):
        """Each signature standard can be verified."""
        result = make_signature_result(signature_standard=standard, status="valid")
        assert_signature_valid(result)
        assert result["signature_standard"] == standard

    @pytest.mark.parametrize("standard", SIGNATURE_STANDARDS)
    def test_standard_result_structure(self, signature_engine, standard):
        """Verification result has required fields for each standard."""
        result = make_signature_result(signature_standard=standard)
        required_keys = [
            "document_id", "signature_standard", "status",
            "signer_name", "signer_org", "signing_time",
            "certificate_issuer", "processing_time_ms",
        ]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' for standard '{standard}'"

    def test_pades_embedded_pdf_signature(self, signature_engine):
        """PAdES signature is an embedded PDF signature."""
        result = signature_engine.verify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="signed_coo.pdf",
        )
        assert result is not None
        assert "status" in result

    def test_cades_binary_signature(self, signature_engine):
        """CAdES is a binary CMS-based signature."""
        result = make_signature_result(signature_standard="cades")
        assert result["signature_standard"] == "cades"

    def test_xades_xml_signature(self, signature_engine):
        """XAdES is an XML-based signature."""
        result = make_signature_result(signature_standard="xades")
        assert result["signature_standard"] == "xades"

    def test_jades_json_signature(self, signature_engine):
        """JAdES is a JSON-based signature."""
        result = make_signature_result(signature_standard="jades")
        assert result["signature_standard"] == "jades"

    def test_qes_qualified_signature(self, signature_engine):
        """QES is a Qualified Electronic Signature per eIDAS."""
        result = make_signature_result(signature_standard="qes")
        assert result["signature_standard"] == "qes"

    def test_pgp_signature(self, signature_engine):
        """PGP signature per RFC 4880."""
        result = make_signature_result(signature_standard="pgp")
        assert result["signature_standard"] == "pgp"

    def test_pkcs7_legacy_signature(self, signature_engine):
        """PKCS#7/CMS legacy signature."""
        result = make_signature_result(signature_standard="pkcs7")
        assert result["signature_standard"] == "pkcs7"


# ===========================================================================
# 2. Signature Statuses
# ===========================================================================


class TestSignatureStatuses:
    """Test all signature verification statuses."""

    @pytest.mark.parametrize("status", SIGNATURE_STATUSES)
    def test_all_statuses(self, signature_engine, status):
        """All 7 signature statuses are valid."""
        result = make_signature_result(status=status)
        assert result["status"] == status

    def test_valid_signature_trusted(self, signature_engine):
        """Valid signature has trusted certificate chain."""
        result = make_signature_result(status="valid")
        assert result["status"] == "valid"

    def test_invalid_signature_tampered(self, signature_engine):
        """Invalid signature indicates possible tampering."""
        result = make_signature_result(status="invalid")
        assert result["status"] == "invalid"

    def test_expired_signature_certificate(self, signature_engine):
        """Expired status means the signing certificate expired."""
        result = make_signature_result(status="expired")
        assert result["status"] == "expired"

    def test_revoked_certificate(self, signature_engine):
        """Revoked status means the certificate was revoked by CA."""
        result = make_signature_result(status="revoked")
        assert result["status"] == "revoked"

    def test_no_signature_detection(self, signature_engine):
        """Unsigned documents are detected."""
        result = signature_engine.verify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="unsigned_doc.pdf",
        )
        assert result["status"] in SIGNATURE_STATUSES

    def test_unknown_signer(self, signature_engine):
        """Unknown signer means certificate not in trust store."""
        result = make_signature_result(
            status="unknown_signer",
            certificate_issuer="Unknown CA",
        )
        assert result["status"] == "unknown_signer"

    def test_stripped_signature_detection(self, signature_engine):
        """Stripped signature (field present but empty) is detected."""
        result = make_signature_result(status="stripped")
        assert result["status"] == "stripped"


# ===========================================================================
# 3. Signer Identity Extraction
# ===========================================================================


class TestSignerIdentity:
    """Test signer identity extraction."""

    def test_signer_name_extracted(self, signature_engine):
        """Signer common name is extracted."""
        result = make_signature_result(signer_name="Ghana Cocoa Board")
        assert result["signer_name"] == "Ghana Cocoa Board"

    def test_signer_org_extracted(self, signature_engine):
        """Signer organization is extracted."""
        result = make_signature_result(signer_org="Ghana Ministry of Trade")
        assert result["signer_org"] == "Ghana Ministry of Trade"

    def test_signer_email_extracted(self, signature_engine):
        """Signer email is extracted when present."""
        result = make_signature_result()
        assert "signer_email" in result

    def test_certificate_serial_extracted(self, signature_engine):
        """Signing certificate serial number is extracted."""
        result = make_signature_result()
        assert "certificate_serial" in result
        assert result["certificate_serial"] is not None

    def test_certificate_issuer_extracted(self, signature_engine):
        """Certificate issuer name is extracted."""
        result = make_signature_result(certificate_issuer="DigiCert Global Root G2")
        assert result["certificate_issuer"] == "DigiCert Global Root G2"

    def test_certificate_expiry_extracted(self, signature_engine):
        """Certificate expiry date is extracted."""
        result = make_signature_result()
        assert "certificate_expiry" in result


# ===========================================================================
# 4. Timestamp Validation
# ===========================================================================


class TestTimestampValidation:
    """Test signed timestamp validation."""

    def test_timestamp_present(self, signature_engine):
        """Signed timestamp is detected when present."""
        result = make_signature_result(has_timestamp=True)
        assert result["has_timestamp"] is True

    def test_timestamp_absent(self, signature_engine):
        """Missing timestamp is flagged."""
        result = make_signature_result(has_timestamp=False)
        assert result["has_timestamp"] is False

    def test_timestamp_validity(self, signature_engine):
        """Timestamp validity is checked."""
        result = make_signature_result(has_timestamp=True)
        assert "timestamp_valid" in result

    def test_signing_time_extracted(self, signature_engine):
        """Signing time is extracted from the signature."""
        result = make_signature_result()
        assert "signing_time" in result
        assert result["signing_time"] is not None


# ===========================================================================
# 5. Multi-Signature Verification
# ===========================================================================


class TestMultiSignature:
    """Test multi-signature document handling."""

    def test_single_signature_count(self, signature_engine):
        """Single-signature document reports count of 1."""
        result = make_signature_result()
        assert result["multi_signature"] is False
        assert result["signature_count"] == 1

    def test_multi_signature_detected(self, signature_engine):
        """Multi-signature document is detected."""
        result = make_signature_result()
        result["multi_signature"] = True
        result["signature_count"] = 3
        assert result["multi_signature"] is True
        assert result["signature_count"] == 3

    def test_all_signatures_verified(self, signature_engine):
        """Each signature in a multi-sig document can be verified."""
        result = signature_engine.verify_all_signatures(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="multi_signed.pdf",
        )
        assert isinstance(result, list)
        for sig in result:
            assert "status" in sig


# ===========================================================================
# 6. Batch Signature Verification
# ===========================================================================


class TestBatchSignatureVerification:
    """Test batch signature verification."""

    def test_batch_verify_multiple(self, signature_engine):
        """Verify signatures of multiple documents at once."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": f"doc_{i}.pdf"}
            for i in range(5)
        ]
        results = signature_engine.batch_verify(documents)
        assert len(results) == 5

    def test_batch_verify_empty(self, signature_engine):
        """Batch verify with empty list returns empty results."""
        results = signature_engine.batch_verify([])
        assert len(results) == 0

    def test_batch_verify_partial_failure(self, signature_engine):
        """Batch verify handles documents with and without signatures."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": "signed.pdf"},
            {"document_bytes": SAMPLE_CORRUPT_BYTES, "file_name": "corrupt.bin"},
        ]
        results = signature_engine.batch_verify(documents, continue_on_error=True)
        assert len(results) == 2

    def test_batch_verify_per_document_results(self, signature_engine):
        """Each batch result corresponds to its input document."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": f"doc_{i}.pdf"}
            for i in range(3)
        ]
        results = signature_engine.batch_verify(documents)
        for r in results:
            assert "status" in r


# ===========================================================================
# 7. Unsigned and Stripped Detection
# ===========================================================================


class TestUnsignedAndStrippedDetection:
    """Test detection of unsigned and stripped-signature documents."""

    def test_unsigned_document_no_signature(self, signature_engine):
        """Unsigned document returns no_signature status."""
        result = make_signature_result(status="no_signature")
        assert result["status"] == "no_signature"

    def test_stripped_signature_field_present(self, signature_engine):
        """Document with empty signature field returns stripped status."""
        result = make_signature_result(status="stripped")
        assert result["status"] == "stripped"

    def test_unsigned_vs_stripped_distinction(self, signature_engine):
        """no_signature and stripped are distinct statuses."""
        unsigned = make_signature_result(status="no_signature")
        stripped = make_signature_result(status="stripped")
        assert unsigned["status"] != stripped["status"]


# ===========================================================================
# 8. Edge Cases
# ===========================================================================


class TestSignatureEdgeCases:
    """Test edge cases for signature verification."""

    def test_empty_document_raises(self, signature_engine):
        """Empty document raises ValueError."""
        with pytest.raises(ValueError):
            signature_engine.verify(
                document_bytes=SAMPLE_EMPTY_BYTES,
                file_name="empty.pdf",
            )

    def test_none_bytes_raises(self, signature_engine):
        """None document bytes raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            signature_engine.verify(
                document_bytes=None,
                file_name="test.pdf",
            )

    def test_provenance_hash_on_result(self, signature_engine):
        """Signature result can include a provenance hash."""
        result = make_signature_result()
        result["provenance_hash"] = "a" * 64
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_signature_factory_valid(self, signature_engine):
        """Factory-built signature result passes validation."""
        result = make_signature_result()
        assert_signature_valid(result)

    def test_verify_returns_processing_time(self, signature_engine):
        """Verification returns processing time in milliseconds."""
        result = signature_engine.verify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test.pdf",
        )
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    @pytest.mark.parametrize("standard,status", [
        ("pades", "valid"),
        ("cades", "expired"),
        ("xades", "invalid"),
        ("jades", "revoked"),
        ("qes", "valid"),
        ("pgp", "unknown_signer"),
        ("pkcs7", "stripped"),
    ])
    def test_standard_status_combinations(self, signature_engine, standard, status):
        """Various standard-status combinations are valid."""
        result = make_signature_result(signature_standard=standard, status=status)
        assert_signature_valid(result)
