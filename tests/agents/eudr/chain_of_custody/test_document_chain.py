# -*- coding: utf-8 -*-
"""
Tests for DocumentChainVerifier - AGENT-EUDR-009 Engine 6: Document Chain

Comprehensive test suite covering:
- Document linking to events (F6.2)
- Required document validation per event type (F6.4)
- Document completeness scoring (F6.3)
- Gap detection (F6.6)
- Quantity cross-reference (F6.7)
- Document expiry monitoring (F6.8)
- Hash registration for tamper detection (F6.9)
- DDS package assembly (F6.10)

Test count: 50+ tests
Coverage target: >= 85% of DocumentChainVerifier module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    DOCUMENT_TYPES,
    EVENT_TYPES,
    REQUIRED_DOCUMENTS,
    OPTIONAL_DOCUMENTS,
    SHA256_HEX_LENGTH,
    SAMPLE_DOCUMENTS,
    EXPORT_COCOA_GH,
    IMPORT_COCOA_NL,
    TRANSFER_COCOA_GH_NL,
    STORAGE_IN_COCOA_NL,
    INSPECTION_COCOA_NL,
    make_document,
    make_event,
    assert_valid_completeness_score,
    compute_sha256,
)


# ===========================================================================
# 1. Document Linking (F6.2)
# ===========================================================================


class TestDocumentLinking:
    """Test linking documents to custody events."""

    def test_link_document_to_event(self, document_chain_verifier):
        """Link a single document to a single event."""
        doc = make_document(doc_type="bill_of_lading", event_ids=["EVT-001"])
        result = document_chain_verifier.link_document(doc)
        assert result is not None
        assert "EVT-001" in result["event_ids"]

    def test_link_document_to_multiple_events(self, document_chain_verifier):
        """Link one document to multiple events (many-to-many)."""
        doc = make_document(doc_type="bill_of_lading",
                           event_ids=["EVT-001", "EVT-002", "EVT-003"])
        result = document_chain_verifier.link_document(doc)
        assert len(result["event_ids"]) == 3

    def test_link_multiple_documents_to_event(self, document_chain_verifier):
        """Link multiple documents to one event."""
        docs = [
            make_document(doc_type="bill_of_lading", event_ids=["EVT-MULTI"]),
            make_document(doc_type="phytosanitary_cert", event_ids=["EVT-MULTI"]),
            make_document(doc_type="certificate_of_origin", event_ids=["EVT-MULTI"]),
        ]
        for doc in docs:
            document_chain_verifier.link_document(doc)
        event_docs = document_chain_verifier.get_documents_for_event("EVT-MULTI")
        assert len(event_docs) == 3

    @pytest.mark.parametrize("doc_type", DOCUMENT_TYPES)
    def test_all_document_types_linkable(self, document_chain_verifier, doc_type):
        """All 15 document types can be linked to events."""
        doc = make_document(doc_type=doc_type, event_ids=["EVT-TYPE-TEST"])
        result = document_chain_verifier.link_document(doc)
        assert result["document_type"] == doc_type

    def test_invalid_document_type_raises(self, document_chain_verifier):
        """Invalid document type raises ValueError."""
        doc = make_document(doc_type="invalid_doc_type")
        with pytest.raises(ValueError):
            document_chain_verifier.link_document(doc)

    def test_document_assigns_id(self, document_chain_verifier):
        """Document linking assigns a unique document_id."""
        doc = make_document(doc_id=None)
        result = document_chain_verifier.link_document(doc)
        assert result.get("document_id") is not None

    def test_duplicate_document_id_raises(self, document_chain_verifier):
        """Linking two documents with same ID raises an error."""
        doc = make_document(doc_id="DOC-DUP-001")
        document_chain_verifier.link_document(doc)
        with pytest.raises((ValueError, KeyError)):
            document_chain_verifier.link_document(copy.deepcopy(doc))


# ===========================================================================
# 2. Required Document Validation (F6.4)
# ===========================================================================


class TestRequiredDocumentValidation:
    """Test required document validation per event type."""

    @pytest.mark.parametrize("event_type,required_docs", list(REQUIRED_DOCUMENTS.items()))
    def test_required_docs_per_event_type(self, document_chain_verifier,
                                          event_type, required_docs):
        """Each event type has correct required documents defined."""
        requirements = document_chain_verifier.get_required_documents(event_type)
        for doc_type in required_docs:
            assert doc_type in requirements, (
                f"{doc_type} should be required for {event_type}"
            )

    def test_export_requires_four_documents(self, document_chain_verifier):
        """Export event requires BL, phyto, CoO, and customs declaration."""
        requirements = document_chain_verifier.get_required_documents("export")
        assert "bill_of_lading" in requirements
        assert "phytosanitary_cert" in requirements
        assert "certificate_of_origin" in requirements
        assert "customs_declaration" in requirements

    def test_validate_export_all_present(self, document_chain_verifier):
        """Export with all required documents validates successfully."""
        event = copy.deepcopy(EXPORT_COCOA_GH)
        docs = [
            make_document("bill_of_lading", event_ids=[event["event_id"]]),
            make_document("phytosanitary_cert", event_ids=[event["event_id"]]),
            make_document("certificate_of_origin", event_ids=[event["event_id"]]),
            make_document("customs_declaration", event_ids=[event["event_id"]]),
        ]
        for doc in docs:
            document_chain_verifier.link_document(doc)
        result = document_chain_verifier.validate_event_documents(event["event_id"], "export")
        assert result["is_complete"] is True

    def test_validate_export_missing_bl(self, document_chain_verifier):
        """Export missing bill of lading fails validation."""
        event_id = "EVT-EXP-MISSING-BL"
        docs = [
            make_document("phytosanitary_cert", event_ids=[event_id]),
            make_document("certificate_of_origin", event_ids=[event_id]),
            make_document("customs_declaration", event_ids=[event_id]),
        ]
        for doc in docs:
            document_chain_verifier.link_document(doc)
        result = document_chain_verifier.validate_event_documents(event_id, "export")
        assert result["is_complete"] is False
        assert "bill_of_lading" in result["missing_documents"]

    def test_sampling_no_required_docs(self, document_chain_verifier):
        """Sampling event has no required documents."""
        requirements = document_chain_verifier.get_required_documents("sampling")
        assert len(requirements) == 0


# ===========================================================================
# 3. Document Completeness Scoring (F6.3)
# ===========================================================================


class TestDocumentCompletenessScoring:
    """Test document completeness scoring for custody chains."""

    def test_full_completeness_score(self, document_chain_verifier):
        """Chain with all required documents scores 100%."""
        event_id = "EVT-COMP-FULL"
        for doc_type in REQUIRED_DOCUMENTS.get("export", []):
            document_chain_verifier.link_document(
                make_document(doc_type, event_ids=[event_id])
            )
        score = document_chain_verifier.calculate_completeness(
            event_ids=[event_id],
            event_types=["export"],
        )
        assert_valid_completeness_score(score)
        assert score >= 90.0

    def test_empty_chain_zero_score(self, document_chain_verifier):
        """Chain with no documents scores zero."""
        score = document_chain_verifier.calculate_completeness(
            event_ids=["EVT-EMPTY"],
            event_types=["export"],
        )
        assert score == pytest.approx(0.0)

    def test_partial_completeness(self, document_chain_verifier):
        """Chain with some documents scores between 0 and 100."""
        event_id = "EVT-PARTIAL"
        required = REQUIRED_DOCUMENTS.get("export", [])
        if len(required) > 1:
            document_chain_verifier.link_document(
                make_document(required[0], event_ids=[event_id])
            )
        score = document_chain_verifier.calculate_completeness(
            event_ids=[event_id],
            event_types=["export"],
        )
        assert 0.0 < score < 100.0

    def test_completeness_score_deterministic(self, document_chain_verifier):
        """Same inputs produce the same completeness score."""
        event_id = "EVT-DETERM"
        document_chain_verifier.link_document(
            make_document("bill_of_lading", event_ids=[event_id])
        )
        s1 = document_chain_verifier.calculate_completeness([event_id], ["export"])
        s2 = document_chain_verifier.calculate_completeness([event_id], ["export"])
        assert s1 == s2


# ===========================================================================
# 4. Gap Detection (F6.6)
# ===========================================================================


class TestDocumentGapDetection:
    """Test detection of events missing required documents."""

    def test_detect_missing_docs(self, document_chain_verifier):
        """Detect events that are missing required documents."""
        event_id = "EVT-GAP-001"
        # Only link 1 out of 4 required for export
        document_chain_verifier.link_document(
            make_document("bill_of_lading", event_ids=[event_id])
        )
        gaps = document_chain_verifier.detect_gaps(
            event_ids=[event_id], event_types=["export"]
        )
        assert len(gaps) > 0
        assert any(g["event_id"] == event_id for g in gaps)

    def test_no_gaps_when_complete(self, document_chain_verifier):
        """No gaps when all required documents are present."""
        event_id = "EVT-NOGAP"
        for doc_type in REQUIRED_DOCUMENTS.get("transfer", []):
            document_chain_verifier.link_document(
                make_document(doc_type, event_ids=[event_id])
            )
        gaps = document_chain_verifier.detect_gaps(
            event_ids=[event_id], event_types=["transfer"]
        )
        missing_for_event = [g for g in gaps if g["event_id"] == event_id]
        assert len(missing_for_event) == 0

    def test_gap_report_identifies_missing_types(self, document_chain_verifier):
        """Gap report specifies which document types are missing."""
        event_id = "EVT-GAPTYP"
        gaps = document_chain_verifier.detect_gaps(
            event_ids=[event_id], event_types=["export"]
        )
        if gaps:
            assert "missing_types" in gaps[0] or "missing_documents" in gaps[0]


# ===========================================================================
# 5. Quantity Cross-Reference (F6.7)
# ===========================================================================


class TestQuantityCrossReference:
    """Test document quantity vs event quantity cross-reference."""

    def test_matching_quantity(self, document_chain_verifier):
        """Document quantity matching event quantity passes validation."""
        doc = make_document(doc_type="weight_cert",
                           event_ids=["EVT-QTY-MATCH"],
                           quantity_kg=5000.0)
        document_chain_verifier.link_document(doc)
        result = document_chain_verifier.validate_quantities(
            event_id="EVT-QTY-MATCH",
            event_quantity_kg=5000.0,
        )
        assert result["quantity_match"] is True

    def test_mismatched_quantity_flagged(self, document_chain_verifier):
        """Document quantity not matching event quantity is flagged."""
        doc = make_document(doc_type="weight_cert",
                           event_ids=["EVT-QTY-MISMATCH"],
                           quantity_kg=4500.0)
        document_chain_verifier.link_document(doc)
        result = document_chain_verifier.validate_quantities(
            event_id="EVT-QTY-MISMATCH",
            event_quantity_kg=5000.0,
        )
        assert result["quantity_match"] is False
        assert result["variance_kg"] == pytest.approx(500.0)

    def test_null_doc_quantity_skipped(self, document_chain_verifier):
        """Documents without quantity are skipped in cross-reference."""
        doc = make_document(doc_type="certificate_of_origin",
                           event_ids=["EVT-QTY-NULL"],
                           quantity_kg=None)
        document_chain_verifier.link_document(doc)
        result = document_chain_verifier.validate_quantities(
            event_id="EVT-QTY-NULL",
            event_quantity_kg=5000.0,
        )
        assert result.get("skipped", True) or result.get("quantity_match", True)


# ===========================================================================
# 6. Document Expiry Monitoring (F6.8)
# ===========================================================================


class TestDocumentExpiryMonitoring:
    """Test monitoring of document expiry dates."""

    def test_detect_expired_document(self, document_chain_verifier):
        """Detect documents that have already expired."""
        doc = make_document(doc_type="phytosanitary_cert",
                           event_ids=["EVT-EXPIRED"],
                           days_until_expiry=-30)  # Expired 30 days ago
        document_chain_verifier.link_document(doc)
        expired = document_chain_verifier.check_expiry()
        assert any(d["document_id"] == doc["document_id"] for d in expired)

    def test_detect_expiring_soon(self, document_chain_verifier):
        """Detect documents expiring within the alert window."""
        doc = make_document(doc_type="quality_cert",
                           event_ids=["EVT-EXPIRING"],
                           days_until_expiry=7)
        document_chain_verifier.link_document(doc)
        expiring = document_chain_verifier.check_expiry(alert_days=30)
        assert any(d["document_id"] == doc["document_id"] for d in expiring)

    def test_valid_document_not_flagged(self, document_chain_verifier):
        """Document with distant expiry is not flagged."""
        doc = make_document(doc_type="quality_cert",
                           event_ids=["EVT-VALID"],
                           days_until_expiry=365)
        document_chain_verifier.link_document(doc)
        expiring = document_chain_verifier.check_expiry(alert_days=30)
        assert not any(d["document_id"] == doc["document_id"] for d in expiring)

    @pytest.mark.parametrize("days,should_alert", [
        (5, True), (7, True), (14, True), (29, True),
        (31, False), (90, False), (365, False),
    ])
    def test_expiry_thresholds(self, document_chain_verifier, days, should_alert):
        """Expiry alerting at various thresholds."""
        doc = make_document(doc_type="bill_of_lading",
                           event_ids=[f"EVT-THR-{days}"],
                           days_until_expiry=days,
                           doc_id=f"DOC-THR-{days}")
        document_chain_verifier.link_document(doc)
        expiring = document_chain_verifier.check_expiry(alert_days=30)
        found = any(d["document_id"] == f"DOC-THR-{days}" for d in expiring)
        assert found == should_alert


# ===========================================================================
# 7. Hash Registration for Tamper Detection (F6.9)
# ===========================================================================


class TestHashRegistration:
    """Test document hash registration for tamper detection."""

    def test_hash_registered_on_link(self, document_chain_verifier):
        """Document file hash is registered when document is linked."""
        doc = make_document(doc_type="bill_of_lading")
        result = document_chain_verifier.link_document(doc)
        assert result.get("file_hash") is not None
        assert len(result["file_hash"]) == SHA256_HEX_LENGTH

    def test_verify_hash_integrity(self, document_chain_verifier):
        """Verify that a document has not been tampered with."""
        doc = make_document(doc_type="bill_of_lading", doc_id="DOC-HASH-001")
        document_chain_verifier.link_document(doc)
        result = document_chain_verifier.verify_hash(
            "DOC-HASH-001", doc["file_hash"]
        )
        assert result["integrity_valid"] is True

    def test_detect_tampered_hash(self, document_chain_verifier):
        """Detect when a document hash does not match registered hash."""
        doc = make_document(doc_type="bill_of_lading", doc_id="DOC-HASH-002")
        document_chain_verifier.link_document(doc)
        tampered_hash = hashlib.sha256(b"tampered content").hexdigest()
        result = document_chain_verifier.verify_hash("DOC-HASH-002", tampered_hash)
        assert result["integrity_valid"] is False


# ===========================================================================
# 8. DDS Package Assembly (F6.10)
# ===========================================================================


class TestDDSPackageAssembly:
    """Test DDS submission document package assembly."""

    def test_assemble_dds_package(self, document_chain_verifier):
        """Assemble a DDS document package for EU submission."""
        event_ids = ["EVT-DDS-001", "EVT-DDS-002"]
        for eid in event_ids:
            for doc_type in ["bill_of_lading", "phytosanitary_cert",
                             "certificate_of_origin", "customs_declaration"]:
                document_chain_verifier.link_document(
                    make_document(doc_type, event_ids=[eid])
                )
        package = document_chain_verifier.assemble_dds_package(event_ids)
        assert package is not None
        assert "documents" in package
        assert len(package["documents"]) >= 4

    def test_dds_package_includes_metadata(self, document_chain_verifier):
        """DDS package includes document metadata."""
        event_ids = ["EVT-DDS-META"]
        document_chain_verifier.link_document(
            make_document("bill_of_lading", event_ids=event_ids, issuer="Maersk")
        )
        package = document_chain_verifier.assemble_dds_package(event_ids)
        assert any(d.get("issuer") == "Maersk" for d in package["documents"])

    def test_dds_package_completeness(self, document_chain_verifier):
        """DDS package reports its own completeness score."""
        event_ids = ["EVT-DDS-COMP"]
        document_chain_verifier.link_document(
            make_document("bill_of_lading", event_ids=event_ids)
        )
        package = document_chain_verifier.assemble_dds_package(event_ids)
        assert "completeness_score" in package


# ===========================================================================
# 9. Document Revocation (F6.11)
# ===========================================================================


class TestDocumentRevocation:
    """Test document revocation and status management."""

    def test_revoke_document(self, document_chain_verifier):
        """Revoke a previously valid document."""
        doc = make_document(doc_type="quality_cert", doc_id="DOC-REV-001")
        document_chain_verifier.link_document(doc)
        result = document_chain_verifier.revoke_document(
            "DOC-REV-001", reason="Quality findings invalidated"
        )
        assert result["status"] == "revoked"

    def test_revoked_doc_fails_validation(self, document_chain_verifier):
        """Revoked document causes event validation to fail."""
        event_id = "EVT-REV-FAIL"
        doc = make_document(doc_type="quality_cert",
                           event_ids=[event_id],
                           doc_id="DOC-REV-002")
        document_chain_verifier.link_document(doc)
        document_chain_verifier.revoke_document("DOC-REV-002", reason="Fraud detected")
        result = document_chain_verifier.validate_event_documents(event_id, "inspection")
        assert result["is_complete"] is False

    def test_revoke_nonexistent_raises(self, document_chain_verifier):
        """Revoking a non-existent document raises an error."""
        with pytest.raises((ValueError, KeyError)):
            document_chain_verifier.revoke_document(
                "DOC-DOES-NOT-EXIST", reason="N/A"
            )

    def test_revoked_doc_in_expiry_check(self, document_chain_verifier):
        """Revoked documents are excluded from expiry monitoring."""
        doc = make_document(doc_type="phytosanitary_cert",
                           event_ids=["EVT-REV-EXP"],
                           days_until_expiry=5,
                           doc_id="DOC-REV-003")
        document_chain_verifier.link_document(doc)
        document_chain_verifier.revoke_document("DOC-REV-003", reason="Superseded")
        expiring = document_chain_verifier.check_expiry(alert_days=30)
        revoked_in_list = any(d["document_id"] == "DOC-REV-003" for d in expiring)
        assert revoked_in_list is False


# ===========================================================================
# 10. Document Version Control (F6.12)
# ===========================================================================


class TestDocumentVersionControl:
    """Test document versioning and replacement."""

    def test_replace_document(self, document_chain_verifier):
        """Replace a document with a newer version."""
        old_doc = make_document(doc_type="weight_cert",
                               event_ids=["EVT-VER-001"],
                               doc_id="DOC-VER-OLD")
        document_chain_verifier.link_document(old_doc)
        new_doc = make_document(doc_type="weight_cert",
                               event_ids=["EVT-VER-001"],
                               doc_id="DOC-VER-NEW")
        result = document_chain_verifier.replace_document(
            old_doc_id="DOC-VER-OLD",
            new_document=new_doc,
        )
        assert result["document_id"] == "DOC-VER-NEW"

    def test_replaced_doc_marked_superseded(self, document_chain_verifier):
        """Old document is marked as superseded after replacement."""
        old_doc = make_document(doc_type="quality_cert",
                               event_ids=["EVT-VER-002"],
                               doc_id="DOC-VER-OLD-2")
        document_chain_verifier.link_document(old_doc)
        new_doc = make_document(doc_type="quality_cert",
                               event_ids=["EVT-VER-002"],
                               doc_id="DOC-VER-NEW-2")
        document_chain_verifier.replace_document("DOC-VER-OLD-2", new_doc)
        old_status = document_chain_verifier.get_document_status("DOC-VER-OLD-2")
        assert old_status == "superseded"

    def test_event_uses_latest_version(self, document_chain_verifier):
        """Event document lookup returns the latest version."""
        event_id = "EVT-VER-003"
        old_doc = make_document(doc_type="bill_of_lading",
                               event_ids=[event_id],
                               doc_id="DOC-VER-V1")
        document_chain_verifier.link_document(old_doc)
        new_doc = make_document(doc_type="bill_of_lading",
                               event_ids=[event_id],
                               doc_id="DOC-VER-V2")
        document_chain_verifier.replace_document("DOC-VER-V1", new_doc)
        event_docs = document_chain_verifier.get_documents_for_event(event_id)
        active_ids = [d["document_id"] for d in event_docs if d.get("status") != "superseded"]
        assert "DOC-VER-V2" in active_ids
