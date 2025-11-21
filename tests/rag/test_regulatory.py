# -*- coding: utf-8 -*-
"""
Regulatory compliance tests for GreenLang RAG system.

Tests all governance features for audit-ready operation:
1. DocumentVersionManager - version tracking and historical compliance
2. RAGGovernance - CSRB approval workflow and allowlist enforcement
3. RAGCitation - complete provenance with 14 required fields
4. IngestionManifest - audit trail completeness

References:
- greenlang/intelligence/rag/version_manager.py
- greenlang/intelligence/rag/governance.py
- greenlang/intelligence/rag/models.py (RAGCitation, IngestionManifest)
"""

import os
from greenlang.determinism import DeterministicClock
# Fix OpenMP conflict between PyTorch and FAISS on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pytest
import tempfile
import json
from pathlib import Path
from datetime import date, datetime
from typing import List

from greenlang.intelligence.rag.version_manager import (
    DocumentVersionManager,
    VersionConflict,
)
from greenlang.intelligence.rag.governance import (
    RAGGovernance,
    ApprovalRequest,
)
from greenlang.intelligence.rag.models import (
    DocMeta,
    Chunk,
    RAGCitation,
    IngestionManifest,
)
from greenlang.intelligence.rag.config import RAGConfig
from greenlang.intelligence.rag.hashing import sha256_str


class TestDocumentVersionManager:
    """
    Test DocumentVersionManager for version tracking and historical compliance.

    Requirements:
    - Register document versions with validation
    - Retrieve versions by date for historical compliance
    - Detect version conflicts (same version, different checksums)
    - Track errata and deprecation
    """

    @pytest.fixture
    def version_manager(self):
        """Create a fresh DocumentVersionManager for each test."""
        return DocumentVersionManager()

    @pytest.fixture
    def ghg_protocol_v1_00(self):
        """GHG Protocol Corporate Standard v1.00 (2004)."""
        return DocMeta(
            doc_id="ghg_protocol_v1.00",
            title="GHG Protocol Corporate Accounting and Reporting Standard",
            collection="ghg_protocol_corp",
            source_uri="https://ghgprotocol.org/corporate-standard",
            publisher="WRI/WBCSD",
            publication_date=date(2004, 9, 1),
            version="1.00",
            content_hash="a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4",
            doc_hash="b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2",
        )

    @pytest.fixture
    def ghg_protocol_v1_05(self):
        """GHG Protocol Corporate Standard v1.05 (2015 revision)."""
        return DocMeta(
            doc_id="ghg_protocol_v1.05",
            title="GHG Protocol Corporate Accounting and Reporting Standard",
            collection="ghg_protocol_corp",
            source_uri="https://ghgprotocol.org/corporate-standard",
            publisher="WRI/WBCSD",
            publication_date=date(2015, 3, 24),
            version="1.05",
            content_hash="c4d1e8f2a5b9c6d3e0f7a4b1c8d5e2f9a6b3c0d7e4f1a8b5c2d9e6f3a0b7c4d1",
            doc_hash="d5e2f9a6b3c0d7e4f1a8b5c2d9e6f3a0b7c4d1e8f5a2b9c6d3e0f7a4b1c8d5e2",
        )

    def test_register_version_success(self, version_manager, ghg_protocol_v1_00):
        """Test successful version registration."""
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")

        # Verify version is registered
        versions = version_manager.list_versions("ghg_protocol_corp")
        assert len(versions) == 1
        assert versions[0].version == "1.00"
        assert versions[0].content_hash == ghg_protocol_v1_00.content_hash

    def test_register_multiple_versions(self, version_manager, ghg_protocol_v1_00, ghg_protocol_v1_05):
        """Test registering multiple versions of the same standard."""
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")
        version_manager.register_version(ghg_protocol_v1_05, standard_id="ghg_protocol_corp")

        # Verify both versions are registered
        versions = version_manager.list_versions("ghg_protocol_corp")
        assert len(versions) == 2
        # Should be sorted by publication_date (earliest first)
        assert versions[0].version == "1.00"
        assert versions[1].version == "1.05"

    def test_version_conflict_detection(self, version_manager, ghg_protocol_v1_00):
        """Test detection of version conflicts (same version, different checksums)."""
        # Register original version
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")

        # Attempt to register same version with different checksum
        tampered_doc = DocMeta(
            doc_id="ghg_protocol_v1.00_tampered",
            title="GHG Protocol Corporate Standard",
            collection="ghg_protocol_corp",
            source_uri="https://ghgprotocol.org/corporate-standard",
            publisher="WRI/WBCSD",
            publication_date=date(2004, 9, 1),
            version="1.00",  # Same version
            content_hash="DIFFERENT_HASH_INDICATING_TAMPERING",
            doc_hash="different_doc_hash",
        )

        # Should raise VersionConflict
        with pytest.raises(VersionConflict, match="Version conflict"):
            version_manager.register_version(tampered_doc, standard_id="ghg_protocol_corp")

    def test_retrieve_by_date_historical_compliance(self, version_manager, ghg_protocol_v1_00, ghg_protocol_v1_05):
        """Test date-based version retrieval for historical compliance."""
        # Register both versions
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")
        version_manager.register_version(ghg_protocol_v1_05, standard_id="ghg_protocol_corp")

        # Test 1: Retrieve version for 2010 report (should use v1.00)
        doc_2010 = version_manager.retrieve_by_date("ghg_protocol_corp", date(2010, 1, 1))
        assert doc_2010 is not None
        assert doc_2010.version == "1.00", "2010 report should use v1.00 (v1.05 not yet published)"

        # Test 2: Retrieve version for 2020 report (should use v1.05)
        doc_2020 = version_manager.retrieve_by_date("ghg_protocol_corp", date(2020, 1, 1))
        assert doc_2020 is not None
        assert doc_2020.version == "1.05", "2020 report should use v1.05 (latest version)"

        # Test 3: Retrieve version before any publication (should return None)
        doc_2000 = version_manager.retrieve_by_date("ghg_protocol_corp", date(2000, 1, 1))
        assert doc_2000 is None, "No version available before 2004"

    def test_retrieve_by_version(self, version_manager, ghg_protocol_v1_00, ghg_protocol_v1_05):
        """Test retrieval of specific version by version string."""
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")
        version_manager.register_version(ghg_protocol_v1_05, standard_id="ghg_protocol_corp")

        # Retrieve specific versions
        v1_00 = version_manager.retrieve_by_version("ghg_protocol_corp", "1.00")
        assert v1_00 is not None
        assert v1_00.version == "1.00"
        assert v1_00.publication_date == date(2004, 9, 1)

        v1_05 = version_manager.retrieve_by_version("ghg_protocol_corp", "1.05")
        assert v1_05 is not None
        assert v1_05.version == "1.05"
        assert v1_05.publication_date == date(2015, 3, 24)

        # Non-existent version
        v2_00 = version_manager.retrieve_by_version("ghg_protocol_corp", "2.00")
        assert v2_00 is None

    def test_check_conflicts(self, version_manager, ghg_protocol_v1_00, ghg_protocol_v1_05):
        """Test conflict detection for ambiguous queries."""
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")
        version_manager.register_version(ghg_protocol_v1_05, standard_id="ghg_protocol_corp")

        # Query that matches multiple versions
        matches = version_manager.check_conflicts("GHG Protocol")
        assert len(matches) == 2, "Query should match both versions"
        assert matches[0].version == "1.00"
        assert matches[1].version == "1.05"

    def test_apply_errata(self, version_manager, ghg_protocol_v1_05):
        """Test errata tracking for document corrections."""
        version_manager.register_version(ghg_protocol_v1_05, standard_id="ghg_protocol_corp")

        # Apply errata
        version_manager.apply_errata(
            doc_id="ghg_protocol_v1.05",
            errata_date=date(2016, 6, 1),
            description="Corrected emission factor for natural gas in Table 7.3",
            sections_affected=["Chapter 7 > Table 7.3"],
        )

        # Retrieve errata
        errata_list = version_manager.get_errata("ghg_protocol_v1.05")
        assert len(errata_list) == 1
        assert errata_list[0]['description'] == "Corrected emission factor for natural gas in Table 7.3"
        assert errata_list[0]['errata_date'] == date(2016, 6, 1)
        assert "Chapter 7 > Table 7.3" in errata_list[0]['sections_affected']

    def test_deprecation_tracking(self, version_manager, ghg_protocol_v1_00):
        """Test deprecation marking and checking."""
        version_manager.register_version(ghg_protocol_v1_00, standard_id="ghg_protocol_corp")

        # Mark as deprecated
        version_manager.mark_deprecated(
            standard_id="ghg_protocol_corp",
            deprecation_date=date(2015, 3, 24),
            replacement="ghg_protocol_corp_v1.05",
            reason="Superseded by revised edition with updated emission factors",
        )

        # Check deprecation status
        is_dep, info = version_manager.is_deprecated("ghg_protocol_corp", date(2020, 1, 1))
        assert is_dep is True
        assert info['replacement'] == "ghg_protocol_corp_v1.05"
        assert info['reason'] == "Superseded by revised edition with updated emission factors"

        # Check before deprecation date
        is_dep_2010, _ = version_manager.is_deprecated("ghg_protocol_corp", date(2010, 1, 1))
        assert is_dep_2010 is False

    def test_missing_required_fields(self, version_manager):
        """Test validation of required fields."""
        # Missing version
        doc_no_version = DocMeta(
            doc_id="test_doc",
            title="Test Document",
            collection="test_collection",
            version=None,  # Missing
            publication_date=date(2020, 1, 1),
            content_hash="hash123",
            doc_hash="hash456",
        )

        with pytest.raises(ValueError, match="missing version field"):
            version_manager.register_version(doc_no_version, standard_id="test_standard")

        # Missing publication_date
        doc_no_date = DocMeta(
            doc_id="test_doc",
            title="Test Document",
            collection="test_collection",
            version="1.0",
            publication_date=None,  # Missing
            content_hash="hash123",
            doc_hash="hash456",
        )

        with pytest.raises(ValueError, match="missing publication_date field"):
            version_manager.register_version(doc_no_date, standard_id="test_standard")


class TestRAGGovernance:
    """
    Test RAGGovernance for CSRB approval workflow and allowlist enforcement.

    Requirements:
    - Submit documents for CSRB approval
    - 2/3 majority vote requirement
    - CSRB member validation
    - Approval status tracking
    - Audit trail persistence
    """

    @pytest.fixture
    def config(self):
        """Create RAG configuration with allowlist."""
        return RAGConfig(
            allowlist=["ghg_protocol_corp", "ipcc_ar6_wg3"],
            verify_checksums=True,
        )

    @pytest.fixture
    def governance(self, config, tmp_path):
        """Create RAGGovernance instance with temporary audit directory."""
        return RAGGovernance(config, audit_dir=tmp_path / "audit")

    @pytest.fixture
    def test_document(self, tmp_path):
        """Create a test document file and metadata."""
        doc_path = tmp_path / "test_standard.pdf"
        doc_path.write_text("Test document content for regulatory compliance testing.")

        # Compute hash
        from greenlang.intelligence.rag.hashing import file_hash
        content_hash = file_hash(str(doc_path))

        doc_meta = DocMeta(
            doc_id="test_standard_v1",
            title="Test Climate Standard",
            collection="test_standard",
            source_uri=str(doc_path),
            publisher="Test Standards Body",
            publication_date=date(2023, 1, 1),
            version="1.0",
            content_hash=content_hash,
            doc_hash="test_doc_hash_123",
        )

        return doc_path, doc_meta

    def test_submit_for_approval_success(self, governance, test_document):
        """Test successful submission for CSRB approval."""
        doc_path, doc_meta = test_document
        approvers = ["climate_scientist_1", "climate_scientist_2", "audit_lead"]

        success = governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
            requested_by="data_engineer",
            verify_checksum=True,
        )

        assert success is True

        # Verify approval request created
        request = governance.get_approval_request("test_standard")
        assert request is not None
        assert request.status == "pending"
        assert request.requested_by == "data_engineer"
        assert len(request.approvers_required) == 3
        assert request.votes_approve == 0
        assert request.votes_reject == 0

    def test_vote_approval_2_3_majority(self, governance, test_document):
        """Test 2/3 majority vote requirement."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        # Submit for approval
        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
            requested_by="requester",
        )

        # Vote 1: Approve
        success = governance.vote_approval("test_standard", "scientist_1", approve=True)
        assert success is True
        request = governance.get_approval_request("test_standard")
        assert request.status == "pending"  # 1/3 votes, not enough

        # Vote 2: Approve (2/3 majority reached)
        success = governance.vote_approval("test_standard", "scientist_2", approve=True)
        assert success is True
        request = governance.get_approval_request("test_standard")
        assert request.status == "approved"  # 2/3 majority reached
        assert "test_standard" in governance.approved_collections
        assert "test_standard" in governance.allowlist

    def test_vote_rejection(self, governance, test_document):
        """Test rejection when majority cannot be reached."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )

        # Vote 1: Reject
        governance.vote_approval("test_standard", "scientist_1", approve=False)
        request = governance.get_approval_request("test_standard")
        assert request.status == "pending"

        # Vote 2: Reject (cannot reach 2/3 approval)
        governance.vote_approval("test_standard", "scientist_2", approve=False)
        request = governance.get_approval_request("test_standard")
        assert request.status == "rejected"

    def test_csrb_member_validation(self, governance, test_document):
        """Test that only authorized CSRB members can vote."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )

        # Attempt to vote with unauthorized approver
        success = governance.vote_approval(
            "test_standard",
            "unauthorized_user",
            approve=True,
        )
        assert success is False

    def test_duplicate_vote_prevention(self, governance, test_document):
        """Test that approvers cannot vote twice."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )

        # First vote
        success1 = governance.vote_approval("test_standard", "scientist_1", approve=True)
        assert success1 is True

        # Attempt second vote
        success2 = governance.vote_approval("test_standard", "scientist_1", approve=True)
        assert success2 is False

    def test_approval_status_tracking(self, governance, test_document):
        """Test complete approval workflow status tracking."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        # Submit
        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )
        assert governance.is_approved("test_standard") is False

        # Vote to approve
        governance.vote_approval("test_standard", "scientist_1", approve=True, comment="Looks good")
        governance.vote_approval("test_standard", "scientist_2", approve=True, comment="Approved")

        # Check approved
        assert governance.is_approved("test_standard") is True

    def test_audit_trail_persistence(self, governance, test_document, tmp_path):
        """Test audit trail is persisted to disk."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )

        governance.vote_approval("test_standard", "scientist_1", approve=True, comment="Test comment")

        # Verify audit file exists
        audit_file = tmp_path / "audit" / "approval_requests.json"
        assert audit_file.exists()

        # Verify audit content
        with open(audit_file, 'r') as f:
            audit_data = json.load(f)

        assert "test_standard" in audit_data
        assert audit_data["test_standard"]["votes_approve"] == 1
        assert len(audit_data["test_standard"]["comments"]) == 1
        assert audit_data["test_standard"]["comments"][0]["comment"] == "Test comment"

    def test_allowlist_enforcement(self, governance):
        """Test runtime allowlist enforcement."""
        # Allowed collection
        assert governance.check_allowlist("ghg_protocol_corp") is True

        # Not allowed collection
        assert governance.check_allowlist("unknown_collection") is False

        # Enforce with exception
        with pytest.raises(ValueError, match="not in allowlist"):
            governance.enforce_allowlist(["ghg_protocol_corp", "unknown_collection"])

    def test_get_audit_trail(self, governance, test_document):
        """Test retrieving full audit trail for a collection."""
        doc_path, doc_meta = test_document
        approvers = ["scientist_1", "scientist_2", "scientist_3"]

        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )

        governance.vote_approval("test_standard", "scientist_1", approve=True)

        # Get audit trail (1/3 votes, still pending)
        audit_trail = governance.get_audit_trail("test_standard")
        assert audit_trail is not None
        assert audit_trail["status"] == "pending"
        assert audit_trail["votes_approve"] == 1
        assert "scientist_1" in audit_trail["approvers_voted"]

    def test_list_pending_approvals(self, governance, test_document):
        """Test listing all pending approval requests."""
        doc_path, doc_meta = test_document

        # Submit document
        governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=["scientist_1", "scientist_2", "scientist_3"],
        )

        # List pending
        pending = governance.list_pending_approvals()
        assert len(pending) == 1
        assert pending[0].metadata.collection == "test_standard"


class TestRAGCitation:
    """
    Test RAGCitation format and compliance with 14 required fields.

    Required Fields:
    1. doc_title
    2. publisher
    3. version
    4. publication_date
    5. section_path
    6. section_hash
    7. page_number
    8. paragraph
    9. uri
    10. uri_fragment
    11. checksum
    12. formatted (citation string)
    13. relevance_score
    14. (implicitly: all fields must be present)
    """

    @pytest.fixture
    def sample_doc_meta(self):
        """Sample document metadata for citation testing."""
        return DocMeta(
            doc_id="ghg_protocol_v1.05",
            title="GHG Protocol Corporate Accounting and Reporting Standard",
            collection="ghg_protocol_corp",
            source_uri="https://ghgprotocol.org/sites/default/files/standards/ghg-protocol-revised.pdf",
            publisher="WRI/WBCSD",
            publication_date=date(2015, 3, 24),
            version="1.05",
            content_hash="a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4",
            doc_hash="b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2",
        )

    @pytest.fixture
    def sample_chunk(self):
        """Sample chunk for citation testing."""
        return Chunk(
            chunk_id="test_chunk_123",
            doc_id="ghg_protocol_v1.05",
            section_path="Chapter 7 > Section 7.3 > 7.3.1 Emission Factors",
            section_hash="d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8",
            page_start=45,
            page_end=46,
            paragraph=2,
            start_char=12450,
            end_char=13100,
            text="For stationary combustion sources, the emission factor...",
            token_count=128,
        )

    def test_citation_all_14_fields_present(self, sample_doc_meta, sample_chunk):
        """Test that RAGCitation includes all 14 required fields."""
        citation = RAGCitation.from_chunk(
            chunk=sample_chunk,
            doc_meta=sample_doc_meta,
            relevance_score=0.87,
        )

        # Verify all 14 fields are present and non-None
        assert citation.doc_title is not None  # 1
        assert citation.publisher is not None  # 2
        assert citation.version is not None  # 3
        assert citation.publication_date is not None  # 4
        assert citation.section_path is not None  # 5
        assert citation.section_hash is not None  # 6
        assert citation.page_number is not None  # 7
        assert citation.paragraph is not None  # 8
        assert citation.uri is not None  # 9
        assert citation.uri_fragment is not None  # 10
        assert citation.checksum is not None  # 11
        assert citation.formatted is not None  # 12
        assert citation.relevance_score is not None  # 13

        # Verify field values
        assert citation.doc_title == "GHG Protocol Corporate Accounting and Reporting Standard"
        assert citation.publisher == "WRI/WBCSD"
        assert citation.version == "1.05"
        assert citation.publication_date == date(2015, 3, 24)
        assert citation.section_path == "Chapter 7 > Section 7.3 > 7.3.1 Emission Factors"
        assert citation.page_number == 45
        assert citation.paragraph == 2
        assert citation.checksum == "a3f5b2c8"  # First 8 chars of content_hash

    def test_citation_checksum_validation(self, sample_doc_meta, sample_chunk):
        """Test checksum is properly included and formatted."""
        citation = RAGCitation.from_chunk(
            chunk=sample_chunk,
            doc_meta=sample_doc_meta,
            relevance_score=0.87,
        )

        # Checksum should be first 8 chars of content_hash
        assert len(citation.checksum) == 8
        assert citation.checksum == sample_doc_meta.content_hash[:8]
        assert citation.checksum in citation.formatted

    def test_citation_formatted_string(self, sample_doc_meta, sample_chunk):
        """Test formatted citation string includes all required components."""
        citation = RAGCitation.from_chunk(
            chunk=sample_chunk,
            doc_meta=sample_doc_meta,
            relevance_score=0.87,
        )

        formatted = citation.formatted

        # Verify all components are in formatted string
        assert "GHG Protocol Corporate Accounting and Reporting Standard" in formatted
        assert "v1.05" in formatted
        assert "WRI/WBCSD" in formatted
        assert "2015-03-24" in formatted
        assert "Chapter 7 > Section 7.3 > 7.3.1 Emission Factors" in formatted
        assert "para 2" in formatted
        assert "p.45" in formatted
        assert "https://ghgprotocol.org" in formatted
        assert "SHA256:a3f5b2c8" in formatted

    def test_citation_provenance_tracking(self, sample_doc_meta, sample_chunk):
        """Test citation provides complete provenance chain."""
        citation = RAGCitation.from_chunk(
            chunk=sample_chunk,
            doc_meta=sample_doc_meta,
            relevance_score=0.92,
        )

        # Provenance chain: Document → Section → Fragment
        assert citation.doc_title is not None
        assert citation.section_path is not None
        assert citation.uri_fragment is not None

        # URI fragment should be constructed from section_path
        assert citation.uri_fragment.startswith("#")
        assert "Chapter_7" in citation.uri_fragment or "7.3.1" in citation.uri_fragment.replace("_", "")

    def test_citation_relevance_score(self, sample_doc_meta, sample_chunk):
        """Test relevance score is included and validated."""
        citation = RAGCitation.from_chunk(
            chunk=sample_chunk,
            doc_meta=sample_doc_meta,
            relevance_score=0.95,
        )

        assert citation.relevance_score == 0.95
        assert 0.0 <= citation.relevance_score <= 1.0

    def test_citation_without_optional_fields(self):
        """Test citation generation when some optional fields are missing."""
        # Document without publisher/date
        doc_meta = DocMeta(
            doc_id="test_doc",
            title="Test Document",
            collection="test_collection",
            publisher=None,
            publication_date=None,
            version=None,
            source_uri=None,
            content_hash="abcd1234" + "0" * 56,
            doc_hash="efgh5678" + "0" * 56,
        )

        chunk = Chunk(
            chunk_id="chunk_1",
            doc_id="test_doc",
            section_path="Introduction",
            section_hash="hash123" + "0" * 57,
            page_start=None,
            paragraph=None,
            start_char=0,
            end_char=100,
            text="Test text",
            token_count=10,
        )

        citation = RAGCitation.from_chunk(
            chunk=chunk,
            doc_meta=doc_meta,
            relevance_score=0.5,
        )

        # Should still create citation with available fields
        assert citation.doc_title == "Test Document"
        assert citation.checksum == "abcd1234"
        assert citation.formatted is not None


class TestIngestionManifest:
    """
    Test IngestionManifest for audit trail completeness.

    Requirements:
    - Complete metadata for all ingested documents
    - Timestamp consistency
    - Pipeline version tracking
    - Deterministic behavior
    """

    @pytest.fixture
    def sample_doc_meta(self):
        """Sample document metadata."""
        return DocMeta(
            doc_id="test_doc_001",
            title="Test Climate Standard",
            collection="test_collection",
            source_uri="file:///path/to/doc.pdf",
            publisher="Test Publisher",
            publication_date=date(2023, 1, 1),
            version="1.0",
            content_hash="hash123" + "0" * 57,
            doc_hash="hash456" + "0" * 57,
            ingested_by="test_user",
            pipeline_version="0.5.0",
            total_chunks=50,
        )

    def test_manifest_creation(self, sample_doc_meta):
        """Test IngestionManifest creation with complete metadata."""
        manifest = IngestionManifest(
            collection_id="test_collection",
            documents=[sample_doc_meta],
            ingestion_duration_seconds=127.5,
            pipeline_config={
                "chunk_size": 512,
                "chunk_overlap": 64,
                "chunking_strategy": "token_aware",
            },
            transformations=[
                {"type": "table_extraction", "tool": "Camelot", "version": "0.10.1"},
                {"type": "section_path_extraction", "tool": "ClimateDocParser"},
            ],
            approved_by=["climate_scientist_1", "audit_lead"],
            approval_date=datetime(2023, 1, 2, 10, 0, 0),
            vector_store_type="FAISS",
            vector_store_config={"index_type": "IndexFlatL2", "dimension": 384},
            total_embeddings=50,
        )

        # Verify all fields
        assert manifest.collection_id == "test_collection"
        assert len(manifest.documents) == 1
        assert manifest.ingestion_duration_seconds == 127.5
        assert manifest.pipeline_config["chunk_size"] == 512
        assert len(manifest.transformations) == 2
        assert "climate_scientist_1" in manifest.approved_by
        assert manifest.vector_store_type == "FAISS"
        assert manifest.total_embeddings == 50

    def test_manifest_metadata_completeness(self, sample_doc_meta):
        """Test manifest includes complete metadata for audit."""
        manifest = IngestionManifest(
            collection_id="test_collection",
            documents=[sample_doc_meta],
            pipeline_config={"chunk_size": 512},
            vector_store_type="FAISS",
            vector_store_config={},
            total_embeddings=50,
        )

        # Verify document metadata
        doc = manifest.documents[0]
        assert doc.doc_id is not None
        assert doc.title is not None
        assert doc.content_hash is not None
        assert doc.doc_hash is not None
        assert doc.ingested_by is not None
        assert doc.pipeline_version is not None

    def test_manifest_timestamp_consistency(self):
        """Test timestamp fields are consistent and properly set."""
        start_time = DeterministicClock.utcnow()

        manifest = IngestionManifest(
            collection_id="test_collection",
            documents=[],
            pipeline_config={},
            vector_store_type="FAISS",
            vector_store_config={},
            total_embeddings=0,
        )

        end_time = DeterministicClock.utcnow()

        # Ingestion timestamp should be within test execution time
        assert start_time <= manifest.ingestion_timestamp <= end_time

    def test_manifest_pipeline_version_tracking(self, sample_doc_meta):
        """Test pipeline configuration and version tracking."""
        manifest = IngestionManifest(
            collection_id="test_collection",
            documents=[sample_doc_meta],
            pipeline_config={
                "chunk_size": 1024,
                "chunk_overlap": 128,
                "chunking_strategy": "semantic",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
            },
            vector_store_type="ChromaDB",
            vector_store_config={"persist_directory": "/data/chroma"},
            total_embeddings=100,
        )

        # Verify pipeline config
        assert manifest.pipeline_config["chunk_size"] == 1024
        assert manifest.pipeline_config["chunking_strategy"] == "semantic"
        assert manifest.pipeline_config["embedding_model"] == "all-MiniLM-L6-v2"

        # Verify document pipeline version
        assert sample_doc_meta.pipeline_version == "0.5.0"

    def test_manifest_deterministic_behavior(self, sample_doc_meta):
        """Test manifest generation is deterministic."""
        # Create two identical manifests
        manifest1 = IngestionManifest(
            collection_id="test_collection",
            documents=[sample_doc_meta],
            pipeline_config={"chunk_size": 512},
            vector_store_type="FAISS",
            vector_store_config={},
            total_embeddings=50,
        )

        manifest2 = IngestionManifest(
            collection_id="test_collection",
            documents=[sample_doc_meta],
            pipeline_config={"chunk_size": 512},
            vector_store_type="FAISS",
            vector_store_config={},
            total_embeddings=50,
        )

        # Verify deterministic fields are identical
        assert manifest1.collection_id == manifest2.collection_id
        assert manifest1.total_embeddings == manifest2.total_embeddings
        assert manifest1.pipeline_config == manifest2.pipeline_config
        assert manifest1.vector_store_type == manifest2.vector_store_type


class TestRegulatoryComplianceIntegration:
    """
    Integration tests for regulatory compliance features.

    Tests complete workflows combining multiple components:
    - Version management + Governance
    - Citation + Manifest generation
    - End-to-end audit trail
    """

    def test_complete_approval_workflow(self, tmp_path):
        """Test complete document approval and version tracking workflow."""
        # Setup
        config = RAGConfig(allowlist=["ghg_protocol_corp"])
        governance = RAGGovernance(config, audit_dir=tmp_path / "audit")
        version_manager = DocumentVersionManager()

        # Create test document
        doc_path = tmp_path / "ghg_protocol_v1.05.pdf"
        doc_path.write_text("GHG Protocol content...")

        from greenlang.intelligence.rag.hashing import file_hash
        content_hash = file_hash(str(doc_path))

        doc_meta = DocMeta(
            doc_id="ghg_protocol_v1.05",
            title="GHG Protocol Corporate Standard",
            collection="ghg_protocol_corp",
            source_uri=str(doc_path),
            publisher="WRI/WBCSD",
            publication_date=date(2015, 3, 24),
            version="1.05",
            content_hash=content_hash,
            doc_hash="test_hash",
        )

        # Step 1: Register version
        version_manager.register_version(doc_meta, standard_id="ghg_protocol_corp")

        # Step 2: Submit for approval
        approvers = ["scientist_1", "scientist_2", "scientist_3"]
        success = governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=approvers,
        )
        assert success is True

        # Step 3: Vote for approval (2/3 majority)
        governance.vote_approval("ghg_protocol_corp", "scientist_1", approve=True)
        governance.vote_approval("ghg_protocol_corp", "scientist_2", approve=True)

        # Step 4: Verify approval
        assert governance.is_approved("ghg_protocol_corp") is True

        # Step 5: Verify version is retrievable
        retrieved = version_manager.retrieve_by_date("ghg_protocol_corp", date(2020, 1, 1))
        assert retrieved is not None
        assert retrieved.version == "1.05"

    def test_audit_trail_error_handling(self, tmp_path):
        """Test error cases produce proper audit trails."""
        config = RAGConfig(allowlist=["test_collection"], verify_checksums=True)
        governance = RAGGovernance(config, audit_dir=tmp_path / "audit")

        # Create document with wrong checksum
        doc_path = tmp_path / "test_doc.pdf"
        doc_path.write_text("Content")

        doc_meta = DocMeta(
            doc_id="test_doc",
            title="Test Document",
            collection="test_collection",
            content_hash="WRONG_CHECKSUM",  # Invalid
            doc_hash="hash",
        )

        # Should fail checksum verification
        success = governance.submit_for_approval(
            doc_path=doc_path,
            metadata=doc_meta,
            approvers=["approver_1"],
        )

        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
