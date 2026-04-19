# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Documentation Engine Tests
===================================================

Tests DocumentationEngine: methodology documentation, assumption tracking,
evidence management, completeness assessment, and assurance readiness.

Target: 40+ test cases.
"""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("documentation")

DocumentationEngine = _mod.DocumentationEngine
DocumentationResult = _mod.DocumentationResult
MethodologyDocument = _mod.MethodologyDocument
Assumption = _mod.Assumption
EvidenceRecord = _mod.EvidenceRecord
CategoryDocumentation = _mod.CategoryDocumentation
DocumentationCompleteness = _mod.DocumentationCompleteness
DocumentType = _mod.DocumentType
SensitivityLevel = _mod.SensitivityLevel
ApprovalStatus = _mod.ApprovalStatus
EvidenceType = _mod.EvidenceType
AssuranceReadiness = _mod.AssuranceReadiness


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh DocumentationEngine."""
    return DocumentationEngine()


@pytest.fixture
def sample_methodology_doc():
    """A sample methodology document."""
    return MethodologyDocument(
        document_type=DocumentType.METHODOLOGY,
        title="Scope 1 Stationary Combustion Methodology",
        category_id="cat-stationary",
        category_name="stationary_combustion",
        scope=1,
        description="Full methodology description for stationary combustion.",
        version="1.0",
        author="sustainability@acme.com",
        approval_status=ApprovalStatus.APPROVED,
    )


@pytest.fixture
def full_category_docs():
    """A full set of approved docs for one category (all required types)."""
    docs = []
    for dt in [
        DocumentType.METHODOLOGY,
        DocumentType.EMISSION_FACTOR,
        DocumentType.ACTIVITY_DATA,
        DocumentType.ASSUMPTION,
        DocumentType.CALCULATION,
        DocumentType.QA_QC,
        DocumentType.UNCERTAINTY,
        DocumentType.CHANGE_LOG,
    ]:
        docs.append(MethodologyDocument(
            document_type=dt,
            title=f"{dt.value} doc for stationary",
            category_id="cat-stationary",
            category_name="stationary_combustion",
            scope=1,
            version="1.0",
            author="user",
            approval_status=ApprovalStatus.APPROVED,
        ))
    return docs


@pytest.fixture
def sample_assumption():
    """A sample assumption record."""
    return Assumption(
        title="German grid emission factor 2025",
        description="Using IEA 2025 grid average for Germany: 0.365 kgCO2e/kWh",
        justification="Best available published data for German grid mix",
        category_id="cat-electricity",
        category_name="purchased_electricity",
        scope=2,
        sensitivity=SensitivityLevel.HIGH,
        impact_description="A 10% change affects total Scope 2 by ~3000 tCO2e",
    )


@pytest.fixture
def sample_evidence():
    """A sample evidence record."""
    return EvidenceRecord(
        evidence_type=EvidenceType.INVOICE,
        title="Natural gas invoice Q1 2025 - Frankfurt Plant",
        description="Gas utility invoice for January-March 2025",
        category_id="cat-stationary",
        category_name="stationary_combustion",
        scope=1,
        document_hash="a1b2c3d4e5f6" + "0" * 52,
        uploaded_by="user-data-entry-001",
    )


# ===================================================================
# Document Registration Tests
# ===================================================================


class TestDocumentRegistration:
    """Tests for register_document."""

    def test_register_document_returns_document(self, engine, sample_methodology_doc):
        result = engine.register_document(sample_methodology_doc)
        assert isinstance(result, MethodologyDocument)

    def test_register_document_with_content_adds_hash(self, engine):
        doc = MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="Test Doc",
            category_id="cat-1",
        )
        result = engine.register_document(doc, content="Hello content")
        assert result.content_hash != ""
        assert len(result.content_hash) == 64  # SHA-256

    def test_register_document_without_content_no_hash(self, engine, sample_methodology_doc):
        result = engine.register_document(sample_methodology_doc)
        # content_hash stays empty or default when no content is passed
        assert isinstance(result.content_hash, str)

    def test_register_document_preserves_fields(self, engine, sample_methodology_doc):
        result = engine.register_document(sample_methodology_doc)
        assert result.title == "Scope 1 Stationary Combustion Methodology"
        assert result.document_type == DocumentType.METHODOLOGY


# ===================================================================
# Model Construction Tests
# ===================================================================


class TestModelConstruction:
    """Tests for Pydantic model creation and defaults."""

    def test_methodology_document_defaults(self):
        doc = MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="Test",
        )
        assert doc.approval_status == ApprovalStatus.DRAFT
        assert doc.version == "1.0"
        assert doc.retention_years == 7
        assert doc.document_id != ""

    def test_assumption_defaults(self):
        asn = Assumption(
            title="Test assumption",
            description="Description of the assumption",
        )
        assert asn.sensitivity == SensitivityLevel.MEDIUM
        assert asn.approval_status == ApprovalStatus.DRAFT
        assert asn.review_frequency_months == 12
        assert asn.assumption_id != ""

    def test_evidence_record_defaults(self):
        ev = EvidenceRecord(
            evidence_type=EvidenceType.INVOICE,
            title="Test evidence",
        )
        assert ev.is_verified is False
        assert ev.file_size_bytes == 0
        assert ev.evidence_id != ""

    def test_category_documentation_defaults(self):
        cd = CategoryDocumentation()
        assert cd.completeness_pct == 0.0
        assert cd.assumptions_count == 0
        assert cd.evidence_count == 0

    def test_documentation_completeness_defaults(self):
        dc = DocumentationCompleteness()
        assert dc.overall_completeness_pct == 0.0
        assert dc.assurance_readiness == "not_ready"
        assert dc.total_documents == 0
        assert dc.total_assumptions == 0
        assert dc.total_evidence == 0

    def test_documentation_result_defaults(self):
        dr = DocumentationResult()
        assert dr.processing_time_ms == 0.0
        assert dr.provenance_hash == ""
        assert dr.document_registry == []
        assert dr.assumption_registry == []
        assert dr.evidence_registry == []


# ===================================================================
# Assess - Completeness Assessment Tests
# ===================================================================


class TestCompletenessAssessment:
    """Tests for engine.assess() completeness assessment."""

    def test_assess_returns_documentation_result(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert isinstance(result, DocumentationResult)

    def test_assess_completeness_populated(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert result.completeness is not None
        assert isinstance(result.completeness, DocumentationCompleteness)

    def test_assess_completeness_score_range(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert 0 <= result.completeness.overall_completeness_pct <= 100

    def test_assess_category_completeness_populated(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert len(result.completeness.category_completeness) >= 1

    def test_assess_provenance_hash(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert len(result.provenance_hash) == 64

    def test_assess_processing_time(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert result.processing_time_ms > 0

    def test_assess_empty_documents(self, engine):
        result = engine.assess(
            documents=[],
            category_ids=["cat-1"],
        )
        assert result.completeness.overall_completeness_pct == 0.0

    def test_assess_empty_categories_derives_from_docs(self, engine, sample_methodology_doc):
        # category_ids=None should derive categories from documents
        result = engine.assess(
            documents=[sample_methodology_doc],
        )
        assert result.completeness is not None

    def test_assess_with_assumptions_and_evidence(
        self, engine, sample_methodology_doc, sample_assumption, sample_evidence
    ):
        result = engine.assess(
            documents=[sample_methodology_doc],
            assumptions=[sample_assumption],
            evidence=[sample_evidence],
            category_ids=["cat-stationary", "cat-electricity"],
        )
        assert result.completeness.total_assumptions >= 1
        assert result.completeness.total_evidence >= 1

    def test_assess_document_registry_populated(
        self, engine, sample_methodology_doc
    ):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert len(result.document_registry) == 1

    def test_assess_assumption_registry_populated(
        self, engine, sample_methodology_doc, sample_assumption
    ):
        result = engine.assess(
            documents=[sample_methodology_doc],
            assumptions=[sample_assumption],
        )
        assert len(result.assumption_registry) == 1

    def test_assess_evidence_registry_populated(
        self, engine, sample_methodology_doc, sample_evidence
    ):
        result = engine.assess(
            documents=[sample_methodology_doc],
            evidence=[sample_evidence],
        )
        assert len(result.evidence_registry) == 1

    def test_full_documentation_high_completeness(self, engine, full_category_docs):
        result = engine.assess(
            documents=full_category_docs,
            category_ids=["cat-stationary"],
        )
        # All 8 required doc types present and approved -> 100%
        assert result.completeness.overall_completeness_pct >= 90.0

    def test_partial_documentation_lower_completeness(self, engine):
        # Only 1 of 8 required types
        docs = [MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="Only methodology",
            category_id="cat-1",
            approval_status=ApprovalStatus.APPROVED,
        )]
        result = engine.assess(
            documents=docs,
            category_ids=["cat-1"],
        )
        assert result.completeness.overall_completeness_pct < 50.0


# ===================================================================
# Category Completeness Tests
# ===================================================================


class TestCategoryCompleteness:
    """Tests for calculate_category_completeness."""

    def test_no_docs_returns_zero(self, engine):
        pct = engine.calculate_category_completeness("cat-1", [])
        assert pct == 0.0

    def test_no_matching_category_returns_zero(self, engine, sample_methodology_doc):
        pct = engine.calculate_category_completeness("cat-OTHER", [sample_methodology_doc])
        assert pct == 0.0

    def test_single_approved_methodology(self, engine, sample_methodology_doc):
        pct = engine.calculate_category_completeness(
            "cat-stationary", [sample_methodology_doc]
        )
        # methodology weight is 20 out of total 100 -> 20%
        assert pct == pytest.approx(20.0, abs=1.0)

    def test_unapproved_doc_gets_70_percent_credit(self, engine):
        doc = MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="Draft methodology",
            category_id="cat-1",
            approval_status=ApprovalStatus.DRAFT,
        )
        pct = engine.calculate_category_completeness("cat-1", [doc])
        # 20 * 0.7 / 100 = 14%
        assert pct == pytest.approx(14.0, abs=1.0)

    def test_full_approved_docs_100_percent(self, engine, full_category_docs):
        pct = engine.calculate_category_completeness("cat-stationary", full_category_docs)
        assert pct == pytest.approx(100.0, abs=0.5)


# ===================================================================
# Integrity Verification Tests
# ===================================================================


class TestDocumentIntegrity:
    """Tests for verify_document_integrity."""

    def test_matching_hash_returns_true(self, engine):
        content = "Test document content"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc = MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="Verified Doc",
            content_hash=content_hash,
        )
        assert engine.verify_document_integrity(doc, content_hash) is True

    def test_mismatching_hash_returns_false(self, engine):
        doc = MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="Tampered Doc",
            content_hash="a" * 64,
        )
        assert engine.verify_document_integrity(doc, "b" * 64) is False

    def test_empty_stored_hash_returns_false(self, engine):
        doc = MethodologyDocument(
            document_type=DocumentType.METHODOLOGY,
            title="No Hash Doc",
            content_hash="",
        )
        assert engine.verify_document_integrity(doc, "abc123" + "0" * 58) is False


# ===================================================================
# Assumptions Due for Review Tests
# ===================================================================


class TestAssumptionsDueReview:
    """Tests for get_assumptions_due_for_review."""

    def test_no_assumptions_returns_empty(self, engine):
        due = engine.get_assumptions_due_for_review([])
        assert due == []

    def test_assumptions_without_valid_until_skipped(self, engine):
        asn = Assumption(
            title="No expiry",
            description="Assumption without validity period",
        )
        due = engine.get_assumptions_due_for_review([asn])
        assert len(due) == 0

    def test_assumption_due_soon_included(self, engine):
        soon = datetime.now(timezone.utc) + timedelta(days=30)
        asn = Assumption(
            title="Due soon",
            description="Expiring within 90 days",
            valid_until=soon,
        )
        due = engine.get_assumptions_due_for_review([asn])
        assert len(due) == 1

    def test_assumption_already_expired_included(self, engine):
        past = datetime.now(timezone.utc) - timedelta(days=10)
        asn = Assumption(
            title="Already expired",
            description="Past validity",
            valid_until=past,
        )
        due = engine.get_assumptions_due_for_review([asn])
        assert len(due) == 1

    def test_assumption_far_future_excluded(self, engine):
        future = datetime.now(timezone.utc) + timedelta(days=365)
        asn = Assumption(
            title="Far future",
            description="Not due yet",
            valid_until=future,
        )
        due = engine.get_assumptions_due_for_review([asn])
        assert len(due) == 0

    def test_custom_reference_date(self, engine):
        valid_until = datetime(2025, 6, 1, tzinfo=timezone.utc)
        asn = Assumption(
            title="Test",
            description="Test assumption",
            valid_until=valid_until,
        )
        # Reference date 30 days before -> within 90 day window
        ref = datetime(2025, 5, 1, tzinfo=timezone.utc)
        due = engine.get_assumptions_due_for_review([asn], as_of=ref)
        assert len(due) == 1


# ===================================================================
# Assurance Readiness Tests
# ===================================================================


class TestAssuranceReadiness:
    """Tests for assurance readiness classification."""

    def test_not_ready_when_empty(self, engine):
        result = engine.assess(
            documents=[],
            category_ids=["cat-1"],
        )
        readiness = result.completeness.assurance_readiness
        assert readiness in ("not_ready", "partially_ready")

    def test_ready_with_full_docs(self, engine, full_category_docs):
        result = engine.assess(
            documents=full_category_docs,
            category_ids=["cat-stationary"],
        )
        readiness = result.completeness.assurance_readiness
        # Full approved docs -> should be ready or mostly_ready
        assert readiness in ("ready", "mostly_ready")

    def test_recommendations_generated(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert isinstance(result.completeness.recommendations, list)


# ===================================================================
# Integrity Checks in Assess Tests
# ===================================================================


class TestAssessIntegrityChecks:
    """Tests for integrity checks performed during assess."""

    def test_integrity_checks_field_populated(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert isinstance(result.integrity_checks, list)

    def test_retention_alerts_field_populated(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert isinstance(result.retention_alerts, list)


# ===================================================================
# Provenance Tests
# ===================================================================


class TestProvenance:
    """Tests for deterministic provenance hashing."""

    def test_provenance_hash_is_64_chars(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert len(result.provenance_hash) == 64

    def test_methodology_notes_populated(self, engine, sample_methodology_doc):
        result = engine.assess(
            documents=[sample_methodology_doc],
            category_ids=["cat-stationary"],
        )
        assert len(result.methodology_notes) >= 1


# ===================================================================
# Enum Value Tests
# ===================================================================


class TestEnumValues:
    """Tests for enum member values."""

    @pytest.mark.parametrize("doc_type", list(DocumentType))
    def test_document_types(self, doc_type):
        assert doc_type.value is not None
        assert isinstance(doc_type.value, str)

    @pytest.mark.parametrize("sensitivity", list(SensitivityLevel))
    def test_sensitivity_levels(self, sensitivity):
        assert sensitivity.value is not None

    @pytest.mark.parametrize("status", list(ApprovalStatus))
    def test_approval_statuses(self, status):
        assert status.value is not None

    @pytest.mark.parametrize("etype", list(EvidenceType))
    def test_evidence_types(self, etype):
        assert etype.value is not None

    @pytest.mark.parametrize("readiness", list(AssuranceReadiness))
    def test_assurance_readiness_values(self, readiness):
        assert readiness.value is not None

    def test_document_type_count(self):
        assert len(DocumentType) == 10

    def test_evidence_type_count(self):
        assert len(EvidenceType) == 10

    def test_approval_status_count(self):
        assert len(ApprovalStatus) == 5

    def test_sensitivity_level_count(self):
        assert len(SensitivityLevel) == 3

    def test_assurance_readiness_count(self):
        assert len(AssuranceReadiness) == 4
